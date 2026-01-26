"""
Phase 1: Write Phase Model - Token-as-Document Learning

핵심 목표 (교수님 2=A):
- z_i만 넣으면 해당 문서 D_i가 생성되도록 학습
- LLM은 freeze, z_i만 최적화
- Loss: -log P(D_i | z_i)

구조:
- z_i: [m_tokens, z_dim] learnable vectors (문서별)
- projection: z_dim → hidden_size
- LLM: frozen decoder
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WritePhaseModel(nn.Module):
    """
    Phase 1: Token-as-Document 학습 모델

    교수님 의도:
    - "토큰만 넣으면 문서가 생성되는지" 확인
    - LLM freeze, z_i만 학습
    - z_i가 "문서의 키"처럼 동작하도록

    학습 후:
    - 학습된 z_i들을 모아서 memory_pool로 저장
    - Phase 3에서 이 pool을 로드해서 사용
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen3-8B",
        m_tokens: int = 4,
        z_dim: int = 256,
        quantization: str = "4bit",
        device: str = "cuda",
    ):
        super().__init__()

        self.llm_name = llm_name
        self.m_tokens = m_tokens
        self.z_dim = z_dim
        self.device = device

        # ============================
        # 1. LLM (Frozen, No LoRA)
        # ============================
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # LLM 완전 freeze (Phase 1 핵심)
        for param in self.llm.parameters():
            param.requires_grad = False

        self.hidden_size = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size

        # ============================
        # 2. z → LLM Embedding Projection (학습 대상)
        # ============================
        # LayerNorm 제거 - std를 1로 강제하면 스케일 문제 발생
        self.z_to_embedding = nn.Sequential(
            nn.Linear(z_dim, z_dim * 2),
            nn.GELU(),
            nn.Linear(z_dim * 2, self.hidden_size),
        ).to(device)

        # α 게이트: 스케일 안정화 (init=1e-2, trainable)
        # z_embed = alpha * projection(z)
        self.alpha = nn.Parameter(torch.tensor(0.01, device=device))

        # ============================
        # 3. Tokenizer
        # ============================
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"[WritePhaseModel] LLM: {llm_name} (frozen)")
        logger.info(f"[WritePhaseModel] m_tokens={m_tokens}, z_dim={z_dim}, hidden_size={self.hidden_size}")

    def create_z_for_doc(self) -> nn.Parameter:
        """
        새 문서용 z_i 생성

        Returns:
            z_i: [m_tokens, z_dim] learnable parameter
        """
        z_i = nn.Parameter(
            torch.randn(self.m_tokens, self.z_dim, device=self.device) * 0.02
        )
        return z_i

    def forward(
        self,
        z_i: torch.Tensor,
        doc_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass: z_i → D_i 생성

        Args:
            z_i: [m_tokens, z_dim] or [batch, m_tokens, z_dim] - 문서별 learned tokens
            doc_ids: [batch, doc_len] - 정답 문서 토큰
            doc_attention_mask: [batch, doc_len] - attention mask

        Returns:
            dict with 'loss', 'logits'
        """
        # z_i shape 처리
        if z_i.dim() == 2:
            # [m_tokens, z_dim] → [1, m_tokens, z_dim]
            z_i = z_i.unsqueeze(0)

        batch_size = doc_ids.size(0)

        # z_i를 batch에 맞게 확장
        if z_i.size(0) == 1 and batch_size > 1:
            z_i = z_i.expand(batch_size, -1, -1)

        # 1. z_i → LLM embedding space (with α gate for scale stabilization)
        z_embed = self.alpha * self.z_to_embedding(z_i)  # [batch, m_tokens, hidden_size]

        # 2. Document → LLM embeddings (teacher forcing)
        doc_embed = self.llm.get_input_embeddings()(doc_ids)  # [batch, doc_len, hidden]

        # 3. Concatenate: [z_i | doc[:-1]] → predict doc
        # Teacher forcing: doc[:-1]를 입력으로, doc[1:]을 타겟으로
        combined_embed = torch.cat([
            z_embed,
            doc_embed[:, :-1, :]  # doc[:-1]
        ], dim=1)

        # Attention mask
        z_mask = torch.ones(batch_size, self.m_tokens, device=doc_ids.device)
        if doc_attention_mask is None:
            doc_attention_mask = torch.ones_like(doc_ids)
        combined_mask = torch.cat([
            z_mask,
            doc_attention_mask[:, :-1].float()
        ], dim=1)

        # 4. Forward through frozen LLM
        with torch.no_grad():
            # LLM은 inference mode (gradient 계산 안 함)
            pass

        # 하지만 embedding → projection은 gradient 필요
        outputs = self.llm(
            inputs_embeds=combined_embed,
            attention_mask=combined_mask,
            output_attentions=False,
        )

        # 5. Loss 계산: doc 토큰 예측
        # logits at positions [m_tokens-1 : -1] → doc[1:] tokens
        m = self.m_tokens
        doc_len = doc_ids.size(1)

        # shift_logits: [m_tokens:] 위치의 logits (doc 생성 부분)
        shift_logits = outputs.logits[:, m-1:m-1+doc_len-1, :]  # [batch, doc_len-1, vocab]
        shift_labels = doc_ids[:, 1:]  # [batch, doc_len-1]

        # Cross entropy loss (padding 무시)
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(
            shift_logits.reshape(-1, self.vocab_size),
            shift_labels.reshape(-1),
        )

        return {
            "loss": loss,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate_from_z(
        self,
        z_i: torch.Tensor,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **generate_kwargs,
    ) -> str:
        """
        z_i로부터 문서 생성 (추론)

        Args:
            z_i: [m_tokens, z_dim] - 학습된 토큰
            max_new_tokens: 최대 생성 토큰 수
            do_sample: 샘플링 여부 (True 권장, greedy는 붕괴 위험)
            temperature: 샘플링 temperature
            top_p: nucleus sampling threshold

        Returns:
            generated_text: 생성된 문서 텍스트
        """
        self.eval()

        if z_i.dim() == 2:
            z_i = z_i.unsqueeze(0)  # [1, m_tokens, z_dim]

        # z_i를 float32로 projection 후 bfloat16으로 변환
        # (z_to_embedding이 float32이므로 float32로 입력 후 출력을 bfloat16으로)
        z_i_float = z_i.float()
        z_embed = self.alpha * self.z_to_embedding(z_i_float)  # [1, m_tokens, hidden] (float32) with α gate
        z_embed = z_embed.to(torch.bfloat16)  # LLM 입력용 bfloat16으로 변환

        # attention_mask 생성 (inputs_embeds 사용 시 필수)
        batch_size, seq_len, _ = z_embed.shape
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=z_embed.device)

        # Generate
        gen_kwargs = dict(
            inputs_embeds=z_embed,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        if do_sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            gen_kwargs["do_sample"] = False

        gen_kwargs.update(generate_kwargs)
        outputs = self.llm.generate(**gen_kwargs)

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def get_z_embed_stats(self, z_i: torch.Tensor) -> dict:
        """
        z_i → embedding의 통계 확인 (디버깅용)

        스케일 붕괴 여부 진단에 사용
        """
        if z_i.dim() == 2:
            z_i = z_i.unsqueeze(0)

        with torch.no_grad():
            # autocast로 dtype 자동 맞춤
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                z_embed = self.alpha * self.z_to_embedding(z_i)

        return {
            "z_i_norm": z_i.float().norm().item(),
            "z_i_mean": z_i.float().mean().item(),
            "z_i_std": z_i.float().std().item(),
            "z_embed_norm": z_embed.float().norm().item(),
            "z_embed_mean": z_embed.float().mean().item(),
            "z_embed_std": z_embed.float().std().item(),
        }

    def get_trainable_params(self, z_i: nn.Parameter, lr_z: float = 1e-2, lr_proj: float = 0) -> list:
        """
        학습 대상 파라미터 반환

        Phase 1에서는:
        - z_i (문서별 learned tokens)
        - z_to_embedding (projection layer) - lr_proj > 0일 때만

        LLM은 freeze

        Args:
            z_i: 학습할 z_i 파라미터
            lr_z: z_i learning rate
            lr_proj: projection learning rate (0이면 freeze)
        """
        params = [{"params": [z_i], "lr": lr_z, "name": "z_i"}]

        # α gate는 항상 학습 (스케일 조절용)
        params.append({
            "params": [self.alpha],
            "lr": lr_z,  # z_i와 같은 lr 사용
            "name": "alpha"
        })

        if lr_proj > 0:
            params.append({
                "params": self.z_to_embedding.parameters(),
                "lr": lr_proj,
                "name": "z_to_embedding"
            })
            logger.info(f"[get_trainable_params] z_i lr={lr_z}, alpha lr={lr_z}, projection lr={lr_proj}")
        else:
            # Projection freeze
            for param in self.z_to_embedding.parameters():
                param.requires_grad = False
            logger.info(f"[get_trainable_params] z_i lr={lr_z}, alpha lr={lr_z}, projection FROZEN")

        return params

    def save_projection(self, path: str):
        """Projection layer 저장 (z_to_embedding)"""
        torch.save({
            "z_to_embedding": self.z_to_embedding.state_dict(),
            "config": {
                "m_tokens": self.m_tokens,
                "z_dim": self.z_dim,
                "hidden_size": self.hidden_size,
            }
        }, path)
        logger.info(f"Saved projection to {path}")

    def load_projection(self, path: str):
        """Projection layer 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.z_to_embedding.load_state_dict(checkpoint["z_to_embedding"])
        logger.info(f"Loaded projection from {path}")


class ZPoolManager:
    """
    학습된 z_i들을 관리하고 저장하는 클래스

    Phase 1 학습 결과물:
    - z_pool: [num_docs, m_tokens, z_dim]
    - doc_id → index 매핑

    Phase 3에서 이 pool을 로드해서 memory_pool로 사용
    """

    def __init__(self, m_tokens: int, z_dim: int, device: str = "cuda"):
        self.m_tokens = m_tokens
        self.z_dim = z_dim
        self.device = device

        self.z_vectors = []  # List of [m_tokens, z_dim]
        self.doc_ids = []    # List of doc_id strings
        self.doc_id_to_idx = {}  # doc_id → index mapping

    def add_z(self, doc_id: str, z_i: torch.Tensor):
        """학습 완료된 z_i 추가"""
        if doc_id in self.doc_id_to_idx:
            # 이미 있으면 업데이트
            idx = self.doc_id_to_idx[doc_id]
            self.z_vectors[idx] = z_i.detach().cpu()
        else:
            # 새로 추가
            self.doc_id_to_idx[doc_id] = len(self.z_vectors)
            self.z_vectors.append(z_i.detach().cpu())
            self.doc_ids.append(doc_id)

    def get_z(self, doc_id: str) -> Optional[torch.Tensor]:
        """doc_id로 z_i 조회"""
        if doc_id not in self.doc_id_to_idx:
            return None
        idx = self.doc_id_to_idx[doc_id]
        return self.z_vectors[idx]

    def get_pool_tensor(self) -> torch.Tensor:
        """
        전체 z_pool을 텐서로 반환

        Returns:
            z_pool: [num_docs, m_tokens, z_dim]
        """
        if not self.z_vectors:
            raise ValueError("No z vectors in pool")
        return torch.stack(self.z_vectors, dim=0)

    def save(self, path: str):
        """
        z_pool 저장 (Phase 3에서 로드 가능한 포맷)

        저장 포맷:
        - z_pool: [num_docs, m_tokens, z_dim]
        - doc_ids: list of doc_id strings
        - doc_id_to_idx: dict mapping
        - config: m_tokens, z_dim
        """
        z_pool = self.get_pool_tensor()

        checkpoint = {
            "z_pool": z_pool,
            "doc_ids": self.doc_ids,
            "doc_id_to_idx": self.doc_id_to_idx,
            "config": {
                "num_docs": len(self.doc_ids),
                "m_tokens": self.m_tokens,
                "z_dim": self.z_dim,
            }
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved z_pool ({len(self.doc_ids)} docs) to {path}")

    def load(self, path: str):
        """z_pool 로드"""
        checkpoint = torch.load(path, map_location="cpu")

        z_pool = checkpoint["z_pool"]
        self.doc_ids = checkpoint["doc_ids"]
        self.doc_id_to_idx = checkpoint["doc_id_to_idx"]

        # z_pool을 개별 z_i로 분리
        self.z_vectors = [z_pool[i] for i in range(z_pool.size(0))]

        self.m_tokens = checkpoint["config"]["m_tokens"]
        self.z_dim = checkpoint["config"]["z_dim"]

        logger.info(f"Loaded z_pool ({len(self.doc_ids)} docs) from {path}")

    def __len__(self):
        return len(self.z_vectors)
