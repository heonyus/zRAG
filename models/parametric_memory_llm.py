"""
Parametric Memory LLM - LLM-as-Memory 핵심 모델

핵심 구조:
- Memory Pool: Z = {z_i} (문서별 learnable vectors)
- 내부 Routing: Z 전체를 prefix로 → LLM 내부 attention이 자동 선택
- 출력: Evidence 텍스트 생성

학습 목표: -log P(evidence | query, Z; θ)
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional


class ParametricMemoryLLM(nn.Module):
    """
    LLM-as-Memory: Vector DB를 대체하는 Parametric Memory 시스템

    구조:
    - Memory Pool: Z = {z_1, ..., z_N} (learnable vectors)
    - Forward: Query → [Z prefix + Query] → LLM → Evidence
    - 외부 Selector 없음: LLM 내부 attention이 relevant z_i 선택
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen3-8B",
        num_docs: int = 2000,
        z_dim: int = 256,
        m_tokens: int = 4,
        quantization: str = "4bit",
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_target_modules: list = None,
        lora_dropout: float = 0.05,
        device: str = "cuda",
    ):
        super().__init__()

        self.num_docs = num_docs
        self.z_dim = z_dim
        self.m_tokens = m_tokens
        self.device = device

        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        # ============================
        # 1. LLM (QLoRA)
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
        self.llm = prepare_model_for_kbit_training(self.llm)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)

        self.hidden_size = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size

        # ============================
        # 2. Memory Pool (Learnable)
        # ============================
        # shape: [num_docs, m_tokens, z_dim]
        self.memory_pool = nn.Parameter(
            torch.randn(num_docs, m_tokens, z_dim) * 0.02
        )

        # ============================
        # 3. z → LLM Embedding Projection
        # ============================
        self.z_to_embedding = nn.Sequential(
            nn.Linear(z_dim, z_dim * 2),
            nn.GELU(),
            nn.Linear(z_dim * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

        # ============================
        # 4. Tokenizer
        # ============================
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_memory_embeddings(self) -> torch.Tensor:
        """
        Memory Pool 전체를 LLM embedding space로 변환

        Returns:
            Z_embed: [num_docs * m_tokens, hidden_size]
        """
        # [num_docs, m_tokens, z_dim] → [num_docs * m_tokens, z_dim]
        Z_flat = self.memory_pool.view(-1, self.z_dim)
        # → [num_docs * m_tokens, hidden_size]
        Z_embed = self.z_to_embedding(Z_flat)
        return Z_embed

    def forward(
        self,
        query_ids: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
        evidence_ids: Optional[torch.Tensor] = None,
        evidence_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass: Query + Z → Evidence 생성

        Args:
            query_ids: [batch, query_len] - 질문 토큰
            evidence_ids: [batch, evidence_len] - 정답 evidence (학습 시)

        Returns:
            dict with 'logits', 'loss' (if evidence_ids provided)
        """
        batch_size = query_ids.size(0)

        # 1. Memory Pool → LLM embeddings
        # [num_docs * m_tokens, hidden_size]
        Z_embed = self.get_memory_embeddings()
        Z_len = Z_embed.size(0)

        # Batch 차원 추가 및 확장: [1, Z_len, hidden] → [batch, Z_len, hidden]
        Z_embed = Z_embed.unsqueeze(0).expand(batch_size, -1, -1)

        # 2. Query → LLM embeddings
        query_embed = self.llm.get_input_embeddings()(query_ids)  # [batch, q_len, hidden]

        # 3. Concatenate: [Z | Query | Evidence(학습시)]
        if evidence_ids is not None:
            # 학습: [Z | Query | Evidence[:-1]] → predict Evidence
            evidence_embed = self.llm.get_input_embeddings()(evidence_ids)
            # Teacher forcing: evidence[:-1]
            combined_embed = torch.cat([
                Z_embed,
                query_embed,
                evidence_embed[:, :-1, :]
            ], dim=1)

            # Attention mask
            Z_mask = torch.ones(batch_size, Z_len, device=query_ids.device)
            if query_attention_mask is None:
                query_attention_mask = torch.ones_like(query_ids)
            if evidence_attention_mask is None:
                evidence_attention_mask = torch.ones_like(evidence_ids)
            combined_mask = torch.cat([
                Z_mask,
                query_attention_mask.float(),
                evidence_attention_mask[:, :-1].float()
            ], dim=1)

            # Forward through LLM
            outputs = self.llm(
                inputs_embeds=combined_embed,
                attention_mask=combined_mask,
                output_attentions=False,  # 메모리 절약
            )

            # Loss 계산: Evidence 토큰 예측
            # logits at positions [Z_len + q_len - 1 : -1] → evidence tokens
            q_len = query_ids.size(1)
            e_len = evidence_ids.size(1)
            shift_start = Z_len + q_len - 1
            shift_end = shift_start + e_len - 1  # -1 for shifted

            shift_logits = outputs.logits[:, shift_start:shift_end, :]
            shift_labels = evidence_ids[:, 1:]  # shifted right

            # Mask padding
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                shift_logits.reshape(-1, self.vocab_size),
                shift_labels.reshape(-1),
            )

            return {"logits": outputs.logits, "loss": loss}

        else:
            # 추론: [Z | Query]
            combined_embed = torch.cat([Z_embed, query_embed], dim=1)
            Z_mask = torch.ones(batch_size, Z_len, device=query_ids.device)
            if query_attention_mask is None:
                query_attention_mask = torch.ones_like(query_ids)
            combined_mask = torch.cat([Z_mask, query_attention_mask.float()], dim=1)

            outputs = self.llm(
                inputs_embeds=combined_embed,
                attention_mask=combined_mask,
            )
            return {"logits": outputs.logits}

    @torch.no_grad()
    def generate_evidence(
        self,
        query_ids: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        **generate_kwargs,
    ) -> str:
        """
        Evidence 텍스트 생성

        Args:
            query_ids: [1, query_len] - 질문 토큰 (batch_size=1)
            max_new_tokens: 최대 생성 토큰 수

        Returns:
            evidence: 생성된 evidence 텍스트
        """
        self.eval()
        batch_size = query_ids.size(0)

        # Memory embeddings
        Z_embed = self.get_memory_embeddings()  # [Z_len, hidden]
        Z_len = Z_embed.size(0)
        Z_embed = Z_embed.unsqueeze(0).expand(batch_size, -1, -1)

        # Query embeddings
        query_embed = self.llm.get_input_embeddings()(query_ids)

        # Combined: [Z | Query]
        combined_embed = torch.cat([Z_embed, query_embed], dim=1)
        Z_mask = torch.ones(batch_size, Z_len, device=query_ids.device)
        if query_attention_mask is None:
            query_attention_mask = torch.ones_like(query_ids)
        combined_mask = torch.cat([Z_mask, query_attention_mask.float()], dim=1)

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=combined_embed,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )

        # Decode (생성된 부분만)
        # generate가 반환하는 것은 전체 시퀀스가 아닌 새로 생성된 토큰
        evidence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return evidence

    def get_trainable_params(self) -> list:
        """학습 대상 파라미터 목록 반환"""
        return [
            {"params": [self.memory_pool], "lr": 1e-3, "name": "memory_pool"},
            {"params": self.z_to_embedding.parameters(), "lr": 1e-4, "name": "z_to_embedding"},
            {"params": filter(lambda p: p.requires_grad, self.llm.parameters()),
             "lr": 2e-5, "name": "llm_lora"},
        ]

    def save_checkpoint(self, path: str):
        """모델 체크포인트 저장"""
        state = {
            "memory_pool": self.memory_pool.data,
            "z_to_embedding": self.z_to_embedding.state_dict(),
            "config": {
                "num_docs": self.num_docs,
                "z_dim": self.z_dim,
                "m_tokens": self.m_tokens,
            },
        }
        torch.save(state, path)
        # LoRA weights 별도 저장
        self.llm.save_pretrained(path + "_lora")

    def load_checkpoint(self, path: str):
        """모델 체크포인트 로드"""
        state = torch.load(path, map_location=self.device)
        self.memory_pool.data = state["memory_pool"]
        self.z_to_embedding.load_state_dict(state["z_to_embedding"])

    def get_memory_stats(self) -> dict:
        """메모리 사용량 통계"""
        Z_params = self.num_docs * self.m_tokens * self.z_dim
        Z_embed_params = self.num_docs * self.m_tokens * self.hidden_size

        return {
            "num_docs": self.num_docs,
            "m_tokens": self.m_tokens,
            "z_dim": self.z_dim,
            "hidden_size": self.hidden_size,
            "memory_pool_params": Z_params,
            "memory_pool_mb": Z_params * 4 / 1024 / 1024,  # float32
            "z_embed_tokens": self.num_docs * self.m_tokens,
            "z_embed_mb": Z_embed_params * 2 / 1024 / 1024,  # bfloat16
        }
