"""
Parametric QA 통합 모델

핵심 구조:
- Document Vectors: per-document learnable embeddings {z_i}
- Query Encoder: E5-base-v2 기반 query → z_space projection
- LLM: QLoRA fine-tuned LLM for answer generation
- z_to_embedding: z vector → LLM input embedding space 변환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Tuple

from .router import CosineSelector, LearnedRouter, AttentionSelector


class ParametricQA(nn.Module):
    """
    Parametric QA System

    Write Phase: Document D_i → Learnable Vector z_i
    Read Phase: Query q → Select z_i → LLM(z_i, q) → Answer
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen3-8B",
        num_docs: int = 5000,
        z_dim: int = 256,
        m_tokens: int = 8,
        selection_method: str = "cosine",
        quantization: str = "4bit",
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_target_modules: list = None,
        lora_dropout: float = 0.05,
        query_encoder_name: str = "intfloat/e5-base-v2",
        device: str = "cuda",
    ):
        super().__init__()

        self.num_docs = num_docs
        self.z_dim = z_dim
        self.m_tokens = m_tokens
        self.selection_method = selection_method
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
        # 2. Document Vectors (Learnable)
        # ============================
        # shape: [num_docs, m_tokens, z_dim]
        self.doc_vectors = nn.Parameter(
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
        # 4. Query Encoder (for selection)
        # ============================
        self.query_encoder = AutoModel.from_pretrained(
            query_encoder_name,
            trust_remote_code=True,
        )
        # Freeze query encoder initially
        for param in self.query_encoder.parameters():
            param.requires_grad = False

        query_hidden = self.query_encoder.config.hidden_size  # 768 for E5-base
        self.query_proj = nn.Linear(query_hidden, z_dim)

        # ============================
        # 5. Selection Module
        # ============================
        self.selector = self._build_selector(selection_method, z_dim)

        # ============================
        # 6. Tokenizer
        # ============================
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_selector(self, method: str, z_dim: int) -> nn.Module:
        """Selection method에 따른 selector 생성"""
        if method == "cosine":
            return CosineSelector()
        elif method == "learned_router":
            return LearnedRouter(z_dim)
        elif method == "attention":
            return AttentionSelector(z_dim)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def get_query_embedding(self, query_input_ids: torch.Tensor,
                            query_attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Query를 z_dim 차원의 embedding으로 변환"""
        with torch.no_grad():
            q_output = self.query_encoder(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
            )
        # [CLS] token representation
        q_hidden = q_output.last_hidden_state[:, 0, :]  # [batch, 768]
        q_proj = self.query_proj(q_hidden)  # [batch, z_dim]
        return q_proj

    def select_documents(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor = None,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query에 대해 top-k document vectors 선택

        Returns:
            indices: [batch, k] - 선택된 doc indices
            scores: [batch, k] - selection scores
        """
        q_embed = self.get_query_embedding(query_input_ids, query_attention_mask)

        # z_i의 mean pooling으로 document representation
        z_repr = self.doc_vectors.mean(dim=1)  # [num_docs, z_dim]

        # Selection
        indices, scores = self.selector(q_embed, z_repr, k=k)
        return indices, scores

    def encode_z_for_llm(self, doc_indices: torch.Tensor) -> torch.Tensor:
        """
        선택된 z_i들을 LLM input embedding으로 변환

        Args:
            doc_indices: [batch, k]

        Returns:
            z_embeds: [batch, k * m_tokens, hidden_size]
        """
        batch_size = doc_indices.size(0)
        k = doc_indices.size(1)

        # Gather selected vectors
        # doc_vectors: [num_docs, m_tokens, z_dim]
        selected_z = self.doc_vectors[doc_indices]  # [batch, k, m_tokens, z_dim]
        selected_z = selected_z.view(batch_size, k * self.m_tokens, self.z_dim)

        # Project to LLM hidden size
        z_embeds = self.z_to_embedding(selected_z)  # [batch, k*m, hidden_size]
        return z_embeds

    def forward(
        self,
        query_ids: torch.Tensor,
        doc_indices: torch.Tensor,
        answer_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        answer_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass: z_i + query → LLM → answer

        Args:
            query_ids: [batch, query_len]
            doc_indices: [batch, k]
            answer_ids: [batch, answer_len] (for training)

        Returns:
            dict with 'logits', 'loss' (if answer_ids provided)
        """
        batch_size = query_ids.size(0)

        # 1. z_i → LLM embeddings
        z_embeds = self.encode_z_for_llm(doc_indices)  # [batch, k*m, hidden]
        z_len = z_embeds.size(1)

        # 2. Query → LLM embeddings
        query_embeds = self.llm.get_input_embeddings()(query_ids)  # [batch, q_len, hidden]

        # 3. Concatenate: [z_tokens | query_tokens]
        if answer_ids is not None:
            # Training: [z | query | answer]
            answer_embeds = self.llm.get_input_embeddings()(answer_ids)
            combined_embeds = torch.cat([z_embeds, query_embeds, answer_embeds], dim=1)

            # Build attention mask
            z_mask = torch.ones(batch_size, z_len, device=query_ids.device)
            if query_attention_mask is None:
                query_attention_mask = torch.ones_like(query_ids)
            if answer_attention_mask is None:
                answer_attention_mask = torch.ones_like(answer_ids)
            combined_mask = torch.cat([z_mask, query_attention_mask.float(),
                                       answer_attention_mask.float()], dim=1)

            # Forward through LLM
            outputs = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
            )

            # Loss: predict answer tokens
            # Shift: logits at positions [z_len + q_len - 1 : -1] predict answer tokens
            q_len = query_ids.size(1)
            a_len = answer_ids.size(1)
            shift_start = z_len + q_len - 1
            shift_end = shift_start + a_len

            shift_logits = outputs.logits[:, shift_start:shift_end, :]
            shift_labels = answer_ids

            # Mask padding in loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                shift_logits.reshape(-1, self.vocab_size),
                shift_labels.reshape(-1),
            )

            return {"logits": outputs.logits, "loss": loss}
        else:
            # Inference: [z | query]
            combined_embeds = torch.cat([z_embeds, query_embeds], dim=1)
            z_mask = torch.ones(batch_size, z_len, device=query_ids.device)
            if query_attention_mask is None:
                query_attention_mask = torch.ones_like(query_ids)
            combined_mask = torch.cat([z_mask, query_attention_mask.float()], dim=1)

            outputs = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
            )
            return {"logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        query_ids: torch.Tensor,
        doc_indices: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Answer 생성 (inference)

        Args:
            query_ids: [batch, query_len]
            doc_indices: [batch, k]

        Returns:
            generated_ids: [batch, max_new_tokens]
        """
        batch_size = query_ids.size(0)

        # z embeddings
        z_embeds = self.encode_z_for_llm(doc_indices)
        z_len = z_embeds.size(1)

        # Query embeddings
        query_embeds = self.llm.get_input_embeddings()(query_ids)

        # Combined
        combined_embeds = torch.cat([z_embeds, query_embeds], dim=1)
        z_mask = torch.ones(batch_size, z_len, device=query_ids.device)
        if query_attention_mask is None:
            query_attention_mask = torch.ones_like(query_ids)
        combined_mask = torch.cat([z_mask, query_attention_mask.float()], dim=1)

        # Generate using LLM
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )

        return outputs

    def write_phase_forward(
        self,
        doc_ids: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Write Phase forward: z_i → reconstruct D_i

        Args:
            doc_ids: [batch] - document indices
            doc_input_ids: [batch, doc_len] - document token ids

        Returns:
            dict with 'loss', 'perplexity'
        """
        batch_size = doc_ids.size(0)

        # z_i for these documents
        z_i = self.doc_vectors[doc_ids]  # [batch, m_tokens, z_dim]
        z_embeds = self.z_to_embedding(z_i)  # [batch, m_tokens, hidden_size]
        z_len = z_embeds.size(1)

        # Document embeddings (teacher forcing)
        doc_embeds = self.llm.get_input_embeddings()(doc_input_ids)  # [batch, doc_len, hidden]

        # Input: [z_i | doc_tokens[:-1]]
        input_embeds = torch.cat([z_embeds, doc_embeds[:, :-1, :]], dim=1)

        # Attention mask
        z_mask = torch.ones(batch_size, z_len, device=doc_ids.device)
        if doc_attention_mask is None:
            doc_attention_mask = torch.ones_like(doc_input_ids)
        # doc mask에서 마지막 하나 제거 (shifted)
        input_mask = torch.cat([z_mask, doc_attention_mask[:, :-1].float()], dim=1)

        # Forward
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=input_mask,
        )

        # Loss: z positions 이후의 logits로 document tokens 예측
        # logits[:, z_len-1:-1] → targets: doc_input_ids[:, 1:]는 아니고
        # logits[:, z_len-1:] 의 첫 doc_len-1개가 doc_input_ids[:,1:]를 예측
        logits = outputs.logits[:, z_len - 1:, :]  # [batch, doc_len, vocab]

        # Targets: doc의 모든 토큰 (첫 토큰 포함)
        # logits[0] → doc_input_ids[0], logits[1] → doc_input_ids[1], ...
        targets = doc_input_ids  # [batch, doc_len]

        # 실제 사용할 길이 맞추기
        min_len = min(logits.size(1), targets.size(1))
        logits = logits[:, :min_len, :]
        targets = targets[:, :min_len]

        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

        perplexity = torch.exp(loss).item()

        return {"loss": loss, "perplexity": perplexity}

    def freeze_llm(self):
        """LLM 파라미터 동결 (Write Phase Stage 1)"""
        for param in self.llm.parameters():
            param.requires_grad = False

    def unfreeze_llm(self):
        """LLM 파라미터 동결 해제 (Read Phase Stage 2)"""
        for param in self.llm.parameters():
            if hasattr(param, '_is_lora') or 'lora' in str(param):
                param.requires_grad = True
        # peft model은 trainable params만 grad=True
        self.llm.train()

    def get_trainable_params(self, stage: str = "write") -> list:
        """학습 대상 파라미터 목록 반환"""
        if stage == "write":
            return [
                {"params": [self.doc_vectors], "lr": 1e-3, "name": "doc_vectors"},
            ]
        elif stage == "read":
            return [
                {"params": [self.doc_vectors], "lr": 1e-3, "name": "doc_vectors"},
                {"params": self.query_proj.parameters(), "lr": 1e-4, "name": "query_proj"},
                {"params": self.z_to_embedding.parameters(), "lr": 1e-4, "name": "z_to_embedding"},
                {"params": self.selector.parameters(), "lr": 1e-4, "name": "selector"},
                {"params": filter(lambda p: p.requires_grad, self.llm.parameters()),
                 "lr": 2e-5, "name": "llm_lora"},
            ]
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def save_checkpoint(self, path: str):
        """모델 체크포인트 저장"""
        state = {
            "doc_vectors": self.doc_vectors.data,
            "z_to_embedding": self.z_to_embedding.state_dict(),
            "query_proj": self.query_proj.state_dict(),
            "selector": self.selector.state_dict(),
            "config": {
                "num_docs": self.num_docs,
                "z_dim": self.z_dim,
                "m_tokens": self.m_tokens,
                "selection_method": self.selection_method,
            },
        }
        torch.save(state, path)
        # LoRA weights 별도 저장
        self.llm.save_pretrained(path + "_lora")

    def load_checkpoint(self, path: str):
        """모델 체크포인트 로드"""
        state = torch.load(path, map_location=self.device)
        self.doc_vectors.data = state["doc_vectors"]
        self.z_to_embedding.load_state_dict(state["z_to_embedding"])
        self.query_proj.load_state_dict(state["query_proj"])
        self.selector.load_state_dict(state["selector"])
