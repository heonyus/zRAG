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
from typing import Optional, Tuple


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

        # ============================
        # 5. Doc embeddings cache (Top-k routing 속도 최적화)
        # ============================
        self._doc_embed_cache = None      # [num_docs, hidden_size]
        self._full_embed_cache = None     # [num_docs, m_tokens, hidden_size]
        self._doc_embed_norm_cache = None # normalized version for cosine sim

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

    @torch.no_grad()
    def precompute_doc_embeddings(self):
        """
        Doc embeddings 사전계산 및 캐시 (Top-k routing 속도 최적화)

        Phase 1 로드 후 한 번만 호출하면 됨.
        이후 select_topk_docs()에서 캐시 사용 → 매 샘플마다 projection 안 함.
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.info("Precomputing doc embeddings for Top-k routing...")

        # [num_docs, m_tokens, z_dim]
        Z = self.memory_pool

        # → [num_docs * m_tokens, z_dim] → [num_docs * m_tokens, hidden_size]
        Z_flat = Z.view(-1, self.z_dim)
        E_flat = self.z_to_embedding(Z_flat)

        # → [num_docs, m_tokens, hidden_size]
        self._full_embed_cache = E_flat.view(self.num_docs, self.m_tokens, -1)

        # 문서별 대표 벡터: token 평균
        self._doc_embed_cache = self._full_embed_cache.mean(dim=1)  # [num_docs, hidden_size]

        # Normalized version for cosine similarity (미리 계산)
        self._doc_embed_norm_cache = torch.nn.functional.normalize(
            self._doc_embed_cache, dim=-1
        )

        # 검증 로그
        logger.info(f"  doc_embed_cache: {self._doc_embed_cache.shape}")
        logger.info(f"  full_embed_cache: {self._full_embed_cache.shape}")
        logger.info(f"  doc_embed mean: {self._doc_embed_cache.mean().item():.4e}, "
                   f"std: {self._doc_embed_cache.std().item():.4e}")

        # NaN/Inf 체크
        if not torch.isfinite(self._doc_embed_cache).all():
            logger.error("WARNING: doc_embed_cache contains NaN/Inf!")
        if not torch.isfinite(self._full_embed_cache).all():
            logger.error("WARNING: full_embed_cache contains NaN/Inf!")

        logger.info("Doc embeddings precomputed and cached.")

    def get_doc_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        문서별 embedding 반환 (캐시 사용)

        Returns:
            doc_embed: [num_docs, hidden_size] - 문서별 대표 벡터 (token 평균)
            full_embed: [num_docs, m_tokens, hidden_size] - 전체 토큰별 embedding
        """
        # 캐시가 있으면 사용 (속도 최적화)
        if self._doc_embed_cache is not None and self._full_embed_cache is not None:
            return self._doc_embed_cache, self._full_embed_cache

        # 캐시 없으면 계산 (첫 호출 시)
        # [num_docs, m_tokens, z_dim]
        Z = self.memory_pool

        # → [num_docs * m_tokens, z_dim] → [num_docs * m_tokens, hidden_size]
        Z_flat = Z.view(-1, self.z_dim)
        E_flat = self.z_to_embedding(Z_flat)

        # → [num_docs, m_tokens, hidden_size]
        full_embed = E_flat.view(self.num_docs, self.m_tokens, -1)

        # 문서별 대표 벡터: token 평균
        doc_embed = full_embed.mean(dim=1)  # [num_docs, hidden_size]

        return doc_embed, full_embed

    def get_query_embedding(self, query_ids: torch.Tensor) -> torch.Tensor:
        """
        Query embedding 계산 (Top-k routing용)

        Args:
            query_ids: [1, query_len]

        Returns:
            query_embed: [1, hidden_size] - query 대표 벡터
        """
        emb = self.llm.get_input_embeddings()(query_ids)  # [1, q_len, hidden]
        q_embed = emb.mean(dim=1)  # [1, hidden]
        return q_embed

    def select_topk_docs(
        self,
        query_ids: torch.Tensor,
        k: int = 8,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query와 가장 관련 있는 Top-k 문서 선택 (cosine similarity)

        Args:
            query_ids: [1, query_len]
            k: 선택할 문서 수
            debug: 디버그 로그 출력 여부

        Returns:
            topk_indices: [k] - 선택된 문서 인덱스
            topk_embed: [k, m_tokens, hidden_size] - 선택된 문서들의 embedding
        """
        import logging
        logger = logging.getLogger(__name__)

        q_embed = self.get_query_embedding(query_ids)  # [1, hidden]

        # 캐시된 doc embeddings 사용 (속도 최적화)
        if self._doc_embed_norm_cache is not None and self._full_embed_cache is not None:
            doc_norm = self._doc_embed_norm_cache  # [num_docs, hidden] - 이미 normalized
            full_embed = self._full_embed_cache    # [num_docs, m_tokens, hidden]
        else:
            # 캐시 없으면 계산 (fallback)
            doc_embed, full_embed = self.get_doc_embeddings()
            doc_norm = torch.nn.functional.normalize(doc_embed, dim=-1)

        # Query normalization
        q_norm = torch.nn.functional.normalize(q_embed, dim=-1)  # [1, hidden]

        # [1, hidden] @ [hidden, num_docs] → [1, num_docs]
        sim = torch.mm(q_norm, doc_norm.t()).squeeze(0)  # [num_docs]

        # Top-k 선택
        k = min(k, sim.numel())
        topk = torch.topk(sim, k=k)
        topk_indices = topk.indices  # [k]
        topk_scores = topk.values    # [k]

        # 선택된 문서들의 full embedding
        topk_embed = full_embed[topk_indices]  # [k, m_tokens, hidden]

        # 디버그 로그
        if debug:
            logger.info(f"[TopK] k={k}, selected_docs={topk_indices.tolist()}")
            logger.info(f"[TopK] scores={[f'{s:.4f}' for s in topk_scores.tolist()]}")
            logger.info(f"[TopK] prefix_tokens={k * self.m_tokens}")
            logger.info(f"[TopK] topk_embed shape={topk_embed.shape}")

        return topk_indices, topk_embed

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
        top_k_docs: Optional[int] = None,
        debug: bool = False,
        **generate_kwargs,
    ) -> str:
        """
        Evidence 텍스트 생성

        Args:
            query_ids: [1, query_len] - 질문 토큰 (batch_size=1)
            max_new_tokens: 최대 생성 토큰 수
            top_k_docs: Top-k 문서만 사용 (None이면 전체 사용)
            debug: 디버그 로그 출력 여부

        Returns:
            evidence: 생성된 evidence 텍스트
        """
        self.eval()
        batch_size = query_ids.size(0)

        # Memory embeddings
        if top_k_docs is not None and top_k_docs < self.num_docs:
            # Top-k routing: query와 유사한 문서만 선택
            topk_indices, topk_embed = self.select_topk_docs(query_ids, k=top_k_docs, debug=debug)
            # [k, m_tokens, hidden] → [1, k*m_tokens, hidden]
            Z_embed = topk_embed.view(1, -1, topk_embed.size(-1))
            Z_len = Z_embed.size(1)
        else:
            # 전체 Z 사용 (기존 동작)
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

    def load_from_phase1(self, z_pool_path: str, projection_path: str = None, lora_path: str = None):
        """
        Phase 1/2 결과물 로드 (LoRA 포함)

        Phase 1/2에서 학습된:
        - z_pool.pt / best.pt: 학습된 z_i들 [num_docs, m_tokens, z_dim]
        - projection.pt: z_to_embedding layer weights
        - best.pt_lora/: LoRA adapter weights (Phase 2에서 학습된 경우)

        Args:
            z_pool_path: z_pool.pt 또는 best.pt 경로
            projection_path: projection.pt 경로 (optional)
            lora_path: LoRA adapter 디렉토리 경로 (optional, 없으면 z_pool_path + "_lora"로 자동 탐색)
        """
        import logging
        logger = logging.getLogger(__name__)

        # z_pool 로드
        checkpoint = torch.load(z_pool_path, map_location=self.device)

        # Phase 1/2 checkpoint format 처리:
        # Option 1: z_pool (tensor) - [num_docs, m_tokens, z_dim]
        # Option 2: z_vectors (dict) - {doc_id: tensor [m_tokens, z_dim]}
        # Option 3: memory_pool (tensor) - Phase 2 save_checkpoint 형식
        if "z_pool" in checkpoint:
            z_pool = checkpoint["z_pool"]  # [num_docs, m_tokens, z_dim]
            doc_ids = checkpoint.get("doc_ids", [f"doc_{i}" for i in range(z_pool.shape[0])])
        elif "z_vectors" in checkpoint:
            # z_vectors dict를 tensor로 변환
            z_vectors = checkpoint["z_vectors"]
            doc_ids = sorted(z_vectors.keys())
            z_list = [z_vectors[doc_id] for doc_id in doc_ids]
            z_pool = torch.stack(z_list, dim=0)  # [num_docs, m_tokens, z_dim]
            logger.info(f"Converted z_vectors dict ({len(doc_ids)} docs) to z_pool tensor")
        elif "memory_pool" in checkpoint:
            # Phase 2 checkpoint 형식 (save_checkpoint으로 저장된 것)
            z_pool = checkpoint["memory_pool"]
            doc_ids = checkpoint.get("doc_ids", [f"doc_{i}" for i in range(z_pool.shape[0])])
            logger.info(f"Loaded memory_pool from Phase 2 checkpoint ({z_pool.shape[0]} docs)")
        else:
            raise KeyError(f"Checkpoint must contain 'z_pool', 'z_vectors', or 'memory_pool'. Found keys: {list(checkpoint.keys())}")

        # Shape 검증
        loaded_num_docs, loaded_m_tokens, loaded_z_dim = z_pool.shape
        if loaded_m_tokens != self.m_tokens:
            raise ValueError(f"m_tokens mismatch: loaded={loaded_m_tokens}, model={self.m_tokens}")
        if loaded_z_dim != self.z_dim:
            raise ValueError(f"z_dim mismatch: loaded={loaded_z_dim}, model={self.z_dim}")

        # num_docs가 다른 경우 처리
        if loaded_num_docs != self.num_docs:
            logger.warning(f"num_docs mismatch: loaded={loaded_num_docs}, model={self.num_docs}")
            if loaded_num_docs < self.num_docs:
                # Phase 1에서 학습한 것보다 모델이 크면, 나머지는 랜덤 초기화 유지
                self.memory_pool.data[:loaded_num_docs] = z_pool.to(self.device)
                logger.info(f"Loaded {loaded_num_docs} z vectors, kept {self.num_docs - loaded_num_docs} random")
            else:
                # Phase 1에서 학습한 것이 더 많으면, 앞부분만 사용
                self.memory_pool.data = z_pool[:self.num_docs].to(self.device)
                logger.info(f"Loaded first {self.num_docs} z vectors from {loaded_num_docs}")
        else:
            self.memory_pool.data = z_pool.to(self.device)
            logger.info(f"Loaded {loaded_num_docs} z vectors")

        # Projection layer 로드 (checkpoint 내부 또는 별도 파일)
        # strict=False: Phase 1과 Phase 2의 projection 구조가 다를 수 있음
        if "z_to_embedding" in checkpoint:
            missing, unexpected = self.z_to_embedding.load_state_dict(
                checkpoint["z_to_embedding"],
                strict=False
            )
            if missing or unexpected:
                logger.warning(f"[load_from_phase1] z_to_embedding strict=False. "
                               f"missing={missing}, unexpected={unexpected}")
            logger.info("Loaded projection from checkpoint (strict=False)")
        elif projection_path:
            proj_checkpoint = torch.load(projection_path, map_location=self.device)
            missing, unexpected = self.z_to_embedding.load_state_dict(
                proj_checkpoint["z_to_embedding"],
                strict=False
            )
            if missing or unexpected:
                logger.warning(f"[load_from_phase1] z_to_embedding strict=False. "
                               f"missing={missing}, unexpected={unexpected}")
            logger.info(f"Loaded projection from {projection_path} (strict=False)")

        # doc_id 매핑 정보 저장 (필요시 사용)
        self.doc_ids = doc_ids
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

        # Device 동기화: LLM embedding과 같은 device로 이동
        try:
            target_device = self.llm.get_input_embeddings().weight.device
        except Exception:
            target_device = self.device

        self.z_to_embedding = self.z_to_embedding.to(target_device)
        self.memory_pool.data = self.memory_pool.data.to(target_device)
        logger.info(f"Moved z_to_embedding and memory_pool to {target_device}")

        # LoRA adapter 로드 (Phase 2에서 학습된 경우)
        from pathlib import Path
        from peft import PeftModel

        # LoRA 경로 결정: 명시적 경로 > z_pool_path + "_lora" 자동 탐색
        if lora_path is None:
            auto_lora_path = Path(z_pool_path).with_suffix("") if z_pool_path.endswith(".pt") else Path(z_pool_path)
            auto_lora_path = str(auto_lora_path) + "_lora"
            if Path(auto_lora_path).exists():
                lora_path = auto_lora_path
                logger.info(f"Auto-detected LoRA path: {lora_path}")

        if lora_path and Path(lora_path).exists():
            try:
                # PEFT adapter 로드
                logger.info(f"Loading LoRA adapter from: {lora_path}")
                # 이미 PeftModel인 경우 load_adapter 사용, 아니면 PeftModel.from_pretrained
                if hasattr(self.llm, 'load_adapter'):
                    self.llm.load_adapter(lora_path, adapter_name="phase2")
                    self.llm.set_adapter("phase2")
                    logger.info("Loaded LoRA adapter via load_adapter (existing PeftModel)")
                else:
                    # Base model에 PEFT 적용
                    self.llm = PeftModel.from_pretrained(self.llm, lora_path)
                    logger.info("Loaded LoRA adapter via PeftModel.from_pretrained")
                logger.info("LoRA adapter loaded successfully!")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapter: {e}")
                logger.warning("Continuing without LoRA...")
        else:
            logger.info("No LoRA adapter found, using base model weights")

        logger.info(f"Phase 1/2 load complete: {len(self.doc_ids)} docs")

        # Top-k routing용 doc embeddings 사전계산 (속도 최적화)
        self.precompute_doc_embeddings()

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
