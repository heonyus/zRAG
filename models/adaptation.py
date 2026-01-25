"""
z Adaptation 모듈
- 새 문서에 대한 빠른 z_i 생성 (few-step optimization)
- 교수님 요구: "initialization 가능한 vector"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ZAdaptation:
    """
    새 문서에 대한 z_i 빠른 적응

    방법:
    1. Random initialization → SGD optimization (baseline)
    2. Encoder-based initialization (후속 연구)

    θ는 frozen, z_new만 학습
    """

    def __init__(
        self,
        model,  # ParametricQA instance
        default_steps: int = 50,
        default_lr: float = 5e-3,
        device: str = "cuda",
    ):
        self.model = model
        self.default_steps = default_steps
        self.default_lr = default_lr
        self.device = device

    @torch.no_grad()
    def initialize_z(
        self,
        doc_input_ids: torch.Tensor,
        method: str = "random",
    ) -> torch.Tensor:
        """
        z_i 초기값 생성

        Methods:
        - random: N(0, 0.02) 초기화
        - encoder: E5 encoder 출력 기반 초기화
        - mean: 기존 z_i들의 평균으로 초기화
        """
        m_tokens = self.model.m_tokens
        z_dim = self.model.z_dim

        if method == "random":
            z_init = torch.randn(1, m_tokens, z_dim, device=self.device) * 0.02

        elif method == "encoder":
            # E5 encoder 출력을 z_dim으로 project
            encoder_output = self.model.query_encoder(
                input_ids=doc_input_ids[:, :512].to(self.device),
            )
            # [CLS] token → z_dim
            cls_hidden = encoder_output.last_hidden_state[:, 0, :]  # [1, 768]
            z_base = self.model.query_proj(cls_hidden)  # [1, z_dim]
            # Repeat for m_tokens
            z_init = z_base.unsqueeze(1).expand(-1, m_tokens, -1).clone()
            z_init += torch.randn_like(z_init) * 0.01  # small noise

        elif method == "mean":
            # 기존 z_i들의 평균
            z_mean = self.model.doc_vectors.data.mean(dim=0, keepdim=True)  # [1, m, z_dim]
            z_init = z_mean.clone()
            z_init += torch.randn_like(z_init) * 0.01

        else:
            raise ValueError(f"Unknown init method: {method}")

        return z_init

    def adapt(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        lr: Optional[float] = None,
        init_method: str = "random",
        verbose: bool = False,
    ) -> dict:
        """
        새 문서에 대한 z adaptation

        Args:
            doc_input_ids: [1, doc_len] or [doc_len]
            doc_attention_mask: [1, doc_len] or [doc_len]
            steps: optimization steps
            lr: learning rate for z
            init_method: initialization method

        Returns:
            dict with:
                - z: adapted z vector [1, m_tokens, z_dim]
                - loss_history: list of losses
                - final_perplexity: final reconstruction perplexity
        """
        if steps is None:
            steps = self.default_steps
        if lr is None:
            lr = self.default_lr

        # Ensure correct dimensions
        if doc_input_ids.dim() == 1:
            doc_input_ids = doc_input_ids.unsqueeze(0)
        if doc_attention_mask is not None and doc_attention_mask.dim() == 1:
            doc_attention_mask = doc_attention_mask.unsqueeze(0)

        doc_input_ids = doc_input_ids.to(self.device)
        if doc_attention_mask is not None:
            doc_attention_mask = doc_attention_mask.to(self.device)

        # Initialize z
        z_new = nn.Parameter(self.initialize_z(doc_input_ids, method=init_method))
        optimizer = torch.optim.Adam([z_new], lr=lr)

        # Freeze everything except z_new
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Document embeddings (cached)
        with torch.no_grad():
            doc_embeds = self.model.llm.get_input_embeddings()(doc_input_ids)

        loss_history = []

        for step in range(steps):
            # z → LLM embedding
            z_embeds = self.model.z_to_embedding(z_new)  # [1, m, hidden]
            z_len = z_embeds.size(1)

            # Teacher forcing: [z | doc[:-1]] → predict doc
            input_embeds = torch.cat([z_embeds, doc_embeds[:, :-1, :]], dim=1)

            # Attention mask
            z_mask = torch.ones(1, z_len, device=self.device)
            if doc_attention_mask is not None:
                input_mask = torch.cat([z_mask, doc_attention_mask[:, :-1].float()], dim=1)
            else:
                input_mask = torch.ones(1, input_embeds.size(1), device=self.device)

            # Forward (no gradient through LLM weights)
            with torch.enable_grad():
                outputs = self.model.llm(
                    inputs_embeds=input_embeds,
                    attention_mask=input_mask,
                )

            # Loss
            logits = outputs.logits[:, z_len - 1:, :]
            targets = doc_input_ids
            min_len = min(logits.size(1), targets.size(1))
            logits = logits[:, :min_len, :]
            targets = targets[:, :min_len]

            loss = F.cross_entropy(
                logits.reshape(-1, self.model.vocab_size),
                targets.reshape(-1),
                ignore_index=self.model.tokenizer.pad_token_id,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if verbose and (step + 1) % 10 == 0:
                ppl = torch.exp(torch.tensor(loss.item())).item()
                logger.info(f"  Adapt step {step+1}/{steps}: loss={loss.item():.4f}, ppl={ppl:.2f}")

        # Restore model gradients
        for param in self.model.parameters():
            if hasattr(param, 'requires_grad'):
                pass  # will be set by caller

        final_ppl = torch.exp(torch.tensor(loss_history[-1])).item()

        return {
            "z": z_new.detach().clone(),
            "loss_history": loss_history,
            "final_loss": loss_history[-1],
            "final_perplexity": final_ppl,
        }

    def adapt_and_evaluate(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        qa_pairs: list,
        steps_list: list = None,
        init_method: str = "random",
    ) -> dict:
        """
        다양한 adaptation step 수에 대한 QA 성능 평가

        Args:
            qa_pairs: [(question_ids, answer_text), ...]
            steps_list: [10, 20, 50, 100, 200]

        Returns:
            dict mapping steps → metrics
        """
        if steps_list is None:
            steps_list = [10, 20, 50, 100, 200]

        results = {}

        for n_steps in steps_list:
            adapt_result = self.adapt(
                doc_input_ids=doc_input_ids,
                doc_attention_mask=doc_attention_mask,
                steps=n_steps,
                init_method=init_method,
            )

            results[n_steps] = {
                "final_perplexity": adapt_result["final_perplexity"],
                "final_loss": adapt_result["final_loss"],
            }

            logger.info(f"Steps={n_steps}: PPL={adapt_result['final_perplexity']:.2f}")

        return results
