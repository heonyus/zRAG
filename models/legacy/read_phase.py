"""
Read Phase Trainer
- QA Fine-tuning: max log P(answer | q, z_selected; θ)
- Stage 2: LLM (QLoRA) + z_i + query encoder 함께 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ReadPhaseTrainer:
    """
    Read Phase: Query + z_i → Answer 학습

    Loss = L_qa + α × L_retrieval
    - L_qa: answer generation loss
    - L_retrieval: gold documents가 top-k에 포함되도록
    """

    def __init__(
        self,
        model,  # ParametricQA instance
        lr_llm: float = 2e-5,
        lr_z: float = 1e-3,
        lr_proj: float = 1e-4,
        alpha_retrieval: float = 0.1,
        top_k: int = 5,
        gradient_accumulation_steps: int = 8,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.alpha_retrieval = alpha_retrieval
        self.top_k = top_k
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.device = device

        # LLM unfreeze (LoRA parameters만)
        self.model.unfreeze_llm()

        # Multi-group optimizer
        param_groups = self.model.get_trainable_params(stage="read")
        self.optimizer = torch.optim.AdamW([
            {"params": pg["params"], "lr": pg.get("lr", lr_llm)}
            for pg in param_groups
            if any(p.requires_grad for p in (pg["params"] if isinstance(pg["params"], list) else [pg["params"]]))
        ])

        self.global_step = 0
        self.accumulation_counter = 0

    def compute_retrieval_loss(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        gold_doc_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieval loss: gold documents가 높은 score를 받도록 학습

        Args:
            query_ids: [batch, q_len]
            query_mask: [batch, q_len]
            gold_doc_ids: [batch, max_gold] (-1 for padding)

        Returns:
            loss: scalar
        """
        # Query embedding
        q_embed = self.model.get_query_embedding(query_ids, query_mask)

        # Document representations
        z_repr = self.model.doc_vectors.mean(dim=1)  # [num_docs, z_dim]

        # Full scores from selector
        if hasattr(self.model.selector, 'compute_full_scores'):
            all_scores = self.model.selector.compute_full_scores(q_embed, z_repr)
        else:
            # Cosine similarity fallback
            q_norm = F.normalize(q_embed, p=2, dim=-1)
            d_norm = F.normalize(z_repr, p=2, dim=-1)
            all_scores = torch.matmul(q_norm, d_norm.t())  # [batch, num_docs]

        # Log-softmax over all documents
        log_probs = F.log_softmax(all_scores, dim=-1)  # [batch, num_docs]

        # Gold document scores
        batch_size = gold_doc_ids.size(0)
        loss = torch.tensor(0.0, device=self.device)
        count = 0

        for i in range(batch_size):
            valid_golds = gold_doc_ids[i][gold_doc_ids[i] >= 0]
            if len(valid_golds) > 0:
                # Negative log-likelihood for gold docs
                gold_log_probs = log_probs[i, valid_golds]
                loss -= gold_log_probs.mean()
                count += 1

        if count > 0:
            loss /= count

        return loss

    def train_step(self, batch: dict) -> dict:
        """Single training step (may accumulate gradients)"""
        question_ids = batch["question_ids"].to(self.device)
        question_mask = batch["question_mask"].to(self.device)
        answer_ids = batch["answer_ids"].to(self.device)
        answer_mask = batch["answer_mask"].to(self.device)
        gold_doc_ids = batch["gold_doc_ids"].to(self.device)

        # 1. Document selection
        with torch.no_grad():
            selected_ids, selected_scores = self.model.select_documents(
                question_ids, question_mask, k=self.top_k
            )

        # 2. QA Forward (answer generation loss)
        qa_result = self.model(
            query_ids=question_ids,
            doc_indices=selected_ids,
            answer_ids=answer_ids,
            query_attention_mask=question_mask,
            answer_attention_mask=answer_mask,
        )
        qa_loss = qa_result["loss"]

        # 3. Retrieval loss
        retrieval_loss = self.compute_retrieval_loss(
            question_ids, question_mask, gold_doc_ids
        )

        # 4. Total loss
        total_loss = qa_loss + self.alpha_retrieval * retrieval_loss

        # Gradient accumulation
        scaled_loss = total_loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        self.accumulation_counter += 1

        step_metrics = {
            "qa_loss": qa_loss.item(),
            "retrieval_loss": retrieval_loss.item(),
            "total_loss": total_loss.item(),
        }

        # Optimizer step after accumulation
        if self.accumulation_counter >= self.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                self.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulation_counter = 0
            self.global_step += 1

        return step_metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
        log_every: int = 50,
    ) -> dict:
        """1 epoch 학습"""
        self.model.train()
        total_qa_loss = 0.0
        total_ret_loss = 0.0
        total_loss = 0.0
        num_steps = 0

        pbar = tqdm(dataloader, desc=f"Read Phase Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)

            total_qa_loss += metrics["qa_loss"]
            total_ret_loss += metrics["retrieval_loss"]
            total_loss += metrics["total_loss"]
            num_steps += 1

            if (batch_idx + 1) % log_every == 0:
                avg_loss = total_loss / num_steps
                avg_qa = total_qa_loss / num_steps
                avg_ret = total_ret_loss / num_steps
                pbar.set_postfix(
                    total=f"{avg_loss:.4f}",
                    qa=f"{avg_qa:.4f}",
                    ret=f"{avg_ret:.4f}",
                )

        epoch_metrics = {
            "qa_loss": total_qa_loss / max(num_steps, 1),
            "retrieval_loss": total_ret_loss / max(num_steps, 1),
            "total_loss": total_loss / max(num_steps, 1),
            "epoch": epoch,
            "global_step": self.global_step,
        }
        return epoch_metrics

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 3,
        log_every: int = 50,
        eval_fn=None,
        save_path: Optional[str] = None,
    ) -> list:
        """전체 Read Phase 학습"""
        all_metrics = []

        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader, epoch=epoch, log_every=log_every)
            all_metrics.append(metrics)

            logger.info(
                f"Read Phase Epoch {epoch}: "
                f"total={metrics['total_loss']:.4f}, "
                f"qa={metrics['qa_loss']:.4f}, "
                f"ret={metrics['retrieval_loss']:.4f}"
            )

            # Evaluation
            if eval_fn is not None:
                eval_result = eval_fn(self.model)
                logger.info(f"  Eval: {eval_result}")
                metrics["eval"] = eval_result

            # Save
            if save_path:
                self.model.save_checkpoint(f"{save_path}/read_epoch_{epoch+1}.pt")

        if save_path:
            self.model.save_checkpoint(f"{save_path}/read_final.pt")

        return all_metrics
