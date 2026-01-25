"""
Write Phase Trainer
- z_i 학습: max log P(D_i | z_i; θ)
- Stage 1: θ frozen, z_i만 학습
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class WritePhaseTrainer:
    """
    Write Phase: Document → z_i 학습

    각 document에 대해 z_i를 최적화하여
    z_i를 prefix로 넣으면 document가 재생성되도록 학습
    """

    def __init__(
        self,
        model,  # ParametricQA instance
        lr: float = 1e-3,
        llm_frozen: bool = True,
        device: str = "cuda",
    ):
        self.model = model
        self.lr = lr
        self.llm_frozen = llm_frozen
        self.device = device

        if llm_frozen:
            self.model.freeze_llm()

        # z_i만 optimizer에 등록
        self.optimizer = torch.optim.AdamW(
            [self.model.doc_vectors],
            lr=lr,
            weight_decay=0.01,
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
        log_every: int = 50,
    ) -> dict:
        """1 epoch 학습"""
        self.model.train()
        total_loss = 0.0
        total_ppl = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Write Phase Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            doc_ids = batch["doc_id"]
            if isinstance(doc_ids, list):
                doc_ids = torch.tensor(doc_ids, dtype=torch.long)
            doc_ids = doc_ids.to(self.device)

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Forward
            result = self.model.write_phase_forward(
                doc_ids=doc_ids,
                doc_input_ids=input_ids,
                doc_attention_mask=attention_mask,
            )

            loss = result["loss"]
            ppl = result["perplexity"]

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for z_i stability
            torch.nn.utils.clip_grad_norm_([self.model.doc_vectors], max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_ppl += ppl
            num_batches += 1

            if (batch_idx + 1) % log_every == 0:
                avg_loss = total_loss / num_batches
                avg_ppl = total_ppl / num_batches
                pbar.set_postfix(loss=f"{avg_loss:.4f}", ppl=f"{avg_ppl:.2f}")
                logger.info(f"Epoch {epoch}, Step {batch_idx+1}: "
                            f"loss={avg_loss:.4f}, ppl={avg_ppl:.2f}")

        metrics = {
            "loss": total_loss / max(num_batches, 1),
            "perplexity": total_ppl / max(num_batches, 1),
            "epoch": epoch,
        }
        return metrics

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 100,
        log_every: int = 50,
        eval_fn=None,
        save_path: Optional[str] = None,
    ) -> list:
        """전체 Write Phase 학습"""
        all_metrics = []

        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader, epoch=epoch, log_every=log_every)
            all_metrics.append(metrics)

            logger.info(f"Write Phase Epoch {epoch}: "
                        f"loss={metrics['loss']:.4f}, ppl={metrics['perplexity']:.2f}")

            # Evaluation
            if eval_fn is not None and (epoch + 1) % 10 == 0:
                eval_result = eval_fn(self.model)
                logger.info(f"  Eval: {eval_result}")

            # Save
            if save_path and (epoch + 1) % 20 == 0:
                self.model.save_checkpoint(f"{save_path}/write_epoch_{epoch+1}.pt")

        if save_path:
            self.model.save_checkpoint(f"{save_path}/write_final.pt")

        return all_metrics

    def train_single_document(
        self,
        doc_id: int,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        steps: int = 100,
        lr: Optional[float] = None,
    ) -> dict:
        """
        단일 문서에 대한 z_i 학습 (adaptation용)

        Returns:
            dict with final loss, perplexity, z_i
        """
        if lr is None:
            lr = self.lr

        # 해당 doc의 z만 별도 optimizer
        z_param = self.model.doc_vectors[doc_id:doc_id+1]
        optimizer = torch.optim.Adam([self.model.doc_vectors], lr=lr)

        doc_ids_tensor = torch.tensor([doc_id], device=self.device)
        input_ids = doc_input_ids.unsqueeze(0).to(self.device) \
            if doc_input_ids.dim() == 1 else doc_input_ids.to(self.device)
        attn_mask = doc_attention_mask.unsqueeze(0).to(self.device) \
            if doc_attention_mask.dim() == 1 else doc_attention_mask.to(self.device)

        losses = []
        for step in range(steps):
            result = self.model.write_phase_forward(
                doc_ids=doc_ids_tensor,
                doc_input_ids=input_ids,
                doc_attention_mask=attn_mask,
            )

            optimizer.zero_grad()
            result["loss"].backward()

            # Only update the specific doc_id's z
            # Mask gradient for other docs
            if self.model.doc_vectors.grad is not None:
                mask = torch.zeros_like(self.model.doc_vectors.grad)
                mask[doc_id] = 1.0
                self.model.doc_vectors.grad *= mask

            optimizer.step()
            losses.append(result["loss"].item())

        return {
            "final_loss": losses[-1],
            "final_perplexity": torch.exp(torch.tensor(losses[-1])).item(),
            "loss_history": losses,
            "z_i": self.model.doc_vectors[doc_id].detach().clone(),
        }
