"""
Evidence Trainer - Evidence 생성 학습기

학습 목표: -log P(evidence | query, Z; θ)

기존 read_phase.py와 다른 점:
1. 외부 selector 없음 → Z 전체가 내부 routing으로 처리
2. Answer 생성이 아닌 Evidence 생성
3. 1-Stage end-to-end 학습 (Write Phase 없음)
"""

import logging
from typing import Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvidenceTrainer:
    """
    Evidence 생성 학습기

    학습 구조:
    - Input: Query
    - Memory: Z (전체가 prefix로 포함)
    - Output: Evidence 텍스트
    - Loss: Cross-entropy on evidence tokens
    """

    def __init__(
        self,
        model,  # ParametricMemoryLLM
        lr_llm: float = 2e-5,
        lr_z: float = 1e-3,
        lr_proj: float = 1e-4,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 8,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
    ):
        """
        Args:
            model: ParametricMemoryLLM instance
            lr_llm: LLM (LoRA) learning rate
            lr_z: Memory pool learning rate
            lr_proj: z_to_embedding learning rate
            warmup_ratio: Warmup 비율
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Gradient clipping norm
            use_amp: Mixed precision 사용 여부
        """
        self.model = model
        self.lr_llm = lr_llm
        self.lr_z = lr_z
        self.lr_proj = lr_proj
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp

        # Optimizer with parameter groups
        self.optimizer = AdamW([
            {"params": [model.memory_pool], "lr": lr_z, "name": "memory_pool"},
            {"params": model.z_to_embedding.parameters(), "lr": lr_proj, "name": "z_to_embedding"},
            {"params": filter(lambda p: p.requires_grad, model.llm.parameters()),
             "lr": lr_llm, "name": "llm_lora"},
        ])

        # Mixed precision
        self.scaler = GradScaler() if use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0

    def train(
        self,
        train_dataloader: DataLoader,
        epochs: int = 5,
        eval_dataloader: Optional[DataLoader] = None,
        eval_steps: int = 500,
        save_path: Optional[str] = None,
        log_steps: int = 50,
    ) -> dict:
        """
        Evidence 생성 학습 실행

        Args:
            train_dataloader: 학습 데이터로더
            epochs: 학습 에폭 수
            eval_dataloader: 평가 데이터로더 (optional)
            eval_steps: 평가 주기
            save_path: 체크포인트 저장 경로
            log_steps: 로깅 주기

        Returns:
            dict with training metrics
        """
        self.model.train()
        total_steps = len(train_dataloader) * epochs

        # Learning rate scheduler with warmup
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = self._get_scheduler(total_steps, warmup_steps)

        metrics_history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
        }

        logger.info(f"Starting training: {epochs} epochs, {len(train_dataloader)} batches/epoch")
        logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"Memory pool size: {self.model.num_docs} docs × {self.model.m_tokens} tokens")

        for epoch in range(epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=True,
            )

            for batch_idx, batch in enumerate(progress_bar):
                loss = self._train_step(batch)
                epoch_loss += loss

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self._update_step(scheduler)
                    num_batches += 1

                    # Logging
                    if self.global_step % log_steps == 0:
                        avg_loss = epoch_loss / max(num_batches, 1)
                        lr = scheduler.get_last_lr()[0]
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                            "step": self.global_step,
                        })
                        metrics_history["train_loss"].append(avg_loss)
                        metrics_history["learning_rate"].append(lr)

                    # Evaluation
                    if eval_dataloader and self.global_step % eval_steps == 0:
                        eval_loss = self.evaluate(eval_dataloader)
                        metrics_history["eval_loss"].append(eval_loss)
                        logger.info(f"Step {self.global_step}: eval_loss = {eval_loss:.4f}")
                        self.model.train()

            # Epoch end
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1}/{epochs} completed. Avg loss: {avg_epoch_loss:.4f}")

            # Save checkpoint
            if save_path:
                checkpoint_path = f"{save_path}/epoch_{epoch + 1}.pt"
                self.model.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")

        return {
            "final_train_loss": avg_epoch_loss,
            "metrics_history": metrics_history,
            "total_steps": self.global_step,
        }

    def _train_step(self, batch: dict) -> float:
        """단일 학습 스텝"""
        # Move to device
        query_ids = batch["query_ids"].to(self.model.device)
        query_mask = batch["query_mask"].to(self.model.device)
        evidence_ids = batch["evidence_ids"].to(self.model.device)
        evidence_mask = batch["evidence_mask"].to(self.model.device)

        # Forward
        if self.use_amp:
            with autocast():
                outputs = self.model(
                    query_ids=query_ids,
                    query_attention_mask=query_mask,
                    evidence_ids=evidence_ids,
                    evidence_attention_mask=evidence_mask,
                )
                loss = outputs["loss"] / self.gradient_accumulation_steps

            # Backward with scaling
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(
                query_ids=query_ids,
                query_attention_mask=query_mask,
                evidence_ids=evidence_ids,
                evidence_attention_mask=evidence_mask,
            )
            loss = outputs["loss"] / self.gradient_accumulation_steps
            loss.backward()

        return loss.item() * self.gradient_accumulation_steps

    def _update_step(self, scheduler):
        """Gradient update step"""
        if self.use_amp:
            # Unscale for clipping
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
        )

        # Optimizer step
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

    def _get_scheduler(self, total_steps: int, warmup_steps: int):
        """Linear warmup + linear decay scheduler"""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )

        return LambdaLR(self.optimizer, lr_lambda)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """평가 (loss 계산)"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            query_ids = batch["query_ids"].to(self.model.device)
            query_mask = batch["query_mask"].to(self.model.device)
            evidence_ids = batch["evidence_ids"].to(self.model.device)
            evidence_mask = batch["evidence_mask"].to(self.model.device)

            outputs = self.model(
                query_ids=query_ids,
                query_attention_mask=query_mask,
                evidence_ids=evidence_ids,
                evidence_attention_mask=evidence_mask,
            )
            total_loss += outputs["loss"].item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def generate_samples(
        self,
        dataloader: DataLoader,
        num_samples: int = 5,
        max_new_tokens: int = 128,
    ) -> list:
        """
        샘플 Evidence 생성 (디버깅/모니터링용)

        Returns:
            List of {"query": str, "gold_evidence": str, "generated_evidence": str}
        """
        self.model.eval()
        samples = []

        for batch in dataloader:
            if len(samples) >= num_samples:
                break

            query_ids = batch["query_ids"].to(self.model.device)
            query_mask = batch["query_mask"].to(self.model.device)

            # Batch의 첫 번째 샘플만
            query_ids_single = query_ids[0:1]
            query_mask_single = query_mask[0:1]

            # Generate
            generated = self.model.generate_evidence(
                query_ids=query_ids_single,
                query_attention_mask=query_mask_single,
                max_new_tokens=max_new_tokens,
            )

            # Decode query and gold evidence
            query_text = self.model.tokenizer.decode(
                query_ids[0],
                skip_special_tokens=True,
            )
            gold_evidence = self.model.tokenizer.decode(
                batch["evidence_ids"][0],
                skip_special_tokens=True,
            )

            samples.append({
                "query": query_text,
                "gold_evidence": gold_evidence,
                "generated_evidence": generated,
            })

        return samples
