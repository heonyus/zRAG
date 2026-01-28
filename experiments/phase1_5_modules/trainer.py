"""
Phase 1.5: LoRA Training for Evidence Generation

This module implements the training loop for Phase 1.5:
- Loads Phase 1 z_pool + projection (FROZEN)
- Attaches LoRA adapters to LLM (TRAINABLE)
- Trains on (z_i, query) -> evidence

Key features:
- Runtime assertions for frozen params
- JSONL-based resumable checkpointing
- Step-level and epoch-level metrics logging
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase1_modules.utils import (
    Timer,
    get_logger,
    save_json,
    append_jsonl,
    load_jsonl_cache,
)
from experiments.phase1_5_modules.model_wrapper import Phase15ForwardWrapper


class Phase15EvidenceDataset(Dataset):
    """
    Dataset for Phase 1.5 evidence generation training.

    Each sample: (doc_id, z_vector, query, evidence)
    """

    def __init__(
        self,
        dataset_path: Path,
        z_pool,  # ZPoolManager
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            dataset_path: Path to dataset.jsonl
            z_pool: ZPoolManager with loaded z vectors
            max_samples: Optional limit on samples
        """
        self.z_pool = z_pool
        self.samples = []

        # Load dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)

                # Skip samples without evidence
                if not sample.get("evidence_text"):
                    continue

                # Skip samples without valid doc_id
                doc_id = sample.get("doc_id")
                if not doc_id or doc_id not in z_pool.doc_ids:
                    continue

                self.samples.append({
                    "sample_id": sample["sample_id"],
                    "doc_id": doc_id,
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "evidence": sample["evidence_text"],
                })

                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Get z vector for this document
        z = self.z_pool.get_z(sample["doc_id"])

        return {
            "sample_id": sample["sample_id"],
            "doc_id": sample["doc_id"],
            "z": z,  # [m_tokens, z_dim]
            "query": sample["question"],
            "evidence": sample["evidence"],
            "answer": sample["answer"],
        }


def collate_phase15_batch(batch: List[Dict]) -> Dict:
    """Collate function for Phase 1.5 dataset."""
    return {
        "sample_ids": [b["sample_id"] for b in batch],
        "doc_ids": [b["doc_id"] for b in batch],
        "z": torch.stack([b["z"] for b in batch]),  # [batch, m_tokens, z_dim]
        "queries": [b["query"] for b in batch],
        "evidences": [b["evidence"] for b in batch],
        "answers": [b["answer"] for b in batch],
    }


def verify_frozen_params(
    model,
    z_pool,
    allow_projection_training: bool = False,
) -> Dict[str, Any]:
    """
    Runtime assertion: Verify z and projection are frozen.

    FATAL if any frozen param has requires_grad=True.

    Args:
        model: WritePhaseModel instance
        z_pool: ZPoolManager instance
        allow_projection_training: If True, skip projection check

    Returns:
        Verification result dict

    Raises:
        AssertionError if any frozen param has requires_grad=True
    """
    result = {
        "z_pool_frozen": True,
        "projection_frozen": True,
        "alpha_frozen": True,
        "lora_trainable_count": 0,
        "all_verified": True,
    }

    # 1. Check z_pool is frozen
    for doc_id in z_pool.doc_ids:
        z = z_pool.get_z(doc_id)
        if z.requires_grad:
            result["z_pool_frozen"] = False
            result["all_verified"] = False
            raise AssertionError(f"FATAL: z for {doc_id} has requires_grad=True")

    # 2. Check projection is frozen (unless explicitly allowed)
    if not allow_projection_training:
        for name, param in model.z_to_embedding.named_parameters():
            if param.requires_grad:
                result["projection_frozen"] = False
                result["all_verified"] = False
                raise AssertionError(f"FATAL: projection param {name} has requires_grad=True")

    # 3. Check alpha is frozen
    if model.alpha.requires_grad:
        result["alpha_frozen"] = False
        result["all_verified"] = False
        raise AssertionError("FATAL: alpha has requires_grad=True")

    # 4. Count LoRA trainable params
    lora_trainable = sum(1 for p in model.llm.parameters() if p.requires_grad)
    result["lora_trainable_count"] = lora_trainable

    if lora_trainable == 0:
        result["all_verified"] = False
        raise AssertionError("FATAL: No trainable LoRA parameters found")

    return result


def setup_lora(
    model,
    lora_config: Dict,
):
    """
    Attach LoRA adapters to the model.

    Args:
        model: WritePhaseModel instance
        lora_config: LoRA configuration dict

    Returns:
        Modified model with LoRA adapters
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logger = get_logger()

    # Freeze projection
    for param in model.z_to_embedding.parameters():
        param.requires_grad = False
    logger.info("Projection frozen")

    # Freeze alpha
    model.alpha.requires_grad = False
    logger.info("Alpha frozen")

    # Prepare for LoRA (if quantized)
    if hasattr(model.llm, 'is_loaded_in_4bit') and model.llm.is_loaded_in_4bit:
        model.llm = prepare_model_for_kbit_training(model.llm)
        logger.info("Model prepared for k-bit training")

    # Create LoRA config
    lora_cfg = LoraConfig(
        r=lora_config.get("r", 32),
        lora_alpha=lora_config.get("alpha", 64),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        lora_dropout=lora_config.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    if not hasattr(model.llm, 'peft_config'):
        model.llm = get_peft_model(model.llm, lora_cfg)
        logger.info(f"LoRA applied: r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}")

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


class Phase15Trainer:
    """
    Trainer for Phase 1.5 LoRA-based evidence generation.
    """

    def __init__(
        self,
        model,  # WritePhaseModel
        z_pool,  # ZPoolManager
        tokenizer,
        train_dataset: Phase15EvidenceDataset,
        val_dataset: Optional[Phase15EvidenceDataset],
        run_dir: Path,
        config: Dict,
    ):
        """
        Args:
            model: WritePhaseModel with LoRA adapters
            z_pool: ZPoolManager with frozen z vectors
            tokenizer: LLM tokenizer
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            run_dir: Run directory
            config: Training configuration
        """
        self.model = model
        self.z_pool = z_pool
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.run_dir = Path(run_dir)
        self.config = config
        self.logger = get_logger()

        # Directories
        self.train_dir = self.run_dir / "02_train"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.train_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.train_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Cache for resumable training
        self.cache_dir = self.run_dir / "05_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.train_dir / "train_metrics.jsonl"

        # Create forward wrapper
        self.wrapper = Phase15ForwardWrapper(
            model, tokenizer, config.get("device", "cuda")
        )

        # Device
        self.device = config.get("device", "cuda")

    def setup_optimizer(self) -> AdamW:
        """
        Setup optimizer with only LoRA params.
        """
        # Collect LoRA params only
        lora_params = [p for p in self.model.llm.parameters() if p.requires_grad]

        # Verify no frozen params in optimizer
        frozen_param_ids = set()
        for doc_id in self.z_pool.doc_ids:
            frozen_param_ids.add(id(self.z_pool.get_z(doc_id)))
        for p in self.model.z_to_embedding.parameters():
            frozen_param_ids.add(id(p))
        frozen_param_ids.add(id(self.model.alpha))

        for p in lora_params:
            if id(p) in frozen_param_ids:
                raise AssertionError("FATAL: Frozen param included in optimizer")

        optimizer = AdamW(
            lora_params,
            lr=self.config.get("lr_lora", 2e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )

        self.logger.info(f"Optimizer: AdamW, lr={self.config.get('lr_lora', 2e-5)}")
        self.logger.info(f"Trainable params in optimizer: {len(lora_params)}")

        return optimizer

    def train(self) -> Dict:
        """
        Main training loop.

        Returns:
            Training summary dict
        """
        # CRITICAL: Verify frozen params before training
        allow_proj = self.config.get("tune_projection", False)
        verification = verify_frozen_params(self.model, self.z_pool, allow_proj)

        self.logger.info("=" * 60)
        self.logger.info("FROZEN PARAM VERIFICATION PASSED")
        self.logger.info(f"  z_pool: FROZEN ({len(self.z_pool.doc_ids)} docs)")
        self.logger.info(f"  projection: {'TRAINABLE' if allow_proj else 'FROZEN'}")
        self.logger.info(f"  alpha: FROZEN")
        self.logger.info(f"  LoRA trainable params: {verification['lora_trainable_count']:,}")
        self.logger.info("=" * 60)

        # Save verification result
        save_json(verification, self.train_dir / "frozen_params_verification.json")

        # Setup
        batch_size = self.config.get("batch_size", 2)
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        use_amp = self.config.get("use_amp", True)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_phase15_batch,
            num_workers=0,
        )

        optimizer = self.setup_optimizer()

        # AMP scaler
        scaler = torch.amp.GradScaler() if use_amp else None

        # Load cached progress
        cached_metrics = load_jsonl_cache(self.metrics_path, key_field="step")
        start_step = len(cached_metrics)
        if start_step > 0:
            self.logger.info(f"Resuming from step {start_step}")

        # Training state
        global_step = start_step
        best_loss = float("inf")
        train_losses = []

        self.model.train()

        with Timer("Training"):
            for epoch in range(epochs):
                epoch_losses = []

                progress = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch+1}/{epochs}",
                    leave=True,
                )

                for batch_idx, batch in enumerate(progress):
                    # Skip already processed steps
                    if global_step < start_step:
                        global_step += 1
                        continue

                    # Move z to device
                    z = batch["z"].to(self.device)

                    # Forward pass
                    if use_amp:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                            outputs = self.wrapper.forward_for_training(
                                z=z,
                                query_text=batch["queries"],
                                evidence_text=batch["evidences"],
                                max_query_length=self.config.get("max_query_length", 128),
                                max_evidence_length=self.config.get("max_evidence_length", 256),
                            )
                            loss = outputs["loss"]
                    else:
                        outputs = self.wrapper.forward_for_training(
                            z=z,
                            query_text=batch["queries"],
                            evidence_text=batch["evidences"],
                        )
                        loss = outputs["loss"]

                    # Backward pass
                    optimizer.zero_grad()

                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            max_grad_norm,
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            max_grad_norm,
                        )
                        optimizer.step()

                    # Log metrics
                    loss_val = loss.item()
                    epoch_losses.append(loss_val)
                    train_losses.append(loss_val)

                    progress.set_postfix({
                        "loss": f"{loss_val:.4f}",
                        "grad_norm": f"{grad_norm:.4f}" if isinstance(grad_norm, float) else f"{grad_norm.item():.4f}",
                    })

                    # Save step metrics
                    step_metrics = {
                        "step": global_step,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "loss": loss_val,
                        "grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    append_jsonl(self.metrics_path, step_metrics)

                    global_step += 1

                # Epoch summary
                epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0

                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {epoch_loss:.4f}"
                )

                # Save checkpoint if best
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.save_checkpoint("best", epoch, epoch_loss)

            # Save final checkpoint
            self.save_checkpoint("last", epochs - 1, epoch_loss)

        # Training summary
        summary = {
            "epochs": epochs,
            "total_steps": global_step,
            "final_loss": epoch_loss if epoch_losses else 0,
            "best_loss": best_loss,
            "train_samples": len(self.train_dataset),
        }

        save_json(summary, self.train_dir / "train_summary.json")

        # Plot loss curve
        self._plot_loss_curve(train_losses)

        return summary

    def save_checkpoint(self, name: str, epoch: int, loss: float):
        """Save checkpoint."""
        ckpt_path = self.ckpt_dir / f"{name}.pt"

        # Save LoRA adapters
        lora_path = self.ckpt_dir / f"{name}.pt_lora"
        self.model.llm.save_pretrained(str(lora_path))

        # Save metadata
        meta = {
            "epoch": epoch,
            "loss": loss,
            "lora_path": str(lora_path),
            "saved_at": datetime.now().isoformat(),
        }
        save_json(meta, ckpt_path.with_suffix(".json"))

        self.logger.info(f"Saved checkpoint: {name} (epoch={epoch}, loss={loss:.4f})")

    def _plot_loss_curve(self, losses: List[float]):
        """Plot training loss curve."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(losses, alpha=0.7)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Phase 1.5 Training Loss")
            ax.grid(True, alpha=0.3)

            # Add smoothed line
            if len(losses) > 10:
                import numpy as np
                window = min(50, len(losses) // 5)
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(losses)), smoothed, 'r-', linewidth=2, label='Smoothed')
                ax.legend()

            plt.tight_layout()
            plt.savefig(self.artifacts_dir / "loss_curve.png", dpi=150)
            plt.savefig(self.artifacts_dir / "loss_curve.pdf")
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to plot loss curve: {e}")


def run_phase15_training(
    model,
    z_pool,
    tokenizer,
    dataset_path: Path,
    run_dir: Path,
    config: Dict,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Entry point for Phase 1.5 training.

    Args:
        model: WritePhaseModel instance
        z_pool: ZPoolManager instance
        tokenizer: LLM tokenizer
        dataset_path: Path to dataset.jsonl
        run_dir: Run directory
        config: Training configuration
        max_samples: Optional sample limit

    Returns:
        Training results dict
    """
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("PHASE 1.5 TRAINING: LoRA Evidence Generation")
    logger.info("=" * 60)

    # Setup LoRA
    lora_config = config.get("lora", {
        "r": 32,
        "alpha": 64,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "dropout": 0.05,
    })
    model = setup_lora(model, lora_config)

    # Create dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    train_dataset = Phase15EvidenceDataset(
        dataset_path,
        z_pool,
        max_samples=max_samples,
    )
    logger.info(f"  Train samples: {len(train_dataset)}")

    # Create trainer
    trainer = Phase15Trainer(
        model=model,
        z_pool=z_pool,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=None,
        run_dir=run_dir,
        config=config,
    )

    # Train
    results = trainer.train()

    logger.info("Training complete!")
    logger.info(f"  Final loss: {results['final_loss']:.4f}")
    logger.info(f"  Best loss: {results['best_loss']:.4f}")

    return results
