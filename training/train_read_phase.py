"""
Phase 2: Read Phase Training Script

핵심 목표:
- Query-conditioned evidence generation 학습
- Phase 1에서 학습한 Z vectors를 query에 따라 "읽는" 법 학습
- 학습 대상: Router alignment + Evidence generation NLL

전제조건:
- Phase 1 완료: z vectors + projection 학습됨
- corpus_builder.py로 QA-aligned corpus 생성됨

사용법:
    python training/train_read_phase.py --config configs/phase2_read.yaml
    python training/train_read_phase.py --config configs/phase2_read.yaml --test
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
from omegaconf import OmegaConf

# Path setup
sys.path.append(str(Path(__file__).parent.parent))

from models.parametric_memory_llm import ParametricMemoryLLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 콘솔 핸들러
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)


def setup_file_logging(log_dir: Path, run_name: str = None):
    """파일 로깅 설정"""
    from datetime import datetime

    log_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = log_dir / f"train_{run_name}.log"

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"Log file: {log_file}")
    return log_file


class ReadPhaseDataset(Dataset):
    """
    Phase 2 Read Phase 학습용 Dataset

    각 샘플: (query, evidence, gold_doc_ids)
    """

    def __init__(
        self,
        qa_pairs: list,
        tokenizer,
        max_query_length: int = 128,
        max_evidence_length: int = 256,
    ):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_evidence_length = max_evidence_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        item = self.qa_pairs[idx]

        query = item["question"]
        evidence = item.get("evidence", "")
        answer = item.get("answer", "")
        gold_doc_ids = item.get("gold_doc_ids", [])

        # Query tokenize
        query_encoded = self.tokenizer(
            query,
            max_length=self.max_query_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Target: Evidence (또는 Evidence + Answer)
        # Phase 2에서는 Evidence 생성을 학습
        target_text = evidence if evidence else f"{answer}"
        target_encoded = self.tokenizer(
            target_text,
            max_length=self.max_evidence_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "query_ids": query_encoded["input_ids"].squeeze(0),
            "query_mask": query_encoded["attention_mask"].squeeze(0),
            "target_ids": target_encoded["input_ids"].squeeze(0),
            "target_mask": target_encoded["attention_mask"].squeeze(0),
            "gold_doc_ids": gold_doc_ids,
        }


def load_qa_data(qa_path: str) -> list:
    """QA 데이터 로드 (corpus_builder 출력)"""
    logger.info(f"Loading QA data from {qa_path}")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    logger.info(f"  Loaded {len(qa_pairs)} QA pairs")
    return qa_pairs


def create_dataloader(
    qa_pairs: list,
    tokenizer,
    batch_size: int = 2,
    max_query_length: int = 128,
    max_evidence_length: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader 생성"""
    dataset = ReadPhaseDataset(
        qa_pairs=qa_pairs,
        tokenizer=tokenizer,
        max_query_length=max_query_length,
        max_evidence_length=max_evidence_length,
    )

    def collate_fn(batch):
        query_ids = torch.stack([b["query_ids"] for b in batch])
        query_mask = torch.stack([b["query_mask"] for b in batch])
        target_ids = torch.stack([b["target_ids"] for b in batch])
        target_mask = torch.stack([b["target_mask"] for b in batch])
        gold_doc_ids = [b["gold_doc_ids"] for b in batch]

        return {
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "gold_doc_ids": gold_doc_ids,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def compute_evidence_nll(
    model: ParametricMemoryLLM,
    query_ids: torch.Tensor,
    query_mask: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    gold_doc_ids: list = None,
    use_topk: bool = True,
    topk: int = 8,
) -> torch.Tensor:
    """
    Evidence generation NLL 계산

    -log P(evidence | query, Z; θ)

    Args:
        model: ParametricMemoryLLM
        query_ids: [B, query_len]
        target_ids: [B, target_len]
        gold_doc_ids: 각 샘플의 gold document indices (routing supervision용)
        use_topk: Top-k routing 사용 여부
        topk: Top-k 문서 수
    """
    batch_size = query_ids.size(0)
    device = query_ids.device

    # 1. Memory embeddings (Top-k 또는 전체)
    if use_topk and topk < model.num_docs:
        # Top-k routing: 배치 내 각 샘플에 대해 독립적으로 Top-k 선택
        # 단순화: 배치 첫 번째 샘플의 query로 Top-k 선택 (full batch support 추후)
        topk_indices, topk_embed = model.select_topk_docs(
            query_ids[:1],  # 첫 번째 query만 사용 (배치 내 동일하다고 가정)
            k=topk,
            debug=False,
        )
        # [k, m_tokens, hidden] → [1, k*m_tokens, hidden] → [B, k*m_tokens, hidden]
        Z_embed = topk_embed.view(1, -1, topk_embed.size(-1))
        Z_embed = Z_embed.expand(batch_size, -1, -1)
        Z_len = Z_embed.size(1)
    else:
        # 전체 Z 사용
        Z_embed = model.get_memory_embeddings()  # [Z_len, hidden]
        Z_len = Z_embed.size(0)
        Z_embed = Z_embed.unsqueeze(0).expand(batch_size, -1, -1)

    # 2. Query embeddings
    query_embed = model.llm.get_input_embeddings()(query_ids)

    # 3. Target embeddings
    target_embed = model.llm.get_input_embeddings()(target_ids)

    # 4. Combined: [Z | Query | Target]
    # Z = prefix (memory), Query = context, Target = what to generate
    combined_embed = torch.cat([Z_embed, query_embed, target_embed], dim=1)

    # Attention mask
    Z_mask = torch.ones(batch_size, Z_len, device=device)
    combined_mask = torch.cat([Z_mask, query_mask.float(), target_mask.float()], dim=1)

    # 5. Labels: only compute loss on target part
    # Labels = -100 for Z + Query, target_ids for Target
    labels = torch.full_like(combined_mask, -100, dtype=torch.long)
    start_idx = Z_len + query_ids.size(1)
    labels[:, start_idx:] = target_ids

    # Mask padding in labels
    labels[labels == model.tokenizer.pad_token_id] = -100

    # 6. Forward pass
    outputs = model.llm(
        inputs_embeds=combined_embed,
        attention_mask=combined_mask,
        labels=labels,
        return_dict=True,
    )

    return outputs.loss


def train_epoch(
    model: ParametricMemoryLLM,
    dataloader: DataLoader,
    optimizer: AdamW,
    scaler: GradScaler,
    config: OmegaConf,
    epoch: int,
) -> dict:
    """한 epoch 학습"""
    model.train()

    total_loss = 0.0
    num_batches = 0

    use_amp = config.training.get("use_amp", True)
    gradient_accumulation = int(config.training.get("gradient_accumulation", 4))
    max_grad_norm = float(config.training.get("max_grad_norm", 1.0))
    use_topk = config.training.get("use_topk_routing", True)
    topk = int(config.training.get("topk_docs", 8))

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(pbar):
        query_ids = batch["query_ids"].cuda()
        query_mask = batch["query_mask"].cuda()
        target_ids = batch["target_ids"].cuda()
        target_mask = batch["target_mask"].cuda()
        gold_doc_ids = batch["gold_doc_ids"]

        # Forward
        if use_amp:
            with autocast("cuda", dtype=torch.bfloat16):
                loss = compute_evidence_nll(
                    model=model,
                    query_ids=query_ids,
                    query_mask=query_mask,
                    target_ids=target_ids,
                    target_mask=target_mask,
                    gold_doc_ids=gold_doc_ids,
                    use_topk=use_topk,
                    topk=topk,
                )
                loss = loss / gradient_accumulation

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss = compute_evidence_nll(
                model=model,
                query_ids=query_ids,
                query_mask=query_mask,
                target_ids=target_ids,
                target_mask=target_mask,
                gold_doc_ids=gold_doc_ids,
                use_topk=use_topk,
                topk=topk,
            )
            loss = loss / gradient_accumulation
            loss.backward()

            if (step + 1) % gradient_accumulation == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation
        num_batches += 1

        pbar.set_postfix({"loss": total_loss / num_batches})

    return {"train_loss": total_loss / num_batches}


@torch.no_grad()
def evaluate(
    model: ParametricMemoryLLM,
    dataloader: DataLoader,
    config: OmegaConf,
) -> dict:
    """Validation 평가"""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    use_topk = config.training.get("use_topk_routing", True)
    topk = int(config.training.get("topk_docs", 8))

    for batch in tqdm(dataloader, desc="Eval"):
        query_ids = batch["query_ids"].cuda()
        query_mask = batch["query_mask"].cuda()
        target_ids = batch["target_ids"].cuda()
        target_mask = batch["target_mask"].cuda()
        gold_doc_ids = batch["gold_doc_ids"]

        with autocast("cuda", dtype=torch.bfloat16):
            loss = compute_evidence_nll(
                model=model,
                query_ids=query_ids,
                query_mask=query_mask,
                target_ids=target_ids,
                target_mask=target_mask,
                gold_doc_ids=gold_doc_ids,
                use_topk=use_topk,
                topk=topk,
            )

        total_loss += loss.item()
        num_batches += 1

    return {"eval_loss": total_loss / num_batches}


@torch.no_grad()
def generate_samples(
    model: ParametricMemoryLLM,
    qa_pairs: list,
    num_samples: int = 5,
    max_new_tokens: int = 128,
    topk_docs: int = 8,
) -> list:
    """샘플 생성"""
    model.eval()
    samples = []

    for qa in qa_pairs[:num_samples]:
        query = qa["question"]
        gold_evidence = qa.get("evidence", "")[:200]
        gold_answer = qa.get("answer", "")

        # Query tokenize
        query_ids = model.tokenizer(
            query,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        )["input_ids"].cuda()

        # Generate
        generated = model.generate_evidence(
            query_ids=query_ids,
            max_new_tokens=max_new_tokens,
            top_k_docs=topk_docs,
        )

        samples.append({
            "query": query,
            "gold_evidence": gold_evidence,
            "gold_answer": gold_answer,
            "generated": generated[:300],
        })

    return samples


def run_read_phase_training(
    config_path: str = None,
    config: OmegaConf = None,
    test_mode: bool = False,
):
    """Phase 2 Read Phase 학습 실행"""

    # Load config
    if config is None:
        with open(config_path, "r") as f:
            config = OmegaConf.create(yaml.safe_load(f))

    # Test mode overrides
    if test_mode:
        config.data.max_samples = 100
        config.training.epochs = 2
        config.training.batch_size = 2
        logger.info("=" * 60)
        logger.info("TEST MODE: max_samples=100, epochs=2")
        logger.info("=" * 60)

    logger.info("=" * 60)
    logger.info("Phase 2: Read Phase Training")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # ==========================================
    # 1. Load QA Data
    # ==========================================
    logger.info("\n[Step 1] Loading QA Data...")

    train_qa = load_qa_data(config.data.train_path)
    val_qa = load_qa_data(config.data.val_path)

    # Max samples 제한
    max_samples = config.data.get("max_samples", None)
    if max_samples:
        train_qa = train_qa[:max_samples]
        val_qa = val_qa[:min(max_samples // 5, len(val_qa))]

    logger.info(f"Train: {len(train_qa)}, Val: {len(val_qa)}")

    # ==========================================
    # 2. Model Initialization + Phase 1 Load
    # ==========================================
    logger.info("\n[Step 2] Model Initialization + Phase 1 Load...")

    model_config = config.model
    memory_config = config.memory

    # ListConfig → list 변환 (JSON 직렬화 문제 방지)
    target_modules = model_config.lora.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
    if hasattr(target_modules, '__iter__') and not isinstance(target_modules, (str, list)):
        target_modules = list(target_modules)

    model = ParametricMemoryLLM(
        llm_name=model_config.llm_name,
        num_docs=memory_config.num_docs,
        z_dim=memory_config.z_dim,
        m_tokens=memory_config.m_tokens,
        quantization=model_config.get("quantization", "4bit"),
        lora_r=int(model_config.lora.get("r", 32)),
        lora_alpha=int(model_config.lora.get("alpha", 64)),
        lora_target_modules=target_modules,
        lora_dropout=float(model_config.lora.get("dropout", 0.05)),
    )

    # Phase 1 checkpoint 로드
    phase1_config = config.phase1
    z_pool_path = Path(phase1_config.checkpoint_dir) / "z_pool_epoch50.pt"
    projection_path = Path(phase1_config.checkpoint_dir) / "projection.pt"

    if not z_pool_path.exists():
        z_pool_path = Path(phase1_config.checkpoint_dir) / "z_pool.pt"

    logger.info(f"Loading Phase 1 from: {z_pool_path}")
    model.load_from_phase1(
        z_pool_path=str(z_pool_path),
        projection_path=str(projection_path) if projection_path.exists() else None,
    )

    # Memory stats
    mem_stats = model.get_memory_stats()
    logger.info(f"Memory Pool: {mem_stats['num_docs']} docs x {mem_stats['m_tokens']} tokens")

    # ==========================================
    # 3. DataLoader
    # ==========================================
    logger.info("\n[Step 3] Creating DataLoaders...")

    train_config = config.training

    train_dataloader = create_dataloader(
        qa_pairs=train_qa,
        tokenizer=model.tokenizer,
        batch_size=train_config.batch_size,
        max_query_length=config.data.get("max_query_length", 128),
        max_evidence_length=config.data.get("max_evidence_length", 256),
        shuffle=True,
    )

    val_dataloader = create_dataloader(
        qa_pairs=val_qa,
        tokenizer=model.tokenizer,
        batch_size=train_config.batch_size,
        max_query_length=config.data.get("max_query_length", 128),
        max_evidence_length=config.data.get("max_evidence_length", 256),
        shuffle=False,
    )

    logger.info(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

    # ==========================================
    # 4. Optimizer
    # ==========================================
    logger.info("\n[Step 4] Setting up Optimizer...")

    # Trainable params based on config
    train_z = train_config.get("train_z", True)
    train_proj = train_config.get("train_proj", True)
    train_lora = train_config.get("train_lora", True)

    param_groups = []

    if train_z:
        lr_z = float(train_config.get("lr_z", 1e-3))
        param_groups.append({
            "params": [model.memory_pool],
            "lr": lr_z,
            "name": "memory_pool",
        })
        logger.info(f"  memory_pool: lr={lr_z}")
    else:
        model.memory_pool.requires_grad = False
        logger.info("  memory_pool: FROZEN")

    if train_proj:
        lr_proj = float(train_config.get("lr_proj", 1e-4))
        param_groups.append({
            "params": model.z_to_embedding.parameters(),
            "lr": lr_proj,
            "name": "z_to_embedding",
        })
        logger.info(f"  z_to_embedding: lr={lr_proj}")
    else:
        for p in model.z_to_embedding.parameters():
            p.requires_grad = False
        logger.info("  z_to_embedding: FROZEN")

    if train_lora:
        lr_lora = float(train_config.get("lr_lora", 2e-5))
        lora_params = [p for p in model.llm.parameters() if p.requires_grad]
        param_groups.append({
            "params": lora_params,
            "lr": lr_lora,
            "name": "lora",
        })
        logger.info(f"  LoRA: lr={lr_lora}, params={len(lora_params)}")

    weight_decay = float(train_config.get("weight_decay", 0.01))
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    scaler = GradScaler("cuda") if train_config.get("use_amp", True) else None

    # ==========================================
    # 5. Training Loop
    # ==========================================
    logger.info("\n[Step 5] Training...")

    save_dir = Path(config.logging.get("save_dir", "./checkpoints/phase2_read"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 파일 로깅 설정
    log_dir = Path(config.logging.get("log_dir", "./logs/phase2_read"))
    setup_file_logging(log_dir, config.logging.get("run_name", None))

    best_eval_loss = float("inf")
    history = []

    for epoch in range(1, train_config.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{train_config.epochs}")
        logger.info("=" * 60)

        # Train
        train_metrics = train_epoch(model, train_dataloader, optimizer, scaler, config, epoch)
        logger.info(f"Train loss: {train_metrics['train_loss']:.4f}")

        # Eval
        eval_metrics = evaluate(model, val_dataloader, config)
        logger.info(f"Eval loss: {eval_metrics['eval_loss']:.4f}")

        # Save best
        if eval_metrics["eval_loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["eval_loss"]
            model.save_checkpoint(str(save_dir / "best.pt"))
            logger.info(f"  Saved best model (eval_loss={best_eval_loss:.4f})")

        # Generate samples
        if epoch % 2 == 0 or epoch == train_config.epochs:
            samples = generate_samples(model, val_qa, num_samples=3, topk_docs=config.training.get("topk_docs", 8))
            logger.info("\nSample generations:")
            for i, s in enumerate(samples):
                logger.info(f"  [{i+1}] Q: {s['query'][:60]}...")
                logger.info(f"      Gold: {s['gold_evidence'][:80]}...")
                logger.info(f"      Gen:  {s['generated'][:80]}...")

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "eval_loss": eval_metrics["eval_loss"],
        })

    # ==========================================
    # 6. Final Save
    # ==========================================
    logger.info("\n[Step 6] Saving final model...")

    model.save_checkpoint(str(save_dir / "final.pt"))

    # Save training history
    torch.save({
        "history": history,
        "config": OmegaConf.to_container(config),
        "best_eval_loss": best_eval_loss,
    }, save_dir / "results.pt")

    # Final samples
    samples = generate_samples(model, val_qa, num_samples=5, topk_docs=config.training.get("topk_docs", 8))

    with open(save_dir / "samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 2 Read Phase Training Complete!")
    logger.info(f"Best eval loss: {best_eval_loss:.4f}")
    logger.info(f"Saved to: {save_dir}")
    logger.info("=" * 60)

    return model, history


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Phase 2 Read Phase Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase2_read.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (small scale)",
    )
    args = parser.parse_args()

    run_read_phase_training(config_path=args.config, test_mode=args.test)


if __name__ == "__main__":
    main()
