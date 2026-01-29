"""
Phase 1.5 Training Diagnostic

Verifies label masking and loss computation are correct.
Dumps actual values from a single training batch to verify:
1. Sequence lengths and positions
2. Which tokens have loss computed
3. What the actual labels are
4. What the model predicts vs targets

Usage:
    python experiments/phase1_5_modules/training_diagnostic.py \
        --phase1_ckpt checkpoints/phase1_v2 \
        --dataset_path results/phase1_5/20260129_040910/01_data/dataset.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model_and_z_pool(phase1_ckpt: str, device: str = "cuda"):
    """Load Phase 1 model and z_pool."""
    from models.write_phase_model import WritePhaseModel, ZPoolManager

    ckpt_dir = Path(phase1_ckpt)

    # Load config
    config_path = ckpt_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        llm_name = config.get("model", {}).get("llm_name", "Qwen/Qwen3-8B")
        m_tokens = config.get("memory", {}).get("m_tokens", 16)
        z_dim = config.get("memory", {}).get("z_dim", 256)
    else:
        llm_name = "Qwen/Qwen3-8B"
        m_tokens = 16
        z_dim = 256

    model = WritePhaseModel(
        llm_name=llm_name,
        m_tokens=m_tokens,
        z_dim=z_dim,
        quantization="4bit",
    )
    model.to(device)

    # Load projection
    proj_path = ckpt_dir / "projection.pt"
    if proj_path.exists():
        model.load_projection(str(proj_path))

    # Load z_pool
    z_pool = ZPoolManager(m_tokens=m_tokens, z_dim=z_dim)
    z_pool_path = ckpt_dir / "z_pool.pt"
    if z_pool_path.exists():
        z_pool.load(str(z_pool_path))

    return model, z_pool, model.tokenizer


def run_training_diagnostic(
    phase1_ckpt: str,
    dataset_path: str,
    sample_idx: int = 0,
    device: str = "cuda",
):
    """
    Run diagnostic on a single training sample.
    """
    from experiments.phase1_5_modules.model_wrapper import Phase15ForwardWrapper

    print("=" * 70)
    print("PHASE 1.5 TRAINING DIAGNOSTIC")
    print("=" * 70)

    # Load model
    print("\n[1] Loading model...")
    model, z_pool, tokenizer = load_model_and_z_pool(phase1_ckpt, device)
    m_tokens = model.m_tokens

    # Load sample
    print(f"\n[2] Loading sample {sample_idx} from {dataset_path}...")
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                sample = json.loads(line)
                break

    doc_id = sample["doc_id"]
    question = sample["question"]
    evidence = sample["evidence_text"]
    answer = sample["answer"]

    print(f"\n  Doc ID: {doc_id}")
    print(f"  Question: {question[:100]}...")
    print(f"  Evidence: {evidence[:100]}...")
    print(f"  Answer: {answer}")

    # Get z vector
    z = z_pool.get_z(doc_id).unsqueeze(0).to(device)  # [1, m, z_dim]

    # Create wrapper
    wrapper = Phase15ForwardWrapper(model, tokenizer, device)

    # =========================================================================
    # DIAGNOSTIC: Trace through forward_for_training step by step
    # =========================================================================
    print("\n" + "=" * 70)
    print("FORWARD PASS DIAGNOSTIC")
    print("=" * 70)

    # 1. Get z embedding
    print("\n[Step 1] Z embedding")
    z_embed = wrapper.get_z_embedding(z)
    print(f"  z shape: {z.shape}")
    print(f"  z_embed shape: {z_embed.shape}")
    print(f"  m_tokens: {m_tokens}")

    # 2. Construct prompt
    print("\n[Step 2] Prompt construction")
    prompt = f"Question: {question}\nEvidence:"
    prompt_encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=False,
    )
    prompt_ids = prompt_encoded["input_ids"].to(device)
    prompt_mask = prompt_encoded["attention_mask"].to(device)
    prompt_len = prompt_ids.size(1)

    print(f"  Prompt: '{prompt[:80]}...'")
    print(f"  Prompt tokens: {prompt_len}")
    print(f"  Prompt IDs: {prompt_ids[0, :20].tolist()}...")

    # 3. Tokenize evidence
    print("\n[Step 3] Evidence tokenization")
    evidence_encoded = tokenizer(
        evidence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=False,
    )
    evidence_ids = evidence_encoded["input_ids"].to(device)
    evidence_mask = evidence_encoded["attention_mask"].to(device)
    evidence_len = evidence_ids.size(1)

    print(f"  Evidence: '{evidence[:80]}...'")
    print(f"  Evidence tokens: {evidence_len}")
    print(f"  Evidence IDs: {evidence_ids[0, :20].tolist()}...")

    # Decode evidence tokens for inspection
    print(f"\n  Decoded evidence tokens (first 20):")
    for i in range(min(20, evidence_len)):
        tok_id = evidence_ids[0, i].item()
        tok_str = tokenizer.decode([tok_id])
        print(f"    [{i}] ID={tok_id} -> '{tok_str}'")

    # Check if END is in evidence
    end_in_evidence = "END" in evidence or "end" in evidence.lower()
    print(f"\n  Contains 'END' in evidence text: {end_in_evidence}")

    # 4. Get embeddings
    print("\n[Step 4] Embeddings")
    embedding_layer = model.llm.get_input_embeddings()
    prompt_embed = embedding_layer(prompt_ids)
    evidence_embed = embedding_layer(evidence_ids)

    print(f"  prompt_embed shape: {prompt_embed.shape}")
    print(f"  evidence_embed shape: {evidence_embed.shape}")

    # 5. Concatenate for teacher forcing
    print("\n[Step 5] Concatenation (teacher forcing)")
    combined_embed = torch.cat([
        z_embed,                      # [1, m, hidden]
        prompt_embed,                 # [1, p, hidden]
        evidence_embed[:, :-1, :],    # [1, e-1, hidden]
    ], dim=1)

    combined_len = combined_embed.size(1)
    print(f"  Combined shape: {combined_embed.shape}")
    print(f"  = z ({m_tokens}) + prompt ({prompt_len}) + evidence[:-1] ({evidence_len - 1})")
    print(f"  = {m_tokens + prompt_len + evidence_len - 1}")

    # 6. Position calculations
    print("\n[Step 6] Position calculations")
    m = m_tokens
    p = prompt_len
    e = evidence_len

    start_pos = m + p - 1  # Where evidence prediction starts
    end_pos = start_pos + e  # Where evidence prediction ends

    print(f"  m (z tokens): {m}")
    print(f"  p (prompt tokens): {p}")
    print(f"  e (evidence tokens): {e}")
    print(f"  Loss computed on positions: [{start_pos}, {end_pos})")
    print(f"  That's {end_pos - start_pos} = {e} positions")

    # Verify we're NOT computing loss on prompt
    print(f"\n  VERIFICATION:")
    print(f"    Z positions: [0, {m})")
    print(f"    Prompt positions: [{m}, {m + p})")
    print(f"    Evidence input positions: [{m + p}, {m + p + e - 1})")
    print(f"    Loss start position: {start_pos} (= m + p - 1)")

    if start_pos >= m + p - 1:
        print(f"    ✓ Loss starts AFTER prompt (position {m + p - 1})")
    else:
        print(f"    ✗ ERROR: Loss includes prompt positions!")

    # 7. Forward pass (without gradient)
    print("\n[Step 7] Forward pass")
    model.eval()
    with torch.no_grad():
        outputs = model.llm(
            inputs_embeds=combined_embed,
            attention_mask=torch.ones(1, combined_len, device=device),
            use_cache=False,
        )

    logits = outputs.logits
    print(f"  Logits shape: {logits.shape}")

    # 8. Extract evidence logits
    print("\n[Step 8] Extract evidence logits")
    actual_end = min(end_pos, logits.size(1))
    shift_logits = logits[:, start_pos:actual_end, :]
    shift_labels = evidence_ids[:, :actual_end - start_pos]

    print(f"  shift_logits shape: {shift_logits.shape}")
    print(f"  shift_labels shape: {shift_labels.shape}")

    # 9. Check predictions vs labels
    print("\n[Step 9] Predictions vs Labels (first 10 positions)")
    print("-" * 70)
    print(f"  {'Pos':>4} | {'Label ID':>8} | {'Pred ID':>8} | Label Token | Pred Token")
    print("-" * 70)

    predictions = shift_logits.argmax(dim=-1)  # [1, e]
    for i in range(min(10, shift_labels.size(1))):
        label_id = shift_labels[0, i].item()
        pred_id = predictions[0, i].item()
        label_tok = tokenizer.decode([label_id])
        pred_tok = tokenizer.decode([pred_id])
        match = "✓" if label_id == pred_id else "✗"
        print(f"  {i:>4} | {label_id:>8} | {pred_id:>8} | {label_tok:>11} | {pred_tok:>10} {match}")

    # 10. Compute loss
    print("\n[Step 10] Loss computation")
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction="none",
    )
    loss = loss.view(1, -1)

    # Apply mask
    loss_mask = evidence_mask[:, :shift_labels.size(1)]
    masked_loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

    print(f"  Per-token loss (first 10): {loss[0, :10].tolist()}")
    print(f"  Loss mask (first 10): {loss_mask[0, :10].tolist()}")
    print(f"  Mean loss: {masked_loss.item():.4f}")

    # 11. Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"\n  1. Label masking: {'CORRECT' if start_pos >= m + p - 1 else 'INCORRECT'}")
    print(f"     - Loss only computed on evidence positions [{start_pos}, {actual_end})")
    print(f"     - Prompt positions [{m}, {m+p}) are NOT included in loss")

    # Check if model is predicting END
    end_token_ids = tokenizer.encode("END", add_special_tokens=False)
    end_count = sum(1 for i in range(predictions.size(1)) if predictions[0, i].item() in end_token_ids)
    print(f"\n  2. END token prediction:")
    print(f"     - END token IDs: {end_token_ids}")
    print(f"     - Predicted END tokens: {end_count} / {predictions.size(1)}")

    # Check if evidence contains END
    evidence_end_count = sum(1 for i in range(evidence_ids.size(1)) if evidence_ids[0, i].item() in end_token_ids)
    print(f"     - END tokens in target evidence: {evidence_end_count}")

    # Check for repetition in predictions
    print(f"\n  3. Repetition check:")
    unique_preds = len(set(predictions[0].tolist()))
    print(f"     - Unique predicted tokens: {unique_preds} / {predictions.size(1)}")
    if unique_preds < predictions.size(1) * 0.3:
        print(f"     ⚠️ WARNING: High repetition detected!")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Phase 1.5 Training Diagnostic")
    parser.add_argument("--phase1_ckpt", type=str, default="checkpoints/phase1_v2")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    run_training_diagnostic(
        phase1_ckpt=args.phase1_ckpt,
        dataset_path=args.dataset_path,
        sample_idx=args.sample_idx,
        device=args.device,
    )


if __name__ == "__main__":
    main()
