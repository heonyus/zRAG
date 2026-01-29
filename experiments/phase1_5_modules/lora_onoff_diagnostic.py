"""
LoRA ON/OFF Diagnostic Script

Compares model output with LoRA enabled vs disabled for the same input.
This helps verify if LoRA is actually affecting generation.

Usage:
    python experiments/phase1_5_modules/lora_onoff_diagnostic.py \
        --phase1_ckpt checkpoints/phase1_v2 \
        --lora_ckpt results/phase1_5/20260129_040910/02_train/checkpoints/best.pt_lora \
        --corpus_dir checkpoints/phase2_corpus \
        --sample_id 8 \
        --doc_id doc_36 \
        --question "Craig Serling and Jeff Celentano, are of which nationality?"

Or use --dataset_path to load sample from dataset:
    python experiments/phase1_5_modules/lora_onoff_diagnostic.py \
        --phase1_ckpt checkpoints/phase1_v2 \
        --lora_ckpt results/phase1_5/20260129_040910/02_train/checkpoints/best.pt_lora \
        --dataset_path results/phase1_5/20260129_040910/01_data/dataset.jsonl \
        --sample_id 8
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model_and_z_pool(phase1_ckpt: str, device: str = "cuda"):
    """Load Phase 1 model and z_pool."""
    from models.write_phase_model import WritePhaseModel, ZPoolManager

    ckpt_dir = Path(phase1_ckpt)
    print(f"Loading model from: {ckpt_dir}")

    # Load config if available
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

    print(f"  Model: {llm_name}, m_tokens={m_tokens}, z_dim={z_dim}")

    # Initialize model
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
        print(f"  Loaded projection from {proj_path}")
        print(f"  Alpha: {model.alpha.item():.4f}")

    # Load z_pool
    z_pool = ZPoolManager(m_tokens=m_tokens, z_dim=z_dim)
    z_pool_path = ckpt_dir / "z_pool.pt"

    if z_pool_path.exists():
        z_pool.load(str(z_pool_path))
        print(f"  Loaded z_pool from {z_pool_path} ({len(z_pool.doc_ids)} docs)")
    else:
        # Try epoch checkpoint
        epoch_path = ckpt_dir / "z_pool_epoch50.pt"
        if epoch_path.exists():
            ckpt = torch.load(epoch_path, map_location="cpu")
            if "z_vectors" in ckpt:
                for doc_id, z_tensor in ckpt["z_vectors"].items():
                    z_pool.add(doc_id, z_tensor)
                print(f"  Loaded z_pool from {epoch_path} ({len(z_pool.doc_ids)} docs)")

    # Get tokenizer
    tokenizer = model.tokenizer

    return model, z_pool, tokenizer


def generate_with_lora_status(
    wrapper,
    z,
    query: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    """Generate evidence with current LoRA status."""
    # Print generation config
    print(f"\n  Generation Config:")
    print(f"    max_new_tokens: {max_new_tokens}")
    print(f"    do_sample: {do_sample}")
    print(f"    temperature: {temperature}")

    evidence = wrapper.generate_evidence(
        z=z,
        query_text=query,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )

    return evidence


def run_lora_onoff_comparison(
    phase1_ckpt: str,
    lora_ckpt: str,
    doc_id: str,
    question: str,
    answer: str = None,
    target_evidence: str = None,
    max_new_tokens: int = 256,
    device: str = "cuda",
):
    """
    Compare LoRA ON vs OFF outputs for the same input.
    """
    from experiments.phase1_5_modules.model_wrapper import Phase15ForwardWrapper
    from peft import PeftModel

    print("=" * 70)
    print("LoRA ON/OFF DIAGNOSTIC")
    print("=" * 70)
    print(f"\nDoc ID: {doc_id}")
    print(f"Question: {question}")
    if answer:
        print(f"Gold Answer: {answer}")
    print()

    # 1. Load base model (no LoRA)
    print("-" * 40)
    print("Step 1: Loading base model (NO LoRA)")
    print("-" * 40)

    model, z_pool, tokenizer = load_model_and_z_pool(phase1_ckpt, device)

    # Get z vector
    z = z_pool.get_z(doc_id).to(device)

    # Create wrapper
    wrapper = Phase15ForwardWrapper(model, tokenizer, device)

    # Generate WITHOUT LoRA
    print("\n[LoRA OFF] Generating evidence...")
    model.eval()
    with torch.no_grad():
        evidence_off = generate_with_lora_status(
            wrapper, z, question, max_new_tokens, do_sample=False
        )

    print(f"\n[LoRA OFF] Generated Evidence:")
    print("-" * 40)
    print(evidence_off)
    print("-" * 40)

    # 2. Load LoRA checkpoint
    print("\n")
    print("-" * 40)
    print("Step 2: Loading LoRA checkpoint")
    print("-" * 40)
    print(f"LoRA path: {lora_ckpt}")

    # Check if LoRA exists
    lora_path = Path(lora_ckpt)
    if not lora_path.exists():
        print(f"ERROR: LoRA checkpoint not found: {lora_path}")
        return

    # Check adapter_config.json
    config_path = lora_path / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            adapter_config = json.load(f)
        print(f"  LoRA rank (r): {adapter_config.get('r', 'N/A')}")
        print(f"  LoRA alpha: {adapter_config.get('lora_alpha', 'N/A')}")
        print(f"  Target modules: {adapter_config.get('target_modules', 'N/A')}")

    # Load LoRA
    model.llm = PeftModel.from_pretrained(model.llm, str(lora_path))
    print("LoRA loaded successfully!")

    # Count trainable params
    total_params = sum(p.numel() for p in model.llm.parameters())
    trainable_params = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable (LoRA): {trainable_params:,}")

    # Recreate wrapper with LoRA-enhanced model
    wrapper = Phase15ForwardWrapper(model, tokenizer, device)

    # Generate WITH LoRA
    print("\n[LoRA ON] Generating evidence...")
    model.eval()
    with torch.no_grad():
        evidence_on = generate_with_lora_status(
            wrapper, z, question, max_new_tokens, do_sample=False
        )

    print(f"\n[LoRA ON] Generated Evidence:")
    print("-" * 40)
    print(evidence_on)
    print("-" * 40)

    # 3. Comparison
    print("\n")
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    # Check if outputs are identical
    if evidence_off == evidence_on:
        print("\n⚠️  WARNING: LoRA ON/OFF outputs are IDENTICAL!")
        print("    This suggests LoRA may not be affecting generation.")
        print("    Possible causes:")
        print("    1. LoRA weights are near-zero (under-trained)")
        print("    2. LoRA not properly loaded")
        print("    3. Wrong target modules")
    else:
        print("\n✓ LoRA ON/OFF outputs are DIFFERENT (expected)")
        print(f"  OFF length: {len(evidence_off)} chars")
        print(f"  ON length: {len(evidence_on)} chars")

        # Show diff
        print("\n[DIFF] First 200 chars:")
        print(f"  OFF: {evidence_off[:200]}...")
        print(f"  ON:  {evidence_on[:200]}...")

    # Check answer coverage
    if answer:
        ans_in_off = answer.lower() in evidence_off.lower()
        ans_in_on = answer.lower() in evidence_on.lower()

        print(f"\nAnswer Coverage:")
        print(f"  LoRA OFF: {'✓ Contains answer' if ans_in_off else '✗ Missing answer'}")
        print(f"  LoRA ON:  {'✓ Contains answer' if ans_in_on else '✗ Missing answer'}")

    # Show target if available
    if target_evidence:
        print(f"\n[TARGET Evidence]:")
        print("-" * 40)
        print(target_evidence[:500])
        if len(target_evidence) > 500:
            print("... (truncated)")
        print("-" * 40)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    return {
        "doc_id": doc_id,
        "question": question,
        "answer": answer,
        "evidence_lora_off": evidence_off,
        "evidence_lora_on": evidence_on,
        "outputs_identical": evidence_off == evidence_on,
        "answer_in_off": answer.lower() in evidence_off.lower() if answer else None,
        "answer_in_on": answer.lower() in evidence_on.lower() if answer else None,
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA ON/OFF Diagnostic")

    parser.add_argument("--phase1_ckpt", type=str, required=True,
                       help="Path to Phase 1 checkpoint")
    parser.add_argument("--lora_ckpt", type=str, required=True,
                       help="Path to LoRA checkpoint (best.pt_lora directory)")

    # Option 1: Direct specification
    parser.add_argument("--doc_id", type=str, help="Document ID")
    parser.add_argument("--question", type=str, help="Question text")
    parser.add_argument("--answer", type=str, help="Gold answer (optional)")

    # Option 2: Load from dataset
    parser.add_argument("--dataset_path", type=str,
                       help="Path to dataset.jsonl to load sample from")
    parser.add_argument("--sample_id", type=int, default=8,
                       help="Sample ID to test (from dataset)")

    parser.add_argument("--corpus_dir", type=str,
                       default="checkpoints/phase2_corpus",
                       help="Corpus directory")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Max tokens to generate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")

    args = parser.parse_args()

    # Get sample info
    doc_id = args.doc_id
    question = args.question
    answer = args.answer
    target_evidence = None

    if args.dataset_path:
        # Load from dataset
        print(f"Loading sample {args.sample_id} from {args.dataset_path}")
        with open(args.dataset_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample.get("sample_id") == args.sample_id:
                    doc_id = sample["doc_id"]
                    question = sample["question"]
                    answer = sample.get("answer")
                    target_evidence = sample.get("evidence_text")
                    break

        if not doc_id:
            print(f"ERROR: Sample ID {args.sample_id} not found in dataset")
            return

    if not doc_id or not question:
        print("ERROR: Must specify --doc_id and --question, or --dataset_path with --sample_id")
        return

    # Run comparison
    run_lora_onoff_comparison(
        phase1_ckpt=args.phase1_ckpt,
        lora_ckpt=args.lora_ckpt,
        doc_id=doc_id,
        question=question,
        answer=answer,
        target_evidence=target_evidence,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()
