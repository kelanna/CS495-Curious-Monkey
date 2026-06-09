"""
Phase IV — LoRA fine-tuning of Llama 3.1 8B on Phase I refusal dataset.

Uses Unsloth 4-bit quantisation + LoRA, TRL SFTTrainer for training.
Saves LoRA adapters AND a merged Q4_K_M GGUF for direct loading in LM Studio.

Run with the llm_ft venv (NOT the project .venv):
    ~/.venvs/llm_ft/bin/python src/finetuning/train.py [OPTIONS]

Prerequisites:
    1. Accept Meta's Llama 3.1 license at:
       https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
    2. Add your HuggingFace token to .env:
       HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

Options:
    --data PATH       Training JSONL  (default: data/phase4_training.jsonl)
    --out DIR         Output root dir (default: models/llama_ft)
    --model ID        HF model ID     (default: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)
    --epochs N        Training epochs (default: 3)
    --rank N          LoRA rank       (default: 16)
    --batch N         Per-device batch size (default: 2)
    --grad-accum N    Gradient accumulation steps (default: 4)
    --lr FLOAT        Learning rate   (default: 2e-4)
    --max-seq N       Max sequence length in tokens (default: 1024)
    --no-gguf         Skip GGUF export (faster, use if only testing training)
"""

import unsloth  # noqa: F401 — must be the very first import for Unsloth patches

import argparse
import json
import os
import sys
import time
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

load_dotenv()

# Reduce VRAM fragmentation — important on 8 GB cards with a live desktop
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_hf_token() -> str:
    token = os.getenv("HF_TOKEN", "")
    if not token:
        print(
            "\nERROR: HF_TOKEN not found in .env\n"
            "Llama 3.1 is a gated model. To fix:\n"
            "  1. Accept the Meta licence at:\n"
            "     https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct\n"
            "  2. Create a token at https://huggingface.co/settings/tokens\n"
            "  3. Add  HF_TOKEN=hf_xxx  to your .env file\n"
        )
        sys.exit(1)
    return token


def _load_dataset(data_path: Path, tokenizer) -> Dataset:
    records = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(records)} training records from {data_path}")

    def _apply_template(rec):
        text = tokenizer.apply_chat_template(
            rec["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(records)
    dataset = dataset.map(_apply_template, remove_columns=["messages"])
    return dataset


def _print_vram() -> None:
    try:
        import torch
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {used:.1f} / {total:.1f} GB used")
    except Exception:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data",      type=Path,  default=Path("data/phase4_training.jsonl"))
    parser.add_argument("--out",       type=Path,  default=Path("models/llama_ft"))
    parser.add_argument("--model",     type=str,   default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument("--epochs",    type=int,   default=3)
    parser.add_argument("--rank",      type=int,   default=16)
    parser.add_argument("--batch",     type=int,   default=1)
    parser.add_argument("--grad-accum",type=int,   default=8,   dest="grad_accum")
    parser.add_argument("--lr",        type=float, default=2e-4)
    parser.add_argument("--max-seq",   type=int,   default=512, dest="max_seq")
    parser.add_argument("--no-gguf",   action="store_true")
    args = parser.parse_args()

    hf_token = _check_hf_token()

    lora_dir  = args.out / "lora_adapters"
    gguf_path = args.out / "llama_ft_q4k"
    log_path  = args.out / "training_log.json"
    args.out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model}")
    print(f"{'='*60}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        token=hf_token,
    )
    _print_vram()

    # ── 2. Apply LoRA ─────────────────────────────────────────────────────────
    print(f"\nApplying LoRA  r={args.rank}  alpha={args.rank * 2}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.rank * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()
    _print_vram()

    # ── 3. Load dataset ───────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.data}")
    dataset = _load_dataset(args.data, tokenizer)
    print(f"Example (truncated):\n  {dataset[0]['text'][:200]!r}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    effective_batch = args.batch * args.grad_accum
    steps_per_epoch = len(dataset) // effective_batch
    total_steps = steps_per_epoch * args.epochs

    print(f"\n{'='*60}")
    print(f"Training plan")
    print(f"  Examples:          {len(dataset)}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Batch (device):    {args.batch}")
    print(f"  Grad accum:        {args.grad_accum}  →  effective batch {effective_batch}")
    print(f"  Steps per epoch:   {steps_per_epoch}")
    print(f"  Total steps:       {total_steps}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  LoRA rank:         {args.rank}")
    print(f"{'='*60}\n")

    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=args.max_seq,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=max(10, total_steps // 10),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        output_dir=str(args.out / "checkpoints"),
        save_strategy="no",          # save only at the end
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Final loss: {result.training_loss:.4f}")
    _print_vram()

    # ── 5. Save log ───────────────────────────────────────────────────────────
    log = {
        "model": args.model,
        "epochs": args.epochs,
        "rank": args.rank,
        "effective_batch": effective_batch,
        "lr": args.lr,
        "total_steps": total_steps,
        "training_loss": result.training_loss,
        "elapsed_seconds": round(elapsed),
        "n_examples": len(dataset),
    }
    log_path.write_text(json.dumps(log, indent=2))
    print(f"Training log → {log_path}")

    # ── 6. Save LoRA adapters ─────────────────────────────────────────────────
    print(f"\nSaving LoRA adapters → {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print("LoRA adapters saved.")

    # ── 7. Export GGUF for LM Studio ─────────────────────────────────────────
    if not args.no_gguf:
        print(f"\nExporting merged Q4_K_M GGUF → {gguf_path}/")
        print("(This merges LoRA weights and quantises — takes ~5 min)")
        model.save_pretrained_gguf(
            str(gguf_path),
            tokenizer,
            quantization_method="q4_k_m",
        )
        gguf_files = list(gguf_path.parent.glob("*.gguf")) + list(gguf_path.glob("*.gguf"))
        if gguf_files:
            print(f"GGUF saved: {gguf_files[0]}")
            print("\nTo use in LM Studio:")
            print(f"  1. Open LM Studio → My Models → Add Model")
            print(f"  2. Point to: {gguf_files[0].resolve()}")
            print(f"  3. Name it:  llama-3.1-8b-injectionguard")
        else:
            print(f"GGUF files written to {gguf_path}/")
    else:
        print("\nSkipping GGUF export (--no-gguf set)")

    print(f"\n{'='*60}")
    print("Phase IV training complete.")
    print(f"  LoRA adapters: {lora_dir}")
    if not args.no_gguf:
        print(f"  GGUF for LM Studio: {gguf_path}/")
    print(f"  Training log:  {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
