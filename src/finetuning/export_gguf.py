"""
Phase IV — export fine-tuned LoRA model to GGUF for LM Studio.

Run this INTERACTIVELY in your terminal (not background) — it will
prompt once for sudo permission to install llama.cpp build tools.

    ~/.venvs/llm_ft/bin/python src/finetuning/export_gguf.py

Output: models/llama_ft/llama_ft_q4k/*.gguf
"""

import unsloth  # noqa: F401 — must be first

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from unsloth import FastLanguageModel

load_dotenv()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LORA_DIR  = Path("models/llama_ft/lora_adapters")
GGUF_PATH = Path("models/llama_ft/llama_ft_q4k")
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

if not LORA_DIR.exists():
    print(f"ERROR: LoRA adapters not found at {LORA_DIR}")
    print("Run train.py first.")
    sys.exit(1)

token = os.getenv("HF_TOKEN", "")
if not token:
    print("ERROR: HF_TOKEN not in .env")
    sys.exit(1)

print(f"Loading base model: {BASE_MODEL}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=512,
    load_in_4bit=True,
    token=token,
)

# Re-apply the same LoRA structure used during training
print("Re-applying LoRA structure (r=16)...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
    random_state=42,
)

# Load the saved adapter weights into the LoRA model
print(f"Loading LoRA weights from: {LORA_DIR}")
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
state_dict = load_file(str(LORA_DIR / "adapter_model.safetensors"))
result = set_peft_model_state_dict(model, state_dict)
if result.unexpected_keys:
    print(f"  Warning: unexpected keys: {result.unexpected_keys[:3]}")
print("  Adapter weights loaded OK.")

print(f"\nExporting Q4_K_M GGUF to: {GGUF_PATH}/")
print("(llama.cpp will be installed if needed — answer the prompt below)\n")
model.save_pretrained_gguf(
    str(GGUF_PATH),
    tokenizer,
    quantization_method="q4_k_m",
)

gguf_files = list(GGUF_PATH.glob("*.gguf"))
if gguf_files:
    gf = gguf_files[0]
    print(f"\nGGUF saved: {gf}")
    print(f"Size: {gf.stat().st_size / 1024**3:.1f} GB")
    print("\nTo load in LM Studio:")
    print("  1. Open LM Studio → My Models → Add Model (folder icon)")
    print(f"  2. Select: {gf.resolve()}")
    print("  3. Name it: llama-3.1-8b-injectionguard")
else:
    print(f"GGUF written to {GGUF_PATH}/ — check directory for .gguf file")
