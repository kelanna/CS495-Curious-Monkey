LOCAL_BASE_URL = "http://127.0.0.1:1234/v1"
LOCAL_API_KEY = "lm-studio"

# Per-model base URL overrides (for models served on a different machine)
LOCAL_MODEL_BASE_URLS: dict[str, str] = {
    "qwen3.6-27b-mtp": "http://192.168.0.84:1234/v1",
    "qwen3.6-35b-a3b-mtp": "http://192.168.0.84:1234/v1",
    # Phase III — 9B and 27B served on secondary machine
    "qwen/qwen3.5-9b":  "http://192.168.0.84:1234/v1",
    "qwen/qwen3.5-27b": "http://192.168.0.84:1234/v1",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# LM Studio model IDs — must match exactly what is shown in LM Studio's model list
# Run one local model at a time: --models <id>
LOCAL_MODELS: list[str] = [
    "meta-llama-3.1-8b-instruct",
    "llama-3.1-8b-injectionguard",   # Phase IV fine-tuned (LoRA SFT)
    "qwen3.6-27b-mtp",
    "qwen3.6-35b-a3b-mtp",
    # Phase III — Qwen 3.5 parameter-size comparison (same series, Q8, reasoning enabled)
    "qwen3.5-0.8b",
    "qwen3.5-2b",
    "qwen3.5-4b",
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
]

# Phase III model list ordered by parameter count (smallest → largest)
PHASE3_MODELS: list[str] = [
    "qwen3.5-0.8b",
    "qwen3.5-2b",
    "qwen3.5-4b",
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
]

# Phase III attacks — top 3 from Phase I by mean ASR
# Role-play/DAN 50.9%, Naive Injection 41.4%, Fake Completion 39.0%
PHASE3_ATTACKS: list[str] = [
    "attack2_roleplay",
    "attack1_naive",
    "attack3_fake_completion",
]

# OpenRouter model IDs — all four can be passed together in one --models call
REMOTE_MODELS: list[str] = [
    "deepseek/deepseek-v4-pro",
    "openai/gpt-5.5",
    "google/gemini-3-flash-preview",
    "anthropic/claude-sonnet-4-6",
]

ALL_MODELS: list[str] = LOCAL_MODELS + REMOTE_MODELS

DEFAULT_MODELS: list[str] = LOCAL_MODELS[:1]  # run one model at a time; pass --models to override

# Phase I domains (low-stake vs high-stake)
PHASE1_DOMAINS: list[str] = ["cooking", "health"]
# Phase IIA personas
PHASE2A_DOMAINS: list[str] = ["compliant_assistant", "truth_teller"]
PHASE2_DOMAINS = PHASE2A_DOMAINS  # backwards compat alias

# Phase IIB languages
PHASE2B_LANGUAGES: list[str] = ["mandarin", "swahili", "welsh"]
# Phase IIB re-uses Phase I domains
PHASE2B_DOMAINS: list[str] = PHASE1_DOMAINS

DEFAULT_DOMAINS: list[str] = PHASE1_DOMAINS
DEFAULT_ATTACKS: list[str] = [
    "attack1_naive",
    "attack2_roleplay",
    "attack3_fake_completion",
    "attack4_extraction",
    "attack5_base64",
]
DEFAULT_REPS: int = 5

CHAT_TEMPERATURE: float = 0.7
CHAT_MAX_TOKENS: int = 200          # remote models — reasoning is handled server-side
LOCAL_MAX_TOKENS: int = 8192        # local reasoning models — reasoning eats into this budget
