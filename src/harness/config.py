LOCAL_BASE_URL = "http://127.0.0.1:1234/v1"
LOCAL_API_KEY = "lm-studio"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# LM Studio model IDs — must match exactly what is shown in LM Studio's model list
# Run one local model at a time: --models <id>
LOCAL_MODELS: list[str] = [
    "meta-llama/llama-3.1-8b-instruct",  # verify exact ID in LM Studio
    "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2",
]

# OpenRouter model IDs — all four can be passed together in one --models call
REMOTE_MODELS: list[str] = [
    "deepseek/deepseek-chat",
    "openai/gpt-5.5",
    "google/gemini-3-flash-preview",
    "anthropic/claude-sonnet-4-6",
]

ALL_MODELS: list[str] = LOCAL_MODELS + REMOTE_MODELS

DEFAULT_MODELS: list[str] = LOCAL_MODELS[:1]  # run one model at a time; pass --models to override
DEFAULT_DOMAINS: list[str] = ["cooking"]
DEFAULT_ATTACKS: list[str] = [
    "attack1_naive",
    "attack2_context_ignore",
    "attack3_extraction",
    "attack4_fake_completion",
]
DEFAULT_REPS: int = 3

CHAT_TEMPERATURE: float = 0.7
CHAT_MAX_TOKENS: int = 200
