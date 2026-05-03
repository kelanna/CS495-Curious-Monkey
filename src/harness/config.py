LOCAL_BASE_URL = "http://127.0.0.1:1234/v1"
LOCAL_API_KEY = "lm-studio"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# LM Studio model IDs — must match exactly what is shown in LM Studio's model list
LOCAL_MODELS: list[str] = [
    "mistralai/ministral-3-3b",
    "nvidia/nemotron-3-nano-4b",
    "qwen3.5-4b",
    "qwen3.6-35b-a3b@iq3_s",
    "qwen3.5-9b-claude-4.6-os-auto-variable-heretic-uncensored-thinking-max-neocode-imatrix",
]

REMOTE_MODELS: list[str] = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "deepseek/deepseek-chat",
    "anthropic/claude-3-5-haiku",
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
CHAT_MAX_TOKENS: int = 1000
