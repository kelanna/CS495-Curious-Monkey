LOCAL_BASE_URL = "http://127.0.0.1:1234/v1"
LOCAL_API_KEY = "lm-studio"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# LM Studio model IDs must match whatever is loaded in LM Studio
LOCAL_MODELS: list[str] = [
    "mistralai/ministral-3b-instruct",
    "nvidia/nemotron-mini-3b-instruct",
    "qwen/qwen3-4b",
    "google/gemma-3-4b-it",
    "thudm/glm-4-9b-chat",
]

REMOTE_MODELS: list[str] = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-haiku",
]

ALL_MODELS: list[str] = LOCAL_MODELS + REMOTE_MODELS

DEFAULT_MODELS: list[str] = LOCAL_MODELS[:2]
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
