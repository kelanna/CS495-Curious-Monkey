import os

from dotenv import load_dotenv
from openai import OpenAI

from . import config

load_dotenv()


def _client(model_id: str) -> OpenAI:
    if model_id in config.REMOTE_MODELS:
        return OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY", ""))
    return OpenAI(base_url=config.LOCAL_BASE_URL, api_key=config.LOCAL_API_KEY)


def chat(model_id: str, system_prompt: str, user_message: str) -> str:
    client = _client(model_id)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=config.CHAT_TEMPERATURE,
        max_tokens=config.CHAT_MAX_TOKENS,
    )
    return response.choices[0].message.content or ""


def chat_turns(model_id: str, system_prompt: str, turns: list[dict]) -> str:
    """Send a multi-turn conversation and return the final assistant reply.

    `turns` is a list of {"role": "user"|"assistant", "content": str} dicts
    in chronological order. The system prompt is prepended automatically.
    """
    client = _client(model_id)
    messages = [{"role": "system", "content": system_prompt}] + turns
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=config.CHAT_TEMPERATURE,
        max_tokens=config.CHAT_MAX_TOKENS,
    )
    return response.choices[0].message.content or ""
