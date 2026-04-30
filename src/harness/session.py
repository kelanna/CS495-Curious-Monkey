import os

from dotenv import load_dotenv
from openai import OpenAI

from . import config

load_dotenv()


def chat(model_id: str, system_prompt: str, user_message: str) -> str:
    if model_id in config.REMOTE_MODELS:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)
    else:
        client = OpenAI(base_url=config.LOCAL_BASE_URL, api_key=config.LOCAL_API_KEY)

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=config.CHAT_TEMPERATURE,
        max_tokens=config.CHAT_MAX_TOKENS,
    )
    return response.choices[0].message.content
