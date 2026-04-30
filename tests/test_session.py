"""Verify session.chat routes to the correct API endpoint (OpenAI client is mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from src.harness import config


def _make_mock_client(response_text: str):
    mock_message = MagicMock()
    mock_message.content = response_text

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@patch("src.harness.session.OpenAI")
def test_local_model_uses_lm_studio_url(mock_openai_cls):
    mock_openai_cls.return_value = _make_mock_client("hello")
    from src.harness.session import chat

    local_model = config.LOCAL_MODELS[0]
    result = chat(local_model, "system", "user msg")

    call_kwargs = mock_openai_cls.call_args.kwargs
    assert call_kwargs["base_url"] == config.LOCAL_BASE_URL
    assert call_kwargs["api_key"] == config.LOCAL_API_KEY
    assert result == "hello"


@patch("src.harness.session.OpenAI")
def test_remote_model_uses_openrouter_url(mock_openai_cls):
    mock_openai_cls.return_value = _make_mock_client("world")
    from src.harness.session import chat

    remote_model = config.REMOTE_MODELS[0]
    result = chat(remote_model, "system", "user msg")

    call_kwargs = mock_openai_cls.call_args.kwargs
    assert call_kwargs["base_url"] == config.OPENROUTER_BASE_URL
    assert result == "world"


@patch("src.harness.session.OpenAI")
def test_chat_passes_system_and_user_messages(mock_openai_cls):
    mock_client = _make_mock_client("ok")
    mock_openai_cls.return_value = mock_client
    from src.harness.session import chat

    chat(config.LOCAL_MODELS[0], "my system prompt", "my user message")

    create_call = mock_client.chat.completions.create.call_args
    messages = create_call.kwargs.get("messages") or create_call.args[0]
    # find messages by key
    msgs = create_call.kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "my system prompt"}
    assert msgs[1] == {"role": "user", "content": "my user message"}
