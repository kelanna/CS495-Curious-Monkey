"""Verify attack modules return the expected dict structure (chat call is mocked)."""

from unittest.mock import patch

import pytest

import src.attacks.naive as naive_mod
import src.attacks.context_ignore as ctx_mod
import src.attacks.extraction as ext_mod
import src.attacks.fake_completion as fake_mod
from src.attacks import REGISTRY

MOCK_RESPONSE = "This is a mocked model response."
MODEL = "test-model"
SYSTEM_PROMPT = "You are a test assistant."


def _mock_chat(model_id, system_prompt, user_message):
    return MOCK_RESPONSE


@patch("src.attacks.naive.chat", side_effect=_mock_chat)
def test_naive_returns_required_keys(mock_chat):
    result = naive_mod.run(MODEL, SYSTEM_PROMPT)
    assert result["attack_id"] == naive_mod.ATTACK_ID
    assert result["attack_name"] == naive_mod.ATTACK_NAME
    assert result["model"] == MODEL
    assert "payload" in result
    assert result["response"] == MOCK_RESPONSE


@patch("src.attacks.context_ignore.chat", side_effect=_mock_chat)
def test_context_ignore_returns_required_keys(mock_chat):
    result = ctx_mod.run(MODEL, SYSTEM_PROMPT)
    assert result["attack_id"] == ctx_mod.ATTACK_ID
    assert result["model"] == MODEL
    assert result["response"] == MOCK_RESPONSE


@patch("src.attacks.extraction.chat", side_effect=_mock_chat)
def test_extraction_returns_required_keys(mock_chat):
    result = ext_mod.run(MODEL, SYSTEM_PROMPT)
    assert result["attack_id"] == ext_mod.ATTACK_ID
    assert result["response"] == MOCK_RESPONSE


@patch("src.attacks.fake_completion.chat", side_effect=_mock_chat)
def test_fake_completion_returns_required_keys(mock_chat):
    result = fake_mod.run(MODEL, SYSTEM_PROMPT)
    assert result["attack_id"] == fake_mod.ATTACK_ID
    assert result["response"] == MOCK_RESPONSE


def test_registry_contains_all_phase1_attacks():
    expected = {
        "attack1_naive",
        "attack2_context_ignore",
        "attack3_extraction",
        "attack4_fake_completion",
    }
    assert expected.issubset(set(REGISTRY.keys()))


def test_registry_modules_have_run_function():
    for attack_id, module in REGISTRY.items():
        assert callable(getattr(module, "run", None)), f"{attack_id} missing run()"
