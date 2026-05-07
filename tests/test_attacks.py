"""Verify attack modules return the expected dict structure (chat call is mocked)."""

from unittest.mock import patch

import src.attacks.naive as naive_mod
import src.attacks.attack2_roleplay as roleplay_mod
import src.attacks.extraction as ext_mod
import src.attacks.fake_completion as fake_mod
import src.attacks.attack5_base64 as base64_mod
import src.attacks.p2_authority_impersonation as auth_mod
from src.attacks import REGISTRY, PHASE1_ATTACKS, PHASE2_ATTACKS

MOCK_RESPONSE = "This is a mocked model response."
MODEL = "test-model"
SYSTEM_PROMPT = "You are a test assistant."


def _mock_chat(model_id, system_prompt, user_message):
    return MOCK_RESPONSE


def _mock_chat_turns(model_id, system_prompt, turns):
    return MOCK_RESPONSE


@patch("src.attacks.naive.chat", side_effect=_mock_chat)
def test_naive_returns_required_keys(mock_chat):
    result = naive_mod.run(MODEL, SYSTEM_PROMPT)
    assert result["attack_id"] == naive_mod.ATTACK_ID
    assert result["attack_name"] == naive_mod.ATTACK_NAME
    assert result["model"] == MODEL
    assert "payload" in result
    assert result["response"] == MOCK_RESPONSE


@patch("src.attacks.attack2_roleplay.chat", side_effect=_mock_chat)
def test_roleplay_returns_required_keys(mock_chat):
    result = roleplay_mod.run(MODEL, SYSTEM_PROMPT)
    assert result["attack_id"] == roleplay_mod.ATTACK_ID
    assert result["model"] == MODEL
    assert result["response"] == MOCK_RESPONSE


@patch("src.attacks.extraction.chat", side_effect=_mock_chat)
def test_extraction_returns_required_keys(mock_chat):
    result = ext_mod.run(MODEL, SYSTEM_PROMPT)
    assert result["attack_id"] == ext_mod.ATTACK_ID
    assert result["response"] == MOCK_RESPONSE


@patch("src.attacks.fake_completion.chat_turns", side_effect=_mock_chat_turns)
@patch("src.attacks.fake_completion.chat", side_effect=_mock_chat)
def test_fake_completion_returns_required_keys(mock_chat, mock_turns):
    result = fake_mod.run(MODEL, SYSTEM_PROMPT)
    assert result["attack_id"] == fake_mod.ATTACK_ID
    assert "turn1_payload" in result
    assert "turn1_response" in result
    assert result["response"] == MOCK_RESPONSE


@patch("src.attacks.attack5_base64.chat", side_effect=_mock_chat)
def test_base64_returns_required_keys(mock_chat):
    result = base64_mod.run(MODEL, SYSTEM_PROMPT)
    assert result["attack_id"] == base64_mod.ATTACK_ID
    assert result["response"] == MOCK_RESPONSE


@patch("src.attacks.p2_authority_impersonation.chat", side_effect=_mock_chat)
def test_authority_impersonation_returns_required_keys(mock_chat):
    result = auth_mod.run(MODEL, SYSTEM_PROMPT)
    assert result["attack_id"] == auth_mod.ATTACK_ID
    assert result["response"] == MOCK_RESPONSE


def test_registry_contains_all_phase1_attacks():
    assert set(PHASE1_ATTACKS).issubset(set(REGISTRY.keys()))


def test_registry_contains_all_phase2_attacks():
    assert set(PHASE2_ATTACKS).issubset(set(REGISTRY.keys()))


def test_registry_modules_have_run_function():
    for attack_id, module in REGISTRY.items():
        assert callable(getattr(module, "run", None)), f"{attack_id} missing run()"
