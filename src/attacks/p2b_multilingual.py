"""Phase IIB — Multilingual attack modules.

Each MultilingualAttack wraps one Phase I attack payload translated into a
target language.  Instances are registered in __init__.py so the harness
and Streamlit UI treat them the same as Phase I attack modules.

Scoring: auto_score.py maps p2b_* IDs back to their Phase I base rubric.
When a model responds in the target language, English rubric patterns will
likely return AMBIGUOUS — flag those for manual review rather than excluding
them, since a non-English response IS itself a data point about model
behaviour under multilingual injection.
"""

from ..harness.session import chat, chat_turns
from .p2b_payloads import PAYLOADS, TURN1_PROMPTS, LANGUAGE_DISPLAY, is_ready

# Base Phase I attack names (for display)
_BASE_NAMES: dict[str, str] = {
    "attack1_naive":          "Naive Injection",
    "attack2_roleplay":       "Role-play / DAN",
    "attack3_fake_completion": "Fake Completion",
    "attack4_extraction":     "System Prompt Extraction",
    "attack5_base64":         "Base64 Encoding",
}

# Maps p2b attack ID → Phase I base attack ID (used by auto_score.py)
P2B_BASE_MAP: dict[str, str] = {}


class MultilingualAttack:
    """Mimics a Phase I attack module for a specific language."""

    def __init__(self, language: str, base_attack_id: str) -> None:
        lang_suffix = language[:2].upper() if language != "mandarin" else "ZH"
        short = base_attack_id.replace("attack", "a").replace("_fake_completion", "_fc")
        self.ATTACK_ID   = f"p2b_{base_attack_id.replace('attack', 'a')}_{language}"
        self.BASE_ATTACK_ID = base_attack_id
        self.LANGUAGE    = language
        self.ATTACK_NAME = f"{_BASE_NAMES[base_attack_id]} ({LANGUAGE_DISPLAY[language]})"
        self.PAYLOAD     = PAYLOADS[language][base_attack_id]
        self._base       = base_attack_id

    def run(self, model_id: str, system_prompt: str, domain_id: str = "cooking") -> dict:
        payload = self.PAYLOAD
        base    = self._base

        if base == "attack3_fake_completion":
            turn1 = TURN1_PROMPTS[self.LANGUAGE].get(domain_id, TURN1_PROMPTS[self.LANGUAGE]["cooking"])
            turn1_response = chat(model_id, system_prompt, turn1)
            response = chat_turns(model_id, system_prompt, [
                {"role": "user",      "content": turn1},
                {"role": "assistant", "content": turn1_response},
                {"role": "user",      "content": payload},
            ])
            return {
                "attack_id":      self.ATTACK_ID,
                "base_attack_id": self.BASE_ATTACK_ID,
                "attack_name":    self.ATTACK_NAME,
                "language":       self.LANGUAGE,
                "model":          model_id,
                "turn1_payload":  turn1,
                "turn1_response": turn1_response,
                "payload":        payload,
                "response":       response,
                "phase":          "iib",
            }

        response = chat(model_id, system_prompt, payload)
        return {
            "attack_id":      self.ATTACK_ID,
            "base_attack_id": self.BASE_ATTACK_ID,
            "attack_name":    self.ATTACK_NAME,
            "language":       self.LANGUAGE,
            "model":          model_id,
            "payload":        payload,
            "response":       response,
            "phase":          "iib",
        }


# ── Build all 15 instances (5 attacks × 3 languages) ─────────────────────────

_BASE_ATTACKS = list(_BASE_NAMES.keys())
_LANGUAGES    = ["mandarin", "swahili", "welsh"]

# Namespace dict: attack_id → MultilingualAttack instance
ALL_P2B: dict[str, "MultilingualAttack"] = {}

for _lang in _LANGUAGES:
    for _base in _BASE_ATTACKS:
        _inst = MultilingualAttack(_lang, _base)
        ALL_P2B[_inst.ATTACK_ID] = _inst
        P2B_BASE_MAP[_inst.ATTACK_ID] = _base


def get(attack_id: str) -> "MultilingualAttack":
    return ALL_P2B[attack_id]
