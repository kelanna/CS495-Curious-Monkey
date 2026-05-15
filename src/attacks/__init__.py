from . import (
    # Phase I
    naive,
    attack2_roleplay,
    fake_completion,
    extraction,
    attack5_base64,
    # Phase IIA
    p2_authority_impersonation,
    p2_moral_paradox,
    p2_recursive_permission,
    # Phase IIB
    p2b_multilingual,
)

REGISTRY: dict = {
    # Phase I — Core Attack × Domain
    naive.ATTACK_ID: naive,
    attack2_roleplay.ATTACK_ID: attack2_roleplay,
    fake_completion.ATTACK_ID: fake_completion,
    extraction.ATTACK_ID: extraction,
    attack5_base64.ATTACK_ID: attack5_base64,
    # Phase IIA — Cognitive Schema Attacks
    p2_authority_impersonation.ATTACK_ID: p2_authority_impersonation,
    p2_moral_paradox.ATTACK_ID: p2_moral_paradox,
    p2_recursive_permission.ATTACK_ID: p2_recursive_permission,
    # Phase IIB — Multilingual (15 instances: 5 attacks × 3 languages)
    **p2b_multilingual.ALL_P2B,
}

PHASE1_ATTACKS = [
    naive.ATTACK_ID,
    attack2_roleplay.ATTACK_ID,
    fake_completion.ATTACK_ID,
    extraction.ATTACK_ID,
    attack5_base64.ATTACK_ID,
]

PHASE2A_ATTACKS = [
    p2_authority_impersonation.ATTACK_ID,
    p2_moral_paradox.ATTACK_ID,
    p2_recursive_permission.ATTACK_ID,
]

PHASE2_ATTACKS = PHASE2A_ATTACKS  # backwards compat alias

PHASE2B_ATTACKS = list(p2b_multilingual.ALL_P2B.keys())
