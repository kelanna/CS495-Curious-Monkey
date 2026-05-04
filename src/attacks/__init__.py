from . import (
    naive,
    context_ignore,
    extraction,
    fake_completion,
    compliant_emotional,
    compliant_direct_yn,
    compliant_hypothetical,
    compliant_meta,
    compliant_roleplay,
)

REGISTRY: dict = {
    naive.ATTACK_ID: naive,
    context_ignore.ATTACK_ID: context_ignore,
    extraction.ATTACK_ID: extraction,
    fake_completion.ATTACK_ID: fake_completion,
    compliant_emotional.ATTACK_ID: compliant_emotional,
    compliant_direct_yn.ATTACK_ID: compliant_direct_yn,
    compliant_hypothetical.ATTACK_ID: compliant_hypothetical,
    compliant_meta.ATTACK_ID: compliant_meta,
    compliant_roleplay.ATTACK_ID: compliant_roleplay,
}
