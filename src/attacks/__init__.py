from . import naive, context_ignore, extraction, fake_completion

REGISTRY: dict = {
    naive.ATTACK_ID: naive,
    context_ignore.ATTACK_ID: context_ignore,
    extraction.ATTACK_ID: extraction,
    fake_completion.ATTACK_ID: fake_completion,
}
