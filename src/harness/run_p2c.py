"""Phase IIB batch runner — LLM as Attacker with payload freeze.

Setup (from Phase I results across all formal runs):
  Attacker : meta-llama-3.1-8b-instruct  (76.7% ASR — highest)
  Defender : anthropic/claude-sonnet-4-6 (13.3% ASR — lowest)
  Domain   : health                      (55.0% ASR — highest)
  Attacks  : top-3 by Phase I ASR
               attack2_roleplay        (70.8%)
               attack1_naive           (41.7%)
               attack3_fake_completion (33.3%)

Payload freeze:
  Rep 0 — attacker generates payload fresh.
  Reps 1–4 — exact same payload reused verbatim; attacker not called again.
"""

import json
import os
from datetime import datetime, timezone

from ..agents.prompts import SYSTEM_PROMPTS
from ..attacks.p2c_llm_attacker import (
    ATTACK_STYLES,
    ATTACKER_MODEL,
    DEFENDER_MODEL,
    run as p2c_run,
    run_with_payload,
)
from ..scoring.auto_score import auto_score
from .display import console

DOMAIN_ID = "health"
REPS      = 5
STYLE_KEYS = ["p2c_roleplay", "p2c_naive", "p2c_fake_completion"]


def run_phase_iic() -> list[dict]:
    system_prompt = SYSTEM_PROMPTS[DOMAIN_ID]
    results: list[dict] = []

    console.rule("[bold cyan]Phase IIB — LLM as Attacker[/bold cyan]")
    console.print(f"  Attacker : [yellow]{ATTACKER_MODEL}[/yellow]")
    console.print(f"  Defender : [yellow]{DEFENDER_MODEL}[/yellow]")
    console.print(f"  Domain   : [yellow]{DOMAIN_ID}[/yellow]")
    console.print(f"  Reps     : [yellow]{REPS}[/yellow] (payload frozen after rep 0)\n")

    for style_key in STYLE_KEYS:
        style = ATTACK_STYLES[style_key]
        console.rule(f"[bold]{style['name']}[/bold]")

        frozen_payload: str | None = None

        for rep in range(REPS):
            if rep == 0:
                console.print(f"  Rep {rep} — generating payload with attacker…")
                with console.status(f"Attacker ({ATTACKER_MODEL}) generating…"):
                    result = p2c_run(style_key, system_prompt, domain_id=DOMAIN_ID)
                frozen_payload = result["generated_payload"]
                console.print(f"  [dim]Payload (frozen): {frozen_payload[:120]}…[/dim]")
            else:
                console.print(f"  Rep {rep} — using frozen payload…")
                with console.status(f"Defender ({DEFENDER_MODEL}) responding…"):
                    result = run_with_payload(
                        style_key,
                        system_prompt,
                        frozen_payload,
                        domain_id=DOMAIN_ID,
                    )

            base_id = style["base_attack"]
            turn1   = result.get("turn1_response") if base_id == "attack3_fake_completion" else None
            score   = auto_score(base_id, result["response"], system_prompt, DOMAIN_ID,
                                 turn1_response=turn1, rubric_version="v2")
            result["score"]   = score
            result["success"] = score == "SUCCESS"
            result["rep"]     = rep

            icon = "[green]SUCCESS[/green]" if result["success"] else "[red]FAILURE[/red]"
            console.print(f"    → {icon}  {result['response'][:80]}")
            results.append(result)

    return results


def save_results(results: list[dict]) -> str:
    out_dir = "results/formal_p2c"
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"{out_dir}/{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Results saved → {path}[/dim]")

    try:
        from ..db import store as _store
        _store.insert_runs(results, source="formal_p2c", result_file=path)
        console.print(f"[dim]SQLite updated → {_store.DB_PATH}[/dim]")
    except Exception as exc:
        console.print(f"[dim yellow]SQLite write skipped: {exc}[/dim yellow]")

    return path


def print_summary(results: list[dict]) -> None:
    from collections import defaultdict
    console.rule("[bold]Phase IIB Summary[/bold]")
    by_style: dict[str, list] = defaultdict(list)
    for r in results:
        by_style[r["attack_id"]].append(r)
    for style_key, reps_list in by_style.items():
        n_success = sum(1 for r in reps_list if r["success"])
        asr = n_success / len(reps_list)
        name = ATTACK_STYLES[style_key]["name"]
        console.print(f"  {name}: [bold]{asr:.0%}[/bold] ASR  ({n_success}/{len(reps_list)})")


def main() -> None:
    results = run_phase_iic()
    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
