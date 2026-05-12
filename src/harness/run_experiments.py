"""CLI entry point: runs the model × attack × domain experiment loop."""

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

from ..attacks import REGISTRY as ATTACK_REGISTRY
from ..agents.prompts import SYSTEM_PROMPTS
from ..scoring.auto_score import auto_score
from . import config
from .display import (
    console,
    print_agent_prompt,
    print_attack,
    print_attack_summary_table,
    print_banner,
    print_model_comparison_table,
    print_phase_header,
    print_response,
    print_run_summary,
    print_section_divider,
    print_spinner,
)

_ATTACK_TYPE: dict[str, str] = {
    # Phase I
    "attack1_naive": "Naive Injection",
    "attack2_roleplay": "Role-play / DAN",
    "attack3_fake_completion": "Fake Completion",
    "attack4_extraction": "System Prompt Extraction",
    "attack5_base64": "Base64 Encoding",
    # Phase II
    "p2_authority_impersonation": "Authority Impersonation",
    "p2_moral_paradox": "Moral Paradox",
    "p2_recursive_permission": "Recursive Permission",
}


def _build_phase_summary(phase_results: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in phase_results:
        key = (r.get("attack_id", ""), r.get("model", ""), r.get("domain", ""))
        groups[key].append(r)
    rows = []
    for (attack_id, model, domain), reps_list in groups.items():
        runs_success = sum(1 for r in reps_list if r.get("success", False))
        runs_total = len(reps_list)
        rows.append({
            "attack_name": reps_list[0].get("attack_name", attack_id),
            "attack_type": _ATTACK_TYPE.get(attack_id, "Unknown"),
            "model": model,
            "domain": domain,
            "runs_success": runs_success,
            "runs_total": runs_total,
            "asr": runs_success / runs_total if runs_total else 0.0,
        })
    return rows


def _build_comparison(results: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        key = (r.get("model", ""), r.get("attack_name", r.get("attack_id", "")))
        groups[key].append(r)
    rows = []
    for (model, attack_name), reps_list in groups.items():
        runs_success = sum(1 for r in reps_list if r.get("success", False))
        runs_total = len(reps_list)
        rows.append({
            "model": model,
            "attack_name": attack_name,
            "asr": runs_success / runs_total if runs_total else 0.0,
        })
    return rows


def run_experiments(
    models: list[str],
    attack_ids: list[str],
    domain_ids: list[str],
    reps: int,
    verbose: bool = False,
) -> list[dict]:
    print_banner()
    results: list[dict] = []
    phase_num = 0

    for model_id in models:
        for domain_id in domain_ids:
            phase_num += 1
            system_prompt = SYSTEM_PROMPTS[domain_id]
            phase_results: list[dict] = []

            print_phase_header(
                phase_num,
                f"{domain_id.upper()} — {model_id}",
                f"Running {len(attack_ids)} attacks × {reps} rep(s) against the {domain_id} agent.",
            )
            print_agent_prompt(domain_id, system_prompt, model_id)

            for attack_num, attack_id in enumerate(attack_ids, start=1):
                attack_module = ATTACK_REGISTRY[attack_id]
                attack_name = getattr(attack_module, "ATTACK_NAME", attack_id)
                attack_type = _ATTACK_TYPE.get(attack_id, "Unknown")
                payload = getattr(attack_module, "PAYLOAD", "")

                print_section_divider(attack_name)

                for rep in range(reps):
                    if reps > 1:
                        print_run_summary(rep + 1, reps)
                    print_attack(attack_num, attack_name, attack_type, payload)
                    try:
                        with print_spinner(f"Querying {model_id}…"):
                            if attack_id == "attack3_fake_completion":
                                result = attack_module.run(model_id, system_prompt, domain_id=domain_id)
                            else:
                                result = attack_module.run(model_id, system_prompt)
                        result["domain"] = domain_id
                        result["rep"] = rep
                        score = auto_score(attack_id, result["response"], system_prompt, domain_id)
                        result["score"] = score
                        result["success"] = score == "SUCCESS"
                        print_response(model_id, result["response"], result["success"])
                    except Exception as exc:
                        result = {
                            "attack_id": attack_id,
                            "attack_name": attack_name,
                            "model": model_id,
                            "domain": domain_id,
                            "rep": rep,
                            "error": str(exc),
                            "score": "ERROR",
                            "success": False,
                        }
                        print_response(model_id, f"[ERROR] {exc}", False)
                    results.append(result)
                    phase_results.append(result)

            print_attack_summary_table(_build_phase_summary(phase_results))

    print_model_comparison_table(_build_comparison(results))
    return results


def save_results(results: list[dict]) -> str:
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"results/{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Results saved → {path}[/dim]")
    return path


def print_summary(results: list[dict]) -> None:
    from rich.table import Table
    scores = Counter(r.get("score", "ERROR") for r in results)
    total = len(results)
    table = Table(title="[bold]Score Summary[/bold]", header_style="bold", padding=(0, 1))
    table.add_column("Score")
    table.add_column("Count", justify="right")
    table.add_column("Pct", justify="right")
    for score, count in sorted(scores.items()):
        pct = 100 * count / total if total else 0
        table.add_row(score, str(count), f"{pct:.1f}%")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt injection experiments")
    parser.add_argument("--models", nargs="+", default=config.DEFAULT_MODELS,
                        help="Model IDs to test")
    parser.add_argument("--attacks", nargs="+", default=config.DEFAULT_ATTACKS,
                        help="Attack IDs to run (see src/attacks/)")
    parser.add_argument("--domains", nargs="+", default=config.DEFAULT_DOMAINS,
                        choices=list(SYSTEM_PROMPTS.keys()),
                        help="Agent domains to test against")
    parser.add_argument("--reps", type=int, default=config.DEFAULT_REPS,
                        help="Repetitions per combination")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment config without making any API calls")
    parser.add_argument("--max-tokens", type=int, default=config.CHAT_MAX_TOKENS,
                        help="Max tokens per model response (default: config.CHAT_MAX_TOKENS)")
    args = parser.parse_args()

    config.CHAT_MAX_TOKENS = args.max_tokens

    if args.dry_run:
        total = len(args.models) * len(args.attacks) * len(args.domains) * args.reps
        console.print(f"[yellow]Models:[/yellow]     {args.models}")
        console.print(f"[yellow]Attacks:[/yellow]    {args.attacks}")
        console.print(f"[yellow]Domains:[/yellow]    {args.domains}")
        console.print(f"[yellow]Reps:[/yellow]       {args.reps}")
        console.print(f"[yellow]Max tokens:[/yellow] {args.max_tokens}")
        console.print(f"[yellow]Total:[/yellow]      {total} API calls")
        return

    results = run_experiments(args.models, args.attacks, args.domains, args.reps)
    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
