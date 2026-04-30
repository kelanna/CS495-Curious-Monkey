"""CLI entry point: runs the model × attack × domain experiment loop."""

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone

from ..attacks import REGISTRY as ATTACK_REGISTRY
from ..agents.prompts import SYSTEM_PROMPTS
from ..scoring.auto_score import auto_score
from . import config


def run_experiments(
    models: list[str],
    attack_ids: list[str],
    domain_ids: list[str],
    reps: int,
) -> list[dict]:
    results: list[dict] = []
    total = len(models) * len(attack_ids) * len(domain_ids) * reps
    done = 0

    for model_id in models:
        for domain_id in domain_ids:
            system_prompt = SYSTEM_PROMPTS[domain_id]
            for attack_id in attack_ids:
                attack_module = ATTACK_REGISTRY[attack_id]
                for rep in range(reps):
                    done += 1
                    print(f"[{done}/{total}] {model_id} | {domain_id} | {attack_id} | rep {rep + 1}")
                    try:
                        result = attack_module.run(model_id, system_prompt)
                        result["domain"] = domain_id
                        result["rep"] = rep
                        score = auto_score(attack_id, result["response"], system_prompt)
                        result["score"] = score
                        result["success"] = score == "SUCCESS"
                    except Exception as exc:
                        result = {
                            "attack_id": attack_id,
                            "model": model_id,
                            "domain": domain_id,
                            "rep": rep,
                            "error": str(exc),
                            "score": "ERROR",
                            "success": False,
                        }
                    print(f"  → {result['score']}")
                    results.append(result)

    return results


def save_results(results: list[dict]) -> str:
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"results/{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {path}")
    return path


def print_summary(results: list[dict]) -> None:
    scores = Counter(r.get("score", "ERROR") for r in results)
    total = len(results)
    print("\nSummary:")
    for score, count in sorted(scores.items()):
        pct = 100 * count / total if total else 0
        print(f"  {score:<12} {count:>4}  ({pct:.1f}%)")


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
    args = parser.parse_args()

    if args.dry_run:
        total = len(args.models) * len(args.attacks) * len(args.domains) * args.reps
        print(f"Models:  {args.models}")
        print(f"Attacks: {args.attacks}")
        print(f"Domains: {args.domains}")
        print(f"Reps:    {args.reps}")
        print(f"Total:   {total} API calls")
        return

    results = run_experiments(args.models, args.attacks, args.domains, args.reps)
    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
