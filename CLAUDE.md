# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Comparing Prompt Injection Robustness Across Pre-Trained LLMs** — a capstone research project that benchmarks 5–7 LLMs against 10 prompt injection attack techniques across 3 agent domains (cooking, finance, medical). Produces a cross-model/cross-domain comparison table, a fine-tuning experiment (LoRA on the weakest model), and a research paper.

## Commands

```bash
make setup   # create .venv and install dependencies
make run     # python src/main.py
make test    # pytest tests/
make lint    # ruff check .
make clean   # remove .venv, caches
```

The venv is `.venv/` (Python 3.13). Activate with `source .venv/bin/activate` or prefix commands with `.venv/bin/`.

## Architecture (planned)

```
src/
  harness/
    run_experiments.py   # main entry point for running attack evaluations
  attacks/               # one module per attack technique (10 total)
  agents/                # system prompt definitions per domain
  scoring/               # auto-scoring logic (success/failure per response)
  finetuning/            # LoRA fine-tuning experiment with Unsloth
tests/
results/                 # JSON logs output by the harness
notebooks/               # Jupyter analysis and figure generation
```

## Model Access

- **Local models** (Ministral 3B, Nemotron 3B, Qwen 3.6B, Gemma 4B, GLM 4.7B): served via LM Studio on an OpenAI-compatible local API. LM Studio must be running before experiments execute.
- **Proprietary models** (GPT-4o-mini, Claude 3.5 Haiku): accessed via OpenRouter using the OpenAI SDK. Requires `OPENROUTER_API_KEY` in `.env`.

Both APIs share the same OpenAI SDK interface, so model endpoints are swapped by changing the `base_url` and `api_key` parameters.

## Key Design Constraints

- Each attack runs in an **independent session** (no shared context between runs) to avoid contamination between trials.
- Results are logged to JSON for downstream Jupyter analysis.
- Fine-tuning uses **Unsloth LoRA** and is designed to run on a laptop GPU.
- The harness must support 10 attack types × 5–7 models × 3 domains × 10–30 repetitions.
