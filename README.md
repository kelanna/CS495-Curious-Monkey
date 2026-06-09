# Comparing Prompt Injection Robustness Across Pre-Trained LLMs

**CS495 Capstone — Kelan Huang, Bellevue College**

A systematic benchmark of six LLMs against five direct prompt injection techniques across two agent domains, extended with multilingual attacks, LLM-as-attacker evaluation, a parameter-size scaling study, and a LoRA fine-tuning defense experiment.

## Research Questions

| # | Question |
|---|---|
| RQ1 | Which LLMs best maintain their agent role under direct prompt injection? |
| RQ2 | Which attack techniques are most effective across models? |
| RQ3 | Does agent domain context moderate injection vulnerability? |
| RQ4 | Do multilingual attacks provide a systematic bypass advantage? |
| RQ5 | Can an LLM functioning as an automated attacker exceed hand-crafted payloads? |
| RQ6 | Does model parameter count affect attack success rate? |
| RQ7 | Can targeted LoRA fine-tuning substantially reduce vulnerability in a susceptible model? |

## Models Evaluated

| Model | Access | Type |
|---|---|---|
| Llama 3.1 8B | Local (LM Studio) | Open-source |
| Qwen 3.6 35B A3B MTP | Local (LM Studio) | Open-source |
| DeepSeek V4 Pro | OpenRouter | Open-weight |
| Gemini 3 Flash Preview | OpenRouter | Proprietary |
| GPT-5.5 | OpenRouter | Proprietary |
| Claude Sonnet 4.6 | OpenRouter | Proprietary |

## Attack Techniques

| Attack | Overall ASR |
|---|---|
| Role-play / DAN | 48% |
| Fake Completion | 40% |
| Naive Injection | 35% |
| Base64 Encoding | 25% |
| System Prompt Extraction | 20% |

ASRs pooled across Phases I–III (1,005 total trials, Phase IV excluded as fine-tuned model returned 0%).

## Experimental Phases

| Phase | Description | Trials |
|---|---|---|
| I | Baseline: 6 models × 5 attacks × 2 domains × 5 reps | 300 |
| IIA | Multilingual: top-3 attacks translated into Mandarin, Swahili, Welsh | 540 |
| IIB | LLM-as-attacker: Llama generates payloads targeting Claude | 15 |
| III | Parameter scaling: Qwen 3.5 series (0.8B → 27B) | 150 |
| IV | LoRA fine-tuning: Llama 3.1 8B hardened with synthetic refusal data | 50 |
| **Total** | | **1,055** |

## Key Findings

- **7× vulnerability spread** across models: Claude Sonnet 4.6 (10% ASR) to Llama 3.1 8B (79.5%)
- **3 statistically distinguishable tiers**: robust (Claude, DeepSeek), moderate (GPT-5.5, Gemini, Qwen), vulnerable (Llama)
- **Domain is the strongest moderator**: health domain 59.7% ASR vs. cooking 18.4% — driven by system prompt structure (directive vs. disclaimer-based framing)
- **Multilingual attacks show no systematic advantage** at the overall level, but attack-level interactions exist (Welsh amplifies Role-play/DAN, Mandarin amplifies Naive Injection)
- **LLM-generated Naive Injection** achieved 100% against Claude (vs. 0% hand-crafted), but the generated payload was structurally a Role-play/DAN attack — the LLM independently converged on identity displacement
- **Parameter count inversely predicts ASR**: Qwen series comparison (0.8B → 27B) shows smaller models are more vulnerable — lower parameter count correlates with higher ASR, larger models resist injection more effectively
- **LoRA fine-tuning** reduced Llama 3.1 8B from 79.5% → 0% ASR across all attacks and domains on consumer laptop hardware
- **Cost ≠ security**: DeepSeek V4 Pro matches Claude Sonnet 4.6 in robustness at a fraction of the API cost

## Repository Structure

```
src/
  harness/          # experiment runner and config
  attacks/          # attack technique implementations
  agents/           # system prompt definitions (cooking, health)
  scoring/          # automated rubric-based scorer
  finetuning/       # LoRA fine-tuning pipeline (Unsloth)
results/
  formal_v2/        # Phase I baseline + Phase IV fine-tuned results
  formal_p2a/       # Phase IIA multilingual results
  formal_p2b/       # Phase IIB LLM-as-attacker results
  formal_p3/        # Phase III Qwen parameter-size results
data/
  phase4_training.jsonl   # synthetic refusal training data
  phase4_test.jsonl       # held-out evaluation set
figures/            # paper figures and summary tables
notebooks/          # Jupyter analysis and figure generation
paper_sections_updated.tex  # LaTeX results, discussion, conclusion
poster.tex / poster.html    # conference poster
```

## Setup

**Prerequisites:** Python 3.13, [LM Studio](https://lmstudio.ai/) (for local models), Git.

```bash
git clone <repo-url>
cd CS495-Curious-Monkey
```

**Create the virtual environment and install dependencies:**

| Platform | Command |
|---|---|
| Mac / Linux | `make setup` |
| Windows | `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt` |

> `make setup` requires `make` to be installed. On Mac it ships with Xcode Command Line Tools (`xcode-select --install`); on Linux via `sudo apt install make`; on Windows via [chocolatey](https://chocolatey.org/) (`choco install make`) or use the manual command above.

**Configure API keys:**

| Platform | Command |
|---|---|
| Mac / Linux | `cp .env.example .env` |
| Windows | `copy .env.example .env` |

Open `.env` and add your `OPENROUTER_API_KEY` (required for DeepSeek, Gemini, GPT-5.5, Claude). Local models (Llama, Qwen) run via LM Studio and need no key.

**Start the results dashboard:**

| Platform | Command |
|---|---|
| Mac / Linux | `.venv/bin/streamlit run streamlit_app.py` |
| Windows | `.venv\Scripts\streamlit run streamlit_app.py` |

Then open `http://localhost:8501` in your browser.

**Run experiments** (requires LM Studio running with the target model loaded):

| Platform | Command |
|---|---|
| Mac / Linux | `make run` or `.venv/bin/python -m src.harness.run_experiments` |
| Windows | `.venv\Scripts\python -m src.harness.run_experiments` |

LM Studio's local API defaults to `http://localhost:1234` — make sure a model is loaded and the server is started before running.

**Other commands (Mac / Linux):**

```bash
make test    # pytest tests/
make lint    # ruff check .
make clean   # remove .venv and caches
```

## Fine-Tuning

Fine-tuning uses [Unsloth](https://github.com/unslothai/unsloth) LoRA on Llama 3.1 8B. **Requires a CUDA-capable GPU** (tested on a consumer laptop GPU).

Install the fine-tuning dependencies separately (not included in the main `requirements.txt` to avoid conflicts on CPU-only machines):

```bash
# Mac / Linux
.venv/bin/pip install -r requirements-finetune.txt

# Windows
.venv\Scripts\pip install -r requirements-finetune.txt
```

The fine-tuned adapter is stored in `models/llama_ft/` (not tracked in git due to size; run `src/finetuning/train.py` to reproduce it).

## Tech Stack

- **Python 3.13**, OpenAI SDK (shared interface for local and remote models)
- **LM Studio** for local model inference (OpenAI-compatible endpoint)
- **OpenRouter** for proprietary model access
- **Unsloth** for LoRA fine-tuning on consumer GPU
- **SQLite** for persistent result storage (`results/results.db`); synced from JSON logs via `src/db/store.py`
- **Streamlit** for the results visualization dashboard
- **Jupyter + pandas + matplotlib/seaborn** for analysis and figures
- **LaTeX** for the paper
