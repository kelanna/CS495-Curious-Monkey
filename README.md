# Project Overview

## Project Theme
Prompt injection security for LLM-powered applications. This project builds a deliberately vulnerable Retrieval-Augmented Generation (RAG) chatbot — a literature research assistant loaded with ~10–15 public-domain books from Project Gutenberg — and uses it as a testbed to measure how effectively the defenses recommended by the [OWASP LLM Prompt Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html) stop known prompt injection attacks.

This project aims to connect with these three areas: 
1. Applied LLM engineering (RAG systems, LangChain)
2. AI security (prompt injection taxonomies, red-teaming)
3. Empirical evaluation (reproducible benchmarking across models and defenses)

## Goals
**Primary goal:** Quantify how much the three defenses — input sanitization, structured prompts with instruction-data separation, and output monitoring — reduce prompt injection attack success rate on a realistic small RAG application, applied individually and layered together.

**Secondary goal:** Test whether that reduction generalizes across multiple LLM vendors, by repeating the headline experiment on models from 4–5 different providers (mixing commercial APIs such as Qwen, DeepSeek, OpenAI, and Anthropic with open-source models accessed via OpenRouter).

**Measurable targets:**
- Implement 5 attacks. Open-Prompt-Injection benchmark (USENIX Security 2024), covering direct and indirect injection
- Implement the 3 primary defenses as independent, composable modules
- Report Attack Success Rate with 95% confidence intervals(CI), False Positive Rate on 50 benign queries, and latency overhead per defense layer
- Produce a single comparison table of attacks, defense configurations and models

**Out of scope:** inventing novel attacks or defenses, adaptive attacks, fine-tuning, multimodal attacks, agent/tool-use attacks, testing against production systems.

## Deliverables

1. **Deliberately vulnerable RAG chatbot** — a Python application built with LangChain and Chroma, loaded with ~10–15 public-domain books from Project Gutenberg, accessed through OpenRouter so the same code runs against multiple LLM backends. System prompt is locked down to make prompt injection a meaningful attack.

2. **Attack suite** — five attack modules replicating the Naive, Context-Ignore, Fake Completion, Document Poisoning (indirect), and Combined (indirect) attacks from the Open-Prompt-Injection benchmark. 

3. **Defense suite** — three independent, composable defense modules: input sanitization, structured prompts with instruction-data separation (spotlighting), and output monitoring via a secondary LLM call.

4. **Evaluation harness**— a reproducible experiment runner that iterates over (attack × defense configuration × model), logs every call to JSON, tracks cost, and emits a CSV of results. Every run records the model name, timestamp, and prompt/response for full traceability.

5. **Cross-model comparison results** — attack and defense comparison across 4–5 LLMs spanning commercial and open-source families (e.g., DeepSeek-Chat, GPT-4o-mini, Claude-3.5-Haiku, and one or two open-source models like Llama 3.1 or Qwen 2.5 via OpenRouter or local Ollama). Headline output is a single results table showing Attack Success Rate, False Positive Rate, and latency for every combination.

6. **Capstone research report** — 12–15 page paper covering threat model, attack taxonomy, methodology, experimental results, discussion, ethics statement, limitations, and future work.

7. **Red-team-style security audit report** — a 3–5 page operational document written from the attacker's perspective, with findings, severity ratings, reproduction steps, and prioritized mitigation recommendations for a hypothetical client deploying this chatbot.

8. **Demo video** — 3–5 minute demo video walkthrough showing one attack succeeding against the unprotected baseline, then being blocked when defenses are enabled.

9. **Presentation materials** — midterm and final powerpoint slides

10. **Public GitHub repository** — the full codebase with listed dependencies, setup instructions, sample attack prompts, defense configurations, the cleaned Gutenberg corpus snapshot, and all logged experiment results for reproducibility.

# 📌 Project Title

<!-- TODO: Add your project title here -->

## 📖 Project Description

<!-- TODO: Describe the problem you are solving -->

## 🎯 Objectives

<!-- TODO: List your main objectives -->

## 🧰 Tools / Technologies

<!-- TODO: e.g., Python, PyTorch, SQL, Docker, etc. -->

## 🚀 How to Run

<!-- TODO: Basic setup and execution instructions -->

## 👥 Team Members

<!-- TODO: List names -->

## 📅 Timeline (Optional)

<!-- TODO: Milestones or schedule -->

---

**Attribution Requirement:**  
Any academic, research, or commercial usage must cite the original repository and authors.