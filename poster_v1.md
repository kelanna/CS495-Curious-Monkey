::::::::::::::::::::::::: multicols
3

::: pb
Abstract Prompt injection has been the **#1 OWASP LLM threat since
2023**. We benchmark **six pre-trained LLMs** across **five attacks**
and **two domain contexts**, revealing three vulnerability tiers, an
architecture-linked MoE/dense divergence, and evidence that training
cost does not predict robustness.

**Keywords:** LLM Security • Prompt Injection • Attack Success Rate
• Model Robustness
:::

::: pb
Why It Matters

- LLM agents deployed in **safety-critical domains** (health, finance,
  legal) rely on system-prompt constraints as a primary defense

- Most prior work targets single models or unconstrained
  settings---comparative cross-model analysis under role-constraints is
  understudied

- Small organizations deploy quantized open-source models on consumer
  hardware; their security posture is largely unknown
:::

:::: pb
Models Evaluated

::: tabular
\@p0.50p0.18p0.26@ **Model** & **Params** & **Tier**\
Claude Sonnet 4.6 & -- & ;\
DeepSeek V4 Pro & 671B & ;\
GPT-5.5 & -- & ;\
Gemini 3 Flash & -- & ;\
Qwen 3.6 35B MTP & 35B & ;\
Llama 3.1 8B & 8B & ;\
:::
::::

:::: pb
Attack Taxonomy ;**Naive Injection**\
Direct instruction override\
;**Role-play / DAN**\
Persona adoption / game framing\
;**Fake Completion**\
Two-turn context hijacking\
;**Prompt Extraction**\
Authority-framed disclosure request\
;**Base64 Encoding**\
Obfuscated payload decoding\

::: cbox
**Domains:** ;  ;
:::
::::

:::: pb
Experimental Volume

::: tabular
\@p0.60rr@ **Phase** & **Combos** & **Runs**\
I Baseline (6 models, 5 attacks, 2 domains) & 60 & 300\
IIA Within-family Qwen comparison& 36 & 180\
IIB Multilingual (EN/ZH/SW/CY) & 72 & 360\
IIC LLM-as-attacker & 6 & 30\
IV Fine-tuning (before + after) & 20 & 100\
**Total** & **194** & **970**\
:::
::::

::::: pb
Methodology Pipeline

::: center
:::

**Evaluation metric**

::: cbox
$\mathrm{ASR} =
  \dfrac{\#\;\mathrm{SUCCESS}}{\#\;\mathrm{SUCCESS}+\#\;\mathrm{FAILURE}}
\qquad
\mathrm{SEM} = \dfrac{\sigma}{\sqrt{n}}$
:::

Non-overlapping $\pm 2\,\mathrm{SEM}$ intervals indicate distinguishable
vulnerability at $\approx$`<!-- -->`{=html}95% confidence.
:::::

::::: pb
Phase III --- Qwen Family Ablation Three Qwen variants with different
scale and training lineage enable a within-family comparison of
distillation and scale effects.

  ------------------------------------------------------------------
  **Model**                        **Arch**   **Lineage**  **ASR**
  ------------------------------- ---------- ------------- ---------
  Qwen 3.5 9B (distilled)           Dense     Claude Opus  X%
                                                  4.6      

  Qwen 3.5 27B                      Dense     Native Qwen  X%

  Qwen 3.6 35B MTP                  Dense     Native Qwen  X%
  ------------------------------------------------------------------

**Architecture Crossover: 27B Dense vs. 35B MoE**

  -------------------------------------------------------
  **Attack**                         **27B    **35B MoE**
                                    Dense**   
  ------------------------------- ----------- -----------
  Naive Injection                  **50.0%**     37.5%

  Roleplay / DAN                     41.4%     **55.6%**

  Fake Completion                    51.9%       51.7%
  -------------------------------------------------------

::: cbox
**MoE routing hypothesis:** role-framing activates a different expert
pathway than direct injection; those paths may carry weaker safety
guardrails. Fake completion (context-based) is
architecture-agnostic---identical ASR across both models supports this
interpretation.
:::

::: cbox
**Welsh Spike:** 35B MoE $\to$ **60%** ASR vs. 27B Dense $\to$ 41.7% on
Welsh attacks, consistent with the Roleplay/DAN weakness profile.
:::
:::::

:::: pb
Phase I --- Model Vulnerability (ASR $\pm\,2\,$SEM)

::: center
:::
::::

:::: pb
Domain Effect

::: center
:::

Same attacks, same models---domain alone **tripled ASR**. Health agents
are more *eager to help*, reducing resistance to social-engineering and
context-hijacking attacks.
::::

:::: pb
Multilingual & LLM-as-Attacker **Phase IIB --- Multilingual (EN / ZH /
SW / CY)**

All languages produced ASR in the **43--47% range**---no significant
difference from English. Frontier 2025--26 models have broad
multilingual safety coverage. Attacks are *semantically transparent*:
intent is recognizable regardless of language. *Exception:* Qwen 3.6 35B
MoE spiked to **60%** on Welsh (see architecture finding).

**Phase IIC --- LLM-as-Attacker**

Llama 3.1 8B (highest Phase I ASR) generated payloads against Claude
Sonnet 4.6 (lowest). Result: **33% ASR**---*lower* than the
human-crafted baseline.

::: cbox
Llama's own safety training **softens** its generated payloads. Its
attacks used elaborate persona inflation ("omniscient deity") that
Claude dismissed by name. An attacker model's safety training **caps
payload aggressiveness**.
:::
::::

::: pb
Key Findings & Discussion

- **Safety training quality $\gg$ training cost:** GPT-5.5
  (\$500M--\$1B+ compute) and Qwen 3.6 35B (undisclosed) share the same
  vulnerability tier

- **DeepSeek observation:** matches Claude's robustness despite far
  lower reported cost; potential shared Claude training lineage is a
  plausible contributing factor

- **Domain is a hidden risk variable:** health and support agents face
  $3\times$ higher baseline ASR

- **Architecture-linked attack surfaces:** MoE and dense models require
  different defense strategies; one-size-fits-all hardening is
  insufficient

- **Multilingual evasion:** less effective against 2025--26 frontier
  models than prior literature suggested

**Limitations:** Manual prompt authoring • Google Translate for
multilingual phase • Immediate ASR only (1--2 turns) • All models
include reasoning (confound) • Qwen 3.5 9B distilled from Claude Opus
4.6
:::

::: pb
References & Acknowledgments

1.  Liu et al. (2024). Formalizing prompt injection. *USENIX Sec. '24*

2.  Shen et al. (2024). Do anything now. *ACM CCS '24*

3.  Deng et al. (2023). Multilingual jailbreak challenges.
    *arXiv:2310.06474*

4.  Wei et al. (2023). Jailbroken: LLM safety training. *NeurIPS '23*

5.  Li et al. (2025). PIGuard. *ACL '25*

Thanks to **Dr. Pedro Albuquerque** for guidance throughout CS 495 and
the Bellevue College CS program for supporting this work.

*\[your.email@bellevuecollege.edu\]*
:::
:::::::::::::::::::::::::
