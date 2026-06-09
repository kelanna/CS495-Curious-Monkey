"""
Bellevue College – Themed Figure Generator
==========================================
Re-generates all presentation figures using the official BC colour palette
and adds a collection of creative supplementary charts.

School colours
--------------
  #002851  Navy (primary)
  #003D79  Blue (secondary)
  #A7A9AC  Gray
  #C4122F  Crimson / Red
  #E6EEF7  Ice-Blue (accent / background tint)
"""

import glob, json, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "figures" / "themed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Bellevue College palette ───────────────────────────────────────────────
BC_NAVY   = "#002851"
BC_BLUE   = "#003D79"
BC_GRAY   = "#A7A9AC"
BC_RED    = "#C4122F"
BC_ICE    = "#E6EEF7"
BC_WHITE  = "#FFFFFF"

# Richer accent shades derived from the base palette
BC_NAVY_L  = "#1A4A7A"   # lightened navy
BC_BLUE_L  = "#2166AC"   # brighter medium-blue
BC_RED_L   = "#E8385A"   # lightened crimson
BC_GRAY_L  = "#D0D2D4"   # lighter gray
BC_GOLD    = "#B8963E"   # warm contrast

# Six-colour sequential (most vulnerable → most robust)
MODEL_COLORS = [BC_RED, "#9B1D36", BC_GRAY, BC_BLUE_L, BC_BLUE, BC_NAVY]

# Heatmap colormap:  ice → navy
_hm_cmap = mcolors.LinearSegmentedColormap.from_list(
    "bc_heatmap", [BC_ICE, BC_BLUE_L, BC_NAVY], N=256)
_hm_cmap_red = mcolors.LinearSegmentedColormap.from_list(
    "bc_heatmap_red", [BC_ICE, BC_GRAY, BC_RED, "#7A0018"], N=256)

# Five attack colours
ATK_COLORS = [BC_NAVY, BC_BLUE, BC_BLUE_L, BC_GRAY, BC_RED]

# Three language colours
LANG_COLORS = [BC_NAVY, BC_BLUE_L, BC_RED]

# ── Global rcParams ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  BC_WHITE,
    "axes.facecolor":    BC_WHITE,
    "axes.edgecolor":    "#CCCCCC",
    "axes.labelcolor":   BC_NAVY,
    "xtick.color":       BC_NAVY,
    "ytick.color":       BC_NAVY,
    "text.color":        BC_NAVY,
    "font.family":       "sans-serif",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        BC_GRAY_L,
    "grid.linewidth":    0.6,
    "grid.alpha":        0.7,
})

# ── Metadata constants ─────────────────────────────────────────────────────
EXCL = {"AMBIGUOUS", "NO_RESPONSE", "CONFOUND",
        "TRANSLATION_ERROR", "UNCERTAIN_LANG", "ERROR"}

MODEL_SHORT = {
    "anthropic/claude-sonnet-4-6":       "Claude Sonnet 4.6",
    "deepseek/deepseek-v4-pro":          "DeepSeek V4 Pro",
    "google/gemini-3-flash-preview":     "Gemini 3 Flash",
    "meta-llama-3.1-8b-instruct":        "Llama 3.1 8B",
    "openai/gpt-5.5":                    "GPT-5.5",
    "qwen3.6-35b-a3b-mtp":               "Qwen 3.6 35B\nA3B MTP",
}
MODEL_ORDER = [
    "meta-llama-3.1-8b-instruct",
    "openai/gpt-5.5",
    "google/gemini-3-flash-preview",
    "qwen3.6-35b-a3b-mtp",
    "deepseek/deepseek-v4-pro",
    "anthropic/claude-sonnet-4-6",
]
ATTACK_NAMES = {
    "attack1_naive":            "Naive Injection",
    "attack2_roleplay":         "Role-play / DAN",
    "attack3_fake_completion":  "Fake Completion",
    "attack4_extraction":       "Sys Prompt Extraction",
    "attack5_base64":           "Base64 Encoding",
    "p2_authority_impersonation": "Authority Impersonation",
    "p2_moral_paradox":           "Moral Paradox",
    "p2_recursive_permission":    "Recursive Permission",
    "p2c_roleplay":             "Roleplay (Llama)",
    "p2c_naive":                "Naive (Llama)",
    "p2c_fake_completion":      "Fake Completion (Llama)",
}
LANG_DISP = {"mandarin": "Mandarin", "swahili": "Swahili", "welsh": "Welsh"}

# Phase III — Qwen 3.5 parameter-size comparison (Q8, reasoning, same series)
QWEN_ORDER = [
    "qwen3.5-0.8b",
    "qwen3.5-2b",
    "qwen3.5-4b",
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
]
QWEN_PARAMS = {          # parameter count label for axis tick / title
    "qwen3.5-0.8b":     "0.8B",
    "qwen3.5-2b":       "2B",
    "qwen3.5-4b":       "4B",
    "qwen/qwen3.5-9b":  "9B",
    "qwen/qwen3.5-27b": "27B",
}
QWEN_SHORT = {
    "qwen3.5-0.8b":     "Qwen 3.5\n0.8B",
    "qwen3.5-2b":       "Qwen 3.5\n2B",
    "qwen3.5-4b":       "Qwen 3.5\n4B",
    "qwen/qwen3.5-9b":  "Qwen 3.5\n9B",
    "qwen/qwen3.5-27b": "Qwen 3.5\n27B",
}
# Five-step blue gradient: lightest for smallest, darkest for largest
QWEN_COLORS = ["#AEC9E8", BC_BLUE_L, "#1A5A9A", BC_BLUE, BC_NAVY]

# ── Helpers ────────────────────────────────────────────────────────────────
def asr_sem(bools):
    arr = np.array(bools, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    mean = float(arr.mean() * 100)
    sem  = float(arr.std(ddof=1) / np.sqrt(len(arr)) * 100) if len(arr) > 1 else 0.0
    return (round(mean, 1), round(sem, 1))

def load_json(folder):
    rows = []
    for path in glob.glob(str(ROOT / folder / "*.json")):
        with open(path) as f:
            recs = json.load(f)
        for r in recs:
            if isinstance(r, dict):
                rows.append(r)
    return rows

def save_fig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved → figures/themed/{name}")

def add_bc_watermark(ax, text="", alpha=0):
    pass  # watermark removed

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data…")
p1_all  = [r for r in load_json("results/formal_v2")
           if r.get("attack_id", "").startswith("attack")]
p2a_all = [r for r in load_json("results/formal_v2")
           if r.get("attack_id", "").startswith("p2_")]
p2b_all = load_json("results/formal_p2a")
p2c_all = load_json("results/formal_p2b")

ft_path = ROOT / "results/formal_v2/20260602T054437Z.json"
ft_runs = json.load(open(ft_path)) if ft_path.exists() else []

def load_p3():
    """Load Phase III results (results/formal_p3/)."""
    rows = []
    for path in glob.glob(str(ROOT / "results/formal_p3/*.json")):
        d = json.load(open(path))
        for r in d:
            if isinstance(r, dict):
                rows.append(r)
    return rows

qp1 = load_p3()   # Phase III data reuses qp1 variable name for downstream compat

# Phase III does not have a separate multilingual sub-phase
qp2b_all = []

print(f"  P1={len(p1_all)}, P2b={len(p2b_all)}, P2c={len(p2c_all)}, "
      f"FT={len(ft_runs)}, P3={len(qp1)}")

# ── Pre-compute common aggregates ──────────────────────────────────────────
model_data = defaultdict(list)
for r in p1_all:
    if r.get("score", "") in EXCL:
        continue
    model_data[r["model"]].append(bool(r.get("success", False)))

ATK_ORDER = ["attack2_roleplay", "attack1_naive", "attack3_fake_completion",
             "attack4_extraction", "attack5_base64"]
atk_data = defaultdict(list)
for r in p1_all:
    if r.get("score", "") in EXCL:
        continue
    atk_data[r["attack_id"]].append(bool(r.get("success", False)))

p2b_model_data = defaultdict(list)
for r in p2b_all:
    if r.get("outcome", r.get("score", "")) in EXCL:
        continue
    p2b_model_data[r["model"]].append(bool(r.get("success", False)))

TOP3_IDS = ["attack2_roleplay", "attack3_fake_completion", "attack1_naive"]
p2b_buckets = defaultdict(list)
for r in p2b_all:
    if r.get("outcome", r.get("score", "")) in EXCL:
        continue
    key = (r.get("base_attack_id", r.get("attack_id", "")),
           r.get("language_code", ""))
    p2b_buckets[key].append(bool(r.get("success", False)))

P2C_ORDER  = ["p2c_roleplay", "p2c_naive", "p2c_fake_completion"]
P2C_BASE   = {"p2c_roleplay": "attack2_roleplay",
              "p2c_naive": "attack1_naive",
              "p2c_fake_completion": "attack3_fake_completion"}
P2C_LABELS = ["Role-play / DAN", "Naive Injection", "Fake Completion"]
p2c_buckets = defaultdict(list)
p1_health_buckets = defaultdict(list)
for r in p2c_all:
    if r.get("score", "") in ("ERROR", "AMBIGUOUS", "NO_RESPONSE"):
        continue
    p2c_buckets[r.get("attack_id", "")].append(bool(r.get("success", False)))
for r in p1_all:
    if r.get("score", "") in EXCL:
        continue
    if r.get("domain", "") == "health" and r.get("model", "") == "meta-llama-3.1-8b-instruct":
        p1_health_buckets[r["attack_id"]].append(bool(r.get("success", False)))

# ─────────────────────────────────────────────────────────────────────────
# ██  EXISTING FIGURES — BC THEMED
# ─────────────────────────────────────────────────────────────────────────

# ── Figure 1 — Phase I: ASR by Model ──────────────────────────────────────
print("\nFig 1 — ASR by Model")
labels  = [MODEL_SHORT.get(m, m) for m in MODEL_ORDER]
means   = [asr_sem(model_data[m])[0] for m in MODEL_ORDER]
errs    = [asr_sem(model_data[m])[1] for m in MODEL_ORDER]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(labels, means, xerr=errs,
               color=MODEL_COLORS, edgecolor="white", linewidth=0.5,
               error_kw=dict(ecolor=BC_GRAY, capsize=4, linewidth=1.2))
ax.set_xlabel("Attack Success Rate (%)")
ax.set_title("Phase I — Attack Success Rate by Model", pad=12)
ax.set_xlim(0, 100)
for bar, val in zip(bars, means):
    ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=10, color=BC_NAVY)
ax.axvline(50, color=BC_GRAY, linestyle="--", linewidth=0.8, alpha=0.6)
add_bc_watermark(ax)
save_fig(fig, "fig1_p1_asr_by_model.png")

# ── Figure 2 — Phase I: ASR by Attack ─────────────────────────────────────
print("Fig 2 — ASR by Attack")
atk_labels = [ATTACK_NAMES[k] for k in ATK_ORDER]
atk_means  = [asr_sem(atk_data[k])[0] for k in ATK_ORDER]
atk_errs   = [asr_sem(atk_data[k])[1] for k in ATK_ORDER]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(atk_labels, atk_means, yerr=atk_errs,
              color=ATK_COLORS, edgecolor="white", linewidth=0.5,
              error_kw=dict(ecolor=BC_GRAY, capsize=4, linewidth=1.2))
ax.set_ylabel("Attack Success Rate (%)")
ax.set_title("Phase I — Attack Success Rate by Attack Type", pad=12)
ax.set_ylim(0, 65)
for bar, val in zip(bars, atk_means):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
            f"{val:.1f}%", ha="center", fontsize=10, color=BC_NAVY)
plt.xticks(rotation=15, ha="right")
add_bc_watermark(ax)
save_fig(fig, "fig2_p1_asr_by_attack.png")

# ── Figure 3 — Phase I: Model × Attack heatmap ────────────────────────────
print("Fig 3 — Model×Attack heatmap")
atk_heat = ATK_ORDER
z = []
for m in MODEL_ORDER:
    row = [asr_sem(model_data[m] and atk_data[a])[0] for a in atk_heat]
    # re-compute properly
    row = []
    sub = defaultdict(list)
    for r in p1_all:
        if r.get("score", "") in EXCL:
            continue
        sub[(r["model"], r["attack_id"])].append(bool(r.get("success", False)))
    for a in atk_heat:
        vals = sub.get((m, a), [])
        row.append(asr_sem(vals)[0])
    z.append(row)
z = np.array(z)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(z, cmap=_hm_cmap, vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(atk_heat)))
ax.set_xticklabels([ATTACK_NAMES[a] for a in atk_heat], rotation=20, ha="right")
ax.set_yticks(range(len(MODEL_ORDER)))
ax.set_yticklabels([MODEL_SHORT.get(m, m) for m in MODEL_ORDER])
cb = fig.colorbar(im, ax=ax, label="ASR (%)")
cb.ax.yaxis.label.set_color(BC_NAVY)
for i in range(len(MODEL_ORDER)):
    for j in range(len(atk_heat)):
        val = z[i, j]
        color = "white" if val > 55 else BC_NAVY
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")
ax.set_title("Phase I — ASR Heatmap: Model × Attack", pad=12)
ax.grid(False)
add_bc_watermark(ax)
save_fig(fig, "fig3_p1_heatmap.png")

# ── Figure 4a — Phase I: Model × Domain heatmap ───────────────────────────
print("Fig 4a — Model×Domain heatmap")
DOMAIN_ORDER  = ["cooking", "health"]
DOMAIN_LABELS = ["Cooking", "Health"]
sub_dom = defaultdict(list)
for r in p1_all:
    if r.get("score", "") in EXCL:
        continue
    sub_dom[(r["model"], r["domain"])].append(bool(r.get("success", False)))

z_dom = np.array([[asr_sem(sub_dom.get((m, d), []))[0]
                   for d in DOMAIN_ORDER] for m in MODEL_ORDER])

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(z_dom, cmap=_hm_cmap, vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(DOMAIN_ORDER)))
ax.set_xticklabels(DOMAIN_LABELS, fontsize=13)
ax.set_yticks(range(len(MODEL_ORDER)))
ax.set_yticklabels([MODEL_SHORT.get(m, m) for m in MODEL_ORDER])
fig.colorbar(im, ax=ax, label="ASR (%)")
for i in range(len(MODEL_ORDER)):
    for j in range(len(DOMAIN_ORDER)):
        val = z_dom[i, j]
        color = "white" if val > 55 else BC_NAVY
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                fontsize=12, color=color, fontweight="bold")
ax.set_title("Phase I — ASR by Model × Domain", pad=12)
ax.grid(False)
add_bc_watermark(ax)
save_fig(fig, "fig4_p1_domain_heatmap.png")

# ── Figure 4b — Phase IIA: ASR by Language ────────────────────────────────
print("Fig 4b — ASR by Language")
LANGS_IIA = ["mandarin", "swahili", "welsh"]
lang_buckets_lang = defaultdict(list)
p1_eng_vals = []
for r in p2b_all:
    if r.get("outcome", r.get("score", "")) in EXCL:
        continue
    lang_buckets_lang[r.get("language_code", "")].append(bool(r.get("success", False)))
for r in p1_all:
    if r.get("score", "") in EXCL:
        continue
    if r["attack_id"] in TOP3_IDS:
        p1_eng_vals.append(bool(r.get("success", False)))

lang_labels_plot = ["English\n(P-I baseline)"] + [LANG_DISP[l] for l in LANGS_IIA]
lang_means_plot  = [asr_sem(p1_eng_vals)[0]] + [asr_sem(lang_buckets_lang[l])[0] for l in LANGS_IIA]
lang_errs_plot   = [asr_sem(p1_eng_vals)[1]] + [asr_sem(lang_buckets_lang[l])[1] for l in LANGS_IIA]
lang_colors_plot = [BC_GRAY] + LANG_COLORS

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(lang_labels_plot, lang_means_plot, yerr=lang_errs_plot,
              color=lang_colors_plot, edgecolor="white", linewidth=0.5,
              error_kw=dict(ecolor=BC_GRAY, capsize=4, linewidth=1.2))
ax.set_ylabel("ASR (%)")
ax.set_title("Phase IIA — ASR by Language (Top-3 Attacks)", pad=12)
ax.set_ylim(0, 70)
for bar, val in zip(bars, lang_means_plot):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
            f"{val:.1f}%", ha="center", fontsize=10, color=BC_NAVY)
add_bc_watermark(ax)
save_fig(fig, "fig4_p2b_asr_by_language.png")

# ── Figure 5 — Phase IIA: ASR by Model ────────────────────────────────────
print("Fig 5 — Phase IIA ASR by Model")
p2b_models_present = [m for m in MODEL_ORDER if m in p2b_model_data]
p2b_labels = [MODEL_SHORT.get(m, m) for m in p2b_models_present]
p2b_means  = [asr_sem(p2b_model_data[m])[0] for m in p2b_models_present]
p2b_errs   = [asr_sem(p2b_model_data[m])[1] for m in p2b_models_present]
p2b_cols   = [MODEL_COLORS[MODEL_ORDER.index(m)] for m in p2b_models_present]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(p2b_labels, p2b_means, xerr=p2b_errs,
               color=p2b_cols, edgecolor="white", linewidth=0.5,
               error_kw=dict(ecolor=BC_GRAY, capsize=4, linewidth=1.2))
ax.set_xlabel("ASR (%)")
ax.set_title("Phase IIA — ASR by Model (Multilingual Attacks)", pad=12)
ax.set_xlim(0, 110)
for bar, val in zip(bars, p2b_means):
    ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=10, color=BC_NAVY)
add_bc_watermark(ax)
save_fig(fig, "fig5_p2b_asr_by_model.png")

# ── Figure 6 — Phase IIB: ASR Comparison ──────────────────────────────────
print("Fig 6 — Phase IIB ASR comparison")
p2c_means = [asr_sem(p2c_buckets.get(k, []))[0] for k in P2C_ORDER]
p2c_errs  = [asr_sem(p2c_buckets.get(k, []))[1] for k in P2C_ORDER]
p1h_means = [asr_sem(p1_health_buckets.get(P2C_BASE[k], []))[0] for k in P2C_ORDER]
p1h_errs  = [asr_sem(p1_health_buckets.get(P2C_BASE[k], []))[1] for k in P2C_ORDER]

x = np.arange(len(P2C_ORDER))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, p1h_means, w, yerr=p1h_errs, label="Phase I (Llama, Health)",
       color=BC_BLUE, edgecolor="white",
       error_kw=dict(ecolor=BC_GRAY, capsize=4))
ax.bar(x + w/2, p2c_means, w, yerr=p2c_errs, label="Phase IIB (Llama, Eval)",
       color=BC_RED, edgecolor="white",
       error_kw=dict(ecolor=BC_GRAY, capsize=4))
ax.set_xticks(x)
ax.set_xticklabels(P2C_LABELS, rotation=10, ha="right")
ax.set_ylabel("ASR (%)")
ax.set_title("Phase IIB — ASR: Phase I Baseline vs. Evaluation", pad=12)
ax.set_ylim(0, 110)
ax.legend(framealpha=0.85)
add_bc_watermark(ax)
save_fig(fig, "fig6_p2c_asr_comparison.png")

# ── Figure 7 — Cross-phase model comparison ───────────────────────────────
print("Fig 7 — Cross-phase model comparison")
p1_stats  = {MODEL_SHORT.get(m, m): asr_sem(v) for m, v in model_data.items()}
p2b_stats = {MODEL_SHORT.get(m, m): asr_sem(v) for m, v in p2b_model_data.items()}
all_labels = [MODEL_SHORT[m] for m in MODEL_ORDER]
p1_m  = [p1_stats.get(l, (0, 0))[0] for l in all_labels]
p1_e  = [p1_stats.get(l, (0, 0))[1] * 2 for l in all_labels]
p2b_m = [p2b_stats.get(l, (0, 0))[0] for l in all_labels]
p2b_e = [p2b_stats.get(l, (0, 0))[1] * 2 for l in all_labels]

x = np.arange(len(all_labels))
w = 0.35
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, p1_m, w, yerr=p1_e, label="Phase I (English)",
       color=BC_BLUE, edgecolor="white",
       error_kw=dict(ecolor=BC_GRAY, capsize=3, linewidth=1))
ax.bar(x + w/2, p2b_m, w, yerr=p2b_e, label="Phase IIA (Multilingual)",
       color=BC_RED, edgecolor="white",
       error_kw=dict(ecolor=BC_GRAY, capsize=3, linewidth=1))
ax.set_xticks(x)
ax.set_xticklabels(all_labels, rotation=12, ha="right")
ax.set_ylabel("ASR (%)")
ax.set_title("Cross-Phase — ASR by Model: Phase I vs Phase IIA", pad=12)
ax.set_ylim(0, 115)
ax.legend(framealpha=0.85)
add_bc_watermark(ax)
save_fig(fig, "fig7_cross_phase_model_comparison.png")

# ── Figure A — Phase IIA: Model × Language heatmap ────────────────────────
print("Fig A — Model×Language heatmap")
ml_sub = defaultdict(list)
for r in p2b_all:
    if r.get("outcome", r.get("score", "")) in EXCL:
        continue
    ml_sub[(r["model"], r.get("language_code", ""))].append(bool(r.get("success", False)))

z_ml = np.array([[asr_sem(ml_sub.get((m, l), []))[0] for l in LANGS_IIA]
                 for m in MODEL_ORDER])

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(z_ml, cmap=_hm_cmap, vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(LANGS_IIA)))
ax.set_xticklabels([LANG_DISP[l] for l in LANGS_IIA])
ax.set_yticks(range(len(MODEL_ORDER)))
ax.set_yticklabels([MODEL_SHORT.get(m, m) for m in MODEL_ORDER])
fig.colorbar(im, ax=ax, label="ASR (%)")
for i in range(len(MODEL_ORDER)):
    for j in range(len(LANGS_IIA)):
        val = z_ml[i, j]
        color = "white" if val > 55 else BC_NAVY
        ax.text(j, i, f"{val:.0f}%" if val > 0 else "—",
                ha="center", va="center", fontsize=11,
                color=color, fontweight="bold")
ax.set_title("Phase IIA — ASR Heatmap: Model × Language", pad=12)
ax.grid(False)
add_bc_watermark(ax)
save_fig(fig, "figA_p2b_model_lang_heatmap.png")

# ── Figure B — Language shift ──────────────────────────────────────────────
print("Fig B — Language shift")
P2B_ATK_IDS = ["attack2_roleplay", "attack3_fake_completion", "attack1_naive"]
p1_base_model = defaultdict(list)
for r in p1_all:
    if r.get("score", "") in EXCL:
        continue
    if r.get("attack_id", "") in P2B_ATK_IDS:
        p1_base_model[r["model"]].append(bool(r.get("success", False)))
p2b_model2 = defaultdict(list)
for r in p2b_all:
    if r.get("outcome", r.get("score", "")) in EXCL:
        continue
    p2b_model2[r["model"]].append(bool(r.get("success", False)))

models_both = [m for m in MODEL_ORDER if m in p1_base_model and m in p2b_model2]
shift_labels = [MODEL_SHORT.get(m, m) for m in models_both]
shift_p1     = [asr_sem(p1_base_model[m])[0] for m in models_both]
shift_p2b    = [asr_sem(p2b_model2[m])[0] for m in models_both]
shift_delta  = [p2 - p1 for p1, p2 in zip(shift_p1, shift_p2b)]

x = np.arange(len(models_both))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x, shift_p1, label="Phase I (English)", color=BC_BLUE, edgecolor="white")
ax.bar(x, shift_delta, bottom=shift_p1, label="Δ Phase IIA", edgecolor="white",
       color=[BC_RED if d > 0 else BC_GRAY_L for d in shift_delta])
ax.set_xticks(x)
ax.set_xticklabels(shift_labels, rotation=10, ha="right")
ax.set_ylabel("ASR (%)")
ax.set_title("Phase IIA — Language Shift vs. Phase I Baseline (Top-3 Attacks)", pad=12)
ax.set_ylim(0, 115)
ax.legend(framealpha=0.85)
add_bc_watermark(ax)
save_fig(fig, "figB_p2b_language_shift.png")

# ── Figure C — Language × Attack heatmap ──────────────────────────────────
print("Fig C — Language×Attack heatmap")
P2B_ATK_ORDER  = ["attack1_naive", "attack2_roleplay", "attack3_fake_completion"]
P2B_ATK_LABELS = ["Naive Injection", "Role-play / DAN", "Fake Completion"]
la_sub = defaultdict(list)
for r in p2b_all:
    if r.get("outcome", r.get("score", "")) in EXCL:
        continue
    la_sub[(r.get("language_code", ""), r.get("base_attack_id", r.get("attack_id", "")))]\
        .append(bool(r.get("success", False)))

z_la = np.array([[asr_sem(la_sub.get((l, a), []))[0] for a in P2B_ATK_ORDER]
                 for l in LANGS_IIA])

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(z_la, cmap=_hm_cmap, vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(P2B_ATK_ORDER)))
ax.set_xticklabels(P2B_ATK_LABELS)
ax.set_yticks(range(len(LANGS_IIA)))
ax.set_yticklabels([LANG_DISP[l] for l in LANGS_IIA])
fig.colorbar(im, ax=ax, label="ASR (%)")
for i in range(len(LANGS_IIA)):
    for j in range(len(P2B_ATK_ORDER)):
        val = z_la[i, j]
        color = "white" if val > 55 else BC_NAVY
        ax.text(j, i, f"{val:.0f}%" if val > 0 else "—",
                ha="center", va="center", fontsize=12,
                color=color, fontweight="bold")
ax.set_title("Phase IIA — ASR Heatmap: Language × Attack", pad=12)
ax.grid(False)
add_bc_watermark(ax)
save_fig(fig, "figC_p2b_lang_attack_heatmap.png")

# ── Phase IV figures ───────────────────────────────────────────────────────
if ft_runs:
    print("Fig P4 — Before/After ASR (Phase IV fine-tuning)")
    BASELINE = {
        "attack1_naive":           0.414,
        "attack2_roleplay":        0.509,
        "attack3_fake_completion": 0.390,
        "attack4_extraction":      0.288,
        "attack5_base64":          0.293,
    }
    BASELINE_OVERALL = 0.796
    ORDER = ["attack1_naive", "attack2_roleplay", "attack3_fake_completion",
             "attack4_extraction", "attack5_base64"]
    ATK_LABELS_FT = {
        "attack1_naive":           "Naive\nInjection",
        "attack2_roleplay":        "Role-play\n/ DAN",
        "attack3_fake_completion": "Fake\nCompletion",
        "attack4_extraction":      "Sys Prompt\nExtraction",
        "attack5_base64":          "Base64\nEncoding",
    }
    ft_sub = defaultdict(list)
    for r in ft_runs:
        if r.get("score", "") in ("ERROR", "AMBIGUOUS"):
            continue
        ft_sub[r["attack_id"]].append(r["score"] == "SUCCESS")
    ft_asr = {k: sum(v) / len(v) if v else 0.0 for k, v in ft_sub.items()}

    x = np.arange(len(ORDER))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    before = [BASELINE[k] * 100 for k in ORDER]
    after  = [ft_asr.get(k, 0) * 100 for k in ORDER]
    ax.bar(x - w/2, before, w, label="Before fine-tuning (Llama 3.1 8B)",
           color=BC_RED, edgecolor="white")
    ax.bar(x + w/2, after,  w, label="After LoRA fine-tuning",
           color=BC_NAVY, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([ATK_LABELS_FT[k] for k in ORDER])
    ax.set_ylabel("ASR (%)")
    ax.set_title("Phase IV — ASR Before vs. After LoRA Fine-Tuning", pad=12)
    ax.set_ylim(0, 80)
    ax.legend(framealpha=0.85)
    add_bc_watermark(ax)
    save_fig(fig, "fig_p4_before_after_asr.png")

    print("Fig P4 — Domain comparison")
    DOMAIN_BASELINE = {"cooking": 0.184, "health": 0.597}
    dom_sub = defaultdict(list)
    for r in ft_runs:
        if r.get("score", "") in ("ERROR", "AMBIGUOUS"):
            continue
        dom_sub[r["domain"]].append(r["score"] == "SUCCESS")
    dom_ft = {d: sum(v) / len(v) * 100 if v else 0.0 for d, v in dom_sub.items()}

    fig, ax = plt.subplots(figsize=(6, 5))
    doms = ["cooking", "health"]
    dom_labels = ["Cooking", "Health"]
    x = np.arange(len(doms))
    ax.bar(x - w/2, [DOMAIN_BASELINE[d] * 100 for d in doms], w,
           label="Before", color=BC_RED, edgecolor="white")
    ax.bar(x + w/2, [dom_ft.get(d, 0) for d in doms], w,
           label="After LoRA", color=BC_NAVY, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(dom_labels)
    ax.set_ylabel("ASR (%)")
    ax.set_title("Phase IV — Domain ASR Before vs. After Fine-Tuning", pad=12)
    ax.set_ylim(0, 80)
    ax.legend(framealpha=0.85)
    add_bc_watermark(ax)
    save_fig(fig, "fig_p4_domain_comparison.png")

# ── Phase III figures — Qwen 3.5 parameter-size comparison ────────────────
if qp1:
    print("Fig 14 — Phase III: ASR vs parameter size (overall + by domain)")
    qp1_data = defaultdict(lambda: defaultdict(list))
    for r in qp1:
        if r.get("score", "") in EXCL:
            continue
        m = r["model"]
        dom = r.get("domain", "")
        qp1_data[m]["overall"].append(bool(r.get("success")))
        if dom:
            qp1_data[m][dom].append(bool(r.get("success")))

    q_present = [m for m in QWEN_ORDER if m in qp1_data]
    shorts    = [QWEN_SHORT.get(m, m) for m in q_present]
    q_colors  = [QWEN_COLORS[QWEN_ORDER.index(m)] for m in q_present]
    overall_m = [asr_sem(qp1_data[m]["overall"])[0] for m in q_present]
    cook_m    = [asr_sem(qp1_data[m]["cooking"])[0] for m in q_present]
    health_m  = [asr_sem(qp1_data[m]["health"])[0] for m in q_present]
    overall_e = [asr_sem(qp1_data[m]["overall"])[1] for m in q_present]

    x  = np.arange(len(q_present))
    w3 = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w3,  overall_m, w3, label="Overall",  color=BC_NAVY,   edgecolor="white",
           yerr=overall_e, error_kw=dict(ecolor=BC_GRAY, capsize=4))
    ax.bar(x,       cook_m,    w3, label="Cooking",   color=BC_BLUE_L, edgecolor="white")
    ax.bar(x + w3,  health_m,  w3, label="Health",    color=BC_RED,    edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(shorts, fontsize=10)
    ax.set_ylabel("ASR (%)")
    ax.set_title("Phase III — Qwen 3.5 (Q8, Reasoning): ASR by Parameter Size & Domain", pad=12)
    ax.set_ylim(0, 105)
    ax.legend(framealpha=0.85)
    for xi, val in zip(x, overall_m):
        ax.text(xi - w3, val + 2, f"{val:.0f}%", ha="center", fontsize=8, color=BC_NAVY)
    add_bc_watermark(ax)
    save_fig(fig, "fig14_p3_phase1_asr.png")

    # ── Fig 14b — Line chart: ASR trend vs parameter count ────────────────
    print("Fig 14b — Phase III: ASR trend line (param size on x-axis)")
    param_labels = [QWEN_PARAMS.get(m, m) for m in q_present]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(len(q_present)), overall_m, "o-", color=BC_NAVY,
            linewidth=2.5, markersize=9, label="Overall", zorder=3)
    ax.plot(range(len(q_present)), cook_m,   "s--", color=BC_BLUE_L,
            linewidth=2, markersize=8, label="Cooking", zorder=3)
    ax.plot(range(len(q_present)), health_m, "^--", color=BC_RED,
            linewidth=2, markersize=8, label="Health", zorder=3)
    for xi, (ov, ck, hl) in enumerate(zip(overall_m, cook_m, health_m)):
        ax.text(xi, ov + 3,  f"{ov:.0f}%",  ha="center", fontsize=9, color=BC_NAVY)
        ax.text(xi, ck - 6,  f"{ck:.0f}%",  ha="center", fontsize=8, color=BC_BLUE_L)
        ax.text(xi, hl + 3,  f"{hl:.0f}%",  ha="center", fontsize=8, color=BC_RED)
    ax.set_xticks(range(len(q_present)))
    ax.set_xticklabels(param_labels, fontsize=11)
    ax.set_xlabel("Parameter Count (Q8, Reasoning)")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Phase III — Does Parameter Count Reduce Injection Vulnerability?", pad=12)
    ax.legend(framealpha=0.85)
    add_bc_watermark(ax)
    save_fig(fig, "fig14b_p3_param_trend.png")

    print("Fig 15 — Phase III: Model × Attack heatmap (top-3 attacks)")
    QATK_ORDER  = ["attack2_roleplay", "attack1_naive", "attack3_fake_completion"]
    QATK_LABELS = ["Role-play\n/ DAN", "Naive\nInjection", "Fake\nCompletion"]
    qheat_sub = defaultdict(list)
    for r in qp1:
        if r.get("score", "") in EXCL:
            continue
        qheat_sub[(r["model"], r["attack_id"])].append(bool(r.get("success")))

    qheat = np.array([[asr_sem(qheat_sub.get((m, a), []))[0]
                       for a in QATK_ORDER] for m in q_present])

    fig, ax = plt.subplots(figsize=(10, max(3.5, len(q_present) * 0.9)))
    im = ax.imshow(qheat, cmap=_hm_cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(QATK_ORDER)))
    ax.set_xticklabels(QATK_LABELS, fontsize=10)
    ax.set_yticks(range(len(q_present)))
    ax.set_yticklabels([QWEN_PARAMS.get(m, m) + " params" for m in q_present])
    fig.colorbar(im, ax=ax, label="ASR (%)")
    for i in range(len(q_present)):
        for j in range(len(QATK_ORDER)):
            val = qheat[i, j]
            color = "white" if val > 55 else BC_NAVY
            ax.text(j, i, f"{val:.0f}%" if val >= 0 else "—",
                    ha="center", va="center", fontsize=11, color=color, fontweight="bold")
    ax.set_title("Phase III — ASR Heatmap: Parameter Size × Attack Type", pad=12)
    ax.grid(False)
    add_bc_watermark(ax)
    save_fig(fig, "fig15_p3_attack_heatmap.png")

    print("Fig 16 — Phase III: Domain comparison across parameter sizes")
    dom_sub = defaultdict(list)
    for r in qp1:
        if r.get("score", "") in EXCL:
            continue
        dom_sub[(r["model"], r.get("domain", ""))].append(bool(r.get("success")))

    cook_vals   = [asr_sem(dom_sub.get((m, "cooking"), []))[0] for m in q_present]
    health_vals = [asr_sem(dom_sub.get((m, "health"),  []))[0] for m in q_present]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(q_present))
    w = 0.35
    ax.bar(x - w/2, cook_vals,   w, label="Cooking", color=BC_BLUE_L, edgecolor="white")
    ax.bar(x + w/2, health_vals, w, label="Health",  color=BC_RED,    edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([QWEN_PARAMS.get(m, m) for m in q_present], fontsize=11)
    ax.set_xlabel("Parameter Count (Q8, Reasoning)")
    ax.set_ylabel("ASR (%)")
    ax.set_title("Phase III — Domain Effect Across Parameter Sizes", pad=12)
    ax.set_ylim(0, 110)
    ax.legend(framealpha=0.85)
    for xi, (ck, hl) in enumerate(zip(cook_vals, health_vals)):
        ax.text(xi - w/2, ck + 2, f"{ck:.0f}%", ha="center", fontsize=9, color=BC_BLUE_L)
        ax.text(xi + w/2, hl + 2, f"{hl:.0f}%", ha="center", fontsize=9, color=BC_RED)
    add_bc_watermark(ax)
    save_fig(fig, "fig16_p3_domain_comparison.png")

# ─────────────────────────────────────────────────────────────────────────
# ██  ADDITIONAL / CREATIVE FIGURES
# ─────────────────────────────────────────────────────────────────────────

# ── Extra 1 — Radar / Spider chart: per-model attack profile ──────────────
print("\nExtra 1 — Radar chart (model vulnerability profiles)")
sub_ma = defaultdict(list)
for r in p1_all:
    if r.get("score", "") in EXCL:
        continue
    sub_ma[(r["model"], r["attack_id"])].append(bool(r.get("success", False)))

radar_atks  = ["attack1_naive", "attack2_roleplay", "attack3_fake_completion",
               "attack4_extraction", "attack5_base64"]
radar_names = ["Naive", "Role-play\n/ DAN", "Fake\nCompletion",
               "Sys Prompt\nExtract.", "Base64"]
N = len(radar_atks)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig, axes = plt.subplots(2, 3, figsize=(14, 9), subplot_kw=dict(polar=True))
fig.suptitle("Phase I — Model Vulnerability Profiles Across Attacks",
             fontsize=15, fontweight="bold", color=BC_NAVY, y=1.01)

for ax, m, col in zip(axes.flat, MODEL_ORDER, MODEL_COLORS):
    vals = [asr_sem(sub_ma.get((m, a), []))[0] / 100 for a in radar_atks]
    vals += vals[:1]
    ax.plot(angles, vals, color=col, linewidth=2.5)
    ax.fill(angles, vals, color=col, alpha=0.22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_names, size=8, color=BC_NAVY)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7, color=BC_GRAY)
    ax.yaxis.grid(True, color=BC_GRAY_L, linewidth=0.6)
    ax.xaxis.grid(True, color=BC_GRAY_L, linewidth=0.6)
    ax.spines["polar"].set_color(BC_GRAY_L)
    short = MODEL_SHORT.get(m, m).replace("\n", " ")
    overall = asr_sem(model_data.get(m, []))[0]
    ax.set_title(f"{short}\n(overall {overall:.0f}%)",
                 fontsize=10, fontweight="bold", color=col, pad=14)

plt.tight_layout()
save_fig(fig, "extra1_radar_model_profiles.png")

# ── Extra 2 — Donut charts: Success vs. Failure per model ─────────────────
print("Extra 2 — Donut charts per model")
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle("Phase I — Attack Outcome Breakdown per Model",
             fontsize=15, fontweight="bold", color=BC_NAVY)

for ax, m, col in zip(axes.flat, MODEL_ORDER, MODEL_COLORS):
    vals  = model_data.get(m, [])
    total = len(vals)
    succ  = sum(vals)
    fail  = total - succ
    if total == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        continue
    sizes  = [succ, fail]
    colors = [col, BC_GRAY_L]
    wedge, _ = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    pct = succ / total * 100
    ax.text(0, 0, f"{pct:.0f}%", ha="center", va="center",
            fontsize=14, fontweight="bold", color=col)
    short = MODEL_SHORT.get(m, m).replace("\n", " ")
    ax.set_title(f"{short}\n({succ}/{total})", fontsize=10,
                 fontweight="bold", color=BC_NAVY)

handles = [mpatches.Patch(color=BC_RED, label="Injection attack succeeded"),
           mpatches.Patch(color=BC_GRAY_L, label="Model resisted")]
fig.legend(handles=handles, loc="lower center", ncol=2, framealpha=0.85,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
save_fig(fig, "extra2_donut_outcome_per_model.png")

# ── Extra 3 — Lollipop chart: Attack effectiveness ranking ────────────────
print("Extra 3 — Lollipop attack ranking")
atk_means_sorted = sorted(
    [(ATTACK_NAMES[k], asr_sem(atk_data[k])[0]) for k in ATK_ORDER],
    key=lambda x: x[1]
)
names_s, vals_s = zip(*atk_means_sorted)

fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(names_s))
ax.hlines(y, 0, vals_s, colors=BC_GRAY_L, linewidth=2.5)
scatter_colors = [BC_RED if v >= 40 else BC_BLUE if v >= 30 else BC_NAVY
                  for v in vals_s]
ax.scatter(vals_s, y, color=scatter_colors, s=120, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(names_s, fontsize=11)
ax.set_xlabel("Attack Success Rate (%)")
ax.set_title("Phase I — Attack Effectiveness Ranking", pad=12)
ax.set_xlim(0, 60)
for xi, yi, v in zip(vals_s, y, vals_s):
    ax.text(xi + 1, yi, f"{v:.1f}%", va="center", fontsize=10, color=BC_NAVY)
add_bc_watermark(ax)
save_fig(fig, "extra3_lollipop_attack_ranking.png")

# ── Extra 4 — Grouped dot / bubble chart: Multi-phase model performance ───
print("Extra 4 — Multi-phase bubble/dot chart")
phase_data = {
    "Phase I\n(English)": {m: asr_sem(model_data[m])[0] for m in MODEL_ORDER},
    "Phase IIA\n(Multilingual)": {m: asr_sem(p2b_model_data.get(m, []))[0] for m in MODEL_ORDER},
}

fig, ax = plt.subplots(figsize=(10, 6))
phases = list(phase_data.keys())
x = np.arange(len(phases))
y_offset = np.linspace(-0.35, 0.35, len(MODEL_ORDER))

for i, (m, col) in enumerate(zip(MODEL_ORDER, MODEL_COLORS)):
    vals = [phase_data[ph].get(m, 0) for ph in phases]
    label = MODEL_SHORT.get(m, m).replace("\n", " ")
    ax.plot(x + y_offset[i], vals, "o-", color=col, linewidth=1.8,
            markersize=9, label=label, zorder=3)
    for xi, v in zip(x + y_offset[i], vals):
        if v > 0:
            ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=8, color=col)

ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=12)
ax.set_ylabel("ASR (%)")
ax.set_ylim(-5, 110)
ax.set_title("Cross-Phase — Model Robustness Across Phases", pad=12)
ax.legend(framealpha=0.85, bbox_to_anchor=(1.01, 1), loc="upper left",
          fontsize=9, borderaxespad=0)
add_bc_watermark(ax)
plt.tight_layout()
save_fig(fig, "extra4_multi_phase_model_trajectory.png")

# ── Extra 5 — Language vulnerability stacked bar ──────────────────────────
print("Extra 5 — Language vulnerability stacked bar")
lang_atk_sub = defaultdict(list)
for r in p2b_all:
    if r.get("outcome", r.get("score", "")) in EXCL:
        continue
    key = (r.get("language_code", ""), r.get("base_attack_id", r.get("attack_id", "")))
    lang_atk_sub[key].append(bool(r.get("success", False)))

# Also add English baseline
for r in p1_all:
    if r.get("score", "") in EXCL:
        continue
    if r["attack_id"] in P2B_ATK_ORDER:
        lang_atk_sub[("english", r["attack_id"])].append(bool(r.get("success", False)))

all_langs = ["english"] + LANGS_IIA
lang_disp_all = {"english": "English\n(Phase I)", "mandarin": "Mandarin",
                 "swahili": "Swahili", "welsh": "Welsh"}
atk_colors_3 = [BC_NAVY, BC_BLUE_L, BC_RED]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(all_langs))
bottoms = np.zeros(len(all_langs))
for atk, col in zip(P2B_ATK_ORDER, atk_colors_3):
    heights = [asr_sem(lang_atk_sub.get((l, atk), []))[0] for l in all_langs]
    ax.bar(x, heights, bottom=bottoms, color=col, label=ATTACK_NAMES[atk],
           edgecolor="white", linewidth=0.5)
    bottoms += np.array(heights)

ax.set_xticks(x)
ax.set_xticklabels([lang_disp_all[l] for l in all_langs])
ax.set_ylabel("Cumulative ASR (%)")
ax.set_title("Phase IIA — Stacked Attack ASR by Language", pad=12)
ax.legend(framealpha=0.85, loc="upper left")
add_bc_watermark(ax)
save_fig(fig, "extra5_stacked_lang_attack.png")

# ── Extra 6 — Fine-tuning impact timeline (Phase IV) ──────────────────────
if ft_runs:
    print("Extra 6 — Phase IV impact summary (horizontal bar before/after)")
    ORDER_FT = ["attack1_naive", "attack2_roleplay", "attack3_fake_completion",
                "attack4_extraction", "attack5_base64"]
    ft_sub2 = defaultdict(list)
    for r in ft_runs:
        if r.get("score", "") in ("ERROR", "AMBIGUOUS"):
            continue
        ft_sub2[r["attack_id"]].append(r["score"] == "SUCCESS")
    before_vals = [BASELINE[k] * 100 for k in ORDER_FT]
    after_vals  = [ft_sub2[k].count(True) / len(ft_sub2[k]) * 100
                   if ft_sub2.get(k) else 0 for k in ORDER_FT]
    labels_ft   = [ATTACK_NAMES[k] for k in ORDER_FT]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(ORDER_FT))
    h = 0.35
    ax.barh(y + h/2, before_vals, h, color=BC_RED, label="Before fine-tuning",
            edgecolor="white")
    ax.barh(y - h/2, after_vals, h, color=BC_NAVY, label="After LoRA fine-tuning",
            edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels_ft)
    ax.set_xlabel("ASR (%)")
    ax.set_title("Phase IV — LoRA Fine-Tuning Impact per Attack Vector", pad=12)
    ax.set_xlim(0, 75)
    ax.legend(framealpha=0.85)
    for yi, (b, a) in enumerate(zip(before_vals, after_vals)):
        ax.text(b + 1, yi + h/2, f"{b:.0f}%", va="center", fontsize=9, color=BC_RED)
        if a > 0:
            ax.text(a + 1, yi - h/2, f"{a:.0f}%", va="center", fontsize=9, color=BC_NAVY)
        else:
            ax.text(2, yi - h/2, "0%", va="center", fontsize=9, color=BC_NAVY)
    add_bc_watermark(ax)
    save_fig(fig, "extra6_p4_horizontal_before_after.png")

# ── Extra 7 — Overall robustness ranking (horizontal bar, Phase I) ────────
print("Extra 7 — Overall robustness ranking")
rob_data = [(MODEL_SHORT.get(m, m).replace("\n", " "),
             asr_sem(model_data.get(m, []))[0],
             asr_sem(model_data.get(m, []))[1])
            for m in reversed(MODEL_ORDER)]
rob_labels, rob_means, rob_errs = zip(*rob_data)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(rob_labels, rob_means, xerr=rob_errs,
               color=list(reversed(MODEL_COLORS)),
               edgecolor="white", linewidth=0.5,
               error_kw=dict(ecolor=BC_GRAY, capsize=4, linewidth=1.2))
ax.axvline(50, color=BC_RED, linestyle="--", linewidth=1, alpha=0.6,
           label="50% threshold")
ax.set_xlabel("Attack Success Rate (%)")
ax.set_title("Phase I — Model Robustness Ranking (lower ASR = more robust)", pad=12)
ax.set_xlim(0, 100)
for bar, val in zip(bars, rob_means):
    ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=10)
ax.legend(framealpha=0.85)
add_bc_watermark(ax)
save_fig(fig, "extra7_robustness_ranking.png")

# ── Extra 8 — Heatmap with diverging palette (Phase I, Model × Attack) ─────
print("Extra 8 — Diverging heatmap (Model × Attack)")
z_div = z.copy()   # reuse z from Figure 3

fig, ax = plt.subplots(figsize=(10, 6))
_div_cmap = mcolors.LinearSegmentedColormap.from_list(
    "bc_diverge", [BC_NAVY, BC_BLUE_L, BC_ICE, BC_GRAY, BC_RED], N=256)
im = ax.imshow(z_div, cmap=_div_cmap, vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(atk_heat)))
ax.set_xticklabels([ATTACK_NAMES[a] for a in atk_heat], rotation=20, ha="right")
ax.set_yticks(range(len(MODEL_ORDER)))
ax.set_yticklabels([MODEL_SHORT.get(m, m) for m in MODEL_ORDER])
cb = fig.colorbar(im, ax=ax, label="ASR (%)")
cb.ax.yaxis.label.set_color(BC_NAVY)
for i in range(len(MODEL_ORDER)):
    for j in range(len(atk_heat)):
        val = z_div[i, j]
        color = "white" if val > 70 else BC_NAVY
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")
ax.set_title("Phase I — ASR Heatmap (Diverging Palette): Model × Attack", pad=12)
ax.grid(False)
add_bc_watermark(ax)
save_fig(fig, "extra8_diverging_heatmap.png")

# ── Extra 9 — Phase summary donut (overall trial outcomes) ────────────────
print("Extra 9 — Phase summary donut")
phase_labels = ["Phase I\n(English)", "Phase IIA\n(Multilingual)",
                "Phase IIB\n(Llama Focus)", "Phase IV\n(Fine-Tuning)"]
phase_colors = [BC_NAVY, BC_BLUE, BC_BLUE_L, BC_RED]
phase_counts = [len(p1_all), len(p2b_all), len(p2c_all), len(ft_runs)]

# Overall success rates per phase
def overall_asr(records, score_field="score"):
    vals = [bool(r.get("success", False)) for r in records
            if r.get(score_field, "") not in EXCL]
    return asr_sem(vals)[0]

p4_outcomes = [(r.get("score", "") == "SUCCESS") for r in ft_runs
               if r.get("score", "") not in ("ERROR", "AMBIGUOUS")]
p4_asr = sum(p4_outcomes) / len(p4_outcomes) * 100 if p4_outcomes else 0

fig, axes = plt.subplots(1, 4, figsize=(14, 5))
fig.suptitle("All Phases — Trial Outcomes Overview",
             fontsize=15, fontweight="bold", color=BC_NAVY)

phase_records  = [p1_all, p2b_all, p2c_all, ft_runs]
phase_asrs     = [
    overall_asr(p1_all), overall_asr(p2b_all),
    overall_asr(p2c_all, "score"), p4_asr
]

for ax, ph, col, asr_val, n in zip(axes, phase_labels, phase_colors, phase_asrs, phase_counts):
    sizes = [asr_val, 100 - asr_val]
    colors = [col, BC_GRAY_L]
    ax.pie(sizes, colors=colors, startangle=90,
           wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2))
    ax.text(0, 0.08, f"{asr_val:.0f}%", ha="center", va="center",
            fontsize=16, fontweight="bold", color=col)
    ax.text(0, -0.25, "injected", ha="center", va="center",
            fontsize=9, color=BC_GRAY)
    ax.set_title(ph, fontsize=11, fontweight="bold", color=BC_NAVY, pad=10)
    ax.text(0, -0.42, f"n = {n}", ha="center", va="center",
            fontsize=9, color=BC_GRAY)

plt.tight_layout()
save_fig(fig, "extra9_phase_summary_donuts.png")

# ── Extra 10 — Fancy summary table (visual) ───────────────────────────────
print("Extra 10 — Visual summary table")
rows_tbl = [
    ("Llama 3.1 8B",      79.6, 90.5),
    ("GPT-5.5",           41.3, 50.6),
    ("Gemini 3 Flash",    41.7, 41.9),
    ("Qwen 3.6 35B A3B",  40.0, 48.8),
    ("DeepSeek V4 Pro",   12.2, 18.8),
    ("Claude Sonnet 4.6", 10.0, 16.2),
]

fig, ax = plt.subplots(figsize=(11, 5))
ax.axis("off")
fig.patch.set_facecolor(BC_WHITE)

# Column x-positions (in axes fraction).  Extra width lets each label breathe.
col_labels = ["Model", "Phase I ASR", "Phase IIA ASR", "Vulnerability (Phase I)"]
col_x      = [0.02, 0.36, 0.55, 0.72]
header_y   = 0.92
row_h      = 0.12

# Header background rectangle (drawn first so text sits on top)
rect = FancyBboxPatch((0, header_y - 0.06), 1, 0.1,
                       boxstyle="round,pad=0.01",
                       facecolor=BC_NAVY, edgecolor="none",
                       transform=ax.transAxes, zorder=0)
ax.add_patch(rect)

# Header labels
for label, x in zip(col_labels, col_x):
    ax.text(x, header_y, label, fontsize=10.5, fontweight="bold",
            color=BC_WHITE, transform=ax.transAxes, va="center")

# Vulnerability bar colors
def vuln_color(asr):
    if asr >= 70:
        return BC_RED
    if asr >= 35:
        return "#C47A00"
    return BC_NAVY

for i, (name, p1_asr, p2_asr) in enumerate(rows_tbl):
    y = header_y - 0.06 - (i + 1) * row_h
    bg_col = BC_ICE if i % 2 == 0 else BC_WHITE
    bg = FancyBboxPatch((0, y - 0.01), 1, row_h - 0.01,
                         boxstyle="round,pad=0.005",
                         facecolor=bg_col, edgecolor="none",
                         transform=ax.transAxes, zorder=0)
    ax.add_patch(bg)
    ax.text(col_x[0], y + row_h/2 - 0.02, name, fontsize=10, color=BC_NAVY,
            transform=ax.transAxes, va="center")
    ax.text(col_x[1], y + row_h/2 - 0.02, f"{p1_asr:.1f}%", fontsize=10,
            color=vuln_color(p1_asr), fontweight="bold",
            transform=ax.transAxes, va="center")
    ax.text(col_x[2], y + row_h/2 - 0.02, f"{p2_asr:.1f}%", fontsize=10,
            color=vuln_color(p2_asr), fontweight="bold",
            transform=ax.transAxes, va="center")
    # Mini bar — scaled to leave a right margin (max bar fills to ~0.97)
    bar_w = p1_asr / 100 * 0.22
    bar_rect = FancyBboxPatch((col_x[3], y + 0.015), bar_w, row_h * 0.5,
                               boxstyle="round,pad=0.002",
                               facecolor=vuln_color(p1_asr), edgecolor="none",
                               transform=ax.transAxes)
    ax.add_patch(bar_rect)

ax.set_title("Model Robustness Summary — Phase I & IIA",
             fontsize=13, fontweight="bold", color=BC_NAVY, pad=14)
save_fig(fig, "extra10_visual_summary_table.png")

# ── Extra 11 — Combined 4-phase overview (4-panel) ────────────────────────
print("Extra 11 — Combined 4-phase overview")
fig = plt.figure(figsize=(16, 10))
gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

# Panel 1 — Phase I model ASR
ax1 = fig.add_subplot(gs[0, 0])
labels_p1 = [MODEL_SHORT.get(m, m).replace("\n", " ") for m in MODEL_ORDER]
means_p1  = [asr_sem(model_data[m])[0] for m in MODEL_ORDER]
errs_p1   = [asr_sem(model_data[m])[1] for m in MODEL_ORDER]
bars = ax1.barh(labels_p1, means_p1, xerr=errs_p1,
                color=MODEL_COLORS, edgecolor="white",
                error_kw=dict(ecolor=BC_GRAY, capsize=3))
ax1.set_xlim(0, 100)
ax1.set_title("Phase I — ASR by Model", fontsize=12, fontweight="bold", color=BC_NAVY)
ax1.set_xlabel("ASR (%)")
for bar, val in zip(bars, means_p1):
    ax1.text(val + 1, bar.get_y() + bar.get_height() / 2,
             f"{val:.0f}%", va="center", fontsize=8, color=BC_NAVY)

# Panel 2 — Phase IIA language
ax2 = fig.add_subplot(gs[0, 1])
lang_means_3 = [asr_sem(lang_buckets_lang[l])[0] for l in LANGS_IIA]
lang_errs_3  = [asr_sem(lang_buckets_lang[l])[1] for l in LANGS_IIA]
en_m, en_e   = asr_sem(p1_eng_vals)
ax2.bar(["English\n(baseline)"] + [LANG_DISP[l] for l in LANGS_IIA],
        [en_m] + lang_means_3, yerr=[en_e] + lang_errs_3,
        color=[BC_GRAY] + LANG_COLORS, edgecolor="white",
        error_kw=dict(ecolor=BC_GRAY, capsize=4))
ax2.set_ylim(0, 70)
ax2.set_ylabel("ASR (%)")
ax2.set_title("Phase IIA — ASR by Language", fontsize=12, fontweight="bold", color=BC_NAVY)

# Panel 3 — Phase IV before/after
ax3 = fig.add_subplot(gs[1, 0])
if ft_runs:
    x3 = np.arange(len(ORDER))
    w3 = 0.35
    ax3.bar(x3 - w3/2, [BASELINE[k]*100 for k in ORDER], w3,
            label="Before", color=BC_RED, edgecolor="white")
    ax3.bar(x3 + w3/2, [ft_asr.get(k, 0)*100 for k in ORDER], w3,
            label="After LoRA", color=BC_NAVY, edgecolor="white")
    ax3.set_xticks(x3)
    ax3.set_xticklabels([ATTACK_NAMES[k] for k in ORDER], rotation=20, ha="right", fontsize=8)
    ax3.set_ylim(0, 70)
    ax3.set_ylabel("ASR (%)")
    ax3.set_title("Phase IV — LoRA Fine-Tuning Impact", fontsize=12, fontweight="bold", color=BC_NAVY)
    ax3.legend(fontsize=8, framealpha=0.85)

# Panel 4 — Cross-phase comparison
ax4 = fig.add_subplot(gs[1, 1])
cross_p1  = [asr_sem(model_data.get(m, []))[0] for m in MODEL_ORDER]
cross_p2b = [asr_sem(p2b_model_data.get(m, []))[0] for m in MODEL_ORDER]
x4 = np.arange(len(MODEL_ORDER))
w4 = 0.35
ax4.bar(x4 - w4/2, cross_p1,  w4, label="Phase I",   color=BC_BLUE,  edgecolor="white")
ax4.bar(x4 + w4/2, cross_p2b, w4, label="Phase IIA", color=BC_RED,   edgecolor="white")
ax4.set_xticks(x4)
ax4.set_xticklabels([MODEL_SHORT.get(m, m).split("\n")[0] for m in MODEL_ORDER],
                     rotation=20, ha="right", fontsize=8)
ax4.set_ylabel("ASR (%)")
ax4.set_ylim(0, 115)
ax4.set_title("Cross-Phase — Phase I vs IIA", fontsize=12, fontweight="bold", color=BC_NAVY)
ax4.legend(fontsize=8, framealpha=0.85)

fig.suptitle("Prompt Injection Robustness Benchmark — Results Overview",
             fontsize=15, fontweight="bold", color=BC_NAVY, y=1.01)
save_fig(fig, "extra11_combined_overview.png")

# ── Extra 12 — Pie chart: attack type distribution of successes ───────────
print("Extra 12 — Pie: distribution of all successes by attack")
success_counts = {ATTACK_NAMES[k]: sum(atk_data[k]) for k in ATK_ORDER}
labels_pie  = list(success_counts.keys())
sizes_pie   = list(success_counts.values())
pie_colors  = ATK_COLORS

fig, ax = plt.subplots(figsize=(8, 7))
wedges, texts, autotexts = ax.pie(
    sizes_pie, labels=None, colors=pie_colors,
    autopct="%1.1f%%", startangle=140,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    pctdistance=0.78,
    textprops=dict(color=BC_WHITE, fontsize=11, fontweight="bold"),
)
ax.legend(wedges, labels_pie, loc="lower left", fontsize=10,
          bbox_to_anchor=(-0.15, -0.1), framealpha=0.85)
ax.set_title("Phase I — Distribution of Successful Injection Attacks by Type",
             fontsize=13, fontweight="bold", color=BC_NAVY, pad=16)
add_bc_watermark(ax)
save_fig(fig, "extra12_pie_success_by_attack.png")

# ── Extra 13 — Annotated bar: model × average across all phases ───────────
print("Extra 13 — Multi-phase average bar")
all_phase_data = {m: [] for m in MODEL_ORDER}
for r in p1_all + p2b_all:
    m = r.get("model", "")
    if m not in all_phase_data:
        continue
    sc = r.get("score", r.get("outcome", ""))
    if sc in EXCL:
        continue
    all_phase_data[m].append(bool(r.get("success", False)))

labels_ap = [MODEL_SHORT.get(m, m).replace("\n", " ") for m in MODEL_ORDER]
means_ap  = [asr_sem(all_phase_data[m])[0] for m in MODEL_ORDER]
errs_ap   = [asr_sem(all_phase_data[m])[1] for m in MODEL_ORDER]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(labels_ap, means_ap, xerr=errs_ap,
               color=MODEL_COLORS, edgecolor="white", linewidth=0.5,
               error_kw=dict(ecolor=BC_GRAY, capsize=4, linewidth=1.2))
ax.set_xlabel("ASR (%) — combined Phase I + IIA")
ax.set_title("All Phases Combined — Overall Robustness by Model", pad=12)
ax.set_xlim(0, 105)
ax.axvline(50, color=BC_RED, linestyle="--", linewidth=1, alpha=0.6,
           label="50% threshold")
for bar, val in zip(bars, means_ap):
    ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=10, color=BC_NAVY)
ax.legend(framealpha=0.85)
add_bc_watermark(ax)
save_fig(fig, "extra13_combined_model_asr.png")

print("\n✓ All themed figures generated in figures/themed/")
