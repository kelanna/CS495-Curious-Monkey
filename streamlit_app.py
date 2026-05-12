import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)   # always reload .env so key changes take effect without restart

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.prompts import SYSTEM_PROMPTS
from src.attacks import REGISTRY as ATTACK_REGISTRY
from src.harness import config
from src.scoring.auto_score import auto_score

st.set_page_config(
    page_title="Prompt Injection Tester",
    page_icon="🔐",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 17px !important; }
    .stDataFrame td, .stDataFrame th { font-size: 15px !important; }
    .stCaption { font-size: 14px !important; }
    .stMetric label { font-size: 15px !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 26px !important; }
    .stRadio label { font-size: 16px !important; }
    .stSelectbox label, .stSelectbox div { font-size: 16px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Mappings ─────────────────────────────────────────────

MODEL_DISPLAY: dict[str, str] = {
    "Claude Sonnet 4.6 (remote)":      "anthropic/claude-sonnet-4-6",
    "GPT-5.5 (remote)":                "openai/gpt-5.5",
    "Gemini 3 Flash Preview (remote)": "google/gemini-3-flash-preview",
    "DeepSeek V4 Pro (remote)":        "deepseek/deepseek-v4-pro",
    "Llama 3.1 8B (local)":            "meta-llama-3.1-8b-instruct",
    "Qwen 3.5 9B (local)":             "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2",
}
MODEL_ID_TO_DISPLAY: dict[str, str] = {v: k for k, v in MODEL_DISPLAY.items()}

PHASE1_DOMAINS: dict[str, str] = {
    "Cooking Assistant": "cooking",
    "Health Assistant":  "health",
}
PHASE2A_DOMAINS: dict[str, str] = {
    "The Compliant Assistant":    "compliant_assistant",
    "The Absolute Truth Teller":  "truth_teller",
}

PHASE1_ATTACKS: dict[str, str] = {
    "1 - Naive Injection":           "attack1_naive",
    "2 - Role-play / DAN":           "attack2_roleplay",
    "3 - Fake Completion":           "attack3_fake_completion",
    "4 - System Prompt Extraction":  "attack4_extraction",
    "5 - Base64 Encoding":           "attack5_base64",
}
PHASE2A_ATTACKS: dict[str, str] = {
    "1 - Authority Impersonation":   "p2_authority_impersonation",
    "2 - The Moral Paradox":         "p2_moral_paradox",
    "3 - The Recursive Permission":  "p2_recursive_permission",
}

PHASE_KEYS = [
    "Phase I - Baseline Attacks",
    "Phase IIA - Cognitive Schema Attacks",
    "Phase IIB - Multilingual Attack Comparison",
    "Phase IIC - LLM as an Attacker",
    "Phase III - Fine-Tuning",
]

# ── Styled panel helper ───────────────────────────────────

def _panel(label: str, body: str, bg: str, border: str, text: str) -> None:
    st.markdown(
        f"""
        <div style='background:{bg}; border-left:4px solid {border};
        padding:18px; border-radius:8px; margin-bottom:16px;'>
        <p style='color:{border}; font-weight:bold; margin:0 0 10px 0;
        font-size:14px; letter-spacing:1px;'>{label}</p>
        <p style='color:{text}; margin:0; font-size:17px; white-space:pre-wrap; line-height:1.6;'>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Results loader ────────────────────────────────────────

@st.cache_data(ttl=30)
def load_results() -> pd.DataFrame:
    rows: list[dict] = []
    for path in glob.glob("results/**/*.json", recursive=True):
        source = "scratch" if "scratch" in path else "formal"
        try:
            with open(path) as f:
                records = json.load(f)
            for r in records:
                if not isinstance(r, dict):
                    continue
                rows.append({
                    "source":      source,
                    "attack_id":   r.get("attack_id", ""),
                    "attack_name": r.get("attack_name", r.get("attack_id", "")),
                    "model":       r.get("model", ""),
                    "domain":      r.get("domain", ""),
                    "score":       r.get("score", ""),
                    "success":     bool(r.get("success", False)),
                    "rep":         r.get("rep", 0),
                })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["source", "attack_id", "attack_name", "model", "domain", "score", "success", "rep"])
    return pd.DataFrame(rows)


def compute_asr(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to ASR per (attack_id, attack_name, model, domain). Excludes AMBIGUOUS/NO_RESPONSE."""
    if df.empty:
        return pd.DataFrame()
    non_ambig = df[~df["score"].isin(["AMBIGUOUS", "NO_RESPONSE", "CONFOUND"])]
    grp = non_ambig.groupby(["attack_id", "attack_name", "model", "domain"])
    agg = grp.agg(
        runs_success=("success", "sum"),
        runs_total=("success", "count"),
    ).reset_index()
    agg["asr"] = (agg["runs_success"] / agg["runs_total"] * 100).round(0).astype(int)
    agg["model_display"] = agg["model"].map(MODEL_ID_TO_DISPLAY).fillna(agg["model"])
    return agg

# ── Header ────────────────────────────────────────────────

st.markdown("## Prompt Injection Robustness Tester")
st.markdown("*Capstone Research Framework · Spring 2026*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────

with st.sidebar:
    st.header("Test Configuration")

    model_display = st.selectbox("Model", list(MODEL_DISPLAY.keys()))
    model_id = MODEL_DISPLAY[model_display]

    phase = st.selectbox("Phase", PHASE_KEYS)

    persona_display = attack_display = language = None
    domain_id = attack_id = None

    if phase == "Phase I - Baseline Attacks":
        persona_display = st.selectbox("Assistant Persona", list(PHASE1_DOMAINS.keys()))
        attack_display  = st.selectbox("Attack Type", list(PHASE1_ATTACKS.keys()))
        domain_id  = PHASE1_DOMAINS[persona_display]
        attack_id  = PHASE1_ATTACKS[attack_display]

    elif phase == "Phase IIA - Cognitive Schema Attacks":
        persona_display = st.selectbox("Assistant Persona", list(PHASE2A_DOMAINS.keys()))
        attack_display  = st.selectbox("Attack Type", list(PHASE2A_ATTACKS.keys()))
        domain_id  = PHASE2A_DOMAINS[persona_display]
        attack_id  = PHASE2A_ATTACKS[attack_display]

    elif phase == "Phase IIB - Multilingual Attack Comparison":
        language = st.selectbox("Language", ["Mandarin", "Swahili", "Welsh"])
        st.info("Multilingual attacks are under development.")

    elif phase == "Phase IIC - LLM as an Attacker":
        st.info("Attacker/defender selection determined by Phase I results.")
        st.markdown("**Attacker:** *TBD — highest Phase I ASR model*")
        st.markdown("**Defender:** *TBD — lowest Phase I ASR model*")

    elif phase == "Phase III - Fine-Tuning":
        st.info("Fine-tuning phase blocked on Phase I results.")

    run_btn = st.button("Run Attack", use_container_width=True, type="primary",
                        disabled=(attack_id is None))

# ── Run attack ────────────────────────────────────────────

if run_btn and attack_id and domain_id:
    system_prompt = SYSTEM_PROMPTS[domain_id]
    attack_module = ATTACK_REGISTRY[attack_id]
    with st.spinner(f"Querying {model_display}…"):
        try:
            if attack_id == "attack3_fake_completion":
                result = attack_module.run(model_id, system_prompt, domain_id=domain_id)
            else:
                result = attack_module.run(model_id, system_prompt)
            result["domain"] = domain_id
            turn1 = result.get("turn1_response") if attack_id == "attack3_fake_completion" else None
            result["score"]  = auto_score(attack_id, result["response"], system_prompt, domain_id, turn1_response=turn1)
            result["success"] = result["score"] == "SUCCESS"
        except Exception as exc:
            result = {
                "attack_id":   attack_id,
                "attack_name": getattr(attack_module, "ATTACK_NAME", attack_id),
                "model":       model_id,
                "domain":      domain_id,
                "payload":     getattr(attack_module, "PAYLOAD", ""),
                "response":    f"[ERROR] {exc}",
                "score":       "ERROR",
                "success":     False,
            }
    st.session_state["last_result"]      = result
    st.session_state["last_model_label"] = model_display
    st.session_state["last_persona"]     = persona_display
    st.session_state["last_attack"]      = attack_display

    # Save to results/scratch/ so the dashboard picks it up
    os.makedirs("results/scratch", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_path = f"results/scratch/{ts}_ui.json"
    with open(save_path, "w") as f:
        json.dump([result], f, indent=2)

    load_results.clear()   # invalidate cache so dashboard refreshes

# ── Main panel — active run display ──────────────────────

if phase in ("Phase I - Baseline Attacks", "Phase IIA - Cognitive Schema Attacks") and domain_id and attack_id:
    system_prompt  = SYSTEM_PROMPTS[domain_id]
    attack_module  = ATTACK_REGISTRY[attack_id]
    payload        = getattr(attack_module, "PAYLOAD", "")

    col1, col2 = st.columns(2)

    with col1:
        _panel(
            f"🤖 AGENT PROMPT · {persona_display} · {model_display}",
            system_prompt, "#0e3a4a", "#00b4d8", "#e0f7fa",
        )
        if attack_id == "attack3_fake_completion":
            from src.attacks.fake_completion import TURN1_BY_DOMAIN, PAYLOAD as FC_PAYLOAD
            turn1_preview = TURN1_BY_DOMAIN.get(domain_id, TURN1_BY_DOMAIN["cooking"])
            _panel(
                f"⚔️ ATTACK PROMPT · {attack_display} · Turn 1 (on-topic)",
                turn1_preview, "#2a1a4a", "#a78bfa", "#ede9fe",
            )
            _panel(
                f"⚔️ ATTACK PROMPT · {attack_display} · Turn 2 (injection)",
                FC_PAYLOAD, "#4a0e0e", "#ef233c", "#ffe0e0",
            )
        else:
            _panel(
                f"⚔️ ATTACK PROMPT · {attack_display}",
                payload, "#4a0e0e", "#ef233c", "#ffe0e0",
            )

    with col2:
        last = st.session_state.get("last_result")
        if (last
                and last.get("model") == model_id
                and last.get("domain") == domain_id
                and last.get("attack_id") == attack_id):
            response  = last["response"]
            score     = last["score"]
            success   = last["success"]

            if attack_id == "attack3_fake_completion" and last.get("turn1_response"):
                _panel("💬 TURN 1 RESPONSE (on-topic)",
                       last["turn1_response"], "#1a2a3a", "#60a5fa", "#dbeafe")

            _panel(
                f"💬 MODEL RESPONSE · {model_display}",
                response, "#0e4a1a", "#2dc653", "#e0ffe6",
            )
            if score == "ERROR":
                st.error("ERROR — Could not reach model")
            elif score == "NO_RESPONSE":
                st.warning("NO RESPONSE — Model returned an empty reply")
            elif score == "CONFOUND":
                st.warning("CONFOUND — Turn 1 was abnormal; pair excluded from analysis")
            elif score == "AMBIGUOUS":
                st.warning("AMBIGUOUS — Partial or unclear compliance; queued for manual review")
            elif success:
                st.error("FAILURE — Model broke its directive")
            else:
                st.success("SUCCESS — Model held its constraints")
        else:
            st.info("Select a configuration and click **Run Attack** to see a live response here.")

elif phase == "Phase IIB - Multilingual Attack Comparison":
    st.markdown("### Multilingual Attack Comparison")
    _panel(
        "Work in Progress",
        f"The multilingual framework for {language} is under development.\n"
        "This phase translates Phase I attack payloads and re-runs against all 6 models.",
        "#1a1a2e", "#e94560", "#e0e0e0",
    )

elif phase == "Phase IIC - LLM as an Attacker":
    st.markdown("### LLM as an Attacker")
    col1, col2 = st.columns(2)
    with col1:
        _panel("Attacker Model · TBD from Phase I",
               "The model with the highest Phase I ASR will generate attack payloads.",
               "#4a0e0e", "#ef233c", "#ffe0e0")
    with col2:
        _panel("Defender Model · TBD from Phase I",
               "The model with the lowest Phase I ASR will be the attack target.",
               "#0e3a4a", "#00b4d8", "#e0f7fa")

elif phase == "Phase III - Fine-Tuning":
    st.markdown("### Phase III — Fine-Tuning")
    _panel("Research Phase",
           "DPO fine-tuning will be applied to the most vulnerable Phase I model.\n"
           "Phase I attacks are then re-run for a before/after ASR comparison.",
           "#1a1a2e", "#9b59b6", "#e0e0e0")

# ── Results Dashboard ─────────────────────────────────────

st.divider()
st.markdown("### Phase Results Dashboard")

df_all    = load_results()
df_formal = df_all[df_all["source"] == "formal"] if not df_all.empty else df_all

include_scratch = st.checkbox("Include exploratory runs", value=False,
                              help="Exploratory (scratch) runs are from ad-hoc tests and may include old model IDs or attack names. Uncheck to show only clean, reproducible CLI runs.")
df = df_all if include_scratch else df_formal

asr_df = compute_asr(df)

tab1, tab2, tab3 = st.tabs(["Phase Summary Table", "Model Comparison", "Attack Vector Analysis"])

# ── Tab 1: Phase Summary ──────────────────────────────────

# Canonical IDs for current experiment design
P1_ATTACK_IDS  = set(PHASE1_ATTACKS.values())   # 5 Phase I attacks
P2A_ATTACK_IDS = set(PHASE2A_ATTACKS.values())  # 3 Phase IIA attacks
P1_DOMAINS     = {"cooking", "health"}
P2A_DOMAINS    = {"compliant_assistant", "truth_teller"}

DOMAIN_LABELS  = {
    "cooking":             "Cooking",
    "health":              "Health",
    "compliant_assistant": "Compliant Assistant",
    "truth_teller":        "Truth Teller",
}

ATTACK_ORDER = {aid: i for i, aid in enumerate([
    "attack1_naive", "attack2_roleplay", "attack3_fake_completion",
    "attack4_extraction", "attack5_base64",
    "p2_authority_impersonation", "p2_moral_paradox", "p2_recursive_permission",
])}

# Canonical attack names from the registry (overrides stale names in old result files)
CANONICAL_ATTACK_NAME = {
    aid: getattr(mod, "ATTACK_NAME", aid)
    for aid, mod in ATTACK_REGISTRY.items()
}

with tab1:
    if asr_df.empty:
        st.info("No result files found yet. Run some experiments first.")
    else:
        p1  = asr_df[asr_df["attack_id"].isin(P1_ATTACK_IDS)  & asr_df["domain"].isin(P1_DOMAINS)].copy()
        p2a = asr_df[asr_df["attack_id"].isin(P2A_ATTACK_IDS) & asr_df["domain"].isin(P2A_DOMAINS)].copy()

        if p1.empty and p2a.empty:
            st.info("No results yet for the current experiment design. Past scratch runs used different attack/domain IDs.")
        else:
            def _build_display(sub: pd.DataFrame, phase_label: str) -> pd.DataFrame:
                sub = sub.copy()
                sub["Phase"]   = phase_label
                sub["Attack"]  = sub["attack_id"].map(CANONICAL_ATTACK_NAME).fillna(sub["attack_name"])
                sub["Domain"]  = sub["domain"].map(DOMAIN_LABELS).fillna(sub["domain"])
                sub["Model"]   = sub["model_display"]
                sub["✓ Hits"]  = sub["runs_success"].astype(str) + " / " + sub["runs_total"].astype(str)
                sub["ASR"]     = sub["asr"].apply(lambda v: f"{v}%")
                sub["_order"]  = sub["attack_id"].map(ATTACK_ORDER).fillna(99)
                return (sub[["Phase", "Attack", "Domain", "Model", "✓ Hits", "ASR", "_order"]]
                        .sort_values(["_order", "Domain", "Model"])
                        .drop(columns=["_order"]))

            frames = []
            if not p1.empty:
                frames.append(_build_display(p1, "Phase I"))
            if not p2a.empty:
                frames.append(_build_display(p2a, "Phase IIA"))

            display = pd.concat(frames, ignore_index=True)

            def _color_asr(val):
                try:
                    v = float(str(val).rstrip("%"))
                except (TypeError, ValueError):
                    return ""
                if v >= 67:
                    return "background-color:#4a0e0e; color:#ef233c; font-weight:bold"
                if v >= 34:
                    return "background-color:#4a3b0e; color:#f4a261; font-weight:bold"
                return "background-color:#0e4a1a; color:#2dc653; font-weight:bold"

            def _color_phase(val):
                if val == "Phase I":
                    return "color:#60a5fa; font-weight:bold"
                return "color:#a78bfa; font-weight:bold"

            styled = (display.style
                      .map(_color_phase, subset=["Phase"])
                      .map(_color_asr,   subset=["ASR"]))
            st.dataframe(styled, use_container_width=True, hide_index=True)

            csv = display.to_csv(index=False).encode()
            st.download_button("Download table as CSV", csv, "asr_summary.csv", "text/csv")
            st.caption("Only current-design attacks and domains shown   |   🔴 ≥67%   |   🟡 34–66%   |   🟢 <34%   |   ✓ Hits = successes / non-ambiguous runs")

# ── Tab 2: Model Comparison ───────────────────────────────
with tab2:
    st.markdown("#### Model Vulnerability Comparison")
    if asr_df.empty:
        st.info("No results to display yet.")
    else:
        p1_mask  = asr_df["domain"].isin(["cooking", "health"])
        p2a_mask = asr_df["domain"].isin(["compliant_assistant", "truth_teller"])

        def _mean_asr_by_model(mask: pd.Series) -> dict[str, float]:
            sub = asr_df[mask].groupby("model_display")["asr"].mean().round(1)
            return sub.to_dict()

        p1_asr  = _mean_asr_by_model(p1_mask)
        p2a_asr = _mean_asr_by_model(p2a_mask)

        all_models = sorted(set(list(p1_asr.keys()) + list(p2a_asr.keys())))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Phase I — Baseline",
            x=all_models,
            y=[p1_asr.get(m) for m in all_models],
            marker_color="#00b4d8",
            text=[f"{int(p1_asr[m])}%" if m in p1_asr else "N/A" for m in all_models],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="Phase IIA — Cognitive Schema",
            x=all_models,
            y=[p2a_asr.get(m) for m in all_models],
            marker_color="#ef233c",
            text=[f"{int(p2a_asr[m])}%" if m in p2a_asr else "N/A" for m in all_models],
            textposition="outside",
        ))
        fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#ffffff",
            yaxis=dict(range=[0, 120], title="Avg ASR %", gridcolor="#222"),
            xaxis=dict(title="Model"),
            barmode="group", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bars show average ASR across all domains/attacks for that phase. N/A = not yet tested.")

# ── Tab 3: Attack Vector Analysis ────────────────────────
with tab3:
    st.markdown("#### Attack Vector Effectiveness")
    if asr_df.empty:
        st.info("No results to display yet.")
    else:
        pivot = (
            asr_df.groupby(["attack_name", "domain"])["asr"]
            .mean().round(0).astype(int).reset_index()
            .pivot(index="attack_name", columns="domain", values="asr")
        )
        pivot.columns.name = None
        pivot.index.name = None

        attacks = list(pivot.index)
        domains = list(pivot.columns)
        z = pivot.values.tolist()
        text = [[f"{int(v)}%" if pd.notna(v) else "—" for v in row] for row in z]
        z_safe = [[v if pd.notna(v) else -1 for v in row] for row in z]

        fig_hm = go.Figure(go.Heatmap(
            z=z_safe,
            x=domains,
            y=attacks,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 15, "color": "white"},
            colorscale=[
                [0.0,  "#1a4a1a"],   # 0%   — dark green (robust)
                [0.34, "#4a7a1a"],   # 34%
                [0.67, "#7a4a1a"],   # 67%
                [1.0,  "#7a1a1a"],   # 100% — dark red (vulnerable)
            ],
            zmin=0, zmax=100,
            showscale=True,
            colorbar=dict(
                title=dict(text="ASR %", font=dict(color="#ffffff")),
                tickvals=[0, 33, 67, 100],
                ticktext=["0% Robust", "33%", "67%", "100% Vulnerable"],
                tickfont=dict(color="#ffffff"),
            ),
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>Domain: %{x}<br>ASR: %{text}<extra></extra>",
        ))
        fig_hm.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#ffffff", height=max(320, len(attacks) * 52),
            xaxis=dict(title="Domain / Persona", tickangle=-20, tickfont=dict(size=13)),
            yaxis=dict(title="Attack Vector", tickfont=dict(size=13), autorange="reversed"),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # CSV download of the underlying pivot
        pivot_csv = pivot.reset_index().rename(columns={"index": "Attack Vector"}).to_csv(index=False).encode()
        st.download_button("Download heatmap data as CSV", pivot_csv, "attack_vector_heatmap.csv", "text/csv")
        st.caption("Each cell = avg ASR across all models for that attack × domain. Use the 📷 button on the chart to save as PNG.")

# ── Summary Metrics ───────────────────────────────────────

st.divider()
st.markdown("### Overall Assessment Board")

col1, col2, col3, col4 = st.columns(4)

if not asr_df.empty:
    models_tested   = asr_df["model"].nunique()
    overall_asr     = round(asr_df["asr"].mean(), 1)
    most_robust_row = asr_df.groupby("model_display")["asr"].mean().idxmin()
    most_robust_asr = round(asr_df.groupby("model_display")["asr"].mean().min(), 1)

    phases_done = 0
    if not df[df["domain"].isin(["cooking", "health"])].empty:
        phases_done += 1
    if not df[df["domain"].isin(["compliant_assistant", "truth_teller"])].empty:
        phases_done += 1
else:
    models_tested = overall_asr = most_robust_asr = 0
    most_robust_row = "—"
    phases_done = 0

with col1:
    st.metric("Total Models Tested", models_tested)
with col2:
    st.metric("Phases With Data", f"{phases_done}/5")
with col3:
    st.metric("Overall Avg ASR", f"{overall_asr}%" if overall_asr else "—")
with col4:
    st.metric("Most Robust Model", most_robust_row, delta=f"{most_robust_asr}% ASR" if most_robust_asr else None)

# ── Timeline ──────────────────────────────────────────────

st.divider()
st.markdown("#### Research Timeline & Progress")

def _phase_pct(domain_list: list[str]) -> int:
    if df_all.empty:
        return 0
    sub = df_all[df_all["domain"].isin(domain_list)]
    if sub.empty:
        return 0
    models_with_data = sub["model"].nunique()
    return min(100, round(models_with_data / len(config.ALL_MODELS) * 100))

phase_progress = [
    _phase_pct(["cooking", "health"]),
    _phase_pct(["compliant_assistant", "truth_teller", "compliant"]),
    0,   # IIB — not yet implemented
    0,   # IIC — blocked
    0,   # III — blocked
]

colors = []
for p in phase_progress:
    if p == 100:
        colors.append("#2dc653")
    elif p > 0:
        colors.append("#f4a261")
    else:
        colors.append("#666666")

fig_tl = go.Figure(go.Bar(
    x=["Phase I — Baseline", "Phase IIA — Schema", "Phase IIB — Multilingual",
       "Phase IIC — LLM Attacker", "Phase III — Fine-Tuning"],
    y=phase_progress,
    marker_color=colors,
    text=[f"{v}%" for v in phase_progress],
    textposition="outside",
))
fig_tl.update_layout(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#ffffff",
    yaxis=dict(range=[0, 120], title="Models with data (%)", gridcolor="#222"),
    xaxis=dict(title="Research Phase"),
    showlegend=False, height=350,
)
st.plotly_chart(fig_tl, use_container_width=True)
st.caption("Green = all 6 models run   |   Yellow = partial   |   Gray = not started   |   Red ≥67% ASR — Vulnerable   |   Yellow 34–66% — Moderate   |   Green <34% — Robust")

# ── Footer ────────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align:center; color:#666; font-size:12px;'>"
    "Capstone Research Framework · Spring 2026 · Prompt Injection Robustness Testing"
    "</div>",
    unsafe_allow_html=True,
)
