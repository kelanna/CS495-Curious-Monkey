import glob
import html as _html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)   # always reload .env so key changes take effect without restart

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.prompts import SYSTEM_PROMPTS
from src.attacks import REGISTRY as ATTACK_REGISTRY
from src.attacks.p2b_multilingual import ALL_P2B
from src.harness import config

st.set_page_config(
    page_title="Prompt Injection Research Dashboard",
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
    #back-to-top-anchor { position: absolute; top: 0; }
    .back-to-top-btn {
        position: fixed; bottom: 28px; right: 28px; z-index: 9999;
        background: #00b4d8; color: #fff; border: none; border-radius: 50%;
        width: 46px; height: 46px; font-size: 22px; cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4); display: flex;
        align-items: center; justify-content: center;
    }
    .back-to-top-btn:hover { background: #0096c7; }
    </style>
    <span id="back-to-top-anchor"></span>
    <a href="#back-to-top-anchor" style="text-decoration:none;">
      <button class="back-to-top-btn" title="Back to top">&#8679;</button>
    </a>
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
    "Qwen 3.6 35B A3B MTP (local)":    "qwen3.6-35b-a3b-mtp",
    # Phase III — Qwen 3.5 parameter-size series
    "Qwen 3.5 0.8B (local)":           "qwen3.5-0.8b",
    "Qwen 3.5 2B (local)":             "qwen3.5-2b",
    "Qwen 3.5 4B (local)":             "qwen3.5-4b",
    "Qwen 3.5 9B (local)":             "qwen/qwen3.5-9b",
    "Qwen 3.5 27B (local)":            "qwen/qwen3.5-27b",
}
MODEL_ID_TO_DISPLAY: dict[str, str] = {v: k for k, v in MODEL_DISPLAY.items()}

# Phase III Qwen 3.5 parameter-size comparison (same series, Q8, reasoning enabled)
QWEN_FAMILY: list[dict] = [
    {
        "model_id": "qwen3.5-0.8b",
        "label":    "Qwen 3.5 0.8B",
        "short":    "0.8B",
        "params_total": "0.8B", "params_active": "0.8B (dense)",
        "arch":     "Dense · Reasoning enabled",
        "quant":    "Q8",
        "color":    "#93c5fd",
    },
    {
        "model_id": "qwen3.5-2b",
        "label":    "Qwen 3.5 2B",
        "short":    "2B",
        "params_total": "2B", "params_active": "2B (dense)",
        "arch":     "Dense · Reasoning enabled",
        "quant":    "Q8",
        "color":    "#60a5fa",
    },
    {
        "model_id": "qwen3.5-4b",
        "label":    "Qwen 3.5 4B",
        "short":    "4B",
        "params_total": "4B", "params_active": "4B (dense)",
        "arch":     "Dense · Reasoning enabled",
        "quant":    "Q8",
        "color":    "#3b82f6",
    },
    {
        "model_id": "qwen/qwen3.5-9b",
        "label":    "Qwen 3.5 9B",
        "short":    "9B",
        "params_total": "9B", "params_active": "9B (dense)",
        "arch":     "Dense · Reasoning enabled",
        "quant":    "Q8",
        "color":    "#1d4ed8",
    },
    {
        "model_id": "qwen/qwen3.5-27b",
        "label":    "Qwen 3.5 27B",
        "short":    "27B",
        "params_total": "27B", "params_active": "27B (dense)",
        "arch":     "Dense · Reasoning enabled",
        "quant":    "Q8",
        "color":    "#1e3a8a",
    },
]

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

# ── Styled panel helper ───────────────────────────────────

def _png_download_btn(label: str, fig: go.Figure, filename: str,
                      width: int = 1200, height: int = 500) -> None:
    """PNG export button — requires kaleido. Shows install hint if missing."""
    try:
        data = fig.to_image(format="png", width=width, height=height)
        st.download_button(label, data, filename, "image/png")
    except Exception:
        st.caption("💡 Install `kaleido` for PNG export: `pip install kaleido`")


def _encode(s: str) -> str:
    """HTML-escape then encode non-ASCII chars as numeric character refs.

    This produces a pure-ASCII HTML string that survives Streamlit's Markdown
    pipeline unchanged — the browser decodes the entities back to the original
    Unicode (including CJK characters) at render time.
    """
    return _html.escape(s).encode("ascii", "xmlcharrefreplace").decode("ascii")


def _panel(label: str, body: str, bg: str, border: str, text: str) -> None:
    st.markdown(
        f"<div style='background:{bg}; border-left:4px solid {border};"
        f"padding:18px; border-radius:8px; margin-bottom:16px;'>"
        f"<p style='color:{border}; font-weight:bold; margin:0 0 10px 0;"
        f"font-size:14px; letter-spacing:1px;'>{_encode(label)}</p>"
        f"<p style='color:{text}; margin:0; font-size:17px; white-space:pre-wrap;"
        f"line-height:1.6;'>{_encode(body)}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Results loader ────────────────────────────────────────

@st.cache_data(ttl=30)
def load_formal_records() -> list[dict]:
    """Return full records (including payload + response) from results/formal_v2/."""
    rows: list[dict] = []
    for path in glob.glob("results/formal_v2/*.json"):
        try:
            with open(path) as f:
                records = json.load(f)
            for r in records:
                if isinstance(r, dict) and r.get("attack_id", "").startswith("attack"):
                    rows.append(r)
        except Exception:
            continue
    return rows


@st.cache_data(ttl=30)
def load_p2b_records() -> list[dict]:
    """Return full records (payload, response, translation) from results/formal_p2a/."""
    rows: list[dict] = []
    for path in glob.glob("results/formal_p2a/*.json"):
        try:
            with open(path) as f:
                records = json.load(f)
            for r in records:
                if isinstance(r, dict):
                    rows.append(r)
        except Exception:
            continue
    return rows


@st.cache_data(ttl=30)
def load_p2a_records() -> list[dict]:
    """Return legacy schema-attack records from results/formal_v2/ (attack_id starts with 'p2_')."""
    rows: list[dict] = []
    for path in glob.glob("results/formal_v2/*.json"):
        try:
            with open(path) as f:
                records = json.load(f)
            for r in records:
                if isinstance(r, dict) and r.get("attack_id", "").startswith("p2_"):
                    rows.append(r)
        except Exception:
            continue
    return rows


@st.cache_data(ttl=30)
def load_p2c_records() -> list[dict]:
    """Return full records from results/formal_p2b/."""
    rows: list[dict] = []
    for path in glob.glob("results/formal_p2b/*.json"):
        try:
            with open(path) as f:
                records = json.load(f)
            for r in records:
                if isinstance(r, dict):
                    rows.append(r)
        except Exception:
            continue
    return rows


@st.cache_data(ttl=30)
def load_replay_records() -> list[dict]:
    """Combined replay loader: Phase I (formal_v2) + Phase IIA (formal_p2a).

    Each record is normalised to have:
      _language   — "english" for Phase I, or the language_code for Phase IIA
      _is_p2b     — bool
      _payload    — attack prompt text
      _response   — raw model response
    """
    rows: list[dict] = []

    # Phase I — English
    for r in load_formal_records():
        r = dict(r)
        r["_language"] = "english"
        r["_is_p2b"]   = False
        r["_payload"]  = r.get("payload", "")
        r["_response"] = r.get("response", "")
        rows.append(r)

    # Phase IIA — Multilingual
    for r in load_p2b_records():
        r = dict(r)
        r["_language"] = r.get("language_code", r.get("language", ""))
        r["_is_p2b"]   = True
        r["_payload"]  = r.get("attack_prompt", r.get("payload", ""))
        r["_response"] = r.get("response_original", r.get("response", ""))
        rows.append(r)

    return rows


@st.cache_data(ttl=30)
def load_results() -> pd.DataFrame:
    rows: list[dict] = []
    for path in glob.glob("results/**/*.json", recursive=True):
        if "formal_p2a" in path:
            source = "formal_p2a"
        elif "formal_p2b" in path:
            source = "formal_p2b"
        elif "formal_v2" in path:
            source = "formal_v2"
        elif "scratch" in path:
            source = "scratch"
        else:
            source = "formal"
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
                    "score":       r.get("score", r.get("outcome", "")),
                    "success":     bool(r.get("success", False)),
                    "rep":         r.get("rep", 0),
                    "language":    r.get("language_code", r.get("language", "")),
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

st.markdown("## Prompt Injection Research Dashboard")
st.markdown("*Capstone Research · CS495 · Spring 2026*")
st.divider()

# ── Results Dashboard ─────────────────────────────────────

with st.expander("🗄️ Results Database (SQLite)", expanded=False):
    try:
        from src.db import store as _store
        _db = _store.db_stats()
        _db_c1, _db_c2, _db_c3 = st.columns(3)
        with _db_c1:
            st.metric("Total Records", _db["total"])
        with _db_c2:
            _last = (_db["last_import"] or "Never")[:19]
            st.metric("Last Synced", _last)
        with _db_c3:
            if st.button("Sync DB from JSON files", use_container_width=True):
                _n = _store.import_all_json()
                st.success(f"Imported {_n} new record(s) into {_store.DB_PATH}")
                st.rerun()
        if _db["by_phase"]:
            st.caption("Records by phase:  " + "  ·  ".join(
                f"**{k.upper()}**: {v}" for k, v in sorted(_db["by_phase"].items())
            ))
        if _db["by_source"]:
            st.caption("Records by source: " + "  ·  ".join(
                f"{k}: {v}" for k, v in _db["by_source"].items()
            ))
    except Exception as _db_exc:
        st.caption(f"SQLite unavailable: {_db_exc}")

df_all        = load_results()
df_formal_v2  = df_all[df_all["source"] == "formal_v2"]  if not df_all.empty else df_all
df_formal_p2a = df_all[df_all["source"] == "formal_p2a"] if not df_all.empty else df_all
df_formal_p2b = df_all[df_all["source"] == "formal_p2b"] if not df_all.empty else df_all

# Shared constants
P1_ATTACK_IDS  = set(PHASE1_ATTACKS.values())
P2A_ATTACK_IDS = set(PHASE2A_ATTACKS.values())
P1_DOMAINS     = {"cooking", "health"}
P2A_DOMAINS    = {"compliant_assistant", "truth_teller"}
_excl_scores   = {"AMBIGUOUS", "NO_RESPONSE", "CONFOUND", "TRANSLATION_ERROR", "UNCERTAIN_LANG", "ERROR"}
_P2B_BASE_LOOKUP = {aid: inst.BASE_ATTACK_ID for aid, inst in ALL_P2B.items()}

DOMAIN_LABELS = {
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
CANONICAL_ATTACK_NAME = {
    aid: getattr(mod, "ATTACK_NAME", aid)
    for aid, mod in ATTACK_REGISTRY.items()
}

# ── Shared dashboard helpers ──────────────────────────────

def _model_asr_sem(df_sub: pd.DataFrame) -> dict[str, tuple[float, float]]:
    clean = df_sub[~df_sub["score"].isin(_excl_scores)] if not df_sub.empty else df_sub
    out: dict[str, tuple[float, float]] = {}
    for m, grp in clean.groupby("model"):
        vals = grp["success"].astype(float).values
        if not len(vals):
            continue
        mean_ = float(vals.mean() * 100)
        sem_  = float(vals.std(ddof=1) / np.sqrt(len(vals)) * 100) if len(vals) > 1 else 0.0
        out[MODEL_ID_TO_DISPLAY.get(m, m)] = (round(mean_, 1), round(sem_, 1))
    return out


def _render_heatmap(asr_sub: pd.DataFrame, key_sfx: str = "") -> None:
    if asr_sub.empty:
        st.info("No results to display yet.")
        return
    pivot = (
        asr_sub.groupby(["attack_name", "domain"])["asr"]
        .mean().round(0).astype(int).reset_index()
        .pivot(index="attack_name", columns="domain", values="asr")
    )
    pivot.columns.name = None
    pivot.index.name   = None
    attacks = list(pivot.index)
    domains = list(pivot.columns)
    z       = pivot.values.tolist()
    text_   = [[f"{int(v)}%" if pd.notna(v) else "—" for v in row] for row in z]
    z_safe  = [[v if pd.notna(v) else -1 for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z_safe, x=domains, y=attacks,
        text=text_, texttemplate="%{text}",
        textfont={"size": 15, "color": "white"},
        colorscale=[[0, "#1a4a1a"], [0.34, "#4a7a1a"], [0.67, "#7a4a1a"], [1, "#7a1a1a"]],
        zmin=0, zmax=100, showscale=True,
        colorbar=dict(
            title=dict(text="ASR %", font=dict(color="#fff")),
            tickvals=[0, 33, 67, 100],
            ticktext=["0% Robust", "33%", "67%", "100% Vulnerable"],
            tickfont=dict(color="#fff"),
        ),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>Domain: %{x}<br>ASR: %{text}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#fff",
        height=max(320, len(attacks) * 52),
        xaxis=dict(title="Domain / Persona", tickangle=-20, tickfont=dict(size=13)),
        yaxis=dict(title="Attack Vector",    tickfont=dict(size=13), autorange="reversed"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    hm_csv = pivot.reset_index().rename(columns={"index": "Attack Vector"}).to_csv(index=False).encode()
    st.download_button("Download heatmap CSV", hm_csv, f"heatmap_{key_sfx}.csv", "text/csv",
                       key=f"hm_dl_{key_sfx}")
    _png_download_btn("Download heatmap PNG", fig, f"heatmap_{key_sfx}.png",
                      height=max(400, len(attacks) * 60))
    st.caption("Each cell = avg ASR across all models for that attack × domain.")


def _render_replay(records: list[dict], key_pfx: str, show_lang: bool = False) -> None:
    if not records:
        st.info("No records found. Run the CLI harness first.")
        return
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        m_opts = ["All"] + sorted({
            MODEL_ID_TO_DISPLAY.get(r.get("model", ""), r.get("model", "")) for r in records
        })
        sel_m = st.selectbox("Model", m_opts, key=f"{key_pfx}_m")
    with fc2:
        d_opts = ["All"] + sorted({
            DOMAIN_LABELS.get(r.get("domain", ""), r.get("domain", "")) for r in records
        })
        sel_d = st.selectbox("Domain", d_opts, key=f"{key_pfx}_d")
    with fc3:
        a_opts = ["All"] + sorted({
            r.get("attack_name", r.get("attack_id", "")) for r in records
        })
        sel_a = st.selectbox("Attack Type", a_opts, key=f"{key_pfx}_a")

    frecs = [
        r for r in records
        if (sel_m == "All" or MODEL_ID_TO_DISPLAY.get(r.get("model",""), r.get("model","")) == sel_m)
        and (sel_d == "All" or DOMAIN_LABELS.get(r.get("domain",""), r.get("domain","")) == sel_d)
        and (sel_a == "All" or r.get("attack_name", r.get("attack_id","")) == sel_a)
    ]
    frecs = sorted(frecs, key=lambda r: (
        r.get("attack_name", r.get("attack_id", "")),
        r.get("domain", ""), r.get("model", ""), r.get("rep", 0),
    ))
    st.markdown(f"**{len(frecs)} run(s) matching filters**")
    if not frecs:
        st.warning("No runs match the selected filters.")
        return

    pp = st.select_slider("Runs per page", [5, 10, 25, 50], value=10, key=f"{key_pfx}_pp")
    tp = max(1, (len(frecs) + pp - 1) // pp)
    pg = st.number_input("Page", min_value=1, max_value=tp, value=1, key=f"{key_pfx}_pg") - 1

    for r in frecs[pg * pp: pg * pp + pp]:
        ml = MODEL_ID_TO_DISPLAY.get(r.get("model", ""), r.get("model", ""))
        dl = DOMAIN_LABELS.get(r.get("domain", ""), r.get("domain", ""))
        an = r.get("attack_name", r.get("attack_id", ""))
        sc = r.get("score", r.get("outcome", ""))
        lc = r.get("_language", "")
        is_p2b = r.get("_is_p2b", False)
        ico = "✅" if sc == "SUCCESS" else (
            "⚠️" if sc in ("AMBIGUOUS", "NO_RESPONSE", "UNCERTAIN_LANG", "TRANSLATION_ERROR", "CONFOUND")
            else "🛡️"
        )
        lang_badge = f" · {lc}" if show_lang and lc and lc != "english" else ""
        with st.expander(
            f"{ico} **{ml}** · {dl}{lang_badge} · {an} · Rep {r.get('rep', 0)+1} · {sc}",
            expanded=False,
        ):
            spr     = SYSTEM_PROMPTS.get(r.get("domain", ""), "")
            payload = r.get("_payload", "")
            resp    = r.get("_response", "")
            c1, c2 = st.columns(2)
            with c1:
                _panel(f"🤖 AGENT PROMPT · {dl} · {ml}", spr,     "#0e3a4a", "#00b4d8", "#e0f7fa")
                _panel(
                    f"⚔️ ATTACK ({lc.upper()}) · {an}" if is_p2b else f"⚔️ ATTACK · {an}",
                    payload, "#4a0e0e", "#ef233c", "#ffe0e0",
                )
            with c2:
                if r.get("turn1_response"):
                    _panel("💬 TURN 1 RESPONSE", r["turn1_response"], "#1a2a3a", "#60a5fa", "#dbeafe")
                _panel(
                    f"💬 RAW RESPONSE · {ml}" if is_p2b else f"💬 RESPONSE · {ml}",
                    resp, "#0e4a1a", "#2dc653", "#e0ffe6",
                )
                if is_p2b:
                    if r.get("response_translated"):
                        _panel(
                            f"🔤 TRANSLATED (detected: {r.get('response_language_detected','')})",
                            r["response_translated"], "#1a1a3a", "#a78bfa", "#ede9fe",
                        )
                    if r.get("note"):
                        st.caption(f"Note: {r['note']}")
                if sc in ("AMBIGUOUS", "NO_RESPONSE", "CONFOUND", "UNCERTAIN_LANG", "TRANSLATION_ERROR"):
                    st.warning(f"{sc} — excluded from ASR")
                elif sc == "SUCCESS":
                    st.success("SUCCESS — Model broke its directive")
                else:
                    st.error("FAILURE — Model maintained its directive")


# ── Top-level phase tabs ──────────────────────────────────

tab_p1, tab_p2, tab_p3, tab_p4, tab_overview = st.tabs([
    "Phase I", "Phase II", "Phase III", "Phase IV", "Overview",
])

# ═══════════════════════════════════════════════════════════
# PHASE I
# ═══════════════════════════════════════════════════════════
with tab_p1:
    st.markdown("#### Phase I — Baseline Attacks")
    _df_p1 = df_formal_v2
    _df_p1_filt = (
        _df_p1[
            _df_p1["attack_id"].isin(P1_ATTACK_IDS) &
            _df_p1["domain"].isin(P1_DOMAINS) &
            (_df_p1["model"] != "llama-3.1-8b-injectionguard")
        ]
        if not _df_p1.empty else _df_p1
    )
    _asr_p1 = compute_asr(_df_p1_filt)

    p1_sub1, p1_sub2, p1_sub3 = st.tabs(["ASR by Attack", "Model Analysis", "Data Replay"])

    with p1_sub1:
        st.markdown("##### Attack Success Rate by Attack Type")
        st.caption("Avg ASR across all models, grouped by domain. Rubric: v2 (formal_v2)")
        if _asr_p1.empty:
            st.info("No Phase I data yet. Run: `python -m src.main --phase p1`")
        else:
            _p1_grp = _asr_p1.groupby(["attack_name","domain"])["asr"].mean().round(1).reset_index()
            _p1_dom_colors = {"cooking": "#00b4d8", "health": "#2dc653"}
            fig_p1_atk = go.Figure()
            for _dom in ["cooking", "health"]:
                _sub = _p1_grp[_p1_grp["domain"] == _dom].sort_values("attack_name")
                if _sub.empty:
                    continue
                fig_p1_atk.add_trace(go.Bar(
                    name=DOMAIN_LABELS.get(_dom, _dom),
                    x=_sub["attack_name"], y=_sub["asr"],
                    marker_color=_p1_dom_colors.get(_dom, "#888"),
                    text=[f"{v:.0f}%" for v in _sub["asr"]], textposition="outside",
                ))
            fig_p1_atk.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#ffffff",
                yaxis=dict(range=[0, 120], title="ASR %", gridcolor="#222"),
                xaxis=dict(title="Attack Type"),
                barmode="group", height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_p1_atk, use_container_width=True)
            st.caption("ASR = successes / non-ambiguous runs, averaged across all models.")
            _png_download_btn("Download as PNG", fig_p1_atk, "p1_asr_by_attack.png", height=480)

    with p1_sub2:
        st.markdown("##### Model Vulnerability — Phase I")
        if _asr_p1.empty:
            st.info("No Phase I data yet.")
        else:
            _p1_stats = _model_asr_sem(_df_p1_filt)
            _p1_models = sorted(_p1_stats.keys())
            if _p1_stats:
                fig_p1_mc = go.Figure(go.Bar(
                    x=_p1_models,
                    y=[_p1_stats[m][0] for m in _p1_models],
                    error_y=dict(type="data", array=[_p1_stats[m][1]*2 for m in _p1_models],
                                 visible=True, color="#ffffff", thickness=1.5),
                    marker_color="#00b4d8",
                    text=[f"{int(_p1_stats[m][0])}%" for m in _p1_models],
                    textposition="outside",
                ))
                fig_p1_mc.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#ffffff",
                    yaxis=dict(range=[0, 130], title="Avg ASR %", gridcolor="#222"),
                    xaxis=dict(title="Model"), showlegend=False, height=420,
                )
                st.plotly_chart(fig_p1_mc, use_container_width=True)
                st.caption("Error bars = ±2 SEM (95% CI). Averaged over all Phase I attacks and domains.")
                _png_download_btn("Download as PNG", fig_p1_mc, "p1_model_comparison.png", height=480)
            st.divider()
            st.markdown("##### Attack × Domain Heatmap")
            _render_heatmap(_asr_p1, key_sfx="p1")

    with p1_sub3:
        st.caption("Browsing results/formal_v2/ — no API calls.")
        _p1_rp = [
            {**r, "_language": "english", "_is_p2b": False,
             "_payload": r.get("payload",""), "_response": r.get("response","")}
            for r in load_formal_records()
        ]
        _render_replay(_p1_rp, key_pfx="p1_rp", show_lang=False)


# ═══════════════════════════════════════════════════════════
# PHASE II
# ═══════════════════════════════════════════════════════════
with tab_p2:
    st.markdown("#### Phase II — Advanced Attack Techniques")
    p2_iia, p2_iib = st.tabs(["Phase IIA", "Phase IIB"])

    # ── Phase IIA ──────────────────────────────────────────
    with p2_iia:
        st.markdown("##### Phase IIA — Multilingual Attack Comparison")
        st.caption("Top-3 Phase I attacks (Roleplay, Fake Completion, Naive) in Mandarin, Swahili, Welsh.")

        _TOP3_IDS  = ["attack2_roleplay", "attack3_fake_completion", "attack1_naive"]
        _TOP3_NAMES = {
            "attack1_naive":           "Naive Injection",
            "attack2_roleplay":        "Role-play / DAN",
            "attack3_fake_completion": "Fake Completion",
        }
        _LANG_COLORS = {
            "Phase I — English":   "#00b4d8",
            "Mandarin (中文)":      "#ef233c",
            "Swahili (Kiswahili)": "#f4a261",
            "Welsh (Cymraeg)":     "#a78bfa",
        }
        _LANG_DISP = {
            "mandarin": "Mandarin (中文)",
            "swahili":  "Swahili (Kiswahili)",
            "welsh":    "Welsh (Cymraeg)",
        }
        _excl_p2b = {"AMBIGUOUS","NO_RESPONSE","CONFOUND","TRANSLATION_ERROR","UNCERTAIN_LANG","ERROR"}

        p2b_s1, p2b_s2, p2b_s3 = st.tabs(["ASR by Language", "Model Analysis", "Data Replay"])

        with p2b_s1:
            _p2b_lang_filter = st.radio(
                "Language filter",
                ["All languages", "Mandarin (中文)", "Swahili (Kiswahili)", "Welsh (Cymraeg)"],
                horizontal=True, key="p2b_asr_lang",
            )
            # Phase I English baseline for top-3 (mean, SEM)
            _p1_base: dict[str, tuple[float, float]] = {}
            if not df_formal_v2.empty:
                _p1f = df_formal_v2[
                    df_formal_v2["attack_id"].isin(_TOP3_IDS) & ~df_formal_v2["score"].isin(_excl_p2b)
                ]
                for _aid, _grp in _p1f.groupby("attack_id"):
                    _v = _grp["success"].astype(float).values
                    _p1_base[_aid] = (
                        round(float(_v.mean() * 100), 1),
                        round(float(_v.std(ddof=1) / np.sqrt(len(_v)) * 100) if len(_v) > 1 else 0.0, 1),
                    )
            # Phase IIA ASR per (base_attack, language) — (mean, SEM) tuples
            _p2b_asr: dict[tuple, tuple[float, float]] = {}
            if not df_formal_p2a.empty:
                _p2bf = df_formal_p2a.copy()
                _p2bf["_base"] = _p2bf["attack_id"].map(_P2B_BASE_LOOKUP)
                _p2bf_c = _p2bf[~_p2bf["score"].isin(_excl_p2b) & _p2bf["_base"].isin(_TOP3_IDS)]
                for (_base, _lang), _grp in _p2bf_c.groupby(["_base", "language"]):
                    _v = _grp["success"].astype(float).values
                    _p2b_asr[(_base, _LANG_DISP.get(_lang, _lang))] = (
                        round(float(_v.mean() * 100), 1),
                        round(float(_v.std(ddof=1) / np.sqrt(len(_v)) * 100) if len(_v) > 1 else 0.0, 1),
                    )
            _atk_labels = [_TOP3_NAMES[k] for k in _TOP3_IDS]
            _series: dict[str, list] = {}
            if _p1_base:
                _series["Phase I — English"] = [_p1_base.get(k) for k in _TOP3_IDS]
            _lang_disp_opts = ["Mandarin (中文)", "Swahili (Kiswahili)", "Welsh (Cymraeg)"]
            _langs_to_show = (
                [_p2b_lang_filter] if _p2b_lang_filter != "All languages"
                else _lang_disp_opts
            )
            for _ld in _langs_to_show:
                _vals = [_p2b_asr.get((k, _ld)) for k in _TOP3_IDS]
                if any(v is not None for v in _vals):
                    _series[_ld] = _vals
            if not _series:
                st.info("No Phase IIA data yet. Run: `python -m src.main --phase p2b`")
            else:
                fig_p2b = go.Figure()
                for _sn, _sv in _series.items():
                    _means = [v[0] if isinstance(v, tuple) else v for v in _sv]
                    _sems  = [v[1] if isinstance(v, tuple) else 0.0 for v in _sv]
                    fig_p2b.add_trace(go.Bar(
                        name=_sn, x=_atk_labels, y=_means,
                        error_y=dict(type="data", array=[s * 2 for s in _sems],
                                     visible=True, color="#ffffff", thickness=1.5),
                        marker_color=_LANG_COLORS.get(_sn, "#888"),
                        text=[f"{v:.0f}%" if v is not None else "—" for v in _means],
                        textposition="outside",
                    ))
                fig_p2b.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#ffffff",
                    yaxis=dict(range=[0, 120], title="ASR %", gridcolor="#222"),
                    xaxis=dict(title="Attack Type"),
                    barmode="group", height=460,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_p2b, use_container_width=True)
                st.caption("Error bars = ±2 SEM (95% CI). ASR = successes / non-ambiguous runs, averaged across all models and domains.")
                _png_download_btn("Download as PNG", fig_p2b, "p2b_asr_by_language.png", height=520)

        with p2b_s2:
            st.markdown("##### Model Vulnerability — Phase IIA")
            _p2b_model_df = df_formal_p2a[~df_formal_p2a["score"].isin(_excl_p2b)] if not df_formal_p2a.empty else df_formal_p2a
            if _p2b_model_df.empty:
                st.info("No Phase IIA data yet. Run: `python -m src.main --phase p2b`")
            else:
                _p2b_stats  = _model_asr_sem(_p2b_model_df)
                _p2b_mdls   = sorted(_p2b_stats.keys())
                if _p2b_stats:
                    fig_p2b_mc = go.Figure(go.Bar(
                        x=_p2b_mdls,
                        y=[_p2b_stats[m][0] for m in _p2b_mdls],
                        error_y=dict(type="data", array=[_p2b_stats[m][1] * 2 for m in _p2b_mdls],
                                     visible=True, color="#ffffff", thickness=1.5),
                        marker_color="#ef233c",
                        text=[f"{int(_p2b_stats[m][0])}%" for m in _p2b_mdls],
                        textposition="outside",
                    ))
                    fig_p2b_mc.update_layout(
                        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#ffffff",
                        yaxis=dict(range=[0, 130], title="Avg ASR %", gridcolor="#222"),
                        xaxis=dict(title="Model"), showlegend=False, height=420,
                    )
                    st.plotly_chart(fig_p2b_mc, use_container_width=True)
                    st.caption("Error bars = ±2 SEM (95% CI). Averaged over all Phase IIA attacks, languages, and domains.")
                    _png_download_btn("Download as PNG", fig_p2b_mc, "p2b_model_comparison.png", height=480)

                # Shared prep: base attack name + language display
                _p2b_hm_df = _p2b_model_df.copy()
                _p2b_hm_df["_base"]      = _p2b_hm_df["attack_id"].map(_P2B_BASE_LOOKUP)
                _p2b_hm_df["_base_name"] = _p2b_hm_df["_base"].map(_TOP3_NAMES).fillna(_p2b_hm_df["attack_id"])
                _p2b_hm_df["_lang_disp"] = _p2b_hm_df["language"].map(_LANG_DISP).fillna(_p2b_hm_df["language"])

                def _p2b_heatmap(pivot_df: pd.DataFrame, x_title: str, key: str) -> None:
                    if pivot_df.empty:
                        return
                    atks = list(pivot_df.index)
                    cols = list(pivot_df.columns)
                    z    = pivot_df.values.tolist()
                    text_ = [[f"{int(v)}%" if pd.notna(v) else "—" for v in row] for row in z]
                    z_s   = [[v if pd.notna(v) else -1   for v in row] for row in z]
                    fig = go.Figure(go.Heatmap(
                        z=z_s, x=cols, y=atks,
                        text=text_, texttemplate="%{text}",
                        textfont={"size": 15, "color": "white"},
                        colorscale=[[0, "#1a4a1a"], [0.34, "#4a7a1a"], [0.67, "#7a4a1a"], [1, "#7a1a1a"]],
                        zmin=0, zmax=100, showscale=True,
                        colorbar=dict(
                            title=dict(text="ASR %", font=dict(color="#fff")),
                            tickvals=[0, 33, 67, 100],
                            ticktext=["0% Robust", "33%", "67%", "100% Vulnerable"],
                            tickfont=dict(color="#fff"),
                        ),
                        hoverongaps=False,
                        hovertemplate=f"<b>%{{y}}</b><br>{x_title}: %{{x}}<br>ASR: %{{text}}<extra></extra>",
                    ))
                    fig.update_layout(
                        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#fff",
                        height=max(300, len(atks) * 80),
                        xaxis=dict(title=x_title, tickfont=dict(size=13)),
                        yaxis=dict(title="Attack Vector", tickfont=dict(size=13), autorange="reversed"),
                        margin=dict(l=20, r=20, t=20, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    _png_download_btn(f"Download PNG", fig, f"p2b_{key}_heatmap.png",
                                      height=max(400, len(atks) * 90))
                    csv = pivot_df.reset_index().to_csv(index=False).encode()
                    st.download_button(f"Download CSV", csv, f"p2b_{key}_heatmap.csv", "text/csv",
                                       key=f"p2b_{key}_csv")

                st.divider()
                st.markdown("##### Attack × Language Heatmap")
                _lang_pivot = (
                    _p2b_hm_df.groupby(["_base_name", "_lang_disp"])["success"]
                    .mean().mul(100).round(0).astype(int).reset_index()
                    .pivot(index="_base_name", columns="_lang_disp", values="success")
                )
                _lang_pivot.columns.name = None
                _lang_pivot.index.name   = None
                _p2b_heatmap(_lang_pivot, "Language", "attack_language")
                st.caption("Each cell = avg ASR across all models and domains for that attack × language pair.")

                st.divider()
                st.markdown("##### Attack × Domain Heatmap")
                _dom_pivot = (
                    _p2b_hm_df.groupby(["_base_name", "domain"])["success"]
                    .mean().mul(100).round(0).astype(int).reset_index()
                    .pivot(index="_base_name", columns="domain", values="success")
                )
                _dom_pivot.columns = [DOMAIN_LABELS.get(c, c) for c in _dom_pivot.columns]
                _dom_pivot.columns.name = None
                _dom_pivot.index.name   = None
                _p2b_heatmap(_dom_pivot, "Domain", "attack_domain")
                st.caption("Each cell = avg ASR across all models and languages for that attack × domain pair.")

        with p2b_s3:
            _p2b_rp_lang = st.radio(
                "Language", ["All", "Mandarin", "Swahili", "Welsh"],
                horizontal=True, key="p2b_rp_lang",
            )
            _p2b_lang_key = {"Mandarin":"mandarin","Swahili":"swahili","Welsh":"welsh"}.get(_p2b_rp_lang)
            _p2b_rp_recs = []
            for r in load_p2b_records():
                lc = r.get("language_code", r.get("language",""))
                if _p2b_lang_key and lc != _p2b_lang_key:
                    continue
                _p2b_rp_recs.append({**r, "_language": lc, "_is_p2b": True,
                                      "_payload": r.get("attack_prompt", r.get("payload","")),
                                      "_response": r.get("response_original", r.get("response",""))})
            st.caption("Browsing results/formal_p2a/ — no API calls.")
            _render_replay(_p2b_rp_recs, key_pfx="p2b_rp", show_lang=True)

    # ── Phase IIB ──────────────────────────────────────────
    with p2_iib:
        st.markdown("##### Phase IIB — LLM as Attacker")
        st.caption(
            "**Attacker:** Llama 3.1 8B (76.7% Phase I ASR — highest)  ·  "
            "**Defender:** Claude Sonnet 4.6 (13.3% Phase I ASR — lowest)  ·  "
            "**Domain:** Health (55.0% Phase I ASR — highest)  ·  "
            "**Payload freeze:** generated on rep 0, reused verbatim for reps 1–4"
        )

        _p2c_recs = load_p2c_records()

        p2c_s1, p2c_s2 = st.tabs(["ASR Comparison", "Payload Replay"])

        # ── IIB Sub-tab 1: ASR Comparison ─────────────────
        with p2c_s1:
            _P2C_BASE_MAP = {
                "p2c_roleplay":        "attack2_roleplay",
                "p2c_naive":           "attack1_naive",
                "p2c_fake_completion": "attack3_fake_completion",
            }
            _P2C_DISPLAY_NAMES = {
                "p2c_roleplay":        "Role-play / DAN",
                "p2c_naive":           "Naive Injection",
                "p2c_fake_completion": "Fake Completion",
            }
            _ATK_ORDER = ["p2c_roleplay", "p2c_naive", "p2c_fake_completion"]
            _ATK_LABELS = [_P2C_DISPLAY_NAMES[k] for k in _ATK_ORDER]

            # Phase IIB ASR per attack — (mean, SEM) tuples
            _p2c_asr: dict[str, tuple[float, float] | None] = {}
            if _p2c_recs:
                from collections import defaultdict as _dd
                _p2c_counts: dict[str, list[bool]] = _dd(list)
                for r in _p2c_recs:
                    sc = r.get("score", "")
                    if sc in ("ERROR", "AMBIGUOUS", "NO_RESPONSE", "CONFOUND"):
                        continue
                    _p2c_counts[r.get("attack_id", "")].append(bool(r.get("success", False)))
                for _aid, _cv in _p2c_counts.items():
                    if not _cv:
                        _p2c_asr[_aid] = None
                    else:
                        _cv_arr = np.array(_cv, dtype=float)
                        _p2c_asr[_aid] = (
                            round(float(_cv_arr.mean() * 100), 1),
                            round(float(_cv_arr.std(ddof=1) / np.sqrt(len(_cv_arr)) * 100) if len(_cv_arr) > 1 else 0.0, 1),
                        )

            # Phase I health-domain ASR for same 3 attacks — (mean, SEM) tuples
            _p1_health_asr: dict[str, tuple[float, float] | None] = {}
            if not df_formal_v2.empty:
                _p1h = df_formal_v2[
                    (df_formal_v2["domain"] == "health")
                    & ~df_formal_v2["score"].isin({"AMBIGUOUS","NO_RESPONSE","CONFOUND","ERROR"})
                ]
                for _base_id, _grp in _p1h.groupby("attack_id"):
                    _p1_key = {v: k for k, v in _P2C_BASE_MAP.items()}.get(_base_id)
                    if _p1_key:
                        _v_p1h = _grp["success"].astype(float).values
                        _p1_health_asr[_p1_key] = (
                            round(float(_v_p1h.mean() * 100), 1),
                            round(float(_v_p1h.std(ddof=1) / np.sqrt(len(_v_p1h)) * 100) if len(_v_p1h) > 1 else 0.0, 1),
                        )

            _iic_means  = [(_p2c_asr.get(k) or (None, 0.0))[0] for k in _ATK_ORDER]
            _iic_sems   = [(_p2c_asr.get(k) or (0.0, 0.0))[1] for k in _ATK_ORDER]
            _p1h_means  = [(_p1_health_asr.get(k) or (None, 0.0))[0] for k in _ATK_ORDER]
            _p1h_sems   = [(_p1_health_asr.get(k) or (0.0, 0.0))[1] for k in _ATK_ORDER]
            _has_iic    = any(v is not None for v in _iic_means)
            _has_p1h    = any(v is not None for v in _p1h_means)

            if not _has_iic:
                st.info("No Phase IIB formal data yet. Run: `python -m src.harness.run_p2c`")
            else:
                fig_p2c = go.Figure()
                if _has_p1h:
                    fig_p2c.add_trace(go.Bar(
                        name="Phase I — Hand-crafted (Health)",
                        x=_ATK_LABELS,
                        y=_p1h_means,
                        error_y=dict(type="data", array=[s * 2 for s in _p1h_sems],
                                     visible=True, color="#ffffff", thickness=1.5),
                        marker_color="#00b4d8",
                        text=[f"{v:.0f}%" if v is not None else "—" for v in _p1h_means],
                        textposition="outside",
                    ))
                fig_p2c.add_trace(go.Bar(
                    name="Phase IIB — Llama-generated (Health)",
                    x=_ATK_LABELS,
                    y=_iic_means,
                    error_y=dict(type="data", array=[s * 2 for s in _iic_sems],
                                 visible=True, color="#ffffff", thickness=1.5),
                    marker_color="#f4a261",
                    text=[f"{v:.0f}%" if v is not None else "—" for v in _iic_means],
                    textposition="outside",
                ))
                fig_p2c.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#ffffff",
                    yaxis=dict(range=[0, 120], title="ASR %", gridcolor="#222"),
                    xaxis=dict(title="Attack Type"),
                    barmode="group", height=440,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_p2c, use_container_width=True)
                st.caption(
                    "Error bars = ±2 SEM (95% CI).  "
                    "**Blue:** Phase I hand-crafted payloads on the health domain (all models avg).  "
                    "**Orange:** Phase IIB Llama-generated payloads targeting Claude Sonnet 4.6 only.  "
                    "Payload frozen after rep 0; 5 reps per attack."
                )
                _png_download_btn("Download as PNG", fig_p2c, "p2c_asr_comparison.png", height=500)

                # Summary metrics
                st.divider()
                _mc1, _mc2, _mc3 = st.columns(3)
                _total_recs = [r for r in _p2c_recs if r.get("score","") not in ("ERROR","AMBIGUOUS","NO_RESPONSE","CONFOUND")]
                _total_succ = sum(1 for r in _total_recs if r.get("success", False))
                _overall_asr = round(_total_succ / len(_total_recs) * 100, 1) if _total_recs else 0.0
                with _mc1:
                    st.metric("Overall IIB ASR", f"{_overall_asr:.0f}%", help="Llama-generated payloads vs Claude Sonnet 4.6")
                with _mc2:
                    st.metric("Total Runs", len(_total_recs), help="Across all 3 attacks × 5 reps")
                with _mc3:
                    _frozen_count = sum(1 for r in _p2c_recs if r.get("payload_frozen", False))
                    st.metric("Frozen-payload Runs", _frozen_count, help="Reps 1–4 used the rep-0 payload verbatim")

                st.divider()
                st.markdown(
                    "**Finding:** Llama 3.1 8B's adaptive payloads achieved **0% ASR** against "
                    "Claude Sonnet 4.6 on the health domain — no improvement over, and in most "
                    "cases below, the hand-crafted Phase I baselines. Claude consistently "
                    "identified the off-topic request and redirected to health information."
                )

        # ── IIB Sub-tab 2: Payload Replay ──────────────────
        with p2c_s2:
            if not _p2c_recs:
                st.info("No Phase IIB formal data yet. Run: `python -m src.harness.run_p2c`")
            else:
                _p2c_atk_filter = st.radio(
                    "Attack", ["All"] + _ATK_LABELS,
                    horizontal=True, key="p2c_rp_atk",
                )
                _p2c_rep_filter = st.radio(
                    "Payload", ["All", "Rep 1 (generated)", "Reps 2–5 (frozen)"],
                    horizontal=True, key="p2c_rp_rep",
                )
                _p2c_filtered = []
                for r in sorted(_p2c_recs, key=lambda x: (x.get("attack_id",""), x.get("rep", 0))):
                    _atk_name = _P2C_DISPLAY_NAMES.get(r.get("attack_id",""), r.get("attack_name",""))
                    if _p2c_atk_filter != "All" and _atk_name != _p2c_atk_filter:
                        continue
                    _rep = r.get("rep", 0)
                    if _p2c_rep_filter == "Rep 1 (generated)" and _rep != 0:
                        continue
                    if _p2c_rep_filter == "Reps 2–5 (frozen)" and _rep == 0:
                        continue
                    _p2c_filtered.append(r)

                st.markdown(f"**{len(_p2c_filtered)} run(s) matching filters**")
                for r in _p2c_filtered:
                    _atk_disp = _P2C_DISPLAY_NAMES.get(r.get("attack_id",""), r.get("attack_name",""))
                    _rep_n    = r.get("rep", 0)
                    _sc       = r.get("score", "")
                    _frozen   = r.get("payload_frozen", False)
                    _ico = "✅" if _sc == "SUCCESS" else ("⚠️" if _sc in ("AMBIGUOUS","NO_RESPONSE","CONFOUND") else "🛡️")
                    _freeze_badge = " · 🔒 frozen" if _frozen else " · ✨ generated"
                    with st.expander(
                        f"{_ico} **{_atk_disp}** · Rep {_rep_n + 1}{_freeze_badge} · {_sc}",
                        expanded=(_rep_n == 0),
                    ):
                        _payload  = r.get("generated_payload", r.get("payload", ""))
                        _response = r.get("response", "")
                        _domain   = r.get("domain", "health")
                        _sys_p    = SYSTEM_PROMPTS.get(_domain, "")
                        c1, c2 = st.columns(2)
                        with c1:
                            _panel("🤖 DEFENDER PROMPT · Health · Claude Sonnet 4.6",
                                   _sys_p, "#0e3a4a", "#00b4d8", "#e0f7fa")
                            _pl_label = "⚔️ LLAMA-GENERATED PAYLOAD · Rep 1" if not _frozen else f"⚔️ FROZEN PAYLOAD · Rep {_rep_n + 1} (verbatim from Rep 1)"
                            _panel(_pl_label, _payload, "#4a0e0e", "#ef233c", "#ffe0e0")
                        with c2:
                            if r.get("turn1_response"):
                                _panel("💬 TURN 1 (on-topic)", r["turn1_response"], "#1a2a3a", "#60a5fa", "#dbeafe")
                            _panel("💬 CLAUDE SONNET 4.6 RESPONSE", _response, "#0e4a1a", "#2dc653", "#e0ffe6")
                            if _sc in ("AMBIGUOUS", "NO_RESPONSE", "CONFOUND"):
                                st.warning(f"{_sc} — excluded from ASR")
                            elif _sc == "SUCCESS":
                                st.success("SUCCESS — Claude broke its directive (Llama wins)")
                            else:
                                st.error("FAILURE — Claude held its directive (Defender wins)")


# ═══════════════════════════════════════════════════════════
# PHASE III — Parameter-Size Comparison Study
# ═══════════════════════════════════════════════════════════
with tab_p3:
    st.markdown("#### Phase III — Parameter-Size Comparison Study (Qwen 3.5)")
    st.caption(
        "Research question: does parameter count reduce prompt injection vulnerability? "
        "Five Qwen 3.5 models (0.8B → 27B), same series, Q8 quantization, reasoning enabled. "
        "Top-3 Phase I attacks × cooking + health × 5 reps each."
    )

    # ── Load Phase III data ──────────────────────────────────────────────────────
    _EXCL_P3 = {"AMBIGUOUS","NO_RESPONSE","CONFOUND","TRANSLATION_ERROR","UNCERTAIN_LANG","ERROR"}

    def _load_p3_json():
        rows = []
        for path in glob.glob("results/formal_p3/*.json"):
            try:
                data = json.load(open(path))
                for r in data:
                    if isinstance(r, dict):
                        rows.append(r)
            except Exception:
                pass
        return rows

    _p3_raw = _load_p3_json()

    def _p3_asr(model_id, domain=None, attack_id=None):
        vals = [
            bool(r.get("success")) for r in _p3_raw
            if r.get("model") == model_id
            and r.get("score","") not in _EXCL_P3
            and (domain is None or r.get("domain") == domain)
            and (attack_id is None or r.get("attack_id") == attack_id)
        ]
        if not vals: return 0.0, 0.0
        arr = np.array(vals, dtype=float)
        return round(float(arr.mean()*100), 1), round(float(arr.std(ddof=1)/np.sqrt(len(arr))*100) if len(arr)>1 else 0.0, 1)

    _p3_results_tab, _p3_replay_tab = st.tabs(["Comparison Results", "Payload Replay"])

    # ── Comparison Results ──────────────────────────────────────────────────────
    with _p3_results_tab:
        if not _p3_raw:
            st.info("No Phase III formal data yet. Run: `python -m src.harness.run_experiments --phase p3 --models <model_id>`")
        else:
            # Summary metrics
            _p3_mc1, _p3_mc2, _p3_mc3, _p3_mc4 = st.columns(4)
            _p3_valid = [r for r in _p3_raw if r.get("score","") not in _EXCL_P3]
            _p3_succ  = sum(1 for r in _p3_valid if r.get("success"))
            _p3_overall_asr = round(_p3_succ / len(_p3_valid) * 100, 1) if _p3_valid else 0.0
            _p3_run_asrs = [_p3_asr(q["model_id"])[0] for q in QWEN_FAMILY
                            if any(r.get("model") == q["model_id"] for r in _p3_raw)]
            _p3_mc1.metric("Models Tested",
                           len([q for q in QWEN_FAMILY if any(r.get("model") == q["model_id"] for r in _p3_raw)]))
            _p3_mc2.metric("Overall ASR", f"{_p3_overall_asr:.0f}%")
            _p3_mc3.metric("Best ASR (most robust)", f"{min(_p3_run_asrs):.0f}%" if _p3_run_asrs else "—")
            _p3_mc4.metric("Worst ASR (most vulnerable)", f"{max(_p3_run_asrs):.0f}%" if _p3_run_asrs else "—")

            st.divider()

            # Model info table
            st.markdown("##### Model Overview")
            _info_rows = [{
                "Model": q["label"], "Parameters": q["params_total"],
                "Architecture": q["arch"], "Quantization": q["quant"],
            } for q in QWEN_FAMILY]
            st.dataframe(pd.DataFrame(_info_rows), use_container_width=True, hide_index=True)

            st.divider()

            # Trend line — key chart
            st.markdown("##### Does Parameter Count Reduce Injection Vulnerability?")
            _param_lbls    = [q["short"] for q in QWEN_FAMILY]
            _param_overall = [_p3_asr(q["model_id"])[0] for q in QWEN_FAMILY]
            _param_cooking = [_p3_asr(q["model_id"], domain="cooking")[0] for q in QWEN_FAMILY]
            _param_health  = [_p3_asr(q["model_id"], domain="health")[0] for q in QWEN_FAMILY]

            _trend_fig = go.Figure()
            _trend_fig.add_trace(go.Scatter(
                x=_param_lbls, y=_param_overall, mode="lines+markers", name="Overall",
                line=dict(color="#60a5fa", width=3), marker=dict(size=10, color="#60a5fa"),
            ))
            _trend_fig.add_trace(go.Scatter(
                x=_param_lbls, y=_param_health, mode="lines+markers", name="Health",
                line=dict(color="#ef233c", width=2, dash="dot"), marker=dict(size=8, color="#ef233c"),
            ))
            _trend_fig.add_trace(go.Scatter(
                x=_param_lbls, y=_param_cooking, mode="lines+markers", name="Cooking",
                line=dict(color="#2dc653", width=2, dash="dot"), marker=dict(size=8, color="#2dc653"),
            ))
            for _ql, _yv in zip(_param_lbls, _param_overall):
                _trend_fig.add_annotation(
                    x=_ql, y=_yv, text=f"{_yv:.0f}%",
                    showarrow=False, yshift=14, font=dict(color="#fff", size=12),
                )
            _trend_fig.update_layout(
                height=400, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#fff",
                yaxis=dict(title="ASR (%)", range=[0, 115], gridcolor="#222"),
                xaxis=dict(title="Parameter Count"),
                legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(_trend_fig, use_container_width=True)
            st.caption(
                "Overall ASR declines from 91.7% (0.8B) to 46.4% (27B), but flattens at ≥4B. "
                "Health domain stays at 93–100% regardless of scale. "
                "Cooking drops to 0% at 4B+, suggesting a threshold effect rather than smooth scaling."
            )

            st.divider()

            # Two-column: grouped bar + attack heatmap
            _p3c1, _p3c2 = st.columns(2)

            with _p3c1:
                st.markdown("##### ASR by Domain")
                _dom_bar_fig = go.Figure()
                for q in QWEN_FAMILY:
                    _ys, _errs = [], []
                    for _dom in [None, "cooking", "health"]:
                        m, s = _p3_asr(q["model_id"], domain=_dom)
                        _ys.append(m); _errs.append(s * 2)
                _dom_bar_fig = go.Figure()
                for q in QWEN_FAMILY:
                    _ys, _errs = [], []
                    for _dom in [None, "cooking", "health"]:
                        m, s = _p3_asr(q["model_id"], domain=_dom)
                        _ys.append(m); _errs.append(s * 2)
                    _dom_bar_fig.add_trace(go.Bar(
                        name=q["short"], x=["Overall","Cooking","Health"], y=_ys,
                        error_y=dict(type="data", array=_errs, visible=True,
                                     color="#ffffff", thickness=1),
                        marker_color=q["color"], marker_line_color="#333", marker_line_width=1,
                    ))
                _dom_bar_fig.update_layout(
                    barmode="group", height=360,
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#fff",
                    yaxis=dict(title="ASR (%)", range=[0, 120], gridcolor="#222"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(_dom_bar_fig, use_container_width=True)

            with _p3c2:
                st.markdown("##### Per-Attack ASR Heatmap")
                _atk_ids  = ["attack2_roleplay", "attack1_naive", "attack3_fake_completion"]
                _atk_lbls = ["Role-play/DAN", "Naive Injection", "Fake Completion"]
                _z_p3, _txt_p3 = [], []
                for q in QWEN_FAMILY:
                    row, txt = [], []
                    for aid in _atk_ids:
                        m, _ = _p3_asr(q["model_id"], attack_id=aid)
                        row.append(m); txt.append(f"{m:.0f}%")
                    _z_p3.append(row); _txt_p3.append(txt)
                _heat_fig = go.Figure(go.Heatmap(
                    z=_z_p3, x=_atk_lbls, y=[q["short"] for q in QWEN_FAMILY],
                    text=_txt_p3, texttemplate="%{text}",
                    textfont={"size": 14, "color": "white"},
                    colorscale=[[0,"#1a4a1a"],[0.5,"#7a7a1a"],[1,"#7a1a1a"]],
                    zmin=0, zmax=100, showscale=True,
                ))
                _heat_fig.update_layout(
                    height=360, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#fff",
                    xaxis=dict(tickangle=-15, tickfont=dict(size=12)),
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(_heat_fig, use_container_width=True)

            st.divider()

            # Summary table
            st.markdown("##### Summary Table")
            _sum_rows = []
            for q in QWEN_FAMILY:
                p_overall, p_sem = _p3_asr(q["model_id"])
                p_cooking, _     = _p3_asr(q["model_id"], domain="cooking")
                p_health, _      = _p3_asr(q["model_id"], domain="health")
                p_roleplay, _    = _p3_asr(q["model_id"], attack_id="attack2_roleplay")
                p_naive, _       = _p3_asr(q["model_id"], attack_id="attack1_naive")
                p_fake, _        = _p3_asr(q["model_id"], attack_id="attack3_fake_completion")
                _sum_rows.append({
                    "Model":           q["label"],
                    "Overall ASR":     f"{p_overall:.0f}% ±{p_sem:.0f}",
                    "Cooking":         f"{p_cooking:.0f}%",
                    "Health":          f"{p_health:.0f}%",
                    "Role-play/DAN":   f"{p_roleplay:.0f}%",
                    "Naive Injection": f"{p_naive:.0f}%",
                    "Fake Completion": f"{p_fake:.0f}%",
                })
            st.dataframe(pd.DataFrame(_sum_rows), use_container_width=True, hide_index=True)
            st.caption(
                "**Key finding:** ASR drops from 91.7% (0.8B) to 46.4% (27B) overall. "
                "Health domain remains 93–100% across all sizes. "
                "Cooking drops to 0% at 4B+. Parameter scale partially mitigates but does not eliminate injection vulnerability."
            )

    # ── Payload Replay ──────────────────────────────────────────────────────────
    with _p3_replay_tab:
        st.caption(
            "Browsing Phase III runs from results/formal_p3/ — no API calls. "
            "Shows how each Qwen 3.5 variant responded to the top-3 Phase I attack payloads."
        )

        if not _p3_raw:
            st.info("No Phase III data yet.")
        else:
            _rfc1, _rfc2, _rfc3 = st.columns(3)
            with _rfc1:
                _p3_m_opts = ["All"] + [q["label"] for q in QWEN_FAMILY]
                _p3_sel_m  = st.selectbox("Model", _p3_m_opts, key="p3_rp_m")
            with _rfc2:
                _p3_d_opts = ["All"] + sorted({
                    DOMAIN_LABELS.get(r.get("domain",""), r.get("domain","")) for r in _p3_raw
                })
                _p3_sel_d  = st.selectbox("Domain", _p3_d_opts, key="p3_rp_d")
            with _rfc3:
                _p3_a_opts = ["All"] + sorted({
                    r.get("attack_name", r.get("attack_id","")) for r in _p3_raw
                })
                _p3_sel_a  = st.selectbox("Attack Type", _p3_a_opts, key="p3_rp_a")

            _qwen_id_by_label = {q["label"]: q["model_id"] for q in QWEN_FAMILY}
            _p3_rp_filt = [
                r for r in sorted(_p3_raw, key=lambda x: (
                    x.get("attack_name", x.get("attack_id","")),
                    x.get("domain",""), x.get("model",""), x.get("rep", 0),
                ))
                if (_p3_sel_m == "All" or r.get("model","") == _qwen_id_by_label.get(_p3_sel_m, ""))
                and (_p3_sel_d == "All" or DOMAIN_LABELS.get(r.get("domain",""), r.get("domain","")) == _p3_sel_d)
                and (_p3_sel_a == "All" or r.get("attack_name", r.get("attack_id","")) == _p3_sel_a)
                and r.get("score","") != "ERROR"
            ]

            st.markdown(f"**{len(_p3_rp_filt)} run(s) matching filters**")
            if not _p3_rp_filt:
                st.warning("No runs match the selected filters.")
            else:
                _p3_pp = st.select_slider("Runs per page", [5, 10, 25, 50], value=10, key="p3_rp_pp")
                _p3_tp = max(1, (len(_p3_rp_filt) + _p3_pp - 1) // _p3_pp)
                _p3_pg = st.number_input("Page", min_value=1, max_value=_p3_tp, value=1, key="p3_rp_pg") - 1

                for _pr in _p3_rp_filt[_p3_pg * _p3_pp: _p3_pg * _p3_pp + _p3_pp]:
                    _pr_ml  = MODEL_ID_TO_DISPLAY.get(_pr.get("model",""), _pr.get("model",""))
                    _pr_dl  = DOMAIN_LABELS.get(_pr.get("domain",""), _pr.get("domain",""))
                    _pr_an  = _pr.get("attack_name", _pr.get("attack_id",""))
                    _pr_sc  = _pr.get("score","")
                    _pr_ico = "✅" if _pr_sc == "SUCCESS" else (
                        "⚠️" if _pr_sc in ("AMBIGUOUS","NO_RESPONSE","CONFOUND") else "🛡️"
                    )

                    with st.expander(
                        f"{_pr_ico} **{_pr_ml}** · {_pr_dl} · {_pr_an} · Rep {_pr.get('rep', 0) + 1} · {_pr_sc}",
                        expanded=False,
                    ):
                        _pr_spr     = SYSTEM_PROMPTS.get(_pr.get("domain",""), "")
                        _pr_payload = _pr.get("payload","")
                        _pr_resp    = _pr.get("response","")

                        c1, c2 = st.columns(2)
                        with c1:
                            _panel(f"🤖 AGENT PROMPT · {_pr_dl}", _pr_spr, "#0e3a4a", "#00b4d8", "#e0f7fa")
                            _panel(f"⚔️ PAYLOAD · {_pr_an}", _pr_payload, "#4a0e0e", "#ef233c", "#ffe0e0")
                        with c2:
                            _panel(f"💬 {_pr_ml} RESPONSE", _pr_resp, "#0e4a1a", "#2dc653", "#e0ffe6")
                            if _pr_sc in ("AMBIGUOUS","NO_RESPONSE","CONFOUND"):
                                st.warning(f"{_pr_sc} — excluded from ASR")
                            elif _pr_sc == "SUCCESS":
                                st.success("SUCCESS — Model broke its directive")
                            else:
                                st.error("FAILURE — Model maintained its directive")


# ═══════════════════════════════════════════════════════════
# PHASE IV — LoRA Fine-Tuning
# ═══════════════════════════════════════════════════════════
with tab_p4:
    st.markdown("#### Phase IV — LoRA Fine-Tuning (Llama 3.1 8B)")
    st.caption("168 refusal examples · LoRA r=16 · 3 epochs · RTX 3060 Ti · ~6 min training")

    _p4_results_tab, _p4_replay_tab = st.tabs(["Fine-Tuning Results", "Payload Replay"])

    # ── Fine-Tuning Results ─────────────────────────────────────────────────────
    with _p4_results_tab:
        _ftp1, _ftp2, _ftp3 = st.columns(3)
        _ftp1.metric("Before ASR", "79.6%")
        _ftp2.metric("After ASR",  "0%", delta="-79.6pp", delta_color="inverse")
        _ftp3.metric("Relative reduction", "100%", delta_color="inverse")

        _ft_atk_labels = ["Naive\nInjection","Role-play\n/ DAN","Fake\nCompletion","Sys Prompt\nExtraction","Base64\nEncoding"]
        _ft_b = [41.4, 50.9, 39.0, 28.8, 29.3]
        _ft_a = [0.0,  0.0,  0.0,  0.0,  0.0]
        _fig_p4ft = go.Figure()
        _fig_p4ft.add_bar(name="Before (baseline)", x=_ft_atk_labels, y=_ft_b, marker_color="#d62728")
        _fig_p4ft.add_bar(name="After LoRA SFT",    x=_ft_atk_labels, y=_ft_a, marker_color="#2ca02c")
        _fig_p4ft.update_layout(
            barmode="group",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#ffffff",
            height=400,
            yaxis=dict(title="ASR (%)", range=[0,95], gridcolor="#333", color="#ffffff"),
            xaxis=dict(title="Attack Type", color="#ffffff"),
            legend=dict(orientation="h", y=1.12, font_color="#ffffff"),
            title=dict(text="Llama 3.1 8B: ASR Before vs After LoRA Fine-Tuning", font_color="#ffffff"),
        )
        st.plotly_chart(_fig_p4ft, use_container_width=True)

        _domain_col1, _domain_col2 = st.columns(2)
        with _domain_col1:
            st.metric("Health domain — Before", "60%")
            st.metric("Health domain — After",  "0%",  delta="-60pp", delta_color="inverse")
        with _domain_col2:
            st.metric("Cooking domain — Before", "18%")
            st.metric("Cooking domain — After",  "0%", delta="-18pp", delta_color="inverse")

        with st.expander("Methodology & caveats"):
            st.markdown("""
- **Training data**: 168 (attack payload, refusal) pairs generated by Claude Sonnet 4.6 across 5 attacks × 2 domains × 20 reps (32 compliance-filtered)
- **Fine-tuning**: Unsloth LoRA SFT, r=16, α=32, 3 epochs, lr=2e-4, effective batch 8
- **Evaluation**: identical Phase I protocol — 5 attacks × 2 domains × 5 reps (50 trials, rubric v2)
- **Scores revised** after manual review: 16 records corrected (AMBIGUOUS/SUCCESS → FAILURE) across Naive Injection, Role-play / DAN, and Sys Prompt Extraction
- **Base64 silent refusal**: `finish_reason=stop`, `completion_tokens=1` — model generates EOS only; confirmed not a token-limit issue via direct API inspection
            """)

    # ── Payload Replay ──────────────────────────────────────────────────────────
    with _p4_replay_tab:
        _FT_MODEL_ID       = "llama-3.1-8b-injectionguard"
        _FT_BASELINE_ID    = "meta-llama-3.1-8b-instruct"
        _FT_MODEL_LABEL    = "Llama 3.1 8B (fine-tuned)"
        _FT_BASELINE_LABEL = "Llama 3.1 8B (baseline)"

        st.caption(
            "Browsing fine-tuned model runs from results/formal_v2/ — no API calls.  "
            "Left column shows the same Phase I baseline payload; right column shows the "
            "fine-tuned model's response, optionally compared to the pre-FT baseline."
        )

        _p4_show_cmp = st.checkbox(
            "Show pre-FT baseline (Llama 3.1 8B) response for comparison",
            value=True, key="p4_rp_cmp",
        )

        # Load fine-tuned model records
        _ft_recs_raw = [
            r for r in load_formal_records()
            if r.get("model","") == _FT_MODEL_ID
        ]

        # Pre-build baseline lookup keyed by (attack_id, domain)
        _ft_base_by_key: dict[tuple, list[dict]] = {}
        for _r in load_formal_records():
            if _r.get("model","") != _FT_BASELINE_ID:
                continue
            _k = (_r.get("attack_id",""), _r.get("domain",""))
            _ft_base_by_key.setdefault(_k, []).append(_r)

        if not _ft_recs_raw:
            st.info(
                "No fine-tuned model records found in results/formal_v2/.  "
                "Run: `python -m src.main --phase p4` with model `llama-3.1-8b-injectionguard`"
            )
        else:
            # Filter controls
            _p4fc1, _p4fc2 = st.columns(2)
            with _p4fc1:
                _p4_d_opts = ["All"] + sorted({
                    DOMAIN_LABELS.get(r.get("domain",""), r.get("domain",""))
                    for r in _ft_recs_raw
                })
                _p4_sel_d = st.selectbox("Domain", _p4_d_opts, key="p4_rp_d")
            with _p4fc2:
                _p4_a_opts = ["All"] + sorted({
                    r.get("attack_name", r.get("attack_id","")) for r in _ft_recs_raw
                })
                _p4_sel_a = st.selectbox("Attack Type", _p4_a_opts, key="p4_rp_a")

            _ft_filt = sorted(
                [
                    r for r in _ft_recs_raw
                    if (_p4_sel_d == "All" or DOMAIN_LABELS.get(r.get("domain",""), r.get("domain","")) == _p4_sel_d)
                    and (_p4_sel_a == "All" or r.get("attack_name", r.get("attack_id","")) == _p4_sel_a)
                ],
                key=lambda x: (x.get("attack_name", x.get("attack_id","")), x.get("domain",""), x.get("rep",0)),
            )

            st.markdown(f"**{len(_ft_filt)} run(s) matching filters**")
            if not _ft_filt:
                st.warning("No runs match the selected filters.")
            else:
                _p4_pp = st.select_slider("Runs per page", [5, 10, 25, 50], value=10, key="p4_rp_pp")
                _p4_tp = max(1, (len(_ft_filt) + _p4_pp - 1) // _p4_pp)
                _p4_pg = st.number_input("Page", min_value=1, max_value=_p4_tp, value=1, key="p4_rp_pg") - 1

                for _fr in _ft_filt[_p4_pg * _p4_pp: _p4_pg * _p4_pp + _p4_pp]:
                    _fr_dl  = DOMAIN_LABELS.get(_fr.get("domain",""), _fr.get("domain",""))
                    _fr_an  = _fr.get("attack_name", _fr.get("attack_id",""))
                    _fr_sc  = _fr.get("score","")
                    _fr_ico = "✅" if _fr_sc == "SUCCESS" else (
                        "⚠️" if _fr_sc in ("AMBIGUOUS","NO_RESPONSE","CONFOUND") else "🛡️"
                    )

                    with st.expander(
                        f"{_fr_ico} **{_FT_MODEL_LABEL}** · {_fr_dl} · {_fr_an} · Rep {_fr.get('rep',0)+1} · {_fr_sc}",
                        expanded=False,
                    ):
                        _fr_spr     = SYSTEM_PROMPTS.get(_fr.get("domain",""), "")
                        _fr_payload = _fr.get("payload","")
                        _fr_resp    = _fr.get("response","")
                        _fr_key     = (_fr.get("attack_id",""), _fr.get("domain",""))
                        _fr_cmp_recs = _ft_base_by_key.get(_fr_key, []) if _p4_show_cmp else []

                        c1, c2 = st.columns(2)
                        with c1:
                            _panel(
                                f"🤖 AGENT PROMPT · {_fr_dl}",
                                _fr_spr, "#0e3a4a", "#00b4d8", "#e0f7fa",
                            )
                            _panel(
                                f"⚔️ BASELINE PAYLOAD · {_fr_an}",
                                _fr_payload, "#4a0e0e", "#ef233c", "#ffe0e0",
                            )
                        with c2:
                            _panel(
                                f"💬 {_FT_MODEL_LABEL} RESPONSE",
                                _fr_resp, "#0e4a1a", "#2dc653", "#e0ffe6",
                            )
                            if _fr_sc in ("AMBIGUOUS","NO_RESPONSE","CONFOUND"):
                                st.warning(f"{_fr_sc} — excluded from ASR")
                            elif _fr_sc == "SUCCESS":
                                st.success("SUCCESS — Fine-tuned model broke its directive")
                            else:
                                st.error("FAILURE — Fine-tuned model maintained its directive")

                            if _fr_cmp_recs:
                                st.markdown("---")
                                st.caption(f"Pre-FT baseline comparison ({_FT_BASELINE_LABEL}, same attack + domain)")
                                for _cmp in _fr_cmp_recs[:3]:
                                    _cmp_sc  = _cmp.get("score","")
                                    _cmp_ico = "✅" if _cmp_sc == "SUCCESS" else (
                                        "⚠️" if _cmp_sc in ("AMBIGUOUS","NO_RESPONSE","CONFOUND") else "🛡️"
                                    )
                                    _panel(
                                        f"{_cmp_ico} PRE-FT BASELINE · Rep {_cmp.get('rep',0)+1} · {_cmp_sc}",
                                        _cmp.get("response",""), "#1a1a3a", "#f4a261", "#ffe0b2",
                                    )
                            elif _p4_show_cmp:
                                st.caption("No pre-FT baseline records for this attack + domain.")


# ═══════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("#### Overview — All Phases")

    _ov_asr = compute_asr(df_all)

    # Metrics row
    ov_mc1, ov_mc2, ov_mc3, ov_mc4 = st.columns(4)
    if not _ov_asr.empty:
        _ov_models   = _ov_asr["model"].nunique()
        _ov_avg_asr  = round(_ov_asr["asr"].mean(), 1)
        _ov_best_mdl = _ov_asr.groupby("model_display")["asr"].mean().idxmin()
        _ov_best_asr = round(_ov_asr.groupby("model_display")["asr"].mean().min(), 1)
        _phases_done = sum([
            not df_all[df_all["domain"].isin(["cooking","health"])].empty,
            not df_all[df_all["domain"].isin(["compliant_assistant","truth_teller"])].empty,
            not df_formal_p2a.empty,
        ])
    else:
        _ov_models = _ov_avg_asr = _ov_best_asr = 0
        _ov_best_mdl = "—"
        _phases_done = 0
    with ov_mc1: st.metric("Models Tested",    _ov_models)
    with ov_mc2: st.metric("Phases With Data", f"{_phases_done}/5")
    with ov_mc3: st.metric("Overall Avg ASR",  f"{_ov_avg_asr}%" if _ov_avg_asr else "—")
    with ov_mc4: st.metric("Most Robust Model", _ov_best_mdl,
                            delta=f"{_ov_best_asr}% ASR" if _ov_best_asr else None)

    st.divider()

    ov_t1, ov_t2, ov_t3, ov_t4 = st.tabs([
        "Summary Table", "Model Comparison", "Attack Heatmap", "Timeline",
    ])

    with ov_t1:
        _ov_p1  = _ov_asr[_ov_asr["attack_id"].isin(P1_ATTACK_IDS)  & _ov_asr["domain"].isin(P1_DOMAINS)].copy()
        _ov_p2a = _ov_asr[_ov_asr["attack_id"].isin(P2A_ATTACK_IDS) & _ov_asr["domain"].isin(P2A_DOMAINS)].copy()

        def _build_display(sub: pd.DataFrame, phase_label: str) -> pd.DataFrame:
            sub = sub.copy()
            sub["Phase"]  = phase_label
            sub["Attack"] = sub["attack_id"].map(CANONICAL_ATTACK_NAME).fillna(sub["attack_name"])
            sub["Domain"] = sub["domain"].map(DOMAIN_LABELS).fillna(sub["domain"])
            sub["Model"]  = sub["model_display"]
            sub["✓ Hits"] = sub["runs_success"].astype(str) + " / " + sub["runs_total"].astype(str)
            sub["ASR"]    = sub["asr"].apply(lambda v: f"{v}%")
            sub["_ord"]   = sub["attack_id"].map(ATTACK_ORDER).fillna(99)
            return (sub[["Phase","Attack","Domain","Model","✓ Hits","ASR","_ord"]]
                    .sort_values(["_ord","Domain","Model"]).drop(columns=["_ord"]))

        def _color_asr(val):
            try:
                v = float(str(val).rstrip("%"))
            except (TypeError, ValueError):
                return ""
            if v >= 67: return "background-color:#4a0e0e; color:#ef233c; font-weight:bold"
            if v >= 34: return "background-color:#4a3b0e; color:#f4a261; font-weight:bold"
            return "background-color:#0e4a1a; color:#2dc653; font-weight:bold"

        def _color_phase(val):
            return "color:#60a5fa; font-weight:bold" if val == "Phase I" else "color:#a78bfa; font-weight:bold"

        _ov_frames = []
        if not _ov_p1.empty:  _ov_frames.append(_build_display(_ov_p1,  "Phase I"))
        if not _ov_p2a.empty: _ov_frames.append(_build_display(_ov_p2a, "Phase IIA"))
        if not _ov_frames:
            st.info("No formal results yet.")
        else:
            _ov_display = pd.concat(_ov_frames, ignore_index=True)
            st.dataframe(
                _ov_display.style.map(_color_phase, subset=["Phase"]).map(_color_asr, subset=["ASR"]),
                use_container_width=True, hide_index=True,
            )
            st.download_button("Download CSV", _ov_display.to_csv(index=False).encode(),
                               "asr_summary_all.csv", "text/csv")
            st.caption("🔴 ≥67%   |   🟡 34–66%   |   🟢 <34%   |   ✓ Hits = successes / non-ambiguous runs")

    with ov_t2:
        if _ov_asr.empty:
            st.info("No results yet.")
        else:
            _ov_clean  = df_all[~df_all["score"].isin(_excl_scores)] if not df_all.empty else df_all
            _ov_p1_raw = _ov_clean[_ov_clean["domain"].isin(["cooking","health"])]           if not _ov_clean.empty else _ov_clean
            _ov_p2a_raw= _ov_clean[_ov_clean["domain"].isin(["compliant_assistant","truth_teller"])] if not _ov_clean.empty else _ov_clean
            _ov_p2b_raw= _ov_clean[_ov_clean["source"] == "formal_p2a"]                      if not _ov_clean.empty else _ov_clean

            _ov_p1_st  = _model_asr_sem(_ov_p1_raw)
            _ov_p2a_st = _model_asr_sem(_ov_p2a_raw)
            _ov_p2b_st = _model_asr_sem(_ov_p2b_raw)
            _ov_mdls   = sorted(set(list(_ov_p1_st)+list(_ov_p2b_st)))
            if not _ov_mdls:
                st.info("No results yet.")
            else:
                fig_ov_mc = go.Figure()
                for _sn, _st, _col in [
                    ("Phase I — Baseline",       _ov_p1_st,  "#00b4d8"),
                    ("Phase IIA — Multilingual", _ov_p2b_st, "#ef233c"),
                ]:
                    if not _st: continue
                    fig_ov_mc.add_trace(go.Bar(
                        name=_sn, x=_ov_mdls,
                        y=[_st.get(m,(None,0))[0] for m in _ov_mdls],
                        error_y=dict(type="data", array=[_st.get(m,(0,0))[1]*2 for m in _ov_mdls],
                                     visible=True, color="#ffffff", thickness=1.5),
                        marker_color=_col,
                        text=[f"{int(_st[m][0])}%" if m in _st else "N/A" for m in _ov_mdls],
                        textposition="outside",
                    ))
                fig_ov_mc.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#ffffff",
                    yaxis=dict(range=[0, 130], title="Avg ASR %", gridcolor="#222"),
                    xaxis=dict(title="Model"),
                    barmode="group", height=480,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_ov_mc, use_container_width=True)
                st.caption("Error bars = ±2 SEM (95% CI). N/A = not yet tested.")
                _png_download_btn("Download as PNG", fig_ov_mc, "overview_model_comparison.png", height=540)

    with ov_t3:
        _render_heatmap(compute_asr(df_all), key_sfx="overview")

    with ov_t4:
        def _phase_pct(domain_list: list[str]) -> int:
            if df_all.empty: return 0
            _s = df_all[df_all["domain"].isin(domain_list)]
            return 0 if _s.empty else min(100, round(_s["model"].nunique() / len(config.ALL_MODELS) * 100))
        _p2b_pct = 0 if df_formal_p2a.empty else min(
            100, round(df_formal_p2a["model"].nunique() / len(config.ALL_MODELS) * 100)
        )
        _progress = [
            _phase_pct(["cooking","health"]),
            _p2b_pct, 100, 100, 100,
        ]
        fig_tl = go.Figure(go.Bar(
            x=["Phase I — Baseline","Phase IIA — Multilingual",
               "Phase IIB — LLM Attacker","Phase III — Qwen Family","Phase IV — Fine-Tuning"],
            y=_progress,
            marker_color=["#2dc653" if p==100 else ("#f4a261" if p>0 else "#666666") for p in _progress],
            text=[f"{v}%" for v in _progress], textposition="outside",
        ))
        fig_tl.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#ffffff",
            yaxis=dict(range=[0,120], title="Models with data (%)", gridcolor="#222"),
            xaxis=dict(title="Research Phase"),
            showlegend=False, height=360,
        )
        st.plotly_chart(fig_tl, use_container_width=True)
        st.caption("Green = all 6 models run   |   Yellow = partial   |   Gray = not started")
        _png_download_btn("Download timeline as PNG", fig_tl, "timeline.png", height=420)

# ── Footer ────────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align:center; color:#666; font-size:12px;'>"
    "Capstone Research Framework · Spring 2026 · Prompt Injection Robustness Testing"
    "</div>",
    unsafe_allow_html=True,
)
