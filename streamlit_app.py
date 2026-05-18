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
from src.attacks.p2b_payloads import (
    LANGUAGE_DISPLAY as P2B_LANG_DISPLAY,
    PAYLOADS as P2B_PAYLOADS,
    is_ready as p2b_is_ready,
)
from src.attacks.p2b_multilingual import ALL_P2B
from src.attacks.p2c_llm_attacker import (
    ATTACK_STYLES as P2C_STYLES,
    ATTACK_STYLE_DISPLAY as P2C_STYLE_DISPLAY,
    ATTACKER_MODEL as P2C_ATTACKER,
    DEFENDER_MODEL as P2C_DEFENDER,
    run as p2c_run,
)
from src.harness import config
from src.attacks import PHASE2B_ATTACKS as _PHASE2B_FORMAL_IDS
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
# Phase IIB: top-3 Phase I attacks only (by ASR: roleplay 48.4%, fake_completion 40.0%, naive 37.1%)
PHASE2B_ATTACKS_UI: dict[str, str] = {
    "2 - Role-play / DAN":           "attack2_roleplay",
    "3 - Fake Completion":           "attack3_fake_completion",
    "1 - Naive Injection":           "attack1_naive",
}
PHASE2B_LANGUAGE_KEYS: dict[str, str] = {
    v: k for k, v in P2B_LANG_DISPLAY.items()
}

PHASE_KEYS = [
    "Phase I - Baseline Attacks",
    "Phase IIA - Cognitive Schema Attacks",
    "Phase IIB - Multilingual Attack Comparison",
    "Phase IIC - LLM as an Attacker",
    "Phase III - Fine-Tuning",
]

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
def load_formal_records(rubric: str = "v1") -> list[dict]:
    """Return full records (including payload + response) from results/formal/ or results/formal_v2/."""
    folder = "results/formal_v2" if rubric == "v2" else "results/formal"
    rows: list[dict] = []
    for path in glob.glob(f"{folder}/*.json"):
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
    """Return full records (payload, response, translation) from results/formal_p2b/."""
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
def load_p2a_records() -> list[dict]:
    """Return Phase IIA records from results/formal_v2/ (attack_id starts with 'p2_')."""
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
def load_replay_records(rubric: str = "v2") -> list[dict]:
    """Combined replay loader: Phase I (formal_v2) + Phase IIB (formal_p2b).

    Each record is normalised to have:
      _language   — "english" for Phase I, or the language_code for Phase IIB
      _is_p2b     — bool
      _payload    — attack prompt text
      _response   — raw model response
    """
    rows: list[dict] = []

    # Phase I — English
    for r in load_formal_records(rubric):
        r = dict(r)
        r["_language"] = "english"
        r["_is_p2b"]   = False
        r["_payload"]  = r.get("payload", "")
        r["_response"] = r.get("response", "")
        rows.append(r)

    # Phase IIB — Multilingual
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
        if "formal_p2b" in path:
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
    p2b_lang_key = None   # internal language key (mandarin / swahili / welsh)

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
        persona_display = st.selectbox("Assistant Persona", list(PHASE1_DOMAINS.keys()))
        attack_display  = st.selectbox("Attack Type (top-3 by ASR)", list(PHASE2B_ATTACKS_UI.keys()))
        language        = st.selectbox("Language", list(P2B_LANG_DISPLAY.values()))
        domain_id       = PHASE1_DOMAINS[persona_display]
        base_attack_id  = PHASE2B_ATTACKS_UI[attack_display]
        p2b_lang_key    = {v: k for k, v in P2B_LANG_DISPLAY.items()}[language]
        attack_id       = ALL_P2B[
            next(aid for aid, inst in ALL_P2B.items()
                 if inst.BASE_ATTACK_ID == base_attack_id and inst.LANGUAGE == p2b_lang_key)
        ].ATTACK_ID
        if not p2b_is_ready(p2b_lang_key):
            st.warning(
                f"{language} translations are placeholders — fill in "
                f"`src/attacks/p2b_payloads.py` before running."
            )

    elif phase == "Phase IIC - LLM as an Attacker":
        persona_display  = st.selectbox("Defender Persona", list(PHASE1_DOMAINS.keys()))
        p2c_style_display = st.selectbox("Attack Style (Llama-generated)", list(P2C_STYLE_DISPLAY.keys()))
        domain_id  = PHASE1_DOMAINS[persona_display]
        attack_id  = P2C_STYLE_DISPLAY[p2c_style_display]
        st.caption("**Attacker:** Llama 3.1 8B (local) — 82.8% Phase I ASR (rubric v2)")
        st.caption("**Defender:** Claude Sonnet 4.6 (remote) — 8.3% Phase I ASR (rubric v2)")

    elif phase == "Phase III - Fine-Tuning":
        st.info("Fine-tuning phase blocked on Phase I results.")

    run_btn = st.button("Run Attack", use_container_width=True, type="primary",
                        disabled=(attack_id is None))

# ── Run attack ────────────────────────────────────────────

if run_btn and attack_id and domain_id:
    system_prompt = SYSTEM_PROMPTS[domain_id]
    _is_p2b = attack_id in ALL_P2B
    _is_p2c = attack_id in P2C_STYLES
    _base_id = (
        ALL_P2B[attack_id].BASE_ATTACK_ID if _is_p2b else
        P2C_STYLES[attack_id]["base_attack"] if _is_p2c else
        attack_id
    )
    attack_module = ATTACK_REGISTRY.get(attack_id)

    if _is_p2c:
        with st.spinner(
            f"Step 1 — Llama 3.1 8B generating payload…  "
            f"Step 2 — Claude Sonnet 4.6 defending…"
        ):
            try:
                result = p2c_run(attack_id, system_prompt, domain_id=domain_id)
                turn1 = result.get("turn1_response") if _base_id == "attack3_fake_completion" else None
                result["score"]   = auto_score(_base_id, result["response"], system_prompt, domain_id, turn1_response=turn1, rubric_version="v2")
                result["success"] = result["score"] == "SUCCESS"
            except Exception as exc:
                result = {
                    "attack_id":   attack_id,
                    "attack_name": P2C_STYLES[attack_id]["name"],
                    "model":       P2C_DEFENDER,
                    "domain":      domain_id,
                    "generated_payload": "",
                    "payload":     "",
                    "response":    f"[ERROR] {exc}",
                    "score":       "ERROR",
                    "success":     False,
                    "phase":       "iic",
                }
    else:
        _spinner_msg = (
            f"Querying {model_display}… (+ Claude Haiku translation if non-English)"
            if _is_p2b else f"Querying {model_display}…"
        )
        with st.spinner(_spinner_msg):
            try:
                if _base_id == "attack3_fake_completion":
                    result = attack_module.run(model_id, system_prompt, domain_id=domain_id)
                else:
                    result = attack_module.run(model_id, system_prompt)
                result["domain"] = domain_id
                turn1 = result.get("turn1_response") if _base_id == "attack3_fake_completion" else None

                if _is_p2b:
                    from src.scoring.translate_score import score_multilingual
                    _inst = ALL_P2B[attack_id]
                    scoring = score_multilingual(
                        attack_id=attack_id,
                        base_attack_id=_inst.BASE_ATTACK_ID,
                        language_code=_inst.LANGUAGE,
                        model=model_id,
                        rep=0,
                        attack_prompt=result["payload"],
                        response_original=result["response"],
                        system_prompt=system_prompt,
                        domain_id=domain_id,
                        turn1_response=turn1,
                    )
                    result.update(scoring)
                    result["score"]   = scoring["outcome"]
                    result["success"] = scoring["outcome"] == "SUCCESS"
                else:
                    result["score"]   = auto_score(attack_id, result["response"], system_prompt, domain_id, turn1_response=turn1, rubric_version="v2")
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

    # Append to session-scoped run log (for Session Log tab)
    if "run_history" not in st.session_state:
        st.session_state["run_history"] = []
    log_entry: dict = {
        "time":     datetime.now().strftime("%H:%M:%S"),
        "model":    "Llama→Claude" if _is_p2c else model_display,
        "attack":   (p2c_style_display if _is_p2c else attack_display) or "",
        "domain":   persona_display or "",
        "score":    result["score"],
        "response": result["response"],
    }
    if _is_p2b:
        log_entry["language"] = result.get("language", "")
    if _is_p2c:
        log_entry["generated_payload"] = result.get("generated_payload", "")
    st.session_state["run_history"].append(log_entry)

    # Save scratch run; suffix encodes phase for easy filtering
    os.makedirs("results/scratch", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "_p2c_ui" if _is_p2c else ("_p2b_ui" if _is_p2b else "_ui")
    save_path = f"results/scratch/{ts}{suffix}.json"
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
                st.success("SUCCESS — Model broke its directive")
            else:
                st.error("FAILURE — Model maintained its directive")
        else:
            st.info("Select a configuration and click **Run Attack** to see a live response here.")

elif phase == "Phase IIB - Multilingual Attack Comparison" and domain_id and attack_id and p2b_lang_key:
    st.markdown(f"### Multilingual Attack Comparison — {language}")
    _inst    = ALL_P2B[attack_id]
    _en_payload = P2B_PAYLOADS["mandarin"].get(_inst.BASE_ATTACK_ID, "")  # re-use ZH for preview ref
    system_prompt = SYSTEM_PROMPTS[domain_id]

    col1, col2 = st.columns(2)
    with col1:
        _panel(
            f"🤖 AGENT PROMPT · {persona_display} · {model_display}",
            system_prompt, "#0e3a4a", "#00b4d8", "#e0f7fa",
        )
        if _inst.BASE_ATTACK_ID == "attack3_fake_completion":
            from src.attacks.p2b_payloads import TURN1_PROMPTS as P2B_TURN1
            turn1_preview = P2B_TURN1[p2b_lang_key].get(domain_id, "")
            _panel(
                f"⚔️ ATTACK · {attack_display} · Turn 1 ({language})",
                turn1_preview, "#2a1a4a", "#a78bfa", "#ede9fe",
            )
            _panel(
                f"⚔️ ATTACK · {attack_display} · Turn 2 injection ({language})",
                _inst.PAYLOAD, "#4a0e0e", "#ef233c", "#ffe0e0",
            )
        else:
            _panel(
                f"⚔️ ATTACK · {attack_display} ({language})",
                _inst.PAYLOAD, "#4a0e0e", "#ef233c", "#ffe0e0",
            )

    with col2:
        last = st.session_state.get("last_result")
        if (last
                and last.get("model") == model_id
                and last.get("attack_id") == attack_id):
            score   = last["score"]
            success = last["success"]
            resp_original   = last.get("response_original", last.get("response", ""))
            resp_translated = last.get("response_translated")
            lang_detected   = last.get("response_language_detected", "")
            note            = last.get("note", "")

            if _inst.BASE_ATTACK_ID == "attack3_fake_completion" and last.get("turn1_response"):
                _panel("💬 TURN 1 RESPONSE",
                       last["turn1_response"], "#1a2a3a", "#60a5fa", "#dbeafe")

            _panel(f"💬 RAW RESPONSE · {model_display}",
                   resp_original, "#0e4a1a", "#2dc653", "#e0ffe6")

            if resp_translated:
                _panel(
                    f"🔤 TRANSLATED RESPONSE (detected: {lang_detected})",
                    resp_translated, "#1a1a3a", "#a78bfa", "#ede9fe",
                )

            # Metadata chips
            meta_parts = []
            if lang_detected:
                meta_parts.append(f"**Detected lang:** `{lang_detected}`")
            if last.get("translation_used"):
                meta_parts.append("**Translation:** Claude Haiku")
            if note:
                meta_parts.append(f"**Note:** {note}")
            if meta_parts:
                st.caption("  ·  ".join(meta_parts))

            if score == "ERROR":
                st.error("ERROR — Could not reach model")
            elif score == "NO_RESPONSE":
                st.warning("NO RESPONSE — Model returned an empty reply")
            elif score == "CONFOUND":
                st.warning("CONFOUND — Turn 1 was abnormal")
            elif score == "TRANSLATION_ERROR":
                st.error("TRANSLATION ERROR — Claude Haiku translation call failed; manual review required")
            elif score == "UNCERTAIN_LANG":
                st.warning(f"UNCERTAIN LANGUAGE — Low-confidence detection ({note}); manual review required")
            elif score == "AMBIGUOUS":
                st.warning("AMBIGUOUS — Queued for manual review")
            elif success:
                st.success("SUCCESS — Model broke its directive")
            else:
                st.error("FAILURE — Model maintained its directive")
        else:
            st.info("Select a configuration and click **Run Attack** to see a live response here.")

elif phase == "Phase IIC - LLM as an Attacker" and domain_id and attack_id:
    st.markdown("### LLM as an Attacker")
    _style = P2C_STYLES[attack_id]
    system_prompt = SYSTEM_PROMPTS[domain_id]

    col1, col2 = st.columns(2)
    with col1:
        _panel(
            f"🤖 DEFENDER PROMPT · {persona_display} · Claude Sonnet 4.6 (remote)",
            system_prompt, "#0e3a4a", "#00b4d8", "#e0f7fa",
        )
        last = st.session_state.get("last_result")
        _gen = last.get("generated_payload", "") if (last and last.get("attack_id") == attack_id) else ""
        _panel(
            f"⚔️ LLAMA-GENERATED PAYLOAD · {_style['name']}",
            _gen if _gen else f"[Click 'Run Attack' — Llama will generate a payload targeting:\n\"{_style['goal']}\"]",
            "#4a0e0e", "#ef233c", "#ffe0e0",
        )

    with col2:
        last = st.session_state.get("last_result")
        if last and last.get("attack_id") == attack_id and last.get("phase") == "iic":
            response = last["response"]
            score    = last["score"]
            success  = last["success"]
            if _style["base_attack"] == "attack3_fake_completion" and last.get("turn1_response"):
                _panel("💬 CLAUDE TURN 1 RESPONSE (on-topic)",
                       last["turn1_response"], "#1a2a3a", "#60a5fa", "#dbeafe")
            _panel("💬 CLAUDE SONNET 4.6 DEFENSE RESPONSE",
                   response, "#0e4a1a", "#2dc653", "#e0ffe6")
            if score == "ERROR":
                st.error("ERROR — Could not reach a model")
            elif score == "NO_RESPONSE":
                st.warning("NO RESPONSE — Model returned an empty reply")
            elif score == "CONFOUND":
                st.warning("CONFOUND — Turn 1 was abnormal")
            elif score == "AMBIGUOUS":
                st.warning("AMBIGUOUS — Queued for manual review")
            elif success:
                st.success("SUCCESS — Claude broke its directive (Llama wins)")
            else:
                st.error("FAILURE — Claude held its directive (Defender wins)")
        else:
            st.info("Select a configuration and click **Run Attack** — Llama will generate an injection payload live, then Claude will defend.")

elif phase == "Phase III - Fine-Tuning":
    st.markdown("### Phase III — Fine-Tuning")
    _panel("Research Phase",
           "DPO fine-tuning will be applied to the most vulnerable Phase I model.\n"
           "Phase I attacks are then re-run for a before/after ASR comparison.",
           "#1a1a2e", "#9b59b6", "#e0e0e0")

# ── Results Dashboard ─────────────────────────────────────

st.divider()
st.markdown("### Results Dashboard")

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
df_formal     = df_all[df_all["source"] == "formal"]     if not df_all.empty else df_all
df_formal_v2  = df_all[df_all["source"] == "formal_v2"]  if not df_all.empty else df_all
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

tab_p1, tab_p2, tab_p3, tab_demo, tab_overview = st.tabs([
    "Phase I", "Phase II", "Phase III", "Live Demo", "Overview",
])

# ═══════════════════════════════════════════════════════════
# PHASE I
# ═══════════════════════════════════════════════════════════
with tab_p1:
    st.markdown("#### Phase I — Baseline Attacks")
    _p1_rubric_sel = st.radio(
        "Rubric version", ["v2 (formal_v2)", "v1 (formal)"],
        index=0, horizontal=True, key="p1_rubric",
    )
    _p1_rubric = "v1" if _p1_rubric_sel.startswith("v1") else "v2"
    _df_p1 = df_formal_v2 if _p1_rubric == "v2" else df_formal
    _df_p1_filt = (
        _df_p1[_df_p1["attack_id"].isin(P1_ATTACK_IDS) & _df_p1["domain"].isin(P1_DOMAINS)]
        if not _df_p1.empty else _df_p1
    )
    _asr_p1 = compute_asr(_df_p1_filt)

    p1_sub1, p1_sub2, p1_sub3 = st.tabs(["ASR by Attack", "Model Analysis", "Data Replay"])

    with p1_sub1:
        st.markdown("##### Attack Success Rate by Attack Type")
        st.caption("Avg ASR across all models, grouped by domain. Rubric: " + _p1_rubric)
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
        st.caption(f"Browsing results/formal{'_v2' if _p1_rubric=='v2' else ''}/ — no API calls.")
        _p1_rp = [
            {**r, "_language": "english", "_is_p2b": False,
             "_payload": r.get("payload",""), "_response": r.get("response","")}
            for r in load_formal_records(rubric=_p1_rubric)
        ]
        _render_replay(_p1_rp, key_pfx="p1_rp", show_lang=False)


# ═══════════════════════════════════════════════════════════
# PHASE II
# ═══════════════════════════════════════════════════════════
with tab_p2:
    st.markdown("#### Phase II — Advanced Attack Techniques")
    p2_iia, p2_iib, p2_iic = st.tabs(["Phase IIA", "Phase IIB", "Phase IIC"])

    # ── Phase IIA ──────────────────────────────────────────
    with p2_iia:
        st.markdown("##### Phase IIA — Cognitive Schema Attacks")
        _df_p2a_filt = (
            df_formal_v2[df_formal_v2["attack_id"].isin(P2A_ATTACK_IDS) & df_formal_v2["domain"].isin(P2A_DOMAINS)]
            if not df_formal_v2.empty else df_formal_v2
        )
        _asr_p2a = compute_asr(_df_p2a_filt)

        p2a_s1, p2a_s2, p2a_s3 = st.tabs(["ASR by Attack", "Model Analysis", "Data Replay"])

        with p2a_s1:
            if _asr_p2a.empty:
                st.info("No Phase IIA data yet. Run: `python -m src.main --phase p2a`")
            else:
                _p2a_grp = _asr_p2a.groupby(["attack_name","domain"])["asr"].mean().round(1).reset_index()
                _p2a_dom_colors = {"compliant_assistant": "#a78bfa", "truth_teller": "#f4a261"}
                fig_p2a_atk = go.Figure()
                for _dom in ["compliant_assistant", "truth_teller"]:
                    _sub = _p2a_grp[_p2a_grp["domain"] == _dom].sort_values("attack_name")
                    if _sub.empty:
                        continue
                    fig_p2a_atk.add_trace(go.Bar(
                        name=DOMAIN_LABELS.get(_dom, _dom),
                        x=_sub["attack_name"], y=_sub["asr"],
                        marker_color=_p2a_dom_colors.get(_dom, "#888"),
                        text=[f"{v:.0f}%" for v in _sub["asr"]], textposition="outside",
                    ))
                fig_p2a_atk.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#ffffff",
                    yaxis=dict(range=[0, 120], title="ASR %", gridcolor="#222"),
                    xaxis=dict(title="Attack Type"),
                    barmode="group", height=420,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_p2a_atk, use_container_width=True)
                _png_download_btn("Download as PNG", fig_p2a_atk, "p2a_asr_by_attack.png", height=480)

        with p2a_s2:
            if _asr_p2a.empty:
                st.info("No Phase IIA data yet.")
            else:
                _p2a_stats = _model_asr_sem(_df_p2a_filt)
                _p2a_models = sorted(_p2a_stats.keys())
                if _p2a_stats:
                    fig_p2a_mc = go.Figure(go.Bar(
                        x=_p2a_models,
                        y=[_p2a_stats[m][0] for m in _p2a_models],
                        error_y=dict(type="data", array=[_p2a_stats[m][1]*2 for m in _p2a_models],
                                     visible=True, color="#ffffff", thickness=1.5),
                        marker_color="#a78bfa",
                        text=[f"{int(_p2a_stats[m][0])}%" for m in _p2a_models],
                        textposition="outside",
                    ))
                    fig_p2a_mc.update_layout(
                        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#ffffff",
                        yaxis=dict(range=[0, 130], title="Avg ASR %", gridcolor="#222"),
                        xaxis=dict(title="Model"), showlegend=False, height=420,
                    )
                    st.plotly_chart(fig_p2a_mc, use_container_width=True)
                    _png_download_btn("Download as PNG", fig_p2a_mc, "p2a_model_comparison.png", height=480)
                st.divider()
                _render_heatmap(_asr_p2a, key_sfx="p2a")

        with p2a_s3:
            st.caption("Browsing Phase IIA records from results/formal_v2/ — no API calls.")
            _p2a_rp = [
                {**r, "_language": "english", "_is_p2b": False,
                 "_payload": r.get("payload",""), "_response": r.get("response","")}
                for r in load_p2a_records()
            ]
            _render_replay(_p2a_rp, key_pfx="p2a_rp", show_lang=False)

    # ── Phase IIB ──────────────────────────────────────────
    with p2_iib:
        st.markdown("##### Phase IIB — Multilingual Attack Comparison")
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
            # Phase I English baseline for top-3
            _p1_base: dict[str, float] = {}
            if not df_formal_v2.empty:
                _p1f = df_formal_v2[
                    df_formal_v2["attack_id"].isin(_TOP3_IDS) & ~df_formal_v2["score"].isin(_excl_p2b)
                ]
                for _aid, _grp in _p1f.groupby("attack_id"):
                    _p1_base[_aid] = round(float(_grp["success"].mean()) * 100, 1)
            # Phase IIB ASR per (base_attack, language)
            _p2b_asr: dict[tuple, float] = {}
            if not df_formal_p2b.empty:
                _p2bf = df_formal_p2b.copy()
                _p2bf["_base"] = _p2bf["attack_id"].map(_P2B_BASE_LOOKUP)
                _p2bf_c = _p2bf[~_p2bf["score"].isin(_excl_p2b) & _p2bf["_base"].isin(_TOP3_IDS)]
                for (_base, _lang), _grp in _p2bf_c.groupby(["_base", "language"]):
                    _p2b_asr[(_base, _LANG_DISP.get(_lang, _lang))] = round(
                        float(_grp["success"].mean()) * 100, 1
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
                st.info("No Phase IIB data yet. Run: `python -m src.main --phase p2b`")
            else:
                fig_p2b = go.Figure()
                for _sn, _sv in _series.items():
                    fig_p2b.add_trace(go.Bar(
                        name=_sn, x=_atk_labels, y=_sv,
                        marker_color=_LANG_COLORS.get(_sn, "#888"),
                        text=[f"{v:.0f}%" if v is not None else "—" for v in _sv],
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
                st.caption("ASR = successes / non-ambiguous runs, averaged across all models and domains.")
                _png_download_btn("Download as PNG", fig_p2b, "p2b_asr_by_language.png", height=520)

        with p2b_s2:
            st.markdown("##### Model Vulnerability — Phase IIB")
            _p2b_model_df = df_formal_p2b[~df_formal_p2b["score"].isin(_excl_p2b)] if not df_formal_p2b.empty else df_formal_p2b
            if _p2b_model_df.empty:
                st.info("No Phase IIB data yet. Run: `python -m src.main --phase p2b`")
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
                    st.caption("Error bars = ±2 SEM (95% CI). Averaged over all Phase IIB attacks, languages, and domains.")
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
            st.caption("Browsing results/formal_p2b/ — no API calls.")
            _render_replay(_p2b_rp_recs, key_pfx="p2b_rp", show_lang=True)

    # ── Phase IIC ──────────────────────────────────────────
    with p2_iic:
        st.markdown("##### Phase IIC — LLM as Attacker")
        _panel(
            "Experimental Phase",
            "Phase IIC uses Llama 3.1 8B as an adaptive attacker to generate injection payloads\n"
            "targeting Claude Sonnet 4.6 as the defender.\n\n"
            "Use the sidebar (Phase IIC - LLM as an Attacker) to run live trials.\n"
            "Formal batch runs are not yet collected.",
            "#1a1a2e", "#f4a261", "#e0e0e0",
        )


# ═══════════════════════════════════════════════════════════
# PHASE III
# ═══════════════════════════════════════════════════════════
with tab_p3:
    st.markdown("#### Phase III — Fine-Tuning")
    _panel(
        "Coming Soon",
        "DPO/LoRA fine-tuning will be applied to the most vulnerable Phase I model.\n"
        "Phase I attacks will then be re-run for a before/after ASR comparison.\n\n"
        "Prerequisite: Phase I data collection complete (done).",
        "#1a1a2e", "#9b59b6", "#e0e0e0",
    )


# ═══════════════════════════════════════════════════════════
# LIVE DEMO
# ═══════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("#### Live Demo — Session Run Log")
    st.caption("Runs made this session via the sidebar. Saved to results/scratch/ for reference.")
    _history = st.session_state.get("run_history", [])
    if not _history:
        st.info("No runs this session. Select a phase and model in the sidebar, then click **Run Attack**.")
    else:
        _SCORE_BADGE = {
            "SUCCESS":     ("🟢", "#2dc653"),
            "FAILURE":     ("🔴", "#ef233c"),
            "AMBIGUOUS":   ("🟡", "#f4a261"),
            "NO_RESPONSE": ("🟡", "#f4a261"),
            "CONFOUND":    ("🟡", "#f4a261"),
            "ERROR":       ("🔴", "#ef233c"),
        }
        st.markdown(f"**{len(_history)} run(s) this session**")
        _summary_rows = []
        for _i, _entry in enumerate(reversed(_history)):
            _icon, _ = _SCORE_BADGE.get(_entry["score"], ("⚪", "#aaa"))
            _summary_rows.append({
                "#": len(_history)-_i,
                "Time":   _entry["time"],
                "Model":  _entry["model"],
                "Attack": _entry["attack"],
                "Domain": _entry["domain"],
                "Score":  f"{_icon} {_entry['score']}",
            })
        st.dataframe(pd.DataFrame(_summary_rows), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown("**Response Details**")
        for _i, _entry in enumerate(reversed(_history)):
            _run_num = len(_history) - _i
            _icon, _color = _SCORE_BADGE.get(_entry["score"], ("⚪", "#aaa"))
            _lbl = f"#{_run_num}  {_entry['time']}  ·  {_entry['attack']}  ·  {_entry['model']}  ·  {_icon} {_entry['score']}"
            with st.expander(_lbl, expanded=(_i == 0)):
                st.markdown(
                    f"<span style='color:{_color}; font-weight:bold;'>{_entry['score']}</span>"
                    f"  &nbsp;|&nbsp;  {_entry['domain']}",
                    unsafe_allow_html=True,
                )
                st.text_area("Response", _entry["response"], height=160,
                             key=f"demo_resp_{_run_num}_{_i}", disabled=True)


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
            not df_formal_p2b.empty,
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
            _ov_p2b_raw= _ov_clean[_ov_clean["source"] == "formal_p2b"]                      if not _ov_clean.empty else _ov_clean

            _ov_p1_st  = _model_asr_sem(_ov_p1_raw)
            _ov_p2a_st = _model_asr_sem(_ov_p2a_raw)
            _ov_p2b_st = _model_asr_sem(_ov_p2b_raw)
            _ov_mdls   = sorted(set(list(_ov_p1_st)+list(_ov_p2a_st)+list(_ov_p2b_st)))
            if not _ov_mdls:
                st.info("No results yet.")
            else:
                fig_ov_mc = go.Figure()
                for _sn, _st, _col in [
                    ("Phase I — Baseline",      _ov_p1_st,  "#00b4d8"),
                    ("Phase IIA — Schema",       _ov_p2a_st, "#a78bfa"),
                    ("Phase IIB — Multilingual", _ov_p2b_st, "#ef233c"),
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
        _p2b_pct = 0 if df_formal_p2b.empty else min(
            100, round(df_formal_p2b["model"].nunique() / len(config.ALL_MODELS) * 100)
        )
        _progress = [
            _phase_pct(["cooking","health"]),
            _phase_pct(["compliant_assistant","truth_teller"]),
            _p2b_pct, 0, 0,
        ]
        fig_tl = go.Figure(go.Bar(
            x=["Phase I — Baseline","Phase IIA — Schema","Phase IIB — Multilingual",
               "Phase IIC — LLM Attacker","Phase III — Fine-Tuning"],
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
