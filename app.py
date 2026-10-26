"""
AgriChainAI — Intelligent Farm & Cold-Chain Assistant (Streamlit UI).

Run:  python3 -m streamlit run app.py

The UI only renders; every prediction is produced by the existing Farm Agent /
Spoilage predictors through agents.orchestrator.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from agents.farm_adapter import FarmService
from agents.orchestrator import ChatState, Orchestrator
from agents.spoilage_adapter import DryFruitService, PaneerService

st.set_page_config(page_title="AgriChainAI", page_icon="🌾", layout="wide")

CSS = """
<style>
:root { --g1:#2e7d32; --g2:#66bb6a; --bg:#f4f8f1; }
.block-container { padding-top: 1.2rem; max-width: 1050px; }
.hero { background: linear-gradient(120deg,#1b5e20,#43a047); color:#fff; padding:18px 24px;
        border-radius:14px; margin-bottom:12px; }
.hero h1 { margin:0; font-size:1.9rem; color:#fff; }
.hero p { margin:2px 0 0 0; opacity:.92; }
.pill { display:inline-block; padding:3px 12px; border-radius:999px; font-weight:600; font-size:.85rem; color:#fff; }
.pill.FRESH{background:#2e7d32} .pill.WARNING{background:#ef6c00} .pill.SPOILED{background:#c62828}
.card { border:1px solid rgba(46,125,50,.35); border-left:6px solid #2e7d32; border-radius:10px;
        padding:14px 18px; margin:8px 0; background:rgba(102,187,106,.08); }
.card h4 { margin:0 0 6px 0; }
.dot { height:10px; width:10px; border-radius:50%; display:inline-block; margin-right:6px; }
.dot.Ready{background:#2e7d32} .dot.Error{background:#c62828}
.dot.Initializing{background:#ef6c00} .dot.NotConfigured{background:#9e9e9e}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def get_farm_service() -> FarmService:
    """Train the existing FarmAgent ONCE per Streamlit process (it has no persistence API)."""
    return FarmService.initialize()


@st.cache_resource(show_spinner=False)
def get_paneer_service() -> PaneerService:
    return PaneerService()


@st.cache_resource(show_spinner=False)
def get_dried_service() -> DryFruitService:
    return DryFruitService()


def build_orchestrator() -> Orchestrator:
    with st.spinner("Initialising Farm Agent (one-time model training, ~20–60 s)…"):
        farm = get_farm_service()
    return Orchestrator(farm, get_paneer_service(), get_dried_service())


orch = build_orchestrator()
if "chat" not in st.session_state:
    st.session_state.chat = ChatState()
    orch.new_conversation(st.session_state.chat)
chat: ChatState = st.session_state.chat

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🌾 System status")
    for name, (status, detail) in orch.statuses().items():
        css = status.replace(" ", "")
        st.markdown(f'<span class="dot {css}"></span>**{name}:** {status}', unsafe_allow_html=True)
        if detail:
            st.caption(detail)
    if any(s == "Error" for s, _ in orch.statuses().values()):
        if st.button("Retry initialisation"):
            st.cache_resource.clear()
            st.rerun()
    st.divider()
    if st.button("🆕 New conversation", use_container_width=True):
        orch.new_conversation(chat)
        st.rerun()
    farm = orch.services["farm"]
    if farm.status == "Ready" and farm.report:
        st.caption(f"Farm model trained on: {farm.report['source']}")
    st.caption("Predictions come from the project's saved/trained models. "
               "No external LLM or internet connection is used.")

# --------------------------------------------------------------------- hero
st.markdown('<div class="hero"><h1>🌾 AgriChainAI</h1><p>Intelligent Farm &amp; Cold-Chain Assistant</p></div>',
            unsafe_allow_html=True)

QUICK = [("🌱 Recommend a Crop", "Recommend a crop"),
         ("🧀 Check Paneer Spoilage", "Check paneer spoilage"),
         ("🥭 Check Dried-Fruit Spoilage", "Check dried-fruit spoilage"),
         ("🔗 Run Both Analyses", "Run both analyses")]
cols = st.columns(4)
pending = None
for col, (label, phrase) in zip(cols, QUICK):
    if col.button(label, use_container_width=True):
        pending = phrase
typed = st.chat_input("Ask for a crop recommendation or a spoilage check…")
pending = typed or pending

if pending:
    with st.spinner("Supervisor is routing your request and running the agents…"):
        orch.handle_message(chat, pending)


# ---------------------------------------------------------------- rendering
def fmt_inr(v):
    return f"₹{v:,.0f}"


def render_farm(d):
    b = d["best"]
    st.markdown(f'<div class="card"><h4>🌾 Farm Agent — recommended crop: {b["crop"].title()}</h4>'
                f'Dataset: {d["dataset"]}</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Model confidence", f'{b["confidence"] * 100:.1f}%')
    c[1].metric("Expected yield", f'{b["expected_yield_t_per_ha"]:.2f} t/ha')
    c[2].metric("Expected profit", fmt_inr(b["expected_profit_inr"]))
    if "harvest_month" in b:
        c[3].metric("Est. harvest", f'{b["harvest_month"]}', b.get("harvest_date_estimate"), delta_color="off")
    if d["alternatives"]:
        st.markdown("**Alternatives**")
        st.dataframe(pd.DataFrame([{
            "Crop": a["crop"].title(), "Confidence": f'{a["confidence"] * 100:.1f}%',
            "Yield (t/ha)": a["expected_yield_t_per_ha"], "Profit": fmt_inr(a["expected_profit_inr"]),
            "Estimate source": a["estimate_source"]} for a in d["alternatives"]]),
            hide_index=True, use_container_width=True)
    with st.expander("Inputs used"):
        st.json({k: str(v) for k, v in d["inputs"].items() if v is not None})
    for line in d.get("defaults_used", []):
        st.caption(f"Not provided → used {line}")
    for w in d.get("warnings", []):
        st.warning(w)
    for cv in d["caveats"]:
        st.caption("ℹ️ " + cv)


def render_spoilage(d):
    st.markdown(f'<div class="card"><h4>🧪 Spoilage Agent — {d["product_subtype"]}</h4>'
                f'Predicted status: <span class="pill {d["status"]}">{d["status"]}</span> &nbsp; '
                f'Model: {d["model_used"]}</div>', unsafe_allow_html=True)
    p = d["probabilities"]
    c = st.columns(4)
    c[0].metric("Fresh", f'{p["FRESH"] * 100:.1f}%')
    c[1].metric("Warning", f'{p["WARNING"] * 100:.1f}%')
    c[2].metric("Spoiled", f'{p["SPOILED"] * 100:.1f}%')
    c[3].metric("Spoilage risk", f'{d["spoilage_risk"]:.1f}%')
    st.caption("Spoilage risk is the value returned by the predictor (equal to the SPOILED probability).")
    with st.expander("Inputs used"):
        st.json({k: str(v) for k, v in d["inputs"].items()})
    st.info(d["interpretation"])
    for line in d.get("defaults_used", []):
        st.caption(f"Not provided → used {line}")
    for w in d.get("warnings", []):
        st.warning(w)
    for n in d["notes"]:
        st.caption("ℹ️ " + n)


def render_summary(farm, sp):
    st.markdown('<div class="card"><h4>📋 Combined summary</h4></div>', unsafe_allow_html=True)
    b = farm["best"]
    st.markdown(
        f"**A. Farm recommendation (Farm Agent):** {b['crop'].title()} — confidence {b['confidence'] * 100:.1f}%, "
        f"expected yield {b['expected_yield_t_per_ha']:.2f} t/ha, expected profit {fmt_inr(b['expected_profit_inr'])}"
        + (f", estimated harvest {b['harvest_month']}." if "harvest_month" in b else "."))
    st.markdown(
        f"**B. Product spoilage assessment (Spoilage Agent):** {sp['product_subtype']} — predicted status "
        f"**{sp['status']}**, spoilage risk {sp['spoilage_risk']:.1f}%.")
    st.markdown("**C. System-level summary:** The supervisor ran both specialist agents in one session — "
                "the Farm Agent and the Spoilage Agent. The two analyses are independent: the product assessed "
                "was chosen by you and is not derived from the crop recommendation.")


def render_trace(trace):
    with st.expander("Agent Execution Trace"):
        st.markdown(" ↓ ".join(f'`{s["step"]}`' for s in trace))
        st.dataframe(pd.DataFrame([{
            "Step": s["step"], "Agent/Tool": s["agent"], "Status": s["status"], "Input summary": s["detail"],
            "Duration (ms)": s["duration_ms"] if s["duration_ms"] is not None else "—"} for s in trace]),
            hide_index=True, use_container_width=True)


for msg in chat.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🌾" if msg["role"] == "user" else "🌾"):
        for blk in msg["blocks"]:
            t = blk["type"]
            if t == "text":
                st.markdown(blk["text"])
            elif t == "error":
                st.error(blk["text"])
            elif t == "farm_result":
                render_farm(blk["data"])
            elif t == "spoilage_result":
                render_spoilage(blk["data"])
            elif t == "summary":
                render_summary(blk["farm"], blk["spoilage"])
        if msg.get("trace"):
            render_trace(msg["trace"])
