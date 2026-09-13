"""
CLO Waterfall Simulator — Streaming Multi-Thread Interface
==========================================================

Features:
• Each CLO run gets its own persistent thread (UUID → SQLite checkpointer)
• Real-time streaming: live per-agent status as the pipeline executes
• Thread sidebar: browse, switch, and reload any past analysis
• Rich result cards: extracted rules, generated code, simulation JSON, report
• Per-PDF isolated vector store with inline management controls
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import sqlite3
from datetime import datetime
from typing import Optional

import streamlit as st

# ── path resolution ────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.checkpoint.sqlite import SqliteSaver
from src.main import build_clo_graph, MAX_ITERATIONS
from src.agents.chatbot_agent import build_chatbot_graph
from src.tools.retriever import get_collection_info, delete_collection

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CLO Waterfall Simulator · AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium dark-accented design
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global typography ──────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Agent timeline cards ───────────────────────────── */
.agent-card {
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    border-left: 4px solid;
    backdrop-filter: blur(4px);
}
.agent-parser  { border-color: #818cf8; background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(99,102,241,0.03) 100%); }
.agent-quant   { border-color: #22d3ee; background: linear-gradient(135deg, rgba(34,211,238,0.08) 0%, rgba(34,211,238,0.03) 100%); }
.agent-critic  { border-color: #f59e0b; background: linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(245,158,11,0.03) 100%); }
.agent-reporter{ border-color: #34d399; background: linear-gradient(135deg, rgba(52,211,153,0.08) 0%, rgba(52,211,153,0.03) 100%); }

/* ── Status badges ──────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-approved { background: rgba(52,211,153,0.15); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.3); }
.badge-rejected { background: rgba(239,68,68,0.15);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.badge-running  { background: rgba(251,191,36,0.15); color: #fde68a; border: 1px solid rgba(251,191,36,0.3); }
.badge-error    { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
.badge-pending  { background: rgba(148,163,184,0.10);color: #94a3b8; border: 1px solid rgba(148,163,184,0.2);}

/* ── Metric tiles ───────────────────────────────────── */
.metric-tile {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-val  { font-size: 1.6rem; font-weight: 700; }
.metric-label{ font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.06em; text-transform: uppercase; margin-top: 2px; }

/* ── Thread list buttons ────────────────────────────── */
div[data-testid="stSidebarContent"] .stButton button {
    text-align: left;
    font-size: 0.82rem;
    padding: 8px 12px;
}

/* ── Section divider ────────────────────────────────── */
.section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin: 16px 0 6px 0;
}

/* ── Welcome hero ───────────────────────────────────── */
.hero {
    text-align: center;
    padding: 60px 20px 40px;
}
.hero h1 { font-size: 2.6rem; font-weight: 700; margin-bottom: 8px; }
.hero p  { font-size: 1.05rem; color: #94a3b8; max-width: 560px; margin: 0 auto 32px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR   = "data/raw_indentures"
DB_PATH    = "data/workflow_threads.db"
KNOWN_NODES = {"parser_node", "quant_node", "critic_node", "reporter_node"}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES — App + Checkpointer (created once per server session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def _get_app_and_checkpointer():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cp   = SqliteSaver(conn=conn)
    app  = build_clo_graph(checkpointer=cp)
    chat = build_chatbot_graph(checkpointer=cp)
    return app, chat, cp

clo_app, chatbot_app, checkpointer = _get_app_and_checkpointer()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def new_thread_id() -> str:
    return str(uuid.uuid4())


def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def list_all_thread_ids() -> list[str]:
    """Return all main workflow thread IDs ever saved by the checkpointer."""
    seen = set()
    try:
        for cp in checkpointer.list(None):
            tid = cp.config["configurable"]["thread_id"]
            if not str(tid).endswith("-chat"):
                seen.add(tid)
    except Exception:
        pass
    return list(seen)


def delete_analysis_thread(tid: str):
    """Delete a thread and its chat history from the database."""
    try:
        # Some langgraph versions have delete_thread, others might need raw sql.
        if hasattr(checkpointer, "delete_thread"):
            checkpointer.delete_thread(tid)
            checkpointer.delete_thread(f"{tid}-chat")
        elif hasattr(checkpointer, "conn"):
            # Raw SQL fallback
            cur = checkpointer.conn.cursor()
            cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (tid,))
            cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (f"{tid}-chat",))
            cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (tid,))
            cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (f"{tid}-chat",))
            checkpointer.conn.commit()
        
        # Clean up session state
        if tid in st.session_state.all_thread_ids:
            st.session_state.all_thread_ids.remove(tid)
        if tid in st.session_state.thread_meta:
            del st.session_state.thread_meta[tid]
        if st.session_state.active_thread == tid:
            st.session_state.active_thread = None
    except Exception as e:
        st.error(f"Failed to delete thread: {e}")


def load_thread_state(thread_id: str) -> Optional[dict]:
    """Load the final persisted state for a given thread."""
    try:
        snap = clo_app.get_state(config=make_config(thread_id))
        return dict(snap.values) if snap and snap.values else None
    except Exception:
        return None


def fmt_money(val) -> str:
    if val is None:
        return "N/A"
    val = float(val)
    if val >= 1_000_000_000:
        return f"${val/1_000_000_000:.2f}B"
    if val >= 1_000_000:
        return f"${val/1_000_000:.0f}M"
    return f"${val:,.0f}"


def status_badge_html(status: str) -> str:
    cls = {
        "approved": "badge-approved",
        "rejected": "badge-rejected",
        "simulated": "badge-running",
        "quant_error": "badge-error",
        "failed": "badge-error",
    }.get(status, "badge-pending")
    return f'<span class="badge {cls}">{status}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
if "active_thread"   not in st.session_state: st.session_state.active_thread   = None
if "pending_pdf"     not in st.session_state: st.session_state.pending_pdf     = None
if "all_thread_ids"  not in st.session_state:
    st.session_state.all_thread_ids = list_all_thread_ids()

if "thread_meta"     not in st.session_state: 
    st.session_state.thread_meta = {}
    for tid in st.session_state.all_thread_ids:
        state = load_thread_state(tid)
        if state:
            pw = state.get("parsed_waterfall") or {}
            deal = pw.get("deal_name", f"Thread {tid[:8]}")
            pdf_path = state.get("pdf_path", "")
            st.session_state.thread_meta[tid] = {
                "deal_name": deal,
                "status": state.get("status", "completed"),
                "pdf_name": os.path.basename(pdf_path) if pdf_path else "",
            }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 CLO Simulator")
    st.caption("Agentic AI · Multi-Thread · Streaming")
    st.divider()

    # ── New Analysis button ────────────────────────────────────────────────
    if st.button("➕  New Analysis", use_container_width=True, type="primary"):
        st.session_state.active_thread = None
        st.session_state.pending_pdf   = None
        st.rerun()

    # ── Document selection ────────────────────────────────────────────────
    st.markdown('<p class="section-title">📁 Document</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload PDF", type=["pdf"], label_visibility="collapsed", key="pdf_upload"
    )
    if uploaded:
        save_path = os.path.join(DATA_DIR, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.pending_pdf = save_path
        st.success(f"✅ {uploaded.name}")

    existing = sorted(os.listdir(DATA_DIR))
    if existing:
        pick = st.selectbox("Or select existing", ["—"] + existing, key="pdf_pick",
                            label_visibility="collapsed")
        if pick != "—":
            st.session_state.pending_pdf = os.path.join(DATA_DIR, pick)

    # ── Vector store status for chosen PDF ───────────────────────────────
    if st.session_state.pending_pdf:
        info = get_collection_info(st.session_state.pending_pdf)
        basename = os.path.basename(st.session_state.pending_pdf)
        st.markdown(f"<small>📄 `{basename}`</small>", unsafe_allow_html=True)
        if info["indexed"]:
            st.success(f"🗄️ Indexed · {info.get('index_size_mb', '?')} MB")
            if st.button("🗑️ Clear Index", key="clear_idx"):
                delete_collection(st.session_state.pending_pdf)
                st.warning("Index cleared — will re-embed on next run.")
                st.rerun()
        else:
            st.info("⏳ Not yet indexed")

    # ── Past threads ──────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🕐 Past Analyses</p>', unsafe_allow_html=True)

    all_tids = st.session_state.all_thread_ids
    if not all_tids:
        st.caption("No analyses yet.")
    else:
        for tid in reversed(all_tids):           # latest first
            meta = st.session_state.thread_meta.get(tid, {})
            deal   = meta.get("deal_name", f"Thread {tid[:8]}")
            status = meta.get("status", "")
            pdf_n  = meta.get("pdf_name", "")
            ts     = meta.get("timestamp", "")

            icon = {"approved": "✅", "rejected": "❌", "quant_error": "⚠️"}.get(status, "⏺")
            label = f"{icon}  {deal[:28]}"
            if pdf_n:
                label += f"\n`{pdf_n[:22]}`"
            if ts:
                label += f"\n{ts[:16]}"

            is_active = tid == st.session_state.active_thread
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                if st.button(label, key=f"th-{tid}", use_container_width=True,
                             type="primary" if is_active else "secondary"):
                    st.session_state.active_thread = tid
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del-{tid}", help="Delete this analysis"):
                    delete_analysis_thread(tid)
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS — agent result cards
# ══════════════════════════════════════════════════════════════════════════════

def _view_json_modal(title: str, data: dict):
    @st.dialog(title)
    def show_json():
        st.json(data)
    show_json()

def _view_code_modal(title: str, code: str):
    @st.dialog(title)
    def show_code():
        st.code(code, language="python")
    show_code()

def _parser_card(state: dict):
    pw = state.get("parsed_waterfall") or {}
    deal  = pw.get("deal_name") or "—"
    par   = pw.get("total_target_par") or 0
    n_tr  = len(pw.get("tranches") or [])
    n_ct  = len(pw.get("coverage_tests") or [])
    vec   = state.get("vector_store_id", "—")

    with st.container(border=True):
        st.markdown(f"#### 📄 Parser Agent &nbsp; {status_badge_html('approved')}", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Deal Name", deal[:22] + ("…" if len(deal) > 22 else ""))
        c2.metric("Total Par", fmt_money(par))
        c3.metric("Tranches", n_tr)
        c4.metric("Coverage Tests", n_ct)

        if st.button("📋 View Extracted IndentureRules JSON", key="btn_parser_json"):
            _view_json_modal("Extracted IndentureRules JSON", pw)

        # ── Data quality warning ──────────────────────────────────────────
        dq = state.get("parser_data_quality") or {}
        if dq and not dq.get("principals_extracted", True):
            st.warning(
                "⚠️ **Principal amounts could not be extracted from the PDF.** "
                "The simulation will use estimated proportional fallbacks. "
                "Results are **not based on the actual deal terms** — "
                "try clearing the vector index and re-running for better extraction.",
                icon="⚠️",
            )
        elif dq and dq.get("principals_extracted"):
            st.success("✅ Tranche principal amounts successfully extracted from the document.")

        st.caption(f"🗄️ Vector Store Collection: `{vec}`")


def _quant_card(state: dict, iteration: int):
    code    = state.get("generated_code", "")
    results = state.get("simulation_results") or {}
    err     = state.get("execution_error")
    status  = state.get("status", "")

    badge_cls  = "badge-approved" if not err else "badge-error"
    badge_text = "✓ Executed" if not err else "⚠ Error"

    with st.container(border=True):
        st.markdown(
            f"#### 📊 Quant Agent &nbsp; <span class='badge {badge_cls}'>{badge_text}</span>"
            f" &nbsp; <small style='color:#64748b'>Iteration {iteration}</small>",
            unsafe_allow_html=True,
        )

        if err:
            st.error(f"**Runtime Error:** {err}")

        cols = st.columns(2)
        with cols[0]:
            if code and st.button("💻 View Generated Python Code", key="btn_quant_code"):
                _view_code_modal("Generated Python Code", code)
        with cols[1]:
            if results and st.button("📈 View Monte Carlo JSON", key="btn_quant_json"):
                _view_json_modal("Monte Carlo Raw JSON", results)

        if results and isinstance(results, dict):
            st.markdown("**Loss Probability per Tranche**")
            rows = [{"Tranche": k, "Loss Probability": f"{v:.2%}"} for k, v in results.items()]
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)


def _critic_card(state: dict, iteration: int):
    feedback = state.get("critic_feedback", "—")
    status   = state.get("status", "")

    badge_cls  = "badge-approved" if status == "approved" else "badge-rejected"
    badge_text = "✅ APPROVED" if status == "approved" else "↩ REJECTED"

    with st.container(border=True):
        st.markdown(
            f"#### ⚖️ Critic Agent &nbsp; <span class='badge {badge_cls}'>{badge_text}</span>"
            f" &nbsp; <small style='color:#64748b'>Iteration {iteration}</small>",
            unsafe_allow_html=True,
        )
        st.info(feedback or "No feedback provided.")


def _reporter_card(state: dict):
    report = state.get("final_report", "")

    with st.container(border=True):
        st.markdown(f"#### 📑 Reporter Agent &nbsp; {status_badge_html('approved')}", unsafe_allow_html=True)
        if report:
            st.markdown(report)
        else:
            st.caption("No report generated.")


def display_full_results(thread_id: str, state: dict):
    """Renders the complete results layout for a finished (or resumed) thread."""
    pw     = state.get("parsed_waterfall") or {}
    status = state.get("status", "unknown")
    iters  = state.get("iteration_count", 0)
    meta   = st.session_state.thread_meta.get(thread_id, {})
    ts     = meta.get("timestamp", "")

    # ── Header ────────────────────────────────────────────────────────────
    deal = pw.get("deal_name", "CLO Analysis")
    st.markdown(f"## {deal}")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown(f'<div class="metric-tile"><div class="metric-val">{status.upper()}</div><div class="metric-label">Status</div></div>', unsafe_allow_html=True)
    with h2:
        st.markdown(f'<div class="metric-tile"><div class="metric-val">{iters}</div><div class="metric-label">Iterations</div></div>', unsafe_allow_html=True)
    with h3:
        st.markdown(f'<div class="metric-tile"><div class="metric-val">{fmt_money(pw.get("total_target_par", 0))}</div><div class="metric-label">Total Par</div></div>', unsafe_allow_html=True)
    with h4:
        st.markdown(f'<div class="metric-tile"><div class="metric-val">{thread_id[:8]}…</div><div class="metric-label">Thread</div></div>', unsafe_allow_html=True)
    
    if ts:
        st.caption(f"Run at: {ts}")
    st.divider()

    # ── Agent Cards ───────────────────────────────────────────────────────
    _parser_card(state)
    _quant_card(state, iters)
    _critic_card(state, iters)

    if state.get("final_report"):
        _reporter_card(state)

def display_chat_interface(thread_id: str, workflow_state: dict):
    st.markdown("### 💬 Chat with your CLO Data")
    st.caption("Ask questions about the document, tranches, or the generated simulation results.")
    
    chat_thread_id = f"{thread_id}-chat"
    config = {"configurable": {"thread_id": chat_thread_id}}

    # Fetch existing messages
    try:
        chat_state = chatbot_app.get_state(config)
        messages = chat_state.values.get("messages", [])
    except Exception:
        messages = []

    # Inject context if starting fresh
    if not messages:
        pdf_path = workflow_state.get("pdf_path") or ""
        pw = workflow_state.get("parsed_waterfall") or {}
        deal_name = pw.get("deal_name") or "Unknown Deal"
        system_injection = (
            f"[System Context]\n"
            f"Active PDF Path: `{pdf_path}`\n"
            f"Active Workflow Thread ID: `{thread_id}`\n"
            f"Deal Name: `{deal_name}`\n"
        )
        chatbot_app.update_state(config, {"messages": [("system", system_injection)]})
        # reload messages
        chat_state = chatbot_app.get_state(config)
        messages = chat_state.values.get("messages", [])

    # Display history
    for msg in messages:
        if msg.type == "system":
            continue
        if msg.type == "human":
            with st.chat_message("user"):
                st.write(msg.content)
        elif msg.type == "ai":
            with st.chat_message("assistant"):
                if msg.content:
                    st.write(msg.content)
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        with st.expander(f"🛠️ Executed Tool: `{tc['name']}`"):
                            st.json(tc["args"])
        elif msg.type == "tool":
            with st.chat_message("tool"):
                with st.expander(f"🔧 Tool Result: `{msg.name}`"):
                    st.write(msg.content)

    # Chat Input
    if prompt := st.chat_input("Ask about the indenture or results..."):
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            st_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    for chunk in chatbot_app.stream(
                        {"messages": [("user", prompt)]},
                        config=config,
                        stream_mode="updates"
                    ):
                        if "agent" in chunk:
                            msg = chunk["agent"]["messages"][-1]
                            if msg.content:
                                st.write(msg.content)
                            if getattr(msg, "tool_calls", None):
                                for tc in msg.tool_calls:
                                    with st.status(f"🛠️ Executing `{tc['name']}`...", state="running") as status:
                                        st.json(tc["args"])
                                        status.update(state="complete")
                        elif "tools" in chunk:
                            msg = chunk["tools"]["messages"][-1]
                            with st.expander(f"🔧 Tool Result: {msg.name}"):
                                st.write(msg.content)
                except Exception as e:
                    st.error(f"Error: {e}")
                
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STREAMING RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_streaming_analysis(pdf_path: str):
    """
    Creates a new thread, streams the full CLO pipeline, displays live agent
    status, then renders the final result cards.
    """
    thread_id = new_thread_id()
    st.session_state.active_thread = thread_id

    # Register this thread immediately so it appears in the sidebar next rerun
    if thread_id not in st.session_state.all_thread_ids:
        st.session_state.all_thread_ids.append(thread_id)

    config = make_config(thread_id)
    pdf_basename = os.path.basename(pdf_path)

    initial_state = {
        "pdf_path": pdf_path,
        "extracted_text_chunks": [],
        "vector_store_id": None,
        "parsed_waterfall": None,
        "senior_tranche_ratio": 0.65,
        "default_correlation": 0.20,
        "base_recovery_rate": 0.65,
        "generated_code": None,
        "simulation_results": None,
        "execution_error": None,
        "critic_feedback": None,
        "iteration_count": 0,
        "max_iterations": MAX_ITERATIONS,
        "status": "pending",
        "final_report": None,
    }

    # ── Check if this PDF is already indexed ──────────────────────────────
    info = get_collection_info(pdf_path)
    is_first_embed = not info["indexed"]
    phase_label = (
        "📥 Embedding PDF into Vector Store (first time — this may take 1–2 min)…"
        if is_first_embed
        else "🤖 Running Multi-Agent CLO Analysis Pipeline…"
    )

    # ── Live streaming status block ───────────────────────────────────────
    collected: dict = {}
    final_state: dict = {}

    with st.status(phase_label, expanded=True) as workflow_status:

        st.write(f"📄 Document: **{pdf_basename}**")
        st.write(f"🔑 Thread ID: `{thread_id[:16]}…`")
        st.markdown("---")

        for chunk in clo_app.stream(
            initial_state,
            config=config,
            stream_mode="updates",
        ):
            # chunk = {node_name: node_output_dict}
            node_name = next(iter(chunk))
            output    = chunk[node_name] or {}

            if node_name not in KNOWN_NODES:
                continue

            # Merge into running state
            collected.update(output)
            final_state.update(output)

            # ── Parser ────────────────────────────────────────────────────
            if node_name == "parser_node":
                pw   = output.get("parsed_waterfall") or {}
                deal = pw.get("deal_name", "Unknown Deal")
                par  = pw.get("total_target_par") or 0
                n_tr = len(pw.get("tranches") or [])
                vec  = output.get("vector_store_id", "—")
                st.write(
                    f"✅ **Parser Agent** — *{deal}* · "
                    f"{fmt_money(par)} par · {n_tr} tranches "
                    f"· collection `{vec[:20]}`"
                )

            # ── Quant ─────────────────────────────────────────────────────
            elif node_name == "quant_node":
                it  = output.get("iteration_count", "?")
                st_ = output.get("status", "")
                err = output.get("execution_error")
                if st_ == "quant_error":
                    st.write(f"⚠️ **Quant Agent** — Code error in iteration {it}, retrying…")
                    if err:
                        st.code(err[:300], language="text")
                else:
                    st.write(f"✅ **Quant Agent** — Simulation executed (iteration {it})")

            # ── Critic ────────────────────────────────────────────────────
            elif node_name == "critic_node":
                it  = output.get("iteration_count", "?")
                st_ = output.get("status", "")
                if st_ == "approved":
                    st.write(f"✅ **Critic Agent** — ✨ Model **APPROVED** on iteration {it}!")
                else:
                    st.write(f"🔄 **Critic Agent** — Rejected (iteration {it}), sending back to Quant…")
                fb = output.get("critic_feedback", "")
                if fb:
                    st.caption(fb[:200] + ("…" if len(fb) > 200 else ""))

            # ── Reporter ─────────────────────────────────────────────────
            elif node_name == "reporter_node":
                st.write("✅ **Reporter Agent** — Executive summary written")

        # Determine final status from collected state
        final_status = final_state.get("status", "unknown")
        label_emoji  = "✅" if final_status == "approved" else "⚠️"
        workflow_status.update(
            label=f"{label_emoji} Pipeline Complete — **{final_status.upper()}**",
            state="complete",
            expanded=False,
        )

    # ── Persist thread metadata ───────────────────────────────────────────
    pw   = final_state.get("parsed_waterfall") or {}
    st.session_state.thread_meta[thread_id] = {
        "deal_name"     : pw.get("deal_name", f"Thread {thread_id[:8]}"),
        "pdf_name"      : pdf_basename,
        "status"        : final_state.get("status", "unknown"),
        "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "vector_store_id": final_state.get("vector_store_id", ""),
        "iterations"    : final_state.get("iteration_count", 0),
    }

    st.divider()
    # ── Render final result cards ─────────────────────────────────────────
    display_full_results(thread_id, final_state)


# ══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════

def show_welcome():
    st.markdown("""
    <div class="hero">
        <h1>📊 CLO Waterfall Simulator</h1>
        <p>Upload a CLO Indenture PDF — multiple AI agents will extract legal rules,
        write a Monte Carlo simulation, stress-test every tranche, and deliver
        an executive summary. All results are persisted per thread so you can
        revisit any past analysis from the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-tile"><div class="metric-val" style="color:#818cf8">📄</div><div class="metric-label">Parser Agent</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-tile"><div class="metric-val" style="color:#22d3ee">📊</div><div class="metric-label">Quant Agent</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-tile"><div class="metric-val" style="color:#f59e0b">⚖️</div><div class="metric-label">Critic Agent</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-tile"><div class="metric-val" style="color:#34d399">📑</div><div class="metric-label">Reporter Agent</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.pending_pdf:
        pdf_name = os.path.basename(st.session_state.pending_pdf)
        st.markdown(f"**Selected:** `{pdf_name}`")
        info = get_collection_info(st.session_state.pending_pdf)

        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            if st.button("🚀  Run Analysis", type="primary", use_container_width=True):
                run_streaming_analysis(st.session_state.pending_pdf)
        with col_info:
            if info["indexed"]:
                st.success(f"🗄️ PDF already indexed ({info.get('index_size_mb', '?')} MB) — analysis will start instantly.")
            else:
                st.info("⏳ First run will embed the full PDF into the vector store before analysis begins.")
    else:
        st.info("← Select or upload a CLO Indenture PDF from the sidebar, then click **Run Analysis**.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTING
# ══════════════════════════════════════════════════════════════════════════════

active = st.session_state.active_thread

if active:
    # ── Viewing a past thread ─────────────────────────────────────────────
    state = load_thread_state(active)
    if state:
        meta = st.session_state.thread_meta.get(active, {})
        # Back button
        if st.button("← New Analysis", key="back_btn"):
            st.session_state.active_thread = None
            st.rerun()
        st.divider()
        
        tab1, tab2 = st.tabs(["📊 Pipeline Results", "💬 AI Assistant"])
        with tab1:
            display_full_results(active, state)
        with tab2:
            display_chat_interface(active, state)
    else:
        st.warning(f"Could not load state for thread `{active}`. It may have been cleared.")
        if st.button("← Back"):
            st.session_state.active_thread = None
            st.rerun()
else:
    # ── No active thread: show welcome + run controls ─────────────────────
    show_welcome()
