import streamlit as st
import os
import sys
import json

# Add root directory to python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import build_clo_graph, MAX_ITERATIONS
from src.tools.retriever import get_collection_info, delete_collection

st.set_page_config(page_title="Agentic AI CLO Simulator", layout="wide")

st.title("📊 Agentic AI CLO Waterfall Simulator")
st.markdown(
    "Upload a CLO Indenture PDF to have multiple AI Agents extract rules, "
    "write a Monte Carlo simulation, and run stress tests."
)

# ---------------------------------------------------------------------------
# Sidebar — RAG Vector Store Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🗄️ Vector Store Status")
    st.caption(
        "Each PDF is indexed into its own isolated ChromaDB collection. "
        "Switching PDFs never mixes data from different documents."
    )

data_dir = "data/raw_indentures"
os.makedirs(data_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# PDF Selection
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload CLO Indenture PDF", type=["pdf"])
selected_file = st.selectbox(
    "Or select an existing PDF",
    ["None"] + os.listdir(data_dir)
)

pdf_path = None
if uploaded_file is not None:
    pdf_path = os.path.join(data_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved uploaded file: **{uploaded_file.name}**")
elif selected_file != "None":
    pdf_path = os.path.join(data_dir, selected_file)

# ---------------------------------------------------------------------------
# Sidebar — show collection info + Clear Index button for selected PDF
# ---------------------------------------------------------------------------
if pdf_path:
    info = get_collection_info(pdf_path)
    with st.sidebar:
        st.markdown(f"**📄 File:** `{os.path.basename(pdf_path)}`")
        st.markdown(f"**🔑 Collection:** `{info['collection_name']}`")

        if info["indexed"]:
            st.success(f"✅ Indexed ({info.get('index_size_mb', '?')} MB on disk)")
            if st.button("🗑️ Clear Index (Force Re-embed)"):
                delete_collection(pdf_path)
                st.warning("Index cleared. Next run will re-embed the full PDF.")
                st.rerun()
        else:
            st.info("⏳ Not yet indexed — will be built on first run.")

# ---------------------------------------------------------------------------
# Run Workflow
# ---------------------------------------------------------------------------
if pdf_path and st.button("🚀 Run Workflow"):
    st.info(f"Starting Multi-Agent Workflow on `{os.path.basename(pdf_path)}`...")

    app = build_clo_graph()

    initial_state = {
        "pdf_path": pdf_path,
        "extracted_text_chunks": [],
        "vector_store_id": None,
        "parsed_waterfall": {},
        "senior_tranche_ratio": 0.65,
        "default_correlation": 0.20,
        "base_recovery_rate": 0.65,
        "simulation_results": {},
        "status": "pending",
        "critic_feedback": "",
        "execution_error": "",
        "iteration_count": 0,
        "max_iterations": MAX_ITERATIONS,
        "generated_code": "",
        "final_report": "",
    }

    # Check if indexing is needed and show appropriate spinner message
    is_first_index = not get_collection_info(pdf_path)["indexed"]
    spinner_msg = (
        "🔍 Indexing PDF into Vector Store (first time — may take 1–2 min for large docs)..."
        if is_first_index
        else "🤖 Agents are analyzing the PDF and simulating the waterfall..."
    )

    with st.spinner(spinner_msg):
        final_state = app.invoke(initial_state)

    st.success("✅ Workflow Complete!")

    # ---------------------------------------------------------------------------
    # Result Summary
    # ---------------------------------------------------------------------------
    status = final_state.get("status", "unknown")
    col_status, col_iter, col_vec = st.columns(3)

    with col_status:
        if status == "approved":
            st.success(f"**Status: APPROVED**")
        else:
            st.error(f"**Status: {status.upper()}**")

    with col_iter:
        st.metric("Iterations", final_state.get("iteration_count", 0))

    with col_vec:
        vec_id = final_state.get("vector_store_id", "N/A")
        st.info(f"**Collection:** `{vec_id}`")

    # Update sidebar after run
    with st.sidebar:
        updated_info = get_collection_info(pdf_path)
        if updated_info["indexed"]:
            st.success(f"✅ Indexed ({updated_info.get('index_size_mb', '?')} MB on disk)")

    if final_state.get("execution_error"):
        st.error(f"**Execution Error:** {final_state.get('execution_error')}")

    st.write("### 📝 Critic Agent Assessment")
    st.info(final_state.get("critic_feedback", "None provided."))

    if final_state.get("final_report"):
        st.write("---")
        st.write("### 📑 Executive Summary")
        st.markdown(final_state.get("final_report"))

    st.write("---")

    # ---------------------------------------------------------------------------
    # Detailed Panels
    # ---------------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 📄 Parsed Rules (JSON)")
        with st.expander("View Data Extracted from PDF"):
            st.json(final_state.get("parsed_waterfall", {}))

        st.write("### 📈 Tranche Simulation Results")
        with st.expander("View Monte Carlo Output"):
            results = final_state.get("simulation_results", {})
            if isinstance(results, str):
                try:
                    results = json.loads(results)
                except Exception:
                    pass
            st.json(results)

    with col2:
        st.write("### 💻 Generated Python Code")
        with st.expander("View Quant Agent Code"):
            code = final_state.get("generated_code", "No code generated.")
            st.code(code, language="python")

        st.write("### 📦 RAG Retrieval Context")
        with st.expander("View chunks retrieved from Vector Store"):
            chunks = final_state.get("extracted_text_chunks", [])
            if chunks:
                st.text("\n\n".join(chunks[:5]))  # Show first 5 chunk headers to avoid overload
            else:
                st.write("No chunks to display.")
