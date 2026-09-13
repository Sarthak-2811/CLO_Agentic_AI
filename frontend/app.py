import streamlit as st
import os
import json
import sys

# Add root directory to python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import build_clo_graph, MAX_ITERATIONS

st.set_page_config(page_title="Agentic AI CLO Simulator", layout="wide")

st.title("📊 Agentic AI CLO Waterfall Simulator")
st.markdown("Upload a CLO Indenture PDF to have multiple AI Agents extract rules, write a Monte Carlo simulation, and run stress tests.")

data_dir = "data/raw_indentures"
os.makedirs(data_dir, exist_ok=True)

uploaded_file = st.file_uploader("Upload CLO Indenture PDF", type=["pdf"])
selected_file = st.selectbox("Or select an existing PDF", ["None"] + os.listdir(data_dir))

pdf_path = None
if uploaded_file is not None:
    pdf_path = os.path.join(data_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved uploaded file: {uploaded_file.name}")
elif selected_file != "None":
    pdf_path = os.path.join(data_dir, selected_file)

if pdf_path and st.button("🚀 Run Workflow"):
    st.info(f"Starting Multi-Agent Workflow on `{os.path.basename(pdf_path)}`...")
    
    app = build_clo_graph()
    
    # Define initial state
    initial_state = {
        "pdf_path": pdf_path,
        "parsed_waterfall": {},
        "simulation_results": {},
        "status": "pending",
        "critic_feedback": "",
        "execution_error": "",
        "iteration_count": 0,
        "max_iterations": MAX_ITERATIONS,
        "generated_code": "",
        "final_report": "",
    }
    
    with st.spinner("Agents are analyzing the PDF and simulating the waterfall... (This may take a minute)"):
        final_state = app.invoke(initial_state)
    
    st.success("Workflow Complete!")
    
    # Extract final results
    status = final_state.get("status", "unknown")
    if status == "approved":
        st.success(f"Status: **APPROVED** by Critic Agent on iteration {final_state.get('iteration_count')}")
    else:
        st.error(f"Status: **{status.upper()}** (Iteration: {final_state.get('iteration_count')})")
        
    if final_state.get("execution_error"):
        st.error(f"Execution Error: {final_state.get('execution_error')}")
        
    st.write("### 📝 Critic Agent Assessment")
    st.info(final_state.get("critic_feedback", "None provided."))
    
    if final_state.get("final_report"):
        st.write("---")
        st.write("### 📑 Executive Summary")
        st.markdown(final_state.get("final_report"))

    st.write("---")
    
    # Expandable sections for detailed data
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
                except:
                    pass
            st.json(results)
            
    with col2:
        st.write("### 💻 Generated Python Code")
        with st.expander("View Quant Agent Code"):
            code = final_state.get("generated_code", "No code generated.")
            st.code(code, language="python")
