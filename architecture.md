# System Architecture: CLO Waterfall Simulator

## 1. Workflow
1. **Document Ingestion:** The user uploads the CLO Indenture (PDF).
2. **Parsing & Structuring (Node 1):** The Indenture Parser searches the document, extracts the Waterfall sequence and numeric thresholds, and outputs a validated JSON object to the `GraphState`.
3. **Model Construction (Node 2 - Generation):** The Cash Flow Quant reads the JSON and writes a Python script to simulate portfolio defaults and calculate the cash flow cascade.
4. **Execution (Node 2 - Tool Call):** The Quant executes the script in a secure Python REPL sandbox, returning a statistical summary of tranche losses to the `GraphState`.
5. **Evaluation (Node 3):** The Rating Agency Critic reviews the results. 
   * If Senior tranche loss > limit: Output `rejected` + `critic_feedback` (e.g., "Reduce Senior tranche by 2%").
   * If Senior tranche loss <= limit: Output `approved`.
6. **Conditional Routing:** 
   * If `rejected` (and iteration < max_loops), route back to Node 2. The Quant integrates the feedback, adjusts the code, and reruns.
   * If `approved`, terminate the graph and surface the final optimal capital structure and codebase.

## 2. Architecture Structure (LangGraph)
* **GraphState (Shared Memory):**
  * `pdf_context`: List[str]
  * `parsed_waterfall`: Dict (JSON schema of rules)
  * `generated_code`: str
  * `simulation_results`: Dict (Metrics like Probability of Default per tranche)
  * `critic_feedback`: str (Instructions for the next loop)
  * `iteration_count`: int
  * `status`: str ('approved' or 'rejected')

* **Nodes:**
  * `parser_node`: LLM configured for strict JSON output via function calling.
  * `quant_node`: LLM equipped with a `Python_REPL` tool.
  * `critic_node`: Deterministic logic or LLM configured for strict binary evaluation and parameter adjustment.

* **Edges:**
  * Linear: `START` -> `parser_node` -> `quant_node` -> `critic_node`
  * Conditional: `critic_node` -> `quant_node` (if rejected) OR `END` (if approved).

## 3. Tech Stack (100% Free & Open-Source Pipeline)
* **Orchestration:** LangGraph (Cyclic Graph state & routing), LangChain (`langchain-groq`).
* **LLM Engine (Inference via Groq Free Tier):**
  * `llama-3.3-70b-versatile`: For Indenture Parsing (complex structured JSON extraction) and Cash Flow Quant (Python code generation).
  * `llama-3.1-8b-instant`: For Rating Agency Critic (ultra-fast evaluation, high rate-limit tolerance for loop retries).
* **Document Processing:** `pypdf` or `pdfplumber` (Local, zero-cost parsing for text and financial tables).
* **Embeddings & Vector Store:**
  * Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (Runs locally on CPU via HuggingFace, no API costs).
  * Vector Database: ChromaDB (Local file-based vector storage).
* **Simulation Engine / Math:** Python 3.11+, `numpy`, `pandas`.
* **Execution Sandbox:** LangChain `PythonREPL` tool (sandboxed local runtime with OS/system call restrictions).

## 4. Folder Structure
```text
clo-simulator/
├── data/
│   ├── raw_indentures/        # Uploaded PDFs
│   └── vector_store/          # Local Chroma DB
├── src/
│   ├── __init__.py
│   ├── main.py                # Graph initialization and entry point
│   ├── state.py               # GraphState TypedDict definition
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── parser_agent.py    # Indenture extraction logic
│   │   ├── quant_agent.py     # Code generation logic
│   │   └── critic_agent.py    # Evaluation logic
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── python_repl.py     # Secure execution environment
│   │   └── retriever.py       # RAG logic for the PDF
│   └── schemas/
│       ├── __init__.py
│       └── waterfall_def.py   # Pydantic models for JSON extraction
├── notebooks/
│   └── graph_viz.ipynb        # Jupyter notebook to test and visualize the graph
├── requirements.txt
└── .env