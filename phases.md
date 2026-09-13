# Execution Phases: CLO Waterfall Simulator

## Phase 1: Foundation (Data Extraction & Structuring)
**Goal:** Successfully convert an unstructured legal PDF into a structured, executable JSON format.
* **Tasks:**
  * Define the target extraction schema using Pydantic (`waterfall_def.py`).
  * Implement PDF parsing (LlamaParse or Unstructured).
  * Build the `parser_agent.py` and test its ability to extract Tranche Sizes, Interest Rates, and OC/IC triggers.
* **Success Criteria:** Given a mock 30-page Indenture, the system reliably outputs a valid JSON object capturing the complete Priority of Payments without hallucinations.

## Phase 2: The Engine (Code Generation & Execution)
**Goal:** Build the `quant_node` capable of writing and executing Monte Carlo simulations based on the JSON constraints.
* **Tasks:**
  * Set up the LangChain `Python_REPL` sandbox tool.
  * Prompt engineer the `quant_agent.py` to write `pandas`/`numpy` code utilizing the Phase 1 JSON.
  * Implement basic Monte Carlo logic (e.g., 1,000 scenarios of varying default rates).
* **Success Criteria:** The Quant node consistently outputs syntactically correct Python that runs in the sandbox and returns a dictionary of loss metrics (e.g., `{"Class A Loss": 0.0%, "Class B Loss": 2.1%}`).

## Phase 3: Orchestration (Critic Node & The Cyclic Graph)
**Goal:** Connect the agents via LangGraph and implement the feedback loop.
* **Tasks:**
  * Build the `critic_agent.py` to evaluate the Quant's output against the loss constraints.
  * Define the `GraphState` TypedDict.
  * Implement the conditional routing logic in `main.py` (Parser -> Quant -> Critic -> Quant / End).
  * Add the `iteration_count` circuit breaker.
* **Success Criteria:** The system successfully completes a multi-iteration loop. When forced to fail initially (by providing a small senior cushion), the Critic correctly forces the Quant to resize the tranches until the loss constraint is satisfied.

## Phase 4: Refinement & Advanced Financial Logic
**Goal:** Upgrade the simulation to handle complex real-world CLO mechanics.
* **Tasks:**
  * Implement CCC-bucket penalty logic in the code generation prompt.
  * Add dynamic recovery rates (e.g., recovery rates drop as default rates spike).
  * Optimize the execution environment (e.g., moving from a local REPL to a fast Dockerized microservice).
* **Success Criteria:** The system accurately models complex edge cases, such as Interest Diversion triggers and Deferred Interest on Mezzanine notes, mirroring the behavior of a professional structuring desk model.