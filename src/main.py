"""
Main entry point for CLO Waterfall Simulator LangGraph workflow.
"""

import os
import sys
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from src.state import GraphState
from src.agents.parser_agent import parser_agent
from src.agents.quant_agent import quant_agent
from src.agents.critic_agent import critic_agent
from src.agents.reporter_agent import reporter_agent

load_dotenv()

MAX_ITERATIONS = 5


def route_after_parser(state: GraphState) -> str:
    """Routes to Quant if parsing succeeded, else terminates."""
    if state.get("status") == "failed":
        return END
    return "quant_node"


def route_after_quant(state: GraphState) -> str:
    """
    Handles script execution outcomes:
    - If code failed to execute, retry quant up to MAX_ITERATIONS.
    - If code executed, proceed to rating agency evaluation.
    """
    iteration = state.get("iteration_count", 0)

    if state.get("status") == "quant_error":
        if iteration >= MAX_ITERATIONS:
            print(f"[!] Reached max iteration limit ({MAX_ITERATIONS}) with errors. Terminating graph.")
            return END
        print(f"[!] Code syntax/runtime error in iteration {iteration}. Retrying Quant...")
        return "quant_node"

    return "critic_node"


def route_after_critic(state: GraphState) -> str:
    """
    Evaluates approval status:
    - If approved: end graph.
    - If rejected: loop back to Quant with feedback if under iteration cap.
    """
    status = state.get("status")
    iteration = state.get("iteration_count", 0)

    if status == "approved":
        print(f"[✓] Model Approved by Rating Agency Critic at iteration {iteration}!")
        return "reporter_node"

    if iteration >= MAX_ITERATIONS:
        print(f"[!] Convergence limit reached without approval. Terminating graph.")
        return END

    print(f"[→] Model Rejected. Routing back to Quant with feedback (Loop {iteration + 1})...")
    return "quant_node"


def build_clo_graph():
    """Builds and compiles the cyclic LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # 1. Register Nodes
    workflow.add_node("parser_node", parser_agent)
    workflow.add_node("quant_node", quant_agent)
    workflow.add_node("critic_node", critic_agent)
    workflow.add_node("reporter_node", reporter_agent)

    # 2. Register Edges
    workflow.add_edge(START, "parser_node")
    workflow.add_conditional_edges("parser_node", route_after_parser)
    workflow.add_conditional_edges("quant_node", route_after_quant)
    workflow.add_conditional_edges("critic_node", route_after_critic)
    workflow.add_edge("reporter_node", END)

    return workflow.compile()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.main <path_to_indenture_pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    if not os.path.exists(pdf_file):
        print(f"Error: File not found at {pdf_file}")
        sys.exit(1)

    # Initial state keys match GraphState TypedDict exactly
    initial_state: GraphState = {
        "pdf_path": pdf_file,
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

    app = build_clo_graph()
    print(f"Starting CLO Waterfall Analysis on {pdf_file}...")

    final_state = app.invoke(initial_state)

    print("\n--- Final Workflow Results ---")
    print("Execution Error:", final_state.get('execution_error'))
    import json
    print("Parsed rules:", json.dumps(final_state.get('parsed_waterfall'), indent=2))
    print(f"Status: {final_state.get('status')}")
    print(f"Iterations Run: {final_state.get('iteration_count')}")
    print(f"Critic Assessment: {final_state.get('critic_feedback')}")
    print(f"Tranche Results: {final_state.get('simulation_results')}")
    print("\n--- Executive Summary ---")
    print(final_state.get('final_report'))