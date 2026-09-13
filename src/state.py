# src/state.py
from typing import TypedDict, List, Optional, Dict, Any


class GraphState(TypedDict):
    """Shared state for the CLO Waterfall LangGraph workflow."""
    # Document inputs
    pdf_path: str
    extracted_text_chunks: List[str]

    # Extracted legal structure (from Parser Node)
    # Stored as a plain dict (Pydantic model serialized via .model_dump())
    parsed_waterfall: Optional[Dict[str, Any]]

    # Structuring & simulation tuning (altered dynamically by feedback loops)
    senior_tranche_ratio: float           # e.g., 0.65 = 65% senior debt
    default_correlation: float            # e.g., 0.20 asset correlation
    base_recovery_rate: float             # e.g., 0.65 recovery upon default

    # Generated simulation code & execution outputs (from Quant Node)
    generated_code: Optional[str]
    simulation_results: Optional[Dict[str, Any]]
    execution_error: Optional[str]

    # Review & routing signals (from Critic Node)
    critic_feedback: Optional[str]
    iteration_count: int
    max_iterations: int
    status: str                           # 'pending', 'approved', or 'rejected'
    
    # Final Executive Summary (from Reporter Node)
    final_report: Optional[str]