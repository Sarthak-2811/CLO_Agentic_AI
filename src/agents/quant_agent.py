"""
Quant Agent: Reads parsed JSON waterfall definition and generates Python simulation scripts
to run Monte Carlo portfolio default & cash flow calculations.
"""

import json
import logging
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import GraphState
from src.tools.python_repl import execute_simulation_code

logger = logging.getLogger(__name__)


def _sanitize_rules(rules: dict) -> dict:
    """
    Fill in None / null values and ensure at least one tranche exists so the
    Quant agent always has valid data to simulate.
    """
    import copy
    rules = copy.deepcopy(rules)

    total_par = rules.get("total_target_par") or 694_100_000.0  # sum of the 4 known tranches
    rules["total_target_par"] = total_par

    tranches = rules.get("tranches") or []

    # If no tranches at all, create a standard 4-tranche CLO structure as fallback
    if not tranches:
        logger.warning("[Quant] No tranches from parser — using standard CLO fallback structure.")
        tranches = [
            {"class_name": "Class A-1", "principal_amount": 399_000_000.0,
             "spread_bps": 158.0, "coupon_type": "floating", "is_equity": False, "target_rating": "AAA"},
            {"class_name": "Class A-2", "principal_amount": 35_000_000.0,
             "spread_bps": 175.0, "coupon_type": "floating", "is_equity": False, "target_rating": "AA"},
            {"class_name": "Class B",   "principal_amount": 42_000_000.0,
             "spread_bps": 185.0, "coupon_type": "floating", "is_equity": False, "target_rating": "A"},
            {"class_name": "Subordinated Notes", "principal_amount": 218_100_000.0,
             "spread_bps": None, "coupon_type": "fixed", "is_equity": True, "target_rating": "NR"},
        ]

    # Fill in any None values on existing tranches
    DEFAULT_RATIOS = [0.57, 0.05, 0.06, 0.31]
    n = len(tranches)
    for idx, t in enumerate(tranches):
        if t.get("principal_amount") is None:
            ratio = DEFAULT_RATIOS[idx] if idx < len(DEFAULT_RATIOS) else (1.0 / n)
            t["principal_amount"] = round(total_par * ratio, 0)
        if t.get("spread_bps") is None and not t.get("is_equity"):
            t["spread_bps"] = 150.0

    rules["tranches"] = tranches

    # FeeStructure fallbacks
    fees = rules.get("fees") or {}
    fees.setdefault("senior_admin_fee_cap",        200_000.0)
    fees.setdefault("senior_mgmt_fee_rate",        0.0015)
    fees.setdefault("subordinated_mgmt_fee_rate",  0.0035)
    fees.setdefault("incentive_fee_hurdle_irr",    0.12)
    fees.setdefault("incentive_fee_share",         0.20)
    for k, v in fees.items():
        if v is None:
            fees[k] = {"senior_admin_fee_cap": 200_000.0, "senior_mgmt_fee_rate": 0.0015,
                       "subordinated_mgmt_fee_rate": 0.0035, "incentive_fee_hurdle_irr": 0.12,
                       "incentive_fee_share": 0.20}[k]
    rules["fees"] = fees

    rules["ccc_bucket_limit"] = rules.get("ccc_bucket_limit") or 0.075

    return rules


def quant_agent(state: GraphState) -> Dict[str, Any]:
    """
    Generates and executes a Python Monte Carlo simulation based on Indenture Rules.
    Returns a partial state update dict.
    """
    rules = state.get("parsed_waterfall")
    if not rules:
        return {"status": "failed", "execution_error": "No indenture rules found in state."}

    # ── Sanitize rules: fill in None principal_amounts with estimated fallbacks ──
    # The LLM may return null for principal_amount when data is scattered across the PDF.
    # We use a proportional fallback so the Quant agent always has valid float values.
    rules = _sanitize_rules(rules)

    # Prepare inputs — rules is already a dict from the parser's .model_dump()
    rules_json = json.dumps(rules, indent=2)
    critic_feedback = state.get("critic_feedback") or "None. This is the initial run."
    execution_error = state.get("execution_error") or "None."
    iteration = state.get("iteration_count", 0)

    # Groq LLM (Using 20B for strong coding logic and higher rate limits)
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.0
    )

    system_prompt = """You are an elite Quantitative Developer at a top-tier Investment Bank.
Your job is to write a standalone Python script to run a Monte Carlo simulation of a CLO Waterfall.

RULES:
1. You MUST use `numpy` and `pandas`. Simulate 10,000 paths (n_paths=10000).
2. Use the provided JSON Indenture Rules to set tranche sizes, triggers, and spreads.
3. Calculate the probability of default (dollar loss > 0) for EACH tranche.
4. IMPORTANT WATERFALL LOSS ORDER: Losses are absorbed BOTTOM-UP. 
   - Subordinated (Equity) takes the FIRST losses (Threshold = 0).
   - Mezzanine/Junior tranches (e.g. Class C, then Class B) take the NEXT losses.
   - Senior tranches (e.g. Class A-1 / AAA) take the LAST losses.
   Do NOT reverse this order. Your cumulative thresholds for loss absorption must reflect this bottom-up sequence.
5. DO NOT make network calls. Assume a flat default correlation matrix. 
6. ASSUME a 2.0% base annual default probability for the underlying loans and a 70% recovery rate upon default.
7. You MUST print the final result to standard output using `print(json.dumps(results_dict))`. The keys must be the tranche class names, and the values must be the probability of loss (a float between 0.0 and 1.0).
8. DO NOT embed the raw JSON string in your code. Instead, extract the specific tranche sizes, spreads, and triggers from the provided rules and define them directly as Python variables (e.g. `tranche_A_size = 100000000`).
9. Output ONLY pure Python code. Do not wrap it in markdown block quotes (```python). Just the code.
10. You MUST include `import json`, `import numpy as np`, and `import pandas as pd` at the top of your script.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Indenture Rules:\n{rules}\n\nCritic Feedback:\n{feedback}\n\nExecution Errors from previous attempt:\n{errors}\n\nWrite the Python simulation code.")
    ])

    chain = prompt | llm
    
    # 1. Generate the Code
    response = chain.invoke({
        "rules": rules_json,
        "feedback": critic_feedback,
        "errors": execution_error
    })
    
    # Clean up markdown if the LLM accidentally included it
    generated_code = response.content.replace("```python", "").replace("```", "").strip()

    # 2. Execute the Code in the Sandbox
    success, results, error_msg = execute_simulation_code(generated_code)

    if not success:
        # If code failed to run, loop back to the Quant Agent (via the conditional router)
        return {
            "generated_code": generated_code,
            "execution_error": error_msg,
            "status": "quant_error",  # Router will send this back to the Quant
            "iteration_count": iteration + 1,
        }

    # 3. Successful Execution
    return {
        "generated_code": generated_code,
        "simulation_results": results,
        "execution_error": None,
        "iteration_count": iteration + 1,
        "status": "simulated",  # Router will send this to the Critic
    }