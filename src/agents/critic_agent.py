"""
Critic Agent: Evaluates simulation metrics against rating agency loss constraints
and provides structured feedback for structural re-sizing.
"""
import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import GraphState


def critic_agent(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates Monte Carlo simulation results against rating agency standards.
    Determines whether to approve the structure or reject with specific feedback.
    Returns a partial state update dict.
    """
    results = state.get("simulation_results")
    rules = state.get("parsed_waterfall")
    iteration = state.get("iteration_count", 1)

    if not results or not rules:
        return {
            "status": "failed",
            "execution_error": "Missing simulation results or indenture rules in Critic node.",
        }

    # rules is a plain dict (serialized from IndentureRules), access via dict keys
    tranches = rules.get("tranches", [])

    # Identify the Senior tranche (typically Class A-1 / AAA)
    senior_tranche = next(
        (t for t in tranches
         if "aaa" in t.get("target_rating", "").lower()
         or "class a" in t.get("class_name", "").lower()),
        tranches[0] if tranches else None
    )

    if not senior_tranche:
        return {
            "status": "failed",
            "execution_error": "No tranches found in parsed waterfall rules.",
        }

    senior_name = senior_tranche["class_name"]
    senior_loss_prob = results.get(senior_name, 0.0)

    # Hard rating threshold: AAA loss probability must be <= 0.01% (0.0001)
    AAA_LOSS_THRESHOLD = 0.0001

    # Fast evaluation model
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.0
    )

    # If the senior tranche fails the stress test
    if senior_loss_prob > AAA_LOSS_THRESHOLD:
        system_prompt = """You are the Lead Structuring Quant and Credit Evaluator for a major investment bank, tasked with stress-testing a CLO structure and ensuring it meets rating agency criteria.

Your objective is to evaluate the Monte Carlo simulation results provided by the Quant Node and issue structured feedback. If the model fails, you must mandate dynamic, real-world structural adjustments to fix it.

CRITICAL RULES & CONSTRAINTS:
1. NO HARDCODED ASSUMPTIONS: You must ONLY use the tranche sizes, interest rates, Overcollateralization (OC) triggers, Interest Coverage (IC) triggers, and fee caps exactly as they were extracted from the PDF into the `IndentureRules` JSON. Do not invent standard market rates.
2. ZERO-SUM CAPITAL STRUCTURE: The Total Target Par of the collateral pool is strictly fixed. If you instruct the Quant to reduce the size of the Senior Tranche to create a safer OC ratio, you MUST instruct the Quant to increase the Subordinated/Equity Tranche (or a Junior Tranche) by the exact same dollar amount. The total capital structure must always sum to the original total.
3. RATING THRESHOLD: The probability of loss for the most senior tranche (e.g., Class A-1 / AAA) must be <= 0.01% (0.0001).

STRATEGY FOR REJECTION (DYNAMIC STRUCTURING):
If the Senior Tranche fails the stress test (loss > 0.01%), DO NOT simply tell the Quant to blindly lower numbers. You must employ dynamic cash flow structuring strategies used in real CLO Indentures:
*   Strategy A (Interest Diversion / Cash Sweep): Instruct the Quant to implement a strict "cash sweep" mechanism in their Python script. If an OC or IC test is breached during a simulation path, the script must IMMEDIATELY halt all dividend payments (Residual Interest) to the Subordinated/Equity investors.
*   Strategy B (Senior Amortization): Instruct the Quant to redirect those halted equity dividends directly to the Senior Tranche to pay down its principal balance rapidly. 
*   Strategy C (Cure & Resume): Instruct the Quant that once the Senior principal is paid down enough that the OC/IC ratio returns to a passing level, the normal flow of the waterfall should resume.

YOUR OUTPUT FORMAT:
If the results fail, provide step-by-step instructions for the Quant detailing exactly which dynamic strategy (A, B, or C) to code into their Python loop and which tranche sizes to adjust (remembering the zero-sum rule)."""

        results_str = json.dumps(results).replace("{", "{{").replace("}", "}}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", (
                f"Senior Tranche: {senior_name}\n"
                f"Observed Loss Probability: {senior_loss_prob:.4f}\n"
                f"Maximum Allowed Threshold: {AAA_LOSS_THRESHOLD:.4f}\n"
                f"Full Simulation Results: {results_str}\n"
                f"Iteration: {iteration}\n\n"
                "Provide exact structuring strategies and Python logic instructions to the Quant to fix this structure."
            ))
        ])
        
        chain = prompt | llm
        response = chain.invoke({})
        
        return {
            "status": "rejected",
            "critic_feedback": response.content,
        }

    # If the senior tranche passes within acceptable limits
    return {
        "status": "approved",
        "critic_feedback": (
            f"Approved: {senior_name} loss rate is {senior_loss_prob:.4f}, "
            f"well within the rating constraint of <= {AAA_LOSS_THRESHOLD:.4f}."
        ),
    }
