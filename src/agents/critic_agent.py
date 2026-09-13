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
        system_prompt = (
            "You are a Senior Credit Officer at a Major Rating Agency (Moody's / S&P).\n"
            "The CLO model FAILED the AAA stress benchmark.\n"
            "Your task is to give precise, actionable instructions to the Quant to re-structure the capital pool.\n"
            "Advise reducing the Senior tranche size by 2% to 4% and increasing junior/equity subordination."
        )
        
        results_str = json.dumps(results).replace("{", "{{").replace("}", "}}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", (
                f"Senior Tranche: {senior_name}\n"
                f"Observed Loss Probability: {senior_loss_prob:.4f}\n"
                f"Maximum Allowed Threshold: {AAA_LOSS_THRESHOLD:.4f}\n"
                f"Full Simulation Results: {results_str}\n"
                f"Iteration: {iteration}\n\n"
                "Provide brief, concrete mathematical feedback on how to resize the tranches."
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
