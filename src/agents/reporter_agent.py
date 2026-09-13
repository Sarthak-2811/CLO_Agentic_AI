"""
Reporter Agent: Generates a plain English summary of the finalized CLO structure, 
explaining the iterations, safety, and final Monte Carlo results.
"""

import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import GraphState

def reporter_agent(state: GraphState) -> Dict[str, Any]:
    """
    Generates a final executive summary of the successful CLO model.
    Returns a partial state update dict containing `final_report`.
    """
    rules = state.get("parsed_waterfall")
    results = state.get("simulation_results")
    iterations = state.get("iteration_count")
    feedback = state.get("critic_feedback")

    rules_json = json.dumps(rules, indent=2) if rules else "{}"
    results_json = json.dumps(results, indent=2) if results else "{}"

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3
    )

    system_prompt = """You are a Lead Quantitative Structurer and CLO Modeling Expert at a top-tier investment bank. Your task is to analyze the complete execution log of a LangGraph multi-agent CLO simulation loop and generate a highly structured, data-dense post-execution report. 

You must synthesize the initial legal constraints, the Monte Carlo loss probabilities, and every iterative adjustment made by the Quant and Critic agents. 

Your response MUST STRICTLY follow the exact structure and headings below. Do not deviate from this format.

## 1. Executive Summary & Final Verdict
*   **Deal Status:** [Approved / Rejected / Failed to Converge]
*   **Total Iterations Required:** [Number]
*   **Final Loss Probabilities:** Briefly state the final expected loss for Senior, Junior, and Equity tranches.
*   **Bottom Line:** One paragraph summarizing if the deal is safe, who bears the risk, and why the final structure was accepted.

## 2. The Delta: Initial vs. Final Structure
Compare the starting parameters (Iteration 1) against the approved parameters (Final Iteration). Use a markdown table to display the numerical shifts. 
*   Include Tranche Sizes, Equity Cushion, Overcollateralization (OC) Triggers, and Fee Caps. 
*   Follow the table with a brief analysis of the most critical structural pivot that saved the deal.

## 3. Iteration-by-Iteration Breakdown
Provide a granular, step-by-step log of the optimization journey. For every iteration, you must define:
*   **Iteration [X]:**
    *   **Starting Condition:** (e.g., Senior tranche at $500M)
    *   **Failure Point:** Exactly what broke? (e.g., OC-test for A-Group failed in 12% of paths).
    *   **Agent Adjustment:** What specific numerical tweaks did the Quant make to fix it?

## 4. Macroeconomic Scenario Analysis
Analyze how the FINAL approved structure will perform across two distinct market environments based on the simulation data:
*   **Base Case (Normal Market):** Describe the expected cash flow, fee generation, and yield behavior when default rates sit at historic averages (e.g., 2-3%).
*   **Severe Recession (High Default Cycle):** Describe exactly how the mechanics hold up when defaults spike (e.g., 10-15%). Detail how the cliff-effects behave, when principal diversions trigger, and how fast the Equity gets wiped out to protect the Senior notes.

## 5. Strategic Recommendations & Areas for Improvement
Provide 2-3 actionable insights for the deal team. Look beyond just passing the rating agency constraints. 
*   Could we increase the Senior tranche size slightly to optimize the spread arbitrage?
*   Are the fee caps too restrictive for the manager? 
*   Is the Junior cushion tranche priced correctly for the risk it takes?

Maintain a highly professional, analytical, and objective tone throughout. Ground all statements in the specific numerical data provided in the execution log."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Final Parsed Rules:\n{rules}\n\nFinal Simulation Results (Loss Probabilities):\n{results}\n\nTotal Iterations Required to Pass:\n{iterations}\n\nFinal Critic Feedback:\n{feedback}\n\nWrite the Executive Summary Report.")
    ])

    chain = prompt | llm
    
    response = chain.invoke({
        "rules": rules_json,
        "results": results_json,
        "iterations": iterations,
        "feedback": feedback
    })
    
    return {
        "final_report": response.content
    }
