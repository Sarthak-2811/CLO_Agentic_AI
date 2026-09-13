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

    system_prompt = """You are a senior financial analyst and communicator. 
Your job is to read the final results of a CLO Waterfall Monte Carlo stress test and explain it in simple, plain English.

The user wants to know:
1. Does the final workflow work and is the deal safe?
2. What structural changes (iterations) were tried before arriving at the final solution?
3. A basic explanation of the final loss attribution across the tranches.

Format your output beautifully using Markdown headers, bullet points, and bold text where appropriate.
Avoid overly complex jargon where possible, or explain it simply if you must use it.
Be concise but comprehensive."""

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
