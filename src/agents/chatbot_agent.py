"""
Chatbot Agent — Interactive AI Assistant for CLO Workflows.

Provides conversational interface with tools to query the PDF and workflow data.
"""

import os
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from src.tools.retriever import search_indenture, get_or_create_retriever

# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def search_document(query: str, pdf_path: str) -> str:
    """
    Search the currently active CLO Indenture PDF for specific keywords or clauses.
    Use this to answer questions about the legal document itself.
    """
    try:
        retriever = get_or_create_retriever(pdf_path, k=4)
        results = search_indenture(retriever, query)
        if not results:
            return "No matching clauses found in the document."
        return "\n\n".join(results)
    except Exception as e:
        return f"Error searching document: {str(e)}"

@tool
def get_workflow_analysis(thread_id: str) -> str:
    """
    Get the parsed Indenture rules, tranches, and Monte Carlo simulation results for the current deal.
    Use this to answer questions about loss probabilities, tranche sizes, tests, and extracted numbers.
    """
    try:
        from src.main import build_clo_graph
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        import json
        
        # We instantiate a temporary checkpointer just to read the state
        # using check_same_thread=False since this might run in a separate thread from Streamlit
        conn = sqlite3.connect("data/workflow_threads.db", check_same_thread=False)
        cp = SqliteSaver(conn)
        app = build_clo_graph(cp)
        
        config = {"configurable": {"thread_id": thread_id}}
        snap = app.get_state(config)
        if not snap or not snap.values:
            return "No data found for this workflow thread."
            
        data = {
            "parsed_waterfall": snap.values.get("parsed_waterfall"),
            "simulation_results": snap.values.get("simulation_results"),
        }
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error fetching workflow analysis: {str(e)}"

# ══════════════════════════════════════════════════════════════════════════════
# AGENT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_chatbot_graph(checkpointer: Optional[SqliteSaver] = None):
    """
    Builds the chatbot reactive agent graph with tools.
    """
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        max_tokens=2048,
    )

    tools = [search_document, get_workflow_analysis]

    system_prompt = (
        "You are an expert AI CLO (Collateralized Loan Obligation) Assistant. "
        "Your role is to help users understand the structure, tranches, rules, and simulation results "
        "of the active CLO workflow.\n\n"
        "You have access to the following tools:\n"
        "1. `get_workflow_analysis`: Call this tool when asked about simulation results, Monte Carlo loss probabilities, extracted tranche sizes, deal name, or coverage tests. You MUST pass the `thread_id` provided in the System Context.\n"
        "2. `search_document`: Call this tool when asked about specific legal clauses, definitions, or raw text directly from the indenture PDF. You MUST pass the exact `pdf_path` provided in the System Context.\n\n"
        "Combine insights from both tools when necessary. Be concise, professional, and explain complex financial concepts clearly."
    )

    # Use the prebuilt ReAct agent graph which handles tool calling and tool-node execution seamlessly.
    chatbot_app = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
    )
    
    return chatbot_app
