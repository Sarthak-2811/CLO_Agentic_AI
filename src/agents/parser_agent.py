# src/agents/parser_agent.py
import os
import pdfplumber
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.schemas.waterfall_def import IndentureRules
from src.state import GraphState

load_dotenv()

# Initialize high-reasoning model for extraction
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_retries=2
)

# Bind Pydantic schema directly to Groq's structured output using json_mode
structured_parser_llm = llm.with_structured_output(IndentureRules, method="json_mode")

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert structured finance and CLO legal attorney. "
        "Your task is to analyze excerpts from a CLO Indenture legal document "
        "and extract the financial parameters into the requested schema.\n\n"
        "Strict Extraction Rules:\n"
        "1. Extract facts explicitly stated in the context.\n"
        "2. If some fields (like deal_name, total_target_par, tranches, fees) are missing from the excerpt, omit them so the schema defaults take over. Do NOT fail the extraction.\n"
        "3. Find all tranches, their exact par amounts, coupon spreads (bps), and target ratings (if present).\n"
        "4. Extract all Overcollateralization (OC) and Interest Coverage (IC) test trigger ratios.\n"
        "5. Capture the administrative fee cap and management fee percentages (if present).\n"
        "6. Reconstruct the sequential priority of interest payments (Priority 1, 2, 3, etc.).\n"
        "7. Output must be a valid JSON object matching the requested schema EXACTLY.\n\n"
        "TARGET JSON SCHEMA:\n"
        "{schema}"
    )),
    ("human", "Indenture Legal Text:\n\n{context}\n\nExtract the structured IndentureRules in JSON format matching the schema:")
])

def extract_relevant_pages(pdf_path: str, max_pages_to_search: int = 40) -> List[str]:
    """
    Scans the PDF for the key sections: Priority of Payments,
    Summary of Terms, and Coverage Tests to avoid feeding 300+ pages to the LLM.
    """
    relevant_chunks = []
    keywords = [
        "priority of payments",
        "priority of interest proceeds",
        "overcollateralization ratio",
        "coverage test",
        "principal amount",
        "spread over sofr"
    ]

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = min(len(pdf.pages), max_pages_to_search)
        for idx in range(total_pages):
            page_text = pdf.pages[idx].extract_text() or ""
            text_lower = page_text.lower()
            if any(k in text_lower for k in keywords):
                relevant_chunks.append(f"--- PAGE {idx + 1} ---\n" + page_text)

    return relevant_chunks


def parser_agent(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node: Ingests the PDF path, extracts the critical sections,
    and returns a partial state update with the validated IndentureRules.
    """
    pdf_path = state.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Indenture PDF not found at path: {pdf_path}")

    # Extract relevant text chunks
    chunks = extract_relevant_pages(pdf_path)
    combined_context = "\n\n".join(chunks[:12])  # Limit to most relevant pages to stay well under token caps

    # Run structured extraction
    chain = EXTRACTION_PROMPT | structured_parser_llm
    schema_str = IndentureRules.schema_json(indent=2)
    parsed_rules: IndentureRules = chain.invoke({"context": combined_context, "schema": schema_str})

    # Convert Pydantic model to dict for LangGraph state serialization
    rules_dict = parsed_rules.model_dump()

    # Initialize structuring ratio based on extracted senior tranche size
    total_par = parsed_rules.total_target_par
    senior_tranche = next((t for t in parsed_rules.tranches if "A" in t.class_name), None)
    if senior_tranche and total_par > 0:
        senior_ratio = round(senior_tranche.principal_amount / total_par, 4)
    else:
        senior_ratio = 0.65  # Fallback default (65% Senior AAA)

    # Return partial state update (LangGraph merges this into the full state)
    return {
        "extracted_text_chunks": chunks,
        "parsed_waterfall": rules_dict,
        "senior_tranche_ratio": senior_ratio,
    }