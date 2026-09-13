# src/agents/parser_agent.py
"""
Parser Agent — RAG-backed CLO Indenture Extraction.

Instead of scanning the first 40 pages, this agent:
1. Indexes the entire PDF into an isolated ChromaDB collection (first run only).
2. Runs 5 targeted semantic queries — one per field group in IndentureRules.
3. Passes only the relevant retrieved chunks to the LLM for structured extraction.

This handles 300+ page documents with scattered data across any page,
stays well within LLM context limits, and avoids rate-limit errors.
"""

import os
import logging
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.schemas.waterfall_def import IndentureRules
from src.state import GraphState
from src.tools.retriever import get_or_create_retriever, search_indenture, get_collection_info

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_retries=2,
)

structured_parser_llm = llm.with_structured_output(IndentureRules, method="json_mode")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert structured finance and CLO legal attorney. "
        "Your task is to analyze excerpts retrieved from a CLO Indenture legal document "
        "and extract the financial parameters into the requested JSON schema.\n\n"
        "Strict Extraction Rules:\n"
        "1. Extract ONLY facts explicitly stated in the provided context excerpts.\n"
        "2. If a field is genuinely missing from the excerpts, use the schema default — do NOT fail.\n"
        "3. Find ALL tranches, their exact par amounts, coupon spreads (bps), and target ratings.\n"
        "4. Extract ALL Overcollateralization (OC) and Interest Coverage (IC) test trigger ratios.\n"
        "5. Capture the administrative fee cap and management fee percentages.\n"
        "6. Reconstruct the FULL sequential priority of interest payments (Priority 1, 2, 3...).\n"
        "7. Output must be valid JSON matching the schema EXACTLY.\n\n"
        "TARGET JSON SCHEMA:\n"
        "{schema}"
    )),
    ("human", (
        "Retrieved Indenture Excerpts (semantically relevant to the schema):\n\n"
        "{context}\n\n"
        "Extract the complete IndentureRules JSON:"
    ))
])

# ---------------------------------------------------------------------------
# Targeted RAG Queries
# Each query is crafted to retrieve the most relevant chunks for a specific
# sub-section of the IndentureRules schema.
# ---------------------------------------------------------------------------

RAG_QUERIES = [
    # Tranche capital structure
    (
        "CLO tranche class names principal amounts par value notes issued "
        "Class A Class B Class C subordinated equity ratings AAA AA BBB "
        "spread basis points SOFR floating fixed coupon"
    ),
    # Coverage tests
    (
        "overcollateralization OC test ratio trigger threshold "
        "interest coverage IC test minimum ratio cure action "
        "Class A Class B OC ratio Class C IC ratio diversion"
    ),
    # Fees
    (
        "senior management fee rate subordinated management fee rate "
        "administrative fee cap trustee fee incentive fee hurdle IRR "
        "basis points annual fee percentage collateral"
    ),
    # Priority of payments / waterfall
    (
        "priority of payments interest proceeds waterfall sequential "
        "first second third fourth administrative fees trustee interest "
        "payment distribution order"
    ),
    # CCC bucket, equity, deal identity
    (
        "CCC rated obligation bucket limit percentage "
        "subordinated notes equity residual deal name CLO vehicle "
        "target par collateral balance IRR hurdle incentive share"
    ),
]


def _build_rag_context(retriever) -> str:
    """
    Runs all targeted queries and merges deduplicated results into one context block.
    """
    seen_content = set()
    all_chunks = []

    for i, query in enumerate(RAG_QUERIES, start=1):
        raw = search_indenture(retriever, query)
        # Deduplicate: skip chunks already retrieved by a prior query
        for chunk in raw.split("\n\n---\n\n"):
            # Use first 120 chars as a fingerprint to detect duplicates
            fingerprint = chunk[:120].strip()
            if fingerprint and fingerprint not in seen_content:
                seen_content.add(fingerprint)
                all_chunks.append(f"[Query {i}]\n{chunk}")

    logger.info(f"[RAG] Retrieved {len(all_chunks)} unique chunks across {len(RAG_QUERIES)} queries.")
    return "\n\n===\n\n".join(all_chunks)


# ---------------------------------------------------------------------------
# LangGraph Node
# ---------------------------------------------------------------------------

def parser_agent(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node: Ingests the PDF via RAG, retrieves targeted excerpts,
    and returns a validated IndentureRules parsed from the vector store.
    """
    pdf_path = state.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Indenture PDF not found at path: {pdf_path}")

    # ------------------------------------------------------------------
    # 1. Index the PDF into an isolated ChromaDB collection (lazy — skips
    #    if already indexed for this exact document).
    # ------------------------------------------------------------------
    logger.info(f"[Parser] Initializing RAG retriever for: {os.path.basename(pdf_path)}")
    retriever, collection_name = get_or_create_retriever(pdf_path)

    collection_info = get_collection_info(pdf_path)
    index_status = "reused" if collection_info["indexed"] else "newly built"
    logger.info(f"[Parser] Collection '{collection_name}' {index_status}.")

    # ------------------------------------------------------------------
    # 2. Run 5 targeted semantic queries and build the merged context.
    # ------------------------------------------------------------------
    combined_context = _build_rag_context(retriever)

    # ------------------------------------------------------------------
    # 3. LLM structured extraction from retrieved chunks only.
    # ------------------------------------------------------------------
    chain = EXTRACTION_PROMPT | structured_parser_llm
    schema_str = IndentureRules.schema_json(indent=2)

    logger.info("[Parser] Sending retrieved context to LLM for structured extraction...")
    parsed_rules: IndentureRules = chain.invoke({
        "context": combined_context,
        "schema": schema_str,
    })

    rules_dict = parsed_rules.model_dump()
    logger.info(f"[Parser] Extraction complete. Deal: '{parsed_rules.deal_name}', "
                f"Tranches: {len(parsed_rules.tranches)}, "
                f"Coverage tests: {len(parsed_rules.coverage_tests)}")

    # ------------------------------------------------------------------
    # 4. Derive senior tranche ratio for the Quant Agent's simulation.
    # ------------------------------------------------------------------
    total_par = parsed_rules.total_target_par
    senior_tranche = next(
        (t for t in parsed_rules.tranches if "A" in t.class_name and not t.is_equity),
        None
    )
    senior_ratio = (
        round(senior_tranche.principal_amount / total_par, 4)
        if senior_tranche and total_par > 0
        else 0.65  # Fallback: 65% senior AAA (market convention)
    )

    # ------------------------------------------------------------------
    # 5. Build the extracted_text_chunks list for display in the UI.
    #    We expose the raw retrieved chunks so the frontend can show them.
    # ------------------------------------------------------------------
    extracted_chunks = [
        f"[RAG Collection: {collection_name}]",
        f"[Index status: {index_status}]",
        combined_context,
    ]

    return {
        "extracted_text_chunks": extracted_chunks,
        "parsed_waterfall": rules_dict,
        "senior_tranche_ratio": senior_ratio,
        "vector_store_id": collection_name,
    }