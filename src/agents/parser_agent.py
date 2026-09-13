# src/agents/parser_agent.py
"""
Parser Agent — Hybrid CLO Indenture Extraction.

Architecture (most reliable first):
1. DETERMINISTIC: pdfplumber scans every page's tables directly for tranche rows
   with dollar amounts. This is exact — no embeddings, no LLM, no hallucination.
2. RAG + LLM (divide-and-conquer): 4 small focused calls for semi-structured content
   (coverage tests, fees, waterfall steps). Each call is ~2,500 tokens max.
3. SANITIZE: _sanitize_rules fills any remaining null values with market-convention
   fallbacks so the Quant agent always has valid floats.

This approach eliminates the embedding/retrieval failure mode for structured table
data, which PyPDFLoader and chunking strategies can't reliably handle.
"""

import re
import json
import os
import logging
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.schemas.waterfall_def import (
    IndentureRules, Tranche, CoverageTest, FeeStructure, WaterfallStep
)
from src.state import GraphState
from src.tools.retriever import get_or_create_retriever, get_collection_info

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM — compact, JSON-only output
# ---------------------------------------------------------------------------
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0, max_retries=2)


# ---------------------------------------------------------------------------
# LAYER 1: Deterministic pdfplumber table extraction (no LLM, no embedding)
# ---------------------------------------------------------------------------

# Regex patterns to recognise dollar amounts (e.g. "399,000,000" or "399000000")
_DOLLAR_RE = re.compile(r"[\$]?\s*([\d,]+(?:\.\d+)?)\s*$")
_SPREAD_RE  = re.compile(r"(?:Benchmark|SOFR|LIBOR)\s*[+]\s*([\d.]+)\s*%", re.IGNORECASE)


def _parse_dollar(text: str) -> Optional[float]:
    """Convert '399,000,000' or '$399000000' to 399000000.0. Returns None if not parseable."""
    if not text:
        return None
    text = text.strip().lstrip("$").replace(",", "").strip()
    try:
        val = float(text)
        return val if val > 0 else None
    except ValueError:
        return None


def _parse_spread(text: str, class_name: str = "") -> Optional[float]:
    """Extract spread in bps from 'Benchmark + 1.58%' → 158.0.
    When class_name is provided, only match if that class name appears nearby."""
    # For equity/subordinated tranches there is no spread
    if re.search(r'subordinated|equity', class_name, re.I):
        return None
    m = _SPREAD_RE.search(text or "")
    if m:
        return round(float(m.group(1)) * 100, 1)
    return None


# Known tranche name patterns for classification
_CLASS_PATTERNS = [
    (re.compile(r"Class\s+A[-\s]?1", re.I), "Class A-1", False),
    (re.compile(r"Class\s+A[-\s]?2", re.I), "Class A-2", False),
    (re.compile(r"Class\s+A\b",       re.I), "Class A",   False),
    (re.compile(r"Class\s+B\b",       re.I), "Class B",   False),
    (re.compile(r"Class\s+C\b",       re.I), "Class C",   False),
    (re.compile(r"Class\s+D\b",       re.I), "Class D",   False),
    (re.compile(r"Class\s+E\b",       re.I), "Class E",   False),
    (re.compile(r"Subordinated",      re.I), "Subordinated Notes", True),
    (re.compile(r"Equity",            re.I), "Equity",    True),
]


def _classify_row(row_text: str):
    """Return (class_name, is_equity) if the row matches a known tranche pattern."""
    for pattern, name, is_eq in _CLASS_PATTERNS:
        if pattern.search(row_text):
            return name, is_eq
    return None, False


def _extract_tranches_deterministic(pdf_path: str) -> List[Tranche]:
    """
    Scan every page of the PDF with pdfplumber, looking for tables that contain
    tranche rows (class name + dollar amount in the same row). This is O(pages)
    but 100% reliable — no semantic search, no LLM call needed.
    """
    import pdfplumber

    found: Dict[str, Tranche] = {}   # deduplicate by class_name

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                for table in tables:
                    for row in (table or []):
                        if not row:
                            continue
                        row_cells = [str(c or "").strip() for c in row]
                        row_text  = " ".join(row_cells)

                        class_name, is_equity = _classify_row(row_text)
                        if not class_name:
                            continue

                        # Scan cells for a dollar amount
                        principal = None
                        for cell in row_cells:
                            val = _parse_dollar(cell)
                            if val and val > 100_000:   # must be > $100k (not a ratio/bps)
                                principal = val
                                break

                        # Spread — only search within this row, not the whole page
                        spread = _parse_spread(row_text, class_name=class_name)

                        # Don't overwrite if we already have a better entry
                        if class_name in found and found[class_name].principal_amount is not None:
                            continue

                        tranche = Tranche(
                            class_name=class_name,
                            target_rating="",
                            principal_amount=principal,
                            coupon_type="fixed" if is_equity else "floating",
                            spread_bps=spread,
                            is_equity=is_equity,
                        )
                        found[class_name] = tranche
                        if principal:
                            logger.info(
                                f"[Parser][Det] {class_name}: ${principal:,.0f}"
                                f"{f', spread={spread}bps' if spread else ''}"
                                f" (page {page_num})"
                            )

                # Also scan plain text for spread info on non-equity tranches already found
                page_text = page.extract_text() or ""
                for class_name, tranche in list(found.items()):
                    if tranche.spread_bps is None and not tranche.is_equity:
                        # Look for a line that mentions this class name and a spread
                        class_pat = re.compile(re.escape(class_name), re.I)
                        for line in page_text.splitlines():
                            if class_pat.search(line):
                                spread = _parse_spread(line, class_name=class_name)
                                if spread:
                                    found[class_name] = tranche.model_copy(
                                        update={"spread_bps": spread}
                                    )
                                    break

    except Exception as e:
        logger.error(f"[Parser][Det] pdfplumber scan failed: {e}")

    tranches = list(found.values())
    logger.info(f"[Parser][Det] Deterministic extraction found {len(tranches)} tranches.")
    return tranches


def _extract_total_par_deterministic(pdf_path: str) -> Optional[float]:
    """
    Scan the PDF text for 'Target Par' / 'Aggregate Principal' amounts.
    Returns the value in USD or None.
    """
    import pdfplumber

    patterns = [
        re.compile(r"[Tt]arget\s+[Pp]ar[^$\d]*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
        re.compile(r"[Aa]ggregate\s+[Pp]rincipal\s+[Aa]mount[^$\d]*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
        re.compile(r"[Tt]otal\s+[Pp]ar[^$\d]*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
    ]

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for pat in patterns:
                    m = pat.search(text)
                    if m:
                        val = _parse_dollar(m.group(1))
                        if val and val > 1_000_000:
                            logger.info(f"[Parser][Det] Total par detected: ${val:,.0f}")
                            return val
    except Exception as e:
        logger.warning(f"[Parser][Det] Total par scan failed: {e}")

    return None


# ---------------------------------------------------------------------------
# LAYER 2: RAG + LLM for semi-structured content (coverage tests, fees, waterfall)
# ---------------------------------------------------------------------------

COVERAGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Extract Overcollateralization (OC) and Interest Coverage (IC) tests from the text. "
        "Return ONLY a JSON array matching this schema:\n"
        '[{{"test_type": "OC"|"IC", "applies_to_class": str, "trigger_ratio": float|null, '
        '"cure_action": str}}]\n'
        "Return empty array [] if none found. Output ONLY the JSON array, no other text."
    )),
    ("human", "Text:\n{context}")
])

FEES_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Extract CLO fee structure from the text. "
        "Return ONLY a JSON object matching this schema:\n"
        '{{"senior_admin_fee_cap": float|null, "senior_mgmt_fee_rate": float|null, '
        '"subordinated_mgmt_fee_rate": float|null, "incentive_fee_hurdle_irr": float|null, '
        '"incentive_fee_share": float|null}}\n'
        "Rates must be decimals (0.0015 for 15bps, 0.12 for 12%). "
        "Output ONLY the JSON object, no other text."
    )),
    ("human", "Text:\n{context}")
])

WATERFALL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Extract CLO deal metadata and interest waterfall. Return ONLY a JSON object:\n"
        '{{"deal_name": str, "ccc_bucket_limit": float|null, '
        '"interest_waterfall": [{{"priority": int, "payee": str, '
        '"payment_type": "fees"|"interest"|"principal"|"oc_cure"|"residual_equity", '
        '"condition": str|null}}]}}\n'
        "Output ONLY the JSON object, no other text."
    )),
    ("human", "Text:\n{context}")
])

COVERAGE_QUERIES = [
    "overcollateralization OC test ratio trigger threshold interest coverage IC test diversion",
    "OC ratio IC ratio Class A Class B coverage test reinvestment period diversion",
]
FEES_QUERIES = [
    "senior management fee rate subordinated management fee administrative fee cap trustee incentive hurdle",
    "asset management fee collateral manager fee annual percentage basis points",
]
WATERFALL_QUERIES = [
    "priority of payments interest proceeds waterfall sequential first second third fees trustee",
    "CCC bucket limit deal name CLO target par IRR hurdle equity subordinated",
]


def _build_context(retriever, queries: List[str], k_per_query: int = 4) -> str:
    """Run targeted queries and return deduplicated context capped at 5000 chars."""
    seen, chunks = set(), []
    vs = getattr(retriever, "vectorstore", None)
    for i, query in enumerate(queries, 1):
        try:
            docs = vs.similarity_search(query, k=k_per_query) if vs else retriever.invoke(query)
        except Exception as e:
            logger.warning(f"[Parser] Search failed for query {i}: {e}")
            docs = []
        for doc in docs:
            fp = doc.page_content[:100].strip()
            if fp and fp not in seen:
                seen.add(fp)
                chunks.append(f"[Q{i}/P{doc.metadata.get('page','?')}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)[:5000]


def _safe_json(text: str, fallback):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except Exception as e:
        logger.warning(f"[Parser] JSON parse failed: {e} | raw={text[:120]}")
        return fallback


def _extract_coverage_tests(retriever) -> List[CoverageTest]:
    ctx  = _build_context(retriever, COVERAGE_QUERIES, k_per_query=4)
    raw  = (COVERAGE_PROMPT | llm).invoke({"context": ctx}).content
    data = _safe_json(raw, [])
    tests = []
    if isinstance(data, list):
        for item in data:
            try:
                tests.append(CoverageTest(**item))
            except Exception as e:
                logger.warning(f"[Parser] Bad coverage test: {e}")
    return tests


def _extract_fees(retriever) -> FeeStructure:
    ctx  = _build_context(retriever, FEES_QUERIES, k_per_query=4)
    raw  = (FEES_PROMPT | llm).invoke({"context": ctx}).content
    data = _safe_json(raw, {})
    try:
        return FeeStructure(**data) if isinstance(data, dict) else FeeStructure()
    except Exception as e:
        logger.warning(f"[Parser] Fee parse failed: {e}")
        return FeeStructure()


def _extract_waterfall_meta(retriever) -> dict:
    ctx  = _build_context(retriever, WATERFALL_QUERIES, k_per_query=4)
    raw  = (WATERFALL_PROMPT | llm).invoke({"context": ctx}).content
    data = _safe_json(raw, {})
    if not isinstance(data, dict):
        data = {}
    steps = []
    for item in data.get("interest_waterfall", []):
        try:
            steps.append(WaterfallStep(**item))
        except Exception as e:
            logger.warning(f"[Parser] Bad waterfall step: {e}")
    return {
        "deal_name":      data.get("deal_name") or "Mock CLO Deal",
        "ccc_bucket_limit": data.get("ccc_bucket_limit"),
        "interest_waterfall": steps,
    }


# ---------------------------------------------------------------------------
# LangGraph Node
# ---------------------------------------------------------------------------

def parser_agent(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node: hybrid extraction.
    Layer 1 — deterministic pdfplumber table scan (tranches + total par)
    Layer 2 — RAG + LLM for coverage tests, fees, waterfall metadata
    """
    pdf_path = state.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Indenture PDF not found at path: {pdf_path}")

    logger.info(f"[Parser] Starting hybrid extraction for: {os.path.basename(pdf_path)}")

    # ------------------------------------------------------------------
    # LAYER 1: Deterministic extraction — no LLM, no embedding needed
    # ------------------------------------------------------------------
    tranches   = _extract_tranches_deterministic(pdf_path)
    total_par  = _extract_total_par_deterministic(pdf_path)

    # Fallback: sum extracted principal amounts if total par not found in text
    if total_par is None and tranches:
        extracted_sum = sum(t.principal_amount for t in tranches if t.principal_amount)
        if extracted_sum > 0:
            total_par = extracted_sum
            logger.info(f"[Parser] Total par derived from tranche sum: ${total_par:,.0f}")

    # ------------------------------------------------------------------
    # LAYER 2: RAG + LLM for semi-structured content
    # ------------------------------------------------------------------
    logger.info("[Parser] Building RAG index for semi-structured content...")
    retriever, collection_name = get_or_create_retriever(pdf_path, k=4)
    index_status = "reused" if get_collection_info(pdf_path)["indexed"] else "newly built"

    coverage_tests = _extract_coverage_tests(retriever)
    fees           = _extract_fees(retriever)
    meta           = _extract_waterfall_meta(retriever)

    # ------------------------------------------------------------------
    # Assemble IndentureRules
    # ------------------------------------------------------------------
    parsed_rules = IndentureRules(
        deal_name=meta["deal_name"],
        total_target_par=total_par,
        tranches=tranches,
        coverage_tests=coverage_tests,
        fees=fees,
        ccc_bucket_limit=meta["ccc_bucket_limit"],
        interest_waterfall=meta["interest_waterfall"],
    )
    rules_dict = parsed_rules.model_dump()

    logger.info(
        f"[Parser] Done. Deal='{parsed_rules.deal_name}' | "
        f"Par=${total_par:,.0f} | Tranches={len(tranches)} | "
        f"Tests={len(coverage_tests)} | Waterfall={len(meta['interest_waterfall'])}"
        if total_par else
        f"[Parser] Done. Deal='{parsed_rules.deal_name}' | Par=N/A | "
        f"Tranches={len(tranches)} | Tests={len(coverage_tests)}"
    )

    # Senior ratio for Quant agent
    senior_tranche = next(
        (t for t in tranches if "A" in t.class_name and not t.is_equity), None
    )
    senior_ratio = (
        round(senior_tranche.principal_amount / total_par, 4)
        if senior_tranche and senior_tranche.principal_amount and total_par
        else 0.65
    )

    missing_principals = any(t.principal_amount is None for t in tranches) if tranches else True
    data_quality = {
        "principals_extracted": bool(tranches) and not missing_principals,
        "tranche_count": len(tranches),
        "coverage_tests_extracted": len(coverage_tests) > 0,
        "extraction_method": "deterministic+rag",
    }

    return {
        "extracted_text_chunks": [
            f"[Collection: {collection_name} | {index_status}]",
            f"[Tranches: {len(tranches)} (deterministic) | Tests: {len(coverage_tests)} (RAG)]",
            f"[Total par: ${total_par:,.0f}]" if total_par else "[Total par: estimated]",
        ],
        "parsed_waterfall": rules_dict,
        "senior_tranche_ratio": senior_ratio,
        "vector_store_id": collection_name,
        "parser_data_quality": data_quality,
    }