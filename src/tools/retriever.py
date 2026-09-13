"""
Retriever Tool: RAG pipeline for querying 300-page CLO Indenture PDFs.

Design:
- Each PDF gets its own isolated ChromaDB collection (keyed by a stable hash of the filename).
- Switching PDFs = different collection = zero data bleed between documents.
- Same PDF uploaded again = collection already exists → skip re-embedding instantly.
- Local HuggingFace embeddings (all-MiniLM-L6-v2) run on CPU at zero cost.
"""

import os
import hashlib
import logging
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

# Root directory where ALL ChromaDB collections are persisted.
# Each PDF collection lives in its own subdirectory underneath this.
CHROMA_DB_ROOT = "./data/vector_store"

# Shared embedding model (loaded once, reused across all calls in the process)
_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily initialise and cache the embedding model (CPU, free, local)."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        logger.info("Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded.")
    return _EMBEDDINGS


def _collection_name_for_pdf(pdf_path: str) -> str:
    """
    Derives a stable, unique ChromaDB collection name from the PDF filename.

    Uses a short SHA-256 hash of the *basename* (not the full path) so that
    moving the file doesn't invalidate the existing index.

    ChromaDB collection names must:
    - Be 3–63 characters
    - Contain only alphanumerics, underscores, and hyphens
    - Not start/end with a hyphen or underscore
    """
    basename = os.path.basename(pdf_path)
    # Remove extension, sanitise special chars → safe prefix
    safe_stem = "".join(c if c.isalnum() else "_" for c in os.path.splitext(basename)[0])[:24]
    short_hash = hashlib.sha256(basename.encode()).hexdigest()[:12]
    return f"{safe_stem}_{short_hash}"


def _persist_dir_for_collection(collection_name: str) -> str:
    """Returns the on-disk path for a given collection's ChromaDB data."""
    return os.path.join(CHROMA_DB_ROOT, collection_name)


def _collection_exists(collection_name: str) -> bool:
    """
    Returns True if this collection has already been embedded and persisted.
    We check for the presence of the ChromaDB SQLite file as the indicator.
    """
    persist_dir = _persist_dir_for_collection(collection_name)
    chroma_db_file = os.path.join(persist_dir, "chroma.sqlite3")
    return os.path.isfile(chroma_db_file)


def get_or_create_retriever(pdf_path: str, k: int = 8):
    """
    Main entry point: Returns a LangChain retriever backed by ChromaDB.

    - First call for a PDF: chunks + embeds the full document, persists to disk.
    - Subsequent calls: loads the existing collection from disk (no re-embedding).

    Args:
        pdf_path: Absolute or relative path to the CLO Indenture PDF.
        k: Number of top-relevant chunks to retrieve per query (default: 8).

    Returns:
        A LangChain VectorStoreRetriever ready for semantic search.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    collection_name = _collection_name_for_pdf(pdf_path)
    persist_dir = _persist_dir_for_collection(collection_name)
    embeddings = _get_embeddings()

    if _collection_exists(collection_name):
        logger.info(f"[RAG] Reusing existing index '{collection_name}' for {os.path.basename(pdf_path)}")
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    else:
        logger.info(f"[RAG] Building new index '{collection_name}' for {os.path.basename(pdf_path)}...")
        os.makedirs(persist_dir, exist_ok=True)

        # 1. Load all pages of the PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        logger.info(f"[RAG] Loaded {len(docs)} pages from PDF.")

        # 2. Chunk into 1000-char segments with 200-char overlap.
        #    Legal docs have long clauses — overlap preserves cross-sentence context.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        splits = text_splitter.split_documents(docs)
        logger.info(f"[RAG] Split into {len(splits)} chunks.")

        # 3. Embed and persist — each collection lives in its own subdirectory.
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
        logger.info(f"[RAG] Index '{collection_name}' built and persisted to {persist_dir}.")

    return vectorstore.as_retriever(search_kwargs={"k": k}), collection_name


def search_indenture(retriever, query: str) -> str:
    """
    Executes a single semantic search and returns the combined text of top-k chunks.

    Args:
        retriever: A LangChain retriever returned by get_or_create_retriever().
        query:     A natural-language description of the information to find.

    Returns:
        A single string of retrieved legal text chunks, separated by dividers.
    """
    relevant_docs = retriever.invoke(query)
    if not relevant_docs:
        return "(No relevant content found for this query)"
    context = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in relevant_docs
    )
    return context


def delete_collection(pdf_path: str) -> bool:
    """
    Deletes the ChromaDB collection for a given PDF, forcing re-indexing next time.

    Args:
        pdf_path: Path to the PDF whose index should be cleared.

    Returns:
        True if the collection existed and was deleted, False if it didn't exist.
    """
    import shutil
    collection_name = _collection_name_for_pdf(pdf_path)
    persist_dir = _persist_dir_for_collection(collection_name)

    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)
        logger.info(f"[RAG] Deleted collection '{collection_name}' at {persist_dir}.")
        return True

    logger.info(f"[RAG] No collection found for '{collection_name}' — nothing to delete.")
    return False


def get_collection_info(pdf_path: str) -> dict:
    """
    Returns metadata about a PDF's index (collection name, path, whether it exists).
    Useful for displaying status in the Streamlit UI.
    """
    collection_name = _collection_name_for_pdf(pdf_path)
    persist_dir = _persist_dir_for_collection(collection_name)
    exists = _collection_exists(collection_name)

    info = {
        "collection_name": collection_name,
        "persist_dir": persist_dir,
        "indexed": exists,
    }

    if exists:
        # Estimate chunk count from the SQLite file size as a rough proxy
        db_file = os.path.join(persist_dir, "chroma.sqlite3")
        info["index_size_mb"] = round(os.path.getsize(db_file) / (1024 * 1024), 2)

    return info
