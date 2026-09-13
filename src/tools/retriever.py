"""
Retriever Tool: RAG logic for querying CLO Indenture text from local Chroma DB vector storage.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# We use a persistent local directory so we don't have to re-embed the same PDF twice
CHROMA_DB_DIR = "./data/vector_store"

def setup_rag_retriever(pdf_path: str):
    """
    Ingests a large PDF, chunks it, embeds it using a free local HuggingFace model, 
    and stores it in a local ChromaDB vector store.
    """
    print(f"Setting up Vector Store for {pdf_path}...")
    
    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split the text into manageable chunks
    # We use a 1000-character chunk with a 200-character overlap to ensure 
    # legal clauses aren't cut in half.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    # 3. Initialize free, local embeddings (Runs on your CPU, 0 cost)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. Create and persist the Vector Database
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    # Return the retriever object (fetches the top 10 most relevant chunks)
    return vectorstore.as_retriever(search_kwargs={"k": 10})


def search_indenture(retriever, query: str) -> str:
    """
    Executes a semantic search against the Vector DB and returns the combined text.
    """
    relevant_docs = retriever.invoke(query)
    
    # Combine the retrieved chunks into a single string
    context = "\n\n...\n\n".join([doc.page_content for doc in relevant_docs])
    return context
