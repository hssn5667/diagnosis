"""
Knowledge Base Ingest Pipeline
Run this ONCE before starting the Diagnosis Agent.
Seeds ChromaDB with medical knowledge documents.

Usage:
    python ingest.py

Requirements:
    - ChromaDB server running (docker-compose up chromadb)
    - OPENAI_API_KEY in environment
    - Documents in ./sources/ directory
"""
import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import logging
import sys

# Configure logging to output to stdout (Docker needs this!)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ingest")

from dotenv import load_dotenv
load_dotenv()


# Configuration
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8008"))
COLLECTION_NAME = "medical_knowledge"
SOURCES_DIR = Path(__file__).parent / "sources"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def load_documents(sources_dir: Path) -> list[Document]:
    """Load all .txt files from the sources directory"""
    documents = []
    
    txt_files = list(sources_dir.glob("**/*.txt"))
    if not txt_files:
        logger.warning(f" ✘  No .txt files found in {sources_dir}")
        return documents
    
    for file_path in txt_files:
        logger.info(f"  Loading: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        doc = Document(
            page_content=content,
            metadata={
                "source": file_path.name,
                "type": "medical_guideline"
            }
        )
        documents.append(doc)
    
    logger.info(f"  ✔  Loaded {len(documents)} documents")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for RAG"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"  ✔  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def get_chroma_client():
    """Get ChromaDB HTTP client (connects to running ChromaDB container)"""
    try:
        client = chromadb.HttpClient(
            host=CHROMADB_HOST,
            port=CHROMADB_PORT
        )
        # Test connection
        client.heartbeat()
        logger.info(f"  ✔  Connected to ChromaDB at {CHROMADB_HOST}:{CHROMADB_PORT}")
        return client
    except Exception as e:
        logger.error(f"  ✘   ChromaDB connection failed: {e}")
        logger.error(f"   Make sure ChromaDB is running: docker-compose up chromadb")
        sys.exit(1)


def ingest_documents(chunks: list[Document], client: chromadb.ClientAPI):
    """Embed chunks and store in ChromaDB"""
    
    embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    logger.info(f"  Embedding {len(chunks)} chunks with Google Gemini Embeddings...")
    
    # Delete existing collection to start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"  Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass  # Collection doesn't exist yet
    
    # Create vectorstore and add documents
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding
    )
    
    # Add in batches of 50 to avoid rate limits
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        logger.info(f"  Ingested batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
    
    logger.info(f"  ✔  Successfully ingested {len(chunks)} chunks into collection '{COLLECTION_NAME}'")
    return vectorstore


def verify_ingestion(vectorstore):
    """Quick verification that the knowledge base works"""
    logger.info("\nVerifying knowledge base...")
    
    test_queries = [
        "What are the symptoms of pneumonia?",
        "How to treat community-acquired pneumonia with penicillin allergy?",
        "What does elevated WBC indicate?",
    ]
    
    for query in test_queries:
        results = vectorstore.similarity_search(query, k=2)
        if results:
            logger.info(f"  ✔ Query '{query[:50]}...' → found {len(results)} relevant chunks")
        else:
            logger.warning(f"  ✘ Query '{query}' returned no results")


def main():
    print("=" * 60)
    print("MediTwin Knowledge Base Ingest Pipeline")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("✘ GOOGLE_API_KEY not set in environment!")
        print("  Set it in .env or pass via docker-compose")
        sys.exit(1)
    
    print(f"✔ GOOGLE_API_KEY loaded (length: {len(api_key)})")
    
    # Step 1: Load documents
    print("\n1. Loading documents...")
    documents = load_documents(SOURCES_DIR)
    if not documents:
        sys.exit(1)
    
    # Step 2: Split into chunks
    print("\n2. Splitting documents...")
    chunks = split_documents(documents)
    
    # Step 3: Connect to ChromaDB
    print("\n3. Connecting to ChromaDB...")
    client = get_chroma_client()
    
    # Step 4: Ingest
    print("\n4. Embedding and ingesting...")
    vectorstore = ingest_documents(chunks, client)
    
    # Step 5: Verify
    print("\n5. Verifying ingestion...")
    verify_ingestion(vectorstore)
    
    print("\n" + "=" * 60)
    print(" ✔  Knowledge base ready! Diagnosis Agent can now start.")
    print ("=" * 60)


if __name__ == "__main__":
    main()