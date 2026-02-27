"""
Agentic RAG Pipeline - Document Ingestion
Handles PDF/text upload, chunking, embedding, and storage in ChromaDB.
"""

import chromadb
from chromadb.utils import embedding_functions
import hashlib
import re
from typing import Optional

ef = embedding_functions.DefaultEmbeddingFunction()

client = chromadb.Client()
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)


def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks for better retrieval."""
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        if len(chunk_text_str.strip()) > 50:
            chunks.append({
                "text": chunk_text_str,
                "start_word": start,
                "end_word": min(end, len(words)),
                "chunk_id": hashlib.md5(chunk_text_str[:100].encode()).hexdigest()[:12]
            })
        start += chunk_size - overlap
    return chunks


def ingest_text(text, source="uploaded_document"):
    """Ingest raw text into ChromaDB."""
    chunks = chunk_text(text)
    if not chunks:
        return {"error": "No valid chunks created from text", "chunks": 0}
    ids = [f"{source}_{c['chunk_id']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": source, "chunk_index": i} for i, c in enumerate(chunks)]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return {
        "source": source,
        "total_chunks": len(chunks),
        "total_characters": sum(len(d) for d in documents),
        "status": "ingested"
    }


def ingest_pdf(pdf_bytes, filename):
    """Extract text from PDF and ingest into ChromaDB."""
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        if len(text.strip()) < 50:
            return {"error": "PDF appears to be empty or image-only"}
        return ingest_text(text, source=filename)
    except ImportError:
        return {"error": "PyMuPDF not installed. Run: pip install PyMuPDF"}
    except Exception as e:
        return {"error": f"PDF extraction failed: {str(e)}"}


def search(query, n_results=5):
    """Retrieve relevant document chunks for a query."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    retrieved = []
    if results and results["documents"]:
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            retrieved.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "relevance_score": round(1 - dist, 4)
            })
    return retrieved


def get_doc_count():
    return collection.count()


def clear_documents():
    global collection
    client.delete_collection("documents")
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return {"status": "cleared"}
