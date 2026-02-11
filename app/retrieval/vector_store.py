
import chromadb

from app.ingestion.formatter import Document
from app.retrieval.embeddings import get_embedding
from fastapi import Request

# Initialize Chroma client (local)

def add_documents(documents: list[Document], request: Request):
    collection = request.app.state.collection

    for doc in documents:
        vector = get_embedding(doc.text)

        collection.add(
            ids=[f"{doc.metadata.get('source')}_{doc.metadata.get('chunk_index', 0)}"],
            documents=[doc.text],
            metadatas=[doc.metadata],
            embeddings=[vector]
        )
