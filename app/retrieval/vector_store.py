
import chromadb

from app.ingestion.formatter import Document
from app.retrieval.embeddings import get_embedding

# Initialize Chroma client (local)
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="rag_documents")


def add_documents(documents: list[Document]):
    """
    Add cleaned and chunked docs to Chroma with embeddings
    """
    for doc in documents:
        vector = get_embedding(doc.text)

        collection.add(
            ids=[f"{doc.metadata.get('source')}_{doc.metadata.get('chunk_index', 0)}"],
            documents=[doc.text],
            metadatas=[doc.metadata],
            embeddings=[vector]
        )
