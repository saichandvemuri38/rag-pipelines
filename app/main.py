from fastapi import FastAPI
from app.ingestion.loader import load_document
from app.api.routes import router
from app.ingestion.cleaner import clean_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.vector_store import add_documents
from app.retrieval.embeddings import get_embedding
from fastapi import Query
from app.retrieval.vector_store import collection


app = FastAPI(
    title="FastAPI Project",
    description="Sample FastAPI application",
    version="1.0.0"
)

app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/ingest")
def ingest_local_document():
    collection.delete(collection.get()["ids"])
    # 1️⃣ Load
    docs = load_document("./data/test_rag.pdf")

    # 2️⃣ Clean
    cleaned = clean_documents(docs)

    # 3️⃣ Chunk
    chunked = chunk_documents(cleaned, max_tokens=500, overlap=50)

    # 4️⃣ Embeddings + Chroma
    add_documents(chunked)

    return {"status": "success", "chunks_added": len(chunked)}


@app.post("/query")
def query_rag(q: str = Query(..., description="Question to search")):
    q_vector = get_embedding(q)
    results = collection.query(
        query_embeddings=[q_vector],
        n_results=3
    )
    
    return results

# @app.post("/query")
# def query_rag(q: str = Query(..., description="Question to search")):
#     q_vector = get_embedding(q)
#     results = collection.query(
#         query_embeddings=[q_vector],
#         n_results=3
#     )
#     return results
# if __name__ == "__main__":
#     load_local_document()

