from fastapi import FastAPI
from app.ingestion.loader import load_document
from app.api.routes import router
from app.ingestion.cleaner import clean_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.vector_store import add_documents
from app.retrieval.embeddings import get_embedding
from fastapi import Query, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))


app = FastAPI(
    title="FastAPI Project",
    description="Sample FastAPI application",
    version="1.0.0"
)

app.include_router(router)


@app.on_event("startup")
def startup_event():
    app.state.chroma_client = chromadb.PersistentClient(path="./chroma_db")
    app.state.collection = app.state.chroma_client.get_or_create_collection(
        name="rag_collection"
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/ingest")
def ingest_local_document(request: Request):
    # collection.delete(collection.get()["ids"])
    # 1️⃣ Load
    docs = load_document("./data/test_rag.pdf")

    # 2️⃣ Clean
    cleaned = clean_documents(docs)

    # 3️⃣ Chunk
    chunked = chunk_documents(cleaned, max_tokens=500, overlap=50)

    # 4️⃣ Embeddings + Chroma
    add_documents(chunked, request)

    return {"status": "success", "chunks_added": len(chunked)}

@app.post("/query")
def query_rag(q: str = Query(..., description="Question to search")):
    q_vector = get_embedding(q)
    results = app.state.collection.query(
        query_embeddings=[q_vector],
        n_results=3
    )
    docs = results.get('documents', [[]])[0]  # default to empty list
    context = "\n".join(docs) if docs else "No relevant context found."
    if not context.strip():
        context = "No relevant context found."

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer based on context only"},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {q}"}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    answer = response.choices[0].message.content.strip()
    return {"context": context, "answer": answer}

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

