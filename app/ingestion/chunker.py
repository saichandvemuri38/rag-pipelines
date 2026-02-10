from typing import List
import re
from app.ingestion.formatter import Document

try:
    import tiktoken  # for token-based splitting
except ImportError:
    tiktoken = None


def count_tokens(text: str, model="text-embedding-3-small") -> int:
    if tiktoken:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    # fallback: approximate
    return len(text.split())


def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap: int = 50,
    model: str = "text-embedding-3-small",
) -> List[str]:
    """Split text into chunks with token overlap"""
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks = []
    chunk = ""
    chunk_tokens = 0

    for sentence in sentences:
        sent_tokens = count_tokens(sentence, model=model)
        if chunk_tokens + sent_tokens > max_tokens:
            if chunk:
                chunks.append(chunk.strip())
            # start new chunk with overlap
            chunk_tokens = 0
            chunk = " ".join(chunks[-overlap:] if overlap else [])
        chunk += " " + sentence
        chunk_tokens += sent_tokens

    if chunk:
        chunks.append(chunk.strip())

    return chunks


def chunk_documents(
    documents: List[Document], max_tokens=500, overlap=50
) -> List[Document]:
    chunked_docs = []

    for doc in documents:
        chunks = chunk_text(doc.text, max_tokens=max_tokens, overlap=overlap)
        for i, chunk in enumerate(chunks):
            # keep metadata + add chunk index
            new_meta = doc.metadata.copy()
            new_meta.update({"chunk_index": i})
            chunked_docs.append(
                Document(text=chunk, metadata=new_meta)
            )

    return chunked_docs
