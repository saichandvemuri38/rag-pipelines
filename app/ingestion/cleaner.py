import re
from typing import List

from app.ingestion.formatter import Document
from app.ingestion.rules import NOISE_PATTERNS


def clean_text(text: str) -> str:
    # Normalize newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # Fix broken sentences
    text = re.sub(r"(?<![.!?])\n", " ", text)

    # Remove excessive spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Apply noise removal rules
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def clean_documents(documents: List[Document]) -> List[Document]:
    cleaned_docs = []

    for doc in documents:
        cleaned_docs.append(
            Document(
                text=clean_text(doc.text),
                metadata=doc.metadata,
            )
        )

    return cleaned_docs
