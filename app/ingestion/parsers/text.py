
from app.ingestion.formatter import Document


def load_text(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [
        Document(
            text=text.strip(),
            metadata={
                "source": file_path,
                "type": "text",
            },
        )
    ]