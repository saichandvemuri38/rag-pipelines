from bs4 import BeautifulSoup
from app.ingestion.formatter import Document

def load_html(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    text = soup.get_text(separator=" ", strip=True)

    return [
        Document(
            text=text,
            metadata={
                "source": file_path,
                "type": "html",
            },
        )
    ]
