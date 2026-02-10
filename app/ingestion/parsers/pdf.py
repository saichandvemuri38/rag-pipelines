from app.ingestion.formatter import Document
from pypdf import PdfReader

def load_pdf(file_path: str) -> list[Document]:
    reader = PdfReader(file_path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            documents.append(
                Document(
                    text=text.strip(),
                    metadata={
                        "source": file_path,
                        "page": page_num + 1,
                        "type": "pdf",
                    },
                )
            )

    return documents