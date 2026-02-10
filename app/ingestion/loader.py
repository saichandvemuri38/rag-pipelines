from pathlib import Path
from app.ingestion.parsers.csv import load_csv
from app.ingestion.parsers.docx import load_docx
from app.ingestion.parsers.html import load_html
from app.ingestion.parsers.pdf import load_pdf
from app.ingestion.parsers.text import load_text

def load_document(file_path: str):
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    if ext == ".docx":
        return load_docx(file_path)
    if ext == ".txt":
        return load_text(file_path)
    if ext in [".html", ".htm"]:
        return load_html(file_path)
    if ext == ".csv":
        return load_csv(file_path)

    raise ValueError(f"Unsupported file type: {ext}")