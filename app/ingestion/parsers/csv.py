from app.ingestion.formatter import Document
import pandas as pd

def load_csv(file_path: str) -> list[Document]:
    df = pd.read_csv(file_path)

    text = df.to_string(index=False)

    return [
        Document(
            text=text,
            metadata={
                "source": file_path,
                "type": "csv",
            },
        )
    ]