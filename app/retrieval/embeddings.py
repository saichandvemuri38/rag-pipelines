import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))  # Make sure OPENAI_API_KEY is in your .env or environment

# def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
#     """
#     Convert text to vector embedding using OpenAI
#     """
#     response = client.embeddings.create(
#         model=model,
#         input=text
#     )
#     return response.data[0].embedding

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # or "text-embedding-3-large"
    )
    return response.data[0].embedding
