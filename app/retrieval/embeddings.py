# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Make sure OPENAI_API_KEY is in your .env or environment

# def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
#     """
#     Convert text to vector embedding using OpenAI
#     """
#     response = client.embeddings.create(
#         model=model,
#         input=text
#     )
#     return response.data[0].embedding


from sentence_transformers import SentenceTransformer

# Load a local embedding model
# "all-MiniLM-L6-v2" is small, fast, good quality
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> list[float]:
    """
    Convert text to a vector embedding using SentenceTransformer
    """
    return model.encode(text).tolist()  # convert numpy array to list
