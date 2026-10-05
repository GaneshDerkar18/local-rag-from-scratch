import requests
from qdrant_client import QdrantClient

OLLAMA_URL = "http://host.docker.internal:11434/api/embed"
MODEL = "nomic-embed-text"
COLLECTION_NAME = "rag_basics"

def embed(text):
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "input": text}
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]

# 1. Connect to Qdrant
qdrant = QdrantClient(host="qdrant", port=6333)

# 2. Define the user's question
question = "How do I run large language models on my own machine?"
print(f"Question: '{question}'\n")

# 3. Convert the question into a vector
print("Embedding the question...")
query_vector = embed(question)

# 4. Search Qdrant using the new Query API
print("Searching Qdrant...")
response = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=1  # We only want the single best match
)

# 5. Display the result
print("\n--- Top Match ---")
for result in response.points:
    # Score is the cosine similarity (closer to 1.0 is better)
    print(f"Similarity Score: {result.score:.4f}")
    # We retrieve the original text from the payload we saved earlier
    print(f"Found Text: {result.payload['text']}")