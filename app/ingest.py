import os
import uuid
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

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

# 1. Connect to Qdrant (using the Docker service name 'qdrant')
print("Connecting to Qdrant...")
qdrant = QdrantClient(host="qdrant", port=6333)

# 2. Create the collection if it doesn't exist
if not qdrant.collection_exists(COLLECTION_NAME):
    print(f"Creating collection: {COLLECTION_NAME}")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

# 3. Read and Chunk the document
# We split by double newlines to isolate paragraphs
print("Reading and chunking knowledge.txt...")
with open("knowledge.txt", "r") as f:
    raw_text = f.read()

chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]

# 4. Embed each chunk and prepare for upload
points = []
for idx, chunk in enumerate(chunks):
    print(f"Embedding chunk {idx + 1}/{len(chunks)}...")
    vector = embed(chunk)
    
    # We store the original text in the 'payload'. 
    # This is vital because the database only returns vectors natively.
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk} 
        )
    )

# 5. Insert into Qdrant
print("Uploading to Qdrant...")
qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)
print("Ingestion complete! Data is now searchable.")