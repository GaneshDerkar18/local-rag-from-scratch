import uuid
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader

OLLAMA_URL = "http://host.docker.internal:11434/api/embed"
MODEL = "nomic-embed-text"
COLLECTION_NAME = "rag_pdf"

def embed(text):
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "input": text})
    response.raise_for_status()
    return response.json()["embeddings"][0]

# The Sliding Window Chunker
def get_chunks(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap) # Move forward, but leave an overlap
    return chunks

qdrant = QdrantClient(host="qdrant", port=6333)

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

print("Extracting text from PDF...")
reader = PdfReader("/documents/bitcoin.pdf")
raw_text = ""
for page in reader.pages:
    raw_text += page.extract_text() + " "

# Clean up excess whitespace
raw_text = " ".join(raw_text.split())

print("Chunking text...")
chunks = get_chunks(raw_text, chunk_size=1000, overlap=200)
print(f"Created {len(chunks)} overlapping chunks.")

points = []
for idx, chunk in enumerate(chunks):
    print(f"Embedding chunk {idx + 1}/{len(chunks)}...")
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embed(chunk),
            payload={"text": chunk, "source": "bitcoin.pdf"} 
        )
    )

print("Uploading to Qdrant...")
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print("PDF ingestion complete!")