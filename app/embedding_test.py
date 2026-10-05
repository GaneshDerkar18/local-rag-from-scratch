import requests
import math


OLLAMA_URL = "http://host.docker.internal:11434/api/embed"
MODEL = "nomic-embed-text"


def embed(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "input": text,
        },
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    return dot_product / (magnitude_a * magnitude_b)


text_a = "Docker containers isolate applications."
text_b = "Containers provide isolated environments for software."
text_c = "The weather is very hot today."


vector_a = embed(text_a)
vector_b = embed(text_b)
vector_c = embed(text_c)


print("Vector dimensions:", len(vector_a))

print("A vs B:", cosine_similarity(vector_a, vector_b))
print("A vs C:", cosine_similarity(vector_a, vector_c))