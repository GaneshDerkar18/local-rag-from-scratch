import json
import requests
from qdrant_client import QdrantClient

OLLAMA_EMBED_URL = "http://host.docker.internal:11434/api/embed"
OLLAMA_CHAT_URL = "http://host.docker.internal:11434/api/generate"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3"
COLLECTION_NAME = "rag_pdf"

def embed(text):
    response = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": text})
    response.raise_for_status()
    return response.json()["embeddings"][0]

def generate_stream(prompt):
    # We tell requests to stream the response, and Ollama to send it chunk-by-chunk
    response = requests.post(
        OLLAMA_CHAT_URL, 
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": True},
        stream=True
    )
    response.raise_for_status()
    
    # Print the text as it arrives
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            if "response" in chunk:
                print(chunk["response"], end="", flush=True)
    print() # Add a final newline when done

print("Connecting to database...")
qdrant = QdrantClient(host="qdrant", port=6333)

print("\n" + "="*50)
print("Bitcoin Whitepaper RAG initialized.")
print("Type 'exit' or 'quit' to stop.")
print("="*50 + "\n")

# The Continuous Chat Loop
while True:
    try:
        question = input("\nYou: ")
        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        if not question.strip():
            continue
        
        # 1. Embed & Search
        query_vector = embed(question)
        search_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3  # Grab top 3 chunks
        )
        
        # 2. Combine Context
        context = "\n\n".join([result.payload['text'] for result in search_results.points])
        
        # 3. Build Prompt
        final_prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the context doesn't contain the answer, say "I don't know based on my current knowledge."

        Context:
        {context}

        Question:
        {question}
        
        Answer:"""

        # 4. Generate & Stream
        print("Assistant: ", end="", flush=True)
        generate_stream(final_prompt)
        
    except KeyboardInterrupt: # Catches Ctrl+C
        print("\nGoodbye!")
        break