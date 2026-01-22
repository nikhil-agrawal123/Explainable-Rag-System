import ollama

# Test direct ollama connection
try:
    response = ollama.embed(model="qwen3-embedding:8b", input="Hello world")
    print(f"Success! Embedding dimension: {len(response['embeddings'][0])}")
except Exception as e:
    print(f"Error: {e}")
