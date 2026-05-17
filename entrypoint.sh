#!/bin/sh
# Wait for Ollama to be ready
until ollama list >/dev/null 2>&1; do
  echo "Waiting for Ollama..."
  sleep 2
done

# ── Pull models ──
# Add the models you need below, one per line:
ollama pull qwen3.5:0.8b
ollama pull qwen3-embedding:8b

# Start the app
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
