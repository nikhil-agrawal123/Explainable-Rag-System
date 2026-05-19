#!/bin/sh
echo "Waiting for Ollama..."
until curl -s http://ollama:11434/api/tags >/dev/null 2>&1; do
  sleep 2
done
echo "Ollama is ready."

echo "Pulling models..."
curl -s -X POST http://ollama:11434/api/pull -d '{"name": "qwen3.5:0.8b"}' >/dev/null
curl -s -X POST http://ollama:11434/api/pull -d '{"name": "qwen3-embedding:8b"}' >/dev/null
echo "Models pulled."

# Start the app
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
