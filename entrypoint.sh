#!/bin/sh
echo "Waiting for Ollama..."
until curl -s http://ollama:11434/api/tags >/dev/null 2>&1; do
  sleep 2
done
echo "Ollama is ready."

echo "Pulling models..."
for model in "${OLLAMA_DOMAIN_MODEL:-qwen3.5:4b}" "${OLLAMA_DOMAIN_MODEL_SMALL:-qwen3.5:0.8b}" "${OLLAMA_EMBEDDING_MODEL:-qwen3-embedding:8b}"; do
  echo "  pulling $model"
  if curl -sS -X POST http://ollama:11434/api/pull -d "{\"name\": \"$model\"}" | grep -q '"error"'; then
    echo "  FAILED to pull $model" >&2
    exit 1
  fi
done
echo "Models pulled."

# Start the app
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
