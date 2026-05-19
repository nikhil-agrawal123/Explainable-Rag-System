FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN chmod +x entrypoint.sh
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
