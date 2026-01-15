# Base image: Ollama preinstalled (Ubuntu)
FROM ollama/ollama:latest

# Install Python toolchain (Python + pip + venv) and minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python-is-python3 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application
COPY . /app

# Optional: preload an Ollama model (override at build time if needed)
ARG OLLAMA_PRELOAD_MODEL=llama3.2
RUN ollama pull ${OLLAMA_PRELOAD_MODEL} || true

# Expose ports: Ollama + Flask API
EXPOSE 11434 5005

# Production defaults (override via docker run -e ...)
ENV PORT=5005 \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=180

# Run: Ollama in background + Gunicorn (WSGI) in foreground
# Flask app instance is `app` inside `app.py` → `app:app`
CMD /bin/bash -lc "ollama serve & exec gunicorn -b 0.0.0.0:${PORT} --workers ${GUNICORN_WORKERS} --threads ${GUNICORN_THREADS} --timeout ${GUNICORN_TIMEOUT} app:app"
