# Production image with guaranteed Python >= 3.11
# (Do NOT copy Python from another base image into Ollama's image; it can break OpenSSL)
FROM python:3.11-slim

WORKDIR /app

# System deps + Ollama install
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
  && rm -rf /var/lib/apt/lists/* \
  && curl -fsSL https://ollama.com/install.sh | sh

# Create a virtual environment (avoids PEP 668 issues on Debian-based hosts)
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
  && python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY . /app

# Ports: Flask API + Ollama
EXPOSE 5005 11434

# Defaults (override via Railway/Docker env vars)
ENV PORT=5005 \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    OLLAMA_PRELOAD_MODEL=llama3.2 \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=180

# Start Ollama, optionally pull a model, then start Gunicorn
CMD ["sh", "-lc", "ollama serve & sleep 1 && (ollama pull ${OLLAMA_PRELOAD_MODEL} || true) && exec gunicorn -b 0.0.0.0:${PORT} --workers ${GUNICORN_WORKERS} --threads ${GUNICORN_THREADS} --timeout ${GUNICORN_TIMEOUT} app:app"]
