# Build Python environment with guaranteed 3.11+
FROM python:3.11-slim AS py

WORKDIR /app

# Create a virtual environment and install dependencies (avoids PEP 668 issues)
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
  && python -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Final image: Ollama preinstalled (Ubuntu)
FROM ollama/ollama:latest

# Copy Python 3.11 runtime from builder image
COPY --from=py /usr/local /usr/local
COPY --from=py /opt/venv /opt/venv
COPY --from=py /app /app

WORKDIR /app
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

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
