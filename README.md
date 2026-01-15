# YouTube Comment Summary

Analyze a YouTube video’s comments end-to-end: fetch comments, store them in a vector DB, generate an overall summary, run sentiment analysis, build visualizations, and answer follow-up questions grounded in the comments.

## Features

- **Comment extraction**: pulls comment threads + replies using the YouTube Data API.
- **Vector DB (Chroma)**: persists embeddings per-video for fast semantic retrieval.
- **Overall comment summary**: generates a concise multi-paragraph summary of discussion themes.
- **Sentiment analysis**: multilingual sentiment classification (Positive / Neutral / Negative).
- **Visuals**: sentiment pie chart + word cloud.
- **Q&A**: ask questions; the system retrieves relevant comments and generates an answer.

## Model usage (Embeddings + LLM)

This project is designed to run with **either Gemini or local Ollama**:

- **Preferred (if available)**: Google Gemini via `langchain-google-genai`
  - **Embeddings**: `models/embedding-001`
  - **LLM**: configurable via `GEMINI_MODEL`
- **Fallback**: Ollama via LangChain community integrations
  - **LLM (default)**: `llama3.2` (`OLLAMA_CHAT_MODEL`)
  - **Embeddings (default)**: `nomic-embed-text` (`OLLAMA_EMBED_MODEL`)

If Gemini fails at runtime (e.g., model `NOT_FOUND` on your API/version), the app retries using Ollama.

## Project structure

```
youtube-comment-summary/
├── app.py                  # Flask API + web UI (templates/)
├── Dockerfile              # Production container (Ollama + Gunicorn)
├── requirements.txt
├── assets/                 # README screenshots
├── templates/              # HTML UI
├── static/                 # CSS, client assets
└── src/
    ├── utils.py            # YouTube API client, preprocessing, plots, wordcloud
    ├── chroma_db.py        # Chroma persistence, summary + Q&A over comments
    ├── models.py           # ModelManager (HF sentiment + LLM/embeddings)
    └── youtube_summary_tool.py  # Orchestrator (pipeline steps)
```

## Results (examples)

**Sentiment pie chart**

![Sentiment pie chart](assets/sentiment_pie_chart.png)

**Comment word cloud**

![Comment word cloud](assets/comment_wordcloud.png)

## Local deployment

### Requirements

- **Python**: 3.10+ recommended
- **YouTube Data API key**
- Optional (recommended): **Ollama** installed + running for local LLM fallback

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure environment

Create a `.env` in the project root:

```env
YOUTUBE_API_KEY=YOUR_YOUTUBE_DATA_API_KEY

# Optional: enable Gemini (if your account/project supports the model)
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
GEMINI_MODEL=gemini-1.5-flash

# Optional: Ollama fallback config
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### 3) If using Ollama (recommended)

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve
```

### 4) Run the Flask app

```bash
python app.py
```

Then open:
- **Web UI**: `http://localhost:5005`

## Docker (production)

This repo ships a production Docker image that runs:
- **Ollama** on `11434`
- **Gunicorn (WSGI)** serving Flask on `5005`

### Build

```bash
docker build -t youtube-comment-summary .
```

### Run

```bash
docker run --rm \
  -p 5005:5005 \
  -p 11434:11434 \
  --env-file .env \
  youtube-comment-summary
```

### Runtime configuration (Docker)

- **Gunicorn**
  - `PORT` (default `5005`)
  - `GUNICORN_WORKERS` (default `1`)
  - `GUNICORN_THREADS` (default `4`)
  - `GUNICORN_TIMEOUT` (default `180`)
- **Ollama**
  - `OLLAMA_BASE_URL` is set to `http://127.0.0.1:11434` inside the container by default
  - `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`

## API (Flask)

### Analyze a video

`POST /api/analyze`

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

### Ask a question

`POST /api/ask`

```json
{
  "question": "What are the main criticisms?",
  "k": 50
}
```

### Status

`GET /api/status`

## Outputs

Per video, results are saved under `chroma/<VIDEO_ID>/`:

```
chroma/<VIDEO_ID>/
├── overall_summary.txt
├── sentiment_summary.txt
├── sentiment_pie_chart.png
├── comment_wordcloud.png
└── video_metadata.json
```
