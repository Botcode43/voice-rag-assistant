# Voice RAG Assistant

A voice-enabled Retrieval-Augmented Generation (RAG) assistant that:

1. Accepts a **PDF upload**
2. Answers questions (text or voice) **only** from that PDF's content
3. **Speaks the answer** using Edge-TTS
4. **Streams both text and audio** for low latency

---

## Architecture

```
FastAPI Backend
├── LangGraph Pipeline
│   ├── Prompt Injection Check (regex/keyword — no LLM)
│   ├── Query Decomposition (session memory, rule-based)
│   ├── Hybrid Retrieval (Qdrant dense+sparse, RRF-fused server-side)
│   └── Grounding Check (similarity threshold — no LLM)
├── Gemini 2.5 Flash (streaming generation)
├── Edge-TTS (per-sentence, parallel with LLM stream)
└── SSE streaming to client

Streamlit Frontend
├── PDF upload + cache indicator
├── Text chat input with streaming response
└── Voice recording + transcription (SpeechRecognition)
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested on 3.11 and 3.12 |
| [Ollama](https://ollama.ai) | Local embedding server |
| `nomic-embed-text` model | `ollama pull nomic-embed-text` |
| Google Gemini API key | [Get one here](https://aistudio.google.com/app/apikey) |
| Internet connection | For Gemini API + Edge-TTS + Google STT |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd voice-rag-assistant
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **PyAudio on Windows**: If `pyaudio` fails to install, download the wheel from
> [Christoph Gohlke's page](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
> or use `pip install pipwin && pipwin install pyaudio`.

### 3. Configure environment variables

```bash
cp .env.example .env   # or just edit .env directly
```

Edit `.env`:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

All other values have sensible defaults (see `.env` for the full list).

### 4. Start Ollama and pull the embedding model

```bash
# In a separate terminal:
ollama serve

# Pull the embedding model (one-time):
ollama pull nomic-embed-text
```

---

## Running

### Backend (FastAPI)

```bash
# From the project root:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### Frontend (Streamlit)

```bash
# In a separate terminal:
streamlit run frontend/app.py
```

Opens at: http://localhost:8501

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Upload a PDF; returns `pdf_hash` |
| `POST` | `/query` | Text query → SSE stream (text + audio) |
| `POST` | `/voice-query` | Audio upload → STT → SSE stream |
| `GET` | `/stream` | GET variant of `/query` (for EventSource clients) |
| `GET` | `/health` | Health check |

### POST /upload

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "pdf_hash": "abc123...",
  "filename": "your_document.pdf",
  "cached": false,
  "num_chunks": 42,
  "message": "PDF processed successfully (42 chunks embedded)."
}
```

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-001",
    "pdf_hash": "abc123...",
    "query": "What is the main topic of this document?"
  }'
```

Returns an SSE stream:
```
data: {"type": "text", "content": "The main topic"}
data: {"type": "text", "content": " is artificial intelligence."}
data: {"type": "audio", "audio_b64": "<base64-encoded-mp3>"}
data: {"type": "done"}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | **required** | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model identifier |
| `GEMINI_TEMPERATURE` | `1.0` | Generation temperature |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model name |
| `EMBEDDING_DIM` | `768` | Embedding vector dimension |
| `QDRANT_PATH` | `./qdrant_storage` | Qdrant storage path |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per query |
| `GROUNDING_THRESHOLD` | `0.30` | Min similarity score for grounding |
| `TTS_VOICE` | `en-US-AriaNeural` | Edge-TTS voice |
| `MEMORY_MAX_TURNS` | `10` | Session memory depth |
| `LLM_CACHE_MAX_SIZE` | `256` | LRU cache size for LLM responses |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI port |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Caching Behaviour

- **PDF Cache**: Keyed by `sha256(pdf_bytes)`. Re-uploading the same PDF skips parsing/chunking/embedding entirely.
- **LLM Response Cache**: Keyed by `sha256(pdf_hash + normalized_query)`. Identical questions (same PDF) return instantly. Fallback responses ("I couldn't find…") are **not** cached.

---

## Prompt Injection Guard

The following patterns are blocked before reaching the LLM (case-insensitive regex):

- `ignore previous/prior instructions`
- `reveal system prompt` / `show your instructions`
- `forget the document` / `ignore the pdf`
- `act as ChatGPT` / `pretend you are…`
- `tell me your hidden instructions`
- `jailbreak` / `bypass safety`

---

## Project Structure

```
voice-rag-assistant/
├── app/
│   ├── main.py             # FastAPI app factory + lifespan
│   ├── config.py           # Pydantic settings (from .env)
│   ├── dependencies.py     # FastAPI DI helpers
│   ├── api/                # Route handlers
│   ├── graph/              # LangGraph nodes, edges, state
│   ├── rag/                # Parser, chunker, embeddings, vector store, retriever
│   ├── llm/                # Gemini client + prompts
│   ├── speech/             # STT + TTS
│   ├── cache/              # LRU in-memory caches
│   ├── memory/             # Session conversation memory
│   ├── guardrails/         # Prompt injection guard
│   ├── models/             # Pydantic request/response schemas
│   ├── utils/              # Logger, helpers
│   └── services/           # Orchestration layer
├── frontend/
│   └── app.py              # Streamlit UI
├── tests/
├── requirements.txt
├── .env                    # Your local config (git-ignored)
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Notes

- Qdrant runs in **local persistent mode** (no Docker needed). Data is stored in `./qdrant_storage/`.
- For pure in-memory (no persistence), set `QDRANT_PATH=:memory:` in `.env`.
- Sparse vectors use **Qdrant's BM25 FastEmbed** model; no separate index needed.
- TTS is streamed per completed sentence — audio starts before the full answer is generated.
