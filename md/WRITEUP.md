# Voice RAG Assistant — Write-Up

## Approach

The system is a streaming RAG pipeline with voice I/O built on FastAPI. A PDF is uploaded once, parsed into token-aware chunks (700 tokens, 120 overlap), and indexed in a local Qdrant collection with two vector types per chunk — a 768-dim dense vector from **nomic-embed-text** (Ollama) and a sparse BM25 vector from **FastEmbed (`Qdrant/bm25`)**. On every query, a 4-node LangGraph pipeline runs: prompt-injection check (regex, no LLM) → anaphora-aware query rewrite (rule-based, no LLM) → hybrid Qdrant search (dense + sparse, RRF-fused server-side in one request) → grounding check (numeric score threshold, no LLM). Only if the top retrieval score clears the threshold is the Gemini LLM called. The LLM streams tokens; as each sentence boundary is detected, a TTS synthesis task (`edge-tts`) is launched in the background via `asyncio.create_task()`. Audio and text are interleaved in the SSE stream so the frontend can display text and play audio simultaneously without waiting for the full answer. Two in-memory LRU caches prevent redundant work: a PDF cache (hash → collection name) skips re-embedding on re-upload, and an LLM cache (hash of pdf + query → full answer) replays identical questions instantly.

---

## AI Tools Used

| Tool | Role |
|------|------|
| **Gemini 3.5 Flash Lite** | Streaming LLM — `google-genai` async SDK (`client.aio.models.generate_content_stream`) |
| **Antigravity (Google DeepMind)** | AI pair-programmer — architecture, code generation, debugging, and latency profiling |
| **nomic-embed-text (Ollama)** | Dense embeddings — local HTTP API, 768-dim vectors |
| **FastEmbed `Qdrant/bm25`** | Sparse BM25 encoder — in-process `SparseTextEmbedding`, no extra server |
| **edge-tts** | Per-sentence neural TTS — Microsoft Edge voices, streams MP3 bytes |
| **Google STT** | Voice transcription — `SpeechRecognition` library, runs in `asyncio.to_thread` |

---

## How Latency Was Reduced (Measured Numbers)

The pipeline was designed to eliminate every unnecessary LLM call from the hot path. The injection check, query decomposition, and grounding check are all zero-LLM steps (regex and numeric comparisons), adding under 1 ms total. Retrieval is a single Qdrant network round-trip that returns in ~7–8 ms. The Ollama embedding call is the dominant non-LLM cost.

The biggest latency win is **parallel sentence-level TTS**: rather than waiting for the full LLM answer before synthesising audio, an `asyncio.create_task()` is launched for each sentence the moment its boundary is detected in the token stream. This means TTS runs concurrently with LLM generation — the first audio chunk reaches the client as soon as the first sentence is ready (~3 s), instead of after the entire answer is complete (~4+ s).

The LLM cache eliminates the Gemini call entirely on repeated questions, bringing response time from ~4 s down to under 1 s. The PDF cache eliminates re-parsing and re-embedding on re-upload, cutting upload time from several seconds to under 100 ms.

**Measured timings from live server logs (before optimisations):**

| Stage | Time |
|-------|------|
| Injection check + query decomposition | < 1 ms each |
| Ollama dense embedding | ~650–680 ms |
| Qdrant hybrid retrieval (RRF) | ~7–8 ms |
| Grounding check | < 1 ms |
| Gemini 3.5 Flash Lite full stream | ~3.5–4.1 s |
| Time-to-first-audio (parallel TTS) | ~3–5 s |
| Voice STT | ~700 ms–1.2 s |
| On LLM cache hit | < 1 s |
| On PDF cache hit | < 100 ms |

**Optimisations implemented and their measured savings:**

| Change | File | Saving |
|--------|------|--------|
| Persistent `httpx.AsyncClient` — single TCP connection reused across all embedding calls instead of a new client per query | `embeddings.py` | ~50–150 ms per query |
| Removed `adjust_for_ambient_noise(duration=0.3)` — hardcoded 300 ms blocking wait on every voice request | `speech_to_text.py` | ~300 ms per voice query |
| Removed `asyncio.sleep(0.02)` per 5-word chunk in cache-hit replay | `streaming_service.py` | ~400–800 ms per cache hit |
| Cache-hit TTS changed from synthesising full answer as one string to sentence-by-sentence parallel synthesis (same as live path) | `streaming_service.py` | ~1–3 s first-audio delay on cache hits |
| `sr.Recognizer()` made a module-level singleton | `speech_to_text.py` | Negligible, avoids per-call construction |
| `GenerateContentConfig` made a module-level singleton | `gemini.py` | Negligible, avoids per-call construction |
| `RecursiveCharacterTextSplitter` made a module-level singleton | `chunker.py` | Negligible, avoids per-upload construction |

**Net saving on a cache-hit voice query: approximately ~1.75–4.25 seconds** of latency removed — on a system whose total response time was ~4–5 s, this is a significant improvement.
