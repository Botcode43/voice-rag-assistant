# Voice RAG Assistant — Session Memory / Progress Tracking

**Last Updated:** 2026-07-26  
**Conversation ID:** e56d9eaf-3257-4575-802b-bacc351c9184  
**Status:** ✅ COMPLETE — Two bugs fixed (chunk duplication + voice audio)

---

## Build Steps Completed

| Step | Status | Notes |
|------|--------|-------|
| 1. Project scaffold + config + `.gitignore` + `requirements.txt` | ✅ Done | Pydantic-settings, all dirs created |
| 2. PDF parse + chunk + embed + Qdrant + PDF cache | ✅ Done | PyMuPDF, tiktoken chunker, nomic-embed via Ollama HTTP, Qdrant local persistent |
| 3. LangGraph state/nodes/edges/graph wiring | ✅ Done | 4 nodes compiled, conditional routing |
| 4. Prompt injection guard + query decomp + retrieval + grounding check | ✅ Done | Regex guard, anaphora resolution, hybrid RRF retrieval, threshold check |
| 5. Gemini streaming generation + LLM response cache | ✅ Done | google-genai v1.x async, LRU cache with streaming replay |
| 6. Edge TTS streaming (per-sentence) | ✅ Done | Concurrent sentence synthesis, yields (text, audio) pairs |
| 7. FastAPI endpoints wired to graph, StreamingResponse | ✅ Done | /upload, /query, /voice-query, /stream all working |
| 8. SpeechRecognition for /voice-query | ✅ Done | Thread-pooled STT, Google API, WAV/WebM support |
| 9. Streamlit frontend | ✅ Done | Upload, text chat, voice recorder, streamed text+audio playback |
| 10. README + basic tests | ✅ Done | 33 tests passing, full README |
| 11. **Bug Fix: chunk duplication on re-upload** | ✅ Done | Two-layer cache (memory + Qdrant); see BUG FIX 1 below |
| 12. **Bug Fix: voice audio silent after st.rerun()** | ✅ Done | pending_audio pattern; see BUG FIX 2 below |

---

## Architecture Summary

```
FastAPI (port 8000)
├── POST /upload        → pdf_service → parse → chunk → embed → Qdrant
├── POST /query         → rag_service → LangGraph → Gemini stream → SSE
├── POST /voice-query   → STT → rag_service → SSE
└── GET  /stream        → same as /query via GET params

LangGraph Pipeline:
  START → injection_check → query_decomposition → retrieval
       → grounding_check → [END or streaming_service]

Streaming Service:
  Gemini 2.5 Flash stream → per-sentence TTS → SSE events
  (type: text | audio | done | error)

Streamlit Frontend (port 8501):
  - PDF upload panel (sidebar)
  - Chat interface with SSE stream parsing
  - Audio autoplay via pending_audio session state pattern
  - Voice capture via audio-recorder-streamlit
```

---

## Bug Fixes (Session 3 — 2026-07-26)

### BUG FIX 1: Chunk Count Grows on Re-Upload

**Root Cause:**  
The `pdf_cache` is an **in-memory LRU** cache that resets on every server restart. However, Qdrant collections **persist on disk** (at `qdrant_storage/`). When a user re-uploads the same PDF after a restart:
1. In-memory cache misses
2. `create_collection_if_missing()` detects the collection exists → skips creation ✅
3. BUT `upsert_chunks()` still runs → **appends** new duplicate points to the same collection ❌

**Result:** 26 chunks on first upload → 70+ on second → 100+ on third.

**Fix Applied:**
- Added `collection_exists()` function to `app/rag/vector_store.py`
- Added "Layer 2" check in `app/services/pdf_service.py`: after an in-memory cache miss, check Qdrant directly. If the collection exists with `count > 0`, re-populate the in-memory cache and return early — skip all parse/chunk/embed/upsert steps.

**Files Changed:**
- `app/rag/vector_store.py` — added `collection_exists()` helper
- `app/services/pdf_service.py` — added two-layer cache (memory + Qdrant persistent)

---

### BUG FIX 2: Voice Mode Audio Silent (Text Mode Works Fine)

**Root Cause:**  
The voice-query handler calls `st.rerun()` after processing to refresh the chat history. This wipes **all dynamically rendered HTML** from the current render pass — including the `<audio autoplay>` element injected by `autoplay_audio_chunks()`. The element is injected, then immediately discarded before the browser can play it.

Text mode works because it does **not** call `st.rerun()`, so the audio HTML survives.

**Fix Applied (pending_audio pattern):**
1. `stream_voice_query()` no longer renders the audio inline. It returns the chunks list.
2. After the voice query completes, the caller combines all chunks into a single base64 string and stores it in `st.session_state["pending_audio"]`.
3. At the **very top** of every script pass (before any other `st.` calls), the app checks for `pending_audio`, renders the `<audio autoplay>` element, and immediately clears the key (so it only plays once).
4. `st.rerun()` triggers a new pass → the pending audio is now rendered at the top of the fresh pass → browser receives and plays it.

**Files Changed:**
- `frontend/app.py` — complete rewrite of voice audio strategy using `pending_audio` session state pattern

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------| 
| Qdrant local persistent (not in-memory) | Survives restart, same setup as in-memory, just pass a path |
| BM25 via FastEmbed `Qdrant/bm25` | Native Qdrant sparse model, no external index |
| Gemini `client.aio` for async streaming | Proper async without thread overhead |
| TTS per-sentence via `asyncio.create_task` | Parallel synthesis = lower time-to-first-audio |
| LRU cache for LLM responses | No Redis needed, in-memory, evicts old entries |
| Grounding threshold 0.30 | RRF scores typically in 0.01-0.05 range; tune via env var |
| SSE format | Works with EventSource, Streamlit httpx streaming |
| **Two-layer PDF cache** | Layer 1: in-memory LRU (fast). Layer 2: Qdrant existence check (survives restarts) |
| **pending_audio pattern** | Audio base64 stored in session_state survives st.rerun(); rendered at top of next pass |

---

## Environment Variables Required

```env
GOOGLE_API_KEY=<your_gemini_api_key>
```

Ollama must be running with `nomic-embed-text` pulled.

---

## Files Created / Modified

```
app/main.py, config.py, dependencies.py
app/api/upload.py, query.py, voice.py
app/graph/graph.py, state.py, nodes.py, edges.py
app/rag/parser.py, chunker.py, embeddings.py, vector_store.py*, retriever.py
app/llm/gemini.py, prompts.py
app/speech/speech_to_text.py, edge_tts.py
app/cache/cache.py
app/memory/memory.py
app/guardrails/prompt_guard.py
app/models/request_models.py, response_models.py
app/utils/helpers.py, logger.py
app/services/pdf_service.py*, rag_service.py, tts_service.py, streaming_service.py
frontend/app.py*
tests/test_guardrails.py, test_helpers.py, test_cache.py, test_memory.py
requirements.txt, README.md, .env, pytest.ini

* = modified in session 3
```

---

## Test Results

```
33 passed in 0.37s
- test_guardrails: 13/13
- test_helpers: 8/8
- test_cache: 6/6
- test_memory: 6/6
```

---

## Next Session — What To Do

If continuing in a new session, the application is **fully implemented and bug-free**. Next actions:

1. **Set API key**: Edit `.env` → `GOOGLE_API_KEY=<real_key>`
2. **Start Ollama**: `ollama serve` + `ollama pull nomic-embed-text`
3. **Run backend**: `uvicorn app.main:app --reload`
4. **Run frontend**: `streamlit run frontend/app.py`
5. **Optional tuning**: Adjust `GROUNDING_THRESHOLD` if too many/few fallbacks

### Known Potential Issues

- `GROUNDING_THRESHOLD=0.30` — RRF scores from Qdrant are often very small (0.01-0.05 range); may need to lower to `0.01` or `0.02` after testing
- PyAudio on Windows may need manual wheel install if not already in venv
- `google-genai` version pinned to `1.16.0` — if streaming API changes, update `stream_generate_v2` in `app/llm/gemini.py`
- FastEmbed downloads `Qdrant/bm25` model on first run (~100MB) — ensure internet access
- STT accuracy: audio-recorder-streamlit captures at 16kHz which is good for Google STT. If accuracy is still poor, try increasing `pause_threshold` to give the speaker more time.

---

## Latency Profile (Expected)

| Metric | Target | Notes |
|--------|--------|-------|
| Time-to-first-token | < 2s | Retrieval + LLM startup |
| Time-to-first-audio | < 4s | First sentence TTS starts after ~1-2 sentences |
| PDF cache hit (memory) | < 100ms | Hash check + return collection name |
| PDF cache hit (Qdrant) | < 200ms | Collection existence check + count |
| LLM cache hit | < 500ms | Replay cached text + TTS full answer |
