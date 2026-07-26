# Voice RAG AI Employee — Project Spec

## 1. Objective

Build a voice-enabled RAG assistant that:

1. Accepts a PDF upload.
2. Answers questions (text or voice) **only** using content from that PDF.
3. Speaks the answer back using TTS.
4. Streams both text and audio for low time-to-first-word / time-to-first-audio.
5. Clearly says *"I couldn't find that information in the uploaded document."* when the answer isn't grounded in the PDF.

This is scoped for a **24-hour take-home**, so the design deliberately favors speed and correctness over retrieval sophistication.

---

## 2. Scope Decisions (what's in, what's cut, and why)

| Feature | Status | Reason |
|---|---|---|
| Query decomposition | **Kept** | Handles follow-ups ("who is the author?" → "when was it published?") using session memory. Cheap, no extra network round trip if done as part of the same LLM call or a lightweight rule-based split. |
| Hybrid search | **Kept — via Qdrant native hybrid search** | Use Qdrant's own hybrid search (dense + sparse vectors in one collection, fused server-side with RRF via the Query API) instead of running BM25 separately and merging in application code. Same retrieval quality benefit, less code, one round trip instead of two, and no manual score-merging logic to get wrong under time pressure. |
| Reranker | **Removed** | Extra model call = extra latency, marginal benefit on a single small-to-medium document. |
| Multi-hop retrieval | **Removed** | Not needed for single-PDF QA; adds retrieval round-trips and complexity with no rubric payoff. |
| Standalone Answer Validation (2nd LLM call) | **Removed** | Replaced with a cheap grounding heuristic: check the top retrieved chunk's similarity score against a threshold *before* generation. If below threshold, skip generation entirely and return the fallback line immediately — this also improves latency since you avoid calling the LLM on ungrounded queries. |
| Prompt Injection Guard | **Kept, but lightweight** | Regex / keyword match against known injection phrases (see below), not an LLM call. Must be near-zero latency. |
| PDF caching | **Kept** | Hash the uploaded file (sha256). If seen before, reuse existing vector collection — skip re-parse + re-embed. |
| LLM response caching | **Kept** | Cache by `(pdf_hash, normalized_query)`. Repeated identical questions return instantly. |
| Embeddings-only cache / retriever-only cache | **Removed** | Redundant once PDF cache exists — the PDF cache already avoids recomputation for anything else the cache would have prevented. |
| Streaming (text + TTS) | **Kept — top priority** | This is what's actually being scored. Stream LLM tokens immediately; kick off TTS per-sentence rather than waiting for the full answer. |

---

## 3. Tech Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI (async endpoints) |
| Agent orchestration | LangGraph |
| LLM | Gemini 2.5 Flash (streaming) |
| Embeddings | nomic-embed-text |
| Vector DB | Qdrant (local/in-memory mode is fine for the take-home — avoids Docker setup time). Store both a dense vector (nomic-embed-text) and a sparse vector (Qdrant/FastEmbed BM25 sparse embedding) per chunk; retrieve via Qdrant's Query API hybrid mode with RRF fusion — no separate BM25 index or manual merge step needed. |
| PDF parsing | PyMuPDF |
| Chunking | `RecursiveCharacterTextSplitter`, token-aware length function, `chunk_size=700`, `chunk_overlap=120` |
| Speech-to-text | `SpeechRecognition` |
| Text-to-speech | `edge-tts` |
| Caching | Local dict/LRU or `diskcache` (no Redis needed for this scope) |
| Frontend | Streamlit |

---

## 4. LangGraph Flow (Simplified)

```
START
  │
  ▼
Prompt Injection Check  (fast, regex/keyword — reject & short-circuit if flagged)
  │
  ▼
Query Decomposition     (uses session memory for follow-ups)
  │
  ▼
Retrieval               (Qdrant native hybrid: dense + sparse, RRF-fused server-side)
  │
  ▼
Context / Grounding Check  (similarity threshold — no LLM call)
  │
  ├── No context ──────────► Return fallback line ──────────► END
  │
  ▼ Has context
Gemini Flash (streaming generation)
  │
  ▼
Stream Text  ──┬─► Generate & Stream Speech (Edge TTS, per-sentence as they complete)
               │
               ▼
              END
```

Removed nodes vs. the original draft: **Reranking**, standalone **Answer Validation**.

---

## 5. Prompt Injection Guard — reject patterns

Block/short-circuit (case-insensitive) on inputs resembling:
- "ignore previous instructions"
- "reveal system prompt" / "show your instructions"
- "forget the uploaded document" / "ignore the pdf"
- "act as ChatGPT" / "pretend you are..."
- "tell me your hidden instructions"

Only document-grounded questions are allowed through.

---

## 6. LLM System Prompt

```
You are an AI assistant that answers questions ONLY from the retrieved document context.

Rules:
- Never use outside knowledge.
- Never guess or fabricate information.
- If the answer cannot be found in the context, reply exactly:
  "I couldn't find that information in the uploaded document."
- Keep answers concise and factual.
```

---

## 7. Session Memory

- In-memory (per session/websocket connection), not persisted.
- Stores last N turns so follow-up questions resolve correctly via query decomposition.

---

## 8. Caching Details

**PDF cache**
- Key: `sha256(pdf_bytes)`
- Value: reference to existing Qdrant collection (or in-memory vector store)
- On upload: if hash exists, skip parse/chunk/embed entirely and reuse.

**LLM response cache**
- Key: `hash(pdf_hash + normalized_query)`
- Value: cached streamed answer (store full text; replay as a stream on cache hit so UX stays consistent)
- Skipped when grounding check fails (no point caching "not found" per near-duplicate phrasing — cache only successful, grounded answers).

---

## 9. FastAPI Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | Upload PDF, parse, chunk, embed, cache |
| POST | `/query` | Text query → streamed answer |
| POST | `/voice-query` | Voice input → STT → same pipeline as `/query` |
| GET | `/stream` | Streaming response endpoint (SSE or chunked) |

---

## 10. File Structure

```
voice-rag-assistant/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   │
│   ├── api/
│   │      upload.py
│   │      query.py
│   │      voice.py
│   │
│   ├── graph/
│   │      graph.py
│   │      state.py
│   │      nodes.py
│   │      edges.py
│   │
│   ├── rag/
│   │      parser.py
│   │      chunker.py
│   │      embeddings.py
│   │      vector_store.py
│   │      retriever.py
│   │
│   ├── llm/
│   │      gemini.py
│   │      prompts.py
│   │
│   ├── speech/
│   │      speech_to_text.py
│   │      edge_tts.py
│   │
│   ├── cache/
│   │      cache.py
│   │
│   ├── memory/
│   │      memory.py
│   │
│   ├── guardrails/
│   │      prompt_guard.py
│   │
│   ├── models/
│   │      request_models.py
│   │      response_models.py
│   │
│   ├── utils/
│   │      helpers.py
│   │      logger.py
│   │
│   └── services/
│          pdf_service.py
│          rag_service.py
│          tts_service.py
│          streaming_service.py
│
├── frontend/
│      app.py
│
├── md/
│      PROJECT.md          ← this file
│      WRITEUP.md           ← 1-page submission write-up
│
├── uploads/
├── qdrant_storage/
├── tests/
├── requirements.txt
├── README.md
├── .env
├── .gitignore
└── docker-compose.yml      (optional — only if Qdrant needs a container; skip if using in-memory/local mode)
```

Note vs. original draft: `reranker.py` and `validator.py` are removed since those layers were cut.

---

## 11. Non-Functional Requirements

- Clean, modular code; type hints throughout; Pydantic models for all request/response schemas.
- Async FastAPI endpoints.
- Structured logging, graceful exception handling.
- Fully runnable after `pip install -r requirements.txt` + `.env` configuration.

---

## 12. Outcome / Definition of Done

- [ ] Upload a PDF → parsed, chunked, embedded, stored.
- [ ] Ask a question grounded in the PDF (text or voice) → correct streamed text + streamed audio answer, fast time-to-first-token.
- [ ] Ask a question NOT in the PDF → correctly returns the fallback line, no hallucination.
- [ ] Re-upload the same PDF → cache hit, no re-embedding.
- [ ] Ask the same question twice → second time served from LLM response cache.
- [ ] Prompt injection attempt → blocked before hitting the LLM.
- [ ] Follow-up question ("when was it published?") resolved correctly using session memory.
- [ ] Latency numbers measured and recorded (time-to-first-token, time-to-first-audio) for the write-up.