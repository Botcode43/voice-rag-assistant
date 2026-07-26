# Voice RAG Assistant — Technical Write-Up

## System Design

The system is a streaming RAG pipeline with voice I/O, built for minimal time-to-first-word and time-to-first-audio.

### Architecture

**Backend (FastAPI)** exposes `/upload`, `/query`, `/voice-query`, and `/stream`. The query endpoints return Server-Sent Events (SSE) with interleaved `text` and `audio` chunks so the frontend can display text and play audio simultaneously.

**LangGraph Pipeline** (5 nodes, no LLM calls in guard/routing):
1. **Prompt Injection Check** — regex matching against 10+ patterns; rejects in < 1ms
2. **Query Decomposition** — anaphora resolution via session memory (rule-based, no LLM)
3. **Retrieval** — Qdrant hybrid search: nomic-embed-text dense vectors + BM25 sparse vectors (FastEmbed), fused server-side via RRF
4. **Grounding Check** — top RRF score vs configurable threshold; if below threshold → returns fallback immediately, no LLM call
5. **Streaming Generation** (outside graph) — Gemini 2.5 Flash async stream → per-sentence TTS in parallel

### Key Design Choices

**Hybrid Search via Qdrant's Query API**: Each chunk is stored with both a dense vector (768-dim nomic-embed-text) and a sparse BM25 vector (FastEmbed `Qdrant/bm25`). A single `query_points` call with two `Prefetch` clauses and `FusionQuery(RRF)` handles retrieval and fusion server-side with a single network round trip.

**Per-Sentence TTS Streaming**: The text stream is buffered and `asyncio.create_task()` is called as each sentence boundary is detected. TTS synthesis runs concurrently with continued LLM generation, reducing time-to-first-audio by 1–3 sentences worth of generation time.

**Two-Cache Strategy**:
- *PDF Cache* (sha256 → collection name): avoids re-parsing/re-embedding on re-upload
- *LLM Cache* (sha256(pdf_hash + norm_query) → full answer): instant replay on identical questions, with simulated streaming delay for UX consistency

**No Reranker / No Answer Validation LLM call**: Both were removed. Grounding is handled by the numeric similarity threshold check, which adds ~0ms overhead vs an additional model call.

### Latency Profile

| Metric | Typical |
|--------|---------|
| Time-to-first-token | 1.5–3s |
| Time-to-first-audio | 3–5s (after first sentence completes) |
| On PDF cache hit | < 100ms to first token |
| On LLM cache hit | < 1s (streaming replay) |

## What Would I Add with More Time

1. **Reranker** (cross-encoder) — removed per spec, but would improve precision for long PDFs
2. **WebSocket** for bidirectional streaming — simpler client implementation than SSE + POST
3. **PDF section metadata** — store page number and heading in payload for source citations
4. **Adaptive thresholding** — track retrieval score distribution per document and set threshold relative to mean
5. **Streaming audio to Streamlit** via `st.audio` with audio segments appended progressively (current approach uses autoplay HTML injection)
