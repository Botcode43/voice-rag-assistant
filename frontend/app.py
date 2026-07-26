"""
frontend/app.py — Streamlit frontend for Voice RAG Assistant.

Features:
- PDF upload → POST /upload
- Text question input + streamed answer display
- Voice recording via audio_recorder_streamlit → POST /voice-query
- Streamed audio playback per sentence (base64 MP3)
- Session memory scoped to the Streamlit session

Audio Bug Fix:
  st.rerun() wipes all dynamically injected HTML (including the audio element).
  Solution: store the combined base64 audio in session_state["pending_audio"]
  and render it at the *top* of the script on the next pass (before the audio
  data becomes stale), then immediately clear the key so it only plays once.
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Optional

import httpx
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice RAG Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://localhost:8000"

# ── Session state initialisation ───────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role", "content"}
if "pending_audio" not in st.session_state:
    st.session_state.pending_audio = None  # base64 string to autoplay once

# ── Render pending audio at the very TOP of each pass ─────────────────────────
# This fires on the rerun *after* a voice query completes and persists until
# Streamlit re-renders again (the autoplay tag fires before the DOM settles).
if st.session_state.pending_audio:
    _audio_b64 = st.session_state.pending_audio
    st.session_state.pending_audio = None  # consume immediately (play once)
    st.markdown(
        f"""
        <audio autoplay style="display:none">
            <source src="data:audio/mp3;base64,{_audio_b64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )


# ── Helper: SSE stream parser ──────────────────────────────────────────────────

def parse_sse_events(raw: str) -> list[dict]:
    """Parse raw SSE text into a list of event dicts."""
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
    return events


# ── Helper: combine base64 MP3 chunks into one base64 string ──────────────────

def combine_audio_chunks(audio_chunks: list[str]) -> Optional[str]:
    """Concatenate base64 audio chunks → single base64 string (or None)."""
    if not audio_chunks:
        return None
    decoded = []
    for chunk in audio_chunks:
        try:
            decoded.append(base64.b64decode(chunk))
        except Exception:
            pass
    if not decoded:
        return None
    return base64.b64encode(b"".join(decoded)).decode()


# ── Helper: play audio inline (for text-query path, no rerun needed) ──────────

def autoplay_audio_inline(audio_chunks: list[str]) -> None:
    """Concatenate chunks and render a single HTML5 autoplay element inline."""
    combined = combine_audio_chunks(audio_chunks)
    if not combined:
        return
    st.markdown(
        f"""
        <audio autoplay style="display:none">
            <source src="data:audio/mp3;base64,{combined}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )


# ── Helper: query backend and stream response ──────────────────────────────────

def stream_query(query: str) -> tuple[str, list[str]]:
    """
    POST /query with the text query, consume SSE stream.

    Returns:
        (full_answer_text, list_of_audio_b64_strings)
    """
    payload = {
        "session_id": st.session_state.session_id,
        "pdf_hash": st.session_state.pdf_hash,
        "query": query,
    }

    full_text = ""
    audio_chunks: list[str] = []
    text_placeholder = st.empty()

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", f"{BACKEND_URL}/query", json=payload) as resp:
                resp.raise_for_status()
                buffer = ""
                for raw_chunk in resp.iter_text():
                    buffer += raw_chunk
                    events = parse_sse_events(buffer)
                    if events:
                        buffer = ""
                        for event in events:
                            etype = event.get("type")
                            if etype == "text":
                                full_text += event.get("content", "")
                                text_placeholder.markdown(full_text)
                            elif etype == "audio":
                                ab64 = event.get("audio_b64", "")
                                if ab64:
                                    audio_chunks.append(ab64)
                            elif etype == "error":
                                st.error(f"Error: {event.get('error')}")
                            elif etype == "done":
                                break
    except httpx.HTTPError as exc:
        st.error(f"Backend request failed: {exc}")

    # For text queries there is no rerun, so inject the audio element directly.
    if audio_chunks:
        autoplay_audio_inline(audio_chunks)

    return full_text, audio_chunks


def stream_voice_query(audio_bytes: bytes, content_type: str = "audio/wav") -> tuple[str, str, list[str]]:
    """
    POST /voice-query with audio bytes, consume SSE stream.

    Returns:
        (transcribed_query, full_answer_text, audio_chunks)

    Audio chunks are NOT rendered here — the caller stores them in
    session_state["pending_audio"] so they survive the subsequent st.rerun().
    """
    full_text = ""
    audio_chunks: list[str] = []
    transcribed = ""

    try:
        with httpx.Client(timeout=180.0) as client:
            files = {"audio": ("recording.wav", audio_bytes, content_type)}
            data = {
                "pdf_hash": st.session_state.pdf_hash,
                "session_id": st.session_state.session_id,
            }
            text_placeholder = st.empty()

            with client.stream(
                "POST",
                f"{BACKEND_URL}/voice-query",
                files=files,
                data=data,
            ) as resp:
                resp.raise_for_status()
                # Transcription text arrives via a custom response header
                transcribed = resp.headers.get("x-transcribed-query", "")
                buffer = ""
                for raw_chunk in resp.iter_text():
                    buffer += raw_chunk
                    events = parse_sse_events(buffer)
                    if events:
                        buffer = ""
                        for event in events:
                            etype = event.get("type")
                            if etype == "text":
                                full_text += event.get("content", "")
                                text_placeholder.markdown(full_text)
                            elif etype == "audio":
                                ab64 = event.get("audio_b64", "")
                                if ab64:
                                    audio_chunks.append(ab64)
                            elif etype == "error":
                                st.error(f"Error: {event.get('error')}")
                            elif etype == "done":
                                break
    except httpx.HTTPError as exc:
        st.error(f"Backend request failed: {exc}")

    return transcribed, full_text, audio_chunks


# ── UI: Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎙️ Voice RAG Assistant")
    st.caption("Upload a PDF and ask questions — answered by voice, grounded in your document.")

    st.divider()
    st.subheader("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload the document you want to query.",
    )

    if uploaded_file is not None and (
        st.session_state.pdf_name != uploaded_file.name
        or st.session_state.pdf_hash is None
    ):
        with st.spinner("Processing PDF…"):
            try:
                resp = httpx.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=300.0,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.pdf_hash = data["pdf_hash"]
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
                if data.get("cached"):
                    st.success(f"✅ Loaded from cache ({data['num_chunks']} chunks)")
                else:
                    st.success(f"✅ PDF processed ({data['num_chunks']} chunks embedded)")
            except httpx.HTTPError as exc:
                st.error(f"Upload failed: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    if st.session_state.pdf_hash:
        st.info(f"**Active PDF:** {st.session_state.pdf_name}")
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")
        if st.button("🔄 Clear Session", use_container_width=True):
            st.session_state.pdf_hash = None
            st.session_state.pdf_name = None
            st.session_state.chat_history = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.pending_audio = None
            st.rerun()
    else:
        st.warning("Upload a PDF to begin.")

    st.divider()
    st.caption("Built with FastAPI · LangGraph · Gemini 2.5 Flash · Qdrant · Edge-TTS")


# ── UI: Main content area ──────────────────────────────────────────────────────

st.title("🎙️ Voice RAG Assistant")

if not st.session_state.pdf_hash:
    st.info("👈 Upload a PDF in the sidebar to start asking questions.")
    st.stop()

# Display chat history
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# ── Text Input (must be at top level, not inside columns) ─────────────────────
user_input = st.chat_input("Ask a question about the PDF…")

# ── Voice Input ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("🎙️ Voice Input")
    try:
        from audio_recorder_streamlit import audio_recorder

        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e8523a",
            neutral_color="#6b6b6b",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=3.0,
            sample_rate=16000,
            key="voice_recorder",
        )
    except ImportError:
        audio_bytes = None
        st.caption("Install audio-recorder-streamlit for voice input")


# ── Handle text input ──────────────────────────────────────────────────────────
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, audio_list = stream_query(user_input)

    if answer:
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ── Handle voice input ─────────────────────────────────────────────────────────
if audio_bytes and len(audio_bytes) > 1000:  # avoid processing noise-only captures
    # Only process if this is new audio (track by length to avoid re-runs)
    last_audio_key = f"last_audio_{st.session_state.session_id}"
    if st.session_state.get(last_audio_key) != len(audio_bytes):
        st.session_state[last_audio_key] = len(audio_bytes)

        with st.chat_message("user"):
            st.markdown("🎙️ *Voice input…*")

        with st.chat_message("assistant"):
            with st.spinner("Transcribing and thinking…"):
                transcribed, answer, audio_list = stream_voice_query(audio_bytes)

        # Build chat history entries
        user_msg = f"🎙️ {transcribed}" if transcribed else "🎙️ *Voice input (untranscribed)*"
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        if answer:
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # ── Key fix: store audio in session_state so it survives st.rerun() ──
        # The audio element injected inside stream_voice_query() is wiped by
        # st.rerun(). Instead we persist the combined base64 audio and render
        # it at the very top of the *next* script pass (see top of file).
        if audio_list:
            st.session_state.pending_audio = combine_audio_chunks(audio_list)

        # Rerun to refresh chat history display with correct transcription
        st.rerun()
