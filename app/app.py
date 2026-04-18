"""Streamlit dashboard for Audio Anomaly Detection System."""

import logging
import os
import tempfile
import time

import streamlit as st

from config import CONFIG
from detector import build_model, detect
from processing import clean_text, extract_features
from transcription import load_model, transcribe

try:
    from realtime import result_queue, start_stream, stop_stream
    REALTIME_AVAILABLE = True
except Exception:
    REALTIME_AVAILABLE = False
logger = logging.getLogger(__name__)


def _save_to_temp(raw_bytes: bytes, suffix: str) -> str:
    """Write raw audio bytes to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        return tmp.name


def _run_pipeline(raw_bytes: bytes, suffix: str, model) -> None:
    """Save audio bytes to a temp file, run the full pipeline, show results."""
    temp_path = None
    try:
        temp_path = _save_to_temp(raw_bytes, suffix)

        if not os.path.exists(temp_path):
            st.error("Failed to create temporary audio file.")
            return

        st.audio(raw_bytes)

        with st.spinner("Processing audio..."):
            transcript = transcribe(temp_path)

        if not transcript:
            st.info("No speech detected in the audio file.")
            return

        cleaned = clean_text(transcript)
        features = extract_features(cleaned, CONFIG["keywords"])
        result = detect(cleaned, features, CONFIG["keywords"], model)

        st.header("📊 Results")

        st.subheader("📝 Transcript")
        st.write(result.transcript)

        if result.keywords_detected:
            st.subheader("🔑 Detected Keywords")
            st.write(", ".join(result.keywords_detected))

        if result.keywords_detected:
            st.error(
                f"🚨 Emergency Alert — Keywords detected: "
                f"{', '.join(result.keywords_detected)}"
            )

        if result.is_anomaly and not result.keywords_detected:
            st.warning("⚠️ Anomaly Detected — Unusual text pattern identified.")
        elif result.is_anomaly and result.keywords_detected:
            st.warning("⚠️ Anomaly Detected — Keywords and unusual pattern found.")
        elif not result.is_anomaly:
            st.success("✅ Normal — No anomalies detected.")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        st.error(f"Error: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _render_live_section() -> None:
    """Pull results from the queue and display live monitoring output."""
    st.subheader("🔴 Live Monitoring Output")

    # Initialize session state for live data
    if "live_data" not in st.session_state:
        st.session_state.live_data = None

    # Drain the queue — keep the latest result
    while not result_queue.empty():
        try:
            st.session_state.live_data = result_queue.get_nowait()
        except Exception:
            break

    data = st.session_state.live_data
    if data:
        st.write(f"📝 {data['text']}")
        if data["keywords"]:
            st.error(f"🚨 Keywords: {', '.join(data['keywords'])}")
        if data["anomaly"]:
            st.warning("⚠️ Anomaly detected")
    else:
        st.caption("Waiting for audio input...")


def main():
    """Run the Audio Anomaly Detection dashboard."""
    st.title("🎙️ Audio Anomaly Detection System")

    try:
        load_model(CONFIG["whisper_model"])
    except RuntimeError as e:
        st.error(f"Transcription unavailable: {e}")
        return

    model = build_model(CONFIG["contamination"])

    # ── Real-Time Monitoring ──
    if REALTIME_AVAILABLE:
        st.header("⚡ Real-Time Monitoring")

        if "stream" not in st.session_state:
            st.session_state.stream = None

        col1, col2 = st.columns(2)

        with col1:
            if st.button("▶️ Start Monitoring"):
                st.session_state.stream = start_stream(model)
                st.success("Real-time monitoring started")

        with col2:
            if st.button("⏹️ Stop Monitoring"):
                if st.session_state.stream:
                    stop_stream(st.session_state.stream)
                    st.session_state.stream = None
                    st.warning("Monitoring stopped")
    else:
        st.info("Real-time monitoring is available only in local environment.")

    if REALTIME_AVAILABLE:
        while not result_queue.empty():
            st.session_state.live_data = result_queue.get()

    # ── File / Mic Input ──
    st.header("🎧 Choose Input Method")

    tab_upload, tab_mic = st.tabs(["📁 Upload File", "🎤 Record from Microphone"])

    with tab_upload:
        uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1].lower()
            raw_bytes = uploaded.getvalue()
            if raw_bytes:
                _run_pipeline(raw_bytes, suffix, model)
            else:
                st.warning("Uploaded file is empty.")

    with tab_mic:
        recorded = st.audio_input("Record audio from your microphone")
        if recorded is not None:
            raw_bytes = recorded.getvalue()
            if raw_bytes:
                _run_pipeline(raw_bytes, ".wav", model)
            else:
                st.warning("No audio recorded.")

    # ── Controlled rerun for live streaming only ──
    if st.session_state.streaming:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
