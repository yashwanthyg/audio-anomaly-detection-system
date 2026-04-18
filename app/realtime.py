"""Real-time audio streaming with queue-based UI communication.

The background thread pushes results into a thread-safe queue.
The Streamlit main thread pulls from the queue on each rerun.
No st.session_state access from the worker thread.
"""

import os
import queue
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import CONFIG
from detector import detect
from processing import clean_text, extract_features
from transcription import transcribe

# Thread-safe queue: worker pushes, UI pulls
result_queue: queue.Queue = queue.Queue()

_running = False
_lock = threading.Lock()


def _is_running() -> bool:
    with _lock:
        return _running


def _set_running(value: bool) -> None:
    global _running
    with _lock:
        _running = value


def _process_loop(model) -> None:
    """Worker thread: buffer mic audio, transcribe chunks, push results."""
    buffer: list[float] = []
    sample_rate = 16000
    chunk_duration = 3  # seconds per chunk
    chunk_size = sample_rate * chunk_duration
    audio_q: queue.Queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}")
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=sample_rate,
        dtype="float32",
    )
    stream.start()

    try:
        while _is_running():
            try:
                data = audio_q.get(timeout=0.5)
                buffer.extend(data.flatten().tolist())
            except queue.Empty:
                continue

            if len(buffer) >= chunk_size:
                chunk = np.array(buffer[:chunk_size], dtype=np.float32)
                buffer = buffer[chunk_size:]

                # Write chunk to temp file for Whisper
                tmp_path = os.path.join(
                    os.path.dirname(__file__), "..", "_rt_chunk.wav"
                )
                tmp_path = os.path.abspath(tmp_path)
                sf.write(tmp_path, chunk, sample_rate)

                try:
                    text = transcribe(tmp_path)
                    if text:
                        cleaned = clean_text(text)
                        keywords = CONFIG["keywords"]
                        features = extract_features(cleaned, keywords)
                        result = detect(cleaned, features, keywords, model)

                        # Push to queue — UI will pull on next rerun
                        result_queue.put({
                            "text": text,
                            "keywords": result.keywords_detected,
                            "anomaly": result.is_anomaly,
                        })
                except Exception as e:
                    print(f"[realtime] Processing error: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
    finally:
        stream.stop()
        stream.close()


def start_stream(model) -> None:
    """Start the real-time monitoring thread."""
    if _is_running():
        return
    _set_running(True)
    t = threading.Thread(target=_process_loop, args=(model,), daemon=True)
    t.start()


def stop_stream() -> None:
    """Stop the real-time monitoring thread."""
    _set_running(False)
