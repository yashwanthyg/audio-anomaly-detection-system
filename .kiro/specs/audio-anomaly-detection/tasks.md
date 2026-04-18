# Implementation Tasks

## Task 1: Project Setup and Configuration

- [x] 1.1 Create project directory structure: `app/` folder with `__init__.py`
- [x] 1.2 Create `requirements.txt` with dependencies: `openai-whisper`, `torch`, `scikit-learn`, `streamlit`, `numpy`, `hypothesis`, `pytest`
- [x] 1.3 Create `app/config.py` with CONFIG dict containing `whisper_model`, `keywords`, `contamination`, and `supported_formats`

## Task 2: Transcription Module

- [x] 2.1 Create `app/transcription.py` with `load_model(size)` function that loads a Whisper model into a module-level variable
- [x] 2.2 Implement `transcribe(file_path)` function that auto-loads the model if needed, calls `model.transcribe()`, and returns stripped text

## Task 3: Text Processing Module

- [x] 3.1 Create `app/processing.py` with `clean_text(raw)` function that lowercases, collapses whitespace, and strips the input
- [x] 3.2 Implement `extract_features(text, keywords)` function that returns a 3-element float list: `[text_length, exclamation_count, keyword_count]`

## Task 4: Detection Module

- [x] 4.1 Create `app/detector.py` with `DetectionResult` dataclass containing exactly three fields: `transcript` (str), `keywords_detected` (list[str]), `is_anomaly` (bool)
- [x] 4.2 Define `NORMAL_SAMPLES` as a hardcoded list of 10 normal feature vectors (3 floats each)
- [x] 4.3 Implement `build_model(contamination)` function that creates and fits an IsolationForest on `NORMAL_SAMPLES`
- [x] 4.4 Implement `scan_keywords(text, keywords)` function that returns keywords found in the text
- [x] 4.5 Implement `detect(text, features, keywords, model)` function that combines keyword scan and model prediction, returning a `DetectionResult` with `is_anomaly = keyword_match OR model_anomaly`

## Task 5: Streamlit Dashboard

- [x] 5.1 Create `app/app.py` with `main()` function that sets up the Streamlit page title, loads the Whisper model, and builds the Isolation Forest inline
- [x] 5.2 Add audio file uploader widget accepting WAV and MP3 formats, writing uploaded file to a temp file for Whisper
- [x] 5.3 Wire the full pipeline on upload: transcribe → clean → extract features → detect
- [x] 5.4 Display transcript, keyword alert (st.error), anomaly warning (st.warning), or success message (st.success)
- [x] 5.5 Handle edge cases: empty transcript shows "No speech detected", model/transcription errors show appropriate messages

## Task 6: Unit and Property Tests

- [ ] 6.1 Create `tests/test_processing.py` with unit tests for `clean_text` (lowercase, whitespace collapse, empty string) and `extract_features` (vector length, text_length correctness, non-negativity)
- [ ] 6.2 Create `tests/test_detector.py` with unit tests for `scan_keywords` (found/not-found), `build_model` (returns fitted model), and `detect` (anomaly flag logic, transcript preservation)
- [ ] 6.3 Add Hypothesis property tests: `clean_text` never grows text, `extract_features` always returns 3 non-negative floats, `scan_keywords` results are subset of both text and keyword list
