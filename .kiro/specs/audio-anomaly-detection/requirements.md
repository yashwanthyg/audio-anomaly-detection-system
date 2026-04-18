# Requirements Document

## Introduction

The Audio Anomaly Detection System is a Python application that processes audio files, transcribes speech to text using OpenAI Whisper, extracts three text features (text length, exclamation count, keyword count), and detects anomalies through keyword matching and an inline Isolation Forest trained on hardcoded normal samples. Results are displayed in a Streamlit dashboard with alerts for emergency keywords and statistical anomalies.

## Glossary

- **Transcription_Module**: The component that loads the Whisper model and converts audio files to plain text (`app/transcription.py`)
- **Processing_Module**: The component that cleans text and extracts a 3-element numerical feature vector from transcripts (`app/processing.py`)
- **Detection_Module**: The component that performs keyword scanning and Isolation Forest anomaly prediction (`app/detector.py`)
- **Dashboard**: The Streamlit web application that orchestrates the pipeline and displays results (`app/app.py`)
- **DetectionResult**: A dataclass holding the output of detection: transcript (str), keywords_detected (list[str]), is_anomaly (bool)
- **Feature_Vector**: A list of exactly 3 floats: [text_length, exclamation_count, keyword_count], used as input to the Isolation Forest model
- **Isolation_Forest**: A scikit-learn unsupervised ML model trained inline on hardcoded normal samples — no model files loaded from disk
- **Emergency_Keywords**: A configurable list of keywords (e.g., "help", "emergency", "fire") that trigger alerts when found in transcripts

## Requirements

### Requirement 1: Audio Transcription

**User Story:** As a user, I want to upload audio files and have them transcribed to text, so that I can analyze spoken content for anomalies.

#### Acceptance Criteria

1. WHEN an audio file in a supported format is provided, THE Transcription_Module SHALL transcribe the audio and return the speech content as a plain text string
2. WHEN the Whisper model has not been loaded, THE Transcription_Module SHALL automatically load the model before transcription
3. WHEN an audio file contains no detectable speech, THE Transcription_Module SHALL return an empty string

### Requirement 2: Text Cleaning and Normalization

**User Story:** As a system operator, I want raw transcripts normalized before analysis, so that feature extraction and keyword matching operate on consistent input.

#### Acceptance Criteria

1. WHEN raw transcript text is provided, THE Processing_Module SHALL return a lowercase string with all consecutive whitespace collapsed to single spaces and leading/trailing whitespace removed
2. THE Processing_Module SHALL produce cleaned text whose length is less than or equal to the length of the raw input for all input strings
3. WHEN an empty string is provided, THE Processing_Module SHALL return an empty string without raising an exception

### Requirement 3: Feature Extraction

**User Story:** As a data scientist, I want three numerical features extracted from transcripts, so that the Isolation Forest model can classify text as normal or anomalous.

#### Acceptance Criteria

1. WHEN cleaned text and a keyword list are provided, THE Processing_Module SHALL return a Feature_Vector of exactly 3 float values: [text_length, exclamation_count, keyword_count]
2. WHEN cleaned text and a keyword list are provided, THE Processing_Module SHALL set the first element (text_length) equal to the character count of the input text
3. THE Processing_Module SHALL produce non-negative values for all three elements of the Feature_Vector

### Requirement 4: Keyword Detection

**User Story:** As a security analyst, I want transcripts scanned for emergency keywords, so that I receive immediate alerts when dangerous terms appear.

#### Acceptance Criteria

1. WHEN text and a keyword list are provided, THE Detection_Module SHALL return only keywords that appear in the input text and are present in the keyword list
2. WHEN no keywords from the list appear in the text, THE Detection_Module SHALL return an empty list

### Requirement 5: Anomaly Detection and Result Composition

**User Story:** As a security analyst, I want transcripts analyzed for anomalies and results combined into a single output, so that I can see all findings in one place.

#### Acceptance Criteria

1. THE Detection_Module SHALL initialize an Isolation Forest inline by fitting it on hardcoded normal samples — no model files shall be loaded from disk
2. THE Detection_Module SHALL flag the result as anomalous if either keywords are detected or the Isolation Forest predicts an anomaly
3. THE Detection_Module SHALL set the transcript field of DetectionResult to the cleaned input text
4. THE DetectionResult SHALL contain exactly three fields: transcript (str), keywords_detected (list[str]), is_anomaly (bool)

### Requirement 6: Dashboard Interface

**User Story:** As a user, I want a web dashboard to upload audio and view detection results, so that I can interact with the system without using the command line.

#### Acceptance Criteria

1. THE Dashboard SHALL provide an audio file upload widget accepting WAV and MP3 formats
2. WHEN a user uploads an audio file, THE Dashboard SHALL execute the full pipeline: transcribe, clean, extract features, and detect anomalies
3. WHEN detection is complete, THE Dashboard SHALL display the transcript text to the user
4. WHEN keywords are found in the transcript, THE Dashboard SHALL display an alert listing the detected keywords
5. WHEN an anomaly is detected, THE Dashboard SHALL display a warning message
6. WHEN no anomalies are detected, THE Dashboard SHALL display a success message
7. THE Dashboard SHALL NOT include history, logs, or any widgets beyond audio upload, transcript display, and alert display

### Requirement 7: Error Handling

**User Story:** As a user, I want the system to handle errors gracefully, so that failures do not crash the application.

#### Acceptance Criteria

1. IF the Whisper model fails to load, THEN THE Dashboard SHALL display a "Transcription unavailable" message
2. IF the audio file is corrupted or unreadable, THEN THE Dashboard SHALL display an error message and allow re-upload
3. WHEN an empty transcript is produced, THE Dashboard SHALL display a "No speech detected" message and skip detection
