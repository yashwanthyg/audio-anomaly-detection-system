# Audio Anomaly Detection System

## Overview
This project transcribes audio and detects emergency keywords and anomalous speech patterns.

## Features
- Audio to text transcription (Whisper)
- Keyword-based alert detection
- Anomaly detection using Isolation Forest
- Microphone-based input (record and process)

## Demo
[[Demo video link]](https://drive.google.com/file/d/1L-OZE-tvgSyUVPVUdJZVCkd3hq_xCzNb/view?usp=drivesdk)

## How to Run
pip install -r requirements.txt
streamlit run app/app.py

## Note
Real-time streaming is implemented locally. The deployed version uses file-based processing.
