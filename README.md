---
title: Meeting Minutes
emoji: 📚
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
short_description: Meeting Minutes
---

# Meeting Minutes Generator

Welcome to the **Meeting Minutes Generator**, a GenAI-powered application designed to transcribe audio recordings of meetings and automatically generate concise, accurate meeting minutes.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Supported Formats](#supported-formats)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview

This app leverages Generative AI and speech-to-text technology to transform audio files into structured meeting summaries. It’s a handy tool for anyone looking to save time on note-taking and ensure meeting information is documented accurately.

## Features

- **Audio Transcription**: Converts uploaded audio files into text.
- **Automated Summarization**: Summarizes the transcriptions into key meeting points and actionable items.
- **Customizable Output**: Choose summary formats based on preferences (bulleted points, narrative summary, etc.).
- **Easy Interface**: Simple and intuitive interface built with Streamlit.
  
## Installation

1. **Clone this repository**:
   ```bash
   git clone <repo-url>
   cd meeting-minutes-generator

2. pip install -r requirements.txt

3.  streamlit run app.py

## Usage

1. Open the app in your browser (usually at http://localhost:8501).
2. Upload an audio file in a supported format (e.g., .wav, .mp3).
3. Click Generate Minutes to transcribe and summarize the audio.
4. Review and download the generated meeting minutes in text format.

## Supported Formats

1. Audio files: .wav, .mp3
2. Summarized output: Plain text, Markdown

## Tech Stack

Frontend: Streamlit
Backend: Python, GenAI for summarization
Transcription: Speech-to-text service integration (like Google Speech-to-Text, Whisper API)

## Future Improvements

1. Add multilingual support for transcription and summarization.
2. Integrate with popular calendar and meeting platforms.
3. Customize summaries based on meeting context (e.g., brainstorming, project updates).

## Contributing
Contributions are welcome! Please open an issue to discuss what you'd like to change or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
