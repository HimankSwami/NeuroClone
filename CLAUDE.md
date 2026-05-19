# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Neuro-Clone is a local AI assistant running on Linux. It features a personality-driven LLM interface (via Ollama), web search capabilities, system monitoring, and a sophisticated voice pipeline (Piper TTS + RVC voice conversion).

## Architecture
- **Core Logic (`brain/`)**: 
  - `core.py`: The `NeuroBrain` class manages the LLM interaction, maintains conversation history, and handles specialized triggers:
    - **Web Search**: Triggered by keywords like "search" or "latest". Uses DuckDuckGo for URL discovery and `crawl4ai` for content extraction.
    - **System Stats**: Triggered by keywords like "cpu" or "ram". Uses `psutil` and `pynvml` to monitor host hardware.
    - **Claw Engineer**: A specialized coding agent triggered via `claw:` prefix. It delegates tasks to a compiled Rust binary.
- **Voice Pipeline (`voice/`)**:
  - `speaker.py`: Implements a three-stage voice output:
    1. **TTS**: Generates base audio using Piper.
    2. **RVC**: Converts base audio to the "Ayaka" voice using RVC (Retrieval-based Voice Conversion).
    3. **Playback**: Plays the final audio via `aplay`.
- **Main Entry (`main.py`)**: Handles the primary execution loop, managing either text-based or voice-based (speech recognition) user interaction.
- **Web Interface (`body/`)**: Contains `app.py` and static assets for a web-based UI.

## Development Commands
- **Run Main Loop (Text Mode)**: `python main.py`
- **Run Main Loop (Voice Mode)**: `python main.py --voice`
- **Environment**: Uses a local virtual environment in `venv/`. Ensure it is activated when running scripts directly.

## Key Technologies
- **LLM Engine**: Ollama (local inference)
- **Voice Input**: `speech_recognition` (Google API)
- **Voice Output**: Piper (TTS) + RVC (Voice Conversion)
- **Web Intelligence**: `duckduckgo_search` + `crawl4ai`
- **System Monitoring**: `psutil` + `pynvml` (NVIDIA)
