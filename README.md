# TalkWithMe

An AI agent pipeline to practice talking in different languages.

## Architecture

The application has been upgraded to a modern Client-Server architecture:
- **Backend**: A `FastAPI` Python server that hosts the heavy AI models (`whisper` for Speech-to-Text, `llama_cpp` for the Brain/LLM, and Coqui `TTS` for Text-to-Speech).
- **Frontend**: A sleek, modern Vanilla HTML/JS web application with a clean light-mode aesthetic.

### Features
- **Hold to Talk**: Record audio via the browser and send it to the backend.
- **Multilingual Support**: Supports practicing in French, Russian, and Arabic.
- **Agentic Pipeline**: Generates AI responses as well as suggestions for how you could reply.

## Setup & Installation

All Python dependencies must be run inside a virtual environment to prevent global package conflicts.

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
2. Install all required dependencies using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the setup script to download the Language Model (The "Brain"):
   ```bash
   python setup_models.py
   ```
   *Note: This script downloads Llama-3.2-3B (~2GB) into the `models/` folder and pre-downloads the `faster-whisper` base model. You only need to run this once. The Text-to-Speech (Piper) models will still automatically download the very first time you select a new language!*

## Running the Application

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
2. Start the FastAPI backend server:
   ```bash
   uvicorn backend.main:app --reload
   ```
3. Open your web browser and navigate to the frontend application:
   [http://localhost:8000/app](http://localhost:8000/app)
