import os
import io
import time
import numpy as np
import soundfile as sf
import subprocess
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import pathlib
import re
import threading

try:
    from llama_cpp import Llama
    from faster_whisper import WhisperModel
except ImportError:
    print("Warning: ML models not found. Running in mock mode.")
    Llama = None
    WhisperModel = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a directory for audio outputs and models
os.makedirs("static", exist_ok=True)
os.makedirs("models", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount frontend
frontend_dir = pathlib.Path(__file__).parent.parent / "frontend"
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")


# Globals
whisper_model = None
llm = None
current_lang = "French"

# Mapping language to the local piper model name
model_paths = {
    "French": "models/fr_FR-upmc-medium.onnx",
    "Russian": "models/ru_RU-irina-medium.onnx",
    "Arabic": "models/ar_JO-kareem-low.onnx"
}

# Mapping for downloading
piper_download_urls = {
    "French": "fr/fr_FR/upmc/medium/fr_FR-upmc-medium",
    "Russian": "ru/ru_RU/irina/medium/ru_RU-irina-medium",
    "Arabic": "ar/ar_JO/kareem/low/ar_JO-kareem-low"
}

def remove_emojis(text):
    emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
    return re.sub(emoji_pattern, '', text)

def download_piper_voice(lang):
    if lang not in piper_download_urls:
        return
        
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
    path_prefix = piper_download_urls[lang]
    local_onnx = model_paths[lang]
    local_json = local_onnx + ".json"
    
    # Download ONNX
    if not os.path.exists(local_onnx):
        print(f"Downloading {lang} ONNX model...")
        r = requests.get(base_url + path_prefix + ".onnx", stream=True)
        with open(local_onnx, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
    # Download JSON Config
    if not os.path.exists(local_json):
        print(f"Downloading {lang} JSON config...")
        r = requests.get(base_url + path_prefix + ".onnx.json")
        with open(local_json, 'wb') as f:
            f.write(r.content)

def init_models(lang="French"):
    global whisper_model, llm, current_lang
    current_lang = lang
    print(f"Loading models for {lang}...")
    
    # Ensure Piper voice is downloaded
    download_piper_voice(lang)
    
    if whisper_model is None and WhisperModel:
        print("Loading Faster Whisper...")
        # device="auto" allows CPU or GPU, compute_type="int8" is memory efficient
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        
    if llm is None and Llama:
        print("Loading Llama...")
        try:
            llm = Llama(model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf", verbose=False)
        except Exception as e:
            print("Could not load Llama model:", e)
            
    print("Models loaded.")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=init_models, args=(current_lang,)).start()

@app.post("/api/models")
def change_model(lang: str = Form(...)):
    if lang not in model_paths:
        return JSONResponse({"error": "Language not supported"}, status_code=400)
    
    threading.Thread(target=init_models, args=(lang,)).start()
    return {"status": "success", "message": f"Switching to {lang}..."}

@app.get("/api/languages")
def get_languages():
    return {"languages": list(model_paths.keys()), "current": current_lang}

@app.post("/api/chat")
async def chat(audio: UploadFile = File(...)):
    try:
        content = await audio.read()
        
        # Browsers record in webm or ogg depending on the engine.
        # We save the raw bytes and let faster-whisper (which uses ffmpeg) decode it natively.
        temp_audio = "temp_recording.webm"
        with open(temp_audio, "wb") as f:
            f.write(content)
        
        # Transcribe with Faster Whisper
        if whisper_model:
            segments, _ = whisper_model.transcribe(temp_audio, beam_size=5)
            user_text = "".join([segment.text for segment in segments]).strip()
        else:
            user_text = "Mock Transcription due to missing model"
            
        if not user_text:
            user_text = "[Silence]"
            
        # Generate AI response
        if llm:
            prompt = f"User says: {user_text}\nRespond as a helpful conversational partner in {current_lang}: "
            output = llm(prompt, max_tokens=50)
            response_text = remove_emojis(output["choices"][0]["text"].strip())
            
            suggest_prompt = f"User said: {user_text}\nAI responded: {response_text}\nSuggest 2 short phrases the user could reply with in {current_lang}:\n1."
            suggest_output = llm(suggest_prompt, max_tokens=40)
            suggestions = "1. " + suggest_output["choices"][0]["text"].strip()
        else:
            response_text = f"Mock response in {current_lang}"
            suggestions = "1. Mock suggestion 1\n2. Mock suggestion 2"
            
        # Text-to-Speech using Piper
        output_audio_path = f"static/output_{int(time.time())}.wav"
        piper_model_file = model_paths[current_lang]
        
        if os.path.exists(piper_model_file):
            # Run piper as a subprocess
            command = [
                "piper",
                "--model", piper_model_file,
                "--output_file", output_audio_path
            ]
            subprocess.run(command, input=response_text.encode('utf-8'), check=False)
        else:
            # Mock audio file if piper isn't set up
            sf.write(output_audio_path, np.zeros(16000), 16000)
            
        return {
            "user_text": user_text,
            "ai_text": response_text,
            "suggestions": suggestions,
            "audio_url": f"/{output_audio_path}"
        }
        
    except Exception as e:
        print("Error processing chat:", e)
        return JSONResponse({"error": str(e)}, status_code=500)
