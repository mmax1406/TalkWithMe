import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import numpy as np
import re
import sounddevice as sd

from llama_cpp import Llama
import whisper
from TTS.api import TTS

# Globals
SAMPLE_RATE = 16000
whisper_model = whisper.load_model("base")
llm = None
tts = None
current_lang = "English"
model_paths = {
    "English": "tts_models/en/ljspeech/tacotron2-DDC",
    "French": "tts_models/fr/css10/vits",
    "Russian": "tts_models/ru/ruslan/vits"
}
is_recording = False
recording_data = []
stream = None


# ---------- Utility Functions ----------
# Restart Button
def restart_models():
    global current_lang
    current_lang = lang_var.get()
    chat_window.insert(tk.END, f"🔁 Reloading models for {current_lang}...\n")
    threading.Thread(target=lambda: load_ai_models(current_lang)).start()

def remove_emojis(text):
    emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
    return re.sub(emoji_pattern, '', text)

def audio_callback(indata, frames, time, status):
    """Callback function for audio recording."""
    global recording_data
    if status:
        print(f"⚠️ Warning: {status}")
    if is_recording:
        recording_data.append(indata.copy())

def start_recording():
    """Start audio recording."""
    global is_recording, recording_data, stream
    is_recording = True
    recording_data = []
    
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype='float32', 
            callback=audio_callback
        )
        stream.start()
        chat_window.insert(tk.END, "🎤 Recording started... Click 'Stop Recording' to finish.\n")
        chat_window.see(tk.END)
        talk_btn.configure(text="⏹️ Stop Recording")
    except Exception as e:
        print(f"Error starting recording: {e}")
        chat_window.insert(tk.END, f"❌ Error starting recording: {e}\n")
        is_recording = False

def stop_recording():
    """Stop audio recording and return the recorded audio."""
    global is_recording, stream
    is_recording = False
    
    if stream:
        stream.stop()
        stream.close()
        stream = None
    
    if not recording_data:
        chat_window.insert(tk.END, "⚠️ No audio recorded.\n")
        talk_btn.configure(text="🎙️ Talk")
        return None
    
    audio_np = np.concatenate(recording_data, axis=0).squeeze()
    chat_window.insert(tk.END, "🎤 Recording stopped. Processing...\n")
    chat_window.see(tk.END)
    talk_btn.configure(text="🎙️ Talk")
    return audio_np

def play_audio_array(audio_data, sample_rate):
    """Plays audio from a NumPy array with proper formatting."""
    try:
        sd.play(audio_data, sample_rate)
        sd.wait()
    except Exception as e:
        print("🔊 Playback error:", e)


# ---------- AI Logic ----------

def load_ai_models(language="English"):
    global llm, tts
    llm = Llama(model_path="models/gemma-2b-it.Q4_K_M.gguf", verbose=False)
    tts = TTS(model_paths[language], gpu=False)

def toggle_recording():
    """Toggle between start and stop recording, then process if stopped."""
    global is_recording
    
    if not is_recording:
        # Start recording
        talk_btn["state"] = "disabled"
        lang_menu["state"] = "disabled"
        threading.Thread(target=start_recording_thread).start()
    else:
        # Stop recording and process
        threading.Thread(target=stop_and_process).start()

def start_recording_thread():
    """Start recording in a separate thread."""
    start_recording()
    talk_btn["state"] = "normal"

def stop_and_process():
    """Stop recording and process the conversation."""
    talk_btn["state"] = "disabled"
    
    audio_data = stop_recording()
    if audio_data is None:
        talk_btn["state"] = "normal"
        lang_menu["state"] = "readonly"
        return

    # Process the recorded audio
    try:
        transcription = whisper_model.transcribe(audio_data, fp16=False)
        user_text = transcription["text"].strip()
        
        if not user_text:
            chat_window.insert(tk.END, "👤 You: [No speech detected]\n")
            talk_btn["state"] = "normal"
            lang_menu["state"] = "readonly"
            return

        chat_window.insert(tk.END, f"👤 You: {user_text}\n")
        chat_window.see(tk.END)

        # Generate AI response
        output = llm(user_text, max_tokens=50)
        response_text = remove_emojis(output["choices"][0]["text"].strip())
        chat_window.insert(tk.END, f"🤖 AI: {response_text}\n\n")
        chat_window.see(tk.END)

        # Generate and play TTS
        tts_audio = tts.tts(response_text)
        tts_sample_rate = 22050  # Default sample rate for most Coqui TTS models
        play_audio_array(tts_audio, tts_sample_rate)

    except Exception as e:
        chat_window.insert(tk.END, f"❌ Error processing audio: {e}\n")
        print(f"Error in processing: {e}")
    
    finally:
        talk_btn["state"] = "normal"
        lang_menu["state"] = "readonly"


# ---------- UI Setup ----------

root = tk.Tk()
root.title("AI Speech Chat")
root.geometry("600x550")

# Chat Window
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Segoe UI", 11))
chat_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Control Frame
controls = tk.Frame(root)
controls.pack(fill=tk.X, padx=10, pady=5)
controls.columnconfigure((0, 1, 2), weight=1)

# Language Dropdown
lang_var = tk.StringVar(value="English")
lang_menu = ttk.Combobox(controls, textvariable=lang_var, values=list(model_paths.keys()), state="readonly", width=10)
lang_menu.grid(row=0, column=0, padx=5)

# Talk Button (now toggle)
talk_btn = ttk.Button(controls, text="🎙️ Talk", command=toggle_recording)
talk_btn.grid(row=0, column=1, padx=5)

# Restart Button
restart_btn = ttk.Button(controls, text="🔄 Restart AI", command=restart_models)
restart_btn.grid(row=0, column=2, padx=5)

# Initialize Models
load_ai_models()

# Cleanup function for proper shutdown
def on_closing():
    global stream, is_recording
    if stream:
        is_recording = False
        stream.stop()
        stream.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()