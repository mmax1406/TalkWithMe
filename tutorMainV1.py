from multiprocessing import Process
from llama_cpp import Llama
import whisper
from TTS.api import TTS
import re
import wave
import soundfile as sf
import numpy as np
import keyboard
import sys

# Audio Recording Settings
SAMPLE_RATE = 16000
DURATION = 5
FILENAME = "input.wav"
OUTPUT_AUDIO = "response.wav"

llm = Llama(model_path="gemma-2b-it.Q4_K_M.gguf", verbose=False)

# Load Whisper for speech recognition
whisper_model = whisper.load_model("base")

# Load Coqui TTS (Change model for different languages)
tts_model = "tts_models/en/ljspeech/tacotron2-DDC"
# tts_model = "tts_models/fr/css10/vits"  # For French
# tts_model = "tts_models/ru/ruslan/vits"  # For Russian
tts = TTS(tts_model, gpu=False)

def remove_emojis(text):
    emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
    return re.sub(emoji_pattern, '', text)

def record_audio(filename): # NOT WORKING AT ALL
    import sounddevice as sd  # Import here to prevent conflicts
    """Records audio while the Enter key is held down using sd.stream."""
    recording = []
    recording_started = False

    def callback(indata, frames, time, status):
        """Callback function to process audio chunks in real time."""
        nonlocal recording_started
        if status:
            print(f"⚠️ Warning: {status}")
        if keyboard.is_pressed("enter"):
            recording.append(indata.copy())  # Store recorded chunk
            recording_started = True

    keyboard.wait("enter")
    # Start streaming audio
    with sd.InputStream(samplerate=16000, channels=1, 
                        dtype=np.int16, callback=callback, device=None):
        while keyboard.is_pressed("enter"):
            pass  # Keep the stream open while Enter is held

    if not recording_started:
        print("⚠ No audio recorded! Check your microphone settings.")
        return None  

    # Save as WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(16000)
        wf.writeframes(np.concatenate(recording, axis=0).tobytes())

    # print("Recording saved!")

def play_audio(file_path):
    import sounddevice as sd  # Import here to prevent conflicts
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()

# Main Conversation Loop
print("Chat started! Press ENTER to talk, or ESC to exit.")

while True:
    if keyboard.is_pressed("esc"):
        print("Exiting chat...")
        break

    # Step 1: Record audio
    record_audio(FILENAME)

    # Step 2: Transcribe
    transcription = whisper_model.transcribe(FILENAME)  
    prompt_text = transcription["text"].strip()
    # print(prompt_text)

    # Step 3: AI Response (Suppress print)
    output = llm(prompt_text, max_tokens=30)
    response_text = remove_emojis(output["choices"][0]["text"].strip())
    # print("AI:", response_text)
    
    # Step 4: Convert to Speech
    tts.tts_to_file(text=response_text, file_path=OUTPUT_AUDIO)
    play_audio(OUTPUT_AUDIO)