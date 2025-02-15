from multiprocessing import Process
from llama_cpp import Llama
import whisper
from TTS.api import TTS
import re
import wave
import soundfile as sf
import numpy as np
import keyboard

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

def record_audio(filename, sample_rate): # NOT WORKING AT ALL
    import sounddevice as sd  # Import here to prevent conflicts
    # print("Press ENTER to start recording...")
    # keyboard.wait("enter")  # Wait for key press
    # print("Recording... Press ENTER again to stop.")
    
    audio = []
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16)
    with stream:
        while not keyboard.is_pressed("enter"):  # Stop when ENTER is pressed
            data, _ = stream.read(1024)
            audio.append(data)
    
    # Save audio
    audio = np.concatenate(audio, axis=0)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    print("Recording saved!")

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

    record_audio(FILENAME, SAMPLE_RATE)  # Step 1: Record audio
    transcription = whisper_model.transcribe(FILENAME)  # Step 2: Transcribe
    prompt_text = transcription["text"].strip()
    print("You:", prompt_text)

    # Step 3: AI Response (Suppress print)
    output = llm(prompt_text, max_tokens=50)
    response_text = remove_emojis(output["choices"][0]["text"].strip())
    
    print("AI:", response_text)
    
    # Step 4: Convert to Speech
    tts.tts_to_file(text=response_text, file_path=OUTPUT_AUDIO)
    play_audio(OUTPUT_AUDIO)