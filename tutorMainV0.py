from multiprocessing import Process
from llama_cpp import Llama
import whisper
from TTS.api import TTS
import re
import wave
import soundfile as sf
import numpy as np

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

def record_audio(filename, duration, sample_rate):
    import sounddevice as sd  # Import here to prevent conflicts
    print("Recording... Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()

    # Save as WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    print("Recording saved!")

def play_audio(file_path):
    import sounddevice as sd  # Import here to prevent conflicts
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()


# Step 1: Record Audio
record_audio(FILENAME, DURATION, SAMPLE_RATE)

# Step 2: Transcribe with Whisper
transcription = whisper_model.transcribe(FILENAME)
prompt_text = transcription["text"].strip()
print("Transcribed Text:", prompt_text)

# Step 3: Generate AI Response
output = llm(prompt_text, max_tokens=50)
response_text = output["choices"][0]["text"].strip()

# Extract the generated answer
response_text = output["choices"][0]["text"].strip()
response_text = remove_emojis(response_text) # Remove Emojis
print("AI Response:", response_text)

# Convert text to speech
tts.tts_to_file(text=response_text, file_path=OUTPUT_AUDIO)
play_audio(OUTPUT_AUDIO)