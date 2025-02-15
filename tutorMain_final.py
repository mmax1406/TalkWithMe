import whisper
from TTS.api import TTS
import torch
import sounddevice as sd
import numpy as np
import wave
import simpleaudio as sa
from llama_cpp import Llama  # Local AI chatbot

# Load Llama model (Replace with your model file)
llm = Llama(
      model_path="gemma-2b-it.Q4_K_M.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)

# Load Whisper for speech recognition
whisper_model = whisper.load_model("base")

# Load Coqui TTS (Change model for different languages)
tts_model = "tts_models/en/ljspeech/tacotron2-DDC"
# tts_model = "tts_models/fr/css10/vits"  # For French
# tts_model = "tts_models/ru/ruslan/vits"  # For Russian
tts = TTS(tts_model).to("cuda" if torch.cuda.is_available() else "cpu")

# Load Llama model (Replace with your model file)
llm = Llama(
      model_path="gemma-2b-it.Q4_K_M.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)

# Function to record user speech
def record_audio(filename="input.wav", duration=5, samplerate=16000):
    print("Listening...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    
    wavefile = wave.open(filename, 'wb')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(samplerate)
    wavefile.writeframes(audio_data.tobytes())
    wavefile.close()
    
    print("Recording saved.")

# Function to transcribe speech using Whisper (forcing English output)
def transcribe_audio(filename="input.wav"):
    result = whisper_model.transcribe(filename, language="en")  # Always transcribe to English
    return result["text"]

# Function to correct grammar and continue the conversation
def get_corrected_response(prompt, conversation_history, target_language="French"):
    system_message = (
        f"You are a language tutor engaging in a conversation. "
        f"If the user's input has mistakes, correct them. "
        f"Then, translate the corrected sentence into {target_language}. "
        f"Continue the conversation naturally after that. "
        f"Format:\n"
        f"Corrected English: [fixed sentence]\n"
        f"{target_language}: [translation]\n"
        f"Continue: [next AI response]"
    )
    
    conversation_history.append(f"User: {prompt}")  # Keep conversation context
    context = "\n".join(conversation_history[-5:])  # Keep last 5 messages
    
    response = llm(f"{system_message}\n{context}\nAI:")
    ai_response = response["choices"][0]["text"].strip()
    
    # Extract corrected English, translated sentence, and AI continuation
    corrected_english = prompt  # Default to original input
    target_text = ""
    ai_continuation = ""

    if "Corrected English:" in ai_response and f"{target_language}:" in ai_response and "Continue:" in ai_response:
        parts = ai_response.split(f"{target_language}:")
        corrected_english = parts[0].replace("Corrected English:", "").strip()
        parts2 = parts[1].split("Continue:")
        target_text = parts2[0].strip()
        ai_continuation = parts2[1].strip()
    else:
        target_text = ai_response.strip()
        ai_continuation = "Can you tell me more?"  # Default fallback

    conversation_history.append(f"AI: {ai_continuation}")  # Store AI's next message
    return corrected_english, target_text, ai_continuation

# Function to generate speech in the target language
def text_to_speech(text, filename="output.wav"):
    tts.tts_to_file(text=text, file_path=filename)
    print("TTS generated:", text)

# Function to play the generated speech
def play_audio(filename="output.wav"):
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()

# Main conversation loop
def main():
    target_language = input("Enter target language (e.g., 'French', 'Russian'): ")
    situation = input("Describe your scenario (e.g., 'I'm in a shop in Paris'): ")
    
    conversation_history = [f"User sets situation: {situation}"]

    print("\nStarting conversation. Speak when prompted.")
    
    while True:
        record_audio()
        user_input = transcribe_audio()
        print("You said:", user_input)

        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Exiting conversation.")
            break

        # Get corrected English, target response, and AI continuation
        corrected_english, target_text, ai_continuation = get_corrected_response(
            user_input, conversation_history, target_language
        )
        
        print(f"Corrected English: {corrected_english}")
        print(f"{target_language}: {target_text}")
        print(f"AI Continues: {ai_continuation}")

        # Convert only the target language text to speech
        text_to_speech(target_text)
        play_audio()

if __name__ == "__main__":
    main()
