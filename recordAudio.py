import sounddevice as sd
import numpy as np
import keyboard
import wave

# Recording parameters
samplerate = 44100  # Sample rate in Hz
channels = 1        # Mono audio
device_id = None  # Change this if necessary

def record_audio():
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
    with sd.InputStream(samplerate=samplerate, channels=channels, 
                        dtype=np.int16, callback=callback, device=device_id):
        while keyboard.is_pressed("enter"):
            pass  # Keep the stream open while Enter is held

    if not recording_started:
        print("⚠ No audio recorded! Check your microphone settings.")
        return None  

    return np.concatenate(recording, axis=0)

def save_audio(filename, data):
    """Saves recorded audio to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM (2 bytes per sample)
        wf.setframerate(samplerate)
        wf.writeframes(data.tobytes())

if __name__ == "__main__":
    while True:
        input("Press Enter to start recording...")

        audio_data = record_audio()
        if audio_data is not None:
            save_audio("recorded_audio.wav", audio_data)
        else:
            print("⚠ No audio detected. Try again.")

        if keyboard.is_pressed("esc"):
            break

