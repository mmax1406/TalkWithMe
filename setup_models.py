import os
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    exit(1)

# We are switching to Llama-3.2-3B-Instruct for significantly better conversational performance
REPO_ID = "bartowski/Llama-3.2-3B-Instruct-GGUF"
FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LOCAL_DIR = "models"

def main():
    print(f"Downloading {FILENAME} from {REPO_ID}...")
    print("This is a ~2GB file, it may take a few minutes depending on your internet connection.")
    
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    downloaded_model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False
    )
    
    print(f"\n✅ Llama Model downloaded successfully to: {downloaded_model_path}")
    
    print("\nDownloading Faster-Whisper 'base' model...")
    try:
        from faster_whisper import download_model
        download_model("base")
        print("✅ Faster-Whisper model downloaded successfully!")
    except ImportError:
        print("⚠️  faster-whisper is not installed. Please run `pip install faster-whisper` first if you want it downloaded now.")
        
    print("\nYou can now start your backend server!")

if __name__ == "__main__":
    main()
