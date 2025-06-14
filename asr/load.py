import whisper
import shutil
import os
import torch

def download_model():
    # Download the model to cache (default behavior)
    model = whisper.load_model("small")

    # Path where Whisper caches the model
    home_dir = os.path.expanduser("~")
    cache_path = os.path.join(home_dir, ".cache", "whisper", "small.pt")

    # Destination where you want to save the model
    destination_dir = "./Model"
    os.makedirs(destination_dir, exist_ok=True)
    destination_path = os.path.join(destination_dir, "small.pt")

    # Copy the cached model to your desired directory
    shutil.copy(cache_path, destination_path)
    print(f"Model saved to {destination_path}")

if __name__ == "__main__":
    download_model()
