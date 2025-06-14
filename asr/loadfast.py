import os
import torchaudio

audio_folder = './Batch'
audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')]

for path in audio_files:
    print(f"Checking: {path}")
    try:
        waveform, sr = torchaudio.load(path)
        print(f"✅ Loaded successfully: {path}, Sample rate: {sr}, Shape: {waveform.shape}")
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
