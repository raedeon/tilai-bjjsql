import os
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Path to local FasterWhisper model
local_model_path = './Faster'
model = WhisperModel(local_model_path, device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)

# Get full paths to all .wav files in ./Batch
audio_folder = './Batch'
audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')]

def test_batch_sizes(audio_files, max_batch_size=16):
    for batch_size in range(1, max_batch_size + 1):
        if len(audio_files) < batch_size:
            print(f"⚠️ Not enough audio files to test batch size {batch_size}")
            break

        test_batch = audio_files[:batch_size]
        print(f"\nTesting batch size: {batch_size}")
        try:
            segments, info = batched_model.transcribe(test_batch, batch_size=batch_size)
            print(f"✅ Batch size {batch_size} succeeded. Detected language: {info.language} (p={info.language_probability:.2f})")
            for segment in segments:
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        except Exception as e:
            print(f"❌ Batch size {batch_size} failed: {e}")
            break

# Run the batch size test
test_batch_sizes(audio_files, max_batch_size=16)
