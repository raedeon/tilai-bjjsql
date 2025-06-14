import io
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel


class ASRManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        # Use int8 quantization or mixed int8/float16 for better performance
        #self.compute_type = "int8"
        # Load local FasterWhisper model once
        self.model = WhisperModel(
            "./Faster",
            device=self.device,
            compute_type=self.compute_type
        )

    def asr(self, audio_bytes: bytes) -> str:
        try:
            # Read audio bytes into numpy array
            audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes))

            # Optional: resample to 16000 Hz if needed
            if sample_rate != 16000:
                import torchaudio
                import torchaudio.transforms as T

                # Convert to torch tensor
                audio_tensor = torch.from_numpy(audio_np).float()
                if len(audio_tensor.shape) > 1:  # If stereo, take the mean to make it mono
                    audio_tensor = audio_tensor.mean(dim=1)

                resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                audio_tensor = resampler(audio_tensor)
                audio_np = audio_tensor.numpy()

            # Perform transcription (no sample_rate argument!)
            segments, info = self.model.transcribe(audio_np, task="translate")
            transcription = "".join([seg.text for seg in segments])

        except Exception as e:
            transcription = f"Transcription failed: {e}"

        return transcription
