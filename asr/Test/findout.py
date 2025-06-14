import base64
import json
import os
from pathlib import Path
from typing import Sequence, Mapping, Any, Iterator
import itertools
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

BATCH_SIZE = 2


def sample_generator(
        instances: Sequence[Mapping[str, Any]],
        data_dir: Path,
) -> Iterator[Mapping[str, Any]]:
    for instance in instances:
        with open(data_dir / instance["audio"], "rb") as audio_file:
            audio_bytes = audio_file.read()
        yield {
            "key": instance["key"],
            "b64": base64.b64encode(audio_bytes).decode("ascii"),
        }

def save_audio_from_base64(base64_audio: str, output_dir: Path, file_name: str):
    # Decode the base64 audio and save it as a .wav file
    audio_data = base64.b64decode(base64_audio)
    output_file_path = output_dir / file_name
    with open(output_file_path, "wb") as audio_file:
        audio_file.write(audio_data)
    print(f"Saved audio to {output_file_path}")


def main():
    data_dir = Path(f"/home/jupyter/{TEAM_TRACK}/asr")
    results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load instances from JSONL file
    with open(data_dir / "asr.jsonl") as f:
        instances = [json.loads(line.strip()) for line in f if line.strip()]
    
    instances = instances[:2]  # Only take the first 2 instances
    
    batch_generator = itertools.batched(sample_generator(instances, data_dir), n=BATCH_SIZE)
    
    for batch in tqdm(batch_generator, total=len(instances) // BATCH_SIZE):
        # Save audio samples to result directory
        for idx, prediction in enumerate(batch):
            base64_audio = prediction.get('b64')  # Assuming audio is under 'b64' key
                
            if base64_audio:
                # Save audio with an indexed filename like audio_1.wav
                file_name = f"audio_{idx+1}.wav"
                save_audio_from_base64(base64_audio, results_dir, file_name)


if __name__ == "__main__":
    main()
