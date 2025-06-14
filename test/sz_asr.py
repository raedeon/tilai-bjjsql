import base64
import json
import math
import os
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence
import jiwer
from typing import Any
import itertools
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

BATCH_SIZE = 4
MAX_SAMPLES = 20  # <-- limit total test samples here

wer_transforms = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.SubstituteRegexes({"-": " "}),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])


def sample_generator(
        instances: Sequence[Mapping[str, Any]],
        data_dir: Path,
        max_samples: int = None
) -> Iterator[Mapping[str, Any]]:
    count = 0
    for instance in instances:
        if max_samples is not None and count >= max_samples:
            break
        with open(data_dir / instance["audio"], "rb") as audio_file:
            audio_bytes = audio_file.read()
        yield {
            "key": instance["key"],
            "b64": base64.b64encode(audio_bytes).decode("ascii"),
        }
        count += 1


def score_asr(truth: list[str], hypothesis: list[str]) -> float:
    return 1 - jiwer.wer(
        truth,
        hypothesis,
        truth_transform=wer_transforms,
        hypothesis_transform=wer_transforms,
    )


def main():
    data_dir = Path(f"/home/jupyter/{TEAM_TRACK}/asr")
    results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "asr.jsonl") as f:
        instances = [json.loads(line.strip()) for line in f if line.strip()]

    # Limit instances for quick test
    limited_instances = instances[:MAX_SAMPLES]

    batch_generator = itertools.batched(sample_generator(limited_instances, data_dir, max_samples=MAX_SAMPLES), n=BATCH_SIZE)

    results = []
    for batch in tqdm(batch_generator, total=math.ceil(len(limited_instances) / BATCH_SIZE)):
        response = requests.post("http://localhost:5001/asr", data=json.dumps({
            "instances": batch,
        }))
        response.raise_for_status()
        results.extend(response.json()["predictions"])

    results_path = results_dir / "asr_results.json"
    print(f"Saving test results to {str(results_path)}")
    with open(results_path, "w") as results_file:
        json.dump(results, results_file)

    ground_truths = [instance["transcript"] for instance in limited_instances]
    score = score_asr(ground_truths, results)
    print("1 - WER:", score)


if __name__ == "__main__":
    main()
