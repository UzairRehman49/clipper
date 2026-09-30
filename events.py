"""events.py - laughter / applause / music, second by second (PANNs SED).

Runs in .venv-diar (PyTorch lives there). A transcript can't hear the room:
"the words are mild but the room exploded" is exactly the clip signal a
text-only LLM misses. clipper.py runs this alongside diarization.

  .venv-diar/bin/python events.py work/<stem>/audio.wav work/<stem>/events.json

Weights: ~/panns_data/Cnn14_DecisionLevelMax.pth (AudioSet, 527 classes).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from panns_inference import SoundEventDetection
from panns_inference.config import labels

SR = 32000                 # PANNs' training rate
CHUNK_S = 60
GROUPS = {
    "laughter": ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh",
                 "Chuckle, chortle"],
    "applause": ["Applause", "Clapping", "Cheering", "Crowd"],
    "music": ["Music"],
}


def main(wav, out_path):
    idx = {n: i for i, n in enumerate(labels)}
    cols = {g: [idx[n] for n in names] for g, names in GROUPS.items()}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sed = SoundEventDetection(
        checkpoint_path=str(Path.home() / "panns_data" / "Cnn14_DecisionLevelMax.pth"),
        device=device)
    pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", wav, "-f", "f32le",
                          "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    per_s = {g: [] for g in GROUPS}
    step = SR * CHUNK_S
    for a in range(0, len(audio), step):
        chunk = audio[a:a + step]
        if len(chunk) < SR:                    # model needs >= 1 s
            chunk = np.pad(chunk, (0, SR - len(chunk)))
        fw = sed.inference(chunk[None, :])[0]  # (frames, 527), 100 frames per second
        secs = int(np.ceil(len(audio[a:a + step]) / SR))
        for g, c in cols.items():
            v = fw[:, c].max(axis=1)
            per_s[g] += [round(float(v[i * 100:(i + 1) * 100].max()), 3)
                         for i in range(secs) if i * 100 < len(v)]
    json.dump({"step": 1.0, **per_s}, open(out_path, "w"))
    print(f"{len(per_s['laughter'])} s analysed on {device} -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
