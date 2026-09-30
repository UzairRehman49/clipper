"""diarize.py - who spoke when, via pyannote's community-1 pipeline.

Runs in its own venv (.venv-diar): pyannote 4 needs numpy>=2 / newer scipy
and protobuf, which break mediapipe in the main venv. clipper.py calls it as
a subprocess, so it also runs alongside Whisper (CPU/MPS here, Whisper on
the GPU via MLX).

  .venv-diar/bin/python diarize.py work/<stem>/audio.wav work/<stem>/speakers.json

Needs a Hugging Face token with the model's terms accepted (one time):
  .venv-diar/bin/hf auth login
"""
import json
import sys
import wave

import numpy as np
import torch
from pyannote.audio import Pipeline

MODEL = "pyannote/speaker-diarization-community-1"


def main(wav_path, out_path):
    # Decode ourselves instead of via torchcodec: audio.wav is already the
    # 16k mono PCM that transcribe extracted.
    with wave.open(wav_path) as f:
        sr = f.getframerate()
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    audio = {"waveform": torch.from_numpy(pcm.astype(np.float32) / 32768.0)[None],
             "sample_rate": sr}

    pipe = Pipeline.from_pretrained(MODEL)
    if pipe is None:
        sys.exit(f"could not load {MODEL} - accept its terms on huggingface.co "
                 "and log in: .venv-diar/bin/hf auth login")
    if torch.backends.mps.is_available():
        pipe.to(torch.device("mps"))

    out = pipe(audio)
    # exclusive = at most one speaker at any instant, which is what word-level
    # alignment with a single-stream ASR transcript needs.
    ann = getattr(out, "exclusive_speaker_diarization", out)
    turns = [{"s": round(t.start, 3), "e": round(t.end, 3), "spk": spk}
             for t, _, spk in ann.itertracks(yield_label=True)]
    with open(out_path, "w") as f:
        json.dump(turns, f)
    print(f"{len(turns)} turns, {len({t['spk'] for t in turns})} speakers -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
