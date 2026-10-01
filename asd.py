"""asd.py - active speaker detection (Light-ASD, CVPR 2023, MIT licence).

Is this face the one talking? Light-ASD reads 25 fps lip crops together with
the audio, so it scores audio-visual sync rather than mouth movement: a
neighbour laughing or a host's beard behind a mic no longer fools it.

Runs in .venv-diar (needs torch); clipper.py prepares the crops and calls:

  .venv-diar/bin/python asd.py in.npz out.npz

in.npz:  audio  int16 16 kHz mono for the clip span
         t<k>   uint8 [frames, 112, 112] grey face crops of track k
         s<k>   first frame of track k (25 fps, from the start of audio)
out.npz: t<k>   float32 [frames] speaking score per frame (> 0 = speaking)
"""
import math
import sys
from pathlib import Path

import numpy as np
import python_speech_features
import torch

REPO = Path(__file__).resolve().parent / "models" / "Light-ASD"
sys.path.insert(0, str(REPO))
from model.Model import ASD_Model  # noqa: E402

# Average over several window lengths (Light-ASD's evaluation). The authors'
# 11-window set took 76 s on a 45 s Whatever clip; this, their shorter set,
# took 30 s and planned the identical framing.
DURATIONS = (1, 2, 4, 6)


def load():
    model, fc = ASD_Model(), torch.nn.Linear(128, 2)
    state = torch.load(REPO / "weight" / "pretrain_AVA_CVPR.model", map_location="cpu")
    model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
    fc.load_state_dict({k[len("lossAV.FC."):]: v for k, v in state.items() if k.startswith("lossAV.FC.")})
    return model.eval(), fc.eval()


def score(model, fc, mfcc, faces):
    n = min(len(faces), (len(mfcc) - len(mfcc) % 4) // 4)
    if n < 25:                                   # under a second: too little to judge
        return np.full(len(faces), np.nan, np.float32)
    mfcc, faces = mfcc[:n * 4], faces[:n]
    runs = []
    with torch.no_grad():
        for d in DURATIONS:
            out = []
            for i in range(math.ceil(n / (d * 25))):
                a = torch.FloatTensor(mfcc[i * d * 100:(i + 1) * d * 100]).unsqueeze(0)
                v = torch.FloatTensor(faces[i * d * 25:(i + 1) * d * 25]).unsqueeze(0)
                x = model.forward_audio_visual_backend(model.forward_audio_frontend(a),
                                                       model.forward_visual_frontend(v))
                out.append(fc(x.reshape(-1, 128))[:, 1].numpy())
            runs.append(np.concatenate(out)[:n])
    s = np.full(len(faces), np.nan, np.float32)
    s[:n] = np.mean(runs, axis=0)
    return s


def main(inp, outp):
    torch.set_num_threads(4)
    d = np.load(inp)
    model, fc = load()
    mfcc = python_speech_features.mfcc(d["audio"], 16000, numcep=13, winlen=0.025, winstep=0.010)
    res = {}
    for k in [k for k in d.files if k.startswith("t")]:
        s0 = int(d["s" + k[1:]])
        res[k] = score(model, fc, mfcc[s0 * 4:], d[k])
    np.savez(outp, **res)


if __name__ == "__main__":
    main(*sys.argv[1:3])
