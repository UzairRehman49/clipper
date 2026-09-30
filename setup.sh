#!/usr/bin/env bash
# setup.sh - one-time install on Apple Silicon (M-series) macOS.
set -euo pipefail

echo "==> checking architecture"
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This targets Apple Silicon. On Intel, MLX will not work." >&2
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install from https://brew.sh then re-run." >&2
  exit 1
fi

echo "==> ffmpeg (+ffmpeg-full for burned-in captions), python, deno (yt-dlp needs a JS runtime for YouTube)"
brew install ffmpeg ffmpeg-full python@3.12 deno || true

echo "==> caption fonts (Anton = the standard short-form look)"
brew install --cask font-anton font-montserrat || true

echo "==> python venv"
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

echo "==> MLX Whisper (runs on the Apple GPU, not the CPU)"
pip install mlx-whisper

echo "==> face tracking for speaker-aware reframing"
# mediapipe>=1.0 dropped the mp.solutions API that face_track() uses (its
# Tasks-API replacement hard-crashes on macOS Metal). Pin to the last
# solutions-API release.
pip install "numpy<2" opencv-python "mediapipe==0.10.21"

echo "==> speaker diarization (own venv: pyannote 4 needs numpy>=2, which breaks mediapipe)"
python3.12 -m venv .venv-diar
.venv-diar/bin/pip install --upgrade pip pyannote.audio panns-inference
echo "==> audio event model (laughter / applause / music) - PANNs weights, ~330 MB"
mkdir -p "$HOME/panns_data"
[ -f "$HOME/panns_data/class_labels_indices.csv" ] || curl -sSL -o "$HOME/panns_data/class_labels_indices.csv" \
  "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
[ -f "$HOME/panns_data/Cnn14_DecisionLevelMax.pth" ] || curl -sSL -o "$HOME/panns_data/Cnn14_DecisionLevelMax.pth" \
  "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
echo "    one-time: accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1"
echo "    then log in with a read token:  .venv-diar/bin/hf auth login"

echo "==> face detector (YuNet, OpenCV model zoo, ~230 KB)"
mkdir -p models
[ -f models/face_detection_yunet_2023mar.onnx ] || curl -sSL -o models/face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

echo "==> youtube publishing (optional)"
pip install google-api-python-client google-auth-oauthlib

mkdir -p sources work out logs
cat > logs/clips.csv <<'CSV'
date,source,clip_id,platform,campaign,hook_type,posted_url,views_24h,views_7d,approved,reject_reason
CSV

cat <<'EOF'

==> done.

Next, install a local LLM server (pick one):

  LM Studio   https://lmstudio.ai   - GUI, start the server on port 1234
  Ollama      brew install ollama && ollama serve

Then pull a model. On 24GB+ unified memory:

  qwen3-30b-a3b        mixture-of-experts, fast, strong at this task
  gemma-3-27b-it       good alternative

On 16GB, use a 12-14B instruct model instead and lower CAND_COUNT.

Point the pipeline at it:

  export CLIPPER_LLM_URL=http://localhost:1234/v1/chat/completions
  export CLIPPER_LLM_MODEL=qwen3-30b-a3b

Ollama instead:
  export CLIPPER_LLM_URL=http://localhost:11434/v1/chat/completions

Then:
  source .venv/bin/activate
  python3 clipper.py all sources/your-episode.mp4 --niche podcast

EOF
