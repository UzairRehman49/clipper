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

echo "==> face identity (SFace, OpenCV model zoo, ~37 MB) - same person across camera angles"
[ -f models/face_recognition_sface_2021dec.onnx ] || curl -sSL -o models/face_recognition_sface_2021dec.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

echo "==> active speaker detection (Light-ASD, MIT, ~4 MB weights) - which face is talking"
[ -d models/Light-ASD ] || git clone -q --depth 1 https://github.com/Junhua-Liao/Light-ASD.git models/Light-ASD
.venv-diar/bin/pip install python_speech_features

echo "==> judge training (optional, own venv): LoRA on the local Qwen from Claude's verdicts"
python3.12 -m venv .venv-train
.venv-train/bin/pip install mlx-lm

echo "==> youtube publishing (optional)"
pip install google-api-python-client google-auth-oauthlib

mkdir -p sources work out logs
touch .env && chmod 600 .env          # settings + keys; git-ignored

set_env() {   # set_env KEY VALUE - replace or add one line in .env
  grep -v "^$1=" .env > .env.tmp || true
  echo "$1=$2" >> .env.tmp && mv .env.tmp .env && chmod 600 .env
}

# ---------------------------------------------------------------- memory --
RAM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
if   (( RAM_GB < 20 )); then REC="qwen3-8b";        TIER="16 GB"
elif (( RAM_GB < 30 )); then REC="qwen3-14b";       TIER="24 GB"
elif (( RAM_GB < 44 )); then REC="qwen3-30b-a3b";   TIER="32-36 GB"
elif (( RAM_GB < 60 )); then REC="qwen3-32b";       TIER="48 GB"
elif (( RAM_GB < 120 )); then REC="llama-3.3-70b-instruct"; TIER="64-96 GB"
else                          REC="gpt-oss-120b";   TIER="128 GB+"
fi

cat <<EOF

==> local model for transcription-side jobs (niche, hooks, find) and the
    local judge. This Mac: ${RAM_GB} GB unified memory -> recommended: ${REC}

    Install LM Studio (https://lmstudio.ai), download the MLX 4-bit build,
    start its server on port 1234. Load it with 32k context, 1 parallel
    request - the pipeline does this itself; higher settings ran a 24 GB
    Mac out of memory (kernel panic).

    What more memory buys you (4-bit MLX; ~8 GB stays free for Whisper,
    face tracking and ffmpeg). Newer releases of the same families in
    LM Studio are fine - prefer the latest:

      16 GB      qwen3-8b                    works; weakest picks and hooks
      24 GB      qwen3-14b                   tested default of this pipeline
      32-36 GB   qwen3-30b-a3b, gpt-oss-20b  mixture-of-experts: bigger AND faster
      48 GB      qwen3-32b, gemma-3-27b-it   dense 30B class: clearly better judgement
      64-96 GB   llama-3.3-70b-instruct      70B class: closest local to Claude
      128 GB+    gpt-oss-120b, qwen3-235b-a22b (3-bit)

EOF

# ----------------------------------------------------------------- judge --
CLI=$(command -v claude || true)
cat <<EOF
==> who judges the clips? (keep/drop, where to cut - the step that decides
    what gets posted; everything else always runs locally)

  1) Claude via your Claude Code login    $( [[ -n "$CLI" ]] && echo "[claude CLI found]" || echo "[needs Claude Code installed]" )
     Best picks: measured far stricter and sharper than a local 14B (kept
     3/10 street-interview candidates vs 9/10, with specific reasons). No
     API key; uses your Claude plan allowance (~17 judgments, ~35k tokens
     per video, ~2 min). Needs internet.

  2) Claude via API key
     Same quality as 1, billed per token on your Anthropic account instead
     of plan allowance. Good for unattended/server runs.

  3) Local model only (${REC})
     Free, offline, private. Weaker selection: a local 14B agreed with a
     careful manual ranking at only +0.46. Improves once trained on Claude's
     verdicts (train_judge.py), or with a bigger model if you have the RAM.

  Whichever you pick, every Claude verdict is saved to logs/judge_data.jsonl
  and becomes training data for your local model - the data carries over
  to future model releases.

EOF
if [[ -t 0 ]]; then
  DEFAULT=$([[ -n "$CLI" ]] && echo 1 || echo 3)
  read -r -p "Choose 1, 2 or 3 [${DEFAULT}]: " CH
  CH=${CH:-$DEFAULT}
else
  CH=$([[ -n "$CLI" ]] && echo 1 || echo 3)   # non-interactive: best available
fi
case "$CH" in
  1) set_env CLIPPER_JUDGE claude
     if [[ -z "$CLI" ]]; then
       echo "   install Claude Code (https://claude.com/claude-code) and log in; the local model is used meanwhile"
     fi ;;
  2) set_env CLIPPER_JUDGE claude
     read -r -s -p "   Anthropic API key (input hidden): " KEY; echo
     if [[ -n "$KEY" ]]; then
       set_env ANTHROPIC_API_KEY "$KEY"; echo "   saved to .env (git-ignored, chmod 600)"
     fi ;;
  *) set_env CLIPPER_JUDGE local ;;
esac
echo "   judge: $(grep '^CLIPPER_JUDGE=' .env | cut -d= -f2)   (change any time: edit .env or re-run setup.sh)"

cat <<'EOF'

==> done. Run:

  source .venv/bin/activate
  python3 clipper.py go "https://www.youtube.com/watch?v=..."

EOF
