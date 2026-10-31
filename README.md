# Clipper — local long-form → short-form pipeline (Apple Silicon)

Turns your own (or a client's) long videos into captioned vertical clips for
TikTok, Instagram and Shorts: link or file in, clips out. Everything runs on
your Mac except the optional Claude judge.

## Install

```bash
chmod +x setup.sh && ./setup.sh
```

Installs ffmpeg (+ffmpeg-full for captions), deno, the main venv,
`.venv-diar` (speaker diarization, audio events, active-speaker detection)
and `.venv-train` (judge training), plus the face and audio models. It asks
**who judges the clips** (Claude via your Claude Code login, Claude via API
key, or local only) and recommends a **local model for your RAM**.

One-time steps it can't do:
1. Hugging Face read token with `pyannote/speaker-diarization-community-1`
   terms accepted: `.venv-diar/bin/hf auth login`
2. LM Studio serving the recommended model on port 1234. The pipeline loads
   it itself at 32k context, one request at a time (more ran a 24 GB Mac out
   of memory).

## Run

```bash
source .venv/bin/activate
python3 clipper.py go "https://www.youtube.com/watch?v=..."   # or a local file
```

Stages can run alone: `transcribe`, `find`, `select` (or `review` to pick
by hand), `render --platform tiktok,instagram`. Output: `out/<id>/<id>_NN_<platform>.mp4`
plus a `.txt` per clip (hook, caption, why, edit and framing summary).
Flags: `--max-clips N`, `--niche auto|podcast|streamer|tutorial|finance|comedy|vlog`,
`--center`, `--no-cuts --no-loop --no-zoom --no-sfx`.
`python3 test_clipper.py` runs the self-checks in seconds; run it after any change.

A 3-hour podcast takes ~20 minutes plus the download.

### Summaries of list and review videos

When the title looks like a top-N list or a review, `go` asks what the
shorts are for, once, at the start:
`clips` (best moments), `summary` (one short running through every item in
the host's own words, with "#5 Name" on screen per item) or `items` (one
short per item or section). Without a terminal, pass `--mode
summary|items|clips`. On an already-transcribed video:
`python3 clipper.py summarize sources/<id>.mp4 --mode items`.
The structure comes from chapters when the video has them, otherwise from
the judge reading the transcript (cached in `work/<id>/structure.json`).

## The judge

`select` decides what to keep and where to cut. Settings live in a
git-ignored `.env`:

- `CLIPPER_JUDGE=claude` uses the API if `ANTHROPIC_API_KEY` is set, else
  `claude -p` on your Claude login (bare: no tools, settings or project
  context). `CLIPPER_JUDGE_MODEL` defaults to `claude-opus-5-5`.
- `CLIPPER_JUDGE=local` uses the LM Studio model. It's also the fallback if
  Claude fails.

Every verdict is logged to `logs/judge_data.jsonl`. With 150+ Claude
verdicts, `.venv-train/bin/python train_judge.py` trains a LoRA on the local
Qwen (~1.5 h for 200 examples); `--eval --adapter <dir>` compares it with
Claude on held-out videos. A new Qwen is retrained from the same log
(`--model <new MLX dir>`).

## Learning from results

Every render adds a row per clip and platform to `logs/clips.csv`. After
posting, fill in `posted_url`, `views_24h`, `views_7d`; `python3 clipper.py stats`
shows median views per pipeline choice and how well each score predicts
views. The per-platform hooks are a free A/B test.

## Posting

- **YouTube:** `python3 publish_youtube.py out/<id>/<clip>.mp4 --title "..." --schedule <ISO time>`.
- **TikTok:** by hand or TikTok Studio's scheduler. Unaudited API apps can
  only post privately.
- Watch every clip first. TikTok demotes reposts and unedited duplicates.
  YouTube's reused-content policy applies even with the creator's permission.
  Never use frame insertion, mirroring or speed/pitch shifts: they're named
  evasion techniques.
- Only post clips you have rights to (e.g. a creator's clipping campaign).

## Tuning

Per-niche criteria: `CRITERIA` in `clipper.py`. Environment knobs:
`CLIPPER_FONT`, `CLIPPER_ASR`, `CLIPPER_FACES=mediapipe`,
`CLIPPER_LLM_URL` / `CLIPPER_LLM_MODEL`,
`CLIPPER_COOKIES_BROWSER=chrome` (only if YouTube's bot check blocks
downloads). Keep `CLIPPER_LLM_PARALLEL=1`.

## Known limits

Faceless people get fit-over-blur. Burned-in tickers and overlays can be
cropped. Small talkers in 1080p wide shots stay small. Hindi-heavy speech
garbles. Hooks are verbatim quotes: accurate, sometimes flat.
