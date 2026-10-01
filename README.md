# Local clipping pipeline — Apple Silicon

Long-form video in, captioned vertical clips out. Everything runs on your Mac.
No subscription, no watermark, no upload limits, no 60-minutes-a-month cap.

---

## What the AI actually does

| Stage | Who | Tool |
|---|---|---|
| Transcribe with word timestamps | AI | MLX Whisper, Apple GPU |
| Find candidate moments (phrase-indexed transcript with speaker names and reaction tags) | AI | local LLM + audio-energy spikes + rapid speaker exchanges |
| Rank by self-containment | AI | local LLM |
| Judge top 14 (+3 reserved for moments with audience signals), plan in/out points + loop payoff, pick the best non-overlapping, non-duplicate clips | AI (`select`) — or you (`review`) | local LLM + embeddings for topic dedup |
| Who is speaking (S1/S2 → names from the title) | AI | pyannote community-1 (own venv, needs a free HF token once) |
| Hooks: verbatim quotes picked from the clip, one per platform; topic-only captions | AI | local LLM picks, code copies the words |
| Chapters, YouTube replay heatmap, SponsorBlock ad segments | auto | yt-dlp metadata + SponsorBlock API |
| Laughter / applause / music per second, tagged into the transcript the LLM reads | AI | PANNs (`events.py`, own venv) |
| Per-shot vertical layout: locked crop on the speaker, facecam-over-game stack, two-person stack, split-screen box zoom, or fit-over-blur for screens | AI | ffmpeg scene cuts + YuNet faces, FaceMesh mouths, speaker-to-face linking |
| Silence + filler-word jump cuts | auto | Whisper word timestamps |
| Cold-open loop (payoff first, ends right before it) | auto | ffmpeg |
| Zoom on jump cuts and vocal emphasis, whoosh/pop SFX | auto | ffmpeg `zoompan`, generated SFX |
| Word-by-word animated captions | AI | Whisper timestamps → ASS |
| Loudness normalise to −14 LUFS | auto | ffmpeg |
| Export 1080×1920 | auto | VideoToolbox hardware encode |
| Publish to YouTube | auto | Data API |
| Publish to TikTok | **you** | see below |

`go` runs everything above end to end with no questions. Your hands-on time
is watching the finished clips before you post them (see below).

---

## Why the review step stays human

Not squeamishness — economics. TikTok detects and demotes exact reposts,
duplicate uploads across accounts, and unedited clips it has seen before,
escalating to account bans. A ban forfeits campaign views that have not
finished verification, so it costs you money you already earned.

YouTube is stricter still: its reused-content policy applies *even when you
have permission from the original creator*, because it is separate from
copyright. What satisfies it is substantive editing that is unique to your
channel.

So the pipeline now makes the picks and writes the hooks itself, but you
still watch every clip before it goes out: that is the check against a wrong
caption, a weak pick or a framing miss. `review` lets you hand-pick instead.
A virality score from any tool - including this one - is a plausible number,
not a measured one; the judge's ranking agreed with a careful manual ranking
at Spearman +0.46 on the one video it was measured on.

Do **not** try to solve this with sandwich frames or subtle frame insertion.
TikTok names both explicitly as evasion techniques that draw harsher
penalties.

---

## Install

```bash
chmod +x setup.sh && ./setup.sh
```

`setup.sh` installs ffmpeg + ffmpeg-full (libass for captions), deno (yt-dlp
needs a JS runtime or YouTube throttles it), the main venv, a separate
`.venv-diar` (pyannote speaker diarization + PANNs audio events - their numpy
2 would break mediapipe), and the YuNet face model and PANNs weights.

Two one-time steps it can't do for you:

1. **Speaker diarization** needs a free Hugging Face read token with the terms
   of `pyannote/speaker-diarization-community-1` accepted:
   `.venv-diar/bin/hf auth login`
2. **A local LLM server** - LM Studio on port 1234 (or Ollama). Measured on a
   24 GB Mac: `qwen/qwen3-14b` at 32k context, one request at a time. The
   pipeline loads the model itself with those settings and reloads it after
   every call, because LM Studio's MLX backend leaks ~2 GB per request; two
   parallel requests or a 65k context ran the Mac out of memory (one kernel
   panic). Point it elsewhere with:

```bash
export CLIPPER_LLM_URL=http://localhost:1234/v1/chat/completions
export CLIPPER_LLM_MODEL=qwen/qwen3-14b
```

---

## Run

```bash
source .venv/bin/activate
python3 clipper.py go "https://www.youtube.com/watch?v=..."
```

Link in, finished clips out: downloads audio (full video downloads in the
background meanwhile), transcribes, detects the niche, finds and judges
candidates, writes hooks, and renders every pick for TikTok, Instagram and
Shorts. Each editing effect can be switched off: `--no-cuts --no-loop
--no-zoom --no-sfx`. `go` also takes a local file.

Or stage by stage:

```bash
python3 clipper.py transcribe sources/ep42.mp4
python3 clipper.py find       sources/ep42.mp4             # --niche auto by default
python3 clipper.py select     sources/ep42.mp4             # or `review` to pick by hand
python3 clipper.py render     sources/ep42.mp4 --platform tiktok,instagram
```

Output lands in `out/ep42/` as `ep42_01_tiktok.mp4` etc., plus a `.txt`
sidecar holding the hook, post caption, why it was picked, the edit summary
(length, cuts, zoom pulses, loop) and a framing summary (layouts used,
talker found %, speaker on screen %, face-edge frames).
`python3 test_clipper.py` runs 28 self-checks (edit timeline, speaker labels,
hooks, signals, framing planner) in a few seconds - run it after any change.

### Measured timings (M-series, 24 GB)

| Source | Stage | Time |
|---|---|---|
| 3 h podcast | transcribe (speaker labels + audio events run alongside) | ~5.5 min |
| | find | ~7 min |
| | select (judge 14-17 candidates) | ~2-3 min |
| | hooks + render 4-6 clips x 3 platforms | ~2-4 min |
| 75 min panel show | whole `go` run with transcript cached | ~9 min |
| any | video download at 1080p | network-bound; ~350 MB per hour |

Unattended. The Mac is kept awake (`caffeinate`) for the run - closing the
lid on battery still sleeps it.

More detail: `PLAN.md` (what was built and why, from a survey of ~20
open-source clippers) and `TEST_REPORT.md` (every test run, measurement and
decision).

---

## Publishing

**YouTube — automated.** Uploads bill to their own quota bucket at 100
calls/day, so volume is not a constraint.

```bash
python3 publish_youtube.py out/ep42/ep42_03.mp4 \
  --title "The first \$10k is the hardest" \
  --schedule 2026-09-16T14:00:00Z
```

**TikTok — manual, and this is not a limitation of the script.** Unaudited
API clients can only post SELF_ONLY (private), capped at 5 users in a 24-hour
window, with accounts required to be private at the time of posting. Lifting
that requires a full app audit with a demo video, privacy policy and domain
verification — an application you would be filing as a software product
serving creators, which you are not.

Any self-hosted scheduler hits the same wall, because it uses *your*
developer credentials. Post by hand, or use TikTok Studio's native scheduler
(10 days ahead, free, unlimited).

**Reframe this as free QC.** The manual TikTok step forces you to look at
every clip before it goes out, which is exactly the check that stops a bad
batch from costing you an account.

---

## Tuning

Per-niche selection criteria live in `CRITERIA` at the top of `clipper.py`.
Rewriting that list for the campaign you are actually working will improve
your hit rate more than swapping any tool in this stack. Finance rewards
specific numbers and contrarian claims; streamer content rewards reaction
beats and escalation.

Other knobs:

```bash
export CLIPPER_FONT="Montserrat ExtraBold"   # caption face
export CLIPPER_ASR=mlx-community/whisper-medium-mlx   # if memory constrained
export CLIPPER_FACES=mediapipe               # old face detector (default: yunet)
export CLIPPER_LLM_PARALLEL=1                # >1 ran a 24 GB Mac out of memory
export CLIPPER_LLM_CYCLE_EVERY=1             # reload the model after N LLM calls
export CLIPPER_COOKIES_BROWSER=chrome        # only if YouTube's bot check blocks downloads
```

Flags: `--center` (skip framing analysis, centre crop), `--no-cuts`,
`--no-loop`, `--no-zoom`, `--no-sfx`, `--max-clips N`, `--niche
auto|podcast|streamer|tutorial|finance|comedy|vlog`, `--platform
tiktok,instagram,shorts`.

---

## The log is the real asset

Every render writes one row per clip and platform to `logs/clips.csv`. Each
row records the choices the pipeline made: origin (llm / exchange / heat),
signals, judge score and final score, loop, main layout, talker coverage,
which hook variant was used, the hook and the caption. Re-rendering updates the
row in place and never touches the columns you fill in.

After posting, fill in `posted_url`, `views_24h`, `views_7d` (and optionally
`likes`). Then run:

    python3 clipper.py stats

It prints median views for each pipeline choice, and the rank correlation
between each score and real views. The three per-platform hooks are a free
A/B test. After ~100 clips, this is the data for retuning the judge and the
signal weights, or for training Laya as a scorer. It is the only part of this
system that compounds.

---

## Known constraints

- **Clip ranking is limited by the 14B model's taste.** Measured: agreement
  with a careful manual ranking +0.46; a comparative-ranking judge did worse
  and was removed. A stronger judge model is the open decision.
- **Who is talking comes from mouth movement sampled 5x/s**, linked to the
  diarized voices. Very short turns and heavy crosstalk can still frame a
  neighbour; an audio-visual active-speaker model (Light-ASD) is the upgrade.
- **Shots with no detectable face** (screen recordings, B-roll, a back of a
  head) show the whole frame over a blurred copy rather than a crop.
- **Hindi-heavy speech transcribes as garbled English** (Whisper forces
  English). Mostly-English Hindi podcasts are fine. The fix (a Hinglish
  Whisper model, romanized captions) was planned and dropped.
- **Hooks are verbatim quotes** chosen by the model - accurate by
  construction, sometimes flat. Written hooks from the 14B model invented
  facts about real people and were removed.
- **Rights:** the pipeline doesn't know whether you're allowed to repost a
  show. Only post clips you have rights to (e.g. a creator's clipping campaign).
- **Single-pass loudnorm.** Fine for speech.
- **Fonts must be installed** for libass to find them. `setup.sh` handles
  Anton and Montserrat; it falls back to Arial Black otherwise.
