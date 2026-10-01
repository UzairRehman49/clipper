#!/usr/bin/env python3
"""
clipper.py - local long-form -> short-form pipeline for Apple Silicon.

Everything runs on your Mac. No cloud, no subscription, no watermark.

  python3 clipper.py go "https://youtube.com/watch?v=..."   # link in, finished clips out

  go = ingest -> transcribe -> find -> select -> render, no questions asked.
  Each stage also runs alone:

  python3 clipper.py transcribe sources/ep42.mp4     # audio -> word-level transcript
  python3 clipper.py find       sources/ep42.mp4     # -> candidates.json (AI shortlist)
  python3 clipper.py select     sources/ep42.mp4     # AI editor judges + plans the cuts
  python3 clipper.py review     sources/ep42.mp4     # (or tick them by hand instead)
  python3 clipper.py render     sources/ep42.mp4 --platform tiktok,instagram
  python3 clipper.py stats                           # which choices earned views (logs/clips.csv)
"""

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- config ----

ROOT = Path(__file__).resolve().parent
# Homebrew's plain `ffmpeg` has no libass, so burned-in captions fail with a
# cryptic filtergraph error - prefer the keg-only ffmpeg-full when present.
# yt-dlp / hf live in this interpreter's venv bin; make them reachable even
# when the venv isn't activated (e.g. `.venv/bin/python clipper.py` from cron)
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
if os.path.isdir("/opt/homebrew/opt/ffmpeg-full/bin"):
    os.environ["PATH"] = "/opt/homebrew/opt/ffmpeg-full/bin" + os.pathsep + os.environ["PATH"]
WORK = ROOT / "work"
OUT = ROOT / "out"
SOURCES = ROOT / "sources"

# ASR. whisper-large-v3-turbo is the accuracy/speed sweet spot on Apple Silicon.
# Drop to "mlx-community/whisper-medium-mlx" if you are memory constrained.
ASR_MODEL = os.environ.get("CLIPPER_ASR", "mlx-community/whisper-large-v3-turbo")

# LLM. Any OpenAI-compatible local server: LM Studio (1234) or Ollama (11434).
LLM_URL = os.environ.get("CLIPPER_LLM_URL", "http://localhost:1234/v1/chat/completions")
LLM_MODEL = os.environ.get("CLIPPER_LLM_MODEL", "qwen3-30b-a3b")
LLM_KEY = os.environ.get("CLIPPER_LLM_KEY", "not-needed")
# Context window + concurrent-request slots to request when (re)loading the
# model via `lms load`. Measured on this model/machine: a 6000-word chunk
# (~52k tokens incl. the "[123.4]word" timestamp overhead, ~8.75 tok/word)
# completed fine at 65536 context, well past the model's advertised
# maxContextLength - trust the measurement over the advertised number, but
# don't push further without re-testing.
LLM_CONTEXT_LENGTH = int(os.environ.get("CLIPPER_LLM_CONTEXT", "32768"))
# 24GB Mac: parallel>1 drove free RAM to 8% and 4x65k kernel-panicked it.
LLM_PARALLEL = int(os.environ.get("CLIPPER_LLM_PARALLEL", "1"))
# Proactively cycle the model every N calls: the MLX leak (see
# restart_llm_backend) grows LM Studio ~2GB per call (13.5GB -> 17GB), and on 24GB
# the machine can panic before a request ever fails.
LLM_CYCLE_EVERY = int(os.environ.get("CLIPPER_LLM_CYCLE_EVERY", "1"))

# Embeddings for semantic dedup (Section 2 of the v2 plan) - LM Studio serves
# an OpenAI-compatible /v1/embeddings endpoint alongside /v1/chat/completions,
# so this reuses the same local server rather than adding a new dependency.
EMBED_URL = os.environ.get("CLIPPER_EMBED_URL",
                            LLM_URL.rsplit("/v1/", 1)[0] + "/v1/embeddings")
EMBED_MODEL = os.environ.get("CLIPPER_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
SEMANTIC_DEDUP = os.environ.get("CLIPPER_SEMANTIC_DEDUP", "1") != "0"
SEMANTIC_DEDUP_THRESHOLD = float(os.environ.get("CLIPPER_SEMANTIC_DEDUP_THRESHOLD", "0.86"))

# Output format
W, H = 1080, 1920
OUT_W, OUT_H = W, H     # (plan_layout's W/H are the SOURCE size)
MAX_ZOOM = 1.9          # upscale limit for crops; 2.8 for one split-screen box
SAME_FACE = 0.363       # SFace cosine: OpenCV's published same-person threshold
ASD_MARGIN = 1.0        # Light-ASD logits: talker must lead the next face by this much
CLIP_MIN, CLIP_MAX = 22, 58          # seconds
CAND_COUNT = 30                      # how many candidates to surface per source
FIND_VERSION = "2"                   # bump when find's inputs/signals change -> recompute
EXCHANGE_MAX = 5                     # rapid speaker-change windows added per source
HEAT_MAX = 3                         # most-replayed regions added when nothing covers them

# Caption look. Install Anton or Montserrat ExtraBold for the standard
# short-form look; falls back to Arial Black which ships with macOS.
FONT = os.environ.get("CLIPPER_FONT", "Anton")
FONT_FALLBACK = "Arial Black"
FONT_SIZE = 96
BASE_COLOR = "&H00FFFFFF"            # white  (ASS is &HAABBGGRR)
HILITE_COLOR = "&H0000E5FF"          # amber
OUTLINE = 6
CAPTION_Y = 0.70                     # fraction of frame height
WORDS_PER_GROUP = 3

# Encoder. h264_videotoolbox = Apple hardware encode, roughly realtime x20.
VCODEC = ["-c:v", "h264_videotoolbox", "-b:v", "10M", "-profile:v", "high"]
SW_VCODEC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-profile:v", "high"]

# Niche-specific selection criteria. Edit these per campaign - this single
# list moves your hit rate more than swapping tools ever will.
CRITERIA = {
    "podcast": [
        "a claim that contradicts what the audience already believes",
        "a specific number, dollar figure, or dated fact",
        "a personal story containing a reversal",
        "a direct disagreement between two speakers",
    ],
    "streamer": [
        "a genuine reaction: shock, outrage, or collapse into laughter",
        "an escalation where the stakes rise across 20 seconds",
        "a moment the chat would clip themselves",
    ],
    "tutorial": [
        "a single named tool or step paired with the concrete problem it solves",
        "a specific number: a price, a file size, a time saved, or a count",
        "a comparison naming a known paid or mainstream product it replaces or resembles",
        "a demonstrable action or visual result described in one sentence",
    ],
    "finance": [
        "a contrarian market or money claim stated plainly",
        "a concrete number with the reasoning behind it",
        "a named mistake the audience is probably making",
    ],
    "comedy": [
        "a joke whose setup and punchline are both inside the segment",
        "the audience laughing or applauding right after a line",
        "an absurd, specific image or scenario described in one breath",
        "a callback or twist that recasts the setup",
    ],
    "vlog": [
        "something the viewer can see happening, not just hear about",
        "a surprising number or fact said aloud or shown on screen",
        "a person's unscripted reaction to what is happening around them",
        "a small story with a beginning and an outcome inside a minute",
    ],
}


NICHE_HINT = {
    "podcast": "a conversation or interview between two or more people - ideas, stories, debate",
    "streamer": "one person reacting live - gaming, chat, big emotional reactions",
    "tutorial": "teaching how to do something - tools, software, step-by-step",
    "finance": "money, markets, investing, business numbers",
    "comedy": "stand-up or sketch comedy - one performer or a lineup, jokes, audience laughter",
    "vlog": "a creator filming their day, a trip or an event - visuals, commentary, music",
}


def log(msg):
    print(f"  {msg}", flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def workdir(source):
    d = WORK / Path(source).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


# ---------------------------------------------------------------- ingest ----

def cmd_ingest(args):
    """Pull a source video from its URL. Capped at 1080p by default: the
    pipeline's final output is a 1080x1920 vertical crop, so a 4K/AV1 pull
    (3.6GB+, tens of minutes on a long podcast) buys nothing but download
    time - cap the source instead of the output."""
    SOURCES.mkdir(parents=True, exist_ok=True)
    if args.audio_only:
        fmt = ["-f", "ba/b"]
    else:
        # res = the SHORT side, so this caps landscape and vertical alike
        # (height<=1080 excluded every 1080x1920 format -> 360p); +size then
        # prefers the smallest stream at that resolution (352 MB AV1 instead
        # of a 970 MB "premium" HLS stream for a 72-min video).
        fmt = ["-f", "bv*+ba/b", "-S", f"res:{args.max_height},+size",
               "--merge-output-format", "mp4"]
    ytdlp([*fmt, "-o", str(SOURCES / "%(id)s.%(ext)s"), args.url])

    # transcribe/find only ever touch the audio-only pull, so the much
    # bigger full-video download doesn't need to happen after them - start
    # it now, in the background, so it overlaps with the (much longer)
    # LLM-chunking stage instead of sitting as a serial wait right before
    # render.
    if args.audio_only and args.prefetch:
        vid = subprocess.run(["yt-dlp", "--get-id", args.url],
                              capture_output=True, text=True, check=True).stdout.strip()
        start_prefetch(args.url, vid, args.max_height)


def start_prefetch(url, vid, max_height=1080):
    pidfile = SOURCES / f".{vid}.prefetch.pid"
    logpath = SOURCES / f".{vid}.prefetch.log"
    if pidfile.exists() and _pid_running(int(pidfile.read_text().strip() or 0)):
        log(f"full-video download already running (pid file {pidfile})")
        return
    log(f"prefetching full video ({vid}) in the background for later render "
        f"(log: {logpath})")
    logf = open(logpath, "w")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "ingest", url,
         "--max-height", str(max_height)],
        stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    pidfile.write_text(str(proc.pid))


def _pid_running(pid):
    """os.kill(pid, 0) alone says a finished-but-unreaped child is still
    alive (a zombie) - reap it first when it's ours, as it is under `go`."""
    if pid <= 0:
        return False
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def ytdlp(args):
    """yt-dlp with a clear way out of YouTube's bot check: set
    CLIPPER_COOKIES_BROWSER=chrome (or safari/firefox) to reuse a signed-in
    browser session. Off by default - cookies grant account access."""
    browser = os.environ.get("CLIPPER_COOKIES_BROWSER")
    cmd = ["yt-dlp", *(["--cookies-from-browser", browser] if browser else []), *args]
    r = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        err = r.stderr.strip()
        if "confirm you" in err.lower() and not browser:
            sys.exit("YouTube bot check blocked the download. Sign in to YouTube in a "
                     "browser, close it, then rerun with e.g. CLIPPER_COOKIES_BROWSER=chrome")
        sys.exit(f"yt-dlp failed:\n{err[-800:]}")


def wait_for_prefetch(src):
    """If `ingest --audio-only --prefetch` kicked off a background full-video
    download for this source, block here (only - not earlier) until it's
    done. Called at the top of render, which is the first step that
    actually needs the full video."""
    pidfile = SOURCES / f".{src.stem}.prefetch.pid"
    if not pidfile.exists():
        return
    pid = int(pidfile.read_text().strip())
    told = False
    while _pid_running(pid):
        if not told:
            log(f"waiting for background video download (pid {pid}) to finish...")
            told = True
        time.sleep(5)
    pidfile.unlink(missing_ok=True)


# ------------------------------------------------------------ transcribe ----

DIAR_PY = ROOT / ".venv-diar" / "bin" / "python"


def start_diarize(wav, wd):
    """Speaker diarization in the background (CPU/MPS, own venv - see
    diarize.py) so it overlaps Whisper instead of adding to it."""
    if (wd / "speakers.json").exists():
        return None
    if not DIAR_PY.exists():
        log("speaker labels skipped: .venv-diar missing (see setup.sh)")
        return None
    log("identifying speakers in the background (pyannote)")
    return subprocess.Popen(
        [str(DIAR_PY), str(ROOT / "diarize.py"), str(wav), str(wd / "speakers.json")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def label_speakers(words, turns):
    """Stamp each word with the diarization turn it overlaps most (or the
    nearest one), renamed S1, S2... in order of first appearance."""
    turns = sorted(turns, key=lambda t: t["s"])
    if not turns:
        return
    names, j = {}, 0
    for w in words:
        while j + 1 < len(turns) and turns[j + 1]["s"] <= w["s"]:
            j += 1
        # overlap, or minus the distance when there is none
        best = max(turns[max(0, j - 1):j + 2],
                   key=lambda t: min(t["e"], w["e"]) - max(t["s"], w["s"]))
        w["spk"] = names.setdefault(best["spk"], f"S{len(names) + 1}")

    # 1-2 word flickers inside one speaker's sentence are noise; a real
    # interjection ("Oh yeah!") is kept because it ends a sentence.
    for i in range(1, len(words) - 1):
        for n in (1, 2):
            if i + n < len(words) and words[i - 1]["spk"] == words[i + n]["spk"] \
                    and all(w["spk"] != words[i - 1]["spk"] for w in words[i:i + n]) \
                    and not re.search(r"[.?!]$", words[i + n - 1]["w"]):
                for w in words[i:i + n]:
                    w["spk"] = words[i - 1]["spk"]

    # Diarization turns and Whisper word times jitter against each other by
    # a word or two (no constant offset - measured), so a change often lands
    # mid-sentence ("two brain | surgeries."). Snap each one to the nearest
    # sentence end or pause within SNAP words, never across another change.
    SNAP = 6
    edge = lambda i: 0 < i < len(words) and (
        re.search(r"[.?!]$", words[i - 1]["w"]) or words[i]["s"] - words[i - 1]["e"] > 0.3)
    i = 1
    while i < len(words):
        old, new = words[i - 1]["spk"], words[i]["spk"]
        if new != old and not edge(i):
            for k in range(1, SNAP + 1):
                fwd, back = i + k, i - k
                if edge(fwd) and all(w["spk"] == new for w in words[i:fwd]):
                    for w in words[i:fwd]:
                        w["spk"] = old
                    i = fwd
                    break
                if edge(back) and all(w["spk"] == old for w in words[back:i]):
                    for w in words[back:i]:
                        w["spk"] = new
                    break
        i += 1


PANNS_WEIGHTS = Path.home() / "panns_data" / "Cnn14_DecisionLevelMax.pth"


def start_events(wav, wd):
    """Laughter/applause/music per second (events.py, own venv) - in the
    background, like diarization; ~2 min per 3 h of audio on the GPU."""
    if (wd / "events.json").exists():
        return None
    if not DIAR_PY.exists() or not PANNS_WEIGHTS.exists():
        log("audio events skipped: .venv-diar or PANNs weights missing (see setup.sh)")
        return None
    return subprocess.Popen(
        [str(DIAR_PY), str(ROOT / "events.py"), str(wav), str(wd / "events.json")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def finish_events(proc):
    if not proc:
        return
    out, _ = proc.communicate()
    if proc.returncode != 0:
        log("audio events skipped - detection failed:\n    "
            + "\n    ".join(out.strip().splitlines()[-4:]))
    else:
        log("audio events: " + out.strip().splitlines()[-1][:80])


def reactions(wd):
    """[(start_s, "laughter"|"applause"), ...] - bursts well above THIS
    video's normal level. Relative, not absolute: comedy-club laughter
    overlapping the comedian peaks near 0.34, so a fixed 0.5 found none;
    the top seconds sat right after the punchline."""
    p = wd / "events.json"
    if not p.exists():
        return []
    ev = json.loads(p.read_text())
    out = []
    for kind, floor, pct in (("laughter", 0.12, 0.97), ("applause", 0.30, 0.98)):
        v = ev.get(kind) or []
        if not v:
            continue
        thr = max(floor, sorted(v)[int(len(v) * pct)])
        i = 0
        while i < len(v):
            if v[i] >= thr:
                j = i
                while j + 1 < len(v) and v[j + 1] >= thr * 0.8:
                    j += 1
                out.append((float(i), kind))
                i = j + 2
            else:
                i += 1
    return sorted(out)


def tags_after(t_end, reacts, used, lo=-1.0, hi=2.5):
    """Reaction tags for a line ending at t_end (each reaction tagged once)."""
    import bisect
    k = bisect.bisect_left(reacts, (t_end + lo, ""))
    got = []
    while k < len(reacts) and reacts[k][0] <= t_end + hi:
        if k not in used:
            used.add(k)
            got.append(f"({reacts[k][1]})")
        k += 1
    return got


def finish_diarize(proc, wd, tpath):
    if proc:
        out, _ = proc.communicate()
        if proc.returncode != 0:
            log("speaker labels skipped - diarization failed:\n    "
                + "\n    ".join(out.strip().splitlines()[-6:]))
            return
    spath = wd / "speakers.json"
    if not spath.exists():
        return
    data = json.loads(tpath.read_text())
    label_speakers(data["words"], json.loads(spath.read_text()))
    tpath.write_text(json.dumps(data, indent=1))
    n = len({w.get("spk") for w in data["words"]})
    log(f"speaker labels: {n} speaker(s) across {len(data['words'])} words")


ASR_VERSION = 3          # bump when transcription settings change -> redo
WHISPER_PROMPT = ("Welcome back to the show. So, what do you think? "
                  "Honestly, I don't know! Let's talk about it.")


def whisper_words(result, offset=0.0):
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            tok = w.get("word", "").strip()
            if tok:
                words.append({"w": tok, "s": round(w["start"] + offset, 3),
                              "e": round(w["end"] + offset, 3)})
    return words


def loop_spans(words, win=30, max_distinct=4):
    """[start, end] of stretches where 30 words use <= 4 distinct words -
    Whisper stuck repeating itself, not speech."""
    spans, i = [], 0
    while i + win <= len(words):
        seg = words[i:i + win]
        if len({re.sub(r"\W", "", w["w"].lower()) for w in seg}) <= max_distinct:
            if spans and seg[0]["s"] <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], seg[-1]["e"])
            else:
                spans.append([seg[0]["s"], seg[-1]["e"]])
            # step half a window, but never past the last full window, or
            # a loop running to the end of the file loses its tail
            i = i + win // 2 if i + win // 2 + win <= len(words) or i == len(words) - win \
                else len(words) - win
        else:
            i += 1
    return spans


def unpunctuated_spans(words, min_words=150):
    """[start, end] of runs of min_words+ words with no sentence end."""
    spans, first, n = [], 0, 0
    for i, w in enumerate(words):
        if re.search(r"[.?!]$", w["w"]):
            if n >= min_words:
                spans.append([words[first]["s"], w["e"]])
            first, n = i + 1, 0
        else:
            n += 1
    if n >= min_words:
        spans.append([words[first]["s"], words[-1]["e"]])
    return spans


def pause_pieces(words, a, b, target=300.0, slack=30.0):
    """Split [a, b] into ~target-second pieces, each cut at the widest
    pause between words within +-slack of the mark, so no word straddles
    two re-transcribed pieces."""
    ws = [w for w in words if a <= w["s"] < b]
    cuts, mark = [a], a + target
    while mark < b - slack:
        near = [(ws[i + 1]["s"] - ws[i]["e"], (ws[i]["e"] + ws[i + 1]["s"]) / 2)
                for i in range(len(ws) - 1) if abs(ws[i]["e"] - mark) <= slack]
        cut = max(near)[1] if near else mark
        cuts.append(cut)
        mark = cut + target
    cuts.append(b)
    return list(zip(cuts, cuts[1:]))


def cmd_transcribe(args):
    src = Path(args.source)
    wd = workdir(src)
    tpath = wd / "transcript.json"
    wav = wd / "audio.wav"
    if not wav.exists():
        log("extracting audio")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])

    events = start_events(wav, wd)
    try:
        _transcribe(args, src, wd, wav, tpath)
    finally:
        finish_events(events)


def _transcribe(args, src, wd, wav, tpath):
    have = tpath.exists() and not args.force
    old_v = json.loads(tpath.read_text()).get("asr_v", 1) if have else ASR_VERSION
    if old_v != ASR_VERSION:
        # older settings left 800-word unpunctuated runs that break hooks
        # and snapping - redo once, keeping each old file under its version
        keep = wd / f"transcript.v{old_v}.json"
        tpath.rename(keep)
        log(f"transcript predates the current ASR settings - re-transcribing (old kept as {keep.name})")
        have = False
    if have and "spk" in (json.loads(tpath.read_text())["words"] or [{"spk": 1}])[0]:
        log(f"transcript exists, skipping ({tpath})")
        return
    diar = start_diarize(wav, wd)
    if have:
        log(f"transcript exists, adding speaker labels only ({tpath})")
        finish_diarize(diar, wd, tpath)
        return

    log(f"transcribing with {ASR_MODEL} (MLX, Apple GPU)")
    try:
        import mlx_whisper
    except ImportError:
        sys.exit("mlx-whisper not installed. Run ./setup.sh")

    # Measured on the worst 5-min excerpt (see TEST_REPORT #18): carrying
    # the previous text plus a punctuated prompt cut the longest
    # unpunctuated run from 216 to 34 words with full coverage.
    # hallucination_silence_threshold is deliberately absent: it silently
    # dropped 28.6s of real speech in the same test.
    result = mlx_whisper.transcribe(
        str(wav), path_or_hf_repo=ASR_MODEL, word_timestamps=True,
        condition_on_previous_text=True, initial_prompt=WHISPER_PROMPT)
    words = whisper_words(result)

    def redo(a, b, **kw):
        nonlocal words
        tmp = wd / "asrfix.wav"
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}",
             "-i", str(wav), str(tmp)])
        new = whisper_words(mlx_whisper.transcribe(
            str(tmp), path_or_hf_repo=ASR_MODEL, word_timestamps=True, **kw), offset=a)
        tmp.unlink(missing_ok=True)
        words = [w for w in words if w["s"] < a] + new + [w for w in words if w["s"] >= b]

    # Carrying text is the setting known to trap Whisper in a repetition
    # loop; redo any looping stretch with the old independent-window setting.
    for s0, e0 in loop_spans(words):
        log(f"  repetition loop at {s0:.0f}-{e0:.0f}s - re-transcribing that stretch")
        redo(max(0.0, s0 - 3), e0 + 3, condition_on_previous_text=False)

    # Its other failure: once a window comes out unpunctuated, carried text
    # keeps it that way (measured: the last 31 min of a 3h podcast, a
    # 3337-word run). Redo such runs in ~5-min pieces cut at pauses, each
    # re-seeded with the punctuated prompt so the style resets.
    for a, b in reversed(unpunctuated_spans(words)):
        log(f"  unpunctuated stretch {a:.0f}-{b:.0f}s - re-transcribing in pieces")
        for p0, p1 in reversed(pause_pieces(words, a, b)):
            redo(p0, p1, condition_on_previous_text=True, initial_prompt=WHISPER_PROMPT)

    tpath.write_text(json.dumps({"words": words, "text": result.get("text", ""),
                                 "asr_v": ASR_VERSION}, indent=1))
    log(f"{len(words)} words -> {tpath}")
    finish_diarize(diar, wd, tpath)


# --------------------------------------------------------- find candidates --

def audio_energy(wav, win=0.5):
    """RMS per window. For streamer content this finds reactions the
    transcript cannot see - a scream has no words."""
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav), "-f", "s16le",
         "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n = int(16000 * win)
    if len(a) < n:
        return []
    a = a[: len(a) // n * n].reshape(-1, n)
    rms = np.sqrt((a ** 2).mean(axis=1) + 1e-9)
    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) + 1e-9
    z = (rms - med) / mad
    return [{"t": round(i * win, 2), "z": round(float(v), 2)}
            for i, v in enumerate(z) if v > 4.0]


def embed_texts(texts):
    body = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(
        EMBED_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {LLM_KEY}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return [d["embedding"] for d in data["data"]]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def semantic_dedup(cands, threshold):
    """Time-overlap dedup (in cmd_find) only catches the same MOMENT scored
    twice by adjacent/overlapping chunks. A long conversation often retells
    the same TOPIC at a completely different timestamp (e.g. a story recapped
    20 minutes later) - those don't overlap in time but are redundant to
    post. Embed each candidate's hook+reason text via the local embedding
    model and greedily drop anything too similar to an already-kept,
    higher-scored candidate."""
    if len(cands) < 2:
        return cands
    texts = [f"{c.get('hook', '')} {c.get('why', '')}".strip() or "." for c in cands]
    try:
        vecs = embed_texts(texts)
    except Exception as e:
        log(f"  semantic dedup skipped (embedding call failed: {e})")
        return cands
    order = sorted(range(len(cands)), key=lambda i: -float(cands[i]["score"]))
    kept, kept_vecs = [], []
    for i in order:
        if any(_cosine(vecs[i], kv) >= threshold for kv in kept_vecs):
            continue
        kept.append(i)
        kept_vecs.append(vecs[i])
    kept = set(kept)
    return [c for i, c in enumerate(cands) if i in kept]


def llm(prompt, system):
    # /no_think: Qwen3 models otherwise burn max_tokens on chain-of-thought
    # reasoning before ever emitting the JSON answer, returning empty content.
    # LM Studio actually serves whatever's loaded regardless of this field,
    # but sending the real identifier keeps the request self-documenting.
    body = json.dumps({
        "model": current_llm_identifier(),
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": "/no_think " + prompt}],
        "temperature": 0.4,
        "max_tokens": 2000,
    }).encode()
    req = urllib.request.Request(
        LLM_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {LLM_KEY}"})
    global _lms_ensured, _lms_calls
    if not _lms_ensured:
        # Whatever LM Studio has loaded may carry settings that crash this
        # machine (e.g. 4x65k parallel) - load it with ours before call 1.
        _lms_ensured = True
        restart_llm_backend(_lms_generation)
    gen = _lms_generation
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)
    except (urllib.error.URLError, TimeoutError) as e:
        detail = e.read().decode(errors="replace")[:300] if isinstance(e, urllib.error.HTTPError) else ""
        raise RuntimeError(
            f"Cannot reach LLM at {LLM_URL}\n  {e} {detail}\n"
            "  Is the LM Studio server started and a model loaded?") from e
    with _lms_model_lock:
        _lms_calls += 1
        due = _lms_calls % LLM_CYCLE_EVERY == 0
    if due:
        restart_llm_backend(gen)
    return data["choices"][0]["message"]["content"]


# LM Studio's bundled MLX backend leaks GPU memory across generate() calls
# (its own BatchGenerator cleanup throws on OOM instead of releasing buffers),
# so long `find` runs reliably crash with a METAL "Insufficient Memory" error
# after ~7 requests, no matter how much memory was free at the start - a
# known, still-open upstream bug, not something this pipeline can fix
# outright. `lms unload`+`lms load` cycles the loaded model in a few seconds
# (measured: ~5-10s) instead of quitting/relaunching the whole GUI app, and
# re-applies our context-length/parallel settings every time so a restart
# can't silently drop back to LM Studio's defaults.
_lms_model_lock = threading.Lock()
_lms_model_identifier = None  # discovered lazily from `lms ps`, cached
_lms_generation = 0  # bumped per reload so racing workers don't reload twice
_lms_ensured = False  # loaded with our settings yet, this process?
_lms_calls = 0


def lms(*args, timeout=60):
    try:
        return subprocess.run(["lms", *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise RuntimeError(
            "`lms` CLI not found - install it from LM Studio (Settings > "
            "Developer > Install CLI) or restart the model manually.") from e


def current_llm_identifier():
    """Whatever chat model is presently loaded in LM Studio, discovered via
    `lms ps` rather than assumed - LLM_MODEL is only a fallback for the first
    load, since the actual downloaded model varies per machine."""
    global _lms_model_identifier
    if _lms_model_identifier:
        return _lms_model_identifier
    r = lms("ps", "--json")
    try:
        for m in json.loads(r.stdout or "[]"):
            if m.get("type") == "llm":
                _lms_model_identifier = m["modelKey"]
                return _lms_model_identifier
    except json.JSONDecodeError:
        pass
    return LLM_MODEL


def wait_for_llm(timeout=180):
    models_url = LLM_URL.rsplit("/v1/", 1)[0] + "/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=5) as r:
                if r.status == 200:
                    return
        except urllib.error.URLError:
            pass
        time.sleep(2)
    raise RuntimeError(f"LLM server did not come back within {timeout}s after restart")


def restart_llm_backend(seen_gen):
    """Serialize restarts across worker threads - concurrent chunk requests
    can all fail at once when the leak hits, and only one of them should
    actually cycle the model. A worker whose request started before someone
    else's reload just retries: reloading again would kill the other
    worker's in-flight request and ping-pong forever."""
    global _lms_generation
    with _lms_model_lock:
        if _lms_generation != seen_gen:
            return
        model = current_llm_identifier()
        log(f"cycling {model} to clear leaked GPU memory "
            f"(context={LLM_CONTEXT_LENGTH}, parallel={LLM_PARALLEL})...")
        lms("server", "start")
        lms("unload", "-a")
        r = lms("load", model, "-c", str(LLM_CONTEXT_LENGTH),
                 "--parallel", str(LLM_PARALLEL), "-y", timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"lms load failed: {r.stderr.strip()}")
        wait_for_llm()
        _lms_generation += 1


def parse_json_block(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def phrases(words, max_words=25, pause=0.3):
    """Transcript -> list of (first, last) word indices. A phrase ends at a
    sentence end, a pause, a speaker change, or max_words - so fast
    back-and-forth, which Whisper often leaves unpunctuated for 100+ words,
    still splits at every turn. Indexing find's prompt by phrase instead of
    by word cuts prompt tokens ~4x (measured 13.6k -> 3.3k per 2000 words)."""
    out, a = [], None
    for i, w in enumerate(words):
        if a is not None and (w.get("spk") != words[i - 1].get("spk")
                              or w["s"] - words[i - 1]["e"] > pause
                              or i - a >= max_words):
            out.append((a, i - 1))
            a = None
        if a is None:
            a = i
        if re.search(r"[.?!]$", w["w"]):
            out.append((a, i))
            a = None
    if a is not None:
        out.append((a, len(words) - 1))
    return out


def chunk_phrases(ph, per=5000, overlap=300):
    """~per words of phrases per LLM call, overlapping by ~overlap words.
    5000 words is ~9k prompt tokens in phrase form - well inside the 32k
    context, and 3x the old 2000-word chunks, so ~3x fewer calls."""
    i = 0
    while i < len(ph):
        j, n = i, 0
        while j < len(ph) and n < per:
            n += ph[j][1] - ph[j][0] + 1
            j += 1
        yield ph[i:j]
        if j >= len(ph):
            return
        k, m = j, 0
        while k > i + 1 and m < overlap:
            k -= 1
            m += ph[k][1] - ph[k][0] + 1
        i = k


PLACEHOLDER = re.compile(r"\b(unknown|unidentified|unnamed|guest|host|speaker|person|"
                         r"audience|narrator|caller|voice|chat|man|woman)\b", re.I)


def meta_text(wd):
    """Title + channel + description, lowercased: the only places a name
    can be confirmed from."""
    m = wd / "meta.json"
    if not m.exists():
        return ""
    meta = json.loads(m.read_text())
    return " ".join(str(meta.get(k, "")) for k in ("title", "channel", "description")).lower()


def verify_names(guesses, labels, haystack, share=None):
    """Keep an LLM's name guess only if every word of it appears in the
    title/channel/description, and it isn't a placeholder. Measured
    inventions it removes: "Rufus Wainwright" and "Dane Cook" (not in the
    title), "Chad" (misheard "chat"), "Unidentified Guest", "Ranveer
    Singh" for Ranveer Allahbadia. A name claimed by two labels goes to
    the one that talks more (labels arrive sorted by share) instead of
    discarding every name."""
    names, used = {}, set()
    for k in labels:
        n = re.sub(r"\(.*?\)", "", guesses.get(k, "")).strip()[:40]
        toks = re.findall(r"[a-z][a-z'.-]+", n.lower())
        if not toks or PLACEHOLDER.search(n) or n.lower() in used:
            continue
        if all(t.strip(".") in haystack for t in toks):
            names[k] = n
            used.add(n.lower())
    # Someone named in the title of a long conversation talks a lot. A
    # confirmed name on a <10% label while a >=20% label has none is a
    # swap (measured: the model called Joe Rogan's 38% label "Grok AI"
    # after he read Grok's replies aloud, and gave his name to a 6% label).
    if share:
        tot = sum(share.values()) or 1
        big = [k for k in labels if share.get(k, 0) / tot >= 0.20 and k not in names]
        for k in [k for k in list(names) if share.get(k, 0) / tot < 0.10]:
            if big:
                names[big.pop(0)] = names.pop(k)
    return names


def speaker_names(words, wd):
    """{"S1": "Joe Rogan", "S2": "JD Vance"} - one cheap LLM call per video,
    cached. Without it the model sees the guest's name in the title but not
    which label is them, and hooks misattribute quotes ("JD Vance: They
    stole the election" was Joe). Unsure -> a role or the raw label."""
    path = wd / "names.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("v") == 2:
            return cached["names"]
    if not words or "spk" not in words[0]:
        return {}
    turns, cur = [], [words[0]]
    for a, b in zip(words, words[1:]):
        if b["spk"] != a["spk"]:
            turns.append(cur)
            cur = []
        cur.append(b)
    turns.append(cur)
    # the six who talk most - numbering is order of appearance, so on a
    # 15-voice compilation S1-S6 skipped the main guest (S7, 38% of words)
    share = {}
    for w in words:
        share[w["spk"]] = share.get(w["spk"], 0) + 1
    labels = sorted(share, key=share.get, reverse=True)

    def sample(k):
        # a spread of mid-length turns (incl. questions) says more about who
        # someone is than their three longest monologues
        ts = [t for t in turns if t[0]["spk"] == k and len(t) >= 8]
        pick = ts[:: max(1, len(ts) // 8)][:8]
        return "\n".join("- " + " ".join(w["w"] for w in t[:45]) for t in pick)

    # share of words per label: hosts and guests talk far more than a clip
    # played on air (a 6% label - mostly an AI voice - got "Joe Rogan")
    samples = "\n\n".join(f"{k} ({100 * share[k] / len(words):.0f}% of all words) said:\n{sample(k)}"
                            for k in labels[:6])
    prompt = (f"{video_context(wd)}Speaker labels in this video's transcript, with sample lines:\n\n"
              f"{samples}\n\n"
              "The people in the video are named in its title (a show named after a person "
              "is hosted by that person; the name after the dash is the guest). Match each "
              "label to a person. The host asks the questions and steers; the guest talks "
              "about their own work and life. Give one short piece of evidence for each.\n"
              "Return JSON only, one entry per label above: {"
              + ", ".join(f'"{k}": {{"name": "...", "evidence": "..."}}' for k in labels[:6]) + "}")
    names = {}
    try:
        obj = llm_json(prompt, "You identify speakers in transcripts. JSON only.") or {}
        guesses = {k: str(v.get("name", "") if isinstance(v, dict) else v) for k, v in obj.items()}
        names = verify_names(guesses, labels, meta_text(wd), share)
    except RuntimeError as e:
        log(f"  speaker naming failed ({e}), keeping S1/S2")
    path.write_text(json.dumps({"v": 2, "names": names}))
    if names:
        log("speakers: " + ", ".join(f"{k} = {v}" for k, v in names.items()))
    return names


def exchange_times(words, win=20.0, step=5.0):
    """Start times of windows with unusually many speaker changes - a
    challenge and a comeback, an argument, people talking over each other.
    Needs speaker labels; [] without them."""
    if not words or "spk" not in words[0]:
        return []
    ch = [words[i]["s"] for i in range(1, len(words)) if words[i]["spk"] != words[i - 1]["spk"]]
    if not ch:
        return []
    import bisect
    t, rows = 0.0, []
    while t < words[-1]["e"]:
        rows.append((bisect.bisect_left(ch, t + win) - bisect.bisect_left(ch, t), t))
        t += step
    counts = sorted(r[0] for r in rows)
    bar = max(6, counts[int(len(counts) * 0.95)])
    picks = []
    for n, t in sorted(rows, reverse=True):
        if n < bar:
            break
        if all(abs(t - p[1]) >= 60 for p in picks):
            k = bisect.bisect_left(ch, t)
            picks.append((n, ch[k] if k < len(ch) else t))
    return picks


def resolve_niche(niche, words, wd):
    """'auto' -> ask the LLM which CRITERIA profile fits, cached per source."""
    if niche != "auto":
        return niche
    cache = wd / "niche.json"      # niche.txt predates the comedy/vlog profiles
    if cache.exists():
        return json.loads(cache.read_text())["niche"]
    n = len(words)
    sample = " ".join(w["w"] for k in (0, n // 3, 2 * n // 3) for w in words[k:k + 500])
    opts = "\n".join(f"- {k}: {v}" for k, v in NICHE_HINT.items())
    meta = load_meta(wd)
    cat = ", ".join(meta.get("categories") or [])
    hint = f"YouTube category: {cat}. Title: {meta.get('title', '')}\n\n" if meta else ""
    prompt = (f"{hint}Transcript excerpts from one video:\n\n{sample}\n\n"
              f"Which content type fits it best?\n{opts}\n\n"
              f'Return JSON only: {{"niche": "<one of: {", ".join(CRITERIA)}>"}}')
    got = "podcast"
    try:
        obj = llm_json(prompt, "You classify video transcripts. JSON only.")
        if obj and obj.get("niche") in CRITERIA:
            got = obj["niche"]
    except RuntimeError as e:
        log(f"  niche detection failed ({e}), assuming podcast")
    cache.write_text(json.dumps({"niche": got}))
    log(f"niche: {got}")
    return got


def load_meta(wd):
    m = wd / "meta.json"
    return json.loads(m.read_text()) if m.exists() else {}


def chapter_bounds(meta, words=None, reach=10.0):
    """Chapter starts, each snapped to the nearest speaker change within
    +-reach s: chapter marks are placed by hand and run a few seconds off
    (stand-up: marked 281s, the next comedian actually took over at 277.7s -
    a clip ran 5 s into the next set and cold-opened on her line)."""
    bounds = [float(c["start_time"]) for c in (meta.get("chapters") or [])[1:]]
    if not words or "spk" not in words[0]:
        return bounds
    changes = [words[i]["s"] for i in range(1, len(words)) if words[i]["spk"] != words[i - 1]["spk"]]
    out = []
    for b in bounds:
        near = [t for t in changes if abs(t - b) <= reach]
        out.append(min(near, key=lambda t: abs(t - b)) if near else b)
    return out


def clamp_to_chapters(s, e, bounds, edge=3.0):
    """A span crossing a chapter boundary -> its longest single-chapter
    piece. On compilations chapters mark where one comedian's set or one
    episode ends (a stand-up clip ran into the next comedian and even took
    its cold-open payoff from them); on podcasts they mark a topic change."""
    inner = [b for b in bounds if s + edge < b < e - edge]
    if not inner:
        return s, e
    cuts = [s, *inner, e]
    return max(zip(cuts, cuts[1:]), key=lambda p: p[1] - p[0])


def overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def usable_heat(meta):
    """Heatmap buckets minus artefacts: the opening minute (always hot -
    everyone starts at 0:00) and the buckets right after a sponsor segment
    (viewers skipping the ad land there; on Rogan-Vance the top two
    replay peaks were the ends of the two ad reads)."""
    heat = meta.get("heatmap") or []
    if not heat:
        return []
    width = heat[0]["end_time"] - heat[0]["start_time"]
    ends = [e for _, e, _ in meta.get("sponsor") or []]
    # any bucket touching [ad end - 5s, ad end + 2 buckets] - the peak bucket
    # usually straddles the ad's end (Rogan: bucket at 1146s, ad ends 1152s)
    return [b for b in heat if b["start_time"] >= 30 and not any(
        b["start_time"] <= se + 2 * width and b["end_time"] >= se - 5 for se in ends)]


def resonate(cands, reacts, heat):
    """Audience reactions inside a clip are direct evidence (+0.2 each, up to
    +0.6), and independent signals agreeing is worth more than any one
    alone - loudness alone may be music, heat alone may be an ad skip
    (hotclip's rule): +0.3 per signal beyond the first, up to +0.6."""
    hot = 0.0
    if heat:
        v = sorted(b["value"] for b in heat)
        hot = v[len(v) // 2] + 0.3 * (v[-1] - v[len(v) // 2])
    for c in cands:
        inside = [k for t, k in reacts if c["start"] <= t <= c["end"] + 2]
        sig = sorted(set(inside))
        if any(b["value"] >= hot and overlap(c["start"], c["end"], b["start_time"], b["end_time"]) > 0
               for b in heat):
            sig.append("replay heat")
        if c.get("source") in ("energy", "exchange"):
            sig.append(c["source"])
        bonus = min(0.6, 0.2 * len(inside)) + min(0.6, 0.3 * max(0, len(sig) - 1))
        if bonus:
            c["score"] = round(float(c["score"]) + bonus, 2)
            c["why"] = (c.get("why", "") + f" [signals: {', '.join(sig)}"
                        + (f"; {len(inside)} reaction(s)" if inside else "") + "]").strip()
        c["signals"] = sig


def apply_signals(cands, meta, words, reacts=()):
    """Chapter/sponsor rules, the replay-heatmap boost and the reaction /
    resonance bonus, applied to every candidate source alike. Returns the
    surviving candidates."""
    bounds, sponsor, heat = chapter_bounds(meta, words), meta.get("sponsor") or [], usable_heat(meta)
    out = []
    for c in cands:
        s, e = clamp_to_chapters(c["start"], c["end"], bounds)
        if e - s < CLIP_MIN - 5:
            continue                                   # nothing usable left inside one chapter
        if sum(overlap(s, e, a, b) for a, b, _ in sponsor) > 0.3 * (e - s):
            continue                                   # mostly an ad read / intro
        c["start"], c["end"] = round(s, 2), round(e, 2)
        out.append(c)
    resonate(out, reacts, heat)
    if not heat:
        return out
    vals = sorted(b["value"] for b in heat)
    med, top = vals[len(vals) // 2], vals[-1]
    lift = lambda v: 1.5 * max(0.0, (v - med) / (top - med + 1e-9))
    for c in out:
        v = max((b["value"] for b in heat if overlap(c["start"], c["end"], b["start_time"], b["end_time"]) > 0),
                default=0.0)
        c["heat"] = round(v, 2)
        if lift(v) > 0.05:
            c["score"] = round(float(c["score"]) + lift(v), 2)
            c["why"] = (c.get("why", "") + f" [replay heat {v:.2f}]").strip()
    # the hottest regions nobody picked become candidates of their own
    added = 0
    for b in sorted(heat, key=lambda b: -b["value"]):
        if added >= HEAT_MAX or b["value"] < med + 0.6 * (top - med):
            break
        s0, _, _ = snap_span(words, max(0.0, b["start_time"] - 5), b["start_time"] + 40, None) \
            if words else (b["start_time"], 0, 0)
        s, e = clamp_to_chapters(s0, s0 + 45, bounds)
        if e - s < CLIP_MIN or sum(overlap(s, e, a, bb) for a, bb, _ in sponsor) > 0:
            continue
        # judge the span we'd actually add, not the bucket's midpoint - a
        # snapped span can land on a moment another candidate already covers
        if any(overlap(s, e, c["start"], c["end"]) > 0.3 * (e - s) for c in out):
            continue
        out.append({"start": round(s, 2), "end": round(e, 2), "hook": "",
                    "why": f"one of the most replayed moments (heat {b['value']:.2f})",
                    "score": round(7.0 + lift(b["value"]) / 2, 2), "heat": round(b["value"], 2),
                    "source": "heat", "selected": False})
        added += 1
    return out


def cmd_find(args):
    src = Path(args.source)
    wd = workdir(src)
    tpath = wd / "transcript.json"
    if not tpath.exists():
        sys.exit("No transcript. Run: clipper.py transcribe <source>")

    words = json.loads(tpath.read_text())["words"]
    crit = CRITERIA[resolve_niche(args.niche, words, wd)]
    reacts = reactions(wd)
    names = speaker_names(words, wd)

    system = (
        "You select segments of long-form video that work as standalone "
        "short-form clips. You return JSON only. No preamble, no commentary."
    )
    crit_text = "\n".join(f"- {c}" for c in crit)

    ph = phrases(words)
    chunks = list(chunk_phrases(ph))

    # Each chunk is one LLM call and the LM Studio backend crashes reliably
    # every ~7 calls (see restart_llm_backend above), so results are
    # checkpointed per-chunk-index (not as a monotonic "done" count) - safe
    # to resume regardless of which chunks a concurrent run had finished
    # when it was interrupted.
    ckpt_path = wd / "find_checkpoint.json"
    raw_by_chunk = {}
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        if ckpt.get("total_chunks") == len(chunks) and ckpt.get("total_words") == len(words) \
                and ckpt.get("fmt") == "phrase-v1":
            raw_by_chunk = {int(k): v for k, v in ckpt.get("raw_by_chunk", {}).items()}
            if raw_by_chunk:
                log(f"resuming from checkpoint: {len(raw_by_chunk)}/{len(chunks)} chunks already done")

    ckpt_lock = threading.Lock()

    def save_checkpoint():
        ckpt_path.write_text(json.dumps({
            "total_chunks": len(chunks), "total_words": len(words), "fmt": "phrase-v1",
            "raw_by_chunk": {str(k): v for k, v in raw_by_chunk.items()},
        }))

    def process_chunk(idx, ch):
        log(f"analysing chunk {idx + 1}/{len(chunks)}")
        # Numbered phrases, not timestamps: the LLM only has to point at a
        # line it can already see, not do arithmetic over decimal seconds -
        # the real timestamp is resolved afterward from Whisper's own
        # per-word timing. Exact word-level in/out points are set later by
        # plan_clip, which sees every word of the short span.
        spk = "spk" in words[0]
        used = set()
        lines = "\n".join(
            f"[{i}] " + (f"{names.get(words[a]['spk'], words[a]['spk'])}: " if spk else "")
            + " ".join([w["w"] for w in words[a:b + 1]] + tags_after(words[b]["e"], reacts, used))
            for i, (a, b) in enumerate(ch))
        who = ("Each line starts with who is speaking. A quick back-and-forth - a "
               "challenge and a comeback, a disagreement, someone cutting in - is "
               "often the strongest material; so is one speaker's uninterrupted "
               "story.\n\n") if spk else ""
        if reacts:
            who += ("(laughter) / (applause) mark real reactions detected in the audio "
                    "right after a line - strong evidence the line landed.\n\n")
        prompt = f"""Numbered transcript phrases below, formatted [index] speaker: text.

{lines}

{who}Return 10-16 segments, each one self-contained moment or topic. For each,
give start_idx: the bracketed index of the phrase where the segment should
START. Always pick a phrase that begins a new sentence or complete thought.
Do not worry about where a segment ends or how long it runs - that is
computed afterwards from the next segment's start.

Prioritise:
{crit_text}

Reject anything that needs setup from earlier in the conversation to make sense.
Self-containment matters more than how interesting the topic is.

Return a JSON array only. Each object:
{{"start_idx": <bracketed index int>,
  "hook": "<first spoken line from that index, verbatim>",
  "reason": "<one sentence: why it holds attention with no surrounding context>",
  "self_contained": <1-10, stands alone with zero prior context - weigh this highest>,
  "hook_strength": <1-10, does the opening line create a reason to keep watching>,
  "specificity": <1-10, named numbers/dates/dollar figures/proper nouns over vague claims>,
  "emotional_turn": <1-10, a reversal/disagreement/genuine reaction>}}"""

        for attempt in range(1, 6):
            gen = _lms_generation
            try:
                content = llm(prompt, system)
                break
            except RuntimeError as e:
                if attempt == 5:
                    raise
                log(f"  chunk {idx + 1} failed (attempt {attempt}/5): {e}")
                restart_llm_backend(gen)

        parsed = []
        found = parse_json_block(content)
        if not found and not re.search(r"\[\s*\]", content or ""):
            # unparseable (not an honest empty list) - one retry with feedback
            found = llm_json(prompt, system, want=list) or []
        for c in found:
            try:
                wi = int(c["start_idx"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= wi < len(ch)):
                continue
            sub = {k: float(c.get(k, 5)) for k in
                   ("self_contained", "hook_strength", "specificity", "emotional_turn")}
            # Weights mirror AGENT.md's selection rank order: self-containment
            # > first-line hook > specificity > emotional/logical turn.
            score = (sub["self_contained"] * 0.4 + sub["hook_strength"] * 0.3
                     + sub["specificity"] * 0.2 + sub["emotional_turn"] * 0.1)
            parsed.append({"start": words[ch[wi][0]]["s"],
                            "hook": str(c.get("hook", ""))[:200],
                            "why": str(c.get("reason", ""))[:200],
                            "score": round(score, 2),
                            "sub_scores": sub})

        with ckpt_lock:
            raw_by_chunk[idx] = parsed
            save_checkpoint()

    with concurrent.futures.ThreadPoolExecutor(max_workers=LLM_PARALLEL) as ex:
        futures = [ex.submit(process_chunk, idx, ch)
                   for idx, ch in enumerate(chunks) if idx not in raw_by_chunk]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # re-raise: every worker already retried 5x internally

    raw = [c for idx in sorted(raw_by_chunk) for c in raw_by_chunk[idx]]

    # The LLM gives accurate *start* points for natural topic boundaries
    # (now exact word timestamps, not drifted arithmetic) but has no
    # reliable sense of where a topic *ends*, so each segment's real end is
    # simply the next segment's start (they are sequential, non-overlapping
    # topics), and short adjacent segments are merged using real timestamps.
    raw.sort(key=lambda c: c["start"])
    video_end = words[-1]["e"] if words else 0
    for i, c in enumerate(raw):
        c["end"] = raw[i + 1]["start"] if i + 1 < len(raw) else video_end

    merged = []
    i = 0
    while i < len(raw):
        cur = dict(raw[i])
        j = i + 1
        while (cur["end"] - cur["start"]) < CLIP_MIN and j < len(raw) \
                and (raw[j]["end"] - cur["start"]) <= CLIP_MAX:
            cur["end"] = raw[j]["end"]
            cur["why"] = (cur["why"] + " " + raw[j]["why"]).strip()
            cur["score"] = min(float(cur["score"]), float(raw[j]["score"]))
            j += 1
        merged.append(cur)
        i = j

    cands = []
    for c in merged:
        if CLIP_MIN <= (c["end"] - c["start"]) <= CLIP_MAX:
            cands.append({**c, "source": "llm", "selected": False})

    if SEMANTIC_DEDUP:
        before = len(cands)
        cands = semantic_dedup(cands, SEMANTIC_DEDUP_THRESHOLD)
        if len(cands) < before:
            log(f"  semantic dedup: {before} -> {len(cands)} candidates "
                f"(collapsed near-duplicate topics)")

    # audio spikes -> extra candidates, deduped against the LLM picks
    spikes = audio_energy(wd / "audio.wav")
    for sp in spikes:
        t = sp["t"]
        if any(c["start"] - 8 < t < c["end"] + 8 for c in cands):
            continue
        s = max(0, t - 18)
        cands.append({"start": round(s, 2), "end": round(s + 35, 2),
                      "hook": "", "why": f"audio energy spike (z={sp['z']})",
                      "score": min(9, 4 + sp["z"] / 4),
                      "source": "energy", "selected": False})

    # Capped and scored below content picks: on a debate show turns change
    # ~8x/min everywhere, and 16 of 30 slots went to these content-blind
    # windows, crowding the LLM's picks out of judging.
    added = 0
    for n, t in exchange_times(words):
        if added >= EXCHANGE_MAX:
            break
        if any(c["start"] - 8 < t < c["end"] + 8 for c in cands):
            continue
        cands.append({"start": round(t, 2), "end": round(t + 35, 2),
                      "hook": "", "why": f"rapid back-and-forth ({n} speaker changes in 20s)",
                      "score": round(min(7.0, 4 + n / 4), 2),
                      "source": "exchange", "selected": False})
        added += 1

    before = len(cands)
    cands = apply_signals(cands, load_meta(wd), words, reacts)
    if len(cands) != before or any(c["source"] == "heat" for c in cands):
        log(f"  metadata signals: {before} -> {len(cands)} candidates "
            f"({sum(c['source'] == 'heat' for c in cands)} added from replay heat)")

    cands.sort(key=lambda c: -float(c["score"]))
    cands = cands[:CAND_COUNT]
    for i, c in enumerate(cands, 1):
        c["id"] = i

    cpath = wd / "candidates.json"
    cpath.write_text(json.dumps(cands, indent=1))
    (wd / "find.version").write_text(FIND_VERSION)
    log(f"{len(cands)} candidates -> {cpath}")
    ckpt_path.unlink(missing_ok=True)


# ------------------------------------------------------------- review ------

def cmd_review(args):
    wd = workdir(args.source)
    cpath = wd / "candidates.json"
    if not cpath.exists():
        sys.exit("No candidates. Run: clipper.py find <source>")
    cands = json.loads(cpath.read_text())

    print("\n  y = keep   n = skip   q = done\n")
    for c in cands:
        dur = c["end"] - c["start"]
        mm, ss = divmod(int(c["start"]), 60)
        print(f"  #{c['id']:>2}  {mm}:{ss:02d}  {dur:.0f}s  [{c['source']}] score {c['score']}")
        if c["hook"]:
            print(f'      hook: "{c["hook"]}"')
        print(f"      {c['why']}")
        ans = input("      keep? [y/N/q] ").strip().lower()
        print()
        if ans == "q":
            break
        c["selected"] = ans == "y"
        if c["selected"] and not c["hook"]:
            c["hook"] = input("      hook text to burn on screen (blank = none): ").strip()

    cpath.write_text(json.dumps(cands, indent=1))
    n = sum(1 for c in cands if c["selected"])
    log(f"{n} selected")


# ------------------------------------------------------ speaker reframing ---

def scene_cuts(src, start, dur, thresh=0.3):
    """Camera cuts (seconds from start) via ffmpeg's own scene score - no
    extra dependency. Crops are planned per shot and never pan across a
    cut: panning through an edit is the most obvious sign of an
    auto-reframed video."""
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
         "-an", "-vf", f"scale=320:-2,select='gt(scene,{thresh})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True)
    cuts = []
    for m in re.finditer(r"pts_time:([\d.]+)", r.stderr):
        t = float(m.group(1))
        if t > 0.4 and (not cuts or t - cuts[-1] > 0.5):
            cuts.append(round(t, 2))
    return cuts


YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SFACE = ROOT / "models" / "face_recognition_sface_2021dec.onnx"


def sample_faces(src, start, dur, fps=5):
    """Faces per sampled frame: [[{cx, cy, w, h, gap}, ...], ...], all
    normalised to the frame; gap = inner-lip opening (None if the mesh
    failed). Detection: YuNet (OpenCV) on a 960 px copy - measured on 877
    Whatever frames it found every person at the table incl. profiles where
    MediaPipe's BlazeFace found 1-2 (BlazeFace shrinks input to ~192 px, so
    more resolution can't help it). Mouths: FaceMesh on the face crop from
    the full-resolution frame - 95% of faces readable vs 65% from the old
    480 px frames. MediaPipe detection remains the fallback
    (CLIPPER_FACES=mediapipe, or the YuNet model file missing).
    emb = SFace identity embedding (unit vector) when that model is present:
    lets plan_layout know the woman in a split screen is the same person
    as a tiny face in the next wide shot."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        return []
    use_yunet = YUNET.exists() and os.environ.get("CLIPPER_FACES", "yunet") == "yunet"
    tmp = Path(tempfile.mkdtemp(prefix="clipfaces_"))
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", f"{max(0.0, start):.3f}",
             "-t", f"{dur:.3f}", "-i", str(src), "-vf", f"fps={fps},scale='min(1920,iw)':-2",
             "-q:v", "3", str(tmp / "f%05d.jpg")], check=True)
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.2)
        if use_yunet:
            net = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.6, 0.3, 5000)
            rec = cv2.FaceRecognizerSF.create(str(SFACE), "") if SFACE.exists() else None
        else:
            det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)
        frames = []
        for f in sorted(tmp.glob("f*.jpg")):
            big = cv2.imread(str(f))
            if big is None:
                frames.append([])
                continue
            H_, W_ = big.shape[:2]
            small = cv2.resize(big, (960, round(H_ * 960 / W_))) if W_ > 960 else big
            boxes, embs = [], []
            if use_yunet:
                net.setInputSize((small.shape[1], small.shape[0]))
                _, found = net.detect(small)
                for x in (found if found is not None else []):
                    boxes.append((x[0] / small.shape[1], x[1] / small.shape[0],
                                  x[2] / small.shape[1], x[3] / small.shape[0]))
                    emb = None
                    if rec is not None:
                        row = x.copy()
                        row[:14] *= W_ / small.shape[1]          # box + landmarks -> full-res coords
                        e = rec.feature(rec.alignCrop(big, row)).flatten()
                        emb = e / (float((e * e).sum()) ** 0.5 + 1e-9)
                    embs.append(emb)
            else:
                for d in (det.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).detections or []):
                    bb = d.location_data.relative_bounding_box
                    boxes.append((bb.xmin, bb.ymin, bb.width, bb.height))
            rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
            faces = []
            for k, (x, y, bw, bh) in enumerate(boxes):
                x0, y0 = max(0, int(x * W_)), max(0, int(y * H_))
                x1, y1 = min(W_, int((x + bw) * W_)), min(H_, int((y + bh) * H_))
                px, py = int((x1 - x0) * 0.3), int((y1 - y0) * 0.3)
                crop = rgb[max(0, y0 - py):min(H_, y1 + py), max(0, x0 - px):min(W_, x1 + px)]
                gap = None
                if crop.size:
                    m = mesh.process(crop)
                    if m.multi_face_landmarks:
                        pts = m.multi_face_landmarks[0].landmark
                        gap = abs(pts[13].y - pts[14].y)       # inner lip top/bottom
                faces.append({"cx": x + bw / 2, "cy": y + bh / 2, "w": bw, "h": bh, "gap": gap,
                              "emb": embs[k] if embs else None})
            frames.append(faces)
        mesh.close()
        return frames
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def speaker_turns(words, start, end, min_len=1.2):
    """[(t0, t1, spk)] relative to start; turns shorter than min_len are
    absorbed (a crop that jumps for an "mm-hm" reads as a glitch)."""
    turns = []
    for w in words:
        if not (start <= w["s"] < end) or "spk" not in w:
            continue
        t0, t1 = w["s"] - start, w["e"] - start
        if turns and turns[-1][2] == w["spk"]:
            turns[-1][1] = t1
        else:
            turns.append([t0, t1, w["spk"]])
    out = []
    for t in turns:
        if out and t[1] - t[0] < min_len:
            out[-1][1] = t[1]
        elif out and out[-1][2] == t[2]:
            out[-1][1] = t[1]
        else:
            out.append(t)
    return [tuple(t) for t in out]


def face_tracks(frames, fps, a, b):
    """Faces within one shot [a, b) linked into tracks [{cx, cy, pts:
    [(t, face)]}]. Matched on x AND y with a tolerance scaled to face size:
    with every person at a panel table detected, neighbours sit ~8% of the
    frame apart - x alone within 8% merged them."""
    tracks = []
    for i in range(len(frames)):
        if not a <= i / fps < b:
            continue
        for x in frames[i]:
            tr = next((t for t in tracks if abs(t["cx"] - x["cx"]) < max(0.03, 0.6 * x["w"])
                       and abs(t["cy"] - x["cy"]) < max(0.05, 0.8 * x["h"])), None)
            if tr is None:
                tr = {"cx": x["cx"], "cy": x["cy"], "pts": []}
                tracks.append(tr)
            tr["pts"].append((i / fps, x))
            tr["cx"], tr["cy"] = x["cx"], x["cy"]
    return tracks


ASD_REPO = ROOT / "models" / "Light-ASD"


def speaking_scores(src, wav, start, dur, frames, cuts, fps=5):
    """Light-ASD (asd.py, in .venv-diar): adds "asd" to every face in
    frames, a speaking score (> 0 = this face is the one talking) from lip
    crops at 25 fps read together with the audio. Mouth-opening variance
    couldn't tell talking from laughing, or read a bearded host behind a
    mic. Leaves frames untouched if the model or venv is missing."""
    import numpy as np
    import cv2
    if not frames or not DIAR_PY.exists() or not (ASD_REPO / "weight").exists():
        return
    R = 25
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "json", str(src)], capture_output=True, text=True, check=True).stdout)["streams"][0]
    W_ = min(1920, probe["width"])
    H_ = round(probe["height"] * W_ / probe["width"] / 2) * 2
    bounds = [0.0, *[c for c in cuts if 0 < c < dur], dur]
    plans = []                               # (track, first frame, cx, cy, half-size) per 25 fps frame
    for a, b in zip(bounds, bounds[1:]):
        for t in face_tracks(frames, fps, a, b):
            if len(t["pts"]) < 5:            # under a second of face: nothing to judge
                continue
            ts = [tt for tt, _ in t["pts"]]
            f0, f1 = round(ts[0] * R), min(round(b * R), round(ts[-1] * R) + R // fps)
            ft = np.arange(f0, f1) / R
            cx = np.interp(ft, ts, [p["cx"] * W_ for _, p in t["pts"]])
            cy = np.interp(ft, ts, [p["cy"] * H_ for _, p in t["pts"]])
            hs = np.interp(ft, ts, [max(p["w"] * W_, p["h"] * H_) / 2 for _, p in t["pts"]])
            plans.append((t, f0, cx, cy, hs, []))
    if not plans:
        return
    # Light-ASD's own crop (Columbia_test.py): box widened by 0.4 per side,
    # grey, 224 px, centre 112 px
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
         "-vf", f"fps={R},scale={W_}:{H_},format=gray", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    i = 0
    while True:
        buf = proc.stdout.read(W_ * H_)
        if len(buf) < W_ * H_:
            break
        img = np.frombuffer(buf, np.uint8).reshape(H_, W_)
        for t, f0, cx, cy, hs, crops in plans:
            k = i - f0
            if 0 <= k < len(cx):
                bs = hs[k]
                x0, x1 = int(cx[k] - bs * 1.4), int(cx[k] + bs * 1.4)
                y0, y1 = int(cy[k] - bs), int(cy[k] + bs * 1.8)
                face = np.full((y1 - y0, x1 - x0), 110, np.uint8)
                sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W_, x1), min(H_, y1)
                if sx1 > sx0 and sy1 > sy0:
                    face[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
                crops.append(cv2.resize(face, (224, 224))[56:168, 56:168])
        i += 1
    proc.wait()
    audio = (load_audio(wav, start, dur) * 32767).astype(np.int16)
    tmp = Path(tempfile.mkdtemp(prefix="clipasd_"))
    try:
        arrs = {"audio": audio}
        for k, (_, f0, _, _, _, crops) in enumerate(plans):
            if crops:
                arrs[f"t{k}"], arrs[f"s{k}"] = np.stack(crops), np.array(f0)
        np.savez(tmp / "in.npz", **arrs)
        r = subprocess.run([str(DIAR_PY), str(ROOT / "asd.py"), str(tmp / "in.npz"), str(tmp / "out.npz")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            log(f"    speaker detection failed ({r.stderr.strip()[-120:]}) - using mouth motion")
            return
        out = np.load(tmp / "out.npz")
        for k, (t, f0, *_) in enumerate(plans):
            if f"t{k}" not in out.files:
                continue
            s = out[f"t{k}"]
            for tt, p in t["pts"]:
                j = round(tt * R) - f0
                w = s[max(0, j - 2):j + 3]   # Columbia's +-2 frame smoothing
                w = w[~np.isnan(w)]
                if len(w):
                    p["asd"] = float(w.mean())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _even(v):
    return int(v) - int(v) % 2


def plan_layout(frames, fps, cuts, turns, dur, sw, sh, center=False):
    """Per-shot layout plan. Returns (segments, qa):
      segments  [(t0, t1, mode, rects)] relative to the clip start, covering
                [0, dur]; mode "R" = one 9:16 rect, "S" = two 9:8 rects
                stacked (facecam over game, or two people in dialogue),
                "F" = whole frame fitted over a blurred copy
      qa        {"on_screen": fraction of speaking time the active speaker
                 is inside the crop (None if unknown), "cut_faces": frames
                 where a face straddles the crop edge}
    Choices, per shot: no face most of the time (screen recording,
    graphics, wide) -> F; a small face pinned near an edge (a facecam) ->
    S; two people trading turns too far apart for one crop -> S; otherwise
    one locked crop per speaker turn on whoever's mouth moves, which only
    moves when the new speaker is meaningfully elsewhere (hysteresis)."""
    W, H = sw, sh
    full_w = _even(min(W, round(H * 9 / 16)))

    def rect_r(cx, cy, fw, cap=None):
        if cap:                             # grid of boxes: exactly one box wide
            rw = max(OUT_W / 2.8, min(full_w, cap))
        else:
            # face about a third of the crop width, whatever the detector's box
            # style (YuNet's whole-head boxes are ~1.3x BlazeFace's, so a fixed
            # "small face" cutoff left medium shots unzoomed and off-centre);
            # at most ~1.9x zoom, never wider than a full-height crop
            # zoom floor in OUTPUT pixels: 0.3 of the source width meant ~1.9x
            # on 1080p but no zoom at all on a 4K source, which has the
            # pixels for a tight, sharp crop
            rw = max(OUT_W / MAX_ZOOM, min(full_w, 3.2 * fw * W))
        rh = min(H, rw * 16 / 9)
        rw = rh * 9 / 16
        x = min(max(0, cx * W - rw / 2), W - rw)
        y = 0 if rh >= H else min(max(0, cy * H - rh * 0.35), H - rh)
        return (_even(x), _even(y), _even(rw), _even(rh))

    def rect_s(cx, cy, fw, scale=2.6):
        rw = max(OUT_W / MAX_ZOOM, min(W, H * 9 / 8, scale * fw * W))
        rh = rw * 8 / 9
        x = min(max(0, cx * W - rw / 2), W - rw)
        y = min(max(0, cy * H - rh * 0.4), H - rh)
        return (_even(x), _even(y), _even(rw), _even(rh))

    centre = rect_r(0.5, 0.5, 1.0)
    bounds = [0.0, *[c for c in cuts if 0 < c < dur], dur]
    shots = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a > 1e-3]

    def shot_tracks(a, b):
        tracks = face_tracks(frames, fps, a, b)
        # Persistence: a face seen in under a quarter of the shot is a false
        # alarm (fire in B-roll, a face on a phone screen), not a person.
        n = sum(1 for i in range(len(frames)) if a <= i / fps < b)
        tracks = [t for t in tracks if len(t["pts"]) >= max(2, 0.25 * n)]
        for t in tracks:
            t["cx"] = statistics.median(p["cx"] for _, p in t["pts"])
            t["cy"] = statistics.median(p["cy"] for _, p in t["pts"])
            t["w"] = statistics.median(p["w"] for _, p in t["pts"])
            g = [p["gap"] for _, p in t["pts"] if p["gap"] is not None]
            t["talk"] = statistics.pvariance(g) if len(g) >= 3 else 0.0
            es = [p["emb"] for _, p in t["pts"] if p.get("emb") is not None]
            t["emb"] = sum(es) / len(es) if es else None
            t["id"] = None
        return tracks

    step = 1.5 / fps

    def diffs(t):
        """Frame-to-frame lip-gap changes [(t1, t2, |d|)] - mouth motion."""
        pts = sorted((tt, p["gap"]) for tt, p in t["pts"] if p["gap"] is not None)
        return [(t1, t2, abs(g2 - g1)) for (t1, g1), (t2, g2) in zip(pts, pts[1:]) if t2 - t1 <= step]

    def link(cands, sp_turns, sp):
        """The face (or identity) that owns voice sp - only as a clear winner:
        its mouth moves >= 3x more while sp talks than otherwise, and >= 1.5x
        more than the runner-up (ratios, eps-stabilised). An off-camera
        speaker otherwise got whichever neighbour happened to move."""
        eps, scores = 0.0005, []
        on = lambda tt: any(t0 <= tt < t1 for t0, t1, s_ in sp_turns if s_ == sp)  # noqa: E731
        for key, ds in cands:
            d_in = [d for t1, t2, d in ds if on(t1) and on(t2)]
            d_out = [d for t1, t2, d in ds if not on(t1) and not on(t2)]
            if len(d_in) >= 2:
                m_out = statistics.mean(d_out) if len(d_out) >= 2 else 0.0
                scores.append(((statistics.mean(d_in) + eps) / (m_out + eps), key))
        scores.sort(key=lambda x: -x[0])
        if len(scores) >= 2 and scores[0][0] >= 3 and scores[0][0] >= 1.5 * scores[1][0]:
            return scores[0][1]
        return None

    def speaking(pts, s0=-1.0, s1=1e9):
        """Mean Light-ASD score of these (t, face) samples in [s0, s1), or None."""
        v = [p["asd"] for tt, p in pts if s0 <= tt < s1 and p.get("asd") is not None]
        return statistics.mean(v) if len(v) >= 2 else None

    def asd_link(cands, sp_turns, sp):
        """Like link(), on speaking scores: the candidate whose score RISES
        while sp talks (vs. while others talk) by >= ASD_MARGIN, and clearly
        more than the runner-up's. Contrast, not level: a mic in front of a
        guest's mouth held her score below zero even while she talked."""
        on = lambda tt: any(t0 <= tt < t1 for t0, t1, s_ in sp_turns if s_ == sp)  # noqa: E731
        other = lambda tt: any(t0 <= tt < t1 for t0, t1, s_ in sp_turns if s_ != sp)  # noqa: E731
        scores = []
        for key, pts in cands:
            m_in = speaking([q for q in pts if on(q[0])])
            if m_in is None:
                continue
            m_out = speaking([q for q in pts if other(q[0])])
            scores.append((m_in - m_out if m_out is not None else m_in, key))
        scores.sort(key=lambda x: -x[0])
        if scores and scores[0][0] >= ASD_MARGIN and (len(scores) == 1 or scores[0][0] - scores[1][0] >= ASD_MARGIN / 2):
            return scores[0][1]
        return None

    pre = {} if center or not frames else {s: shot_tracks(*s) for s in shots}
    has_asd = any(p.get("asd") is not None for f in frames for p in f)
    # Identity across shots (SFace): a multi-cam show cuts every 2-5 s, too
    # little mouth motion per shot to tell who talks - the host's 2 s
    # question never cleared the bar, so the crop stayed on the guest. Faces
    # are grouped into people over the whole clip and each voice is linked
    # to a person on all of their footage at once. Two faces in one shot are
    # never the same person.
    people = []
    for s, trs in pre.items():
        for t in sorted(trs, key=lambda t: -len(t["pts"])):
            if t["emb"] is None:
                continue
            e = t["emb"] / (float((t["emb"] * t["emb"]).sum()) ** 0.5 + 1e-9)
            sim, k = max(((float(e @ p["emb"]) / (float((p["emb"] * p["emb"]).sum()) ** 0.5 + 1e-9), k)
                          for k, p in enumerate(people) if s not in p["shots"]), default=(0.0, None))
            if sim < SAME_FACE:
                k = len(people)
                people.append({"emb": 0 * e, "shots": set(), "diffs": []})
            people[k]["emb"] = people[k]["emb"] + e
            people[k]["shots"].add(s)
            people[k]["diffs"] += diffs(t)
            people[k].setdefault("pts", []).extend(t["pts"])
            t["id"] = k
    person_of = {}
    for sp in {sp for _, _, sp in turns}:
        k = asd_link([(k, p["pts"]) for k, p in enumerate(people)], turns, sp) if has_asd else \
            link([(k, p["diffs"]) for k, p in enumerate(people)], turns, sp)
        if k is not None:
            person_of[sp] = k

    segs, spoke, seen, cut_faces = [], 0.0, 0.0, 0
    for a, b in shots:
        fr = [frames[i] for i in range(len(frames)) if a <= i / fps < b]
        if center or not frames:
            segs.append((a, b, "R", [centre]))
            continue
        if not fr:                          # shot shorter than one sample: keep previous framing
            segs.append((a, b, *segs[-1][2:]) if segs else (a, b, "R", [centre]))
            continue
        tracks = pre[(a, b)]
        covered = {round(tt * fps) for t in tracks for tt, _ in t["pts"]}
        if not tracks or len(covered) / len(fr) < 0.4:
            segs.append((a, b, "F", []))
            continue
        # Facecam: a small face pinned near an edge for most of the shot whose
        # mouth moves. Judged per track - a game HUD's static character
        # portrait sits at the same edge (Jynxzi/R6) and never talks.
        cams = [t for t in tracks
                if len(t["pts"]) >= 0.6 * len(fr) and t["w"] < 0.22 and (t["cx"] < 0.3 or t["cx"] > 0.7)
                and statistics.pstdev([p["cx"] for _, p in t["pts"]]) < 0.05]
        talking = [t for t in cams if t["talk"] > 0] or (cams if len(tracks) == 1 else [])
        peak = max((t["talk"] for t in tracks), default=0.0)
        # two faces talking comparably = a conversation shot with people near
        # the edges, not a facecam (a two-guest wide shot looked like one)
        conversation = sum(1 for t in tracks if peak and t["talk"] >= 0.3 * peak) >= 2
        cam = max(talking, key=lambda t: t["talk"]) if talking else None
        # the facecam is the shot's main talker - a neighbour fidgeting near the
        # frame edge while someone centre-frame talks is a panel, not a stream
        if cam and len(tracks) <= 3 and not conversation and cam["talk"] >= 0.7 * peak:
            # game panel: centred, but pushed clear of the webcam overlay (about
            # 3.4 face-widths wide) so the face doesn't appear twice
            gw = min(W, H * 9 / 8)
            gx = (W - gw) / 2
            if cam["cx"] < 0.5:
                gx = max(gx, (cam["cx"] + 1.7 * cam["w"]) * W)
            else:
                gx = min(gx, (cam["cx"] - 1.7 * cam["w"]) * W - gw)
            gx = min(max(0, gx), W - gw)
            game = (_even(gx), 0, _even(gw), _even(gw * 8 / 9))
            segs.append((a, b, "S", [rect_s(cam["cx"], cam["cy"], cam["w"], 2.2), game]))
            continue
        def by_var(s0, s1):
            best, var = None, 0.0
            for t in tracks:
                g = [p["gap"] for tt, p in t["pts"] if s0 <= tt < s1 and p["gap"] is not None]
                if len(g) >= 3 and statistics.pvariance(g) > var:
                    best, var = t, statistics.pvariance(g)
            return best

        # Voice -> face: a face belongs to a diarized speaker if its mouth
        # moves MORE while that speaker talks than while they're silent.
        # Picking the busiest mouth per turn (by_var) broke once every
        # person at a panel table was detected: a neighbour laughing or
        # drinking out-moved the actual speaker in short windows.
        shot_turns = [(max(a, t0), min(b, t1), sp) for t0, t1, sp in turns if t1 > a and t0 < b]
        owner = {}
        for sp in {sp for _, _, sp in shot_turns}:
            if sp in person_of:             # known person: their face here, or None = off camera
                owner[sp] = next((t for t in tracks if t["id"] == person_of[sp]), None)
                continue
            t = asd_link([(t, t["pts"]) for t in tracks], shot_turns, sp) if has_asd else \
                link([(t, diffs(t)) for t in tracks], shot_turns, sp)
            if t is not None:
                owner[sp] = t

        def talker(s0, s1):
            # the face that is visibly talking right now wins outright -
            # robust to a mislabeled diarization turn
            sc = sorted(((m, t) for t in tracks if (m := speaking(t["pts"], s0, s1)) is not None),
                        key=lambda x: -x[0])
            if sc and sc[0][0] > 0 and (len(sc) == 1 or sc[0][0] - sc[1][0] >= ASD_MARGIN):
                return sc[0][1]
            best_asd = sc[0][1] if sc else None
            mid = (s0 + s1) / 2
            sp = next((s_ for t0, t1, s_ in shot_turns if t0 <= mid < t1), None)
            if sp in owner:
                return owner[sp]            # None: known speaker, off camera in this shot
            # a diarized voice with no face of its own here is off camera:
            # hold the framing on a known speaker rather than guess a face
            if sp is not None and any(owner.values()):
                return None
            # nobody decisive and the voice unknown: the face reading most
            # like a talker is a better guess than the busiest mouth
            return best_asd or by_var(s0, s1)

        # A row of 3+ small faces at one height is a split-screen grid (news
        # panel): crop exactly one box, sized by the spacing between faces -
        # the 0.3W zoom floor showed half the neighbour's box (Piers Morgan).
        cap = None
        row = sorted(t["cx"] for t in tracks if len(t["pts"]) >= 0.3 * len(fr))
        if len(row) >= 3 and max(t["w"] for t in tracks) < 0.1 and \
                statistics.pstdev([t["cy"] for t in tracks]) < 0.06:
            cap = 0.92 * min(q - p for p, q in zip(row, row[1:])) * W

        marks = sorted({a, b, *[t0 for t0, _, _ in turns if a < t0 < b]})
        subs = []
        for s0, s1 in zip(marks, marks[1:]):
            tr = talker(s0, s1)
            spoke += (s1 - s0) if tr else 0
            subs.append([s0, s1, tr, tr is not None])
        for k, sub in enumerate(subs):      # unknown talker -> previous, else next known, else biggest face
            if sub[2] is None:
                nxt = next((x[2] for x in subs[k + 1:] if x[2] is not None), None)
                sub[2] = subs[k - 1][2] if k else (nxt or max(tracks, key=lambda t: t["w"]))
        share = {}
        for s0, s1, tr, _ in subs:
            share[id(tr)] = share.get(id(tr), 0) + (s1 - s0)
        top = sorted(tracks, key=lambda t: -share.get(id(t), 0))[:2]
        if len(top) == 2 and all(share.get(id(t), 0) >= 0.25 * (b - a) for t in top) \
                and abs(top[0]["cx"] - top[1]["cx"]) * W > 0.9 * full_w:
            first = next(tr for _, _, tr, _ in subs if any(tr is t for t in top))
            pair = [first, top[1] if first is top[0] else top[0]]
            segs.append((a, b, "S", [rect_s(t["cx"], t["cy"], t["w"]) for t in pair]))
            seen += sum(s1 - s0 for s0, s1, _, known in subs if known)
            continue
        prev = None
        for s0, s1, tr, known in subs:
            r = rect_r(tr["cx"], tr["cy"], tr["w"], cap)
            if prev and abs((prev[0] + prev[2] / 2) - (r[0] + r[2] / 2)) < 0.06 * W and prev[2] == r[2]:
                r = prev                    # same place: don't nudge the camera
            if segs and segs[-1][2] == "R" and segs[-1][3][0] == r and segs[-1][1] == s0:
                segs[-1] = (segs[-1][0], s1, "R", [r])
            else:
                segs.append((s0, s1, "R", [r]))
            prev = r
            if known and r[0] <= tr["cx"] * W <= r[0] + r[2]:
                seen += s1 - s0
            for tt, p in [q for t in tracks for q in t["pts"] if s0 <= q[0] < s1]:
                inside = max(0.0, min(r[0] + r[2], (p["cx"] + p["w"] / 2) * W) - max(r[0], (p["cx"] - p["w"] / 2) * W))
                if 0.2 < inside / max(1e-6, p["w"] * W) < 0.8:
                    cut_faces += 1
    return segs, {"on_screen": round(seen / spoke, 3) if spoke else None, "cut_faces": cut_faces,
                  "talker": round(spoke / dur, 3) if dur else 0.0, "people": len(people),
                  "voices": person_of,
                  "ids": [(a, round(t["cx"], 2), t["id"]) for (a, _), trs in pre.items() for t in trs]}


def layout_pieces(blocks, segs, start, fps):
    """Split the edit (blocks of kept spans, playback order) at layout
    segment boundaries -> [(seek_abs, [(a, b) rel to seek], seg)]. Split
    points are snapped to each block's frame grid so caption timing can't
    drift from the video."""
    segs = sorted(segs)
    frags = []
    for bs, spans in blocks:
        grid = lambda t: bs + round((t - bs) * fps) / fps
        for ra, rb in spans:
            a, b = bs + ra, bs + rb
            for k, sg in enumerate(segs):
                s0 = -1e9 if k == 0 else grid(start + sg[0])
                s1 = 1e9 if k == len(segs) - 1 else grid(start + sg[1])
                x0, x1 = max(a, s0), min(b, s1)
                if x1 - x0 > 0.5 / fps:
                    frags.append((x0, x1, k))
    pieces = []
    for x0, x1, k in frags:
        if pieces and pieces[-1][2] == k and x0 >= pieces[-1][3] - 1e-6:
            pieces[-1][1].append((x0, x1))
            pieces[-1][3] = x1
        else:
            pieces.append([x0, [(x0, x1)], k, x1])
    return [(p[0], [(a - p[0], b - p[0]) for a, b in p[1]], segs[p[2]]) for p in pieces]


def piece_filter(i, spans, seg, fps):
    """Filter chain for one piece -> [v{i}] 1080x1920 and [a{i}]."""
    h = 0.5 / fps                            # half-frame: exactly (b-a)*fps frames
    vsel = "+".join(f"gte(t,{a - h:.4f})*lt(t,{b - h:.4f})" for a, b in spans)
    asel = "+".join(f"gte(t,{a:.4f})*lt(t,{b:.4f})" for a, b in spans)
    # Exactly round(duration x fps) frames per piece: each piece seeks on its
    # own and can land a frame early/late, which summed to 0.8 s of A/V
    # drift over a 20-piece clip. Clone the last frame if short, trim if long.
    n = round(sum(b - a for a, b in spans) * fps)
    head = (f"[{i}:v]select='{vsel}',setpts=N/FRAME_RATE/TB,"
            f"tpad=stop_mode=clone:stop=3,trim=end_frame={n}")
    mode, rects = seg[2], seg[3]
    if mode == "R":
        x, y, w, hh = rects[0]
        v = f"{head},crop={w}:{hh}:{x}:{y},scale={W}:{H}:flags=lanczos,setsar=1[v{i}]"
    elif mode == "S":
        (x1, y1, w1, h1), (x2, y2, w2, h2) = rects
        v = (f"{head},split=2[t{i}][b{i}];"
             f"[t{i}]crop={w1}:{h1}:{x1}:{y1},scale={W}:{H // 2}:flags=lanczos[tt{i}];"
             f"[b{i}]crop={w2}:{h2}:{x2}:{y2},scale={W}:{H // 2}:flags=lanczos[bb{i}];"
             f"[tt{i}][bb{i}]vstack,setsar=1[v{i}]")
    else:                                     # F: whole frame over a blurred copy of itself
        v = (f"{head},split=2[g{i}][f{i}];"
             f"[g{i}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=24:4[gb{i}];"
             f"[f{i}]scale={W}:-2:flags=lanczos[ff{i}];"
             f"[gb{i}][ff{i}]overlay=(W-w)/2:(H-h)/2,setsar=1[v{i}]")
    return v + f";[{i}:a]aselect='{asel}',asetpts=N/SR/TB[a{i}]"


# -------------------------------------------------------------- editing ----

FILLERS = {"um", "uh", "uhm", "umm", "uhh", "erm", "er", "ah", "hmm", "mm"}
GAP_MAX = 0.45            # silence longer than this between words is cut
PAD_IN, PAD_OUT = 0.06, 0.10
ZOOM_CUT = 1.06           # framing alternates 1.0 / this across jump cuts
ZOOM_PULSE = 0.10         # extra punch-in on emphasised words
SFX_VOL = {"whoosh": 0.22, "pop": 0.12}


def _is_filler(w):
    return re.sub(r"[^a-z]", "", w["w"].lower()) in FILLERS


def build_timeline(words, start, end, loop=None, cuts=True, fps=30.0, quiet=None, edge=None):
    """Plan the edit as blocks of kept spans, in playback order.

    Returns (blocks, out_words, total, bounds, seam):
      blocks    [(block_src_start, [(a, b), ...])] - spans in seconds
                relative to block_src_start, snapped to the frame grid so
                caption timing can't drift from the cut video
      out_words words re-timed onto the output timeline (+ src_s for loudness)
      bounds    output times of every cut
      seam      output time where the cold open hands over, else None

    cuts: drop silences > GAP_MAX and filler words, using Whisper's own word
    boundaries - more precise than any amplitude-based silence detector.
    quiet(a, b) -> bool vetoes cutting a gap that isn't actually silent:
    between words there may be music, laughter or a crowd reacting, and on
    a stage show or stream that IS the content.
    edge(t, lo, hi) -> t' moves each block's outer cuts to the quietest
    point nearby (Whisper word times drift ~0.2s, so a cut on the word
    boundary can clip an attack or a tail); a block may then start up to
    0.3s before its first word.
    loop=(ps, pe): cold open - play the payoff line first, then the clip
    from `start` up to just before the payoff, so the last frame flows
    straight back into the first on replay."""
    blocks_src = [(loop[0], loop[1]), (start, loop[0])] if loop else [(start, end)]
    blocks, out, bounds, t = [], [], [], 0.0
    for bs, be in blocks_src:
        ws = [w for w in words if bs - 0.02 <= w["s"] < be - 0.02]
        parts = []
        if cuts:
            prev, broke = None, False
            for w in ws:
                if _is_filler(w):
                    broke = True
                    continue
                a, b = max(bs, w["s"] - PAD_IN), min(be, w["e"] + PAD_OUT)
                if parts and not broke and (w["s"] - prev["e"] <= GAP_MAX or (
                        quiet and not quiet(prev["e"], w["s"]))):
                    parts[-1][1] = b
                else:
                    parts.append([max(a, parts[-1][1]) if parts else a, b])
                prev, broke = w, False
        if not parts:
            parts = [[bs, be]]
        if edge:
            parts[0][0] = max(bs - 0.3, edge(parts[0][0], -0.3, 0.02))
            parts[-1][1] = min(be + 0.3, edge(parts[-1][1], -0.02, 0.3))
            bs = min(bs, parts[0][0])                      # seek point may move earlier
        spans = []
        for a, b in parts:
            ra, rb = round((a - bs) * fps) / fps, round((b - bs) * fps) / fps
            if rb - ra < 2 / fps:
                continue
            for w in ws:
                if bs + ra <= w["s"] < bs + rb and not (cuts and _is_filler(w)):
                    out.append({"w": w["w"], "s": t + w["s"] - bs - ra,
                                "e": t + min(w["e"], bs + rb) - bs - ra, "src_s": w["s"]})
            if t > 0:
                bounds.append(t)
            spans.append((ra, rb))
            t += rb - ra
        if not spans:
            if loop:  # payoff too short to survive the cuts - play it straight
                return build_timeline(words, start, end, None, cuts, fps)
            spans = [(0.0, be - bs)]
            t += be - bs
        blocks.append((bs, spans))
    seam = sum(b - a for a, b in blocks[0][1]) if loop else None
    return blocks, out, t, bounds, seam


def load_audio(wav, start, dur):
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(wav),
         "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def edge_fn(a, start):
    """edge(t, lo, hi): centre of the quietest 10 ms window in [t+lo, t+hi]."""
    import numpy as np

    def edge(t, lo, hi):
        i0, i1 = int((t + lo - start) * 16000), int((t + hi - start) * 16000)
        i0, i1 = max(0, i0), min(len(a), i1)
        if i1 - i0 < 320:
            return t
        n = (i1 - i0) // 160
        rms = np.sqrt((a[i0:i0 + n * 160].reshape(n, 160) ** 2).mean(axis=1))
        return start + (i0 + int(rms.argmin()) * 160 + 80) / 16000
    return edge


def quiet_fn(a, start, words):
    """quiet(t0, t1): is the gap well below this clip's speech level?
    ponytail: one global ratio vs median word loudness; per-speaker
    levels if quiet talkers get their pauses kept."""
    import numpy as np
    rms = lambda t0, t1: float(np.sqrt((a[max(0, int((t0 - start) * 16000)):
                                          max(0, int((t1 - start) * 16000))] ** 2).mean() + 1e-12))
    ref = np.median([rms(w["s"], w["e"]) for w in words if w["e"] > w["s"]] or [0.1])
    return lambda t0, t1: t1 - t0 < 0.1 or rms(t0 + 0.05, t1 - 0.05) < ref * 0.18


def emphasis_times(a, start, out_words, total):
    """Output times of the loudest words, >= 4s apart, so zoom pulses land
    on real vocal emphasis instead of at random."""
    import numpy as np
    if len(out_words) < 5:
        return []
    lv = []
    for w in out_words:
        i0 = max(0, int((w["src_s"] - start) * 16000))
        seg = a[i0:i0 + int(max(0.08, w["e"] - w["s"]) * 16000)]
        lv.append(float(np.sqrt((seg ** 2).mean())) if len(seg) else 0.0)
    lv = np.array(lv)
    z = (lv - lv.mean()) / (lv.std() + 1e-9)
    picks, cap = [], max(1, int(total // 10))
    for i in np.argsort(-z):
        if z[i] < 1.0 or len(picks) >= cap:
            break
        t = out_words[i]["s"]
        if 1.0 <= t <= total - 1.5 and all(abs(t - p) >= 4 for p in picks):
            picks.append(t)
    return sorted(picks)


def zoom_expr(toggles, pulses):
    """zoompan z(it). Alternating framing across jump cuts makes a cut read
    as an edit rather than a glitch; pulses are a quick punch-in (0.1s in,
    ~1s hold, 0.25s out) on emphasis."""
    lvl = lambda k: "1" if k % 2 == 0 else f"{ZOOM_CUT}"
    expr = lvl(len(toggles))
    for i in range(len(toggles) - 1, -1, -1):
        expr = f"if(lt(it,{toggles[i]:.3f}),{lvl(i)},{expr})"
    for p in pulses:
        expr += (f"+{ZOOM_PULSE}*clip((it-{p:.3f})/0.1,0,1)"
                 f"*clip(({p + 1.3:.3f}-it)/0.25,0,1)")
    return expr


def missing_frames(path, fps):
    """How many frames short a rendered file is of duration x fps (0 = ok)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,duration", "-of", "json", str(path)],
        capture_output=True, text=True).stdout
    st = (json.loads(out or "{}").get("streams") or [{}])[0]
    try:
        return max(0, round(float(st["duration"]) * fps) - int(st["nb_frames"]) - 1)
    except (KeyError, ValueError):
        return 0


def sfx_files():
    """Procedural SFX, generated once with ffmpeg itself - no asset
    downloads or licensing to track."""
    d = WORK / "_sfx"
    d.mkdir(parents=True, exist_ok=True)
    specs = {
        "whoosh": "anoisesrc=d=0.5:c=pink:a=0.5,highpass=f=500,lowpass=f=6000,"
                  "afade=t=in:d=0.32:curve=qsin,afade=t=out:st=0.32:d=0.18",
        "pop": "sine=f=1200:d=0.1,afade=t=out:d=0.1:curve=exp",
    }
    for name, graph in specs.items():
        p = d / f"{name}.wav"
        if not p.exists():
            run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                 graph + ",aformat=sample_rates=48000:channel_layouts=stereo", str(p)])
    return {k: d / f"{k}.wav" for k in specs}


# ----------------------------------------------------------- captions ------

def ass_time(t):
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def esc(s):
    return s.replace("\\", "").replace("{", "").replace("}", "")


def build_ass(seg, total, hook, path):
    """Word-by-word highlighted captions, the standard short-form look.
    One Dialogue event per word-state so the active word pops. `seg` is
    already on the output timeline (see build_timeline)."""
    y = int(H * CAPTION_Y)

    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Cap,{FONT},{FONT_SIZE},{BASE_COLOR},&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{OUTLINE},3,5,60,60,0,1
Style: Hook,{FONT},{int(FONT_SIZE*0.92)},{BASE_COLOR},&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{OUTLINE},3,5,110,110,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    ev = []

    if hook:
        txt = esc(hook.upper())
        # WrapStyle 0 (smart wrap) breaks long hooks over 2-3 balanced
        # lines; with wrapping off they ran off both edges of the frame.
        if len(txt) > 90:
            txt = txt[:87] + "..."
        ev.append(f"Dialogue: 0,{ass_time(0)},{ass_time(2.2)},Hook,,0,0,0,,"
                  f"{{\\pos({W//2},{int(H*0.26)})\\fad(120,220)}}{txt}")

    caps = []                                   # (s, e, text) per word-state
    groups = [seg[i:i + WORDS_PER_GROUP] for i in range(0, len(seg), WORDS_PER_GROUP)]
    for g in groups:
        toks = [esc(w["w"]).upper() for w in g]
        for i, w in enumerate(g):
            s = max(0.0, w["s"])
            e = max(s + 0.08, w["e"])
            if i == len(g) - 1:
                e = max(e, min(g[-1]["e"] + 0.06, total))
            parts = []
            for j, tok in enumerate(toks):
                if j == i:
                    # Eased pop instead of an instant scale jump: overshoot
                    # to 118% in the first ~80ms then settle to 108%, timed
                    # relative to this dialogue event's own start (\t times
                    # are ms-from-line-start, not the \k centisecond clock).
                    dur_ms = max(1, int((e - s) * 1000))
                    pop = min(80, dur_ms // 2)
                    parts.append(
                        f"{{\\c{HILITE_COLOR}\\fscx100\\fscy100"
                        f"\\t(0,{pop},\\fscx118\\fscy118)"
                        f"\\t({pop},{dur_ms},\\fscx108\\fscy108)}}{tok}"
                        f"{{\\c{BASE_COLOR}\\fscx100\\fscy100}}")
                else:
                    parts.append(tok)
            caps.append([s, e, " ".join(parts)])
    # Silence cuts pull words closer than their padded display times, so
    # clamp each state to end where the next begins - otherwise two caption
    # lines render on top of each other.
    for i in range(len(caps) - 1):
        caps[i][1] = max(caps[i][0] + 0.02, min(caps[i][1], caps[i + 1][0]))
    for s, e, txt in caps:
        ev.append(f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Cap,,0,0,0,,"
                  f"{{\\pos({W//2},{y})}}" + txt)

    Path(path).write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


# --------------------------------------------------------------- hooks -----

# Order matters: index N here is what --platform picks. Never bake the same
# on-screen hook text into copies meant for different platforms - identical
# hook text across platforms is a duplicate-content signal that gets clips
# suppressed.
PLATFORMS = ("shorts", "tiktok", "instagram")


def parse_json_obj(text):
    m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", text or "", flags=re.S), flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def llm_json(prompt, system, want=dict):
    """llm_retry + parse, with one retry that tells the model what was wrong.
    Small local models break the JSON contract often enough that feeding
    the problem back converts most failures (pattern from artbyjazi/autoclip).
    Returns the parsed dict/list, or None."""
    parse = parse_json_obj if want is dict else parse_json_block
    reply = llm_retry(prompt, system)
    got = parse(reply)
    if got or (want is list and re.search(r"\[\s*\]", reply or "")):
        return got
    kind = "JSON object {...}" if want is dict else "JSON array [...]"
    log(f"  reply wasn't a valid {kind} - asking again with feedback")
    fix = (f"{prompt}\n\nYour previous reply could not be parsed as a {kind}:\n"
           f"{(reply or '')[:400]}\n\nReply again with ONLY the {kind}, nothing else.")
    return parse(llm_retry(fix, system)) or None


def llm_retry(prompt, system, tries=3):
    """llm() with backend-restart retries; raises RuntimeError after `tries`."""
    for attempt in range(1, tries + 1):
        gen = _lms_generation
        try:
            return llm(prompt, system)
        except RuntimeError as e:
            if attempt == tries:
                raise
            log(f"  LLM call failed (attempt {attempt}/{tries}): {e}")
            restart_llm_backend(gen)


OPENERS = re.compile(r"^(?:(?:so|and|but|like|okay|ok|alright|all right|um|uh|yeah|well|"
                     r"look|you know)[,.]?\s+)+", re.I)
# swearing and explicit sexual terms: both cost reach / ad eligibility as the lead hook
PROFANE = re.compile(r"\b(fuck\w*|f\*+\w*|shit\w*|bitch\w*|cunt|dick\w*|asshole\w*|motherf\w*|"
                     r"clit\w*|puss(y|ies)|cocks?|tits?|boobs?|cum|porn\w*|horny|orgasm\w*|"
                     r"blowjob\w*|dildo\w*|anal)\b", re.I)


# lines that announce the point instead of making it ("Let me read the quote.")
TRANSITION = re.compile(r"(next up|let me (read|tell you|explain|ask you)|here'?s the thing|"
                        r"moving on|first of all|number (one|two|three))\b", re.I)


def tidy(line):
    """Trim filler openers ("So, ", "And ", "Like, ") - still verbatim,
    just starting where the point starts."""
    line = OPENERS.sub("", line).strip(" ,;")
    return line[:1].upper() + line[1:]


def clauses(ws, lo=4, hi=12, pieces=False):
    """Clip words -> the menu hook quotes are chosen from: whole sentences
    (split at sentence ends and speaker changes) of lo..hi words. With
    pieces=True, a longer sentence also offers its opening comma piece -
    only the opening one: a middle piece ("if they should", "26 year
    old.") doesn't stand alone. Transition lines ("Next up, ...") are
    dropped: they introduce the point instead of making it."""
    sents, cur = [], []
    for i, w in enumerate(ws):
        if cur and w.get("spk") != ws[i - 1].get("spk"):
            sents.append(cur)
            cur = []
        cur.append(w["w"])
        if re.search(r"[.?!]$", w["w"]):
            sents.append(cur)
            cur = []
    sents.append(cur)
    out = []
    for sent in sents:
        cand, piece = " ".join(sent), False
        if pieces and len(sent) > hi:
            piece = True
            head = []
            for t in sent:
                head.append(t)
                if re.search(r"[,;]$", t):
                    break
            cand = " ".join(head)
        line = tidy(cand)
        # a short line with no sentence end was cut off by the next speaker
        # ("It's a bit of a") - punctuation is reliable since the ASR fix
        cut_off = not piece and not re.search(r"[.?!]$", line) and len(line.split()) < 7
        if lo <= len(line.split()) <= hi and not TRANSITION.match(line) and not cut_off \
                and line not in out:
            out.append(line)
    return out


def generate_hooks(ws, why, context="", names=None, payoff=""):
    """On-screen hooks as VERBATIM quotes picked from a numbered menu of the
    clip's own clauses. Written hooks from a 14B model invented facts about
    named people ("VP admits secret facility exists"), a second model pass
    could not reliably catch it (measured: dropped true hooks, kept false
    ones), and free word-index picks came back as fragments ("companies
    that had"). Choosing whole clauses from a list can't invent and can't
    cut mid-thought. No name prefix: someone reading a quote aloud isn't its
    source, and the viewer sees who is talking anyway.
    A loop clip's payoff is left out: it already plays (and is captioned)
    in the first seconds, so as hook text it would show the same words twice.
    Returns {"hooks": [...], "caption": "..."}."""
    names = names or {}
    menu = clauses(ws)
    if len(menu) < 2 * len(PLATFORMS):
        menu += [m for m in clauses(ws, pieces=True) if m not in menu]
    if payoff:
        p = tidy(payoff)
        menu = [m for m in menu if m != p and p not in m]
    who = sorted({names.get(w.get("spk"), "") for w in ws} - {""})
    listing = "\n".join(f"{i}. {m}" for i, m in enumerate(menu))
    system = ("You pick on-screen hook quotes for vertical short-form video clips. "
              "You return JSON only. No preamble, no commentary.")
    prompt = f"""{context}Lines spoken in one clip{(" by " + " and ".join(who)) if who else ""}:

{listing}

Why the clip holds attention: {why}

Pick the 5 lines that would work best as the on-screen hook - the words
shown for the first 2 seconds to stop a stranger from scrolling: a shocking
claim, a punchline, a vivid specific detail (a name, a number, an object).
A line must make sense on its own, without the rest of the clip.

Also write a caption: one sentence for the post text naming who is talking
and the TOPIC only - e.g. "Joe Rogan and JD Vance on alien remains and
government secrecy". Do not state or attribute any claim in it.

Return JSON only: {{"best": [<line number>, ...], "caption": "..."}}"""
    caption = f"{' and '.join(who)} in conversation" if who else ""
    picked = []
    if menu:
        try:
            obj = llm_json(prompt, system) or {}
            picked = [menu[k] for k in obj.get("best", [])
                      if isinstance(k, int) and 0 <= k < len(menu)]
            caption = str(obj.get("caption") or caption)[:280]
        except RuntimeError as e:
            log(f"  hook pick failed ({e}) - using the clip's own lines in order")
    # model's picks first, then the rest of the menu so every platform gets
    # its own hook; clean lines before profane ones (reach/ad risk on the
    # lead hook), otherwise keeping the model's order
    lines = list(dict.fromkeys(picked + menu))
    lines.sort(key=lambda l: bool(PROFANE.search(l)))
    hooks = [f'"{l}"' for l in lines[:len(PLATFORMS)]]
    if 0 < len(hooks) < len(PLATFORMS):
        log(f"  only {len(hooks)} distinct hook line(s) in this clip - platforms will share one")
    return {"hooks": hooks, "caption": caption} if hooks else None


# ------------------------------------------------------- auto selection ----

N_JUDGE = 14          # top-N find candidates that get a full editorial pass
SIGNAL_SLOTS = 3      # + this many more that carry audience signals (heat, reactions)
FLOOR_FINAL = 6.0     # never render below this, whatever the ranking says
MIN_OUT_S = 15.0      # a looped (payoff-first) clip shorter than this plays straight
TOPIC_DUP = 0.75      # full-text cosine at/above which two picks are one topic


def plan_clip(c, words, niche, context="", names=None, reacts=()):
    """Second, editorial pass over one candidate with its real text in view.
    find scores a moment from where it *starts*; this pass sees the whole
    span (plus some runway) and decides: keep or not, the real in/out
    points, and the payoff line to cold-open and loop on."""
    lo, hi = c["start"] - 5, c["end"] + 15
    sw = [w for w in words if lo <= w["s"] < hi]
    if len(sw) < 20:
        return {"keep": False, "final": 0, "reason": f"only {len(sw)} words - nothing to judge"}
    k0 = next((i for i, w in enumerate(sw) if w["s"] >= c["start"]), 0)
    k1 = next((i for i, w in enumerate(sw) if w["s"] >= c["end"]), len(sw))
    lines, prev, used = [], None, set()
    for i, w in enumerate(sw):
        if w.get("spk") and w["spk"] != prev:
            lines.append(f"\n{(names or {}).get(w['spk'], w['spk'])}:")
            prev = w["spk"]
        lines.append(f"[{i}]{w['w']}")
        lines += tags_after(w["e"], reacts, used, lo=-0.3, hi=2.0)
    lines = " ".join(lines)
    heard = sorted({t for t in lines if t.startswith("(")})
    system = ("You are a ruthless short-form video editor. You return JSON only. "
              "No preamble, no commentary.")
    prompt = f"""{context}Numbered words from a {niche} video, formatted [index]word. The
current proposed clip runs from word {k0} to word {k1 - 1}; words outside that
range are runway you may extend into or trim away. "Name:" marks who is speaking.
{"(laughter)/(applause) mark reactions detected in the audio - real evidence of a payoff, usually right after it." if heard else ""}
{lines}

Judge it for a cold audience scrolling TikTok/Reels/Shorts, then plan the cut.
Be harsh: most candidates are NOT worth posting. Scores: 5 = an ordinary
moment, 8 = better than 9 in 10 clips you have seen, 10 = rare.
- keep: false if it needs earlier context, rambles without a point, or has no payoff
- start_idx: first word of the clip - the start of a sentence that sets up the
  moment immediately. Cut throat-clearing and filler lead-ins.
- end_idx: last word of the clip - the end of a complete sentence right after
  the punchline. No trailing ramble. Aim for {CLIP_MIN}-{CLIP_MAX} seconds.
- payoff_start_idx, payoff_end_idx: the single most gripping line (punchline,
  shocking claim, or emotional peak), 3-12 words, inside the clip, not in its
  first 15 seconds, ideally near the end. It will be shown first as a cold
  open and the clip will loop back into it.
- standalone: 1-10, makes complete sense with zero prior context
- virality: 1-10, would a stranger stop, share, or comment
- reason: one sentence on why it works (or doesn't)

Return a JSON object only:
{{"keep": true, "start_idx": 0, "end_idx": 0, "payoff_start_idx": 0,
  "payoff_end_idx": 0, "standalone": 0, "virality": 0, "reason": ""}}"""

    try:
        obj = llm_json(prompt, system)
    except RuntimeError as e:
        log(f"    plan failed: {e}")
        return None
    if not obj:
        return None

    def idx(key, default):
        try:
            k = int(obj.get(key))
        except (TypeError, ValueError):
            return default
        return k if 0 <= k < len(sw) else default

    def num(key):
        try:
            return max(0.0, min(10.0, float(obj.get(key, 5))))
        except (TypeError, ValueError):
            return 5.0

    si, ei = idx("start_idx", k0), idx("end_idx", k1 - 1)
    s, e = sw[si]["s"], sw[ei]["e"]
    if si >= ei or not (CLIP_MIN - 7 <= e - s <= CLIP_MAX + 5):
        s, e = c["start"], c["end"]           # LLM's cut is unusable - keep find's

    loop, payoff = None, ""
    pi, pj = idx("payoff_start_idx", -1), idx("payoff_end_idx", -1)
    if 0 <= pi <= pj:
        ps, pe = sw[pi]["s"], sw[pj]["e"]
        if s + 15 <= ps and pe <= e + 0.05 and 0.8 <= pe - ps <= 6:
            loop = [ps, pe]
            payoff = " ".join(w["w"] for w in sw[pi:pj + 1])

    standalone, virality = num("standalone"), num("virality")
    return {"keep": obj.get("keep") is not False, "start": s, "end": e, "loop": loop,
            "payoff": payoff, "standalone": standalone, "virality": virality,
            "final": round(0.3 * float(c["score"]) + 0.35 * standalone + 0.35 * virality, 2),
            "reason": str(obj.get("reason", ""))[:200]}


def snap_span(words, s, e, loop):
    """Snap planned in/out points (and the loop payoff) onto sentence or
    speaker-turn boundaries - the judge's word picks sometimes land
    mid-sentence ("religion on a child, right?")."""
    import bisect
    # a speaker change counts only with a pause: one still mid-sentence after
    # label_speakers' snapping is a diarization miss, not a real turn
    # ...and a 0.5s+ pause is a boundary on its own: Whisper leaves long
    # crosstalk stretches unpunctuated, with no sentence start to snap to.
    gap = lambda i: words[i + 1]["s"] - words[i]["e"]
    brk = lambda i: re.search(r"[.?!]$", words[i]["w"]) or gap(i) > 0.5 or (
        words[i].get("spk") != words[i + 1].get("spk") and gap(i) > 0.3)
    st = [words[i]["s"] for i in range(len(words)) if i == 0 or brk(i - 1)]
    en = [words[i]["e"] for i in range(len(words)) if i == len(words) - 1 or brk(i)]

    def back(t, lim):                    # latest sentence start <= t
        k = bisect.bisect_right(st, t + 1e-3) - 1
        return st[k] if k >= 0 and t - st[k] <= lim else t

    def fwd(t, lim):                     # earliest sentence end >= t
        k = bisect.bisect_left(en, t - 1e-3)
        return en[k] if k < len(en) and en[k] - t <= lim else t

    s2, e2 = back(s, 12), fwd(e, 5)
    if s2 == s:
        # Nothing within 12s. Reach further back for the setup - on a
        # debate show the answer ("stay the f*** home") needs the question
        # asked 19s earlier - while the clip still fits the platforms (75s).
        far = back(s, 25)
        if far != s and e2 - far <= 75:
            s2 = far
        else:                            # last resort: any speaker change
            ch = [words[i]["s"] for i in range(1, len(words))
                  if words[i].get("spk") != words[i - 1].get("spk")]
            k = bisect.bisect_right(ch, s + 1e-3) - 1
            s2 = ch[k] if k >= 0 and s - ch[k] <= 12 else s
    if loop:
        ps, pe = back(loop[0], 4), fwd(loop[1], 3)
        loop = [ps, pe] if 0.8 <= pe - ps <= 7 and s2 + 12 <= ps and pe <= e2 + 0.05 else None
    return s2, e2, loop


def cmd_select(args):
    """Replaces the interactive review: judge the top candidates, pick the
    best non-overlapping ones, and apply each one's planned cut."""
    wd = workdir(args.source)
    cpath = wd / "candidates.json"
    if not cpath.exists():
        sys.exit("No candidates. Run: clipper.py find <source>")
    cands = json.loads(cpath.read_text())
    words = json.loads((wd / "transcript.json").read_text())["words"]
    niche = resolve_niche(args.niche, words, wd)
    ctx = video_context(wd)
    bounds = chapter_bounds(load_meta(wd), words)
    reacts = reactions(wd)
    names = speaker_names(words, wd)

    by_score = sorted(cands, key=lambda c: -float(c["score"]))
    judged = by_score[:N_JUDGE]
    # reserved slots: moments the audience marked (replay heat, laughter,
    # applause) that ranked just below the cut - two such moments on the
    # Rogan-Vance run never reached judging
    judged += [c for c in by_score[N_JUDGE:] if c.get("signals") or c.get("source") == "heat"][:SIGNAL_SLOTS]
    for c in judged:
        if "plan" in c:
            continue
        log(f"judging #{c['id']} ({c['end'] - c['start']:.0f}s, find score {c['score']})")
        c["plan"] = plan_clip(c, words, niche, ctx, names, reacts) or {"keep": False, "final": 0, "reason": "plan failed"}
        p = c["plan"]
        log(f"    {'KEEP' if p['keep'] else 'drop'} final={p['final']} "
            f"loop={'yes' if p.get('loop') else 'no'} - {p['reason']}")
        cpath.write_text(json.dumps(cands, indent=1))    # checkpoint per judgement

    # A comparative ranking pass (all candidates side by side, two orders)
    # was tried here and measured WORSE than these per-clip scores against a
    # careful manual ranking (Spearman +0.39 vs +0.46, top-4 overlap 1 vs 2),
    # and blends didn't beat either - so the per-clip score orders.
    ranked = sorted(judged, key=lambda c: (not c["plan"].get("keep", True), -c["plan"].get("final", 0)))
    # Same-topic check on the full clip text: find's dedup only compares a
    # one-line hook + reason, which let two "religion in schools" clips
    # through (full-text cosine 0.80; unrelated clips sit around 0.54).
    vec = {}
    try:
        judged_ok = [c for c in ranked if "start" in c["plan"]]
        texts = [" ".join(w["w"] for w in words if c["plan"]["start"] <= w["s"] < c["plan"]["end"])[:900]
                 or "." for c in judged_ok]
        vec = {c["id"]: v for c, v in zip(judged_ok, embed_texts(texts))}
    except Exception as e:
        log(f"  topic dedup skipped (embedding call failed: {e})")
    picked = []
    for c in ranked:
        p = c["plan"]
        if len(picked) >= args.max_clips:
            break
        # never a clip the judge dropped or one below the floor (it once forced
        # three 5.6s from a small-talk stream)
        if not p.get("keep", True) or p.get("final", 0) < FLOOR_FINAL:
            continue
        if "start" not in p:
            continue
        # Adjacent/overlapping spans are usually one story split in two -
        # posting both is posting the same clip twice.
        if any(p["start"] < q["plan"]["end"] + 5 and q["plan"]["start"] < p["end"] + 5
               for q in picked):
            continue
        twin = next((q for q in picked if c["id"] in vec and q["id"] in vec
                     and _cosine(vec[c["id"]], vec[q["id"]]) >= TOPIC_DUP), None)
        if twin:
            log(f"  skipping #{c['id']}: same topic as #{twin['id']}")
            continue
        picked.append(c)

    for c in cands:
        c["selected"] = c in picked
    for c in picked:
        p = c["plan"]
        s, e, loop = snap_span(words, p["start"], p["end"], p["loop"])
        s, e = clamp_to_chapters(s, e, bounds)
        if loop and not (s <= loop[0] and loop[1] <= e):
            loop = None                        # payoff fell outside this chapter
        if [c["start"], c["end"]] != [s, e] or c.get("loop") != loop:
            c.setdefault("orig", [c["start"], c["end"]])
            c["start"], c["end"], c["loop"] = s, e, loop
            c.pop("hooks", None)                          # span changed - rewrite hooks
        payoff = " ".join(w["w"] for w in words if loop and loop[0] <= w["s"] < loop[1])
        c["why"] = p["reason"] + (f' Payoff line: "{payoff}"' if payoff else "")
    cpath.write_text(json.dumps(cands, indent=1))
    if not picked:
        log(f"no postable moments: every judged candidate scored below {FLOOR_FINAL}")
    log(f"selected {len(picked)}: " + ", ".join(
        f"#{c['id']} ({c['plan']['final']})" for c in picked))


# ------------------------------------------------------------- render ------

def cmd_render(args):
    src = Path(args.source)
    wait_for_prefetch(src)
    wd = workdir(src)
    cpath = wd / "candidates.json"
    cands = json.loads(cpath.read_text())
    words = json.loads((wd / "transcript.json").read_text())["words"]
    picks = [c for c in cands if c.get("selected")]
    if args.pick:
        ids = {int(x) for x in args.pick.split(",")}
        picks = [c for c in cands if c["id"] in ids]
    if not picks:
        sys.exit("Nothing selected. Run: clipper.py select <source> (or review)")
    platforms = [p.strip() for p in args.platform.split(",")] if args.platform else [""]
    bad = [p for p in platforms if p and p not in PLATFORMS]
    if bad:
        sys.exit(f"unknown platform(s) {bad}; choose from {PLATFORMS}")

    # Write hooks once per candidate (cached on the candidate itself so a
    # re-render, or a render with a different --platform, doesn't re-call
    # the LLM) rather than reusing the raw transcript line.
    ctx = video_context(wd)
    names = speaker_names(words, wd)
    for c in picks:
        if not c.get("hooks") or c.get("hooks_v") != 10:
            log(f"  writing hooks for #{c['id']}...")
            loop = c.get("loop")
            payoff = " ".join(w["w"] for w in words if loop and loop[0] <= w["s"] < loop[1])
            gen = generate_hooks([w for w in words if c["start"] <= w["s"] < c["end"]],
                                 c.get("why", ""), ctx, names, payoff)
            if not gen:                          # never keep hooks from an older method
                c.pop("hooks", None)
            if gen:
                c["hooks"], c["caption"], c["hooks_v"] = gen["hooks"], gen["caption"], 10
                c["hook"] = gen["hooks"][0]
                cpath.write_text(json.dumps(cands, indent=1))

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(src)],
        capture_output=True, text=True, check=True).stdout
    streams = json.loads(probe).get("streams")
    if not streams:
        sys.exit(f"{src} has no video stream (audio-only pull?) - render the full video file instead.")
    st = streams[0]
    sw, sh = int(st["width"]), int(st["height"])
    fps_str = st.get("r_frame_rate", "30/1")
    num_, den_ = (float(x) for x in fps_str.split("/"))
    fps = num_ / den_ if den_ else 30.0
    sfx = sfx_files() if not args.no_sfx else {}

    OUT.mkdir(parents=True, exist_ok=True)
    outdir = OUT / src.stem
    outdir.mkdir(exist_ok=True)

    for c in picks:
        start = c["start"]
        loop = None if args.no_loop else c.get("loop")
        span_end = max(c["end"], loop[1] if loop else 0)
        a0 = max(0.0, start - 0.5)                  # room for the edge search before the first word
        audio = load_audio(wd / "audio.wav", a0, span_end - a0 + 0.8)
        clip_words = [w for w in words if start <= w["s"] < span_end]
        blocks, out_words, total, bounds, seam = build_timeline(
            words, start, c["end"], loop, cuts=not args.no_cuts, fps=fps,
            quiet=quiet_fn(audio, a0, clip_words), edge=edge_fn(audio, a0))
        if seam and total < MIN_OUT_S:
            # the loop drops everything after the payoff; with silence cuts
            # that left 13s clips - play it straight instead
            loop = None
            blocks, out_words, total, bounds, seam = build_timeline(
                words, start, c["end"], None, cuts=not args.no_cuts, fps=fps,
                quiet=quiet_fn(audio, a0, clip_words), edge=edge_fn(audio, a0))
        dur_in = max(bs + spans[-1][1] for bs, spans in blocks) - start + 0.2
        log(f"clip #{c['id']}  {start:.1f}s: {total:.1f}s out of {c['end'] - start:.0f}s source, "
            f"{len(bounds)} cut(s)" + (", cold-open loop" if seam else ""))

        frames = [] if args.center else sample_faces(src, start, dur_in)
        cuts = [] if args.center else scene_cuts(src, start, dur_in)
        speaking_scores(src, wd / "audio.wav", start, dur_in, frames, cuts)
        segs, qa = plan_layout(frames, 5, cuts, speaker_turns(words, start, start + dur_in),
                               dur_in, sw, sh, center=args.center)
        modes = {m: sum(1 for s_ in segs if s_[2] == m) for m in ("R", "S", "F")}
        framing = (f"{len(cuts) + 1} shot(s); crop {modes['R']}, stack {modes['S']}, fit {modes['F']}"
                   + f"; talker found {qa['talker']:.0%} of the time"
                   + (f"; speaker on screen {qa['on_screen']:.0%}" if qa["on_screen"] is not None else "")
                   + (f"; {qa['cut_faces']} face-edge frames" if qa["cut_faces"] else "")
                   + (f"; voices matched to faces {len(qa['voices'])}/{len({s_ for *_, s_ in speaker_turns(words, start, start + dur_in)})}"
                      if qa.get("voices") is not None else ""))
        log(f"    framing: {framing}")

        pulses = [] if args.no_zoom else emphasis_times(audio, a0, out_words, total)
        toggles, last = [], 0.0
        for b in bounds:                  # don't flicker framing on rapid-fire cuts
            if b - last >= 1.5:
                toggles.append(b)
                last = b
        events = ([("whoosh", max(0.0, seam - 0.32))] if seam and sfx else []) \
            + ([("pop", p) for p in pulses] if sfx else [])

        # One seeked input per piece (a stretch of the edit with one layout)
        # rather than split+trim over one input: split buffers every decoded
        # frame until concat asks for it - GBs of RAM on a 50s 1080p clip.
        # Pieces are what let a clip mix layouts (crop, stack, fit): an
        # ffmpeg stream can't change its framing mode mid-way.
        inputs, fc = [], []
        pieces = layout_pieces(blocks, segs, start, fps)
        for i, (seek, spans, seg) in enumerate(pieces):
            inputs += ["-ss", f"{seek:.3f}", "-t", f"{spans[-1][1] + 0.3:.3f}", "-i", str(src)]
            fc.append(piece_filter(i, spans, seg, fps))
        n = len(pieces)
        fc.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
        zoom = "" if args.no_zoom else (
            f"zoompan=z='{zoom_expr(toggles, pulses)}':d=1:x='iw/2-iw/zoom/2':"
            f"y='(ih-ih/zoom)*0.35':s={W}x{H}:fps={fps_str},")

        amix = "[ac]"
        for k, (kind, t) in enumerate(events):
            inputs += ["-i", str(sfx[kind])]
            ms = int(t * 1000)
            fc.append(f"[{n + k}:a]adelay={ms}|{ms},volume={SFX_VOL[kind]}[s{k}]")
        if events:
            fc.append("[ac]" + "".join(f"[s{k}]" for k in range(len(events)))
                      + f"amix=inputs={len(events) + 1}:normalize=0:duration=first[am]")
            amix = "[am]"
        fc.append(f"{amix}loudnorm=I=-14:TP=-1.5:LRA=11,aformat=sample_rates=48000[a]")

        for plat in platforms:
            hooks = c.get("hooks") or [c.get("hook", "")]
            hook_text = hooks[min(PLATFORMS.index(plat) if plat else 0, len(hooks) - 1)]
            stem = f"{src.stem}_{c['id']:02d}" + (f"_{plat}" if plat else "")
            tmp = Path(tempfile.mkdtemp(prefix="cliprender_"))
            try:
                assf = tmp / "c.ass"
                build_ass(out_words, total, hook_text, assf)
                graph = ";".join(fc)
                vfinal = f"[vc]{zoom}ass='{assf}'[v]"
                dest = outdir / f"{stem}.mp4"
                cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs,
                       "-filter_complex", graph + ";" + vfinal,
                       "-map", "[v]", "-map", "[a]", *VCODEC,
                       "-c:a", "aac", "-b:a", "192k", "-shortest",   # drop the loudnorm/AAC tail
                       "-movflags", "+faststart", str(dest)]
                r = subprocess.run(cmd, capture_output=True, text=True)
                # VideoToolbox sometimes drops frames - with a -12912 error,
                # or with none at all - and ffmpeg still exits 0. Check the
                # result itself, and redo in software if frames are missing.
                short = missing_frames(dest, fps) if r.returncode == 0 else None
                if r.returncode != 0 or "Error encoding" in r.stderr or short:
                    log(f"    hardware encode bad ({short or r.stderr.strip()[:80]} frames missing"
                        f" or error) - redoing with libx264")
                    i = cmd.index(VCODEC[0])
                    cmd[i:i + len(VCODEC)] = SW_VCODEC
                    run(cmd)
                (outdir / f"{stem}.txt").write_text(
                    f"hook (on-screen): {hook_text}\n"
                    f"caption (post text): {c.get('caption', '')}\n"
                    f"why: {c.get('why', '')}\n"
                    f"edit: {total:.1f}s, {len(bounds)} cuts, {len(pulses)} zoom pulses"
                    + (", cold-open loop" if seam else "") + "\n"
                    f"framing: {framing}\n"
                    f"source: {src.name} @ {start:.1f}-{c['end']:.1f}\n")
                log(f"    -> {dest}")
                log_clip({"date": time.strftime("%Y-%m-%d"), "source": src.stem, "clip_id": c["id"],
                          "platform": plat or "default", "file": dest.name, "duration": f"{total:.1f}",
                          "origin": c.get("source", ""), "signals": "+".join(c.get("signals") or []),
                          "score": c.get("score", ""), "final": (c.get("plan") or {}).get("final", ""),
                          "loop": int(bool(seam)), "layout": max(modes, key=modes.get),
                          "talker": f"{qa['talker']:.2f}", "hook_variant": hooks.index(hook_text),
                          "hook": hook_text, "caption": c.get("caption", "")})
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    log(f"\n  done: {outdir}")


# ------------------------------------------------------------ learning -----

CLIP_LOG = ROOT / "logs" / "clips.csv"
AUTO_COLS = ["date", "source", "clip_id", "platform", "file", "duration", "origin", "signals",
             "score", "final", "loop", "layout", "talker", "hook_variant", "hook", "caption"]
MANUAL_COLS = ["posted_url", "views_24h", "views_7d", "likes", "approved", "reject_reason"]


def log_clip(row, path=None):
    """Upsert one rendered clip into the log, keyed by source/clip/platform.
    A re-render refreshes what the pipeline knows and never touches the
    columns you fill in by hand (url, views, ...)."""
    import csv
    path = path or CLIP_LOG
    path.parent.mkdir(exist_ok=True)
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    key = lambda r: (r.get("source"), str(r.get("clip_id")), r.get("platform"))  # noqa: E731
    row = {k: str(v) for k, v in row.items()}
    for r in rows:
        if key(r) == key(row):
            r.update(row)
            break
    else:
        rows.append(row)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, AUTO_COLS + MANUAL_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)                       # never leave a half-written log


def views_of(r):
    for k in ("views_7d", "views_24h"):
        try:
            return float(r.get(k, "").replace(",", ""))
        except ValueError:
            pass
    return None


def clip_stats(rows):
    """Median views per value of each pipeline choice, plus whether the
    judge's score ranks clips the way the audience did."""
    rows = [(r, v) for r in rows if (v := views_of(r)) is not None]
    out = {"n": len(rows), "groups": {}, "corr": {}}
    for col in ("platform", "origin", "loop", "layout", "hook_variant", "signals"):
        g = {}
        for r, v in rows:
            g.setdefault(r.get(col) or "-", []).append(v)
        out["groups"][col] = {k: (len(vs), statistics.median(vs)) for k, vs in g.items()}
    for col in ("final", "score", "talker", "duration"):
        pairs = [(float(r[col]), v) for r, v in rows if r.get(col) not in (None, "")]
        if len(pairs) >= 5 and len({p[0] for p in pairs}) > 1 and len({p[1] for p in pairs}) > 1:
            out["corr"][col] = statistics.correlation(*zip(*pairs), method="ranked")
    return out


def cmd_stats(args):
    import csv
    if not CLIP_LOG.exists():
        sys.exit(f"no {CLIP_LOG} yet - render something first")
    s = clip_stats(list(csv.DictReader(CLIP_LOG.open())))
    if not s["n"]:
        sys.exit(f"no views logged yet - fill views_24h / views_7d in {CLIP_LOG}")
    print(f"{s['n']} clips with views (views_7d, else views_24h)\n")
    for col, g in s["groups"].items():
        print(f"{col}:")
        for k, (n, med) in sorted(g.items(), key=lambda x: -x[1][1]):
            print(f"  {k:24s} n={n:<4d} median views {med:,.0f}")
    print("\nrank correlation with views (+1 = predicts perfectly, 0 = noise):")
    for col, r in s["corr"].items():
        print(f"  {col:10s} {r:+.2f}")
    if s["n"] < 30:
        print(f"\n  only {s['n']} clips - treat every number above as a hint, not a finding")


# ------------------------------------------------------------------ go -----

AUDIO_EXTS = (".m4a", ".webm", ".opus", ".mp3", ".wav")


def free_whisper():
    """On 24GB, LM Studio holds the LLM while this process holds Whisper.
    Drop Whisper's cached model before the LLM stages or both compete."""
    try:
        import gc
        import mlx.core as mx
        t = sys.modules.get("mlx_whisper.transcribe")
        if t:
            t.ModelHolder.model = None
        gc.collect()
        mx.clear_cache()
        log("released Whisper GPU memory")
    except Exception:
        pass


def video_context(wd):
    """'Video: "<title>" (<channel>)' - who the speakers are, which S1/S2
    labels can't tell the LLM. Written by `go` from the source URL."""
    m = wd / "meta.json"
    if not m.exists():
        return ""
    meta = json.loads(m.read_text())
    return f'Video: "{meta.get("title", "")}" ({meta.get("channel", "")})\n'


def short_side(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return min(int(x) for x in out.split(",")[:2])
    except ValueError:
        return 0


SPONSOR_CATS = ["sponsor", "selfpromo", "interaction", "intro", "outro"]


def sponsor_segments(vid):
    """[[start, end, category], ...] from SponsorBlock's crowd-sourced
    database: ad reads, self-promo, subscribe reminders, intros/outros.
    None of it is clip material, and it explains fake heatmap peaks (people
    replaying the spot right after an ad they skipped)."""
    if not vid:
        return []
    url = ("https://sponsor.ajay.app/api/skipSegments?"
           + urllib.parse.urlencode({"videoID": vid, "categories": json.dumps(SPONSOR_CATS)}))
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code != 404:                      # 404 = nobody marked this video
            log(f"  SponsorBlock lookup failed ({e})")
        return []
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        log(f"  SponsorBlock lookup failed ({e})")
        return []
    return sorted([round(x["segment"][0], 1), round(x["segment"][1], 1), x["category"]]
                  for x in data)


def fetch_meta(url, path):
    """One yt-dlp -J call: title/channel/description confirm speaker names;
    chapters, heatmap (YouTube "most replayed"), category and tags are kept
    for selection signals."""
    r = subprocess.run(["yt-dlp", "--skip-download", "-J", url], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        log(f"  metadata fetch failed: {r.stderr.strip()[-200:]}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "sponsor": sponsor_segments(d.get("id", "")),
        "title": d.get("title", ""), "channel": d.get("channel", ""), "url": url,
        "description": (d.get("description") or "")[:5000],
        "categories": d.get("categories") or [], "tags": (d.get("tags") or [])[:40],
        "duration": d.get("duration"), "chapters": d.get("chapters") or [],
        "heatmap": d.get("heatmap") or []}, indent=1))


def cmd_go(args):
    """Link (or local file) in, finished platform clips out - no questions."""
    target = args.source
    if re.match(r"https?://", target) and not shutil.which("deno"):
        log("warning: deno not installed - YouTube throttles yt-dlp without a JS "
            "runtime (measured ~60 KB/s). brew install deno")
    if re.match(r"https?://", target):
        vid = subprocess.run(["yt-dlp", "--get-id", target], capture_output=True,
                             text=True, check=True).stdout.strip().splitlines()[-1]
        video = SOURCES / f"{vid}.mp4"
        if video.exists() and short_side(video) < 720:
            # pulled by the old height<=1080 format cap, which turned vertical
            # sources into 360x640 - set it aside and fetch it again
            video.rename(video.with_suffix(".lowres.mp4"))
            log(f"existing video is {short_side(video.with_suffix('.lowres.mp4'))}p - re-downloading "
                f"(old copy kept as {video.stem}.lowres.mp4)")
        meta = WORK / vid / "meta.json"
        if not meta.exists() or "sponsor" not in json.loads(meta.read_text()):
            fetch_meta(target, meta)
        find_audio = lambda: next((p for p in sorted(SOURCES.glob(f"{vid}.*"))
                                   if p.suffix in AUDIO_EXTS), None)
        if not find_audio() and not video.exists():
            cmd_ingest(argparse.Namespace(url=target, audio_only=True, prefetch=True,
                                          max_height=1080))
        elif not video.exists():
            start_prefetch(target, vid)
        audio = find_audio() or video
    else:
        audio = video = Path(target)
        if not video.exists():
            sys.exit(f"{video} not found")
        if video.suffix in AUDIO_EXTS and video.with_suffix(".mp4").exists():
            video = video.with_suffix(".mp4")

    ns = argparse.Namespace(**vars(args))
    ns.source = str(audio)
    cmd_transcribe(ns)
    free_whisper()

    cpath = workdir(audio) / "candidates.json"
    wd = workdir(audio)
    # reuse candidates only if newer than every input (transcript, audio
    # events, metadata) and made by the current find - otherwise new
    # signals silently never apply
    inputs = [p for p in (wd / "transcript.json", wd / "events.json", wd / "meta.json") if p.exists()]
    stamp = wd / "find.version"
    fresh = (cpath.exists() and not args.force
             and all(cpath.stat().st_mtime > p.stat().st_mtime for p in inputs)
             and stamp.exists() and stamp.read_text().strip() == FIND_VERSION)
    if fresh:
        log(f"candidates exist, skipping find ({cpath})")
    else:
        cmd_find(ns)
    cmd_select(ns)
    if not any(c.get("selected") for c in json.loads(cpath.read_text())):
        log("nothing to render")
        return
    ns.source = str(video)
    ns.pick = ""
    cmd_render(ns)


# --------------------------------------------------------------- cli -------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ingest")
    sp.add_argument("url")
    sp.add_argument("--audio-only", action="store_true",
                     help="~50MB audio only, for triage before committing to full video")
    sp.add_argument("--max-height", type=int, default=1080,
                     help="cap source video resolution (default 1080, plenty for a 1080x1920 crop)")
    sp.add_argument("--prefetch", action="store_true",
                     help="with --audio-only: also start the full video downloading in the "
                          "background now, so it overlaps with transcribe/find instead of "
                          "waiting until render")

    for name in ("go", "all", "transcribe", "find", "select", "review", "render"):
        sp = sub.add_parser(name)
        sp.add_argument("source", help="URL (go) or source file")
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--niche", default="auto", choices=["auto", *CRITERIA])
        sp.add_argument("--pick", default="", help="render only these ids, e.g. 1,4,9")
        sp.add_argument("--center", action="store_true", help="skip face tracking")
        sp.add_argument("--platform",
                         default=",".join(PLATFORMS[1:] + PLATFORMS[:1]) if name in ("go", "all") else "",
                         help="comma list of " + ",".join(PLATFORMS) + " - one render each, "
                              "each with its own hook (default for render: shorts hook, no suffix)")
        sp.add_argument("--max-clips", type=int, default=6)
        sp.add_argument("--no-cuts", action="store_true", help="keep silences and filler words")
        sp.add_argument("--no-loop", action="store_true", help="no cold-open loop")
        sp.add_argument("--no-zoom", action="store_true", help="no jump-cut / emphasis zooms")
        sp.add_argument("--no-sfx", action="store_true", help="no whoosh / pop")
    sub.add_parser("stats", help="which pipeline choices earned views (reads logs/clips.csv)")
    a = p.parse_args()
    # Long runs otherwise stall for 15+ min at a time when the Mac idles
    # into sleep (seen on battery). -w: released the moment we exit.
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    # ffmpeg treats stdin as live commands, so terminal or batch input got
    # injected into renders ("Enter command: ... Parse error"). Every child
    # inherits fd 0; only `review` actually reads the keyboard.
    if a.cmd != "review":
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    {"ingest": cmd_ingest, "go": cmd_go, "all": cmd_go, "transcribe": cmd_transcribe,
     "find": cmd_find, "select": cmd_select, "review": cmd_review,
     "render": cmd_render, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
