"""train_judge.py - teach the local model to judge clips like Claude.

clipper.py logs every judge call to logs/judge_data.jsonl. Claude's verdicts
there are the training set; this fits a LoRA adapter on the local Qwen (MLX)
so `select` can run free and offline. The dataset is model-agnostic: when a
new Qwen ships, point --model at it and train again - nothing re-collected.

Runs in .venv-train (mlx-lm). Unloads LM Studio first: a 14B model training
next to one being served would not fit in 24 GB.

  .venv-train/bin/python train_judge.py            # build data + train
  .venv-train/bin/python train_judge.py --eval     # student vs Claude on held-out videos
"""
import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "logs" / "judge_data.jsonl"
DATA = ROOT / "work" / "judge-train"
BASE = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3-14B-MLX-4bit"
MIN_EXAMPLES = 150


def obj_of(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def teacher_rows():
    """Claude's verdicts, latest per prompt, with the reply reduced to its JSON."""
    rows = {}
    for line in LOG.open():
        r = json.loads(line)
        o = obj_of(r["reply"])
        # '#' tags are clip judgements (vid#id); structure/summary-piece calls differ in kind
        if r["model"].startswith("claude") and o and "#" in r.get("tag", ""):
            o.pop("_by", None)
            rows[r["prompt"]] = {**r, "answer": json.dumps(o, ensure_ascii=False)}
    return list(rows.values())


def split(rows):
    """Held out by VIDEO, not by row: clips of one video share a transcript,
    so a row-level split would test on near-copies of the training set."""
    vids = sorted({r["tag"].split("#")[0] for r in rows})
    random.Random(0).shuffle(vids)
    n = max(1, len(vids) // 7)
    test, valid = set(vids[:n]), set(vids[n:2 * n])
    part = lambda r: "test" if r["tag"].split("#")[0] in test else "valid" if r["tag"].split("#")[0] in valid else "train"
    return {k: [r for r in rows if part(r) == k] for k in ("train", "valid", "test")}


def chat(r):
    # same prompt the local judge gets (/no_think: Qwen3 otherwise reasons first)
    return {"messages": [{"role": "system", "content": r["system"]},
                         {"role": "user", "content": "/no_think " + r["prompt"]},
                         {"role": "assistant", "content": r["answer"]}]}


def unload_lm_studio():
    subprocess.run(["lms", "unload", "--all"], capture_output=True)


def train(args):
    rows = teacher_rows()
    if len(rows) < MIN_EXAMPLES:
        sys.exit(f"{len(rows)} Claude verdicts logged - need {MIN_EXAMPLES}+ before training "
                 f"(each video adds ~17; `clipper.py select <source>` with ANTHROPIC_API_KEY "
                 f"set re-judges a past video)")
    parts = split(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    for k, rs in parts.items():
        with (DATA / f"{k}.jsonl").open("w") as f:
            for r in rs:
                f.write(json.dumps(chat(r), ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in parts.items()}, "examples")
    out = ROOT / "models" / "judge-lora" / (Path(args.model).name + time.strftime("-%Y%m%d"))
    unload_lm_studio()
    subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "--model", str(args.model), "--train",
                    "--data", str(DATA), "--adapter-path", str(out), "--mask-prompt",
                    "--batch-size", "1", "--num-layers", "16", "--grad-checkpoint",
                    "--max-seq-length", "4096", "--learning-rate", "1e-5",
                    "--iters", str(args.epochs * len(parts["train"])),
                    "--steps-per-eval", "100", "--val-batches", "25", "--save-every", "100"], check=True)
    print(f"adapter: {out}\nnext: .venv-train/bin/python train_judge.py --eval --adapter {out}")


def score(o):
    try:
        return 0.35 * float(o.get("standalone", 5)) + 0.35 * float(o.get("virality", 5))
    except (TypeError, ValueError):
        return None


def evaluate(args):
    from mlx_lm import generate, load
    rows = split(teacher_rows())["test"]
    if not rows:
        sys.exit("no held-out videos yet")
    unload_lm_studio()
    for adapter in [None, args.adapter] if args.adapter else [None]:
        model, tok = load(str(args.model), adapter_path=adapter)
        pairs, keep_agree, bad = [], 0, 0
        for r in rows:
            msgs = chat(r)["messages"][:2]
            text = generate(model, tok, tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                            max_tokens=400)
            o, t = obj_of(text), json.loads(r["answer"])
            if not o or score(o) is None:
                bad += 1
                continue
            pairs.append((score(o), score(t)))
            keep_agree += (o.get("keep") is not False) == (t.get("keep") is not False)
        rho = statistics.correlation(*zip(*pairs), method="ranked") if len(pairs) > 2 else float("nan")
        print(f"{'adapter ' + Path(adapter).name if adapter else 'base model':30s} "
              f"rank agreement with Claude {rho:+.2f}   keep/drop agree {keep_agree}/{len(pairs)}   "
              f"unparseable {bad}/{len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(BASE), help="MLX model dir to train (a newer Qwen: point here)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--adapter", help="adapter dir to evaluate against the base model")
    a = ap.parse_args()
    evaluate(a) if a.eval else train(a)
