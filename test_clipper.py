"""python3 test_clipper.py - checks the edit timeline maths (no LLM, no video)."""
from clipper import build_timeline


def w(t, s, e):
    return {"w": t, "s": s, "e": e}


words = [w("Hello", 10.0, 10.4), w("there", 10.45, 10.8),        # 0.05 gap: kept
         w("um", 11.0, 11.3),                                     # filler: cut
         w("big", 11.4, 11.7), w("news", 11.75, 12.0),
         w("today.", 13.5, 13.9),                                 # 1.5s silence: cut
         w("Payoff", 20.0, 20.5), w("line!", 20.55, 21.0)]

blocks, out, total, bounds, seam = build_timeline(words, 10.0, 21.0, fps=30.0)
assert [x["w"] for x in out] == ["Hello", "there", "big", "news", "today.", "Payoff", "line!"], out
assert len(bounds) == 3, bounds                    # filler, 1.5s gap, gap before payoff
assert total < 11.0 - 1.4, total                   # silences actually removed
assert all(abs(a * 30 - round(a * 30)) < 1e-6 for _, sp in blocks for a, _ in sp)  # frame grid
assert all(0 <= x["s"] < x["e"] <= total + 1e-6 for x in out)

blocks, out, total, bounds, seam = build_timeline(words, 10.0, 21.0, loop=(20.0, 21.0), fps=30.0)
assert [x["w"] for x in out][:2] == ["Payoff", "line!"], out   # cold open first
assert [x["w"] for x in out][-1] == "today.", out              # ends right before payoff
assert blocks[0][0] == 20.0 and blocks[1][0] == 10.0
assert seam and abs(seam - out[2]["s"]) < 0.2, (seam, out[2])

blocks, out, total, *_ = build_timeline(words, 10.0, 21.0, cuts=False, fps=30.0)
assert abs(total - 11.0) < 1 / 30 and len(out) == len(words)
print("ok")

# a finished-but-unreaped child (zombie) must not count as running
import subprocess, time
from clipper import _pid_running
p = subprocess.Popen(["true"]); time.sleep(0.3)
assert _pid_running(p.pid) is False
print("ok pid")

# a loud gap (music/crowd) between words must survive the cuts
blocks, out, total, bounds, _ = build_timeline(words, 10.0, 21.0, fps=30.0, quiet=lambda a, b: False)
assert len(bounds) == 1, bounds          # only the filler still forces a cut
print("ok quiet")

# speaker labels: word -> overlapping (or nearest) turn, named in order of appearance
from clipper import label_speakers, phrases, exchange_times
ws = [w("Wait,", 0.0, 0.3), w("really?", 0.35, 0.7), w("Oh", 0.9, 1.0), w("yeah!", 1.0, 1.3), w("Wow.", 2.0, 2.2)]
label_speakers(ws, [{"s": 0.0, "e": 0.75, "spk": "SPEAKER_07"}, {"s": 0.85, "e": 1.35, "spk": "SPEAKER_02"},
                    {"s": 2.4, "e": 3.0, "spk": "SPEAKER_07"}])
assert [x["spk"] for x in ws] == ["S1", "S1", "S2", "S2", "S1"], ws   # "Wow." has no overlap -> nearest
# a phrase breaks at every speaker change, even with no punctuation
assert phrases(ws) == [(0, 1), (2, 3), (4, 4)], phrases(ws)
# no punctuation, one speaker: split every 25 words
run_on = [dict(w("x", i * 0.2, i * 0.2 + 0.15), spk="S1") for i in range(60)]
assert [b - a + 1 for a, b in phrases(run_on)] == [25, 25, 10]
# rapid turn-taking is found as an exchange
talk = [dict(w("x", i * 0.5, i * 0.5 + 0.4), spk="S1") for i in range(400)]
for i in range(200, 240):
    talk[i]["spk"] = "S2" if i % 4 < 2 else "S1"
ex = exchange_times(talk)
assert ex and 90 <= ex[0][1] <= 120, ex
print("ok speakers")

# a mid-sentence change snaps to the sentence end; a 1-word flicker is folded back
ws = [dict(w(t, i, i + 1.0)) for i, t in enumerate(
    ["I", "had", "two", "brain", "surgeries.", "He", "had", "two", "brain", "surgeries", "to", "repair"])]
label_speakers(ws, [{"s": 0, "e": 3.9, "spk": "A"}, {"s": 3.9, "e": 8.9, "spk": "B"},
                    {"s": 8.9, "e": 9.6, "spk": "A"}, {"s": 9.6, "e": 12, "spk": "B"}])
assert [x["spk"] for x in ws] == ["S1"] * 5 + ["S2"] * 7, [x["spk"] for x in ws]
print("ok snap")

# planned cut points snap to sentence starts / ends
from clipper import snap_span
ws = [dict(w(t, i, i + 0.9)) for i, t in enumerate("A b c. Force religion on kids. D e f.".split())]
s, e, _ = snap_span(ws, ws[4]["s"], ws[5]["e"], None)          # picked "religion on"
assert (s, e) == (ws[3]["s"], ws[6]["e"]), (s, e)             # -> "Force ... kids."
print("ok snap span")

# out-point on the next speaker's first word after a 7 s title card: end the answer, don't run on
ti = [dict(w("So", 19.2, 19.3), spk="G"), dict(w("it", 19.5, 19.7), spk="G"), dict(w("depends.", 19.7, 20.4), spk="G"),
      dict(w("You", 27.7, 28.2), spk="I"), dict(w("haven't", 28.2, 28.4), spk="I"), dict(w("shared?", 28.4, 30.2), spk="I")]
assert snap_span(ti, 19.2, 27.7, None)[1] == 20.4, snap_span(ti, 19.2, 27.7, None)
assert snap_span(ti, 19.2, 28.3, None)[1] == 30.2                # mid-sentence: still finishes it
for x in ti[3:]:
    x["spk"] = "G"                                               # same speaker after the pause: include it
assert snap_span(ti, 19.2, 27.7, None)[1] == 30.2
print("ok snap end")

# hook menu: whole clauses of 3-12 words, split at punctuation and speaker changes
from clipper import clauses
ws = [dict(w(t, i, i + 0.5), spk=("S1" if i < 8 else "S2")) for i, t in enumerate(
    "Wait, Biden had two brain surgeries? Oh yeah! I didn't know that".split())]
assert clauses(ws) == ["Wait, Biden had two brain surgeries?"], clauses(ws)   # unpunctuated 4-word tail = cut off
long = [dict(w(t, i, i + .5), spk="S1") for i, t in enumerate(
    "So what he is talking about, is that these very wealthy guys, are trying to fund Christian schools.".split())]
assert clauses(long) == [], clauses(long)                                  # too long as a whole sentence
assert clauses(long, pieces=True) == ["What he is talking about"], clauses(long, pieces=True)  # opener trimmed, mid pieces dropped
print("ok clauses")

# no boundary within 12s: reach back to the setup question if the clip still fits 75s
q = [dict(w(t, i * 0.25, i * 0.25 + 0.25), spk="A") for i, t in enumerate(
    ("So should she just stop? " + "blah " * 70 + "none of these are exclusive").split())]
s, e, _ = snap_span(q, q[-4]["s"], q[-1]["e"], None)
assert s == q[5]["s"], (s, q[5]["s"])   # latest sentence start, 17.5s back
print("ok setup")

# repetition-loop detector: a Whisper loop is flagged, ordinary speech is not
from clipper import loop_spans
speech = [w(t, i, i + 0.3) for i, t in enumerate(("so what we found is that people who start early " * 5).split())]
loop = [w("thank you.", 100 + i, 100.3 + i) for i in range(40)]
assert loop_spans(speech) == [], loop_spans(speech)
sp = loop_spans(speech + loop); assert len(sp) == 1 and sp[0][0] >= 40 and sp[0][1] >= 139, sp   # covers the loop (+ a few edge words)
print("ok loops")

# speaker names: only confirmed by title/description; no placeholders; duplicates go to the bigger speaker
from clipper import verify_names
hay = "joe rogan experience #2281 - elon musk powerfuljre"
assert verify_names({"S4": "Elon Musk", "S3": "Joe Rogan", "S5": "Joe Rogan"}, ["S4", "S3", "S5"], hay) == \
    {"S4": "Elon Musk", "S3": "Joe Rogan"}
assert verify_names({"S1": "Chad", "S2": "Unidentified Guest", "S3": "Rufus Wainwright"}, ["S1", "S2", "S3"], hay) == {}
assert verify_names({"S1": "Ranveer Singh (The Ranveer Show)"}, ["S1"], "kareena kapoor on trs with ranveer allahbadia") == {}
print("ok names")

# hook lines: filler openers trimmed, "Next up" dropped, profanity ranked last
from clipper import tidy, PROFANE
assert tidy("So, what difference would it make?") == "What difference would it make?"
ws = [dict(w(t, i, i + .4), spk="S1") for i, t in enumerate(
    "Next up, we have Hand Mirror. Let me read the quote. What the fuck did that mean? I opened the door to a loose bottle of ranch.".split())]
assert clauses(ws) == ["What the fuck did that mean?", "I opened the door to a loose bottle of ranch."], clauses(ws)
assert sorted(clauses(ws), key=lambda l: bool(PROFANE.search(l)))[0].startswith("I opened")
print("ok hook lines")

# broken JSON from the model gets one retry with feedback; an honest [] does not
import clipper
_real = clipper.llm_retry
calls, replies = [], iter(['Sure! Here you go: {"niche": "podcast"', '{"niche": "podcast"}'])
clipper.llm_retry = lambda p, s, tries=3: (calls.append(p), next(replies))[1]
assert clipper.llm_json("classify", "sys") == {"niche": "podcast"} and "could not be parsed" in calls[1]
replies = iter(["[]"])
assert clipper.llm_json("find", "sys", want=list) == []
clipper.llm_retry = _real
print("ok json retry")

# punctuation repair: finds long unpunctuated runs, splits them at pauses
from clipper import unpunctuated_spans, pause_pieces
runon = [w("word", i * 0.4, i * 0.4 + 0.3) for i in range(1600)]            # 640s, no full stop
runon[800] = w("word", 320.0 + 1.5, 320.3 + 1.5)                            # a clear pause near 320s
for x in runon[801:]: x["s"] += 1.5; x["e"] += 1.5
sp = unpunctuated_spans([w("Hi.", -1, -0.5)] + runon)
assert len(sp) == 1 and sp[0][0] == 0.0, sp
pieces = pause_pieces(runon, sp[0][0], sp[0][1])
assert len(pieces) == 2 and 319 < pieces[0][1] < 322, pieces                # cut in the pause, not mid-word
print("ok punct repair")

# a confirmed name stuck on a tiny label moves to the big unnamed one (Joe read Grok's lines aloud)
assert verify_names({"S4": "Elon Musk", "S3": "Grok AI", "S5": "Joe Rogan"}, ["S4", "S3", "S5"], hay,
                    {"S4": 557, "S3": 377, "S5": 58}) == {"S4": "Elon Musk", "S3": "Joe Rogan"}
print("ok name share")

# metadata signals: chapters trim, sponsor reads dropped, ad-skip heat ignored, hot regions boosted/added
from clipper import clamp_to_chapters, usable_heat, apply_signals
assert clamp_to_chapters(250.0, 310.0, [270.0]) == (270.0, 310.0)            # longest single-chapter piece
assert clamp_to_chapters(250.0, 310.0, [252.0]) == (250.0, 310.0)            # boundary within 3s of an edge: ignored
peaks = {0: 1.0, 1100: 1.0, 2640: 0.9, 6710: 0.85}                             # intro, post-ad, 2 real peaks
heat = [{"start_time": t, "end_time": t + 110, "value": peaks.get(t, 0.2 + (t % 7) / 100)}
        for t in range(0, 11000, 110)]
meta = {"heatmap": heat, "sponsor": [[1074, 1152, "sponsor"]], "chapters": []}
kept = {b["start_time"] for b in usable_heat(meta)}
assert 0 not in kept and 1100 not in kept and 2640 in kept                     # intro + post-ad spike removed
words = [dict(w("Hi.", t, t + 0.3)) for t in range(0, 7000, 2)]
cands = [{"start": 1080, "end": 1130, "score": 8, "why": "", "source": "llm"},      # inside the ad read
         {"start": 6690, "end": 6740, "score": 6, "why": "", "source": "llm"}]      # in a hot region
out = apply_signals(cands, meta, words)
assert all(c["start"] != 1080 for c in out)                                      # ad read dropped
hot = next(c for c in out if c["start"] == 6690)
assert hot["score"] > 6.5 and "replay heat" in hot["why"]                        # boosted
assert any(c["source"] == "heat" and 2600 <= c["start"] <= 2760 for c in out)   # uncovered hot region added
print("ok signals")

# reactions: relative thresholds find comedy-club laughter peaking ~0.34; tags attach once; resonance bonus
import tempfile, json as _json
from pathlib import Path as _P
from clipper import reactions, tags_after, resonate
d = _P(tempfile.mkdtemp())
lau = [0.02] * 300; lau[229] = lau[230] = 0.34; lau[140] = 0.30     # two laughs, well under 0.5
app = [0.01] * 300; app[72] = app[73] = 0.69
(d / "events.json").write_text(_json.dumps({"step": 1.0, "laughter": lau, "applause": app, "music": [0] * 300}))
r = reactions(d)
assert r == [(72.0, "applause"), (140.0, "laughter"), (229.0, "laughter")], r
used = set()
assert tags_after(228.5, r, used) == ["(laughter)"] and tags_after(228.9, r, used) == []   # tagged once
c = [{"start": 200, "end": 240, "score": 7.0, "why": "", "source": "exchange"}]
resonate(c, r, [])
assert c[0]["score"] == 7.5 and c[0]["signals"] == ["laughter", "exchange"], c      # +0.2 reaction +0.3 resonance
print("ok reactions")

# edges: outer cuts move to the quiet point; a block may start before its first word, never mid-array
import numpy as _np
from clipper import edge_fn
a = _np.ones(16000 * 30, dtype=_np.float32) * 0.2
a[int(9.8 * 16000):int(9.84 * 16000)] = 0.0                       # silence 0.2s before the first word (10.0)
a[int(21.25 * 16000):int(21.29 * 16000)] = 0.0                    # silence just after the last word
e = edge_fn(a, 0.0)
assert abs(e(10.0, -0.3, 0.02) - 9.82) < 0.02
words2 = [w("Hello", 10.0, 10.4), w("there.", 10.45, 21.0)]
blocks, out, total, _, _ = build_timeline(words2, 10.0, 21.0, fps=30.0, edge=e)
bs, spans = blocks[0]
assert bs <= 9.84 and spans[0][0] >= 0 and abs(bs + spans[-1][1] - 21.27) < 0.05, (bs, spans)
print("ok edges")

# a short line cut off by the next speaker is not a hook
cut = [dict(w(t, i, i + .4), spk=("S1" if i < 5 else "S2")) for i, t in enumerate(
    "It's a bit of a Typically we can get requested.".split())]
assert clauses(cut) == ["Typically we can get requested."], clauses(cut)
print("ok cut-off lines")

# chapter marks snap to the real speaker handover (marked 281s, next comedian at 277.7s)
from clipper import chapter_bounds
sw_ = [dict(w("x", t, t + .3), spk=("A" if t < 277.7 else "B")) for t in [270 + i * 0.5 for i in range(15)] + [277.7, 278.2]]
assert chapter_bounds({"chapters": [{"start_time": 0}, {"start_time": 281}]}, sw_) == [277.7]
assert clamp_to_chapters(228.0, 283.0, [277.7]) == (228.0, 277.7)
print("ok chapter snap")

# explicit terms rank a hook last; ordinary words that contain them do not
from clipper import PROFANE
assert PROFANE.search("Or tickle my clit, you know?") and PROFANE.search("What the fuck did that mean?")
assert not any(PROFANE.search(x) for x in ["Cocktail hour at the classic analysis.", "A cumulative title."])
print("ok explicit")

# framing plan: fit when faceless, facecam stack, dialogue stack, locked crop per speaker turn
from clipper import plan_layout, layout_pieces, speaker_turns
F = lambda cx, cy=0.4, w_=0.12, gap=None: {"cx": cx, "cy": cy, "w": w_, "h": w_, "gap": gap}
talk = lambda i: 0.02 + 0.03 * (i % 2)                               # a moving mouth
# 1) no faces -> fit;  2) facecam bottom-left -> stack
segs, _ = plan_layout([[]] * 20, 5, [], [], 4.0, 1920, 1080)
assert [s[2] for s in segs] == ["F"]
segs, _ = plan_layout([[F(0.12, 0.75, 0.1, 0.03)]] * 20, 5, [], [], 4.0, 1920, 1080)
assert segs[0][2] == "S" and segs[0][3][0][0] < 400                  # top panel = the facecam, left side
# 3) two people far apart trading turns -> stack; 4) one speaker -> one locked full-height crop
two = [[F(0.2, gap=talk(i) if i < 25 else 0.02), F(0.8, gap=talk(i) if i >= 25 else 0.02)] for i in range(50)]
segs, qa = plan_layout(two, 5, [], [(0, 5, "A"), (5, 10, "B")], 10.0, 1920, 1080)
assert [s[2] for s in segs] == ["S"] and qa["on_screen"] == 1.0
one = [[F(0.3, gap=talk(i)), F(0.8, gap=0.02)] for i in range(50)]
segs, qa = plan_layout(one, 5, [], [(0, 10, "A")], 10.0, 1920, 1080)
assert len(segs) == 1 and segs[0][2] == "R" and segs[0][3][0][3] == 1080 and qa["on_screen"] == 1.0
x, _, rw, _ = segs[0][3][0]
assert x <= 0.3 * 1920 <= x + rw                                     # the talker is in the crop
# 5) a camera cut splits shots; small faces (panel) get a zoomed crop no tighter than 0.3 W
cut = [[F(0.3, gap=talk(i))] if i < 25 else [F(0.5, w_=0.05, gap=talk(i))] for i in range(50)]
segs, _ = plan_layout(cut, 5, [5.0], [], 10.0, 1920, 1080)
assert len(segs) == 2 and segs[1][3][0][2] >= 1080 / 1.9 - 2 and segs[1][3][0][3] < 1080   # zoom capped at 1.9x
# pieces: split at segment bounds on the frame grid; each piece keeps one layout
blocks = [(100.0, [(0.0, 4.0), (5.0, 9.0)])]
pcs = layout_pieces(blocks, [(0, 6.0, "R", [(0, 0, 608, 1080)]), (6.0, 20, "F", [])], 100.0, 30.0)
assert [p[2][2] for p in pcs] == ["R", "F"] and abs(sum(b - a for _, sp, _ in pcs for a, b in sp) - 8.0) < 1e-6
assert pcs[1][0] == 106.0 and pcs[0][1] == [(0.0, 4.0), (5.0, 6.0)]
print("ok framing")

# facecam next to a static HUD portrait at the same edge (Jynxzi/R6): still a stack on the talking face
hud = [[F(0.05, 0.3, 0.08, None), F(0.18 + 0.02 * (i % 3), 0.75, 0.12, talk(i))] for i in range(40)]
segs, _ = plan_layout(hud, 5, [], [], 8.0, 1920, 1080)
assert segs[0][2] == "S", segs
cx_top = segs[0][3][0][0] + segs[0][3][0][2] / 2
assert abs(cx_top - 0.2 * 1920) < 120, cx_top                      # top panel centred on the talker, not the portrait
print("ok facecam+hud")
game = segs[0][3][1]
assert game[0] >= min((0.2 + 1.7 * 0.12) * 1920, 1920 - game[2]) - 2, game   # as far from the webcam as the frame allows
print("ok game panel")

# split-screen grid (5 boxes): the crop is one box wide, not half the neighbour too
grid = [[F(0.1 + 0.2 * k, 0.4, 0.05, talk(i) if k == 2 else 0.02) for k in range(5)] for i in range(30)]
segs, _ = plan_layout(grid, 5, [], [], 6.0, 1920, 1080)
x, y, rw, rh = segs[0][3][0]
assert rw <= 1080 / 2.8 + 2 and x <= 0.5 * 1920 <= x + rw, segs[0]      # one box (<= 2.8x upscale)
print("ok grid box")


# persistence: 2 stray detections in a faceless 20-frame shot (fire B-roll) don't block "fit"
stray = [[] for _ in range(20)]; stray[3] = [F(0.4, 0.4, 0.25)]; stray[11] = [F(0.6, 0.5, 0.2)]
assert plan_layout(stray, 5, [], [], 4.0, 1920, 1080)[0][0][2] == "F"
# neighbours 8% apart at a panel table stay separate tracks; the talker's crop lands on the talker
panel = [[F(0.42, 0.35, 0.05, 0.02), F(0.50, 0.36, 0.05, talk(i)), F(0.58, 0.35, 0.05, 0.02)] for i in range(40)]
segs, _ = plan_layout(panel, 5, [], [(0, 8, "A")], 8.0, 1920, 1080)
x, _, rw, _ = segs[0][3][0]
assert segs[0][2] == "R" and abs((x + rw / 2) - 0.50 * 1920) < 60, segs[0]
print("ok face tracks")

# voice->face: a neighbour laughing throughout (bigger mouth motion) must not steal a speaker's turn
laugh = lambda i: 0.02 + 0.06 * (i % 2)                          # always moving, big
fr_ = [[F(0.3, gap=(talk(i) if i < 20 else 0.02)),            # A talks only in the first 4 s
        F(0.7, gap=laugh(i)),                                  # neighbour laughs the whole time
        F(0.5, gap=(talk(i) if i >= 20 else 0.02))] for i in range(40)]  # B talks in the last 4 s
segs, _ = plan_layout(fr_, 5, [], [(0, 4, "A"), (4, 8, "B")], 8.0, 1920, 1080)
first = [s_ for s_ in segs if s_[0] < 2][0]
x, _, rw, _ = first[3][0] if first[2] == "R" else (0, 0, 0, 0)
assert first[2] == "S" or x <= 0.3 * 1920 <= x + rw, segs      # A's turn frames A, not the laugher
print("ok voice-face")

# 4K source: the zoom floor is in output pixels, so a medium shot gets a tight crop, not a near-full one
seg4k, _ = plan_layout([[F(0.42, 0.4, 0.068, talk(i))] for i in range(20)], 5, [], [], 4.0, 3840, 2160)
rw4k = seg4k[0][3][0][2]
assert rw4k < 0.25 * 3840 and abs(rw4k - 3.2 * 0.068 * 3840) < 4, rw4k                # face ~1/3 of the crop
print("ok 4k zoom")

# an off-camera host speaks first: the crop holds on the on-screen speaker, not a random neighbour
offcam = [[F(0.43, gap=(talk(i) if i >= 12 else 0.02)),        # woman talks after 2.4 s
           F(0.75, gap=0.02 + 0.006 * (i % 2)),                        # neighbour fidgets throughout
           F(0.17, gap=0.02)] for i in range(38)]
segs, _ = plan_layout(offcam, 5, [], [(0, 2.3, "HOST"), (2.5, 7.6, "WOMAN")], 7.6, 1920, 1080)
centres = [(g[3][0][0] + g[3][0][2] / 2) / 1920 for g in segs if g[2] == "R"]
assert segs and all(g[2] == "R" for g in segs) and all(abs(c - 0.43) < 0.05 for c in centres), segs
print("ok off-camera")

# clip log: a re-render refreshes pipeline columns but keeps the views typed in by hand
import csv as _csv, tempfile as _tf
from pathlib import Path
from clipper import log_clip, clip_stats
lp = Path(_tf.mkdtemp()) / "clips.csv"
log_clip({"source": "v", "clip_id": 1, "platform": "tiktok", "hook": "old", "final": 7}, lp)
rows = list(_csv.DictReader(lp.open()))
rows[0]["views_7d"] = "1,200"
with lp.open("w", newline="") as f:
    w = _csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
log_clip({"source": "v", "clip_id": 1, "platform": "tiktok", "hook": "new", "final": 8}, lp)
log_clip({"source": "v", "clip_id": 2, "platform": "tiktok", "hook": "x", "final": 6}, lp)
rows = list(_csv.DictReader(lp.open()))
assert len(rows) == 2 and rows[0]["hook"] == "new" and rows[0]["views_7d"] == "1,200", rows
st = clip_stats([{"platform": "a", "final": str(i), "views_24h": str(i * 10)} for i in range(6)]
                + [{"platform": "b", "final": "9"}])                     # no views: ignored
assert st["n"] == 6 and st["groups"]["platform"]["a"] == (6, 25.0) and st["corr"]["final"] > 0.99, st
print("ok clip log")

# multi-cam panel (Whatever): split screen host|guest, then a wide shot where the guest is a small
# face. The host's 2 s question must get the host; the guest's mic keeps her speaking score below 0
# even while she talks - she must still be linked (contrast), and found again in the wide shot (identity)
import numpy as _np
_e = {k: _np.eye(128, dtype=_np.float32)[k] for k in range(4)}
A = lambda cx, cy, w_, who, asd: dict(F(cx, cy, w_), emb=_e[who], asd=asd)
split = [[A(0.17, 0.3, 0.07, 0, (1.0 if i < 12 else -2.0)),          # host talks 0-2.3 s
          A(0.75, 0.4, 0.07, 1, (-0.8 if 13 <= i else -2.5)),         # guest, mic-muffled
          A(0.43, 0.3, 0.05, 2, -2.0)] for i in range(38)]            # neighbour
wide = [[A(0.14, 0.4, 0.03, 2, -2.0), A(0.52, 0.4, 0.03, 3, -1.5),    # pink, never talks here
         A(0.80, 0.4, 0.03, 1, (0.5 if 3 <= i else -1.5))] for i in range(30)]
segs, qa = plan_layout(split + wide, 5, [7.6], [(0, 2.3, "HOST"), (2.5, 7.6, "GUEST"), (8.2, 13.6, "GUEST")],
                       13.6, 3840, 2160)
assert qa["voices"] == {"HOST": 0, "GUEST": 1}, qa["voices"]
at = lambda t: next(g for g in segs if g[0] <= t < g[1])
cxs = lambda g: [(r[0] + r[2] / 2) / 3840 for r in g[3]]
assert any(abs(c - 0.17) < 0.06 for c in cxs(at(1.0))), segs        # host framed for his question
assert any(abs(c - 0.75) < 0.06 for c in cxs(at(5.0))), segs        # guest during her answer
assert any(abs(c - 0.80) < 0.06 for c in cxs(at(11.0))), segs       # same guest in the wide shot, not pink
print("ok multi-cam panel")

# a long video's sign-off/end card and "closing" chapters are off limits; its opening is not
from clipper import video_edges
ed = video_edges({"duration": 600, "chapters": [{"start_time": 0, "title": "Disclosing partner count"},
                                               {"start_time": 500, "title": "Closing remarks"}]})
assert ed == [(570.0, 600.0), (500, 600.0)], ed
assert video_edges({"duration": 200}) == []                          # short video: nothing
print("ok video edges")

# IRL: one long take, the talker walks across the frame - the crop pans with him (smoothly, ends
# held), no two-box stack; a seated talker still gets one locked crop
from clipper import path_x, path_expr
walk = [[F(0.2 + 0.6 * i / 49, 0.4, 0.06, talk(i)), F(0.85, 0.4, 0.06, 0.02)] for i in range(50)]
segs, _ = plan_layout(walk, 5, [], [(0, 10, "A")], 10.0, 1920, 1080)
assert len(segs) == 1 and segs[0][2] == "R" and len(segs[0]) == 5, segs
path, rw = segs[0][4], segs[0][3][0][2]
assert path_x(path, 1.0) + rw / 2 < path_x(path, 9.0) + rw / 2, path     # moves right with him
assert abs(path_x(path, 5.0) + rw / 2 - 0.5 * 1920) < 0.08 * 1920, path_x(path, 5.0)
assert all(abs(b[1] - a[1]) / (b[0] - a[0]) < 0.2 * 1920 for a, b in zip(path, path[1:])), path  # glides
assert path_expr([(0, 10), (2, 30)], 1.0) == "if(lt(t,-1.000),10,if(lt(t,1.000),10+(20)*(t--1.000)/2.000,30))"
still = [[F(0.42, 0.4, 0.06, talk(i))] for i in range(30)]
assert len(plan_layout(still, 5, [], [(0, 6, "A")], 6.0, 1920, 1080)[0][0]) == 4    # locked
print("ok follow")


# Claude judge: verdict returned, logged as a training example (model, prompt, reply); offline (API faked)
import json as _json, tempfile as _tf2
from pathlib import Path as _P
import clipper as _c
_c.JUDGE, _c.JUDGE_LOG = "claude", _P(_tf2.mkdtemp()) / "judge.jsonl"
_c.claude = lambda prompt, system, **k: ('Sure:\n{"keep": true, "standalone": 8, "virality": 7}', {"input_tokens": 9})
got = _c.judge_json("PROMPT", "SYS", tag="vid#3")
row = _json.loads(_c.JUDGE_LOG.read_text().splitlines()[0])
assert got["keep"] is True and got["_by"] == _c.JUDGE_MODEL, got
assert row["model"] == _c.JUDGE_MODEL and row["prompt"] == "PROMPT" and row["tag"] == "vid#3", row
_c.JUDGE = "local"
print("ok claude judge")

# judge training data: Claude rows only, latest per prompt, JSON-only answers, held out by VIDEO
import train_judge as _tj
_tj.LOG = _P(_tf2.mkdtemp()) / "j.jsonl"
with _tj.LOG.open("w") as f:
    for v in range(14):
        for k in range(3):
            f.write(_json.dumps({"tag": f"v{v}#{k}", "model": "claude-x", "system": "S", "prompt": f"p{v}-{k}",
                                 "reply": 'ok {"keep": true, "virality": 7}'}) + "\n")
    f.write(_json.dumps({"tag": "v0#0", "model": "local:qwen", "system": "S", "prompt": "q", "reply": "{}"}) + "\n")
    f.write(_json.dumps({"tag": "v0#0", "model": "claude-x", "system": "S", "prompt": "p0-0",
                         "reply": '{"keep": false}'}) + "\n")          # re-judged: latest wins
rows = _tj.teacher_rows()
assert len(rows) == 42 and next(r for r in rows if r["prompt"] == "p0-0")["answer"] == '{"keep": false}'
parts = _tj.split(rows)
vids = {k: {r["tag"].split("#")[0] for r in v} for k, v in parts.items()}
assert vids["test"] and vids["valid"] and not (vids["test"] & vids["train"]) and not (vids["valid"] & vids["train"]), vids
assert _tj.chat(rows[0])["messages"][1]["content"].startswith("/no_think ")
print("ok judge data")

# summarize: list/review titles detected; chapters become the structure (minus intro/outro);
# a judge pick that runs long is trimmed back to a sentence end near the budget
from clipper import video_kind, video_structure, pick_piece
_w = lambda t, a, b: {"w": t, "s": a, "e": b}
assert video_kind({"title": "Top 10 Open World Games of 2026"}) == "list"
assert video_kind({"title": "10 Movies That Changed Horror"}) == "list"
assert video_kind({"title": "Elden Ring Nightreign Review - Worth It?"}) == "review"
assert video_kind({"title": "Joe Rogan Experience #2281 - Elon Musk"}) is None
_wd = _P(_tf2.mkdtemp())
_ws = [_w("x.", i, i + 0.5) for i in range(400)]
chaps = [{"title": "Intro", "start_time": 0, "end_time": 30}] + \
        [{"title": f"#{5 - k} Game {k}", "start_time": 30 + 60 * k, "end_time": 90 + 60 * k} for k in range(5)] + \
        [{"title": "Outro", "start_time": 330, "end_time": 400}]
st = video_structure(_wd, _ws, {"chapters": chaps}, "list")
assert [x["label"] for x in st] == [f"#{5 - k} Game {k}" for k in range(5)] and st[0]["start"] == 30, st
assert video_structure(_wd, [], {}, "list") == st                       # cached
long = [_w(t, i * 0.4, i * 0.4 + 0.35) for i, t in enumerate(
    ("It is the best game. " * 2 + "The combat is amazing and deep. " + "blah " * 60).split())]
_orig = _c.judge_json
_c.judge_json = lambda *a, **k: {"start_idx": 0, "end_idx": len(long) - 1}
s_, e_ = pick_piece(long, {"label": "#1", "start": 0, "end": 99}, 4.0, "", [], "t")
assert s_ == 0 and e_ <= 2 * 4.0 + 2 and long[[x["e"] for x in long].index(e_)]["w"].endswith("."), (s_, e_)
_c.judge_json = _orig
print("ok summarize")
