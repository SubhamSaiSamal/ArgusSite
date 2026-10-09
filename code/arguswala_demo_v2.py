# ===== CELL D / DEMO SNAPSHOT (current pipeline) -- run AFTER Cell 1 and Cell 2 =====
# Writes /kaggle/working/argus_demo_snapshot_v2.json, which site/index.html replays offline
# via "Load snapshot". No web server and no tunnel, so nothing here depends on the network.
#
# Same design as the July demo: real held-out White-bellied Sholakili calls (never used as
# the five examples) planted at known times into real Western Ghats soundscapes, one call per
# loudness level from +12 dB down to -3 dB, so the hard cases stay visible.
#
# One change, for honesty: the detection thresholds are chosen on CALIBRATION soundscapes and
# then applied unchanged to different soundscapes that are shown. That is the paper's protocol.
# The July demo picked its threshold on the very soundscapes it displayed, which flatters it.
import io, json, time, base64
import numpy as np, librosa, soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEMO_SPECIES = "whbsho3"
DEMO_SEC     = 120.0
DEMO_SNRS    = (12, 9, 6, 3, 0, -3)
N_SHOW, N_CALIB = 3, 3
OUT_JSON     = "/kaggle/working/argus_demo_snapshot_v2.json"

t_demo = time.time()
assert DEMO_SPECIES in TARGET_LIST, f"{DEMO_SPECIES} is not a pre-registered target"
demo_ctx = build_target(DEMO_SPECIES)
demo_enc = encoders[(FEATURE, ENCODER_SEEDS[0])]
print(f"[demo] {DEMO_SPECIES}: {len(demo_ctx['support'])} support views, "
      f"{len(demo_ctx['target_pool'])} held-out calls, decoys {demo_ctx['DECOYS']}")


def wav_b64(y, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(y, dtype="float32"), -1, 1), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def spec_b64(y, sr=SR):
    # No axes or padding: the image spans exactly 0 -> duration, so the page can place
    # markers by percentage and have them line up.
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
    keep = librosa.fft_frequencies(sr=sr, n_fft=1024) <= 11000
    fig = plt.figure(figsize=(16, 4.2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(S[keep], origin="lower", aspect="auto", cmap="magma", vmin=-70, vmax=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, pad_inches=0); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_scenario(bed, seed):
    r = np.random.default_rng(seed)
    n = int(DEMO_SEC * SR)
    off = int(r.integers(0, max(1, len(bed) - n)))
    y = bed[off:off + n].copy()
    pool = demo_ctx["target_pool"]
    spans, truth = [], []
    for snr in DEMO_SNRS:
        placed = None
        for _ in range(120):
            pos = int(r.integers(0, len(y) - CLEN))
            if any(pos < e + int(1.2 * SR) and pos + CLEN > s - int(1.2 * SR) for s, e in spans):
                continue
            placed = pos; break
        if placed is None:
            continue
        c = pool[int(r.integers(len(pool)))][:CLEN]
        if len(c) < CLEN: c = np.pad(c, (0, CLEN - len(c)))
        g = rms(y[placed:placed + CLEN]) * (10 ** (snr / 20)) / (rms(c) + 1e-9)
        y[placed:placed + CLEN] += c.astype("float32") * g
        spans.append((placed, placed + CLEN))
        truth.append({"start": placed / SR, "end": (placed + CLEN) / SR, "snr": int(snr)})
    peak = float(np.max(np.abs(y))) or 1.0
    if peak > 0.99: y = y * (0.99 / peak)
    return y.astype("float32"), sorted(truth, key=lambda t: t["start"])


def make(beds, base_seed, n, label):
    out = []
    for i in range(n):
        y, truth = build_scenario(beds[i % len(beds)], seed=base_seed + i)
        out.append({"id": i, "name": f"{label} #{i + 1}", "seconds": round(len(y) / SR, 1),
                    "audio": y, "truth": truth})
    return out


CALIB = make(calib_beds, 200, N_CALIB, "calibration soundscape")
SHOW = make(eval_beds, 100, N_SHOW, "Western Ghats soundscape")

for s in CALIB + SHOW:
    t0 = time.time()
    dets = detect_pv(demo_enc, demo_ctx, s["audio"], np.random.default_rng(0))
    # Round once, here, and compute every score from the rounded values -- the page does the
    # same arithmetic in JavaScript, so a borderline call cannot flip between the two.
    s["raw"] = [(float(a), float(b), round(float(p), 4), round(float(v), 4)) for a, b, p, v in dets]
    print(f"[demo] {s['name']}: {len(dets)} candidates in {time.time() - t0:.0f}s")

FUSE = {"p": lambda p, v: p, "v": lambda p, v: v, "pv": lambda p, v: p * v}


def match(events, truth):
    pairs = sorted(((iou((e[0], e[1]), (t["start"], t["end"])), i, j)
                    for i, e in enumerate(events) for j, t in enumerate(truth)), reverse=True)
    ue, ut = set(), set()
    for ov, i, j in pairs:
        if ov <= IOU_THR: break
        if i in ue or j in ut: continue
        ue.add(i); ut.add(j)
    return ue, ut


def tally(scenes, fn, thr):
    tp = fp = nt = 0
    for s in scenes:
        keep = [(a, b, fn(p, v)) for a, b, p, v in s["raw"] if fn(p, v) >= thr]
        ue, ut = match(keep, s["truth"])
        tp += len(ut); fp += len(keep) - len(ue); nt += len(s["truth"])
    return tp, fp, nt


def best_threshold(fn):
    scores = sorted({fn(p, v) for s in CALIB for _, _, p, v in s["raw"]})
    best, best_f1 = 1.0, -1.0
    for thr in scores:
        tp, fp, nt = tally(CALIB, fn, thr)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / nt if nt else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        if f1 > best_f1: best_f1, best = f1, thr
    return best, best_f1


THR = {}
for k, fn in FUSE.items():
    THR[k], f1 = best_threshold(fn)
    print(f"[demo] threshold[{k}] = {THR[k]:.4f}  (best F1 {f1:.3f}, chosen on CALIBRATION soundscapes only)")

print("[demo] results on the SHOWN soundscapes at the calibrated p x v threshold:")
for s in SHOW:
    tp, fp, nt = tally([s], FUSE["pv"], THR["pv"])
    print(f"   {s['name']}: {tp}/{nt} found, {fp} false alarms")
tp, fp, nt = tally(SHOW, FUSE["pv"], THR["pv"])
print(f"   ALL SHOWN: {tp}/{nt} found, {fp} false alarms")

sup = demo_ctx["support"]
one_per_shot = sup[1::3] if len(sup) >= 15 else sup[:5]   # the unshifted view of each recording
snapshot = {
    "target": DEMO_SPECIES,
    "support": {"count": len(one_per_shot), "clips": [wav_b64(w) for w in one_per_shot[:5]],
                "note": "The only labelled examples the model is given. Everything else is found unaided."},
    "analysis": None,
    "note": ("Current ARGUS pipeline (two-stage, Perch v2 verifier). Thresholds chosen on separate "
             "calibration soundscapes, then applied unchanged to the soundscapes shown."),
    "scenarios": [],
}
for s in SHOW:
    cands = []
    for a, b, p, v in s["raw"]:
        lo, hi = int(a * SR), int(b * SR)
        clip = s["audio"][max(0, lo):min(len(s["audio"]), hi)]
        if len(clip) < int(0.4 * SR): clip = np.pad(clip, (0, int(0.4 * SR) - len(clip)))
        cands.append({"start": round(a, 2), "end": round(b, 2), "p": p, "v": v, "audio": wav_b64(clip)})
    cands.sort(key=lambda c: c["start"])
    snapshot["scenarios"].append({"id": s["id"], "name": s["name"], "seconds": s["seconds"],
                                  "audio": wav_b64(s["audio"]), "spectrogram": spec_b64(s["audio"]),
                                  "candidates": cands, "truth": s["truth"], "thresholds": THR})
with open(OUT_JSON, "w") as fh:
    json.dump(snapshot, fh)
import os
print(f"[demo] wrote {OUT_JSON} ({os.path.getsize(OUT_JSON) / 1e6:.1f} MB) in {(time.time() - t_demo) / 60:.1f} min")
