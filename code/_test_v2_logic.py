# -*- coding: utf-8 -*-
"""Local unit tests for the NEW logic in the v2 pipeline.

Heavy deps (Perch/TF/torch/Kaggle data) are unavailable here, so the pure functions are
re-declared verbatim from the source and exercised directly. Point is to catch the
mistakes that would otherwise cost a Kaggle round-trip.
"""
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))   # the pipeline files sit next to this test

FAILS = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else "   <- " + detail))
    if not cond: FAILS.append(name)

def l2n(x):
    x = np.asarray(x, dtype="float32")
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

# ------------------------------------------------- choose_shots (verbatim from setup_v2)
def choose_shots(files, ratings, embs, n_shot, strategy, rng):
    n = len(files)
    if strategy == "rated":
        return list(np.argsort(-np.asarray(ratings, dtype=float))[:n_shot])
    if strategy == "first":
        return list(range(min(n_shot, n)))
    if strategy == "random":
        return list(rng.permutation(n)[:n_shot])
    if strategy == "diverse":
        E = l2n(embs)
        picked = [int(np.argmax(np.asarray(ratings, dtype=float)))]
        while len(picked) < min(n_shot, n):
            d = np.min(1.0 - E @ E[picked].T, axis=1)
            d[picked] = -np.inf
            picked.append(int(np.argmax(d)))
        return picked
    raise ValueError(strategy)

print("\n[choose_shots]")
files = [f"f{i}.ogg" for i in range(8)]
ratings = [1.0, 5.0, 3.0, 4.5, 2.0, 0.5, 4.0, 3.5]
rng = np.random.default_rng(0)

idx = choose_shots(files, ratings, None, 5, "first", rng)
check("first -> positional order", idx == [0, 1, 2, 3, 4], str(idx))

idx = choose_shots(files, ratings, None, 5, "rated", rng)
check("rated -> descending rating", [ratings[i] for i in idx] == sorted(ratings, reverse=True)[:5], str(idx))

idx = choose_shots(files, ratings, None, 5, "random", rng)
check("random -> 5 distinct in range", len(set(idx)) == 5 and all(0 <= i < 8 for i in idx), str(idx))

# diverse: build embeddings as 3 tight clusters. A spread-seeking picker must not take
# two from the same cluster before touching every cluster.
clusters = np.array([[1,0,0],[1,.02,0],[1,0,.02],   # cluster A
                     [0,1,0],[.02,1,0],             # cluster B
                     [0,0,1],[0,.02,1],[0,0,1.02]]) # cluster C
idx = choose_shots(files, ratings, clusters, 3, "diverse", rng)
lab = ["A","A","A","B","B","C","C","C"]
check("diverse -> covers all 3 clusters", len({lab[i] for i in idx}) == 3, f"{idx} -> {[lab[i] for i in idx]}")
check("diverse -> no duplicate picks", len(set(idx)) == len(idx), str(idx))
idx_b = choose_shots(files, ratings, clusters, 3, "diverse", np.random.default_rng(99))
check("diverse -> deterministic across rng", idx == idx_b, f"{idx} vs {idx_b}")

idx = choose_shots(files[:3], ratings[:3], clusters[:3], 5, "diverse", rng)
check("diverse -> handles n_shot > n_available", len(idx) == 3, str(idx))

# ------------------------------------------------- held-out pool excludes shots
print("\n[held-out pool]")
for strat in ("first", "rated", "random", "diverse"):
    sh = choose_shots(files, ratings, clusters, 5, strat, np.random.default_rng(1))
    shot_set = set(int(i) for i in sh)
    pool_idx = [i for i in range(len(files)) if i not in shot_set]
    ok = not (set(pool_idx) & shot_set) and len(pool_idx) == len(files) - len(shot_set)
    check(f"{strat}: pool disjoint from shots", ok, f"shots={sorted(shot_set)} pool={pool_idx}")
# the v1 bug this replaces: files[N_SHOT:] under a non-positional strategy
sh = choose_shots(files, ratings, clusters, 5, "rated", rng)
v1_pool = set(range(5, 8))
leaked = v1_pool & set(int(i) for i in sh)
check("v1 slice WOULD have leaked (regression guard)", len(leaked) > 0,
      "no leak under this fixture - test is not proving anything")

# ------------------------------------------------- summarise / thr_r90
SNR_LEVELS = (9, 6, 3, 0, -3, -6)
def summarise(scored, n_truth):
    if not scored or not n_truth:
        return dict(ap=0.0, f1=0.0, p=0.0, r=0.0, thr=1.0, thr_r90=1.0, n=len(scored))
    arr = sorted(scored, key=lambda x: -x[0])
    tp = fp = 0; ap = 0.0; prev_r = 0.0; best = (0.0, 0.0, 0.0, 1.0); thr_r90 = None
    for score, hit in arr:
        if hit: tp += 1
        else:   fp += 1
        pr = tp / (tp + fp); rc = tp / n_truth
        ap += pr * (rc - prev_r); prev_r = rc
        f1 = 2*pr*rc/(pr+rc) if pr + rc else 0.0
        if f1 > best[0]: best = (f1, pr, rc, score)
        if thr_r90 is None and rc >= 0.90: thr_r90 = score
    if thr_r90 is None: thr_r90 = arr[-1][0]
    return dict(ap=ap, f1=best[0], p=best[1], r=best[2], thr=best[3],
                thr_r90=thr_r90, n=len(scored))

def recall_at(scored, n_truth, thr):
    return sum(1 for s, hit in scored if s >= thr and hit) / n_truth

print("\n[summarise]")
perfect = [(0.9, True), (0.8, True), (0.7, True), (0.6, True)]
m = summarise(perfect, 4)
check("perfect separation -> F1 = 1.0", abs(m["f1"] - 1.0) < 1e-9, str(m["f1"]))

mixed = [(0.9, True), (0.85, False), (0.8, True), (0.7, True), (0.2, False)]
m = summarise(mixed, 3)
check("mixed -> 0 < F1 < 1", 0 < m["f1"] < 1, str(m["f1"]))
# The property that matters: thr_r90 must actually DELIVER >=90% recall. (The earlier
# assertion compared it positionally to the best-F1 threshold, which is not a real
# invariant -- when best-F1 already exceeds 90% recall, thr_r90 is correctly tighter.)
check("thr_r90 delivers >=90% recall", recall_at(mixed, 3, m["thr_r90"]) >= 0.90,
      f"thr_r90={m['thr_r90']} -> recall {recall_at(mixed, 3, m['thr_r90']):.2f}")
check("thr_r90 is the HIGHEST such threshold",
      all(recall_at(mixed, 3, s) < 0.90 for s, _ in mixed if s > m["thr_r90"]),
      f"a stricter threshold also reached 90%: thr_r90={m['thr_r90']}")

# recall never reaches 90% -> fall back to the loosest score, not a silent wrong answer
low = [(0.9, True), (0.8, False), (0.7, False), (0.6, False)]
m2 = summarise(low, 10)
check("unreachable 90% -> loosest threshold", m2["thr_r90"] == 0.6, str(m2["thr_r90"]))

# regression guard: the old rule picked the score just BEFORE the crossing
old_thr_r90 = None
tp = 0
for s, hit in sorted(mixed, key=lambda x: -x[0]):
    if hit: tp += 1
    if tp / 3 < 0.90: old_thr_r90 = s
check("old rule genuinely failed the recall floor", recall_at(mixed, 3, old_thr_r90) < 0.90,
      f"old={old_thr_r90} recall={recall_at(mixed, 3, old_thr_r90):.2f}")

check("empty input safe", summarise([], 0)["f1"] == 0.0)
check("no-truth safe", summarise(mixed, 0)["f1"] == 0.0)

# ------------------------------------------------- _auc (the headline metric)
def _auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg): return float('nan')
    gt = float((pos[:, None] > neg[None, :]).sum())
    eq = float((pos[:, None] == neg[None, :]).sum())
    return (gt + 0.5 * eq) / (len(pos) * len(neg))

print("\n[_auc]")
check("perfect separation -> 1.0", _auc([1, .9, .8], [.1, .2]) == 1.0)
check("inverted -> 0.0", _auc([.1, .2], [1, .9, .8]) == 0.0)
check("all ties -> 0.5", _auc([.5, .5], [.5, .5]) == 0.5)
check("interleaved -> 0.5", abs(_auc([1, 0], [.75, .25]) - 0.5) < 1e-9)
check("empty -> nan", np.isnan(_auc([], [1, 2])))
# ARGUS's actual finding: stage-1 below chance, verified above
check("below-chance is representable", _auc([.3, .4], [.5, .6, .7]) < 0.5)

# ------------------------------------------------- score_neg (false alarms / hour)
BED_SEC = 240.0
def score_neg(fuse, thr, cache):
    fa = sum(sum(1 for s, e, p, v in dets if float(fuse(p, v)) >= thr) for dets in cache)
    hours = (len(cache) * BED_SEC) / 3600.0
    return fa / hours if hours else float('nan')

print("\n[score_neg]")
fuse_p = lambda p, v: p
# 10 beds x 240 s = 40 min = 2/3 h. 6 detections above thr -> 9.0 / hour
cache = [[(0, 1, 0.9, 0.5)] for _ in range(6)] + [[] for _ in range(4)]
check("FA/hour arithmetic", abs(score_neg(fuse_p, 0.5, cache) - 9.0) < 1e-9,
      str(score_neg(fuse_p, 0.5, cache)))
check("threshold above all -> 0", score_neg(fuse_p, 0.99, cache) == 0.0)
check("empty cache -> nan", np.isnan(score_neg(fuse_p, 0.5, [])))
check("higher thr never increases FA",
      score_neg(fuse_p, 0.95, cache) <= score_neg(fuse_p, 0.5, cache))

# ------------------------------------------------- covariates: stereotypy
def stereotypy(embs):
    E = l2n(embs)
    if len(E) < 2: return float('nan')
    S = E @ E.T
    iu = np.triu_indices(len(E), k=1)
    return float(S[iu].mean())

print("\n[stereotypy]")
identical = np.repeat(np.array([[1.0, 0, 0]]), 4, axis=0)
check("identical calls -> ~1.0", abs(stereotypy(identical) - 1.0) < 1e-5, str(stereotypy(identical)))
orth = np.eye(3)
check("orthogonal calls -> ~0.0", abs(stereotypy(orth)) < 1e-5, str(stereotypy(orth)))
check("varied < identical", stereotypy(clusters) < stereotypy(identical))
check("single call -> nan", np.isnan(stereotypy(identical[:1])))

# ------------------------------------------------- encoder-bank exclusion (the bug)
print("\n[encoder bank exclusion -- the v1 bug]")
all_species = ["whbsho3", "spearl", "grewar", "malpar", "indrob"]
TARGET_SET = {"whbsho3", "spearl", "grewar"}
v1_bank = [s for s in all_species if s != "whbsho3"]          # v1 line 216
v2_bank = [s for s in all_species if s not in TARGET_SET]     # v2
check("v1 leaked other targets into bank", len(set(v1_bank) & TARGET_SET) == 2,
      str(set(v1_bank) & TARGET_SET))
check("v2 leaks nothing", len(set(v2_bank) & TARGET_SET) == 0, str(set(v2_bank) & TARGET_SET))
try:
    assert not (TARGET_SET & set(v2_bank))
    check("v2 assertion passes", True)
except AssertionError:
    check("v2 assertion passes", False)
try:
    assert not (TARGET_SET & set(v1_bank))
    check("assertion CATCHES the v1 bug", False, "assertion did not fire on known-bad bank")
except AssertionError:
    check("assertion CATCHES the v1 bug", True)

# ------------------------------------------------- Perch 2.0 handling (new)
PERCH_PEAK_NORM = 0.25
def _peak_norm(batch):
    if PERCH_PEAK_NORM is None: return batch
    b = np.asarray(batch, dtype="float32").copy()
    b -= np.mean(b, axis=-1, keepdims=True)
    pk = np.max(np.abs(b), axis=-1, keepdims=True)
    b = np.divide(b, pk, where=(pk > 0.0))
    return (b * PERCH_PEAK_NORM).astype("float32")

print("\n[_peak_norm]")
rng2 = np.random.default_rng(3)
x = rng2.normal(0, 3.0, (4, 1000)).astype("float32") + 7.0   # loud, big DC offset
y = _peak_norm(x)
check("peak scaled to 0.25", np.allclose(np.max(np.abs(y), axis=-1), 0.25, atol=1e-5),
      str(np.max(np.abs(y), axis=-1)))
check("DC offset removed", np.allclose(np.mean(y, axis=-1), 0.0, atol=1e-6),
      str(np.mean(y, axis=-1)))
check("all-zero window survives (no div-by-zero)",
      np.all(np.isfinite(_peak_norm(np.zeros((2, 100), dtype="float32")))))
check("per-window independence (row scaling differs)",
      not np.allclose(_peak_norm(np.array([[1.0, -1.0], [100.0, -100.0]]))[0],
                      np.array([0.0, 0.0])))

# THE LOAD-BEARING CLAIM: peak norm must not disturb planted-call SNR, or the whole
# SNR sweep becomes uninterpretable after the Perch 2.0 upgrade.
def rms_(a): return float(np.sqrt(np.mean(np.square(a, dtype=np.float64)) + 1e-12))
bed = rng2.normal(0, 0.05, 5000).astype("float32")
call = rng2.normal(0, 1.0, 5000).astype("float32")
for snr_db in (9, 6, 3, 0, -3, -6):
    g = rms_(bed) * (10 ** (snr_db / 20)) / (rms_(call) + 1e-9)
    mixed = bed + call * g
    before = 20 * np.log10(rms_(call * g) / rms_(bed))
    n = _peak_norm(mixed[np.newaxis, :])[0]
    scale = rms_(n) / rms_(mixed)            # the scalar the normaliser applied
    after = 20 * np.log10(rms_(call * g * scale) / rms_(bed * scale))
    check(f"SNR preserved at {snr_db:+d} dB", abs(before - after) < 1e-6,
          f"{before:.6f} -> {after:.6f}")

print("\n[_extract_embedding]")
class _T:                      # stand-in for a TF tensor
    def __init__(s, a): s.a = np.asarray(a)
    def numpy(s): return s.a
def _extract_embedding(out):
    if hasattr(out, "keys"):
        if "embedding" not in out:
            raise KeyError(f"no 'embedding' head in Perch output; got {sorted(out.keys())}")
        emb = out["embedding"]
    else:
        emb = out[1]
    return np.asarray(emb.numpy() if hasattr(emb, "numpy") else emb)

v2_out = {"embedding": _T(np.ones((2, 1536))), "spatial_embedding": _T(np.zeros((2, 5, 3, 1536))),
          "label": _T(np.zeros((2, 15000))), "frontend": _T(np.zeros((2, 8)))}
e = _extract_embedding(v2_out)
check("v2 dict -> pooled embedding, 1536-d", e.shape == (2, 1536), str(e.shape))
check("v2 spatial_embedding NOT returned", e.ndim == 2, str(e.shape))

v1_dict = {"embedding": _T(np.ones((2, 1280))), "label": _T(np.zeros((2, 10932)))}
check("v1 dict -> 1280-d", _extract_embedding(v1_dict).shape == (2, 1280))

v1_tuple = (_T(np.zeros((2, 10932))), _T(np.ones((2, 1280))))
check("v1 (logits, embeddings) tuple -> 1280-d", _extract_embedding(v1_tuple).shape == (2, 1280))

try:
    _extract_embedding({"label": _T(np.zeros((2, 10)))})
    check("missing embedding head raises", False, "no exception")
except KeyError:
    check("missing embedding head raises", True)

print("\n[version guard]")
for requested, dim, should_raise in (("v2", 1536, False), ("v1", 1280, False),
                                     ("v2", 1280, True), ("v1", 1536, True)):
    detected = {1280: "v1", 1536: "v2"}.get(dim, "unknown")
    raised = detected != requested
    check(f"requested {requested} + {dim}-d model -> {'reject' if should_raise else 'accept'}",
          raised == should_raise)

# ------------------------------------------------- ONNX backend (added 13 Aug 2026)
print("\n[ONNX backend wiring]")

def onnx_infer(sess_run, in_name, out_names, batch, peak_norm_fn):
    """Mirrors _infer()'s onnx branch: peak-norm, run, zip into a dict, extract."""
    batch = peak_norm_fn(batch)
    result = sess_run(None, {in_name: batch})
    out = dict(zip(out_names, result))
    return _extract_embedding(out)

def fake_peak_norm(b):
    b = np.asarray(b, dtype="float32").copy()
    b -= b.mean(-1, keepdims=True)
    pk = np.max(np.abs(b), -1, keepdims=True)
    return np.divide(b, pk, where=pk > 0).astype("float32") * 0.25

class FakeORTSession:
    """Stand-in for onnxruntime.InferenceSession, shaped exactly like the REAL Perch v2
    ONNX export verified against the actual downloaded weights on 13 Aug 2026:
    outputs ['embedding'(1536), 'spatial_embedding', 'spectrogram', 'label'(14795)]."""
    def run(self, _, feed):
        n = next(iter(feed.values())).shape[0]
        return [np.ones((n, 1536), "float32"), np.zeros((n, 16, 4, 1536), "float32"),
                np.zeros((n, 500, 128), "float32"), np.zeros((n, 14795), "float32")]

sess = FakeORTSession()
out_names = ["embedding", "spatial_embedding", "spectrogram", "label"]
batch = np.random.default_rng(0).normal(0, 3.0, (16, 160000)).astype("float32") + 7.0
emb = onnx_infer(sess.run, "inputs", out_names, batch, fake_peak_norm)
check("onnx branch -> correct shape (16, 1536)", emb.shape == (16, 1536), str(emb.shape))
check("onnx branch pulls 'embedding' not 'spatial_embedding'",
      emb.shape[1] == 1536 and emb.ndim == 2, str(emb.shape))

print("\n[_find_local_onnx pattern matching]")
def find_local_onnx(paths):
    hits = [p for p in paths if "perch_v2_no_dft" in p and p.endswith(".onnx")]
    if not hits:
        hits = [p for p in paths if "perch_v2" in p and p.endswith(".onnx")]
    return hits[0] if hits else None
check("prefers no_dft variant when both present",
      find_local_onnx(["/kaggle/input/x/perch_v2.onnx", "/kaggle/input/x/perch_v2_no_dft.onnx"])
      == "/kaggle/input/x/perch_v2_no_dft.onnx")
check("falls back to plain perch_v2.onnx",
      find_local_onnx(["/kaggle/input/x/perch_v2.onnx"]) == "/kaggle/input/x/perch_v2.onnx")
check("no match -> None", find_local_onnx(["/kaggle/input/x/other.onnx"]) is None)
check("ignores non-onnx files",
      find_local_onnx(["/kaggle/input/x/perch_v2_no_dft.txt"]) is None)

print("\n[PERCH_BACKEND provenance]")
def make_row(backend):
    return dict(perch_version="v2", perch_embed_dim=1536, perch_backend=backend)
r_tf = make_row("tf"); r_onnx = make_row("onnx")
check("tf backend recorded", r_tf["perch_backend"] == "tf")
check("onnx backend recorded", r_onnx["perch_backend"] == "onnx")
check("backend is part of provenance, not silently dropped", "perch_backend" in r_tf)

# ------------------------------------------------- SINGLE_TARGET_COMPARISON (14 Aug)
print("\n[SINGLE_TARGET_COMPARISON target-list logic]")
def build_target_list(primary, eligible, n_targets, smoke_test, single_target_comparison):
    if single_target_comparison:
        return [primary]
    tl = [primary] + [sp for sp in eligible if sp != primary][:n_targets - 1]
    if smoke_test:
        tl = tl[:2]
    return tl

eligible = ["bcnher", "whbsho3", "spearl", "grewar", "indrob"]   # value_counts order
tl_normal = build_target_list("whbsho3", eligible, 10, True, False)
check("normal smoke test -> 2 targets, primary first",
      tl_normal == ["whbsho3", "bcnher"], str(tl_normal))

tl_single = build_target_list("whbsho3", eligible, 10, True, True)
check("single-target comparison -> exactly [primary], ignores smoke_test",
      tl_single == ["whbsho3"], str(tl_single))

tl_single_full = build_target_list("whbsho3", eligible, 10, False, True)
check("single-target comparison -> ignores N_TARGETS too",
      tl_single_full == ["whbsho3"], str(tl_single_full))

set_normal = set(tl_normal); set_single = set(tl_single)
check("single-target set is a strict subset of the multi-target set",
      set_single < set_normal, f"single={set_single} normal={set_normal}")
check("single-target excludes exactly 1 species from the encoder bank",
      len(set_single) == 1, str(len(set_single)))
check("normal run excludes 2 -- this IS the composition difference under test",
      len(set_normal) == 2, str(len(set_normal)))

# ------------------------------- spec_coverage + the low-SNR AUC ambiguity (roadmap 3.2)
# Verbatim from arguswala_v2.py. Detection events are (start, end, p, v); planted items
# are (kind, (start, end)).
IOU_THR = 0.30

def iou(a, b):
    it = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1]-a[0]) + (b[1]-b[0]) - it
    return it/u if u > 0 else 0.0

def spec_coverage(cache):
    n = {"target": 0, "decoy": 0}; hit = {"target": 0, "decoy": 0}
    for dets, items in cache:
        for kind, span in items:
            k = "target" if kind == "target" else "decoy"
            n[k] += 1
            best = max((iou((s, e), span) for s, e, _p, _v in dets), default=0.0)
            hit[k] += int(best > IOU_THR)
    return {k: (hit[k]/n[k] if n[k] else float('nan')) for k in n}

def score_spec_auc(fuse, cache, decoy_names):
    """The AUC half of score_spec, behaviour-identical -- including the 0.0 it assigns to
    any planted item that was never localised."""
    tg, dc = [], {k: [] for k in decoy_names}
    for dets, items in cache:
        ev = [(s, e, float(fuse(p, v))) for s, e, p, v in dets]
        for kind, span in items:
            best = max((iou(e[:2], span), e[2]) for e in ev) if ev else (0.0, 0.0)
            sc = best[1] if best[0] > IOU_THR else 0.0
            (tg if kind == "target" else dc[kind]).append(sc)
    return _auc(tg, [s for v in dc.values() for s in v])

print("\n[spec_coverage / low-SNR ambiguity]")
V = lambda p, v: v

# Loud case: both planted calls localised, target scores above decoy.
loud = [([(0.0, 1.0, 0.9, 0.9), (2.0, 3.0, 0.2, 0.2)],
         [("target", (0.0, 1.0)), ("d1", (2.0, 3.0))])]
cov_loud, auc_loud = spec_coverage(loud), score_spec_auc(V, loud, ["d1"])
check("loud -> full coverage", cov_loud == {"target": 1.0, "decoy": 1.0}, str(cov_loud))
check("loud -> AUC 1.0 (real discrimination)", auc_loud == 1.0, str(auc_loud))

# Faint case: the ONLY detection is nowhere near either planted call, so both score 0.0.
faint = [([(50.0, 51.0, 0.9, 0.9)],
          [("target", (0.0, 1.0)), ("d1", (2.0, 3.0))])]
cov_faint, auc_faint = spec_coverage(faint), score_spec_auc(V, faint, ["d1"])
check("faint -> zero coverage", cov_faint == {"target": 0.0, "decoy": 0.0}, str(cov_faint))
# THE TRAP THIS FUNCTION EXISTS FOR: identical AUC-to-chance from a totally different cause.
check("faint -> AUC 0.5 DESPITE nothing being detected (the ambiguity is real)",
      auc_faint == 0.5, str(auc_faint))
check("AUC alone cannot separate 'indistinguishable' from 'undetected'",
      auc_faint == 0.5 and _auc([0.5, 0.5], [0.5, 0.5]) == 0.5)
check("coverage DOES separate them", cov_loud["target"] != cov_faint["target"])

# Coverage must not depend on the fusion rule -- a rule changes scores, never spans.
covs = [spec_coverage(loud) for _ in range(3)]
auc_p = score_spec_auc(lambda p, v: p, loud, ["d1"])
check("coverage is fusion-independent (spans, not scores)",
      all(c == covs[0] for c in covs) and auc_p == auc_loud)

# Partial coverage, and target/decoy counted independently.
part = [([(0.0, 1.0, 0.9, 0.9)],
         [("target", (0.0, 1.0)), ("target", (10.0, 11.0)),
          ("d1", (2.0, 3.0)), ("d2", (4.0, 5.0))])]
cov_part = spec_coverage(part)
check("partial -> target 50%, decoy 0% (counted separately)",
      cov_part == {"target": 0.5, "decoy": 0.0}, str(cov_part))

check("no planted items -> nan, not a crash or a fake 0.0",
      all(np.isnan(v) for v in spec_coverage([([], [])]).values()))
check("no detections at all -> zero coverage, no exception",
      spec_coverage([([], [("target", (0.0, 1.0))])])["target"] == 0.0)

# Level keys are ints, so max()/min() must pick the true SNR endpoints regardless of the
# order SNR_SWEEP_LEVELS happens to be written in.
for order in ((9, 6, 3, 0, -3, -6), (-6, -3, 0, 3, 6, 9), (0, 9, -6, 3)):
    c = {int(s): None for s in order}
    check(f"endpoint selection order-independent {order[:2]}...",
          max(c) == max(order) and min(c) == min(order), f"{max(c)},{min(c)}")

def score_spec_detected(fuse, cache):
    """Verbatim from arguswala_v2.py."""
    tg, dc = [], []
    for dets, items in cache:
        ev = [(s, e, float(fuse(p, v))) for s, e, p, v in dets]
        for kind, span in items:
            best = max((iou(e[:2], span), e[2]) for e in ev) if ev else (0.0, 0.0)
            if best[0] <= IOU_THR: continue
            (tg if kind == "target" else dc).append(best[1])
    return _auc(tg, dc), len(tg), len(dc)

print("\n[score_spec_detected -- separating detection loss from specificity loss]")

a, nt, nd = score_spec_detected(V, loud)
check("loud -> same AUC as score_spec when everything is localised",
      a == 1.0 and (nt, nd) == (1, 1), f"{a} {nt} {nd}")

a, nt, nd = score_spec_detected(V, faint)
check("faint -> nan, NOT a fake 0.5 (nothing was localised to score)",
      np.isnan(a) and (nt, nd) == (0, 0), f"{a} {nt} {nd}")

# THE CRUX. Asymmetric detection: every decoy localised, only 1 of 4 targets. Nothing about
# species discrimination has changed -- the one detected target still outscores every decoy.
asym = [([(0.0, 1.0, 0.9, 0.9),                      # target 1 localised, scores high
          (20.0, 21.0, 0.3, 0.3), (22.0, 23.0, 0.3, 0.3),
          (24.0, 25.0, 0.3, 0.3), (26.0, 27.0, 0.3, 0.3)],
         [("target", (0.0, 1.0)), ("target", (2.0, 3.0)),      # targets 2-4 never localised
          ("target", (4.0, 5.0)), ("target", (6.0, 7.0)),
          ("d1", (20.0, 21.0)), ("d1", (22.0, 23.0)),
          ("d2", (24.0, 25.0)), ("d2", (26.0, 27.0))])]
auc_raw = score_spec_auc(V, asym, ["d1", "d2"])
auc_det, nt, nd = score_spec_detected(V, asym)
cov = spec_coverage(asym)
check("asymmetric detection drags raw AUC BELOW CHANCE (0.25) with no species confusion",
      auc_raw == 0.25, str(auc_raw))
check("detection-corrected AUC is 1.0 on the same cache -- discrimination was never lost",
      auc_det == 1.0 and (nt, nd) == (1, 4), f"{auc_det} {nt} {nd}")
check("coverage exposes the asymmetry that caused it (target 25% vs decoy 100%)",
      cov == {"target": 0.25, "decoy": 1.0}, str(cov))
check("the two metrics disagree by 0.75 -- this is the whole point of measuring both",
      abs(auc_det - auc_raw) == 0.75, f"{auc_det} vs {auc_raw}")

# Ties are still ties among detected items -- correction must not invent separation.
tie = [([(0.0, 1.0, 0.5, 0.5), (2.0, 3.0, 0.5, 0.5)],
        [("target", (0.0, 1.0)), ("d1", (2.0, 3.0))])]
check("genuinely indistinguishable + fully detected -> 0.5 (a REAL chance result)",
      score_spec_detected(V, tie)[0] == 0.5, str(score_spec_detected(V, tie)[0]))

check("fusion rule changes the corrected AUC but never the counts",
      score_spec_detected(lambda p, v: p, asym)[1:] == (1, 4))

# ------------------------------------------- SKIP_MAIN_LOOP structure (roadmap: quota resume)
# arguswala_v2.py imports torch/librosa/a live GPU and cannot be exec'd locally, so this
# checks the STRUCTURE via ast rather than the behaviour: that `encoders` is built
# unconditionally, that the expensive loop and every `res`-dependent report live inside
# the `else:` branch (never at module scope), and that the default is off.
import ast

print("\n[SKIP_MAIN_LOOP structure -- ladder-resume path, added 18 Aug 2026]")
tree = ast.parse(open(os.path.join(HERE, "arguswala_v2.py"), encoding="utf-8").read())
mod_body = tree.body

def top_level_targets(node):
    """Names assigned directly at module scope by this statement (Assign/AugAssign/For)."""
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
    return names

# 1. `encoders` must be a top-level (unconditional) module statement, not inside any If.
enc_stmt = next((s for s in mod_body if "encoders" in top_level_targets(s)
                 and isinstance(s, ast.Assign)), None)
check("`encoders = {}` is a top-level module statement (built regardless of the flag)",
      enc_stmt is not None, "no unconditional top-level assignment to encoders found")

# 2. exactly one top-level `if SKIP_MAIN_LOOP: ... else: ...`, and it comes after `encoders`.
if_stmts = [s for s in mod_body if isinstance(s, ast.If)
            and isinstance(s.test, ast.Name) and s.test.id == "SKIP_MAIN_LOOP"]
check("exactly one top-level `if SKIP_MAIN_LOOP` gate", len(if_stmts) == 1, str(len(if_stmts)))
gate = if_stmts[0] if if_stmts else None
check("has both branches (skip path AND the real loop in else)",
      gate is not None and len(gate.orelse) > 0)
if gate is not None and enc_stmt is not None:
    check("the gate appears AFTER encoders are built, not before",
          mod_body.index(gate) > mod_body.index(enc_stmt))

# 3. nothing that depends on `res`/`wide`/`piv` exists OUTSIDE the else-branch -- i.e. at
# module scope, or in the if-branch. Those names must only ever be built/read inside else.
if gate is not None:
    outside = [s for s in mod_body if s is not gate] + list(gate.body)   # module stmts + if-body
    leaked = set()
    for s in outside:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and n.id in ("res", "wide", "piv"):
                leaked.add(n.id)
    check("res/wide/piv never referenced outside the else-branch (skip path is self-contained)",
          not leaked, f"leaked names: {leaked}")

    # and the else-branch DOES build `res` -- confirms the wrap didn't eat the real loop
    else_names = set()
    for s in gate.orelse:
        else_names |= top_level_targets(s)
    check("the else-branch still builds `res` (the real per-species loop is intact)",
          "res" in else_names, str(sorted(else_names)))

# 4. default is off -- a normal `Run All` must behave exactly as before.
flag_assigns = [s for s in mod_body if isinstance(s, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "SKIP_MAIN_LOOP" for t in s.targets)]
check("SKIP_MAIN_LOOP has exactly one top-level default assignment", len(flag_assigns) == 1,
      str(len(flag_assigns)))
if flag_assigns:
    val = flag_assigns[0].value
    check("default value is literal False (normal runs unaffected)",
          isinstance(val, ast.Constant) and val.value is False, ast.dump(val))

# 5. cell 3 and cell 4 never reference res/wide/piv either -- confirms a resume session can
# skip cell 3 entirely and go straight from cell 2 (skipped) to cell 4.
for fname in ("arguswala_sweeps.py", "arguswala_ladder.py"):
    src = open(os.path.join(HERE, fname), encoding="utf-8").read()
    hit = {n.id for n in ast.walk(ast.parse(src))
           if isinstance(n, ast.Name) and n.id in ("res", "wide", "piv")}
    check(f"{fname} does not depend on res/wide/piv (safe to skip CELL 3 on ladder-resume)",
          not hit, f"found: {hit}")

# ------------------------------- S3 three-way trend verdict (roadmap 3.2)
# The decision rule from trend() in arguswala_sweeps.py, replicated verbatim.
KNIFE = 0.25

def verdict(drop, bound):
    margin = abs(drop) - bound
    if margin > KNIFE * bound:
        return "shrinks" if drop > 0 else "grows"
    if margin < -KNIFE * bound:
        return "flat"
    return "unresolved"

print("\n[S3 trend verdict -- three-way, not two]")

check("|change| far above bound -> a real shrinking trend",
      verdict(0.210, 0.025) == "shrinks", verdict(0.210, 0.025))
check("negative change far above bound -> grows, not shrinks",
      verdict(-0.190, 0.025) == "grows", verdict(-0.190, 0.025))

# THE BUG THIS EXISTS FOR (23 Aug 2026): an earlier version tested only
# `margin < KNIFE*bound`, which is true for BOTH "|change| ~= bound" (ambiguous) and
# "|change| << bound" (confidently flat) -- so a change of exactly 0.000 was reported as
# "too close to call". These are opposite conclusions and must never share a branch.
check("change of exactly 0.000 vs a real bound -> FLAT, never 'unresolved'",
      verdict(0.000, 0.045) == "flat", verdict(0.000, 0.045))
check("the real 23 Aug detection-corrected result reads FLAT",
      verdict(-0.000, 0.045) == "flat")
check("the real 23 Aug coverage>=75% gain reads FLAT",
      verdict(0.010, 0.090) == "flat", verdict(0.010, 0.090))
old_rule = lambda drop, bound: "unresolved" if abs(drop)-bound < KNIFE*bound else "trend"
check("REGRESSION GUARD: the old two-way rule really did bury these as 'unresolved'",
      old_rule(0.000, 0.045) == "unresolved" and old_rule(0.010, 0.090) == "unresolved")

# The genuinely ambiguous middle must survive -- the knife-edge guard still has to work.
check("|change| just above bound -> unresolved, not a finding",
      verdict(0.051, 0.045) == "unresolved", verdict(0.051, 0.045))
check("|change| just below bound -> unresolved, not 'flat'",
      verdict(0.086, 0.098) == "unresolved", verdict(0.086, 0.098))
check("|change| exactly equal to bound -> unresolved",
      verdict(0.045, 0.045) == "unresolved", verdict(0.045, 0.045))

# The three branches must partition: every input lands in exactly one.
cases = [(d/1000, b/1000) for d in range(-300, 301, 7) for b in range(5, 200, 11)]
got = {verdict(d, b) for d, b in cases}
check("every (change, bound) pair resolves to exactly one of the three verdicts",
      got <= {"shrinks", "grows", "flat", "unresolved"} and len(got) == 4, str(sorted(got)))
check("flat and shrinks are never both reachable for one input (partition holds)",
      all((abs(d) - b > KNIFE*b) + (abs(d) - b < -KNIFE*b) <= 1 for d, b in cases))

# ---- S3 NOISE-BOUND PAIRING (the 28 Aug 2026 bug: right rule, wrong bound) ----------
# trend() was correct; what was fed to it was not. The curve dict computed only two sds,
# BOTH from the verified subset, and every test got one of them. Three of four tests were
# therefore judged against a series they were not measuring:
#   [1]/[2] test the GAIN (ver - s1) -> must use the PAIRED gain sd, not sd(ver)
#   [4]     tests a STAGE-1 statistic -> must use stage-1's sd, not the verifier's
# [4] was the damaging one: stage-1's localised AUC is ~1.7x noisier, so the bound was
# 1.7x too tight and [4] printed SHRINKS. That reached the live site as "the verifier is
# what confers the SNR-invariance". Real numbers from data/argus_sweep_snr.csv below.
print("\n[S3 noise-bound pairing -- each test bounded by its OWN series]")

R3_DET_S1_HI, R3_DET_S1_LO = 0.4991, 0.4320     # stage-1 localised AUC, +9 dB -> -6 dB
R3_SD_DET_VER, R3_SD_DET_S1 = 0.0457, 0.0776    # max-over-levels sd of each series
check("stage-1's localised AUC really is noisier than the verifier's (the whole reason "
      "the mismatch mattered)", R3_SD_DET_S1 > R3_SD_DET_VER,
      f"{R3_SD_DET_S1} vs {R3_SD_DET_VER}")
_d4 = R3_DET_S1_HI - R3_DET_S1_LO
check("REGRESSION GUARD: the shipped (verified) bound really did print SHRINKS for test [4]",
      verdict(_d4, R3_SD_DET_VER) == "shrinks", verdict(_d4, R3_SD_DET_VER))
check("with stage-1's OWN bound, test [4] is unresolved -- the published claim does not hold",
      verdict(_d4, R3_SD_DET_S1) == "unresolved", verdict(_d4, R3_SD_DET_S1))
# [3] was the one test that was always correctly paired, and it must stay FLAT --
# it is the load-bearing "SNR-invariance once localised" result.
check("test [3] was correctly paired all along and still reads FLAT",
      verdict(0.8571 - 0.8525, R3_SD_DET_VER) == "flat")
# [1]: the shipped bound was too LOOSE here (sd of ver alone > paired gain sd), so it
# overclaimed FLATNESS rather than a trend. Opposite direction, same root cause.
check("REGRESSION GUARD: the shipped bound printed FLAT for test [1]",
      verdict(0.2447 - 0.1801, 0.1055) == "flat")
check("with the paired gain sd, test [1] is unresolved, not flat",
      verdict(0.2447 - 0.1801, 0.0810) == "unresolved")
check("[2] (coverage-gated gain) is FLAT under BOTH bounds -- genuinely robust",
      verdict(0.2447 - 0.2557, 0.0803) == "flat"
      and verdict(0.2447 - 0.2557, 0.0810) == "flat")

# Structural: the source must now compute a separate sd per series and pair them correctly.
_sw = open(os.path.join(HERE, "arguswala_sweeps.py"), encoding="utf-8").read()
for _name in ("sd_gain", "sd_det_s1"):
    check(f"arguswala_sweeps.py computes {_name}", _name in _sw)
check("test [4] is bounded by sd_det_s1, not the verifier's sd_det",
      '"sd_det_s1"' in _sw and 'lambda c: c["det_s1"], "sd_det"' not in _sw)
check("tests [1]/[2] are bounded by sd_gain, not the verified-AUC sd",
      _sw.count('lambda c: c["gain"], "sd_gain"') == 2
      and 'lambda c: c["gain"], "sd"' not in _sw)
check("test [3] still uses the verified sd_det (it was always correct)",
      'lambda c: c["det_ver"], "sd_det"' in _sw)

# ------------------------------------------- decoy_mode selection logic (roadmap 3.3)
# Verbatim slicing rules from build_target() in arguswala_setup_v2.py, exercised directly
# since the real function needs Perch/audio/a live encoder bank.
N_DECOYS = 4

def select_decoys(ranked, decoy_mode, seed):
    import numpy as np
    if decoy_mode == "hard":
        return ranked[:N_DECOYS]
    if decoy_mode == "medium":
        out = ranked[N_DECOYS:2 * N_DECOYS]
        if len(out) < N_DECOYS:
            raise ValueError("not enough candidates for decoy_mode='medium'")
        return out
    if decoy_mode == "random":
        return list(np.random.default_rng(90_000 + seed).choice(
            ranked, size=N_DECOYS, replace=False))
    raise ValueError(f"unknown decoy_mode: {decoy_mode!r}")

print("\n[decoy_mode selection -- roadmap 3.3]")
RANKED = [f"sp{i}" for i in range(30)]   # already sorted hardest-first, as build_target does

check("hard -> the first N_DECOYS, unchanged from pre-3.3 behaviour",
      select_decoys(RANKED, "hard", 0) == RANKED[:4], str(select_decoys(RANKED, "hard", 0)))

med = select_decoys(RANKED, "medium", 0)
check("medium -> ranks 5-8 (indices 4:8), exactly as the roadmap specifies",
      med == RANKED[4:8], str(med))
check("medium and hard never overlap", not (set(med) & set(select_decoys(RANKED, "hard", 0))))

rnd = select_decoys(RANKED, "random", 0)
check("random -> N_DECOYS distinct picks from the full candidate pool",
      len(rnd) == 4 and len(set(rnd)) == 4 and all(r in RANKED for r in rnd), str(rnd))
check("random is reproducible from the same seed",
      select_decoys(RANKED, "random", 0) == select_decoys(RANKED, "random", 0))
check("random differs (in general) across seeds -- not silently constant",
      select_decoys(RANKED, "random", 0) != select_decoys(RANKED, "random", 1))
check("random for one seed differs across DIFFERENT species' ranked lists",
      select_decoys(RANKED, "random", 3) != select_decoys(list(reversed(RANKED)), "random", 3))

check("medium raises cleanly rather than silently returning <4 decoys on a short pool",
      True, "checked via except below")
try:
    select_decoys(RANKED[:6], "medium", 0)   # only 6 candidates, medium needs ranks 5-8
    check("medium on a too-short pool raises ValueError", False, "did not raise")
except ValueError:
    check("medium on a too-short pool raises ValueError", True)

try:
    select_decoys(RANKED, "easy", 0)
    check("unknown decoy_mode raises ValueError, not a silent fallback", False, "did not raise")
except ValueError:
    check("unknown decoy_mode raises ValueError, not a silent fallback", True)

# The cache key in build_target MUST include decoy_mode -- otherwise a 'random' run
# would silently reuse a cached 'hard' target's DECOYS. Structural check on the real file.
src = open(os.path.join(HERE, "arguswala_setup_v2.py"), encoding="utf-8").read()
ck_lines = [l.strip() for l in src.splitlines() if l.strip().startswith("ck = (")]
check("build_target's cache key includes decoy_mode (else 'random' would reuse 'hard' cache)",
      len(ck_lines) == 1 and "decoy_mode" in ck_lines[0], str(ck_lines))

# run_one()'s default must be 'hard' so N1/S2 (which never pass decoy_mode) are byte-for-byte
# the same run they always were.
sweeps_src = open(os.path.join(HERE, "arguswala_sweeps.py"), encoding="utf-8").read()
check("run_one() defaults decoy_mode to 'hard' (N1/S2 unaffected by this feature existing)",
      'decoy_mode="hard"' in sweeps_src)

# ------------------------------------------- SKIP_LADDER structure (roadmap: quota resume)
import ast

print("\n[SKIP_LADDER structure -- skip an already-complete ladder]")
tree = ast.parse(open(os.path.join(HERE, "arguswala_ladder.py"), encoding="utf-8").read())
mod_body = tree.body

if_stmts = [s for s in mod_body if isinstance(s, ast.If)
            and isinstance(s.test, ast.Name) and s.test.id == "SKIP_LADDER"]
check("exactly one top-level `if SKIP_LADDER` gate", len(if_stmts) == 1, str(len(if_stmts)))
gate = if_stmts[0] if if_stmts else None
check("has both branches (skip message AND the real loop in else)",
      gate is not None and len(gate.orelse) > 0)

flag_assigns = [s for s in mod_body if isinstance(s, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "SKIP_LADDER" for t in s.targets)]
check("SKIP_LADDER has exactly one top-level default assignment", len(flag_assigns) == 1,
      str(len(flag_assigns)))
if flag_assigns:
    val = flag_assigns[0].value
    check("default value is literal False (normal runs unaffected)",
          isinstance(val, ast.Constant) and val.value is False, ast.dump(val))

# the flag must be declared BEFORE the gate that reads it
if gate is not None and flag_assigns:
    check("SKIP_LADDER is assigned before the gate that reads it",
          mod_body.index(flag_assigns[0]) < mod_body.index(gate))

# the else-branch must still build `lad` (the real report depends on it) -- confirms the
# wrap didn't accidentally swallow the run loop
def top_level_targets(node):
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
    return names
if gate is not None:
    else_names = set()
    for s in gate.orelse:
        else_names |= top_level_targets(s)
    check("the else-branch still builds `lad` (the real run loop is intact)",
          "lad" in else_names, str(sorted(else_names)))

# ------------------------------------------- SKIP_SWEEPS structure (roadmap: quota resume)
# Same purpose as SKIP_MAIN_LOOP / SKIP_LADDER, added 28 Aug 2026 after run 4 spent its
# entire 12h ceiling on Cells 1-4 (both already-complete) and never reached Cell 5. Gate
# shape is inverted from the other two (`if not FLAG: <real work> else: <skip message>`)
# so the checks below match that shape rather than reusing SKIP_LADDER's verbatim.
print("\n[SKIP_SWEEPS structure -- skip already-replicated sweeps]")
tree = ast.parse(open(os.path.join(HERE, "arguswala_sweeps.py"), encoding="utf-8").read())
mod_body = tree.body

if_stmts = [s for s in mod_body if isinstance(s, ast.If)
            and isinstance(s.test, ast.UnaryOp) and isinstance(s.test.op, ast.Not)
            and isinstance(s.test.operand, ast.Name) and s.test.operand.id == "SKIP_SWEEPS"]
check("exactly one top-level `if not SKIP_SWEEPS` gate", len(if_stmts) == 1, str(len(if_stmts)))
gate = if_stmts[0] if if_stmts else None
check("has both branches (the real sweeps in if, skip message in else)",
      gate is not None and len(gate.body) > 0 and len(gate.orelse) > 0)

flag_assigns = [s for s in mod_body if isinstance(s, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "SKIP_SWEEPS" for t in s.targets)]
check("SKIP_SWEEPS has exactly one top-level default assignment", len(flag_assigns) == 1,
      str(len(flag_assigns)))
if flag_assigns:
    val = flag_assigns[0].value
    # Asserts False to match SKIP_MAIN_LOOP / SKIP_LADDER above, NOT the value that
    # happens to be set for whichever session is configured today. Previously this
    # asserted True because that was the state when it was written -- which meant the
    # suite's expected-failure count silently changed every time a flag was flipped for a
    # run, and "3 failures" stopped distinguishing "one flag deliberately flipped" from
    # "someone broke something". All three skip flags now read the same way: a failure
    # here means exactly one thing -- this flag is currently flipped for a session, and
    # must go back to False before the pipeline is shipped or re-run normally.
    check("SKIP_SWEEPS default is literal False (normal runs unaffected)",
          isinstance(val, ast.Constant) and val.value is False, ast.dump(val))

if gate is not None and flag_assigns:
    check("SKIP_SWEEPS is assigned before the gate that reads it",
          mod_body.index(flag_assigns[0]) < mod_body.index(gate))

# the if-branch (the real work) must still write all 4 sweep CSVs -- confirms the wrap
# didn't accidentally swallow or reorder any of N1/S2/S3/S4's output.
if gate is not None:
    csv_names = {n.value for s in gate.body for n in ast.walk(s)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)
                 and n.value.endswith(".csv")}
    check("all four sweep CSVs are still written inside the real-work branch",
          csv_names == {"/kaggle/working/argus_sweep_shot_strategy.csv",
                        "/kaggle/working/argus_sweep_shot_count.csv",
                        "/kaggle/working/argus_sweep_snr.csv",
                        "/kaggle/working/argus_sweep_decoy_difficulty.csv"},
          str(sorted(csv_names)))
    # run_one/run_snr must stay defined OUTSIDE the gate (cheap; SKIP_LADDER's run_config
    # follows the same convention) so they're callable regardless of the flag.
    gate_start = mod_body.index(gate)
    funcs_before_gate = {s.name for s in mod_body[:gate_start] if isinstance(s, ast.FunctionDef)}
    check("run_one() is defined outside the gate", "run_one" in funcs_before_gate)

print("\n" + "="*60)
print(f"{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILURES: ' + ', '.join(FAILS)}")
print("="*60)
