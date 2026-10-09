# -*- coding: utf-8 -*-
"""Local tests for the ablation-ladder additions: PrototypicalProbe, TimeFilterAug,
hard-negative selection. Functions are re-declared verbatim from the pipeline sources."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import os
HERE = os.path.dirname(os.path.abspath(__file__))   # the pipeline files sit next to this test

FAILS = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else "   <- " + detail))
    if not cond: FAILS.append(name)

def l2n(x):
    x = np.asarray(x, dtype="float32")
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

# ------------------------------------------------- PrototypicalProbe (verbatim)
class PrototypicalProbe:
    def __init__(self, n_prototypes=4, random_state=0, class_weight="balanced"):
        self.n_prototypes = n_prototypes
        self.random_state = random_state
        self.class_weight = class_weight
    def _protos(self, Xc):
        k = int(min(self.n_prototypes, len(Xc)))
        if k <= 1: return l2n(Xc.mean(0, keepdims=True))
        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(Xc)
        return l2n(km.cluster_centers_)
    def _scores(self, X):
        Xn = l2n(X)
        return np.stack([(Xn @ self.protos_[c].T).max(1) for c in (0, 1)], axis=1)
    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True); e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    def _fit_temperature(self, S, y, w, eps=0.05):
        best_T, best_ll = 1.0, np.inf
        for T in np.geomspace(0.02, 3.0, 60):
            p_true = np.clip(self._softmax(S / T)[np.arange(len(y)), y], 1e-9, 1.0)
            p_false = np.clip(1.0 - p_true, 1e-9, 1.0)
            ll = float(-(w * ((1.0 - eps) * np.log(p_true)
                              + eps * np.log(p_false))).sum() / w.sum())
            if ll < best_ll: best_T, best_ll = float(T), ll
        return best_T
    def fit(self, X, y):
        X = np.asarray(X, dtype="float32"); y = np.asarray(y).astype(int)
        self.protos_ = {c: self._protos(X[y == c]) for c in (0, 1)}
        if self.class_weight == "balanced":
            cw = {c: len(y) / (2.0 * max(1, (y == c).sum())) for c in (0, 1)}
        else: cw = {0: 1.0, 1: 1.0}
        w = np.array([cw[int(v)] for v in y], dtype="float64")
        self.temperature_ = self._fit_temperature(self._scores(X), y, w)
        return self
    def predict_proba(self, X):
        return self._softmax(self._scores(X) / self.temperature_)
    def score(self, X, y):
        return float((self.predict_proba(X).argmax(1) == np.asarray(y).astype(int)).mean())

def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = float((pos[:, None] > neg[None, :]).sum()); eq = float((pos[:, None] == neg[None, :]).sum())
    return (gt + 0.5 * eq) / (len(pos) * len(neg))

print("\n[PrototypicalProbe: basics]")
rng = np.random.default_rng(0)
Xp = l2n(rng.normal(0, 1, (60, 32)) + np.r_[3.0, np.zeros(31)])
Xn = l2n(rng.normal(0, 1, (60, 32)) - np.r_[3.0, np.zeros(31)])
X = np.vstack([Xn, Xp]); y = np.r_[np.zeros(60), np.ones(60)].astype(int)
pp = PrototypicalProbe().fit(X, y)
P = pp.predict_proba(X)
check("predict_proba shape (n,2)", P.shape == (120, 2), str(P.shape))
check("rows sum to 1", np.allclose(P.sum(1), 1.0))
check("in [0,1]", P.min() >= 0 and P.max() <= 1)
check("separable data -> AUC 1.0", auc(P[y==1, 1], P[y==0, 1]) == 1.0)
check("temperature fitted > 0", pp.temperature_ > 0, str(pp.temperature_))
check("not saturated to hard 0/1 (v must rank, not veto)",
      P[:, 1].max() < 0.999 and P[:, 1].min() > 1e-3,
      f"max={P[:,1].max():.6f} min={P[:,1].min():.6f}")
check("v retains ranking resolution among positives",
      len(np.unique(np.round(P[y == 1, 1], 6))) > 5,
      f"{len(np.unique(np.round(P[y==1,1],6)))} distinct values")
check("deterministic", np.allclose(PrototypicalProbe().fit(X, y).predict_proba(X), P))

print("\n[PrototypicalProbe: k <= n guards]")
Xs = np.vstack([Xn[:1], Xp[:1]]); ys = np.array([0, 1])
ps = PrototypicalProbe(n_prototypes=4).fit(Xs, ys)
check("1 sample/class -> 1 prototype each",
      all(ps.protos_[c].shape[0] == 1 for c in (0, 1)))
check("still predicts", ps.predict_proba(Xs).shape == (2, 2))
p2 = PrototypicalProbe(n_prototypes=3).fit(np.vstack([Xn[:2], Xp[:5]]),
                                           np.r_[0, 0, 1, 1, 1, 1, 1].astype(int))
check("k clipped to class size", p2.protos_[0].shape[0] == 2 and p2.protos_[1].shape[0] == 3,
      f"{p2.protos_[0].shape} {p2.protos_[1].shape}")

# ---- THE SCIENTIFIC CLAIM ----
# A species with two distinct call types = a bimodal positive class. A single averaged
# prototype lands BETWEEN the modes -- exactly where the confusable negatives sit -- which
# is the failure Nolasco et al. sec.4.1 describe and the reason ARGUS scored below chance.
# Multi-prototype should survive it; single-prototype should not.
print("\n[PrototypicalProbe: multi-modal positive class -- the actual hypothesis]")
d = 32
modeA = np.zeros(d); modeA[0] = 1.0          # call type 1
modeB = np.zeros(d); modeB[1] = 1.0          # call type 2
between = l2n((modeA + modeB)[None, :])[0]   # decoy sits between the two call types
r2 = np.random.default_rng(7)
pos = l2n(np.vstack([modeA + 0.10 * r2.normal(0, 1, (40, d)),
                     modeB + 0.10 * r2.normal(0, 1, (40, d))]))
neg = l2n(between + 0.10 * r2.normal(0, 1, (80, d)))
Xm = np.vstack([neg, pos]); ym = np.r_[np.zeros(80), np.ones(80)].astype(int)

single = PrototypicalProbe(n_prototypes=1).fit(Xm, ym)
multi  = PrototypicalProbe(n_prototypes=4).fit(Xm, ym)
a1 = auc(single.predict_proba(Xm)[ym==1, 1], single.predict_proba(Xm)[ym==0, 1])
a4 = auc(multi.predict_proba(Xm)[ym==1, 1],  multi.predict_proba(Xm)[ym==0, 1])
print(f"    single-prototype AUC {a1:.3f}   |   4-prototype AUC {a4:.3f}")
check("single averaged prototype degrades badly on bimodal class", a1 < 0.75, f"AUC {a1:.3f}")
check("multi-prototype RECOVERS it", a4 > 0.90, f"AUC {a4:.3f}")
check("multi beats single by a wide margin", a4 - a1 > 0.30, f"delta {a4-a1:.3f}")
# sanity: on a unimodal class the two should be comparable (no free lunch claimed)
su = PrototypicalProbe(n_prototypes=1).fit(X, y); mu = PrototypicalProbe(n_prototypes=4).fit(X, y)
au1 = auc(su.predict_proba(X)[y==1,1], su.predict_proba(X)[y==0,1])
au4 = auc(mu.predict_proba(X)[y==1,1], mu.predict_proba(X)[y==0,1])
check("unimodal: multi does not hurt", au4 >= au1 - 1e-9, f"{au1:.3f} -> {au4:.3f}")

print("\n[PrototypicalProbe vs LogisticRegression on the bimodal case]")
lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(Xm, ym)
alr = auc(lr.predict_proba(Xm)[ym==1, 1], lr.predict_proba(Xm)[ym==0, 1])
print(f"    LogReg(C=0.1) AUC {alr:.3f}  vs  4-prototype AUC {a4:.3f}")
check("proto probe >= logreg on multi-modal positives", a4 >= alr - 1e-6,
      f"logreg {alr:.3f} vs proto {a4:.3f}")

# ------------------------------------------------- TimeFilterAug (verbatim logic)
FEATURE = "pcen"
def timefilteraug(x, rng, m_range=(3, 6), lo_db=-6.0, hi_db=8.0):
    T = x.shape[1]
    m = int(rng.integers(m_range[0], m_range[1] + 1))
    if T < m + 1: return x
    cuts = np.sort(rng.choice(np.arange(1, T), size=m - 1, replace=False))
    edges = np.concatenate([[0], cuts, [T]])
    g = rng.random(m + 1)
    alpha = np.concatenate([np.linspace(g[i], g[i+1], int(edges[i+1]-edges[i]), endpoint=False)
                            for i in range(m)])
    beta = (lo_db + (hi_db - lo_db) * alpha).astype("float32")
    if FEATURE == "logmel":
        return x + beta[np.newaxis, :]
    return x * (10.0 ** (beta / 20.0))[np.newaxis, :].astype("float32")

print("\n[TimeFilterAug]")
r3 = np.random.default_rng(11)
spec = np.abs(r3.normal(1.0, 0.2, (128, 87))).astype("float32")   # PCEN-like, 1 s window
out = timefilteraug(spec, r3)
check("shape preserved", out.shape == spec.shape, str(out.shape))
ratio = out / np.maximum(spec, 1e-9)
check("gain identical across mel bins (time-only filter)",
      np.allclose(ratio, ratio[0][np.newaxis, :], atol=1e-4))
gdb = 20 * np.log10(np.maximum(ratio[0], 1e-12))
check("gain within [-6, +8] dB", gdb.min() >= -6.01 and gdb.max() <= 8.01,
      f"[{gdb.min():.2f}, {gdb.max():.2f}]")
check("gain actually varies over time", gdb.std() > 0.5, f"sd={gdb.std():.3f}")
# piecewise-linear in dB => second difference is zero except at the few breakpoints
d2 = np.abs(np.diff(gdb, 2))
check("piecewise-linear in dB (<=5 breakpoints)", int((d2 > 1e-3).sum()) <= 5,
      f"{int((d2>1e-3).sum())} kinks")
check("PCEN output stays non-negative", out.min() >= 0.0)

FEATURE = "logmel"
lm = (r3.normal(-40, 10, (128, 87))).astype("float32")            # dB-domain
lout = timefilteraug(lm, r3)
delta = lout - lm
check("logmel: gain is ADDITIVE in dB", np.allclose(delta, delta[0][np.newaxis, :], atol=1e-4))
check("logmel: delta within [-6, +8] dB", delta.min() >= -6.01 and delta.max() <= 8.01,
      f"[{delta.min():.2f}, {delta.max():.2f}]")
FEATURE = "pcen"
check("short window returns unchanged",
      np.array_equal(timefilteraug(spec[:, :3], np.random.default_rng(1)), spec[:, :3]))

print("\n[hard-negative selection]")
def pick_hard(neg_scores, keep):
    k = max(4, int(round(len(neg_scores) * keep)))
    return np.argsort(-np.asarray(neg_scores))[:k]
scores = np.linspace(0, 1, 200)
idx = pick_hard(scores, 0.25)
check("keeps 25%", len(idx) == 50, str(len(idx)))
check("keeps the HIGHEST-scoring (most confusable)", scores[idx].min() > scores.mean(),
      f"min kept {scores[idx].min():.3f}")
check("floor of 4 on tiny pools", len(pick_hard(np.linspace(0, 1, 5), 0.25)) == 4)
check("never exceeds pool size", len(pick_hard(np.linspace(0, 1, 3), 0.25)) <= 3)

# ------------------------------------------------- front-end as a ladder factor
import itertools
print("\n[feat() front-end resolution]")
def resolve(feature, global_feature):
    return feature or global_feature
check("explicit arg wins over global", resolve("logmel", "pcen") == "logmel")
check("None falls back to global", resolve(None, "pcen") == "pcen")
check("explicit pcen under logmel global", resolve("pcen", "logmel") == "pcen")

print("\n[factorial design]")
def factorial():
    out = []
    for v, t, hn, ft in itertools.product(["logreg", "proto"], [False, True],
                                          [False, True], ["pcen", "logmel"]):
        tag = ft + ("|proto" if v == "proto" else "|logreg")
        tag += "+tfa" if t else ""
        tag += "+hn" if hn else ""
        out.append((tag, dict(VERIFIER=v, USE_TIMEFILTER=t, USE_HARD_NEG=hn, FEATURE=ft)))
    return out
F = factorial()
check("2^4 = 16 configs", len(F) == 16, str(len(F)))
check("all labels unique", len(set(t for t, _ in F)) == 16)
for flag, levels in (("VERIFIER", ["logreg", "proto"]), ("USE_TIMEFILTER", [False, True]),
                     ("USE_HARD_NEG", [False, True]), ("FEATURE", ["pcen", "logmel"])):
    counts = [sum(1 for _, c in F if c[flag] == lv) for lv in levels]
    check(f"{flag} balanced (8/8)", counts == [8, 8], str(counts))
check("baseline present exactly once",
      sum(1 for _, c in F if c == dict(VERIFIER="logreg", USE_TIMEFILTER=False,
                                       USE_HARD_NEG=False, FEATURE="pcen")) == 1)

print("\n[encoder key resolution -- silent-garbage guard]")
ENCODER_SEEDS = (0,)
encoders = {("pcen", 0): "ENC_PCEN", ("logmel", 0): "ENC_LOGMEL"}
def resolve_enc(cfg_feature, enc_map, seeds=ENCODER_SEEDS):
    key = (cfg_feature, seeds[0])
    if key not in enc_map:
        raise KeyError(f"no encoder for {key}; have {sorted(enc_map)}")
    return enc_map[key]
check("pcen config -> pcen encoder", resolve_enc("pcen", encoders) == "ENC_PCEN")
check("logmel config -> logmel encoder", resolve_enc("logmel", encoders) == "ENC_LOGMEL")
check("every factorial config resolves an encoder",
      all(resolve_enc(c["FEATURE"], encoders) for _, c in F))
try:
    resolve_enc("mfcc", encoders); check("unknown front-end raises", False, "no exception")
except KeyError:
    check("unknown front-end raises", True)
try:
    resolve_enc("logmel", {("pcen", 0): "ENC_PCEN"})
    check("missing encoder raises rather than silently reusing", False, "no exception")
except KeyError:
    check("missing encoder raises rather than silently reusing", True)

print("\n[cumulative ladder]")
CUM = [
    ("rung 0", dict(VERIFIER="logreg", USE_TIMEFILTER=False, USE_HARD_NEG=False, FEATURE="pcen")),
    ("rung 1", dict(VERIFIER="proto",  USE_TIMEFILTER=False, USE_HARD_NEG=False, FEATURE="pcen")),
    ("rung 2", dict(VERIFIER="proto",  USE_TIMEFILTER=True,  USE_HARD_NEG=False, FEATURE="pcen")),
    ("rung 3", dict(VERIFIER="proto",  USE_TIMEFILTER=True,  USE_HARD_NEG=True,  FEATURE="pcen")),
    ("rung 4", dict(VERIFIER="proto",  USE_TIMEFILTER=True,  USE_HARD_NEG=True,  FEATURE="logmel")),
]
check("rung 0 is the untouched original pipeline",
      CUM[0][1] == dict(VERIFIER="logreg", USE_TIMEFILTER=False, USE_HARD_NEG=False, FEATURE="pcen"))
diffs = [sum(1 for k in CUM[i][1] if CUM[i][1][k] != CUM[i-1][1][k]) for i in range(1, len(CUM))]
check("each rung changes exactly ONE factor", diffs == [1, 1, 1, 1], str(diffs))
check("every cumulative rung is a factorial cell",
      all(c in [cc for _, cc in F] for _, c in CUM))

# ------------------------------------------- revised ladder: ordering + resume (14 Aug)
print("\n[config ordering -- distance from baseline]")
BASELINE_CFG = dict(VERIFIER="logreg", USE_TIMEFILTER=False, USE_HARD_NEG=False, FEATURE="pcen")

def _tag(c):
    t = f"{c['FEATURE']}|{c['VERIFIER']}"
    if c["USE_TIMEFILTER"]: t += "+tfa"
    if c["USE_HARD_NEG"]:   t += "+hn"
    return t

def _dist(c):
    return sum(1 for k, v in BASELINE_CFG.items() if c[k] != v)

def factorial_ordered(skip_tfa):
    tfa = (False,) if skip_tfa else (False, True)
    out = []
    for v, t, hn, ft in itertools.product(("logreg", "proto"), tfa, (False, True), ("pcen", "logmel")):
        c = dict(VERIFIER=v, USE_TIMEFILTER=t, USE_HARD_NEG=hn, FEATURE=ft)
        out.append((_tag(c), c))
    out.sort(key=lambda kc: (_dist(kc[1]), kc[0]))
    return out

F16 = factorial_ordered(False)
F8  = factorial_ordered(True)
check("skip_tfa=False -> 16 configs", len(F16) == 16, str(len(F16)))
check("skip_tfa=True  -> 8 configs",  len(F8) == 8,  str(len(F8)))
check("baseline is always first", F16[0][0] == "pcen|logreg" and F8[0][0] == "pcen|logreg")
check("no tfa configs when skipped",
      all(not c["USE_TIMEFILTER"] for _, c in F8))
check("distances are non-decreasing (truncation-safe)",
      all(_dist(F16[i][1]) <= _dist(F16[i+1][1]) for i in range(len(F16)-1)))

# THE regression this ordering exists to prevent: the old ordering put every proto config
# last, so a run killed at 6/16 produced zero data on the prototypical probe.
def factorial_OLD():
    out = []
    for v, t, hn, ft in itertools.product(("logreg", "proto"), (False, True), (False, True), ("pcen", "logmel")):
        c = dict(VERIFIER=v, USE_TIMEFILTER=t, USE_HARD_NEG=hn, FEATURE=ft)
        out.append((_tag(c), c))
    return out

old6 = [c for _, c in factorial_OLD()[:6]]
new6 = [c for _, c in F16[:6]]
check("OLD ordering: first 6 configs contain ZERO proto (the actual failure)",
      not any(c["VERIFIER"] == "proto" for c in old6))
check("NEW ordering: first 6 configs DO contain proto",
      any(c["VERIFIER"] == "proto" for c in new6))
first5 = [c for _, c in F16[:5]]
for factor, off in (("VERIFIER", "logreg"), ("USE_TIMEFILTER", False),
                    ("USE_HARD_NEG", False), ("FEATURE", "pcen")):
    check(f"first 5 configs cover the '{factor}' main effect",
          any(c[factor] != off for c in first5))

print("\n[checkpoint / resume]")
def build_plan(configs, baseline_label, baseline_seeds, other_seed):
    plan = []
    for label, cfg in configs:
        for es in (baseline_seeds if label == baseline_label else (other_seed,)):
            plan.append((label, cfg, es))
    return plan

plan = build_plan(F8, "pcen|logreg", (0, 1, 2), 0)
check("baseline repeated across all encoder seeds (noise floor)",
      sum(1 for l, _, _ in plan if l == "pcen|logreg") == 3, str(len(plan)))
check("non-baseline configs run once each",
      all(sum(1 for l, _, _ in plan if l == lab) == 1
          for lab in {l for l, _ in F8} - {"pcen|logreg"}))
check("total planned runs = 8 configs + 2 extra baseline = 10", len(plan) == 10, str(len(plan)))

def filter_todo(plan, done):
    return [(l, c, e) for (l, c, e) in plan if (l, int(e)) not in done]
check("empty checkpoint -> run everything", len(filter_todo(plan, set())) == 10)
done_partial = {("pcen|logreg", 0), ("pcen|logreg", 1), ("logmel|logreg", 0)}
check("partial checkpoint -> resumes remainder",
      len(filter_todo(plan, done_partial)) == 7, str(len(filter_todo(plan, done_partial))))
check("resume never re-runs a completed pair",
      all((l, int(e)) not in done_partial for l, _, e in filter_todo(plan, done_partial)))
check("full checkpoint -> nothing to do",
      len(filter_todo(plan, {(l, int(e)) for l, _, e in plan})) == 0)

print("\n[noise floor gating]")
def verdict(delta, noise):
    return "within noise" if abs(delta) < noise else "REAL"
NOISE = 0.035   # measured: same config scored 0.696 and 0.731 across two sessions
check("proto delta (-0.005) judged within noise", verdict(-0.005, NOISE) == "within noise")
check("tfa delta (-0.155) judged REAL",           verdict(-0.155, NOISE) == "REAL")
check("feature delta (-0.148) judged REAL",       verdict(-0.148, NOISE) == "REAL")
check("hard-neg delta (-0.047) judged REAL but marginal",
      verdict(-0.047, NOISE) == "REAL" and abs(-0.047) < 2 * NOISE)

# ---- main-effect weighting (the 28 Aug 2026 bug) ------------------------------------
# The baseline config is repeated once per BASELINE_ENCODER_SEEDS to measure the noise
# floor, so it carries 3x the rows of every challenger. A row-weighted
# groupby("species").mean() therefore counts the (highest-scoring) baseline three times
# inside the "off" arm of VERIFIER, USE_HARD_NEG and FEATURE(pcen), inflating every
# deficit. Real numbers from data/argus_ablation_ladder.csv.
print("\n[main-effect weighting -- each config must count once]")

def main_effect(rows, flag, on_vals, off_vals, weighting):
    """rows: list of dicts with species/config/auc_ver/<flag>. Returns (mean_delta, d_over_se)."""
    import statistics
    def agg(sub):
        by_sp = {}
        if weighting == "row":                      # the shipped (buggy) path
            for r in sub:
                by_sp.setdefault(r["species"], []).append(r["auc_ver"])
            return {s: sum(v)/len(v) for s, v in by_sp.items()}
        cfg = {}                                     # config-weighted (correct)
        for r in sub:
            cfg.setdefault((r["species"], r["config"]), []).append(r["auc_ver"])
        cfg = {k: sum(v)/len(v) for k, v in cfg.items()}
        for (s, _c), v in cfg.items():
            by_sp.setdefault(s, []).append(v)
        return {s: sum(v)/len(v) for s, v in by_sp.items()}
    on, off = agg([r for r in rows if r[flag] in on_vals]), agg([r for r in rows if r[flag] in off_vals])
    common = sorted(set(on) & set(off))
    d = [on[s] - off[s] for s in common]
    m = sum(d)/len(d)
    se = statistics.stdev(d)/(len(d) ** 0.5)
    return m, m/se

import csv, os
_LAD = os.path.join(HERE, "..", "data", "argus_ablation_ladder.csv")
if os.path.exists(_LAD):
    with open(_LAD, encoding="utf-8") as fh:
        _rows = [{**r, "auc_ver": float(r["auc_ver"])} for r in csv.DictReader(fh)]
    counts = {}
    for r in _rows: counts[r["config"]] = counts.get(r["config"], 0) + 1
    base_n = counts.get("pcen|logreg", 0)
    other_n = max(v for k, v in counts.items() if k != "pcen|logreg")
    check("the baseline really is over-represented (this is what made it a bug)",
          base_n == 3 * other_n, f"baseline {base_n} rows vs {other_n} for others")

    for flag, on, off, want_row, want_cfg in (
            ("VERIFIER", ["proto"], ["logreg"], -0.0477, -0.0227),
            ("USE_HARD_NEG", ["True"], ["False"], -0.0442, -0.0184),
            ("FEATURE", ["logmel"], ["pcen"], -0.1065, -0.0933)):
        m_row, dse_row = main_effect(_rows, flag, on, off, "row")
        m_cfg, dse_cfg = main_effect(_rows, flag, on, off, "config")
        check(f"REGRESSION GUARD: row-weighting reproduces the published {flag} effect "
              f"({want_row:+.4f})", abs(m_row - want_row) < 5e-4, f"{m_row:+.4f}")
        check(f"config-weighting gives the corrected {flag} effect ({want_cfg:+.4f})",
              abs(m_cfg - want_cfg) < 5e-4, f"{m_cfg:+.4f}")
        check(f"{flag}: row-weighting inflated the deficit", abs(m_row) > abs(m_cfg))

    # The consequence that reached the site: proto crossed the project's own evidence bar.
    _, dse_row = main_effect(_rows, "VERIFIER", ["proto"], ["logreg"], "row")
    _, dse_cfg = main_effect(_rows, "VERIFIER", ["proto"], ["logreg"], "config")
    check("REGRESSION GUARD: row-weighted proto passed the |d/SE|>2 bar (it was published)",
          abs(dse_row) > 2, f"{dse_row:.2f}")
    check("config-weighted proto FAILS the project's own |d/SE|>2 bar",
          abs(dse_cfg) < 2, f"{dse_cfg:.2f}")
    check("config-weighted FEATURE still clears the bar (this effect is real)",
          abs(main_effect(_rows, "FEATURE", ["logmel"], ["pcen"], "config")[1]) > 2)
else:
    check("ladder CSV present for the weighting regression test", False, _LAD)

# The source must no longer use the row-weighted path.
_ld = open(os.path.join(HERE, "arguswala_ladder.py"), encoding="utf-8").read()
check("arguswala_ladder.py collapses per (species, config) before averaging",
      'groupby(["species", "config"]).auc_ver.mean().groupby("species").mean()' in _ld)
check("the bare row-weighted groupby is gone",
      'g_on.groupby("species").auc_ver.mean()' not in _ld)

print("\n" + "="*62)
print(f"{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILURES: ' + ', '.join(FAILS)}")
print("="*62)
