# ===== CELL 2 v2 / TWO-STAGE across N species =====
# Needs CELL 1 v2 (arguswala_setup_v2.py).
#
# WHAT CHANGED vs v1:
#
#  (1) THE BUG. v1 line 216 read:
#          sps = [s for s in meta.primary_label.unique() if s != "whbsho3"]
#      The target was hard-coded while CELL 1 defined it as a variable. The moment a
#      second target is evaluated, that line puts the new target INTO the encoder's
#      training bank -- the encoder is pre-trained on the species it must never have
#      seen, every number inflates, and nothing crashes. Now excludes all of TARGET_SET,
#      with an assertion so it can never regress silently.
#
#  (2) ONE ENCODER, MANY TARGETS. Legitimate because no target is in the bank, and it
#      turns O(species x encoder-trainings) into O(1).
#
#  (3) PURE-NEGATIVE CONTROL (neg_pass). Nothing in v1 ever ran on a recording with zero
#      target calls -- in deployment that is the common case, and false alarms per hour
#      on clean audio is the first number a conservation biologist asks for.
#
#  (4) FEATURE TOGGLE. Liang et al. (DCASE 2024 baseline) Table III: PCEN gives the best
#      PRECISION of any front-end tested (68.0%) while log-Mel gives the best F1 (63.7 vs
#      60.0). v1 assumed PCEN. Now it is a measured choice.
#
#  (5) COVARIATES + per-species results written to CSV, so the difficulty-factor analysis
#      Nolasco et al. attempted and failed can actually be run afterwards.
import numpy as np, os, glob, copy, time, librosa, torch, torch.nn as nn, torch.optim as optim
import pandas as pd

SR, N_FFT, HOP, N_MELS = 22050, 1024, 256, 128   # identical to the DCASE 2024 baseline
                                                 # (22.05kHz / 1024 / 256 / 128) -- keep it
                                                 # that way, it is what makes a DCASE run cheap
SEG_SEC = 1.0; SEG_FR = int(np.ceil(SEG_SEC*SR/HOP)); CLEN = int(SEG_SEC*SR); EMB = 128
FT_STEPS, FT_LR, FT_BATCH = 250, 1e-4, 16
N_INJECT, N_NEG, INJ_SNR = 45, 200, (-6, 12)
CAND_FLOOR, STRIDE, IOU_THR = 0.10, max(1, SEG_FR//3), 0.30
MAX_EVENT_SEC = 10.0
CALLS_PER_BED = 12

FEATURE = "pcen"          # 'pcen' | 'logmel'
SEEDS   = (7, 8, 9)       # raise to >=10 once the timing measurement says it fits
ENCODER_SEEDS = (0, 1, 2) # >1 entry measures encoder-training variance (roadmap 3.4).
                          # Widened 14 Aug 2026 after the first real run: whbsho3 stage-1
                          # AUC came back 0.517 here vs. the earlier single-target baseline
                          # of 0.459. This tests ONE candidate cause -- training-run
                          # stochasticity on a FIXED bank (build_banks() always uses
                          # seed=0 regardless of this tuple, so bank COMPOSITION doesn't
                          # vary here). A separate single-target comparison run is still
                          # the only way to test the other candidate cause (bank
                          # composition differs because this run excludes 2 species from
                          # the encoder bank, the original baseline only excluded 1).

# 18 Aug 2026: a Kaggle GPU-quota cutoff killed a run mid-way through CELL 4 after
# CELL 2's 5.5-hour species loop had already completed and its CSV was safely
# downloaded. Restarting the whole notebook to reach CELL 4 again would re-pay that
# 5.5 hours for a result already in hand. SKIP_MAIN_LOOP=True runs bank-building and
# encoder training (~10-20 min, everything CELL 4 actually needs) and stops there --
# `res`/`wide`/the printed report below never run, and never need to. CELL 4's own
# checkpointing (RESUME=True) then picks up the ladder at the first unfinished config.
# Leave False for a normal full run.
SKIP_MAIN_LOOP = False

# ---------------------------------------------------------------- ablation ladder
# Stage-1 rungs. Both default OFF so rung 0 reproduces the original pipeline exactly.
# (Stage-2 rung lives in CELL 1 as VERIFIER = 'logreg' | 'proto'.)
USE_TIMEFILTER = False    # rung 2: TimeFilterAug on the support during fine-tuning
USE_HARD_NEG   = False    # rung 3: hard-negative mining in the fine-tuning loop
HARD_NEG_KEEP  = 0.25     # fraction of candidate negatives retained as "hard"

# rung 5 (CELL 5, confound-closing run 2 -- roadmap "owed" list): stage-1's per-target
# fine-tuning has always drawn its negatives as arbitrary random crops of the bed
# (whatever silence/wind/distant noise happens to be there) -- never a labelled OTHER
# SPECIES call. The stage-2 Perch verifier already trains against real other-species
# calls (calib_neg_species in build_target); stage 1 never has. Default OFF reproduces
# every previously-published number byte-for-byte. When True, stage1() injects real
# calls from ctx['neg_calls'] (the SAME non-decoy species pool the verifier uses) into a
# copy of the clean recording and crops THOSE positions as negatives -- but N_NEG=200
# 1s calls with 0.4s gaps need >=280s in a 240s bed, which is geometrically impossible
# (measured: rejection-sampling places ~126 of the 200 target, min 122/max 130 over 10
# draws), so on THIS side the "instead of arbitrary background" framing overclaims --
# ~74 of the ~200 negatives are still background top-up on every call (see the top-up
# branch in stage1() below). What this rung actually tests is "~126 species-matched +
# ~74 background" vs "~200 background", not a clean substitution. Restated honestly in
# both places 28 Aug 2026, after an audit caught the original wording.
USE_SPECIES_NEG = False

# CELL 5, confound-closing run 1: how far a sliding Perch-embedding + verifier scan
# steps between windows when scoring a recording with NO stage-1 CNN in the loop at all
# (detect_perch_only below). Perch inference is the pipeline's dominant per-call cost, so
# this trades runtime against localisation precision; narrow it only after the first
# timing print says the wider stride is affordable.
PERCH_ALONE_STRIDE_S = 0.5

def timefilteraug(x, rng, m_range=(3, 6), lo_db=-6.0, hi_db=8.0):
    """TimeFilterAug -- Zou et al. 2024 (DCASE 2023 Task 5, 1st place, 63.8% F), eq. 6.

    Motivation, in their words: the five shots "are typically clear, whereas the query set
    for predictions often contains interference noise, predominantly from far-field sound
    and background impulse noise." This simulates that by applying a random piecewise-
    LINEAR gain envelope across TIME (not a mask, and not across frequency):

      partition the window into m in [3,6] segments; draw a breakpoint gain g_i ~ U[0,1]
      per boundary; within each segment interpolate linearly between adjacent g's; map
      that ramp onto [lo_db, hi_db] = [-6, +8] dB; apply.

    Their front-end is PCEN at 22050 Hz / n_fft 1024 / hop 256 -- identical to ARGUS, so
    the dB bounds transfer directly rather than needing rescaling.
    """
    T = x.shape[1]
    m = int(rng.integers(m_range[0], m_range[1] + 1))
    if T < m + 1:
        return x
    cuts  = np.sort(rng.choice(np.arange(1, T), size=m - 1, replace=False))
    edges = np.concatenate([[0], cuts, [T]])
    g = rng.random(m + 1)
    alpha = np.concatenate([
        np.linspace(g[i], g[i + 1], int(edges[i + 1] - edges[i]), endpoint=False)
        for i in range(m)
    ])
    beta = (lo_db + (hi_db - lo_db) * alpha).astype("float32")   # per-frame gain in dB
    if FEATURE == "logmel":
        # amplitude_to_db output is ALREADY logarithmic -> a dB gain is additive.
        return x + beta[np.newaxis, :]
    # PCEN is a linear-domain magnitude -> a dB gain is multiplicative.
    return x * (10.0 ** (beta / 20.0))[np.newaxis, :].astype("float32")

def aug_pos(x, rng):
    """Augmentation applied to positives (the injected support) during fine-tuning."""
    x = specaug(x, rng)
    if USE_TIMEFILTER:
        x = timefilteraug(x, rng)
    return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device:", device)

def rms(x): return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))
def ld(p, dur=None):
    y, _ = librosa.load(p, sr=SR, mono=True, duration=dur); return y.astype("float32")

FEATURES = ("pcen", "logmel")   # every front-end an encoder is trained for

def feat(y, feature=None):
    """Front-end. PCEN vs log-Mel is an experiment, not an assumption.

    `feature=None` reads the FEATURE global -- that is what makes the ladder able to
    switch front-ends by rebinding a global. Pass it explicitly when featurising for a
    front-end other than the currently-selected one (e.g. building both encoder banks).
    """
    feature = feature or FEATURE
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, power=1.0)
    if feature == "pcen":
        return librosa.pcen(m * (2**31), sr=SR, hop_length=HOP).astype("float32")
    if feature == "logmel":
        return librosa.amplitude_to_db(m, ref=np.max).astype("float32")
    raise ValueError(f"unknown feature: {feature}")

def crop(f, c):
    a = c - SEG_FR//2; p = np.full((N_MELS, SEG_FR), f.min(), dtype="float32")
    lo, hi = max(0, a), min(f.shape[1], a+SEG_FR)
    if hi > lo: p[:, (lo-a):(lo-a)+(hi-lo)] = f[:, lo:hi]
    return p
def loud_off(y, n=CLEN):
    if len(y) <= n: return 0
    e = np.square(y, dtype=np.float64); cs = np.concatenate([[0.0], np.cumsum(e)])
    return int(np.argmax(cs[n:]-cs[:-n]))
def specaug(x, rng):
    if rng.random() >= 0.6: return x
    x = x.copy(); fl = x.min()
    for _ in range(2):
        k = int(rng.integers(0, 21))
        if k: a = int(rng.integers(0, max(1, N_MELS-k))); x[a:a+k, :] = fl
        k = int(rng.integers(0, 11))
        if k: a = int(rng.integers(0, max(1, SEG_FR-k))); x[:, a:a+k] = fl
    return x
def inject(bed, calls, n, snr, rng, gap_s=0.4):
    y = bed.astype("float32").copy(); L = len(y); gap = int(gap_s*SR); sp = []
    if L <= CLEN: return y, sp
    for _ in range(n*60):
        if len(sp) >= n: break
        pos = int(rng.integers(0, L-CLEN))
        if any(pos < e+gap and pos+CLEN > s-gap for s, e in sp): continue
        c = calls[int(rng.integers(len(calls)))][:CLEN]
        if len(c) < CLEN: c = np.pad(c, (0, CLEN-len(c)))
        g = rms(y[pos:pos+CLEN])*(10**(float(rng.uniform(*snr))/20))/(rms(c)+1e-9)
        y[pos:pos+CLEN] += c.astype("float32")*g; sp.append((pos, pos+CLEN))
    return y, sorted(sp)
def iou(a, b):
    it = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1]-a[0]) + (b[1]-b[0]) - it
    return it/u if u > 0 else 0.0

class Enc(nn.Module):
    def __init__(s, emb=EMB):
        super().__init__()
        def B(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                          nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.e = nn.Sequential(B(1,128), B(128,128), B(128,128), B(128,emb)); s.p = nn.AdaptiveAvgPool2d(1)
    def forward(s, x): return s.p(s.e(x.unsqueeze(1))).flatten(1)

def train_enc(bank, rng, episodes=1000, nway=6, k=4, q=4):
    m = Enc().to(device); opt = optim.Adam(m.parameters(), 1e-3); ce = nn.CrossEntropyLoss()
    cl = list(bank); m.train()
    for ep in range(episodes):
        ch = rng.choice(cl, size=min(nway, len(cl)), replace=False)
        sx, sy, qx, qy = [], [], [], []
        for lab, c in enumerate(ch):
            idx = rng.permutation(len(bank[c]))[:k+q]
            for j, i in enumerate(idx):
                (sx if j < k else qx).append(torch.tensor(specaug(bank[c][i], rng)))
                (sy if j < k else qy).append(lab)
        sx = torch.stack(sx).to(device); qx = torch.stack(qx).to(device)
        sy = torch.tensor(sy).to(device); qy = torch.tensor(qy).to(device)
        se, qe = m(sx), m(qx)
        pr = torch.stack([se[sy == c].mean(0) for c in torch.unique(sy)])
        loss = ce(-torch.cdist(qe, pr), qy); opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 250 == 0: print(f"    enc ep {ep+1}/{episodes} loss {loss.item():.3f}")
    return m

def stitch(frames, probs):
    ab = probs >= CAND_FLOOR; ev = []; i = 0
    while i < len(frames):
        if not ab[i]: i += 1; continue
        j = i
        while j+1 < len(frames) and ab[j+1] and frames[j+1]-frames[j] <= STRIDE*1.5: j += 1
        pk = i + int(np.argmax(probs[i:j+1]))
        s = max(0.0, frames[i]*HOP/SR - SEG_SEC/2); e = frames[j]*HOP/SR + SEG_SEC/2
        if e - s > MAX_EVENT_SEC:
            c = frames[pk]*HOP/SR; s, e = max(0.0, c-SEG_SEC/2), c+SEG_SEC/2
        ev.append((s, e, float(probs[pk]))); i = j+1
    return ev

def stitch_time(times, probs, floor=CAND_FLOOR, gap_s=None, max_event_sec=MAX_EVENT_SEC,
                 seg_sec=SEG_SEC):
    """Same merge-adjacent-above-floor logic as stitch(), but for detectors that never
    build a mel spectrogram (CELL 5's Perch-alone-as-detector), so positions are already
    in seconds rather than STRIDE-spaced spectrogram frame indices. `times` must be sorted
    ascending. gap_s defaults to a spacing consistent with stitch()'s STRIDE*1.5 in frames,
    converted to seconds."""
    gap_s = gap_s if gap_s is not None else (STRIDE * HOP / SR) * 1.5
    times = np.asarray(times, dtype="float64")
    ab = np.asarray(probs) >= floor; ev = []; i = 0
    while i < len(times):
        if not ab[i]: i += 1; continue
        j = i
        while j+1 < len(times) and ab[j+1] and times[j+1]-times[j] <= gap_s: j += 1
        pk = i + int(np.argmax(probs[i:j+1]))
        s = max(0.0, times[i] - seg_sec/2); e = times[j] + seg_sec/2
        if e - s > max_event_sec:
            c = times[pk]; s, e = max(0.0, c - seg_sec/2), c + seg_sec/2
        ev.append((s, e, float(probs[pk]))); i = j+1
    return ev

def stage1(enc, rec, sup, rng, neg_species_calls=None):
    n_inj = max(8, int(round(N_INJECT * len(rec) / (240.0 * SR))))
    aw, sp = inject(rec, sup, n_inj, INJ_SNR, rng)
    af, sf = feat(aw), feat(rec)
    pc = [((a+b)//2)//HOP for a, b in sp]
    if not pc: return []
    pos = [crop(af, c) for c in pc]
    g = SEG_FR; hi = af.shape[1]-g

    def _background_neg():
        out = []
        for _ in range(N_NEG*40):
            if len(out) >= N_NEG: break
            f = int(rng.integers(g, max(g+1, hi)))
            if all(abs(f-p) > SEG_FR for p in pc): out.append(crop(af, f))
        return out

    if USE_SPECIES_NEG and neg_species_calls:
        # Inject real OTHER-SPECIES calls into a FRESH copy of the clean recording (not
        # `aw`, which already carries the positive injections -- a separate copy means
        # positive and negative spans can never collide or corrupt each other) and crop
        # those exact centres as negatives. N_NEG=200 1s calls at 0.4s gaps cannot all
        # fit in a 240s bed (needs >=280s) -- inject() places ~126 of them, and the
        # len(neg)<N_NEG branch below tops up the rest with background. This arm is
        # therefore "~126 species-matched + ~74 background", not a clean 200-call
        # substitution -- see USE_SPECIES_NEG's definition above for the real numbers.
        nw, nsp = inject(rec, neg_species_calls, N_NEG, INJ_SNR, rng)
        nf = feat(nw)
        nc = [((a+b)//2)//HOP for a, b in nsp]
        neg = [crop(nf, c) for c in nc if g <= c <= nf.shape[1]-g]
        if len(neg) < N_NEG:
            # short recording / crowded bed couldn't place enough species-matched calls --
            # top up with background rather than silently training on fewer negatives.
            neg += _background_neg()
    else:
        neg = _background_neg()
    if not neg:
        neg = [np.full((N_MELS, SEG_FR), af.min(), dtype="float32")]
    m = copy.deepcopy(enc).to(device); h = nn.Linear(EMB, 2).to(device)
    opt = optim.Adam(list(m.parameters())+list(h.parameters()), FT_LR)
    ce = nn.CrossEntropyLoss(); m.train(); h.train(); hf = FT_BATCH//2

    def _train(steps, neg_pool):
        for _ in range(steps):
            xs  = [aug_pos(pos[int(rng.integers(len(pos)))], rng) for _ in range(hf)]
            xs += [neg_pool[int(rng.integers(len(neg_pool)))] for _ in range(hf)]
            xb = torch.tensor(np.stack(xs)).to(device)
            yb = torch.tensor([1]*hf+[0]*hf).to(device)
            loss = ce(h(m(xb)), yb); opt.zero_grad(); loss.backward(); opt.step()

    def _score(items):
        m.eval(); h.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(items), 256):
                b = torch.tensor(np.stack(items[i:i+256])).to(device)
                out.append(torch.softmax(h(m(b)), 1)[:, 1].cpu().numpy())
        m.train(); h.train()
        return np.concatenate(out)

    if USE_HARD_NEG and len(neg) >= 8:
        # Negative choice is named as decisive twice in the literature: Nolasco et al.
        # sec.4.2 ("prototype-based meta-learning works well when taking care about ...
        # the choice of negative examples") and Liang et al., who measure +5.31 F1 from
        # negative hard sampling. Random background windows are mostly trivial negatives
        # -- silence and wind -- so half the budget is spent learning nothing. Train
        # briefly, find the background the model currently CONFUSES with the target, and
        # spend the rest of the budget there.
        half = FT_STEPS // 2
        _train(half, neg)
        k = max(4, int(round(len(neg) * HARD_NEG_KEEP)))
        hard = [neg[i] for i in np.argsort(-_score(neg))[:k]]
        _train(FT_STEPS - half, hard)
    else:
        _train(FT_STEPS, neg)
    m.eval(); h.eval()
    fr = list(range(g, max(g+1, sf.shape[1]-g), STRIDE)); pb = np.empty(len(fr), dtype="float32")
    with torch.no_grad():
        for i in range(0, len(fr), 256):
            b = torch.tensor(np.stack([crop(sf, f) for f in fr[i:i+256]])).to(device)
            p = torch.softmax(h(m(b)), 1)[:, 1]; pb[i:i+len(p)] = p.cpu().numpy()
    return stitch(fr, pb)

def detect_pv(enc, ctx, rec, rng):
    """-> [(start, end, p, v)]. Target context is now explicit, not a global."""
    ev = stage1(enc, rec, ctx["support"], rng, neg_species_calls=ctx.get("neg_calls"))
    if not ev: return []
    centres = [(s + e) / 2 for s, e, _ in ev]
    # verifier_embed_ctx (CELL 1) dispatches on EMBED_BACKEND. build_target() FITS
    # ctx["perch_clf"] through the same function, so the classifier is always scored in the
    # space it was fit in -- under EMBED_BACKEND="distilled" no Perch inference runs here.
    emb = l2n(verifier_embed_ctx(rec, centres))
    v = ctx["perch_clf"].predict_proba(emb)[:, 1]
    return [(s, e, p, float(vi)) for (s, e, p), vi in zip(ev, v)]

def detect_perch_only(enc, ctx, rec, rng, stride_s=None):
    """-> [(start, end, p, v)], p := v. CELL 5, confound-closing run 1: "what did the CNN
    contribute?" No stage-1 encoder, no mel front-end, no fine-tuning loop anywhere in
    this path -- Perch embeddings are computed directly on a sliding window across the
    raw recording and scored with the SAME calibrated verifier (ctx['perch_clf']) every
    other detector in this file uses, then merged into events with stitch_time(). `enc`
    and `rng` are accepted only for call-site parity with detect_pv (so eval_pass /
    spec_pass / neg_pass can swap detectors via one `detect_fn` parameter) -- this path
    is deterministic given `rec`, so both are unused. p is set equal to v so the FUSIONS
    machinery still works mechanically, but only the "perch only (v)" fusion is a
    meaningful score here -- any p-dependent fusion rule applied to this detector's
    output is measuring nothing, since p carries no independent information."""
    stride_s = stride_s if stride_s is not None else PERCH_ALONE_STRIDE_S
    L = len(rec) / SR
    times = ([L / 2.0] if L <= SEG_SEC else
             list(np.arange(SEG_SEC / 2.0, L - SEG_SEC / 2.0 + 1e-9, stride_s)))
    if not times: return []
    emb = l2n(perch_embed_ctx(to32k(rec), times))
    v = ctx["perch_clf"].predict_proba(emb)[:, 1]
    # max_event_sec=SEG_SEC and an explicit gap_s are BOTH load-bearing; without them this
    # detector is at serious risk of scoring zero true positives in its normal operating
    # range, and failing silently in the direction of our own pre-registered prediction.
    # Geometry: Perch's window is 5 s, so a 1 s call sits inside it for ~11 consecutive
    # 0.5 s-strided positions, and if all of them clear CAND_FLOOR, stitch_time merges them
    # (0.5 < the default gap of 0.505 s) into one event spanning 0.5(R-1)+1 = 6 s. Truth
    # spans are 1 s, so IoU = 1/6 = 0.17, under IOU_THR=0.30 -- no match. Every merged-run
    # length R in [6,19] is unmatchable this way, and the default MAX_EVENT_SEC=10 only
    # rescues R>=20 (truncated to the peak window). This is a geometric hazard, not a
    # proven outcome -- whether real above-floor runs actually land in [6,19] depends on
    # the empirical density of the score field, which was never measured before this fix
    # landed (found by audit 28 Aug 2026, before Cell 5 ever ran). If it did land there
    # for most calls, the failure mode is silent and severe: near-zero true positives push
    # thr toward 1.0, drive FA/h toward 0 and AUC toward the 0.500 all-ties value, which
    # verdict() would report as "Perch alone is WORSE -- stage-1's localisation is doing
    # real work": our own pre-registered prediction, confirmed by an artefact rather than
    # a result. Capping at SEG_SEC removes the hazard rather than characterising it: it
    # collapses every run to a 1 s span at its peak window (R=1 is already 1 s and passes
    # through untouched), which is also the honest semantic -- a 5 s window cannot localise
    # better than its argmax. gap_s is passed explicitly so widening stride_s past 0.505 s
    # cannot silently flip merging off. Found by audit 28 Aug 2026, before this ever ran.
    ev = stitch_time(np.asarray(times), v, gap_s=max(stride_s * 1.5, 1e-6),
                     max_event_sec=SEG_SEC)
    return [(s, e, vi, vi) for s, e, vi in ev]

FUSIONS = {
    "stage-1 only  (p)":  lambda p, v: p,
    "perch only    (v)":  lambda p, v: v,
    "p * v":              lambda p, v: p * v,
    "p * sqrt(v)":        lambda p, v: p * np.sqrt(v),
    "sqrt(p) * v":        lambda p, v: np.sqrt(p) * v,
    "geo mean sqrt(p*v)": lambda p, v: np.sqrt(p * v),
}

# ------------------------------------------------------------------- metrics
def match(events, truth, iou_thresh):
    pairs = sorted(((iou(e[:2], t), i, j) for i, e in enumerate(events)
                    for j, t in enumerate(truth)), reverse=True)
    used_e, used_t = set(), set()
    for score, i, j in pairs:
        if score <= iou_thresh: break
        if i in used_e or j in used_t: continue
        used_e.add(i); used_t.add(j)
    return [(e[2], i in used_e) for i, e in enumerate(events)]

def match_truth(events, truth, iou_thresh):
    pairs = sorted(((iou(e[:2], t), i, j) for i, e in enumerate(events)
                    for j, t in enumerate(truth)), reverse=True)
    ue, ut, out = set(), set(), [0.0] * len(truth)
    for score, i, j in pairs:
        if score <= iou_thresh: break
        if i in ue or j in ut: continue
        ue.add(i); ut.add(j); out[j] = events[i][2]
    return out

def fp_per_tp(scored, n_truth, target_recall=0.5):
    arr = sorted(scored, key=lambda x: -x[0]); tp = fp = 0
    for score, hit in arr:
        if hit: tp += 1
        else:   fp += 1
        if tp / n_truth >= target_recall:
            return fp / tp if tp else float('inf')
    return float('inf')

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
        # Recall-first operating point (N4): the HIGHEST threshold still achieving >=90%
        # recall, i.e. best precision subject to a recall floor. Scores descend, so the
        # first crossing is that threshold. Setting it to the score just BEFORE the
        # crossing (the earlier bug) yields a point that never actually reaches 90%.
        if thr_r90 is None and rc >= 0.90: thr_r90 = score
    if thr_r90 is None: thr_r90 = arr[-1][0]   # 90% unreachable -> loosest available
    return dict(ap=ap, f1=best[0], p=best[1], r=best[2], thr=best[3],
                thr_r90=thr_r90, n=len(scored))

def _auc(pos, neg):
    """P(random target scores above random decoy), ties 0.5. Threshold-FREE.
    Same metric family BirdCLEF 2024 adopted as its official score (macro ROC-AUC):
    0.50 = cannot tell target from decoy, 1.00 = perfect species discrimination."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg): return float('nan')
    gt = float((pos[:, None] > neg[None, :]).sum())
    eq = float((pos[:, None] == neg[None, :]).sum())
    return (gt + 0.5 * eq) / (len(pos) * len(neg))

# ------------------------------------------------------------------- passes
# detect_fn is swappable (default detect_pv) so CELL 5 can run the identical planting /
# matching logic against a different detector -- e.g. detect_perch_only -- and get a
# directly comparable number rather than a separately-implemented, possibly-diverging one.
def eval_pass(enc, ctx, seed, detect_fn=detect_pv):
    r = np.random.default_rng(seed); cache = []
    for bed in eval_beds:
        y = bed.copy(); L = len(y); spans, planted, snrs = [], [], []
        for k in range(CALLS_PER_BED):
            snr = SNR_LEVELS[k % len(SNR_LEVELS)]
            placed = None
            for _ in range(80):
                pos = int(r.integers(0, L-CLEN))
                if any(pos < e+int(0.4*SR) and pos+CLEN > s-int(0.4*SR) for s, e in spans): continue
                placed = pos; break
            if placed is None: continue
            c = ctx["target_pool"][int(r.integers(len(ctx["target_pool"])))][:CLEN]
            if len(c) < CLEN: c = np.pad(c, (0, CLEN-len(c)))
            g = rms(y[placed:placed+CLEN])*(10**(snr/20))/(rms(c)+1e-9)
            y[placed:placed+CLEN] += c.astype("float32")*g
            spans.append((placed, placed+CLEN))
            planted.append((placed/SR, (placed+CLEN)/SR)); snrs.append(snr)
        cache.append((detect_fn(enc, ctx, y, r), planted, snrs))
    return cache

def spec_pass(enc, ctx, seed, snr_db=6.0, detect_fn=detect_pv):
    """Specificity control. snr_db is now a parameter: v1 fixed it at +6 dB, so species
    discrimination had only ever been measured on loud, easy calls (roadmap 3.2)."""
    r = np.random.default_rng(seed); cache = []
    for bed in eval_beds:
        y = bed.copy(); L = len(y); spans, items = [], []
        plan = [("target", ctx["target_pool"])]*6
        for n_, p_ in ctx["decoy_pools"].items(): plan += [(n_, p_)]*3
        for kind, pool in plan:
            placed = None
            for _ in range(80):
                pos = int(r.integers(0, L-CLEN))
                if any(pos < e+int(0.4*SR) and pos+CLEN > s-int(0.4*SR) for s, e in spans): continue
                placed = pos; break
            if placed is None or not pool: continue
            c = pool[int(r.integers(len(pool)))][:CLEN]
            if len(c) < CLEN: c = np.pad(c, (0, CLEN-len(c)))
            g = rms(y[placed:placed+CLEN])*(10**(snr_db/20))/(rms(c)+1e-9)
            y[placed:placed+CLEN] += c.astype("float32")*g
            spans.append((placed, placed+CLEN))
            items.append((kind, (placed/SR, (placed+CLEN)/SR)))
        cache.append((detect_fn(enc, ctx, y, r), items))
    return cache

def neg_pass(enc, ctx, seed, detect_fn=detect_pv):
    """PURE NEGATIVE control -- unmodified beds, zero target calls planted.
    Every detection here is a false alarm by construction. This is the deployment-
    relevant number (false alarms per hour of clean audio) and v1 never measured it."""
    r = np.random.default_rng(seed + 500); cache = []
    for bed in eval_beds:
        cache.append(detect_fn(enc, ctx, bed.copy(), r))
    return cache

# ------------------------------------------------------------------- scoring
def score_eval(fuse, cache):
    scored, n_truth, by_snr = [], 0, []
    for dets, planted, snrs in cache:
        ev = [(s, e, float(fuse(p, v))) for s, e, p, v in dets]
        scored += match(ev, planted, IOU_THR); n_truth += len(planted)
        by_snr += list(zip(snrs, match_truth(ev, planted, IOU_THR)))
    m = summarise(scored, n_truth)
    m["fp_per_tp@R50"] = fp_per_tp(scored, n_truth, 0.5)
    m["recall_by_snr"] = {s: float(np.mean([sc >= m["thr"] for ss, sc in by_snr if ss == s]))
                          for s in SNR_LEVELS}
    return m

def score_spec(fuse, thr, cache, decoy_names):
    tg, dc = [], {k: [] for k in decoy_names}
    for dets, items in cache:
        ev = [(s, e, float(fuse(p, v))) for s, e, p, v in dets]
        for kind, span in items:
            best = max((iou(e[:2], span), e[2]) for e in ev) if ev else (0.0, 0.0)
            sc = best[1] if best[0] > IOU_THR else 0.0
            (tg if kind == "target" else dc[kind]).append(sc)
    t = float(np.mean([s >= thr for s in tg])) if tg else 0.0
    d = {k: float(np.mean([s >= thr for s in v])) for k, v in dc.items() if v}
    md = float(np.mean(list(d.values()))) if d else 0.0
    all_dc = [s for v in dc.values() for s in v]
    return t, md, (md/t if t else float('nan')), d, _auc(tg, all_dc)

def spec_coverage(cache):
    """-> fraction of planted items stage 1 LOCALISED at all (IoU > IOU_THR), by kind.

    Only the SNR sweep (roadmap 3.2) needs this, and there it is not optional. score_spec
    assigns 0.0 to any planted item that was never localised, and _auc counts a 0.0-vs-0.0
    pair as a tie worth 0.5. So an AUC drifting toward 0.50 at low SNR is ambiguous between
    two different findings: "target and decoy became indistinguishable" and "nothing was
    detected, so everything tied". Coverage separates them. Reporting the SNR curve without
    it would invite exactly the misreading the curve is meant to settle.

    Fusion-independent by construction -- a fusion rule changes an event's score, never its
    span -- so this is computed once per cache rather than once per fusion rule.
    """
    n = {"target": 0, "decoy": 0}; hit = {"target": 0, "decoy": 0}
    for dets, items in cache:
        for kind, span in items:
            k = "target" if kind == "target" else "decoy"
            n[k] += 1
            best = max((iou((s, e), span) for s, e, _p, _v in dets), default=0.0)
            hit[k] += int(best > IOU_THR)
    return {k: (hit[k]/n[k] if n[k] else float('nan')) for k in n}

def score_spec_detected(fuse, cache):
    """-> (AUC over LOCALISED items only, n_target_localised, n_decoy_localised).

    The companion to spec_coverage, and the direct measurement it was a proxy for.
    score_spec's AUC mixes two effects that pull in the same direction at low SNR: genuine
    species confusion, and stage 1 failing to localise the call at all -- which forces a
    0.0 and drags the AUC toward 0.50 for a reason that has nothing to do with species
    identity. Excluding unlocalised items (rather than scoring them 0.0) separates the two,
    so "specificity degraded" and "detection degraded" can be told apart from the numbers
    instead of argued about afterwards.

    Returns nan when either side is empty -- at low enough SNR that is the honest answer,
    and it must not be silently averaged in as if it were a measurement.
    """
    tg, dc = [], []
    for dets, items in cache:
        ev = [(s, e, float(fuse(p, v))) for s, e, p, v in dets]
        for kind, span in items:
            best = max((iou(e[:2], span), e[2]) for e in ev) if ev else (0.0, 0.0)
            if best[0] <= IOU_THR: continue      # never localised -> excluded, NOT zeroed
            (tg if kind == "target" else dc).append(best[1])
    return _auc(tg, dc), len(tg), len(dc)

def score_neg(fuse, thr, cache):
    """-> false alarms per hour on audio containing no target calls."""
    fa = sum(sum(1 for s, e, p, v in dets if float(fuse(p, v)) >= thr) for dets in cache)
    hours = (len(cache) * BED_SEC) / 3600.0
    return fa / hours if hours else float('nan')

# ------------------------------- stage-1 encoder (ALL targets excluded -- the bug fix)
def build_banks(exclude, features=FEATURES, seed=0, max_species=40):
    """One bank per front-end, sharing a single pass over the audio.

    An encoder trained on PCEN cannot be evaluated on log-Mel inputs, so the front-end
    cannot be a same-encoder ladder rung -- it needs its own encoder. But the expensive
    part is decoding audio, not featurising it, so each recording is loaded ONCE and
    featurised for every front-end. Cost is one extra encoder training, not a second
    pass over the dataset.
    """
    rng = np.random.default_rng(seed)
    sps = [s for s in meta.primary_label.unique() if s not in exclude]
    rng.shuffle(sps)
    banks = {f: {} for f in features}
    for sp in sps:
        if len(banks[features[0]]) >= max_species: break
        segs = {f: [] for f in features}
        for fn in meta[meta.primary_label == sp]["filename"].tolist()[:12]:
            try:
                y = ld(os.path.join(AUD, fn), 15)
                c = loud_off(y)//HOP + SEG_FR//2
                for f in features:
                    segs[f].append(crop(feat(y, f), c))
            except Exception:
                pass
        if len(segs[features[0]]) >= 8:
            for f in features:
                banks[f][sp] = segs[f]
    return banks

print(f"\nbuilding encoder banks for {list(FEATURES)}, excluding all {len(TARGET_SET)} "
      f"targets: {sorted(TARGET_SET)}")
banks = build_banks(TARGET_SET, FEATURES, seed=0)
for _f, _b in banks.items():
    assert not (TARGET_SET & set(_b)), \
        f"LEAKAGE: target species present in {_f} encoder bank: {TARGET_SET & set(_b)}"
    print(f"  {_f:<7} bank: {len(_b)} species, none of them targets (assertion passed)")

# Keyed by (feature, seed): a rung that switches the front-end must switch the encoder
# with it, or it is measuring an encoder/feature mismatch rather than the front-end.
encoders = {}
for _f in FEATURES:
    for es in ENCODER_SEEDS:
        print(f"training stage-1 encoder (feature={_f}, seed={es})...")
        torch.manual_seed(es)
        encoders[(_f, es)] = train_enc(banks[_f], np.random.default_rng(es))

if SKIP_MAIN_LOOP:
    print("SKIP_MAIN_LOOP=True -- stopping after CELL 2's encoder training.")
    print("res/wide/the per-species report do not exist this run. Proceed to CELL 4;")
    print("its own checkpointing resumes the ladder at the first unfinished config.")
else:
    # ------------------------------------------------------------------- main loop
    rows, t_start = [], time.time()
    for sp in TARGET_LIST:
        print(f"\n{'='*78}\nTARGET: {sp}\n{'='*78}")
        t_sp = time.time()
        try:
            ctx = build_target(sp)
        except Exception as ex:
            print(f"  SKIP {sp}: {type(ex).__name__}: {ex}"); continue

        for es in ENCODER_SEEDS:
            enc = encoders[(FEATURE, es)]
            per_seed = {name: [] for name in FUSIONS}
            for sd in SEEDS:
                t0 = time.time()
                ec = eval_pass(enc, ctx, sd)
                sc = spec_pass(enc, ctx, sd)
                nc = neg_pass(enc, ctx, sd)
                for name, fuse in FUSIONS.items():
                    m = score_eval(fuse, ec)
                    t, md, ratio, per, auc = score_spec(fuse, m["thr"], sc, ctx["DECOYS"])
                    per_seed[name].append(dict(
                        ap=m["ap"], f1=m["f1"], p=m["p"], r=m["r"],
                        fptp=m["fp_per_tp@R50"], auc=auc, ratio=ratio,
                        fa_per_hr=score_neg(fuse, m["thr"], nc),
                        fa_per_hr_r90=score_neg(fuse, m["thr_r90"], nc)))
                print(f"  seed {sd} done in {time.time()-t0:.0f}s")

            for name in FUSIONS:
                agg = {k: float(np.mean([d[k] for d in per_seed[name]])) for k in per_seed[name][0]}
                sds = {k + "_sd": float(np.std([d[k] for d in per_seed[name]], ddof=1))
                       if len(SEEDS) > 1 else 0.0 for k in per_seed[name][0]}
                rows.append({**ctx["covariates"], "encoder_seed": es, "fusion": name,
                             "feature": FEATURE, "n_seeds": len(SEEDS),
                             "use_timefilter": USE_TIMEFILTER, "use_hard_neg": USE_HARD_NEG,
                             **agg, **sds})
        print(f"  [{sp}] total {time.time()-t_sp:.0f}s")

    res = pd.DataFrame(rows)
    res.to_csv("/kaggle/working/argus_multispecies_results.csv", index=False)
    print(f"\nwrote {len(res)} rows -> argus_multispecies_results.csv "
          f"| wall clock {(time.time()-t_start)/60:.1f} min")

    # ------------------------------------------------------------------- headline table
    BASE, CAND = "stage-1 only  (p)", "perch only    (v)"
    print(f"\n{'='*100}\nSPECIFICITY AUC BY SPECIES  (threshold-free; 0.50 = target "
          f"indistinguishable from decoys)\n{'='*100}")
    print(f"{'species':<12}{'n_rec':>6}{'decoy_sim':>10}{'stereo':>8}"
          f"{'AUC stage-1':>13}{'AUC verified':>14}{'delta':>8}{'F1 s1':>7}{'F1 ver':>8}")
    print("-"*100)
    # Average over encoder seeds first, so a multi-encoder-seed run reports the mean rather
    # than silently taking whichever row happened to land first.
    NUM = ["auc", "f1", "fa_per_hr", "fa_per_hr_r90", "n_recordings",
           "decoy_sim_top1", "stereotypy"]
    piv = (res[res.fusion.isin([BASE, CAND])]
           .groupby(["species", "fusion"], sort=False)[NUM].mean().reset_index())
    wide = piv.pivot(index="species", columns="fusion")

    for sp in [s for s in TARGET_LIST if s in wide.index]:
        r = wide.loc[sp]
        print(f"{sp:<12}{int(r[('n_recordings', BASE)]):>6}{r[('decoy_sim_top1', BASE)]:>10.3f}"
              f"{r[('stereotypy', BASE)]:>8.3f}{r[('auc', BASE)]:>13.3f}{r[('auc', CAND)]:>14.3f}"
              f"{r[('auc', CAND)] - r[('auc', BASE)]:>+8.3f}"
              f"{r[('f1', BASE)]:>7.3f}{r[('f1', CAND)]:>8.3f}")

    # ---------------------------------------------------------- encoder-seed spread
    # The table above averages over encoder seeds -- which hides exactly the variance
    # ENCODER_SEEDS was widened to measure. Same bank, same data, same eval seeds; the only
    # thing differing is the encoder training run. Anything smaller than this spread is not
    # a real effect, no matter what a within-run SE says about it.
    if len(ENCODER_SEEDS) > 1:
        print(f"\n{'='*100}\nENCODER-SEED SPREAD -- pure training-run variance "
              f"(same bank, same data, {len(ENCODER_SEEDS)} encoder seeds)\n{'='*100}")
        es_piv = (res[res.fusion.isin([BASE, CAND])]
                  .groupby(["species", "fusion", "encoder_seed"])["auc"].mean().reset_index())
        worst = 0.0
        for sp in [s for s in TARGET_LIST if s in set(es_piv.species)]:
            for fus, tag in ((BASE, "stage-1"), (CAND, "verified")):
                g = es_piv[(es_piv.species == sp) & (es_piv.fusion == fus)].sort_values("encoder_seed")
                if g.empty: continue
                v = g.auc.to_numpy(); spread = float(v.max() - v.min())
                worst = max(worst, spread)
                print(f"  {sp:<10} {tag:<9} " + "  ".join(f"{x:.3f}" for x in v)
                      + f"   | spread {spread:+.3f}  sd {v.std(ddof=1):.3f}")
        print(f"\n  Largest spread from encoder training alone: {worst:.3f}")
        print(f"  Treat any AUC difference below ~{worst:.3f} as noise, not signal.")

    if len(wide) > 1:
        d = (wide[("auc", CAND)] - wide[("auc", BASE)]).to_numpy()
        se = d.std(ddof=1)/np.sqrt(len(d))
        print(f"\nACROSS SPECIES (n={len(d)}): mean AUC gain {d.mean():+.3f}, SE {se:.3f}, "
              f"delta/SE {d.mean()/se if se else float('nan'):.2f}   (|d/SE| < 2 is NOT evidence)")
        print(f"  verification helped:            {int((d > 0).sum())}/{len(d)} species")
        print(f"  below chance WITHOUT verifying: "
              f"{int((wide[('auc', BASE)] < 0.5).sum())}/{len(d)} species")
        print(f"  below chance WITH verifying:    "
              f"{int((wide[('auc', CAND)] < 0.5).sum())}/{len(d)} species")
        # The pre-registered outcome test (roadmap sec.2): does the original single-species
        # finding generalise, hold only conditionally, or fail? Reported either way.
        frac = (d > 0).mean()
        print("  -> outcome " + ("A (generalises)" if frac >= 0.8 else
                                 "B (conditional -- find what separates them)" if frac >= 0.4 else
                                 "C/D (does NOT generalise -- rebuild narrative around the boundary)"))

    print(f"\nFALSE ALARMS PER HOUR on clean audio (no target present)")
    print(f"{'species':<12}{'best-F1 thr':>24}{'recall-first thr':>26}")
    for sp in [s for s in TARGET_LIST if s in wide.index]:
        r = wide.loc[sp]
        print(f"  {sp:<10}{r[('fa_per_hr', BASE)]:>10.1f} ->{r[('fa_per_hr', CAND)]:>8.1f}"
              f"{r[('fa_per_hr_r90', BASE)]:>16.1f} ->{r[('fa_per_hr_r90', CAND)]:>8.1f}")
