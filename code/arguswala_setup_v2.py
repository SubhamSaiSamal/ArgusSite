# ===== CELL 1 v2 / SETUP -- multi-species, shot-strategy aware =====
# Replaces arguswala_setup.py. Run BEFORE arguswala_v2.py.
#
# WHAT CHANGED vs v1, and why:
#
#  (1) MULTI-TARGET. v1 hard-wired one TARGET. Everything per-species is now behind
#      build_target(sp), so the same pipeline evaluates N species. This pays the debt
#      arguswala.py:375 admits to ("one target species is not a validated finding").
#
#  (2) SPECIES CACHE COMPUTED ONCE. v1 embedded 30 candidate species per target. With
#      10 targets that is 10x redundant Perch inference -- the dominant cost. The cache
#      is now global; build_target() only ranks against it. ~10x saving on setup.
#
#  (3) SHOT-SELECTION STRATEGY is now a variable, not an assumption. v1 always took the
#      5 highest-`rating` recordings. DCASE always takes the first 5. Nolasco et al.
#      (2023) sec.4.1 report that unrepresentative support sets were the dominant failure
#      cause their expert annotators identified, and call better selection an open
#      problem. So it becomes an axis we measure instead of a default we inherit.
#
#  (4) PER-SPECIES COVARIATES are recorded at build time (decoy proximity, stereotypy,
#      spectral centre/bandwidth, call duration, n recordings). Nolasco et al. tried
#      multivariate regression to find what predicts task difficulty and FAILED to find
#      any consistent factor. These covariates are free at runtime and impossible to
#      reconstruct afterwards -- logging them is what makes that analysis possible later.
#
#  (5) The held-out pool is now derived from the shots actually chosen, not from a
#      positional slice. Under a non-'first' strategy, files[N_SHOT:] would have leaked
#      support recordings into the evaluation pool.
import os, glob, time
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# ---------------------------------------------------------------- ablation ladder
# Each rung is independently switchable so marginal contributions are measurable.
# Rung 0 is the original pipeline; nothing below changes it when all flags are False.
VERIFIER = "logreg"        # 'logreg' (rung 0) | 'proto' (rung 1, stage 2)

# ------------------------------------------------------------------ configuration
SR = 22050
SEG_SEC = 1.0
CLEN = int(SEG_SEC * SR)

N_SHOT          = 5
SHOT_STRATEGY   = "rated"     # 'rated' | 'first' | 'random' | 'diverse'
N_DECOYS        = 4
N_CANDIDATES    = 30          # non-target species considered for decoy ranking
CALIB_PER_SPECIES = 6
DECOY_RECORDINGS  = 20
N_CALIB_BEDS, N_EVAL_BEDS = 4, 10
SNR_LEVELS = (9, 6, 3, 0, -3, -6)

# Targets are PRE-REGISTERED: chosen by a rule, before any result is seen.
# Rule: species with >= MIN_RECS recordings, ranked by recording count, top N_TARGETS,
# with the original study species forced in first. Deterministic -- no cherry-picking.
PRIMARY_TARGET = "whbsho3"    # White-bellied Sholakili, the original study species
N_TARGETS      = 10
MIN_RECS       = N_SHOT + 4   # 5 shots + >=4 genuinely held out

SMOKE_TEST = False            # 15 Aug 2026: timing measured (10 min/species/3seeds),
                              # config locked (PCEN/logreg/no-tfa/no-hn all confirmed
                              # winners in the ladder), noise floor known (0.025-0.035).
                              # Centrepiece run: full N_TARGETS species, no more reason
                              # to stay on the 2-species smoke test.

# Added 14 Aug 2026, after the first real run: whbsho3 stage-1-alone AUC came back 0.517
# in a run that excludes 2 species from the encoder bank (this multi-target pipeline),
# vs. the original single-species baseline of 0.459, which only ever excluded whbsho3.
# ENCODER_SEEDS (widened separately, in CELL 2) tests training-run-to-training-run noise
# on a FIXED bank. This flag tests the OTHER candidate cause: does the bank's COMPOSITION
# itself change when only 1 species is excluded instead of 2? Set True to force
# TARGET_LIST down to [PRIMARY_TARGET] alone -- the exact exclusion set the original
# baseline used -- so build_banks() in CELL 2 produces a directly comparable bank.
# Re-run CELL 1 then CELL 2 after flipping this; no kernel restart needed, everything
# downstream is rebuilt from these globals. CELL 3/4 are not needed for this comparison.
SINGLE_TARGET_COMPARISON = False

# Added 29 Aug 2026 for CELL 6 (Perch distillation). Selects which embedding space the
# STAGE-2 VERIFIER is trained and run in:
#   "perch"     -- Perch v2's 1536-d embeddings. Every result published before this flag
#                  existed used this, and it stays the default.
#   "distilled" -- a ~445k-param CNN trained (CELL 6) to reproduce Perch's embeddings.
#                  At detection time NO Perch inference runs at all.
#
# DELIBERATELY NARROW. Only the verifier's embeddings switch. Decoy ranking
# (SPECIES_CACHE, perch_embed) and 'diverse' shot selection stay on Perch under BOTH
# settings, because those choose WHAT THE TEST IS -- which species count as decoys, and
# which five calls are the support set. If the student picked its own decoys, the two arms
# would be scored against different test sets and the comparison would be meaningless.
# This is the same trap the literature review flagged for the Perch v1->v2 upgrade
# ("change Perch, and the decoy set changes, and the difficulty of the specificity test
# changes with it"). Holding the test fixed is what makes teacher-vs-student a clean A/B.
EMBED_BACKEND = "perch"
DISTILL_EMBED_CTX = None      # set by CELL 6 to the trained student's embedder

def verifier_embed_ctx(y22, centres_sec):
    """Embeddings the stage-2 verifier is FIT on and SCORED with. y22 is a 22050 Hz
    recording; centres_sec are event centres in seconds. Dispatches on EMBED_BACKEND so
    that build_target()'s verifier fit and CELL 2's detect_pv() always agree about which
    embedding space they are in -- a mismatch there would fit the classifier in one space
    and score candidates in another, which produces plausible-looking nonsense rather
    than an error."""
    if EMBED_BACKEND == "perch":
        return perch_embed_ctx(to32k(y22), centres_sec)
    if EMBED_BACKEND == "distilled":
        assert DISTILL_EMBED_CTX is not None, \
            "EMBED_BACKEND='distilled' but CELL 6 has not set DISTILL_EMBED_CTX"
        return DISTILL_EMBED_CTX(y22, centres_sec)
    raise ValueError(f"unknown EMBED_BACKEND {EMBED_BACKEND!r}")

root = os.path.dirname(glob.glob('/kaggle/input/**/train_metadata.csv', recursive=True)[0])
AUD = os.path.join(root, 'train_audio')
SS  = os.path.join(root, 'unlabeled_soundscapes')
meta = pd.read_csv(os.path.join(root, 'train_metadata.csv'))

# Recordist identity column, for the recordist-disjoint confound check (CELL 5, roadmap
# "owed" list: rules out "did it learn the microphone, not the bird"). BirdCLEF inherits
# Xeno-Canto's schema, where the recordist's name is the 'author' field -- but that is a
# CONVENTION, not something this script has verified against the actual attached dataset,
# so it is printed here rather than assumed silently. If build_target(recordist_disjoint=
# True) raises a KeyError, check this list against the real column name and fix RECORDIST_COL.
RECORDIST_COL = "author"
print(f"train_metadata.csv columns: {meta.columns.tolist()}")
print(f"  RECORDIST_COL = {RECORDIST_COL!r} "
      f"({'present' if RECORDIST_COL in meta.columns else 'NOT FOUND -- fix before using recordist_disjoint'})")

# ------------------------------------------------------------------ audio helpers
def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))

def ld(p, dur=None):
    y, _ = librosa.load(p, sr=SR, mono=True, duration=dur)
    return y.astype('float32')

def l2n(x):
    x = np.asarray(x, dtype="float32")
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))

def loudest_offset(y, n=CLEN):
    if len(y) <= n: return 0
    e = np.square(y, dtype=np.float64); cs = np.concatenate([[0.0], np.cumsum(e)])
    return int(np.argmax(cs[n:] - cs[:-n]))

def peak_windows(y, k=1, n=CLEN):
    """Top-k non-overlapping highest-energy windows of length n."""
    if len(y) <= n: return [np.pad(y, (0, n - len(y))).astype('float32')]
    e = np.square(y, dtype=np.float64); cs = np.concatenate([[0.0], np.cumsum(e)])
    w = cs[n:] - cs[:-n]
    out, taken = [], []
    for _ in range(k):
        pick = next((int(i) for i in np.argsort(w)[::-1] if all(abs(int(i) - t) >= n for t in taken)), None)
        if pick is None: break
        taken.append(pick); out.append(y[pick:pick + n].astype('float32'))
    return out

def build_support_views(recordings, shifts=(-0.25, 0.0, 0.25)):
    out = []
    for y in recordings:
        base = loudest_offset(y)
        for shift in shifts:
            a = int(np.clip(base + shift * SR, 0, max(0, len(y) - CLEN)))
            w = y[a:a + CLEN]
            if len(w) < CLEN: w = np.pad(w, (0, CLEN - len(w)))
            out.append(w.astype('float32'))
    return out

def place(y, pool, rng, spans, snr_db, gap_s=0.4):
    """Mix one random call from `pool` into `y` at a free position -> centre in seconds."""
    for _ in range(80):
        pos = int(rng.integers(0, len(y) - CLEN))
        if any(pos < e + int(gap_s*SR) and pos + CLEN > s - int(gap_s*SR) for s, e in spans):
            continue
        c = pool[int(rng.integers(len(pool)))][:CLEN]
        if len(c) < CLEN: c = np.pad(c, (0, CLEN - len(c)))
        g = rms(y[pos:pos+CLEN]) * (10 ** (float(snr_db)/20)) / (rms(c) + 1e-9)
        y[pos:pos+CLEN] += c.astype('float32') * g
        spans.append((pos, pos + CLEN))
        return (pos + CLEN/2) / SR
    return None

# ------------------------------------------------------------------ Perch
# PERCH 2.0 UPGRADE (Aug 2025 release, arXiv 2508.04665). Three things changed and each
# one silently breaks the v1 code path, so all three are handled explicitly. Verified
# against the official loader: google-research/perch-hoplite, zoo/taxonomy_model_tf.py.
#
#  (a) NO `infer_tf`. Perch 1.0 was a TF-Jax export exposing model.infer_tf(). Perch 2.0
#      exposes only the standard serving signature. perch-hoplite branches on exactly
#      this: `elif hasattr(self.model, 'infer_tf') ... else: # Perch v2 style
#      outputs = self.model.signatures['serving_default'](inputs=rebatched_audio)`.
#      Calling infer_tf on v2 raises AttributeError.
#
#  (b) EMBEDDING WIDTH 1280 -> 1536 (EfficientNet-B1 -> B3). Probed, never assumed.
#      v2 also emits a `spatial_embedding` (5,3,1536) head that must be ignored --
#      `embedding` is the pooled mean vector we want.
#
#  (c) AUDIO IS PEAK-NORMALISED BEFORE INFERENCE. The official wrapper removes DC offset
#      and scales each window to a peak of 0.25 (zoo_interface.normalize_audio,
#      target_peak=0.25). ARGUS v1 fed raw waveforms to Perch and did NEITHER -- so
#      Perch was being given amplitudes outside the distribution it is served with.
#      This affects the EXISTING v1 numbers too, not just the upgrade.
#      Note it does NOT disturb the SNR sweep: peak norm is a per-window scalar multiply,
#      and SNR is a ratio within the window, so planted-call SNR is preserved exactly.
PERCH_VERSION   = "v2"     # "v2" | "v1"  -- recorded in the results CSV
PERCH_PEAK_NORM = 0.25     # None to disable (v1 behaviour); 0.25 matches the official wrapper
PERCH_BACKEND   = "tf"     # "tf" | "onnx" -- see below. Default "tf" -> ZERO behaviour
                           # change to the tested GPU pipeline unless explicitly opted in.

PERCH_SR, PERCH_WIN = 32000, 5.0     # unchanged between versions
PERCH_LEN = int(PERCH_WIN * PERCH_SR)
PERCH_BATCH = 16

_V2_SLUGS = ("perch_v2", "perch_v2_cpu")
def _find_local_perch(prefer_v2):
    """Kaggle attaches models as input dirs. Pick one matching the requested version."""
    hits = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/saved_model.pb', recursive=True)
            if 'bird-vocalization' in p.lower() or 'perch' in p.lower()]
    if not hits: return None
    is_v2 = lambda p: any(s in p.lower() for s in _V2_SLUGS)
    pool = [p for p in hits if is_v2(p) == prefer_v2] or hits
    def _ver(p):
        try: return int(os.path.basename(p))
        except ValueError: return -1
    return max(pool, key=_ver)

def _remote_handle():
    if PERCH_VERSION != "v2":
        # perch_8 in perch-hoplite's config; v8 is the last 1280-d release.
        return ("https://www.kaggle.com/models/google/bird-vocalization-classifier"
                "/tensorFlow2/bird-vocalization-classifier/8")
    # perch-hoplite ships separate GPU and CPU builds at different version numbers.
    try:
        import tensorflow as tf
        gpu = bool(tf.config.list_physical_devices('GPU'))
    except Exception:
        gpu = False
    slug, ver = ("perch_v2", 2) if gpu else ("perch_v2_cpu", 1)
    return ("https://www.kaggle.com/models/google/bird-vocalization-classifier"
            f"/tensorFlow2/{slug}/{ver}")

# ---------------------------------------------------------------- ONNX backend (opt-in)
# For CPU-constrained / edge-hardware experiments (roadmap: ARGUS_Hardware_Deployment_
# Roadmap.md), not for the main GPU pipeline. Uses a pre-converted, community-published,
# Apache-2.0 ONNX export (justinchuby/Perch-onnx on Hugging Face) rather than converting
# ourselves -- same I/O signature, independently verified against the real 413MB weights
# on 13 Aug 2026: input 'inputs' [batch,160000], output 'embedding' [batch,1536], real
# inference measured at ~156ms/clip (batch=16, 4 CPU threads) on a laptop CPU -- NOT the
# target embedded device, a reference point only. Peak-norm was confirmed to materially
# change the real embedding (mean abs delta 0.065), i.e. the v1 peak-norm bug was real,
# not theoretical, on the actual weights.
_ONNX_HF_URL = "https://huggingface.co/justinchuby/Perch-onnx/resolve/main/perch_v2_no_dft.onnx"

def _find_local_onnx():
    hits = (glob.glob('/kaggle/input/**/perch_v2_no_dft*.onnx', recursive=True)
            or glob.glob('/kaggle/input/**/perch_v2*.onnx', recursive=True))
    return hits[0] if hits else None

def _load_onnx_perch():
    import onnxruntime as ort
    path = _find_local_onnx()
    if path is None:
        # Falls back to a live download (~413MB) if nothing is attached as a Kaggle
        # input -- slow and wasteful on repeated runs. Attaching a Kaggle dataset
        # mirror (e.g. tuckerarrants/perch-v2-no-dft-onnx) is the fast path.
        import requests
        path = "/kaggle/working/perch_v2_no_dft.onnx"
        print(f"  no local ONNX Perch found; downloading from {_ONNX_HF_URL} ...")
        r = requests.get(_ONNX_HF_URL, stream=True, timeout=300)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    print(f"  ONNX Perch loaded from {path}  |  input='{in_name}'  outputs={out_names}")
    return sess, in_name, out_names

if PERCH_BACKEND == "onnx":
    _ort_sess, _ort_in, _ort_out_names = _load_onnx_perch()
    _HAS_INFER_TF = False; _SERVING = None; perch = None   # not used on this path
    print(f"  calling convention: onnxruntime (CPU)  |  peak-norm: {PERCH_PEAK_NORM}")
else:
    PERCH_HANDLE = _find_local_perch(PERCH_VERSION == "v2") or _remote_handle()
    print(f"Perch {PERCH_VERSION} source: {PERCH_HANDLE}")
    perch = hub.load(PERCH_HANDLE)

    # Which calling convention does THIS artefact expose? Detected, not assumed -- so a
    # mis-attached Kaggle input degrades to a clear message instead of a confusing crash.
    _HAS_INFER_TF = hasattr(perch, "infer_tf")
    _SERVING = None
    if not _HAS_INFER_TF:
        try:
            _SERVING = perch.signatures["serving_default"]
        except Exception as ex:
            raise RuntimeError(
                "Loaded Perch artefact exposes neither infer_tf nor a 'serving_default' "
                f"signature ({type(ex).__name__}). Check the attached Kaggle model."
            ) from ex
    print(f"  calling convention: {'infer_tf (Perch 1.x)' if _HAS_INFER_TF else 'serving_default (Perch 2.x)'}"
          f" | peak-norm: {PERCH_PEAK_NORM}")
_perch_batched = True

def _probe_embed_dim():
    try:
        return int(_perch_run(np.zeros((1, PERCH_LEN), dtype="float32")).shape[-1])
    except Exception as ex:
        print("  could not probe embedding dim, assuming 1280:", type(ex).__name__)
        return 1280

def to32k(y):
    return librosa.resample(np.asarray(y, dtype="float32"), orig_sr=SR, target_sr=PERCH_SR)

def _peak_norm(batch):
    """DC-remove and scale each window to PERCH_PEAK_NORM -- zoo_interface.normalize_audio."""
    if PERCH_PEAK_NORM is None: return batch
    b = np.asarray(batch, dtype="float32").copy()
    b -= np.mean(b, axis=-1, keepdims=True)
    pk = np.max(np.abs(b), axis=-1, keepdims=True)
    b = np.divide(b, pk, where=(pk > 0.0))
    return (b * PERCH_PEAK_NORM).astype("float32")

def _extract_embedding(out):
    """Perch 2.x -> dict with 'embedding' (pooled) plus 'spatial_embedding'/'frontend'/
    logit heads. Perch 1.x -> dict with 'embedding', or a bare (logits, embeddings) tuple."""
    if hasattr(out, "keys"):
        if "embedding" not in out:
            raise KeyError(f"no 'embedding' head in Perch output; got {sorted(out.keys())}")
        emb = out["embedding"]
    else:
        emb = out[1]
    return np.asarray(emb.numpy() if hasattr(emb, "numpy") else emb)

def _infer(batch):
    batch = _peak_norm(batch)
    if PERCH_BACKEND == "onnx":
        result = _ort_sess.run(None, {_ort_in: batch})
        out = dict(zip(_ort_out_names, result))   # verified real keys: embedding/
                                                   # spatial_embedding/spectrogram/label
    else:
        out = perch.infer_tf(batch) if _HAS_INFER_TF else _SERVING(inputs=batch)
    return _extract_embedding(out)

def _perch_run(waves):
    global _perch_batched
    n = len(waves)
    if n == 0: return np.zeros((0, EMBED_DIM if "EMBED_DIM" in globals() else 1280), dtype="float32")
    if _perch_batched:
        try:
            out = []
            for i in range(0, n, PERCH_BATCH):
                chunk = waves[i:i+PERCH_BATCH]; k = len(chunk)
                if k < PERCH_BATCH:
                    chunk = np.concatenate([chunk, np.zeros((PERCH_BATCH-k, PERCH_LEN), dtype="float32")])
                out.append(_infer(chunk)[:k])
            return np.concatenate(out)
        except Exception as ex:
            print("batched Perch inference failed, falling back to per-clip:", type(ex).__name__)
            _perch_batched = False
    return np.stack([_infer(w[np.newaxis, :])[0] for w in waves])

def perch_embed(waveforms):
    """Isolated clips -> (n, D). CENTRED zero-padding, Perch's documented convention."""
    ws = []
    for w in waveforms:
        w = to32k(w)
        if len(w) >= PERCH_LEN:
            w = w[:PERCH_LEN]
        else:
            pad = PERCH_LEN - len(w); lo = pad // 2
            w = np.pad(w, (lo, pad - lo))
        ws.append(w.astype("float32"))
    return _perch_run(np.stack(ws)) if ws else np.zeros((0, EMBED_DIM), dtype="float32")

def perch_embed_ctx(y32, centres_sec):
    """Real 5s windows cut from a 32kHz recording, centred on each event time."""
    ws = []
    for c in centres_sec:
        a = int(round(c * PERCH_SR)) - PERCH_LEN // 2
        a = max(0, min(a, max(0, len(y32) - PERCH_LEN)))
        w = y32[a:a + PERCH_LEN]
        if len(w) < PERCH_LEN:
            pad = PERCH_LEN - len(w); lo = pad // 2
            w = np.pad(w, (lo, pad - lo))
        ws.append(w.astype("float32"))
    return _perch_run(np.stack(ws)) if ws else np.zeros((0, EMBED_DIM), dtype="float32")

EMBED_DIM = _probe_embed_dim()
_DETECTED = {1280: "v1", 1536: "v2"}.get(EMBED_DIM, "unknown")
print(f"Perch embedding dim: {EMBED_DIM}  (detected {_DETECTED}, requested {PERCH_VERSION})")
if _DETECTED != PERCH_VERSION:
    # Wrong artefact attached is the likeliest cause, and it would otherwise produce
    # perfectly plausible numbers from the wrong model. Refuse rather than mislead.
    raise RuntimeError(
        f"Perch version mismatch: requested {PERCH_VERSION} but the loaded model emits "
        f"{EMBED_DIM}-d embeddings ({_DETECTED}). Loaded from: {PERCH_HANDLE}\n"
        f"  -> On Kaggle, attach the model variation matching PERCH_VERSION:\n"
        f"     v2: bird-vocalization-classifier/tensorFlow2/perch_v2 (GPU, v2) "
        f"or perch_v2_cpu (CPU, v1)\n"
        f"     v1: bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier (v8)")

# ------------------------------------------------- pre-registered target selection
_counts = meta.primary_label.value_counts()
_eligible = [sp for sp in _counts.index if _counts[sp] >= MIN_RECS]
if PRIMARY_TARGET not in _eligible:
    raise ValueError(f"{PRIMARY_TARGET} has only {_counts.get(PRIMARY_TARGET, 0)} recordings "
                     f"(need >= {MIN_RECS})")
if SINGLE_TARGET_COMPARISON:
    TARGET_LIST = [PRIMARY_TARGET]
else:
    TARGET_LIST = [PRIMARY_TARGET] + [sp for sp in _eligible if sp != PRIMARY_TARGET][:N_TARGETS - 1]
    if SMOKE_TEST:
        TARGET_LIST = TARGET_LIST[:2]
TARGET_SET = set(TARGET_LIST)     # <-- consumed by CELL 2 to exclude ALL targets from the
                                  #     encoder bank. v1 hard-coded "whbsho3" there, which
                                  #     silently trained the encoder on every OTHER target.
print(f"\nPRE-REGISTERED TARGETS (n={len(TARGET_LIST)}, smoke_test={SMOKE_TEST}, "
      f"single_target_comparison={SINGLE_TARGET_COMPARISON}) "
      f"-- copy this list into the logbook BEFORE running:")
for sp in TARGET_LIST:
    print(f"   {sp}  ({_counts[sp]} recordings)")

# ------------------------------------------------- global species cache (ONCE)
_cand = (meta[~meta.primary_label.isin(TARGET_SET)]['primary_label']
         .value_counts().head(N_CANDIDATES).index.tolist())
SPECIES_CACHE = {}
_t0 = time.time()
for sp in _cand:
    calls = []
    for f in meta[meta.primary_label == sp]['filename'].tolist()[:CALIB_PER_SPECIES]:
        try: calls += peak_windows(ld(os.path.join(AUD, f), 30), k=1)
        except Exception: pass
    if not calls: continue
    SPECIES_CACHE[sp] = {"calls": calls, "emb": l2n(perch_embed(calls)).mean(0)}
print(f"species cache: {len(SPECIES_CACHE)}/{len(_cand)} candidates embedded "
      f"in {time.time()-_t0:.0f}s (computed once, reused by every target)")

_decoy_pool_cache = {}
def _deep_decoy_pool(sp):
    if sp not in _decoy_pool_cache:
        pool = []
        for f in meta[meta.primary_label == sp]['filename'].tolist()[:DECOY_RECORDINGS]:
            try: pool += peak_windows(ld(os.path.join(AUD, f), 30), k=1)
            except Exception: pass
        _decoy_pool_cache[sp] = pool or SPECIES_CACHE[sp]["calls"]
    return _decoy_pool_cache[sp]

# ------------------------------------------------------------------ soundscape beds
bed_paths = sorted(glob.glob(os.path.join(SS, "*.ogg")))[:N_CALIB_BEDS + N_EVAL_BEDS]
all_beds = [ld(p, 240.0) for p in bed_paths]
calib_beds, eval_beds = all_beds[:N_CALIB_BEDS], all_beds[N_CALIB_BEDS:]
BED_SEC = 240.0
print(f"beds: {len(calib_beds)} calibration + {len(eval_beds)} evaluation (disjoint)")

# ------------------------------------------------------------ shot selection
def choose_shots(files, ratings, embs, n_shot, strategy, rng):
    """-> indices into `files`. Each strategy is a different answer to the open question
    Nolasco et al. sec.4.1 raise: which 5 examples should a deployed system be given?
      rated   : highest Xeno-Canto rating (ARGUS v1 behaviour -- best-quality audio)
      first   : first n in metadata order (DCASE Task 5 behaviour -- no curation)
      random  : uniform sample (control)
      diverse : greedy max-min distance in Perch space (maximum coverage of the
                call repertoire -- the thing the expert annotators said was missing)
    """
    n = len(files)
    if strategy == "rated":
        return list(np.argsort(-np.asarray(ratings, dtype=float))[:n_shot])
    if strategy == "first":
        return list(range(min(n_shot, n)))
    if strategy == "random":
        return list(rng.permutation(n)[:n_shot])
    if strategy == "diverse":
        E = l2n(embs)
        picked = [int(np.argmax(np.asarray(ratings, dtype=float)))]   # deterministic seed point
        while len(picked) < min(n_shot, n):
            d = np.min(1.0 - E @ E[picked].T, axis=1)   # cosine distance to nearest picked
            d[picked] = -np.inf
            picked.append(int(np.argmax(d)))
        return picked
    raise ValueError(f"unknown shot strategy: {strategy}")

# ------------------------------------------------------------ stage-2 verifiers
class PrototypicalProbe:
    """Prototype-distance classifier. Drop-in for LogisticRegression (fit/predict_proba).

    WHY: Perch 2.0's own benchmark (Table 3, BEANS) reports prototypical probing beating
    linear probing on DETECTION tasks by +0.078 mAP (0.426 -> 0.504). ARGUS stage 2 is a
    detection task, and currently uses a linear probe.

    K prototypes per class rather than one mean, following the ProtoPNet lineage Perch 2.0
    adopts (4 prototypes/class, prediction = max activation across them). This is also the
    direct answer to Nolasco et al. sec.4.1: a single averaged prototype "implies that the
    examples are in some sense all of one kind" -- false for a species with several call
    types, which is exactly ARGUS's failure mode.

    Scores are cosine similarities (inputs are L2-normalised upstream), turned into
    calibrated probabilities by a fitted temperature. Calibration matters here because the
    fusion rules multiply p*v -- an uncalibrated v that saturates to 0/1 becomes a hard
    veto and destroys ranking, which is the same trap that forced C=0.1 on the LogReg.
    """

    def __init__(self, n_prototypes=4, random_state=0, class_weight="balanced"):
        self.n_prototypes = n_prototypes
        self.random_state = random_state
        self.class_weight = class_weight

    def _protos(self, Xc):
        k = int(min(self.n_prototypes, len(Xc)))
        if k <= 1:
            return l2n(Xc.mean(0, keepdims=True))
        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(Xc)
        return l2n(km.cluster_centers_)

    def _scores(self, X):
        """-> (n, 2): max cosine similarity to each class's prototype set."""
        Xn = l2n(X)
        return np.stack([(Xn @ self.protos_[c].T).max(1) for c in (0, 1)], axis=1)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _fit_temperature(self, S, y, w, eps=0.05):
        """Temperature by weighted log-loss with LABEL SMOOTHING.

        Smoothing is not cosmetic here. The calibration set separates almost perfectly in
        Perch space, so an unsmoothed objective drives T -> 0 and saturates predict_proba
        to hard 0/1. That turns v into a veto rather than a score, which destroys the
        ranking that AP and the p*v fusion rules depend on -- the identical failure mode
        that forced C=0.1 on the LogisticRegression verifier. With targets smoothed to
        1-eps the optimum is finite and the probabilities stay soft.
        """
        best_T, best_ll = 1.0, np.inf
        for T in np.geomspace(0.02, 3.0, 60):
            p_true = np.clip(self._softmax(S / T)[np.arange(len(y)), y], 1e-9, 1.0)
            p_false = np.clip(1.0 - p_true, 1e-9, 1.0)
            ll = float(-(w * ((1.0 - eps) * np.log(p_true)
                              + eps * np.log(p_false))).sum() / w.sum())
            if ll < best_ll:
                best_T, best_ll = float(T), ll
        return best_T

    def fit(self, X, y):
        X = np.asarray(X, dtype="float32"); y = np.asarray(y).astype(int)
        self.protos_ = {c: self._protos(X[y == c]) for c in (0, 1)}
        if self.class_weight == "balanced":
            cw = {c: len(y) / (2.0 * max(1, (y == c).sum())) for c in (0, 1)}
        else:
            cw = {0: 1.0, 1: 1.0}
        w = np.array([cw[int(v)] for v in y], dtype="float64")
        self.temperature_ = self._fit_temperature(self._scores(X), y, w)
        return self

    def predict_proba(self, X):
        return self._softmax(self._scores(X) / self.temperature_)

    def score(self, X, y):
        return float((self.predict_proba(X).argmax(1) == np.asarray(y).astype(int)).mean())


def _make_verifier():
    if VERIFIER == "logreg":
        # C deliberately small: 1280/1536-d embeddings with ~100 samples separate perfectly
        # at C=1, saturating predict_proba to 0/1 and turning v into a hard veto.
        return LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
    if VERIFIER == "proto":
        return PrototypicalProbe(n_prototypes=4, random_state=0)
    raise ValueError(f"unknown VERIFIER: {VERIFIER}")


# ------------------------------------------------------------ per-species covariates
def _covariates(support_views, own_embs, decoy_sims):
    """Recorded so the difficulty-factor regression Nolasco et al. could not resolve
    becomes possible later. Free now; unreconstructable after the fact."""
    E = l2n(own_embs)
    if len(E) > 1:
        S = E @ E.T
        iu = np.triu_indices(len(E), k=1)
        stereotypy = float(S[iu].mean())        # high = calls resemble each other
    else:
        stereotypy = float('nan')
    cents, bws, durs = [], [], []
    for w in support_views:
        cents.append(float(np.mean(librosa.feature.spectral_centroid(y=w, sr=SR))))
        bws.append(float(np.mean(librosa.feature.spectral_bandwidth(y=w, sr=SR))))
        e = np.square(w, dtype=np.float64)
        thr = 0.1 * e.max() if e.max() > 0 else 0.0
        durs.append(float((e > thr).sum()) / SR)   # seconds above 10% of peak energy
    return dict(
        stereotypy=stereotypy,
        spectral_centroid_hz=float(np.mean(cents)),
        spectral_bandwidth_hz=float(np.mean(bws)),
        call_duration_s=float(np.mean(durs)),
        decoy_sim_top1=float(max(decoy_sims)) if decoy_sims else float('nan'),
        decoy_sim_mean=float(np.mean(decoy_sims)) if decoy_sims else float('nan'),
    )

# ------------------------------------------------------------ per-target builder
# Everything except the verifier fit is deterministic given (species, n_shot, strategy,
# seed) -- and it is nearly all Perch inference, the dominant cost. The ablation ladder
# rebuilds targets once per rung, so without this cache every rung would re-embed the same
# audio. Cached across rungs; only the (cheap) classifier is refitted.
_TARGET_CACHE = {}

def build_target(sp, n_shot=N_SHOT, strategy=SHOT_STRATEGY, seed=0, verbose=True,
                  decoy_mode="hard", recordist_disjoint=False):
    """Everything v1 did at module level for one hard-coded species, now per species.
    Returns a self-contained context dict consumed by CELL 2.

    decoy_mode selects WHICH species (out of every candidate ranked by Perch-embedding
    similarity to this target) become the N_DECOYS used at test time -- roadmap 3.3:
      'hard'   (default) -- the top N_DECOYS nearest. Unchanged behaviour; every result
                published before this flag existed used this mode.
      'medium' -- ranks N_DECOYS..2*N_DECOYS-1 (5th-8th nearest at N_DECOYS=4).
      'random' -- N_DECOYS drawn uniformly from the full non-target candidate pool,
                  seeded from this call's own `seed` so it's reproducible, not a
                  fresh draw on every call.
    Only what a run is SCORED against changes; the verifier is still trained against
    every non-decoy candidate as negatives regardless of mode (calib_neg_species below),
    so 'easier' decoys are a harder TEST, not a easier training signal.

    recordist_disjoint -- confound-closing run 3 (roadmap "owed" list): "did it learn the
    microphone, not the bird?" Default False reproduces every previously-published number
    byte-for-byte (held-out pool = every recording not used as a shot, regardless of who
    recorded it). True additionally excludes, from the held-out pool, any recording whose
    RECORDIST_COL matches one of the n_shot support recordings -- so if a species'
    apparent AUC were actually partly "recognise this recordist's mic/room", a
    recordist-disjoint held-out pool would show it as a drop. Can raise ValueError if a
    species has too few distinct recordists (the held-out pool empties out) -- that itself
    is worth logging, not a bug to route around."""
    # EMBED_BACKEND is part of the key for the same reason recordist_disjoint is: the cache
    # stores the fitted verifier's TRAINING MATRIX (c["X"]) and refits only the cheap
    # classifier on it. Without the backend in the key, switching to the distilled encoder
    # would silently refit on Perch's cached embeddings and report the result as distilled.
    ck = (sp, int(n_shot), strategy, int(seed), decoy_mode, bool(recordist_disjoint),
          EMBED_BACKEND)
    if ck in _TARGET_CACHE:
        c = _TARGET_CACHE[ck]
        clf = _make_verifier().fit(c["X"], c["yv"])
        cov = dict(c["cov"]); cov["clf_train_acc"] = float(clf.score(c["X"], c["yv"]))
        cov["verifier"] = VERIFIER
        if verbose:
            print(f"  [{sp}] cached build reused; verifier={VERIFIER} "
                  f"(train acc {cov['clf_train_acc']:.2f})")
        return dict(species=sp, support=c["support"], target_pool=c["target_pool"],
                    PROTO=c["PROTO"], DECOYS=c["DECOYS"], decoy_pools=c["decoy_pools"],
                    perch_clf=clf, covariates=cov, neg_calls=c["neg_calls"])
    rng = np.random.default_rng(seed)
    rows = meta[meta.primary_label == sp]
    files   = rows['filename'].tolist()
    ratings = rows['rating'].tolist() if 'rating' in rows else [0.0] * len(files)
    if recordist_disjoint and RECORDIST_COL not in rows:
        raise KeyError(f"recordist_disjoint=True but {RECORDIST_COL!r} is not a column "
                        f"of train_metadata.csv; have {rows.columns.tolist()}")
    authors = (rows[RECORDIST_COL].tolist() if recordist_disjoint
               else [None] * len(files))
    if len(files) < n_shot + 2:
        raise ValueError(f"not enough recordings for {sp}: {len(files)}")

    raw = [ld(os.path.join(AUD, f), 30) for f in files]
    # 'diverse' needs one embedding per recording to pick a spread; others don't.
    rec_embs = (perch_embed([peak_windows(y, k=1)[0] for y in raw])
                if strategy == "diverse" else np.zeros((len(raw), 1)))

    shot_idx = choose_shots(files, ratings, rec_embs, n_shot, strategy, rng)
    shot_set = set(int(i) for i in shot_idx)

    support = build_support_views([raw[i] for i in shot_idx])
    # Held-out pool = every recording NOT used as a shot. v1 sliced files[N_SHOT:],
    # which under any non-'first' strategy would have leaked support into evaluation.
    excluded_idx = set(shot_set)
    if recordist_disjoint:
        # pandas gives every missing value in a string column the SAME NaN singleton, and
        # `a in shot_authors` does an identity check before equality -- so a NaN-authored
        # shot correctly excludes every other NaN-authored recording as one shared
        # pseudo-recordist. This is conservative (over-exclusion, shrinking the held-out
        # pool), not a leak -- verified 28 Aug 2026 against the real pandas behaviour after
        # an audit initially suspected the opposite (NaN != NaN silently under-excluding).
        # It fails safe, so the only thing worth doing is making it visible.
        n_nan = sum(1 for a in authors if a is not None and a != a)
        if n_nan and verbose:
            print(f"  [{sp}] {n_nan}/{len(authors)} recordings have a missing author -- "
                  f"treated as one shared pseudo-recordist for recordist_disjoint")
        shot_authors = {authors[i] for i in shot_set if authors[i] is not None}
        excluded_idx |= {i for i, a in enumerate(authors) if a in shot_authors}
    target_pool = [w for i, y in enumerate(raw) if i not in excluded_idx
                   for w in peak_windows(y, k=1)]
    if not target_pool:
        extra = (f" ({len(excluded_idx) - len(shot_set)} extra recordings excluded for "
                 f"sharing a recordist with a shot)" if recordist_disjoint else "")
        raise ValueError(f"{sp}: no held-out recordings left after taking {n_shot} "
                          f"shots{extra}")

    own_embs = perch_embed(support)
    PROTO = l2n(own_embs).mean(0); PROTO /= np.linalg.norm(PROTO) + 1e-9

    pool_sp = {k: v for k, v in SPECIES_CACHE.items() if k != sp}
    ranked  = sorted(pool_sp, key=lambda o: -cos(pool_sp[o]["emb"], PROTO))
    if decoy_mode == "hard":
        DECOYS = ranked[:N_DECOYS]
    elif decoy_mode == "medium":
        DECOYS = ranked[N_DECOYS:2 * N_DECOYS]
        if len(DECOYS) < N_DECOYS:
            raise ValueError(f"{sp}: only {len(ranked)} candidates ranked, need "
                              f"{2*N_DECOYS} for decoy_mode='medium'")
    elif decoy_mode == "random":
        # Seeded from this call's `seed`, offset so it never coincides with any other
        # rng drawn from the same seed in this function (shot choice, calibration).
        DECOYS = list(np.random.default_rng(90_000 + seed).choice(
            ranked, size=N_DECOYS, replace=False))
    else:
        raise ValueError(f"unknown decoy_mode: {decoy_mode!r} (want hard|medium|random)")
    decoy_sims  = [cos(pool_sp[o]["emb"], PROTO) for o in DECOYS]
    decoy_pools = {o: _deep_decoy_pool(o) for o in DECOYS}
    calib_neg_species = [o for o in ranked if o not in DECOYS]

    # ---- domain-matched perch_clf (identical protocol to v1, per target) ----
    neg_calls = [w for o in calib_neg_species for w in pool_sp[o]["calls"]]
    rng_cal = np.random.default_rng(11 + seed)
    X_parts, y_all = [], []
    for bed in calib_beds:
        y_bed = bed.copy(); spans, centres, labels = [], [], []
        plan = [(1, support)] * 12 + [(0, neg_calls)] * 12
        plan = [plan[i] for i in rng_cal.permutation(len(plan))]
        for i, (lab, pl) in enumerate(plan):
            c = place(y_bed, pl, rng_cal, spans, SNR_LEVELS[i % len(SNR_LEVELS)])
            if c is None: continue
            centres.append(c + float(rng_cal.uniform(-0.25, 0.25))); labels.append(lab)
        for _ in range(8):
            pos = int(rng_cal.integers(0, len(y_bed) - CLEN))
            if any(pos < e and pos + CLEN > s for s, e in spans): continue
            centres.append((pos + CLEN/2) / SR); labels.append(0)
        X_parts.append(verifier_embed_ctx(y_bed, centres)); y_all += labels

    X = l2n(np.concatenate(X_parts)); yv = np.array(y_all)
    clf = _make_verifier().fit(X, yv)

    cov = _covariates(support, own_embs, decoy_sims)
    cov.update(species=sp, n_recordings=len(files), n_shot=n_shot,
               shot_strategy=strategy, n_heldout_calls=len(target_pool),
               clf_train_acc=float(clf.score(X, yv)),
               decoy_mode=decoy_mode,
               recordist_disjoint=bool(recordist_disjoint),
               n_distinct_recordists=(len(set(authors)) if recordist_disjoint else -1),
               # provenance: a results CSV that does not say which Perch produced it
               # is not comparable to any other run.
               perch_version=PERCH_VERSION, perch_embed_dim=EMBED_DIM,
               perch_peak_norm=(PERCH_PEAK_NORM if PERCH_PEAK_NORM is not None else 0.0),
               perch_backend=PERCH_BACKEND,
               verifier=VERIFIER)

    if verbose:
        rd_note = (f"  distinct_recordists={cov['n_distinct_recordists']}"
                   if recordist_disjoint else "")
        print(f"  [{sp}] shots={strategy}({n_shot})  heldout={len(target_pool)} calls  "
              f"decoys={DECOYS} ({decoy_mode})  top-sim={cov['decoy_sim_top1']:.3f}  "
              f"stereotypy={cov['stereotypy']:.3f}  verifier={VERIFIER}{rd_note}")

    _TARGET_CACHE[ck] = dict(support=support, target_pool=target_pool, PROTO=PROTO,
                             DECOYS=DECOYS, decoy_pools=decoy_pools, X=X, yv=yv,
                             neg_calls=neg_calls,
                             cov={k: v for k, v in cov.items() if k != "clf_train_acc"})
    return dict(species=sp, support=support, target_pool=target_pool, PROTO=PROTO,
                DECOYS=DECOYS, decoy_pools=decoy_pools, perch_clf=clf, covariates=cov,
                neg_calls=neg_calls)

print("\nCELL 1 v2 ready. build_target(sp) is the entry point; TARGET_SET is exported "
      "for the encoder-bank exclusion in CELL 2.")
