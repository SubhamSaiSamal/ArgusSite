"""Local tests for CELL 6 (arguswala_distill.py) and the EMBED_BACKEND plumbing it adds
to CELLS 1/2/5. No GPU, no Perch, no dataset -- pure logic and structural checks, run
before spending a Kaggle session.

The failure modes these exist to catch are all SILENT ones. A distillation run that is
subtly wrong does not crash; it reports a plausible AUC that means nothing, which is the
single most expensive kind of bug this project can ship.
"""
import ast, re
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))   # the pipeline files sit next to this test

FAILS = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else "   <- " + detail))
    if not cond: FAILS.append(name)

SETUP = open(os.path.join(HERE, "arguswala_setup_v2.py"), encoding="utf-8").read()
V2    = open(os.path.join(HERE, "arguswala_v2.py"), encoding="utf-8").read()
CONF  = open(os.path.join(HERE, "arguswala_confounds.py"), encoding="utf-8").read()
DIST  = open(os.path.join(HERE, "arguswala_distill.py"), encoding="utf-8").read()

# ---------------------------------------------------------------------------
# 1. _peak_offsets must select the SAME windows peak_windows() does.
#    If it drifts, the student's spectrogram and the teacher's waveform describe
#    different seconds of audio and the whole distillation target is noise.
# ---------------------------------------------------------------------------
print("\n[_peak_offsets vs peak_windows -- same selection rule]")
CLEN = 22050

def peak_windows_ref(y, k=1, n=CLEN):
    """Verbatim reimplementation of CELL 1's peak_windows()."""
    if len(y) <= n: return [np.pad(y, (0, n - len(y))).astype('float32')]
    e = np.square(y, dtype=np.float64); cs = np.concatenate([[0.0], np.cumsum(e)])
    w = cs[n:] - cs[:-n]
    out, taken = [], []
    for _ in range(k):
        pick = next((int(i) for i in np.argsort(w)[::-1]
                     if all(abs(int(i) - t) >= n for t in taken)), None)
        if pick is None: break
        taken.append(pick); out.append(y[pick:pick + n].astype('float32'))
    return out

def peak_offsets_impl(y, k, n=CLEN):
    """Verbatim copy of CELL 6's _peak_offsets()."""
    if len(y) <= n: return [0]
    e = np.square(y, dtype=np.float64); cs = np.concatenate([[0.0], np.cumsum(e)])
    w = cs[n:] - cs[:-n]
    order = np.argsort(w)[::-1]
    taken = []
    for _ in range(k):
        pick = next((int(i) for i in order if all(abs(int(i) - t) >= n for t in taken)), None)
        if pick is None: break
        taken.append(pick)
    return taken

rng = np.random.default_rng(0)
agree = True
for trial in range(25):
    y = (rng.standard_normal(CLEN * 6) * 0.01).astype("float32")
    for _ in range(rng.integers(1, 4)):
        at = int(rng.integers(0, len(y) - CLEN))
        y[at:at + CLEN] += rng.standard_normal(CLEN).astype("float32") * rng.uniform(0.2, 1.0)
    k = int(rng.integers(1, 5))
    offs = peak_offsets_impl(y, k)
    wins = peak_windows_ref(y, k)
    if len(offs) != len(wins):
        agree = False; break
    for o, w in zip(offs, wins):
        if not np.array_equal(y[o:o + CLEN].astype("float32"), w):
            agree = False; break
check("_peak_offsets returns offsets whose audio is byte-identical to peak_windows() "
      "(25 randomised recordings, k=1..4)", agree)

y_short = (rng.standard_normal(CLEN // 2)).astype("float32")
check("both handle a recording shorter than one window without raising",
      peak_offsets_impl(y_short, 3) == [0] and len(peak_windows_ref(y_short, 3)) == 1)

y_flat = np.zeros(CLEN * 4, dtype="float32")
offs_flat = peak_offsets_impl(y_flat, 3)
check("silent recording still yields non-overlapping offsets (no infinite loop, no dupes)",
      len(offs_flat) == len(set(offs_flat))
      and all(abs(a - b) >= CLEN for i, a in enumerate(offs_flat) for b in offs_flat[i+1:]),
      str(offs_flat))

# ---------------------------------------------------------------------------
# 2. EMBED_BACKEND plumbing in CELL 1
# ---------------------------------------------------------------------------
print("\n[CELL 1 -- EMBED_BACKEND dispatch and cache key]")
tree1 = ast.parse(SETUP)
top1 = {t.id: n.value for n in tree1.body if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)}
check("EMBED_BACKEND exists and defaults to 'perch' (every archived result used it)",
      "EMBED_BACKEND" in top1 and ast.literal_eval(top1["EMBED_BACKEND"]) == "perch")
check("DISTILL_EMBED_CTX exists and defaults to None",
      "DISTILL_EMBED_CTX" in top1 and ast.literal_eval(top1["DISTILL_EMBED_CTX"]) is None)

funcs1 = {n.name: n for n in tree1.body if isinstance(n, ast.FunctionDef)}
check("verifier_embed_ctx() is defined", "verifier_embed_ctx" in funcs1)
vec = ast.dump(funcs1["verifier_embed_ctx"]) if "verifier_embed_ctx" in funcs1 else ""
check("verifier_embed_ctx dispatches on EMBED_BACKEND", "EMBED_BACKEND" in vec)
check("verifier_embed_ctx routes 'perch' to perch_embed_ctx", "perch_embed_ctx" in vec)
check("verifier_embed_ctx routes 'distilled' to DISTILL_EMBED_CTX", "DISTILL_EMBED_CTX" in vec)
check("verifier_embed_ctx raises on an unknown backend rather than silently defaulting",
      "ValueError" in vec)
check("verifier_embed_ctx asserts DISTILL_EMBED_CTX was actually set by CELL 6",
      "assert" in vec.lower() or "AssertionError" in vec)

# The load-bearing one. _TARGET_CACHE stores the verifier's TRAINING MATRIX and refits only
# the classifier; without the backend in the key a distilled run silently refits on Perch's
# cached embeddings and reports the result as distilled.
m = re.search(r"^\s*ck = \((.*?)\)\s*$", SETUP, re.M | re.S)
ck_src = m.group(1) if m else ""
check("build_target()'s cache key includes EMBED_BACKEND (else a distilled run silently "
      "reuses Perch's cached embeddings)", "EMBED_BACKEND" in ck_src, ck_src[:120])
for part in ("recordist_disjoint", "decoy_mode", "strategy"):
    check(f"cache key still includes {part} (no regression from the EMBED_BACKEND edit)",
          part in ck_src)

check("the verifier's training matrix is built through verifier_embed_ctx, not "
      "perch_embed_ctx directly", "X_parts.append(verifier_embed_ctx(" in SETUP)

# Decoy ranking and shot selection must NOT go through the dispatch -- they define the
# test, and must be identical across both arms.
sp_cache_line = next((l for l in SETUP.splitlines() if "SPECIES_CACHE[sp] =" in l), "")
check("decoy ranking (SPECIES_CACHE) still embeds with perch_embed -- the TEST must be "
      "identical in both arms", "perch_embed(" in sp_cache_line, sp_cache_line.strip()[:100])
check("'diverse' shot selection still embeds with perch_embed (rec_embs)",
      "rec_embs = (perch_embed(" in SETUP)

# ---------------------------------------------------------------------------
# 3. CELL 2 -- detect_pv must score in the space the verifier was fit in
# ---------------------------------------------------------------------------
print("\n[CELL 2 -- detect_pv / detect_perch_only]")
tree2 = ast.parse(V2)
funcs2 = {n.name: n for n in tree2.body if isinstance(n, ast.FunctionDef)}
dpv = ast.dump(funcs2["detect_pv"]) if "detect_pv" in funcs2 else ""
check("detect_pv embeds via verifier_embed_ctx (so it follows EMBED_BACKEND)",
      "verifier_embed_ctx" in dpv)
check("detect_pv no longer calls perch_embed_ctx directly (that would pin it to Perch "
      "while build_target fits the classifier on the student)",
      "perch_embed_ctx" not in dpv)
dpo = ast.dump(funcs2["detect_perch_only"]) if "detect_perch_only" in funcs2 else ""
check("detect_perch_only DOES still call perch_embed_ctx directly (P1 is about Perch "
      "specifically; routing it through the dispatch would break that experiment)",
      "perch_embed_ctx" in dpo)

# ---------------------------------------------------------------------------
# 4. CELL 5 must refuse to run under a distilled backend
# ---------------------------------------------------------------------------
print("\n[CELL 5 -- backend guard]")
check("CELL 5 asserts EMBED_BACKEND == 'perch'",
      'assert EMBED_BACKEND == "perch"' in CONF)
check("that assert sits BEFORE the first build_target() call in CELL 5",
      CONF.index('assert EMBED_BACKEND == "perch"') < CONF.index("build_target(sp"))

# ---------------------------------------------------------------------------
# 5. CELL 6 structure
# ---------------------------------------------------------------------------
print("\n[CELL 6 -- distillation structure]")
tree6 = ast.parse(DIST)
top6 = {t.id: n.value for n in tree6.body if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)}
funcs6 = {n.name: n for n in tree6.body if isinstance(n, ast.FunctionDef)}
cls6   = {n.name: n for n in tree6.body if isinstance(n, ast.ClassDef)}

for f in ("_peak_offsets", "build_distill_set", "train_distill", "make_distill_embedder"):
    check(f"{f}() is defined", f in funcs6)
check("DistillStudent class is defined", "DistillStudent" in cls6)
check("SKIP_DISTILL exists (so a session can skip CELL 6 like every other cell)",
      "SKIP_DISTILL" in top6)

student_src = ast.dump(cls6["DistillStudent"]) if "DistillStudent" in cls6 else ""
check("the student's trunk is Enc() -- the SAME network stage 1 deploys, not a bigger "
      "one introduced for the occasion", "'Enc'" in student_src or "Enc()" in DIST)
check("the projection head is a single nn.Linear (a stronger claim than needing an MLP)",
      "nn.Linear(EMB" in DIST)

td = ast.dump(funcs6["train_distill"]) if "train_distill" in funcs6 else ""
check("train_distill splits validation by SPECIES, not by crop (a crop split would put "
      "windows from the same recording on both sides and measure memorisation)",
      "uniq" in td and "np.unique" in DIST)
check("train_distill uses cosine loss (every consumer L2-normalises, so only direction "
      "is ever read)", "cosine_similarity" in DIST)
check("train_distill selects the best epoch on held-out species, not the last epoch",
      "best_state" in DIST and "load_state_dict" in DIST)

# The single most dangerous line in the file: if EMBED_BACKEND is left flipped, every
# later cell scores a classifier in the wrong space, silently.
def _restores_backend_in_finally(tree):
    """Walk for a Try whose finalbody assigns globals()['EMBED_BACKEND'] = 'perch'.
    Checked structurally rather than by regex: the restore is separated from `finally:` by
    a comment block, and a text match would break on any future re-wording while a
    genuinely missing finally would still slip through."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not node.finalbody:
            continue
        for stmt in ast.walk(ast.Module(body=node.finalbody, type_ignores=[])):
            if not isinstance(stmt, ast.Assign):
                continue
            for t in stmt.targets:
                if (isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Call)
                        and getattr(t.value.func, "id", None) == "globals"
                        and ast.literal_eval(t.slice) == "EMBED_BACKEND"
                        and ast.literal_eval(stmt.value) == "perch"):
                    return True
    return False

check("EMBED_BACKEND is restored inside a finally block (not merely at the end of the "
      "happy path -- a mid-run exception would otherwise leave every later cell scoring "
      "a classifier in the wrong embedding space)",
      _restores_backend_in_finally(tree6))
check("the A/B loop that flips EMBED_BACKEND is itself inside that try",
      any(isinstance(n, ast.Try) and n.finalbody
          and any('EMBED_BACKEND' in ast.dump(b) for b in n.body)
          for n in ast.walk(tree6)))
check("CELL 6 asserts it STARTS from the perch backend too",
      'assert EMBED_BACKEND == "perch"' in DIST)
check("both arms are evaluated with the SAME stage-1 encoder (ENC_D), so only the "
      "embedding space differs", DIST.count("ENC_D") >= 4 and "ENC_D  = encoders[" in DIST)
check("both arms are scored with the same verifier-only fusion (CAND_D)",
      DIST.count("CAND_D") >= 3)
check("the A/B loop iterates ('perch', 'distilled') in that order",
      'for backend in ("perch", "distilled")' in DIST)
check("results are checkpointed per species (a killed session keeps partial work)",
      "to_csv(DISTILL_CKPT" in DIST)
check("the verdict reports a change against a noise bound, not a bare comparison",
      "DISTILL_NOISE_FLOOR" in DIST and "bound" in DIST)
check("the verdict has a branch warning about leakage if the student BEATS the teacher",
      "leakage" in DIST.lower())

print("\n" + "=" * 60)
print(f"{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILURES: ' + ', '.join(FAILS)}")
print("=" * 60)
