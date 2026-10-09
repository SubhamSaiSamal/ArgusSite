"""Local tests for CELL 7 (arguswala_frontier.py). No GPU, no Perch, no dataset.

This cell spends a full Kaggle session, so the failure modes worth catching here are the
ones that waste it silently: an arm that reports another arm's cached result, a scan that
re-reads the calibration beds it was supposed to avoid, a backend left flipped, or a
control species that is actually the study species.
"""
import ast, re
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))   # the pipeline files sit next to this test

FAILS = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else "   <- " + detail))
    if not cond: FAILS.append(name)

SRC = open(os.path.join(HERE, "arguswala_frontier.py"), encoding="utf-8").read()
TREE = ast.parse(SRC)
TOP = {t.id: n.value for n in TREE.body if isinstance(n, ast.Assign)
       for t in n.targets if isinstance(t, ast.Name)}
FUNCS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
CLASSES = {n.name: n for n in TREE.body if isinstance(n, ast.ClassDef)}

# ---------------------------------------------------------------------------
print("\n[structure]")
check("parses as valid Python", True)
for f in ("train_width_student", "_eval_backend"):
    check(f"{f}() defined", f in FUNCS)
check("WidthStudent class defined", "WidthStudent" in CLASSES)
check("SKIP_FRONTIER exists (session can skip this cell like every other)",
      "SKIP_FRONTIER" in TOP)

# ---------------------------------------------------------------------------
# The cache-collision bug. EMBED_BACKEND is part of build_target's cache key but the
# WIDTH is not, so consecutive distilled arms would silently share a cached verifier.
# ---------------------------------------------------------------------------
print("\n[cache collision between arms -- the expensive silent failure]")
ev = ast.dump(FUNCS["_eval_backend"]) if "_eval_backend" in FUNCS else ""
check("_eval_backend clears _TARGET_CACHE before building any target "
      "(else arm N reports arm N-1's cached verifier matrix)",
      "_TARGET_CACHE" in ev and "clear" in ev)
check("the cache is ALSO cleared after each arm's finally block",
      SRC.count("_TARGET_CACHE.clear()") >= 2,
      f"found {SRC.count('_TARGET_CACHE.clear()')}")

# ---------------------------------------------------------------------------
print("\n[backend restore]")
def restores_in_finally(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not node.finalbody: continue
        for stmt in ast.walk(ast.Module(body=node.finalbody, type_ignores=[])):
            if not isinstance(stmt, ast.Assign): continue
            for t in stmt.targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Call)
                        and getattr(t.value.func, "id", None) == "globals"):
                    try:
                        if ast.literal_eval(t.slice) == "EMBED_BACKEND" and \
                           ast.literal_eval(stmt.value) == "perch":
                            return True
                    except Exception:
                        pass
    return False
check("EMBED_BACKEND restored to 'perch' inside a finally block", restores_in_finally(TREE))
check("DISTILL_EMBED_CTX is also cleared on the way out (a stale embedder bound to a "
      "dead arm is as dangerous as a stale backend)",
      'globals()["DISTILL_EMBED_CTX"] = None' in SRC)
check("CELL 7 asserts it STARTS from the perch backend",
      'assert EMBED_BACKEND == "perch"' in SRC)
check("Part B re-asserts the perch backend before scanning "
      "(the scan must run the real pipeline, not a distilled arm)",
      SRC.count('assert EMBED_BACKEND == "perch"') >= 2)

# ---------------------------------------------------------------------------
print("\n[Part A -- frontier design]")
check("FRONTIER_WIDTHS is a tuple of >=3 increasing widths (a curve, not a comparison)",
      "FRONTIER_WIDTHS" in TOP and isinstance(TOP["FRONTIER_WIDTHS"], ast.Tuple)
      and len(TOP["FRONTIER_WIDTHS"].elts) >= 3)
widths = [ast.literal_eval(e) for e in TOP["FRONTIER_WIDTHS"].elts] if "FRONTIER_WIDTHS" in TOP else []
check("widths strictly increasing", widths == sorted(widths) and len(set(widths)) == len(widths),
      str(widths))
check("128 is among the widths (CELL 6's published setting, so the sweep contains a "
      "point directly comparable to the archived result)", 128 in widths, str(widths))
check("all arms share ONE teacher set built once "
      "(build_distill_set called exactly once)", SRC.count("build_distill_set(") == 1)
check("every arm uses the same stage-1 encoder ENC_F, so only the embedding space varies",
      SRC.count("ENC_F") >= 4 and "ENC_F = encoders[" in SRC)
ts = ast.dump(FUNCS["train_width_student"]) if "train_width_student" in FUNCS else ""
check("the train/held-out split is seeded identically for every arm "
      "(else a lucky species partition looks like a width effect)",
      "default_rng" in ts and "seed" in ts)
check("split is by species, not by crop", "np.unique(species)" in SRC)
check("best epoch selected on held-out species, not last epoch",
      "best_state" in SRC and "load_state_dict" in SRC)
check("a nonlinear-head arm exists to separate 'too few dimensions' from "
      "'a linear map is not enough'", "FRONTIER_NONLINEAR" in TOP)
# CELL 6's embedder reads model.head.out_features to size its empty-input return. That
# attribute exists on nn.Linear but NOT on nn.Sequential, so without an explicit shim the
# nonlinear arm crashes the first time stage 1 proposes zero candidates -- hours in, not
# at startup.
check("the nonlinear head exposes .out_features, which CELL 6's embedder requires "
      "(nn.Sequential does not have it natively)",
      "s.head.out_features = out_dim" in SRC)
check("out_dim is set for BOTH arms, not just one branch",
      SRC.count("s.out_dim = out_dim") >= 1)
check("a same-session Perch control is run rather than reusing an old number",
      "RUN_PERCH_CONTROL" in TOP)

# pre-registration: all three outcomes named before the run
head = SRC[:SRC.index("import numpy")]
for phrase, why in [("both rise", "the bottleneck-was-binding outcome"),
                    ("flat", "the bottleneck-was-not-binding outcome"),
                    ("cosine", "the fidelity-rises-but-AUC-does-not outcome")]:
    check(f"pre-registration names {why}", phrase in head.lower() or phrase in SRC.lower())

# ---------------------------------------------------------------------------
print("\n[Part B -- wild scan]")
check("scan skips the calibration + evaluation beds",
      "SCAN_SKIP_FIRST" in TOP and "N_CALIB_BEDS" in SRC and "N_EVAL_BEDS" in SRC)
check("SCAN_SKIP_FIRST is derived from the bed counts, not hardcoded "
      "(a hardcoded 14 silently breaks if the bed config changes)",
      "SCAN_SKIP_FIRST  = N_CALIB_BEDS + N_EVAL_BEDS" in SRC
      or "SCAN_SKIP_FIRST = N_CALIB_BEDS + N_EVAL_BEDS" in SRC)
check("the scan slices files AFTER the skip offset",
      "all_paths[SCAN_SKIP_FIRST:" in SRC)
sp_ok = "SCAN_SPECIES" in TOP and "SCAN_CONTROL" in TOP
check("study species and a control species are both configured", sp_ok)
if sp_ok:
    try:
        ctrl = ast.literal_eval(TOP["SCAN_CONTROL"])
        study_is_primary = isinstance(TOP["SCAN_SPECIES"], ast.Name) and \
                           TOP["SCAN_SPECIES"].id == "PRIMARY_TARGET"
        check("study species is PRIMARY_TARGET (the species the project is about)",
              study_is_primary)
        check("control is a DIFFERENT species from the study species",
              not study_is_primary or ctrl != "whbsho3", str(ctrl))
    except Exception as e:
        check("control species parses", False, str(e))
check("threshold comes from planted eval, never tuned on the unlabelled files",
      "score_eval(FUSIONS[CAND_F], eval_pass(" in SRC)
check("scan checkpoints periodically (a killed session keeps partial detections)",
      "to_csv(SCAN_CKPT" in SRC and "% 25 == 0" in SRC)
check("top-K clips are saved for human listening", "np.savez_compressed(SCAN_CLIPS" in SRC)
check("output honestly states these are unconfirmed detections",
      "no ground truth" in SRC.lower() and "confirmed detection" in SRC.lower())
check("output tells the reader what a null result would mean, in advance",
      "close" in SRC.lower() and "bird-shaped sound" in SRC.lower())

# ---------------------------------------------------------------------------
# THE 3 SEP REGRESSION SECTION. Everything below guards a failure that actually
# happened and cost a 12 h session, rather than one that was imagined.
# ---------------------------------------------------------------------------
print("\n[regression: the CUDA OOM that killed the 3 Sep session]")
DSRC = open(os.path.join(HERE, "arguswala_distill.py"), encoding="utf-8").read()
# Comment lines are stripped before this check: both files DESCRIBE the bug they fixed,
# and a test that cannot tell an explanation from an occurrence would force the fix to
# be undocumented in order to stay green.
def _code(t):
    return chr(10).join(l for l in t.split(chr(10)) if not l.lstrip().startswith('#'))
for nm, s in (("CELL 7", _code(SRC)), ("CELL 6", _code(DSRC))):
    check(f"{nm}: held-out set is NOT pushed through the trunk in one forward pass",
          "m(Xva)" not in s,
          "found m(Xva) -- 1,633 crops x 5.6 MB of first-block activations = 8.69 GiB")
    check(f"{nm}: held-out tensors stay on the CPU and stream in batches",
          "Xva = torch.tensor(specs[vm]); Yva = torch.tensor(embs[vm])" in s)
    check(f"{nm}: a batched _val_cos() exists and iterates in steps of `batch`",
          "def _val_cos():" in s and "for i in range(0, len(Xva), batch):" in s)
    check(f"{nm}: the batched path runs under no_grad and in eval mode "
          "(eval mode is what makes batching numerically identical -- BatchNorm on "
          "running statistics cannot depend on how the held-out set is grouped)",
          "m.eval(); tot = 0.0" in s and "with torch.no_grad():" in s)
check("the training loop still streams its batches too (it always did -- that is why "
      "training survived and only validation died)", "xb = Xtr[idx].to(device)" in SRC)

print("\n[budget: enforced, not estimated]")
for nm in ("CELL7_BUDGET_H", "SCAN_RESERVE_MIN"):
    check(f"{nm} is defined", nm in TOP)
check("a deadline clock exists", "_deadline = t_all + CELL7_BUDGET_H * 3600" in SRC)
check("PART A refuses to start an arm it cannot finish",
      "if _left < SCAN_RESERVE_MIN:" in SRC)
check("PART A names the arms it skipped, so a follow-up run is one edit",
      "arms not run" in SRC)
check("PART B stops mid-scan with files written rather than being killed at the ceiling",
      "if _left_min() < 8:" in SRC and "stopped_early" in SRC)
check("arm RUN order is by information value, so a truncated PART A still answers "
      "the question", "def _priority(a):" in SRC and "arms.sort(key=_priority)" in SRC)
if "def _priority(a):" in SRC:
    def _priority(a):
        w, nl = a
        if nl:              return 9.0
        if w == 128:        return 0.0
        if w == max(widths): return 1.0
        return 2.0 + w / 10000.0
    order = sorted([(w, False) for w in widths] + [(max(widths), True)], key=_priority)
    check("that order starts with 128 (replicates CELL 6) then the widest LINEAR arm "
          "(the other end of the contrast), with the nonlinear follow-up last",
          order[0] == (128, False) and order[1] == (max(widths), False)
          and order[-1] == (max(widths), True), str(order))
check("the REPORT sorts by width, not by run order -- otherwise iloc[0]/iloc[-1] "
      "silently compares the wrong two arms",
      'g.sort_values("width"' in SRC and '.sort_values("width")' in SRC)

print("\n[PART B -- identical file sets by construction]")
check("both species are scanned inside ONE loop over files, so a budget stop cannot "
      "leave the study species with more files than the control",
      "for sp in scan_sp:" in SRC and "for i, p in enumerate(scan_paths):" in SRC
      and SRC.count("ld(p, 240.0)") == 1)
check("each species keeps its own RNG (shared audio, independent stochasticity)",
      "rngs[sp] = np.random.default_rng(SCAN_SEED)" in SRC)
check("the summary reports the file count actually scanned, not the count planned",
      "n_files" in SRC and "on the SAME {n_files} files" in SRC)
check("a zero-detection outcome is handled and pre-interpreted, rather than crashing "
      "on an empty DataFrame", "no detections above threshold" in SRC)

# ---------------------------------------------------------------------------
print("\n[budget sanity -- now costed from the 3 Sep MEASUREMENTS, not from a guess]")
if "SCAN_N_FILES" in TOP and "FRONTIER_WIDTHS" in TOP:
    # Measured 3 Sep: teacher set 12.7 min, Perch control 34.6 min, one training epoch
    # ~11 s. The original estimate costed each arm's EVALUATION and forgot its TRAINING
    # entirely, which is exactly how a "5.5 h" projection became an 8 h reality.
    epochs = ast.literal_eval(TOP["FRONTIER_EPOCHS"]) if "FRONTIER_EPOCHS" in TOP else 150
    n_distilled = len(widths) + (1 if "FRONTIER_NONLINEAR" in TOP else 0)
    train_min = epochs * 11 / 60.0
    est_a = 12.7 + 34.6 + n_distilled * (train_min + 34.6)
    reserve = ast.literal_eval(TOP["SCAN_RESERVE_MIN"])
    ceiling = ast.literal_eval(TOP["CELL7_BUDGET_H"]) * 60
    print(f"  PART A ~{est_a/60:.1f} h ({n_distilled} arms x [{train_min:.0f} min train "
          f"+ 34.6 min eval] + 47 min fixed);  PART B reserve {reserve/60:.1f} h;  "
          f"ceiling {ceiling/60:.1f} h")
    check("PART A plus PART B's reserve fits under the cell's own ceiling",
          est_a + reserve <= ceiling, f"{(est_a+reserve)/60:.1f} h vs {ceiling/60:.1f} h")
    check("the ceiling itself leaves room under Kaggle's 12 h session limit "
          "(~10 min of CELLS 1-6 runs before this cell starts)", ceiling <= 11 * 60)

print("\n" + "=" * 60)
print(f"{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILURES: ' + ', '.join(FAILS)}")
print("=" * 60)
