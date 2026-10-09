# -*- coding: utf-8 -*-
"""Local unit tests for CELL 5 (arguswala_confounds.py) and the CELL 1/2 additions it
depends on (recordist_disjoint, USE_SPECIES_NEG, detect_perch_only, stitch_time,
swappable detect_fn). Heavy deps unavailable here -- pure functions are re-declared
verbatim and exercised directly; everything else is checked structurally with ast."""
import ast
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))   # the pipeline files sit next to this test

FAILS = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else "   <- " + detail))
    if not cond: FAILS.append(name)

# ------------------------------------------------- stitch_time (verbatim from v2)
SR, HOP, STRIDE, SEG_SEC, MAX_EVENT_SEC, CAND_FLOOR = 22050, 256, 29, 1.0, 10.0, 0.10

def stitch_time(times, probs, floor=CAND_FLOOR, gap_s=None, max_event_sec=MAX_EVENT_SEC,
                 seg_sec=SEG_SEC):
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

print("\n[stitch_time]")
# two isolated bumps, well separated -> two events
times = np.arange(0, 20, 0.5)
probs = np.zeros_like(times)
probs[(times >= 2.0) & (times <= 3.0)] = 0.8
probs[(times >= 10.0) & (times <= 11.0)] = 0.9
ev = stitch_time(times, probs)
check("two separated bumps -> two events", len(ev) == 2, str(ev))
check("peak score preserved per event", abs(ev[0][2] - 0.8) < 1e-9 and abs(ev[1][2] - 0.9) < 1e-9, str(ev))
check("event span brackets the bump with seg_sec/2 padding",
      ev[0][0] <= 2.0 and ev[0][1] >= 3.0, str(ev[0]))

# nothing above floor -> no events
ev0 = stitch_time(times, np.zeros_like(times))
check("all-below-floor -> zero events", ev0 == [], str(ev0))

# adjacent samples within gap_s merge into ONE event even with a dip between, as long as
# both flank samples are >= floor (mirrors stitch()'s "merge across small gaps" behaviour)
times2 = np.array([0.0, 0.3, 0.6, 0.9])
probs2 = np.array([0.5, 0.5, 0.5, 0.5])
ev2 = stitch_time(times2, probs2, gap_s=0.35)
check("closely-spaced equal-floor samples merge into one event", len(ev2) == 1, str(ev2))

# very long merged run gets clipped to max_event_sec around its peak
times3 = np.arange(0, 30, 0.5)
probs3 = np.full_like(times3, 0.5); probs3[20] = 0.99   # peak in the middle
ev3 = stitch_time(times3, probs3, gap_s=1.0, max_event_sec=5.0)
check("run longer than max_event_sec is clipped", len(ev3) == 1 and (ev3[0][1]-ev3[0][0]) <= 5.0 + 1e-9,
      str(ev3))
check("clip is centred on the peak, not the run start", abs(ev3[0][2] - 0.99) < 1e-9, str(ev3))

# ------------------------------------------------- recordist_disjoint exclusion logic
# Re-implements just the exclusion-set computation from build_target's recordist_disjoint
# branch, so the leakage-prevention logic is tested without the real pipeline around it.
def excluded_indices(shot_idx, authors, recordist_disjoint):
    shot_set = set(int(i) for i in shot_idx)
    excluded = set(shot_set)
    if recordist_disjoint:
        shot_authors = {authors[i] for i in shot_set if authors[i] is not None}
        excluded |= {i for i, a in enumerate(authors) if a in shot_authors}
    return excluded

print("\n[recordist_disjoint exclusion logic]")
authors = ["alice", "bob", "alice", "carol", "bob", "dave", "carol", "eve"]
shot_idx = [0, 1]   # alice, bob
exc_off = excluded_indices(shot_idx, authors, False)
check("disjoint=False excludes only the shots themselves", exc_off == {0, 1}, str(exc_off))
exc_on = excluded_indices(shot_idx, authors, True)
check("disjoint=True also excludes every OTHER recording by alice/bob",
      exc_on == {0, 1, 2, 4}, str(exc_on))
check("disjoint=True does not touch unrelated recordists (carol/dave/eve)",
      {3, 5, 6, 7} & exc_on == set(), str(exc_on))

# every recording shares a recordist with a shot -> held-out pool must come back empty,
# which is exactly the case build_target is required to raise ValueError on, not silently
# proceed with zero data.
mono_authors = ["only_one"] * 8
exc_mono = excluded_indices([0], mono_authors, True)
check("single-recordist species -> everything excluded (build_target must raise, not "
      "silently return an empty pool)", exc_mono == set(range(8)), str(exc_mono))

# ---- real pandas NaN semantics (the 28 Aug 2026 correction) -------------------------
# An earlier audit pass suspected `authors[i] is not None` would leave NaN != NaN and
# silently UNDER-exclude (a leak). That was wrong: pandas gives every missing value in a
# string column the SAME NaN singleton object, and `in` on a set checks identity before
# equality, so NaN-authored recordings correctly collapse into one shared pseudo-recordist
# and get OVER-excluded (conservative -- shrinks the pool, never leaks). Verified against
# the REAL pandas object, not a plain-Python stand-in, since that is exactly where the
# first analysis went wrong.
import pandas as pd, math
print("\n[recordist_disjoint -- real pandas NaN identity semantics]")
real_authors = pd.DataFrame({"author": ["alice", "bob", None, "carol", float("nan"), "dave"]})["author"].tolist()
check("pandas gives every missing author the SAME NaN object (the load-bearing fact)",
      real_authors[2] is real_authors[4], str(real_authors))
exc_nan_shot = excluded_indices([2], real_authors, True)   # the NaN-authored recording is the shot
check("a NaN-authored SHOT excludes every other NaN-authored recording too (over-exclusion, "
      "not a leak)", exc_nan_shot == {2, 4}, str(exc_nan_shot))
check("a NaN-authored shot does NOT touch named recordists (no false match)",
      {0, 1, 3, 5} & exc_nan_shot == set(), str(exc_nan_shot))

# ------------------------------------------------- verdict() three-way margin logic
# Re-implements CELL 5's verdict() decision boundary (same shape as CELL 3's trend(),
# tested there) so the exact same "0.000 change must read as FLAT, not unresolved" case
# that was once a real bug (roadmap: S3, 23 Aug 2026) cannot silently regress here too.
NOISE_FLOOR, KNIFE = 0.025, 0.25

def verdict_label(val_a, val_b, sd_a, sd_b, noise_floor=NOISE_FLOOR, knife=KNIFE):
    drop = val_b - val_a
    bound = max(noise_floor, sd_a, sd_b)
    margin = abs(drop) - bound
    if margin > knife * bound:
        return "RISES" if drop > 0 else "FALLS"
    if margin < -knife * bound:
        return "FLAT"
    return "UNRESOLVED"

print("\n[verdict three-way logic]")
check("large rise, low noise -> RISES", verdict_label(0.50, 0.60, 0.01, 0.01) == "RISES")
check("large fall, low noise -> FALLS", verdict_label(0.60, 0.50, 0.01, 0.01) == "FALLS")
check("exact zero change against a real bound -> FLAT, not UNRESOLVED (the regression "
      "the old two-way S3 rule got wrong)", verdict_label(0.70, 0.70, 0.01, 0.01) == "FLAT")
check("change within noise -> FLAT", verdict_label(0.70, 0.705, 0.02, 0.02) == "FLAT")
check("change right at the knife edge (margin == 0) -> UNRESOLVED",
      verdict_label(0.70, 0.725, 0.025, 0.025) == "UNRESOLVED")

# ------------------------------------------------- structural checks on the real source
print("\n[CELL 2 structure -- USE_SPECIES_NEG, detect_perch_only, swappable detect_fn]")
src2 = open(os.path.join(HERE, "arguswala_v2.py"), encoding="utf-8").read()
tree2 = ast.parse(src2)
mod2 = tree2.body

flag_assigns = [s for s in mod2 if isinstance(s, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "USE_SPECIES_NEG" for t in s.targets)]
check("USE_SPECIES_NEG has exactly one top-level default assignment", len(flag_assigns) == 1,
      str(len(flag_assigns)))
if flag_assigns:
    val = flag_assigns[0].value
    check("USE_SPECIES_NEG default is literal False",
          isinstance(val, ast.Constant) and val.value is False, ast.dump(val))

funcs2 = {n.name: n for n in mod2 if isinstance(n, ast.FunctionDef)}
check("stage1() exists", "stage1" in funcs2)
check("stage1() accepts neg_species_calls",
      "neg_species_calls" in {a.arg for a in funcs2["stage1"].args.args} if "stage1" in funcs2 else False)
check("detect_perch_only() exists", "detect_perch_only" in funcs2)
check("detect_pv() passes neg_species_calls through to stage1()",
      "neg_species_calls" in ast.dump(funcs2["detect_pv"]) if "detect_pv" in funcs2 else False)

for fname in ("eval_pass", "spec_pass", "neg_pass"):
    check(f"{fname}() accepts a detect_fn parameter",
          fname in funcs2 and "detect_fn" in {a.arg for a in funcs2[fname].args.args},
          "missing" if fname not in funcs2 else str([a.arg for a in funcs2[fname].args.args]))

# stage1()'s USE_SPECIES_NEG branch must not be able to silently train on an empty
# negative pool -- the source must still fall back to background crops.
check("stage1() has a background-negative fallback path even when USE_SPECIES_NEG is True",
      "_background_neg" in ast.dump(funcs2["stage1"]) if "stage1" in funcs2 else False)

print("\n[CELL 1 structure -- recordist_disjoint, neg_calls exposure]")
src1 = open(os.path.join(HERE, "arguswala_setup_v2.py"), encoding="utf-8").read()
tree1 = ast.parse(src1)
mod1 = tree1.body
funcs1 = {n.name: n for n in mod1 if isinstance(n, ast.FunctionDef)}
check("build_target() exists", "build_target" in funcs1)
bt_src = ast.dump(funcs1["build_target"]) if "build_target" in funcs1 else ""
check("build_target() accepts recordist_disjoint",
      "recordist_disjoint" in {a.arg for a in funcs1["build_target"].args.args} if "build_target" in funcs1 else False)
check("build_target()'s cache key includes recordist_disjoint "
      "(else True/False would silently share a cached build)",
      "recordist_disjoint" in bt_src)
check("build_target() returns neg_calls (CELL 5 / USE_SPECIES_NEG needs it)",
      "neg_calls" in bt_src)
check("build_target() raises on an empty recordist-disjoint held-out pool "
      "rather than silently returning nothing",
      "no held-out recordings left" in bt_src)

# ---------------------------------------------------------------------------
# CELL 5 structure -- per-experiment RUN_P* gating and the P2-only seed widening
# (added 29 Aug 2026 alongside the P2 re-run). These are the same class of structural
# check the SKIP_* flags already carry: a silently-wrong flag here does not crash, it
# just spends GPU hours reproducing a settled result or, worse, writes 6-seed rows
# alongside stale 2-seed ones in the same CSV.
# ---------------------------------------------------------------------------
print("\n[CELL 5 structure -- RUN_P* gating and P2_SEEDS]")
src5 = open(os.path.join(HERE, "arguswala_confounds.py"), encoding="utf-8").read()
tree5 = ast.parse(src5)
top5 = {t.id: n.value for n in tree5.body if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)}

for flag in ("RUN_P1", "RUN_P2", "RUN_P3"):
    check(f"{flag} exists as a top-level literal", flag in top5)

check("P2_SEEDS exists and is a tuple", "P2_SEEDS" in top5 and isinstance(top5["P2_SEEDS"], ast.Tuple))
check("P2_SEEDS is strictly wider than CONFOUND_SEEDS (else the re-run buys nothing)",
      len(top5["P2_SEEDS"].elts) > len(top5["CONFOUND_SEEDS"].elts)
      if {"P2_SEEDS", "CONFOUND_SEEDS"} <= top5.keys() else False)
check("P2_SEEDS values are unique (a repeated seed would silently narrow the widening)",
      len({ast.literal_eval(e) for e in top5["P2_SEEDS"].elts}) == len(top5["P2_SEEDS"].elts)
      if "P2_SEEDS" in top5 else False)

# The load-bearing pairing: if P2 is re-run at a new seed count, its old rows MUST be
# dropped, or the CSV silently mixes 2-seed and 6-seed rows under one run_type.
p2_on = top5.get("RUN_P2") is not None and ast.literal_eval(top5["RUN_P2"]) is True
forced = ({ast.literal_eval(e) for e in top5["FORCE_RERUN"].elts}
          if "FORCE_RERUN" in top5 and isinstance(top5["FORCE_RERUN"], ast.Set) else set())
check("if RUN_P2 is True, 'species_neg' is in FORCE_RERUN "
      "(else RESUME skips P2 entirely, or mixes 2-seed and 6-seed rows)",
      (not p2_on) or ("species_neg" in forced),
      f"RUN_P2={p2_on} FORCE_RERUN={forced}")
check("FORCE_RERUN only ever names real run_types",
      forced <= {"perch_alone", "species_neg", "recordist_disjoint"}, str(forced))

# The mirror-image failure of the one above, and the more dangerous of the two: dropping a
# run_type whose RUN_P* flag is OFF deletes checkpointed work that nothing will rebuild,
# and the session still exits cleanly. Found 29 Aug 2026 by switching RUN_P2 off while
# FORCE_RERUN still named "species_neg".
check("the resume block filters FORCE_RERUN by the RUN_P* flags before dropping rows "
      "(else a force-listed but not-run experiment is silently deleted)",
      "_RUN_FLAGS" in src5 and "_force = {rt for rt in FORCE_RERUN if _RUN_FLAGS" in src5)
check("rows are dropped using the FILTERED set (_force), never FORCE_RERUN directly",
      'r["run_type"] not in _force' in src5
      and 'r["run_type"] not in FORCE_RERUN' not in src5)
check("the skipped-but-force-listed case prints a warning rather than passing silently",
      "_skipped" in src5 and "KEEPING their checkpointed rows" in src5)

check("P2's seed loop uses P2_SEEDS, not CONFOUND_SEEDS",
      "for sd in P2_SEEDS:" in src5)
check("P2's row writer records len(P2_SEEDS) as n_seeds (not the P1/P3 count)",
      'all_rows.append(_row("species_neg", sp, arm, len(P2_SEEDS)' in src5)
check("P1 still records len(CONFOUND_SEEDS) as n_seeds",
      'all_rows.append(_row("perch_alone", sp, arm, len(CONFOUND_SEEDS)' in src5)
check("P3 still records len(CONFOUND_SEEDS) as n_seeds",
      'all_rows.append(_row("recordist_disjoint", sp, arm, len(CONFOUND_SEEDS)' in src5)

for flag, rt in (("RUN_P1", "perch_alone"), ("RUN_P2", "species_neg"), ("RUN_P3", "recordist_disjoint")):
    check(f"{flag} gates its loop via `CONFOUND_SPECIES if {flag} else ()`",
          f"if {flag} else ()" in src5)

# A skipped experiment must still REPORT from the checkpoint, or turning a flag off would
# silently delete its verdict from the session log.
for rt in ("perch_alone", "species_neg", "recordist_disjoint"):
    check(f"{rt}'s report block reads all_rows (so a skipped run still prints its verdict)",
          f'run_type"] == "{rt}"' in src5)

print("\n" + "="*60)
print(f"{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILURES: ' + ', '.join(FAILS)}")
print("="*60)
