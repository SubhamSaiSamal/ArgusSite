# ===== CELL 5 / CONFOUND-CLOSING RUNS =====
# Run AFTER arguswala_v2.py (reuses its trained `encoders`, passes and scorers). Does NOT
# depend on CELL 3 or CELL 4 -- like them, it only needs CELL 1+2, so it can run in any
# order relative to the sweeps/ladder.
#
# Three items the 25 Aug 2026 adversarial review named as genuinely unrun (lab notebook,
# Day 31): "None of these are required before Oct 3; they close INTERVIEW risk, not
# submission risk" -- a judge is expected to ask each of these questions live.
#
#  P1  PERCH ALONE AS A STANDALONE DETECTOR. Every number this project has ever reported
#      uses stage-1 (the trained CNN) to propose candidate windows and Perch only to
#      VERIFY them. Nobody has asked: if you skip stage-1 entirely and just slide Perch +
#      the calibrated verifier across the recording, what do you get? Answers "what did
#      the CNN actually contribute" before a judge asks it live.
#
#  P2  STAGE-1 FINE-TUNED WITH SPECIES-MATCHED NEGATIVES. Stage-2's Perch verifier has
#      always trained against real other-species calls as negatives (calib_neg_species in
#      build_target). Stage-1's per-target fine-tuning (stage1() in CELL 2) never has --
#      its negatives have always been arbitrary random crops of whatever's in the bed.
#      USE_SPECIES_NEG (CELL 2) tests whether that choice matters.
#
#  P3  RECORDIST-DISJOINT HELD-OUT SPLIT. "Did it learn the microphone, not the bird?"
#      The held-out pool has never been checked against WHO recorded the 5 support shots.
#      recordist_disjoint=True (build_target, CELL 1) additionally excludes any held-out
#      recording sharing a recordist with a shot.
import numpy as np, pandas as pd, time, os, gc

# 28 Aug 2026, widened after run 4: species count doubled from 3 to 5 now that
# SKIP_MAIN_LOOP + SKIP_LADDER + SKIP_SWEEPS free up enough of the 12h ceiling to afford
# it (budgeted ~3.8h total for a Cell-5-focused session; see the lab notebook Day 34).
CONFOUND_SPECIES = TARGET_LIST[:5]
CONFOUND_SEEDS   = (7, 8)

# --- 29 Aug 2026: P2-only re-run, to settle the one confound that came back unresolved ---
# The 29 Aug session (lab notebook Day 35) settled P1 (perch-alone, FALLS, clear of bound)
# and P3 (recordist-disjoint, FLAT) at 2 seeds. P2 returned TOO CLOSE TO CALL in BOTH
# directions, and verdict()'s own advice on that outcome is "widen CONFOUND_SEEDS before
# claiming a direction" -- so that is exactly what this does, for P2 alone.
#
# P1/P3 stay at CONFOUND_SEEDS and are NOT re-run: their archived rows already answer their
# questions, and re-running them at P2's seed count would triple their cost to reproduce a
# result that is already settled. Their verdicts still print below, read from the
# checkpoint rather than recomputed.
# 29 Aug 2026, second revision: RUN_P2 is OFF for the distillation session. Cell 6 runs
# AFTER this cell, so bundling them would mean an overrun kills the distillation result --
# the one that answers the novelty question -- while preserving P2, which is a smaller
# claim. P2 is fully set up below and costs ~3.5h whenever it is switched back on.
RUN_P1   = False
RUN_P2   = False
RUN_P3   = False
P2_SEEDS = (7, 8, 9, 10, 11, 12)   # 3x CONFOUND_SEEDS; P2 only

# P2's existing rows were computed at 2 seeds and must not survive alongside 6-seed rows in
# the same CSV -- RESUME would otherwise skip P2 entirely as "already done". Listing a
# run_type here drops its rows from the checkpoint before resuming, so it recomputes clean.
FORCE_RERUN = {"species_neg"}
# 0.025 was checked against run 4, not just carried over: the configuration-matched floor
# (same 3 species, same 2 seeds, encoder seeds 0/1/2 -- run 4's own ladder, baseline
# verified AUC 0.754/0.739/0.756) is 0.017; per-species encoder-seed spread across the
# CONFOUND_SPECIES themselves maxes at 0.025 (barswa). The pipeline's own run-wide 0.062
# is driven entirely by eaywag1, which is not among CONFOUND_SPECIES -- using it here
# would be a >2x over-correction that suppresses real effects rather than guarding
# against noise. 0.025 is the right bound for what this file actually compares.
NOISE_FLOOR      = 0.025
KNIFE            = 0.25
BASE, CAND       = "stage-1 only  (p)", "perch only    (v)"
ENC              = encoders[(FEATURE, ENCODER_SEEDS[0])]

# SKIP_LADDER's _apply() (CELL 4) rebinds VERIFIER/FEATURE/USE_TIMEFILTER/USE_HARD_NEG as
# globals and never restores them -- harmless while SKIP_LADDER=True, but if this cell is
# ever run after a completed ladder, `ENC` above could silently bind a non-baseline
# encoder and every build_target() here would refit a non-baseline verifier. Refuse rather
# than produce confound numbers measured against a config nobody chose.
assert (FEATURE, VERIFIER, USE_TIMEFILTER, USE_HARD_NEG, USE_SPECIES_NEG) == \
       ("pcen", "logreg", False, False, False), \
       "CELL 5 inherited a non-baseline config -- check CELL 4 ran with SKIP_LADDER=True"
# Same guard, for CELL 6's flag. P1 in particular would be actively wrong under a distilled
# backend: detect_perch_only() embeds with Perch by design, but ctx["perch_clf"] would have
# been fit on the student's embeddings -- scoring one space with a classifier fit in
# another. That produces plausible numbers rather than an error, so refuse instead.
assert EMBED_BACKEND == "perch", \
       f"CELL 5 requires EMBED_BACKEND='perch', found {EMBED_BACKEND!r} -- CELL 6 must " \
       f"restore it (it does so in a finally block) before the confound runs are valid"

CKPT_PATH = "/kaggle/working/argus_confounds.csv"
RESUME    = True     # skip (run_type, species) pairs already checkpointed

# ------------------------------------------------------------------ shared plumbing
def _arm_metrics(fuse, ec, sc, nc, ctx):
    m = score_eval(fuse, ec)
    _, _, _, _, auc = score_spec(fuse, m["thr"], sc, ctx["DECOYS"])
    fa = score_neg(fuse, m["thr"], nc)
    return dict(auc=auc, f1=m["f1"], prec=m["p"], rec=m["r"], fa_per_hr=fa)

def verdict(label, val_a, val_b, sd_a, sd_b, pos_note, neg_note, flat_note,
            noise_floor=NOISE_FLOOR, knife=KNIFE):
    """Two-arm version of CELL 3's trend() margin test -- same three-way logic (real /
    flat / unresolved), just for an A-vs-B comparison instead of a multi-point curve. A
    bare `b - a > 0` would turn a 0.001 difference into a printed verdict, which is
    exactly the mistake CELL 3's own trend() was written to stop making."""
    drop = val_b - val_a
    bound = max(noise_floor, sd_a, sd_b)
    margin = abs(drop) - bound
    print(f"\n{label}")
    print(f"  {val_a:+.3f} -> {val_b:+.3f}   change {drop:+.3f}   bound {bound:.3f}   "
          f"margin {margin:+.4f}")
    if margin > knife * bound:
        print(f"  -> {'RISES' if drop > 0 else 'FALLS'} by {abs(drop):.3f}, clear of the "
              f"noise bound. {pos_note if drop > 0 else neg_note}")
    elif margin < -knife * bound:
        print(f"  -> FLAT within noise ({abs(drop):.3f} vs bound {bound:.3f}). {flat_note}")
    else:
        print(f"  -> TOO CLOSE TO CALL, within {knife:.0%} of the bound. Unresolved at "
              f"this seed count -- widen this experiment's seed tuple (CONFOUND_SEEDS for "
              f"P1/P3, P2_SEEDS for P2) before claiming a direction.")

done = set()
all_rows = []
if RESUME and os.path.exists(CKPT_PATH):
    prev = pd.read_csv(CKPT_PATH)
    if {"run_type", "species"}.issubset(prev.columns):
        all_rows = prev.to_dict("records")
        # Only drop what will actually be RECOMPUTED this session. Dropping a run_type
        # whose RUN_P* flag is off would delete it from the checkpoint and never rebuild
        # it -- silent data loss that looks like a successful run. Found 29 Aug 2026 when
        # RUN_P2 was switched off for the distillation session while FORCE_RERUN still
        # named "species_neg".
        _RUN_FLAGS = {"perch_alone": RUN_P1, "species_neg": RUN_P2,
                      "recordist_disjoint": RUN_P3}
        _skipped = {rt for rt in FORCE_RERUN if not _RUN_FLAGS.get(rt, False)}
        if _skipped:
            print(f"FORCE_RERUN names {sorted(_skipped)} but their RUN_P* flags are off "
                  f"-- KEEPING their checkpointed rows rather than deleting work nothing "
                  f"will rebuild.")
        _force = {rt for rt in FORCE_RERUN if _RUN_FLAGS.get(rt, False)}
        if _force:
            n_before = len(all_rows)
            all_rows = [r for r in all_rows if r["run_type"] not in _force]
            print(f"FORCE_RERUN {sorted(_force)}: dropped "
                  f"{n_before - len(all_rows)} checkpointed rows -- they will be recomputed")
        done = {(r["run_type"], r["species"]) for r in all_rows}
        print(f"RESUMING: {len(done)} (run_type, species) pairs already in {CKPT_PATH}")

def _save():
    pd.DataFrame(all_rows).to_csv(CKPT_PATH, index=False)

def _row(run_type, species, arm, n_seeds, metrics_mean, metrics_sd, **extra):
    return dict(run_type=run_type, species=species, arm=arm, n_seeds=n_seeds,
                auc=metrics_mean["auc"], auc_sd=metrics_sd["auc"],
                f1=metrics_mean["f1"], prec=metrics_mean["prec"], rec=metrics_mean["rec"],
                fa_per_hr=metrics_mean["fa_per_hr"], **extra)

print(f"\n{'='*90}\nCELL 5  CONFOUND-CLOSING RUNS\n{'='*90}")
print(f"species={list(CONFOUND_SPECIES)}  seeds={CONFOUND_SEEDS}  checkpoint={CKPT_PATH}")
print(f"this session: P1={'run' if RUN_P1 else 'from checkpoint'}  "
      f"P2={'run' if RUN_P2 else 'from checkpoint'} (seeds={P2_SEEDS})  "
      f"P3={'run' if RUN_P3 else 'from checkpoint'}")

# ======================================================================= P1: Perch alone
# PRE-REGISTERED: Perch was pretrained as a clip-level species classifier, never as a
# step-by-step localiser; stage-1's CNN is specifically trained per-target, few-shot, to
# localise. Prediction: Perch-alone, as an end-to-end detector, scores LOWER than the
# full stage-1+verify pipeline -- possibly lower than stage-1-alone too on F1/localisation
# -- even though Perch's species-discrimination given an already-localised candidate stays
# strong. Consistent with this project's own AP-vs-AUC finding: Perch's strength is
# discrimination given a candidate, not candidate generation. Reported either way.
print(f"\n{'-'*90}\nP1  PERCH ALONE AS A STANDALONE DETECTOR\n{'-'*90}")
t0 = time.time(); first_timed = False
if not RUN_P1:
    print("  RUN_P1=False -- settled 29 Aug at 2 seeds; verdict below reads the checkpoint.")
for sp in (CONFOUND_SPECIES if RUN_P1 else ()):
    if ("perch_alone", sp) in done:
        print(f"  [{sp}] already checkpointed, skipping"); continue
    try:
        ctx = build_target(sp, verbose=False)
    except Exception as ex:
        print(f"  skip {sp}: {type(ex).__name__}: {ex}"); continue
    acc = {"s1": [], "ver": [], "perch_alone": []}
    for sd in CONFOUND_SEEDS:
        ts = time.time()
        ec_f, sc_f, nc_f = eval_pass(ENC, ctx, sd), spec_pass(ENC, ctx, sd), neg_pass(ENC, ctx, sd)
        ec_p = eval_pass(ENC, ctx, sd, detect_fn=detect_perch_only)
        sc_p = spec_pass(ENC, ctx, sd, detect_fn=detect_perch_only)
        nc_p = neg_pass(ENC, ctx, sd, detect_fn=detect_perch_only)
        acc["s1"].append(_arm_metrics(FUSIONS[BASE], ec_f, sc_f, nc_f, ctx))
        acc["ver"].append(_arm_metrics(FUSIONS[CAND], ec_f, sc_f, nc_f, ctx))
        acc["perch_alone"].append(_arm_metrics(FUSIONS[CAND], ec_p, sc_p, nc_p, ctx))
        if not first_timed:
            first_timed = True
            per = time.time() - ts
            print(f"  [timing] one (species, seed) with BOTH detectors took {per:.0f}s "
                  f"-> ~{per * len(CONFOUND_SPECIES) * len(CONFOUND_SEEDS) / 60:.1f} min "
                  f"projected for all of P1. Widen PERCH_ALONE_STRIDE_S (CELL 2) if that's "
                  f"too slow.")
    for arm in acc:
        mean = {k: np.mean([d[k] for d in acc[arm]]) for k in acc[arm][0]}
        sd_  = {k: (np.std([d[k] for d in acc[arm]], ddof=1) if len(acc[arm]) > 1 else 0.0)
                for k in acc[arm][0]}
        all_rows.append(_row("perch_alone", sp, arm, len(CONFOUND_SEEDS), mean, sd_))
    _save()
    print(f"  [{sp}] stage1-alone AUC {all_rows[-3]['auc']:.3f}  "
          f"stage1+verify AUC {all_rows[-2]['auc']:.3f}  "
          f"perch-alone AUC {all_rows[-1]['auc']:.3f}  [{time.time()-t0:.0f}s elapsed]")
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
print(f"[P1 took {(time.time()-t0)/60:.1f} min]")

p1 = pd.DataFrame([r for r in all_rows if r["run_type"] == "perch_alone"])
if len(p1):
    g = p1.groupby("arm").agg(auc=("auc", "mean"), auc_sd=("auc_sd", "mean"),
                              f1=("f1", "mean")).reindex(["s1", "ver", "perch_alone"])
    print(f"\n{'arm':<14}{'AUC':>8}{'F1':>8}")
    for arm, r in g.iterrows():
        print(f"{arm:<14}{r['auc']:>8.3f}{r['f1']:>8.3f}")
    sd_ver = float(p1[p1.arm == "ver"].auc_sd.mean())
    sd_pa  = float(p1[p1.arm == "perch_alone"].auc_sd.mean())
    verdict("Does removing stage-1 entirely change detector AUC? (verified vs perch-alone)",
            float(g.loc["ver", "auc"]), float(g.loc["perch_alone", "auc"]), sd_ver, sd_pa,
            pos_note="Perch alone is BETTER than the full pipeline -- stage-1 is not "
                     "adding value as a candidate proposer for these species; worth "
                     "re-examining whether stage-1 is pulling its weight at all.",
            neg_note="Perch alone is WORSE than the full pipeline -- stage-1's "
                     "localisation is doing real work the verifier alone cannot replace. "
                     "Answers the interview question directly: the CNN's contribution is "
                     "candidate generation, not species discrimination (Perch already "
                     "wins that, per the AP-vs-AUC finding).",
            flat_note="Removing stage-1 doesn't move the number -- suggests Perch's own "
                      "sliding-window scores are enough to localise these calls, and "
                      "stage-1's contribution is smaller than assumed. Worth a wider "
                      "species set before leaning on this.")

# ======================================================================= P2: species-matched negatives
# PRE-REGISTERED: negative CHOICE is named as decisive twice already in this project's own
# literature review (Nolasco et al. sec.4.2; Liang et al., +5.31 F1 from hard-negative
# mining). Species-matched negatives are a strictly more informative signal than arbitrary
# background. Prediction: stage-1-alone AUC/F1 improves with USE_SPECIES_NEG=True. No
# strong prior on whether the gain propagates to the VERIFIED number -- the verifier
# already handles species discrimination and may already correct for whatever stage-1
# misses. Reported either way.
# CAVEAT, found by audit 28 Aug 2026 before this ever ran: the "species_matched" arm is
# not a clean substitution. N_NEG=200 1s calls at 0.4s gaps cannot all fit a 240s bed
# (needs >=280s); USE_SPECIES_NEG's real placement rate is ~126/200, background-padded to
# reach 200 total (see USE_SPECIES_NEG, CELL 2). So this tests "~126 species-matched + ~74
# background" against "~200 background", not "species-matched" against "background" as a
# pure factor. State results that way rather than as a clean substitution.
print(f"\n{'-'*90}\nP2  STAGE-1 FINE-TUNED WITH SPECIES-MATCHED NEGATIVES\n{'-'*90}")

def _set_species_neg(flag):
    g = globals()
    g["USE_SPECIES_NEG"] = flag
    assert USE_SPECIES_NEG == flag, "USE_SPECIES_NEG rebind failed"

# NOT a byte-identical A/B. eval_pass/spec_pass/neg_pass create one rng per seed and pass
# it into detect_fn for every bed in sequence; stage1() then consumes it by an amount that
# depends on USE_SPECIES_NEG (the True branch runs a full inject() call the False branch
# never does), so bed 2 onward draws different planting positions between the two arms of
# a "same seed" comparison. This is not new: USE_HARD_NEG in this same function already has
# the identical property, and the whole codebase's noise-floor discipline exists because of
# it -- CONFOUND_SEEDS averages over 2 seeds and verdict() compares against NOISE_FLOOR
# specifically so this is a fair STATISTICAL comparison, not a claim of paired identical
# draws. Stated explicitly here because P2 is the confound run where it matters most.
t0 = time.time()
_set_species_neg(False)   # defensive reset -- P1 never touches this flag, but don't assume
if not RUN_P2:
    print("  RUN_P2=False -- verdict below reads the checkpoint.")
for sp in (CONFOUND_SPECIES if RUN_P2 else ()):
    if ("species_neg", sp) in done:
        print(f"  [{sp}] already checkpointed, skipping"); continue
    try:
        ctx = build_target(sp, verbose=False)
    except Exception as ex:
        print(f"  skip {sp}: {type(ex).__name__}: {ex}"); continue
    if not ctx.get("neg_calls"):
        print(f"  skip {sp}: no neg_calls in ctx (no non-decoy candidate species found)")
        continue
    acc = {"background|s1": [], "background|ver": [],
           "species_matched|s1": [], "species_matched|ver": []}
    for use_species_neg, tag in ((False, "background"), (True, "species_matched")):
        _set_species_neg(use_species_neg)
        for sd in P2_SEEDS:
            ec, sc, nc = eval_pass(ENC, ctx, sd), spec_pass(ENC, ctx, sd), neg_pass(ENC, ctx, sd)
            acc[f"{tag}|s1"].append(_arm_metrics(FUSIONS[BASE], ec, sc, nc, ctx))
            acc[f"{tag}|ver"].append(_arm_metrics(FUSIONS[CAND], ec, sc, nc, ctx))
    _set_species_neg(False)   # reset before the next species / next confound section
    for arm in acc:
        mean = {k: np.mean([d[k] for d in acc[arm]]) for k in acc[arm][0]}
        sd_  = {k: (np.std([d[k] for d in acc[arm]], ddof=1) if len(acc[arm]) > 1 else 0.0)
                for k in acc[arm][0]}
        all_rows.append(_row("species_neg", sp, arm, len(P2_SEEDS), mean, sd_))
    _save()
    r = {rw["arm"]: rw for rw in all_rows if rw["run_type"] == "species_neg" and rw["species"] == sp}
    print(f"  [{sp}] s1 AUC {r['background|s1']['auc']:.3f} -> {r['species_matched|s1']['auc']:.3f}   "
          f"ver AUC {r['background|ver']['auc']:.3f} -> {r['species_matched|ver']['auc']:.3f}")
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
print(f"[P2 took {(time.time()-t0)/60:.1f} min]")

p2 = pd.DataFrame([r for r in all_rows if r["run_type"] == "species_neg"])
if len(p2):
    for tag in ("s1", "ver"):
        bg = p2[p2.arm == f"background|{tag}"]
        sm = p2[p2.arm == f"species_matched|{tag}"]
        if bg.empty or sm.empty: continue
        verdict(f"Does a species-matched negative pool change stage-1 fine-tuning? "
                f"({'stage-1 alone' if tag == 's1' else 'verified'} AUC)",
                float(bg.auc.mean()), float(sm.auc.mean()),
                float(bg.auc_sd.mean()), float(sm.auc_sd.mean()),
                pos_note="Species-matched negatives HELP -- stage-1's fine-tuning negative "
                         "choice matters, matching Liang et al./Nolasco et al. Worth "
                         "adopting as a rung, not just a confound check.",
                neg_note="Species-matched negatives HURT -- plausible if the pool is small "
                         "or dominated by acoustically-distant species, giving stage-1 "
                         "less varied negative exposure than random background did.",
                flat_note="Negative source doesn't move stage-1 -- background windows were "
                          "already an adequate negative signal for this task; the "
                          "literature's finding may not transfer to this planting protocol.")

# ======================================================================= P3: recordist-disjoint
# PRE-REGISTERED: if ARGUS's specificity number is genuinely about the SPECIES rather than
# the recording SETUP, a recordist-disjoint held-out split should barely move AUC. A real
# drop would mean part of the measured signal has been "which microphone/recordist", not
# "which bird" -- a genuine confound this project has not yet ruled out. No strong prior
# on magnitude either way; reported honestly regardless of outcome, per this project's
# established practice throughout this notebook.
print(f"\n{'-'*90}\nP3  RECORDIST-DISJOINT HELD-OUT SPLIT\n{'-'*90}")
t0 = time.time()
if not RUN_P3:
    print("  RUN_P3=False -- settled 29 Aug at 2 seeds; verdict below reads the checkpoint.")
for sp in (CONFOUND_SPECIES if RUN_P3 else ()):
    if ("recordist_disjoint", sp) in done:
        print(f"  [{sp}] already checkpointed, skipping"); continue
    acc = {"shared_ok|s1": [], "shared_ok|ver": [],
           "disjoint|s1": [], "disjoint|ver": []}
    n_heldout, n_recordists = {}, {}
    skip_species = False
    for disjoint, tag in ((False, "shared_ok"), (True, "disjoint")):
        try:
            ctx = build_target(sp, recordist_disjoint=disjoint, verbose=False)
        except (ValueError, KeyError) as ex:
            print(f"  [{sp}] {tag}: {type(ex).__name__}: {ex}")
            if disjoint:
                print(f"  skip {sp}: recordist-disjoint split leaves no held-out pool "
                      f"(too few distinct recordists for this species) -- itself worth "
                      f"logging, not a bug to route around.")
            skip_species = True
            break
        n_heldout[tag] = ctx["covariates"]["n_heldout_calls"]
        n_recordists[tag] = ctx["covariates"]["n_distinct_recordists"]
        for sd in CONFOUND_SEEDS:
            ec, sc, nc = eval_pass(ENC, ctx, sd), spec_pass(ENC, ctx, sd), neg_pass(ENC, ctx, sd)
            acc[f"{tag}|s1"].append(_arm_metrics(FUSIONS[BASE], ec, sc, nc, ctx))
            acc[f"{tag}|ver"].append(_arm_metrics(FUSIONS[CAND], ec, sc, nc, ctx))
    if skip_species:
        continue
    for arm in acc:
        mean = {k: np.mean([d[k] for d in acc[arm]]) for k in acc[arm][0]}
        sd_  = {k: (np.std([d[k] for d in acc[arm]], ddof=1) if len(acc[arm]) > 1 else 0.0)
                for k in acc[arm][0]}
        tag = arm.split("|")[0]
        all_rows.append(_row("recordist_disjoint", sp, arm, len(CONFOUND_SEEDS), mean, sd_,
                             n_heldout_calls=n_heldout[tag], n_distinct_recordists=n_recordists[tag]))
    _save()
    r = {rw["arm"]: rw for rw in all_rows
         if rw["run_type"] == "recordist_disjoint" and rw["species"] == sp}
    print(f"  [{sp}] shared-ok ver AUC {r['shared_ok|ver']['auc']:.3f} "
          f"(n_heldout={n_heldout['shared_ok']})  ->  "
          f"disjoint ver AUC {r['disjoint|ver']['auc']:.3f} "
          f"(n_heldout={n_heldout['disjoint']}, distinct_recordists={n_recordists['disjoint']})")
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
print(f"[P3 took {(time.time()-t0)/60:.1f} min]")

p3 = pd.DataFrame([r for r in all_rows if r["run_type"] == "recordist_disjoint"])
if len(p3):
    for tag in ("s1", "ver"):
        so = p3[p3.arm == f"shared_ok|{tag}"]
        dj = p3[p3.arm == f"disjoint|{tag}"]
        if so.empty or dj.empty: continue
        verdict(f"Does a recordist-disjoint held-out split change AUC? "
                f"({'stage-1 alone' if tag == 's1' else 'verified'})",
                float(so.auc.mean()), float(dj.auc.mean()),
                float(so.auc_sd.mean()), float(dj.auc_sd.mean()),
                pos_note="AUC is HIGHER once recordist overlap is removed -- unexpected; "
                         "check the held-out pool sizes above before trusting this, small "
                         "pools swing more.",
                neg_note="AUC FALLS once recordist overlap is removed -- part of the "
                         "measured signal was 'recognise this recordist's setup', not "
                         "purely the species. A real, reportable limitation -- state the "
                         "size of the drop, don't just flag its direction.",
                flat_note="AUC does not depend on recordist overlap -- the specificity "
                          "result is not an artefact of recording-setup leakage. Rules "
                          "out the 'did it learn the microphone, not the bird' concern "
                          "for these species.")

print(f"\n{'='*90}\nALL CONFOUND-CLOSING RUNS DONE -- {len(all_rows)} rows -> {CKPT_PATH}")
print(f"Log the three verdicts above in the bound logbook the same day.\n{'='*90}")
