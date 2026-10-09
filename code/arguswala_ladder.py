# ===== CELL 4 / ABLATION LADDER =====
# Run AFTER arguswala_v2.py (reuses its trained `encoders`, passes and scorers).
#
# REVISED 14 Aug 2026 after the first factorial run was killed at config 6/16.
# Three faults in the previous version, all fixed here:
#
#  (1) NO CHECKPOINTING. Results were only written to CSV after the whole loop finished,
#      so a killed session lost every completed config. ~82 min of real compute was
#      recoverable only from scrollback. Now each config is appended to CSV the moment
#      it finishes, and a re-run resumes instead of restarting.
#
#  (2) CATASTROPHIC CONFIG ORDERING. itertools.product with VERIFIER first made it the
#      slowest-varying factor -- all 8 logreg configs ran before any proto config. The
#      run died at 6/16, so it produced ZERO data on the prototypical probe, which was
#      the entire reason for running a factorial. Configs are now ordered by DISTANCE
#      FROM BASELINE, so the first 5 cover the baseline plus every single-factor main
#      effect. A truncated run now still answers the main question.
#
#  (3) NO NOISE FLOOR. The previous run accidentally measured one: `pcen|logreg` scored
#      0.696 in the cumulative run and 0.731 in the factorial run -- identical config,
#      identical seeds, 0.035 apart, purely from GPU non-determinism (cuDNN conv backward
#      is not bit-reproducible even with a fixed seed). Without that number, a delta of
#      -0.005 was reported as "HURTS (d/SE -3.02)" when it is in fact indistinguishable
#      from zero. The baseline is now deliberately repeated across encoder seeds to
#      measure the floor, and every delta is judged against it.
import numpy as np, pandas as pd, time, itertools, os, gc

LADDER_SPECIES = TARGET_LIST[:3]     # widen once the timing measurement says it fits
LADDER_SEEDS   = (7, 8)
FULL_FACTORIAL = True

# TimeFilterAug is EXCLUDED by default as of 14 Aug 2026. Measured in isolation
# (hn=False, both front-ends) it hurt every metric: verified AUC -0.155 (pcen) and
# -0.088 (logmel), F1 0.713->0.522 and 0.483->0.362, and stage-1-alone AUC 0.493->0.407
# -- that last one is verifier-independent, so the damage is unambiguously in stage-1
# fine-tuning, not an interaction with the probe. At 4.4x the measured noise floor this
# is a real effect, not scatter. Dropping it halves the design (16 -> 8 configs).
# Set False to put it back and re-test.
SKIP_TIMEFILTER = True

# The baseline is re-run once per encoder seed to establish the run-to-run noise floor.
# Deltas smaller than this are not evidence, regardless of what a within-run SE says.
BASELINE_ENCODER_SEEDS = ENCODER_SEEDS          # from CELL 2; (0,1,2) after the widening
OTHER_ENCODER_SEED     = ENCODER_SEEDS[0]

CKPT_PATH = "/kaggle/working/argus_ablation_ladder.csv"
RESUME    = True     # skip (config, encoder_seed) pairs already present in CKPT_PATH

# 24 Aug 2026: the ladder finished a complete 8-config run (all 10 planned rows) that day
# -- see the lab notebook Day 29 entry. RESUME's checkpoint only helps within ONE Kaggle
# session (/kaggle/working/ is wiped between sessions unless manually re-attached), so a
# fresh session re-derives an answer this project already has rather than resuming past
# it. SKIP_LADDER=True skips this cell's run loop entirely when you only need CELL 3
# (e.g. just the decoy-difficulty sweep, S4) and the ladder's own conclusion isn't what
# you're re-running for. Leave False for a normal run, or when re-verifying the ladder is
# actually the point (e.g. re-testing at a wider species count).
SKIP_LADDER = False

BASE, CAND = "stage-1 only  (p)", "perch only    (v)"
BASELINE_CFG = dict(VERIFIER="logreg", USE_TIMEFILTER=False, USE_HARD_NEG=False, FEATURE="pcen")

CUMULATIVE = [
    ("rung 0  baseline",        dict(VERIFIER="logreg", USE_TIMEFILTER=False, USE_HARD_NEG=False, FEATURE="pcen")),
    ("rung 1  +proto probe",    dict(VERIFIER="proto",  USE_TIMEFILTER=False, USE_HARD_NEG=False, FEATURE="pcen")),
    ("rung 2  +timefilter",     dict(VERIFIER="proto",  USE_TIMEFILTER=True,  USE_HARD_NEG=False, FEATURE="pcen")),
    ("rung 3  +hard negatives", dict(VERIFIER="proto",  USE_TIMEFILTER=True,  USE_HARD_NEG=True,  FEATURE="pcen")),
    ("rung 4  +log-Mel",        dict(VERIFIER="proto",  USE_TIMEFILTER=True,  USE_HARD_NEG=True,  FEATURE="logmel")),
]

def _tag(cfg):
    t = f"{cfg['FEATURE']}|{cfg['VERIFIER']}"
    if cfg["USE_TIMEFILTER"]: t += "+tfa"
    if cfg["USE_HARD_NEG"]:   t += "+hn"
    return t

def _dist_from_baseline(cfg):
    return sum(1 for k, v in BASELINE_CFG.items() if cfg[k] != v)

def _factorial():
    tfa_levels = (False,) if SKIP_TIMEFILTER else (False, True)
    out = []
    for v, t, hn, ft in itertools.product(("logreg", "proto"), tfa_levels,
                                          (False, True), ("pcen", "logmel")):
        cfg = dict(VERIFIER=v, USE_TIMEFILTER=t, USE_HARD_NEG=hn, FEATURE=ft)
        out.append((_tag(cfg), cfg))
    # Order by how many factors differ from baseline. With 4 binary factors this yields
    # 1 baseline, then all single-factor changes, then pairs, then triples. A run killed
    # partway still delivers every main effect -- which is what the previous ordering
    # failed to do.
    out.sort(key=lambda kc: (_dist_from_baseline(kc[1]), kc[0]))
    return out

CONFIGS = _factorial() if FULL_FACTORIAL else CUMULATIVE
BASELINE_LABEL = _tag(BASELINE_CFG)

def _apply(cfg):
    """CELL 1 and CELL 2 share one notebook namespace, and _make_verifier(), aug_pos(),
    stage1() and feat() all read these as globals at CALL time -- so rebinding here is
    what switches the config. Verified by asserting the read-back below."""
    g = globals()
    for k, val in cfg.items():
        g[k] = val
    assert (VERIFIER, USE_TIMEFILTER, USE_HARD_NEG, FEATURE) == \
           (cfg["VERIFIER"], cfg["USE_TIMEFILTER"], cfg["USE_HARD_NEG"], cfg["FEATURE"]), \
           "flag rebind failed"

def run_config(label, cfg, enc_seed):
    _apply(cfg)
    # The encoder MUST match the front-end: one trained on PCEN cannot be evaluated on
    # log-Mel inputs, and doing so measures a mismatch rather than the front-end.
    key = (FEATURE, enc_seed)
    if key not in encoders:
        raise KeyError(f"no encoder for {key}; add '{FEATURE}' to FEATURES in CELL 2 "
                       f"and re-run the encoder training. Have: {sorted(encoders)}")
    ENC = encoders[key]
    rows = []
    for sp in LADDER_SPECIES:
        try:
            ctx = build_target(sp, verbose=False)     # cached; only the verifier refits
        except Exception as ex:
            print(f"    skip {sp}: {type(ex).__name__}: {ex}"); continue
        acc = {BASE: [], CAND: []}
        for sd in LADDER_SEEDS:
            ec = eval_pass(ENC, ctx, sd)
            sc = spec_pass(ENC, ctx, sd)
            nc = neg_pass(ENC, ctx, sd)
            for name in (BASE, CAND):
                mm = score_eval(FUSIONS[name], ec)
                _, _, _, _, auc = score_spec(FUSIONS[name], mm["thr"], sc, ctx["DECOYS"])
                acc[name].append((auc, mm["f1"], mm["p"], mm["r"],
                                  score_neg(FUSIONS[name], mm["thr"], nc)))
        row = dict(config=label, species=sp, encoder_seed=enc_seed, **{k: cfg[k] for k in cfg})
        for name, tag in ((BASE, "s1"), (CAND, "ver")):
            a = np.array(acc[name], dtype=float)
            row[f"auc_{tag}"]  = a[:, 0].mean()
            row[f"f1_{tag}"]   = a[:, 1].mean()
            row[f"prec_{tag}"] = a[:, 2].mean()
            row[f"rec_{tag}"]  = a[:, 3].mean()
            row[f"fa_{tag}"]   = a[:, 4].mean()
        rows.append(row)
    return rows

# ------------------------------------------------------------------ build the run plan
# Baseline gets every encoder seed (noise floor); everything else gets one.
PLAN = []
for label, cfg in CONFIGS:
    seeds = BASELINE_ENCODER_SEEDS if label == BASELINE_LABEL else (OTHER_ENCODER_SEED,)
    for es in seeds:
        PLAN.append((label, cfg, es))

done = set()
all_rows = []
if RESUME and os.path.exists(CKPT_PATH):
    prev = pd.read_csv(CKPT_PATH)
    if {"config", "encoder_seed"}.issubset(prev.columns):
        all_rows = prev.to_dict("records")
        done = {(r["config"], int(r["encoder_seed"])) for r in all_rows}
        print(f"RESUMING: {len(done)} (config, encoder_seed) pairs already in {CKPT_PATH}")

todo = [(l, c, e) for (l, c, e) in PLAN if (l, int(e)) not in done]
print(f"\n{'='*100}\nABLATION LADDER  |  {len(PLAN)} runs planned "
      f"({len(CONFIGS)} configs, baseline x{len(BASELINE_ENCODER_SEEDS)} for noise floor) "
      f"| {len(todo)} to run\n"
      f"  species={list(LADDER_SPECIES)}  eval seeds={LADDER_SEEDS}  "
      f"timefilter={'EXCLUDED' if SKIP_TIMEFILTER else 'included'}\n"
      f"  checkpointing to {CKPT_PATH} after EVERY config -- a killed session resumes, "
      f"it does not restart.\n{'='*100}")

if SKIP_LADDER:
    print("SKIP_LADDER=True -- skipping the ladder's run loop entirely.")
    print("This cell already has a complete answer from a prior session (see the")
    print("lab notebook Day 29 entry); re-running it here would just re-derive it.")
else:
    t0 = time.time()
    for i, (label, cfg, es) in enumerate(todo, 1):
        t1 = time.time()
        r = run_config(label, cfg, es)
        all_rows += r
        # Write immediately. This is the whole point -- 82 min of compute was lost last run.
        pd.DataFrame(all_rows).to_csv(CKPT_PATH, index=False)
        if r:
            d = pd.DataFrame(r)
            print(f"[{i:2d}/{len(todo)}] {label:<22} enc{es}  "
                  f"AUC {d.auc_s1.mean():.3f} -> {d.auc_ver.mean():.3f}   "
                  f"F1 {d.f1_ver.mean():.3f}   FA/h {d.fa_ver.mean():5.1f}   "
                  f"[{time.time()-t1:.0f}s]  saved")
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    lad = pd.DataFrame(all_rows)
    print(f"\ntotal {(time.time()-t0)/60:.1f} min | {len(lad)} rows -> {CKPT_PATH}")

    # ------------------------------------------------------------------ noise floor first
    noise = None
    b = lad[lad.config == BASELINE_LABEL]
    if not b.empty and b.encoder_seed.nunique() > 1:
        per_seed = b.groupby("encoder_seed").auc_ver.mean()
        noise = float(per_seed.max() - per_seed.min())
        print(f"\n{'='*100}\nNOISE FLOOR -- baseline '{BASELINE_LABEL}' repeated across "
              f"{per_seed.size} encoder seeds\n{'='*100}")
        print("  verified AUC per encoder seed: " + "  ".join(f"{v:.3f}" for v in per_seed))
        print(f"  spread {noise:+.3f}   sd {per_seed.std(ddof=1):.3f}")
        print("  -> any delta below this spread is NOT evidence, whatever a within-run SE says.")
        print("  (Cross-session noise is larger still: the same config scored 0.696 and 0.731 "
              "in two\n   separate sessions -- 0.035 apart -- from GPU non-determinism alone.)")

    # ------------------------------------------------------------------ report
    print(f"\n{'='*100}\n{'config':<24}{'enc':>4}{'AUC stage-1':>13}{'AUC verified':>14}"
          f"{'delta vs base':>15}{'F1 ver':>9}{'FA/hr':>8}\n{'-'*100}")
    base_auc = b.auc_ver.mean() if not b.empty else np.nan
    for label, cfg in CONFIGS:
        g = lad[lad.config == label]
        if g.empty: continue
        a = g.auc_ver.mean()
        d = a - base_auc
        flag = ""
        if noise and label != BASELINE_LABEL:
            flag = "  <- within noise" if abs(d) < noise else ("  <- REAL" if d < 0 else "  <- REAL +")
        print(f"{label:<24}{g.encoder_seed.nunique():>4}{g.auc_s1.mean():>13.3f}{a:>14.3f}"
              f"{d:>+15.3f}{g.f1_ver.mean():>9.3f}{g.fa_ver.mean():>8.1f}{flag}")

    # ------------------------------------------------------------------ main effects
    if FULL_FACTORIAL and not lad.empty:
        print(f"\nMAIN EFFECTS (technique ON minus OFF, paired within species, "
              f"averaged over all other settings):")
        for flag, on_vals, off_vals in (("VERIFIER", ["proto"], ["logreg"]),
                                        ("USE_TIMEFILTER", [True], [False]),
                                        ("USE_HARD_NEG", [True], [False]),
                                        ("FEATURE", ["logmel"], ["pcen"])):
            if flag not in lad.columns: continue
            g_on, g_off = lad[lad[flag].isin(on_vals)], lad[lad[flag].isin(off_vals)]
            if g_on.empty or g_off.empty:
                print(f"  {flag:<34} not tested in this run"); continue
            # Collapse each CONFIG to one value per species BEFORE averaging configs.
            # A plain groupby("species").mean() is row-weighted, and the baseline carries
            # 3x the rows of every other config (it is repeated across BASELINE_ENCODER_SEEDS
            # to measure the noise floor, while challengers run at OTHER_ENCODER_SEED only).
            # So the baseline -- the highest-scoring config -- was counted three times inside
            # the "off" arm of VERIFIER, USE_HARD_NEG and FEATURE(pcen), inflating every
            # deficit: VERIFIER read -0.048 (d/SE -5.35) when the paired value is -0.023
            # (d/SE -1.84), which does not clear this file's own "|d/SE| < 2 is NOT evidence"
            # bar. That inflated number reached the public site; retracted 28 Aug 2026, see
            # the lab notebook Day 34. Config-weighting makes each config count once.
            on_s  = g_on.groupby(["species", "config"]).auc_ver.mean().groupby("species").mean()
            off_s = g_off.groupby(["species", "config"]).auc_ver.mean().groupby("species").mean()
            common = on_s.index.intersection(off_s.index)
            if len(common) < 2: continue
            d = (on_s.loc[common] - off_s.loc[common]).to_numpy()
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            verdict = ""
            if noise:
                verdict = "  within noise floor" if abs(d.mean()) < noise else "  exceeds noise floor"
            print(f"  {flag} ({on_vals[0]} vs {off_vals[0]}):".ljust(36)
                  + f"{d.mean():+.3f}  SE {se:.3f}  d/SE {d.mean()/se if se else float('nan'):>6.2f}{verdict}")
        print("  NOTE: within-run SE ignores GPU non-determinism between runs. Trust the "
              "noise floor over d/SE.")

    print("\nLog the noise floor and the surviving effects in the bound logbook the same day.")
