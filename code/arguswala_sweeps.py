# ===== CELL 3 / SWEEPS -- shot selection, shot count, front-end =====
# Run AFTER arguswala_v2.py (reuses its trained `encoders`, scoring fns and eval passes).
#
# Five experiments the literature says matter and nobody has published answers for on
# this data:
#
#  N1  SHOT-SELECTION STRATEGY.  Nolasco et al. (2023) sec.4.1: unrepresentative support
#      sets were the single failure cause their expert annotators raised most often --
#      "the five shots ... are not representative of the range of possible [calls]".
#      DCASE always takes the FIRST five. ARGUS v1 always took the highest-RATED five.
#      Nobody has measured what that choice is worth. Four strategies, same everything else.
#
#  S2  SHOT COUNT.  ARGUS is a few-shot project that has never varied the number of shots.
#      First thing a judge asks. Where the curve plateaus is a deployable field number.
#
#  S3  SPECIFICITY vs SNR.  Every specificity number this project has reported was
#      measured at a fixed +6 dB -- loud, close calls. Roadmap 3.2 calls this a real hole,
#      with a pre-registered prediction attached. A deployment cannot choose its SNR.
#
#  S4  DECOY DIFFICULTY.  Decoys have always been the top-N_DECOYS hardest confusers, on
#      purpose -- never measured against the alternative. Roadmap 3.3.
#
#  N2  FRONT-END.  Liang et al. (DCASE 2024 baseline) Table III: PCEN has the best
#      PRECISION of any front-end tested (68.0%), log-Mel the best F1 (63.7 vs 60.0).
#      ARGUS uses PCEN. Confirm or contradict that trade on Western Ghats data.
#      NOTE: changing FEATURE changes the encoder bank too, so this needs a fresh
#      encoder -- it is NOT a free re-score. Budget one extra encoder training.
import numpy as np, pandas as pd, time, torch

# 29 Aug 2026: widened from 3 species x 2 seeds (= 6 measurements per condition) to
# 5 x 3 (= 15). The old figure sat below the 10-per-condition bar the IRIS/ISEF rubric
# names explicitly as a pitfall ("sample size of three ... won't survive statistical
# testing"), and the sweeps were the weakest link in the project on that criterion -- the
# centrepiece already carries n=10 species for its paired test, the confounds carry 5.
#
# Species were widened in preference to seeds at equal cost: 5x3 and 3x5 both cost 2.5x
# the old run, but extra species test whether a sweep's verdict GENERALISES, where extra
# seeds only tighten the noise estimate on the same three. It also aligns SWEEP_SPECIES
# with CONFOUND_SPECIES (both TARGET_LIST[:5]), so Cell 3 and Cell 5 now describe the same
# species set rather than two overlapping ones.
#
# Projected cost from run 3's timings (N1 3350s + S2 3331s + S3 2869s + S4 2614s = 3.38h
# at 3x2): ~8.5h at 5x3, plus ~10min setup/encoders. Fits the 12h ceiling with ~3h spare.
# Each sweep writes its CSV on completion, so a session killed during S4 still keeps
# N1/S2/S3.
SWEEP_SPECIES = TARGET_LIST[:5]      # sweeps are O(strategies x species x seeds); keep narrow
SWEEP_SEEDS   = (7, 8, 9)            # fewer than the main run -- these are relative comparisons
BASE, CAND    = "stage-1 only  (p)", "perch only    (v)"
ENC           = encoders[(FEATURE, ENCODER_SEEDS[0])]   # encoders are keyed by front-end

# 28 Aug 2026, after run 4 (12h ceiling, killed mid-ladder, Cell 5 never reached):
# all four sweeps below (N1/S2/S3/S4) have now independently replicated twice -- 16 Aug
# and 25 Aug -- with the same qualitative verdict both times (audited against run 4's own
# console log the same day; see the lab notebook Day 34). SKIP_SWEEPS=True skips this
# cell's ~3.4h of already-answered compute so a Cell-5-focused session goes straight to
# the one thing genuinely unrun. Same revert-after-use discipline as SKIP_MAIN_LOOP /
# SKIP_LADDER: leave False for a normal run, or when re-verifying a sweep is the point.
#
# 29 Aug 2026 -- THREE SESSIONS, RUN SEPARATELY. Never combine this cell with another.
#   [DONE 29 Aug] P2 re-run: SKIP_SWEEPS=True, RUN_P2=True -> ran 3h40m, settled P2
#                 (stage-1 FLAT, verified FALLS -0.050, negative in 5/5 species).
#   [THIS BUILD]  widened sweeps: SKIP_SWEEPS=False, everything else off -> ~8.7h.
#   [PENDING]     distillation: Cell 6 SKIP_DISTILL=False, this cell back to True -> ~3h.
# The sweeps alone project to ~8.7h against a 12h ceiling; adding either other session
# overruns it, which is how an earlier run was lost mid-ladder.
SKIP_SWEEPS = False

def run_one(sp, n_shot, strategy, seeds=SWEEP_SEEDS, decoy_mode="hard"):
    """One (species, n_shot, strategy, decoy_mode) cell -> mean metrics over seeds.
    decoy_mode defaults to 'hard' (roadmap 3.3's meaning of "unchanged") so N1/S2 above
    are byte-for-byte the same run they always were; only S4 below passes anything else."""
    try:
        ctx = build_target(sp, n_shot=n_shot, strategy=strategy, verbose=False,
                            decoy_mode=decoy_mode)
    except Exception as ex:
        print(f"    skip {sp} n_shot={n_shot} {strategy} {decoy_mode}: "
              f"{type(ex).__name__}: {ex}")
        return None
    acc = {BASE: [], CAND: []}
    for sd in seeds:
        ec = eval_pass(ENC, ctx, sd)
        sc = spec_pass(ENC, ctx, sd)
        for name in (BASE, CAND):
            m = score_eval(FUSIONS[name], ec)
            _, _, _, _, auc = score_spec(FUSIONS[name], m["thr"], sc, ctx["DECOYS"])
            acc[name].append((auc, m["f1"], m["p"], m["r"]))
    out = dict(species=sp, n_shot=n_shot, strategy=strategy, decoy_mode=decoy_mode,
               n_heldout=ctx["covariates"]["n_heldout_calls"],
               decoy_sim_top1=ctx["covariates"]["decoy_sim_top1"])
    for name, tag in ((BASE, "s1"), (CAND, "ver")):
        a = np.array(acc[name], dtype=float)
        out[f"auc_{tag}"] = a[:, 0].mean(); out[f"auc_{tag}_sd"] = a[:, 0].std(ddof=1) if len(a) > 1 else 0.0
        out[f"f1_{tag}"]  = a[:, 1].mean()
        out[f"prec_{tag}"] = a[:, 2].mean(); out[f"rec_{tag}"] = a[:, 3].mean()
    return out

if not SKIP_SWEEPS:
    # ------------------------------------------------------------------ N1: shot selection
    print(f"\n{'='*84}\nN1  SHOT-SELECTION STRATEGY  (n_shot={N_SHOT} held constant)\n{'='*84}")
    t0 = time.time(); rows_n1 = []
    for strategy in ("first", "rated", "random", "diverse"):
        for sp in SWEEP_SPECIES:
            r = run_one(sp, N_SHOT, strategy)
            if r: rows_n1.append(r); print(f"  {strategy:<8} {sp:<12} "
                                           f"AUC {r['auc_s1']:.3f} -> {r['auc_ver']:.3f}   "
                                           f"F1 {r['f1_s1']:.3f} -> {r['f1_ver']:.3f}")
    n1 = pd.DataFrame(rows_n1)
    if len(n1):
        n1.to_csv("/kaggle/working/argus_sweep_shot_strategy.csv", index=False)
        print(f"\n{'strategy':<10}{'AUC stage-1':>13}{'AUC verified':>14}{'F1 stage-1':>12}{'F1 verified':>13}")
        print("-"*62)
        for s, g in n1.groupby("strategy", sort=False):
            print(f"{s:<10}{g.auc_s1.mean():>13.3f}{g.auc_ver.mean():>14.3f}"
                  f"{g.f1_s1.mean():>12.3f}{g.f1_ver.mean():>13.3f}")
        best = n1.groupby("strategy").auc_ver.mean().idxmax()
        spread = n1.groupby("strategy").auc_ver.mean()
        print(f"\nbest strategy by verified AUC: {best}  "
              f"(spread across strategies: {spread.max()-spread.min():.3f} AUC)")
        print("If that spread exceeds the verification gain itself, then HOW the five shots are"
              "\nchosen matters more than the architecture -- which is a publishable observation"
              "\nand exactly the open problem Nolasco et al. sec.4.1 raise.")
    print(f"[N1 took {(time.time()-t0)/60:.1f} min]")

    # ------------------------------------------------------------------ S2: shot count
    print(f"\n{'='*84}\nS2  SHOT COUNT  (strategy='{SHOT_STRATEGY}' held constant)\n{'='*84}")
    t0 = time.time(); rows_s2 = []
    for n_shot in (1, 3, 5, 10):
        for sp in SWEEP_SPECIES:
            r = run_one(sp, n_shot, SHOT_STRATEGY)
            if r: rows_s2.append(r); print(f"  n_shot={n_shot:<3} {sp:<12} "
                                           f"AUC {r['auc_s1']:.3f} -> {r['auc_ver']:.3f}   "
                                           f"F1 {r['f1_s1']:.3f} -> {r['f1_ver']:.3f}")
    s2 = pd.DataFrame(rows_s2)
    if len(s2):
        s2.to_csv("/kaggle/working/argus_sweep_shot_count.csv", index=False)
        print(f"\n{'n_shot':<8}{'AUC stage-1':>13}{'AUC verified':>14}{'F1 verified':>13}")
        print("-"*48)
        for n, g in s2.groupby("n_shot"):
            print(f"{n:<8}{g.auc_s1.mean():>13.3f}{g.auc_ver.mean():>14.3f}{g.f1_ver.mean():>13.3f}")
        print("\nRead the plateau, not the peak: the smallest n_shot within noise of the best is"
              "\nthe number a field team actually has to collect.")
    print(f"[S2 took {(time.time()-t0)/60:.1f} min]")

    # ------------------------------------------------------------- S3: specificity vs SNR
    # Roadmap 3.2 -- the last unmeasured axis in the scenario matrix. spec_pass() has always
    # planted at a fixed +6 dB, so EVERY species-discrimination number this project has ever
    # reported -- including the 0.751 centrepiece -- describes loud, close calls only. A field
    # deployment does not get to choose the SNR.
    #
    # PRE-REGISTERED PREDICTION (roadmap 3.2, written before this ever ran):
    #   "the verifier's advantage shrinks as calls get fainter."
    #   If true  -> an honest limitation with a number attached, and the centrepiece result
    #               must be quoted as a HIGH-SNR result from then on.
    #   If false -> a stronger claim than the project currently makes.
    # Both outcomes get reported. Nothing about the verdict is decided after seeing the curve.
    SNR_SWEEP_SPECIES = SWEEP_SPECIES     # same 3 as N1/S2, so all three sweeps stay comparable
    SNR_SWEEP_SEEDS   = SWEEP_SEEDS
    SNR_SWEEP_LEVELS  = SNR_LEVELS        # (9, 6, 3, 0, -3, -6); trim first if budget is tight
    NOISE_FLOOR       = 0.025             # measured in-session Day 26-27. Nothing below it is real.

    def run_snr(sp, seeds=SNR_SWEEP_SEEDS, levels=SNR_SWEEP_LEVELS):
        """One species -> one row per (seed, SNR level, fusion).

        eval_pass runs ONCE per seed rather than once per level. It exists here only to set the
        operating threshold, and its own planting protocol is SNR-independent by design (it
        rotates through SNR_LEVELS internally). Re-running it per level would cost 6x for an
        identical threshold -- and would also be the wrong experiment. Calibrating once and then
        varying the call level IS the deployment case: you fix a detector's operating point, and
        the forest hands you whatever it hands you.
        """
        try:
            ctx = build_target(sp, verbose=False)
        except Exception as ex:
            print(f"    skip {sp}: {type(ex).__name__}: {ex}")
            return []
        rows = []
        for sd in seeds:
            ec  = eval_pass(ENC, ctx, sd)
            thr = {name: score_eval(FUSIONS[name], ec)["thr"] for name in (BASE, CAND)}
            for snr in levels:
                sc  = spec_pass(ENC, ctx, sd, snr_db=float(snr))
                cov = spec_coverage(sc)          # fusion-independent, so computed once per level
                for name, tag in ((BASE, "s1"), (CAND, "ver")):
                    t, md, ratio, _per, auc = score_spec(FUSIONS[name], thr[name], sc, ctx["DECOYS"])
                    auc_det, n_td, n_dd = score_spec_detected(FUSIONS[name], sc)
                    rows.append(dict(species=sp, seed=sd, snr_db=int(snr), fusion=tag,
                                     auc=auc, auc_det=auc_det, n_tgt_det=n_td, n_dec_det=n_dd,
                                     tgt_rate=t, decoy_rate=md, ratio=ratio,
                                     cov_target=cov["target"], cov_decoy=cov["decoy"],
                                     thr=float(thr[name])))
            print(f"    {sp:<12} seed {sd} done ({len(levels)} levels)")
        return rows

    _np = len(SNR_SWEEP_SPECIES) * len(SNR_SWEEP_SEEDS) * (1 + len(SNR_SWEEP_LEVELS))
    print(f"\n{'='*84}\nS3  SPECIFICITY vs SNR  (roadmap 3.2)\n{'='*84}")
    print(f"{len(SNR_SWEEP_SPECIES)} species x {len(SNR_SWEEP_SEEDS)} seeds x "
          f"({len(SNR_SWEEP_LEVELS)} levels + 1 threshold pass) = {_np} detection passes "
          f"~= {_np*1.1:.0f} min at the measured 1.1 min/pass.")
    print("PRE-REGISTERED: the verification gain is predicted to SHRINK as SNR falls.")
    print("Reported either way -- a flat curve is the stronger result, not a failed run.\n")

    t0 = time.time(); rows_s3 = []
    for sp in SNR_SWEEP_SPECIES:
        rows_s3 += run_snr(sp)
    s3 = pd.DataFrame(rows_s3)

    if len(s3):
        s3.to_csv("/kaggle/working/argus_sweep_snr.csv", index=False)
        print(f"\n{'SNR':>6}{'AUC s1':>9}{'AUC ver':>9}{'gain':>8}{'sd':>7}"
              f"{'AUCdet s1':>11}{'AUCdet ver':>12}{'cov tgt':>9}{'cov dec':>9}")
        print("-"*80)
        curve = {}
        for snr in sorted(s3.snr_db.unique(), reverse=True):
            d  = s3[s3.snr_db == snr]
            s1 = d[d.fusion == "s1"]; ve = d[d.fusion == "ver"]
            # coverage is fusion-independent, so either subset reports it
            # EVERY statistic gets its OWN noise bound. Until 28 Aug 2026 this dict computed
            # exactly two sds -- both from `ve`, the VERIFIED subset -- and trend() was handed
            # one of them for all four tests below. That silently mismatched three of the four:
            #   [1]/[2] test the GAIN (ver - s1) but were bounded by the sd of `ver` ALONE;
            #   [4] tests a STAGE-1 statistic but was bounded by the VERIFIED series' sd.
            # Only [3] (a verified statistic bounded by verified sd) was ever scored correctly.
            # The [4] mismatch was the damaging one: stage-1's localised AUC is ~1.7x noisier
            # than the verifier's, so the bound was 1.7x too TIGHT and [4] printed SHRINKS for a
            # change that does not clear stage-1's own noise. That verdict reached the public
            # site as "the verifier is what confers the SNR-invariance" -- retracted 28 Aug 2026,
            # see the lab notebook Day 34. The gain bound is paired per (species, seed) because
            # sd(A-B) != sd(A) when A and B are correlated across the same eval pass.
            gain_paired = (ve.sort_values(["species", "seed"]).auc.to_numpy()
                           - s1.sort_values(["species", "seed"]).auc.to_numpy())
            curve[int(snr)] = c = dict(
                auc_s1=float(s1.auc.mean()), auc_ver=float(ve.auc.mean()),
                gain=float(ve.auc.mean() - s1.auc.mean()),
                det_s1=float(s1.auc_det.mean()), det_ver=float(ve.auc_det.mean()),
                cov_t=float(ve.cov_target.mean()), cov_d=float(ve.cov_decoy.mean()),
                sd=float(ve.auc.std(ddof=1)) if len(ve) > 1 else 0.0,
                sd_gain=float(gain_paired.std(ddof=1)) if len(gain_paired) > 1 else 0.0,
                sd_det=float(ve.auc_det.std(ddof=1)) if len(ve) > 1 else 0.0,
                sd_det_s1=float(s1.auc_det.std(ddof=1)) if len(s1) > 1 else 0.0)
            print(f"{snr:>+5d}dB{c['auc_s1']:>9.3f}{c['auc_ver']:>9.3f}{c['gain']:>+8.3f}"
                  f"{c['sd']:>7.3f}{c['det_s1']:>11.3f}{c['det_ver']:>12.3f}"
                  f"{c['cov_t']:>9.0%}{c['cov_d']:>9.0%}")

        COV_MIN = 0.75    # below this, raw AUC is contaminated by detection failure (see caveat)
        KNIFE   = 0.25    # a margin under 25% of the bound is a coin flip, not a finding

        def trend(label, keys, get, sd_key, note, flat_note):
            """One trend test, reporting the MARGIN explicitly.

            A bare `drop > bound` turns a 0.001 difference into a printed 'CONFIRMED' -- which
            is how this project has produced retractions before. The margin is printed every
            time, and the verdict is THREE-WAY, not two:

              margin >  +KNIFE*bound : |change| clears the noise -> a real trend
              margin <  -KNIFE*bound : |change| sits far BELOW the noise -> confidently FLAT
              otherwise              : |change| ~= the noise -> genuinely unresolved

            The middle and bottom cases are opposites and must not be merged. An earlier
            version of this function tested only `margin < KNIFE*bound` and so reported a
            change of exactly 0.000 against a 0.045 bound -- the strongest possible evidence
            of flatness -- as "too close to call", burying the actual result (23 Aug 2026).
            """
            if len(keys) < 2:
                print(f"\n{label}\n  -> NO VERDICT: {len(keys)} level(s) is a point, not a trend.")
                return
            v_hi, v_lo = get(curve[keys[0]]), get(curve[keys[-1]])
            if not (v_hi == v_hi and v_lo == v_lo):          # nan guard
                print(f"\n{label}\n  -> NO VERDICT: not measurable at an endpoint (nan).")
                return
            drop   = v_hi - v_lo
            bound  = max(NOISE_FLOOR, max(curve[k][sd_key] for k in keys))
            margin = abs(drop) - bound
            print(f"\n{label}")
            print(f"  {v_hi:+.3f} at {keys[0]:+d} dB -> {v_lo:+.3f} at {keys[-1]:+d} dB    "
                  f"change {drop:+.3f}   bound {bound:.3f}   margin {margin:+.4f}")
            if margin > KNIFE * bound:
                if drop > 0:
                    print(f"  -> SHRINKS by {drop:.3f}, clear of the noise bound. {note}")
                else:
                    print(f"  -> GROWS by {-drop:.3f} as calls get fainter. Report as observed; do"
                          f"\n     not rationalise it into the write-up without a separate test.")
            elif margin < -KNIFE * bound:
                print(f"  -> FLAT within noise. |change| ({abs(drop):.3f}) sits well below the "
                      f"noise bound\n     ({bound:.3f}) -- this is positive evidence of no trend, "
                      f"not an absent result. {flat_note}")
            else:
                print(f"  -> TOO CLOSE TO CALL. |change| and the noise bound differ by "
                      f"{margin:+.4f}, within\n     {KNIFE:.0%} of the bound itself -- a coin flip. "
                      f"Unresolved: widen SNR_SWEEP_SEEDS\n     before claiming flat OR a direction.")

        allk   = sorted(curve, reverse=True)
        validk = sorted([k for k, c in curve.items() if c["cov_t"] >= COV_MIN], reverse=True)

        trend("[1] verification GAIN, raw AUC, full SNR range", allk,
              lambda c: c["gain"], "sd_gain",
              "Quote the centrepiece as a high-SNR result from here on.",
              "The verification gain holds across the whole tested SNR range.")
        if validk != allk:
            trend(f"[2] verification GAIN, raw AUC, coverage>={COV_MIN:.0%} levels only", validk,
                  lambda c: c["gain"], "sd_gain",
                  "Holds where detection is still intact -- a real specificity effect.",
                  "Where detection is intact, the gain does not depend on SNR.")
        trend("[3] verified AUC among LOCALISED items only (detection-corrected)", allk,
              lambda c: c["det_ver"], "sd_det",
              "Specificity itself degrades, independently of detection.",
              "Species discrimination is SNR-INVARIANT once a call is localised -- so any fall"
              "\n     in raw AUC is detection loss, not species confusion. This is the strong result.")
        trend("[4] stage-1 AUC among LOCALISED items only (control)", allk,
              lambda c: c["det_s1"], "sd_det_s1",
              "Stage 1's own specificity degrades too.",
              "Stage 1's own specificity is stable too, so [3] is not an artefact of the verifier.")

        print(f"\n{'-'*80}\nHOW TO READ [1] AND [3] TOGETHER -- they answer different questions")
        print("  raw falls + detection-corrected FLAT  -> DETECTION degrades with SNR; species")
        print("                                           discrimination, given a detection, does not.")
        print("  raw falls + detection-corrected FALLS -> specificity genuinely degrades; the")
        print("                                           limitation is real and belongs in the paper.")
        print("  both flat                             -> robust across the whole tested range.")
        print("  [4] is the control: if stage-1's corrected curve moves while the verified one does")
        print("  not, the verifier is what stabilises it -- and that is the claim worth making.")

        lo = min(curve)
        if curve[lo]["cov_t"] < COV_MIN:
            bad = [k for k in allk if curve[k]["cov_t"] < COV_MIN]
            print(f"\n  CAVEAT -- raw AUC at {bad} dB is CONTAMINATED. Target coverage there is "
                  f"{curve[lo]['cov_t']:.0%};\n  score_spec scores an unlocalised item 0.0 and _auc "
                  f"counts 0.0-vs-0.0 as a tie, so those levels\n  measure detection failure as much "
                  f"as species confusion. Worse, coverage is ASYMMETRIC\n  (target "
                  f"{curve[lo]['cov_t']:.0%} vs decoy {curve[lo]['cov_d']:.0%}): unmatched zeros bias "
                  f"the AUC in a fixed\n  direction rather than merely adding noise. Use the AUCdet "
                  f"columns at those levels, or\n  restrict the claim to the coverage>={COV_MIN:.0%} "
                  f"range and say so explicitly.")
    print(f"[S3 took {(time.time()-t0)/60:.1f} min]")

    # ---------------------------------------------------------- S4: decoy difficulty sweep
    # Roadmap 3.3. Decoys have always been the top-N_DECOYS nearest species in Perch space --
    # the hardest possible confusers, chosen on purpose. That choice was never actually
    # measured against the alternative: how much of the headline number is because the decoys
    # are hard? This turns one number into a difficulty curve, and pre-empts "you picked easy
    # decoys" -- you didn't, but until this ran that was an assertion, not a result.
    #
    # PRE-REGISTERED, and split into what's actually at risk of being wrong:
    #   Not a real prediction (true by construction of how DECOYS is ranked): AUC should rise
    #   hard -> medium -> random, for both stage-1 and verified. Decoys get progressively less
    #   similar to the target by definition, so this ordering isn't a finding, it's a sanity
    #   check on the ranking itself.
    #   The actual open question: does VERIFICATION'S GAIN (ver - s1) shrink, grow, or stay
    #   flat as decoys get easier? No strong prior either way -- reported honestly, whichever
    #   way it lands.
    DECOY_SWEEP_SPECIES = SWEEP_SPECIES
    DECOY_SWEEP_SEEDS   = SWEEP_SEEDS
    DECOY_MODES         = ("hard", "medium", "random")

    print(f"\n{'='*84}\nS4  DECOY DIFFICULTY  (roadmap 3.3)\n{'='*84}")
    t0 = time.time(); rows_s4 = []
    for mode in DECOY_MODES:
        for sp in DECOY_SWEEP_SPECIES:
            r = run_one(sp, N_SHOT, SHOT_STRATEGY, seeds=DECOY_SWEEP_SEEDS, decoy_mode=mode)
            if r: rows_s4.append(r); print(f"  {mode:<7} {sp:<12} decoy-sim {r['decoy_sim_top1']:.3f}  "
                                           f"AUC {r['auc_s1']:.3f} -> {r['auc_ver']:.3f}   "
                                           f"F1 {r['f1_s1']:.3f} -> {r['f1_ver']:.3f}")
    s4 = pd.DataFrame(rows_s4)

    if len(s4):
        s4.to_csv("/kaggle/working/argus_sweep_decoy_difficulty.csv", index=False)
        print(f"\n{'mode':<9}{'decoy-sim':>11}{'AUC stage-1':>13}{'AUC verified':>14}{'gain':>8}{'gain sd':>9}")
        print("-"*64)
        agg = {}
        for mode, g in s4.groupby("decoy_mode", sort=False):
            gains = g.auc_ver - g.auc_s1
            agg[mode] = dict(sim=g.decoy_sim_top1.mean(), s1=g.auc_s1.mean(), ver=g.auc_ver.mean(),
                             gain=gains.mean(), gain_sd=gains.std(ddof=1) if len(gains) > 1 else 0.0)
            c = agg[mode]
            print(f"{mode:<9}{c['sim']:>11.3f}{c['s1']:>13.3f}{c['ver']:>14.3f}{c['gain']:>+8.3f}"
                  f"{c['gain_sd']:>9.3f}")

        ordered = [agg[m] for m in DECOY_MODES if m in agg]
        print(f"\nsanity checks -- judged on AUC, which is what 'harder' actually MEANS:")
        # AUC ordering is the real difficulty check. If the tiers are separating at all, a
        # harder decoy set must score LOWER, for stage-1 and verified alike.
        auc_ok_s1  = all(ordered[i]["s1"]  <= ordered[i+1]["s1"]  + 1e-9 for i in range(len(ordered)-1))
        auc_ok_ver = all(ordered[i]["ver"] <= ordered[i+1]["ver"] + 1e-9 for i in range(len(ordered)-1))
        print(f"  AUC rises hard -> medium -> random, stage-1:  "
              f"{'OK' if auc_ok_s1 else 'VIOLATED -- tiers are not separating, investigate'}")
        print(f"  AUC rises hard -> medium -> random, verified: "
              f"{'OK' if auc_ok_ver else 'VIOLATED -- tiers are not separating, investigate'}")
        # The one similarity relation that IS true by construction: 'hard' takes ranked[0],
        # the literal maximum, so nothing can beat it on top-1.
        if "hard" in agg:
            hard_is_max = all(agg["hard"]["sim"] >= agg[m]["sim"] - 1e-9 for m in agg)
            print(f"  'hard' holds the highest top-1 similarity (true by construction):  "
                  f"{'OK' if hard_is_max else 'VIOLATED -- decoy RANKING itself is broken'}")
        sim_str = "   ".join(f"{m}: sim {agg[m]['sim']:.3f}" for m in DECOY_MODES if m in agg)
        print(f"\n  {sim_str}")
        print("  NOTE: top-1 similarity does NOT fall hard > medium > random, and that is")
        print("  EXPECTED rather than a fault. 'medium' deliberately excludes ranks 1..N_DECOYS,")
        print("  so its top-1 is a deterministic ceiling at ranked[N_DECOYS]. 'random' samples the")
        print("  WHOLE pool -- with 30 candidates and 4 draws it catches one of the four hardest")
        print("  ~45% of the time -- and top-1 is a MAX over that draw, so it lands ABOVE medium's")
        print("  ceiling. 'random' is therefore average-difficulty-with-high-variance, not an easy")
        print("  tier. Read difficulty off the AUC columns, never off this scalar: the same lesson")
        print("  the decoy-pool confound already taught (Day 28) -- one similarity number does not")
        print("  capture a decoy SET's difficulty.")

        if "hard" in agg and "random" in agg:
            drop = agg["random"]["gain"] - agg["hard"]["gain"]
            bound = max(NOISE_FLOOR, agg["hard"]["gain_sd"], agg["random"]["gain_sd"])
            margin = abs(drop) - bound
            KNIFE = 0.25
            print(f"\nTHE ACTUAL QUESTION -- does verification's gain depend on decoy difficulty?")
            print(f"  gain on hardest decoys   {agg['hard']['gain']:+.3f}")
            print(f"  gain on random decoys    {agg['random']['gain']:+.3f}")
            print(f"  change {drop:+.3f}   bound {bound:.3f}   margin {margin:+.4f}")
            if margin > KNIFE * bound:
                if drop > 0:
                    print(f"  -> GAIN IS LARGER against easy decoys, clear of noise. Verification's")
                    print(f"     advantage is partly a property of how hard the decoys are, not just")
                    print(f"     the target species -- the 0.751-class headline number is a HARD-DECOY")
                    print(f"     number and should keep being described as one.")
                else:
                    print(f"  -> GAIN IS LARGER against the hardest decoys, clear of noise. Verification")
                    print(f"     earns its keep precisely where it's needed most -- the headline number")
                    print(f"     is not being flattered by an easy comparison; if anything it's the")
                    print(f"     hardest case, understating what an easier deployment would see.")
            elif margin < -KNIFE * bound:
                print(f"  -> FLAT within noise. Verification's advantage does not depend on how hard")
                print(f"     the decoys are -- the 0.751-class headline number generalises across the")
                print(f"     difficulty curve, not just at the one operating point it was measured at.")
            else:
                print(f"  -> TOO CLOSE TO CALL, within {KNIFE:.0%} of the bound. Unresolved at this")
                print(f"     seed count -- widen DECOY_SWEEP_SEEDS before claiming a direction.")

        print(f"\nWhy this matters beyond the sanity check: the headline 0.751-class AUC has always")
        print(f"been reported on the HARDEST decoy set on purpose -- the number a skeptical judge")
        print(f"would ask for. The random-decoy row above is the number if we'd taken the easy path")
        print(f"instead, so the difference between them is now a stated fact, not an assertion.")
    print(f"[S4 took {(time.time()-t0)/60:.1f} min]")

else:
    print(f"\n{'='*84}")
    print("SKIP_SWEEPS=True -- skipping N1/S2/S3/S4 (already answered twice, "
          "16 Aug and 25 Aug; see the lab notebook Day 34 for the run-4 cross-check).")
    print(f"{'='*84}")
# ------------------------------------------------------------------ N7: Perch version
# Two-run protocol: a kernel holds ONE Perch model, so this cannot be a re-score.
# DO THIS BEFORE THE FULL MULTI-SPECIES RUN -- decoys are ranked by Perch embedding
# proximity, so changing Perch changes the decoy sets and makes runs incomparable.
print(f"\n{'='*84}\nN7  PERCH 1.0 vs 2.0 -- manual two-run protocol, DO THIS FIRST\n{'='*84}")
print(f"""Currently loaded: Perch {PERCH_VERSION}, {EMBED_DIM}-d embeddings, peak-norm {PERCH_PEAK_NORM}

To compare:
  1. Run the whole notebook with PERCH_VERSION = "v2" (default), SMOKE_TEST = True
     -> save argus_multispecies_results.csv as ..._perchv2.csv
  2. Set PERCH_VERSION = "v1" in CELL 1 and attach the v1 Kaggle model variation
     (bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier, version 8)
  3. Restart the kernel and re-run -> save as ..._perchv1.csv
  4. Compare verified AUC per species.

Published expectation (Perch 2.0 paper, Table 3): BirdSet AUROC 0.839 -> 0.908 for the
same embeddings, no fine-tuning. If ARGUS sees no gain, that is worth reporting -- it
would suggest the bottleneck is stage 1, not the embedding.

NOTE the confound: peak normalisation was ALSO wrong in v1 (raw waveforms were fed to
Perch; the official wrapper DC-removes and scales to peak 0.25). To attribute a change to
the model rather than the preprocessing, run v1 twice -- PERCH_PEAK_NORM = None and 0.25 --
or accept that the comparison bundles both fixes and say so explicitly.""")

# ------------------------------------------------------------------ N2: front-end
# NO LONGER A MANUAL PROTOCOL. CELL 2 now trains one encoder per front-end in a single
# pass over the audio (build_banks), so PCEN vs log-Mel is a proper factor in the
# ablation ladder (CELL 4) rather than a two-run comparison across kernels.
print(f"\n{'='*84}\nN2  FRONT-END (PCEN vs log-Mel) -- now handled by CELL 4\n{'='*84}")
print("""Run arguswala_ladder.py. It appears as:
  - cumulative "rung 4  +log-Mel", or
  - a main effect in the 2^4 design with FULL_FACTORIAL = True (16 configs).

The ladder resolves encoders[(FEATURE, seed)] per config, so the encoder always matches
the front-end -- evaluating a PCEN-trained encoder on log-Mel inputs would measure a
mismatch, not the front-end.

Expected, and genuinely contested:
  - DCASE 2024 official table, same team and architecture: log-Mel 65.2% vs PCEN 61.5%
  - Liang et al. Table III: log-Mel 63.67 vs PCEN 59.97 F1 -- but PCEN wins precision
    (68.0 vs 66.4), and species specificity is a precision-shaped problem
  - DCASE 2023 winner (Zou et al.) used PCEN at exactly ARGUS's 22050/1024/256 settings
Two of three point to log-Mel on F1; the 2023 winner and the precision column point back
at PCEN. This is why it is measured rather than assumed.""")

print(f"\n{'='*84}\nALL SWEEPS DONE. Copy the three CSVs out of /kaggle/working/ and log the"
      f"\nheadline numbers in the bound logbook the same day.\n{'='*84}")
