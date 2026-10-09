# ===== CELL 7 / BOTTLENECK FRONTIER + WILD-AUDIO SCAN =====
# Run AFTER arguswala_v2.py and arguswala_distill.py (reuses CELL 6's build_distill_set and
# make_distill_embedder, which are defined even when SKIP_DISTILL=True). Independent of
# CELLS 3-5.
#
# Two experiments, chosen because each attacks a limitation the research paper currently
# states outright, and because neither can be answered by more of the same compute.
#
# ---------------------------------------------------------------------------------------
# PART A -- THE BOTTLENECK FRONTIER. Why the distilled verifier failed, and whether it has
# to.
#
# CELL 6 distilled Perch into the stage-1 network and lost 0.136 mean species AUC, with
# S. albiventris collapsing to 0.511 (chance). That was reported honestly as bounding "a
# modest first attempt", not distillation in general. This tests the most specific
# mechanical explanation available for why it failed:
#
#   Perch emits 1536 dimensions. The student's trunk emits 128. A linear head from 128-d
#   can only ever reproduce a 128-dimensional SUBSPACE of Perch's space, no matter how
#   long it trains or how much data it sees. If Perch's species-discriminative structure
#   needs more than 128 dimensions, the student is not undertrained -- it is architecturally
#   incapable, and CELL 6's held-out-species cosine of 0.399 is a ceiling, not a symptom.
#
# So: sweep the bottleneck width and watch what happens to distillation fidelity AND to
# downstream species AUC. This produces an accuracy-vs-size curve rather than a single
# pass/fail, which is what the paper's own framing ("an accuracy/deployability frontier")
# actually promised but did not yet deliver.
#
# PRE-REGISTERED, before running. If the bottleneck is the binding constraint, held-out
# cosine and species AUC both rise monotonically with width, and the curve says how much
# model a field deployment has to carry to keep how much discrimination -- a genuinely
# useful number nobody has published. If both stay flat, the bottleneck is NOT the
# constraint, CELL 6's failure is about data or optimisation instead, and that redirects
# the future work in the paper. A third outcome is possible and must not be hidden: fidelity
# (cosine) rises while species AUC does not, which would mean the student is reproducing
# Perch's embedding geometry in aggregate while losing exactly the part that separates
# confusable species -- the most interesting outcome of the three, and the one most relevant
# to this project's own thesis about metrics measuring the wrong thing.
#
# A nonlinear-head arm at the widest setting separates "not enough dimensions" from "not
# enough expressiveness in the map". Same trunk, same data, same everything else.
#
# ---------------------------------------------------------------------------------------
# PART B -- WILD-AUDIO SCAN. The pipeline has never been asked to find a real call.
#
# Limitation #2 in the paper: every reported number comes from target calls PLANTED into
# real soundscape at controlled SNR. The system has never once been pointed at continuous
# unlabelled forest audio and asked what is in it. That is the gap between a benchmark and
# an instrument.
#
# This scans soundscapes the pipeline has never touched -- strictly disjoint from the 14
# used for calibration and evaluation -- at the same best-F1 threshold every other number
# in this project uses, and saves the highest-scoring detections as listenable clips.
#
# WHAT THIS CAN AND CANNOT ESTABLISH, stated before it runs. It cannot confirm a detection:
# nobody in this project can reliably identify S. albiventris by ear, and there is no ground
# truth for these files. It CAN establish a detection RATE on genuinely wild audio, produce
# a ranked shortlist short enough for a human (or later, an expert) to adjudicate, and
# compare that rate against the same pipeline hunting a different species in the identical
# audio. If the study species and the control fire at indistinguishable rates, that is
# evidence the detector is responding to generic bird-shaped sound rather than to this
# species, and it must be reported that way.
# ---------------------------------------------------------------------------------------
# RUN HISTORY. The 3 Sep session died 15 s into the first distilled arm with a CUDA OOM,
# after spending 57 min on setup, the teacher set and the Perch control. Cause: the
# held-out set was pushed through the trunk in ONE forward pass (inherited from CELL 6,
# where it was safe at 2,636 pairs and ~400 held-out crops). At this cell's 10,163 pairs
# that is ~1,633 crops, and the first conv block emits 128 channels at the full 128x86
# spectrogram resolution -- 5.6 MB of activations per crop, so its BatchNorm asked for
# 8.69 GiB in a single allocation. Fixed in train_width_student below.
#
# The wall-clock estimate was wrong too, in a way worth recording: it costed each arm's
# EVALUATION and forgot the training entirely. The session did produce one real
# measurement before dying -- the Perch control takes 34.6 min, the teacher set 12.7 min,
# and one training epoch ~11 s -- so this cell no longer trusts an estimate at all. It
# measures as it goes, refuses to start an arm it cannot finish, and stops the scan while
# there is still time to write files.
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
import time, os, gc, glob

SKIP_FRONTIER = False

# ------------------------------------------------------------------ Part A config
FRONTIER_WIDTHS      = (64, 128, 256, 512)
FRONTIER_NONLINEAR   = True          # extra arm: MLP head at the widest width
FRONTIER_EVAL_SPECIES = TARGET_LIST[:5]
FRONTIER_EVAL_SEEDS   = (7, 8)
RUN_PERCH_CONTROL     = True         # same-session Perch baseline, not a carried-over number

# Distillation set, scaled up from CELL 6's 2,636 pairs / 60 species. Species diversity is
# raised in preference to crops-per-recording: CELL 6 hit train cosine 0.669 against
# held-out-species 0.399, and a train/held-out gap that wide is a generalisation problem
# across species, which more crops of the SAME species does not fix.
FRONTIER_SPECIES_N   = 98            # effectively every non-target species available
FRONTIER_RECS_PER_SP = 20
FRONTIER_CROPS_PER_REC = 6
FRONTIER_EPOCHS      = 90            # 90 x 8,530 pairs = 5.7x CELL 6's total gradient
                                     # steps, so a null result cannot be blamed on
                                     # undertraining -- which matters, because "both flat"
                                     # is one of the pre-registered outcomes.
FRONTIER_BATCH       = 128
FRONTIER_LR          = 1e-3
FRONTIER_VAL_FRAC    = 0.15          # held out BY SPECIES, as in CELL 6
FRONTIER_SEED        = 0
FRONTIER_CKPT        = "/kaggle/working/argus_frontier.csv"

# ------------------------------------------------------------------ Part B config
SCAN_ENABLED     = True
SCAN_SPECIES     = PRIMARY_TARGET            # whbsho3 -- the species the project is about
SCAN_CONTROL     = "bkwsti"                  # a target species with a very different niche
SCAN_N_FILES     = 400                       # 400 x 240 s = ~27 h of unlabelled audio
SCAN_SKIP_FIRST  = N_CALIB_BEDS + N_EVAL_BEDS   # NEVER rescan calibration/eval beds
SCAN_TOP_K       = 60                        # clips saved for listening, per species
SCAN_SEED        = 7
SCAN_CKPT        = "/kaggle/working/argus_wildscan.csv"
SCAN_CLIPS       = "/kaggle/working/argus_wildscan_clips.npz"

# ------------------------------------------------------------------ session budget
# Measured, not estimated: setup through CELL 6 = 10.2 min, teacher set = 12.7 min,
# Perch control = 34.6 min, one training epoch ~11 s. Everything else is enforced rather
# than predicted. PART A stops launching arms once PART B's reserve is threatened, and
# PART B stops mid-scan with its CSV and clips written rather than being killed at the
# 12 h ceiling holding everything in memory.
CELL7_BUDGET_H   = 10.0    # hard ceiling for CELL 7's own wall clock (~10 min of setup precedes it)
SCAN_RESERVE_MIN = 180     # never start another frontier arm if less than this remains for PART B

assert EMBED_BACKEND == "perch", \
    f"CELL 7 must start from the Perch backend, found {EMBED_BACKEND!r}"

BASE_F, CAND_F = "stage-1 only  (p)", "perch only    (v)"
ENC_F = encoders[(FEATURE, ENCODER_SEEDS[0])]   # stage 1 is IDENTICAL in every arm


# ------------------------------------------------------------------ Part A machinery
class WidthStudent(nn.Module):
    """CELL 6's DistillStudent with the bottleneck width exposed, and an optional
    nonlinear head.

    The trunk is Enc(emb=width) -- the same four-conv-block architecture stage 1 deploys,
    just with the final block emitting `width` channels instead of the fixed 128. Nothing
    else about the network changes, so a difference between arms is attributable to the
    bottleneck and not to a different model having been swapped in.

    `nonlinear=True` inserts one hidden layer of the same width before the projection. It
    exists to separate two explanations that a width sweep alone cannot: "128 dimensions
    cannot carry Perch's structure" versus "a single linear map cannot carry it, whatever
    the width". Those imply different fixes, so they are worth telling apart."""
    def __init__(s, width, out_dim, nonlinear=False):
        super().__init__()
        s.width = width
        s.trunk = Enc(emb=width)
        if nonlinear:
            s.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, out_dim))
            # CELL 6's make_distill_embedder() reads `model.head.out_features` to size the
            # empty-input return. nn.Linear has that attribute; nn.Sequential does not, so
            # the nonlinear arm would crash the first time stage 1 proposed zero candidates
            # for a recording -- which happens routinely, and would happen hours into the
            # session rather than at the start. Set it explicitly so both arms satisfy the
            # same contract.
            s.head.out_features = out_dim
        else:
            s.head = nn.Linear(width, out_dim)
        s.out_dim = out_dim
    def forward(s, x): return s.head(s.trunk(x))
    def embed(s, x):   return s.head(s.trunk(x))
    def n_params(s):   return sum(p.numel() for p in s.parameters())


def train_width_student(specs, embs, species, width, nonlinear=False,
                        epochs=FRONTIER_EPOCHS, batch=FRONTIER_BATCH, lr=FRONTIER_LR,
                        val_frac=FRONTIER_VAL_FRAC, seed=FRONTIER_SEED):
    """Cosine-loss distillation at a given bottleneck width, validated on HELD-OUT SPECIES.

    Identical protocol to CELL 6's train_distill (same loss, same species-level split, same
    best-epoch selection on held-out species) so the width arms are comparable to CELL 6's
    published 128-d linear result as well as to each other. The split is regenerated from
    the same seed for every arm, so all arms see the same train/held-out species partition
    -- otherwise a lucky partition would masquerade as a width effect."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(species); rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_sp = set(uniq[:n_val].tolist())
    vm = np.array([s in val_sp for s in species])

    print(f"  split: {(~vm).sum()} train pairs / {vm.sum()} held-out pairs "
          f"({len(uniq)-n_val} train species / {n_val} held-out species)")

    Xtr = torch.tensor(specs[~vm]); Ytr = torch.tensor(embs[~vm])
    # The held-out tensors stay on the CPU and are streamed in batches. This is the line
    # that killed the 3 Sep session: `m(Xva)` ran all ~1,633 held-out crops through the
    # trunk at once, and the first conv block's BatchNorm asked CUDA for 8.69 GiB. Batched
    # evaluation is numerically identical here -- m.eval() puts BatchNorm on its running
    # statistics, so the result cannot depend on how the held-out set is grouped -- which
    # is why this is a pure bug fix and leaves CELL 6's archived 0.399 comparable.
    Xva = torch.tensor(specs[vm]); Yva = torch.tensor(embs[vm])

    m = WidthStudent(width, embs.shape[1], nonlinear=nonlinear).to(device)

    def _val_cos():
        m.eval(); tot = 0.0
        with torch.no_grad():
            for i in range(0, len(Xva), batch):
                xb = Xva[i:i + batch].to(device); yb = Yva[i:i + batch].to(device)
                tot += float(nn.functional.cosine_similarity(m(xb), yb, dim=1).sum())
        return tot / len(Xva)

    opt = optim.Adam(m.parameters(), lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(Xtr); best_val = -1.0; best_state = None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xtr[idx].to(device); yb = Ytr[idx].to(device)
            loss = (1 - nn.functional.cosine_similarity(m(xb), yb, dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        vsim = _val_cos()
        if vsim > best_val:
            best_val = vsim
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if (ep + 1) % 30 == 0 or ep == 0:
            print(f"      ep {ep+1:3d}/{epochs}  train cos {1-tot/n:.4f}  held-out cos {vsim:.4f}")
    m.load_state_dict(best_state); m.eval()
    return m, best_val


def _eval_backend(label, rows, extra=None):
    """Run the standard specificity test for whatever EMBED_BACKEND is currently active.

    _TARGET_CACHE is cleared first, and this is load-bearing rather than defensive: the
    cache key includes EMBED_BACKEND but NOT which student is bound to it, so running a
    64-d arm and then a 256-d arm would otherwise hand the second arm the FIRST arm's
    cached verifier matrix and report it as the second's. That failure is silent and
    produces entirely plausible numbers, which is what makes it worth an explicit line."""
    _TARGET_CACHE.clear()
    for sp in FRONTIER_EVAL_SPECIES:
        try:
            ctx = build_target(sp, verbose=False)
        except Exception as ex:
            print(f"    skip {sp} ({label}): {type(ex).__name__}: {ex}"); continue
        per = []
        for sd in FRONTIER_EVAL_SEEDS:
            ec = eval_pass(ENC_F, ctx, sd)
            sc = spec_pass(ENC_F, ctx, sd)
            nc = neg_pass(ENC_F, ctx, sd)
            m = score_eval(FUSIONS[CAND_F], ec)
            _, _, _, _, auc = score_spec(FUSIONS[CAND_F], m["thr"], sc, ctx["DECOYS"])
            per.append(dict(auc=auc, f1=m["f1"], prec=m["p"], rec=m["r"],
                            fa_per_hr=score_neg(FUSIONS[CAND_F], m["thr"], nc)))
        mean = {k: float(np.mean([d[k] for d in per])) for k in per[0]}
        sd_ = {k: float(np.std([d[k] for d in per], ddof=1)) if len(per) > 1 else 0.0
               for k in per[0]}
        row = dict(arm=label, species=sp, n_seeds=len(FRONTIER_EVAL_SEEDS),
                   auc=mean["auc"], auc_sd=sd_["auc"], f1=mean["f1"], prec=mean["prec"],
                   rec=mean["rec"], fa_per_hr=mean["fa_per_hr"])
        row.update(extra or {})
        rows.append(row)
        pd.DataFrame(rows).to_csv(FRONTIER_CKPT, index=False)
        print(f"    [{label:16s} {sp:9s}] AUC {mean['auc']:.3f}  F1 {mean['f1']:.3f}  "
              f"FA/hr {mean['fa_per_hr']:.1f}")
        gc.collect()
        try: torch.cuda.empty_cache()
        except Exception: pass


# ------------------------------------------------------------------ run
frontier_rows = []
if SKIP_FRONTIER:
    print("SKIP_FRONTIER=True -- skipping CELL 7 entirely.")
else:
    print(f"\n{'='*90}\nCELL 7  PART A -- BOTTLENECK FRONTIER\n{'='*90}")
    t_all = time.time()
    _deadline = t_all + CELL7_BUDGET_H * 3600
    def _left_min(): return (_deadline - time.time()) / 60.0

    # ---- teacher set, built ONCE and shared by every arm ----
    t0 = time.time()
    specs, embs, species = build_distill_set(
        TARGET_SET, n_species=FRONTIER_SPECIES_N, recs_per_sp=FRONTIER_RECS_PER_SP,
        crops_per_rec=FRONTIER_CROPS_PER_REC, seed=FRONTIER_SEED)
    print(f"[teacher set: {len(specs)} pairs, {len(np.unique(species))} species, "
          f"{(time.time()-t0)/60:.1f} min]")

    # ---- Perch control, same session, same species, same seeds ----
    if RUN_PERCH_CONTROL:
        print(f"\n{'-'*90}\nPerch control (EMBED_BACKEND='perch')\n{'-'*90}")
        t0 = time.time()
        _eval_backend("perch", frontier_rows,
                      extra=dict(width=np.nan, nonlinear=False, params=np.nan, val_cosine=np.nan))
        print(f"[perch control took {(time.time()-t0)/60:.1f} min -- every distilled arm "
              f"below costs about the same]")

    # ---- one arm per bottleneck width ----
    arms = [(w, False) for w in FRONTIER_WIDTHS]
    if FRONTIER_NONLINEAR:
        arms.append((max(FRONTIER_WIDTHS), True))

    # Run order is by information value, not by width. If the budget guard stops PART A
    # early, the arms that survive should be the ones that answer the question: 128
    # replicates CELL 6's published setting and 512 is the widest bottleneck, so those two
    # alone already say whether width moves anything at all. 64 and 256 fill in the shape
    # of the curve, and the nonlinear arm is a follow-up that only matters if width does
    # move something. The report below sorts by width regardless of run order.
    def _priority(a):
        w, nl = a
        if nl:                        return 9.0
        if w == 128:                  return 0.0
        if w == max(FRONTIER_WIDTHS): return 1.0
        return 2.0 + w / 10000.0
    arms.sort(key=_priority)
    print("arm order (by information value, not by width): " +
          ", ".join(f"{w}-mlp" if nl else str(w) for w, nl in arms))

    for _ai, (width, nonlin) in enumerate(arms):
        _left = _left_min()
        if _left < SCAN_RESERVE_MIN:
            _skipped = [f"{w}-mlp" if nl else str(w) for w, nl in arms[_ai:]]
            print()
            print(f"[budget] {_left:.0f} min left, below the {SCAN_RESERVE_MIN} min reserved "
                  f"for PART B -- stopping PART A here rather than crowding out the scan.")
            print(f"[budget] arms not run: {', '.join(_skipped)}. To finish the curve, re-run "
                  f"with FRONTIER_WIDTHS trimmed to those and RUN_PERCH_CONTROL=False.")
            break
        tag = f"distil-{width}{'-mlp' if nonlin else ''}"
        print(f"\n{'-'*90}\n{tag}  (bottleneck {width}-d"
              f"{', nonlinear head' if nonlin else ', linear head'})\n{'-'*90}")
        t0 = time.time()
        student, val_cos = train_width_student(specs, embs, species, width, nonlinear=nonlin)
        npar = student.n_params()
        print(f"    held-out-species cosine {val_cos:.4f}   params {npar:,}   "
              f"{12_000_000/npar:.1f}x smaller than Perch   [{(time.time()-t0)/60:.1f} min]")

        globals()["DISTILL_EMBED_CTX"] = make_distill_embedder(student)
        try:
            globals()["EMBED_BACKEND"] = "distilled"
            _eval_backend(tag, frontier_rows,
                          extra=dict(width=width, nonlinear=nonlin, params=npar,
                                     val_cosine=val_cos))
        finally:
            globals()["EMBED_BACKEND"] = "perch"
            globals()["DISTILL_EMBED_CTX"] = None
            _TARGET_CACHE.clear()
        print(f"  [{tag} total {(time.time()-t0)/60:.1f} min, "
              f"{(time.time()-t_all)/60:.1f} min elapsed in CELL 7, "
              f"{_left_min():.0f} min of budget left]")

    # ---- frontier report ----
    fr = pd.DataFrame(frontier_rows)
    if len(fr):
        print()
        print("=" * 90); print("FRONTIER"); print("=" * 90)
        g = fr.groupby("arm", sort=False).agg(
            auc=("auc", "mean"), fa=("fa_per_hr", "mean"), width=("width", "first"),
            params=("params", "first"), cos=("val_cosine", "first"))
        # Display and analysis both sort by width. Arms are RUN in priority order so that
        # a budget-truncated session still answers the question, which means run order and
        # width order differ -- reading iloc[0]/iloc[-1] off run order would silently
        # compare the wrong two arms.
        g = g.sort_values("width", na_position="first")
        print(f"{'arm':<18}{'species AUC':>12}{'FA/hr':>9}{'params':>12}{'cos(teacher)':>14}")
        for arm, r in g.iterrows():
            p = "--" if np.isnan(r["params"]) else f"{int(r['params']):,}"
            c = "--" if np.isnan(r["cos"]) else f"{r['cos']:.4f}"
            print(f"{arm:<18}{r['auc']:>12.4f}{r['fa']:>9.1f}{p:>12}{c:>14}")
        d = g[g.index != "perch"]
        if "perch" in g.index and len(d):
            base = float(g.loc["perch", "auc"])
            best = d.auc.idxmax()
            print()
            print(f"Perch {base:.4f}  |  best distilled '{best}' {d.auc.max():.4f}  "
                  f"(gap {d.auc.max()-base:+.4f})")
            lin = d[~d.index.str.endswith("-mlp")].sort_values("width")
            if len(lin) >= 2:
                w0, w1 = int(lin.width.iloc[0]), int(lin.width.iloc[-1])
                rise = lin.auc.iloc[-1] - lin.auc.iloc[0]
                crise = lin.cos.iloc[-1] - lin.cos.iloc[0]
                print(f"linear arms {w0}-d -> {w1}-d ({len(lin)} widths): "
                      f"AUC {lin.auc.iloc[0]:.4f} -> {lin.auc.iloc[-1]:.4f} ({rise:+.4f}), "
                      f"teacher cosine {lin.cos.iloc[0]:.4f} -> {lin.cos.iloc[-1]:.4f} ({crise:+.4f})")
                print("Read this against the pre-registered outcomes at the top of this cell:")
                print("  both rise      -> the bottleneck WAS the constraint; this is the frontier curve.")
                print("  both flat      -> it was not; CELL 6's failure is data or optimisation, not width.")
                print("  cosine rises,  -> the student reproduces Perch's geometry but loses exactly the")
                print("  AUC does not      part that separates confusable species. Report this one loudly.")
                mlp = d[d.index.str.endswith("-mlp")]
                if len(mlp):
                    same = lin[lin.width == mlp.width.iloc[0]]
                    if len(same):
                        print(f"nonlinear head at {int(mlp.width.iloc[0])}-d: "
                              f"AUC {float(same.auc.iloc[0]):.4f} -> {float(mlp.auc.iloc[0]):.4f}, "
                              f"cosine {float(same.cos.iloc[0]):.4f} -> {float(mlp.cos.iloc[0]):.4f}. "
                              f"A gain here means the LINEAR MAP was the limit, not the dimension count.")
        print()
        print(f"wrote {FRONTIER_CKPT}  ({len(fr)} rows)")

    # =================================================================== PART B
    if SCAN_ENABLED:
        print()
        print("=" * 90); print("CELL 7  PART B -- WILD-AUDIO SCAN"); print("=" * 90)
        all_paths = sorted(glob.glob(os.path.join(SS, "*.ogg")))
        scan_paths = all_paths[SCAN_SKIP_FIRST:SCAN_SKIP_FIRST + SCAN_N_FILES]
        print(f"{len(all_paths)} soundscapes available; scanning {len(scan_paths)}, "
              f"skipping the first {SCAN_SKIP_FIRST} (calibration + evaluation beds).")
        assert EMBED_BACKEND == "perch", "PART B must run on the real Perch pipeline"

        # Both species are hunted inside ONE pass over the files rather than one full pass
        # each. The cheap reason is that each .ogg is decoded once instead of twice. The
        # real reason is that under a wall-clock budget a two-pass version can scan 400
        # files for the study species and then run out after 150 for the control, and a
        # rate comparison across two different file sets is not a comparison. Interleaving
        # makes the file sets identical by construction, however early the run stops.
        _TARGET_CACHE.clear()
        scan_sp = [SCAN_SPECIES, SCAN_CONTROL]
        ctxs, thrs, rngs = {}, {}, {}
        for sp in scan_sp:
            ctxs[sp] = build_target(sp, verbose=False)
            # Operating threshold: the same best-F1 point every other number in this
            # project uses, derived the same way, from planted audio -- NOT tuned on the
            # wild files, which have no labels to tune against.
            thrs[sp] = score_eval(FUSIONS[CAND_F], eval_pass(ENC_F, ctxs[sp], SCAN_SEED))["thr"]
            rngs[sp] = np.random.default_rng(SCAN_SEED)
            print(f"  {sp}: operating threshold (best-F1, from planted eval) {thrs[sp]:.4f}")

        hits = []; secs = 0.0; n_files = 0; t0 = time.time(); stopped_early = False
        for i, p in enumerate(scan_paths):
            if _left_min() < 8:
                stopped_early = True
                print(f"    [budget] {_left_min():.0f} min left -- stopping the scan after "
                      f"{n_files} files so the CSV and the clips still get written.")
                break
            try:
                y = ld(p, 240.0)
            except Exception as ex:
                print(f"    skip {os.path.basename(p)}: {type(ex).__name__}"); continue
            secs += len(y) / SR; n_files += 1
            for sp in scan_sp:
                for (s0, s1, pp, vv) in detect_pv(ENC_F, ctxs[sp], y, rngs[sp]):
                    if vv >= thrs[sp]:
                        hits.append(dict(species=sp, file=os.path.basename(p),
                                         start=float(s0), end=float(s1),
                                         p=float(pp), v=float(vv)))
            if i == 0:
                per = time.time() - t0
                print(f"  [timing] first file {per:.1f}s for both species -> "
                      f"~{per*len(scan_paths)/60:.0f} min projected for all {len(scan_paths)}")
            if (i + 1) % 25 == 0:
                pd.DataFrame(hits).to_csv(SCAN_CKPT, index=False)
                print(f"    {i+1}/{len(scan_paths)} files, {len(hits)} detections, "
                      f"{(time.time()-t0)/60:.1f} min elapsed, "
                      f"{_left_min():.0f} min of budget left")
        hrs = secs / 3600.0
        print(f"  scanned {n_files} files = {hrs:.1f} h of audio in "
              f"{(time.time()-t0)/60:.1f} min" + (" (STOPPED EARLY ON BUDGET)" if stopped_early else ""))

        # top-K clips per species, for a human to listen to
        clip_store = {}
        for sp in scan_sp:
            top = sorted([h for h in hits if h["species"] == sp],
                         key=lambda h: -h["v"])[:SCAN_TOP_K]
            for j, h in enumerate(top):
                try:
                    y = ld(os.path.join(SS, h["file"]), 240.0)
                    a = max(0, int((h["start"] - 0.5) * SR))
                    b = min(len(y), int((h["end"] + 0.5) * SR))
                    clip_store[f"{sp}_{j:03d}_v{h['v']:.3f}_{h['file'].replace('.ogg','')}"] = y[a:b]
                except Exception:
                    pass
        gc.collect()
        try: torch.cuda.empty_cache()
        except Exception: pass

        sc = pd.DataFrame(hits)
        if len(sc):
            sc.to_csv(SCAN_CKPT, index=False)
            np.savez_compressed(SCAN_CLIPS, **clip_store)
            print()
            print("=" * 90); print("WILD SCAN"); print("=" * 90)
            summ = sc.groupby("species").agg(detections=("v", "size"), mean_v=("v", "mean"),
                                             max_v=("v", "max"), files=("file", "nunique"))
            summ["per_hour"] = summ["detections"] / max(hrs, 1e-9)
            print(summ.to_string())
            a = sc[sc.species == SCAN_SPECIES]; b = sc[sc.species == SCAN_CONTROL]
            print()
            print(f"{SCAN_SPECIES} {len(a)} detections vs control {SCAN_CONTROL} {len(b)}, "
                  f"on the SAME {n_files} files ({hrs:.1f} h).")
            print(f"rates: {len(a)/max(hrs,1e-9):.1f}/h vs {len(b)/max(hrs,1e-9):.1f}/h  "
                  f"(ratio {len(a)/max(len(b),1):.2f}x)")
            print("Neither number is a confirmed detection -- there is no ground truth for these")
            print("files and nobody on this project can identify the species by ear. What this")
            print("supports is a RATE, a ranked shortlist, and a comparison. If the two rates are")
            print("close, say so: that is evidence the detector is firing on bird-shaped sound")
            print("rather than on this species, and it belongs in the paper either way.")
            print()
            print(f"wrote {SCAN_CKPT} ({len(sc)} rows) and {SCAN_CLIPS} "
                  f"({len(clip_store)} clips, load with np.load and listen)")
        else:
            print("no detections above threshold in either species -- that is itself a result,")
            print("and it means the operating point transferred from planted audio to wild audio")
            print("as total silence. Report it; do not lower the threshold to find something.")

    print()
    print(f"[CELL 7 total {(time.time()-t_all)/60:.1f} min]")
