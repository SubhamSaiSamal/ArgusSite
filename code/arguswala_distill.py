# ===== CELL 6 / PERCH DISTILLATION =====
# Run AFTER arguswala_v2.py (needs its `encoders`, `Enc`, `feat`/`crop`, and the three
# evaluation passes). Independent of CELLS 3-5.
#
# THE QUESTION. Every species-discrimination number this project reports comes out of
# Perch v2 -- a ~12M-parameter model Google trained on 14,795 classes. Stage 1, the part
# built here, is a ~445k-parameter trunk (642k once the projection head needed to actually
# stand in for Perch is included) trained on 40 species, and CELL 5's P1 established
# that neither stage does the other's job: Perch alone collapses to 0.526 AUC / 2.5%
# recall as a detector, stage-1 alone sits at 0.585 on discrimination. The pipeline works;
# but the DISCRIMINATION half of it is Perch's.
#
# That is also a deployment problem, not only an authorship one. Perch cannot run on a
# solar-powered recorder in a shola forest. A 642k-parameter CNN can. So:
#
#   Can Perch's species knowledge be compressed ~19x into the stage-1 encoder, and what
#   does that cost in species discrimination?
#
# METHOD. Standard knowledge distillation. The student is the SAME Enc architecture stage 1
# already uses, plus a linear head projecting its 128-d output to Perch's 1536-d space. It
# is trained to reproduce Perch's embedding of the same 1-second audio, by cosine loss.
# Perch is the TEACHER, used at training time only. At detection time under
# EMBED_BACKEND="distilled", no Perch inference runs at all.
#
# WHAT IS HELD FIXED, AND WHY IT MATTERS. Only the verifier's embedding space changes.
# Decoy ranking and shot selection stay on Perch under both arms (see EMBED_BACKEND in
# CELL 1) -- those decide WHAT THE TEST IS. Changing them alongside the thing being
# measured is the exact error the literature review flagged for the v1->v2 upgrade. Same
# species, same shots, same decoys, same beds, same seeds: only the embedding space moves.
#
# PRE-REGISTERED PREDICTION, written before the first run. The student will lose to the
# teacher -- a 19x smaller model trained on 40 species cannot carry all of a 14,795-class
# model's discriminative structure. The open question is HOW MUCH is lost, and the honest
# outcomes are: (a) small loss, inside the noise floor -> compression is nearly free and
# field deployment is real; (b) large but bounded loss -> the accuracy/deployability
# frontier gets a number nobody has published; (c) collapse to chance -> distillation does
# not transfer at this scale, reported as a negative result. All three are publishable.
# What is NOT acceptable is quietly reporting only whichever the run happens to produce.
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, time, os, gc

SKIP_DISTILL   = True
DISTILL_SPECIES_N   = 60      # bank species for the teacher-student set (targets excluded)
DISTILL_RECS_PER_SP = 12      # recordings per species
DISTILL_CROPS_PER_REC = 4     # peak windows per recording -> ~2,880 pairs at the defaults
DISTILL_EPOCHS = 60
DISTILL_BATCH  = 64
DISTILL_LR     = 1e-3
DISTILL_VAL_FRAC = 0.15       # held-out species, not held-out crops -- see below
DISTILL_SEED   = 0
DISTILL_EVAL_SPECIES = TARGET_LIST[:5]
DISTILL_EVAL_SEEDS   = (7, 8)
ENC_D  = encoders[(FEATURE, ENCODER_SEEDS[0])]   # stage 1 is IDENTICAL in both arms
CAND_D = "perch only    (v)"                     # the verifier-only fusion, both arms
DISTILL_CKPT   = "/kaggle/working/argus_distill.csv"
# Same bound CELL 5 uses and for the same reason: it is the configuration-matched
# encoder-seed spread over these species, not the pipeline's run-wide figure.
DISTILL_NOISE_FLOOR = 0.025

assert EMBED_BACKEND == "perch", \
    f"CELL 6 must start from the Perch backend, found {EMBED_BACKEND!r}"

# ------------------------------------------------------------------ teacher-student data
def _peak_offsets(y, k, n=CLEN):
    """Same selection rule as CELL 1's peak_windows(), but returns the OFFSETS rather than
    the audio. Needed because the teacher and the student must be handed the same second:
    peak_windows() returns only the waveform, so pairing it with a spectrogram crop would
    mean re-deriving where it came from. Returning offsets makes the correspondence exact
    by construction instead of by reconstruction."""
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

def build_distill_set(exclude, feature=FEATURE, n_species=DISTILL_SPECIES_N,
                      recs_per_sp=DISTILL_RECS_PER_SP, crops_per_rec=DISTILL_CROPS_PER_REC,
                      seed=DISTILL_SEED):
    """-> (specs, embs, species) : student inputs, teacher targets, and the species each
    pair came from.

    Deliberately a SEPARATE builder from CELL 2's build_banks(), not a modification of it.
    build_banks() feeds the stage-1 encoder that produced every archived result in this
    project; changing it would invalidate the centrepiece. This function reads the same
    dataset with the same exclusions and the same front-end, but keeps the raw waveform
    crop alongside the spectrogram crop, because the teacher needs audio and the student
    needs a spectrogram -- of the SAME one second, which is the whole point.

    crops_per_rec exists because ~480 pairs (40 species x 12 recordings) is far too few to
    fit a 128->1536 projection, which alone has ~197k parameters. Taking the top-k loudest
    non-overlapping windows per recording multiplies the pair count without touching a new
    file, and peak_windows() is already the project's way of finding where a bird actually
    calls in a mostly-silent recording."""
    rng = np.random.default_rng(seed)
    sps = [s for s in meta.primary_label.unique() if s not in exclude]
    rng.shuffle(sps)
    specs, waves, species = [], [], []
    picked = 0
    for sp in sps:
        if picked >= n_species: break
        got = 0
        for fn in meta[meta.primary_label == sp]["filename"].tolist()[:recs_per_sp]:
            try:
                y = ld(os.path.join(AUD, fn), 15)
                # Featurise the WHOLE recording then crop, exactly as build_banks() and
                # stage1()'s scan do. Featurising an isolated 1 s clip instead would give
                # different mel edge behaviour from what the student meets at detection
                # time -- a train/test mismatch introduced by the data builder.
                f = feat(y, feature)
                for off in _peak_offsets(y, k=crops_per_rec, n=CLEN):
                    w = y[off:off + CLEN]
                    if len(w) < CLEN: continue
                    specs.append(crop(f, off // HOP + SEG_FR // 2))
                    waves.append(w.astype("float32"))
                    species.append(sp)
                    got += 1
            except Exception:
                pass
        if got >= 8:
            picked += 1
    if not specs:
        raise RuntimeError("distillation set is empty -- check AUD/meta are loaded")
    specs = np.stack(specs)
    print(f"  distill set: {len(specs)} (spectrogram, waveform) pairs "
          f"from {picked} species")
    t0 = time.time()
    embs = []
    for i in range(0, len(waves), 256):
        embs.append(perch_embed(waves[i:i + 256]))
        if i == 0:
            print(f"  [timing] first 256 teacher embeddings took {time.time()-t0:.0f}s "
                  f"-> ~{(time.time()-t0)*len(waves)/256/60:.1f} min projected")
    embs = l2n(np.concatenate(embs))
    return specs, embs.astype("float32"), np.array(species)


class DistillStudent(nn.Module):
    """The stage-1 encoder, plus a linear projection into Perch's embedding width.

    The trunk is Enc() unchanged -- the same 4-conv-block, ~445k-parameter network stage 1
    already runs -- so a positive result here is a statement about the model this project
    actually deploys, not about some larger network introduced for the occasion. Only the
    projection head is new, and it is deliberately linear: if a single matrix can carry
    Perch's structure, that is a stronger and more interpretable claim than if an MLP is
    needed to rescue it."""
    def __init__(s, out_dim):
        super().__init__()
        s.trunk = Enc()
        s.head  = nn.Linear(EMB, out_dim)
    def forward(s, x):
        return s.head(s.trunk(x))
    def embed(s, x):
        return s.head(s.trunk(x))


def train_distill(specs, embs, species, epochs=DISTILL_EPOCHS, batch=DISTILL_BATCH,
                  lr=DISTILL_LR, val_frac=DISTILL_VAL_FRAC, seed=DISTILL_SEED):
    """Cosine-loss distillation, validated on HELD-OUT SPECIES.

    The validation split is by species, not by crop. Splitting crops at random would put
    other windows from the same recording -- often the same individual bird, the same
    microphone, the same afternoon -- on both sides, and the reported validation number
    would then measure memorisation rather than transfer. Since the entire point is
    whether the student generalises to species it has never met (which is what a five-shot
    target IS), the split has to be at the species level to mean anything.

    Cosine rather than MSE because every downstream consumer of these embeddings
    L2-normalises first (l2n in CELL 1, applied in build_target and detect_pv), so only
    direction is ever used. Training to match magnitudes would spend capacity on a
    quantity nothing reads."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(species); rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_sp = set(uniq[:n_val].tolist())
    vm = np.array([s in val_sp for s in species])
    print(f"  split: {(~vm).sum()} train pairs / {vm.sum()} val pairs "
          f"({len(uniq)-n_val} train species / {n_val} held-out species)")

    Xtr = torch.tensor(specs[~vm]); Ytr = torch.tensor(embs[~vm])
    # Held-out crops stream in batches rather than going through the trunk in one forward
    # pass. At this cell's own defaults (~400 held out) the single-pass version fits and
    # produced the archived 0.399; CELL 7 raised the teacher set to 10,163 pairs and the
    # identical line asked CUDA for 8.69 GiB and killed the 3 Sep session. Fixed in both
    # places rather than only where it bit, since the trap is the dataset size and not the
    # cell. m.eval() puts BatchNorm on running statistics, so batching cannot change the
    # number -- the archived result stays comparable.
    Xva = torch.tensor(specs[vm]); Yva = torch.tensor(embs[vm])

    m = DistillStudent(embs.shape[1]).to(device)

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
        m.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xtr[idx].to(device); yb = Ytr[idx].to(device)
            pred = m(xb)
            loss = (1 - nn.functional.cosine_similarity(pred, yb, dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        vsim = _val_cos()
        if vsim > best_val:
            best_val = vsim
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    ep {ep+1:3d}/{epochs}  train cos {1-tot/n:.4f}  "
                  f"held-out-species cos {vsim:.4f}")
    # Select on held-out species rather than taking the last epoch: the last epoch is
    # whatever the schedule happened to end on, and reporting it would confound "how well
    # this distils" with "where training stopped".
    m.load_state_dict(best_state)
    m.eval()
    print(f"  best held-out-species cosine similarity to Perch: {best_val:.4f}")
    return m, best_val


def make_distill_embedder(model, feature=FEATURE):
    """-> a function with verifier_embed_ctx's contract: (y22, centres_sec) -> (n, D).

    Mirrors perch_embed_ctx's geometry at the student's own scale. Perch cuts a 5-second
    window around each event because that is its input length; the student cuts SEG_SEC
    (1 s), because that is what it was distilled on and what stage 1 scans with. Feeding
    it a 5-second window would be a train/test mismatch, not a fairer comparison."""
    def _embed_ctx(y22, centres_sec):
        if len(centres_sec) == 0:
            return np.zeros((0, model.head.out_features), dtype="float32")
        f = feat(y22, feature)
        xs = [crop(f, int(round(c * SR / HOP))) for c in centres_sec]
        out = []
        with torch.no_grad():
            for i in range(0, len(xs), 256):
                b = torch.tensor(np.stack(xs[i:i + 256])).to(device)
                out.append(model.embed(b).cpu().numpy())
        return np.concatenate(out).astype("float32")
    return _embed_ctx


# ------------------------------------------------------------------ run
rows = []
if SKIP_DISTILL:
    print("SKIP_DISTILL=True -- skipping CELL 6 entirely.")
else:
    print(f"\n{'='*90}\nCELL 6  PERCH DISTILLATION\n{'='*90}")
    print(f"teacher=Perch {PERCH_VERSION} ({EMBED_DIM}-d)   "
          f"student=Enc + Linear({EMB}->{EMBED_DIM})")
    t0 = time.time()
    specs, embs, species = build_distill_set(TARGET_SET)
    print(f"[teacher embeddings took {(time.time()-t0)/60:.1f} min]")

    t0 = time.time()
    student, val_cos = train_distill(specs, embs, species)
    print(f"[distillation took {(time.time()-t0)/60:.1f} min]")

    n_student = sum(p.numel() for p in student.parameters())
    print(f"\nstudent parameters: {n_student:,}  "
          f"(trunk {sum(p.numel() for p in student.trunk.parameters()):,} + "
          f"head {sum(p.numel() for p in student.head.parameters()):,})")

    # ---------------------------------------------------------- teacher vs student, A/B
    # Same species, same shots, same decoys, same beds, same seeds. The ONLY difference
    # between the two arms is which embedding space the verifier lives in.
    print(f"\n{'-'*90}\nTEACHER vs STUDENT (same test, only the embedding space differs)"
          f"\n{'-'*90}")
    DISTILL_EMBED_CTX = make_distill_embedder(student)
    globals()["DISTILL_EMBED_CTX"] = DISTILL_EMBED_CTX

    acc = {"perch": [], "distilled": []}
    try:
        for backend in ("perch", "distilled"):
            globals()["EMBED_BACKEND"] = backend
            assert EMBED_BACKEND == backend, "EMBED_BACKEND rebind failed"
            for sp in DISTILL_EVAL_SPECIES:
                try:
                    ctx = build_target(sp, verbose=False)
                except Exception as ex:
                    print(f"  skip {sp} ({backend}): {type(ex).__name__}: {ex}"); continue
                per_seed = []
                for sd in DISTILL_EVAL_SEEDS:
                    ec = eval_pass(ENC_D, ctx, sd)
                    sc = spec_pass(ENC_D, ctx, sd)
                    nc = neg_pass(ENC_D, ctx, sd)
                    m = score_eval(FUSIONS[CAND_D], ec)
                    _, _, _, _, auc = score_spec(FUSIONS[CAND_D], m["thr"], sc, ctx["DECOYS"])
                    per_seed.append(dict(auc=auc, f1=m["f1"], prec=m["p"], rec=m["r"],
                                         fa_per_hr=score_neg(FUSIONS[CAND_D], m["thr"], nc)))
                mean = {k: float(np.mean([d[k] for d in per_seed])) for k in per_seed[0]}
                sd_  = {k: float(np.std([d[k] for d in per_seed], ddof=1))
                        if len(per_seed) > 1 else 0.0 for k in per_seed[0]}
                acc[backend].append(mean)
                rows.append(dict(backend=backend, species=sp,
                                 n_seeds=len(DISTILL_EVAL_SEEDS),
                                 student_params=n_student, val_cosine=val_cos,
                                 auc=mean["auc"], auc_sd=sd_["auc"], f1=mean["f1"],
                                 prec=mean["prec"], rec=mean["rec"],
                                 fa_per_hr=mean["fa_per_hr"]))
                pd.DataFrame(rows).to_csv(DISTILL_CKPT, index=False)
                print(f"  [{backend:9s} {sp}] AUC {mean['auc']:.3f}  F1 {mean['f1']:.3f}  "
                      f"FA/hr {mean['fa_per_hr']:.1f}")
                gc.collect()
                try: torch.cuda.empty_cache()
                except Exception: pass
    finally:
        # Restore unconditionally. CELL 5 asserts EMBED_BACKEND=="perch" precisely because
        # leaving it flipped would make every later cell score a classifier in the wrong
        # space -- silently, with plausible numbers.
        globals()["EMBED_BACKEND"] = "perch"
        print(f"\n  EMBED_BACKEND restored to {EMBED_BACKEND!r}")

    # ---------------------------------------------------------- verdict
    if acc["perch"] and acc["distilled"]:
        pa = float(np.mean([d["auc"] for d in acc["perch"]]))
        da = float(np.mean([d["auc"] for d in acc["distilled"]]))
        df = pd.DataFrame(rows)
        sd_p = float(df[df.backend == "perch"].auc_sd.mean())
        sd_d = float(df[df.backend == "distilled"].auc_sd.mean())
        drop = da - pa
        bound = max(DISTILL_NOISE_FLOOR, sd_p, sd_d)
        print(f"\n{'='*90}")
        print(f"Perch verifier      mean species AUC  {pa:.4f}")
        print(f"Distilled verifier  mean species AUC  {da:.4f}")
        print(f"change {drop:+.4f}   bound {bound:.4f}   "
              f"student is {12_000_000/n_student:.0f}x smaller than Perch")
        if abs(drop) - bound < -0.25 * bound:
            print("-> COMPRESSION IS ESSENTIALLY FREE at this scale: the student matches "
                  "the teacher within the measured noise bound, with no Perch inference "
                  "at detection time. This is the deployable result.")
        elif drop < 0:
            print(f"-> The student LOSES {abs(drop):.3f} AUC. Report it as the measured "
                  f"cost of {12_000_000/n_student:.0f}x compression -- an "
                  f"accuracy/deployability frontier, not a failure, provided it stays "
                  f"clear of chance (0.50).")
        else:
            print("-> The student MATCHES OR BEATS the teacher. Check for leakage before "
                  "believing it: confirm the validation split was by species, and that "
                  "decoy ranking still ran on Perch under both arms.")
        print(f"{'='*90}")
        print(f"wrote {DISTILL_CKPT}  ({len(rows)} rows)")
