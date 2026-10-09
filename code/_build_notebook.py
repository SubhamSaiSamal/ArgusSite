# -*- coding: utf-8 -*-
"""Build ARGUS_Kaggle_Notebook.ipynb from the tested .py source files.

Deliberately NOT hand-typed: each source file becomes exactly one code cell (matching
the "CELL 1/2/3/4" naming already baked into each file's own header comment), so the
notebook is byte-identical to code that already passed 94 local assertions rather than a
transcription that could silently diverge. Re-run this after any edit to the .py files;
do not hand-edit the .ipynb.
"""
import json, os, hashlib, uuid

ROOT = os.path.dirname(os.path.abspath(__file__))   # this folder (code/)
OUT = os.path.join(ROOT, "ARGUS_Kaggle_Notebook.ipynb")

FILES = [
    ("arguswala_setup_v2.py",   "Cell 1 — Setup"),
    ("arguswala_v2.py",         "Cell 2 — Two-Stage Detection, Multi-Species"),
    ("arguswala_sweeps.py",     "Cell 3 — Shot-Selection & Shot-Count Sweeps"),
    ("arguswala_ladder.py",     "Cell 4 — Ablation Ladder"),
    ("arguswala_confounds.py",  "Cell 5 — Confound-Closing Runs"),
    ("arguswala_distill.py",    "Cell 6 — Perch Distillation"),
    ("arguswala_frontier.py",   "Cell 7 — Bottleneck Frontier & Wild-Audio Scan"),
]

def _cid():
    # nbformat 4.5 requires a cell id; deterministic-length hex, not tied to wall clock
    # since this script must be re-runnable byte-for-byte (Date.now()-style calls aren't
    # available/desired for a build script either) -- random is fine, ids just need to
    # be unique within the notebook.
    return uuid.uuid4().hex[:8]

def md(src):
    return {"cell_type": "markdown", "id": _cid(), "metadata": {},
            "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "id": _cid(), "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md(
"""# ARGUS — Multi-Species Few-Shot Bioacoustic Detection

Kaggle notebook, built from the tested pipeline (`arguswala_setup_v2.py`,
`arguswala_v2.py`, `arguswala_sweeps.py`, `arguswala_ladder.py`, `arguswala_confounds.py`,
`arguswala_distill.py`, `arguswala_frontier.py` — 394 local assertions passing as of the
last edit; see `_test_v2_logic.py` / `_test_ladder.py` / `_test_confounds.py` /
`_test_distill.py` / `_test_frontier.py` in the project repo). One notebook cell per
source file, run top to bottom.

## Current state of this build (4 Sep 2026) — FRONTIER + WILD-SCAN, ATTEMPT 2

`SMOKE_TEST = False`. **Do not set it True** — it cuts `TARGET_LIST` to 2 species and
breaks the `[:5]` slices in Cells 5–7.

### What happened on 3 Sep, and what changed

The first attempt died at **57m51s with a CUDA out-of-memory error**, 15 seconds into the
first distilled arm. It had already spent that hour usefully — setup, the 10,163-pair
teacher set, and the full Perch control — but wrote only the control's five rows.

**Cause.** The held-out set was pushed through the student's trunk in a *single* forward
pass. Cell 6 does the same thing safely because its teacher set is 2,636 pairs (~400 held
out); Cell 7 raised it to 10,163 pairs (~1,633 held out), and the first conv block emits
128 channels at the full 128×86 spectrogram resolution — 5.6 MB of activations per crop.
Its BatchNorm asked CUDA for **8.69 GiB in one allocation**. Fixed by streaming the
held-out set in batches, in **both** Cell 6 and Cell 7 — the trap is the dataset size, not
the cell. `m.eval()` puts BatchNorm on running statistics, so batching cannot change the
number and Cell 6's archived 0.399 stays comparable.

**The time estimate was also wrong,** in a way worth recording: it costed each arm's
*evaluation* and forgot its *training* entirely. That is how "~5.5 h" became ~8 h. This
build no longer trusts an estimate. The failed run measured the pieces — teacher set
12.7 min, Perch control 34.6 min, one training epoch ~11 s — and the cell now enforces a
budget instead of predicting one:

- `CELL7_BUDGET_H = 10.0` — a deadline clock started when Cell 7 begins.
- `SCAN_RESERVE_MIN = 180` — Part A refuses to start an arm it cannot finish, and names
  the arms it skipped so a follow-up run is a one-line edit.
- Part B stops mid-scan with its CSV and clips **written**, rather than being killed at
  the ceiling holding everything in memory.

**Arms now run in order of information value, not width** — 128 (replicates Cell 6), then
512 (the widest), then 64, 256, and the nonlinear head last. If the budget guard trims the
tail, what survives is still the contrast that answers the question. The report sorts by
width regardless of run order.

**Part B now scans both species in one pass over the files.** Each `.ogg` is decoded once
instead of twice, but the real reason is that a two-pass version under a deadline can scan
400 files for the study species and only 150 for the control — and a rate comparison
across two different file sets is not a comparison.

Also trimmed: `FRONTIER_EPOCHS` 150 → **90** (still 5.7× Cell 6's total gradient steps, so
a null result cannot be blamed on undertraining — which matters, because "both flat" is
one of the pre-registered outcomes), and `SCAN_N_FILES` 500 → **400** (~27 h of audio).

### What the two parts are for

**Part A — bottleneck frontier.** Cell 6 distilled Perch into the stage-1 network and lost
0.136 AUC. This tests the most specific explanation for *why*: Perch emits 1536 dimensions,
the student's trunk emits 128, and a linear head from 128-d can only reproduce a
128-dimensional subspace of Perch's space no matter how long it trains. Sweeping the
bottleneck (64 / 128 / 256 / 512, plus a nonlinear head at 512) turns a single pass/fail
into an accuracy-vs-size curve — 25× down to 7× smaller than Perch. All three outcomes are
pre-registered in the cell header, including the awkward one where fidelity rises but
species AUC does not.

**Part B — wild-audio scan.** Every number in this project so far comes from calls
*planted* into soundscape. This scans 400 unlabelled soundscapes (~27 h), strictly disjoint
from the 14 calibration/evaluation beds, at the same best-F1 threshold everything else
uses — plus the same scan for a control species on the identical files. Saves the top 60
detections per species as listenable clips. It cannot *confirm* a detection (no ground
truth, no expert ear), but it establishes a rate, a ranked shortlist, and a comparison.

Flag state: `SKIP_MAIN_LOOP` / `SKIP_LADDER` / `SKIP_SWEEPS` / `SKIP_DISTILL` **`True`**;
Cell 5 `RUN_P1`/`RUN_P2`/`RUN_P3` **`False`**; Cell 7 `SKIP_FRONTIER` **`False`**.
Cell 6 skips but still defines `build_distill_set` and `make_distill_embedder`, which
Cell 7 reuses — so it must still execute.

**Session log:**

| Session | Status | Result |
|---|---|---|
| P2 re-run, 6 seeds | done 29 Aug, 3h40m | Settled: verified **falls −0.050**, negative 5/5. |
| Widened sweeps, 5sp×3seed | done 30 Aug, 8h13m | Two conclusions strengthened, shot-count gain shrank 0.116→0.078. |
| Perch distillation | done 30 Aug, 1h26m | −0.136 mean; free for one species, chance-level for the study species. |
| Frontier + wild scan | **failed 3 Sep, 57m51s** | CUDA OOM in held-out validation. Perch control salvaged: mean AUC 0.713 over 5 species. |
| Frontier + wild scan, attempt 2 | **this build**, ~8 h budgeted, 10 h ceiling | — |
| DCASE external validation | not started | Roadmap §4, 5-day timebox. |

## Before running — checklist

1. **Notebook settings (right sidebar):** Accelerator = **GPU**, Internet = **On**.
2. **Add data**, attach:
   - **BirdCLEF 2024** competition data (`birdclef-2024`)
   - **A Perch model.** Which one depends on `PERCH_VERSION` in Cell 1 (default `"v2"`):
     - `PERCH_VERSION = "v2"` → attach `google/bird-vocalization-classifier` →
       variation **`perch_v2`** (GPU build) — the CPU fallback `perch_v2_cpu` also works
       if no GPU is attached, but detection is automatic either way.
     - `PERCH_VERSION = "v1"` → attach `google/bird-vocalization-classifier` →
       variation **`bird-vocalization-classifier`**, version **8**.
   - A version mismatch between the attached model and `PERCH_VERSION` raises
     immediately with a clear message rather than silently scoring the wrong model.
3. Cells 1–7 build on each other's names — run in order, don't skip cells manually (the
   skip *flags* inside Cells 2–6 handle that for you; the cells themselves still need to
   execute so their functions and cached state exist for Cell 7).
4. Watch Cell 1's first print line — `train_metadata.csv columns: [...]` — and confirm
   `author` is listed (Cell 5's recordist-disjoint test needs it; already confirmed
   present in the 27 Aug run, but re-check if the attached dataset version ever changes).
5. Watch Cell 5's own timing print after the very first (species, seed) pair completes —
   it projects a total for that section. Stop and widen `PERCH_ALONE_STRIDE_S` if it
   looks too slow rather than finding out hours later.

See `ARGUS_Roadmap_Aug_Sept_2026.md` (§1b) for why the ablation ladder (Cell 4) and a
Perch-version decision should both happen on a *few* species before committing to the
full multi-species run in Cell 2 — decoys are ranked by Perch-embedding proximity, so
changing the verifier config after Cell 2 has started makes species runs incomparable.
"""))

for fname, title in FILES:
    path = os.path.join(ROOT, fname)
    src = open(path, encoding="utf-8").read()
    cells.append(md(f"## {title}\n\n`{fname}`"))
    cells.append(code(src))

cells.append(md(
"""## After running

- **This build newly writes three files** — `argus_frontier.csv` (one row per
  species per arm: width, parameter count, teacher cosine, species AUC, F1, FA/hr),
  `argus_wildscan.csv` (every detection above threshold, with file, timestamps and both
  scores), and `argus_wildscan_clips.npz` (the top 60 clips per species, as raw waveforms —
  `np.load` it and listen). Nothing else is touched: Cells 2–6 all exit early, so every
  existing CSV in the repo stays as it is.
- **Download all three.** The `.npz` is the one that is easy to forget and the only one
  that cannot be regenerated without re-running the scan.
- **Check the console for a `[budget]` line before reading the frontier table.** If one
  appears, Part A stopped early and the table has fewer arms than planned; the line names
  exactly which widths are missing. A truncated curve is still publishable — a curve with
  an unnoticed hole in it is not.
- **Listen to the clips before believing anything in the scan CSV.** The detection counts
  are real; whether any of them is actually *S. albiventris* is not established by this
  cell and cannot be. Compare the study-species rate against the control rate first — if
  they are close, that is the finding, and it belongs in the paper as one.
- **Download the new CSV before the session ends** (or commit the notebook). Kaggle's
  Output preview is virtualised and its plain-text extraction silently stops short of the
  full table — a real download, or reading the preview's underlying accessibility table
  down to its "No more data to show" marker, is the only reliable capture. See lab
  notebook Day 35 for how six rows were nearly lost this way.
- **Re-verify before publishing.** Every aggregate quoted anywhere public should be
  recomputed from the downloaded CSV, not copied from the console. A stale console
  number is how the README came to disagree with its own archived data (found 29 Aug).
- Log the headline numbers — per-species AUC before/after, the winning ladder
  config, which Perch version was used, and Cell 5's three confound verdicts — in the
  bound logbook **the same day**.
- If this was a `SMOKE_TEST` run: record hours-per-species-per-seed, then come back,
  set `SMOKE_TEST = False`, and size `SEEDS` / `N_TARGETS` to the real budget before
  the next run.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11", "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3", "file_extension": ".py"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# --------------------------------------------------------------------- verification
print(f"wrote {OUT}")
with open(OUT, encoding="utf-8") as f:
    reloaded = json.load(f)
print(f"cells: {len(reloaded['cells'])}  "
      f"(md={sum(1 for c in reloaded['cells'] if c['cell_type']=='markdown')}, "
      f"code={sum(1 for c in reloaded['cells'] if c['cell_type']=='code')})")

# Byte-exact reconstruction check: concatenating each source file's own code cell must
# reproduce the original .py file exactly -- proves this is packaging, not transcription.
ok = True
code_cells = [c for c in reloaded["cells"] if c["cell_type"] == "code"]
for (fname, _), cell in zip(FILES, code_cells):
    original = open(os.path.join(ROOT, fname), encoding="utf-8").read()
    rebuilt = "".join(cell["source"])
    h1 = hashlib.sha256(original.encode()).hexdigest()[:12]
    h2 = hashlib.sha256(rebuilt.encode()).hexdigest()[:12]
    match = original == rebuilt
    ok &= match
    print(f"  {fname:<26} original={h1} notebook={h2}  {'MATCH' if match else 'MISMATCH!!'}")

# Every code cell must independently satisfy ast.parse -- confirms valid, complete Python,
# not merely correct text.
import ast
for (fname, _), cell in zip(FILES, code_cells):
    src = "".join(cell["source"])
    try:
        ast.parse(src)
        print(f"  {fname:<26} ast.parse: OK")
    except SyntaxError as e:
        ok = False
        print(f"  {fname:<26} ast.parse: FAIL {e}")

print("\n" + ("ALL CHECKS PASSED" if ok else "CHECKS FAILED -- do not ship"))
