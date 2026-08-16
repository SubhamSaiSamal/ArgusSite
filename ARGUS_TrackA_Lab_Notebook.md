# Project ARGUS — Track A Lab Notebook

**Scope:** Track A only (bioacoustics few-shot detector, DCASE Task 5 baseline). Track B (ESP32 hardware) and Track C (video/submission) are owned by teammates and are out of scope here.

**Purpose:** This is the dated, step-by-step engineering log for Track A's software-polish week (the 7 days before regionals). It's a companion to `ARGUS_Official_PRD.md` — the PRD states the plan and requirements; this notebook records what was actually done, what was found, and why decisions were made. It satisfies the "dated lab notebook" rigor requirement in PRD §18 and is meant to be **appended to after every work session**, not rewritten.

A note on screenshots: for Day 1–3's own work, this environment couldn't save in-session browser screenshots as image files, so those entries describe what was seen in words/tables instead of embedding pictures. Starting with the noise-robustness figures below, Akshath supplied real screenshots directly (saved to `screenshots/` next to this file) — those are embedded where relevant.

---

## Day 1 — 2026-07-01 — Refactor into setup()/detect() + speed experiments

**Goal (per roadmap):** turn the working baseline notebook into two callable functions — `setup()` (one-time cost: install/clone, load checkpoint, compute training-set stats) and `detect(audio_file)` (per-clip cost) — with a target of **under ~15 seconds** per `detect()` call on a fresh clip, so Day 2's app can call it repeatedly without re-paying the one-time cost.

**What was done:**
- Verified both existing baselines still reproduce cleanly on a fresh runtime: cross-correlation F = 44.444%, prototypical network F = 52.857%.
- Wrote `setup()` and `detect()` in a new notebook cell (`dcase-few-shot-bioacoustic` repo + config + checkpoint load happen once in `setup()`; feature extraction + scoring happen in `detect()`).

**Bug found and fixed:** first working version of `detect()` took **37.03s** per clip — far over target. Root cause: the feature-extraction step was processing *every* audio file sitting in the shared validation folder (both `ME1.wav` and `ME2.wav`), not just the one clip requested. Fix: each call now copies the requested clip (+ its annotation CSV) into its own isolated scratch folder before running feature extraction. This dropped the time to **26.09s** — better, but still over target, which motivated the speed experiments below.

**Speed experiments (user-authorized — "try cutting the iteration count... let's see how much it helps and how much it makes results go down"):**

The official scorer's `evaluate_prototypes()` function (in `util.py`) averages its prediction over `conf.eval.iterations` random draws of negative examples before thresholding — a config value, not hardcoded. Three levers were tested:

| Lever | Official default | Tested | Effect on speed | Effect on accuracy |
|---|---|---|---|---|
| `iterations` (negative-resampling draws) | 10 | 10 / 5 / 3 / 1 (3 repeats each, one validation clip) | 10→3 cut scoring time ~3x (~16s → ~5s) | F-score stayed in the same 65–72% band at every setting tested — no clear degradation down to 3 iterations in this test. Caveat: only 3 repeats per setting, so this shows it held on this clip, not that it's safe in general. Going to 1 iteration removes the averaging safety net entirely even though it didn't visibly hurt here. |
| `query_batch_size` | 8 | 8 → 32 | ~20% additional speedup (pure batching, no semantic change) | None (same math, just less Python/GPU-transfer overhead per call) |
| `torch.no_grad()` around inference | not used | added | ~3% faster | None (this loop was never doing backprop; the win is negligible but free, so kept anyway) |

**Decision:** shipped `detect(audio_file, support_examples=None, iterations=3, query_batch_size=32)` with `torch.no_grad()` wrapping the scoring call. Both `iterations` and `query_batch_size` remain real function parameters (not hardcoded) — pass `iterations=10, query_batch_size=8` to get the exact official behavior back if the Day 4 reliability rehearsal finds 3 iterations too noisy on other clips.

**Result — confirmed via the "Day 1 test" cell on a fresh runtime:**

| Stage | detect() time |
|---|---|
| First working version (redundant file processing bug) | 37.03s |
| After isolating each clip into its own scratch folder | 26.09s |
| Final (iterations=3, query_batch_size=32, no_grad) | **13.15s** |

Target of <15s met. Re-verified again later in the session on a completely fresh Colab runtime: `setup()` in 12.6s, `detect(ME2.wav)` in 13.87s.

**Known open gap (flagged, not yet solved):** `detect()` requires a DCASE-format CSV (first 5 POS rows = the 5-shot support set) sitting next to the audio file — it does not yet accept arbitrary user-picked/marked examples. Carried into Day 2's design decision below.

---

## Day 2 — 2026-07-01 — Gradio app (v1, then v2 visual/functional polish)

**Goal (per roadmap):** build a small app on top of `detect()` — pick a clip, click Detect, see the result — as the first interactive demo layer for the reliability rehearsal and eventually the live regionals demo.

### v1 — first working app

- Built with `gr.Blocks`: a clip dropdown (starts with one preset, "Meerkat calls (ME2)"), a Detect button, a matplotlib mel-spectrogram with detections shaded in red, a results table, and a plain-English result line.
- **Design decision on the Day 1 support-set gap:** rather than build a full "mark your own 5 examples on a waveform" UI (a much larger, riskier feature this close to regionals), the app offers **preset clips only** — audio that already has its matching support-set CSV sitting next to it in the validation data. This directly matches the roadmap's Day 3 plan ("add a second target-sound preset") — adding a preset going forward is a one-line addition to a dictionary, not new UI work.
- **Verified twice:** once embedded in the Colab notebook, once on the public Gradio share link opened in a separate browser tab, confirming the app works independent of the notebook's own rendering. Results across these runs: 43 / 45 / 46 detections on the same clip, 13.9–16.9s each — the run-to-run count difference is expected noise from the `iterations=3` random negative sampling characterized on Day 1, not a bug.

### v2 — visual + functional polish (this session, in response to explicit feedback: "make the app more visually appealing as well as more functionally appealing")

Visual changes:
- Custom theme (`gr.themes.Soft`, emerald/amber palette) and header with an owl icon + "ARGUS" branding, matching the PRD's "hundred-eyed watchman" framing.
- Collapsible "How this works" panel explaining the 5-shot support-set concept in plain language, for judges/visitors who haven't seen the project before.
- Spectrogram now has a title, a legend, and a max-width layout so it doesn't sprawl on wide screens.

Functional changes:
- **Support-set vs. detection shading:** the spectrogram now shades the 5-shot support set (the examples the model was *given*) in green and detected events (what it found *on its own*) in red, with a legend. This matters scientifically, not just visually — DCASE scoring only counts detections after the 5th support event ends, so the earlier version's plot (which only showed red) could have been read as "the model detected the support-set calls too," which isn't what's happening.
- **Audio player** so a judge/visitor can actually listen to the clip, not just look at a spectrogram.
- **Adjustable `iterations` / `query_batch_size` sliders**, defaulting to the Day 1 tuned values (3 / 32) with the official values (10 / 8) reachable by sliding — lets someone feel the speed/accuracy tradeoff live instead of reading about it.
- **Downloadable CSV** of the detections table (`last_detections.csv`), for a mentor or judge who wants the raw numbers.
- **Progress indicator** during detection ("Running the detector..." → "Rendering spectrogram...") and **basic error handling** (a failed run now shows an error message in the Result box instead of crashing the app).

**Bug hit and fixed during this build:** the header's owl emoji was written as a Python unicode escape (`\U0001F989`) inside a JavaScript string used to set the clipboard; JavaScript doesn't recognize `\U` as an escape sequence and silently dropped the backslash, so the literal text `U0001F989` appeared in the rendered header instead of 🦉. Fixed by typing the actual emoji character directly into the cell instead of relying on an escape sequence.

**Verification:** re-ran the enhanced app on a new public Gradio link (previous v1 server closed cleanly via `gr.close_all()` first) and clicked through it end-to-end: header/theme rendered correctly, audio player loaded and was scrubbable, Detect produced "Found 45 likely calls in 16.9s" with the green/red legend visible on the spectrogram, the detections table scrolled correctly, and the CSV download produced a real 2.0 KB file.

**Known gap, unchanged from v1:** still preset-clips-only; no free-form audio upload yet (needs the support-set-marking UI decision, still not built).

---

## Day 3 — 2026-07-01 — Second preset + CPU-runtime findings

**Goal (per roadmap):** add a second target-sound preset to the Gradio app, per the Day 1/2 design decision (presets are a one-line dictionary addition, no new UI work).

**Environment change first (not part of the plan, but it shaped everything below):** Colab's GPU quota was exhausted on reconnect this session ("Cannot connect to a GPU due to usage limits"). Chose "Connect without GPU" to keep moving, since picking a second preset is data curation, not heavy computation — but this choice ended up mattering a lot for the timing numbers below.

**Finding: this repo's local data only has one species.** Searched `Development_Set/Validation_Set/` and `Development_Set/Training_Set/` for any audio folder other than `ME` (Meerkat) with a matching annotation CSV. Result: only two folders exist anywhere in this environment — `Validation_Set/ME` and `Training_Set/MT` (also Meerkat). This is a reduced/lightweight copy of the DCASE data for the hackathon, not the full multi-species dataset. So a genuinely different *species* preset isn't available without downloading more DCASE data, which is out of scope for this week.

**Decision:** added `ME1.wav` (16 POS rows, 23 total annotated rows) as the second preset — a different Meerkat recording, not a different species. Documented this honestly in the app's own code comment rather than implying it's a different target sound. `PRESETS` now has two entries:
- `"Meerkat calls (ME2)"` → `ME2.wav` (Day 2's original preset)
- `"Meerkat calls (ME1)"` → `ME1.wav` (new)

**Bug hit while reconnecting (worth flagging for future sessions):** the "Run before" (Ctrl+F8) chain from the previous disconnect had stalled on an old cross-correlation-baseline cell positioned *above* the `setup()`/`detect()` definitions — that cell reprocesses the entire validation set and was still showing "Executing" after 3 minutes on the fresh CPU runtime. Used Runtime → Interrupt execution to kill it, then ran the `setup()`/`detect()` definition cell and a `setup()` call directly instead of relying on "Run before" again. Lesson: after a disconnect, explicitly re-run the specific cells needed rather than trusting "Run before" to land cleanly when older, unrelated cells sit earlier in the notebook.

**Verification — ran on the live Gradio app (public link), not just re-used old results:**

| Step | Result |
|---|---|
| `setup()` on fresh CPU runtime | 45.6s (vs. ~12.6s on GPU on Day 1 — CPU is meaningfully slower even for the one-time setup cost) |
| Dropdown shows both presets | Confirmed — "Meerkat calls (ME2)" and "Meerkat calls (ME1)" both selectable |
| Clicked Detect on ME1 (new preset), default sliders (iterations=3, batch=32) | **331.6s**, 58 detections. Spectrogram rendered with correct green (support set) / red (detections) shading and legend, audio player worked, detections table populated, CSV downloaded (2.6 KB, real file) |

**Honest timing note — do not carry the 331.6s number forward as "how slow the app is":** this is CPU, not GPU, and ME1 has more annotated segments to score than ME2. Day 1's GPU numbers (13–16s per `detect()` call) are the relevant figures for judging demo-day responsiveness, since regionals will presumably have GPU access or at least won't have hit a quota wall mid-demo. The 331.6s CPU run confirms *correctness* (the new preset works end-to-end), not *speed* — those are different questions and this entry keeps them separate on purpose.

**Known gap, unchanged from Day 2:** still preset-clips-only, still single-species (Meerkat) only, given local data constraints. If a genuinely different species is wanted before regionals, it requires pulling additional DCASE Development_Set folders (e.g. from the Zenodo archive already referenced in `setup()`'s zip-download fallback path) — not attempted this session, flagged for Day 4+ if there's time.

### Noise-robustness prototype (supplied by Akshath, run separately from this session)

Akshath ran a noise-robustness check on `ME1.wav` and supplied the screenshots below — this satisfies the roadmap's Day 3 line item "one noise-robustness idea prototyped." **Flagging honestly: the exact methodology (how the noise was synthetically added, what SNR=0dB/−6dB/+10dB means precisely — e.g. additive white noise scaled to those ratios — and whether this ran with the `iterations=3, query_batch_size=32` Day 1 defaults or the official 10/8 settings) was not confirmed with Akshath before this entry was written.** The numbers below are read directly off the supplied charts; if these get cited at regionals, confirm the methodology first so a judge's question about "how was noise added" has a real answer.

**F-score vs. SNR on ME1.wav — four separate runs:**

| Run | "Clean" baseline (single run, no fixed seed) | SNR = −6 dB | SNR = 0 dB | SNR = +10 dB |
|---|---|---|---|---|
| 1 | 28.1% | 8.1% | 26.1% | 41.7% |
| 2 | 28.6% | 8.2% | 17.4% | 38.5% |
| 3 | 33.3% | 4.4% | 28.1% | 35.7% |
| 4 | 25.8% | 11.9% | 20.3% | 38.5% |

![F-score vs SNR, run 1](<screenshots/fscore_vs_snr (1).png>)
![F-score vs SNR, run 2](<screenshots/fscore_vs_snr (2).png>)
![F-score vs SNR, run 3](<screenshots/fscore_vs_snr (3).png>)
![F-score vs SNR, run 4](screenshots/fscore_vs_snr.png)

**Reading on the pattern:** F-score rises monotonically with SNR in every one of the 4 runs (more background noise → worse detection, as expected — this is a sanity-check result, not a surprise). What's more interesting is the spread: the "clean" baseline alone ranges from 25.8% to 33.3% across 4 single-draw runs (a 7.5-point spread with nothing else changed), and the 0 dB point ranges 17.4%–28.1% (10.7 points). This is the same "no fixed seed, one draw" issue already flagged on Day 1 for the `iterations` experiments, just showing up again here — **any single noise-robustness number from one run should be treated as noisy, not a precise measurement,** until averaged over multiple seeded runs (not done here or on Day 1).

**Detection "hero shot" — spectrogram with support/detections/ground-truth rows:**

![ARGUS detection hero shot on ME1.wav](screenshots/hero_shot_ME1.png)

This shows the first 38s of the 652s `ME1.wav` recording with three annotated rows above the spectrogram: the 5-shot support set (white, hatched), the model's detections (red), and ground truth (green) — letting you visually check detections against ground truth rather than just the support-set-vs-detection view the Gradio app shows. Caption states **F=55.6% "on ME validation slice."** Note this number doesn't match the "clean, single run" values in the table above (25.8%–33.3%) — likely a different evaluation subset or scoring setup, but this wasn't confirmed either; treat the two F-scores as separate measurements, not the same number restated. (All 4 copies of this screenshot Akshath supplied were pixel-identical, so this is one run/figure, not 4.)

---

## Day 3.5 — 2026-07-02 — ISEF-worthiness audit (Fable, via Claude Code) + PRD corrections

**What happened:** Akshath ran a deep-research critical-analysis pass using Fable (Claude Code) against the PRD and this notebook, asking: is the project ISEF-grand-award-worthy as scoped, what are the flaws, and what new directions could raise the ceiling. Full report saved as `ARGUS_ISEF_Audit_2026-07-02.md`. This entry records what changed as a result.

**Verdict, in short:** as a *plan*, the PRD is solid — IRIS-qualifying / special-award territory with a real (if narrow) grand-award path. As *executed* (this notebook), the project is at "strong regional PoC" and no further — every logged day so far (Day 1–3, all dated 2026-07-01) is baseline reproduction and demo polish; zero rungs of the §4.2 method ladder above Baseline B exist. That's not a surprise reading this notebook honestly, but it's useful to have it stated plainly by an independent pass.

**Two corrections that matter regardless of timeline:**
- DCASE Task 5 was discontinued after the 2024 edition (2025's DCASE replaced it with a different task). The frozen 2024 leaderboard is still valid to benchmark against, but "active/live scoreboard" framing is now stale and should not be used in any write-up or pitch.
- The PRD's SOTA/baseline numbers were a year stale: current 2024 SOTA is 65.2% (frame-level embedding models, not prototypical/transductive — that family has been superseded), baselines are 41.6% (prototypical) / 14.9% (template-matching), not the "~63%"/"~30%" figures used before. This also means Baseline B in the method ladder should be rebuilt on a frame-level-embedding backbone, not prototypical, once Phase 2 starts.

**Novelty framing was oversold and has been rewritten (PRD §3.4, §19):** the original claim ("rare + weak + on-device is an unclaimed intersection") doesn't survive a literature check — edge/TinyML bioacoustics is a crowded published subfield (TinyChirp, hornbill/burrowing-owl TinyML, Bird@Edge), and a July 2025 paper already pairs few-shot detection with an extremely lightweight model (directly pre-empting the PRD's own §4.4 "associative-memory head" throwaway idea). The corrected, defensible claim: isolating **low-SNR robustness as a controlled variable** in few-shot bioacoustic detection, measured with an ablation on the DCASE benchmark — that specific framing does appear to still be open. Reflected in the PRD; also added the prior-art citations directly into the Q&A prep (§19) so the team can name them first rather than get caught not knowing them.

**Statistics plan was under-scoped for the sample size available.** This notebook's own Day 3 data is the proof: the "clean" single-run F-score swung 25.8%–33.3% across four runs with nothing else changed, and the 0 dB point swung 17.4%–28.1% — roughly ±10 F-points of pure run-to-run noise from unseeded `iterations=3`. A naive per-class-per-SNR significance claim will not survive that. PRD §7 now specifies matched-pair bootstrap on shared noise realizations, aggregated across classes per SNR bin, with ≥10 seeds per point — this needs to be the actual protocol once real experiments start, not the demo's speed-tuned unseeded default.

**Data gap confirmed, not new:** the audit independently found the same thing Day 3 already flagged — the local repo only has Meerkat data, not the real class-disjoint multi-class DCASE set. This is now explicitly the first task of Phase 2 (PRD §10): pull the real 2024 Development + Evaluation set from Zenodo before re-citing any F-score past the regional stage.

**Roadmap changes (PRD §10 Phase 2):** added an explicit Week-9 go/no-go gate (if no significant low-SNR gain by then, pivot the headline to the edge Pareto result and lean EBED — decided now, not in October); demoted the field pilot from a multi-week campaign to one chainsaw-at-distance session (the single highest-impact-per-hour experiment, per the audit's §3.6) plus one real field session; **cut RQ4 (open-set detection) entirely** — a one-class detector for a critically endangered species was published Oct 2025, so this is occupied prior art, not a stretch goal worth the remaining time.

**Important scope note, so this doesn't get misread later:** none of the above changes the Jul-8 CBSE Skill Expo sprint (`ARGUS_SkillExpo_Roadmap_to_Jul8.md`). That file already correctly scopes this week as "a proof-of-concept + pitch, NOT the full ISEF research" — the audit is entirely about the Phase 2 window that starts *after* the 8th. Only two cheap fixes were pulled into this week: seed-averaging Friday's F-vs-SNR figure if there's time, and an honesty-pass on benchmark/data-scope language in the write-up and video before Monday's mentor review. Everything else in the audit — real data, a real mechanism, corrected statistics, category decision, Track B logging — is Phase 2 work and is now written into the PRD roadmap instead of just this notebook.

---

## Day 4 — 2026-07-02 — Closing two Day 3 loose ends: noise methodology confirmed, hero-shot bug found and fixed

**Goal:** per the roadmap's own Friday note ("don't present a single unseeded run as 'the' result if it can be avoided") and the audit's §2.9 fix ("nail down and document the noise-injection method... reconcile the two F-numbers"), go back into the actual `ARGUS_TrackA_DCASE_Baseline.ipynb` Colab notebook (account: `valkblox0@gmail.com` — the notebook is not shared with the `suvransu.samal476@gmail.com` account Akshath mentioned switching to; confirmed via "Access denied" when trying) and read the real code behind Day 3's SNR figure and hero shot, rather than guessing.

### 1. Noise-injection methodology — confirmed from code (previously "not confirmed with Akshath")

Read `make_snr_wav()` directly. What it actually does:
- **Noise source:** not forest noise or synthetic white noise — a 60-second segment of ME1.wav *itself* (150.0s–210.0s, presumably a quiet stretch), tiled to the full clip length and added back to the original signal.
- **SNR definition:** whole-clip RMS ratio in dB — `target_noise_rms = sig_rms / (10**(snr_db/20))`, `scaled_noise = noise * (target_noise_rms/noise_rms)`, `mixed = clean_signal + scaled_noise`. Both `sig_rms` and `noise_rms` are computed over the *entire* clip, not just the event regions — so this is a global/whole-clip SNR, not an event-level target-to-noise ratio.
- Peak-normalized to 0.99 after mixing if it would otherwise clip.
- No fixed seed anywhere in this function — confirms the "unseeded, one draw" caveat already on the chart is accurate, not overcautious.

This replaces the "e.g. additive white noise" guess in the Day 3 entry — the real method is self-referential (uses a noise-only segment of the same recording), which is a reasonable and defensible choice for a quick demo but is worth stating precisely if a judge asks "how was noise added."

### 2. Hero-shot bug found: it was never showing prototypical-network detections

This is bigger than "two F-scores don't match." Tracing the code:
- The hero-shot figure was titled "Prototypical-Network Detection on ME1.wav (F=55.6%...)", but `F=55.6%` is a **hardcoded string** in the `fig.suptitle(...)` call — not a number computed anywhere in that cell.
- The predictions actually plotted (`pred_me1`) are loaded from `stretch_output.csv`, which is Section 5's **cross-correlation/template-matching** baseline output — not the prototypical network.
- The prototypical-network scoring cell in this notebook was manually `KeyboardInterrupt`ed partway through feature extraction (stopped at iteration 2 of 3, ~14% through) and never finished — so no real prototypical predictions existed anywhere in this notebook to plot. The figure's title was aspirational, not descriptive.

**Fix (done, verified live in Colab today):**
- Reconnected the Colab runtime (CPU only — GPU quota still exhausted on this account, same as Day 3; "Cannot connect to a GPU due to usage limits" on first attempt).
- Re-ran the notebook from the top through Section 5 (env setup → clone `dcase-few-shot-bioacoustic` → pull the ME subset via `remotezip` → cross-correlation baseline → official evaluator). Skipped the prototypical/"Stretch" section entirely (not needed for this fix, and it's the slow part).
- Fresh, verified official-evaluator output today: `ME1.wav {TP:5, FP:0, FN:6}`, `ME2.wav {TP:11, FP:4, FN:30}`, **Overall: precision=0.80, recall=0.308, F=44.444%** (ME1+ME2 combined) — matches Day 1's originally reported cross-correlation F=44.444% exactly, so that number was always correct; only the hero-shot's *label* was wrong.
- Corrected the caption to: *"ARGUS Track A — Cross-Correlation (Template-Matching) Baseline Detection on ME1.wav (F=44.4%, official evaluator, ME1+ME2 combined)"* and regenerated the actual figure from the real, fresh `output.csv` (not the stale `stretch_output.csv` reference, which doesn't exist in a clean runtime — `os.path.exists` confirmed `False`). New figure shows only 3 detections in the first 38s (vs. the old mislabeled figure's denser-looking red bars), consistent with the weaker template-matching method.
- The new `hero_shot_ME1.png` is saved in the Colab session (`/content/hero_shot_ME1.png`) and downloads to Akshath's local machine via the notebook's existing `files.download()` cell — **not yet pulled into this workspace's `screenshots/` folder**, since Cowork's browser control and this file-tool sandbox are different environments. Akshath needs to grab it from his Downloads folder and drop it into `screenshots/` to replace the old mislabeled one before it goes in the submission image set.
- **Still open, not done today:** an honest prototypical-network hero shot doesn't exist yet. If a "the model actually learned this" hero shot is wanted (rather than the weaker template-matching baseline), the prototypical section needs to be run to completion on this notebook — expect several minutes on CPU based on today's timings.

### 3. Seed-averaging the F-vs-SNR figure — attempted, not completed (infeasible in one sitting on current compute)

Akshath opted to try this live. Real evidence gathered before stopping:
- The cross-correlation baseline alone — the *fast*, no-training baseline — took **3 minutes** on this fresh CPU runtime.
- Day 3 already measured the neural prototypical `detect()` call at **331.6s (~5.5 min) per single run** on CPU with the tuned `iterations=3`.
- Seed-averaging even a minimal 3 seeds × 4 SNR conditions (clean, +10, 0, −6 dB) with the real neural pipeline would be on the order of **60–70+ minutes minimum** of continuous compute — not practical to run and babysit live in this session, and GPU is still unavailable on this account (quota exhausted, same as Day 3).
- **Decision: did not run it.** The existing single-run SNR figure (already honestly caveated on-chart: "no fixed seed, one draw") stands as-is. If seed-averaging is wanted before Monday's mentor review, it needs either (a) a dedicated block of unattended time (kick it off and let it run in the background rather than live in a Cowork session), or (b) GPU quota becoming available (try again later, or on a different account that has quota).

### Housekeeping note

While reading the notebook's code, two accidental edits were made and caught/reverted before running anything: stray text typed into a code cell from a misdirected keystroke (undone), and an empty text cell that got inserted (deleted). Mentioning this so a future session isn't confused by notebook version history showing brief unrelated-looking edits around 2026-07-02.

**Net effect on what can be cited:** the cross-correlation F=44.4% (ME1+ME2 combined) is now doubly confirmed (Day 1 and today, both fresh runs). The SNR-vs-F curve's methodology is now documented precisely. The hero shot is now honest about what it shows. Nothing about the noise-robustness *mechanism* (still doesn't exist — that's Phase 2, per the audit) changed today; this was purely closing correctness/honesty gaps in existing Day 3 work, per the roadmap's own instruction not to let this pull into Phase 2 scope before Jul 8.

---

## Day 4 (continued) — 2026-07-02, later same day — Real prototypical hero shot finished + seed-averaged SNR sweep completed (GPU)

**Goal:** close the two items Day 4 left open above — an honest prototypical-network hero shot, and seed-averaging the F-vs-SNR figure — following Akshath's "let's follow the roadmap, start coding" instruction to push through both today.

### Session reliability note (read this before trusting timestamps above)

Partway through, the Colab runtime silently disconnected — "Runtime disconnected due to inactivity or reaching its maximum duration" — after the account's free daily compute allowance ran out (confirmed via Colab's own Resources panel: "zero compute units available"). Real time elapsed across the gap was roughly two hours, though almost none of that was active work. On reconnect, a `torch.cuda.is_available()` check unexpectedly returned `True` — the fresh session landed on a real T4 GPU (quota reset), not the CPU-only runtime used all week. Everything below this point ran on GPU, which is why the timings are dramatically faster than Day 3/4's CPU numbers.

One near-miss worth recording: after reconnecting, a reflexive click on "Run all" started re-running the entire notebook from scratch, including the ~10+ minute prototypical feature-extraction loop that had *just* been interrupted by the disconnect. Caught and interrupted it before it wasted the GPU session re-doing completed work — lesson: after any reconnect, check what actually needs re-running before mashing "Run all."

### 1. Real prototypical-network hero shot — done

Re-ran Section 5 (cross-correlation, unchanged) through the Stretch/prototypical section on the fresh GPU runtime. `evaluate_prototypes()` at the official default `iterations=10` (not the speed-tuned `iterations=3`) processed 491+450 segments in **under 2 minutes total** for both ME1.wav and ME2.wav combined — for comparison, Day 3's CPU timing for one call alone was 331.6s. GPU speedup here was roughly 60–80x on the inner iteration loop (torch tqdm bars went from ~4.5 it/s on CPU to ~260–390 it/s on GPU).

Official evaluator, fresh run, `ARGUS_stretch` team name, `stretch_output.csv`, ME1+ME2 combined:
- `ME1.wav {TP:8, FP:27, FN:3}`, `ME2.wav {TP:28, FP:13, FN:13}`
- **precision = 0.4737, recall = 0.6923, F = 56.25%**

This is the first time a real (not hardcoded) prototypical F-score has backed this figure. Re-pointed the hero-shot plotting cell at `stretch_output.csv` (was `output.csv`) and corrected the title to *"ARGUS Track A — Prototypical Network Baseline Detection on ME1.wav (F=56.2%, official evaluator, ME1+ME2 combined)"*. Regenerated figure shows a visibly denser set of detections than the cross-correlation baseline's figure, consistent with the stronger method. Re-running the identical pipeline from a second fresh runtime (after the disconnect) reproduced the same 56.25% — good sign the pipeline is behaving deterministically apart from the known unseeded negative-sampling variance.

**Note for the record:** this 56.25% is not directly comparable to the SNR sweep's "clean" single-clip numbers below (~28–31%) — different eval splits/scoring scope (ME1+ME2 combined vs. ME1-alone with a different feature/query split), as already flagged and reconciled earlier this week. Don't quote one number as if it supersedes the other.

`hero_shot_ME1.png` re-saved to `/content/hero_shot_ME1.png` and downloaded via the notebook's `files.download()` cell — **still needs to be manually moved from Akshath's Downloads folder into `screenshots/`** to replace the old (now doubly-outdated) file, same manual step flagged in the original Day 4 entry.

### 2. Seed-averaged F-vs-SNR sweep — done, real numbers with error bars

Added new cells (did not modify the existing single-run cells, which are kept as-is for comparison) that call the existing `run_snr_eval()` function **5 times per condition** (clean + SNR +10/0/−6 dB — the roadmap's own suggested "3–5 seeds"), collecting the F-score each time and computing mean ± std. This was the item Day 4 explicitly could not do on CPU (estimated 60–70+ min); on GPU the full 20-call sweep (5 seeds × 4 conditions) took **under 15 minutes**.

Real results (ME1.wav, N=5 runs per point, official evaluator each time):

| Condition | Individual F-scores (%) | Mean | Std dev |
|---|---|---|---|
| Clean (no added noise) | 28.07, 39.13, 24.24, 28.57, 34.48 | 30.90% | 5.26 |
| SNR = +10 dB | 32.26, 40.82, 28.07, 33.33, 34.78 | 33.85% | 4.14 |
| SNR = 0 dB | 21.74, 24.62, 30.77, 25.35, 30.00 | 26.49% | 3.41 |
| SNR = −6 dB | 6.90, 8.99, 17.39, 17.14, 8.60 | 11.80% | 4.52 |

This confirms two things the roadmap flagged as open questions: (1) the expected trend holds — F-score drops as noise increases, roughly 34% → 26% → 12% from +10 dB down to −6 dB; (2) the single-run swing the roadmap warned about is real and now quantified — std dev of 3.4–5.3 F-points per point, i.e. a single unseeded run can land anywhere in roughly a ±1 std band purely from negative-sampling luck, matching Day 3's own observed ±10-point swings.

New figure `fscore_vs_snr_seedavg.png` — same style as the original single-run chart plus error bars (±1 std) on each noisy point and a shaded band around the clean-baseline reference line — saved and downloaded the same way; also needs manual move into `screenshots/`. The original single-run `fscore_vs_snr.png` was also re-generated and re-downloaded today as a freshness check (still shows the same single-draw-noise caveat on-chart; kept as a "why we did this" comparison artifact, not the number to quote).

### Housekeeping note

Two more small Colab-editing mistakes happened and were caught before they mattered: a keyboard shortcut (`b`, "insert cell below") landed as literal text at the start of a download cell, causing a `SyntaxError` on first run — fixed and re-run cleanly. Also hit the same "browser zoom shortcut typed literal characters into a cell" issue as a minor annoyance (`ctrl+-` doesn't zoom out in this control environment) — worked around by scrolling rather than zooming for the rest of the session.

**Net effect on what can be cited:** both of Day 4's "still open" items are now closed with real, verified numbers. The prototypical hero shot shows a genuine model output for the first time. The SNR-vs-F relationship now has honest uncertainty bars instead of a single unseeded draw. GPU access (free-tier T4) turned out to be available after a runtime cycle — worth trying "disconnect and reconnect" early in a session if CPU-only was assumed, rather than treating GPU quota exhaustion as fixed for the day.

---

## Day 4 (continued, part 3) — 2026-07-02 — Local demo app live; Task 7 (reliability rehearsal) closed with real numbers

Custom local demo app (`argus_app/`, FastAPI + custom HTML/CSS/JS, replacing Gradio — built via Claude Code this week) got its environment finished and its data populated today, and the actual Task 7 reliability rehearsal ran for real.

**Bugs hit and fixed along the way (kept honest, not swept under the rug):**
- Windows `cd D:\...` from a C:-drive shell silently doesn't switch drives (classic cmd gotcha) — looked like a missing-file error, wasn't. Fixed with `cd /d`.
- `fetch_data.py` (written to pull just the needed Zenodo slices via `remotezip` instead of the full multi-GB zip) crashed with `PermissionError` on its first real run — its file-selection filter matched a bare zip directory entry (`Development_Set/Training_Set/MT/`, trailing slash, not a real file) and tried `shutil.copy2()` on it. Also would have pulled in macOS `__MACOSX/._*` resource-fork junk from the zip. Fixed by adding an `is_real_file()` filter excluding trailing-slash entries and `__MACOSX/`/`._`-prefixed paths before either member list is built.

**Task 7 — real results, all 10 runs on CPU, local Windows machine, no GPU:**

`setup()` (one-time training-set feature extraction): 41.5s.

ME1.wav — 5 repeated `detect()` calls, `iterations=3`, `query_batch_size=32` (same tuned config as the notebook):
| run | detections | time | F (indicative) |
|---|---|---|---|
| 1 | 43 | 79.40s | 40.7% |
| 2 | 44 | 78.15s | 40.0% |
| 3 | 58 | 81.04s | 31.9% |
| 4 | 57 | 77.11s | 32.4% |
| 5 | 50 | 78.67s | 36.1% |

Avg time 78.87s (range 77.11–81.04s). Detection count range 43–58 (spread 15). F-score range 31.9–40.7% (spread 8.8 points) — consistent with the unseeded-negative-resampling variance already documented this week.

ME2.wav — 5 repeated `detect()` calls:
| run | detections | time | F (indicative) |
|---|---|---|---|
| 1 | 45 | 65.13s | 93.0% |
| 2 | 43 | 65.94s | 92.9% |
| 3 | 43 | 74.23s | 92.9% |
| 4 | 47 | 78.24s | 90.9% |
| 5 | 43 | 81.24s | 92.9% |

Avg time 72.96s (range 65.13–81.24s). Detection count range 43–47 (spread 4, much tighter than ME1). F-score range 90.9–93.0% (spread 2.1 points).

**Reliability verdict: 10/10 `detect()` calls completed with zero crashes**, across two different clips, back to back. No slowdown or memory-leak trend across repeated calls — timings stayed in a stable band rather than climbing run over run. This is a clean pass on Task 7's actual question (does the app hold up under repeated live use).

**Honest discrepancy — not glossed over:** Day 1's notebook log recorded `detect()` at 13.15s/clip on Colab CPU with this identical config. Real local timing is 65–81s/clip, roughly 5-6x slower. `argus_core.py`'s `detect()` is an unmodified port of the notebook function, so this isn't a logic bug — most likely explanation is weaker/contended local CPU (Task Manager showed Brave, Discord, WhatsApp, and BlueStacks all running during the test) rather than Colab's allocated CPU. Not independently isolated which factor dominates. **Practical takeaway: budget ~70-80s per live detection when demoing or narrating, not the old 13s figure.**

**Scoring caveat:** the F-scores above are the app's own "indicative interval-overlap" metric (explicitly labelled as such in the UI, per §4 of `ARGUS_TrackA_Project_Status_2026-07-02.md`), not the official DCASE evaluator that produced the 56.25% hero-shot number. ME2's ~93% here reflects a looser overlap-matching method on a single clip, not a validated official score — the two numbers must not be conflated in submission materials.

**Decision: Task 7 is closed.** The app is reliable enough for a live or recorded demo. GPU (DirectML, since the machine is AMD) was considered as a way to cut the ~70-80s/detection time down, but deliberately not attempted today — deadline is too close to risk destabilizing a now-working, carefully-pinned environment for a speed win that isn't blocking anything. Left as optional future polish only if there's real spare time before the 8th.

---

## Day 5 — 2026-07-04 — DirectML/Fable5 status check + first noise-robustness mechanism prototype (code written, NOT yet executed)

**Context:** picking Track A back up in a new Cowork session per
`ARGUS_TrackA_Project_Status_2026-07-02.md`. Two things §5b of that file left
as "handed off, outcome unknown" were asked about directly rather than assumed:

- **DirectML GPU (AMD RX 7600): confirmed working, kept.** Akshath confirmed
  directly. Verified independently by reading the actual code/docs (not by
  running anything): `argus_core._select_device()` now tries CUDA then
  DirectML behind an `ARGUS_DEVICE` env var (default still `cpu` — the
  validated path is unchanged unless GPU is explicitly requested), and
  `README.md` documents measured numbers on the real RX 7600: **CPU ~69–84s/
  detection vs DirectML ~14–19s/detection (~5x faster)**, detection counts/F
  in the same ballpark but not bit-identical (different float kernels, on top
  of the existing unseeded-negative-sampling variance). `torch-directml==
  0.2.1.dev240521` is installed, pinning `torch==2.2.1`; a
  `requirements.lock.cpu-verified.txt` snapshot and an explicit uninstall/
  reinstall command are documented for reverting. Treat these numbers as
  "documented by whoever ran the DirectML build," not "independently
  re-measured this session" — nobody has re-run `gpu_bench.py` today.
- **Fable 5 full-project audit: run, but the output isn't anywhere in this
  folder.** Akshath confirmed he ran it and has results. Searched the whole
  `D:\TARANG - Copy` tree for anything resembling a saved audit output —
  found only the prompt file (`ARGUS_Fable5_FullAudit_Prompt.md`), no
  response/transcript. **Still open:** ask Akshath to paste or save the
  actual Fable 5 output somewhere in this folder if he wants its findings
  reflected in the PRD/notebook — nothing from it is incorporated below
  because it was never located.

**Main work this session: first prototype of a PRD §4.3 noise-robustness
mechanism, plus a real ablation script to test it — NOT YET RUN.**

Per Akshath's instruction to mine the PRD for post-regionals (Phase 2)
material that's still achievable in the ~2 days before Tuesday: the ISEF
audit's own #1-ranked action item (after pulling real DCASE data, which is
not feasible in 2 days) is *"Build ONE §4.3 noise-robustness mechanism... and
produce the F-vs-SNR ablation curve (with vs. without the mechanism)"* —
called out as "the single decisive issue" / "the ceiling" on the whole
project, since zero rungs of the method ladder above Baseline B existed.
That's the item pulled forward here, scoped honestly for what 2 days can
actually support:

- **`argus_core.spectral_subtract_denoise()`** (new) — a blind spectral-
  subtraction denoiser: per-frequency-bin noise floor estimated as a low
  percentile of that bin's magnitude across the whole clip (no external/known
  noise reference — has to be blind, or the ablation would be circular),
  subtract an over-subtracted version, keep original phase, floor to avoid
  musical-noise artifacts. This is explicitly a first, simple DSP prototype of
  the mechanism the PRD scopes for Phase 2 (a co-trained/learned denoiser),
  not that mechanism itself.
- **`detect(..., denoise=False)`** (new param, default off) — when `True`,
  runs the denoiser on the clip's isolated scratch copy only, before feature
  extraction. Original file untouched; default unchanged, so every existing
  validated number (13.15s Colab GPU, 70–85s local CPU, 14–19s DirectML, all
  F-scores reported to date) is unaffected unless a caller opts in.
- **`snr_ablation.py`** (new script, mirrors `gpu_bench.py`'s style) — ports
  the exact `make_snr_wav()` methodology already traced out of the Colab
  notebook (Day 4 entry above: a 150–210s quiet segment of the same clip,
  tiled, added back at a target whole-clip RMS SNR, fully deterministic) to
  build clean/+10/0/−6 dB test clips, then runs `detect()` N times per
  condition with `denoise=False` and `denoise=True` and reports mean±std F,
  same seed-averaging logic as the existing SNR sweep (needed because
  `evaluate_prototypes()`'s negative sampling is unseeded — a single run per
  cell would not be a fair comparison).
- **App wiring:** `/api/detect` takes a `denoise` form field (default off),
  returned back in the JSON; the frontend's Advanced panel got a "Noise-
  robustness front end (experimental)" checkbox, and the result line now
  shows `denoiser: ON` when used.
- **Fixed in passing:** `static/index.html`'s run-note still said "~13
  seconds" (the original Colab GPU figure) as the CPU expectation, which
  contradicts the app's own verified README numbers (70–85s CPU). Corrected
  to "roughly 70–85s CPU / ~15–20s GPU."

**Honesty flag — this is the important part.** None of the denoiser code above
has been executed or verified this session. Two reasons: (1) this Cowork
session's sandboxed shell reported "Workspace unavailable... VM service not
running" for the entire session, so no command could be run at all; (2) even
when that sandbox is up, it's an isolated Linux environment, not the real
Windows machine + pinned `.venv` this app actually runs in — so meaningful
execution has to happen on Akshath's machine (or a local Claude Code session
there), same handoff pattern already used for the DirectML and demo-app work.
**No F-score number for the denoiser exists anywhere yet.** Do not repeat "the
denoiser helps" or "the denoiser hurts" until `snr_ablation.py` has actually
been run and the real output — including if it makes things worse, which is a
legitimate and useful result — is logged here.

**Next step (for Akshath or a local Claude Code session, not done here):**

```bash
cd argus_app
.venv\Scripts\python.exe snr_ablation.py --clip ME1 --seeds 5
# or, faster — the verified DirectML path:
$env:ARGUS_DEVICE = "gpu"; .venv\Scripts\python.exe snr_ablation.py --clip ME1 --seeds 5
```

CPU estimate: ~2.5–3h for the full 4-condition grid; DirectML estimate:
~30–40min. Run it unattended, then append the real numbers as a follow-up
entry here — positive, negative, or inconclusive, whichever it actually is.

**Unchanged from §5b of the status doc, still open, not touched this
session (Track A scope only):** verifying which `screenshots/` PNGs are
current, pinging Vlkram on Track B, and all of Track C (narration, demo
recording, video edit, portal submission).

---

## Day 6 — 2026-07-05 — Learned front-end mechanism (PRD §4.3(c)): code written and reasoned through end-to-end; NOT yet executed on real ME1 audio

**Context:** picked up via a Cowork session using `ARGUS_NoiseRobustness_3Day_Prompt.md`
(the 3-day plan pulled forward from the ISEF audit's #1 action item). Goal: a
second, independent, genuinely trainable noise-robustness mechanism — separate
from Day 5's still-unrun blind spectral-subtraction denoiser — scoped to PRD
§4.3(c) ("learned matched-filter / spectro-temporal receptive fields").

**Infrastructure constraint hit immediately, worth recording:** this session's
sandboxed shell has no access to `D:\TARANG - Copy` at all (unlike the Day 5
session, where the shell was simply "unavailable"; here it's a genuinely
separate Linux machine, reachable only through the file-read/write tools) —
and plain PyPI `torch` on Linux pulls several GB of bundled CUDA/nvidia
packages that stalled indefinitely through this environment's network
allowlist (`download.pytorch.org`, the CPU-only wheel index, is blocked
outright; Zenodo, HuggingFace, Kaggle, and archive.org are also blocked, so
the real ME1/ME2 audio and training slice couldn't be fetched independently
either). Net effect: **no code below has been executed against real audio
this session.** This is the same category of handoff already logged on Day 4
and Day 5 (Colab GPU quota, Windows-drive quirks, "VM service not running") —
meaningful execution needs Akshath's actual machine + pinned `.venv`, same as
every other real number in this notebook.

**What was still done, and how it was verified without execution:**
- Read the real source directly (not guessed): `Model.py` (ProtoNet/ResNet
  both take `x` shaped `(N, seq_len, mel_bins)`), `Datagenerator.py` /
  `Feature_extract.py` (confirms `feat_pos`/`feat_neg`/`feat_query` are what
  `Datagen_test.generate_eval()` loads and normalizes — exactly what
  `evaluate_prototypes()` already uses), and `util.py` in full.
- Confirmed via a fresh `git clone` of the public
  `c4dm/dcase-few-shot-bioacoustic` repo (network-reachable even though
  Zenodo isn't) that `best_model.pth` — the frozen checkpoint this whole
  project is built on — is committed directly in the repo, not
  Zenodo-only. Useful to know for any future session that also lacks direct
  machine access.
- Every new/edited Python file was written directly into `argus_app/` via the
  file tools (real edits on the real project, not a copy), then verified with
  `python3 -m py_compile` (syntax-only, doesn't need torch) — all six files
  (see below) compile cleanly. This is NOT the same as running them; it only
  rules out typos/syntax errors, not logic bugs that only show up at runtime.
- Traced BatchNorm/Dropout behavior through `ResNet`/`BasicBlock` by hand to
  confirm the frozen backbone is fully deterministic whenever `.eval()` is
  set (which `FrontendWrappedEncoder` enforces permanently) — this is what
  makes the Day 1 zero-init identity check meaningful rather than "probably
  fine."

**What was built (six files, all under `argus_app/` unless noted):**
1. `noise_frontend.py` (new) — `LearnableSpectroTemporalFrontend`: a
   per-mel-bin learnable gain + one small depthwise spectro-temporal Conv2d +
   a **zero-init residual mix (`alpha=0` at construction)**, so a freshly
   built, untrained instance is the *exact identity function* regardless of
   the conv's own small random init. `FrontendWrappedEncoder` combines it
   with the frozen backbone, freezing the backbone's parameters
   (`requires_grad=False`) and forcing it to stay in `.eval()` even while the
   front end is in `.train()` mode.
2. `vendor/.../deep_learning/util.py` (edited, minimal/additive) — extracted
   the inline "build ResNet/ProtoNet, load `best_model.pth`" block into a new
   `build_frozen_encoder(conf, device)` helper, and added one new, **default-
   `None`** parameter to `evaluate_prototypes()`: `encoder_override`. When
   `None` (every existing call site — the notebook, `snr_ablation.py`,
   `detect()` without `frontend_ckpt`), behavior is byte-for-byte identical
   to before. This is the only change to the "official evaluator," and it's
   the same additive/off-by-default pattern this project already used for
   `denoise` and the `iterations`/`query_batch_size` knobs.
3. `argus_core.py` (edited) — extracted the "isolate clip → `feature_transform
   (mode='eval')` → open the `.h5`" block (previously inline in `detect()`)
   into `build_eval_features()`, reused by both `detect()` and
   `train_frontend.py`. Added `detect(..., frontend_ckpt=None)`: when set,
   loads a trained front end, wraps the same frozen backbone with it, and
   passes the wrapped module in as `encoder_override`. `denoise` and
   `frontend_ckpt` are independent and not meant to be combined.
4. `train_frontend.py` (new) — Day 2's training script. Builds the 0dB
   condition clip via `snr_ablation.build_condition_clip()` (reused, not
   rebuilt), loads `feat_pos`/`feat_neg` via the DCASE repo's own
   `Datagen_test.generate_eval()` (same call `evaluate_prototypes()` makes —
   no new data-loading logic), then trains only the front end's parameters
   with a differentiable rewrite of `util.get_probability()`'s exact
   softmax-over-negative-euclidean-distance formula (that function itself is
   unmodified and still used, unmodified, at scoring time — it can't be
   reused verbatim for backprop because it deliberately detaches to a plain
   Python list). Binary cross-entropy pushes held-out positive queries toward
   the positive prototype and held-out negative queries toward the negative
   prototype, both built from the same per-step resampled pool
   `evaluate_prototypes()` draws from.
5. `day1_sanity_check.py` (new) — the actual Day-1 plumbing check the prompt
   asked for. Builds the 0dB clip, saves a fresh untrained (identity) front
   end, then calls `detect()` twice with `torch.manual_seed()` pinned
   identically right before each call — once with `frontend_ckpt=None`, once
   with the untrained checkpoint — and asserts the detections and metrics
   are exactly equal. Prints PASS/FAIL. This is a stronger check than "it
   didn't crash": an identity module MUST reproduce the baseline exactly
   given the same negative-sampling draw, so a PASS is real evidence the new
   code path is wired correctly, not just that it ran.
6. `snr_ablation_frontend.py` (new) — Day 2's comparison sweep. Same
   structure as the existing (still-unrun) `snr_ablation.py`, with
   `denoise=True/False` swapped for `frontend_ckpt=<path>/None`, run in the
   same script invocation across clean/+10/0/−6 dB × N seeds so the
   "without" and "with" columns are a matched, same-session comparison.

**Honest status, plainly:** this is real, reviewed code grounded in the
actual DCASE source (not a guess at what the interfaces look like), and it
compiles — but **zero lines of it have touched real audio yet, and no
F-score number exists for this mechanism.** Do not cite "the learned front
end helps" or "hurts" from anything in this entry.

**Next step (for Akshath or a local Claude Code session, same handoff
pattern as Day 5's denoiser — run in order, each depends on the last):**

```bash
cd argus_app
.venv\Scripts\python.exe day1_sanity_check.py --clip ME1
# expect: "PASS" -- if FAIL, stop and debug noise_frontend.py / detect()'s
# encoder_override wiring before touching training.

.venv\Scripts\python.exe train_frontend.py --clip ME1 --condition 0dB --steps 300
# watch the printed loss: first10-avg vs last10-avg should show a real
# decrease. If it doesn't, say so honestly in the next entry rather than
# running the sweep on an unconverged checkpoint.

.venv\Scripts\python.exe snr_ablation_frontend.py --clip ME1 --seeds 5 --frontend_ckpt data\frontend_ME1.pt
# or, faster: ARGUS_DEVICE=gpu prefixed on all three, same DirectML path
# already verified for snr_ablation.py.
```

CPU estimate for the sweep: same ballpark as `snr_ablation.py`, ~2.5–3h for
the full grid; DirectML: ~30–40min. Append the real PASS/FAIL and the real
with-vs-without table as a follow-up dated entry — positive, negative, or
inconclusive, whichever it actually is. That entry is Day 3 of the prompt's
plan (the honest write-up + Phase 2 decision) and cannot be written honestly
until these three commands have actually produced output.

---

## Day 7 — 2026-07-05 — Learned front-end mechanism: real with-vs-without numbers in (Day 3 of the 3-day plan)

**What ran:** Akshath ran all three scripts from the Day 6 entry on his own
machine. `snr_ablation_frontend.py --clip ME1 --seeds 5 --frontend_ckpt
data\frontend_ME1.pt` produced the first real numbers for this mechanism.
(Day 1 sanity-check PASS/FAIL and the training loss curve from
`train_frontend.py` were not pasted back this session — flagged as an open
follow-up below, since they matter for diagnosing *why* the result below
looks the way it does.)

**Real result, N=5 seeds per cell, official evaluator via `detect()`:**

| Condition | Without mechanism | With mechanism | Delta |
|---|---|---|---|
| Clean | 24.3 ± 5.4 | 26.6 ± 2.5 | +2.3 |
| +10 dB | 23.5 ± 3.6 | 25.2 ± 3.9 | +1.7 |
| 0 dB | 19.2 ± 1.6 | 22.7 ± 3.9 | +3.5 |
| −6 dB | 14.0 ± 1.2 | 13.3 ± 2.1 | **−0.7** |

**Sanity check against this week's existing baseline table (Day 4 GPU sweep,
same clip: clean 30.90±5.26, +10dB 33.85±4.14, 0dB 26.49±3.41, −6dB
11.80±4.52):** this run's own "without mechanism" column reads noticeably
*lower* at three of four conditions (clean −6.6, +10dB −10.35, 0dB −7.3) but
slightly *higher* at −6dB (+2.2). Flagging this plainly rather than ignoring
it, per the script's own printed reminder. Two candidate explanations: (a)
plain N=5 seed-to-seed noise — the direction isn't consistent (three down,
one up), which is what you'd expect from noise rather than a systematic
environment/device difference (a real device/precision shift would more
likely push every condition the same way); (b) a genuine environment
difference (this run's device — CPU vs the verified DirectML GPU path — and
exact settings weren't confirmed back to this notebook entry). **Open item:**
confirm which device this ran on and whether `iterations`/`query_batch_size`
matched the Day 1 defaults (3/32) before citing the "without mechanism"
column here as reproducing Day 4's numbers. This does NOT invalidate the
with-vs-without comparison itself, though — both columns came from the same
script run, same device, same settings, so that comparison is still a fair,
matched one regardless of how it lines up with Day 4's separately-run table.

**Is any of these four deltas distinguishable from noise?** Using each
condition's own standard error of the difference (sqrt(std_without² +
std_with²) / sqrt(N), N=5) as a rough (not the full Phase-2 matched-pair
bootstrap) gauge:

| Condition | Delta | SE of the difference | Delta / SE |
|---|---|---|---|
| Clean | +2.3 | 2.66 | 0.86 |
| +10 dB | +1.7 | 2.37 | 0.72 |
| 0 dB | +3.5 | 1.89 | **1.85** |
| −6 dB | −0.7 | 1.08 | −0.65 |

Only 0 dB gets close to conventional significance (~2 SE), and it doesn't
clear it. Clean and +10dB are clearly within noise. **−6dB — the condition
the whole mechanism was supposed to help most, per the PRD and the 3-day
prompt — is not just "within noise," it's slightly negative.**

**Honest verdict:** this is not a demonstrated noise-robustness gain. The
pattern is: a mild, not-quite-significant positive signal at 0dB; noise-level
positives at clean/+10dB; a small negative at −6dB. If the mechanism worked
the way it's meant to, the biggest, most consistent gain should show up at
the noisiest condition — instead that's the one place it went the wrong way
(though also within noise, at N=5). Reporting this straight, in the language
the prompt itself asked for: **"we prototyped a first mechanism; the early
signal is inconclusive, with a suggestive-but-not-significant gain at 0dB and
no gain (a small negative) at −6dB where it mattered most; a properly powered
version of this comparison (≥10 seeds, matched-pair bootstrap, per PRD §7) is
Phase 2 work."** This is not "we beat the baseline," and it should not be
written up or pitched as one.

**Why this might have happened (candidate causes, for Phase 2 triage, not
yet confirmed):**
1. **Data-starved training.** `train_frontend.py` trains on a single clip's
   own positive pool (ME1.wav has on the order of 5-16 support segments per
   the Day 3 notebook entry) — a handful of positive examples is thin for a
   metric-learning objective, even a small one. If the training loss barely
   moved (not yet confirmed — see open item below), the front end may have
   learned close to nothing, in which case a near-zero/noisy delta is
   exactly what you'd expect, not evidence the *mechanism idea* itself is
   wrong.
2. **Trained on 0dB specifically, evaluated on −6dB too** — the front end
   was never shown −6dB-level noise during training (by design, per the Day
   6 entry, to keep training fast and avoid overfitting to one specific
   noise draw), so it's plausible it doesn't generalize to a noisier regime
   than it trained on. Worth testing training directly on −6dB as a Phase 2
   variant if this mechanism is kept.
3. Plain N=5 noise, full stop — genuinely possible given every other number
   this project has produced this week has shown ±3-10 point swings at this
   sample size.

**Open follow-ups before a real go/no-go decision (not done this session):**
- Get the Day 1 `day1_sanity_check.py` PASS/FAIL and the `train_frontend.py`
  loss log (first10-avg vs last10-avg) back into this notebook — these
  determine whether "training barely worked" or "training worked fine but
  the mechanism doesn't help" is the right diagnosis, and those two cases
  point to very different Phase 2 next steps.
- Confirm the device/settings this sweep ran on, to reconcile the "without
  mechanism" column against Day 4's table (see above).

**Recommendation, given what's known right now:** do not carry this forward
as "our noise-robustness mechanism works" in any pitch or write-up. It is a
legitimate, honestly-reported first attempt per the 3-day prompt's own
instructions — plumbing built and (per Day 6) reasoned through, trained once,
compared once, real numbers logged. The correct framing for now is "prototyped,
inconclusive, Phase 2 needed" — either to re-run with more seeds and confirm
training actually converged, or to try training with more/noisier data before
concluding this specific architecture doesn't help.

---

## Day 8 — 2026-07-05 — Day 1 FAIL root-caused and fixed; training-loop improved on real evidence

**What happened:** the real PowerShell run (all three scripts queued together)
came back with Day 1's sanity check printing **FAIL** — an untrained,
identity-init front end changed `detect()`'s output vs. no front end at all,
which per the script's own logic means something in the wiring was broken.
Root-caused and fixed this session; also used the real training log from the
same run to make two evidence-based improvements to `train_frontend.py`.
Nothing below has been re-run yet — same handoff constraint as every other
entry this week.

### 1. Root cause of the Day 1 FAIL

`LearnableSpectroTemporalFrontend.__init__` initialized its conv filter with
`nn.init.normal_(self.filt.weight, ...)`, which draws from **torch's global
RNG**. `day1_sanity_check.py` pins `torch.manual_seed(seed)` immediately
before each of its two `detect()` calls specifically so the only difference
between them is whether the front end is engaged — but Run B's call path
(`frontend_ckpt` set) *constructs* a fresh front end **after** that seed reset
(inside `argus_core.detect()`, via `noise_frontend.load_frontend()`), which
silently consumed random draws from the just-pinned global stream that Run
A's call (no front end at all) never consumed. That shifted
`evaluate_prototypes()`'s later, intentionally-unseeded negative-sampling
draw (`torch.randperm(len(X_neg))[:conf.eval.samples_neg]`) out of alignment
between the two runs. So the FAIL was real, but **it was a test-methodology
bug, not evidence the front end's own math is broken**: `alpha=0` still
mathematically zeroes the conv branch regardless of the conv's weight values
— constructing the module just happened to perturb *unrelated* downstream
randomness, which is what actually caused the mismatch.

**Fix:** `noise_frontend.py`'s `LearnableSpectroTemporalFrontend.__init__`
now draws its conv weight init from a **private, local `torch.Generator`**
(seeded independently, stored/restored as `init_seed` in the checkpoint),
never touching the global RNG. Added an explicit regression guard to
`day1_sanity_check.py` that captures `torch.get_rng_state()` before/after
constructing a front end and asserts it's unchanged, so this exact bug class
can't silently reappear and go unnoticed again.

**This does NOT invalidate the Day 2 with-vs-without numbers already logged
in Day 7** — `snr_ablation_frontend.py` never relied on seed-pinning between
its "without"/"with" columns (both are independent unseeded draws, same as
the project's existing `snr_ablation.py` convention), so that comparison's
fairness doesn't depend on the bug above. **Next step: re-run
`day1_sanity_check.py` after this fix and confirm it now prints PASS** —
still open, not done this session (same sandbox-can't-execute constraint as
every prior entry this week).

### 2. Training-loop improvements, motivated by the real log (not guessed)

The real `train_frontend.py` run showed two concrete problems worth fixing
before trying again:
- `Loaded features: X_pos=(5, 28, 128)` — only 5 positive segments on
  ME1/0dB, triggering the leave-one-out branch. Consistent with what Day 7's
  entry predicted as the likely reason for a near-zero/noisy training signal.
- The per-step loss **did not converge** — it swung between ~0.15 and ~1.22
  with no visible downward trend (step 60: 0.85, step 120: 1.22, step 180:
  0.15, step 200: 0.80, ...), consistent with a single-random-leave-one-out-
  sample-per-step gradient estimate being too high-variance for only 5 data
  points, possibly compounded by too high a learning rate.

**Two fixes made to `train_frontend.py`:**
1. **Full leave-one-out enumeration per step**, replacing "sample one random
   held-out fold." When `len(X_pos) <= n_shot` (the case actually hit), every
   step now builds all N leave-one-out folds (each fold's own prototype from
   the other N-1 positives, scored against its own held-out query) and
   averages their loss together, instead of using a random 1-of-N slice of
   the signal each step. Uses the full available positive data every step.
2. **Lower default learning rate (1e-3 -> 3e-4) + gradient-norm clipping
   (max_norm=1.0, new `--grad_clip` flag)** — directly targets the observed
   loss instability.

**Not implemented this session (documented here instead of shipped
untested):** pooling positive/negative pools across multiple preset clips
(e.g. `--clips ME1 ME2`) to get more raw training data than one clip's
5-16 support segments can offer. Investigated and deliberately deferred:
`Feature_extract.py` computes `seg_len` *adaptively per file* (based on that
file's own support-event durations — ME1 got `seg_len=28` this run; ME2 could
easily differ), so naively concatenating `X_pos`/`X_neg` tensors across clips
risks a shape mismatch that can't be verified without actually running it in
this environment. Real next step, not done: either resample/pad each clip's
segments to a common length before pooling, or run alternating per-clip
training episodes instead of pooling raw tensors. Flagging honestly rather
than shipping an untested guess.

### Next step (same handoff pattern, run in order on Akshath's machine)

```bash
cd argus_app
.venv\Scripts\python.exe day1_sanity_check.py --clip ME1
# expect PASS this time -- if still FAIL, the RNG-purity assert added above
# will point at exactly where it diverges.

.venv\Scripts\python.exe train_frontend.py --clip ME1 --condition 0dB --steps 300
# watch for the loss to actually trend down now, not just swing -- compare
# first10-avg vs last10-avg like before.

.venv\Scripts\python.exe snr_ablation_frontend.py --clip ME1 --seeds 5 --frontend_ckpt data\frontend_ME1.pt
```

Also worth trying, cheap (no code change, existing `--condition` flag):
`train_frontend.py --clip ME1 --condition -6dB --steps 300` — trains directly
in the noisiest regime, which is where Day 7's real numbers showed the
mechanism doing worst (the one place it's supposed to help most). Compare
whether training on -6dB directly changes that condition's delta.

**Honest framing for whoever picks this up:** the mechanism has not been
disproven — the one real training run so far had a data-starvation problem
that's now been partially addressed (full LOO enumeration, LR/clip fix).
Whether that's enough to produce a real, consistent gain is still an open
question, not a settled one either way.

---

## Day 9 — 2026-07-05 — Real fix for the Day 1 crash; second real Day 2 run shows the most promising signal yet at −6dB

**What happened:** Akshath re-ran the same three-command sequence after Day
8's fix. `day1_sanity_check.py` this time didn't print FAIL — it **crashed**
with the new RNG-purity guard's own `AssertionError` firing on the very first
front-end construction, before Run A/B even started. Root-caused and fixed
for real this session. Training and the sweep ran anyway (same PowerShell
queuing behavior as before), and produced the most interesting numbers yet.

### 1. Why Day 8's fix was incomplete

Day 8 replaced `nn.init.normal_(self.filt.weight, ...)` (global-RNG-draining)
with a local `torch.Generator`. That fixed the *explicit* re-init line, but
missed something more basic: **`nn.Conv2d(...)`'s own constructor** runs its
own default `reset_parameters()` (`kaiming_uniform_` + `uniform_`, both from
the *global* RNG) as a normal part of `__init__` — before any code afterward
gets a chance to overwrite those weights. So the mere act of writing
`nn.Conv2d(1, 1, kernel_size=..., ...)` already perturbs the caller's global
RNG state, regardless of what happens to the weights next. This is standard,
expected PyTorch behavior (every parameterized layer self-initializes on
construction) — the bug was not accounting for it, not a strange edge case.

**Real fix:** wrap the entire `LearnableSpectroTemporalFrontend.__init__`
body in `torch.random.fork_rng(devices=[])`, which saves the CPU global RNG
state on entry and restores it on exit no matter what happens inside —
`nn.Conv2d`'s own init, the explicit re-init, or anything else. This is the
standard PyTorch idiom for "run some code that touches the global RNG without
that leaking to the caller," and it's a complete fix rather than a partial
one (it doesn't matter what internally consumes randomness, it's all
contained). `devices=[]` is correct/sufficient here since the CPU RNG is
*always* forked regardless of that argument (it only controls which CUDA
device states also get saved — irrelevant, since this project uses CPU or
DirectML, never CUDA).

**Not independently re-verified this session** (same constraint as every
entry this week) — next step is still to actually re-run
`day1_sanity_check.py` and confirm PASS.

### 2. Real Day 2 numbers, run 2 (after the training-stability fixes from Day 8)

Training loss this time was visibly more stable than run 1 — swung between
~0.22 and ~0.48 (was ~0.15–1.22 in run 1), consistent with the lower LR +
gradient clipping doing what they were meant to. Net change was still small
(first10-avg 0.3531 -> last10-avg 0.3505), but the per-step P(pos|pos_query)
stayed around 0.75–0.89 and P(pos|neg_query) around 0.03–0.14 throughout —
a real, meaningful separation between classes existed from early on, which
may just mean the frozen backbone + near-identity front end was already
fairly good at this before much training happened, leaving less headroom for
the loss to visibly drop further. Worth remembering when interpreting "loss
barely moved" — it doesn't necessarily mean nothing is working.

**With-vs-without table, N=5 seeds, this run:**

| Condition | Without mechanism | With mechanism | Delta |
|---|---|---|---|
| Clean | 25.1 ± 6.8 | 29.1 ± 4.9 | +4.0 |
| +10 dB | 25.7 ± 3.5 | 25.2 ± 4.4 | −0.5 |
| 0 dB | 19.9 ± 3.6 | 20.8 ± 3.0 | +0.9 |
| −6 dB | 12.9 ± 1.4 | 18.5 ± 4.2 | **+5.6** |

Same informal delta/SE-of-difference gauge as Day 7's entry:

| Condition | Delta | SE of the difference | Delta / SE |
|---|---|---|---|
| Clean | +4.0 | 3.75 | 1.07 |
| +10 dB | −0.5 | 2.51 | −0.20 |
| 0 dB | +0.9 | 2.10 | 0.43 |
| −6 dB | **+5.6** | 1.98 | **2.83** |

**−6dB is the first condition across both real runs to informally clear a
~2-SE bar — and it's exactly the condition the mechanism is supposed to help
most.** Run 1 (before the training-stability fix) showed −6dB at −0.7
(slightly negative); run 2 (after full leave-one-out enumeration + lower
LR + gradient clipping) shows +5.6. That direction of change is consistent
with the diagnosis that run 1's training was too unstable/high-variance to
produce a useful front end, and fixing that instability is what let a real
signal show up. This is the most encouraging result yet, but it is still
**one run, N=5, not the ≥10-seed matched-pair bootstrap PRD §7 calls for** —
informally clearing ~2 SE at N=5 is suggestive, not confirmed.

**Baseline-vs-Day-4 mismatch, now seen twice:** this run's own "without
mechanism" column (25.1 / 25.7 / 19.9 / 12.9) is close to run 1's (24.3 /
23.5 / 19.2 / 14.0) — the two local runs agree with each other reasonably
well — but both consistently read ~6-8 points lower than Day 4's original
Colab-T4 table (30.90 / 33.85 / 26.49 / 11.80) at clean/+10dB/0dB, while
matching closely at −6dB. Seeing the same pattern twice, in the same
direction, on the same local DirectML setup makes "plain seed noise" less
likely and "a real, reproducible device/precision difference between this
local DirectML path and the original Colab T4 GPU" more likely — consistent
with the README's own caveat that DirectML uses different floating-point
kernels than CUDA/CPU. **Practical implication: don't compare a local run's
"without mechanism" column against the old Day 4 Colab table as if they were
interchangeable — always compare within the same script invocation (which
these ablation scripts already do by construction), not across environments.**

### Honest read, updated from Day 7's

Day 7's verdict ("inconclusive, slightly negative at −6dB") no longer holds
as the latest evidence — it was measured on a training run that has since
been shown to be unstable. The current, honest read: **a real, promising
signal at −6dB (the condition that matters most), not yet confirmed
(N=5, one run), with the other three conditions still within noise.** This
justifies continuing rather than abandoning the mechanism, but does not yet
justify claiming it works. Legitimate next-step framing: "an early result
suggests a real low-SNR gain; a properly powered replication (more seeds,
ideally the matched-pair bootstrap) is the next step before this can be
cited as a finding."

### Next steps, in priority order

1. Re-run `day1_sanity_check.py` and confirm it now prints PASS (still open).
2. Re-run `snr_ablation_frontend.py` a second time (fresh training +
   fresh sweep, i.e. redo all three commands once more) to see if the −6dB
   gain replicates — one run at N=5 could still be a lucky draw either way.
3. If it replicates, increase `--seeds` (e.g. 10) specifically at −6dB to
   tighten the estimate before treating this as a real finding.

---

## Day 10 — 2026-07-05 — Day 1 finally PASSES; the −6dB signal from Day 9 did NOT replicate; training still isn't clearly converging in any of 3 real attempts

**What happened:** Akshath re-ran the same three-command sequence a third
time. `day1_sanity_check.py` now genuinely **PASSES** — "Same detection
count: True", "Same detections: True", "Same metrics dict: True" —
confirming the `torch.random.fork_rng()` fix from Day 9 is a real, complete
fix. That question is now closed.

The other two scripts produced this pass's real numbers, and they change
last session's read.

### 1. Training: third attempt, third different (still inconclusive) outcome

| Run | first10-avg loss | last10-avg loss | delta |
|---|---|---|---|
| Run 1 (Day 7, unstable loop) | 0.3708 | 0.3637 | −0.0071 |
| Run 2 (Day 9, fixed loop) | 0.3531 | 0.3505 | −0.0026 |
| Run 3 (this run, fixed loop) | 0.3698 | 0.3883 | **+0.0185** |

This run's loss went *up* slightly, and `train_frontend.py`'s own built-in
check correctly fired: `WARNING: loss did not decrease over training...`.
Looking at all three real attempts together rather than one at a time: none
of them show a clear, meaningful loss decrease — all three deltas are small
and close to zero in both directions (−0.0071, −0.0026, +0.0185), which is
itself the more important finding than any single run's number. With a
fixed, stable training loop (runs 2 and 3 both have it) and still no
consistent decrease across two attempts, "unstable training" is no longer
a sufficient explanation — three tries in and this objective, on this much
data (5 positive segments), is not reliably moving the loss anywhere. The
front end is very likely staying close to its zero-init identity state in
all three checkpoints, just perturbed by different small amounts of noise
from each run's own random negative-sampling draws.

### 2. Real Day 2 numbers, run 3

| Condition | Without mechanism | With mechanism | Delta |
|---|---|---|---|
| Clean | 25.4 ± 4.8 | 23.7 ± 4.1 | −1.7 |
| +10 dB | 27.1 ± 3.4 | 25.5 ± 3.3 | −1.6 |
| 0 dB | 17.9 ± 2.7 | 20.5 ± 1.3 | +2.6 |
| −6 dB | 14.9 ± 1.4 | 14.1 ± 1.7 | −0.8 |

Same delta/SE-of-difference gauge as Days 7 and 9:

| Condition | Delta | SE of the difference | Delta / SE |
|---|---|---|---|
| Clean | −1.7 | 2.82 | −0.60 |
| +10 dB | −1.6 | 2.12 | −0.76 |
| 0 dB | +2.6 | 1.34 | **1.94** |
| −6 dB | −0.8 | 0.98 | −0.81 |

**Day 9's −6dB result (+5.6, delta/SE ≈ 2.83) did NOT replicate.** This run's
−6dB delta is back to small and negative (−0.8), matching run 1's direction
(−0.7) rather than run 2's. Flagged explicitly in Day 9's own "next steps" as
the thing to check — it's now been checked, and the honest answer is that it
did not hold up. Read as a single-run fluke (well within N=5 seed noise, not
a real effect) unless a future run shows it again.

**0 dB is the one condition that has now come out positive in all three real
runs** (+3.5, +0.9, +2.6 — Days 7/9/10 respectively), though it only reaches
the informal ~2-SE bar in runs 1 and 3, not run 2. This is a different,
smaller, less dramatic pattern than Day 9's −6dB excitement, but it's the
more *replicated* one of the two. One plausible explanation, given point 1
above: `train_frontend.py` trains on 0dB by default — if the front end is
only barely moving from identity, whatever tiny, consistent shift it does
pick up would plausibly show up most at the exact condition it saw during
training, and not necessarily generalize to other SNR levels. That would be
mild memorization of one condition, not a general noise-robustness gain —
worth designing the next experiment specifically to tell those two apart
(see below).

### Honest read, updated from Day 9's

Day 9's read ("a real, promising signal at −6dB, not yet confirmed")
does not survive this run. Updated honest verdict, all three runs
together: **no condition shows a delta that is both large and replicated.**
−6dB flip-flopped sign-and-magnitude across runs 1/2/3 (−0.7 / +5.6 / −0.8)
in a way that looks like noise around zero, not a real effect. 0dB is
positive in all three runs, modestly and only sometimes at the informal
~2-SE bar — a small, currently-unexplained-but-consistent pattern that is
worth a targeted follow-up (see below), not yet a claim. Clean and +10dB show
no consistent direction at all. Combined with point 1 (training not clearly
converging in any of 3 attempts), the most likely honest explanation for
everything observed so far is: **the front end has not been trained far
enough from its zero-init identity state, in any of the three checkpoints
produced this week, to produce a reliable measured effect at this SNR
sweep's sample size.** This is not the same as "the mechanism doesn't work"
— it has not really been tried yet in a way that would show whether it
works. Per the 3-day prompt's own instruction not to overclaim: the honest
Day 3 answer is **inconclusive, leaning toward "training hasn't converged
yet" rather than "no effect exists,"** and continuing to Phase 2 should be
framed as "fix training/get more data first," not "the mechanism is proven"
or "the mechanism has failed."

### Next steps, in priority order

1. **Give the front end more, and more varied, training signal** — the
   likeliest single fix given point 1. Two concrete levers, both scoped and
   built this session (see the 2026-07-05 (autonomous session) entry below):
   training across multiple SNR conditions of the same clip (not just 0dB),
   and pooling positive/negative segments across more than one preset clip.
2. **Test the "memorized 0dB, didn't generalize" hypothesis directly** —
   train on 0dB only (as done so far) vs. train on multiple conditions
   jointly, and compare whether the with-mechanism gain at 0dB shrinks while
   gains elsewhere grow. If 0dB's gain disappears once training also sees
   other conditions, that confirms memorization rather than a general effect.
3. Once training shows a real, clean, repeated loss decrease (not just a
   flat/noisy line), redo the N=5 sweep again before trusting any single
   condition's delta.
4. Still deferred: the real PRD §7 matched-pair bootstrap with ≥10 seeds.
   A cheaper, still-useful intermediate step (paired-seed sweep, same seed
   pinned for both the with/without call at each seed-index) was built this
   session too — see below.

---

## 2026-07-05 (autonomous session, Akshath AFK) — Three levers built to give Day 11 a better shot: multi-condition training, matched-pair sweep, multi-clip pooling, unattended runner

**Context:** written in the same session as the Day 10 analysis above, while
Akshath was away from keyboard, per his explicit instruction to keep making
real progress without stopping to ask permission at each step. Everything
below is new code, verified by manual review + `python -m py_compile` only
(this assistant's sandbox cannot run torch or touch the real venv/audio —
same standing constraint as every entry this week) — **none of it has been
run on real data yet.** Do not treat any of it as validated until Akshath (or
a future session) actually executes it and the output is pasted back in.

### 1. `train_frontend.py` — train across multiple SNR conditions, not just one

Motivated directly by Day 10's "did it just memorize 0dB" hypothesis. Added
`--conditions` (plural, e.g. `--conditions 0dB -6dB clean +10dB`), replacing
the old singular `--condition`. Features for every requested condition are
built once, up front, via the existing `snr_ablation.build_condition_clip()`
(reused, not rebuilt), then their positive/negative segments are pooled
(concatenated) into one larger set before training starts — so a 4-condition
run gives the same full-leave-one-out training loop 4x as many positive
segments to fold over every step (e.g. 20 instead of 5 for ME1), not just a
random one-condition-per-step draw. This is safe to
do within one clip because within-clip `seg_len` is fixed by the annotation
CSV's own event durations, which do not change when noise is mixed in at
different SNRs — confirmed from this week's own logs (`ME1`'s `seg_len=28`
appeared identically at clean/+10dB/0dB/−6dB in every run so far). So pooling
across conditions of the SAME clip carries none of the shape-mismatch risk
that pooling across DIFFERENT clips would (see item 2). `--condition`
(singular) still works as a one-item alias for backward compatibility with
this week's existing commands.

### 2. `train_frontend.py` — optional multi-clip positive/negative pooling

Added `--clips` (e.g. `--clips ME1 ME2`), previously flagged (Day 6 entry) as
deferred because different clips can have different `seg_len` (computed from
each clip's own annotation durations). Implemented via a simple, explicit
resize-to-common-length step: each clip's segments are center-cropped or
edge-padded (repeat last frame) to `max(seg_len across requested clips)`
frames before pooling into one tensor. This is a first, honest prototype of
that resize — not claimed to be the "right" way to reconcile different
segment lengths, just a documented, inspectable one. Off by default
(`--clips` defaults to just the `--clip` value), so every command run this
week still works unchanged.

### 3. `snr_ablation_frontend_paired.py` — new, matched-pair seeding

New script, structurally `snr_ablation_frontend.py`'s exact pattern
(reused CONDITIONS/build_condition_clip from `snr_ablation.py`, same as
always) with one change: at each seed index `i`, `torch.manual_seed(base_seed
+ i)` is called immediately before BOTH the without-mechanism and
with-mechanism `detect()` calls for that seed — the same trick
`day1_sanity_check.py` already uses to make exactly two calls comparable,
extended here across N real seeds instead of just one. This cancels out
`evaluate_prototypes()`'s shared unseeded-negative-sampling draw as a source
of noise between the two columns, which should make genuinely small deltas
easier to distinguish from seed noise at the same N=5 — directly in the
spirit of PRD §7's matched-pair recommendation, without building its full
bootstrap machinery. Reports both the usual per-condition means/stds AND the
paired per-seed deltas' own mean/std (a paired comparison's std is usually
much smaller than computing means separately then subtracting, when the
pairing is doing its job) — if it isn't smaller here, that itself is useful
information (means the negative-sampling draw isn't actually the dominant
noise source).

### 4. `run_experiment_suite.py` — unattended multi-config runner

New driver script for exactly this situation (Akshath stepping away for an
extended stretch): runs a small, fixed sequence of train+sweep configs back
to back with no user interaction, catching and logging (not crashing on) any
individual config's failure so one bad config doesn't stop the rest, and
writing one consolidated, timestamped results file
(`data/experiment_suite_<timestamp>.log` + a parsed summary table at the
end) instead of Akshath needing to babysit three separate commands per
config. Configured to run, in order: (a) the existing single-condition
(0dB) baseline again as a same-conditions replication check, (b)
multi-condition training (`--conditions 0dB -6dB`), (c) multi-condition
training across all four conditions, (d) the paired-seed sweep on
whichever checkpoint looks most promising from (a)-(c). Estimated runtime
on the verified DirectML path: roughly 45-70 minutes for all four configs
(each config's own train+sweep is in the same ballpark as this week's
existing ~15-20 min per full run) — sized to fit inside an AFK stretch
without needing to be split up further.

**Usage (from `argus_app/`, with the venv active):**
```
ARGUS_DEVICE=gpu .venv\Scripts\python.exe run_experiment_suite.py --clip ME1
```

**Honest caveat, stated up front rather than buried:** none of these four
files have been executed even once as of this entry. They are today's
equivalent of the Day 6 entry's "code written and reasoned through end-to-end,
NOT yet executed" milestone, not new validated results. The actual next step
is exactly what it's been every day this week — run them on the real venv and
paste the output back so it can be read honestly, same as the three runs
before this one.

---

## Day 11 — 2026-07-05 — Matched-pair sweep shows EXACTLY zero effect (config A); configs B/C blocked by a real argparse bug (now fixed)

**What happened:** Akshath ran `day1_sanity_check.py` (4th confirmed PASS)
then `run_experiment_suite.py --clip ME1` (the new unattended runner from the
last entry) and went AFK again. Config A (single-condition 0dB, same setup
as every prior real run) completed its train+paired-sweep. Configs B and C
(the new multi-condition pooling) both crashed immediately with exit code 2
— a real bug in code from the last session, root-caused and fixed below.

### 1. Config A's matched-pair sweep: exactly zero effect, every condition

| Condition | Without | With | Unpaired delta | Paired delta (mean±std) | Paired SE |
|---|---|---|---|---|---|
| Clean | 21.5 ± 1.7 | 21.5 ± 1.7 | +0.0 | +0.0 ± 0.0 | 0.00 |
| +10 dB | 22.9 ± 2.6 | 22.9 ± 2.6 | +0.0 | +0.0 ± 0.0 | 0.00 |
| 0 dB | 20.3 ± 1.5 | 20.3 ± 1.5 | +0.0 | +0.0 ± 0.0 | 0.00 |
| −6 dB | 15.2 ± 0.5 | 15.2 ± 0.5 | +0.0 | +0.0 ± 0.0 | 0.00 |

Not "small" — **bit-for-bit identical** F-scores (and per the script, identical
precision/recall/detections) between the without and with columns, at every
one of 5 paired seeds, at every condition. This is the matched-pair sweep
doing exactly its job: with the shared negative-sampling draw held fixed
between the two columns, the ONLY remaining source of any difference is the
trained front end itself — and it produced none, anywhere, at all.

Training's own loss log this run: first10-avg=0.3706, last10-avg=0.3895,
delta=+0.0189 — again didn't decrease (own `WARNING` fired). That's now
**4 for 4** real training attempts (Days 7/9/10, and this run) on the
single-clip/single-condition setup, with deltas −0.0071 / −0.0026 / +0.0185 /
+0.0189 — all close to zero, none showing real, repeated improvement. This is
no longer "one unlucky run" territory; it's a systematic property of training
5 positive segments on 0dB alone with this objective.

**This changes the read on Days 7/9/10 significantly.** Those with/without
deltas (−6dB swinging −0.7 / +5.6 / −0.8; 0dB positive all three times) came
from the *unpaired* sweep, where the without/with columns each draw their own
independent, unseeded negative sample. Given config A's checkpoint here is
essentially still the zero-init identity map (see below) and shows *exactly*
zero effect once that shared noise source is controlled for, the far more
likely explanation for every earlier swing is: **it was entirely
negative-sampling noise between two independently-sampled columns, not
anything the mechanism was doing.** The "0dB positive in 3/3 runs" pattern
that Day 10 flagged as the one replicated signal worth taking seriously —
this new evidence weakens that read considerably; it's more likely to have
been a coincidence of unpaired sampling than a real, if small, effect. Not
fully ruled out (all three of those runs used slightly different training
seeds/checkpoints than config A's), but the burden of proof just went up a
lot.

### 2. Root cause: alpha very likely never moved, and now there's a script to check it directly

Added `inspect_frontend_checkpoint.py` this session — loads a saved
checkpoint (no audio/backbone/setup() needed) and reports how far `alpha`
(the zero-init residual weight that gates whether the learned conv branch
affects the output *at all*) and `gain` have moved from their known init
values. Not yet run against `data/frontend_ME1.pt` — that's a natural next
step, cheap enough to run before any full sweep. Given the exact-zero
result above, the most likely finding is `alpha` sitting at (or extremely
close to) 0.0 — meaning the front end is still functioning as the identity
map regardless of what the conv filter itself learned, because `alpha` is
the single multiplier gating whether that filter matters at all.

**Working hypothesis for why alpha isn't moving** (flagged honestly as a
hypothesis, not verified): the frozen backbone already separates
pos/neg queries fairly well before any training (P(pos|pos_query) ~0.75–0.88,
P(pos|neg_query) ~0.03–0.13 from the start, per this and prior runs' logs) —
if the loss is already fairly low at alpha=0, the gradient signal pushing
alpha away from 0 may simply be very small. A concrete Phase-2-scoped idea
worth trying (not implemented — needs real experimentation to check, not
just code review): initialize `alpha` to a small nonzero value instead of
exactly 0, breaking the possible symmetry point, or increase `lr` for
`alpha` specifically (a per-parameter learning rate) while keeping `gain`/
`filt` at the current rate.

### 3. Bug found and fixed: `-6dB` broke `--conditions` (exit code 2)

Root cause: `train_frontend.py`'s `--conditions` was `nargs="+"` (space-
separated). `run_experiment_suite.py` passed it the tokens `0dB -6dB` (config
B) / `clean +10dB 0dB -6dB` (config C). Argparse classifies any token
starting with `-` that isn't a plain negative number as a *new option flag*
— `-6dB` isn't a plain negative number (trailing letters), so argparse tried
to treat it as an unrecognized flag rather than a value, and errored out
(exit code 2 is argparse's own error code — confirms this diagnosis without
needing to reproduce it). Quoting in the shell doesn't help; the token
argparse sees is identical either way.

**Fix:** `--conditions` and `--clips` (both `train_frontend.py` and
`snr_ablation_frontend_paired.py`) now take a single comma-separated string
(e.g. `--conditions 0dB,-6dB`) instead of `nargs="+"`. `run_experiment_suite.py`'s
configs B/C updated to match. The combined token never starts with `-`, so
argparse never misclassifies it. **Not independently re-verified this
session** (same standing constraint — sandbox can't run torch) — next run
should confirm configs B/C actually execute now.

The pre-existing, already-validated `snr_ablation.py` / `snr_ablation_frontend.py`
have the identical latent bug in their own `--conditions nargs="+"` flags —
left untouched (nobody has explicitly typed a bare `-6dB` token for those
two scripts, so it's never fired), but noting it here so it isn't a surprise
later if someone does.

### Honest read, updated from Day 10's

Day 10 framed the mechanism as "inconclusive, leaning toward hasn't-trained-
enough." This run sharpens that considerably: **the single-condition
training setup produces a checkpoint with a cleanly measurable ZERO effect**,
not just a noisy/small one — and the tool built specifically to distinguish
"real small effect" from "sampling noise" (the matched-pair sweep) says,
unambiguously, zero. Every positive-looking number from Days 7/9/10 should
now be treated as more likely noise than signal. The path forward is
squarely: get configs B/C (multi-condition, more data) actually running now
that the bug is fixed, and use `inspect_frontend_checkpoint.py` on every new
checkpoint before spending a full sweep's runtime on it — if alpha still
hasn't moved on B/C either, that points at the training objective/learning
rate rather than the data-scarcity hypothesis.

### Next steps, in priority order

1. Re-run `run_experiment_suite.py` — configs B and C should now execute
   with the argparse fix. This is the actual test of Day 10's "more/more
   varied data" hypothesis, which hasn't been tried yet (both attempts so
   far crashed before training even started).
2. Run `inspect_frontend_checkpoint.py` on `data/frontend_ME1.pt` (config A's
   checkpoint, already saved) and on whatever B/C produce, to check the
   alpha-didn't-move hypothesis directly and cheaply before any full sweep.
3. If B/C's checkpoints also show alpha stuck near 0, the next lever is the
   training objective/optimizer (per-parameter LR for alpha, or a nonzero
   alpha init) rather than more data — flagged above as a Phase-2 idea, not
   yet tried.
4. Still deferred: the real PRD §7 matched-pair bootstrap with ≥10 seeds.

---

## Day 12 — 2026-07-06 — Independent review of Claude Code's detection post-processing addition: logic verified correct, one real methodological bug found and fixed

**Context:** Akshath ran a separate Claude Code session (not this one) which added
opt-in detection post-processing to `argus_core.py` (`postprocess_detections()` +
`suggest_postprocess_params()` + a new `detect(..., postprocess=...)` parameter),
plus a new `postprocess_ablation.py` A/B harness and a README section. Reviewed
that work independently here before Akshath reports anything to his teacher.

### What it does

A post-hoc cleanup step on `detect()`'s predicted events only (never touches the
model): merges detections that are within `merge_gap` seconds of each other (or
overlapping), then drops anything still shorter than `min_duration` after
merging. `postprocess="auto"` derives both params from the median duration of
the 5 support-set events (`min_duration = 0.3×median`, `merge_gap = 0.5×median`).
Off by default (`postprocess=None`); existing behavior is unaffected unless a
caller opts in.

### Verification performed (this assistant's sandbox still can't run torch/real
audio, but this logic needed no model at all, so it WAS actually executed —
unlike most of this week's other verification, which was code-review-only)

1. Re-read `detect()`'s new code path line by line: confirmed that when
   `postprocess=None` (the default), the new block is skipped entirely and
   `df_out` is never touched -- output is byte-identical to before this change,
   only two new dict keys (`n_raw_detections`, `postprocess`) are added to the
   return value, which no existing caller destructures in a way that would break
   (`day1_sanity_check.py` only compares the `metrics` sub-dict).
2. Extracted `postprocess_detections()` verbatim and ran it directly (real
   execution, not just review) against 6 hand-built cases: merging two close
   fragments into one, dropping a short blip while keeping a real event, an
   empty input staying empty, two well-separated real events staying untouched,
   a fully-nested duplicate interval not shrinking the merged interval, and
   confirming the documented risk case (`merge_gap` set too large genuinely
   fuses two distinct real events into one) is real and reachable, not just a
   hypothetical caveat. **All 6 passed.**
3. Extracted `suggest_postprocess_params()` and ran it against a support set
   with known median duration (0.5s) plus extra held-out events with very
   different durations (0.3s, 0.9s): confirmed the returned `min_duration`/
   `merge_gap` are computed ONLY from the first 5 (support-set) events -- the
   held-out ground-truth events' durations do not leak into the heuristic.
   **Confirms the "not leakage" claim is actually true, not just asserted.**

### One real bug found and fixed: `postprocess_ablation.py`'s OFF/ON calls weren't seed-matched

The harness called `argus_core.detect(wav)` (OFF) then `argus_core.detect(wav,
postprocess="auto")` (ON) back to back with no shared seed pinned between them.
`evaluate_prototypes()`'s negative-resampling draw is unseeded and freshly drawn
on EVERY `detect()` call (documented and relied upon all week), so the OFF and
ON calls in each "pair" could each see a DIFFERENT raw set of detections even
before post-processing was applied -- the measured delta would have been
confounded with ordinary seed-to-seed model noise, not isolating the effect of
post-processing alone. Exactly the same pitfall this project already diagnosed
and fixed for the noise-frontend ablation (Day 11's matched-pair sweep). Fixed
the same way: `torch.manual_seed(base_seed + i)` pinned immediately before both
calls at each seed index, so OFF and ON now start from identical raw detections
and the only measured difference is the post-processing step itself.

### Honest bottom line

The core logic is correct and does what it claims, verified by actually running
it (not just reading it) -- this is a legitimate, non-leaky lever. The ablation
harness had a real, fixable methodology gap (not a logic bug in the post-
processing itself) which is now fixed. **Still not run on real audio** -- same
standing rule as every mechanism this project has tried: no number gets cited
until `postprocess_ablation.py --clip ME1 --seeds 3` is actually executed and
the real output logged here.

---

## Day 13 — 2026-07-06 — F-score plan Tier-1 item 1.1 (transductive inference): built additively, and it's a NULL on ME1 clean (single lucky seed did not survive seed-averaging). Neg-only variant neutral; any positive-prototype refinement actively hurts precision.

**Context:** Autonomous session (Akshath AFK), working the new
`ARGUS_TrackA_FScore_Improvement_ResearchPlan.md`, Tier 1 top-to-bottom. This
entry covers item **1.1 transductive inference** — the plan's flagged
"single best-evidenced lever" (DCASE 2024 Task 5 tech reports report up to
P +18.1 / R +4.64 / F1 +11.58 absolute for it).

**Environment note (important, and different from Day 12):** unlike the Day 12
reviewer's sandbox, THIS session can actually run torch + real audio on
Akshath's machine. CPU path ≈ **118s/detect** on ME1. Midway, Akshath asked to
use his GPU — `torch-directml` is installed and sees the **AMD Radeon RX 7600**;
the existing `ARGUS_DEVICE=gpu` DirectML path works out of the box and runs
≈ **20–21s/detect (~5.7× faster)**. Caveat learned and applied: DirectML changes
the encoder's float outputs, so the *absolute* indicative-F differs by device
(e.g. seed 2000 baseline: CPU F≈28.6 vs GPU F≈17.9). So **CPU and GPU numbers
are never mixed here**. The matched-pair *delta* stays valid within a device
because the negative-sampling `torch.randperm` in `evaluate_prototypes()` is
CPU-side and thus seed-reproducible regardless of the compute device (verified:
seed-pinned OFF/OFF gave byte-identical metrics).

### What was built (additive, opt-in, same pattern as `encoder_override`)
- `util._transductive_refine()` — NEW helper (added like `build_frozen_encoder()`
  was), called ONLY from `evaluate_prototypes(transductive=True)`. Uses the query
  clip's OWN embeddings (no held-out labels): score queries with the current
  prototypes, fold the most-confident pseudo-positive/pseudo-negative query
  windows back into the respective prototype as a weighted mean with the original
  5-shot / sampled-negative prototype, re-score. Reuses `get_probability()`'s
  distance math rather than re-deriving it.
- `evaluate_prototypes(..., transductive=False, transductive_pos_conf=0.7,
  transductive_neg_conf=0.3, transductive_weight=1.0, transductive_steps=1,
  transductive_refine_pos=True, transductive_refine_neg=True)` — all default so
  the OFF path is byte-identical (the whole block is skipped; `prob_pos_iter` is
  the same per-batch `get_probability()` result as always). **Verified identical**
  by seed-pinned OFF/OFF reproducibility.
- `argus_core.detect(..., transductive=False, transductive_kwargs=None)` threads
  it through; adds a `"transductive"` key to the result dict (no existing caller
  destructures beyond `metrics`, per the Day 12 audit).
- `transductive_ablation_paired.py` — NEW matched-pair harness (ported from
  `snr_ablation_frontend_paired.py`): pins the same seed before the OFF and ON
  call at each seed index, defaults to `--conditions clean` (plan says clean
  first), prints P/R alongside F because recall is already saturated.

### Real numbers

1) **Single-seed variant probe (CPU, seed=2000, ME1 clean, matched draw).** Ran
   5 configs at one shared seed to find *direction* before spending seed budget:

   | variant | P | R | F | n_pred | ΔF vs OFF |
   |---|---|---|---|---|---|
   | OFF baseline | 16.7 | 100 | 28.6 | 66 | — |
   | pos+neg, w=0.2 | 12.1 | 100 | 21.6 | 91 | −7.0 |
   | **neg-only, w=1.0** | **20.0** | 100 | **33.3** | **55** | **+4.8** |
   | pos-only, gate0.9, w=0.3 | 12.8 | 100 | 22.7 | 86 | −5.9 |
   | pos+neg, gate0.8/0.2, w=0.5 | 10.0 | 100 | 18.2 | 110 | −10.4 |

   Mechanism, clearly: recall is already 1.0, so the only lever is precision.
   Folding confident query *positives* into the positive prototype **expands**
   the positive region → more windows cross threshold → n_pred UP → precision (and
   F) DOWN. Every variant that touches the positive prototype loses. Only
   **neg-only** (sharpen the background prototype with abundant confident query
   negatives, leave the trusted 5-shot positive prototype alone) went the right
   way at this seed: n_pred 66→55, P 16.7→20.0, F +4.8.

2) **Seed-averaged matched-pair (GPU/DirectML, seeds 2000–2005, N=6, ME1 clean,
   neg-only w=1.0).** The properly-disciplined test of the one promising variant:

   - per-seed ΔF: +1.6, −1.1, +1.5, +0.6, −2.0, 0.0
   - **mean paired ΔF = +0.1 ± 1.3, ΔF/SE = +0.20**, precision 12.3 → 12.3 (no move)

### Honest verdict
**Item 1.1 is a NULL as implemented.** The single-seed +4.8 was sampling noise,
not an effect — it vanished under N=6 seed-averaging (mean ΔF +0.1, |ΔF/SE| 0.20,
nowhere near the ~2 bar). This is the *exact* Day-11 lesson repeating (an
attractive single/unpaired number that goes to ~zero when properly paired and
averaged), which is precisely why the plan mandates seed-averaging. Simple
transductive prototype re-estimation doesn't move F here because: touching the
positive prototype hurts precision (shown above), and the neg-only side is inert
(the confident-negative query mean ≈ the existing random-negative mean, so the
negative prototype barely moves — P 12.3→12.3).

**Why this isn't a refutation of the DCASE +11.58% figure, and what's left
untested:** that figure is the *official evaluator over the full validation set*
with the default `iterations=10 / query_batch_size=8`, and DCASE transductive
entries typically use a more sophisticated objective (transductive
information-maximization / label-propagation), not a one-step prototype fold.
Here the test was: the *indicative overlap* metric, a *single clip* (ME1), the
*demo* `iterations=3 / query_batch_size=32`, recall already saturated. So the
honest scope of this null is narrow — "the simplest prototype-folding form of
transductive inference gives no F gain on ME1-clean under demo settings." A
properly-powered transductive method (TIM / label propagation) and the official
settings remain the real, untested version of this lever; not attempted this
session (would be a much bigger build). ME2 and noise conditions also untested —
no reason to expect a neutral-at-clean effect flips under noise, so deferred.

**Code disposition:** kept — it's opt-in, `transductive=False` by default, proven
byte-identical when off, and compiles. NOT adopted as any default; nothing about
the validated demo path changes. Moving on to Tier-1 item **1.2 (hard-negative
mining)**, which targets precision more directly than a mean-prototype fold can.

Cross-ref: research plan §3 Tier-1 1.1 and §2; PRD §7 (matched-pair discipline)
and the §4.2 baseline detector these knobs wrap.

---

## Day 14 — 2026-07-07 — F-score plan Tier-1 item 1.2 (hard-negative mining): a HUGE, seed-robust win on ME1 (+30.5 F) that is an equally huge, seed-robust LOSS on ME2 (−47.4 F). Clip-dependent, net-negative, NOT adopted. The multi-clip check earned its keep.

**Context:** Same autonomous session as Day 13, continuing the F-score plan Tier 1.
Item **1.2 hard-negative mining**. All runs on the **DirectML GPU (RX 7600)** —
see Day 13 for the CPU-vs-GPU caveat (absolute F differs by device; matched-pair
deltas valid within a device because `randperm`/`multinomial` are CPU-seeded).

### What was built (additive, opt-in, default OFF)
- `util.evaluate_prototypes(..., hard_negative=False, hard_neg_temp=1.0)`. When
  ON, the FULL candidate-negative pool is encoded once (in query_batch_size
  chunks so the 8GB GPU doesn't OOM), each candidate is weighted by
  `softmax(-standardized_distance_to_pos_proto / hard_neg_temp)`, and each
  iteration's negative draw is a `torch.multinomial` over those weights instead
  of uniform `torch.randperm` — i.e. oversample the negatives most confusable
  with the positive prototype. Default path (`hard_negative=False`) is the
  original uniform draw, byte-identical (the else-branch is the original 4 lines).
- `argus_core.detect(..., hard_negative=False, eval_kwargs=None)` threads it
  through (I also collapsed Day-13's `transductive_kwargs` into a single generic
  `eval_kwargs` dict while here, since that param was one session old and only my
  own harness used it — cleaner for future items).
- `hardneg_ablation_paired.py` — matched-pair harness; computes the OFF baseline
  once per seed and reuses it across `--temps` (OFF doesn't depend on temp).
- No new leakage: the weights use only the 5-shot positive prototype; the
  candidate pool is the SAME whole-file negative set the baseline already uses.

### Real numbers (N=10 matched-pair per (clip,temp), GPU, clean)

ME1 (sparse events; baseline over-predicts, recall already saturated at 1.0):

| T | without F | with F | paired ΔF ± std | ΔF/SE | P off→on |
|---|---|---|---|---|---|
| 0.3 | 22.6 | 53.0 | **+30.5 ± 5.0** | **+19.46** | 12.8 → 36.1 |
| 0.5 | 22.6 | 43.2 | +20.7 ± 4.6 | +14.28 | 12.8 → 27.6 |

Every one of the 10 seeds was a large positive delta (T=0.3 range +17.8…+35.2),
recall held at 1.0 throughout. On ME1 alone this looks like the plan's first real
win — precision nearly triples with zero recall cost.

ME2 (dense events, 46 POS/508s; baseline ALREADY precise, F≈84):

| T | without F | with F | paired ΔF ± std | ΔF/SE | P off→on |
|---|---|---|---|---|---|
| 0.3 | 84.4 | 37.0 | **−47.4 ± 3.4** | **−44.50** | 74.5 → 100.0 |
| 0.5 | 84.4 | 62.0 | −22.4 ± 4.2 | −16.78 | 74.5 → 94.9 |

Every one of the 10 seeds was a large NEGATIVE delta. Precision goes to ~100 but
predictions collapse from ~53 to ~9 → recall craters → F falls off a cliff.

### Mechanism (why the sign flips between clips)
The lever sharpens the negative prototype toward whatever is most positive-
confusable. Its effect depends entirely on the baseline's operating point:
- **ME1**: baseline emits ~66–113 detections for 11 true events (recall 1.0, huge
  FP surplus). Cutting FPs is almost free → precision up, recall unaffected → big F gain.
- **ME2**: baseline is already balanced (P 74.5, recall healthy) and events are
  DENSE, so the "hardest negatives" (whole-file negative windows nearest the
  positive prototype) are largely *actual near-positive segments*. Oversampling
  them drags the negative prototype onto the positive cluster, so almost
  everything scores negative → true positives get suppressed → recall collapse.
No single temperature fixes both — the two clips need opposite treatment. Softer
T (0.5) shrinks both the ME1 gain and the ME2 loss but doesn't change the signs.

### Honest verdict
**NOT adopted.** Across the only two clips available (ME1, ME2) the effect is
clip-dependent and net-negative (ME1 +30.5 / ME2 −47.4 at T=0.3), so as a default
it would make the system worse on balanced clips. Kept in the code, opt-in and
OFF by default (validated path unchanged); it is a real, sometimes-large lever,
just not a safe general one. **This is precisely why the plan mandates multi-clip
validation before adoption** (research plan §4, and the project's own "does it
hold on other clips" rule): ME1 in isolation would have been logged as a +30
F-point breakthrough — the ME2 run turned that into an honest "no."

### Caveats and what's untested
- **Indicative metric, not the official evaluator.** All numbers here are the
  demo's interval-overlap score on query-only audio, and are GPU-internal — NOT
  the official DCASE evaluator and NOT comparable to the 56.25% hero-shot. A real
  adoption claim would need the official evaluator.
- **Noise untested and now moot** for adoption — a mechanism that's already
  net-negative at clean audio doesn't need an SNR sweep to be rejected. Skipped
  to save GPU time (deliberate, not an oversight).
- **Possible future rescue (speculative, not attempted):** make the strength
  ADAPTIVE — gate hard-neg mining (or scale `hard_neg_temp`) on a support-set-
  derivable signal of how much the baseline over-predicts, so it engages on
  ME1-like clips and stays off on ME2-like ones. That's a bigger, self-adaptive
  design, out of scope for a clean Tier-1 measurement; flagged for later.

### Tier-1 item 1.3 (feature-stream concatenation) — BLOCKED, needs Akshath
1.3 requires retraining the backbone's first conv layer (input channel count
changes from 1 to 2 when concatenating log-mel/delta-MFCC alongside PCEN). The
local setup only has the **MT training slice (2 wavs)** — used solely for
feature-normalization stats — NOT the full DCASE Development training set the
shipped `best_model.pth` was trained on. Retraining a backbone here would produce
a different, almost certainly *worse* model than the official baseline (cf.
[[argus-model-is-official-baseline]]), and there's no local path to reproduce the
official training. This is a "spend GPU hours / provide full training data"
judgment call only Akshath can make. Per the plan's loop rule (§5: if blocked on a
decision only Akshath can make, write it down and continue to the next unblocked
item), **1.3 is parked pending his call**; proceeding to the inference-only items
1.4 (distance metric) and 1.5 (sub-prototype clustering), which are unblocked.

Cross-ref: research plan §3 Tier-1 1.2/1.3 and §2; the recall-saturated ME1 vs
balanced ME2 split is a concrete instance of PRD §13's generalization risk.

---

## Day 15 — 2026-07-07 — F-score plan Tier-1 item 1.4 (cosine / L2-normalized distance): robustly NEGATIVE on both clips. This backbone's embedding *norm* carries signal; normalizing it away collapses the ranking. Rejected.

**Context:** Same autonomous session, F-score plan Tier 1, GPU (RX 7600).
Item **1.4 distance-metric ablation** — L2-normalize the query/prototype
embeddings before `euclidean_dist()` so scoring is cosine-style, the "commonly
reported improvement in metric-learning literature" the plan cites.

### What was built (additive, opt-in, default OFF)
- `util.get_probability(..., l2_normalize=False)` — when True, `F.normalize` the
  positive prototype, negative prototype, and query embeddings before the
  distance (unit vectors: `||a-b||² = 2 - 2cos`). Default False = raw euclidean,
  byte-identical.
- `util.evaluate_prototypes(..., l2_normalize=False)` passes it through; reaches
  `detect()` via the generic `eval_kwargs` dict (no new detect() param needed).
- `lever_ablation_paired.py` — NEW generic matched-pair harness for on/off
  `eval_kwargs` levers, multi-clip by default. Pairing here is CLEANER than the
  hard-neg sweep's: L2-normalization doesn't touch the negative draw, so a shared
  seed gives OFF and ON the identical `randperm` negatives (a true shared-draw
  pair). Reused for 1.5 as well.

### Real numbers (N=5 matched-pair per clip, GPU, clean)

| clip | without F | with F | paired ΔF ± std | ΔF/SE | P off→on |
|---|---|---|---|---|---|
| ME1 | 21.7 | 3.4 | −18.3 ± 4.1 | −9.94 | 12.2 → 1.8 |
| ME2 | 84.1 | 33.7 | −50.4 ± 2.2 | −51.07 | 73.9 → 21.6 |

Robustly negative on BOTH clips (unlike hard-neg's split sign), large |ΔF/SE|.

### Mechanism / honest read
The single-seed probe showed the tell: on ME1, cosine scoring dropped precision
to ~1.8% AND recall to ~45% simultaneously — i.e. it didn't just move the
operating point, it *collapsed the ranking* (ME1 emitted ~278 predictions for 11
events, ~5 correct). That rules out "just needs a re-tuned threshold": no
threshold recovers a ranking that's fundamentally worse. Root cause: the shipped
backbone was trained with the raw squared-euclidean prototypical loss, so the
**magnitude** of its embeddings carries discriminative information; L2-normalizing
throws that away and leaves only direction, which is not separable enough for this
task. The metric-learning prior that "cosine usually helps" assumes a model
trained *for* cosine (normalized embeddings / angular loss) — which is exactly
what Tier-2 item 2.1 proposes to build, and is the only context in which 1.4
would be expected to help. On the frozen euclidean-trained backbone it's a clear
loss.

**Verdict:** REJECTED as a default; kept opt-in and OFF (validated path
unchanged). Genuinely useful negative result — it also sharpens the case for
Tier-2 2.1 (contrastive/angular retrain), the setting where normalized-distance
scoring should finally pay off.

Cross-ref: research plan §3 Tier-1 1.4 (and its own "check empirically" caveat),
§3 Tier-2 2.1 (contrastive/angular backbone — where cosine would fit).

---

## Day 16 — 2026-07-07 — F-score plan Tier-1 item 1.5 (sub-prototype clustering): the plan-mandated "check empirically first" says the 5-shot positive embeddings are UNIMODAL, so not pursued. Tier-1 complete; Tier-2 is blocked on the same retrain-data decision as 1.3.

**Context:** Same autonomous session, closing out Tier 1. Item **1.5 sub-prototype
clustering** — the plan explicitly says to *cluster the 5 support embeddings and
only invest further if they are genuinely multi-modal*. Did exactly that
diagnostic (`subproto_diag.py`, GPU) rather than jumping to build the mechanism.

### Diagnostic (real numbers, both clips, the frozen backbone's own embeddings)
Both clips: exactly **n_pos = 5** positive segments (each of the 5 support events
fits inside one adaptive segment — no chunking, so 5 embeddings, ||emb||≈6.2–6.4).

| clip | k=2 cluster sizes | inertia2/inertia1 | silhouette | centroid_sep/within_rms |
|---|---|---|---|---|
| ME1 | [4, 1] | 0.572 | 0.235 | 2.16 |
| ME2 | [2, 3] | 0.583 | 0.166 | 1.73 |

**Read:** neither clip is meaningfully multi-modal. Silhouette 0.17–0.24 is far
below the ~0.5 that indicates real 2-cluster separation; the inertia ratio ~0.57
is just the mechanical gain of fitting 2 centroids to 5 points (2 centroids
*always* beat 1), not structure. ME1's "best" split is a single outlier ([4,1]).

### Decision (plan-aligned): not pursued
On 5 unimodal points, splitting into 2 sub-prototypes would either (a) promote a
lone outlier to its own prototype (ME1) or (b) split a single blob into two
high-variance halves (ME2) — a sub-prototype estimated from 1–2 points is
extremely noisy and, scored by nearest-sub-prototype, would most plausibly ADD
false positives (a query near the outlier now scores positive). The plan's own
guidance is explicit: "likely lower-impact than 1.1–1.2 unless the 5-shot examples
genuinely are multi-modal — check this empirically before investing more time
here." The check says they are not, so per that instruction I did **not** build
the full mechanism (no code added for 1.5). If Akshath wants an empirical ΔF
confirmation rather than the diagnostic, it's a ~1h add (nearest-sub-proto scoring
+ a 2-clip N=5 run), but the diagnostic already answers it and the plan advises
against the spend.

---

## Tier-1 wrap-up (F-score improvement plan) — what the autonomous session found

| item | mechanism | status | ME1 clean | ME2 clean | verdict |
|---|---|---|---|---|---|
| 1.1 | transductive re-estimation | built, tested | +0.1 ± 1.3 (N=6) | — | **null** (single-seed +4.8 didn't survive averaging) |
| 1.2 | hard-negative mining | built, tested | **+30.5 ± 5.0** (N=10) | **−47.4 ± 3.4** (N=10) | **clip-dependent, net-negative → not adopted** |
| 1.3 | feature-stream concat | not built | — | — | **BLOCKED** (needs backbone retrain + full training data) |
| 1.4 | cosine / L2-norm distance | built, tested | −18.3 ± 4.1 (N=5) | −50.4 ± 2.2 (N=5) | **rejected** (norm carries signal; ranking collapses) |
| 1.5 | sub-prototype clustering | diagnosed | unimodal (sil 0.24) | unimodal (sil 0.17) | **not warranted** (per plan's check-first rule) |

**All numbers are the demo's indicative interval-overlap metric on GPU (DirectML),
NOT the official DCASE evaluator and NOT comparable to the 56.25% hero-shot.**
Every lever is opt-in and OFF by default; the validated demo path is byte-for-byte
unchanged (verified by seed-pinned OFF/OFF reproducibility on Day 13).

**The one real signal:** hard-negative mining (1.2) genuinely and largely helps
clips in ME1's regime — baseline massively over-predicts with recall already
saturated — and genuinely hurts balanced clips (ME2). A *regime-adaptive* version
(gate/scale it on a support-derived over-prediction signal) is the obvious next
idea, BUT it cannot be honestly validated on only 2 clips (a 2-point gate is just
memorizing "ME1 vs ME2"). That needs more validation clips from the full
Development set — which isn't local. Flagged for Akshath, not built.

## Tier-2 — BLOCKED (all four items), needs Akshath's decision
2.1 (supervised-contrastive/angular retrain), 2.2 (frame-level embeddings),
2.3 (foundation-model backbone: BEATs/PANNs), 2.4 (SpecAugment/mixup) ALL require
retraining or replacing the backbone, which needs (a) the full DCASE Development
training set — not local (only the 2-file MT slice, for stats) — and (b) real GPU
hours. Same block as 1.3. Per the plan's §5 loop rule ("if blocked on a decision
only Akshath can make, write down exactly what's blocked and why, then continue")
— there is no further *unblocked* item to continue to, so the autonomous loop
stops here with this handoff.

**Decisions needed from Akshath to unblock Tier 2 / item 1.3:**
1. Provide the full DCASE Development_Set Training data locally (or a Colab/Kaggle
   path), and approve spending GPU hours on a real backbone retrain.
2. For 2.3 specifically: approve adding a large pretrained-audio dependency
   (BEATs/PANNs checkpoint + its runtime) as a separate experiment track.
3. Optionally: supply ≥3–4 more validation clips so the one lever with real signal
   (regime-adaptive hard-negative mining) could be built and validated without
   overfitting to ME1-vs-ME2.

Cross-ref: research plan §3 Tier-1 1.5, Tier-2 all; §5 loop/stop rule; PRD §13
(generalization risk — the ME1-vs-ME2 sign flip is the headline example).

---

## Day 17 — 2026-07-07 — Tier-2 item 2.3 (frozen PANNs CNN14 backbone swap, Akshath-approved): another clip-dependent, net-negative lever. Helps ME1 (+7.5 F), hurts ME2 (−11.2 F). Frozen out-of-domain AudioSet model does not beat the from-scratch ResNet overall. Not adopted.

**Context:** Akshath approved adding a large pretrained-audio dependency for an
INFERENCE-ONLY frozen-backbone test (explicitly *not* the retrain, which stays
blocked on the still-downloading full training set). Did item **2.3** with PANNs
CNN14 (AudioSet-pretrained, 81.8M params, 2048-d embeddings); chose PANNs over
BEATs because it's a clean frozen-embedding CNN with no fairseq/unilm stack.

### Integration (separate track, nothing in the validated path touched)
- New `pretrained_backbone.py`: loads frozen CNN14 and, because PANNs consumes
  RAW 32kHz audio (its own internal 64-mel log-mel) while the ResNet consumes
  128-mel PCEN patches, it cuts raw-audio windows at the SAME segment length /
  hop as the baseline eval `.h5` (reusing `argus_core.build_eval_features()` for
  those, converted frames→seconds), embeds them, and reuses `util.get_probability`
  + the baseline threshold/onset-offset math + `score_against_ground_truth` so the
  F-numbers are directly comparable. Only the embedding function differs.
- New `pretrained_ablation.py`: matched-pair harness, both clips, tests PANNs raw
  AND PANNs+L2-normalized vs the ResNet baseline. Embeddings are deterministic so
  they're extracted ONCE per clip (ME1: 5 pos / 4007 neg / 3925 query windows;
  ME2: 5 / 3650 / 3593) and every seed just re-samples negatives from the cache.
- Deps: `panns_inference` + `torchlibrosa` installed (torch 2.2.1+cpu / DirectML
  unaffected — verified). The 327MB `Cnn14_mAP=0.431.pth` + labels CSV were staged
  manually into `~/panns_data/` because panns_inference's auto-download uses `wget`
  (absent on Windows). PANNs runs on CPU (its torchlibrosa STFT isn't DirectML-
  friendly); the ResNet baseline still uses the DirectML GPU.

### Real numbers (N=10 matched-pair per clip, clean)

| clip | variant | baseline F | with F | paired ΔF ± std | ΔF/SE | P b→w | R b→w |
|---|---|---|---|---|---|---|---|
| ME1 | PANNs raw | 22.4 | 30.0 | **+7.5 ± 4.7** | +5.09 | 12.7→17.9 | 100→90.9 |
| ME2 | PANNs raw | 84.3 | 73.1 | **−11.2 ± 2.6** | −13.63 | 74.3→57.6 | 97.6→100 |
| ME1 | PANNs L2 | 22.4 | 3.5 | −18.9 ± 4.8 | −12.42 | 12.7→1.8 | 100→90.9 |
| ME2 | PANNs L2 | 84.3 | 54.2 | −30.1 ± 2.6 | −37.19 | 74.3→37.5 | 97.6→97.6 |

### Honest verdict
**Not adopted.** PANNs-raw is *clip-dependent and net-negative*: it genuinely and
robustly helps ME1 (+7.5, ΔF/SE 5.1 — more precise, one recall miss) but robustly
hurts ME2 (−11.2, ΔF/SE −13.6, larger and more certain than the ME1 gain). The
same directional split as Day-14's hard-negative mining — and for the same
underlying reason: ME1's baseline massively over-predicts (P 12.7), so almost any
*more conservative* representation helps it, while ME2's baseline is already
balanced (P 74.3) so an out-of-domain AudioSet embedding just degrades it. PANNs
+L2 collapses on both, consistent with Day 15 (this euclidean scorer wants raw,
not normalized, embeddings). A frozen general-audio model, with zero adaptation,
does not beat the task-trained ResNet across clips.

### Caveats / what's untested
- **Indicative metric, GPU-baseline vs CPU-PANNs**, not the official evaluator —
  the standing caveat. The paired delta is still the honest read within this
  metric; the PANNs column being CPU doesn't confound it (the embedding is
  deterministic, not device-noisy).
- **Frozen only.** The plan's 2.3 also allows "combine via a small adapter." A
  trained linear adapter / fine-tune on top of PANNs embeddings could plausibly
  turn the ME1 gain into an everywhere-gain — but that's a *training* job (needs
  the same data/GPU decision as 1.3 / the rest of Tier 2), so it stays blocked.
  What's reported here is strictly the frozen, inference-only swap Akshath scoped.
- **Recurring theme worth flagging for the writeup:** every precision-boosting
  lever tried so far (hard-neg 1.2, PANNs 2.3) helps ME1 and hurts ME2. ME1 and
  ME2 sit at opposite operating points and want opposite treatment; no single
  static change has helped both. This is the strongest argument yet that the real
  win is *regime-adaptive* behavior (the Day-14 idea, task 3 below) — but that
  still can't be validated on only 2 clips.

Cross-ref: research plan §3 Tier-2 2.3; PRD §13 (generalization). Next: item 2.4
(support-set augmentation), then the regime-adaptive hard-negative prototype.

---

## Day 18 — 2026-07-07 — Tier-2 item 2.4 (support-set augmentation): a clean, UNIFORM reject. Folding pitch/stretch/noise variants into the mean positive prototype blurs it off the query signal → recall collapses. Hurts both clips, at clean AND under noise, even noise-only. Not adopted.

**Context:** Akshath's task-2.4 framing = augment the 5-shot support examples
(pitch shift, time-stretch, added noise) BEFORE computing the positive prototype,
using only local data. (Note this is the *inference-time* version; the research
plan's own 2.4 is *training-time* SpecAugment/mixup during a backbone retrain,
which stays blocked with the rest of Tier 2.)

### What was built (additive, opt-in, default OFF)
- `util.evaluate_prototypes(..., extra_pos_feats=None)` — when given, augmented
  support patches (M, seg_len, n_mels) are encoded the same way as the real
  support patches and folded into the positive prototype (originals + augmented).
  Default None → prototype is exactly the 5-shot mean, unchanged. Reaches detect()
  via the generic `eval_kwargs`.
- `support_augment.py` — builds pitch(±1,±2) / stretch(0.9,1.1) / Gaussian-noise
  (15,8dB) variants of each support event, using the SAME Feature_Extractor / PCEN
  / (2**32) scaling as the baseline.
- `support_aug_ablation.py` — matched-pair harness (aug only changes pos_proto →
  clean shared-draw pairing), with `--condition` (SNR) and `--aug_mode`
  (full/noise/mild) so the robustness claim can be tested under noise.

### One real bug found and fixed before any number was trusted
First run returned **0 detections on both clips** (F=0). Root cause: the real
support patches `X_pos` are normalized by `Datagen_test.feature_scale(X) =
(X - train_mean)/train_std` (training-set stats), but my augmented patches were
raw PCEN → out-of-distribution to the encoder → corrupted prototype. Fixed by
applying the identical `Datagen(conf)` mean/std normalization to the augmented
patches (verified: augmented embedding norms then matched the originals, 6.14 vs
6.19). Only after this fix are the numbers below meaningful.

### Real numbers (N=10 matched-pair per clip)

| condition | aug_mode | clip | without F | with F | paired ΔF ± std | ΔF/SE | P off→on | R off→on |
|---|---|---|---|---|---|---|---|---|
| clean | full (40 patch) | ME1 | 22.4 | 11.4 | −11.0 ± 7.6 | −4.61 | 12.7→6.8 | 100→34.5 |
| clean | full (40 patch) | ME2 | 84.1 | 9.9 | −74.2 ± 4.4 | −53.67 | 74.1→66.8 | 97.6→5.4 |
| 0 dB | noise-only (15) | ME1 | 19.6 | 4.9 | −14.7 ± 2.5 | −18.36 | 11.0→3.2 | 90.9→10.9 |
| 0 dB | noise-only (15) | ME2 | 76.0 | 38.3 | −37.7 ± 5.7 | −20.75 | 62.7→71.1 | 96.8→26.3 |

### Honest verdict
**Rejected — the cleanest negative of the whole plan.** Unlike hard-neg (1.2) and
PANNs (2.3), which were clip-dependent, support augmentation hurts *uniformly*:
both clips, at clean and under noise, and even with the gentlest noise-only
variant. Every case is a **recall collapse** (ME2 clean: recall 97.6→5.4). The
mechanism is inherent to the design: averaging pitch/stretch/noise variants into
the single MEAN positive prototype drags it toward a generic "distorted-call"
region that is farther from the actual query events than the clean 5-shot mean —
so fewer queries clear the positive threshold. The robustness benefit does NOT
appear even where it was most expected (noise-augmented prototype on 0 dB query);
noise augmentation just makes the prototype more conservative (ME2 precision even
rose 62.7→71.1) at a catastrophic recall cost.

### Alternatives (untested / blocked), for the writeup
- **Nearest-sub-prototype augmentation** (don't average — keep each augmented
  variant as its own sub-prototype and score a query against the *nearest*). This
  directly attacks the blur (a clean query still matches the clean sub-proto), so
  it might avoid the recall collapse. It's the augmentation-flavored version of
  item 1.5 and is a genuinely different mechanism — untested, flagged for later.
- **Training-time augmentation** (the plan's original 2.4: SpecAugment/mixup during
  a backbone retrain) — blocked with the rest of Tier 2 on the training-data/GPU
  decision. What's reported here is strictly the inference-time prototype
  augmentation Akshath scoped.

Cross-ref: research plan §3 Tier-2 2.4; item 1.5 (sub-prototype, the untested
rescue). Next: the regime-adaptive hard-negative prototype (Day-14 idea).

---

## Day 19 — 2026-07-07 — Task 3: regime-adaptive hard-negative. On ME1+ME2 it converts the +30/−47 clip-dependent lever into +30.6 / EXACTLY 0.0 — the best two-clip result in the plan. But the gate is fit on 2 clips; this is a PROVISIONAL prototype at real risk of memorizing ME1-vs-ME2, NOT a validated win. Do not claim it works until more clips exist.

**Context:** Akshath's task 3. Day-14 found hard-negative mining (1.2) helps the
over-predicting clip (ME1, +30.5) but hurts the balanced one (ME2, −47.4). Idea:
detect the regime from the support set + unlabeled negatives (no query labels) and
engage hard-neg ONLY when it will help.

### Finding the regime signal (one refuted, one that works)
- **Refuted — support event rate.** First guess was "sparse-support clips
  over-predict." Measured: ME1 and ME2 both have support rate ≈ **0.747/s** (the 5
  support events are front-loaded into the first ~6.7s in *both* clips), and the
  support-implied expected count (477 vs 374) is nowhere near the true post-support
  GT (11 vs 41). Support rate does NOT separate the regimes — dropped it.
- **Works — prototype/background separability.** `separability =
  distance(pos_proto, neg_pool_mean) / mean_spread(support_embeddings)`. Low means
  the positive prototype sits close to the background relative to how tight the
  5 support examples are → mushy boundary → over-prediction → hard-neg helps.
  Measured (deterministic; uses the full-pool mean, so seed-independent):

  | clip | separability | true regime |
  |---|---|---|
  | ME1 | **1.06** | over-predicts (baseline P≈13) |
  | ME2 | **1.72** | balanced (baseline P≈74) |

  A clean, *mechanistically motivated* gap (not an arbitrary fit statistic).

### Implementation (additive, opt-in, default OFF)
`util.evaluate_prototypes(..., adaptive_hard_neg=False, adaptive_gate=1.4)`:
encode the negative pool once, compute separability, and set `hard_negative` on
iff separability < `adaptive_gate`, then reuse the exact item-1.2 hard-neg
machinery (temp 0.3). When the gate is OFF the run is byte-identical to baseline
(verified: ME2 ΔF = 0.0 ± 0.0 at every seed). Reaches detect() via `eval_kwargs`.

### Real numbers (N=10 matched-pair per clip, GPU, clean)

| clip | separability → gate | without F | with F | paired ΔF ± std | ΔF/SE | P off→on |
|---|---|---|---|---|---|---|
| ME1 | 1.062 → ON | 22.4 | 53.0 | +30.6 ± 5.0 | +19.48 | 12.7→36.1 |
| ME2 | 1.723 → OFF | 84.1 | 84.1 | +0.0 ± 0.0 | n/a | 74.1→74.1 |

On these two clips the adaptive detector captures the full ME1 hard-neg gain AND
does exactly zero harm to ME2 — turning the clip-dependent +30/−47 into +30/neutral.

### Honest verdict — PROVISIONAL, not validated (this is the important part)
**This is a 2-clip fit and must not be reported as a working method.** The gate
threshold 1.4 was chosen to sit between ME1's 1.06 and ME2's 1.72, so "it helps
ME1 and doesn't hurt ME2" is very close to *tautological* — I built the gate to do
exactly that. Two genuine reasons for cautious optimism, and the hard limits on
each:
- The signal is **mechanistically derived** (low separability → over-prediction →
  hard-neg helps), not a blind label-fit — that's more likely to generalize than
  an arbitrary threshold. BUT the mechanism itself is only *demonstrated* on 2
  points; a third clip could have separability 1.3 and behave either way.
- The gate is **deterministic and byte-safe when OFF** (ME2 provably unchanged),
  so at worst it's a no-op on balanced clips it correctly identifies. BUT whether
  it *correctly identifies* an unseen clip's regime is exactly what's untested.

**What a validated version needs (all blocked on more data):** measure separability
AND the actual hard-neg ΔF across ≥3–4 more clips from the full Development set, (a)
confirm separability correlates with hard-neg benefit beyond these 2 points, and
(b) set the threshold from that distribution instead of the midpoint of 2 values.
Until then this is a promising prototype and a good story for *why* hard-neg is
clip-dependent — not a result to cite as "ARGUS improved by +30."

Cross-ref: research plan §3 (Tier-1 1.2, the lever this gates) and §4/§5
(multi-clip validation, the exact discipline this entry is bounded by); PRD §13.

---

## Tasks 2.3 / 2.4 / adaptive — session wrap-up (Akshath's Tier-2 batch)

| task | mechanism | ME1 | ME2 | verdict |
|---|---|---|---|---|
| 2.3 | frozen PANNs CNN14 backbone | +7.5 (N=10) | −11.2 (N=10) | clip-dependent, net-negative → not adopted |
| 2.4 | support-set augmentation (inference) | −11.0 clean / −14.7 @0dB | −74.2 clean / −37.7 @0dB | uniform reject (recall collapse) |
| 3 | regime-adaptive hard-negative | +30.6 (N=10) | +0.0 (N=10) | **provisional** — 2-clip fit, not validated |

All are additive, opt-in, default OFF; the validated demo path is unchanged. All
numbers are the indicative overlap metric on GPU, NOT the official DCASE evaluator
(not comparable to 56.25%). New deps this batch: `panns_inference` + `torchlibrosa`
(+ a manually-staged 327MB Cnn14 checkpoint in ~/panns_data/). New files:
`pretrained_backbone.py`, `pretrained_ablation.py`, `support_augment.py`,
`support_aug_ablation.py`, plus the reused `lever_ablation_paired.py`.

**Standing block unchanged:** items 1.3 and the *training-based* half of Tier 2
(2.1, 2.2, 2.3-with-adapter, 2.4-training-time) remain blocked on the full DCASE
Development training set + GPU-hours decision. Per Akshath's instruction, did NOT
touch config.yaml's train_dir or start the 1.3 retrain (zip still downloading).
The single most valuable *unblocked* next step is validating the Day-19 adaptive
gate on ≥3–4 more clips — which itself needs those extra clips.

---

## Day 20 — 2026-07-09 — Regime-adaptive hard-negative gate: VALIDATED ON THE FULL DCASE VALIDATION SET (12 clips, 6 folders) — and it does NOT generalize. The ME1 +30 was clip-specific. The one live thread is honestly closed.

**Context:** The full DCASE 2024 Development Set is now local. Akshath's #1
priority: test whether the Day-19 regime-adaptive gate (engage hard-neg only when
support-derived *separability* is low) actually generalizes beyond ME1/ME2, before
spending GPU on the Tier-2 retrain he conditioned on it.

**Design (`gate_generalization.py`):** MT-slice normalization held FIXED (the gate
was calibrated under it — changing normalization would confound "new clips" with
"new stats"; the full training set is reserved for the retrain). 12 clips, the 2
highest-POS from each of HB/PB/ME/PB24/RD/PW, each truncated to support_end+720s
for bounded, comparable runtime. Per clip: separability (deterministic) + N=5
matched-pair baseline vs. hard-neg (temp 0.3). Recording both columns per seed
lets any gate threshold be evaluated post-hoc. **Sanity: ME1/ME2 reproduced the
Day-19 anchors** (ME1 sep 1.06 → dF +29.3 vs Day-19's +30.6; ME2 sep 1.72 → −48.6)
— the harness is sound.

### Real numbers (N=5 matched-pair per clip, sorted by separability)

| sep | folder | baseF | P | R | hardF | dF ± sd |
|---|---|---|---|---|---|---|
| 0.33 | HB | 51.5 | 40.2 | 71.5 | 44.3 | **−7.1** ± 3.9 |
| 0.38 | RD | 16.4 | 9.2 | 75.0 | 27.4 | +11.0 ± 7.6 |
| 0.45 | PB | 4.4 | 2.3 | 61.2 | 5.5 | +1.1 ± 0.2 |
| 0.46 | HB | 80.8 | 72.0 | 92.3 | 73.8 | **−6.9** ± 3.4 |
| 0.52 | RD | 6.9 | 3.6 | 71.4 | 6.9 | +0.1 ± 0.5 |
| 0.60 | PB24 | 15.4 | 8.8 | 60.3 | 18.4 | +3.0 ± 1.3 |
| 0.67 | PW | 9.3 | 4.9 | 100 | 10.3 | +1.0 ± 2.2 |
| 0.69 | PB | 2.4 | 1.2 | 86.1 | 3.1 | +0.6 ± 0.2 |
| 0.76 | PW | 41.0 | 25.8 | 100 | 44.7 | +3.7 ± 3.7 |
| 1.06 | ME1 | 25.5 | 14.7 | 100 | 54.8 | **+29.3** ± 6.8 |
| 1.09 | PB24 | 7.1 | 16.7 | 4.5 | 0.0 | **−7.1** ± 6.4 |
| 1.72 | ME2 | 84.6 | 74.8 | 97.6 | 36.0 | **−48.6** ± 2.1 |

### Honest verdict — the gate does NOT generalize
- **The helped and hurt clips interleave across the separability axis; no threshold
  separates them.** The *lowest*-separability clip (HB, 0.33) is HURT (−7.1) — a
  direct counterexample to the gate's premise — while RD at 0.38 right next to it
  is helped (+11.0). HB 0.46 hurt, PB24 0.60 helped, ME1 1.06 helped huge, PB24
  1.09 hurt. corr(sep, dF) = **−0.53** (moderate, driven by the ME extremes; not a
  usable separation).
- **The Day-19 threshold (1.4) fails outright here:** every new clip has sep < 1.4,
  so the gate would engage hard-neg on all of them — producing a mix of tiny gains
  and real losses, and notably hurting the balanced HB clips.
- **The "best re-fit threshold gives +2.9 over baseline" is an artifact:** it is
  dominated by *including ME1 (+29.3, a calibration clip)* while *excluding ME2
  (−48.6)*. On the **10 genuinely-new clips alone, hard-neg nets ≈ 0** (sum of dF
  ≈ −0.6 over 10 clips) and the separability signal does not predict which clips
  benefit. Mean F across all 12: baseline 28.8, always-hard-neg 27.1 (worse).
- **Why ME1 was special:** ME1 is an extreme recall-saturated over-predictor
  (P 14.7, R 100) — the one regime where trading recall for precision is almost
  free. Most real clips are recall-*limited* (PB R 61, RD R 75, PB24 R 60), where
  hard-neg's recall cost isn't offset; and balanced clips (HB P 40–72, ME2 P 75)
  are hurt outright. The ME1 win did not represent a general phenomenon.

**Disposition:** the regime-adaptive hard-negative gate (Day 19) and hard-negative
mining generally (Day 14) are **NOT adopted** — now confirmed on 12 clips, not 2.
The "one live thread" from the Tier-1/2 batch is closed, honestly. This is exactly
what the multi-clip discipline (research plan §4; the Day-19 caveat "at real risk
of memorizing ME1-vs-ME2") existed to catch, and it caught it. All the levers'
code stays opt-in/OFF; the validated demo path is unchanged.

### Consequence for Akshath's task 2 (retrain)
His go-ahead was explicit: *"If that holds up, proceed with item 1.3 and Tier 2 …
GPU hours."* **It did not hold up.** So the condition for spending the GPU hours on
a backbone retrain is not met — PAUSED for Akshath's call rather than auto-spending
the compute. Note the strategic reframing this result forces: with the adaptive
gate dead, the remaining high-ceiling direction is the PRD §4.5 one — rebuild
Baseline B as a **frame-level embedding model** (the ~65% SOTA family) rather than
incrementally patching the 41.6% prototypical backbone. That's a bigger decision
than "run 1.3/2.1" and worth an explicit choice from him before the compute spend.

Cross-ref: research plan §3 Tier-1 1.2 / Task-3, §4.5 (frame-level baseline), §5
(multi-clip validation before adoption); PRD §13 generalization risk; Day 14, Day 19.

---

## Day 21 — 2026-07-22 — Real TIM implemented (distinct from the Day-13 null result) and smoke-tested; official full-validation-set scoring resumed after being found incomplete

**Gap note, logged honestly per this notebook's own standing rule:** no entry exists between Day 20 (2026-07-09) and today (2026-07-22), a 13-day gap. The session handoff covering that period (`ARGUS_Session_Handoff.md`) documents the CBSE Skill Expo / CLM Presentation Day likely happening ~2026-07-10, but nothing between then and now is logged anywhere. This entry starts a new work session picking up directly from the Day-20 disposition and the subsequent literature-review pass (`ARGUS_TrackA_Literature_Collection_2026-07-21.md`, `ARGUS_TrackA_Implementation_Plan_2026-07-21.md`), not from anything done in the gap.

**A1 (official full-validation-set score) — status: IN PROGRESS, not yet complete.** `official_eval.py`'s prediction CSV (`pred_val_prototypical.csv`) turned out to only cover 28 of the 43 official clips — the other 15 (all of RD's 6 files, 8 of PW's 15) were never run, likely because the original job didn't finish (these are the longest clips in the set: RD alone is ~186 min/file × 6 files, PW's missing files total ~817 min). Scoring the partial CSV as-is would have been misleading — the official evaluator counts a clip with no submitted predictions as 100% missed (all false negatives) for that file, which would make an incomplete run look like a real, terrible score rather than an incomplete one. Resumed `official_eval.py --resume` on GPU (DirectML, RX 7600) instead of scoring the partial set. First clip observed: pw3.wav (103.2 min audio) took 380s, i.e. ~3.7s of processing per minute of audio — extrapolating, the remaining ~1900 min of audio should take roughly 1.5-2 hours, not the much longer estimate a naive per-file-count projection would suggest. **Real score not yet in hand; do not cite a full-eval F1 number until this finishes and the report JSON is read.**

**A2 (real TIM) — implemented, wired in, smoke-tested; full paired result pending.**
- Read the actual official implementation (github.com/mboudiaf/TIM, `src/tim.py`) rather than working from a paraphrased summary of the paper — this caught a real discrepancy: an earlier automated paper-summary had guessed the q-update's normalization exponent as a fixed `1/2`, but the real code computes it as `beta = marg_weight/(marg_weight+1)`, which only equals 0.5 under the *default* hyperparameters (marg_weight=1.0). Implemented faithfully in `util.py` as new functions `_tim_get_logits()` and `_tim_refine()`, wired into `evaluate_prototypes(tim=False, ...)` (new params, default off, existing `transductive=`/`hard_negative=` paths byte-for-byte unchanged) and `argus_core.detect(tim=False)`. New ablation harness `tim_ablation_paired.py`, same matched-pair/seed-paired discipline as every other lever this project has tested.
- **Confirmed empirically important design fact:** TIM-ADM only ever updates the two prototype vectors, never the frozen encoder — verified both in the source code and in this port, so this lever cannot touch the validated frozen backbone regardless of outcome.
- **Single-seed smoke test (ME1, clean, seed=4000 — NOT a result, just confirms the code runs correctly):** baseline F=43.1% (P=27.5, R=100.0) vs. TIM-on F=1.8% (P=0.9, R=63.6) — precision collapsed. N_s=55 (5 positive shots + 50 sampled negatives) vs. N_q=3926 query windows for this clip — a support:query ratio far more extreme than the standard few-shot image benchmarks TIM's default hyperparameters (temp=15, λ=0.1, 150 iters) were tuned on. This matches, and does not yet confirm, the exact caveat flagged in `_tim_refine()`'s own docstring before this was ever run.
- **Full N=5 matched-pair run on ME1/clean launched, not yet complete as this entry is written** (background job `tim_ablation_run_2026-07-22.log`) — a single unseeded run is exactly the kind of result this project's own discipline says not to cite as "the" answer. Follow-up entry to come with the real mean±std once it finishes.
- Learned a process lesson worth recording: this codebase's scratch paths (`app_run/`, `detect_scratch/`) are fixed, shared, absolute paths — running two `detect()`-calling scripts concurrently (the A1 resume job and the A2 smoke test) collided on the same feature-extraction directory and threw a Windows file-lock error on the very first call. Not a bug in the new TIM code; a pre-existing single-process assumption in the pipeline. Fix used: never run two of these scripts at the same time — stop one (safe, since `official_eval.py --resume` and the ablation scripts are all checkpointed/idempotent-per-clip) before starting the other.

Cross-ref: `ARGUS_TrackA_Implementation_Plan_2026-07-21.md` §1 (A1-A2), `ARGUS_TrackA_Literature_Collection_2026-07-21.md` §B.1 (TIM citation trail); Day 13 (the `_transductive_refine()` null result this is explicitly NOT a repeat of).

---

## Day 21 (continued) — 2026-07-22 — Real TIM, full N=5 paired result: clear, consistent, strongly NEGATIVE at default hyperparameters

**Real numbers** (`tim_ablation_paired.py --clip ME1 --seeds 5`, matched-pair, seeds 4000-4004, clean condition):

| condition | without F | with F | paired ΔF (mean±sd) | ΔF/SE | P off→on | R off→on |
|---|---|---|---|---|---|---|
| clean | 38.2 ± 6.0 | 2.7 ± 0.6 | **−35.5 ± 6.2** | **−12.82** | 23.8 → 1.4 | 100.0 → 83.6 |

**This is not noise.** |ΔF/SE| = 12.82 is far past the ~2 threshold this project treats as meaningful (see every prior ablation's own reminder). Real TIM, at the official repo's default hyperparameters (temp=15, λ=0.1, marg_weight=1.0, cond_weight=0.1, 150 ADM iterations), makes this specific detector dramatically worse on ME1 — consistently, across all 5 seeds, not an artifact of one bad draw.

**Mechanistic explanation, not just "it broke":** the collapse is almost entirely a precision problem (23.8%→1.4%) with recall barely moving (100%→83.6%) — i.e., TIM makes the model call vastly more query windows positive. TIM-ADM's marginal-entropy term (the `beta`-powered normalization in `q_update`) is specifically designed to keep the query set's *predicted* class split from collapsing onto one class — which implicitly assumes/encourages something closer to a **balanced** class distribution across the query set. Standard few-shot image benchmarks are constructed with balanced query sets per episode, so this assumption is reasonable there. A real bioacoustic detection query set is the opposite: overwhelmingly background/negative, with the true positive class being rare. Pushing the predicted distribution toward balance in that setting means predicting far more positives than actually exist — which is exactly the precision collapse observed. This is a plausible, mechanistically-grounded explanation for why an evidenced literature lever (+27%/+10% F1 on DCASE-2021/2022 per the MDPI paper) fails badly here, not a contradiction of that literature — that paper's own query sets were presumably far closer to the small, more-balanced regime TIM was designed for.

**Disposition:** real TIM at default hyperparameters is **NOT adopted** — a clear, well-evidenced negative result, not a repeat of the Day-13 heuristic's null result (this is a genuinely different, correctly-implemented mechanism that was tested and found to actively hurt). Two honest options for anyone who wants to pursue this further, neither attempted yet: (a) retune away from the balanced-class assumption specifically — e.g. lower `marg_weight` toward 0 to weaken the marginal-entropy term's pull, or (b) treat this negative result itself as the finding (an off-the-shelf transductive few-shot method's core assumption doesn't transfer to imbalanced few-shot *detection*) rather than chasing a hyperparameter fix. Neither has been tried; this entry reports what was actually run, not what could be tried next.

**Process note:** this result required stopping the concurrently-running A1 official-eval resume job first (same shared-scratch-path collision noted in the entry above) — restarted after this ablation completed.

Cross-ref: `ARGUS_TrackA_Implementation_Plan_2026-07-21.md` §1 (A2); `ARGUS_TrackA_Literature_Collection_2026-07-21.md` §B.1/B.2 (the TIM citations this result qualifies, not contradicts).

---

## Day 21 (continued) — 2026-07-22 — Post-processing ablation run on real audio for the first time (A3): exact-zero effect, and precisely why

**Real numbers** (`postprocess_ablation.py --clip ME1 --seeds 5`, matched-pair, seeds 2000-2004):

| seed | raw n (OFF=ON) | P (OFF=ON) | R (OFF=ON) | F (OFF=ON) |
|---|---|---|---|---|
| 2000 | 66 | 0.1667 | 1.0 | 0.2857 |
| 2001 | 47 | 0.234 | 1.0 | 0.3793 |
| 2002 | 68 | 0.1618 | 1.0 | 0.2785 |
| 2003 | 55 | 0.2 | 1.0 | 0.3333 |
| 2004 | 47 | 0.234 | 1.0 | 0.3793 |

Mean over 5 seeds: OFF P=0.1993 R=1.0 F=0.3312 — ON P=0.1993 R=1.0 F=0.3312. **ΔF = +0.0 exactly, at every single seed independently**, not just on average.

**This is a more specific finding than "no effect on average" — the raw detection count is identical to the post-processed count at every seed (66=66, 47=47, 68=68, 55=55, 47=47), even though the 5 seeds produced 4 genuinely different raw detection counts (66/47/68/55) because negative sampling differs per seed.** `postprocess_detections()` (auto params: `min_duration=0.0765s`, `merge_gap=0.1275s`, derived from ME1's own support-event median duration) made literally zero merges and dropped literally zero events across all 5 different raw detection sets tried. This means, on ME1 specifically, none of the raw detected event gaps ever fall within 127.5ms of each other, and no raw detected event is shorter than 76.5ms — i.e., the fragmentation problem this feature was built to fix (per its own header comment in `argus_core.py`) **does not appear to occur at this granularity on this clip**, not that fixing it doesn't help.

**Disposition: honest null result, but a structural one, not a "no effect" one — worth distinguishing.** Do not read this as "post-processing doesn't help" in general; read it as "the current auto-derived thresholds are too conservative to ever activate on ME1's actual detection pattern." A fair next step (not attempted here) would be testing explicit, larger `min_duration`/`merge_gap` values (or a different clip whose raw detections are known to fragment more) to see whether the mechanism does anything when it actually engages — this run only establishes that it currently doesn't engage on this clip under the auto heuristic.

Cross-ref: `ARGUS_TrackA_Implementation_Plan_2026-07-21.md` §1 (A3); `postprocess_detections()`/`suggest_postprocess_params()` docstrings in `argus_core.py` (2026-07-06, this session's own earlier feature).

---

## Day 21 (continued) — 2026-07-22 — A1's first "complete" run was NOT valid: disk-full failure on 11/43 clips, caught before citing the number

The `official_eval.py --resume` run reported an overall F-measure of **0.006%** — nowhere near any number this project has ever cited, which is itself the reason it got investigated rather than reported. Cause found in the run's own log, not a model problem: **D: filled to 100% (29GB, 1.4MB free) partway through**, and 11 of the 43 clips (pw5, pw6, pw7, pw8, pw9, and all 6 of RD_01–RD_06) failed outright with `OSError: [Errno 28] No space left on device` / `WinError 112`. `official_eval.py`'s per-clip try/except caught these and moved on without submitting predictions for those files, and the official evaluator correctly (per its own, unmodified logic) scores a file with zero submitted predictions as 100% missed — which is exactly why RD's per-file scores all landed on the exact `MIN_EVAL_VALUE` floor (1e-05) and why the harmonic-mean-of-subsets aggregation collapsed the *overall* number to near-zero. **The model was never actually evaluated on those 11 clips in that run; the 0.006% figure is an artifact of the disk filling up, not a measurement of anything real, and must not be cited or compared to any other number in this project.**

Root cause: `argus_app/detect_scratch/` (each clip gets an isolated scratch folder for feature extraction, per `build_eval_features()`) had silently grown to **17GB across 45 leftover folders** accumulated since 2026-07-02 — the code deletes and rebuilds a clip's OWN scratch folder on reuse, but never cleans up a DIFFERENT clip's folder once that clip is done, so every unique clip ever processed across this project's history (ablations, demo runs, prior validation attempts) left its folder behind. Confirmed every folder name matched the expected per-clip pattern before deleting (`ls detect_scratch/`) — deleted the lot, freeing 17GB → 29GB total, 17GB available.

**Re-running now** (`resume_run_2026-07-22_c.log`) with an active watch on both disk space and new failure lines while it proceeds, given ~26 hours of audio (the 11 failed clips) still needs to be processed and there's no guarantee 17GB is enough headroom without checking. Do not treat any full-eval number as final until this entry is followed up with a clean run (0 FAILED lines) and a fresh report JSON.

**Process lesson for future sessions:** if a real evaluation number looks absurd relative to every other number already in this project, investigate the run's own log before reporting it — the DCASE evaluator's harmonic-mean-of-subsets aggregation makes the overall score extremely sensitive to even one wholly-failed subset, so an infrastructure failure and a genuine domain-generalization collapse can look identical in the summary number alone. The per-file/per-subset breakdown (not just the headline number) is what actually distinguishes them.

Cross-ref: `ARGUS_TrackA_Implementation_Plan_2026-07-21.md` §1 (A1).

---

## Day 21 (continued) — 2026-07-22 — Adversarial-novelty research pass CORRECTS the TIM framing: the diagnosis isn't ours, and a published paper reports the OPPOSITE result

A 25-agent literature+novelty-verification workflow (`argus-research-cook`; full report: `ARGUS_Research_Directions_2026-07-22.md`) was run while the local machine is down (RAM stick burnt out 2026-07-22 — no Python/heavy compute until repaired). It found two things that revise, honestly, the framing written earlier today in `ARGUS_Problem_Selection_2026-07-22.md`:

1. **The "TIM assumes balanced classes → collapses under imbalance" mechanism is NOT an ARGUS discovery.** It is the central, proven thesis of **α-TIM (Veilleux, Boudiaf, Piantanida, Ben Ayed, NeurIPS 2021, arXiv:2204.11181)** — from the same lab that built TIM. Must be cited as the origin of the diagnosis, never claimed. This is the exact novelty risk flagged in the implementation plan (A4) — now confirmed real.
2. **A peer-reviewed paper reports TIM HELPING on this exact task.** Ijaz, Banoori & Koo (2024, *Bioengineering* 11(7):685 / PMC11274013) — *the same MDPI paper ARGUS already cited in its own literature collection §B.2 as the reason to try real TIM* — apply TIM *with* the marginal-entropy term to DCASE Task 5 and report F 29.6%→56.0% with precision **rising** 36→55%. ARGUS's own result (F 38→3, P 24→1, Day-21 entry above) **directly contradicts** a published result on the same benchmark.

**What this does NOT change:** the Day-21 TIM ablation numbers are real and correctly measured (N=5 paired, ΔF/SE=−12.8). The result stands as an observation.

**What it DOES change:** the *interpretation*. "We discovered TIM fails because of imbalance" is not defensible (both the mechanism and the fix-family are prior work). The honest, and actually stronger, hook is **the contradiction itself** — two teams, one method, one task, opposite signs, unexplained in the literature. Resolving it is now the gating first step, and it is *cheap*: on cached embeddings, rerun TIM with the marginal-entropy weight λ→0 and with the standard duration post-filter on/off (4 numbers, matrix-op scale). If the collapse vanishes when λ→0 or when the post-filter is enabled, ARGUS's collapse was a configuration artifact — still an honest, reportable finding, and it half-resolves the contradiction. **This experiment is deferred until the machine's RAM is repaired (or moved to Colab).**

**Genuinely novel white space the pass DID find (verification-backed, boundaries stated in the report):** (a) reconciling the Ijaz-vs-ARGUS contradiction + mapping the detection-grade (>99% background) imbalance breakpoint α-TIM's authors explicitly declined to test; (b) **Positive-Unlabeled (PU/nnPU) reframing of 5-shot detection** — the DCASE POS/NEG/UNK scheme is literally a PU generating process, and this method-cell is empty for Task 5 (cleanest white space found); (c) a **verification-cost + calibration evaluation protocol** for few-shot detection (first instantiation + a possible F-score rank-inversion). All three are inference-only on cached embeddings — feasible on one consumer GPU, no retrain.

**Process win:** this is exactly what the adversarial-verification discipline is for. The earlier same-day framing over-credited ARGUS with a known diagnosis; the verification pass caught it *before* it went on a poster, where a judge who knows α-TIM would have caught it instead. Log stands as the honest record of both the over-claim and its correction.

Cross-ref: `ARGUS_Research_Directions_2026-07-22.md` (full report), `ARGUS_Problem_Selection_2026-07-22.md` (the framing this corrects), `ARGUS_TrackA_Literature_Collection_2026-07-21.md` §B.2 (the Ijaz paper).

---

## Day 22 — 2026-07-29 (experiment executed on Colab on/after 2026-07-23) — Reconciliation experiment: marg_weight=0 fully rescues the collapse, and beats the plain baseline

**Date-accuracy note (correction):** this entry was initially drafted under the heading "Day 21 (continued) — 2026-07-22," which was wrong — that was the date of the *preceding* local-CPU work, not of this run. The Colab log's own `cloudflared version 2026.7.3 (built 2026-07-23-09:58 UTC)` line pins the run to on/after 2026-07-23, and the session clock at logging time read 2026-07-29. Exact run date within that window is not recoverable from the pasted output, so it is stated as a range rather than guessed. Corrected here rather than silently rewritten, per this notebook's append-only convention.

**Real numbers** (`tim_reconcile_experiment.py --clip ME1 --seeds 5`, Colab T4 GPU, matched-pair):

| cell | F (mean±sd) | P | R |
|---|---|---|---|
| marg=1.0 (default) + postproc OFF [reproduces the collapse] | 2.8±0.3 | 1.4 | 89.1 |
| **marg=0.0 (no balance pull) + postproc OFF** | **49.9±9.5** | **33.8** | 100.0 |
| marg=1.0 (default) + postproc ON (auto) | 2.8±0.3 | 1.4 | 89.1 |
| marg=0.0 (no balance pull) + postproc ON (auto) | 49.9±9.5 | 33.8 | 100.0 |

**Cross-device sanity check:** the marg=1.0/postproc-off cell (F=2.8±0.3, P=1.4, R=89.1) closely reproduces the local-CPU Day-21 result (F=2.7±0.6, P=1.4, R=83.6) — the collapse is real and reproducible across hardware (AMD DirectML CPU path vs. Colab CUDA T4), not a device-specific artifact.

**Headline result: disabling the marginal-entropy term (marg_weight=0) fully rescues the collapse — and the fixed version BEATS the plain non-TIM baseline**, not just "stops hurting." Day-21's plain baseline (`tim=False`, from `tim_ablation_paired.py`): F=38.2±6.0, P=23.8, R=100.0. With marg_weight=0: F=49.9±9.5, P=33.8, R=100.0 — a real, positive lever. This is a direct, causal confirmation (an on/off manipulation, not a correlational story) that the marginal-entropy/balanced-class-assumption term is what drives the collapse — exactly the mechanism diagnosed in `ARGUS_Research_Directions_2026-07-22.md` Direction 1, now empirically demonstrated, not just hypothesized.

**Postprocess made exactly zero difference in every condition** (identical to 1 decimal place, both marg settings) — consistent with the earlier `postprocess_ablation.py` finding: the auto-derived thresholds simply never trigger on ME1's detection pattern, regardless of upstream method.

**Critical caveat — do not repeat the Day-19/20 mistake.** This is N=5 on ME1 alone, and ME1 is already known (from the regime-adaptive-gate story) to be an atypical extreme over-predictor (R pinned near 100%, "free" precision-recall tradeoff) whose lever results have previously NOT generalized to other clips. **This is a strong, promising go/no-go signal to proceed to Direction 1's multi-clip breakpoint-map step — it is NOT yet evidence that marg_weight=0 generally fixes TIM for detection.** Next step: repeat this exact marg=1.0-vs-0.0 comparison across the same ≥8–12 clip spread `gate_generalization.py` (Day 20) used, before drawing any general conclusion. Do not put this on a poster as "solved" until that's done.

**Process note:** ran on Colab after the local machine's RAM stick failed (see the Colab re-upload work same day) — first real validation that the Colab pipeline can pick up exactly where local execution left off, using the same code, same seeds, same clip.

Cross-ref: `ARGUS_Research_Directions_2026-07-22.md` (Direction 1, the experiment this executes), `tim_ablation_paired.py` (the Day-21 baseline this reproduces and beats), `gate_generalization.py` (the multi-clip discipline the next step must follow).

---

## Day 23 — 2026-07-30 — NEW CODEBASE (BirdCLEF + Perch two-stage), and a leakage bug that retracts the previous session's headline number

**Read this first — scope change.** Days 1–22 are all `argus_app/` (DCASE Task 5, frozen ProtoNet baseline, TIM/imbalance thread). This entry covers a **different codebase** (`D:\arrgusdadai\argus.py` + `arguswala.py`), which is:

| | Days 1–22 (`argus_app/`) | Day 23 (`arrgusdadai/`) |
|---|---|---|
| Data | DCASE Task 5, 43-clip validation set | **BirdCLEF 2024**, target `whbsho3` (White-bellied Sholakili) |
| Backbone | **Frozen** official ProtoNet/ResNet, never retrained | Custom CNN trained from scratch (40 species, 1000 episodes) |
| Adaptation | Inference-only levers, additive/default-off | **Per-file fine-tuning** (deepcopy + 250 Adam steps) — mutates the encoder |
| Metric | Official DCASE evaluator (Path A) / interval-overlap (Path C) | Planted-call event F1 with this repo's own matcher |

These threads do **not** share numbers, code, or discipline. PRD v2.0 §3.2's number-scope rule applies with full force — see the scope warning at the bottom of this entry.

### 1. What was built

A two-stage architecture: stage 1 (the existing few-shot detector) **proposes** candidate events; stage 2 (**Google Perch** `bird-vocalization-classifier` v8, frozen, 1280-d embeddings) **verifies** them. Note this is not a re-try of `argus.py` ablation #1 — that rejected Perch as a *detector* ("clip-level model, 5 s window, blind to faint calls"); here it only ever scores already-localised candidates.

Verification is a logistic regression on Perch embeddings, and decoy species for the specificity control are now chosen by **cosine distance to the target in Perch space** (the 4 nearest of 30 candidates) rather than by recording count — i.e. genuinely confusable species instead of an arbitrarily easy set.

### 2. 🔴 Leakage bug — the previous session's 0.34 specificity ratio is RETRACTED

An earlier run this session reported specificity ratio **1.31 → 0.34**, i.e. "crossed into the species-specific bucket." **That number was produced by a leaking classifier and must not be cited.**

Cause: `perch_clf` was fit on `support` **+ `target_pool`** — but `target_pool` is exactly the pool the evaluation plants its test calls from. The verifier had memorised the specific test calls, not learned the species.

Fix: calibrate on `support` only (the 5 labelled events, injected into real soundscape beds at the evaluation SNRs with centre jitter), leaving `target_pool` purely held-out. Also made calibration and evaluation beds disjoint, and matched the calibration domain to inference (previously the classifier trained on clean Xeno-Canto peaks and was applied to noisy soundscape events).

**Effect of removing the leak: ratio 0.34 → 1.12.** The entire apparent gain was leakage. Two further fixes went in alongside: Perch was being fed a 1 s call plus 4 s of *trailing digital silence* (its documented convention is centred padding, and in a soundscape real context is available — it now gets a real 5 s window), and `LogisticRegression(C=0.1)` replaced the default, since 1280-d embeddings with ~100 samples separate perfectly at C=1 and saturate `predict_proba` into a hard veto.

Logging this the way Day 21's disk-full incident was logged: the number looked good, the number was wrong, and it was caught before it went on a poster.

### 3. Real numbers — N=1 seed (seed 7), 10 beds, 120 planted calls, strict IoU≥0.3

| fusion rule | AP | best F1 | prec | rec | FP/TP | ratio | **specAUC** |
|---|---|---|---|---|---|---|---|
| stage-1 only (p) | 0.533 | 0.599 | 71.3% | 51.7% | 0.4 | 1.12 | **0.477** |
| **perch only (v)** | 0.536 | **0.622** | **82.2%** | 50.0% | **0.2** | **0.73** | **0.580** |
| p × v | 0.539 | 0.608 | 73.8% | 51.7% | 0.3 | 1.09 | 0.490 |
| p × √v | 0.535 | 0.605 | 72.9% | 51.7% | 0.4 | 1.11 | 0.483 |
| **√p × v** | **0.546** | 0.607 | 69.1% | 54.2% | 0.3 | 1.11 | **0.500** |
| √(p·v) | 0.539 | 0.608 | 73.8% | 51.7% | 0.3 | 1.09 | 0.490 |

`specAUC` = P(a random target call scores above a random decoy call), threshold-free. 0.50 = cannot distinguish target from decoy at all. Introduced because the hit-rate ratio is measured at the best-F1 threshold, which is **not binding** on the +6 dB planted calls in the specificity pass — five of six variants returned bit-identical ratios, which is a measurement artifact, not agreement.

### 4. Three findings

**(a) Stage-1's species discrimination is at or below chance.** specAUC **0.477** — it scores decoy species marginally *higher* than the target. This is the threshold-free confirmation of `argus.py` ablations #10/#12 (bird-activity detector, and the frozen encoder is equally unspecific, so the fault is in the representation). A detector with F1 ≈ 0.60 on event detection carries **zero** species information in its scores.

**(b) 🟢 Rank inversion — this is the result worth having.** By **AP**, the best variant is `√p × v` (0.546) — and its specAUC is **exactly 0.500**, i.e. no species discrimination whatsoever. By **FP-per-TP** and **specAUC**, the best variant is `perch only` (0.2 and 0.580). *The standard detection metric and the deployment-relevant metric rank these methods in opposite orders.* Optimising AP here would have selected the least species-specific model in the set. This is a direct test of **PRD v2.0 H4** ("method rankings under verification-cost/calibration disagree with the F-score ranking — Status: untested") and of RQ3, on our own data.

**(c) Perch verification halves verification burden at matched recall.** FP/TP 0.4 → 0.2, precision 71.3% → 82.2%, recall essentially unchanged (51.7% → 50.0%). That is PRD v2.0 §2.3's "verification cost" currency — the constraint the deployment literature names as the #1 adoption blocker.

**Also: `argus.py` ablation #1's prediction did NOT transfer.** Recall by planted-call SNR:

| variant | +9 dB | +6 | +3 | 0 | −3 | −6 |
|---|---|---|---|---|---|---|
| stage-1 only (p) | 70% | 70% | 50% | 50% | 40% | 30% |
| perch only (v) | 55% | 75% | 50% | 45% | 40% | 35% |

Perch verification does **not** collapse at low SNR — at −6 dB it is marginally *better* (35% vs 30%). The concern that Perch is "blind to faint calls" applies to its use as a detector, not as a verifier of already-localised candidates. Caveat: n=20 per SNR cell, underpowered per PRD §7.4; the +9 dB dip (55% vs 70%) is 11/20 vs 14/20 and is not distinguishable from noise.

### 5. What is owed before any of this is claimed

- **N=1.** Every number above is a single seed. PRD §7.2 requires N≥5 matched-pair and "treat |Δ/SE| < ~2 as not evidence." The F1 spread across variants (0.599–0.622) is well inside the ±3–10 point run-to-run noise this notebook has documented since Day 3. `arguswala.py` has since been changed to loop `SEEDS = (7, 8, 9)` and print paired Δ/SE for `perch only` vs `stage-1`; **not yet run.**
- **One target species.** The multi-species analogue of the §7.3 twelve-clip rule has not been done. Day 19→20 is the standing warning: `whbsho3` alone may be this thread's ME1.
- **Novelty unchecked.** Perch embeddings + a light classifier is established practice (Nature Sci Rep 2023, "Global birdsong embeddings…"); propose-then-verify is decades old. The defensible object is more likely the **measured rank inversion** than the architecture. Run the §3.4 adversarial search before this goes near a poster.

### 6. 🔴 Scope warning — do NOT compare this F1 to 65.2%

The F1 above (0.599–0.622) is on **BirdCLEF planted calls with this repo's own matcher**. DCASE SOTA 65.2% is the **official DCASE evaluator on the DCASE Task 5 evaluation set**. Per PRD v2.0 §3.2 these are different scoring paths — this is in fact a *fifth* path, beyond the four the PRD enumerates. "We reached 62%, SOTA is 65%" is a category error and a judge who knows this benchmark will catch it instantly. If a leaderboard comparison is wanted, it has to come from `argus_app/` and PRD §13's S1 (official full-eval baseline), which remains 🔴 uncomputed.

Cross-ref: PRD v2.0 §2.3 (verification cost), §3.2 (number scope), §7.2/§7.3 (seeds, multi-clip), H4 (rank inversion), RQ2/RQ3, §4.5 Directions 2–3; `argus.py` ablations #1, #10, #12.

---

## Day 24 — 2026-08-06 — Perch 1.0 → 2.0 upgrade, verified against real downloaded weights before touching the pipeline

**Why:** Day 23 ran Perch v1 (`bird-vocalization-classifier` v8, 1280-d). Google shipped
Perch 2.0 in Aug 2025 (BirdSet AUROC 0.839 → 0.908, no fine-tuning) — a straight embedding
upgrade, but not a config flip. `perch-hoplite` source was read directly (no assumptions):
v2 exposes `signatures['serving_default']`, **not** `infer_tf` — calling the old convention
on the new model raises `AttributeError` on the first embed.

**Verification before deployment, not after:** downloaded the real 413 MB
`perch_v2_no_dft.onnx` (public, Apache-2.0, `justinchuby/Perch-onnx` on Hugging Face) and ran
it locally with `onnxruntime` against synthetic audio. Six checks, all against real weights:
input `'inputs'` `[batch,160000]` matches `PERCH_LEN` exactly; output `embedding` is
1536-d; measured latency ~156 ms/clip (batch=16, CPU); **peak-normalisation changes the
real embedding materially** (mean abs delta 0.065) — independent confirmation that v1
feeding Perch raw, un-normalised waveforms was a live bug, not a theoretical one.

**Decision:** `PERCH_VERSION="v2"`, `PERCH_PEAK_NORM=0.25` (matches the official wrapper's
`normalize_audio`, `target_peak=0.25`). Both dispatch conventions (`infer_tf` / `signatures`)
kept, auto-detected — a mis-attached Kaggle model degrades to a clear `RuntimeError` naming
the mismatch, not a silent wrong-shape bug. `PERCH_BACKEND="onnx"` wired in as an opt-in
CPU path for the hardware-deployment roadmap, default `"tf"` — zero behaviour change unless
explicitly switched.

**Multi-species rebuild, same session:** `arguswala_setup.py`/`arguswala.py` → `_v2.py`.
The rewrite fixed the hard-coded `s != "whbsho3"` exclusion (a leak: any second target would
have trained the encoder on itself), and derived the held-out target pool from the shots
*actually chosen* rather than `files[N_SHOT:]` positionally — under the `rated` strategy the
positional slice does not guarantee no overlap with the shots.

Cross-ref: PRD v2.0 §7.3 (multi-species not yet run — this rebuild is the prerequisite, not
the run itself). 94 local regression tests written and passing before any of this touched
Kaggle (`_test_v2_logic.py`); none of it had run against real infrastructure yet.

---

## Day 25 — 2026-08-13 — First real Kaggle run: pipeline validates end-to-end on live infra, and a real discrepancy surfaces

**First contact with real infrastructure.** Every fix in Day 24 was written against the
`perch-hoplite` source and tested with synthetic data — untested against the actual Kaggle
GPU environment until now. It worked: `calling convention: serving_default (Perch 2.x)`
printed correctly (2× Tesla T4), `Perch embedding dim: 1536 (detected v2, requested v2)`
confirmed the version guard, both encoder-bank leakage assertions passed on real data.

**2-species smoke test** (`whbsho3` forced primary, `bcnher` — 500 recordings, highest
count in corpus — filled the second `SMOKE_TEST` slot): 20.5 min wall clock for the full
per-species eval loop. **This settles PRD-adjacent budget planning:** scaling to the full
pre-registered 8–12 species at 5 seeds projects to 2–3 h, nowhere near the ~30 GPU-h/week
ceiling.

**🔴 Discrepancy, flagged not resolved:** `whbsho3` stage-1-alone AUC came back **0.517**.
Day 23's retired number was 0.459 (below chance). Candidate cause identified but not yet
isolated: decoys are ranked by Perch-embedding proximity, so the v1→v2 swap re-drew the
entire decoy set independent of any bug fix — the two numbers may simply not be measuring
the same test. Not resolved this session; see Day 26/28.

---

## Day 26 — 2026-08-14 — Encoder-seed widening rules out pure training noise; the ladder's first real run dies at 6/16 with zero data on the one technique that mattered most

**`ENCODER_SEEDS` widened `(0,)→(0,1,2)`** to test one candidate explanation for Day 25's
gap: retraining stochasticity on an *identical* bank (bank composition is fixed by a
hardcoded `seed=0` in `build_banks`, independent of `ENCODER_SEEDS` — this measures training
noise only, not bank-composition variance). Result: `whbsho3` stage-1 spread **0.020**
across 3 seeds (0.501/0.521/0.516) — real, but far smaller than the 0.459-vs-0.517 gap.
Training noise alone does not explain it. Bank composition (Day 25's candidate cause)
remains the live hypothesis.

**Ablation ladder, first attempt: `FULL_FACTORIAL=True`, 16 configs.** Killed after config
6/16 — no traceback, no OOM signature, consistent ~800s/config timing right up to the stop.
Most likely a session limit or disconnect, not a code fault; cause not conclusively
identified.

**🔴 Root-caused, not just unlucky:** the factorial was built with
`itertools.product(["logreg","proto"], ...)` — `VERIFIER` as the *slowest*-varying factor.
All 8 `logreg` configs ran before any `proto` config. **The one technique this whole ladder
exists to test — the prototypical probe — had zero data after the crash.** This is a design
bug, not bad luck; recorded as such rather than blamed on the infrastructure.

**Fix, before re-running anything:** configs reordered by Hamming distance from baseline
(first 5 configs now cover every single-factor main effect, truncation-safe); checkpointing
added so each config appends to CSV on completion — a killed session resumes, it does not
restart; `TimeFilterAug` dropped from the design (16→8 configs) after this session's
cumulative-ladder data (not shown here, see repo history) showed it damaging *stage-1-alone*
AUC — verifier-independent, so unambiguously a fine-tuning-time effect, not a verifier
interaction. Regression test written asserting the OLD ordering's first 6 configs contain
zero `proto` entries (it does) and the NEW ordering's first 6 do not (they don't) — the bug
is now unable to silently return.

---

## Day 27 — 2026-08-15 — Ladder rebuilt and completed clean: all four imported techniques rejected, baseline wins outright, noise floor formally measured

**Full 8-config factorial completed, checkpointed, no crash.** 130 min.
`ENCODER_SEEDS` widening also re-run (3 seeds, 2 species): spread 0.020–0.025 across every
species/stage combination — consistent with Day 26, confirms this is a real, repeatable
floor rather than a one-off.

**🟢 The noise floor, formally measured, not assumed.** Baseline (`pcen|logreg`) repeated
across the 3 encoder seeds: **0.719 / 0.724 / 0.703** — spread **0.021**. Combined with an
earlier cross-session observation (same config, two separate Kaggle sessions: 0.696 and
0.731 — 0.035 apart, from GPU non-determinism in cuDNN's backward pass, not from anything in
this codebase), the working floor is **~0.025 within a session, ~0.035 across sessions.**
Every effect below is judged against this, not against a within-run SE.

**🔴 A verdict retracted, then correctly re-earned.** The FIRST ladder attempt (pre-fix)
reported the prototypical probe as harmful at `d/SE=-3.02` — computed across 2 species
*within one run*, blind to the cross-run floor just measured. That verdict is retracted:
`-0.005` is inside a `0.035` floor, not evidence. The REBUILT ladder measured it again,
properly: **`-0.034` AUC** — still inside the floor on AUC alone (edge case), but decisive
on **false-alarm rate**: `pcen` 7.9→18.0/hr, `log-Mel` 5.2→35.6/hr, both directions, plus F1
down in both. Same final call (reject) as the flawed first attempt, but earned honestly the
second time, on the metric that actually separates it.

**All four results:**

| technique | source | effect (AUC) | vs 0.035 floor |
|---|---|---|---|
| TimeFilterAug | DCASE 2023 winner | −0.155 | 4.4× — real |
| log-Mel front-end | DCASE 2024 winner | −0.114 | 3.3× — real |
| Hard negatives | Liang et al. 2024 | −0.047 | 1.3× — real, marginal |
| Prototypical probe | Perch 2.0 paper, Table 3 | −0.034 | edge — decided on FA/hr instead |

The untouched baseline beat every published technique tried against it. Reported as the
honest result it is, not re-tuned to find a win: in a five-shot regime with a 250-step
fine-tune, techniques validated at full-scale training budgets did not transfer.

**What is owed, unchanged from Day 23:** this is still 2–3 species. The ladder's ranking of
techniques is an assumption carried into the centrepiece run below, not independently
re-verified at 10-species scale.

---

## Day 28 — 2026-08-16 — THE CENTREPIECE: full 10-species run. "Below chance" does not survive at scale — retracted and replaced with a stronger result. Two other in-session claims also retracted. One real new lead.

**`SMOKE_TEST=False`.** Winning ladder config (Day 27) was already every Cell-2 default —
no config change needed beyond the flag. 10 species (pre-registered rule: `whbsho3` forced
primary + top-9 by recording count), 3 encoder seeds × 3 eval seeds each.

**🟢 The core claim, properly earned this time — n=10, not n=2.**

| | n=2 (Day 25/27) | n=10 (this entry) |
|---|---|---|
| mean gain | +0.202 | **+0.165** |
| SE | 0.070 | **0.019** |
| d/SE | 2.77 | **8.83** |
| species improved | 2/2 | **10/10** |
| gain exceeds noise floor, per species | not tested | **10/10** |

Every individual species' gain clears the 0.025 in-session floor on its own — not a few
noisy outliers riding a mean. FA/hr drops are large throughout (best case 55.2→0.7/hr).

**🔴 "Stage-1 sits at chance" — retracted as a general claim.** Range across 10 species:
**0.469–0.694**. Two below 0.50, four within ±0.05 of it, **six clearly above it**
(0.62–0.69). The claim only held for the 2 species in the smoke test, and — worth stating
plainly rather than letting it be found — **those two were the two lowest-scoring species
in the full set of ten**, `whbsho3` lowest of all. Corrected claim, and it is the stronger
one: verification helps every species tested, by a margin that clears the noise floor in
all ten cases, independent of whether that species' baseline was at chance or not.

**🔴 Day 25's discrepancy, resolved (partially) — and an honest coincidence flagged.**
`whbsho3` stage-1 across every correct-pipeline measurement to date: 0.517, 0.509 (2-target
runs) → **0.469** (0.456–0.480 across 3 seeds, this 10-target run). The 3-seed spread here
(0.024) matches the established training-noise floor — so within THIS run it's stable. The
move between runs is the OTHER candidate cause from Day 25/26, now confirmed: decoy pool
composition depends on how many other species are excluded as targets. Checked directly —
`decoy_sim_top1` for `whbsho3` is 0.575 in the 2-target run, 0.419 here (a *lower* peak
similarity, nominally easier), yet the score came out harder. One scalar decoy-similarity
number does not capture full decoy-set difficulty. **0.469 sits close to the Day 23 retired
0.459 — flagged explicitly as coincidence, not relapse:** the retired number came from a
different Perch version, missing peak-norm, and a different decoy pool entirely; this
number comes from the corrected pipeline measured three times consistently. Both facts are
true at once and both are now stated on the public site rather than one hiding the other.

**🔴 Two more retractions, both from claims made in this project during the smoke-test
phase, before the wider sweep re-ran on 3 species instead of 2:**

- *"Rated shot-selection wins."* Spread across strategies (0.068) is smaller than the
  measured run-to-run sd at this sample size (0.024 mean, up to 0.062) — not separable.
  Per-species winners actively disagree: `rated` for barswa, `random` for bcnher, `diverse`
  for whbsho3. Only "first five, uncurated" (the DCASE default) reads consistently worse.
- *"Plateau at 5 shots."* Step gains 1→3 **+0.061**, 3→5 **+0.031**, 5→10 **+0.024**, against
  a mean sd of 0.043. Only the first step clearly clears noise. Honest claim: one shot is
  clearly insufficient; beyond three, this study cannot resolve further gains from noise.

**🟢 New lead — the difficulty-factor question Nolasco et al. (2023, §4.1) raised and could
not resolve with multivariate regression on their own dataset.** Correlated 6 per-species
covariates against stage-1 AUC and verification gain. **Call stereotypy** (how self-similar
a species' calls are) is the standout: `r=0.590` vs stage-1 AUC, `r=-0.557` vs gain — more
stereotyped calls score higher before verification and gain less from it, which is
mechanistically the expected direction (a single embedding-space prototype fits a
consistent call well; it blurs across a variable one). `n_recordings` also correlates
(0.531) but is confounded — 9/10 species are tied at 500 recordings, `whbsho3` alone differs
(22), so that correlation is one data point, not a relationship; disregarded. n=10 is below
the ~0.63 two-SE threshold for this sample size — directional, not proof. `whbsho3` has the
lowest stereotypy (0.268) *and* the lowest stage-1 AUC in the set, consistent with the
pattern.

**What is owed, going forward:** the ladder's technique ranking (Day 27) has not been
re-verified at 10-species scale — carried as an assumption. The stereotypy correlation needs
either more species or an experiment that manipulates it directly (e.g. synthetic bimodal
decoys, already validated as a mechanism locally — see `_test_ladder.py`'s prototype-probe
synthetic test) before it's anything stronger than a lead. Both `site/index.html` and this
notebook were updated together, same day, from the same source CSVs — no daylight between
the public-facing numbers and this log.

Cross-ref: PRD v2.0 §7.3 (multi-species — now run, n=10); Nolasco et al. 2023 §4.1
(difficulty-factor question — stereotypy is the first candidate answer this project has for
it); Day 23 §5 ("one target species... may be this thread's ME1" — resolved: it wasn't, but
it was the hardest case in the set, which is itself worth knowing).

---

## Standing notes for future entries

- Add a new dated entry per work session, oldest to newest — don't overwrite previous entries.
- Record: what was attempted, what broke, the actual numbers observed, and the decision made — not just the final happy-path outcome.
- Cross-reference the PRD section number when an entry resolves or informs a PRD risk/requirement (e.g., PRD §12 success criteria, §13 risk register).
