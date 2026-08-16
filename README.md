# ARGUS

**Few-shot bioacoustic detection for an endangered Western Ghats bird — learn a species from five examples, then find it in hours of unlabelled forest audio.**

ARGUS is trained to recognise *Sholicola albiventris* (White-bellied Sholakili), a Western Ghats endemic with almost no labelled recordings available anywhere — and to generalise the same method to any species with just five example calls. It's built on BirdCLEF 2024 data and verified with Google's Perch model.

**[Live demo →](#) <!-- TODO: add your Vercel URL here -->**

---

## The finding

A model that reliably detects *a* bird call can still carry almost no information about *which* species it heard. We measured this directly and fixed it.

Every result below is from the actual pipeline output, not illustrative — see [`ARGUS_TrackA_Lab_Notebook.md`](ARGUS_TrackA_Lab_Notebook.md) for the full dated log and [`argus_multispecies_results.csv`](argus_multispecies_results.csv) for the raw numbers.

| | Stage 1 alone (matching) | + Verification (Perch) |
|---|---|---|
| Mean species-discrimination AUC, 10 species | 0.585 | **0.751** |
| Species improved by verification | — | **10 / 10** |
| Confidence (Δ / SE) | — | **8.83** |
| False alarms / hour, best case | up to 78 | as low as **0.7** |

0.50 AUC means the model cannot tell the target species from an acoustically similar decoy at all — a coin flip. Stage 1 alone lands there or close to it for several species; adding a Perch-embedding verification step lifts every single one of the ten species tested, by a margin larger than the measured run-to-run noise floor in every case.

We also found what *doesn't* help: four published techniques (a DCASE-2023-winning augmentation, a DCASE-2024-winning front-end, hard-negative mining, and the prototypical-probe design from Google's own Perch 2.0 paper) were each tested against a measured noise floor. All four made things worse in this five-shot, short-fine-tune regime. The untouched baseline beat every one of them. That's reported as the honest result it is, not tuned away.

## How it works

```
5 example recordings
        │
        ▼
┌───────────────────┐      ┌──────────────────────┐
│  Stage 1 — propose │ ───▶ │  Stage 2 — verify     │ ───▶ confirmed detection
│  Custom CNN,       │      │  Google Perch v2      │
│  prototypical       │      │  embeddings + a       │
│  few-shot, fine-    │      │  trained classifier   │
│  tuned per clip     │      │                       │
└───────────────────┘      └──────────────────────┘
   fast, high recall,          slower, catches what
   weak on species ID          stage 1 gets wrong
```

- **Stage 1** is a small CNN trained from scratch on 40 non-target species (prototypical-network style), then fine-tuned in seconds on the five labelled examples for whatever species you're pointing it at. It's fast and rarely misses a candidate — but on its own it barely knows *which* bird it heard.
- **Stage 2** takes every candidate stage 1 flags, embeds it with Google's Perch v2, and scores it with a classifier trained on the same five examples. This is what recovers species identity.

Decoy species for testing are picked automatically by acoustic similarity in Perch's own embedding space — the four *hardest*, most confusable species, not an easy set.

## Try it live

The demo (`site/index.html`) is a single-file, no-framework web app that connects to a live model running on Kaggle:

- Record straight from your microphone in the browser, or upload a clip
- Runs the real two-stage pipeline, not a canned response
- Every chart on the results pages is real data from the runs above — interactive, not static images

To run your own backend: open [`ARGUS_Kaggle_Notebook.ipynb`](ARGUS_Kaggle_Notebook.ipynb) on Kaggle, attach the BirdCLEF 2024 competition data and a Perch v2 model, enable GPU + Internet, and run the cells in order. It exposes a public URL via Cloudflare Tunnel that the demo site connects to.

## Repository layout

```
arguswala_setup_v2.py     Cell 1 — data loading, decoy selection, verifier calibration
arguswala_v2.py           Cell 2 — two-stage detection across all target species
arguswala_sweeps.py       Cell 3 — shot-count and shot-selection-strategy experiments
arguswala_ladder.py       Cell 4 — ablation ladder (tests published techniques honestly)
ARGUS_Kaggle_Notebook.ipynb   The four cells above, packaged to import directly into Kaggle

site/index.html           Live demo — mic capture, results dashboard, all analysis charts

ARGUS_TrackA_Lab_Notebook.md   Full dated engineering log — every bug, every retraction, every real number
ARGUS_Literature_Review.md     ~20 papers reviewed and positioned against this project
ARGUS_Roadmap_Aug_Sept_2026.md What's automated, what's manual, what's next
ARGUS_Hardware_Deployment_Roadmap.md   Honest split: what's built vs. what's planned for field deployment

argus_multispecies_results.csv   Raw output of the full 10-species run
```

Files prefixed `_` are working/scratch scripts, not part of the pipeline itself.

## Built with

Python · PyTorch (stage-1 encoder, trained from scratch) · TensorFlow / TensorFlow Hub (Perch v2) · scikit-learn · ONNX Runtime · librosa · FastAPI · vanilla JavaScript (hand-rolled SVG charts, Web Audio API — no charting library, no frontend framework) · Kaggle (GPU training/inference)

## Honesty, by design

This project keeps a public record of what didn't work alongside what did — including retracted claims, a discovered noise floor that overturned an earlier conclusion, and real bugs found and fixed with regression tests. See the "What did not work" section of the live demo, or `ARGUS_TrackA_Lab_Notebook.md` for the full account. We think that record is worth more than a version of this README that only shows the wins.

## Team

Subham Sai Samal · Vlkramzharis
Mentor: Mrs. Jolly Verghese, Jawahar Vidyalaya Senior Secondary School

<!-- TODO: add a LICENSE file and reference it here if you want this repo publicly licensed -->
