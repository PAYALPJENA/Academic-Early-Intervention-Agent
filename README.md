# Academic Early-Intervention Framework

Predicts a student's academic risk (High / Medium / Low) before the final examination from 23 mid-semester
attributes, and turns the prediction into support actions for faculty advisors through a Streamlit dashboard.

## Folder structure

| Folder / file | Purpose |
|---|---|
| `app.py` | Streamlit dashboard for advisors |
| `src/risk_prediction.py` | Prediction module used by the app (loads the validation-selected pipeline model and its validation thresholds; falls back to `old_models/`) |
| `src/pipeline/` | Reproducible experiment code: config, data and labels, models, metrics, plots, operating points |
| `src/pipeline/run_pipeline.py` | Runs the full experiment (70/15/15 split, 5-fold CV × 5 seeds, both decision rules) |
| `data/raw/` | User-supplied *Student Performance and Behaviour* dataset; not distributed here pending license and privacy review |
| `data/processed/` | 80/20 arrays from the initial experiment (used only by the archived notebooks) |
| `models/*.pkl` | Initial-experiment model artifacts |
| `models/pipeline/` | Models saved by the pipeline (first seed) |
| `old_models/*.pkl` | Legacy dashboard fallback artifacts |
| `results/` | Pipeline results (`*.json`) and per-seed checkpoints in `runs/`; the paper is built from these |
| `figures/` | Figures produced by the pipeline |
| `report/` | Paper builder (`build_paper.js`, `figs.py`, `assets/`) and the existing DOCX output; review and regenerate the paper before release |
| `brag-output/` | 23-second launch video with voiceover, thumbnail and share caption |
| `DEMO_VIDEO_SCRIPT.md` | Script for a narrated demo video |
| `_archive/` | Superseded files (original notebooks, old paper, logs, video sources); see `_archive/MANIFEST.md` |

## Setup

Python 3.13 and Node.js are required.

```
py -3.13 -m pip install -r requirements.txt
py -3.13 -m pip install -r requirements-extras.txt  # optional model families
cd report && npm ci && cd ..
```

## Run the app

```
py -3.13 -m streamlit run app.py
```

## Dashboard model

The dashboard serves the validation-selected pipeline model: **CatBoost, seed 0**. It reads the selected model name from `results/selection.json` and that model's seed-0 validation thresholds from `results/thresholds_validation.json`.

## Reproduce the results

```
py -3.13 src/pipeline/run_pipeline.py                          # full 12-model run
py -3.13 src/pipeline/run_pipeline.py --operating-points-only   # recompute operating points from saved predictions
py -3.13 src/pipeline/run_pipeline.py --time-budget 30          # stop cleanly after 30 min; rerun to continue
py -3.13 src/pipeline/run_pipeline.py --device cuda --time-budget 30  # optional CUDA for supported models
```

`--device` accepts `cpu`, `cuda` or `auto` and defaults to `cpu`. LightGBM stays on CPU. The CUDA run requires a CUDA-enabled PyTorch environment and GPU-capable XGBoost/CatBoost installs. CUDA fold checkpoints are written under `results/runs/folds/cuda/` so existing CPU fold checkpoints remain intact.

Runs resume from the checkpoints in `results/runs/`, so completed work is reused. Labels are checked on every run
(High Risk < 40, Medium 40–69, Low ≥ 70 on the final exam score).

The saved comparison contains all 12 models and five seeds. The pipeline resumes from saved run records and predictions; a fresh run requires the raw dataset and can take several hours. Install `requirements-extras.txt` for XGBoost, LightGBM, PyTorch, and the supplementary imbalance experiment.

## Rebuild the paper

```
Push-Location report; py -3.13 figs.py; Pop-Location  # review generated figures before replacing report assets
node report/build_paper.js   # writes report/Academic_Early_Intervention_Framework_Paper.docx
```

All numbers in the paper are read from `results/*.json`. To make the PDF, open the `.docx` in Word and save it
as PDF.
