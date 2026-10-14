# Colab runbook

## Before opening Colab

Create `MyDrive/AcademicEarlyIntervention/` and upload:

1. `Academic-Early-Intervention-Agent.zip`, with a top-level `Academic-Early-Intervention-Agent/` folder. Include `src/pipeline/run_pipeline.py`, `src/`, `requirements*.txt`, the current `results/` directory (including saved model runs, predictions, tuning JSON, and XGBoost CPU fold checkpoints), and `figures/`. The audit created `results_backup_oct11/` locally; it is a rollback copy and is not needed to run Colab.
2. `Student Performance and Behaviour.zip`, containing exactly one `Student Performance and Behaviour.csv`.

The notebook expects those exact archive names. It expands the project and dataset under the Drive folder, placing the CSV at `data/raw/Student Performance and Behaviour.csv`. It works from that Drive copy, so both model and fold checkpoints survive a disconnect. Do not upload the local project by itself without its existing `results/` cache.

## Cell order

Open `notebooks/colab_run.ipynb` and run cells from top to bottom:

1. Mount Drive and expand the archives. Extraction is skipped on later runs if the project and CSV already exist.
2. Install `requirements.txt` and `requirements-extras.txt`.
3. In Colab, select **Runtime → Change runtime type → T4 GPU** (or another available GPU), then confirm `torch.cuda.is_available()` and note the GPU name.
4. Run one 45-minute pipeline chunk. Re-run this cell to resume until it reports exit code 0.
5. Export `results/` and `figures/` plus any first-seed artifacts for the five added models to a timestamped Drive folder and zip for download.

The run command uses the full default model list. Existing complete model/seed outputs and tuning files are reused. Keep `--device cuda` the same after every reconnect. CUDA CV checkpoints are stored in `results/runs/folds/cuda/`, leaving the existing CPU XGBoost fold files in place. The five existing XGBoost seed-0 CPU fold files are therefore recalculated on CUDA; the CPU and CUDA fold scores are not mixed. The run log is at `MyDrive/AcademicEarlyIntervention/colab_pipeline.log`.

## Resume after a disconnect

Reconnect to the same Drive account and rerun cells 1–4. The setup cell leaves the existing extracted project and CSV in place. The pipeline checks its saved `results/runs/` files and continues from the next missing fold or seed. If you switch to a CPU runtime or change library versions mid-run, do not continue in the same output folder; use a separate project copy to avoid mixing device/platform results.

## Merge the exported bundle locally

Keep `results_backup_oct11/` as the original snapshot. Unzip the downloaded bundle into a staging folder such as `colab_import/` under the project root. Do **not** copy the entire Colab `results/` over the local `results/`: that would replace aggregate files and needlessly copy the seven finished models.

The following PowerShell copies only the five new models' seed records and predictions, plus CUDA fold checkpoints. Existing destination files are preserved if byte-identical; a conflicting destination stops the merge for review.

```powershell
$stage = '.\colab_import\results\runs'
$destination = '.\results\runs'
$models = @('xgb', 'lgbm', 'catboost', 'mlp', 'ft_transformer')

foreach ($model in $models) {
    Get-ChildItem -LiteralPath $stage -File |
        Where-Object { $_.Name -like "$model`_seed*" } |
        ForEach-Object {
            $target = Join-Path $destination $_.Name
            if (Test-Path -LiteralPath $target) {
                if ((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash -ne
                    (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash) {
                    throw "Conflict; preserving existing file: $target"
                }
            } else {
                Copy-Item -LiteralPath $_.FullName -Destination $target
            }
        }
}

$cudaFolds = Join-Path $stage 'folds\cuda'
if (Test-Path -LiteralPath $cudaFolds) {
    $targetFolds = Join-Path $destination 'folds\cuda'
    New-Item -ItemType Directory -Force -Path $targetFolds | Out-Null
    foreach ($model in $models) {
        Get-ChildItem -LiteralPath $cudaFolds -File |
            Where-Object { $_.Name -like "$model`_seed*_fold*.json" } |
            ForEach-Object {
                $target = Join-Path $targetFolds $_.Name
                if (Test-Path -LiteralPath $target) {
                    if ((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash -ne
                        (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash) {
                        throw "Conflict; preserving existing file: $target"
                    }
                } else {
                    Copy-Item -LiteralPath $_.FullName -Destination $target
                }
            }
    }
}
```

The export includes the full pipeline aggregates and figures created by the completed Colab run. Import the new model artifacts and the generated aggregates selectively; do not replace the whole `results/` directory. First confirm the run completed with exit code 0, and confirm every added model has seeds 0–4 plus validation/test predictions. Import the per-seed artifacts with the guarded script above. If `models/pipeline/<name>.joblib` files exist in the bundle, import only the five new models with the same hash guard:

```powershell
$stageModels = '.\colab_import\models\pipeline'
$destModels = '.\models\pipeline'
if (Test-Path -LiteralPath $stageModels) {
    foreach ($model in @('xgb', 'lgbm', 'catboost', 'mlp', 'ft_transformer')) {
        $source = Join-Path $stageModels "$model.joblib"
        if (Test-Path -LiteralPath $source) {
            $target = Join-Path $destModels "$model.joblib"
            if (Test-Path -LiteralPath $target) {
                if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne
                    (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash) {
                    throw "Conflict; preserving existing file: $target"
                }
            } else {
                Copy-Item -LiteralPath $source -Destination $target
            }
        }
    }
}
```

Only after the seven-model per-seed hash check below passes, replace the aggregate files and figures with the full 12-model versions from the successful Colab export. This allow-list contains pipeline-generated summaries; it does not include `tuning_*.json`, `legacy_80_20.json`, or any per-model run files:

```powershell
$stageResults = '.\colab_import\results'
$liveResults = '.\results'
$aggregateFiles = @(
    'config.json', 'data_split.json', 'cv.json', 'validation.json',
    'thresholds_validation.json', 'selection.json',
    'test_argmax.json', 'test_tuned_thresholds.json', 'summary_table.json',
    'operating_points.json'
)
$models = @('xgb', 'lgbm', 'catboost', 'mlp', 'ft_transformer')
foreach ($model in $models) {
    foreach ($seed in 0..4) {
        $stem = "${model}_seed${seed}"
        foreach ($suffix in @('.json', '_val_predictions.npz', '_test_predictions.npz')) {
            $path = Join-Path $stageResults "runs\${stem}${suffix}"
            if (-not (Test-Path -LiteralPath $path)) { throw "Missing completed model artifact: $path" }
        }
    }
}
$selection = Get-Content -Raw (Join-Path $stageResults 'selection.json') | ConvertFrom-Json
if (-not $selection.decided_before_test_evaluation) {
    throw 'Colab selection file does not confirm validation-first selection.'
}
if ($selection.argmax_rule.ranking.Count -ne 12 -or $selection.tuned_threshold_rule.ranking.Count -ne 12) {
    throw 'Selection does not contain all 12 models; do not import a subset run.'
}
foreach ($name in $aggregateFiles) {
    $source = Join-Path $stageResults $name
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing aggregate: $source" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $liveResults $name) -Force
}
if (-not (Test-Path -LiteralPath '.\figures_backup_oct11')) {
    Copy-Item -LiteralPath '.\figures' -Destination '.\figures_backup_oct11' -Recurse
}
Copy-Item -Path '.\colab_import\figures\*' -Destination '.\figures' -Recurse -Force
```

The aggregate JSON files intentionally become 12-model outputs. The original seven-model aggregate files remain in `results_backup_oct11/`. The guarded seed merge never overwrites any existing per-seed record or prediction, and the hash check below verifies all seven models' per-seed files against the backup after import.

To rebuild all aggregates locally instead of importing the Colab-generated aggregates, first ensure all 12 models have all five seed JSONs and validation/test predictions, then run the full default model list:

```powershell
python src/pipeline/run_pipeline.py --device cpu
```

This reuses complete seed checkpoints and does not refit those models. It does recompute validation thresholds, selection, and test metrics from the saved probabilities, then rewrites the summary and figures. Selection is written before that test-metric pass. Use the Python environment whose version and dependencies match the saved run as closely as possible, and do not pass a reduced `--models` list. Importing the successful Colab aggregates is preferable when you want to preserve the single completed Colab scoring pass without recalculating test metrics locally.

The protected-file check (run after importing the Colab seed artifacts and again after updating aggregates) is:

```powershell
$protected = @('majority', 'midterm_lr', 'logreg', 'logreg_balanced', 'rf_capped', 'rf_unconstrained', 'hgb')
foreach ($model in $protected) {
    Get-ChildItem -LiteralPath '.\results_backup_oct11\runs' -File |
        Where-Object { $_.Name -like "$model`_seed*" } |
        ForEach-Object {
            $live = Join-Path '.\results\runs' $_.Name
            if (-not (Test-Path -LiteralPath $live) -or
                (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash -ne
                (Get-FileHash -LiteralPath $live -Algorithm SHA256).Hash) {
                throw "Protected seven-model artifact changed or missing: $live"
            }
        }
}
```

## Risks and interpretation

- The saved run used Python 3.13.13, scikit-learn 1.7.1, NumPy 2.3.1, and pandas 2.3.1. Colab's Sep. 16, 2026 release notes report Python 3.13.15 across its images, so the current default is close but not identical; the notebook prints the actual runtime version. Colab updates runtimes, and requirements use minimum versions, so library versions can drift. See [Colab release notes](https://colab.research.google.com/notebooks/relnotes.ipynb?trk=article-ssr-frontend-pulse_little-text-block).
- XGBoost/CatBoost/PyTorch CUDA results may differ slightly from CPU results due to kernel and floating-point differences. CatBoost documents its GPU implementation as non-deterministic. Seeds and hyperparameters stay fixed, but results are not guaranteed bit-for-bit identical. LightGBM is explicitly kept on CPU. See [CatBoost GPU documentation](https://catboost.ai/docs/en/features/training-on-gpu) and [XGBoost GPU documentation](https://xgboost.readthedocs.io/en/stable/gpu/).
- Existing `results/tuning_*.json` files are reused before any search begins, so the saved best parameters will **not** change just because CUDA is enabled. If tuning files are absent in a fresh copy, a GPU tuning pass can select different parameters. Do not remove the existing tuning files to force retuning.
- The run JSON schema is unchanged and does not record a device field. Keep the Drive run log with the exported bundle as provenance. Resume with one device consistently; the CUDA CV folder is separate from the older CPU folds, but completed seed records are intentionally reused regardless of device.
- `selection.json` will include the five new models only when the full default pipeline finishes with all model/seed artifacts available. The notebook uses that full list; a subset run would produce subset selection. The final local full-list aggregation recomputes selection from stored validation results before it calculates test metrics.
- In `src/pipeline/run_pipeline.py`, each completed model run generates and stores test probabilities before `selection.json` is written; the test labels and metrics are only used in the later evaluation stage. Those probabilities do not affect tuning or selection, but “the test set is untouched until after selection” is not literally true for test-feature inference.
- Colab GPU support must be available for XGBoost and CatBoost as well as PyTorch. The notebook checks PyTorch CUDA; unsupported booster/runtime builds can still fail when those models start.
