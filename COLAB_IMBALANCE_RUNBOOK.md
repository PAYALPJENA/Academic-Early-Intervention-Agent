# Colab runbook: supplementary class-imbalance experiment

The smoke test passed locally on a stratified 6,000-row subset in about 29 seconds. A rough extrapolation for the full data is 30-45 CPU hours for three seeds and 8-16 hours on a Colab T4. Treat these as planning estimates; SMOTE neighbor search and Colab runtime availability make the real duration variable. Run the full experiment on Colab. The notebook uses three seeds (`0, 1, 2`), which is allowed when five seeds are too slow, and 120-minute resumable chunks.

## Copy the small code bundle to Drive

1. Upload the local `colab_imbalance_code.zip` from this project root to `MyDrive/AcademicEarlyIntervention/workspace/Academic-Early-Intervention-Agent` in Drive.
2. In Colab, upload and open `notebooks/colab_imbalance.ipynb` from this project.
3. In the Colab runtime, select **Runtime > Change runtime type > T4 GPU**.
4. Run notebook cells from top to bottom. The first cell mounts Drive, checks that the existing project contains the dataset, CatBoost tuning file and saved CatBoost model, and copies only these three bundle members into the project: `requirements-extras.txt`, `run_imbalance_experiment.py`, and `notebooks/colab_imbalance.ipynb`. It saves an existing `requirements-extras.txt` as `requirements-extras.pre-imbalance.bak` before replacing it. It refuses to replace a differing script or notebook.
5. Run the dependency install and GPU-info cells, then the smoke-test cell. The smoke test uses `--device cuda --smoke --time-budget 10`.
6. Run the full-run cell. It executes `python run_imbalance_experiment.py --device cuda --time-budget 120 --seeds 0 1 2`. If the cell stops at a budget boundary or Colab disconnects, reconnect, rerun the setup cells as needed, and rerun the full-run cell. It resumes from `results/imbalance/checkpoints/` in Drive. Do not remove that directory before completion.
7. When both `results/imbalance_experiment.json` and `figures/imbalance_comparison.png` exist, run the export cell. It creates `imbalance_results_export.zip` in the Drive project folder and downloads it. If that name already exists, rename the old export before running the cell again.

## Merge the completed new results locally

The export should contain only `results/imbalance_experiment.json`, `figures/imbalance_comparison.png`, and full-run files under `results/imbalance/checkpoints/`. Do not copy the smoke folder. Extract the ZIP to a fresh temporary directory first. The commands below validate allowed paths, require both deliverables, refuse to overwrite any local destination, then SHA256-check each copy:

```powershell
$Project = (Get-Location).Path
$Bundle = 'C:\path\to\imbalance_results_export.zip'
$Temp = Join-Path $env:TEMP 'imbalance-results-import-check'
if (Test-Path -LiteralPath $Temp) { throw "Choose a fresh temp directory: $Temp" }
New-Item -ItemType Directory -Path $Temp | Out-Null
Expand-Archive -LiteralPath $Bundle -DestinationPath $Temp
$Files = Get-ChildItem -LiteralPath $Temp -File -Recurse
foreach ($File in $Files) {
    $Rel = $File.FullName.Substring($Temp.Length).TrimStart('\','/').Replace('\','/')
    if ($Rel -notin @('results/imbalance_experiment.json','figures/imbalance_comparison.png') -and $Rel -notlike 'results/imbalance/checkpoints/*') { throw "Unexpected archive member: $Rel" }
}
foreach ($Required in @('results/imbalance_experiment.json','figures/imbalance_comparison.png')) {
    if (!(Test-Path -LiteralPath (Join-Path $Temp $Required) -PathType Leaf)) { throw "Missing required output: $Required" }
}
foreach ($File in $Files) {
    $Rel = $File.FullName.Substring($Temp.Length).TrimStart('\','/')
    $Target = Join-Path $Project $Rel
    if (Test-Path -LiteralPath $Target) { throw "Refusing to overwrite existing file: $Target" }
}
foreach ($File in $Files) {
    $Rel = $File.FullName.Substring($Temp.Length).TrimStart('\','/')
    $Target = Join-Path $Project $Rel
    New-Item -ItemType Directory -Path (Split-Path -Parent $Target) -Force | Out-Null
    Copy-Item -LiteralPath $File.FullName -Destination $Target
    if ((Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash) { throw "SHA256 mismatch after copy: $Rel" }
}
```

These paths are separate from the original 12-model files. The experiment script refuses to replace an existing aggregate with another run signature, and refuses to replace an existing comparison figure. The merge commands also stop if any destination already exists.
