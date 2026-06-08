# AIMS.au C3 Risk-Description Detection Pipeline

This repository contains a reproducible end-to-end thesis pipeline for detecting **Criterion 3 (modern slavery risk-description disclosure)** in corporate modern slavery statements from the public AIMS.au dataset.

The pipeline performs:

- automatic AIMS.au dataset download from Hugging Face;
- data cleaning and C3 task construction;
- exploratory data analysis and leakage/overlap checks;
- TF-IDF feature extraction;
- baseline modelling;
- Optuna-tuned classical machine-learning models;
- optional DistilBERT fine-tuning;
- sentence-level and document-level evaluation;
- error analysis;
- SHAP-based interpretability;
- thesis-ready tables, figures, predictions, metrics, and run manifest.

## Repository structure

```text
.
├── run_pipeline.py              # Main executable pipeline for GitHub/CLI use
├── final_op_clean.ipynb         # Clean notebook version of the same pipeline
├── requirements.txt             # Python dependencies
├── README.md                    # Reproducibility instructions
├── REPRODUCIBILITY_CHECKLIST.md # Sprint-6 style reproducibility checklist
├── data/                        # Placeholder; dataset is downloaded automatically
└── outputs/                     # Placeholder; generated outputs are written by the run
```

The actual run output is saved under:

```text
aims_c3_reproducible_project/outputs/
```

Key output folders:

```text
outputs/metrics/          # model-comparison metrics, Optuna trials, rankings
outputs/predictions/      # sentence-level and document-level predictions
outputs/figures/          # general figures
outputs/thesis_figures/   # thesis-ready figures
outputs/thesis_tables/    # thesis-ready CSV/LaTeX tables
outputs/thesis_reports/   # text summaries for thesis writing
outputs/logs/             # environment/config logs
outputs/models/           # saved models if enabled
```

## Installation

Create and activate a virtual environment.

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick reproducibility smoke test

The full thesis run can be slow because it includes 50 Optuna trials per classical model, DistilBERT fine-tuning, and SHAP. To check that the repository is runnable, start with a lighter smoke test:

### macOS/Linux

```bash
RUN_TRANSFORMER=false RUN_SHAP=false OPTUNA_N_TRIALS=2 python run_pipeline.py
```

### Windows PowerShell

```powershell
$env:RUN_TRANSFORMER="false"
$env:RUN_SHAP="false"
$env:OPTUNA_N_TRIALS="2"
python run_pipeline.py
```

## Full thesis run

Use this command to reproduce the full intended thesis pipeline:

### macOS/Linux

```bash
RUN_TRANSFORMER=true RUN_SHAP=true RUN_OPTUNA_TUNING=true OPTUNA_N_TRIALS=50 python run_pipeline.py
```

### Windows PowerShell

```powershell
$env:RUN_TRANSFORMER="true"
$env:RUN_SHAP="true"
$env:RUN_OPTUNA_TUNING="true"
$env:OPTUNA_N_TRIALS="50"
python run_pipeline.py
```

A GPU is recommended for the DistilBERT part. CPU execution is possible but slow.

## Configuration via environment variables

The main runtime options can be changed without editing the source code:

| Variable | Default | Meaning |
|---|---:|---|
| `PROJECT_DIR` | `./aims_c3_reproducible_project` | Output/project directory |
| `RUN_XGBOOST` | `true` | Train XGBoost model |
| `RUN_TRANSFORMER` | `true` | Fine-tune DistilBERT |
| `RUN_SHAP` | `true` | Run SHAP interpretability |
| `RUN_OPTUNA_TUNING` | `true` | Tune classical models with Optuna |
| `OPTUNA_N_TRIALS` | `50` | Number of Optuna trials per tunable classical model |
| `TRANSFORMER_EPOCHS` | `2` | DistilBERT fine-tuning epochs |
| `TRANSFORMER_BATCH_SIZE` | `8` | Transformer batch size |
| `TRANSFORMER_MAX_LENGTH` | `256` | Maximum transformer token length |
| `TRANSFORMER_LEARNING_RATE` | `2e-5` | Transformer learning rate |
| `MLP_MAX_ITER` | `40` | Maximum MLP iterations |

## Dataset

The dataset is downloaded automatically from:

```text
mila-ai4h/AIMS.au
```

No private local data path is required.

## Reproducibility notes

The pipeline fixes the random seed, keeps train/validation/test splits separate, fits TF-IDF only on the training split, records package versions and configuration, and saves a run manifest at:

```text
aims_c3_reproducible_project/outputs/run_manifest.json
```

## Main command for reviewers

For a complete run:

```bash
pip install -r requirements.txt
RUN_TRANSFORMER=true RUN_SHAP=true RUN_OPTUNA_TUNING=true OPTUNA_N_TRIALS=50 python run_pipeline.py
```

For a quick structural check:

```bash
pip install -r requirements.txt
RUN_TRANSFORMER=false RUN_SHAP=false OPTUNA_N_TRIALS=2 python run_pipeline.py
```
