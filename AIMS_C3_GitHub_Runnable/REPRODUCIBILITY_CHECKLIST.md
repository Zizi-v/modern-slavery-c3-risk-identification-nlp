# Reproducibility Checklist

This repository was cleaned to make the C3 thesis pipeline understandable, executable, and reproducible.

## Included

- [x] Main executable script: `run_pipeline.py`
- [x] Clean notebook version: `final_op_clean.ipynb`
- [x] Dependency file: `requirements.txt`
- [x] README with installation and run commands
- [x] Automatic public dataset download from Hugging Face
- [x] Fixed random seed
- [x] Environment/package logging
- [x] Output manifest
- [x] Structured output folders
- [x] Git ignore rules for generated outputs and local environments

## Main reproducibility command

```bash
RUN_TRANSFORMER=true RUN_SHAP=true RUN_OPTUNA_TUNING=true OPTUNA_N_TRIALS=50 python run_pipeline.py
```

## Smoke test command

```bash
RUN_TRANSFORMER=false RUN_SHAP=false OPTUNA_N_TRIALS=2 python run_pipeline.py
```

## Expected generated output root

```text
aims_c3_reproducible_project/outputs/
```
