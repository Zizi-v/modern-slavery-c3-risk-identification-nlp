"""
AIMS.au Criterion 3 reproducible end-to-end thesis pipeline.

This script is generated from final_op_clean.ipynb and is intended for
GitHub execution. It downloads the public AIMS.au dataset, runs EDA,
trains/evaluates classical ML and Transformer models, and saves thesis-ready
outputs.
"""



# %%

# ============================================================
# 1. Reproducibility setup
# ============================================================
import os
import random
import json
import time
import platform
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Headless-safe plotting for GitHub/servers.
os.environ.setdefault("MPLBACKEND", "Agg")

CONFIG = {
    "SEED": 42,
    "PROJECT_DIR": os.getenv("PROJECT_DIR", "./aims_c3_reproducible_project"),
    "HF_REPO_ID": "mila-ai4h/AIMS.au",
    "HF_REPO_TYPE": "dataset",
    "TEXT_COL": "sentence",
    "GROUP_COL": "statement_id",
    "TARGET_COL": "C3 (risk description)",
    "NA_VALUES": [-1, "-1"],
    "TFIDF_MAX_FEATURES": 5000,
    "TFIDF_NGRAM_RANGE": (1, 2),
    "RUN_XGBOOST": True,
    "RUN_TRANSFORMER": True,
    "TRANSFORMER_MODEL_NAME": "distilbert-base-uncased",
    "TRANSFORMER_MAX_LENGTH": 256,
    "TRANSFORMER_EPOCHS": 2,
    "TRANSFORMER_BATCH_SIZE": 8,
    "TRANSFORMER_LEARNING_RATE": 2e-5,
    "SAVE_MODELS": True,
    "RUN_SHAP": True,
    "SHAP_TEST_SAMPLE_SIZE": 1000,
    "SHAP_BACKGROUND_SAMPLE_SIZE": 500,
    "RUN_OPTUNA_TUNING": True,
    "OPTUNA_N_TRIALS": 50,
    "OPTUNA_TIMEOUT_SECONDS": None,
    "OPTUNA_OPTIMIZE_METRIC": "f1_c3",
    "MLP_MAX_ITER": 40,
}


def set_global_seed(seed: int = 42) -> None:
    """Set seeds and deterministic flags for reproducible execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)

# Optional environment overrides for GitHub/CLI runs.
CONFIG["RUN_XGBOOST"] = _env_bool("RUN_XGBOOST", CONFIG["RUN_XGBOOST"])
CONFIG["RUN_TRANSFORMER"] = _env_bool("RUN_TRANSFORMER", CONFIG["RUN_TRANSFORMER"])
CONFIG["RUN_SHAP"] = _env_bool("RUN_SHAP", CONFIG["RUN_SHAP"])
CONFIG["RUN_OPTUNA_TUNING"] = _env_bool("RUN_OPTUNA_TUNING", CONFIG["RUN_OPTUNA_TUNING"])
CONFIG["OPTUNA_N_TRIALS"] = _env_int("OPTUNA_N_TRIALS", CONFIG["OPTUNA_N_TRIALS"])
CONFIG["TRANSFORMER_EPOCHS"] = _env_int("TRANSFORMER_EPOCHS", CONFIG["TRANSFORMER_EPOCHS"])
CONFIG["TRANSFORMER_BATCH_SIZE"] = _env_int("TRANSFORMER_BATCH_SIZE", CONFIG["TRANSFORMER_BATCH_SIZE"])
CONFIG["TRANSFORMER_MAX_LENGTH"] = _env_int("TRANSFORMER_MAX_LENGTH", CONFIG["TRANSFORMER_MAX_LENGTH"])
CONFIG["TRANSFORMER_LEARNING_RATE"] = _env_float("TRANSFORMER_LEARNING_RATE", CONFIG["TRANSFORMER_LEARNING_RATE"])
CONFIG["MLP_MAX_ITER"] = _env_int("MLP_MAX_ITER", CONFIG["MLP_MAX_ITER"])


set_global_seed(CONFIG["SEED"])

PROJECT_DIR = Path(CONFIG["PROJECT_DIR"])
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw_hf_snapshot"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
THESIS_TABLE_DIR = OUTPUT_DIR / "thesis_tables"
THESIS_FIGURE_DIR = OUTPUT_DIR / "thesis_figures"
THESIS_REPORT_DIR = OUTPUT_DIR / "thesis_reports"

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, FIGURE_DIR, METRICS_DIR, PREDICTIONS_DIR, MODELS_DIR, LOGS_DIR, THESIS_TABLE_DIR, THESIS_FIGURE_DIR, THESIS_REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("Project directory:", PROJECT_DIR.resolve())
print("Seed:", CONFIG["SEED"])


# %%

# ============================================================
# 2. Environment and dependency logging
# ============================================================
def get_package_version(package_name: str) -> str:
    try:
        import importlib.metadata as metadata
        return metadata.version(package_name)
    except Exception:
        return "not installed"


PACKAGE_NAMES = [
    "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "joblib",
    "scipy", "tqdm", "huggingface-hub", "datasets", "pyarrow", "xgboost",
    "transformers", "torch", "accelerate", "tokenizers", "shap"
]

environment_report = {
    "created_at": datetime.utcnow().isoformat() + "Z",
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "processor": platform.processor(),
    "package_versions": {name: get_package_version(name) for name in PACKAGE_NAMES},
    "config": CONFIG,
}

with open(LOGS_DIR / "environment_report.json", "w", encoding="utf-8") as f:
    json.dump(environment_report, f, indent=2)

pd.DataFrame([
    {"package": key, "version": value}
    for key, value in environment_report["package_versions"].items()
]).to_csv(LOGS_DIR / "package_versions.csv", index=False)

print(json.dumps(environment_report, indent=2)[:2000])


# %%

# ============================================================
# 3. Automatic dataset download
# ============================================================
def download_hf_dataset_snapshot() -> Path:
    """Download the Hugging Face dataset snapshot and return the local path."""
    from huggingface_hub import snapshot_download

    print("Downloading dataset snapshot from Hugging Face...")
    snapshot_path = snapshot_download(
        repo_id=CONFIG["HF_REPO_ID"],
        repo_type=CONFIG["HF_REPO_TYPE"],
        local_dir=str(RAW_DATA_DIR),
        local_dir_use_symlinks=False,
    )
    print("Dataset snapshot path:", snapshot_path)
    return Path(snapshot_path)


def list_candidate_data_files(root: Path) -> list[Path]:
    """List candidate tabular data files in the downloaded snapshot."""
    extensions = {".csv", ".tsv", ".jsonl", ".json", ".parquet"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    return sorted(files)


snapshot_path = download_hf_dataset_snapshot()
candidate_files = list_candidate_data_files(snapshot_path)

print("Candidate data files:")
for path in candidate_files[:50]:
    print("-", path.relative_to(snapshot_path))
print("Total candidate files:", len(candidate_files))

with open(LOGS_DIR / "downloaded_files.txt", "w", encoding="utf-8") as f:
    for path in candidate_files:
        f.write(str(path.relative_to(snapshot_path)) + "\n")


# %%

# ============================================================
# 4. Data loading
# ============================================================
def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV, TSV, JSONL, JSON, or Parquet file as a pandas DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, na_values=CONFIG["NA_VALUES"], low_memory=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", na_values=CONFIG["NA_VALUES"], low_memory=False)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def find_split_file(files: list[Path], split_name: str) -> Path | None:
    """Find the most likely file for a split based on file names."""
    split_aliases = {
        "train": ["train"],
        "validation": ["validation", "val", "dev"],
        "test": ["test"],
    }
    aliases = split_aliases[split_name]
    matches = []
    for path in files:
        name = path.name.lower()
        stem = path.stem.lower()
        if any(alias in name or alias in stem for alias in aliases):
            matches.append(path)
    if not matches:
        return None
    priority = {".csv": 0, ".parquet": 1, ".jsonl": 2, ".json": 3, ".tsv": 4}
    matches = sorted(matches, key=lambda p: (priority.get(p.suffix.lower(), 99), len(str(p))))
    return matches[0]


def load_splits_from_snapshot(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test splits from the downloaded files."""
    split_paths = {
        "train": find_split_file(files, "train"),
        "validation": find_split_file(files, "validation"),
        "test": find_split_file(files, "test"),
    }
    print("Detected split files:")
    for split, path in split_paths.items():
        print(split, "->", path)

    if all(path is not None for path in split_paths.values()):
        train_df = read_table(split_paths["train"])
        val_df = read_table(split_paths["validation"])
        test_df = read_table(split_paths["test"])
        return train_df, val_df, test_df

    raise FileNotFoundError(
        "Could not automatically identify train/validation/test files in the downloaded snapshot. "
        "Inspect logs/downloaded_files.txt and update find_split_file() if the upstream file names changed."
    )


train_raw, val_raw, test_raw = load_splits_from_snapshot(candidate_files)

train_raw["split"] = "train"
val_raw["split"] = "validation"
test_raw["split"] = "test"

df_all_raw = pd.concat([train_raw, val_raw, test_raw], ignore_index=True)

print("Train shape:", train_raw.shape)
print("Validation shape:", val_raw.shape)
print("Test shape:", test_raw.shape)
print("All shape:", df_all_raw.shape)
print("Columns:", df_all_raw.columns.tolist())

# Save local copies to make the exact run auditable.
train_raw.to_csv(PROCESSED_DATA_DIR / "train_downloaded.csv", index=False)
val_raw.to_csv(PROCESSED_DATA_DIR / "validation_downloaded.csv", index=False)
test_raw.to_csv(PROCESSED_DATA_DIR / "test_downloaded.csv", index=False)


# %%

# ============================================================
# 5. Original preprocessing logic for the C3 task
# ============================================================
TEXT_COL = CONFIG["TEXT_COL"]
GROUP_COL = CONFIG["GROUP_COL"]
TARGET_COL = CONFIG["TARGET_COL"]

required_columns = [TEXT_COL, GROUP_COL, TARGET_COL]
missing_columns = [col for col in required_columns if col not in df_all_raw.columns]
if missing_columns:
    raise KeyError(f"Missing required columns: {missing_columns}")

# Replace -1 values that may remain after non-CSV loading.
for df in [train_raw, val_raw, test_raw, df_all_raw]:
    df.replace({-1: np.nan, "-1": np.nan}, inplace=True)

# Keep only C3-labelled, non-empty text rows.
df_c3 = df_all_raw[df_all_raw[TARGET_COL].notna()].copy()
df_c3 = df_c3.dropna(subset=[TEXT_COL])
df_c3 = df_c3[df_c3[TEXT_COL].astype(str).str.strip() != ""]
df_c3[TARGET_COL] = df_c3[TARGET_COL].astype(int)
df_c3[GROUP_COL] = df_c3[GROUP_COL].astype(str)

train_c3 = df_c3[df_c3["split"] == "train"].copy()
val_c3 = df_c3[df_c3["split"] == "validation"].copy()
test_c3 = df_c3[df_c3["split"] == "test"].copy()

X_train = train_c3[TEXT_COL].astype(str)
y_train = train_c3[TARGET_COL].astype(int)
X_val = val_c3[TEXT_COL].astype(str)
y_val = val_c3[TARGET_COL].astype(int)
X_test = test_c3[TEXT_COL].astype(str)
y_test = test_c3[TARGET_COL].astype(int)

print("C3 modelling dataset shape:", df_c3.shape)
print("Train/validation/test sizes:", len(train_c3), len(val_c3), len(test_c3))
print("C3 distribution by split:")
print(df_c3.groupby("split")[TARGET_COL].value_counts(dropna=False).unstack(fill_value=0))

df_c3.to_csv(PROCESSED_DATA_DIR / "c3_modelling_dataset.csv", index=False)
train_c3.to_csv(PROCESSED_DATA_DIR / "train_c3.csv", index=False)
val_c3.to_csv(PROCESSED_DATA_DIR / "validation_c3.csv", index=False)
test_c3.to_csv(PROCESSED_DATA_DIR / "test_c3.csv", index=False)


# %%

# ============================================================
# 6. Exploratory Data Analysis
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

LABEL_COLS = [
    "Approval", "Signature", "C1 (reporting entity)", "C2 (structure)",
    "C2 (operations)", "C2 (supply chains)", "C3 (risk description)",
    "C4 (risk mitigation)", "C4 (remediation)", "C5 (effectiveness)",
    "C6 (consultation)"
]
LABEL_COLS = [col for col in LABEL_COLS if col in df_all_raw.columns]

eda_summary = {}

# Split-level rows and documents.
eda_summary["split_rows"] = df_all_raw.groupby("split").size().to_dict()
eda_summary["split_unique_statements"] = df_all_raw.groupby("split")[GROUP_COL].nunique().to_dict()

# Missing and empty text.
empty_text_mask = df_all_raw[TEXT_COL].isna() | (df_all_raw[TEXT_COL].astype(str).str.strip() == "")
eda_summary["empty_text_rows"] = int(empty_text_mask.sum())

# Sentence length.
df_all_raw["char_len"] = df_all_raw[TEXT_COL].fillna("").astype(str).apply(len)
df_all_raw["word_len"] = df_all_raw[TEXT_COL].fillna("").astype(str).apply(lambda x: len(x.split()))
eda_summary["length_summary"] = df_all_raw[["char_len", "word_len"]].describe().to_dict()

# Unusual symbols and URLs.
weird_mask = df_all_raw[TEXT_COL].fillna("").astype(str).str.contains(r"[^a-zA-Z0-9\s\.,;:!\?()\-\&/%']", regex=True)
url_mask = df_all_raw[TEXT_COL].fillna("").astype(str).str.contains(r"http|www", case=False, regex=True)
eda_summary["rows_with_unusual_symbols"] = int(weird_mask.sum())
eda_summary["rows_with_urls"] = int(url_mask.sum())

# Label sparsity.
if LABEL_COLS:
    df_all_raw["positive_label_count"] = (df_all_raw[LABEL_COLS] == 1).sum(axis=1)
    eda_summary["positive_label_count"] = df_all_raw["positive_label_count"].value_counts().sort_index().to_dict()

# C3 distribution.
eda_summary["c3_distribution_after_preprocessing"] = df_c3.groupby("split")[TARGET_COL].value_counts().unstack(fill_value=0).to_dict()

# Duplicates and leakage checks.
duplicate_sentence_count = int(df_all_raw.duplicated(subset=[TEXT_COL]).sum())
eda_summary["duplicate_sentence_count"] = duplicate_sentence_count

train_texts = set(train_c3[TEXT_COL].astype(str))
val_texts = set(val_c3[TEXT_COL].astype(str))
test_texts = set(test_c3[TEXT_COL].astype(str))
eda_summary["exact_text_overlap_train_val"] = len(train_texts.intersection(val_texts))
eda_summary["exact_text_overlap_train_test"] = len(train_texts.intersection(test_texts))
eda_summary["exact_text_overlap_val_test"] = len(val_texts.intersection(test_texts))

with open(METRICS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(eda_summary, f, indent=2, default=str)

print(json.dumps(eda_summary, indent=2, default=str)[:3000])

# Plot sentence length distribution.
plt.figure(figsize=(8, 5))
sns.histplot(df_all_raw["word_len"], bins=50)
plt.title("Sentence Length Distribution")
plt.xlabel("Word count")
plt.ylabel("Number of sentences")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "eda_sentence_length_distribution.png", dpi=300)
plt.close()

# Plot C3 class distribution by split.
plt.figure(figsize=(7, 5))
sns.countplot(data=df_c3, x="split", hue=TARGET_COL)
plt.title("C3 Class Distribution by Split")
plt.xlabel("Split")
plt.ylabel("Number of sentences")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "eda_c3_class_distribution_by_split.png", dpi=300)
plt.close()

# Save EDA samples for inspection.
df_all_raw.loc[empty_text_mask].head(100).to_csv(PREDICTIONS_DIR / "eda_empty_text_rows_sample.csv", index=False)
df_all_raw.loc[weird_mask, [GROUP_COL, TEXT_COL, "split"]].head(500).to_csv(PREDICTIONS_DIR / "eda_unusual_symbol_rows_sample.csv", index=False)
df_all_raw.loc[url_mask, [GROUP_COL, TEXT_COL, "split"]].head(500).to_csv(PREDICTIONS_DIR / "eda_url_rows_sample.csv", index=False)


# %%

# ============================================================
# 6B. Thesis-ready EDA reporting
# ============================================================
from IPython.display import display, Image, Markdown

THESIS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
THESIS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
THESIS_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- EDA tables ----------
split_overview = (
    df_c3.groupby("split")
    .agg(
        n_sentences=(TEXT_COL, "size"),
        n_documents=(GROUP_COL, "nunique"),
        c3_positive=(TARGET_COL, "sum"),
        c3_rate=(TARGET_COL, "mean"),
    )
    .reset_index()
)
split_overview["c3_rate_pct"] = (split_overview["c3_rate"] * 100).round(2)
split_overview.to_csv(THESIS_TABLE_DIR / "table_eda_split_overview.csv", index=False)
split_overview.to_latex(THESIS_TABLE_DIR / "table_eda_split_overview.tex", index=False, float_format="%.3f")

length_stats = (
    df_c3.assign(
        word_len=df_c3[TEXT_COL].fillna("").astype(str).apply(lambda x: len(x.split())),
        char_len=df_c3[TEXT_COL].fillna("").astype(str).apply(len),
    )
    .groupby(["split", TARGET_COL])[["word_len", "char_len"]]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(2)
)
length_stats.to_csv(THESIS_TABLE_DIR / "table_eda_sentence_length_by_split_and_label.csv")
length_stats.to_latex(THESIS_TABLE_DIR / "table_eda_sentence_length_by_split_and_label.tex", float_format="%.2f")

if LABEL_COLS:
    label_distribution = (df_all_raw[LABEL_COLS] == 1).sum().sort_values(ascending=False).reset_index()
    label_distribution.columns = ["criterion", "positive_sentences"]
    label_distribution.to_csv(THESIS_TABLE_DIR / "table_eda_all_criteria_positive_counts.csv", index=False)
    label_distribution.to_latex(THESIS_TABLE_DIR / "table_eda_all_criteria_positive_counts.tex", index=False)

leakage_table = pd.DataFrame([
    {"check": "Exact sentence overlap: train-validation", "value": eda_summary.get("exact_text_overlap_train_val")},
    {"check": "Exact sentence overlap: train-test", "value": eda_summary.get("exact_text_overlap_train_test")},
    {"check": "Exact sentence overlap: validation-test", "value": eda_summary.get("exact_text_overlap_val_test")},
    {"check": "Duplicate sentences in full raw data", "value": eda_summary.get("duplicate_sentence_count")},
    {"check": "Rows with empty text", "value": eda_summary.get("empty_text_rows")},
    {"check": "Rows with unusual symbols", "value": eda_summary.get("rows_with_unusual_symbols")},
    {"check": "Rows with URLs", "value": eda_summary.get("rows_with_urls")},
])
leakage_table.to_csv(THESIS_TABLE_DIR / "table_eda_data_quality_and_leakage_checks.csv", index=False)
leakage_table.to_latex(THESIS_TABLE_DIR / "table_eda_data_quality_and_leakage_checks.tex", index=False)

# ---------- EDA figures ----------
plt.figure(figsize=(8, 5))
sns.barplot(data=split_overview, x="split", y="n_sentences")
plt.title("Number of Sentences by Split")
plt.xlabel("Split")
plt.ylabel("Number of sentences")
plt.tight_layout()
plt.savefig(THESIS_FIGURE_DIR / "fig_eda_sentences_by_split.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=split_overview, x="split", y="n_documents")
plt.title("Number of Documents by Split")
plt.xlabel("Split")
plt.ylabel("Number of documents")
plt.tight_layout()
plt.savefig(THESIS_FIGURE_DIR / "fig_eda_documents_by_split.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=split_overview, x="split", y="c3_rate_pct")
plt.title("C3 Positive Rate by Split")
plt.xlabel("Split")
plt.ylabel("C3-positive sentences (%)")
plt.tight_layout()
plt.savefig(THESIS_FIGURE_DIR / "fig_eda_c3_positive_rate_by_split.png", dpi=300)
plt.close()

plot_len_df = df_c3.copy()
plot_len_df["word_len"] = plot_len_df[TEXT_COL].fillna("").astype(str).apply(lambda x: len(x.split()))
plot_len_df["C3_label"] = plot_len_df[TARGET_COL].map({0: "Non-C3", 1: "C3"})

plt.figure(figsize=(9, 5))
sns.histplot(data=plot_len_df, x="word_len", hue="C3_label", bins=60, stat="density", common_norm=False)
plt.title("Sentence Length Distribution by C3 Label")
plt.xlabel("Word count")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(THESIS_FIGURE_DIR / "fig_eda_sentence_length_by_c3_label.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
sns.boxplot(data=plot_len_df, x="split", y="word_len", hue="C3_label")
plt.title("Sentence Length by Split and C3 Label")
plt.xlabel("Split")
plt.ylabel("Word count")
plt.tight_layout()
plt.savefig(THESIS_FIGURE_DIR / "fig_eda_sentence_length_boxplot_by_split_label.png", dpi=300)
plt.close()

if LABEL_COLS:
    plt.figure(figsize=(9, 6))
    sns.barplot(data=label_distribution.head(15), y="criterion", x="positive_sentences")
    plt.title("Positive Sentence Count by Reporting Criterion")
    plt.xlabel("Number of positive sentences")
    plt.ylabel("Criterion")
    plt.tight_layout()
    plt.savefig(THESIS_FIGURE_DIR / "fig_eda_positive_counts_by_criterion.png", dpi=300)
    plt.close()

# ---------- EDA display ----------
display(Markdown("### Thesis-ready EDA tables"))
display(split_overview)
display(leakage_table)

eda_figures_to_display = [
    "fig_eda_sentences_by_split.png",
    "fig_eda_documents_by_split.png",
    "fig_eda_c3_positive_rate_by_split.png",
    "fig_eda_sentence_length_by_c3_label.png",
    "fig_eda_sentence_length_boxplot_by_split_label.png",
    "fig_eda_positive_counts_by_criterion.png",
]
for fig_name in eda_figures_to_display:
    fig_path = THESIS_FIGURE_DIR / fig_name
    if fig_path.exists():
        display(Markdown(f"**{fig_name}**"))
        display(Image(filename=str(fig_path)))


# %%

# ============================================================
# 7. TF-IDF feature extraction
# ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=CONFIG["TFIDF_MAX_FEATURES"],
    ngram_range=CONFIG["TFIDF_NGRAM_RANGE"],
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF validation shape:", X_val_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)

if CONFIG["SAVE_MODELS"]:
    import joblib
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")


# %%

# ============================================================
# 8. Evaluation helper functions
# ============================================================
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)


def safe_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def safe_average_precision(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan


def get_scores(model, X):
    """Return continuous scores for ranking metrics where possible."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores
    return model.predict(X)


def compute_sentence_metrics(y_true, y_pred, y_score=None, model_name="model", split="validation"):
    if y_score is None:
        y_score = y_pred
    return {
        "model": model_name,
        "split": split,
        "level": "sentence",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_c3": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_c3": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_c3": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": safe_auc(y_true, y_score),
        "average_precision": safe_average_precision(y_true, y_score),
        "n": len(y_true),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
    }


def document_level_from_sentence(df: pd.DataFrame, y_true_col: str, y_pred_col: str, y_score_col: str | None = None) -> pd.DataFrame:
    agg_dict = {
        y_true_col: "max",
        y_pred_col: "max",
    }
    if y_score_col is not None:
        agg_dict[y_score_col] = "max"
    doc = df.groupby(GROUP_COL).agg(agg_dict).reset_index()
    return doc


def compute_document_metrics(sentence_df: pd.DataFrame, model_name="model", split="validation"):
    doc_df = document_level_from_sentence(
        sentence_df,
        y_true_col="true_C3",
        y_pred_col="pred_C3",
        y_score_col="score_C3" if "score_C3" in sentence_df.columns else None,
    )
    y_true_doc = doc_df["true_C3"].astype(int)
    y_pred_doc = doc_df["pred_C3"].astype(int)
    y_score_doc = doc_df["score_C3"] if "score_C3" in doc_df.columns else y_pred_doc
    metrics = {
        "model": model_name,
        "split": split,
        "level": "document",
        "accuracy": accuracy_score(y_true_doc, y_pred_doc),
        "precision_c3": precision_score(y_true_doc, y_pred_doc, pos_label=1, zero_division=0),
        "recall_c3": recall_score(y_true_doc, y_pred_doc, pos_label=1, zero_division=0),
        "f1_c3": f1_score(y_true_doc, y_pred_doc, pos_label=1, zero_division=0),
        "roc_auc": safe_auc(y_true_doc, y_score_doc),
        "average_precision": safe_average_precision(y_true_doc, y_score_doc),
        "n": len(y_true_doc),
        "positive_rate_true": float(np.mean(y_true_doc)),
        "positive_rate_pred": float(np.mean(y_pred_doc)),
    }
    return metrics, doc_df


def make_sentence_prediction_frame(df_split: pd.DataFrame, y_true, y_pred, y_score, model_name: str, split: str) -> pd.DataFrame:
    result = df_split[[GROUP_COL, TEXT_COL]].copy()
    result["split"] = split
    result["model"] = model_name
    result["true_C3"] = np.asarray(y_true).astype(int)
    result["pred_C3"] = np.asarray(y_pred).astype(int)
    result["score_C3"] = np.asarray(y_score).astype(float)
    result["error_type"] = np.where(
        (result["true_C3"] == 1) & (result["pred_C3"] == 0), "false_negative",
        np.where((result["true_C3"] == 0) & (result["pred_C3"] == 1), "false_positive", "correct")
    )
    return result


all_metrics = []
validation_prediction_frames = []
test_prediction_frames = []
document_prediction_frames = []
trained_models = {}


# %%

# ============================================================
# 9. Baseline and Optuna-tuned classical machine-learning models
# ============================================================
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

models_to_train = []
optuna_best_rows = []
optuna_trial_rows = []

# ------------------------------------------------------------
# Untuned lower-bound baselines
# ------------------------------------------------------------
# The dummy model is intentionally not tuned. It provides the minimum reference point.
models_to_train.append((
    "Dummy_Most_Frequent",
    DummyClassifier(strategy="most_frequent", random_state=CONFIG["SEED"])
))

# Transparent baseline: a default TF-IDF + Logistic Regression model.
models_to_train.append((
    "TFIDF_LogisticRegression_Baseline",
    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=CONFIG["SEED"])
))


def validation_f1(model) -> float:
    """Fit a model on the training split and return C3 F1 on the validation split."""
    model.fit(X_train_tfidf, y_train)
    pred_val = model.predict(X_val_tfidf)
    return f1_score(y_val, pred_val, pos_label=1, zero_division=0)


def record_study(study, model_name: str) -> None:
    """Save full Optuna trial history and best hyperparameters."""
    for t in study.trials:
        row = {
            "model": model_name,
            "trial_number": t.number,
            "value_validation_f1_c3": t.value,
            "state": str(t.state),
        }
        row.update({f"param_{k}": v for k, v in t.params.items()})
        optuna_trial_rows.append(row)

    best_row = {
        "model": model_name,
        "best_validation_f1_c3": study.best_value,
    }
    best_row.update({f"best_{k}": v for k, v in study.best_params.items()})
    optuna_best_rows.append(best_row)


def make_study(model_name: str):
    import optuna
    sampler = optuna.samplers.TPESampler(seed=CONFIG["SEED"])
    return optuna.create_study(direction="maximize", sampler=sampler, study_name=model_name)


if CONFIG.get("RUN_OPTUNA_TUNING", True):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = int(CONFIG.get("OPTUNA_N_TRIALS", 50))
        timeout = CONFIG.get("OPTUNA_TIMEOUT_SECONDS", None)

        print(f"Running Optuna hyperparameter tuning with {n_trials} trials per tunable classical model.")

        # ------------------------------------------------------------
        # Logistic Regression tuning
        # ------------------------------------------------------------
        def objective_lr(trial):
            C = trial.suggest_float("C", 1e-3, 100.0, log=True)
            class_weight_choice = trial.suggest_categorical("class_weight", ["balanced", None])
            model = LogisticRegression(
                C=C,
                class_weight=class_weight_choice,
                max_iter=2000,
                random_state=CONFIG["SEED"],
                solver="lbfgs",
            )
            return validation_f1(model)

        study_lr = make_study("TFIDF_LogisticRegression_Optuna")
        study_lr.optimize(objective_lr, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        record_study(study_lr, "TFIDF_LogisticRegression_Optuna")
        best_lr = LogisticRegression(
            C=study_lr.best_params["C"],
            class_weight=study_lr.best_params["class_weight"],
            max_iter=2000,
            random_state=CONFIG["SEED"],
            solver="lbfgs",
        )
        models_to_train.append(("TFIDF_LogisticRegression_Optuna", best_lr))

        # ------------------------------------------------------------
        # Linear SVM tuning
        # ------------------------------------------------------------
        def objective_svm(trial):
            C = trial.suggest_float("C", 1e-3, 100.0, log=True)
            class_weight_choice = trial.suggest_categorical("class_weight", ["balanced", None])
            model = LinearSVC(
                C=C,
                class_weight=class_weight_choice,
                random_state=CONFIG["SEED"],
                dual="auto",
                max_iter=5000,
            )
            return validation_f1(model)

        study_svm = make_study("TFIDF_LinearSVM_Optuna")
        study_svm.optimize(objective_svm, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        record_study(study_svm, "TFIDF_LinearSVM_Optuna")
        best_svm = LinearSVC(
            C=study_svm.best_params["C"],
            class_weight=study_svm.best_params["class_weight"],
            random_state=CONFIG["SEED"],
            dual="auto",
            max_iter=5000,
        )
        models_to_train.append(("TFIDF_LinearSVM_Optuna", best_svm))

        # ------------------------------------------------------------
        # Random Forest tuning
        # ------------------------------------------------------------
        def objective_rf(trial):
            n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
            max_depth = trial.suggest_categorical("max_depth", [None, 10, 20, 40, 60])
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            class_weight_choice = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight_choice,
                random_state=CONFIG["SEED"],
                n_jobs=-1,
            )
            return validation_f1(model)

        study_rf = make_study("TFIDF_RandomForest_Optuna")
        study_rf.optimize(objective_rf, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        record_study(study_rf, "TFIDF_RandomForest_Optuna")
        best_rf = RandomForestClassifier(
            n_estimators=study_rf.best_params["n_estimators"],
            max_depth=study_rf.best_params["max_depth"],
            min_samples_split=study_rf.best_params["min_samples_split"],
            min_samples_leaf=study_rf.best_params["min_samples_leaf"],
            max_features=study_rf.best_params["max_features"],
            class_weight=study_rf.best_params["class_weight"],
            random_state=CONFIG["SEED"],
            n_jobs=-1,
        )
        models_to_train.append(("TFIDF_RandomForest_Optuna", best_rf))

        # ------------------------------------------------------------
        # XGBoost tuning
        # ------------------------------------------------------------
        if CONFIG.get("RUN_XGBOOST", True):
            try:
                from xgboost import XGBClassifier
                scale_pos_weight = max(1, (y_train == 0).sum() / max(1, (y_train == 1).sum()))

                def objective_xgb(trial):
                    model = XGBClassifier(
                        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
                        max_depth=trial.suggest_int("max_depth", 2, 8),
                        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        subsample=trial.suggest_float("subsample", 0.6, 1.0),
                        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
                        gamma=trial.suggest_float("gamma", 0.0, 5.0),
                        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=CONFIG["SEED"],
                        n_jobs=-1,
                        verbosity=0,
                    )
                    return validation_f1(model)

                study_xgb = make_study("TFIDF_XGBoost_Optuna")
                study_xgb.optimize(objective_xgb, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
                record_study(study_xgb, "TFIDF_XGBoost_Optuna")
                best_xgb = XGBClassifier(
                    n_estimators=study_xgb.best_params["n_estimators"],
                    max_depth=study_xgb.best_params["max_depth"],
                    learning_rate=study_xgb.best_params["learning_rate"],
                    subsample=study_xgb.best_params["subsample"],
                    colsample_bytree=study_xgb.best_params["colsample_bytree"],
                    min_child_weight=study_xgb.best_params["min_child_weight"],
                    gamma=study_xgb.best_params["gamma"],
                    reg_lambda=study_xgb.best_params["reg_lambda"],
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=CONFIG["SEED"],
                    n_jobs=-1,
                    verbosity=0,
                )
                models_to_train.append(("TFIDF_XGBoost_Optuna", best_xgb))
            except Exception as e:
                print("XGBoost Optuna tuning skipped:", e)

        # ------------------------------------------------------------
        # MLP tuning
        # ------------------------------------------------------------
        def objective_mlp(trial):
            hidden_layer_sizes = trial.suggest_categorical(
                "hidden_layer_sizes",
                [(64,), (100,), (128,), (64, 32), (100, 50), (128, 64)],
            )
            alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
            learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True)
            activation = trial.suggest_categorical("activation", ["relu", "tanh"])
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                activation=activation,
                max_iter=int(CONFIG.get("MLP_MAX_ITER", 40)),
                random_state=CONFIG["SEED"],
                early_stopping=True,
                validation_fraction=0.1,
            )
            return validation_f1(model)

        study_mlp = make_study("TFIDF_MLP_Optuna")
        study_mlp.optimize(objective_mlp, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        record_study(study_mlp, "TFIDF_MLP_Optuna")
        best_mlp = MLPClassifier(
            hidden_layer_sizes=study_mlp.best_params["hidden_layer_sizes"],
            alpha=study_mlp.best_params["alpha"],
            learning_rate_init=study_mlp.best_params["learning_rate_init"],
            activation=study_mlp.best_params["activation"],
            max_iter=int(CONFIG.get("MLP_MAX_ITER", 40)),
            random_state=CONFIG["SEED"],
            early_stopping=True,
            validation_fraction=0.1,
        )
        models_to_train.append(("TFIDF_MLP_Optuna", best_mlp))

        optuna_trials_df = pd.DataFrame(optuna_trial_rows)
        optuna_best_df = pd.DataFrame(optuna_best_rows)
        optuna_trials_df.to_csv(METRICS_DIR / "optuna_trials_all_models.csv", index=False)
        optuna_best_df.to_csv(METRICS_DIR / "optuna_best_params_all_models.csv", index=False)
        optuna_best_df.to_csv(THESIS_TABLE_DIR / "table_optuna_best_hyperparameters.csv", index=False)

        print("Optuna tuning finished. Best validation scores:")
        display(optuna_best_df.sort_values("best_validation_f1_c3", ascending=False))

    except Exception as e:
        print("Optuna tuning failed or Optuna is not installed. Falling back to fixed classical models.")
        print("Error:", e)
        models_to_train.extend([
            ("TFIDF_LogisticRegression_Fixed", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=CONFIG["SEED"])),
            ("TFIDF_LinearSVM_Fixed", LinearSVC(class_weight="balanced", random_state=CONFIG["SEED"], dual="auto")),
            ("TFIDF_RandomForest_Fixed", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=CONFIG["SEED"], n_jobs=-1)),
            ("TFIDF_MLP_Fixed", MLPClassifier(hidden_layer_sizes=(100,), max_iter=int(CONFIG.get("MLP_MAX_ITER", 40)), random_state=CONFIG["SEED"], early_stopping=True)),
        ])
else:
    print("Optuna tuning disabled. Training fixed classical models.")
    models_to_train.extend([
        ("TFIDF_LogisticRegression_Fixed", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=CONFIG["SEED"])),
        ("TFIDF_LinearSVM_Fixed", LinearSVC(class_weight="balanced", random_state=CONFIG["SEED"], dual="auto")),
        ("TFIDF_RandomForest_Fixed", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=CONFIG["SEED"], n_jobs=-1)),
        ("TFIDF_MLP_Fixed", MLPClassifier(hidden_layer_sizes=(100,), max_iter=int(CONFIG.get("MLP_MAX_ITER", 40)), random_state=CONFIG["SEED"], early_stopping=True)),
    ])

# ------------------------------------------------------------
# Train final selected classical models and evaluate validation/test
# ------------------------------------------------------------
for model_name, model in models_to_train:
    start_time = time.time()
    print("\nTraining final model:", model_name)
    model.fit(X_train_tfidf, y_train)
    trained_models[model_name] = model

    for split_name, X_split, y_split, df_split in [
        ("validation", X_val_tfidf, y_val, val_c3),
        ("test", X_test_tfidf, y_test, test_c3),
    ]:
        pred = model.predict(X_split)
        score = get_scores(model, X_split)
        sentence_metrics = compute_sentence_metrics(y_split, pred, score, model_name=model_name, split=split_name)
        sentence_metrics["runtime_seconds_train_plus_eval"] = round(time.time() - start_time, 2)
        all_metrics.append(sentence_metrics)

        pred_frame = make_sentence_prediction_frame(df_split, y_split, pred, score, model_name, split_name)
        if split_name == "validation":
            validation_prediction_frames.append(pred_frame)
        else:
            test_prediction_frames.append(pred_frame)

        doc_metrics, doc_df = compute_document_metrics(pred_frame, model_name=model_name, split=split_name)
        doc_metrics["runtime_seconds_train_plus_eval"] = round(time.time() - start_time, 2)
        all_metrics.append(doc_metrics)
        doc_df["model"] = model_name
        doc_df["split"] = split_name
        document_prediction_frames.append(doc_df)

    if CONFIG["SAVE_MODELS"]:
        import joblib
        joblib.dump(model, MODELS_DIR / f"{model_name}.joblib")

print("Finished baseline and Optuna-tuned classical model training.")



# %%

# ============================================================
# 10. advanced transformer model
# ============================================================
def run_transformer_model():
    import time
    import inspect
    import numpy as np
    import pandas as pd
    import torch

    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )

    set_global_seed(CONFIG["SEED"])

    model_name = CONFIG["TRANSFORMER_MODEL_NAME"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def make_hf_dataset(df: pd.DataFrame):
        tmp = pd.DataFrame({
            "text": df[TEXT_COL].astype(str).tolist(),
            "labels": df[TARGET_COL].astype(int).tolist(),
            GROUP_COL: df[GROUP_COL].astype(str).tolist(),
        })
        return Dataset.from_pandas(tmp, preserve_index=False)

    train_ds = make_hf_dataset(train_c3)
    val_ds = make_hf_dataset(val_c3)
    test_ds = make_hf_dataset(test_c3)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=CONFIG["TRANSFORMER_MAX_LENGTH"],
        )

    train_tok = train_ds.map(tokenize, batched=True)
    val_tok = val_ds.map(tokenize, batched=True)
    test_tok = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]

        preds = np.argmax(logits, axis=-1)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        scores = probs[:, 1]

        return {
            "accuracy": accuracy_score(labels, preds),
            "precision_c3": precision_score(labels, preds, pos_label=1, zero_division=0),
            "recall_c3": recall_score(labels, preds, pos_label=1, zero_division=0),
            "f1_c3": f1_score(labels, preds, pos_label=1, zero_division=0),
            "roc_auc": safe_auc(labels, scores),
            "average_precision": safe_average_precision(labels, scores),
        }

    # Version-safe TrainingArguments. This handles transformers versions that use
    # either eval_strategy or evaluation_strategy, and avoids unsupported arguments.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters

    training_kwargs = {
        "output_dir": str(MODELS_DIR / "transformer_checkpoints"),
        "seed": CONFIG["SEED"],
        "learning_rate": CONFIG["TRANSFORMER_LEARNING_RATE"],
        "per_device_train_batch_size": CONFIG["TRANSFORMER_BATCH_SIZE"],
        "per_device_eval_batch_size": CONFIG["TRANSFORMER_BATCH_SIZE"],
        "num_train_epochs": CONFIG["TRANSFORMER_EPOCHS"],
        "weight_decay": 0.01,
    }

    if "data_seed" in ta_params:
        training_kwargs["data_seed"] = CONFIG["SEED"]
    if "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = "epoch"
    if "save_strategy" in ta_params:
        training_kwargs["save_strategy"] = "epoch"
    if "logging_strategy" in ta_params:
        training_kwargs["logging_strategy"] = "steps"
    if "logging_steps" in ta_params:
        training_kwargs["logging_steps"] = 100
    if "load_best_model_at_end" in ta_params and ("eval_strategy" in training_kwargs or "evaluation_strategy" in training_kwargs):
        training_kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in ta_params:
        training_kwargs["metric_for_best_model"] = "f1_c3"
    if "greater_is_better" in ta_params:
        training_kwargs["greater_is_better"] = True
    if "report_to" in ta_params:
        training_kwargs["report_to"] = []
    if "remove_unused_columns" in ta_params:
        training_kwargs["remove_unused_columns"] = True
    if "fp16" in ta_params:
        training_kwargs["fp16"] = torch.cuda.is_available()

    training_args = TrainingArguments(**training_kwargs)

    # Version-safe Trainer. Some versions use tokenizer=, newer versions use processing_class=.
    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_tok,
        "eval_dataset": val_tok,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    start_time = time.time()
    trainer.train()
    runtime = round(time.time() - start_time, 2)

    trainer.save_model(str(MODELS_DIR / "transformer_final_model"))
    tokenizer.save_pretrained(str(MODELS_DIR / "transformer_final_model"))

    for split_name, tokenized_ds, original_df in [
        ("validation", val_tok, val_c3),
        ("test", test_tok, test_c3),
    ]:
        output = trainer.predict(tokenized_ds)
        logits = output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        scores = probs[:, 1]
        preds = np.argmax(logits, axis=1)
        y_true_split = np.array(tokenized_ds["labels"])
        model_label = f"Transformer_{model_name.replace('/', '_')}"

        sentence_metrics = compute_sentence_metrics(
            y_true_split,
            preds,
            scores,
            model_name=model_label,
            split=split_name,
        )
        sentence_metrics["runtime_seconds_train_plus_eval"] = runtime
        all_metrics.append(sentence_metrics)

        pred_frame = make_sentence_prediction_frame(
            original_df,
            y_true_split,
            preds,
            scores,
            model_label,
            split_name,
        )

        if split_name == "validation":
            validation_prediction_frames.append(pred_frame)
        else:
            test_prediction_frames.append(pred_frame)

        doc_metrics, doc_df = compute_document_metrics(
            pred_frame,
            model_name=model_label,
            split=split_name,
        )
        doc_metrics["runtime_seconds_train_plus_eval"] = runtime
        all_metrics.append(doc_metrics)

        doc_df["model"] = model_label
        doc_df["split"] = split_name
        document_prediction_frames.append(doc_df)


if CONFIG["RUN_TRANSFORMER"]:
    run_transformer_model()
else:
    print("Transformer model skipped. Set CONFIG['RUN_TRANSFORMER'] = True to run the advanced model.")



# %%

# ============================================================
# 11. Model comparison and saved evaluation outputs
# ============================================================
metrics_df = pd.DataFrame(all_metrics)
metrics_df = metrics_df.sort_values(["split", "level", "f1_c3"], ascending=[True, True, False])
metrics_df.to_csv(METRICS_DIR / "model_comparison_all_metrics.csv", index=False)

validation_predictions_df = pd.concat(validation_prediction_frames, ignore_index=True) if validation_prediction_frames else pd.DataFrame()
test_predictions_df = pd.concat(test_prediction_frames, ignore_index=True) if test_prediction_frames else pd.DataFrame()
document_predictions_df = pd.concat(document_prediction_frames, ignore_index=True) if document_prediction_frames else pd.DataFrame()

validation_predictions_df.to_csv(PREDICTIONS_DIR / "validation_sentence_predictions_all_models.csv", index=False)
test_predictions_df.to_csv(PREDICTIONS_DIR / "test_sentence_predictions_all_models.csv", index=False)
document_predictions_df.to_csv(PREDICTIONS_DIR / "document_predictions_all_models.csv", index=False)

print("Model comparison:")
print(metrics_df)

# Plot validation and test F1 by model and level.
plt.figure(figsize=(12, 6))
plot_df = metrics_df[(metrics_df["split"].isin(["validation", "test"])) & (metrics_df["level"].isin(["sentence", "document"]))]
sns.barplot(data=plot_df, x="model", y="f1_c3", hue="level")
plt.xticks(rotation=45, ha="right")
plt.title("C3 F1 by Model and Evaluation Level")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "model_comparison_f1_by_level.png", dpi=300)
plt.close()


# %%

# ============================================================
# 12. Feature analysis for RQ2
# ============================================================
def save_logistic_regression_feature_importance():
    model_name = "TFIDF_LogisticRegression_Tuned"
    if model_name not in trained_models:
        print("Tuned Logistic Regression not available for feature analysis.")
        return
    model = trained_models[model_name]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]
    feature_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    }).sort_values("coefficient", ascending=False)

    top_positive = feature_df.head(50).copy()
    top_negative = feature_df.tail(50).sort_values("coefficient", ascending=True).copy()
    top_abs = feature_df.sort_values("abs_coefficient", ascending=False).head(100).copy()

    top_positive.to_csv(METRICS_DIR / "top_positive_tfidf_features_c3.csv", index=False)
    top_negative.to_csv(METRICS_DIR / "top_negative_tfidf_features_c3.csv", index=False)
    top_abs.to_csv(METRICS_DIR / "top_absolute_tfidf_features_c3.csv", index=False)

    plt.figure(figsize=(8, 8))
    sns.barplot(data=top_positive.head(20), y="feature", x="coefficient")
    plt.title("Top Positive TF-IDF Features for C3")
    plt.xlabel("Logistic Regression coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "top_positive_tfidf_features_c3.png", dpi=300)
    plt.close()

    print("Top positive C3 features:")
    print(top_positive.head(20))


save_logistic_regression_feature_importance()


# %%

# ============================================================
# 12B. SHAP analysis for selected model interpretability
# ============================================================

def run_shap_analysis_for_selected_model():
    """
    Apply SHAP to the selected/best-performing model.

    Selection logic:
    1. Prefer best test document-level model by F1/F1_C3.
    2. If unavailable, use best test sentence-level model.
    3. If unavailable, use CONFIG['SELECTED_MODEL_FOR_SHAP'].
    4. If unavailable, fall back to the best available trained model.

    Explanation logic:
    - Logistic Regression / Linear SVM: Linear SHAP
    - Random Forest / XGBoost: Tree SHAP
    - MLP: Kernel SHAP on a small sampled dense TF-IDF matrix
    - Transformer / DistilBERT / BERT / ModernBERT: SHAP Text Explainer
    """

    if not CONFIG.get("RUN_SHAP", True):
        print("SHAP skipped because CONFIG['RUN_SHAP'] is False.")
        return

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # ------------------------------------------------------------
    # Safe defaults
    # ------------------------------------------------------------

    CONFIG.setdefault("SELECTED_MODEL_FOR_SHAP", None)
    CONFIG.setdefault("SHAP_BACKGROUND_SAMPLE_SIZE", 100)
    CONFIG.setdefault("SHAP_TEST_SAMPLE_SIZE", 300)
    CONFIG.setdefault("SHAP_TRANSFORMER_SAMPLE_SIZE", 30)
    CONFIG.setdefault("SHAP_RANDOM_STATE", CONFIG.get("SEED", 42))

    for folder in [
        METRICS_DIR,
        PREDICTIONS_DIR,
        FIGURE_DIR,
        THESIS_TABLE_DIR,
        THESIS_FIGURE_DIR,
        THESIS_REPORT_DIR,
    ]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------

    def detect_column(df, candidates):
        lower_map = {str(c).lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    def get_metrics_dataframe():
        if "all_metrics" in globals() and len(all_metrics) > 0:
            return pd.DataFrame(all_metrics)

        possible_files = [
            THESIS_TABLE_DIR / "table_final_model_ranking.csv",
            THESIS_TABLE_DIR / "table_document_level_model_comparison.csv",
            THESIS_TABLE_DIR / "table_sentence_level_model_comparison.csv",
            METRICS_DIR / "all_model_metrics.csv",
            METRICS_DIR / "metrics_all_models.csv",
        ]

        for file_path in possible_files:
            if Path(file_path).exists():
                return pd.read_csv(file_path)

        return pd.DataFrame()

    def select_best_model_name():
        manual = CONFIG.get("SELECTED_MODEL_FOR_SHAP", None)
        if manual:
            print(f"Manual selected model for SHAP: {manual}")
            return manual

        metrics_df = get_metrics_dataframe()

        if not metrics_df.empty:
            model_col = detect_column(metrics_df, ["model", "model_name", "classifier", "method"])
            split_col = detect_column(metrics_df, ["split", "dataset"])
            level_col = detect_column(metrics_df, ["level", "evaluation_level", "granularity"])
            f1_col = detect_column(metrics_df, ["f1_c3", "f1", "F1", "test_f1", "document_f1"])

            if model_col and f1_col:
                df = metrics_df.copy()

                if split_col:
                    df = df[df[split_col].astype(str).str.lower().str.contains("test", na=False)]

                if level_col:
                    doc_df = df[df[level_col].astype(str).str.lower().str.contains("document", na=False)]
                    if not doc_df.empty:
                        df = doc_df

                df[f1_col] = pd.to_numeric(df[f1_col], errors="coerce")
                df = df.dropna(subset=[f1_col])

                if not df.empty:
                    best_row = df.sort_values(f1_col, ascending=False).iloc[0]
                    best_model = str(best_row[model_col])
                    print(f"Best model selected from metrics: {best_model}")
                    print(f"Selection metric: {f1_col} = {best_row[f1_col]}")
                    return best_model

        available = list(trained_models.keys()) if "trained_models" in globals() else []

        priority = [
            "TFIDF_XGBoost_Optuna",
            "TFIDF_LinearSVM_Optuna",
            "TFIDF_LogisticRegression_Optuna",
            "TFIDF_RandomForest_Optuna",
            "TFIDF_MLP_Optuna",
            "TFIDF_XGBoost_Fixed",
            "TFIDF_LinearSVM_Fixed",
            "TFIDF_LogisticRegression_Fixed",
            "TFIDF_RandomForest_Fixed",
            "TFIDF_MLP_Fixed",
            "TFIDF_LogisticRegression_Baseline",
        ]

        for name in priority:
            if name in available:
                print(f"No metrics-based best model found. Falling back to: {name}")
                return name

        if available:
            print(f"No priority model found. Falling back to first available model: {available[0]}")
            return available[0]

        raise ValueError("No model available for SHAP analysis.")

    def resolve_trained_model_key(selected_name):
        if "trained_models" not in globals():
            return None

        available = list(trained_models.keys())

        if selected_name in trained_models:
            return selected_name

        clean_selected = selected_name.lower().replace(" ", "").replace("-", "").replace("_", "")

        for name in available:
            clean_name = name.lower().replace(" ", "").replace("-", "").replace("_", "")
            if clean_selected == clean_name or clean_selected in clean_name or clean_name in clean_selected:
                return name

        aliases = {
            "logistic": ["TFIDF_LogisticRegression_Optuna", "TFIDF_LogisticRegression_Fixed", "TFIDF_LogisticRegression_Baseline"],
            "svm": ["TFIDF_LinearSVM_Optuna", "TFIDF_LinearSVM_Fixed"],
            "linearsvm": ["TFIDF_LinearSVM_Optuna", "TFIDF_LinearSVM_Fixed"],
            "randomforest": ["TFIDF_RandomForest_Optuna", "TFIDF_RandomForest_Fixed"],
            "forest": ["TFIDF_RandomForest_Optuna", "TFIDF_RandomForest_Fixed"],
            "xgboost": ["TFIDF_XGBoost_Optuna", "TFIDF_XGBoost_Fixed"],
            "xgb": ["TFIDF_XGBoost_Optuna", "TFIDF_XGBoost_Fixed"],
            "mlp": ["TFIDF_MLP_Optuna", "TFIDF_MLP_Fixed"],
        }

        for key, candidates in aliases.items():
            if key in clean_selected:
                for candidate in candidates:
                    if candidate in trained_models:
                        return candidate

        return None

    selected_model_name = select_best_model_name()
    trained_model_key = resolve_trained_model_key(selected_model_name)

    is_transformer_selected = any(
        token in selected_model_name.lower()
        for token in ["transformer", "distilbert", "bert", "modernbert"]
    )

    # ============================================================
    # Route A: SHAP for selected Transformer model
    # ============================================================

    if is_transformer_selected:
        print("Selected model is transformer-based. Running token-level SHAP.")

        try:
            import shap
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            transformer_path = MODELS_DIR / "transformer_final_model"

            if not Path(transformer_path).exists():
                transformer_path = CONFIG.get("TRANSFORMER_MODEL_NAME", "distilbert-base-uncased")

            tokenizer = AutoTokenizer.from_pretrained(transformer_path)
            transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_path)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            transformer_model.to(device)
            transformer_model.eval()

            rng = np.random.default_rng(CONFIG["SHAP_RANDOM_STATE"])
            n_samples = min(CONFIG["SHAP_TRANSFORMER_SAMPLE_SIZE"], len(test_c3))

            sample_idx = rng.choice(len(test_c3), size=n_samples, replace=False)

            text_samples = test_c3.iloc[sample_idx][TEXT_COL].astype(str).tolist()
            meta_samples = test_c3.iloc[sample_idx][[GROUP_COL, TEXT_COL, TARGET_COL]].reset_index(drop=True)

            def predict_proba_text(texts):
                if isinstance(texts, np.ndarray):
                    texts = texts.tolist()

                texts = [str(t) for t in texts]
                all_probs = []

                batch_size = 8

                for start in range(0, len(texts), batch_size):
                    batch_texts = texts[start:start + batch_size]

                    encoded = tokenizer(
                        batch_texts,
                        truncation=True,
                        padding=True,
                        max_length=CONFIG.get("TRANSFORMER_MAX_LENGTH", 256),
                        return_tensors="pt",
                    ).to(device)

                    with torch.no_grad():
                        logits = transformer_model(**encoded).logits
                        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                        all_probs.append(probs)

                return np.vstack(all_probs)

            masker = shap.maskers.Text(tokenizer)

            explainer = shap.Explainer(
                predict_proba_text,
                masker,
                output_names=["non_C3", "C3"],
            )

            shap_values = explainer(text_samples)

            local_rows = []

            for i in range(len(text_samples)):
                tokens = shap_values.data[i]
                values = shap_values.values[i]

                if values.ndim == 2:
                    token_values = values[:, 1]
                else:
                    token_values = values

                top_idx = np.argsort(np.abs(token_values))[-15:][::-1]

                for rank, token_id in enumerate(top_idx, start=1):
                    local_rows.append({
                        "sample_id": i,
                        GROUP_COL: meta_samples.loc[i, GROUP_COL],
                        "true_C3": meta_samples.loc[i, TARGET_COL],
                        "sentence": meta_samples.loc[i, TEXT_COL],
                        "rank": rank,
                        "token": str(tokens[token_id]),
                        "shap_value": float(token_values[token_id]),
                        "direction": "towards_C3" if token_values[token_id] >= 0 else "away_from_C3",
                        "selected_model": selected_model_name,
                        "explained_model": str(transformer_path),
                        "explanation_method": "shap.TextExplainer",
                    })

            shap_local = pd.DataFrame(local_rows)

            shap_global = (
                shap_local
                .groupby("token", as_index=False)
                .agg(
                    mean_shap_value=("shap_value", "mean"),
                    mean_abs_shap_value=("shap_value", lambda x: np.mean(np.abs(x)))
                )
                .sort_values("mean_abs_shap_value", ascending=False)
            )

            shap_global["direction"] = np.where(
                shap_global["mean_shap_value"] >= 0,
                "towards_C3",
                "away_from_C3"
            )

            shap_global["selected_model"] = selected_model_name
            shap_global["explanation_method"] = "shap.TextExplainer"

            shap_global.to_csv(
                METRICS_DIR / "shap_selected_transformer_global_token_importance.csv",
                index=False
            )

            shap_local.to_csv(
                PREDICTIONS_DIR / "shap_selected_transformer_local_token_examples.csv",
                index=False
            )

            shap_global.head(30).to_csv(
                THESIS_TABLE_DIR / "table_rq2_shap_selected_transformer_top_tokens.csv",
                index=False
            )

            shap_local.head(150).to_csv(
                THESIS_TABLE_DIR / "table_rq2_shap_selected_transformer_local_examples.csv",
                index=False
            )

            plot_df = shap_global.head(25).sort_values("mean_abs_shap_value", ascending=True)

            plt.figure(figsize=(9, 8))
            sns.barplot(data=plot_df, y="token", x="mean_abs_shap_value")
            plt.title("SHAP Token Importance for Selected Transformer Model")
            plt.xlabel("Mean absolute SHAP value")
            plt.ylabel("Token")
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / "shap_selected_transformer_token_importance.png", dpi=300)
            plt.savefig(THESIS_FIGURE_DIR / "fig_rq2_shap_selected_transformer_token_importance.png", dpi=300)
            plt.close()

            summary_text = [
                "# SHAP Analysis for Selected Transformer Model\n\n",
                f"Selected model: `{selected_model_name}`\n\n",
                f"Explained model path/name: `{transformer_path}`\n\n",
                "Explanation method: `shap.TextExplainer`\n\n",
                f"Explained test sentences: {n_samples}\n\n",
                "The analysis reports token-level contributions towards or away from the C3 class.\n",
            ]

            with open(
                THESIS_REPORT_DIR / "shap_selected_transformer_analysis_summary.md",
                "w",
                encoding="utf-8"
            ) as f:
                f.writelines(summary_text)

            print("Transformer SHAP finished.")
            print("Selected model:", selected_model_name)
            display(shap_global.head(20))
            return

        except Exception as exc:
            print("Transformer SHAP failed.")
            print("Reason:", exc)
            print("Falling back to the best available TF-IDF model for explanation.")

            trained_model_key = resolve_trained_model_key("TFIDF_LogisticRegression_Optuna")
            if trained_model_key is None:
                trained_model_key = resolve_trained_model_key("TFIDF_LogisticRegression_Fixed")
            if trained_model_key is None:
                trained_model_key = resolve_trained_model_key("TFIDF_LogisticRegression_Baseline")

    # ============================================================
    # Route B: SHAP for selected TF-IDF classical model
    # ============================================================

    if trained_model_key is None:
        raise ValueError(
            f"Selected model `{selected_model_name}` could not be matched to trained_models."
        )

    print(f"Running SHAP for selected classical model: {trained_model_key}")

    import shap

    model = trained_models[trained_model_key]
    model_class_name = model.__class__.__name__.lower()
    feature_names = np.array(vectorizer.get_feature_names_out())

    rng = np.random.default_rng(CONFIG["SHAP_RANDOM_STATE"])

    background_size = min(CONFIG["SHAP_BACKGROUND_SAMPLE_SIZE"], X_train_tfidf.shape[0])
    test_size = min(CONFIG["SHAP_TEST_SAMPLE_SIZE"], X_test_tfidf.shape[0])

    background_idx = rng.choice(X_train_tfidf.shape[0], size=background_size, replace=False)
    test_idx = rng.choice(X_test_tfidf.shape[0], size=test_size, replace=False)

    X_background = X_train_tfidf[background_idx]
    X_explain = X_test_tfidf[test_idx]

    explained_text = (
        test_c3
        .iloc[test_idx][[GROUP_COL, TEXT_COL, TARGET_COL]]
        .reset_index(drop=True)
        .copy()
    )

    shap_method = None
    shap_values = None

    # ------------------------------------------------------------
    # Linear models
    # ------------------------------------------------------------

    if hasattr(model, "coef_"):
        try:
            shap_method = "shap.LinearExplainer"
            explainer = shap.LinearExplainer(model, X_background)
            raw_values = explainer.shap_values(X_explain)

            if isinstance(raw_values, list):
                shap_values = raw_values[1]
            elif hasattr(raw_values, "values"):
                shap_values = raw_values.values
            else:
                shap_values = raw_values

            shap_values = np.asarray(shap_values)

        except Exception as exc:
            print("Linear SHAP failed; using linear contribution fallback.")
            print("Reason:", exc)

            shap_method = "linear_contribution_fallback"
            coef = model.coef_[0]
            background_mean = np.asarray(X_background.mean(axis=0)).ravel()
            X_dense = X_explain.toarray()
            shap_values = (X_dense - background_mean) * coef

    # ------------------------------------------------------------
    # Tree models
    # ------------------------------------------------------------

    elif "forest" in model_class_name or "xgb" in model_class_name or "boost" in model_class_name:
        try:
            shap_method = "shap.TreeExplainer"

            X_explain_dense = X_explain.toarray()

            explainer = shap.TreeExplainer(model)
            raw_values = explainer.shap_values(X_explain_dense)

            if isinstance(raw_values, list):
                shap_values = raw_values[1]
            elif hasattr(raw_values, "values"):
                shap_values = raw_values.values
            else:
                shap_values = raw_values

            shap_values = np.asarray(shap_values)

            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

        except Exception as exc:
            raise RuntimeError(f"Tree SHAP failed for selected model `{trained_model_key}`: {exc}")

    # ------------------------------------------------------------
    # MLP / black-box TF-IDF models
    # ------------------------------------------------------------

    else:
        try:
            shap_method = "shap.KernelExplainer"

            small_background_size = min(50, X_background.shape[0])
            small_test_size = min(100, X_explain.shape[0])

            X_background_dense = X_background[:small_background_size].toarray()
            X_explain_dense = X_explain[:small_test_size].toarray()

            explained_text = explained_text.iloc[:small_test_size].reset_index(drop=True)

            def predict_model_dense(x):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(x)

                scores = model.decision_function(x)

                if scores.ndim == 1:
                    probs_pos = 1 / (1 + np.exp(-scores))
                    return np.vstack([1 - probs_pos, probs_pos]).T

                return scores

            explainer = shap.KernelExplainer(
                predict_model_dense,
                X_background_dense
            )

            raw_values = explainer.shap_values(
                X_explain_dense,
                nsamples=100
            )

            if isinstance(raw_values, list):
                shap_values = raw_values[1]
            else:
                shap_values = raw_values

        except Exception as exc:
            raise RuntimeError(f"Kernel SHAP failed for selected model `{trained_model_key}`: {exc}")

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # ------------------------------------------------------------
    # Global SHAP summary
    # ------------------------------------------------------------

    mean_shap = shap_values.mean(axis=0)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_global = pd.DataFrame({
        "feature": feature_names,
        "mean_shap_value": mean_shap,
        "mean_abs_shap_value": mean_abs_shap,
        "direction": np.where(mean_shap >= 0, "towards_C3", "away_from_C3"),
        "selected_model": selected_model_name,
        "explained_model": trained_model_key,
        "explanation_method": shap_method,
    }).sort_values("mean_abs_shap_value", ascending=False)

    shap_positive = shap_global.sort_values("mean_shap_value", ascending=False).head(50).copy()
    shap_negative = shap_global.sort_values("mean_shap_value", ascending=True).head(50).copy()
    shap_top_abs = shap_global.head(100).copy()

    # ------------------------------------------------------------
    # Save SHAP tables
    # ------------------------------------------------------------

    shap_global.to_csv(
        METRICS_DIR / "shap_selected_model_global_feature_importance_all_features.csv",
        index=False
    )

    shap_positive.to_csv(
        METRICS_DIR / "shap_selected_model_top_positive_features_towards_c3.csv",
        index=False
    )

    shap_negative.to_csv(
        METRICS_DIR / "shap_selected_model_top_negative_features_away_from_c3.csv",
        index=False
    )

    shap_top_abs.to_csv(
        METRICS_DIR / "shap_selected_model_top_absolute_features_c3.csv",
        index=False
    )

    shap_top_abs.head(30).to_csv(
        THESIS_TABLE_DIR / "table_rq2_shap_selected_model_top_absolute_features.csv",
        index=False
    )

    shap_positive.head(30).to_csv(
        THESIS_TABLE_DIR / "table_rq2_shap_selected_model_top_positive_features_towards_c3.csv",
        index=False
    )

    shap_negative.head(30).to_csv(
        THESIS_TABLE_DIR / "table_rq2_shap_selected_model_top_negative_features_away_from_c3.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Local SHAP examples
    # ------------------------------------------------------------

    local_rows = []
    top_k_local = 10
    n_local_examples = min(30, shap_values.shape[0])

    for row_idx in range(n_local_examples):
        row_values = shap_values[row_idx]
        top_idx = np.argsort(np.abs(row_values))[-top_k_local:][::-1]

        for rank, feature_idx in enumerate(top_idx, start=1):
            local_rows.append({
                "sample_id": row_idx,
                GROUP_COL: explained_text.loc[row_idx, GROUP_COL],
                "true_C3": explained_text.loc[row_idx, TARGET_COL],
                "sentence": explained_text.loc[row_idx, TEXT_COL],
                "rank": rank,
                "feature": feature_names[feature_idx],
                "shap_value": row_values[feature_idx],
                "direction": "towards_C3" if row_values[feature_idx] >= 0 else "away_from_C3",
                "selected_model": selected_model_name,
                "explained_model": trained_model_key,
                "explanation_method": shap_method,
            })

    shap_local_examples = pd.DataFrame(local_rows)

    shap_local_examples.to_csv(
        PREDICTIONS_DIR / "shap_selected_model_local_explanation_examples.csv",
        index=False
    )

    shap_local_examples.head(100).to_csv(
        THESIS_TABLE_DIR / "table_rq2_shap_selected_model_local_explanation_examples.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Save figures
    # ------------------------------------------------------------

    plot_abs = shap_top_abs.head(25).sort_values("mean_abs_shap_value", ascending=True)

    plt.figure(figsize=(9, 8))
    sns.barplot(data=plot_abs, y="feature", x="mean_abs_shap_value")
    plt.title("SHAP Global Feature Importance for Selected Model")
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_selected_model_global_feature_importance_c3.png", dpi=300)
    plt.savefig(THESIS_FIGURE_DIR / "fig_rq2_shap_selected_model_global_feature_importance_c3.png", dpi=300)
    plt.close()

    directional = pd.concat([
        shap_positive.head(15).assign(group="Towards C3"),
        shap_negative.head(15).assign(group="Away from C3"),
    ], ignore_index=True)

    directional["plot_value"] = directional["mean_shap_value"]
    directional = directional.sort_values("plot_value")

    plt.figure(figsize=(9, 9))
    sns.barplot(data=directional, y="feature", x="plot_value", hue="group", dodge=False)
    plt.axvline(0, linewidth=1)
    plt.title("Directional SHAP Contributions for Selected Model")
    plt.xlabel("Mean SHAP value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_selected_model_directional_features_c3.png", dpi=300)
    plt.savefig(THESIS_FIGURE_DIR / "fig_rq2_shap_selected_model_directional_features_c3.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    summary_text = [
        "# SHAP Analysis for Selected Model\n\n",
        f"Selected model from evaluation: `{selected_model_name}`\n\n",
        f"Explained model: `{trained_model_key}`\n\n",
        f"Explanation method: `{shap_method}`\n\n",
        f"Background samples: {background_size}\n\n",
        f"Explained test samples: {shap_values.shape[0]}\n\n",
        "This analysis applies SHAP to the model selected by the evaluation pipeline. "
        "For transformer models, token-level SHAP explanations are generated. "
        "For TF-IDF-based classical models, feature-level SHAP explanations are generated.\n",
    ]

    with open(
        THESIS_REPORT_DIR / "shap_selected_model_analysis_summary.md",
        "w",
        encoding="utf-8"
    ) as f:
        f.writelines(summary_text)

    print("Selected-model SHAP finished.")
    print("Selected model from evaluation:", selected_model_name)
    print("Explained model:", trained_model_key)
    print("Explanation method:", shap_method)

    print("\nTop SHAP features towards C3:")
    display(shap_positive.head(15))

    print("\nTop SHAP features away from C3:")
    display(shap_negative.head(15))

    print("\nTop absolute SHAP features:")
    display(shap_top_abs.head(15))


run_shap_analysis_for_selected_model()


# %%

# ============================================================
# 13. Error analysis
# ============================================================
def select_best_test_model(metrics: pd.DataFrame) -> str:
    test_doc = metrics[(metrics["split"] == "test") & (metrics["level"] == "document")].copy()
    if len(test_doc) == 0:
        test_sent = metrics[(metrics["split"] == "test") & (metrics["level"] == "sentence")].copy()
        return test_sent.sort_values("f1_c3", ascending=False).iloc[0]["model"]
    return test_doc.sort_values("f1_c3", ascending=False).iloc[0]["model"]


best_model_name = select_best_test_model(metrics_df)
print("Best test model for error analysis:", best_model_name)

best_test_predictions = test_predictions_df[test_predictions_df["model"] == best_model_name].copy()
best_test_predictions["word_len"] = best_test_predictions[TEXT_COL].fillna("").astype(str).apply(lambda x: len(x.split()))
best_test_predictions["char_len"] = best_test_predictions[TEXT_COL].fillna("").astype(str).apply(len)

false_positives = best_test_predictions[best_test_predictions["error_type"] == "false_positive"].copy()
false_negatives = best_test_predictions[best_test_predictions["error_type"] == "false_negative"].copy()
correct_predictions = best_test_predictions[best_test_predictions["error_type"] == "correct"].copy()

false_positives.to_csv(PREDICTIONS_DIR / "best_model_false_positives.csv", index=False)
false_negatives.to_csv(PREDICTIONS_DIR / "best_model_false_negatives.csv", index=False)
correct_predictions.head(500).to_csv(PREDICTIONS_DIR / "best_model_correct_predictions_sample.csv", index=False)

error_summary = best_test_predictions.groupby("error_type").agg(
    n=("error_type", "size"),
    mean_word_len=("word_len", "mean"),
    median_word_len=("word_len", "median"),
    mean_score=("score_C3", "mean"),
).reset_index()
error_summary.to_csv(METRICS_DIR / "best_model_error_summary.csv", index=False)

plt.figure(figsize=(8, 5))
sns.boxplot(data=best_test_predictions, x="error_type", y="word_len")
plt.title("Sentence Length by Error Type")
plt.xlabel("Error type")
plt.ylabel("Word count")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "error_analysis_sentence_length_by_error_type.png", dpi=300)
plt.close()

print("Error summary:")
print(error_summary)


# %%

# ============================================================
# 14. Document-level error analysis
# ============================================================
best_doc_predictions = document_predictions_df[
    (document_predictions_df["model"] == best_model_name) &
    (document_predictions_df["split"] == "test")
].copy()

if not best_doc_predictions.empty:
    best_doc_predictions["doc_error_type"] = np.where(
        (best_doc_predictions["true_C3"] == 1) & (best_doc_predictions["pred_C3"] == 0), "document_false_negative",
        np.where((best_doc_predictions["true_C3"] == 0) & (best_doc_predictions["pred_C3"] == 1), "document_false_positive", "document_correct")
    )
    best_doc_predictions.to_csv(PREDICTIONS_DIR / "best_model_document_level_errors.csv", index=False)
    doc_error_summary = best_doc_predictions["doc_error_type"].value_counts().reset_index()
    doc_error_summary.columns = ["doc_error_type", "n"]
    doc_error_summary.to_csv(METRICS_DIR / "best_model_document_error_summary.csv", index=False)
    print(doc_error_summary)


# %%

# ============================================================
# 15. Thesis-ready results and discussion reporting
# ============================================================
from IPython.display import display, Image, Markdown
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve

THESIS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
THESIS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
THESIS_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helper functions ----------
def save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    csv_path = THESIS_TABLE_DIR / f"{name}.csv"
    tex_path = THESIS_TABLE_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=index)
    try:
        df.to_latex(tex_path, index=index, float_format="%.4f")
    except Exception as exc:
        print(f"LaTeX export skipped for {name}: {exc}")


def display_figures(fig_names):
    for fig_name in fig_names:
        fig_path = THESIS_FIGURE_DIR / fig_name
        if fig_path.exists():
            display(Markdown(f"**{fig_name}**"))
            display(Image(filename=str(fig_path)))


def make_metric_table(level: str, split: str = "test") -> pd.DataFrame:
    out = metrics_df[(metrics_df["split"] == split) & (metrics_df["level"] == level)].copy()
    preferred_cols = [
        "model", "split", "level", "accuracy", "precision_c3", "recall_c3", "f1_c3",
        "roc_auc", "average_precision", "runtime_seconds_train_plus_eval"
    ]
    existing_cols = [c for c in preferred_cols if c in out.columns]
    out = out[existing_cols].sort_values("f1_c3", ascending=False)
    return out

# ---------- Main thesis tables ----------
sentence_test_table = make_metric_table("sentence", "test")
document_test_table = make_metric_table("document", "test")
sentence_val_table = make_metric_table("sentence", "validation")
document_val_table = make_metric_table("document", "validation")

save_table(sentence_test_table, "table_results_sentence_level_test_model_comparison")
save_table(document_test_table, "table_results_document_level_test_model_comparison")
save_table(sentence_val_table, "table_results_sentence_level_validation_model_comparison")
save_table(document_val_table, "table_results_document_level_validation_model_comparison")

# Final ranking prioritises document-level F1, because RQ1 focuses on document-level C3 prediction.
if not document_test_table.empty:
    final_ranking = document_test_table.copy()
    final_ranking["rank_by_document_f1"] = final_ranking["f1_c3"].rank(ascending=False, method="dense").astype(int)
else:
    final_ranking = sentence_test_table.copy()
    final_ranking["rank_by_sentence_f1"] = final_ranking["f1_c3"].rank(ascending=False, method="dense").astype(int)
save_table(final_ranking, "table_results_final_model_ranking")

best_model_summary = pd.DataFrame([{
    "selection_rule": "Highest test document-level F1; sentence-level F1 used only if document-level metrics are unavailable",
    "best_model": best_model_name,
    "available_models": ", ".join(sorted(metrics_df["model"].dropna().unique()))
}])
save_table(best_model_summary, "table_results_best_model_summary")

# Runtime table.
runtime_cols = [c for c in ["model", "split", "level", "runtime_seconds_train_plus_eval"] if c in metrics_df.columns]
if "runtime_seconds_train_plus_eval" in metrics_df.columns:
    runtime_table = (
        metrics_df[runtime_cols]
        .dropna(subset=["runtime_seconds_train_plus_eval"])
        .drop_duplicates()
        .sort_values("runtime_seconds_train_plus_eval", ascending=False)
    )
    save_table(runtime_table, "table_results_runtime_comparison")

# Error-analysis tables.
if "error_summary" in globals():
    save_table(error_summary, "table_error_analysis_best_model_sentence_errors")
if "doc_error_summary" in globals():
    save_table(doc_error_summary, "table_error_analysis_best_model_document_errors")
if "false_positives" in globals():
    false_positives.head(25).to_csv(THESIS_TABLE_DIR / "table_error_analysis_false_positive_examples.csv", index=False)
if "false_negatives" in globals():
    false_negatives.head(25).to_csv(THESIS_TABLE_DIR / "table_error_analysis_false_negative_examples.csv", index=False)

# ---------- Model-comparison figures ----------
test_metrics_for_plot = metrics_df[(metrics_df["split"] == "test") & (metrics_df["level"].isin(["sentence", "document"]))].copy()

plt.figure(figsize=(12, 6))
sns.barplot(data=test_metrics_for_plot, x="model", y="f1_c3", hue="level")
plt.xticks(rotation=45, ha="right")
plt.title("Test F1 for C3 by Model and Evaluation Level")
plt.xlabel("Model")
plt.ylabel("F1 score for C3")
plt.tight_layout()
plt.savefig(THESIS_FIGURE_DIR / "fig_results_test_f1_by_model_and_level.png", dpi=300)
plt.close()

for level in ["sentence", "document"]:
    level_df = metrics_df[(metrics_df["split"] == "test") & (metrics_df["level"] == level)].copy()
    if not level_df.empty:
        level_df = level_df.sort_values("f1_c3", ascending=False)
        plt.figure(figsize=(11, 5))
        sns.barplot(data=level_df, x="model", y="f1_c3")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Test {level.capitalize()}-Level F1 by Model")
        plt.xlabel("Model")
        plt.ylabel("F1 score for C3")
        plt.tight_layout()
        plt.savefig(THESIS_FIGURE_DIR / f"fig_results_{level}_level_test_f1_by_model.png", dpi=300)
        plt.close()

metric_long = test_metrics_for_plot.melt(
    id_vars=["model", "level"],
    value_vars=[c for c in ["precision_c3", "recall_c3", "f1_c3"] if c in test_metrics_for_plot.columns],
    var_name="metric",
    value_name="score",
)
if not metric_long.empty:
    plt.figure(figsize=(14, 6))
    sns.barplot(data=metric_long, x="model", y="score", hue="metric")
    plt.xticks(rotation=45, ha="right")
    plt.title("Test Precision, Recall, and F1 by Model")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(THESIS_FIGURE_DIR / "fig_results_precision_recall_f1_grouped_by_model.png", dpi=300)
    plt.close()

# ---------- ROC and Precision-Recall curves for sentence-level predictions ----------
if not test_predictions_df.empty:
    plt.figure(figsize=(8, 6))
    for model_label, group in test_predictions_df.groupby("model"):
        y_true = group["true_C3"].astype(int).values
        scores = group["score_C3"].astype(float).values
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, scores)
            plt.plot(fpr, tpr, label=model_label)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Sentence-Level ROC Curves on Test Set")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(THESIS_FIGURE_DIR / "fig_results_sentence_level_roc_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for model_label, group in test_predictions_df.groupby("model"):
        y_true = group["true_C3"].astype(int).values
        scores = group["score_C3"].astype(float).values
        if len(np.unique(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, scores)
            plt.plot(recall, precision, label=model_label)
    plt.title("Sentence-Level Precision-Recall Curves on Test Set")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(THESIS_FIGURE_DIR / "fig_results_sentence_level_precision_recall_curves.png", dpi=300)
    plt.close()

# ---------- Confusion matrices for the best model ----------
if "best_test_predictions" in globals() and not best_test_predictions.empty:
    cm_sentence = confusion_matrix(best_test_predictions["true_C3"], best_test_predictions["pred_C3"], labels=[0, 1])
    cm_sentence_df = pd.DataFrame(cm_sentence, index=["True Non-C3", "True C3"], columns=["Pred Non-C3", "Pred C3"])
    save_table(cm_sentence_df.reset_index().rename(columns={"index": "true_label"}), "table_results_best_model_sentence_confusion_matrix")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_sentence, display_labels=["Non-C3", "C3"])
    disp.plot(values_format="d")
    plt.title(f"Sentence-Level Confusion Matrix: {best_model_name}")
    plt.tight_layout()
    plt.savefig(THESIS_FIGURE_DIR / "fig_results_best_model_sentence_confusion_matrix.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=best_test_predictions, x="score_C3", hue="true_C3", bins=50, stat="density", common_norm=False)
    plt.title(f"Predicted C3 Probability Distribution: {best_model_name}")
    plt.xlabel("Predicted probability for C3")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(THESIS_FIGURE_DIR / "fig_results_best_model_probability_distribution.png", dpi=300)
    plt.close()

    # Error rate by sentence length bin.
    error_len_df = best_test_predictions.copy()
    error_len_df["is_error"] = (error_len_df["true_C3"] != error_len_df["pred_C3"]).astype(int)
    error_len_df["length_bin"] = pd.qcut(error_len_df["word_len"], q=5, duplicates="drop")
    error_rate_by_length = error_len_df.groupby("length_bin").agg(
        n=("is_error", "size"),
        error_rate=("is_error", "mean"),
        mean_word_len=("word_len", "mean"),
    ).reset_index()
    error_rate_by_length["length_bin"] = error_rate_by_length["length_bin"].astype(str)
    save_table(error_rate_by_length, "table_error_analysis_error_rate_by_sentence_length")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=error_rate_by_length, x="length_bin", y="error_rate")
    plt.xticks(rotation=35, ha="right")
    plt.title("Best Model Error Rate by Sentence Length")
    plt.xlabel("Sentence length bin")
    plt.ylabel("Error rate")
    plt.tight_layout()
    plt.savefig(THESIS_FIGURE_DIR / "fig_error_analysis_error_rate_by_sentence_length.png", dpi=300)
    plt.close()

if "best_doc_predictions" in globals() and not best_doc_predictions.empty:
    cm_doc = confusion_matrix(best_doc_predictions["true_C3"], best_doc_predictions["pred_C3"], labels=[0, 1])
    cm_doc_df = pd.DataFrame(cm_doc, index=["True Non-C3", "True C3"], columns=["Pred Non-C3", "Pred C3"])
    save_table(cm_doc_df.reset_index().rename(columns={"index": "true_label"}), "table_results_best_model_document_confusion_matrix")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_doc, display_labels=["Non-C3", "C3"])
    disp.plot(values_format="d")
    plt.title(f"Document-Level Confusion Matrix: {best_model_name}")
    plt.tight_layout()
    plt.savefig(THESIS_FIGURE_DIR / "fig_results_best_model_document_confusion_matrix.png", dpi=300)
    plt.close()

# ---------- Feature-importance figures for RQ2 ----------
pos_path = METRICS_DIR / "top_positive_tfidf_features_c3.csv"
neg_path = METRICS_DIR / "top_negative_tfidf_features_c3.csv"
if pos_path.exists():
    top_positive = pd.read_csv(pos_path).head(20)
    save_table(top_positive, "table_rq2_top_positive_tfidf_features")
    plt.figure(figsize=(8, 8))
    sns.barplot(data=top_positive, y="feature", x="coefficient")
    plt.title("Top Positive TF-IDF Features for C3")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(THESIS_FIGURE_DIR / "fig_rq2_top_positive_tfidf_features.png", dpi=300)
    plt.close()

if neg_path.exists():
    top_negative = pd.read_csv(neg_path).head(20)
    save_table(top_negative, "table_rq2_top_negative_tfidf_features")
    plt.figure(figsize=(8, 8))
    sns.barplot(data=top_negative, y="feature", x="coefficient")
    plt.title("Top Negative TF-IDF Features for C3")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(THESIS_FIGURE_DIR / "fig_rq2_top_negative_tfidf_features.png", dpi=300)
    plt.close()

# ---------- Compact thesis summary markdown ----------
summary_lines = []
summary_lines.append("# Thesis-ready Results Summary\n")
summary_lines.append(f"Best selected model: **{best_model_name}**.\n")
summary_lines.append("Selection rule: highest test document-level F1, because the main task is document-level C3 risk disclosure prediction.\n")
summary_lines.append("\n## Key tables\n")
for p in sorted(THESIS_TABLE_DIR.glob("*.csv")):
    summary_lines.append(f"- `{p.name}`\n")
summary_lines.append("\n## Key figures\n")
for p in sorted(THESIS_FIGURE_DIR.glob("*.png")):
    summary_lines.append(f"- `{p.name}`\n")

with open(THESIS_REPORT_DIR / "thesis_ready_results_summary.md", "w", encoding="utf-8") as f:
    f.writelines(summary_lines)

# ---------- Display key tables and figures in notebook output ----------
display(Markdown("### Thesis-ready model-comparison tables"))
if not sentence_test_table.empty:
    display(Markdown("**Sentence-level test comparison**"))
    display(sentence_test_table)
if not document_test_table.empty:
    display(Markdown("**Document-level test comparison**"))
    display(document_test_table)
if not final_ranking.empty:
    display(Markdown("**Final model ranking**"))
    display(final_ranking)

figures_to_display = [
    "fig_results_test_f1_by_model_and_level.png",
    "fig_results_sentence_level_test_f1_by_model.png",
    "fig_results_document_level_test_f1_by_model.png",
    "fig_results_precision_recall_f1_grouped_by_model.png",
    "fig_results_sentence_level_roc_curves.png",
    "fig_results_sentence_level_precision_recall_curves.png",
    "fig_results_best_model_sentence_confusion_matrix.png",
    "fig_results_best_model_document_confusion_matrix.png",
    "fig_results_best_model_probability_distribution.png",
    "fig_error_analysis_error_rate_by_sentence_length.png",
    "fig_rq2_top_positive_tfidf_features.png",
    "fig_rq2_top_negative_tfidf_features.png",
]
display(Markdown("### Thesis-ready figures"))
display_figures(figures_to_display)

print("Thesis reporting finished.")
print("Tables saved to:", THESIS_TABLE_DIR.resolve())
print("Figures saved to:", THESIS_FIGURE_DIR.resolve())
print("Markdown summary saved to:", (THESIS_REPORT_DIR / "thesis_ready_results_summary.md").resolve())


# %%

# ============================================================
# 15. Final run manifest
# ============================================================
manifest = {
    "created_at": datetime.utcnow().isoformat() + "Z",
    "project_dir": str(PROJECT_DIR.resolve()),
    "raw_data_dir": str(RAW_DATA_DIR.resolve()),
    "processed_data_dir": str(PROCESSED_DATA_DIR.resolve()),
    "output_dir": str(OUTPUT_DIR.resolve()),
    "best_model_name": best_model_name,
    "files": {
        "metrics": [str(p.relative_to(PROJECT_DIR)) for p in sorted(METRICS_DIR.glob("*"))],
        "figures": [str(p.relative_to(PROJECT_DIR)) for p in sorted(FIGURE_DIR.glob("*"))],
        "predictions": [str(p.relative_to(PROJECT_DIR)) for p in sorted(PREDICTIONS_DIR.glob("*"))],
        "models": [str(p.relative_to(PROJECT_DIR)) for p in sorted(MODELS_DIR.glob("*"))],
        "logs": [str(p.relative_to(PROJECT_DIR)) for p in sorted(LOGS_DIR.glob("*"))],
    }
}

with open(OUTPUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("Pipeline finished successfully.")
print("Outputs saved to:", OUTPUT_DIR.resolve())
print(json.dumps(manifest, indent=2)[:3000])


# %%

# ============================================================
# 16. Optional local project archive
# ============================================================
import shutil

archive_path = shutil.make_archive(
    str(PROJECT_DIR.resolve()),
    "zip",
    str(PROJECT_DIR.resolve())
)

print("ZIP file created successfully:", archive_path)
