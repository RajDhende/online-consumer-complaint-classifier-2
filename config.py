"""
Central configuration for the Online Consumer Complaint Classification System.

All paths, dataset constants, class labels and display helpers live here so that
the training script, the analytics layer and the Streamlit app share one source
of truth.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
ASSETS_DIR = BASE_DIR / "assets"

RAW_ZIP_PATH = DATA_DIR / "complaints.csv.zip"
PARQUET_PATH = DATA_DIR / "cfpb_full.parquet"
DATASET_PATH = DATA_DIR / "complaints.csv"
DATASET_GZIP_PATH = DATA_DIR / "complaints.csv.gz"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "trained_model.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"
CONFUSION_PATH = MODEL_DIR / "confusion_matrix.csv"
REPORT_PATH = MODEL_DIR / "classification_report.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"
CLASS_DISTRIBUTION_PATH = MODEL_DIR / "class_distribution.csv"

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
# Official public download (CFPB Consumer Complaint Database).
CFPB_BULK_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
CFPB_HOME = "https://www.consumerfinance.gov/data-research/consumer-complaints/"
CFPB_API_DOCS = "https://cfpb.github.io/api/ccdb/"
# The current official bulk export does not carry the free-text narrative
# column, so the project samples a public mirror of the same CFPB database
# that preserves narratives. Provenance is documented in the README/report.
CFPB_MIRROR_URL = "https://huggingface.co/datasets/sovai/cfpb_complaints"
CFPB_MIRROR_PARQUET = "https://huggingface.co/datasets/sovai/cfpb_complaints/resolve/main/cfpb_complaints.parquet"

# Number of narrative-bearing complaints kept for the class project.
# Chosen so training finishes in well under a minute on a laptop while still
# giving every product category thousands of examples.
SAMPLE_ROWS = 40_000
RANDOM_STATE = 42

# A product must have at least this many complaints to become a model class.
MIN_CLASS_COUNT = 150
MAX_CLASSES = 10
# Cap the dominant categories so one product cannot dominate the benchmark.
MAX_PER_CLASS = 4_000

# Narrative length guards (in words) applied during cleaning.
MIN_WORDS = 8
MAX_WORDS = 2_000

# Train/test split.
TEST_SIZE = 0.20

# TF-IDF hyper-parameters.
TFIDF_MAX_FEATURES = 60_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.90

# --------------------------------------------------------------------------- #
# Canonical columns
# --------------------------------------------------------------------------- #
# Internal names used everywhere after data loading.
CANONICAL_COLUMNS = [
    "complaint_id",
    "date_received",
    "product",
    "sub_product",
    "issue",
    "narrative",
    "company",
    "state",
    "submitted_via",
    "company_response",
    "timely_response",
    "consumer_disputed",
]

TEXT_COLUMN = "narrative"
TARGET_COLUMN = "product"

# Aliases accepted from the CFPB bulk file and from user CSV uploads.
# Keys are canonical names, values are normalised (lower-case) aliases.
COLUMN_ALIASES: dict[str, list[str]] = {
    "complaint_id": ["complaint id", "complaint_id", "id", "complaint number"],
    "date_received": ["date received", "date_received", "received date", "date"],
    "product": ["product", "category", "complaint category", "product category"],
    "sub_product": ["sub-product", "sub_product", "subproduct"],
    "issue": ["issue", "complaint issue"],
    "narrative": [
        "consumer complaint narrative",
        "consumer_complaint_narrative",
        "narrative",
        "complaint narrative",
        "complaint text",
        "complaint",
        "text",
        "review",
    ],
    "company": ["company", "company name"],
    "state": ["state", "consumer state"],
    "submitted_via": ["submitted via", "submitted_via", "submission channel"],
    "company_response": [
        "company response to consumer",
        "company_response",
        "company response",
        "response",
    ],
    "timely_response": [
        "timely response?",
        "timely_response",
        "timely",
        "timely response",
    ],
    "consumer_disputed": [
        "consumer disputed?",
        "consumer_disputed",
        "consumer disputed",
        "disputed",
    ],
}

# Friendly labels for the long official product names.
PRODUCT_DISPLAY_NAMES: dict[str, str] = {
    "Credit reporting": "Credit Reporting",
    "Debt collection": "Debt Collection",
    "Mortgage": "Mortgage",
    "Credit card": "Credit Card",
    "Checking or savings account": "Checking / Savings",
    "Money transfer": "Money Transfer",
    "Vehicle loan or lease": "Vehicle Loan / Lease",
    "Student loan": "Student Loan",
    "Payday or personal loan": "Payday / Personal Loan",
    "Prepaid card": "Prepaid Card",
    "Other financial service": "Other Financial Service",
}

# CFPB renamed several products over the years. Merging the historical names
# turns a fragmented 21-label problem into a clean, comparable one.
PRODUCT_NORMALIZATION: dict[str, str] = {
    "Credit reporting": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Debt collection": "Debt collection",
    "Debt or credit management": "Debt collection",
    "Mortgage": "Mortgage",
    "Credit card": "Credit card",
    "Credit card or prepaid card": "Credit card",
    "Checking or savings account": "Checking or savings account",
    "Bank account or service": "Checking or savings account",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Money transfers": "Money transfer",
    "Vehicle loan or lease": "Vehicle loan or lease",
    "Consumer Loan": "Vehicle loan or lease",
    "Student loan": "Student loan",
    "Payday loan, title loan, or personal loan": "Payday or personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday or personal loan",
    "Payday loan": "Payday or personal loan",
    "Prepaid card": "Prepaid card",
    "Other financial service": "Other financial service",
    "Virtual currency": "Virtual currency",
}

# Colour palette shared by Plotly and Matplotlib charts.
PALETTE = [
    "#4F46E5",  # indigo
    "#06B6D4",  # cyan
    "#8B5CF6",  # violet
    "#F59E0B",  # amber
    "#10B981",  # emerald
    "#EF4444",  # red
    "#EC4899",  # pink
    "#14B8A6",  # teal
    "#F97316",  # orange
    "#6366F1",  # indigo light
    "#84CC16",  # lime
]


def display_name(product: str) -> str:
    """Return a short, chart-friendly label for a product/category."""
    return PRODUCT_DISPLAY_NAMES.get(product, product)


def ensure_directories() -> None:
    """Create the folders the application writes to."""
    for folder in (DATA_DIR, MODEL_DIR, ASSETS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
