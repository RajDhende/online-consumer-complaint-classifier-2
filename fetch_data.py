"""
Build the real-world dataset used by the project.

Usage:
    uv run python fetch_data.py                # build a 40k complaint sample
    uv run python fetch_data.py --rows 60000   # larger sample
    uv run python fetch_data.py --force        # rebuild the sample

Dataset provenance
------------------
The Consumer Financial Protection Bureau (CFPB) publishes the Consumer
Complaint Database at:
    https://www.consumerfinance.gov/data-research/consumer-complaints/

The current official bulk export (complaints.csv.zip) contains complaint
metadata but NOT the free-text consumer narrative, so it cannot train a text
classifier. This project therefore samples a public mirror of the same CFPB
database that preserves the narrative column:
    https://huggingface.co/datasets/sovai/cfpb_complaints

Both sources describe the same real complaints. No synthetic records are
created. The application also accepts any user CSV that carries a text column.
"""

from __future__ import annotations

import argparse

import config
import data_loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare CFPB complaint data."
    )
    parser.add_argument(
        "--rows", type=int, default=config.SAMPLE_ROWS, help="sample size to keep"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild data/complaints.csv even if it exists",
    )
    args = parser.parse_args()

    config.ensure_directories()

    if config.DATASET_PATH.exists() and not args.force:
        print(
            f"Dataset already exists at {config.DATASET_PATH}. Use --force to rebuild."
        )
        return

    print("CFPB Consumer Complaint Database")
    print(f"  official page : {config.CFPB_HOME}")
    print(f"  mirror used   : {config.CFPB_MIRROR_URL}")
    print(f"  sample target : {args.rows:,} narrative complaints\n")

    try:
        df = data_loader.prepare_dataset(force=args.force, sample_rows=args.rows)
    except Exception as exc:  # noqa: BLE001 - clear message beats a traceback
        raise SystemExit(
            f"\nDataset preparation failed: {exc}\n"
            "Place the public parquet mirror at data/cfpb_full.parquet and rerun, "
            "or upload your own CSV from the Dataset Explorer page."
        ) from exc

    print("\nDataset ready.")
    print(f"  rows        : {len(df):,}")
    print(f"  categories  : {df[config.TARGET_COLUMN].nunique():,}")
    if "date_received" in df.columns:
        dates = df["date_received"].dropna()
        if not dates.empty:
            print(f"  date range  : {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}")
    active_path = (
        config.DATASET_PATH
        if config.DATASET_PATH.exists()
        else config.DATASET_GZIP_PATH
    )
    print(f"  saved to    : {active_path}")

    # A gzip copy keeps the repository clone small (GitHub blocks files >100 MB
    # and its web uploader stops at 25 MB). pandas reads .csv.gz transparently.
    # Only build it when a plain CSV exists; the shipped package already carries
    # the gzip and no plain CSV.
    if config.DATASET_PATH.exists():
        import gzip
        import shutil

        with (
            open(config.DATASET_PATH, "rb") as source,
            gzip.open(config.DATASET_GZIP_PATH, "wb", compresslevel=9) as target,
        ):
            shutil.copyfileobj(source, target)

        size_mb = config.DATASET_GZIP_PATH.stat().st_size / 1e6
        print(f"  gzip copy   : {config.DATASET_GZIP_PATH} ({size_mb:,.1f} MB)")
    print("\nNext step:  uv run python train_model.py")


if __name__ == "__main__":
    main()
