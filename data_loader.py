"""
Data acquisition and loading utilities.

Two genuine data paths are supported:

1. The official CFPB Consumer Complaint Database bulk download
   (``complaints.csv.zip``), streamed and reservoir-sampled so the class
   project stays a sensible size.
2. Any user-supplied CSV that contains a complaint-text column and,
   optionally, a category column.

No synthetic or fake records are created anywhere in this module.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import requests

import config


# --------------------------------------------------------------------------- #
# Column handling
# --------------------------------------------------------------------------- #
def _normalise(name: str) -> str:
    """Lower-case, collapse whitespace and strip a column name."""
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _alias_lookup() -> dict[str, str]:
    """Map every accepted alias to its canonical column name."""
    lookup: dict[str, str] = {}
    for canonical, aliases in config.COLUMN_ALIASES.items():
        lookup[_normalise(canonical)] = canonical
        for alias in aliases:
            lookup[_normalise(alias)] = canonical
    return lookup


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename recognised columns to canonical names, keep unknown columns."""
    lookup = _alias_lookup()
    rename: dict[str, str] = {}
    used: set[str] = set()
    for column in df.columns:
        canonical = lookup.get(_normalise(column))
        if canonical and canonical not in used:
            rename[column] = canonical
            used.add(canonical)
    return df.rename(columns=rename)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes: dates to datetime, text columns to stripped strings."""
    df = df.copy()
    if "date_received" in df.columns:
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

    text_columns = [
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
    for column in text_columns:
        if column in df.columns:
            series = df[column].astype("string")
            series = series.str.replace(r"\s+", " ", regex=True).str.strip()
            series = series.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            df[column] = series
    return df


def normalize_products(df: pd.DataFrame) -> pd.DataFrame:
    """Merge historical CFPB product names into one comparable label set."""
    if config.TARGET_COLUMN not in df.columns:
        return df
    work = df.copy()
    work[config.TARGET_COLUMN] = work[config.TARGET_COLUMN].astype("string").str.strip()
    work[config.TARGET_COLUMN] = work[config.TARGET_COLUMN].replace(
        config.PRODUCT_NORMALIZATION
    )
    return work


# --------------------------------------------------------------------------- #
# Bulk download and sampling
# --------------------------------------------------------------------------- #
def download_file(
    url: str,
    destination: Path,
    timeout: int = 120,
    min_bytes: int = 1_000_000,
) -> Path:
    """Stream a large public file to disk, reusing a valid cached copy."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > min_bytes:
        return destination

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    handle.write(block)
                    downloaded += len(block)
                    if total and downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                        print(
                            f"  downloaded {downloaded / 1e6:,.1f} / {total / 1e6:,.1f} MB",
                            flush=True,
                        )
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return destination


def download_bulk_dataset(
    url: str = config.CFPB_BULK_URL,
    destination: Path = config.RAW_ZIP_PATH,
    timeout: int = 120,
) -> Path:
    """Backward-compatible wrapper around :func:`download_file`."""
    return download_file(url, destination, timeout=timeout)


def _bulk_has_narrative(zip_path: Path) -> bool:
    """Check the official archive header for a narrative column."""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            member = _csv_member(archive)
            with archive.open(member) as handle:
                header = handle.readline(4096).decode("utf-8", errors="replace").lower()
        return "narrative" in header
    except Exception:
        return False


def build_sample_from_parquet(
    parquet_path: Path = config.PARQUET_PATH,
    output_path: Path = config.DATASET_PATH,
    sample_rows: int = config.SAMPLE_ROWS,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    Draw a reproducible random sample from the public CFPB parquet mirror.

    DuckDB filters the narrative-bearing rows first, then samples, so the
    result is an unbiased sample of the eligible population. The mirror is a
    public snapshot of the same CFPB Consumer Complaint Database; it is used
    because the current official bulk CSV no longer carries narrative text.
    """
    import duckdb

    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found.")

    escaped = str(parquet_path).replace("'", "''")
    query = f"""
        SELECT * FROM (
            SELECT
                product,
                consumer_complaint_narrative,
                date,
                company,
                state,
                company_response_to_consumer,
                timely_response,
                consumer_disputed,
                sub_product,
                issue,
                submitted_via
            FROM read_parquet('{escaped}')
            WHERE consumer_complaint_narrative IS NOT NULL
              AND length(trim(consumer_complaint_narrative)) > 80
        ) AS eligible
        USING SAMPLE {int(sample_rows)} ROWS (reservoir, {int(seed)})
    """
    connection = duckdb.connect()
    try:
        frame = connection.execute(query).fetchdf()
    finally:
        connection.close()

    frame = normalize_columns(frame)
    frame = coerce_types(frame)
    frame = normalize_products(frame)

    ordered = [column for column in config.CANONICAL_COLUMNS if column in frame.columns]
    extra = [column for column in frame.columns if column not in ordered]
    frame = frame[ordered + extra]
    frame = frame.sort_values("date_received", ascending=False, na_position="last")
    frame = frame.reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"  saved {len(frame):,} complaints to {output_path}")
    return frame


def _csv_member(archive: zipfile.ZipFile) -> str:
    """Return the name of the first CSV member inside the archive."""
    members = [
        info.filename
        for info in archive.infolist()
        if info.filename.lower().endswith(".csv")
    ]
    if not members:
        raise ValueError("The downloaded archive does not contain a CSV file.")
    return members[0]


def _iter_chunks(zip_path: Path, chunk_size: int = 25_000) -> Iterator[pd.DataFrame]:
    """Stream chunks from the CSV stored inside the ZIP archive."""
    with zipfile.ZipFile(zip_path) as archive:
        member = _csv_member(archive)
        with archive.open(member) as handle:
            text_stream = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
            yield from pd.read_csv(
                text_stream,
                chunksize=chunk_size,
                low_memory=False,
                on_bad_lines="skip",
            )


def build_sample_from_bulk(
    zip_path: Path = config.RAW_ZIP_PATH,
    output_path: Path = config.DATASET_PATH,
    sample_rows: int = config.SAMPLE_ROWS,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    Stream the bulk file, keep narrative-bearing rows, and draw an unbiased
    reservoir sample.

    Reservoir sampling (A-Res) gives every narrative-bearing complaint the same
    chance of being kept, regardless of where it sits in the 350 MB file.
    """
    zip_path = Path(zip_path)
    output_path = Path(output_path)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Run `python fetch_data.py` first, or place the "
            "official CFPB complaints.csv.zip in the data/ folder."
        )

    rng = np.random.default_rng(seed)
    reservoir: pd.DataFrame | None = None
    scanned = 0
    eligible = 0

    for chunk in _iter_chunks(zip_path):
        scanned += len(chunk)
        chunk = normalize_columns(chunk)
        if config.TEXT_COLUMN not in chunk.columns:
            raise ValueError(
                "Could not find a complaint narrative column. "
                f"Columns seen: {list(chunk.columns)[:15]}"
            )

        narrative = chunk[config.TEXT_COLUMN].astype("string").str.strip()
        keep = narrative.notna() & (narrative.str.len() > 0)
        chunk = chunk.loc[keep].copy()
        if chunk.empty:
            continue

        chunk["_reservoir_key"] = rng.random(len(chunk))
        eligible += len(chunk)
        reservoir = (
            chunk
            if reservoir is None
            else pd.concat([reservoir, chunk], ignore_index=True)
        )
        if len(reservoir) > sample_rows:
            reservoir = reservoir.nlargest(sample_rows, "_reservoir_key")

        if scanned % 500_000 < 25_000:
            print(
                f"  scanned {scanned:,} rows | narratives seen {eligible:,} | "
                f"reservoir {len(reservoir):,}",
                flush=True,
            )

    if reservoir is None or reservoir.empty:
        raise ValueError("No complaint narratives were found in the bulk dataset.")

    reservoir = reservoir.drop(columns=["_reservoir_key"], errors="ignore")
    reservoir = coerce_types(reservoir)

    ordered = [
        column for column in config.CANONICAL_COLUMNS if column in reservoir.columns
    ]
    extra = [column for column in reservoir.columns if column not in ordered]
    reservoir = reservoir[ordered + extra]
    reservoir = reservoir.sort_values(
        "date_received", ascending=False, na_position="last"
    )
    reservoir = reservoir.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reservoir.to_csv(output_path, index=False)

    print(
        f"  saved {len(reservoir):,} narrative complaints to {output_path} "
        f"(scanned {scanned:,} rows, {eligible:,} narratives)"
    )
    return reservoir


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def prepare_dataset(
    force: bool = False, sample_rows: int | None = None
) -> pd.DataFrame:
    """
    Return the modelling dataset, building it from the best available source.

    Order of preference:
    1. an existing ``data/complaints.csv``;
    2. the public CFPB parquet mirror (keeps narrative text);
    3. the official bulk ZIP, but only if it actually contains narratives.
    """
    rows = config.SAMPLE_ROWS if sample_rows is None else int(sample_rows)

    if config.DATASET_PATH.exists() and not force:
        return load_dataset()

    if config.DATASET_GZIP_PATH.exists() and not force:
        return load_dataset()

    if config.PARQUET_PATH.exists():
        print("Building the sample from the public CFPB parquet mirror ...")
        return build_sample_from_parquet(sample_rows=rows)

    if config.RAW_ZIP_PATH.exists():
        if _bulk_has_narrative(config.RAW_ZIP_PATH):
            print("Building the sample from the official CFPB bulk CSV ...")
            return build_sample_from_bulk(sample_rows=rows)
        print(
            "The official ZIP does not contain a narrative column "
            "(checked its header), so it cannot train a text model."
        )

    print("Downloading the public CFPB mirror that preserves narratives ...")
    download_file(config.CFPB_MIRROR_PARQUET, config.PARQUET_PATH, timeout=180)
    return build_sample_from_parquet(sample_rows=rows)


def load_dataset(path: Path = config.DATASET_PATH) -> pd.DataFrame:
    """Load a complaints CSV (plain or gzip) and normalise it for the app."""
    path = Path(path)
    if not path.exists():
        compressed = path.parent / f"{path.name}.gz"
        if compressed.exists():
            path = compressed
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path} (or {compressed.name}). "
                "Run `python fetch_data.py` to build it."
            )
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)
    df = coerce_types(df)
    df = normalize_products(df)
    return df


def validate_upload(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check that an uploaded dataframe can be used by the application.

    Returns ``(ok, messages)`` where messages explain every problem found.
    """
    messages: list[str] = []
    df = normalize_columns(df)

    if df.empty:
        messages.append("The uploaded file contains no rows.")
    if config.TEXT_COLUMN not in df.columns:
        messages.append(
            "No complaint-text column found. Expected one of: "
            + ", ".join(config.COLUMN_ALIASES["narrative"])
        )
    else:
        non_empty = (
            df[config.TEXT_COLUMN]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .notna()
        )
        if non_empty.sum() == 0:
            messages.append("The complaint-text column is empty for every row.")
        elif non_empty.sum() < len(df):
            messages.append(
                f"{len(df) - int(non_empty.sum()):,} row(s) have no complaint text "
                "and will be skipped during classification."
            )

    return (
        not any(
            m.startswith(("The uploaded file contains no rows", "No complaint-text"))
            for m in messages
        ),
        messages,
    )


def dataset_summary(df: pd.DataFrame) -> dict:
    """Small, JSON-friendly summary used by the Dataset Explorer page."""
    summary: dict = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
    }
    if "date_received" in df.columns:
        dates = pd.to_datetime(df["date_received"], errors="coerce").dropna()
        if not dates.empty:
            summary["date_min"] = dates.min().strftime("%Y-%m-%d")
            summary["date_max"] = dates.max().strftime("%Y-%m-%d")
    if config.TARGET_COLUMN in df.columns:
        summary["categories"] = int(df[config.TARGET_COLUMN].nunique(dropna=True))
    return summary
