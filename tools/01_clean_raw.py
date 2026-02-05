'''
tools/01_clean_raw.py
    Walmart sales forecast: step01 clean & merge raw data 
Notes:
    Output is a single clean table: one row per (Store, Dept, Date)
'''
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def _parse_date(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Missing date column '{col}'")
    df[col] = pd.to_datetime(df[col], errors="coerce")
    bad = df[col].isna().sum()
    if bad > 0:
        raise ValueError(f"Found {bad} rows with invalid Date parsing.")
    return df


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def _ensure_bool_is_holiday(df: pd.DataFrame, col: str = "IsHoliday") -> pd.DataFrame:
    if col not in df.columns:
        return df

    if df[col].dtype == bool:
        return df

    s = df[col].astype(str).str.strip().str.lower()
    df[col] = s.isin(["true", "1", "t", "yes", "y"])
    return df


def _dedup_features(features: pd.DataFrame) -> pd.DataFrame:
    # -- features.csv sometimes has duplicates per (Store, Date) -- 
    # -- We aggregate numeric columns by mean and IsHoliday by max -- 
    keys = ["Store", "Date"]
    if not all(k in features.columns for k in keys):
        raise ValueError("features.csv must contain columns: Store, Date")

    features = features.copy()

    agg = {}
    for c in features.columns:
        if c in keys:
            continue
        if c == "IsHoliday":
            agg[c] = "max"
        else:
            if pd.api.types.is_numeric_dtype(features[c]):
                agg[c] = "mean"
            else:
                # -- keep the most frequent non-null value
                agg[c] = lambda x: x.dropna().mode().iloc[0] if x.dropna().size > 0 else np.nan

    out = features.groupby(keys, as_index=False).agg(agg)
    return out


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Basic cleaning rules:
        MarkDown1-5: fill missing with 0 (common convention in this dataset)
        Keep other missing values for modeling stage (we can impute later)
    '''
    df = df.copy()

    md_cols = [c for c in df.columns if c.lower().startswith("markdown")]
    for c in md_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(0.0)

    # -- Make sure key columns are correct dtypes --
    for c in ["Store", "Dept"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise").astype(int)

    if "Weekly_Sales" in df.columns:
        df["Weekly_Sales"] = pd.to_numeric(df["Weekly_Sales"], errors="coerce")
        # -- Drop rows with invalid target values
        df = df.dropna(subset=["Weekly_Sales"])

    # -- Optional: remove negative sales (rare but sometimes appears) --
    if "Weekly_Sales" in df.columns:
        df = df[df["Weekly_Sales"] >= 0].copy()

    return df


def build_clean_table(raw_dir: Path) -> pd.DataFrame:
    # -- Read raw files --
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    features_path = raw_dir / "features.csv"
    stores_path = raw_dir / "stores.csv"

    train = _normalize_cols(_read_csv(train_path))
    test = _normalize_cols(_read_csv(test_path))
    features = _normalize_cols(_read_csv(features_path))
    stores = _normalize_cols(_read_csv(stores_path))

    # -- Parse dates --
    train = _parse_date(train, "Date")
    test = _parse_date(test, "Date")
    features = _parse_date(features, "Date")

    # -- Normalize holiday flags --
    train = _ensure_bool_is_holiday(train, "IsHoliday")
    test = _ensure_bool_is_holiday(test, "IsHoliday")
    features = _ensure_bool_is_holiday(features, "IsHoliday")

    # -- Deduplicate features --
    features = _dedup_features(features)

    # -- Merge: train + features + stores --
    merged = train.merge(features, on=["Store", "Date"], how="left", suffixes=("", "_feat"))
    merged = merged.merge(stores, on=["Store"], how="left", suffixes=("", "_store"))

    # -- Some files have both train.IsHoliday and features.IsHoliday
    # -- Prefer the one from train if exists; otherwise use features
    if "IsHoliday_feat" in merged.columns:
        merged["IsHoliday"] = merged["IsHoliday"].fillna(merged["IsHoliday_feat"])
        merged = merged.drop(columns=["IsHoliday_feat"])

    merged = _basic_clean(merged)

    # -- Sort for time-based processing later
    sort_cols = [c for c in ["Store", "Dept", "Date"] if c in merged.columns]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)

    return merged


def save_outputs(df: pd.DataFrame, out_dir: Path, fmt: str = "parquet") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        out_path = out_dir / "clean.parquet"
        df.to_parquet(out_path, index=False)
        return out_path

    if fmt == "csv":
        out_path = out_dir / "clean.csv"
        df.to_csv(out_path, index=False)
        return out_path

    raise ValueError("fmt must be one of: parquet, csv")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=str(Path.home() / "projects" / "datasets" / "walmart_sales_data"),
        help="Raw data folder containing train.csv/test.csv/features.csv/stores.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path.home() / "projects" / "walmart_sale_forecast" / "data"),
        help="Output folder for cleaned data",
    )
    parser.add_argument(
        "--fmt",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Output format",
    )
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    df = build_clean_table(raw_dir)
    out_path = save_outputs(df, out_dir, fmt=args.fmt)

    # -- Basic sanity prints --
    print(f"[OK] saved: {out_path}")
    print(f"[INFO] rows={len(df)} cols={df.shape[1]}")
    must_cols = ["Store", "Dept", "Date", "Weekly_Sales"]
    missing = [c for c in must_cols if c not in df.columns]
    if missing:
        print(f"[WARN] missing expected columns: {missing}")
    else:
        dmin, dmax = df["Date"].min(), df["Date"].max()
        print(f"[INFO] date range: {dmin.date()} -> {dmax.date()}")


if __name__ == "__main__":
    main()

