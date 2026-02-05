# tools/03_split_dataset.py
# -- Walmart sales forecast: step03 time-based train/valid split --
# -- Split tabular.parquet and seq.npz using the same date boundary --

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def split_tabular(
    df: pd.DataFrame,
    valid_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    # -- Time-based split using Date quantile --
    if "Date" not in df.columns:
        raise ValueError("tabular data must contain Date column")

    df = df.sort_values("Date").reset_index(drop=True)
    split_date = df["Date"].quantile(1.0 - valid_ratio)

    train_df = df[df["Date"] < split_date].reset_index(drop=True)
    valid_df = df[df["Date"] >= split_date].reset_index(drop=True)

    return train_df, valid_df, split_date


def split_seq(
    seq: dict[str, np.ndarray],
    split_date_ns: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # -- Split sequence samples by target date --
    mask_train = seq["date_ns"] < split_date_ns
    mask_valid = seq["date_ns"] >= split_date_ns

    train = {k: v[mask_train] for k, v in seq.items()}
    valid = {k: v[mask_valid] for k, v in seq.items()}

    return train, valid


def save_npz(path: Path, arrs: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrs)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path.home() / "projects" / "walmart_sale_forecast" / "data"),
        help="Data directory",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.2,
        help="Validation ratio by time",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)

    # -- Load tabular -- 
    tab_path = data_dir / "tabular.parquet"
    df_tab = pd.read_parquet(tab_path)
    df_tab["Date"] = pd.to_datetime(df_tab["Date"])

    train_tab, valid_tab, split_date = split_tabular(df_tab, args.valid_ratio)

    train_tab.to_parquet(data_dir / "tabular_train.parquet", index=False)
    valid_tab.to_parquet(data_dir / "tabular_valid.parquet", index=False)

    # -- Load seq -- 
    seq = dict(np.load(data_dir / "seq.npz"))

    train_seq, valid_seq = split_seq(seq, split_date_ns=split_date.value)

    save_npz(data_dir / "seq_train.npz", train_seq)
    save_npz(data_dir / "seq_valid.npz", valid_seq)

    # -- Logs --
    print(f"[OK] split date = {split_date.date()}")
    print(f"[INFO] tabular train={len(train_tab)} valid={len(valid_tab)}")
    print(f"[INFO] seq train={train_seq['X'].shape[0]} valid={valid_seq['X'].shape[0]}")


if __name__ == "__main__":
    main()
