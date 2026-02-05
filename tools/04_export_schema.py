# tools/04_export_schema.py
# -- Walmart sales forecast: step04 export feature schema --
# -- Purpose: fix feature columns to avoid train/valid/test mismatch --

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def export_schema(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    # -- Define roles
    target_col = "Weekly_Sales"
    meta_cols = ["Store", "Dept", "Date"]

    # -- Feature columns = numeric / one-hot cols except target & meta --
    feature_cols = [
        c for c in df.columns
        if c not in meta_cols + [target_col]
    ]

    schema = {
        "target": target_col,
        "features": feature_cols,
        "meta": meta_cols,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path.home() / "projects" / "walmart_sale_forecast" / "data"),
        help="Data directory",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)

    train_path = data_dir / "tabular_train.parquet"
    if not train_path.exists():
        raise FileNotFoundError("tabular_train.parquet not found")

    df_train = pd.read_parquet(train_path)

    schema_path = data_dir / "feature_schema.json"
    export_schema(df_train, schema_path)

    # -- Logs
    print(f"[OK] saved: {schema_path}")
    print(f"[INFO] feature count = {len(df_train.columns) - 4}")
    print("[INFO] schema locked for all downstream models")


if __name__ == "__main__":
    main()
