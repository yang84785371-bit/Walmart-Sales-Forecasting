# scripts/train_rf.py
# -- Walmart sales forecast: RandomForest baseline model --
# -- Train on tabular_train.parquet, validate on tabular_valid.parquet --

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_schema(schema_path: Path) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1.0, None)))
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path.home() / "projects" / "walmart_sale_forecast" / "data"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.home() / "projects" / "walmart_sale_forecast" / "output"),
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load data --
    train_df = pd.read_parquet(data_dir / "tabular_train.parquet")
    valid_df = pd.read_parquet(data_dir / "tabular_valid.parquet")

    schema = load_schema(data_dir / "feature_schema.json")
    feat_cols = schema["features"]
    target_col = schema["target"]
    meta_cols = schema["meta"]

    X_train = train_df[feat_cols].to_numpy()
    y_train = train_df[target_col].to_numpy()

    X_valid = valid_df[feat_cols].to_numpy()
    y_valid = valid_df[target_col].to_numpy()

    # -- Model --
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # -- Predict --
    y_pred = model.predict(X_valid)

    # -- Evaluate --
    metrics = evaluate(y_valid, y_pred)

    # -- Save metrics --
    with open(output_dir / "rf_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # -- Save predictions --
    out_df = valid_df[meta_cols].copy()
    out_df["y_true"] = y_valid
    out_df["y_pred"] = y_pred
    out_df.to_csv(output_dir / "rf_pred_valid.csv", index=False)

    # -- Logs --
    print("[OK] RandomForest training finished")
    for k, v in metrics.items():
        print(f"[METRIC] {k} = {v:.4f}")


if __name__ == "__main__":
    main()
