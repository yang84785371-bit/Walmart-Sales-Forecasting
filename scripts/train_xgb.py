# scripts/train_xgb.py
# -- Walmart sales forecast: XGBoost with log1p target + manual early stopping --
# -- Train on tabular_train.parquet, validate on tabular_valid.parquet --
# -- Outputs:
# -- - output/xgb_metrics.json
# -- - output/xgb_pred_valid.csv

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_schema(schema_path: Path) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1.0, None)))
    wape = np.sum(np.abs(y_true - y_pred)) / np.clip(np.sum(np.abs(y_true)), 1.0, None)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "WAPE": float(wape),
    }


def train_with_manual_early_stop(
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_valid: np.ndarray,
    y_valid_log: np.ndarray,
    max_rounds: int,
    step_rounds: int,
    patience_steps: int,
    params: dict,
) -> Tuple[xgb.XGBRegressor, int, float]:
    # -- Manual early stopping for older xgboost sklearn wrapper
    best_rmse = float("inf")
    best_rounds = step_rounds
    bad = 0

    for n_estimators in range(step_rounds, max_rounds + 1, step_rounds):
        model = xgb.XGBRegressor(**params, n_estimators=n_estimators)
        model.fit(X_train, y_train_log, eval_set=[(X_valid, y_valid_log)], verbose=False)

        pred_log = model.predict(X_valid)
        rmse = float(np.sqrt(np.mean((pred_log - y_valid_log) ** 2)))

        if rmse < best_rmse - 1e-6:
            best_rmse = rmse
            best_rounds = n_estimators
            bad = 0
        else:
            bad += 1
            if bad >= patience_steps:
                break

    best_model = xgb.XGBRegressor(**params, n_estimators=best_rounds)
    best_model.fit(X_train, y_train_log, eval_set=[(X_valid, y_valid_log)], verbose=False)
    return best_model, best_rounds, best_rmse


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
    parser.add_argument("--max_rounds", type=int, default=3000)
    parser.add_argument("--step_rounds", type=int, default=200)
    parser.add_argument("--patience_steps", type=int, default=3)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(data_dir / "tabular_train.parquet")
    valid_df = pd.read_parquet(data_dir / "tabular_valid.parquet")

    schema = load_schema(data_dir / "feature_schema.json")
    feat_cols = schema["features"]
    target_col = schema["target"]
    meta_cols = schema["meta"]

    X_train = train_df[feat_cols].to_numpy()
    y_train = train_df[target_col].to_numpy(dtype=np.float32)

    X_valid = valid_df[feat_cols].to_numpy()
    y_valid = valid_df[target_col].to_numpy(dtype=np.float32)

    # -- log1p target
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    # -- Params
    params = {
        "learning_rate": 0.03,
        "max_depth": 10,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    }

    model, best_rounds, best_valid_rmse_log = train_with_manual_early_stop(
        X_train=X_train,
        y_train_log=y_train_log,
        X_valid=X_valid,
        y_valid_log=y_valid_log,
        max_rounds=args.max_rounds,
        step_rounds=args.step_rounds,
        patience_steps=args.patience_steps,
        params=params,
    )

    y_pred_log = model.predict(X_valid)
    y_pred = np.expm1(y_pred_log)

    metrics = evaluate(y_valid, y_pred)

    with open(output_dir / "xgb_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    out_df = valid_df[meta_cols].copy()
    out_df["y_true"] = y_valid.astype(float)
    out_df["y_pred"] = y_pred.astype(float)
    out_df.to_csv(output_dir / "xgb_pred_valid.csv", index=False)

    print("[OK] XGBoost training finished")
    print(f"[INFO] best_rounds = {best_rounds}")
    print(f"[INFO] best_valid_rmse_log = {best_valid_rmse_log:.6f}")
    for k, v in metrics.items():
        print(f"[METRIC] {k} = {v:.4f}")


if __name__ == "__main__":
    main()


