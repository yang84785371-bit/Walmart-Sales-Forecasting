# tools/07_plot_valid_compare.py
# -- Plot valid predictions vs truth for multiple models --
# -- Fix: align sampling indices across models by using min length --

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_pred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    need = {"Date", "y_true", "y_pred"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"missing columns in {path.name}: {sorted(list(miss))}")
    return df


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--models",
        type=str,
        default="rf,xgb,lstm,transformer_embed",
        help="Comma-separated model keys",
    )
    parser.add_argument("--max_points", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # -- Map model key to prediction file
    pred_map: Dict[str, Path] = {
        "rf": output_dir / "rf_pred_valid.csv",
        "xgb": output_dir / "xgb_pred_valid.csv",
        "lstm": output_dir / "lstm_pred_valid.csv",
        "transformer": output_dir / "transformer_pred_valid.csv",
        "lstm_embed": output_dir / "lstm_embed_pred_valid.csv",
        "lstm_embed_res": output_dir / "lstm_embed_res_pred_valid.csv",
        "transformer_embed": output_dir / "transformer_embed_pred_valid.csv",
    }

    dfs: Dict[str, pd.DataFrame] = {}
    lengths: List[int] = []

    for m in models:
        if m not in pred_map:
            raise ValueError(f"unknown model key: {m}")
        p = pred_map[m]
        if not p.exists():
            raise FileNotFoundError(f"pred file not found: {p}")
        df = load_pred_csv(p)
        dfs[m] = df
        lengths.append(len(df))

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -- Plot 1: total sales by date (each model uses its own df)
    plt.figure()
    plotted_true = False
    for m, df in dfs.items():
        g = df.groupby("Date", as_index=False)[["y_true", "y_pred"]].sum()
        if not plotted_true:
            plt.plot(g["Date"], g["y_true"], label="true_total")
            plotted_true = True
        plt.plot(g["Date"], g["y_pred"], label=f"{m}_pred_total")
    plt.title("Valid: Total Weekly Sales (sum over Store/Dept)")
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales")
    plt.legend()
    out1 = fig_dir / "valid_total_sales_timeseries.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"[OK] saved: {out1}")

    # -- Sampling indices aligned by min length
    n = int(min(lengths))
    rng = np.random.default_rng(args.seed)
    k = int(min(args.max_points, n))
    if k < n:
        idx = rng.choice(n, size=k, replace=False)
    else:
        idx = np.arange(n)

    # -- Plot 2: scatter y_true vs y_pred (sampled)
    plt.figure()
    any_key = models[0]
    y_true = dfs[any_key]["y_true"].to_numpy()[:n][idx]
    plt.scatter(y_true, y_true, s=4, alpha=0.25, label="ideal")
    for m in models:
        y_pred = dfs[m]["y_pred"].to_numpy()[:n][idx]
        plt.scatter(y_true, y_pred, s=4, alpha=0.25, label=m)
    plt.title("Valid: y_true vs y_pred (sampled)")
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.legend()
    out2 = fig_dir / "valid_scatter_true_vs_pred.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"[OK] saved: {out2}")

    # -- Plot 3: abs error distribution (sampled)
    plt.figure()
    for m in models:
        yt = dfs[m]["y_true"].to_numpy()[:n][idx]
        yp = dfs[m]["y_pred"].to_numpy()[:n][idx]
        err = np.abs(yt - yp)
        plt.hist(err, bins=80, alpha=0.35, label=m)
    plt.title("Valid: Absolute Error Distribution (sampled)")
    plt.xlabel("|y_true - y_pred|")
    plt.ylabel("count")
    plt.legend()
    out3 = fig_dir / "valid_abs_error_hist.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"[OK] saved: {out3}")

    print(f"[DONE] figures saved in: {fig_dir}")
    print(f"[INFO] aligned length n = {n}, sampled k = {len(idx)}")


if __name__ == "__main__":
    main()

