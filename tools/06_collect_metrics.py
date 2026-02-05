# tools/06_collect_metrics.py
# -- Collect metrics json files into one csv --

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--models",
        type=str,
        default="rf,xgb,lstm,transformer_embed",
        help="Comma-separated model keys",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # -- Map model key to metrics file
    metrics_map = {
        "rf": output_dir / "rf_metrics.json",
        "xgb": output_dir / "xgb_metrics.json",
        "lstm": output_dir / "lstm_metrics.json",
        "transformer": output_dir / "transformer_metrics.json",
        "lstm_embed": output_dir / "lstm_embed_metrics.json",
        "lstm_embed_res": output_dir / "lstm_embed_res_metrics.json",
        "transformer_embed": output_dir / "transformer_embed_metrics.json",
    }

    rows: List[Dict[str, Any]] = []
    for m in models:
        if m not in metrics_map:
            raise ValueError(f"unknown model key: {m}")
        p = metrics_map[m]
        if not p.exists():
            raise FileNotFoundError(f"metrics not found: {p}")
        d = read_json(p)
        d["model"] = m
        rows.append(d)

    df = pd.DataFrame(rows)

    # -- Order columns
    cols = ["model"]
    for c in ["MAE", "RMSE", "MAPE", "WAPE"]:
        if c in df.columns:
            cols.append(c)
    other = [c for c in df.columns if c not in cols]
    df = df[cols + other]

    out_path = output_dir / "model_metrics_summary.csv"
    df.to_csv(out_path, index=False)

    print(f"[OK] saved: {out_path}")
    print(df.sort_values("RMSE").to_string(index=False))


if __name__ == "__main__":
    main()
