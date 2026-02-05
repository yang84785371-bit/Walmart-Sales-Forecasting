# tools/02_build_features.py
'''
Walmart sales forecast: step02 build features for tabular + sequence models 
Outputs:
    data/tabular.parquet : for RF/XGB
    data/seq.npz : for LSTM/Transformer (sliding windows)
Notes:
    We do NOT split train/valid here; split is step03
'''
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _load_clean(clean_path: Path) -> pd.DataFrame:
    if not clean_path.exists():
        raise FileNotFoundError(f"clean file not found: {clean_path}")
    df = pd.read_parquet(clean_path)
    need = ["Store", "Dept", "Date", "Weekly_Sales"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"clean file missing columns: {miss}")
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["Date"])
    df["year"] = dt.dt.year.astype(int)
    df["month"] = dt.dt.month.astype(int)
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["dayofweek"] = dt.dt.dayofweek.astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)

    if "IsHoliday" in df.columns:
        df["is_holiday"] = df["IsHoliday"].astype(bool).astype(int)
    else:
        df["is_holiday"] = 0

    return df


def _coerce_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fill_missing_basic(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Minimal imputation:
        Numeric columns: fill NaN with group median (Store,Dept), then global median
        Categorical: fill with 'Unknown'
    '''
    df = df.copy()

    cat_cols = []
    for c in df.columns:
        if df[c].dtype == "object":
            cat_cols.append(c)

    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    grp_keys = ["Store", "Dept"]

    for c in num_cols:
        if c in grp_keys:
            continue
        if c == "Weekly_Sales":
            continue
        gmed = df.groupby(grp_keys)[c].transform("median")
        df[c] = df[c].fillna(gmed)
        df[c] = df[c].fillna(df[c].median())

    return df


def _make_lag_rolling(df: pd.DataFrame, lags: List[int], rolls: List[int]) -> pd.DataFrame:
    # -- Build lag + rolling features safely per (Store, Dept) --
    df = df.copy()
    g = df.groupby(["Store", "Dept"], sort=False)

    # -- Lags of target
    for k in lags:
        df[f"lag_{k}"] = g["Weekly_Sales"].shift(k)

    # -- Rolling stats on shifted target to avoid leakage --
    # -- Use group-wise apply to avoid MultiIndex assumptions --
    shifted = g["Weekly_Sales"].shift(1)

    for w in rolls:
        df[f"roll_mean_{w}"] = shifted.groupby([df["Store"], df["Dept"]]).transform(
            lambda s: s.rolling(window=w, min_periods=1).mean()
        )
        df[f"roll_std_{w}"] = shifted.groupby([df["Store"], df["Dept"]]).transform(
            lambda s: s.rolling(window=w, min_periods=1).std()
        )
        df[f"roll_min_{w}"] = shifted.groupby([df["Store"], df["Dept"]]).transform(
            lambda s: s.rolling(window=w, min_periods=1).min()
        )
        df[f"roll_max_{w}"] = shifted.groupby([df["Store"], df["Dept"]]).transform(
            lambda s: s.rolling(window=w, min_periods=1).max()
        )

    return df



def _one_hot(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    if not cat_cols:
        return df
    return pd.get_dummies(df, columns=cat_cols, dummy_na=False)


def _build_tabular(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # -- Prepare base --
    df = df.copy()

    # -- Numeric cols in Walmart dataset --
    numeric_candidates = [
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "Size",
    ]
    df = _coerce_numeric_cols(df, numeric_candidates)

    # -- Calendar + lag/rolling --
    df = _add_calendar_features(df)
    df = _make_lag_rolling(df, lags=[1, 2, 4, 8, 12], rolls=[4, 8, 12])

    # -- Identify categorical columns  --
    cat_cols = []
    if "Type" in df.columns and df["Type"].dtype == "object":
        cat_cols.append("Type")

    df = _fill_missing_basic(df)
    df = _one_hot(df, cat_cols)

    # -- Define usable feature columns --
    drop_cols = ["Weekly_Sales", "Date", "IsHoliday"]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    # -- Remove rows where lag_1 is NaN  --
    if "lag_1" in df.columns:
        df = df.dropna(subset=["lag_1"]).reset_index(drop=True)

    return df, feat_cols


def _build_sequence_npz(
    df_tabular: pd.DataFrame,
    feat_cols: List[str],
    window: int,
    horizon: int,
) -> Dict[str, np.ndarray]:
    # -- Build (X_seq, y, meta) from tabular rows, grouped by (Store,Dept) --
    # -- Each sample uses past 'window' rows to predict the row at t+horizon-1 --
    if window <= 0:
        raise ValueError("window must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    df = df_tabular.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    X_list = []
    y_list = []
    meta_store = []
    meta_dept = []
    meta_date = []

    for (s, d), g in df.groupby(["Store", "Dept"], sort=False):
        g = g.sort_values("Date")
        Xg = g[feat_cols].to_numpy(dtype=np.float32)
        yg = g["Weekly_Sales"].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(g["Date"]).to_numpy()

        n = len(g)
        last_start = n - (window + horizon) + 1
        if last_start <= 0:
            continue

        for i in range(last_start):
            x_win = Xg[i : i + window]
            y_t = yg[i + window + horizon - 1]
            date_t = dates[i + window + horizon - 1]

            X_list.append(x_win)
            y_list.append(y_t)
            meta_store.append(int(s))
            meta_dept.append(int(d))
            meta_date.append(date_t.astype("datetime64[ns]").astype(np.int64))

    if not X_list:
        raise RuntimeError("No sequence samples were created. Try smaller window/horizon.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    store = np.array(meta_store, dtype=np.int32)
    dept = np.array(meta_dept, dtype=np.int32)
    date_ns = np.array(meta_date, dtype=np.int64)

    return {
        "X": X,
        "y": y,
        "store": store,
        "dept": dept,
        "date_ns": date_ns,
    }


def save_tabular(df_tab: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_tab.to_parquet(out_path, index=False)


def save_seq_npz(arrs: Dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrs)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean_path",
        type=str,
        default=str(Path.home() / "projects" / "walmart_sale_forecast" / "data" / "clean.parquet"),
        help="Path to clean.parquet",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path.home() / "projects" / "walmart_sale_forecast" / "data"),
        help="Output directory for tabular + seq",
    )
    parser.add_argument("--window", type=int, default=12, help="Sequence window length (weeks)")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon (steps ahead)")
    args = parser.parse_args(argv)

    clean_path = Path(args.clean_path)
    out_dir = Path(args.out_dir)

    df_clean = _load_clean(clean_path)

    df_tab, feat_cols = _build_tabular(df_clean)

    tab_path = out_dir / "tabular.parquet"
    save_tabular(df_tab, tab_path)

    seq_arrs = _build_sequence_npz(df_tab, feat_cols=feat_cols, window=args.window, horizon=args.horizon)
    seq_path = out_dir / "seq.npz"
    save_seq_npz(seq_arrs, seq_path)

    # -- Logs
    print(f"[OK] saved: {tab_path}")
    print(f"[INFO] tabular rows={len(df_tab)} cols={df_tab.shape[1]}")
    print(f"[INFO] feature cols={len(feat_cols)}")

    X = seq_arrs["X"]
    y = seq_arrs["y"]
    print(f"[OK] saved: {seq_path}")
    print(f"[INFO] seq X shape={X.shape} y shape={y.shape}")
    print(f"[INFO] window={args.window} horizon={args.horizon}")


if __name__ == "__main__":
    main()
