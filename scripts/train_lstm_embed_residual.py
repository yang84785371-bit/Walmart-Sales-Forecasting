# scripts/train_lstm_embed_residual.py
# -- Walmart sales forecast: LSTM + Store/Dept embeddings residual calibration --
# -- Design:
# -- - LSTM only models temporal pattern from X
# -- - Store/Dept embeddings calibrate level via residual head
# -- - X standardization + log1p target
# -- Outputs:
# -- - output/lstm_embed_res_metrics.json
# -- - output/lstm_embed_res_pred_valid.csv
# -- - output/lstm_embed_res_preprocess_stats.json

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1.0, None))))
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.clip(np.sum(np.abs(y_true)), 1.0, None))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"npz not found: {path}")
    obj = dict(np.load(path))
    need = ["X", "y", "store", "dept", "date_ns"]
    miss = [k for k in need if k not in obj]
    if miss:
        raise ValueError(f"npz missing keys: {miss}")
    return obj


def compute_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1]).astype(np.float32)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    store: np.ndarray,
    dept: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    s_t = torch.from_numpy(store).long()
    d_t = torch.from_numpy(dept).long()
    ds = TensorDataset(X_t, y_t, s_t, d_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


class LSTMEmbedResidual(nn.Module):
    def __init__(
        self,
        input_dim: int,
        store_vocab: int,
        dept_vocab: int,
        store_emb_dim: int,
        dept_emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.store_emb = nn.Embedding(store_vocab, store_emb_dim)
        self.dept_emb = nn.Embedding(dept_vocab, dept_emb_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.calib_head = nn.Sequential(
            nn.Linear(store_emb_dim + dept_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, store_id: torch.Tensor, dept_id: torch.Tensor) -> torch.Tensor:
        # -- x: (B, T, F)
        # -- store_id/dept_id: (B,)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        base = self.base_head(h).squeeze(-1)

        se = self.store_emb(store_id)
        de = self.dept_emb(dept_id)
        sd = torch.cat([se, de], dim=-1)
        bias = self.calib_head(sd).squeeze(-1)

        return base + bias


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for xb, _, sb, db in loader:
        xb = xb.to(device)
        sb = sb.to(device)
        db = db.to(device)
        yb = model(xb, sb, db).detach().cpu().numpy()
        preds.append(yb)
    return np.concatenate(preds, axis=0)


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    n = 0
    loss_fn = nn.SmoothL1Loss(beta=0.2)
    for xb, yb, sb, db in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        sb = sb.to(device)
        db = db.to(device)

        optim.zero_grad(set_to_none=True)
        pred = model(xb, sb, db)
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def valid_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    loss_fn = nn.SmoothL1Loss(beta=0.2)
    for xb, yb, sb, db in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        sb = sb.to(device)
        db = db.to(device)

        pred = model(xb, sb, db)
        loss = loss_fn(pred, yb)

        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


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
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--store_emb_dim", type=int, default=16)
    parser.add_argument("--dept_emb_dim", type=int, default=16)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_npz = load_npz(data_dir / "seq_train.npz")
    valid_npz = load_npz(data_dir / "seq_valid.npz")

    X_train_raw = train_npz["X"].astype(np.float32)
    y_train_raw = train_npz["y"].astype(np.float32)
    s_train = train_npz["store"].astype(np.int64)
    d_train = train_npz["dept"].astype(np.int64)

    X_valid_raw = valid_npz["X"].astype(np.float32)
    y_valid_raw = valid_npz["y"].astype(np.float32)
    s_valid = valid_npz["store"].astype(np.int64)
    d_valid = valid_npz["dept"].astype(np.int64)

    mean, std = compute_standardizer(X_train_raw)
    X_train = apply_standardizer(X_train_raw, mean, std)
    X_valid = apply_standardizer(X_valid_raw, mean, std)

    y_train = np.log1p(y_train_raw).astype(np.float32)
    y_valid = np.log1p(y_valid_raw).astype(np.float32)

    store_vocab = int(s_train.max()) + 1
    dept_vocab = int(d_train.max()) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(X_train, y_train, s_train, d_train, batch_size=args.batch_size, shuffle=True)
    valid_loader = make_loader(X_valid, y_valid, s_valid, d_valid, batch_size=args.batch_size, shuffle=False)

    model = LSTMEmbedResidual(
        input_dim=X_train.shape[-1],
        store_vocab=store_vocab,
        dept_vocab=dept_vocab,
        store_emb_dim=args.store_emb_dim,
        dept_emb_dim=args.dept_emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_v = float("inf")
    bad = 0
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, device)
        va = valid_loss(model, valid_loader, device)
        print(f"[EPOCH] {ep:03d} train_loss={tr:.6f} valid_loss={va:.6f}")

        if va < best_v - 1e-6:
            best_v = va
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_pred_log = predict(model, valid_loader, device=device)
    y_pred = np.expm1(y_pred_log).astype(np.float32)

    metrics = evaluate(y_valid_raw, y_pred)

    with open(output_dir / "lstm_embed_res_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    out_df = pd.DataFrame(
        {
            "Store": s_valid.astype(int),
            "Dept": d_valid.astype(int),
            "Date": pd.to_datetime(valid_npz["date_ns"].astype(np.int64)),
            "y_true": y_valid_raw.astype(float),
            "y_pred": y_pred.astype(float),
        }
    )
    out_df.to_csv(output_dir / "lstm_embed_res_pred_valid.csv", index=False)

    stats = {
        "x_mean": mean.tolist(),
        "x_std": std.tolist(),
        "y_transform": "log1p",
        "store_vocab": store_vocab,
        "dept_vocab": dept_vocab,
        "store_emb_dim": args.store_emb_dim,
        "dept_emb_dim": args.dept_emb_dim,
        "arch": "lstm + residual calibration (store/dept embeddings)",
    }
    with open(output_dir / "lstm_embed_res_preprocess_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("[OK] LSTM(emb_res) training finished")
    for k, v in metrics.items():
        print(f"[METRIC] {k} = {v:.4f}")


if __name__ == "__main__":
    main()
