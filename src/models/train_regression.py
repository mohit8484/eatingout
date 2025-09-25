# src/models/train_regression.py
# Train two regression models to predict rating_number:
# (A) scikit-learn LinearRegression
# (B) custom batch Gradient Descent (from scratch)
#
# Inputs:  data/processed/X_train.parquet, X_test.parquet, yreg_*.parquet
# Outputs: models/regression/*.joblib and reports/metrics.json (appended/created)

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# ---------- Args ----------
p = argparse.ArgumentParser()
p.add_argument("--in_dir", default="data/processed")
p.add_argument("--model_dir", default="models/regression")
p.add_argument("--report_path", default="reports/metrics.json")
p.add_argument("--lr", type=float, default=0.05, help="GD learning rate")
p.add_argument("--epochs", type=int, default=2000, help="GD max epochs")
p.add_argument("--tol", type=float, default=1e-7, help="GD early-stop tolerance on loss")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

IN = Path(args.in_dir)
MDIR = Path(args.model_dir); MDIR.mkdir(parents=True, exist_ok=True)
RPT = Path(args.report_path); RPT.parent.mkdir(parents=True, exist_ok=True)

# ---------- Load data ----------
X_train = pd.read_parquet(IN / "X_train.parquet")
X_test  = pd.read_parquet(IN / "X_test.parquet")
ytr = pd.read_parquet(IN / "yreg_train.parquet")["rating_number"].to_numpy().astype(float)
yte = pd.read_parquet(IN / "yreg_test.parquet")["rating_number"].to_numpy().astype(float)

# Ensure numeric arrays
Xtr = X_train.to_numpy().astype(float)
Xte = X_test.to_numpy().astype(float)

# ---------- (A) Linear Regression ----------
lin = LinearRegression(n_jobs=None)
lin.fit(Xtr, ytr)
pred_lin = lin.predict(Xte)
mse_lin = float(mean_squared_error(yte, pred_lin))
joblib.dump(lin, MDIR / "linear.joblib")

# ---------- (B) Custom Gradient Descent ----------
# Standardise features (fit on train, apply to test)
mu = Xtr.mean(axis=0)
sigma = Xtr.std(axis=0, ddof=0)
sigma[sigma == 0] = 1.0
Xtr_s = (Xtr - mu) / sigma
Xte_s = (Xte - mu) / sigma

# Add intercept column
def add_intercept(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

Xtr_b = add_intercept(Xtr_s)
Xte_b = add_intercept(Xte_s)

rng = np.random.default_rng(args.seed)
w = rng.normal(scale=0.01, size=Xtr_b.shape[1])

def mse_loss(X, y, w):
    yhat = X @ w
    return np.mean((yhat - y) ** 2)

prev = np.inf
for epoch in range(args.epochs):
    yhat = Xtr_b @ w
    grad = (2.0 / Xtr_b.shape[0]) * (Xtr_b.T @ (yhat - ytr))
    w -= args.lr * grad
    loss = mse_loss(Xtr_b, ytr, w)
    if abs(prev - loss) < args.tol:
        break
    prev = loss

pred_gd = Xte_b @ w
mse_gd = float(mean_squared_error(yte, pred_gd))

# Save the GD "model" (weights + scaler)
gd_artifact = {
    "weights": w.tolist(),
    "mu": mu.tolist(),
    "sigma": sigma.tolist(),
    "feature_order": X_train.columns.tolist()
}
joblib.dump(gd_artifact, MDIR / "gd_model.joblib")

# ---------- Write/append metrics ----------
metrics = {}
if RPT.exists():
    try:
        metrics = json.loads(RPT.read_text())
    except Exception:
        metrics = {}

metrics.setdefault("regression", {})
metrics["regression"]["linear_regression_mse"] = mse_lin
metrics["regression"]["gradient_descent_mse"] = mse_gd

RPT.write_text(json.dumps(metrics, indent=2))

print("✅ Regression done")
print(f"- LinearRegression MSE: {mse_lin:.4f}")
print(f"- GD (custom) MSE:     {mse_gd:.4f}")
print(f"- Saved models in:      {MDIR.resolve()}")
print(f"- Metrics in:           {RPT.resolve()}")
