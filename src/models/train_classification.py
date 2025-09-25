# src/models/train_classification.py
# Inputs:  data/processed/X_train.parquet, X_test.parquet, yclf_*.parquet
# Outputs: models/classification/*, figures/confmat_*.png, reports/metrics.json (+ CSV table)

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, classification_report
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- args ----------
p = argparse.ArgumentParser()
p.add_argument("--in_dir", default="data/processed")
p.add_argument("--model_dir", default="models/classification")
p.add_argument("--report_path", default="reports/metrics.json")
p.add_argument("--table_csv", default="reports/classification_metrics.csv")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

IN = Path(args.in_dir)
MDIR = Path(args.model_dir); MDIR.mkdir(parents=True, exist_ok=True)
RPT = Path(args.report_path); RPT.parent.mkdir(parents=True, exist_ok=True)
TABLE = Path(args.table_csv)

# ---------- load ----------
X_train = pd.read_parquet(IN / "X_train.parquet")
X_test  = pd.read_parquet(IN / "X_test.parquet")
ytr = pd.read_parquet(IN / "yclf_train.parquet")["rating_binary"].to_numpy().astype(int)
yte = pd.read_parquet(IN / "yclf_test.parquet")["rating_binary"].to_numpy().astype(int)

Xtr = X_train.values
Xte = X_test .values

# ---------- models ----------
models = {
    "logreg": LogisticRegression(max_iter=2000, n_jobs=None, random_state=args.seed),
    "rf": RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=-1),
    "gbt": GradientBoostingClassifier(random_state=args.seed),
    "svm_linear": LinearSVC(random_state=args.seed)
}

rows = []
fig_dir = Path("figures"); fig_dir.mkdir(exist_ok=True)

def save_confmat(cm, name):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix — {name}")
    plt.colorbar()
    tick = np.array([0,1])
    plt.xticks(tick, ["Class 0 (Poor/Average)", "Class 1 (Good→Excellent)"], rotation=30, ha="right")
    plt.yticks(tick, ["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out = fig_dir / f"confmat_{name}.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out

for name, model in models.items():
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)

    cm = confusion_matrix(yte, yp, labels=[0,1])
    prec, rec, f1, _ = precision_recall_fscore_support(yte, yp, labels=[0,1], average="macro", zero_division=0)

    # store
    joblib.dump(model, MDIR / f"{name}.joblib")
    out_png = save_confmat(cm, name)

    rows.append({
        "model": name,
        "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4),
        "f1_macro": round(float(f1), 4),
        "confmat_png": str(out_png)
    })

# table csv
df_rows = pd.DataFrame(rows)
df_rows.to_csv(TABLE, index=False)

# write/append json metrics
metrics = {}
if RPT.exists():
    try:
        metrics = json.loads(RPT.read_text())
    except Exception:
        metrics = {}
metrics.setdefault("classification", {})
for r in rows:
    metrics["classification"][r["model"]] = {
        "precision_macro": r["precision_macro"],
        "recall_macro": r["recall_macro"],
        "f1_macro": r["f1_macro"]
    }
RPT.write_text(json.dumps(metrics, indent=2))

print("✅ Classification done")
print(df_rows.to_string(index=False))
print(f"- Models in:   {MDIR.resolve()}")
print(f"- Conf mats:   {fig_dir.resolve()}")
print(f"- Metrics CSV: {TABLE.resolve()}")
print(f"- Metrics JSON:{RPT.resolve()}")
