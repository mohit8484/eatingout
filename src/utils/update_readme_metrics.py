# src/utils/update_readme_metrics.py
from pathlib import Path
import json
import csv
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

# -------- Paths --------
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
PROC = ROOT / "data" / "processed"

metrics_json = REPORTS / "metrics.json"
clf_csv = REPORTS / "classification_metrics.csv"
X_test_pq = PROC / "X_test.parquet"
yreg_test_pq = PROC / "yreg_test.parquet"
feats_list_csv = PROC / "features_list.csv"

# -------- Utils --------
def read_json(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def read_clf_table(path: Path):
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return rows

def fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def try_spark_metrics(metrics: dict):
    """Return (mse, rmse, acc, f1) if present in reports/metrics.json."""
    mse = rmse = acc = f1 = None
    if not isinstance(metrics, dict):
        return mse, rmse, acc, f1

    spark = metrics.get("spark", {})
    if isinstance(spark, dict):
        reg = spark.get("regression", {})
        if isinstance(reg, dict):
            mse = reg.get("mse", None)
            rmse = reg.get("rmse", None)
        clf = spark.get("classification", {})
        if isinstance(clf, dict):
            acc = clf.get("accuracy", None)
            f1 = clf.get("f1", None)
    return mse, rmse, acc, f1

def predict_gd_dict(gd_obj, X_df: pd.DataFrame, feature_order=None):
    """
    Compute predictions for a gradient-descent model saved as a dict/tuple.
    Handles:
      - dict with weights and optional bias
      - tuple/list like (weights, bias)
      - weights including intercept (len = n_features + 1)
      - column reordering to training order
    Returns np.ndarray or None.
    """
    w = b = None

    # Unpack tuple/list case: (weights, bias) or [weights, bias]
    if isinstance(gd_obj, (tuple, list)) and len(gd_obj) >= 1:
        w = np.array(gd_obj[0]).reshape(-1)
        if len(gd_obj) >= 2:
            try:
                b = float(np.array(gd_obj[1]).reshape(()))
            except Exception:
                b = 0.0

    # Dict case with various key names
    if isinstance(gd_obj, dict):
        keys_w = ["w", "weights", "coef_", "coefficients", "theta", "params"]
        keys_b = ["b", "bias", "intercept_", "intercept"]
        for k in keys_w:
            if k in gd_obj:
                w = np.array(gd_obj[k]).reshape(-1)
                break
        for k in keys_b:
            if k in gd_obj:
                try:
                    b = float(np.array(gd_obj[k]).reshape(()))
                except Exception:
                    b = 0.0

    # Estimator-like? (just in case)
    if hasattr(gd_obj, "predict"):
        try:
            return gd_obj.predict(X_df.values)
        except Exception:
            pass

    if w is None:
        return None

    # Reorder columns to training order if provided
    X = X_df.copy()
    if feature_order:
        cols = [c for c in feature_order if c in X.columns]
        if cols:
            X = X[cols]

    Xn = X.values  # numpy matrix

    # Case 1: weights include intercept at the END
    if w.shape[0] == Xn.shape[1] + 1:
        b_from_w = float(w[-1])
        w_core = w[:-1]
        return Xn @ w_core + b_from_w

    # Case 2: weights match features; use explicit b (or 0 if missing)
    if w.shape[0] == Xn.shape[1]:
        if b is None:
            b = 0.0
        return Xn @ w + b

    # Case 3: auto-trim or pad conservatively
    if w.shape[0] > Xn.shape[1]:
        # Trim extra tail weights; if exactly +1, treat extra as bias
        w_trim = w[:Xn.shape[1]]
        if b is None and w.shape[0] == Xn.shape[1] + 1:
            b = float(w[-1])
        if b is None:
            b = 0.0
        return Xn @ w_trim + b

    if w.shape[0] < Xn.shape[1]:
        # Pad weights with zeros
        pad = Xn.shape[1] - w.shape[0]
        w_pad = np.pad(w, (0, pad))
        if b is None:
            b = 0.0
        return Xn @ w_pad + b

    return None

# -------- Main --------
if __name__ == "__main__":
    # 1) Load processed test data
    lr_mse = gd_mse = None
    X_test = y_test = None
    feature_order = None

    if X_test_pq.exists() and yreg_test_pq.exists():
        X_test = pd.read_parquet(X_test_pq)
        y_test = pd.read_parquet(yreg_test_pq).iloc[:, 0]

        # Apply original training column order if available
        if feats_list_csv.exists():
            try:
                feature_order = pd.read_csv(feats_list_csv, header=None)[0].tolist()
                cols_present = [c for c in feature_order if c in X_test.columns]
                if cols_present:
                    X_test = X_test[cols_present]
            except Exception as e:
                print(f"[warn] could not apply feature order: {e}")

    # 2) Compute sklearn regression metrics if we have data
    if X_test is not None and y_test is not None:
        # Linear Regression
        lin_path = MODELS / "regression" / "linear.joblib"
        if lin_path.exists():
            try:
                lin = joblib.load(lin_path)
                yhat_lin = lin.predict(X_test.values)  # pass numpy to avoid feature-names warning
                lr_mse = mean_squared_error(y_test, yhat_lin)
            except Exception as e:
                print(f"[warn] linear.joblib predict failed: {e}")

        # Gradient Descent model (dict/tuple/estimator)
        gd_path = MODELS / "regression" / "gd_model.joblib"
        if gd_path.exists():
            try:
                gd = joblib.load(gd_path)
                yhat_gd = None

                # estimator-like path
                if hasattr(gd, "predict"):
                    try:
                        yhat_gd = gd.predict(X_test.values)
                    except Exception:
                        yhat_gd = None

                # dict/tuple path
                if yhat_gd is None:
                    yhat_gd = predict_gd_dict(gd, X_test, feature_order=feature_order)

                if yhat_gd is not None:
                    gd_mse = mean_squared_error(y_test, np.asarray(yhat_gd).reshape(-1))
            except Exception as e:
                print(f"[warn] gd_model load/predict failed: {e}")

    # 3) Read sklearn classification table
    clf_rows = read_clf_table(clf_csv)

    # 4) Read PySpark metrics (mse, rmse, accuracy, f1)
    mj = read_json(metrics_json)
    spark_mse, spark_rmse, spark_acc, spark_f1 = try_spark_metrics(mj)

    # 5) Print block for README.md
    print("## 📊 Results\n")

    print("### Regression (Scikit-Learn)")
    print(f"- Linear Regression MSE: **{fmt(lr_mse, 2) if lr_mse is not None else 'N/A'}**")
    print(f"- Gradient Descent MSE: **{fmt(gd_mse, 2) if gd_mse is not None else 'N/A'}**\n")

    print("### Classification (Scikit-Learn)")
    if clf_rows:
        print("| Model | Precision | Recall | F1 | Confusion Matrix |")
        print("|---|---:|---:|---:|---|")
        for r in clf_rows:
            prec = r.get("precision_macro", r.get("precision", "N/A"))
            rec  = r.get("recall_macro",    r.get("recall", "N/A"))
            f1   = r.get("f1_macro",        r.get("f1", "N/A"))
            cm   = r.get("confmat_png", "")
            print(f"| {r.get('model','')} | {fmt(prec)} | {fmt(rec)} | {fmt(f1)} | `{cm}` |")
    else:
        print("_classification_metrics.csv not found_")
    print()

    print("### PySpark")
    print(f"- Regression MSE: **{fmt(spark_mse, 6) if spark_mse is not None else 'N/A'}**")
    if spark_rmse is not None:
        print(f"- Regression RMSE: **{fmt(spark_rmse, 6)}**")
    print(f"- Classification — Accuracy: **{fmt(spark_acc, 6) if spark_acc is not None else 'N/A'}**, F1: **{fmt(spark_f1, 6) if spark_f1 is not None else 'N/A'}**")
