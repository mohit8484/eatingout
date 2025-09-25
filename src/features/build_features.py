# src/features/build_features.py
# Purpose: impute, encode, engineer features; 80/20 split; write data/processed/*.parquet

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", default="data/interim/preprocessed.parquet")
p.add_argument("--outdir", dest="outdir", default="data/processed")
p.add_argument("--top_cuisines", type=int, default=20)
p.add_argument("--top_categories", type=int, default=50, help="top N for suburb/type OHE")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

INP = Path(args.inp)
OUTDIR = Path(args.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INP)

# Column handles
cols = {c.lower(): c for c in df.columns}
rating_text = cols.get("rating_text", "rating_text")
rating_num  = cols.get("rating_number", "rating_number")
cost        = cols.get("cost", "cost")
cuisine     = cols.get("cuisine", "cuisine")
lat         = cols.get("lat", "lat")
lng         = cols.get("lng", "lng")
suburb      = cols.get("subzone", "subzone")
rtype       = cols.get("type", "type")
votes       = cols.get("votes", "votes")

# --------- Impute basic numerics ----------
for c in [cost, rating_num, votes, lat, lng]:
    if c in df.columns:
        med = df[c].median()
        df[c] = df[c].fillna(med)

# --------- Categorical clean ----------
for c in [rating_text, suburb, rtype]:
    if c in df.columns:
        mode = df[c].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else "Unknown"
        df[c] = df[c].fillna(fill)

# --------- Cuisine features ----------
def split_cuisines(s):
    if pd.isna(s) or not str(s).strip():
        return []
    return [x.strip().lower() for x in str(s).split(",") if x.strip()]

df["_cuisine_list"] = df[cuisine].apply(split_cuisines)

# Top-K cuisine tokens
from collections import Counter
ctr = Counter()
for lst in df["_cuisine_list"]:
    ctr.update(lst)
topk = [w for w, _ in ctr.most_common(args.top_cuisines)]

# One-hot for top cuisines + cuisine_diversity
for w in topk:
    df[f"cuisine__{w}"] = df["_cuisine_list"].apply(lambda lst, w=w: int(w in lst))
df["cuisine_diversity"] = df["_cuisine_list"].apply(len)

# --------- Limited OHE for suburb & type (top-N only) ----------
def top_ohe(series: pd.Series, topn: int, prefix: str):
    top = series.value_counts().head(topn).index.tolist()
    out = pd.DataFrame(index=series.index)
    for v in top:
        out[f"{prefix}__{v}"] = (series == v).astype(int)
    return out

ohe_blocks = []

if suburb in df.columns:
    ohe_blocks.append(top_ohe(df[suburb].astype(str), args.top_categories, "suburb"))
if rtype in df.columns:
    ohe_blocks.append(top_ohe(df[rtype].astype(str), args.top_categories, "type"))

ohe_df = pd.concat(ohe_blocks, axis=1) if ohe_blocks else pd.DataFrame(index=df.index)

# --------- Assemble feature table ----------
num_keep = [c for c in [cost, votes, lat, lng] if c in df.columns]
X = pd.concat([df[num_keep], ohe_df, df[[f"cuisine__{w}" for w in topk] + ["cuisine_diversity"]]], axis=1)
X = X.fillna(0)

# --------- Targets ----------
# Regression target
y_reg = df[rating_num].astype(float)

# Classification target: Class 1 = Poor/Average; Class 2 = Good/Very Good/Excellent
def to_binary(rt: str) -> int:
    rt = str(rt).strip().title()
    if rt in ("Poor", "Average"):
        return 0
    elif rt in ("Good", "Very Good", "Excellent"):
        return 1
    return 1  # default to positive class if unknown

y_clf = df[rating_text].apply(to_binary).astype(int)

# --------- Split ----------
X_train, X_test, yreg_train, yreg_test, yclf_train, yclf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=args.seed, stratify=y_clf
)

# Save
X_train.to_parquet(OUTDIR / "X_train.parquet")
X_test.to_parquet(OUTDIR / "X_test.parquet")
yreg_train.to_frame("rating_number").to_parquet(OUTDIR / "yreg_train.parquet")
yreg_test.to_frame("rating_number").to_parquet(OUTDIR / "yreg_test.parquet")
yclf_train.to_frame("rating_binary").to_parquet(OUTDIR / "yclf_train.parquet")
yclf_test.to_frame("rating_binary").to_parquet(OUTDIR / "yclf_test.parquet")

# Also save a features summary for your report
feat_summary = pd.DataFrame({"feature": X.columns.tolist()})
feat_summary.to_csv(OUTDIR / "features_list.csv", index=False)

print("✅ Features built and split:")
print(f"- X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"- yreg_train/test: {yreg_train.shape} / {yreg_test.shape}")
print(f"- yclf_train/test: {yclf_train.shape} / {yclf_test.shape}")
