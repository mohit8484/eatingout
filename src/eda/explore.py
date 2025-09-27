import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # use non-GUI backend (no Tk needed)
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

# --- CLI args (optional, defaults are fine) ---
p = argparse.ArgumentParser(description="Basic EDA for Sydney restaurants dataset")
p.add_argument("--csv", default="data/raw/zomato_df_final_data.csv", help="Path to CSV")
p.add_argument("--outdir", default="figures", help="Directory to save figures")
p.add_argument("--reportdir", default="reports", help="Directory to save text/metrics")
args = p.parse_args()

CSV_PATH = Path(args.csv)
FIG_DIR = Path(args.outdir)
REP_DIR = Path(args.reportdir)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

# --- 1) Load dataset ---
df = pd.read_csv(CSV_PATH)

# Standardise column handles
cols = {c.lower(): c for c in df.columns}
rating_text_col = cols.get("rating_text", "rating_text")
rating_num_col  = cols.get("rating_number", "rating_number")
cost_col        = cols.get("cost", "cost")
cuisine_col     = cols.get("cuisine", "cuisine")
votes_col       = cols.get("votes", "votes")
suburb_col      = cols.get("subzone", "subzone")
type_col        = cols.get("type", "type")

# --- 2) Basic cleaning (trim strings, coerce numerics) ---
for c in [suburb_col, cuisine_col, rating_text_col, type_col]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

for c in [cost_col, rating_num_col, votes_col, "lat", "lng"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# --- 3) Missingness summary ---
missing_pct = (df.isna().mean().sort_values(ascending=False) * 100).round(2)
miss_tbl = pd.DataFrame({"column": missing_pct.index, "missing_%": missing_pct.values})
miss_tbl.to_csv(REP_DIR / "missingness_summary.csv", index=False)

# --- 4) Helpful derived items ---
# 4a) Unique cuisines (approx): split by comma and normalise
unique_cuisines = set()
if cuisine_col in df.columns:
    cuisines_series = df[cuisine_col].dropna().astype(str).str.split(",")
    for lst in cuisines_series:
        for x in lst:
            name = x.strip().lower()
            if name:
                unique_cuisines.add(name)

approx_unique_cuisines_count = len(unique_cuisines)

# 4b) Top 3 suburbs by restaurant count
top_suburbs = []
if suburb_col in df.columns:
    top_suburbs = (
        df[suburb_col]
        .dropna()
        .astype(str).str.strip()
        .value_counts()
        .head(3)
        .to_dict()
    )

# --- 5) Plots (Matplotlib, default styles, single plots) ---
# 5a) Distribution: cost
if cost_col in df.columns:
    plt.figure()
    df[cost_col].dropna().plot(kind="hist", bins=40)
    plt.title("Distribution of Cost (for two)")
    plt.xlabel("Cost (AUD)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dist_cost.png")
    plt.close()

# 5b) Distribution: rating_number
if rating_num_col in df.columns:
    plt.figure()
    df[rating_num_col].dropna().plot(kind="hist", bins=30)
    plt.title("Distribution of Rating Number")
    plt.xlabel("Rating Number")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dist_rating_number.png")
    plt.close()

# 5c) Distribution: type (bar of top 15)
if type_col in df.columns:
    plt.figure()
    df[type_col].value_counts().head(15).plot(kind="bar")
    plt.title("Top Restaurant Types (Top 15)")
    plt.xlabel("Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bar_type_top15.png")
    plt.close()

# 5d) Top 3 suburbs bar (if we have suburb_col)
if top_suburbs:
    plt.figure()
    s = pd.Series(top_suburbs)
    s.sort_values(ascending=True).plot(kind="barh")
    plt.title("Top 3 Suburbs by Restaurant Count")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top3_suburbs.png")
    plt.close()

# 5e) Cost vs Votes correlation + scatter
corr_cost_votes = None
if cost_col in df.columns and votes_col in df.columns:
    subset = df[[cost_col, votes_col]].dropna()
    if len(subset) > 2:
        corr_cost_votes = float(subset[cost_col].corr(subset[votes_col]))
        plt.figure()
        subset.plot(kind="scatter", x=cost_col, y=votes_col)
        plt.title(f"Cost vs Votes (corr={corr_cost_votes:.3f})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "scatter_cost_votes.png")
        plt.close()

# 5f) Are “Excellent” more expensive than “Poor”? (boxplot + stats)
price_comp_stats = {}
if rating_text_col in df.columns and cost_col in df.columns:
    mask = df[rating_text_col].notna() & df[cost_col].notna()
    small = df.loc[mask, [rating_text_col, cost_col]].copy()
    # Normalise categories
    small[rating_text_col] = small[rating_text_col].str.strip().str.title()
    groups = {
        "Poor": small.loc[small[rating_text_col] == "Poor", cost_col].dropna(),
        "Excellent": small.loc[small[rating_text_col] == "Excellent", cost_col].dropna()
    }
    # Save basic stats
    for k, s in groups.items():
        if len(s) > 0:
            price_comp_stats[k] = {
                "n": int(len(s)),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else float("nan")
            }
    # Boxplot (if both groups non-empty)
    if all(len(v) > 0 for v in groups.values()):
        plt.figure()
        plt.boxplot([groups["Poor"], groups["Excellent"]], labels=["Poor", "Excellent"])
        plt.title("Cost by Rating Category: Poor vs Excellent")
        plt.ylabel("Cost (AUD)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "box_cost_poor_vs_excellent.png")
        plt.close()

# --- 6) Write a small text summary for your PDF later ---
summary = {
    "rows": int(df.shape[0]),
    "cols": int(df.shape[1]),
    "columns": df.columns.tolist(),
    "approx_unique_cuisines_count": approx_unique_cuisines_count,
    "top_3_suburbs": top_suburbs,
    "corr_cost_votes": corr_cost_votes,
    "price_comparison_stats": price_comp_stats,
}
with open(REP_DIR / "eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("=== EDA SUMMARY ===\n")
    f.write(json.dumps(summary, indent=2))
    f.write("\n\nMissingness (top 15):\n")
    f.write(miss_tbl.head(15).to_string(index=False))
    f.write("\n")

print("✅ EDA complete.")
print(f"- Saved figures to: {FIG_DIR.resolve()}")
print(f"- Saved reports to: {REP_DIR.resolve()}")
