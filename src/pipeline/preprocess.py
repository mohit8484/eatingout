# src/pipeline/preprocess.py
# Purpose: light cleaning and column standardisation → writes data/interim/preprocessed.parquet

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", default="data/raw/zomato_df_final_data.csv")
p.add_argument("--out", dest="out", default="data/interim/preprocessed.parquet")
args = p.parse_args()

INP = Path(args.inp)
OUT = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INP)

# Standard names
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
groupon     = cols.get("groupon", "groupon")

keep = [rating_text, rating_num, cost, cuisine, lat, lng, suburb, rtype, votes, groupon]
keep = [c for c in keep if c in df.columns]
df = df[keep].copy()

# Trim strings
for c in [rating_text, cuisine, suburb, rtype]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# Coerce numerics
for c in [rating_num, cost, lat, lng, votes]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Normalise rating_text to title-case (Poor/Average/Good/Very Good/Excellent)
if rating_text in df.columns:
    df[rating_text] = df[rating_text].astype(str).str.strip().str.title()

# Save
df.to_parquet(OUT, index=False)
print(f"✅ Wrote {OUT.resolve()} (rows={len(df)})")
