# Geospatial cuisine density choropleth
# Usage:
#   python src/eda/geo_choropleth.py --cuisine "indian"
#   python src/eda/geo_choropleth.py --cuisine "thai"
#
# Outputs:
#   figures/choropleth_<cuisine>.png
#   reports/cuisine_<cuisine>_top_suburbs.csv

import argparse
import json
from pathlib import Path

import pandas as pd

# Force non-GUI backend for Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try geopandas (Common Windows fix: use Python 3.10/3.11 if 3.13 wheels missing)
try:
    import geopandas as gpd
except Exception as e:
    raise SystemExit(
        f"[ERROR] geopandas failed to import: {e}\n"
        "Fix: in Windows, use Python 3.11 virtual env and reinstall:\n"
        "  py -3.11 -m venv .venv311 && .\\.venv311\\Scripts\\Activate.ps1\n"
        "  pip install -r requirements.txt\n"
    )

p = argparse.ArgumentParser(description="Cuisine density choropleth by suburb")
p.add_argument("--csv", default="data/raw/zomato_df_final_data.csv", help="Path to restaurants CSV")
p.add_argument("--geojson", default="data/raw/sydney.geojson", help="Path to Sydney GeoJSON")
p.add_argument("--cuisine", required=True, help="Cuisine name to map (e.g., 'indian')")
p.add_argument("--outdir", default="figures", help="Where to save the PNG")
p.add_argument("--reportdir", default="reports", help="Where to save the top-suburbs CSV")
args = p.parse_args()

CSV_PATH = Path(args.csv)
GEO_PATH = Path(args.geojson)
FIG_DIR = Path(args.outdir); FIG_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR = Path(args.reportdir); REP_DIR.mkdir(parents=True, exist_ok=True)

target_cuisine = args.cuisine.strip().lower()

# 1) Load CSV
df = pd.read_csv(CSV_PATH)

# Identify key columns
cols = {c.lower(): c for c in df.columns}
suburb_col = cols.get("subzone", "subzone")
cuisine_col = cols.get("cuisine", "cuisine")

# 2) Build a boolean flag "serves_target" per row
def serves(row):
    raw = str(row.get(cuisine_col, "")).lower()
    if not raw:
        return False
    # split and allow substring match so "north indian" counts for "indian"
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return any(target_cuisine in p for p in parts)


df["serves_target"] = df.apply(serves, axis=1)

# 3) Count restaurants serving target cuisine by suburb
# Normalise suburb names for joining: strip + uppercase
df["suburb_norm"] = df[suburb_col].astype(str).str.strip().str.upper()
counts = (
    df.loc[df["serves_target"], "suburb_norm"]
      .value_counts()
      .rename_axis("suburb_norm")
      .reset_index(name="count")
)

# Save top 20 suburbs to CSV (for report)
counts.head(20).to_csv(REP_DIR / f"cuisine_{target_cuisine}_top_suburbs.csv", index=False)

# 4) Load GeoJSON (Sydney suburbs), guess the suburb name key (SSC_NAME)
gdf = gpd.read_file(GEO_PATH)
# Many ABS GeoJSONs use SSC_NAME for suburb names
suburb_key = None
for k in gdf.columns:
    if "SSC_NAME" == k or k.lower() in ("ssc_name", "name", "suburb", "suburb_name"):
        suburb_key = k
        break
if suburb_key is None:
    # Fallback to first string-like column
    for k in gdf.columns:
        if gdf[k].dtype == "object":
            suburb_key = k; break

# Normalise geo suburb names same as CSV: strip + uppercase
gdf["suburb_norm"] = gdf[suburb_key].astype(str).str.strip().str.upper()

# 5) Merge counts onto shapes
merged = gdf.merge(counts, on="suburb_norm", how="left").fillna({"count": 0})

# 6) Plot choropleth
vmax = max(1, int(merged["count"].max()))
ax = merged.plot(
    column="count",
    cmap="viridis",
    legend=True,
    edgecolor="black",
    linewidth=0.2,
    figsize=(10, 10),
    vmin=0, vmax=vmax,
    missing_kwds={"color": "lightgrey", "edgecolor": "white", "hatch": "///", "label": "No data"},
)
ax.set_title(f"Density of '{target_cuisine.title()}' Restaurants by Suburb", fontsize=14)
ax.set_axis_off()

# Tidy legend title
leg = ax.get_figure().axes[-1]  # colorbar axis
leg.set_ylabel("Count")


out_path = FIG_DIR / f"choropleth_{target_cuisine}.png"
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.close()

print("✅ Choropleth done")
print(f"- Map:     {out_path.resolve()}")
print(f"- Top 20:  {(REP_DIR / f'cuisine_{target_cuisine}_top_suburbs.csv').resolve()}")
