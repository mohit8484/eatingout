import argparse
from pathlib import Path
import pandas as pd
import plotly.express as px

p = argparse.ArgumentParser(description="Interactive cost vs rating scatter (Plotly)")
p.add_argument("--csv", default="data/raw/zomato_df_final_data.csv")
p.add_argument("--out", default="figures/interactive_cost_rating.html")
args = p.parse_args()

CSV_PATH = Path(args.csv)
OUT = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_csv(CSV_PATH)

# Standardise key columns
cols = {c.lower(): c for c in df.columns}
rating_text = cols.get("rating_text", "rating_text")
rating_num  = cols.get("rating_number", "rating_number")
cost        = cols.get("cost", "cost")
suburb      = cols.get("subzone", "subzone")
cuisine     = cols.get("cuisine", "cuisine")
rtype       = cols.get("type", "type")
votes       = cols.get("votes", "votes")

# Keep useful cols and drop nulls for this plot
keep = [cost, rating_num, rating_text, suburb, cuisine, rtype, votes]
use = df[keep].dropna(subset=[cost, rating_num]).copy()

# Basic cleaning
use[suburb] = use[suburb].astype(str).str.strip().str.title()
use[rating_text] = use[rating_text].astype(str).str.strip().str.title()
use[rtype] = use[rtype].astype(str).str.strip().str.title()

# Make an informative hover string
use["hover"] = (
    "Suburb: " + use[suburb].astype(str) +
    "<br>Cuisine: " + use[cuisine].astype(str) +
    "<br>Type: " + use[rtype].astype(str) +
    "<br>Votes: " + use[votes].astype(str)
)

fig = px.scatter(
    use,
    x=cost, y=rating_num,
    color=rating_text,
    hover_name=suburb,
    hover_data={"hover": True, cost: True, rating_num: True, rating_text: True, suburb: False, cuisine: False, rtype: False, votes: False},
    opacity=0.6,
    template="plotly_white",
)

fig.update_traces(
    hovertemplate="%{hover}<extra></extra>"
)

fig.update_layout(
    title="Interactive: Cost vs Rating Number (hover for details; click legend to filter)",
    xaxis_title="Cost for Two (AUD)",
    yaxis_title="Rating Number",
    legend_title="Rating Text",
)

fig.write_html(str(OUT), include_plotlyjs="cdn")
print(f"✅ Saved interactive plot: {OUT.resolve()}")
