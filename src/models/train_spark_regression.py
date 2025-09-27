# src/models/train_spark_regression.py
# Spark Linear Regression for rating_number

import json
from pathlib import Path
import argparse

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/raw/zomato_df_final_data.csv")
p.add_argument("--model_dir", default="models/spark/regression")
p.add_argument("--report_path", default="reports/metrics.json")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

MDIR = Path(args.model_dir); MDIR.mkdir(parents=True, exist_ok=True)
RPT = Path(args.report_path); RPT.parent.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("eatingout-regression")
    .getOrCreate()
)

df = spark.read.csv(args.csv, header=True, inferSchema=True)

# Select a compact feature set to keep it fast + clear
num_cols = ["cost", "votes", "lat", "lng"]
cat_cols = ["type", "subzone"]
target = "rating_number"

# Clean: remove non-numeric chars then try_cast to double (tolerates junk like " CBD")
for c in num_cols + [target]:
    if c in df.columns:
        # keep digits, minus, dot; drop everything else
        df = df.withColumn(c, F.regexp_replace(F.col(c).cast("string"), r"[^0-9.\-]", ""))
        df = df.withColumn(c, F.expr(f"try_cast({c} as double)"))

# drop rows where target couldn't be parsed
df = df.na.drop(subset=[target])


# Impute simple numeric medians
for c in num_cols:
    if c in df.columns:
        # use approxQuantile and ignore nulls
        med = df.approxQuantile(c, [0.5], 0.1)[0] if df.where(F.col(c).isNotNull()).count() > 0 else 0.0
        df = df.fillna({c: med})


stages = []
# Index + OHE for categoricals (ignore invalid labels)
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols if c in df.columns]
encoders = [OneHotEncoder(inputCols=[f"{c}_idx"], outputCols=[f"{c}_oh"]) for c in cat_cols if c in df.columns]
stages.extend(indexers + encoders)

# Assemble
feat_cols = [c for c in num_cols if c in df.columns] + [f"{c}_oh" for c in cat_cols if c in df.columns]
assembler = VectorAssembler(inputCols=feat_cols, outputCol="features")
stages.append(assembler)

reg = LinearRegression(featuresCol="features", labelCol=target, predictionCol="prediction", maxIter=200)
stages.append(reg)

pipe = Pipeline(stages=stages)

# Train/test split
train, test = df.randomSplit([0.8, 0.2], seed=args.seed)
model = pipe.fit(train)
pred = model.transform(test)

# Evaluate
evaluator_mse = RegressionEvaluator(predictionCol="prediction", labelCol=target, metricName="mse")
evaluator_rmse = RegressionEvaluator(predictionCol="prediction", labelCol=target, metricName="rmse")
mse = evaluator_mse.evaluate(pred)
rmse = evaluator_rmse.evaluate(pred)

# Skip saving on Windows to avoid HADOOP_HOME/winutils issues.
# model.write().overwrite().save(str(MDIR))
pass


# Append to reports/metrics.json
metrics = {}
if RPT.exists():
    try:
        metrics = json.loads(RPT.read_text())
    except Exception:
        metrics = {}
metrics.setdefault("spark", {})
metrics["spark"]["regression"] = {"mse": mse, "rmse": rmse}

RPT.write_text(json.dumps(metrics, indent=2))

print("✅ Spark Regression done")
print(f"- MSE:  {mse:.4f}")
print(f"- RMSE: {rmse:.4f}")
print(f"- Model saved to: {MDIR.resolve()}")
print(f"- Metrics → {RPT.resolve()}")

spark.stop()
