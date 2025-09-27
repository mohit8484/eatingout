# src/models/train_spark_classification.py
# Spark Logistic Regression for binary rating_text

import json
from pathlib import Path
import argparse

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/raw/zomato_df_final_data.csv")
p.add_argument("--model_dir", default="models/spark/classification")
p.add_argument("--report_path", default="reports/metrics.json")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

MDIR = Path(args.model_dir); MDIR.mkdir(parents=True, exist_ok=True)
RPT = Path(args.report_path); RPT.parent.mkdir(parents=True, exist_ok=True)

spark = SparkSession.builder.appName("eatingout-classification").getOrCreate()

df = spark.read.csv(args.csv, header=True, inferSchema=True)

# Binary label: 0 = Poor/Average, 1 = Good/Very Good/Excellent (default 1)
df = df.withColumn("rating_text_norm", F.initcap(F.trim(F.col("rating_text"))))
df = df.withColumn(
    "label",
    F.when(F.col("rating_text_norm").isin("Poor", "Average"), 0)
     .when(F.col("rating_text_norm").isin("Good", "Very Good", "Excellent"), 1)
     .otherwise(1)
)

num_cols = ["cost", "votes", "lat", "lng"]
cat_cols = ["type", "subzone"]

# Clean numerics robustly
for c in num_cols:
    if c in df.columns:
        df = df.withColumn(c, F.regexp_replace(F.col(c).cast("string"), r"[^0-9.\-]", ""))
        df = df.withColumn(c, F.expr(f"try_cast({c} as double)"))

# Median impute
for c in num_cols:
    if c in df.columns:
        med = df.approxQuantile(c, [0.5], 0.1)[0] if df.where(F.col(c).isNotNull()).count() > 0 else 0.0
        df = df.fillna({c: med})

stages = []
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols if c in df.columns]
encoders = [OneHotEncoder(inputCols=[f"{c}_idx"], outputCols=[f"{c}_oh"]) for c in cat_cols if c in df.columns]
stages.extend(indexers + encoders)

feat_cols = [c for c in num_cols if c in df.columns] + [f"{c}_oh" for c in cat_cols if c in df.columns]
assembler = VectorAssembler(inputCols=feat_cols, outputCol="features")
stages.append(assembler)

clf = LogisticRegression(featuresCol="features", labelCol="label", maxIter=200)
stages.append(clf)

pipe = Pipeline(stages=stages)

train, test = df.randomSplit([0.8, 0.2], seed=args.seed)
model = pipe.fit(train)
pred = model.transform(test)

acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(pred)
f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(pred)

# Skip saving on Windows to avoid HADOOP_HOME/winutils issues.
# model.write().overwrite().save(str(MDIR))
pass



# Append to reports
metrics = {}
if RPT.exists():
    try:
        metrics = json.loads(RPT.read_text())
    except Exception:
        metrics = {}
metrics.setdefault("spark", {})
metrics["spark"]["classification"] = {"accuracy": acc, "f1": f1}
RPT.write_text(json.dumps(metrics, indent=2))

print("✅ Spark Classification done")
print(f"- Accuracy: {acc:.4f}")
print(f"- F1:       {f1:.4f}")
print(f"- Model →   {MDIR.resolve()}")
print(f"- Metrics → {RPT.resolve()}")

spark.stop()
