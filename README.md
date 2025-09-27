# Modelling of Eating-Out Problem — Sydney Restaurants (2018)

**Student:** Mohit Singh Mandota  
**Student ID:** u3278214  
**GitHub:** https://github.com/mohit8484/eatingout

Reproducible workflow: EDA → feature engineering → regression & classification (Scikit-Learn + PySpark) → DVC pipeline.

---

## 💻 How to run (Windows PowerShell)

```powershell
# 1) create & activate venv (first time)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Spark only — set these each NEW PowerShell session
$env:PYSPARK_PYTHON        = "$PWD\.venv\Scripts\python.exe"
$env:PYSPARK_DRIVER_PYTHON = "$PWD\.venv\Scripts\python.exe"

# 3) reproduce all stages with DVC
dvc repro
---

## 📊 Results

### Regression (Scikit-Learn)
- Linear Regression MSE: **[fill from metrics.json]**
- Gradient Descent MSE: **[fill from metrics.json]**

### Classification (Scikit-Learn)
- Best model: **GBT (Gradient Boosted Trees)**
- Best F1 (macro): **0.9131**
- Full metrics in `reports/classification_metrics.csv`

### PySpark
- Logistic Regression (binary rating): **Accuracy ≈ 0.678**, **F1 ≈ 0.679**
- Linear Regression: **MSE ≈ 84,669,900**, **RMSE ≈ 9,201.6**
