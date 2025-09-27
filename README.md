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
- Linear Regression MSE: **0.09**
- Gradient Descent MSE: **31942.16**

### Classification (Scikit-Learn)
| Model | Precision | Recall | F1 | Confusion Matrix |
|---|---:|---:|---:|---|
| logreg | 0.7513 | 0.7533 | 0.7518 | `figures\confmat_logreg.png` |
| rf | 0.8998 | 0.8889 | 0.8924 | `figures\confmat_rf.png` |
| gbt | 0.9145 | 0.9121 | 0.9131 | `figures\confmat_gbt.png` |
| svm_linear | 0.7319 | 0.7338 | 0.7323 | `figures\confmat_svm_linear.png` |

### PySpark
- Regression MSE: **84669899.979090**
- Regression RMSE: **9201.624855**
- Classification — Accuracy: **0.678325**, F1: **0.678526**