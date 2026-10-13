# Implementation of ML Minor Project

Fraud detection in financial transactions using machine learning and imbalance mitigation, based on the attached Expert Systems paper by Al-dahasi et al.

## What This Project Contains

- A complete Python training pipeline.
- Six ML models from the paper: Logistic Regression, Decision Tree, Gaussian Naive Bayes, Random Forest, XGBoost, and KNN.
- Random undersampling for class imbalance.
- 5-fold and 10-fold cross-validation.
- Accuracy, precision, recall, F1-score, and ROC-AUC reports.
- ROC curve, confusion matrix, and optional SHAP explainability plot.
- A small Streamlit app for checking one transaction.
- A generated demo dataset so the project runs even before downloading Kaggle data.

## Setup

Open PowerShell in this folder:

```powershell
cd "$env:USERPROFILE\Desktop\implementation of ml (minor project)"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Train With Demo Data

```powershell
python -m fraud_detection.train
```

The script will create `data/sample/demo_transactions.csv`, train every model, save the best model, and write reports.

## Train With Real Kaggle Data

Download the Online Payments Fraud Detection CSV from Kaggle and save it as:

```text
data/raw/online_payments.csv
```

The expected columns are:

```text
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
```

Then run:

```powershell
python -m fraud_detection.train
```

## Predict From A CSV

```powershell
python -m fraud_detection.predict --input data/sample/demo_transactions.csv --output reports/predictions.csv
```

## Run The Mini App

```powershell
streamlit run app.py
```

## Output Files

- `models/best_fraud_model.joblib`: trained model bundle.
- `reports/metrics.csv`: cross-validation and holdout scores.
- `reports/roc_curve.png`: ROC curve for the best model.
- `reports/confusion_matrix.png`: confusion matrix for the best model.
- `reports/shap_summary.png`: SHAP feature contribution plot, when available.

## Source Paper

The implementation is based on the paper's methodology: one-hot encoding, feature selection, random undersampling, standardization, six supervised models, cross-validation, and SHAP-based explainability.
