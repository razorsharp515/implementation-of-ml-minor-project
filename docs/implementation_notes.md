# Implementation Notes

This project implements the methodology from the attached paper as a runnable minor project.

## Paper Method Followed

- Dataset shape expected: online payment transactions with `step`, `type`, `amount`, account balance columns, recipient/origin names, `isFraud`, and `isFlaggedFraud`.
- Preprocessing:
  - One-hot encode the `type` column.
  - Drop identifier columns `nameOrig` and `nameDest`.
  - Drop `isFlaggedFraud`.
  - Drop `newbalanceDest` and `newbalanceOrig` because the paper removes them for high correlation.
  - Standardize numerical features.
- Imbalance mitigation:
  - Use random undersampling on the majority non-fraud class.
- Models:
  - Logistic Regression
  - Decision Tree
  - Gaussian Naive Bayes
  - Random Forest
  - XGBoost
  - K-Nearest Neighbors
- Validation:
  - 5-fold and 10-fold stratified cross-validation.
  - Accuracy, precision, recall, F1-score, and ROC-AUC.
- Explainability:
  - SHAP summary plot for the selected best model when SHAP supports the fitted estimator.

## Practical Difference From The Paper

The paper uses a large Kaggle dataset with over one million rows. This folder includes a synthetic demo dataset generator so the project runs immediately on any computer. For final college results, download the Kaggle dataset and place it at:

```text
data/raw/online_payments.csv
```

Then rerun training.
