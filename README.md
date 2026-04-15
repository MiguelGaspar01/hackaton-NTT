# Bank Marketing Conversion Prediction

End-to-end ML workflow to predict whether a client subscribes to a term deposit (`y`), using a notebook-driven pipeline from EDA to deployment artifacts.

## What Was Done

| Stage | Notebook | Main work | Main outputs |
|---|---|---|---|
| 1 | `EDA.ipynb` | Exploratory analysis and profiling | Data understanding and profiling report |
| 2 | `Preprocessment.ipynb` | Feature engineering, outlier handling, imputation, encoding, class imbalance treatment | `data/X_train_processed.csv`, `data/X_test_processed.csv`, `data/y_train_processed.csv`, `data/y_test_processed.csv` |
| 3 | `FeatureSelection.ipynb` | Correlation analysis + embedded feature selection | `data/X_train_selected.csv`, `data/X_test_selected.csv`, `data/y_train.csv`, `data/y_test.csv` |
| 4 | `Modelling.ipynb` | Baseline training, 5-fold CV, operating-point policy definition | `models/*.joblib`, `models/model_operating_points.csv`, `models/model_selection_rate_validation_metrics.csv` |
| 5 | `Evaluation.ipynb` | Frozen-threshold evaluation on test set, reports, SHAP | `data/test_set_predictions.csv`, `data/test_set_final_metrics_frozen_policy.csv` |

## Data and Preprocessing Summary

- Initial dataset shape: `45,211 x 17`
- Train/test split: `36,168 / 9,043`
- Target is imbalanced; undersampling is applied on train only.
- Main preprocessing steps:

1. `log1p` transform on `previous`
2. Cyclical encoding for `month` and `day` (`sin/cos`)
3. Numeric scaling (`StandardScaler` + `MinMaxScaler`)
4. Unknown categories converted to missing
5. Outlier treatment with z-score percentile cutoff
6. Imputation (`KNNImputer` for numeric, `SimpleImputer` for categorical)
7. One-hot encoding (`drop='first'`)
8. Undersampling

After preprocessing/undersampling:
- Train shape: `13,423 x 30`
- Test shape: `9,043 x 30`

## Feature Selection Summary

- Pearson correlation filter inspected relevance to target.
- Embedded selection with regularized Logistic Regression tested multiple `C` values.
- Random Forest importance ranking was also used.
- Final selected matrix used in modelling: `29` features.

## Models Trained

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost

### 5-Fold Cross-Validation (Train)

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7921 | 0.6594 | 0.7041 | 0.6809 |
| Random Forest | 0.8828 | 0.9366 | 0.6738 | 0.7837 |
| CatBoost | 0.8839 | 0.9296 | 0.6835 | 0.7878 |
| XGBoost | 0.8835 | 0.9169 | 0.6932 | 0.7895 |

## Operating Policy and Final Evaluation

Instead of using a fixed `0.5` threshold, thresholds were frozen from validation using a selection-rate/lift rule (`first lift drop >= 10% or max lift fallback`).

### Frozen Policy on Test Set

| Model | Frozen selection rate | Threshold | Test precision | Test recall | Test F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.20 | 0.6937 | 0.2481 | 0.5435 | 0.3406 | 0.7250 |
| Random Forest | 0.30 | 0.3300 | 0.2146 | 0.7665 | 0.3353 | 0.7809 |
| XGBoost | 0.30 | 0.3258 | 0.1787 | 0.8469 | 0.2952 | 0.7877 |
| CatBoost | 0.30 | 0.3199 | 0.2232 | 0.7647 | 0.3456 | 0.7842 |

Key takeaway: the project emphasizes threshold policy and lift trade-offs, not only default-threshold accuracy.

## How To Run

### Option A: Run notebooks in order (recommended for analysis)

1. `EDA.ipynb`
2. `Preprocessment.ipynb`
3. `FeatureSelection.ipynb`
4. `Modelling.ipynb`
5. `Evaluation.ipynb`
6. `Deployment.ipynb`


## Next Improvements

- Improve minority-class precision/recall balance under frozen policy.
- Try alternative imbalance strategies (`SMOTE`, weighted losses, focal approaches).
- Calibrate probabilities before threshold policy freeze.
- Add experiment tracking and model registry metadata.
