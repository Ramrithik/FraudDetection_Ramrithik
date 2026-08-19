# Fraud Detection — IEEE-CIS Transaction Dataset

An end-to-end machine learning pipeline for detecting fraudulent transactions, built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) Kaggle dataset (590,540 transactions, 434 features after identity merge).

## Live Dashboard
https://ramtithiksfrauddetection.streamlit.app/

---

## Dataset & Class Imbalance

| Class | Count | Percentage |
|---|---|---|
| Legitimate | 455,902 (train) | 96.5% |
| Fraud | 16,530 (train) | 3.5% |

**Imbalance handling:**
- **SMOTE** (Synthetic Minority Oversampling) applied on training set only — resampled to 50/50 balance (911,804 training samples)
- **RobustScaler** for feature scaling (robust to outliers in skewed fraud amounts)
- **Stratified 80/20 train-test split** to preserve class ratios in evaluation

## Feature Engineering

Five engineered features added to the 220 columns retained after dropping >50% missing:

| Feature | Logic |
|---|---|
| `AmtToMeanRatio` | `TransactionAmt / mean(TransactionAmt)` |
| `HourOfDay` | Extracted from `TransactionDT` (mod 24) |
| `DayOfWeek` | Extracted from `TransactionDT` (mod 7) |
| `DeviceRisk` | Binary flag from `DeviceType` + `DeviceInfo` |
| `LogTransactionAmt` | `log1p(TransactionAmt)` for skew reduction |

## Model Comparison

Three models trained on SMOTE-balanced data. Evaluated on the original (imbalanced) held-out test set (118,108 transactions):

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **LightGBM (deployed)** | **88.89%** | **50.35%** | **64.29%** | **0.9484** | **0.7287** |
| XGBoost | 82.66% | 44.06% | 57.48% | 0.9143 | 0.6160 |
| Isolation Forest | 14.27% | 5.71% | 8.16% | 0.6615 | 0.0767 |

> **Note:** LightGBM outperforms XGBoost on every metric and is the deployed model (n_estimators=500, learning_rate=0.05, num_leaves=63). PR-AUC is used as the primary evaluation metric because accuracy is misleading at 96.5% class imbalance — a model predicting "always legitimate" achieves 96.5% accuracy while catching zero fraud.

### Threshold Analysis

- **F1-optimal decision threshold:** 0.3653 (F1 = 0.6018 at that point), found via precision-recall curve analysis
- **Operational risk tiers** (used in the dashboard): 0.75 / 0.40 — these are manually chosen triage cutoffs for the investigation team, not optimized thresholds

### Hyperparameter Tuning (Optuna)

Optuna was used to search XGBoost hyperparameters (including `scale_pos_weight` for class imbalance) over 10 trials with 3-fold stratified CV. Best trial F1: 0.5543 — worse than the untuned SMOTE-trained baseline (F1: 0.5748). The tuning validated that the default hyperparameters already outperformed the searched configurations, so the baseline model was kept. This is a case where automated tuning did not improve performance, and the result was correctly discarded rather than force-adopted.

## Explainability (SHAP)

SHAP (TreeExplainer) is used at two levels:

1. **Global feature importance:** Summary plot of top 20 features across the full test set, showing directional impact (which feature values push toward fraud vs. legitimate)
2. **Per-transaction explanations:** The dashboard generates SHAP waterfall plots on-the-fly for any individual transaction, translating model math into a plain-language **Analyst Briefing**:
   - 🔴 **Critical alerts** explain the primary fraud signal and recommend immediate action
   - 🟡 **Borderline cases** explain the conflicting signals and recommend manual review
   - 🟢 **Cleared transactions** confirm which features support legitimacy

This SHAP-to-English translation is a deliberate product decision — fraud investigation teams need actionable explanations, not probability scores.

## Dashboard

An interactive Streamlit dashboard for exploring model predictions on a held-out test sample (batch inference on static data, not real-time):

- **System Overview:** KPIs (total transactions, confirmed fraud count, detection rate) + risk tier distribution and amount breakdowns
- **Transaction Explorer:** Filterable, sortable transaction table with risk-score heatmapping and tier-based filtering (🔴 Critical / 🟡 Suspicious / 🟢 Clear)
- **Decision Explainer:** Enter any Transaction ID → generates SHAP waterfall + plain-language Analyst Briefing explaining *why* the model scored it that way

> **Note:** `TransactionAmt` values displayed in the dashboard are post-RobustScaler output (standardized), not original dollar amounts. The raw training data is not included in this repository, so inverse-transforming is not possible without re-running the pipeline.

### Risk Tiers (Operational)

| Tier | Threshold | Action |
|---|---|---|
| 🔴 Critical | Probability ≥ 0.75 | Immediate block or highest-priority manual review |
| 🟡 Suspicious | 0.40 ≤ Probability < 0.75 | Queued for secondary review / step-up authentication |
| 🟢 Clear | Probability < 0.40 | Processed normally |

## Tech Stack

| Component | Technology |
|---|---|
| ML Models | XGBoost, LightGBM, Isolation Forest |
| Imbalance Handling | SMOTE (imblearn) |
| Feature Scaling | RobustScaler (scikit-learn) |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP (TreeExplainer) |
| Dashboard | Streamlit, Plotly |
| Data Processing | Pandas, NumPy |

## Project Structure

```
├── analysis.py             # Full pipeline as Python script (editable version of analysis.ipynb)
├── app.py                  # Streamlit dashboard (deployed)
├── fraud_model.pkl         # Serialized LightGBM model
├── test_features.csv       # Held-out test features (1K-row sample, scaled)
├── test_labels.csv         # Held-out test labels
├── requirements.txt        # Python dependencies
├── plot_*.png              # Confusion matrices, ROC/PR curves, threshold optimization
├── shap_*.png              # SHAP summary, dependence, and waterfall plots
├── chart_*.png             # Additional visualizations (hourly fraud, amount distribution, etc.)
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```
