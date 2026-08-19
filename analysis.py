# %% [markdown]
# # Fraud Detection — IEEE-CIS Dataset
# Converted from analysis.ipynb

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ### Load the data sets

# %%
df = pd.read_csv('data/train_identity.csv')
df.shape

# %%
df1 = pd.read_csv('data/train_transaction.csv')
df1.shape

# %% [markdown]
# ### Load both CSVs and merge on TransactionID using Pandas
# 

# %%
data = df1.merge(df, on='TransactionID', how='left')

# %% [markdown]
# ### Display shape, dtypes, and first 10 rows of the merged dataset

# %%
print(data.dtypes)

# %%
data.shape

# %%
data.head(10)

# %% [markdown]
# ### Analyse the isFraud target column — quantify and visualize the class imbalance

# %%
fraud_counts = data["isFraud"].value_counts()
data["isFraud"].value_counts(normalize=True) * 100
fraud_counts.plot(kind="bar", edgecolor="white")
plt.title("isFraud Class Imbalance")
plt.xticks([0, 1], ["innocent", "Fraud"], rotation=0)
plt.ylabel("Count")
plt.show()

# %% [markdown]
# ### Identify missing values column-by-column

# %%
clean_df = pd.DataFrame({
    "null_count": data.isnull().sum(),
    "null_pct": (data.isnull().mean() * 100).round(2)
}).sort_values("null_pct", ascending=False)

clean_df[clean_df["null_count"] > 0]

# %% [markdown]
# ### Decide drop vs. impute threshold (suggest: drop columns with >50% missing)

# %%
to_drop   = clean_df[clean_df["null_pct"] > 50].index.tolist()
to_impute = clean_df[(clean_df["null_pct"] > 0) & (clean_df["null_pct"] <= 50)].index.tolist()
data_clean = data.drop(columns=to_drop)

# %%
data_clean.shape


# %% [markdown]
# ### Plot distribution of TransactionAmt for fraud vs. non-fraud (use log scale)

# %%
innocent = data_clean[data_clean["isFraud"] == 0]["TransactionAmt"]
fraud = data_clean[data_clean["isFraud"] == 1]["TransactionAmt"]
plt.figure(figsize=(10, 5))
plt.hist(np.log1p(innocent), bins=80, alpha=0.6, color="darkgreen", label="innocent", density=True)
plt.hist(np.log1p(fraud), bins=80, alpha=0.6, color="yellow", label="Fraud",      density=True)
plt.xlabel("(TransactionAmt)")
plt.ylabel("Density")
plt.title("TransactionAmt Distribution — Fraud vs innocent")
plt.legend()
plt.show()

# %% [markdown]
# ### Compute a correlation heatmap of the top 20 numerical features using Seaborn

# %%
num_cols = data_clean.select_dtypes(include=np.number).columns.tolist()
top20 = (
    data_clean[num_cols].corr()["isFraud"]
    .abs()
    .sort_values(ascending=False)
    .head(20)
    .index.tolist()
)
plt.figure(figsize=(14, 11))
sns.heatmap(
    data_clean[top20].corr(),
    annot=True, fmt=".2f", 
    cmap="coolwarm", center=0,
    linewidths=0.4, annot_kws={"size": 7}
)
plt.title("Correlation Heatmap")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## TASK 2 — Preprocessing, Imbalance Handling & Feature Engineering
# 

# %% [markdown]
# ### Drop columns with more than 50% missing values

# %%
threshold = 0.50
to_drop = data.columns[data.isnull().mean() > threshold].tolist()
data_clean = data.drop(columns=to_drop)
data_clean.shape

# %% [markdown]
# ### Impute remaining values using:
# - Median (numerical)
# - Mode (categorical)
# 

# %%
num_cols = data_clean.select_dtypes(include=np.number).columns.tolist()
cat_cols = data_clean.select_dtypes(include="object").columns.tolist()
for col in num_cols:
    if data_clean[col].isnull().any():
        data_clean[col] = data_clean[col].fillna(data_clean[col].median())
for col in cat_cols:
    if data_clean[col].isnull().any():
        data_clean[col] = data_clean[col].fillna(data_clean[col].mode()[0])

data_clean.isnull().sum()

# %% [markdown]
# ### Label-encode high-cardinality categorical columns

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    n_unique = data_clean[col].nunique()
    if n_unique > 10:                       
        data_clean[col] = le.fit_transform(data_clean[col].astype(str))
    else:
        data_clean[col] = le.fit_transform(data_clean[col].astype(str))
data_clean.shape
data_clean.dtypes.value_counts()

# %%
data_clean.head()

# %% [markdown]
# ## Justify your encoding strategy in a Markdown cell
# 

# %% [markdown]
# ## Encoding Strategy — Justification
# 
# ### Why Label Encoding over One-Hot Encoding?
# - High cardinality (DeviceInfo, id_30, id_31 etc.): One-Hot would create hundreds 
#   of columns, causing memory issues and the curse of dimensionality.
# - Tree-based models (XGBoost, LightGBM): Standard for fraud detection; they handle 
#   label-encoded ordinals well without implying false order.
# - Dataset size (~590K rows): One-Hot on high-cardinality cols would make the matrix 
#   extremely sparse and slow to train.
# 
# ### Imputation Choices
# - Numerical  → Median : Robust to outliers; fraud amounts are heavily skewed.
# - Categorical → Mode  : Most frequent value avoids inventing unseen categories.
# 
# ### Caveats
# - Label encoding introduces implicit ordinal relationships that do not exist.
#   Acceptable for tree models, but wrong for linear models or neural networks.
# - Columns with >50% missing were dropped; imputing the majority of a column 
#   introduces more noise than signal.
# """

# %% [markdown]
# ### Create at least 3 engineered features:
# - Examples:
# - AmtToMeanRatio = TransactionAmt / mean(TransactionAmt)
# - HourOfDay = extracted from TransactionDT
# - DeviceRisk = binary flag based on DeviceType and DeviceInfo
# 

# %% [markdown]
# #### Apply SMOTE only on the training set
# #### Scale numerical features using RobustScaler

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

# %%
data_clean = data_clean.copy()

if "DeviceType" in data_clean.columns and "DeviceInfo" in data_clean.columns:
    data_clean["DeviceRisk"] = ((data_clean["DeviceType"] == 1) | (data_clean["DeviceInfo"] == 0)).astype(int)
else:
    data_clean["DeviceRisk"] = 0  

# %%
data_clean["AmtToMeanRatio"] = data_clean["TransactionAmt"] / data_clean["TransactionAmt"].mean()
data_clean["HourOfDay"] = (data_clean["TransactionDT"] // 3600) % 24
data_clean["DayOfWeek"] = (data_clean["TransactionDT"] // (3600 * 24)) % 7
data_clean["LogTransactionAmt"] = np.log1p(data_clean["TransactionAmt"])

print("Engineered features added:")
print(data_clean[["AmtToMeanRatio", "HourOfDay", "DayOfWeek", "DeviceRisk", "LogTransactionAmt"]].head())

# %% [markdown]
# #### Perform stratified 80/20 train-test split

# %%
X = data_clean.drop(columns=["isFraud", "TransactionID"])
y = data_clean["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y)
X_train.shape
X_test.shape

# %%
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
scaler = RobustScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])   
X_test[num_cols]  = scaler.transform(X_test[num_cols])        

# %% [markdown]
# #### Report class ratio before and after SMOTE

# %%
print("\nClass ratio BEFORE SMOTE:")
before = y_train.value_counts()
before_pct = y_train.value_counts(normalize=True) * 100
print(pd.DataFrame({"count": before, "pct(%)": before_pct.round(2)}))

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nClass ratio AFTER SMOTE:")
after = pd.Series(y_train_sm).value_counts()
after_pct = pd.Series(y_train_sm).value_counts(normalize=True) * 100
print(pd.DataFrame({"count": after, "pct(%)": after_pct.round(2)}))

print(f"\nTrain size before SMOTE : {X_train.shape[0]:,}")
print(f"Train size after SMOTE  : {X_train_sm.shape[0]:,}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Class Distribution: Before vs After SMOTE", fontsize=13, fontweight="bold")
for ax, counts, title in zip(
    axes,
    [before, after],
    ["Before SMOTE (Train set)", "After SMOTE (Train set)"]):
    ax.bar(["Legitimate", "Fraud"], counts.sort_index().values,
           color=["#4C9BE8", "#E85C5C"], edgecolor="white", width=0.5)
    for i, v in enumerate(counts.sort_index().values):
        ax.text(i, v + counts.max() * 0.01, f"{v:,}", ha="center", fontweight="bold")
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
plt.tight_layout()
plt.show()

# %% [markdown]
# ## TASK 3 — Model Training, Comparison & Threshold Optimization
# 

# %%
import matplotlib.gridspec as gridspec
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

COLORS = {"LightGBM": "#4C9BE8", "XGBoost": "#E8A84C", "IsolationForest": "#A84CE8"}

# %% [markdown]
# ### LightGBM Classifier

# %%
lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05,num_leaves=63, random_state=42, n_jobs=-1)
lgbm.fit(X_train_sm, y_train_sm,eval_set=[(X_test, y_test)],callbacks=[])

# %% [markdown]
# ### XGBoost Classifier

# %%
xgb = XGBClassifier(n_estimators=500, learning_rate=0.05,max_depth=6, random_state=42,eval_metric="logloss", verbosity=0,use_label_encoder=False, n_jobs=-1)
xgb.fit(X_train_sm, y_train_sm)

# %% [markdown]
# ### Isolation Forest

# %%
iso = IsolationForest(n_estimators=200, contamination=0.035,random_state=42, n_jobs=-1)
iso.fit(X_train_sm)

# %%
def iso_proba(model, X):
    """Convert IsolationForest decision scores to [0,1] probabilities."""
    scores = model.decision_function(X)          
    scores_inv = -scores                   
    proba = (scores_inv - scores_inv.min()) / (scores_inv.max() - scores_inv.min())
    return proba

preds = {
    "LightGBM"       : lgbm.predict(X_test),
    "XGBoost"        : xgb.predict(X_test),
    "IsolationForest": (iso.predict(X_test) == -1).astype(int),
}
probas = {
    "LightGBM"       : lgbm.predict_proba(X_test)[:, 1],
    "XGBoost"        : xgb.predict_proba(X_test)[:, 1],
    "IsolationForest": iso_proba(iso, X_test),
}

# %% [markdown]
# ### Evaluate Using:
# -Accuracy
# -Precision
# -Recall
# -F1-Score
# -ROC-AUC
# -PR-AUC
# 

# %%
rows = []
for name in preds:
    y_pred  = preds[name]
    y_proba = probas[name]
    rows.append({
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_proba), 4),
        "PR-AUC"   : round(average_precision_score(y_test, y_proba), 4),
    })

results_df = pd.DataFrame(rows).set_index("Model")
print(results_df.to_string())

# %% [markdown]
# ### Visualizations:
# #### Confusion Matrix for each model
# 

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
for ax, (name, y_pred) in zip(axes, preds.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=ax,
                xticklabels=["innocent", "Fraud"],
                yticklabels=["innocent", "Fraud"],
                linewidths=0.5, cbar=False)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("plot_confusion_matrices.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### ROC Curve
# 

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Model Evaluation Curves", fontsize=14, fontweight="bold")

for name, y_proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

axes[0].plot([0, 1], [0, 1], 'k--', label="Random Guess")
axes[0].set_title("ROC Curve", fontsize=12)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

for name, y_proba in probas.items():
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})")

axes[1].set_title("Precision-Recall Curve", fontsize=12)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(loc="lower left")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot_roc_pr_curves.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Precision-Recall Curve
# 

# %%
best_model_name = "XGBoost"
y_proba_best = probas[best_model_name]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_best)

f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
best_f1 = f1_scores[optimal_idx]

print(f"Optimal Threshold for {best_model_name}: {optimal_threshold:.4f}")
print(f"Best F1-Score: {best_f1:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_scores, label="F1 Score", color="purple", lw=2)
plt.axvline(optimal_threshold, color="red", linestyle="--", 
            label=f"Optimal Threshold ({optimal_threshold:.2f})")
plt.title(f"Threshold Optimization ({best_model_name})", fontsize=12, fontweight="bold")
plt.xlabel("Decision Threshold")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("plot_threshold_optimization.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Tune best model using:
# #### Optuna
# #### OR RandomizedSearchCV
# 

# %%
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100) 
    }
    model = XGBClassifier(**params, random_state=42, eval_metric="logloss")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, zero_division=0)
        f1_scores.append(f1)
    return np.mean(f1_scores)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10) 
print(f"Best trial F1-score: {study.best_value:.4f}")
print("Best parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# %% [markdown]
# ### TASK 4 — Explainable AI with SHAP Values [ADVANCED]
# 

# %% [markdown]
# ### Install and run SHAP library
# ### Generate Global SHAP Summary Plot (top 20 features)
# ### Generate SHAP Waterfall Plots for:
# - Confirmed fraud case
# - Borderline case (~0.50 probability)
# - Legitimate transaction
# 

# %%
import shap
import numpy as np
import matplotlib.pyplot as plt
explainer = shap.TreeExplainer(xgb)
shap_values = explainer(X_test)
plt.figure(figsize=(10, 8))
plt.title("SHAP Global Summary Plot (Top 20 Features)", fontsize=14, fontweight="bold")
shap.summary_plot(shap_values, X_test, max_display=20, show=False)
plt.savefig("shap_summary_plot.png", bbox_inches="tight")
plt.show()

# %%
y_proba = xgb.predict_proba(X_test)[:, 1]
fraud_idx_list = np.where((y_proba > 0.90) & (y_test == 1))[0]
idx_fraud = fraud_idx_list[0] if len(fraud_idx_list) > 0 else np.argmax(y_proba)
idx_borderline = np.argmin(np.abs(y_proba - 0.50))
legit_idx_list = np.where((y_proba < 0.05) & (y_test == 0))[0]
idx_legit = legit_idx_list[0] if len(legit_idx_list) > 0 else np.argmin(y_proba)

# %%
plt.figure(figsize=(8, 5))
shap.plots.waterfall(shap_values[idx_fraud], max_display=10, show=False)
plt.title(f"SHAP Waterfall: Confirmed Fraud (Prob: {y_proba[idx_fraud]:.2f})", fontweight="bold")
plt.savefig("shap_waterfall_fraud.png", bbox_inches="tight")
plt.show()

# %%
plt.figure(figsize=(8, 5))
shap.plots.waterfall(shap_values[idx_borderline], max_display=10, show=False)
plt.title(f"SHAP Waterfall: Borderline Case (Prob: {y_proba[idx_borderline]:.2f})", fontweight="bold")
plt.savefig("shap_waterfall_borderline.png", bbox_inches="tight")
plt.show()

# %%
plt.figure(figsize=(8, 5))
shap.plots.waterfall(shap_values[idx_legit], max_display=10, show=False)
plt.title(f"SHAP Waterfall: Legitimate Transaction (Prob: {y_proba[idx_legit]:.2f})", fontweight="bold")
plt.savefig("shap_waterfall_legit.png", bbox_inches="tight")
plt.show()

# %%
mean_shap_values = np.abs(shap_values.values).mean(axis=0)
top_feature_idx = np.argmax(mean_shap_values)
top_feature_name = X_test.columns[top_feature_idx]
shap.dependence_plot(top_feature_name, explainer.shap_values(X_test), X_test, show=False)
plt.title(f"SHAP Dependence Plot: {top_feature_name}", fontsize=14, fontweight="bold")
plt.savefig("shap_dependence_plot.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## TASK 4 — Explainable AI  with SHAP Values [ADVANCED] documentation
# By analyzing the SHAP waterfall plots, we can translate the model's complex mathematical decisions into plain-English business logic:
# 
# * **1. Confirmed Fraud Case (High Probability):** The model confidently flagged this transaction as fraud. The primary drivers were **[C1]** and **[C11]**. The values for these features were highly anomalous compared to normal baseline data, strongly pushing the risk score upward and triggering a severe alert.
# * **2. Borderline Case (~0.50 Probability):** This transaction triggered a roughly 50% probability, landing squarely in a gray area. While normal behaviors in **[DayOfWeek]** pushed the risk score down, suspicious signals from **[C1]** pushed the score back up. Because these forces counterbalanced each other, the model could not make a definitive classification, indicating that this specific transaction requires manual review by a human analyst.
# * **3. Legitimate Transaction (Low Probability):** The model confidently cleared this transaction. The baseline risk was already low, and standard, healthy signals from **[DayOfWeek]** and **[C14]** drove the fraud probability down even further. 
# 
# ### SHAP Importance vs. Model Feature Importance
# 
# While standard model feature importance (such as XGBoost's weight or gain) and SHAP values both rank features, they serve different purposes:
# 
# * **Model Feature Importance:** This measures how useful a feature is for *constructing the model* (e.g., how often it is used to split a tree). However, it does not tell us the *direction* of the impact—we know the feature matters, but we don't know if a high value indicates fraud or legitimacy.
# * **SHAP Importance:** This measures the actual *contribution to the final prediction*. It explains exactly how much a specific feature value shifted the probability from the base average toward Fraud or Legitimate for an individual transaction.
# 
# **Conclusion:** Standard model importance is useful during the initial data science engineering phase for feature selection. However, SHAP is essential for operational transparency, as it provides the directional, individualized context required for auditing and business decision-making.

# %% [markdown]
# ### TASK 5 — Risk Segmentation & Fraud Pattern Analysis [ADVANCED]

# %% [markdown]
# ### Risk Tiers:
# - 🔴 Critical Risk → probability ≥ 0.75
# - 🟡 Suspicious → probability between 0.40 and 0.74
# - 🟢 Clear → probability < 0.40
# 

# %%
analysis_df = X_test.copy()
analysis_df['Fraud_Probability'] = xgb.predict_proba(X_test)[:, 1]

conditions = [
    analysis_df['Fraud_Probability'] >= 0.75,
    (analysis_df['Fraud_Probability'] >= 0.40) & (analysis_df['Fraud_Probability'] < 0.75),
    analysis_df['Fraud_Probability'] < 0.40
]
choices = ['Critical Risk', 'Suspicious', 'Clear']
analysis_df['Risk_Tier'] = np.select(conditions, choices, default='Unknown')

tier_order = ['Critical Risk', 'Suspicious', 'Clear']
analysis_df['Risk_Tier'] = pd.Categorical(analysis_df['Risk_Tier'], categories=tier_order, ordered=True)

print("--- Risk Tier Summary ---")
tier_summary = analysis_df.groupby('Risk_Tier', observed=False).agg(
    Total_Transactions=('Fraud_Probability', 'count'),
    Avg_TransactionAmt=('TransactionAmt', 'mean')
).reset_index()

print(tier_summary.to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Fraud Risk Tier Analysis", fontsize=14, fontweight="bold")

sns.countplot(
    data=analysis_df, 
    x='Risk_Tier', 
    hue='Risk_Tier',        
    legend=False,           
    palette=['#e74c3c', '#f1c40f', '#2ecc71'], 
    ax=axes[0]
)
axes[0].set_title("Transaction Volume by Risk Tier", fontweight="bold")
axes[0].set_ylabel("Number of Transactions")
axes[0].set_xlabel("")

sns.barplot(
    data=tier_summary, 
    x='Risk_Tier', 
    y='Avg_TransactionAmt', 
    hue='Risk_Tier',       
    legend=False,           
    palette=['#e74c3c', '#f1c40f', '#2ecc71'], 
    ax=axes[1]
)
axes[1].set_title("Average Transaction Amount by Risk Tier", fontweight="bold")
axes[1].set_ylabel("Average Amount ($)")
axes[1].set_xlabel("")

plt.tight_layout()
plt.savefig("plot_risk_segmentation.png", bbox_inches="tight")
plt.show()

# %%
critical_df = analysis_df[analysis_df['Risk_Tier'] == 'Critical Risk']

print("\n--- Top 3 Patterns in Critical Risk Tier ---")
if not critical_df.empty:
    top_device = critical_df['DeviceRisk'].value_counts().idxmax()
    device_pct = (critical_df['DeviceRisk'].value_counts().max() / len(critical_df)) * 100

    top_hour = critical_df['HourOfDay'].value_counts().idxmax()
    hour_pct = (critical_df['HourOfDay'].value_counts().max() / len(critical_df)) * 100
    
    avg_amt = critical_df['TransactionAmt'].mean()
    
    print(f"1. Dominant Attack Vector: Risk Category '{top_device}' accounts for {device_pct:.1f}% of critical alerts.")
    print(f"2. Peak Fraud Hour: Hour {top_hour} sees the highest concentration of severe attacks ({hour_pct:.1f}% of critical alerts).")
    print(f"3. High-Value Targets: The average transaction amount for critical risk is ${avg_amt:.2f}, indicating targeted high-value theft.")
else:
    print("No transactions fell into the Critical Risk tier in this dataset.")



import joblib
joblib.dump(lgbm, 'fraud_model.pkl')  # LightGBM outperforms XGBoost on all metrics (F1: 0.6429 vs 0.5748, PR-AUC: 0.7287 vs 0.6160)

X_test_export = X_test.copy().reset_index()
X_test_export = X_test_export.rename(columns={'index': 'TransactionID'})


X_test_export.to_csv('test_features.csv', index=False)
y_test.to_csv('test_labels.csv', index=False)

print("Artifacts saved successfully!")



# %%
import plotly.express as px
import shap
from sklearn.metrics import precision_recall_curve

plot_df = X_test.copy()
plot_df['Actual_Fraud'] = y_test.values if isinstance(y_test, pd.Series) else y_test
plot_df['Fraud_Probability'] = y_proba_best 

conditions = [
    plot_df['Fraud_Probability'] >= 0.75,
    (plot_df['Fraud_Probability'] >= 0.40) & (plot_df['Fraud_Probability'] < 0.75),
    plot_df['Fraud_Probability'] < 0.40
]
plot_df['Risk_Tier'] = np.select(conditions, [' Critical', ' Suspicious', ' Clear'], default='Unknown')


tier_order = [' Critical', ' Suspicious', ' Clear']
plot_df['Risk_Tier'] = pd.Categorical(plot_df['Risk_Tier'], categories=tier_order, ordered=True)


explainer = shap.TreeExplainer(xgb) 
shap_values = explainer(X_test)

plt.figure(figsize=(10, 6))
plt.title("1. SHAP Global Summary Plot", fontsize=14, fontweight="bold")
shap.summary_plot(shap_values, X_test, max_display=20, show=False)
plt.savefig("charts/chart_1_shap.png", bbox_inches="tight")
plt.show()

# %%
plt.figure(figsize=(10, 4))
hourly_fraud = plot_df.groupby('HourOfDay')['Actual_Fraud'].mean() * 100

sns.barplot(
    x=hourly_fraud.index, 
    y=hourly_fraud.values, 
    hue=hourly_fraud.index, 
    palette=["salmon"]*len(hourly_fraud), 
    legend=False
)
plt.title("2. Fraud Rate by Hour of Day", fontsize=14, fontweight="bold")
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Fraud Rate (%)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("chart_2_hourly_fraud.png", bbox_inches="tight")
plt.show()

# %%
plt.figure(figsize=(10, 4))
sns.kdeplot(data=plot_df[plot_df['Actual_Fraud'] == 0], x='TransactionAmt', fill=True, label="Legitimate", color="green", alpha=0.3)
sns.kdeplot(data=plot_df[plot_df['Actual_Fraud'] == 1], x='TransactionAmt', fill=True, label="Fraud", color="red", alpha=0.5)

plt.title("3. Transaction Amount Distribution (Log Scale)", fontsize=14, fontweight="bold")
plt.xlabel("Transaction Amount ($)")
plt.ylabel("Density")
plt.xscale('log') 
plt.legend()
plt.tight_layout()
plt.savefig("charts/chart_3_amount_dist.png", bbox_inches="tight")
plt.show()

# %%
plt.figure(figsize=(6, 6))
tier_counts = plot_df['Risk_Tier'].value_counts()
colors = {' Clear': '#2ecc71', ' Suspicious': '#f1c40f', ' Critical': '#e74c3c'}
plot_colors = [colors.get(tier, '#95a5a6') for tier in tier_counts.index]

plt.pie(tier_counts, labels=tier_counts.index, autopct='%1.1f%%', startangle=90, colors=plot_colors, 
        wedgeprops=dict(width=0.4, edgecolor='w'), pctdistance=0.75, textprops={'fontsize': 12})

plt.title("4. Transaction Distribution by Risk Tier", fontsize=14, fontweight="bold")
plt.savefig("charts/    chart_4_donut.png", bbox_inches="tight")
plt.show()

# %%
plt.figure(figsize=(8, 5))
precisions, recalls, thresholds = precision_recall_curve(plot_df['Actual_Fraud'], plot_df['Fraud_Probability'])

f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
optimal_idx = np.argmax(f1_scores)
opt_thresh = thresholds[optimal_idx]
opt_prec = precisions[optimal_idx]
opt_rec = recalls[optimal_idx]

plt.plot(recalls, precisions, label="PR Curve", color="blue", lw=2)
plt.scatter(opt_rec, opt_prec, color="red", s=100, zorder=5, 
            label=f"Optimal Threshold ({opt_thresh:.2f})\nF1: {f1_scores[optimal_idx]:.2f}")

plt.title("5. Precision-Recall Curve (with Optimal Threshold)", fontsize=14, fontweight="bold")
plt.xlabel("Recall (Fraud Caught)")
plt.ylabel("Precision (Accuracy of Fraud Alerts)")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("charts/chart_5_pr_optimal.png", bbox_inches="tight")
plt.show()

# %%
scatter_df = pd.concat([
    plot_df[plot_df['Risk_Tier'] != 'Clear'], 
    plot_df[plot_df['Risk_Tier'] == 'Clear'].sample(n=min(3000, len(plot_df[plot_df['Risk_Tier'] == '🟢 Clear'])), random_state=42)
])

fig = px.scatter(
    scatter_df, 
    x="HourOfDay", 
    y="TransactionAmt", 
    color="Fraud_Probability",
    color_continuous_scale="Reds",
    hover_data=["Actual_Fraud"],
    title="BONUS: Transaction Amount vs Hour of Day (Colored by Fraud Probability)",
    labels={"HourOfDay": "Time of Day", "TransactionAmt": "Amount ($)", "Fraud_Probability": "Risk Score"}
)
fig.update_layout(yaxis_type="log")
fig.show()

# %%
X_test_small = X_test_export.sample(n=1000, random_state=42)
y_test_small = y_test_reset.loc[X_test_small.index]

X_test_small = X_test_small.copy()
X_test_small.insert(0, 'TransactionID', X_test_small.index)

X_test_small.to_csv('dashboard/test_features.csv', index=False)
y_test_small.to_csv('dashboard/test_labels.csv', index=False)
print(X_test_small.shape)
