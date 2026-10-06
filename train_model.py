"""
train_model.py
Customer Account Takeover Detection
Trains and compares Logistic Regression, Decision Tree, and Random Forest.
Saves the best model to models/model.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend so plots save without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# ---------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------
DATA_PATH = "dataset/dataset_sample.csv"
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

# ---------------------------------------------------------------
# 2. Clean & prepare
#    - Drop identifier / free-text columns with very high cardinality
#    - Extract simple time features from the timestamp
# ---------------------------------------------------------------
drop_cols = ["index", "IP Address", "User Agent String", "User ID"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

if "Login Timestamp" in df.columns:
    df["Login Timestamp"] = pd.to_datetime(
        df["Login Timestamp"], errors="coerce", utc=True
    )
    df["login_hour"] = df["Login Timestamp"].dt.hour
    df["login_day_of_week"] = df["Login Timestamp"].dt.dayofweek
    df = df.drop(columns=["Login Timestamp"])

# Target column
TARGET = "Is Account Takeover"
df[TARGET] = df[TARGET].astype(int)

print("\nTarget distribution:")
print(df[TARGET].value_counts())

# ---------------------------------------------------------------
# 3. Separate features (X) and target (y)
# ---------------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

print("\nCategorical features:", categorical_cols)
print("Numeric features:", numeric_cols)

# ---------------------------------------------------------------
# 4. Preprocessing pipeline
#    - Numeric: fill missing with median, then standardize
#    - Categorical: fill missing, then one-hot encode
#      (min_frequency groups rare categories to avoid huge matrices)
# ---------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# ---------------------------------------------------------------
# 5. Train / test split
#    stratify=y keeps the same ATO ratio in both splits
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ---------------------------------------------------------------
# 6. Models
#    class_weight='balanced' compensates for the rare ATO class
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
}

# ---------------------------------------------------------------
# 7. Train and evaluate every model
# ---------------------------------------------------------------
results = []
trained_pipelines = {}

for name, model in models.items():
    print(f"\n=== {name} ===")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")

    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "ROC_AUC": auc
    })
    trained_pipelines[name] = pipe

# ---------------------------------------------------------------
# 8. Compare models
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------
# 9. Pick the best model
#    Because ATO is rare, we rank by F1 (balances precision & recall)
# ---------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_name]
print(f"\nBest model (by F1): {best_name}")

# ---------------------------------------------------------------
# 10. Save the trained pipeline
# ---------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, "models/model.pkl")
print("Saved -> models/model.pkl")

os.makedirs("results", exist_ok=True)
results_df.to_csv("results/model_comparison.csv", index=False)
print("Saved -> results/model_comparison.csv")

# ---------------------------------------------------------------
# 11. Confusion matrix plot
# ---------------------------------------------------------------
y_pred_best = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Compromised"],
            yticklabels=["Normal", "Compromised"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=120)
plt.close()
print("Saved -> results/confusion_matrix.png")

# ---------------------------------------------------------------
# 12. Feature importance plot (tree models only)
# ---------------------------------------------------------------
final_model = best_pipeline.named_steps["model"]
if hasattr(final_model, "feature_importances_"):
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = final_model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx])
    plt.title(f"Top 15 Feature Importances - {best_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=120)
    plt.close()
    print("Saved -> results/feature_importance.png")
else:
    print(f"{best_name} does not expose feature_importances_. Skipping plot.")

print("\nDone. Training complete.")