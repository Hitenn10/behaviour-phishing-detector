"""
Train and evaluate phishing-detection models.

Loads the feature table, splits and scales the data, trains three ML models
and a rule-based baseline, then reports metrics, confusion matrices, and
Random Forest feature importances. All outputs are saved to results/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)


# ---------------------------------------------------------------------------
# 1. Load the feature table (built by build_dataset.py)
# ---------------------------------------------------------------------------
df = pd.read_csv("data/features.csv")
X = df.drop(columns=["label"])
y = df["label"]


# ---------------------------------------------------------------------------
# 2. Stratified 80/20 train/test split (reproducible)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# ---------------------------------------------------------------------------
# 3. Scale features (fit on train only, apply to both)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------------------------
# 4. Train the three machine learning models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)


# ---------------------------------------------------------------------------
# 5. Rule-based baseline (no training): urgency AND a suspicious link
# ---------------------------------------------------------------------------
def rule_based_predict(row):
    if row["urgency_score"] > 0 and row["suspicious_url"] == 1:
        return 1
    return 0

y_pred_rule = X_test.apply(rule_based_predict, axis=1)


# ---------------------------------------------------------------------------
# 6. Evaluate all approaches and save the metrics table
# ---------------------------------------------------------------------------
def score(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }

results = [score(name, y_test, model.predict(X_test_scaled))
           for name, model in models.items()]
results.append(score("Rule-based baseline", y_test, y_pred_rule))

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("results/metrics.csv", index=False)


# ---------------------------------------------------------------------------
# 7. Confusion matrix for each model (saved as PNGs)
# ---------------------------------------------------------------------------
for name, model in models.items():
    cm = confusion_matrix(y_test, model.predict(X_test_scaled))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Legitimate", "Phishing"])
    disp.plot()
    plt.title(name)
    plt.savefig(f"results/confusion_{name.replace(' ', '_')}.png",
                bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 8. Random Forest feature importances (saved as a chart)
# ---------------------------------------------------------------------------
rf = models["Random Forest"]
importances = rf.feature_importances_
order = np.argsort(importances)[::-1]

print("\nFeature importances (Random Forest):")
for i in order:
    print(f"  {X.columns[i]:22s} {importances[i]:.3f}")

plt.figure(figsize=(10, 4))
plt.barh([X.columns[i] for i in order][::-1],
         [importances[i] for i in order][::-1])
plt.xlabel("Importance")
plt.title("Random Forest feature importance")
plt.tight_layout()          # keeps labels from being clipped at the wide size
plt.savefig("results/feature_importance.png", dpi=200, bbox_inches="tight")
plt.close()