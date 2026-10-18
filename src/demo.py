# %%
import os, sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

assert os.path.exists("data/features.csv"), "Missing data/features.csv - run build_dataset.py first"
assert os.path.isdir("data/raw/nazario"), "Missing data/raw/nazario - raw emails not on this machine"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_emails, get_headers, get_body_text, get_urls
from features import extract_features

df = pd.read_csv("data/features.csv")
FEATURE_COLS = [c for c in df.columns if c != "label"]
X = df[FEATURE_COLS]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler().fit(X_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler.transform(X_train), y_train)

def rule_based_predict(feats):
    return 1 if (feats["urgency_score"] > 0 and feats["suspicious_url"] == 1) else 0

def classify(message):
    feats = extract_features(message)
    row = pd.DataFrame([feats]).reindex(columns=FEATURE_COLS, fill_value=0)
    scaled = scaler.transform(row)
    rf_pred = rf.predict(scaled)[0]
    conf = rf.predict_proba(scaled)[0].max()
    return feats, rf_pred, conf, rule_based_predict(feats)

records = load_emails()
demo_phish = None
for msg, label in records:
    if label == 1:
        feats, rf_pred, _, rule_pred = classify(msg)
        if rf_pred == 1 and rule_pred == 0:
            demo_phish = msg
            break
demo_legit = next(m for m, label in records if label == 0)
print("Ready. Demo emails selected.")

# %%
h = get_headers(demo_phish)
print("From:       ", h["from"])
print("Return-Path:", h["return_path"])
print("Subject:    ", h["subject"])
print("URLs:       ", get_urls(demo_phish)[:5])
print("\nBody (first 600 chars):\n", get_body_text(demo_phish)[:600])

# %%
feats, rf_pred, conf, rule_pred = classify(demo_phish)
pd.DataFrame([feats]).T.rename(columns={0: "value"})

# %%
print("Random Forest:", "PHISHING" if rf_pred == 1 else "legitimate", "| confidence", round(conf, 3))

# %%
pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).round(3)

# %%
print("Baseline:", "PHISHING" if rule_pred == 1 else "legitimate  <-- MISSED IT")

# %%
h = get_headers(demo_legit)
print("Subject:", h["subject"])
feats2, rf_pred2, conf2, _ = classify(demo_legit)
print("Random Forest:", "PHISHING" if rf_pred2 == 1 else "legitimate", "| confidence", round(conf2, 3))
pd.DataFrame([feats2]).T.rename(columns={0: "value"})
# %%
 