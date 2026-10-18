import pandas as pd
from preprocess import load_emails
from features import extract_features


def build_dataset():
    records = load_emails()

    rows = []
    for message, label in records:
        feats = extract_features(message)
        feats["label"] = label          
        rows.append(feats)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = build_dataset()

    
    df.to_csv("data/features.csv", index=False)

    
    print(df.shape)
    print(df["label"].value_counts())
    print(df.head())
    print(df.describe())