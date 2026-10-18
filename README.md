# Behaviour-Based Phishing Email Detection

A Python system that detects phishing emails from behavioural features rather than content alone. It compares Logistic Regression, Naive Bayes and Random Forest classifiers.

## Features extracted

- Suspicious links and URL length
- Urgency indicators in the text
- Sender-domain mismatches

## Project structure

- `src/preprocess.py` - loads and parses emails
- `src/features.py` - extracts behavioural features
- `src/build_dataset.py` - builds the feature dataset
- `src/train.py` - trains and evaluates the classifiers
- `src/demo.py` - runs a demo on sample emails
- `results/` - confusion matrices and feature-importance chart

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/build_dataset.py
python src/train.py
python src/demo.py
```

## Data

Datasets are not included. Download them from their original sources and place them in a `data/` folder:

- Nazario phishing corpus
- SpamAssassin / Enron
- Kaggle phishing email datasets

## Note

Created as part of an MSc Cybersecurity project at Newcastle University.
