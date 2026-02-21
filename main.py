

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#defining current_folder

try:
    current_folder = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_folder = os.getcwd()

# LOADING DATA

df = pd.read_excel("data/app_data.xlsx")

print("First 5 rows:")
print(df.head())

print("\nShape of data:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())



# DROPPING COLUMNS WITH >50% MISSING + REMOVE LEAKAGE COLS

threshold = len(df) * 0.5
df = df.dropna(axis=1, thresh=threshold)

leakage_cols = ['Length_of_Stay', 'Management', 'Severity', 'Diagnosis_Presumptive', 'US_Number']
df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

# Dropping rows where Diagnosis is missing
df = df.dropna(subset=["Diagnosis"]).reset_index(drop=True)

# FIXING TARGET LABELS (Diagnosis -> 0/1)
# 0 = appendicitis, 1 = no appendicitis


# Normalizing strings safely
df["Diagnosis"] = df["Diagnosis"].astype(str).str.strip().str.lower()

diagnosis_map = {
    "appendicitis": 0,
    "no appendicitis": 1,
    "no_appendicitis": 1,
    "no-appendicitis": 1
}

df["Diagnosis"] = df["Diagnosis"].map(diagnosis_map)

print("\nDiagnosis distribution after mapping (including NaN):")
print(df["Diagnosis"].value_counts(dropna=False))

# Dropping rows that didn't map (rare/unknown labels)
df = df.dropna(subset=["Diagnosis"]).copy()
df["Diagnosis"] = df["Diagnosis"].astype(int)

print("\nFinal Diagnosis distribution:")
print(df["Diagnosis"].value_counts())


#  EDA CHARTS

# Chart 1 — Diagnosis Distribution
plt.figure(figsize=(6, 4))
df["Diagnosis"].value_counts().plot(kind="bar")
plt.title("Appendicitis vs No Appendicitis")
plt.xlabel("Diagnosis (0=Appendicitis, 1=No Appendicitis)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(os.path.join(current_folder, "diagnosis_chart.png"))
plt.close()
print("\nDiagnosis chart saved!")

# Chart 2 — Age Distribution
if "Age" in df.columns:
    plt.figure(figsize=(8, 4))
    plt.hist(df["Age"].dropna(), bins=20, edgecolor="black")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(os.path.join(current_folder, "age_distribution.png"))
    plt.close()
    print("Age distribution chart saved!")

# Chart 3 — WBC Count by Diagnosis
if "WBC_Count" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Diagnosis", y="WBC_Count", data=df)
    plt.title("WBC Count by Diagnosis")
    plt.xlabel("Diagnosis (0=Appendicitis, 1=No Appendicitis)")
    plt.ylabel("WBC Count")
    plt.tight_layout()
    plt.savefig(os.path.join(current_folder, "wbc_chart.png"))
    plt.close()
    print("WBC chart saved!")

# SPLITTING X AND y
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining patients:", X_train.shape[0])
print("Testing patients:", X_test.shape[0])


#  PREPROCESSING PIPELINE

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object" ,"string"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

# BUILDING MODELS AS PIPELINES


dt_pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", DecisionTreeClassifier(random_state=42))
])

rf_pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

xgb_pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss"
    ))
])

def evaluate(name, pipe):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print(classification_report(y_test, preds, target_names=["appendicitis", "no appendicitis"]))
    return acc, preds, pipe

dt_acc, dt_preds, dt_fitted = evaluate("DECISION TREE", dt_pipe)
rf_acc, rf_preds, rf_fitted = evaluate("RANDOM FOREST", rf_pipe)
xgb_acc, xgb_preds, xgb_fitted = evaluate("XGBOOST", xgb_pipe)

#  MODEL COMPARISON CHART

models = ["Decision Tree", "Random Forest", "XGBoost"]
accuracies = [dt_acc, rf_acc, xgb_acc]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f"{acc:.2%}",
             ha="center", fontsize=12)

plt.title("Model Comparison (Pipeline, No Leakage)")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(current_folder, "model_comparison.png"))
plt.close()
print("\nModel comparison chart saved!")

#  CONFUSION MATRIX

cm = confusion_matrix(y_test, xgb_preds, labels=[0, 1])
print("\nConfusion Matrix (labels [0,1]):\n", cm)

tp_app = cm[0, 0]
fn_app = cm[0, 1]
fp_app = cm[1, 0]
tn_app = cm[1, 1]

print("Appendicitis(0) -> TP:", tp_app, "FN:", fn_app)
print("Appendicitis(0) -> FP:", fp_app, "TN:", tn_app)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0 (appendicitis)", "Pred 1 (no app)"],
            yticklabels=["Actual 0 (appendicitis)", "Actual 1 (no app)"])
plt.title("Confusion Matrix - XGBoost (Pipeline)")
plt.tight_layout()
plt.savefig(os.path.join(current_folder, "confusion_matrix.png"))
plt.close()
print("Confusion matrix saved!")

# HYPERPARAMETER TUNING

param_grid = {
    "model__n_estimators":  [100, 200, 300, 500],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.3],
    "model__max_depth":     [3, 5, 7, 9],
    "model__subsample":     [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_pipe,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_pipe = random_search.best_estimator_
best_preds = best_pipe.predict(X_test)
best_acc = accuracy_score(y_test, best_preds)

print("\nBest Parameters:", random_search.best_params_)
print("Tuned XGBoost Accuracy:", best_acc)
print("Base XGBoost Accuracy:", xgb_acc)

# SAVING BEST MODEL

with open(os.path.join(current_folder, "best_model.pkl"), "wb") as f:
    pickle.dump(best_pipe, f)

print("\nBest PIPELINE model saved as best_model.pkl")

# Test load
with open(os.path.join(current_folder, "best_model.pkl"), "rb") as f:
    loaded = pickle.load(f)

print("Loaded model OK. Example prediction:", loaded.predict(X_test.iloc[:1]))

# FINAL SUMMARY

print("\n" + "="*50)
print("FINAL RESULTS SUMMARY (PIPELINE)")
print("="*50)
print(f"Decision Tree Accuracy : {dt_acc:.2%}")
print(f"Random Forest Accuracy : {rf_acc:.2%}")
print(f"XGBoost Accuracy       : {xgb_acc:.2%}")
print(f"Tuned XGBoost Accuracy : {best_acc:.2%}")
print("="*50)
print("Primary clinical focus: minimize FN for appendicitis (missed cases).")
print("="*50)