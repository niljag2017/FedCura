"""
federated_icu/data_loaders.py
==============================
Dataset loaders for generalisation experiments.

Supported datasets:
  "eicu"         — existing eICU ICU readmission (default, unchanged)
  "heart"        — UCI Heart Disease (4 hospital sites: Cleveland, Hungarian,
                   Switzerland, VA Long Beach)
  "bank"         — Bank Customer Churn Modelling (3 regions: France, Germany, Spain)

Each loader returns a pd.DataFrame compatible with the rest of the pipeline:
  - FEATURE_COLS  (numeric, pre-imputed, pre-encoded)
  - TARGET_COL    ("readmitted" column — binary 0/1)
  - "region"      column used by FedAvg Region strategy
  - "hospitalid"  column (optional, used by Quality-Gated Hospital strategy)
  - "is_synthetic" column = False for all real rows
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

# ── Heart Disease feature columns (13 + 3 interactions) ──────────────────────
HEART_FEATURE_COLS: List[str] = [
    # Raw clinical features
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope",
    # Interaction features
    "age_thalach",       # age × max heart rate (severity × function)
    "oldpeak_slope",     # ST depression × slope
    "cp_exang",          # chest pain type × exercise-induced angina
]

HEART_FEATURE_LABELS: dict = {
    "age":           "Age (years)",
    "sex":           "Sex (1=male)",
    "cp":            "Chest pain type (1–4)",
    "trestbps":      "Resting blood pressure",
    "chol":          "Serum cholesterol",
    "fbs":           "Fasting blood sugar > 120",
    "restecg":       "Resting ECG (0–2)",
    "thalach":       "Max heart rate achieved",
    "exang":         "Exercise-induced angina",
    "oldpeak":       "ST depression",
    "slope":         "Slope of peak ST",
    "age_thalach":   "Age × Max HR interaction",
    "oldpeak_slope": "ST depression × slope",
    "cp_exang":      "Chest pain × angina",
}

# ── Bank Churn feature columns ────────────────────────────────────────────────
BANK_FEATURE_COLS: List[str] = [
    # Raw features
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Gender_enc",
    # Interaction features
    "balance_salary_ratio",   # financial health
    "age_tenure",             # loyalty signal
    "products_active",        # engagement signal
]

BANK_FEATURE_LABELS: dict = {
    "CreditScore":          "Credit score",
    "Age":                  "Customer age",
    "Tenure":               "Years with bank",
    "Balance":              "Account balance",
    "NumOfProducts":        "Number of products",
    "HasCrCard":            "Has credit card",
    "IsActiveMember":       "Is active member",
    "EstimatedSalary":      "Estimated salary",
    "Gender_enc":           "Gender (1=male)",
    "balance_salary_ratio": "Balance / Salary ratio",
    "age_tenure":           "Age × Tenure",
    "products_active":      "Products × Active",
}


# ── Heart Disease loader ──────────────────────────────────────────────────────
def load_heart_disease(data_dir: str) -> pd.DataFrame:
    """
    Load the 4-site UCI Heart Disease dataset.
    Each site becomes one FL "region" (hospital).

    Expected files in data_dir:
      processed.cleveland.data
      processed.hungarian.data
      processed.switzerland.data
      processed.va.data

    Returns a DataFrame with HEART_FEATURE_COLS + "readmitted" + "region" +
    "hospitalid" + "is_synthetic".
    """
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]
    sites = [
        ("processed.cleveland.data",  "Cleveland",  1),
        ("processed.hungarian.data",  "Hungarian",  2),
        ("processed.switzerland.data","Switzerland",3),
        ("processed.va.data",         "VA",         4),
    ]
    frames = []
    for fname, region, hid in sites:
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path, header=None, names=cols, na_values="?")
        df["region"]    = region
        df["hospitalid"] = hid
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Binary target: any disease (1–4) → 1,  no disease (0) → 0
    df["readmitted"] = (df["target"] > 0).astype(int)

    # Impute missing values with column median (per full dataset)
    for col in ["trestbps","chol","fbs","restecg","thalach","exang",
                "oldpeak","slope","ca","thal"]:
        df[col] = df[col].fillna(df[col].median())

    # Interaction features
    df["age_thalach"]   = df["age"]     * df["thalach"]
    df["oldpeak_slope"] = df["oldpeak"] * df["slope"]
    df["cp_exang"]      = df["cp"]      * df["exang"]

    df["is_synthetic"] = False
    return df


# ── Bank Churn loader ─────────────────────────────────────────────────────────
def load_bank_churn(data_dir: str) -> pd.DataFrame:
    """
    Load the Bank Customer Churn Modelling dataset.
    Partitioned by Geography (France, Germany, Spain) as FL regions.

    Expected file: Churn_Modelling.csv

    Returns a DataFrame with BANK_FEATURE_COLS + "readmitted" + "region" +
    "hospitalid" + "is_synthetic".
    """
    # Try common filenames
    for fname in ["Churn_Modelling.csv", "churn_modelling.csv",
                  "Churn_Modelling.csv".lower()]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError(
            f"Could not find Churn_Modelling.csv in {data_dir}"
        )

    df = pd.read_csv(path)

    # Binary target: Exited = 1 (churned), 0 (retained)
    df["readmitted"] = df["Exited"].astype(int)

    # Encode gender
    df["Gender_enc"] = (df["Gender"].str.lower() == "male").astype(int)

    # Interaction features
    df["balance_salary_ratio"] = np.where(
        df["EstimatedSalary"] > 0,
        df["Balance"] / (df["EstimatedSalary"] + 1),
        0.0,
    )
    df["age_tenure"]    = df["Age"]           * df["Tenure"]
    df["products_active"] = df["NumOfProducts"] * df["IsActiveMember"]

    # Region = Geography; hospitalid = numeric mapping
    geo_map = {"France": 1, "Germany": 2, "Spain": 3}
    df["region"]     = df["Geography"]
    df["hospitalid"] = df["Geography"].map(geo_map).fillna(0).astype(int)

    df["is_synthetic"] = False
    return df


# ── Dataset registry ──────────────────────────────────────────────────────────
DATASET_REGISTRY = {
    "eicu":  {
        "label":        "eICU — ICU Readmission (5 US regions)",
        "task":         "ICU 30-day readmission prediction",
        "domain":       "Healthcare",
        "n_clients":    5,
        "partition_by": "Geographic region",
    },
    "heart": {
        "label":        "UCI Heart Disease (4 hospital sites)",
        "task":         "Cardiac disease diagnosis",
        "domain":       "Healthcare",
        "n_clients":    4,
        "partition_by": "Hospital (Cleveland / Hungarian / Switzerland / VA)",
        "feature_cols": HEART_FEATURE_COLS,
        "feature_labels": HEART_FEATURE_LABELS,
        "loader":       load_heart_disease,
    },
    "bank":  {
        "label":        "Bank Customer Churn (3 geographies)",
        "task":         "Customer churn prediction",
        "domain":       "Finance / Banking",
        "n_clients":    3,
        "partition_by": "Geography (France / Germany / Spain)",
        "feature_cols": BANK_FEATURE_COLS,
        "feature_labels": BANK_FEATURE_LABELS,
        "loader":       load_bank_churn,
    },
}


def get_dataset_info(dataset_type: str) -> dict:
    return DATASET_REGISTRY.get(dataset_type, DATASET_REGISTRY["eicu"])
