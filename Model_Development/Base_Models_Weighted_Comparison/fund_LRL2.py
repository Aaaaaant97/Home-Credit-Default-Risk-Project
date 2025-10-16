#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - Logistic Regression (L2) Baseline (Main Table Only)
=================================================================
- Reads application_train.csv / application_test.csv
- Preprocessing: categorical One-Hot encoding + numeric median imputation + standardization
- 5-fold StratifiedKFold training of LogisticRegression (L2)
- Outputs: submission / OOF predictions / coefficient importance / run configuration
"""

import os, gc, json, time, warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

# ========== Configuration ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # "train_path": "application_train.csv",
    # "test_path":  "application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_logreg_l2",

    # Logistic Regression parameters (can be tuned or grid-searched)
    "lr_params": {
        "penalty": "l2",
        "solver": "saga",       # stable for sparse and high-dimensional features
        "C": 1.0,               # regularization strength (e.g., 0.2, 0.5, 1, 2, 5)
        "max_iter": 2000,
        "tol": 1e-4,
        "n_jobs": -1,
        "class_weight": None    # or "balanced"
    }
}


def ensure_dir(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def read_data(train_path, test_path):
    """Read training and test CSV files."""
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    test = pd.read_csv(test_path)

    if os.path.isfile(train_path):
        train = pd.read_csv(train_path)
    else:
        raise FileNotFoundError(f"Training file not found: {train_path}")
    return train, test


def preprocess(train: pd.DataFrame, test: pd.DataFrame, target: str, id_col: str):
    """
    Basic preprocessing:
    - Numeric: missing values → median, standardized via StandardScaler
    - Categorical: missing values → '__MISSING__', One-Hot encoding (fit on combined train+test to avoid unseen)
    - Returns: sparse design matrices X_train, X_test, labels y, feature names
    """
    y = train[target].astype(int).values
    train = train.drop(columns=[target])

    # Combine train and test for consistent encoding
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # Identify column types
    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols]

    # Handle missing values
    combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")
    for col in num_cols:
        if col == id_col:
            continue
        if combined[col].isna().any():
            med = combined[col].median()
            combined[col] = combined[col].fillna(med)

    # Numeric standardization (set with_mean=False for sparse compatibility)
    scaler = StandardScaler(with_mean=False)
    X_num = csr_matrix(scaler.fit_transform(combined[num_cols].astype(float)))

    # One-Hot encoding for categorical features
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_cat = ohe.fit_transform(combined[cat_cols].astype(str)) if cat_cols else None

    # Combine numeric + categorical
    if X_cat is not None:
        X_all = hstack([X_num, X_cat], format="csr")
        feature_names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))
    else:
        X_all = X_num
        feature_names = list(num_cols)

    # Split back into train/test
    n_train = len(train)
    X_train = X_all[:n_train]
    X_test = X_all[n_train:]

    return X_train, X_test, y, feature_names, cat_cols, num_cols


def run_cv_logreg(X, y, X_test, feature_names, cfg):
    """Perform StratifiedKFold cross-validation and model training."""
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )

    oof = np.zeros(X.shape[0], dtype=float)
    test_pred = np.zeros(X_test.shape[0], dtype=float)
    coef_importance = np.zeros(len(feature_names), dtype=float)

    print(f"Starting {cfg['n_splits']}-fold cross-validation ...")
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = LogisticRegression(random_state=cfg["seed"], **cfg["lr_params"])
        model.fit(X_trn, y_trn)

        # Validation prediction (probabilities)
        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        print(f"[Fold {fold}] AUC = {val_auc:.6f}")

        # Accumulate test predictions
        test_pred += model.predict_proba(X_test)[:, 1] / cfg["n_splits"]

        # Coefficient importance (absolute values, averaged later)
        coef = np.abs(model.coef_.ravel())
        coef_importance += coef

        del model, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")

    coef_importance /= cfg["n_splits"]
    fi = pd.DataFrame({"feature": feature_names, "coef_abs": coef_importance}) \
        .sort_values("coef_abs", ascending=False)

    return oof, test_pred, oof_auc, fi


def main():
    """Main training and evaluation pipeline."""
    t0 = time.time()
    ensure_dir(CFG["out_dir"])

    print("Reading data...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (One-Hot + missing value imputation + standardization)...")
    X_train, X_test, y, feature_names, cat_cols, num_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"]
    )
    print(f"Feature count: {len(feature_names)} (including {len(cat_cols)} categorical columns)")

    print("Cross-validation + Logistic Regression (L2) training ...")
    oof, test_pred, oof_auc, fi = run_cv_logreg(
        X_train, y, X_test, feature_names, CFG
    )

    # Save outputs
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(CFG["out_dir"], f"submission_{stamp}.csv")
    oof_path = os.path.join(CFG["out_dir"], f"oof_{stamp}.csv")
    fi_path = os.path.join(CFG["out_dir"], f"feature_importance_{stamp}.csv")
    cfg_path = os.path.join(CFG["out_dir"], f"run_config_{stamp}.json")

    submission = test[[CFG["id_col"]]].copy()
    submission[CFG["target"]] = test_pred
    submission.to_csv(sub_path, index=False)

    oof_df = train[[CFG["id_col"]]].copy()
    oof_df["oof_pred"] = oof
    oof_df[CFG["target"]] = y
    oof_df.to_csv(oof_path, index=False)

    fi.to_csv(fi_path, index=False)

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n"
          f"- Coefficient importance: {fi_path}\n- Run configuration: {cfg_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
