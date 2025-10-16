#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - RidgeClassifier + Platt Calibration Baseline (Main Table Only)
============================================================================
- Reads application_train.csv / application_test.csv
- Preprocessing: categorical LabelEncoding + numeric median imputation + numeric standardization
- Outer 5-fold StratifiedKFold
- Model: RidgeClassifier (alpha grid) + CalibratedClassifierCV(method='sigmoid')
- Outputs: submission / OOF predictions / (|coef|) feature importance / run configuration
"""

import os, gc, json, time
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

# Optional progress bar
try:
    from tqdm import tqdm
    _use_tqdm = True
except Exception:
    _use_tqdm = False


# ========== Configuration ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # "train_path": "application_train.csv",
    # "test_path":  "application_test.csv",
    # "train_path": "fli_application_train.csv",
    # "test_path":  "fli_application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path":  "../Features/All_application_test.csv",
    "out_dir": "baseline_ridge_calibrated",

    # Ridge alpha grid (tune as needed)
    "alpha_grid": [0.1, 0.3, 1.0, 3.0, 10.0],

    # Calibration method: 'sigmoid' (Platt) or 'isotonic' (more flexible, higher overfitting risk)
    "calibration": "sigmoid",

    # Whether to standardize only numeric features (linear models often benefit)
    "scale_numeric_only": True
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


def preprocess(train, test, target, id_col, scale_numeric_only=True):
    """
    Basic preprocessing: categorical encoding + missing value imputation + (optional) standardization.
    - LabelEncode categorical columns on combined train+test (avoids unseen categories)
    - Median-impute numeric columns
    - Standardize features (numeric only or all, depending on flag)
    """
    y = train[target].astype(int).values
    train = train.drop(columns=[target])

    # Combine for consistent category handling and missing-value treatment
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # Identify column types
    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols]

    # Missing values
    combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")
    for col in num_cols:
        if col == id_col:
            continue
        if combined[col].isna().any():
            median_val = combined[col].median()
            combined[col] = combined[col].fillna(median_val)

    # Label Encoding (fit on combined to include all categories)
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Standardization
    if scale_numeric_only:
        scaler = StandardScaler()
        to_scale = [c for c in num_cols if c != id_col]
        combined[to_scale] = scaler.fit_transform(combined[to_scale])
    else:
        scaler = StandardScaler()
        feats = [c for c in combined.columns if c != id_col]
        combined[feats] = scaler.fit_transform(combined[feats])

    # Split back to train/test
    train_pre = combined.iloc[:len(train)].reset_index(drop=True)
    test_pre  = combined.iloc[len(train):].reset_index(drop=True)

    # Feature list
    features = [c for c in train_pre.columns if c != id_col]
    return train_pre, test_pre, y, features, cat_cols


def fit_calibrated_ridge(X_trn, y_trn, alpha, method="sigmoid", seed=42):
    """
    Fit RidgeClassifier and wrap with CalibratedClassifierCV for probability calibration.
    Uses inner CV (CalibratedClassifierCV(cv=5)) on the training fold to avoid leakage.
    """
    base = RidgeClassifier(alpha=alpha)
    clf = CalibratedClassifierCV(estimator=base, method=method, cv=5)
    clf.fit(X_trn, y_trn)
    return clf


def run_cv_ridge_calibrated(train_pre, y, test_pre, features, id_col, cfg):
    """Outer CV: choose best alpha per fold, calibrate, evaluate, and aggregate feature importances."""
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )

    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)

    # Linear-model "importance": average |coef| across inner calibrated estimators and folds
    coef_importance = pd.DataFrame(0.0, index=features, columns=["coef_abs_sum"])

    X = train_pre[features].values
    X_test = test_pre[features].values

    print(f"Starting {cfg['n_splits']}-fold cross-validation ...")
    iterator = folds.split(X, y)
    if _use_tqdm:
        iterator = tqdm(list(iterator), total=cfg['n_splits'])

    for fold_id, (trn_idx, val_idx) in enumerate(iterator, 1):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Simple per-fold selection of best alpha on the outer fold
        best_alpha, best_auc = None, -1.0
        for a in cfg["alpha_grid"]:
            tmp_clf = fit_calibrated_ridge(
                X_trn, y_trn, alpha=a, method=cfg["calibration"], seed=cfg["seed"]
            )
            val_prob = tmp_clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_prob)
            if auc > best_auc:
                best_auc, best_alpha = auc, a
            del tmp_clf

        # Retrain with the best alpha for this fold
        clf = fit_calibrated_ridge(
            X_trn, y_trn, alpha=best_alpha, method=cfg["calibration"], seed=cfg["seed"]
        )

        # Validation predictions (probabilities)
        val_prob = clf.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_prob
        val_auc = roc_auc_score(y_val, val_prob)
        print(f"[Fold {fold_id}] AUC = {val_auc:.6f}  (best alpha = {best_alpha}, "
              f"alpha_grid={cfg['alpha_grid']})")

        # Test predictions (average over folds)
        test_prob = clf.predict_proba(X_test)[:, 1]
        test_pred += test_prob / cfg["n_splits"]

        # Collect coefficients from inner calibrated estimators
        # Note: probabilities are calibrated; we only use the linear part's coefficients as a reference.
        coefs = []
        for est in clf.calibrated_classifiers_:
            if hasattr(est.estimator, "coef_"):
                coefs.append(est.estimator.coef_.ravel())
        if len(coefs) > 0:
            mean_coef = np.mean(np.vstack(coefs), axis=0)
            coef_importance["coef_abs_sum"] += np.abs(mean_coef)

        del clf, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")

    coef_importance = coef_importance.sort_values("coef_abs_sum", ascending=False)
    return oof, test_pred, oof_auc, coef_importance


def main():
    t0 = time.time()
    ensure_dir(CFG["out_dir"])

    print("Reading data...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (categorical encoding, imputation, standardization)...")
    train_pre, test_pre, y, features, cat_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"], CFG["scale_numeric_only"]
    )
    print(f"Feature count: {len(features)} (including {len(cat_cols)} categorical columns)")

    print("Cross-validation + Ridge (calibrated) training ...")
    oof, test_pred, oof_auc, fi = run_cv_ridge_calibrated(
        train_pre, y, test_pre, features, CFG["id_col"], CFG
    )

    # Save outputs
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(CFG["out_dir"], f"submission_{stamp}.csv")
    oof_path = os.path.join(CFG["out_dir"], f"oof_{stamp}.csv")
    fi_path  = os.path.join(CFG["out_dir"], f"feature_importance_{stamp}.csv")
    cfg_path = os.path.join(CFG["out_dir"], f"run_config_{stamp}.json")

    submission = test[[CFG["id_col"]]].copy()
    submission[CFG["target"]] = test_pred
    submission.to_csv(sub_path, index=False)

    oof_df = train[[CFG["id_col"]]].copy()
    oof_df["oof_pred"] = oof
    oof_df[CFG["target"]] = y
    oof_df.to_csv(oof_path, index=False)

    # "Importance" here aggregates per-fold average |coef| from inner estimators (linear reference only)
    fi.reset_index().rename(columns={"index": "feature"}).to_csv(fi_path, index=False)

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n"
          f"- Linear coefficient importance: {fi_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
