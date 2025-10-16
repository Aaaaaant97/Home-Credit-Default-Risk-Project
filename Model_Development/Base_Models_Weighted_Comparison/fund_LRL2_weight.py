#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - Logistic Regression (L2) Baseline (Main Table Only)
=================================================================
- Read application_train.csv / application_test.csv
- Preprocessing: One-Hot for categorical (version compatible) + numerical median imputation + numerical standardization
- 5-fold StratifiedKFold training of LogisticRegression (L2)
- Class imbalance handling: auto_weight / class_weight=balanced / undersample / none
- Evaluation: ROC-AUC + PR-AUC (Average Precision)
- Outputs: submission / OOF / coefficient importance / run configuration
"""

import os, gc, json, time, warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

# ========== CONFIG ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # "train_path": "fli_application_train.csv",
    # "test_path":  "fli_application_test.csv",
    # "train_path": "application_train.csv",
    # "test_path":  "application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_logreg_l2_weight",

    # Logistic Regression parameters (tweakable / grid-searchable if needed)
    "lr_params": {
        "penalty": "l2",
        "solver": "saga",       # Stable for sparse & high-dimensional feature spaces
        "C": 1.0,               # Regularization strength (example grid: 0.2, 0.5, 1, 2, 5)
        "max_iter": 2000,
        "tol": 1e-4,
        "n_jobs": -1,
        "class_weight": None    # Or "balanced"; keep None if using auto_weight below
    },

    # ======== Class imbalance strategies (aligned with XGB version) ========
    # method:
    #   - "auto_weight": instance-level weighting (pos weight = neg/pos, with cap) [default]
    #   - "balanced":    use LR's class_weight="balanced" (choose either this or auto_weight)
    #   - "undersample": downsample negatives to the specified neg:pos ratio
    #   - "none":        no handling
    "imbalance": {
        "method": "auto_weight",
        "undersample_neg_pos_ratio": 3.0,
        "max_pos_weight_cap": 50.0
    },

    # Whether to report PR-AUC (AP)
    "report_pr_auc": True,
}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_data(train_path, test_path):
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    test = pd.read_csv(test_path)

    if os.path.isfile(train_path):
        train = pd.read_csv(train_path)
    else:
        raise FileNotFoundError(
            f"Training file not found: {train_path} (application_train.csv required)"
        )
    return train, test


def _make_ohe():
    """
    Compatible across sklearn versions:
    - Newer: OneHotEncoder(..., sparse_output=True)
    - Older: OneHotEncoder(..., sparse=True)
    """
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def preprocess(train: pd.DataFrame, test: pd.DataFrame, target: str, id_col: str):
    """
    Basic preprocessing:
    - Numerical: median imputation, StandardScaler (with_mean=False for CSR compatibility)
    - Categorical: missing -> '__MISSING__', One-Hot (fit on train+test to avoid misalignment)
    - Returns: sparse design matrices X_train, X_test, y, and feature name list
    """
    y = train[target].astype(int).values
    train = train.drop(columns=[target])

    # Concatenate to ensure consistent encoding
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # Column types
    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols]

    # Missing values
    if cat_cols:
        combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")
    for col in num_cols:
        if col == id_col:
            continue
        if combined[col].isna().any():
            med = combined[col].median()
            combined[col] = combined[col].fillna(med)

    # Numerical standardization (CSR-friendly with_mean=False)
    scaler = StandardScaler(with_mean=False)
    X_num = csr_matrix(scaler.fit_transform(combined[num_cols].astype(float)))

    # Categorical One-Hot (version compatible)
    if cat_cols:
        ohe = _make_ohe()
        X_cat = ohe.fit_transform(combined[cat_cols].astype(str))
        X_all = hstack([X_num, X_cat], format="csr")
        feature_names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))
    else:
        X_all = X_num
        feature_names = list(num_cols)

    # Split back to train/test
    n_train = len(train)
    X_train = X_all[:n_train]
    X_test  = X_all[n_train:]

    return X_train, X_test, y, feature_names


def _compute_pos_weight(y_arr, cap=50.0):
    pos = int(np.sum(y_arr == 1))
    neg = int(np.sum(y_arr == 0))
    if pos == 0:
        return 1.0
    w = neg / max(1, pos)
    if cap is not None:
        w = min(w, float(cap))
    return float(w)


def _undersample_indices(y_trn, desired_neg_pos_ratio=3.0, rng=None):
    """Return boolean mask for undersampling."""
    if rng is None:
        rng = np.random.RandomState(42)
    pos_idx = np.where(y_trn == 1)[0]
    neg_idx = np.where(y_trn == 0)[0]
    n_pos = len(pos_idx)
    n_neg_keep = int(min(len(neg_idx), desired_neg_pos_ratio * n_pos))
    if n_neg_keep <= 0:
        return np.ones_like(y_trn, dtype=bool)
    keep_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False)
    keep = np.zeros_like(y_trn, dtype=bool)
    keep[pos_idx] = True
    keep[keep_neg] = True
    return keep


def run_cv_logreg(X, y, X_test, feature_names, cfg):
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )

    oof = np.zeros(X.shape[0], dtype=float)
    test_pred = np.zeros(X_test.shape[0], dtype=float)
    coef_importance = np.zeros(len(feature_names), dtype=float)

    imb = cfg.get("imbalance", {"method": "none"})
    method = imb.get("method", "none")
    print(f"Imbalance handling: {method}")

    rng = np.random.RandomState(cfg["seed"])
    print(f"Starting {cfg['n_splits']}-fold cross-validation ...")

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # ====== Imbalance handling ======
        sample_weight = None
        lr_params_fold = dict(cfg["lr_params"])

        if method == "undersample":
            mask = _undersample_indices(
                y_trn,
                desired_neg_pos_ratio=float(imb.get("undersample_neg_pos_ratio", 3.0)),
                rng=rng
            )
            X_trn = X_trn[mask]
            y_trn = y_trn[mask]
            lr_params_fold["class_weight"] = None  # avoid mixing with other weighting
            print(f"[Fold {fold}] After undersampling: n={X_trn.shape[0]} (pos={np.sum(y_trn==1)}, neg={np.sum(y_trn==0)})")

        elif method == "auto_weight":
            # Instance-level weights: pos = neg/pos (capped)
            pos_w = _compute_pos_weight(y_trn, cap=float(imb.get("max_pos_weight_cap", 50.0)))
            sample_weight = np.where(y_trn == 1, pos_w, 1.0).astype(float)
            lr_params_fold["class_weight"] = None
            print(f"[Fold {fold}] auto_weight: pos_weight≈{pos_w:.2f}")

        elif method == "balanced":
            lr_params_fold["class_weight"] = "balanced"
            print(f"[Fold {fold}] Using class_weight='balanced'")

        else:
            print(f"[Fold {fold}] No imbalance handling")

        # ====== Training ======
        model = LogisticRegression(random_state=cfg["seed"], **lr_params_fold)
        model.fit(X_trn, y_trn, sample_weight=sample_weight)

        # Validation predictions (probabilities)
        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred

        val_auc = roc_auc_score(y_val, val_pred)
        if cfg.get("report_pr_auc", True):
            val_ap = average_precision_score(y_val, val_pred)
            print(f"[Fold {fold}] AUC = {val_auc:.6f} | AP = {val_ap:.6f}")
        else:
            print(f"[Fold {fold}] AUC = {val_auc:.6f}")

        # Test predictions (averaged over folds)
        test_pred += model.predict_proba(X_test)[:, 1] / cfg["n_splits"]

        # Coefficient importance (absolute value, averaged later)
        coef_importance += np.abs(model.coef_.ravel())

        del model, X_trn, X_val, y_trn, y_val, sample_weight
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    if cfg.get("report_pr_auc", True):
        oof_ap = average_precision_score(y, oof)
        print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} | OOF AP = {oof_ap:.6f} ===")
    else:
        print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")

    coef_importance /= cfg["n_splits"]
    fi = (
        pd.DataFrame({"feature": feature_names, "coef_abs": coef_importance})
        .sort_values("coef_abs", ascending=False)
        .reset_index(drop=True)
    )

    return oof, test_pred, oof_auc, fi


def main():
    t0 = time.time()
    ensure_dir(CFG["out_dir"])

    print("Reading data ...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (One-Hot + imputation + standardization) ...")
    X_train, X_test, y, feature_names = preprocess(
        train, test, CFG["target"], CFG["id_col"]
    )
    print(f"Number of features: {len(feature_names)}")

    # Label distribution for sanity check
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    print(f"Label distribution: neg={neg}, pos={pos}, neg:pos≈{(neg/max(1,pos)):.2f}:1")

    print("Cross-validation + training Logistic Regression (L2) ...")
    oof, test_pred, oof_auc, fi = run_cv_logreg(
        X_train, y, X_test, feature_names, CFG
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

    fi.to_csv(fi_path, index=False)

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n- Coefficient importance: {fi_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
