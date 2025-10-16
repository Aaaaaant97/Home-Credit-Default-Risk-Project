#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - RidgeClassifier + Platt Calibration Baseline (Main Table Only)
============================================================================
- Read application_train.csv / application_test.csv
- Preprocessing:
    * Categorical: LabelEncode-style mapping based on training only (unseen -> -1)
    * Numerical: median imputation using training statistics
    * Standardize numerical features with StandardScaler fitted on training only
- Outer 5-fold StratifiedKFold
- Model: RidgeClassifier (select best alpha from alpha_grid per fold)
         + CalibratedClassifierCV(method='sigmoid')
- Class imbalance handling: auto_weight / balanced / undersample / none
- Evaluation: ROC-AUC + PR-AUC (AP)
- Outputs: submission / OOF / (|coef|) feature_importance / run configuration
"""

import os, gc, json, time
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

# Progress bar (optional)
try:
    from tqdm import tqdm
    _use_tqdm = True
except Exception:
    _use_tqdm = False


# ========== CONFIG ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # "train_path": "application_train.csv",
    # "test_path":  "application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_ridge_calibrated_weight",

    # Ridge alpha grid (select best per fold)
    "alpha_grid": [0.1, 0.3, 1.0, 3.0, 10.0],

    # Calibration: 'sigmoid' (Platt) or 'isotonic'
    "calibration": "sigmoid",

    # Standardize numeric columns only (recommended True; if False, all encoded features are standardized)
    "scale_numeric_only": True,

    # ======== Class imbalance handling ========
    # method:
    #   - "auto_weight": instance-level weighting (pos weight = neg/pos, with cap) [default]
    #   - "balanced":    RidgeClassifier(class_weight='balanced')
    #   - "undersample": downsample negatives to desired neg:pos ratio
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


# ---------------- Preprocessing (fit strictly on training) ----------------
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

def _label_encode_with_train_map(series_train: pd.Series, series_test: pd.Series):
    """
    Build mapping based on training only: category -> int.
    Unseen categories in test are mapped to -1.
    """
    # Replace NaN with placeholder to include in mapping
    st = series_train.astype(str).replace("nan", np.nan).fillna("__MISSING__")
    se = series_test.astype(str).replace("nan", np.nan).fillna("__MISSING__")

    uniq = pd.Series(st.unique())
    mapping = {v: i for i, v in enumerate(uniq)}
    tr_enc = st.map(mapping).fillna(-1).astype(int)  # unseen -> -1
    te_enc = se.map(mapping).fillna(-1).astype(int)
    return tr_enc, te_enc, mapping


def preprocess(train, test, target, id_col, scale_numeric_only=True):
    """
    Basic preprocessing (leakage-safe):
      - Categorical: train-only mapping (LabelEncode style), unseen -> -1
      - Numerical: median imputation from training
      - Standardization: StandardScaler fitted on training only (numeric-only by default)
    """
    y = train[target].astype(int).values
    train_nolab = train.drop(columns=[target])

    # Column types
    cat_cols = train_nolab.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in train_nolab.columns if c not in cat_cols]

    # Numerical imputation: training median only
    for col in num_cols:
        if col == id_col:
            continue
        med = train_nolab[col].median() if pd.api.types.is_numeric_dtype(train_nolab[col]) else None
        if med is not None:
            train_nolab[col] = train_nolab[col].fillna(med)
            test[col] = test[col].fillna(med)

    # Categorical encoding (train-only), unseen in test -> -1
    for col in cat_cols:
        tr_enc, te_enc, _ = _label_encode_with_train_map(train_nolab[col], test[col])
        train_nolab[col] = tr_enc
        test[col] = te_enc

    # Standardization
    if scale_numeric_only:
        scaler = StandardScaler()
        to_scale = [c for c in num_cols if c != id_col]
        train_nolab[to_scale] = scaler.fit_transform(train_nolab[to_scale])
        test[to_scale] = scaler.transform(test[to_scale])
    else:
        scaler = StandardScaler()
        feats = [c for c in train_nolab.columns if c != id_col]
        train_nolab[feats] = scaler.fit_transform(train_nolab[feats])
        test[feats] = scaler.transform(test[feats])

    train_pre = train_nolab.reset_index(drop=True)
    test_pre  = test.reset_index(drop=True)

    features = [c for c in train_pre.columns if c != id_col]
    return train_pre, test_pre, y, features, cat_cols


# ---------------- Training / CV ----------------
def fit_calibrated_ridge(X_trn, y_trn, alpha, method="sigmoid", seed=42, sample_weight=None):
    """
    Fit RidgeClassifier on the training fold and calibrate probabilities via CalibratedClassifierCV.
    CalibratedClassifierCV(cv=5) performs an inner 5-fold on the training fold; sample_weight is passed if supported.
    """
    base = RidgeClassifier(alpha=alpha, random_state=seed, class_weight=None)
    clf = CalibratedClassifierCV(estimator=base, method=method, cv=5)
    try:
        clf.fit(X_trn, y_trn, sample_weight=sample_weight)  # sklearn >= 1.1
    except TypeError:
        clf.fit(X_trn, y_trn)
    return clf


def run_cv_ridge_calibrated(train_pre, y, test_pre, features, id_col, cfg):
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )

    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)

    # Linear-model "importance": mean |coef| over inner estimators, accumulated across folds
    coef_sum = np.zeros(len(features), dtype=float)

    X = train_pre[features].values
    X_test = test_pre[features].values

    imb = cfg.get("imbalance", {"method": "none"})
    method = imb.get("method", "none")
    rng = np.random.RandomState(cfg["seed"])

    print(f"Imbalance handling: {method}")
    print(f"Starting {cfg['n_splits']}-fold CV ...")
    iterator = folds.split(X, y)
    if _use_tqdm:
        iterator = tqdm(list(iterator), total=cfg['n_splits'])

    for fold_id, (trn_idx, val_idx) in enumerate(iterator, 1):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # ===== Imbalance handling (applied before selecting alpha) =====
        sample_weight = None
        class_weight_for_grid = None  # used only when method == "balanced"
        X_grid, y_grid = X_trn, y_trn

        if method == "undersample":
            mask = _undersample_indices(
                y_trn,
                desired_neg_pos_ratio=float(imb.get("undersample_neg_pos_ratio", 3.0)),
                rng=rng
            )
            X_grid = X_trn[mask]
            y_grid = y_trn[mask]
            class_weight_for_grid = None
            sample_weight = None
            print(f"[Fold {fold_id}] After undersampling: n={X_grid.shape[0]} (pos={np.sum(y_grid==1)}, neg={np.sum(y_grid==0)})")

        elif method == "auto_weight":
            pos_w = _compute_pos_weight(y_trn, cap=float(imb.get("max_pos_weight_cap", 50.0)))
            sample_weight = np.where(y_trn == 1, pos_w, 1.0).astype(float)
            class_weight_for_grid = None
            print(f"[Fold {fold_id}] auto_weight: pos_weight≈{pos_w:.2f}")

        elif method == "balanced":
            class_weight_for_grid = "balanced"
            sample_weight = None
            print(f"[Fold {fold_id}] Using class_weight='balanced'")

        else:
            print(f"[Fold {fold_id}] No imbalance handling")

        # ===== Select best alpha on this fold =====
        best_alpha, best_auc = None, -1.0
        for a in cfg["alpha_grid"]:
            base = RidgeClassifier(alpha=a, random_state=cfg["seed"], class_weight=class_weight_for_grid)
            tmp = CalibratedClassifierCV(estimator=base, method=cfg["calibration"], cv=5)
            try:
                tmp.fit(X_grid, y_grid, sample_weight=(sample_weight if method == "auto_weight" else None))
            except TypeError:
                tmp.fit(X_grid, y_grid)
            val_prob = tmp.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_prob)
            if auc > best_auc:
                best_auc, best_alpha = auc, a
            del tmp

        # ===== Retrain on the fold with best alpha and chosen strategy =====
        if method == "undersample":
            X_trn_use, y_trn_use = X_grid, y_grid
            sample_weight_use = None
            class_weight_use = None
        elif method == "balanced":
            X_trn_use, y_trn_use = X_trn, y_trn
            sample_weight_use = None
            class_weight_use = "balanced"
        elif method == "auto_weight":
            X_trn_use, y_trn_use = X_trn, y_trn
            sample_weight_use = sample_weight
            class_weight_use = None
        else:
            X_trn_use, y_trn_use = X_trn, y_trn
            sample_weight_use = None
            class_weight_use = None

        base = RidgeClassifier(alpha=best_alpha, random_state=cfg["seed"], class_weight=class_weight_use)
        clf = CalibratedClassifierCV(estimator=base, method=cfg["calibration"], cv=5)
        try:
            clf.fit(X_trn_use, y_trn_use, sample_weight=sample_weight_use)
        except TypeError:
            clf.fit(X_trn_use, y_trn_use)

        # Validation predictions
        val_prob = clf.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_prob
        val_auc = roc_auc_score(y_val, val_prob)
        if cfg.get("report_pr_auc", True):
            val_ap = average_precision_score(y_val, val_prob)
            print(f"[Fold {fold_id}] AUC = {val_auc:.6f} | AP = {val_ap:.6f}  (best alpha = {best_alpha})")
        else:
            print(f"[Fold {fold_id}] AUC = {val_auc:.6f}  (best alpha = {best_alpha})")

        # Test predictions
        test_pred += clf.predict_proba(X_test)[:, 1] / cfg["n_splits"]

        # Aggregate linear coefficients (mean of inner estimator coef_)
        coef_stack = []
        for est in clf.calibrated_classifiers_:
            if hasattr(est.estimator, "coef_"):
                coef_stack.append(est.estimator.coef_.ravel())
        if len(coef_stack) > 0:
            mean_coef = np.mean(np.vstack(coef_stack), axis=0)
            coef_sum += np.abs(mean_coef)

        del clf, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    if cfg.get("report_pr_auc", True):
        oof_ap = average_precision_score(y, oof)
        print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} | OOF AP = {oof_ap:.6f} ===")
    else:
        print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")

    fi = pd.DataFrame({"feature": features, "coef_abs": coef_sum / cfg["n_splits"]}) \
           .sort_values("coef_abs", ascending=False) \
           .reset_index(drop=True)

    return oof, test_pred, oof_auc, fi


def main():
    t0 = time.time()
    ensure_dir(CFG["out_dir"])

    print("Reading data ...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (categorical encoding, imputation, standardization; all fitted on training only) ...")
    train_pre, test_pre, y, features, cat_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"], CFG["scale_numeric_only"]
    )
    print(f"Number of features: {len(features)} (categorical columns: {len(cat_cols)})")

    print("Cross-validation + Ridge (Calibrated) training ...")
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

    fi.to_csv(fi_path, index=False)

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n- Linear-coefficient importance: {fi_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
