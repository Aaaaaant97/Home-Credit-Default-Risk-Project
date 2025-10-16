#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - RidgeClassifier + Calibration (Grid Search)
=========================================================
- Baseline on main tables (application_train/test)
- Preprocessing: LabelEncode + numeric median imputation + standardization
- Model: RidgeClassifier + CalibratedClassifierCV
- Grid search: alpha × calibration_method (sigmoid/isotonic)
- Outer 5-fold StratifiedKFold for OOF; inner CV handled by CalibratedClassifierCV
- Exports: submission / OOF / linear-coefficient "importance" (|coef|) / run & best configs / grid table
"""

import os, gc, json, time
from datetime import datetime
from copy import deepcopy
from itertools import product

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


# ========== CONFIG ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # "train_path": "fli_application_train.csv",
    # "test_path":  "fli_application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_ridge_calibrated_grid",

    # Preprocessing
    "scale_numeric_only": True,  # standardization is recommended for linear models

    # Grid ranges
    "grid_alpha": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    "grid_calib": ["sigmoid", "isotonic"],  # Platt / Isotonic

    # Inner CV folds for CalibratedClassifierCV
    "inner_cv": 5
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
        raise FileNotFoundError(f"Training file not found: {train_path} (expected application_train.csv)")
    return train, test


def preprocess(train, test, target, id_col, scale_numeric_only=True):
    """Basic preprocessing: LabelEncode + missing-value imputation + standardization."""
    y = train[target].astype(int).values
    train = train.drop(columns=[target])

    combined = pd.concat([train, test], axis=0, ignore_index=True)

    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols]

    # Missing values: categorical -> "__MISSING__", numeric -> median
    combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")
    for col in num_cols:
        if col == id_col:
            continue
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())

    # Label encoding (fit on train+test together)
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Standardization
    scaler = StandardScaler()
    if scale_numeric_only:
        to_scale = [c for c in num_cols if c != id_col]
        combined[to_scale] = scaler.fit_transform(combined[to_scale])
    else:
        feats = [c for c in combined.columns if c != id_col]
        combined[feats] = scaler.fit_transform(combined[feats])

    train_pre = combined.iloc[:len(train)]
    test_pre  = combined.iloc[len(train):].reset_index(drop=True)

    features = [c for c in train_pre.columns if c != id_col]
    return train_pre, test_pre, y, features, cat_cols


def fit_calibrated_ridge(X_trn, y_trn, alpha, method="sigmoid", inner_cv=5):
    """Fit on the training fold: RidgeClassifier -> CalibratedClassifierCV (with inner CV)."""
    base = RidgeClassifier(alpha=alpha)
    clf = CalibratedClassifierCV(estimator=base, method=method, cv=inner_cv)
    clf.fit(X_trn, y_trn)
    return clf


def _extract_mean_coef_from_calibrator(clf):
    """
    Extract average linear coefficients from the calibrated model across inner folds.
    Handles scikit-learn variants where attributes may be named 'estimator' or 'base_estimator'.
    """
    coefs = []
    if hasattr(clf, "calibrated_classifiers_"):
        for est in clf.calibrated_classifiers_:
            base = None
            # Try common attribute names
            if hasattr(est, "estimator"):
                base = getattr(est, "estimator")
            elif hasattr(est, "base_estimator"):
                base = getattr(est, "base_estimator")
            if base is not None and hasattr(base, "coef_"):
                coefs.append(base.coef_.ravel())
    if len(coefs) == 0:
        return None
    return np.mean(np.vstack(coefs), axis=0)


def run_cv_with_params(train_pre, y, test_pre, features, params, cfg):
    """
    Run one outer CV pass for given (alpha, calibration_method).
    Returns oof, test_pred, oof_auc, coef_importance (|coef| aggregated across folds).
    """
    folds = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)
    coef_importance = pd.DataFrame(0.0, index=features, columns=["coef_abs_sum"])

    X = train_pre[features].values
    X_test = test_pre[features].values

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = fit_calibrated_ridge(
            X_trn, y_trn,
            alpha=params["alpha"],
            method=params["calibration"],
            inner_cv=cfg["inner_cv"]
        )

        val_prob = clf.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_prob

        test_prob = clf.predict_proba(X_test)[:, 1]
        test_pred += test_prob / cfg["n_splits"]

        # Coefficients: average inner base estimators' coef_
        mean_coef = _extract_mean_coef_from_calibrator(clf)
        if mean_coef is not None:
            coef_importance["coef_abs_sum"] += np.abs(mean_coef)

        del clf, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    coef_importance = coef_importance.sort_values("coef_abs_sum", ascending=False)
    return oof, test_pred, oof_auc, coef_importance


def grid_search_ridge(train_pre, y, test_pre, features, cfg, save_csv_path=None):
    """
    Grid search over alpha × calibration_method; returns the best setup and full results table.
    """
    pairs = list(product(cfg["grid_alpha"], cfg["grid_calib"]))
    records = []
    best_auc, best_detail, best_params = -1.0, None, None

    iterator = tqdm(pairs, desc="Grid search (alpha × calibration)") if _use_tqdm else pairs

    for alpha, calib in iterator:
        params = {"alpha": alpha, "calibration": calib}
        print(f"\n[Try] alpha={alpha}, calibration={calib}")
        oof, test_pred, oof_auc, fi = run_cv_with_params(
            train_pre, y, test_pre, features, params, cfg
        )

        records.append({"alpha": alpha, "calibration": calib, "oof_auc": oof_auc})
        if _use_tqdm:
            iterator.set_postfix({"alpha": alpha, "calib": calib, "auc": f"{oof_auc:.6f}"})

        if oof_auc > best_auc:
            best_auc = oof_auc
            best_params = params
            best_detail = (oof, test_pred, fi, params)

        gc.collect()

    tried_df = pd.DataFrame(records).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    if save_csv_path is not None:
        tried_df.to_csv(save_csv_path, index=False)

    return {"best_auc": best_auc, "best_params": best_params, "detail": best_detail}, tried_df


def main():
    t0 = time.time()
    ensure_dir(CFG["out_dir"])

    print("Reading data ...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (label encoding, missing value imputation, standardization) ...")
    train_pre, test_pre, y, features, cat_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"], CFG["scale_numeric_only"]
    )
    print(f"Number of features: {len(features)} (categorical: {len(cat_cols)})")

    # ====== Grid search ======
    grid_csv = os.path.join(CFG["out_dir"], f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    best_pack, tried_df = grid_search_ridge(train_pre, y, test_pre, features, CFG, save_csv_path=grid_csv)

    print("\n[GridSearch] Top-10 (sorted by OOF AUC):")
    print(tried_df.head(10).to_string(index=False))

    print(f"\n[Best] alpha={best_pack['best_params']['alpha']}, "
          f"calibration={best_pack['best_params']['calibration']}, "
          f"OOF AUC={best_pack['best_auc']:.6f}")

    # ====== Retrain with best params and save outputs ======
    oof_g, test_pred_g, fi_g, best_params = best_pack["detail"]  # (oof, test_pred, fi, params)

    print("\n[Retrain] Retraining full CV with best parameters and exporting files ...")
    oof, test_pred, oof_auc, fi = run_cv_with_params(
        train_pre, y, test_pre, features, best_params, CFG
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path  = os.path.join(CFG["out_dir"], f"submission_{stamp}.csv")
    oof_path  = os.path.join(CFG["out_dir"], f"oof_{stamp}.csv")
    fi_path   = os.path.join(CFG["out_dir"], f"feature_importance_{stamp}.csv")
    cfg_path  = os.path.join(CFG["out_dir"], f"run_config_{stamp}.json")
    best_path = os.path.join(CFG["out_dir"], f"best_params_{stamp}.json")

    # Write submission
    submission = test[[CFG["id_col"]]].copy()
    submission[CFG["target"]] = test_pred
    submission.to_csv(sub_path, index=False)

    # Write OOF
    oof_df = train[[CFG["id_col"]]].copy()
    oof_df["oof_pred"] = oof
    oof_df[CFG["target"]] = y
    oof_df.to_csv(oof_path, index=False)

    # Write "importance" (linear coefficients)
    fi.reset_index().rename(columns={"index": "feature"}).to_csv(fi_path, index=False)

    # Save run config + best parameters
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)
    out_cfg = {
        "best_params": best_params,
        "best_oof_auc": float(oof_auc),
        "grid": {
            "alpha": CFG["grid_alpha"],
            "calibration": CFG["grid_calib"],
            "inner_cv": CFG["inner_cv"]
        }
    }
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print(f"\nSaved files:\n- Grid results: {grid_csv}\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n- Linear-coefficient importance: {fi_path}\n- Best parameters: {best_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
