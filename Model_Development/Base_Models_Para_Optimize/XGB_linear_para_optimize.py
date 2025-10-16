#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost (gblinear) baseline + small grid search (reg_alpha × reg_lambda × scale_pos_weight)
- Preprocessing: categorical LabelEncode + numeric median imputation
- Keeps folds/rounds/early-stopping consistent with your LGB script
- Small grid by default (8 combos) for time-friendly runs
- Feature importance: prefer linear coefficients (coef_); otherwise fall back to permutation importance (scoring='roc_auc')
- Exports: submission / OOF / feature_importance / run config / best params
"""

import os, gc, json, time, warnings
from datetime import datetime
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

import xgboost as xgb

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
    # "train_path": "application_train.csv",
    # "test_path":  "application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_xgblinear_grid",

    # ========== Base params for XGBoost (gblinear) ==========
    "xgb_params": {
        "booster": "gblinear",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        # eta has little effect for gblinear; regularization + optimizer matter more,
        # but it's harmless to keep:
        "learning_rate": 0.1,
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": 42,
        # Regularization (overridden by grid):
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        # Class imbalance weight (grid or 'auto'):
        "scale_pos_weight": 1.0,
    },

    # Rounds & early stopping (aligned with LGB style)
    "num_boost_round": 10000,
    "early_stopping_rounds": 200,

    # ====== Small grid (time-friendly) ======
    # Tweak as needed, e.g., alpha: [0, 1], lambda: [0, 5], spw: [1.0, "auto"]
    "grid_alpha":   [0.0, 1.0],
    "grid_lambda":  [0.0, 5.0],
    "grid_spw":     [1.0, "auto"],   # "auto" -> n_neg / n_pos
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
        raise FileNotFoundError(f"Training file not found: {train_path}")
    return train, test


def preprocess(train, test, target, id_col):
    """Basic preprocessing: label-encode categoricals + median-impute numerics (same style as LGB)."""
    y = train[target].astype(int).values
    train = train.drop(columns=[target])

    combined = pd.concat([train, test], axis=0, ignore_index=True)

    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols]

    combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")
    for col in num_cols:
        if col == id_col:
            continue
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())

    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    train_pre = combined.iloc[:len(train)]
    test_pre  = combined.iloc[len(train):].reset_index(drop=True)
    features = [c for c in train_pre.columns if c != id_col]
    return train_pre, test_pre, y, features, cat_cols


def _auto_scale_pos_weight(y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def _coef_importance_or_none(model, features):
    """
    Prefer extracting linear coefficients from XGBClassifier(booster='gblinear').
    If unavailable, return None so we can fall back to permutation importance.
    """
    try:
        coefs = getattr(model, "coef_", None)
        if coefs is None:
            return None
        coefs = np.ravel(coefs)
        if len(coefs) != len(features):
            return None
        fi = pd.DataFrame({
            "feature": features,
            "importance": np.abs(coefs),
            "coef": coefs
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return fi
    except Exception:
        return None


def _permutation_importance_safe(model, X_val, y_val, features, seed=42):
    """Lightweight PI: n_repeats=3, scoring='roc_auc'. On failure, return zeros and never raise."""
    try:
        # For speed, optionally subsample large validation sets (e.g., max 20k)
        if len(X_val) > 20000:
            rs = np.random.RandomState(seed)
            idx = rs.choice(len(X_val), size=20000, replace=False)
            X_val_s = X_val.iloc[idx]
            y_val_s = y_val[idx]
        else:
            X_val_s = X_val
            y_val_s = y_val

        pi = permutation_importance(
            model, X_val_s, y_val_s,
            n_repeats=3,
            scoring="roc_auc",
            random_state=seed,
            n_jobs=-1
        )
        fi = pd.DataFrame({
            "feature": features,
            "importance": pi.importances_mean
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return fi
    except Exception as e:
        warnings.warn(f"Permutation importance failed (skipped): {e}")
        return pd.DataFrame({"feature": features, "importance": 0.0})


def _predict_proba_best(model, X):
    """
    Predict using the best iteration across xgboost versions:
    - Prefer iteration_range with best_iteration_ if available
    - Fall back to ntree_limit if needed
    """
    try:
        # xgboost >= 1.6
        return model.predict_proba(X, iteration_range=(0, model.best_iteration_ + 1))[:, 1]
    except Exception:
        # Older versions
        ntree_limit = getattr(model, "best_ntree_limit", None)
        if ntree_limit is None:
            return model.predict_proba(X)[:, 1]
        return model.predict_proba(X, ntree_limit=ntree_limit)[:, 1]


def run_cv_xgb_linear(train_pre, y, test_pre, features, id_col, cfg):
    """5-fold CV with XGBClassifier(booster='gblinear') + early stopping.
    Returns OOF, test predictions, OOF AUC, and aggregated feature importance.
    """
    folds = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])

    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)
    importance_sum = np.zeros(len(features), dtype=float)

    X = train_pre[features]
    X_test = test_pre[features]

    print(f"Starting {cfg['n_splits']}-fold CV ...")
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        params = deepcopy(cfg["xgb_params"])
        clf = xgb.XGBClassifier(**params, n_estimators=cfg["num_boost_round"])
        clf.fit(
            X_trn, y_trn,
            eval_set=[(X_trn, y_trn), (X_val, y_val)],
            eval_metric="auc",
            verbose=False,
            early_stopping_rounds=cfg["early_stopping_rounds"]
        )

        # Validation predictions
        val_pred = _predict_proba_best(clf, X_val)
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        best_iter = getattr(clf, "best_iteration_", None)
        print(f"[Fold {fold}] AUC = {val_auc:.6f} best_iter = {best_iter}")

        # Test predictions (average over folds)
        test_pred += _predict_proba_best(clf, X_test) / cfg["n_splits"]

        # Feature importance this fold: prefer coef_, else PI
        fi_fold = _coef_importance_or_none(clf, features)
        if fi_fold is None:
            fi_fold = _permutation_importance_safe(clf, X_val, y_val, features, seed=cfg["seed"])

        # Align by feature order and sum
        imp_map = dict(zip(fi_fold["feature"], fi_fold["importance"]))
        importance_sum += np.array([imp_map.get(f, 0.0) for f in features], dtype=float)

        del clf, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")
    fi_sorted = pd.DataFrame({"feature": features, "importance": importance_sum}) \
                   .sort_values("importance", ascending=False).reset_index(drop=True)
    return oof, test_pred, oof_auc, fi_sorted


def grid_search_xgblinear(train_pre, y, test_pre, features, id_col, base_cfg,
                          alpha_list, lambda_list, spw_list, save_csv_path=None):
    """Grid over reg_alpha × reg_lambda × scale_pos_weight (small, time-friendly)."""
    auto_spw = _auto_scale_pos_weight(y)

    pairs = list(product(alpha_list, lambda_list, spw_list))
    records = []

    iterator = tqdm(pairs, desc="Grid search (alpha × lambda × spw)") if _use_tqdm else pairs

    best_auc = -1.0
    best_pack = None

    for a, lmbd, spw in iterator:
        cfg = deepcopy(base_cfg)
        cfg["xgb_params"].update({
            "reg_alpha": a,
            "reg_lambda": lmbd,
            "scale_pos_weight": (auto_spw if (isinstance(spw, str) and spw == "auto") else float(spw)),
        })

        print(f"\n[Try] reg_alpha={a}, reg_lambda={lmbd}, scale_pos_weight={cfg['xgb_params']['scale_pos_weight']:.4f}")
        oof, test_pred, oof_auc, fi = run_cv_xgb_linear(
            train_pre, y, test_pre, features, id_col, cfg
        )

        records.append({
            "reg_alpha": a,
            "reg_lambda": lmbd,
            "scale_pos_weight": cfg["xgb_params"]["scale_pos_weight"],
            "oof_auc": oof_auc
        })

        if _use_tqdm:
            iterator.set_postfix({"alpha": a, "lambda": lmbd, "spw": cfg["xgb_params"]["scale_pos_weight"], "auc": f"{oof_auc:.6f}"})

        if oof_auc > best_auc:
            best_auc = oof_auc
            best_pack = {
                "reg_alpha": a,
                "reg_lambda": lmbd,
                "scale_pos_weight": cfg["xgb_params"]["scale_pos_weight"],
                "oof_auc": oof_auc,
                "detail": (oof, test_pred, fi, cfg)
            }
        gc.collect()

    tried_df = pd.DataFrame(records).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    if save_csv_path is not None:
        tried_df.to_csv(save_csv_path, index=False)

    return best_pack, tried_df


def main():
    t0 = time.time()
    ensure_dir(CFG["out_dir"])

    print("Reading data ...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (label encoding, median imputation) ...")
    train_pre, test_pre, y, features, cat_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"]
    )
    print(f"Number of features: {len(features)} (categorical: {len(cat_cols)})")

    # ====== Small grid search (faster) ======
    grid_csv = os.path.join(CFG["out_dir"], f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    best_pack, tried_df = grid_search_xgblinear(
        train_pre, y, test_pre, features, CFG["id_col"], CFG,
        alpha_list=CFG["grid_alpha"],
        lambda_list=CFG["grid_lambda"],
        spw_list=CFG["grid_spw"],
        save_csv_path=grid_csv
    )

    print("\n[GridSearch] Top-10 (sorted by OOF AUC):")
    print(tried_df.head(10).to_string(index=False))

    print(f"\n[Best] reg_alpha={best_pack['reg_alpha']}, "
          f"reg_lambda={best_pack['reg_lambda']}, "
          f"scale_pos_weight={best_pack['scale_pos_weight']}, "
          f"OOF AUC={best_pack['oof_auc']:.6f}")

    # ====== Retrain with best params and save outputs ======
    # Retrieve the best config (includes other xgb_params)
    oof_g, test_pred_g, fi_g, best_cfg = best_pack["detail"]  # (oof, test_pred, fi, cfg)

    print("\n[Retrain] Retraining full CV with best parameters and exporting files ...")
    oof, test_pred, oof_auc, fi = run_cv_xgb_linear(
        train_pre, y, test_pre, features, CFG["id_col"], best_cfg
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path  = os.path.join(CFG["out_dir"], f"submission_{stamp}.csv")
    oof_path  = os.path.join(CFG["out_dir"], f"oof_{stamp}.csv")
    fi_path   = os.path.join(CFG["out_dir"], f"feature_importance_{stamp}.csv")
    cfg_path  = os.path.join(CFG["out_dir"], f"run_config_{stamp}.json")
    best_path = os.path.join(CFG["out_dir"], f"best_params_{stamp}.json")

    submission = test[[CFG["id_col"]]].copy()
    submission[CFG["target"]] = test_pred
    submission.to_csv(sub_path, index=False)

    oof_df = train[[CFG["id_col"]]].copy()
    oof_df["oof_pred"] = oof
    oof_df[CFG["target"]] = y
    oof_df.to_csv(oof_path, index=False)

    fi.to_csv(fi_path, index=False)

    # Save run config + best parameters
    out_cfg = deepcopy(best_cfg)
    out_cfg["best_search"] = {
        "reg_alpha": best_pack["reg_alpha"],
        "reg_lambda": best_pack["reg_lambda"],
        "scale_pos_weight": best_pack["scale_pos_weight"],
        "oof_auc": best_pack["oof_auc"]
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print(f"\nSaved files:\n- Grid results: {grid_csv}\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n- Feature importance: {fi_path}\n- Best parameters: {best_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
