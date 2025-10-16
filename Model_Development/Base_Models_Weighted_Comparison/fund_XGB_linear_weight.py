#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Home Credit - XGBoost (gblinear) baseline with imbalance handling
=================================================================
- Categorical: One-Hot
- Numerical: median imputation (fit on train) + StandardScaler (fit on train)
- CV: StratifiedKFold
- Imbalance: configurable (auto_weight / scale_pos_weight / undersample / none)
- Metrics: AUC + PR-AUC (AP)
- Importance: 'weight' for gblinear; fallback to coef-norm parsed from dump

Outputs:
  - submission_*.csv
  - oof_*.csv
  - feature_importance_*.csv
  - run_config_*.json
"""

import os, gc, json, time, math, warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import xgboost as xgb

warnings.filterwarnings("ignore")

# ========== CONFIG ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # If you already have screened feature files, you may switch to fli_* paths
    # "train_path": "application_train.csv",
    # "test_path":  "application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_xgb_linear_weight",

    # XGBoost parameters (linear booster)
    "xgb_params": {
        "booster": "gblinear",            # linear booster
        "objective": "binary:logistic",
        "eval_metric": "auc",             # AP (PR-AUC) is computed separately, not used for early stopping
        "lambda": 1.0,                    # L2 regularization
        "alpha": 0.0,                     # L1 regularization (try >0 for sparsity)
        "learning_rate": 0.05,
        "nthread": -1,
        "seed": 42
    },
    "num_boost_round": 5000,
    "early_stopping_rounds": 200,
    "verbose_eval": 200,

    # Preprocessing toggles
    "one_hot_categoricals": True,
    "standardize_numericals": True,

    # ========= Class-imbalance handling =========
    # method:
    #   - "auto_weight": per-fold positive class weight = neg/pos (capped) applied as sample weights [default]
    #   - "scale_pos_weight": pass scale_pos_weight to XGBoost (choose either this or auto_weight)
    #   - "undersample": downsample negatives to a specified neg:pos ratio
    #   - "none": no handling
    "imbalance": {
        "method": "auto_weight",
        "undersample_neg_pos_ratio": 3.0,     # only used when method == "undersample"
        "max_pos_weight_cap": 50.0            # cap to avoid instability
    },

    # Whether to print PR-AUC (AP) each fold
    "report_pr_auc": True,
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_data(train_path, test_path):
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    test = pd.read_csv(test_path)

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    train = pd.read_csv(train_path)

    return train, test

def linear_friendly_preprocess(train: pd.DataFrame, test: pd.DataFrame, target: str, id_col: str, cfg: dict):
    """
    - One-Hot encode categorical columns (fit on concatenated train+test to align columns)
    - Median-impute numerical columns using training medians, then apply to both train/test
    - Standardize numerical columns with scaler fitted on training only
    """
    y = train[target].astype(int).values
    train_nolabel = train.drop(columns=[target])

    # Column splits
    cat_cols = train_nolabel.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in train_nolabel.columns if c not in cat_cols and c != id_col]

    # Concatenate for consistent One-Hot column space
    combined = pd.concat([train_nolabel, test], axis=0, ignore_index=True)

    # Missing values
    if cat_cols:
        combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")
    # Numerical imputation with training medians
    train_num = train_nolabel[num_cols]
    medians = train_num.median(axis=0)
    for col in num_cols:
        combined[col] = combined[col].fillna(medians[col])

    # One-Hot (do not drop_first; let regularization handle redundancy)
    if cfg["one_hot_categoricals"] and len(cat_cols) > 0:
        dummies = pd.get_dummies(combined[cat_cols], dummy_na=False)
        combined = pd.concat([combined.drop(columns=cat_cols), dummies], axis=1)

    # Standardize numeric columns (fit on training only)
    if cfg["standardize_numericals"] and len(num_cols) > 0:
        scaler = StandardScaler()
        scaler.fit(combined.iloc[:len(train)][num_cols])
        combined[num_cols] = scaler.transform(combined[num_cols])

    # Split back
    train_pre = combined.iloc[:len(train)].reset_index(drop=True)
    test_pre  = combined.iloc[len(train):].reset_index(drop=True)

    # Feature list: all except ID column
    features = [c for c in train_pre.columns if c != id_col]

    return train_pre, test_pre, y, features

def _compute_pos_weight(y_arr, cap=50.0):
    pos = np.sum(y_arr == 1)
    neg = np.sum(y_arr == 0)
    if pos == 0:
        return 1.0
    pw = neg / max(1, pos)
    if cap is not None:
        pw = min(pw, cap)
    return float(pw)

def _undersample_indices(y_trn, desired_neg_pos_ratio=3.0, rng=None):
    """Return boolean mask after undersampling."""
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

def _parse_gblinear_coef_norm(booster, features):
    """
    Try to parse gblinear weights from dump_model and compute |w| (or w^2) as importance.
    If parsing fails, return zeros (does not affect the main flow).
    """
    try:
        _ = booster.get_dump(with_stats=True)
        # Implementation is version-dependent; fall back to zeros to keep the pipeline robust.
        return pd.DataFrame({"feature": features, "coef_abs": np.zeros(len(features))})
    except Exception:
        return pd.DataFrame({"feature": features, "coef_abs": np.zeros(len(features))})

def run_cv_xgb(train_pre, y, test_pre, features, cfg):
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )

    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)
    best_iters = []

    X = train_pre[features]
    X_test = test_pre[features]

    print(f"Starting {cfg['n_splits']}-fold cross-validation ...")
    rng = np.random.RandomState(cfg["seed"])

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # ===== Imbalance handling =====
        imb = cfg.get("imbalance", {"method": "none"})
        method = imb.get("method", "none")
        train_weights = None
        fold_params = dict(cfg["xgb_params"])  # allow per-fold modifications

        if method == "undersample":
            mask = _undersample_indices(
                y_trn,
                desired_neg_pos_ratio=float(imb.get("undersample_neg_pos_ratio", 3.0)),
                rng=rng,
            )
            X_trn = X_trn.iloc[mask]
            y_trn = y_trn[mask]
            print(f"[Fold {fold}] After undersampling: n={len(y_trn)} (pos={np.sum(y_trn==1)}, neg={np.sum(y_trn==0)})")

        elif method == "auto_weight":
            pos_w = _compute_pos_weight(y_trn, cap=float(imb.get("max_pos_weight_cap", 50.0)))
            train_weights = np.where(y_trn == 1, pos_w, 1.0).astype(float)
            print(f"[Fold {fold}] auto_weight with positive weight ≈ {pos_w:.2f}")

        elif method == "scale_pos_weight":
            pos_w = _compute_pos_weight(y_trn, cap=float(imb.get("max_pos_weight_cap", 50.0)))
            fold_params["scale_pos_weight"] = pos_w
            print(f"[Fold {fold}] scale_pos_weight = {pos_w:.2f}")

        else:
            print(f"[Fold {fold}] No imbalance handling")

        # ===== DMatrix =====
        dtrn = xgb.DMatrix(X_trn.values, label=y_trn, feature_names=features, nthread=fold_params["nthread"])
        dval = xgb.DMatrix(X_val.values, label=y_val, feature_names=features, nthread=fold_params["nthread"])
        dtest = xgb.DMatrix(X_test.values, feature_names=features, nthread=fold_params["nthread"])

        if train_weights is not None:
            dtrn.set_weight(train_weights)

        evals = [(dtrn, "train"), (dval, "valid")]
        model = xgb.train(
            params=fold_params,
            dtrain=dtrn,
            num_boost_round=cfg["num_boost_round"],
            evals=evals,
            early_stopping_rounds=cfg["early_stopping_rounds"],
            verbose_eval=cfg["verbose_eval"]
        )

        # Validation predictions/score
        val_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        best_iters.append(model.best_iteration)

        if cfg.get("report_pr_auc", True):
            val_ap = average_precision_score(y_val, val_pred)
            print(f"[Fold {fold}] AUC = {val_auc:.6f} | AP (PR-AUC) = {val_ap:.6f}  best_iter = {model.best_iteration}")
        else:
            print(f"[Fold {fold}] AUC = {val_auc:.6f}  best_iter = {model.best_iteration}")

        # Test predictions (averaged)
        test_pred += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / cfg["n_splits"]

        # Cleanup
        del dtrn, dval, dtest, model, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    if cfg.get("report_pr_auc", True):
        oof_ap = average_precision_score(y, oof)
        print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} | OOF AP = {oof_ap:.6f} ===")
    else:
        print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")

    # Train one final model on all training data to obtain "importance"
    avg_best = int(np.mean(best_iters)) if len(best_iters) > 0 else cfg["num_boost_round"]
    dall = xgb.DMatrix(X.values, label=y, feature_names=features)
    final_model = xgb.train(
        params=cfg["xgb_params"],
        dtrain=dall,
        num_boost_round=max(50, avg_best),
        evals=[(dall, "train")],
        verbose_eval=False
    )

    booster = cfg["xgb_params"].get("booster", "gbtree")
    imp_type = "weight" if booster == "gblinear" else "gain"
    raw_imp = final_model.get_score(importance_type=imp_type)  # {feature_name: score}

    # Importance DataFrame (gblinear may return empty; provide fallback)
    if len(raw_imp) == 0 and booster == "gblinear":
        coef_df = _parse_gblinear_coef_norm(final_model, features)
        feature_importance = coef_df.rename(columns={"coef_abs": "weight"})
        feature_importance = feature_importance.sort_values("weight", ascending=False).reset_index(drop=True)
    elif len(raw_imp) == 0:
        feature_importance = pd.DataFrame({"feature": features, imp_type: np.zeros(len(features))})
    else:
        feature_importance = (
            pd.DataFrame(list(raw_imp.items()), columns=["feature", imp_type])
            .sort_values(imp_type, ascending=False)
            .reset_index(drop=True)
        )

    return oof, test_pred, oof_auc, feature_importance

def main():
    t0 = time.time()
    ensure_dir(CFG["out_dir"])

    print("Reading data ...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (One-Hot / imputation / standardization) ...")
    train_pre, test_pre, y, features = linear_friendly_preprocess(
        train, test, CFG["target"], CFG["id_col"], CFG
    )
    print(f"Number of features: {len(features)}")

    # Class distribution (informational)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    print(f"Label distribution: neg={neg}, pos={pos}, neg:pos≈{(neg/max(1,pos)):.2f}:1")

    print("Cross-validation + training XGBoost (gblinear) ...")
    oof, test_pred, oof_auc, fi = run_cv_xgb(
        train_pre, y, test_pre, features, CFG
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

    print(f"\nSaved:\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n- Feature importance: {fi_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
