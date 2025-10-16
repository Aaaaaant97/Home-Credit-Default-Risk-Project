#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RandomForest (Gini) baseline + Coarse Grid Search (max_depth × min_samples_leaf)
- Automatically finds the best combination by OOF AUC
- Retrains with the best params and exports submission / OOF / feature importance / run config
"""

import os, gc, json, time
from datetime import datetime
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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
    # If you want to use screened F.L.I features, switch the two lines below
    # "train_path": "fli_application_train.csv",
    # "test_path":  "fli_application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_rf_gini_grid",

    # RF base params (grid will override max_depth & min_samples_leaf; others fixed)
    "rf_params": {
        "n_estimators": 800,          # prioritize stability; reduce to ~500 for speed
        "criterion": "gini",
        "max_depth": None,            # overridden by grid
        "min_samples_split": 2,
        "min_samples_leaf": 1,        # overridden by grid
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": "balanced"    # handle class imbalance
    },

    # Grid ranges (two parameters)
    "grid_max_depth": [10, 15, 20, None],
    "grid_min_leaf": [1, 2, 4, 8],
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
            f"Training file not found: {train_path}"
        )
    return train, test


def preprocess(train, test, target, id_col):
    """Basic preprocessing: label-encode categoricals + median imputation for numerics."""
    y = train[target].astype(int).values
    train = train.drop(columns=[target])

    # Combine for unified label encoding
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # Column types
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

    # Label Encoding (fit on train+test together to avoid unseen categories)
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Split back
    train_pre = combined.iloc[:len(train)]
    test_pre  = combined.iloc[len(train):].reset_index(drop=True)

    # Feature columns
    features = [c for c in train_pre.columns if c != id_col]
    return train_pre, test_pre, y, features, cat_cols


def run_cv_rf(train_pre, y, test_pre, features, cfg):
    """5-fold CV for RandomForest; returns OOF, test predictions, OOF AUC, and feature importance."""
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )

    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)
    fi_sum = np.zeros(len(features), dtype=float)

    X = train_pre[features].values
    X_test = test_pre[features].values

    print(f"Starting {cfg['n_splits']}-fold CV ...")
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = RandomForestClassifier(**cfg["rf_params"])
        model.fit(X_trn, y_trn)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        print(f"[Fold {fold}] AUC = {val_auc:.6f}")

        # Aggregate test predictions
        test_pred += model.predict_proba(X_test)[:, 1] / cfg["n_splits"]

        # Impurity-based importance
        fi_sum += model.feature_importances_

        del model, X_trn, X_val, y_trn, y_val, val_pred
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")

    fi_df = pd.DataFrame({"feature": features, "importance": fi_sum})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return oof, test_pred, oof_auc, fi_df


def grid_search_rf_complexity(train_pre, y, test_pre, features, base_cfg,
                              depths, min_leaf_list, save_csv_path=None):
    """
    Search only the two complexity parameters: max_depth × min_samples_leaf.
    Other parameters follow base_cfg['rf_params'].
    """
    pairs = list(product(depths, min_leaf_list))
    records = []

    iterator = pairs
    if _use_tqdm:
        iterator = tqdm(pairs, desc="Grid search (max_depth × min_samples_leaf)")

    best_auc = -1.0
    best_pack = None

    for md, mleaf in iterator:
        cfg = deepcopy(base_cfg)
        cfg["rf_params"].update({
            "max_depth": md,
            "min_samples_leaf": mleaf,
        })

        print(f"\n[Try] max_depth={md}, min_samples_leaf={mleaf}")
        oof, test_pred, oof_auc, fi = run_cv_rf(
            train_pre, y, test_pre, features, cfg
        )

        records.append({
            "max_depth": (md if md is not None else "None"),
            "min_samples_leaf": mleaf,
            "oof_auc": oof_auc
        })

        if _use_tqdm:
            iterator.set_postfix({"depth": md, "min_leaf": mleaf, "auc": f"{oof_auc:.6f}"})

        if oof_auc > best_auc:
            best_auc = oof_auc
            best_pack = {
                "max_depth": md,
                "min_samples_leaf": mleaf,
                "oof_auc": oof_auc,
                "detail": (oof, test_pred, fi, cfg)  # same structure as your LGB script
            }

        gc.collect()

    tried_df = pd.DataFrame(records).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    if save_csv_path is not None:
        tried_df.to_csv(save_csv_path, index=False)

    return best_pack, tried_df


def main():
    t0 = time.time()
    ensure_dir(CFG["out_dir"])

    print("Loading data...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (label encoding + median imputation)...")
    train_pre, test_pre, y, features, cat_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"]
    )
    print(f"Number of features: {len(features)} (categorical columns: {len(cat_cols)})")

    # ====== Coarse Grid Search ======
    grid_csv = os.path.join(CFG["out_dir"], f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    best_pack, tried_df = grid_search_rf_complexity(
        train_pre, y, test_pre, features, CFG,
        depths=CFG["grid_max_depth"],
        min_leaf_list=CFG["grid_min_leaf"],
        save_csv_path=grid_csv
    )

    print("\n[GridSearch] Top-10 (sorted by OOF AUC):")
    print(tried_df.head(10).to_string(index=False))

    print(f"\n[Best] max_depth={best_pack['max_depth']}, "
          f"min_samples_leaf={best_pack['min_samples_leaf']}, "
          f"OOF AUC={best_pack['oof_auc']:.6f}")

    # ====== Retrain with best params and save outputs ======
    # Same detail structure as your LGB version: (oof, test_pred, fi, cfg)
    oof_g, test_pred_g, fi_g, best_cfg = best_pack["detail"]

    print("\n[Retrain] Re-training full CV with best params and exporting files ...")
    oof, test_pred, oof_auc, fi = run_cv_rf(
        train_pre, y, test_pre, features, best_cfg
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

    # Save original run config + best params
    out_cfg = deepcopy(best_cfg)
    out_cfg["best_search"] = {
        "max_depth": best_pack["max_depth"],
        "min_samples_leaf": best_pack["min_samples_leaf"],
        "oof_auc": best_pack["oof_auc"]
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:\n- Grid results: {grid_csv}\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n"
          f"- Feature importance: {fi_path}\n- Best params: {best_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
