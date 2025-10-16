#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HistGradientBoosting baseline + coarse grid search (max_leaf_nodes × min_samples_leaf)
- Automatically finds the best-performing combination (small grid, time-friendly)
- Retrains with the best parameters and exports submission/OOF/feature importance/grid results
- Number of folds, max iterations, and early-stopping rounds aligned with your LGBM template
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
from sklearn.ensemble import HistGradientBoostingClassifier

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
    # Default to All_XX; change to application_train/test if needed
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_hgbc_grid",

    # Map LGBM semantics to HGBC
    "num_boost_round": 10000,        # -> max_iter
    "early_stopping_rounds": 200,    # -> n_iter_no_change

    # Base parameters (grid will override max_leaf_nodes / min_samples_leaf)
    "hgbc_params": {
        "loss": "log_loss",
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 100,
        "max_bins": 255,
        "l2_regularization": 0.0,
        "max_depth": None,
        "early_stopping": True,
        "n_iter_no_change": 200,
        "validation_fraction": 0.1,
        "class_weight": None,  # use "balanced" for imbalanced datasets
        "random_state": 42
    },

    # ===== Grid search range (small for speed) =====
    "grid_max_leaf_nodes": [31, 63, 127],
    "grid_min_samples_leaf": [60, 120, 240],
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
    """Basic preprocessing: label encoding + numeric imputation (median fill)."""
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
            median_val = combined[col].median()
            combined[col] = combined[col].fillna(median_val)

    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    train_pre = combined.iloc[:len(train)]
    test_pre = combined.iloc[len(train):].reset_index(drop=True)

    features = [c for c in train_pre.columns if c != id_col]
    return train_pre, test_pre, y, features, cat_cols


def run_cv_hgbc(train_pre, y, test_pre, features, id_col, cfg):
    """
    5-fold CV for HistGradientBoostingClassifier.
    Returns OOF predictions, test predictions, OOF AUC, and feature importance.
    """
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )

    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)
    feature_importance = pd.DataFrame(0.0, index=features, columns=["importance"])

    X = train_pre[features].values
    X_test = test_pre[features].values

    print(f"Starting {cfg['n_splits']}-fold CV ...")
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        params = deepcopy(cfg["hgbc_params"])
        params["max_iter"] = cfg["num_boost_round"]
        params["n_iter_no_change"] = cfg["early_stopping_rounds"]
        params["random_state"] = cfg["seed"] + fold

        model = HistGradientBoostingClassifier(**params)
        model.fit(X_trn, y_trn)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        used_iter = getattr(model, "n_iter_", None)
        print(f"[Fold {fold}] AUC = {val_auc:.6f}  used_iter = {used_iter}")

        test_pred += model.predict_proba(X_test)[:, 1] / cfg["n_splits"]

        fi = pd.Series(model.feature_importances_, index=features)
        feature_importance["importance"] += fi

        del model, X_trn, X_val, y_trn, y_val, fi
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    return oof, test_pred, oof_auc, feature_importance


def grid_search_hgbc_complexity(train_pre, y, test_pre, features, id_col, base_cfg,
                                leaf_nodes_list, min_samples_list, save_csv_path=None):
    """
    Grid search on two complexity parameters: max_leaf_nodes × min_samples_leaf.
    All other parameters remain fixed.
    """
    pairs = list(product(leaf_nodes_list, min_samples_list))
    records = []

    iterator = tqdm(pairs, desc="Grid search (max_leaf_nodes × min_samples_leaf)") if _use_tqdm else pairs

    best_auc = -1.0
    best_pack = None

    for mln, msl in iterator:
        cfg = deepcopy(base_cfg)
        cfg["hgbc_params"].update({
            "max_leaf_nodes": int(mln),
            "min_samples_leaf": int(msl),
        })

        print(f"\n[Try] max_leaf_nodes={mln}, min_samples_leaf={msl}")
        oof, test_pred, oof_auc, fi = run_cv_hgbc(
            train_pre, y, test_pre, features, id_col, cfg
        )

        records.append({
            "max_leaf_nodes": mln,
            "min_samples_leaf": msl,
            "oof_auc": oof_auc
        })

        if _use_tqdm:
            iterator.set_postfix({"leaf_nodes": mln, "min_leaf": msl, "auc": f"{oof_auc:.6f}"})

        if oof_auc > best_auc:
            best_auc = oof_auc
            best_pack = {
                "max_leaf_nodes": mln,
                "min_samples_leaf": msl,
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

    print("Preprocessing (categorical encoding, missing value imputation) ...")
    train_pre, test_pre, y, features, cat_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"]
    )
    print(f"Feature count: {len(features)} (categorical: {len(cat_cols)})")

    grid_csv = os.path.join(CFG["out_dir"], f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    best_pack, tried_df = grid_search_hgbc_complexity(
        train_pre, y, test_pre, features, CFG["id_col"], CFG,
        leaf_nodes_list=CFG["grid_max_leaf_nodes"],
        min_samples_list=CFG["grid_min_samples_leaf"],
        save_csv_path=grid_csv
    )

    print("\n[GridSearch] Top-10 (sorted by OOF AUC):")
    print(tried_df.head(10).to_string(index=False))

    print(f"\n[Best] max_leaf_nodes={best_pack['max_leaf_nodes']}, "
          f"min_samples_leaf={best_pack['min_samples_leaf']}, "
          f"OOF AUC={best_pack['oof_auc']:.6f}")

    oof_g, test_pred_g, fi_g, best_cfg = best_pack["detail"]

    print("\n[Retrain] Retraining full CV with best parameters and exporting files ...")
    oof, test_pred, oof_auc, fi = run_cv_hgbc(
        train_pre, y, test_pre, features, CFG["id_col"], best_cfg
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(CFG["out_dir"], f"submission_{stamp}.csv")
    oof_path = os.path.join(CFG["out_dir"], f"oof_{stamp}.csv")
    fi_path = os.path.join(CFG["out_dir"], f"feature_importance_{stamp}.csv")
    cfg_path = os.path.join(CFG["out_dir"], f"run_config_{stamp}.json")
    best_path = os.path.join(CFG["out_dir"], f"best_params_{stamp}.json")

    submission = test[[CFG["id_col"]]].copy()
    submission[CFG["target"]] = test_pred
    submission.to_csv(sub_path, index=False)

    oof_df = train[[CFG["id_col"]]].copy()
    oof_df["oof_pred"] = oof
    oof_df[CFG["target"]] = y
    oof_df.to_csv(oof_path, index=False)

    fi.reset_index().rename(columns={"index": "feature"}).to_csv(fi_path, index=False)

    out_cfg = deepcopy(best_cfg)
    out_cfg["best_search"] = {
        "max_leaf_nodes": best_pack["max_leaf_nodes"],
        "min_samples_leaf": best_pack["min_samples_leaf"],
        "oof_auc": best_pack["oof_auc"]
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print(f"\nSaved files:\n- Grid results: {grid_csv}\n- Submission: {sub_path}\n- OOF: {oof_path}\n- Feature importance: {fi_path}\n- Best parameters: {best_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
