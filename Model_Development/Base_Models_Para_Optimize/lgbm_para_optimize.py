#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM baseline + coarse grid search (num_leaves × min_data_in_leaf)
- Automatically finds the best combination
- Retrains with the best parameters and exports submission/OOF/feature importance
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
import lightgbm as lgb

# Optional progress bar
try:
    from tqdm import tqdm
    _use_tqdm = True
except Exception:
    _use_tqdm = False


# ========== CONFIGURATION ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # If you want to use filtered F.L.I features, switch to these:
    # "train_path": "fli_application_train.csv",
    # "test_path":  "fli_application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_lgbm_grid",
    "lgb_params": {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,            # will be overridden by grid
        "min_data_in_leaf": 100,     # will be overridden by grid
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "max_depth": -1,
        "verbosity": -1,
        "boosting_type": "gbdt",
        # Threading and determinism parameters (recommended to keep)
        "num_threads": -1,
        "deterministic": True,
        "force_row_wise": True,
        "seed": 42
    },
    "num_boost_round": 10000,
    "early_stopping_rounds": 200,

    # Grid search ranges
    "grid_leaves": [63, 127, 255],
    "grid_minleaf": [60, 120, 240, 400],
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
            f"Training file not found: {train_path} (expected application_train.csv)"
        )
    return train, test


def preprocess(train, test, target, id_col):
    """Basic preprocessing: label encoding + missing value imputation."""
    y = train[target].astype(int).values
    train = train.drop(columns=[target])

    # Merge to apply consistent label encoding
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # Identify categorical and numeric columns
    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols]

    # Handle missing values
    combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")
    for col in num_cols:
        if col == id_col:
            continue
        if combined[col].isna().any():
            median_val = combined[col].median()
            combined[col] = combined[col].fillna(median_val)

    # Label encoding (fit on combined train+test to avoid unseen labels)
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Split back
    train_pre = combined.iloc[:len(train)]
    test_pre = combined.iloc[len(train):].reset_index(drop=True)

    features = [c for c in train_pre.columns if c != id_col]
    return train_pre, test_pre, y, features, cat_cols


def run_cv_lgbm(train_pre, y, test_pre, features, id_col, cfg):
    """5-fold CV for LightGBM. Returns OOF predictions, test predictions, OOF AUC, and feature importance."""
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )

    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)
    feature_importance = pd.DataFrame(0, index=features, columns=["importance"])

    X = train_pre[features]
    X_test = test_pre[features]

    print(f"Starting {cfg['n_splits']}-fold CV ...")
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        lgb_train = lgb.Dataset(X_trn, label=y_trn, free_raw_data=True)
        lgb_valid = lgb.Dataset(X_val, label=y_val, free_raw_data=True)

        callbacks = [
            lgb.early_stopping(cfg["early_stopping_rounds"]),
            lgb.log_evaluation(200)
        ]

        model = lgb.train(
            cfg["lgb_params"],
            lgb_train,
            num_boost_round=cfg["num_boost_round"],
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            callbacks=callbacks
        )

        # Validation predictions
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        print(f"[Fold {fold}] AUC = {val_auc:.6f}  best_iter = {model.best_iteration}")

        # Test predictions (average over folds)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / cfg["n_splits"]

        # Feature importance
        fi = pd.DataFrame({
            "feature": features,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split")
        }).set_index("feature")
        feature_importance["importance"] += fi["gain"]

        del lgb_train, lgb_valid, model, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    return oof, test_pred, oof_auc, feature_importance


def grid_search_lgbm_complexity(train_pre, y, test_pre, features, id_col, base_cfg,
                                leaves_list, min_data_list, save_csv_path=None):
    """Grid search for two complexity parameters: num_leaves × min_data_in_leaf."""
    pairs = list(product(leaves_list, min_data_list))
    records = []

    iterator = tqdm(pairs, desc="Grid search (leaves × min_data_in_leaf)") if _use_tqdm else pairs

    best_auc = -1.0
    best_pack = None

    for nl, mdl in iterator:
        cfg = deepcopy(base_cfg)
        cfg["lgb_params"].update({
            "num_leaves": nl,
            "min_data_in_leaf": mdl,
            "num_threads": cfg["lgb_params"].get("num_threads", -1),
            "deterministic": cfg["lgb_params"].get("deterministic", True),
            "force_row_wise": cfg["lgb_params"].get("force_row_wise", True),
        })

        print(f"\n[Try] num_leaves={nl}, min_data_in_leaf={mdl}")
        oof, test_pred, oof_auc, fi = run_cv_lgbm(
            train_pre, y, test_pre, features, id_col, cfg
        )

        records.append({
            "num_leaves": nl,
            "min_data_in_leaf": mdl,
            "oof_auc": oof_auc
        })

        if _use_tqdm:
            iterator.set_postfix({"leaves": nl, "min_leaf": mdl, "auc": f"{oof_auc:.6f}"})

        if oof_auc > best_auc:
            best_auc = oof_auc
            best_pack = {
                "num_leaves": nl,
                "min_data_in_leaf": mdl,
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

    print("Preprocessing (label encoding, missing value imputation) ...")
    train_pre, test_pre, y, features, cat_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"]
    )
    print(f"Number of features: {len(features)} (categorical: {len(cat_cols)})")

    # ===== Coarse grid search =====
    grid_csv = os.path.join(CFG["out_dir"], f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    best_pack, tried_df = grid_search_lgbm_complexity(
        train_pre, y, test_pre, features, CFG["id_col"], CFG,
        leaves_list=CFG["grid_leaves"],
        min_data_list=CFG["grid_minleaf"],
        save_csv_path=grid_csv
    )

    print("\n[GridSearch] Top-10 (sorted by OOF AUC):")
    print(tried_df.head(10).to_string(index=False))

    print(f"\n[Best] num_leaves={best_pack['num_leaves']}, "
          f"min_data_in_leaf={best_pack['min_data_in_leaf']}, "
          f"OOF AUC={best_pack['oof_auc']:.6f}")

    # Retrain with best parameters and save outputs
    oof_g, test_pred_g, fi_g, best_cfg = best_pack["detail"]

    print("\n[Retrain] Retraining full CV with best parameters and exporting files ...")
    oof, test_pred, oof_auc, fi = run_cv_lgbm(
        train_pre, y, test_pre, features, CFG["id_col"], best_cfg
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(CFG["out_dir"], f"submission_{stamp}.csv")
    oof_path = os.path.join(CFG["out_dir"], f"oof_{stamp}.csv")
    fi_path  = os.path.join(CFG["out_dir"], f"feature_importance_{stamp}.csv")
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

    # Save run configuration and best parameters
    out_cfg = deepcopy(best_cfg)
    out_cfg["best_search"] = {
        "num_leaves": best_pack["num_leaves"],
        "min_data_in_leaf": best_pack["min_data_in_leaf"],
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
