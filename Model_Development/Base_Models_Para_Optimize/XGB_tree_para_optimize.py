#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - XGBoost (tree) Small Grid Search (max_depth × min_child_weight)
=============================================================================
- Reads All_application_train.csv / All_application_test.csv (switch back to application_*.csv if needed)
- Preprocessing: categorical LabelEncoding + numeric median imputation (aligned with LGBM)
- 5-fold StratifiedKFold (folds/rounds/early stopping aligned with your LGBM version)
- Grid: max_depth × min_child_weight (small grid, time-friendly)
- Outputs: submission / OOF / feature_importance / run config / best params
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

import xgboost as xgb

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
    # "train_path": "application_train.csv",
    # "test_path":  "application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_xgb_grid",

    # ===== XGBoost base params (grid will override the targeted ones) =====
    "xgb_params": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,                 # = learning_rate
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "lambda": 0.0,               # L2
        "alpha": 0.0,                # L1
        "max_depth": 6,              # overridden by grid
        "min_child_weight": 1,       # overridden by grid
        "tree_method": "hist",
        "max_bin": 256,
        "verbosity": 0,
        "seed": 42
        # Note: if nthread is not set, XGBoost will use available cores; add "nthread": os.cpu_count() to fix it.
    },

    "num_boost_round": 10000,        # aligned with your LGBM version
    "early_stopping_rounds": 200,    # aligned with your LGBM version

    # ===== Small grid (time-friendly) =====
    "grid_max_depth": [4, 6, 8],
    "grid_min_child_weight": [1, 5, 10],
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
            f"Training file not found: {train_path} (expected {os.path.basename(train_path)})"
        )
    return train, test


def preprocess(train, test, target, id_col):
    """Basic preprocessing: label-encode categoricals + median-impute numerics."""
    y = train[target].astype(int).values
    train = train.drop(columns=[target])

    combined = pd.concat([train, test], axis=0, ignore_index=True)

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

    train_pre = combined.iloc[:len(train)]
    test_pre = combined.iloc[len(train):].reset_index(drop=True)

    features = [c for c in train_pre.columns if c != id_col]
    return train_pre, test_pre, y, features, cat_cols


def _booster_feature_importance(booster: xgb.Booster, features: list) -> pd.DataFrame:
    """Map XGBoost gain/split importance safely onto the full feature set, filling missing with 0."""
    gain_map = booster.get_score(importance_type="gain")
    split_map = booster.get_score(importance_type="weight")  # aka "split"

    gain_vals = [float(gain_map.get(f, 0.0)) for f in features]
    split_vals = [float(split_map.get(f, 0.0)) for f in features]

    fi = pd.DataFrame({"feature": features, "gain": gain_vals, "split": split_vals})
    fi = fi.set_index("feature")
    return fi


def run_cv_xgb(train_pre, y, test_pre, features, id_col, cfg):
    """K-fold CV with XGBoost; returns OOF, test predictions, OOF AUC, and feature importance (gain-accumulated)."""
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )

    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)
    feature_importance = pd.DataFrame(0.0, index=features, columns=["importance"])

    X = train_pre[features]
    X_test = test_pre[features]

    print(f"Starting {cfg['n_splits']}-fold CV ...")
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        dtrn = xgb.DMatrix(X_trn, label=y_trn, feature_names=features)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
        dtest = xgb.DMatrix(X_test, feature_names=features)

        watchlist = [(dtrn, "train"), (dval, "valid")]

        model = xgb.train(
            params=cfg["xgb_params"],
            dtrain=dtrn,
            num_boost_round=cfg["num_boost_round"],
            evals=watchlist,
            early_stopping_rounds=cfg["early_stopping_rounds"],
            verbose_eval=200
        )

        # Validation predictions
        val_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        print(f"[Fold {fold}] AUC = {val_auc:.6f} best_iter = {model.best_iteration}")

        # Aggregate test predictions
        test_pred += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / cfg["n_splits"]

        # Importance (accumulate gain)
        fi = _booster_feature_importance(model, features)
        feature_importance["importance"] += fi["gain"]

        del dtrn, dval, dtest, model, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    return oof, test_pred, oof_auc, feature_importance


def grid_search_xgb_complexity(train_pre, y, test_pre, features, id_col, base_cfg,
                               depth_list, mcw_list, save_csv_path=None):
    """
    Search only two complexity parameters: max_depth × min_child_weight.
    Other params follow base_cfg['xgb_params']; early stopping controls the number of rounds.
    """
    pairs = list(product(depth_list, mcw_list))
    records = []

    iterator = pairs
    if _use_tqdm:
        iterator = tqdm(pairs, desc="Grid search (max_depth × min_child_weight)")

    best_auc = -1.0
    best_pack = None

    for md, mcw in iterator:
        cfg = deepcopy(base_cfg)
        cfg["xgb_params"].update({
            "max_depth": int(md),
            "min_child_weight": float(mcw),
            # keep stability-related params
            "eta": cfg["xgb_params"].get("eta", 0.05),
            "subsample": cfg["xgb_params"].get("subsample", 0.8),
            "colsample_bytree": cfg["xgb_params"].get("colsample_bytree", 0.8),
            "tree_method": cfg["xgb_params"].get("tree_method", "hist"),
            "seed": cfg["xgb_params"].get("seed", base_cfg["seed"]),
        })

        print(f"\n[Try] max_depth={md}, min_child_weight={mcw}")
        oof, test_pred, oof_auc, fi = run_cv_xgb(
            train_pre, y, test_pre, features, id_col, cfg
        )

        records.append({
            "max_depth": md,
            "min_child_weight": mcw,
            "oof_auc": oof_auc
        })

        if _use_tqdm:
            iterator.set_postfix({"max_depth": md, "min_child_weight": mcw, "auc": f"{oof_auc:.6f}"})

        if oof_auc > best_auc:
            best_auc = oof_auc
            best_pack = {
                "max_depth": md,
                "min_child_weight": mcw,
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

    print("Loading data...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (label encoding, median imputation)...")
    train_pre, test_pre, y, features, cat_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"]
    )
    print(f"Number of features: {len(features)} (categorical columns: {len(cat_cols)})")

    # ====== Small grid search ======
    grid_csv = os.path.join(CFG["out_dir"], f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    best_pack, tried_df = grid_search_xgb_complexity(
        train_pre, y, test_pre, features, CFG["id_col"], CFG,
        depth_list=CFG["grid_max_depth"],
        mcw_list=CFG["grid_min_child_weight"],
        save_csv_path=grid_csv
    )

    print("\n[GridSearch] Top-10 (sorted by OOF AUC):")
    print(tried_df.head(10).to_string(index=False))

    print(f"\n[Best] max_depth={best_pack['max_depth']}, "
          f"min_child_weight={best_pack['min_child_weight']}, "
          f"OOF AUC={best_pack['oof_auc']:.6f}")

    # ====== Retrain with best params and save outputs ======
    # Retrieve the best config (includes other xgb_params)
    oof_g, test_pred_g, fi_g, best_cfg = best_pack["detail"]  # (oof, test_pred, fi, cfg)

    print("\n[Retrain] Re-training full CV with best params and exporting files ...")
    oof, test_pred, oof_auc, fi = run_cv_xgb(
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

    fi.reset_index().rename(columns={"index": "feature"}).to_csv(fi_path, index=False)

    # Save run config + best params
    out_cfg = deepcopy(best_cfg)
    out_cfg["best_search"] = {
        "max_depth": best_pack["max_depth"],
        "min_child_weight": best_pack["min_child_weight"],
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
