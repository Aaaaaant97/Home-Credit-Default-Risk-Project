#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost (with optional feature binning) + Light Grid Search
- The grid searches only a few key params (time-friendly): max_depth × min_child_weight × subsample × colsample_bytree
- All other params stay the same as in CFG['xgb_params']
- The number of folds (n_splits), boosting rounds (num_boost_round), and early stopping (early_stopping_rounds) are fixed by CFG
- Matches the fund_*.py style used in the project: saves submission / OOF / feature importance / run config
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

# ========== Configuration ==========
CFG = {
    "seed": 42,
    "n_splits": 5,                      # number of CV folds
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # Use the main table or your integrated features
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "fund_xgb_binning_grid",

    # Boosting rounds & early stopping (adjust if needed)
    "num_boost_round": 10000,
    "early_stopping_rounds": 200,

    # Optional numeric feature binning. Keep consistent with your original fund_XGB_binning.py.
    "enable_binning": True,
    "bin_num": 64,                      # number of qcut bins per numeric column (missing has its own category upstream)

    # Base XGBoost params (the grid will override a subset)
    "xgb_params": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 6,                 # overridden by grid
        "min_child_weight": 1.0,        # overridden by grid
        "subsample": 0.8,               # overridden by grid
        "colsample_bytree": 0.8,        # overridden by grid
        "lambda": 0.0,
        "alpha": 0.0,
        "tree_method": "hist",          # fast & stable on CPU
        "max_bin": 256,                 # histogram binning; pairs well with feature binning
        "verbosity": 0,
        "seed": 42,
        "nthread": -1
    },

    # Light grid (time-friendly) — adjust as you like
    "grid_max_depth": [5, 7],
    "grid_min_child_weight": [1.0, 3.0],
    "grid_subsample": [0.8, 1.0],
    "grid_colsample_bytree": [0.7, 0.9],
}

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def read_data(train_path: str, test_path: str):
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    test = pd.read_csv(test_path)

    if os.path.isfile(train_path):
        train = pd.read_csv(train_path)
    else:
        raise FileNotFoundError(f"Training file not found: {train_path}")
    return train, test

def preprocess(train: pd.DataFrame, test: pd.DataFrame, target: str, id_col: str, enable_binning: bool, bin_num: int):
    """Basic preprocessing: label-encode categoricals + median-impute numerics; optional numeric binning (applied to train+test uniformly)."""
    y = train[target].astype(int).values
    train = train.drop(columns=[target])

    # Combine for unified transforms
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # Column types
    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols]

    # Missing values (categorical / numeric)
    combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")
    for col in num_cols:
        if col == id_col:
            continue
        if combined[col].isna().any():
            median_val = combined[col].median()
            combined[col] = combined[col].fillna(median_val)

    # Optional: qcut binning for numeric columns (pairs well with tree_method='hist' / max_bin)
    if enable_binning:
        for col in num_cols:
            if col == id_col:
                continue
            try:
                # Use rank to avoid duplicate-quantile issues; labels=False gives 0..(bin_num-1)
                combined[col], _ = pd.qcut(
                    combined[col].rank(method="first"),
                    q=bin_num, labels=False, retbins=True, duplicates="drop"
                )
            except Exception:
                # If too few unique values, qcut may fail — skip binning for that column
                pass

    # Label encode categoricals (fit on train+test together to avoid unseen categories)
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Split back
    train_pre = combined.iloc[:len(train)]
    test_pre  = combined.iloc[len(train):].reset_index(drop=True)
    features = [c for c in train_pre.columns if c != id_col]
    return train_pre, test_pre, y, features, cat_cols

def run_cv_xgb(train_pre, y, test_pre, features, id_col, cfg):
    """K-fold CV training with XGBoost; returns OOF, test predictions, OOF AUC, and feature importance (gain)."""
    folds = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)
    fi_gain_acc = pd.Series(0.0, index=features, dtype=float)

    X = train_pre[features]
    X_test = test_pre[features]

    print(f"Starting {cfg['n_splits']}-fold CV ...")
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_trn, label=y_trn, nthread=cfg["xgb_params"].get("nthread", None))
        dvalid = xgb.DMatrix(X_val,  label=y_val,  nthread=cfg["xgb_params"].get("nthread", None))
        dtest  = xgb.DMatrix(X_test,                nthread=cfg["xgb_params"].get("nthread", None))

        watchlist = [(dtrain, "train"), (dvalid, "valid")]
        model = xgb.train(
            params=cfg["xgb_params"],
            dtrain=dtrain,
            num_boost_round=cfg["num_boost_round"],
            evals=watchlist,
            early_stopping_rounds=cfg["early_stopping_rounds"],
            verbose_eval=200
        )

        val_pred = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        print(f"[Fold {fold}] AUC = {val_auc:.6f} best_iter = {model.best_iteration}")

        test_pred += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / cfg["n_splits"]

        # Feature importance (gain). XGBoost names features as f0, f1, ... in the booster.
        score = model.get_score(importance_type="gain")
        # Fill missing features with 0
        gain = pd.Series({f: score.get(f, 0.0) for f in model.feature_names})
        # Map f0/f1/... back to original column names
        fmap = {f"f{i}": feat for i, feat in enumerate(features)}
        gain_mapped = pd.Series({fmap.get(k, k): v for k, v in gain.items()})
        fi_gain_acc = fi_gain_acc.add(gain_mapped.reindex(features).fillna(0.0), fill_value=0.0)

        del dtrain, dvalid, dtest, model, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")

    fi_df = pd.DataFrame({"feature": features, "gain": fi_gain_acc.values}).sort_values("gain", ascending=False)
    return oof, test_pred, oof_auc, fi_df

def grid_search_xgb(train_pre, y, test_pre, features, id_col, base_cfg,
                    grid_max_depth, grid_min_child_weight, grid_subsample, grid_colsample_bytree,
                    save_csv_path=None):
    """Light grid over: max_depth × min_child_weight × subsample × colsample_bytree."""
    combos = list(product(grid_max_depth, grid_min_child_weight, grid_subsample, grid_colsample_bytree))
    records = []
    best_auc = -1.0
    best_pack = None

    for (md, mcw, ss, cs) in combos:
        cfg = deepcopy(base_cfg)
        cfg["xgb_params"].update({
            "max_depth": int(md),
            "min_child_weight": float(mcw),
            "subsample": float(ss),
            "colsample_bytree": float(cs),
        })

        print(f"\n[Try] max_depth={md}, min_child_weight={mcw}, subsample={ss}, colsample_bytree={cs}")
        oof, test_pred, oof_auc, fi = run_cv_xgb(train_pre, y, test_pre, features, id_col, cfg)

        records.append({
            "max_depth": md,
            "min_child_weight": mcw,
            "subsample": ss,
            "colsample_bytree": cs,
            "oof_auc": oof_auc
        })

        if oof_auc > best_auc:
            best_auc = oof_auc
            best_pack = {
                "params": {"max_depth": md, "min_child_weight": mcw, "subsample": ss, "colsample_bytree": cs},
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

    print("Preprocessing (categorical encoding, median imputation{})...".format(" + binning" if CFG["enable_binning"] else ""))
    train_pre, test_pre, y, features, cat_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"],
        enable_binning=CFG["enable_binning"],
        bin_num=CFG["bin_num"]
    )
    print(f"Number of features: {len(features)} (categorical columns: {len(cat_cols)})")

    # ====== Light Grid Search ======
    grid_csv = os.path.join(CFG["out_dir"], f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    best_pack, tried_df = grid_search_xgb(
        train_pre, y, test_pre, features, CFG["id_col"], CFG,
        grid_max_depth=CFG["grid_max_depth"],
        grid_min_child_weight=CFG["grid_min_child_weight"],
        grid_subsample=CFG["grid_subsample"],
        grid_colsample_bytree=CFG["grid_colsample_bytree"],
        save_csv_path=grid_csv
    )

    print("\n[GridSearch] Top-10 (sorted by OOF AUC):")
    print(tried_df.head(10).to_string(index=False))

    best_params = best_pack["params"]
    print(f"\n[Best] max_depth={best_params['max_depth']}, "
          f"min_child_weight={best_params['min_child_weight']}, "
          f"subsample={best_params['subsample']}, "
          f"colsample_bytree={best_params['colsample_bytree']}, "
          f"OOF AUC={best_pack['oof_auc']:.6f}")

    # ====== Retrain with best params and save outputs ======
    oof_g, test_pred_g, fi_g, best_cfg = best_pack["detail"]  # (oof, test_pred, fi, cfg)

    print("\n[Retrain] Re-training full CV with best params and exporting files ...")
    oof, test_pred, oof_auc, fi = run_cv_xgb(
        train_pre, y, test_pre, features, CFG["id_col"], best_cfg
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(CFG["out_dir"], f"submission_{stamp}.csv")
    oof_path = os.path.join(CFG["out_dir"], f"oof_{stamp}.csv")
    fi_path  = os.path.join(CFG["out_dir"], f"feature_importance_{stamp}.csv")
    cfg_path = os.path.join(CFG["out_dir"], f"run_config_{stamp}.json")
    best_path = os.path.join(CFG["out_dir"], f"best_params_{stamp}.json")

    # Submission
    submission = test[[CFG["id_col"]]].copy()
    submission[CFG["target"]] = test_pred
    submission.to_csv(sub_path, index=False)

    # OOF predictions
    oof_df = train[[CFG["id_col"]]].copy()
    oof_df["oof_pred"] = oof
    oof_df[CFG["target"]] = y
    oof_df.to_csv(oof_path, index=False)

    # Feature importance
    fi.to_csv(fi_path, index=False)

    # Save run config + best params
    out_cfg = deepcopy(best_cfg)
    out_cfg["best_search"] = {**best_params, "oof_auc": best_pack["oof_auc"]}
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:\n- Grid results: {grid_csv}\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n"
          f"- Feature importance: {fi_path}\n- Best params: {best_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
