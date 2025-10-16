#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Home Credit - XGBoost (gblinear) Baseline with Linear-Friendly Preprocessing
============================================================================
- Categorical: One-Hot
- Numerical : median imputation (fit on train) + StandardScaler (fit on train)
- CV        : StratifiedKFold
- Importance: 'weight' for gblinear; 'gain' for tree boosters

Outputs:
  - submission_*.csv
  - oof_*.csv
  - feature_importance_*.csv
  - run_config_*.json
"""

import os, gc, json, time
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb

# ========== Configuration ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # If you already have filtered-feature files, you may switch to fli_* paths.
    # "train_path": "application_train.csv",
    # "test_path":  "application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path":  "../Features/All_application_test.csv",
    "out_dir": "baseline_xgb_linear",

    # XGBoost parameters (linear booster)
    "xgb_params": {
        "booster": "gblinear",            # linear booster
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "lambda": 1.0,                    # L2 regularization
        "alpha": 0.0,                     # L1 regularization (try >0 for sparsity)
        "learning_rate": 0.05,
        "n_jobs": -1,
        "seed": 42
    },
    "num_boost_round": 5000,
    "early_stopping_rounds": 200,
    "verbose_eval": 200,

    # Preprocessing switches
    "one_hot_categoricals": True,
    "standardize_numericals": True,
}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_data(train_path, test_path):
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    test = pd.read_csv(test_path)

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    train = pd.read_csv(train_path)

    return train, test


def linear_friendly_preprocess(train: pd.DataFrame, test: pd.DataFrame,
                               target: str, id_col: str, cfg: dict):
    """
    - One-Hot encode categorical columns (fit on train+test combined to align columns)
    - Numerical columns: fill missing with train medians, then apply the same to test
    - Numerical columns: standardize (fit scaler on training rows only)
    """
    y = train[target].astype(int).values
    train_nolabel = train.drop(columns=[target])

    # Column types
    cat_cols = train_nolabel.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in train_nolabel.columns if c not in cat_cols and c != id_col]

    # Combine for column-space alignment (especially for OHE)
    combined = pd.concat([train_nolabel, test], axis=0, ignore_index=True)

    # Missing values
    if cat_cols:
        combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")

    # Numericals: median imputation using train medians
    train_num = train_nolabel[num_cols]
    medians = train_num.median(axis=0)
    for col in num_cols:
        combined[col] = combined[col].fillna(medians[col])

    # One-Hot encode categoricals (do not drop first; let regularization handle redundancy)
    if cfg["one_hot_categoricals"] and len(cat_cols) > 0:
        dummies = pd.get_dummies(combined[cat_cols], dummy_na=False)
        combined = pd.concat([combined.drop(columns=cat_cols), dummies], axis=1)

    # Standardize numericals (fit on train rows only)
    if cfg["standardize_numericals"] and len(num_cols) > 0:
        scaler = StandardScaler()
        scaler.fit(combined.iloc[:len(train)][num_cols])
        combined[num_cols] = scaler.transform(combined[num_cols])

    # Split back
    train_pre = combined.iloc[:len(train)].reset_index(drop=True)
    test_pre  = combined.iloc[len(train):].reset_index(drop=True)

    # Feature list = all except ID
    features = [c for c in train_pre.columns if c != id_col]

    return train_pre, test_pre, y, features


def run_cv_xgb(train_pre, y, test_pre, features, cfg):
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )

    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)
    best_iters = []

    X = train_pre[features]
    X_test = test_pre[features]

    print(f"Starting {cfg['n_splits']}-fold CV ...")
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        dtrn = xgb.DMatrix(X_trn.values, label=y_trn, feature_names=features)
        dval = xgb.DMatrix(X_val.values, label=y_val, feature_names=features)
        dtest = xgb.DMatrix(X_test.values, feature_names=features)

        evals = [(dtrn, "train"), (dval, "valid")]
        model = xgb.train(
            params=cfg["xgb_params"],
            dtrain=dtrn,
            num_boost_round=cfg["num_boost_round"],
            evals=evals,
            early_stopping_rounds=cfg["early_stopping_rounds"],
            verbose_eval=cfg["verbose_eval"]
        )

        # Validation prediction/score
        val_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y[val_idx], val_pred)
        best_iters.append(model.best_iteration)
        print(f"[Fold {fold}] AUC = {val_auc:.6f}  best_iter = {model.best_iteration}")

        # Test predictions (averaged over folds)
        test_pred += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / cfg["n_splits"]

        # Cleanup
        del dtrn, dval, dtest, model, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")

    # Train one final model on all training data to get importances
    avg_best = int(np.mean(best_iters)) if len(best_iters) > 0 else cfg["num_boost_round"]
    avg_best = max(50, avg_best)  # be a bit more stable
    dall = xgb.DMatrix(X.values, label=y, feature_names=features)
    final_model = xgb.train(
        params=cfg["xgb_params"],
        dtrain=dall,
        num_boost_round=avg_best,
        evals=[(dall, "train")],
        verbose_eval=False
    )

    booster = cfg["xgb_params"].get("booster", "gbtree")
    imp_type = "weight" if booster == "gblinear" else "gain"
    raw_imp = final_model.get_score(importance_type=imp_type)  # {feature_name: score}

    # Importance DataFrame
    if len(raw_imp) == 0:
        # In rare cases gblinear may return empty weights; fall back to zeros.
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

    print("Reading data...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (One-Hot / median imputation / standardization) ...")
    train_pre, test_pre, y, features = linear_friendly_preprocess(
        train, test, CFG["target"], CFG["id_col"], CFG
    )
    print(f"Feature count: {len(features)}")

    print("Cross-validation + training XGBoost (gblinear) ...")
    oof, test_pred, oof_auc, fi = run_cv_xgb(train_pre, y, test_pre, features, CFG)

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
