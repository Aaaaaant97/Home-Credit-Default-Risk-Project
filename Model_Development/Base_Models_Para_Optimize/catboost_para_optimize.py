#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, gc, json, time
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool


# ========== Global Configuration ==========
CFG = {
    "seed": 42,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "catboost_grid",

    # --- Grid Search Phase (fast) ---
    "grid_cv_splits": 3,
    "grid_iterations": 4000,
    "grid_es_rounds": 200,
    "grid_log_every_n": 200,

    # --- Final Training Phase (robust) ---
    "final_cv_splits": 5,
    "final_iterations": 12000,
    "final_es_rounds": 300,
    "final_log_every_n": 200,

    # Fixed parameters during grid search
    "base_cat_params": {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.03,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.3,
        "border_count": 254,
        "thread_count": -1,
        "random_seed": 42,
        "scale_pos_weight": 5   # handles imbalance (pos:neg ≈ 1:10)
    },

    # Grid search: only tune these
    "grid_depth": [8, 10],
    "grid_l2": [3, 4, 6]
}


# ========== Utilities ==========
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


def preprocess_for_catboost(train: pd.DataFrame, test: pd.DataFrame, target: str, id_col: str):
    """
    Minimal preprocessing:
      - Numeric columns: keep NaNs (CatBoost can handle them)
      - Categorical columns: convert to string, fill missing with "__MISSING__"
    """
    if target not in train.columns:
        raise ValueError(f"Target column '{target}' not found in training data.")

    y = train[target].astype(int).values
    train_nolabel = train.drop(columns=[target])

    combined = pd.concat([train_nolabel, test], axis=0, ignore_index=True)

    cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        combined[c] = combined[c].astype("string").fillna("__MISSING__")

    train_pre = combined.iloc[: len(train_nolabel)].reset_index(drop=True)
    test_pre = combined.iloc[len(train_nolabel):].reset_index(drop=True)

    features = [c for c in train_pre.columns if c != id_col]
    cat_features_idx = [features.index(c) for c in cat_cols if c in features]
    return train_pre, test_pre, y, features, cat_cols, cat_features_idx


def cv_eval_catboost(X, y, features, cat_idx, params, iterations, es_rounds, n_splits, log_every_n):
    """Return mean_auc, std_auc, mean_best_iter, mean_time_s."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG["seed"])
    aucs, best_iters, times = [], [], []

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X[features], y), 1):
        X_trn, y_trn = X.iloc[trn_idx][features], y[trn_idx]
        X_val, y_val = X.iloc[val_idx][features], y[val_idx]

        train_pool = Pool(X_trn, label=y_trn, cat_features=cat_idx)
        valid_pool = Pool(X_val, label=y_val, cat_features=cat_idx)

        model = CatBoostClassifier(iterations=iterations, **params)

        t0 = time.time()
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            verbose=log_every_n,
            early_stopping_rounds=es_rounds
        )
        sec = time.time() - t0

        val_pred = model.predict_proba(valid_pool)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        best_iter = model.get_best_iteration() or model.tree_count_

        aucs.append(auc)
        best_iters.append(best_iter)
        times.append(sec)

        del train_pool, valid_pool, model, X_trn, X_val, y_trn, y_val
        gc.collect()

    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(best_iters)), float(np.mean(times))


def run_final_training(train_pre, y, test_pre, features, cat_idx, best_depth, best_l2,
                       out_dir, iterations, es_rounds, log_every_n):
    """Perform final 5-fold training with best parameters and export submission/OoF/importance."""
    folds = StratifiedKFold(n_splits=CFG["final_cv_splits"], shuffle=True, random_state=CFG["seed"])

    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)
    fi_accum = np.zeros(len(features), dtype=float)

    X = train_pre[features]
    X_test = test_pre[features]

    print(f"\n[Final] Starting {CFG['final_cv_splits']}-fold training (Ordered boosting)...")
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        train_pool = Pool(X_trn, label=y_trn, cat_features=cat_idx)
        valid_pool = Pool(X_val, label=y_val, cat_features=cat_idx)
        test_pool = Pool(X_test, cat_features=cat_idx)

        params = {
            **CFG["base_cat_params"],
            "depth": best_depth,
            "l2_leaf_reg": best_l2,
            "boosting_type": "Ordered"
        }
        model = CatBoostClassifier(iterations=iterations, **params)

        t0 = time.time()
        model.fit(
            train_pool, eval_set=valid_pool, use_best_model=True,
            verbose=log_every_n, early_stopping_rounds=es_rounds
        )
        sec = time.time() - t0

        val_pred = model.predict_proba(valid_pool)[:, 1]
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        best_iter = model.get_best_iteration() or model.tree_count_
        print(f"[Final Fold {fold}] AUC={val_auc:.6f}  best_iter={best_iter}  time={sec/60:.1f}min")

        test_pred += model.predict_proba(test_pool)[:, 1] / CFG["final_cv_splits"]

        fi = model.get_feature_importance(train_pool, type="PredictionValuesChange")
        fi_accum += fi

        del train_pool, valid_pool, test_pool, model, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    fi_df = pd.DataFrame({"feature": features, "importance": fi_accum}).sort_values("importance", ascending=False)

    # Save results
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(out_dir, f"submission_{stamp}.csv")
    oof_path = os.path.join(out_dir, f"oof_{stamp}.csv")
    fi_path = os.path.join(out_dir, f"feature_importance_{stamp}.csv")
    cfg_path = os.path.join(out_dir, f"best_config_{stamp}.json")

    submission = test_pre[[CFG["id_col"]]].copy() if CFG["id_col"] in test_pre.columns else None
    if submission is None:
        raise RuntimeError("test_pre does not contain the ID column; please check preprocessing.")
    submission[CFG["target"]] = test_pred
    submission.to_csv(sub_path, index=False)

    oof_df = train_pre[[CFG["id_col"]]].copy()
    oof_df["oof_pred"] = oof
    oof_df[CFG["target"]] = y
    oof_df.to_csv(oof_path, index=False)

    fi_df.to_csv(fi_path, index=False)

    best_pack = {
        "best_depth": best_depth,
        "best_l2_leaf_reg": best_l2,
        "final_iterations": iterations,
        "final_early_stopping_rounds": es_rounds,
        "final_oof_auc": oof_auc
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(best_pack, f, ensure_ascii=False, indent=2)

    print(f"\n[Final] OOF AUC = {oof_auc:.6f}")
    print(f"Saved outputs:\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n"
          f"- Feature importance: {fi_path}\n- Best config: {cfg_path}")

    return oof_auc, sub_path, oof_path, fi_path, cfg_path


def main():
    t_all = time.time()
    ensure_dir(CFG["out_dir"])

    print("Loading data...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Basic preprocessing (CatBoost native categorical + NaN support)...")
    train_pre, test_pre, y, features, cat_cols, cat_idx = preprocess_for_catboost(
        train, test, CFG["target"], CFG["id_col"]
    )
    print(f"Number of features: {len(features)} (categorical: {len(cat_cols)})")

    # ========== Grid Search ==========
    print("\n[Grid] Starting grid search (3 folds, Plain boosting for speed)...")
    grid_records = []
    for d in CFG["grid_depth"]:
        for l2 in CFG["grid_l2"]:
            params = {
                **CFG["base_cat_params"],
                "depth": d,
                "l2_leaf_reg": l2,
                "boosting_type": "Plain"
            }
            mean_auc, std_auc, mean_best_iter, mean_time = cv_eval_catboost(
                train_pre, y, features, cat_idx, params,
                iterations=CFG["grid_iterations"],
                es_rounds=CFG["grid_es_rounds"],
                n_splits=CFG["grid_cv_splits"],
                log_every_n=CFG["grid_log_every_n"]
            )
            grid_records.append({
                "depth": d,
                "l2_leaf_reg": l2,
                "mean_auc": round(mean_auc, 6),
                "std_auc": round(std_auc, 6),
                "mean_best_iter": int(round(mean_best_iter)),
                "mean_time_s": round(mean_time, 1)
            })
            print(f"[Grid] depth={d}, l2={l2} -> AUC={mean_auc:.6f} ± {std_auc:.6f} "
                  f"(best_iter≈{int(round(mean_best_iter))}, ~{mean_time/60:.1f} min/fold)")

    grid_df = pd.DataFrame(grid_records).sort_values(["mean_auc", "depth"], ascending=[False, True])
    grid_path = os.path.join(CFG["out_dir"], "grid_results.csv")
    grid_df.to_csv(grid_path, index=False)
    print(f"\n[Grid] Results saved: {grid_path}")
    print(grid_df.head(10))

    # Select best parameters
    best_row = grid_df.iloc[0]
    best_depth = int(best_row["depth"])
    best_l2 = float(best_row["l2_leaf_reg"])
    print(f"\n[Grid] Best params: depth={best_depth}, l2_leaf_reg={best_l2}, mean_auc={best_row['mean_auc']:.6f}")

    # ========== Final 5-fold Training ==========
    oof_auc, sub_path, oof_path, fi_path, cfg_path = run_final_training(
        train_pre, y, test_pre, features, cat_idx,
        best_depth=best_depth, best_l2=best_l2,
        out_dir=CFG["out_dir"],
        iterations=CFG["final_iterations"],
        es_rounds=CFG["final_es_rounds"],
        log_every_n=CFG["final_log_every_n"]
    )

    print(f"\nTotal time: {(time.time() - t_all)/60:.1f} minutes")


if __name__ == "__main__":
    main()
