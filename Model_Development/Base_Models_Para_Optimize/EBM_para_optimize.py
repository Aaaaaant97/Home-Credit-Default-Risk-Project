#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - EBM Baseline + Coarse Grid Search
(interactions × max_bins × learning_rate × max_rounds)
- Uses only the main tables: application_train.csv / application_test.csv
- Outputs: submission / OOF / term_importance / run_config / grid_records
- Compatible with interpret v0.5.0 (term_names / term_importances API)
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

# Optional progress bar
try:
    from tqdm import tqdm
    _use_tqdm = True
except Exception:
    _use_tqdm = False

from interpret.glassbox import ExplainableBoostingClassifier


# ========== Global Configuration ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",

    # "train_path": "fli_application_train.csv",
    # "test_path":  "fli_application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",

    "out_dir": "baseline_ebm_grid",

    # Base EBM parameters (some overridden during grid search)
    "ebm_params": {
        "outer_bags": 2,
        "interactions": 8,         # overridden in grid search
        "learning_rate": 0.05,     # overridden in grid search
        "max_bins": 128,           # overridden in grid search
        "max_interaction_bins": 64,
        "max_rounds": 4000,        # overridden in grid search
        "min_samples_leaf": 2,
        "validation_size": 0.1,
        "early_stopping_rounds": 200,
        "random_state": 42
    },

    # ===== Coarse Grid Search Ranges =====
    "grid_interactions": [0, 8],
    "grid_max_bins": [64, 128],
    "grid_lr": [0.03, 0.05],
    "grid_max_rounds": [2000, 4000],
}


# ---------- Basic Utilities ----------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def read_data(train_path: str, test_path: str):
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    test = pd.read_csv(test_path)

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    train = pd.read_csv(train_path)
    return train, test


def preprocess(train: pd.DataFrame, test: pd.DataFrame, target: str, id_col: str):
    """Category imputation + numeric median fill + LabelEncoding (consistent with LGBM style)."""
    y = train[target].astype(int).values
    train = train.drop(columns=[target])

    combined = pd.concat([train, test], axis=0, ignore_index=True)

    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols]

    combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")
    for col in num_cols:
        if col != id_col and combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())

    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    train_pre = combined.iloc[:len(train)]
    test_pre = combined.iloc[len(train):].reset_index(drop=True)

    features = [c for c in train_pre.columns if c != id_col]
    return train_pre, test_pre, y, features, cat_cols


# ---------- interpret Compatibility ----------
def _get_term_names(model) -> list:
    names = getattr(model, "term_names_", None)
    if names is None:
        names = getattr(model, "term_names", None)
        if callable(names):
            names = names()
    return list(names)


def _get_term_importances(model) -> np.ndarray:
    imps = getattr(model, "term_importances_", None)
    if imps is None:
        imps = getattr(model, "term_importances", None)
        if callable(imps):
            imps = imps()
    return np.asarray(imps, dtype=float)


# ---------- EBM Training ----------
def run_cv_ebm(train_pre, y, test_pre, features, id_col, cfg):
    """5-fold CV training for EBM; returns OOF, test prediction, OOF AUC, and term importances."""
    folds = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)

    all_term_names, term_importance_sum = None, None

    X, X_test = train_pre[features], test_pre[features]
    print(f"Starting {cfg['n_splits']}-fold cross-validation...")

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = ExplainableBoostingClassifier(**cfg["ebm_params"])
        model.fit(X_trn, y_trn)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        print(f"[Fold {fold}] AUC = {val_auc:.6f}")

        test_pred += model.predict_proba(X_test)[:, 1] / cfg["n_splits"]

        term_names = _get_term_names(model)
        term_imps = _get_term_importances(model)

        if all_term_names is None:
            all_term_names = term_names
            term_importance_sum = np.array(term_imps, dtype=float)
        else:
            if term_names != all_term_names:
                m = dict(zip(term_names, term_imps))
                term_importance_sum += np.array([m.get(n, 0.0) for n in all_term_names], dtype=float)
            else:
                term_importance_sum += np.array(term_imps, dtype=float)

        del model, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")

    term_importance_df = pd.DataFrame({
        "term": all_term_names,
        "cv_importance_sum": term_importance_sum
    }).sort_values("cv_importance_sum", ascending=False).reset_index(drop=True)

    return oof, test_pred, oof_auc, term_importance_df


def grid_search_ebm(train_pre, y, test_pre, features, id_col, base_cfg,
                    grid_interactions, grid_max_bins, grid_lr, grid_max_rounds,
                    save_csv_path=None):
    """
    Coarse grid search over (interactions × max_bins × learning_rate × max_rounds).
    Other parameters follow base_cfg['ebm_params']; early stopping controls iterations.
    """
    combos = list(product(grid_interactions, grid_max_bins, grid_lr, grid_max_rounds))
    records = []

    iterator = combos
    if _use_tqdm:
        iterator = tqdm(combos, desc="Grid search (EBM)", leave=False)

    best_auc = -1.0
    best_pack = None

    for inter, bins, lr, rounds in iterator:
        cfg = deepcopy(base_cfg)
        cfg["ebm_params"].update({
            "interactions": inter,
            "max_bins": bins,
            "learning_rate": lr,
            "max_rounds": rounds,
            "outer_bags": cfg["ebm_params"].get("outer_bags", 2),
            "random_state": cfg["ebm_params"].get("random_state", 42),
            "early_stopping_rounds": cfg["ebm_params"].get("early_stopping_rounds", 200),
            "validation_size": cfg["ebm_params"].get("validation_size", 0.1),
        })

        print(f"\n[Try] interactions={inter}, max_bins={bins}, lr={lr}, max_rounds={rounds}")
        oof, test_pred, oof_auc, term_imp = run_cv_ebm(
            train_pre, y, test_pre, features, id_col, cfg
        )

        records.append({
            "interactions": inter,
            "max_bins": bins,
            "learning_rate": lr,
            "max_rounds": rounds,
            "oof_auc": oof_auc
        })

        if _use_tqdm:
            iterator.set_postfix({
                "inter": inter, "bins": bins, "lr": lr,
                "rounds": rounds, "auc": f"{oof_auc:.6f}"
            })

        if oof_auc > best_auc:
            best_auc = oof_auc
            best_pack = {
                "hp": {
                    "interactions": inter,
                    "max_bins": bins,
                    "learning_rate": lr,
                    "max_rounds": rounds
                },
                "oof_auc": oof_auc,
                "detail": (oof, test_pred, term_imp, cfg)
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

    print("Preprocessing (encoding categories and imputing missing values)...")
    train_pre, test_pre, y, features, cat_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"]
    )
    print(f"Number of features: {len(features)} (categorical: {len(cat_cols)})")

    # ====== Coarse Grid Search ======
    grid_csv = os.path.join(CFG["out_dir"], f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    best_pack, tried_df = grid_search_ebm(
        train_pre, y, test_pre, features, CFG["id_col"], CFG,
        grid_interactions=CFG["grid_interactions"],
        grid_max_bins=CFG["grid_max_bins"],
        grid_lr=CFG["grid_lr"],
        grid_max_rounds=CFG["grid_max_rounds"],
        save_csv_path=grid_csv
    )

    print("\n[GridSearch] Top-10 (sorted by OOF AUC):")
    print(tried_df.head(10).to_string(index=False))

    hp = best_pack["hp"]
    print(f"\n[Best] interactions={hp['interactions']}, max_bins={hp['max_bins']}, "
          f"lr={hp['learning_rate']}, max_rounds={hp['max_rounds']}, "
          f"OOF AUC={best_pack['oof_auc']:.6f}")

    # ====== Retrain with Best Parameters and Save Outputs ======
    oof_g, test_pred_g, term_imp_g, best_cfg = best_pack["detail"]

    print("\n[Retrain] Re-training with best parameters and exporting results...")
    oof, test_pred, oof_auc, term_imp = run_cv_ebm(
        train_pre, y, test_pre, features, CFG["id_col"], best_cfg
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(CFG["out_dir"], f"submission_{stamp}.csv")
    oof_path = os.path.join(CFG["out_dir"], f"oof_{stamp}.csv")
    imp_path = os.path.join(CFG["out_dir"], f"ebm_terms_importance_{stamp}.csv")
    cfg_path = os.path.join(CFG["out_dir"], f"run_config_{stamp}.json")
    best_path = os.path.join(CFG["out_dir"], f"best_params_{stamp}.json")

    # Save submission
    submission = test[[CFG["id_col"]]].copy()
    submission[CFG["target"]] = test_pred
    submission.to_csv(sub_path, index=False)

    # Save OOF predictions
    oof_df = train[[CFG["id_col"]]].copy()
    oof_df["oof_pred"] = oof
    oof_df[CFG["target"]] = y
    oof_df.to_csv(oof_path, index=False)

    # Save term importances
    term_imp.to_csv(imp_path, index=False)

    # Save configurations
    out_cfg = deepcopy(best_cfg)
    out_cfg["best_search"] = {**hp, "oof_auc": best_pack["oof_auc"]}
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print(f"\nSaved files:\n- Grid results: {grid_csv}\n- Submission: {sub_path}\n"
          f"- OOF predictions: {oof_path}\n- EBM term importance: {imp_path}\n- Best params: {best_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
