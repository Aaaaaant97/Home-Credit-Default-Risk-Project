#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - Logistic Regression (L2) Baseline + Coarse Grid Search (C)
========================================================================
- Reads application_train.csv / application_test.csv (main tables)
- Preprocessing: categorical One-Hot + numeric median imputation + numeric standardization (sparse-safe)
- 5-fold StratifiedKFold training with LogisticRegression (L2, solver='saga')
- Grid: C ∈ {0.2, 0.5, 1, 2, 5}
- Outputs: submission / OOF / coefficient importance / grid results / best params / run config
"""

import os, gc, json, time, warnings
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

# Optional progress bar
try:
    from tqdm import tqdm
    _use_tqdm = True
except Exception:
    _use_tqdm = False

warnings.filterwarnings("ignore")

# ========== Configuration ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # If you want to use screened F.L.I features, switch the following two lines
    # "train_path": "fli_application_train.csv",
    # "test_path":  "fli_application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_logreg_l2_grid",

    # Logistic Regression base params (C will be tuned)
    "lr_params": {
        "penalty": "l2",
        "solver": "saga",       # friendly to sparse & high-dimensional data
        "max_iter": 2000,
        "tol": 1e-4,
        "n_jobs": -1,
        "class_weight": None,   # or "balanced" for more robustness
        "random_state": 42
    },

    # Grid range (feel free to adjust)
    "grid_C": [0.2, 0.5, 1.0, 2.0, 5.0],
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


def preprocess(train: pd.DataFrame, test: pd.DataFrame, target: str, id_col: str):
    """
    Basic preprocessing:
    - Numeric: fill missing with median, then StandardScaler(with_mean=False) to keep sparsity
    - Categorical: fill missing with '__MISSING__', One-Hot encode (fit on train+test to avoid unseen)
    - Returns: sparse matrices X_train, X_test; y; and feature name list
    """
    y = train[target].astype(int).values
    train_nolab = train.drop(columns=[target])

    combined = pd.concat([train_nolab, test], axis=0, ignore_index=True)

    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols]

    # Missing values
    combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")
    for col in num_cols:
        if col == id_col:
            continue
        if combined[col].isna().any():
            med = combined[col].median()
            combined[col] = combined[col].fillna(med)

    # Numeric standardization (sparse-safe)
    scaler = StandardScaler(with_mean=False)
    X_num = csr_matrix(scaler.fit_transform(combined[num_cols].astype(float)))

    # Categorical One-Hot
    if len(cat_cols) > 0:
        # For scikit-learn >= 1.2, use sparse_output=True; for older versions, use sparse=True
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

        X_cat = ohe.fit_transform(combined[cat_cols].astype(str))
        X_all = hstack([X_num, X_cat], format="csr")
        try:
            ohe_names = list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            # Fallback for very old sklearn
            ohe_names = list(ohe.get_feature_names(cat_cols))
        feature_names = list(num_cols) + ohe_names
    else:
        X_all = X_num
        feature_names = list(num_cols)

    n_train = len(train)
    X_train = X_all[:n_train]
    X_test  = X_all[n_train:]

    return X_train, X_test, y, feature_names, cat_cols, num_cols


def run_cv_logreg(X, y, X_test, feature_names, cfg):
    """5-fold CV for Logistic Regression; returns OOF, test predictions, OOF AUC, and |coef| importance."""
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )

    oof = np.zeros(X.shape[0], dtype=float)
    test_pred = np.zeros(X_test.shape[0], dtype=float)
    coef_importance = np.zeros(len(feature_names), dtype=float)

    print(f"Starting {cfg['n_splits']}-fold cross-validation...")
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = LogisticRegression(**cfg["lr_params"])
        model.fit(X_trn, y_trn)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        print(f"[Fold {fold}] AUC = {val_auc:.6f}")

        test_pred += model.predict_proba(X_test)[:, 1] / cfg["n_splits"]

        coef_importance += np.abs(model.coef_.ravel())

        del model, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")

    coef_importance /= cfg["n_splits"]
    fi = pd.DataFrame({"feature": feature_names, "coef_abs": coef_importance}) \
         .sort_values("coef_abs", ascending=False)
    return oof, test_pred, oof_auc, fi


def grid_search_logreg_C(X, y, X_test, feature_names, base_cfg, C_list, save_csv_path=None):
    """Search only C; other LR params follow base_cfg['lr_params']."""
    records = []
    iterator = C_list
    if _use_tqdm:
        from tqdm import tqdm
        iterator = tqdm(C_list, desc="Grid search (C)")

    best_auc = -1.0
    best_pack = None

    for c in iterator:
        cfg = deepcopy(base_cfg)
        cfg["lr_params"].update({"C": float(c)})

        print(f"\n[Try] C={c}")
        oof, test_pred, oof_auc, fi = run_cv_logreg(X, y, X_test, feature_names, cfg)

        records.append({"C": float(c), "oof_auc": oof_auc})

        if _use_tqdm:
            iterator.set_postfix({"C": c, "auc": f"{oof_auc:.6f}"})

        if oof_auc > best_auc:
            best_auc = oof_auc
            best_pack = {
                "C": float(c),
                "oof_auc": oof_auc,
                "detail": (oof, test_pred, fi, cfg)  # aligned with your LGBM unpack format
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

    print("Preprocessing (One-Hot + median imputation + standardization)...")
    X_train, X_test, y, feature_names, cat_cols, num_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"]
    )
    print(f"Number of features: {len(feature_names)} (categorical columns: {len(cat_cols)})")

    # ====== Coarse grid search over C ======
    grid_csv = os.path.join(CFG["out_dir"], f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    best_pack, tried_df = grid_search_logreg_C(
        X_train, y, X_test, feature_names, CFG, C_list=CFG["grid_C"], save_csv_path=grid_csv
    )

    print("\n[GridSearch] Top-10 (sorted by OOF AUC):")
    print(tried_df.head(10).to_string(index=False))

    print(f"\n[Best] C={best_pack['C']}, OOF AUC={best_pack['oof_auc']:.6f}")

    # ====== Retrain with best C and save outputs ======
    oof_g, test_pred_g, fi_g, best_cfg = best_pack["detail"]  # (oof, test_pred, fi, cfg)

    print("\n[Retrain] Re-training full CV with best C and exporting files...")
    oof, test_pred, oof_auc, fi = run_cv_logreg(
        X_train, y, X_test, feature_names, best_cfg
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

    # Importance
    fi.to_csv(fi_path, index=False)

    # Save run config + best params
    out_cfg = deepcopy(best_cfg)
    out_cfg["best_search"] = {
        "C": best_pack["C"],
        "oof_auc": best_pack["oof_auc"]
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:\n- Grid results: {grid_csv}\n- Submission: {sub_path}\n"
          f"- OOF predictions: {oof_path}\n- Coefficient importance: {fi_path}\n- Best params: {best_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
