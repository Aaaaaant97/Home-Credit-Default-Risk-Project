#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - Logistic Regression (BIN: Quantile Binning + WOE) + Coarse Grid Search (C)
========================================================================================
- Reads application_train.csv / application_test.csv (main tables)
- Preprocessing:
    * Numeric: quantile binning (qcut) -> compute WOE on training set (smoothed)
    * Categorical: compute WOE per category on training set (smoothed)
    * Unseen/test bins: fall back to the feature-level global WOE
- 5-fold StratifiedKFold training with LogisticRegression (L2, solver='saga')
- Grid: C ∈ {0.2, 0.5, 1, 2, 5}
- Outputs: submission / OOF / coefficient importance / grid results / best params / run config
"""

import os, gc, json, time, warnings
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
    # "train_path": "fli_application_train.csv",
    # "test_path":  "fli_application_test.csv"
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_logreg_bin_grid",

    # Binning & WOE
    "num_bins": 10,             # number of quantile bins for numeric columns
    "min_bin_size": 200,        # if tiny bins appear, reduce bins or fallback
    "laplace_alpha": 0.5,       # Laplace smoothing for WOE

    # Logistic Regression base params (C will be tuned)
    "lr_params": {
        "penalty": "l2",
        "solver": "saga",
        "max_iter": 2000,
        "tol": 1e-4,
        "n_jobs": -1,
        "class_weight": None,   # or "balanced"
        "random_state": 42
    },

    # Grid range
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


def _safe_qcut(s: pd.Series, q: int, min_bin_size: int):
    """Quantile binning for numeric columns; if qcut fails (many ties/too small bins),
    gradually reduce the number of bins; if still failing, return a 2-level NA/non-NA scheme."""
    s_nonnull = s.dropna()
    bins_try = q
    while bins_try >= 2:
        try:
            cats = pd.qcut(s_nonnull, q=bins_try, duplicates="drop")
            # enforce minimum bin size
            vc = cats.value_counts()
            if (vc.min() < min_bin_size) and bins_try > 2:
                bins_try -= 1
                continue
            out = pd.Series(index=s.index, dtype="object")
            out.loc[s_nonnull.index] = cats.astype(str).values
            out.loc[s.isna()] = "__BIN_NA__"
            return out, bins_try
        except Exception:
            bins_try -= 1
    # Fallback: no real binning, only NA vs non-NA
    out = pd.Series(index=s.index, dtype="object")
    out.loc[s.isna()] = "__BIN_NA__"
    out.loc[s.notna()] = "__BIN_ALL__"
    return out, 1


def _woe_from_counts(pos, neg, pos_tot, neg_tot, alpha=0.5):
    # Convention: TARGET=1 = "bad" (default), TARGET=0 = "good"
    # WOE = ln( (good_i / bad_i) / (good_all / bad_all) )
    gi = neg + alpha     # good = label==0
    bi = pos + alpha     # bad  = label==1
    gA = neg_tot + alpha
    bA = pos_tot + alpha
    return np.log((gi / bi) / (gA / bA))


def _fit_feature_woe(series: pd.Series, y: np.ndarray, alpha=0.5):
    """Given a discretized feature (string bins), compute per-bin WOE on the training set
    and also the global WOE for fallback. Returns (mapping dict, global_woe)."""
    df = pd.DataFrame({"bin": series, "y": y})
    grp = df.groupby("bin")["y"]
    pos = grp.sum()
    cnt = grp.count()
    neg = cnt - pos

    pos_tot = pos.sum()
    neg_tot = neg.sum()

    woe_map = {}
    for b in cnt.index:
        w = _woe_from_counts(pos.get(b, 0.0), neg.get(b, 0.0), pos_tot, neg_tot, alpha=alpha)
        woe_map[b] = float(w)

    # Global WOE (used for unseen bins)
    global_woe = _woe_from_counts(pos_tot, neg_tot, pos_tot, neg_tot, alpha=alpha)
    return woe_map, float(global_woe)


def preprocess_bin_woe(train: pd.DataFrame, test: pd.DataFrame, target: str, id_col: str,
                       num_bins: int, min_bin_size: int, alpha: float):
    """Quantile binning + WOE encoding. All statistics are computed strictly on the training set to avoid leakage."""
    y = train[target].astype(int).values
    train_nolab = train.drop(columns=[target])

    # Column types
    cat_cols = train_nolab.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in train_nolab.columns if (c not in cat_cols and c != id_col)]

    # Concatenate only to unify transform; we will compute stats on training slice
    all_df = pd.concat([train_nolab, test], axis=0, ignore_index=True)
    n_train = len(train_nolab)

    # === Numeric: quantile binning ===
    binned_num = {}
    used_bins = {}
    for col in num_cols:
        bins_col, bins_used = _safe_qcut(all_df[col], q=num_bins, min_bin_size=min_bin_size)
        binned_num[col] = bins_col
        used_bins[col] = bins_used

    # === Categorical: treat original categories as bins; fill missing with __CAT_NA__ ===
    binned_cat = {}
    for col in cat_cols:
        s = all_df[col].astype(str)
        s = s.fillna("__CAT_NA__")
        s.loc[s == "nan"] = "__CAT_NA__"
        binned_cat[col] = s.astype("object")

    # === For each feature, compute per-bin WOE on the training slice and map to train+test ===
    feature_names = []
    X_all = pd.DataFrame(index=all_df.index)

    for col in num_cols:
        series = binned_num[col].iloc[:n_train]    # training only
        wmap, global_woe = _fit_feature_woe(series, y, alpha=alpha)
        full_bins = binned_num[col]
        X_all[col] = full_bins.map(wmap).fillna(global_woe).astype(float)
        feature_names.append(col)

    for col in cat_cols:
        series = binned_cat[col].iloc[:n_train]
        wmap, global_woe = _fit_feature_woe(series, y, alpha=alpha)
        full_bins = binned_cat[col]
        X_all[col] = full_bins.map(wmap).fillna(global_woe).astype(float)
        feature_names.append(col)

    # Optional: standardize (helps some LR solvers; WOE is already log-ratio but scaling can stabilize)
    scaler = StandardScaler()
    X_all[feature_names] = scaler.fit_transform(X_all[feature_names])

    X_train = X_all.iloc[:n_train][feature_names].values
    X_test  = X_all.iloc[n_train:][feature_names].values

    return X_train, X_test, y, feature_names, cat_cols, num_cols


def run_cv_logreg(X, y, X_test, feature_names, cfg):
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

    print("Binning + WOE encoding (strictly based on training statistics)...")
    X_train, X_test, y, feature_names, cat_cols, num_cols = preprocess_bin_woe(
        train, test, CFG["target"], CFG["id_col"],
        num_bins=CFG["num_bins"],
        min_bin_size=CFG["min_bin_size"],
        alpha=CFG["laplace_alpha"]
    )
    print(f"Number of features: {len(feature_names)} (numeric: {len(num_cols)} / categorical: {len(cat_cols)})")

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

    submission = test[[CFG["id_col"]]].copy()
    submission[CFG["target"]] = test_pred
    submission.to_csv(sub_path, index=False)

    oof_df = train[[CFG["id_col"]]].copy()
    oof_df["oof_pred"] = oof
    oof_df[CFG["target"]] = y
    oof_df.to_csv(oof_path, index=False)

    fi.to_csv(fi_path, index=False)

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
