#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - Logistic Regression (BIN + WOE, No Grid Search)
==============================================================
- Reads application_train.csv / application_test.csv (main tables)
- Preprocessing: numeric quantile binning (qcut) + categorical bins -> WOE encoding
  (based strictly on training set, with Laplace smoothing)
- 5-fold StratifiedKFold training of LogisticRegression (L2, solver='saga')
- Outputs: submission / OOF predictions / coefficient importance / run configuration
"""

import os, gc, json, time, warnings
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ========== Configuration ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # "train_path": "application_train.csv",
    # "test_path":  "application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_logreg_bin_nogrid",

    # Binning & WOE
    "num_bins": 10,
    "min_bin_size": 200,
    "laplace_alpha": 0.5,

    # Logistic Regression fixed parameters (no grid search)
    "lr_params": {
        "penalty": "l2",
        "solver": "saga",
        "C": 1.0,              # fixed value (can be changed if needed)
        "max_iter": 2000,
        "tol": 1e-4,
        "n_jobs": -1,
        "class_weight": None,  # or "balanced"
        "random_state": 42
    },
}


def ensure_dir(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def read_data(train_path, test_path):
    """Read training and test datasets."""
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    test = pd.read_csv(test_path)
    if os.path.isfile(train_path):
        train = pd.read_csv(train_path)
    else:
        raise FileNotFoundError(f"Train file not found: {train_path}")
    return train, test


def _safe_qcut(s: pd.Series, q: int, min_bin_size: int):
    """Perform quantile binning safely with minimum bin size constraint."""
    s_nonnull = s.dropna()
    bins_try = q
    while bins_try >= 2:
        try:
            cats = pd.qcut(s_nonnull, q=bins_try, duplicates="drop")
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
    out = pd.Series(index=s.index, dtype="object")
    out.loc[s.isna()] = "__BIN_NA__"
    out.loc[s.notna()] = "__BIN_ALL__"
    return out, 1


def _woe_from_counts(pos, neg, pos_tot, neg_tot, alpha=0.5):
    """Compute Weight of Evidence (WOE) from positive/negative counts."""
    gi = neg + alpha  # good = y==0
    bi = pos + alpha  # bad  = y==1
    gA = neg_tot + alpha
    bA = pos_tot + alpha
    return np.log((gi / bi) / (gA / bA))


def _fit_feature_woe(series: pd.Series, y: np.ndarray, alpha=0.5):
    """Compute WOE mapping for one feature."""
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
    global_woe = _woe_from_counts(pos_tot, neg_tot, pos_tot, neg_tot, alpha=alpha)
    return woe_map, float(global_woe)


def preprocess_bin_woe(train: pd.DataFrame, test: pd.DataFrame, target: str, id_col: str,
                       num_bins: int, min_bin_size: int, alpha: float):
    """Apply quantile binning and WOE encoding to numeric and categorical features."""
    y = train[target].astype(int).values
    train_nolab = train.drop(columns=[target])

    cat_cols = train_nolab.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in train_nolab.columns if (c not in cat_cols and c != id_col)]

    all_df = pd.concat([train_nolab, test], axis=0, ignore_index=True)
    n_train = len(train_nolab)

    # Numeric binning
    binned_num = {}
    for col in num_cols:
        bins_col, _ = _safe_qcut(all_df[col], q=num_bins, min_bin_size=min_bin_size)
        binned_num[col] = bins_col

    # Treat categorical columns as "bins"
    binned_cat = {}
    for col in cat_cols:
        s = all_df[col].astype(str)
        s = s.fillna("__CAT_NA__")
        s.loc[s == "nan"] = "__CAT_NA__"
        binned_cat[col] = s.astype("object")

    # Compute WOE based on training data and apply to both train/test
    feature_names = []
    X_all = pd.DataFrame(index=all_df.index)

    for col in num_cols:
        series = binned_num[col].iloc[:n_train]
        wmap, global_woe = _fit_feature_woe(series, y, alpha=alpha)
        X_all[col] = binned_num[col].map(wmap).fillna(global_woe).astype(float)
        feature_names.append(col)

    for col in cat_cols:
        series = binned_cat[col].iloc[:n_train]
        wmap, global_woe = _fit_feature_woe(series, y, alpha=alpha)
        X_all[col] = binned_cat[col].map(wmap).fillna(global_woe).astype(float)
        feature_names.append(col)

    # Optional standardization
    scaler = StandardScaler()
    X_all[feature_names] = scaler.fit_transform(X_all[feature_names])

    X_train = X_all.iloc[:n_train][feature_names].values
    X_test  = X_all.iloc[n_train:][feature_names].values
    return X_train, X_test, y, feature_names, cat_cols, num_cols


def run_cv_logreg(X, y, X_test, feature_names, cfg):
    """Perform cross-validation training and evaluation."""
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    )
    oof = np.zeros(X.shape[0], dtype=float)
    test_pred = np.zeros(X_test.shape[0], dtype=float)
    coef_importance = np.zeros(len(feature_names), dtype=float)

    print(f"Starting {cfg['n_splits']}-fold cross-validation ...")
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


def main():
    """Main execution pipeline."""
    t0 = time.time()
    ensure_dir(CFG["out_dir"])

    print("Reading data...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Performing binning + WOE encoding (based strictly on training data)...")
    X_train, X_test, y, feature_names, cat_cols, num_cols = preprocess_bin_woe(
        train, test, CFG["target"], CFG["id_col"],
        num_bins=CFG["num_bins"],
        min_bin_size=CFG["min_bin_size"],
        alpha=CFG["laplace_alpha"]
    )
    print(f"Feature count: {len(feature_names)} (numeric: {len(num_cols)} / categorical: {len(cat_cols)})")

    print("\n[Train] Logistic Regression with fixed C, exporting outputs ...")
    oof, test_pred, oof_auc, fi = run_cv_logreg(
        X_train, y, X_test, feature_names, CFG
    )

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

    print(f"\nSaved files:\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n"
          f"- Coefficient importance: {fi_path}\n- Run config: {cfg_path}")
    print(f"Total time elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
