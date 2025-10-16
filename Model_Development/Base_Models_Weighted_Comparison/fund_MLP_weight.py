#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - MLP Baseline (Main Table Only) — Imbalance-Safe
=============================================================
- Read application_train.csv / application_test.csv
- Preprocessing: categorical mapped using training-only mapping (LabelEncode style; unseen -> -1)
                 + numerical median imputation (fit on training) + standardization (fit on training)
- 5-fold StratifiedKFold training of Keras MLP
- Imbalance handling: auto_weight / balanced / undersample / none
- Evaluation: ROC-AUC + PR-AUC (AP)
- Outputs: submission / OOF / (first-layer weights as approximate) feature_importance / run configuration
"""

import os, gc, json, time, random
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

# ====== Environment variables (set before importing TF) ======
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"             # Force CPU for stability; comment out to use GPU/Metal
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ========== CONFIG ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # "train_path": "application_train.csv",
    # "test_path":  "application_test.csv"
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_mlp_weight",

    # ====== MLP parameters ======
    "mlp_params": {
        "hidden_layers": [256, 128, 64],
        "dropout": 0.2,
        "batch_norm": True,
        "l2": 1e-6,
        "learning_rate": 1e-3,
        "epochs": 100,
        "batch_size": 4096,
        "patience": 10,
        "reduce_lr_patience": 5,
        "label_smoothing": 0.0
    },

    # ====== Imbalance handling ======
    # method:
    #   - "auto_weight": use class_weight={0:1, 1:neg/pos (capped)} [default]
    #   - "balanced":    use class weights computed as in sklearn's 'balanced' formula
    #   - "undersample": downsample negatives to the specified neg:pos ratio
    #   - "none":        no handling
    "imbalance": {
        "method": "auto_weight",
        "undersample_neg_pos_ratio": 3.0,
        "max_pos_weight_cap": 50.0
    },

    # Whether to report PR-AUC (AP)
    "report_pr_auc": True,
}


# ---------------- Utils ----------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def read_data(train_path, test_path):
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    test = pd.read_csv(test_path)
    if os.path.isfile(train_path):
        train = pd.read_csv(train_path)
    else:
        raise FileNotFoundError(f"Training file not found: {train_path}")
    return train, test

def _compute_pos_weight(y_arr, cap=50.0):
    pos = int(np.sum(y_arr == 1)); neg = int(np.sum(y_arr == 0))
    if pos == 0: return 1.0
    w = neg / max(1, pos)
    if cap is not None: w = min(w, float(cap))
    return float(w)

def _undersample_indices(y_trn, desired_neg_pos_ratio=3.0, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    pos_idx = np.where(y_trn == 1)[0]; neg_idx = np.where(y_trn == 0)[0]
    n_pos = len(pos_idx); n_neg_keep = int(min(len(neg_idx), desired_neg_pos_ratio * n_pos))
    if n_neg_keep <= 0: return np.ones_like(y_trn, dtype=bool)
    keep_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False)
    keep = np.zeros_like(y_trn, dtype=bool); keep[pos_idx] = True; keep[keep_neg] = True
    return keep

def _label_encode_with_train_map(series_train: pd.Series, series_test: pd.Series):
    """Train-only mapping for categories; unseen in test -> -1."""
    st = series_train.astype(str).replace("nan", np.nan).fillna("__MISSING__")
    se = series_test.astype(str).replace("nan", np.nan).fillna("__MISSING__")
    uniq = pd.Series(st.unique())
    mapping = {v: i for i, v in enumerate(uniq)}
    tr_enc = st.map(mapping).fillna(-1).astype(int)
    te_enc = se.map(mapping).fillna(-1).astype(int)
    return tr_enc, te_enc, mapping


# ---------------- Preprocess (no leakage) ----------------
def preprocess(train, test, target, id_col):
    """Categoricals are mapped using training data only; numerical imputation/standardization fitted on training only."""
    y = train[target].astype(int).values
    train_nolab = train.drop(columns=[target]).copy()

    # Column types
    cat_cols = train_nolab.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in train_nolab.columns if c not in cat_cols]

    # Numerical imputation: median from training only
    for col in num_cols:
        if col == id_col: continue
        med = train_nolab[col].median() if pd.api.types.is_numeric_dtype(train_nolab[col]) else None
        if med is not None:
            train_nolab[col] = train_nolab[col].fillna(med)
            test[col] = test[col].fillna(med)

    # Categorical encoding (train-only mapping), unseen in test -> -1
    for col in cat_cols:
        tr_enc, te_enc, _ = _label_encode_with_train_map(train_nolab[col], test[col])
        train_nolab[col] = tr_enc
        test[col] = te_enc

    # Standardization: fit on training only; apply to all features except id
    features = [c for c in train_nolab.columns if c != id_col]
    scaler = StandardScaler()
    train_nolab[features] = scaler.fit_transform(train_nolab[features])
    test[features] = scaler.transform(test[features])

    train_pre = train_nolab.reset_index(drop=True)
    test_pre  = test.reset_index(drop=True)
    return train_pre, test_pre, y, features, cat_cols


# ---------------- Model ----------------
def build_mlp(input_dim: int, cfg: dict) -> keras.Model:
    p = cfg["mlp_params"]
    reg = regularizers.l2(p["l2"]) if p["l2"] and p["l2"] > 0 else None

    inp = keras.Input(shape=(input_dim,), name="features")
    x = inp
    for i, h in enumerate(p["hidden_layers"]):
        x = layers.Dense(h, kernel_regularizer=reg, name=f"dense_{i}")(x)
        if p["batch_norm"]:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        if p["dropout"] and p["dropout"] > 0:
            x = layers.Dropout(p["dropout"], name=f"dropout_{i}")(x)
    out = layers.Dense(1, activation="sigmoid", name="logit")(x)

    model = keras.Model(inputs=inp, outputs=out)
    opt = keras.optimizers.Adam(learning_rate=p["learning_rate"])
    model.compile(
        optimizer=opt,
        loss=keras.losses.BinaryCrossentropy(label_smoothing=p["label_smoothing"]),
        metrics=[keras.metrics.AUC(name="auc")]
    )
    return model


def first_layer_importance(model: keras.Model, feature_names):
    """Use |w| aggregated from the first Dense layer as a rough proxy for feature importance."""
    first_dense = next((layer for layer in model.layers if isinstance(layer, layers.Dense)), None)
    if first_dense is None:
        return pd.DataFrame({"feature": feature_names, "importance": 0.0})
    w, _ = first_dense.get_weights()
    imp = np.abs(w).sum(axis=1)
    df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
    return df


# ---------------- CV Train ----------------
def run_cv_mlp(train_pre, y, test_pre, features, id_col, cfg):
    folds = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])

    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)
    fi_accum = np.zeros(len(features), dtype=float)

    X = train_pre[features].astype(np.float32)
    X_test = test_pre[features].astype(np.float32)

    imb = cfg.get("imbalance", {"method": "none"})
    method = imb.get("method", "none")
    rng = np.random.RandomState(cfg["seed"])

    print(f"Imbalance handling: {method}")
    print(f"Starting {cfg['n_splits']}-fold cross-validation ...")

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        set_seed(cfg["seed"] + fold)
        X_trn_full, y_trn_full = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # ===== Imbalance handling =====
        class_weight = None
        X_trn, y_trn = X_trn_full, y_trn_full

        if method == "undersample":
            mask = _undersample_indices(y_trn_full,
                                        desired_neg_pos_ratio=float(imb.get("undersample_neg_pos_ratio", 3.0)),
                                        rng=rng)
            X_trn = X_trn_full.iloc[mask]
            y_trn = y_trn_full[mask]
            print(f"[Fold {fold}] After undersampling: n={X_trn.shape[0]} (pos={np.sum(y_trn==1)}, neg={np.sum(y_trn==0)})")

        elif method == "auto_weight":
            pw = _compute_pos_weight(y_trn_full, cap=float(imb.get("max_pos_weight_cap", 50.0)))
            class_weight = {0: 1.0, 1: float(pw)}
            print(f"[Fold {fold}] auto_weight: class_weight={{0:1, 1:{pw:.2f}}}")

        elif method == "balanced":
            # Keras does not support class_weight='balanced' directly; compute weights manually as in sklearn.
            n_pos = int(np.sum(y_trn_full == 1)); n_neg = int(np.sum(y_trn_full == 0))
            total = n_pos + n_neg
            w0 = total / (2.0 * max(1, n_neg))
            w1 = total / (2.0 * max(1, n_pos))
            class_weight = {0: float(w0), 1: float(w1)}
            print(f"[Fold {fold}] balanced: class_weight={{0:{w0:.2f}, 1:{w1:.2f}}}")

        else:
            print(f"[Fold {fold}] No imbalance handling")

        # ===== Model =====
        model = build_mlp(input_dim=X.shape[1], cfg=cfg)
        cbs = [
            EarlyStopping(monitor="val_auc", mode="max",
                          patience=cfg["mlp_params"]["patience"],
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_auc", mode="max",
                              patience=cfg["mlp_params"]["reduce_lr_patience"],
                              factor=0.5, min_lr=1e-5, verbose=1)
        ]

        history = model.fit(
            X_trn, y_trn,
            validation_data=(X_val, y_val),
            epochs=cfg["mlp_params"]["epochs"],
            batch_size=cfg["mlp_params"]["batch_size"],
            callbacks=cbs,
            class_weight=class_weight,
            verbose=2
        )

        # Validation predictions
        val_pred = model.predict(X_val, verbose=0).reshape(-1)
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        if cfg.get("report_pr_auc", True):
            val_ap = average_precision_score(y_val, val_pred)
            print(f"[Fold {fold}] AUC = {val_auc:.6f} | AP = {val_ap:.6f} | best_epoch = {np.argmax(history.history['val_auc'])+1}")
        else:
            print(f"[Fold {fold}] AUC = {val_auc:.6f} | best_epoch = {np.argmax(history.history['val_auc'])+1}")

        # Test predictions (averaged across folds)
        test_pred += model.predict(X_test, verbose=0).reshape(-1) / cfg["n_splits"]

        # Simple "importance" accumulation (first-layer |w| sum)
        fi_fold = first_layer_importance(model, features)["importance"].values
        fi_accum += fi_fold

        del model, X_trn, X_val, y_trn, y_val, history
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    if cfg.get("report_pr_auc", True):
        oof_ap = average_precision_score(y, oof)
        print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} | OOF AP = {oof_ap:.6f} ===")
    else:
        print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")

    fi_df = pd.DataFrame({"feature": features, "importance": fi_accum})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return oof, test_pred, oof_auc, fi_df


# ---------------- Main ----------------
def main():
    t0 = time.time()
    set_seed(CFG["seed"])
    ensure_dir(CFG["out_dir"])

    print("Reading data ...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (category mapping, imputation, standardization; all fitted on training only) ...")
    train_pre, test_pre, y, features, cat_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"]
    )
    print(f"Number of features: {len(features)} (categorical columns: {len(cat_cols)})")

    # Label distribution
    pos = int(np.sum(y == 1)); neg = int(np.sum(y == 0))
    print(f"Label distribution: neg={neg}, pos={pos}, neg:pos≈{(neg/max(1,pos)):.2f}:1")

    print("Cross-validation + MLP training ...")
    oof, test_pred, oof_auc, fi = run_cv_mlp(
        train_pre, y, test_pre, features, CFG["id_col"], CFG
    )

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
    oof_df[CFG["TARGET"]] = y
    oof_df.to_csv(oof_path, index=False)

    fi.to_csv(fi_path, index=False)

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n- (First-layer weights) feature importance: {fi_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
