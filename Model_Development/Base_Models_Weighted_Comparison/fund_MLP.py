#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - MLP Baseline (Main Table Only)
============================================
- Reads application_train.csv / application_test.csv
- Preprocessing: categorical LabelEncoding + numeric median imputation + numeric feature standardization
- 5-fold StratifiedKFold training with Keras MLP
- Outputs: submission / OOF predictions / (first-layer weight) feature importance / run configuration
"""

import os, gc, json, time, random
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# ====== Environment variables (set before importing TensorFlow) ======
# For CPU-only stability; comment out if you want to use GPU/Metal on macOS.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ========== Configuration ==========
CFG = {
    "seed": 42,
    "n_splits": 5,
    "target": "TARGET",
    "id_col": "SK_ID_CURR",
    # "train_path": "application_train.csv",
    # "test_path":  "application_test.csv",
    # "train_path": "fli_application_train.csv",
    # "test_path":  "fli_application_test.csv",
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_mlp",

    # ====== MLP hyperparameters ======
    "mlp_params": {
        "hidden_layers": [256, 128, 64],   # hidden layer widths
        "dropout": 0.2,                    # dropout probability
        "batch_norm": True,                # whether to use BatchNorm
        "l2": 1e-6,                        # L2 regularization
        "learning_rate": 1e-3,             # initial learning rate
        "epochs": 100,                     # max epochs (with early stopping)
        "batch_size": 4096,                # large batch for big data
        "patience": 10,                    # EarlyStopping patience
        "reduce_lr_patience": 5,           # ReduceLROnPlateau patience
        "label_smoothing": 0.0             # e.g., 0.01 for slight smoothing
    }
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_data(train_path, test_path):
    """Read training and test CSV files."""
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    test = pd.read_csv(test_path)

    if os.path.isfile(train_path):
        train = pd.read_csv(train_path)
    else:
        raise FileNotFoundError(f"Training file not found: {train_path}")
    return train, test


def preprocess(train, test, target, id_col):
    """
    Basic preprocessing: categorical encoding + missing value imputation + standardization.
    - LabelEncode categorical columns (fit on combined train+test to avoid unseen categories).
    - Numeric: fill missing with median; standardize (fit on training set only).
    """
    y = train[target].astype(int).values
    train = train.drop(columns=[target])

    # Combine for consistent category handling (avoids unseen categories in test)
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # Identify column types
    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols]

    # Missing value handling
    combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")
    for col in num_cols:
        if col == id_col:
            continue
        if combined[col].isna().any():
            median_val = combined[col].median()
            combined[col] = combined[col].fillna(median_val)

    # Label Encoding (fit on train+test combined)
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Split back to train/test
    train_pre = combined.iloc[:len(train)].reset_index(drop=True)
    test_pre  = combined.iloc[len(train):].reset_index(drop=True)

    # Standardize numeric + encoded categorical features (fit on train to avoid leakage)
    features = [c for c in train_pre.columns if c != id_col]
    scaler = StandardScaler()
    scaler.fit(train_pre[features])
    train_pre[features] = scaler.transform(train_pre[features])
    test_pre[features]  = scaler.transform(test_pre[features])

    return train_pre, test_pre, y, features, cat_cols


def build_mlp(input_dim: int, cfg: dict) -> keras.Model:
    """Build a plain MLP with optional BatchNorm, Dropout, and L2 regularization."""
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
    """
    Approximate feature importance using |weights| of the first Dense layer (sum over outputs).
    This is a heuristic and should be interpreted cautiously.
    """
    first_dense = None
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            first_dense = layer
            break
    if first_dense is None:
        return pd.DataFrame({"feature": feature_names, "importance": 0.0})

    w, b = first_dense.get_weights()
    # importance_i = sum_j |w_ij|
    imp = np.abs(w).sum(axis=1)
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False)
    return df


def run_cv_mlp(train_pre, y, test_pre, features, id_col, cfg):
    """Cross-validated MLP training and evaluation with simple weight-based importances."""
    folds = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])

    oof = np.zeros(len(train_pre), dtype=float)
    test_pred = np.zeros(len(test_pre), dtype=float)
    fi_accum = np.zeros(len(features), dtype=float)

    X = train_pre[features].astype(np.float32)
    X_test = test_pre[features].astype(np.float32)

    # Class weights (mitigate class imbalance)
    classes = np.array([0, 1], dtype=int)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw_map = {int(c): float(w) for c, w in zip(classes, class_weights)}

    print(f"Starting {cfg['n_splits']}-fold cross-validation ...")
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        set_seed(cfg["seed"] + fold)  # ensure deterministic behavior per fold
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

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
            class_weight=cw_map,
            verbose=2
        )

        # Validation predictions
        val_pred = model.predict(X_val, verbose=0).reshape(-1)
        oof[val_idx] = val_pred
        val_auc = roc_auc_score(y_val, val_pred)
        best_epoch = np.argmax(history.history["val_auc"]) + 1
        print(f"[Fold {fold}] AUC = {val_auc:.6f}  best_epoch = {best_epoch}")

        # Test predictions (average over folds)
        test_pred += model.predict(X_test, verbose=0).reshape(-1) / cfg["n_splits"]

        # Accumulate simple "importance" from first-layer weights
        fi_fold = first_layer_importance(model, features)["importance"].values
        fi_accum += fi_fold

        del model, X_trn, X_val, y_trn, y_val, history
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n=== CV AUC (OOF) = {oof_auc:.6f} ===")

    fi_df = pd.DataFrame({"feature": features, "importance": fi_accum})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return oof, test_pred, oof_auc, fi_df


def main():
    t0 = time.time()
    set_seed(CFG["seed"])
    ensure_dir(CFG["out_dir"])

    print("Reading data...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (categorical encoding, missing value imputation, standardization)...")
    train_pre, test_pre, y, features, cat_cols = preprocess(
        train, test, CFG["target"], CFG["id_col"]
    )
    print(f"Feature count: {len(features)} (including {len(cat_cols)} categorical columns)")

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
    oof_df[CFG["target"]] = y
    oof_df.to_csv(oof_path, index=False)

    fi.to_csv(fi_path, index=False)

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n"
          f"- (first-layer weight) feature importance: {fi_path}\n- Run configuration: {cfg_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
