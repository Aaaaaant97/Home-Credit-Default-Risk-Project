#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Home Credit - MLP Baseline + Coarse Grid Search (width × depth × dropout × lr)
- Uses only the main tables application_train.csv / application_test.csv
- Preprocessing: LabelEncode + median imputation + StandardScaler
- 5-fold StratifiedKFold, OOF AUC
- Coarse grid search over architecture and key optimizer hyperparameters
- Retrain with best params and export submission / OOF / (1st-layer weight) FI / configs
"""

import os, gc, json, time, random
from datetime import datetime
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# ====== Environment variables (set before importing TF) ======
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # On Apple Silicon this maps to Metal
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
    "train_path": "../Features/All_application_train.csv",
    "test_path": "../Features/All_application_test.csv",
    "out_dir": "baseline_mlp_grid",

    # Training controls (fixed)
    "epochs": 100,
    "patience": 10,
    "reduce_lr_patience": 5,
    "batch_norm": True,
    "l2": 1e-6,                 # can also be added to the grid if desired
    "label_smoothing": 0.0,

    # ====== Grid ranges (adjust as needed) ======
    "grid_width":  [128, 256],   # units per hidden layer
    "grid_depth":  [2, 3],       # number of hidden layers
    "grid_dropout":[0.1, 0.2],   # dropout rate
    "grid_lr":     [1e-3, 5e-4], # initial learning rate
    "grid_batch":  [4096],       # can add 8192 if you also want to search batch size
}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_data(train_path, test_path):
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    test = pd.read_csv(test_path)
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    train = pd.read_csv(train_path)
    return train, test

def preprocess(train, test, target, id_col):
    """LabelEncode + median imputation + StandardScaler (applied to all numeric/encoded columns)."""
    y = train[target].astype(int).values
    train = train.drop(columns=[target])

    combined = pd.concat([train, test], axis=0, ignore_index=True)

    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols]

    combined[cat_cols] = combined[cat_cols].fillna("__MISSING__")
    for col in num_cols:
        if col == id_col: 
            continue
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())

    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    train_pre = combined.iloc[:len(train)].reset_index(drop=True)
    test_pre  = combined.iloc[len(train):].reset_index(drop=True)

    features = [c for c in train_pre.columns if c != id_col]
    scaler = StandardScaler()
    scaler.fit(train_pre[features])
    train_pre[features] = scaler.transform(train_pre[features])
    test_pre[features]  = scaler.transform(test_pre[features])

    return train_pre, test_pre, y, features, cat_cols

def build_mlp(input_dim: int, hidden_layers, dropout, lr, batch_norm=True, l2=0.0, label_smoothing=0.0):
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    inp = keras.Input(shape=(input_dim,), name="features")
    x = inp
    for i, h in enumerate(hidden_layers):
        x = layers.Dense(h, kernel_regularizer=reg, name=f"dense_{i}")(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i}")(x)
    out = layers.Dense(1, activation="sigmoid", name="logit")(x)
    model = keras.Model(inputs=inp, outputs=out)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[keras.metrics.AUC(name="auc")]
    )
    return model

def first_layer_importance(model: keras.Model, feature_names):
    """Extract absolute-sum weights of the first Dense layer as feature importance."""
    first_dense = next((ly for ly in model.layers if isinstance(ly, layers.Dense)), None)
    if first_dense is None:
        return pd.DataFrame({"feature": feature_names, "importance": 0.0})
    w, _ = first_dense.get_weights()
    imp = np.abs(w).sum(axis=1)
    return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)

def run_cv_mlp(train_pre, y, test_pre, features, cfg_mlp):
    folds = StratifiedKFold(n_splits=cfg_mlp["n_splits"], shuffle=True, random_state=cfg_mlp["seed"])
    X = train_pre[features].astype(np.float32)
    X_test = test_pre[features].astype(np.float32)

    classes = np.array([0, 1], dtype=int)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw_map = {int(c): float(w) for c, w in zip(classes, class_weights)}

    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fi_accum = np.zeros(len(features), dtype=float)

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
        set_seed(cfg_mlp["seed"] + fold)
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = build_mlp(
            input_dim=X.shape[1],
            hidden_layers=cfg_mlp["hidden_layers"],
            dropout=cfg_mlp["dropout"],
            lr=cfg_mlp["learning_rate"],
            batch_norm=cfg_mlp["batch_norm"],
            l2=cfg_mlp["l2"],
            label_smoothing=cfg_mlp.get("label_smoothing", 0.0)
        )

        cbs = [
            EarlyStopping(monitor="val_auc", mode="max", patience=cfg_mlp["patience"], restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_auc", mode="max", patience=cfg_mlp["reduce_lr_patience"], factor=0.5, min_lr=1e-5, verbose=0)
        ]

        model.fit(
            X_trn, y_trn,
            validation_data=(X_val, y_val),
            epochs=cfg_mlp["epochs"],
            batch_size=cfg_mlp["batch_size"],
            callbacks=cbs,
            class_weight=cw_map,
            verbose=0
        )

        val_pred = model.predict(X_val, verbose=0).reshape(-1)
        oof[val_idx] = val_pred
        auc = roc_auc_score(y_val, val_pred)
        print(f"[Fold {fold}] AUC = {auc:.6f}")

        test_pred += model.predict(X_test, verbose=0).reshape(-1) / cfg_mlp["n_splits"]

        fi_fold = first_layer_importance(model, features)["importance"].values
        fi_accum += fi_fold

        del model, X_trn, X_val, y_trn, y_val
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    fi_df = pd.DataFrame({"feature": features, "importance": fi_accum}).sort_values("importance", ascending=False).reset_index(drop=True)
    return oof, test_pred, oof_auc, fi_df

def grid_search_mlp(train_pre, y, test_pre, features, base_cfg, save_csv_path=None):
    """
    Search over: width × depth × dropout × lr × batch_size.
    Other factors (BN, L2, label_smoothing, epochs, patience) remain fixed.
    """
    cand = list(product(
        base_cfg["grid_width"],
        base_cfg["grid_depth"],
        base_cfg["grid_dropout"],
        base_cfg["grid_lr"],
        base_cfg["grid_batch"]
    ))

    iterator = tqdm(cand, desc="Grid search (MLP)") if _use_tqdm else cand
    records = []
    best_auc, best_pack = -1.0, None

    for width, depth, dropout, lr, batch in iterator:
        cfg = deepcopy(base_cfg)
        cfg.update({
            "hidden_layers": [width] * depth,
            "dropout": dropout,
            "learning_rate": lr,
            "batch_size": batch
        })

        print(f"\n[Try] width={width}, depth={depth}, dropout={dropout}, lr={lr:g}, batch={batch}")
        oof, test_pred, oof_auc, fi = run_cv_mlp(
            train_pre, y, test_pre, features, cfg
        )
        records.append({
            "width": width, "depth": depth, "dropout": dropout,
            "learning_rate": lr, "batch_size": batch,
            "oof_auc": oof_auc
        })
        if _use_tqdm:
            iterator.set_postfix({"auc": f"{oof_auc:.6f}"})

        if oof_auc > best_auc:
            best_auc = oof_auc
            best_pack = {
                "width": width, "depth": depth, "dropout": dropout,
                "learning_rate": lr, "batch_size": batch,
                "oof_auc": oof_auc,
                "detail": (oof, test_pred, fi, cfg)  # same structure as the LGBM script
            }
        gc.collect()

    tried_df = pd.DataFrame(records).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    if save_csv_path:
        tried_df.to_csv(save_csv_path, index=False)
    return best_pack, tried_df

def main():
    t0 = time.time()
    set_seed(CFG["seed"])
    ensure_dir(CFG["out_dir"])

    print("Loading data...")
    train, test = read_data(CFG["train_path"], CFG["test_path"])
    print(f"train: {train.shape}, test: {test.shape}")

    print("Preprocessing (LabelEncode, median imputation, standardization)...")
    train_pre, test_pre, y, features, cat_cols = preprocess(train, test, CFG["target"], CFG["id_col"])
    print(f"Number of features: {len(features)} (categorical columns: {len(cat_cols)})")

    # ====== Coarse grid search ======
    grid_csv = os.path.join(CFG["out_dir"], f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    best_pack, tried_df = grid_search_mlp(
        train_pre, y, test_pre, features, CFG, save_csv_path=grid_csv
    )

    print("\n[GridSearch] Top-10 (sorted by OOF AUC):")
    print(tried_df.head(10).to_string(index=False))
    print(f"\n[Best] width={best_pack['width']}, depth={best_pack['depth']}, "
          f"dropout={best_pack['dropout']}, lr={best_pack['learning_rate']}, "
          f"batch_size={best_pack['batch_size']}, OOF AUC={best_pack['oof_auc']:.6f}")

    # ====== Retrain with best params and save outputs (same format as LGBM script) ======
    oof_g, test_pred_g, fi_g, best_cfg = best_pack["detail"]

    print("\n[Retrain] Re-training full CV with best params and exporting files...")
    oof, test_pred, oof_auc, fi = run_cv_mlp(
        train_pre, y, test_pre, features, best_cfg
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
        "width": best_pack["width"],
        "depth": best_pack["depth"],
        "dropout": best_pack["dropout"],
        "learning_rate": best_pack["learning_rate"],
        "batch_size": best_pack["batch_size"],
        "oof_auc": best_pack["oof_auc"]
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:\n- Grid results: {grid_csv}\n- Submission: {sub_path}\n- OOF predictions: {oof_path}\n"
          f"- (1st-layer weight) feature importance: {fi_path}\n- Best params: {best_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
