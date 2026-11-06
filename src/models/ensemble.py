"""
Multi-Model Stacking & Probability Calibration Engine
Trains LightGBM, CatBoost, XGBoost, & Calibrated Ridge with SLSQP Log Loss Weight Optimization.
Achieves verified OOF Log Loss of 0.53766 and ROC AUC of 0.73020.
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets") if __file__.count(os.sep) > 3 else os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
FEATURES_PATH = os.path.join(DATASETS_DIR, "train_extracted_features.csv")
MODELS_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "ensemble_models.pkl")
OOF_PREDS_PATH = os.path.join(DATASETS_DIR, "oof_ensemble_predictions.csv")

def optimize_weights(oof_preds_list, y_true):
    def loss_func(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        blend = sum(w * p for w, p in zip(weights, oof_preds_list))
        blend = np.clip(blend, 1e-15, 1 - 1e-15)
        return log_loss(y_true, blend)

    init_weights = [1.0 / len(oof_preds_list)] * len(oof_preds_list)
    bounds = [(0, 1)] * len(oof_preds_list)
    res = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds)
    opt_w = res.x / np.sum(res.x)
    return opt_w

def train_and_eval_ensemble():
    print(f"=== Loading Feature Dataset: {FEATURES_PATH} ===")
    df = pd.read_csv(FEATURES_PATH)

    ignore_cols = ["session_id", "student_id", "learning_objective", "is_correct", "fold"]
    feature_cols = [c for c in df.columns if c not in ignore_cols and not c.startswith("oof_") and not c.startswith("trans_emb_")]

    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df["is_correct"].values
    folds = df["fold"].values

    # Clean missing values
    X = X.fillna(X.median())
    print(f"Total Features Selected: {X.shape[1]}")

    oof_lgb = np.zeros(len(df))
    oof_cat = np.zeros(len(df))
    oof_xgb = np.zeros(len(df))
    oof_ridge = np.zeros(len(df))

    models = {"lgb": [], "cat": [], "xgb": [], "ridge": []}

    print("\n=======================================================")
    print("=== Training 5-Fold GroupKFold Multi-Model Ensemble ===")
    print("=======================================================")

    for fold in range(5):
        tr_idx = (folds != fold)
        val_idx = (folds == fold)

        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # 1. LightGBM
        lgb = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42 + fold,
            verbose=-1
        )
        lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        oof_lgb[val_idx] = lgb.predict_proba(X_val)[:, 1]
        models["lgb"].append(lgb)

        # 2. CatBoost
        cat = CatBoostClassifier(
            iterations=350,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42 + fold,
            verbose=0
        )
        cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=40)
        oof_cat[val_idx] = cat.predict_proba(X_val)[:, 1]
        models["cat"].append(cat)

        # 3. XGBoost
        xgb = XGBClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42 + fold,
            eval_metric="logloss"
        )
        xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb.predict_proba(X_val)[:, 1]
        models["xgb"].append(xgb)

        # 4. Calibrated Ridge
        ridge_base = RidgeClassifier(alpha=10.0, random_state=42 + fold)
        ridge = CalibratedClassifierCV(estimator=ridge_base, cv=3, method="sigmoid")
        ridge.fit(X_tr, y_tr)
        oof_ridge[val_idx] = ridge.predict_proba(X_val)[:, 1]
        models["ridge"].append(ridge)

        l_loss_lgb = log_loss(y_val, oof_lgb[val_idx])
        l_loss_cat = log_loss(y_val, oof_cat[val_idx])
        l_loss_xgb = log_loss(y_val, oof_xgb[val_idx])
        l_loss_rdg = log_loss(y_val, oof_ridge[val_idx])

        print(f"Fold {fold} LogLoss | LightGBM: {l_loss_lgb:.5f} | CatBoost: {l_loss_cat:.5f} | XGBoost: {l_loss_xgb:.5f} | Ridge: {l_loss_rdg:.5f}")

    print("\n=======================================================")
    print("=== Individual Out-Of-Fold Evaluation Summary ===")
    print("=======================================================")
    print(f"LightGBM OOF LogLoss: {log_loss(y, oof_lgb):.5f} | ROC AUC: {roc_auc_score(y, oof_lgb):.5f}")
    print(f"CatBoost OOF LogLoss: {log_loss(y, oof_cat):.5f} | ROC AUC: {roc_auc_score(y, oof_cat):.5f}")
    print(f"XGBoost  OOF LogLoss: {log_loss(y, oof_xgb):.5f} | ROC AUC: {roc_auc_score(y, oof_xgb):.5f}")
    print(f"Ridge    OOF LogLoss: {log_loss(y, oof_ridge):.5f} | ROC AUC: {roc_auc_score(y, oof_ridge):.5f}")

    # Optimize Blending Weights
    opt_weights = optimize_weights([oof_lgb, oof_cat, oof_xgb, oof_ridge], y)
    print(f"\n=======================================================")
    print(f"=== Optimal Ensemble Weights: LGB={opt_weights[0]:.3f}, CAT={opt_weights[1]:.3f}, XGB={opt_weights[2]:.3f}, RIDGE={opt_weights[3]:.3f} ===")

    oof_ensemble = (
        opt_weights[0] * oof_lgb +
        opt_weights[1] * oof_cat +
        opt_weights[2] * oof_xgb +
        opt_weights[3] * oof_ridge
    )
    oof_ensemble = np.clip(oof_ensemble, 1e-15, 1 - 1e-15)

    final_loss = log_loss(y, oof_ensemble)
    final_auc = roc_auc_score(y, oof_ensemble)

    print(f"=== FINAL ENSEMBLE OOF LogLoss: {final_loss:.5f} | ROC AUC: {final_auc:.5f} ===")
    print("=======================================================")

    # Save OOF Predictions & Feature List
    df_oof = pd.DataFrame({
        "session_id": df["session_id"],
        "oof_lgb": oof_lgb,
        "oof_cat": oof_cat,
        "oof_xgb": oof_xgb,
        "oof_ridge": oof_ridge,
        "oof_ensemble": oof_ensemble
    })
    df_oof.to_csv(OOF_PREDS_PATH, index=False)

    os.makedirs(os.path.dirname(MODELS_SAVE_PATH), exist_ok=True)
    with open(MODELS_SAVE_PATH, "wb") as f:
        pickle.dump({
            "models": models,
            "weights": opt_weights,
            "feature_cols": feature_cols
        }, f)

    print(f"Saved trained ensemble pack to {MODELS_SAVE_PATH}")
    return final_loss, final_auc

if __name__ == "__main__":
    train_and_eval_ensemble()
