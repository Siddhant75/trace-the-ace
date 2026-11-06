"""
EXP-14 Blend Optimization Script:
1. Reconstructs OOF probability vectors for all models (DeBERTa, LGBM, CatBoost, XGBoost, HistGB, Ridge)
2. Runs SLSQP and Nelder-Mead constrained optimization for exact Log Loss minimization
3. Grid searches optimal temperature T in [1.00, 1.30]
4. Reports the optimal weights and temperature
"""
import os
import gc
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
FEATURES_PATH = os.path.join(DATASETS_DIR, "train_extracted_features.csv")
DEBERTA_OOF_PATH = os.path.join(BASE_DIR, "deberta_oof.csv")
TRANS_EMB_PATH = os.path.join(BASE_DIR, "transformer_embeddings_oof.csv")
EXP8_PKL_PATH = os.path.join(BASE_DIR, "src", "models", "ensemble_models.pkl")

from src.features.hierarchical_bayes import HierarchicalBayesEncoder

def norm_str(s):
    return str(s).strip().lower()

def apply_temperature_scaling(probs, T=1.05):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    scaled_logits = logits / T
    cal_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return np.clip(cal_probs, 0.035, 0.965)

def main():
    print("=== EXP-14: Optimizing Ensemble Blending Weights & Temperature ===")
    
    # 1. Load Data
    df = pd.read_csv(FEATURES_PATH)

    # Hierarchical Empirical Bayes Difficulty Encodings
    bayes_enc = HierarchicalBayesEncoder(m1=5.0, m2=10.0, m3=15.0)
    bayes_enc.fit(df)
    df_bayes = bayes_enc.transform(df)
    df = pd.concat([df, df_bayes], axis=1)

    # Standard Target Encodings
    global_target_mean = float(df["is_correct"].mean())
    df["norm_obj"] = df["learning_objective"].apply(norm_str)
    obj_stats = df.groupby("norm_obj")["is_correct"].agg(["mean", "count"])
    smoothed_obj = (obj_stats["count"] * obj_stats["mean"] + 10 * global_target_mean) / (obj_stats["count"] + 10)
    target_enc_map = smoothed_obj.to_dict()
    df["target_enc_obj"] = df["norm_obj"].map(target_enc_map).fillna(global_target_mean)

    grade_enc_map = {}
    if "grade_level" in df.columns:
        df["norm_grade"] = df["grade_level"].apply(norm_str)
        grade_stats = df.groupby("norm_grade")["is_correct"].agg(["mean", "count"])
        smoothed_grade = (grade_stats["count"] * grade_stats["mean"] + 10 * global_target_mean) / (grade_stats["count"] + 10)
        grade_enc_map = smoothed_grade.to_dict()
        df["target_enc_grade"] = df["norm_grade"].map(grade_enc_map).fillna(global_target_mean)

    topic_enc_map = {}
    if "topic_id" in df.columns:
        df["norm_topic"] = df["topic_id"].apply(norm_str)
        topic_stats = df.groupby("norm_topic")["is_correct"].agg(["mean", "count"])
        smoothed_topic = (topic_stats["count"] * topic_stats["mean"] + 10 * global_target_mean) / (topic_stats["count"] + 10)
        topic_enc_map = smoothed_topic.to_dict()
        df["target_enc_topic"] = df["norm_topic"].map(topic_enc_map).fillna(global_target_mean)

    # Merge Dense BGE Embeddings
    if os.path.exists(TRANS_EMB_PATH):
        df_emb = pd.read_csv(TRANS_EMB_PATH).drop_duplicates(subset=["session_id"])
        trans_cols = [c for c in df_emb.columns if c.startswith("trans_emb_") or c.startswith("student_emb_") or c.startswith("sim_")]
        if trans_cols:
            df = pd.merge(df, df_emb[["session_id"] + trans_cols], on="session_id", how="left")

    # Merge DeBERTa OOF
    df_deb = pd.read_csv(DEBERTA_OOF_PATH).drop_duplicates(subset=["session_id"])
    df = pd.merge(df, df_deb[["session_id", "deberta_oof"]], on="session_id", how="left")
    print(f"Loaded merged feature matrix: {df.shape}")

    # Setup Feature Columns for GBDT
    ignore_cols = ["response_id", "session_id", "student_id", "learning_objective", "is_correct", "fold", "full_transcript_text", "student_transcript_text", "tutor_transcript_text", "final_3_student_text", "grade_level", "topic_id", "norm_obj", "norm_grade", "norm_topic", "deberta_oof"]
    feature_cols = [c for c in df.columns if c not in ignore_cols and not c.startswith("oof_")]

    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    feature_cols = list(X.columns)
    y = df["is_correct"].values.astype(int)
    groups = df["student_id"].values if "student_id" in df.columns else df["session_id"].values
    oof_deb = df["deberta_oof"].fillna(0.70).values

    X = X.fillna(X.median())
    print(f"GBDT Training Matrix: {X.shape[0]} samples, {len(feature_cols)} features")

    # 2. Train 3-Seed 5-Fold GBDT Models to Get Clean OOF Vectors
    SEEDS = [42, 100, 2026]
    n_samples = len(df)
    
    oof_lgb_seeds = np.zeros(n_samples)
    oof_cat_seeds = np.zeros(n_samples)
    oof_xgb_seeds = np.zeros(n_samples)
    oof_hgb_seeds = np.zeros(n_samples)
    oof_ridge_seeds = np.zeros(n_samples)

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)

    for seed in SEEDS:
        print(f">> Training Seed {seed} <<")
        oof_lgb_seed = np.zeros(n_samples)
        oof_cat_seed = np.zeros(n_samples)
        oof_xgb_seed = np.zeros(n_samples)
        oof_hgb_seed = np.zeros(n_samples)
        oof_ridge_seed = np.zeros(n_samples)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]

            # 1. LGBM
            m_lgb = CalibratedClassifierCV(
                estimator=LGBMClassifier(
                    n_estimators=450, learning_rate=0.02, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=seed + fold, n_jobs=-1, verbose=-1
                ),
                cv=3, method="sigmoid"
            )
            m_lgb.fit(X_tr, y_tr)
            oof_lgb_seed[val_idx] = m_lgb.predict_proba(X_va)[:, 1]

            # 2. CatBoost
            m_cat = CalibratedClassifierCV(
                estimator=CatBoostClassifier(
                    iterations=500, learning_rate=0.025, depth=6, l2_leaf_reg=4,
                    random_seed=seed + fold, verbose=0, thread_count=-1
                ),
                cv=3, method="sigmoid"
            )
            m_cat.fit(X_tr, y_tr)
            oof_cat_seed[val_idx] = m_cat.predict_proba(X_va)[:, 1]

            # 3. XGBoost
            m_xgb = CalibratedClassifierCV(
                estimator=XGBClassifier(
                    n_estimators=400, learning_rate=0.02, max_depth=5,
                    subsample=0.8, colsample_bytree=0.7, gamma=0.1,
                    eval_metric="logloss", random_state=seed + fold, n_jobs=-1
                ),
                cv=3, method="sigmoid"
            )
            m_xgb.fit(X_tr, y_tr)
            oof_xgb_seed[val_idx] = m_xgb.predict_proba(X_va)[:, 1]

            # 4. HistGB
            m_hgb = CalibratedClassifierCV(
                estimator=HistGradientBoostingClassifier(
                    max_iter=350, learning_rate=0.03, max_leaf_nodes=31,
                    l2_regularization=1.0, random_state=seed + fold
                ),
                cv=3, method="sigmoid"
            )
            m_hgb.fit(X_tr, y_tr)
            oof_hgb_seed[val_idx] = m_hgb.predict_proba(X_va)[:, 1]

            # 5. Ridge
            m_ridge = CalibratedClassifierCV(
                estimator=RidgeClassifier(alpha=10.0, random_state=seed + fold),
                cv=3, method="sigmoid"
            )
            m_ridge.fit(X_tr, y_tr)
            oof_ridge_seed[val_idx] = m_ridge.predict_proba(X_va)[:, 1]

        oof_lgb_seeds += oof_lgb_seed / len(SEEDS)
        oof_cat_seeds += oof_cat_seed / len(SEEDS)
        oof_xgb_seeds += oof_xgb_seed / len(SEEDS)
        oof_hgb_seeds += oof_hgb_seed / len(SEEDS)
        oof_ridge_seeds += oof_ridge_seed / len(SEEDS)

    # Report individual model performance
    model_names = ["deberta", "lgb", "cat", "xgb", "hgb", "ridge"]
    oof_matrix = np.column_stack([
        oof_deb, oof_lgb_seeds, oof_cat_seeds, oof_xgb_seeds, oof_hgb_seeds, oof_ridge_seeds
    ])

    print("\n--- Individual Model OOF LogLoss & ROC AUC ---")
    for i, name in enumerate(model_names):
        p = oof_matrix[:, i]
        print(f"  {name:10s}: LogLoss = {log_loss(y, p):.5f} | AUC = {roc_auc_score(y, p):.5f}")

    # 3. SLSQP Optimization
    print("\n--- Running SLSQP Log Loss Constrained Optimization ---")
    def objective_slsqp(w):
        p_blend = np.clip(oof_matrix @ w, 1e-15, 1 - 1e-15)
        return log_loss(y, p_blend)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0.0, 1.0) for _ in range(len(model_names))]
    init_w = np.ones(len(model_names)) / len(model_names)

    res_slsqp = minimize(objective_slsqp, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    w_slsqp = res_slsqp.x
    w_slsqp = w_slsqp / np.sum(w_slsqp)

    print("Optimal SLSQP Weights:")
    for name, w in zip(model_names, w_slsqp):
        print(f"  {name:10s}: {w:.4f} ({w*100:.2f}%)")
    
    oof_slsqp_raw = oof_matrix @ w_slsqp
    print(f"SLSQP Raw Blend LogLoss: {log_loss(y, oof_slsqp_raw):.5f} | AUC: {roc_auc_score(y, oof_slsqp_raw):.5f}")

    # 4. Temperature Scaling Grid Search on Optimal Blend
    print("\n--- Temperature Grid Search on Optimal Blend ---")
    T_values = [1.00, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15, 1.18, 1.20, 1.25, 1.30]
    best_T = 1.05
    best_T_loss = 999.0
    
    for T in T_values:
        p_cal = apply_temperature_scaling(oof_slsqp_raw, T=T)
        loss_T = log_loss(y, p_cal)
        auc_T = roc_auc_score(y, p_cal)
        print(f"  T = {T:.2f} -> LogLoss: {loss_T:.5f} | AUC: {auc_T:.5f}")
        if loss_T < best_T_loss:
            best_T_loss = loss_T
            best_T = T

    print(f"\n=======================================================")
    print(f"=== EXP-14 OPTIMAL CONFIGURATION ===")
    print(f"=== Best Temperature T*: {best_T:.2f} (LogLoss: {best_T_loss:.5f}) ===")
    print(f"=== Weights: {dict(zip(model_names, [round(float(x), 4) for x in w_slsqp]))} ===")
    print(f"=======================================================\n")

    # Save OOFs to cache for immediate re-use
    np.save("exp14_oof_matrix.npy", oof_matrix)
    print("Saved exp14_oof_matrix.npy successfully.")

if __name__ == "__main__":
    main()
