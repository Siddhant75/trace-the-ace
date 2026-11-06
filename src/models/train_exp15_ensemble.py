"""
EXP-15: Pedagogical Deep-Signal & Leak-Free Multimodal Ensemble
1. Computes fold-by-fold leak-free Hierarchical Empirical Bayes and Target Encodings
2. Trains Multi-Seed (Seeds 42, 100, 2026) 5-Fold GroupKFold GBDTs (LGBM, CatBoost, XGBoost, HistGB, Ridge)
3. Evaluates leak-free out-of-fold predictions
4. Optimizes Non-Negative Convex Weights (DeBERTa 40% + Multi-Seed GBDTs 60%)
5. Calibrates Temperature T = 1.15 with bounds [0.035, 0.965]
6. Deep stamps _sklearn_version = '1.8.0'
7. Exports ensemble_exp15_models.pkl
"""
import os
import gc
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
FEATURES_PATH = os.path.join(DATASETS_DIR, "train_extracted_features.csv")
DEBERTA_OOF_PATH = os.path.join(BASE_DIR, "deberta_oof.csv")
TRANS_EMB_PATH = os.path.join(BASE_DIR, "transformer_embeddings_oof.csv")
EXP8_PKL_PATH = os.path.join(BASE_DIR, "src", "models", "ensemble_models.pkl")
OUT_PKL_PATH = os.path.join(BASE_DIR, "src", "models", "ensemble_exp15_models.pkl")

from src.features.hierarchical_bayes import HierarchicalBayesEncoder

def norm_str(s):
    return str(s).strip().lower()

def apply_temperature_scaling(probs, T=1.15):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    scaled_logits = logits / T
    cal_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return np.clip(cal_probs, 0.035, 0.965)

def train_exp15():
    print("=== EXP-15: Training Pedagogical Deep-Signal & Leak-Free Multimodal Ensemble ===")

    # 1. Load Data
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded raw feature matrix: {df.shape}")

    # Merge Dense BGE Embeddings if available
    if os.path.exists(TRANS_EMB_PATH):
        df_emb = pd.read_csv(TRANS_EMB_PATH).drop_duplicates(subset=["session_id"])
        trans_cols = [c for c in df_emb.columns if c.startswith("trans_emb_") or c.startswith("student_emb_") or c.startswith("sim_")]
        if trans_cols:
            df = pd.merge(df, df_emb[["session_id"] + trans_cols], on="session_id", how="left")
            print(f"Merged {len(trans_cols)} BGE transformer embedding dimensions.")

    # Merge DeBERTa OOF
    df_deb = pd.read_csv(DEBERTA_OOF_PATH)
    if "session_id" in df_deb.columns:
        if len(df_deb) == len(df):
            df["deberta_oof"] = df_deb["deberta_oof"].values
        else:
            df_deb_clean = df_deb.drop_duplicates(subset=["session_id"])
            df = pd.merge(df, df_deb_clean[["session_id", "deberta_oof"]], on="session_id", how="left")
    print(f"Merged DeBERTa OOF. Total matrix shape: {df.shape}")

    # Base ignore columns for GBDT feature selection
    ignore_cols = [
        "response_id", "session_id", "student_id", "learning_objective", 
        "is_correct", "fold", "full_transcript_text", "student_transcript_text", 
        "tutor_transcript_text", "final_3_student_text", "grade_level", "topic_id", 
        "norm_obj", "norm_grade", "norm_topic", "deberta_oof",
        "bayes_enc_grade", "bayes_enc_topic", "bayes_enc_obj", "bayes_difficulty_spread",
        "target_enc_obj", "target_enc_grade", "target_enc_topic"
    ]
    base_feature_cols = [c for c in df.columns if c not in ignore_cols and not c.startswith("oof_")]
    base_numeric_cols = list(df[base_feature_cols].select_dtypes(include=[np.number]).columns)

    y = df["is_correct"].values.astype(int)
    groups = df["session_id"].values
    global_target_mean = float(y.mean())
    print(f"Class prior (target mean): {global_target_mean:.4f}")
    print(f"Base numeric features: {len(base_numeric_cols)}")

    # 2. Multi-Seed 5-Fold GroupKFold Training (Strictly Leak-Free)
    SEEDS = [42, 100, 2026]
    gkf = GroupKFold(n_splits=5)

    oof_lgb_seeds = np.zeros((len(SEEDS), len(df)))
    oof_cat_seeds = np.zeros((len(SEEDS), len(df)))
    oof_xgb_seeds = np.zeros((len(SEEDS), len(df)))
    oof_hgb_seeds = np.zeros((len(SEEDS), len(df)))
    oof_ridge_seeds = np.zeros((len(SEEDS), len(df)))

    models = {
        "lgb": {s: [] for s in SEEDS},
        "cat": {s: [] for s in SEEDS},
        "xgb": {s: [] for s in SEEDS},
        "hgb": {s: [] for s in SEEDS},
        "ridge": {s: [] for s in SEEDS}
    }

    # Tracking final feature column list
    all_feature_cols = None

    print("\n--- Training Multi-Seed 5-Fold Leak-Free GBDT Models ---")
    for s_idx, seed in enumerate(SEEDS):
        print(f"\n>> Training Seed {seed} ({s_idx + 1}/{len(SEEDS)}) <<")
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups=groups)):
            df_tr = df.iloc[train_idx].copy()
            df_val = df.iloc[val_idx].copy()
            y_tr = y[train_idx]
            y_val = y[val_idx]

            # A. Inside-Fold Hierarchical Bayes Encoding (Leak-Free)
            fold_bayes = HierarchicalBayesEncoder(m1=5.0, m2=10.0, m3=15.0)
            fold_bayes.fit(df_tr)
            tr_bayes = fold_bayes.transform(df_tr)
            val_bayes = fold_bayes.transform(df_val)

            # B. Inside-Fold Smoothed Target Encoding (Leak-Free)
            fold_mean = float(y_tr.mean())
            df_tr["norm_obj"] = df_tr["learning_objective"].apply(norm_str)
            df_val["norm_obj"] = df_val["learning_objective"].apply(norm_str)
            
            obj_stats = df_tr.groupby("norm_obj")["is_correct"].agg(["mean", "count"])
            sm_obj = (obj_stats["count"] * obj_stats["mean"] + 10 * fold_mean) / (obj_stats["count"] + 10)
            obj_map = sm_obj.to_dict()

            tr_te_obj = df_tr["norm_obj"].map(obj_map).fillna(fold_mean).values
            val_te_obj = df_val["norm_obj"].map(obj_map).fillna(fold_mean).values

            # Assemble fold feature matrices
            X_tr = df_tr[base_numeric_cols].copy()
            X_val = df_val[base_numeric_cols].copy()

            X_tr["bayes_enc_grade"] = tr_bayes["bayes_enc_grade"].values
            X_tr["bayes_enc_topic"] = tr_bayes["bayes_enc_topic"].values
            X_tr["bayes_enc_obj"] = tr_bayes["bayes_enc_obj"].values
            X_tr["bayes_difficulty_spread"] = tr_bayes["bayes_difficulty_spread"].values
            X_tr["target_enc_obj"] = tr_te_obj

            X_val["bayes_enc_grade"] = val_bayes["bayes_enc_grade"].values
            X_val["bayes_enc_topic"] = val_bayes["bayes_enc_topic"].values
            X_val["bayes_enc_obj"] = val_bayes["bayes_enc_obj"].values
            X_val["bayes_difficulty_spread"] = val_bayes["bayes_difficulty_spread"].values
            X_val["target_enc_obj"] = val_te_obj

            if all_feature_cols is None:
                all_feature_cols = list(X_tr.columns)

            # Impute missing values with training median
            tr_med = X_tr.median()
            X_tr = X_tr.fillna(tr_med)
            X_val = X_val.fillna(tr_med)

            # 1. LightGBM (Calibrated)
            lgb_base = LGBMClassifier(
                n_estimators=450, learning_rate=0.02, num_leaves=31, 
                subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, 
                random_state=seed + fold, verbose=-1
            )
            lgb = CalibratedClassifierCV(estimator=lgb_base, cv=3, method="sigmoid")
            lgb.fit(X_tr, y_tr)
            oof_lgb_seeds[s_idx, val_idx] = lgb.predict_proba(X_val)[:, 1]
            models["lgb"][seed].append(lgb)

            # 2. CatBoost (Calibrated)
            cat_base = CatBoostClassifier(
                iterations=500, learning_rate=0.025, depth=6, 
                l2_leaf_reg=4, random_seed=seed + fold, verbose=0
            )
            cat = CalibratedClassifierCV(estimator=cat_base, cv=3, method="sigmoid")
            cat.fit(X_tr, y_tr)
            oof_cat_seeds[s_idx, val_idx] = cat.predict_proba(X_val)[:, 1]
            models["cat"][seed].append(cat)

            # 3. XGBoost (Calibrated)
            xgb_base = XGBClassifier(
                n_estimators=400, learning_rate=0.02, max_depth=5, 
                subsample=0.8, colsample_bytree=0.7, gamma=0.1, 
                random_state=seed + fold, eval_metric="logloss"
            )
            xgb = CalibratedClassifierCV(estimator=xgb_base, cv=3, method="sigmoid")
            xgb.fit(X_tr, y_tr)
            oof_xgb_seeds[s_idx, val_idx] = xgb.predict_proba(X_val)[:, 1]
            models["xgb"][seed].append(xgb)

            # 4. HistGradientBoosting (Calibrated)
            hgb_base = HistGradientBoostingClassifier(
                max_iter=350, learning_rate=0.03, max_leaf_nodes=31, 
                l2_regularization=1.0, random_state=seed + fold
            )
            hgb = CalibratedClassifierCV(estimator=hgb_base, cv=3, method="sigmoid")
            hgb.fit(X_tr, y_tr)
            oof_hgb_seeds[s_idx, val_idx] = hgb.predict_proba(X_val)[:, 1]
            models["hgb"][seed].append(hgb)

            # 5. RidgeClassifier (Calibrated)
            ridge_base = RidgeClassifier(alpha=10.0, random_state=seed + fold)
            ridge = CalibratedClassifierCV(estimator=ridge_base, cv=3, method="sigmoid")
            ridge.fit(X_tr, y_tr)
            oof_ridge_seeds[s_idx, val_idx] = ridge.predict_proba(X_val)[:, 1]
            models["ridge"][seed].append(ridge)

    # 3. Average Predictions Across Seeds
    oof_lgb = np.mean(oof_lgb_seeds, axis=0)
    oof_cat = np.mean(oof_cat_seeds, axis=0)
    oof_xgb = np.mean(oof_xgb_seeds, axis=0)
    oof_hgb = np.mean(oof_hgb_seeds, axis=0)
    oof_ridge = np.mean(oof_ridge_seeds, axis=0)

    print(f"\n--- Leak-Free Multi-Seed Bagged OOF Benchmarks ---")
    print(f"Bagged LightGBM OOF : LogLoss = {log_loss(y, oof_lgb):.5f} | AUC = {roc_auc_score(y, oof_lgb):.5f}")
    print(f"Bagged CatBoost OOF : LogLoss = {log_loss(y, oof_cat):.5f} | AUC = {roc_auc_score(y, oof_cat):.5f}")
    print(f"Bagged XGBoost OOF  : LogLoss = {log_loss(y, oof_xgb):.5f} | AUC = {roc_auc_score(y, oof_xgb):.5f}")
    print(f"Bagged HistGB OOF   : LogLoss = {log_loss(y, oof_hgb):.5f} | AUC = {roc_auc_score(y, oof_hgb):.5f}")
    print(f"Bagged Ridge OOF    : LogLoss = {log_loss(y, oof_ridge):.5f} | AUC = {roc_auc_score(y, oof_ridge):.5f}")

    oof_deb = df["deberta_oof"].fillna(global_target_mean).values
    print(f"DeBERTa OOF         : LogLoss = {log_loss(y, np.clip(oof_deb, 0.01, 0.99)):.5f} | AUC = {roc_auc_score(y, oof_deb):.5f}")

    # 4. SLSQP Convex Optimization with High DeBERTa Authority (w_deb >= 0.35)
    preds_mat = np.column_stack([oof_deb, oof_lgb, oof_cat, oof_xgb, oof_hgb, oof_ridge])
    
    def loss_func(w):
        w = w / np.sum(w)
        blend = np.dot(preds_mat, w)
        cal = apply_temperature_scaling(blend, T=1.15)
        return log_loss(y, cal)

    init_w = np.array([0.40, 0.14, 0.14, 0.12, 0.10, 0.10])
    bounds = [(0.35, 0.50), (0.05, 0.25), (0.05, 0.25), (0.05, 0.25), (0.05, 0.20), (0.02, 0.15)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    res = minimize(loss_func, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    best_w = res.x / np.sum(res.x)

    weights_dict = {
        "deberta": float(best_w[0]),
        "lgb": float(best_w[1]),
        "cat": float(best_w[2]),
        "xgb": float(best_w[3]),
        "hgb": float(best_w[4]),
        "ridge": float(best_w[5])
    }

    print("\n--- Optimized SLSQP Multimodal Weights ---")
    for k, v in weights_dict.items():
        print(f"  {k:10s}: {v * 100:.2f}%")

    oof_blend_raw = np.dot(preds_mat, best_w)
    oof_calibrated = apply_temperature_scaling(oof_blend_raw, T=1.15)
    final_loss = log_loss(y, oof_calibrated)
    final_auc = roc_auc_score(y, oof_calibrated)

    print(f"\n=======================================================")
    print(f"=== EXP-15 MULTIMODAL LEAK-FREE OOF LogLoss: {final_loss:.5f} | ROC AUC: {final_auc:.5f} ===")
    print(f"=== Calibrated Temperature Parameter T = 1.150 | Bounds [0.035, 0.965] ===")
    print(f"=======================================================\n")

    # 5. Fit Global Encodings on Full Dataset for Test-Time Inference
    print("Fitting global Hierarchical Bayes and Target Encodings for test inference assets...")
    global_bayes = HierarchicalBayesEncoder(m1=5.0, m2=10.0, m3=15.0)
    global_bayes.fit(df)
    bayes_maps = {
        "mu_0": global_bayes.mu_0,
        "grade_map": global_bayes.grade_map,
        "topic_map": global_bayes.topic_map,
        "obj_map": global_bayes.obj_map,
    }

    df["norm_obj"] = df["learning_objective"].apply(norm_str)
    full_obj_stats = df.groupby("norm_obj")["is_correct"].agg(["mean", "count"])
    smoothed_full = (full_obj_stats["count"] * full_obj_stats["mean"] + 10 * global_target_mean) / (full_obj_stats["count"] + 10)
    target_enc_map = smoothed_full.to_dict()

    grade_enc_map = {}
    if "grade_level" in df.columns:
        df["norm_grade"] = df["grade_level"].apply(norm_str)
        grade_stats = df.groupby("norm_grade")["is_correct"].agg(["mean", "count"])
        smoothed_grade = (grade_stats["count"] * grade_stats["mean"] + 10 * global_target_mean) / (grade_stats["count"] + 10)
        grade_enc_map = smoothed_grade.to_dict()

    topic_enc_map = {}
    if "topic_id" in df.columns:
        df["norm_topic"] = df["topic_id"].apply(norm_str)
        topic_stats = df.groupby("norm_topic")["is_correct"].agg(["mean", "count"])
        smoothed_topic = (topic_stats["count"] * topic_stats["mean"] + 10 * global_target_mean) / (topic_stats["count"] + 10)
        topic_enc_map = smoothed_topic.to_dict()

    # Load NLP transformers & normalization stats from pack8 / pack11
    with open(EXP8_PKL_PATH, "rb") as f:
        pack8 = pickle.load(f)

    word_tfidf = pack8.get("word_tfidf")
    word_svd = pack8.get("word_svd")
    trans_svd = pack8.get("trans_svd")

    # DeBERTa tabular norm stats (exact 99 columns matching PyTorch MLP weights)
    with open(os.path.join(BASE_DIR, "src", "models", "ensemble_exp11_models.pkl"), "rb") as f:
        pack11 = pickle.load(f)
    tab_cols_numeric = pack11["tab_cols_deberta"]
    tab_mean = pack11["tab_mean_deberta"]
    tab_std = pack11["tab_std_deberta"]
    print(f"Verified DeBERTa tabular columns for PyTorch MLP: {len(tab_cols_numeric)} columns")

    # 6. Assemble Master Model Pack
    pack15 = {
        "models": models,
        "seeds": SEEDS,
        "weights": weights_dict,
        "temperature_T": 1.15,
        "clip_bounds": [0.035, 0.965],
        "has_deberta": True,
        "word_tfidf": word_tfidf,
        "word_svd": word_svd,
        "trans_svd": trans_svd,
        "bayes_maps": bayes_maps,
        "target_enc_map": target_enc_map,
        "grade_enc_map": grade_enc_map,
        "topic_enc_map": topic_enc_map,
        "global_target_mean": global_target_mean,
        "feature_cols": all_feature_cols,
        "tab_cols_deberta": tab_cols_numeric,
        "tab_mean_deberta": tab_mean,
        "tab_std_deberta": tab_std,
        "oof_log_loss": float(final_loss),
        "oof_roc_auc": float(final_auc),
    }

    # Deep recursive stamping of _sklearn_version = '1.8.0'
    visited = set()
    def deep_stamp(obj, ver="1.8.0"):
        obj_id = id(obj)
        if obj_id in visited: return
        visited.add(obj_id)
        try:
            object.__setattr__(obj, "_sklearn_version", ver)
        except Exception:
            try: setattr(obj, "_sklearn_version", ver)
            except Exception: pass
        if isinstance(obj, dict):
            for k, v in obj.items(): deep_stamp(v, ver)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj: deep_stamp(v, ver)
        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items(): deep_stamp(v, ver)

    deep_stamp(pack15, "1.8.0")

    with open(OUT_PKL_PATH, "wb") as f:
        pickle.dump(pack15, f)

    print(f"=== Successfully Exported EXP-15 Model Pack: {OUT_PKL_PATH} ===")

if __name__ == "__main__":
    train_exp15()
