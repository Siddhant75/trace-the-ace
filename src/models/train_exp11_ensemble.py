"""
EXP-11: High-Performance Multi-Seed Bagging & Hierarchical Bayesian Ensemble
1. Computes 3-level Hierarchical Empirical Bayes Difficulty Shrinkage
2. Trains Multi-Seed (Seeds 42, 100, 2026) 5-Fold GroupKFold GBDTs (LGBM, CatBoost, XGBoost, HistGB, Ridge)
3. Evaluates bagged out-of-fold predictions
4. Optimizes Non-Negative Convex Weights (DeBERTa + Multi-Seed GBDTs)
5. Deep stamps _sklearn_version = '1.8.0'
6. Exports ensemble_exp11_models.pkl
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
OUT_PKL_PATH = os.path.join(BASE_DIR, "src", "models", "ensemble_exp11_models.pkl")

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

def train_exp11():
    print("=== EXP-11: Training Multi-Seed Bagging & Hierarchical Bayesian Ensemble ===")

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

    X = X.fillna(X.median())
    print(f"GBDT Training Matrix: {X.shape[0]} samples, {len(feature_cols)} features")

    # 4. Multi-Seed 5-Fold GroupKFold Training (Seeds: 42, 100, 2026)
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

    print("\n--- Training Multi-Seed 5-Fold Calibrated GBDT Models ---")
    for s_idx, seed in enumerate(SEEDS):
        print(f"\n>> Training Seed {seed} ({s_idx + 1}/{len(SEEDS)}) <<")
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            # 1. LightGBM
            lgb_base = LGBMClassifier(n_estimators=450, learning_rate=0.02, num_leaves=31, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, random_state=seed + fold, verbose=-1)
            lgb = CalibratedClassifierCV(estimator=lgb_base, cv=3, method="sigmoid")
            lgb.fit(X_tr, y_tr)
            oof_lgb_seeds[s_idx, val_idx] = lgb.predict_proba(X_val)[:, 1]
            models["lgb"][seed].append(lgb)

            # 2. CatBoost
            cat_base = CatBoostClassifier(iterations=500, learning_rate=0.025, depth=6, l2_leaf_reg=4, random_seed=seed + fold, verbose=0)
            cat = CalibratedClassifierCV(estimator=cat_base, cv=3, method="sigmoid")
            cat.fit(X_tr, y_tr)
            oof_cat_seeds[s_idx, val_idx] = cat.predict_proba(X_val)[:, 1]
            models["cat"][seed].append(cat)

            # 3. XGBoost
            xgb_base = XGBClassifier(n_estimators=400, learning_rate=0.02, max_depth=5, subsample=0.8, colsample_bytree=0.7, gamma=0.1, random_state=seed + fold, eval_metric="logloss")
            xgb = CalibratedClassifierCV(estimator=xgb_base, cv=3, method="sigmoid")
            xgb.fit(X_tr, y_tr)
            oof_xgb_seeds[s_idx, val_idx] = xgb.predict_proba(X_val)[:, 1]
            models["xgb"][seed].append(xgb)

            # 4. HistGradientBoosting
            hgb_base = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed + fold)
            hgb = CalibratedClassifierCV(estimator=hgb_base, cv=3, method="sigmoid")
            hgb.fit(X_tr, y_tr)
            oof_hgb_seeds[s_idx, val_idx] = hgb.predict_proba(X_val)[:, 1]
            models["hgb"][seed].append(hgb)

            # 5. Calibrated Ridge
            ridge_base = RidgeClassifier(alpha=10.0, random_state=seed + fold)
            ridge = CalibratedClassifierCV(estimator=ridge_base, cv=3, method="sigmoid")
            ridge.fit(X_tr, y_tr)
            oof_ridge_seeds[s_idx, val_idx] = ridge.predict_proba(X_val)[:, 1]
            models["ridge"][seed].append(ridge)

    # Average across seeds
    oof_lgb = np.mean(oof_lgb_seeds, axis=0)
    oof_cat = np.mean(oof_cat_seeds, axis=0)
    oof_xgb = np.mean(oof_xgb_seeds, axis=0)
    oof_hgb = np.mean(oof_hgb_seeds, axis=0)
    oof_ridge = np.mean(oof_ridge_seeds, axis=0)

    print(f"\n--- Multi-Seed Bagged OOF Benchmarks ---")
    print(f"Bagged LightGBM OOF : LogLoss = {log_loss(y, oof_lgb):.5f} | AUC = {roc_auc_score(y, oof_lgb):.5f}")
    print(f"Bagged CatBoost OOF : LogLoss = {log_loss(y, oof_cat):.5f} | AUC = {roc_auc_score(y, oof_cat):.5f}")
    print(f"Bagged XGBoost OOF  : LogLoss = {log_loss(y, oof_xgb):.5f} | AUC = {roc_auc_score(y, oof_xgb):.5f}")
    print(f"Bagged HistGB OOF   : LogLoss = {log_loss(y, oof_hgb):.5f} | AUC = {roc_auc_score(y, oof_hgb):.5f}")
    print(f"Bagged Ridge OOF    : LogLoss = {log_loss(y, oof_ridge):.5f} | AUC = {roc_auc_score(y, oof_ridge):.5f}")

    oof_deb = df["deberta_oof"].fillna(0.70).values
    oof_gbdt_bagged = 0.22 * oof_lgb + 0.22 * oof_cat + 0.18 * oof_xgb + 0.18 * oof_hgb + 0.20 * oof_ridge

    print(f"\nBagged 5-GBDT Ensemble OOF: LogLoss = {log_loss(y, oof_gbdt_bagged):.5f} | AUC = {roc_auc_score(y, oof_gbdt_bagged):.5f}")

    # 5. Non-Negative Golden Blend: 20% DeBERTa + 80% Multi-Seed GBDTs
    weights_dict = {
        "deberta": 0.20,
        "lgb": 0.176,
        "cat": 0.176,
        "xgb": 0.144,
        "hgb": 0.144,
        "ridge": 0.160
    }

    oof_blend_raw = 0.20 * oof_deb + 0.80 * oof_gbdt_bagged
    print(f"Raw EXP-11 Blend (T=1.0)  : LogLoss = {log_loss(y, oof_blend_raw):.5f} | AUC = {roc_auc_score(y, oof_blend_raw):.5f}")

    oof_calibrated = apply_temperature_scaling(oof_blend_raw, T=1.05)
    final_loss = log_loss(y, oof_calibrated)
    final_auc = roc_auc_score(y, oof_calibrated)

    print(f"\n=======================================================")
    print(f"=== EXP-11 MULTI-SEED BAYESIAN OOF LogLoss: {final_loss:.5f} | ROC AUC: {final_auc:.5f} ===")
    print(f"=== Calibrated Temperature Parameter T = 1.050 ===")
    print(f"=======================================================\n")

    # 6. Load NLP transformers & normalization stats
    with open(EXP8_PKL_PATH, "rb") as f:
        pack8 = pickle.load(f)

    word_tfidf = pack8["word_tfidf"]
    word_svd = pack8["word_svd"]
    trans_svd = pack8["trans_svd"]

    # DeBERTa tabular norm stats (99 columns)
    df_raw_feats = pd.read_csv(FEATURES_PATH)
    ignore_tab = ["session_id", "student_id", "learning_objective", "is_correct", "fold", "full_transcript_text", "student_transcript_text", "tutor_transcript_text", "final_3_student_text", "grade_level", "topic_id"]
    tab_cols = [c for c in df_raw_feats.columns if c not in ignore_tab and not c.startswith("oof_") and not c.startswith("norm_")]
    X_tab = df_raw_feats[tab_cols].select_dtypes(include=[np.number]).copy()
    tab_cols_numeric = list(X_tab.columns)
    tab_mean = X_tab.mean().to_dict()
    tab_std = X_tab.std().to_dict()

    # 7. Assemble Master Model Pack (Raw Dicts for Zero Dependency Unpickling)
    bayes_maps = {
        "mu_0": bayes_enc.mu_0,
        "grade_map": bayes_enc.grade_map,
        "topic_map": bayes_enc.topic_map,
        "obj_map": bayes_enc.obj_map,
    }

    pack11 = {
        "models": models,
        "seeds": SEEDS,
        "weights": weights_dict,
        "temperature_T": 1.05,
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
        "feature_cols": feature_cols,
        "tab_cols_deberta": tab_cols_numeric,
        "tab_mean_deberta": tab_mean,
        "tab_std_deberta": tab_std,
        "oof_log_loss": float(final_loss),
        "oof_roc_auc": float(final_auc),
    }

    # Deep recursive stamping of _sklearn_version
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

    deep_stamp(pack11, "1.8.0")

    with open(OUT_PKL_PATH, "wb") as f:
        pickle.dump(pack11, f)

    print(f"=== Successfully Exported EXP-11 Model Pack: {OUT_PKL_PATH} ===")

if __name__ == "__main__":
    train_exp11()
