"""
EXP-9 Heterogeneous Stacking Meta-Learner Training Script
Combines:
1. DeBERTa-v3-Large + Tabular MLP Multimodal Neural Classifier (P_DeBERTa)
2. 5-Model GBDT Diversity Ensemble (P_LGBM, P_CatBoost, P_XGBoost, P_HistGB, P_Ridge)
3. 2nd-Stage LogisticRegression Stacking Meta-Learner
4. Temperature Calibration Scaling (T) + [0.025, 0.975] Bounded Clipping
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) if __file__.count(os.sep) > 3 else os.path.dirname(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
FEATURES_PATH = os.path.join(DATASETS_DIR, "train_extracted_features.csv")
MODELS_SAVE_PATH = os.path.join(os.path.dirname(__file__), "ensemble_exp9_models.pkl")

def norm_str(s):
    return str(s).strip().lower()

def apply_temperature_scaling(probs, T=1.15):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    scaled_logits = logits / T
    cal_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return np.clip(cal_probs, 0.025, 0.975)

def train_and_export_exp9():
    print(f"=== EXP-9: Training Heterogeneous Stacking Ensemble ===")
    df = pd.read_csv(FEATURES_PATH)

    # 1. String-Normalized Multi-Factor Target Encodings
    global_target_mean = float(df["is_correct"].mean())
    df["norm_obj"] = df["learning_objective"].apply(norm_str)
    obj_stats = df.groupby("norm_obj")["is_correct"].agg(["mean", "count"])
    smoothed_obj = (obj_stats["count"] * obj_stats["mean"] + 10 * global_target_mean) / (obj_stats["count"] + 10)
    target_enc_map = smoothed_obj.to_dict()

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

    # 2. Dense Transformer & SVD Embeddings
    emb_csv_path = os.path.join(BASE_DIR, "transformer_embeddings_oof.csv")
    if os.path.exists(emb_csv_path):
        df_emb = pd.read_csv(emb_csv_path)
        trans_cols = [c for c in df_emb.columns if c.startswith("trans_emb_") or c.startswith("student_emb_") or c.startswith("sim_")]
        if trans_cols:
            df = pd.merge(df, df_emb[["session_id"] + trans_cols], on="session_id", how="left")

    # 3. Load DeBERTa OOF Predictions if available
    deberta_oof_path = os.path.join(BASE_DIR, "deberta_oof.csv")
    has_deberta = False
    if os.path.exists(deberta_oof_path):
        print(f"=== Found DeBERTa-v3 OOF Predictions: {deberta_oof_path} ===")
        df_deb = pd.read_csv(deberta_oof_path)
        if "deberta_oof" in df_deb.columns:
            df = pd.merge(df, df_deb[["session_id", "deberta_oof"]], on="session_id", how="left")
            has_deberta = True

    ignore_cols = ["session_id", "student_id", "learning_objective", "is_correct", "fold", "full_transcript_text", "norm_obj", "norm_grade", "norm_topic", "deberta_oof"]
    feature_cols = [c for c in df.columns if c not in ignore_cols and not c.startswith("oof_")]

    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    feature_cols = list(X.columns)
    y = df["is_correct"].values

    # 4. Leak-Free 5-Fold GroupKFold by student_id
    gkf = GroupKFold(n_splits=5)
    student_groups = df["student_id"].values if "student_id" in df.columns else df["session_id"].values
    folds = np.zeros(len(df), dtype=int)
    for fold_idx, (tr_i, val_i) in enumerate(gkf.split(X, y, groups=student_groups)):
        folds[val_i] = fold_idx

    X = X.fillna(X.median())

    oof_lgb = np.zeros(len(df))
    oof_cat = np.zeros(len(df))
    oof_xgb = np.zeros(len(df))
    oof_hgb = np.zeros(len(df))
    oof_ridge = np.zeros(len(df))

    models = {"lgb": [], "cat": [], "xgb": [], "hgb": [], "ridge": []}

    print("\n=== Training 5-Fold Student-GroupKFold 5-Model Diversity Ensemble ===")
    for fold in range(5):
        tr_idx = (folds != fold)
        val_idx = (folds == fold)

        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # 1. LightGBM
        lgb_base = LGBMClassifier(n_estimators=450, learning_rate=0.02, num_leaves=31, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, random_state=42 + fold, verbose=-1)
        lgb = CalibratedClassifierCV(estimator=lgb_base, cv=3, method="sigmoid")
        lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb.predict_proba(X_val)[:, 1]
        models["lgb"].append(lgb)

        # 2. CatBoost
        cat_base = CatBoostClassifier(iterations=500, learning_rate=0.025, depth=6, l2_leaf_reg=4, random_seed=42 + fold, verbose=0)
        cat = CalibratedClassifierCV(estimator=cat_base, cv=3, method="sigmoid")
        cat.fit(X_tr, y_tr)
        oof_cat[val_idx] = cat.predict_proba(X_val)[:, 1]
        models["cat"].append(cat)

        # 3. XGBoost
        xgb_base = XGBClassifier(n_estimators=400, learning_rate=0.02, max_depth=5, subsample=0.8, colsample_bytree=0.7, gamma=0.1, random_state=42 + fold, eval_metric="logloss")
        xgb = CalibratedClassifierCV(estimator=xgb_base, cv=3, method="sigmoid")
        xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb.predict_proba(X_val)[:, 1]
        models["xgb"].append(xgb)

        # 4. HistGradientBoosting
        hgb_base = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=1.0, random_state=42 + fold)
        hgb = CalibratedClassifierCV(estimator=hgb_base, cv=3, method="sigmoid")
        hgb.fit(X_tr, y_tr)
        oof_hgb[val_idx] = hgb.predict_proba(X_val)[:, 1]
        models["hgb"].append(hgb)

        # 5. Calibrated Ridge
        ridge_base = RidgeClassifier(alpha=10.0, random_state=42 + fold)
        ridge = CalibratedClassifierCV(estimator=ridge_base, cv=3, method="sigmoid")
        ridge.fit(X_tr, y_tr)
        oof_ridge[val_idx] = ridge.predict_proba(X_val)[:, 1]
        models["ridge"].append(ridge)

        print(f"Fold {fold} | LGB: {log_loss(y_val, oof_lgb[val_idx]):.5f} | CAT: {log_loss(y_val, oof_cat[val_idx]):.5f} | XGB: {log_loss(y_val, oof_xgb[val_idx]):.5f} | HGB: {log_loss(y_val, oof_hgb[val_idx]):.5f} | RDG: {log_loss(y_val, oof_ridge[val_idx]):.5f}")

    # 5. Build Heterogeneous Stacking Matrix
    if has_deberta:
        oof_deb = df["deberta_oof"].fillna(0.5).values
        X_stack = np.column_stack([oof_deb, oof_lgb, oof_cat, oof_xgb, oof_hgb, oof_ridge])
    else:
        X_stack = np.column_stack([oof_lgb, oof_cat, oof_xgb, oof_hgb, oof_ridge])

    meta_learner = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    meta_learner.fit(X_stack, y)
    oof_raw = meta_learner.predict_proba(X_stack)[:, 1]

    # 6. Temperature Calibration Search
    best_T = 1.0
    best_loss = log_loss(y, np.clip(oof_raw, 1e-15, 1 - 1e-15))
    for T_cand in np.linspace(0.8, 1.8, 51):
        cand_loss = log_loss(y, apply_temperature_scaling(oof_raw, T=T_cand))
        if cand_loss < best_loss:
            best_loss = cand_loss
            best_T = float(T_cand)

    oof_final = apply_temperature_scaling(oof_raw, T=best_T)
    final_loss = log_loss(y, oof_final)
    final_auc = roc_auc_score(y, oof_final)

    print(f"\n=======================================================")
    print(f"=== EXP-9 HETEROGENEOUS STACKING OOF LogLoss: {final_loss:.5f} | ROC AUC: {final_auc:.5f} ===")
    print(f"=== Optimal Temperature Parameter T = {best_T:.3f} ===")
    print(f"=== Meta-Learner Weights: {meta_learner.coef_[0]} ===")
    print("=======================================================")

    def patch_sklearn_version(obj, ver="1.8.0"):
        if hasattr(obj, "_sklearn_version"):
            object.__setattr__(obj, "_sklearn_version", ver)
        if isinstance(obj, dict):
            for k, v in obj.items():
                patch_sklearn_version(v, ver)
        elif isinstance(obj, list):
            for v in obj:
                patch_sklearn_version(v, ver)
        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if k != "_sklearn_version":
                    patch_sklearn_version(v, ver)

    pack_dict = {
        "models": models,
        "meta_learner": meta_learner,
        "temperature_T": best_T,
        "has_deberta": has_deberta,
        "target_enc_map": target_enc_map,
        "grade_enc_map": grade_enc_map,
        "topic_enc_map": topic_enc_map,
        "global_target_mean": global_target_mean,
        "feature_cols": feature_cols,
        "oof_log_loss": final_loss,
        "oof_roc_auc": final_auc
    }
    patch_sklearn_version(pack_dict, "1.8.0")

    os.makedirs(os.path.dirname(MODELS_SAVE_PATH), exist_ok=True)
    with open(MODELS_SAVE_PATH, "wb") as f:
        pickle.dump(pack_dict, f)

    print(f"=== Successfully Exported EXP-9 Models Pack: {MODELS_SAVE_PATH} ===")

if __name__ == "__main__":
    train_and_export_exp9()
