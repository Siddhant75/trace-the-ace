"""
EXP-10: True Multimodal Convex Fusion & Calibration Training Engine
Solves the meta-learner negative weight cancellation issue:
1. Fits 5-Fold GroupKFold Diversity GBDT Models (LGBM, CatBoost, XGBoost, HistGB, Ridge) on 195 features
2. Optimizes Non-Negative Convex Weights (SLSQP: w_i >= 0, sum(w_i) = 1.0) combining DeBERTa + 5 GBDTs
3. Calibrates Temperature Scaling Parameter T (optimal around T=1.20)
4. Deep stamps _sklearn_version = '1.8.0'
5. Exports ensemble_exp10_models.pkl
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
FEATURES_PATH = os.path.join(DATASETS_DIR, "train_extracted_features.csv")
DEBERTA_OOF_PATH = os.path.join(BASE_DIR, "deberta_oof.csv")
TRANS_EMB_PATH = os.path.join(BASE_DIR, "transformer_embeddings_oof.csv")
EXP8_PKL_PATH = os.path.join(BASE_DIR, "src", "models", "ensemble_models.pkl")
OUT_PKL_PATH = os.path.join(BASE_DIR, "src", "models", "ensemble_exp10_models.pkl")

def norm_str(s):
    return str(s).strip().lower()

def apply_temperature_scaling(probs, T=1.20):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    scaled_logits = logits / T
    cal_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return np.clip(cal_probs, 0.025, 0.975)

def train_exp10():
    print("=== EXP-10: Training True Multimodal Convex Fusion Ensemble ===")

    # 1. Load Data
    df = pd.read_csv(FEATURES_PATH)

    # Multi-Factor M-Estimate Target Encodings
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

    # 4. 5-Fold GroupKFold GBDT Training
    gkf = GroupKFold(n_splits=5)
    oof_lgb = np.zeros(len(df))
    oof_cat = np.zeros(len(df))
    oof_xgb = np.zeros(len(df))
    oof_hgb = np.zeros(len(df))
    oof_ridge = np.zeros(len(df))

    models = {"lgb": [], "cat": [], "xgb": [], "hgb": [], "ridge": []}

    print("\n--- Training 5-Fold Calibrated GBDT Models ---")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
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

    # 5. Non-Negative Convex Optimization (SLSQP)
    oof_deb = df["deberta_oof"].fillna(0.70).values
    oof_gbdt_avg = (oof_lgb + oof_cat + oof_xgb + oof_hgb + oof_ridge) / 5.0

    print(f"\n--- Standalone OOF Benchmarks ---")
    print(f"DeBERTa Standalone OOF : LogLoss = {log_loss(y, oof_deb):.5f} | AUC = {roc_auc_score(y, oof_deb):.5f}")
    print(f"GBDT 5-Model Avg OOF   : LogLoss = {log_loss(y, oof_gbdt_avg):.5f} | AUC = {roc_auc_score(y, oof_gbdt_avg):.5f}")

    # Optimize non-negative convex weights across all 6 model streams: [DeBERTa, LGB, CAT, XGB, HGB, Ridge]
    P_matrix = np.column_stack([oof_deb, oof_lgb, oof_cat, oof_xgb, oof_hgb, oof_ridge])
    model_names = ["deberta", "lgb", "cat", "xgb", "hgb", "ridge"]

    def loss_func(w):
        w = w / np.sum(w)
        p_blend = np.dot(P_matrix, w)
        p_blend = np.clip(p_blend, 1e-15, 1 - 1e-15)
        return log_loss(y, p_blend)

    # Initial guess: 55% DeBERTa, 9% each GBDT
    init_w = np.array([0.55, 0.09, 0.09, 0.09, 0.09, 0.09])
    bounds = [(0.45, 0.75), (0.03, 0.25), (0.03, 0.25), (0.03, 0.25), (0.03, 0.25), (0.01, 0.15)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(loss_func, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    opt_weights = res.x / np.sum(res.x)

    weights_dict = {name: float(w) for name, w in zip(model_names, opt_weights)}
    print(f"\n--- Optimal Non-Negative Convex Weights ---")
    for name, w in weights_dict.items():
        print(f"  {name:<10}: {w:.4f} ({w*100:.1f}%)")

    oof_blend_raw = np.dot(P_matrix, opt_weights)
    print(f"Convex Blend (Raw T=1.0) : LogLoss = {log_loss(y, oof_blend_raw):.5f} | AUC = {roc_auc_score(y, oof_blend_raw):.5f}")

    # 6. Temperature Scaling Search on Convex Blend
    best_T = 1.20
    best_loss = log_loss(y, apply_temperature_scaling(oof_blend_raw, T=1.20))
    for T_cand in np.linspace(1.0, 1.5, 51):
        cand_loss = log_loss(y, apply_temperature_scaling(oof_blend_raw, T=T_cand))
        if cand_loss < best_loss:
            best_loss = cand_loss
            best_T = float(T_cand)

    oof_calibrated = apply_temperature_scaling(oof_blend_raw, T=best_T)
    final_loss = log_loss(y, oof_calibrated)
    final_auc = roc_auc_score(y, oof_calibrated)

    print(f"\n=======================================================")
    print(f"=== EXP-10 TRUE MULTIMODAL CONVEX OOF LogLoss: {final_loss:.5f} | ROC AUC: {final_auc:.5f} ===")
    print(f"=== Calibrated Temperature Parameter T = {best_T:.3f} ===")
    print(f"=======================================================\n")

    # 7. Load NLP transformers & normalization stats
    with open(EXP8_PKL_PATH, "rb") as f:
        pack8 = pickle.load(f)

    word_tfidf = pack8["word_tfidf"]
    word_svd = pack8["word_svd"]
    trans_svd = pack8["trans_svd"]

    # DeBERTa tabular norm stats
    ignore_tab = ["session_id", "student_id", "learning_objective", "is_correct", "fold", "full_transcript_text", "student_transcript_text", "tutor_transcript_text", "final_3_student_text", "grade_level", "topic_id", "deberta_oof"]
    tab_cols = [c for c in df.columns if c not in ignore_tab and not c.startswith("oof_") and not c.startswith("norm_")]
    X_tab = df[tab_cols].select_dtypes(include=[np.number]).copy()
    tab_cols_numeric = list(X_tab.columns)
    tab_mean = X_tab.mean().to_dict()
    tab_std = X_tab.std().to_dict()

    # 8. Assemble Master Model Pack
    pack10 = {
        "models": models,
        "weights": weights_dict,
        "temperature_T": best_T,
        "clip_bounds": [0.035, 0.965],
        "has_deberta": True,
        "word_tfidf": word_tfidf,
        "word_svd": word_svd,
        "trans_svd": trans_svd,
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

    deep_stamp(pack10, "1.8.0")

    with open(OUT_PKL_PATH, "wb") as f:
        pickle.dump(pack10, f)

    print(f"=== Successfully Exported EXP-10 Model Pack: {OUT_PKL_PATH} ===")

if __name__ == "__main__":
    train_exp10()
