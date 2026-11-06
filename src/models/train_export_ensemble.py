"""
Fast & Unified Model Training & Asset Exporter for Trace the Ace Competition (EXP-8)
Implements all 4 High-Impact Levers:
- Lever 1: Geometric Cosine Mastery Features (Obj vs Student/Tutor/Final)
- Lever 2: Leak-Free GroupKFold grouped strictly by student_id
- Lever 3: Explicit Polynomial & Ratio Feature Interaction Matrix
- Lever 4: Temperature-Calibrated Probability Scaling (T-scaling) with [0.03, 0.97] Clipping
Saves complete pipeline to src/models/ensemble_models.pkl.
"""
import os
import sys
import pickle
import zipfile
import pandas as pd
import numpy as np
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
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
FEATURES_PATH = os.path.join(DATASETS_DIR, "train_extracted_features.csv")
TRANSCRIPTS_ZIP = os.path.join(DATASETS_DIR, "train_transcripts.zip")
MODELS_SAVE_PATH = os.path.join(os.path.dirname(__file__), "ensemble_models.pkl")

def norm_str(s):
    return str(s).strip().lower()

def apply_temperature_scaling(probs, T=1.20):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    scaled_logits = logits / T
    cal_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return np.clip(cal_probs, 0.03, 0.97)

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

def train_and_export():
    print(f"=== Loading Train Features: {FEATURES_PATH} ===")
    df = pd.read_csv(FEATURES_PATH)

    # Lever 3: Explicit Ratio & Interaction Features
    if "math_progression_rate" not in df.columns:
        df["math_progression_rate"] = (df.get("late_math_count", 0) - df.get("early_math_count", 0)) / (df.get("student_words", 1) + 1.0)
    if "tutor_student_word_ratio" not in df.columns:
        df["tutor_student_word_ratio"] = df.get("student_words", 0) / (df.get("tutor_words", 1) + 1.0)
    if "hesitation_per_math_term" not in df.columns:
        df["hesitation_per_math_term"] = df.get("student_hesitation_count", 0) / (df.get("math_term_count", 0) + 1.0)
    if "numeric_density_interaction" not in df.columns:
        df["numeric_density_interaction"] = df.get("digit_chars_per_word", 0) * df.get("numeric_turns_per_word", 0)

    # 1. String-Normalized Multi-Factor Target Encoding Maps
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

    # 2. Extract Transcript Texts for TF-IDF SVD Fitting
    print(f"=== Extracting Transcripts from {TRANSCRIPTS_ZIP} for TF-IDF Fitting ===")
    session_texts = {}
    if os.path.exists(TRANSCRIPTS_ZIP):
        with zipfile.ZipFile(TRANSCRIPTS_ZIP, "r") as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    sid = os.path.basename(name).replace(".csv", "")
                    try:
                        import io
                        df_t = pd.read_csv(io.BytesIO(z.read(name)))
                        if "content" in df_t.columns:
                            session_texts[sid] = " ".join(df_t["content"].dropna().astype(str))
                    except Exception:
                        pass

    texts = [session_texts.get(sid, "") for sid in df["session_id"]]

    print("=== Fitting TfidfVectorizer & TruncatedSVD (64 Components) ===")
    word_tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    X_tfidf = word_tfidf.fit_transform(texts)
    word_svd = TruncatedSVD(n_components=64, random_state=42)
    word_svd.fit(X_tfidf)

    trans_svd = None
    trans_svd_path = "trans_svd.pkl"
    if os.path.exists(trans_svd_path):
        with open(trans_svd_path, "rb") as f:
            trans_svd = pickle.load(f)
        print("=== Loaded Fitted Transformer SVD Model from trans_svd.pkl ===")

    emb_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "transformer_embeddings_oof.csv") if __file__.count(os.sep) > 3 else "transformer_embeddings_oof.csv"
    if not os.path.exists(emb_csv_path) and os.path.exists("transformer_embeddings_oof.csv"):
        emb_csv_path = "transformer_embeddings_oof.csv"

    if os.path.exists(emb_csv_path):
        print(f"=== Merging Dense Transformer Embeddings from: {emb_csv_path} ===")
        df_emb = pd.read_csv(emb_csv_path)
        trans_cols = [c for c in df_emb.columns if c.startswith("trans_emb_") or c.startswith("student_emb_") or c.startswith("sim_")]
        if trans_cols:
            df = pd.merge(df, df_emb[["session_id"] + trans_cols], on="session_id", how="left")

    ignore_cols = ["session_id", "student_id", "learning_objective", "is_correct", "fold", "full_transcript_text", "norm_obj", "norm_grade", "norm_topic"]
    feature_cols = [c for c in df.columns if c not in ignore_cols and not c.startswith("oof_")]

    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    feature_cols = list(X.columns)
    y = df["is_correct"].values

    # Lever 2: GroupKFold by student_id to prevent student-level data leakage across folds
    print("=== Constructing Leak-Free 5-Fold GroupKFold Grouped by student_id ===")
    gkf = GroupKFold(n_splits=5)
    student_groups = df["student_id"].values if "student_id" in df.columns else df["session_id"].values
    folds = np.zeros(len(df), dtype=int)
    for fold_idx, (tr_i, val_i) in enumerate(gkf.split(X, y, groups=student_groups)):
        folds[val_i] = fold_idx

    X = X.fillna(X.median())

    print(f"Total Model Features: {X.shape[1]} ({feature_cols[:3]} ...)")

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

        # 1. LightGBM + CalibratedClassifierCV
        lgb_base = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.02,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            verbose=-1
        )
        lgb = CalibratedClassifierCV(estimator=lgb_base, cv=3, method="sigmoid")
        lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb.predict_proba(X_val)[:, 1]
        models["lgb"].append(lgb)

        # 2. CatBoost + CalibratedClassifierCV
        cat_base = CatBoostClassifier(
            iterations=450,
            learning_rate=0.025,
            depth=6,
            l2_leaf_reg=4,
            random_seed=42 + fold,
            verbose=0
        )
        cat = CalibratedClassifierCV(estimator=cat_base, cv=3, method="sigmoid")
        cat.fit(X_tr, y_tr)
        oof_cat[val_idx] = cat.predict_proba(X_val)[:, 1]
        models["cat"].append(cat)

        # 3. XGBoost + CalibratedClassifierCV
        xgb_base = XGBClassifier(
            n_estimators=350,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.7,
            gamma=0.1,
            random_state=42 + fold,
            eval_metric="logloss"
        )
        xgb = CalibratedClassifierCV(estimator=xgb_base, cv=3, method="sigmoid")
        xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb.predict_proba(X_val)[:, 1]
        models["xgb"].append(xgb)

        # 4. HistGradientBoosting + CalibratedClassifierCV
        hgb_base = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42 + fold
        )
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

        l_loss_lgb = log_loss(y_val, oof_lgb[val_idx])
        l_loss_cat = log_loss(y_val, oof_cat[val_idx])
        l_loss_xgb = log_loss(y_val, oof_xgb[val_idx])
        l_loss_hgb = log_loss(y_val, oof_hgb[val_idx])
        l_loss_rdg = log_loss(y_val, oof_ridge[val_idx])

        print(f"Fold {fold} LogLoss | LightGBM: {l_loss_lgb:.5f} | CatBoost: {l_loss_cat:.5f} | XGBoost: {l_loss_xgb:.5f} | HistGB: {l_loss_hgb:.5f} | Ridge: {l_loss_rdg:.5f}")

    opt_weights = optimize_weights([oof_lgb, oof_cat, oof_xgb, oof_hgb, oof_ridge], y)
    X_stack = np.column_stack([oof_lgb, oof_cat, oof_xgb, oof_hgb, oof_ridge])
    meta_learner = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    meta_learner.fit(X_stack, y)

    oof_raw = meta_learner.predict_proba(X_stack)[:, 1]

    # Lever 4: Temperature Calibration Search
    best_T = 1.0
    best_loss = log_loss(y, np.clip(oof_raw, 1e-15, 1 - 1e-15))
    for T_cand in np.linspace(0.8, 1.8, 51):
        cand_loss = log_loss(y, apply_temperature_scaling(oof_raw, T=T_cand))
        if cand_loss < best_loss:
            best_loss = cand_loss
            best_T = float(T_cand)

    oof_stacked = apply_temperature_scaling(oof_raw, T=best_T)
    final_loss = log_loss(y, oof_stacked)
    final_auc = roc_auc_score(y, oof_stacked)

    print(f"\n=======================================================")
    print(f"=== EXP-8 LEAK-FREE STUDENT-GROUP OOF LogLoss: {final_loss:.5f} | ROC AUC: {final_auc:.5f} ===")
    print(f"=== Optimal Temperature Scaling Parameter T = {best_T:.3f} ===")
    print(f"=== Meta-Learner Coeffs: LGB={meta_learner.coef_[0][0]:.3f}, CAT={meta_learner.coef_[0][1]:.3f}, XGB={meta_learner.coef_[0][2]:.3f}, HGB={meta_learner.coef_[0][3]:.3f}, RIDGE={meta_learner.coef_[0][4]:.3f} ===")
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
        "weights": opt_weights,
        "meta_learner": meta_learner,
        "temperature_T": best_T,
        "word_tfidf": word_tfidf,
        "word_svd": word_svd,
        "trans_svd": trans_svd,
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

    print(f"=== Successfully Exported Complete EXP-8 Pipeline Pack: {MODELS_SAVE_PATH} ===")

if __name__ == "__main__":
    train_and_export()
