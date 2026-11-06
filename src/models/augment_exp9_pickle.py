"""
Augment EXP-9 Ensemble Models Pack with:
1. word_tfidf, word_svd, trans_svd from EXP-8 ensemble_models.pkl
2. DeBERTa tabular normalization stats (tab_mean, tab_std, tab_cols_deberta)
3. Deep recursive stamping of _sklearn_version = "1.8.0" on all estimators
"""
import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
FEATURES_PATH = os.path.join(DATASETS_DIR, "train_extracted_features.csv")
EXP9_PKL_PATH = os.path.join(BASE_DIR, "src", "models", "ensemble_exp9_models.pkl")
EXP8_PKL_PATH = os.path.join(BASE_DIR, "src", "models", "ensemble_models.pkl")

def augment_pickle():
    print(f"=== Augmenting EXP-9 Models Pack ===")
    with open(EXP9_PKL_PATH, "rb") as f:
        pack9 = pickle.load(f)

    with open(EXP8_PKL_PATH, "rb") as f:
        pack8 = pickle.load(f)

    # 1. Inject NLP feature transformers
    pack9["word_tfidf"] = pack8["word_tfidf"]
    pack9["word_svd"] = pack8["word_svd"]
    pack9["trans_svd"] = pack8["trans_svd"]
    print("Injected word_tfidf, word_svd, and trans_svd from EXP-8 pack.")

    # 2. Compute and inject DeBERTa tabular normalization stats (99 dimensions)
    df_feats = pd.read_csv(FEATURES_PATH)
    ignore_cols = ["session_id", "student_id", "learning_objective", "is_correct", "fold", "full_transcript_text", "student_transcript_text", "tutor_transcript_text", "final_3_student_text", "grade_level", "topic_id"]
    tab_cols = [c for c in df_feats.columns if c not in ignore_cols and not c.startswith("oof_") and not c.startswith("norm_")]
    X_tab = df_feats[tab_cols].select_dtypes(include=[np.number]).copy()

    tab_cols_numeric = list(X_tab.columns)
    tab_mean = X_tab.mean().to_dict()
    tab_std = X_tab.std().to_dict()

    pack9["tab_cols_deberta"] = tab_cols_numeric
    pack9["tab_mean_deberta"] = tab_mean
    pack9["tab_std_deberta"] = tab_std
    print(f"Computed DeBERTa normalization stats for {len(tab_cols_numeric)} tabular features.")

    # 3. Deep recursive stamping of _sklearn_version
    visited = set()
    def deep_stamp(obj, ver="1.8.0"):
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        try:
            object.__setattr__(obj, "_sklearn_version", ver)
        except Exception:
            try:
                setattr(obj, "_sklearn_version", ver)
            except Exception:
                pass

        if isinstance(obj, dict):
            for k, v in obj.items():
                deep_stamp(v, ver)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                deep_stamp(v, ver)
        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                deep_stamp(v, ver)

    deep_stamp(pack9, "1.8.0")
    print(f"Deep stamped _sklearn_version = '1.8.0' across {len(visited)} objects.")
    print(f"Meta-learner _sklearn_version: {getattr(pack9['meta_learner'], '_sklearn_version', 'None')}")

    with open(EXP9_PKL_PATH, "wb") as f:
        pickle.dump(pack9, f)

    print(f"=== Successfully Saved Augmented EXP-9 Pack: {EXP9_PKL_PATH} ===")

if __name__ == "__main__":
    augment_pickle()
