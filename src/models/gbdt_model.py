"""
Gradient Boosted Decision Trees Baseline Model & Probability Calibration
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets") if __file__.count(os.sep) > 3 else os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")

FEATURE_COLS = [
    "total_turns", "n_tutor_turns", "n_student_turns", "tutor_words", "student_words",
    "turn_ratio", "word_ratio", "avg_tutor_words", "avg_student_words",
    "tutor_questions", "student_questions", "tutor_q_ratio",
    "encouragement_count", "uncertainty_count", "duration_seconds"
]

def train_lgb_baseline():
    extracted_path = os.path.join(DATASETS_DIR, "train_extracted_features.csv")
    if not os.path.exists(extracted_path):
        print(f"Error: {extracted_path} not found. Run feature extraction first.")
        return

    df = pd.read_csv(extracted_path)
    print(f"Loaded dataset for training: {df.shape}")

    # Handle learning_objective_id as frequency encoding
    if "learning_objective_id" in df.columns:
        freq = df["learning_objective_id"].value_counts()
        df["learning_obj_freq"] = df["learning_objective_id"].map(freq)
        features = FEATURE_COLS + ["learning_obj_freq"]
    else:
        features = FEATURE_COLS

    df[features] = df[features].fillna(0)

    oof_preds = np.zeros(len(df))
    feature_importances = np.zeros(len(features))

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": 42
    }

    for fold in range(5):
        train_mask = df["fold"] != fold
        val_mask = df["fold"] == fold

        X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, "is_correct"]
        X_val, y_val = df.loc[val_mask, features], df.loc[val_mask, "is_correct"]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_mask] = val_preds
        feature_importances += model.feature_importance(importance_type="gain") / 5.0

        fold_loss = log_loss(y_val, val_preds)
        print(f"Fold {fold} Log Loss: {fold_loss:.5f}")

    raw_oof_loss = log_loss(df["is_correct"], oof_preds)
    raw_oof_auc = roc_auc_score(df["is_correct"], oof_preds)
    print(f"\n>>> Raw OOF Log Loss: {raw_oof_loss:.5f} | ROC AUC: {raw_oof_auc:.5f} <<<")

    # Platt Calibration
    eps = 1e-15
    clipped_preds = np.clip(oof_preds, eps, 1 - eps)
    logits = np.log(clipped_preds / (1 - clipped_preds)).reshape(-1, 1)

    calibrator = LogisticRegression(C=1e5, solver="lbfgs")
    calibrator.fit(logits, df["is_correct"])
    calibrated_preds = calibrator.predict_proba(logits)[:, 1]

    cal_oof_loss = log_loss(df["is_correct"], calibrated_preds)
    print(f">>> Calibrated OOF Log Loss: {cal_oof_loss:.5f} <<<")

    # Feature Importance Table
    fi_df = pd.DataFrame({"feature": features, "importance": feature_importances}).sort_values("importance", ascending=False)
    print("\nFeature Importances (Gain):\n", fi_df.head(10))

if __name__ == "__main__":
    train_lgb_baseline()
