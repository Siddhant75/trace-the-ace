"""
GroupKFold Cross-Validation Framework for Trace the Ace Competition
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")

def create_folds(n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    features_path = os.path.join(DATASETS_DIR, "train_features_TMQTWsB.csv")
    labels_path = os.path.join(DATASETS_DIR, "train_labels_44ujmj2.csv")

    df_feat = pd.read_csv(features_path)
    df_labels = pd.read_csv(labels_path)

    # Merge features and labels on response_id
    df = pd.merge(df_feat, df_labels, on="response_id", how="inner")
    
    # Initialize fold column
    df["fold"] = -1
    
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["session_id"]
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["is_correct"], groups)):
        df.loc[val_idx, "fold"] = fold

    print(f"=== GroupKFold Created ({n_splits} Folds) ===")
    print(f"Total Samples: {len(df)}")
    print(f"Total Unique Sessions: {df['session_id'].nunique()}")
    
    # Verify zero leakage across folds
    for f in range(n_splits):
        val_sessions = set(df[df["fold"] == f]["session_id"])
        train_sessions = set(df[df["fold"] != f]["session_id"])
        overlap = val_sessions.intersection(train_sessions)
        print(f"Fold {f}: Val Samples = {len(df[df['fold']==f])}, Val Sessions = {len(val_sessions)}, Leakage Overlap = {len(overlap)}")
        assert len(overlap) == 0, f"Data leakage detected in fold {f}!"
        
    folds_path = os.path.join(DATASETS_DIR, "train_folds.csv")
    df.to_csv(folds_path, index=False)
    print(f"Folds saved to: {folds_path}")
    return df

if __name__ == "__main__":
    create_folds()
