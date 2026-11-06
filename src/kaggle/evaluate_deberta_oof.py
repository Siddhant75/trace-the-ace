import os
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

def evaluate():
    oof_path = os.path.join("datasets", "deberta_outputs", "deberta_oof.csv")
    df_oof = pd.read_csv(oof_path)
    print("OOF DataFrame shape:", df_oof.shape)
    print("Columns:", df_oof.columns.tolist())
    print(df_oof.head())

    mask = df_oof["deberta_oof"] > 0
    print(f"Number of evaluated samples: {mask.sum()} / {len(df_oof)}")

    y_true = df_oof.loc[mask, "is_correct"].values
    y_pred = np.clip(df_oof.loc[mask, "deberta_oof"].values, 0.025, 0.975)

    cv_logloss = log_loss(y_true, y_pred)
    cv_auc = roc_auc_score(y_true, y_pred)

    print("\n" + "="*60)
    print(f"DEBERTA-V3-LARGE 5-FOLD CV LOG LOSS: {cv_logloss:.5f}")
    print(f"DEBERTA-V3-LARGE 5-FOLD CV ROC AUC : {cv_auc:.5f}")
    print("="*60)

    # Copy to datasets/deberta_oof.csv
    shutil.copy(oof_path, os.path.join("datasets", "deberta_oof.csv"))
    print("Copied deberta_oof.csv to datasets/")

if __name__ == "__main__":
    evaluate()
