"""
Official DrivenData Local Test Harness & Leaderboard Simulator for Trace the Ace.
Replicates the platform cloud container execution, format validation, and Log Loss scoring.
"""
import os
import shutil
import zipfile
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

import sys
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
if len(sys.argv) > 1 and sys.argv[1].endswith(".zip"):
    ZIP_PATH = os.path.abspath(sys.argv[1]) if os.path.isabs(sys.argv[1]) else os.path.join(BASE_DIR, sys.argv[1])
elif os.path.exists(os.path.join(BASE_DIR, "submission_EXP9_deberta.zip")):
    ZIP_PATH = os.path.join(BASE_DIR, "submission_EXP9_deberta.zip")
else:
    ZIP_PATH = os.path.join(BASE_DIR, "submission.zip")
HARNESS_DIR = os.path.join(BASE_DIR, "local_test_harness")
SANDBOX_DIR = os.path.join(HARNESS_DIR, "code_execution")
DATA_DIR = os.path.join(SANDBOX_DIR, "data")
LOG_PATH = os.path.join(HARNESS_DIR, "log.txt")

def setup_harness_environment():
    print("=================================================================")
    print("=== Setting Up Official DrivenData Local Test Harness Sandbox ===")
    print("=================================================================")

    # Clean existing harness sandbox
    if os.path.exists(HARNESS_DIR):
        shutil.rmtree(HARNESS_DIR)

    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Prepare simulated test data from train_folds.csv (Holdout Fold 4 or full set)
    folds_path = os.path.join(DATASETS_DIR, "train_folds.csv")
    df_full = pd.read_csv(folds_path)

    # Use Fold 4 as holdout test set to get realistic OOF evaluation
    df_holdout = df_full[df_full["fold"] == 4].copy()

    # Create test_features.csv (no labels)
    df_holdout[["response_id", "session_id", "learning_objective"]].to_csv(
        os.path.join(DATA_DIR, "test_features.csv"), index=False
    )

    # Create submission_format.csv
    df_holdout[["response_id"]].assign(probability=0.5).to_csv(
        os.path.join(DATA_DIR, "submission_format.csv"), index=False
    )

    # Copy test_transcripts if available
    transcripts_src = os.path.join(DATASETS_DIR, "train_transcripts")
    transcripts_dst = os.path.join(DATA_DIR, "test_transcripts")

    if os.path.exists(transcripts_src):
        os.makedirs(transcripts_dst, exist_ok=True)
        sample_sids = set(df_holdout["session_id"].unique())
        for sid in sample_sids:
            src_csv = os.path.join(transcripts_src, f"{sid}.csv")
            if os.path.exists(src_csv):
                shutil.copy(src_csv, os.path.join(transcripts_dst, f"{sid}.csv"))
    elif os.path.exists(os.path.join(DATASETS_DIR, "train_transcripts.zip")):
        print("Extracting holdout test transcripts from train_transcripts.zip...")
        os.makedirs(transcripts_dst, exist_ok=True)
        sample_sids = set(df_holdout["session_id"].unique())
        with zipfile.ZipFile(os.path.join(DATASETS_DIR, "train_transcripts.zip"), "r") as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    sid = os.path.basename(name).replace(".csv", "")
                    if sid in sample_sids:
                        target_file = os.path.join(transcripts_dst, f"{sid}.csv")
                        with open(target_file, "wb") as f_out:
                            f_out.write(z.read(name))

    print(f"Sandbox Data Ready: {len(df_holdout)} holdout samples, {len(os.listdir(transcripts_dst)) if os.path.exists(transcripts_dst) else 0} transcripts.")
    return df_holdout

def test_submission_harness(df_holdout):
    print("\n=================================================================")
    print("=== Step 1: Validating Submission Zip Archive ===")
    print("=================================================================")

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Missing submission.zip at {ZIP_PATH}!")

    # Check zip contents
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        namelist = z.namelist()
        print("Archive entries:")
        for name in namelist:
            print(f" - {name}")

        if "main.py" not in namelist:
            raise ValueError("FAIL: main.py is missing from root of submission.zip!")

        for name in namelist:
            if name.startswith("data/") or name.startswith("data\\"):
                raise ValueError(f"FAIL: Archive contains prohibited folder '{name}'. data/ must NOT be in submission.zip!")

    print("=== Zip Validation PASSED ===")

    print("\n=================================================================")
    print("=== Step 2: Unpacking & Executing Entrypoint (main.py) ===")
    print("=================================================================")

    # Unpack submission into sandbox root
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(SANDBOX_DIR)

    main_py_path = os.path.join(SANDBOX_DIR, "main.py")
    out_csv_path = os.path.join(SANDBOX_DIR, "submission.csv")

    # Run main.py non-interactively
    with open(LOG_PATH, "w") as log_file:
        proc = subprocess.Popen(
            ["python", "main.py"],
            cwd=SANDBOX_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"FAIL: main.py exited with error code {proc.returncode}! Check log: {LOG_PATH}")

    print("=== Execution Completed Successfully ===")

    print("\n=================================================================")
    print("=== Step 3: Validating Output submission.csv Format ===")
    print("=================================================================")

    if not os.path.exists(out_csv_path):
        raise FileNotFoundError(f"FAIL: submission.csv was not generated at {out_csv_path}!")

    df_sub = pd.read_csv(out_csv_path)
    sub_format = pd.read_csv(os.path.join(DATA_DIR, "submission_format.csv"))

    # Validate shape
    if len(df_sub) != len(sub_format):
        raise ValueError(f"FAIL: Row count mismatch! Expected {len(sub_format)}, got {len(df_sub)}.")

    # Validate columns
    expected_cols = ["response_id", "probability"]
    if list(df_sub.columns) != expected_cols:
        raise ValueError(f"FAIL: Column mismatch! Expected {expected_cols}, got {list(df_sub.columns)}.")

    # Validate response_ids match exactly
    if not (df_sub["response_id"].values == sub_format["response_id"].values).all():
        raise ValueError("FAIL: response_id alignment mismatch with submission_format.csv!")

    # Check NaNs
    null_count = df_sub["probability"].isnull().sum()
    if null_count > 0:
        raise ValueError(f"FAIL: Output submission.csv contains {null_count} NaN values!")

    # Check bounds
    p_min = df_sub["probability"].min()
    p_max = df_sub["probability"].max()
    print(f"Predicted Probability Bounds: Min={p_min:.5f}, Max={p_max:.5f}, Mean={df_sub['probability'].mean():.5f}")

    if p_min < 0.0 or p_max > 1.0:
        raise ValueError(f"FAIL: Probabilities outside [0, 1] range! Min={p_min}, Max={p_max}")

    print("=== Format & Bound Validation PASSED ===")

    print("\n=================================================================")
    print("=== Step 4: Scoring Prediction Quality (Log Loss & ROC AUC) ===")
    print("=================================================================")

    y_true = df_holdout["is_correct"].values
    y_pred = df_sub["probability"].values

    loss = log_loss(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    print(f"[RESULTS] Local Holdout Performance Results:")
    print(f" - Log Loss (Leaderboard Metric): {loss:.5f}")
    print(f" - ROC AUC  (Secondary Metric)  : {auc:.5f}")
    print("=================================================================\n")

    return loss, auc

def run_harness():
    df_holdout = setup_harness_environment()
    loss, auc = test_submission_harness(df_holdout)
    return loss, auc

if __name__ == "__main__":
    run_harness()
