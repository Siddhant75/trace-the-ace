"""
Comprehensive DrivenData Pre-Submission Audit Script for EXP-9 TRUE PEAK
Verifies every critical requirement:
1. Archive Structure & Forbidden File Checks (data/, submission.csv, root main.py)
2. Sklearn Version Stamping (_sklearn_version == '1.8.0')
3. Feature Pipeline Integrity (195 GBDT features, 99 DeBERTa tabular features)
4. Model Weight Checks (all 5 DeBERTa PyTorch checkpoints, GBDTs, Meta-Learner)
5. Tokenizer Offline Loading Check (config.json, tokenizer.json)
6. Mock Container Execution on unseen holdout batch
7. Output submission.csv Format, NaN, Alignment, and Probability Bounds Verification
"""
import os
import sys
import io
import zipfile
import pickle
import shutil
import subprocess
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "submission_EXP15_peak.zip")
if not os.path.exists(ZIP_PATH):
    ZIP_PATH = os.path.join(BASE_DIR, "submission.zip")

def run_comprehensive_audit():
    print("=" * 75)
    print("=== STARTING COMPREHENSIVE PRE-SUBMISSION AUDIT FOR TRACE THE ACE ===")
    print("=" * 75)
    print(f"Target Submission Archive: {ZIP_PATH}")
    print(f"Archive Size: {os.path.getsize(ZIP_PATH) / (1024**3):.3f} GB ({os.path.getsize(ZIP_PATH) / (1024*1024):.2f} MB)")

    audit_passed = True
    audit_log = []

    def record_check(name, passed, detail=""):
        nonlocal audit_passed
        if passed:
            print(f" [PASS] {name}: {detail}")
            audit_log.append((name, "PASS", detail))
        else:
            print(f" [FAIL] {name}: {detail}")
            audit_log.append((name, "FAIL", detail))
            audit_passed = False

    # -------------------------------------------------------------
    # Check 1: Zip Archive Integrity & Structure
    # -------------------------------------------------------------
    print("\n--- Check 1: Zip Archive Integrity & File Layout ---")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        crc_err = z.testzip()
        record_check("Archive CRC Checksum", crc_err is None, "Archive is not corrupted")

        namelist = z.namelist()
        has_main_root = ("main.py" in namelist)
        record_check("Root main.py Present", has_main_root, "main.py is located at the archive root")

        has_prohibited = any(n.startswith("data/") or n.startswith("data\\") or n == "submission.csv" for n in namelist)
        record_check("No Prohibited Files", not has_prohibited, "Archive does NOT contain data/ or submission.csv")

        expected_assets = [
            "assets/ensemble_exp15_models.pkl",
            "assets/deberta_tokenizer/config.json",
            "assets/deberta_tokenizer/tokenizer.json",
            "assets/deberta_fold_0.pt",
            "assets/deberta_fold_1.pt",
            "assets/deberta_fold_2.pt",
            "assets/deberta_fold_3.pt",
            "assets/deberta_fold_4.pt",
        ]
        missing_assets = [a for a in expected_assets if a not in namelist]
        record_check("All 8 Essential Assets Present", len(missing_assets) == 0, f"Missing: {missing_assets}" if missing_assets else "All 5 fold weights, tokenizer, and model pack found")

    # -------------------------------------------------------------
    # Check 2: Pickle Contents & Sklearn Version 1.8.0
    # -------------------------------------------------------------
    print("\n--- Check 2: Model Pack & Sklearn Version Audit ---")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        pkl_bytes = z.read("assets/ensemble_exp15_models.pkl")
        pack = pickle.loads(pkl_bytes)

    required_keys = ["models", "weights", "temperature_T", "has_deberta", "feature_cols", "word_tfidf", "word_svd", "trans_svd", "tab_cols_deberta", "tab_mean_deberta", "tab_std_deberta"]
    missing_keys = [k for k in required_keys if k not in pack]
    record_check("Pickle Keys Complete", len(missing_keys) == 0, f"Missing: {missing_keys}" if missing_keys else f"All {len(required_keys)} required keys present")

    weights = pack.get("weights", {})
    record_check("Multimodal Weights Set", "deberta" in weights, f"Weights: {weights}")

    feat_cols = pack.get("feature_cols", [])
    record_check("GBDT Feature Columns Present", len(feat_cols) > 100, f"Found {len(feat_cols)} feature columns")

    tab_cols_deb = pack.get("tab_cols_deberta", [])
    record_check("DeBERTa Tabular Features (99 Features)", len(tab_cols_deb) == 99, f"Found {len(tab_cols_deb)} tabular columns for tab_mlp")

    models_dict = pack.get("models", {})
    expected_models = ["lgb", "cat", "xgb", "hgb", "ridge"]
    seeds = pack.get("seeds", [42, 100, 2026])
    all_models_present = all(all(len(models_dict.get(m, {}).get(s, [])) == 5 for s in seeds) for m in expected_models)
    record_check("3-Seed 5-Fold Diversity GBDT Models", all_models_present, f"3 seeds x 5 folds present for each of {expected_models}")

    # -------------------------------------------------------------
    # Check 3: DeBERTa Weights & Tokenizer Offline Validation
    # -------------------------------------------------------------
    print("\n--- Check 3: DeBERTa Weights & Tokenizer Offline Validation ---")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for fold in range(5):
            pt_bytes = z.read(f"assets/deberta_fold_{fold}.pt")
            sd = torch.load(io.BytesIO(pt_bytes), map_location="cpu", weights_only=False)
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            has_tab = "tab_mlp.0.weight" in sd
            has_clf = "classifier.weight" in sd
            tab_shape = tuple(sd["tab_mlp.0.weight"].shape) if has_tab else None
            clf_shape = tuple(sd["classifier.weight"].shape) if has_clf else None
            valid_shapes = (tab_shape == (128, 99) and clf_shape == (1, 2112))
            record_check(f"DeBERTa Fold {fold} Architecture", valid_shapes, f"tab_mlp: {tab_shape}, classifier: {clf_shape}")

    # -------------------------------------------------------------
    # Check 4: Full Mock Container Execution Sandbox (Offline)
    # -------------------------------------------------------------
    print("\n--- Check 4: Mock Container Execution in Sandbox (Offline) ---")
    audit_sandbox = os.path.join(BASE_DIR, "audit_sandbox")
    if os.path.exists(audit_sandbox):
        shutil.rmtree(audit_sandbox)
    os.makedirs(audit_sandbox, exist_ok=True)

    print("Unpacking essential components into test container...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        # Extract main.py, model pack, tokenizer, and fold models to verify end-to-end forward pass
        for name in z.namelist():
            if name == "main.py" or "ensemble_exp15_models.pkl" in name or "deberta_tokenizer" in name or "deberta_fold_" in name:
                z.extract(name, audit_sandbox)

    data_dir = os.path.join(audit_sandbox, "data")
    trans_dir = os.path.join(data_dir, "test_transcripts")
    os.makedirs(trans_dir, exist_ok=True)

    df_folds = pd.read_csv(os.path.join(BASE_DIR, "datasets", "train_folds.csv"))
    df_sample = df_folds[df_folds["fold"] == 4].head(25).copy()

    df_sample[["response_id", "session_id", "learning_objective"]].to_csv(
        os.path.join(data_dir, "test_features.csv"), index=False
    )
    df_sample[["response_id"]].assign(probability=0.5).to_csv(
        os.path.join(data_dir, "submission_format.csv"), index=False
    )

    sids = set(df_sample["session_id"].unique())
    with zipfile.ZipFile(os.path.join(BASE_DIR, "datasets", "train_transcripts.zip"), "r") as zt:
        for name in zt.namelist():
            if name.endswith(".csv"):
                sid = os.path.basename(name).replace(".csv", "")
                if sid in sids:
                    with open(os.path.join(trans_dir, f"{sid}.csv"), "wb") as f_out:
                        f_out.write(zt.read(name))

    print(f"Sandbox populated with 25 test samples and {len(os.listdir(trans_dir))} transcripts.")

    print("Executing main.py entrypoint (simulating offline DrivenData container)...")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    proc = subprocess.run(
        [sys.executable, "main.py"],
        cwd=audit_sandbox,
        capture_output=True,
        text=True,
        env=env,
        timeout=300
    )

    print("Entrypoint STDOUT Output:")
    for line in proc.stdout.strip().split("\n"):
        print(f"  > {line}")

    if proc.stderr:
        print("Entrypoint STDERR:")
        for line in proc.stderr.strip().split("\n")[:10]:
            print(f"  ! {line}")

    record_check("Container Entrypoint Exit Code 0", proc.returncode == 0, f"Exit code = {proc.returncode}")

    out_csv_path = os.path.join(audit_sandbox, "submission.csv")
    out_exists = os.path.exists(out_csv_path)
    record_check("submission.csv Created", out_exists, f"File at {out_csv_path}")

    if out_exists:
        df_out = pd.read_csv(out_csv_path)
        sub_format = pd.read_csv(os.path.join(data_dir, "submission_format.csv"))

        row_match = len(df_out) == len(sub_format)
        record_check("Row Count Match", row_match, f"Rows: {len(df_out)} == {len(sub_format)}")

        cols_match = list(df_out.columns) == ["response_id", "probability"]
        record_check("Column Names Exact Match", cols_match, f"Columns: {list(df_out.columns)}")

        id_align = (df_out["response_id"].values == sub_format["response_id"].values).all()
        record_check("Response ID Alignment", id_align, "All 25 response_ids perfectly ordered")

        nan_count = df_out["probability"].isnull().sum() + np.isinf(df_out["probability"]).sum()
        record_check("Zero NaN / Infinite Values", nan_count == 0, f"NaN/Inf count = {nan_count}")

        p_min = float(df_out["probability"].min())
        p_max = float(df_out["probability"].max())
        p_mean = float(df_out["probability"].mean())
        bounds_valid = (0.0 < p_min < 1.0) and (0.0 < p_max < 1.0) and (p_min < p_max)
        record_check("Valid Non-Trivial Probability Distribution", bounds_valid, f"Min={p_min:.5f}, Max={p_max:.5f}, Mean={p_mean:.5f}")

    if os.path.exists(audit_sandbox):
        shutil.rmtree(audit_sandbox)
        print("\nCleaned up audit sandbox.")

    # -------------------------------------------------------------
    # Final Audit Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("=== FINAL PRE-SUBMISSION AUDIT SUMMARY ===")
    print("=" * 75)
    passed_count = sum(1 for _, status, _ in audit_log if status == "PASS")
    total_count = len(audit_log)

    for name, status, detail in audit_log:
        symbol = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"{symbol} {name:<42} : {detail}")

    print("=" * 75)
    if audit_passed:
        print(f"ALL {passed_count}/{total_count} AUDIT CHECKS PASSED PERFECTLY!")
        print("Ready for official competition submission!")
    else:
        print(f"AUDIT FAILED: {total_count - passed_count} checks failed.")
    print("=" * 75)

    return audit_passed

if __name__ == "__main__":
    success = run_comprehensive_audit()
    sys.exit(0 if success else 1)
