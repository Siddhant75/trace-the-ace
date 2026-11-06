"""
Trace the Ace - Code Submission Package Builder (EXP-8)
Creates a fully compliant DrivenData code execution package (submission.zip)
containing main.py and assets/ensemble_models.pkl with Temperature Scaling Calibration & Ratio Interactions.
"""
import os
import zipfile
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BUILD_DIR = os.path.join(BASE_DIR, "submission_package")
ZIP_OUT_PATH = os.path.join(BASE_DIR, "submission.zip")

MAIN_PY_CONTENT = '''"""
Trace the Ace - Inference Entrypoint (main.py - EXP-8)
Executed inside DrivenData Cloud Runtime for Trace the Ace competition.
Includes string-normalized target encodings, explicit ratio interactions, dual-stream BGE embeddings, and Temperature Scaling calibration.
"""
import os
import glob
import pickle
import zipfile
import io
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

ENCOURAGEMENT_WORDS = set(["great", "good", "well", "excellent", "awesome", "nice", "yes", "exactly", "correct", "perfect", "brilliant", "spot", "super", "wonderful"])
UNCERTAINTY_WORDS = set(["dont", "don't", "know", "sure", "confused", "maybe", "hard", "help", "sorry", "difficult", "stuck", "what", "huh", "um", "uh"])
MATH_WORDS = set(["fraction", "numerator", "denominator", "divide", "division", "multiply", "multiplication", "percent", "percentage", "equation", "angle", "triangle", "ratio", "slope", "variable", "equals", "equal", "add", "addition", "subtract", "subtraction", "decimal", "graph", "plot", "formula", "algebra", "geometry", "area", "perimeter", "volume", "square", "circle", "radius", "diameter"])

def norm_str(s):
    return str(s).strip().lower()

def apply_temperature_scaling(probs, T=1.20):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    scaled_logits = logits / T
    cal_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return np.clip(cal_probs, 0.03, 0.97)

def parse_transcript_df(sid, df_trans):
    if df_trans is None or df_trans.empty:
        return {"session_id": sid, "full_transcript_text": "", "student_transcript_text": "", "tutor_transcript_text": "", "final_3_student_text": ""}

    total_turns = len(df_trans)
    roles = df_trans["role"].astype(str).str.strip().str.lower()
    tutor_turns = df_trans[roles.isin(["tutor"])]
    student_turns = df_trans[roles.isin(["student", "user"])]

    n_tutor_turns = len(tutor_turns)
    n_student_turns = len(student_turns)

    tutor_contents = tutor_turns["content"].dropna().astype(str)
    student_contents = student_turns["content"].dropna().astype(str)

    tutor_text = " ".join(tutor_contents)
    student_text = " ".join(student_contents)
    full_text = " ".join(df_trans["content"].dropna().astype(str))

    tutor_words = len(tutor_text.split())
    student_words = len(student_text.split())

    turn_ratio = n_student_turns / max(n_tutor_turns, 1)
    word_ratio = student_words / max(tutor_words, 1)

    avg_tutor_words = tutor_words / max(n_tutor_turns, 1)
    avg_student_words = student_words / max(n_student_turns, 1)

    student_turn_lengths = [len(c.split()) for c in student_contents] if len(student_contents) > 0 else [0]
    student_word_std = float(np.std(student_turn_lengths)) if len(student_turn_lengths) > 1 else 0.0
    student_word_max = int(np.max(student_turn_lengths)) if len(student_turn_lengths) > 0 else 0

    early_student_turns = student_contents.iloc[:3] if len(student_contents) >= 3 else student_contents
    late_student_turns = student_contents.iloc[-3:] if len(student_contents) >= 3 else student_contents

    early_words = sum(len(c.split()) for c in early_student_turns)
    late_words = sum(len(c.split()) for c in late_student_turns)
    early_vs_late_ratio = late_words / max(early_words, 1)

    tutor_questions = sum(tutor_contents.str.contains(r"\\?", regex=True))
    student_questions = sum(student_contents.str.contains(r"\\?", regex=True))
    tutor_q_ratio = tutor_questions / max(n_tutor_turns, 1)

    tutor_words_list = tutor_text.lower().split()
    student_words_list = student_text.lower().split()

    encouragement_count = sum(1 for w in tutor_words_list if w in ENCOURAGEMENT_WORDS)
    uncertainty_count = sum(1 for w in student_words_list if w in UNCERTAINTY_WORDS)
    math_term_count = sum(1 for w in student_words_list if w in MATH_WORDS)
    math_term_density = math_term_count / max(student_words, 1)

    late_student_turns = student_contents.iloc[-3:] if len(student_contents) >= 3 else student_contents
    final_3_student_text = " ".join(late_student_turns)
    hesitation_phrases = ["um", "uh", "maybe", "i think", "not sure", "is it", "sorry", "stuck", "what if"]
    student_hesitation_count = sum(student_text.lower().count(h) for h in hesitation_phrases)

    n_opening_turns = max(int(total_turns * 0.2), 1)
    opening_student_turns = student_contents.iloc[:n_opening_turns] if len(student_contents) >= n_opening_turns else student_contents
    opening_turn_ratio = len(opening_student_turns) / max(total_turns, 1)

    n_ending_turns = max(int(total_turns * 0.2), 1)
    ending_student_turns = student_contents.iloc[-n_ending_turns:] if len(student_contents) >= n_ending_turns else student_contents
    ending_words = sum(len(c.split()) for c in ending_student_turns)
    ending_word_ratio = ending_words / max(student_words, 1)

    early_math_count = sum(1 for w in " ".join(opening_student_turns).lower().split() if w in MATH_WORDS)
    late_math_count = sum(1 for w in " ".join(ending_student_turns).lower().split() if w in MATH_WORDS)
    math_growth_ratio = (late_math_count + 1.0) / (early_math_count + 1.0)

    duration_seconds = 0.0
    if "timestamp" in df_trans.columns and len(df_trans) > 1:
        try:
            ts = pd.to_datetime(df_trans["timestamp"], format="mixed", errors="coerce").dropna().sort_values()
            if len(ts) > 1:
                duration_seconds = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
        except Exception:
            duration_seconds = 0.0

    duration_per_turn = duration_seconds / max(total_turns, 1)

    import re
    numeric_turns = sum(student_contents.str.contains(r"\\d", regex=True)) if len(student_contents) > 0 else 0
    numeric_turns_per_word = numeric_turns / max(student_words, 1)

    digit_char_count = sum(len(re.findall(r"\\d", c)) for c in student_contents) if len(student_contents) > 0 else 0
    digit_chars_per_word = digit_char_count / max(student_words, 1)

    is_quiet_session = 1.0 if student_words < 100 else 0.0

    # EXP-8 Lever 3 Ratio Interactions
    math_progression_rate = (late_math_count - early_math_count) / (student_words + 1.0)
    tutor_student_word_ratio = student_words / (tutor_words + 1.0)
    hesitation_per_math_term = student_hesitation_count / (math_term_count + 1.0)
    numeric_density_interaction = digit_chars_per_word * numeric_turns_per_word

    return {
        "session_id": sid,
        "total_turns": total_turns,
        "n_tutor_turns": n_tutor_turns,
        "n_student_turns": n_student_turns,
        "tutor_words": tutor_words,
        "student_words": student_words,
        "turn_ratio": turn_ratio,
        "word_ratio": word_ratio,
        "avg_tutor_words": avg_tutor_words,
        "avg_student_words": avg_student_words,
        "student_word_std": student_word_std,
        "student_word_max": student_word_max,
        "early_vs_late_ratio": early_vs_late_ratio,
        "tutor_questions": tutor_questions,
        "student_questions": student_questions,
        "tutor_q_ratio": tutor_q_ratio,
        "encouragement_count": encouragement_count,
        "uncertainty_count": uncertainty_count,
        "math_term_count": math_term_count,
        "math_term_density": math_term_density,
        "early_math_count": early_math_count,
        "late_math_count": late_math_count,
        "math_growth_ratio": math_growth_ratio,
        "student_hesitation_count": student_hesitation_count,
        "opening_turn_ratio": opening_turn_ratio,
        "ending_word_ratio": ending_word_ratio,
        "duration_seconds": duration_seconds,
        "duration_per_turn": duration_per_turn,
        "numeric_turns_per_word": numeric_turns_per_word,
        "digit_chars_per_word": digit_chars_per_word,
        "is_quiet_session": is_quiet_session,
        "math_progression_rate": math_progression_rate,
        "tutor_student_word_ratio": tutor_student_word_ratio,
        "hesitation_per_math_term": hesitation_per_math_term,
        "numeric_density_interaction": numeric_density_interaction,
        "full_transcript_text": full_text,
        "student_transcript_text": student_text,
        "tutor_transcript_text": tutor_text,
        "final_3_student_text": final_3_student_text
    }

def run_inference():
    test_features_path = os.path.join(DATA_DIR, "test_features.csv")
    sub_format_path = os.path.join(DATA_DIR, "submission_format.csv")
    transcripts_dir = os.path.join(DATA_DIR, "test_transcripts")

    if not os.path.exists(test_features_path):
        print(f"Warning: {test_features_path} not found. Searching fallback...")
        test_features_path = glob.glob(os.path.join(DATA_DIR, "*features*.csv"))[0]

    df_test = pd.read_csv(test_features_path)
    df_sub = pd.read_csv(sub_format_path)

    # 1. Load Complete Asset Pipeline Pack
    model_pack_path = os.path.join(ASSETS_DIR, "ensemble_models.pkl")
    with open(model_pack_path, "rb") as f:
        pack = pickle.load(f)

    models = pack["models"]
    temperature_T = pack.get("temperature_T", 1.20)
    word_tfidf = pack["word_tfidf"]
    word_svd = pack["word_svd"]
    trans_svd = pack.get("trans_svd")
    target_enc_map = pack.get("target_enc_map", {})
    grade_enc_map = pack.get("grade_enc_map", {})
    topic_enc_map = pack.get("topic_enc_map", {})
    global_target_mean = pack.get("global_target_mean", 0.5)
    feature_cols = pack.get("feature_cols", [])

    # 2. Extract Pedagogical Dialogue Features
    transcript_rows = []
    if os.path.exists(transcripts_dir) and os.path.isdir(transcripts_dir):
        for sid in df_test["session_id"].unique():
            csv_path = os.path.join(transcripts_dir, f"{sid}.csv")
            if not os.path.exists(csv_path):
                candidates = glob.glob(os.path.join(transcripts_dir, "**", f"{sid}.csv"), recursive=True)
                if candidates:
                    csv_path = candidates[0]
            if os.path.exists(csv_path):
                try:
                    df_t = pd.read_csv(csv_path)
                    transcript_rows.append(parse_transcript_df(sid, df_t))
                except Exception:
                    transcript_rows.append({"session_id": sid, "full_transcript_text": "", "student_transcript_text": "", "tutor_transcript_text": "", "final_3_student_text": ""})
            else:
                transcript_rows.append({"session_id": sid, "full_transcript_text": "", "student_transcript_text": "", "tutor_transcript_text": "", "final_3_student_text": ""})

    df_trans_feats = pd.DataFrame(transcript_rows)
    if not df_trans_feats.empty and "session_id" in df_trans_feats.columns:
        df_merged = pd.merge(df_test, df_trans_feats, on="session_id", how="left")
    else:
        df_merged = df_test.copy()
        df_merged["full_transcript_text"] = ""
        df_merged["student_transcript_text"] = ""
        df_merged["tutor_transcript_text"] = ""
        df_merged["final_3_student_text"] = ""

    # 3. Compute String-Normalized Multi-Factor Target Encodings
    df_merged["target_enc_obj"] = df_merged["learning_objective"].apply(norm_str).map(target_enc_map).fillna(global_target_mean)
    if "grade_level" in df_merged.columns:
        df_merged["target_enc_grade"] = df_merged["grade_level"].apply(norm_str).map(grade_enc_map).fillna(global_target_mean)
    if "topic_id" in df_merged.columns:
        df_merged["target_enc_topic"] = df_merged["topic_id"].apply(norm_str).map(topic_enc_map).fillna(global_target_mean)

    # 4. Compute Word TF-IDF & TruncatedSVD Text Embeddings
    texts = df_merged["full_transcript_text"].fillna("").astype(str)
    X_tfidf_test = word_tfidf.transform(texts)
    svd_feats_test = word_svd.transform(X_tfidf_test)
    for i in range(64):
        df_merged[f"svd_{i}"] = svd_feats_test[:, i]

    # 4b. Pre-loaded HuggingFace Dense Transformer Dual-Stream Embeddings (BAAI/bge-large-en-v1.5)
    hf_model_path = "/code_execution/huggingface_models/BAAI/bge-large-en-v1.5"
    if not os.path.exists(hf_model_path) and os.path.exists(os.path.join(ASSETS_DIR, "bge-large-en-v1.5")):
        hf_model_path = os.path.join(ASSETS_DIR, "bge-large-en-v1.5")

    if os.path.exists(hf_model_path):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(hf_model_path, local_files_only=True)
            model = AutoModel.from_pretrained(hf_model_path, local_files_only=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                model = model.half().to(device)
            else:
                model = model.to(device)
            model.eval()

            full_texts = ("Task: Assess student math concept mastery | Objective: " + df_merged["learning_objective"].fillna("Mathematics") + " | Student Responses: " + df_merged["student_transcript_text"].fillna("") + " | Tutor Guidance: " + df_merged["tutor_transcript_text"].fillna("") + " | Final Outcome: " + df_merged["final_3_student_text"].fillna("")).tolist()
            student_texts = ("Task: Assess student response accuracy | Objective: " + df_merged["learning_objective"].fillna("Mathematics") + " | Student Responses: " + df_merged["student_transcript_text"].fillna("")).tolist()

            def embed_texts(t_list, max_l):
                embs = []
                batch_sz = 128 if device.type == "cuda" else 32
                with torch.no_grad():
                    for b_idx in range(0, len(t_list), batch_sz):
                        b_texts = t_list[b_idx:b_idx+batch_sz]
                        encoded = tokenizer(b_texts, padding=True, truncation=True, max_length=max_l, return_tensors="pt").to(device)
                        out = model(**encoded)
                        mask = encoded["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
                        pooled = torch.sum(out[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                        embs.append(pooled.float().cpu().numpy())
                return np.vstack(embs)

            mat_full = embed_texts(full_texts, max_l=256)
            mat_student = embed_texts(student_texts, max_l=192)

            if isinstance(trans_svd, dict):
                comp_full = trans_svd["svd_full"].transform(mat_full)
                comp_student = trans_svd["svd_student"].transform(mat_student)
            elif trans_svd is not None:
                comp_full = trans_svd.transform(mat_full)
                comp_student = mat_student[:, :32]
            else:
                comp_full = mat_full[:, :64]
                comp_student = mat_student[:, :32]

            for i in range(comp_full.shape[1]):
                df_merged[f"trans_emb_{i}"] = comp_full[:, i]
            for i in range(comp_student.shape[1]):
                df_merged[f"student_emb_{i}"] = comp_student[:, i]

        except Exception as e:
            for i in range(64):
                df_merged[f"trans_emb_{i}"] = 0.0
            for i in range(32):
                df_merged[f"student_emb_{i}"] = 0.0
    else:
        for i in range(64):
            df_merged[f"trans_emb_{i}"] = 0.0
        for i in range(32):
            df_merged[f"student_emb_{i}"] = 0.0

    missing_cols = [col for col in feature_cols if col not in df_merged.columns]
    if missing_cols:
        missing_df = pd.DataFrame({col: 0.0 for col in missing_cols}, index=df_merged.index)
        df_merged = pd.concat([df_merged, missing_df], axis=1)

    X_test = df_merged[feature_cols].copy()
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    X_test = X_test[feature_cols]

    # 5. Generate Calibrated 5-Model Ensemble Predictions & Stacking
    preds_lgb = np.mean([m.predict_proba(X_test)[:, 1] for m in models["lgb"]], axis=0)
    preds_cat = np.mean([m.predict_proba(X_test)[:, 1] for m in models["cat"]], axis=0)
    preds_xgb = np.mean([m.predict_proba(X_test)[:, 1] for m in models["xgb"]], axis=0)
    preds_hgb = np.mean([m.predict_proba(X_test)[:, 1] for m in models.get("hgb", models["lgb"])], axis=0)
    preds_ridge = np.mean([m.predict_proba(X_test)[:, 1] for m in models["ridge"]], axis=0)

    meta_learner = pack.get("meta_learner")
    if meta_learner is not None:
        X_stack = np.column_stack([preds_lgb, preds_cat, preds_xgb, preds_hgb, preds_ridge])
        blend_raw = meta_learner.predict_proba(X_stack)[:, 1]
    else:
        weights = pack.get("weights", [0.2, 0.2, 0.2, 0.2, 0.2])
        blend_raw = (
            weights[0] * preds_lgb +
            weights[1] * preds_cat +
            weights[2] * preds_xgb +
            weights[3] * preds_hgb +
            weights[4] * preds_ridge
        )

    # Lever 4: Temperature Calibration & [0.03, 0.97] Clipping
    blend = apply_temperature_scaling(blend_raw, T=temperature_T)

    df_sub["probability"] = df_merged["response_id"].map(dict(zip(df_merged["response_id"], blend))).fillna(float(np.mean(blend)))

    out_csv = os.path.join(BASE_DIR, "submission.csv")
    df_sub.to_csv(out_csv, index=False)
    print(f"=== Successfully wrote test predictions to {out_csv} (Shape: {df_sub.shape}) ===")

if __name__ == "__main__":
    run_inference()
'''

def build_submission_package():
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR, exist_ok=True)
    assets_dir = os.path.join(BUILD_DIR, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Write updated main.py
    main_py_path = os.path.join(BUILD_DIR, "main.py")
    with open(main_py_path, "w") as f:
        f.write(MAIN_PY_CONTENT)
    print(f"=== Generated main.py at {main_py_path} ===")

    # 2. Copy trained model pack
    model_src = os.path.join(BASE_DIR, "src", "models", "ensemble_models.pkl")
    model_dst = os.path.join(assets_dir, "ensemble_models.pkl")
    shutil.copy(model_src, model_dst)
    print(f"=== Copied model weights to {model_dst} ===")

    # 3. Create ZIP archive
    if os.path.exists(ZIP_OUT_PATH):
        os.remove(ZIP_OUT_PATH)

    with zipfile.ZipFile(ZIP_OUT_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(BUILD_DIR):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, BUILD_DIR)
                if rel_path.startswith("data") or rel_path.startswith("submission.csv") or "bge-large-en-v1.5" in rel_path:
                    continue
                z.write(abs_path, rel_path)

    print(f"=== Successfully built DrivenData Code Submission ZIP: {ZIP_OUT_PATH} ===")
    print(f"Zip File Size: {os.path.getsize(ZIP_OUT_PATH) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    build_submission_package()
