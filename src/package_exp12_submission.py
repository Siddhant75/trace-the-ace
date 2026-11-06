"""
Trace the Ace - EXP-12 Submission Package Builder (Grandmaster Multimodal Edition)
Builds submission_EXP12_grandmaster.zip containing:
1. main.py (Dialogue Trajectory Dynamics + Hierarchical Bayes + 3-Seed Bagged GBDT Ensemble + Joint SLSQP Calibrated Fusion)
2. assets/ensemble_exp12_models.pkl (with TF-IDF, SVD, trans_svd, bayes_maps, SLSQP optimal weights & temperature)
3. assets/deberta_tokenizer/ (Cleaned local tokenizer and config files)
4. assets/deberta_fold_0.pt ... assets/deberta_fold_4.pt (All 5 fine-tuned PyTorch state dicts)
"""
import os
import zipfile
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ZIP_OUT_PATH = os.path.join(BASE_DIR, "submission_EXP12_grandmaster.zip")

MAIN_PY_EXP12_CONTENT = '''"""
Trace the Ace - EXP-12 Inference Entrypoint (main.py - Grandmaster Multimodal Edition)
Executed inside DrivenData Cloud Runtime for Trace the Ace competition.
Implements:
1. Dialogue Trajectory Dynamics & Pedagogical Dialogue Feature Pipeline
2. Hierarchical Empirical Bayes Difficulty Shrinkage (Grade -> Topic -> Objective)
3. Multi-Seed 5-Fold Calibrated GBDT Bagged Ensemble (LGBM, CatBoost, XGBoost, HistGB, Ridge)
4. 5-Fold DeBERTa-v3-Large Multimodal Neural Network
5. Joint SLSQP-Optimized Non-Negative Convex Weights & Calibrated Temperature Scaling
"""
import os
import re
import glob
import pickle
import zipfile
import io
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

ENCOURAGEMENT_WORDS = set(["great", "good", "well", "excellent", "awesome", "nice", "yes", "exactly", "correct", "perfect", "brilliant", "spot", "super", "wonderful"])
UNCERTAINTY_WORDS = set(["dont", "don't", "know", "sure", "confused", "maybe", "hard", "help", "sorry", "difficult", "stuck", "what", "huh", "um", "uh"])
MATH_WORDS = set(["fraction", "numerator", "denominator", "divide", "division", "multiply", "multiplication", "percent", "percentage", "equation", "angle", "triangle", "ratio", "slope", "variable", "equals", "equal", "add", "addition", "subtract", "subtraction", "decimal", "graph", "plot", "formula", "algebra", "geometry", "area", "perimeter", "volume", "square", "circle", "radius", "diameter"])

def norm_str(s):
    return str(s).strip().lower()

def apply_temperature_scaling(probs, T=1.05):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    scaled_logits = logits / T
    cal_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return np.clip(cal_probs, 0.035, 0.965)

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

    numeric_turns = sum(student_contents.str.contains(r"\\d", regex=True)) if len(student_contents) > 0 else 0
    numeric_turns_per_word = numeric_turns / max(student_words, 1)

    digit_char_count = sum(len(re.findall(r"\\d", c)) for c in student_contents) if len(student_contents) > 0 else 0
    digit_chars_per_word = digit_char_count / max(student_words, 1)

    is_quiet_session = 1.0 if student_words < 100 else 0.0

    math_progression_rate = (late_math_count - early_math_count) / (student_words + 1.0)
    tutor_student_word_ratio = student_words / (tutor_words + 1.0)
    hesitation_per_math_term = student_hesitation_count / (math_term_count + 1.0)
    numeric_density_interaction = digit_chars_per_word * numeric_turns_per_word

    # Trajectory Dynamics
    if len(student_turn_lengths) >= 2:
        t_arr = np.arange(len(student_turn_lengths))
        t_mean = np.mean(t_arr)
        l_mean = np.mean(student_turn_lengths)
        denom = np.sum((t_arr - t_mean) ** 2)
        student_turn_length_slope = float(np.sum((t_arr - t_mean) * (student_turn_lengths - l_mean)) / max(denom, 1e-7))
    else:
        student_turn_length_slope = 0.0

    student_hesitation_per_turn = [sum(c.lower().count(h) for h in hesitation_phrases) for c in student_contents] if len(student_contents) > 0 else [0]
    if len(student_hesitation_per_turn) >= 2:
        t_arr = np.arange(len(student_hesitation_per_turn))
        t_mean = np.mean(t_arr)
        h_mean = np.mean(student_hesitation_per_turn)
        denom = np.sum((t_arr - t_mean) ** 2)
        student_hesitation_slope = float(np.sum((t_arr - t_mean) * (student_hesitation_per_turn - h_mean)) / max(denom, 1e-7))
    else:
        student_hesitation_slope = 0.0

    mid_start = int(total_turns * 0.2)
    mid_end = max(int(total_turns * 0.8), mid_start + 1)
    df_mid = df_trans.iloc[mid_start:mid_end]
    mid_tutor_q = sum(df_mid[df_mid["role"].astype(str).str.lower().isin(["tutor", "assistant", "teacher"])]["content"].dropna().astype(str).str.contains(r"\\?", regex=True))
    mid_student_turns = len(df_mid[df_mid["role"].astype(str).str.lower().isin(["student", "user"])])
    tutor_scaffolding_intensity = float(mid_tutor_q / max(mid_student_turns, 1))

    student_confidence_proxy = float((student_word_max * math_growth_ratio) / (student_hesitation_count + 1.0))

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
        "student_turn_length_slope": student_turn_length_slope,
        "student_hesitation_slope": student_hesitation_slope,
        "tutor_scaffolding_intensity": tutor_scaffolding_intensity,
        "student_confidence_proxy": student_confidence_proxy,
        "full_transcript_text": full_text,
        "student_transcript_text": student_text,
        "tutor_transcript_text": tutor_text,
        "final_3_student_text": final_3_student_text
    }

def format_multimodal_text(row):
    obj = str(row.get("learning_objective", "Mathematics")).strip()
    grade = str(row.get("grade_level", "Unknown")).strip()
    topic = str(row.get("topic_id", "Unknown")).strip()

    student_text = str(row.get("student_transcript_text", "")).strip()
    tutor_text = str(row.get("tutor_transcript_text", "")).strip()
    final_text = str(row.get("final_3_student_text", "")).strip()

    header = f"Task: Assess student math mastery | Grade: {grade} | Topic: {topic} | Objective: {obj} [SEP] "
    dialogue = f"Tutor: {tutor_text} [SEP] Student: {student_text} [SEP] Final Student Response: {final_text}"
    return header + dialogue

class DeBERTaMultimodalModel(nn.Module):
    def __init__(self, config_obj, num_tab_features=99, num_dropouts=5, dropout_rate=0.20):
        super().__init__()
        from transformers import AutoModel
        self.config = config_obj
        self.config.update({
            "output_hidden_states": True,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "use_cache": False
        })
        self.backbone = AutoModel.from_config(self.config)
        hidden_dim = self.config.hidden_size # 1024

        self.tab_mlp = nn.Sequential(
            nn.Linear(num_tab_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        fused_dim = (hidden_dim * 2) + 64
        self.classifier = nn.Linear(fused_dim, 1)

    def extract_text_features(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()

        sum_hidden = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_hidden / sum_mask

        last_hidden_masked = last_hidden.clone()
        last_hidden_masked[mask_expanded == 0] = -1e9
        max_pooled = torch.max(last_hidden_masked, dim=1)[0]

        return torch.cat([mean_pooled, max_pooled], dim=1)

    def forward(self, input_ids, attention_mask, tab_features):
        text_emb = self.extract_text_features(input_ids, attention_mask)
        tab_emb = self.tab_mlp(tab_features)
        fused = torch.cat([text_emb, tab_emb], dim=1)
        return self.classifier(fused).squeeze(-1)

def apply_bayes_maps(df, bayes_maps):
    mu_0 = bayes_maps.get("mu_0", 0.7035)
    grade_map = bayes_maps.get("grade_map", {})
    topic_map = bayes_maps.get("topic_map", {})
    obj_map = bayes_maps.get("obj_map", {})

    enc_grade = []
    enc_topic = []
    enc_obj = []

    for _, row in df.iterrows():
        g = norm_str(row.get("grade_level", "unknown")) if "grade_level" in df else "unknown"
        t = norm_str(row.get("topic_id", "unknown")) if "topic_id" in df else "unknown"
        o = norm_str(row.get("learning_objective", ""))

        g_val = grade_map.get(g, mu_0)
        t_val = topic_map.get((g, t), g_val)
        o_val = obj_map.get((g, t, o), t_val)

        enc_grade.append(g_val)
        enc_topic.append(t_val)
        enc_obj.append(o_val)

    res = pd.DataFrame(index=df.index)
    res["bayes_enc_grade"] = enc_grade
    res["bayes_enc_topic"] = enc_topic
    res["bayes_enc_obj"] = enc_obj
    res["bayes_difficulty_spread"] = np.array(enc_obj) - np.array(enc_grade)
    return res

def run_inference():
    print("=== DrivenData Container Inference Entrypoint (EXP-12 Grandmaster Multimodal Peak) ===")

    test_csv = os.path.join(DATA_DIR, "test_features.csv")
    if not os.path.exists(test_csv):
        test_csv = os.path.join(DATA_DIR, "test.csv")

    sub_format = os.path.join(DATA_DIR, "submission_format.csv")
    transcripts_dir = os.path.join(DATA_DIR, "test_transcripts")
    if not os.path.exists(transcripts_dir):
        transcripts_dir = os.path.join(DATA_DIR, "transcripts")

    df_test = pd.read_csv(test_csv)
    df_sub = pd.read_csv(sub_format)

    # 1. Load Complete Asset Pipeline Pack
    model_pack_path = os.path.join(ASSETS_DIR, "ensemble_exp12_models.pkl")
    with open(model_pack_path, "rb") as f:
        pack = pickle.load(f)

    models = pack["models"]
    seeds = pack.get("seeds", [42, 100, 2026])
    weights = pack.get("weights", {"deberta": 0.20, "lgb": 0.176, "cat": 0.176, "xgb": 0.144, "hgb": 0.144, "ridge": 0.160})
    temperature_T = pack.get("temperature_T", 1.05)
    word_tfidf = pack.get("word_tfidf")
    word_svd = pack.get("word_svd")
    trans_svd = pack.get("trans_svd")
    bayes_maps = pack.get("bayes_maps", {})
    target_enc_map = pack.get("target_enc_map", {})
    grade_enc_map = pack.get("grade_enc_map", {})
    topic_enc_map = pack.get("topic_enc_map", {})
    global_target_mean = pack.get("global_target_mean", 0.5)
    feature_cols = pack.get("feature_cols", [])
    tab_cols_deb = pack.get("tab_cols_deberta", [])
    tab_mean_deb = pack.get("tab_mean_deberta", {})
    tab_std_deb = pack.get("tab_std_deberta", {})

    print(f"Loaded EXP-12 Grandmaster Ensemble with SLSQP Weights: {weights} (Temperature T={temperature_T:.4f})")

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

    # 3. Compute Hierarchical Empirical Bayes Difficulty Encodings
    if bayes_maps:
        try:
            df_bayes_test = apply_bayes_maps(df_merged, bayes_maps)
            df_merged = pd.concat([df_merged, df_bayes_test], axis=1)
        except Exception:
            for c in ["bayes_enc_grade", "bayes_enc_topic", "bayes_enc_obj", "bayes_difficulty_spread"]:
                df_merged[c] = global_target_mean

    # 3b. Standard Target Encodings
    df_merged["target_enc_obj"] = df_merged["learning_objective"].apply(norm_str).map(target_enc_map).fillna(global_target_mean)
    if "grade_level" in df_merged.columns:
        df_merged["target_enc_grade"] = df_merged["grade_level"].apply(norm_str).map(grade_enc_map).fillna(global_target_mean)
    if "topic_id" in df_merged.columns:
        df_merged["target_enc_topic"] = df_merged["topic_id"].apply(norm_str).map(topic_enc_map).fillna(global_target_mean)

    # 4. Compute Word TF-IDF & TruncatedSVD Text Embeddings
    if word_tfidf is not None and word_svd is not None:
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
            from transformers import AutoTokenizer as BGETokenizer, AutoModel as BGEModel
            bge_tok = BGETokenizer.from_pretrained(hf_model_path, local_files_only=True)
            bge_mod = BGEModel.from_pretrained(hf_model_path, local_files_only=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                bge_mod = bge_mod.half().to(device)
            else:
                bge_mod = bge_mod.to(device)
            bge_mod.eval()

            full_texts = ("Task: Assess student math concept mastery | Objective: " + df_merged["learning_objective"].fillna("Mathematics") + " | Student Responses: " + df_merged["student_transcript_text"].fillna("") + " | Tutor Guidance: " + df_merged["tutor_transcript_text"].fillna("") + " | Final Outcome: " + df_merged["final_3_student_text"].fillna("")).tolist()
            student_texts = ("Task: Assess student response accuracy | Objective: " + df_merged["learning_objective"].fillna("Mathematics") + " | Student Responses: " + df_merged["student_transcript_text"].fillna("")).tolist()

            def embed_texts(t_list, max_l):
                embs = []
                batch_sz = 128 if device.type == "cuda" else 32
                with torch.no_grad():
                    for b_idx in range(0, len(t_list), batch_sz):
                        b_texts = t_list[b_idx:b_idx+batch_sz]
                        encoded = bge_tok(b_texts, padding=True, truncation=True, max_length=max_l, return_tensors="pt").to(device)
                        out = bge_mod(**encoded)
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

            del bge_mod, bge_tok
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"BGE extraction warning: {e}")
            for i in range(64):
                df_merged[f"trans_emb_{i}"] = 0.0
            for i in range(32):
                df_merged[f"student_emb_{i}"] = 0.0
    else:
        for i in range(64):
            df_merged[f"trans_emb_{i}"] = 0.0
        for i in range(32):
            df_merged[f"student_emb_{i}"] = 0.0

    # 5. Multi-Seed Bagged GBDT Tabular Predictions
    missing_cols = [col for col in feature_cols if col not in df_merged.columns]
    if missing_cols:
        missing_df = pd.DataFrame({col: 0.0 for col in missing_cols}, index=df_merged.index)
        df_merged = pd.concat([df_merged, missing_df], axis=1)

    X_test = df_merged[feature_cols].copy()
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    X_test = X_test[feature_cols]

    def predict_bagged(model_dict):
        all_preds = []
        for s in seeds:
            for m in model_dict[s]:
                all_preds.append(m.predict_proba(X_test)[:, 1])
        return np.mean(all_preds, axis=0)

    preds_lgb = predict_bagged(models["lgb"])
    preds_cat = predict_bagged(models["cat"])
    preds_xgb = predict_bagged(models["xgb"])
    preds_hgb = predict_bagged(models["hgb"])
    preds_ridge = predict_bagged(models["ridge"])

    # 6. Fine-Tuned DeBERTa-v3 Multimodal PyTorch Inference
    preds_deb = None
    tokenizer_dir = os.path.join(ASSETS_DIR, "deberta_tokenizer")
    deberta_checkpoints = sorted(glob.glob(os.path.join(ASSETS_DIR, "deberta_fold_*.pt")))

    if os.path.exists(tokenizer_dir) and deberta_checkpoints:
        try:
            print(f"=== Running Real DeBERTa-v3 Multimodal Inference across {len(deberta_checkpoints)} Folds ===")
            from transformers import AutoTokenizer, AutoConfig

            try:
                tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
            except Exception:
                from transformers import DebertaV2TokenizerFast
                tok = DebertaV2TokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True)

            cfg = AutoConfig.from_pretrained(tokenizer_dir, local_files_only=True)

            # Format multimodal text
            texts_deb = [format_multimodal_text(row) for _, row in df_merged.iterrows()]

            # Format and normalize 99 tabular features
            tab_data = []
            for col in tab_cols_deb:
                if col in df_merged.columns:
                    val = pd.to_numeric(df_merged[col], errors='coerce').fillna(0.0).values
                else:
                    val = np.zeros(len(df_merged))
                m = tab_mean_deb.get(col, 0.0)
                s = tab_std_deb.get(col, 1.0)
                norm_val = (val - m) / (s + 1e-7)
                tab_data.append(norm_val)

            tab_array = np.column_stack(tab_data).astype(np.float32)

            enc = tok(texts_deb, truncation=True, max_length=384, padding="max_length", return_tensors="pt")
            inp_ids = enc["input_ids"]
            att_masks = enc["attention_mask"]
            tab_tensors = torch.tensor(tab_array, dtype=torch.float32)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            use_amp = (device.type == "cuda")

            ds = TensorDataset(inp_ids, att_masks, tab_tensors)
            dl = DataLoader(ds, batch_size=32 if device.type == "cuda" else 8, shuffle=False)

            fold_preds = []
            for ckpt_path in deberta_checkpoints:
                print(f"Evaluating {os.path.basename(ckpt_path)}...")
                model = DeBERTaMultimodalModel(cfg, num_tab_features=tab_array.shape[1])
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(cleaned_state_dict, strict=True)

                if device.type == "cuda":
                    model = model.to(device)
                model.eval()

                preds_list = []
                with torch.no_grad():
                    for b_ids, b_mask, b_tab in dl:
                        b_ids = b_ids.to(device)
                        b_mask = b_mask.to(device)
                        b_tab = b_tab.to(device)
                        if use_amp:
                            with torch.amp.autocast('cuda', dtype=torch.float16):
                                logits = model(b_ids, b_mask, b_tab)
                        else:
                            logits = model(b_ids, b_mask, b_tab)
                        probs = torch.sigmoid(logits).cpu().numpy().flatten()
                        preds_list.extend(probs)

                fold_preds.append(np.array(preds_list))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            preds_deb = np.mean(fold_preds, axis=0)
            print(f"=== DeBERTa Inference Complete! Mean Prob: {np.mean(preds_deb):.4f}, Std: {np.std(preds_deb):.4f} ===")

        except Exception as e:
            print(f"DeBERTa PyTorch Inference Exception: {e}")
            preds_deb = (preds_lgb * 0.4 + preds_cat * 0.3 + preds_xgb * 0.3)
    else:
        print("DeBERTa models or tokenizer not found in assets, falling back to GBDT blend.")
        preds_deb = (preds_lgb * 0.4 + preds_cat * 0.3 + preds_xgb * 0.3)

    # 7. True Multimodal Non-Negative Convex Fusion (SLSQP Optimal Weights)
    w_deb = weights.get("deberta", 0.20)
    w_lgb = weights.get("lgb", 0.176)
    w_cat = weights.get("cat", 0.176)
    w_xgb = weights.get("xgb", 0.144)
    w_hgb = weights.get("hgb", 0.144)
    w_rdg = weights.get("ridge", 0.160)

    total_w = w_deb + w_lgb + w_cat + w_xgb + w_hgb + w_rdg
    w_deb, w_lgb, w_cat, w_xgb, w_hgb, w_rdg = [w / total_w for w in [w_deb, w_lgb, w_cat, w_xgb, w_hgb, w_rdg]]

    blend_raw = (
        w_deb * preds_deb +
        w_lgb * preds_lgb +
        w_cat * preds_cat +
        w_xgb * preds_xgb +
        w_hgb * preds_hgb +
        w_rdg * preds_ridge
    )

    # Apply Optimal Temperature Scaling
    blend = apply_temperature_scaling(blend_raw, T=temperature_T)

    df_sub["probability"] = df_merged["response_id"].map(dict(zip(df_merged["response_id"], blend))).fillna(float(np.mean(blend)))

    out_csv = os.path.join(BASE_DIR, "submission.csv")
    df_sub.to_csv(out_csv, index=False)
    print(f"=== Successfully wrote test predictions to {out_csv} (Shape: {df_sub.shape}) ===")

if __name__ == "__main__":
    run_inference()
'''

def build_exp12_package():
    if os.path.exists(ZIP_OUT_PATH):
        os.remove(ZIP_OUT_PATH)

    print(f"=== Creating Direct Stream Zip Archive: {ZIP_OUT_PATH} ===")
    with zipfile.ZipFile(ZIP_OUT_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        # 1. Write main.py directly from memory
        z.writestr("main.py", MAIN_PY_EXP12_CONTENT)
        print("  + Added main.py")

        # 2. Write model pack directly from src/models/ensemble_exp12_models.pkl
        model_src = os.path.join(BASE_DIR, "src", "models", "ensemble_exp12_models.pkl")
        z.write(model_src, "assets/ensemble_exp12_models.pkl")
        print("  + Added assets/ensemble_exp12_models.pkl")

        # 3. Write tokenizer files from src/models/deberta_tokenizer
        tok_dir = os.path.join(BASE_DIR, "src", "models", "deberta_tokenizer")
        if os.path.exists(tok_dir):
            for f in os.listdir(tok_dir):
                abs_f = os.path.join(tok_dir, f)
                if os.path.isfile(abs_f):
                    z.write(abs_f, f"assets/deberta_tokenizer/{f}")
                    print(f"  + Added assets/deberta_tokenizer/{f}")

        # 4. Write all 5 DeBERTa fold model checkpoints directly from src/models/
        for i in range(5):
            ckpt_src = os.path.join(BASE_DIR, "src", "models", f"deberta_fold_{i}.pt")
            if os.path.exists(ckpt_src):
                sz_mb = os.path.getsize(ckpt_src) / (1024 * 1024)
                z.write(ckpt_src, f"assets/deberta_fold_{i}.pt")
                print(f"  + Added assets/deberta_fold_{i}.pt ({sz_mb:.1f} MB)")

    print(f"=== Built: {ZIP_OUT_PATH} (Size: {os.path.getsize(ZIP_OUT_PATH) / (1024 * 1024):.2f} MB) ===")

if __name__ == "__main__":
    build_exp12_package()
