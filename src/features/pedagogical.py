"""
Pedagogical Dialogue Feature Extraction & Word TF-IDF SVD Pipeline
Extracts high-signal educational features & 64-component Word TF-IDF TruncatedSVD.
"""
import os
import zipfile
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets") if __file__.count(os.sep) > 3 else os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
TRANSCRIPTS_ZIP = os.path.join(DATASETS_DIR, "train_transcripts.zip")

ENCOURAGEMENT_WORDS = set(["great", "good", "well", "excellent", "awesome", "nice", "yes", "exactly", "correct", "perfect", "brilliant", "spot", "super", "wonderful"])
UNCERTAINTY_WORDS = set(["dont", "don't", "know", "sure", "confused", "maybe", "hard", "help", "sorry", "difficult", "stuck", "what", "huh", "um", "uh"])
MATH_WORDS = set(["fraction", "numerator", "denominator", "divide", "division", "multiply", "multiplication", "percent", "percentage", "equation", "angle", "triangle", "ratio", "slope", "variable", "equals", "equal", "add", "addition", "subtract", "subtraction", "decimal", "graph", "plot", "formula", "algebra", "geometry", "area", "perimeter", "volume", "square", "circle", "radius", "diameter"])

def parse_transcript_bytes(args):
    sid, csv_bytes = args
    if not csv_bytes:
        return {"session_id": sid, "full_transcript_text": ""}

    import io
    try:
        df_trans = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception:
        return {"session_id": sid, "full_transcript_text": ""}

    if df_trans.empty:
        return {"session_id": sid, "full_transcript_text": ""}

    total_turns = len(df_trans)
    roles = df_trans["role"].astype(str).str.strip().str.lower()
    tutor_mask = roles.isin(["tutor", "assistant", "teacher"])
    student_mask = roles.isin(["student", "user"])

    tutor_turns = df_trans[tutor_mask]
    student_turns = df_trans[student_mask]

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

    tutor_questions = sum(tutor_contents.str.contains(r"\?", regex=True))
    student_questions = sum(student_contents.str.contains(r"\?", regex=True))
    tutor_q_ratio = tutor_questions / max(n_tutor_turns, 1)

    tutor_words_list = tutor_text.lower().split()
    student_words_list = student_text.lower().split()

    encouragement_count = sum(1 for w in tutor_words_list if w in ENCOURAGEMENT_WORDS)
    uncertainty_count = sum(1 for w in student_words_list if w in UNCERTAINTY_WORDS)
    math_term_count = sum(1 for w in student_words_list if w in MATH_WORDS)
    math_term_density = math_term_count / max(student_words, 1)

    # Specific Speaker & Dialogue Phase Features
    final_3_student_text = " ".join(late_student_turns)
    hesitation_phrases = ["um", "uh", "maybe", "i think", "not sure", "is it", "sorry", "stuck", "what if"]
    student_hesitation_count = sum(student_text.lower().count(h) for h in hesitation_phrases)

    n_opening_turns = max(int(total_turns * 0.2), 1)
    opening_student_turns = student_contents.iloc[:n_opening_turns] if len(student_contents) >= n_opening_turns else student_contents
    opening_words = sum(len(c.split()) for c in opening_student_turns)
    opening_turn_ratio = len(opening_student_turns) / max(total_turns, 1)

    n_ending_turns = max(int(total_turns * 0.2), 1)
    ending_student_turns = student_contents.iloc[-n_ending_turns:] if len(student_contents) >= n_ending_turns else student_contents
    ending_words = sum(len(c.split()) for c in ending_student_turns)
    ending_word_ratio = ending_words / max(student_words, 1)

    # Dialogue Phase Math Term Progression
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
    numeric_turns = sum(student_contents.str.contains(r"\d", regex=True)) if len(student_contents) > 0 else 0
    numeric_turns_per_word = numeric_turns / max(student_words, 1)

    digit_char_count = sum(len(re.findall(r"\d", c)) for c in student_contents) if len(student_contents) > 0 else 0
    digit_chars_per_word = digit_char_count / max(student_words, 1)

    is_quiet_session = 1.0 if student_words < 100 else 0.0

    # EXP-8 Lever 3: Explicit Polynomial & Ratio Feature Interactions
    math_progression_rate = (late_math_count - early_math_count) / (student_words + 1.0)
    tutor_student_word_ratio = student_words / (tutor_words + 1.0)
    hesitation_per_math_term = student_hesitation_count / (math_term_count + 1.0)
    numeric_density_interaction = digit_chars_per_word * numeric_turns_per_word

    # EXP-13: Robust Non-Parametric Quantile Dynamics & Scaffolding Ratios
    if len(student_turn_lengths) > 0:
        student_turn_p75 = float(np.percentile(student_turn_lengths, 75))
        student_turn_p25 = float(np.percentile(student_turn_lengths, 25))
        student_turn_iqr = float(student_turn_p75 - student_turn_p25)
    else:
        student_turn_p75 = 0.0
        student_turn_p25 = 0.0
        student_turn_iqr = 0.0

    tutor_math_term_count = sum(1 for w in tutor_words_list if w in MATH_WORDS)
    tutor_math_scaffold_ratio = float((tutor_math_term_count + 1.0) / (math_term_count + 1.0))
    student_hesitation_density = float(student_hesitation_count / (student_words + 1.0))

    # Middle phase scaffolding ratio (turns between 0.2*T and 0.8*T)
    mid_start = int(total_turns * 0.2)
    mid_end = max(int(total_turns * 0.8), mid_start + 1)
    df_mid = df_trans.iloc[mid_start:mid_end]
    mid_tutor_q = sum(df_mid[df_mid["role"].astype(str).str.lower().isin(["tutor", "assistant", "teacher"])]["content"].dropna().astype(str).str.contains(r"\?", regex=True))
    mid_student_turns = len(df_mid[df_mid["role"].astype(str).str.lower().isin(["student", "user"])])
    tutor_scaffolding_intensity = float(mid_tutor_q / max(mid_student_turns, 1))
    student_confidence_proxy = float((student_word_max * math_growth_ratio) / (student_hesitation_count + 1.0))

    # === EXP-15: Domain-Invariant Pedagogical Dynamics & Tutoring Moves ===
    SCAFFOLD_PHRASES = ["can you", "what do you think", "how would you", "why", "explain", "what is", "how many", "try", "show me", "tell me", "which column", "shall we", "look at"]
    DIRECT_PHRASES = ["the answer is", "it is", "you need to", "remember that", "so it's", "that's because", "the way to", "let me show", "we write"]
    CAUSAL_WORDS = ["because", "since", "therefore", "so that", "means that", "if we", "which makes", "implies"]
    SELF_REPAIR_WORDS = ["oh wait", "never mind", "no wait", "wrong one", "actually", "i mean", "oh i see", "wait, ", "no, wait"]
    TRANSITION_PHRASES = ["next lesson", "complete this lesson", "moving on to", "next slide", "new lesson", "upload another", "completed this lesson"]
    WHY_PHRASES = ["why", "how did you", "how come", "reason", "explain", "how would you"]

    scaffold_count = sum(tutor_text.lower().count(p) for p in SCAFFOLD_PHRASES)
    direct_count = sum(tutor_text.lower().count(p) for p in DIRECT_PHRASES)
    scaffold_to_direct_ratio = float(scaffold_count / max(direct_count, 1.0))

    causal_count = sum(student_text.lower().count(w) for w in CAUSAL_WORDS)
    student_causal_reasoning_density = float(causal_count / max(student_words, 1.0))

    self_repair_arc_count = sum(student_text.lower().count(w) for w in SELF_REPAIR_WORDS)

    # Key moments: Student hesitation followed by tutor scaffolding in next turns
    key_moments_count = 0
    if len(df_trans) > 1:
        for i in range(len(df_trans) - 1):
            if student_mask.iloc[i]:
                s_txt = str(df_trans.iloc[i]["content"]).lower()
                if any(w in s_txt for w in ["um", "uh", "don't know", "dont know", "not sure", "confused", "stuck", "wait", "huh"]):
                    for j in range(i + 1, min(i + 3, len(df_trans))):
                        if tutor_mask.iloc[j]:
                            t_txt = str(df_trans.iloc[j]["content"]).lower()
                            if any(p in t_txt for p in SCAFFOLD_PHRASES):
                                key_moments_count += 1
                            break

    # Passive agreement vs Substantive knowledge construction
    n_monosyllabic_student = sum(1 for c in student_contents if len(c.split()) <= 2)
    student_monosyllabic_ratio = float(n_monosyllabic_student / max(n_student_turns, 1.0))

    n_substantive_student = sum(1 for c in student_contents if len(c.split()) >= 12)
    student_substantive_turn_count = float(n_substantive_student)

    n_monologue_tutor = sum(1 for c in tutor_contents if len(c.split()) >= 40)
    tutor_monologue_ratio = float(n_monologue_tutor / max(n_tutor_turns, 1.0))

    curriculum_velocity_count = float(sum(tutor_text.lower().count(p) for p in TRANSITION_PHRASES))

    # Early vs Late Epistemic Uncertainty Decay
    early_confusion = sum(sum(str(c).lower().count(w) for w in UNCERTAINTY_WORDS) for c in opening_student_turns)
    late_confusion = sum(sum(str(c).lower().count(w) for w in ending_student_turns) for c in ending_student_turns)
    epistemic_uncertainty_decay = float((early_confusion - late_confusion) / max(early_confusion + 1.0, 1.0))

    tutor_why_count = sum(1 for c in tutor_contents if any(w in str(c).lower() for w in WHY_PHRASES) and "?" in str(c))
    tutor_why_prompt_ratio = float(tutor_why_count / max(n_tutor_turns, 1.0))

    early_tutor_turns = tutor_contents.iloc[:n_opening_turns] if len(tutor_contents) >= n_opening_turns else tutor_contents
    early_tutor_math = sum(1 for w in " ".join(early_tutor_turns).lower().split() if w in MATH_WORDS)
    student_math_uptake_rate = float((late_math_count + 1.0) / (early_tutor_math + 1.0))

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
        "student_turn_p75": student_turn_p75,
        "student_turn_p25": student_turn_p25,
        "student_turn_iqr": student_turn_iqr,
        "early_vs_late_ratio": early_vs_late_ratio,
        "tutor_questions": tutor_questions,
        "student_questions": student_questions,
        "tutor_q_ratio": tutor_q_ratio,
        "encouragement_count": encouragement_count,
        "uncertainty_count": uncertainty_count,
        "math_term_count": math_term_count,
        "math_term_density": math_term_density,
        "tutor_math_term_count": tutor_math_term_count,
        "tutor_math_scaffold_ratio": tutor_math_scaffold_ratio,
        "early_math_count": early_math_count,
        "late_math_count": late_math_count,
        "math_growth_ratio": math_growth_ratio,
        "student_hesitation_count": student_hesitation_count,
        "student_hesitation_density": student_hesitation_density,
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
        "tutor_scaffolding_intensity": tutor_scaffolding_intensity,
        "student_confidence_proxy": student_confidence_proxy,
        "scaffold_to_direct_ratio": scaffold_to_direct_ratio,
        "student_causal_reasoning_density": student_causal_reasoning_density,
        "self_repair_arc_count": self_repair_arc_count,
        "key_moments_count": key_moments_count,
        "student_monosyllabic_ratio": student_monosyllabic_ratio,
        "student_substantive_turn_count": student_substantive_turn_count,
        "tutor_monologue_ratio": tutor_monologue_ratio,
        "curriculum_velocity_count": curriculum_velocity_count,
        "epistemic_uncertainty_decay": epistemic_uncertainty_decay,
        "tutor_why_prompt_ratio": tutor_why_prompt_ratio,
        "student_math_uptake_rate": student_math_uptake_rate,
        "full_transcript_text": full_text,
        "student_transcript_text": student_text,
        "tutor_transcript_text": tutor_text,
        "final_3_student_text": final_3_student_text
    }

def process_all_transcripts() -> pd.DataFrame:
    folds_path = os.path.join(DATASETS_DIR, "train_folds.csv")
    if os.path.exists(folds_path):
        df_meta = pd.read_csv(folds_path)
    else:
        df_meta = pd.read_csv(os.path.join(DATASETS_DIR, "train_features_TMQTWsB.csv"))

    print(f"=== Reading Zip Bytes ({TRANSCRIPTS_ZIP}) ===")
    payloads = []
    with zipfile.ZipFile(TRANSCRIPTS_ZIP, 'r') as z:
        filenames = {f: f for f in z.namelist() if f.endswith('.csv')}
        unique_sessions = df_meta["session_id"].unique()
        for sid in unique_sessions:
            zip_filename = f"train_transcripts/{sid}.csv" if f"train_transcripts/{sid}.csv" in filenames else f"{sid}.csv"
            if zip_filename in filenames:
                payloads.append((sid, z.read(zip_filename)))

    print(f"Read {len(payloads)} file payloads. Processing in parallel via threads...")
    feature_rows = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(parse_transcript_bytes, p) for p in payloads]
        for idx, f in enumerate(as_completed(futures)):
            feature_rows.append(f.result())

    df_feats = pd.DataFrame(feature_rows)
    df_merged = pd.merge(df_meta, df_feats, on="session_id", how="left")

    if "is_correct" in df_merged.columns and "fold" in df_merged.columns:
        df_merged["target_enc_obj"] = 0.0
        global_mean = df_merged["is_correct"].mean()
        for fold in range(5):
            tr = df_merged[df_merged["fold"] != fold]
            val_idx = df_merged[df_merged["fold"] == fold].index
            means = tr.groupby("learning_objective")["is_correct"].agg(["mean", "count"])
            smoothed = (means["count"] * means["mean"] + 10 * global_mean) / (means["count"] + 10)
            df_merged.loc[val_idx, "target_enc_obj"] = df_merged.loc[val_idx, "learning_objective"].map(smoothed).fillna(global_mean)

    print("=== Computing Word-Level TF-IDF & TruncatedSVD (64 Components) ===")
    texts = df_merged["full_transcript_text"].fillna("").astype(str)
    word_tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    X_word_tfidf = word_tfidf.fit_transform(texts)
    word_svd = TruncatedSVD(n_components=64, random_state=42)
    word_svd_feats = word_svd.fit_transform(X_word_tfidf)

    for i in range(64):
        df_merged[f"svd_{i}"] = word_svd_feats[:, i]

    # Save speaker texts for multi-stream dense embedding extraction
    speaker_texts_df = df_merged[["session_id", "full_transcript_text", "student_transcript_text", "tutor_transcript_text", "final_3_student_text"]].copy()
    speaker_texts_path = os.path.join(DATASETS_DIR, "parsed_speaker_texts.csv")
    speaker_texts_df.to_csv(speaker_texts_path, index=False)
    print(f"=== Saved Parsed Speaker Texts to {speaker_texts_path} ===")

    df_merged.drop(columns=["full_transcript_text", "student_transcript_text", "tutor_transcript_text", "final_3_student_text"], inplace=True, errors="ignore")

    out_path = os.path.join(DATASETS_DIR, "train_extracted_features.csv")
    df_merged.to_csv(out_path, index=False)
    print(f"=== Extracted Features Saved to {out_path} (Shape: {df_merged.shape}) ===")
    return df_merged

if __name__ == "__main__":
    process_all_transcripts()
