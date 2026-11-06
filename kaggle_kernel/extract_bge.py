"""
Kaggle Extraction Script for BAAI/bge-large-en-v1.5 Embeddings (Ultra-Optimized v14).
Supports float32 CUDA execution on P100/T4 + torch.inference_mode() + 24-thread CPU parallelization.
"""
import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import TruncatedSVD

sys.stderr = sys.stdout

# Set maximum CPU thread parallelism
num_cpus = os.cpu_count() or 8
torch.set_num_threads(num_cpus)
torch.set_num_interop_threads(min(8, num_cpus))
os.environ["OMP_NUM_THREADS"] = str(num_cpus)
os.environ["MKL_NUM_THREADS"] = str(num_cpus)

def find_file_kaggle(filename):
    for root, dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            full_p = os.path.join(root, filename)
            print(f"Found input file: {full_p}", flush=True)
            return full_p
    raise FileNotFoundError(f"Could not find {filename} in /kaggle/input")

def get_device_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"=== CUDA GPU Detected: {gpu_name} ===", flush=True)
        # Try float32 CUDA first, then float16
        for precision, use_half in [("Float32 CUDA", False), ("Float16 CUDA", True)]:
            try:
                print(f"Testing {precision} GPU Model Initialization...", flush=True)
                gpu_model = AutoModel.from_pretrained(model_name)
                if use_half:
                    gpu_model = gpu_model.half()
                gpu_model = gpu_model.to("cuda")
                gpu_model.eval()

                dummy_in = tokenizer(["Task: Assess student mastery"], return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    _ = gpu_model(**dummy_in)
                print(f"=== Active {precision} Acceleration Verified! Using CUDA GPU ===", flush=True)
                return torch.device("cuda"), gpu_model, tokenizer
            except Exception as e:
                print(f"=== {precision} failed ({e}). Trying fallback... ===", flush=True)
                torch.cuda.empty_cache()

    print(f"=== Instantiating Ultra-Fast CPU Model ({num_cpus} threads) ===", flush=True)
    cpu_model = AutoModel.from_pretrained(model_name).to("cpu")
    cpu_model.eval()
    return torch.device("cpu"), cpu_model, tokenizer

def extract_stream_embeddings(model, tokenizer, texts, device, max_len=160, batch_size=128):
    total = len(texts)
    embeddings = []
    with torch.inference_mode():
        for i in range(0, total, batch_size):
            b_texts = texts[i:i + batch_size]
            encoded = tokenizer(b_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
            pooled = torch.sum(out[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.float().cpu().numpy())

            if (i // batch_size) % 50 == 0:
                print(f" Processed {min(i+batch_size, total)} / {total} samples...", flush=True)
    return np.vstack(embeddings)

def main():
    print("=== Kaggle BAAI/bge-large-en-v1.5 Fast Embedding Extractor (v14) ===", flush=True)

    feats_path = find_file_kaggle("train_extracted_features.csv")
    spk_path = find_file_kaggle("parsed_speaker_texts.csv")

    df_feats = pd.read_csv(feats_path)
    df_spk = pd.read_csv(spk_path).drop_duplicates(subset=["session_id"])
    df = df_feats.merge(df_spk[["session_id", "student_transcript_text", "tutor_transcript_text", "final_3_student_text"]], on="session_id", how="left")

    df["full_text"] = "Task: Assess student math concept mastery | Objective: " + df["learning_objective"].fillna("Mathematics") + " | Student Responses: " + df["student_transcript_text"].fillna("") + " | Tutor Guidance: " + df["tutor_transcript_text"].fillna("") + " | Final Outcome: " + df["final_3_student_text"].fillna("")
    df["student_text"] = "Task: Assess student response accuracy | Objective: " + df["learning_objective"].fillna("Mathematics") + " | Student Responses: " + df["student_transcript_text"].fillna("")

    full_texts = df["full_text"].values.tolist()
    student_texts = df["student_text"].values.tolist()

    model_name = "BAAI/bge-large-en-v1.5"
    device, model, tokenizer = get_device_and_model(model_name)

    batch_sz = 128 if device.type == "cuda" else 64
    max_l_full = 256 if device.type == "cuda" else 160
    max_l_stud = 192 if device.type == "cuda" else 128

    print(f"--- Extracting Stream 1: Full Dialogue Arc (Max Tokens: {max_l_full}, Batch: {batch_sz}) ---", flush=True)
    embs_full = extract_stream_embeddings(model, tokenizer, full_texts, device, max_len=max_l_full, batch_size=batch_sz)

    print(f"--- Extracting Stream 2: Student Responses Only (Max Tokens: {max_l_stud}, Batch: {batch_sz}) ---", flush=True)
    embs_student = extract_stream_embeddings(model, tokenizer, student_texts, device, max_len=max_l_stud, batch_size=batch_sz)

    print("Fitting TruncatedSVD on Stream 1 (64 Components)...", flush=True)
    svd_full = TruncatedSVD(n_components=64, random_state=42)
    comp_full = svd_full.fit_transform(embs_full)

    print("Fitting TruncatedSVD on Stream 2 (32 Components)...", flush=True)
    svd_student = TruncatedSVD(n_components=32, random_state=42)
    comp_student = svd_student.fit_transform(embs_student)

    for i in range(64):
        df[f"trans_emb_{i}"] = comp_full[:, i]
    for i in range(32):
        df[f"student_emb_{i}"] = comp_student[:, i]

    with open("trans_svd.pkl", "wb") as f:
        pickle.dump({"svd_full": svd_full, "svd_student": svd_student}, f)

    df.drop(columns=["full_text", "student_text", "student_transcript_text", "tutor_transcript_text", "final_3_student_text"], inplace=True, errors="ignore")
    out_csv = "transformer_embeddings_oof.csv"
    df.to_csv(out_csv, index=False)
    print(f"=== Saved Upgraded Dual-Stream Feature Matrix to {out_csv} ({os.path.getsize(out_csv)/(1024*1024):.2f} MB) ===", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"=== ERROR ENCOUNTERED: {e} ===", flush=True)
        import traceback
        traceback.print_exc()
        with open("error_log.txt", "w") as f:
            f.write(str(e) + "\n" + traceback.format_exc())
        sys.exit(1)
