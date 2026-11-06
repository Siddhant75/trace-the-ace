"""
Local Lightweight Transformer Embedding Extractor (all-MiniLM-L6-v2)
Extracts 32-component SVD dense semantic embeddings from student-tutor transcripts using PyTorch locally.
"""
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import TruncatedSVD

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets") if __file__.count(os.sep) > 3 else os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 128
MAX_LEN = 256
torch.set_num_threads(8)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)

def find_file(target_filename):
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if target_filename in files:
                return os.path.join(root, target_filename)
    if os.path.exists(os.path.join("datasets", target_filename)):
        return os.path.join("datasets", target_filename)
    if os.path.exists(target_filename):
        return target_filename
    return target_filename

def load_all_session_texts(session_ids):
    session_texts = {}
    dir_path = None
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "train_transcripts" in dirs:
                dir_path = os.path.join(root, "train_transcripts")
                break
    if not dir_path and os.path.exists("datasets/train_transcripts"):
        dir_path = "datasets/train_transcripts"

    if dir_path and os.path.isdir(dir_path):
        print(f"=== Reading transcripts from unzipped directory: {dir_path} ===")
        for sid in session_ids:
            csv_file = os.path.join(dir_path, f"{sid}.csv")
            if os.path.exists(csv_file):
                try:
                    df_t = pd.read_csv(csv_file)
                    session_texts[sid] = " ".join(df_t["content"].dropna().astype(str))
                except Exception:
                    session_texts[sid] = ""
        return session_texts

    zip_path = find_file("train_transcripts.zip")
    print(f"=== Reading transcripts from ZIP file: {zip_path} ===")
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            filenames = {f: f for f in z.namelist() if f.endswith('.csv')}
            for sid in session_ids:
                zip_fn = f"train_transcripts/{sid}.csv" if f"train_transcripts/{sid}.csv" in filenames else f"{sid}.csv"
                if zip_fn in filenames:
                    with z.open(zip_fn) as f:
                        try:
                            df_t = pd.read_csv(f)
                            session_texts[sid] = " ".join(df_t["content"].dropna().astype(str))
                        except Exception:
                            session_texts[sid] = ""
    return session_texts

def get_safe_device():
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device="cuda")
            _ = test_tensor + 1
            major_ver = torch.cuda.get_device_capability()[0]
            if major_ver >= 7:
                return torch.device("cuda")
        except Exception as e:
            print(f"CUDA verification failed ({e}). Falling back to CPU.")
    return torch.device("cpu")

def extract_transformer_embeddings():
    features_path = find_file("train_extracted_features.csv")
    print(f"=== Loading Features from: {features_path} ===")
    df = pd.read_csv(features_path)

    speaker_texts_path = find_file("parsed_speaker_texts.csv")
    if os.path.exists(speaker_texts_path):
        print(f"=== Loading Speaker Partitioned Texts from: {speaker_texts_path} ===")
        df_spk = pd.read_csv(speaker_texts_path).drop_duplicates(subset=["session_id"])
        df = df.merge(df_spk, on="session_id", how="left")
    else:
        session_texts = load_all_session_texts(df["session_id"].unique())
        df["full_transcript_text"] = df["session_id"].map(session_texts).fillna("")
        df["student_transcript_text"] = df["full_transcript_text"]
        df["tutor_transcript_text"] = df["full_transcript_text"]
        df["final_3_student_text"] = df["full_transcript_text"]

    df["full_text"] = "Task: Assess student math concept mastery | Objective: " + df["learning_objective"].fillna("Mathematics") + " | Student Responses: " + df["student_transcript_text"].fillna("") + " | Tutor Guidance: " + df["tutor_transcript_text"].fillna("") + " | Final Outcome: " + df["final_3_student_text"].fillna("")
    texts = df["full_text"].values.tolist()

    print(f"=== Loading Transformer Model: {MODEL_NAME} ===")
    device = get_safe_device()
    print(f"Using PyTorch device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    embeddings = []
    total = len(texts)

    print(f"=== Computing Multi-Stream Embeddings in Batches of {BATCH_SIZE} ===")
    with torch.no_grad():
        for i in range(0, total, BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(device)
            
            out = model(**encoded)
            pooled = mean_pooling(out, encoded["attention_mask"])
            embeddings.append(pooled.cpu().numpy())

            if (i + BATCH_SIZE) % 5000 < BATCH_SIZE:
                print(f"Processed {min(i + BATCH_SIZE, total)} / {total} embeddings...")

    all_embs = np.vstack(embeddings)
    print(f"Full Multi-Stream Embeddings Matrix Shape: {all_embs.shape}")

    # Compress 384 dimensions to 64 dense TruncatedSVD components
    print("=== Compressing Multi-Stream Embeddings to 64 SVD Components ===")
    svd = TruncatedSVD(n_components=64, random_state=42)
    compressed_embs = svd.fit_transform(all_embs)

    emb_cols = {}
    for i in range(64):
        emb_cols[f"trans_emb_{i}"] = compressed_embs[:, i]
    emb_df = pd.DataFrame(emb_cols, index=df.index)
    df = pd.concat([df, emb_df], axis=1)

    import pickle
    with open("trans_svd.pkl", "wb") as f:
        pickle.dump(svd, f)
    print("=== Saved fitted Multi-Stream Transformer SVD model to trans_svd.pkl ===")

    df.drop(columns=["full_text", "full_transcript_text", "student_transcript_text", "tutor_transcript_text", "final_3_student_text"], inplace=True, errors="ignore")
    out_emb_path = "transformer_embeddings_oof.csv"
    df.to_csv(out_emb_path, index=False)
    print(f"=== Saved Upgraded Multi-Stream Feature Matrix to {out_emb_path} (Shape: {df.shape}) ===")

if __name__ == "__main__":
    extract_transformer_embeddings()
