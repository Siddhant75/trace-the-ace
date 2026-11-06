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
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
MAX_LEN = 256

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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

    session_texts = load_all_session_texts(df["session_id"].unique())
    df["full_text"] = "Objective: " + df["learning_objective"].fillna("") + " | Dialogue: " + df["session_id"].map(session_texts).fillna("")
    texts = df["full_text"].values.tolist()

    print(f"=== Loading Transformer Model: {MODEL_NAME} ===")
    device = get_safe_device()
    print(f"Using PyTorch device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    embeddings = []
    total = len(texts)

    print(f"=== Computing Embeddings in Batches of {BATCH_SIZE} ===")
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
    print(f"Full Embeddings Matrix Shape: {all_embs.shape}")

    # Compress 384 dimensions to 32 dense TruncatedSVD components
    print("=== Compressing Embeddings to 32 SVD Components ===")
    svd = TruncatedSVD(n_components=32, random_state=42)
    compressed_embs = svd.fit_transform(all_embs)

    for i in range(32):
        df[f"trans_emb_{i}"] = compressed_embs[:, i]

    df.drop(columns=["full_text"], inplace=True, errors="ignore")
    out_emb_path = "transformer_embeddings_oof.csv"
    df.to_csv(out_emb_path, index=False)
    print(f"=== Saved Upgraded Feature Matrix to {out_emb_path} (Shape: {df.shape}) ===")

if __name__ == "__main__":
    extract_transformer_embeddings()
