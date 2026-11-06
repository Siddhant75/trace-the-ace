"""
DeBERTa-v3 / ModernBERT Kaggle Free GPU Fine-Tuning Pipeline
Run this script inside a Kaggle GPU Notebook (T4/P100) or Google Colab.
"""
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_cosine_schedule_with_warmup
from sklearn.metrics import log_loss, roc_auc_score

# Hyperparameters & Settings
MODEL_NAME = "microsoft/deberta-v3-base" # Or "answerdotai/ModernBERT-base"
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
def get_safe_device():
    if torch.cuda.is_available():
        try:
            # Perform a test tensor allocation to verify GPU kernel compatibility
            test_tensor = torch.zeros(1, device="cuda")
            cap = torch.cuda.get_device_capability()
            print(f"CUDA Available: GPU {torch.cuda.get_device_name(0)} (CUDA Capability: {cap[0]}.{cap[1]})")
            if cap[0] < 7:
                print(f"Warning: GPU capability {cap[0]}.{cap[1]} < 7.0 (sm_70). Falling back to CPU.")
                return torch.device("cpu")
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA verification failed ({e}). Falling back to CPU.")
            return torch.device("cpu")
    print("CUDA not available. Using CPU.")
    return torch.device("cpu")

DEVICE = get_safe_device()

class DialogueDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        data = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
        if self.labels is not None:
            data["label"] = torch.tensor(self.labels[item], dtype=torch.float)
        return data

def train_fn(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def valid_fn(model, dataloader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)
    return np.array(preds)

def run_kaggle_training(folds_df_path=None):
    print(f"Using device: {DEVICE}")
    
    if folds_df_path is None:
        import glob
        matches = glob.glob("/kaggle/input/**/train_folds.csv", recursive=True)
        if matches:
            folds_df_path = matches[0]
        elif os.path.exists("datasets/train_folds.csv"):
            folds_df_path = "datasets/train_folds.csv"
        else:
            folds_df_path = "train_folds.csv"

    print(f"Loading folds from: {folds_df_path}")
    df = pd.read_csv(folds_df_path)
    
    # Text formatting: combine learning objective + session summary
    df["full_text"] = "Objective: " + df["learning_objective"].fillna("") + " | Session: " + df["session_id"]
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    oof_preds = np.zeros(len(df))

    for fold in range(5):
        print(f"\n======== Training Fold {fold} ========")
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        val_indices = df[df["fold"] == fold].index

        train_ds = DialogueDataset(train_df["full_text"].values, train_df["is_correct"].values, tokenizer, MAX_LEN)
        val_ds = DialogueDataset(val_df["full_text"].values, val_df["is_correct"].values, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=1)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(EPOCHS):
            train_loss = train_fn(model, train_loader, optimizer, scheduler, criterion)
            val_preds = valid_fn(model, val_loader)
            val_loss = log_loss(val_df["is_correct"].values, val_preds)
            print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} | Val LogLoss: {val_loss:.5f}")

        oof_preds[val_indices] = val_preds

    df["deberta_pred"] = oof_preds
    df.to_csv("deberta_oof_predictions.csv", index=False)
    print("Saved OOF predictions to deberta_oof_predictions.csv")

if __name__ == "__main__":
    print("Starting Kaggle DeBERTa-v3 GPU Training Pipeline...")
    run_kaggle_training()
