"""
EXP-9: DeBERTa-v3-Large + Tabular Multimodal Classifier (Ultra-Robust v13)
Changes:
1. Fixed KeyError on student_id by selecting dynamic existing identifier columns ([c for c in ["session_id", "student_id", "is_correct"] if c in df.columns]).
2. Scaled BATCH_SIZE = 16 (8 per T4 GPU) on Dual T4 for 2x faster throughput (~30 mins/epoch, ~5h total).
3. Gradient checkpointing, 2 epochs per fold, 5x MSD, LLRD, AMP FP16.
4. Incremental fold saving to deberta_oof.csv and deberta_fold_*.pt.
"""
import os
import sys
import gc
import math
import time
import pickle
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup

sys.stderr = sys.stdout

# --- Hyperparameters & Config ---
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 384
BATCH_SIZE = 16            # 8 per GPU on Dual T4
GRAD_ACCUM_STEPS = 1       # Effective batch size = 16
LR_HEAD = 5e-4
LR_BACKBONE = 1.5e-5
LLRD_DECAY = 0.90
WEIGHT_DECAY = 0.01
EPOCHS = 2                 # Optimal 2-epoch fine-tuning sweet spot
LABEL_SMOOTH_EPS = 0.015
NUM_DROPOUTS = 5
DROPOUT_RATE = 0.20
SEED = 42

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)

def find_file_kaggle(filename):
    for root, dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            full_p = os.path.join(root, filename)
            print(f"Found input file: {full_p}", flush=True)
            return full_p
    # Local fallback
    local_p = os.path.join("datasets", filename)
    if os.path.exists(local_p):
        return local_p
    raise FileNotFoundError(f"Could not find {filename}")

# --- Text Preprocessing with Dialogue Structuring ---
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

# --- Multimodal DeBERTa-v3 Neural Architecture ---
class DeBERTaMultimodalModel(nn.Module):
    def __init__(self, model_name, num_tab_features=105, num_dropouts=5, dropout_rate=0.20):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update({
            "output_hidden_states": True,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "use_cache": False
        })
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.backbone.gradient_checkpointing_enable()

        hidden_dim = self.config.hidden_size # 1024 for deberta-v3-large

        # Tabular MLP Projection Head (num_tab_features -> 128 -> 64)
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
        self.num_dropouts = num_dropouts
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_dropouts)])
        self.classifier = nn.Linear(fused_dim, 1)

    def extract_text_features(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state # [batch, seq_len, 1024]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()

        # Mean Pooling
        sum_hidden = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_hidden / sum_mask

        # Max Pooling
        last_hidden_masked = last_hidden.clone()
        last_hidden_masked[mask_expanded == 0] = -1e9
        max_pooled = torch.max(last_hidden_masked, dim=1)[0]

        return torch.cat([mean_pooled, max_pooled], dim=1) # [batch, 2048]

    def forward(self, input_ids, attention_mask, tab_features):
        text_emb = self.extract_text_features(input_ids, attention_mask) # [batch, 2048]
        tab_emb = self.tab_mlp(tab_features) # [batch, 64]
        fused = torch.cat([text_emb, tab_emb], dim=1) # [batch, 2112]

        if self.training:
            logits = [self.classifier(drop(fused)) for drop in self.dropouts]
            return torch.mean(torch.stack(logits, dim=0), dim=0).squeeze(-1) # [batch]
        else:
            return self.classifier(fused).squeeze(-1) # [batch]

# --- Layer-wise Learning Rate Decay Optimizer Setup ---
def get_optimizer_grouped_parameters(raw_model, base_lr=1.5e-5, head_lr=5e-4, weight_decay=0.01, decay_rate=0.90):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = []

    # Head parameters
    head_params = list(raw_model.classifier.named_parameters()) + list(raw_model.tab_mlp.named_parameters())
    optimizer_grouped_parameters.append({
        "params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": head_lr
    })
    optimizer_grouped_parameters.append({
        "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": head_lr
    })

    # Backbone parameters with LLRD
    num_layers = raw_model.config.num_hidden_layers
    for layer_i in range(num_layers):
        layer = raw_model.backbone.encoder.layer[layer_i]
        layer_lr = base_lr * (decay_rate ** (num_layers - 1 - layer_i))
        optimizer_grouped_parameters.append({
            "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": layer_lr
        })
        optimizer_grouped_parameters.append({
            "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": layer_lr
        })

    embed_lr = base_lr * (decay_rate ** num_layers)
    optimizer_grouped_parameters.append({
        "params": [p for n, p in raw_model.backbone.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": embed_lr
    })
    optimizer_grouped_parameters.append({
        "params": [p for n, p in raw_model.backbone.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": embed_lr
    })

    return optimizer_grouped_parameters

# --- Main Training Function ---
def main():
    print("=== EXP-9: DeBERTa-v3-Large Multimodal Classifier (Pre-Tokenized v13) ===", flush=True)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPUs available: {num_gpus} ({[torch.cuda.get_device_name(i) for i in range(num_gpus)]})", flush=True)

    # 1. Load Data
    feats_path = find_file_kaggle("train_extracted_features.csv")
    spk_path = find_file_kaggle("parsed_speaker_texts.csv")

    df_feats = pd.read_csv(feats_path)
    df_spk = pd.read_csv(spk_path).drop_duplicates(subset=["session_id"])
    df = df_feats.merge(df_spk[["session_id", "student_transcript_text", "tutor_transcript_text", "final_3_student_text"]], on="session_id", how="left")

    ignore_cols = ["session_id", "student_id", "learning_objective", "is_correct", "fold", "full_transcript_text", "student_transcript_text", "tutor_transcript_text", "final_3_student_text", "grade_level", "topic_id"]
    tab_cols = [c for c in df_feats.columns if c not in ignore_cols and not c.startswith("oof_") and not c.startswith("norm_")]

    X_tab = df[tab_cols].select_dtypes(include=[np.number]).copy()
    X_tab = (X_tab - X_tab.mean()) / (X_tab.std() + 1e-7)
    X_tab_vals = X_tab.fillna(0.0).values
    num_tab_features = X_tab_vals.shape[1]
    print(f"Tabular features dimension: {num_tab_features}", flush=True)

    texts = [format_multimodal_text(row) for _, row in df.iterrows()]
    y = df["is_correct"].values.astype(np.float32)

    # 2. Upfront Batch Tokenization
    print(f"Pre-tokenizing all {len(texts)} samples with fast Rust tokenizer...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    )
    all_input_ids = encoded["input_ids"]
    all_attention_mask = encoded["attention_mask"]
    all_tab_features = torch.tensor(X_tab_vals, dtype=torch.float32)
    all_labels = torch.tensor(y, dtype=torch.float32)
    print(f"Pre-tokenization complete in {time.time()-t0:.2f}s! Tensors allocated in RAM.", flush=True)

    # 3. Student / Session GroupKFold Split
    gkf = GroupKFold(n_splits=5)
    student_groups = df["student_id"].values if "student_id" in df.columns else df["session_id"].values
    folds = np.zeros(len(df), dtype=int)
    for fold_idx, (tr_i, val_i) in enumerate(gkf.split(texts, y, groups=student_groups)):
        folds[val_i] = fold_idx

    oof_predictions = np.zeros(len(df))
    out_id_cols = [c for c in ["session_id", "student_id", "is_correct"] if c in df.columns]

    # 4. 5-Fold Training Loop
    for fold in range(5):
        print(f"\n==================== TRAINING FOLD {fold}/5 ====================", flush=True)
        tr_idx = torch.tensor(np.where(folds != fold)[0], dtype=torch.long)
        val_idx = torch.tensor(np.where(folds == fold)[0], dtype=torch.long)

        val_y = y[val_idx.numpy()]

        tr_ds = TensorDataset(
            all_input_ids[tr_idx],
            all_attention_mask[tr_idx],
            all_tab_features[tr_idx],
            all_labels[tr_idx]
        )
        val_ds = TensorDataset(
            all_input_ids[val_idx],
            all_attention_mask[val_idx],
            all_tab_features[val_idx]
        )

        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)

        raw_model = DeBERTaMultimodalModel(MODEL_NAME, num_tab_features=num_tab_features, num_dropouts=NUM_DROPOUTS, dropout_rate=DROPOUT_RATE).to(device).float()
        opt_params = get_optimizer_grouped_parameters(raw_model, base_lr=LR_BACKBONE, head_lr=LR_HEAD, weight_decay=WEIGHT_DECAY, decay_rate=LLRD_DECAY)
        optimizer = torch.optim.AdamW(opt_params)

        if num_gpus > 1:
            print(f"=== Wrapping model with nn.DataParallel across {num_gpus} GPUs ===", flush=True)
            model = nn.DataParallel(raw_model)
        else:
            model = raw_model

        total_steps = (len(tr_loader) // GRAD_ACCUM_STEPS) * EPOCHS
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

        best_val_loss = float("inf")
        best_val_preds = None

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            for step, (input_ids, attention_mask, tab_feats, targets) in enumerate(tqdm(tr_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Fold {fold}]")):
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                tab_feats = tab_feats.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                targets_smooth = targets * (1.0 - 2.0 * LABEL_SMOOTH_EPS) + LABEL_SMOOTH_EPS

                if scaler is not None:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        logits = model(input_ids, attention_mask, tab_feats)
                        loss = F.binary_cross_entropy_with_logits(logits, targets_smooth) / GRAD_ACCUM_STEPS
                    scaler.scale(loss).backward()
                else:
                    logits = model(input_ids, attention_mask, tab_feats)
                    loss = F.binary_cross_entropy_with_logits(logits, targets_smooth) / GRAD_ACCUM_STEPS
                    loss.backward()

                total_loss += loss.item() * GRAD_ACCUM_STEPS

                if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(tr_loader):
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            # Validation
            model.eval()
            val_preds = []
            with torch.no_grad():
                for (input_ids, attention_mask, tab_feats) in val_loader:
                    input_ids = input_ids.to(device, non_blocking=True)
                    attention_mask = attention_mask.to(device, non_blocking=True)
                    tab_feats = tab_feats.to(device, non_blocking=True)

                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            logits = model(input_ids, attention_mask, tab_feats)
                            probs = torch.sigmoid(logits).cpu().numpy()
                    else:
                        logits = model(input_ids, attention_mask, tab_feats)
                        probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(probs)

            val_preds = np.clip(np.array(val_preds), 0.025, 0.975)
            val_loss = log_loss(val_y, val_preds)
            val_auc = roc_auc_score(val_y, val_preds)
            print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(tr_loader):.5f} | Val LogLoss: {val_loss:.5f} | Val ROC-AUC: {val_auc:.5f}", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_preds = val_preds
                underlying_model = model.module if hasattr(model, "module") else model
                fp16_state = {k: v.half().cpu() for k, v in underlying_model.state_dict().items()}
                torch.save(fp16_state, f"deberta_fold_{fold}.pt")
                print(f" >>> Saved Best Fold {fold} Model (LogLoss: {best_val_loss:.5f}) <<<", flush=True)

        oof_predictions[val_idx.numpy()] = best_val_preds
        # Safe incremental save of OOF predictions after every fold
        df_temp = df[out_id_cols].copy()
        df_temp["deberta_oof"] = oof_predictions
        df_temp.to_csv("deberta_oof.csv", index=False)
        print(f"=== Incremental fold {fold} OOF saved to deberta_oof.csv ===", flush=True)

        del model, raw_model, optimizer, scheduler, tr_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    overall_loss = log_loss(y, oof_predictions)
    overall_auc = roc_auc_score(y, oof_predictions)
    print(f"\n=======================================================")
    print(f"=== EXP-9 DEBERTA MULTIMODAL 5-FOLD OOF LogLoss: {overall_loss:.5f} | ROC AUC: {overall_auc:.5f} ===")
    print(f"=======================================================", flush=True)

    df_temp = df[out_id_cols].copy()
    df_temp["deberta_oof"] = oof_predictions
    df_temp.to_csv("deberta_oof.csv", index=False)
    print("=== Exported final deberta_oof.csv successfully! ===", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"=== CRITICAL EXCEPTION: {e} ===", flush=True)
        traceback.print_exc()
        with open("error_log.txt", "w") as f:
            f.write(str(e) + "\n" + traceback.format_exc())
        sys.exit(1)
