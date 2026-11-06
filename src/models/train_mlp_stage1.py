"""
PyTorch Multi-Layer Perceptron (MLP) Stage-1 Model
Trains a 2-layer Neural Network with BatchNorm, Dropout, and Cosine Annealing under 5-Fold GroupKFold.
Exports out-of-fold probability vectors to mlp_oof.csv and saves model weights to mlp_models.pth.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) if __file__.count(os.sep) > 3 else os.path.dirname(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class TabularMLP(nn.Module):
    def __init__(self, input_dim):
        super(TabularMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_pytorch_mlp():
    print("=== Loading Extracted Features & Multi-Stream Embeddings for PyTorch MLP ===")
    feats_path = os.path.join(DATASETS_DIR, "train_extracted_features.csv")
    df_feats = pd.read_csv(feats_path)

    trans_oof_path = os.path.join(BASE_DIR, "transformer_embeddings_oof.csv")
    if os.path.exists(trans_oof_path):
        df_trans = pd.read_csv(trans_oof_path)
        trans_cols = [c for c in df_trans.columns if c.startswith("trans_emb_")]
        if "response_id" in df_trans.columns and "response_id" in df_feats.columns:
            df = df_feats.merge(df_trans[["response_id"] + trans_cols], on="response_id", how="left")
        else:
            df = df_feats.copy()
            for c in trans_cols:
                df[c] = df_trans[c]
    else:
        df = df_feats.copy()

    # M-estimate Target Encoding for learning_objective
    gkf = GroupKFold(n_splits=5)
    df["target_enc_obj"] = np.nan
    global_mean = df["is_correct"].mean()
    for train_idx, val_idx in gkf.split(df, df["is_correct"], groups=df["session_id"]):
        tr = df.iloc[train_idx]
        means = tr.groupby("learning_objective")["is_correct"].agg(["mean", "count"])
        smoothed = (means["count"] * means["mean"] + 10 * global_mean) / (means["count"] + 10)
        df.iloc[val_idx, df.columns.get_loc("target_enc_obj")] = df.iloc[val_idx]["learning_objective"].map(smoothed)
    df["target_enc_obj"] = df["target_enc_obj"].fillna(global_mean)

    exclude_cols = ["response_id", "session_id", "learning_objective", "learning_objective_id", "is_correct", "fold"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
    print(f"Total PyTorch MLP Features: {len(feature_cols)}")

    X = df[feature_cols].fillna(0.0).values
    y = df["is_correct"].values
    groups = df["session_id"].values

    oof_mlp = np.zeros(len(df))
    models_dict = {}
    scalers_dict = {}

    device = torch.device("cpu")
    print(f"Using PyTorch device: {device}")

    epochs = 20
    batch_size = 256

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        scalers_dict[fold] = scaler

        train_ds = TensorDataset(torch.tensor(X_tr_scaled, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_va_scaled, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = TabularMLP(input_dim=len(feature_cols)).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = 999.0
        best_state = None

        for ep in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(device)
                    out = torch.sigmoid(model(bx))
                    val_preds.extend(out.cpu().numpy())
                    val_targets.extend(by.numpy())
            val_loss = log_loss(val_targets, val_preds)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()

        model.load_state_dict(best_state)
        models_dict[fold] = best_state

        model.eval()
        val_preds = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                out = torch.sigmoid(model(bx))
                val_preds.extend(out.cpu().numpy())
        oof_mlp[val_idx] = val_preds
        print(f"Fold {fold} PyTorch MLP LogLoss: {log_loss(y[val_idx], oof_mlp[val_idx]):.5f}")

    overall_loss = log_loss(y, np.clip(oof_mlp, 0.01, 0.99))
    overall_auc = roc_auc_score(y, oof_mlp)
    print(f"\n=======================================================")
    print(f"=== PyTorch MLP 5-Fold OOF LogLoss: {overall_loss:.5f} | ROC AUC: {overall_auc:.5f} ===")
    print(f"=======================================================")

    # Save weights and scalers
    save_path = os.path.join(MODELS_DIR, "mlp_models.pth")
    torch.save({
        "models": models_dict,
        "scalers": scalers_dict,
        "feature_cols": feature_cols
    }, save_path)
    print(f"=== Saved PyTorch MLP weights to {save_path} ===")

    # Save OOF predictions
    oof_df = pd.DataFrame({"response_id": df["response_id"], "mlp_prob": oof_mlp})
    oof_out_path = os.path.join(BASE_DIR, "mlp_oof.csv")
    oof_df.to_csv(oof_out_path, index=False)
    print(f"=== Saved PyTorch MLP OOF Predictions to {oof_out_path} ===")

if __name__ == "__main__":
    train_pytorch_mlp()
