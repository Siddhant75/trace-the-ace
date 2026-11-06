"""
Data Exploration & Inspection Script for Trace the Ace Competition
"""
import os
import zipfile
import pandas as pd
import numpy as np

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
TRANSCRIPTS_ZIP = os.path.join(DATASETS_DIR, "train_transcripts.zip")
EXTRACT_DIR = os.path.join(DATASETS_DIR, "train_transcripts")

def inspect_dataset():
    print("=== 1. Checking Datasets Directory ===")
    files = os.listdir(DATASETS_DIR)
    print("Files found:", files)

    # 1. Inspect Features
    features_path = os.path.join(DATASETS_DIR, "train_features_TMQTWsB.csv")
    if os.path.exists(features_path):
        df_feat = pd.read_csv(features_path)
        print(f"\n=== Train Features Shape: {df_feat.shape} ===")
        print("Columns:", df_feat.columns.tolist())
        print("Head:\n", df_feat.head(3))
        print("Unique sessions:", df_feat['session_id'].nunique())
        print("Unique response_ids:", df_feat['response_id'].nunique())

    # 2. Inspect Labels
    labels_path = os.path.join(DATASETS_DIR, "train_labels_44ujmj2.csv")
    if os.path.exists(labels_path):
        df_labels = pd.read_csv(labels_path)
        print(f"\n=== Train Labels Shape: {df_labels.shape} ===")
        print("Labels Columns:", df_labels.columns.tolist())
        print("Labels Head:\n", df_labels.head(3))
        target_col = [c for c in df_labels.columns if c != 'response_id'][0]
        print(f"Target distribution ('{target_col}'):")
        print(df_labels[target_col].value_counts(normalize=True))

    # 3. Unzip sample transcript if needed
    if os.path.exists(TRANSCRIPTS_ZIP) and not os.path.exists(EXTRACT_DIR):
        print(f"\n=== Extracting {TRANSCRIPTS_ZIP} ... ===")
        with zipfile.ZipFile(TRANSCRIPTS_ZIP, 'r') as zip_ref:
            # Extract first 5 files for fast sample inspection
            namelist = zip_ref.namelist()
            print(f"Total transcript CSV files in zip: {len(namelist)}")
            sample_files = namelist[:5]
            for f in sample_files:
                zip_ref.extract(f, DATASETS_DIR)

    # Inspect a transcript file
    if os.path.exists(EXTRACT_DIR):
        sample_csvs = [f for f in os.listdir(EXTRACT_DIR) if f.endswith('.csv')]
        if sample_csvs:
            sample_path = os.path.join(EXTRACT_DIR, sample_csvs[0])
            df_transcript = pd.read_csv(sample_path)
            print(f"\n=== Sample Transcript ({sample_csvs[0]}) Shape: {df_transcript.shape} ===")
            print("Columns:", df_transcript.columns.tolist())
            print("Head:\n", df_transcript.head(5))

if __name__ == "__main__":
    inspect_dataset()
