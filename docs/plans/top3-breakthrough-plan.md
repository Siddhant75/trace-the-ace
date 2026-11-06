# Top-3 Leaderboard Breakthrough Plan (Target Log Loss < 0.5964)

## Goal
Bridge the 0.0218 Log Loss gap to reach Top 3 by implementing **Speaker Partitioning (Tutor vs. Student)**, **3-Phase Dialogue Arc Features**, **Multi-Stream Heavy Transformer Embeddings (`bge-large` / `MiniLM`)**, **Stage-1 PyTorch MLP**, **Smoothed M-Estimate Target Encodings**, and **5-Model Stacking & SLSQP Ensembling**.

## Tasks
- [ ] Task 1: **Speaker & 3-Phase Dialogue Features** (`src/features/pedagogical.py`) → Extract `student_transcript`, `tutor_transcript`, `final_3_student_turns`, `student_math_term_density`, `tutor_encouragement_count`, and student hesitation lexicons (`"um"`, `"maybe"`, `"i think"`). → Verify: `python src/features/pedagogical.py` outputs 105+ features.
- [ ] Task 2: **Multi-Stream Transformer Embeddings with Fallback** (`src/features/transformer_embeddings.py`) → Extract dual embeddings for Student vs. Tutor turns using `bge-large-en-v1.5` (or `all-MiniLM-L6-v2`), fit 64-dim TruncatedSVD, and save `trans_svd.pkl` + `transformer_embeddings_oof.csv`. → Verify: Output shape contains 64 embedding components.
- [ ] Task 3: **PyTorch MLP Stage-1 Model Training** (`src/models/train_mlp_stage1.py`) → Train a 2-layer PyTorch MLP with Dropout and Cosine Annealing on tabular + embedding features under 5-Fold GroupKFold; save `mlp_oof.csv` and `mlp_models.pth`. → Verify: OOF predictions written with Log Loss logged.
- [ ] Task 4: **5-Model Stage-2 Stacking & M-Estimate Target Encoding** (`src/models/train_export_ensemble.py`) → Add smoothed M-estimate out-of-fold target encoding (`m=10.0`) for `learning_objective`/`subject`; merge PyTorch MLP OOF probabilities into Stage-2 Stacking Meta-Learner (LightGBM + CatBoost + XGBoost + Ridge + MLP); stamp `_sklearn_version = "1.8.0"`. → Verify: CV Log Loss drops below `0.50500`.
- [ ] Task 5: **Updated Code Package Generator** (`src/package_submission.py`) → Update `main.py` template to execute multi-stream speaker embeddings + PyTorch MLP + Stacking Meta-Learner on DrivenData runtime; copy `mlp_models.pth` to `assets/` and build lightweight `submission.zip`. → Verify: `submission.zip` generated under 20 MB.
- [ ] Task 6: **Local Test Harness Sandbox Verification** (`src/evaluate_local_harness.py`) → Execute full DrivenData local test harness on 7,014 holdout samples with live multi-stream embeddings and PyTorch MLP. → Verify: Local holdout Log Loss drops below `< 0.47000` with exit code 0.

## Done When
- [ ] 5-Fold Cross-Validation Log Loss drops below **`0.50500`**.
- [ ] Local holdout Log Loss drops below **`0.47000`**.
- [ ] `submission.zip` passes DrivenData local test harness with 100% format compliance, zero warnings, and exit code 0.
