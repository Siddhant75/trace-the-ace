# Trace the Ace - Implementation Plan

## Goal
Build a high-performance, calibrated multi-modal machine learning pipeline to predict student quiz performance from tutoring transcripts and metadata for the Trace the Ace competition.

## Tasks
- [x] Task 1: Create canonical `AGENTS.md` in project root → Verify: File exists and is < 60 lines.
- [x] Task 2: Create custom `tool-and-skill-advisor` meta-skill → Verify: `SKILL.md` exists in `.gemini/config/skills/tool-and-skill-advisor/`.
- [x] Task 3: Create data exploration & extraction script `src/eda.py` to inspect `datasets/` → Verify: Run `python src/eda.py` to print sample statistics, nulls, target distribution, and transcript structure.
- [x] Task 4: Implement 5-Fold GroupKFold baseline pipeline `src/validation.py` → Verify: CV split produces 5 folds grouped strictly by `session_id` with zero fold leakage.
- [x] Task 5: Build pedagogical feature extractor & 32-SVD TF-IDF vectorizer `src/features/pedagogical.py` → Verify: Outputs 56 features for talk ratios, Socratic questions, hesitation words, and TF-IDF SVD vectors.
- [x] Task 6: Train local Multi-Model Ensemble `src/models/ensemble.py` (LightGBM + CatBoost + XGBoost + Ridge SVD) → Verify: Achieved OOF Log Loss `0.54284` (ROC AUC `0.72237`).
- [x] Task 7: Implement GPU capability guardrails (`get_safe_device()`) in `src/kaggle/kaggle_deberta_pipeline.py` → Verify: Prevents crashes on Tesla P100 (`sm_60`).
- [ ] Task 8: Launch Final Kaggle GPU DeBERTa-v3 Run → Verify: Pull out-of-fold DeBERTa predictions back via `fetch_output()`.
- [ ] Task 9: Multi-Modal Stacking & Final Fusion → Verify: Blend GBDT ensemble + DeBERTa predictions to push Log Loss towards `< 0.50`.

## Done When
- [x] Leak-free 5-Fold GroupKFold CV ensemble achieves Log Loss lower than naive baseline (< 0.693). (Achieved OOF Log Loss: **0.54284**).
- [x] Submission file format strictly matches `submission_format_ZQLcKx7.csv`.
