# Math-Talk & Transformer Score Improvement Plan

## Goal
Incorporate math-talk density features (word count ratios, digit-character counts, <100 word fallback) and Hugging Face sentence-transformers into our feature pipeline and GBDT/Linear ensemble to lower holdout Log Loss (< 0.45000) and maximize competition leaderboard rank.

## Tasks
- [ ] Task 1: Update `src/features/extract_features.py` to calculate `numeric_turns_per_word`, `digit_chars_per_word`, `avg_student_words_per_turn`, and `quiet_session_flag` (< 100 student words). -> Verify: `python src/features/extract_features.py` executes without errors and outputs non-NaN columns.
- [ ] Task 2: Update `main.py` and `src/package_submission.py` to extract Sentence Transformer embeddings when available (`sentence-transformers/all-MiniLM-L6-v2`) with local fallback to TF-IDF SVD. -> Verify: Test embedding generation on 10 transcripts in `src/scratch_test.py`.
- [ ] Task 3: Update `src/models/train_export_ensemble.py` to retrain 5-Fold `GroupKFold` ensemble (LightGBM, XGBoost, CatBoost, Calibrated Ridge) with Platt scaling (`CalibratedClassifierCV`) and SLSQP probability weight optimization on the new feature set. -> Verify: Save `src/models/ensemble_models.pkl` with CV Log Loss < 0.530.
- [ ] Task 4: Rebuild `submission.zip` using `python src/package_submission.py`. -> Verify: Confirm `main.py` and `assets/ensemble_models.pkl` exist in `submission.zip` root with valid size (< 60GB).
- [ ] Task 5: Run `python src/evaluate_local_harness.py` to evaluate holdout Log Loss and ROC AUC against test container constraints. -> Verify: Ensure holdout Log Loss < 0.45000 and 100% format/bound compliance.

## Done When
- [ ] Local holdout Log Loss drops below `0.45000` (beating current leader `0.5964` and previous submission `0.9408`).
- [ ] `submission.zip` passes all DrivenData format checks without logging test set transcripts or aggregations.
