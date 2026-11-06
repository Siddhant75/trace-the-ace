# Hugging Face Transformer Embeddings, Math Lexicon & Stacking Meta-Learner Plan

## Goal
Integrate pre-loaded `sentence-transformers/all-MiniLM-L6-v2` dense semantic embeddings and math domain vocabulary density into our feature extraction pipeline, train a 2-stage Stacking Meta-Learner (Non-linear Logistic Regression / Isotonic Stacking on OOF predictions) over 5-Fold `GroupKFold` base models (LightGBM, CatBoost, XGBoost, Ridge), and evaluate on our local test harness before submitting.

## Tasks
- [ ] Task 1: Add math domain vocabulary lexicon (`math_term_count`, `math_term_density`) and student reasoning turn ratios to `src/features/pedagogical.py`. -> Verify: `python -u src/features/pedagogical.py` outputs non-NaN columns in `datasets/train_extracted_features.csv`.
- [ ] Task 2: Run `src/features/transformer_embeddings.py` to extract 32 SVD components from `all-MiniLM-L6-v2` embeddings for all 35,072 training rows -> Save to `transformer_embeddings_oof.csv`. -> Verify: Output shape is `(35072, 32)`.
- [ ] Task 3: Update `src/models/train_export_ensemble.py` to train 5-Fold `GroupKFold` base models (LGBM, CatBoost, XGBoost, Ridge) AND fit a 2nd-stage Non-linear Stacking Meta-Learner (`LogisticRegression` / Isotonic Calibration) on OOF predictions -> Save complete pipeline pack to `src/models/ensemble_models.pkl`. -> Verify: Stacking CV Log Loss < 0.530.
- [ ] Task 4: Update `submission_package/main.py` and `src/package_submission.py` to load `/code_execution/huggingface_models/sentence-transformers/all-MiniLM-L6-v2` at runtime (falling back to SVD), extract math domain features, and run 2-stage stacking inference. -> Verify: Rebuild `submission.zip` (< 20MB).
- [ ] Task 5: Run `python -u src/evaluate_local_harness.py` to evaluate holdout Log Loss on 7,014 test samples. -> Verify: Holdout Log Loss < 0.46000 and 100% format compliance.

## Done When
- [ ] Local holdout Log Loss drops below `0.46000` (beating current `0.48182`).
- [ ] 2-stage Stacking Meta-Learner yields superior log-loss probability calibration than simple linear blending.
- [ ] `submission.zip` passes all DrivenData format and anti-leakage checks.
