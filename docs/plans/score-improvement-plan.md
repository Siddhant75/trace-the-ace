# Score Improvement & Local Harness Validation Plan

## Goal
Improve model Log Loss score below `0.50` and validate locally using the replicated platform test harness before uploading to the leaderboard.

## Tasks
- [ ] Task 1: Fine-tune `SentenceTransformer` / `MiniLM` text embeddings on session transcripts → Verify: Extract 64-dim embeddings and check OOF Log Loss in `src/features/pedagogical.py`.
- [ ] Task 2: Implement Isotonic Calibration & Temperature Scaling per fold → Verify: Compare calibrated OOF Log Loss against baseline `0.53985`.
- [ ] Task 3: Retrain 5-Fold GroupKFold Ensemble (LGBM, CatBoost, XGBoost, Calibrated Ridge) with new features → Verify: Output `src/models/ensemble_models.pkl` with new transformers.
- [ ] Task 4: Rebuild submission package (`submission.zip`) with updated `main.py` entrypoint → Verify: Ensure `submission.zip` size < 50MB and valid structure.
- [ ] Task 5: Run full local container harness test (`python src/evaluate_local_harness.py`) → Verify: Check 0 NaNs, correct format, and local holdout Log Loss < 0.50.

## Done When
- [ ] Local holdout Log Loss score drops below `0.50`.
- [ ] `submission.zip` passes all 4 steps of `evaluate_local_harness.py` cleanly.
