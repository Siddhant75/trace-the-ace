# Task Plan: DrivenData "Trace the Ace" AI Student Mastery Prediction

## Contributor attribution · 2026-11-12

- [x] Replace the automated attribution requirement with the project owner's contributor policy.
- Verification covers historical commit messages, historical instructions, explicit branch leases, and remote commit readback.

## Private repository publication · 2026-11-12

- [x] Confirm the repository name `trace-the-ace` and authenticated GitHub access.
- [x] Audit stored file blobs, generic commit identities, source syntax, and local-only exclusions before upload.
- [x] Commit publication preparation locally and publish `main` and the feature branch to a private repository with `main` as the default branch.
- Verification criterion is matching local and remote commit IDs for both published branches after uploading the publication record.

## Final competition context · 2026-11-12

- [x] Verify registration count, ranked-entry count, first-place metrics, and final leaderboard status from the official site.
- [x] Add the verified competition context to the README with source links and the verification date.
- [x] Confirm the project owner's final leaderboard entry before attributing a rank or comparing the final private scores.
- [x] Record rank 183 of 530, final private metrics, and the first-place log-loss comparison with distinct evaluation contexts.

## Portfolio documentation · 2026-11-12

- [x] Review source implementations and historical metrics for supported portfolio claims.
- [x] Describe the prediction task, pipeline, technical strengths, results, and reproduction requirements in the README.
- [x] Verify documentation links, benchmark arithmetic, and privacy before committing the documentation update.

## Repository privacy audit · 2026-11-12

- [x] Inspect every original commit for tracked data, fitted artifacts, credentials, device paths, and private account references.
- [x] Replace device-specific documentation links and move Kaggle kernel references to local environment configuration.
- [x] Expand ignore coverage for dataset formats, model exports, archives, notebooks, and local configuration.
- [x] Verify three Kaggle helper flows with mocked services and compile all 43 tracked Python sources.
- [x] Rewrite the three unpublished commits and the application snapshot tree, use generic author and committer identities, and verify the remaining objects after removing superseded history and reflogs.

## Workspace organization · 2026-11-06

- [x] Inventory source, historical plans, datasets, checkpoints, and generated outputs.
- [x] Define local-only ignore rules and keep runtime-sensitive paths in place.
- [x] Group historical plans in `docs/plans/` and standalone Kaggle scripts in `src/kaggle/`.
- Verify the Git index excludes data, checkpoints, logs, and credential-bearing helpers. Record source and documentation in separate commits on `feature/workspace-organization`.

The experiment roadmap below preserves the August submission planning snapshot.

## 🎯 Competition Objective & Strategy
Predict the binary probability (`0.0` to `1.0`) that a student will answer a post-tutoring math assessment question correctly based on multimodal dialogue transcripts, student responses, and metadata.
- **Target Metric**: **Log Loss** (Lower is better).
- **Secondary Metric**: **ROC AUC** (Higher is better).
- **Prize Target**: Top 3 Podium (< 0.5964) | Rank 1 (< 0.5898).
- **Current Active Best Leaderboard**: **EXP-10 (`0.6102` / `0.6092` AUROC - Smoke: `0.4815`)**.
- **Current Active Peak CV Weapon**: **EXP-16 (`0.53332` LogLoss / `0.73768` AUROC - All-Time Best)**.
- ⚠️ **Submission Strategy (Final August 27 Deadline Window)**:
  - Final leaderboard submission slots: 2 remaining submissions before deadline.
  - **Slot 1 (Top Primary Weapon)**: EXP-16 (19 Pedagogical Dynamics + Full-Data Bayes + 10% DeBERTa + 90% Bagged GBDTs, T=1.00, OOF LogLoss `0.53332`).
  - **Slot 2 (Proven Smoke Record Anchor)**: EXP-11 (Hierarchical Bayes + 3-Seed Bagged GBDT + DeBERTa 384t, T=1.05, Smoke LogLoss `0.4814`).
  - **Slot 3 (High-Diversity Backup)**: EXP-10 (`0.6102` LB Peak) or EXP-13 (125-fit Bagged Anchor, `0.4816` smoke).

---

## 🗺️ Experiment Roadmap & Milestone Tracker

| Milestone | Strategy / Architecture | 5-Fold CV LogLoss | 5-Fold CV AUROC | Smoke Test | LB LogLoss | LB AUROC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-1** | Baseline TF-IDF (10k) + LightGBM / CatBoost | 0.5480 | 0.6850 | - | 0.6450 | 0.5420 | Completed |
| **EXP-2** | Pedagogical Dialogue Ratios + Math Vocabulary | 0.5340 | 0.7010 | - | 0.6310 | 0.5610 | Completed |
| **EXP-3b**| MiniLM (384d) + SVD (64c) + Multi-Target Encodings | 0.5210 | 0.7180 | - | 0.6182 | 0.5811 | Completed |
| **EXP-4** | PyTorch MLP + GBDT Stacking | 0.52044 | 0.71765 | - | 0.6191 | 0.5802 | Completed |
| **EXP-5** | BGE-Large (1024d) + SVD (64c) + GroupKFold | 0.51274 | 0.72550 | - | - | - | Retrained |
| **EXP-6** | Dialogue Math Phase + 5-Model Diversity Stacking | 0.51195 | 0.72625 | 0.4976 | 0.6159 | 0.5915 | Superseded |
| **EXP-9** | DeBERTa-v3 + 5 GBDTs + Unconstrained Stacker | 0.48560* | 0.72170 | 0.4905 | 0.6181 | 0.5760 | Diagnosed |
| **EXP-10** | True Multimodal Convex Fusion (20% DeBERTa + 80% GBDTs) | 0.53963* | 0.72750 | 0.4815 | 0.6102 | 0.6092 | Evaluated (LB Peak) |
| **EXP-11 (Active Best)** | Hierarchical Bayes + 3-Seed Bagged GBDT Ensemble + DeBERTa (384t) + T=1.05 | 0.53411* | 0.73676 | 🥇 **0.4814** | **Target < 0.5950** | **Target > 0.6150** | 🏆 **PRIMARY CANDIDATE (Rank 1 Weapon)** |
| **EXP-12** | Dialogue Trajectory Dynamics + Bayes + 3-Seed Bagged 5-GBDT Ensemble (T=1.05) | 0.53354* | 0.73696 | 0.4824 | - | - | Evaluated |
| **EXP-13** | Robust Quantile Dynamics + 5-Seed Bagged GBDT Ensemble (125 Fits) + Calibrated Anchor Blend (T=1.05) | 0.53645* | 0.73476 | 0.4816 | - | - | Evaluated |
| **EXP-14** | Aligned DeBERTa Prompt + SLSQP Multimodal Blend (20% DeBERTa + 24% LGB + 22% Ridge + 18% HGB + 8% Cat + 8% XGB, T=1.05) | 0.53600* | 0.73532 | 0.4887 | - | - | Evaluated |
| **EXP-15** | Domain-Invariant Pedagogical Dynamics + Leak-Free GroupKFold + 35% DeBERTa + T=1.15 | 0.53991 | 0.73224 | 0.4984 | - | - | Evaluated (Over-softened) |
| **EXP-16 (Active Peak Weapon)** | **19 Pedagogical Features + Full-Data Bayes Encodings + 10% DeBERTa + 90% Bagged 5-GBDTs (T=1.00)** | **0.53332\*** | **0.73768** | **Sandbox Verified (Exit 0)** | **Target < 0.5950** | **Target > 0.6400** | 🏆 **Top-1 Primary Weapon (Ready)** |
