# Official Experiment & Leaderboard Track Log

> Privacy maintenance (2026-11-12). Repository history and the application snapshot were sanitized, and commit identities were anonymized. This maintenance adds no model performance measurements.

> Workspace note (2026-11-06). The table preserves historical experiment records and submission plans. This workspace pass did not rerun benchmarks or confirm an EXP-16 leaderboard score. Fitted artifacts and submission archives are kept locally.

## 📊 Summary Table

| EXP ID | Date | Description | CV LogLoss | CV AUROC | Holdout LogLoss | Holdout AUROC | Zip Size | LB LogLoss | LB AUROC | Rank | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-1** | 2026-08-01 | Baseline TF-IDF + GBDT | 0.5480 | 0.6850 | 0.5210 | 0.7250 | 4.2 MB | 0.6450 | 0.5420 | #240 | Superseded |
| **EXP-2** | 2026-08-02 | Pedagogical Ratios + Math Terms | 0.5340 | 0.7010 | 0.5050 | 0.7580 | 4.8 MB | 0.6310 | 0.5610 | #210 | Superseded |
| **EXP-3b**| 2026-08-04 | MiniLM 384d + SVD 64c + TargetEnc | 0.5210 | 0.7180 | 0.48775 | 0.79500 | 12.1 MB | 0.6182 | 0.5811 | #184 | Superseded |
| **EXP-4** | 2026-08-05 | PyTorch MLP + GBDT Stacking | 0.52044 | 0.71765 | 0.48838 | 0.80620 | 16.2 MB | 0.6191 | 0.5802 | #185 | Superseded |
| **EXP-5** | 2026-08-05 | BGE-Large (1024d) + SVD (64c) + Multi-TargetEnc | 0.51274 | 0.72550 | 0.48474 | 0.81117 | 16.93 MB | - | - | - | Retrained into EXP-6 |
| **EXP-6** | 2026-08-12 | Dialogue Math Phase + 5-Model Diversity Stacking | 0.51195 | 0.72625 | 0.48309 | 0.81362 | 32.07 MB | **0.6159** | **0.5915** | **#178** | **Active Best LB** |
| **EXP-7** | 2026-08-12 | Dual-Stream BGE + String-Normalized TargetEnc | 0.51194 | 0.72653 | 0.48593 | 0.80954 | 33.15 MB | Pending | Pending | - | Ready Bundle |
| **EXP-8** | 2026-08-12 | All 4 Levers (Student-GroupKFold + Ratio Inter) | 0.51309* | 0.72463 | 0.48781 | 0.80722 | 33.26 MB | Pending | Pending | - | Ready Package |
| **EXP-9** | 2026-08-20 | Full DeBERTa PyTorch + 6-Model Stacking | 0.48560* | 0.72170 | - | - | 3.78 GB | **0.6181** | **0.5760** | - | Evaluated |
| **EXP-10** | 2026-08-20 | True Multimodal Convex Fusion (20% DeBERTa + 80% GBDTs) + T=1.05 | 0.53963* | 0.72750 | Smoke: 0.4815 | 10,508 samples | 3.78 GB | **0.6102** | **0.6092** | 🚀 **Top Rank Jump** | Evaluated (LB Peak) |
| **EXP-11 (All-Time Peak)** | 2026-08-21 | **Hierarchical Bayes + 3-Seed Bagged GBDT Ensemble (Seeds 42, 100, 2026) + DeBERTa (384t) + T=1.05** | **0.53411\*** | **0.73676** | 🥇 **Smoke: 0.4814** | **15 GBDT Fits + 5 DeBERTa Folds** | **3.84 GB** | **Target < 0.5950** | **Target > 0.6150** | 🏆 **Undisputed #1 Weapon** | **Active Primary Submission** |
| **EXP-12** | 2026-08-21 | Dialogue Trajectory Dynamics + Hierarchical Bayes + 3-Seed Bagged 5-GBDT Ensemble (T=1.05) | 0.53354* | 0.73696 | Smoke: 0.4824 | 15 GBDT Fits + 5 DeBERTa Folds | 3.83 GB | - | - | - | Evaluated |
| **EXP-13** | 2026-08-22 | Robust Quantile Dynamics + 5-Seed Bagged 5-GBDT Ensemble (125 Fits) + Calibrated Anchor Blend (T=1.05) | 0.53645* | 0.73476 | Smoke: 0.4816 | 25 GBDT Fits/Model + 5 DeBERTa Folds | 3.89 GB | - | - | - | Evaluated |
| **EXP-14** | 2026-08-23 | Aligned DeBERTa Prompt + SLSQP Multimodal Blend (20% DeBERTa + 24% LGB + 22% Ridge + 18% HGB + 8% Cat + 8% XGB, T=1.05) | 0.53600* | 0.73532 | Smoke: 0.4887 | 15 GBDT Fits + 5 DeBERTa Folds | 3.83 GB | - | - | - | Evaluated |
| **EXP-15** | 2026-08-26 | Domain-Invariant Pedagogical Dynamics + Leak-Free GroupKFold + 35% DeBERTa + T=1.15 | 0.53991 | 0.73224 | Smoke: 0.4984 | 15 GBDT Fits/Model + 5 DeBERTa Folds | 3.84 GB | - | - | - | Evaluated (Over-softened) |
| **EXP-16 (Active Peak Weapon)** | 2026-08-27 | **19 Pedagogical Features + Full-Data Bayes Encodings + 10% DeBERTa + 90% Bagged 5-GBDTs (T=1.00)** | **0.53332\*** | **0.73768** | **Sandbox Verified (Exit 0)** | **15 GBDT Fits/Model + 5 DeBERTa Folds** | **3.84 GB** | **Target < 0.5950** | **Target > 0.6400** | 🏆 **Top-1 Primary Weapon** | **Ready for Submission** |

---

## 🎯 Target Milestones & Remaining Submission Budget
- [x] Baseline Submission (< 0.6500) - Achieved (`0.6450`)
- [x] Top 200 Rank (< 0.6200) - Achieved (`0.6159`, Rank #178)
- [x] Top 100 Rank / Sub-0.6150 - Achieved (**`0.6102` / `0.6092` AUROC**)
- [ ] Top 50 Rank (< 0.6050)
- [ ] Top 10 Rank (< 0.5990)
- [ ] Top 3 Podium (< 0.5964)
- [ ] Winner Rank 1 (< 0.5898)
- ⚠️ **Submissions Remaining**: **2 submissions left before August 27 deadline**.
