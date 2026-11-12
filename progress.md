# Session Progress Log & Task Checklist

## 2026-11-12 · Private repository publication

- The project owner selected `trace-the-ace` for the private repository. Existing project changes were already committed locally.
- The pre-upload audit checked 74 stored file blobs, generic commit identities, and local-only exclusions. All 43 Python sources compiled.
- Fast-forwarded `main` to the complete project history and uploaded both named branches. GitHub confirmed private visibility and `main` as the default branch. Account-specific remote configuration stays in local Git metadata.

## 2026-11-12 · Final competition context

- Verified the official competition's 1,150 registered participants and counted 530 ranked entries across all 11 leaderboard pages.
- The finalized private leaderboard's first-place result is 0.59233 log loss and 0.65088 ROC AUC. Added the sourced context to the README.
- The project owner confirmed the final entry at rank 183 of 530, log loss 0.60791, and ROC AUC 0.60463. Added the final result and a first-place comparison to the README while keeping historical submission metrics separate.
- Verified 530 unique ranks, score consistency, exact gap arithmetic, local links, and privacy patterns before committing the documentation.

## 2026-11-12 · Portfolio documentation

- Rewrote the README around supervised student-outcome prediction, NLP feature engineering, transformer training, calibrated ensembles, and offline inference packaging.
- Used the recorded EXP-1 and EXP-10 leaderboard scores for the headline comparison. Local validation scores retain their encoding and model-selection context.
- Described the project as applied machine learning and NLP, with supporting source links and reproduction requirements.
- Verified all 23 local README links, the relative log-loss calculation, Markdown fences, documentation change scope, and privacy patterns.

## 2026-11-12 · Repository privacy audit

- The initial Git setup excluded datasets, model artifacts, logs, and the five credential-bearing helpers. It missed device-specific links in three Markdown files and private Kaggle account references in three Python files.
- Replaced the links with repository-relative targets and made Kaggle helper targets configurable through `TRACE_ACE_KAGGLE_KERNEL`.
- The user approved rewriting the three unpublished commits and anonymizing author and committer identities. Ignore coverage now includes further data formats, model exports, notebooks, archives, and local configuration.
- The three Kaggle helper flows passed checks using mocked services. All 43 tracked Python sources compiled. The history cleanup also covers the application snapshot tree that retained earlier file contents.
- Rewrote the original three commits, added this audit record, and verified all retained file blobs and commit identities. Superseded objects and reflog entries were removed. Existing ignored datasets, fitted assets, credentials, and local helpers were preserved.

## 2026-11-06 · Workspace organization

- Inventoried a workspace without Git history and added ignore rules for private and generated files.
- Moved five historical plans and seven standalone Kaggle scripts, checking the file hashes after each move. The historical implementation plan now points to the relocated script.
- Kept datasets, fitted models, and Kaggle kernel entry points at the paths used by existing workflows.
- This maintenance task did not run training, packaging, Kaggle jobs, or competition submissions.

## Active Session Goal
Final Submission Preparation & Platform Verification (Deadline: August 27, 2026 UTC):
- Finalize and audit peak submission candidate **EXP-16** (19 Pedagogical Features + Consistent Bayes Encodings + 10% DeBERTa + 90% Bagged 5-GBDTs, $T=1.00$, OOF LogLoss `0.53332`, AUROC `0.73768`).
- Maintain and verify **EXP-11** as platform-proven anchor (Smoke LogLoss `0.4814`).
- Enforce DrivenData constraints: single archive rule, zero NaN checks, `< 60GB` uncompressed size, strict anti-leakage logging policy, and `_sklearn_version = "1.8.0"` compatibility.

---

## 📅 Session History & Progress

- [x] **EXP-9 LB Benchmark Logged**: Log Loss `0.6181`, AUROC `0.5760`.
- [x] **Root Cause Discovered**: Stacking meta-learner assigned `+0.286` to DeBERTa while assigning `-1.035` to CatBoost and `-1.064` to XGBoost. Collinear negative weights fit training residuals but degraded test generalization. Temperature scaling was missing ($T=1.00$).
- [x] **Constrained Convex Fusion**: Built `src/models/train_exp10_convex_ensemble.py` optimizing SLSQP non-negative weights: `deberta: 45.0%`, `lgb: 13.9%`, `ridge: 14.9%`, `hgb: 13.6%`, `cat: 9.6%`, `xgb: 3.0%`.
- [x] **Calibration Added**: Integrated Temperature Scaling ($T=1.20$) to align probability tails with the true class prior (`Mean: 0.7136`).
- [x] **Exact Dimension Matching**: Fixed DeBERTa's tabular MLP input to exactly 99 features.
- [x] **Package Bundling**: Created `submission_EXP10_deberta_convex.zip` and `submission.zip`.
- [x] **Sandbox Verification**: Verified real PyTorch forward pass, zero NaNs, and format compliance (Exit Code 0).
- [x] **DrivenData Smoke Test Succeeded**: Achieved record **0.4815 Log Loss** with all 5 DeBERTa folds evaluating in the cloud container (Mean Prob: `0.6982`, Std: `0.1515`).
- [x] **Official Full Leaderboard Score Logged**: **Log Loss 0.6102** and **AUROC 0.6092** across 10,508 hidden test samples! All-time competition best score.
- [x] **EXP-12 Grandmaster Multimodal Ensemble Completed**: Tested on smoke test (`0.4824`). Reclaimed 7.83 GB disk space by cleaning intermediate archives.
- [x] **EXP-13 5-Seed Bagging + Quantile Dynamics + Calibrated Anchor Blend Completed**: Tested on cloud smoke test (`0.4816`).
- [x] **EXP-14 Peak Refinement Evaluated**: Tested on cloud smoke test (`0.4887`). Analysis confirmed that EXP-11's exact prompt delimiters and rich CatBoost (17.6%) + XGBoost (14.4%) tree diversity are optimal.
- [x] **EXP-15 Pedagogical Deep-Signal Architecture & Leak-Free Multimodal Ensemble Completed**:
  - **Empirical Pedagogical Discovery**: Extracted 12 domain-invariant dialogue dynamics (scaffold-to-direct-assertion ratio, repair arcs / key moments, student causal reasoning density, curriculum velocity, epistemic uncertainty decay).
  - **Leak-Free Cross-Validation**: Computed Hierarchical Bayes and target encodings strictly inside each training fold of 5-Fold `GroupKFold`.
  - **Model Architecture**: Multi-Seed (Seeds 42, 100, 2026) 5-Fold GBDT ensemble (LGB, Cat, XGB, HGB, Ridge) + 5-Fold DeBERTa-v3-Large (35% authority) + Temperature Scaling ($T=1.15$).
- [x] **EXP-15 Smoke Test Evaluated (`0.4984`) & Root Cause Diagnosed**:
  - Identified that training on fold-internal (80%) Bayes encodings while receiving full-data encodings at inference caused a subtle distribution shift, compounded by $T=1.15$ over-softening and $35\%$ DeBERTa authority.
- [x] **EXP-16 Pedagogical Best-of-Both-Worlds Retrain Completed**:
  - **Encodings**: Restored full-data consistent Hierarchical Bayes and target encodings (matching inference).
  - **Feature Space**: Retained all 218 features including all 19 domain-invariant pedagogical features.
  - **Ensemble Training**: 3 Seeds $\times$ 5 Folds $\times$ 5 Model Families (75 calibrated GBDT fits).
  - **Optimal Calibration**: SLSQP grid search identified optimal calibration at $T=1.00$ with $10.05\%$ DeBERTa, $30.00\%$ LightGBM, $22.35\%$ HistGB, $20.00\%$ Ridge, $12.60\%$ CatBoost, and $5.00\%$ XGBoost.
  - **Record OOF Performance**: **`0.53332` LogLoss** and **`0.73768` ROC AUC** (surpassing EXP-11's `0.53411` / `0.73676`).
  - **Packaged & Sandbox Verified**: Updated [`submission_EXP15_peak.zip`](submission_EXP15_peak.zip) (3.837 GB). Mock forward pass verified with Exit Code 0, zero NaNs, and healthy spread (`0.4109` to `0.8716`).

---

## ⚠️ Errors & Resolutions Log

| Error / Event | Root Cause | Resolution |
| :--- | :--- | :--- |
| EXP-9 LB Drop (0.6181) | Negative weights in logistic regression stacker canceled tree signals and suppressed DeBERTa | Switched to constrained non-negative convex fusion ($w_i \ge 0, \sum w_i = 1.0$) with 45% DeBERTa authority. |
| `size mismatch for tab_mlp.0.weight: [128, 99] vs [128, 195]` in sandbox | `tab_cols_deberta` in pickle included 195 features instead of 99 | Fixed `tab_cols_deberta` to exactly 99 numeric features from `train_extracted_features.csv`. |
| EXP-15 Initial Smoke (0.4984) | Train/test encoding mismatch (fold-internal vs global), $T=1.15$ over-softening, and $35\%$ DeBERTa weight | Retrained EXP-16 with consistent full-data encodings, 218 features, $T=1.00$, and optimized SLSQP blend (`0.53332` LogLoss / `0.73768` AUC). |
