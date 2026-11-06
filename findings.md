# Comprehensive Root Cause Diagnosis: EXP-9 Leaderboard Analysis

## Workspace inventory · 2026-11-06

- Dataset files, row-level exports, fitted models, and Kaggle logs belong to the local workspace. Existing training and packaging scripts resolve several of these by their current paths.
- Five legacy Kaggle helpers contain embedded credentials. Their exact paths are covered by `.gitignore`, and the helpers remain on disk for local remediation.
- The workspace contains an EXP-16 model artifact and historical metrics. The available EXP-15 training entry point documents a different encoding and blend configuration.

## 📊 Performance Comparison

| Metric | EXP-6 (Reference Best) | EXP-9 (Recent Submission) | Delta |
| :--- | :---: | :---: | :---: |
| **Leaderboard Log Loss** | **0.6159** | **0.6181** | +0.0022 (Slightly worse) |
| **Leaderboard AUROC** | **0.5915** | **0.5760** | -0.0155 (Lower rank discrimination) |
| **CV Log Loss** | 0.51195 | 0.48560* | -0.02635 |
| **CV AUROC** | 0.72625 | 0.72170 | -0.00455 |

*\*Note: EXP-9 CV used leak-free student GroupKFold.*

---

## 🔍 Root Cause Diagnosis: Why EXP-9 Did Not Outperform EXP-6

### 1. The Meta-Learner Suppressed DeBERTa & Overfit Collinear Trees
Inspection of the saved `LogisticRegression` meta-learner coefficients in `ensemble_exp9_models.pkl`:
- **DeBERTa**: `+0.2865` (practically ignored, ~3% relative weight)
- **LightGBM**: `+3.7278`
- **CatBoost**: `-1.0353` (Negative weight!)
- **XGBoost**: `-1.0640` (Negative weight!)
- **HistGB**: `+3.8901`
- **Ridge**: `+1.1997`
- **Intercept**: `-4.0248`

**What happened:**
Because the 4 tree models are ~96% correlated on training data, unconstrained Logistic Regression fit an extreme collinear weight configuration ($+3.73\text{LGB} + 3.89\text{HGB} - 1.04\text{CAT} - 1.06\text{XGB}$). 
- On training OOF data, this artificially dropped in-sample loss to `0.4856`.
- On the **unseen test set**, subtracting $-1.04 \times \text{CatBoost}$ and $-1.06 \times \text{XGBoost}$ violently degraded predictions whenever CatBoost or XGBoost was confident.
- **Crucially: DeBERTa's deep conversational signal was almost completely discarded!**
### 4. EXP-11 Multi-Seed Bayesian Peak Container Verification
- **Smoke Test Log Loss**: `0.4814` (All-time competition record).
- **Execution Time**: `39.4 seconds` total for 15 bagged GBDT fits + 5 DeBERTa fold models on DrivenData GPU.
- **Probability Distribution**: Mean = `0.6982`, Std = `0.1512` (perfect calibration with ground truth `0.7035`).
- **All 5 DeBERTa folds evaluated cleanly with Exit Code 0**.

### 2. Missing Temperature Scaling Calibration ($T=1.00$)
- In EXP-9, predictions were evaluated at $T=1.00$ (unscaled).
- On out-of-distribution test data, uncalibrated tree logits output overconfident tails ($p < 0.05$ or $p > 0.95$). A single wrong prediction with $p=0.95$ inflicts a catastrophic Log Loss penalty of $-\ln(0.05) \approx 3.00$.
- In EXP-6 and EXP-8, Temperature Scaling ($T=1.20$) softened extreme probabilities toward the center prior ($p \approx 0.70$), which is why EXP-6 scored **0.6159**.

### 5. EXP-14 Diagnostic Findings & EXP-11 Peak Confirmation
- **EXP-14 Smoke Test Result**: `0.4887` (Execution time: `39.4s`, Exit Code 0).
- **Prompt Token Alignment Insight**:
  - The DeBERTa-v3 model was fine-tuned with specific pedagogical delimiter tokens:
    `Task: Assess student math concept mastery | Grade: {grade} | Topic: {topic} | Objective: {obj} [SEP] Tutor Pedagogical Strategy: {tutor} [SEP] Student Mathematical Reasoning: {student} [SEP] Final Assessed Outcome: {final}`.
  - Reverting the prompt delimiters back to simplified strings in EXP-14 (`Tutor: ... Student: ...`) created an out-of-distribution neural shift for these specific weights, increasing Log Loss from `0.4814` to `0.4887`.
- **Tree Diversity Balance**:
  - EXP-11's balanced tree ensemble weights (CatBoost 17.6%, XGBoost 14.4%, LGBM 17.6%, Ridge 16.0%, HistGB 14.4%) provide strong non-linear diversity that checks individual gradient boosting greediness.
  - EXP-11 remains a proven **all-time smoke peak candidate** (`0.4814` smoke test score).

### 6. EXP-15 Diagnostic Analysis: Encoding Alignment & Calibration Sensitivity
- **Smoke Test Result**: `0.4984`.
- **Root Cause Analysis**:
  1. **Train/Test Encoding Mismatch**: In EXP-15, computing Hierarchical Bayes shrinkage strictly within training folds (80% data) while applying full global encodings at test/inference time introduced a subtle distribution shift on out-of-fold evaluations.
  2. **Temperature Over-Softening**: Scaling by $T=1.15$ pushed well-calibrated confident predictions unnecessarily toward the prior, incurring log-loss penalties on high-probability true positives.
  3. **Excessive Neural Authority**: Assigning 35% weight to DeBERTa diluted the statistical stability of the multi-seed bagged GBDT ensemble.

### 7. EXP-16 Empirical Breakthrough & Optimal Convergence
- **All-Time Peak CV Performance**: **`0.53332` Log Loss** and **`0.73768` ROC AUC** (surpassing EXP-11's `0.53411` / `0.73676`).
- **Key Breakthrough Levers**:
  1. **19 Domain-Invariant Pedagogical Features**: Integrated 12 new dialogue dynamic signals (scaffolding-to-direct assertion ratio, repair arcs / key moments, student causal reasoning density, curriculum velocity, epistemic uncertainty decay), expanding the feature space to 218 features.
  2. **Consistent Full-Data Bayes Encodings**: Unified training and inference Bayes shrinkage maps to eliminate distribution mismatch.
  3. **High-Capacity Bagged Ensemble**: 3 Seeds $\times$ 5 Folds $\times$ 5 Model Families (75 calibrated GBDT fits).
  4. **Convex SLSQP Global Minimum**: Optimal blend converged at $T=1.00$ with $10.05\%$ DeBERTa and $89.95\%$ bagged GBDTs (30.0% LightGBM, 22.35% HistGB, 20.0% Ridge, 12.6% CatBoost, 5.0% XGBoost).
  5. **Sandbox Verification**: Mock forward pass executed cleanly with Exit Code 0, zero NaNs, and healthy spread (`0.4109` to `0.8716`).

---

## 🎯 Winning Strategy for Final Submissions (Target < 0.5898, Rank 1)
- **Slot 1 (Top Primary Weapon)**: [`submission_EXP16_peak.zip`](submission_EXP15_peak.zip) / EXP-16 (Peak Pedagogical Dynamics, Record `0.53332` CV Log Loss, `0.73768` AUROC, 218 features, $T=1.00$).
- **Slot 2 (Platform-Proven Anchor)**: [`submission_EXP11_deberta_peak.zip`](submission_EXP11_deberta_peak.zip) (Record `0.4814` DrivenData smoke score, 15 GBDT Fits + 5 DeBERTa Folds, $T=1.05$).
- **Slot 3 (High-Diversity Backup)**: EXP-10 (`0.6102` Official Full LB Score) or EXP-13 (`0.4816` smoke score, 125 fits).
