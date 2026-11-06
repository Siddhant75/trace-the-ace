# Agent Instructions

## Working Memory & Planning Artifacts
Always inspect and update the following project root and documentation markdown files at the start of any new session or task:
- [`task_plan.md`](task_plan.md): Active experiment plan, task roadmap, and verification criteria.
- [`progress.md`](progress.md): Chronological session history, task completion status, and error resolution log.
- [`findings.md`](findings.md): Competition discoveries, diagnostic findings, and technical benchmark logs.
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md): Official CV benchmarks, local holdout metrics, and submission history table.
- [`challenge.md`](challenge.md): Official DrivenData competition rules, target metrics, and container execution constraints.

## Package Manager
Use **pip** or **uv**:
- Install dependencies: `uv pip install -r requirements.txt` or `pip install -r requirements.txt`

## Commit Attribution
Do not list AI assistants as project contributors or commit co-authors. Use the project owner's GitHub-linked noreply identity and omit automated contributor attribution.

## Repository Privacy
Keep device paths, private account references, credentials, datasets, row-level outputs, fitted artifacts, and local configuration out of tracked files. Use relative documentation links and environment configuration for account-specific values. Check historical objects and commit identities when auditing repository privacy. Use the project owner's GitHub noreply identity for commit metadata while keeping personal email addresses private.

## File-Scoped Commands
| Task | Command |
|------|---------|
| Typecheck | `mypy path/to/file.py` |
| Lint | `ruff check path/to/file.py` |
| Format | `ruff format path/to/file.py` |
| Test | `pytest tests/test_file.py` |

## Key Conventions & Validation Rules
- **Cross-Validation**: Always use leak-free 5-Fold `GroupKFold` grouped by `session_id` or `student_id`.
- **Target Metric**: Optimize strictly for **Log Loss** ($-(y \log p + (1-y)\log(1-p))$).
- **Probability Calibration**: Apply Platt Scaling (`CalibratedClassifierCV(method='sigmoid')`) on base GBDT models and 2nd-stage Non-linear Stacking (`LogisticRegression`).
- **Probability Clipping**: Bounded strictly to `[0.01, 0.99]` to prevent infinity penalties under Log Loss.
- **Single Archive Rule**: NEVER create duplicate `submission.zip` copies alongside named submission archives. Maintain only one designated `.zip` archive to preserve disk storage.
- **Code Structure**: Store feature engineering in `src/features/`, models in `src/models/`, scripts in `src/`.

## DrivenData Platform & Cloud Runtime Environment
- **Hardware & Limit**: 1x NVIDIA A100 GPU (80GB VRAM), 24 vCPUs, 6-hour total execution limit, 60GB max zip limit.
- **Python Runtime**: Python 3.12, PyTorch, CUDA 12.9, `scikit-learn 1.8.0` (stamp `_sklearn_version = "1.8.0"` on pickled estimators).
- **Pre-loaded HF Models**: Mounted at `/code_execution/huggingface_models/BAAI/bge-large-en-v1.5` (335M params, 24 layers, 1024 dims). Load with `local_files_only=True` (`BLOCK_INTERNET=true`).
- **Anti-Leakage Logging Rule**: **NEVER** print or log test set transcripts, transcript snippets, or summary statistics to `stdout`/`stderr`.

## Model & Feature Pipeline Architecture (EXP-16 Active Peak)
- **Active Feature Matrix (218 Features)**:
  - 19 Domain-Invariant Pedagogical Dynamics: scaffolding-to-direct assertion ratio, repair arcs, student cognitive agency & causal reasoning density, epistemic uncertainty decay, dialogue arcs (`src/features/pedagogical.py`)
  - DrivenData Math-Talk ratios (`numeric_turns_per_word`, `digit_chars_per_word`, `<100` word low-talk fallback)
  - Math Domain Vocabulary Density (`math_term_count`, `math_term_density`)
  - Hierarchical Empirical Bayes Difficulty Encodings & Multi-Factor Target Encodings (`learning_objective`, `grade_level`, `topic_id`)
  - 64-component Word TF-IDF TruncatedSVD
  - 64-component Dense Semantic Transformer Embeddings (SVD projection via `trans_svd.pkl`)
- **Multi-Modal Ensemble Architecture**:
  - Stage 1: 3 Seeds (42, 100, 2026) × 5 Folds × 5 Model Families (75 Calibrated Fits: LightGBM, CatBoost, XGBoost, HistGB, Ridge) + 5-Fold Fine-Tuned DeBERTa-v3-Large (384-token pedagogical prompt).
  - Stage 2: SLSQP Convex Fusion (10.05% DeBERTa + 89.95% Bagged GBDTs) with Temperature Scaling ($T=1.00$) and `[0.035, 0.965]` probability bounding.
