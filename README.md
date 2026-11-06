# Trace the Ace

This project predicts whether a student will answer a post-tutoring math assessment question correctly. It combines dialogue features and transformer representations with a calibrated tree ensemble, using log loss as the primary metric. See the [DrivenData reference solution](https://blog.drivendata.org/blog/productive-math-talk-reference) for the task and data context.

## Workspace map

| Path | Contents |
| --- | --- |
| [`src/features/`](src/features/) | Pedagogical dialogue features, semantic embeddings, and Bayesian encodings |
| [`src/models/`](src/models/) | Training, ensemble, and calibration code |
| [`src/`](src/) | Grouped validation, submission builders, and local evaluation |
| [`src/kaggle/`](src/kaggle/) | Standalone Kaggle training, download, and evaluation scripts |
| [`kaggle_kernel/`](kaggle_kernel/) | Kernel entry points with their existing deployment paths |
| [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) | Recorded CV, smoke test, and leaderboard results |
| [`docs/plans/`](docs/plans/) | Historical implementation and experiment plans |

Start with the [challenge brief](challenge.md), [task plan](task_plan.md), and [findings](findings.md). Run local Python commands from the repository root. The Kaggle kernel entry points use their own runtime paths, and scripts that call Kaggle require separately configured account access.

## Local inputs and outputs

Git tracks source and documentation. Keep competition data under `datasets/`, fitted DeBERTa assets under `src/models/`, and generated CSV and SVD artifacts at the paths the training scripts already use. Submission archives, scratch directories, logs, private Kaggle configuration, and legacy helpers with embedded credentials are local-only. See [`.gitignore`](.gitignore) for the exact rules.

The experiment table records historical runs. The EXP-16 checkpoint is a local artifact, and the checked-in `src/models/train_exp15_ensemble.py` describes EXP-15. The source snapshot alone cannot reproduce the recorded EXP-16 result. Some legacy submission builders create more than one archive. Follow the [single-archive rule](AGENTS.md) when choosing a builder.
