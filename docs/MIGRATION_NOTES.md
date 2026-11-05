# Ubuntu → Windows migration notes

The project was developed on Ubuntu (`/home/shivam/CNWT_CP`, venv `.venv` with Python 3.12.3)
and moved to Windows 11. This page records what changed.

## Environment

| Before (Ubuntu) | After (Windows) |
|---|---|
| `.venv/` Linux venv, 2.6 GB, committed to git | new `.venv\` from `setup_env.ps1`, git-ignored |
| `files/requirements.txt` from the old v1 (sklearn 1.3, pandas 1.5) | `requirements.txt` matching the code (sklearn 1.8.0 pinned) |
| `streamlit run network_app_final.py` (file did not exist) | `streamlit run src\app.py` |

The old Linux venv was renamed to `.venv_linux_old` (git-ignored) and can be deleted.

## Layout

| Old (`files/`) | New |
|---|---|
| `network_troubleshooting_dataset.py` | `src/dataset.py` |
| `network_troubleshooting_training.py` | `src/train.py` |
| `rule_based_engine.py` | `src/rule_engine.py` |
| `rf_uncertainty.py`, `bayesian_fusion.py`, `hybrid_system.py` | `src/` (same names) |
| `app_streamlit.py` | `src/app.py` |
| `network_dataset_v3.csv` | `data/` |
| `models/pipeline.pkl`, `encoders.pkl` | `models/` |
| `network_inference.py`, `dt_model.pkl`, `metadata.json`, `decision_tree.*` | `archive/v1_decision_tree/` (older Decision-Tree version, unused) |
| `README.md`, `EXECUTION_GUIDE.md`, `INDEX.md`, `QUICK_START.txt` | `archive/v1_docs/` (they describe the v1 system with 86.7 % accuracy) |
| `results.txt` | `archive/misplaced_suspension_project_results.txt` (it was the suspension project's output) |
| `CNWT_Problems.txt` | turned into `tests/scenarios.json` (original kept in `archive/`) |

Unmodified copies of all original sources are in `archive/original_sources/`.

## Code changes

- **Paths:** `src/config.py` resolves everything from the project root with `pathlib`.
  Scripts no longer depend on the current working directory.
- **No duplication:** feature lists, class names, remediation steps and severities were
  copied into 3–4 files. They now live once in `config.py`.
- **Cross-validation fixed:** `train.py` ran 5-fold CV on the 240-sample **test** set. It now
  runs stratified 5-fold CV on the training set.
- **New `evaluate.py`:** measures RF, rules and hybrid on the test set, the ablations,
  text-template diagnostics, calibration and the hand-written scenarios. It writes
  `results/metrics.json`.
- **App shows measured numbers:** the sidebar previously hardcoded "99.6 %" and
  "98.75 % ± 1.67 %". It now reads `results/metrics.json` and shows both RF and hybrid accuracy.
- **Fusion naming:** `bayesian_fusion.py` still used `lstm_*` names from an earlier LSTM
  version. They are renamed to `rf_*` (strategy `lstm_override` becomes `rf_override`).
  The maths is unchanged.
- **Inference speed:** the saved RF uses `n_jobs=1`, which is faster for single-row
  predictions in the app. Predictions are identical.
- **Windows console:** non-ASCII symbols removed from `print` output. Files are read and
  written as UTF-8.
- **Dataset CSV:** written with `\n` line endings, so `python src/dataset.py` reproduces
  `data/network_dataset_v3.csv` byte for byte on Windows. A test checks this.
- `pandas.date_range(freq="H")` changed to `freq="h"` (deprecated alias).
- **Tests added:** `tests/` holds 23 pytest checks for the rules, fusion, dataset
  reproducibility and the hybrid pipeline.

## Verification

- The retrained pipeline makes **identical predictions** to the original Linux `pipeline.pkl`
  (test accuracy 99.58 %, the same single error).
- The regenerated dataset is byte-identical to the committed CSV.
- The Streamlit app starts and completes the 3-step flow with no exceptions (checked with
  `streamlit.testing.AppTest`).

## Git

- The local commit "Initial commit of all files" contained the whole Linux venv
  (42,455 files) and was never pushed. It is replaced by a clean commit on top of the GitHub history.
- `core.filemode = false` and `.gitattributes` normalise line endings.
