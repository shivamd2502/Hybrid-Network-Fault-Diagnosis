# Windows setup

Tested on Windows 11 with Python 3.12, PowerShell 5.1, scikit-learn 1.8.0 and Streamlit 1.x.

## 1. Prerequisites

- **Python 3.10–3.12** from [python.org](https://www.python.org/downloads/windows/). Tick
  *"Add python.exe to PATH"* during installation.
- **Git for Windows**: `winget install --id Git.Git -e`

## 2. Environment

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

Manual alternative:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`scikit-learn` is pinned to **1.8.0** because `models/pipeline.pkl` is a pickled sklearn
pipeline. If you change the version, retrain with `python src\train.py`; it takes about 10 seconds.

## 3. Running

```powershell
streamlit run src\app.py        # opens http://localhost:8501
python src\train.py             # retrain the model
python src\evaluate.py          # regenerate results\ and figures
python -m pytest tests -q       # run the test suite
```

All scripts resolve paths from the project root, so you can start them from any folder.

## 4. Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or run `.\.venv\Scripts\python.exe` directly |
| `streamlit` is not recognised | Activate the venv, or use `python -m streamlit run src\app.py` |
| `InconsistentVersionWarning` / errors loading `pipeline.pkl` | Your scikit-learn version differs from 1.8.0. Run `python src\train.py` |
| Port 8501 already in use | `streamlit run src\app.py --server.port 8502` |
| `UnicodeEncodeError` in an old script | Run `$env:PYTHONIOENCODING = "utf-8"` first. The current scripts print ASCII only |
