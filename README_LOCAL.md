# Run locally (Windows)
1. Install Python 3.11 or 3.12 (python.org; tick "Add to PATH").
2. In PowerShell, inside this folder:
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install -r requirements.txt
3. Check everything works:   python -m pytest -q tests      (expect 63 passed)
4. Rebuild outputs:          python build_labels.py ; python build_features.py
5. Try the prototype:   python prototype/app.py   (open http://127.0.0.1:8000)
