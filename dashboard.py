"""traderbridge Dashboard — Streamlit web UI.

Usage:
    pipenv run streamlit run dashboard.py
"""

import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

# Runtime bootstrap — Streamlit may pre-import matplotlib so this must
# run before `from dashboard import main`.
from utils.bootstrap import setup_runtime
setup_runtime()

from dashboard import main

main()
