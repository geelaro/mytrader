"""mytrader Dashboard — Streamlit web UI.

Usage:
    pipenv run streamlit run dashboard.py
"""

import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from dashboard import main

main()
