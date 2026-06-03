"""Tests for utils/bootstrap.py — explicit runtime initialisation."""

import os

import pytest

from utils import bootstrap


def test_idempotent_call():
    """Multiple invocations must not error or repeat side effects."""
    bootstrap.setup_runtime()
    bootstrap.setup_runtime()
    assert bootstrap.is_setup_done() is True


def test_matplotlib_backend_env_set():
    """setup_runtime must export MPLBACKEND so any subsequent matplotlib
    import defaults to the non-interactive Agg backend.  Without this
    headless deployments (Docker / Streamlit) would crash on chart code."""
    bootstrap.setup_runtime()
    assert os.environ.get("MPLBACKEND") == "Agg"


def test_setup_runtime_exported_from_utils():
    """Public API check — `from utils import setup_runtime` works."""
    from utils import setup_runtime
    assert setup_runtime is bootstrap.setup_runtime
