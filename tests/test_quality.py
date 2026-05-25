"""Tests for data/quality.py — data quality checks."""

import numpy as np
import pandas as pd
import pytest
from data.quality import (
    flag_missing, flag_price_jumps, flag_non_trading,
    quality_report, validate_ohlcv, clean,
)


@pytest.fixture
def clean_df():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame({
        "Open": close - 0.3, "High": close + 1,
        "Low": close - 1, "Close": close,
        "Volume": np.random.randint(1000, 10000, 100),
    }, index=dates)


class TestFlagMissing:
    def test_no_missing(self, clean_df):
        df = flag_missing(clean_df)
        assert "_missing" in df.columns
        assert not df["_missing"].any()

    def test_detects_nan(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[10], "Close"] = np.nan
        result = flag_missing(df)
        assert result["_missing"].sum() == 1
        assert result.loc[df.index[10], "_missing"]

    def test_detects_multi_column_nan(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[5], ["Open", "High"]] = np.nan
        result = flag_missing(df)
        assert result["_missing"].sum() == 1


class TestFlagPriceJumps:
    def test_no_jumps(self, clean_df):
        df = flag_price_jumps(clean_df)
        assert "_price_jump" in df.columns
        assert df["_price_jump"].sum() == 0

    def test_detects_jump(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[20], "Close"] = df.loc[df.index[19], "Close"] * 1.5
        result = flag_price_jumps(df)
        assert result["_price_jump"].sum() >= 1

    def test_jump_below_threshold(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[20], "Close"] = df.loc[df.index[19], "Close"] * 1.05
        result = flag_price_jumps(df, threshold_pct=20.0)
        assert result["_price_jump"].sum() == 0


class TestFlagNonTrading:
    def test_no_flag_on_normal_volume(self, clean_df):
        df = flag_non_trading(clean_df)
        assert "_non_trading" in df.columns

    def test_flags_low_volume(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[-1], "Volume"] = 1
        result = flag_non_trading(df)
        assert bool(result.loc[df.index[-1], "_non_trading"])


class TestValidateOhlcv:
    def test_valid(self, clean_df):
        ok, reason = validate_ohlcv(clean_df)
        assert ok
        assert reason == "ok"

    def test_empty(self):
        ok, reason = validate_ohlcv(pd.DataFrame())
        assert not ok
        assert "empty" in reason

    def test_none(self):
        ok, _ = validate_ohlcv(None)
        assert not ok

    def test_missing_column(self, clean_df):
        df = clean_df.drop(columns=["High"])
        ok, reason = validate_ohlcv(df)
        assert not ok
        assert "High" in reason

    def test_close_nan(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[-1], "Close"] = np.nan
        ok, reason = validate_ohlcv(df)
        assert not ok
        assert "nan" in reason

    def test_high_lt_low(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[-1], "High"] = 50
        df.loc[df.index[-1], "Low"] = 100
        ok, reason = validate_ohlcv(df)
        assert not ok

    def test_close_oob(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[-1], "Close"] = 999
        ok, reason = validate_ohlcv(df)
        assert not ok
        assert "oob" in reason


class TestQualityReport:
    def test_returns_dict(self, clean_df):
        df = flag_missing(clean_df)
        df = flag_price_jumps(df)
        df = flag_non_trading(df)
        report = quality_report(df)
        assert isinstance(report, dict)
        assert "bars" in report
        assert report["bars"] == 100
        assert report["missing_pct"] == 0.0
        assert report["price_jumps"] == 0

    def test_missing_bars_reported(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[5], "Close"] = np.nan
        df = flag_missing(df)
        df = flag_price_jumps(df)
        report = quality_report(df)
        assert report["missing_pct"] > 0


class TestClean:
    def test_drop_missing(self, clean_df):
        df = clean_df.copy()
        df.loc[df.index[10], "Close"] = np.nan
        result = clean(df, drop_missing=True)
        assert len(result) < len(df)

    def test_keep_all_when_no_drop(self, clean_df):
        result = clean(clean_df, drop_missing=False, drop_jumps=False)
        assert len(result) == len(clean_df)
