"""Tests for data/realtime.py — Yahoo intraday VIX fetch."""

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from data.realtime import _extract_latest_quote, get_realtime_vix, reset_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    """Each test starts with empty realtime cache."""
    reset_cache()
    yield
    reset_cache()


def _mock_yahoo_response(payload: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.status_code = status
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status}")
    return resp


# ===================================================================
# _extract_latest_quote — pure parser
# ===================================================================


class TestExtractLatestQuote:
    def test_prefers_regular_market_price(self):
        data = {
            "chart": {"result": [{
                "meta": {"regularMarketPrice": 18.45},
                "indicators": {"quote": [{"close": [17.0, 17.5]}]},
            }]},
        }
        assert _extract_latest_quote(data) == 18.45

    def test_falls_back_to_last_close(self):
        """When meta.regularMarketPrice missing, scan close array."""
        data = {
            "chart": {"result": [{
                "meta": {},
                "indicators": {"quote": [{"close": [17.0, 17.5, 18.2]}]},
            }]},
        }
        assert _extract_latest_quote(data) == 18.2

    def test_skips_trailing_nulls_in_close(self):
        """Yahoo often pads with null for incomplete bars."""
        data = {
            "chart": {"result": [{
                "meta": {},
                "indicators": {"quote": [{"close": [17.0, 17.5, None, None]}]},
            }]},
        }
        assert _extract_latest_quote(data) == 17.5

    def test_zero_or_negative_price_rejected(self):
        data = {
            "chart": {"result": [{
                "meta": {"regularMarketPrice": 0},
                "indicators": {"quote": [{"close": [-1.0, 0.0]}]},
            }]},
        }
        assert _extract_latest_quote(data) is None

    def test_malformed_response_returns_none(self):
        assert _extract_latest_quote({}) is None
        assert _extract_latest_quote({"chart": None}) is None
        assert _extract_latest_quote({"chart": {"result": []}}) is None
        assert _extract_latest_quote({"chart": {"result": [None]}}) is None


# ===================================================================
# get_realtime_vix — fetch wrapper with cache
# ===================================================================


class TestGetRealtimeVix:
    def test_happy_path_returns_price(self):
        payload = {
            "chart": {"result": [{
                "meta": {"regularMarketPrice": 18.45},
                "indicators": {"quote": [{"close": []}]},
            }]},
        }
        session = MagicMock()
        session.get.return_value = _mock_yahoo_response(payload)
        with patch("data.realtime._yahoo_session", return_value=session):
            assert get_realtime_vix() == 18.45

    def test_returns_none_on_request_error(self):
        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("offline")
        with patch("data.realtime._yahoo_session", return_value=session):
            assert get_realtime_vix() is None

    def test_returns_none_on_timeout(self):
        session = MagicMock()
        session.get.side_effect = requests.Timeout("slow")
        with patch("data.realtime._yahoo_session", return_value=session):
            assert get_realtime_vix() is None

    def test_returns_none_on_http_error(self):
        session = MagicMock()
        session.get.return_value = _mock_yahoo_response({}, status=403)
        with patch("data.realtime._yahoo_session", return_value=session):
            assert get_realtime_vix() is None

    def test_returns_none_on_invalid_json(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.side_effect = ValueError("not json")
        session = MagicMock()
        session.get.return_value = resp
        with patch("data.realtime._yahoo_session", return_value=session):
            assert get_realtime_vix() is None

    def test_cache_hit_skips_network(self):
        """Second call within TTL should not hit Yahoo."""
        payload = {
            "chart": {"result": [{
                "meta": {"regularMarketPrice": 18.45},
                "indicators": {"quote": [{"close": []}]},
            }]},
        }
        session = MagicMock()
        session.get.return_value = _mock_yahoo_response(payload)
        with patch("data.realtime._yahoo_session", return_value=session):
            v1 = get_realtime_vix(ttl=60)
            v2 = get_realtime_vix(ttl=60)
            v3 = get_realtime_vix(ttl=60)
        assert v1 == v2 == v3 == 18.45
        # Only the first call hit Yahoo
        assert session.get.call_count == 1

    def test_cache_expires_after_ttl(self):
        """Calls after TTL expiry should re-hit Yahoo."""
        payload1 = {
            "chart": {"result": [{
                "meta": {"regularMarketPrice": 18.45},
                "indicators": {"quote": [{"close": []}]},
            }]},
        }
        payload2 = {
            "chart": {"result": [{
                "meta": {"regularMarketPrice": 19.20},
                "indicators": {"quote": [{"close": []}]},
            }]},
        }
        session = MagicMock()
        # First call returns 18.45, subsequent returns 19.20
        session.get.side_effect = [
            _mock_yahoo_response(payload1),
            _mock_yahoo_response(payload2),
        ]
        with patch("data.realtime._yahoo_session", return_value=session):
            v1 = get_realtime_vix(ttl=0.05)
            time.sleep(0.1)  # let TTL expire
            v2 = get_realtime_vix(ttl=0.05)
        assert v1 == 18.45
        assert v2 == 19.20
        assert session.get.call_count == 2

    def test_failure_not_cached_so_next_call_retries(self):
        """A failed fetch shouldn't poison the cache."""
        session = MagicMock()
        # First fails, second succeeds
        session.get.side_effect = [
            requests.ConnectionError("offline"),
            _mock_yahoo_response({
                "chart": {"result": [{
                    "meta": {"regularMarketPrice": 18.45},
                    "indicators": {"quote": [{"close": []}]},
                }]},
            }),
        ]
        with patch("data.realtime._yahoo_session", return_value=session):
            assert get_realtime_vix() is None
            assert get_realtime_vix() == 18.45  # retried, succeeded
        assert session.get.call_count == 2

    def test_reset_cache_forces_refetch(self):
        payload = {
            "chart": {"result": [{
                "meta": {"regularMarketPrice": 18.45},
                "indicators": {"quote": [{"close": []}]},
            }]},
        }
        session = MagicMock()
        session.get.return_value = _mock_yahoo_response(payload)
        with patch("data.realtime._yahoo_session", return_value=session):
            get_realtime_vix(ttl=60)
            assert session.get.call_count == 1
            reset_cache()
            get_realtime_vix(ttl=60)
        assert session.get.call_count == 2
