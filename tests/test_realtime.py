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


def _spark_payload(price: float) -> dict:
    """Build a Yahoo spark-shaped response with the given final close."""
    return {"^VIX": {"timestamp": [1, 2], "close": [17.0, price]}}


class TestGetRealtimeVix:
    def test_happy_path_returns_price(self):
        session = MagicMock()
        session.get.return_value = _mock_yahoo_response(_spark_payload(18.45))
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
        """All three endpoints fail in different ways → None."""
        # JSON failure for spark + chart, HTML response without the pattern
        json_fail = MagicMock()
        json_fail.raise_for_status = MagicMock()
        json_fail.json.side_effect = ValueError("not json")
        html_fail = MagicMock()
        html_fail.raise_for_status = MagicMock()
        html_fail.text = "<html>no vix here</html>"
        session = MagicMock()
        session.get.side_effect = [json_fail, json_fail, html_fail]
        with patch("data.realtime._yahoo_session", return_value=session):
            assert get_realtime_vix() is None

    def test_cache_hit_skips_network(self):
        """Second call within TTL should not hit Yahoo."""
        session = MagicMock()
        session.get.return_value = _mock_yahoo_response(_spark_payload(18.45))
        with patch("data.realtime._yahoo_session", return_value=session):
            v1 = get_realtime_vix(ttl=60)
            v2 = get_realtime_vix(ttl=60)
            v3 = get_realtime_vix(ttl=60)
        assert v1 == v2 == v3 == 18.45
        # Only the first call hit Yahoo (spark succeeds on first try)
        assert session.get.call_count == 1

    def test_cache_expires_after_ttl(self):
        """Calls after TTL expiry should re-hit Yahoo."""
        session = MagicMock()
        session.get.side_effect = [
            _mock_yahoo_response(_spark_payload(18.45)),
            _mock_yahoo_response(_spark_payload(19.20)),
        ]
        with patch("data.realtime._yahoo_session", return_value=session):
            v1 = get_realtime_vix(ttl=0.05)
            time.sleep(0.1)  # let TTL expire
            v2 = get_realtime_vix(ttl=0.05)
        assert v1 == 18.45
        assert v2 == 19.20
        assert session.get.call_count == 2

    def test_failure_not_cached_so_next_call_retries(self):
        """A failed fetch shouldn't poison the cache.

        With three-endpoint fallback (spark, chart, HTML), a total failure
        consumes 3 calls.  Then a successful spark call on retry adds 1 more.
        """
        session = MagicMock()
        session.get.side_effect = [
            # First attempt: all three endpoints down
            requests.ConnectionError("offline"),  # spark
            requests.ConnectionError("offline"),  # chart
            requests.ConnectionError("offline"),  # html
            # Second attempt: spark succeeds immediately
            _mock_yahoo_response(_spark_payload(18.45)),
        ]
        with patch("data.realtime._yahoo_session", return_value=session):
            assert get_realtime_vix() is None
            assert get_realtime_vix() == 18.45
        assert session.get.call_count == 4

    def test_reset_cache_forces_refetch(self):
        session = MagicMock()
        session.get.return_value = _mock_yahoo_response(_spark_payload(18.45))
        with patch("data.realtime._yahoo_session", return_value=session):
            get_realtime_vix(ttl=60)
            assert session.get.call_count == 1
            reset_cache()
            get_realtime_vix(ttl=60)
        assert session.get.call_count == 2

    def test_falls_back_to_chart_when_spark_fails(self):
        """If spark endpoint fails, chart endpoint is tried as backup."""
        session = MagicMock()
        chart_payload = {
            "chart": {"result": [{
                "meta": {"regularMarketPrice": 18.45},
                "indicators": {"quote": [{"close": []}]},
            }]},
        }
        session.get.side_effect = [
            _mock_yahoo_response({}, status=429),         # spark rate-limited
            _mock_yahoo_response(chart_payload),          # chart works
        ]
        with patch("data.realtime._yahoo_session", return_value=session):
            assert get_realtime_vix() == 18.45
        assert session.get.call_count == 2

    def test_falls_back_to_html_when_both_apis_fail(self):
        """spark + chart 429 → HTML scrape succeeds."""
        html_with_vix = (
            '<html><body>'
            '<fin-streamer data-symbol="^VIX" '
            'data-field="regularMarketPrice" value="22.18"></fin-streamer>'
            '</body></html>'
        )
        session = MagicMock()
        # Build the HTML response mock — needs .text, not .json
        html_resp = MagicMock()
        html_resp.status_code = 200
        html_resp.text = html_with_vix
        html_resp.raise_for_status = MagicMock()
        session.get.side_effect = [
            _mock_yahoo_response({}, status=429),  # spark
            _mock_yahoo_response({}, status=429),  # chart
            html_resp,                              # HTML
        ]
        with patch("data.realtime._yahoo_session", return_value=session):
            assert get_realtime_vix() == 22.18
        assert session.get.call_count == 3

    def test_html_missing_fin_streamer_returns_none(self):
        """If HTML doesn't contain the expected pattern, return None."""
        bad_html = "<html><body>price page rearranged</body></html>"
        session = MagicMock()
        html_resp = MagicMock()
        html_resp.status_code = 200
        html_resp.text = bad_html
        html_resp.raise_for_status = MagicMock()
        session.get.side_effect = [
            _mock_yahoo_response({}, status=429),
            _mock_yahoo_response({}, status=429),
            html_resp,
        ]
        with patch("data.realtime._yahoo_session", return_value=session):
            assert get_realtime_vix() is None
