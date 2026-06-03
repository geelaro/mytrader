"""Cron entry point — build the weekly risk report and push to Feishu.

Usage
-----
    # Manual run (writes report to logs/, sends Feishu)
    pipenv run python scripts/weekly_risk_report.py

    # Dry-run (markdown only, no Feishu send)
    pipenv run python scripts/weekly_risk_report.py --dry-run

    # Custom output path
    pipenv run python scripts/weekly_risk_report.py --out reports/2026-06.md

Scheduled cron example (weekly Monday 9am Beijing):
    0 9 * * 1 cd /path/to/traderbridge && \
        pipenv run python scripts/weekly_risk_report.py >> logs/weekly.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# Ensure project root is importable when run via ``python scripts/...``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Runtime bootstrap — must run before matplotlib / dotenv-aware imports.
from utils.bootstrap import setup_runtime  # noqa: E402
setup_runtime()

from analysis.risk_report import RiskReport  # noqa: E402
from config import config as runtime_config  # noqa: E402
from data import DataProvider  # noqa: E402
from data.cache import CacheManager  # noqa: E402
from utils import load_toml  # noqa: E402
from utils.notify import Notifier  # noqa: E402

logger = logging.getLogger("weekly_risk_report")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="Write Markdown to this file (default: logs/risk_report_YYYY-MM-DD.md)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Feishu send, only generate Markdown")
    parser.add_argument("--config", default="watchlist.toml",
                        help="watchlist config path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    config = load_toml(args.config)
    provider = DataProvider()
    cache = CacheManager()

    logger.info("Building weekly risk report...")
    report = RiskReport(config, provider, cache, target_date=date.today())
    data = report.build()
    md = report.to_markdown()

    # Write Markdown
    out_path = Path(args.out) if args.out else Path(
        "logs") / f"risk_report_{data['as_of']}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    logger.info("Markdown written to %s (%d bytes)", out_path, len(md))

    # Send Feishu unless --dry-run
    if args.dry_run:
        logger.info("--dry-run: skipping Feishu send")
        print(md)
        return

    notifier = Notifier(async_mode=False)
    if not notifier.available:
        logger.warning("Feishu notifier unavailable; skipping send")
        return
    card = report.to_feishu_card()
    ok = notifier._send({"msg_type": "interactive", "card": card})
    if ok:
        logger.info("Feishu card sent ✓")
    else:
        logger.warning("Feishu send failed")


if __name__ == "__main__":
    main()
