"""US macro release calendar — NFP / CPI / FOMC tagging.

V1: heuristic only — uses calendar rules instead of actual published
release dates.  This gives ~90% accuracy on identifying release days:

- **NFP** (Non-Farm Payrolls) — first Friday of month, 8:30 AM ET.
  Edge cases: NFP shifts to second Friday if first Friday is a federal
  holiday (rare).
- **CPI** — typically Tue/Wed/Thu in second week of month, day 10-15,
  8:30 AM ET.  Edge cases: occasionally day 8 or 16-17.
- **FOMC** — 8 meetings/year, typically Tue-Wed.  Without an external
  schedule of actual meeting dates, this is currently a stub returning
  ``False``.  Plug in actual dates when needed.

Why heuristic?
--------------
For "historical analog" / event-study analyses, perfect accuracy isn't
needed — we want to bucket past drops by likely driver.  A 90%-accurate
tag is far better than no tag, and avoids a maintenance burden of a
hand-curated calendar file.

Future: replace ``is_*_release`` implementations with a lookup against
a release-dates CSV.  The :func:`macro_tag` API stays stable.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def is_nfp_release(date) -> bool:
    """First Friday of the month (NFP release).

    Accepts anything :class:`pandas.Timestamp` can parse.  Returns False
    on parse failure rather than raising — calendar checks are best-effort.
    """
    try:
        d = pd.Timestamp(date)
    except Exception:
        return False
    if d.weekday() != 4:  # Friday
        return False
    return d.day <= 7


def is_cpi_release(date) -> bool:
    """Second-week Tue/Wed/Thu (CPI release window)."""
    try:
        d = pd.Timestamp(date)
    except Exception:
        return False
    if d.weekday() not in (1, 2, 3):  # Tue/Wed/Thu
        return False
    return 10 <= d.day <= 15


def is_fomc_release(date) -> bool:
    """FOMC meeting day.  Currently a stub — returns False.

    Plug in actual meeting dates here (or load from an external file)
    when needed.  The function signature is kept so callers don't need
    to change.
    """
    return False


def macro_tag(date) -> Optional[str]:
    """Return a single tag describing the macro release for ``date``.

    Priority: FOMC → NFP → CPI → None.  Returns the first match;
    multi-event days (rare) get the highest-priority tag.

    Returns one of: ``"FOMC"``, ``"NFP"``, ``"CPI"``, or ``None``.
    """
    if is_fomc_release(date):
        return "FOMC"
    if is_nfp_release(date):
        return "NFP"
    if is_cpi_release(date):
        return "CPI"
    return None
