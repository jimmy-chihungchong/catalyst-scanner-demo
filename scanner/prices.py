"""Price data for charts + deterministic move verification.

PriceSource mirrors NewsSource: YFinancePriceSource is DEMO-ONLY (unofficial Yahoo access, no SLA).
Production swap: AlpacaPriceSource using the official market-data API — nothing else changes.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

log = logging.getLogger(__name__)

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_FULL_DATE = re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}}),\s*(\d{{4}})", re.I)
_END_OF_MONTH = re.compile(rf"\bend of ({_MONTHS})\s+(\d{{4}})", re.I)
_MONTH_YEAR = re.compile(rf"\b({_MONTHS})\s+(\d{{4}})", re.I)


def parse_event_date(text: str | None) -> tuple[date | None, str]:
    """Deterministically pull the event date from source text. Returns (date, precision)."""
    if not text:
        return None, "none"
    if m := _FULL_DATE.search(text):
        return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y").date(), "day"
    if m := _END_OF_MONTH.search(text):
        first = datetime.strptime(f"{m.group(1)} 1 {m.group(2)}", "%B %d %Y").date()
        nxt = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        return nxt - timedelta(days=1), "approx (end of month)"
    if m := _MONTH_YEAR.search(text):
        return datetime.strptime(f"{m.group(1)} 1 {m.group(2)}", "%B %d %Y").date(), "approx (month)"
    return None, "none"


@dataclass
class MoveCheck:
    event_date: date | None
    trading_date: date | None      # first trading session on/after the event date
    verified_pct: float | None     # close-to-close % on that session
    claimed_pct: float | None
    status: str                    # "match" | "mismatch" | "unverifiable"
    detail: str


class PriceSource(ABC):
    name = "base"
    demo_only = False

    @abstractmethod
    def history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Daily bars indexed by naive date with columns Open, High, Low, Close, Volume."""


class YFinancePriceSource(PriceSource):
    name = "yfinance"
    demo_only = True

    def history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf
        df = yf.Ticker(ticker).history(start=start, end=end + timedelta(days=1), interval="1d", auto_adjust=False)
        if df.empty:
            return df
        df.index = df.index.tz_localize(None).normalize()
        return df[["Open", "High", "Low", "Close", "Volume"]]


def verify_move(df: pd.DataFrame, event_date: date | None, claimed_pct: float | None, window: str,
                tolerance_pts: float = 5.0) -> MoveCheck:
    """Compare the source's claimed % move against actual closes. Multi-period claims are measured
    from the event date to the latest close."""
    if event_date is None or df.empty:
        return MoveCheck(event_date, None, None, claimed_pct, "unverifiable", "no event date or no price data")
    closes = df["Close"]
    after = closes.loc[pd.Timestamp(event_date):]
    before = closes.loc[: pd.Timestamp(event_date) - pd.Timedelta(days=1)]
    if after.empty or before.empty:
        return MoveCheck(event_date, None, None, claimed_pct, "unverifiable", "event date outside price history")
    session = after.index[0]
    if window == "multi-period":
        start_close = before.iloc[-1]  # close into the event date
        pct = (closes.iloc[-1] / start_close - 1) * 100
        detail = f"close {before.index[-1].date()} ${start_close:,.2f} -> latest {closes.index[-1].date()} ${closes.iloc[-1]:,.2f}"
    else:
        pct = (after.iloc[0] / before.iloc[-1] - 1) * 100
        detail = f"prev close ${before.iloc[-1]:,.2f} -> {session.date()} close ${after.iloc[0]:,.2f}"
    if claimed_pct is None:
        status = "unverifiable"
    else:
        status = "match" if abs(pct - claimed_pct) <= tolerance_pts else "mismatch"
    return MoveCheck(event_date, session.date(), round(float(pct), 1), claimed_pct, status, detail)
