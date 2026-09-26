"""Data sources. The pipeline only depends on NewsSource.fetch() -> NewsEvent.

Going live = implement another NewsSource subclass and set NEWS_SOURCE in .env.
Nothing downstream (rules, classifier, alerting, storage) changes.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .config import Settings
from .models import NewsEvent, TestCase

log = logging.getLogger(__name__)

REQUIRED_FIELDS = ("ticker", "company", "headline", "expected_classification", "source")


class NewsSource(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self) -> Iterator[NewsEvent]:
        """Yield normalized events. Live sources: poll/stream since last cursor."""


class FixtureNewsSource(NewsSource):
    """Replays canned, sourced test cases from JSON. Skips cases with null required fields."""
    name = "fixture"

    def __init__(self, path: Path):
        self.path = path
        self.skipped: list[tuple[str, list[str]]] = []

    def load_cases(self) -> list[TestCase]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        cases: list[TestCase] = []
        self.skipped = []
        for item in raw.get("test_cases", []):
            missing = [f for f in REQUIRED_FIELDS if not item.get(f)]
            if missing:
                self.skipped.append((item.get("id", "<no id>"), missing))
                log.info("Skipping %s: null fields %s", item.get("id"), missing)
                continue
            cases.append(TestCase(
                case_id=item["id"],
                category=item.get("category", ""),
                expected_classification=item["expected_classification"],
                event=self._to_event(item),
            ))
        return cases

    def fetch(self) -> Iterator[NewsEvent]:
        for case in self.load_cases():
            yield case.event

    @staticmethod
    def _to_event(item: dict) -> NewsEvent:
        src = item["source"]
        category = item.get("category", "")
        hint = "small" if category.startswith("small_cap") else "large" if category.startswith("large_cap") else None
        published = item.get("published_at")  # optional ISO-8601 with offset; not in current fixtures
        return NewsEvent(
            event_id=item["id"],
            ticker=item["ticker"].upper(),
            company=item["company"],
            headline=item["headline"],
            source_publisher=src.get("publisher", "unknown"),
            source_url=src.get("url"),
            source_title=src.get("title"),
            body=item.get("body"),
            published_at=datetime.fromisoformat(published) if published else None,
            price_move_text=item.get("price_move"),
            market_cap_usd=item.get("market_cap_usd"),
            cap_tier_hint=hint,
            is_replay=True,
            # NOTE: reasoning_notes and expected_classification deliberately NOT copied
            # onto the event — they are answer-key data and must never reach the AI.
        )


# --- Deferred live sources (post-hire). Stubs document where each plugs in. ---

class AlpacaNewsSource(NewsSource):
    """Alpaca real-time news WebSocket (wss://stream.data.alpaca.markets/v1beta1/news).
    Benzinga-sourced; push-based so breaking news never waits for the 15-min scan."""
    name = "alpaca"

    def fetch(self) -> Iterator[NewsEvent]:
        raise NotImplementedError("Deferred: Alpaca account not set up yet.")


class EdgarFilingSource(NewsSource):
    """SEC EDGAR: poll the 'latest filings' Atom feed for 8-K/6-K/S-1/S-3/424B*, resolve
    CIK->ticker via data.sec.gov company_tickers.json. Used to verify/enrich, not replace news."""
    name = "edgar"

    def fetch(self) -> Iterator[NewsEvent]:
        raise NotImplementedError("Deferred: EDGAR polling is out of demo scope.")


def get_source(settings: Settings) -> NewsSource:
    if settings.news_source == "fixture":
        return FixtureNewsSource(settings.fixture_path)
    if settings.news_source == "alpaca":
        return AlpacaNewsSource()
    if settings.news_source == "edgar":
        return EdgarFilingSource()
    raise ValueError(f"Unknown NEWS_SOURCE={settings.news_source!r}")
