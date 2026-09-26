"""Deterministic-layer tests. Synthetic strings here are unit-test inputs, not test-case news."""
from datetime import datetime, timedelta, timezone

from scanner import rules
from scanner.config import Settings
from scanner.models import NewsEvent

NOW = datetime(2026, 8, 12, 14, 0, tzinfo=timezone.utc)


def ev(**kw):
    base = dict(event_id="t", ticker="TEST", company="Test Co", headline="Test headline", source_publisher="unit")
    base.update(kw)
    return NewsEvent(**base)


def test_parse_move_fixture_formats():
    assert rules.parse_move("Jumped 17.3% on August 12, 2026") == (17.3, "single-session")
    assert rules.parse_move("Stock surged approximately 22% on August 6, 2026 despite the earnings miss") == (22.0, "single-session")
    assert rules.parse_move("Stock up over 660% since end of July 2026, trading around $99.30") == (660.0, "multi-period")
    assert rules.parse_move("Offering closed February 9, 2026; gross proceeds approximately $7.4 million") == (None, "unknown")
    assert rules.parse_move("Shares fell 12% premarket") == (-12.0, "single-session")
    assert rules.parse_move(None) == (None, "unknown")


def test_freshness_rules():
    s = Settings()
    assert rules.freshness(ev(published_at=NOW - timedelta(minutes=3)), NOW, 90) == (3.0, "fresh")
    assert rules.freshness(ev(published_at=NOW - timedelta(days=30)), NOW, 90)[1] == "stale"
    assert rules.freshness(ev(published_at=None), NOW, 90) == (None, "unverified")
    assert rules.freshness(ev(published_at=None, is_replay=True), NOW, 90) == (None, "replay")
    assert rules.freshness(ev(published_at=datetime(2026, 8, 12, 13, 0)), NOW, 90)[1] == "unverified"  # naive tz
    assert rules.freshness(ev(published_at=NOW + timedelta(hours=1)), NOW, 90)[1] == "unverified"  # future-dated
    assert s.max_news_age_minutes > 0


def test_dilution_detection():
    e = ev(headline="Upexi announces registered direct offering of 6,337,000 shares plus warrants at $1.17 per share")
    found = rules.dilution_keywords(e)
    assert "registered direct offering" in found and "warrants" in found
    assert rules.dilution_keywords(ev(headline="Company reports record Q2 revenue")) == []


def test_reverse_split():
    assert rules.has_reverse_split(ev(headline="Acme announces 1-for-20 reverse stock split"))


def test_dedup_exact_and_near():
    a = "ACME Corp (NASDAQ: ACME) - ACME Announces Strategic Partnership with Widget Inc."
    b = "ACME Announces Strategic Partnership With Widget Inc"
    assert rules.fingerprint("ACME", a) == rules.fingerprint("acme", b)
    assert rules.is_near_duplicate("ACME announces strategic partnership with Widget Inc.",
                                   "ACME announces a strategic partnership with Widget, Inc.", 0.9)
    assert not rules.is_near_duplicate("ACME announces partnership", "ACME prices $5M public offering", 0.9)


def test_monitor_window_and_holidays():
    assert rules.in_monitor_window(datetime(2026, 8, 12, 13, 0, tzinfo=timezone.utc))      # Wed 06:00 PT
    assert not rules.in_monitor_window(datetime(2026, 8, 12, 11, 0, tzinfo=timezone.utc))  # Wed 04:00 PT
    assert not rules.in_monitor_window(datetime(2026, 9, 7, 17, 0, tzinfo=timezone.utc))   # Labor Day
    assert not rules.in_monitor_window(datetime(2026, 8, 15, 17, 0, tzinfo=timezone.utc))  # Saturday


def test_cap_tier_and_threshold():
    s = Settings()
    assert rules.cap_tier(ev(market_cap_usd=5e8))[0] == "Small Cap"
    assert rules.cap_tier(ev(market_cap_usd=5e10))[0] == "Large Cap"
    assert rules.cap_tier(ev(cap_tier_hint="large"))[0] == "Large Cap"
    assert rules.cap_tier(ev())[0] == "Unknown"
    assert rules.meets_move_threshold(12, "Small Cap", s) is True
    assert rules.meets_move_threshold(6, "Small Cap", s) is False
    assert rules.meets_move_threshold(6, "Large Cap", s) is True
    assert rules.meets_move_threshold(None, "Large Cap", s) is None
