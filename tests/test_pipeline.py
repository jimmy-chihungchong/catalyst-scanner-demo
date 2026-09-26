"""Pipeline tests with a stub LLM (no network). Synthetic ACME strings are unit-test inputs only —
they are NOT used as demo test cases and are not presented as real news."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from scanner import evaluation
from scanner.alerting import format_alert
from scanner.classifier import LLMProvider
from scanner.config import Settings
from scanner.models import Assessment, CatalystAnalysis, NewsEvent
from scanner.pipeline import Pipeline
from scanner.storage import Store

NOW = datetime(2026, 8, 12, 14, 0, tzinfo=timezone.utc)


def analysis(**kw) -> CatalystAnalysis:
    base = dict(catalyst_category="strategic_partnership", assessment="PUFFY_WEAK", financial_value_disclosed="no",
                revenue_impact_disclosed="no", binding_agreement="not_verified", clinical_fda_significance="not_applicable",
                dilution_risk="no", promotional_or_vague_language=True, reaction_disproportionate="yes",
                evidence=["stub"], risk_factors=[], reasoning="stub", confidence=0.7)
    base.update(kw)
    return CatalystAnalysis(**base)


class StubProvider(LLMProvider):
    name, model = "stub", "stub-1"

    def __init__(self, result=None, fail=False):
        self.result, self.fail, self.calls = result or analysis(), fail, 0

    def _call(self, system, user):
        self.calls += 1
        if self.fail:
            raise ConnectionError("provider down")
        return self.result.model_dump(mode="json")


def ev(**kw):
    base = dict(event_id="e1", ticker="ACME", company="Acme", source_publisher="GlobeNewswire",
                headline="ACME announces strategic partnership with Widget Inc.",
                published_at=NOW - timedelta(minutes=3), price_move_text="up 48% premarket", market_cap_usd=5e7)
    base.update(kw)
    return NewsEvent(**base)


@pytest.fixture
def settings():
    return replace(Settings(), ai_max_retries=1)


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def test_alert_format_matches_spec(settings, store):
    res = Pipeline(settings, store, StubProvider()).process(ev(), NOW)
    keys = [line.split(":")[0] for line in res.alert_text.splitlines()]
    assert keys == ["Ticker", "Type", "Move", "News age", "Source", "Catalyst", "Financial value disclosed",
                    "Revenue impact disclosed", "Binding agreement", "Recent dilution", "Assessment", "Watch"]
    assert "Short Watch — Wait for Technical Setup" in res.alert_text
    assert "News age: 3 minutes" in res.alert_text


def test_duplicate_alert_suppressed_and_llm_not_recalled(settings, store):
    p = StubProvider()
    pipe = Pipeline(settings, store, p)
    assert pipe.process(ev(), NOW).alert_status == "SENT"
    second = pipe.process(ev(), NOW)
    assert second.alert_status == "SUPPRESSED" and second.cached and p.calls == 1


def test_cross_wire_duplicate_merges_sources(settings, store):
    pipe = Pipeline(settings, store, StubProvider())
    pipe.process(ev(), NOW)
    dup = pipe.process(ev(event_id="e2", source_publisher="Company IR",
                          headline="ACME Announces Strategic Partnership With Widget, Inc"), NOW)
    assert dup.duplicate_of is not None and dup.alert_status == "SUPPRESSED"
    assert dup.alert.source == "GlobeNewswire, Company IR"


def test_stale_news_forced_to_no_fresh_catalyst(settings, store):
    strong = analysis(assessment="STRONG_CATALYST")
    res = Pipeline(settings, store, StubProvider(strong)).process(ev(published_at=NOW - timedelta(days=40)), NOW)
    assert res.raw_analysis.assessment == Assessment.STRONG_CATALYST
    assert res.analysis.assessment == Assessment.NO_FRESH_CATALYST
    assert "GUARDRAIL" in res.guardrail_notes[0]


def test_missing_timestamp_live_forced_to_no_fresh(settings, store):
    res = Pipeline(settings, store, StubProvider(analysis(assessment="STRONG_CATALYST"))).process(ev(published_at=None), NOW)
    assert res.analysis.assessment == Assessment.NO_FRESH_CATALYST


def test_offering_cannot_be_strong(settings, store):
    a = analysis(assessment="STRONG_CATALYST", catalyst_category="offering_dilution")
    res = Pipeline(settings, store, StubProvider(a)).process(
        ev(headline="ACME prices $5M registered direct offering with warrants"), NOW)
    assert res.analysis.assessment == Assessment.MIXED_UNCLEAR and res.analysis.dilution_risk == "yes"


def test_provider_outage_degrades_gracefully(settings, store):
    res = Pipeline(settings, store, StubProvider(fail=True)).process(ev(), NOW)
    assert res.alert_status == "ERROR" and "provider down" in res.error


def test_classification_change_triggers_update(settings, store):
    Pipeline(settings, store, StubProvider()).process(ev(), NOW)
    store.conn.execute("DELETE FROM analyses")
    res = Pipeline(settings, store, StubProvider(analysis(assessment="MIXED_UNCLEAR"))).process(ev(), NOW)
    assert res.alert_status == "SENT" and "classification changed" in res.alert_reason


@pytest.mark.parametrize("expected,a,status", [
    ("STRONG CATALYST - Potential Long Watch", analysis(assessment="STRONG_CATALYST"), "PASS"),
    ("STRONG CATALYST - Potential Long Watch", analysis(assessment="MIXED_UNCLEAR"), "FAIL"),
    ("MIXED / UNCLEAR - Requires manual review", analysis(assessment="MIXED_UNCLEAR"), "PASS"),
    ("Dilution event - flag 'Recent dilution: Yes'; not itself a bullish catalyst", analysis(dilution_risk="yes"), "PASS"),
    ("Dilution event - flag 'Recent dilution: Yes'; not itself a bullish catalyst",
     analysis(dilution_risk="yes", assessment="STRONG_CATALYST"), "FAIL"),
    ("something unparseable", analysis(), "UNGRADED"),
])
def test_evaluation_parsing(expected, a, status):
    assert evaluation.evaluate(expected, a).status == status
