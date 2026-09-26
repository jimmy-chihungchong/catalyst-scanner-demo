"""Orchestration: event -> rules -> dedup -> AI -> guardrails -> alert. Source-agnostic.

Live mode would call Pipeline.process() from (a) a news-stream callback (breaking news, immediate)
and (b) a 15-min scanner job (movers) — same function, no rewrite.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from . import rules
from .alerting import build_alert, format_alert
from .classifier import LLMProvider, classify_catalyst, get_provider
from .config import Settings
from .models import Alert, CatalystAnalysis, NewsEvent, RuleResult
from .storage import Store

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    event: NewsEvent
    rules: RuleResult
    raw_analysis: CatalystAnalysis | None = None
    analysis: CatalystAnalysis | None = None
    guardrail_notes: list[str] = field(default_factory=list)
    alert: Alert | None = None
    alert_text: str = ""
    alert_status: str = ""          # SENT | SUPPRESSED | SKIPPED | ERROR | NOT_RUN
    alert_reason: str = ""
    duplicate_of: str | None = None
    cached: bool = False
    latency_s: float = 0.0
    error: str | None = None


class Pipeline:
    def __init__(self, settings: Settings, store: Store, provider: LLMProvider | None = None):
        self.s = settings
        self.store = store
        self._provider = provider  # created lazily: cached results work even if the provider's key is missing
        self.provider_name = provider.name if provider else settings.ai_provider
        self.model = provider.model if provider else settings.model_for(settings.ai_provider)

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_provider(self.s)
        return self._provider

    def process(self, event: NewsEvent, now: datetime | None = None, cache_only: bool = False) -> PipelineResult:
        """cache_only=True: never call the LLM; return NOT_RUN if no cached analysis (used by the UI on load)."""
        t0 = time.perf_counter()
        r = rules.evaluate(event, self.s, now)
        res = PipelineResult(event=event, rules=r)

        if self.s.watchlist_enforce and not r.on_watchlist:
            res.alert_status, res.alert_reason = "SKIPPED", "not on watchlist"
            return res

        # News dedup: same story from another wire/provider -> attach source, reuse analysis.
        dup = self.store.find_duplicate(event.ticker, r.fingerprint, event.headline, self.s.dedup_similarity)
        fp = dup or r.fingerprint
        if dup:
            res.duplicate_of = dup
            self.store.add_source(dup, event.source_publisher, event.source_url)
        else:
            self.store.save_event(fp, event)

        # AI analysis (cached per story + provider + model).
        cached = self.store.get_analysis(fp, self.provider_name, self.model)
        if cache_only and not cached:
            res.alert_status, res.alert_reason = "NOT_RUN", "no analysis yet for this provider"
            return res
        try:
            if cached:
                raw = CatalystAnalysis.model_validate(cached)
                res.cached = True
            else:
                raw = classify_catalyst(event, r, self.provider, self.s)
                self.store.save_analysis(fp, self.provider_name, self.model, raw.model_dump(mode="json"))
        except Exception as e:  # provider outage: degrade, never crash the scanner
            log.error("Classification failed for %s: %s", event.ticker, e)
            res.error = str(e)
            res.alert_status, res.alert_reason = "ERROR", "AI provider unavailable — manual review required"
            res.latency_s = time.perf_counter() - t0
            return res

        res.raw_analysis = raw
        res.analysis, res.guardrail_notes = rules.apply_guardrails(raw, r)
        res.alert = build_alert(event, r, res.analysis, res.guardrail_notes, self.store.sources_for(fp))
        res.alert_text = format_alert(res.alert)

        send, why = self.store.should_alert(event.ticker, fp, res.analysis.assessment.value, r.move_pct)
        res.alert_status, res.alert_reason = ("SENT" if send else "SUPPRESSED"), why
        if send:
            self.store.record_alert(event.ticker, fp, res.analysis.assessment.value, r.move_pct, res.alert_text)
        res.latency_s = time.perf_counter() - t0
        return res
