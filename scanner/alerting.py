"""Alert construction + formatting. Watch recommendation is deterministic, derived from assessment + cap tier.
Decision support only: nothing here (or anywhere in this project) places orders."""
from __future__ import annotations

from .models import Alert, Assessment, CatalystAnalysis, NewsEvent, RuleResult

# Original-source publishers; everything else is treated as secondary/aggregator coverage.
PRIMARY_SOURCES = {"sec edgar", "globenewswire", "business wire", "pr newswire", "accesswire",
                   "newsfile", "company ir", "investor relations"}

ASSESSMENT_DISPLAY = {
    Assessment.STRONG_CATALYST: "STRONG CATALYST — Potential Long Watch",
    Assessment.PUFFY_WEAK: "POTENTIAL PUFFY / WEAK CATALYST — Potential Short Watch",
    Assessment.MIXED_UNCLEAR: "MIXED / UNCLEAR — Requires manual review",
    Assessment.NO_FRESH_CATALYST: "NO FRESH CATALYST — No current verifiable company-specific catalyst found",
}

YES_NO = {"yes": "Yes", "no": "No", "not_verified": "Not verified", "not_applicable": "N/A", "unknown": "Unknown"}


def watch_recommendation(a: Assessment, tier: str, move_pct: float | None) -> str:
    wait = "Wait for Technical Setup"
    if a == Assessment.STRONG_CATALYST:
        if tier == "Large Cap":
            return f"Large-Cap Long Watch — {wait}"
        if tier == "Small Cap":
            return f"Small-Cap Long Watch — {wait}"
        return f"Long Watch (cap tier unverified) — {wait}"
    if a == Assessment.PUFFY_WEAK:
        if tier == "Large Cap":
            return "No Watch — large caps require a strong catalyst"
        if move_pct is None or move_pct <= 0:
            return "Risk flag only — no qualifying up-move to fade"
        return f"Short Watch — {wait}"
    if a == Assessment.MIXED_UNCLEAR:
        return "Manual Review — conflicting / unclear catalyst"
    return "No Watch — no verifiable fresh catalyst"


def _move(r: RuleResult) -> str:
    if r.move_pct is None:
        return "n/a (no % move in source data)"
    if r.move_window == "multi-period":
        return f"{r.move_pct:+.1f}% (multi-period, NOT a single-session move)"
    return f"{r.move_pct:+.1f}%"


def _age(r: RuleResult) -> str:
    if r.freshness == "replay":
        return "n/a — historical replay (fixture has no published timestamp)"
    if r.news_age_minutes is None:
        return "UNVERIFIED (no reliable timestamp)"
    m = r.news_age_minutes
    txt = f"{m:.0f} minutes" if m < 120 else f"{m / 60:.1f} hours" if m < 2880 else f"{m / 1440:.0f} days"
    return txt + (" (STALE)" if r.freshness == "stale" else "")


def build_alert(event: NewsEvent, r: RuleResult, a: CatalystAnalysis, notes: list[str],
                all_sources: list[str] | None = None) -> Alert:
    notes = list(notes)
    sources = all_sources or [event.source_publisher]
    if not any(s.lower() in PRIMARY_SOURCES for s in sources):
        notes.append(f"Source '{event.source_publisher}' is secondary coverage; original release not verified.")
    if r.cap_tier_source != "market cap":
        notes.append(f"Cap tier source: {r.cap_tier_source}.")
    if not r.on_watchlist:
        notes.append(f"{event.ticker} is not on the configured watchlist.")
    if not event.body:
        notes.append("Headline-only analysis (full text not ingested).")
    return Alert(
        ticker=event.ticker,
        type=r.cap_tier,
        move=_move(r),
        news_age=_age(r),
        source=", ".join(sources),
        catalyst=a.catalyst_category.replace("_", " ").title(),
        financial_value_disclosed=YES_NO[a.financial_value_disclosed],
        revenue_impact_disclosed=YES_NO[a.revenue_impact_disclosed],
        binding_agreement=YES_NO[a.binding_agreement],
        recent_dilution=YES_NO[a.dilution_risk],
        assessment=ASSESSMENT_DISPLAY[a.assessment],
        watch=watch_recommendation(a.assessment, r.cap_tier, r.move_pct),
        evidence=a.evidence,
        reasoning=a.reasoning,
        notes=notes,
    )


def format_alert(al: Alert) -> str:
    """Exact client-specified field order."""
    return "\n".join([
        f"Ticker: {al.ticker}",
        f"Type: {al.type}",
        f"Move: {al.move}",
        f"News age: {al.news_age}",
        f"Source: {al.source}",
        f"Catalyst: {al.catalyst}",
        f"Financial value disclosed: {al.financial_value_disclosed}",
        f"Revenue impact disclosed: {al.revenue_impact_disclosed}",
        f"Binding agreement: {al.binding_agreement}",
        f"Recent dilution: {al.recent_dilution}",
        f"Assessment: {al.assessment}",
        f"Watch: {al.watch}",
    ])
