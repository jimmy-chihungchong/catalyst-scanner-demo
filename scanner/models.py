"""Shared data models. NewsEvent is the contract between ANY data source and the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Assessment(str, Enum):
    STRONG_CATALYST = "STRONG_CATALYST"
    PUFFY_WEAK = "PUFFY_WEAK"
    MIXED_UNCLEAR = "MIXED_UNCLEAR"
    NO_FRESH_CATALYST = "NO_FRESH_CATALYST"

    @property
    def label(self) -> str:
        return {
            "STRONG_CATALYST": "STRONG CATALYST",
            "PUFFY_WEAK": "POTENTIAL PUFFY / WEAK CATALYST",
            "MIXED_UNCLEAR": "MIXED / UNCLEAR",
            "NO_FRESH_CATALYST": "NO FRESH CATALYST",
        }[self.value]


CatalystCategory = Literal[
    "clinical_trial_data", "fda_regulatory", "material_contract", "earnings_guidance", "m_and_a",
    "strategic_partnership", "offering_dilution", "reverse_split", "product_launch", "legal_regulatory",
    "analyst_action", "sector_macro", "other",
]


@dataclass
class NewsEvent:
    """One piece of news/filing tied to a ticker. Every source (fixture, Alpaca, Benzinga,
    EDGAR, wire RSS) must normalize into this shape — that is what makes sources swappable."""
    event_id: str
    ticker: str
    company: str
    headline: str
    source_publisher: str
    source_url: str | None = None
    source_title: str | None = None
    body: str | None = None                 # full text when available (AI prefers this over headline)
    published_at: datetime | None = None    # tz-aware; None = unverified timestamp
    price_move_text: str | None = None      # raw description from the feed / fixture
    market_cap_usd: float | None = None
    cap_tier_hint: str | None = None        # "small" | "large" | None (fixture category, no market data)
    is_replay: bool = False                 # True for historical fixtures (freshness judged at publish time)


@dataclass
class TestCase:
    """Fixture wrapper: the event plus grading metadata that is NEVER shown to the AI."""
    case_id: str
    category: str
    expected_classification: str
    event: NewsEvent


class CatalystAnalysis(BaseModel):
    """Structured AI output. Same schema for every provider.
    Field ORDER matters: evidence -> risks -> facts -> reasoning -> verdict, so the model
    weighs the material before committing to an assessment (verdict-last)."""
    catalyst_category: CatalystCategory = Field(description="Type of event (NOT the assessment)")
    evidence: list[str] = Field(description="Positive/substantive facts quoted from the source text")
    risk_factors: list[str] = Field(description="Negative or offsetting facts in the text (misses, dilution, "
                                                "regulatory actions, vague terms). Empty list only if truly none.")
    financial_value_disclosed: Literal["yes", "no", "not_applicable"]
    revenue_impact_disclosed: Literal["yes", "no", "not_applicable"]
    binding_agreement: Literal["yes", "no", "not_verified", "not_applicable"]
    clinical_fda_significance: Literal["high", "moderate", "low", "not_applicable"]
    dilution_risk: Literal["yes", "no", "unknown"]
    promotional_or_vague_language: bool
    reaction_disproportionate: Literal["yes", "no", "unknown"]
    reasoning: str = Field(description="1-3 sentences weighing evidence against risk_factors")
    assessment: Assessment
    confidence: float = Field(ge=0, le=1, description="Decimal between 0.0 and 1.0 (e.g. 0.85)")


@dataclass
class RuleResult:
    """Output of the deterministic layer (no LLM involved)."""
    move_pct: float | None
    move_window: str                 # "single-session" | "multi-period" | "unknown"
    cap_tier: str                    # "Small Cap" | "Large Cap" | "Unknown"
    cap_tier_source: str
    news_age_minutes: float | None
    freshness: str                   # "fresh" | "stale" | "unverified" | "replay"
    on_watchlist: bool
    dilution_keywords: list[str] = field(default_factory=list)
    fingerprint: str = ""


@dataclass
class Alert:
    ticker: str
    type: str
    move: str
    news_age: str
    source: str
    catalyst: str
    financial_value_disclosed: str
    revenue_impact_disclosed: str
    binding_agreement: str
    recent_dilution: str
    assessment: str
    watch: str
    evidence: list[str]
    reasoning: str
    notes: list[str] = field(default_factory=list)  # guardrail overrides, data caveats
