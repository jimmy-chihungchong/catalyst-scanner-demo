"""Central configuration. Everything tunable lives in env vars (.env), never in code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Hardcoded demo watchlist (production: pulled from Trade Ideas / market-data scanner).
# Ordered as provided by the client (tuple keeps display order; membership checks still work).
WATCHLIST: tuple[str, ...] = tuple(
    """AAPL MSFT NVDA AMZN TSLA GOOGL META AMD NFLX JPM XOM PFE JNJ BA DIS INTC ORCL WMT KO CRM
    MARA RIOT PLUG FCEL SNDL ATER PROG MULN NKLA SAVA AXSM OCGN NVAX BKKT GREE IINN SLNO CYCC
    ATXI ATNF XELB ANY IMPP KAVL BENF HUBC TOP INDO GNS CISS""".split()
)

MARKET_TZ = "America/New_York"   # exchange time: sessions, holidays
TRADER_TZ = "America/Los_Angeles"  # monitoring window 05:00-16:00 PT
MONITOR_START_PT = "05:00"
MONITOR_END_PT = "16:00"


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    ai_provider: str = field(default_factory=lambda: os.getenv("AI_PROVIDER", "lmstudio").lower())
    lmstudio_base_url: str = field(default_factory=lambda: os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"))
    lmstudio_model: str = field(default_factory=lambda: os.getenv("LMSTUDIO_MODEL", "google/gemma-4-12b-qat"))
    # none | low | medium | high. "none" disables Gemma-4 thinking (much faster; thinking can exceed the timeout)
    lmstudio_reasoning_effort: str = field(default_factory=lambda: os.getenv("LMSTUDIO_REASONING_EFFORT", "none"))
    nim_base_url: str = field(default_factory=lambda: os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    nim_model: str = field(default_factory=lambda: os.getenv("NIM_MODEL", "google/gemma-4-31b-it"))
    # Which providers the UI offers. Hosted/public deploy: set to "nim" so Anthropic/LM Studio are never exposed.
    enabled_providers: tuple[str, ...] = field(default_factory=lambda: tuple(
        p.strip().lower() for p in os.getenv("ENABLED_PROVIDERS", "anthropic,lmstudio,nim").split(",") if p.strip()))
    # If set, the UI's "Re-run all" (which spends API credits) requires this passcode.
    rerun_passcode: str = field(default_factory=lambda: os.getenv("RERUN_PASSCODE", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"))
    ai_timeout: float = field(default_factory=lambda: float(os.getenv("AI_TIMEOUT_SECONDS", "120")))
    ai_max_retries: int = field(default_factory=lambda: int(os.getenv("AI_MAX_RETRIES", "3")))

    news_source: str = field(default_factory=lambda: os.getenv("NEWS_SOURCE", "fixture").lower())
    fixture_path: Path = field(default_factory=lambda: PROJECT_ROOT / os.getenv("FIXTURE_PATH", "data/test_cases.json"))
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / os.getenv("DB_PATH", "data/scanner.db"))

    min_move_pct_small_cap: float = field(default_factory=lambda: float(os.getenv("MIN_MOVE_PCT_SMALL_CAP", "10")))
    min_move_pct_large_cap: float = field(default_factory=lambda: float(os.getenv("MIN_MOVE_PCT_LARGE_CAP", "4")))
    max_news_age_minutes: float = field(default_factory=lambda: float(os.getenv("MAX_NEWS_AGE_MINUTES", "90")))
    dedup_similarity: float = field(default_factory=lambda: float(os.getenv("DEDUP_SIMILARITY", "0.90")))
    watchlist_enforce: bool = field(default_factory=lambda: _bool("WATCHLIST_ENFORCE", False))


    def model_for(self, provider: str) -> str:
        return {"anthropic": self.anthropic_model, "lmstudio": self.lmstudio_model, "nim": self.nim_model}[provider]


def get_settings() -> Settings:
    return Settings()
