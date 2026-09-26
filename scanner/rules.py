"""Deterministic logic. Numbers, timestamps, eligibility, dedup — never delegated to the LLM."""
from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, time, timezone
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo

from .config import MARKET_TZ, MONITOR_END_PT, MONITOR_START_PT, TRADER_TZ, WATCHLIST, Settings
from .models import Assessment, CatalystAnalysis, NewsEvent, RuleResult

SMALL_CAP_MAX_USD = 2_000_000_000

# NYSE full-day closures 2026. Simplified: no early-close handling (post-hire: exchange_calendars lib).
NYSE_HOLIDAYS_2026 = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
}

_PCT_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")
_NEG_WORDS = re.compile(r"\b(down|fell|falls|dropped|drops|plunged|plunges|slid|sank|declined|tumbled)\b", re.I)
_MULTI_PERIOD = re.compile(r"\b(since|over the past|year[- ]to[- ]date|ytd|this year|in \d+ (days|weeks|months))\b", re.I)

DILUTION_PATTERNS = {
    "registered direct offering": r"registered direct offering",
    "public offering": r"(underwritten |public )offering",
    "private placement / PIPE": r"private placement|\bpipe\b",
    "warrants": r"\bwarrants?\b",
    "at-the-market (ATM)": r"at[- ]the[- ]market|\batm (program|offering|facility)\b",
    "shelf / prospectus (S-1/S-3/424B)": r"\bS-1\b|\bS-3\b|\b424B\d?\b|prospectus supplement",
    "convertible notes": r"convertible (notes?|debentures?|preferred)",
}
REVERSE_SPLIT_RE = re.compile(r"reverse (stock )?split", re.I)


def parse_move(text: str | None) -> tuple[float | None, str]:
    """Extract signed % move and whether it's a single-session or multi-period move."""
    if not text:
        return None, "unknown"
    m = _PCT_RE.search(text)
    if not m:
        return None, "unknown"
    pct = float(m.group(1))
    if pct > 0 and _NEG_WORDS.search(text[: m.start()]):
        pct = -pct
    window = "multi-period" if _MULTI_PERIOD.search(text) else "single-session"
    return pct, window


def cap_tier(event: NewsEvent) -> tuple[str, str]:
    if event.market_cap_usd is not None:
        return ("Small Cap" if event.market_cap_usd < SMALL_CAP_MAX_USD else "Large Cap"), "market cap"
    if event.cap_tier_hint == "small":
        return "Small Cap", "fixture category (no market-cap feed)"
    if event.cap_tier_hint == "large":
        return "Large Cap", "fixture category (no market-cap feed)"
    return "Unknown", "no market-cap data"


def freshness(event: NewsEvent, now: datetime, max_age_min: float) -> tuple[float | None, str]:
    """No verifiable timestamp => 'unverified' (live mode treats that as NOT fresh).
    Replayed fixtures are evaluated as-of publication, so they are 'replay'."""
    if event.published_at is None:
        return None, "replay" if event.is_replay else "unverified"
    if event.published_at.tzinfo is None:
        return None, "unverified"  # naive timestamps are ambiguous -> refuse to trust
    age = (now - event.published_at).total_seconds() / 60
    if age < -5:
        return age, "unverified"   # future-dated: clock/feed problem
    return age, "fresh" if age <= max_age_min else "stale"


def dilution_keywords(event: NewsEvent) -> list[str]:
    text = f"{event.headline}\n{event.body or ''}\n{event.source_title or ''}"
    return [label for label, pat in DILUTION_PATTERNS.items() if re.search(pat, text, re.I)]


def has_reverse_split(event: NewsEvent) -> bool:
    return bool(REVERSE_SPLIT_RE.search(f"{event.headline}\n{event.body or ''}"))


def normalize_headline(headline: str) -> str:
    h = headline.lower()
    h = re.sub(r"^\s*[\w .,&-]{0,40}\((nasdaq|nyse|nyse american|otc)[: ]+[a-z.]+\)\s*[-–—:]?\s*", "", h)  # wire prefixes
    h = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", h)  # keep decimal points ($1.17), drop sentence dots
    h = re.sub(r"[^a-z0-9$%. ]+", " ", h)
    return re.sub(r"\s+", " ", h).strip()


def fingerprint(ticker: str, headline: str) -> str:
    return hashlib.sha256(f"{ticker.upper()}|{normalize_headline(headline)}".encode()).hexdigest()[:16]


def is_near_duplicate(a: str, b: str, threshold: float) -> bool:
    """Cross-wire duplicates: same release on GlobeNewswire + company IR with tiny wording diffs."""
    return SequenceMatcher(None, normalize_headline(a), normalize_headline(b)).ratio() >= threshold


def in_monitor_window(now: datetime) -> bool:
    """05:00-16:00 PT on NYSE trading days."""
    pt = now.astimezone(ZoneInfo(TRADER_TZ))
    et_day = now.astimezone(ZoneInfo(MARKET_TZ)).date()
    if et_day.weekday() >= 5 or et_day in NYSE_HOLIDAYS_2026:
        return False
    start, end = time.fromisoformat(MONITOR_START_PT), time.fromisoformat(MONITOR_END_PT)
    return start <= pt.time() <= end


def meets_move_threshold(move_pct: float | None, tier: str, s: Settings) -> bool | None:
    if move_pct is None or tier == "Unknown":
        return None
    need = s.min_move_pct_small_cap if tier == "Small Cap" else s.min_move_pct_large_cap
    return abs(move_pct) >= need


_FINANCING_CATEGORY = re.compile(r"offering|financ|dilut|capital raise|placement", re.I)


def apply_guardrails(analysis: CatalystAnalysis, r: RuleResult) -> tuple[CatalystAnalysis, list[str]]:
    """Hard rules the LLM cannot override. Returns (possibly adjusted analysis, notes explaining changes)."""
    a = analysis.model_copy(deep=True)
    notes: list[str] = []
    # 1. No false catalyst matching: unverifiable or stale timestamp => NO FRESH CATALYST.
    if r.freshness in {"stale", "unverified"} and a.assessment != Assessment.NO_FRESH_CATALYST:
        notes.append(f"GUARDRAIL: news timestamp {r.freshness} -> forced NO FRESH CATALYST "
                     f"(AI said {a.assessment.value}).")
        a.assessment = Assessment.NO_FRESH_CATALYST
    # 2. Deterministic dilution detection wins over the model.
    if r.dilution_keywords and a.dilution_risk != "yes":
        notes.append(f"GUARDRAIL: dilution keywords {r.dilution_keywords} -> dilution_risk=yes (AI said {a.dilution_risk}).")
        a.dilution_risk = "yes"
    # 3. A financing/offering is never itself a STRONG bullish catalyst.
    if r.dilution_keywords and _FINANCING_CATEGORY.search(a.catalyst_category) and a.assessment == Assessment.STRONG_CATALYST:
        notes.append("GUARDRAIL: financing/offering event cannot be STRONG CATALYST -> downgraded to MIXED / UNCLEAR.")
        a.assessment = Assessment.MIXED_UNCLEAR
    return a, notes


def evaluate(event: NewsEvent, s: Settings, now: datetime | None = None) -> RuleResult:
    now = now or datetime.now(timezone.utc)
    pct, window = parse_move(event.price_move_text)
    tier, tier_src = cap_tier(event)
    age, fresh = freshness(event, now, s.max_news_age_minutes)
    return RuleResult(
        move_pct=pct, move_window=window, cap_tier=tier, cap_tier_source=tier_src,
        news_age_minutes=age, freshness=fresh, on_watchlist=event.ticker in WATCHLIST,
        dilution_keywords=dilution_keywords(event),
        fingerprint=fingerprint(event.ticker, event.headline),
    )
