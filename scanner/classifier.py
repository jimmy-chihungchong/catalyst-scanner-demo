"""Provider-agnostic AI catalyst classification.

classify_catalyst(event, rules) is the only entry point the pipeline uses. Providers are
selected by AI_PROVIDER in .env; each returns the same validated CatalystAnalysis.
Adding OpenAI/Google = one new subclass of LLMProvider, nothing else changes.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import time
from abc import ABC, abstractmethod

from pydantic import ValidationError

from .config import Settings, get_settings
from .models import Assessment, CatalystAnalysis, NewsEvent, RuleResult

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a catalyst analyst supporting a discretionary U.S. equity day trader.
You do NOT give trade signals. You assess the SUBSTANCE of company news relative to the stock's reaction.

Base your analysis ONLY on the source text provided. If a fact is not stated in the text, treat it as
not disclosed / not verified — do not fill gaps from memory.

Assessment definitions:
- STRONG_CATALYST: fresh, company-specific, materially significant news with concrete substance
  (e.g. disclosed-value contracts, positive pivotal/Phase 3 data or FDA decisions, acquisitions,
  earnings/guidance beats or raises, significant business developments).
- PUFFY_WEAK: news that is vague, promotional, non-binding, financially insignificant, lacks
  disclosed terms, or is disproportionate to the price reaction. Company financings/offerings are
  NOT bullish catalysts: they dilute existing holders.
- MIXED_UNCLEAR: the text contains material signals pointing in conflicting directions of comparable
  weight (e.g. a miss alongside a reaffirmation), or substance cannot be judged — needs manual review.
- NO_FRESH_CATALYST: no company-specific catalyst can be identified in the text.

Field rules:
- financial_value_disclosed: "yes" only if a dollar/unit value of the deal/event is stated.
- revenue_impact_disclosed: "yes" only if revenue, guidance, or contract value tied to revenue is stated.
- binding_agreement: "yes" only if the text indicates a definitive/executed agreement; "not_verified" if an
  agreement is mentioned without that; "not_applicable" if no agreement is involved.
- dilution_risk: "yes" if the text describes share issuance, offerings, warrants, convertibles, ATM.
- reaction_disproportionate: compare the stated price move to the substance; "unknown" if no move given.
- catalyst_category: the TYPE of event. Any issuance of shares/warrants/convertibles is "offering_dilution";
  "m_and_a" only for acquisitions/mergers.
- evidence: short quotes or facts copied from the text supporting the catalyst.
- risk_factors: list every negative or offsetting fact in the text before deciding. If positives and
  material negatives coexist, weigh them explicitly in reasoning.
- reasoning: 1-3 sentences; decide the assessment only after listing evidence and risk_factors.
- confidence: a decimal from 0.0 to 1.0 (e.g. 0.8), NOT a 1-5 or 1-10 score.
Return ONLY the structured result."""


def build_user_prompt(event: NewsEvent, r: RuleResult) -> str:
    move = f"{r.move_pct:+.1f}% ({r.move_window})" if r.move_pct is not None else "not provided"
    lines = [
        f"Ticker: {event.ticker}",
        f"Company: {event.company}",
        f"Source publisher: {event.source_publisher}",
        f"Source title: {event.source_title or 'n/a'}",
        f"Headline: {event.headline}",
        f"Full text: {event.body or '(not available — headline only)'}",
        f"Price move description: {event.price_move_text or 'not provided'}",
        f"Deterministic facts (computed, trust these): move={move}; cap tier={r.cap_tier}; "
        f"dilution keywords detected={r.dilution_keywords or 'none'}",
    ]
    return "\n".join(lines)


def _inline_schema() -> dict:
    """Pydantic emits $defs/$ref; many local servers need a flat schema."""
    schema = CatalystAnalysis.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    flat = resolve(schema)
    flat["required"] = list(flat["properties"].keys())
    flat["additionalProperties"] = False
    return flat


ANALYSIS_SCHEMA = _inline_schema()


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


class LLMProvider(ABC):
    name = "base"
    model = ""

    @abstractmethod
    def _call(self, system: str, user: str) -> dict:
        """Return raw dict matching ANALYSIS_SCHEMA."""

    def analyze(self, system: str, user: str, max_retries: int) -> CatalystAnalysis:
        last_err: Exception | None = None
        prompt = user
        for attempt in range(1, max_retries + 1):
            try:
                return CatalystAnalysis.model_validate(self._call(system, prompt))
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                last_err = e
                log.warning("%s: invalid structured output (attempt %d): %s", self.name, attempt, e)
                # Feed the validation error back so the retry can self-correct (temp=0 would repeat otherwise).
                prompt = f"{user}\n\nYour previous response was rejected by the validator:\n{e}\nFix it and respond again."
                continue
            except Exception as e:  # network / rate limit / server error
                last_err = e
                log.warning("%s: call failed (attempt %d): %s", self.name, attempt, e)
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 20))  # exponential backoff
        raise RuntimeError(f"{self.name} failed after {max_retries} attempts: {last_err}")


class LMStudioProvider(LLMProvider):
    name = "lmstudio"

    def __init__(self, s: Settings):
        from openai import OpenAI
        self.client = OpenAI(base_url=s.lmstudio_base_url, api_key="lm-studio", timeout=s.ai_timeout, max_retries=0)
        self.model = s.lmstudio_model
        self.reasoning_effort = s.lmstudio_reasoning_effort

    def _call(self, system: str, user: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            extra_body={"reasoning_effort": self.reasoning_effort},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "catalyst_analysis", "strict": True, "schema": ANALYSIS_SCHEMA}},
        )
        return _extract_json(resp.choices[0].message.content or "")


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, s: Settings):
        import anthropic
        self.client = anthropic.Anthropic(timeout=s.ai_timeout, max_retries=0)  # key from ANTHROPIC_API_KEY
        self.model = s.anthropic_model

    def _call(self, system: str, user: str) -> dict:
        # Forced tool call = schema-constrained structured output.
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[{"name": "record_catalyst_analysis",
                    "description": "Record the structured catalyst analysis.",
                    "input_schema": ANALYSIS_SCHEMA}],
            tool_choice={"type": "tool", "name": "record_catalyst_analysis"},
        )
        for block in resp.content:
            if block.type == "tool_use":
                return block.input
        raise ValueError("No tool_use block in Anthropic response")


class NvidiaNimProvider(LLMProvider):
    """NVIDIA NIM hosted catalog (build.nvidia.com), OpenAI-compatible. Key: NVIDIA_API_KEY (nvapi-...).
    Structured output = schema-in-prompt + the shared validator/retry loop. Tested 2026-09 on the hosted
    gemma-4-31b-it: nvext.guided_json is rejected (400) and response_format=json_schema degenerates into
    whitespace until max_tokens, so neither is used."""
    name = "nim"

    def __init__(self, s: Settings):
        import os
        from openai import OpenAI
        key = os.getenv("NVIDIA_API_KEY", "").strip()
        if not key:
            raise RuntimeError("NVIDIA_API_KEY is not set (add it to .env locally or Streamlit secrets when hosted).")
        self.client = OpenAI(base_url=s.nim_base_url, api_key=key, timeout=s.ai_timeout, max_retries=0)
        self.model = s.nim_model

    def _call(self, system: str, user: str) -> dict:
        schema_hint = ("\n\nRespond with ONLY a JSON object matching this JSON Schema (no prose, no code fences):\n"
                       + json.dumps(ANALYSIS_SCHEMA))
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=2000,
            messages=[{"role": "system", "content": system + schema_hint}, {"role": "user", "content": user}])
        return _extract_json(resp.choices[0].message.content or "")


PROVIDERS = {"lmstudio": LMStudioProvider, "anthropic": AnthropicProvider, "nim": NvidiaNimProvider}


def get_provider(s: Settings | None = None) -> LLMProvider:
    s = s or get_settings()
    try:
        return PROVIDERS[s.ai_provider](s)
    except KeyError:
        raise ValueError(f"Unknown AI_PROVIDER={s.ai_provider!r}; options: {list(PROVIDERS)}") from None


def classify_catalyst(event: NewsEvent, rule_result: RuleResult, provider: LLMProvider | None = None,
                      settings: Settings | None = None) -> CatalystAnalysis:
    """Qualitative catalyst analysis. Deterministic guardrails are applied separately (rules.apply_guardrails)."""
    s = settings or get_settings()
    provider = provider or get_provider(s)
    return provider.analyze(SYSTEM_PROMPT, build_user_prompt(event, rule_result), s.ai_max_retries)


__all__ = ["classify_catalyst", "get_provider", "Assessment"]
