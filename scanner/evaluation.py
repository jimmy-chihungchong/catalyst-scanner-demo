"""PASS/FAIL grading against each fixture's free-text expected_classification.

The expected text is parsed into explicit, inspectable checks so it's clear WHY a case passed/failed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Assessment, CatalystAnalysis

# Expected text must START with one of these to assert an exact assessment.
LABEL_PATTERNS = [
    ("STRONG CATALYST", Assessment.STRONG_CATALYST),
    ("POTENTIAL PUFFY", Assessment.PUFFY_WEAK),
    ("PUFFY", Assessment.PUFFY_WEAK),
    ("MIXED", Assessment.MIXED_UNCLEAR),
    ("NO FRESH CATALYST", Assessment.NO_FRESH_CATALYST),
]


@dataclass
class Check:
    description: str
    passed: bool
    actual: str


@dataclass
class EvalResult:
    checks: list[Check] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.checks:
            return "UNGRADED"
        return "PASS" if all(c.passed for c in self.checks) else "FAIL"


def evaluate(expected: str, a: CatalystAnalysis) -> EvalResult:
    up = expected.strip().upper()
    res = EvalResult()
    for pattern, label in LABEL_PATTERNS:
        if up.startswith(pattern):
            res.checks.append(Check(f"assessment == {label.value}", a.assessment == label, a.assessment.value))
            break
    if "RECENT DILUTION: YES" in up:
        res.checks.append(Check("recent dilution == Yes", a.dilution_risk == "yes", a.dilution_risk))
    if "NOT ITSELF A BULLISH" in up or "NOT A BULLISH" in up:
        res.checks.append(Check("assessment != STRONG_CATALYST (not bullish)",
                                a.assessment != Assessment.STRONG_CATALYST, a.assessment.value))
    return res
