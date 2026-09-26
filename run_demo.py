"""CLI demo: replay fixture test cases through the full pipeline and grade each against expectations.

    python run_demo.py                       # uses AI_PROVIDER from .env (default: lmstudio)
    python run_demo.py --provider anthropic  # quality comparison
    python run_demo.py --provider nim        # NVIDIA NIM hosted (needs NVIDIA_API_KEY)
    python run_demo.py --fresh               # re-analyze: clears THIS provider's cache + alerts only
    python run_demo.py --case case_03        # run a single case (prefix match)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from dataclasses import replace
from datetime import datetime, timezone

from scanner import evaluation
from scanner.config import get_settings
from scanner.ingestion import FixtureNewsSource
from scanner.pipeline import Pipeline
from scanner.storage import Store

W = 88


def rule(ch="="):
    print(ch * W)


def wrap(text: str, indent="    ") -> str:
    return textwrap.fill(text, W, initial_indent=indent, subsequent_indent=indent)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["lmstudio", "anthropic", "nim"])
    ap.add_argument("--fresh", action="store_true", help="re-analyze: clear this provider's cached analyses and alerts")
    ap.add_argument("--case", help="only run cases whose id starts with this")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    s = get_settings()
    if args.provider:
        s = replace(s, ai_provider=args.provider)
    source = FixtureNewsSource(s.fixture_path)
    cases = [c for c in source.load_cases() if not args.case or c.case_id.startswith(args.case)]
    store = Store(s.db_path)
    pipe = Pipeline(s, store)
    if args.fresh:  # scoped: other providers' cached analyses survive
        store.clear_provider(pipe.provider_name, pipe.model)

    rule()
    print(f"CATALYST SCANNER DEMO  |  provider={pipe.provider_name}  model={pipe.model}")
    print(f"source=fixture ({s.fixture_path.name})  db={s.db_path.name}  run={datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    print("Decision support only — this system never places trades.")
    rule()
    for sid, missing in source.skipped:
        print(f"SKIPPED {sid}: null fields {', '.join(missing)} (awaiting real sourced data)")

    summary = []
    for case in cases:
        res = pipe.process(case.event)
        print()
        rule()
        print(f"{case.case_id}   [{case.category}]")
        rule("-")
        if res.error:
            print(f"ERROR: {res.alert_reason}\n  {res.error}")
            summary.append((case.case_id, case.event.ticker, "ERROR", "-", res.latency_s))
            continue
        print(res.alert_text)
        print(f"\nAlert status: {res.alert_status} ({res.alert_reason})"
              + ("  [analysis from cache]" if res.cached else f"  [AI latency {res.latency_s:.1f}s]"))
        print("\nEvidence:")
        for e in res.analysis.evidence:
            print(wrap(f"- {e}"))
        print("Reasoning:")
        print(wrap(res.analysis.reasoning))
        a = res.analysis
        print(f"Detail: clinical/FDA={a.clinical_fda_significance}  promotional={a.promotional_or_vague_language}  "
              f"disproportionate={a.reaction_disproportionate}  confidence={a.confidence:.2f}")
        if a.risk_factors:
            print("Risk factors:")
            for rf in a.risk_factors:
                print(wrap(f"- {rf}"))
        if res.alert.notes:
            print("Notes:")
            for n in res.alert.notes:
                print(wrap(f"* {n}"))

        ev_res = evaluation.evaluate(case.expected_classification, res.analysis)
        print(f"\nExpected: {case.expected_classification}")
        for c in ev_res.checks:
            print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.description}  (actual: {c.actual})")
        print(f">>> RESULT: {ev_res.status}")
        summary.append((case.case_id, case.event.ticker, ev_res.status, a.assessment.value, res.latency_s))

    print()
    rule()
    print("SUMMARY")
    rule("-")
    for cid, t, st, assess, lat in summary:
        print(f"{st:<9}{t:<6}{assess:<20}{lat:>6.1f}s  {cid}")
    passed = sum(1 for x in summary if x[2] == "PASS")
    print(f"\n{passed}/{len(summary)} graded cases passed; {len(source.skipped)} cases skipped (null data).")
    rule()

    report = [dict(case=c, ticker=t, status=st, assessment=a, latency_s=round(l, 2)) for c, t, st, a, l in summary]
    (s.db_path.parent / f"last_run_{pipe.provider_name}.json").write_text(json.dumps(report, indent=2))
    store.close()
    return 0 if passed == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
