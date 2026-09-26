from datetime import date

import pandas as pd

from scanner.prices import parse_event_date, verify_move


def test_parse_event_date_fixture_formats():
    assert parse_event_date("Jumped 17.3% on August 12, 2026") == (date(2026, 8, 12), "day")
    assert parse_event_date("Offering closed February 9, 2026; gross proceeds") == (date(2026, 2, 9), "day")
    assert parse_event_date("Stock up over 660% since end of July 2026") == (date(2026, 7, 31), "approx (end of month)")
    assert parse_event_date(None) == (None, "none")


def _df(closes: dict) -> pd.DataFrame:
    idx = pd.to_datetime(list(closes))
    return pd.DataFrame({"Close": list(closes.values())}, index=idx)


def test_verify_move_single_session():
    df = _df({"2026-08-10": 100.0, "2026-08-11": 100.0, "2026-08-12": 117.0, "2026-08-13": 110.0})
    m = verify_move(df, date(2026, 8, 12), 17.3, "single-session")
    assert m.verified_pct == 17.0 and m.status == "match"
    assert verify_move(df, date(2026, 8, 12), 40.0, "single-session").status == "mismatch"


def test_verify_move_weekend_event_uses_next_session():
    df = _df({"2026-08-14": 100.0, "2026-08-17": 110.0})  # Fri, Mon
    m = verify_move(df, date(2026, 8, 15), 10.0, "single-session")
    assert m.trading_date == date(2026, 8, 17) and m.status == "match"


def test_verify_move_multi_period_and_unverifiable():
    df = _df({"2026-07-30": 10.0, "2026-08-03": 12.0, "2026-09-25": 30.0})
    assert verify_move(df, date(2026, 7, 31), 200.0, "multi-period").verified_pct == 200.0
    assert verify_move(df, None, 5.0, "single-session").status == "unverifiable"
    assert verify_move(df, date(2026, 8, 3), None, "single-session").status == "unverifiable"
