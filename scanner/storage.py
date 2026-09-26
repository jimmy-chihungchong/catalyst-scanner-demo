"""SQLite persistence: seen news (dedup), cached analyses, sent alerts (duplicate-alert prevention)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .rules import is_near_duplicate

SCHEMA = """
CREATE TABLE IF NOT EXISTS news_events (
    fingerprint   TEXT PRIMARY KEY,
    event_id      TEXT, ticker TEXT, headline TEXT, publisher TEXT, url TEXT,
    published_at  TEXT, first_seen_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_news_ticker ON news_events(ticker);
CREATE TABLE IF NOT EXISTS news_sources (          -- every provider that carried the same story
    fingerprint TEXT, publisher TEXT, url TEXT, seen_at TEXT,
    PRIMARY KEY (fingerprint, publisher)
);
CREATE TABLE IF NOT EXISTS analyses (
    fingerprint TEXT, provider TEXT, model TEXT, analysis_json TEXT, created_at TEXT,
    PRIMARY KEY (fingerprint, provider, model)
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT, fingerprint TEXT, assessment TEXT, move_pct REAL, alert_text TEXT, sent_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_alerts_ticker ON alerts(ticker, fingerprint);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path | str):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)  # Streamlit reruns on worker threads
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    # --- news dedup ---
    def find_duplicate(self, ticker: str, fingerprint: str, headline: str, threshold: float) -> str | None:
        """Exact fingerprint match, else fuzzy headline match for the same ticker (cross-wire copies)."""
        row = self.conn.execute("SELECT fingerprint FROM news_events WHERE fingerprint=?", (fingerprint,)).fetchone()
        if row:
            return row["fingerprint"]
        for r in self.conn.execute("SELECT fingerprint, headline FROM news_events WHERE ticker=? "
                                   "ORDER BY first_seen_at DESC LIMIT 200", (ticker,)):
            if is_near_duplicate(headline, r["headline"], threshold):
                return r["fingerprint"]
        return None

    def save_event(self, fp: str, ev) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO news_events VALUES (?,?,?,?,?,?,?,?)",
                (fp, ev.event_id, ev.ticker, ev.headline, ev.source_publisher, ev.source_url,
                 ev.published_at.isoformat() if ev.published_at else None, _now()))
            self.add_source(fp, ev.source_publisher, ev.source_url)

    def add_source(self, fp: str, publisher: str, url: str | None) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO news_sources VALUES (?,?,?,?)", (fp, publisher, url, _now()))

    def sources_for(self, fp: str) -> list[str]:
        return [r["publisher"] for r in self.conn.execute(
            "SELECT publisher FROM news_sources WHERE fingerprint=? ORDER BY rowid", (fp,))]  # first-seen wire first

    # --- analysis cache (same story is never sent to the LLM twice per provider/model) ---
    def get_analysis(self, fp: str, provider: str, model: str) -> dict | None:
        r = self.conn.execute("SELECT analysis_json FROM analyses WHERE fingerprint=? AND provider=? AND model=?",
                              (fp, provider, model)).fetchone()
        return json.loads(r["analysis_json"]) if r else None

    def save_analysis(self, fp: str, provider: str, model: str, analysis: dict) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?)",
                              (fp, provider, model, json.dumps(analysis), _now()))

    def clear_provider(self, provider: str, model: str) -> None:
        """Scoped reset for re-runs: drops one provider/model's analyses + sent alerts, keeps everything else."""
        with self.conn:
            self.conn.execute("DELETE FROM analyses WHERE provider=? AND model=?", (provider, model))
            self.conn.execute("DELETE FROM alerts")

    # --- alert dedup ---
    def should_alert(self, ticker: str, fp: str, assessment: str, move_pct: float | None,
                     move_change_pts: float = 10.0) -> tuple[bool, str]:
        """Re-alert only on meaningful change: new story, classification change, or big further move."""
        last = self.conn.execute("SELECT assessment, move_pct FROM alerts WHERE ticker=? AND fingerprint=? "
                                 "ORDER BY id DESC LIMIT 1", (ticker, fp)).fetchone()
        if last is None:
            return True, "new catalyst"
        if last["assessment"] != assessment:
            return True, f"classification changed {last['assessment']} -> {assessment}"
        if move_pct is not None and last["move_pct"] is not None and abs(move_pct - last["move_pct"]) >= move_change_pts:
            return True, f"move changed {last['move_pct']:+.1f}% -> {move_pct:+.1f}%"
        return False, "identical alert already sent (same story, same classification)"

    def record_alert(self, ticker: str, fp: str, assessment: str, move_pct: float | None, text: str) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO alerts (ticker, fingerprint, assessment, move_pct, alert_text, sent_at) "
                              "VALUES (?,?,?,?,?,?)", (ticker, fp, assessment, move_pct, text, _now()))
