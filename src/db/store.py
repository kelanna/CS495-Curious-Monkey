"""SQLite-backed results store for the prompt injection harness.

Records are de-duplicated by (attack_id, model, domain, rep, language_code,
result_file) so re-importing the same JSON file is always idempotent.
"""
import glob
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = "results/results.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    source                      TEXT    NOT NULL,
    phase                       TEXT    NOT NULL,
    attack_id                   TEXT,
    attack_name                 TEXT,
    base_attack_id              TEXT,
    model                       TEXT,
    domain                      TEXT,
    language_code               TEXT,
    rep                         INTEGER DEFAULT 0,
    score                       TEXT,
    success                     INTEGER DEFAULT 0,
    payload                     TEXT,
    response                    TEXT,
    response_translated         TEXT,
    response_language_detected  TEXT,
    translation_used            INTEGER DEFAULT 0,
    note                        TEXT,
    result_file                 TEXT,
    imported_at                 TEXT    DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_dedup
    ON runs (
        COALESCE(attack_id,    ''),
        COALESCE(model,        ''),
        COALESCE(domain,       ''),
        COALESCE(rep,          0),
        COALESCE(language_code,''),
        COALESCE(result_file,  '')
    );
"""


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        con.executescript(_SCHEMA)


# ── Source / phase inference ──────────────────────────────────────────────────

def _infer_source(path: str) -> str:
    for key in ("formal_p2a", "formal_v2", "scratch"):
        if key in path:
            return key
    return "formal"


def _infer_phase(source: str, attack_id: str) -> str:
    if source == "formal_p2a" or attack_id.startswith("p2b_"):
        return "p2b"
    if attack_id.startswith("p2_"):
        return "p2a"
    if attack_id.startswith("attack"):
        return "p1"
    return "unknown"


# ── Record normalisation ──────────────────────────────────────────────────────

def _record_to_row(r: dict, source: str, result_file: str) -> dict:
    attack_id = r.get("attack_id", "")
    is_p2b    = source == "formal_p2a" or attack_id.startswith("p2b_")
    lang      = r.get("language_code", r.get("language", "")) or None
    return {
        "source":                     source,
        "phase":                      _infer_phase(source, attack_id),
        "attack_id":                  attack_id,
        "attack_name":                r.get("attack_name", ""),
        "base_attack_id":             r.get("base_attack_id", ""),
        "model":                      r.get("model", ""),
        "domain":                     r.get("domain", ""),
        "language_code":              lang,
        "rep":                        r.get("rep", 0),
        "score":                      r.get("score", r.get("outcome", "")),
        "success":                    int(bool(r.get("success", False))),
        "payload":                    (r.get("attack_prompt") or r.get("payload", "")) if is_p2b
                                      else r.get("payload", ""),
        "response":                   (r.get("response_original") or r.get("response", "")) if is_p2b
                                      else r.get("response", ""),
        "response_translated":        r.get("response_translated"),
        "response_language_detected": r.get("response_language_detected", ""),
        "translation_used":           int(bool(r.get("translation_used", False))),
        "note":                       r.get("note", ""),
        "result_file":                result_file,
    }


_INSERT_SQL = """
    INSERT OR IGNORE INTO runs
        (source, phase, attack_id, attack_name, base_attack_id,
         model, domain, language_code, rep, score, success,
         payload, response, response_translated,
         response_language_detected, translation_used, note, result_file)
    VALUES
        (:source, :phase, :attack_id, :attack_name, :base_attack_id,
         :model, :domain, :language_code, :rep, :score, :success,
         :payload, :response, :response_translated,
         :response_language_detected, :translation_used, :note, :result_file)
"""


# ── Public API ────────────────────────────────────────────────────────────────

def import_json(path: str) -> int:
    """Import all records from one JSON result file. Returns new rows inserted."""
    source = _infer_source(path)
    try:
        with open(path) as f:
            records = json.load(f)
    except Exception:
        return 0

    inserted = 0
    with _conn() as con:
        for r in records:
            if not isinstance(r, dict):
                continue
            row = _record_to_row(r, source, path)
            try:
                con.execute(_INSERT_SQL, row)
                inserted += con.execute("SELECT changes()").fetchone()[0]
            except Exception:
                continue
    return inserted


def import_all_json() -> int:
    """Scan the full results/ tree and import every JSON file. Returns new rows."""
    init_db()
    total = 0
    for path in sorted(glob.glob("results/**/*.json", recursive=True)):
        total += import_json(path)
    return total


def insert_runs(records: list[dict], source: str, result_file: str = "") -> None:
    """Insert a batch of harness run records. Silently skips duplicates."""
    init_db()
    with _conn() as con:
        for r in records:
            if not isinstance(r, dict):
                continue
            row = _record_to_row(r, source, result_file)
            try:
                con.execute(_INSERT_SQL, row)
            except Exception:
                continue


def db_stats() -> dict:
    """Return record counts and last-import time for the dashboard."""
    try:
        with _conn() as con:
            total = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            by_phase = {
                row["phase"]: row["cnt"]
                for row in con.execute(
                    "SELECT phase, COUNT(*) AS cnt FROM runs GROUP BY phase"
                ).fetchall()
            }
            by_source = {
                row["source"]: row["cnt"]
                for row in con.execute(
                    "SELECT source, COUNT(*) AS cnt FROM runs GROUP BY source ORDER BY cnt DESC"
                ).fetchall()
            }
            last_import = con.execute("SELECT MAX(imported_at) FROM runs").fetchone()[0]
            return {
                "total":       total,
                "by_phase":    by_phase,
                "by_source":   by_source,
                "last_import": last_import,
            }
    except Exception:
        return {"total": 0, "by_phase": {}, "by_source": {}, "last_import": None}
