"""SQLite evidence log for violations."""
import sqlite3, datetime, os

DB = "violations.db"
SCHEMA = """CREATE TABLE IF NOT EXISTS violations(
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 ts TEXT, track_id INTEGER, reason TEXT, snapshot TEXT, conf REAL)"""

def conn():
    c = sqlite3.connect(DB)
    c.execute(SCHEMA)
    return c

def log_violation(track_id, reason, snapshot="", conf=0.0):
    c = conn()
    c.execute("INSERT INTO violations(ts,track_id,reason,snapshot,conf) VALUES(?,?,?,?,?)",
              (datetime.datetime.now().isoformat(timespec="seconds"), track_id, reason, snapshot, conf))
    c.commit(); c.close()

def recent(limit=100):
    c = conn()
    rows = c.execute("SELECT * FROM violations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    c.close()
    return rows
