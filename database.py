import sqlite3
import json
from datetime import datetime

DB_PATH = "scans.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            mode TEXT NOT NULL,
            product_count INTEGER DEFAULT 0,
            avg_freshness_score REAL DEFAULT 0,
            flagged_count INTEGER DEFAULT 0,
            shelf_gaps INTEGER DEFAULT 0,
            detections_json TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_scan(mode, product_count, avg_freshness, flagged_count, shelf_gaps, detections):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO scans (timestamp, mode, product_count, avg_freshness_score, flagged_count, shelf_gaps, detections_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        mode,
        product_count,
        round(avg_freshness, 2),
        flagged_count,
        shelf_gaps,
        json.dumps(detections)
    ))
    conn.commit()
    conn.close()


def get_history(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, timestamp, mode, product_count, avg_freshness_score, flagged_count, shelf_gaps
        FROM scans ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": r[0], "timestamp": r[1], "mode": r[2],
            "product_count": r[3], "avg_freshness": r[4],
            "flagged_count": r[5], "shelf_gaps": r[6]
        }
        for r in rows
    ]


def get_trend_data(limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT timestamp, avg_freshness_score, product_count, flagged_count
        FROM scans WHERE mode='fruit' ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))
