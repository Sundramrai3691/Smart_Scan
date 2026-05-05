"""
Run this once to populate the database with realistic demo scan data.
Usage: python seed_demo.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from database import init_db, log_scan
import random
from datetime import datetime, timedelta

PRODUCT_LABELS = ["Amul Butter", "Maggi Noodles", "Parle G", "Lays Classic", "Britannia Bread", "Nestle KitKat", "Haldirams Bhujia", "Horlicks"]
FRUIT_LABELS = ["Apple", "Banana", "Tomato", "Mango", "Orange", "Grapes", "Watermelon"]


def seed_product_scans(n=8):
    base_time = datetime.now() - timedelta(days=7)
    for i in range(n):
        count = random.randint(4, 8)
        detections = [
            {
                "label": random.choice(PRODUCT_LABELS),
                "confidence": round(random.uniform(0.72, 0.97), 2),
                "freshness_score": None,
                "bbox": [random.randint(10, 100), random.randint(10, 100), random.randint(150, 300), random.randint(150, 300)]
            }
            for _ in range(count)
        ]
        ts = (base_time + timedelta(hours=i * 6)).isoformat()
        # Patch timestamp directly since log_scan uses datetime.now()
        import sqlite3, json
        conn = sqlite3.connect("scans.db")
        conn.execute("""
            INSERT INTO scans (timestamp, mode, product_count, avg_freshness_score, flagged_count, shelf_gaps, detections_json)
            VALUES (?,?,?,?,?,?,?)
        """, (ts, "product", count, 0.0, random.randint(0, 2), max(0, 8 - count), json.dumps(detections)))
        conn.commit()
        conn.close()


def seed_fruit_scans(n=10):
    base_time = datetime.now() - timedelta(days=5)
    for i in range(n):
        count = random.randint(3, 7)
        scores = [random.randint(30, 95) for _ in range(count)]
        avg = round(sum(scores) / len(scores), 1)
        flagged = sum(1 for s in scores if s < 60)
        detections = [
            {
                "label": random.choice(FRUIT_LABELS),
                "confidence": round(random.uniform(0.70, 0.95), 2),
                "freshness_score": scores[j],
                "bbox": [random.randint(10, 100), random.randint(10, 100), random.randint(150, 300), random.randint(150, 300)]
            }
            for j in range(count)
        ]
        ts = (base_time + timedelta(hours=i * 8)).isoformat()
        import sqlite3, json
        conn = sqlite3.connect("scans.db")
        conn.execute("""
            INSERT INTO scans (timestamp, mode, product_count, avg_freshness_score, flagged_count, shelf_gaps, detections_json)
            VALUES (?,?,?,?,?,?,?)
        """, (ts, "fruit", count, avg, flagged, max(0, 8 - count), json.dumps(detections)))
        conn.commit()
        conn.close()


if __name__ == "__main__":
    init_db()
    seed_product_scans(8)
    seed_fruit_scans(10)
    print("Seeded 18 demo scans into scans.db")
    print("Now run: python app.py — and open the Dashboard view.")
