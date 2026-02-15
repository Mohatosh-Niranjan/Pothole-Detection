import os
import sqlite3
from dataclasses import asdict
from typing import Iterable, List, Optional, Tuple

from .detection import DetectionRecord


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "realtime.db")


def ensure_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                confidence REAL,
                area_px REAL,
                risk_level TEXT,
                latitude REAL,
                longitude REAL,
                detection_time TEXT,
                device_id TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS gps (
                device_id TEXT PRIMARY KEY,
                latitude REAL,
                longitude REAL,
                updated_at TEXT
            )
            """
        )
        try:
            c.execute("ALTER TABLE detections ADD COLUMN device_id TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()


def insert_detection(record: DetectionRecord) -> None:
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO detections (
                frame_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence,
                area_px, risk_level, latitude, longitude, detection_time, device_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.frame_id,
                record.bbox_x1,
                record.bbox_y1,
                record.bbox_x2,
                record.bbox_y2,
                record.confidence,
                record.area_px,
                record.risk_level,
                record.latitude,
                record.longitude,
                record.detection_time,
                record.device_id,
            ),
        )
        conn.commit()


def upsert_gps(device_id: str, latitude: float, longitude: float, updated_at: str) -> None:
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO gps (device_id, latitude, longitude, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
              latitude=excluded.latitude,
              longitude=excluded.longitude,
              updated_at=excluded.updated_at
            """,
            (device_id, latitude, longitude, updated_at),
        )
        conn.commit()


def get_recent_detections(limit: int = 1000):
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        return [dict(row) for row in rows][::-1]


def get_latest_gps(device_id: str):
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT latitude, longitude, updated_at FROM gps WHERE device_id = ?",
            (device_id,),
        )
        row = c.fetchone()
        if row:
            return float(row["latitude"]), float(row["longitude"]), row["updated_at"]
        return None


