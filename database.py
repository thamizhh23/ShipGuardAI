"""
SHIPGUARD AI - Local Inspection Database
Lightweight SQLite persistence for ship inspection records, defect analytics, and history tracking.
"""

import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional


DEFAULT_DB_PATH = "shipguard.db"


class InspectionDatabase:
    """
    SQLite database manager for persistent inspection records.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Creates table schema if it does not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    defect_count INTEGER NOT NULL,
                    defect_types_json TEXT NOT NULL,
                    highest_confidence REAL NOT NULL,
                    severity TEXT NOT NULL,
                    risk_score INTEGER NOT NULL,
                    recommended_action TEXT NOT NULL,
                    total_area_pct REAL NOT NULL,
                    notes TEXT,
                    raw_data_json TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_inspection(
        self,
        image_name: str,
        defect_count: int,
        defect_types: Dict[str, int],
        highest_confidence: float,
        severity: str,
        risk_score: int,
        recommended_action: str,
        total_area_pct: float,
        raw_data: Dict[str, Any],
        notes: str = ""
    ) -> int:
        """
        Inserts a completed inspection into the database.
        Returns the new row ID.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inspections (
                    timestamp, image_name, defect_count, defect_types_json,
                    highest_confidence, severity, risk_score, recommended_action,
                    total_area_pct, notes, raw_data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                image_name,
                defect_count,
                json.dumps(defect_types),
                round(highest_confidence, 4),
                severity,
                risk_score,
                recommended_action,
                round(total_area_pct, 2),
                notes,
                json.dumps(raw_data)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all_inspections(self) -> List[Dict[str, Any]]:
        """Retrieves all inspection records ordered by timestamp descending."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM inspections ORDER BY id DESC")
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "image_name": row["image_name"],
                    "defect_count": row["defect_count"],
                    "defect_types": json.loads(row["defect_types_json"]),
                    "highest_confidence": row["highest_confidence"],
                    "severity": row["severity"],
                    "risk_score": row["risk_score"],
                    "recommended_action": row["recommended_action"],
                    "total_area_pct": row["total_area_pct"],
                    "notes": row["notes"]
                })
            return results

    def get_inspection_by_id(self, inspection_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves full details including raw detection data for an inspection."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "image_name": row["image_name"],
                "defect_count": row["defect_count"],
                "defect_types": json.loads(row["defect_types_json"]),
                "highest_confidence": row["highest_confidence"],
                "severity": row["severity"],
                "risk_score": row["risk_score"],
                "recommended_action": row["recommended_action"],
                "total_area_pct": row["total_area_pct"],
                "notes": row["notes"],
                "raw_data": json.loads(row["raw_data_json"])
            }

    def delete_inspection(self, inspection_id: int) -> bool:
        """Deletes an inspection record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM inspections WHERE id = ?", (inspection_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_statistics(self) -> Dict[str, Any]:
        """Returns overall registry summary statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(risk_score), AVG(defect_count) FROM inspections")
            total, avg_risk, avg_defects = cursor.fetchone()
            if not total:
                return {"total_inspections": 0, "avg_risk": 0.0, "avg_defects": 0.0, "severity_distribution": {}}

            cursor.execute("SELECT severity, COUNT(*) FROM inspections GROUP BY severity")
            sev_dist = dict(cursor.fetchall())

            return {
                "total_inspections": total,
                "avg_risk": round(avg_risk or 0.0, 1),
                "avg_defects": round(avg_defects or 0.0, 1),
                "severity_distribution": sev_dist
            }
