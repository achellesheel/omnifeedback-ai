"""SQLite star-schema data warehouse with ACID batch ingestion."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "warehouse" / "omnifeedback.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS Dim_User (
    user_id INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS Dim_Channel (
    channel_id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS Dim_Product (
    aspect_id INTEGER PRIMARY KEY AUTOINCREMENT,
    aspect_category TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS Fact_Feedback (
    feedback_id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    aspect_id INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    cleaned_text TEXT NOT NULL,
    urgency_score REAL NOT NULL CHECK (urgency_score >= 0.0 AND urgency_score <= 1.0),
    label INTEGER,
    predicted_cluster INTEGER,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES Dim_User(user_id),
    FOREIGN KEY (channel_id) REFERENCES Dim_Channel(channel_id),
    FOREIGN KEY (aspect_id) REFERENCES Dim_Product(aspect_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_timestamp ON Fact_Feedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_fact_channel ON Fact_Feedback(channel_id);
"""


class WarehouseManager:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self):
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def _get_or_create_dim(self, conn, table: str, name_col: str, value: str) -> int:
        cur = conn.execute(f"SELECT rowid FROM {table} WHERE {name_col} = ?", (value,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur = conn.execute(f"INSERT INTO {table} ({name_col}) VALUES (?)", (value,))
        return cur.lastrowid

    def insert_feedback_batch(self, df: pd.DataFrame) -> int:
        """Insert a batch of cleaned feedback records inside one ACID transaction.

        Expects columns: feedback_id, user_id, raw_text, cleaned_text, channel,
        aspect_category, urgency_score, label, timestamp. Rolls back entirely on
        any constraint violation so partial/corrupt batches never land.
        """
        required = {
            "feedback_id", "user_id", "raw_text", "cleaned_text",
            "channel", "aspect_category", "urgency_score", "timestamp",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        inserted = 0
        with self.get_connection() as conn:
            try:
                conn.execute("BEGIN")
                for _, row in df.iterrows():
                    conn.execute(
                        "INSERT OR IGNORE INTO Dim_User (user_id) VALUES (?)", (int(row["user_id"]),)
                    )
                    channel_id = self._get_or_create_dim(conn, "Dim_Channel", "channel_name", row["channel"])
                    aspect_id = self._get_or_create_dim(conn, "Dim_Product", "aspect_category", row["aspect_category"])
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO Fact_Feedback
                        (feedback_id, user_id, channel_id, aspect_id, raw_text, cleaned_text,
                         urgency_score, label, predicted_cluster, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(row["feedback_id"]), int(row["user_id"]), channel_id, aspect_id,
                            row["raw_text"], row["cleaned_text"], float(row["urgency_score"]),
                            int(row.get("label", 0)) if pd.notna(row.get("label", None)) else None,
                            int(row["predicted_cluster"]) if "predicted_cluster" in row and pd.notna(row["predicted_cluster"]) else None,
                            str(row["timestamp"]),
                        ),
                    )
                    inserted += 1
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return inserted

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self.get_connection() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def fetch_all_feedback(self) -> pd.DataFrame:
        return self.query(
            """
            SELECT f.feedback_id, f.user_id, c.channel_name AS channel,
                   p.aspect_category, f.raw_text, f.cleaned_text,
                   f.urgency_score, f.label, f.predicted_cluster, f.timestamp
            FROM Fact_Feedback f
            JOIN Dim_Channel c ON f.channel_id = c.channel_id
            JOIN Dim_Product p ON f.aspect_id = p.aspect_id
            ORDER BY f.timestamp
            """
        )
