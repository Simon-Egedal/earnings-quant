from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

import pandas as pd


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def read_parquet_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@contextmanager
def metadata_db(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS artifacts (
        key TEXT PRIMARY KEY, path TEXT NOT NULL, updated_at TEXT NOT NULL,
        row_count INTEGER, metadata_json TEXT NOT NULL DEFAULT '{}')"""
    )
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def record_artifact(
    db_path: Path, key: str, path: Path, row_count: int | None = None, **metadata: object
) -> None:
    with metadata_db(db_path) as connection:
        connection.execute(
            """INSERT INTO artifacts(key,path,updated_at,row_count,metadata_json)
            VALUES(?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET path=excluded.path,
            updated_at=excluded.updated_at,row_count=excluded.row_count,
            metadata_json=excluded.metadata_json""",
            (key, str(path), datetime.now(UTC).isoformat(), row_count, json.dumps(metadata, default=str)),
        )

