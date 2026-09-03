"""Engine/connection helpers for the synthetic transaction database."""

from pathlib import Path

from sqlalchemy import Engine, create_engine

from finance_qna.data.schema import metadata


def get_engine(db_path: Path) -> Engine:
    """Create an engine for the SQLite database at `db_path`, creating the schema if needed."""
    engine = create_engine(f"sqlite:///{db_path}")
    metadata.create_all(engine)
    return engine
