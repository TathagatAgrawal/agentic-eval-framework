"""Shared pytest fixtures: a seeded fixture database reused across the tool tests."""

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from finance_qna.data.db import get_engine
from finance_qna.data.generate import generate

FIXTURE_SEED = 42


@pytest.fixture(scope="session")
def fixture_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the seeded synthetic dataset once per test session and return its path."""
    db_dir = tmp_path_factory.mktemp("fixture_db")
    db_path = db_dir / "fixture.db"
    generate(seed=FIXTURE_SEED, db_path=db_path)
    return db_path


@pytest.fixture(scope="session")
def ground_truth(fixture_db_path: Path) -> dict[str, Any]:
    """Load the ground_truth.json sidecar written alongside the fixture database."""
    gt_path = fixture_db_path.parent / "ground_truth.json"
    return dict(json.loads(gt_path.read_text()))


@pytest.fixture(scope="session")
def fixture_engine(fixture_db_path: Path) -> Engine:
    """Return a SQLAlchemy engine bound to the seeded fixture database."""
    return get_engine(fixture_db_path)
