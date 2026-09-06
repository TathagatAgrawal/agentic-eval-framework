"""The regression store: persists `RunRecord`s to disk so eval runs accumulate
into a history instead of each being a disconnected, one-off number.

Flat JSON files under a `runs_dir` (default `eval/runs/`), git-tracked -- each
record is a few KB, and the point (per design/eval-harness-plan.md §7) is that
run-to-run history stays directly diffable in a PR, not a database to query.
"""

import json
from pathlib import Path

from eval.results import RunRecord

DEFAULT_RUNS_DIR = Path("eval/runs")


def save_run(record: RunRecord, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Write `record` to `runs_dir/<run_id>.json`, creating `runs_dir` if needed."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record.run_id}.json"
    path.write_text(record.model_dump_json(indent=2))
    return path


def load_all_runs(runs_dir: Path = DEFAULT_RUNS_DIR) -> list[RunRecord]:
    """Load every saved `RunRecord` from `runs_dir`, oldest first.

    Returns an empty list if `runs_dir` doesn't exist yet (no runs saved).
    """
    if not runs_dir.exists():
        return []
    records = [
        RunRecord.model_validate(json.loads(path.read_text()))
        for path in sorted(runs_dir.glob("*.json"))
    ]
    return sorted(records, key=lambda record: record.timestamp)
