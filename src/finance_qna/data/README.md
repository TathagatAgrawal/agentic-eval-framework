# `data/`

The deterministic, LLM-free foundation everything else depends on: the SQLite schema, an engine factory, and the synthetic transaction generator.

## Files

- **`schema.py`** — SQLAlchemy Core table definitions: `accounts`, `categories` (self-referencing `parent_category_id`), `merchants` (each with a `default_category_id`), and `transactions` (amounts as `Numeric(12, 2)`, never `float`, plus an `is_subscription` flag used by the trend tools). No ORM models — every query elsewhere in the project builds parameterized `select()` statements directly against these tables.
- **`db.py`** — `get_engine(db_path)`, the one place a SQLAlchemy `Engine` is constructed.
- **`generate.py`** — `generate(seed: int, db_path: Path) -> None`, the deterministic synthetic data generator (`faker.Faker(seed)` + `random.Random(seed)`). Produces 18 months of transactions (July 2024 – December 2025) across 11 categories, 22 merchants, and 2 accounts, and writes `data/ground_truth.json` alongside the database with the same run's computed aggregates — so a test-case label can never silently drift from the data it's meant to check, since both come from the same generation run.

## Injected fixtures

Two anomalies are deliberately generated so the trend/anomaly-detection tools (`tools/trend_tools.py`) have a known, labeled target to find:

- A recurring subscription (StreamFlix) whose price increases from $13.99 to $17.99 in June 2025.
- A spending spike in the Shopping category in November 2025 ($1,402.09, roughly double a typical month).

Both are recorded in `data/ground_truth.json` and are what `eval/testset/single_turn.yaml`'s `trend_001`/`trend_002` cases check for.

## Usage

Not called directly in normal operation — `finance-qna data generate` (see `cli/`) is the entry point. Called directly only in tests, against a temporary database, to assert row counts and that the injected events match `ground_truth.json`.
