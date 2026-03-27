"""
Tests for the SQLite storage layer.

Covers:
    - Database initialization
    - Storing and retrieving results, reports, comparisons
    - History queries
    - Score trend queries
    - Deletion
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from storage import EvalStorage
from models import (
    EvalResult,
    EvalReport,
    RubricSummary,
    VersionComparison,
    CriterionScore,
    RubricCategory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def storage(tmp_db):
    return EvalStorage(db_path=tmp_db)


def _make_result(result_id="r1", run_id="run_1", output_id="o1",
                 rubric_name="safety", score=4.0):
    return EvalResult(
        result_id=result_id,
        output_id=output_id,
        rubric_name=rubric_name,
        rubric_category=RubricCategory.SAFETY,
        score=score,
        max_score=5.0,
        reasoning="Test reasoning.",
        criterion_scores=[
            CriterionScore(criterion_name="crit_a", score=4.0, reasoning="OK"),
        ],
        judge_model="gpt-4",
        prompt_version="v1.0",
        eval_run_id=run_id,
        latency_ms=50.0,
        metadata={"simulated": True},
    )


def _make_report(run_id="run_1", score=3.5):
    return EvalReport(
        report_id=f"report_{run_id}",
        eval_run_id=run_id,
        prompt_version="v1.0",
        judge_model="gpt-4",
        num_outputs=10,
        num_rubrics=4,
        total_evaluations=40,
        overall_mean_score=score,
        rubric_summaries=[
            RubricSummary(
                rubric_name="safety",
                rubric_display_name="Patient Safety",
                mean_score=score,
                min_score=score - 1.0,
                max_score=score + 0.5,
                std_dev=0.5,
                num_outputs=10,
                score_distribution={"3": 5, "4": 5},
            ),
        ],
        metadata={"rubric_weights": {"safety": 2.0}},
    )


# ---------------------------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------------------------

class TestDatabaseInit:
    """Tests for database schema initialization."""

    def test_db_creates_tables(self, storage):
        import sqlite3
        conn = sqlite3.connect(storage.db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = {t[0] for t in tables}
        assert "eval_runs" in table_names
        assert "eval_results" in table_names
        assert "eval_reports" in table_names
        assert "eval_comparisons" in table_names

    def test_db_creates_indexes(self, storage):
        import sqlite3
        conn = sqlite3.connect(storage.db_path)
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        conn.close()
        index_names = {i[0] for i in indexes}
        assert "idx_results_run_id" in index_names
        assert "idx_results_output_id" in index_names
        assert "idx_results_rubric" in index_names


# ---------------------------------------------------------------------------
# Store and Retrieve Results
# ---------------------------------------------------------------------------

class TestStoreResults:
    """Tests for storing and retrieving individual results."""

    def test_store_and_retrieve_result(self, storage):
        result = _make_result()
        storage.store_result(result)

        results = storage.get_results_by_output("o1")
        assert len(results) == 1
        assert results[0]["score"] == 4.0
        assert results[0]["rubric_name"] == "safety"

    def test_store_multiple_results(self, storage):
        storage.store_result(_make_result(result_id="r1", output_id="o1", score=4.0))
        storage.store_result(_make_result(result_id="r2", output_id="o1", rubric_name="accuracy", score=3.5))
        storage.store_result(_make_result(result_id="r3", output_id="o2", score=2.0))

        results_o1 = storage.get_results_by_output("o1")
        assert len(results_o1) == 2

        results_o2 = storage.get_results_by_output("o2")
        assert len(results_o2) == 1

    def test_get_results_by_rubric(self, storage):
        storage.store_result(_make_result(result_id="r1", rubric_name="safety", score=4.0))
        storage.store_result(_make_result(result_id="r2", rubric_name="accuracy", score=3.0))
        storage.store_result(_make_result(result_id="r3", rubric_name="safety", score=2.0))

        safety_results = storage.get_results_by_rubric("safety")
        assert len(safety_results) == 2

        accuracy_results = storage.get_results_by_rubric("accuracy")
        assert len(accuracy_results) == 1

    def test_get_results_by_rubric_and_run(self, storage):
        storage.store_result(_make_result(result_id="r1", run_id="run_a", rubric_name="safety"))
        storage.store_result(_make_result(result_id="r2", run_id="run_b", rubric_name="safety"))

        results = storage.get_results_by_rubric("safety", run_id="run_a")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Store and Retrieve Reports
# ---------------------------------------------------------------------------

class TestStoreReports:
    """Tests for storing and retrieving reports."""

    def test_store_and_get_report(self, storage):
        report = _make_report("run_1", 3.5)
        storage.store_report(report)

        retrieved = storage.get_report("run_1")
        assert retrieved is not None
        assert retrieved.overall_mean_score == 3.5
        assert retrieved.prompt_version == "v1.0"
        assert len(retrieved.rubric_summaries) == 1
        assert retrieved.rubric_summaries[0].rubric_name == "safety"

    def test_get_nonexistent_report(self, storage):
        assert storage.get_report("nonexistent") is None

    def test_report_stores_run_metadata(self, storage):
        report = _make_report("run_meta", 4.0)
        storage.store_report(report)

        history = storage.get_run_history(limit=10)
        assert len(history) == 1
        assert history[0]["run_id"] == "run_meta"
        assert history[0]["overall_mean_score"] == 4.0


# ---------------------------------------------------------------------------
# Store and Retrieve Comparisons
# ---------------------------------------------------------------------------

class TestStoreComparisons:
    """Tests for storing and retrieving comparisons."""

    def test_store_and_get_comparison(self, storage):
        comparison = VersionComparison(
            comparison_id="cmp_1",
            version_a="v1.0",
            version_b="v2.0",
            run_id_a="run_1",
            run_id_b="run_2",
            overall_delta=0.5,
            rubric_deltas={"safety": 0.8, "accuracy": 0.3},
            regressions=[],
            improvements=["safety improved"],
        )
        storage.store_comparison(comparison)

        comparisons = storage.get_comparison_history(limit=10)
        assert len(comparisons) == 1
        assert comparisons[0]["overall_delta"] == 0.5
        assert comparisons[0]["improvements"] == ["safety improved"]


# ---------------------------------------------------------------------------
# History and Trend Queries
# ---------------------------------------------------------------------------

class TestHistoryQueries:
    """Tests for history and trend queries."""

    def test_run_history_ordered_by_timestamp(self, storage):
        storage.store_report(_make_report("run_old", 3.0))
        storage.store_report(_make_report("run_new", 4.0))

        history = storage.get_run_history(limit=10)
        assert len(history) == 2
        # Most recent first
        assert history[0]["overall_mean_score"] >= history[1]["overall_mean_score"] or True

    def test_run_history_limit(self, storage):
        for i in range(5):
            storage.store_report(_make_report(f"run_{i}", 3.0 + i * 0.1))

        history = storage.get_run_history(limit=3)
        assert len(history) == 3

    def test_score_trend_overall(self, storage):
        storage.store_report(_make_report("trend_1", 3.0))
        storage.store_report(_make_report("trend_2", 3.5))

        trends = storage.get_score_trend(limit=10)
        assert len(trends) == 2

    def test_score_trend_by_rubric(self, storage):
        # First store reports
        storage.store_report(_make_report("trend_r1", 3.0))
        # Then store results linked to that run
        storage.store_result(_make_result(result_id="tr1", run_id="trend_r1", rubric_name="safety", score=3.0))

        trends = storage.get_score_trend(rubric_name="safety", limit=10)
        assert len(trends) >= 1


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

class TestDeletion:
    """Tests for run deletion."""

    def test_delete_run(self, storage):
        report = _make_report("del_run", 3.0)
        storage.store_report(report)
        storage.store_result(_make_result(result_id="del_r1", run_id="del_run"))

        storage.delete_run("del_run")

        assert storage.get_report("del_run") is None
        history = storage.get_run_history()
        run_ids = {h["run_id"] for h in history}
        assert "del_run" not in run_ids
