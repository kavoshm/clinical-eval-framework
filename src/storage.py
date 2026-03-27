"""
SQLite Storage Layer for Evaluation Results

Stores evaluation results, reports, and comparisons in a SQLite database
for historical tracking and analysis.

Schema:
    - eval_results: individual output x rubric evaluation results
    - eval_reports: aggregated evaluation reports per run
    - eval_comparisons: version comparison records
    - eval_runs: metadata about each evaluation run
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from models import (
    EvalResult,
    EvalReport,
    VersionComparison,
    RubricCategory,
    RubricSummary,
    CriterionScore,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    prompt_version TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    num_outputs INTEGER NOT NULL DEFAULT 0,
    num_rubrics INTEGER NOT NULL DEFAULT 0,
    total_evaluations INTEGER NOT NULL DEFAULT 0,
    overall_mean_score REAL NOT NULL DEFAULT 0.0,
    weighted_mean_score REAL NOT NULL DEFAULT 0.0,
    timestamp TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS eval_results (
    result_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    output_id TEXT NOT NULL,
    rubric_name TEXT NOT NULL,
    rubric_category TEXT NOT NULL,
    score REAL NOT NULL,
    max_score REAL NOT NULL DEFAULT 5.0,
    reasoning TEXT DEFAULT '',
    criterion_scores TEXT DEFAULT '[]',
    judge_model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    latency_ms REAL DEFAULT 0.0,
    timestamp TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
);

CREATE TABLE IF NOT EXISTS eval_reports (
    report_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    prompt_version TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    num_outputs INTEGER NOT NULL DEFAULT 0,
    num_rubrics INTEGER NOT NULL DEFAULT 0,
    total_evaluations INTEGER NOT NULL DEFAULT 0,
    overall_mean_score REAL NOT NULL DEFAULT 0.0,
    rubric_summaries TEXT DEFAULT '[]',
    timestamp TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
);

CREATE TABLE IF NOT EXISTS eval_comparisons (
    comparison_id TEXT PRIMARY KEY,
    version_a TEXT NOT NULL,
    version_b TEXT NOT NULL,
    run_id_a TEXT NOT NULL,
    run_id_b TEXT NOT NULL,
    overall_delta REAL NOT NULL DEFAULT 0.0,
    rubric_deltas TEXT DEFAULT '{}',
    regressions TEXT DEFAULT '[]',
    improvements TEXT DEFAULT '[]',
    timestamp TEXT NOT NULL,
    FOREIGN KEY (run_id_a) REFERENCES eval_runs(run_id),
    FOREIGN KEY (run_id_b) REFERENCES eval_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_results_run_id ON eval_results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_output_id ON eval_results(output_id);
CREATE INDEX IF NOT EXISTS idx_results_rubric ON eval_results(rubric_name);
CREATE INDEX IF NOT EXISTS idx_runs_prompt_version ON eval_runs(prompt_version);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON eval_runs(timestamp);
"""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class EvalStorage:
    """
    SQLite storage layer for evaluation results.

    Provides CRUD operations for evaluation data with support
    for historical queries and version comparison.
    """

    def __init__(self, db_path: str = "eval_results.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        logger.debug("Initializing database at '%s'", self.db_path)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            logger.info("Database initialized at '%s'", self.db_path)
        finally:
            conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- Store Operations ---

    def store_result(self, result: EvalResult) -> None:
        """Store a single evaluation result."""
        logger.debug(
            "Storing result: result_id='%s' output_id='%s' rubric='%s' score=%.1f",
            result.result_id, result.output_id, result.rubric_name, result.score,
        )
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO eval_results
                (result_id, run_id, output_id, rubric_name, rubric_category,
                 score, max_score, reasoning, criterion_scores, judge_model,
                 prompt_version, latency_ms, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.result_id,
                    result.eval_run_id,
                    result.output_id,
                    result.rubric_name,
                    result.rubric_category.value,
                    result.score,
                    result.max_score,
                    result.reasoning,
                    json.dumps([cs.model_dump() for cs in result.criterion_scores]),
                    result.judge_model,
                    result.prompt_version,
                    result.latency_ms,
                    result.timestamp.isoformat(),
                    json.dumps(result.metadata),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def store_report(self, report: EvalReport) -> None:
        """Store an evaluation report and its run metadata."""
        logger.info(
            "Storing report: run_id='%s' prompt_version='%s' overall_mean=%.3f",
            report.eval_run_id, report.prompt_version, report.overall_mean_score,
        )
        conn = self._get_conn()
        try:
            # Store run metadata
            conn.execute(
                """INSERT OR REPLACE INTO eval_runs
                (run_id, prompt_version, judge_model, num_outputs, num_rubrics,
                 total_evaluations, overall_mean_score, weighted_mean_score,
                 timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report.eval_run_id,
                    report.prompt_version,
                    report.judge_model,
                    report.num_outputs,
                    report.num_rubrics,
                    report.total_evaluations,
                    report.overall_mean_score,
                    report.weighted_mean_score,
                    report.timestamp.isoformat(),
                    json.dumps(report.metadata),
                ),
            )

            # Store report
            conn.execute(
                """INSERT OR REPLACE INTO eval_reports
                (report_id, run_id, prompt_version, judge_model, num_outputs,
                 num_rubrics, total_evaluations, overall_mean_score,
                 rubric_summaries, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report.report_id,
                    report.eval_run_id,
                    report.prompt_version,
                    report.judge_model,
                    report.num_outputs,
                    report.num_rubrics,
                    report.total_evaluations,
                    report.overall_mean_score,
                    json.dumps([s.model_dump() for s in report.rubric_summaries]),
                    report.timestamp.isoformat(),
                    json.dumps(report.metadata),
                ),
            )

            conn.commit()
        finally:
            conn.close()

    def store_comparison(self, comparison: VersionComparison) -> None:
        """Store a version comparison record."""
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO eval_comparisons
                (comparison_id, version_a, version_b, run_id_a, run_id_b,
                 overall_delta, rubric_deltas, regressions, improvements, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    comparison.comparison_id,
                    comparison.version_a,
                    comparison.version_b,
                    comparison.run_id_a,
                    comparison.run_id_b,
                    comparison.overall_delta,
                    json.dumps(comparison.rubric_deltas),
                    json.dumps(comparison.regressions),
                    json.dumps(comparison.improvements),
                    comparison.timestamp.isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # --- Query Operations ---

    def get_report(self, run_id: str) -> Optional[EvalReport]:
        """Retrieve an evaluation report by run ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM eval_reports WHERE run_id = ?",
                (run_id,),
            ).fetchone()

            if not row:
                return None

            # Parse rubric summaries
            rubric_summaries_data = json.loads(row["rubric_summaries"])
            rubric_summaries = [
                RubricSummary(**s) for s in rubric_summaries_data
            ]

            # Fetch associated results
            result_rows = conn.execute(
                "SELECT * FROM eval_results WHERE run_id = ?",
                (run_id,),
            ).fetchall()

            results = []
            for rr in result_rows:
                criterion_scores_data = json.loads(rr["criterion_scores"])
                criterion_scores = [CriterionScore(**cs) for cs in criterion_scores_data]

                results.append(EvalResult(
                    result_id=rr["result_id"],
                    output_id=rr["output_id"],
                    rubric_name=rr["rubric_name"],
                    rubric_category=RubricCategory(rr["rubric_category"]),
                    score=rr["score"],
                    max_score=rr["max_score"],
                    reasoning=rr["reasoning"],
                    criterion_scores=criterion_scores,
                    judge_model=rr["judge_model"],
                    prompt_version=rr["prompt_version"],
                    eval_run_id=rr["run_id"],
                    latency_ms=rr["latency_ms"],
                    timestamp=datetime.fromisoformat(rr["timestamp"]),
                    metadata=json.loads(rr["metadata"]),
                ))

            return EvalReport(
                report_id=row["report_id"],
                eval_run_id=row["run_id"],
                prompt_version=row["prompt_version"],
                judge_model=row["judge_model"],
                num_outputs=row["num_outputs"],
                num_rubrics=row["num_rubrics"],
                total_evaluations=row["total_evaluations"],
                overall_mean_score=row["overall_mean_score"],
                rubric_summaries=rubric_summaries,
                results=results,
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"]),
            )
        finally:
            conn.close()

    def get_run_history(self, limit: int = 10) -> list[dict]:
        """Get evaluation run history, most recent first."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT run_id, prompt_version, judge_model, num_outputs,
                          total_evaluations, overall_mean_score, weighted_mean_score,
                          timestamp
                   FROM eval_runs
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_results_by_output(self, output_id: str) -> list[dict]:
        """Get all evaluation results for a specific output."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT result_id, run_id, rubric_name, rubric_category,
                          score, max_score, reasoning, prompt_version, timestamp
                   FROM eval_results
                   WHERE output_id = ?
                   ORDER BY timestamp DESC""",
                (output_id,),
            ).fetchall()

            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_results_by_rubric(
        self,
        rubric_name: str,
        run_id: Optional[str] = None,
    ) -> list[dict]:
        """Get all results for a specific rubric, optionally filtered by run."""
        conn = self._get_conn()
        try:
            if run_id:
                rows = conn.execute(
                    """SELECT * FROM eval_results
                       WHERE rubric_name = ? AND run_id = ?
                       ORDER BY score DESC""",
                    (rubric_name, run_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM eval_results
                       WHERE rubric_name = ?
                       ORDER BY timestamp DESC""",
                    (rubric_name,),
                ).fetchall()

            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_score_trend(
        self,
        rubric_name: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Get score trend over time for regression monitoring.

        Returns average scores per run, ordered by time.
        """
        conn = self._get_conn()
        try:
            if rubric_name:
                rows = conn.execute(
                    """SELECT r.run_id, r.prompt_version, r.timestamp,
                              AVG(e.score) as avg_score,
                              MIN(e.score) as min_score,
                              MAX(e.score) as max_score,
                              COUNT(e.result_id) as num_evals
                       FROM eval_runs r
                       JOIN eval_results e ON r.run_id = e.run_id
                       WHERE e.rubric_name = ?
                       GROUP BY r.run_id
                       ORDER BY r.timestamp DESC
                       LIMIT ?""",
                    (rubric_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT run_id, prompt_version, timestamp,
                              overall_mean_score as avg_score,
                              total_evaluations as num_evals
                       FROM eval_runs
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (limit,),
                ).fetchall()

            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_comparison_history(self, limit: int = 10) -> list[dict]:
        """Get comparison history."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT comparison_id, version_a, version_b,
                          overall_delta, regressions, improvements, timestamp
                   FROM eval_comparisons
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

            results = []
            for row in rows:
                d = dict(row)
                d["regressions"] = json.loads(d["regressions"])
                d["improvements"] = json.loads(d["improvements"])
                results.append(d)

            return results
        finally:
            conn.close()

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all its results."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM eval_results WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM eval_reports WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM eval_runs WHERE run_id = ?", (run_id,))
            conn.commit()
        finally:
            conn.close()
