"""
Main Evaluation Engine

Orchestrates the full evaluation pipeline:
    1. Load rubrics from YAML files
    2. Run LLM-as-judge evaluation on each output x rubric pair
    3. Aggregate scores into a structured report
    4. Store results in SQLite for historical tracking
    5. Support comparing multiple evaluation runs
"""

from __future__ import annotations

import logging
import uuid
import math
from typing import Optional
from datetime import datetime, timezone

from models import (
    EvalRubric,
    EvalResult,
    EvalReport,
    ClinicalOutput,
    RubricSummary,
    VersionComparison,
)
from rubric_loader import RubricLoader
from judge import ClinicalJudge, JudgeConfig
from storage import EvalStorage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ClinicalEvaluator:
    """
    Main evaluation engine for clinical LLM outputs.

    Usage:
        evaluator = ClinicalEvaluator()
        report = evaluator.run_evaluation(
            outputs=clinical_outputs,
            prompt_version="v1.2",
        )
        evaluator.compare_runs(run_id_a, run_id_b)
    """

    def __init__(
        self,
        rubric_loader: Optional[RubricLoader] = None,
        judge: Optional[ClinicalJudge] = None,
        storage: Optional[EvalStorage] = None,
        db_path: Optional[str] = None,
    ):
        self.rubric_loader = rubric_loader or RubricLoader()
        self.judge = judge or ClinicalJudge(JudgeConfig(simulate=True))
        self.storage = storage or EvalStorage(db_path=db_path or "eval_results.db")

    def run_evaluation(
        self,
        outputs: list[ClinicalOutput],
        prompt_version: str = "unknown",
        rubric_names: Optional[list[str]] = None,
        run_id: Optional[str] = None,
    ) -> EvalReport:
        """
        Run a complete evaluation across all outputs and rubrics.

        Args:
            outputs: List of clinical outputs to evaluate
            prompt_version: Version string for the prompt that generated these outputs
            rubric_names: Specific rubrics to use (default: all available)
            run_id: Optional run ID (generated if not provided)

        Returns:
            EvalReport with all results aggregated
        """
        # Generate run ID
        eval_run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        logger.info(
            "Starting evaluation run '%s' | prompt_version='%s' | outputs=%d",
            eval_run_id, prompt_version, len(outputs),
        )

        # Load rubrics
        if rubric_names:
            rubrics = [self.rubric_loader.load(name) for name in rubric_names]
        else:
            rubrics = self.rubric_loader.load_all()

        if not rubrics:
            raise ValueError("No rubrics found. Check the rubrics directory.")

        # Run evaluations
        all_results: list[EvalResult] = []

        for output in outputs:
            for rubric in rubrics:
                result = self.judge.evaluate(
                    output=output,
                    rubric=rubric,
                    eval_run_id=eval_run_id,
                )
                all_results.append(result)

                # Store individual result
                self.storage.store_result(result)

        # Build report
        report = self._build_report(
            results=all_results,
            eval_run_id=eval_run_id,
            prompt_version=prompt_version,
            rubrics=rubrics,
            num_outputs=len(outputs),
        )

        # Store report
        self.storage.store_report(report)

        logger.info(
            "Evaluation run '%s' complete | overall_mean=%.3f | weighted_mean=%.3f | total_evals=%d",
            eval_run_id, report.overall_mean_score, report.weighted_mean_score, len(all_results),
        )
        return report

    def _build_report(
        self,
        results: list[EvalResult],
        eval_run_id: str,
        prompt_version: str,
        rubrics: list[EvalRubric],
        num_outputs: int,
    ) -> EvalReport:
        """Build an aggregated evaluation report from individual results."""

        # Compute rubric summaries
        rubric_summaries = []
        rubric_weights = {}

        for rubric in rubrics:
            rubric_results = [r for r in results if r.rubric_name == rubric.name]
            if not rubric_results:
                continue

            scores = [r.score for r in rubric_results]
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / max(len(scores), 1)
            std_dev = math.sqrt(variance)

            # Score distribution
            distribution = {}
            for s in scores:
                bucket = str(round(s))
                distribution[bucket] = distribution.get(bucket, 0) + 1

            rubric_summaries.append(RubricSummary(
                rubric_name=rubric.name,
                rubric_display_name=rubric.display_name,
                mean_score=round(mean_score, 4),
                min_score=round(min(scores), 4),
                max_score=round(max(scores), 4),
                std_dev=round(std_dev, 4),
                num_outputs=len(rubric_results),
                score_distribution=distribution,
            ))

            rubric_weights[rubric.name] = rubric.weight

        # Compute overall mean
        all_scores = [r.score for r in results]
        overall_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0

        report_id = f"report_{eval_run_id}"

        return EvalReport(
            report_id=report_id,
            eval_run_id=eval_run_id,
            prompt_version=prompt_version,
            judge_model=self.judge.config.model,
            num_outputs=num_outputs,
            num_rubrics=len(rubrics),
            total_evaluations=len(results),
            overall_mean_score=round(overall_mean, 4),
            rubric_summaries=rubric_summaries,
            results=results,
            metadata={"rubric_weights": rubric_weights},
        )

    def compare_runs(
        self,
        run_id_a: str,
        run_id_b: str,
        regression_threshold: float = 0.2,
    ) -> VersionComparison:
        """
        Compare two evaluation runs.

        Args:
            run_id_a: First run ID (typically the baseline / earlier version)
            run_id_b: Second run ID (typically the candidate / newer version)
            regression_threshold: Minimum score drop to flag as regression

        Returns:
            VersionComparison with deltas and regression analysis
        """
        logger.info(
            "Comparing runs '%s' vs '%s' | regression_threshold=%.2f",
            run_id_a, run_id_b, regression_threshold,
        )
        report_a = self.storage.get_report(run_id_a)
        report_b = self.storage.get_report(run_id_b)

        if not report_a or not report_b:
            missing = []
            if not report_a:
                missing.append(run_id_a)
            if not report_b:
                missing.append(run_id_b)
            raise ValueError(f"Run(s) not found: {', '.join(missing)}")

        # Compute deltas
        overall_delta = report_b.overall_mean_score - report_a.overall_mean_score

        rubric_deltas = {}
        regressions = []
        improvements = []

        summaries_a = {s.rubric_name: s for s in report_a.rubric_summaries}
        summaries_b = {s.rubric_name: s for s in report_b.rubric_summaries}

        for rubric_name in set(summaries_a.keys()) | set(summaries_b.keys()):
            score_a = summaries_a.get(rubric_name, None)
            score_b = summaries_b.get(rubric_name, None)

            if score_a and score_b:
                delta = score_b.mean_score - score_a.mean_score
                rubric_deltas[rubric_name] = round(delta, 4)

                if delta < -regression_threshold:
                    regressions.append(
                        f"{rubric_name}: {score_a.mean_score:.3f} -> "
                        f"{score_b.mean_score:.3f} (delta={delta:+.3f})"
                    )
                elif delta > regression_threshold:
                    improvements.append(
                        f"{rubric_name}: {score_a.mean_score:.3f} -> "
                        f"{score_b.mean_score:.3f} (delta={delta:+.3f})"
                    )

        comparison = VersionComparison(
            comparison_id=f"cmp_{run_id_a}_{run_id_b}",
            version_a=report_a.prompt_version,
            version_b=report_b.prompt_version,
            run_id_a=run_id_a,
            run_id_b=run_id_b,
            overall_delta=round(overall_delta, 4),
            rubric_deltas=rubric_deltas,
            regressions=regressions,
            improvements=improvements,
        )

        # Store comparison
        self.storage.store_comparison(comparison)

        logger.info(
            "Comparison complete: %s vs %s | delta=%+.3f | regressions=%d | improvements=%d | recommendation='%s'",
            comparison.version_a, comparison.version_b,
            comparison.overall_delta, len(comparison.regressions),
            len(comparison.improvements), comparison.recommendation,
        )
        if comparison.has_regressions:
            for reg in comparison.regressions:
                logger.warning("REGRESSION DETECTED: %s", reg)

        return comparison

    def get_history(self, limit: int = 10) -> list[dict]:
        """Get evaluation run history."""
        return self.storage.get_run_history(limit)

    def get_report(self, run_id: str) -> Optional[EvalReport]:
        """Retrieve a stored report by run ID."""
        return self.storage.get_report(run_id)
