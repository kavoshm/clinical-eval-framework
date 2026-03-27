"""
Tests for Pydantic data models.

Covers:
    - Score aggregation and weighted mean computation
    - Score label and normalized score computation
    - Version comparison regression detection
    - Recommendation logic
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from models import (
    EvalResult,
    EvalReport,
    RubricSummary,
    ClinicalOutput,
    VersionComparison,
    CriterionScore,
    RubricCategory,
    OutputType,
    EvalStatus,
    ScoringLevel,
    EvalCriterion,
    EvalRubric,
)


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

class TestEvalResult:
    """Tests for individual evaluation result scoring."""

    def test_normalized_score(self):
        result = EvalResult(
            result_id="r1",
            output_id="o1",
            rubric_name="test",
            rubric_category=RubricCategory.ACCURACY,
            score=4.0,
            max_score=5.0,
        )
        assert result.normalized_score == 0.8

    def test_normalized_score_zero_max(self):
        result = EvalResult(
            result_id="r1",
            output_id="o1",
            rubric_name="test",
            rubric_category=RubricCategory.ACCURACY,
            score=4.0,
            max_score=0.0,
        )
        assert result.normalized_score == 0.0

    @pytest.mark.parametrize(
        "score,expected_label",
        [
            (5.0, "Excellent"),
            (4.5, "Excellent"),
            (4.0, "Good"),
            (3.5, "Good"),
            (3.0, "Adequate"),
            (2.5, "Adequate"),
            (2.0, "Poor"),
            (1.5, "Poor"),
            (1.0, "Critical"),
            (0.5, "Critical"),
        ],
    )
    def test_score_label(self, score, expected_label):
        result = EvalResult(
            result_id="r1",
            output_id="o1",
            rubric_name="test",
            rubric_category=RubricCategory.SAFETY,
            score=score,
        )
        assert result.score_label == expected_label


# ---------------------------------------------------------------------------
# EvalReport — Score Aggregation & Weight Application
# ---------------------------------------------------------------------------

class TestEvalReportAggregation:
    """Tests for report-level score aggregation and weighted means."""

    def _make_report(self, summaries, weights=None):
        metadata = {}
        if weights:
            metadata["rubric_weights"] = weights
        return EvalReport(
            report_id="rpt_1",
            eval_run_id="run_1",
            prompt_version="v1.0",
            judge_model="gpt-4",
            num_outputs=10,
            num_rubrics=len(summaries),
            total_evaluations=40,
            overall_mean_score=sum(s.mean_score for s in summaries) / len(summaries),
            rubric_summaries=summaries,
            metadata=metadata,
        )

    def test_unweighted_mean(self):
        summaries = [
            RubricSummary(
                rubric_name="accuracy", rubric_display_name="Accuracy",
                mean_score=4.0, min_score=3.0, max_score=5.0, std_dev=0.5, num_outputs=10,
            ),
            RubricSummary(
                rubric_name="safety", rubric_display_name="Safety",
                mean_score=2.0, min_score=1.0, max_score=3.0, std_dev=0.8, num_outputs=10,
            ),
        ]
        report = self._make_report(summaries)
        assert report.overall_mean_score == 3.0
        # Without weights in metadata, weighted_mean falls back to overall_mean
        assert report.weighted_mean_score == report.overall_mean_score

    def test_weighted_mean_with_safety_2x(self):
        """Safety at 2x weight should pull overall score toward the safety score."""
        summaries = [
            RubricSummary(
                rubric_name="accuracy", rubric_display_name="Accuracy",
                mean_score=4.0, min_score=3.0, max_score=5.0, std_dev=0.5, num_outputs=10,
            ),
            RubricSummary(
                rubric_name="safety", rubric_display_name="Safety",
                mean_score=2.0, min_score=1.0, max_score=3.0, std_dev=0.8, num_outputs=10,
            ),
        ]
        weights = {"accuracy": 1.0, "safety": 2.0}
        report = self._make_report(summaries, weights)
        # Expected: (4.0*1.0 + 2.0*2.0) / (1.0 + 2.0) = 8.0/3.0 = 2.6667
        assert abs(report.weighted_mean_score - 2.6667) < 0.001

    def test_weighted_mean_all_equal_weights(self):
        summaries = [
            RubricSummary(
                rubric_name="a", rubric_display_name="A",
                mean_score=3.0, min_score=1.0, max_score=5.0, std_dev=1.0, num_outputs=5,
            ),
            RubricSummary(
                rubric_name="b", rubric_display_name="B",
                mean_score=5.0, min_score=4.0, max_score=5.0, std_dev=0.2, num_outputs=5,
            ),
        ]
        weights = {"a": 1.0, "b": 1.0}
        report = self._make_report(summaries, weights)
        assert abs(report.weighted_mean_score - 4.0) < 0.001

    def test_safety_weight_hides_failure_without_weighting(self):
        """Demonstrate that unweighted scores can hide safety failures.

        Four rubrics: accuracy=4.5, safety=1.5, completeness=4.0, appropriateness=4.0
        Unweighted mean = 3.5 (looks "Good")
        Weighted mean (safety 2x) = 3.019 (reveals "Adequate" / borderline)
        """
        summaries = [
            RubricSummary(
                rubric_name="accuracy", rubric_display_name="Accuracy",
                mean_score=4.5, min_score=4.0, max_score=5.0, std_dev=0.3, num_outputs=10,
            ),
            RubricSummary(
                rubric_name="safety", rubric_display_name="Safety",
                mean_score=1.5, min_score=1.0, max_score=2.0, std_dev=0.5, num_outputs=10,
            ),
            RubricSummary(
                rubric_name="completeness", rubric_display_name="Completeness",
                mean_score=4.0, min_score=3.5, max_score=4.5, std_dev=0.3, num_outputs=10,
            ),
            RubricSummary(
                rubric_name="appropriateness", rubric_display_name="Appropriateness",
                mean_score=4.0, min_score=3.5, max_score=4.5, std_dev=0.3, num_outputs=10,
            ),
        ]
        weights = {
            "accuracy": 1.5,
            "safety": 2.0,
            "completeness": 1.0,
            "appropriateness": 0.8,
        }
        report = self._make_report(summaries, weights)
        unweighted = report.overall_mean_score
        weighted = report.weighted_mean_score
        # Unweighted mean: (4.5+1.5+4.0+4.0)/4 = 3.5
        assert abs(unweighted - 3.5) < 0.01
        # Weighted mean should be lower because safety=1.5 carries 2x weight
        assert weighted < unweighted
        # The weighted mean should be noticeably lower than the unweighted mean
        assert weighted < 3.25

    def test_empty_report_weighted_mean(self):
        report = EvalReport(
            report_id="rpt",
            eval_run_id="run",
            prompt_version="v1",
            judge_model="gpt-4",
        )
        assert report.weighted_mean_score == 0.0


# ---------------------------------------------------------------------------
# VersionComparison — Regression Detection
# ---------------------------------------------------------------------------

class TestVersionComparison:
    """Tests for version comparison and regression detection."""

    def test_no_regressions(self):
        comparison = VersionComparison(
            comparison_id="cmp_1",
            version_a="v1.0",
            version_b="v2.0",
            run_id_a="run_1",
            run_id_b="run_2",
            overall_delta=0.5,
            rubric_deltas={"accuracy": 0.3, "safety": 0.7},
            regressions=[],
            improvements=["safety: 2.8 -> 3.5 (delta=+0.700)"],
        )
        assert comparison.has_regressions is False
        assert comparison.recommendation == "DEPLOY - significant improvement"

    def test_with_regressions(self):
        comparison = VersionComparison(
            comparison_id="cmp_2",
            version_a="v1.0",
            version_b="v2.0",
            run_id_a="run_1",
            run_id_b="run_2",
            overall_delta=0.1,
            rubric_deltas={"accuracy": 0.5, "safety": -0.3},
            regressions=["safety: 3.5 -> 3.2 (delta=-0.300)"],
            improvements=["accuracy: 3.0 -> 3.5 (delta=+0.500)"],
        )
        assert comparison.has_regressions is True
        assert comparison.recommendation == "HOLD - regressions detected"

    def test_marginal_improvement(self):
        comparison = VersionComparison(
            comparison_id="cmp_3",
            version_a="v1.0",
            version_b="v2.0",
            run_id_a="run_1",
            run_id_b="run_2",
            overall_delta=0.05,
            regressions=[],
        )
        assert comparison.recommendation == "DEPLOY - marginal improvement"

    def test_no_improvement(self):
        comparison = VersionComparison(
            comparison_id="cmp_4",
            version_a="v1.0",
            version_b="v2.0",
            run_id_a="run_1",
            run_id_b="run_2",
            overall_delta=-0.05,
            regressions=[],
        )
        assert comparison.recommendation == "HOLD - no improvement"


# ---------------------------------------------------------------------------
# ClinicalOutput
# ---------------------------------------------------------------------------

class TestClinicalOutput:
    """Tests for clinical output model."""

    def test_word_count(self):
        output = ClinicalOutput(
            output_id="o1",
            text="This is a five word sentence.",
        )
        assert output.word_count == 6  # "This", "is", "a", "five", "word", "sentence."

    def test_default_fields(self):
        output = ClinicalOutput(output_id="o1", text="Hello")
        assert output.output_type == OutputType.SESSION_SUMMARY
        assert output.model == "unknown"
        assert output.prompt_version == "unknown"

    def test_output_types(self):
        for otype in OutputType:
            output = ClinicalOutput(output_id="o1", text="Test", output_type=otype)
            assert output.output_type == otype


# ---------------------------------------------------------------------------
# EvalRubric
# ---------------------------------------------------------------------------

class TestEvalRubric:
    """Tests for rubric model methods."""

    def test_get_scoring_level_found(self):
        rubric = EvalRubric(
            name="test",
            display_name="Test",
            description="Desc",
            category=RubricCategory.ACCURACY,
            scoring_levels=[
                ScoringLevel(score=1, label="Bad", description="Bad desc"),
                ScoringLevel(score=5, label="Good", description="Good desc"),
            ],
        )
        level = rubric.get_scoring_level(1)
        assert level is not None
        assert level.label == "Bad"

    def test_get_scoring_level_not_found(self):
        rubric = EvalRubric(
            name="test",
            display_name="Test",
            description="Desc",
            category=RubricCategory.ACCURACY,
        )
        assert rubric.get_scoring_level(3) is None
