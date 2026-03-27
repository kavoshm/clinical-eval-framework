"""
Tests for the report generation module.

Covers:
    - Evaluation report generation (markdown format)
    - Comparison report generation
    - Score-to-rating mapping
    - Score-to-status mapping
    - Safety alert section
    - Histogram rendering
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from reporter import ReportGenerator, generate_report, generate_comparison
from models import (
    EvalReport,
    EvalResult,
    RubricSummary,
    VersionComparison,
    CriterionScore,
    RubricCategory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generator():
    return ReportGenerator()


@pytest.fixture
def sample_report():
    return EvalReport(
        report_id="rpt_test",
        eval_run_id="run_test",
        prompt_version="v1.0",
        judge_model="gpt-4",
        num_outputs=10,
        num_rubrics=4,
        total_evaluations=40,
        overall_mean_score=3.248,
        rubric_summaries=[
            RubricSummary(
                rubric_name="clinical_accuracy",
                rubric_display_name="Clinical Accuracy",
                mean_score=3.380,
                min_score=1.500,
                max_score=4.800,
                std_dev=1.1,
                num_outputs=10,
                score_distribution={"2": 2, "3": 3, "4": 3, "5": 2},
            ),
            RubricSummary(
                rubric_name="patient_safety",
                rubric_display_name="Patient Safety",
                mean_score=2.860,
                min_score=1.000,
                max_score=4.800,
                std_dev=1.3,
                num_outputs=10,
                score_distribution={"1": 2, "2": 2, "3": 2, "4": 2, "5": 2},
            ),
        ],
        results=[
            EvalResult(
                result_id="r1",
                output_id="out_001",
                rubric_name="clinical_accuracy",
                rubric_category=RubricCategory.ACCURACY,
                score=4.8,
                reasoning="All clinical facts accurate.",
            ),
            EvalResult(
                result_id="r2",
                output_id="out_001",
                rubric_name="patient_safety",
                rubric_category=RubricCategory.SAFETY,
                score=4.8,
                reasoning="Output is safe.",
            ),
            EvalResult(
                result_id="r3",
                output_id="out_009",
                rubric_name="patient_safety",
                rubric_category=RubricCategory.SAFETY,
                score=1.0,
                reasoning="CRITICAL SAFETY FAILURE: Output describes abnormal findings as normal.",
            ),
        ],
    )


@pytest.fixture
def sample_comparison():
    return VersionComparison(
        comparison_id="cmp_test",
        version_a="v1.0",
        version_b="v2.0",
        run_id_a="run_v1",
        run_id_b="run_v2",
        overall_delta=0.644,
        rubric_deltas={
            "clinical_accuracy": 0.650,
            "patient_safety": 0.860,
        },
        regressions=[],
        improvements=[
            "clinical_accuracy: 3.380 -> 4.030 (delta=+0.650)",
            "patient_safety: 2.860 -> 3.720 (delta=+0.860)",
        ],
    )


# ---------------------------------------------------------------------------
# Evaluation Report Generation
# ---------------------------------------------------------------------------

class TestEvalReportGeneration:
    """Tests for evaluation report markdown generation."""

    def test_report_contains_header(self, generator, sample_report):
        md = generator.generate_eval_report(sample_report)
        assert "# Evaluation Report: v1.0" in md

    def test_report_contains_run_metadata(self, generator, sample_report):
        md = generator.generate_eval_report(sample_report)
        assert "run_test" in md
        assert "gpt-4" in md
        assert "10" in md  # num_outputs

    def test_report_contains_overall_score(self, generator, sample_report):
        md = generator.generate_eval_report(sample_report)
        assert "3.248" in md

    def test_report_contains_rubric_table(self, generator, sample_report):
        md = generator.generate_eval_report(sample_report)
        assert "Clinical Accuracy" in md
        assert "Patient Safety" in md
        assert "3.380" in md
        assert "2.860" in md

    def test_report_contains_status_indicators(self, generator, sample_report):
        md = generator.generate_eval_report(sample_report)
        # Safety mean 2.860 should be FAIL
        assert "FAIL" in md

    def test_report_contains_safety_alerts(self, generator, sample_report):
        md = generator.generate_eval_report(sample_report)
        assert "Safety Alerts" in md
        assert "out_009" in md
        assert "CRITICAL" in md

    def test_report_contains_detailed_results(self, generator, sample_report):
        md = generator.generate_eval_report(sample_report)
        assert "out_001" in md
        assert "Detailed Results" in md

    def test_report_contains_score_distributions(self, generator, sample_report):
        md = generator.generate_eval_report(sample_report)
        assert "Score Distribution" in md

    def test_report_no_safety_alerts_when_all_safe(self, generator):
        safe_report = EvalReport(
            report_id="rpt_safe",
            eval_run_id="run_safe",
            prompt_version="v2.0",
            judge_model="gpt-4",
            num_outputs=5,
            num_rubrics=1,
            total_evaluations=5,
            overall_mean_score=4.5,
            rubric_summaries=[],
            results=[
                EvalResult(
                    result_id="rs1",
                    output_id="o1",
                    rubric_name="patient_safety",
                    rubric_category=RubricCategory.SAFETY,
                    score=4.5,
                    reasoning="Safe output.",
                ),
            ],
        )
        md = generator.generate_eval_report(safe_report)
        assert "Safety Alerts" not in md


# ---------------------------------------------------------------------------
# Comparison Report Generation
# ---------------------------------------------------------------------------

class TestComparisonReportGeneration:
    """Tests for version comparison report generation."""

    def test_comparison_report_header(self, generator, sample_comparison):
        md = generator.generate_comparison_report(sample_comparison)
        assert "v1.0 vs v2.0" in md

    def test_comparison_report_contains_delta(self, generator, sample_comparison):
        md = generator.generate_comparison_report(sample_comparison)
        assert "+0.644" in md

    def test_comparison_report_shows_improvements(self, generator, sample_comparison):
        md = generator.generate_comparison_report(sample_comparison)
        assert "Improvements" in md
        assert "patient_safety" in md

    def test_comparison_report_shows_recommendation(self, generator, sample_comparison):
        md = generator.generate_comparison_report(sample_comparison)
        assert "DEPLOY" in md

    def test_comparison_with_side_by_side(self, generator, sample_comparison, sample_report):
        report_b = EvalReport(
            report_id="rpt_v2",
            eval_run_id="run_v2",
            prompt_version="v2.0",
            judge_model="gpt-4",
            overall_mean_score=3.892,
            rubric_summaries=[
                RubricSummary(
                    rubric_name="clinical_accuracy",
                    rubric_display_name="Clinical Accuracy",
                    mean_score=4.030,
                    min_score=2.300,
                    max_score=4.800,
                    std_dev=0.8,
                    num_outputs=10,
                ),
                RubricSummary(
                    rubric_name="patient_safety",
                    rubric_display_name="Patient Safety",
                    mean_score=3.720,
                    min_score=2.000,
                    max_score=4.800,
                    std_dev=0.9,
                    num_outputs=10,
                ),
            ],
        )
        md = generator.generate_comparison_report(sample_comparison, sample_report, report_b)
        assert "Side-by-Side" in md
        assert "4.030" in md
        assert "3.720" in md


# ---------------------------------------------------------------------------
# Score Helpers
# ---------------------------------------------------------------------------

class TestScoreHelpers:
    """Tests for score-to-rating and score-to-status conversions."""

    @pytest.mark.parametrize(
        "score,expected",
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
        ],
    )
    def test_score_to_rating(self, score, expected):
        assert ReportGenerator._score_to_rating(score) == expected

    @pytest.mark.parametrize(
        "score,expected",
        [
            (4.5, "PASS"),
            (4.0, "PASS"),
            (3.5, "WARN"),
            (3.0, "WARN"),
            (2.5, "FAIL"),
            (1.0, "FAIL"),
        ],
    )
    def test_score_to_status(self, score, expected):
        assert ReportGenerator._score_to_status(score) == expected


# ---------------------------------------------------------------------------
# Histogram Rendering
# ---------------------------------------------------------------------------

class TestHistogramRendering:
    """Tests for ASCII histogram rendering."""

    def test_histogram_renders(self, generator):
        summary = RubricSummary(
            rubric_name="test",
            rubric_display_name="Test Rubric",
            mean_score=3.0,
            min_score=1.0,
            max_score=5.0,
            std_dev=1.0,
            num_outputs=10,
            score_distribution={"1": 1, "2": 2, "3": 4, "4": 2, "5": 1},
        )
        histogram = generator._render_histogram(summary)
        assert "Test Rubric" in histogram
        assert "Score" in histogram
        assert "#" in histogram

    def test_histogram_empty_distribution(self, generator):
        summary = RubricSummary(
            rubric_name="empty",
            rubric_display_name="Empty",
            mean_score=0.0,
            min_score=0.0,
            max_score=0.0,
            std_dev=0.0,
            num_outputs=0,
        )
        histogram = generator._render_histogram(summary)
        assert "Empty" in histogram


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_generate_report_function(self, sample_report):
        md = generate_report(sample_report)
        assert "# Evaluation Report:" in md

    def test_generate_comparison_function(self, sample_comparison):
        md = generate_comparison(sample_comparison)
        assert "Version Comparison" in md
