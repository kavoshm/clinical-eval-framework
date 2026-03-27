"""
Tests for the main evaluation engine.

Covers:
    - Full evaluation pipeline execution
    - Score aggregation across rubrics
    - Version comparison logic
    - Regression detection thresholds
    - Report building
"""

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluator import ClinicalEvaluator
from rubric_loader import RubricLoader
from judge import ClinicalJudge, JudgeConfig
from storage import EvalStorage
from models import ClinicalOutput, OutputType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluator(rubrics_dir, tmp_db):
    rubric_loader = RubricLoader(rubrics_dir=rubrics_dir)
    judge = ClinicalJudge(JudgeConfig(model="gpt-4", simulate=True))
    storage = EvalStorage(db_path=tmp_db)
    return ClinicalEvaluator(
        rubric_loader=rubric_loader,
        judge=judge,
        storage=storage,
    )


@pytest.fixture
def good_outputs(sample_source_text):
    return [
        ClinicalOutput(
            output_id="good_001",
            text=(
                "Assessment and Plan:\n"
                "67M with T2DM (HbA1c 8.2%), HTN, stage IIIA NSCLC s/p C1 carboplatin/pemetrexed.\n\n"
                "1. Oncology: WBC 3.2 (LOW). Monitor. Continue ondansetron 8mg PRN.\n"
                "2. Endocrine: Fasting BG 180-240. HbA1c 8.2%. Consider insulin.\n"
                "3. Renal: Creatinine 1.3 (elevated). Monitor.\n"
                "4. CV: HTN stable on lisinopril 20mg.\n"
                "Medications: metformin 1000mg BID, lisinopril 20mg, ondansetron 8mg PRN.\n"
                "Follow-up: 2 weeks."
            ),
            source_text=sample_source_text,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v1.0",
        ),
    ]


@pytest.fixture
def poor_outputs(sample_source_text):
    return [
        ClinicalOutput(
            output_id="poor_001",
            text="Patient is doing well after chemotherapy. Blood work is normal. Continue current medications.",
            source_text=sample_source_text,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-3.5-turbo",
            prompt_version="v1.0",
        ),
    ]


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Tests for full evaluation pipeline execution."""

    def test_evaluation_produces_report(self, evaluator, good_outputs):
        report = evaluator.run_evaluation(
            outputs=good_outputs,
            prompt_version="v1.0",
            run_id="test_run_1",
        )
        assert report.eval_run_id == "test_run_1"
        assert report.prompt_version == "v1.0"
        assert report.num_outputs == 1
        assert report.num_rubrics == 4
        assert report.total_evaluations == 4  # 1 output x 4 rubrics
        assert report.overall_mean_score > 0

    def test_report_has_rubric_summaries(self, evaluator, good_outputs):
        report = evaluator.run_evaluation(
            outputs=good_outputs,
            prompt_version="v1.0",
            run_id="test_run_summaries",
        )
        assert len(report.rubric_summaries) == 4
        rubric_names = {s.rubric_name for s in report.rubric_summaries}
        assert "clinical_accuracy" in rubric_names
        assert "patient_safety" in rubric_names
        assert "clinical_completeness" in rubric_names
        assert "clinical_appropriateness" in rubric_names

    def test_report_stores_all_results(self, evaluator, good_outputs):
        report = evaluator.run_evaluation(
            outputs=good_outputs,
            prompt_version="v1.0",
            run_id="test_run_results",
        )
        assert len(report.results) == 4

    def test_good_output_scores_higher_than_poor(self, evaluator, good_outputs, poor_outputs):
        report_good = evaluator.run_evaluation(
            outputs=good_outputs,
            prompt_version="v1.0",
            run_id="test_good_run",
        )
        report_poor = evaluator.run_evaluation(
            outputs=poor_outputs,
            prompt_version="v1.0",
            run_id="test_poor_run",
        )
        assert report_good.overall_mean_score > report_poor.overall_mean_score

    def test_evaluation_with_specific_rubrics(self, evaluator, good_outputs):
        report = evaluator.run_evaluation(
            outputs=good_outputs,
            prompt_version="v1.0",
            rubric_names=["accuracy", "safety"],
            run_id="test_specific_rubrics",
        )
        assert report.num_rubrics == 2
        assert report.total_evaluations == 2

    def test_no_rubrics_raises(self, tmp_db, tmp_path):
        """Evaluation with an empty rubrics directory should raise ValueError."""
        empty_dir = tmp_path / "empty_rubrics"
        empty_dir.mkdir()
        rubric_loader = RubricLoader(rubrics_dir=empty_dir)
        judge = ClinicalJudge(JudgeConfig(simulate=True))
        storage = EvalStorage(db_path=tmp_db)
        evaluator = ClinicalEvaluator(
            rubric_loader=rubric_loader,
            judge=judge,
            storage=storage,
        )
        with pytest.raises(ValueError, match="No rubrics found"):
            evaluator.run_evaluation(
                outputs=[ClinicalOutput(output_id="x", text="Test")],
                prompt_version="v1.0",
            )


# ---------------------------------------------------------------------------
# Version Comparison
# ---------------------------------------------------------------------------

class TestVersionComparison:
    """Tests for version comparison logic."""

    def test_compare_runs_returns_comparison(self, evaluator, good_outputs, poor_outputs):
        evaluator.run_evaluation(outputs=poor_outputs, prompt_version="v1.0", run_id="cmp_v1")
        evaluator.run_evaluation(outputs=good_outputs, prompt_version="v2.0", run_id="cmp_v2")

        comparison = evaluator.compare_runs("cmp_v1", "cmp_v2")
        assert comparison.version_a == "v1.0"
        assert comparison.version_b == "v2.0"
        assert comparison.overall_delta > 0

    def test_compare_detects_improvements(self, evaluator, good_outputs, poor_outputs):
        evaluator.run_evaluation(outputs=poor_outputs, prompt_version="v1.0", run_id="imp_v1")
        evaluator.run_evaluation(outputs=good_outputs, prompt_version="v2.0", run_id="imp_v2")

        comparison = evaluator.compare_runs("imp_v1", "imp_v2", regression_threshold=0.2)
        assert len(comparison.improvements) > 0

    def test_compare_missing_run_raises(self, evaluator, good_outputs):
        evaluator.run_evaluation(outputs=good_outputs, prompt_version="v1.0", run_id="exists")
        with pytest.raises(ValueError, match="Run.*not found"):
            evaluator.compare_runs("exists", "nonexistent_run")


# ---------------------------------------------------------------------------
# Regression Detection Thresholds
# ---------------------------------------------------------------------------

class TestRegressionDetection:
    """Tests for regression detection threshold logic."""

    def test_default_threshold_0_2(self, evaluator, good_outputs, poor_outputs):
        """Default threshold is 0.2; verify it is applied correctly."""
        evaluator.run_evaluation(outputs=good_outputs, prompt_version="v1.0", run_id="reg_v1")
        evaluator.run_evaluation(outputs=poor_outputs, prompt_version="v2.0", run_id="reg_v2")

        comparison = evaluator.compare_runs("reg_v1", "reg_v2", regression_threshold=0.2)
        # Good -> Poor is a regression, so at least some rubrics should regress
        assert comparison.overall_delta < 0

    def test_high_threshold_fewer_regressions(self, evaluator, good_outputs, poor_outputs):
        """With a very high threshold (e.g., 10.0), no regressions should be flagged."""
        evaluator.run_evaluation(outputs=good_outputs, prompt_version="v1.0", run_id="ht_v1")
        evaluator.run_evaluation(outputs=poor_outputs, prompt_version="v2.0", run_id="ht_v2")

        comparison = evaluator.compare_runs("ht_v1", "ht_v2", regression_threshold=10.0)
        assert len(comparison.regressions) == 0

    def test_zero_threshold_catches_all(self, evaluator, good_outputs, poor_outputs):
        """With threshold=0.0, even tiny regressions should be flagged."""
        evaluator.run_evaluation(outputs=good_outputs, prompt_version="v1.0", run_id="zt_v1")
        evaluator.run_evaluation(outputs=poor_outputs, prompt_version="v2.0", run_id="zt_v2")

        comparison = evaluator.compare_runs("zt_v1", "zt_v2", regression_threshold=0.0)
        # All rubrics should show regression since good > poor
        assert len(comparison.regressions) > 0


# ---------------------------------------------------------------------------
# Score Aggregation
# ---------------------------------------------------------------------------

class TestScoreAggregation:
    """Tests for score aggregation in report building."""

    def test_overall_mean_is_average_of_all_scores(self, evaluator, good_outputs):
        report = evaluator.run_evaluation(
            outputs=good_outputs,
            prompt_version="v1.0",
            run_id="agg_test",
        )
        all_scores = [r.score for r in report.results]
        expected_mean = sum(all_scores) / len(all_scores)
        assert abs(report.overall_mean_score - expected_mean) < 0.01

    def test_rubric_summary_stats(self, evaluator, good_outputs):
        report = evaluator.run_evaluation(
            outputs=good_outputs,
            prompt_version="v1.0",
            run_id="stats_test",
        )
        for summary in report.rubric_summaries:
            assert summary.min_score <= summary.mean_score <= summary.max_score
            assert summary.std_dev >= 0
            assert summary.num_outputs == 1

    def test_weighted_mean_stored_in_metadata(self, evaluator, good_outputs):
        report = evaluator.run_evaluation(
            outputs=good_outputs,
            prompt_version="v1.0",
            run_id="weight_meta_test",
        )
        assert "rubric_weights" in report.metadata
        weights = report.metadata["rubric_weights"]
        assert weights.get("patient_safety") == 2.0
        assert weights.get("clinical_accuracy") == 1.5


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    """Tests for evaluation history retrieval."""

    def test_get_history(self, evaluator, good_outputs):
        evaluator.run_evaluation(outputs=good_outputs, prompt_version="v1.0", run_id="hist_1")
        evaluator.run_evaluation(outputs=good_outputs, prompt_version="v2.0", run_id="hist_2")

        history = evaluator.get_history(limit=10)
        assert len(history) == 2
        run_ids = {h["run_id"] for h in history}
        assert "hist_1" in run_ids
        assert "hist_2" in run_ids

    def test_get_report(self, evaluator, good_outputs):
        evaluator.run_evaluation(outputs=good_outputs, prompt_version="v1.0", run_id="rpt_test")
        report = evaluator.get_report("rpt_test")
        assert report is not None
        assert report.prompt_version == "v1.0"

    def test_get_nonexistent_report(self, evaluator):
        report = evaluator.get_report("nonexistent")
        assert report is None
