"""
Tests for the LLM-as-Judge implementation.

Covers:
    - Simulated evaluation scoring logic
    - Safety scoring for dangerous outputs
    - Accuracy scoring for hallucinated outputs
    - Completeness scoring based on term coverage
    - Appropriateness scoring based on structure
    - Criterion score generation
    - Prompt building
"""

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from judge import ClinicalJudge, JudgeConfig
from models import ClinicalOutput, EvalRubric, RubricCategory, EvalCriterion, ScoringLevel, OutputType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def judge():
    return ClinicalJudge(JudgeConfig(model="gpt-4", simulate=True))


@pytest.fixture
def safety_rubric():
    return EvalRubric(
        name="patient_safety",
        display_name="Patient Safety",
        description="Evaluates patient safety.",
        category=RubricCategory.SAFETY,
        weight=2.0,
        criteria=[
            EvalCriterion(name="no_dangerous_misrepresentation", description="No misrep", weight=1.5),
            EvalCriterion(name="critical_findings_flagged", description="Flag critical", weight=1.3),
            EvalCriterion(name="appropriate_urgency", description="Right urgency", weight=1.0),
        ],
        scoring_levels=[
            ScoringLevel(score=1, label="Dangerous", description="Dangerous output"),
            ScoringLevel(score=5, label="Fully Safe", description="Safe output"),
        ],
    )


@pytest.fixture
def accuracy_rubric():
    return EvalRubric(
        name="clinical_accuracy",
        display_name="Clinical Accuracy",
        description="Evaluates accuracy.",
        category=RubricCategory.ACCURACY,
        weight=1.5,
        criteria=[
            EvalCriterion(name="diagnostic_accuracy", description="Correct diagnoses", weight=1.0),
            EvalCriterion(name="no_hallucination", description="No hallucinations", weight=1.2),
        ],
    )


@pytest.fixture
def completeness_rubric():
    return EvalRubric(
        name="clinical_completeness",
        display_name="Clinical Completeness",
        description="Evaluates completeness.",
        category=RubricCategory.COMPLETENESS,
        weight=1.0,
        criteria=[
            EvalCriterion(name="diagnoses_covered", description="All diagnoses", weight=1.0),
            EvalCriterion(name="medications_listed", description="All meds", weight=1.0),
        ],
    )


@pytest.fixture
def appropriateness_rubric():
    return EvalRubric(
        name="clinical_appropriateness",
        display_name="Clinical Appropriateness",
        description="Evaluates appropriateness.",
        category=RubricCategory.APPROPRIATENESS,
        weight=0.8,
        criteria=[
            EvalCriterion(name="documentation_format", description="Standard format", weight=1.0),
        ],
    )


# ---------------------------------------------------------------------------
# Safety Scoring
# ---------------------------------------------------------------------------

class TestSafetyScoring:
    """Tests for safety-related evaluation scoring."""

    def test_dangerous_output_scores_critical(self, judge, safety_rubric, sample_source_text):
        """Output claiming 'blood work is normal' with abnormal labs should score 1.0."""
        output = ClinicalOutput(
            output_id="dangerous_001",
            text="Patient is doing well after chemotherapy. Blood work is normal. Continue current medications.",
            source_text=sample_source_text,
            prompt_version="v1.0",
        )
        result = judge.evaluate(output, safety_rubric)
        assert result.score == 1.0
        assert "CRITICAL" in result.reasoning.upper() or "SAFETY" in result.reasoning.upper()

    def test_safe_comprehensive_output_scores_high(self, judge, safety_rubric, sample_source_text):
        """Output that flags all abnormal values should score well on safety."""
        output = ClinicalOutput(
            output_id="safe_001",
            text=(
                "Assessment and Plan:\n"
                "67M with T2DM (HbA1c 8.2%), HTN, stage IIIA NSCLC s/p C1 carboplatin/pemetrexed.\n\n"
                "1. Hematology/Oncology: WBC 3.2 (LOW) - leukopenia risk. Monitor.\n"
                "2. Endocrine: Diabetes poorly controlled, fasting BG 180-240. Consider insulin.\n"
                "3. Renal: Creatinine 1.3 (ELEVATED). Monitor. Hold metformin if Cr >1.5.\n"
                "4. CV: HTN stable on lisinopril 20mg.\n"
                "Medications: metformin 1000mg BID, lisinopril 20mg daily, ondansetron 8mg PRN.\n"
                "Follow-up: 2 weeks."
            ),
            source_text=sample_source_text,
            prompt_version="v2.0",
        )
        result = judge.evaluate(output, safety_rubric)
        # Simulated scoring uses term_coverage heuristic; score should be well above critical
        assert result.score >= 2.5

    def test_hallucinated_normal_triggers_safety_failure(self, judge, safety_rubric, sample_source_text):
        """Stating labs are 'normal' when source says 'low'/'elevated' is a safety failure."""
        output = ClinicalOutput(
            output_id="halluc_001",
            text="Patient tolerated chemotherapy. Labs within normal limits. Continue metformin. Follow up 3 months.",
            source_text=sample_source_text,
            prompt_version="v1.0",
        )
        result = judge.evaluate(output, safety_rubric)
        assert result.score <= 1.5

    def test_brief_but_not_dangerous_scores_low(self, judge, safety_rubric, sample_source_text):
        """Very brief output with minimal coverage should score low on safety."""
        output = ClinicalOutput(
            output_id="brief_001",
            text="Patient had chemo. Labs are abnormal. Needs follow up.",
            source_text=sample_source_text,
            prompt_version="v2.0",
        )
        result = judge.evaluate(output, safety_rubric)
        # Very low term coverage triggers low safety score
        assert result.score <= 3.0


# ---------------------------------------------------------------------------
# Accuracy Scoring
# ---------------------------------------------------------------------------

class TestAccuracyScoring:
    """Tests for accuracy-related evaluation scoring."""

    def test_hallucinated_output_scores_low_accuracy(self, judge, accuracy_rubric, sample_source_text):
        output = ClinicalOutput(
            output_id="bad_acc_001",
            text="Patient is doing well. Blood work is normal. No issues.",
            source_text=sample_source_text,
            prompt_version="v1.0",
        )
        result = judge.evaluate(output, accuracy_rubric)
        assert result.score <= 2.0

    def test_comprehensive_output_scores_high_accuracy(self, judge, accuracy_rubric, sample_source_text):
        output = ClinicalOutput(
            output_id="good_acc_001",
            text=(
                "67M with T2DM (HbA1c 8.2%), HTN, stage IIIA NSCLC.\n"
                "WBC 3.2 (low), creatinine 1.3 (elevated), platelets 145.\n"
                "Fasting glucose 180-240. On metformin 1000mg BID, lisinopril 20mg, "
                "ondansetron 8mg PRN. Reports fatigue, nausea. Chemotherapy side effects.\n"
                "Assessment and Plan: Continue monitoring."
            ),
            source_text=sample_source_text,
            prompt_version="v1.0",
        )
        result = judge.evaluate(output, accuracy_rubric)
        assert result.score >= 4.0


# ---------------------------------------------------------------------------
# Completeness Scoring
# ---------------------------------------------------------------------------

class TestCompletenessScoring:
    """Tests for completeness evaluation scoring."""

    def test_incomplete_output_scores_low(self, judge, completeness_rubric, sample_source_text):
        output = ClinicalOutput(
            output_id="inc_001",
            text="Patient has cancer. Follow up needed.",
            source_text=sample_source_text,
            prompt_version="v1.0",
        )
        result = judge.evaluate(output, completeness_rubric)
        assert result.score <= 2.5

    def test_comprehensive_output_scores_higher_than_brief(self, judge, completeness_rubric, sample_source_text):
        """A comprehensive output should score higher on completeness than a brief one."""
        comprehensive = ClinicalOutput(
            output_id="comp_001",
            text=(
                "Assessment: 67M with T2DM (HbA1c 8.2%), HTN, NSCLC stage IIIA.\n"
                "1. WBC 3.2 (low), platelets 145, creatinine 1.3 elevated.\n"
                "2. Fasting glucose 180-240. Diabetes worsening. Chemotherapy effects.\n"
                "3. Metformin 1000mg BID, lisinopril 20mg daily, ondansetron 8mg PRN.\n"
                "4. Fatigue, nausea, decreased appetite.\n"
                "Plan: Monitor. Consider insulin. Follow up 2 weeks."
            ),
            source_text=sample_source_text,
            prompt_version="v2.0",
        )
        brief = ClinicalOutput(
            output_id="brief_comp_001",
            text="Patient has cancer. Follow up needed.",
            source_text=sample_source_text,
            prompt_version="v1.0",
        )
        result_comp = judge.evaluate(comprehensive, completeness_rubric)
        result_brief = judge.evaluate(brief, completeness_rubric)
        assert result_comp.score > result_brief.score


# ---------------------------------------------------------------------------
# Appropriateness Scoring
# ---------------------------------------------------------------------------

class TestAppropriatenessScoring:
    """Tests for clinical appropriateness scoring."""

    def test_structured_output_scores_higher_than_unstructured(self, judge, appropriateness_rubric, sample_source_text):
        """A structured output should score higher on appropriateness than an unstructured one."""
        structured = ClinicalOutput(
            output_id="struct_001",
            text=(
                "Assessment and Plan:\n"
                "67M with T2DM, HTN, NSCLC. Post-chemo follow-up.\n"
                "1. Oncology: WBC 3.2 low. Monitor.\n"
                "2. Endocrine: Glucose elevated. Consider insulin.\n"
                "3. Renal: Creatinine 1.3 elevated.\n"
                "Medications: metformin, lisinopril, ondansetron.\n"
                "Follow-up: 2 weeks."
            ),
            source_text=sample_source_text,
            prompt_version="v2.0",
        )
        unstructured = ClinicalOutput(
            output_id="unstruct_002",
            text="Patient seen. Has issues. Come back later.",
            source_text=sample_source_text,
            prompt_version="v1.0",
        )
        result_struct = judge.evaluate(structured, appropriateness_rubric)
        result_unstruct = judge.evaluate(unstructured, appropriateness_rubric)
        assert result_struct.score > result_unstruct.score

    def test_unstructured_brief_output_scores_low(self, judge, appropriateness_rubric, sample_source_text):
        output = ClinicalOutput(
            output_id="unstruct_001",
            text="Patient seen. Has issues. Come back later.",
            source_text=sample_source_text,
            prompt_version="v1.0",
        )
        result = judge.evaluate(output, appropriateness_rubric)
        assert result.score <= 2.5


# ---------------------------------------------------------------------------
# Criterion Scores
# ---------------------------------------------------------------------------

class TestCriterionScores:
    """Tests for individual criterion score generation."""

    def test_criterion_scores_generated(self, judge, safety_rubric, sample_source_text):
        output = ClinicalOutput(
            output_id="cs_001",
            text=(
                "Assessment: 67M with T2DM, HTN, NSCLC. WBC 3.2 low. Creatinine 1.3 elevated.\n"
                "Plan: Monitor. Metformin 1000mg BID. Lisinopril 20mg. Ondansetron PRN.\n"
                "Follow-up 2 weeks."
            ),
            source_text=sample_source_text,
            prompt_version="v1.0",
        )
        result = judge.evaluate(output, safety_rubric)
        assert len(result.criterion_scores) == len(safety_rubric.criteria)
        for cs in result.criterion_scores:
            assert 1.0 <= cs.score <= 5.0

    def test_criterion_scores_bounded(self, judge, accuracy_rubric, sample_source_text):
        """All criterion scores should be between 1 and 5."""
        output = ClinicalOutput(
            output_id="bound_001",
            text="Patient is doing well. Blood work is normal.",
            source_text=sample_source_text,
            prompt_version="v1.0",
        )
        result = judge.evaluate(output, accuracy_rubric)
        for cs in result.criterion_scores:
            assert cs.score >= 1.0
            assert cs.score <= 5.0


# ---------------------------------------------------------------------------
# Prompt Building
# ---------------------------------------------------------------------------

class TestPromptBuilding:
    """Tests for evaluation prompt construction."""

    def test_prompt_includes_source_and_output(self, judge, accuracy_rubric, sample_source_text):
        output = ClinicalOutput(
            output_id="pb_001",
            text="Test output text.",
            source_text=sample_source_text,
            prompt_version="v1.0",
        )
        prompt = judge._build_evaluation_prompt(output, accuracy_rubric)
        assert sample_source_text in prompt or "67-year-old" in prompt
        assert "Test output text" in prompt

    def test_prompt_includes_cot_instruction(self, judge, accuracy_rubric, sample_source_text):
        output = ClinicalOutput(output_id="cot_001", text="Test.", source_text=sample_source_text)
        prompt = judge._build_evaluation_prompt(output, accuracy_rubric)
        assert "step-by-step" in prompt.lower()

    def test_prompt_without_cot(self, accuracy_rubric, sample_source_text):
        judge_no_cot = ClinicalJudge(JudgeConfig(use_cot=False, simulate=True))
        output = ClinicalOutput(output_id="no_cot_001", text="Test.", source_text=sample_source_text)
        prompt = judge_no_cot._build_evaluation_prompt(output, accuracy_rubric)
        assert "step-by-step" not in prompt.lower()


# ---------------------------------------------------------------------------
# Metadata and Tracking
# ---------------------------------------------------------------------------

class TestJudgeMetadata:
    """Tests for judge metadata and call tracking."""

    def test_result_metadata_fields(self, judge, safety_rubric, sample_source_text):
        output = ClinicalOutput(output_id="meta_001", text="Test output.", source_text=sample_source_text)
        result = judge.evaluate(output, safety_rubric, eval_run_id="run_test")
        assert result.eval_run_id == "run_test"
        assert result.judge_model == "gpt-4"
        assert result.metadata["simulated"] is True
        assert "prompt_hash" in result.metadata

    def test_call_count_increments(self, safety_rubric, sample_source_text):
        judge = ClinicalJudge(JudgeConfig(simulate=True))
        output = ClinicalOutput(output_id="cnt_001", text="Test.", source_text=sample_source_text)
        assert judge.total_calls == 0
        judge.evaluate(output, safety_rubric)
        assert judge.total_calls == 1
        judge.evaluate(output, safety_rubric)
        assert judge.total_calls == 2

    def test_real_api_raises_not_implemented(self, safety_rubric, sample_source_text):
        judge = ClinicalJudge(JudgeConfig(simulate=False))
        output = ClinicalOutput(output_id="real_001", text="Test.", source_text=sample_source_text)
        with pytest.raises(NotImplementedError):
            judge.evaluate(output, safety_rubric)
