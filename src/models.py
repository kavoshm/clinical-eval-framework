"""
Pydantic Models for the Clinical Output Evaluation Framework

Defines the core data structures used throughout the evaluation pipeline:
    - EvalCriterion: a single evaluation criterion within a rubric
    - EvalRubric: a complete evaluation rubric with scoring levels
    - EvalResult: the result of evaluating one output against one rubric
    - EvalReport: aggregated results for a complete evaluation run
    - PromptVersion: metadata for tracking prompt versions
    - ClinicalOutput: a clinical LLM output to evaluate
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RubricCategory(str, Enum):
    """Categories of evaluation rubrics."""
    ACCURACY = "accuracy"
    SAFETY = "safety"
    COMPLETENESS = "completeness"
    APPROPRIATENESS = "appropriateness"


class OutputType(str, Enum):
    """Types of clinical outputs that can be evaluated."""
    SESSION_SUMMARY = "session_summary"
    TRIAGE_NARRATIVE = "triage_narrative"
    DISCHARGE_NOTE = "discharge_note"
    CLINICAL_LETTER = "clinical_letter"
    REFERRAL_NOTE = "referral_note"


class EvalStatus(str, Enum):
    """Status of an evaluation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Rubric Models
# ---------------------------------------------------------------------------

class ScoringLevel(BaseModel):
    """A single scoring level within a rubric (e.g., score=3, description='...')."""
    score: int
    label: str
    description: str
    examples: list[str] = Field(default_factory=list)


class EvalCriterion(BaseModel):
    """A single evaluation criterion within a rubric."""
    name: str
    description: str
    weight: float = 1.0


class EvalRubric(BaseModel):
    """
    A complete evaluation rubric.

    Loaded from YAML files in the rubrics/ directory.
    """
    name: str
    display_name: str
    description: str
    category: RubricCategory
    weight: float = 1.0
    version: str = "1.0"
    scale_min: int = 1
    scale_max: int = 5
    criteria: list[EvalCriterion] = Field(default_factory=list)
    scoring_levels: list[ScoringLevel] = Field(default_factory=list)
    evaluation_prompt_template: str = ""

    def get_scoring_level(self, score: int) -> Optional[ScoringLevel]:
        """Get the scoring level description for a given score."""
        for level in self.scoring_levels:
            if level.score == score:
                return level
        return None

    def format_rubric_for_prompt(self) -> str:
        """Format the rubric for inclusion in an LLM evaluation prompt."""
        lines = [
            f"Rubric: {self.display_name}",
            f"Description: {self.description}",
            "",
            "Scoring Scale:",
        ]
        for level in sorted(self.scoring_levels, key=lambda x: x.score):
            lines.append(f"  {level.score} - {level.label}: {level.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Clinical Output Models
# ---------------------------------------------------------------------------

class ClinicalOutput(BaseModel):
    """A clinical LLM output to be evaluated."""
    output_id: str
    text: str
    source_text: str = ""
    output_type: OutputType = OutputType.SESSION_SUMMARY
    model: str = "unknown"
    prompt_version: str = "unknown"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def word_count(self) -> int:
        return len(self.text.split())


# ---------------------------------------------------------------------------
# Evaluation Result Models
# ---------------------------------------------------------------------------

class CriterionScore(BaseModel):
    """Score for a single criterion within a rubric evaluation."""
    criterion_name: str
    score: float
    max_score: float = 5.0
    reasoning: str = ""


class EvalResult(BaseModel):
    """
    Result of evaluating one clinical output against one rubric.

    Stores the score, reasoning, and metadata needed for analysis.
    """
    result_id: str
    output_id: str
    rubric_name: str
    rubric_category: RubricCategory
    score: float
    max_score: float = 5.0
    reasoning: str = ""
    criterion_scores: list[CriterionScore] = Field(default_factory=list)
    judge_model: str = "gpt-4"
    prompt_version: str = "unknown"
    eval_run_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def normalized_score(self) -> float:
        """Score normalized to 0-1 range."""
        if self.max_score == 0:
            return 0.0
        return round(self.score / self.max_score, 4)

    @computed_field
    @property
    def score_label(self) -> str:
        """Human-readable score label."""
        if self.score >= 4.5:
            return "Excellent"
        elif self.score >= 3.5:
            return "Good"
        elif self.score >= 2.5:
            return "Adequate"
        elif self.score >= 1.5:
            return "Poor"
        else:
            return "Critical"


# ---------------------------------------------------------------------------
# Report Models
# ---------------------------------------------------------------------------

class RubricSummary(BaseModel):
    """Summary statistics for one rubric across all evaluated outputs."""
    rubric_name: str
    rubric_display_name: str
    mean_score: float
    min_score: float
    max_score: float
    std_dev: float
    num_outputs: int
    score_distribution: dict[str, int] = Field(default_factory=dict)


class EvalReport(BaseModel):
    """
    Aggregated evaluation report for a complete run.

    A run evaluates a set of clinical outputs across all rubrics.
    """
    report_id: str
    eval_run_id: str
    prompt_version: str
    judge_model: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    num_outputs: int = 0
    num_rubrics: int = 0
    total_evaluations: int = 0
    overall_mean_score: float = 0.0
    rubric_summaries: list[RubricSummary] = Field(default_factory=list)
    results: list[EvalResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def weighted_mean_score(self) -> float:
        """Weighted mean score across rubrics (using rubric weights)."""
        if not self.rubric_summaries:
            return 0.0
        # Weights are stored in metadata if available
        weights = self.metadata.get("rubric_weights", {})
        if not weights:
            return self.overall_mean_score

        total_weight = sum(weights.get(rs.rubric_name, 1.0) for rs in self.rubric_summaries)
        weighted_sum = sum(
            rs.mean_score * weights.get(rs.rubric_name, 1.0)
            for rs in self.rubric_summaries
        )
        return round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0


# ---------------------------------------------------------------------------
# Prompt Version Model
# ---------------------------------------------------------------------------

class PromptVersion(BaseModel):
    """Metadata for a prompt version, used for tracking and comparison."""
    version: str
    name: str
    description: str = ""
    author: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_eval_score: Optional[float] = None
    last_eval_date: Optional[datetime] = None
    parent_version: Optional[str] = None
    changes_from_parent: str = ""
    tags: list[str] = Field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Comparison Models
# ---------------------------------------------------------------------------

class VersionComparison(BaseModel):
    """Comparison between two evaluation runs (prompt versions)."""
    comparison_id: str
    version_a: str
    version_b: str
    run_id_a: str
    run_id_b: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Score deltas
    overall_delta: float = 0.0
    rubric_deltas: dict[str, float] = Field(default_factory=dict)

    # Regression detection
    regressions: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0

    @computed_field
    @property
    def recommendation(self) -> str:
        if self.has_regressions:
            return "HOLD - regressions detected"
        elif self.overall_delta > 0.1:
            return "DEPLOY - significant improvement"
        elif self.overall_delta > 0:
            return "DEPLOY - marginal improvement"
        else:
            return "HOLD - no improvement"
