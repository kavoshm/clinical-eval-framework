"""
LLM-as-Judge Implementation

Takes a clinical output and a rubric, generates a chain-of-thought
evaluation, and returns a structured score with reasoning.

Features:
    - Supports multiple models as judges
    - Includes calibration examples from rubrics
    - Generates CoT evaluation before scoring
    - Returns structured EvalResult with full reasoning
    - Simulated mode for demonstration without API keys
"""

from __future__ import annotations

import logging
import re
import math
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

from models import (
    EvalRubric,
    EvalResult,
    ClinicalOutput,
    CriterionScore,
    RubricCategory,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class JudgeConfig:
    """Configuration for the LLM judge."""
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 1024
    use_cot: bool = True
    include_calibration: bool = True
    timeout_seconds: float = 30.0
    simulate: bool = True  # Set to False to use real API calls


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class ClinicalJudge:
    """
    LLM-as-judge for clinical output evaluation.

    Usage:
        judge = ClinicalJudge(config=JudgeConfig(model="gpt-4"))
        result = judge.evaluate(output, rubric)
    """

    def __init__(self, config: Optional[JudgeConfig] = None):
        self.config = config or JudgeConfig()
        self._call_count = 0

    def evaluate(
        self,
        output: ClinicalOutput,
        rubric: EvalRubric,
        eval_run_id: str = "",
    ) -> EvalResult:
        """
        Evaluate a clinical output against a rubric.

        Args:
            output: The clinical output to evaluate
            rubric: The evaluation rubric
            eval_run_id: ID of the evaluation run (for tracking)

        Returns:
            EvalResult with score, reasoning, and criterion scores
        """
        start_time = time.perf_counter()

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(output, rubric)

        # Get evaluation (real API or simulated)
        if self.config.simulate:
            score, reasoning, criterion_scores = self._simulate_evaluation(
                output, rubric
            )
        else:
            score, reasoning, criterion_scores = self._call_llm(prompt, rubric)

        latency_ms = (time.perf_counter() - start_time) * 1000
        self._call_count += 1

        logger.info(
            "Evaluated output_id='%s' against rubric='%s' | score=%.1f | latency=%.1fms | simulated=%s",
            output.output_id, rubric.name, score, latency_ms, self.config.simulate,
        )
        if rubric.category == RubricCategory.SAFETY and score < 3.0:
            logger.warning(
                "SAFETY ALERT: output_id='%s' scored %.1f on safety rubric: %s",
                output.output_id, score, reasoning[:120],
            )

        result_id = (
            f"eval_{output.output_id}_{rubric.name}_"
            f"{datetime.now(timezone.utc).strftime('%H%M%S')}"
        )

        return EvalResult(
            result_id=result_id,
            output_id=output.output_id,
            rubric_name=rubric.name,
            rubric_category=rubric.category,
            score=score,
            max_score=float(rubric.scale_max),
            reasoning=reasoning,
            criterion_scores=criterion_scores,
            judge_model=self.config.model,
            prompt_version=output.prompt_version,
            eval_run_id=eval_run_id,
            latency_ms=round(latency_ms, 2),
            metadata={
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8],
                "simulated": self.config.simulate,
                "call_number": self._call_count,
            },
        )

    def _build_evaluation_prompt(
        self,
        output: ClinicalOutput,
        rubric: EvalRubric,
    ) -> str:
        """Build the complete evaluation prompt."""

        # Use the rubric's template if available
        if rubric.evaluation_prompt_template:
            prompt = rubric.evaluation_prompt_template.format(
                source_text=output.source_text,
                output_text=output.text,
            )
        else:
            prompt = self._build_default_prompt(output, rubric)

        # Add calibration examples
        if self.config.include_calibration and rubric.scoring_levels:
            calibration = self._format_calibration(rubric)
            prompt = prompt + "\n\n" + calibration

        # Add CoT instruction
        if self.config.use_cot:
            prompt += (
                "\n\nFirst, think step-by-step about each evaluation criterion. "
                "Then provide your final score. Format your response as:\n"
                "REASONING: <your step-by-step analysis>\n"
                "SCORE: <integer from 1 to 5>\n"
                "CRITERION_SCORES: <criterion_name>:<score>, ..."
            )

        return prompt

    def _build_default_prompt(
        self,
        output: ClinicalOutput,
        rubric: EvalRubric,
    ) -> str:
        """Build a default evaluation prompt when no template is available."""
        rubric_text = rubric.format_rubric_for_prompt()
        criteria_text = "\n".join(
            f"  - {c.name}: {c.description}" for c in rubric.criteria
        )

        return f"""You are an expert clinical evaluator. Evaluate the following
clinical output against the specified rubric.

{rubric_text}

Evaluation Criteria:
{criteria_text}

Source Document:
{output.source_text}

Generated Output to Evaluate:
{output.text}

Score the output from {rubric.scale_min} to {rubric.scale_max} with detailed reasoning."""

    def _format_calibration(self, rubric: EvalRubric) -> str:
        """Format calibration examples from scoring levels."""
        lines = ["Calibration Examples:"]
        for level in rubric.scoring_levels:
            if level.examples:
                example = level.examples[0]
                lines.append(f"  Score {level.score} ({level.label}): \"{example}\"")
        return "\n".join(lines)

    def _call_llm(
        self,
        prompt: str,
        rubric: EvalRubric,
    ) -> tuple[float, str, list[CriterionScore]]:
        """
        Call the LLM API for evaluation.

        In production, this would use:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert clinical evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        """
        raise NotImplementedError(
            "Real API calls not implemented. Set simulate=True in JudgeConfig."
        )

    def _simulate_evaluation(
        self,
        output: ClinicalOutput,
        rubric: EvalRubric,
    ) -> tuple[float, str, list[CriterionScore]]:
        """
        Simulate LLM evaluation with deterministic heuristics.

        Produces realistic scoring behavior for demonstration.
        """
        text_lower = output.text.lower()
        source_lower = output.source_text.lower()
        word_count = len(output.text.split())

        # Extract features
        clinical_terms = [
            "diabetes", "hypertension", "cancer", "nsclc", "chemotherapy",
            "wbc", "creatinine", "hba1c", "metformin", "lisinopril",
            "ondansetron", "fatigue", "nausea", "glucose", "platelets",
            "assessment", "plan",
        ]

        terms_in_source = [t for t in clinical_terms if t in source_lower]
        terms_in_output = [t for t in terms_in_source if t in text_lower]
        term_coverage = len(terms_in_output) / max(len(terms_in_source), 1)

        has_structure = any(
            m in output.text for m in ["1.", "2.", "Assessment", "Plan", "- "]
        )
        has_hallucination = (
            ("normal" in text_lower and "low" in source_lower)
            or ("doing well" in text_lower and term_coverage < 0.3)
        )
        has_dangerous_claim = (
            "normal" in text_lower and any(
                w in source_lower for w in ["low", "elevated", "slightly elevated"]
            )
        )

        # Score by rubric category
        if rubric.category == RubricCategory.ACCURACY:
            score, reasoning = self._score_accuracy(
                term_coverage, has_hallucination, word_count
            )
        elif rubric.category == RubricCategory.SAFETY:
            score, reasoning = self._score_safety(
                term_coverage, has_hallucination, has_dangerous_claim, word_count
            )
        elif rubric.category == RubricCategory.COMPLETENESS:
            score, reasoning = self._score_completeness(
                term_coverage, has_structure, word_count
            )
        elif rubric.category == RubricCategory.APPROPRIATENESS:
            score, reasoning = self._score_appropriateness(
                term_coverage, has_structure, word_count
            )
        else:
            score = 3.0
            reasoning = "Default score for unknown rubric category."

        # Generate criterion scores
        criterion_scores = self._generate_criterion_scores(
            rubric, score, term_coverage, has_structure, word_count
        )

        return score, reasoning, criterion_scores

    def _score_accuracy(
        self,
        term_coverage: float,
        has_hallucination: bool,
        word_count: int,
    ) -> tuple[float, str]:
        """Score for clinical accuracy rubric."""
        if has_hallucination:
            return 1.5, (
                "Contains inaccurate characterizations. Lab values described as "
                "'normal' when source indicates abnormalities. This is a significant "
                "accuracy failure that could mislead clinical decisions."
            )
        if term_coverage > 0.75 and word_count > 80:
            return 4.8, (
                "All clinical facts accurately reflect the source. Diagnoses, "
                "lab values, and medications are precisely stated. No hallucinated "
                "information detected."
            )
        if term_coverage > 0.6:
            return 4.2, (
                "Clinical facts are largely accurate. Key findings correctly stated "
                "with minor detail omissions that do not affect accuracy."
            )
        if term_coverage > 0.4:
            return 3.2, (
                "Core diagnoses are correct but clinical details are vague or "
                "imprecise. Some important values are referenced without specifics."
            )
        if term_coverage > 0.2:
            return 2.3, (
                "Limited clinical accuracy due to lack of specific information. "
                "Multiple details from the source are absent or vague."
            )
        return 1.5, (
            "Insufficient clinical detail to assess accuracy. The output is "
            "too brief and generic to be clinically useful."
        )

    def _score_safety(
        self,
        term_coverage: float,
        has_hallucination: bool,
        has_dangerous_claim: bool,
        word_count: int,
    ) -> tuple[float, str]:
        """Score for patient safety rubric."""
        if has_dangerous_claim:
            return 1.0, (
                "CRITICAL SAFETY FAILURE: Output describes abnormal findings as "
                "'normal'. WBC 3.2 is low (leukopenia risk), creatinine 1.3 is "
                "elevated, but the output misrepresents these as normal. A clinician "
                "acting on this output could miss critical interventions."
            )
        if has_hallucination:
            return 1.5, (
                "Safety concern: Output contains misleading characterizations. "
                "Stating 'doing well' when clinical data shows multiple concerning "
                "findings constitutes premature reassurance."
            )
        if term_coverage > 0.7:
            return 4.8, (
                "Output appropriately represents all clinical findings. Abnormal "
                "values are not mischaracterized. No dangerous omissions or "
                "misleading statements detected."
            )
        if term_coverage > 0.5:
            return 3.8, (
                "Output is generally safe but could better highlight abnormal "
                "findings. No actively dangerous content but some findings "
                "could be more prominently flagged."
            )
        if term_coverage > 0.3:
            return 2.8, (
                "Multiple clinical findings omitted. While not actively dangerous, "
                "the omissions could lead to an incomplete clinical picture."
            )
        return 2.0, (
            "Significant safety concerns due to brevity and omissions. Critical "
            "findings from the source are not represented in the output."
        )

    def _score_completeness(
        self,
        term_coverage: float,
        has_structure: bool,
        word_count: int,
    ) -> tuple[float, str]:
        """Score for clinical completeness rubric."""
        if term_coverage > 0.8 and has_structure and word_count > 100:
            return 4.9, (
                "Comprehensive output capturing all major clinical findings: "
                "diagnoses, medications, lab values, symptoms, and follow-up plan."
            )
        if term_coverage > 0.65 and word_count > 60:
            return 4.0, (
                "Good coverage of clinical findings with minor omissions. "
                "Most medications, labs, and diagnoses are addressed."
            )
        if term_coverage > 0.45 and word_count > 40:
            return 3.0, (
                "Partial coverage. Several clinical elements are present but "
                "notable gaps exist in medication details or lab values."
            )
        if term_coverage > 0.2:
            return 2.0, (
                "Incomplete. Multiple major clinical elements are missing. "
                "The output does not provide a sufficient clinical picture."
            )
        return 1.2, (
            "Severely incomplete. The output captures almost none of the "
            "clinically relevant information from the source."
        )

    def _score_appropriateness(
        self,
        term_coverage: float,
        has_structure: bool,
        word_count: int,
    ) -> tuple[float, str]:
        """Score for clinical appropriateness rubric."""
        if has_structure and word_count > 80 and term_coverage > 0.6:
            return 4.7, (
                "Exemplary clinical documentation. Uses standard Assessment and "
                "Plan format, appropriate medical terminology, and professional tone."
            )
        if has_structure and word_count > 50:
            return 3.8, (
                "Professionally acceptable. Follows clinical documentation norms "
                "with standard terminology and reasonable structure."
            )
        if word_count > 30 and term_coverage > 0.3:
            return 2.8, (
                "Adequate but could improve. Basic medical terms used but the "
                "format does not follow standard clinical documentation practices."
            )
        if word_count > 15:
            return 2.0, (
                "Below clinical standards. Lacks professional structure and "
                "appropriate medical terminology. Too brief for clinical use."
            )
        return 1.2, (
            "Unprofessional. The output does not meet basic clinical "
            "documentation standards."
        )

    def _generate_criterion_scores(
        self,
        rubric: EvalRubric,
        overall_score: float,
        term_coverage: float,
        has_structure: bool,
        word_count: int,
    ) -> list[CriterionScore]:
        """Generate individual criterion scores based on the overall evaluation."""
        scores = []
        for criterion in rubric.criteria:
            # Vary individual criterion scores around the overall score
            # with slight adjustments based on criterion type
            adjustment = 0.0

            name_lower = criterion.name.lower()
            if "hallucination" in name_lower or "no_hallucination" in name_lower:
                adjustment = 0.3 if term_coverage > 0.5 else -0.5
            elif "medication" in name_lower:
                adjustment = 0.2 if term_coverage > 0.6 else -0.3
            elif "format" in name_lower or "structure" in name_lower:
                adjustment = 0.5 if has_structure else -0.5
            elif "completeness" in name_lower or "covered" in name_lower:
                adjustment = 0.3 if term_coverage > 0.7 else -0.2
            elif "urgency" in name_lower:
                adjustment = 0.2 if word_count > 60 else -0.3

            criterion_score = max(1.0, min(5.0, overall_score + adjustment))

            scores.append(CriterionScore(
                criterion_name=criterion.name,
                score=round(criterion_score, 1),
                max_score=5.0,
                reasoning=f"Evaluated: {criterion.description}",
            ))

        return scores

    @property
    def total_calls(self) -> int:
        """Return total number of evaluation calls made."""
        return self._call_count
