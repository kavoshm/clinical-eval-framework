"""
Clinical Output Evaluation Framework — Main Entry Point

Demonstrates the complete evaluation pipeline:
    1. Load sample clinical outputs
    2. Load evaluation rubrics from YAML
    3. Run LLM-as-judge evaluation on all outputs x rubrics
    4. Store results in SQLite
    5. Generate evaluation reports
    6. Compare prompt versions
    7. Detect regressions

This script can be run standalone to see the full framework in action
without any API keys (uses simulated evaluation).
"""

from __future__ import annotations

import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure structured logging for the evaluation framework
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from models import ClinicalOutput, OutputType
from rubric_loader import RubricLoader
from judge import ClinicalJudge, JudgeConfig
from evaluator import ClinicalEvaluator
from storage import EvalStorage
from reporter import ReportGenerator


# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------

SOURCE_DOCUMENT = """
Patient: 67-year-old male with Type 2 diabetes mellitus (HbA1c 8.2%),
hypertension (on lisinopril 20mg daily), and recently diagnosed stage IIIA
non-small cell lung cancer. Patient presents for follow-up after first cycle
of carboplatin/pemetrexed chemotherapy. Reports moderate fatigue, decreased
appetite, and occasional nausea. Blood glucose readings have been elevated
(180-240 mg/dL fasting) since starting chemotherapy. Current medications:
metformin 1000mg BID, lisinopril 20mg daily, ondansetron 8mg PRN nausea.
Lab results: WBC 3.2 (low), platelets 145, creatinine 1.3 (slightly elevated).
"""


def get_v1_outputs() -> list[ClinicalOutput]:
    """Sample outputs from prompt v1.0 (baseline — less refined prompt)."""
    return [
        ClinicalOutput(
            output_id="v1_good_001",
            text=(
                "67M with T2DM, HTN, stage IIIA NSCLC s/p C1 carboplatin/pemetrexed.\n\n"
                "1. Oncology: Tolerating chemo. WBC 3.2 (low). Monitor. Continue ondansetron.\n"
                "2. Diabetes: Fasting BG 180-240. HbA1c 8.2%. Consider insulin.\n"
                "3. Renal: Cr 1.3 (elevated). Monitor.\n"
                "4. HTN: Stable on lisinopril 20mg.\n\n"
                "F/u 2 weeks."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v1.0",
        ),
        ClinicalOutput(
            output_id="v1_good_002",
            text=(
                "Assessment and Plan for 67-year-old male:\n\n"
                "Active Problems:\n"
                "1. Stage IIIA NSCLC - after first cycle carboplatin/pemetrexed, "
                "experiencing expected side effects (fatigue, nausea, decreased appetite). "
                "WBC 3.2 is concerning for neutropenia. Monitor closely.\n"
                "2. T2DM - poor control with HbA1c 8.2%, worsened by chemotherapy. "
                "Fasting glucose 180-240. May need insulin initiation.\n"
                "3. Renal function - creatinine 1.3, mildly elevated. Could be "
                "chemotherapy effect. Recheck in 1 week.\n\n"
                "Medications: metformin 1000mg BID, lisinopril 20mg daily, "
                "ondansetron 8mg PRN.\n\n"
                "Return: 2 weeks."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v1.0",
        ),
        ClinicalOutput(
            output_id="v1_good_003",
            text=(
                "Clinical Summary:\n"
                "67M with diabetes (HbA1c 8.2%), hypertension, and newly diagnosed "
                "NSCLC stage IIIA. Completed first cycle of carboplatin/pemetrexed. "
                "Reports fatigue, poor appetite, intermittent nausea.\n\n"
                "Labs: WBC 3.2 (low), platelets 145, creatinine 1.3 (elevated).\n"
                "Glucose: Fasting 180-240, poorly controlled.\n\n"
                "Plan:\n"
                "- Monitor blood counts, especially WBC\n"
                "- Consider adding insulin for glucose management\n"
                "- Recheck renal function; hold metformin if Cr rises\n"
                "- Continue ondansetron for nausea\n"
                "- Follow up in 2 weeks"
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v1.0",
        ),
        ClinicalOutput(
            output_id="v1_mediocre_004",
            text=(
                "Patient is a 67 year old male with diabetes and cancer. He had "
                "chemotherapy and is feeling tired and nauseous. His blood sugar is "
                "high. He should continue his medications and come back in 2 weeks."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-3.5-turbo",
            prompt_version="v1.0",
        ),
        ClinicalOutput(
            output_id="v1_mediocre_005",
            text=(
                "Follow-up visit for 67M with multiple conditions including diabetes "
                "and lung cancer. Patient reports side effects from chemotherapy. "
                "Some labs are abnormal. Continue current management and follow up."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.TRIAGE_NARRATIVE,
            model="gpt-3.5-turbo",
            prompt_version="v1.0",
        ),
        ClinicalOutput(
            output_id="v1_mediocre_006",
            text=(
                "67 year old man with Type 2 diabetes, high blood pressure, and stage 3 "
                "lung cancer. Had chemotherapy recently. Feeling tired and sick. Blood "
                "sugar levels are high. Some blood tests are not great. Needs to come "
                "back to see the doctor in a couple of weeks."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-3.5-turbo",
            prompt_version="v1.0",
        ),
        ClinicalOutput(
            output_id="v1_poor_007",
            text=(
                "Patient is doing well after chemotherapy. Blood work is normal. "
                "Continue current medications. No changes needed. Follow up in "
                "one month."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.DISCHARGE_NOTE,
            model="gpt-3.5-turbo",
            prompt_version="v1.0",
        ),
        ClinicalOutput(
            output_id="v1_poor_008",
            text=(
                "Patient tolerated chemotherapy. Labs within normal limits. "
                "Recommend increasing metformin to 2000mg BID. No follow-up needed "
                "for 3 months."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-3.5-turbo",
            prompt_version="v1.0",
        ),
        ClinicalOutput(
            output_id="v1_good_009",
            text=(
                "Assessment: 67M, complex patient with T2DM (HbA1c 8.2%), HTN (on "
                "lisinopril), and stage IIIA NSCLC post-C1 carboplatin/pemetrexed.\n\n"
                "Key Findings:\n"
                "- Chemo side effects: fatigue, appetite loss, nausea (expected)\n"
                "- WBC 3.2 (LOW) - leukopenia concern\n"
                "- Creatinine 1.3 (ELEVATED) - monitor for nephrotoxicity\n"
                "- Fasting BG 180-240 - diabetes worsening\n"
                "- Platelets 145 (borderline low)\n\n"
                "Plan: Monitor CBC closely. Consider G-CSF if WBC drops further. "
                "Evaluate insulin initiation. Recheck Cr in 1 week. Continue "
                "ondansetron, metformin 1000mg BID (monitor renal), lisinopril 20mg.\n"
                "Follow up: 2 weeks."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v1.0",
        ),
        ClinicalOutput(
            output_id="v1_good_010",
            text=(
                "Encounter Summary:\n\n"
                "Patient: 67-year-old male\n"
                "Diagnoses: T2DM, HTN, Stage IIIA NSCLC\n"
                "Treatment: C1 carboplatin/pemetrexed completed\n\n"
                "Symptoms: Moderate fatigue, decreased appetite, occasional nausea\n"
                "Medications: Metformin 1000mg BID, lisinopril 20mg daily, "
                "ondansetron 8mg PRN\n\n"
                "Pertinent Labs:\n"
                "- WBC: 3.2 (low - monitor for neutropenia)\n"
                "- Platelets: 145\n"
                "- Creatinine: 1.3 (slightly elevated)\n"
                "- HbA1c: 8.2% (above target)\n"
                "- Fasting BG: 180-240 mg/dL\n\n"
                "Plan:\n"
                "- Heme: Monitor WBC, consider growth factor support\n"
                "- Endo: Evaluate insulin, monitor renal before metformin changes\n"
                "- Renal: Recheck Cr in 1 week\n"
                "- F/u: 2 weeks"
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v1.0",
        ),
    ]


def get_v2_outputs() -> list[ClinicalOutput]:
    """
    Sample outputs from prompt v2.0 (improved prompt with better instructions).

    v2 prompt includes:
    - Explicit instruction to flag abnormal values
    - Required Assessment and Plan format
    - Instruction to never state 'doing well' without evidence
    - Required medication reconciliation
    """
    return [
        ClinicalOutput(
            output_id="v2_excellent_001",
            text=(
                "Assessment and Plan:\n"
                "67M with T2DM (HbA1c 8.2%), HTN, stage IIIA NSCLC s/p C1 "
                "carboplatin/pemetrexed.\n\n"
                "1. Hematology/Oncology:\n"
                "   - Tolerating chemo with expected toxicities\n"
                "   - WBC 3.2 (LOW) - leukopenia. Monitor closely, consider G-CSF "
                "if <2.0. Precautions for infection prevention.\n"
                "   - Platelets 145 (stable, monitor)\n"
                "   - Continue ondansetron 8mg PRN nausea\n\n"
                "2. Endocrine:\n"
                "   - Diabetes poorly controlled: HbA1c 8.2%, fasting BG 180-240\n"
                "   - Likely worsened by chemotherapy\n"
                "   - Consider basal insulin initiation\n"
                "   - Must assess renal function before metformin dose changes\n\n"
                "3. Renal:\n"
                "   - Creatinine 1.3 (ELEVATED) - possible chemo nephrotoxicity\n"
                "   - Recheck in 1 week\n"
                "   - HOLD metformin if Cr rises >1.5\n\n"
                "4. Cardiovascular:\n"
                "   - HTN stable on lisinopril 20mg daily\n"
                "   - Continue current dose\n\n"
                "Medications Reconciled:\n"
                "- Metformin 1000mg PO BID (monitor renal)\n"
                "- Lisinopril 20mg PO daily\n"
                "- Ondansetron 8mg PO PRN nausea\n\n"
                "Follow-up: 2 weeks or sooner if fever/worsening symptoms.\n"
                "Patient counseled on neutropenic precautions."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v2.0",
        ),
        ClinicalOutput(
            output_id="v2_excellent_002",
            text=(
                "Assessment and Plan - 67M - Oncology Follow-up\n\n"
                "Active Diagnoses:\n"
                "1. Stage IIIA NSCLC (dx recent)\n"
                "2. Type 2 Diabetes Mellitus (HbA1c 8.2%)\n"
                "3. Hypertension (controlled)\n\n"
                "Current Visit:\n"
                "Post-C1 carboplatin/pemetrexed. Expected toxicity profile: "
                "moderate fatigue, decreased appetite, intermittent nausea.\n\n"
                "ABNORMAL FINDINGS:\n"
                "- WBC 3.2 (LOW) -> leukopenia risk, monitor CBC, consider G-CSF\n"
                "- Creatinine 1.3 (ELEVATED) -> possible nephrotoxicity, recheck 1wk\n"
                "- Fasting BG 180-240 (ELEVATED) -> worsening diabetes control\n\n"
                "Plan by Problem:\n"
                "1. NSCLC: Continue chemo per oncology. Monitor counts closely.\n"
                "2. DM: Consider insulin. Monitor Cr before metformin adjustment.\n"
                "3. HTN: Continue lisinopril 20mg. Stable.\n"
                "4. Renal: Trend creatinine. Hold metformin if Cr >1.5.\n\n"
                "Medications: metformin 1000mg BID, lisinopril 20mg daily, "
                "ondansetron 8mg PRN\n\n"
                "F/u: 2 weeks. Urgent visit if fever >38C or bleeding."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v2.0",
        ),
        ClinicalOutput(
            output_id="v2_excellent_003",
            text=(
                "FOLLOW-UP NOTE\n\n"
                "Patient: 67M\n"
                "Reason for Visit: Post-chemotherapy follow-up\n\n"
                "Assessment and Plan:\n\n"
                "Problem List:\n"
                "1. Stage IIIA NSCLC on carboplatin/pemetrexed\n"
                "   - C1 completed. Side effects: fatigue (moderate), appetite "
                "decrease, nausea (controlled with ondansetron 8mg PRN)\n"
                "   - WBC 3.2 **LOW** - neutropenia watch. If WBC drops below 2.0 "
                "or ANC <1000, initiate G-CSF. Neutropenic precautions counseled.\n"
                "   - Platelets 145 - adequate, continue monitoring.\n\n"
                "2. T2DM (HbA1c 8.2%, fasting BG 180-240)\n"
                "   - Glycemic control worsened since chemo initiation\n"
                "   - Current: metformin 1000mg BID\n"
                "   - Plan: Start basal insulin (glargine 10 units nightly). "
                "Must verify renal function first.\n\n"
                "3. CKD concern (Cr 1.3 - **ELEVATED**)\n"
                "   - Possible chemotherapy-related nephrotoxicity\n"
                "   - If Cr >1.5: HOLD metformin\n"
                "   - Recheck BMP in 1 week\n\n"
                "4. HTN (controlled)\n"
                "   - Lisinopril 20mg daily - continue\n\n"
                "Follow-up: 2 weeks or sooner for fever, bleeding, or worsening renal function."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v2.0",
        ),
        ClinicalOutput(
            output_id="v2_good_004",
            text=(
                "A&P:\n"
                "67M with DM2, HTN, stage IIIA NSCLC post C1 carbo/pem.\n\n"
                "Chemo: Tolerating with fatigue, nausea. WBC 3.2 (low) - monitor.\n"
                "DM: Poorly controlled (HbA1c 8.2, fasting BG 180-240). Consider insulin.\n"
                "Renal: Cr 1.3 (elevated). Recheck. Watch metformin.\n"
                "HTN: Stable. Continue lisinopril.\n\n"
                "Meds: metformin 1000 BID, lisinopril 20 daily, ondansetron 8 PRN.\n"
                "F/u 2 weeks."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v2.0",
        ),
        ClinicalOutput(
            output_id="v2_good_005",
            text=(
                "Assessment and Plan:\n\n"
                "67-year-old male with T2DM, HTN, NSCLC.\n\n"
                "Post-chemo follow-up. WBC low at 3.2. Creatinine mildly elevated "
                "at 1.3. Blood glucose elevated (180-240 fasting). Symptoms of "
                "fatigue and nausea are expected chemotherapy side effects.\n\n"
                "Plan: Monitor blood counts. Adjust diabetes management. Check "
                "renal function again. Continue all current medications. Return "
                "in 2 weeks."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v2.0",
        ),
        ClinicalOutput(
            output_id="v2_mediocre_006",
            text=(
                "Patient has diabetes, hypertension, and lung cancer. Had first "
                "round of chemotherapy. Labs show low white blood cells and elevated "
                "creatinine. Blood sugars are high. Needs monitoring and follow-up "
                "in two weeks."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.TRIAGE_NARRATIVE,
            model="gpt-3.5-turbo",
            prompt_version="v2.0",
        ),
        ClinicalOutput(
            output_id="v2_mediocre_007",
            text=(
                "67M post-chemo. Multiple issues: WBC low, creatinine up, sugars "
                "high. Continue meds, follow up soon. Consider insulin for glucose."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-3.5-turbo",
            prompt_version="v2.0",
        ),
        ClinicalOutput(
            output_id="v2_good_008",
            text=(
                "Assessment and Plan:\n\n"
                "Oncology: 67M with stage IIIA NSCLC. Completed C1 "
                "carboplatin/pemetrexed. Side effects manageable. WBC 3.2 (low) - "
                "leukopenia risk. Platelets 145 stable.\n\n"
                "Endocrine: T2DM with HbA1c 8.2%, glucose 180-240 fasting. Poor "
                "control likely worsened by chemo. On metformin 1000mg BID. "
                "Consider adding insulin.\n\n"
                "Renal: Cr 1.3 elevated. May be chemo-related. Recheck 1 week. "
                "Watch metformin if Cr rises.\n\n"
                "CV: HTN on lisinopril 20mg. Stable.\n\n"
                "Follow-up: 2 weeks."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v2.0",
        ),
        ClinicalOutput(
            output_id="v2_good_009",
            text=(
                "Clinical Note:\n\n"
                "67M with T2DM (A1c 8.2), HTN, stage IIIA NSCLC. S/p first cycle "
                "carboplatin/pemetrexed.\n\n"
                "Reports: fatigue, decreased appetite, occasional nausea.\n"
                "Concerning labs: WBC 3.2 (low), Cr 1.3 (elevated).\n"
                "Glucose: 180-240 fasting (uncontrolled).\n"
                "Other: Plt 145 (adequate), on ondansetron PRN.\n\n"
                "Plan: Close CBC monitoring. Renal recheck 1 week. Diabetes "
                "escalation (consider insulin). Continue current meds. "
                "Return 2 weeks."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-4",
            prompt_version="v2.0",
        ),
        ClinicalOutput(
            output_id="v2_poor_010",
            text=(
                "Patient had chemo. Labs are abnormal. Needs follow up."
            ),
            source_text=SOURCE_DOCUMENT,
            output_type=OutputType.SESSION_SUMMARY,
            model="gpt-3.5-turbo",
            prompt_version="v2.0",
        ),
    ]


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline() -> None:
    """Run the complete evaluation pipeline demonstration."""

    print("=" * 70)
    print("CLINICAL OUTPUT EVALUATION FRAMEWORK")
    print("Complete Pipeline Demonstration")
    print("=" * 70)

    # Use in-memory database for demonstration
    db_path = ":memory:"

    # Initialize components
    rubric_dir = Path(__file__).parent.parent / "rubrics"
    rubric_loader = RubricLoader(rubrics_dir=rubric_dir)
    judge = ClinicalJudge(JudgeConfig(model="gpt-4", simulate=True))
    storage = EvalStorage(db_path=db_path)
    evaluator = ClinicalEvaluator(
        rubric_loader=rubric_loader,
        judge=judge,
        storage=storage,
        db_path=db_path,
    )
    reporter = ReportGenerator()

    # --- Step 1: Load rubrics ---
    print("\n--- Step 1: Loading Rubrics ---\n")
    rubrics = rubric_loader.load_all()
    for rubric in rubrics:
        print(f"  Loaded: {rubric.display_name} (weight={rubric.weight})")

    # --- Step 2: Evaluate v1.0 outputs ---
    print("\n--- Step 2: Evaluating Prompt v1.0 ---\n")
    v1_outputs = get_v1_outputs()
    print(f"  Evaluating {len(v1_outputs)} outputs against {len(rubrics)} rubrics...")

    report_v1 = evaluator.run_evaluation(
        outputs=v1_outputs,
        prompt_version="v1.0",
        run_id="run_v1_demo",
    )

    print(f"  Run ID: {report_v1.eval_run_id}")
    print(f"  Overall Score: {report_v1.overall_mean_score:.3f} / 5.0")
    for summary in report_v1.rubric_summaries:
        print(f"    {summary.rubric_display_name:30s} {summary.mean_score:.3f}")

    # --- Step 3: Evaluate v2.0 outputs ---
    print("\n--- Step 3: Evaluating Prompt v2.0 ---\n")
    v2_outputs = get_v2_outputs()
    print(f"  Evaluating {len(v2_outputs)} outputs against {len(rubrics)} rubrics...")

    report_v2 = evaluator.run_evaluation(
        outputs=v2_outputs,
        prompt_version="v2.0",
        run_id="run_v2_demo",
    )

    print(f"  Run ID: {report_v2.eval_run_id}")
    print(f"  Overall Score: {report_v2.overall_mean_score:.3f} / 5.0")
    for summary in report_v2.rubric_summaries:
        print(f"    {summary.rubric_display_name:30s} {summary.mean_score:.3f}")

    # --- Step 4: Compare versions ---
    print("\n--- Step 4: Version Comparison ---\n")
    comparison = evaluator.compare_runs("run_v1_demo", "run_v2_demo")

    print(f"  Overall Delta: {comparison.overall_delta:+.3f}")
    print(f"  Recommendation: {comparison.recommendation}")

    if comparison.improvements:
        print(f"\n  Improvements:")
        for imp in comparison.improvements:
            print(f"    + {imp}")

    if comparison.regressions:
        print(f"\n  Regressions:")
        for reg in comparison.regressions:
            print(f"    - {reg}")

    print(f"\n  Rubric Deltas:")
    for rubric_name, delta in comparison.rubric_deltas.items():
        direction = "^" if delta > 0 else "v" if delta < 0 else "="
        print(f"    {rubric_name:30s} {delta:+.4f} {direction}")

    # --- Step 5: Generate reports ---
    print("\n--- Step 5: Generating Reports ---\n")

    report_v1_md = reporter.generate_eval_report(report_v1)
    report_v2_md = reporter.generate_eval_report(report_v2)
    comparison_md = reporter.generate_comparison_report(comparison, report_v1, report_v2)

    # Save reports
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    (outputs_dir / "eval_report_v1.md").write_text(report_v1_md)
    (outputs_dir / "eval_report_v2.md").write_text(report_v2_md)
    (outputs_dir / "version_comparison.md").write_text(comparison_md)

    # Save raw scores as JSON
    scores_data = {
        "v1": {
            "run_id": report_v1.eval_run_id,
            "prompt_version": "v1.0",
            "overall_mean": report_v1.overall_mean_score,
            "results": [
                {
                    "output_id": r.output_id,
                    "rubric": r.rubric_name,
                    "score": r.score,
                    "label": r.score_label,
                    "reasoning": r.reasoning[:100],
                }
                for r in report_v1.results
            ],
        },
        "v2": {
            "run_id": report_v2.eval_run_id,
            "prompt_version": "v2.0",
            "overall_mean": report_v2.overall_mean_score,
            "results": [
                {
                    "output_id": r.output_id,
                    "rubric": r.rubric_name,
                    "score": r.score,
                    "label": r.score_label,
                    "reasoning": r.reasoning[:100],
                }
                for r in report_v2.results
            ],
        },
        "comparison": {
            "overall_delta": comparison.overall_delta,
            "rubric_deltas": comparison.rubric_deltas,
            "recommendation": comparison.recommendation,
        },
    }

    with open(outputs_dir / "sample_scores.json", "w") as f:
        json.dump(scores_data, f, indent=2, default=str)

    print(f"  Reports saved to {outputs_dir}/")
    print(f"    - eval_report_v1.md")
    print(f"    - eval_report_v2.md")
    print(f"    - version_comparison.md")
    print(f"    - sample_scores.json")

    # --- Step 6: Show history ---
    print("\n--- Step 6: Evaluation History ---\n")
    history = evaluator.get_history()
    for run in history:
        print(
            f"  {run['run_id']:30s} v={run['prompt_version']:6s} "
            f"score={run['overall_mean_score']:.3f} evals={run['total_evaluations']}"
        )

    print("\n" + "=" * 70)
    print("PIPELINE DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
