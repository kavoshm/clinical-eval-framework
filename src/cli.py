"""
CLI Interface for the Clinical Output Evaluation Framework

Commands:
    eval     — Run evaluation on a set of clinical outputs
    compare  — Compare two evaluation runs
    report   — Generate an evaluation report
    history  — Show evaluation run history
"""

from __future__ import annotations

import logging
import sys
import json
from pathlib import Path
from typing import Optional

# Configure structured logging for the evaluation framework
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Attempt to import click; provide fallback if not installed
try:
    import click
except ImportError:
    print("Error: 'click' package not installed. Run: pip install click")
    sys.exit(1)

from models import ClinicalOutput, OutputType
from rubric_loader import RubricLoader
from judge import ClinicalJudge, JudgeConfig
from evaluator import ClinicalEvaluator
from storage import EvalStorage
from reporter import ReportGenerator


# ---------------------------------------------------------------------------
# CLI Group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--db",
    default="eval_results.db",
    help="Path to the SQLite database file.",
)
@click.option(
    "--rubrics-dir",
    default=None,
    help="Path to the rubrics directory.",
)
@click.pass_context
def cli(ctx: click.Context, db: str, rubrics_dir: Optional[str]) -> None:
    """Clinical Output Evaluation Framework

    Evaluate clinical LLM outputs against structured rubrics using
    LLM-as-judge scoring with chain-of-thought reasoning.
    """
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db
    ctx.obj["rubrics_dir"] = rubrics_dir


# ---------------------------------------------------------------------------
# eval command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--input",
    "input_dir",
    required=True,
    help="Directory containing clinical output JSON files.",
)
@click.option(
    "--prompt-version",
    default="unknown",
    help="Prompt version that generated these outputs.",
)
@click.option(
    "--rubrics",
    default=None,
    help="Comma-separated list of rubric names (default: all).",
)
@click.option(
    "--model",
    default="gpt-4",
    help="Judge model to use (default: gpt-4).",
)
@click.option(
    "--output-report",
    default=None,
    help="Path to write the evaluation report markdown.",
)
@click.pass_context
def eval_cmd(
    ctx: click.Context,
    input_dir: str,
    prompt_version: str,
    rubrics: Optional[str],
    model: str,
    output_report: Optional[str],
) -> None:
    """Run evaluation on a set of clinical outputs."""
    db_path = ctx.obj["db_path"]
    rubrics_dir = ctx.obj.get("rubrics_dir")

    click.echo(f"Loading outputs from: {input_dir}")

    # Load clinical outputs
    outputs = _load_outputs_from_dir(input_dir)
    if not outputs:
        click.echo("Error: No output files found in the input directory.", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(outputs)} clinical outputs")

    # Parse rubric names
    rubric_names = rubrics.split(",") if rubrics else None

    # Initialize evaluator
    rubric_loader = RubricLoader(
        rubrics_dir=Path(rubrics_dir) if rubrics_dir else None
    )
    judge = ClinicalJudge(JudgeConfig(model=model, simulate=True))
    storage = EvalStorage(db_path=db_path)
    evaluator = ClinicalEvaluator(
        rubric_loader=rubric_loader,
        judge=judge,
        storage=storage,
    )

    # Run evaluation
    click.echo(f"Running evaluation with prompt version: {prompt_version}")
    report = evaluator.run_evaluation(
        outputs=outputs,
        prompt_version=prompt_version,
        rubric_names=rubric_names,
    )

    # Display summary
    click.echo(f"\nEvaluation Complete!")
    click.echo(f"  Run ID:          {report.eval_run_id}")
    click.echo(f"  Overall Score:   {report.overall_mean_score:.3f} / 5.0")
    click.echo(f"  Total Evals:     {report.total_evaluations}")

    for summary in report.rubric_summaries:
        click.echo(
            f"  {summary.rubric_display_name:30s}  "
            f"mean={summary.mean_score:.3f}  "
            f"min={summary.min_score:.3f}  max={summary.max_score:.3f}"
        )

    # Write report if requested
    if output_report:
        generator = ReportGenerator()
        report_md = generator.generate_eval_report(report)
        Path(output_report).write_text(report_md)
        click.echo(f"\nReport written to: {output_report}")


# ---------------------------------------------------------------------------
# compare command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("run_id_a")
@click.argument("run_id_b")
@click.option(
    "--threshold",
    default=0.2,
    help="Regression detection threshold (default: 0.2).",
)
@click.option(
    "--output",
    "output_file",
    default=None,
    help="Path to write the comparison report.",
)
@click.pass_context
def compare(
    ctx: click.Context,
    run_id_a: str,
    run_id_b: str,
    threshold: float,
    output_file: Optional[str],
) -> None:
    """Compare two evaluation runs."""
    db_path = ctx.obj["db_path"]

    storage = EvalStorage(db_path=db_path)
    evaluator = ClinicalEvaluator(storage=storage)

    try:
        comparison = evaluator.compare_runs(run_id_a, run_id_b, threshold)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Display comparison
    click.echo(f"\nVersion Comparison: {comparison.version_a} vs {comparison.version_b}")
    click.echo(f"  Overall Delta:   {comparison.overall_delta:+.3f}")
    click.echo(f"  Recommendation:  {comparison.recommendation}")

    if comparison.regressions:
        click.echo(f"\n  Regressions:")
        for reg in comparison.regressions:
            click.echo(f"    - {reg}")

    if comparison.improvements:
        click.echo(f"\n  Improvements:")
        for imp in comparison.improvements:
            click.echo(f"    + {imp}")

    click.echo(f"\n  Rubric Deltas:")
    for rubric, delta in comparison.rubric_deltas.items():
        direction = "^" if delta > 0 else "v" if delta < 0 else "="
        click.echo(f"    {rubric:30s} {delta:+.3f} {direction}")

    # Write report if requested
    if output_file:
        generator = ReportGenerator()
        report_a = storage.get_report(run_id_a)
        report_b = storage.get_report(run_id_b)
        report_md = generator.generate_comparison_report(
            comparison, report_a, report_b
        )
        Path(output_file).write_text(report_md)
        click.echo(f"\nComparison report written to: {output_file}")


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("run_id")
@click.option(
    "--output",
    "output_file",
    default=None,
    help="Path to write the report (default: stdout).",
)
@click.pass_context
def report(ctx: click.Context, run_id: str, output_file: Optional[str]) -> None:
    """Generate an evaluation report for a run."""
    db_path = ctx.obj["db_path"]
    storage = EvalStorage(db_path=db_path)

    eval_report = storage.get_report(run_id)
    if not eval_report:
        click.echo(f"Error: Run not found: {run_id}", err=True)
        sys.exit(1)

    generator = ReportGenerator()
    report_md = generator.generate_eval_report(eval_report)

    if output_file:
        Path(output_file).write_text(report_md)
        click.echo(f"Report written to: {output_file}")
    else:
        click.echo(report_md)


# ---------------------------------------------------------------------------
# history command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--limit",
    default=10,
    help="Number of runs to display (default: 10).",
)
@click.pass_context
def history(ctx: click.Context, limit: int) -> None:
    """Show evaluation run history."""
    db_path = ctx.obj["db_path"]
    storage = EvalStorage(db_path=db_path)

    runs = storage.get_run_history(limit)

    if not runs:
        click.echo("No evaluation runs found.")
        return

    click.echo(f"\n{'Run ID':<40s} {'Version':<12s} {'Score':>7s} {'Evals':>6s} {'Timestamp'}")
    click.echo(f"{'-' * 40} {'-' * 12} {'-' * 7} {'-' * 6} {'-' * 20}")

    for run in runs:
        click.echo(
            f"{run['run_id']:<40s} {run['prompt_version']:<12s} "
            f"{run['overall_mean_score']:7.3f} {run['total_evaluations']:6d} "
            f"{run['timestamp'][:19]}"
        )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _load_outputs_from_dir(dir_path: str) -> list[ClinicalOutput]:
    """Load clinical outputs from JSON files in a directory."""
    outputs = []
    input_path = Path(dir_path)

    if not input_path.exists():
        return outputs

    for json_file in sorted(input_path.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Support both single output and list of outputs
            if isinstance(data, list):
                for item in data:
                    outputs.append(_parse_output(item))
            else:
                outputs.append(_parse_output(data))
        except (json.JSONDecodeError, KeyError) as e:
            click.echo(f"Warning: Could not parse {json_file}: {e}", err=True)

    return outputs


def _parse_output(data: dict) -> ClinicalOutput:
    """Parse a dictionary into a ClinicalOutput model."""
    output_type = data.get("output_type", "session_summary")
    try:
        otype = OutputType(output_type)
    except ValueError:
        otype = OutputType.SESSION_SUMMARY

    return ClinicalOutput(
        output_id=data.get("output_id", "unknown"),
        text=data["text"],
        source_text=data.get("source_text", ""),
        output_type=otype,
        model=data.get("model", "unknown"),
        prompt_version=data.get("prompt_version", "unknown"),
        metadata=data.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
