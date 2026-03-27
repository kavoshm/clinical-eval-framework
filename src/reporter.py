"""
Reporting Module — Generate Markdown Reports

Produces professional evaluation reports including:
    - Summary statistics per rubric
    - Score distributions
    - Rubric-level breakdowns
    - Regression detection between prompt versions
    - Version comparison reports
"""

from __future__ import annotations

import logging
import math
from typing import Optional
from datetime import datetime, timezone

from models import EvalReport, VersionComparison, RubricSummary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generate markdown evaluation reports."""

    def generate_eval_report(self, report: EvalReport) -> str:
        """Generate a complete markdown evaluation report."""
        logger.info(
            "Generating evaluation report for run '%s' (prompt_version='%s')",
            report.eval_run_id, report.prompt_version,
        )
        lines = [
            f"# Evaluation Report: {report.prompt_version}",
            "",
            f"**Run ID:** `{report.eval_run_id}`",
            f"**Timestamp:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Judge Model:** {report.judge_model}",
            f"**Outputs Evaluated:** {report.num_outputs}",
            f"**Rubrics Applied:** {report.num_rubrics}",
            f"**Total Evaluations:** {report.total_evaluations}",
            "",
            "---",
            "",
            "## Overall Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Overall Mean Score | **{report.overall_mean_score:.3f}** / 5.0 |",
            f"| Weighted Mean Score | **{report.weighted_mean_score:.3f}** / 5.0 |",
            f"| Quality Rating | {self._score_to_rating(report.overall_mean_score)} |",
            "",
        ]

        # Rubric summaries table
        lines.extend([
            "## Rubric-Level Breakdown",
            "",
            "| Rubric | Mean | Min | Max | Std Dev | Status |",
            "|--------|------|-----|-----|---------|--------|",
        ])

        for summary in report.rubric_summaries:
            status = self._score_to_status(summary.mean_score)
            lines.append(
                f"| {summary.rubric_display_name} | "
                f"{summary.mean_score:.3f} | {summary.min_score:.3f} | "
                f"{summary.max_score:.3f} | {summary.std_dev:.3f} | "
                f"{status} |"
            )

        # Score distributions
        lines.extend(["", "## Score Distributions", ""])

        for summary in report.rubric_summaries:
            lines.append(f"### {summary.rubric_display_name}")
            lines.append("")
            lines.append("```")
            lines.append(self._render_histogram(summary))
            lines.append("```")
            lines.append("")

        # Individual results
        lines.extend(["## Detailed Results", ""])

        # Group by output
        outputs_seen = {}
        for result in report.results:
            if result.output_id not in outputs_seen:
                outputs_seen[result.output_id] = []
            outputs_seen[result.output_id].append(result)

        for output_id, results in outputs_seen.items():
            mean_score = sum(r.score for r in results) / len(results)
            lines.append(f"### Output: `{output_id}` (mean: {mean_score:.2f})")
            lines.append("")
            lines.append("| Rubric | Score | Label | Key Finding |")
            lines.append("|--------|-------|-------|-------------|")

            for result in results:
                reasoning_short = result.reasoning[:80].replace("|", "/")
                lines.append(
                    f"| {result.rubric_name} | {result.score:.1f}/5 | "
                    f"{result.score_label} | {reasoning_short}... |"
                )

            lines.append("")

        # Alerts
        safety_alerts = [
            r for r in report.results
            if r.rubric_name == "patient_safety" and r.score < 3.0
        ]

        if safety_alerts:
            logger.warning(
                "Report for run '%s' contains %d safety alert(s)",
                report.eval_run_id, len(safety_alerts),
            )
            lines.extend([
                "## Safety Alerts",
                "",
                "The following outputs scored below 3.0 on patient safety:",
                "",
            ])
            for alert in safety_alerts:
                lines.append(
                    f"- **{alert.output_id}**: Score {alert.score:.1f}/5 -- "
                    f"{alert.reasoning[:100]}"
                )
            lines.append("")

        lines.extend([
            "---",
            f"*Report generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        ])

        return "\n".join(lines)

    def generate_comparison_report(
        self,
        comparison: VersionComparison,
        report_a: Optional[EvalReport] = None,
        report_b: Optional[EvalReport] = None,
    ) -> str:
        """Generate a markdown version comparison report."""
        lines = [
            f"# Version Comparison: {comparison.version_a} vs {comparison.version_b}",
            "",
            f"**Comparison ID:** `{comparison.comparison_id}`",
            f"**Timestamp:** {comparison.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Recommendation:** **{comparison.recommendation}**",
            "",
            "---",
            "",
            "## Overall Score Change",
            "",
            f"| Version | Score | Delta |",
            f"|---------|-------|-------|",
        ]

        if report_a and report_b:
            lines.append(
                f"| {comparison.version_a} (baseline) | "
                f"{report_a.overall_mean_score:.3f} | -- |"
            )
            lines.append(
                f"| {comparison.version_b} (candidate) | "
                f"{report_b.overall_mean_score:.3f} | "
                f"{comparison.overall_delta:+.3f} |"
            )
        else:
            lines.append(
                f"| {comparison.version_a} -> {comparison.version_b} | "
                f"-- | {comparison.overall_delta:+.3f} |"
            )

        # Rubric deltas
        lines.extend([
            "",
            "## Rubric-Level Changes",
            "",
            "| Rubric | Delta | Direction |",
            "|--------|-------|-----------|",
        ])

        for rubric_name, delta in sorted(comparison.rubric_deltas.items()):
            direction = "IMPROVED" if delta > 0 else "REGRESSED" if delta < 0 else "NO CHANGE"
            arrow = "^" if delta > 0 else "v" if delta < 0 else "="
            lines.append(f"| {rubric_name} | {delta:+.3f} | {arrow} {direction} |")

        # Regressions
        if comparison.regressions:
            lines.extend([
                "",
                "## Regressions Detected",
                "",
                "The following rubrics showed significant score decreases:",
                "",
            ])
            for regression in comparison.regressions:
                lines.append(f"- {regression}")

        # Improvements
        if comparison.improvements:
            lines.extend([
                "",
                "## Improvements",
                "",
                "The following rubrics showed significant score increases:",
                "",
            ])
            for improvement in comparison.improvements:
                lines.append(f"- {improvement}")

        # Side-by-side rubric comparison
        if report_a and report_b:
            lines.extend([
                "",
                "## Side-by-Side Rubric Comparison",
                "",
                f"| Rubric | {comparison.version_a} | {comparison.version_b} | Delta |",
                f"|--------|{'-' * len(comparison.version_a) + '--'}|{'-' * len(comparison.version_b) + '--'}|-------|",
            ])

            summaries_a = {s.rubric_name: s for s in report_a.rubric_summaries}
            summaries_b = {s.rubric_name: s for s in report_b.rubric_summaries}

            for name in sorted(set(summaries_a.keys()) | set(summaries_b.keys())):
                score_a = summaries_a.get(name)
                score_b = summaries_b.get(name)
                sa = f"{score_a.mean_score:.3f}" if score_a else "--"
                sb = f"{score_b.mean_score:.3f}" if score_b else "--"
                delta = comparison.rubric_deltas.get(name, 0.0)
                lines.append(f"| {name} | {sa} | {sb} | {delta:+.3f} |")

        lines.extend([
            "",
            "---",
            f"*Report generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        ])

        return "\n".join(lines)

    # --- Helpers ---

    @staticmethod
    def _score_to_rating(score: float) -> str:
        """Convert a numeric score to a human-readable rating."""
        if score >= 4.5:
            return "Excellent"
        elif score >= 3.5:
            return "Good"
        elif score >= 2.5:
            return "Adequate"
        elif score >= 1.5:
            return "Poor"
        else:
            return "Critical"

    @staticmethod
    def _score_to_status(score: float) -> str:
        """Convert a score to a status indicator."""
        if score >= 4.0:
            return "PASS"
        elif score >= 3.0:
            return "WARN"
        else:
            return "FAIL"

    @staticmethod
    def _render_histogram(summary: RubricSummary) -> str:
        """Render a simple ASCII histogram of score distribution."""
        max_count = max(summary.score_distribution.values()) if summary.score_distribution else 1
        lines = [f"  {summary.rubric_display_name} Score Distribution"]
        lines.append(f"  {'Score':<8s} {'Count':<6s} {'Bar'}")

        for score_str in ["1", "2", "3", "4", "5"]:
            count = summary.score_distribution.get(score_str, 0)
            bar_len = int((count / max_count) * 30) if max_count > 0 else 0
            bar = "#" * bar_len
            lines.append(f"  {score_str:<8s} {count:<6d} {bar}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------

def generate_report(report: EvalReport) -> str:
    """Generate a markdown evaluation report."""
    generator = ReportGenerator()
    return generator.generate_eval_report(report)


def generate_comparison(
    comparison: VersionComparison,
    report_a: Optional[EvalReport] = None,
    report_b: Optional[EvalReport] = None,
) -> str:
    """Generate a markdown version comparison report."""
    generator = ReportGenerator()
    return generator.generate_comparison_report(comparison, report_a, report_b)
