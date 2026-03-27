#!/usr/bin/env python3
"""
Generate All Figures for the Clinical Output Evaluation Framework

Reads outputs/sample_scores.json and produces publication-quality visualizations
for the README and documentation.

Figures generated:
    1. eval_architecture.png       -- System architecture diagram
    2. rubric_scores_v1.png        -- Grouped bar chart of v1 scores
    3. rubric_scores_v2.png        -- Grouped bar chart of v2 scores
    4. version_comparison.png      -- Radar/spider chart v1 vs v2
    5. score_distribution.png      -- Violin/box plot per rubric
    6. improvement_delta.png       -- Bar chart of score deltas v1->v2
    7. safety_heatmap.png          -- Heatmap of safety scores

Color palette (dark theme):
    Primary:    #9b6b9e (muted purple)
    Danger:     #b85450 (clinical red)
    Good:       #4a7c59 (clinical green)
    Warning:    #c47e3a (amber)
    Background: #1a1a2e (dark navy)
    Surface:    #16213e (dark blue-gray)
    Text:       #e8e8e8 (light gray)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORES_PATH = REPO_ROOT / "outputs" / "sample_scores.json"
OUTPUT_DIR = REPO_ROOT / "docs" / "images"

# Color palette
COLORS = {
    "primary": "#9b6b9e",
    "danger": "#b85450",
    "good": "#4a7c59",
    "warning": "#c47e3a",
    "bg": "#1a1a2e",
    "surface": "#16213e",
    "text": "#e8e8e8",
    "text_dim": "#8888aa",
    "grid": "#2a2a4a",
    "v1": "#c47e3a",
    "v2": "#9b6b9e",
    "accent1": "#5b8a72",
    "accent2": "#d4a574",
    "accent3": "#7b9ec4",
    "accent4": "#c4857b",
}

RUBRIC_COLORS = {
    "clinical_accuracy": "#9b6b9e",
    "patient_safety": "#b85450",
    "clinical_completeness": "#4a7c59",
    "clinical_appropriateness": "#c47e3a",
}

RUBRIC_LABELS = {
    "clinical_accuracy": "Clinical\nAccuracy",
    "patient_safety": "Patient\nSafety",
    "clinical_completeness": "Clinical\nCompleteness",
    "clinical_appropriateness": "Clinical\nAppropriateness",
}

RUBRIC_LABELS_SINGLE = {
    "clinical_accuracy": "Clinical Accuracy",
    "patient_safety": "Patient Safety",
    "clinical_completeness": "Clinical Completeness",
    "clinical_appropriateness": "Clinical Appropriateness",
}

DPI = 180


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_scores() -> dict:
    """Load sample_scores.json."""
    with open(SCORES_PATH, "r") as f:
        return json.load(f)


def setup_dark_theme():
    """Configure matplotlib for dark theme."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text_dim"],
        "ytick.color": COLORS["text_dim"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "legend.facecolor": COLORS["surface"],
        "legend.edgecolor": COLORS["grid"],
        "legend.labelcolor": COLORS["text"],
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def pivot_scores(results: list[dict]) -> dict[str, dict[str, float]]:
    """
    Pivot results into {output_id: {rubric: score}}.
    """
    pivot = {}
    for r in results:
        oid = r["output_id"]
        if oid not in pivot:
            pivot[oid] = {}
        pivot[oid][r["rubric"]] = r["score"]
    return pivot


def save_fig(fig, name: str):
    """Save figure to output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1: Architecture Diagram
# ---------------------------------------------------------------------------

def generate_architecture_diagram():
    """Generate the system architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Title
    ax.text(
        7, 5.15, "Clinical Output Evaluation Framework",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color=COLORS["text"],
        path_effects=[pe.withStroke(linewidth=2, foreground=COLORS["bg"])],
    )
    ax.text(
        7, 4.75, "Rubric-Based LLM-as-Judge Pipeline with Version Tracking",
        ha="center", va="center", fontsize=10, color=COLORS["text_dim"],
    )

    # Box positions: (x_center, y_center, width, height)
    boxes = [
        (1.4, 2.7, 2.2, 2.6, "Clinical\nOutputs", COLORS["accent3"],
         ["Session Summary", "Triage Note", "Discharge Note", "Clinical Letter"]),
        (4.2, 2.7, 2.2, 2.6, "Rubric\nEngine", COLORS["primary"],
         ["accuracy.yaml", "safety.yaml", "completeness.yaml", "appropriate.yaml"]),
        (7.0, 2.7, 2.2, 2.6, "LLM-as-Judge", COLORS["warning"],
         ["Build CoT Prompt", "Call LLM Judge", "Extract Score", "Parse Reasoning"]),
        (9.8, 2.7, 2.2, 2.6, "SQLite\nStorage", COLORS["accent1"],
         ["eval_runs", "eval_results", "eval_reports", "comparisons"]),
        (12.6, 2.7, 2.2, 2.6, "Reporting", COLORS["danger"],
         ["eval_report.md", "comparison.md", "scores.json", "regression alerts"]),
    ]

    for (cx, cy, w, h, title, color, items) in boxes:
        # Box background
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color + "22",
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)

        # Title
        ax.text(
            cx, cy + h / 2 - 0.42, title,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=color,
        )

        # Items
        for i, item in enumerate(items):
            ax.text(
                cx, cy + 0.1 - i * 0.38, item,
                ha="center", va="center", fontsize=7.5,
                color=COLORS["text_dim"],
            )

    # Arrows between boxes
    arrow_props = dict(
        arrowstyle="->,head_width=0.3,head_length=0.15",
        color=COLORS["text_dim"],
        linewidth=1.5,
        connectionstyle="arc3,rad=0.0",
    )

    arrow_positions = [
        (2.55, 2.7, 3.05, 2.7),
        (5.35, 2.7, 5.85, 2.7),
        (8.15, 2.7, 8.65, 2.7),
        (10.95, 2.7, 11.45, 2.7),
    ]

    for (x1, y1, x2, y2) in arrow_positions:
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=arrow_props,
        )

    # Bottom annotation
    file_labels = [
        (1.4, "data/sample_outputs/"),
        (4.2, "rubrics/*.yaml"),
        (7.0, "src/judge.py\nsrc/evaluator.py"),
        (9.8, "src/storage.py"),
        (12.6, "src/reporter.py"),
    ]

    for (cx, label) in file_labels:
        ax.text(
            cx, 0.95, label,
            ha="center", va="center", fontsize=7,
            color=COLORS["text_dim"], fontstyle="italic",
        )

    save_fig(fig, "eval_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2 & 3: Grouped Bar Charts (v1 and v2)
# ---------------------------------------------------------------------------

def generate_rubric_scores_bar(data: dict, version: str, filename: str):
    """Generate grouped bar chart of scores across rubrics for a version."""
    results = data[version]["results"]
    pivot = pivot_scores(results)

    output_ids = list(pivot.keys())
    rubrics = ["clinical_accuracy", "patient_safety",
               "clinical_completeness", "clinical_appropriateness"]

    n_outputs = len(output_ids)
    n_rubrics = len(rubrics)
    x = np.arange(n_outputs)
    bar_width = 0.18

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    for i, rubric in enumerate(rubrics):
        scores = [pivot[oid].get(rubric, 0) for oid in output_ids]
        offset = (i - n_rubrics / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, scores, bar_width,
            label=RUBRIC_LABELS_SINGLE[rubric],
            color=RUBRIC_COLORS[rubric],
            edgecolor=RUBRIC_COLORS[rubric],
            alpha=0.85,
            linewidth=0.5,
        )
        # Value labels on bars
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{score:.1f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color=COLORS["text_dim"],
                )

    # Threshold lines
    ax.axhline(y=4.0, color=COLORS["good"], linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=3.0, color=COLORS["warning"], linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=2.0, color=COLORS["danger"], linestyle="--", alpha=0.4, linewidth=1)

    ax.text(13.6, 4.1, "PASS", fontsize=7, color=COLORS["good"], alpha=0.6)
    ax.text(13.6, 3.1, "WARN", fontsize=7, color=COLORS["warning"], alpha=0.6)
    ax.text(13.6, 2.1, "FAIL", fontsize=7, color=COLORS["danger"], alpha=0.6)

    # Clean short labels for x-axis
    short_labels = []
    for oid in output_ids:
        # Shorten: out_001 -> #01, v2_excellent_001 -> #01
        parts = oid.split("_")
        num = parts[-1] if parts[-1].isdigit() else parts[-1][-3:]
        short_labels.append(f"#{num}")

    ax.set_xlabel("Clinical Output", fontsize=12)
    ax.set_ylabel("Score (1-5)", fontsize=12)
    prompt_ver = data[version]["prompt_version"]
    ax.set_title(
        f"Evaluation Scores by Rubric -- Prompt {prompt_ver}",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylim(0, 5.6)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.2)

    save_fig(fig, filename)


# ---------------------------------------------------------------------------
# Figure 4: Version Comparison Radar/Spider Chart
# ---------------------------------------------------------------------------

def generate_version_comparison_radar(data: dict):
    """Generate radar/spider chart comparing v1 and v2 average scores."""
    rubrics = ["clinical_accuracy", "patient_safety",
               "clinical_completeness", "clinical_appropriateness"]

    v1_means = data["v1"]["rubric_means"]
    v2_means = data["v2"]["rubric_means"]

    v1_values = [v1_means[r] for r in rubrics]
    v2_values = [v2_means[r] for r in rubrics]

    # Close the polygon
    v1_values += v1_values[:1]
    v2_values += v2_values[:1]

    labels = [RUBRIC_LABELS_SINGLE[r] for r in rubrics]
    num_vars = len(rubrics)

    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor(COLORS["surface"])
    fig.patch.set_facecolor(COLORS["bg"])

    # Draw radar grid
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color=COLORS["text_dim"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, color=COLORS["text"], fontweight="bold")

    # Grid styling
    ax.spines["polar"].set_color(COLORS["grid"])
    ax.grid(color=COLORS["grid"], alpha=0.3)
    ax.tick_params(axis="x", pad=15)

    # Plot v1
    ax.plot(angles, v1_values, "o-", linewidth=2.5, label=f"v1.0 (mean {data['v1']['overall_mean']:.3f})",
            color=COLORS["v1"], markersize=8)
    ax.fill(angles, v1_values, alpha=0.15, color=COLORS["v1"])

    # Plot v2
    ax.plot(angles, v2_values, "o-", linewidth=2.5, label=f"v2.0 (mean {data['v2']['overall_mean']:.3f})",
            color=COLORS["v2"], markersize=8)
    ax.fill(angles, v2_values, alpha=0.15, color=COLORS["v2"])

    # Title and legend
    ax.set_title(
        "Version Comparison: v1.0 vs v2.0\nAverage Scores per Rubric",
        fontsize=15, fontweight="bold", pad=30,
        color=COLORS["text"],
    )
    ax.legend(
        loc="lower right", bbox_to_anchor=(1.25, -0.05),
        fontsize=11, framealpha=0.8,
    )

    save_fig(fig, "version_comparison.png")


# ---------------------------------------------------------------------------
# Figure 5: Score Distribution (Box + Strip Plot)
# ---------------------------------------------------------------------------

def generate_score_distribution(data: dict):
    """Generate box plot with strip overlay showing score distribution per rubric."""
    rubrics = ["clinical_accuracy", "patient_safety",
               "clinical_completeness", "clinical_appropriateness"]

    # Collect all scores per rubric across both versions
    all_data = []
    for version in ["v1", "v2"]:
        for r in data[version]["results"]:
            all_data.append({
                "Rubric": RUBRIC_LABELS_SINGLE[r["rubric"]],
                "Score": r["score"],
                "Version": data[version]["prompt_version"],
                "rubric_key": r["rubric"],
            })

    # Sort by rubric order
    rubric_order = [RUBRIC_LABELS_SINGLE[r] for r in rubrics]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Use seaborn for box plots
    import pandas as pd
    df = pd.DataFrame(all_data)

    # Color palette for boxplot
    palette = {RUBRIC_LABELS_SINGLE[r]: RUBRIC_COLORS[r] for r in rubrics}

    # Box plot
    bp = sns.boxplot(
        data=df, x="Rubric", y="Score", hue="Rubric",
        order=rubric_order,
        palette=palette,
        width=0.5,
        linewidth=1.5,
        fliersize=0,
        boxprops=dict(alpha=0.6),
        medianprops=dict(color=COLORS["text"], linewidth=2),
        whiskerprops=dict(color=COLORS["text_dim"]),
        capprops=dict(color=COLORS["text_dim"]),
        ax=ax,
        legend=False,
    )

    # Strip (jitter) overlay
    sns.stripplot(
        data=df, x="Rubric", y="Score", hue="Version",
        order=rubric_order,
        dodge=True,
        jitter=0.15,
        size=6,
        alpha=0.7,
        palette={"v1.0": COLORS["v1"], "v2.0": COLORS["v2"]},
        edgecolor=COLORS["text_dim"],
        linewidth=0.5,
        ax=ax,
    )

    # Threshold lines
    ax.axhline(y=4.0, color=COLORS["good"], linestyle="--", alpha=0.3, linewidth=1)
    ax.axhline(y=3.0, color=COLORS["warning"], linestyle="--", alpha=0.3, linewidth=1)
    ax.axhline(y=2.0, color=COLORS["danger"], linestyle="--", alpha=0.3, linewidth=1)

    ax.set_ylim(0.5, 5.5)
    ax.set_ylabel("Score (1-5)", fontsize=12)
    ax.set_xlabel("")
    ax.set_title(
        "Score Distribution per Rubric (v1.0 + v2.0 Combined)",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax.legend(title="Prompt Version", loc="lower left", fontsize=10, title_fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    save_fig(fig, "score_distribution.png")


# ---------------------------------------------------------------------------
# Figure 6: Improvement Delta Bar Chart
# ---------------------------------------------------------------------------

def generate_improvement_delta(data: dict):
    """Generate bar chart showing score improvement from v1 to v2 per rubric."""
    comparison = data["comparison"]
    rubric_deltas = comparison["rubric_deltas"]

    rubrics = ["clinical_accuracy", "patient_safety",
               "clinical_completeness", "clinical_appropriateness"]

    labels = [RUBRIC_LABELS_SINGLE[r] for r in rubrics]
    deltas = [rubric_deltas[r] for r in rubrics]

    # Color bars by positive/negative
    bar_colors = [COLORS["good"] if d > 0 else COLORS["danger"] for d in deltas]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(rubrics))
    bars = ax.bar(
        x, deltas, width=0.55,
        color=bar_colors, alpha=0.85,
        edgecolor=[c for c in bar_colors],
        linewidth=1.5,
    )

    # Value labels
    for bar, delta in zip(bars, deltas):
        y_pos = bar.get_height() + 0.02 if delta > 0 else bar.get_height() - 0.04
        va = "bottom" if delta > 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}",
            ha="center", va=va, fontsize=13, fontweight="bold",
            color=COLORS["text"],
        )

    # Zero line
    ax.axhline(y=0, color=COLORS["text_dim"], linewidth=1, alpha=0.5)

    # Highlight largest improvement
    max_idx = np.argmax(deltas)
    bars[max_idx].set_edgecolor(COLORS["text"])
    bars[max_idx].set_linewidth(2.5)
    ax.annotate(
        "Largest\nimprovement",
        xy=(max_idx, deltas[max_idx]),
        xytext=(max_idx + 0.8, deltas[max_idx] + 0.12),
        fontsize=9, color=COLORS["text"],
        arrowprops=dict(arrowstyle="->", color=COLORS["text_dim"], linewidth=1),
        ha="center",
    )

    # Overall delta annotation
    overall = comparison["overall_delta"]
    ax.text(
        0.98, 0.95,
        f"Overall: +{overall:.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=13, fontweight="bold", color=COLORS["primary"],
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=COLORS["primary"] + "22",
            edgecolor=COLORS["primary"],
            linewidth=1.5,
        ),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score Delta (v2.0 - v1.0)", fontsize=12)
    ax.set_title(
        "Score Improvement from v1.0 to v2.0",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(-0.2, max(deltas) + 0.35)

    save_fig(fig, "improvement_delta.png")


# ---------------------------------------------------------------------------
# Figure 7: Safety Heatmap
# ---------------------------------------------------------------------------

def generate_safety_heatmap(data: dict):
    """Generate heatmap of safety scores across all outputs, highlighting danger."""
    rubrics = ["clinical_accuracy", "patient_safety",
               "clinical_completeness", "clinical_appropriateness"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"wspace": 0.35})

    for idx, (version, ax) in enumerate(zip(["v1", "v2"], axes)):
        results = data[version]["results"]
        pivot = pivot_scores(results)

        output_ids = list(pivot.keys())
        n_outputs = len(output_ids)

        # Build matrix
        matrix = np.zeros((n_outputs, len(rubrics)))
        for i, oid in enumerate(output_ids):
            for j, rubric in enumerate(rubrics):
                matrix[i, j] = pivot[oid].get(rubric, 0)

        # Short output labels
        short_labels = []
        for oid in output_ids:
            parts = oid.split("_")
            num = parts[-1] if parts[-1].isdigit() else parts[-1][-3:]
            short_labels.append(f"#{num}")

        rubric_labels = [RUBRIC_LABELS_SINGLE[r] for r in rubrics]

        # Custom colormap: red for low, yellow for mid, green for high
        from matplotlib.colors import LinearSegmentedColormap
        cmap_colors = [COLORS["danger"], COLORS["warning"], "#e8d44d", COLORS["good"]]
        cmap = LinearSegmentedColormap.from_list("safety_cmap", cmap_colors, N=256)

        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=1, vmax=5)

        # Annotate cells
        for i in range(n_outputs):
            for j in range(len(rubrics)):
                score = matrix[i, j]
                text_color = COLORS["text"] if score < 3.5 else COLORS["bg"]
                fontweight = "bold" if score < 2.5 else "normal"

                # Add warning symbol for critical scores
                label = f"{score:.1f}"
                if score < 2.0:
                    label = f"{score:.1f}"
                    text_color = "#ffffff"

                ax.text(
                    j, i, label,
                    ha="center", va="center",
                    fontsize=10, fontweight=fontweight,
                    color=text_color,
                )

        ax.set_xticks(np.arange(len(rubrics)))
        ax.set_xticklabels(rubric_labels, fontsize=9, rotation=25, ha="right")
        ax.set_yticks(np.arange(n_outputs))
        ax.set_yticklabels(short_labels, fontsize=9)
        prompt_ver = data[version]["prompt_version"]
        mean_score = data[version]["overall_mean"]
        ax.set_title(
            f"Prompt {prompt_ver} (mean: {mean_score:.3f})",
            fontsize=13, fontweight="bold", pad=10,
        )

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([1, 2, 3, 4, 5])
        cbar.set_ticklabels(["1 Critical", "2 Poor", "3 Adequate", "4 Good", "5 Excellent"])
        cbar.ax.tick_params(labelsize=8, colors=COLORS["text_dim"])

    fig.suptitle(
        "Clinical Evaluation Heatmap -- All Rubrics x Outputs",
        fontsize=16, fontweight="bold", y=1.02,
        color=COLORS["text"],
    )

    save_fig(fig, "safety_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Generate all figures."""
    print("Clinical Output Evaluation Framework -- Figure Generation")
    print("=" * 60)

    setup_dark_theme()

    print("\nLoading scores from:", SCORES_PATH)
    data = load_scores()

    print(f"  v1 results: {len(data['v1']['results'])} evaluations")
    print(f"  v2 results: {len(data['v2']['results'])} evaluations")
    print(f"  Output directory: {OUTPUT_DIR}\n")

    print("Generating figures:")

    print("\n[1/7] Architecture diagram...")
    generate_architecture_diagram()

    print("\n[2/7] Rubric scores v1...")
    generate_rubric_scores_bar(data, "v1", "rubric_scores_v1.png")

    print("\n[3/7] Rubric scores v2...")
    generate_rubric_scores_bar(data, "v2", "rubric_scores_v2.png")

    print("\n[4/7] Version comparison radar...")
    generate_version_comparison_radar(data)

    print("\n[5/7] Score distribution...")
    generate_score_distribution(data)

    print("\n[6/7] Improvement delta...")
    generate_improvement_delta(data)

    print("\n[7/7] Safety heatmap...")
    generate_safety_heatmap(data)

    print("\n" + "=" * 60)
    print("All 7 figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
