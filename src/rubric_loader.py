"""
Rubric Loader — Load evaluation rubrics from YAML files.

Parses YAML rubric definitions and converts them into EvalRubric models.
Supports loading individual rubrics or all rubrics from a directory.
"""

from __future__ import annotations

import logging
import yaml
from pathlib import Path
from typing import Optional

from models import EvalRubric, EvalCriterion, ScoringLevel, RubricCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RUBRICS_DIR = Path(__file__).parent.parent / "rubrics"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class RubricLoader:
    """
    Load evaluation rubrics from YAML files.

    Usage:
        loader = RubricLoader()
        rubrics = loader.load_all()
        accuracy_rubric = loader.load("accuracy")
    """

    def __init__(self, rubrics_dir: Optional[Path] = None):
        self.rubrics_dir = rubrics_dir or DEFAULT_RUBRICS_DIR
        self._cache: dict[str, EvalRubric] = {}

    def load(self, rubric_name: str) -> EvalRubric:
        """
        Load a single rubric by name.

        Args:
            rubric_name: Name of the rubric (without .yaml extension)

        Returns:
            Parsed EvalRubric model

        Raises:
            FileNotFoundError: if rubric file does not exist
            ValueError: if rubric file is malformed
        """
        if rubric_name in self._cache:
            logger.debug("Rubric '%s' loaded from cache", rubric_name)
            return self._cache[rubric_name]

        # Try exact filename first, then with .yaml extension
        yaml_path = self.rubrics_dir / f"{rubric_name}.yaml"
        if not yaml_path.exists():
            yaml_path = self.rubrics_dir / f"{rubric_name}.yml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Rubric file not found: {rubric_name}.yaml in {self.rubrics_dir}"
            )

        rubric = self._parse_yaml(yaml_path)
        self._cache[rubric_name] = rubric
        logger.info(
            "Loaded rubric '%s' (category=%s, weight=%.1f, criteria=%d)",
            rubric.display_name, rubric.category.value, rubric.weight, len(rubric.criteria),
        )
        return rubric

    def load_all(self) -> list[EvalRubric]:
        """Load all rubrics from the rubrics directory."""
        rubrics = []
        if not self.rubrics_dir.exists():
            raise FileNotFoundError(f"Rubrics directory not found: {self.rubrics_dir}")

        for yaml_file in sorted(self.rubrics_dir.glob("*.yaml")):
            rubric = self._parse_yaml(yaml_file)
            self._cache[rubric.name] = rubric
            rubrics.append(rubric)

        # Also check for .yml files
        for yml_file in sorted(self.rubrics_dir.glob("*.yml")):
            if yml_file.stem not in self._cache:
                rubric = self._parse_yaml(yml_file)
                self._cache[rubric.name] = rubric
                rubrics.append(rubric)

        logger.info("Loaded %d rubrics from %s", len(rubrics), self.rubrics_dir)
        return rubrics

    def _parse_yaml(self, yaml_path: Path) -> EvalRubric:
        """Parse a YAML file into an EvalRubric model."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty rubric file: {yaml_path}")

        # Parse criteria
        criteria = []
        for criterion_data in data.get("criteria", []):
            criteria.append(EvalCriterion(
                name=criterion_data["name"],
                description=criterion_data["description"],
                weight=criterion_data.get("weight", 1.0),
            ))

        # Parse scoring levels
        scoring_levels = []
        scoring_rubric = data.get("scoring_rubric", {})
        for score_val, level_data in scoring_rubric.items():
            score_int = int(score_val)
            if isinstance(level_data, dict):
                scoring_levels.append(ScoringLevel(
                    score=score_int,
                    label=level_data.get("label", f"Level {score_int}"),
                    description=level_data.get("description", ""),
                    examples=level_data.get("examples", []),
                ))
            elif isinstance(level_data, str):
                scoring_levels.append(ScoringLevel(
                    score=score_int,
                    label=f"Level {score_int}",
                    description=level_data,
                ))

        # Parse scale
        scale = data.get("scale", {})

        # Map category string to enum
        category_str = data.get("category", "accuracy")
        try:
            category = RubricCategory(category_str)
        except ValueError:
            category = RubricCategory.ACCURACY

        return EvalRubric(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            description=data.get("description", ""),
            category=category,
            weight=data.get("weight", 1.0),
            version=data.get("version", "1.0"),
            scale_min=scale.get("min", 1),
            scale_max=scale.get("max", 5),
            criteria=criteria,
            scoring_levels=scoring_levels,
            evaluation_prompt_template=data.get("evaluation_prompt_template", ""),
        )

    def get_rubric_names(self) -> list[str]:
        """Return names of all available rubrics."""
        names = []
        if self.rubrics_dir.exists():
            for yaml_file in sorted(self.rubrics_dir.glob("*.yaml")):
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                    if data and "name" in data:
                        names.append(data["name"])
        return names

    def clear_cache(self) -> None:
        """Clear the rubric cache."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def load_rubric(name: str, rubrics_dir: Optional[Path] = None) -> EvalRubric:
    """Load a single rubric by name."""
    loader = RubricLoader(rubrics_dir)
    return loader.load(name)


def load_all_rubrics(rubrics_dir: Optional[Path] = None) -> list[EvalRubric]:
    """Load all rubrics from the directory."""
    loader = RubricLoader(rubrics_dir)
    return loader.load_all()


# ---------------------------------------------------------------------------
# Main (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading all rubrics...\n")
    loader = RubricLoader()

    try:
        rubrics = loader.load_all()
        for rubric in rubrics:
            print(f"  {rubric.display_name} ({rubric.name})")
            print(f"    Category: {rubric.category.value}")
            print(f"    Weight: {rubric.weight}")
            print(f"    Criteria: {len(rubric.criteria)}")
            print(f"    Scoring Levels: {len(rubric.scoring_levels)}")
            print(f"    Scale: {rubric.scale_min}-{rubric.scale_max}")
            print()
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("  Run from the project root directory.")
