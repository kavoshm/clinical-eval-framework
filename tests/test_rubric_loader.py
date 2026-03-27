"""
Tests for rubric YAML parsing and loading.

Covers:
    - Loading individual rubrics by name
    - Loading all rubrics from the directory
    - Parsing criteria, scoring levels, and weights
    - Cache behavior
    - Error handling for missing/malformed files
"""

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from rubric_loader import RubricLoader, load_rubric, load_all_rubrics
from models import RubricCategory


# ---------------------------------------------------------------------------
# Rubric Loading
# ---------------------------------------------------------------------------

class TestRubricLoading:
    """Tests for loading rubrics from YAML files."""

    def test_load_accuracy_rubric(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric = loader.load("accuracy")
        assert rubric.name == "clinical_accuracy"
        assert rubric.display_name == "Clinical Accuracy"
        assert rubric.category == RubricCategory.ACCURACY
        assert rubric.weight == 1.5

    def test_load_safety_rubric(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric = loader.load("safety")
        assert rubric.name == "patient_safety"
        assert rubric.display_name == "Patient Safety"
        assert rubric.category == RubricCategory.SAFETY
        assert rubric.weight == 2.0

    def test_load_completeness_rubric(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric = loader.load("completeness")
        assert rubric.name == "clinical_completeness"
        assert rubric.category == RubricCategory.COMPLETENESS
        assert rubric.weight == 1.0

    def test_load_appropriateness_rubric(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric = loader.load("clinical_appropriateness")
        assert rubric.name == "clinical_appropriateness"
        assert rubric.category == RubricCategory.APPROPRIATENESS
        assert rubric.weight == 0.8

    def test_load_all_rubrics(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubrics = loader.load_all()
        assert len(rubrics) == 4
        names = {r.name for r in rubrics}
        assert "clinical_accuracy" in names
        assert "patient_safety" in names
        assert "clinical_completeness" in names
        assert "clinical_appropriateness" in names

    def test_load_nonexistent_rubric_raises(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_rubric")

    def test_load_from_nonexistent_directory_raises(self, tmp_path):
        loader = RubricLoader(rubrics_dir=tmp_path / "no_such_dir")
        with pytest.raises(FileNotFoundError):
            loader.load_all()

    def test_load_empty_yaml_raises(self, tmp_path):
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        loader = RubricLoader(rubrics_dir=tmp_path)
        with pytest.raises(ValueError, match="Empty rubric file"):
            loader.load("empty")


# ---------------------------------------------------------------------------
# YAML Parsing
# ---------------------------------------------------------------------------

class TestYAMLParsing:
    """Tests for correct parsing of YAML rubric fields."""

    def test_criteria_parsed(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric = loader.load("accuracy")
        assert len(rubric.criteria) == 5
        criterion_names = [c.name for c in rubric.criteria]
        assert "diagnostic_accuracy" in criterion_names
        assert "no_hallucination" in criterion_names

    def test_criteria_weights_parsed(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric = loader.load("accuracy")
        halluc = next(c for c in rubric.criteria if c.name == "no_hallucination")
        assert halluc.weight == 1.2
        temporal = next(c for c in rubric.criteria if c.name == "temporal_accuracy")
        assert temporal.weight == 0.8

    def test_scoring_levels_parsed(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric = loader.load("safety")
        assert len(rubric.scoring_levels) == 5
        scores = sorted([sl.score for sl in rubric.scoring_levels])
        assert scores == [1, 2, 3, 4, 5]

    def test_scoring_level_labels(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric = loader.load("safety")
        level_1 = rubric.get_scoring_level(1)
        assert level_1 is not None
        assert level_1.label == "Dangerous"
        level_5 = rubric.get_scoring_level(5)
        assert level_5 is not None
        assert level_5.label == "Fully Safe"

    def test_scoring_level_examples(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric = loader.load("accuracy")
        level_1 = rubric.get_scoring_level(1)
        assert level_1 is not None
        assert len(level_1.examples) > 0
        assert any("WBC" in ex for ex in level_1.examples)

    def test_scale_min_max(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric = loader.load("safety")
        assert rubric.scale_min == 1
        assert rubric.scale_max == 5

    def test_evaluation_prompt_template(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric = loader.load("accuracy")
        assert "{source_text}" in rubric.evaluation_prompt_template
        assert "{output_text}" in rubric.evaluation_prompt_template

    def test_format_rubric_for_prompt(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric = loader.load("accuracy")
        formatted = rubric.format_rubric_for_prompt()
        assert "Clinical Accuracy" in formatted
        assert "Scoring Scale:" in formatted

    def test_custom_yaml_parsing(self, tmp_path):
        """Test parsing a custom minimal rubric YAML."""
        custom_rubric = {
            "name": "test_rubric",
            "display_name": "Test Rubric",
            "description": "A test rubric.",
            "category": "accuracy",
            "weight": 1.5,
            "scale": {"min": 1, "max": 5},
            "criteria": [
                {"name": "criterion_a", "description": "First criterion", "weight": 1.0},
                {"name": "criterion_b", "description": "Second criterion"},
            ],
            "scoring_rubric": {
                1: {"label": "Bad", "description": "Very bad"},
                5: {"label": "Good", "description": "Very good"},
            },
            "evaluation_prompt_template": "Evaluate: {source_text} | {output_text}",
        }
        yaml_path = tmp_path / "test_rubric.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(custom_rubric, f)

        loader = RubricLoader(rubrics_dir=tmp_path)
        rubric = loader.load("test_rubric")
        assert rubric.name == "test_rubric"
        assert rubric.weight == 1.5
        assert len(rubric.criteria) == 2
        assert rubric.criteria[1].weight == 1.0  # default weight
        assert len(rubric.scoring_levels) == 2


# ---------------------------------------------------------------------------
# Cache Behavior
# ---------------------------------------------------------------------------

class TestRubricCache:
    """Tests for rubric loader caching."""

    def test_cache_returns_same_object(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        rubric_1 = loader.load("accuracy")
        rubric_2 = loader.load("accuracy")
        assert rubric_1 is rubric_2

    def test_clear_cache(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        loader.load("accuracy")
        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_get_rubric_names(self, rubrics_dir):
        loader = RubricLoader(rubrics_dir=rubrics_dir)
        names = loader.get_rubric_names()
        assert len(names) == 4
        assert "clinical_accuracy" in names
        assert "patient_safety" in names


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_load_rubric_function(self, rubrics_dir):
        rubric = load_rubric("safety", rubrics_dir=rubrics_dir)
        assert rubric.name == "patient_safety"

    def test_load_all_rubrics_function(self, rubrics_dir):
        rubrics = load_all_rubrics(rubrics_dir=rubrics_dir)
        assert len(rubrics) == 4
