"""
Shared fixtures for the Clinical Eval Framework test suite.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the src/ directory is on the import path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

RUBRICS_DIR = Path(__file__).parent.parent / "rubrics"


@pytest.fixture
def rubrics_dir():
    """Return the path to the project rubrics directory."""
    return RUBRICS_DIR


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary SQLite database path."""
    return str(tmp_path / "test_eval.db")


@pytest.fixture
def sample_source_text():
    """Return the canonical sample clinical source document."""
    return (
        "Patient: 67-year-old male with Type 2 diabetes mellitus (HbA1c 8.2%), "
        "hypertension (on lisinopril 20mg daily), and recently diagnosed stage IIIA "
        "non-small cell lung cancer. Patient presents for follow-up after first cycle "
        "of carboplatin/pemetrexed chemotherapy. Reports moderate fatigue, decreased "
        "appetite, and occasional nausea. Blood glucose readings have been elevated "
        "(180-240 mg/dL fasting) since starting chemotherapy. Current medications: "
        "metformin 1000mg BID, lisinopril 20mg daily, ondansetron 8mg PRN nausea. "
        "Lab results: WBC 3.2 (low), platelets 145, creatinine 1.3 (slightly elevated)."
    )
