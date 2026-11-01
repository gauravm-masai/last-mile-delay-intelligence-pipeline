"""Validate Part A outputs against validation/part_a_reference_metrics.csv.

Run from the project root, after part_a_clean_eda.py:
    python validation/validate_part_a.py
"""
import sys
from pathlib import Path
from validations_common_utility import validate

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "references" / "part_a_reference_metrics.csv"

if __name__ == "__main__":
    sys.exit(validate("A", REFERENCE_PATH))
