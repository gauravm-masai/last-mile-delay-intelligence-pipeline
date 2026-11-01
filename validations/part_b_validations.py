"""Validate Part B outputs against validation/part_b_reference_metrics.csv.

Run from the project root, after part_b_regression.py:
    python validation/validate_part_b.py
"""
import sys
from pathlib import Path
from validations_common_utility import validate

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "references" / "part_b_reference_metrics.csv"

if __name__ == "__main__":
    sys.exit(validate("B", REFERENCE_PATH))
