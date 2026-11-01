"""Validate Part D outputs against validation/part_d_reference_metrics.csv.

Run from the project root, after part_d_segments.py, package_model.py and predict_delay.py:
    python validation/validate_part_d.py
"""
import sys
from pathlib import Path
from validations_common_utility import validate

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "references" / "part_d_reference_metrics.csv"

if __name__ == "__main__":
    sys.exit(validate("D", REFERENCE_PATH))
