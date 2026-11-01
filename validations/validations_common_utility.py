"""Shared validation utility used by every part (validate_part_a.py, _b, _c, _d).

Everything that is checked comes from the part's reference CSV. One row = one check.

Required columns
    source_file   file to read, relative to the project root (.csv, .py, .md, ...)
    metric_name   name of the check (also used to find the value, see below)
    value_type    ACCEPTANCE -> a miss is a FAIL (graded criterion)
                  REFERENCE  -> a miss is a WARN (value from the pinned configuration;
                                the brief grades the band, not this exact number)
    metric_value  reference value from the brief
    tolerance     allowed absolute difference for EQ (0 = exact)

Optional columns (Part A works without them)
    comparison    EQ (default) |generated - reference| <= tolerance
                  GE generated >= reference      LE generated <= reference
                  GT generated >  reference      LT generated <  reference
    expression    Python expression that computes the generated value.
                  Blank -> the value is looked up with the metric_name convention.
    description   plain-words meaning of the check
    brief_ref     where the reference comes from in the brief

Metric_name lookup convention (used when expression is blank, and by get()):
    raw_rows                       row whose first column is "raw_rows"; value in the last column
    R2__linear_regression__val     column "R2" on the row whose leading columns are
                                   linear_regression, val (in order)
    ROW_COUNT                      number of data rows in the file

Names available inside an expression
    df            the source_file as a DataFrame (CSV files only)
    src, code     the source_file as text; code has # comments and all whitespace removed
    get(name)     a value from df by the lookup convention; get(name, other_df) for another table
    read(path)    another CSV file as a DataFrame
    clean_csv(df) 1 if df has no stray index column and no value with more than 4 decimals
    confusion_ok(df) 1 if TN/FP/FN/TP in every row reproduce accuracy/precision/recall/f1
    literal(text) turn text such as "{'max_depth': 6}" into a Python value
    rerun_identical(script, output) 1 if running the script twice gives a byte-identical output file
    (any file type can be a source_file; a missing file is reported as ERROR, so "1" checks existence)
    re, np, math, pd
Booleans are written as int(...) so that a pass is 1 and the reference is 1.
"""
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = PROJECT_ROOT / "validations"
REPORT_DIR = VALIDATION_DIR / "reports"

COMPARE = {
    "EQ": lambda a, e, t: abs(a - e) <= t + 1e-9,
    "GE": lambda a, e, t: a >= e,
    "LE": lambda a, e, t: a <= e,
    "GT": lambda a, e, t: a > e,
    "LT": lambda a, e, t: a < e,
}
SYMBOL = {"EQ": "=", "GE": "≥", "LE": "≤", "GT": ">", "LT": "<"}


def as_number(x):
    """Return x as a float, or None if it is not a number."""
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def find_value(df, metric_name):
    """Pick one value out of a table using the metric_name convention."""
    if metric_name == "ROW_COUNT":
        return len(df)
    parts = metric_name.split("__")
    column, keys = (df.columns[-1], parts) if len(parts) == 1 else (parts[0], parts[1:])
    rows = df
    for i, key in enumerate(keys):
        rows = rows[rows.iloc[:, i].astype(str) == key]
    if len(rows) != 1:
        raise LookupError(f"{len(rows)} rows match {keys}")
    if column not in df.columns:
        raise LookupError(f"no column '{column}'")
    return rows[column].iloc[0]


def clean_csv(df):
    """1 when the table has no 'Unnamed' index column and no value with more than 4 decimals."""
    floats = df.select_dtypes("float")
    over = ((floats - floats.round(4)).abs() > 1e-9).any().any()
    unnamed = any(str(c).startswith("Unnamed") for c in df.columns)
    return int(not over and not unnamed)

def confusion_ok(df, tol=0.0001):
    """1 when TN/FP/FN/TP in every row reproduce accuracy, precision, recall and f1
    (precision/recall/f1 taken as 0 when undefined), within rounding."""
    n = df.TN + df.FP + df.FN + df.TP
    expected = {
        "accuracy": (df.TP + df.TN) / n,
        "precision": (df.TP / (df.TP + df.FP)).fillna(0),
        "recall": (df.TP / (df.TP + df.FN)).fillna(0),
        "f1": (2 * df.TP / (2 * df.TP + df.FP + df.FN)).fillna(0),
    }
    return int(all(((df[c] - v).abs() <= tol).all() for c, v in expected.items()))

def generated_value(r, root):
    """Compute the value to compare for one reference row."""
    path = root / r["source_file"]
    is_csv = path.suffix.lower() == ".csv"
    df = pd.read_csv(path) if is_csv else None
    src = path.read_text(encoding="utf-8", errors="replace")  # errors=replace: binary files such as .pkl
    expression = str(r.get("expression", "") or "").strip()
    if not expression:
        return find_value(df, r["metric_name"])
    names = {
        "df": df, "src": src, "code": re.sub(r"\s+", "", re.sub(r"#.*", "", src)),
        "get": lambda name, table=None: find_value(df if table is None else table, name),
        "read": lambda p: pd.read_csv(root / p),
        "clean_csv": clean_csv, "confusion_ok": confusion_ok,
        "literal": __import__("ast").literal_eval,
        "re": re, "np": np, "math": math, "pd": pd,
    }
    value = eval(expression, names)  # expressions come only from your own reference CSV
    return value.item() if hasattr(value, "item") else value


def judge(r, actual):
    """Return (status, comment) for one generated value."""
    op = (str(r.get("comparison", "") or "").strip().upper()) or "EQ"
    ref, tol_text = r["metric_value"], (r["tolerance"] or "0")
    a, e, tol = as_number(actual), as_number(ref), as_number(tol_text) or 0.0
    miss = "WARN" if r["value_type"].strip().upper() == "REFERENCE" else "FAIL"

    if a is None or e is None:  # text, e.g. a model name
        ok = str(actual).strip() == str(ref).strip()
        return ("PASS", "Matches") if ok else (miss, f"Expected '{ref}'")

    if COMPARE[op](a, e, tol):
        if op != "EQ":
            return "PASS", f"Meets {SYMBOL[op]} {ref}"
        return "PASS", "Exact match" if tol == 0 else f"Within ±{tol_text} (diff {a - e:+.4g})"
    if op == "EQ":
        limit = "exact match required" if tol == 0 else f"allowed ±{tol_text}"
        return miss, f"Off by {a - e:+.4g} ({limit})"
    return miss, f"Needs {SYMBOL[op]} {ref}; short by {abs(a - e):.4g}"


def check(r, root):
    """Run one check. Returns (generated value, status, comment)."""
    try:
        actual = generated_value(r, root)
    except FileNotFoundError:
        return "", "ERROR", f"File not found: {r['source_file']}"
    except Exception as ex:
        # A REFERENCE row may name a model your run did not pick (e.g. a different CV pair): only warn.
        status = "WARN" if r["value_type"].strip().upper() == "REFERENCE" else "ERROR"
        return "", status, f"Could not compute value: {type(ex).__name__}: {ex}"
    status, comment = judge(r, actual)
    return actual, status, comment


def show(x):
    """Readable value: 1224.0 -> 1224, True -> 1, long floats shortened."""
    v = as_number(x)
    if v is None:
        return str(x)
    return str(int(v)) if v == int(v) else f"{v:.6g}"


def reference_text(r):
    op = (str(r.get("comparison", "") or "").strip().upper()) or "EQ"
    if op == "EQ":
        tol = as_number(r["tolerance"]) or 0
        return r["metric_value"] if tol == 0 else f"{r['metric_value']} ± {r['tolerance']}"
    return f"{SYMBOL[op]} {r['metric_value']}"


def validate(part, reference_csv):
    """Run every check in the reference CSV, print a summary, write the report.
    Returns 0 when there is no FAIL or ERROR (WARN does not fail the run)."""
    ref = pd.read_csv(VALIDATION_DIR / reference_csv, dtype=str, keep_default_na=False)

    results = []
    for _, r in ref.iterrows():
        actual, status, comment = check(r, PROJECT_ROOT)
        results.append({
            "source_file": r["source_file"], "metric_name": r["metric_name"],
            "value_type": r["value_type"], "description": r.get("description", ""),
            "reference": reference_text(r), "generated": show(actual),
            "status": status, "comment": comment, "brief_ref": r.get("brief_ref", ""),
        })
        #print(f"[{status:<5}] {r['metric_name']:<46} ref={reference_text(r):<14} "
        #      f"got={show(actual):<10} {comment}")

    report = pd.DataFrame(results)
    n = report["status"].value_counts()
    counts = {k: int(n.get(k, 0)) for k in ("PASS", "FAIL", "ERROR", "WARN")}
    print(f"\nPart {part}: {counts['PASS']}/{len(report)} passed, {counts['FAIL']} failed, "
          f"{counts['ERROR']} errors, {counts['WARN']} warnings")

    stem = f"part_{part.lower()}_validation_report"
    report.to_csv(REPORT_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    write_markdown(report, part, counts, REPORT_DIR / f"{stem}.md")
    print(f"Report: validation/{stem}.csv and validation/{stem}.md")
    return 0 if counts["FAIL"] == 0 and counts["ERROR"] == 0 else 1

def write_markdown(report, part, counts, path):
    """Same results as a readable table; problems first."""
    summary = " · ".join(f"{k} {v}" for k, v in counts.items())
    lines = [f"# Part {part} validation report", "",
             f"Run: {datetime.now():%Y-%m-%d %H:%M} · **{counts['PASS']}/{len(report)} passed** · {summary}",
             "",
             "FAIL = graded criterion not met · ERROR = value could not be computed · "
             "WARN = differs from the pinned-configuration reference (graded on the band, not this number)",
             "",
             "| Metric | Type | Reference | Generated | Status | Comment |",
             "|---|---|---|---|---|---|"]
    order = {"ERROR": 0, "FAIL": 1, "WARN": 2, "PASS": 3}
    sorted_report = report.assign(_o=report["status"].map(order)).sort_values("_o", kind="stable")
    for _, r in sorted_report.iterrows():
        status = r["status"] if r["status"] == "PASS" else f"**{r['status']}**"
        cells = [r["metric_name"], r["value_type"], r["reference"], r["generated"], status, r["comment"]]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
