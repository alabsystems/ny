#!/usr/bin/env python3
"""Where are ny's remaining VNN-COMP points, measured rather than guessed.

For every category this reports the rows where the published 2025 tools AGREE on
a verdict and ny's own bank carries none -- i.e. rows that are provably decidable
and that ny is provably not deciding. At +10 a row that is the recoverable score,
and it costs no box time to compute.

    python3 scripts/recoverable_rows_census.py                # summary table
    python3 scripts/recoverable_rows_census.py --csv out.csv  # + per-row dump

WHY GT IS TAKEN AS *AGREEMENT*, AND WHY THAT IS A LOWER BOUND
Rows where the reporting tools disagree are EXCLUDED entirely rather than
resolved by majority. A disagreement between competition entrants is not
something this script is entitled to adjudicate, and excluding those rows means
every number it prints understates the opportunity instead of inflating it.
`gt_tools_agreeing` is carried per row in the CSV so a single-tool row can be
told apart from a corroborated one before anyone spends a day on it.

WHAT THIS IS NOT
A bank is a DATED MEASUREMENT, not a capability fact. A row listed here may
already decide on the current binary -- that is exactly how cctsdb's three
patch-3 rows came back on 2026-09-02. Re-measure a sample before building
anything for a category. Conversely this is blind to rows the bank already
decides, so it can never report a REGRESSION; that is what
`scripts/verdict_row_moat.py` is for. The two are complements: the moat re-runs
decided rows to catch flips, this one counts rows that were never proved.
"""

import argparse
import collections
import csv
import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECIDED = ("sat", "unsat")


# The published results live OUTSIDE the repo, one CSV per tool covering EVERY
# category, with the category in column 0. This script used to glob
# `<repo>/external_tools/vnncomp2025_results/*/2025_<category>/*.csv`, a layout
# that does not exist on this host -- so it printed `TOTAL 0 0` and read as "there
# is nothing recoverable" when it had simply found no files. That is the
# empty-result trap: a zero from a wrong path and a zero from a closed front are
# byte-identical. `--results` overrides, and `census()` now FAILS LOUDLY if the
# directory yields no tools rather than reporting a confident zero.
RESULTS_DIR = os.environ.get(
    "VNNCOMP2025_RESULTS", os.path.expanduser("~/vnncomp2025_results")
)
# `SCORING-*` are pseudo-tools (re-scorings of the field, not entrants). Counting
# them inflates both the agreement count and the tool census.
PSEUDO_TOOL_PREFIXES = ("SCORING",)


def tool_dirs(results_dir=None):
    """Real entrant directories under the published results tree."""
    base = results_dir or RESULTS_DIR
    out = []
    for entry in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(entry):
            continue
        name = os.path.basename(entry)
        if name.startswith(PSEUDO_TOOL_PREFIXES):
            continue
        if glob.glob(os.path.join(entry, "*.csv")):
            out.append(entry)
    return out


def ground_truth(category, results_dir=None):
    """{(onnx, vnnlib): verdict} for rows every reporting tool agrees on."""
    per_row = collections.defaultdict(dict)
    for tool_dir in tool_dirs(results_dir):
        tool = os.path.basename(tool_dir)
        for path in sorted(glob.glob(os.path.join(tool_dir, "*.csv"))):
            with open(path, newline="") as handle:
                for row in csv.reader(handle):
                    if len(row) < 4:
                        continue
                    # one file per tool, every category in it
                    if row[0].strip() != category:
                        continue
                    key = (os.path.basename(row[1]), os.path.basename(row[2]))
                    verdicts = [c.strip() for c in row if c.strip() in DECIDED]
                    if verdicts:
                        per_row[key][tool] = verdicts[0]
    return {
        key: next(iter(set(tools.values())))
        for key, tools in per_row.items()
        if len(set(tools.values())) == 1
    }, {key: len(tools) for key, tools in per_row.items()}


def census():
    rows = []
    # FAIL LOUDLY rather than reporting a confident zero. This script's whole
    # output is "rows the field decided and ny did not"; with no tool CSVs it
    # prints TOTAL 0 and looks like good news.
    tools = tool_dirs()
    if not tools:
        raise SystemExit(
            f"recoverable_rows_census: no tool CSVs under {RESULTS_DIR!r}.\n"
            "  A zero from a missing results tree is indistinguishable from a\n"
            "  zero from a closed front. Set VNNCOMP2025_RESULTS or pass --results."
        )
    print(f"# ground truth from {len(tools)} tools: "
          f"{', '.join(os.path.basename(t) for t in tools)}")
    for bank in sorted(glob.glob(os.path.join(REPO, "reports", "measured", "*.csv"))):
        category = os.path.basename(bank)[:-4]
        if not os.path.isdir(
            os.path.join(REPO, "benchmarks", "vnncomp2025", "benchmarks", category)
        ):
            continue
        gt, n_tools = ground_truth(category)
        if not gt:
            continue
        with open(bank, newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 5 or row[0] != category:
                    continue
                key = (os.path.basename(row[1]), os.path.basename(row[2]))
                if key in gt and row[4] not in DECIDED:
                    rows.append(
                        [category, row[1], row[2], gt[key], n_tools[key], row[4]]
                    )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="also write the per-row dump here")
    args = ap.parse_args()

    rows = census()
    by_cat = collections.Counter(r[0] for r in rows)
    by_verdict = collections.Counter(r[3] for r in rows)

    print(f"{'category':<34}{'recoverable':>12}{'points':>9}")
    for category, n in by_cat.most_common():
        print(f"{category:<34}{n:>12}{n * 10:>9}")
    print(f"{'TOTAL':<34}{len(rows):>12}{len(rows) * 10:>9}")
    print()
    for verdict, n in by_verdict.most_common():
        print(f"  {verdict:<6} {n}")
    # The split is the point: a set that is overwhelmingly `unsat` says the gap
    # is proving power, not counterexample search, and should redirect work.
    if rows and by_verdict.get("sat", 0) * 20 < len(rows):
        print("\n  -> the gap is almost entirely UNSAT: proving power, not falsification")

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "category",
                    "onnx",
                    "vnnlib",
                    "gt_verdict",
                    "gt_tools_agreeing",
                    "ny_bank_verdict",
                ]
            )
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
