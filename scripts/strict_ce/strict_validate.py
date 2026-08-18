#!/usr/bin/env python3
"""Validate regenerated witnesses with the official zero-tolerance checker."""

from __future__ import annotations

import argparse
import collections
import importlib
import os
import sys
from pathlib import Path
from typing import Callable


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
SCORING_ENV = "VNNCOMP_SCORING_DIR"
RESULTS_ENV = "VNNCOMP2026_RESULTS"


def _repo_root(explicit: Path | None) -> Path:
    candidate = explicit or (
        Path(os.environ["NY_ROOT"]) if os.environ.get("NY_ROOT") else SCRIPT_REPO_ROOT
    )
    return candidate.expanduser().resolve()


def _scoring_dir(explicit: Path | None, repo_root: Path) -> Path:
    if explicit is not None:
        candidate = explicit
    elif os.environ.get(SCORING_ENV):
        candidate = Path(os.environ[SCORING_ENV])
    elif os.environ.get(RESULTS_ENV):
        candidate = Path(os.environ[RESULTS_ENV]) / "SCORING"
    else:
        candidate = repo_root / "external_tools" / "vnncomp2026_results" / "SCORING"
    return candidate.expanduser().resolve()


def _load_checker(scoring_dir: Path) -> Callable[..., tuple[object, str]]:
    counterexamples = scoring_dir / "counterexamples.py"
    if not scoring_dir.is_dir() or not counterexamples.is_file():
        raise RuntimeError(
            f"official SCORING directory is unavailable at {scoring_dir}; "
            f"set --scoring-dir or {SCORING_ENV}"
        )

    # The organizer module imports its sibling settings/vnnlib modules by
    # name. Put only the explicitly resolved directory first, then verify that
    # Python did not satisfy `counterexamples` from an unrelated installation.
    sys.path.insert(0, str(scoring_dir))
    try:
        module = importlib.import_module("counterexamples")
    except Exception as error:
        raise RuntimeError(
            f"could not import official checker from {scoring_dir}: {error}"
        ) from error
    finally:
        sys.path.pop(0)

    module_path = Path(getattr(module, "__file__", "")).resolve()
    if module_path != counterexamples.resolve():
        raise RuntimeError(
            f"counterexamples resolved outside configured SCORING directory: {module_path}"
        )
    checker = getattr(module, "get_ce_diff", None)
    if not callable(checker):
        raise RuntimeError(f"{counterexamples} does not define callable get_ce_diff")
    return checker


def _witness_dirs(root: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, int | str]:
        return (0, int(path.name)) if path.name.isdigit() else (1, path.name)

    return sorted((path for path in root.iterdir() if path.is_dir()), key=key)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("witness_root", type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="NY repository root (default: NY_ROOT or this script's checkout)",
    )
    parser.add_argument(
        "--scoring-dir",
        type=Path,
        help=(
            "official vnncomp2026_results/SCORING directory "
            f"(default: {SCORING_ENV}, {RESULTS_ENV}/SCORING, or repository copy)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    witness_root = args.witness_root.expanduser().resolve()
    if not witness_root.is_dir():
        parser.error(f"witness root is not a directory: {witness_root}")

    repo_root = _repo_root(args.repo_root)
    scoring_dir = _scoring_dir(args.scoring_dir, repo_root)
    try:
        get_ce_diff = _load_checker(scoring_dir)
    except RuntimeError as error:
        parser.error(str(error))

    tally: collections.Counter[str] = collections.Counter()
    for directory in _witness_dirs(witness_root):
        counterexample = directory / "ce.gz"
        if not counterexample.is_file():
            continue
        try:
            result, message = get_ce_diff(
                str(directory / "m.onnx"),
                str(directory / "p.vnnlib"),
                str(counterexample),
                0.0,
                0.0,
            )
        except Exception as error:  # organizer checker owns the exception types
            tally["EXCEPTION"] += 1
            print(directory.name, "EXC", repr(error)[:120])
            continue
        name = str(getattr(result, "value", result)).lower()
        tally[name] += 1
        if name != "correct":
            print(directory.name, name, str(message).replace("\n", " | ")[:200])

    print()
    print("STRICT (abs_tol=0, rel_tol=0) RESULTS:", dict(tally))
    total = sum(tally.values())
    correct = tally.get("correct", 0)
    if total:
        print(f"strict yield: {correct}/{total} = {100.0 * correct / total:.1f}%")
    else:
        print("no data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
