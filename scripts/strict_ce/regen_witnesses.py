#!/usr/bin/env python3
"""Regenerate NY SAT witnesses for strict counterexample validation."""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_root(explicit: Path | None) -> Path:
    candidate = explicit or (
        Path(os.environ["NY_ROOT"]) if os.environ.get("NY_ROOT") else SCRIPT_REPO_ROOT
    )
    return candidate.expanduser().resolve()


def _configured_path(
    explicit: Path | None,
    environment_name: str,
    default: Path,
) -> Path:
    candidate = explicit
    if candidate is None and os.environ.get(environment_name):
        candidate = Path(os.environ[environment_name])
    return (candidate or default).expanduser().resolve()


def _category_root(benchmark_root: Path, category: str) -> Path:
    if not category or category in {".", ".."} or Path(category).name != category:
        raise ValueError(f"category must be one path component: {category!r}")
    root = benchmark_root.resolve()
    selected = (root / category).resolve()
    try:
        selected.relative_to(root)
    except ValueError as error:
        raise ValueError(f"category escapes benchmark root: {category!r}") from error
    return selected


def _source_path(category_root: Path, relative: str) -> Path:
    candidate = (category_root / relative).resolve()
    try:
        candidate.relative_to(category_root.resolve())
    except ValueError as error:
        raise ValueError(f"benchmark input escapes category root: {relative!r}") from error
    return candidate


def _materialize(category_root: Path, relative: str, destination: Path) -> bool:
    source = _source_path(category_root, relative)
    if source.is_file():
        shutil.copyfile(source, destination)
        return True
    compressed = Path(f"{source}.gz")
    if compressed.is_file():
        with gzip.open(compressed, "rb") as input_file, destination.open(
            "wb"
        ) as output_file:
            shutil.copyfileobj(input_file, output_file)
        return True
    return False


def _allocate_workdir(explicit: Path | None, category: str) -> Path:
    if explicit is None and os.environ.get("NY_STRICT_CE_WORK_DIR"):
        explicit = Path(os.environ["NY_STRICT_CE_WORK_DIR"])
    if explicit is None:
        return Path(tempfile.mkdtemp(prefix=f"ny-strict-ce-{category}-")).resolve()

    workdir = explicit.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if any(workdir.iterdir()):
        raise ValueError(f"explicit work directory must be empty: {workdir}")
    return workdir


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category")
    parser.add_argument("limit", nargs="?", type=_positive_int, default=999)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="NY repository root (default: NY_ROOT or this script's checkout)",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        help="2025 benchmark root (default: NY_BROOT or repository benchmark copy)",
    )
    parser.add_argument(
        "--measured-dir",
        type=Path,
        help="measured CSV directory (default: NY_MEASURED_DIR or reports/measured)",
    )
    parser.add_argument(
        "--ny-bin",
        type=Path,
        help="NY executable (default: NY_BIN or target/release/ny)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help=(
            "empty retained output directory (default: a unique tempfile.mkdtemp "
            "directory; NY_STRICT_CE_WORK_DIR is also accepted)"
        ),
    )
    parser.add_argument("--solver-timeout", type=_positive_int, default=60)
    parser.add_argument("--process-timeout", type=_positive_int, default=180)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    repo_root = _repo_root(args.repo_root)
    benchmark_root = _configured_path(
        args.benchmark_root,
        "NY_BROOT",
        repo_root / "benchmarks" / "vnncomp2025" / "benchmarks",
    )
    measured_dir = _configured_path(
        args.measured_dir,
        "NY_MEASURED_DIR",
        repo_root / "reports" / "measured",
    )
    ny_binary = _configured_path(
        args.ny_bin,
        "NY_BIN",
        repo_root / "target" / "release" / "ny",
    )

    try:
        category_root = _category_root(benchmark_root, args.category)
        workdir = _allocate_workdir(args.work_dir, args.category)
    except ValueError as error:
        parser.error(str(error))
    measured_csv = measured_dir / f"{args.category}.csv"
    if not category_root.is_dir():
        parser.error(f"benchmark category directory is unavailable: {category_root}")
    if not measured_csv.is_file():
        parser.error(f"measured CSV is unavailable: {measured_csv}")
    if not ny_binary.is_file() or not os.access(ny_binary, os.X_OK):
        parser.error(f"NY binary is missing or not executable: {ny_binary}")

    with measured_csv.open(newline="", encoding="utf-8") as input_file:
        rows = [
            row
            for row in csv.reader(input_file)
            if len(row) >= 6 and row[4].strip().lower() == "sat"
        ]
    print(f"{args.category}: {len(rows)} sat rows; testing up to {args.limit}")

    tally: dict[str, int] = {}

    def count(key: str) -> None:
        tally[key] = tally.get(key, 0) + 1

    for index, row in enumerate(rows[: args.limit]):
        onnx_relative, vnnlib_relative = row[1], row[2]
        instance_dir = workdir / str(index)
        instance_dir.mkdir()
        model = instance_dir / "m.onnx"
        specification = instance_dir / "p.vnnlib"
        try:
            materialized = _materialize(
                category_root, onnx_relative, model
            ) and _materialize(category_root, vnnlib_relative, specification)
        except (OSError, ValueError) as error:
            print(f"{index}\tinput-error\t{error}")
            count("missing-asset")
            continue
        if not materialized:
            count("missing-asset")
            continue

        result_file = instance_dir / "res.txt"
        try:
            process = subprocess.run(
                [
                    str(ny_binary),
                    "vnncomp",
                    "v1",
                    args.category,
                    str(model),
                    str(specification),
                    str(result_file),
                    str(args.solver_timeout),
                ],
                capture_output=True,
                text=True,
                cwd=repo_root,
                timeout=args.process_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            count("ny:process-timeout")
            continue

        if process.returncode != 0:
            count(f"ny:exit-{process.returncode}")
            diagnostic = process.stderr.strip().replace("\n", " | ")[:200]
            if diagnostic:
                print(f"{index}\tny-exit-{process.returncode}\t{diagnostic}")
            continue
        if not result_file.is_file():
            count("ny:NO-FILE")
            continue
        result_text = result_file.read_text(encoding="utf-8")
        verdict, separator, body = result_text.partition("\n")
        verdict = verdict.strip()
        if verdict != "sat":
            count(f"ny:{verdict or 'EMPTY'}")
            continue
        if not separator or not body.strip():
            count("ny:sat-without-witness")
            continue

        with gzip.open(instance_dir / "ce.gz", "wt", encoding="utf-8") as output:
            output.write(body)
        print(
            f"{index}\t{Path(onnx_relative).name}\t"
            f"{Path(vnnlib_relative).name}\tsat",
            flush=True,
        )
        count("ny-sat")

    print("NY-side tally:", tally)
    print("MANIFEST", workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
