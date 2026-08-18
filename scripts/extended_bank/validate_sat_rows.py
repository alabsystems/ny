#!/usr/bin/env python3
"""Revalidate recorded SAT rows against their complete VNNLIB specifications.

For every ``sat``/``violated`` row in ``reports/measured/<bench>.csv``, rerun
ny, capture its witness, and ask :mod:`vnnlib_ce` to check that the witness is
strictly in-box and violates the full output relation.  Measured and bank files
are read-only; the audit is written to ``NY_SAT_AUDIT_OUT`` or a temporary-file
default.

Usage: validate_sat_rows.py <bench> [rerun_budget] [max_rows]

Runtime paths are portable and overridable with ``NY_ROOT``, ``NY_BIN``,
``NY_AY``, and ``NY_SAT_AUDIT_OUT``.  Missing executables, inputs, or Python
runtime packages fail closed; validator errors yield a distinct
``ENVIRONMENT/ERROR`` conclusion, never ``SOUND``.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import vnnlib_ce  # noqa: E402

RETRIES = 2
BENCHMARK_ROOT = Path("benchmarks/vnncomp2025/benchmarks")
# Shared strict witness grammar: a naive [-\d.eE]+ value pattern would drop
# '+'-signed exponents and silently lose assignments.
RAW_ASSIGNMENT = vnnlib_ce.COUNTEREXAMPLE_ASSIGNMENT


def _configured_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _runtime_paths(bench: str) -> tuple[Path, Path, Path, Path, Path]:
    root = _configured_path("NY_ROOT", REPO_ROOT)
    ny_bin = _configured_path("NY_BIN", root / "target/release/ny")
    ay_bin = _configured_path(
        "NY_AY", root.parent / "ay" / "target/release/ay"
    )
    bench_dir = root / BENCHMARK_ROOT / bench
    measured_csv = root / "reports" / "measured" / f"{bench}.csv"
    default_output = (
        Path(tempfile.gettempdir()) / f"ny_sat_audit_{bench}_{os.getpid()}.csv"
    )
    output = _configured_path("NY_SAT_AUDIT_OUT", default_output)
    return ny_bin, ay_bin, bench_dir, measured_csv, output


def _require_executable(path: Path, label: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"{label} is missing or not executable: {path}")


def _load_budgets(instances_csv: Path, cap: int) -> dict[tuple[str, str], int]:
    budgets: dict[tuple[str, str], int] = {}
    with instances_csv.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            try:
                budget = int(float(row[2]))
            except ValueError:
                budget = cap
            budgets[(Path(row[0]).name, Path(row[1]).name)] = min(budget, cap)
    return budgets


def _resolve_benchmark_input(bench_dir: Path, raw_path: str) -> Path:
    relative = raw_path.strip().removeprefix("./")
    candidates = [bench_dir / relative, bench_dir / Path(relative).name]
    candidates.extend(
        bench_dir / subdir / Path(relative).name for subdir in ("onnx", "vnnlib")
    )
    resolved_bench_dir = bench_dir.resolve()
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(resolved_bench_dir)
        except ValueError:
            continue
        if resolved_candidate.is_file():
            return resolved_candidate
    return (bench_dir / Path(relative).name).resolve()


def _run_ny(
    ny_bin: Path,
    ay_bin: Path,
    bench: str,
    model: Path,
    specification: Path,
    budget: int,
) -> tuple[str, str]:
    environment = dict(os.environ)
    environment["NY_AY"] = str(ay_bin)
    environment.setdefault("RUST_LOG", "error")

    with tempfile.NamedTemporaryFile(
        prefix="ny-sat-audit-result-", suffix=".txt", delete=False
    ) as result_file:
        result_path = Path(result_file.name)
    try:
        try:
            process = subprocess.Popen(
                [
                    str(ny_bin),
                    "vnncomp",
                    "v1",
                    bench,
                    str(model),
                    str(specification),
                    str(result_path),
                    str(budget),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
            )
        except OSError:
            return "process-error", ""
        try:
            return_code = process.wait(timeout=budget + 40)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return "process-error", ""
        if return_code != 0:
            return "process-error", ""
        contents = result_path.read_text(encoding="utf-8")
        return contents.split("\n", 1)[0].strip(), contents
    finally:
        result_path.unlink(missing_ok=True)


def _parse_args(argv: list[str]) -> tuple[str, int, int]:
    if not 2 <= len(argv) <= 4:
        raise RuntimeError(
            "usage: validate_sat_rows.py <bench> [rerun_budget] [max_rows]"
        )
    bench = argv[1]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", bench):
        raise RuntimeError(f"invalid benchmark category: {bench!r}")
    try:
        cap = int(argv[2]) if len(argv) > 2 else 300
        maximum = int(argv[3]) if len(argv) > 3 else 100_000
    except ValueError as error:
        raise RuntimeError("rerun_budget and max_rows must be integers") from error
    if cap < 0 or maximum < 0:
        raise RuntimeError("rerun_budget and max_rows must be non-negative")
    return bench, cap, maximum


def _audit_conclusion(
    valid: int, out_of_box: int, errored: int, other: int, not_reproduced: int
) -> tuple[str, int]:
    if out_of_box:
        return "*** OUT-OF-BOX SAT FOUND — moat concern ***", 1
    if errored:
        return (
            "ENVIRONMENT/ERROR: validation errored instead of producing verdict "
            "evidence; fix the environment and rerun.",
            2,
        )
    if other or not_reproduced or not valid:
        return (
            "INCONCLUSIVE: recorded SAT rows were not all reproduced and validated.",
            1,
        )
    return (
        "SOUND: every reproduced ny=sat is a genuine strictly-in-box counterexample.",
        0,
    )


def main(argv: list[str]) -> int:
    try:
        bench, cap, maximum = _parse_args(argv)
        ny_bin, ay_bin, bench_dir, measured_csv, output = _runtime_paths(bench)
        _require_executable(ny_bin, "ny binary")
        _require_executable(ay_bin, "AY binary")
        instances_csv = bench_dir / "instances.csv"
        for required, label in (
            (bench_dir, "benchmark directory"),
            (instances_csv, "benchmark instances.csv"),
            (measured_csv, "measured SAT report"),
        ):
            if not required.exists():
                raise RuntimeError(f"{label} is missing: {required}")
    except RuntimeError as error:
        print(f"validate_sat_rows: error: {error}", file=sys.stderr)
        return 2

    budgets = _load_budgets(instances_csv, cap)

    def budget_for(model: Path, specification: Path) -> int:
        return budgets.get((model.name, specification.name), cap)

    raw_rows: list[tuple[str, str]] = []
    with measured_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 5 and row[4].strip().lower() in ("sat", "violated"):
                raw_rows.append((row[1], row[2]))

    rows: list[tuple[Path, Path]] = []
    for model_raw, specification_raw in raw_rows[:maximum]:
        model = _resolve_benchmark_input(bench_dir, model_raw)
        specification = _resolve_benchmark_input(bench_dir, specification_raw)
        for required, label in ((model, "ONNX model"), (specification, "VNNLIB file")):
            if not required.is_file():
                print(
                    f"validate_sat_rows: error: {label} is missing: {required}",
                    file=sys.stderr,
                )
                return 2
        rows.append((model, specification))

    # Validate cheap, repository-owned inputs before probing optional heavy
    # packages. This keeps a missing model/property diagnostic deterministic
    # even on a minimal host, while still failing before any verifier rerun can
    # produce evidence that we would be unable to validate.
    try:
        vnnlib_ce.require_runtime_dependencies()
    except RuntimeError as error:
        print(f"validate_sat_rows: error: {error}", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    valid = out_of_box = errored = not_reproduced = other = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["bench", "onnx", "vnnlib", "rerun_verdict", "in_box", "is_ce", "detail"]
        )
        for index, (model, specification) in enumerate(rows):
            reproduced = False
            for _ in range(RETRIES):
                verdict, counterexample = _run_ny(
                    ny_bin,
                    ay_bin,
                    bench,
                    model,
                    specification,
                    budget_for(model, specification),
                )
                if verdict != "sat":
                    continue
                validation_errored = False
                try:
                    assignments = {
                        int(name): float(value)
                        for name, value in RAW_ASSIGNMENT.findall(counterexample)
                    }
                    in_box, is_counterexample, detail = vnnlib_ce.validate(
                        model, specification, assignments
                    )
                except Exception as error:  # validator errors belong in the audit
                    in_box, is_counterexample, detail = None, None, f"ERR {error}"
                    validation_errored = True
                writer.writerow(
                    [
                        bench,
                        model.name,
                        specification.name,
                        verdict,
                        in_box,
                        is_counterexample,
                        detail,
                    ]
                )
                handle.flush()
                if is_counterexample:
                    valid += 1
                elif validation_errored:
                    errored += 1
                    print(f"  !! {specification.name}: {detail}")
                elif in_box is False:
                    out_of_box += 1
                    print(f"  !!! OUT-OF-BOX {specification.name}: {detail}")
                else:
                    other += 1
                    print(f"  ?? {specification.name}: {detail}")
                reproduced = True
                break
            if not reproduced:
                writer.writerow(
                    [bench, model.name, specification.name, "not-repro", "", "", ""]
                )
                handle.flush()
                not_reproduced += 1
            if (index + 1) % 25 == 0:
                print(
                    f"  .. {index + 1}/{len(rows)} "
                    f"(valid={valid} out={out_of_box} norepro={not_reproduced})"
                )

    print(
        f"{bench}: SAT rows audited={len(rows)} in-box-CE={valid} "
        f"OUT-OF-BOX={out_of_box} errored={errored} other={other} "
        f"not-reproduced={not_reproduced}"
    )
    conclusion, exit_code = _audit_conclusion(
        valid, out_of_box, errored, other, not_reproduced
    )
    print(f"{conclusion} -> {output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
