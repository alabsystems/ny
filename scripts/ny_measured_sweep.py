#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Run a legacy, unsealed NY sweep over one VNN-COMP category.

The output uses the organizer-compatible six-column row shape, but it is not
provenance-complete evidence. Use ``scripts/measure_ny_scorecard.sh`` for any
published score claim. This helper remains useful for quick local triage.

By default each row uses its official ``instances.csv`` budget. ``--timeout N``
sets a lower-bound cap of ``min(official_budget, N)``; it never grants a row more
than its official budget.

BANKING GATE: an ``unsat`` on an instance the VNN-COMP field already falsified
with an ACCEPTED counterexample is refused here, before it can reach a ledger.
See scripts/emit_ce_falsified_watchlist.py for why the published ``unsat`` label
cannot be trusted on those rows. The gate fails CLOSED: a missing watchlist
aborts the sweep rather than reading as "nothing is listed".
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_TOKENS = frozenset({"sat", "unsat", "unknown", "timeout", "error"})

_WATCHLIST_PATH = REPO_ROOT / "scripts" / "emit_ce_falsified_watchlist.py"
_spec = importlib.util.spec_from_file_location(
    "emit_ce_falsified_watchlist", _WATCHLIST_PATH
)
if _spec is None or _spec.loader is None:  # pragma: no cover - repo layout invariant
    raise SystemExit(f"cannot import the watchlist module at {_WATCHLIST_PATH}")
watchlist_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = watchlist_mod
_spec.loader.exec_module(watchlist_mod)

# Only `unsat` contradicts an accepted counterexample; `sat`/`timeout`/`unknown`
# on a falsified row are legitimate outcomes and must still bank.
_UNSAT_TOKENS = frozenset({"unsat", "holds"})


class SweepError(RuntimeError):
    """The sweep configuration or an instance row is invalid."""


def _instance_key(onnx: str, vnnlib: str, occurrence: int) -> tuple[str, str, int]:
    """The scorecard's subdir-aware, occurrence-aware instance key."""
    return (*watchlist_mod.scorecard.key(onnx, vnnlib), occurrence)


def refused_falsified_unsats(
    category: str,
    rows: Sequence[tuple[str, str, int]],
    results: dict[int, tuple[str, float]],
    watchlist: dict[str, set[tuple[str, str, int]]],
) -> list[tuple[str, str, str]]:
    """Rows this sweep must NOT bank: ``unsat`` on a field-falsified instance."""
    listed = watchlist.get(category, set())
    if not listed:
        return []
    occurrences: Counter[tuple[str, str]] = Counter()
    refused: list[tuple[str, str, str]] = []
    for index, (onnx, vnnlib, _timeout) in enumerate(rows):
        base = watchlist_mod.scorecard.key(onnx, vnnlib)
        occurrence = occurrences[base]
        occurrences[base] += 1
        token = (results.get(index, ("unknown", 0.0))[0] or "").strip().lower()
        if token not in _UNSAT_TOKENS:
            continue
        if (*base, occurrence) in listed:
            refused.append((onnx, vnnlib, token))
    return refused


def _repo_relative_or_absolute(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_instance_input(
    benchmark_dir: Path, raw_path: str, suffix: str
) -> tuple[str, Path]:
    value = raw_path.strip()
    if not value or "\\" in value:
        raise SweepError(f"invalid benchmark path {raw_path!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SweepError(f"benchmark path escapes category: {raw_path!r}")
    normalized = PurePosixPath(
        *(part for part in relative.parts if part != ".")
    ).as_posix()
    if not normalized.endswith(suffix):
        raise SweepError(f"benchmark path lacks {suffix}: {raw_path!r}")
    resolved = (benchmark_dir / normalized).resolve()
    try:
        resolved.relative_to(benchmark_dir)
    except ValueError as error:
        raise SweepError(f"benchmark path escapes category: {raw_path!r}") from error
    if not resolved.is_file():
        raise SweepError(f"benchmark input is missing: {resolved}")
    return normalized, resolved


def _official_budget(raw: str, source: str) -> int:
    try:
        numeric = float(raw.strip())
    except ValueError as error:
        raise SweepError(f"{source}: invalid official timeout {raw!r}") from error
    if not math.isfinite(numeric) or numeric < 1:
        raise SweepError(f"{source}: official timeout must be finite and positive")
    budget = int(numeric)
    if budget < 1:
        raise SweepError(f"{source}: official timeout floors below one second")
    return budget


def load_instances(
    instances_csv: Path, benchmark_dir: Path, timeout_cap: int
) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    with instances_csv.open(newline="", encoding="utf-8") as handle:
        for line_number, fields in enumerate(csv.reader(handle), 1):
            if not fields or not any(field.strip() for field in fields):
                continue
            if len(fields) < 3:
                raise SweepError(
                    f"{instances_csv}:{line_number}: expected ONNX, VNN-LIB, timeout"
                )
            onnx, _ = _resolve_instance_input(benchmark_dir, fields[0], ".onnx")
            vnnlib, _ = _resolve_instance_input(benchmark_dir, fields[1], ".vnnlib")
            official = _official_budget(fields[2], f"{instances_csv}:{line_number}")
            timeout = min(official, timeout_cap) if timeout_cap else official
            rows.append((onnx, vnnlib, timeout))
    if not rows:
        raise SweepError(f"instances file has no runnable rows: {instances_csv}")
    return rows


def run_instance(
    ny: Path,
    corpus: Path,
    category: str,
    onnx: str,
    vnnlib: str,
    timeout: int,
    watchdog_grace: int = 15,
) -> tuple[str, float]:
    """Run one instance with an invocation-unique result file."""
    benchmark_dir = corpus / category
    result_fd, result_name = tempfile.mkstemp(
        prefix="ny-measured-sweep-", suffix=".result"
    )
    os.close(result_fd)
    result_path = Path(result_name)
    command = [
        str(ny),
        "vnncomp",
        "v1",
        category,
        onnx,
        vnnlib,
        str(result_path),
        str(timeout),
    ]
    started = time.monotonic()
    try:
        try:
            completed = subprocess.run(
                command,
                cwd=benchmark_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout + watchdog_grace,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "timeout", time.monotonic() - started
        except OSError:
            return "error", time.monotonic() - started

        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            return "error", elapsed
        # The subprocess grace exists only so NY can shut down and publish its
        # timeout state cleanly. A decision returned during that grace is not an
        # official-budget solve and must never be scored as SAT/UNSAT.
        if elapsed > timeout:
            return "timeout", elapsed
        try:
            token = (
                result_path.read_text(encoding="utf-8").splitlines()[0].strip().lower()
            )
        except (OSError, IndexError, UnicodeError):
            return "error", elapsed
        return (token, elapsed) if token in RESULT_TOKENS else ("error", elapsed)
    finally:
        result_path.unlink(missing_ok=True)


def _write_results_atomic(
    output_csv: Path,
    category: str,
    rows: Sequence[tuple[str, str, int]],
    results: dict[int, tuple[str, float]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=output_csv.parent,
            prefix=f".{output_csv.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            writer = csv.writer(temporary)
            for index, (onnx, vnnlib, _timeout) in enumerate(rows):
                token, runtime = results[index]
                counts[token] = counts.get(token, 0) + 1
                writer.writerow(
                    [category, onnx, vnnlib, "prepared", token, f"{runtime:.2f}"]
                )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output_csv)
    except BaseException:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        raise
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="optional lower-bound cap; 0 uses each official row budget",
    )
    parser.add_argument("--limit", type=int, default=0, help="row cap; 0 means all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--watchdog-grace", type=int, default=15)
    parser.add_argument("--corpus", default="benchmarks/vnncomp2025/benchmarks")
    parser.add_argument("--ny", default="target/release/ny")
    parser.add_argument("--out", default="reports/measured")
    parser.add_argument(
        "--watchlist",
        default=str(watchlist_mod.DEFAULT_OUT),
        help=(
            "field-falsified watchlist JSON; an unsat on any listed instance is "
            "REFUSED and the ledger is not written"
        ),
    )
    args = parser.parse_args(argv)

    if args.timeout < 0:
        parser.error("--timeout must be non-negative")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.watchdog_grace < 0:
        parser.error("--watchdog-grace must be non-negative")
    if (
        not args.category
        or Path(args.category).name != args.category
        or args.category in {".", ".."}
    ):
        parser.error("category must be one path component")

    # Fail CLOSED, and do it BEFORE burning a sweep: an absent watchlist is not
    # evidence that nothing is listed.
    watchlist_path = _repo_relative_or_absolute(args.watchlist, REPO_ROOT)
    if not watchlist_path.is_file():
        print(
            f"REFUSING TO SWEEP: no field-falsified watchlist at {watchlist_path}.\n"
            "  Without it an unsat on an instance the field already falsified with "
            "an accepted counterexample would be banked unchecked.\n"
            "  Generate it: scripts/emit_ce_falsified_watchlist.py",
            file=sys.stderr,
        )
        return 3
    watchlist = watchlist_mod.load(watchlist_path)

    corpus = _repo_relative_or_absolute(args.corpus, REPO_ROOT)
    ny = _repo_relative_or_absolute(args.ny, REPO_ROOT)
    benchmark_dir = (corpus / args.category).resolve()
    try:
        benchmark_dir.relative_to(corpus)
    except ValueError:
        parser.error("category escapes the benchmark corpus")
    if not benchmark_dir.is_dir():
        print(f"benchmark category is missing: {benchmark_dir}", file=sys.stderr)
        return 2
    if not ny.is_file() or not os.access(ny, os.X_OK):
        print(f"NY binary is missing or not executable: {ny}", file=sys.stderr)
        return 2
    instances_csv = benchmark_dir / "instances.csv"
    if not instances_csv.is_file():
        print(f"instances.csv is missing: {instances_csv}", file=sys.stderr)
        return 2

    try:
        rows = load_instances(instances_csv, benchmark_dir, args.timeout)
    except SweepError as error:
        print(f"ny_measured_sweep: error: {error}", file=sys.stderr)
        return 2
    if args.limit:
        rows = rows[: args.limit]

    output_dir = _repo_relative_or_absolute(args.out, REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{args.category}.csv"
    print(
        "WARNING: legacy unsealed sweep; use measure_ny_scorecard.sh for score evidence.",
        file=sys.stderr,
    )
    print(
        f"[{args.category}] {len(rows)} instances, timeout_cap={args.timeout}s, "
        f"workers={args.workers} -> {output_csv}",
        file=sys.stderr,
    )

    results: dict[int, tuple[str, float]] = {}
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_instance,
                ny,
                corpus,
                args.category,
                onnx,
                vnnlib,
                timeout,
                args.watchdog_grace,
            ): index
            for index, (onnx, vnnlib, timeout) in enumerate(rows)
        }
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception:
                results[index] = ("error", 0.0)
            completed_count += 1
            token, runtime = results[index]
            print(
                f"  [{completed_count}/{len(rows)}] "
                f"{Path(rows[index][1]).name:40s} {token:9s} {runtime:7.2f}s",
                file=sys.stderr,
            )

    refused = refused_falsified_unsats(args.category, rows, results, watchlist)
    if refused:
        print(
            f"\n*** BANKING REFUSED: {len(refused)} unsat verdict(s) on instances "
            f"the VNN-COMP field FALSIFIED with an ACCEPTED counterexample ***",
            file=sys.stderr,
        )
        print(
            "    Those tools were paid +10 falsifier credit for those witnesses; "
            "an ny unsat on the same instance cannot be sound, and the published "
            "'unsat' label is an artifact of process_results.py promoting "
            "true_result only on an exactly-CORRECT witness.",
            file=sys.stderr,
        )
        for onnx, vnnlib, token in refused:
            print(f"    REFUSED {token}: {onnx} {vnnlib}", file=sys.stderr)
        print(f"    {output_csv} NOT written.", file=sys.stderr)
        return 3

    counts = _write_results_atomic(output_csv, args.category, rows, results)
    wall = time.monotonic() - started
    print(f"[{args.category}] done in {wall:.1f}s counts={counts}", file=sys.stderr)
    print(
        f"  unsat={counts.get('unsat', 0)} sat={counts.get('sat', 0)} "
        f"timeout={counts.get('timeout', 0)} unknown={counts.get('unknown', 0)} "
        f"error={counts.get('error', 0)}",
        file=sys.stderr,
    )
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
