#!/usr/bin/env python3
"""Run immutable high-impact experiments; this launcher is not proof authority."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from high_impact_experiments.evidence import ExperimentError, strict_json
from high_impact_experiments.process import worker
from high_impact_experiments.runner import execute, freeze


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    if len(sys.argv) == 3 and sys.argv[1] == "__worker":
        return worker(strict_json(Path(sys.argv[2])))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--mode", choices=("measure", "correctness"), default="measure")
    args = parser.parse_args()
    try:
        if args.command == "check":
            plan, evidence = freeze(args.plan.resolve(strict=True), repo)
            print(json.dumps({"status": "frozen_plan_valid", "rows": len(evidence["rows"]),
                              "identity_sha256": evidence["identity_sha256"],
                              "note": "no binary execution or certificate claim"}, sort_keys=True))
            return 0
        if args.archive is None:
            parser.error("run requires --archive outside the source worktree")
        result = execute(args.plan.resolve(strict=True), args.archive, repo=repo, mode=args.mode)
        print(json.dumps({"status": result["status"], "counts": result.get("counts", {}),
                          "archive": str(args.archive.absolute())}, sort_keys=True))
        disqualified = any(result.get("counts", {}).get(key, 0) for key in ("void", "refused", "unqualified"))
        return 0 if result["status"] == "completed" and not disqualified else 2
    except (ExperimentError, OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"experiment refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
