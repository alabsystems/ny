#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Compare every actual NY decoder output with a pinned original runtime.

This is diagnostic qualification, never a numerical or trajectory certificate.
Tolerances are declared here before collecting NY observations. No top-K or
last-position-only comparison may substitute for complete tensor comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ATOL = 2e-4
RTOL = 2e-4


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def owned_file(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("archive file must be a direct child")
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("archive file missing or symlinked")
    return path


def compare(reference: Path, observed: Path, binary: Path, commit: str) -> dict:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("an exact committed source frontier is required")
    source = owned_file(reference, "reference_manifest.json")
    run_path = owned_file(observed, "run.json")
    original = json.loads(source.read_text())
    run = json.loads(run_path.read_text())
    if original.get("certificate_authority") is not False:
        raise ValueError("reference must explicitly disclaim certificate authority")
    fixtures, prompts = original["fixtures"], run["prompts"]
    if not fixtures or len(fixtures) != len(prompts):
        raise ValueError("every original prompt must be observed")
    vocab, hidden = original["head"]["vocab_size"], original["head"]["hidden_size"]
    # Include inputs, executable, comparator and all observations in one stable
    # before/after frontier. A changing input invalidates the whole comparison.
    files = [source, run_path, binary, Path(__file__)]
    model_directory = Path(run["model_directory"])
    for name, expected_hash in original["original_files"].items():
        path = owned_file(model_directory, name)
        if digest(path) != expected_hash:
            raise ValueError("NY model input differs from the pinned original runtime")
        files.append(path)
    for fixture, prompt in zip(fixtures, prompts):
        files.append(owned_file(reference, fixture["tensors"]))
        files.append(owned_file(observed, prompt["manifest_file"]))
        files.extend(owned_file(observed, t["file"]) for t in prompt["tensors"])
    before = {str(p.resolve()): digest(p) for p in files}
    records = []
    passed = True
    for fixture, prompt in zip(fixtures, prompts):
        ids = fixture["input_ids"]
        if (prompt["token_ids"] != ids
                or prompt["positions"] != list(range(len(ids)))
                or prompt["name"] != fixture["prompt"]):
            raise ValueError("original token IDs, positions or prompt order changed")
        archive = owned_file(reference, fixture["tensors"])
        if digest(archive) != fixture["sha256"]:
            raise ValueError("original reference tensor hash changed")
        manifest = json.loads(owned_file(observed, prompt["manifest_file"]).read_text())
        if (manifest["parameters"]["vocab_size"] != vocab
                or manifest["parameters"]["hidden_size"] != hidden
                or manifest["positions"] != prompt["positions"]
                or manifest["lowering_digest"] != prompt["lowering_digest"]):
            raise ValueError("lowered vocabulary/hidden dimensions changed")
        count = fixture["raw_block_count"]
        if manifest["parameters"]["num_layers"] != count:
            raise ValueError("original block count differs from lowering")
        aliases = [f"block.{i}" for i in range(count)] + ["hidden", "logits"]
        if [t["output_alias"] for t in prompt["tensors"]] != aliases:
            raise ValueError("all blocks, final hidden and full logits are required exactly once")
        with np.load(archive, allow_pickle=False) as tensors:
            if tensors["input_ids"].tolist() != [ids]:
                raise ValueError("reference token tensor changed")
            for record, alias in zip(prompt["tensors"], aliases):
                width = vocab if alias == "logits" else hidden
                shape = [len(ids), width]
                if (record["shape"] != shape or record["dtype"] != "F32"
                        or record["byte_order"] != "little"
                        or record["graph_shape"] not in (shape, [1] + shape)
                        or record["elements"] != len(ids) * width
                        or record["bytes"] != len(ids) * width * 4):
                    raise ValueError("complete observed shape/dtype/size required")
                path = owned_file(observed, record["file"])
                if path.stat().st_size != record["bytes"]:
                    raise ValueError("observed tensor truncated or oversized")
                value = np.fromfile(path, dtype="<f4").reshape(shape)
                key = ("hidden_" + str(fixture["hidden_count"] - 1) if alias == "hidden"
                       else alias.replace(".", "_"))
                expected = tensors[key]
                if expected.shape != (1, *shape):
                    raise ValueError("original full tensor shape changed")
                expected = expected[0]
                if not np.isfinite(value).all() or not np.isfinite(expected).all():
                    raise ValueError("nonfinite parity tensor")
                delta = np.abs(value.astype(np.float64) - expected.astype(np.float64))
                limit = ATOL + RTOL * np.abs(expected.astype(np.float64))
                bad = int(np.count_nonzero(delta > limit))
                row = {"prompt": fixture["prompt"], "output": alias,
                       "shape": shape, "elements_compared": value.size,
                       "max_absolute_delta": float(np.max(delta)),
                       "max_tolerance_ratio": float(np.max(delta / limit)),
                       "outside_tolerance": bad, "pass": bad == 0}
                if alias == "logits":
                    original_ids = np.argmax(expected, axis=-1).tolist()
                    observed_ids = np.argmax(value, axis=-1).tolist()
                    row.update(original_argmax=original_ids, observed_argmax=observed_ids)
                    row["pass"] &= original_ids == observed_ids
                passed &= row["pass"]
                records.append(row)
    if any(digest(p) != before[str(p.resolve())] for p in files):
        raise ValueError("source, binary, reference or observation changed during comparison")
    return {"schema": "ny.decoder_parity.v1", "status": "pass" if passed else "failed",
            "claim": "diagnostic_complete_tensor_parity", "certificate_authority": False,
            "upstream_trajectory_certificate": "not_provided", "source_commit": commit,
            "binary_sha256": before[str(binary.resolve())],
            "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
            "repository": original["repository"], "revision": original["revision"],
            "full_vocabulary": vocab, "frontier_sha256": before, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.reference, args.observed, args.binary, args.source_commit)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "complete_tensors": len(report["records"]),
                      "full_vocabulary": report["full_vocabulary"],
                      "certificate_authority": False}))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
