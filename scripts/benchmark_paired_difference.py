#!/usr/bin/env python3
"""Measure the native paired certifier without presenting harness runs as submissions.

Requires exact exported PAIRED_V1 inputs and records executable/input hashes,
completed stdout, return codes, and external elapsed time. Historical bank
comparisons are explicitly distinct from a same-binary paired/plain ablation.
"""
import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


PROOF_SOURCES = (
    "crates/ny-propagate/src/paired_relu_hull.rs",
    "crates/ny-propagate/src/paired_difference/engine.rs",
    "crates/ny-propagate/src/paired_difference/engine_tests.rs",
    "crates/ny-propagate/src/paired_difference/forward.rs",
    "crates/ny-propagate/src/paired_difference/forward_affine_hull.rs",
    "crates/ny-propagate/src/paired_difference/forward_coupled_hull.rs",
    "crates/ny-propagate/src/paired_difference/forward_meet.rs",
    "crates/ny-propagate/src/paired_difference/lp_certificate.rs",
    "crates/ny-propagate/src/paired_difference/lp_model.rs",
    "crates/ny-propagate/src/paired_difference/lp_affine_hull.rs",
    "crates/ny-propagate/src/paired_difference/lp_coupled_affine.rs",
    "crates/ny-propagate/src/paired_difference/lp_cover.rs",
    "crates/ny-propagate/src/paired_difference/lp_group.rs",
    "crates/ny-propagate/src/paired_difference/lp_template.rs",
    "crates/ny-propagate/src/paired_difference/lp_template_tests.rs",
    "crates/ny-propagate/src/paired_difference/lp_template_output.rs",
    "crates/ny-propagate/src/paired_difference/lp_template_output_tests.rs",
    "crates/ny-propagate/src/paired_difference/lp_template_soft_guard.rs",
    "crates/ny-propagate/src/paired_difference/lp_template_soft_guard_tests.rs",
    "crates/ny-propagate/src/paired_difference/lp_phase.rs",
    "crates/ny-propagate/src/paired_difference/lp_native.rs",
    "crates/ny-propagate/src/paired_difference/lp_optimize.rs",
    "crates/ny-propagate/src/paired_difference/refine.rs",
    "crates/ny-propagate/src/paired_difference/refine_tests.rs",
    "crates/ny-propagate/src/paired_difference/search.rs",
    "crates/ny-propagate/src/paired_difference/search_tests.rs",
    "crates/ny-propagate/src/paired_difference/chain.rs",
    "crates/ny-propagate/src/paired_difference/mod.rs",
    "tools/paired-difference/src/main.rs",
    "tools/paired-difference/Cargo.toml",
    "tools/paired-difference/Cargo.lock",
    "tools/paired-difference/build.rs",
    "scripts/export_paired_difference_instances.py",
    "scripts/benchmark_paired_difference.py",
)


def source_provenance(root):
    result = {}
    for relative in PROOF_SOURCES:
        path = root / relative
        result[relative] = {"path": str(path.resolve()), "sha256": sha256(path)}
    return result


def validate_export(index, path, benchmark_root=None, manifest_row=None):
    """Check available exporter receipts before trusting an input identity.

    Receipt hashes establish identity, not correctness of the proof kernel or
    a correspondence between a binary and a source snapshot.
    """
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        if manifest_row is not None:
            raise ValueError(f"missing manifest-backed exporter receipt for row {index}")
        return {"status": "no_exporter_receipt", "source_hash_checks": []}
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("index") != index or metadata.get("export_sha256") != sha256(path):
        raise ValueError(f"export identity/hash mismatch on row {index}")
    if manifest_row is not None and manifest_row != metadata:
        raise ValueError(f"manifest and row receipt disagree on row {index}")
    checks = []
    if benchmark_root is not None:
        models = ast.literal_eval(metadata["models"])
        if len(models) != 2 or [m[0] for m in models] != ["f", "g"]:
            raise ValueError(f"unexpected source model tuple on row {index}")
        root = benchmark_root.resolve()
        for label, relative, field in (
            ("f", models[0][1], "f_sha256"),
            ("g", models[1][1], "g_sha256"),
            ("property", metadata["property"], "property_sha256"),
        ):
            source = (root / relative).resolve()
            if not source.is_relative_to(root):
                raise ValueError("source receipt escapes the benchmark root")
            actual = sha256(source)
            if metadata.get(field) != actual:
                raise ValueError(f"source {label} hash mismatch on row {index}")
            checks.append({"source": label, "path": str(source), "sha256": actual})
    return {"status": "verified", "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": sha256(metadata_path), "metadata": metadata,
            "source_hash_checks": checks,
            "source_hash_validation": "verified" if benchmark_root is not None else "source_paths_unavailable"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rows", default="", help="comma separated row numbers; absent runs every exported row")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--domain-cap", type=int, default=100000)
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--native-lp", action="store_true", help="native joint relaxation with source dual replay")
    ap.add_argument("--refined-search", action="store_true", help="search with source-owned root intermediate references")
    ap.add_argument("--lp-steps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8, help="explicit native worker count in [1,16]")
    ap.add_argument("--bank", type=Path, help="explicit historical bank; required for additional-proofs comparison")
    ap.add_argument("--benchmark-root", type=Path, help="validate source ONNX/property and instance-list hashes against exporter receipts")
    ap.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--validate-only", action="store_true", help="validate identities/provenance and write evidence without executing the certifier")
    args = ap.parse_args()
    if sum([args.native_lp, args.plain, args.refined_search]) > 1 or not 0 <= args.lp_steps <= 100000:
        ap.error("native LP has no plain ablation; LP steps must be in [0,100000]")
    if not 0 < args.seconds <= 100 or not 0 < args.domain_cap <= 2000000:
        ap.error("seconds must be in (0,100], domain cap in [1,2000000]")
    if not 1 <= args.workers <= 16:
        ap.error("workers must be in [1,16]")
    files = {}
    for path in list(args.inputs.glob("instance_*.txt")) + list(args.inputs.glob("instance_*.paired")):
        match = re.fullmatch(r"instance_(\d+)", path.stem)
        if match is None:
            continue
        index = int(match[1])
        if index in files:
            ap.error(f"duplicate exported formats for row {index}")
        files[index] = path
    selected = sorted(files) if not args.rows else [int(s) for s in args.rows.split(",")]
    if not selected or any(i not in files for i in selected):
        ap.error("nonempty exported row selection required")
    bank = {}
    if args.bank:
        with args.bank.open(newline="") as stream:
            for row in csv.reader(stream):
                match = re.search(r"instance_(\d+)\.vnnlib", row[2])
                if match:
                    if int(match[1]) in bank:
                        ap.error("historical bank contains duplicate instance identities")
                    bank[int(match[1])] = row[4]
        if any(index not in bank for index in selected):
            ap.error("historical bank does not cover every selected row")
    manifest_path = args.inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    manifest_rows = {row["index"]: row for row in manifest["rows"]} if manifest else {}
    if manifest and any(index not in manifest_rows for index in selected):
        ap.error("exporter manifest does not cover every selected row")
    if manifest and args.benchmark_root:
        if sha256(args.benchmark_root / "instances.csv") != manifest.get("instances_sha256"):
            ap.error("source instances.csv hash differs from exporter manifest")
    source_hashes = source_provenance(args.source_root.resolve())
    receipts = {index: validate_export(index, files[index], args.benchmark_root, manifest_rows.get(index)) for index in selected}
    binary = args.binary.resolve()
    executable_hash = sha256(binary)
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "scope": "standalone identical-source certificate harness; not a VNN-COMP protocol run",
        "binary": str(binary), "binary_sha256": executable_hash,
        "paired": not args.plain, "seconds": args.seconds, "domain_cap": args.domain_cap,
        "native_lp": args.native_lp, "lp_steps": args.lp_steps,
        "refined_search": args.refined_search,
        "workers": args.workers,
        "historical_bank": str(args.bank) if args.bank else None,
        "historical_bank_sha256": sha256(args.bank) if args.bank else None,
        "historical_comparison_available": args.bank is not None,
        "source_hashes_snapshot": source_hashes,
        "binary_source_correspondence": "source hashes record the measurement snapshot; hashes alone do not establish which source produced the executable",
        "exporter_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path), "metadata": manifest} if manifest else None,
        "benchmark_root": str(args.benchmark_root.resolve()) if args.benchmark_root else None,
        "validate_only": args.validate_only,
        "rows": [],
    }
    if args.validate_only:
        evidence["export_validation"] = receipts
        (args.output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(f"validated={len(receipts)} executed=0 historical_comparison={args.bank is not None}")
        return
    for index in selected:
        path = files[index].resolve()
        command = [str(binary), str(path), str(args.seconds), str(args.domain_cap),
                   "lp" if args.native_lp else ("refined" if args.refined_search else ("plain" if args.plain else "paired")), str(args.workers)]
        if args.native_lp: command.append(str(args.lp_steps))
        start = time.monotonic()
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.seconds + 2)
            stdout, stderr, code = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = "external harness deadline exceeded"
            code = 124
        elapsed = time.monotonic() - start
        fields = dict(re.findall(r"([a-z_]+)=([^\s]+)", stdout))
        if code == 0 and fields.get("workers") != str(args.workers):
            raise RuntimeError("certifier did not confirm the requested worker count; executable may use an older protocol")
        proved = code == 0 and fields.get("proved") == "true" and elapsed <= args.seconds + 1
        entry = {"row": index, "input_sha256": sha256(path), "command": command,
                 "return_code": code, "wall_seconds": elapsed, "proved": proved,
                 "historical_bank_verdict": bank.get(index), "fields": fields,
                 "stdout": stdout, "stderr": stderr, "exporter_receipt": receipts[index]}
        evidence["rows"].append(entry)
        (args.output / f"instance_{index}.log").write_text(stdout + stderr)
        (args.output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(f"row={index} proved={proved} wall_s={elapsed:.3f} historical={bank.get(index)} {fields.get('reason', 'error')} evaluated={fields.get('evaluated', 'n/a')}", flush=True)
        if proved and bank.get(index) == "sat":
            raise RuntimeError(f"SOUNDNESS CONFLICT on historical SAT row {index}; stop and investigate")
        if sha256(binary) != executable_hash:
            raise RuntimeError("executable changed during measurement")
        if source_provenance(args.source_root.resolve()) != source_hashes:
            raise RuntimeError("certificate source changed during measurement; recorded run is incomplete")
        validate_export(index, path, args.benchmark_root, manifest_rows.get(index))
    gains = [r["row"] for r in evidence["rows"] if r["proved"] and r["historical_bank_verdict"] in {"timeout", "unknown", "error"}]
    evidence["summary"] = {"proved": sum(r["proved"] for r in evidence["rows"]), "rows": len(selected),
                           "additional_proofs_vs_historical_bank": gains if args.bank else None,
                           "historical_comparison_available": args.bank is not None,
                           "score_claim": "none; standalone proofs are not adjudicated competition points"}
    (args.output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
