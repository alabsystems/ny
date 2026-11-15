#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Generate conditional original-Pythia final-LayerNorm/full-head fixtures.

Reads immutable existing archives, never runs a decoder or verifier. These
supplied raw-state boxes are conditional mathematical-real properties, not
upstream trajectory or deployed floating-point normalization certificates.
Run this source-only prepared script under NYbox after the N12 CLI gates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from fractions import Fraction
from pathlib import Path

REPOSITORY = "EleutherAI/pythia-14m"
REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
VOCAB, HIDDEN, BLOCKS = 50304, 128, 6
RADII = ("0", "1e-6", "1e-5", "1e-4")
PINNED = {"numpy": "2.0.2", "safetensors": "0.6.2"}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def owned_file(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("archive inputs must be direct children")
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked archive input: {path}")
    return path


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def decimal_box(anchor: list[float], radius: str) -> tuple[list[float], list[float]]:
    """Outward binary64 endpoints of exact decimal-radius real intervals."""
    r = Fraction(radius)
    lower, upper = [], []
    for coordinate in anchor:
        center = Fraction.from_float(coordinate)
        lo_real, hi_real = center - r, center + r
        lo, hi = float(lo_real), float(hi_real)
        if Fraction.from_float(lo) > lo_real:
            lo = math.nextafter(lo, -math.inf)
        if Fraction.from_float(hi) < hi_real:
            hi = math.nextafter(hi, math.inf)
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError("nonfinite generated input box")
        if Fraction.from_float(lo) > lo_real or Fraction.from_float(hi) < hi_real:
            raise ValueError("decimal box did not enclose its declared interval")
        lower.append(lo)
        upper.append(hi)
    return lower, upper


def generate(args: argparse.Namespace) -> None:
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    if versions != PINNED:
        raise ValueError(f"fixture decoding environment differs: {versions}")
    import numpy as np
    from safetensors import safe_open

    frontier: dict[str, str] = {}

    def bind(path: Path, expected: str | None = None) -> str:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or symlinked input: {path}")
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"original source digest differs: {path}")
        frontier[str(path.resolve())] = actual
        return actual

    bind(Path(__file__).resolve())
    reference_path = owned_file(args.reference, "reference_manifest.json")
    bind(reference_path)
    original = json.loads(reference_path.read_text())
    if (original.get("repository") != REPOSITORY or original.get("revision") != REVISION
            or original.get("certificate_authority") is not False
            or original.get("claim") != "diagnostic_pinned_official_cpu_float32_eager"
            or original["head"]["vocab_size"] != VOCAB
            or original["head"]["hidden_size"] != HIDDEN
            or len(original["fixtures"]) != 3):
        raise ValueError("expected the complete pinned three-prompt Pythia archive")
    for name, digest in original["original_files"].items():
        bind(owned_file(args.model_directory, name), digest)
    config_path = owned_file(args.model_directory, "config.json")
    config = json.loads(config_path.read_text())
    if (config["hidden_size"] != HIDDEN or config["vocab_size"] != VOCAB
            or config["num_hidden_layers"] != BLOCKS
            or config.get("model_type") != "gpt_neox"):
        raise ValueError("original Pythia configuration changed")
    epsilon = float(config["layer_norm_eps"])
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid authored original LayerNorm epsilon")

    weights = owned_file(args.reference, "head_weights.f64le")
    bias = owned_file(args.reference, "head_bias.f64le")
    bind(weights, original["head"]["weights_sha256"])
    bind(bias, original["head"]["bias_sha256"])
    if weights.stat().st_size != VOCAB * HIDDEN * 8 or bias.stat().st_size != VOCAB * 8:
        raise ValueError("complete original vocabulary head shape changed")

    # Check all original rows, including their low bits, rather than trusting
    # only a head dimension or a selected candidate subset. Chunked conversion
    # has no full second binary64 vocabulary matrix.
    model_path = owned_file(args.model_directory, "model.safetensors")
    archived_weights = np.memmap(weights, mode="r", dtype="<f8", shape=(VOCAB, HIDDEN))
    archived_bias = np.memmap(bias, mode="r", dtype="<f8", shape=(VOCAB,))
    with safe_open(str(model_path), framework="numpy") as tensors:
        keys = set(tensors.keys())
        head_slice = tensors.get_slice("embed_out.weight")
        if head_slice.get_shape() != [VOCAB, HIDDEN]:
            raise ValueError("original output head is not the complete vocabulary")
        for start in range(0, VOCAB, 256):
            end = min(start + 256, VOCAB)
            row = np.asarray(head_slice[start:end], dtype="<f8")
            if not np.isfinite(row).all() or not np.array_equal(
                    row.view("<u8"), archived_weights[start:end].view("<u8")):
                raise ValueError("archived f64 head differs from original safetensors")
        original_bias = (tensors.get_tensor("embed_out.bias").astype("<f8")
                         if "embed_out.bias" in keys else np.zeros(VOCAB, dtype="<f8"))
        if not np.array_equal(original_bias.view("<u8"), archived_bias.view("<u8")):
            raise ValueError("archived head bias differs from original model")
        gamma = tensors.get_tensor("gpt_neox.final_layer_norm.weight").astype("<f8")
        beta = tensors.get_tensor("gpt_neox.final_layer_norm.bias").astype("<f8")
    if gamma.shape != (HIDDEN,) or beta.shape != (HIDDEN,) or not (
            np.isfinite(gamma).all() and np.isfinite(beta).all()):
        raise ValueError("original final LayerNorm parameter shape or values")

    observed_run = None
    observed_identity = None
    if args.observed is not None:
        if args.observed_binary_manifest is None:
            raise ValueError("observed raw-state hull requires its pinned binary manifest")
        run_path = owned_file(args.observed, "run.json")
        bind(run_path)
        observed_run = json.loads(run_path.read_text())
        bind(args.observed_binary_manifest)
        observed_identity = json.loads(args.observed_binary_manifest.read_text())
        commit = observed_identity["source_commit"]
        if (len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)
                or observed_identity.get("dirty") != ""
                or len(observed_run["prompts"]) != 3):
            raise ValueError("observations lack a clean pinned three-prompt source frontier")
        bind(Path(observed_identity["parity_binary"]), observed_identity["parity_sha256"])
        for name, digest in original["original_files"].items():
            bind(owned_file(Path(observed_run["model_directory"]), name), digest)
    elif args.observed_binary_manifest is not None:
        raise ValueError("binary manifest requires --observed")

    # A new destination prevents mutation of archived experiment evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    gamma_path = args.output / "original_final_layernorm_gamma.f64le"
    beta_path = args.output / "original_final_layernorm_beta.f64le"
    with gamma_path.open("xb") as stream:
        stream.write(gamma.tobytes(order="C"))
    with beta_path.open("xb") as stream:
        stream.write(beta.tobytes(order="C"))
    normalization = {"kind": "LayerNorm", "arithmetic": "ExactReal",
                     "input_domain": "pre_normalization_box",
                     "gamma_f64le": str(gamma_path.resolve()),
                     "beta_f64le": str(beta_path.resolve()), "epsilon": epsilon}
    records = []
    for index, fixture in enumerate(original["fixtures"]):
        archive = owned_file(args.reference, fixture["tensors"])
        bind(archive, fixture["sha256"])
        ids = fixture["input_ids"]
        with np.load(archive, allow_pickle=False) as arrays:
            raw = arrays["block_5"]
            logits = arrays["logits"]
            if (raw.shape != (1, len(ids), HIDDEN) or raw.dtype != np.dtype("float32")
                    or logits.shape != (1, len(ids), VOCAB)
                    or arrays["input_ids"].tolist() != [ids]
                    or fixture["raw_block_count"] != BLOCKS
                    or fixture["full_vocab"] != VOCAB
                    or not np.isfinite(raw).all() or not np.isfinite(logits).all()
                    or int(np.argmax(logits[0, -1])) != fixture["target"]):
                raise ValueError("original raw-state/full-vocabulary fixture changed")
            anchor = raw[0, -1].astype(np.float64).tolist()
        requests = []

        def emit(label: str, lower: list[float], upper: list[float], domain: dict) -> None:
            path = args.output / f"fixture_{index}_normalized_{label}.json"
            request = {"vocab_size": VOCAB, "hidden_size": HIDDEN,
                       "weights_f64le": str(weights.resolve()), "bias_f64le": str(bias.resolve()),
                       "arithmetic": "ExactReal", "target": fixture["target"],
                       "lo": lower, "hi": upper, "anchor": anchor,
                       "final_normalization": normalization}
            write_json(path, request)
            requests.append({"path": path.name, "sha256": sha256(path), "domain": domain,
                             "upstream_enclosure": "not_provided",
                             "normalization_execution_enclosure": "not_provided"})

        for radius in RADII:
            lower, upper = decimal_box(anchor, radius)
            emit(f"radius_{radius}", lower, upper,
                 {"kind": "exact_decimal_linf_radius", "radius": radius,
                  "center": "original_official_raw_block5_last_position"})
        if observed_run is not None:
            prompt = observed_run["prompts"][index]
            if (prompt["name"] != fixture["prompt"] or prompt["token_ids"] != ids
                    or prompt["positions"] != list(range(len(ids)))):
                raise ValueError("observed prompt IDs or positions differ")
            lowering_path = owned_file(args.observed, prompt["manifest_file"])
            bind(lowering_path)
            lowering = json.loads(lowering_path.read_text())
            if (lowering["parameters"]["vocab_size"] != VOCAB
                    or lowering["parameters"]["hidden_size"] != HIDDEN
                    or lowering["parameters"]["num_layers"] != BLOCKS
                    or lowering["lowering_digest"] != prompt["lowering_digest"]):
                raise ValueError("observed original model/lowering changed")
            candidates = [record for record in prompt["tensors"]
                          if record["output_alias"] == "block.5"]
            if len(candidates) != 1:
                raise ValueError("exactly one original final raw block observation required")
            record = candidates[0]
            if (record["dtype"] != "F32" or record["byte_order"] != "little"
                    or record["shape"] != [len(ids), HIDDEN]
                    or record["elements"] != len(ids) * HIDDEN
                    or record["bytes"] != len(ids) * HIDDEN * 4):
                raise ValueError("observed raw-state shape/dtype changed")
            observed_path = owned_file(args.observed, record["file"])
            bind(observed_path)
            if observed_path.stat().st_size != record["bytes"]:
                raise ValueError("observed raw-state file size changed")
            value = np.fromfile(observed_path, dtype="<f4").reshape(record["shape"])
            if not np.isfinite(value).all():
                raise ValueError("nonfinite observed raw state")
            last = value[-1].astype(np.float64).tolist()
            emit("observed_raw_hull", [min(x, y) for x, y in zip(anchor, last)],
                 [max(x, y) for x, y in zip(anchor, last)],
                 {"kind": "coordinate_hull_of_two_finite_diagnostic_observations",
                  "observed_max_abs_last_position_drift": max(abs(x-y) for x, y in zip(anchor, last)),
                  "trajectory_enclosure": "not_established"})
        records.append({"prompt": fixture["prompt"], "input_ids": ids,
                        "original_target": fixture["target"], "requests": requests})

    if any(sha256(Path(path)) != expected for path, expected in frontier.items()):
        raise ValueError("model/reference/observation/script frontier changed during generation")
    manifest = {"schema": "ny.pythia_normalized_head_fixtures.v1", "repository": REPOSITORY,
                "revision": REVISION, "full_vocabulary": VOCAB, "hidden_size": HIDDEN,
                "claim": "conditional_mathematical_real_original_final_normalization_and_full_head",
                "certificate_authority": False, "upstream_trajectory_certificate": "not_provided",
                "normalization_execution_certificate": "not_provided",
                "performance_claim": False, "packages": versions,
                "authored_layer_norm_epsilon": epsilon,
                "decoded_parameters": {"gamma_sha256": sha256(gamma_path),
                                       "beta_sha256": sha256(beta_path),
                                       "original_tensor_keys": ["gpt_neox.final_layer_norm.weight",
                                                                "gpt_neox.final_layer_norm.bias"]},
                "original_frontier_sha256": frontier, "observed_binary_identity": observed_identity,
                "anchor_semantics": "legacy_metadata_only;normalized_proof_uses_lo_and_hi",
                "fixtures": records}
    write_json(args.output / "fixture_manifest.json", manifest)
    print(json.dumps({"status": "fixtures_generated", "requests": sum(len(r["requests"]) for r in records),
                      "full_vocabulary": VOCAB, "certificate_authority": False}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observed", type=Path)
    parser.add_argument("--observed-binary-manifest", type=Path)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
