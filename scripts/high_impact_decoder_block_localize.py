#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Same-input, original-block diagnostic localization for pinned Pythia14M.

prepare writes both archived official and NY previous-block inputs, then a
request for decoder_block_observe. compare executes the unchanged official
Torch block with hooks and evaluates individual original Torch operators on
captured NY operands. NY outputs always come from its actual graph observer.
No reference value here provides a numerical certificate. Fixed tolerances
match the original full-tensor parity gate; local references rank causes.
Run through the parent's serialized ny_box correctness lane.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

ATOL = RTOL = 2e-4
PINNED = {"torch": "2.8.0", "transformers": "4.57.1", "numpy": "2.0.2",
          "tokenizers": "0.22.1", "safetensors": "0.6.2"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def prepare(args):
    import numpy as np
    reference = json.loads((args.reference / "reference_manifest.json").read_text())
    ny_run = json.loads((args.ny_parity / "run.json").read_text())
    if args.block < 1 or args.block >= 6:
        raise ValueError("this archived-input diagnostic requires block1..5")
    args.inputs.mkdir(parents=True, exist_ok=False)
    fixtures, bindings = [], []
    for index, original in enumerate(reference["fixtures"]):
        archive_path = args.reference / original["tensors"]
        if sha(archive_path) != original["sha256"]:
            raise ValueError("official archive identity changed")
        with np.load(archive_path, allow_pickle=False) as archive:
            previous = archive[f"block_{args.block - 1}"].astype("<f4", copy=False)
        recorded = ny_run["prompts"][index]
        if recorded["token_ids"] != original["input_ids"]:
            raise ValueError("original/NY input IDs differ")
        positions = recorded["positions"]
        if positions != list(range(previous.shape[1])):
            raise ValueError("official archive uses these contiguous positions only")
        record = next(r for r in recorded["tensors"] if r["output_alias"] == f"block.{args.block-1}")
        ny_path = args.ny_parity / record["file"]
        ny_previous = np.fromfile(ny_path, dtype="<f4").reshape(previous.shape)
        for kind, values in (("official", previous), ("ny", ny_previous)):
            name = f"fixture-{index:02}-{kind}"
            path = args.inputs / f"{name}.input.f32"
            if not np.isfinite(values).all():
                raise ValueError("nonfinite archived boundary")
            values.tofile(path)
            fixtures.append({"name": name, "positions": positions, "shape": list(values.shape),
                             "input_f32": str(path.resolve())})
            bindings.append({"name": name, "fixture_index": index, "input_kind": kind,
                             "input_sha256": sha(path), "official_archive": str(archive_path.resolve()),
                             "official_archive_sha256": sha(archive_path), "ny_previous_sha256": sha(ny_path)})
    json_write(args.request, {"model_directory": str(args.model.resolve()),
               "output_directory": str(args.observations.resolve()), "block_index": args.block,
               "fixtures": fixtures})
    json_write(args.inputs / "bindings.json", {"schema_version": 1, "certificate_authority": False,
               "reference_manifest": str((args.reference / "reference_manifest.json").resolve()),
               "reference_manifest_sha256": sha(args.reference / "reference_manifest.json"),
               "ny_run_sha256": sha(args.ny_parity / "run.json"), "bindings": bindings})


def compare_array(observed, reference):
    import numpy as np
    observed, reference = np.asarray(observed, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if observed.shape != reference.shape:
        raise ValueError(f"stage shape mismatch {observed.shape} vs {reference.shape}")
    if not np.isfinite(observed).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite compared stage")
    delta = np.abs(observed - reference)
    allowance = ATOL + RTOL * np.abs(reference)
    index = np.unravel_index(int(np.argmax(delta)), delta.shape)
    return {"max_absolute_delta": float(delta.max()), "max_tolerance_ratio": float((delta / allowance).max()),
            "outside_tolerance": int((delta > allowance).sum()), "elements_compared": int(delta.size),
            "max_delta_index": [int(i) for i in index], "pass": bool((delta <= allowance).all())}


def capture_official(model, block_index, boundary, positions):
    """Call the original Torch block, retaining actual module outputs."""
    import torch
    import transformers.models.gpt_neox.modeling_gpt_neox as implementation
    block = model.gpt_neox.layers[block_index]
    prefix = f"decoder.block.{block_index}"
    stages, handles = {}, []
    modules = {"input_norm": block.input_layernorm, "post_attention_norm": block.post_attention_layernorm,
               "packed_qkv": block.attention.query_key_value, "attention_output": block.attention.dense,
               "mlp_up": block.mlp.dense_h_to_4h, "gelu_erf": block.mlp.act,
               "mlp_down": block.mlp.dense_4h_to_h}
    def capture(name):
        def hook(_module, _inputs, output):
            stages[f"{prefix}.{name}"] = output.detach().clone()
        return hook
    for name, module in modules.items():
        handles.append(module.register_forward_hook(capture(name)))
    original_eager = implementation.eager_attention_forward
    def attention_observer(module, query, key, value, attention_mask, scaling, **kwargs):
        # Actual original-function inputs and returned probabilities/context.
        stages[f"{prefix}.query_rope.concat"] = query.detach().clone()
        stages[f"{prefix}.key_rope.concat"] = key.detach().clone()
        output, probabilities = original_eager(module, query, key, value, attention_mask,
                                                scaling=scaling, **kwargs)
        stages[f"{prefix}.scores"] = (torch.matmul(query, key.transpose(-2, -1)) * scaling).detach()
        stages[f"{prefix}.causal_probabilities"] = probabilities.detach().clone()
        stages[f"{prefix}.weighted_values"] = output.transpose(1, 2).detach().clone()
        stages[f"{prefix}.token_major_context"] = output.detach().clone()
        stages[f"{prefix}.flatten_context"] = output.reshape(1, boundary.shape[1], -1).detach()
        return output, probabilities
    implementation.eager_attention_forward = attention_observer
    try:
        position_ids = torch.tensor(positions, dtype=torch.long).unsqueeze(0)
        sequence = boundary.shape[1]
        causal = torch.full((sequence, sequence), torch.finfo(torch.float32).min).triu(1)[None, None]
        with torch.no_grad():
            cos, sin = model.gpt_neox.rotary_emb(boundary, position_ids)
            value = block(boundary, attention_mask=causal, position_ids=position_ids,
                          use_cache=False, position_embeddings=(cos, sin))[0]
        stages[f"{prefix}.output"] = value.detach()
        packed = stages[f"{prefix}.packed_qkv"]
        heads, dim = block.attention.config.num_attention_heads, block.attention.head_size
        stages[f"{prefix}.packed_heads"] = packed.reshape(1, sequence, heads, 3 * dim)
        viewed = stages[f"{prefix}.packed_heads"].permute(0, 2, 1, 3)
        stages[f"{prefix}.head_major_qkv"] = viewed
        for name, number in (("query", 0), ("key", 1), ("value", 2)):
            stages[f"{prefix}.{name}"] = viewed[..., number * dim:(number + 1) * dim]
        rotary = block.attention.rotary_ndims
        c, s = cos[..., :rotary//2].unsqueeze(1), sin[..., :rotary//2].unsqueeze(1)
        # These split-form intermediates use original Torch operators on the
        # captured official operands. The concat itself above is captured from
        # the unchanged official attention function, not substituted here.
        for side in ("query", "key"):
            source = stages[f"{prefix}.{side}"]
            base = f"{prefix}.{side}_rope"
            a, b = source[..., :rotary//2], source[..., rotary//2:rotary]
            stages.update({f"{base}.first_half": a, f"{base}.second_half": b,
                           f"{base}.a_cos": a*c, f"{base}.b_sin": b*s,
                           f"{base}.b_cos": b*c, f"{base}.a_sin": a*s,
                           f"{base}.rotated_first": a*c-b*s,
                           f"{base}.rotated_second": b*c+a*s})
            if rotary < dim:
                stages[f"{base}.pass"] = source[..., rotary:]
        stages[f"{prefix}.parallel_branches"] = stages[f"{prefix}.mlp_down"] + stages[f"{prefix}.attention_output"]
        return stages, modules
    finally:
        implementation.eager_attention_forward = original_eager
        for handle in handles:
            handle.remove()


def local_operator(op, operands, modules):
    """Original Torch operators on identical captured NY operands; diagnostic."""
    import torch
    import torch.nn.functional as functional
    name = op["name"].split(".", 3)[-1]
    if name in modules:
        module = modules[name]
        official = module(operands[0])
        linear64 = None
        if isinstance(module, torch.nn.Linear):
            linear64 = functional.linear(operands[0].double(), module.weight.double(),
                         None if module.bias is None else module.bias.double()).float()
        return official, linear64
    kind, params = op["operator"], op["parameters"]
    if kind == "Add": return operands[0] + operands[1], None
    if kind == "Sub": return operands[0] - operands[1], None
    if kind == "Mul": return operands[0] * operands[1], None
    if kind == "MatMul":
        right = operands[1].transpose(-2, -1) if params.get("transpose_b", 0) else operands[1]
        value = torch.matmul(operands[0], right)
        if "scale" in params: value = value * params["scale"]
        # This separates accumulation/staging from the actual standard f32
        # operation, without treating this f64 variant as kernel authority.
        wide = torch.matmul(operands[0].double(), right.double())
        if "scale" in params: wide = wide * params["scale"]
        return value, wide.float()
    if kind == "CausalSoftmax":
        sequence = operands[0].shape[-1]
        causal = torch.full((sequence, sequence), torch.finfo(torch.float32).min).triu(1)
        return functional.softmax(operands[0] + causal, dim=-1, dtype=torch.float32), None
    return None, None


def compare(args):
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    if versions != PINNED: raise ValueError(f"reference environment differs: {versions}")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    bindings = json.loads((args.inputs / "bindings.json").read_text())
    reference_path = Path(bindings["reference_manifest"])
    if sha(reference_path) != bindings["reference_manifest_sha256"]:
        raise ValueError("reference manifest changed")
    reference = json.loads(reference_path.read_text())
    for file, identity in reference["original_files"].items():
        if sha(args.model / file) != identity: raise ValueError("original model file changed")
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.float32, attn_implementation="eager").cpu().eval()
    source = Path(inspect.getfile(type(model)))
    if sha(source) != reference["implementation_sha256"]: raise ValueError("official implementation changed")
    run = json.loads((args.observations / "run.json").read_text())
    if run["certificate_authority"] or model.config.model_type != "gpt_neox":
        raise ValueError("this comparator supports original GPTNeoX diagnostic observations only")
    rows, fixtures = [], []
    for record in run["fixtures"]:
        fixture = record["fixture"]
        binding = next(b for b in bindings["bindings"] if b["name"] == fixture["name"])
        if sha(fixture["input_f32"]) != binding["input_sha256"]: raise ValueError("input file changed")
        input_np = np.fromfile(fixture["input_f32"], dtype="<f4").reshape(fixture["shape"])
        metadata = json.loads((args.observations / record["metadata_file"]).read_text())
        boundary = metadata["boundary"]
        if metadata["certificate_authority"] or boundary["certificate_authority"]:
            raise ValueError("unexpected certificate authority")
        block_index = boundary["block_index"]
        with torch.no_grad():
            official, modules = capture_official(model, block_index, torch.from_numpy(input_np), fixture["positions"])
        with np.load(binding["official_archive"], allow_pickle=False) as archive:
            archive_output = archive[f"block_{block_index}"]
        reproduction = compare_array(official[f"decoder.block.{block_index}.output"].numpy(), archive_output)
        if binding["input_kind"] == "official" and reproduction["max_absolute_delta"] != 0:
            raise ValueError("unchanged official block did not exactly reproduce original archive")
        captured, constants = {}, {}
        for constant in record["constants"]:
            path = args.observations / constant["file"]
            if sha(path) != bytes(constant["file_sha256"]).hex(): raise ValueError("table export changed")
            constants[constant["name"]] = torch.from_numpy(np.fromfile(path, dtype="<f4").reshape(constant["shape"]))
        for tensor in record["tensors"]:
            path = args.observations / tensor["file"]
            if sha(path) != bytes(tensor["file_sha256"]).hex(): raise ValueError("stage export changed")
            values = np.fromfile(path, dtype="<f4").reshape(tensor["shape"])
            captured[tensor["tensor"]] = torch.from_numpy(values.reshape(tensor["original_shape"]))
        block = next(b for b in metadata["original_lowering_manifest"]["blocks"] if b["name"] == f"block.{block_index}")
        first_local, ranked = None, []
        for op in block["operations"]:
            observed = captured[op["output"]]
            reference_stage = official.get(op["name"])
            if reference_stage is None: raise ValueError(f"missing original reference stage {op['name']}")
            total = compare_array(observed.numpy(), reference_stage.numpy())
            operands = []
            for name in op["inputs"]:
                if name == boundary["input_tensor"]: operands.append(torch.from_numpy(input_np))
                elif name in captured: operands.append(captured[name])
                elif name in constants: operands.append(constants[name])
                # Original module-owned weights are used by Linear/Norm itself.
            with torch.no_grad(): local, wide = local_operator(op, operands, modules)
            row = {"fixture": fixture["name"], "node": op["name"], "operator": op["operator"],
                   "actual_completed_arm": next(s["arm"] for s in metadata["trace"]["steps"] if s["node"] == op["name"]),
                   "official_same_block_input": total,
                   "reference_stage_kind": "captured original module/function, or original split/view operator on captured operands"}
            if local is not None:
                local_result = compare_array(observed.numpy(), local.detach().numpy())
                row["official_operator_on_identical_ny_operands"] = local_result
                if wide is not None: row["f64_operator_on_identical_ny_operands"] = compare_array(observed.numpy(), wide.detach().numpy())
                if first_local is None and local_result["max_absolute_delta"] != 0:
                    first_local = op["name"]
                ranked.append((local_result["max_absolute_delta"], op["name"]))
            rows.append(row)
        fixtures.append({"name": fixture["name"], "input_kind": binding["input_kind"],
                         "original_archive_reproduction_or_input_propagation": reproduction,
                         "first_nonzero_local_operator_discrepancy": first_local,
                         "ranked_local_absolute_deltas": sorted(ranked, reverse=True),
                         "execution_digest": record["execution_digest"]})
    json_write(args.report, {"schema_version":1, "certificate_authority":False,
        "claim":"diagnostic_same_input_original_block_and_operator_localization",
        "absolute_tolerance":ATOL, "relative_tolerance":RTOL, "packages":versions,
        "torch_implementation_sha256":sha(source), "script_sha256":sha(Path(__file__)),
        "run_sha256":sha(args.observations / "run.json"), "bindings_sha256":sha(args.inputs / "bindings.json"),
        "kernel_qualification":"not_provided; f64 variants diagnose staging/reduction only",
        "fixtures":fixtures, "records":rows})
    print(json.dumps({"report":str(args.report), "fixtures":len(fixtures), "stages":len(rows)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    for name in ("reference", "ny-parity", "model", "inputs", "observations", "request"):
        preparation.add_argument("--" + name, type=Path, required=True)
    preparation.add_argument("--block", type=int, default=5)
    comparison = commands.add_parser("compare")
    for name in ("model", "inputs", "observations", "report"):
        comparison.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else compare)(args)


if __name__ == "__main__": main()
