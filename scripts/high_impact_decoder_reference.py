#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Archive pinned official CPU decoder intermediates for lowering parity.

This is a diagnostic reference collector, never certificate authority. Run it
through ny_box.sh build for correctness fixtures or measure for fair clocks.
Every fixture includes the full vocabulary and original head coefficients.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

from high_impact_decoders import MODELS, sha256

PINNED = {"torch": "2.8.0", "transformers": "4.57.1", "tokenizers": "0.22.1",
          "safetensors": "0.6.2", "numpy": "2.0.2"}
PROMPTS = ("The capital of France is", "A proof must cover every token.", "Once upon a time")


def collect(cache: Path, output: Path, slot: str, manifest: Path) -> None:
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    if versions != PINNED:
        raise ValueError(f"reference environment differs: {versions}")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, repository, revision = next(item for item in MODELS if item[0] == slot)
    original = cache / repository.replace("/", "--") / revision
    registry = json.loads(manifest.read_text())
    identity = next(m for m in registry["models"] if m["slot"] == slot)
    if identity["repository"] != repository or identity["revision"] != revision:
        raise ValueError("original model registry differs")
    for record in identity["files"]:
        path = original / record["path"]
        expected = record["remote_lfs_sha256"] or record.get("observed_sha256")
        if expected is None or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"original fixture file absent or altered: {path.name}")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Serialized weights are loaded without remote code or a mutable revision.
    tokenizer = AutoTokenizer.from_pretrained(original, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        original, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).cpu().eval()
    files = {path.name: sha256(path) for path in original.iterdir()
             if path.is_file() and path.suffix in {".json", ".safetensors"}}
    source = Path(inspect.getfile(type(model)))
    profile = {
        "claim": "diagnostic_pinned_official_cpu_float32_eager",
        "certificate_authority": False,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions, "threads": 1,
        "implementation_file": source.name, "implementation_sha256": sha256(source),
        "collector_sha256": sha256(Path(__file__)),
        "model_registry_sha256": sha256(manifest),
        "repository": repository, "revision": revision, "original_files": files,
    }
    head = model.get_output_embeddings()
    weights = head.weight.detach().cpu().double().numpy().astype("<f8", copy=False)
    bias = np.zeros(weights.shape[0], dtype="<f8") if head.bias is None else head.bias.detach().cpu().double().numpy().astype("<f8", copy=False)
    weights_path = output / "head_weights.f64le"
    bias_path = output / "head_bias.f64le"
    weights.tofile(weights_path)
    bias.tofile(bias_path)
    profile["head"] = {"vocab_size": weights.shape[0], "hidden_size": weights.shape[1],
                       "weights_sha256": sha256(weights_path), "bias_sha256": sha256(bias_path)}
    fixtures = []
    # HF hidden_states replaces the final raw block with its post-norm value.
    # Hooks retain every original block output for actual lowering parity.
    blocks = model.gpt_neox.layers if hasattr(model, "gpt_neox") else model.model.layers
    captured = {}
    def capture(index):
        def hook(_module, _inputs, outputs):
            value = outputs[0] if isinstance(outputs, tuple) else outputs
            captured[index] = value.detach().cpu().numpy().copy()
        return hook
    handles = [block.register_forward_hook(capture(index))
               for index, block in enumerate(blocks)]
    for number, prompt in enumerate(PROMPTS):
        captured.clear()
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            result = model(**encoded, output_hidden_states=True, use_cache=False)
        states = [state.detach().cpu().numpy() for state in result.hidden_states]
        if sorted(captured) != list(range(len(blocks))):
            raise ValueError("original runtime did not expose every block")
        logits = result.logits.detach().cpu().numpy()
        if logits.shape[-1] != weights.shape[0] or not np.isfinite(logits).all():
            raise ValueError("original runtime output shape or values")
        tensors_path = output / f"fixture_{number}.npz"
        np.savez(tensors_path, input_ids=encoded["input_ids"].numpy(), logits=logits,
                 **{f"hidden_{index}": state for index, state in enumerate(states)},
                 **{f"block_{index}": state for index, state in captured.items()})
        anchor = states[-1][0, -1].astype(np.float64)
        target = int(np.argmax(logits[0, -1]))
        # These standalone boxes are properties of a supplied head domain.
        # They are NOT claims that the decoder's perturbed trajectory lies here.
        requests = []
        for epsilon in (0.0, 1e-8, 1e-6):
            request = {
                "vocab_size": int(weights.shape[0]), "hidden_size": int(weights.shape[1]),
                "weights_f64le": str(weights_path.resolve()),
                "bias_f64le": str(bias_path.resolve()), "arithmetic": "ExactReal",
                "target": target, "anchor": anchor.tolist(),
                "lo": np.nextafter(anchor - epsilon, -np.inf).tolist(),
                "hi": np.nextafter(anchor + epsilon, np.inf).tolist(),
            }
            request_path = output / f"fixture_{number}_head_eps_{epsilon:g}.json"
            request_path.write_text(json.dumps(request, separators=(",", ":")) + "\n")
            requests.append({"epsilon": epsilon, "path": request_path.name,
                             "sha256": sha256(request_path), "upstream_enclosure": "not_provided"})
        fixtures.append({"prompt": prompt, "input_ids": encoded["input_ids"][0].tolist(),
                         "target": target, "hidden_count": len(states),
                         "raw_block_count": len(captured),
                         "tensors": tensors_path.name, "sha256": sha256(tensors_path),
                         "full_vocab": int(logits.shape[-1]), "head_requests": requests})
    profile["fixtures"] = fixtures
    for handle in handles:
        handle.remove()
    (output / "reference_manifest.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps({"slot": slot, "fixtures": len(fixtures), "full_vocab": int(weights.shape[0]),
                      "status": "reference_archived", "certificate_authority": False}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slot", choices=[m[0] for m in MODELS], default="LLM-A")
    parser.add_argument("--model-manifest", type=Path,
                        default=Path(__file__).resolve().parent.parent / "benchmarks/llm/manifests/pretrained_decoders_v1.json")
    args = parser.parse_args()
    collect(args.cache, args.output, args.slot, args.model_manifest)


if __name__ == "__main__":
    main()
