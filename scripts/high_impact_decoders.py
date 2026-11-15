#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Fetch and hash immutable pretrained decoder fixtures, without claiming parity.

Weights live in an external cache; only identities and hashes belong in Git.
The original publisher's LFS SHA256 is checked before a new file is installed.
Use --weights LLM-B or --weights LLM-C to stage later scale experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

MODELS = (
    ("LLM-A", "EleutherAI/pythia-14m", "cf967c0a9a04383db6f7b1108d86b2962634b4ac"),
    ("LLM-B", "Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
    ("LLM-C", "Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796"),
)
METADATA = {
    "config.json", "generation_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "tokenizer.json", "vocab.json", "merges.txt",
    "model.safetensors.index.json", "LICENSE", "LICENSE.txt",
}
CONFIG_KEYS = (
    "model_type", "architectures", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "vocab_size",
    "layer_norm_eps", "rms_norm_eps", "rotary_pct", "rope_theta",
    "hidden_act", "tie_word_embeddings", "torch_dtype",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_original(path: Path, expected: str | None, git_blob: str | None) -> str:
    digest = sha256(path)
    if expected is not None:
        if digest != expected:
            raise ValueError(f"original file has wrong SHA256: {path}")
    elif git_blob is not None:
        # Ordinary repository files have a Git blob identity at the pinned
        # revision. Checking only a locally observed SHA would accept a stale
        # or modified cache as if it were publisher metadata.
        blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                blob.update(chunk)
        if blob.hexdigest() != git_blob:
            raise ValueError(f"original metadata has wrong Git blob: {path}")
    else:
        raise ValueError(f"original file lacks publisher identity: {path.name}")
    return digest


def fetch(url: str, destination: Path, expected: str | None, git_blob: str | None) -> str:
    if destination.exists():
        return verify_original(destination, expected, git_blob)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".decoder-", delete=False) as out:
            temporary = Path(out.name)
            with urllib.request.urlopen(url, timeout=120) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    out.write(chunk)
        digest = verify_original(temporary, expected, git_blob)
        temporary.replace(destination)
        temporary = None
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def pin(cache: Path, weight_slots: set[str]) -> dict:
    models = []
    for slot, repository, revision in MODELS:
        url = f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true"
        with urllib.request.urlopen(url, timeout=30) as response:
            metadata = json.load(response)
        if metadata["sha"] != revision:
            raise ValueError(f"publisher returned a different revision for {repository}")
        license_name = metadata.get("cardData", {}).get("license")
        if license_name != "apache-2.0":
            raise ValueError(f"unexpected fixture license: {repository}: {license_name}")
        directory = cache / repository.replace("/", "--") / revision
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for blob in metadata["siblings"]:
            name = blob["rfilename"]
            is_weight = name.endswith(".safetensors")
            if name not in METADATA and not is_weight:
                continue
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"unsafe publisher filename: {name}")
            expected = blob.get("lfs", {}).get("sha256")
            git_blob = blob.get("blobId")
            if is_weight and expected is None:
                raise ValueError(f"original weight lacks publisher SHA256: {name}")
            info = {"path": name, "size_bytes": blob.get("size"), "remote_lfs_sha256": expected,
                    "remote_git_blob": git_blob}
            if name in METADATA or slot in weight_slots:
                original = f"https://huggingface.co/{repository}/resolve/{revision}/{name}"
                info["observed_sha256"] = fetch(original, directory / name, expected, git_blob)
            files.append(info)
        configuration = json.loads((directory / "config.json").read_text())
        models.append({
            "slot": slot, "repository": repository, "revision": revision,
            "source": f"https://huggingface.co/{repository}", "license": license_name,
            "configuration": {key: configuration.get(key) for key in CONFIG_KEYS},
            "files": sorted(files, key=lambda item: item["path"]),
            "parity_status": "not_run", "runtime_certificate_status": "not_qualified",
        })
        print(f"{slot}: {repository}@{revision}: original metadata/hashes pinned", flush=True)
    return {
        "schema_version": 1,
        "copyright": "Copyright 2026 Andrew Yates; SPDX-License-Identifier: Apache-2.0",
        "selection_date": "2026-10-01",
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", nargs="*", choices=[item[0] for item in MODELS], default=["LLM-A"])
    args = parser.parse_args()
    manifest = pin(args.cache, set(args.weights))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
