#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Materialize the two pinned setup-hosted VNN-COMP 2025 model payloads.

Dry-run is the default.  The command never downloads data and never overwrites
an existing path.  ``--apply`` installs only the two files declared by the
hard-pinned retained manifest and the one vggnet16 symlink declared by the
pinned official ``setup.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import regular_bank_evidence as evidence  # noqa: E402

SCHEMA = "ny_vnncomp2025_large_model_materialization_v1"
VGG_SOURCE_RELATIVE = PurePosixPath("vggnet16_2023/onnx/vgg16-7.onnx")
VGG_LINK_RELATIVE = PurePosixPath("vggnet16_2022/onnx/vgg16-7.onnx")
VGG_LINK_TARGET = "../../vggnet16_2023/onnx/vgg16-7.onnx"


@dataclass(frozen=True)
class Target:
    logical_path: str
    category: str
    declared_name: str
    physical_relative: PurePosixPath
    symlink_relative: PurePosixPath | None = None
    symlink_target: str | None = None


TARGETS = (
    Target(
        logical_path=(
            "benchmarks/cgan_2023/onnx/"
            "cGAN_imgSz32_nCh_3_small_transformer.onnx"
        ),
        category="cgan_2023",
        declared_name="onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx",
        physical_relative=PurePosixPath(
            "cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx"
        ),
    ),
    Target(
        logical_path="benchmarks/vggnet16_2022/onnx/vgg16-7.onnx",
        category="vggnet16_2022",
        declared_name="onnx/vgg16-7.onnx",
        physical_relative=VGG_SOURCE_RELATIVE,
        symlink_relative=VGG_LINK_RELATIVE,
        symlink_target=VGG_LINK_TARGET,
    ),
)


class MaterializationError(RuntimeError):
    """The pinned model worktree cannot be materialized safely."""


def _directory(path: Path, *, create: bool, root: Path) -> Path:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise MaterializationError(
            f"materialization directory escapes benchmark root: {path}"
        ) from error
    if path.is_symlink():
        raise MaterializationError(
            f"materialization directory must not be a symlink: {path}"
        )
    if path.exists():
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise MaterializationError(
                f"materialization directory is unavailable: {path}"
            ) from error
        if not resolved.is_dir() or resolved != path:
            raise MaterializationError(
                f"materialization directory is not canonical: {path}"
            )
        return resolved
    if not create:
        return path
    parent = _directory(path.parent, create=True, root=root)
    try:
        path.mkdir(mode=0o755)
    except FileExistsError:
        return _directory(path, create=False, root=root)
    except OSError as error:
        raise MaterializationError(
            f"could not create materialization directory: {path}"
        ) from error
    _fsync_directory(parent)
    return path


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise MaterializationError(
            f"could not synchronize directory: {path}"
        ) from error


def _file_state(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> str:
    if path.is_symlink():
        raise MaterializationError(
            f"refusing linked materialized payload path: {path}"
        )
    if not path.exists():
        return "missing"
    try:
        mode = path.lstat().st_mode
        links = path.lstat().st_nlink
        digest, _ = evidence.provenance._stable_file_hash(path)
        size = path.stat().st_size
    except (OSError, evidence.provenance.ProvenanceError) as error:
        raise MaterializationError(
            f"could not verify materialized payload: {path}"
        ) from error
    if (
        not stat.S_ISREG(mode)
        or links != 1
        or digest != expected_sha256
        or size != expected_size
    ):
        raise MaterializationError(
            f"existing materialized payload differs from pinned bytes: {path}"
        )
    return "ready" if stat.S_IMODE(mode) == 0o444 else "fix_mode"


def _install_file(path: Path, payload: bytes, root: Path) -> None:
    parent = _directory(path.parent, create=True, root=root)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".materializing",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise MaterializationError(
                f"payload path appeared during materialization: {path}"
            ) from error
        temporary.unlink()
        _fsync_directory(parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _symlink_state(path: Path, *, target: str, physical: Path) -> str:
    if path.is_symlink():
        if os.readlink(path) != target:
            raise MaterializationError(
                f"existing vgg symlink target differs from setup.sh: {path}"
            )
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise MaterializationError(
                f"existing vgg symlink is dangling: {path}"
            ) from error
        if resolved != physical:
            raise MaterializationError(
                f"existing vgg symlink resolves outside the pinned target: {path}"
            )
        return "ready"
    if path.exists():
        raise MaterializationError(
            f"refusing to replace non-symlink vgg path: {path}"
        )
    return "missing"


def _apply_symlink(path: Path, *, target: str, root: Path) -> None:
    parent = _directory(path.parent, create=False, root=root)
    if not parent.exists():
        raise MaterializationError(
            f"official vgg symlink parent is unavailable: {parent}"
        )
    try:
        path.symlink_to(target)
    except FileExistsError as error:
        raise MaterializationError(
            f"vgg symlink path appeared during materialization: {path}"
        ) from error
    except OSError as error:
        raise MaterializationError(
            f"could not create official vgg symlink: {path}"
        ) from error
    _fsync_directory(parent)


def materialize(benchmark_root: Path, *, apply: bool = False) -> dict[str, Any]:
    benchmark = evidence.validate_official_benchmark(benchmark_root)
    root = benchmark.benchmark_root
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        authoritative, payload = evidence.authoritative_benchmark_input(
            benchmark=benchmark,
            category=target.category,
            declared_name=target.declared_name,
            label="onnx",
        )
        source = authoritative.retained_setup_payload
        if (
            source is None
            or source.get("logical_path") != target.logical_path
            or authoritative.git_path is not None
            or authoritative.git_blob is not None
        ):
            raise MaterializationError(
                f"target is not bound to the retained setup payload: "
                f"{target.logical_path}"
            )
        physical = root.joinpath(*target.physical_relative.parts)
        state_before = _file_state(
            physical,
            expected_sha256=authoritative.sha256,
            expected_size=authoritative.size_bytes,
        )
        action = {
            "missing": "create",
            "fix_mode": "set_read_only",
            "ready": "none",
        }[state_before]
        if apply:
            if state_before == "missing":
                _install_file(physical, payload, root)
            elif state_before == "fix_mode":
                physical.chmod(0o444)
                _fsync_directory(physical.parent)
        final_state = _file_state(
            physical,
            expected_sha256=authoritative.sha256,
            expected_size=authoritative.size_bytes,
        )
        if apply and final_state != "ready":
            raise MaterializationError(
                f"materialized payload did not reach immutable state: {physical}"
            )

        symlink_action = "none"
        symlink_state = None
        if target.symlink_relative is not None:
            assert target.symlink_target is not None
            link = root.joinpath(*target.symlink_relative.parts)
            symlink_state = _symlink_state(
                link,
                target=target.symlink_target,
                physical=physical,
            )
            symlink_action = "create" if symlink_state == "missing" else "none"
            if apply and symlink_state == "missing":
                _apply_symlink(
                    link,
                    target=target.symlink_target,
                    root=root,
                )
            if apply:
                symlink_state = _symlink_state(
                    link,
                    target=target.symlink_target,
                    physical=physical,
                )

        rows.append(
            {
                "logical_path": target.logical_path,
                "physical_path": str(physical),
                "payload_action": action,
                "payload_state": final_state,
                "sha256": authoritative.sha256,
                "size_bytes": authoritative.size_bytes,
                "symlink_action": symlink_action,
                "symlink_state": symlink_state,
            }
        )
        del payload
    evidence.revalidate_official_benchmark(benchmark)
    return {
        "schema": SCHEMA,
        "applied": apply,
        "benchmark_root": str(root),
        "claim_scope": evidence.CLAIM_SCOPE,
        "retained_manifest_sha256": (
            evidence.PINNED_LARGE_MODEL_MANIFEST_SHA256
        ),
        "targets": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="install missing payloads/symlink; default is read-only dry-run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = materialize(args.benchmark_root, apply=args.apply)
    except (MaterializationError, evidence.EvidenceError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
