# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "materialize_vnncomp2025_large_models.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "materialize_vnncomp2025_large_models",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


materializer = _load_module()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, bytes]]:
    repository = tmp_path / "official-benchmark"
    root = repository / "benchmarks"
    (root / "cgan_2023/onnx").mkdir(parents=True)
    (root / "vggnet16_2022/onnx").mkdir(parents=True)
    benchmark = materializer.evidence.PinnedOfficialBenchmark(
        benchmark_root=root,
        repository_root=repository,
        identity={
            "commit": materializer.evidence.OFFICIAL_BENCHMARK_COMMIT,
            "origin": materializer.evidence.OFFICIAL_BENCHMARK_ORIGIN,
        },
    )
    payloads = {
        materializer.TARGETS[0].logical_path: b"cgan retained payload\n",
        materializer.TARGETS[1].logical_path: b"vgg retained payload\n",
    }

    monkeypatch.setattr(
        materializer.evidence,
        "validate_official_benchmark",
        lambda requested: benchmark
        if requested.resolve() == root.resolve()
        else pytest.fail("unexpected benchmark root"),
    )
    monkeypatch.setattr(
        materializer.evidence,
        "revalidate_official_benchmark",
        lambda observed: None
        if observed == benchmark
        else pytest.fail("unexpected benchmark snapshot"),
    )

    def authoritative(
        *,
        benchmark: materializer.evidence.PinnedOfficialBenchmark,
        category: str,
        declared_name: str,
        label: str,
        **_kwargs: object,
    ) -> tuple[materializer.evidence.AuthoritativeInput, bytes]:
        assert label == "onnx"
        target = next(
            item
            for item in materializer.TARGETS
            if item.category == category and item.declared_name == declared_name
        )
        payload = payloads[target.logical_path]
        source = {
            "kind": "official_setup_retained_payload_v1",
            "logical_path": target.logical_path,
        }
        return (
            materializer.evidence.AuthoritativeInput(
                declared_name=declared_name,
                git_path=None,
                git_blob=None,
                compression="gzip",
                compressed_sha256="1" * 64,
                compressed_size_bytes=1,
                sha256=_sha(payload),
                size_bytes=len(payload),
                retained_setup_payload=source,
            ),
            payload,
        )

    monkeypatch.setattr(
        materializer.evidence,
        "authoritative_benchmark_input",
        authoritative,
    )
    return root, payloads


def test_materializer_is_read_only_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _fixture(tmp_path, monkeypatch)

    result = materializer.materialize(root)

    assert result["applied"] is False
    assert [row["payload_action"] for row in result["targets"]] == [
        "create",
        "create",
    ]
    assert not (
        root
        / "cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx"
    ).exists()
    assert not (root / materializer.VGG_SOURCE_RELATIVE).exists()
    assert not (root / materializer.VGG_LINK_RELATIVE).is_symlink()


def test_materializer_applies_atomically_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, payloads = _fixture(tmp_path, monkeypatch)

    applied = materializer.materialize(root, apply=True)

    assert applied["applied"] is True
    for target in materializer.TARGETS:
        physical = root.joinpath(*target.physical_relative.parts)
        assert physical.read_bytes() == payloads[target.logical_path]
        assert physical.stat().st_mode & 0o777 == 0o444
        assert physical.stat().st_nlink == 1
    link = root / materializer.VGG_LINK_RELATIVE
    assert link.is_symlink()
    assert link.readlink().as_posix() == materializer.VGG_LINK_TARGET
    assert link.resolve(strict=True) == (
        root / materializer.VGG_SOURCE_RELATIVE
    )

    repeated = materializer.materialize(root, apply=True)

    assert [row["payload_action"] for row in repeated["targets"]] == [
        "none",
        "none",
    ]
    assert repeated["targets"][1]["symlink_action"] == "none"


def test_materializer_never_overwrites_different_existing_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _fixture(tmp_path, monkeypatch)
    target = (
        root
        / "cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx"
    )
    target.write_bytes(b"different")

    with pytest.raises(
        materializer.MaterializationError,
        match="differs from pinned bytes",
    ):
        materializer.materialize(root, apply=True)

    assert target.read_bytes() == b"different"
    assert not (root / materializer.VGG_SOURCE_RELATIVE).exists()


def test_materializer_never_replaces_wrong_vgg_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _fixture(tmp_path, monkeypatch)
    materializer.materialize(root, apply=True)
    link = root / materializer.VGG_LINK_RELATIVE
    link.unlink()
    link.symlink_to("../../wrong/model.onnx")

    with pytest.raises(
        materializer.MaterializationError,
        match="symlink target differs",
    ):
        materializer.materialize(root, apply=True)

    assert link.readlink().as_posix() == "../../wrong/model.onnx"
