# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import gzip
import os
import shlex
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "benchmarks" / "download_benchmarks.sh"


def _install_fake_git_clone(tmp_path: Path, *, corrupt_gzip: bool = False) -> Path:
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()

    # Keep the subprocess hermetic and deliberately omit wget so the production
    # curl-backed compatibility path is exercised even on hosts that install it.
    for tool in (
        "bash",
        "basename",
        "cat",
        "chmod",
        "cp",
        "dirname",
        "find",
        "grep",
        "gunzip",
        "gzip",
        "ln",
        "mkdir",
        "mktemp",
        "mv",
        "rm",
        "rmdir",
        "touch",
        "tr",
        "unzip",
        "wc",
    ):
        tool_path = shutil.which(tool)
        assert tool_path is not None, f"test requires {tool}"
        (fakebin / tool).symlink_to(tool_path)

    archive = tmp_path / "fake-large-models.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        for archive_path in (
            "vnncomp2024/nn4sys_2023/seed_896832480/onnx/mscn_2048d.onnx",
            "vnncomp2024/nn4sys_2023/seed_896832480/onnx/mscn_2048d_dual.onnx",
            "vnncomp2024/vggnet16_2023/seed_896832480/onnx/vgg16-7.onnx",
            "vnncomp2024/cgan_2023/seed_896832480/onnx/"
            "cGAN_imgSz32_nCh_3_small_transformer.onnx",
        ):
            zip_file.writestr(archive_path, b"\x08\x01")
        zip_file.writestr(
            "vnncomp2024/example/seed_896832480/onnx/model.onnx.gz",
            gzip.compress(b"\x08\x01"),
        )
        zip_file.writestr(
            "vnncomp2024/example/seed_896832480/README.txt",
            b"fixture\n",
        )

    corrupt_fixture = ""
    if corrupt_gzip:
        corrupt_fixture = (
            'printf "not a gzip stream\\n" '
            '> "$target/benchmarks/example/onnx/corrupt.onnx.gz"'
        )

    git_shim = fakebin / "git"
    git_shim.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            if [ "$1" != "clone" ]; then
                echo "unexpected git args: $*" >&2
                exit 1
            fi
            target=""
            for arg in "$@"; do
                target="$arg"
            done
            mkdir -p \
                "$target/benchmarks/example/onnx" \
                "$target/benchmarks/nn4sys/onnx" \
                "$target/benchmarks/vggnet16_2022/onnx"
            __CORRUPT_FIXTURE__
            cat > "$target/setup.sh" <<'EOF'
            #!/bin/sh
            exit 0
            EOF
            chmod +x "$target/setup.sh"
            """
        ).replace("__CORRUPT_FIXTURE__", corrupt_fixture),
        encoding="utf-8",
    )
    git_shim.chmod(0o755)

    curl_shim = fakebin / "curl"
    curl_shim.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            output=""
            while [ "$#" -gt 0 ]; do
                case "$1" in
                    -o)
                        output="$2"
                        shift 2
                        ;;
                    *)
                        shift
                        ;;
                esac
            done
            cp __ARCHIVE__ "$output"
            """
        ).replace("__ARCHIVE__", shlex.quote(str(archive))),
        encoding="utf-8",
    )
    curl_shim.chmod(0o755)
    return fakebin


def _copy_download_script(tmp_path: Path) -> Path:
    script_copy = tmp_path / "download_benchmarks.sh"
    script_copy.write_text(SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    script_copy.chmod(0o755)
    return script_copy


def _write_existing_vnncomp2025_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "vnncomp2025"
    (checkout / "benchmarks" / "nn4sys" / "onnx").mkdir(parents=True)
    (checkout / "benchmarks" / "nn4sys_2023" / "onnx").mkdir(parents=True)
    (checkout / "benchmarks" / "vggnet16_2022" / "onnx").mkdir(parents=True)
    (checkout / "benchmarks" / "vggnet16_2023" / "onnx").mkdir(parents=True)
    (checkout / "benchmarks" / "cgan_2023" / "onnx").mkdir(parents=True)

    (checkout / "benchmarks" / "nn4sys" / "onnx" / "mscn_2048d.onnx").write_text(
        "placeholder\n",
        encoding="utf-8",
    )
    (checkout / "benchmarks" / "nn4sys" / "onnx" / "mscn_2048d_dual.onnx").write_text(
        "placeholder\n",
        encoding="utf-8",
    )
    (checkout / "benchmarks" / "vggnet16_2022" / "onnx" / "vgg16-7.onnx").write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    (checkout / "setup.sh").write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            set -e
            mkdir -p .setup benchmarks/nn4sys_2023/onnx benchmarks/vggnet16_2023/onnx benchmarks/cgan_2023/onnx
            echo "reran" > .setup/invoked
            wget https://example.invalid/model.onnx -O .setup/downloaded.bin
            cp .setup/downloaded.bin benchmarks/nn4sys_2023/onnx/mscn_2048d.onnx
            cp .setup/downloaded.bin benchmarks/nn4sys_2023/onnx/mscn_2048d_dual.onnx
            cp .setup/downloaded.bin benchmarks/vggnet16_2023/onnx/vgg16-7.onnx
            cp .setup/downloaded.bin benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx
            ln -sf ../../nn4sys_2023/onnx/mscn_2048d.onnx benchmarks/nn4sys/onnx/mscn_2048d.onnx
            ln -sf ../../nn4sys_2023/onnx/mscn_2048d_dual.onnx benchmarks/nn4sys/onnx/mscn_2048d_dual.onnx
            ln -sf ../../vggnet16_2023/onnx/vgg16-7.onnx benchmarks/vggnet16_2022/onnx/vgg16-7.onnx
            """
        ),
        encoding="utf-8",
    )
    (checkout / "setup.sh").chmod(0o755)
    return checkout


def _write_existing_vnncomp2025_checkout_missing_cgan_asset(tmp_path: Path) -> Path:
    checkout = tmp_path / "vnncomp2025"
    (checkout / "benchmarks" / "nn4sys" / "onnx").mkdir(parents=True)
    (checkout / "benchmarks" / "nn4sys_2023" / "onnx").mkdir(parents=True)
    (checkout / "benchmarks" / "vggnet16_2022" / "onnx").mkdir(parents=True)
    (checkout / "benchmarks" / "vggnet16_2023" / "onnx").mkdir(parents=True)
    (checkout / "benchmarks" / "cgan_2023" / "onnx").mkdir(parents=True)

    for path in (
        checkout / "benchmarks" / "nn4sys_2023" / "onnx" / "mscn_2048d.onnx",
        checkout / "benchmarks" / "nn4sys_2023" / "onnx" / "mscn_2048d_dual.onnx",
        checkout / "benchmarks" / "vggnet16_2023" / "onnx" / "vgg16-7.onnx",
    ):
        path.write_bytes(b"\x08\x01")

    os.symlink(
        "../../nn4sys_2023/onnx/mscn_2048d.onnx",
        checkout / "benchmarks" / "nn4sys" / "onnx" / "mscn_2048d.onnx",
    )
    os.symlink(
        "../../nn4sys_2023/onnx/mscn_2048d_dual.onnx",
        checkout / "benchmarks" / "nn4sys" / "onnx" / "mscn_2048d_dual.onnx",
    )
    os.symlink(
        "../../vggnet16_2023/onnx/vgg16-7.onnx",
        checkout / "benchmarks" / "vggnet16_2022" / "onnx" / "vgg16-7.onnx",
    )

    (checkout / "setup.sh").write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            set -e
            mkdir -p .setup benchmarks/cgan_2023/onnx
            echo "reran" > .setup/invoked
            wget https://example.invalid/model.onnx -O .setup/downloaded.bin
            cp .setup/downloaded.bin benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx
            """
        ),
        encoding="utf-8",
    )
    (checkout / "setup.sh").chmod(0o755)
    return checkout


def _write_existing_vnncomp2025_checkout_missing_cgan_asset_with_gunzip_rerun(
    tmp_path: Path,
) -> Path:
    checkout = _write_existing_vnncomp2025_checkout_missing_cgan_asset(tmp_path)
    existing_model = checkout / "benchmarks" / "example" / "onnx" / "model.onnx"
    existing_model.parent.mkdir(parents=True)
    existing_model.write_bytes(b"already present\n")
    return checkout


def test_download_benchmarks_runs_vnncomp2025_setup(tmp_path: Path) -> None:
    fakebin = _install_fake_git_clone(tmp_path)
    script_copy = _copy_download_script(tmp_path)

    env = os.environ.copy()
    env["PATH"] = str(fakebin)

    result = subprocess.run(
        ["bash", str(script_copy), "2025"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, (
        f"download script failed: {result.stderr}\n{result.stdout}"
    )
    checkout = tmp_path / "vnncomp2025"
    assert (
        checkout
        / "benchmarks"
        / "cgan_2023"
        / "onnx"
        / "cGAN_imgSz32_nCh_3_small_transformer.onnx"
    ).read_bytes() == b"\x08\x01", (
        "expected the fake large-model archive to be installed"
    )
    assert (
        checkout / "benchmarks" / "nn4sys" / "onnx" / "mscn_2048d.onnx"
    ).is_symlink(), "expected the 2025 setup repair symlink to be installed"
    assert (
        "Running upstream setup.sh for large models and broken symlinks"
        in result.stdout
    ), f"expected setup invocation log, got: {result.stdout}"
    assert (
        "wget not found; providing curl-backed compatibility shim for setup.sh"
        in result.stdout
    ), f"expected curl compatibility log, got: {result.stdout}"


def test_download_benchmarks_refreshes_existing_vnncomp2025_checkout(
    tmp_path: Path,
) -> None:
    fakebin = _install_fake_git_clone(tmp_path)
    script_copy = _copy_download_script(tmp_path)
    checkout = _write_existing_vnncomp2025_checkout(tmp_path)

    env = os.environ.copy()
    env["PATH"] = str(fakebin)

    result = subprocess.run(
        ["bash", str(script_copy), "2025"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, (
        f"download script failed: {result.stderr}\n{result.stdout}"
    )
    assert (
        checkout / "benchmarks" / "nn4sys" / "onnx" / "mscn_2048d.onnx"
    ).is_symlink(), (
        "expected stale nn4sys placeholder to be replaced by the upstream repair symlink"
    )
    assert (
        checkout / "benchmarks" / "nn4sys" / "onnx" / "mscn_2048d_dual.onnx"
    ).is_symlink(), (
        "expected stale nn4sys dual placeholder to be replaced by the upstream repair symlink"
    )
    assert (
        checkout / "benchmarks" / "vggnet16_2022" / "onnx" / "vgg16-7.onnx"
    ).is_symlink(), (
        "expected stale vggnet16 placeholder to be replaced by the upstream repair symlink"
    )
    assert (
        checkout
        / "benchmarks"
        / "cgan_2023"
        / "onnx"
        / "cGAN_imgSz32_nCh_3_small_transformer.onnx"
    ).read_bytes() == b"\x08\x01", (
        "expected setup refresh to restore the cgan small_transformer asset"
    )
    assert (
        "Existing checkout needs setup refresh for large models and broken symlinks"
        in result.stdout
    ), f"expected existing-checkout refresh log, got: {result.stdout}"


def test_download_benchmarks_refreshes_existing_vnncomp2025_checkout_when_cgan_asset_missing(
    tmp_path: Path,
) -> None:
    fakebin = _install_fake_git_clone(tmp_path)
    script_copy = _copy_download_script(tmp_path)
    checkout = _write_existing_vnncomp2025_checkout_missing_cgan_asset(tmp_path)

    env = os.environ.copy()
    env["PATH"] = str(fakebin)

    result = subprocess.run(
        ["bash", str(script_copy), "2025"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, (
        f"download script failed: {result.stderr}\n{result.stdout}"
    )
    assert (
        checkout
        / "benchmarks"
        / "cgan_2023"
        / "onnx"
        / "cGAN_imgSz32_nCh_3_small_transformer.onnx"
    ).read_bytes() == b"\x08\x01", (
        "expected missing cgan asset to be restored by the refresh path"
    )
    assert (
        "Existing checkout needs setup refresh for large models and broken symlinks"
        in result.stdout
    ), f"expected existing-checkout refresh log, got: {result.stdout}"


def test_download_benchmarks_refreshes_existing_vnncomp2025_checkout_when_setup_gunzip_is_idempotent(
    tmp_path: Path,
) -> None:
    fakebin = _install_fake_git_clone(tmp_path)
    script_copy = _copy_download_script(tmp_path)
    checkout = (
        _write_existing_vnncomp2025_checkout_missing_cgan_asset_with_gunzip_rerun(
            tmp_path
        )
    )

    env = os.environ.copy()
    env["PATH"] = str(fakebin)

    result = subprocess.run(
        ["bash", str(script_copy), "2025"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, (
        f"download script failed: {result.stderr}\n{result.stdout}"
    )
    assert (
        checkout
        / "benchmarks"
        / "cgan_2023"
        / "onnx"
        / "cGAN_imgSz32_nCh_3_small_transformer.onnx"
    ).read_bytes() == b"\x08\x01", (
        "expected cgan asset restoration to survive idempotent gunzip warnings"
    )
    assert (
        checkout / "benchmarks" / "example" / "onnx" / "model.onnx"
    ).read_bytes() == b"already present\n", (
        "expected idempotent gunzip to preserve the existing model"
    )


def test_download_benchmarks_rejects_corrupt_gzip(tmp_path: Path) -> None:
    fakebin = _install_fake_git_clone(tmp_path, corrupt_gzip=True)
    script_copy = _copy_download_script(tmp_path)

    env = os.environ.copy()
    env["PATH"] = str(fakebin)

    result = subprocess.run(
        ["bash", str(script_copy), "2024"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode != 0, (
        "corrupt benchmark archives must fail the download instead of being ignored"
    )
    assert "not in gzip format" in result.stderr, (
        f"expected an actionable gunzip diagnostic, got: {result.stderr}"
    )
