# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Offline regression tests for the VNN-COMP prebuilt installer path."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_TOOL = REPO_ROOT / "install_tool.sh"
RECEIPT_HELPER = REPO_ROOT / "vnncomp_scripts" / "submission_binary_receipt.sh"
AY_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(textwrap.dedent(contents), encoding="utf-8")
    path.chmod(0o755)


def _make_install_fixture(tmp_path: Path, *, prebuilt_exit: int = 0) -> Path:
    """Copy the installer and provide local-only command/build shims."""
    real_sha256sum = shutil.which("sha256sum")
    assert real_sha256sum is not None, "sha256sum is required by this fixture"
    installer = tmp_path / "install_tool.sh"
    shutil.copy(INSTALL_TOOL, installer)

    prebuilt = tmp_path / "dist" / "bin" / "ny-x86_64-linux.xz"
    prebuilt.parent.mkdir(parents=True)
    _write_executable(
        prebuilt,
        f"""\
        #!/bin/bash
        echo ny-fixture-1.0
        exit {prebuilt_exit}
        """,
    )
    prebuilt.with_suffix(prebuilt.suffix + ".sha256").write_text(
        "fixture  ny-x86_64-linux.xz\n", encoding="utf-8"
    )
    (prebuilt.parent / "ny-x86_64-linux.provenance.txt").write_text(
        "fixture-provenance\n", encoding="utf-8"
    )

    scripts = tmp_path / "vnncomp_scripts"
    scripts.mkdir(parents=True)
    shutil.copy(RECEIPT_HELPER, scripts / "submission_binary_receipt.sh")
    fallback = scripts / "build_submission_binary.sh"
    _write_executable(
        fallback,
        f"""\
        #!/bin/bash
        : > "{tmp_path / "source-fallback-used"}"
        """,
    )
    _write_executable(
        fallback.parent / "verify_prebuilt.py",
        """\
        #!/usr/bin/env python3
        import argparse
        import os
        import shutil

        parser = argparse.ArgumentParser()
        parser.add_argument("--repo-root", required=True)
        parser.add_argument("--archive", required=True)
        parser.add_argument("--checksum", required=True)
        parser.add_argument("--provenance", required=True)
        parser.add_argument("--output", required=True)
        arguments = parser.parse_args()
        with open(os.environ["FAKE_COMMAND_LOG"], "a", encoding="utf-8") as log:
            log.write("verify-prebuilt\\n")
        if int(os.environ.get("FAKE_VERIFY_EXIT", "0")):
            raise SystemExit(int(os.environ["FAKE_VERIFY_EXIT"]))
        shutil.copyfile(arguments.archive, arguments.output)
        """,
    )
    cargo_lock = tmp_path / "Cargo.lock"
    cargo_lock.write_text(
        "version = 4\n\n"
        "[[package]]\n"
        'name = "ay-milp"\n'
        'version = "0.1.0"\n'
        f'source = "git+https://github.com/alabsystems/ay.git?rev={AY_COMMIT}#{AY_COMMIT}"\n',
        encoding="utf-8",
    )
    lock_sha256 = hashlib.sha256(cargo_lock.read_bytes()).hexdigest()
    binary_sha256 = hashlib.sha256(prebuilt.read_bytes()).hexdigest()
    digest = "1" * 64
    (prebuilt.parent / "ny-x86_64-linux.provenance.txt").write_text(
        "\n".join(
            [
                "schema=ny-vnncomp-prebuilt-v1",
                "target=x86_64-unknown-linux-gnu",
                "features=mip,cuda",
                f"trust_commit={AY_COMMIT}",
                "trust_bootstrap_mode=seed",
                "trust_gate_status=passed",
                f"trust_gate_receipt_sha256={digest}",
                f"trust_gate_commands_sha256={digest}",
                f"trust_gate_log_sha256={digest}",
                f"trustc_sha256={digest}",
                f"trustc_version_sha256={digest}",
                f"ny_commit={AY_COMMIT}",
                f"cargo_lock_sha256={lock_sha256}",
                f"ay_lock_commit={AY_COMMIT}",
                f"builder_script_sha256={digest}",
                f"onnxruntime_static_sha256={digest}",
                f"binary_sha256={binary_sha256}",
                f"package_sha256={digest}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "uname",
        """\
        #!/bin/bash
        case "${1:-}" in
            -s) echo Linux ;;
            -m) echo x86_64 ;;
            *) exit 2 ;;
        esac
        """,
    )
    _write_executable(
        fake_bin / "sha256sum",
        f"""\
        #!/bin/bash
        echo sha256sum >> "${{FAKE_COMMAND_LOG}}"
        if [ "${{1:-}}" = "-c" ]; then
            exit "${{FAKE_SHA_EXIT:-0}}"
        fi
        exec {shlex.quote(real_sha256sum)} "$@"
        """,
    )
    _write_executable(
        fake_bin / "getconf",
        """\
        #!/bin/bash
        echo getconf >> "${FAKE_COMMAND_LOG}"
        if [ -n "${FAKE_GLIBC_VERSION:-}" ]; then
            echo "glibc ${FAKE_GLIBC_VERSION}"
        else
            exit 1
        fi
        """,
    )
    _write_executable(
        fake_bin / "ldd",
        """\
        #!/bin/bash
        echo ldd >> "${FAKE_COMMAND_LOG}"
        echo "musl libc fixture"
        """,
    )
    _write_executable(
        fake_bin / "xz",
        """\
        #!/bin/bash
        echo xz >> "${FAKE_COMMAND_LOG}"
        for argument in "$@"; do
            artifact="${argument}"
        done
        command cat "${artifact}"
        """,
    )
    _write_executable(
        fake_bin / "apt-get",
        """\
        #!/bin/bash
        echo apt-get >> "${FAKE_COMMAND_LOG}"
        exit "${FAKE_APT_EXIT:-0}"
        """,
    )
    _write_executable(
        fake_bin / "id",
        """\
        #!/bin/bash
        if [ "${1:-}" = "-u" ]; then
            echo 0
        else
            exit 2
        fi
        """,
    )
    _write_executable(fake_bin / "cargo", "#!/bin/bash\nexit 99\n")
    _write_executable(
        fake_bin / "git",
        '#!/bin/bash\necho git >> "${FAKE_COMMAND_LOG}"\nexit 1\n',
    )
    return installer


def _run_installer(
    tmp_path: Path,
    installer: Path,
    *,
    glibc_version: str | None,
    apt_exit: int = 0,
    verify_exit: int = 0,
) -> subprocess.CompletedProcess[str]:
    command_log = tmp_path / "commands.log"
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'fake-bin'}:/usr/bin:/bin"
    env["HOME"] = str(tmp_path / "home")
    env["FAKE_COMMAND_LOG"] = str(command_log)
    env["FAKE_APT_EXIT"] = str(apt_exit)
    env["FAKE_VERIFY_EXIT"] = str(verify_exit)
    if glibc_version is None:
        env.pop("FAKE_GLIBC_VERSION", None)
    else:
        env["FAKE_GLIBC_VERSION"] = glibc_version
    return subprocess.run(
        [str(installer), "v1"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _seed_existing_binary(tmp_path: Path) -> Path:
    target = tmp_path / "target" / "release" / "ny"
    target.parent.mkdir(parents=True)
    target.write_text("existing-binary\n", encoding="utf-8")
    return target


def _command_log(tmp_path: Path) -> list[str]:
    log = tmp_path / "commands.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def test_documented_glibc_floor_matches_installer() -> None:
    """The installer constants are the source of truth for published docs."""
    installer = INSTALL_TOOL.read_text(encoding="utf-8")
    major = re.search(r"^PREBUILT_MIN_GLIBC_MAJOR=(\d+)$", installer, re.MULTILINE)
    minor = re.search(r"^PREBUILT_MIN_GLIBC_MINOR=(\d+)$", installer, re.MULTILINE)
    assert major is not None
    assert minor is not None
    floor = f"{major.group(1)}.{minor.group(1)}"

    submission_readme = (REPO_ROOT / "vnncomp_scripts/README.md").read_text(
        encoding="utf-8"
    )
    trust_build_doc = (REPO_ROOT / "docs/VNNCOMP_2026_TRUST_LINUX_BUILD.md").read_text(
        encoding="utf-8"
    )
    assert f"{floor} or newer" in submission_readme
    assert f"requires ≥ {floor}" in trust_build_doc


def test_compatible_glibc_installs_only_after_provenance_and_sanity_run(
    tmp_path: Path,
) -> None:
    installer = _make_install_fixture(tmp_path)
    target = _seed_existing_binary(tmp_path)

    result = _run_installer(tmp_path, installer, glibc_version="2.39")

    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8").startswith("#!/bin/bash")
    receipt = target.with_suffix(".receipt")
    assert receipt.is_file()
    receipt_text = receipt.read_text(encoding="utf-8")
    assert "schema=ny-submission-binary-receipt-v1\n" in receipt_text
    assert "source_kind=prebuilt\n" in receipt_text
    assert f"source_commit={AY_COMMIT}\n" in receipt_text
    assert "toolchain_kind=trust-sealed\n" in receipt_text
    assert not (tmp_path / "source-fallback-used").exists()
    commands = _command_log(tmp_path)
    assert commands.index("verify-prebuilt") < commands.index("getconf")
    assert commands.index("getconf") < commands.index("sha256sum")
    assert "xz" not in commands
    assert not list(target.parent.glob(".ny-prebuilt.*"))
    assert not list(target.parent.glob(".ny-prebuilt-receipt.*"))


def test_prebuilt_receipt_accepts_manifest_key_reordering(tmp_path: Path) -> None:
    """Runtime parsing matches the packager's order-independent strict map."""
    installer = _make_install_fixture(tmp_path)
    provenance = tmp_path / "dist" / "bin" / "ny-x86_64-linux.provenance.txt"
    lines = provenance.read_text(encoding="utf-8").splitlines()
    provenance.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

    result = _run_installer(tmp_path, installer, glibc_version="2.39")

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "target" / "release" / "ny.receipt").is_file()


def test_old_glibc_preserves_existing_binary_and_uses_source_fallback(
    tmp_path: Path,
) -> None:
    installer = _make_install_fixture(tmp_path)
    target = _seed_existing_binary(tmp_path)

    result = _run_installer(tmp_path, installer, glibc_version="2.37")

    assert result.returncode == 0, result.stderr
    assert "requires GNU glibc >= 2.39; detected 2.37" in result.stderr
    assert target.read_text(encoding="utf-8") == "existing-binary\n"
    assert (tmp_path / "source-fallback-used").is_file()
    commands = _command_log(tmp_path)
    assert commands.index("verify-prebuilt") < commands.index("getconf")
    assert "xz" not in commands
    assert "git" not in commands, "installer must not persist Git credential rewrites"


def test_unknown_libc_fails_closed_for_prebuilt_but_keeps_source_fallback(
    tmp_path: Path,
) -> None:
    installer = _make_install_fixture(tmp_path)
    target = _seed_existing_binary(tmp_path)

    result = _run_installer(tmp_path, installer, glibc_version=None)

    assert result.returncode == 0, result.stderr
    assert "unable to confirm GNU glibc >= 2.39" in result.stderr
    assert target.read_text(encoding="utf-8") == "existing-binary\n"
    assert (tmp_path / "source-fallback-used").is_file()
    assert _command_log(tmp_path)[:3] == ["verify-prebuilt", "getconf", "ldd"]


def test_missing_checksum_stops_before_compatibility_or_fallback(
    tmp_path: Path,
) -> None:
    installer = _make_install_fixture(tmp_path)
    target = _seed_existing_binary(tmp_path)
    (tmp_path / "dist" / "bin" / "ny-x86_64-linux.xz.sha256").unlink()

    result = _run_installer(tmp_path, installer, glibc_version="2.39")

    assert result.returncode == 1
    assert "refusing unchecked prebuilt binary" in result.stderr
    assert target.read_text(encoding="utf-8") == "existing-binary\n"
    assert not (tmp_path / "source-fallback-used").exists()
    assert _command_log(tmp_path) == []


def test_missing_provenance_stops_before_verifier_or_fallback(tmp_path: Path) -> None:
    installer = _make_install_fixture(tmp_path)
    target = _seed_existing_binary(tmp_path)
    (tmp_path / "dist" / "bin" / "ny-x86_64-linux.provenance.txt").unlink()

    result = _run_installer(tmp_path, installer, glibc_version="2.39")

    assert result.returncode == 1
    assert "refusing unproven prebuilt binary" in result.stderr
    assert target.read_text(encoding="utf-8") == "existing-binary\n"
    assert not (tmp_path / "source-fallback-used").exists()
    assert _command_log(tmp_path) == []


def test_failed_provenance_verification_stops_without_source_fallback(
    tmp_path: Path,
) -> None:
    installer = _make_install_fixture(tmp_path)
    target = _seed_existing_binary(tmp_path)

    result = _run_installer(tmp_path, installer, glibc_version="2.39", verify_exit=23)

    assert result.returncode == 1
    assert "failed provenance validation" in result.stderr
    assert target.read_text(encoding="utf-8") == "existing-binary\n"
    assert not (tmp_path / "source-fallback-used").exists()
    assert _command_log(tmp_path) == ["verify-prebuilt"]
    assert not list(target.parent.glob(".ny-prebuilt.*"))


def test_failed_sanity_run_does_not_replace_existing_binary(tmp_path: Path) -> None:
    installer = _make_install_fixture(tmp_path, prebuilt_exit=7)
    target = _seed_existing_binary(tmp_path)

    result = _run_installer(tmp_path, installer, glibc_version="2.39")

    assert result.returncode == 0, result.stderr
    assert "failed its sanity run" in result.stderr
    assert target.read_text(encoding="utf-8") == "existing-binary\n"
    assert (tmp_path / "source-fallback-used").is_file()
    assert not list(target.parent.glob(".ny-prebuilt.*"))
    assert not list(target.parent.glob(".ny-prebuilt-receipt.*"))


def test_symlinked_prebuilt_provenance_preserves_existing_binary(tmp_path: Path) -> None:
    installer = _make_install_fixture(tmp_path)
    target = _seed_existing_binary(tmp_path)
    provenance = tmp_path / "dist" / "bin" / "ny-x86_64-linux.provenance.txt"
    external = tmp_path / "external-provenance.txt"
    external.write_bytes(provenance.read_bytes())
    provenance.unlink()
    provenance.symlink_to(external)

    result = _run_installer(tmp_path, installer, glibc_version="2.39")

    assert result.returncode == 1
    assert "prebuilt provenance must be a regular non-symlink file" in result.stderr
    assert target.read_text(encoding="utf-8") == "existing-binary\n"
    assert not (tmp_path / "source-fallback-used").exists()


def test_mismatched_prebuilt_manifest_preserves_existing_binary(tmp_path: Path) -> None:
    installer = _make_install_fixture(tmp_path)
    target = _seed_existing_binary(tmp_path)
    provenance = tmp_path / "dist" / "bin" / "ny-x86_64-linux.provenance.txt"
    provenance.write_text(
        provenance.read_text(encoding="utf-8").replace(
            "binary_sha256=" + hashlib.sha256(
                (tmp_path / "dist" / "bin" / "ny-x86_64-linux.xz").read_bytes()
            ).hexdigest(),
            "binary_sha256=" + "0" * 64,
        ),
        encoding="utf-8",
    )

    result = _run_installer(tmp_path, installer, glibc_version="2.39")

    assert result.returncode == 1
    assert "installed prebuilt bytes do not match binary_sha256" in result.stderr
    assert target.read_text(encoding="utf-8") == "existing-binary\n"
    assert not (tmp_path / "source-fallback-used").exists()


def test_source_fallback_stops_when_package_bootstrap_fails(tmp_path: Path) -> None:
    installer = _make_install_fixture(tmp_path)

    result = _run_installer(
        tmp_path,
        installer,
        glibc_version="2.37",
        apt_exit=9,
    )

    assert result.returncode == 9
    assert not (tmp_path / "source-fallback-used").exists()


def test_source_fallback_installs_python_for_build_and_verification() -> None:
    installer = INSTALL_TOOL.read_text(encoding="utf-8")
    assert re.search(r"apt-get install -y[^\n]*\bpython3\b", installer)
