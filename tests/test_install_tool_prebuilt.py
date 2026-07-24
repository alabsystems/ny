# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Offline regression tests for the VNN-COMP prebuilt installer path."""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

INSTALL_TOOL = Path(__file__).resolve().parent.parent / "install_tool.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(textwrap.dedent(contents), encoding="utf-8")
    path.chmod(0o755)


def _make_install_fixture(tmp_path: Path, *, prebuilt_exit: int = 0) -> Path:
    """Copy the installer and provide local-only command/build shims."""
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

    fallback = tmp_path / "vnncomp_scripts" / "build_submission_binary.sh"
    fallback.parent.mkdir(parents=True)
    _write_executable(
        fallback,
        f"""\
        #!/bin/bash
        : > "{tmp_path / "source-fallback-used"}"
        """,
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
        """\
        #!/bin/bash
        echo sha256sum >> "${FAKE_COMMAND_LOG}"
        exit "${FAKE_SHA_EXIT:-0}"
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
        "#!/bin/bash\necho git >> \"${FAKE_COMMAND_LOG}\"\nexit 1\n",
    )
    return installer


def _run_installer(
    tmp_path: Path, installer: Path, *, glibc_version: str | None
) -> subprocess.CompletedProcess[str]:
    command_log = tmp_path / "commands.log"
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'fake-bin'}:/usr/bin:/bin"
    env["HOME"] = str(tmp_path / "home")
    env["FAKE_COMMAND_LOG"] = str(command_log)
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


def test_compatible_glibc_installs_only_after_checksum_and_sanity_run(
    tmp_path: Path,
) -> None:
    installer = _make_install_fixture(tmp_path)
    target = _seed_existing_binary(tmp_path)

    result = _run_installer(tmp_path, installer, glibc_version="2.39")

    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8").startswith("#!/bin/bash")
    assert not (tmp_path / "source-fallback-used").exists()
    commands = _command_log(tmp_path)
    assert (
        commands.index("sha256sum") < commands.index("getconf") < commands.index("xz")
    )
    assert not list(target.parent.glob(".ny-prebuilt.*"))


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
    assert commands.index("sha256sum") < commands.index("getconf")
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
    assert _command_log(tmp_path)[:3] == ["sha256sum", "getconf", "ldd"]


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


def test_failed_sanity_run_does_not_replace_existing_binary(tmp_path: Path) -> None:
    installer = _make_install_fixture(tmp_path, prebuilt_exit=7)
    target = _seed_existing_binary(tmp_path)

    result = _run_installer(tmp_path, installer, glibc_version="2.39")

    assert result.returncode == 0, result.stderr
    assert "failed its sanity run" in result.stderr
    assert target.read_text(encoding="utf-8") == "existing-binary\n"
    assert (tmp_path / "source-fallback-used").is_file()
    assert not list(target.parent.glob(".ny-prebuilt.*"))
