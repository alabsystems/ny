# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""The gate's docs and pinned Trust frontend resolution stay fail-closed."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
GATE = REPO_ROOT / "scripts/gate.sh"


def hermetic_environment() -> dict[str, str]:
    """A copy of os.environ with the GitHub token variables removed.

    These tests inherit the developer's real environment. gate.sh now runs a
    github.com credential precondition (added 2026-08-28) that refuses with
    exit 2 when an env token is set and does not authenticate — which is
    exactly the state this box was in when these tests were written, so they
    were passing only because gate.sh did not yet look. Popping the variables
    makes the outcome a property of the test's own fixture rather than of
    whoever's shell launched pytest.

    gh reads GH_TOKEN in preference to GITHUB_TOKEN (measured: with both set,
    `gh auth status` names GH_TOKEN as the failing one), so both must go.
    """
    environment = os.environ.copy()
    for variable in ("GH_TOKEN", "GITHUB_TOKEN"):
        environment.pop(variable, None)
    return environment


def run_with_rejecting_rustup(tmp_path: Path, *arguments: str):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "rustup-was-invoked"
    rustup = fake_bin / "rustup"
    rustup.write_text(
        "#!/bin/sh\n"
        'printf invoked > "$NY_FAKE_RUSTUP_MARKER"\n'
        "exit 99\n",
        encoding="utf-8",
    )
    rustup.chmod(0o755)

    bash = shutil.which("bash")
    assert bash is not None
    environment = hermetic_environment()
    environment.update(
        {
            "NY_FAKE_RUSTUP_MARKER": str(marker),
            "NY_RUSTUP": str(rustup),
            # Keep only basic system utilities behind the deliberately broken
            # rustup. The test must not inherit a working Trust frontend.
            "PATH": f"{fake_bin}{os.pathsep}/usr/bin{os.pathsep}/bin",
        }
    )
    result = subprocess.run(
        [bash, str(GATE), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result, marker


def test_help_does_not_resolve_the_trust_frontend(tmp_path: Path) -> None:
    result, marker = run_with_rejecting_rustup(tmp_path, "--help")

    assert result.returncode == 0, result.stderr
    assert "--fast        run the bounded edit-loop lane (checks a-h" in result.stdout
    assert not marker.exists(), "--help invoked rustup before returning usage"


def test_executing_gate_still_requires_the_trust_frontend(tmp_path: Path) -> None:
    result, marker = run_with_rejecting_rustup(tmp_path, "--only", "a")

    assert result.returncode == 2
    assert marker.read_text(encoding="utf-8") == "invoked"
    assert "cannot resolve the pinned Trust frontend" in result.stderr


@pytest.mark.parametrize("executable_suffix", ("", ".exe"))
def test_executing_gate_uses_rustup_pinned_frontend_not_ambient_path(
    tmp_path: Path,
    executable_suffix: str,
) -> None:
    fake_bin = tmp_path / "bin"
    sysroot_bin = tmp_path / "trust-sysroot" / "bin"
    fake_bin.mkdir()
    sysroot_bin.mkdir(parents=True)
    rustup_marker = tmp_path / "rustup-was-invoked"
    rustc_marker = tmp_path / "pinned-rustc-was-invoked"
    targo_marker = tmp_path / "pinned-targo-was-invoked"
    cargo_marker = tmp_path / "pinned-cargo-was-invoked"

    rustup = fake_bin / "rustup"
    rustup.write_text(
        "#!/bin/sh\n"
        'printf invoked > "$NY_FAKE_RUSTUP_MARKER"\n'
        'printf "%s\\n" "$NY_FAKE_RUSTC"\n',
        encoding="utf-8",
    )
    rustup.chmod(0o755)
    rustc = sysroot_bin / f"rustc{executable_suffix}"
    rustc.write_text(
        "#!/bin/sh\n"
        'printf invoked > "$NY_FAKE_RUSTC_MARKER"\n'
        'printf "%s\\n" "$NY_FAKE_SYSROOT"\n',
        encoding="utf-8",
    )
    rustc.chmod(0o755)
    targo = sysroot_bin / f"targo{executable_suffix}"
    targo.write_text(
        "#!/bin/sh\n"
        'printf invoked > "$NY_FAKE_TARGO_MARKER"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    targo.chmod(0o755)
    cargo = sysroot_bin / f"cargo{executable_suffix}"
    cargo.write_text(
        "#!/bin/sh\n"
        'printf invoked > "$NY_FAKE_CARGO_MARKER"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    cargo.chmod(0o755)

    bash = shutil.which("bash")
    assert bash is not None
    environment = hermetic_environment()
    environment.update(
        {
            "NY_FAKE_RUSTUP_MARKER": str(rustup_marker),
            "NY_FAKE_RUSTC": str(rustc),
            "NY_FAKE_RUSTC_MARKER": str(rustc_marker),
            "NY_FAKE_TARGO_MARKER": str(targo_marker),
            "NY_FAKE_CARGO_MARKER": str(cargo_marker),
            "NY_FAKE_SYSROOT": str(sysroot_bin.parent),
            "NY_RUSTUP": str(rustup),
            # Ambient PATH deliberately has no compiler frontend. Resolution
            # must use the rustup-selected executable paths above.
            "PATH": f"{fake_bin}{os.pathsep}/usr/bin{os.pathsep}/bin",
        }
    )
    result = subprocess.run(
        [bash, str(GATE), "--only", "a"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert rustup_marker.read_text(encoding="utf-8") == "invoked"
    assert rustc_marker.read_text(encoding="utf-8") == "invoked"
    assert targo_marker.read_text(encoding="utf-8") == "invoked"
    assert cargo_marker.read_text(encoding="utf-8") == "invoked"
    assert "cannot resolve the pinned Trust frontend" not in result.stderr


# ---------------------------------------------------------------------------
# Lane reachability: the three places a check letter must appear stay in sync.
#
# A gate lane is reachable only if it is in ALL THREE of: the run_check
# dispatch, the --only whitelist, and the --help header. Lane [k] shipped with
# the whitelist updated to a-k while the header (line 99) and the arity error
# (line 127) still said a-j, so the newest lane was undocumented and
# unadvertised for its whole life. Nothing caught it: the only assertion on the
# header pinned the --fast line, which does not carry the letter range.
#
# These parse gate.sh rather than pinning a literal range, so they keep working
# when lane [l] lands — and fail if it lands in only two of the three places.

_DISPATCH = re.compile(r"^run_check ([a-z]) ", re.MULTILINE)
_WHITELIST = re.compile(r"^\s*((?:[a-z]\|)+[a-z])\)\s*;;\s*$", re.MULTILINE)
_HEADER_ENTRY = re.compile(r"^#   ([a-z])  \S", re.MULTILINE)


def gate_source() -> str:
    return GATE.read_text(encoding="utf-8")


def dispatched_letters() -> list[str]:
    letters = _DISPATCH.findall(gate_source())
    # A regex that matched nothing would make every assertion below vacuous —
    # the same "ok. 0 passed is VOID, not pass" trap the gate's own lanes guard
    # against. The gate has had at least 11 lanes since [k] landed.
    assert len(letters) >= 11, f"run_check dispatch parse found only {letters}"
    return letters


def test_only_whitelist_covers_exactly_the_dispatched_lanes() -> None:
    dispatched = dispatched_letters()
    candidates = _WHITELIST.findall(gate_source())
    # Exactly one, not "at least one": taking candidates[0] from an ambiguous
    # match would silently check some other `a|b)` case arm and pass while the
    # real whitelist drifted. Verified 2026-08-28 — this pattern matches once.
    assert len(candidates) == 1, (
        f"expected exactly one --only whitelist case arm, found {candidates}"
    )
    whitelist = sorted(set(candidates[0].split("|")))

    assert whitelist == sorted(set(dispatched)), (
        "the --only whitelist and the run_check dispatch disagree; a lane in "
        "the dispatch but not the whitelist is UNREACHABLE via --only, and a "
        "lane in the whitelist but not the dispatch silently runs nothing"
    )


def test_documented_only_range_matches_the_whitelist() -> None:
    source = gate_source()
    dispatched = dispatched_letters()
    expected = f"{min(dispatched)}-{max(dispatched)}"

    # Both user-facing statements of the range, asserted BY ANCHORED PATTERN so
    # a substring cannot produce a false green: "a-j" is a substring of nothing
    # useful, but an unanchored search for the range would also match prose.
    header = re.search(
        r"^#   --only <x>    run a single check by letter \(([a-z]-[a-z])\)$",
        source,
        re.MULTILINE,
    )
    assert header is not None, "the --help header no longer documents --only"
    assert header.group(1) == expected, (
        f"--help header advertises {header.group(1)} but the dispatch has {expected}"
    )

    arity = re.search(
        r'^\s*echo "gate\.sh: --only requires a check letter ([a-z]-[a-z])" >&2$',
        source,
        re.MULTILINE,
    )
    assert arity is not None, "the --only arity error no longer states a range"
    assert arity.group(1) == expected, (
        f"--only arity error states {arity.group(1)} but the dispatch has {expected}"
    )


def test_every_dispatched_lane_is_described_in_the_help_header() -> None:
    described = set(_HEADER_ENTRY.findall(gate_source()))
    missing = sorted(set(dispatched_letters()) - described)
    assert not missing, (
        f"lanes {missing} run but have no '#   <letter>  <name>' header entry, "
        "so `gate.sh --help` never mentions them"
    )


def test_documented_fast_lane_matches_FAST_CHECKS() -> None:
    """The --fast description must state the range the code actually runs.

    Found 2026-08-28 alongside the a-j/a-k drift: the header said --fast "skips
    heavy i-j" while FAST_CHECKS="a b c d e f g h" also skipped the newer lane
    [k]. Same defect class, same line block, and equally invisible — the only
    assertion on this line pinned the text up to "(checks a-h", which is the
    half that was still true.

    NOTE for whoever adds the next cheap lane: this asserts the DOCS match the
    CODE, not that the split is correct. check_k is a seconds-long lane and by
    the FAST_CHECKS rationale ("a cheap check belongs where it will be run")
    arguably belongs in --fast. Moving it is a behaviour change and is
    deliberately NOT made here; this test will simply require the header to be
    updated with it.
    """
    source = gate_source()
    dispatched = set(dispatched_letters())

    fast = re.search(r'^FAST_CHECKS="([a-z ]+)"$', source, re.MULTILINE)
    assert fast is not None, "FAST_CHECKS assignment not found in gate.sh"
    included = sorted(fast.group(1).split())
    skipped = sorted(dispatched - set(included))
    assert skipped, "FAST_CHECKS covers every lane; the --fast wording needs rethinking"

    # The range wording is only honest if both halves are contiguous.
    letters = sorted(dispatched)
    for group in (included, skipped):
        span = letters[letters.index(group[0]) : letters.index(group[-1]) + 1]
        assert span == group, (
            f"{group} is not a contiguous run of {letters}; the header's "
            "'x-y' range wording cannot describe it"
        )

    expected = (
        f"#   --fast        run the bounded edit-loop lane "
        f"(checks {included[0]}-{included[-1]}; skips {skipped[0]}-{skipped[-1]})"
    )
    assert expected in source, (
        "the --fast header line does not match FAST_CHECKS; expected:\n"
        f"  {expected}"
    )
