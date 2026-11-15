# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""The gate's docs and pinned Trust frontend resolution stay fail-closed."""

from __future__ import annotations

import os
import re
import shlex
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
@pytest.mark.parametrize("gate_letter", ("a", "k"))
def test_executing_gate_uses_rustup_pinned_frontend_not_ambient_path(
    tmp_path: Path,
    executable_suffix: str,
    gate_letter: str,
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
        # Every build/test lane must name the unverified lane explicitly.
        'case "$1" in fmt|tippy|metadata) ;; --unverified) ;; '
        '*) printf "fake targo: no lane: %s\\n" "$*" >&2; exit 97 ;; esac\n'
        "printf '     Running tests/lipschitz_sound.rs (fake)\\n'\n"
        "printf '     Running tests/facade_expansion_contract.rs (fake)\\n'\n"
        "printf 'test result: ok. 65 passed; 0 failed; 0 ignored\\n'\n"
        "exit 0\n",
        encoding="utf-8",
    )
    targo.chmod(0o755)
    # The sysroot also ships an upstream-compatible cargo; no lane may use it.
    cargo = sysroot_bin / f"cargo{executable_suffix}"
    cargo.write_text(
        "#!/bin/sh\n"
        'printf invoked > "$NY_FAKE_CARGO_MARKER"\n'
        "exit 98\n",
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
        [bash, str(GATE), "--only", gate_letter],
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
    assert not cargo_marker.exists(), "a gate lane ran the sysroot's upstream cargo"
    assert "cannot resolve the pinned Trust frontend" not in result.stderr


@pytest.mark.parametrize(
    ("identity", "executable_suffix", "has_targo", "expected_returncode"),
    (
        ("trust: 0.12.0+gnative", "", True, 0),
        ("trust: 0.12.0+gnative", ".exe", True, 0),
        ("release: 1.101.0", "", True, 2),
        ("trust: 0.12.0+gnative", "", False, 2),
    ),
)
def test_native_trust_requires_identity_and_its_own_sysroot_frontend(
    tmp_path: Path,
    identity: str,
    executable_suffix: str,
    has_targo: bool,
    expected_returncode: int,
) -> None:
    fake_bin = tmp_path / "bin"
    sysroot_bin = tmp_path / "native Trust sysroot" / "bin"
    fake_bin.mkdir()
    sysroot_bin.mkdir(parents=True)
    ambient_marker = tmp_path / "ambient-frontend-was-invoked"
    native_marker = tmp_path / "native-targo-was-invoked"
    trustc = fake_bin / "trustc"
    trustc.write_text(
        '#!/bin/sh\ncase "$1" in\n'
        f'  -vV) printf "%s\\n" "{identity}" ;;\n'
        '  --print) printf "%s\\n" "$NY_FAKE_SYSROOT" ;;\n'
        '  *) exit 99 ;;\nesac\n',
        encoding="utf-8",
    )
    trustc.chmod(0o755)
    # Even a discoverable rustup or ambient frontend cannot rescue a broken
    # native installation by silently changing the selected compiler.
    for name in ("rustup", "targo", "cargo"):
        executable = fake_bin / name
        executable.write_text(
            '#!/bin/sh\nprintf invoked > "$NY_FAKE_AMBIENT_MARKER"\nexit 98\n',
            encoding="utf-8",
        )
        executable.chmod(0o755)
    if has_targo:
        targo = sysroot_bin / f"targo{executable_suffix}"
        targo.write_text(
            '#!/bin/sh\nprintf invoked > "$NY_FAKE_NATIVE_MARKER"\n'
            'case "$1" in --unverified) ;; *) exit 97 ;; esac\n'
            "printf '     Running tests/lipschitz_sound.rs (fake)\\n'\n"
            "printf '     Running tests/facade_expansion_contract.rs (fake)\\n'\n"
            "printf 'test result: ok. 65 passed; 0 failed; 0 ignored\\n'\n",
            encoding="utf-8",
        )
        targo.chmod(0o755)

    bash = shutil.which("bash")
    assert bash is not None
    environment = hermetic_environment()
    environment.pop("NY_RUSTUP", None)
    environment.update(
        {
            "NY_FAKE_SYSROOT": str(sysroot_bin.parent),
            "NY_FAKE_AMBIENT_MARKER": str(ambient_marker),
            "NY_FAKE_NATIVE_MARKER": str(native_marker),
            "PATH": f"{fake_bin}{os.pathsep}/usr/bin{os.pathsep}/bin",
        }
    )
    result = subprocess.run(
        [bash, str(GATE), "--only", "k"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == expected_returncode, result.stderr
    assert native_marker.exists() == (expected_returncode == 0)
    assert not ambient_marker.exists(), "native resolution changed frontend"
    if expected_returncode != 0:
        assert "cannot resolve the pinned Trust frontend" in result.stderr


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


# ---------------------------------------------------------------------------
# Lane [o]: the ny-cert verified lane (advisory ramp) fails closed.
#
# The lane trusts neither targo's exit code nor its silence: it runs
# `targo trust check` once for ny-cert's library and once per binary, parses
# each run's `Gate:` line and requires `0 failed` there. These drive the real
# gate.sh against a fake pinned sysroot whose targo replays a chosen Gate line
# per run (keyed by `--lib` / `--bin <name>` in its argv), so each verdict
# below is a property of the lane's parser.

_REAL_GATE_LINE = (
    "  Gate: CONDITIONAL-PASS [advisory lane, exit 0] (309 proved, 0 failed, "
    "0 runtime-checked, 6 assumed, 10 mandated, 0 contract-panic, "
    "561 unknown / 886 total)"
)


def _bin_line(proved: int, mandated: int, unknown: int, total: int) -> str:
    return (
        f"  Gate: CONDITIONAL-PASS [advisory lane, exit 0] ({proved} proved, "
        f"0 failed, 0 runtime-checked, 0 assumed, {mandated} mandated, "
        f"0 contract-panic, {unknown} unknown / {total} total)"
    )


# Each binary's own run, as measured on the fix/ny-cert-bins-rows tree, and
# the floor gate.sh holds it to.
_REAL_BIN_LINES = {
    "certify_onnx": _bin_line(65, 90, 711, 866),
    "certify_onnx_diff": _bin_line(23, 19, 251, 293),
    "frontier": _bin_line(7, 2, 26, 35),
    "lrat_to_lean": _bin_line(6, 3, 29, 38),
}
_BIN_FLOORS = {
    "certify_onnx": 50,
    "certify_onnx_diff": 15,
    "frontier": 5,
    "lrat_to_lean": 4,
}
_BINS = sorted(_REAL_BIN_LINES)


def run_lane_o(
    tmp_path: Path,
    gate_line: str,
    exit_code: int = 0,
    bins: dict[str, tuple[str, int]] | None = None,
    version: str | None = None,
):
    """Lane [o] with `gate_line` for the `--lib` run; each `--bin <name>` run
    replays the measured passing line unless `bins` overrides it. The prover
    is m29's 321aaeda7 unless `version` names another."""
    per_target = {f"--bin {name}": (line, 0) for name, line in _REAL_BIN_LINES.items()}
    for name, replay in (bins or {}).items():
        per_target[f"--bin {name}"] = replay
    tmp_path.mkdir(parents=True, exist_ok=True)
    return run_lane(
        tmp_path,
        "o",
        gate_line,
        exit_code,
        per_target,
        version=_M29_VERSION if version is None else version,
    )


# What the fake targo answers to `--version`: the prover lane [o] keys its
# proved floors by. 321aaeda7 is the m29 build the original floors (and the
# _REAL_* lines above) were measured under.
_M29_VERSION = "targo 1.99.0-dev (321aaeda7 2026-09-17) (targo 0.10.0+g321aaeda7)"


def run_lane(
    tmp_path: Path,
    lane: str,
    gate_line: str,
    exit_code: int = 0,
    per_target: dict[str, tuple[str, int]] | None = None,
    version: str = _M29_VERSION,
    gate: Path = GATE,
    env: dict[str, str] | None = None,
):
    """Run `gate.sh --only <lane>` against a fake pinned sysroot.

    `gate` may be a copy of gate.sh under a synthetic repository root (it
    resolves the root from its own path), for lanes whose prerequisites are
    files in the tree. `env` adds to the hermetic environment.
    """
    fake_bin = tmp_path / "bin"
    sysroot_bin = tmp_path / "trust-sysroot" / "bin"
    fake_bin.mkdir()
    sysroot_bin.mkdir(parents=True)
    argv_log = tmp_path / "targo-argv"
    rustc = sysroot_bin / "rustc"
    rustup = fake_bin / "rustup"
    rustup.write_text(f'#!/bin/sh\nprintf "%s\\n" "{rustc}"\n', encoding="utf-8")
    rustup.chmod(0o755)
    rustc.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "{sysroot_bin.parent}"\n', encoding="utf-8"
    )
    rustc.chmod(0o755)
    line_file = tmp_path / "gate-line"
    line_file.write_text(gate_line + "\n" if gate_line else "", encoding="utf-8")
    # A run whose argv contains a `per_target` key replays that key's line
    # and exit code; every other invocation replays `gate_line` / `exit_code`.
    cases = ""
    for index, (key, (line, code)) in enumerate((per_target or {}).items()):
        target_file = tmp_path / f"gate-line-{index}"
        target_file.write_text(line + "\n" if line else "", encoding="utf-8")
        cases += f'  *" {key} "*) cat "{target_file}"; exit {code} ;;\n'
    version_file = tmp_path / "targo-version"
    version_file.write_text(version + "\n" if version else "", encoding="utf-8")
    targo = sysroot_bin / "targo"
    targo.write_text(
        "#!/bin/sh\n"
        # `--version` is the prover probe, not a lane run: answer it unlogged.
        f'if [ "$*" = "--version" ]; then cat "{version_file}"; exit 0; fi\n'
        f'printf "%s\\n" "$*" >> "{argv_log}"\n'
        'case " $* " in\n'
        f"{cases}"
        "esac\n"
        f'cat "{line_file}"\n'
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    targo.chmod(0o755)

    bash = shutil.which("bash")
    assert bash is not None
    environment = hermetic_environment()
    environment.update(
        {
            "NY_RUSTUP": str(rustup),
            "NY_SKIP_CREDENTIAL_CHECK": "1",
            "PATH": f"{fake_bin}{os.pathsep}/usr/bin{os.pathsep}/bin",
        }
    )
    environment.update(env or {})
    result = subprocess.run(
        [bash, str(gate), "--only", lane],
        cwd=gate.parent.parent,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    argv = argv_log.read_text(encoding="utf-8") if argv_log.exists() else ""
    return result, argv


def test_lane_o_runs_the_advisory_verified_lane_on_the_lib_and_each_bin(
    tmp_path: Path,
) -> None:
    result, argv = run_lane_o(tmp_path, _REAL_GATE_LINE)

    assert result.returncode == 0, result.stdout + result.stderr
    # One run for the library first, then one per binary, each over exactly
    # one target selection.
    assert argv.splitlines() == [
        "trust check --allow-l0-gaps --locked -p ny-cert --lib",
        *(f"trust check --allow-l0-gaps --locked -p ny-cert --bin {b}" for b in _BINS),
    ]
    # Every condition of each pass is disclosed: each ok line echoes that
    # run's whole Gate line, so unknown, assumed and mandated rows all appear
    # and none is reported as proved.
    assert "0 failed, 309 proved (floor 250)" in result.stdout
    assert result.stdout.count("CONDITION, not proved") == 1 + len(_BINS)
    lib_tail = result.stdout.split("ok    o/lib:", 1)[1].split("+ targo", 1)[0]
    for condition in ("561 unknown / 886 total", "6 assumed", "10 mandated"):
        assert condition in lib_tail
    for name, line in _REAL_BIN_LINES.items():
        tail = result.stdout.split(f"ok    o/bin:{name}:", 1)[1].split("+ targo", 1)[0]
        assert f"(floor {_BIN_FLOORS[name]})" in tail
        assert line.strip() in tail
    assert "PASS  [o]" in result.stdout


def _floor_sets() -> dict[str, dict[str, int]]:
    """LANE_O_FLOOR_SETS as {prover: {target: floor}}."""
    source = GATE.read_text(encoding="utf-8")
    declared = re.search(r"^LANE_O_FLOOR_SETS='\n(.*?)\n'$", source, re.M | re.S)
    assert declared is not None, "LANE_O_FLOOR_SETS not found in gate.sh"
    sets = {}
    for line in declared.group(1).splitlines():
        prover, *entries = line.split()
        sets[prover] = {k: int(v) for k, v in (e.split(":") for e in entries)}
    return sets


def test_lane_o_bin_list_matches_the_crate_binaries() -> None:
    # The fake replays the m29 floors gate.sh declares; every prover's set
    # names exactly the binaries under crates/ny-cert/src/bin.
    sets = _floor_sets()
    assert sets["321aaeda7"] == {"lib": 250, **_BIN_FLOORS}
    on_disk = sorted(
        p.stem for p in (REPO_ROOT / "crates/ny-cert/src/bin").glob("*.rs")
    )
    assert on_disk == _BINS
    for prover, floors in sets.items():
        assert sorted(k for k in floors if k != "lib") == _BINS, prover
        assert all(floor >= 1 for floor in floors.values()), prover


def test_lane_o_fails_on_a_failed_row_even_when_targo_exits_zero(
    tmp_path: Path,
) -> None:
    line = _REAL_GATE_LINE.replace("0 failed", "3 failed")
    result, _ = run_lane_o(tmp_path, line, exit_code=0)

    assert result.returncode == 1
    assert "FAIL  o/lib: 3 FAILED verified-lane row(s)" in result.stdout


@pytest.mark.parametrize("name", _BINS)
def test_lane_o_fails_on_a_failed_row_in_any_one_bin_behind_passing_others(
    tmp_path: Path, name: str
) -> None:
    line = _REAL_BIN_LINES[name].replace("0 failed", "2 failed")
    result, argv = run_lane_o(tmp_path, _REAL_GATE_LINE, bins={name: (line, 0)})

    assert result.returncode == 1
    assert "ok    o/lib:" in result.stdout
    assert f"FAIL  o/bin:{name}: 2 FAILED verified-lane row(s)" in result.stdout
    for other in _BINS:
        if other != name:
            assert f"ok    o/bin:{other}:" in result.stdout
    assert len(argv.splitlines()) == 1 + len(_BINS)


def test_lane_o_still_runs_every_bin_when_the_lib_fails(tmp_path: Path) -> None:
    # A library failure must not hide the binaries' verdicts: all run, and
    # every verdict is reported.
    lib = _REAL_GATE_LINE.replace("0 failed", "1 failed")
    first = _BINS[0]
    bad = _REAL_BIN_LINES[first].replace("0 failed", "4 failed")
    result, argv = run_lane_o(tmp_path, lib, bins={first: (bad, 0)})

    assert result.returncode == 1
    assert "FAIL  o/lib: 1 FAILED" in result.stdout
    assert f"FAIL  o/bin:{first}: 4 FAILED" in result.stdout
    assert [a.rsplit(" ", 1)[-1] for a in argv.splitlines()[1:]] == _BINS


@pytest.mark.parametrize("name", _BINS)
def test_lane_o_one_bin_going_dark_fails_even_when_the_rest_are_high(
    tmp_path: Path, name: str
) -> None:
    # The review's blind spot under one `--bins` floor: certify_onnx_diff's
    # 22 proved could vanish (102 - 22 = 80, the old floor) and the lane
    # stayed green. Each binary now has its own floor, so any one binary
    # whose solver goes dark (0 proved) fails the lane.
    line = re.sub(r"\(\d+ proved", "(0 proved", _REAL_BIN_LINES[name])
    result, _ = run_lane_o(tmp_path, _REAL_GATE_LINE, bins={name: (line, 0)})

    assert result.returncode == 1
    assert (
        f"FAIL  o/bin:{name}: only 0 proved of "
        in result.stdout
    )
    assert f"below the floor of {_BIN_FLOORS[name]}" in result.stdout


def test_lane_o_a_bin_without_a_gate_line_is_void(tmp_path: Path) -> None:
    name = _BINS[-1]
    result, _ = run_lane_o(tmp_path, _REAL_GATE_LINE, bins={name: ("", 0)})

    assert result.returncode == 1
    assert f"FAIL  o/bin:{name}: no 'Gate:' line" in result.stdout


def test_lane_o_a_bin_nonzero_exit_with_zero_failed_still_fails(
    tmp_path: Path,
) -> None:
    name = _BINS[0]
    result, _ = run_lane_o(
        tmp_path, _REAL_GATE_LINE, bins={name: (_REAL_BIN_LINES[name], 101)}
    )

    assert result.returncode == 1
    assert (
        f"FAIL  o/bin:{name}: targo trust check exited 101 with 0 failed rows"
        in result.stdout
    )


def test_lane_o_without_a_gate_line_is_void_not_pass(tmp_path: Path) -> None:
    # A frontend that prints nothing (e.g. stock cargo under targo's name)
    # and exits 0 must not read as a verified pass.
    result, _ = run_lane_o(tmp_path, "", exit_code=0)

    assert result.returncode == 1
    assert "FAIL  o/lib: no 'Gate:' line" in result.stdout


def test_lane_o_with_zero_obligations_is_void(tmp_path: Path) -> None:
    line = (
        "  Gate: PASS (0 proved, 0 failed, 0 runtime-checked, 0 assumed, "
        "0 mandated, 0 contract-panic, 0 unknown / 0 total)"
    )
    result, _ = run_lane_o(tmp_path, line)

    assert result.returncode == 1
    assert "verified NOTHING" in result.stdout


def test_lane_o_nonzero_exit_with_zero_failed_still_fails(tmp_path: Path) -> None:
    result, _ = run_lane_o(tmp_path, _REAL_GATE_LINE, exit_code=101)

    assert result.returncode == 1
    assert "exited 101 with 0 failed rows" in result.stdout


def test_lane_o_all_unknown_is_not_a_pass(tmp_path: Path) -> None:
    # A solver that goes dark turns every row unknown: 0 failed, non-zero
    # total, exit 0. That verified nothing and must not read as green.
    line = (
        "  Gate: CONDITIONAL-PASS [advisory lane, exit 0] (0 proved, 0 failed, "
        "0 runtime-checked, 0 assumed, 0 mandated, 0 contract-panic, "
        "886 unknown / 886 total)"
    )
    result, _ = run_lane_o(tmp_path, line)

    assert result.returncode == 1
    assert "FAIL  o/lib: only 0 proved of 886, below the floor of 250" in result.stdout


def test_lane_o_proved_below_floor_fails(tmp_path: Path) -> None:
    line = _REAL_GATE_LINE.replace("309 proved", "40 proved").replace(
        "561 unknown", "830 unknown"
    )
    result, _ = run_lane_o(tmp_path, line)

    assert result.returncode == 1
    assert "only 40 proved of 886" in result.stdout


@pytest.mark.parametrize("verdict", ["FAIL", "INCONCLUSIVE", "VOID", "WEIRD"])
def test_lane_o_non_pass_verdict_fails_at_zero_failed_and_exit_zero(
    tmp_path: Path, verdict: str
) -> None:
    line = _REAL_GATE_LINE.replace("CONDITIONAL-PASS", verdict)
    result, _ = run_lane_o(tmp_path, line, exit_code=0)

    assert result.returncode == 1
    assert f"Gate verdict is {verdict}, not PASS/CONDITIONAL-PASS" in result.stdout


def test_lane_o_bare_pass_verdict_is_accepted(tmp_path: Path) -> None:
    line = _REAL_GATE_LINE.replace("CONDITIONAL-PASS", "PASS")
    result, _ = run_lane_o(tmp_path, line)

    assert result.returncode == 0, result.stdout + result.stderr


def test_lane_o_a_failing_gate_line_cannot_hide_behind_a_later_pass(
    tmp_path: Path,
) -> None:
    failing = _REAL_GATE_LINE.replace("0 failed", "5 failed").replace(
        "CONDITIONAL-PASS", "FAIL"
    )
    result, _ = run_lane_o(tmp_path, failing + "\n" + _REAL_GATE_LINE)

    assert result.returncode == 1
    assert "FAIL  o/lib: 2 'Gate:' lines, expected exactly 1" in result.stdout


def test_lane_o_a_bin_failing_gate_line_cannot_hide_behind_a_later_pass(
    tmp_path: Path,
) -> None:
    name = _BINS[1]
    real = _REAL_BIN_LINES[name]
    failing = real.replace("0 failed", "5 failed").replace("CONDITIONAL-PASS", "FAIL")
    result, _ = run_lane_o(
        tmp_path, _REAL_GATE_LINE, bins={name: (failing + "\n" + real, 0)}
    )

    assert result.returncode == 1
    assert f"FAIL  o/bin:{name}: 2 'Gate:' lines, expected exactly 1" in result.stdout


def test_lane_m_nonzero_exit_names_its_cause(tmp_path: Path) -> None:
    # Before the set +e window, a failing targo aborted lane [m] under
    # `set -euo pipefail` before it could print why it failed.
    result, _ = run_lane(tmp_path, "m", "error: could not compile", exit_code=101)

    assert result.returncode == 1
    assert "exited 101" in result.stdout


# ---------------------------------------------------------------------------
# VOID: a lane that cannot run its whole contract on this host says so with
# its own status and exit code (3), never PASS and never a test FAIL.

_M21_VERSION = "targo 1.99.0-dev (6606d3bdf 2026-09-25) (targo 0.10.0+g6606d3bdf)"
_UNMEASURED_VERSION = "targo 1.99.0-dev (abcdef012 2026-10-04) (targo 0.10.0+gabcdef012)"


def test_lane_o_under_an_unmeasured_prover_is_void_not_pass(tmp_path: Path) -> None:
    # A proved count is a property of (source, prover): under a prover with no
    # recorded floor set the counts are not compared — in either direction.
    result, argv = run_lane_o(tmp_path, _REAL_GATE_LINE, version=_UNMEASURED_VERSION)

    assert result.returncode == 3, result.stdout + result.stderr
    assert "NO proved floors are recorded for it" in result.stdout
    assert "VOID  [o]" in result.stdout
    assert "none measured under prover targo abcdef012" in result.stdout
    assert "GATE: VOID (0 pass, 0 fail, 1 void, 14 skipped)" in result.stdout
    # It still ran every target.
    assert len(argv.splitlines()) == 1 + len(_BINS)


def test_lane_o_unparseable_prover_is_void(tmp_path: Path) -> None:
    result, _ = run_lane_o(tmp_path, _REAL_GATE_LINE, version="cargo 1.95.0")

    assert result.returncode == 3, result.stdout + result.stderr
    assert "prover targo <unparsed>" in result.stdout


def test_lane_o_unmeasured_prover_still_fails_a_refuted_row(tmp_path: Path) -> None:
    line = _REAL_GATE_LINE.replace("0 failed", "3 failed")
    result, _ = run_lane_o(tmp_path, line, version=_UNMEASURED_VERSION)

    assert result.returncode == 1
    assert "FAIL  o/lib: 3 FAILED verified-lane row(s)" in result.stdout
    assert "FAIL  [o]" in result.stdout


def test_lane_o_unmeasured_prover_still_fails_a_dark_solver(tmp_path: Path) -> None:
    name = _BINS[0]
    line = re.sub(r"\(\d+ proved", "(0 proved", _REAL_BIN_LINES[name])
    result, _ = run_lane_o(
        tmp_path, _REAL_GATE_LINE, bins={name: (line, 0)}, version=_UNMEASURED_VERSION
    )

    assert result.returncode == 1
    assert f"FAIL  o/bin:{name}: only 0 proved of" in result.stdout


def test_lane_o_floors_follow_the_running_prover(tmp_path: Path) -> None:
    # The same input proves frontier 2 / lrat_to_lean 2 under the m21 prover
    # (measured 2026-10-04): that clears 6606d3bdf's floors and must fail
    # 321aaeda7's, which were measured where the same binaries prove 7 and 6.
    m21_lines = {
        "certify_onnx": _bin_line(66, 70, 307, 788),
        "certify_onnx_diff": _bin_line(24, 19, 99, 289),
        "frontier": _bin_line(2, 3, 23, 33),
        "lrat_to_lean": _bin_line(2, 4, 9, 40),
    }
    lib = _REAL_GATE_LINE.replace("309 proved", "274 proved")
    bins = {name: (line, 0) for name, line in m21_lines.items()}
    floors = _floor_sets()["6606d3bdf"]

    result, _ = run_lane_o(tmp_path / "m21", lib, bins=bins, version=_M21_VERSION)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"0 failed, 2 proved (floor {floors['frontier']})" in result.stdout

    result, _ = run_lane_o(tmp_path / "m29", lib, bins=bins, version=_M29_VERSION)
    assert result.returncode == 1
    assert "FAIL  o/bin:frontier: only 2 proved of 33, below the floor of 5" in result.stdout


def _gate_with_floor_sets(tmp_path: Path, old: str, new: str) -> Path:
    """A copy of gate.sh under a synthetic root with one edit to its floor
    table, beside the ny-cert binaries the lane checks the table against."""
    source = GATE.read_text(encoding="utf-8")
    assert old in source, f"gate.sh no longer contains {old!r}"
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    gate = root / "scripts/gate.sh"
    gate.write_text(source.replace(old, new), encoding="utf-8")
    gate.chmod(0o755)
    shutil.copy2(REPO_ROOT / "rust-toolchain.toml", root / "rust-toolchain.toml")
    bins = root / "crates/ny-cert/src/bin"
    bins.mkdir(parents=True)
    for name in _BINS:
        (bins / f"{name}.rs").write_text("fn main() {}\n", encoding="utf-8")
    return gate


_M21_BIN_LINES = {
    "certify_onnx": _bin_line(66, 70, 307, 788),
    "certify_onnx_diff": _bin_line(24, 19, 99, 289),
    "frontier": _bin_line(2, 3, 23, 33),
    "lrat_to_lean": _bin_line(2, 4, 9, 40),
}


def _run_lane_o_m21(tmp_path: Path, gate: Path, lib_proved: int):
    per_target = {f"--bin {name}": (line, 0) for name, line in _M21_BIN_LINES.items()}
    lib = _REAL_GATE_LINE.replace("309 proved", f"{lib_proved} proved")
    run = tmp_path / "run"
    run.mkdir()
    return run_lane(run, "o", lib, 0, per_target, version=_M21_VERSION, gate=gate)


def test_lane_o_a_floor_set_without_the_library_floor_fails(tmp_path: Path) -> None:
    # Reviewer plant, 2026-10-08: with `lib:` dropped from a recorded set the
    # library was gated at the unmeasured-prover fallback of 1, and 5 proved
    # (270 is typical) passed. A recorded set must carry the library's floor.
    gate = _gate_with_floor_sets(
        tmp_path, "6606d3bdf lib:200 certify_onnx:35", "6606d3bdf certify_onnx:35"
    )
    result, argv = _run_lane_o_m21(tmp_path, gate, lib_proved=5)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL  o/floors: prover 6606d3bdf's floor set is malformed: [no lib: floor]" in result.stdout
    assert "FAIL  o/lib: proved floor '' is not a positive integer" in result.stdout
    assert "ok    o/lib:" not in result.stdout
    # Every target still ran, and the binaries kept their own floors.
    assert len(argv.splitlines()) == 1 + len(_BINS)
    assert "ok    o/bin:frontier: 0 failed, 2 proved (floor 1)" in result.stdout


def test_lane_o_a_non_integer_floor_fails(tmp_path: Path) -> None:
    # `lib:2OO` (letter O): `[ 5 -lt 2OO ]` is a bash error an `if` reads as
    # false, so the mistyped floor passed every proved count.
    gate = _gate_with_floor_sets(tmp_path, "6606d3bdf lib:200 ", "6606d3bdf lib:2OO ")
    result, _ = _run_lane_o_m21(tmp_path, gate, lib_proved=5)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL  o/floors: prover 6606d3bdf's floor set is malformed: [lib:2OO]" in result.stdout
    assert "FAIL  o/lib: proved floor '2OO' is not a positive integer" in result.stdout
    assert "integer expression expected" not in result.stdout + result.stderr


def test_lane_o_a_zero_or_repeated_floor_fails(tmp_path: Path) -> None:
    gate = _gate_with_floor_sets(tmp_path / "zero", "frontier:1 lrat_to_lean:1", "frontier:0 lrat_to_lean:1")
    result, _ = _run_lane_o_m21(tmp_path / "zero", gate, lib_proved=274)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "floor set is malformed: [frontier:0]" in result.stdout

    gate = _gate_with_floor_sets(
        tmp_path / "twice", "6606d3bdf lib:200 ", "6606d3bdf lib:1 lib:200 "
    )
    result, _ = _run_lane_o_m21(tmp_path / "twice", gate, lib_proved=5)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "floor set is malformed: [more than one floor for: lib]" in result.stdout


def test_lane_o_a_malformed_set_of_another_prover_still_fails_the_lane(
    tmp_path: Path,
) -> None:
    # The table is checked whole: the running prover's set is fine here, but a
    # host under the other prover would be gated by the broken one.
    gate = _gate_with_floor_sets(tmp_path, "321aaeda7 lib:250 ", "321aaeda7 lib:25O ")
    result, _ = _run_lane_o_m21(tmp_path, gate, lib_proved=274)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL  o/floors: prover 321aaeda7's floor set is malformed: [lib:25O]" in result.stdout
    assert "ok    o/lib: 0 failed, 274 proved (floor 200)" in result.stdout


def test_lane_o_the_unmodified_table_passes_under_a_synthetic_root(tmp_path: Path) -> None:
    # Control for the four plants above: the same harness with the table
    # untouched passes, and the real set fails 5 proved on its own floor.
    gate = _gate_with_floor_sets(tmp_path / "ok", "LANE_O_FLOOR_SETS='", "LANE_O_FLOOR_SETS='")
    result, _ = _run_lane_o_m21(tmp_path / "ok", gate, lib_proved=274)
    assert result.returncode == 0, result.stdout + result.stderr

    gate = _gate_with_floor_sets(tmp_path / "low", "LANE_O_FLOOR_SETS='", "LANE_O_FLOOR_SETS='")
    result, _ = _run_lane_o_m21(tmp_path / "low", gate, lib_proved=5)
    assert result.returncode == 1
    assert "FAIL  o/lib: only 5 proved of 886, below the floor of 200" in result.stdout


# targo's refusal to START the verified lane, verbatim in shape (measured on
# m21, 2026-10-04, while a Trust stage-1 build's trustd held the socket):
# the reason spans lines, there is no Gate line, and targo exits non-zero.
_SOCKET = "/private/tmp/trustd-runtime-locks-501/trust-memory-jobserver.sock"
_NO_AUTHORITY = (
    "targo trust: using codegen backend `llvm`\n"
    "targo trust: could not configure Cargo verifier options: could not establish "
    f"the selected same-sysroot trustd memory authority at {_SOCKET}: same-sysroot "
    "trustd /t/bin/trustd exited before exact-identity readiness with exit status: 1; "
    f"bounded stderr: trustd: memory-coordination daemon binding {_SOCKET}\n"
    f"trustd: fatal: cannot bind socket {_SOCKET}: another trustd owns {_SOCKET}\n"
    "; refusing to launch a split-authority Cargo fan-out. A trustd of a DIFFERENT "
    "build owns that socket; WAIT for it to retire. Do NOT signal it"
)


def test_lane_o_a_trustd_authority_refusal_is_void_naming_the_socket(
    tmp_path: Path,
) -> None:
    # The host could not run the verified lane at all: nothing was compiled or
    # verified, and no change to the tree causes or cures it. Not a FAIL (it
    # read as five on 2026-10-04) and never a pass.
    bins = dict.fromkeys(_BINS, (_NO_AUTHORITY, 101))
    result, argv = run_lane_o(tmp_path, _NO_AUTHORITY, exit_code=101, bins=bins)

    assert result.returncode == 3, result.stdout + result.stderr
    assert "VOID  [o]" in result.stdout
    assert "FAIL  o/" not in result.stdout
    assert (
        "NOT CHECKED: lane [o] lib: the verified lane did not start — targo could "
        f"not establish this host's trustd memory authority at {_SOCKET} "
        f"(trustd: fatal: cannot bind socket {_SOCKET}: another trustd owns {_SOCKET})"
        in result.stdout
    )
    assert f"(+{len(_BINS)} more)" in result.stdout
    assert "GATE: VOID (0 pass, 0 fail, 1 void, 14 skipped)" in result.stdout
    # Every target was still attempted, and each refusal is named.
    assert len(argv.splitlines()) == 1 + len(_BINS)
    for name in _BINS:
        assert f"VOID  lane [o] bin:{name}: the verified lane did not start" in result.stdout


def test_lane_o_an_authority_refusal_without_a_fatal_line_quotes_targos_reason(
    tmp_path: Path,
) -> None:
    # The other variant of the refusal (sibling trustd missing, or its crash
    # epoch not CLEAN) carries no `trustd: fatal:` line: the reason is the
    # text after the socket.
    refusal = (
        "targo trust: could not configure Cargo verifier options: could not establish "
        f"the selected same-sysroot trustd memory authority at {_SOCKET}: trustd crash "
        "epoch is not CLEAN; refusing to launch a split-authority Cargo fan-out. "
        "Confirm the sibling trustd is installed/executable"
    )
    bins = dict.fromkeys(_BINS, (refusal, 101))
    result, _ = run_lane_o(tmp_path, refusal, exit_code=101, bins=bins)

    assert result.returncode == 3, result.stdout + result.stderr
    assert f"authority at {_SOCKET} (trustd crash epoch is not CLEAN; refusing" in result.stdout


def test_lane_o_a_refuted_row_wins_over_an_authority_refusal(tmp_path: Path) -> None:
    # The library ran and refuted a row; a later target was refused. FAIL wins.
    lib = _REAL_GATE_LINE.replace("0 failed", "3 failed")
    name = _BINS[-1]
    result, argv = run_lane_o(tmp_path, lib, bins={name: (_NO_AUTHORITY, 101)})

    assert result.returncode == 1
    assert "FAIL  o/lib: 3 FAILED verified-lane row(s)" in result.stdout
    assert f"VOID  lane [o] bin:{name}: the verified lane did not start" in result.stdout
    assert "FAIL  [o]" in result.stdout
    assert "VOID  [o]" not in result.stdout
    assert len(argv.splitlines()) == 1 + len(_BINS)


def test_lane_o_one_refused_target_makes_the_lane_void_not_pass(tmp_path: Path) -> None:
    # Four targets verified and one was refused: the lane did not run its
    # whole contract, so it is VOID naming that target, never a PASS.
    name = _BINS[0]
    result, _ = run_lane_o(tmp_path, _REAL_GATE_LINE, bins={name: (_NO_AUTHORITY, 101)})

    assert result.returncode == 3, result.stdout + result.stderr
    assert "ok    o/lib:" in result.stdout
    assert f"NOT CHECKED: lane [o] bin:{name}: the verified lane did not start" in result.stdout
    assert "PASS  [o]" not in result.stdout


def test_lane_o_an_authority_refusal_that_exits_zero_is_a_fail(tmp_path: Path) -> None:
    # Only targo's refusal WITH its non-zero exit is the host condition. The
    # same words on a zero exit are not a refusal anyone can account for.
    bins = dict.fromkeys(_BINS, (_NO_AUTHORITY, 0))
    result, _ = run_lane_o(tmp_path, _NO_AUTHORITY, exit_code=0, bins=bins)

    assert result.returncode == 1
    assert "FAIL  o/lib: no 'Gate:' line" in result.stdout
    assert "VOID  [o]" not in result.stdout


def test_lane_o_other_silence_with_a_nonzero_exit_is_still_a_fail(tmp_path: Path) -> None:
    # A compile error also ends with no Gate line and a non-zero exit; that is
    # a defect in the tree and must stay a FAIL.
    result, _ = run_lane_o(tmp_path, "error: could not compile `ny-cert`", exit_code=101)

    assert result.returncode == 1
    assert "FAIL  o/lib: no 'Gate:' line" in result.stdout
    assert "VOID  [o]" not in result.stdout


def test_a_lane_exiting_3_without_a_recorded_reason_is_a_fail(tmp_path: Path) -> None:
    # VOID needs BOTH the code and a reason recorded with lane_void: a tool's
    # own exit 3 (pytest: internal error) must not read as "data absent".
    tmp_path.mkdir(exist_ok=True)
    result, _ = run_lane(tmp_path, "d", "", exit_code=3)

    assert result.returncode == 1
    assert "FAIL  [d]" in result.stdout
    assert "VOID" not in result.stdout


# Lane [i]: the guard run (`--skip` the data test) and the data run (`--exact`).
_PRESENCE = "preset::model_load_smoke_tests::shipped_preset_benchmark_data_is_present"
_GUARD = "preset::model_load_smoke_tests::every_shipped_preset_model_still_loads"
_GUARD_OK = "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 2209 filtered out"
_DATA_OK = "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2213 filtered out"
_LOADS_HELD = "preset load smoke: 35 model file(s) checked across 24 preset(s) (mode=family-sample)"
_GUARD_ABSENT_PANIC = (
    "BENCHMARK DATA IS NOT ON THIS BOX — every model that IS here loaded (35 model file(s) "
    "across 24 preset(s)), but 31 input(s) could NOT be checked because this box lacks them."
)


def _guard_failed(names: list[str], passed: int, body: str) -> str:
    """A libtest `--nocapture` log whose failure list is `names`."""
    listed = "".join(f"    {name}\n" for name in names)
    return (
        f"{body}\n\nfailures:\n\nfailures:\n{listed}\n"
        f"test result: FAILED. {passed} passed; {len(names)} failed; 0 ignored; "
        "0 measured; 2209 filtered out"
    )


# The guard's absent-data outcome: its loads held (the line), then it failed
# for the inputs it could not check, and it is the only failed test.
_GUARD_ABSENT = _guard_failed([_GUARD], 4, f"{_LOADS_HELD}\n{_GUARD_ABSENT_PANIC}")
_DATA_ABSENT = (
    "BENCHMARK DATA IS NOT ON THIS BOX — the model-load guard could not check 2 input(s)\n"
    "  configs/vnncomp25/acasxu_2023.yaml\n"
    "      root:    benchmarks/vnncomp2025/benchmarks/acasxu_2023\n"
    "      missing: /r/benchmarks/vnncomp2025/benchmarks/acasxu_2023\n"
    "  configs/vnncomp25/cifar100_2024.yaml\n"
    "      root:    benchmarks/vnncomp2025/benchmarks/cifar100_2024\n"
    "      missing: /r/benchmarks/vnncomp2025/benchmarks/cifar100_2024/instances.csv\n"
    "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 2213 filtered out"
)


def run_lane_i(tmp_path: Path, guard: tuple[str, int], data: tuple[str, int]):
    return run_lane(tmp_path, "i", "", 0, {"--skip": guard, "--exact": data})


def test_lane_i_passes_when_every_model_loads_and_all_data_is_present(
    tmp_path: Path,
) -> None:
    result, argv = run_lane_i(tmp_path, (_GUARD_OK, 0), (_DATA_OK, 0))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  [i]" in result.stdout
    runs = argv.splitlines()
    assert len(runs) == 2
    assert runs[0].endswith(f"-- --nocapture --skip {_PRESENCE}")
    assert runs[1].endswith(f"{_PRESENCE} -- --nocapture --exact")


def test_lane_i_absent_data_is_void_naming_every_path(tmp_path: Path) -> None:
    # The guard loaded everything that is here and then failed for what is
    # not; the data test names the paths. VOID, exit 3: not a pass, not a FAIL.
    result, _ = run_lane_i(tmp_path, (_GUARD_ABSENT, 101), (_DATA_ABSENT, 101))

    assert result.returncode == 3, result.stdout + result.stderr
    assert (
        "VOID  [i] preset model-load smoke (every shipped preset's models load)"
        in result.stdout
    )
    assert (
        "NOT CHECKED: benchmark data absent: "
        "/r/benchmarks/vnncomp2025/benchmarks/acasxu_2023 (+1 more)" in result.stdout
    )
    assert (
        "VOID  benchmark data absent: "
        "/r/benchmarks/vnncomp2025/benchmarks/cifar100_2024/instances.csv"
        in result.stdout
    )
    assert "FAIL  i:" not in result.stdout


def test_lane_i_no_data_at_all_is_void(tmp_path: Path) -> None:
    # Nothing is on the box: the guard loads nothing, so its loads-held line is
    # the first thing it writes and (one test thread) continues libtest's
    # `test <name> ... ` line instead of starting its own.
    guard = _guard_failed(
        [_GUARD],
        4,
        f"test {_GUARD} ... preset load smoke: 0 model file(s) checked across 0 preset(s) "
        f"(mode=family-sample)\n\nthread '{_GUARD}' panicked at x.rs:1:1:\n{_GUARD_ABSENT_PANIC}",
    )
    result, _ = run_lane_i(tmp_path, (guard, 101), (_DATA_ABSENT, 101))

    assert result.returncode == 3, result.stdout + result.stderr
    assert "VOID  [i]" in result.stdout


def test_lane_i_a_load_failure_is_a_fail_even_with_data_absent(tmp_path: Path) -> None:
    # The guard died on a model that stopped loading: no loads-held line, and
    # the absent-data words do not appear in ITS panic.
    guard = _guard_failed([_GUARD], 4, "SHIPPED PRESET MODELS NO LONGER LOAD — THIS PUTS 100.0")
    result, argv = run_lane_i(tmp_path, (guard, 101), (_DATA_ABSENT, 101))

    assert result.returncode == 1
    assert "FAIL  [i]" in result.stdout
    assert "the model-load guard failed" in result.stdout
    assert len(argv.splitlines()) == 1


def test_lane_i_absent_data_cannot_hide_a_second_failing_guard_test(
    tmp_path: Path,
) -> None:
    # The guard reports absent data, but another test of the guard run failed
    # too (the stake registry): that is a FAIL, not a VOID.
    registry = "preset::model_load_smoke_tests::every_shipped_preset_declares_its_banked_score_at_risk"
    guard = _guard_failed(
        [registry, _GUARD], 3, f"{_LOADS_HELD}\n{_GUARD_ABSENT_PANIC}"
    )
    result, argv = run_lane_i(tmp_path, (guard, 101), (_DATA_ABSENT, 101))

    assert result.returncode == 1
    assert "the model-load guard failed" in result.stdout
    assert "VOID  [i]" not in result.stdout
    assert len(argv.splitlines()) == 1


def test_lane_i_a_failure_of_another_guard_test_alone_is_a_fail(tmp_path: Path) -> None:
    # Exactly one failed test, with the absent-data words in the log — but the
    # failed test is not the guard, so nothing says the loads held for it.
    registry = "preset::model_load_smoke_tests::every_shipped_preset_declares_its_banked_score_at_risk"
    guard = _guard_failed([registry], 4, f"{_LOADS_HELD}\n{_GUARD_ABSENT_PANIC}")
    result, _ = run_lane_i(tmp_path, (guard, 101), (_DATA_ABSENT, 101))

    assert result.returncode == 1
    assert "the model-load guard failed" in result.stdout


def test_lane_i_absent_data_words_without_the_loads_held_line_is_a_fail(
    tmp_path: Path,
) -> None:
    guard = _guard_failed([_GUARD], 4, _GUARD_ABSENT_PANIC)
    result, _ = run_lane_i(tmp_path, (guard, 101), (_DATA_ABSENT, 101))

    assert result.returncode == 1
    assert "the model-load guard failed" in result.stdout


def test_lane_i_a_guard_that_passes_over_absent_data_is_a_fail(tmp_path: Path) -> None:
    # The pre-2026-10-09 guard: it returned success when inputs were absent and
    # left the failing to the data test, so selected by name it was a vacuous
    # green. The two halves disagreeing is a FAIL in either direction.
    result, _ = run_lane_i(tmp_path, (_GUARD_OK, 0), (_DATA_ABSENT, 101))

    assert result.returncode == 1
    assert "the data test reports absent benchmark data but the guard PASSED" in result.stdout
    assert "VOID  [i]" not in result.stdout

    other = tmp_path / "other"
    other.mkdir()
    result, _ = run_lane_i(other, (_GUARD_ABSENT, 101), (_DATA_OK, 0))

    assert result.returncode == 1
    assert "the two halves disagree" in result.stdout


def test_lane_i_an_absent_data_guard_run_below_the_floor_is_a_fail(tmp_path: Path) -> None:
    # The filter matched too little: three passed beside the guard's failure
    # where the floor needs four.
    guard = _guard_failed([_GUARD], 3, f"{_LOADS_HELD}\n{_GUARD_ABSENT_PANIC}")
    result, _ = run_lane_i(tmp_path, (guard, 101), (_DATA_ABSENT, 101))

    assert result.returncode == 1
    assert "the model-load guard failed" in result.stdout


def test_lane_i_a_guard_filter_matching_too_little_is_a_fail(tmp_path: Path) -> None:
    guard = "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 2211 filtered out"
    result, _ = run_lane_i(tmp_path, (guard, 0), (_DATA_OK, 0))

    assert result.returncode == 1
    assert "the guard filter ran 2 test(s), floor is 5" in result.stdout


def test_lane_i_a_data_run_that_matched_nothing_is_a_fail(tmp_path: Path) -> None:
    data = "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 2213 filtered out"
    result, _ = run_lane_i(tmp_path, (_GUARD_OK, 0), (data, 0))

    assert result.returncode == 1
    assert "neither a pass nor a data report" in result.stdout


# Lane [b]: a copy of gate.sh under a synthetic root, so the pinned Lake
# checkout's presence is the test's fixture rather than this host's state.
def _fake_root(tmp_path: Path, with_lake: bool) -> Path:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(GATE, root / "scripts/gate.sh")
    shutil.copy2(REPO_ROOT / "rust-toolchain.toml", root / "rust-toolchain.toml")
    lean = root / "crates/ny-cert/proofs/lean/.lake/packages"
    if with_lake:
        (lean / "crownproof/crown-proofs/lean/Crownproof").mkdir(parents=True)
        (lean / "mathlib/Mathlib").mkdir(parents=True)
    return root / "scripts/gate.sh"


_HERMETIC_OK = "test result: ok. 172 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
_PINNED_OK = "test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 176 filtered out"


def test_lane_b_without_the_lake_checkout_is_void_naming_both_paths(
    tmp_path: Path,
) -> None:
    gate = _fake_root(tmp_path, with_lake=False)
    result, argv = run_lane(
        tmp_path, "b", "", 0, {"--all-targets": (_HERMETIC_OK, 0)}, gate=gate
    )

    assert result.returncode == 3, result.stdout + result.stderr
    assert "VOID  [b]" in result.stdout
    root = gate.parent.parent
    for missing in (
        "crates/ny-cert/proofs/lean/.lake/packages/crownproof/crown-proofs/lean/Crownproof",
        "crates/ny-cert/proofs/lean/.lake/packages/mathlib/Mathlib",
    ):
        assert f"pinned Lake checkout absent: {root}/{missing}" in result.stdout
    # The hermetic suite ran; the grounding slice did not pretend to.
    assert argv.splitlines() == ["--unverified test --locked -p ny-cert --all-targets"]


def test_lane_b_a_hermetic_failure_is_a_fail_not_void(tmp_path: Path) -> None:
    gate = _fake_root(tmp_path, with_lake=False)
    hermetic = "test result: FAILED. 171 passed; 1 failed; 0 ignored; 0 measured"
    result, _ = run_lane(
        tmp_path, "b", "", 0, {"--all-targets": (hermetic, 101)}, gate=gate
    )

    assert result.returncode == 1
    assert "FAIL  [b]" in result.stdout
    assert "VOID" not in result.stdout


def test_lane_b_with_the_checkout_runs_the_pinned_corpus_slice(tmp_path: Path) -> None:
    gate = _fake_root(tmp_path, with_lake=True)
    result, argv = run_lane(
        tmp_path,
        "b",
        "",
        0,
        {"--all-targets": (_HERMETIC_OK, 0), "external-lean": (_PINNED_OK, 0)},
        gate=gate,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 pinned-corpus grounding test(s) passed" in result.stdout
    assert argv.splitlines()[1] == (
        "--unverified test --locked -p ny-cert --features external-lean --lib pinned_corpus"
    )


def test_lane_b_an_ungrounded_cite_is_a_fail(tmp_path: Path) -> None:
    gate = _fake_root(tmp_path, with_lake=True)
    pinned = "test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured"
    result, _ = run_lane(
        tmp_path,
        "b",
        "",
        0,
        {"--all-targets": (_HERMETIC_OK, 0), "external-lean": (pinned, 101)},
        gate=gate,
    )

    assert result.returncode == 1
    assert "the pinned-corpus grounding slice failed" in result.stdout


def test_lane_b_a_pinned_filter_matching_nothing_is_a_fail(tmp_path: Path) -> None:
    gate = _fake_root(tmp_path, with_lake=True)
    pinned = "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 180 filtered out"
    result, _ = run_lane(
        tmp_path,
        "b",
        "",
        0,
        {"--all-targets": (_HERMETIC_OK, 0), "external-lean": (pinned, 0)},
        gate=gate,
    )

    assert result.returncode == 1
    assert "the pinned_corpus filter ran 0 test(s), floor is 4" in result.stdout



# ---------------------------------------------------------------------------
# One failure must not hide another. Lanes [a] and [g] run every step and name
# each one that failed: on 2026-10-08 lane [a] showed fmt drift while three
# more reds sat unreported behind it, and lane [g]'s stale Python version had
# hidden a failing repository-policy contract for eight days.

_TEST_OK = "test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
_TEST_FAILED = "test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"


def _lane_a_steps(argv: str) -> set[str]:
    """Classify executed scopes without fixing command spelling or ordering."""
    reached = set()
    for command in map(shlex.split, argv.splitlines()):
        def option(name: str) -> str | None:
            for index, argument in enumerate(command):
                if argument == name:
                    return command[index + 1] if index + 1 < len(command) else None
                if argument.startswith(f"{name}="):
                    return argument.split("=", 1)[1]
            return None

        if command[0] == "fmt" and "--check" in command:
            reached.add("format")
        elif command[:2] == ["--unverified", "test"]:
            package = option("-p") or option("--package")
            if package == "ny-test-utils":
                reached.add(option("--test"))
            elif package == "ny-levers":
                reached.add("levers")
        elif command[0] == "tippy":
            if "--workspace" in command and "--all-targets" in command:
                reached.add("workspace-lint")
            if (
                (option("-p") or option("--package")) == "ny-cli"
                and option("--bin") == "ny"
                and {"mip", "cuda"} <= set((option("--features") or "").split(","))
            ):
                reached.add("scored-cli-lint")
    return reached


_LANE_A_SCOPES = {
    "format", "source_policy", "python_correctness_migration_manifest", "levers",
    "workspace-lint", "scored-cli-lint",
}


def test_lane_a_runs_every_step_and_names_each_failure(tmp_path: Path) -> None:
    # fmt and the ny-levers ratchets fail; source policy, the migration
    # inventory and both Clippy tiers pass. All six steps still run.
    result, argv = run_lane(
        tmp_path,
        "a",
        _TEST_OK,
        0,
        {
            "fmt --all": ("Diff in crates/x/src/lib.rs:1:", 1),
            "-p ny-levers": (_TEST_FAILED, 101),
        },
    )

    assert result.returncode == 1
    assert _LANE_A_SCOPES <= _lane_a_steps(argv)
    assert "FAIL  a: failing step(s): fmt ny-levers — every step ran" in result.stdout
    assert "FAIL  [a]" in result.stdout


def test_lane_a_a_late_failure_alone_still_fails_the_lane(tmp_path: Path) -> None:
    result, argv = run_lane(
        tmp_path, "a", _TEST_OK, 0, {"mip,cuda": ("error: lint", 101)}
    )

    assert result.returncode == 1
    assert "FAIL  a: failing step(s): tippy-ny-cli-mip-cuda — every step ran" in result.stdout
    assert _LANE_A_SCOPES <= _lane_a_steps(argv)


def test_lane_a_passes_when_every_step_passes(tmp_path: Path) -> None:
    result, argv = run_lane(tmp_path, "a", _TEST_OK)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  [a]" in result.stdout
    assert _LANE_A_SCOPES <= _lane_a_steps(argv)


def _fake_gate_python(tmp_path: Path, pytest_exit: int) -> Path:
    """An interpreter stand-in for lane [g]: the import probe (`-c`) succeeds
    and the plugin suite (`-m pytest`) exits `pytest_exit`."""
    python = tmp_path / "fake-python"
    python.write_text(
        "#!/bin/sh\n"
        'case " $* " in\n'
        f'  *" -m pytest "*) echo "plugin suite ran"; exit {pytest_exit} ;;\n'
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return python


_LANE_G_CARGO = (
    "--unverified test --locked --no-fail-fast -p ny-test-utils --features python-tools --tests"
)


def test_lane_g_runs_the_cargo_contracts_when_the_plugin_suite_fails(
    tmp_path: Path,
) -> None:
    # The stale-version failure must not keep the repository-policy and
    # tooling contracts from running (it hid one for eight days).
    python = _fake_gate_python(tmp_path, pytest_exit=1)
    result, argv = run_lane(
        tmp_path, "g", _TEST_FAILED, 101, env={"NY_GATE_PYTHON": str(python)}
    )

    assert result.returncode == 1
    assert "plugin suite ran" in result.stdout
    assert argv.splitlines() == [_LANE_G_CARGO]
    assert "FAIL  g: the pytest plugin suite failed" in result.stdout
    assert (
        "FAIL  g: a Cargo-owned Python tooling or repository-policy contract failed"
        in result.stdout
    )
    assert "FAIL  g: failing half(s): plugin-suite cargo-contracts — both ran" in result.stdout


def test_lane_g_a_cargo_contract_failure_alone_fails_the_lane(tmp_path: Path) -> None:
    python = _fake_gate_python(tmp_path, pytest_exit=0)
    result, _ = run_lane(
        tmp_path, "g", _TEST_FAILED, 101, env={"NY_GATE_PYTHON": str(python)}
    )

    assert result.returncode == 1
    assert "FAIL  g: failing half(s): cargo-contracts — both ran" in result.stdout
    assert "the pytest plugin suite failed" not in result.stdout


def test_lane_g_passes_when_both_halves_pass(tmp_path: Path) -> None:
    python = _fake_gate_python(tmp_path, pytest_exit=0)
    result, argv = run_lane(
        tmp_path, "g", _TEST_OK, 0, env={"NY_GATE_PYTHON": str(python)}
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  [g]" in result.stdout
    assert argv.splitlines() == [_LANE_G_CARGO]
