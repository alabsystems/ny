# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed coherence checks for NY's first-party dependency policy."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
AY_URL = "https://github.com/alabsystems/ay.git"
CLEAN_URL = "https://github.com/alabsystems/clean.git"
FORBIDDEN_INTERNAL_COPY_ROOTS = (
    Path("vendor/ay"),
    Path("vendor/ny"),
    Path("vendor/trust"),
    Path("vendor/clean"),
    # These generic-path helpers were byte-identical, unused AY source copies.
    Path("vendor/build_support"),
    Path("crates/ay"),
    Path("crates/ny"),
    Path("crates/trust"),
    Path("crates/clean"),
    Path("crates/trust-spec"),
    # The former in-tree Clean corpus; Clean now resolves through Lake.
    Path("crates/ny-cert/proofs/lean/Crownproof"),
)
UNTRACKED_DEPENDENCY_CACHE_ROOTS = (
    # Proper Lake dependency checkout: allowed as ignored machine state, never
    # as tracked or submission-packaged source.
    Path("crates/ny-cert/proofs/lean/.lake/packages/crownproof"),
)
FORBIDDEN_TRACKED_INTERNAL_COPY_ROOTS = (
    *FORBIDDEN_INTERNAL_COPY_ROOTS,
    *UNTRACKED_DEPENDENCY_CACHE_ROOTS,
)
LEGACY_TRUST_COPY_PACKAGES = {"ny-trust-spec", "trust-spec"}


def test_internal_repositories_are_not_copied_into_ny() -> None:
    """Permit NY-owned ports and third-party vendor code, never repo mirrors."""
    def has_source_content(relative: Path) -> bool:
        path = REPO_ROOT / relative
        if path.is_file() or path.is_symlink():
            return True
        return path.is_dir() and any(
            child.is_file() or child.is_symlink() for child in path.rglob("*")
        )

    existing = [
        str(relative)
        for relative in FORBIDDEN_INTERNAL_COPY_ROOTS
        if has_source_content(relative)
    ]
    assert not existing, f"internal repository source copies remain: {existing!r}"

    tracked = {
        Path(path)
        for path in subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if (REPO_ROOT / path).exists() or (REPO_ROOT / path).is_symlink()
    }
    tracked_internal = [
        str(path)
        for path in tracked
        if any(
            path == root or root in path.parents
            for root in FORBIDDEN_TRACKED_INTERNAL_COPY_ROOTS
        )
    ]
    assert not tracked_internal, (
        "tracked internal repository source copies remain: "
        f"{tracked_internal!r}"
    )

    workspace = tomllib.loads(
        (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
    )["workspace"]
    members = {Path(member) for member in workspace["members"]}
    assert Path("crates/trust-spec") not in members

    lock_data = tomllib.loads(
        (REPO_ROOT / "Cargo.lock").read_text(encoding="utf-8")
    )
    copied_packages = [
        package
        for package in lock_data["package"]
        if package.get("name") in LEGACY_TRUST_COPY_PACKAGES
        and "source" not in package
    ]
    assert not copied_packages, (
        "legacy in-tree Trust shim packages remain in Cargo.lock: "
        f"{copied_packages!r}"
    )


def test_internal_copy_roots_do_not_block_ny_or_external_vendor_content() -> None:
    """Keep the denylist narrow while covering known first-party copy roots."""
    denied = tuple(
        root / "source.rs" for root in FORBIDDEN_TRACKED_INTERNAL_COPY_ROOTS
    )
    for relative in denied:
        assert any(
            relative == root or root in relative.parents
            for root in FORBIDDEN_TRACKED_INTERNAL_COPY_ROOTS
        ), f"known internal copy root escaped: {relative}"

    allowed = (
        Path("vendor/serde/build_support/generated.rs"),
        Path("vendor/clean-room-parser/src/lib.rs"),
        Path("crates/ny-contracts/src/lib.rs"),
        Path("crates/ny-mip/corpus/hard-six/instance.smt2.zst"),
        Path("crates/ny-cert/proofs/lean/NyProof/NYOwned.lean"),
    )
    for relative in allowed:
        assert not any(
            relative == root or root in relative.parents
            for root in FORBIDDEN_TRACKED_INTERNAL_COPY_ROOTS
        ), f"legitimate NY/external path was overblocked: {relative}"


def _canonical_ay_packages(lock_data):
    ay_packages = [
        package
        for package in lock_data["package"]
        if package.get("name") == "ay"
        or str(package.get("name", "")).startswith("ay-")
        or "alabsystems/ay" in str(package.get("source", "")).lower()
    ]
    assert ay_packages, "Cargo.lock must contain at least one pinned AY package"
    canonical_source = re.compile(
        rf"^git\+{re.escape(AY_URL)}\?rev=([0-9a-f]{{40}})"
        rf"#([0-9a-f]{{40}})$"
    )
    matches = [
        canonical_source.fullmatch(str(package.get("source", "")))
        for package in ay_packages
    ]
    assert all(match is not None for match in matches), (
        f"non-canonical AY Cargo.lock source: {ay_packages!r}"
    )
    lock_pairs = {match.groups() for match in matches if match is not None}
    assert len(lock_pairs) == 1, f"incoherent AY Cargo.lock pins: {lock_pairs!r}"
    requested, resolved = next(iter(lock_pairs))
    assert requested == resolved
    return resolved, ay_packages


def test_no_suffix_ay_source_cannot_hide_beside_the_canonical_pin() -> None:
    commit = "a" * 40
    lock_data = tomllib.loads(
        f"""\
version = 4

[[package]]
name = "ay-canonical"
version = "0.11.0"
source = "git+{AY_URL}?rev={commit}#{commit}"

[[package]]
name = "ay-hidden"
version = "0.11.0"
source = "git+https://github.com/alabsystems/ay?branch=main#{commit}"
"""
    )
    with pytest.raises(AssertionError, match="non-canonical AY Cargo.lock source"):
        _canonical_ay_packages(lock_data)


def test_path_sourced_ay_package_cannot_hide_beside_the_canonical_pin() -> None:
    commit = "a" * 40
    lock_data = tomllib.loads(
        f"""\
version = 4

[[package]]
name = "ay-canonical"
version = "0.11.0"
source = "git+{AY_URL}?rev={commit}#{commit}"

[[package]]
name = "ay-hidden-path-package"
version = "0.11.0"
"""
    )
    with pytest.raises(AssertionError, match="non-canonical AY Cargo.lock source"):
        _canonical_ay_packages(lock_data)


def test_git_sourced_ay_revision_is_coherent_and_not_vendor_replaced() -> None:
    assert not (REPO_ROOT / "vendor" / "ay").exists()

    lock = (REPO_ROOT / "Cargo.lock").read_text(encoding="utf-8")
    resolved, _ay_packages = _canonical_ay_packages(tomllib.loads(lock))

    ny_mip = tomllib.loads(
        (REPO_ROOT / "crates/ny-mip/Cargo.toml").read_text(encoding="utf-8")
    )
    dependency_pin = ny_mip["dependencies"]["ay-milp"]
    assert dependency_pin["git"] == AY_URL
    assert dependency_pin["rev"] == resolved

    cargo_config = tomllib.loads(
        (REPO_ROOT / ".cargo/config.toml").read_text(encoding="utf-8")
    )
    source_config = cargo_config.get("source", {})
    ay_source_entries = [
        (key, value)
        for key, value in source_config.items()
        if "alabsystems/ay" in key.lower()
        or (
            isinstance(value, dict)
            and "alabsystems/ay" in str(value.get("git", "")).lower()
        )
    ]
    assert not ay_source_entries, (
        "AY must remain the canonical Git dependency, not a Cargo "
        f"source replacement: {ay_source_entries!r}"
    )
    assert all(
        not (
            isinstance(value, dict)
            and str(value.get("directory", "")).rstrip("/") == "vendor/ay"
        )
        for value in source_config.values()
    ), "internal AY must not be redirected to vendor/ay"

    forced_env = cargo_config.get("env", {})
    assert "AY_SOURCE_GIT_COMMIT" not in forced_env
    assert "AY_SOURCE_GIT_DIRTY" not in forced_env


def test_internal_scored_audit_names_the_canonical_ay_revision() -> None:
    """Keep the private claims audit coherent without breaking public clones."""
    scored_audit_path = REPO_ROOT / "docs/SCORED_REPRO_AUDIT_2026-07-19.md"
    if not scored_audit_path.is_file():
        pytest.skip("the claims-of-record audit is intentionally not published")

    lock = (REPO_ROOT / "Cargo.lock").read_text(encoding="utf-8")
    resolved, _ay_packages = _canonical_ay_packages(tomllib.loads(lock))

    scored_audit = scored_audit_path.read_text(encoding="utf-8")
    full_audit_pins = re.findall(
        r"revision-pinned\s+to AY at\s+`([0-9a-f]{40})`", scored_audit
    )
    assert full_audit_pins == [resolved], (
        "the active scored-path audit must name the exact Git-sourced AY revision"
    )
    short_audit_pins = re.findall(
        r"(?:revision-pinned to|Exact Git-pinned) AY `([0-9a-f]{8})`",
        scored_audit,
    )
    assert short_audit_pins
    assert set(short_audit_pins) == {resolved[:8]}, (
        "the active scored-path audit contains a stale abbreviated AY revision"
    )
    assert "vendor/ay" not in scored_audit


def test_branch_hint_canary_names_the_canonical_ay_revision() -> None:
    """A dependency bump must not leave the active branch-hint claim stale."""
    canary_path = REPO_ROOT / "docs/AY_BRANCH_HINT_CANARY.md"
    if not canary_path.is_file():
        pytest.skip("the branch-hint canary is intentionally not published")

    lock = (REPO_ROOT / "Cargo.lock").read_text(encoding="utf-8")
    resolved, _ay_packages = _canonical_ay_packages(tomllib.loads(lock))
    canary = canary_path.read_text(encoding="utf-8")
    pins = re.findall(
        r"canonical revision-pinned\s+`ay-milp` Git dependency at\s+"
        r"`([0-9a-f]{40})`",
        canary,
    )
    assert pins == [resolved], (
        "the active AY branch-hint canary must name the canonical AY revision"
    )


def test_clean_is_an_exact_lake_subrepo_with_publication_mapping() -> None:
    lean_root = REPO_ROOT / "crates/ny-cert/proofs/lean"
    lakefile = tomllib.loads((lean_root / "lakefile.toml").read_text(encoding="utf-8"))
    crownproof = [
        dependency
        for dependency in lakefile["require"]
        if dependency.get("name") == "crownproof"
    ]
    assert len(crownproof) == 1
    dependency = crownproof[0]
    assert dependency["git"] == CLEAN_URL
    assert re.fullmatch(r"[0-9a-f]{40}", dependency["rev"])
    assert dependency["subDir"] == "crown-proofs/lean"

    manifest = json.loads((lean_root / "lake-manifest.json").read_text(encoding="utf-8"))
    packages = [
        package
        for package in manifest["packages"]
        if package.get("name") == "crownproof"
    ]
    assert len(packages) == 1
    package = packages[0]
    assert package["type"] == "git"
    assert package["url"] == CLEAN_URL
    assert package["rev"] == dependency["rev"]
    assert package["inputRev"] == dependency["rev"]
    assert package["subDir"] == "crown-proofs/lean"
    assert package["inherited"] is False

    clean_rev = dependency["rev"]
    cite_source = (REPO_ROOT / "crates/ny-cert/src/cite_check.rs").read_text(
        encoding="utf-8"
    )
    cite_rev = re.findall(
        r'pub const CLEAN_GIT_REV: &str = "([0-9a-f]{40})";', cite_source
    )
    assert cite_rev == [clean_rev], (
        "ny-cert's citation resolver must use the exact declared Clean revision"
    )

    cite_map = json.loads(
        (REPO_ROOT / "crates/ny-cert/proofs/cite-map.json").read_text(
            encoding="utf-8"
        )
    )
    assert cite_map["corpus"]["rev"] == clean_rev

    documented_pin_fragments = {
        Path("crates/ny-cert/SPEC.md"): f"uses private Clean commit `{clean_rev}`",
        Path("crates/ny-cert/proofs/README.md"): f"`{clean_rev}`. NY does not copy",
        Path("crates/ny-cert/proofs/lean/PROVENANCE.md"): f"revision:   {clean_rev}",
    }
    for relative, expected in documented_pin_fragments.items():
        assert expected in (REPO_ROOT / relative).read_text(encoding="utf-8"), (
            f"{relative} must document the exact declared Clean revision"
        )

    transform = (REPO_ROOT / "publish/transforms.sh").read_text(encoding="utf-8")
    for required in (
        'ledger["mappings"]["clean"][clean_dev_rev]',
        "clean-mapping.json",
        "https://github.com/alabsystems/clean.git",
        "lake update",
        "checked-out public Clean HEAD",
    ):
        assert required in transform


def test_clean_cross_repo_checks_do_not_copy_internal_source() -> None:
    scripts = REPO_ROOT / "crates/ny-cert/scripts"
    checked = (
        scripts / "_clean_pinned.sh",
        scripts / "roundtrip_with_clean.sh",
        scripts / "clean_differential.sh",
        scripts / "clean_sbar.sh",
        scripts / "certify_cersyve_v2.sh",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in checked)
    assert "crates/clean-elab/src" not in combined
    assert 'cp "$CERT_SRC' not in combined
    assert "clean-extcert-verify" not in combined
    assert "_clean_pinned.sh" in combined
    assert "lake-manifest.json" in combined
