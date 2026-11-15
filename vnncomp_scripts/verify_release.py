#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed verifier for a `dist/bin/<target>/` release artifact.

`scripts/release/make_submission.sh` writes, per target triple,

    dist/bin/<target>/MANIFEST    strict key=value record (schema ny-release-manifest-v1)
    dist/bin/<target>/ny.xz       the stripped executable, xz-compressed
    dist/bin/<target>/ny.sha256   "<binary_sha256>  ny\\n"

and `install_tool.sh` calls this verifier before it runs or installs anything.
It is standard-library only and never executes the artifact: every check is a
static comparison of bytes against the manifest, of the manifest against the
source checkout (the only program it runs is `git`, read-only, to bind the
checkout), and of the executable's own ELF records against the manifest.

The checkout is bound by content, so a SHALLOW clone binds as a full one does:
HEAD's tree with the release files removed must be MANIFEST's `ny_commit_tree`,
and MANIFEST's `ny_commit_object` -- ny_commit's raw commit object -- must hash
to ny_commit (which the binary embeds) and name `ny_commit_tree`, so the tree
id is ny_commit's own without any history. When ny_commit's history is present
it must also be an ancestor of HEAD differing only in the release files. Every
`git` call names the checkout as `safe.directory`, so a clone owned by another
user is read rather than refused (docs/RELEASE.md).

The artifact this admits is an UNSEALED, DECLARED build lane: `targo
--unverified build` (built WITH the Trust toolchain, not verified BY it), either
native or cross-compiled with `-Zbuild-std`. It never carries, and this
verifier refuses, the `ny.vnncomp.build.v1|status=sealed|` record that only the
seed-mode Ubuntu 24.04 builder may embed; a sealed binary belongs to the
`dist/bin/ny-x86_64-linux.*` triplet and `verify_prebuilt.py`.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import lzma
import os
import re
import stat
import struct
import subprocess
import sys
from pathlib import Path

SCHEMA = "ny-release-manifest-v1"
FEATURES = "mip,cuda"
UNSEALED_PROVENANCE = "ny.vnncomp.unsealed.v1|status=unsealed|"
SEALED_PREFIX = b"ny.vnncomp.build.v1|status=sealed|"
MACHINES = {
    "x86_64-unknown-linux-gnu": 62,
    "aarch64-unknown-linux-gnu": 183,
}
LANES = {"cross-build-std", "native"}
MANIFEST_KEYS = (
    "schema",
    "target",
    "build_lane",
    "build_host",
    "features",
    "ny_commit",
    "ny_tree",
    "ny_commit_tree",
    "ny_commit_object",
    "release_paths",
    "cargo_lock_sha256",
    "ay_lock_commit",
    "trustc_commit",
    "trustc_version_sha256",
    "trustc_sha256",
    "targo_version_sha256",
    "std_source",
    "cross_sysroot_sha256",
    "linker_sha256",
    "onnxruntime_static_sha256",
    "binary_sha256",
    "binary_size",
    "package",
    "package_sha256",
    "build_id",
    "build_stamp",
    "vnncomp_build_provenance",
    "needed",
    "max_glibc",
    "max_glibcxx",
    "cuda_runtime",
    "cuda_fatbin_manifest_sha256",
    "cuda_fatbins",
    "self_identified_by",
    "make_submission_sha256",
)
HASH_KEYS = (
    "cargo_lock_sha256",
    "trustc_version_sha256",
    "trustc_sha256",
    "targo_version_sha256",
    "onnxruntime_static_sha256",
    "binary_sha256",
    "package_sha256",
    "cuda_fatbin_manifest_sha256",
    "make_submission_sha256",
)
COMMIT_KEYS = ("ny_commit", "ny_commit_tree", "ay_lock_commit", "trustc_commit")
RELEASE_FILES = ("MANIFEST", "ny.xz", "ny.sha256")
# Every path a release commit may add on top of ny_commit, for EVERY target (an
# operator may commit both targets' files): make_submission.sh records this set
# as `release_paths`, and submission_binary_receipt.sh carries the same list
# (release_pipeline_black_box.rs pins all three equal). Sorted, comma-joined.
RELEASE_PATHS = (
    "dist/bin/aarch64-unknown-linux-gnu/MANIFEST",
    "dist/bin/aarch64-unknown-linux-gnu/ny.sha256",
    "dist/bin/aarch64-unknown-linux-gnu/ny.xz",
    "dist/bin/x86_64-unknown-linux-gnu/MANIFEST",
    "dist/bin/x86_64-unknown-linux-gnu/ny.sha256",
    "dist/bin/x86_64-unknown-linux-gnu/ny.xz",
)
RELEASE_PATH_BYTES = frozenset(path.encode("ascii") for path in RELEASE_PATHS)
# Only a regular file at a release path is a release file; a symlink, directory
# or submodule there stays in the tree, so the tree comparison refuses it.
RELEASE_FILE_MODES = (b"100644", b"100755")
# Same package roots verify_prebuilt.py binds: untracked, non-ignored files
# under them would be inputs the source identity does not cover.
PACKAGE_ROOTS = (
    ".cargo",
    "benchmarks/download_benchmarks.sh",
    "Cargo.lock",
    "Cargo.toml",
    "LICENSE",
    "README.md",
    "_typos.toml",
    "clippy.toml",
    "configs",
    "crates",
    "install_tool.sh",
    "prepare_instance.sh",
    "requirements.txt",
    "rust-toolchain.toml",
    "run_instance.sh",
    "scripts/vnncomp_coverage.py",
    "scripts/release",
    "vnncomp_scripts",
)
FORBIDDEN_NEEDED = ("onnxruntime", "libcuda", "libcudart", "libcublas", "libnvrtc", "libvulkan")
BUILD_ID_FIELDS = (
    "version",
    "features",
    "backend",
    "markers",
    "commit",
    "dirty",
    "revision",
    "source",
    "profile",
    "target",
    "rustc",
    "stamp",
)
FATBIN_MANIFEST = Path("crates/ny-cuda/kernels/search_kernels.manifest")
MAKE_SUBMISSION = Path("scripts/release/make_submission.sh")

# MANIFEST carries ny_commit's raw commit object (base64, <= 32 KiB decoded:
# the longest of 3000 recent ny commits is 10.7 KB), so its bound is 64 KiB.
MAX_MANIFEST_BYTES = 64 * 1024
MAX_COMMIT_OBJECT_BYTES = 32 * 1024
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
MAX_BINARY_BYTES = 512 * 1024 * 1024
HEX40 = re.compile(r"[0-9a-f]{40}", re.ASCII)
HEX64 = re.compile(r"[0-9a-f]{64}", re.ASCII)
GLIBC_VERSION = re.compile(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", re.ASCII)
AY_SOURCE = re.compile(
    r'^source = "git\+https://github\.com/alabsystems/ay\.git'
    r'\?rev=([0-9a-f]{40})#([0-9a-f]{40})"$',
    re.ASCII,
)
AY_BUILD_COMMIT = re.compile(rb"build\.commit=([0-9a-f]{40})(-dirty)?", re.ASCII)


class VerificationError(RuntimeError):
    """The artifact failed one of the release-admission invariants."""


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def read_regular(path: Path, limit: int, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise VerificationError(f"cannot inspect {label} {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise VerificationError(f"{label} must be a regular file, not a symlink: {path}")
    if metadata.st_size > limit:
        raise VerificationError(f"{label} is larger than {limit} bytes: {path}")
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise VerificationError(f"cannot read {label} {path}: {error}") from error
    if len(contents) > limit:
        raise VerificationError(f"{label} grew past its size limit: {path}")
    return contents


def parse_manifest(contents: bytes) -> dict[str, str]:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError("MANIFEST is not UTF-8") from error
    if not text.endswith("\n"):
        raise VerificationError("MANIFEST must end with a newline")
    values: dict[str, str] = {}
    allowed = set(MANIFEST_KEYS)
    for number, line in enumerate(text[:-1].split("\n"), start=1):
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise VerificationError(f"MANIFEST line {number} is not key=value")
        if key not in allowed:
            raise VerificationError(f"MANIFEST has unknown key {key!r}")
        if key in values:
            raise VerificationError(f"MANIFEST has duplicate key {key!r}")
        if not value or value != value.strip() or "\r" in value:
            raise VerificationError(f"MANIFEST value for {key!r} is empty or padded")
        values[key] = value
    missing = [key for key in MANIFEST_KEYS if key not in values]
    if missing:
        raise VerificationError(f"MANIFEST is missing keys {missing}")
    return values


def require(values: dict[str, str], key: str, expected: str) -> None:
    if values[key] != expected:
        raise VerificationError(
            f"MANIFEST {key!r} mismatch: expected {expected!r}, got {values[key]!r}"
        )


def parse_identity_line(line: str, schema: str) -> dict[str, str]:
    parts = line.split(" ")
    if not parts or parts[0] != schema:
        raise VerificationError(f"identity line does not start with {schema}: {line!r}")
    fields: dict[str, str] = {}
    for part in parts[1:]:
        key, separator, value = part.partition("=")
        if not separator or not key or not value or key in fields:
            raise VerificationError(f"malformed {schema} field {part!r}")
        fields[key] = value
    return fields


def verify_identity(values: dict[str, str], target: str) -> None:
    build_id = parse_identity_line(values["build_id"], "NY_BUILD_ID_V1")
    stamp = parse_identity_line(values["build_stamp"], "NY_BUILD_STAMP_V1")
    for key in (*BUILD_ID_FIELDS, "exe", "id"):
        if key not in build_id:
            raise VerificationError(f"build_id lacks the {key}= field")
    for key in BUILD_ID_FIELDS:
        if stamp.get(key) != build_id[key]:
            raise VerificationError(
                f"build_stamp {key}={stamp.get(key)!r} disagrees with build_id {key}={build_id[key]!r}"
            )
    expected = {
        "commit": values["ny_commit"],
        "dirty": "false",
        "source": "clean",
        "features": FEATURES,
        "profile": "release",
        "target": target,
        "exe": values["binary_sha256"],
    }
    for key, want in expected.items():
        if build_id[key] != want:
            raise VerificationError(
                f"the binary's own build_id says {key}={build_id[key]!r}, the release needs {want!r}"
            )
    if values["trustc_commit"][:9] not in build_id["rustc"]:
        raise VerificationError(
            f"build_id rustc={build_id['rustc']!r} does not name trustc {values['trustc_commit'][:9]}"
        )


def exact_ay_lock_commit(lock_contents: bytes) -> str:
    try:
        lines = lock_contents.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise VerificationError("Cargo.lock is not UTF-8") from error
    commits: set[str] = set()
    for line in lines:
        if "alabsystems/ay" not in line.lower():
            continue
        match = AY_SOURCE.fullmatch(line)
        if match is None:
            raise VerificationError(f"non-canonical AY Cargo.lock source: {line}")
        requested, resolved = match.groups()
        if requested != resolved:
            raise VerificationError(f"AY revision {requested} resolved to {resolved}")
        commits.add(resolved)
    if len(commits) != 1:
        raise VerificationError(f"expected exactly one AY commit in Cargo.lock, found {sorted(commits)}")
    return commits.pop()


class Elf:
    """Just enough ELF64-LE parsing to read DT_NEEDED and the version needs."""

    def __init__(self, data: bytes) -> None:
        if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
            raise VerificationError("binary is not an ELF64 little-endian file")
        self.data = data
        self.machine = struct.unpack_from("<H", data, 18)[0]
        shoff = struct.unpack_from("<Q", data, 40)[0]
        shentsize, shnum = struct.unpack_from("<HH", data, 58)
        if shoff == 0 or shentsize != 64 or shoff + shnum * 64 > len(data):
            raise VerificationError("binary has no readable section header table")
        self.sections = []
        for index in range(shnum):
            (_name, kind, _flags, _addr, offset, size, link, _info, _align, entsize) = struct.unpack_from(
                "<IIQQQQIIQQ", data, shoff + index * 64
            )
            self.sections.append((kind, offset, size, link, entsize))

    def _string(self, section: int, offset: int) -> str:
        _kind, base, size, _link, _entsize = self.sections[section]
        if offset >= size:
            raise VerificationError("ELF string offset out of range")
        end = self.data.index(b"\0", base + offset)
        return self.data[base + offset : end].decode("ascii", errors="strict")

    def needed(self) -> list[str]:
        out: list[str] = []
        for kind, offset, size, link, _entsize in self.sections:
            if kind != 6:  # SHT_DYNAMIC
                continue
            for cursor in range(offset, offset + size, 16):
                tag, value = struct.unpack_from("<qQ", self.data, cursor)
                if tag == 0:
                    break
                if tag == 1:  # DT_NEEDED
                    out.append(self._string(link, value))
        return out

    def version_needs(self) -> list[str]:
        out: list[str] = []
        for kind, offset, size, link, _entsize in self.sections:
            if kind != 0x6FFFFFFE:  # SHT_GNU_verneed
                continue
            cursor = offset
            while cursor < offset + size:
                _version, count, _file, aux, following = struct.unpack_from("<HHIII", self.data, cursor)
                aux_cursor = cursor + aux
                for _ in range(count):
                    _hash, _flags, _other, name, aux_next = struct.unpack_from(
                        "<IHHII", self.data, aux_cursor
                    )
                    out.append(self._string(link, name))
                    if aux_next == 0:
                        break
                    aux_cursor += aux_next
                if following == 0:
                    break
                cursor += following
        return out


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def max_symbol_version(names: list[str], prefix: str) -> str:
    versions = [name[len(prefix) :] for name in names if name.startswith(prefix)]
    versions = [version for version in versions if GLIBC_VERSION.fullmatch(version)]
    if not versions:
        return "none"
    return max(versions, key=version_key)


def verify_binary(binary: bytes, values: dict[str, str], target: str, root: Path) -> None:
    if sha256_bytes(binary) != values["binary_sha256"]:
        raise VerificationError("decompressed binary does not match binary_sha256")
    if str(len(binary)) != values["binary_size"]:
        raise VerificationError("decompressed binary does not match binary_size")
    elf = Elf(binary)
    if elf.machine != MACHINES[target]:
        raise VerificationError(
            f"binary e_machine={elf.machine} is not {target} (e_machine={MACHINES[target]})"
        )
    needed = elf.needed()
    if not needed:
        raise VerificationError("binary has no DT_NEEDED entries; refusing an unreadable link record")
    require(values, "needed", ",".join(needed))
    for library in needed:
        if any(token in library for token in FORBIDDEN_NEEDED):
            raise VerificationError(
                f"binary links {library} at load time; CUDA/Vulkan are dlopen'd and ONNX Runtime is static"
            )
    requirements = elf.version_needs()
    require(values, "max_glibc", max_symbol_version(requirements, "GLIBC_"))
    require(values, "max_glibcxx", max_symbol_version(requirements, "GLIBCXX_"))

    if SEALED_PREFIX in binary:
        raise VerificationError(
            "binary embeds a SEALED ny.vnncomp.build.v1 record; sealed builds ship only as the "
            "dist/bin/ny-x86_64-linux.* triplet (verify_prebuilt.py), never as a declared lane"
        )
    if UNSEALED_PROVENANCE.encode("ascii") not in binary:
        raise VerificationError("binary does not embed the unsealed build-provenance record")
    if values["ny_commit"].encode("ascii") not in binary:
        raise VerificationError("binary does not embed its ny_commit")
    commits = {
        match.group(1).decode("ascii") + ("-dirty" if match.group(2) else "")
        for match in AY_BUILD_COMMIT.finditer(binary)
    }
    if commits != {values["ay_lock_commit"]}:
        raise VerificationError(
            f"embedded AY build.commit {sorted(commits)} is not exactly {values['ay_lock_commit']}"
        )
    verify_fatbins(binary, values, root)


def verify_fatbins(binary: bytes, values: dict[str, str], root: Path) -> None:
    require(values, "cuda_runtime", "dlopen")
    manifest_bytes = read_regular(root / FATBIN_MANIFEST, 64 * 1024, "CUDA kernel manifest")
    require(values, "cuda_fatbin_manifest_sha256", sha256_bytes(manifest_bytes))
    fields: dict[str, str] = {}
    for line in manifest_bytes.decode("utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            fields[key] = value
    images: list[tuple[str, str]] = []
    for image_key, sha_key in (("image", "image_sha256"), ("wave_image", "wave_image_sha256")):
        if image_key not in fields or sha_key not in fields:
            raise VerificationError(f"CUDA kernel manifest lacks {image_key}/{sha_key}")
        images.append((fields[image_key], fields[sha_key]))
    recorded = ",".join(f"{name}:{digest}" for name, digest in images)
    require(values, "cuda_fatbins", recorded)
    for name, digest in images:
        if "/" in name or name.startswith("."):
            raise VerificationError(f"CUDA kernel image name is not a plain file name: {name!r}")
        image = read_regular(root / FATBIN_MANIFEST.parent / name, 64 * 1024 * 1024, "CUDA fatbin")
        if sha256_bytes(image) != digest:
            raise VerificationError(f"committed {name} does not match the kernel manifest")
        if binary.find(image) < 0:
            raise VerificationError(f"binary does not embed the committed {name} byte-for-byte")


def git_command(root: Path, *arguments: str) -> list[str]:
    """`git` on the checkout ROOT (already resolved): read-only (no optional
    index refresh), and with ROOT -- exactly ROOT, never `*` -- as
    safe.directory. A clone owned by another user than the installer is then
    read instead of refused as "dubious ownership"; every comparison below is
    the one the owner would run. git matches the entry against the WORKTREE,
    so a linked worktree's gitdir (a gitfile target) is opened under it too.

    This trusts nothing new, but it does make the clone's own `.git` live for
    these calls -- its config (a `core.fsmonitor` hook there runs as the
    caller), index flags, refs/replace, `shallow` file. Whoever can write that
    `.git` can already edit this verifier, install_tool.sh and run_instance.sh,
    which the same caller runs from the same clone; the binding is of committed
    content under git's own clone state, never a defence against the clone's
    owner (docs/RELEASE.md)."""
    return ["git", "-c", f"safe.directory={os.fspath(root)}", "--no-optional-locks", "-C", os.fspath(root), *arguments]


def git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(git_command(root, *arguments), capture_output=True, check=False)
    except OSError as error:
        raise VerificationError(f"cannot run git: {error}") from error
    if check and result.returncode != 0:
        raise VerificationError(
            f"git {' '.join(arguments)} failed ({result.returncode}): "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result


def tree_without_release_paths(root: Path, tree: str, prefix: bytes = b"") -> str | None:
    """The id of TREE with every RELEASE_PATHS file removed; None if nothing is
    left (git never stores an empty directory, so an emptied one goes too).

    Computable in a shallow clone: it reads only HEAD's own trees (`ls-tree`)
    and hashes the rebuilt ones in memory -- nothing is written to the
    repository. Unchanged trees keep their ids; a changed tree is re-serialized
    from its `ls-tree` entries in their stored order. submission_binary_receipt.sh
    receipt_tree_without_release_paths() is the same computation in bash."""
    listing = git(root, "ls-tree", "--full-tree", "-z", tree).stdout
    body = bytearray()
    changed = False
    for record in listing.split(b"\0"):
        if not record:
            continue
        meta, tab, name = record.partition(b"\t")
        fields = meta.split(b" ")
        if not tab or not name or len(fields) != 3 or HEX40.fullmatch(fields[2].decode("ascii", "replace")) is None:
            raise VerificationError(f"unexpected git ls-tree record {record!r}")
        mode, kind, oid = fields[0], fields[1], fields[2].decode("ascii")
        path = prefix + name
        if path in RELEASE_PATH_BYTES and kind == b"blob" and mode in RELEASE_FILE_MODES:
            changed = True
            continue
        if kind == b"tree" and any(release.startswith(path + b"/") for release in RELEASE_PATH_BYTES):
            stripped = tree_without_release_paths(root, oid, path + b"/")
            if stripped != oid:
                changed = True
            if stripped is None:
                continue
            oid = stripped
        body += (mode[1:] if mode.startswith(b"0") else mode) + b" " + name + b"\0" + bytes.fromhex(oid)
    if not changed:
        return tree
    if not body:
        return None
    return hashlib.sha1(b"tree %d\0" % len(body) + bytes(body)).hexdigest()


def verify_commit_object(values: dict[str, str]) -> None:
    """MANIFEST `ny_commit_object` is ny_commit's raw commit object, base64:
    its git SHA-1 must be ny_commit -- the commit the binary embeds -- and its
    first header must name `ny_commit_tree`. That binds the tree id to ny_commit
    without any history, so in a shallow clone (no ny_commit object) a MANIFEST
    whose `ny_commit_tree` was re-pointed at an edited source tree is refused
    exactly as a full clone refuses it."""
    encoded = values["ny_commit_object"]
    try:
        # validate=True: any character outside the base64 alphabet refuses.
        body = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise VerificationError(f"MANIFEST 'ny_commit_object' is not base64: {error}") from error
    if base64.b64encode(body).decode("ascii") != encoded:
        raise VerificationError("MANIFEST 'ny_commit_object' is not canonical base64")
    if len(body) > MAX_COMMIT_OBJECT_BYTES:
        raise VerificationError(f"MANIFEST 'ny_commit_object' is larger than {MAX_COMMIT_OBJECT_BYTES} bytes")
    oid = hashlib.sha1(b"commit %d\0" % len(body) + body).hexdigest()
    if oid != values["ny_commit"]:
        raise VerificationError(
            f"MANIFEST ny_commit_object hashes to {oid}, not ny_commit {values['ny_commit']}"
        )
    header = body.split(b"\n", 1)[0]
    if header != b"tree " + values["ny_commit_tree"].encode("ascii"):
        raise VerificationError(
            f"ny_commit {values['ny_commit']} names {header.decode('utf-8', errors='replace')!r}, "
            f"not MANIFEST ny_commit_tree {values['ny_commit_tree']}"
        )


def verify_source_binding(root: Path, ny_commit: str, ny_commit_tree: str) -> str:
    """Bind the artifact to the checkout. Returns a short description."""
    if not (root / ".git").exists():
        marker = root / ".ny-vnncomp-source.txt"
        if not marker.exists():
            raise VerificationError(
                "no Git checkout and no .ny-vnncomp-source.txt marker: the source identity is unknown"
            )
        lines = read_regular(marker, 4096, "source marker").decode("utf-8").splitlines()
        if len(lines) != 3 or lines[0] != "schema=ny-vnncomp-source-v1":
            raise VerificationError("malformed .ny-vnncomp-source.txt")
        if lines[1] != f"ny_commit={ny_commit}":
            raise VerificationError(f"source marker names {lines[1]!r}, the artifact ny_commit={ny_commit}")
        return "archive-marker"

    top = git(root, "rev-parse", "--show-toplevel").stdout.decode("utf-8").strip()
    if Path(top).resolve() != root:
        raise VerificationError(f"installer root {root} is not the Git worktree root {top}")
    dirty = git(root, "diff", "--quiet", "HEAD", "--", check=False)
    if dirty.returncode == 1:
        raise VerificationError("tracked files are modified relative to HEAD")
    if dirty.returncode != 0:
        raise VerificationError(f"git diff failed while checking the source state ({dirty.returncode})")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", *PACKAGE_ROOTS).stdout
    stray = [item.decode("utf-8", errors="replace") for item in untracked.split(b"\0") if item]
    if stray:
        raise VerificationError(f"untracked files under package roots are not covered by ny_commit: {stray}")
    head = git(root, "rev-parse", "--verify", "HEAD^{commit}").stdout.decode("ascii").strip()
    present = git(root, "cat-file", "-e", f"{ny_commit}^{{commit}}", check=False).returncode == 0
    if present:
        commit_tree = git(root, "rev-parse", "--verify", f"{ny_commit}^{{tree}}").stdout.decode("ascii").strip()
        if commit_tree != ny_commit_tree:
            raise VerificationError(
                f"MANIFEST ny_commit_tree {ny_commit_tree} is not the tree of ny_commit {ny_commit} ({commit_tree})"
            )
    if head == ny_commit:
        binding = "head"
    elif present and git(root, "merge-base", "--is-ancestor", ny_commit, head, check=False).returncode == 0:
        changed = [
            item.decode("utf-8", errors="replace")
            for item in git(root, "diff", "--name-only", "-z", ny_commit, head, "--").stdout.split(b"\0")
            if item
        ]
        outside = [path for path in changed if path not in RELEASE_PATHS]
        if outside:
            raise VerificationError(
                f"HEAD {head} differs from ny_commit {ny_commit} outside dist/bin/<target>/ release files: {outside}"
            )
        binding = "artifact-only-descendant"
    elif git(root, "rev-parse", "--is-shallow-repository").stdout.strip() == b"true":
        # A shallow clone (`git clone --depth 1` of the release commit) cuts the
        # history above HEAD, so ancestry is unknowable; the tree check below
        # binds it by content instead, and verify_commit_object() proves the
        # recorded tree is ny_commit's own.
        binding = "artifact-only-tree"
    elif not present:
        raise VerificationError(f"ny_commit {ny_commit} is not in this checkout (HEAD {head})")
    else:
        raise VerificationError(f"ny_commit {ny_commit} is not an ancestor of HEAD {head}")
    # Every binding, with or without history: HEAD's tree with the release
    # files removed IS ny_commit's tree. In a full clone this agrees with the
    # ancestry/diff rule above; in a shallow clone it is the whole binding.
    head_tree = git(root, "rev-parse", "--verify", "HEAD^{tree}").stdout.decode("ascii").strip()
    source_tree = tree_without_release_paths(root, head_tree)
    if source_tree != ny_commit_tree:
        raise VerificationError(
            f"HEAD {head}'s tree without the release files is {source_tree or 'empty'}, not MANIFEST "
            f"ny_commit_tree {ny_commit_tree}: the checkout is not ny_commit's source plus release files"
        )
    return binding


def decompress_bounded(package: bytes, output: Path) -> bytes:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(output, flags, 0o600)
    except OSError as error:
        raise VerificationError(f"cannot exclusively create {output}: {error}") from error
    chunks: list[bytes] = []
    total = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            with lzma.LZMAFile(io.BytesIO(package), "rb") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_BINARY_BYTES:
                        raise VerificationError(f"decompressed binary exceeds {MAX_BINARY_BYTES} bytes")
                    destination.write(chunk)
                    chunks.append(chunk)
            destination.flush()
            os.fsync(destination.fileno())
    except VerificationError:
        output.unlink(missing_ok=True)
        raise
    except (lzma.LZMAError, EOFError, OSError) as error:
        output.unlink(missing_ok=True)
        raise VerificationError(f"cannot decompress ny.xz: {error}") from error
    return b"".join(chunks)


def verify(root: Path, target: str, output: Path) -> dict[str, str]:
    if target not in MACHINES:
        raise VerificationError(f"unsupported target {target!r}; supported: {sorted(MACHINES)}")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise VerificationError(f"cannot resolve the repository root: {error}") from error
    for relative in ("dist", "dist/bin", f"dist/bin/{target}"):
        try:
            metadata = (root / relative).lstat()
        except OSError as error:
            raise VerificationError(f"cannot inspect {relative}: {error}") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise VerificationError(f"{relative} must be a real directory, not a symlink")
    artifact = root / "dist" / "bin" / target

    values = parse_manifest(read_regular(artifact / "MANIFEST", MAX_MANIFEST_BYTES, "MANIFEST"))
    require(values, "schema", SCHEMA)
    require(values, "target", target)
    require(values, "features", FEATURES)
    require(values, "ny_tree", "clean")
    require(values, "release_paths", ",".join(RELEASE_PATHS))
    require(values, "package", "ny.xz")
    require(values, "vnncomp_build_provenance", UNSEALED_PROVENANCE)
    for key in HASH_KEYS:
        if HEX64.fullmatch(values[key]) is None:
            raise VerificationError(f"MANIFEST {key!r} is not 64 lowercase hex")
    for key in COMMIT_KEYS:
        if HEX40.fullmatch(values[key]) is None:
            raise VerificationError(f"MANIFEST {key!r} is not 40 lowercase hex")
    if not values["binary_size"].isdigit():
        raise VerificationError("MANIFEST binary_size is not a decimal integer")
    if values["build_lane"] not in LANES:
        raise VerificationError(f"MANIFEST build_lane {values['build_lane']!r} is not one of {sorted(LANES)}")
    if values["build_host"] not in MACHINES:
        raise VerificationError(f"MANIFEST build_host {values['build_host']!r} is not a supported triple")
    if values["build_lane"] == "native":
        if values["build_host"] != target:
            raise VerificationError("a native build_lane must have build_host == target")
        for key, want in (("std_source", "prebuilt"), ("cross_sysroot_sha256", "none")):
            require(values, key, want)
        expected_identifier = "native"
    else:
        if values["build_host"] == target:
            raise VerificationError("a cross build_lane must have build_host != target")
        std = values["std_source"]
        if not std.startswith("build-std:") or HEX64.fullmatch(std[len("build-std:") :]) is None:
            raise VerificationError("a cross build_lane must record std_source=build-std:<sha256>")
        for key in ("cross_sysroot_sha256", "linker_sha256"):
            if HEX64.fullmatch(values[key]) is None:
                raise VerificationError(f"a cross build_lane must record {key}")
        expected_identifier = "qemu-user"
    if not values["self_identified_by"].startswith(expected_identifier):
        raise VerificationError(
            f"self_identified_by={values['self_identified_by']!r} is not a {expected_identifier} run"
        )
    verify_identity(values, target)

    sidecar = read_regular(artifact / "ny.sha256", 1024, "ny.sha256")
    if sidecar != f"{values['binary_sha256']}  ny\n".encode("ascii"):
        raise VerificationError("ny.sha256 is not the exact sha256sum record of the binary")
    package = read_regular(artifact / "ny.xz", MAX_PACKAGE_BYTES, "ny.xz")
    if sha256_bytes(package) != values["package_sha256"]:
        raise VerificationError("ny.xz does not match package_sha256")

    lock = read_regular(root / "Cargo.lock", 64 * 1024 * 1024, "Cargo.lock")
    require(values, "cargo_lock_sha256", sha256_bytes(lock))
    require(values, "ay_lock_commit", exact_ay_lock_commit(lock))
    builder = read_regular(root / MAKE_SUBMISSION, 1024 * 1024, "make_submission.sh")
    require(values, "make_submission_sha256", sha256_bytes(builder))
    binding = verify_source_binding(root, values["ny_commit"], values["ny_commit_tree"])
    verify_commit_object(values)

    binary = decompress_bounded(package, output)
    try:
        verify_binary(binary, values, target, root)
        raw = artifact / "ny"
        if raw.exists() and sha256_bytes(read_regular(raw, MAX_BINARY_BYTES, "dist ny")) != values["binary_sha256"]:
            raise VerificationError("dist/bin/<target>/ny is present and differs from binary_sha256")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    values["_binding"] = binding
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        values = verify(arguments.repo_root, arguments.target, arguments.output)
    except VerificationError as error:
        print(f"ERROR: release artifact verification failed: {error}", file=sys.stderr)
        return 1
    print(
        "NY_RELEASE_VERIFY_OK "
        f"target={values['target']} lane={values['build_lane']} ny_commit={values['ny_commit']} "
        f"binary_sha256={values['binary_sha256']} max_glibc={values['max_glibc']} "
        f"source_binding={values['_binding']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
