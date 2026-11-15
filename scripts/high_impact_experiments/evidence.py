"""Stable byte identities and immutable external archives."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


class ExperimentError(ValueError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def fingerprint(path: Path) -> tuple[int, ...]:
    s = path.stat()
    if not stat.S_ISREG(s.st_mode):
        raise ExperimentError(f"not a regular file: {path}")
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def stable_read(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Resolve a regular file, then reject mutation during the read."""
    resolved = path.resolve(strict=True)
    before = fingerprint(resolved)
    data = resolved.read_bytes()
    if before != fingerprint(resolved) or len(data) != before[2]:
        raise ExperimentError(f"file changed while hashing: {resolved}")
    return data, {"path": str(resolved), "declared_path": str(path.absolute()), "sha256": digest(data),
                  "size_bytes": len(data), "fingerprint": list(before)}


def pin(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    before = fingerprint(resolved)
    sha = hashlib.sha256()
    size = 0
    with resolved.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            sha.update(data)
            size += len(data)
    if before != fingerprint(resolved) or size != before[2]:
        raise ExperimentError(f"file changed while hashing: {resolved}")
    return {"path": str(resolved), "declared_path": str(path.absolute()), "sha256": sha.hexdigest(),
            "size_bytes": size, "fingerprint": list(before)}


def unchanged(record: dict[str, Any]) -> None:
    observed = pin(Path(record.get("declared_path", record["path"])))
    if observed != record:
        raise ExperimentError(f"frozen file changed: {record['path']}")


def immutable(path: Path, data: bytes) -> dict[str, Any]:
    """Never replace an archive member, even a symlink or partial prior run."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return pin(path)


def immutable_json(path: Path, value: Any) -> dict[str, Any]:
    return immutable(path, encode(value))


def seal_copy(source: Path, target: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    original = pin(source)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    unchanged(original)
    sealed = pin(target)
    if sealed["sha256"] != original["sha256"]:
        raise ExperimentError("artifact changed while sealing")
    return sealed, original


def strict_json(path: Path) -> dict[str, Any]:
    def bad_constant(value: str) -> None:
        raise ExperimentError(f"nonfinite JSON value: {value}")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ExperimentError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ExperimentError("JSON receipt exceeds 32 MiB orchestration limit")
    value = json.loads(stable_read(path)[0], parse_constant=bad_constant,
                       object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ExperimentError(f"JSON object required: {path}")
    return value


def positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ExperimentError(f"{label} must be a positive finite number")
    return result


def count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExperimentError(f"{label} must be a nonnegative integer")
    return value


def git_state(repo: Path) -> dict[str, str]:
    def git(*args: str) -> bytes:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                           check=False, timeout=30)
        if p.returncode:
            raise ExperimentError(f"cannot inspect source tree: {p.stderr.decode(errors='replace')}")
        return p.stdout
    return {"head": git("rev-parse", "HEAD").decode().strip(),
            "tracked_diff_sha256": digest(git("diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--")),
            "status_sha256": digest(git("status", "--porcelain=v1", "-z"))}


def ny_build_source(repo: Path) -> dict[str, str]:
    """Mirror the current ny-cli/build.rs stamp, independently of its claim."""
    paths = ["crates", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]
    def git(*args: str) -> bytes:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                           check=False, timeout=30)
        if p.returncode:
            raise ExperimentError("cannot inspect NY build source")
        return p.stdout
    if git("ls-files", "--others", "--exclude-standard", "--", *paths):
        raise ExperimentError("untracked compiled sources cannot be bound by the NY build stamp; commit the source snapshot first")
    dirty = bool(git("status", "--porcelain=v1", "--untracked-files=no", "--", *paths))
    diff = git("diff", "--binary", "--no-ext-diff", "HEAD", "--", *paths)
    return {"dirty": "true" if dirty else "false",
            "source": digest(b"ny.build-source-diff.v1" + diff)[:16] if dirty else "clean"}


def pin_sources(paths: list[Path]) -> list[dict[str, Any]]:
    files: set[Path] = set()
    for raw in paths:
        path = raw.resolve(strict=True)
        if path.is_dir():
            # Explicit source roots only. Build/cache/benchmark roots must not be
            # selected accidentally through a full worktree directory.
            for entry in path.rglob("*"):
                if entry.is_symlink():
                    raise ExperimentError(f"source tree contains a symlink: {entry}")
                if entry.is_file() and "__pycache__" not in entry.parts:
                    files.add(entry.resolve())
        else:
            files.add(path)
    if not files:
        raise ExperimentError("source_paths must freeze at least one source file")
    return [pin(path) for path in sorted(files)]


def archive_directory(path: Path, repo: Path) -> Path:
    repo = repo.resolve(strict=True)
    path = path.absolute()
    parent = path.parent.resolve(strict=True)
    resolved = parent / path.name
    if resolved == repo or repo in resolved.parents:
        raise ExperimentError("measurement archive must be outside the source worktree")
    resolved.mkdir(mode=0o700)  # exclusive; existing directory also refuses
    return resolved


def bounded_artifact(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ExperimentError(f"unsafe artifact path: {relative}")
    resolved = path.resolve(strict=True)
    if root.resolve() not in resolved.parents:
        raise ExperimentError(f"artifact escaped run directory: {relative}")
    if path.is_symlink():
        raise ExperimentError(f"artifact may not be a symlink: {relative}")
    return resolved
