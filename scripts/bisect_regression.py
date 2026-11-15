# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Automated first-bad-commit search for VNN-COMP row regressions.

Why this exists
---------------
This campaign has hand-bisected the same class of regression at least six times
(``1ede1d30`` x3, ``25dee0c5``, ``44d9e46d``, ``38a2fecf``, ``819b0554``), each
time re-deriving the same workflow by hand: create a worktree at a SHA, symlink
the benchmark tree, build with an isolated ``CARGO_TARGET_DIR``, run one row
through ``ny vnncomp``, compare against the banked verdict, halve the interval.
It is mechanical, slow (a release build is tens of minutes), and it has already
produced at least one wrong answer from a *stale binary* -- a binary that
existed and looked new enough, but did not contain the commit under test.

A stale binary does not merely waste a probe.  It silently inverts a bisect: the
harness attributes the previous SHA's behaviour to the SHA under test, the
interval is halved in the wrong direction, and the search converges confidently
on an innocent commit.  Every other guarantee here is worthless without a build
attestation, so that is the part this module treats as load-bearing.

The build attestation
---------------------
Each candidate SHA gets a deterministic *build token* derived from the SHA.  The
final link is performed through ``targo --unverified rustc -- -C link-arg=-Wl,--build-id=0x<token>``,
which stamps the token into the produced ELF's ``NT_GNU_BUILD_ID`` note.  The
token therefore travels *inside the artifact*, not alongside it.  A binary is
accepted for a probe only when all of the following hold:

1. the worktree's ``HEAD`` resolved to the requested SHA and carried no
   modifications to tracked files, checked immediately before the build;
2. the build subprocess **exited**, with status 0 (never inferred from the
   presence of a file);
3. the binary's mtime advanced past a stamp file created after the source
   attestation and before the build started;
4. the ELF's embedded ``NT_GNU_BUILD_ID`` equals the token for *this* SHA;
5. the binary's SHA-256, recorded at build time, still matches at run time.

Gate 4 is the one that cannot be satisfied by an older artifact: the token is a
function of the commit, so a binary built from any other commit carries a
different note.  Gates 1-3 and 5 are cheap and catch the adjacent failures
(dirty tree, crashed build, later overwrite).

Because the token is part of the top crate's rustc invocation, cargo's own
fingerprint forces a relink whenever the SHA changes.  That makes a *shared*
``CARGO_TARGET_DIR`` safe as well as fast -- dependency artifacts are reused
across probes (a release build of this workspace is dominated by the fat-LTO
link of ``ny-cli``, not by third-party crates), while the artifact under test is
always re-linked and always re-attested.  ``--target-mode per-sha`` restores
fully isolated target directories for anyone who wants belt and braces.

Search semantics
----------------
The search is ``git bisect``'s, driven through ``git rev-list --bisect-vars``,
so it walks the *full commit DAG* rather than the first-parent spine.  That
matters here: ``1ede1d30``, the known answer for relusplitter row 34, is not on
main's first-parent spine, and a first-parent-only search cannot name it.

Several rows are bisected in one campaign.  A release build costs tens of
minutes and a probe costs a few, so every unresolved row is evaluated at every
probe SHA; each row then folds that observation into its own interval if the SHA
lies inside it.  Rows whose regressions share a commit converge together for the
price of one search.

Probe classification
--------------------
``GOOD`` means the row still reaches its banked verdict, ``BAD`` means it does
not, ``SKIP`` means the probe could not answer (build failure, missing preset
line, watchdog SIGKILL with no sealed verdict).  Per the campaign's measurement
hygiene, *a single failed re-run is not a dead row*: a probe is BAD only after
``--repeats`` consecutive runs all fail to reach the expected verdict, and any
single expected verdict makes it GOOD immediately.  A run that produces the
*opposite* decided verdict is not a regression at all -- it is a soundness
event, and the campaign aborts on it.

``--control N:verdict`` names rows that are *known to survive* across the whole
interval.  They run first at every probe SHA, and a control failure makes the
probe SKIP instead of charging it to the commit.

Controls alone are not enough, and this harness has the measurement to prove it.
``relusplitter`` rows 24 and 40 were probed at ``e4b647ff`` at loadavg 32-53 and
returned 3/3 timeouts at 95.4 s.  Three hours later the *identical attested
binary* at the *identical SHA* decided both on the first attempt -- ``unsat`` in
42.9 s and 39.9 s at loadavg ~11.  Both controls passed on every one of those
probes.  They could not have done otherwise: the controls decide in ~10 s of a
90 s budget (9x margin) while rows 24/40 need ~40 s (2.2x margin), so a 2x
slowdown kills the rows and leaves the controls untouched.  A control only
detects the contention that matters when its margin resembles the margin of the
rows under test.

So the expensive verdict is not accepted from a loaded box at all.  Before a
probe may be recorded BAD it must reproduce once with the 1-minute load average
at or below ``--bad-confirm-load``; the harness holds for that window up to
``--bad-confirm-wait``.  If the quiet run decides, the probe is GOOD and the
earlier failures are recorded as contention.  If the window never arrives the
probe is SKIP -- "I could not tell" is the honest answer, and it must not move
an interval.

Usage
-----
::

    scripts/bisect_regression.py bisect \
        --category relusplitter \
        --row 34:unsat --row 24:unsat --row 40:unsat \
        --good e4b647ff --bad HEAD \
        --state reports/measured-runs/lane4-bisect

    scripts/bisect_regression.py report --state reports/measured-runs/lane4-bisect

The state directory is written after every probe, so an interrupted campaign
resumes by re-running the identical command line.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

STATE_VERSION = 3
STATE_FILE = "bisect_state.json"

GOOD = "good"
BAD = "bad"
SKIP = "skip"

DECIDED = ("sat", "unsat")
UNDECIDED = ("timeout", "unknown", "error", "none")

PRESET_MARKER = "Loading preset:"

# `ny`'s own watchdog commits a verdict and then SIGKILLs the process, so 137 is
# only fatal when it arrives without a sealed result.  See the campaign note:
# "rc=137 is NOT automatically a harness kill".
SIGKILL_RC = 137


class BisectError(RuntimeError):
    """Fatal, non-recoverable harness condition."""


class SoundnessAlarm(BisectError):
    """A probe produced the opposite decided verdict.  Never a bisect result."""


# --------------------------------------------------------------------------
# small process helpers
# --------------------------------------------------------------------------


def run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = True,
    log: Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run ``cmd``, optionally teeing combined output to ``log``.

    ``check`` raises :class:`BisectError` rather than ``CalledProcessError`` so
    every fatal path in this module surfaces the same way.
    """
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as handle:
            handle.write(f"\n$ {' '.join(cmd)}\n".encode())
            handle.flush()
            proc = subprocess.run(
                cmd,
                cwd=None if cwd is None else str(cwd),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        result = subprocess.CompletedProcess(cmd, proc.returncode, "", "")
    else:
        result = subprocess.run(
            cmd,
            cwd=None if cwd is None else str(cwd),
            env=env,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
    if check and result.returncode != 0:
        tail = ""
        if log is not None and log.exists():
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
        else:
            tail = (result.stdout or "") + (result.stderr or "")
        raise BisectError(f"command failed ({result.returncode}): {' '.join(cmd)}\n{tail}")
    return result


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = run(["git", "-C", str(repo), *args], check=check)
    return (result.stdout or "").strip()


def loadavg() -> float:
    return os.getloadavg()[0]


def gpu_busy_mib() -> int | None:
    """Total GPU memory in use, or ``None`` when nvidia-smi cannot answer.

    Load average does not observe GPU contention -- that has already produced
    one false regression in this campaign -- so probes record this too.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    total = 0
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            total += int(line)
    return total


# --------------------------------------------------------------------------
# ELF build-id extraction (no external tool dependency at probe time)
# --------------------------------------------------------------------------


def read_build_id(binary: Path) -> str | None:
    """Return the ``NT_GNU_BUILD_ID`` note of an ELF64 little-endian file.

    Parsed directly so the attestation does not depend on ``readelf`` being
    installed on whatever host the harness runs on.
    """
    try:
        data = binary.read_bytes()
    except OSError:
        return None
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return None
    if data[4] != 2 or data[5] != 1:  # ELF64, little-endian
        return None

    def u16(off: int) -> int:
        return int.from_bytes(data[off : off + 2], "little")

    def u32(off: int) -> int:
        return int.from_bytes(data[off : off + 4], "little")

    def u64(off: int) -> int:
        return int.from_bytes(data[off : off + 8], "little")

    e_phoff = u64(0x20)
    e_phentsize = u16(0x36)
    e_phnum = u16(0x38)
    PT_NOTE = 4
    for index in range(e_phnum):
        base = e_phoff + index * e_phentsize
        if base + e_phentsize > len(data):
            break
        if u32(base) != PT_NOTE:
            continue
        offset = u64(base + 0x08)
        size = u64(base + 0x20)
        cursor = offset
        end = min(offset + size, len(data))
        while cursor + 12 <= end:
            n_namesz = u32(cursor)
            n_descsz = u32(cursor + 4)
            n_type = u32(cursor + 8)
            name_off = cursor + 12
            desc_off = name_off + ((n_namesz + 3) // 4) * 4
            desc_end = desc_off + n_descsz
            if desc_end > end:
                break
            name = data[name_off : name_off + n_namesz].rstrip(b"\x00")
            if n_type == 3 and name == b"GNU":
                return data[desc_off:desc_end].hex()
            cursor = desc_off + ((n_descsz + 3) // 4) * 4
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_token(sha: str) -> str:
    """A 16-byte token that is a pure function of the commit under test."""
    return hashlib.sha256(f"ny-bisect-build-token:{sha}".encode()).hexdigest()[:32]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class RowSpec:
    row: int
    expect: str
    onnx: str = ""
    vnnlib: str = ""
    budget: int = 0

    @property
    def key(self) -> str:
        return str(self.row)


@dataclass
class Config:
    repo: Path
    category: str
    rows: list[RowSpec]
    controls: list[RowSpec]
    good: str
    bad: str
    state_dir: Path
    cache_dir: Path
    benchmarks: Path
    bench_year: str
    repeats: int
    budget_override: int | None
    max_load: float
    load_wait_s: float
    target_mode: str
    features: str
    jobs: int | None
    grace: int
    build_timeout_s: float
    control_attempts: int
    contention_backoff_s: float
    bad_confirm_load: float
    bad_confirm_wait_s: float
    keep_worktrees: bool

    def to_json(self) -> dict:
        return {
            "repo": str(self.repo),
            "category": self.category,
            "rows": [
                {
                    "row": r.row,
                    "expect": r.expect,
                    "onnx": r.onnx,
                    "vnnlib": r.vnnlib,
                    "budget": r.budget,
                }
                for r in self.rows
            ],
            "controls": [{"row": r.row, "expect": r.expect} for r in self.controls],
            "good": self.good,
            "bad": self.bad,
            "benchmarks": str(self.benchmarks),
            "bench_year": self.bench_year,
            "repeats": self.repeats,
            "budget_override": self.budget_override,
            "max_load": self.max_load,
            "bad_confirm_load": self.bad_confirm_load,
            "target_mode": self.target_mode,
            "features": self.features,
            "grace": self.grace,
        }


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


@dataclass
class RowState:
    spec: RowSpec
    goods: list[str] = field(default_factory=list)
    bad: str = ""
    skips: list[str] = field(default_factory=list)
    first_bad: str | None = None
    status: str = "searching"  # searching | resolved | unresolved | endpoint-failure
    note: str = ""

    def to_json(self) -> dict:
        return {
            "row": self.spec.row,
            "expect": self.spec.expect,
            "onnx": self.spec.onnx,
            "vnnlib": self.spec.vnnlib,
            "budget": self.spec.budget,
            "goods": self.goods,
            "bad": self.bad,
            "skips": self.skips,
            "first_bad": self.first_bad,
            "status": self.status,
            "note": self.note,
        }


class State:
    def __init__(self, config: Config):
        self.config = config
        self.path = config.state_dir / STATE_FILE
        self.rows: dict[str, RowState] = {}
        self.probes: dict[str, dict] = {}
        self.builds: dict[str, dict] = {}
        self.events: list[dict] = []

    # -- persistence -------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "version": STATE_VERSION,
            "config": self.config.to_json(),
            "rows": {k: v.to_json() for k, v in self.rows.items()},
            "probes": self.probes,
            "builds": self.builds,
            "events": self.events[-500:],
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True))
        os.replace(tmp, self.path)

    def load(self) -> bool:
        if not self.path.exists():
            return False
        blob = json.loads(self.path.read_text())
        if blob.get("version") != STATE_VERSION:
            self.log(f"state version {blob.get('version')} != {STATE_VERSION}; starting fresh")
            return False
        stored = blob.get("config", {})
        current = self.config.to_json()
        # A changed search question invalidates the recorded intervals, but the
        # build cache and probe observations stay valid and are worth keeping.
        invalidating = ("category", "good", "bad", "rows", "repeats", "budget_override")
        mismatched = [k for k in invalidating if stored.get(k) != current.get(k)]
        self.probes = blob.get("probes", {})
        self.builds = blob.get("builds", {})
        self.events = blob.get("events", [])
        if mismatched:
            self.log(f"config changed ({', '.join(mismatched)}); keeping probe/build cache, resetting intervals")
            return False
        for key, row in blob.get("rows", {}).items():
            spec = RowSpec(
                row=row["row"],
                expect=row["expect"],
                onnx=row.get("onnx", ""),
                vnnlib=row.get("vnnlib", ""),
                budget=row.get("budget", 0),
            )
            self.rows[key] = RowState(
                spec=spec,
                goods=row.get("goods", []),
                bad=row.get("bad", ""),
                skips=row.get("skips", []),
                first_bad=row.get("first_bad"),
                status=row.get("status", "searching"),
                note=row.get("note", ""),
            )
        return True

    # -- logging -----------------------------------------------------------
    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        self.events.append({"t": stamp, "msg": message})
        log_path = self.config.state_dir / "bisect.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


class Builder:
    def __init__(self, config: Config, state: State):
        self.config = config
        self.state = state
        self.worktree_root = config.cache_dir / "wt"
        self.bin_root = config.cache_dir / "bin"
        self.target_root = config.cache_dir / "target"

    def worktree(self, sha: str) -> Path:
        return self.worktree_root / sha

    def binary(self, sha: str) -> Path:
        return self.bin_root / sha / "ny"

    def build_record_path(self, sha: str) -> Path:
        """Where a build record lives.

        Deliberately in the *cache*, not in the campaign state: a build belongs
        to a commit, not to a question about a commit.  Two campaigns pointed at
        the same ``--cache`` therefore share the expensive artifact even though
        their intervals and probe histories are independent.
        """
        return self.bin_root / sha / "build.json"

    def cached_record(self, sha: str) -> dict | None:
        record = self.state.builds.get(sha)
        if record is not None:
            return record
        path = self.build_record_path(sha)
        if path.exists():
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                return None
            self.state.builds[sha] = record
            return record
        return None

    def store_record(self, sha: str, record: dict) -> None:
        self.state.builds[sha] = record
        path = self.build_record_path(sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True))

    def target_dir(self, sha: str) -> Path:
        if self.config.target_mode == "per-sha":
            return self.target_root / sha
        return self.target_root / "shared"

    # -- worktree ----------------------------------------------------------
    def ensure_worktree(self, sha: str) -> Path:
        path = self.worktree(sha)
        if not (path / ".git").exists():
            if path.exists():
                shutil.rmtree(path)
            # Clear any registration left by an interrupted campaign, otherwise
            # `worktree add` refuses the path.
            run(["git", "-C", str(self.config.repo), "worktree", "prune"], check=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            run(
                ["git", "-C", str(self.config.repo), "worktree", "add", "--detach", str(path), sha],
                log=self.config.state_dir / "logs" / f"worktree-{sha}.log",
            )
        # A fresh worktree has no benchmarks/ tree.  Symlink the year the
        # category lives in; /benchmarks/vnncomp20XX/ is gitignored, so the
        # symlink cannot dirty the tree the source attestation checks.
        bench_link = path / "benchmarks" / self.config.bench_year
        if not bench_link.exists():
            bench_link.parent.mkdir(parents=True, exist_ok=True)
            if bench_link.is_symlink():
                bench_link.unlink()
            os.symlink(self.config.benchmarks, bench_link)
        return path

    def attest_source(self, sha: str, path: Path) -> dict:
        """Gate 1: the tree really is this commit, and nothing tracked differs."""
        head = git(path, "rev-parse", "HEAD")
        full = git(self.config.repo, "rev-parse", sha)
        if head != full:
            raise BisectError(f"worktree {path} is at {head}, expected {full}")
        dirty = git(path, "status", "--porcelain", "--untracked-files=no")
        if dirty:
            raise BisectError(f"worktree {path} has modified tracked files:\n{dirty}")
        return {"head": head, "clean": True}

    # -- build -------------------------------------------------------------
    def build(self, sha: str, *, force: bool = False) -> dict:
        """Build ``ny`` at ``sha`` and return its verified build record."""
        full = git(self.config.repo, "rev-parse", sha)
        token = build_token(full)
        record = self.cached_record(full)
        binary = self.binary(full)

        if record and not force and record.get("ok") and binary.exists():
            # Cache hit still re-attests: the file could have been replaced
            # since it was recorded.
            if read_build_id(binary) == token and sha256_file(binary) == record.get("sha256"):
                self.state.log(f"build cache HIT {full[:8]} (build-id re-verified)")
                return record
            self.state.log(f"build cache STALE for {full[:8]}; rebuilding")
        if record and not force and record.get("ok") is False:
            self.state.log(f"build cache HIT {full[:8]}: known build failure")
            return record

        worktree = self.ensure_worktree(full)
        source = self.attest_source(full, worktree)

        target = self.target_dir(full)
        target.mkdir(parents=True, exist_ok=True)
        log = self.config.state_dir / "logs" / f"build-{full[:12]}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        if log.exists():
            log.unlink()

        env = dict(os.environ)
        env["CARGO_TARGET_DIR"] = str(target)
        env.setdefault("OPENSSL_DIR", str(Path.home() / ".local" / "openssl"))
        env.pop("RUSTFLAGS", None)

        # Gate 3's reference point: a stamp created after attestation and before
        # the build.  Any binary older than this is by definition not ours.
        stamp = target / ".bisect-stamp"
        stamp.write_text(full)
        stamp_mtime = stamp.stat().st_mtime

        out_bin = target / "release" / "ny"
        if out_bin.exists():
            # Remove first so a *failed* relink cannot leave the previous SHA's
            # artifact in place looking plausible.
            out_bin.unlink()

        cmd = [
            "targo", "--unverified",
            "rustc",
            "--release",
            "--features",
            self.config.features,
            "-p",
            "ny-cli",
            "--bin",
            "ny",
        ]
        if self.config.jobs:
            cmd += ["-j", str(self.config.jobs)]
        # `targo --unverified rustc` applies trailing flags to the top crate only, so the
        # token does not perturb the dependency fingerprints and the shared
        # target dir keeps its value -- while still forcing a relink per SHA.
        cmd += ["--", "-C", f"link-arg=-Wl,--build-id=0x{token}"]

        self.state.log(f"building {full[:8]} (token {token[:12]}..., target {target})")
        started = time.time()
        try:
            proc = run(cmd, cwd=worktree, env=env, check=False, log=log,
                       timeout=self.config.build_timeout_s)
        except subprocess.TimeoutExpired:
            proc = subprocess.CompletedProcess(cmd, -1, "", "")
        elapsed = time.time() - started

        failure: str | None = None
        # Gate 2: the process EXITED, and with status 0.
        if proc.returncode != 0:
            failure = f"targo exited {proc.returncode}"
        elif not out_bin.exists():
            failure = "targo reported success but produced no binary"
        else:
            mtime = out_bin.stat().st_mtime
            # Gate 3: mtime advanced past the stamp.
            if mtime <= stamp_mtime:
                failure = (
                    f"binary mtime {mtime} did not advance past pre-build stamp "
                    f"{stamp_mtime}: STALE ARTIFACT"
                )
            else:
                # Gate 4: the embedded token is this SHA's.
                observed = read_build_id(out_bin)
                if observed != token:
                    failure = (
                        f"embedded build-id {observed} != expected {token} for {full}: "
                        "the binary does NOT contain this commit"
                    )

        if failure:
            record = {
                "sha": full,
                "ok": False,
                "error": failure,
                "elapsed_s": round(elapsed, 1),
                "log": str(log),
                "source": source,
            }
            self.store_record(full, record)
            self.state.log(f"BUILD FAILED {full[:8]}: {failure}")
            self.state.save()
            return record

        binary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_bin, binary)
        digest = sha256_file(binary)
        # Gate 4 again on the copy that probes will actually execute.
        if read_build_id(binary) != token:
            raise BisectError(f"cached copy of {full} lost its build-id note")

        record = {
            "sha": full,
            "ok": True,
            "binary": str(binary),
            "build_id": token,
            "sha256": digest,
            "mtime": binary.stat().st_mtime,
            "elapsed_s": round(elapsed, 1),
            "log": str(log),
            "worktree": str(worktree),
            "source": source,
        }
        self.store_record(full, record)
        self.state.log(
            f"BUILD OK {full[:8]} in {elapsed / 60:.1f} min; build-id attested; sha256 {digest[:12]}"
        )
        self.state.save()
        return record

    def verify_binary(self, record: dict) -> Path:
        """Gate 5: re-attest immediately before a probe uses the binary."""
        binary = Path(record["binary"])
        if not binary.exists():
            raise BisectError(f"cached binary vanished: {binary}")
        if read_build_id(binary) != record["build_id"]:
            raise BisectError(f"build-id changed under us for {record['sha']}")
        if sha256_file(binary) != record["sha256"]:
            raise BisectError(f"binary content changed under us for {record['sha']}")
        return binary


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------


VERDICT_RE = re.compile(r"^\s*(sat|unsat|timeout|unknown|error)\b", re.IGNORECASE)


def parse_results_file(path: Path) -> str:
    if not path.exists():
        return "none"
    text = path.read_text(errors="replace").strip()
    if not text:
        return "none"
    match = VERDICT_RE.match(text.splitlines()[0])
    if match:
        return match.group(1).lower()
    return "unparsed:" + text.splitlines()[0][:40]


class Prober:
    def __init__(self, config: Config, state: State, builder: Builder):
        self.config = config
        self.state = state
        self.builder = builder

    def wait_for_quiet(self, deadline_s: float | None = None, force_wait: bool = False) -> float:
        """Hold until the 1-minute load average is under the cap, or give up.

        Giving up is deliberate: on a shared box the load may never settle, and a
        harness that blocks forever is worse than one that measures under load
        and records the load it measured under.  The control rows, not this
        function, are what decide whether a loaded probe is trustworthy.
        """
        budget = self.config.load_wait_s if deadline_s is None else deadline_s
        deadline = time.time() + budget
        current = loadavg()
        if current <= self.config.max_load and not force_wait:
            return current
        self.state.log(f"loadavg {current:.2f} > {self.config.max_load}; waiting up to {budget:.0f}s")
        while time.time() < deadline:
            time.sleep(20)
            current = loadavg()
            if current <= self.config.max_load:
                self.state.log(f"loadavg settled to {current:.2f}")
                return current
        self.state.log(
            f"loadavg still {current:.2f} after {budget:.0f}s; proceeding and recording it"
        )
        return current

    def run_once(self, sha: str, record: dict, spec: RowSpec, attempt) -> dict:
        binary = self.builder.verify_binary(record)
        worktree = Path(record["worktree"])
        configs = worktree / "configs"
        if not configs.is_dir():
            raise BisectError(f"no configs dir at {configs}")

        bench = self.config.benchmarks / "benchmarks" / self.config.category
        onnx = (bench / spec.onnx).resolve()
        vnnlib = (bench / spec.vnnlib).resolve()
        for path in (onnx, vnnlib):
            if not path.exists():
                raise BisectError(f"missing instance file {path}")

        budget = self.config.budget_override or spec.budget
        run_dir = self.config.state_dir / "runs" / f"{sha[:12]}" / f"row{spec.row}-a{attempt}"
        run_dir.mkdir(parents=True, exist_ok=True)
        results = run_dir / "result.txt"
        if results.exists():
            results.unlink()
        log = run_dir / "run.log"
        if log.exists():
            log.unlink()

        load0 = self.wait_for_quiet()
        gpu0 = gpu_busy_mib()

        # `-v` is a GLOBAL flag and must precede the subcommand -- `ny vnncomp
        # v1 ... -v` is rejected by clap outright.  RUST_LOG is ignored on this
        # path, so `-v` is the only way to get the `Loading preset:` line the
        # probe asserts on.
        cmd = [
            "timeout",
            "-k",
            "30",
            str(budget + self.config.grace),
            str(binary),
            "-v",
            "vnncomp",
            "v1",
            "--configs-dir",
            str(configs),
            self.config.category,
            str(onnx),
            str(vnnlib),
            str(results),
            str(budget),
        ]
        (run_dir / "cmd.txt").write_text(" ".join(cmd) + "\n")

        env = dict(os.environ)
        env.pop("RUST_LOG", None)  # RUST_LOG is ignored by ny; -v is the lever.

        started = time.time()
        proc = run(cmd, cwd=worktree, env=env, check=False, log=log)
        wall = time.time() - started

        text = log.read_text(errors="replace") if log.exists() else ""
        preset_lines = text.count(PRESET_MARKER)
        verdict = parse_results_file(results)

        out = {
            "sha": sha,
            "row": spec.row,
            "attempt": attempt,
            "rc": proc.returncode,
            "wall_s": round(wall, 2),
            "budget": budget,
            "verdict": verdict,
            "preset_lines": preset_lines,
            "load0": round(load0, 2),
            "gpu_mib0": gpu0,
            "gpu_mib1": gpu_busy_mib(),
            "load1": round(loadavg(), 2),
            "run_dir": str(run_dir),
            "binary_sha256": record["sha256"],
            "build_id": record["build_id"],
        }

        # Invalidity checks, in the order that makes the cause unambiguous.
        if preset_lines == 0:
            out["invalid"] = "no `Loading preset:` line -- the run did not use the category preset"
        elif proc.returncode == SIGKILL_RC and verdict in ("none", "unparsed"):
            out["invalid"] = "rc=137 with no sealed verdict -- unbankable in either direction"
        elif proc.returncode not in (0, SIGKILL_RC) and verdict == "none":
            out["invalid"] = f"rc={proc.returncode} with no verdict"
        return out

    def controls_ok(self, sha: str) -> tuple[bool, str]:
        """Run the control rows at ``sha``; a failure invalidates the probe.

        Cached per SHA, because the controls answer a question about the SHA and
        the box, not about any one row under test.
        """
        if not self.config.controls:
            return True, ""
        key = f"{sha}|controls"
        cached = self.state.probes.get(key)
        # Only a PASS is cached.  A control failure is evidence about the host at
        # a moment in time, not a property of the commit, so it must not become a
        # permanent verdict about the SHA.
        if cached and cached.get("class") == GOOD:
            return True, ""

        record = self.builder.cached_record(sha)
        if record is None or not record.get("ok"):
            return False, "build unavailable"

        runs: list[dict] = []
        reason = ""
        for round_index in range(1, self.config.control_attempts + 1):
            failures = []
            for spec in self.config.controls:
                ok = False
                for attempt in range(1, self.config.repeats + 1):
                    single = self.run_once(sha, record, spec, f"c{round_index}-{attempt}")
                    single["control"] = True
                    runs.append(single)
                    self.state.log(
                        f"  {sha[:8]} CONTROL row {spec.row} round {round_index} attempt {attempt}: "
                        f"verdict={single['verdict']} wall={single['wall_s']}s load={single['load0']}"
                    )
                    if single["verdict"] == spec.expect and not single.get("invalid"):
                        ok = True
                        break
                if not ok:
                    failures.append(str(spec.row))
            if not failures:
                self.state.probes[key] = {
                    "sha": sha, "row": "controls", "class": GOOD, "reason": "", "runs": runs,
                }
                self.state.save()
                return True, ""
            reason = (
                f"control row(s) {', '.join(failures)} failed at {sha[:8]} "
                f"(round {round_index}/{self.config.control_attempts}); the host, not the commit, "
                "is the likely cause"
            )
            self.state.log(f"CONTROLS FAILED at {sha[:8]}: {reason}")
            self.state.probes[key] = {
                "sha": sha, "row": "controls", "class": "contended", "reason": reason, "runs": runs,
            }
            self.state.save()
            if round_index < self.config.control_attempts:
                self.state.log(
                    f"  backing off {self.config.contention_backoff_s:.0f}s and re-arming controls"
                )
                self.wait_for_quiet(deadline_s=self.config.contention_backoff_s, force_wait=True)
        return False, reason + " -- probe not charged to this commit"

    def confirm_bad(self, sha: str, record: dict, spec: RowSpec, runs: list[dict]) -> bool | None:
        """Re-run a failing row on a quiet box before charging it to the commit.

        Why this exists, in one measurement.  ``relusplitter`` rows 24 and 40
        were probed at ``e4b647ff`` at loadavg 32-53 and returned 3/3 timeouts at
        95.4 s.  Three hours later the *identical attested binary* at the
        *identical SHA* decided both on the first attempt -- ``unsat`` in 42.9 s
        and 39.9 s at loadavg ~11.  Both control rows passed on every one of
        those probes, so the controls did not catch it: they decide in ~10 s of a
        90 s budget (9x margin) while rows 24/40 need ~40 s (2.2x margin), and a
        2x slowdown kills the second and not the first.

        A control can only detect the contention that would matter if its margin
        resembles the margin of the rows under test.  Rather than demand that of
        the caller, refuse to accept the expensive verdict from a loaded box at
        all: wait for quiet, run one more repeat, and let that decide.

        Returns ``True`` when the failure reproduces quiet, ``False`` when the
        quiet run decides (the earlier failures were contention), and ``None``
        when the box never went quiet -- in which case the probe is SKIP, because
        "I could not tell" is the honest answer and it must not move an interval.
        """
        load = self.wait_for_quiet(deadline_s=self.config.bad_confirm_wait_s)
        if load > self.config.bad_confirm_load:
            self.state.log(
                f"  {sha[:8]} row {spec.row}: no quiet window (loadavg {load:.2f} > "
                f"{self.config.bad_confirm_load}); refusing to charge the failure to this commit"
            )
            return None
        single = self.run_once(sha, record, spec, "quiet-confirm")
        single["quiet_confirm"] = True
        runs.append(single)
        self.state.log(
            f"  {sha[:8]} row {spec.row} QUIET CONFIRM: verdict={single['verdict']} "
            f"wall={single['wall_s']}s load={single['load0']}"
        )
        self.state.save()
        if single.get("invalid"):
            return None
        if single["verdict"] == spec.expect:
            return False
        opposite = "sat" if spec.expect == "unsat" else "unsat"
        if single["verdict"] == opposite:
            raise SoundnessAlarm(
                f"row {spec.row} at {sha}: quiet confirmation produced {opposite}, expected "
                f"{spec.expect}. Evidence: {single['run_dir']}"
            )
        return True

    def probe(self, sha: str, spec: RowSpec) -> dict:
        """Classify ``sha`` for one row.  GOOD on any expected verdict."""
        key = f"{sha}|{spec.row}"
        cached = self.state.probes.get(key)
        if cached and cached.get("class") in (GOOD, BAD, SKIP):
            self.state.log(f"probe cache HIT {sha[:8]} row {spec.row} -> {cached['class']}")
            return cached

        record = self.builder.cached_record(sha)
        if record is None or not record.get("ok"):
            result = {
                "sha": sha,
                "row": spec.row,
                "class": SKIP,
                "reason": (record or {}).get("error", "build unavailable"),
                "runs": [],
            }
            self.state.probes[key] = result
            self.state.save()
            return result

        controls_pass, controls_reason = self.controls_ok(sha)
        if not controls_pass:
            result = {
                "sha": sha,
                "row": spec.row,
                "class": SKIP,
                "reason": controls_reason,
                "runs": [],
            }
            self.state.probes[key] = result
            self.state.save()
            return result

        runs: list[dict] = []
        classification = BAD
        reason = ""
        opposite = "sat" if spec.expect == "unsat" else "unsat"

        for attempt in range(1, self.config.repeats + 1):
            single = self.run_once(sha, record, spec, attempt)
            runs.append(single)
            self.state.log(
                f"  {sha[:8]} row {spec.row} attempt {attempt}/{self.config.repeats}: "
                f"verdict={single['verdict']} rc={single['rc']} wall={single['wall_s']}s "
                f"preset={single['preset_lines']} load={single['load0']}"
                + (f" INVALID: {single['invalid']}" if single.get("invalid") else "")
            )
            self.state.probes[key] = {
                "sha": sha,
                "row": spec.row,
                "class": "in-progress",
                "runs": runs,
            }
            self.state.save()

            if single["verdict"] == opposite:
                raise SoundnessAlarm(
                    f"row {spec.row} at {sha}: expected {spec.expect} but got {opposite}. "
                    f"This is a soundness event, not a regression. Evidence: {single['run_dir']}"
                )
            if single.get("invalid"):
                continue
            if single["verdict"] == spec.expect:
                classification = GOOD
                reason = f"decided {spec.expect} in {single['wall_s']}s on attempt {attempt}"
                break

        if classification != GOOD:
            valid = [r for r in runs if not r.get("invalid")]
            if not valid:
                classification = SKIP
                reason = "every repeat was invalid: " + "; ".join(
                    sorted({r.get("invalid", "?") for r in runs})
                )
            else:
                # A BAD verdict is the expensive one -- it moves the interval and
                # it is what gets reported as a dead row. It is not accepted from
                # a contended box.
                confirmed = self.confirm_bad(sha, record, spec, runs)
                if confirmed is None:
                    classification = SKIP
                    reason = (
                        f"row failed {len(valid)}/{len(valid)} repeats but the box never went "
                        f"quiet (loadavg <= {self.config.bad_confirm_load}) within "
                        f"{self.config.bad_confirm_wait_s:.0f}s, so the failure cannot be "
                        "attributed to this commit"
                    )
                elif confirmed:
                    classification = BAD
                    reason = (
                        f"{len(valid)} valid repeats plus a quiet-box confirmation, none reached "
                        f"{spec.expect} (saw {sorted({r['verdict'] for r in valid})})"
                    )
                else:
                    classification = GOOD
                    last = runs[-1]
                    reason = (
                        f"{len(valid)} repeats failed under load but the quiet-box confirmation "
                        f"decided {spec.expect} in {last['wall_s']}s — the earlier failures were "
                        "contention, not this commit"
                    )

        result = {
            "sha": sha,
            "row": spec.row,
            "class": classification,
            "reason": reason,
            "runs": runs,
        }
        self.state.probes[key] = result
        self.state.log(f"PROBE {sha[:8]} row {spec.row} => {classification.upper()} ({reason})")
        self.state.save()
        return result


# --------------------------------------------------------------------------
# DAG bisection (git bisect semantics, full topology)
# --------------------------------------------------------------------------


def exclusions(goods: Sequence[str]) -> list[str]:
    """Render known-good commits as ``^sha`` exclusions.

    NOT ``--not a --not b``.  In git's revision grammar ``--not`` *reverses* the
    sense of everything that follows it until the next ``--not``, so a second
    ``--not`` flips it back and silently re-includes that commit.  A synthetic
    replay of relusplitter row 34 caught exactly that: accumulated goods stopped
    shrinking the interval and the search re-proposed one candidate forever.
    The ``^sha`` prefix has no such positional meaning.
    """
    return [f"^{good}" for good in goods]


def bisect_candidate(repo: Path, bad: str, goods: Sequence[str], skips: Sequence[str]) -> tuple[str | None, int]:
    """Return git's own next bisect midpoint and the remaining candidate count.

    Uses ``git rev-list --bisect-vars`` so the search matches ``git bisect``
    exactly, including merge topology.  ``1ede1d30`` is not on main's
    first-parent spine, so a spine-only search could never name it; this walks
    the full DAG.
    """
    args = ["git", "-C", str(repo), "rev-list", "--bisect-vars", bad, *exclusions(goods)]
    result = run(args, check=False)
    if result.returncode != 0:
        return None, 0
    variables: dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        if "=" in line:
            name, _, value = line.partition("=")
            variables[name.strip()] = value.strip().strip("'")
    remaining = int(variables.get("bisect_all", 0) or 0)
    # bisect_all counts the candidate set including the bad commit itself.
    remaining = max(remaining - 1, 0)
    candidate = variables.get("bisect_rev")
    if not candidate:
        return None, remaining
    if candidate in skips:
        candidate = pick_non_skipped(repo, bad, goods, skips)
    return candidate, remaining


def pick_non_skipped(repo: Path, bad: str, goods: Sequence[str], skips: Sequence[str]) -> str | None:
    """Fall back to the best-ranked non-skipped commit, as ``git bisect skip`` does."""
    args = ["git", "-C", str(repo), "rev-list", "--bisect-all", bad, *exclusions(goods)]
    result = run(args, check=False)
    if result.returncode != 0:
        return None
    for line in (result.stdout or "").splitlines():
        sha = line.split()[0] if line.split() else ""
        if sha and sha not in skips and sha != bad:
            return sha
    return None


def candidates_between(repo: Path, bad: str, goods: Sequence[str]) -> list[str]:
    args = ["git", "-C", str(repo), "rev-list", bad, *exclusions(goods)]
    result = run(args, check=False)
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant], check=False
    )
    return result.returncode == 0


def in_open_interval(repo: Path, sha: str, row: RowState) -> bool:
    """True when ``sha`` is still an undecided candidate for this row."""
    if not is_ancestor(repo, sha, row.bad):
        return False
    return all(not is_ancestor(repo, sha, good) for good in row.goods)


# --------------------------------------------------------------------------
# campaign driver
# --------------------------------------------------------------------------


def load_instances(config: Config) -> list[tuple[str, str, int]]:
    path = config.benchmarks / "benchmarks" / config.category / "instances.csv"
    if not path.exists():
        raise BisectError(f"no instances.csv at {path}")
    rows: list[tuple[str, str, int]] = []
    with path.open() as handle:
        for record in csv.reader(handle):
            if len(record) < 3:
                continue
            rows.append((record[0].strip(), record[1].strip(), int(float(record[2]))))
    return rows


def resolve_rows(config: Config) -> None:
    """Bind each ``--row N:verdict`` to its instances.csv line (1-based)."""
    instances = load_instances(config)
    for spec in [*config.rows, *config.controls]:
        if spec.row < 1 or spec.row > len(instances):
            raise BisectError(
                f"row {spec.row} out of range for {config.category} (1..{len(instances)})"
            )
        onnx, vnnlib, budget = instances[spec.row - 1]
        spec.onnx, spec.vnnlib = onnx, vnnlib
        if spec.budget == 0:
            spec.budget = budget


def campaign(config: Config) -> dict:
    state = State(config)
    resumed = state.load()
    builder = Builder(config, state)
    prober = Prober(config, state, builder)

    good_full = git(config.repo, "rev-parse", config.good)
    bad_full = git(config.repo, "rev-parse", config.bad)
    if not is_ancestor(config.repo, good_full, bad_full):
        raise BisectError(f"--good {config.good} is not an ancestor of --bad {config.bad}")

    # `probe` runs record observations without creating row intervals, so a
    # state directory can legitimately carry a warm probe cache and no rows.
    # Seed the intervals in that case too -- the cached probes then make the
    # endpoint verification free.
    if not resumed or not state.rows:
        state.rows = {}
        for spec in config.rows:
            state.rows[spec.key] = RowState(spec=spec, goods=[good_full], bad=bad_full)
    state.log(
        f"campaign {'resumed' if resumed else 'started'}: {config.category} rows "
        f"{[r.spec.row for r in state.rows.values()]} over "
        f"{len(candidates_between(config.repo, bad_full, [good_full]))} candidate commits "
        f"({good_full[:8]}..{bad_full[:8]})"
    )
    state.save()

    # ---- endpoint verification ------------------------------------------
    # A bisect whose endpoints are not verified is a guess.  Both anchors are
    # probed before any halving, and a row whose anchors do not bracket the
    # regression is reported as such rather than silently converging.
    for anchor, expect_class in ((bad_full, BAD), (good_full, GOOD)):
        unresolved = [r for r in state.rows.values() if r.status == "searching"]
        if not unresolved:
            break
        builder.build(anchor)
        for row in unresolved:
            result = prober.probe(anchor, row.spec)
            if result["class"] == SKIP:
                row.status = "endpoint-failure"
                row.note = f"anchor {anchor[:8]} unusable: {result['reason']}"
                continue
            if anchor == bad_full and result["class"] == GOOD:
                row.status = "unresolved"
                row.note = (
                    f"row is NOT regressed at --bad {bad_full[:8]}: it still reaches "
                    f"{row.spec.expect}. Nothing to bisect."
                )
            if anchor == good_full and result["class"] == BAD and row.status == "searching":
                row.status = "unresolved"
                row.note = (
                    f"row is ALREADY failing at --good {good_full[:8]}: the first-bad commit "
                    "is older than the search interval. Widen --good."
                )
        state.save()

    # ---- shared halving --------------------------------------------------
    while True:
        searching = [r for r in state.rows.values() if r.status == "searching"]
        if not searching:
            break

        # Pick the probe SHA from whichever row has the widest interval, then
        # spend it on every row it can inform.  A build costs tens of minutes
        # and a probe a few, so a shared SHA is close to free for extra rows.
        best_sha: str | None = None
        best_remaining = -1
        for row in searching:
            candidate, remaining = bisect_candidate(config.repo, row.bad, row.goods, row.skips)
            if candidate is None or remaining == 0:
                row.first_bad = row.bad
                row.status = "resolved"
                row.note = "interval collapsed"
                continue
            if remaining > best_remaining:
                best_remaining, best_sha = remaining, candidate
        state.save()
        searching = [r for r in state.rows.values() if r.status == "searching"]
        if not searching or best_sha is None:
            continue

        subject = git(config.repo, "log", "-1", "--format=%s", best_sha)
        state.log(
            f"probing {best_sha[:8]} ({best_remaining} candidates in the widest open interval): {subject}"
        )
        record = builder.build(best_sha)

        before = {key: (row.bad, len(row.goods), len(row.skips)) for key, row in state.rows.items()}
        for row in list(searching):
            if not in_open_interval(config.repo, best_sha, row):
                continue
            result = prober.probe(best_sha, row.spec)
            if result["class"] == SKIP:
                row.skips.append(best_sha)
                state.log(f"row {row.spec.row}: SKIP {best_sha[:8]} -- {result['reason']}")
            elif result["class"] == GOOD:
                row.goods.append(best_sha)
            else:
                row.bad = best_sha
            state.save()

        # Liveness guard.  A probe that moves no interval means the next
        # iteration would choose the same SHA and spin forever -- which is
        # exactly what the `--not a --not b` defect did before it was fixed.
        # Fail loudly rather than burn the box in a silent loop.
        after = {key: (row.bad, len(row.goods), len(row.skips)) for key, row in state.rows.items()}
        if before == after:
            raise BisectError(
                f"probe at {best_sha[:8]} advanced no row's interval; refusing to loop. "
                "State is on disk and the campaign can resume once this is understood."
            )

        # Convergence check.  Log each row's surviving interval so an
        # interrupted campaign still reports "first-bad is somewhere in these N
        # commits, ending at <sha>" rather than nothing.
        for row in [r for r in state.rows.values() if r.status == "searching"]:
            candidate, remaining = bisect_candidate(config.repo, row.bad, row.goods, row.skips)
            state.log(
                f"row {row.spec.row}: {remaining} candidate(s) remain, interval ends at "
                f"{row.bad[:8]}"
            )
            if remaining == 0 or candidate is None:
                row.first_bad = row.bad
                row.status = "resolved"
                if row.skips:
                    row.note = (
                        f"{len(row.skips)} commit(s) skipped in this interval; "
                        "the answer is the first bad commit among the probeable ones"
                    )
        state.save()

    summary = summarise(config, state)
    (config.state_dir / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    state.save()
    return summary


def summarise(config: Config, state: State) -> dict:
    rows = []
    for row in sorted(state.rows.values(), key=lambda r: r.spec.row):
        entry = {
            "row": row.spec.row,
            "onnx": row.spec.onnx,
            "vnnlib": row.spec.vnnlib,
            "budget": row.spec.budget,
            "expect": row.spec.expect,
            "status": row.status,
            "first_bad": row.first_bad,
            "note": row.note,
            "skips": row.skips,
        }
        if row.first_bad:
            entry["first_bad_subject"] = git(config.repo, "log", "-1", "--format=%s", row.first_bad)
            entry["first_bad_date"] = git(config.repo, "log", "-1", "--format=%ci", row.first_bad)
            parents = git(config.repo, "rev-list", "--parents", "-n", "1", row.first_bad).split()
            entry["first_bad_parents"] = parents[1:]
        rows.append(entry)
    builds = {
        sha[:12]: {"ok": rec.get("ok"), "elapsed_s": rec.get("elapsed_s"), "error": rec.get("error")}
        for sha, rec in state.builds.items()
    }
    return {
        "category": config.category,
        "good": config.good,
        "bad": config.bad,
        "repeats": config.repeats,
        "rows": rows,
        "builds": builds,
        "probe_count": len([p for p in state.probes.values() if p.get("class") != "in-progress"]),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_row(value: str) -> RowSpec:
    parts = value.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"--row wants N:verdict[:budget], got {value!r}")
    row = int(parts[0])
    expect = parts[1].lower()
    if expect not in DECIDED:
        raise argparse.ArgumentTypeError(f"expected verdict must be sat|unsat, got {expect!r}")
    budget = int(parts[2]) if len(parts) > 2 else 0
    return RowSpec(row=row, expect=expect, budget=budget)


def build_config(args: argparse.Namespace) -> Config:
    repo = Path(args.repo).resolve()
    state_dir = Path(args.state).resolve()
    cache_dir = Path(args.cache).resolve() if args.cache else state_dir / "cache"
    benchmarks = Path(args.benchmarks).resolve()
    config = Config(
        repo=repo,
        category=args.category,
        rows=args.row,
        controls=args.control or [],
        good=args.good,
        bad=args.bad,
        state_dir=state_dir,
        cache_dir=cache_dir,
        benchmarks=benchmarks,
        bench_year=benchmarks.name,
        repeats=args.repeats,
        budget_override=args.budget,
        max_load=args.max_load,
        load_wait_s=args.load_wait,
        target_mode=args.target_mode,
        features=args.features,
        jobs=args.jobs,
        grace=args.grace,
        build_timeout_s=args.build_timeout,
        control_attempts=args.control_attempts,
        contention_backoff_s=args.contention_backoff,
        bad_confirm_load=args.bad_confirm_load,
        bad_confirm_wait_s=args.bad_confirm_wait,
        keep_worktrees=True,
    )
    resolve_rows(config)
    return config


def cmd_bisect(args: argparse.Namespace) -> int:
    config = build_config(args)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    summary = campaign(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Build and attest a single SHA -- useful for pre-warming the cache."""
    config = build_config(args)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    state = State(config)
    state.load()
    builder = Builder(config, state)
    record = builder.build(args.sha, force=args.force)
    state.save()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record.get("ok") else 1


def cmd_probe(args: argparse.Namespace) -> int:
    """Classify one SHA for the given rows without moving any interval."""
    config = build_config(args)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    state = State(config)
    state.load()
    builder = Builder(config, state)
    prober = Prober(config, state, builder)
    builder.build(args.sha)
    out = {}
    for spec in config.rows:
        result = prober.probe(git(config.repo, "rev-parse", args.sha), spec)
        out[spec.key] = {"class": result["class"], "reason": result.get("reason", "")}
    state.save()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Replay the search against a synthetic oracle -- no builds, no runs.

    The oracle says "BAD exactly at commits that have ``--culprit`` as an
    ancestor", which is the shape of a real regression.  A correct DAG search
    must return ``--culprit`` itself.  This is cheap enough to run before every
    campaign, and it earns its keep: it is what caught the ``--not a --not b``
    defect (git re-flips the sense on the second ``--not``, so accumulated goods
    stopped shrinking the interval and the search re-proposed one candidate
    forever) before that defect could spend a day of builds converging on an
    innocent commit.
    """
    repo = Path(args.repo).resolve()
    culprit = git(repo, "rev-parse", args.culprit)
    good = git(repo, "rev-parse", args.good)
    bad = git(repo, "rev-parse", args.bad)

    probes: list[str] = []

    def fake_build(self, sha, force=False):  # noqa: ANN001
        full = git(self.config.repo, "rev-parse", sha)
        record = {"sha": full, "ok": True, "binary": "/dev/null", "build_id": "x",
                  "sha256": "y", "worktree": str(self.config.repo)}
        self.state.builds[full] = record
        return record

    def fake_cached(self, sha):  # noqa: ANN001
        return self.state.builds.get(sha)

    def fake_probe(self, sha, spec):  # noqa: ANN001
        probes.append(sha)
        bad_here = is_ancestor(self.config.repo, culprit, sha)
        return {"sha": sha, "row": spec.row, "class": BAD if bad_here else GOOD,
                "reason": "synthetic oracle", "runs": []}

    saved = (Builder.build, Builder.cached_record, Prober.probe, Prober.controls_ok)
    Builder.build, Builder.cached_record = fake_build, fake_cached
    Prober.probe, Prober.controls_ok = fake_probe, lambda self, sha: (True, "")
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config = Config(
                repo=repo, category=args.category, rows=[RowSpec(row=args.row, expect="unsat")],
                controls=[], good=good, bad=bad, state_dir=Path(tmp),
                cache_dir=Path(tmp) / "cache", benchmarks=Path(args.benchmarks).resolve(),
                bench_year=Path(args.benchmarks).name, repeats=1, budget_override=None,
                max_load=1e9, load_wait_s=0.0, target_mode="shared", features="",
                jobs=None, grace=0, build_timeout_s=1.0, control_attempts=1,
                contention_backoff_s=0.0, bad_confirm_load=1e9, bad_confirm_wait_s=0.0,
                keep_worktrees=True,
            )
            resolve_rows(config)
            summary = campaign(config)
    finally:
        Builder.build, Builder.cached_record, Prober.probe, Prober.controls_ok = saved

    found = summary["rows"][0]["first_bad"]
    ok = found == culprit
    print(json.dumps({
        "culprit_expected": culprit,
        "first_bad_found": found,
        "match": ok,
        "probes": len(probes),
        "candidates": len(candidates_between(repo, bad, [good])),
    }, indent=2))
    return 0 if ok else 1


def cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.state).resolve() / STATE_FILE
    if not path.exists():
        print(f"no state at {path}", file=sys.stderr)
        return 1
    blob = json.loads(path.read_text())
    print(json.dumps({"rows": blob.get("rows"), "builds": blob.get("builds")}, indent=2, sort_keys=True))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
        p.add_argument("--category", required=True)
        p.add_argument("--row", action="append", type=parse_row, required=True,
                       help="N:verdict[:budget], 1-based line in instances.csv")
        p.add_argument("--control", action="append", type=parse_row, default=None,
                       help="N:verdict for a row known-good across the interval; its "
                            "failure marks the whole probe SKIP (contention guard)")
        p.add_argument("--good", default="HEAD")
        p.add_argument("--bad", default="HEAD")
        p.add_argument("--state", required=True)
        p.add_argument("--cache", default=None,
                       help="build cache root (default <state>/cache); share it across campaigns")
        p.add_argument("--benchmarks", default="<home>/ny/benchmarks/vnncomp2025")
        p.add_argument("--repeats", type=int, default=3,
                       help="consecutive failing runs required before a row counts as BAD")
        p.add_argument("--budget", type=int, default=None, help="override the instance budget")
        p.add_argument("--max-load", type=float, default=12.0)
        p.add_argument("--load-wait", type=float, default=900.0)
        p.add_argument("--target-mode", choices=("shared", "per-sha"), default="shared")
        p.add_argument("--features", default="cuda,mip")
        p.add_argument("--jobs", type=int, default=None)
        p.add_argument("--grace", type=int, default=120, help="timeout(1) grace over the budget")
        p.add_argument("--bad-confirm-load", type=float, default=16.0,
                       help="loadavg a BAD probe must be re-confirmed under; above it the probe "
                            "is SKIP rather than charged to the commit")
        p.add_argument("--bad-confirm-wait", type=float, default=5400.0,
                       help="seconds to hold for that quiet window before giving up")
        p.add_argument("--control-attempts", type=int, default=3,
                       help="rounds of control re-arming before a probe is abandoned as SKIP")
        p.add_argument("--contention-backoff", type=float, default=1800.0,
                       help="seconds to hold for a quiet box after a control failure")
        p.add_argument("--build-timeout", type=float, default=9000.0,
                       help="seconds before a candidate build is abandoned as SKIP")

    p_bisect = sub.add_parser("bisect", help="run the first-bad-commit search")
    common(p_bisect)
    p_bisect.set_defaults(func=cmd_bisect)

    p_build = sub.add_parser("build", help="build + attest one SHA")
    common(p_build)
    p_build.add_argument("--sha", required=True)
    p_build.add_argument("--force", action="store_true")
    p_build.set_defaults(func=cmd_build)

    p_probe = sub.add_parser("probe", help="classify one SHA for the given rows")
    common(p_probe)
    p_probe.add_argument("--sha", required=True)
    p_probe.set_defaults(func=cmd_probe)

    p_self = sub.add_parser("selftest", help="replay the search against a synthetic oracle")
    p_self.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    p_self.add_argument("--category", default="relusplitter")
    p_self.add_argument("--row", type=int, default=34)
    p_self.add_argument("--benchmarks", default="<home>/ny/benchmarks/vnncomp2025")
    p_self.add_argument("--culprit", required=True, help="the commit the oracle makes bad")
    p_self.add_argument("--good", required=True)
    p_self.add_argument("--bad", required=True)
    p_self.set_defaults(func=cmd_selftest)

    p_report = sub.add_parser("report", help="print the current state")
    p_report.add_argument("--state", required=True)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return args.func(args)
    except SoundnessAlarm as exc:
        print(f"\nSOUNDNESS ALARM: {exc}", file=sys.stderr)
        return 3
    except BisectError as exc:
        print(f"\nharness error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
