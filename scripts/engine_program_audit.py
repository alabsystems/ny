#!/usr/bin/env python3
"""Read-only auditor for the ENGINE_PROGRAM capability checklist.

It re-derives, from a machine-readable registry (docs/engine_program_capabilities.yaml),
the stages of every capability that can be checked without cargo, a ny binary or the box:

  S1  built + adversarially reviewed   the recorded review commit exists and its message says so, or the
                                       recorded review report names that commit and carries its verdict
  S2  merged to origin/main            `git merge-base --is-ancestor <head> <main-ref>` for every head
  S3  named tests green in a main gate each named test's own `test ... ok` line in the gate leg logs;
                                       "main gate" = a gate run whose HEAD is on <main-ref>. Only the newest
                                       heartbeat block per leg prefix owns its leg files, dirty runs and legs
                                       older than their block do not count, and a device-gated test counts
                                       only from a leg that ran with its output visible and printed no SKIP
  S4  delivered                        LeverDecl, space.rs Deliver::PresetKey axis for the SAME yaml path,
                                       preset schema field, apply.rs mapping (generic axis loop or an
                                       explicit field), ny-cli bridge + ny-propagate cell, consumer sites.
                                       A consumer counts only on a PRODUCTION line (not a comment, not test
                                       code); a definition or import counts only with a production CALLER
                                       A key recorded `retired: {refusal: ...}` is audited in REVERSE: n/a
                                       only if it has no PresetKey axis, no bridge write, no ledger row, and
                                       its refusal string is on a production line of preset/apply.rs
  S5  arrives                          a recorded run log on a real row contains `Loading preset: <yaml>`,
                                       THAT yaml arms the key, the log carries the lever's own marker, the
                                       recorded runner refuses NY_* on a code line, and the latest pin
                                       still carries the capability's markers (`strings -a <pin> | grep -cF`)
  S6-S8, S10, S11                      the recorded evidence files exist (and contain the recorded pattern);
                                       an `na` needs a recorded reason. When the evidence names a seal STAGE record
                                       (`evidence.<S5..S8>.seal`, written by scripts/seal/seal.py record), the stage
                                       is DERIVED from it: scripts/seal/seallib.py re-judges the record from its own
                                       inputs (sha-bound plan, specs, judged.tsv re-derived from the run artefacts,
                                       facts re-scanned) with the judge the runner used into the SAME record, and the
                                       record's pin commit must contain the capability's heads. S5
                                       with a seal record is the lower of the arrival check and the seal record
  SR  routable (ROUTER_PROGRAM 3.5)    SR.seal: S7 and S8 derived done from seal records, `routable.envelope` (op
                                       kinds) and `envelope_specs` recorded, both inside the op sets / spec
                                       archetypes of the S7/S8 evidence rows on which the capability engaged, and
                                       every moat row the envelope admits (both years, every scored category) an
                                       S8 row (seallib.sr_seal); n/a for a capability with no `routable:` block
  S9  decision                         yaml arming in configs/vnncomp*, resolved the way the preset resolver
                                       resolves each SCORED category (newest year first; full name, then the
                                       name without its _20NN suffix), so a file no scored category loads is
                                       a placebo; or a recorded 'not armed' decision with a reason; or the
                                       decision of the capability it inherits from

Every negative ("0 production sites") claim carries a positive control that must match, so a zero from a
renamed symbol or a wrong path is an ERROR, never an absence (the CLAUDE.md empty-grep trap).

It never writes anything except, with --update-md, the two generated regions of the checklist.
Exit status: 0 = every capability is at or above its recorded stages; 1 = a regression (a stage
derived below its recorded state, a recorded arming that is gone, or an arming the registry does
not record); 2 = usage / registry / input error (a missing pin, gate log, gate script, config year,
benchmark root, or a failed positive control) — an input the audit cannot read never passes.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as _dt
import fnmatch
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is present on every ny host
    sys.stderr.write("engine_program_audit: PyYAML is required (python3 -m pip install pyyaml)\n")
    sys.exit(2)

STAGES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "SR", "S9", "S10", "S11"]
RANK = {"open": 0, "partial": 1, "done": 2, "na": 2}
SYMBOL = {"done": "x", "na": "-", "partial": "~", "open": " "}
DERIVED = {"S1", "S2", "S3", "S4", "S5", "S9", "SR"}
# review(eng11): `record.SR` may also carry a level of the router's SR ladder (ROUTER_PROGRAM 3.5, #router-r1 /
# #router-r5: decl, obs, seal), which `derive_sr` and router_r5_registry_ladders_match_the_capability_registry read.
# This auditor derives SR.seal only, so a ladder level is compared as the RANK state it claims of SR.seal.
SR_LADDER = {"decl": "open", "obs": "open", "seal": "done"}
LEG_EXCLUDE = re.compile(r"_(strings|strings_recheck|release|check|check_cli|regen)\.txt$")
TEST_LINE = re.compile(r"^test (\S+)(?: - should panic)? \.\.\.(.*)$")
DOCTEST_LINE = re.compile(r"^test (crates/\S+\.rs) - (.*?) \.\.\.(.*)$")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TEST_FILE_RX = re.compile(r"(^|_)tests?\.rs$|_tests?_")
TEST_DIRS = {"tests", "benches", "examples", "target", ".git"}
DEF_LINE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:(?:const|async|unsafe|default|extern\s+\"[^\"]*\")\s+)*fn\s+\w+"
    r"|^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+\w+\s*:"
    r"|^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|type|mod|impl)\b"
    r"|^\s*(?:pub(?:\([^)]*\))?\s+)?use\s")


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------

def run(cmd, cwd=None, check=False):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {p.stderr.strip()}")
    return p


class Git:
    def __init__(self, tree: str):
        self.tree = tree
        self._anc: dict[tuple[str, str], bool] = {}
        self._sha: dict[str, str | None] = {}

    def sha(self, rev: str) -> str | None:
        if rev not in self._sha:
            p = run(["git", "-C", self.tree, "rev-parse", "--verify", "--quiet", rev + "^{commit}"])
            self._sha[rev] = p.stdout.strip() or None
        return self._sha[rev]

    def is_ancestor(self, a: str, b: str) -> bool | None:
        sa, sb = self.sha(a), self.sha(b)
        if sa is None or sb is None:
            return None
        key = (sa, sb)
        if key not in self._anc:
            self._anc[key] = run(["git", "-C", self.tree, "merge-base", "--is-ancestor", sa, sb]).returncode == 0
        return self._anc[key]

    def message(self, rev: str) -> str:
        return run(["git", "-C", self.tree, "log", "-1", "--format=%B", rev]).stdout

    def subject(self, rev: str) -> str:
        return run(["git", "-C", self.tree, "log", "-1", "--format=%s", rev]).stdout.strip()


def seal_module():
    """scripts/seal/seallib.py: the seal pipeline's stage judge (imported, never re-implemented here)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seal")
    if p not in sys.path:
        sys.path.insert(0, p)
    import seallib  # noqa: E402
    return seallib


def read(path: str) -> str:
    with open(path, errors="replace") as fh:
        return fh.read()


def parse_ts(s: str) -> float | None:
    try:
        return _dt.datetime.fromisoformat(s).timestamp()
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------------
# source scanning: comment-, string- and test-region-aware. A match in a comment, in a test module or
# in a test-only file is never evidence of production use.
# --------------------------------------------------------------------------------------------

_RS_TOK = re.compile(r"//[^\n]*|/\*|(?<![A-Za-z0-9_])b?r#*\"|b?\"|b?'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]{1,6}\}|.)|[^\\'\n])'|[{}]")
_STR_END = re.compile(r'(?:[^"\\]|\\.)*"', re.S)
_BLOCK = re.compile(r"/\*|\*/")
_TEST_ATTR = re.compile(r"#!?\[\s*(?:cfg\s*\(\s*(?:test|all\s*\(\s*test\b[^\]]*)\)|test|tokio::test[^\]]*)\s*\]")
_INNER_TEST = re.compile(r"#!\[\s*cfg\s*\(\s*(?:test|all\s*\(\s*test\b[^\]]*)\)\s*\]")
_CFG_TEST_MOD = re.compile(
    r"#\[\s*cfg\s*\(\s*(?:test|all\s*\(\s*test\b[^\]]*)\)\s*\]\s*((?:#\[[^\]]*\]\s*)*)(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;")
_ANY_MOD = re.compile(r"^[ \t]*((?:#\[[^\]]*\][ \t]*\n?[ \t]*)*)(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;", re.M)
_PATH_ATTR = re.compile(r'#\[\s*path\s*=\s*"([^"]+)"\s*\]')
_USE_STMT = re.compile(r"(?:^|(?<=[;{}]))\s*(?:pub(?:\([^)]*\))?\s+)?use\s[^;]*;", re.S)


class SrcFile:
    """One source file with its comment / string spans and (for Rust) its test regions."""

    def __init__(self, rel: str, text: str):
        self.rel = rel
        self.text = text
        self.lines = text.split("\n")
        self.starts = [0]
        for ln in self.lines[:-1]:
            self.starts.append(self.starts[-1] + len(ln) + 1)
        self.spans: list[tuple[int, int, str]] = []   # (start, end, 'c' comment | 's' string), sorted
        self.span_starts: list[int] = []
        self.match: dict[int, int] = {}              # code '{' -> its '}'
        self.test_ranges: list[tuple[int, int]] = []  # 1-based inclusive line ranges compiled only under test
        self.whole_file_test = False
        self._use_spans: list[tuple[int, int]] | None = None
        if rel.endswith(".rs"):
            self._scan_rs()
        else:
            self._scan_hash()
        self.span_starts = [s for s, _, _ in self.spans]

    # ---- scanning ----------------------------------------------------------------------------
    def _scan_hash(self) -> None:
        for i, ln in enumerate(self.lines):
            q = None
            for j, ch in enumerate(ln):
                if q:
                    if ch == q:
                        q = None
                    continue
                if ch in "\"'":
                    q = ch
                elif ch == "#" and (j == 0 or ln[j - 1] in " \t"):
                    self.spans.append((self.starts[i] + j, self.starts[i] + len(ln), "c"))
                    break

    def _scan_rs(self) -> None:
        t, n, pos, stack = self.text, len(self.text), 0, []
        while True:
            m = _RS_TOK.search(t, pos)
            if not m:
                break
            s, tok = m.start(), m.group(0)
            if tok.startswith("//"):
                self.spans.append((s, m.end(), "c"))
                pos = m.end()
            elif tok == "/*":
                depth, j = 1, n
                for mm in _BLOCK.finditer(t, m.end()):
                    depth += 1 if mm.group(0) == "/*" else -1
                    if depth == 0:
                        j = mm.end()
                        break
                self.spans.append((s, j, "c"))
                pos = j
            elif tok == "{":
                stack.append(s)
                pos = m.end()
            elif tok == "}":
                if stack:
                    self.match[stack.pop()] = s
                pos = m.end()
            elif tok.endswith('"'):
                if "r" in tok:
                    h = tok.count("#")
                    e = t.find('"' + "#" * h, m.end())
                    j = n if e < 0 else e + 1 + h
                else:
                    mm = _STR_END.match(t, m.end())
                    j = mm.end() if mm else n
                self.spans.append((s, j, "s"))
                pos = j
            else:  # char literal
                self.spans.append((s, m.end(), "s"))
                pos = m.end()
        self.span_starts = [s for s, _, _ in self.spans]
        # test regions
        for m in _TEST_ATTR.finditer(t):
            if self.cls(m.start()) is not None:
                continue
            if m.group(0).startswith("#!"):
                if _INNER_TEST.match(m.group(0)):
                    self.whole_file_test = True
                continue
            end = self._item_end(m.end())
            self.test_ranges.append((self.line_of(m.start()), self.line_of(end)))
        self.test_ranges.sort()

    def _item_end(self, j: int) -> int:
        t, n, depth = self.text, len(self.text), 0
        while j < n:
            k = self._span_end(j)
            if k != j:
                j = k
                continue
            ch = t[j]
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif depth <= 0 and ch == ";":
                return j
            elif depth <= 0 and ch == "{":
                return self.match.get(j, n - 1)
            j += 1
        return n - 1

    def _span_end(self, pos: int) -> int:
        i = bisect.bisect_right(self.span_starts, pos) - 1
        if i >= 0 and self.spans[i][0] <= pos < self.spans[i][1]:
            return self.spans[i][1]
        return pos

    # ---- queries -------------------------------------------------------------------------------
    def cls(self, pos: int) -> str | None:
        """'c' inside a comment, 's' inside a string / char literal, None in code."""
        i = bisect.bisect_right(self.span_starts, pos) - 1
        if i >= 0 and self.spans[i][0] <= pos < self.spans[i][1]:
            return self.spans[i][2]
        return None

    def line_of(self, pos: int) -> int:
        return bisect.bisect_right(self.starts, pos)

    def in_test(self, line: int) -> bool:
        if self.whole_file_test:
            return True
        for a, b in self.test_ranges:
            if a > line:
                return False
            if a <= line <= b:
                return True
        return False

    def in_use_stmt(self, pos: int) -> bool:
        if self._use_spans is None:
            self._use_spans = [(m.start(), m.end()) for m in _USE_STMT.finditer(self.text)
                               if self.cls(m.start() + len(m.group(0)) - len(m.group(0).lstrip())) is None]
        return any(a <= pos < b for a, b in self._use_spans)


class SourceIndex:
    """The Rust tree under <tree>/crates, with test-only files (by name, by directory, by an
    `#[cfg(test)] mod x;` declaration or an inner `#![cfg(test)]`, transitively) told apart."""

    def __init__(self, tree: str):
        self.tree = tree
        self._raw: dict[str, str] = {}
        self._files: dict[str, SrcFile] = {}
        self._rs: list[str] | None = None
        self._test_only: set[str] | None = None

    def rs_files(self) -> list[str]:
        if self._rs is None:
            out = []
            for dp, dn, fns in os.walk(os.path.join(self.tree, "crates")):
                dn[:] = [d for d in dn if d not in ("target", ".git")]
                for f in fns:
                    if f.endswith(".rs"):
                        out.append(os.path.relpath(os.path.join(dp, f), self.tree))
            self._rs = sorted(out)
        return self._rs

    def raw(self, rel: str) -> str | None:
        if rel not in self._raw:
            try:
                self._raw[rel] = read(os.path.join(self.tree, rel))
            except OSError:
                return None
        return self._raw[rel]

    def file(self, rel: str) -> SrcFile | None:
        if rel not in self._files:
            t = self.raw(rel)
            if t is None:
                return None
            self._files[rel] = SrcFile(rel, t)
        return self._files[rel]

    @staticmethod
    def _children(rel: str, name: str, attrs: str) -> list[str]:
        d, b = os.path.dirname(rel), os.path.basename(rel)
        pm = _PATH_ATTR.search(attrs or "")
        if pm:
            return [os.path.normpath(os.path.join(d, pm.group(1)))]
        base = d if b in ("mod.rs", "lib.rs", "main.rs") else os.path.join(d, b[:-3])
        return [os.path.join(base, name + ".rs"), os.path.join(base, name, "mod.rs")]

    def test_only(self) -> set[str]:
        if self._test_only is None:
            files = set(self.rs_files())
            tst = set()
            for rel in files:
                parts = rel.split("/")
                if any(p in TEST_DIRS for p in parts[:-1]) or TEST_FILE_RX.search(parts[-1]):
                    tst.add(rel)
            for rel in files:
                t = self.raw(rel) or ""
                if "cfg" not in t or "test" not in t:
                    continue
                if _INNER_TEST.search(t[:4000]):
                    tst.add(rel)
                for m in _CFG_TEST_MOD.finditer(t):
                    for ch in self._children(rel, m.group(2), m.group(1)):
                        if ch in files:
                            tst.add(ch)
            frontier = list(tst)
            while frontier:
                rel = frontier.pop()
                for m in _ANY_MOD.finditer(self.raw(rel) or ""):
                    for ch in self._children(rel, m.group(2), m.group(1)):
                        if ch in files and ch not in tst:
                            tst.add(ch)
                            frontier.append(ch)
            self._test_only = tst
        return self._test_only

    def is_prod(self, rel: str) -> bool:
        return rel.endswith(".rs") and rel not in self.test_only()

    def prod_uses(self, ident: str, limit: int = 5) -> list[str]:
        """Production code occurrences of `ident` that are neither its definition nor an import."""
        rx = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(ident) + r"(?![A-Za-z0-9_])")
        defn = re.compile(r"\b(?:fn|const|static|struct|enum|trait|type|mod)\s+" + re.escape(ident) + r"\b"
                          r"|^\s*(?:unsafe\s+)?impl\b[^{]*\b" + re.escape(ident) + r"\b")
        hits = []
        for rel in self.rs_files():
            t = self.raw(rel)
            if not t or ident not in t or not self.is_prod(rel):
                continue
            sf = self.file(rel)
            for m in rx.finditer(t):
                if sf.cls(m.start()) is not None:
                    continue
                ln = sf.line_of(m.start())
                if sf.in_test(ln) or defn.search(sf.lines[ln - 1]) or sf.in_use_stmt(m.start()):
                    continue
                hits.append(f"{rel}:{ln}")
                if len(hits) >= limit:
                    return hits
        return hits


def retired_key_findings(path: str, axes: "dict | list[dict] | None", refusal: str | None,
                         refusal_line: int | None, bridge_line: int | None, ledger_row: bool) -> list[str]:
    """What is WRONG with a key the registry records as RETIRED (`retired: {refusal: ...}`): a retired typed
    key stays in the preset deserializer only so an old yaml is REFUSED with its reason, so it must be refused,
    never delivered. Each finding is a hard S4 miss; an empty list means the retirement holds.

    `axes` is EVERY space.rs Axis literal that names the key's lever OR delivers the key's yaml path (a single
    dict is accepted as a one-element list). Fail closed: ANY `Deliver::PresetKey` among them is a delivery,
    whichever array it sits in and whichever yaml path it names, so an early `EnvOnly` literal cannot mask a
    later delivering one and a re-promotion under another spelling is still caught."""
    out = []
    listed = [axes] if isinstance(axes, dict) else list(axes or [])
    delivering = [a for a in listed if a and (a.get("deliver") == "PresetKey" or a.get("key") == path)]
    if delivering:
        where = ", ".join(f"{a.get('name', '?')}@{a.get('line', '?')}->{a.get('deliver')}({a.get('key')})"
                          for a in delivering)
        out.append(f"RETIRED key is still a space.rs Deliver::PresetKey axis (still delivered: {where})")
    if not refusal:
        out.append("RETIRED key records no `retired.refusal` string (preset validation must refuse it by name)")
    elif refusal_line is None:
        out.append(f"RETIRED key has no production refusal '{refusal}' in crates/ny-cli/src/preset/apply.rs "
                   "(a retired key must be refused, not silently ignored)")
    if bridge_line is not None:
        out.append("RETIRED key is still bridged (crates/ny-cli/src/preset_gates.rs writes its cell)")
    if ledger_row:
        out.append("RETIRED key still has a preset_unarmed_promotions.tsv row (it is no longer a promotion)")
    return out


def find_code_line(sf: SrcFile | None, needle: str | None = None, pattern: str | None = None,
                   allow_string: bool = True) -> int | None:
    """First line where the needle / pattern matches OUTSIDE a comment and outside test code
    (inside a string literal only when allow_string). No comment fallback: a match that exists only
    in a comment is absent."""
    if sf is None:
        return None
    rx = re.compile(pattern if pattern is not None else re.escape(needle))
    for m in rx.finditer(sf.text):
        c = sf.cls(m.start())
        if c == "c" or (c == "s" and not allow_string):
            continue
        ln = sf.line_of(m.start())
        if sf.rel.endswith(".rs") and sf.in_test(ln):
            continue
        return ln
    return None


# --------------------------------------------------------------------------------------------
# source test index: every #[test] fn in the tree, so a renamed test cannot pass by absence
# --------------------------------------------------------------------------------------------

def index_source_tests(tree: str) -> dict[str, dict]:
    tests: dict[str, dict] = {}
    fn_rx = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)\s*[(<]")
    for dirpath, dirnames, filenames in os.walk(os.path.join(tree, "crates")):
        dirnames[:] = [d for d in dirnames if d not in ("target", ".git")]
        for fn in filenames:
            if not fn.endswith(".rs"):
                continue
            path = os.path.join(dirpath, fn)
            text = read(path)
            if "#[test]" not in text and "#[tokio::test]" not in text:
                continue
            lines = text.split("\n")
            for i, line in enumerate(lines):
                m = fn_rx.match(line)
                if not m:
                    continue
                # walk up over the attribute / doc-comment run directly above the fn (multi-line
                # attributes included: an unbalanced `]` means we are still inside one)
                attrs, bal, j = [], 0, i - 1
                while j >= 0:
                    s_ = lines[j].strip()
                    bal += s_.count("]") - s_.count("[")
                    if s_.startswith(("#[", "///", "//")) or s_ == "" or bal > 0:
                        attrs.append(s_)
                        j -= 1
                        continue
                    break
                a = "\n".join(attrs)
                if "#[test]" in a or "#[tokio::test]" in a:
                    tests.setdefault(m.group(1), {"file": os.path.relpath(path, tree), "line": i + 1,
                                                  "ignored": "#[ignore" in a or "ignore =" in a,
                                                  "dups": 0})
                    tests[m.group(1)]["dups"] += 1
    return tests


# --------------------------------------------------------------------------------------------
# gate runs: one run per leg prefix, with the HEAD of the heartbeat block that reported it
# --------------------------------------------------------------------------------------------

@dataclass
class GateRun:
    log: str
    prefix: str
    head: str
    ts: str
    done: str | None
    dirty: str | None = None
    legs: list[str] = field(default_factory=list)
    status: dict[str, str] = field(default_factory=dict)      # test name -> ok|FAILED|ignored
    device_ok: dict[str, bool] = field(default_factory=dict)  # ok seen with output visible and no SKIP
    doctests: dict[str, dict[str, int]] = field(default_factory=dict)  # file -> {ok, FAILED, ignored}
    on_main: bool | None = None
    excluded: str | None = None

    @property
    def label(self) -> str:
        return f"{self.prefix}@{self.head[:9]}"


def parse_gate_log(path: str, pending: list[dict]) -> list[GateRun]:
    text = read(path).split("\n")
    blocks: list[list[str]] = []
    for line in text:
        if "heartbeat start" in line and line.startswith("==="):
            blocks.append([line])
        elif blocks:
            blocks[-1].append(line)
    runs: list[GateRun] = []
    leg_rx = re.compile(r"^\s*\[([a-z0-9]+)_([a-z0-9]+)\] (?:rc=\d+ )?passed=")
    for blk in blocks:
        head, dirty = "", None
        for line in blk:
            m = re.match(r"^\s*HEAD: ([0-9a-f]{7,40})", line)
            if m:
                head = m.group(1)
                dm = re.search(r"\bdirty=(\S+)", line)
                dirty = dm.group(1) if dm else None
                break
        ts_m = re.search(r"heartbeat start (\S+)", blk[0])
        done = next((l.strip() for l in blk if re.search(r"^[A-Z0-9]+_DONE ", l.strip())), None)
        prefixes = []
        for line in blk:
            m = leg_rx.match(line)
            if m and m.group(1) not in prefixes:
                prefixes.append(m.group(1))
        for pfx in prefixes:
            runs.append(GateRun(log=path, prefix=pfx, head=head, ts=ts_m.group(1) if ts_m else "",
                                done=done, dirty=dirty))
        if not prefixes:
            pending.append({"log": path, "head": head, "ts": ts_m.group(1) if ts_m else "",
                            "done": done, "last": next((l.strip() for l in reversed(blk) if l.strip()), "")})
    return runs


def attach_legs(runs: list[GateRun]) -> None:
    """Leg files are named <prefix>_<leg>.txt beside the log and are OVERWRITTEN by a re-run, so only the
    newest block reporting a prefix (per directory) owns them; a leg older than its block is stale."""
    newest: dict[tuple[str, str], GateRun] = {}
    for r in runs:
        k = (os.path.dirname(r.log), r.prefix)
        if k not in newest or (parse_ts(r.ts) or 0) >= (parse_ts(newest[k].ts) or 0):
            newest[k] = r
    for r in runs:
        if r.dirty not in (None, "0", "false"):
            r.excluded = f"dirty={r.dirty}: the gate HEAD does not describe what was tested"
            continue
        if newest[(os.path.dirname(r.log), r.prefix)] is not r:
            r.excluded = "superseded: a newer block with the same leg prefix overwrote its leg files"
            continue
        start = parse_ts(r.ts)
        d = os.path.dirname(r.log)
        legs = sorted(p for p in glob.glob(os.path.join(d, f"{r.prefix}_*.txt")) if not LEG_EXCLUDE.search(p))
        stale = [p for p in legs if start is not None and os.path.getmtime(p) < start]
        r.legs = [p for p in legs if p not in stale]
        if stale:
            r.excluded = None
            r.done = (r.done or "") + f" [stale legs ignored: {', '.join(os.path.basename(p) for p in stale)}]"
        for leg in r.legs:
            parse_leg(leg, r)


def parse_leg(path: str, run_: GateRun) -> None:
    raw = read(path)
    lines = [ANSI.sub("", l) for l in raw.split("\n")]
    cmd = next((l for l in lines if l.startswith("[ny-box] cmd:")), "")
    visible = "--nocapture" in cmd or "--show-output" in cmd
    skipped = set(re.findall(r"(\w+): SKIP=VOID", raw))
    for i, line in enumerate(lines):
        if not line.startswith("test ") or line.startswith("test result"):
            continue
        dm = DOCTEST_LINE.match(line)
        if dm:
            st = verdict_from(dm.group(3), lines, i)
            if st:
                run_.doctests.setdefault(dm.group(1), {}).setdefault(st, 0)
                run_.doctests[dm.group(1)][st] += 1
            continue
        m = TEST_LINE.match(line)
        if not m:
            continue
        name = m.group(1).split("::")[-1]
        st = verdict_from(m.group(2), lines, i)
        if st is None:
            continue
        prev = run_.status.get(name)
        # FAILED wins over ok within one run (a same-named test in two binaries)
        if prev == "FAILED":
            continue
        run_.status[name] = st
        if st == "ok":
            run_.device_ok[name] = visible and name not in skipped


def verdict_from(tail: str, lines: list[str], i: int) -> str | None:
    """The verdict on the test's own line, or on a BARE `ok` / `FAILED` line before the next test line
    (interleaved stderr). A line that merely ends in ' ok' is not a verdict."""
    t = tail.strip()
    if t == "ok":
        return "ok"
    if t == "FAILED":
        return "FAILED"
    if t == "ignored" or t.startswith("ignored, "):
        return "ignored"
    for j in range(i + 1, min(i + 400, len(lines))):
        s = lines[j].strip()
        if s.startswith("test "):
            break
        if s == "ok":
            return "ok"
        if s == "FAILED":
            return "FAILED"
    return None


# --------------------------------------------------------------------------------------------
# pins
# --------------------------------------------------------------------------------------------

class Pin:
    def __init__(self, path: str | None):
        self.path = path
        self.lines: list[str] | None = None
        self.sha = None
        m = re.search(r"_([0-9a-f]{7,40})$", os.path.basename(path or ""))
        self.commit = m.group(1) if m else None
        if path and os.path.exists(path):
            p = run(["strings", "-a", path])
            if p.returncode == 0:
                self.lines = p.stdout.split("\n")
            sha_p = run(["sha256sum", path])
            if sha_p.returncode == 0:
                self.sha = sha_p.stdout.split()[0][:16]

    @property
    def name(self) -> str:
        return os.path.basename(self.path) if self.path else "-"

    def count(self, marker: str) -> int | None:
        if self.lines is None:
            return None
        return sum(1 for l in self.lines if marker in l)


# --------------------------------------------------------------------------------------------
# configs: yaml arming, resolved per SCORED category exactly like `resolve_preset_path`
# (crates/ny-cli/src/commands/vnncomp.rs): year directories newest first; within one, the full
# lower-cased category name, then the name without a trailing _20NN.
# --------------------------------------------------------------------------------------------

REQUIRED_YEARS = ["vnncomp24", "vnncomp25", "vnncomp26"]


def strip_year_suffix(s: str) -> str:
    return s[:-5] if re.search(r"_20\d\d$", s) else s


def preset_candidates(category: str) -> list[str]:
    lower = category.lower()
    base = strip_year_suffix(lower)
    return [lower] if base == lower else [lower, base]


class Configs:
    def __init__(self, tree: str, categories: list[str], errors: list[str]):
        self.tree = tree
        cdir = os.path.join(tree, "configs")
        self.years = sorted((d for d in os.listdir(cdir) if d.startswith("vnncomp")
                             and os.path.isdir(os.path.join(cdir, d))), reverse=True) if os.path.isdir(cdir) else []
        for y in REQUIRED_YEARS:
            if y not in self.years or not glob.glob(os.path.join(cdir, y, "*.yaml")):
                errors.append(f"configs/{y} missing or empty: arming cannot be derived")
        self.files: list[tuple[str, str, dict]] = []  # (year, name, data)
        for y in self.years:
            for p in sorted(glob.glob(os.path.join(cdir, y, "*.yaml"))):
                try:
                    data = yaml.safe_load(read(p))
                except yaml.YAMLError as exc:
                    errors.append(f"configs/{y}/{os.path.basename(p)} does not parse ({str(exc)[:80]}): "
                                  "its armings cannot be derived")
                    data = None
                if data is not None and not isinstance(data, dict):
                    errors.append(f"configs/{y}/{os.path.basename(p)} is not a mapping")
                    data = None
                self.files.append((y, os.path.basename(p), data or {}))
        self.categories = sorted(set(categories))
        self.resolved: dict[str, list[str]] = {}   # configs/<y>/<name> -> scored categories loading it
        self.resolution: dict[str, str | None] = {}
        for cat in self.categories:
            f = self.resolve(cat)
            self.resolution[cat] = f
            if f:
                self.resolved.setdefault(f, []).append(cat)

    def resolve(self, category: str) -> str | None:
        for y in self.years:
            for cand in preset_candidates(category):
                if os.path.isfile(os.path.join(self.tree, "configs", y, cand + ".yaml")):
                    return f"configs/{y}/{cand}.yaml"
        return None

    def placebo_reason(self, year: str, name: str) -> str | None:
        """None when a scored category loads this file; otherwise why none does."""
        f = f"configs/{year}/{name}"
        if f in self.resolved:
            return None
        stem = name[:-5]
        wanting = [c for c in self.categories if stem in preset_candidates(c)]
        if wanting:
            return "shadowed by " + ", ".join(sorted({f"{self.resolution[c]} for {c}" for c in wanting}))
        return "no scored category resolves to it"

    @staticmethod
    def lookup(data: dict, dotted: str):
        node = data
        for part in dotted.split("."):
            if not isinstance(node, dict):
                return None
            cands = [part, part.replace("-", "_"), part.replace("_", "-")]
            hit = next((c for c in cands if c in node), None)
            if hit is None:
                return None
            node = node[hit]
        return node

    def arming(self, key: dict) -> list[dict]:
        out = []
        paths = [key["path"]] + list(key.get("alt_paths", []))
        default = key.get("default")
        for year, name, data in self.files:
            for p in paths:
                v = self.lookup(data, p)
                if v is None:
                    continue
                armed = not equal_default(v, default)
                canary = "_canary" in name
                out.append({"file": f"configs/{year}/{name}", "path": p, "value": v, "armed": armed,
                            "canary": canary, "shadowed_by": None if canary else self.placebo_reason(year, name),
                            "categories": self.resolved.get(f"configs/{year}/{name}", [])})
        return out


def equal_default(v, default) -> bool:
    if isinstance(v, bool) or isinstance(default, bool):
        return bool(v) == bool(default) if isinstance(v, bool) and isinstance(default, bool) else v == default
    try:
        return float(v) == float(default)
    except (TypeError, ValueError):
        return str(v) == str(default)


def key_armed_in(data: dict, key: dict) -> bool:
    for p in [key["path"]] + list(key.get("alt_paths", [])):
        v = Configs.lookup(data, p)
        if v is not None and not equal_default(v, key.get("default")):
            return True
    return False


# --------------------------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------------------------

@dataclass
class StageResult:
    state: str
    note: str = ""


class Auditor:
    def __init__(self, reg: dict, args):
        self.reg = reg
        self.args = args
        self.tree = os.path.abspath(args.tree)
        self.scratch = args.scratch_root or reg.get("scratch_root", "")
        self.errors: list[str] = []
        self.pending: list[dict] = []
        if not self.scratch or not os.path.isdir(self.scratch):
            self.errors.append(f"scratch root missing: {self.scratch!r} (every evidence path resolves against it)")
        self.git = Git(self.tree)
        self.main_ref = args.main_ref or reg.get("defaults", {}).get("main_ref", "origin/main")
        self.main_sha = self.git.sha(self.main_ref)
        if self.main_sha is None:
            self.errors.append(f"main ref {self.main_ref} does not resolve in {self.tree}")
        if args.check_remote and self.main_sha:
            self._check_remote()
        d = reg.get("defaults", {})
        pins = d.get("pins", {})
        self.pin = Pin(self.path(args.pin or pins.get("latest")))
        self.pin_shipped = Pin(self.path(args.shipped_pin or pins.get("shipped")))
        self.pin_x86 = Pin(self.path(args.x86_pin or pins.get("x86")))
        for label, pin in (("latest", self.pin), ("shipped", self.pin_shipped), ("x86", self.pin_x86)):
            if pin.lines is None:
                self.errors.append(f"{label} pin unreadable: {pin.path} (its marker counts cannot be derived)")
        if self.pin.commit and not self.git.sha(self.pin.commit):
            self.errors.append(f"latest pin commit {self.pin.commit} (from its name) is not in this repository")
        gate_logs = [] if args.only_gates else list(d.get("gate_logs", []))
        gate_logs += args.gate_log or []
        self.runs: list[GateRun] = []
        for gl in gate_logs:
            p = self.path(gl)
            if not p or not os.path.exists(p):
                self.errors.append(f"gate log missing: {p}")
                continue
            self.runs.extend(parse_gate_log(p, self.pending))
        attach_legs(self.runs)
        self.excluded_runs = [r for r in self.runs if r.excluded]
        self.runs = [r for r in self.runs if not r.excluded]
        for r in self.runs:
            r.on_main = self.git.is_ancestor(r.head, self.main_ref) if r.head else None
        self.runs.sort(key=lambda r: r.ts)
        self.src = SourceIndex(self.tree)
        if not os.path.isdir(os.path.join(self.tree, "crates")):
            self.errors.append(f"{self.tree}/crates missing: no source to audit")
        self.src_tests = index_source_tests(self.tree)
        self.configs = Configs(self.tree, self._categories(d), self.errors)
        self.keymap = {k["path"]: k for c in reg["capabilities"] for k in c.get("keys", [])}
        self.space = self._parse_space()
        self.decls = self._parse_decls()
        self.ledger = self._parse_tsv_col("crates/ny-levers/tests/data/preset_unarmed_promotions.tsv", 0)
        self.frontier = self._parse_frontier()
        self.results: dict[str, dict] = {}
        self._seal_records: dict[str, dict] = {}
        self._cur_runs: list[GateRun] = []
        self.used_prefixes: set[str] = set()
        self._census = self.census()

    def _check_remote(self) -> None:
        remote, _, branch = self.main_ref.partition("/")
        p = run(["git", "-C", self.tree, "ls-remote", remote, f"refs/heads/{branch}"])
        live = p.stdout.split()[0] if p.returncode == 0 and p.stdout.strip() else None
        if live is None:
            self.errors.append(f"--check-remote: ls-remote {remote} {branch} failed: {p.stderr.strip()[:120]}")
        elif live != self.main_sha:
            self.errors.append(f"--check-remote: local {self.main_ref}={self.main_sha[:9]} but the remote is at "
                               f"{live[:9]}: fetch before trusting S2/S3")

    def _categories(self, d: dict) -> list[str]:
        roots = d.get("benchmark_roots", [])
        cats: list[str] = []
        if not roots:
            self.errors.append("registry defaults.benchmark_roots is empty: the scored categories (and so the "
                               "live yaml armings) cannot be derived")
        for r in roots:
            p = self.path(r)
            subs = sorted(x for x in os.listdir(p) if os.path.isdir(os.path.join(p, x))) if p and os.path.isdir(p) else []
            if not subs:
                self.errors.append(f"benchmark root missing or empty: {p}")
            cats += subs
        return cats

    # ---- paths -----------------------------------------------------------------------------
    def path(self, p: str | None) -> str | None:
        if not p:
            return None
        if os.path.isabs(p):
            return p
        return os.path.join(self.scratch, p)

    def tpath(self, rel: str) -> str:
        return os.path.join(self.tree, rel)

    # ---- source indexes ----------------------------------------------------------------------
    def _parse_space(self) -> dict[str, dict]:
        """NY name -> {deliver key, line} for every Axis literal in space.rs outside test code (the FIRST literal
        per name). `self.space_all` keeps EVERY literal per name, so a retired key's reverse audit cannot be
        satisfied by an early EnvOnly literal while a later one still delivers it."""
        sf = self.src.file("crates/ny-levers/src/space.rs")
        out: dict[str, dict] = {}
        self.space_all: dict[str, list[dict]] = {}
        if sf is None:
            self.errors.append("space.rs missing")
            return out
        text = sf.text
        for m in re.finditer(r"(?<![A-Za-z0-9_])Axis\s*\{", text):
            if sf.cls(m.start()) is not None or sf.in_test(sf.line_of(m.start())):
                continue
            body = text[m.end(): m.end() + 4000]
            nm = re.search(r'name:\s*"(NY_[A-Z0-9_]+)"', body)
            if not nm:
                continue
            nxt = re.search(r"Axis\s*\{", body)
            if nxt and nxt.start() < nm.start():
                continue
            scope = body[: nxt.start()] if nxt else body
            dm = re.search(r'deliver:\s*Deliver::(\w+)(?:\("([^"]+)"\))?', scope)
            line = text.count("\n", 0, m.end() + nm.start()) + 1
            entry = {"deliver": dm.group(1) if dm else None, "key": dm.group(2) if dm else None, "line": line}
            out.setdefault(nm.group(1), entry)
            self.space_all.setdefault(nm.group(1), []).append(dict(entry, name=nm.group(1)))
        if not out:
            self.errors.append("space.rs parsed to 0 axes: the Axis literal shape changed (S4 cannot be derived)")
        return out

    def _parse_decls(self) -> dict[str, str]:
        out: dict[str, str] = {}
        paths = sorted(glob.glob(self.tpath("crates/ny-levers/src/decls/*.rs")))
        if not paths:
            self.errors.append("crates/ny-levers/src/decls/*.rs missing: LeverDecls cannot be derived")
        for p in paths:
            sf = self.src.file(os.path.relpath(p, self.tree))
            for i, line in enumerate(sf.lines, 1):
                m = re.search(r'^\s*name:\s*"(NY_[A-Z0-9_]+)"', line)
                if m and not sf.in_test(i):
                    out.setdefault(m.group(1), f"{sf.rel}:{i}")
        return out

    def _parse_tsv_col(self, rel: str, col: int) -> set[str]:
        out = set()
        try:
            for line in read(self.tpath(rel)).split("\n"):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) > col:
                    out.add(parts[col])
        except OSError:
            self.errors.append(f"missing {rel}")
        return out

    def _parse_frontier(self) -> dict[str, int]:
        out: dict[str, int] = {}
        try:
            for line in read(self.tpath("crates/ny-levers/tests/data/ny_literal_frontier.tsv")).split("\n"):
                parts = line.split("\t")
                if len(parts) >= 4 and parts[3].startswith("NY_"):
                    out[parts[3]] = out.get(parts[3], 0) + 1
        except OSError:
            self.errors.append("missing ny_literal_frontier.tsv")
        if not out:
            self.errors.append("ny_literal_frontier.tsv parsed to 0 rows: frontier coverage cannot be derived")
        return out

    # ---- per capability ------------------------------------------------------------------------
    def audit_cap(self, cap: dict) -> dict:
        res: dict = {"stages": {}, "notes": {}, "flags": []}
        rec = cap.get("record", {})
        ev = cap.get("evidence", {}) or {}
        heads = cap.get("heads", [])

        # S1 ---------------------------------------------------------------------------------
        res["stages"]["S1"] = self.audit_review(cap, heads)

        # S2 ---------------------------------------------------------------------------------
        if heads:
            anc = {h: self.git.is_ancestor(h, self.main_ref) for h in heads}
            missing = [h for h, a in anc.items() if a is None]
            if missing:
                res["stages"]["S2"] = StageResult("open", f"unknown commit(s) {missing}")
            elif all(anc.values()):
                res["stages"]["S2"] = StageResult("done", f"{', '.join(heads)} on {self.main_ref}@{(self.main_sha or '?')[:9]}")
            elif any(anc.values()):
                res["stages"]["S2"] = StageResult("partial", "on main: " + ", ".join(h for h, a in anc.items() if a)
                                                  + "; NOT on main: " + ", ".join(h for h, a in anc.items() if not a))
            else:
                res["stages"]["S2"] = StageResult("open", f"none of {', '.join(heads)} is on {self.main_ref}")
        else:
            res["stages"]["S2"] = StageResult("open", "no heads recorded")

        # S3 ---------------------------------------------------------------------------------
        res["stages"]["S3"], res["tests"] = self.audit_tests(cap, heads)

        # S4 ---------------------------------------------------------------------------------
        res["stages"]["S4"], res["keys"], res["consumers"], extra = self.audit_delivery(cap)
        res.update(extra)

        # markers in pins -------------------------------------------------------------------
        res["markers"] = []
        censused = self._census["censused"]
        pin_has_cap = bool(heads) and self.pin.commit is not None and all(
            self.git.is_ancestor(h, self.pin.commit) for h in heads)
        for mk in cap.get("markers", []):
            latest = self.pin.count(mk)
            res["markers"].append({"marker": mk, "latest": latest,
                                   "shipped": self.pin_shipped.count(mk), "x86": self.pin_x86.count(mk),
                                   # the gate proves a marker only by grepping a string that CONTAINS it
                                   "censused": None if censused is None else any(mk in c for c in censused)})
            if pin_has_cap and latest == 0:
                res["flags"].append(f"marker '{mk}' is absent from {self.pin.name} although its commit contains "
                                    "the capability: its code is not linked (dead) or the marker spans a format "
                                    "placeholder")

        # S5 ---------------------------------------------------------------------------------
        res["stages"]["S5"] = self.audit_arrival(cap, res["markers"])
        if (ev.get("S5") or {}).get("seal"):
            sr = self.audit_seal_stage(cap, "S5", ev["S5"])
            a5 = res["stages"]["S5"]
            res["stages"]["S5"] = StageResult(min((a5.state, sr.state), key=lambda s: RANK[s]),
                                              a5.note + "; " + sr.note)

        # S6-S8, S10, S11 ----------------------------------------------------------------------
        for st in ("S6", "S7", "S8", "S10", "S11"):
            if st in ("S6", "S7", "S8") and (ev.get(st) or {}).get("seal") and rec.get(st, "open") != "na":
                res["stages"][st] = self.audit_seal_stage(cap, st, ev[st])
            else:
                res["stages"][st] = self.audit_evidence(cap, st, ev.get(st), rec.get(st, "open"))

        # SR (routable) ------------------------------------------------------------------------
        res["stages"]["SR"] = self.audit_routable(cap, res["stages"])

        # S9 (inheriting capabilities are resolved after their parents) -----------------------------
        res["stages"]["S9"], res["arming"] = self.audit_arming(cap)
        self._finish(cap, res)
        return res

    def _finish(self, cap: dict, res: dict) -> None:
        rec = cap.get("record", {})
        res["flags"] = [f for f in res["flags"] if not f.startswith(("REGRESSION", "improved"))]
        for st in STAGES:
            want = rec.get(st, "na" if st == "SR" and not cap.get("routable") else "open")
            got = res["stages"][st].state
            if st == "SR" and want in SR_LADDER:
                # review(eng11): a router SR-ladder level (decl / obs / seal) is compared on the one bit this
                # auditor derives, SR.seal; decl and obs are recorded facts it does not re-derive.
                if SR_LADDER[want] == "done" and got != "done":
                    res["flags"].append(f"REGRESSION SR: recorded {want}, derived {got}")
                elif SR_LADDER[want] != "done" and got == "done":
                    res["flags"].append(f"improved SR: recorded {want}, derived {got} (record SR: seal)")
                continue
            if want not in RANK:
                self.errors.append(f"{cap['id']}: bad recorded state {st}={want}")
                continue
            if RANK[got] < RANK[want]:
                res["flags"].append(f"REGRESSION {st}: recorded {want}, derived {got}")
            elif RANK[got] > RANK[want]:
                res["flags"].append(f"improved {st}: recorded {want}, derived {got} (update the registry)")
        # review(eng11): the router routes at seal from its COMPILED ladder, which mirrors the record (S7 + S8 done
        # and an envelope: `derive_sr`, cross-checked by router_r5_registry_ladders_match_the_capability_registry),
        # not from the seal records. A recorded ladder that reaches seal where SR.seal does not derive would let the
        # router configure a capability the seal pipeline never sealed: fail closed.
        if cap.get("routable") and derive_sr(cap) == "seal" and res["stages"]["SR"].state != "done":
            res["flags"].append("REGRESSION SR: the router's ladder reaches seal (record S7 + S8 done and an envelope) "
                                f"but SR.seal does not derive ({res['stages']['SR'].note})")
        killed = cap.get("status") == "killed"
        res["ladder"] = "KILLED" if killed else ladder(res["stages"])
        res["highest"] = next((st for st in reversed(STAGES) if res["stages"][st].state == "done"), "S0")
        res["recorded_ladder"] = "KILLED" if killed else ladder(
            {s: StageResult(SR_LADDER.get(rec.get(s, "open"), rec.get(s, "open")) if s == "SR" else rec.get(s, "open"))
             for s in STAGES})

    # ---- S1 -----------------------------------------------------------------------------------
    def audit_review(self, cap: dict, heads: list[str]) -> StageResult:
        rv = cap.get("review")
        if not rv:
            built = all(self.git.sha(h) for h in heads) if heads else False
            return StageResult("partial" if built else "open",
                               "built; no adversarial-review commit recorded" if built else "no head commit")
        sha = self.git.sha(rv["commit"])
        if sha is None:
            return StageResult("partial", f"{rv['commit']} (commit not found)")
        msg = self.git.message(rv["commit"])
        by_msg = "grep" in rv and re.search(rv["grep"], msg, re.I) is not None
        if "grep" not in rv and "doc" not in rv:
            by_msg = re.search(r"\breview", msg, re.I) is not None
        note = f"{rv['commit']} \"{self.git.subject(rv['commit'])[:110]}\""
        by_doc = False
        if rv.get("doc"):
            p = self.path(rv["doc"])
            if p and os.path.exists(p):
                t = read(p)
                by_doc = rv.get("doc_grep", "Verdict") in t and sha[:9] in t
                note += f"; review report {rv['doc']} " + ("names the commit and carries the verdict" if by_doc
                                                          else "does NOT name the commit with the verdict")
            else:
                note += f"; review report {rv['doc']} MISSING"
        return StageResult("done" if (by_msg or by_doc) else "partial", note)

    # ---- S3 -----------------------------------------------------------------------------------
    def audit_tests(self, cap: dict, heads: list[str]):
        entries = []
        for t in cap.get("tests", []):
            if isinstance(t, str):
                t = {"name": t}
            if "glob" in t:
                names = sorted(n for n in self.src_tests if fnmatch.fnmatchcase(n, t["glob"]))
                if len(names) < t.get("min", 1):
                    entries.append({"name": t["glob"], "status": "NOT_IN_SOURCE",
                                    "why": f"{len(names)} source tests match (min {t.get('min', 1)})", "role": t.get("role", "contract")})
                for n in names:
                    entries.append(self._test_status(n, t))
            elif "doctest" in t:
                entries.append(self._doctest_status(t))
            elif "file" in t:
                names = sorted(n for n, v in self.src_tests.items() if v["file"] == t["file"])
                if len(names) < t.get("min", 1):
                    entries.append({"name": t["file"], "status": "NOT_IN_SOURCE", "role": t.get("role", "contract"),
                                    "why": f"{len(names)} #[test] fns in the file (min {t.get('min', 1)})"})
                for n in names:
                    entries.append(self._test_status(n, t))
            else:
                entries.append(self._test_status(t["name"], t))
        contract = [e for e in entries if e.get("role", "contract") == "contract"]
        if not contract:
            return StageResult("na" if not cap.get("tests") else "open", "no contract tests recorded"), entries
        n_ok = sum(1 for e in contract if e["status"] == "ok")
        failed = [e for e in contract if e["status"] in ("FAILED", "NOT_IN_SOURCE", "AMBIGUOUS")]
        ok_on_main = sum(1 for e in contract if e["status"] == "ok" and e.get("on_main"))
        gates = sorted({e["gate"] for e in contract if e.get("gate")})
        mains = sorted({e["main_gate"] for e in contract if e.get("main_gate")})
        note = f"{n_ok}/{len(contract)} ok"
        if failed:
            note += "; FAILED/renamed/ambiguous: " + ", ".join(e["name"] for e in failed[:6])
        unproven = [e for e in contract if e["status"] == "ok-unproven"]
        if unproven:
            note += (f"; {len(unproven)} device test(s) reported ok from a leg that hid their output or printed "
                     "SKIP=VOID (not a pass): " + ", ".join(e["name"] for e in unproven[:3]))
        other = [e for e in contract if e["status"] not in ("ok", "ok-unproven", "FAILED", "NOT_IN_SOURCE", "AMBIGUOUS")]
        if other:
            note += f"; never ok in a gate containing the capability: {len(other)} (" + ", ".join(e["name"] for e in other[:4]) + ("…" if len(other) > 4 else "") + ")"
        note += f"; newest status from: {', '.join(gates) or '-'}; ok in main gate(s): {', '.join(mains) or 'none'}"
        if n_ok and ok_on_main < n_ok:
            note += f"; {ok_on_main}/{len(contract)} ok in a MAIN gate"
        if failed or n_ok == 0:
            state = "open"
        elif n_ok == len(contract) and ok_on_main == len(contract):
            state = "done"
        else:
            state = "partial"
            if n_ok == len(contract):
                note += "; green only in an integration gate (HEAD not on main)"
        return StageResult(state, note), entries

    def _relevant_runs(self, cap: dict):
        heads = cap.get("heads", [])
        rel = []
        for r in self.runs:
            if not r.head:
                continue
            if heads and not all(self.git.is_ancestor(h, r.head) for h in heads):
                continue
            rel.append(r)
        return rel

    def _test_status(self, name: str, t: dict) -> dict:
        src = self.src_tests.get(name)
        e = {"name": name, "role": t.get("role", "contract"), "status": "missing", "gate": None, "on_main": None,
             "src": f"{src['file']}:{src['line']}" if src else None, "ignored_in_source": bool(src and src["ignored"])}
        if src is None:
            e["status"] = "NOT_IN_SOURCE"
            return e
        if src["dups"] > 1:
            # gate lines are keyed by the bare fn name: two #[test] fns with this name cannot be told apart
            e["status"] = "AMBIGUOUS"
            e["why"] = f"{src['dups']} #[test] fns named {name}"
            return e
        device = bool(t.get("device"))

        def st_of(r: GateRun) -> str | None:
            st = r.status.get(name)
            if st == "ok" and device and not r.device_ok.get(name):
                return "ok-unproven"
            return st

        for r in reversed(self._cur_runs):
            st = st_of(r)
            if st in ("ok", "ok-unproven", "FAILED", "ignored"):
                e.update(status=st, gate=r.label)
                break
        e["on_main"] = any(r.on_main and st_of(r) == "ok" for r in self._cur_runs)
        e["main_gate"] = next((r.label for r in reversed(self._cur_runs) if r.on_main and st_of(r) == "ok"), None)
        return e

    def _doctest_status(self, t: dict) -> dict:
        f = t["doctest"]
        e = {"name": f"doctests in {f}", "role": t.get("role", "contract"), "status": "missing", "gate": None, "on_main": None}
        for r in reversed(self._cur_runs):
            d = r.doctests.get(f)
            if d:
                okc = d.get("ok", 0)
                e.update(status="ok" if okc >= t.get("min", 1) and not d.get("FAILED") else "FAILED",
                         gate=r.label, on_main=bool(r.on_main), main_gate=r.label if r.on_main else None,
                         why=json.dumps(d))
                break
        return e

    # ---- S4 -----------------------------------------------------------------------------------
    def _consumer(self, c: dict) -> dict:
        rel = c["file"]
        sf = self.src.file(rel)
        out = {"file": rel, "grep": c.get("grep") or c.get("regex"), "call": c.get("call"), "line": None, "why": None}
        if sf is None:
            out["why"] = "file missing"
            return out
        if rel.endswith(".rs") and not self.src.is_prod(rel):
            out["why"] = "a test-only file"
            return out
        ln = find_code_line(sf, c.get("grep"), c.get("regex"))
        if not ln:
            out["why"] = "no production (non-comment, non-test) line matches"
            return out
        text = sf.lines[ln - 1]
        if c.get("call"):
            uses = self.src.prod_uses(c["call"])
            out["callers"] = uses
            if not uses:
                out["why"] = f"no production caller of `{c['call']}` (test-only or dead)"
                return out
        elif rel.endswith(".rs") and DEF_LINE.match(text):
            out["why"] = "a definition or import, not a use: record `call:` so a production caller is proven"
            return out
        out["line"] = ln
        return out

    def audit_delivery(self, cap: dict):
        keys = cap.get("keys", [])
        consumers = [self._consumer(c) for c in cap.get("consumers", [])]
        cons_ok = bool(consumers) and all(c["line"] for c in consumers)
        miss_note = ", ".join(f"{c['file']} ({c['why']})" for c in consumers if not c["line"])
        krows = [self._key_row(k) for k in keys]
        callers = [self._callers(c) for c in cap.get("callers", [])]
        caller_hard = [c for c in callers if not c["ok"] and c["severity"] == "open"]
        caller_soft = [c for c in callers if not c["ok"] and c["severity"] == "partial"]
        caller_note = "; ".join(f"{c['what']}: {c['found']} production site(s) (need {c['min']})"
                                for c in caller_hard + caller_soft)
        info_note = "; ".join(f"{c['what']}: {c['found']} production site(s)" for c in callers if c["severity"] == "info")
        res_extra = {"callers": callers}
        if not keys:
            if not consumers:
                return StageResult("open", "no key and no consumer recorded"), krows, consumers, res_extra
            if not cons_ok:
                return StageResult("open", "keyless; consumer site MISSING: " + miss_note), krows, consumers, res_extra
            if caller_hard:
                return StageResult("open", "keyless; " + caller_note), krows, consumers, res_extra
            if caller_soft:
                return StageResult("partial", "keyless; " + caller_note), krows, consumers, res_extra
            return StageResult("na", "keyless by design; production consumer sites present"
                               + (f"; {info_note}" if info_note else "")), krows, consumers, res_extra
        hard = [f"{r['path']}: " + ", ".join(r["missing"]) for r in krows if r["missing"]]
        soft = [f"{r['path']}: " + ", ".join(r["warn"]) for r in krows if r["warn"]]
        if all(k.get("retired") for k in keys):
            # A RETIRED key is audited in reverse: refused at preset validation, never delivered.
            if hard:
                return StageResult("open", "; ".join(hard)), krows, consumers, res_extra
            return StageResult("na", f"{len(krows)} RETIRED key(s): refused at preset validation, no PresetKey "
                               "axis, no bridge write, no ledger row"), krows, consumers, res_extra
        if not cons_ok:
            hard.append("consumer site missing: " + miss_note if consumers else "no consumer recorded")
        if caller_hard:
            hard.append(caller_note)
        elif caller_soft:
            soft.append(caller_note)
        governed = all(r["decl"] for r in krows)
        if hard:
            return StageResult("open", "; ".join(hard)), krows, consumers, res_extra
        if not governed:
            return StageResult("partial", "delivered config-direct (BabPreset/AlphaCrownPreset -> apply.rs -> config), "
                               "outside ny-levers governance: no LeverDecl / space.rs axis / ledger row (§1.10)"
                               + ("; " + "; ".join(soft) if soft else "")), krows, consumers, res_extra
        if soft:
            return StageResult("partial", "delivered, but: " + "; ".join(soft)), krows, consumers, res_extra
        return StageResult("done", f"{len(krows)} key(s): LeverDecl + space.rs PresetKey axis + schema + apply "
                           "+ bridge/cell + production consumer"), krows, consumers, res_extra

    def _callers(self, c: dict) -> dict:
        """Production CODE sites matching a regex (never comments, strings or test code). Every entry carries
        a positive `control` that must match somewhere in the tree, so a zero is an absence and not a typo."""
        rx = re.compile(c["regex"])
        excl = [re.compile(x) for x in c.get("exclude", [])]
        root = c.get("root", "crates").rstrip("/") + "/"
        out = {"what": c.get("what", c["regex"]), "regex": c["regex"], "min": c.get("min", 1), "found": 0,
               "hits": [], "ok": False, "severity": c.get("severity", "open"), "control": None}
        if not os.path.isdir(self.tpath(root)):
            self.errors.append(f"callers root missing: {root} ({out['what']})")
            return out
        ctl = c.get("control")
        if not ctl:
            self.errors.append(f"callers '{out['what']}' has no positive control: a zero would be uninterpretable")
        else:
            crx = re.compile(ctl)
            where = next((r for r in self.src.rs_files() if r.startswith(root) and crx.search(self.src.raw(r) or "")), None)
            out["control"] = where
            if where is None:
                self.errors.append(f"callers '{out['what']}': positive control /{ctl}/ matches nothing under {root}: "
                                   "the symbol was renamed or moved, so the count proves nothing")
        hits = []
        for rel in self.src.rs_files():
            if not rel.startswith(root) or not self.src.is_prod(rel):
                continue
            t = self.src.raw(rel) or ""
            if not rx.search(t):
                continue
            sf = self.src.file(rel)
            for i, line in enumerate(sf.lines, 1):
                ms = [m for m in rx.finditer(line) if sf.cls(sf.starts[i - 1] + m.start()) is None]
                if not ms or sf.in_test(i) or any(x.search(line) for x in excl):
                    continue
                hits.append(f"{rel}:{i}")
        out.update(found=len(hits), hits=hits[:8], ok=len(hits) >= out["min"])
        return out

    def _retired_key_row(self, k: dict, row: dict) -> dict:
        path = k["path"]
        leaf = path.split(".")[-1]
        decl = k.get("decl")
        # EVERY Axis literal naming the lever, plus any literal (any lever) delivering the retired yaml path.
        axes = list(self.space_all.get(decl, [])) if decl else []
        axes += [a for lits in self.space_all.values() for a in lits if a.get("key") == path and a not in axes]
        row["decl"] = self.decls.get(decl) if decl else None
        row["axis"] = ", ".join(f"crates/ny-levers/src/space.rs:{a['line']} ({a['deliver']})" for a in axes) or None
        row["frontier_rows"] = self.frontier.get(decl, 0) if decl else None
        row["ledger_row"] = path in self.ledger
        refusal = (k.get("retired") or {}).get("refusal") if isinstance(k.get("retired"), dict) else None
        apply_rel = "crates/ny-cli/src/preset/apply.rs"
        rl = find_code_line(self.src.file(apply_rel), refusal) if refusal else None
        row["apply"] = f"REFUSED {apply_rel}:{rl}" if rl else None
        b = find_code_line(self.src.file("crates/ny-cli/src/preset_gates.rs"),
                           pattern=rf"config\.margin_row\.{re.escape(leaf)}\b", allow_string=False)
        row["bridge"] = f"crates/ny-cli/src/preset_gates.rs:{b}" if b else None
        row["cell"] = None
        schema_field = k.get("schema_field", leaf)
        sfile = k.get("schema_file", "crates/ny-cli/src/preset/mod.rs")
        ln = find_code_line(self.src.file(sfile), pattern=rf"\bpub(\(crate\))?\s+{re.escape(schema_field)}\s*:",
                            allow_string=False)
        row["schema"] = f"{sfile}:{ln} (kept to refuse)" if ln else None
        row["missing"].extend(retired_key_findings(path, axes, refusal, rl, b, row["ledger_row"]))
        return row

    def _key_row(self, k: dict) -> dict:
        path = k["path"]
        leaf = path.split(".")[-1]
        row = {"path": path, "type": k.get("type"), "default": k.get("default"), "route": k.get("route"),
               "missing": [], "warn": []}
        if k.get("retired"):
            return self._retired_key_row(k, row)
        decl = k.get("decl")
        if decl:
            row["decl"] = self.decls.get(decl)
            if not row["decl"]:
                row["missing"].append(f"LeverDecl {decl}")
            ax = self.space.get(decl)
            if ax and ax["deliver"] == "PresetKey" and ax["key"] == path:
                row["axis"] = f"crates/ny-levers/src/space.rs:{ax['line']}"
            else:
                row["axis"] = None
                row["missing"].append("space.rs Deliver::PresetKey axis " + (
                    f"(axis delivers {ax['deliver']}({ax['key']}))" if ax else "(no axis: DARK on scored rows)"))
            row["frontier_rows"] = self.frontier.get(decl, 0)
            if row["frontier_rows"] == 0:
                row["warn"].append("no ny_literal_frontier.tsv rows (source_frontier goes red)")
        else:
            row["decl"] = row["axis"] = row["frontier_rows"] = None
        row["ledger_row"] = path in self.ledger
        # schema: the leaf (or the recorded parent struct field for nested keys)
        schema_field = k.get("schema_field", leaf)
        sfile = k.get("schema_file", "crates/ny-cli/src/preset/mod.rs")
        ln = find_code_line(self.src.file(sfile), pattern=rf"\bpub(\(crate\))?\s+{re.escape(schema_field)}\s*:",
                            allow_string=False)
        row["schema"] = f"{sfile}:{ln}" if ln else None
        if not ln:
            row["missing"].append(f"schema field `{schema_field}`")
        # apply
        apply_rel = "crates/ny-cli/src/preset/apply.rs"
        if decl and k.get("apply", "generic") == "generic":
            gl = find_code_line(self.src.file(apply_rel), "fn validate_delivered_lever_values", allow_string=False)
            vl = find_code_line(self.src.file("crates/ny-cli/src/commands/vnncomp.rs"),
                                "fn preset_delivered_config_value", allow_string=False)
            ok = gl and vl and row["axis"]
            row["apply"] = f"generic axis projection apply.rs:{gl} + vnncomp.rs:{vl}" if ok else None
            if not ok:
                row["missing"].append("apply (the generic projection only sees keys WITH an axis)")
        else:
            needle = k.get("apply") if k.get("apply") not in (None, "generic") else schema_field
            ln = find_code_line(self.src.file(apply_rel), pattern=rf"\b{re.escape(needle)}\b", allow_string=False)
            row["apply"] = f"{apply_rel}:{ln}" if ln else None
            if not ln:
                row["missing"].append(f"apply.rs mapping `{needle}`")
        # bridge + governed cell (every declared margin_row key, §1.10); word-bounded so `bi_patch` never
        # borrows `bi_patch_ratio`'s line
        if decl and path.startswith("margin_row."):
            b = find_code_line(self.src.file("crates/ny-cli/src/preset_gates.rs"),
                               pattern=rf"config\.margin_row\.{re.escape(leaf)}\b", allow_string=False)
            c = find_code_line(self.src.file("crates/ny-propagate/src/preset_gates.rs"),
                               pattern=rf"\bset_margin_row_{re.escape(leaf)}_preset\b", allow_string=False)
            row["bridge"] = f"crates/ny-cli/src/preset_gates.rs:{b}" if b else None
            row["cell"] = f"crates/ny-propagate/src/preset_gates.rs:{c}" if c else None
            load_bearing = k.get("route") == "cell"
            for ok, what in ((b, "ny-cli preset_gates bridge write"), (c, "ny-propagate governed cell")):
                if not ok:
                    (row["missing"] if load_bearing else row["warn"]).append(what)
        else:
            row["bridge"] = row["cell"] = None
        return row

    # ---- S5 -----------------------------------------------------------------------------------
    def audit_arrival(self, cap: dict, markers: list[dict]) -> StageResult:
        rec = (cap.get("record") or {}).get("S5", "open")
        ev = (cap.get("evidence") or {}).get("S5")
        keys = [k["path"] for k in cap.get("keys", [])]
        in_pin = [m for m in markers if m["latest"]]
        pin_note = f"latest pin carries {len(in_pin)}/{len(markers)} of its markers"
        if not ev:
            if rec == "na":
                return StageResult("na", "structural: no runtime observable")
            return StageResult("open", pin_note + "; no armed-yaml, zero-NY_* run on a real row recorded")
        arrivals = ev.get("arrivals", [])
        covered, notes = set(), []
        for a in arrivals:
            files = sorted(set(sum((glob.glob(self.path(g)) for g in a.get("logs", [])), [])))
            hits, armed_keys, gone = [], set(), set()
            for f in files:
                t = read(f)
                if a["marker"] not in t:
                    continue
                if ev.get("keyless"):
                    hits.append(f)
                    continue
                pm = re.search(r"Loading preset: (\S+)", t)
                if not pm or "Running VNN-COMP instance:" not in t:
                    continue
                if not os.path.exists(pm.group(1)):
                    gone.add(pm.group(1))
                    continue
                try:
                    data = yaml.safe_load(read(pm.group(1))) or {}
                except yaml.YAMLError:
                    continue
                these = {kp for kp in a.get("keys", []) if kp in self.keymap and key_armed_in(data, self.keymap[kp])}
                unknown = [kp for kp in a.get("keys", []) if kp not in self.keymap]
                if unknown:
                    self.errors.append(f"{cap['id']}: S5 arrival names unregistered key(s) {unknown}")
                if these:
                    hits.append(f)
                    armed_keys |= these
            covered |= armed_keys
            notes.append(f"'{a['marker']}' in {len(hits)}/{len(files)} run logs"
                         + ("" if ev.get("keyless") else " on a real row whose loaded preset arms the key")
                         + (f" ({len(gone)} loaded preset(s) no longer on disk)" if gone else ""))
        zero = ev.get("zero_ny")
        zsf = SrcFile(zero["file"], read(self.path(zero["file"]))) if zero and self.path(zero["file"]) \
            and os.path.exists(self.path(zero["file"])) else None
        zero_ok = find_code_line(zsf, zero["grep"]) is not None if zsf else False
        uncovered = [k for k in keys if k not in covered]
        note = "; ".join(notes) + (f" (pin {ev['pin']})" if ev.get("pin") else "")
        note += "; runner's zero-NY_* refusal " + (f"found on a code line of {zero['file']}" if zero_ok else "NOT FOUND")
        if uncovered:
            note += "; keys no armed yaml ever set: " + ", ".join(uncovered)
        if not covered and not (ev.get("keyless") and any("in 0/" not in n for n in notes)):
            return StageResult("open", note)
        if not zero_ok:
            return StageResult("open", note)
        if self.pin.lines is None:
            return StageResult("partial", note + "; the latest pin is unreadable, so marker survival is unknown")
        if markers and not in_pin:
            return StageResult("partial", note + f"; but the latest pin {self.pin.name} carries NONE of its markers "
                               "(the arrival evidence is from a binary the current code no longer matches)")
        if uncovered or ev.get("partial"):
            return StageResult("partial", note + (f"; {ev['partial']}" if isinstance(ev.get("partial"), str) else ""))
        return StageResult("done", note)

    # ---- S6-S8, S10, S11 ----------------------------------------------------------------------
    def audit_evidence(self, cap: dict, st: str, ev: dict | None, recorded: str) -> StageResult:
        ev = ev or {}
        if recorded == "na":
            if not ev.get("note"):
                self.errors.append(f"{cap['id']}: {st} recorded n/a without a reason (evidence.{st}.note)")
                return StageResult("open", "recorded n/a WITHOUT a reason")
            return StageResult("na", ev["note"])
        if recorded == "open" or not ev.get("logs"):
            return StageResult("open", ev.get("note", "") if recorded == "open" else
                               "recorded " + recorded + " but no evidence path recorded")
        files = sorted(set(sum((glob.glob(self.path(g)) for g in ev.get("logs", [])), [])))
        if not files:
            return StageResult("open", "recorded evidence files are MISSING: " + ", ".join(ev["logs"]))
        if ev.get("grep") and not any(ev["grep"] in read(f) for f in files):
            return StageResult("open", f"recorded evidence lacks '{ev['grep']}'")
        return StageResult(recorded, ev.get("note", "") + f" [{len(files)} evidence file(s)]")

    # ---- seal records (S5-S8) and SR ----------------------------------------------------------
    def audit_seal_stage(self, cap: dict, st: str, ev: dict) -> StageResult:
        """A stage DERIVED from a seal STAGE record: re-judged from its own inputs by scripts/seal/seallib.py (the
        judge the runner used), never read off the record; the record's pin must contain the capability's heads."""
        try:
            seal = seal_module()
        except Exception as exc:  # pragma: no cover - the module ships beside this script
            self.errors.append(f"{cap['id']}: {st} names a seal record but scripts/seal/seallib.py is unimportable: {exc}")
            return StageResult("open", "seal library unimportable")
        path = self.path(ev["seal"])
        state, note, rec = seal.verify_stage_record(path, cap["id"], st)
        if rec is not None and state != "open":
            commit = (rec.get("pin") or {}).get("commit") or ""
            bad = [h for h in cap.get("heads", []) if self.git.is_ancestor(h, commit) is not True]
            if bad:
                return StageResult("open", f"seal record pin {commit[:9]} does not contain head(s) {bad}: {note}")
            self._seal_records.setdefault(cap["id"], {})[st] = rec
        return StageResult(state, note)

    def audit_routable(self, cap: dict, stages: dict) -> StageResult:
        """SR.seal (ROUTER_PROGRAM 3.5): S7 + S8 done from seal records, the recorded envelope inside the op sets
        and spec archetypes of the S7/S8 rows on which the capability engaged, and every moat row the envelope
        admits (both years, every scored category) run in S8 -- scripts/seal/seallib.py sr_seal, the check
        `seal record` applies before it writes SR."""
        rt = cap.get("routable")
        if not rt:
            return StageResult("na", "not routable (no routable: block)")
        env = list(rt.get("envelope") or [])
        specs = list(rt.get("envelope_specs") or [])
        done = [s for s in ("S7", "S8") if stages[s].state == "done"]
        if len(done) < 2:
            return StageResult("partial" if done or env else "open",
                               f"SR.seal needs S7 + S8 done (done: {', '.join(done) or 'none'})"
                               + (f"; envelope recorded {env}" if env else "; no envelope recorded"))
        recs = self._seal_records.get(cap["id"], {})
        if "S7" not in recs or "S8" not in recs:
            return StageResult("partial", "S7/S8 are done but not from seal records: the envelope cannot be checked "
                                          "against evidence rows")
        ok, why = seal_module().sr_seal(env, specs, recs)
        return StageResult("done" if ok else "partial", "SR.seal: " + why)

    # ---- S9 -----------------------------------------------------------------------------------
    def audit_arming(self, cap: dict):
        rec = (cap.get("record") or {}).get("S9", "open")
        arm = cap.get("arming") or {}
        found = []
        for k in cap.get("keys", []):
            found.extend(self.configs.arming(k))
        live = sorted({h["file"] for h in found if h["armed"] and not h["canary"] and not h["shadowed_by"]})
        dead = sorted({f"{h['file']} ({h['shadowed_by']})" for h in found if h["armed"] and h["shadowed_by"]})
        canary = sorted({h["file"] for h in found if h["armed"] and h["canary"]})
        recorded = sorted(arm.get("armed", []))
        info = {"live": live, "shadowed": dead, "canary": canary, "values": found,
                "categories": sorted({c for h in found if h["armed"] and not h["canary"] and not h["shadowed_by"]
                                      for c in h["categories"]})}
        flags = []
        extra = [f for f in live if f not in recorded]
        gone = [f for f in recorded if f not in live]
        if extra:
            flags.append("UNRECORDED ARMING: " + ", ".join(extra))
        if gone:
            flags.append("RECORDED ARMING GONE: " + ", ".join(gone))
        # ledger coherence: a dark declared key keeps its unarmed-promotion row; an armed one drops it
        for k in cap.get("keys", []):
            if not k.get("decl") or k.get("retired"):
                # a RETIRED key's ledger rule (no row) is part of its S4 audit
                continue
            armed_k = any(h["armed"] and not h["canary"] and not h["shadowed_by"] and h["path"] == k["path"] for h in found)
            has_row = k["path"] in self.ledger
            if armed_k and has_row:
                flags.append(f"ledger: armed {k['path']} still has an unarmed-promotion row")
            if not armed_k and not has_row:
                flags.append(f"ledger: dark {k['path']} has no preset_unarmed_promotions.tsv row")
        info["flags"] = flags
        parents = arm.get("inherit") or []
        if isinstance(parents, str):
            parents = [parents]
        if live:
            note = "ARMED (live for " + ", ".join(info["categories"]) + "): " + ", ".join(live)
        elif parents:
            pstates = {p: (self.results.get(p) or {}).get("stages", {}).get("S9") for p in parents}
            missing = [p for p, s in pstates.items() if s is None]
            if missing:
                self.errors.append(f"{cap['id']}: S9 inherits from unknown or unaudited capabilities {missing}")
            note = "inherits the arming decision of " + ", ".join(
                f"{p} ({pstates[p].state if pstates[p] else '?'})" for p in parents) + (
                f": {arm['note']}" if arm.get("note") else "")
        elif arm.get("decision"):
            note = "not armed — decision: " + arm["decision"]
        elif rec == "na":
            note = arm.get("note", "no arming applies")
        else:
            note = "not armed in configs/vnncomp*; no decision yet" + (f" ({arm['note']})" if arm.get("note") else "")
        if dead:
            note += "; PLACEBO armings (no scored category loads them): " + ", ".join(dead)
        if canary:
            note += "; canary armings: " + ", ".join(canary)
        if flags:
            note += "; " + "; ".join(flags)
        if extra or gone:
            state = "open"
        elif live or arm.get("decision"):
            state = "done"
        elif parents:
            ps = [(self.results.get(p) or {}).get("stages", {}).get("S9") for p in parents]
            state = max((s.state for s in ps if s), key=lambda x: RANK[x], default="open")
            state = "done" if state in ("done", "na") else state
        else:
            state = "na" if rec == "na" else "open"
        return StageResult(state, note), info

    # ---- gate marker census -----------------------------------------------------------------------
    def census(self) -> dict:
        path = self.path(self.args.gate_script or self.reg.get("defaults", {}).get("gate_script"))
        if not path or not os.path.exists(path):
            self.errors.append(f"gate script missing: {path} (the marker census cannot be compared)")
            return {"script": path, "censused": None}
        text = read(path)
        loops = re.findall(r"for mk in (.*?); do", text, re.S)
        if not loops:
            self.errors.append(f"{path}: no `for mk in ...; do` marker census found")
        censused = set()
        for body in loops:
            censused |= set(re.findall(r"'([^']*)'", body)) | set(re.findall(r'"([^"]*)"', body))
        return {"script": path, "censused": censused}


def ladder(stages: dict) -> str:
    top = "S0"
    for st in STAGES:
        s = stages.get(st)
        if s is None or s.state not in ("done", "na"):
            break
        top = st
    return top


# --------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------

def stage_symbols(stages: dict) -> str:
    return "".join(f"[{SYMBOL[stages[s].state]}]" for s in STAGES)


def render_table(aud: Auditor) -> str:
    out = []
    out.append(f"engine_program_audit: tree={aud.tree} HEAD={(aud.git.sha('HEAD') or '?')[:9]} "
               f"{aud.main_ref}={(aud.main_sha or '?')[:9]}")
    out.append(f"  pin latest={aud.pin.name} sha256={aud.pin.sha or '-'}; shipped={aud.pin_shipped.name}; x86={aud.pin_x86.name}")
    out.append(f"  scored categories: {len(aud.configs.categories)} (from benchmark_roots); "
               f"{len(aud.configs.resolved)} yaml files are loaded by one")
    for r in aud.runs:
        if r.prefix in aud.used_prefixes:
            out.append(f"  gate {r.label} {'MAIN' if r.on_main else 'integration'} {os.path.basename(r.log)} "
                       f"legs={len(r.legs)} tests={len(r.status)} {(r.done or 'NO DONE LINE (in progress?)')[:60]}")
    for r in aud.excluded_runs:
        out.append(f"  gate {r.label} {os.path.basename(r.log)} EXCLUDED: {r.excluded}")
    for b in aud.pending:
        if b["done"] is None:
            out.append(f"  gate block WITHOUT test legs: {os.path.basename(b['log'])} HEAD={b['head'][:9]} start={b['ts']} "
                       f"IN PROGRESS (no DONE line); last: {b['last'][:80]}")
    out.append("  legend: [x] done  [-] n/a  [~] partial  [ ] open;  stages S1..S8, SR (routable), S9..S11 left to right")
    out.append("")
    w = max(len(c["id"]) for c in aud.reg["capabilities"])
    out.append(f"  {'capability':<{w}}  {'milestone':<10} {' 1  2  3  4  5  6  7  8  R  9 10 11':<36}  ladder  flags")
    for cap in aud.reg["capabilities"]:
        res = aud.results[cap["id"]]
        flags = [f for f in res["flags"] if f.startswith("REGRESSION")] + res["arming"].get("flags", [])
        out.append(f"  {cap['id']:<{w}}  {str(cap.get('milestone_short', ''))[:10]:<10} {stage_symbols(res['stages'])}  "
                   f"{res['ladder']:<6}  {'; '.join(flags)[:120]}")
    out.append("")
    counts: dict[str, int] = {}
    for cap in aud.reg["capabilities"]:
        counts[aud.results[cap["id"]]["ladder"]] = counts.get(aud.results[cap["id"]]["ladder"], 0) + 1
    out.append("  ladder stage counts: " + ", ".join(f"{k}={counts[k]}" for k in ["S0"] + STAGES + ["KILLED"] if k in counts))
    live_caps = [c for c in aud.reg["capabilities"] if c.get("status") != "killed"]
    per = {s: sum(1 for c in live_caps if aud.results[c["id"]]["stages"][s].state in ("done", "na")) for s in STAGES}
    out.append("  per-stage done|n/a (killed excluded): " + ", ".join(f"{s}={per[s]}" for s in STAGES)
               + f"  (of {len(live_caps)})")
    cen = aud._census
    if cen["censused"] is not None:
        miss = sorted({m["marker"] for c in aud.reg["capabilities"] for m in aud.results[c["id"]]["markers"]
                       if not m["censused"]})
        out.append(f"  gate census ({os.path.basename(cen['script'])}): {len(miss)} registry markers the next gate does NOT grep "
                   "(a marker counts only when a census string contains it):")
        for m in miss:
            out.append(f"      '{m}'")
    return "\n".join(out)


def render_details(aud: Auditor) -> str:
    out = []
    for cap in aud.reg["capabilities"]:
        res = aud.results[cap["id"]]
        out.append(f"--- {cap['id']}  ladder={res['ladder']} (recorded {res['recorded_ladder']})")
        for st in STAGES:
            s = res["stages"][st]
            out.append(f"  {st:<3} [{SYMBOL[s.state]}] {s.note}"[:400])
        bad = [e for e in res["tests"] if e["status"] != "ok" and e.get("role", "contract") == "contract"]
        for e in bad[:25]:
            out.append(f"      test {e['status']:<13} {e['name']} {e.get('src') or ''} {e.get('why') or ''}")
        for k in res["keys"]:
            if k["missing"]:
                out.append(f"      key {k['path']}: missing {', '.join(k['missing'])}")
        for c in res["consumers"]:
            if not c["line"]:
                out.append(f"      consumer MISSING {c['file']} /{c['grep']}/: {c['why']}")
        for c in res.get("callers", []):
            out.append(f"      callers {c['what']}: {c['found']} (control at {c['control']}) {c['hits'][:3]}")
        for m in res["markers"]:
            if not m["latest"]:
                out.append(f"      marker NOT in latest pin: '{m['marker']}'")
        for f in res["flags"]:
            out.append(f"      {f}")
    return "\n".join(out)


def yaml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return "null" if v is None else str(v)


def anchor(cap: dict) -> str:
    return "cap-" + cap["id"]


def md_escape(s) -> str:
    return str(s).replace("|", "\\|")


def derive_sr(cap: dict) -> str:
    """#router-r5 (ROUTER_PROGRAM §3.5): the routability level of a capability with a `routable:` block,
    DERIVED from its S-ladder: `seal` needs S7 AND S8 done AND a recorded envelope; `obs` needs a recorded
    shadow observation (`record.SR` obs/seal/done); otherwise `decl`. The router routes only at `seal`
    (`ny_core::router::decl::SrLadder::level`, cross-checked against this file by
    `router_r5_registry_ladders_match_the_capability_registry`)."""
    rec = cap.get("record") or {}
    block = cap.get("routable") or {}
    envelope = block.get("envelope") or []
    if rec.get("S7") == "done" and rec.get("S8") == "done" and envelope:
        return "seal"
    if rec.get("SR") in ("obs", "seal", "done"):
        return "obs"
    return "decl"


def render_sr(aud: Auditor) -> list:
    """#router-r5: one row per routable capability: its derived SR level, its ledger, the keys a route sets."""
    rows = [c for c in aud.reg["capabilities"] if c.get("routable")]
    if not rows:
        return []
    out = ["### Stage SR — routable (derived; ROUTER_PROGRAM §3.5, R5)\n",
           "The router CONFIGURES a capability on a row only at `seal` (S7 + S8 done and the envelope recorded), name-blind, "
           "where its own precondition admits the row and its sealed envelope covers it; a shipped yaml line that sets the route "
           "of a `seal` capability is pin debt (`crates/ny-cli/src/router/router_yaml_debt.tsv`, may only fall). "
           "review(eng11): the router's ladder (from the record, what the compiled registry mirrors) must never reach "
           "`seal` where SR.seal does not derive from seal records (the SR stage column); the audit flags that as a "
           "REGRESSION.\n",
           "| capability | kind | SR (router ladder) | SR.seal (seal records) | SR (recorded) | ledger | route sets | envelope |",
           "|---|---|---|---|---|---|---|---|"]
    for cap in rows:
        block = cap["routable"]
        route = block.get("route") or []
        sets = ", ".join(f"`{r.get('key')}={r.get('value')}`" for r in route if isinstance(r, dict)) or "—"
        env = ", ".join(str(e) for e in (block.get("envelope") or [])) or "—"
        rec = (cap.get("record") or {}).get("SR", "open")
        sealed = aud.results.get(cap["id"], {}).get("stages", {}).get("SR")
        sealed = f"[{SYMBOL[sealed.state]}]" if sealed is not None else "?"
        out.append(f"| `{cap['id']}` | {block.get('kind', '')} | **{derive_sr(cap)}** | {sealed} | {rec} | "
                   f"{block.get('ledger', '')} | {sets} | {env} |")
    out.append("")
    return out


def render_md(aud: Auditor) -> str:
    """The per-capability checklist, generated from the registry + this audit."""
    out = []
    labels = aud.reg.get("stages", {})
    out.append("## Capability summary (generated)\n")
    used = [r for r in aud.runs if r.prefix in aud.used_prefixes]
    if used:
        out.append("Gate runs cited below as `<prefix>@<HEAD>`: " + "; ".join(
            f"`{r.label}` = `{os.path.relpath(r.log, aud.scratch) if aud.scratch else r.log}` + legs "
            f"`{r.prefix}_*.txt` ({'MAIN' if r.on_main else 'integration'})" for r in used) + ".\n")
    out.append("| capability | milestone | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | SR | S9 | S10 | S11 | ladder | armed (live) |")
    out.append("|---|---|" + "---|" * len(STAGES) + "---|---|")
    for cap in aud.reg["capabilities"]:
        r = aud.results[cap["id"]]
        cells = " | ".join(f"[{SYMBOL[r['stages'][s].state]}]" for s in STAGES)
        live = ", ".join(os.path.basename(f) for f in r["arming"]["live"]) or "—"
        tag = f" **{cap['status']}**" if cap.get("status") else ""
        out.append(f"| [`{cap['id']}`](#{anchor(cap)}){tag} | {md_escape(cap.get('milestone_short', ''))} | {cells} | {r['ladder']} | {live} |")
    out.append("")
    out.extend(render_sr(aud))
    fam = None
    for cap in aud.reg["capabilities"]:
        res = aud.results[cap["id"]]
        if cap.get("family") != fam:
            fam = cap.get("family")
            out.append(f"\n## Family: {fam}\n")
        status = f" — **{cap['status'].upper()}**" if cap.get("status") else ""
        out.append(f'<a id="{anchor(cap)}"></a>\n')
        out.append(f"### `{cap['id']}` — {cap['name']}{status}\n")
        out.append(f"- **Milestone:** {cap.get('milestone', '')}")
        br = cap.get("branches") or ""
        out.append(f"- **Branch / heads:** {br} — heads {', '.join('`' + h + '`' for h in cap.get('heads', []))}")
        out.append(f"- **Ladder (derived):** {res['ladder']} · recorded {res['recorded_ladder']}\n")
        for st in STAGES:
            s = res["stages"][st]
            box = "x" if s.state in ("done",) else ("-" if s.state == "na" else ("~" if s.state == "partial" else " "))
            label = labels.get(st, st)
            extra = ""
            ev = (cap.get("evidence") or {}).get(st) or {}
            if st not in DERIVED and ev.get("note") and ev.get("note") not in s.note:
                extra = " — " + ev["note"]
            note = s.note or ("n/a" if s.state == "na" else "no evidence recorded")
            out.append(f"- [{box}] **{st}** {label}: {md_escape(note)}{md_escape(extra)}")
        if res["keys"]:
            out.append("\n| yaml key | type · default | LeverDecl | space.rs axis | schema | apply | bridge · cell | frontier rows · ledger row |")
            out.append("|---|---|---|---|---|---|---|---|")
            for k in res["keys"]:
                out.append("| `{}` | {} · `{}` | {} | {} | {} | {} | {} | {} |".format(
                    k["path"], md_escape(k.get("type")), yaml_scalar(k.get("default")),
                    md_escape(k.get("decl") or "—"), md_escape(k.get("axis") or "—"), md_escape(k.get("schema") or "MISSING"),
                    md_escape(k.get("apply") or "MISSING"),
                    md_escape(" · ".join(x for x in (k.get("bridge"), k.get("cell")) if x) or "—"),
                    f"{k.get('frontier_rows') if k.get('frontier_rows') is not None else '—'} · {'yes' if k.get('ledger_row') else 'no'}"))
        if res["consumers"]:
            out.append("\n**Consumer sites** (production lines; a definition is backed by a production caller): " + "; ".join(
                (f"`{c['file']}:{c['line']}`" + (f" (called at `{c['callers'][0]}`)" if c.get("callers") else ""))
                if c["line"] else f"`{c['file']}`: MISSING ({md_escape(c['why'])})" for c in res["consumers"]))
        if res.get("callers"):
            out.append("\n**Production-site counts** (code only, test code excluded, each with a positive control): " + "; ".join(
                f"{md_escape(c['what'])}: **{c['found']}**" + (f" (e.g. `{c['hits'][0]}`)" if c["hits"] else "")
                + (f"; control `{c['control']}`" if c.get("control") else "; control FAILED") for c in res["callers"]))
        if res["markers"]:
            out.append("\n**Own observables** (`strings -a <pin> | grep -cF`): " + "; ".join(
                f"`{m['marker']}` latest={m['latest']} shipped={m['shipped']} x86={m['x86']}" for m in res["markers"]))
        tests = res["tests"]
        if tests:
            ok = sum(1 for e in tests if e["status"] == "ok")
            out.append(f"\n**Named tests:** {ok}/{len(tests)} ok in a gate containing the capability.")
            bad = [e for e in tests if e["status"] != "ok"]
            if bad:
                out.append("Not ok: " + "; ".join(f"`{e['name']}` ({e['status']}{', ' + e['role'] if e.get('role') != 'contract' else ''})" for e in bad[:30]))
        m = cap.get("measurement") or {}
        if m:
            out.append("\n**Measurement:** " + " ".join(f"*{k}:* {md_escape(v)}" for k, v in m.items()))
        gaps = list(cap.get("gaps", []))
        for k in res["keys"]:
            for miss in k["missing"]:
                g = f"`{k['path']}`: {miss} (derived)"
                if g not in gaps:
                    gaps.append(g)
        for f in res["arming"].get("flags", []):
            gaps.append(f + " (derived)")
        for f in res["flags"]:
            if f.startswith("marker "):
                gaps.append(f + " (derived)")
        if gaps:
            out.append("\n**Open gaps:**")
            for g in gaps:
                out.append(f"- {g}")
        out.append("")
    return "\n".join(out)


def update_md(path: str, aud: Auditor, audit_text: str) -> None:
    text = read(path)
    gen = render_md(aud)
    for tag, body in (("CAPABILITIES", gen), ("AUDIT", "```\n" + audit_text.rstrip() + "\n```")):
        b, e = f"<!-- BEGIN GENERATED {tag} -->", f"<!-- END GENERATED {tag} -->"
        if b not in text or e not in text:
            raise SystemExit(f"{path}: missing {b} / {e}")
        pre, rest = text.split(b, 1)
        _, post = rest.split(e, 1)
        text = pre + b + "\n" + body.rstrip() + "\n" + e + post
    with open(path, "w") as fh:
        fh.write(text)


# --------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.path.join(repo, "docs", "engine_program_capabilities.yaml"))
    ap.add_argument("--tree", default=repo, help="source tree to audit (read-only; default: this repo)")
    ap.add_argument("--main-ref", default=None, help="ref that counts as main (default: registry, origin/main)")
    ap.add_argument("--check-remote", action="store_true",
                    help="error (rc 2) when the local main ref differs from the remote's (git ls-remote)")
    ap.add_argument("--pin", default=None, help="release pin whose strings carry the markers (default: registry latest)")
    ap.add_argument("--shipped-pin", default=None)
    ap.add_argument("--x86-pin", default=None)
    ap.add_argument("--gate-log", action="append", help="extra gate log(s); legs are found beside it by prefix")
    ap.add_argument("--only-gates", action="store_true", help="use only --gate-log, not the registry's gate list")
    ap.add_argument("--gate-script", default=None, help="gate runner whose marker census is compared (default: registry)")
    ap.add_argument("--scratch-root", default=None, help="root for the registry's relative evidence paths")
    ap.add_argument("--details", action="store_true", help="print per-stage evidence for every capability")
    ap.add_argument("--json", default=None, help="write the derived results as JSON to this path")
    ap.add_argument("--update-md", default=None, help="rewrite the generated regions of this checklist")
    args = ap.parse_args(argv)

    try:
        reg = yaml.safe_load(read(args.registry))
    except (OSError, yaml.YAMLError) as exc:
        sys.stderr.write(f"engine_program_audit: cannot read registry {args.registry}: {exc}\n")
        return 2
    if not isinstance(reg, dict) or not reg.get("capabilities"):
        sys.stderr.write(f"engine_program_audit: registry {args.registry} has no capabilities\n")
        return 2
    ids = [c["id"] for c in reg["capabilities"]]
    if len(ids) != len(set(ids)):
        sys.stderr.write("engine_program_audit: duplicate capability ids in the registry\n")
        return 2
    aud = Auditor(reg, args)
    # every capability sees only the gate runs whose HEAD contains all of its heads; a capability whose
    # S9 inherits another's decision is audited after it
    order = sorted(reg["capabilities"], key=lambda c: 1 if (c.get("arming") or {}).get("inherit") else 0)
    for cap in order:
        aud._cur_runs = aud._relevant_runs(cap)
        aud.used_prefixes.update(r.prefix for r in aud._cur_runs)
        aud.results[cap["id"]] = aud.audit_cap(cap)
    regress = [(c["id"], f) for c in reg["capabilities"] for f in aud.results[c["id"]]["flags"] if f.startswith("REGRESSION")]
    arming = [(c["id"], f) for c in reg["capabilities"] for f in aud.results[c["id"]]["arming"].get("flags", [])
              if f.startswith(("UNRECORDED", "RECORDED ARMING GONE"))]
    rc = 1 if (regress or arming) else (2 if aud.errors else 0)
    tail = [f"  ERROR {e}" for e in aud.errors]
    tail += [f"  {cid}: {f}" for cid, f in regress + arming]
    tail.append(f"AUDIT_DONE rc={rc} capabilities={len(reg['capabilities'])} regressions={len(regress)} "
                f"arming_mismatches={len(arming)} errors={len(aud.errors)}")
    summary = render_table(aud) + "\n" + "\n".join(tail)
    print(render_table(aud) + ("\n\n" + render_details(aud) if args.details else "") + "\n" + "\n".join(tail))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({cid: {"ladder": r["ladder"], "stages": {s: vars(v) for s, v in r["stages"].items()},
                             "flags": r["flags"], "arming": {k: v for k, v in r["arming"].items() if k != "values"}}
                       for cid, r in aud.results.items()}, fh, indent=1, default=str)
    if args.update_md:
        update_md(args.update_md, aud, summary)
    return rc


if __name__ == "__main__":
    sys.exit(main())
