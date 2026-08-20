#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Lever search harness: drive `ny benchmarks run` over arms, and REFUSE the fakes.

WHAT IT DOES. Generates arms over the axes of `crates/ny-levers/src/space.rs`,
runs each as `ny benchmarks run --year Y --category C --lever NAME=VALUE ...`,
then validates the artifacts the run already wrote and records one JSONL row per
(arm, category). An arm that fails validation is reported INVALID and EXCLUDED —
never averaged in, never counted as a negative result.

WHY VALIDATION IS THE POINT, NOT THE PLUMBING. Every failure mode below has
already produced a confident-but-empty conclusion in this repository, and each
one looks exactly like a legitimate measurement from the outside:

  * A REJECTED LEVER VALUE resolves to the declaration default. `resolve_raw`
    arms a Bool on the exact byte string "1" and disarms on exact "0"; "true",
    "01", " 1" and "" are REJECTIONS recorded in `rejected_raw`. A harness that
    emits one of those spends a full instance budget measuring the BASELINE and
    reports it as a treatment. T1 therefore asserts `rejected_raw` is empty on
    every row of every completed run, which is precisely what the module
    docstring of `space.rs` tells callers to do.
  * An INERT ARM is the same waste without the receipt: a child axis whose
    prerequisite is unmet is read by nobody. The lattice is mirrored here so
    such a sample is never GENERATED — the binary would refuse it, but a
    refusal round-trip per sample is silly, and in grid mode most corners are
    forbidden.
  * A VALUE EQUAL TO THE COMPILED DEFAULT is a no-op run wearing a treatment
    label. `NY_INTERM_ROW_CHUNKS=1` is the live example: `DefaultSpec::U64(1)`,
    and `space.rs` says in as many words that 1 is byte-identical to the
    historical single sweep. T0 (below) drops those before they cost 100 s.
  * An AMBIENT LEVER exported into the calling shell is inherited by every
    child and is invisible in the result. This harness scrubs `NY_*` from the
    child environment and then CHECKS the sealed `ambient_env` in the manifest,
    because trusting its own scrub would be the same class of mistake.
  * A SAT ROW WITHOUT A RETAINED WITNESS cannot be revalidated by anyone. A
    run with any such row is invalid here, full stop.
  * A ROW SERVED FROM THE VERDICT CACHE is a replay, not a measurement, and it
    is the newest member of this list. `--cache` embeds the EARLIER run's
    flight record in the bank verbatim, so every receipt check below (T1.1,
    T1.2, T1.4) passes against evidence produced by a different process: a
    sweep in which zero children ran would otherwise be byte-indistinguishable
    from one that ran in full. T1.10 reads the per-row `from_cache` marker,
    fails a bank in which EVERY row was served, and warns on any partial
    replay; T1.7 additionally refuses a summary claiming more `cache_hits`
    than the bank marks.

T0 (pre-flight, free) and T1 (post-run, from artifacts):

  T0  every (axis, value) about to be emitted must PARSE under the axis's
      declared `LeverKind` (scraped from `crates/ny-levers/src/decls/*.rs`) and
      must DIFFER from the declared default. Axes whose kind cannot be read are
      excluded unless `--allow-unchecked-domains`; fail-closed, because the
      failure mode of guessing is a whole sweep of fake treatments.
  T1  per completed run: empty `rejected_raw` on every row, a resolved lever
      receipt, the row `arm` and the child's `ambient_env` equal to what was
      requested, a clean parent ambient, `sat_rows_without_witness == 0`, no
      `error` rows, at least one row actually MEASURED rather than served from
      the verdict cache, and the printed summary agreeing with the metadata
      bank.

WHY THERE IS NO ASHA / MULTI-FIDELITY SCHEDULER HERE. Deliberate. A
multi-fidelity scheduler needs a cheap signal that RANKS arms the same way the
expensive one does, and the obvious candidate on this codebase — the root
objective census (`NY_ACASXU_PROF=1`) — is SUSPECT, not merely unproven: it has
already moved 0/99 -> 92/99 while converting ZERO rows. An arm that improves the
census by 92 objectives and solves nothing would be promoted by ASHA over an arm
that solves a row, which is not early stopping, it is an automated wrong answer.
The same applies to shrinking the budget as a fidelity: 5 cifar100 timeout rows
re-run at {30 s, 50 s} converted 0/10, so budget is not a cheap proxy either. A
scheduler attaches at `select_next_arms` below, and it must not be attached
until some cheap signal has been CALIBRATED against measured row conversions.

WHAT THE BINARY CURRENTLY OFFERS (probed at run time, never assumed):
`benchmarks run --lever NAME=VALUE` IS wired and is the only way arms are
delivered — if a binary lacks it this harness REFUSES to run rather than fall
back to exporting the arm into the environment, which is the invisible-arm hole
`ambient_env` sealing exists to close. `--cache` is passed automatically when
the probed binary has it (older binaries simply measure every (arm, row) pair
fresh); a served row is then marked `from_cache` in the bank and T1.10 is what
stops that saving from becoming a fake measurement. There is no lever/space
subcommand, so `--list-space` parses `space.rs` and says so.

Examples
  scripts/ny_search.py --list-space
  scripts/ny_search.py --category cifar100 --mode random --samples 8 --dry-run
  scripts/ny_search.py --category cifar100_2024 --mode random --samples 8 \
      --configs-dir configs --out-dir reports/search/2026-08-16
  scripts/ny_search.py --self-test        # no GPU, no binary, no benchmarks
"""

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_RS = REPO_ROOT / "crates" / "ny-levers" / "src" / "space.rs"
DECLS_DIR = REPO_ROOT / "crates" / "ny-levers" / "src" / "decls"
RESULT_SCHEMA = "ny-search/result/v1"

# Probes tried in order when the binary might be able to print the space itself.
# None of these subcommands exists today (`ny --help` lists no lever/space
# command), so the parsed-source path is the live one; the probe stays because a
# binary that CAN print its own space is authoritative and the source parse is
# not. `--list-space` always states which source it used.
SPACE_PROBES = (
    ("levers", "space", "--json"),
    ("levers", "--json"),
    ("lever-space", "--json"),
    ("benchmarks", "space", "--json"),
)


# --------------------------------------------------------------------------
# The search space, mirrored from crates/ny-levers/src/space.rs
# --------------------------------------------------------------------------


class Axis:
    """One searchable dimension: name, admissible tokens, class, delivery."""

    def __init__(self, name, domain_kind, values, cls, deliver, why=""):
        self.name = name
        self.domain_kind = domain_kind  # "bool" | "grid" | "enum"
        self.values = list(values)  # emitted verbatim, exactly as the parser sees them
        self.cls = cls  # "VerdictAffecting" | "SafeToSearch" | "Unsafe"
        self.deliver = deliver  # "preset:<key>" | "env-only"
        self.why = why

    def as_json(self):
        return {
            "name": self.name,
            "domain": self.domain_kind,
            "values": self.values,
            "class": self.cls,
            "deliver": self.deliver,
        }


class Edge:
    """`child` does nothing unless `requires` holds. Mirrors `space::Edge`."""

    def __init__(self, child, kind, parent, bound=None, site=""):
        self.child = child
        self.kind = kind  # "armed" | "non_zero" | "greater_than" | "not_armed"
        self.parent = parent
        self.bound = bound
        self.site = site

    def render(self):
        if self.kind == "armed":
            return 'requires `%s` = "1" (%s)' % (self.parent, self.site)
        if self.kind == "non_zero":
            return 'requires `%s` present and not "0" (%s)' % (self.parent, self.site)
        if self.kind == "greater_than":
            return "requires `%s` > %d (%s)" % (self.parent, self.bound, self.site)
        return 'suppressed while `%s` = "1" (%s)' % (self.parent, self.site)


class Space:
    def __init__(self, axes, edges, unsafe_names, instrument_only, test_only, source):
        self.axes = axes
        self.edges = edges
        self.unsafe_names = list(unsafe_names)
        self.instrument_only = list(instrument_only)
        self.test_only = list(test_only)
        self.source = source
        self.by_name = {axis.name: axis for axis in axes}
        self.edges_by_child = {}
        for edge in edges:
            self.edges_by_child.setdefault(edge.child, []).append(edge)


class Inert(Exception):
    """Why a sample was refused. Mirrors `space::Inert`."""

    def __init__(self, axis, because):
        super().__init__("`%s` is inert: %s" % (axis, because))
        self.axis = axis
        self.because = because


def _rust_block(text, header):
    """Body of `const NAME: ... = &[ ... ];`, matched on the closing `\n];`."""
    start = text.find(header)
    if start < 0:
        return ""
    start += len(header)
    end = text.find("\n];", start)
    return text[start:end] if end > 0 else text[start:]


def _rust_str_list(body):
    return re.findall(r'"([^"]+)"', body)


def _rust_prose(body):
    """Join a Rust string literal (or a run of them) back into one line.

    The declarations wrap `why`/`site` with `\\`-continuations, so the raw match
    carries backslashes and the indentation of the next source line.
    """
    return re.sub(r"\s+", " ", " ".join(_rust_str_list(body)).replace("\\", " ")).strip()


def _parse_axis_records(body):
    """Split an AXES/UNSAFE_AXES body into per-`Axis {}` chunks."""
    chunks = []
    for match in re.finditer(r"Axis\s*\{", body):
        depth = 1
        index = match.end()
        while index < len(body) and depth:
            if body[index] == "{":
                depth += 1
            elif body[index] == "}":
                depth -= 1
            index += 1
        chunks.append(body[match.end() : index - 1])
    return chunks


def _parse_domain(chunk):
    domain = re.search(r"domain:\s*Domain::(\w+)", chunk)
    if not domain:
        raise ValueError("axis record has no domain: %r" % chunk[:120])
    kind = domain.group(1)
    if kind == "Bool":
        # `Domain::Bool` only ever emits "0"/"1" — every other spelling is a
        # parser REJECTION that silently resolves to the default.
        return "bool", ["0", "1"]
    if kind == "Grid":
        inner = re.search(r"Domain::Grid\(&\[([^\]]*)\]", chunk).group(1)
        return "grid", [tok.strip().replace("_", "") for tok in inner.split(",") if tok.strip()]
    if kind == "Enum":
        inner = re.search(r"Domain::Enum\(&\[([^\]]*)\]", chunk, re.S).group(1)
        return "enum", _rust_str_list(inner)
    raise ValueError("unknown Domain::%s" % kind)


def _parse_axes(body):
    axes = []
    for chunk in _parse_axis_records(body):
        name = re.search(r'name:\s*"([^"]+)"', chunk).group(1)
        kind, values = _parse_domain(chunk)
        cls = re.search(r"class:\s*Class::(\w+)", chunk).group(1)
        preset = re.search(r'deliver:\s*Deliver::PresetKey\("([^"]+)"\)', chunk)
        deliver = "preset:%s" % preset.group(1) if preset else "env-only"
        why = re.search(r"why:(.*)$", chunk, re.S)
        prose = _rust_prose(why.group(1)) if why else ""
        axes.append(Axis(name, kind, values, cls, deliver, prose))
    return axes


def _parse_edges(body):
    edges = []
    for match in re.finditer(
        r'child:\s*"([^"]+)",\s*requires:\s*Requirement::(\w+)\(\s*"([^"]+)"'
        r'(?:\s*,\s*(\d+))?\s*\),\s*site:\s*((?:"[^"]*"\s*)+)',
        body,
        re.S,
    ):
        child, requirement, parent, bound, site = match.groups()
        kind = {
            "Armed": "armed",
            "NonZero": "non_zero",
            "GreaterThan": "greater_than",
            "NotArmed": "not_armed",
        }[requirement]
        edges.append(Edge(child, kind, parent, int(bound) if bound else None, _rust_prose(site)))
    return edges


def parse_space_rs(path=SPACE_RS):
    """Parse the search space out of `space.rs` — the fallback source."""
    text = path.read_text(encoding="utf-8")
    axes = _parse_axes(_rust_block(text, "const AXES: &[Axis] = &["))
    edges = _parse_edges(_rust_block(text, "const EDGES: &[Edge] = &["))
    unsafe_axes = _parse_axes(_rust_block(text, "const UNSAFE_AXES: &[Axis] = &["))
    instruments = _rust_str_list(_rust_block(text, "const INSTRUMENT_ONLY: &[&str] = &["))
    test_only = _rust_str_list(_rust_block(text, "const TEST_ONLY: &[&str] = &["))
    if not axes:
        raise ValueError("parsed no axes from %s" % path)
    known = {axis.name for axis in axes}
    for edge in edges:
        if edge.child not in known or edge.parent not in known:
            raise ValueError("edge %s <- %s names an unknown axis" % (edge.child, edge.parent))
    return Space(
        axes,
        edges,
        [axis.name for axis in unsafe_axes],
        instruments,
        test_only,
        "parsed %s" % path.relative_to(REPO_ROOT),
    )


def probe_space_binary(ny_bin, timeout=20):
    """Ask the binary to print its own space. Returns (Space, argv) or None.

    The binary is authoritative when it can answer; the source parse cannot see
    a `#[cfg]`-gated axis or a binary built from a different revision. Only
    subcommands the binary actually advertises are attempted, so a miss costs
    one `--help`.
    """
    if not ny_bin or not Path(ny_bin).exists():
        return None
    try:
        help_text = subprocess.run(
            [str(ny_bin), "--help"],
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    advertised = {
        line.split()[0]
        for line in help_text.splitlines()
        if line.startswith("  ") and line.split()
    }
    for probe in SPACE_PROBES:
        if probe[0] not in advertised:
            continue
        try:
            done = subprocess.run(
                [str(ny_bin), *probe], capture_output=True, text=True, timeout=timeout
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode != 0:
            continue
        try:
            payload = json.loads(done.stdout)
        except ValueError:
            continue
        axes = payload.get("axes")
        if not isinstance(axes, list) or not axes:
            continue
        parsed = [
            Axis(
                item["name"],
                item.get("domain", "bool"),
                item.get("values", ["0", "1"]),
                item.get("class", "VerdictAffecting"),
                item.get("deliver", "env-only"),
                item.get("why", ""),
            )
            for item in axes
        ]
        edges = [
            Edge(item["child"], item["kind"], item["parent"], item.get("bound"),
                 item.get("site", ""))
            for item in payload.get("edges", [])
        ]
        return Space(
            parsed,
            edges,
            payload.get("unsafe", []),
            payload.get("instrument_only", []),
            payload.get("test_only", []),
            "binary `%s %s`" % (Path(ny_bin).name, " ".join(probe)),
        ), list(probe)
    return None


def load_space(ny_bin):
    probed = probe_space_binary(ny_bin)
    if probed:
        return probed[0]
    return parse_space_rs()


# --------------------------------------------------------------------------
# The interaction lattice, mirrored from `space::expand`
# --------------------------------------------------------------------------


def _armed(sample, name):
    return sample.get(name) == "1"


def _non_zero(sample, name):
    return name in sample and sample[name] != "0"


def _greater_than(sample, name, bound):
    try:
        return int(sample[name]) > bound
    except (KeyError, TypeError, ValueError):
        return False


def expand(sample, space):
    """Mirror of `space::expand`: the sorted assignment, or raise `Inert`.

    An axis left at its default cannot be inert — it is not being moved — which
    is why an explicit "0" is exempt, exactly as the Rust does.

    Deliberately NOT preset-aware, unlike the T0 filter. `space::expand` judges
    the arm alone, so an arm whose parent is armed only by the preset is refused
    by the binary; the repair below supplies the redundant parent token to get
    past that. Teaching this mirror about presets would emit arms the binary then
    rejects.
    """
    for name in sorted(sample):
        value = sample[name]
        if name in space.unsafe_names:
            raise Inert(name, "axis is Class::Unsafe and must never be set by a search")
        if value == "0":
            continue
        for edge in space.edges_by_child.get(name, []):
            if edge.kind == "armed":
                ok = _armed(sample, edge.parent)
            elif edge.kind == "non_zero":
                ok = _non_zero(sample, edge.parent)
            elif edge.kind == "greater_than":
                ok = _greater_than(sample, edge.parent, edge.bound)
            else:
                ok = not _armed(sample, edge.parent)
            if not ok:
                raise Inert(name, edge.render())
    return sorted(sample.items())


def is_inert(sample, space):
    try:
        expand(sample, space)
    except Inert as inert:
        return inert
    return None


def _satisfying_value(axis, edge, kinds=None):
    """Smallest token for `edge.parent` that satisfies `edge`, or None.

    Only tokens the declared parser accepts are eligible: satisfying an edge
    with a value that lands in `rejected_raw` would leave the child inert AND
    the run mislabelled, which is both failures at once.
    """
    if axis is None:
        return None
    kind = (kinds or {}).get(axis.name)
    eligible = [
        token
        for token in axis.values
        if kinds is None or kind is None or value_admissible(kind, token)
    ]
    if edge.kind in ("armed", "non_zero"):
        return "1" if "1" in eligible else None
    if edge.kind == "greater_than":
        numeric = []
        for token in eligible:
            try:
                numeric.append((int(token), token))
            except ValueError:
                continue
        for value, token in sorted(numeric):
            if value > edge.bound:
                return token
    return None


def repair(sample, space, kinds=None):
    """Close a sample under the lattice. Returns `(sample, prerequisites_added)`.

    Constructive rather than rejective, so random sampling does not degenerate
    into resampling. Where a `NotArmed` edge fires, the PARENT owns the slot
    (that is what the edge means), so the child goes.

    A prerequisite is emitted even when the compiled default already satisfies
    it — `NY_BAB_RESNET_WIDE=1` under `DefaultSpec::Bool(true)` — because
    `space::expand` judges the SAMPLE alone: it refuses a child whose parent is
    merely defaulted, so the redundant pair is what gets the arm past the
    binary's own validation. Those additions are reported separately so nothing
    mistakes them for treatments.
    """
    sample = dict(sample)
    added = set()
    for _ in range(len(sample) * len(space.edges) + 16):
        changed = False
        for name in sorted(sample):
            if name not in sample or sample[name] == "0":
                continue
            for edge in space.edges_by_child.get(name, []):
                parent_axis = space.by_name.get(edge.parent)
                if edge.kind == "not_armed":
                    if _armed(sample, edge.parent):
                        del sample[name]
                        added.discard(name)
                        changed = True
                        break
                    continue
                if edge.kind == "armed" and _armed(sample, edge.parent):
                    continue
                if edge.kind == "non_zero" and _non_zero(sample, edge.parent):
                    continue
                if edge.kind == "greater_than" and _greater_than(sample, edge.parent, edge.bound):
                    continue
                fix = _satisfying_value(parent_axis, edge, kinds)
                if fix is None or edge.parent in added:
                    # Either the prerequisite is unreachable within the declared
                    # domain, or arming it was already tried and something else
                    # suppressed it again. Drop the CHILD rather than oscillate:
                    # it can never be measured in this sample.
                    del sample[name]
                    added.discard(name)
                    changed = True
                    break
                sample[edge.parent] = fix
                added.add(edge.parent)
                changed = True
        if not changed:
            break
    # Belt and braces: whatever the closure could not settle, drop the axis the
    # lattice names until the sample is clean. This terminates (the sample only
    # shrinks) and guarantees no inert sample is ever emitted.
    while sample:
        inert = is_inert(sample, space)
        if inert is None:
            break
        del sample[inert.axis]
        added.discard(inert.axis)
    return sample, added


# --------------------------------------------------------------------------
# T0: does the value survive the declared parser, and is it a treatment at all?
# --------------------------------------------------------------------------


def parse_lever_kinds(decls_dir=DECLS_DIR):
    """`{NY_NAME: {"kind": ..., "min": ..., "max": ..., "default": token|None}}`.

    Scraped from the `declare_levers!` blocks. `kind:` always immediately
    follows `name:`, and `default:` is the next `DefaultSpec` before the next
    declaration, which is what bounds the search.
    """
    kinds = {}
    for path in sorted(Path(decls_dir).glob("*.rs")):
        text = path.read_text(encoding="utf-8")
        starts = [
            (match.start(), match.group(1))
            for match in re.finditer(r'name:\s*"(NY_[A-Z0-9_]+)"', text)
        ]
        for index, (offset, name) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
            block = text[offset:end]
            kind_match = re.search(r"kind:\s*LeverKind::(\w+)", block)
            if not kind_match:
                continue
            record = {"kind": kind_match.group(1), "default": None, "file": path.name}
            if record["kind"] in ("F64Open", "F64ClosedTrimmed"):
                bounds = re.search(
                    r"F64(?:Open|ClosedTrimmed)\s*\{\s*min:\s*([-\d._eE]+)"
                    r"\s*,\s*max:\s*([-\d._eE]+)",
                    block,
                )
                if bounds:
                    record["min"] = float(bounds.group(1))
                    record["max"] = float(bounds.group(2))
            if record["kind"] == "Enum":
                members = re.search(r"LeverKind::Enum\(&\[([^\]]*)\]", block, re.S)
                record["members"] = _rust_str_list(members.group(1)) if members else []
            default = re.search(r"default:\s*DefaultSpec::(\w+)(?:\(([^)]*)\))?", block)
            if default:
                record["default"] = _default_token(default.group(1), default.group(2))
            kinds[name] = record
    return kinds


def _default_token(variant, payload):
    """The raw token an environment value must DIFFER from to be a treatment."""
    if variant == "Unset":
        return None
    if variant == "Bool":
        return "1" if (payload or "").strip() == "true" else "0"
    if payload is None:
        return None
    return (payload or "").strip().replace("_", "")


def value_admissible(kind, token):
    """Does `token` survive `ny_levers::env::parse` for this kind?

    Mirrors the chokepoint exactly. A value that does not survive resolves to
    the declaration default and lands in `rejected_raw`: the run measures the
    baseline and the harness would report it as a treatment.
    """
    if kind is None:
        return False
    name = kind["kind"]
    if name == "Bool":
        return token in ("0", "1")
    if name == "Presence":
        return True  # cannot reject: presence alone arms it
    if name == "U64":
        return bool(re.fullmatch(r"\d+", token))
    if name in ("U64Trimmed", "UsizeTrimmed"):
        return bool(re.fullmatch(r"\d+", token.strip()))
    if name == "F64Open":
        try:
            value = float(token)
        except ValueError:
            return False
        low = kind.get("min", float("-inf"))
        high = kind.get("max", float("inf"))
        return value == value and abs(value) != float("inf") and low < value < high
    if name == "F64ClosedTrimmed":
        # CLOSED at both ends, which is the whole reason the kind exists: for a
        # fraction, 0.0 ("this phase is worth nothing here") and 1.0 are
        # meaningful settings rather than degenerate ones. Rejecting the
        # endpoints would drop the disarmed control arm — the baseline every
        # other arm is scored against.
        #
        # Without this branch the function fell through to `return False`, so
        # EVERY token of this kind was rejected as "would resolve to the
        # default" and the axis silently vanished from the search.
        try:
            value = float(token.strip())
        except ValueError:
            return False
        low = kind.get("min", float("-inf"))
        high = kind.get("max", float("inf"))
        return value == value and abs(value) != float("inf") and low <= value <= high
    if name == "Secs":
        try:
            value = float(token)
        except ValueError:
            return False
        return value == value and abs(value) != float("inf") and value >= 0.0
    if name == "Enum":
        return token in kind.get("members", [])
    if name == "Text":
        return True
    return False


def resolve_preset_path(configs_dir, category):
    """Mirror of `vnncomp::resolve_preset_path` — `configs_dir/vnncomp*/<name>.yaml`.

    The Rust side iterates `vnncomp*` SUBDIRECTORIES of `configs_dir`, newest name
    first, trying the full category name before the year-stripped base. Passing a
    year directory itself (`configs/vnncomp25`) therefore resolves NOTHING and the
    run silently applies no preset at all.
    """
    configs_dir = Path(configs_dir)
    lower = category.lower()
    candidates = [lower]
    if len(lower) >= 5 and re.match(r"^_20\d\d$", lower[-5:]):
        candidates.append(lower[:-5])
    if not configs_dir.is_dir():
        return None
    year_dirs = sorted((p for p in configs_dir.iterdir()
                        if p.is_dir() and p.name.startswith("vnncomp")),
                       key=lambda p: p.name, reverse=True)
    for year_dir in year_dirs:
        for candidate in candidates:
            preset = year_dir / ("%s.yaml" % candidate)
            if preset.is_file():
                return preset
    return None


def load_preset_doc(preset_path):
    """Parse a preset YAML, or None if it cannot be read (PyYAML absent, bad file)."""
    if preset_path is None:
        return None
    try:
        import yaml  # optional: --self-test and --list-space never need it
    except ImportError:
        return None
    try:
        with open(preset_path, "r", encoding="utf-8") as handle:
            doc = yaml.safe_load(handle)
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def preset_effective_token(preset_doc, deliver):
    """Token the LOADED PRESET already delivers for a `preset:<dotted.key>` axis.

    Returns None when the axis is env-only, no preset was read, or the key is
    absent from it (in which case the DefaultSpec default is genuinely the
    baseline). Values are normalised to the exact spelling `Domain` emits:
    booleans to "0"/"1", integers verbatim.
    """
    if not deliver.startswith("preset:") or not isinstance(preset_doc, dict):
        return None
    node = preset_doc
    for part in deliver[len("preset:"):].split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool):
        return "1" if node else "0"
    if isinstance(node, int):
        return str(node)
    if isinstance(node, float):
        return repr(node)
    if isinstance(node, str):
        return node
    return None


def admissible_values(axis, kinds, allow_unchecked, preset_doc=None, preset_known=False):
    """T0-filtered tokens for one axis, plus the reasons anything was dropped.

    The baseline a token must differ from is the value THIS RUN will actually
    load, which for a `Deliver::PresetKey` axis is the preset's value and not
    `DefaultSpec`. Comparing against `DefaultSpec` alone inverts the filter
    wherever a preset ships an axis armed: on cifar100 the preset carries
    `bab.root_comprehensive_gpu_interm: true` and `..._chunks: 64` against
    declared defaults of `false`/`1`, so the DefaultSpec comparison KEEPS the two
    tokens that merely re-measure the shipped baseline and DROPS `=0`/`=1`, the
    only genuine treatments — on the one benchmark that decides the score.

    When an axis is preset-delivered but the preset could not be read, no token is
    dropped and the reason is reported. Silently falling back to `DefaultSpec`
    there is exactly the inversion above.
    """
    kind = kinds.get(axis.name)
    dropped = []
    if kind is None:
        if not allow_unchecked:
            return [], ["%s: no LeverKind found in decls (excluded; --allow-unchecked-domains "
                        "to search it anyway)" % axis.name]
        return list(axis.values), ["%s: kind unknown, values UNCHECKED" % axis.name]

    is_preset_axis = axis.deliver.startswith("preset:")
    preset_token = preset_effective_token(preset_doc, axis.deliver)
    baseline_token = kind.get("default")
    baseline_source = "DefaultSpec default"
    if is_preset_axis and preset_token is not None:
        baseline_token = preset_token
        baseline_source = "the value preset %s already delivers" % axis.deliver[len("preset:"):]
    elif is_preset_axis and not preset_known:
        baseline_token = None
        dropped.append(
            "%s is preset-delivered (%s) but no preset was resolved for this run — keeping "
            "every token, because DefaultSpec is NOT the baseline a preset run measures "
            "against" % (axis.name, axis.deliver)
        )

    kept = []
    for token in axis.values:
        if not value_admissible(kind, token):
            dropped.append(
                "%s=%s is REJECTED by LeverKind::%s — it would resolve to the default and "
                "measure the baseline" % (axis.name, token, kind["kind"])
            )
            continue
        if baseline_token is not None and token == baseline_token:
            dropped.append(
                "%s=%s equals %s — a no-op run, not a treatment"
                % (axis.name, token, baseline_source)
            )
            continue
        kept.append(token)
    return kept, dropped


# --------------------------------------------------------------------------
# Arm generation
# --------------------------------------------------------------------------


def arm_id(arm):
    """Stable, canonical identity for an assignment."""
    if not arm:
        return "baseline"
    payload = json.dumps(sorted(arm), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def arm_label(arm):
    return ",".join("%s=%s" % (name, value) for name, value in sorted(arm)) or "(default)"


def _dedupe(arms):
    seen = {}
    for arm in arms:
        seen.setdefault(arm_id(arm), arm)
    return list(seen.values())


def generate_arms(space, pool, mode, samples, seed, max_axes, baseline=True, max_arms=256,
                  kinds=None):
    """Arms that are valid BY CONSTRUCTION: never unsafe, never lattice-inert.

    `pool` is the T0-filtered `[(Axis, [tokens])]` the search may move. Random
    mode picks a k-subset and repairs it under the lattice; grid mode takes the
    full product and DROPS the forbidden corners (a grid over a constrained
    lattice legitimately has holes; repairing them would silently change the
    grid).
    """
    rng = random.Random(seed)
    arms = [[]] if baseline else []
    dropped_inert = 0
    if not pool:
        return arms, 0
    if mode == "grid":
        # `None` is a real grid level: an ABSENT axis means "leave at the
        # compiled default", which is the only way a grid can express the
        # mutually exclusive root-phase family at all. Without it every point
        # sets every axis, and the ownership edges annihilate the whole grid.
        levels = [[None] + list(values) for _, values in pool]
        total = 1
        for level in levels:
            total *= len(level)
        if total > max_arms:
            raise SystemExit(
                "grid over %d axes is %d arms (> --max-arms %d); narrow it with --axis"
                % (len(pool), total, max_arms)
            )
        for point in itertools.product(*levels):
            sample = {
                axis.name: token for (axis, _), token in zip(pool, point) if token is not None
            }
            if not sample:
                continue  # the all-default point IS the baseline arm
            if is_inert(sample, space):
                dropped_inert += 1
                continue
            arms.append(sorted(sample.items()))
    else:
        attempts = 0
        while len(arms) < samples + (1 if baseline else 0) and attempts < samples * 64:
            attempts += 1
            width = rng.randint(1, max(1, min(max_axes, len(pool))))
            chosen = rng.sample(pool, width)
            sample = {axis.name: rng.choice(values) for axis, values in chosen}
            sample, added = repair(sample, space, kinds)
            # Everything the operator picked may have been dropped by the
            # lattice, leaving only prerequisites: that is the baseline wearing
            # a treatment label, so it does not get a run.
            if not sample or not set(sample) - added:
                dropped_inert += 1
                continue
            if is_inert(sample, space):
                # Unreachable by construction; counted rather than emitted,
                # because emitting it would cost a full instance budget.
                dropped_inert += 1
                continue
            arms.append(sorted(sample.items()))
            arms = _dedupe(arms)
    return _dedupe(arms), dropped_inert


# --------------------------------------------------------------------------
# THE SCHEDULER HOOK  (deliberately inert -- read the docstring before wiring)
# --------------------------------------------------------------------------


def select_next_arms(pending, results):
    """Order the remaining work. TODAY: FIFO, nothing dropped.

    THIS IS WHERE A MULTI-FIDELITY SCHEDULER (ASHA / successive halving) WOULD
    ATTACH, AND IT IS NOT ATTACHED ON PURPOSE. Such a scheduler kills arms early
    on a cheap signal; on this codebase every cheap signal available is
    uncalibrated or actively misleading:

      * the root objective census moved 0/99 -> 92/99 with ZERO row
        conversions, so ranking by it would promote arms that solve nothing;
      * a reduced budget is not a fidelity either — 5 cifar100 timeout rows at
        {30 s, 50 s} converted 0/10;
      * a single run cannot see a 1-in-3 failure, so any halving decision taken
        on one measurement per rung is noise (marginal rows need >= 3 runs/arm).

    Before wiring anything here, CALIBRATE: measure rank correlation between the
    proposed cheap signal and measured row conversions across >= 20 arms, and
    write the artifact down. Until then FIFO with full-budget runs is the only
    honest schedule, and `--repeats` is the only variance control.
    """
    del results  # unused until a calibrated signal exists
    return list(pending)


# --------------------------------------------------------------------------
# Running one arm
# --------------------------------------------------------------------------


def sidecar(output, extension):
    """Mirror of Rust `Path::with_extension`: replace after the last dot."""
    name = output.name
    base = name[: name.rfind(".")] if "." in name else name
    return output.with_name(base + extension)


def probe_run_flags(ny_bin):
    """Which flags of `benchmarks run` this binary actually has."""
    try:
        done = subprocess.run(
            [str(ny_bin), "benchmarks", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    text = done.stdout + done.stderr
    return {flag for flag in ("--lever", "--cache", "--configs-dir", "--timeout-cap", "--limit",
                             "--vnnlib-version", "--overwrite", "--json") if flag in text}


def child_env(keep_ambient=False):
    """The child environment, with every `NY_*` scrubbed.

    A lever exported into the calling shell is inherited by every child and used
    to be INVISIBLE in the artifact. Scrubbing is not enough on its own — T1
    re-checks the manifest's sealed `ambient_env` — but it stops the harness
    from being the contamination it is trying to detect. `NY_BIN` is scrubbed
    too: it is not a lever, but `ambient_env_from` captures every `NY_*`, so
    leaving it set would trip the check for a real reason.
    """
    env = dict(os.environ)
    if keep_ambient:
        return env, []
    scrubbed = sorted(name for name in env if name.startswith("NY_"))
    for name in scrubbed:
        env.pop(name)
    return env, scrubbed


def build_command(ny_bin, arm, category, output, opts, flags):
    cmd = [
        str(ny_bin),
        "benchmarks",
        "run",
        "--year",
        str(opts.year),
        "--category",
        category,
        "--output",
        str(output),
        "--json",
    ]
    if opts.vnnlib_version and "--vnnlib-version" in flags:
        cmd += ["--vnnlib-version", opts.vnnlib_version]
    if opts.configs_dir:
        cmd += ["--configs-dir", str(opts.configs_dir)]
    if opts.limit is not None:
        cmd += ["--limit", str(opts.limit)]
    if opts.timeout_cap is not None:
        cmd += ["--timeout-cap", str(opts.timeout_cap)]
    if opts.overwrite:
        cmd += ["--overwrite"]
    if opts.cache != "off" and "--cache" in flags:
        cmd += ["--cache", "read-write" if opts.cache == "auto" else opts.cache]
    for name, value in sorted(arm):
        cmd += ["--lever", "%s=%s" % (name, value)]
    return cmd


def run_arm(ny_bin, arm, category, run_dir, opts, flags):
    """Run one (arm, category). Returns the raw run record; validity comes later."""
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / ("%s.csv" % category)
    cmd = build_command(ny_bin, arm, category, output, opts, flags)
    env, scrubbed = child_env(opts.keep_ambient)
    stdout_path = run_dir / "sweep.stdout.json"
    stderr_path = run_dir / "sweep.stderr.log"
    started = time.time()
    timed_out = False
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=env, cwd=str(REPO_ROOT))
        try:
            code = proc.wait(timeout=opts.run_timeout_secs)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            code = proc.wait()
    summary = None
    try:
        summary = json.loads(stdout_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        summary = None
    return {
        "cmd": cmd,
        "exit_code": code,
        "harness_timeout": timed_out,
        "duration_secs": round(time.time() - started, 2),
        "output": str(output),
        "summary": summary,
        "scrubbed_ambient": scrubbed,
        "stderr_tail": _tail(stderr_path),
    }


def _tail(path, limit=4096):
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-limit:].decode("utf-8", errors="replace")


def read_rows(metadata_path):
    rows = []
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as error:
                raise ValueError("%s line %d: %s" % (metadata_path, lineno, error))
    return rows


# --------------------------------------------------------------------------
# T1 RUN-VALIDITY
# --------------------------------------------------------------------------


def _receipt_levers(row):
    """`(status, entries)` from the row's embedded flight record."""
    flight = row.get("flight")
    if not isinstance(flight, dict):
        return "absent", []
    levers = flight.get("levers")
    if not isinstance(levers, dict):
        return "absent", []
    status = levers.get("status", "absent")
    receipt = levers.get("receipt")
    entries = receipt.get("levers", []) if isinstance(receipt, dict) else []
    return status, entries


def _row_name(row):
    return "%s/%s/%s" % (row.get("category", "?"), row.get("onnx", "?"), row.get("vnnlib", "?"))


def validate_run(
    requested_arm,
    requested_year,
    category,
    rows,
    manifest,
    summary,
    strict_receipt=True,
    allow_error_rows=False,
    allow_cached_rows=False,
):
    """T1: is this completed run a measurement of what was requested?

    Every check reads an artifact the run already wrote. A run failing ANY check
    is INVALID and must be excluded from results — not averaged, not counted as
    a negative result. Returns `(valid, failures, warnings, recomputed)`.
    """
    failures = []
    warnings = []
    wanted = sorted((str(name), str(value)) for name, value in requested_arm)
    wanted_map = dict(wanted)

    def fail(check, detail):
        failures.append({"check": check, "detail": detail})

    def warn(check, detail):
        warnings.append({"check": check, "detail": detail})

    # T1.0 -- a run with no rows measured nothing.
    if not rows:
        fail("rows_present", "the bank has zero rows; nothing was measured")

    recomputed = {
        "rows": len(rows),
        "sat": 0,
        "unsat": 0,
        "timeout": 0,
        "unknown": 0,
        "error": 0,
        "capped_rows": 0,
        "sat_rows_without_witness": 0,
        # Rows the bank says were SERVED from the verdict cache rather than
        # measured here. Absent marker == measured: every pre-cache bank, and
        # every row from a binary without `--cache`, reads as a real run.
        "cached_rows": 0,
        "seconds_total": 0.0,
    }
    for row in rows:
        verdict = row.get("verdict")
        if verdict in recomputed:
            recomputed[verdict] += 1
        try:
            recomputed["seconds_total"] += float(row.get("seconds") or 0.0)
        except (TypeError, ValueError):
            pass
        if row.get("capped_from") is not None:
            recomputed["capped_rows"] += 1
        if verdict == "sat" and not isinstance(row.get("witness"), dict):
            recomputed["sat_rows_without_witness"] += 1
        if row.get("from_cache") is True:
            recomputed["cached_rows"] += 1

        # T1.1 -- a rejected raw value silently resolved to the DEFAULT: this
        # row measured the baseline and would be reported as a treatment.
        status, entries = _receipt_levers(row)
        for entry in entries:
            if "rejected_raw" in entry:
                fail(
                    "rejected_raw",
                    "%s: %s rejected raw %r -> resolved to the default, so this row "
                    "measured the BASELINE"
                    % (_row_name(row), entry.get("name"), entry["rejected_raw"]),
                )

        # T1.2 -- without a resolved receipt there is no evidence the arm was
        # applied at all, which is the same hole one layer up.
        if status != "resolved":
            detail = "%s: lever receipt status is %r, so the arm cannot be verified" % (
                _row_name(row),
                status,
            )
            if wanted and strict_receipt:
                fail("receipt_resolved", detail)
            else:
                warn("receipt_resolved", detail)

        # T1.3 -- the row's own record of the arm it was measured under.
        row_arm = sorted((str(name), str(value)) for name, value in row.get("arm", []))
        if row_arm != wanted:
            fail(
                "row_arm_matches",
                "%s: row arm %s != requested %s"
                % (_row_name(row), row_arm or "[]", wanted or "[]"),
            )

        # T1.4 -- and the environment the child actually saw.
        flight = row.get("flight") if isinstance(row.get("flight"), dict) else {}
        ambient = flight.get("ambient_env") if isinstance(flight.get("ambient_env"), dict) else None
        if ambient is not None:
            for name, value in wanted:
                if ambient.get(name) != value:
                    fail(
                        "child_env_carries_arm",
                        "%s: child ambient_env has %s=%r, requested %r"
                        % (_row_name(row), name, ambient.get(name), value),
                    )

        if row.get("category") and category and row["category"] != category:
            fail(
                "category_matches",
                "%s: row category %r != requested %r" % (_row_name(row), row["category"], category),
            )

    # T1.5 -- the PARENT's sealed ambient set. A lever exported into the calling
    # shell is inherited by every child and is invisible in the result.
    ambient_env = (manifest or {}).get("ambient_env") or {}
    for name, value in sorted(ambient_env.items()):
        if not name.startswith("NY_"):
            continue
        if wanted_map.get(name) == value:
            warn(
                "ambient_clean",
                "%s=%s is also exported in the parent shell; it matches the arm, so the "
                "treatment is still the requested one" % (name, value),
            )
        else:
            fail(
                "ambient_clean",
                "parent shell exported %s=%r, which every child inherited and no arm requested"
                % (name, value),
            )

    # T1.6 -- a sat row nobody can revalidate.
    reported_gap = (summary or {}).get("sat_rows_without_witness")
    gap = recomputed["sat_rows_without_witness"] if reported_gap is None else reported_gap
    if gap:
        fail(
            "sat_rows_without_witness",
            "%d sat row(s) banked without a retained witness; organizer-style replay cannot "
            "revalidate them" % gap,
        )

    # T1.7 -- the printed summary must agree with the authoritative bank.
    if summary:
        for key in ("rows", "sat", "unsat", "timeout", "unknown", "error", "capped_rows",
                    "sat_rows_without_witness"):
            if key in summary and summary[key] != recomputed[key]:
                fail(
                    "summary_matches_metadata",
                    "summary %s=%s but the metadata bank has %s"
                    % (key, summary[key], recomputed[key]),
                )
        # `cache_hits` is NOT compared for equality: it counts what THIS
        # invocation served, while the bank accumulates every row a previous
        # invocation banked as served too (`--resume` reuses rows without
        # recounting them). One direction is still an invariant -- a row served
        # here is banked with the marker, so the bank can never mark FEWER rows
        # than the summary claims. A summary reporting hits against an unmarked
        # bank is exactly the byte-indistinguishable replay this check exists
        # for: an old `--cache` binary that predates the marker, or a bank whose
        # rows were rewritten to hide it.
        hits = summary.get("cache_hits")
        if isinstance(hits, int) and hits > recomputed["cached_rows"]:
            fail(
                "cache_hits_accounted",
                "summary claims %d row(s) served from the verdict cache but the bank marks "
                "only %d with from_cache; the served rows cannot be identified"
                % (hits, recomputed["cached_rows"]),
            )

    # T1.8 -- a harness error is a NON-measurement, never a negative result.
    if recomputed["error"] and not allow_error_rows:
        fail("no_error_rows", "%d row(s) failed to run" % recomputed["error"])

    # T1.9 -- the run is of the requested corpus.
    if manifest:
        if requested_year is not None and manifest.get("year") != requested_year:
            fail("manifest_matches_request",
                 "manifest year %s != requested %s" % (manifest.get("year"), requested_year))
        instances = manifest.get("instances_csv") or {}
        if category and instances and category not in instances:
            fail("manifest_matches_request",
                 "manifest pins no instances.csv for category %r" % category)

    # T1.10 -- a bank nobody actually ran. `--cache` serves a row by copying the
    # EARLIER measurement's `SweepRow` wholesale, flight record included, so
    # T1.1/T1.2/T1.4 above all pass against receipts written by a different
    # process on a different day. Every check up to here would therefore green
    # a sweep that started ZERO children. Only the per-row `from_cache` marker
    # separates the two, and this is the check that reads it.
    #
    # Fully served == INVALID, not a warning: an arm whose every row is a replay
    # is evidence about the arm that produced the cache entries, and reporting
    # it as this arm's measurement is the "confident wrong measurement" class
    # this harness exists to refuse. A partial replay is legitimate (it is the
    # saving the cache is for) but is still reported on every run, because a
    # conversion claim resting on replayed rows must be traceable to the run
    # that measured them.
    if rows and recomputed["cached_rows"] == len(rows):
        detail = ("all %d row(s) were SERVED from the verdict cache; this run started no "
                  "child and measured nothing" % len(rows))
        if allow_cached_rows:
            warn("rows_were_measured", detail + " (allowed by --allow-cached-rows)")
        else:
            fail("rows_were_measured", detail)
    elif recomputed["cached_rows"]:
        warn(
            "rows_were_measured",
            "%d of %d row(s) were served from the verdict cache, not measured here"
            % (recomputed["cached_rows"], len(rows)),
        )

    recomputed["seconds_total"] = round(recomputed["seconds_total"], 2)
    if recomputed["capped_rows"]:
        warnings.append({
            "check": "budget_is_official",
            "detail": "%d row(s) ran BELOW their official budget (--timeout-cap): a LOWER BOUND, "
                      "not a competition-comparable score" % recomputed["capped_rows"],
        })

    return (not failures), failures, warnings, recomputed


def validate_artifacts(run_record, arm, category, opts):
    """Load a completed run's artifacts and apply T1. Missing artifacts = INVALID."""
    output = Path(run_record["output"])
    metadata_path = sidecar(output, ".metadata.jsonl")
    manifest_path = sidecar(output, ".manifest.json")
    failures = []
    if run_record.get("harness_timeout"):
        failures.append({"check": "child_completed",
                         "detail": "killed by --run-timeout-secs; a partial sweep is not a result"})
    if run_record["exit_code"] != 0:
        failures.append({"check": "child_completed",
                         "detail": "ny exited %s; stderr tail: %s"
                                   % (run_record["exit_code"], run_record["stderr_tail"][-600:])})
    rows, manifest = [], None
    try:
        rows = read_rows(metadata_path)
    except (OSError, ValueError) as error:
        failures.append({"check": "artifacts_present", "detail": "metadata: %s" % error})
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        failures.append({"check": "artifacts_present", "detail": "manifest: %s" % error})
    valid, t1_failures, warnings, recomputed = validate_run(
        arm,
        opts.year,
        category,
        rows,
        manifest,
        run_record.get("summary"),
        strict_receipt=not opts.allow_unverified_receipt,
        allow_error_rows=opts.allow_error_rows,
        allow_cached_rows=opts.allow_cached_rows,
    )
    failures.extend(t1_failures)
    return {
        "valid": not failures,
        "failures": failures,
        "warnings": warnings,
        "counts": recomputed,
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "executable_sha256": ((manifest or {}).get("executable") or {}).get("sha256"),
        "build_provenance": (manifest or {}).get("build_provenance"),
        "compute_backend": (manifest or {}).get("compute_backend"),
        "host": (manifest or {}).get("host"),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_space(space, kinds, allow_unchecked, as_json=False):
    if as_json:
        payload = {
            "source": space.source,
            "axes": [axis.as_json() for axis in space.axes],
            "edges": [
                {"child": edge.child, "kind": edge.kind, "parent": edge.parent,
                 "bound": edge.bound, "site": edge.site}
                for edge in space.edges
            ],
            "unsafe": space.unsafe_names,
            "instrument_only": space.instrument_only,
            "test_only": space.test_only,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("source: %s" % space.source)
    print()
    width = max(len(axis.name) for axis in space.axes)
    print("%-*s  %-16s  %-46s  %s" % (width, "AXIS", "CLASS", "DELIVERY", "SEARCHABLE VALUES (T0)"))
    for axis in space.axes:
        kept, dropped = admissible_values(axis, kinds, allow_unchecked)
        print("%-*s  %-16s  %-46s  %s" % (width, axis.name, axis.cls, axis.deliver,
                                          ",".join(kept) if kept else "-- none --"))
        for note in dropped:
            print("%-*s  %s" % (width, "", "T0: %s" % note))
    print()
    print("interaction lattice (a violated edge = a full instance budget spent on the baseline):")
    for edge in space.edges:
        print("  %s %s" % (edge.child, edge.render()))
    print()
    print("excluded, Class::Unsafe:   %s" % ", ".join(space.unsafe_names))
    print("excluded, instrument-only: %s" % ", ".join(space.instrument_only))
    print("excluded, test-only:       %s" % ", ".join(space.test_only))
    print()
    print("NOTE: an `env-only` axis is measurable but CANNOT reach a scored run — the scored")
    print("      entry point exports exactly one NY_* variable. Its result is worth zero")
    print("      points until a typed preset key exists.")


def print_leaderboard(records):
    valid = [record for record in records if record["valid"]]
    invalid = [record for record in records if not record["valid"]]
    print()
    print("=" * 78)
    print("%d run(s): %d valid, %d INVALID and excluded" % (len(records), len(valid), len(invalid)))
    for record in invalid:
        print("  INVALID %-10s %-16s %s"
              % (record["arm_id"], record["category"],
                 "; ".join("%s: %s" % (f["check"], f["detail"]) for f in record["failures"])[:160]))
    binaries = {
        record.get("executable_sha256") for record in valid if record.get("executable_sha256")
    }
    if len(binaries) > 1:
        print("REFUSING to rank: these runs used %d DIFFERENT ny binaries (%s). Arms measured "
              "with different executables are not comparable."
              % (len(binaries), ", ".join(sorted(sha[:8] for sha in binaries))))
        return
    by_category = {}
    for record in valid:
        by_category.setdefault(record["category"], []).append(record)
    for category, group in sorted(by_category.items()):
        print()
        print("%s" % category)
        group.sort(key=lambda record: (-record["counts"]["sat"] - record["counts"]["unsat"],
                                       record["wall_secs"]))
        for record in group:
            counts = record["counts"]
            print("  solved %3d (sat %3d unsat %3d) timeout %3d  %7.1fs  %-10s %s"
                  % (counts["sat"] + counts["unsat"], counts["sat"], counts["unsat"],
                     counts["timeout"], record["wall_secs"], record["arm_id"],
                     arm_label(record["arm"])))
    # Only a run whose rows were ALL measured counts as an independent repeat.
    # T1.10 invalidates a fully replayed run, but a PARTIALLY replayed one stays
    # valid, and admission rule 2 makes that reachable without tampering: a row
    # measured under load is stored and then refused on read, so on a busy box
    # repeats 2 and 3 can be mostly byte-copies of repeat 1. Counting those
    # toward the ">= 3 runs per arm" caveat would let the variance control the
    # design doc's W6 rests on be satisfied by replaying one measurement.
    repeats, partial = {}, {}
    for record in valid:
        key = (record["arm_id"], record["category"])
        cached = record.get("counts", {}).get("cached_rows", 0)
        if cached:
            partial[key] = partial.get(key, 0) + 1
            continue
        repeats[key] = repeats.get(key, 0) + 1
    if partial:
        print()
        print("NOTE: %d valid run(s) served some rows from the cache and are NOT counted as "
              "independent repeats. Re-run with --cache off for variance." % sum(partial.values()))
    worst = max(repeats.values()) if repeats else 0
    if valid and worst < 3:
        print()
        print("CAVEAT: at most %d fully-measured run(s) per (arm, category). A one- or two-row "
              "difference is NOT a result at this sample size — a single run cannot see a "
              "1-in-3 failure, and marginal rows need >= 3 runs per arm." % worst)


# --------------------------------------------------------------------------
# Self-test: arm generation and the T1 validators, against synthetic artifacts
# --------------------------------------------------------------------------


def _clean_row(arm=(), verdict="unsat", category="cifar100"):
    """A synthetic bank row shaped like `SweepRow` with a flight-v3 sidecar."""
    ambient = {name: value for name, value in arm}
    return {
        "category": category,
        "onnx": "onnx/m.onnx",
        "vnnlib": "vnnlib/p.vnnlib",
        "occurrence": 0,
        "instance_index": 0,
        "verdict": verdict,
        "seconds": 12.5,
        "budget_secs": 100,
        "capped_from": None,
        "detail": None,
        "arm": [list(pair) for pair in sorted(arm)],
        "flight": {
            "schema_version": 3,
            "ambient_env": ambient,
            "levers": {
                "status": "resolved",
                "receipt": {
                    "schema": "ny-levers/receipt/v2",
                    "levers": [{"name": name, "value": value, "source": "legacy_env"}
                               for name, value in sorted(arm)],
                },
            },
        },
        **({"witness": {"path": "witnesses/w.counterexample"}} if verdict == "sat" else {}),
    }


def _clean_manifest(year=2025, category="cifar100"):
    return {
        "schema_version": 1,
        "year": year,
        "timeout_cap": None,
        "executable": {"canonical_path": "/repo/target/release/ny", "sha256": "a" * 64},
        "build_provenance": "test",
        "compute_backend": "cuda",
        "host": "gb10",
        "ambient_env": {},
        "instances_csv": {category: {"canonical_path": "/i.csv", "sha256": "b" * 64}},
    }


def _summary_of(rows):
    counts = {"rows": len(rows), "sat": 0, "unsat": 0, "timeout": 0, "unknown": 0, "error": 0,
              "capped_rows": 0, "sat_rows_without_witness": 0, "wall_secs": 1.0}
    for row in rows:
        counts[row["verdict"]] += 1
        if row.get("capped_from") is not None:
            counts["capped_rows"] += 1
        if row["verdict"] == "sat" and not isinstance(row.get("witness"), dict):
            counts["sat_rows_without_witness"] += 1
    return counts


def self_test():
    results = []

    def check(name, condition, detail=""):
        results.append((name, bool(condition), detail))

    space = parse_space_rs()
    kinds = parse_lever_kinds()

    # --- the mirrored lattice must reproduce space.rs's own documented refusals
    check("space parses", len(space.axes) >= 10 and len(space.edges) >= 5,
          "%d axes, %d edges" % (len(space.axes), len(space.edges)))
    inert = is_inert({"NY_INTERM_ROW_CHUNKS": "64"}, space)
    check("chunks without the comprehensive sweep are refused",
          inert and inert.axis == "NY_INTERM_ROW_CHUNKS"
          and "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN" in inert.because)
    check("chunks with the comprehensive sweep are accepted",
          not is_inert({"NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN": "1",
                        "NY_INTERM_ROW_CHUNKS": "64"}, space))
    inert = is_inert({"NY_ROOT_PHASE_RESIDENT_CROWN": "1",
                      "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN": "1"}, space)
    check("phase-resident suppresses the comprehensive sweep",
          inert and inert.axis == "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"
          and "suppressed" in inert.because)
    inert = is_inert({"NY_MARGIN_ROW_GPU_BATCH": "1"}, space)
    check("margin-row batch requires the seam",
          inert and inert.axis == "NY_MARGIN_ROW_GPU_BATCH")
    check("wide-demanded is reachable at (0,0,1)",
          not is_inert({"NY_ROOT_WIDE_DEMANDED_INTERM_CROWN": "1"}, space))
    check("an explicit disarm is never inert",
          not is_inert({"NY_INTERM_ROW_CHUNKS": "0"}, space)
          and not is_inert({"NY_MARGIN_ROW_GPU_BATCH": "0"}, space))
    check("an Unsafe axis is refused",
          (is_inert({"NY_STRIP_TERMINAL_SOFTMAX": "1"}, space) or None)
          and "Unsafe" in is_inert({"NY_STRIP_TERMINAL_SOFTMAX": "1"}, space).because)

    # --- repair closes a sample constructively rather than by rejection
    repaired, added = repair({"NY_MARGIN_ROW_GPU_BATCH": "1"}, space, kinds)
    check("repair arms the prerequisite",
          repaired.get("NY_MARGIN_ROW_GPU") == "1" and added == {"NY_MARGIN_ROW_GPU"}
          and not is_inert(repaired, space), str(repaired))
    # The comprehensive sweep can never be armed while the resident phase owns
    # the slot, so the chunks axis that depends on it has to go -- and repair
    # must NOT oscillate between arming and dropping the parent.
    repaired, _added = repair(
        {"NY_ROOT_PHASE_RESIDENT_CROWN": "1", "NY_INTERM_ROW_CHUNKS": "16"}, space, kinds)
    check("repair drops a child the owning phase suppresses",
          not is_inert(repaired, space) and "NY_INTERM_ROW_CHUNKS" not in repaired, str(repaired))

    # --- T0: every emitted value must parse, and must not equal the default
    check("Bool rejects 'true' (the parser arms on exact '1')",
          not value_admissible({"kind": "Bool"}, "true")
          and value_admissible({"kind": "Bool"}, "1"))
    check("F64ClosedTrimmed admits BOTH endpoints (0.0 is the disarmed control)",
          value_admissible({"kind": "F64ClosedTrimmed", "min": 0.0, "max": 1.0}, "0.0")
          and value_admissible({"kind": "F64ClosedTrimmed", "min": 0.0, "max": 1.0}, "1.0")
          and value_admissible({"kind": "F64ClosedTrimmed", "min": 0.0, "max": 1.0}, "0.35")
          and not value_admissible({"kind": "F64ClosedTrimmed", "min": 0.0, "max": 1.0}, "1.5")
          and not value_admissible({"kind": "F64ClosedTrimmed", "min": 0.0, "max": 1.0}, "nope"))
    check("F64Open is OPEN at both ends",
          not value_admissible({"kind": "F64Open", "min": 0.0, "max": 0.9}, "0")
          and not value_admissible({"kind": "F64Open", "min": 0.0, "max": 0.9}, "25")
          and value_admissible({"kind": "F64Open", "min": 0.0, "max": 0.9}, "0.25"))
    check("decls scrape found the live kinds",
          kinds.get("NY_MARGIN_ROW_GPU", {}).get("kind") == "Bool"
          and kinds.get("NY_INTERM_ROW_CHUNKS", {}).get("kind") == "UsizeTrimmed",
          "%d levers scraped" % len(kinds))

    pool = []
    t0_notes = []
    for axis in space.axes:
        kept, dropped = admissible_values(axis, kinds, False)
        t0_notes.extend(dropped)
        if kept:
            pool.append((axis, kept))
    check("T0 drops at least one declared value", bool(t0_notes), "; ".join(t0_notes)[:200])

    arms, dropped_inert = generate_arms(space, pool, "random", 120, 7, 3, kinds=kinds)
    bad_lattice = [arm for arm in arms if is_inert(dict(arm), space)]
    check("no generated arm is lattice-inert", not bad_lattice, str(bad_lattice[:2]))
    bad_value = [
        (name, value)
        for arm in arms
        for name, value in arm
        if not value_admissible(kinds.get(name), value)
    ]
    check("no generated value is rejected by its declared parser",
          not bad_value, str(bad_value[:3]))
    # A value equal to the compiled default is a no-op; it may appear ONLY as a
    # lattice prerequisite (`space::expand` refuses a child whose parent is
    # merely defaulted), never as the only thing an arm moves.
    # ... and "its default" means DefaultSpec only for an ENV-ONLY axis. For a
    # preset-delivered one the preset key is the baseline, so `NAME=0` against a
    # preset that sets it true is a real treatment and DefaultSpec says nothing
    # about it. This exemption is the same rule `admissible_values` applies at
    # T0; without it the two disagree, and the invariant fires on arms T0
    # deliberately kept.
    preset_delivered = {
        axis.name for axis in space.axes
        if str(getattr(axis, "deliver", "") or "").startswith("preset")
    }
    no_treatment = [
        arm for arm in arms
        if arm and not [
            (name, value) for name, value in arm
            if name in preset_delivered
            or kinds.get(name, {}).get("default") is None
            or value != kinds[name]["default"]
        ]
    ]
    check("every non-baseline arm moves at least one value off its default",
          not no_treatment, str(no_treatment[:2]))
    check("the baseline arm is always generated", [] in arms)

    # A grid must carry an ABSENT level per axis. Without it every point sets
    # every axis, the root-phase ownership edges refuse all of them, and the
    # grid silently collapses to nothing but the baseline.
    root_family = [
        (axis, values)
        for axis, values in pool
        if axis.name in ("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN", "NY_INTERM_ROW_CHUNKS",
                         "NY_ROOT_PHASE_RESIDENT_CROWN")
    ]
    grid_arms, grid_inert = generate_arms(space, root_family, "grid", 0, 0, 3, kinds=kinds)
    check("a grid over mutually exclusive axes still yields arms",
          len(grid_arms) > 1 and grid_inert > 0
          and not [arm for arm in grid_arms if is_inert(dict(arm), space)],
          "%d arms, %d inert corners dropped" % (len(grid_arms), grid_inert))
    check("arm ids are canonical and order-free",
          arm_id([("A", "1"), ("B", "2")]) == arm_id([("B", "2"), ("A", "1")])
          and arm_id([]) == "baseline")
    check("generation is deterministic under a seed",
          generate_arms(space, pool, "random", 20, 11, 3, kinds=kinds)[0]
          == generate_arms(space, pool, "random", 20, 11, 3, kinds=kinds)[0])
    check("sidecar paths mirror Rust with_extension",
          sidecar(Path("/b/cifar100.csv"), ".metadata.jsonl") == Path("/b/cifar100.metadata.jsonl"))
    del dropped_inert

    # --- T1 against synthetic artifacts
    arm = [("NY_MARGIN_ROW_GPU", "1")]
    rows = [_clean_row(arm), _clean_row(arm, verdict="timeout")]
    rows[1]["instance_index"] = 1
    valid, failures, _warnings, counts = validate_run(
        arm, 2025, "cifar100", rows, _clean_manifest(), _summary_of(rows))
    check("a clean run is VALID", valid, str(failures))
    check("counts are recomputed from the bank",
          counts["unsat"] == 1 and counts["timeout"] == 1)

    dirty = [json.loads(json.dumps(row)) for row in rows]
    dirty[0]["flight"]["levers"]["receipt"]["levers"][0]["rejected_raw"] = "true"
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", dirty, _clean_manifest(), _summary_of(dirty))
    check("a rejected_raw row is INVALID",
          not valid and any(f["check"] == "rejected_raw" for f in failures))

    dirty = [json.loads(json.dumps(row)) for row in rows]
    dirty[0]["arm"] = [["NY_MARGIN_ROW_GPU", "0"]]
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", dirty, _clean_manifest(), _summary_of(dirty))
    check("a row measured under a different arm is INVALID",
          not valid and any(f["check"] == "row_arm_matches" for f in failures))

    dirty = [json.loads(json.dumps(row)) for row in rows]
    dirty[0]["flight"]["ambient_env"] = {}
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", dirty, _clean_manifest(), _summary_of(dirty))
    check("a child that never saw the arm is INVALID",
          not valid and any(f["check"] == "child_env_carries_arm" for f in failures))

    dirty = [json.loads(json.dumps(row)) for row in rows]
    dirty[0]["flight"]["levers"] = {"status": "not_materialized"}
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", dirty, _clean_manifest(), _summary_of(dirty))
    check("an unresolved lever receipt is INVALID under a non-empty arm",
          not valid and any(f["check"] == "receipt_resolved" for f in failures))
    unresolved = _clean_row()
    unresolved["flight"]["levers"] = {"status": "not_materialized"}
    valid, _f, warnings, _c = validate_run(
        [], 2025, "cifar100", [unresolved], _clean_manifest(), None)
    check("the same receipt gap is only a WARNING on a baseline run",
          valid and any(w["check"] == "receipt_resolved" for w in warnings))

    sat_rows = [_clean_row(arm, verdict="sat")]
    del sat_rows[0]["witness"]
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", sat_rows, _clean_manifest(), _summary_of(sat_rows))
    check("a sat row without a witness is INVALID",
          not valid and any(f["check"] == "sat_rows_without_witness" for f in failures))

    manifest = _clean_manifest()
    manifest["ambient_env"] = {"NY_BAB_RESNET_WIDE": "1"}
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", rows, manifest, _summary_of(rows))
    check("an ambient lever nobody requested is INVALID",
          not valid and any(f["check"] == "ambient_clean" for f in failures))
    manifest["ambient_env"] = {"NY_MARGIN_ROW_GPU": "1"}
    valid, _f, warnings, _c = validate_run(
        arm, 2025, "cifar100", rows, manifest, _summary_of(rows))
    check("an ambient lever equal to the arm is only a WARNING",
          valid and any(w["check"] == "ambient_clean" for w in warnings))

    bad_summary = dict(_summary_of(rows), unsat=99)
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", rows, _clean_manifest(), bad_summary)
    check("summary/metadata disagreement is INVALID",
          not valid and any(f["check"] == "summary_matches_metadata" for f in failures))

    # --- the verdict cache must not be able to fake a run (T1.10 / T1.7)
    # Every row served, so no child ever started. Note what this bank does NOT
    # trip: the arm, the receipt and the child's ambient_env are all present and
    # correct, because the served rows carry an earlier run's flight record
    # verbatim. Only `from_cache` separates it from a real measurement.
    replayed = [dict(row, from_cache=True) for row in rows]
    valid, failures, _w, counts = validate_run(
        arm, 2025, "cifar100", replayed, _clean_manifest(), _summary_of(replayed))
    check("a bank whose rows were ALL served from the cache is INVALID",
          not valid and any(f["check"] == "rows_were_measured" for f in failures)
          and not any(f["check"] in ("row_arm_matches", "child_env_carries_arm")
                      for f in failures),
          str(failures))
    check("cached rows are counted from the bank", counts["cached_rows"] == 2)

    valid, _f, warnings, _c = validate_run(
        arm, 2025, "cifar100", replayed, _clean_manifest(), _summary_of(replayed),
        allow_cached_rows=True)
    check("--allow-cached-rows downgrades a full replay to a WARNING",
          valid and any(w["check"] == "rows_were_measured" for w in warnings))

    half = [dict(rows[0], from_cache=True), json.loads(json.dumps(rows[1]))]
    valid, _f, warnings, counts = validate_run(
        arm, 2025, "cifar100", half, _clean_manifest(), _summary_of(half))
    check("a partly served bank is VALID but reported",
          valid and counts["cached_rows"] == 1
          and any(w["check"] == "rows_were_measured" for w in warnings))

    valid, _f, warnings, counts = validate_run(
        arm, 2025, "cifar100", rows, _clean_manifest(), _summary_of(rows))
    check("an unmarked bank reads as measured, so pre-cache banks stay VALID",
          valid and counts["cached_rows"] == 0
          and not any(w["check"] == "rows_were_measured" for w in warnings))

    # A binary that serves rows without marking them -- the exact hole the
    # marker closes -- is caught by the summary it prints about itself.
    unaccounted = dict(_summary_of(rows), cache_hits=2, cache_misses=0)
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", rows, _clean_manifest(), unaccounted)
    check("a summary claiming more cache hits than the bank marks is INVALID",
          not valid and any(f["check"] == "cache_hits_accounted" for f in failures))
    accounted = dict(_summary_of(half), cache_hits=1, cache_misses=1)
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", half, _clean_manifest(), accounted)
    check("a summary whose hits the bank accounts for is VALID", valid, str(failures))

    error_rows = [_clean_row(arm, verdict="error")]
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", error_rows, _clean_manifest(), _summary_of(error_rows))
    check("a harness-error row is INVALID (a non-measurement, not a negative result)",
          not valid and any(f["check"] == "no_error_rows" for f in failures))

    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", [], _clean_manifest(), {"rows": 0})
    check("an empty bank is INVALID",
          not valid and any(f["check"] == "rows_present" for f in failures))

    valid, failures, _w, _c = validate_run(
        arm, 2026, "cifar100", rows, _clean_manifest(year=2025), _summary_of(rows))
    check("a bank from another year is INVALID",
          not valid and any(f["check"] == "manifest_matches_request" for f in failures))

    capped = [json.loads(json.dumps(row)) for row in rows]
    capped[0]["capped_from"] = 100
    valid, _f, warnings, _c = validate_run(
        arm, 2025, "cifar100", capped, _clean_manifest(), _summary_of(capped))
    check("a capped row is flagged as a LOWER BOUND, not failed",
          valid and any(w["check"] == "budget_is_official" for w in warnings))

    # --- artifact loading end to end, still with no binary and no GPU
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        output = run_dir / "cifar100.csv"
        sidecar(output, ".metadata.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        sidecar(output, ".manifest.json").write_text(
            json.dumps(_clean_manifest()), encoding="utf-8")
        opts = argparse.Namespace(year=2025, allow_unverified_receipt=False,
                                  allow_error_rows=False, allow_cached_rows=False)
        verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("artifacts load and validate from disk", verdict["valid"], str(verdict["failures"]))
        verdict = validate_artifacts(
            {"output": str(output), "exit_code": 101, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": "boom"},
            arm, "cifar100", opts)
        check("a non-zero child exit is INVALID",
              not verdict["valid"]
              and any(f["check"] == "child_completed" for f in verdict["failures"]))

    failed = 0
    for name, ok, detail in results:
        note = ("  [%s]" % detail) if detail and not ok else ""
        print("%s - %s%s" % ("ok  " if ok else "FAIL", name, note))
        failed += 0 if ok else 1
    print()
    print("%d/%d checks passed" % (len(results) - failed, len(results)))
    if t0_notes:
        print()
        print("T0 pre-flight findings on the LIVE space (these axes/values are excluded):")
        for note in sorted(set(t0_notes)):
            print("  %s" % note)
    return 1 if failed else 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="Search ny's lever space over `ny benchmarks run`, and refuse invalid runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list-space", action="store_true",
                        help="print the searchable axes with class + delivery, then exit")
    parser.add_argument("--json", action="store_true", help="machine-readable --list-space output")
    parser.add_argument("--self-test", action="store_true",
                        help="exercise arm generation and the T1 validators against synthetic "
                             "artifacts; no binary, no benchmarks, no GPU")
    parser.add_argument("--ny-bin",
                        default=os.environ.get("NY_BIN", str(REPO_ROOT / "target/release/ny")),
                        help="ny binary (default: target/release/ny)")
    parser.add_argument("--year", type=int, default=2025, help="VNN-COMP year (default: 2025)")
    parser.add_argument("--category", action="append", default=[], dest="categories",
                        help="category to sweep; repeatable")
    parser.add_argument("--vnnlib-version", default=None, help="forwarded when the binary has it")
    parser.add_argument("--configs-dir", default=None, help="preset directory, forwarded per run")
    parser.add_argument("--limit", type=int, default=None, help="first N instances per category")
    parser.add_argument("--timeout-cap", type=int, default=None,
                        help="LOWER the per-instance budget; capped runs are flagged as a "
                             "lower bound")
    parser.add_argument("--mode", choices=("random", "grid"), default="random")
    parser.add_argument("--samples", type=int, default=8, help="random arms to generate")
    parser.add_argument("--max-axes", type=int, default=2, help="axes moved per random arm")
    parser.add_argument("--max-arms", type=int, default=256, help="refuse a grid larger than this")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axis", action="append", default=[], dest="axis_filter",
                        help="restrict the search to this axis; repeatable")
    parser.add_argument("--class", dest="class_filter", action="append", default=[],
                        choices=("VerdictAffecting", "SafeToSearch"),
                        help="restrict to axes of this class; repeatable")
    parser.add_argument("--repeats", type=int, default=1,
                        help="runs per (arm, category); marginal rows need >= 3. NOTE: only "
                             "FULLY-MEASURED runs count as repeats, so pair this with "
                             "--cache off — under the default --cache auto a repeat can be "
                             "served from the first run and buys no variance")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip the default arm (you almost never want this: without a "
                             "baseline measured on the same binary nothing is interpretable)")
    parser.add_argument("--out-dir", default=None,
                        help="artifact root (default: reports/search/<timestamp>)")
    parser.add_argument("--results", default=None,
                        help="JSONL results file (default: <out-dir>/results.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="print the planned arms and exit")
    parser.add_argument("--cache", choices=("auto", "off", "read", "read-write"), default="auto",
                        help="verdict cache mode, used only if the binary has --cache; a run "
                             "whose rows were ALL served is INVALID (T1.10) because it started "
                             "no child, and a PARTIALLY served run is valid but does not count "
                             "toward --repeats. Use off for anything quoting a time or a spread")
    parser.add_argument("--run-timeout-secs", type=int, default=None,
                        help="kill a sweep that exceeds this wall clock (its result is discarded)")
    parser.add_argument("--overwrite", action="store_true", help="pass --overwrite to each sweep")
    parser.add_argument("--reuse", action="store_true",
                        help="re-validate existing artifacts instead of re-running a measured pair")
    parser.add_argument("--keep-ambient", action="store_true",
                        help="do NOT scrub NY_* from the child environment (contaminated runs "
                             "then fail T1, which is the point)")
    parser.add_argument("--allow-unchecked-domains", action="store_true",
                        help="search axes whose LeverKind could not be read (fail-open; T0 off)")
    parser.add_argument("--allow-unverified-receipt", action="store_true",
                        help="downgrade a missing lever receipt to a warning")
    parser.add_argument("--allow-error-rows", action="store_true",
                        help="do not invalidate a run that contains harness-error rows")
    parser.add_argument("--allow-cached-rows", action="store_true",
                        help="downgrade a FULLY cache-served run (zero children started) from "
                             "INVALID to a warning; the replay is reported either way")
    return parser


def main(argv=None):
    opts = build_parser().parse_args(argv)
    if opts.self_test:
        return self_test()

    space = load_space(opts.ny_bin)
    kinds = parse_lever_kinds()
    if opts.list_space:
        print_space(space, kinds, opts.allow_unchecked_domains, as_json=opts.json)
        return 0

    if not opts.categories:
        print("error: --category is required (or use --list-space / --self-test)", file=sys.stderr)
        return 2

    # T1.1: prove the preset this run will load, before spending an instance budget.
    # `--configs-dir` pointing at a YEAR directory resolves nothing and the binary
    # then runs on bare defaults WITHOUT saying so, which silently turns every arm
    # and its baseline into the same unpresetted configuration.
    preset_doc, preset_known = None, False
    if opts.configs_dir:
        resolved = {}
        for category in opts.categories:
            path = resolve_preset_path(opts.configs_dir, category)
            if path is None:
                print("error: --configs-dir %s resolves NO preset for category %r.\n"
                      "       resolve_preset_path searches <configs-dir>/vnncomp*/ "
                      "SUBDIRECTORIES, so pass the parent (e.g. `configs`), not a year "
                      "directory (`configs/vnncomp25`).\n"
                      "       Running anyway would measure every arm against bare "
                      "defaults and silently invalidate the comparison."
                      % (opts.configs_dir, category), file=sys.stderr)
                return 2
            resolved[category] = path
            print("preset: %s -> %s" % (category, path))
        docs = [load_preset_doc(path) for path in resolved.values()]
        if len(docs) == 1 and docs[0] is not None:
            preset_doc, preset_known = docs[0], True
        elif any(doc is None for doc in docs):
            print("T0: preset(s) resolved but could not be parsed (PyYAML absent?) — "
                  "no preset-delivered token will be dropped")
        else:
            print("T0: %d categories with different presets — no preset-delivered token "
                  "will be dropped, because the baseline differs per category"
                  % len(docs))
    else:
        print("WARNING: no --configs-dir, so every arm AND the baseline run on bare "
              "defaults.\n"
              "         That is a valid A/B against each other, but it is NOT the scored "
              "configuration, and\n"
              "         a preset-delivered axis measured this way says nothing about the "
              "row as it competes.\n"
              "         Pass `--configs-dir configs` unless you specifically want the "
              "unpresetted comparison.")

    pool, notes = [], []
    for axis in space.axes:
        if opts.axis_filter and axis.name not in opts.axis_filter:
            continue
        if opts.class_filter and axis.cls not in opts.class_filter:
            continue
        kept, dropped = admissible_values(axis, kinds, opts.allow_unchecked_domains,
                                          preset_doc=preset_doc, preset_known=preset_known)
        notes.extend(dropped)
        if kept:
            pool.append((axis, kept))
    for note in notes:
        print("T0: %s" % note)
    if not pool and not opts.no_baseline:
        print("T0 left no searchable axis; only the baseline arm will run.")

    arms, dropped_inert = generate_arms(
        space, pool, opts.mode, opts.samples, opts.seed, opts.max_axes,
        baseline=not opts.no_baseline, max_arms=opts.max_arms, kinds=kinds)
    arms = select_next_arms(arms, [])

    print()
    print("space source: %s" % space.source)
    print("%d arm(s) over %d axis/axes; %d lattice-inert sample(s) never generated"
          % (len(arms), len(pool), dropped_inert))
    for arm in arms:
        print("  %-10s %s" % (arm_id(arm), arm_label(arm)))
    print("categories: %s | year %d | repeats %d | %d run(s) total"
          % (", ".join(opts.categories), opts.year, opts.repeats,
             len(arms) * len(opts.categories) * opts.repeats))
    if opts.dry_run:
        return 0

    ny_bin = Path(opts.ny_bin)
    if not ny_bin.exists():
        print("error: ny binary not found at %s" % ny_bin, file=sys.stderr)
        return 2
    flags = probe_run_flags(ny_bin)
    if "--lever" not in flags and any(arms):
        # Falling back to exporting the arm into the environment would defeat
        # the sealing the arm exists to provide: invisible in the artifact,
        # inherited by every later process, silently mixed across a --resume.
        print("error: this ny has no `benchmarks run --lever`; REFUSING to deliver arms through "
              "the environment instead. Rebuild, or run with --no-baseline removed and no arms.",
              file=sys.stderr)
        return 2
    if opts.cache != "off" and "--cache" not in flags:
        print("note: this ny has no `benchmarks run --cache`; every (arm, row) pair will be "
              "measured fresh.")

    default_out = REPO_ROOT / "reports" / "search" / time.strftime("%Y%m%dT%H%M%S")
    out_dir = Path(opts.out_dir or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(opts.results) if opts.results else out_dir / "results.jsonl"
    records = []
    for repeat in range(opts.repeats):
        for arm in arms:
            for category in opts.categories:
                identity = arm_id(arm)
                run_dir = out_dir / identity / ("%s.r%d" % (category, repeat))
                output = run_dir / ("%s.csv" % category)
                reused = False
                if opts.reuse and sidecar(output, ".metadata.jsonl").exists():
                    reused = True
                    run_record = {
                        "cmd": [], "exit_code": 0, "harness_timeout": False, "duration_secs": 0.0,
                        "output": str(output), "summary": None, "scrubbed_ambient": [],
                        "stderr_tail": "",
                    }
                else:
                    print()
                    print("[%s] %s | %s (repeat %d)" % (time.strftime("%H:%M:%S"), identity,
                                                        arm_label(arm), repeat))
                    run_record = run_arm(ny_bin, arm, category, run_dir, opts, flags)
                    if run_record["scrubbed_ambient"]:
                        print("  scrubbed from the child environment: %s"
                              % ", ".join(run_record["scrubbed_ambient"]))
                verdict = validate_artifacts(run_record, arm, category, opts)
                counts = verdict["counts"]
                record = {
                    "schema": RESULT_SCHEMA,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "arm_id": identity,
                    "arm": [list(pair) for pair in arm],
                    "arm_label": arm_label(arm),
                    "category": category,
                    "year": opts.year,
                    "repeat": repeat,
                    "reused_artifacts": reused,
                    "valid": verdict["valid"],
                    "failures": verdict["failures"],
                    "warnings": verdict["warnings"],
                    "counts": counts,
                    "solved": counts["sat"] + counts["unsat"],
                    "lower_bound": bool(counts["capped_rows"]),
                    # Reused artifacts have no parent-side clock, so fall back to
                    # the bank's own per-row seconds rather than reporting 0.
                    "wall_secs": (run_record["summary"] or {}).get(
                        "wall_secs", run_record["duration_secs"] or counts["seconds_total"]),
                    "duration_secs": run_record["duration_secs"],
                    "exit_code": run_record["exit_code"],
                    "space_source": space.source,
                    "executable_sha256": verdict["executable_sha256"],
                    "build_provenance": verdict["build_provenance"],
                    "compute_backend": verdict["compute_backend"],
                    "host": verdict["host"],
                    "output": run_record["output"],
                    "metadata": verdict["metadata"],
                    "manifest": verdict["manifest"],
                    "cmd": run_record["cmd"],
                }
                with open(results_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                records.append(record)
                if verdict["valid"]:
                    print("  VALID   solved %d (sat %d unsat %d) timeout %d"
                          % (record["solved"], counts["sat"], counts["unsat"], counts["timeout"]))
                else:
                    print("  INVALID -- EXCLUDED from results:")
                    for failure in verdict["failures"]:
                        print("    %s: %s" % (failure["check"], failure["detail"]))
                for warning in verdict["warnings"]:
                    print("  warning %s: %s" % (warning["check"], warning["detail"]))

    print_leaderboard(records)
    print()
    print("results: %s" % results_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
