#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Search typed lever arms through ``ny benchmarks run`` and validate results.

Each (arm, category) produces a JSONL record. Invalid runs are excluded from
comparisons. Measured runs and dry runs require the selected binary's compiled
``benchmarks space --json`` schema and ``--lever NAME=VALUE`` delivery. Source
parsing is limited to self-tests and ``--list-space --dev-source-schema``.

Validation contract:
  T0: exclude values that do not parse, match their effective baseline, have
      unmet dependencies, or do not apply to the category. Contextual baselines
      require an axis-specific engagement contract before search.
  T1: check resolved receipts, requested arm, clean ambient environment, retained
      SAT witnesses, executable and model/property hashes, execution evidence
      and agreement between summary and row metadata. Reject errors and wholly
      cache-served runs. Every graduated treatment axis must prove engagement
      on each scored row; other axes establish delivery only. The authoritative
      graduated-family list is ``_AXIS_SPECIFIC_T1_CONTRACTS``.

Random mode changes one sampled axis by default; required parent controls may
be added. Multi-axis results belong to the joint treatment. Paired comparisons
require matching category, repeat, binary, backend and host, and omit invalid,
cache-served and unmatched runs. ``pair-cover`` requires an explicit axis list
and refuses enumeration beyond its limits. No multi-fidelity scheduler is used.

``--instances FILE`` selects exact ``category,onnx,vnnlib`` CSV triples. The
harness seals selector identity, measured rows and resolved input hashes, and
forwards the native ``--instances`` selector rather than approximating a limit.

Examples:
  scripts/ny_search.py --list-space
  scripts/ny_search.py --category cifar100_2024 --mode cover --dry-run
  scripts/ny_search.py --category cifar100_2024 --mode random --samples 8 --dry-run
  scripts/ny_search.py --category cifar100_2024 --instances near_misses.csv \
      --mode cover --configs-dir configs --cache off
  scripts/ny_search.py --self-test
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
import stat
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_RS = REPO_ROOT / "crates" / "ny-levers" / "src" / "space.rs"
DECLS_DIR = REPO_ROOT / "crates" / "ny-levers" / "src" / "decls"
RESULT_SCHEMA = "ny-search/result/v2"
DEFAULT_MAX_CANDIDATES = 100_000
INSTANCE_SELECTOR_SCHEMA = "ny-search/instance-selectors/v1"
MEASURED_ROW_SET_SCHEMA = "ny-search/measured-row-set/v1"
SELECTED_ROW_INPUTS_SCHEMA = "ny-cli/selected-row-inputs/v2"
AMBIENT_ENV_SCHEMA = "ny-cli/ambient-env/v3"
RAYON_WORKER_ENV = frozenset(("RAYON_NUM_THREADS", "RAYON_RS_NUM_CPUS"))
SELF_TEST_BUILD_ID = "NY_BUILD_ID_V1 self-test"

# This exact command is part of the measured-binary contract. Historical command
# spellings are deliberately not accepted: a competition search must prove that
# the executable it will run publishes the same schema used to generate arms.
SPACE_PROBE = ("benchmarks", "space", "--json")
SPACE_SCHEMA = "ny-levers/search-space/v6"


class SpaceSchemaUnavailable(RuntimeError):
    """The selected executable cannot authoritatively describe its lever space."""


# --------------------------------------------------------------------------
# The search space, mirrored from crates/ny-levers/src/space.rs
# --------------------------------------------------------------------------


class Axis:
    """One searchable dimension: name, admissible tokens, class, delivery."""

    def __init__(self, name, domain_kind, values, cls, deliver, why="",
                 absence_baseline=None, category_prefixes=None,
                 preset_baseline_keys=None):
        self.name = name
        self.domain_kind = domain_kind  # "bool" | "grid" | "enum"
        self.values = list(values)  # emitted verbatim, exactly as the parser sees them
        self.cls = cls  # "VerdictAffecting" | "SafeToSearch" | "Unsafe"
        self.deliver = deliver  # "preset:<key>" | "env-only"
        self.why = why
        self.absence_baseline = dict(
            absence_baseline or {"kind": "declaration_default"}
        )
        self.category_prefixes = list(category_prefixes or [])
        if preset_baseline_keys is None:
            preset_baseline_keys = (
                [deliver[len("preset:"):]] if deliver.startswith("preset:") else []
            )
        self.preset_baseline_keys = list(preset_baseline_keys)

    def as_json(self):
        return {
            "name": self.name,
            "domain": self.domain_kind,
            "values": self.values,
            "class": self.cls,
            "deliver": self.deliver,
            "absence_baseline": self.absence_baseline,
            "category_prefixes": self.category_prefixes,
            "preset_baseline_keys": self.preset_baseline_keys,
        }


class Edge:
    """`child` does nothing unless `requires` holds. Mirrors `space::Edge`."""

    def __init__(self, child, kind, parent=None, parents=None, bound=None, token=None,
                 site=""):
        self.child = child
        # "armed" | "any_armed" | "non_zero" | "greater_than" | "not_armed" |
        # "not_equals" | "absent" | "equals"
        self.kind = kind
        self.parent = parent
        self.parents = list(parents or [])
        self.bound = bound
        self.token = token
        self.site = site

    def render(self):
        if self.kind == "armed":
            return 'requires `%s` = "1" (%s)' % (self.parent, self.site)
        if self.kind == "any_armed":
            alternatives = " or ".join('`%s`' % parent for parent in self.parents)
            return 'requires any of %s = "1" (%s)' % (alternatives, self.site)
        if self.kind == "non_zero":
            return 'requires `%s` present and not "0" (%s)' % (self.parent, self.site)
        if self.kind == "greater_than":
            return "requires `%s` > %d (%s)" % (self.parent, self.bound, self.site)
        if self.kind == "equals":
            return "requires `%s` = %r (%s)" % (self.parent, self.token, self.site)
        if self.kind == "not_equals":
            return "suppressed while `%s` = %r (%s)" % (
                self.parent, self.token, self.site)
        if self.kind == "absent":
            return "requires `%s` to remain absent (%s)" % (self.parent, self.site)
        return 'suppressed while `%s` = "1" (%s)' % (self.parent, self.site)


class Space:
    def __init__(self, axes, edges, unsafe_names, instrument_only, test_only, source,
                 declarations=None):
        self.axes = axes
        self.edges = edges
        self.unsafe_names = list(unsafe_names)
        self.instrument_only = list(instrument_only)
        self.test_only = list(test_only)
        self.source = source
        # `None` is reserved for the explicit developer source parser. Compiled
        # Current compiled schemas must publish a non-empty declaration contract;
        # they fail before construction rather than mixing checkout data into a
        # measured executable.
        self.declarations = declarations
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
    """Rust string literals in code, excluding line and nested block comments.

    A regex over every quoted span also reads examples in comments. That made
    `INSTRUMENT_ONLY` contain prose such as ``"never arm on a scored run"`` and
    meant the source fallback did not describe the Rust constant at all. This
    small lexer only needs ordinary Rust strings (the search contract uses no
    raw/byte strings), but correctly keeps comment markers inside a string.
    """
    values = []
    index = 0
    length = len(body)
    while index < length:
        if body.startswith("//", index):
            newline = body.find("\n", index + 2)
            index = length if newline < 0 else newline + 1
            continue
        if body.startswith("/*", index):
            depth = 1
            index += 2
            while index < length and depth:
                if body.startswith("/*", index):
                    depth += 1
                    index += 2
                elif body.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            continue
        if body[index] != '"':
            index += 1
            continue

        index += 1
        chars = []
        while index < length:
            char = body[index]
            if char == '"':
                index += 1
                values.append("".join(chars))
                break
            if char == "\\" and index + 1 < length:
                escaped = body[index + 1]
                if escaped in ('"', "\\"):
                    chars.append(escaped)
                else:
                    # Keep unfamiliar escapes and line continuations in the
                    # same form the old parser returned. `_rust_prose` performs
                    # its own whitespace/backslash normalization.
                    chars.extend((char, escaped))
                index += 2
                continue
            chars.append(char)
            index += 1
    return values


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


def _declared_lever_name(decls_dir, module, constant):
    """Resolve exactly one declaration in the source checkout."""
    path = Path(decls_dir) / (module + ".rs")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("cannot resolve axis declaration %s::%s: %s is unreadable: %s"
                         % (module, constant, path, error)) from error
    declarations = list(re.finditer(
        r"\bpub\s+" + re.escape(constant) + r"\s*=\s*LeverDecl\s*\{", text,
    ))
    if len(declarations) != 1:
        raise ValueError("cannot uniquely resolve axis declaration %s::%s: "
                         "%s does not declare exactly one matching LeverDecl"
                         % (module, constant, path))
    name = re.match(r'\s*name:\s*"([^"]+)"', text[declarations[0].end():])
    if not name:
        raise ValueError("%s declares %s without a leading `name:` literal" % (path, constant))
    return name.group(1)


def _axis_name(chunk, decls_dir):
    literal = re.search(r'name:\s*"([^"]+)"', chunk)
    if literal:
        return literal.group(1)
    reference = re.search(r"name:\s*crate::decls::(\w+)::(\w+)\.name\b", chunk)
    if reference:
        return _declared_lever_name(decls_dir, *reference.groups())
    raise ValueError("axis record has an unsupported name: %r" % chunk[:120])


def _parse_axes(body, decls_dir=DECLS_DIR):
    axes = []
    for chunk in _parse_axis_records(body):
        name = _axis_name(chunk, decls_dir)
        kind, values = _parse_domain(chunk)
        cls = re.search(r"class:\s*Class::(\w+)", chunk).group(1)
        preset = re.search(r'deliver:\s*Deliver::PresetKey\("([^"]+)"\)', chunk)
        deliver = "preset:%s" % preset.group(1) if preset else "env-only"
        why = re.search(r"why:(.*)$", chunk, re.S)
        prose = _rust_prose(why.group(1)) if why else ""
        axes.append(Axis(name, kind, values, cls, deliver, prose))
    return axes


def _parse_edges(body):
    positioned = []
    for match in re.finditer(
        r'child:\s*"([^"]+)",\s*requires:\s*Requirement::(\w+)\(\s*"([^"]+)"'
        r'(?:\s*,\s*(?:"([^"]+)"|(\d+)))?\s*\),\s*site:\s*((?:"[^"]*"\s*)+)',
        body,
        re.S,
    ):
        child, requirement, parent, token, bound, site = match.groups()
        kind = {
            "Armed": "armed",
            "NonZero": "non_zero",
            "GreaterThan": "greater_than",
            "NotArmed": "not_armed",
            "NotEquals": "not_equals",
            "Absent": "absent",
            "Equals": "equals",
        }[requirement]
        positioned.append(
            (match.start(), Edge(
                child,
                kind,
                parent,
                bound=int(bound) if bound else None,
                token=token,
                site=_rust_prose(site),
            ))
        )
    for match in re.finditer(
        r'child:\s*"([^"]+)",\s*requires:\s*Requirement::AnyArmed\('
        r'\s*&\[(.*?)\]\s*\),\s*site:\s*((?:"[^"]*"\s*)+)',
        body,
        re.S,
    ):
        child, raw_parents, site = match.groups()
        positioned.append(
            (match.start(), Edge(
                child,
                "any_armed",
                parents=_rust_str_list(raw_parents),
                site=_rust_prose(site),
            ))
        )
    return [edge for _offset, edge in sorted(positioned, key=lambda item: item[0])]


def _parse_absence_baselines(text):
    """Parse `ABSENCE_BASELINES` into the compiled-schema object shape."""
    body = _rust_block(
        text, "const ABSENCE_BASELINES: &[(&str, AbsenceBaseline)] = &["
    )
    if not body.strip():
        raise ValueError("source search space has no ABSENCE_BASELINES contract")
    entry_re = re.compile(
        r'\(\s*"(NY_[A-Z0-9_]+)"\s*,\s*AbsenceBaseline::'
        r'(DeclarationDefault|Distinct|Equivalent|Contextual)'
        r'(?:\s*\(\s*"([^"]+)"\s*\))?\s*,?\s*\)',
        re.S,
    )
    mentioned = re.findall(
        r'\(\s*"(NY_[A-Z0-9_]+)"\s*,\s*AbsenceBaseline::', body, re.S
    )
    parsed = {}
    names = {
        "DeclarationDefault": "declaration_default",
        "Distinct": "distinct",
        "Equivalent": "equivalent",
        "Contextual": "contextual",
    }
    for match in entry_re.finditer(body):
        name, variant, token = match.groups()
        if name in parsed:
            raise ValueError("duplicate absence baseline for %s" % name)
        if variant == "Equivalent":
            if token is None:
                raise ValueError("Equivalent absence baseline for %s has no token" % name)
            parsed[name] = {"kind": names[variant], "token": token}
        else:
            if token is not None:
                raise ValueError("non-Equivalent absence baseline for %s has a token" % name)
            parsed[name] = {"kind": names[variant]}
    if len(parsed) != len(mentioned):
        raise ValueError(
            "parsed %d of %d ABSENCE_BASELINES entries" % (len(parsed), len(mentioned))
        )
    return parsed


def _parse_category_prefixes(text):
    """Parse `CATEGORY_PREFIXES` into `{axis: [lowercase category prefix]}`."""
    body = _rust_block(
        text, "const CATEGORY_PREFIXES: &[(&str, &[&str])] = &["
    )
    if not body.strip():
        raise ValueError("source search space has no CATEGORY_PREFIXES contract")
    entry_re = re.compile(
        r'\(\s*"(NY_[A-Z0-9_]+)"\s*,\s*&\[([^\]]*)\]\s*,?\s*\)', re.S
    )
    mentioned = re.findall(
        r'\(\s*"(NY_[A-Z0-9_]+)"\s*,\s*&\[', body, re.S
    )
    parsed = {}
    for match in entry_re.finditer(body):
        name, raw_prefixes = match.groups()
        if name in parsed:
            raise ValueError("duplicate category applicability for %s" % name)
        prefixes = _rust_str_list(raw_prefixes)
        if (not prefixes or any(not prefix or prefix != prefix.lower()
                                for prefix in prefixes)
                or len(prefixes) != len(set(prefixes))):
            raise ValueError("invalid category prefixes for %s" % name)
        parsed[name] = prefixes
    if len(parsed) != len(mentioned):
        raise ValueError(
            "parsed %d of %d CATEGORY_PREFIXES entries" % (len(parsed), len(mentioned))
        )
    return parsed


def _parse_preset_baseline_keys(text):
    """Parse ordered preset-key fallback overrides from `space.rs`."""
    body = _rust_block(
        text, "const PRESET_BASELINE_KEYS: &[(&str, &[&str])] = &["
    )
    if not body.strip():
        raise ValueError("source search space has no PRESET_BASELINE_KEYS contract")
    entry_re = re.compile(
        r'\(\s*"(NY_[A-Z0-9_]+)"\s*,\s*&\[([^\]]*)\]\s*,?\s*\)', re.S
    )
    mentioned = re.findall(
        r'\(\s*"(NY_[A-Z0-9_]+)"\s*,\s*&\[', body, re.S
    )
    parsed = {}
    for match in entry_re.finditer(body):
        name, raw_keys = match.groups()
        if name in parsed:
            raise ValueError("duplicate preset baseline keys for %s" % name)
        keys = _rust_str_list(raw_keys)
        if (not keys or any(not key for key in keys)
                or len(keys) != len(set(keys))):
            raise ValueError("invalid preset baseline keys for %s" % name)
        parsed[name] = keys
    if len(parsed) != len(mentioned):
        raise ValueError(
            "parsed %d of %d PRESET_BASELINE_KEYS entries"
            % (len(parsed), len(mentioned))
        )
    return parsed


def parse_space_rs(path=SPACE_RS):
    """Parse the search space out of `space.rs` — the fallback source."""
    text = path.read_text(encoding="utf-8")
    decls_dir = Path(path).parent / "decls"
    axes = _parse_axes(_rust_block(text, "const AXES: &[Axis] = &["), decls_dir)
    edges = _parse_edges(_rust_block(text, "const EDGES: &[Edge] = &["))
    unsafe_axes = _parse_axes(
        _rust_block(text, "const UNSAFE_AXES: &[Axis] = &["), decls_dir
    )
    instruments = _rust_str_list(_rust_block(text, "const INSTRUMENT_ONLY: &[&str] = &["))
    test_only = _rust_str_list(_rust_block(text, "const TEST_ONLY: &[&str] = &["))
    if not axes:
        raise ValueError("parsed no axes from %s" % path)
    known = {axis.name for axis in axes}
    absence_baselines = _parse_absence_baselines(text)
    category_prefixes = _parse_category_prefixes(text)
    preset_baseline_keys = _parse_preset_baseline_keys(text)
    unknown_absence = set(absence_baselines).difference(known)
    if unknown_absence:
        raise ValueError(
            "absence metadata names unknown axes: %s"
            % ", ".join(sorted(unknown_absence))
        )
    unknown_categories = set(category_prefixes).difference(known)
    if unknown_categories:
        raise ValueError(
            "category metadata names unknown axes: %s"
            % ", ".join(sorted(unknown_categories))
        )
    unknown_preset_baselines = set(preset_baseline_keys).difference(known)
    if unknown_preset_baselines:
        raise ValueError(
            "preset baseline metadata names unknown axes: %s"
            % ", ".join(sorted(unknown_preset_baselines))
        )
    for axis in axes:
        axis.absence_baseline = dict(
            absence_baselines.get(axis.name, {"kind": "declaration_default"})
        )
        if (axis.absence_baseline["kind"] == "equivalent"
                and axis.absence_baseline["token"] not in axis.values):
            raise ValueError(
                "absence token %r is outside %s's domain"
                % (axis.absence_baseline["token"], axis.name)
            )
        axis.category_prefixes = list(category_prefixes.get(axis.name, []))
        direct_key = (
            axis.deliver[len("preset:"):]
            if axis.deliver.startswith("preset:")
            else None
        )
        axis.preset_baseline_keys = list(
            preset_baseline_keys.get(
                axis.name,
                [direct_key] if direct_key is not None else [],
            )
        )
        if (
            (direct_key is None and axis.preset_baseline_keys)
            or (direct_key is not None and (
                not axis.preset_baseline_keys
                or axis.preset_baseline_keys[0] != direct_key
            ))
        ):
            raise ValueError(
                "preset baseline keys for %s do not begin with its delivery key"
                % axis.name
            )
    for edge in edges:
        parents = edge.parents if edge.kind == "any_armed" else [edge.parent]
        if (edge.child not in known or not parents
                or len(parents) != len(set(parents))
                or any(parent not in known for parent in parents)):
            raise ValueError(
                "edge %s <- %s names an unknown/invalid axis"
                % (edge.child, parents)
            )
    return Space(
        axes,
        edges,
        [axis.name for axis in unsafe_axes],
        instruments,
        test_only,
        "parsed %s" % path.relative_to(REPO_ROOT),
    )


def _declarations_from_payload(raw):
    """Normalize the optional binary declaration map to `parse_lever_kinds` form."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        entries = raw.items()
    elif isinstance(raw, list):
        entries = []
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ValueError("binary search-space declaration list has a malformed record")
            entries.append((item["name"], item))
    else:
        raise ValueError("binary search-space `declarations` must be an object or list")

    declarations = {}
    for name, item in entries:
        if not isinstance(name, str) or not isinstance(item, dict):
            raise ValueError("binary search-space declaration map has a malformed record")
        kind = item.get("kind")
        if not isinstance(kind, str):
            raise ValueError("binary declaration %r has no string `kind`" % name)
        record = {"kind": kind, "default": item.get("default")}
        default = record["default"]
        if isinstance(default, bool):
            record["default"] = "1" if default else "0"
        elif default is not None and not isinstance(default, str):
            record["default"] = str(default)
        if item.get("min") is not None:
            if kind in ("Usize", "UsizeTrimmedOverride", "UsizeOverride"):
                minimum = item["min"]
                try:
                    normalized_minimum = int(minimum)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(
                        "binary %s declaration %r has an invalid `min`" % (kind, name)
                    ) from error
                maximum_usize = (1 << (8 * struct.calcsize("P"))) - 1
                if (
                    isinstance(minimum, bool)
                    or normalized_minimum < 0
                    or normalized_minimum > maximum_usize
                    or float(normalized_minimum) != float(minimum)
                ):
                    raise ValueError(
                        "binary %s declaration %r has a non-integral/out-of-range `min`"
                        % (kind, name)
                    )
                record["min"] = normalized_minimum
            else:
                record["min"] = float(item["min"])
        if item.get("max") is not None:
            record["max"] = float(item["max"])
        if item.get("members") is not None:
            if not isinstance(item["members"], list):
                raise ValueError("binary declaration %r has non-list `members`" % name)
            record["members"] = [str(member) for member in item["members"]]
        declarations[name] = record
    return declarations


def _absence_baseline_from_payload(raw, axis_name, axis_values):
    """Validate one required, versioned effective-absence contract."""
    if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
        raise ValueError("compiled axis %s has no typed absence baseline" % axis_name)
    kind = raw["kind"]
    if kind == "equivalent":
        if set(raw) != {"kind", "token"} or not isinstance(raw.get("token"), str):
            raise ValueError("compiled axis %s has malformed equivalent baseline" % axis_name)
        if raw["token"] not in axis_values:
            raise ValueError(
                "compiled axis %s has an absence token outside its domain" % axis_name
            )
        return {"kind": kind, "token": raw["token"]}
    if kind not in ("declaration_default", "distinct", "contextual"):
        raise ValueError("compiled axis %s has unknown absence baseline" % axis_name)
    if set(raw) != {"kind"}:
        raise ValueError("compiled axis %s has malformed absence baseline" % axis_name)
    return {"kind": kind}


def _space_from_binary_payload(payload, ny_bin):
    """Validate the complete compiled contract before constructing a `Space`.

    Defaults are intentionally absent here. A missing field is not an older
    spelling of the current schema; it is a different schema, and accepting it
    would eventually refill the hole from checkout source.
    """
    if not isinstance(payload, dict) or payload.get("schema") != SPACE_SCHEMA:
        raise ValueError("wrong or missing compiled search-space schema")
    for field in ("axes", "edges", "unsafe", "instrument_only", "test_only",
                  "declarations"):
        if field not in payload:
            raise ValueError("compiled search-space schema has no %r field" % field)

    raw_axes = payload["axes"]
    if not isinstance(raw_axes, list) or not raw_axes:
        raise ValueError("compiled search-space schema has no axes")
    parsed, names = [], set()
    for item in raw_axes:
        if not isinstance(item, dict):
            raise ValueError("compiled search-space axis is not an object")
        required = (
            "name", "domain", "values", "class", "deliver", "why", "absence_baseline",
            "category_prefixes", "preset_baseline_keys",
        )
        if any(field not in item for field in required):
            raise ValueError("compiled search-space axis is missing a required field")
        name, domain, values = item["name"], item["domain"], item["values"]
        cls, deliver, why = item["class"], item["deliver"], item["why"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("compiled search-space axis has an invalid or duplicate name")
        if domain not in ("bool", "enum", "grid"):
            raise ValueError("compiled search-space axis %s has invalid domain" % name)
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) for value in values)):
            raise ValueError("compiled search-space axis %s has invalid values" % name)
        if domain == "bool" and values != ["0", "1"]:
            raise ValueError("compiled Bool axis %s does not emit exact 0/1 tokens" % name)
        if cls not in ("Unsafe", "VerdictAffecting", "SafeToSearch"):
            raise ValueError("compiled search-space axis %s has invalid class" % name)
        if (not isinstance(deliver, str)
                or (deliver != "env-only"
                    and not (deliver.startswith("preset:") and deliver != "preset:"))):
            raise ValueError("compiled search-space axis %s has invalid delivery" % name)
        if not isinstance(why, str):
            raise ValueError("compiled search-space axis %s has non-string rationale" % name)
        absence_baseline = _absence_baseline_from_payload(
            item["absence_baseline"], name, values
        )
        category_prefixes = item["category_prefixes"]
        if (not isinstance(category_prefixes, list)
                or any(not isinstance(prefix, str) or not prefix
                       or prefix != prefix.lower() for prefix in category_prefixes)
                or len(category_prefixes) != len(set(category_prefixes))):
            raise ValueError("compiled axis %s has invalid category prefixes" % name)
        preset_baseline_keys = item["preset_baseline_keys"]
        direct_key = deliver[len("preset:"):] if deliver.startswith("preset:") else None
        if (
            not isinstance(preset_baseline_keys, list)
            or any(not isinstance(key, str) or not key
                   for key in preset_baseline_keys)
            or len(preset_baseline_keys) != len(set(preset_baseline_keys))
            or (direct_key is None and preset_baseline_keys)
            or (direct_key is not None and (
                not preset_baseline_keys or preset_baseline_keys[0] != direct_key
            ))
        ):
            raise ValueError("compiled axis %s has invalid preset baseline keys" % name)
        names.add(name)
        parsed.append(
            Axis(
                name, domain, values, cls, deliver, why, absence_baseline,
                category_prefixes, preset_baseline_keys,
            )
        )

    if not isinstance(payload["declarations"], dict) or not payload["declarations"]:
        raise ValueError("compiled search-space declarations are missing or empty")
    declarations = _declarations_from_payload(payload["declarations"])
    missing = names.difference(declarations)
    if missing:
        raise ValueError("compiled search-space declarations omit %s" % ", ".join(sorted(missing)))

    raw_edges = payload["edges"]
    if not isinstance(raw_edges, list):
        raise ValueError("compiled search-space edges are not a list")
    axes_by_name = {axis.name: axis for axis in parsed}
    edges = []
    for item in raw_edges:
        required = ("child", "kind", "parent", "parents", "bound", "token", "site")
        if not isinstance(item, dict) or any(field not in item for field in required):
            raise ValueError("compiled search-space edge is malformed")
        child, kind, parent = item["child"], item["kind"], item["parent"]
        parents = item["parents"]
        bound, token, site = item["bound"], item["token"], item["site"]
        if child not in names:
            raise ValueError("compiled search-space edge names an unknown child axis")
        if kind not in (
                "armed", "any_armed", "non_zero", "greater_than", "not_armed",
                "not_equals", "absent", "equals"):
            raise ValueError("compiled search-space edge has invalid requirement")
        if kind == "any_armed":
            if parent is not None:
                raise ValueError("compiled any_armed edge has a scalar parent")
            if (not isinstance(parents, list) or not parents
                    or any(not isinstance(candidate, str) or candidate not in names
                           for candidate in parents)
                    or len(parents) != len(set(parents))):
                raise ValueError("compiled any_armed edge has invalid parents")
        else:
            if not isinstance(parent, str) or parent not in names:
                raise ValueError("compiled search-space edge names an unknown parent axis")
            if parents is not None:
                raise ValueError("compiled scalar edge has plural parents")
        if (kind == "greater_than"
                and (not isinstance(bound, int) or isinstance(bound, bool) or bound < 0)):
            raise ValueError("compiled greater-than edge has invalid bound")
        if kind != "greater_than" and bound is not None:
            raise ValueError("compiled search-space edge has an unexpected bound")
        if kind in ("equals", "not_equals"):
            if not isinstance(token, str):
                raise ValueError("compiled equality edge has invalid token")
            if kind == "equals" and token not in axes_by_name[parent].values:
                raise ValueError("compiled equality edge token is outside its parent domain")
            if kind == "not_equals":
                if not value_admissible(declarations[parent], token):
                    raise ValueError(
                        "compiled not-equals sentinel is rejected by its parent declaration"
                    )
                if not any(
                    not tokens_semantically_equal(declarations[parent], value, token)
                    for value in axes_by_name[parent].values
                ):
                    raise ValueError(
                        "compiled not-equals edge has no searchable parent repair"
                    )
        elif token is not None:
            raise ValueError("compiled search-space edge has an unexpected token")
        if not isinstance(site, str):
            raise ValueError("compiled search-space edge has non-string site")
        edges.append(Edge(
            child,
            kind,
            parent=parent,
            parents=parents,
            bound=bound,
            token=token,
            site=site,
        ))

    for field in ("unsafe", "instrument_only", "test_only"):
        values = payload[field]
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError("compiled search-space %s is not a string list" % field)
    for axis in parsed:
        declaration = declarations[axis.name]
        absence = axis.absence_baseline
        if absence["kind"] == "declaration_default" and declaration.get("default") is None:
            raise ValueError(
                "compiled axis %s has an unset declaration default but no absence contract"
                % axis.name
            )
        if (absence["kind"] == "equivalent"
                and not value_admissible(declaration, absence["token"])):
            raise ValueError(
                "compiled axis %s has an absence token rejected by its declaration"
                % axis.name
            )

    return Space(
        parsed,
        edges,
        payload["unsafe"],
        payload["instrument_only"],
        payload["test_only"],
        "binary `%s %s`" % (Path(ny_bin).name, " ".join(SPACE_PROBE)),
        declarations=declarations,
    )


def probe_space_binary(ny_bin, timeout=20, keep_ambient=False):
    """Ask the binary to print its own space. Returns (Space, argv) or None.

    The binary is authoritative when it can answer; the source parse cannot see
    a `#[cfg]`-gated axis or a binary built from a different revision. Only the
    current measured-binary contract is accepted.
    """
    if not ny_bin or not Path(ny_bin).exists():
        return None
    try:
        probe_env, _scrubbed = child_env(keep_ambient)
        done = subprocess.run(
            [str(ny_bin), *SPACE_PROBE], capture_output=True, text=True,
            timeout=timeout, env=probe_env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    try:
        payload = json.loads(done.stdout)
        space = _space_from_binary_payload(payload, ny_bin)
    except (KeyError, TypeError, ValueError):
        return None
    return space, list(SPACE_PROBE)


def load_space(ny_bin, allow_source_schema=False, keep_ambient=False):
    """Load the selected executable's schema, failing closed by default.

    `allow_source_schema` exists only for explicit developer inspection. Callers
    that generate or run arms must leave it false: the source checkout is not
    evidence about an existing binary, even when that binary lives under this
    checkout's target directory.
    """
    probed = probe_space_binary(ny_bin, keep_ambient=keep_ambient)
    if probed:
        return probed[0]
    if allow_source_schema:
        return parse_space_rs()
    target = str(ny_bin) if ny_bin else "<unset>"
    raise SpaceSchemaUnavailable(
        "selected ny binary %s does not provide a valid `%s`; refusing to pair "
        "that executable with the checkout's source schema"
        % (target, " ".join(SPACE_PROBE))
    )


# --------------------------------------------------------------------------
# The interaction lattice, mirrored from `space::expand`
# --------------------------------------------------------------------------


def _armed(sample, name):
    return sample.get(name) == "1"


def _non_zero(sample, name):
    """Mirror of `space::non_zero` — NUMERIC, not textual.

    A fraction disarms with "0.0", which a string compare against "0" calls
    ARMED. That stopped the lattice suppressing children of a disarmed parent,
    so those samples ran as full-budget measurements of the baseline labelled as
    treatments. Non-numeric tokens keep the textual rule.
    """
    if name not in sample:
        return False
    raw = sample[name].strip()
    try:
        return float(raw) != 0.0
    except ValueError:
        return raw != "0"


def _greater_than(sample, name, bound):
    try:
        return int(sample[name]) > bound
    except (KeyError, TypeError, ValueError):
        return False


def _dependency_tokens_equal(kind, left, right):
    if left is None or right is None:
        return False
    return left == right or (
        kind is not None and tokens_semantically_equal(kind, left, right)
    )


def expand(sample, space):
    """Mirror of `space::expand`: the sorted assignment, or raise `Inert`.

    Every labelled value, including an explicit control value of "0", must be
    in the exact search domain and satisfy its dependency edges. Otherwise a
    child control spends a row re-measuring the baseline without its parent.

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
        axis = space.by_name.get(name)
        if axis is None:
            raise Inert(name, "name is not a searchable axis")
        if value not in axis.values:
            raise Inert(name, "value %r is outside the exact search domain %r"
                        % (value, axis.values))
        for edge in space.edges_by_child.get(name, []):
            if edge.kind == "armed":
                ok = _armed(sample, edge.parent)
            elif edge.kind == "any_armed":
                ok = any(_armed(sample, parent) for parent in edge.parents)
            elif edge.kind == "non_zero":
                ok = _non_zero(sample, edge.parent)
            elif edge.kind == "greater_than":
                ok = _greater_than(sample, edge.parent, edge.bound)
            elif edge.kind == "equals":
                ok = sample.get(edge.parent) == edge.token
            elif edge.kind == "not_equals":
                kind = (space.declarations or {}).get(edge.parent)
                ok = not _dependency_tokens_equal(
                    kind, sample.get(edge.parent), edge.token
                )
            elif edge.kind == "absent":
                ok = edge.parent not in sample
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
    if edge.kind == "armed":
        return "1" if "1" in eligible else None
    if edge.kind == "non_zero":
        for token in eligible:
            if _non_zero({axis.name: token}, axis.name):
                return token
        return None
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
    if edge.kind == "equals":
        return edge.token if edge.token in eligible else None
    if edge.kind == "not_equals":
        return next(
            (
                token for token in eligible
                if not _dependency_tokens_equal(kind, token, edge.token)
            ),
            None,
        )
    return None


def _arming_value(axis, kinds=None):
    """Exact parser-valid `"1"` for one Bool-like arming parent, or None."""
    if axis is None or "1" not in axis.values:
        return None
    kind = (kinds or {}).get(axis.name)
    if kinds is not None and kind is not None and not value_admissible(kind, "1"):
        return None
    return "1"


def _normalize_repair_baselines(baseline_values, repair_baselines):
    """Per-category facts for runtime dependency repair.

    `baseline_values` is the historical single-map API. Treating every key in
    that map as present preserves its exact behavior for callers and tests that
    do not have category-preset facts. New configured searches pass one repair
    baseline per category so disagreements are never collapsed away.
    """
    if repair_baselines is not None:
        return tuple(repair_baselines)
    values = dict(baseline_values or {})
    return ({
        "values": values,
        "present": frozenset(values),
        "unknown": frozenset(),
    },)


def _baseline_known_armed(repair_baselines, name):
    return any(_armed(profile["values"], name) for profile in repair_baselines)


def _baseline_may_be_armed(repair_baselines, name):
    return any(
        name in profile["unknown"] or _armed(profile["values"], name)
        for profile in repair_baselines
    )


def _baseline_may_equal(repair_baselines, name, token, kind):
    return any(
        name in profile["unknown"]
        or _dependency_tokens_equal(kind, profile["values"].get(name), token)
        for profile in repair_baselines
    )


def _baseline_may_be_present(repair_baselines, name):
    return any(
        name in profile["present"] or name in profile["unknown"]
        for profile in repair_baselines
    )


def _any_armed_repair(edge, sample, space, kinds, repair_baselines, added):
    """Choose one non-confounding OR satisfier, preserving declared order.

    A parent explicitly present as `0` is a treatment coordinate and must not be
    overwritten. An already-armed runtime baseline is preferred before the
    declared fallback order, so closure-adaptive experiments do not acquire a
    new root-time direct-beta treatment merely to satisfy arm-only validation.
    """
    for parent in edge.parents:
        if parent in sample or parent in added:
            continue
        if not _baseline_known_armed(repair_baselines, parent):
            continue
        value = _arming_value(space.by_name.get(parent), kinds)
        if value is not None:
            return parent, value
    for parent in edge.parents:
        if parent in sample or parent in added:
            continue
        value = _arming_value(space.by_name.get(parent), kinds)
        if value is not None:
            return parent, value
    return None


def _disarming_value(axis, kinds=None):
    if axis is None or "0" not in axis.values:
        return None
    kind = (kinds or {}).get(axis.name)
    if kinds is not None and kind is not None and not value_admissible(kind, "0"):
        return None
    return "0"


def repair(sample, space, kinds=None, baseline_values=None, repair_baselines=None):
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

    Runtime baseline facts additionally close ownership, exclusion, and absence
    edges against every loaded category preset. If any category would suppress
    a child, repair emits a common explicit control where one exists; a required
    absence cannot be reconstructed by an override, so that child is dropped.
    `baseline_values` remains the compatible single-baseline input, while
    `repair_baselines` preserves category disagreements and preset presence.
    """
    if kinds is None and space.declarations is not None:
        kinds = space.declarations
    repair_baselines = _normalize_repair_baselines(
        baseline_values, repair_baselines)
    sample = dict(sample)
    added = set()
    for _ in range(len(sample) * len(space.edges) + 16):
        changed = False
        for name in sorted(sample):
            if name not in sample:
                continue
            for edge in space.edges_by_child.get(name, []):
                if edge.kind == "any_armed":
                    if any(_armed(sample, parent) for parent in edge.parents):
                        continue
                    fix = _any_armed_repair(
                        edge, sample, space, kinds, repair_baselines, added)
                    if fix is None:
                        del sample[name]
                        added.discard(name)
                        changed = True
                        break
                    parent, value = fix
                    sample[parent] = value
                    added.add(parent)
                    changed = True
                    continue
                parent_axis = space.by_name.get(edge.parent)
                if edge.kind == "not_armed":
                    if _armed(sample, edge.parent):
                        del sample[name]
                        added.discard(name)
                        changed = True
                        break
                    if (edge.parent not in sample
                            and _baseline_may_be_armed(
                                repair_baselines, edge.parent)):
                        fix = _disarming_value(parent_axis, kinds)
                        if fix is None or edge.parent in added:
                            del sample[name]
                            added.discard(name)
                            changed = True
                            break
                        # The typed preset/default would otherwise suppress the
                        # child at runtime even though the arm-only Rust lattice
                        # accepts it. Override that owner explicitly.
                        sample[edge.parent] = fix
                        added.add(edge.parent)
                        changed = True
                    continue
                if edge.kind == "absent":
                    if (edge.parent in sample
                            or _baseline_may_be_present(
                                repair_baselines, edge.parent)):
                        del sample[name]
                        added.discard(name)
                        changed = True
                        break
                    continue
                if edge.kind == "not_equals":
                    parent_kind = (kinds or {}).get(edge.parent)
                    if _dependency_tokens_equal(
                        parent_kind, sample.get(edge.parent), edge.token
                    ):
                        del sample[name]
                        added.discard(name)
                        changed = True
                        break
                    if (
                        edge.parent not in sample
                        and _baseline_may_equal(
                            repair_baselines,
                            edge.parent,
                            edge.token,
                            parent_kind,
                        )
                    ):
                        fix = _satisfying_value(parent_axis, edge, kinds)
                        if fix is None or edge.parent in added:
                            del sample[name]
                            added.discard(name)
                            changed = True
                            break
                        sample[edge.parent] = fix
                        added.add(edge.parent)
                        changed = True
                    continue
                if edge.kind == "armed" and _armed(sample, edge.parent):
                    continue
                if edge.kind == "non_zero" and _non_zero(sample, edge.parent):
                    continue
                if edge.kind == "greater_than" and _greater_than(sample, edge.parent, edge.bound):
                    continue
                if edge.kind == "equals" and sample.get(edge.parent) == edge.token:
                    continue
                if (
                    edge.kind == "not_equals"
                    and not _dependency_tokens_equal(
                        (kinds or {}).get(edge.parent),
                        sample.get(edge.parent),
                        edge.token,
                    )
                ):
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
            declaration_syntax = re.sub(r"//[^\n]*", "", block)
            declaration_syntax = re.sub(
                r"/\*.*?\*/", "", declaration_syntax, flags=re.S
            )
            if record["kind"] in (
                "F64Open",
                "F64OpenOverride",
                "F64OpenUnicodeOwned",
                "F64ClosedTrimmed",
            ):
                bounds = re.search(
                    r"F64(?:Open(?:Override|UnicodeOwned)?|ClosedTrimmed)"
                    r"\s*\{\s*min:\s*([^,\n]+)"
                    r"\s*,\s*max:\s*([^,}\n]+)",
                    declaration_syntax,
                )
                if bounds:
                    try:
                        record["min"] = _rust_f64_expression(bounds.group(1))
                        record["max"] = _rust_f64_expression(bounds.group(2))
                    except (OverflowError, ValueError):
                        # Missing bounds make bounded kinds fail closed in
                        # `value_admissible`; never guess a source-only parser.
                        pass
            if record["kind"] in (
                "Usize", "UsizeTrimmedOverride", "UsizeOverride",
            ):
                lower_bound = re.search(
                    r"LeverKind::(?:UsizeTrimmedOverride|UsizeOverride|Usize)\s*"
                    r"\{\s*min:\s*([^,}\n]+)",
                    declaration_syntax,
                )
                if lower_bound:
                    try:
                        record["min"] = _rust_usize_expression(lower_bound.group(1))
                    except (OverflowError, ValueError):
                        # As with floating bounds, an unreadable source parser
                        # contract is excluded later rather than approximated.
                        pass
            if record["kind"] in ("Enum", "LayeredEnum"):
                if record["kind"] == "LayeredEnum":
                    members = re.search(r"env_members:\s*&\[([^\]]*)\]", block, re.S)
                else:
                    members = re.search(r"LeverKind::Enum\(&\[([^\]]*)\]", block, re.S)
                record["members"] = _rust_str_list(members.group(1)) if members else []
            default = re.search(r"default:\s*DefaultSpec::(\w+)(?:\(([^)]*)\))?", block)
            if default:
                record["default"] = _default_token(default.group(1), default.group(2))
            kinds[name] = record
    return kinds


def _rust_f64_expression(expression):
    """Evaluate the tiny constant grammar used by float lever declarations."""
    source = expression.strip()
    if source == "f64::MAX":
        return sys.float_info.max
    from_bits = re.fullmatch(
        r"f64::from_bits\((0x[0-9a-fA-F_]+|[0-9_]+)\)", source
    )
    if from_bits:
        bits = int(from_bits.group(1).replace("_", ""), 0)
        if not 0 <= bits <= (1 << 64) - 1:
            raise ValueError("f64 bit pattern outside u64")
        return struct.unpack(">d", bits.to_bytes(8, "big"))[0]
    return float(source.replace("_", ""))


def _rust_usize_expression(expression):
    """Evaluate the literal grammar used by bounded `usize` lever kinds."""
    source = expression.strip().replace("_", "")
    maximum = (1 << (8 * struct.calcsize("P"))) - 1
    if source == "usize::MAX":
        return maximum
    if re.fullmatch(r"[0-9]+", source) is None:
        raise ValueError("unsupported usize declaration expression")
    value = int(source, 10)
    if value > maximum:
        raise OverflowError("usize declaration expression is out of range")
    return value


def lever_kinds_for_space(space):
    """Parser contracts from the measured binary, or the source fallback.

    Do not merge the two maps. A measured binary reaches this point only after
    its current declaration map was proved non-empty and complete; filling any hole
    from the checkout could approve a token the executable will reject.
    """
    if space.declarations is not None:
        return space.declarations
    return parse_lever_kinds()


def _default_token(variant, payload):
    """The raw token an environment value must DIFFER from to be a treatment."""
    if variant == "Unset":
        return None
    if variant == "Bool":
        return "1" if (payload or "").strip() == "true" else "0"
    if variant == "Text":
        strings = _rust_str_list(payload or "")
        return strings[0] if strings else None
    if payload is None:
        return None
    return (payload or "").strip().replace("_", "")


def _parsed_numeric_lever_value(kind, token):
    """Mirror Rust numeric `FromStr`, including trim and integer width."""
    if kind is None or not isinstance(token, str):
        return None
    name = kind.get("kind")
    if name in (
        "U64", "U64Override", "U64Trimmed", "UsizeTrimmed",
        "UsizeTrimmedUnicodeOwned", "UsizeTrimmedOverride",
        "UsizeOverride", "Usize"
    ):
        source = (
            token
            if name in ("U64", "U64Override", "UsizeOverride", "Usize")
            else token.strip(_RUST_WHITESPACE)
        )
        # Rust unsigned `FromStr` accepts one leading `+`, but neither Unicode
        # digits nor a minus sign (including `-0`). Python's `int` is broader.
        if re.fullmatch(r"\+?[0-9]+", source) is None:
            return None
        value = int(source, 10)
        maximum = (
            (1 << (8 * struct.calcsize("P"))) - 1
            if name in (
                "UsizeTrimmed", "UsizeTrimmedUnicodeOwned",
                "UsizeTrimmedOverride", "UsizeOverride", "Usize",
            )
            else (1 << 64) - 1
        )
        if value > maximum:
            return None
        if name in ("Usize", "UsizeTrimmedOverride", "UsizeOverride"):
            minimum = kind.get("min")
            if not isinstance(minimum, int) or isinstance(minimum, bool):
                return None
            return value if value >= minimum else None
        return value
    if name in ("F32Positive", "F32NonNegative"):
        value = _parsed_ascii_f32(token)
        if value is None:
            return None
        if name == "F32Positive":
            return value if value > 0.0 else None
        return value if value >= 0.0 else None
    if name not in (
        "F64Open", "F64OpenOverride", "F64OpenUnicodeOwned",
        "F64ClosedTrimmed", "Secs",
    ):
        return None
    source = (
        token.strip(_RUST_WHITESPACE)
        if name == "F64ClosedTrimmed"
        else token
    )
    # `str::parse::<f64>` does not trim and accepts only ASCII numeric syntax.
    # Python's `float` silently accepts surrounding whitespace and Unicode
    # decimal digits, either of which would turn a rejected Rust arm into a
    # purported treatment here.
    if re.fullmatch(
        r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))"
        r"(?:[eE][+-]?[0-9]+)?",
        source,
    ) is None:
        return None
    try:
        value = float(source)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if name in ("F64Open", "F64OpenOverride", "F64OpenUnicodeOwned"):
        if "min" not in kind or "max" not in kind:
            return None
        return value if kind["min"] < value < kind["max"] else None
    if name == "F64ClosedTrimmed":
        if "min" not in kind or "max" not in kind:
            return None
        return value if kind["min"] <= value <= kind["max"] else None
    return value if value >= 0.0 else None


_ASCII_DECIMAL_RE = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))"
    r"(?:[eE][+-]?[0-9]+)?"
)

# Rust `str::trim()` follows `char::is_whitespace`, whose Unicode White_Space
# set is slightly narrower than Python's `str.strip()`: Python additionally
# strips the C0 information separators U+001C..U+001F. Keep every trimmed
# parser byte-for-byte aligned with the Rust runtime.
_RUST_WHITESPACE = (
    "\u0009\u000a\u000b\u000c\u000d\u0020\u0085\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)


def _parsed_ascii_f32(token):
    """Rust-like untrimmed finite decimal `f32`, returned after narrowing."""
    if not isinstance(token, str) or _ASCII_DECIMAL_RE.fullmatch(token) is None:
        return None
    try:
        wide = float(token)
        if not math.isfinite(wide):
            return None
        value = struct.unpack(">f", struct.pack(">f", wide))[0]
    except (OverflowError, struct.error, ValueError):
        return None
    return value if math.isfinite(value) else None


def tokens_semantically_equal(kind, left, right):
    """Whether two accepted raw tokens resolve to the same typed value."""
    if kind is None or left is None or right is None:
        return False
    name = kind.get("kind")
    if name == "Presence":
        # Every present byte string resolves to `LeverValue::Bool(true)`.
        return True
    if name == "TruthyTrimmed":
        # Preserve the reviewed legacy root-f32 grammar exactly. Every UTF-8
        # token is accepted; trimming to one of these three spellings selects
        # the true Boolean value and every other spelling selects false.
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        truthy = {"1", "true", "on"}
        return (left.strip(_RUST_WHITESPACE) in truthy) == (
            right.strip(_RUST_WHITESPACE) in truthy
        )
    if name == "TrimmedExactOne":
        # Every Unicode token is accepted by this legacy parser. Exactly one
        # trimmed spelling selects true; every other spelling is the same
        # explicit false override. Measurement launch syntax remains narrower
        # (canonical 0/1) in ny_measurement_provenance.py.
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        return (left.strip(_RUST_WHITESPACE) == "1") == (
            right.strip(_RUST_WHITESPACE) == "1"
        )
    if name in (
        "U64", "U64Override", "U64Trimmed", "UsizeTrimmed",
        "UsizeTrimmedUnicodeOwned", "UsizeTrimmedOverride",
        "UsizeOverride", "Usize", "F32Positive", "F32NonNegative",
        "F64Open", "F64OpenOverride", "F64OpenUnicodeOwned",
        "F64ClosedTrimmed", "Secs",
    ):
        lhs = _parsed_numeric_lever_value(kind, left)
        rhs = _parsed_numeric_lever_value(kind, right)
        return lhs is not None and rhs is not None and lhs == rhs
    # Boolean kinds (including one-way kill overrides), Enum, LayeredEnum and
    # Text retain exact-byte semantics.
    return str(left) == str(right)


def value_admissible(kind, token):
    """Does `token` survive `ny_levers::env::parse` for this kind?

    Mirrors the chokepoint exactly. A value that does not survive resolves to
    the declaration default and lands in `rejected_raw`: the run measures the
    baseline and the harness would report it as a treatment.
    """
    if kind is None:
        return False
    name = kind["kind"]
    if name in ("Bool", "BoolOverride"):
        return token in ("0", "1")
    if name == "BoolKillOverride":
        return token == "0"
    if name in ("TruthyTrimmed", "TrimmedExactOne"):
        return isinstance(token, str)
    if name == "Presence":
        return True  # cannot reject: presence alone arms it
    if name in (
        "U64", "U64Override", "U64Trimmed", "UsizeTrimmed",
        "UsizeTrimmedUnicodeOwned", "UsizeTrimmedOverride",
        "UsizeOverride", "Usize", "F32Positive", "F32NonNegative",
        "F64Open", "F64OpenOverride", "F64OpenUnicodeOwned",
    ):
        return _parsed_numeric_lever_value(kind, token) is not None
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
        return _parsed_numeric_lever_value(kind, token) is not None
    if name == "Secs":
        return _parsed_numeric_lever_value(kind, token) is not None
    if name in ("Enum", "LayeredEnum"):
        return token in kind.get("members", [])
    if name == "Text":
        return True
    return False


def benchmark_configs_dir(configs_dir, year):
    """Exact `vnncompYY` directory for an explicitly requested corpus year.

    Accepts either the parent `configs/` directory or an already scoped
    `configs/vnncompYY`. A differently scoped directory and a missing requested
    year both return None: a 2025 measurement must never fall forward into a
    same-named 2026 preset merely because it is lexically newer.
    """
    root = Path(configs_dir)
    expected = "vnncomp%02d" % (int(year) % 100)
    if re.fullmatch(r"vnncomp\d{2}", root.name):
        if root.name != expected:
            return None
        selected = root
    else:
        selected = root / expected
    return selected.resolve() if selected.is_dir() else None


def effective_configs_root(explicit, repo_root=REPO_ROOT):
    """Mirror the repo-local auto-config case the sweep child will actually use."""
    if explicit:
        return Path(explicit)
    automatic = Path(repo_root) / "configs"
    return automatic if automatic.is_dir() else None


def resolve_preset_path(configs_dir, category, year):
    """Resolve a category only inside the requested corpus year's preset directory."""
    year_dir = benchmark_configs_dir(configs_dir, year)
    if year_dir is None:
        return None
    lower = category.lower()
    candidates = [lower]
    if len(lower) >= 5 and re.match(r"^_20\d\d$", lower[-5:]):
        candidates.append(lower[:-5])
    for candidate in candidates:
        preset = year_dir / ("%s.yaml" % candidate)
        if preset.is_file():
            return preset.resolve()
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


_PORTABLE_YAML_KEY_RE = re.compile(r"[A-Za-z0-9_-]+")
_PORTABLE_YAML_INTEGER_RE = re.compile(r"[+-]?(?:0|[1-9][0-9]*)")
_PORTABLE_YAML_FLOAT_RE = re.compile(
    r"[+-]?(?:(?:[0-9]+\.[0-9]*)|(?:\.[0-9]+))(?:[eE][+-]?[0-9]+)?"
)
_PORTABLE_YAML_BARE_STRING_RE = re.compile(r"[A-Za-z0-9_./+-]+")


def _portable_yaml_content(line):
    """Strip one YAML comment while respecting the supported quoted scalars."""
    quote = None
    escaped = False
    index = 0
    while index < len(line):
        character = line[index]
        if quote == '"':
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif quote == "'":
            if character == quote:
                if index + 1 < len(line) and line[index + 1] == quote:
                    index += 1
                else:
                    quote = None
        elif character in {'"', "'"}:
            quote = character
        elif character == "#" and (
            index == 0 or line[index - 1].isspace()
        ):
            return line[:index].rstrip()
        index += 1
    if quote is not None or escaped:
        raise ValueError("unterminated quoted YAML scalar")
    return line.rstrip()


def _portable_yaml_scalar(source):
    """Parse the closed scalar subset used by the shipped VNN-COMP presets."""
    if not source:
        raise ValueError("empty YAML scalar")
    if source.startswith('"'):
        try:
            value = json.loads(source)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid double-quoted YAML scalar") from error
        if not isinstance(value, str):
            raise ValueError("double-quoted YAML scalar is not a string")
        return value
    if source.startswith("'"):
        if len(source) < 2 or not source.endswith("'"):
            raise ValueError("invalid single-quoted YAML scalar")
        body = source[1:-1]
        index = 0
        decoded = []
        while index < len(body):
            character = body[index]
            if character == "'":
                if index + 1 >= len(body) or body[index + 1] != "'":
                    raise ValueError("invalid single-quoted YAML escape")
                decoded.append("'")
                index += 2
                continue
            decoded.append(character)
            index += 1
        return "".join(decoded)
    lower = source.lower()
    if lower in {"null", "~"}:
        return None
    if lower in {"true", "false"}:
        return lower == "true"
    if _PORTABLE_YAML_INTEGER_RE.fullmatch(source):
        return int(source, 10)
    if _PORTABLE_YAML_FLOAT_RE.fullmatch(source):
        value = float(source)
        if not math.isfinite(value):
            raise ValueError("non-finite YAML scalar")
        return value
    if _PORTABLE_YAML_BARE_STRING_RE.fullmatch(source):
        return source
    raise ValueError("unsupported YAML scalar")


def _parse_portable_preset_yaml(text):
    """Parse the dependency-free, mapping-and-scalar YAML preset grammar.

    This is deliberately not a general YAML implementation. The scorecard
    invokes Python with ``-S``, while the solver's reviewed VNN-COMP presets use
    only indentation maps and one scalar-list spelling. Rejecting every YAML
    feature outside that closed grammar keeps provenance identical with and
    without user-site PyYAML instead of guessing at anchors, tags, merges, or
    implicit scalar types.
    """
    if not isinstance(text, str):
        raise ValueError("preset YAML is not text")
    logical = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        if "\t" in raw_line:
            raise ValueError(f"line {line_number}: tabs are unsupported")
        content = _portable_yaml_content(raw_line)
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        if indent % 2:
            raise ValueError(f"line {line_number}: indentation is not two-space aligned")
        stripped = content[indent:]
        if stripped in {"---", "..."}:
            raise ValueError(f"line {line_number}: YAML document markers are unsupported")
        logical.append((line_number, indent, stripped))
    if not logical:
        raise ValueError("preset YAML is empty")

    def parse_block(index, indent):
        first_line, first_indent, first = logical[index]
        if first_indent != indent:
            raise ValueError(f"line {first_line}: unexpected indentation")
        sequence = first.startswith("- ")
        result = [] if sequence else {}
        while index < len(logical):
            line_number, current_indent, source = logical[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"line {line_number}: unexpected indentation")
            if source.startswith("- ") != sequence:
                raise ValueError(f"line {line_number}: mixed mapping and sequence")
            if sequence:
                item = source[2:].strip()
                if not item:
                    raise ValueError(f"line {line_number}: nested list items are unsupported")
                result.append(_portable_yaml_scalar(item))
                index += 1
                continue
            key, separator, value = source.partition(":")
            if (
                not separator
                or _PORTABLE_YAML_KEY_RE.fullmatch(key) is None
                or key in result
            ):
                raise ValueError(f"line {line_number}: invalid or duplicate mapping key")
            value = value.strip()
            if value:
                result[key] = _portable_yaml_scalar(value)
                index += 1
                continue
            index += 1
            if index >= len(logical) or logical[index][1] <= indent:
                raise ValueError(f"line {line_number}: empty mapping value")
            if logical[index][1] != indent + 2:
                raise ValueError(
                    f"line {logical[index][0]}: nested indentation skips a level"
                )
            child, index = parse_block(index, indent + 2)
            result[key] = child
        return result, index

    if logical[0][1] != 0 or logical[0][2].startswith("- "):
        raise ValueError("preset YAML root must be a zero-indented mapping")
    document, consumed = parse_block(0, 0)
    if consumed != len(logical) or not isinstance(document, dict):
        raise ValueError("preset YAML did not form one mapping document")
    return document


def load_preset_doc_portable(preset_path):
    """Read a preset with only stdlib modules, returning None on any ambiguity."""
    if preset_path is None:
        return None
    try:
        with open(preset_path, "r", encoding="utf-8") as handle:
            document = _parse_portable_preset_yaml(handle.read())
    except (OSError, UnicodeError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _preset_path_token(preset_doc, dotted_key):
    """Return `(state, token)` for one typed dotted key.

    `absent` is distinct from `invalid`: a malformed value at an earlier key in
    a runtime fallback chain must not silently reveal a later fallback and
    invent a baseline the engine would never load.
    """
    if not isinstance(preset_doc, dict):
        return "absent", None
    node = preset_doc
    for part in dotted_key.split("."):
        if not isinstance(node, dict):
            return "invalid", None
        if part not in node:
            return "absent", None
        node = node[part]
    # Serde `Option<T>` fields treat an explicit YAML null exactly like an
    # omitted key. Runtime therefore continues through a layered fallback (or
    # reaches the axis absence contract) instead of rejecting the preset.
    if node is None:
        return "absent", None
    if isinstance(node, bool):
        return "token", "1" if node else "0"
    if isinstance(node, int):
        return "token", str(node)
    if isinstance(node, float):
        return "token", repr(node)
    if isinstance(node, str):
        return "token", node
    return "invalid", None


def preset_effective_token(preset_doc, deliver):
    """Token one direct `preset:<dotted.key>` delivery, if present and scalar."""
    if not deliver.startswith("preset:"):
        return None
    state, token = _preset_path_token(preset_doc, deliver[len("preset:"):])
    return token if state == "token" else None


def preset_effective_axis_token(preset_doc, axis):
    """Resolve the first present typed key in an axis's runtime fallback chain."""
    for key in axis.preset_baseline_keys:
        state, token = _preset_path_token(preset_doc, key)
        if state == "absent":
            continue
        return state, token, key
    return "absent", None, None


def _effective_axis_baseline(axis, kind, preset_doc=None, preset_known=False,
                             preset_expected=False, preset_docs=None):
    """Return `(state, token, source)` for T0 and lattice repair.

    `state == "known"` is the only state with a comparable token. `distinct`
    means absence itself is a real fourth state, `contextual` means the live
    baseline cannot be proven before executing the exact category/preset/lane
    context, and `unknown` means a requested preset could not be parsed or
    category baselines genuinely differ for this axis.
    """
    if preset_docs:
        category_baselines = [
            _effective_axis_baseline(
                axis,
                kind,
                preset_doc=doc,
                preset_known=True,
                preset_expected=True,
            )
            for doc in preset_docs
        ]
        states = {state for state, _token, _source in category_baselines}
        if states == {"known"}:
            first_token = category_baselines[0][1]
            if all(tokens_semantically_equal(kind, first_token, token)
                   for _state, token, _source in category_baselines[1:]):
                return (
                    "known",
                    first_token,
                    "the common effective value across %d loaded category presets"
                    % len(category_baselines),
                )
        elif len(states) == 1 and states <= {"distinct", "contextual"}:
            state = category_baselines[0][0]
            return (
                state,
                None,
                "the common %s absence policy across %d loaded category presets"
                % (state, len(category_baselines)),
            )
        return (
            "unknown",
            None,
            "category-specific effective values across %d loaded presets"
            % len(category_baselines),
        )

    is_preset_axis = axis.deliver.startswith("preset:")
    preset_state, preset_token, preset_key = preset_effective_axis_token(
        preset_doc, axis
    )
    if is_preset_axis and preset_state == "token":
        inherited = preset_key != axis.deliver[len("preset:"):]
        return (
            "known",
            preset_token,
            "the value preset %s already delivers%s"
            % (preset_key, " through runtime fallback" if inherited else ""),
        )
    if is_preset_axis and preset_state == "invalid":
        return "unknown", None, "a malformed value at preset key %s" % preset_key
    if is_preset_axis and preset_expected and not preset_known:
        return "unknown", None, "an unresolved requested preset"

    contract = axis.absence_baseline
    contract_kind = contract["kind"]
    absent_source = (
        "the effective value with preset key %s absent"
        % axis.deliver[len("preset:"):]
        if is_preset_axis
        else "the effective value with the environment override absent"
    )
    if contract_kind == "equivalent":
        return "known", contract["token"], absent_source
    if contract_kind == "distinct":
        return "distinct", None, absent_source
    if contract_kind == "contextual":
        return "contextual", None, absent_source
    if contract_kind != "declaration_default":
        return "unknown", None, "an unrecognised absence contract"
    token = kind.get("default")
    if token is None:
        # The Rust space ratchet forbids this combination for searchable axes.
        # Keep the Python mirror fail-closed if a malformed external schema
        # somehow reaches this layer anyway.
        return "unknown", None, "a missing declaration default"
    return "known", token, "the DefaultSpec declaration default"


def effective_baseline_values(space, kinds, preset_doc=None, preset_known=False,
                              preset_expected=False, preset_docs=None):
    """Known effective tokens used only to repair preset/default conflicts.

    Explicit typed YAML values win. With an absent override, each axis's
    `absence_baseline` contract decides whether a concrete token is known.
    Multiple category presets are compared per axis; a common effective token
    remains usable even when unrelated YAML fields differ. Contextual,
    distinct, and unresolved-preset axes are omitted rather than invented.
    """
    baseline = {}
    for axis in space.axes:
        kind = kinds.get(axis.name) or {}
        state, token, _source = _effective_axis_baseline(
            axis,
            kind,
            preset_doc=preset_doc,
            preset_known=preset_known,
            preset_expected=preset_expected,
            preset_docs=preset_docs,
        )
        if state == "known" and token is not None:
            baseline[axis.name] = token
    return baseline


def effective_repair_baselines(space, kinds, preset_docs):
    """Preserve effective values and presence for every category preset.

    T0 needs the common effective value across categories, but dependency repair
    has a stricter question: whether one shared arm is live in *every* category.
    Keep each parsed document separate here. `present` follows the axis's typed
    runtime fallback chain, so an explicit false/zero still differs from an
    omitted or null key; `unknown` makes repair fail conservative.
    """
    profiles = []
    for preset_doc in preset_docs:
        if not isinstance(preset_doc, dict):
            profiles.append({
                "values": {},
                "present": frozenset(),
                "unknown": frozenset(axis.name for axis in space.axes),
            })
            continue
        values = {}
        present = set()
        unknown = set()
        for axis in space.axes:
            kind = kinds.get(axis.name) or {}
            state, token, _source = _effective_axis_baseline(
                axis,
                kind,
                preset_doc=preset_doc,
                preset_known=True,
                preset_expected=True,
            )
            if state == "known" and token is not None:
                values[axis.name] = token
            elif state == "unknown":
                unknown.add(axis.name)
            if axis.deliver.startswith("preset:"):
                preset_state, _preset_token, _preset_key = (
                    preset_effective_axis_token(preset_doc, axis)
                )
                if preset_state == "token":
                    present.add(axis.name)
                elif preset_state == "invalid":
                    unknown.add(axis.name)
        profiles.append({
            "values": values,
            "present": frozenset(present),
            "unknown": frozenset(unknown),
        })
    return tuple(profiles)


_MARGIN_ROW_EPOCH_T1_COMPARATOR_BASELINES = {
    # The v1 epoch terminal counts causal differences against these exact
    # historical lower-policy values. Until the terminal freezes an arbitrary
    # category-preset comparator, a different preset baseline cannot receive
    # T1 credit: doing so would compare the treatment with the wrong control.
    "NY_EPOCH_DEPTH": "0",
    "NY_EPOCH_ATTEMPTS": "2",
    "NY_EPOCH_TIER0": "0",
    "NY_EPOCH_UNIVERSE": "64",
    "NY_EPOCH_RETAIN": "128",
}

_MARGIN_ROW_DOMAIN_ALPHA_T1_COMPARATOR_BASELINES = {
    # The v1 domain-alpha terminal's counterfactual counters are defined
    # against the historical dark policy.  A category preset that changes any
    # member changes the control, so it must be split out until a later schema
    # freezes an arbitrary comparator beside the treatment.
    "NY_MARGIN_ROW_DOMAIN_ALPHA": "0",
    "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS": "2",
    "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA": "0.25",
}


_CONTEXTUAL_T1_ENGAGEMENT_TOKENS = {
    # A force has direct work-start/budget-gate evidence.  A contextual kill
    # does not yet carry the parsed lower admission needed to prove that the
    # absent policy would have consulted this particular row.
    "NY_UPFRONT_ATTACK": frozenset({"1"}),
    # The terminal margin-row frontier receipt freezes both the requested
    # effective width and the exact lower-layer width in the same Rayon
    # context, then counts only live tree decisions whose batches differ.
    "NY_MARGIN_ROW_PARALLEL": frozenset({"0", "2", "4", "8", "16"}),
    # Both optional exact-VJP controls publish their configured request and
    # effective execution value in ny-cli/exact-vjp-execution/v1.  T1 below
    # still rejects a structural bypass, a width clamp, or a step-cap clamp;
    # listing the domain here only permits those treatments to reach T1.
    "NY_PGD_VJP_K": frozenset({"8", "16", "32", "52", "64", "96", "128"}),
    "NY_PGD_VJP_STEPS": frozenset({"100", "250", "500", "800", "1000"}),
    # Both route selectors publish the exact effective ceiling alongside a
    # nonzero selected target set and a completed tightener dispatch.
}

# A disabled concurrent lane has no row-local counterfactual proving that a
# worker would have overlapped the verifier.  Keep the explicit off spelling
# readable and deliverable, but never spend or score an automated experiment
# on it until a later terminal can prove that counterfactual.
_ONE_SIDED_T1_ENGAGEMENT_TOKENS = {
    "NY_MARGIN_ROW_CONCURRENT": frozenset({"1"}),
}


def admissible_values(axis, kinds, allow_unchecked, preset_doc=None, preset_known=False,
                      preset_expected=False, preset_docs=None):
    """T0-filtered tokens for one axis, plus the reasons anything was dropped.

    The baseline a token must differ from is the value THIS RUN will actually
    load, which for a `Deliver::PresetKey` axis may be the preset's value, an
    explicit absent-key equivalent, a distinct absent state, or a contextual
    runtime policy. Comparing against `DefaultSpec` alone inverts the filter
    wherever a preset ships an axis armed: on cifar100 the preset carries
    `bab.root_comprehensive_gpu_interm: true` and `..._chunks: 64` against
    declared defaults of `false`/`1`, so the DefaultSpec comparison KEEPS the two
    tokens that merely re-measure the shipped baseline and DROPS `=0`/`=1`, the
    only genuine treatments — on the one benchmark that decides the score.

    When a requested preset could not be read, no token is dropped. A
    contextual baseline has no standalone token to compare: it is a live
    category/model/device policy. Such an axis remains searchable only when T1
    has an explicit, versioned execution contract for it; otherwise every token
    is excluded rather than allowing an inert override to masquerade as a
    treatment.
    """
    kind = kinds.get(axis.name)
    dropped = []
    if kind is None:
        if not allow_unchecked:
            return [], ["%s: no LeverKind found in decls (excluded; --allow-unchecked-domains "
                        "to search it anyway)" % axis.name]
        return list(axis.values), ["%s: kind unknown, values UNCHECKED" % axis.name]

    state, baseline_token, baseline_source = _effective_axis_baseline(
        axis,
        kind,
        preset_doc=preset_doc,
        preset_known=preset_known,
        preset_expected=preset_expected,
        preset_docs=preset_docs,
    )
    comparator_family = None
    comparator_baseline = _MARGIN_ROW_EPOCH_T1_COMPARATOR_BASELINES.get(axis.name)
    if comparator_baseline is not None:
        comparator_family = "epoch"
    else:
        comparator_baseline = (
            _MARGIN_ROW_DOMAIN_ALPHA_T1_COMPARATOR_BASELINES.get(axis.name)
        )
        if comparator_baseline is not None:
            comparator_family = "domain-alpha"
    if comparator_baseline is not None and (
        state != "known"
        or baseline_token is None
        or not tokens_semantically_equal(kind, baseline_token, comparator_baseline)
    ):
        observed = (
            repr(baseline_token)
            if state == "known" and baseline_token is not None
            else "%s (%s)" % (state, baseline_source)
        )
        return [], [
            "%s=%s cannot earn T1 against category baseline %s; the v1 "
            "%s terminal's sealed causal comparator is %s"
            % (
                axis.name,
                value,
                observed,
                comparator_family,
                comparator_baseline,
            )
            for value in axis.values
        ]
    if state == "contextual":
        engagement_tokens = _CONTEXTUAL_T1_ENGAGEMENT_TOKENS.get(axis.name)
        if engagement_tokens is None:
            return [], [
                "%s has a contextual absent-key policy but no T1 engagement "
                "contract — excluded fail-closed" % axis.name
            ]
        dropped.append(
            "%s has a contextual absent-key policy — keeping only tokens with sound "
            "counterfactual or execution evidence; "
            "its versioned T1 engagement contract must prove the override affected a row"
            % axis.name
        )
    elif state == "unknown":
        dropped.append(
            "%s has %s — keeping every token, because DefaultSpec is not proven to be "
            "the execution baseline" % (axis.name, baseline_source)
        )
    elif state == "distinct":
        dropped.append(
            "%s absence is distinct from every explicit domain token — keeping every token"
            % axis.name
        )

    kept = []
    for token in axis.values:
        one_sided_tokens = _ONE_SIDED_T1_ENGAGEMENT_TOKENS.get(axis.name)
        if one_sided_tokens is not None and token not in one_sided_tokens:
            dropped.append(
                "%s=%s has no row-local causal witness in the current one-sided "
                "T1 contract — excluded fail-closed" % (axis.name, token)
            )
            continue
        if (
            state == "contextual"
            and token not in _CONTEXTUAL_T1_ENGAGEMENT_TOKENS[axis.name]
        ):
            dropped.append(
                "%s=%s has no sound contextual counterfactual in the current T1 "
                "contract — excluded fail-closed" % (axis.name, token)
            )
            continue
        if not value_admissible(kind, token):
            dropped.append(
                "%s=%s is REJECTED by LeverKind::%s — it would resolve to the default and "
                "measure the baseline" % (axis.name, token, kind["kind"])
            )
            continue
        if baseline_token is not None and tokens_semantically_equal(
                kind, token, baseline_token):
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


def delivery_bucket(axis):
    return "preset" if axis.deliver.startswith("preset:") else "env-only"


def axis_applies_to_categories(axis, categories):
    """Whether a hard-scoped axis applies to every category in a shared arm."""
    if not axis.category_prefixes:
        return True
    normalized = [str(category).lower() for category in categories]
    return all(
        any(category.startswith(prefix.lower()) for prefix in axis.category_prefixes)
        for category in normalized
    )


def category_applicability_finding(axis, categories):
    """T0 finding for a hard-scoped axis that would be inert in this plan."""
    if axis_applies_to_categories(axis, categories):
        return None
    return (
        "%s is scoped to category prefix(es) %s and does not apply to every "
        "requested category %s — excluded as category-inert"
        % (axis.name, ",".join(axis.category_prefixes), ",".join(categories))
    )


_POSTBAB_RESERVE_BYPASS_CATEGORIES = frozenset({
    "isomorphic_acasxu_2026", "monotonic_acasxu_2026",
})


def runtime_applicability_finding(axis, categories, preset_docs=None,
                                  preset_expected=False):
    """T0 finding for reachability facts richer than a positive prefix list.

    Concurrent margin-row search also needs a common false and provably
    reserve-free typed baseline: otherwise the only currently creditable ON arm
    is either a no-op in some categories or guaranteed to pay the rejected
    verifier-contention + held-reserve double tax. Adaptive release is
    row/budget dependent, so T0 deliberately accepts only an exact typed zero;
    T1 still checks the allocation actually executed. The post-BaB reserve is
    structurally general, but two relational entry points bypass the standard
    wrapper and a category preset may explicitly disable it.
    """
    if axis.name == "NY_MARGIN_ROW_CONCURRENT":
        if not preset_expected:
            return (
                "%s requires a parsed typed preset with margin_row.reserve_secs=0 "
                "to avoid the unmeasured held-reserve double tax — excluded fail-closed"
                % axis.name
            )
        if not preset_docs or len(preset_docs) != len(categories):
            return (
                "%s requires every category preset to prove an exact reserve-free "
                "scheduler baseline — excluded fail-closed" % axis.name
            )
        blocked = []
        malformed = []
        already_armed = []
        for category, doc in zip(categories, preset_docs):
            node = doc
            state = "present"
            for part in ("margin_row", "reserve_secs"):
                if not isinstance(node, dict):
                    state = "invalid"
                    break
                if part not in node:
                    state = "absent"
                    break
                node = node[part]
            if state == "invalid":
                malformed.append(str(category))
                continue
            if state == "absent" or node is None:
                reserve = 45
            elif type(node) is int and 0 <= node <= (1 << 64) - 1:
                reserve = node
            else:
                malformed.append(str(category))
                continue
            if reserve != 0:
                blocked.append(str(category))

            node = doc
            state = "present"
            for part in ("margin_row", "concurrent"):
                if not isinstance(node, dict):
                    state = "invalid"
                    break
                if part not in node:
                    state = "absent"
                    break
                node = node[part]
            if state == "invalid":
                malformed.append(str(category))
                continue
            if state == "absent" or node is None:
                concurrent = False
            elif type(node) is bool:
                concurrent = node
            else:
                malformed.append(str(category))
                continue
            if concurrent:
                already_armed.append(str(category))
        if malformed:
            return (
                "%s cannot authenticate typed margin_row.reserve_secs/concurrent in "
                "category preset(s) %s — excluded fail-closed"
                % (axis.name, ",".join(malformed))
            )
        if already_armed:
            return (
                "%s=1 is already the typed baseline in category preset(s) %s; a shared "
                "ON arm would be a no-op there — excluded fail-closed"
                % (axis.name, ",".join(already_armed))
            )
        if blocked:
            return (
                "%s has no T0-safe reserve-free treatment in category preset(s) %s; "
                "only explicit typed reserve_secs=0 can enter the isolated concurrency A/B"
                % (axis.name, ",".join(blocked))
            )
        return None
    if axis.name != "NY_POSTBAB_RESERVE_SECS":
        return None
    bypassed = [
        category for category in categories
        if str(category).lower() in _POSTBAB_RESERVE_BYPASS_CATEGORIES
    ]
    if bypassed:
        return (
            "%s cannot reach the standard post-BaB wrapper in relational "
            "category invocation(s) %s — excluded as route-inert"
            % (axis.name, ",".join(bypassed))
        )
    if not preset_expected:
        return None
    if not preset_docs or len(preset_docs) != len(categories):
        return (
            "%s requires parsed category presets to prove its outer wrapper "
            "is not explicitly disabled — excluded fail-closed"
            % axis.name
        )
    wrapper_key = "preset:bab.phase_budget.vnncomp_post_bab_attack"
    disabled = [
        category for category, doc in zip(categories, preset_docs)
        if preset_effective_token(doc, wrapper_key) == "0"
    ]
    if disabled:
        return (
            "%s is disabled by bab.phase_budget.vnncomp_post_bab_attack=false "
            "in category preset(s) %s — excluded as preset-inert"
            % (axis.name, ",".join(disabled))
        )
    return None


def effective_delivery_filter(configured, requested):
    """Configured/scored searches exclude unshippable axes unless explicitly requested."""
    return list(requested) if requested else (["preset"] if configured else [])


def _dedupe(arms):
    seen = {}
    for arm in arms:
        # Exact canonical tuples avoid making a truncated display hash part of
        # the experiment's correctness contract.
        key = tuple(sorted(arm))
        seen.setdefault(key, list(key))
    return list(seen.values())


SEARCH_DESIGN_SCHEMA = "ny-search/contrast-design/v1"

# The finite-authority CUDA root lane is a recipe, not a bag of independent
# switches.  In particular, the call-local sound fold admits at most eight
# selected rows; the cGAN extraction needs one of the suffix admissions and
# BatchNorm extraction; and the root master historically removes the optional
# CPU proof handle whenever WGPU was not independently qualified.  A one-axis
# search that repaired only `child -> master` therefore spent its budget on a
# master-only arm with MAX_SEL=512 and usually never dispatched the lane.
#
# Keep this as PLANNING context rather than adding false `space::EDGES`: those
# edges describe values that are provably inert everywhere, whereas frozen-stop
# and BatchNorm are topology-specific execution cofactors.  For a child main
# effect these controls are added to both treatment and exact comparator, so
# the requested coordinate remains the only causal delta.  The master itself
# is excluded from automated planning below: switching the admission master on
# necessarily creates a composite recipe and cannot be called an independent
# one-axis effect.
_ROOT_JOINT_RECIPE_MASTER = "NY_ROOT_JOINT_INTERM_ALPHA"
_ROOT_JOINT_RECIPE_TRIGGERS = frozenset({
    "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT",
    "NY_ROOT_JOINT_INTERM_ALPHA_SECS",
    "NY_ROOT_JOINT_INTERM_ALPHA_ITERS",
    "NY_ROOT_JOINT_INTERM_ALPHA_LR",
    "NY_ROOT_JOINT_MIN_REMAINING_SECS",
    "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM",
    "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL",
    "NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP",
    "NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN",
    "NY_BN_GPU_EXTRACT",
})
_ROOT_JOINT_RECIPE_DEFAULTS = (
    (_ROOT_JOINT_RECIPE_MASTER, "1"),
    ("NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT", "1"),
    ("NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL", "8"),
    ("NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP", "1"),
    ("NY_BN_GPU_EXTRACT", "1"),
    # Equalize the master's historical CPU-handle suppression across each
    # child treatment/comparator pair.  A qualified WGPU proof device remains
    # retained by the runtime regardless of this kill switch.
    ("NY_CPU_COMPUTE_ENGINE", "0"),
)
_AUTOMATED_COMPOSITE_AXES = frozenset({_ROOT_JOINT_RECIPE_MASTER})


def _root_joint_recipe_prerequisites(requested):
    """Return the shared finite-CUDA context for a root-family child.

    `None` rejects an explicit master+child interaction: removing the master
    from its comparator would leave all root child controls lattice-inert, so
    that design cannot express the exact causal comparison it claims.
    """
    requested_names = set(requested)
    triggered = requested_names.intersection(_ROOT_JOINT_RECIPE_TRIGGERS)
    if not triggered:
        return {}
    if _ROOT_JOINT_RECIPE_MASTER in requested_names:
        return None
    context = dict(_ROOT_JOINT_RECIPE_DEFAULTS)
    if requested.get("NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN") == "1":
        # The dedicated pure-chain receipt is attributable only when frozen
        # stop was not an independent admitting authority.
        context["NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP"] = "0"
    for name in requested_names:
        context.pop(name, None)
    return context


def _repair_preserving(requested, space, kinds=None, baseline_values=None,
                       return_contrast=False, repair_baselines=None):
    """Repair prerequisites without changing any coordinate the design requested.

    Dependency repair may add controls, but overwriting or deleting a requested
    coordinate turns the candidate into a different experiment. Return `None`
    for that forbidden corner instead of silently relabelling it.
    """
    recipe = _root_joint_recipe_prerequisites(requested)
    if recipe is None:
        return None
    candidate = dict(requested)
    candidate.update(recipe)
    sample, added = repair(
        candidate,
        space,
        kinds,
        baseline_values=baseline_values,
        repair_baselines=repair_baselines,
    )
    added = set(added).union(recipe)
    if (any(sample.get(name) != value for name, value in requested.items())
            or is_inert(sample, space)):
        return None
    repaired = sorted(sample.items())
    if not return_contrast:
        return repaired
    requested_assignment = sorted(requested.items())
    prerequisites_added = sorted(
        (name, sample[name]) for name in added if name in sample
    )
    comparison_assignment = list(prerequisites_added)
    payload = {
        "schema": SEARCH_DESIGN_SCHEMA,
        "requested_axes": sorted(requested),
        "requested_assignment": [list(pair) for pair in requested_assignment],
        "prerequisites_added": [list(pair) for pair in prerequisites_added],
        "treatment_assignment": [list(pair) for pair in repaired],
        "comparison_assignment": [list(pair) for pair in comparison_assignment],
    }
    payload["contrast_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return repaired, payload


def _guard_candidates(mode, count, max_candidates):
    if max_candidates < 1:
        raise SystemExit("--max-candidates must be at least 1")
    if count > max_candidates:
        raise SystemExit(
            "%s would enumerate %d raw candidate(s) (> --max-candidates %d); "
            "narrow it with --axis or raise the explicit enumeration bound"
            % (mode, count, max_candidates)
        )


def _guard_arms(mode, arms, max_arms):
    if max_arms < 1:
        raise SystemExit("--max-arms must be at least 1")
    if len(arms) > max_arms:
        raise SystemExit(
            "%s requires %d executable arm(s) (> --max-arms %d); narrow it with "
            "--axis or raise the explicit run bound" % (mode, len(arms), max_arms)
        )


def _pair_cover_request_error(mode, axis_filter, no_baseline=False):
    """CLI-only prerequisites for a deliberately bounded interaction design."""
    if mode != "pair-cover":
        return None
    if len(set(axis_filter)) < 2:
        return "pair-cover requires at least two distinct explicit --axis filters"
    if no_baseline:
        return "pair-cover always includes its control; --no-baseline is incompatible"
    return None


def generate_arms(space, pool, mode, samples, seed, max_axes, baseline=True, max_arms=256,
                  kinds=None, baseline_values=None,
                  max_candidates=DEFAULT_MAX_CANDIDATES,
                  return_design=False, repair_baselines=None):
    """Arms that are valid BY CONSTRUCTION: never unsafe, never lattice-inert.

    `pool` is the T0-filtered `[(Axis, [tokens])]` the search may move. Random
    mode picks a k-subset and repairs it under the lattice. With the CLI default
    `max_axes=1`, every random arm has one sampled treatment (plus any explicit
    dependency controls repair must add), so a paired delta is attributable to
    that treatment. `max_axes > 1` deliberately screens joint interactions.
    Cover mode emits a repaired candidate for every T0-kept `(axis, value)`.
    Pair-cover adds every two-axis value combination, but is deliberately
    restricted by the CLI to an explicit shortlist. Grid takes the full product.
    Both modes repair by ADDING prerequisite controls and drop any forbidden
    corner whose requested coordinate would have to change. `max_candidates`
    bounds raw enumeration work; `max_arms` independently bounds the repaired,
    deduplicated experiments that would consume benchmark budgets.
    """
    rng = random.Random(seed)
    arms = [[]] if baseline else []
    contrasts_by_assignment = {}
    dropped_inert = 0

    def add_requested(requested):
        result = _repair_preserving(
            requested,
            space,
            kinds,
            baseline_values=baseline_values,
            return_contrast=True,
            repair_baselines=repair_baselines,
        )
        if result is None:
            return None
        repaired, contrast = result
        arms.append(repaired)
        key = tuple(repaired)
        known = contrasts_by_assignment.setdefault(key, [])
        if all(item["contrast_id"] != contrast["contrast_id"] for item in known):
            known.append(contrast)
        return repaired
    if mode == "pair-cover" and (len(pool) < 2 or not baseline):
        if not baseline:
            raise SystemExit("pair-cover always includes its baseline control")
        raise SystemExit("pair-cover requires at least two T0-kept axes")
    if mode == "random" and samples < 0:
        raise SystemExit("random --samples must be non-negative")
    if not pool:
        if mode == "random" and samples:
            raise SystemExit(
                "random requested %d unique treatment arm(s), but T0 left no searchable axis"
                % samples
            )
        if mode in ("cover", "grid"):
            _guard_arms(mode, arms, max_arms)
        empty_design = {
            "schema": SEARCH_DESIGN_SCHEMA,
            "contrasts_by_assignment": {},
            "comparison_for_assignment": {},
        }
        return (arms, 0, empty_design) if return_design else (arms, 0)
    if mode == "grid":
        # `None` is a real grid level: an ABSENT axis means "leave at the
        # compiled default", which is the only way a grid can express the
        # mutually exclusive root-phase family at all. Without it every point
        # sets every axis, and the ownership edges annihilate the whole grid.
        levels = [[None] + list(values) for _, values in pool]
        total = 1
        for level in levels:
            total *= len(level)
        _guard_candidates("grid", total, max_candidates)
        for point in itertools.product(*levels):
            sample = {
                axis.name: token for (axis, _), token in zip(pool, point) if token is not None
            }
            if not sample:
                continue  # the all-default point IS the baseline arm
            requested = dict(sample)
            # Repair is allowed to add explicit controls for owners armed by
            # the typed preset. It must not silently turn a forbidden grid
            # corner into some different treatment by deleting/changing a
            # coordinate the grid requested.
            repaired = add_requested(requested)
            if repaired is None:
                dropped_inert += 1
                continue
        arms = _dedupe(arms)
        _guard_arms("grid", arms, max_arms)
    elif mode == "cover":
        requested = {
            (axis.name, token)
            for axis, values in pool
            for token in values
        }
        covered = set()
        for axis, values in pool:
            for token in values:
                repaired = add_requested({axis.name: token})
                if repaired is None:
                    dropped_inert += 1
                    continue
                covered.add((axis.name, token))
        arms = _dedupe(arms)
        missing = sorted(requested - covered)
        if missing:
            raise SystemExit(
                "cover cannot exercise %d axis/value pair(s) after dependency repair: %s"
                % (len(missing), ", ".join("%s=%s" % pair for pair in missing[:8]))
            )
        _guard_arms("cover", arms, max_arms)
    elif mode == "pair-cover":
        main_count = sum(len(values) for _axis, values in pool)
        pair_count = sum(
            len(left_values) * len(right_values)
            for (_left, left_values), (_right, right_values)
            in itertools.combinations(pool, 2)
        )
        raw_candidates = len(arms) + main_count + pair_count
        # Count from domain cardinalities before constructing the requested
        # design sets: this guard exists specifically to stop a huge shortlist
        # before its Cartesian products consume memory.
        _guard_candidates("pair-cover", raw_candidates, max_candidates)
        requested_main = {
            (axis.name, token)
            for axis, values in pool
            for token in values
        }
        requested_pairs = set()
        for (left, left_values), (right, right_values) in itertools.combinations(pool, 2):
            for left_token, right_token in itertools.product(left_values, right_values):
                requested_pairs.add(
                    ((left.name, left_token), (right.name, right_token))
                )

        covered_main = set()
        covered_pairs = set()
        for axis, values in pool:
            for token in values:
                requested = {axis.name: token}
                repaired = add_requested(requested)
                if repaired is None:
                    dropped_inert += 1
                    continue
                covered_main.add((axis.name, token))
        for (left, left_values), (right, right_values) in itertools.combinations(pool, 2):
            for left_token, right_token in itertools.product(left_values, right_values):
                requested = {left.name: left_token, right.name: right_token}
                repaired = add_requested(requested)
                pair = ((left.name, left_token), (right.name, right_token))
                if repaired is None:
                    dropped_inert += 1
                    continue
                covered_pairs.add(pair)
        missing_main = sorted(requested_main - covered_main)
        missing_pairs = sorted(requested_pairs - covered_pairs)
        if missing_main or missing_pairs:
            examples = ["%s=%s" % pair for pair in missing_main[:4]]
            examples.extend(
                "%s=%s + %s=%s" % (pair[0][0], pair[0][1], pair[1][0], pair[1][1])
                for pair in missing_pairs[:max(0, 8 - len(examples))]
            )
            raise SystemExit(
                "pair-cover cannot exercise %d main effect(s) and %d value pair(s) after "
                "dependency repair without changing a requested coordinate: %s"
                % (len(missing_main), len(missing_pairs), ", ".join(examples))
            )
        arms = _dedupe(arms)
        _guard_arms("pair-cover", arms, max_arms)
    elif mode == "random":
        attempts = 0
        while len(arms) < samples + (1 if baseline else 0) and attempts < samples * 64:
            attempts += 1
            width = rng.randint(1, max(1, min(max_axes, len(pool))))
            chosen = rng.sample(pool, width)
            requested = {axis.name: rng.choice(values) for axis, values in chosen}
            repaired = add_requested(requested)
            if repaired is None:
                # A changed or unreachable requested coordinate is not the arm
                # random sampled. Resample rather than silently relabel it.
                dropped_inert += 1
                continue
            arms = _dedupe(arms)
        treatment_count = len(arms) - (1 if baseline else 0)
        if treatment_count < samples:
            raise SystemExit(
                "random requested %d unique treatment arm(s), but generated only %d after "
                "%d attempts; narrow the pool, lower --samples, or use cover for a "
                "deterministic complete design" % (samples, treatment_count, attempts)
            )
    else:
        raise ValueError("unknown generation mode %r" % mode)

    # A repaired prerequisite is part of the treatment's execution context,
    # not part of its causal contrast. Schedule the exact prerequisite-only
    # assignment as the comparator. Distinct intents survive assignment
    # deduplication: one executable arm can legitimately carry several
    # contrast records (for example a child main effect and an explicit
    # parent+child interaction).
    comparison_for_assignment = {}
    for contrasts in contrasts_by_assignment.values():
        contrasts.sort(key=lambda item: item["contrast_id"])
        for contrast in contrasts:
            comparison = [tuple(pair) for pair in contrast["comparison_assignment"]]
            key = tuple(comparison)
            comparison_for_assignment.setdefault(key, []).append(
                contrast["contrast_id"]
            )
            arms.append(comparison)
    for contrast_ids in comparison_for_assignment.values():
        contrast_ids.sort()
    arms = _dedupe(arms)
    _guard_arms(mode, arms, max_arms)
    design = {
        "schema": SEARCH_DESIGN_SCHEMA,
        "contrasts_by_assignment": contrasts_by_assignment,
        "comparison_for_assignment": comparison_for_assignment,
    }
    if return_design:
        return arms, dropped_inert, design
    return arms, dropped_inert


def _assignment_design_roles(design, arm):
    key = tuple(sorted(arm))
    return (
        list(design.get("contrasts_by_assignment", {}).get(key, [])),
        list(design.get("comparison_for_assignment", {}).get(key, [])),
    )


def empty_comparator_contrast_ids(design):
    """Contrasts whose exact prerequisite-only control is the default arm."""
    return sorted(
        contrast["contrast_id"]
        for contrasts in design.get("contrasts_by_assignment", {}).values()
        for contrast in contrasts
        if not contrast.get("comparison_assignment")
    )


def search_design_manifest(arms, design, *, mode, seed, max_axes, categories,
                           year, repeats, space_source, evidence_scope):
    """Seal causal intents separately from the native full-arm manifest."""
    assignments = []
    for arm in sorted((sorted(arm) for arm in arms), key=lambda item: tuple(item)):
        contrasts, comparison_for = _assignment_design_roles(design, arm)
        assignments.append({
            "arm_id": arm_id(arm),
            "assignment": [list(pair) for pair in arm],
            "contrasts": contrasts,
            "comparison_for": comparison_for,
        })
    payload = {
        "schema": SEARCH_DESIGN_SCHEMA,
        "mode": mode,
        "seed": seed,
        "max_axes": max_axes,
        "categories": list(categories),
        "year": year,
        "repeats": repeats,
        "space_source": space_source,
        "evidence_scope": evidence_scope,
        "assignments": assignments,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["identity"] = {
        "schema": "ny-search/contrast-design-identity/v1",
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "assignment_count": len(assignments),
        "contrast_count": sum(len(item["contrasts"]) for item in assignments),
    }
    return payload


def search_design_manifest_issues(document):
    """Strictly revalidate the persisted contrast plan and its identity."""
    if not isinstance(document, dict):
        return ["search design is not an object"]
    expected_fields = {
        "schema", "mode", "seed", "max_axes", "categories", "year",
        "repeats", "space_source", "evidence_scope", "assignments", "identity",
    }
    issues = []
    if set(document) != expected_fields or document.get("schema") != SEARCH_DESIGN_SCHEMA:
        issues.append("search design has an unknown schema or field set")
        return issues
    identity = document.get("identity")
    payload = {key: value for key, value in document.items() if key != "identity"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    expected_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if not isinstance(identity, dict) or set(identity) != {
        "schema", "sha256", "assignment_count", "contrast_count",
    } or identity.get("schema") != "ny-search/contrast-design-identity/v1" \
            or identity.get("sha256") != expected_sha:
        issues.append("search design identity does not match its canonical payload")
        return issues
    assignments = document.get("assignments")
    if not isinstance(assignments, list):
        return issues + ["search design assignments is not a list"]
    by_assignment = {}
    contrast_comparators = {}
    seen_contrast_ids = set()
    seen_arm_ids = set()
    contrast_count = 0
    assignment_order = []
    observed_scopes = set()
    for item in assignments:
        if not isinstance(item, dict) or set(item) != {
            "arm_id", "assignment", "contrasts", "comparison_for",
        }:
            issues.append("search design assignment has an unknown field set")
            continue
        assignment = _canonical_record_assignment(item.get("assignment"))
        if assignment is None or item.get("arm_id") != arm_id(assignment):
            issues.append("search design assignment is noncanonical or misidentified")
            continue
        if assignment in by_assignment or item["arm_id"] in seen_arm_ids:
            issues.append(
                "search design repeats an executable assignment or truncated arm identity"
            )
            continue
        contrasts = item.get("contrasts")
        comparison_for = item.get("comparison_for")
        if (
            not isinstance(contrasts, list)
            or not isinstance(comparison_for, list)
            or not all(
                isinstance(value, str)
                and re.fullmatch(r"[0-9a-f]{16}", value) is not None
                for value in comparison_for
            )
            or comparison_for != sorted(set(comparison_for))
        ):
            issues.append("search design assignment roles are malformed")
            continue
        contrast_ids = []
        valid_contrasts = True
        for contrast in contrasts:
            if not _contrast_contract_valid(contrast, assignment):
                issues.append("search design contains a malformed contrast")
                valid_contrasts = False
                continue
            contrast_id = contrast["contrast_id"]
            if contrast_id in contrast_ids or contrast_id in seen_contrast_ids:
                issues.append("search design repeats a contrast entry or identifier")
                valid_contrasts = False
                continue
            contrast_ids.append(contrast_id)
        if contrast_ids != sorted(contrast_ids):
            issues.append("search design contrasts are not in canonical identifier order")
            valid_contrasts = False
        if not valid_contrasts:
            continue
        by_assignment[assignment] = item
        seen_arm_ids.add(item["arm_id"])
        assignment_order.append(assignment)
        for contrast in contrasts:
            contrast_id = contrast["contrast_id"]
            seen_contrast_ids.add(contrast_id)
            contrast_count += 1
            observed_scopes.add(contrast["evidence_scope"])
            contrast_comparators[contrast_id] = _canonical_record_assignment(
                contrast["comparison_assignment"]
            )
    if assignment_order != sorted(assignment_order):
        issues.append("search design assignments are not in canonical order")
    if identity.get("assignment_count") != len(assignments) \
            or identity.get("contrast_count") != contrast_count:
        issues.append("search design identity counts disagree with its assignments")
    for contrast_id, comparator in contrast_comparators.items():
        comparator_item = by_assignment.get(comparator)
        if comparator_item is None or contrast_id not in comparator_item["comparison_for"]:
            issues.append("search design does not schedule/seal an exact comparator")
    declared_comparisons = {
        contrast_id
        for item in by_assignment.values()
        for contrast_id in item["comparison_for"]
    }
    if declared_comparisons != set(contrast_comparators):
        issues.append("search design comparator roles contain an unknown/missing contrast")
    expected_scope = (
        next(iter(observed_scopes)) if len(observed_scopes) == 1
        else "mixed" if observed_scopes else "scored"
    )
    if document.get("evidence_scope") != expected_scope:
        issues.append("search design evidence scope disagrees with its contrasts")
    return issues


def _atomic_write_utf8(path, text):
    """Atomically replace one UTF-8 design artifact in its destination directory."""
    path = Path(path)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _resolved_output_path(path):
    """Canonicalize an output path, including a not-yet-created final name."""
    return Path(path).resolve(strict=False)


def output_layout_issues(out_dir, results_path, design_path, arms, categories, repeats):
    """Reject result targets that can alias a plan or native run artifact."""
    raw_results_path = Path(results_path)
    raw_design_path = Path(design_path)
    out_dir = _resolved_output_path(out_dir)
    results_path = _resolved_output_path(results_path)
    design_path = _resolved_output_path(design_path)
    default_results = out_dir / "results.jsonl"
    issues = []
    if raw_results_path.is_symlink():
        issues.append("results path must not be a symlink")
    if raw_design_path.is_symlink():
        issues.append("search_plan.json must not be a symlink")
    if results_path == design_path:
        issues.append("results path aliases search_plan.json")
    if results_path == out_dir or results_path.is_dir():
        issues.append("results path resolves to a directory")
    try:
        inside_out_dir = results_path.is_relative_to(out_dir)
    except AttributeError:  # Python 3.8 compatibility for research hosts.
        inside_out_dir = out_dir == results_path or out_dir in results_path.parents
    if inside_out_dir and results_path != default_results:
        issues.append(
            "a results file inside --out-dir must be its root results.jsonl, "
            "not a selector or per-arm artifact path"
        )
    for repeat in range(repeats):
        for arm in arms:
            for category in categories:
                run_dir = out_dir / arm_id(arm) / ("%s.r%d" % (category, repeat))
                if results_path == run_dir or run_dir in results_path.parents:
                    issues.append("results path aliases a planned native run directory")
                    return sorted(set(issues))
    return sorted(set(issues))


def causal_result_stream_issues(records, expected_design_sha256):
    """Validate exact causal roles and uniqueness in one result stream."""
    issues = []
    seen_runs = set()
    for index, record in enumerate(records, 1):
        if (
            not isinstance(record, dict)
            or record.get("schema") != RESULT_SCHEMA
            or record.get("search_design_schema") != SEARCH_DESIGN_SCHEMA
            or record.get("search_design_sha256") != expected_design_sha256
        ):
            issues.append(
                "record %d belongs to another result schema or search design"
                % index
            )
            continue
        assignment = _canonical_record_assignment(record.get("arm"))
        if (
            assignment is None
            or _validated_result_design_roles(record, assignment) is None
            or not isinstance(record.get("category"), str)
            or not record["category"]
            or type(record.get("repeat")) is not int
            or record["repeat"] < 0
        ):
            issues.append("record %d has malformed causal result roles" % index)
            continue
        run_key = (assignment, record["category"], record["repeat"])
        if run_key in seen_runs:
            issues.append("record %d duplicates an earlier planned run" % index)
        seen_runs.add(run_key)
    return issues


def existing_result_stream_issues(path, expected_design_sha256):
    """Prove an existing reuse stream belongs wholly to the selected plan.

    A successful reuse replaces this stream atomically.  Until then, leave the
    previous file untouched and refuse to overwrite an unrelated/mixed stream.
    """
    path = Path(path)
    if not path.exists():
        return []
    if not path.is_file():
        return ["existing results target is not a regular file"]
    issues = []
    records = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as error:
                    issues.append("line %d is not JSON: %s" % (lineno, error))
                    continue
                records.append(record)
    except (OSError, UnicodeError) as error:
        return ["cannot read existing results: %s" % error]
    return issues + causal_result_stream_issues(records, expected_design_sha256)


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


def _parse_selector_csv_line(line):
    """Mirror Rust `bench_vnncomp::parse_csv_line` for one physical line."""
    fields = []
    field = []
    state = "start"
    index = 0
    while index < len(line):
        char = line[index]
        if state == "start":
            if char == ",":
                fields.append("")
            elif char == '"':
                state = "quoted"
            elif char in "\r\n":
                raise ValueError("CSV record contains a line separator")
            else:
                field.append(char)
                state = "unquoted"
        elif state == "unquoted":
            if char == ",":
                fields.append("".join(field).strip())
                field = []
                state = "start"
            elif char == '"':
                raise ValueError("quote appears inside an unquoted CSV field")
            elif char in "\r\n":
                raise ValueError("CSV record contains a line separator")
            else:
                field.append(char)
        elif state == "quoted":
            if char == '"' and index + 1 < len(line) and line[index + 1] == '"':
                field.append('"')
                index += 1
            elif char == '"':
                state = "after_quote"
            elif char in "\r\n":
                raise ValueError("CSV record contains a line separator")
            else:
                field.append(char)
        else:
            if char == ",":
                fields.append("".join(field).strip())
                field = []
                state = "start"
            elif not char.isspace():
                raise ValueError("non-whitespace data follows a closing CSV quote")
        index += 1
    if state == "quoted":
        raise ValueError("unterminated quoted CSV field")
    fields.append("".join(field).strip())
    return fields


def _semantic_identity(schema, entries):
    """Stable order-insensitive identity for a set represented by sorted entries."""
    canonical = sorted([list(entry) for entry in entries])
    encoded = json.dumps(
        {"schema": schema, "entries": canonical},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": schema,
        "count": len(canonical),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def parse_instance_selectors(body):
    """Parse Rust-compatible `category,onnx,vnnlib` selector lines.

    Input order is retained for native error/report order. The semantic identity
    is set-like because Rust converts the selectors to a `HashSet` for selection;
    reordering comments or selectors must not split otherwise identical trials.
    """
    ordered = []
    seen = set()
    for lineno, raw in enumerate(body.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            fields = _parse_selector_csv_line(line)
        except ValueError as error:
            raise ValueError("line %d: malformed CSV: %s" % (lineno, error))
        if len(fields) != 3:
            raise ValueError(
                "line %d: expected exactly 3 fields (category,onnx,vnnlib), got %d"
                % (lineno, len(fields))
            )
        if any(not field for field in fields):
            raise ValueError(
                "line %d: category, model, and property fields must be non-empty" % lineno
            )
        selector = tuple(fields)
        if selector in seen:
            raise ValueError(
                "line %d: %s is listed twice; a row subset must be exact"
                % (lineno, ",".join(selector))
            )
        seen.add(selector)
        ordered.append(selector)
    if not ordered:
        raise ValueError("the file names no rows; an empty subset would measure nothing")
    return ordered


def load_instance_selection(path):
    """Read, validate, canonicalize and fingerprint one native selector file."""
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("--instances is not a file: %s" % source)
    raw = source.read_bytes()
    try:
        body = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("--instances is not UTF-8: %s" % error)
    ordered = parse_instance_selectors(body)
    by_category = {}
    for selector in ordered:
        by_category.setdefault(selector[0], []).append(selector)
    return {
        "source": {
            "canonical_path": str(source),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "ordered": ordered,
        "by_category": by_category,
        "identity": _semantic_identity(INSTANCE_SELECTOR_SCHEMA, ordered),
        "category_identities": {
            category: _semantic_identity(INSTANCE_SELECTOR_SCHEMA, selectors)
            for category, selectors in by_category.items()
        },
    }


def instance_request_error(selection, categories, limit):
    if selection is None:
        return None
    if limit is not None:
        return "--instances and --limit are mutually exclusive"
    requested = set(categories)
    selected = set(selection["by_category"])
    if requested != selected:
        return (
            "--instances categories [%s] must exactly match requested --category values [%s]"
            % (", ".join(sorted(selected)), ", ".join(sorted(requested)))
        )
    return None


def duplicate_category_error(categories):
    """Refuse repeated native invocations masquerading as independent repeats."""
    seen = set()
    duplicates = set()
    for category in categories:
        if category in seen:
            duplicates.add(category)
        seen.add(category)
    if not duplicates:
        return None
    return "--category repeated: %s" % ", ".join(sorted(duplicates))


def limit_request_error(limit):
    if limit is not None and limit < 1:
        return "--limit must be at least 1"
    return None


def _selector_csv_escape(field):
    if any(char in field for char in (",", '"', "\n", "\r")):
        return '"%s"' % field.replace('"', '""')
    return field


def materialize_instance_selections(selection, directory):
    """Write one canonical native selector file per category-specific child."""
    root = Path(directory) / "instance-selectors"
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for category, selectors in sorted(selection["by_category"].items()):
        body = "".join(
            ",".join(_selector_csv_escape(field) for field in selector) + "\n"
            # Native selection is set-like after parsing. Sorting makes the
            # projected artifact canonical while `selection["ordered"]` still
            # retains source order for faithful diagnostics.
            for selector in sorted(selectors)
        )
        category_key = hashlib.sha256(category.encode("utf-8")).hexdigest()[:12]
        path = root / ("%s.%s.csv" % (selection["identity"]["sha256"][:16], category_key))
        if path.exists():
            if path.read_text(encoding="utf-8") != body:
                raise ValueError("existing canonical selector file disagrees: %s" % path)
        else:
            path.write_text(body, encoding="utf-8")
        paths[category] = str(path.resolve())
    return paths


def measured_row_set_identity(rows):
    """Exact multiset identity, including duplicate-corpus occurrence and index."""
    identities = []
    for row in rows:
        fields = [row.get("category"), row.get("onnx"), row.get("vnnlib")]
        if any(not isinstance(field, str) or not field for field in fields):
            raise ValueError("metadata row has an incomplete category/model/property identity")
        occurrence = row.get("occurrence")
        instance_index = row.get("instance_index")
        if (not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 0
                or not isinstance(instance_index, int) or isinstance(instance_index, bool)
                or instance_index < 0):
            raise ValueError("metadata row has an invalid occurrence or instance_index")
        identities.append(tuple(fields + [occurrence, instance_index]))
    if len(set(identities)) != len(identities):
        raise ValueError("metadata bank repeats an exact row identity")
    return _semantic_identity(MEASURED_ROW_SET_SCHEMA, identities)


def _manifest_semantic_identity(identity, schema, what, allow_empty):
    if not isinstance(identity, dict):
        raise ValueError("manifest %s is not an object" % what)
    if identity.get("schema") != schema:
        raise ValueError("manifest %s has the wrong identity schema" % what)
    digest = identity.get("sha256")
    count = identity.get("count")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not isinstance(count, int) or isinstance(count, bool)
            or count < (0 if allow_empty else 1)):
        raise ValueError("manifest %s has an invalid sha256/count" % what)
    return {"schema": schema, "sha256": digest, "count": count}


def _native_instance_selection(manifest):
    """Strict normalized native selection seal; legacy absence fails closed."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not an object")
    native = manifest.get("instance_selection")
    if not isinstance(native, dict):
        raise ValueError("manifest instance_selection is missing or is not an object")
    mode = native.get("mode")
    if mode == "all":
        return {"mode": "all"}
    if mode == "limit":
        per_category = native.get("per_category")
        if (not isinstance(per_category, int) or isinstance(per_category, bool)
                or per_category < 0):
            raise ValueError("manifest limit instance_selection has invalid per_category")
        return {"mode": "limit", "per_category": per_category}
    if mode == "exact":
        identity = _manifest_semantic_identity(
            native.get("identity"), INSTANCE_SELECTOR_SCHEMA,
            "instance_selection identity", allow_empty=False)
        return {"mode": "exact", "identity": identity}
    raise ValueError("manifest instance_selection has unknown mode %r" % mode)


def _native_selected_rows(manifest, category):
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not an object")
    selected = manifest.get("selected_rows")
    if not isinstance(selected, dict):
        raise ValueError("manifest selected_rows is missing or is not an object")
    if category not in selected:
        raise ValueError("manifest selected_rows has no seal for category %r" % category)
    return _manifest_semantic_identity(
        selected[category], MEASURED_ROW_SET_SCHEMA,
        "selected_rows[%r]" % category, allow_empty=True)


def _validated_current_artifact(artifact, what, fingerprint_cache=None):
    """Strictly validate one native path/hash seal against current file bytes."""
    if not isinstance(artifact, dict):
        raise ValueError("manifest %s is not an object" % what)
    canonical_path = artifact.get("canonical_path")
    expected_sha = artifact.get("sha256")
    if (not isinstance(canonical_path, str) or not canonical_path
            or not isinstance(expected_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)):
        raise ValueError("manifest %s has an invalid canonical_path/sha256" % what)

    try:
        resolved = Path(canonical_path).resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise ValueError("manifest %s cannot be inspected: %s" % (what, error)) from error
    if str(resolved) != canonical_path or not resolved.is_file():
        raise ValueError(
            "manifest %s canonical path no longer resolves to the sealed regular file"
            % what
        )
    stat_identity = (
        metadata.st_dev, metadata.st_ino, metadata.st_size,
        metadata.st_mtime_ns, metadata.st_ctime_ns,
    )
    cache = fingerprint_cache if fingerprint_cache is not None else {}
    cache_key = (canonical_path, expected_sha, stat_identity)
    current = cache.get(cache_key)
    if current is None:
        try:
            current = fingerprint_file(canonical_path)
        except (OSError, ValueError) as error:
            raise ValueError("manifest %s cannot be re-read: %s" % (what, error)) from error
        cache[cache_key] = current
    try:
        after = resolved.stat()
    except OSError as error:
        raise ValueError("manifest %s disappeared while it was checked" % what) from error
    after_identity = (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    )
    if after_identity != stat_identity:
        raise ValueError("manifest %s changed while its identity was checked" % what)
    expected = {"canonical_path": canonical_path, "sha256": expected_sha}
    if current != expected:
        raise ValueError(
            "manifest %s no longer matches current file identity %r (sealed %r)"
            % (what, current, expected)
        )
    return expected


def _protobuf_varint(payload, offset, what):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(payload):
            raise ValueError("%s has a truncated protobuf varint" % what)
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7f) << shift
        if not (byte & 0x80):
            return value, offset
    raise ValueError("%s has an overlong protobuf varint" % what)


def _protobuf_fields(payload, what):
    """Minimal fail-closed protobuf wire walk for the public ONNX schema."""
    offset = 0
    while offset < len(payload):
        key, offset = _protobuf_varint(payload, offset, what)
        field, wire = key >> 3, key & 7
        if field == 0:
            raise ValueError("%s has protobuf field zero" % what)
        if wire == 0:
            value, offset = _protobuf_varint(payload, offset, what)
        elif wire == 1:
            end = offset + 8
            if end > len(payload):
                raise ValueError("%s has a truncated fixed64 field" % what)
            value, offset = payload[offset:end], end
        elif wire == 2:
            length, offset = _protobuf_varint(payload, offset, what)
            end = offset + length
            if end > len(payload):
                raise ValueError("%s has a truncated length-delimited field" % what)
            value, offset = payload[offset:end], end
        elif wire == 5:
            end = offset + 4
            if end > len(payload):
                raise ValueError("%s has a truncated fixed32 field" % what)
            value, offset = payload[offset:end], end
        else:
            raise ValueError("%s uses unsupported protobuf wire type %d" % (what, wire))
        yield field, wire, value


def _protobuf_utf8(payload, what):
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("%s is not UTF-8" % what) from error


def _normalize_onnx_external_location(raw, what):
    if not isinstance(raw, str) or not raw or "\0" in raw:
        raise ValueError("%s has an empty or NUL-containing location" % what)
    if ("\\" in raw or raw.startswith("/")
            or (len(raw) >= 2 and raw[0].isascii()
                and raw[0].isalpha() and raw[1] == ":")):
        raise ValueError("%s location %r is not a portable relative POSIX path" % (what, raw))
    parts = []
    for part in raw.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError("%s location %r escapes the model directory" % (what, raw))
        parts.append(part)
    if not parts:
        raise ValueError("%s location %r does not name a file" % (what, raw))
    return "/".join(parts)


def _onnx_string_map_entry(payload, what):
    key, value = None, None
    for field, wire, raw in _protobuf_fields(payload, what):
        if field == 1 and wire == 2:
            key = _protobuf_utf8(raw, "%s key" % what)
        elif field == 2 and wire == 2:
            value = _protobuf_utf8(raw, "%s value" % what)
    if key is None or value is None:
        raise ValueError("%s is missing key or value" % what)
    return key, value


def _onnx_tensor_external_location(payload, what):
    data_location = 0
    entries = []
    for field, wire, raw in _protobuf_fields(payload, what):
        if field == 13 and wire == 2:
            entries.append(_onnx_string_map_entry(raw, "%s external_data" % what))
        elif field == 14 and wire == 0:
            data_location = raw
    if data_location == 0:
        if entries:
            raise ValueError("%s has external_data but DEFAULT data_location" % what)
        return None
    if data_location != 1:
        raise ValueError("%s has unknown data_location %r" % (what, data_location))

    values = {}
    for key, value in entries:
        if key in values:
            raise ValueError("%s repeats external_data key %r" % (what, key))
        if key not in ("location", "offset", "length", "checksum", "basepath"):
            raise ValueError("%s has unknown external_data key %r" % (what, key))
        values[key] = value
    if "location" not in values:
        raise ValueError("%s external_data has no location" % what)
    return _normalize_onnx_external_location(values["location"], what)


def _onnx_attribute_external_locations(payload, what):
    attribute_type = 0
    tensors = []
    unmodeled_containers = []
    for field, wire, raw in _protobuf_fields(payload, what):
        if field == 5 and wire == 2:
            tensors.append(raw)
        elif field in (6, 10, 11, 22, 23) and wire == 2:
            unmodeled_containers.append(field)
        elif field == 20 and wire == 0:
            attribute_type = raw
    if bool(tensors) != (attribute_type == 4):
        raise ValueError("%s has inconsistent Tensor payload/type metadata" % what)
    if unmodeled_containers or attribute_type in (5, 9, 10, 11, 12):
        raise ValueError("%s uses an unsupported nested tensor container" % what)
    if not tensors:
        return []
    # Protobuf merges repeated encodings of one singular message. Concatenating
    # their inner wire streams is the equivalent merge for the TensorProto
    # fields this fail-closed walker consumes.
    location = _onnx_tensor_external_location(b"".join(tensors), "%s Tensor" % what)
    return [] if location is None else [location]


def _onnx_node_external_locations(payload, what):
    locations = []
    index = 0
    for field, wire, raw in _protobuf_fields(payload, what):
        if field == 5 and wire == 2:
            locations.extend(_onnx_attribute_external_locations(
                raw, "%s attribute[%d]" % (what, index)))
            index += 1
    return locations


def _onnx_graph_external_locations(payload, what):
    locations = []
    initializer_index, node_index = 0, 0
    for field, wire, raw in _protobuf_fields(payload, what):
        if field == 1 and wire == 2:
            locations.extend(_onnx_node_external_locations(
                raw, "%s node[%d]" % (what, node_index)))
            node_index += 1
        elif field == 5 and wire == 2:
            location = _onnx_tensor_external_location(
                raw, "%s initializer[%d]" % (what, initializer_index))
            initializer_index += 1
            if location is not None:
                locations.append(location)
        elif field == 15 and wire == 2:
            raise ValueError("%s contains an unsupported sparse initializer" % what)
    return locations


def _current_onnx_external_locations(model_artifact, fingerprint_cache=None):
    canonical_path, expected_sha = (
        model_artifact["canonical_path"], model_artifact["sha256"])
    cache = fingerprint_cache if fingerprint_cache is not None else {}
    cache_key = ("onnx-external-data-v1", canonical_path, expected_sha)
    cached = cache.get(cache_key)
    if cached is not None:
        return list(cached)
    try:
        container = Path(canonical_path).read_bytes()
        observed_sha = hashlib.sha256(container).hexdigest()
        if observed_sha != expected_sha:
            raise ValueError(
                "ONNX container changed while its dependency closure was read "
                "(%s != %s)" % (observed_sha, expected_sha)
            )
        # flate2::read::GzDecoder (used by native sealing and the actual loader)
        # consumes exactly the first gzip member. Python's gzip.decompress
        # expands every concatenated member, so use zlib's single-stream mode.
        payload = (zlib.decompress(container, 16 + zlib.MAX_WBITS)
                   if Path(canonical_path).suffix == ".gz" else container)
    except (OSError, zlib.error) as error:
        raise ValueError("cannot decode ONNX dependency closure: %s" % error) from error
    graphs = [raw for field, wire, raw in _protobuf_fields(payload, "ONNX ModelProto")
              if field == 7 and wire == 2]
    locations = [] if not graphs else _onnx_graph_external_locations(
        b"".join(graphs), "ONNX GraphProto")
    normalized = sorted(set(locations))
    cache[cache_key] = tuple(normalized)
    return normalized


def _resolved_onnx_external_path(model_path, location, what):
    candidate = Path(model_path).parent
    parts = location.split("/")
    for index, part in enumerate(parts):
        candidate = candidate / part
        try:
            metadata = candidate.lstat()
        except OSError as error:
            raise ValueError("manifest %s cannot inspect %s: %s" % (
                what, candidate, error)) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("manifest %s traverses sidecar symlink %s" % (what, candidate))
        if index + 1 < len(parts) and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("manifest %s traverses non-directory %s" % (what, candidate))
        if index + 1 == len(parts) and not stat.S_ISREG(metadata.st_mode):
            raise ValueError("manifest %s does not resolve to a regular file" % what)
    return candidate.resolve(strict=True)


def validate_selected_row_inputs(manifest, category, rows, fingerprint_cache=None):
    """Prove selected semantic rows still resolve to their sealed input bytes.

    `selected_rows` deliberately stays the stable cross-language semantic-set
    contract. This companion native seal names every resolved relational-model
    member, its complete external tensor-data closure, and the VNN-LIB file.
    Legacy absence fails closed: unchanged CSV strings cannot prove unchanged
    model/property contents.
    """
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not an object")
    schema = manifest.get("selected_row_inputs_schema")
    if schema != SELECTED_ROW_INPUTS_SCHEMA:
        raise ValueError(
            "manifest selected_row_inputs_schema is %r, expected %r"
            % (schema, SELECTED_ROW_INPUTS_SCHEMA)
        )
    by_category = manifest.get("selected_row_inputs")
    if not isinstance(by_category, dict):
        raise ValueError("manifest selected_row_inputs is missing or is not an object")
    sealed_rows = by_category.get(category)
    if not isinstance(sealed_rows, list):
        raise ValueError(
            "manifest selected_row_inputs has no row list for category %r" % category
        )

    expected_keys = []
    for row in rows:
        key = (
            row.get("onnx"), row.get("vnnlib"), row.get("occurrence"),
            row.get("instance_index"),
        )
        if (not all(isinstance(value, str) and value for value in key[:2])
                or not all(isinstance(value, int) and not isinstance(value, bool)
                           and value >= 0 for value in key[2:])):
            raise ValueError("metadata row has an invalid selected-input identity")
        expected_keys.append(key)
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("metadata bank repeats a selected-input row identity")

    normalized = []
    found_keys = []
    for index, seal in enumerate(sealed_rows):
        what = "selected_row_inputs[%r][%d]" % (category, index)
        if not isinstance(seal, dict):
            raise ValueError("manifest %s is not an object" % what)
        key = (
            seal.get("onnx"), seal.get("vnnlib"), seal.get("occurrence"),
            seal.get("instance_index"),
        )
        if (not all(isinstance(value, str) and value for value in key[:2])
                or not all(isinstance(value, int) and not isinstance(value, bool)
                           and value >= 0 for value in key[2:])):
            raise ValueError("manifest %s has an invalid row identity" % what)
        found_keys.append(key)

        inputs = seal.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("manifest %s.inputs is not an object" % what)
        models = inputs.get("onnx")
        if not isinstance(models, list) or not models:
            raise ValueError("manifest %s.inputs.onnx is not a non-empty list" % what)
        normalized_models = []
        labels = []
        for model_index, model in enumerate(models):
            model_what = "%s.inputs.onnx[%d]" % (what, model_index)
            if not isinstance(model, dict):
                raise ValueError("manifest %s is not an object" % model_what)
            label = model.get("label")
            if label is not None and (not isinstance(label, str) or not label):
                raise ValueError("manifest %s has an invalid label" % model_what)
            labels.append(label)
            model_artifact = _validated_current_artifact(
                model.get("artifact"), "%s.artifact" % model_what,
                fingerprint_cache,
            )
            expected_locations = _current_onnx_external_locations(
                model_artifact, fingerprint_cache)
            dependencies = model.get("external_data")
            if not isinstance(dependencies, list):
                raise ValueError(
                    "manifest %s.external_data is not an affirmative list" % model_what)
            normalized_dependencies = []
            found_locations = []
            for dependency_index, dependency in enumerate(dependencies):
                dependency_what = "%s.external_data[%d]" % (
                    model_what, dependency_index)
                if not isinstance(dependency, dict):
                    raise ValueError("manifest %s is not an object" % dependency_what)
                location = dependency.get("location")
                if not isinstance(location, str) or not location:
                    raise ValueError("manifest %s has an invalid location" % dependency_what)
                if _normalize_onnx_external_location(location, dependency_what) != location:
                    raise ValueError(
                        "manifest %s location is not canonical" % dependency_what)
                found_locations.append(location)
                resolved_dependency = _resolved_onnx_external_path(
                    model_artifact["canonical_path"], location, dependency_what)
                dependency_artifact = _validated_current_artifact(
                    dependency.get("artifact"), "%s.artifact" % dependency_what,
                    fingerprint_cache,
                )
                if dependency_artifact["canonical_path"] != str(resolved_dependency):
                    raise ValueError(
                        "manifest %s artifact path %r does not resolve from model %r"
                        % (dependency_what, dependency_artifact["canonical_path"],
                           model_artifact["canonical_path"])
                    )
                normalized_dependencies.append({
                    "location": location,
                    "artifact": dependency_artifact,
                })
            if found_locations != sorted(set(found_locations)):
                raise ValueError(
                    "manifest %s external_data is not sorted and duplicate-free" % model_what)
            if found_locations != expected_locations:
                raise ValueError(
                    "manifest %s external-data locations %r do not equal model closure %r"
                    % (model_what, found_locations, expected_locations)
                )
            normalized_model = {
                "artifact": model_artifact,
                "external_data": normalized_dependencies,
            }
            if label is not None:
                normalized_model["label"] = label
            normalized_models.append(normalized_model)
        if len(set(labels)) != len(labels):
            raise ValueError("manifest %s repeats a relational-model label" % what)
        if len(models) > 1 and any(label is None for label in labels):
            raise ValueError(
                "manifest %s has an unlabeled member in a relational model" % what
            )

        normalized.append({
            "onnx": key[0],
            "vnnlib": key[1],
            "occurrence": key[2],
            "instance_index": key[3],
            "inputs": {
                "onnx": normalized_models,
                "vnnlib": _validated_current_artifact(
                    inputs.get("vnnlib"), "%s.inputs.vnnlib" % what,
                    fingerprint_cache,
                ),
            },
        })

    if len(set(found_keys)) != len(found_keys):
        raise ValueError("manifest selected_row_inputs repeats a row identity")
    if sorted(found_keys) != sorted(expected_keys):
        raise ValueError(
            "manifest selected-input rows %r do not equal metadata rows %r"
            % (sorted(found_keys), sorted(expected_keys))
        )
    normalized.sort(
        key=lambda seal: (
            seal["instance_index"], seal["occurrence"], seal["onnx"], seal["vnnlib"]
        )
    )
    return normalized


def requested_instance_selection(selection, category, limit, global_intent=False):
    """Normalized request identity, globally or for one native child."""
    if selection is not None:
        identity = (selection["identity"] if global_intent
                    else selection["category_identities"][category])
        return {"mode": "exact", "identity": identity}
    if limit is not None:
        return {"mode": "limit", "per_category": limit}
    return {"mode": "all"}


def validate_instance_selection(selection, category, limit, rows, manifest):
    """Prove request mode and the native planned row set equal the actual bank."""
    failures = []
    row_identity = None
    if rows:
        try:
            row_identity = measured_row_set_identity(rows)
        except ValueError as error:
            failures.append({"check": "measured_row_set_identity", "detail": str(error)})

    if selection is not None:
        expected_rows = set(selection.get("by_category", {}).get(category, ()))
        actual_rows = set()
        malformed = False
        for row in rows:
            identity = (row.get("category"), row.get("onnx"), row.get("vnnlib"))
            if any(not isinstance(field, str) or not field for field in identity):
                malformed = True
                continue
            actual_rows.add(identity)
        if malformed or actual_rows != expected_rows:
            missing = sorted(expected_rows - actual_rows)
            extra = sorted(actual_rows - expected_rows)
            failures.append({
                "check": "instance_subset_matches_request",
                "detail": (
                    "metadata rows do not exactly realize --instances for %s "
                    "(missing=%r, extra=%r%s)"
                    % (category, missing[:10], extra[:10],
                       ", malformed row identity present" if malformed else "")
                ),
            })

    wanted = requested_instance_selection(selection, category, limit)
    try:
        native = _native_instance_selection(manifest)
    except ValueError as error:
        failures.append({"check": "manifest_subset_matches_request", "detail": str(error)})
    else:
        if native != wanted:
            failures.append({
                "check": "manifest_subset_matches_request",
                "detail": (
                    "manifest selection %r != requested selection %r" % (native, wanted)
                ),
            })

    try:
        planned_rows = _native_selected_rows(manifest, category)
    except ValueError as error:
        failures.append({"check": "manifest_selected_rows_match_bank", "detail": str(error)})
    else:
        if row_identity is None or planned_rows != row_identity:
            failures.append({
                "check": "manifest_selected_rows_match_bank",
                "detail": "manifest planned row set %r != measured bank %r"
                          % (planned_rows, row_identity),
            })
    return row_identity, failures


def probe_run_flags(ny_bin, keep_ambient=False):
    """Which flags of `benchmarks run` this binary actually has."""
    try:
        probe_env, _scrubbed = child_env(keep_ambient)
        done = subprocess.run(
            [str(ny_bin), "benchmarks", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
            env=probe_env,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    text = done.stdout + done.stderr
    return {flag for flag in ("--lever", "--cache", "--configs-dir", "--timeout-cap", "--limit",
                             "--instances", "--dry-run", "--vnnlib-version", "--overwrite",
                             "--json")
            if flag in text}


def child_env(keep_ambient=False, source_env=None):
    """The child environment, with every `NY_*` and Rayon override scrubbed.

    A lever exported into the calling shell is inherited by every child and used
    to be INVISIBLE in the artifact. Scrubbing is not enough on its own — T1
    re-checks the manifest's sealed `ambient_env` — but it stops the harness
    from being the contamination it is trying to detect. `NY_BIN` is scrubbed
    too: it is not a lever, but `ambient_env_from` captures every `NY_*`, so
    leaving it set would trip the check for a real reason. Both
    `RAYON_NUM_THREADS` and deprecated `RAYON_RS_NUM_CPUS` are equally material:
    rayon-core 1.13 still uses the latter when the former is absent, changing
    the worker count behind the contextual `NY_MARGIN_ROW_PARALLEL` default.
    `source_env` is an injected test seam; production always copies `os.environ`.
    """
    env = dict(os.environ if source_env is None else source_env)
    if keep_ambient:
        return env, []
    scrubbed = sorted(
        name for name in env
        if name.startswith("NY_") or name in RAYON_WORKER_ENV
    )
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
    selector_files = getattr(opts, "instances_by_category", {}) or {}
    selector_file = selector_files.get(category)
    if selector_file is None and getattr(opts, "instances", None):
        # This fallback keeps the helper useful for a direct, single-category
        # call. Main materializes category projections before running because
        # native `--instances` rejects selectors owned by another invocation's
        # category.
        selector_file = str(opts.instances)
    if selector_file is not None:
        if opts.limit is not None:
            raise ValueError("--instances and --limit are mutually exclusive")
        if "--instances" not in flags:
            raise ValueError("selected ny lacks `benchmarks run --instances`")
        cmd += ["--instances", str(selector_file)]
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


def preflight_instance_selection(ny_bin, selection, opts, flags, runner=subprocess.run):
    """Ask the native planner to resolve every exact selector before measurement.

    Parsing the CSV proves only its syntax. Native dry-run is the authoritative
    corpus join: it catches a typoed category/model/property without producing a
    bank or spending an instance budget. The injected runner keeps this contract
    self-testable without a benchmark binary.
    """
    if selection is None:
        return
    if "--instances" not in flags:
        raise ValueError("selected ny lacks `benchmarks run --instances`")
    if "--dry-run" not in flags:
        raise ValueError("selected ny lacks `benchmarks run --dry-run`")

    with tempfile.TemporaryDirectory(prefix="ny-search-selector-preflight-") as tmp:
        root = Path(tmp)
        probe_opts = argparse.Namespace(**vars(opts))
        probe_opts.instances_by_category = materialize_instance_selections(selection, root)
        # Planning must neither consult nor populate the verdict cache.
        probe_opts.cache = "off"
        env, _scrubbed = child_env(getattr(opts, "keep_ambient", False))
        for category in opts.categories:
            output = root / ("%s.csv" % hashlib.sha256(
                category.encode("utf-8")).hexdigest()[:16])
            command = build_command(ny_bin, [], category, output, probe_opts, flags)
            command.append("--dry-run")
            try:
                done = runner(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env,
                    cwd=str(REPO_ROOT),
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise ValueError("native selector preflight failed for %s: %s"
                                 % (category, error)) from error
            if done.returncode != 0:
                detail = ((done.stderr or "") + "\n" + (done.stdout or "")).strip()
                if len(detail) > 1200:
                    detail = detail[-1200:]
                raise ValueError(
                    "native selector preflight rejected %s%s"
                    % (category, ": " + detail if detail else "")
                )


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
    """`(status, receipt, entries)` from the row's embedded flight record."""
    flight = row.get("flight")
    if not isinstance(flight, dict):
        return "absent", None, []
    levers = flight.get("levers")
    if not isinstance(levers, dict):
        return "absent", None, []
    status = levers.get("status", "absent")
    receipt = levers.get("receipt")
    entries = receipt.get("levers", []) if isinstance(receipt, dict) else []
    return status, receipt, entries if isinstance(entries, list) else []


def _receipt_value_matches_token(value, token):
    """Whether one typed receipt value is the accepted form of an env token."""
    if isinstance(value, bool):
        return token == ("1" if value else "0")
    if isinstance(value, int):
        try:
            return int(token.strip(), 10) == value
        except ValueError:
            return False
    if isinstance(value, float):
        try:
            parsed = float(token.strip())
        except ValueError:
            return False
        return math.isfinite(parsed) and parsed == value
    return isinstance(value, str) and value == token


def _receipt_v3_errors(receipt, wanted):
    """Structural and arm-delivery errors in the current layered receipt."""
    if not isinstance(receipt, dict):
        return ["resolved state has no receipt object"]
    errors = []
    if receipt.get("schema") != "ny-levers/receipt/v3":
        errors.append(
            "schema %r != 'ny-levers/receipt/v3'" % receipt.get("schema")
        )
    entries = receipt.get("levers")
    if not isinstance(entries, list):
        return errors + ["levers is not an array"]
    if not entries:
        errors.append("lever array is empty")

    count_fields = (
        "lever_count", "env_present", "env_accepted", "env_rejected", "env_ignored",
    )
    for field in count_fields:
        value = receipt.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append("%s is not a nonnegative integer" % field)
    if isinstance(receipt.get("lever_count"), int) and not isinstance(
        receipt.get("lever_count"), bool
    ) and receipt["lever_count"] != len(entries):
        errors.append(
            "lever_count=%d but the array has %d entries"
            % (receipt["lever_count"], len(entries))
        )

    by_name = {}
    accepted = rejected = ignored = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append("lever entry %d is not an object" % index)
            continue
        name = entry.get("name")
        source = entry.get("source")
        if not isinstance(name, str) or not re.fullmatch(r"NY_[A-Z0-9_]+", name):
            errors.append("lever entry %d has invalid name %r" % (index, name))
            continue
        if name in by_name:
            errors.append("lever %s appears more than once" % name)
        else:
            by_name[name] = entry
        if source not in {
            "default", "config", "legacy_env", "legacy_env_rejected",
        }:
            errors.append("%s has unknown resolution source %r" % (name, source))
        if source == "legacy_env":
            accepted += 1
        if source == "legacy_env_rejected":
            rejected += 1
            if "rejected_raw" not in entry:
                errors.append("%s is legacy_env_rejected without rejected_raw" % name)
        elif "rejected_raw" in entry:
            errors.append("%s has rejected_raw with source %r" % (name, source))
        if "ignored_raw" in entry:
            ignored += 1
            if source in ("legacy_env", "legacy_env_rejected"):
                errors.append("%s has ignored_raw with environment source %r" % (name, source))
        if "ignored_raw" in entry and "rejected_raw" in entry:
            errors.append("%s has both ignored_raw and rejected_raw" % name)

    actual_counts = {
        "env_accepted": accepted,
        "env_rejected": rejected,
        "env_ignored": ignored,
        "env_present": accepted + rejected + ignored,
    }
    for field, actual in actual_counts.items():
        declared = receipt.get(field)
        if isinstance(declared, int) and not isinstance(declared, bool) and declared != actual:
            errors.append("%s=%d but entries imply %d" % (field, declared, actual))

    for name, token in wanted:
        entry = by_name.get(name)
        if entry is None:
            errors.append("requested %s=%s is absent from the receipt" % (name, token))
        elif entry.get("source") != "legacy_env":
            errors.append(
                "requested %s=%s resolved from %r, not legacy_env"
                % (name, token, entry.get("source"))
            )
        elif not _receipt_value_matches_token(entry.get("value"), token):
            errors.append(
                "requested %s=%s receipted typed value %r"
                % (name, token, entry.get("value"))
            )
    return errors


def _row_name(row):
    return "%s/%s/%s" % (row.get("category", "?"), row.get("onnx", "?"), row.get("vnnlib", "?"))


def fingerprint_file(path):
    """Content identity of one selected executable."""
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("selected ny binary is not a regular file: %s" % resolved)
    before = resolved.stat()
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    after = resolved.stat()
    before_identity = (
        before.st_dev, before.st_ino, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    )
    if before_identity != after_identity or Path(path).resolve(strict=True) != resolved:
        raise ValueError("selected file changed while it was being fingerprinted: %s" % resolved)
    return {"canonical_path": str(resolved), "sha256": digest.hexdigest()}


def probe_binary_identity(ny_bin, timeout=30):
    """Bind a path hash to the binary's readable `NY_BUILD_ID_V1` identity.

    `--build-id` hashes the executable image opened by the child. Re-hashing
    the selected canonical path after the child exits closes replacement during
    the probe: disagreement is a hard error, never an identity guess.
    """
    resolved = Path(ny_bin).resolve(strict=True)
    probe_env, _scrubbed = child_env(False)
    try:
        done = subprocess.run(
            [str(resolved), "--build-id"], capture_output=True, text=True,
            timeout=timeout, env=probe_env,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("cannot read selected ny binary identity: %s" % error) from error
    line = done.stdout.strip()
    if done.returncode != 0 or "\n" in line or not line.startswith("NY_BUILD_ID_V1 "):
        raise ValueError(
            "selected ny binary did not emit one canonical NY_BUILD_ID_V1 line"
        )
    fields = {
        key: value
        for token in line.split()[1:]
        if "=" in token
        for key, value in [token.split("=", 1)]
    }
    embedded_sha = fields.get("exe")
    if not isinstance(embedded_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", embedded_sha):
        raise ValueError("selected ny build identity has no canonical executable digest")
    fingerprint = fingerprint_file(resolved)
    if fingerprint["sha256"] != embedded_sha:
        raise ValueError(
            "selected ny binary changed while its identity was being probed "
            "(%s != %s)" % (fingerprint["sha256"], embedded_sha)
        )
    fingerprint["build_id"] = line
    return fingerprint


def fingerprint_tree(path):
    """Mirror Rust `vnncomp_sweep::fingerprint_tree` for T1 config identity."""
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("configuration path is not a directory: %s" % root)
    files = []

    def collect(directory):
        # Rust sorts each directory's immediate entries, then recursively walks
        # a directory before visiting its next sibling. A global file sort is
        # observably different for `a/inside` beside `a.x`.
        for entry in sorted(directory.iterdir()):
            if entry.is_symlink():
                raise ValueError("configuration tree contains a symlink: %s" % entry)
            if entry.is_dir():
                collect(entry)
            elif entry.is_file():
                files.append(entry)
            else:
                raise ValueError("configuration tree contains a non-file entry: %s" % entry)

    collect(root)
    digest = hashlib.sha256()
    for entry in files:
        relative = str(entry.relative_to(root)).encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        size = entry.stat().st_size
        digest.update(size.to_bytes(8, "little"))
        with open(entry, "rb") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return {"canonical_path": str(root), "sha256": digest.hexdigest()}


def _same_resolved_path(left, right):
    if not isinstance(left, str) or not left:
        return False
    try:
        return Path(left).resolve(strict=True) == Path(right).resolve(strict=True)
    except (OSError, RuntimeError):
        return False


def _flight_base_preset_path(flight):
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return None
    for event in events:
        if (isinstance(event, dict) and event.get("method") == "preset_base"
                and event.get("status") == "ran" and isinstance(event.get("reason"), str)):
            return event["reason"]
    return None


_RUST_DURATION_MAX_NS = ((1 << 64) - 1) * 1_000_000_000 + 999_999_999


def _rust_duration_as_secs_f64(duration_ns):
    """Exact operation order of Rust `Duration::as_secs_f64`."""
    if not isinstance(duration_ns, int) or not 0 <= duration_ns <= _RUST_DURATION_MAX_NS:
        raise ValueError("nanoseconds are outside Rust Duration")
    seconds, nanoseconds = divmod(duration_ns, 1_000_000_000)
    return float(seconds) + float(nanoseconds) / 1_000_000_000.0


def _rust_duration_from_secs_f64_ns(seconds):
    """Rust `Duration::try_from_secs_f64`, returned as exact nanoseconds."""
    if not isinstance(seconds, float):
        seconds = float(seconds)
    if not math.isfinite(seconds) or seconds < 0.0:
        return None
    numerator, denominator = seconds.as_integer_ratio()
    quotient, remainder = divmod(numerator * 1_000_000_000, denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient % 2 == 1):
        quotient += 1
    return quotient if quotient <= _RUST_DURATION_MAX_NS else None


def _rust_duration_mul_f64_ns(duration_ns, multiplier):
    """Exact observable semantics of Rust `Duration::mul_f64`."""
    try:
        product = float(multiplier) * _rust_duration_as_secs_f64(duration_ns)
    except (TypeError, ValueError, OverflowError):
        return None
    return _rust_duration_from_secs_f64_ns(product)


def _rust_duration_format_ns(duration_ns):
    """Render the producer's `format!(\"{:.6}\", duration.as_secs_f64())`."""
    try:
        return f"{_rust_duration_as_secs_f64(duration_ns):.6f}"
    except ValueError:
        return None


def _fixed_six_duration_center_ns(token):
    """The exact decimal nanosecond center of a canonical six-place receipt."""
    if not isinstance(token, str) or re.fullmatch(
        r"(?:0|[1-9][0-9]*)\.[0-9]{6}", token
    ) is None:
        return None
    seconds, fraction = token.split(".", 1)
    return int(seconds) * 1_000_000_000 + int(fraction) * 1_000


def _rust_duration_receipt_candidates(token):
    """Nanosecond Durations that can produce one six-decimal receipt.

    Competition budgets are ordinary VNN-COMP deadlines, where one receipt bin
    spans about 1,000 source nanoseconds. At astronomically large seconds an
    f64 bin can be much wider; reject those fail-closed instead of letting an
    untrusted artifact trigger an unbounded enumeration.
    """
    center = _fixed_six_duration_center_ns(token)
    if center is None:
        return []
    try:
        receipt_ulp_ns = math.ceil(math.ulp(float(token)) * 1_000_000_000.0)
    except (OverflowError, ValueError):
        return []
    radius = max(2_000, receipt_ulp_ns * 2 + 2_000)
    if radius > 100_000:
        return []
    lower = max(0, center - radius)
    upper = min(_RUST_DURATION_MAX_NS, center + radius)
    return [
        nanoseconds
        for nanoseconds in range(lower, upper + 1)
        if _rust_duration_format_ns(nanoseconds) == token
    ]


def _machine_event_fields(row, method, schema):
    """Strict key/value records for one versioned flight-event method."""
    route_values = {
        "auto_disjunction", "preset_relational_conjunction",
        "disabled_by_environment", "disabled_by_preset",
        "disabled_by_invalid_preset", "forced_by_environment", "forced_by_preset",
    }
    policy_v2 = schema == "ny-cli/upfront-policy/v2"
    execution_v2 = schema == "ny-cli/upfront-execution/v2"
    if method == "upfront_attack_policy" and schema in {
        "ny-cli/upfront-policy/v1", "ny-cli/upfront-policy/v2",
    }:
        fields = {
            "schema", "master", "route", "lower_master", "lower_route",
            "wall_cap_secs", "auto_wall_cap_secs", "budget_fraction",
            "probe_first", "lower_probe_first",
        }
        if policy_v2:
            fields |= {"bnn_wall_cap_secs", "bnn_budget_fraction"}
        contract = {"fields": fields, "status": {"ran"}}
    elif method == "upfront_attack_execution" and schema in {
        "ny-cli/upfront-execution/v1", "ny-cli/upfront-execution/v2",
    }:
        fields = {
            "schema", "consultation", "route", "admission", "sign_net",
            "requested_window", "requested_cap_secs", "budget_authority",
            "binding", "remaining_secs", "effective_budget_secs",
            "inactive_reason",
        }
        if execution_v2:
            fields.add("budget_competition")
        contract = {"fields": fields, "status": {"ran", "skipped"}}
    else:
        return []

    def finite_number(token, minimum=0.0, maximum=None):
        try:
            value = float(token)
        except (TypeError, ValueError):
            return False
        return (
            math.isfinite(value)
            and value >= minimum
            and (maximum is None or value <= maximum)
        )

    def canonical_u64(token):
        return (
            isinstance(token, str)
            and re.fullmatch(r"0|[1-9][0-9]*", token) is not None
            and int(token) <= (1 << 64) - 1
        )

    def contract_valid(fields, status):
        if set(fields) != contract["fields"] or status not in contract["status"]:
            return False
        if method == "upfront_attack_policy":
            masters = {
                "contextual", "disabled_by_environment", "forced_by_environment",
                "disabled_by_preset", "forced_by_preset",
            }
            valid = (
                fields["master"] in masters
                and fields["lower_master"] in masters
                and fields["route"] in route_values
                and fields["lower_route"] in route_values
                and canonical_u64(fields["wall_cap_secs"])
                and canonical_u64(fields["auto_wall_cap_secs"])
                and finite_number(fields["budget_fraction"], minimum=0.0, maximum=1.0)
                and float(fields["budget_fraction"]) > 0.0
                and fields["probe_first"] in {"true", "false"}
                and fields["lower_probe_first"] in {"true", "false"}
            )
            route_by_master = {
                "contextual": {
                    "auto_disjunction", "preset_relational_conjunction",
                    "disabled_by_preset", "disabled_by_invalid_preset",
                },
                "disabled_by_environment": {
                    "disabled_by_environment", "disabled_by_invalid_preset",
                },
                "forced_by_environment": {
                    "forced_by_environment", "disabled_by_invalid_preset",
                },
                "disabled_by_preset": {
                    "disabled_by_preset", "disabled_by_invalid_preset",
                },
                "forced_by_preset": {
                    "forced_by_preset", "disabled_by_invalid_preset",
                },
            }
            valid = (
                valid
                and fields["route"] in route_by_master.get(fields["master"], set())
                and fields["lower_master"]
                in {"contextual", "disabled_by_preset", "forced_by_preset"}
                and fields["lower_route"]
                in route_by_master.get(fields["lower_master"], set())
            )
            if policy_v2:
                valid = (
                    valid
                    and canonical_u64(fields["bnn_wall_cap_secs"])
                    and finite_number(
                        fields["bnn_budget_fraction"], minimum=0.0, maximum=1.0
                    )
                    and float(fields["bnn_budget_fraction"]) > 0.0
                )
            return valid

        binding_parts = fields["binding"].split("+")
        binding_values = {
            "none", "budget_fraction", "wall_cap", "auto_wall_cap",
            "probe_cap", "committed_cap", "live_remaining", "allocator_zero",
            "frame_zero",
        }
        if execution_v2:
            binding_values |= {"bnn_budget_fraction", "bnn_wall_cap"}
        else:
            binding_values.add("bnn_policy")
        inactive_values = {
            "none", "probe_disarmed", "allocator_skip", "frame_skip",
            "wrapper_disabled",
            "vnnlib_load_failed", "deadline_elapsed", "admission_declined",
            "committed_budget_unusable", "private_live_budget_unusable",
            "private_base_below_min", "private_effective_below_min",
            "invalid_effective_deadline", "effective_deadline_elapsed",
            "unsupported_search_box", "trusted_forward_unavailable",
            "setup_exhausted_budget",
        }
        max_duration_secs = float(1 << 64)
        serialized_duration_or = lambda token, sentinel: (
            token == sentinel
            or (
                isinstance(token, str)
                and re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]{6}", token)
                is not None
                and finite_number(
                    token, minimum=0.0, maximum=max_duration_secs
                )
            )
        )
        status_consistent = (
            status == "ran" and fields["inactive_reason"] == "none"
        ) or (
            status == "skipped" and fields["inactive_reason"] != "none"
        )
        window_consistent = (
            fields["requested_window"] == "cap"
        ) == (fields["requested_cap_secs"] != "none")
        authority_consistent = (
            fields["budget_authority"] == "private"
            and fields["consultation"] == "ordinary"
            and fields["requested_window"] == "private"
        ) or (
            fields["budget_authority"] == "probe_cap"
            and fields["consultation"] == "probe_first"
            and fields["requested_window"] == "cap"
        ) or (
            fields["budget_authority"] == "committed_cap"
            and fields["consultation"] == "ordinary"
            and fields["requested_window"] in {"cap", "skip"}
        ) or fields["budget_authority"] == "none"
        admission_route_consistent = (
            fields["admission"] == "none"
            or (
                fields["admission"] == "forced"
                and fields["route"]
                in {"forced_by_environment", "forced_by_preset"}
            )
            or (
                # #ml4acopf-upfront-single: the typed preset force keeps a
                # multi-clause disjunction on its automatic admission (and
                # automatic cap); only the environment force re-admits it as
                # `forced`.
                fields["admission"] == "auto_disjunction"
                and fields["route"]
                in {
                    "auto_disjunction",
                    "preset_relational_conjunction",
                    "forced_by_preset",
                }
            )
            or (
                fields["admission"] == "preset_relational_conjunction"
                and fields["route"] == "preset_relational_conjunction"
            )
        )
        zero_parts = set(binding_parts) & {"allocator_zero", "frame_zero"}
        zero_reason = fields["inactive_reason"] in {"allocator_skip", "frame_skip"}
        zero_consistent = not (zero_parts or zero_reason) or (
            len(zero_parts) == 1
            and len(binding_parts) == 1
            and (
                ("allocator_zero" in zero_parts
                 and fields["inactive_reason"] == "allocator_skip")
                or ("frame_zero" in zero_parts
                    and fields["inactive_reason"] == "frame_skip")
            )
            and (
                fields["consultation"] == "ordinary"
                and fields["requested_window"] == "skip"
                and fields["budget_authority"] == "committed_cap"
                and fields["effective_budget_secs"] == "none"
                and status == "skipped"
            )
        )
        competition_consistent = True
        binding_family_consistent = True
        source_binding_consistent = True
        budget_stage_consistent = True
        route_stage_consistent = True
        cap_clamp_consistent = True
        if execution_v2:
            competition = fields["budget_competition"]
            execution_enabled_routes = {
                "auto_disjunction", "preset_relational_conjunction",
                "forced_by_environment", "forced_by_preset",
            }
            competition_consistent = (
                competition == "not_evaluated"
                or (competition == "not_sign" and fields["sign_net"] == "false")
                or (
                    competition in {"ordinary_wins", "bnn_wins", "tie"}
                    and fields["sign_net"] == "true"
                )
            )
            if (
                fields["consultation"] == "probe_first"
                or fields["budget_authority"] == "none"
                or zero_parts
                or zero_reason
            ):
                competition_consistent = (
                    competition_consistent and competition == "not_evaluated"
                )
            if status == "ran" and fields["budget_authority"] == "private":
                competition_consistent = (
                    competition_consistent and competition != "not_evaluated"
                )
            if competition != "not_evaluated":
                competition_consistent = (
                    competition_consistent
                    and fields["route"] in execution_enabled_routes
                    and fields["admission"] != "none"
                )
            ordinary_parts = set(binding_parts) & {"budget_fraction", "wall_cap"}
            bnn_parts = set(binding_parts) & {
                "bnn_budget_fraction", "bnn_wall_cap",
            }
            auto_parts = set(binding_parts) & {"auto_wall_cap"}
            parts = set(binding_parts)
            numeric_parts = ordinary_parts | bnn_parts | auto_parts
            binding_family_consistent = {
                "not_evaluated": not bnn_parts,
                "not_sign": bool(ordinary_parts or auto_parts) and not bnn_parts,
                "ordinary_wins": (
                    bool(ordinary_parts) and not bnn_parts and not auto_parts
                ),
                "bnn_wins": bool(bnn_parts) and not ordinary_parts and not auto_parts,
                "tie": bool(ordinary_parts) and bool(bnn_parts) and not auto_parts,
            }.get(competition, False)
            authority = fields["budget_authority"]
            if authority == "none":
                source_binding_consistent = (
                    parts == {"none"} and competition == "not_evaluated"
                )
            elif authority == "probe_cap":
                source_binding_consistent = (
                    competition == "not_evaluated"
                    and parts
                    in (
                        {"probe_cap"},
                        {"live_remaining"},
                        {"probe_cap", "live_remaining"},
                    )
                )
            elif authority == "private":
                source_binding_consistent = (
                    (parts == {"live_remaining"} and competition == "not_evaluated")
                    or (
                        bool(numeric_parts)
                        and parts == numeric_parts
                    )
                )
            elif authority == "committed_cap":
                source_binding_consistent = (
                    bool(zero_parts)
                    or (parts == {"live_remaining"} and competition == "not_evaluated")
                    or (
                        competition == "not_evaluated"
                        and parts
                        in (
                            {"committed_cap"},
                            {"committed_cap", "live_remaining"},
                        )
                    )
                    or (
                        bool(numeric_parts)
                        and (
                            parts == numeric_parts
                            or parts == numeric_parts | {"live_remaining"}
                        )
                    )
                )
            else:
                source_binding_consistent = False
            reason = fields["inactive_reason"]
            admission_known = fields["admission"] != "none"
            sign_known = fields["sign_net"] in {"true", "false"}
            effective_numeric = finite_number(
                fields["effective_budget_secs"], minimum=0.0
            )
            pre_admission_reasons = {
                "wrapper_disabled", "vnnlib_load_failed", "deadline_elapsed",
                "admission_declined",
            }
            post_budget_reasons = {
                "none", "invalid_effective_deadline",
                "effective_deadline_elapsed", "unsupported_search_box",
                "trusted_forward_unavailable", "setup_exhausted_budget",
            }
            disabled_routes = {
                "disabled_by_environment", "disabled_by_preset",
                "disabled_by_invalid_preset",
            }
            if reason == "wrapper_disabled":
                route_stage_consistent = fields["route"] in disabled_routes
            elif reason in {"vnnlib_load_failed", "deadline_elapsed"}:
                route_stage_consistent = fields["route"] in execution_enabled_routes
            elif reason == "admission_declined":
                route_stage_consistent = fields["route"] in {
                    "auto_disjunction", "preset_relational_conjunction",
                }
            if reason == "probe_disarmed":
                budget_stage_consistent = (
                    status == "skipped"
                    and fields["consultation"] == "probe_first"
                    and fields["requested_window"] == "skip"
                    and fields["admission"] == "none"
                    and fields["sign_net"] == "unknown"
                    and authority == "none"
                    and parts == {"none"}
                    and competition == "not_evaluated"
                    and fields["effective_budget_secs"] == "none"
                )
            elif reason in {"allocator_skip", "frame_skip"}:
                budget_stage_consistent = (
                    zero_consistent
                    and fields["admission"] == "none"
                    and fields["sign_net"] == "unknown"
                    and competition == "not_evaluated"
                )
            elif reason in pre_admission_reasons:
                budget_stage_consistent = (
                    status == "skipped"
                    and fields["admission"] == "none"
                    and fields["sign_net"] == "unknown"
                    and authority == "none"
                    and parts == {"none"}
                    and competition == "not_evaluated"
                    and fields["effective_budget_secs"] == "none"
                )
            elif reason == "committed_budget_unusable":
                budget_stage_consistent = (
                    status == "skipped"
                    and admission_known
                    and sign_known
                    and fields["requested_window"] == "cap"
                    and authority in {"probe_cap", "committed_cap"}
                    and parts == {"live_remaining"}
                    and competition == "not_evaluated"
                    and fields["effective_budget_secs"] == "none"
                )
            elif reason == "private_live_budget_unusable":
                budget_stage_consistent = (
                    status == "skipped"
                    and admission_known
                    and sign_known
                    and fields["consultation"] == "ordinary"
                    and fields["requested_window"] == "private"
                    and authority == "private"
                    and parts == {"live_remaining"}
                    and competition == "not_evaluated"
                    and fields["effective_budget_secs"] == "none"
                    and fields["remaining_secs"] != "unbounded"
                    and finite_number(fields["remaining_secs"], minimum=0.0)
                    and float(fields["remaining_secs"]) <= 3.000000
                )
            elif reason == "private_base_below_min":
                budget_stage_consistent = (
                    status == "skipped"
                    and admission_known
                    and sign_known
                    and authority == "private"
                    and fields["requested_window"] == "private"
                    and competition == "not_evaluated"
                    and bool(ordinary_parts)
                    and parts == ordinary_parts
                    and effective_numeric
                    and float(fields["effective_budget_secs"]) <= 0.8
                )
            elif reason == "private_effective_below_min":
                budget_stage_consistent = (
                    status == "skipped"
                    and admission_known
                    and authority == "private"
                    and fields["requested_window"] == "private"
                    and competition == "not_sign"
                    and fields["sign_net"] == "false"
                    and fields["admission"] == "auto_disjunction"
                    and parts == {"auto_wall_cap"}
                    and effective_numeric
                    and float(fields["effective_budget_secs"]) <= 0.8
                )
            elif reason in post_budget_reasons:
                budget_stage_consistent = (
                    admission_known
                    and sign_known
                    and authority != "none"
                    and fields["requested_window"] != "skip"
                    and effective_numeric
                    and (
                        float(fields["effective_budget_secs"]) > 0.0
                        or (
                            status == "skipped"
                            and authority in {"probe_cap", "committed_cap"}
                        )
                    )
                )
            else:
                budget_stage_consistent = False
            if (
                authority in {"probe_cap", "committed_cap"}
                and fields["requested_window"] == "cap"
            ):
                if authority == "probe_cap":
                    cap_candidates = [5_000_000_000]
                    cap_clamp_consistent = (
                        fields["requested_cap_secs"] == "5.000000"
                    )
                else:
                    cap_candidates = [
                        cap_ns
                        for cap_ns in _rust_duration_receipt_candidates(
                            fields["requested_cap_secs"]
                        )
                        if cap_ns >= 800_000_000
                    ]
                    cap_clamp_consistent = bool(cap_candidates)
                source = parts - {"live_remaining"}
                live = "live_remaining" in parts
                if authority == "probe_cap" and source and source != {"probe_cap"}:
                    cap_clamp_consistent = False
                remaining_token = fields["remaining_secs"]
                if remaining_token == "unbounded":
                    cap_clamp_consistent = (
                        cap_clamp_consistent
                        and reason in post_budget_reasons
                        and not live
                        and bool(source)
                        and any(
                            _rust_duration_format_ns(cap_ns)
                                == fields["effective_budget_secs"]
                            for cap_ns in cap_candidates
                        )
                    )
                else:
                    remaining_candidates = _rust_duration_receipt_candidates(
                        remaining_token
                    )
                    usable_candidates = [
                        remaining_ns - 3_000_000_000
                        for remaining_ns in remaining_candidates
                        if remaining_ns >= 3_000_000_000
                    ]
                    if live and not source:
                        if reason == "committed_budget_unusable":
                            cap_clamp_consistent = (
                                cap_clamp_consistent
                                and competition == "not_evaluated"
                                and (
                                    any(remaining_ns < 3_000_000_000
                                        for remaining_ns in remaining_candidates)
                                    or any(
                                        usable_ns == 0 and cap_ns > 0
                                        for usable_ns in usable_candidates
                                        for cap_ns in cap_candidates
                                    )
                                )
                            )
                        else:
                            effective_usable = [
                                usable_ns
                                for usable_ns in usable_candidates
                                if usable_ns > 0
                                and _rust_duration_format_ns(usable_ns)
                                    == fields["effective_budget_secs"]
                            ]
                            cap_clamp_consistent = (
                                cap_clamp_consistent
                                and reason in post_budget_reasons
                                and competition == "not_evaluated"
                                and bool(effective_usable)
                                and max(cap_candidates) > min(effective_usable)
                            )
                    elif live and source:
                        usable_set = set(usable_candidates)
                        cap_clamp_consistent = (
                            cap_clamp_consistent
                            and reason in post_budget_reasons
                            and any(
                                cap_ns > 0
                                and cap_ns in usable_set
                                and _rust_duration_format_ns(cap_ns)
                                    == fields["effective_budget_secs"]
                                for cap_ns in cap_candidates
                            )
                        )
                    elif source:
                        effective_caps = [
                            cap_ns
                            for cap_ns in cap_candidates
                            if cap_ns > 0
                            and _rust_duration_format_ns(cap_ns)
                                == fields["effective_budget_secs"]
                        ]
                        cap_clamp_consistent = (
                            cap_clamp_consistent
                            and reason in post_budget_reasons
                            and bool(effective_caps)
                            and bool(usable_candidates)
                            and max(usable_candidates) > min(effective_caps)
                        )
                    else:
                        cap_clamp_consistent = False
        return (
            fields["consultation"] in {"probe_first", "ordinary"}
            and fields["route"] in route_values
            and fields["admission"]
            in {"none", "forced", "auto_disjunction", "preset_relational_conjunction"}
            and fields["sign_net"] in {"true", "false", "unknown"}
            and fields["requested_window"] in {"private", "skip", "cap"}
            and serialized_duration_or(fields["requested_cap_secs"], "none")
            and fields["budget_authority"]
            in {"none", "private", "probe_cap", "committed_cap"}
            and binding_parts
            and len(binding_parts) == len(set(binding_parts))
            and all(part in binding_values for part in binding_parts)
            and ("none" not in binding_parts or len(binding_parts) == 1)
            and serialized_duration_or(fields["remaining_secs"], "unbounded")
            and serialized_duration_or(fields["effective_budget_secs"], "none")
            and fields["inactive_reason"] in inactive_values
            and status_consistent
            and window_consistent
            and authority_consistent
            and admission_route_consistent
            and zero_consistent
            and competition_consistent
            and binding_family_consistent
            and source_binding_consistent
            and budget_stage_consistent
            and route_stage_consistent
            and cap_clamp_consistent
        )

    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    parsed_events = []
    for event in events:
        if not isinstance(event, dict) or event.get("method") != method:
            continue
        reason = event.get("reason")
        if not isinstance(reason, str):
            continue
        fields = {}
        malformed = False
        for token in reason.split():
            if token.count("=") != 1:
                malformed = True
                break
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                malformed = True
                break
            fields[key] = value
        status = event.get("status")
        if (
            malformed
            or fields.get("schema") != schema
            or not contract_valid(fields, status)
        ):
            continue
        fields["_status"] = status
        parsed_events.append(fields)
    return parsed_events


_MARGIN_ROW_FRONTIER_EVENT_SCHEMA = "ny-cli/margin-row-frontier-execution/v1"
_MARGIN_ROW_FRONTIER_EVENT_FIELDS = frozenset({
    "schema", "phase", "route", "outcome", "tree_started", "configured",
    "contextual", "steps", "counterfactual_differences", "max_open", "max_batch",
})
_MARGIN_ROW_FRONTIER_INTEGER_FIELDS = frozenset({
    "configured", "contextual", "steps", "counterfactual_differences",
    "max_open", "max_batch",
})


def _margin_row_frontier_contract_valid(fields, status):
    """Validate one terminal margin-row frontier receipt in isolation."""
    if (
        set(fields) != _MARGIN_ROW_FRONTIER_EVENT_FIELDS
        or fields.get("schema") != _MARGIN_ROW_FRONTIER_EVENT_SCHEMA
        or status != "ran"
        or fields["phase"] != "terminal"
        or fields["route"] not in {"joint", "classwise"}
        or fields["outcome"] not in {"unsat", "unknown"}
        or fields["tree_started"] not in {"true", "false"}
    ):
        return False

    integers = {}
    for name in _MARGIN_ROW_FRONTIER_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        integers[name] = value

    configured = integers["configured"]
    contextual = integers["contextual"]
    steps = integers["steps"]
    differences = integers["counterfactual_differences"]
    max_open = integers["max_open"]
    max_batch = integers["max_batch"]
    tree_started = fields["tree_started"] == "true"
    if configured == 0 or contextual == 0 or differences > steps:
        return False
    if steps == 0:
        return (
            not tree_started
            and differences == 0
            and max_open == 0
            and max_batch == 0
            and (
                fields["route"] != "classwise"
                or (configured == 1 and contextual == 1)
            )
        )
    if (
        not tree_started
        or max_open == 0
        or max_batch == 0
        or max_batch != min(configured, max_open)
        or (differences > 0)
        != (min(configured, max_open) != min(contextual, max_open))
    ):
        return False
    if fields["route"] == "classwise":
        return configured == 1 and contextual == 1 and differences == 0
    return True


def _margin_row_frontier_terminal_evidences(row):
    """Strict independently closed margin-row frontier invocations for one row.

    Any malformed same-method event poisons this row. Terminal-only v1 events
    need no cross-event counter folding, so a later root-only invocation cannot
    erase an earlier, independently closed engagement receipt.
    """
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if not isinstance(event, dict) or event.get("method") != "margin_row_frontier_execution":
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii()
            or not reason
            or reason != reason.strip(" ")
            or "  " in reason
            or any(character.isspace() and character != " " for character in reason)
        ):
            return []
        tokens = reason.split(" ")
        if not tokens or tokens[0] != "schema=" + _MARGIN_ROW_FRONTIER_EVENT_SCHEMA:
            return []
        fields = {}
        for token in tokens:
            if token.count("=") != 1:
                return []
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                return []
            fields[key] = value
        if not _margin_row_frontier_contract_valid(fields, event.get("status")):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    return records


def _margin_row_frontier_parameter_engagement(wanted, rows):
    """Whether the requested width changed a live joint-tree batch decision."""
    requested = dict(wanted).get("NY_MARGIN_ROW_PARALLEL")
    if requested is None:
        return set()
    expected = 1 if requested == "0" else _canonical_event_u64(requested)
    if expected is None or requested not in {"0", "2", "4", "8", "16"}:
        return set()
    for row in rows:
        for event in _margin_row_frontier_terminal_evidences(row):
            if (
                event["route"] == "joint"
                and _canonical_event_u64(event["configured"]) == expected
                and _canonical_event_u64(event["counterfactual_differences"]) > 0
            ):
                return {"NY_MARGIN_ROW_PARALLEL"}
    return set()


_MARGIN_ROW_WIDTH_FAMILY = frozenset({
    "NY_MARGIN_ROW_K_HEAD",
    "NY_MARGIN_ROW_K_TRUNK",
    "NY_MARGIN_ROW_K_ADAPT",
})
_MARGIN_ROW_WIDTH_EVENT_SCHEMA = "ny-margin-row-width-execution/v2"
_MARGIN_ROW_WIDTH_EVENT_FIELDS = frozenset({
    "schema", "phase", "outcome", "configured_k_head",
    "contextual_k_head", "configured_k_trunk", "contextual_k_trunk",
    "adaptive_enabled", "contextual_adaptive", "tier0_exact",
    "effective_k_head_max", "effective_k_trunk_max",
    "effective_contextual_k_head_max", "effective_contextual_k_trunk_max",
    "outer_head_selection_calls", "outer_head_available_max",
    "outer_head_selected_total", "outer_head_binding_scored_calls",
    "outer_trunk_selection_calls", "outer_trunk_available_max",
    "outer_trunk_selected_total", "outer_trunk_binding_scored_calls",
    "nested_head_selection_calls", "nested_trunk_selection_calls",
    "configured_adaptive_threshold_opportunities",
    "configured_adaptive_escalations",
    "contextual_adaptive_threshold_opportunities",
    "contextual_adaptive_escalations",
    "policy_mismatch",
})
_MARGIN_ROW_WIDTH_BOOLEAN_FIELDS = frozenset({
    "adaptive_enabled", "contextual_adaptive", "policy_mismatch",
})
_MARGIN_ROW_WIDTH_INTEGER_FIELDS = (
    _MARGIN_ROW_WIDTH_EVENT_FIELDS
    - _MARGIN_ROW_WIDTH_BOOLEAN_FIELDS
    - {"schema", "phase", "outcome"}
)


def _margin_row_width_contract_valid(fields, status):
    """Validate one immutable branch-width terminal in isolation."""
    if (
        set(fields) != _MARGIN_ROW_WIDTH_EVENT_FIELDS
        or fields.get("schema") != _MARGIN_ROW_WIDTH_EVENT_SCHEMA
        or status != "ran"
        or fields["phase"] != "terminal"
        or fields["outcome"] not in {"unsat", "unknown"}
        or any(
            fields[name] not in {"true", "false"}
            for name in _MARGIN_ROW_WIDTH_BOOLEAN_FIELDS
        )
    ):
        return False

    numbers = {}
    for name in _MARGIN_ROW_WIDTH_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        numbers[name] = value

    if any(
        numbers[name] == 0
        for name in {
            "configured_k_head", "contextual_k_head",
            "configured_k_trunk", "contextual_k_trunk",
        }
    ):
        return False
    if (
        numbers["configured_adaptive_escalations"]
        > numbers["configured_adaptive_threshold_opportunities"]
        or numbers["contextual_adaptive_escalations"]
        > numbers["contextual_adaptive_threshold_opportunities"]
    ):
        return False

    for route in ("head", "trunk"):
        calls = numbers[f"outer_{route}_selection_calls"]
        available = numbers[f"outer_{route}_available_max"]
        selected = numbers[f"outer_{route}_selected_total"]
        binding = numbers[f"outer_{route}_binding_scored_calls"]
        effective = numbers[f"effective_k_{route}_max"]
        configured = numbers[f"configured_k_{route}"]
        contextual = numbers[f"contextual_k_{route}"]
        contextual_effective = numbers[f"effective_contextual_k_{route}_max"]
        if (
            effective == 0
            or contextual_effective == 0
            or effective < configured
            or contextual_effective < contextual
            or binding > calls
            or selected > calls * effective
            or selected > calls * available
            or (calls == 0 and (available != 0 or selected != 0 or binding != 0))
            or (
                binding > 0
                and (
                    selected == 0
                    or (
                        configured == contextual
                        and fields["adaptive_enabled"]
                            == fields["contextual_adaptive"]
                    )
                    or available <= min(configured, contextual)
                    or (
                        configured == contextual
                        and effective == contextual_effective
                    )
                )
            )
        ):
            return False

    adaptive_sides = (
        (
            "adaptive_enabled",
            "configured_k_{}",
            "effective_k_{}_max",
            "configured_adaptive_threshold_opportunities",
            "configured_adaptive_escalations",
        ),
        (
            "contextual_adaptive",
            "contextual_k_{}",
            "effective_contextual_k_{}_max",
            "contextual_adaptive_threshold_opportunities",
            "contextual_adaptive_escalations",
        ),
    )
    for enabled_name, base_pattern, effective_pattern, opportunities, escalations in adaptive_sides:
        base_head = numbers[base_pattern.format("head")]
        base_trunk = numbers[base_pattern.format("trunk")]
        effective_head = numbers[effective_pattern.format("head")]
        effective_trunk = numbers[effective_pattern.format("trunk")]
        changed = effective_head > base_head or effective_trunk > base_trunk
        if fields[enabled_name] == "false":
            if (
                changed
                or numbers[opportunities] != 0
                or numbers[escalations] != 0
            ):
                return False
        elif changed != (numbers[escalations] > 0):
            # Tier-0 bypasses ordinary selectors but historically still
            # ratchets operational widths. A later per-domain snapshot can
            # therefore retain the changed maximum without an ordinary-route
            # escalation, including after the unsafe incremental route drops
            # retention. Tier-0 rows are rejected before any width-axis
            # engagement, so accepting this live telemetry shape cannot create
            # score credit.
            if not (
                numbers["tier0_exact"] > 0
                and changed
                and numbers[escalations] == 0
            ):
                return False
    if (
        fields["adaptive_enabled"] == "true"
        and fields["contextual_adaptive"] == "true"
        and numbers["configured_adaptive_threshold_opportunities"]
        != numbers["contextual_adaptive_threshold_opportunities"]
    ):
        return False
    return True


def _margin_row_width_terminal_evidences(row):
    """Return the row's one exact width terminal, or fail closed.

    Width policy is frozen once for an invocation. Multiple same-method
    terminals therefore make ownership ambiguous even when byte-identical;
    malformed, duplicate, and conflicting terminals all poison the row.
    """
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if not isinstance(event, dict) or event.get("method") != "margin_row_width_execution":
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii()
            or not reason
            or reason != reason.strip(" ")
            or "  " in reason
            or any(character.isspace() and character != " " for character in reason)
        ):
            return []
        tokens = reason.split(" ")
        if not tokens or tokens[0] != "schema=" + _MARGIN_ROW_WIDTH_EVENT_SCHEMA:
            return []
        fields = {}
        for token in tokens:
            if token.count("=") != 1:
                return []
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                return []
            fields[key] = value
        if not _margin_row_width_contract_valid(fields, event.get("status")):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    return records if len(records) == 1 else []


def _margin_row_width_parameter_engagement(wanted, rows):
    """Width axes that changed an ordinary outer exact-scored selection."""
    requested = {
        name: token for name, token in wanted
        if name in _MARGIN_ROW_WIDTH_FAMILY
    }
    if not requested:
        return set()
    engaged = set()
    for row in rows:
        evidence = _margin_row_width_terminal_evidences(row)
        if len(evidence) != 1:
            continue
        event = evidence[0]
        if (
            event["policy_mismatch"] != "false"
            or _canonical_event_u64(event["tier0_exact"]) != 0
        ):
            continue
        fixed_requested = {
            name: token for name, token in requested.items()
            if name != "NY_MARGIN_ROW_K_ADAPT"
        }
        fixed_policy = (
            event["adaptive_enabled"] == "false"
            and event["contextual_adaptive"] == "false"
            and all(
                _canonical_event_u64(event[f"effective_k_{route}_max"])
                    == _canonical_event_u64(event[f"configured_k_{route}"])
                and _canonical_event_u64(
                    event[f"effective_contextual_k_{route}_max"]
                ) == _canonical_event_u64(event[f"contextual_k_{route}"])
                for route in ("head", "trunk")
            )
        )
        for name, token in fixed_requested.items():
            configured = _canonical_event_u64(token)
            route = "head" if name == "NY_MARGIN_ROW_K_HEAD" else "trunk"
            if (
                fixed_policy
                and configured is not None
                and configured > 0
                and _canonical_event_u64(event[f"configured_k_{route}"])
                    == configured
                and _canonical_event_u64(event[f"contextual_k_{route}"])
                    != configured
                and _canonical_event_u64(event[f"outer_{route}_selection_calls"])
                    > 0
                and _canonical_event_u64(
                    event[f"outer_{route}_binding_scored_calls"]
                ) > 0
            ):
                engaged.add(name)

        adaptive_token = requested.get("NY_MARGIN_ROW_K_ADAPT")
        if adaptive_token not in {"0", "1"}:
            continue
        configured_adaptive = event["adaptive_enabled"] == "true"
        contextual_adaptive = event["contextual_adaptive"] == "true"
        requested_adaptive = adaptive_token == "1"
        bases_match = all(
            _canonical_event_u64(event[f"configured_k_{route}"])
            == _canonical_event_u64(event[f"contextual_k_{route}"])
            for route in ("head", "trunk")
        )
        binding_scored = any(
            _canonical_event_u64(
                event[f"outer_{route}_binding_scored_calls"]
            ) > 0
            for route in ("head", "trunk")
        )
        if (
            configured_adaptive != requested_adaptive
            or contextual_adaptive == requested_adaptive
            or not bases_match
            or not binding_scored
        ):
            continue
        configured_changed = any(
            _canonical_event_u64(event[f"effective_k_{route}_max"])
            > _canonical_event_u64(event[f"configured_k_{route}"])
            for route in ("head", "trunk")
        )
        contextual_changed = any(
            _canonical_event_u64(
                event[f"effective_contextual_k_{route}_max"]
            ) > _canonical_event_u64(event[f"contextual_k_{route}"])
            for route in ("head", "trunk")
        )
        configured_escalated = (
            _canonical_event_u64(
                event["configured_adaptive_threshold_opportunities"]
            ) > 0
            and _canonical_event_u64(
                event["configured_adaptive_escalations"]
            ) > 0
            and configured_changed
        )
        contextual_escalated = (
            _canonical_event_u64(
                event["contextual_adaptive_threshold_opportunities"]
            ) > 0
            and _canonical_event_u64(
                event["contextual_adaptive_escalations"]
            ) > 0
            and contextual_changed
        )
        if (
            requested_adaptive
            and configured_escalated
            and not contextual_escalated
            and not contextual_changed
        ) or (
            not requested_adaptive
            and contextual_escalated
            and not configured_escalated
            and not configured_changed
        ):
            engaged.add("NY_MARGIN_ROW_K_ADAPT")
    return engaged


_MARGIN_ROW_SCHEDULE_FAMILY = frozenset({"NY_MARGIN_ROW_CONCURRENT"})
_MARGIN_ROW_SCHEDULE_CONFIGURED_FAMILY = frozenset({
    "NY_MARGIN_ROW_RESERVE_SECS",
    "NY_MARGIN_ROW_ADAPTIVE_RESERVE",
    "NY_MARGIN_ROW_RESERVE_MAX_FRAC",
    "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC",
    "NY_MARGIN_ROW_CONCURRENT",
    "NY_MARGIN_ROW_EARLY_EXIT",
    "NY_VERIFIER_THREADS",
    "NY_MARGIN_ROW_CONCURRENT_THREADS",
})
_MARGIN_ROW_SCHEDULE_EVENT_SCHEMA = (
    "ny-cli/margin-row-schedule-execution/v2"
)
_MARGIN_ROW_SCHEDULE_EVENT_FIELD_ORDER = (
    "schema",
    "phase",
    "outer_route",
    "configured_reserve_secs",
    "contextual_reserve_secs",
    "configured_adaptive_reserve",
    "contextual_adaptive_reserve",
    "configured_reserve_max_frac_bits",
    "contextual_reserve_max_frac_bits",
    "configured_release_frac_bits",
    "contextual_release_frac_bits",
    "configured_concurrent",
    "contextual_concurrent",
    "configured_early_exit",
    "contextual_early_exit",
    "configured_verifier_threads",
    "contextual_verifier_threads",
    "configured_concurrent_threads",
    "contextual_concurrent_threads",
    "host_parallelism",
    "global_pool_route",
    "global_pool_requested_threads",
    "global_pool_actual_threads",
    "lane_pool_route",
    "lane_pool_threads",
    "private_pool_route",
    "concurrent_route",
    "concurrent_spawned",
    "internal_before_secs",
    "reserve_route",
    "uncapped_reserve_secs",
    "capped_reserve_secs",
    "held_reserve_secs",
    "internal_after_secs",
    "sparse_root_requested",
    "policy_mismatch",
)
_MARGIN_ROW_SCHEDULE_EVENT_FIELDS = frozenset(
    _MARGIN_ROW_SCHEDULE_EVENT_FIELD_ORDER
)
_MARGIN_ROW_SCHEDULE_BOOLEAN_FIELDS = frozenset({
    "configured_adaptive_reserve",
    "contextual_adaptive_reserve",
    "configured_concurrent",
    "contextual_concurrent",
    "configured_early_exit",
    "contextual_early_exit",
    "concurrent_spawned",
    "sparse_root_requested",
    "policy_mismatch",
})
_MARGIN_ROW_SCHEDULE_INTEGER_FIELDS = frozenset({
    "configured_reserve_secs",
    "contextual_reserve_secs",
    "internal_before_secs",
    "uncapped_reserve_secs",
    "capped_reserve_secs",
    "held_reserve_secs",
    "internal_after_secs",
})
_MARGIN_ROW_SCHEDULE_OPTIONAL_FRACTION_BITS_FIELDS = frozenset({
    "configured_reserve_max_frac_bits",
    "contextual_reserve_max_frac_bits",
})
_MARGIN_ROW_SCHEDULE_FRACTION_BITS_FIELDS = frozenset({
    "configured_release_frac_bits",
    "contextual_release_frac_bits",
})
_MARGIN_ROW_SCHEDULE_OPTIONAL_USIZE_FIELDS = frozenset({
    "configured_verifier_threads",
    "contextual_verifier_threads",
    "configured_concurrent_threads",
    "contextual_concurrent_threads",
    "host_parallelism",
    "global_pool_requested_threads",
    "global_pool_actual_threads",
    "lane_pool_threads",
})
_MARGIN_ROW_SCHEDULE_OUTER_ROUTES = frozenset({
    "actionable", "proof_tail", "deferred_pgd", "memory",
})
_MARGIN_ROW_SCHEDULE_CONCURRENT_ROUTES = frozenset({
    "outer_suppressed", "disabled", "authority_disabled", "cpu_device",
    "no_deadline", "short_budget", "model", "property", "input_shape",
    "no_safe_remainder", "global_unavailable", "private_pool_failed",
    "spawn_failed", "spawned",
})
_MARGIN_ROW_SCHEDULE_RESERVE_ROUTES = frozenset({
    "not_applicable", "fixed", "adaptive_preserved", "adaptive_released",
})
_MARGIN_ROW_SCHEDULE_GLOBAL_POOL_ROUTES = frozenset({
    "not_requested", "applied", "already_initialized",
})
_MARGIN_ROW_SCHEDULE_LANE_POOL_ROUTES = frozenset({
    "not_consulted", "shared_global", "explicit_private",
    "verifier_remainder", "no_safe_remainder", "global_unavailable",
})
_MARGIN_ROW_SCHEDULE_PRIVATE_POOL_ROUTES = frozenset({
    "not_requested", "built", "build_failed",
})
_MARGIN_ROW_CONCURRENT_OVERLAP_EVENT_SCHEMA = (
    "ny-cli/margin-row-concurrent-overlap-execution/v1"
)
_MARGIN_ROW_CONCURRENT_OVERLAP_EVENT_FIELD_ORDER = (
    "schema", "phase", "worker_active", "verifier_active",
)


def _strict_ordered_event_fields(event, method, schema, field_order):
    """Parse one producer-owned ASCII ``key=value`` reason without aliases."""
    if not isinstance(event, dict) or event.get("method") != method:
        return None
    reason = event.get("reason")
    if (
        not isinstance(reason, str)
        or not reason.isascii()
        or not reason
        or reason != reason.strip(" ")
        or "  " in reason
        or any(
            character.isspace() and character != " "
            for character in reason
        )
    ):
        return None
    tokens = reason.split(" ")
    if len(tokens) != len(field_order):
        return None
    fields = {}
    for token, expected_name in zip(tokens, field_order):
        if token.count("=") != 1:
            return None
        name, value = token.split("=", 1)
        if name != expected_name or not value or name in fields:
            return None
        fields[name] = value
    if fields.get("schema") != schema:
        return None
    return fields


def _canonical_event_usize(token):
    """Return one canonical native ``usize`` event token, or ``None``."""
    value = _canonical_event_u64(token)
    if value is None or value > (sys.maxsize * 2) + 1:
        return None
    return value


def _optional_event_usize(token):
    """Parse producer-owned ``none|usize`` without conflating zero and none."""
    if token == "none":
        return None, True
    value = _canonical_event_usize(token)
    return value, value is not None


def _margin_row_schedule_contract_valid(fields, status):
    """Validate one immutable allocator receipt and its exact arithmetic."""
    if (
        set(fields) != _MARGIN_ROW_SCHEDULE_EVENT_FIELDS
        or fields.get("schema") != _MARGIN_ROW_SCHEDULE_EVENT_SCHEMA
        or status != "ran"
        or fields["phase"] != "allocation"
        or fields["outer_route"] not in _MARGIN_ROW_SCHEDULE_OUTER_ROUTES
        or fields["concurrent_route"]
        not in _MARGIN_ROW_SCHEDULE_CONCURRENT_ROUTES
        or fields["reserve_route"] not in _MARGIN_ROW_SCHEDULE_RESERVE_ROUTES
        or fields["global_pool_route"]
        not in _MARGIN_ROW_SCHEDULE_GLOBAL_POOL_ROUTES
        or fields["lane_pool_route"]
        not in _MARGIN_ROW_SCHEDULE_LANE_POOL_ROUTES
        or fields["private_pool_route"]
        not in _MARGIN_ROW_SCHEDULE_PRIVATE_POOL_ROUTES
        or any(
            fields[name] not in {"true", "false"}
            for name in _MARGIN_ROW_SCHEDULE_BOOLEAN_FIELDS
        )
        or fields["policy_mismatch"] != "false"
    ):
        return False

    numbers = {}
    for name in _MARGIN_ROW_SCHEDULE_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        numbers[name] = value

    fractions = {}
    for name in _MARGIN_ROW_SCHEDULE_FRACTION_BITS_FIELDS:
        value = _event_f64_from_bits(fields[name], positive=True)
        if value is None or value >= 1.0:
            return False
        fractions[name] = value
    optional_fractions = {}
    for name in _MARGIN_ROW_SCHEDULE_OPTIONAL_FRACTION_BITS_FIELDS:
        value, valid = _optional_event_f64_from_bits(fields[name])
        if not valid or (value is not None and not 0.0 < value < 1.0):
            return False
        optional_fractions[name] = value

    optional_usizes = {}
    for name in _MARGIN_ROW_SCHEDULE_OPTIONAL_USIZE_FIELDS:
        value, valid = _optional_event_usize(fields[name])
        if not valid:
            return False
        optional_usizes[name] = value
    if optional_usizes["host_parallelism"] == 0:
        return False
    for name in (
        "global_pool_requested_threads", "global_pool_actual_threads",
    ):
        if optional_usizes[name] == 0:
            return False

    configured_secs = numbers["configured_reserve_secs"]
    before = numbers["internal_before_secs"]
    uncapped = numbers["uncapped_reserve_secs"]
    capped = numbers["capped_reserve_secs"]
    held = numbers["held_reserve_secs"]
    after = numbers["internal_after_secs"]
    outer_route = fields["outer_route"]
    concurrent_route = fields["concurrent_route"]
    reserve_route = fields["reserve_route"]
    adaptive = fields["configured_adaptive_reserve"] == "true"
    configured_verifier_threads = optional_usizes[
        "configured_verifier_threads"
    ]
    configured_concurrent_threads = optional_usizes[
        "configured_concurrent_threads"
    ]
    host_parallelism = optional_usizes["host_parallelism"]
    global_pool_route = fields["global_pool_route"]
    global_pool_requested_threads = optional_usizes[
        "global_pool_requested_threads"
    ]
    global_pool_actual_threads = optional_usizes[
        "global_pool_actual_threads"
    ]
    lane_pool_route = fields["lane_pool_route"]
    lane_pool_threads = optional_usizes["lane_pool_threads"]
    private_pool_route = fields["private_pool_route"]

    verifier_request = (
        configured_verifier_threads
        if configured_verifier_threads not in {None, 0}
        else None
    )
    if global_pool_route == "not_requested":
        if (
            verifier_request is not None
            or global_pool_requested_threads is not None
            or global_pool_actual_threads is not None
        ):
            return False
    else:
        if (
            verifier_request is None
            or global_pool_requested_threads != verifier_request
        ):
            return False
        if global_pool_route == "applied":
            if global_pool_actual_threads != verifier_request:
                return False
        elif global_pool_route == "already_initialized":
            if global_pool_actual_threads is None:
                return False

    max_fraction = optional_fractions["configured_reserve_max_frac_bits"]
    expected_capped = uncapped
    if max_fraction is not None:
        ceiling = math.floor(float(before) * max_fraction)
        expected_capped = min(uncapped, ceiling)
    expected_after = (
        max(max(before - capped, 0), 10) if capped > 0 else before
    )
    if (
        capped != expected_capped
        or after != expected_after
        or held != max(before - after, 0)
    ):
        return False

    spawned = fields["concurrent_spawned"] == "true"
    configured_concurrent = fields["configured_concurrent"] == "true"
    if spawned != (concurrent_route == "spawned"):
        return False
    if outer_route != "actionable":
        if concurrent_route != "outer_suppressed":
            return False
    elif concurrent_route == "outer_suppressed":
        return False
    if concurrent_route == "disabled" and configured_concurrent:
        return False
    if concurrent_route in {
        "cpu_device", "no_deadline", "short_budget", "model", "property",
        "input_shape", "no_safe_remainder", "global_unavailable",
        "private_pool_failed", "spawn_failed", "spawned",
    } and not configured_concurrent:
        return False

    pre_resource_routes = {
        "outer_suppressed", "disabled", "authority_disabled", "cpu_device",
        "no_deadline", "short_budget", "model", "property", "input_shape",
    }
    if lane_pool_route == "not_consulted":
        if (
            lane_pool_threads is not None
            or private_pool_route != "not_requested"
            or concurrent_route not in pre_resource_routes
        ):
            return False
    elif lane_pool_route == "shared_global":
        if (
            lane_pool_threads != 0
            or private_pool_route != "not_requested"
            or concurrent_route not in {"spawn_failed", "spawned"}
            or not (
                configured_concurrent_threads == 0
                or (
                    configured_concurrent_threads is None
                    and global_pool_route == "not_requested"
                )
            )
        ):
            return False
    elif lane_pool_route == "explicit_private":
        if (
            configured_concurrent_threads is None
            or configured_concurrent_threads == 0
            or lane_pool_threads != configured_concurrent_threads
        ):
            return False
    elif lane_pool_route == "verifier_remainder":
        if (
            configured_concurrent_threads is not None
            or global_pool_route not in {"applied", "already_initialized"}
            or host_parallelism is None
            or global_pool_actual_threads is None
            or host_parallelism <= global_pool_actual_threads
            or lane_pool_threads
            != host_parallelism - global_pool_actual_threads
        ):
            return False
    elif lane_pool_route == "no_safe_remainder":
        if (
            configured_concurrent_threads is not None
            or global_pool_route not in {"applied", "already_initialized"}
            or host_parallelism is None
            or global_pool_actual_threads is None
            or host_parallelism > global_pool_actual_threads
            or lane_pool_threads is not None
            or private_pool_route != "not_requested"
            or concurrent_route != "no_safe_remainder"
        ):
            return False
    elif lane_pool_route == "global_unavailable":
        if (
            configured_concurrent_threads is not None
            or verifier_request is None
            or global_pool_route not in {"applied", "already_initialized"}
            or global_pool_actual_threads is None
            or host_parallelism is not None
            or lane_pool_threads is not None
            or private_pool_route != "not_requested"
            or concurrent_route != "global_unavailable"
        ):
            return False

    if lane_pool_route in {"explicit_private", "verifier_remainder"}:
        if private_pool_route == "built":
            if concurrent_route not in {"spawn_failed", "spawned"}:
                return False
        elif private_pool_route == "build_failed":
            if concurrent_route != "private_pool_failed":
                return False
        else:
            return False
    elif private_pool_route != "not_requested":
        return False

    if concurrent_route == "no_safe_remainder" and (
        lane_pool_route != "no_safe_remainder"
    ):
        return False
    if concurrent_route == "global_unavailable" and (
        lane_pool_route != "global_unavailable"
    ):
        return False
    if concurrent_route == "private_pool_failed" and (
        lane_pool_route not in {"explicit_private", "verifier_remainder"}
        or private_pool_route != "build_failed"
    ):
        return False

    if outer_route != "actionable":
        if (
            reserve_route != "not_applicable"
            or uncapped != 0
            or capped != 0
            or held != 0
        ):
            return False
    if configured_secs == 0 and reserve_route != "not_applicable":
        return False
    if reserve_route == "not_applicable":
        if uncapped != 0 or capped != 0:
            return False
    elif reserve_route == "fixed":
        if configured_secs == 0 or adaptive or uncapped != configured_secs:
            return False
    elif reserve_route == "adaptive_preserved":
        release_fraction = fractions["configured_release_frac_bits"]
        release_target = (
            before != 0
            and float(configured_secs) >= float(before) * release_fraction
        )
        if (
            configured_secs == 0
            or not adaptive
            or release_target
            or uncapped != configured_secs
        ):
            return False
    elif reserve_route == "adaptive_released":
        release_fraction = fractions["configured_release_frac_bits"]
        release_target = (
            before != 0
            and float(configured_secs) >= float(before) * release_fraction
        )
        if (
            configured_secs == 0
            or not adaptive
            or not release_target
            or uncapped != 0
            or capped != 0
            or held != 0
        ):
            return False

    sparse = fields["sparse_root_requested"] == "true"
    if sparse != (reserve_route == "adaptive_released"):
        return False
    return True


def _margin_row_schedule_terminal_evidences(row):
    """Return exactly one strict allocation receipt; ambiguity poisons a row."""
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("method") != "margin_row_schedule_execution"
        ):
            continue
        fields = _strict_ordered_event_fields(
            event,
            "margin_row_schedule_execution",
            _MARGIN_ROW_SCHEDULE_EVENT_SCHEMA,
            _MARGIN_ROW_SCHEDULE_EVENT_FIELD_ORDER,
        )
        if fields is None or not _margin_row_schedule_contract_valid(
            fields, event.get("status")
        ):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    return records if len(records) == 1 else []


def _margin_row_concurrent_overlap_evidences(row):
    """Accept the producer's one-shot worker/verifier overlap witness only."""
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("method")
            != "margin_row_concurrent_overlap_execution"
        ):
            continue
        fields = _strict_ordered_event_fields(
            event,
            "margin_row_concurrent_overlap_execution",
            _MARGIN_ROW_CONCURRENT_OVERLAP_EVENT_SCHEMA,
            _MARGIN_ROW_CONCURRENT_OVERLAP_EVENT_FIELD_ORDER,
        )
        if (
            fields is None
            or event.get("status") != "ran"
            or fields["phase"] != "work"
            or fields["worker_active"] != "true"
            or fields["verifier_active"] != "true"
        ):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    return records if len(records) == 1 else []


def _margin_row_schedule_configuration_signature(event):
    """Frozen schedule/resource identity used to reject cross-row stitching."""
    return tuple(
        event[name]
        for name in (
            "configured_reserve_secs",
            "contextual_reserve_secs",
            "configured_adaptive_reserve",
            "contextual_adaptive_reserve",
            "configured_reserve_max_frac_bits",
            "contextual_reserve_max_frac_bits",
            "configured_release_frac_bits",
            "contextual_release_frac_bits",
            "configured_concurrent",
            "contextual_concurrent",
            "configured_early_exit",
            "contextual_early_exit",
            "configured_verifier_threads",
            "contextual_verifier_threads",
            "configured_concurrent_threads",
            "contextual_concurrent_threads",
            "host_parallelism",
            "global_pool_route",
            "global_pool_requested_threads",
            "global_pool_actual_threads",
            "lane_pool_route",
            "lane_pool_threads",
            "private_pool_route",
        )
    )


def _margin_row_schedule_parameter_engagement(wanted, rows):
    """Credit only a clean, real concurrent overlap on every supplied row."""
    requested = {
        name: token for name, token in wanted
        if name in _MARGIN_ROW_SCHEDULE_FAMILY
    }
    # The off arm remains intentionally ungraduated: a refusal or absence is
    # not causal evidence that concurrency would otherwise have overlapped.
    if requested != {"NY_MARGIN_ROW_CONCURRENT": "1"} or not rows:
        return set()

    signatures = set()
    for row in rows:
        schedule = _margin_row_schedule_terminal_evidences(row)
        overlap = _margin_row_concurrent_overlap_evidences(row)
        width = _margin_row_width_terminal_evidences(row)
        if len(schedule) != 1 or len(overlap) != 1 or len(width) != 1:
            return set()
        flight = row.get("flight") if isinstance(row, dict) else None
        events = flight.get("events") if isinstance(flight, dict) else None
        if not isinstance(events, list):
            return set()
        positions = {}
        for method in (
            "margin_row_schedule_execution",
            "margin_row_concurrent_overlap_execution",
            "margin_row_width_execution",
        ):
            matches = [
                index for index, candidate in enumerate(events)
                if isinstance(candidate, dict)
                and candidate.get("method") == method
            ]
            if len(matches) != 1:
                return set()
            positions[method] = matches[0]
        if not (
            positions["margin_row_schedule_execution"]
            < positions["margin_row_concurrent_overlap_execution"]
            < positions["margin_row_width_execution"]
        ):
            return set()
        event = schedule[0]
        if (
            event["configured_concurrent"] != "true"
            or event["contextual_concurrent"] != "false"
            or event["outer_route"] != "actionable"
            or event["concurrent_route"] != "spawned"
            or event["concurrent_spawned"] != "true"
            or event["capped_reserve_secs"] != "0"
            or event["held_reserve_secs"] != "0"
            or event["internal_after_secs"] != event["internal_before_secs"]
            or event["policy_mismatch"] != "false"
            or width[0]["policy_mismatch"] != "false"
            # Once a spawned lane reaches the stats-owning width terminal, the
            # producer has passed the same model/property admission used by
            # reserve allocation. A positive configured reserve therefore
            # cannot truthfully report NotApplicable on this row.
            or (
                int(event["configured_reserve_secs"]) > 0
                and event["reserve_route"] == "not_applicable"
            )
        ):
            return set()
        signatures.add(_margin_row_schedule_configuration_signature(event))
    return (
        {"NY_MARGIN_ROW_CONCURRENT"}
        if len(signatures) == 1
        else set()
    )


_MARGIN_ROW_POLICY_FAMILY = frozenset({
    "NY_MARGIN_ROW_BETA_PERCOL",
    "NY_MARGIN_ROW_BETA_COLS",
    "NY_MARGIN_ROW_COL_RETIRE",
    "NY_MARGIN_ROW_PARALLEL_PREBUILD",
})
_MARGIN_ROW_POLICY_EVENT_SCHEMA = "ny-cli/margin-row-policy-execution/v1"
_MARGIN_ROW_POLICY_EVENT_FIELDS = frozenset({
    "schema", "phase", "outcome", "beta_per_column", "beta_columns",
    "column_retire", "parallel_prebuild", "beta_per_column_evals",
    "beta_columns_binding_evals", "beta_columns_max_candidates",
    "newly_retired_columns", "inherited_retirement_evals",
    "parallel_prebuild_batches", "parallel_prebuild_domains", "policy_mismatch",
})
_MARGIN_ROW_POLICY_INTEGER_FIELDS = frozenset({
    "beta_columns", "beta_per_column_evals", "beta_columns_binding_evals",
    "beta_columns_max_candidates", "newly_retired_columns",
    "inherited_retirement_evals", "parallel_prebuild_batches",
    "parallel_prebuild_domains",
})


def _margin_row_policy_contract_valid(fields, status):
    """Validate one immutable margin-row policy execution receipt."""
    if (
        set(fields) != _MARGIN_ROW_POLICY_EVENT_FIELDS
        or fields.get("schema") != _MARGIN_ROW_POLICY_EVENT_SCHEMA
        or status != "ran"
        or fields["phase"] != "terminal"
        or fields["outcome"] not in {"unsat", "unknown"}
        or any(fields[name] not in {"true", "false"} for name in (
            "beta_per_column", "column_retire", "parallel_prebuild",
            "policy_mismatch",
        ))
        or fields["policy_mismatch"] != "false"
    ):
        return False

    integers = {}
    for name in _MARGIN_ROW_POLICY_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        integers[name] = value
    if not 1 <= integers["beta_columns"] <= 32:
        return False

    per_column = fields["beta_per_column"] == "true"
    per_column_evals = integers["beta_per_column_evals"]
    binding_evals = integers["beta_columns_binding_evals"]
    max_candidates = integers["beta_columns_max_candidates"]
    if (
        binding_evals > per_column_evals
        or ((per_column_evals == 0) != (max_candidates == 0))
        or (
            binding_evals > 0
            and (
                (integers["beta_columns"] < 8
                 and max_candidates <= integers["beta_columns"])
                or (integers["beta_columns"] > 8 and max_candidates <= 8)
                or integers["beta_columns"] == 8
            )
        )
    ):
        return False
    if not per_column and (per_column_evals or binding_evals or max_candidates):
        return False

    retire = fields["column_retire"] == "true"
    if not retire and (
        integers["newly_retired_columns"]
        or integers["inherited_retirement_evals"]
    ):
        return False

    prebuild = fields["parallel_prebuild"] == "true"
    batches = integers["parallel_prebuild_batches"]
    domains = integers["parallel_prebuild_domains"]
    if (not prebuild and (batches or domains)) or ((batches == 0) != (domains == 0)):
        return False
    return domains >= batches


def _margin_row_policy_terminal_evidences(row):
    """Strict terminal policy receipts; one malformed family event poisons the row."""
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if not isinstance(event, dict) or event.get("method") != "margin_row_policy_execution":
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii()
            or not reason
            or reason != reason.strip(" ")
            or "  " in reason
            or any(character.isspace() and character != " " for character in reason)
        ):
            return []
        tokens = reason.split(" ")
        if not tokens or tokens[0] != "schema=" + _MARGIN_ROW_POLICY_EVENT_SCHEMA:
            return []
        fields = {}
        for token in tokens:
            if token.count("=") != 1:
                return []
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                return []
            fields[key] = value
        if not _margin_row_policy_contract_valid(fields, event.get("status")):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    return records


def _margin_row_policy_configuration_matches(event, requested):
    expected_booleans = {
        "NY_MARGIN_ROW_BETA_PERCOL": "beta_per_column",
        "NY_MARGIN_ROW_COL_RETIRE": "column_retire",
        "NY_MARGIN_ROW_PARALLEL_PREBUILD": "parallel_prebuild",
    }
    for name, field in expected_booleans.items():
        token = requested.get(name)
        if token is not None and (
            token not in {"0", "1"}
            or event[field] != ("true" if token == "1" else "false")
        ):
            return False
    columns = requested.get("NY_MARGIN_ROW_BETA_COLS")
    if columns is not None and (
        columns not in {"1", "4", "8", "16", "32"}
        or _canonical_event_u64(event["beta_columns"])
        != _canonical_event_u64(columns)
    ):
        return False
    return True


def _margin_row_policy_parameter_engagement(wanted, rows):
    """Prove live work for every requested member of the immutable policy family."""
    requested = {
        name: token for name, token in wanted if name in _MARGIN_ROW_POLICY_FAMILY
    }
    if not requested:
        return set()
    engaged = set()
    for row in rows:
        for event in _margin_row_policy_terminal_evidences(row):
            if not _margin_row_policy_configuration_matches(event, requested):
                continue
            per_column_evals = _canonical_event_u64(event["beta_per_column_evals"])
            binding_evals = _canonical_event_u64(event["beta_columns_binding_evals"])
            if (
                requested.get("NY_MARGIN_ROW_BETA_PERCOL") == "1"
                and event["beta_per_column"] == "true"
                and per_column_evals > 0
            ):
                engaged.add("NY_MARGIN_ROW_BETA_PERCOL")
            if (
                "NY_MARGIN_ROW_BETA_COLS" in requested
                and event["beta_per_column"] == "true"
                and per_column_evals > 0
                and binding_evals > 0
            ):
                engaged.add("NY_MARGIN_ROW_BETA_COLS")
            if (
                requested.get("NY_MARGIN_ROW_COL_RETIRE") == "1"
                and event["column_retire"] == "true"
                and _canonical_event_u64(event["inherited_retirement_evals"]) > 0
            ):
                engaged.add("NY_MARGIN_ROW_COL_RETIRE")
            batches = _canonical_event_u64(event["parallel_prebuild_batches"])
            domains = _canonical_event_u64(event["parallel_prebuild_domains"])
            if (
                requested.get("NY_MARGIN_ROW_PARALLEL_PREBUILD") == "1"
                and event["parallel_prebuild"] == "true"
                and batches > 0
                and domains >= 2 * batches
            ):
                engaged.add("NY_MARGIN_ROW_PARALLEL_PREBUILD")
    return engaged


_MARGIN_ROW_ROOT_PRECISION_FAMILY = frozenset({
    "NY_MARGIN_ROW_ROOT_F32",
    "NY_MARGIN_ROW_TABLEAU_F32",
})
_MARGIN_ROW_ROOT_PRECISION_EVENT_SCHEMA = (
    "ny-cli/margin-row-root-precision-execution/v1"
)
_MARGIN_ROW_ROOT_PRECISION_EVENT_FIELDS = frozenset({
    "schema", "phase", "outcome", "configured_root_f32",
    "configured_tableau_f32", "outward_eligible", "eligible_conv_ops",
    "root_f32_conv_ops", "root_f64_conv_ops", "eligible_store_ops",
    "tableau_f32_store_ops", "tableau_f64_store_ops", "tableau_refused",
    "policy_mismatch",
})
_MARGIN_ROW_ROOT_PRECISION_BOOLEAN_FIELDS = frozenset({
    "configured_root_f32", "configured_tableau_f32", "outward_eligible",
    "tableau_refused", "policy_mismatch",
})
_MARGIN_ROW_ROOT_PRECISION_INTEGER_FIELDS = frozenset({
    "eligible_conv_ops", "root_f32_conv_ops", "root_f64_conv_ops",
    "eligible_store_ops", "tableau_f32_store_ops", "tableau_f64_store_ops",
})


def _margin_row_root_precision_contract_valid(fields, status):
    """Validate one complete root precision execution receipt."""
    if (
        set(fields) != _MARGIN_ROW_ROOT_PRECISION_EVENT_FIELDS
        or fields.get("schema") != _MARGIN_ROW_ROOT_PRECISION_EVENT_SCHEMA
        or status != "ran"
        or fields["phase"] != "terminal"
        or fields["outcome"] != "completed"
        or any(
            fields[name] not in {"true", "false"}
            for name in _MARGIN_ROW_ROOT_PRECISION_BOOLEAN_FIELDS
        )
        or fields["policy_mismatch"] != "false"
    ):
        return False

    counters = {}
    for name in _MARGIN_ROW_ROOT_PRECISION_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        counters[name] = value
    eligible_conv = counters["eligible_conv_ops"]
    eligible_store = counters["eligible_store_ops"]
    if (
        counters["root_f32_conv_ops"] + counters["root_f64_conv_ops"]
        != eligible_conv
        or counters["tableau_f32_store_ops"]
        + counters["tableau_f64_store_ops"]
        != eligible_store
    ):
        return False

    configured_root = fields["configured_root_f32"] == "true"
    configured_tableau = fields["configured_tableau_f32"] == "true"
    outward = fields["outward_eligible"] == "true"
    root_f32_effective = configured_root and outward
    tableau_f32_effective = configured_tableau and root_f32_effective
    if (
        (fields["tableau_refused"] == "true")
        != (configured_tableau and not root_f32_effective)
    ):
        return False
    if not outward:
        return all(value == 0 for value in counters.values())
    if root_f32_effective:
        if counters["root_f32_conv_ops"] != eligible_conv:
            return False
    elif counters["root_f64_conv_ops"] != eligible_conv:
        return False
    if tableau_f32_effective:
        return counters["tableau_f32_store_ops"] == eligible_store
    return counters["tableau_f64_store_ops"] == eligible_store


def _margin_row_root_precision_terminal_evidences(row):
    """Strict terminals; malformed or mixed frozen policies poison one row."""
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("method") != "margin_row_root_precision_execution"
        ):
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii()
            or not reason
            or reason != reason.strip(" ")
            or "  " in reason
            or any(
                character.isspace() and character != " "
                for character in reason
            )
        ):
            return []
        tokens = reason.split(" ")
        if (
            not tokens
            or tokens[0]
            != "schema=" + _MARGIN_ROW_ROOT_PRECISION_EVENT_SCHEMA
        ):
            return []
        fields = {}
        for token in tokens:
            if token.count("=") != 1:
                return []
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                return []
            fields[key] = value
        if not _margin_row_root_precision_contract_valid(
            fields, event.get("status")
        ):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    configurations = {
        (fields["configured_root_f32"], fields["configured_tableau_f32"])
        for fields in records
    }
    if len(configurations) > 1:
        return []
    return records


def _margin_row_root_precision_configuration_matches(event, requested):
    fields = {
        "NY_MARGIN_ROW_ROOT_F32": "configured_root_f32",
        "NY_MARGIN_ROW_TABLEAU_F32": "configured_tableau_f32",
    }
    for name, field in fields.items():
        token = requested.get(name)
        if token is not None and (
            token not in {"0", "1"}
            or event[field] != ("true" if token == "1" else "false")
        ):
            return False
    return True


def _margin_row_root_precision_parameter_engagement(wanted, rows):
    """Prove effective compute/storage work for each precision treatment."""
    requested = {
        name: token
        for name, token in wanted
        if name in _MARGIN_ROW_ROOT_PRECISION_FAMILY
    }
    if not requested:
        return set()
    engaged = set()
    for row in rows:
        for event in _margin_row_root_precision_terminal_evidences(row):
            if not _margin_row_root_precision_configuration_matches(
                event, requested
            ):
                continue
            outward = event["outward_eligible"] == "true"
            eligible_conv = _canonical_event_u64(event["eligible_conv_ops"])
            eligible_store = _canonical_event_u64(event["eligible_store_ops"])
            assert eligible_conv is not None and eligible_store is not None
            root = requested.get("NY_MARGIN_ROW_ROOT_F32")
            if outward and eligible_conv > 0 and (
                (
                    root == "1"
                    and event["root_f32_conv_ops"] == event["eligible_conv_ops"]
                )
                or (
                    root == "0"
                    and event["root_f64_conv_ops"] == event["eligible_conv_ops"]
                )
            ):
                engaged.add("NY_MARGIN_ROW_ROOT_F32")
            tableau = requested.get("NY_MARGIN_ROW_TABLEAU_F32")
            root_effective = (
                outward and event["configured_root_f32"] == "true"
            )
            if root_effective and eligible_store > 0 and (
                (
                    tableau == "1"
                    and event["tableau_refused"] == "false"
                    and event["tableau_f32_store_ops"]
                    == event["eligible_store_ops"]
                )
                or (
                    tableau == "0"
                    and event["tableau_f64_store_ops"]
                    == event["eligible_store_ops"]
                )
            ):
                engaged.add("NY_MARGIN_ROW_TABLEAU_F32")
    return engaged


_MARGIN_ROW_DOMAIN_ALPHA_FAMILY = frozenset({
    "NY_MARGIN_ROW_DOMAIN_ALPHA",
    "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS",
    "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA",
})
_MARGIN_ROW_DOMAIN_ALPHA_EVENT_SCHEMA = (
    "ny-cli/margin-row-domain-alpha-execution/v1"
)
_MARGIN_ROW_DOMAIN_ALPHA_EVENT_FIELDS = frozenset({
    "schema", "phase", "route", "outcome", "configured_enabled",
    "configured_iters", "configured_eta_bits", "policy_mismatch",
    "gate_evals", "movable_domains", "movable_slots",
    "warm_entries_applied", "ascent_domains", "outer_ascent_domains",
    "proposals", "trial_passes_completed", "trial_pass_failures",
    "finite_trial_scores", "accepted_trials", "carried_entries",
    "iters_counterfactual_domains", "eta_counterfactual_scored_domains",
    "outer_finite_trial_scores", "outer_iters_counterfactual_domains",
    "outer_eta_counterfactual_scored_domains",
})
_MARGIN_ROW_DOMAIN_ALPHA_BOOLEAN_FIELDS = frozenset({
    "configured_enabled", "policy_mismatch",
})
_MARGIN_ROW_DOMAIN_ALPHA_INTEGER_FIELDS = (
    _MARGIN_ROW_DOMAIN_ALPHA_EVENT_FIELDS
    - _MARGIN_ROW_DOMAIN_ALPHA_BOOLEAN_FIELDS
    - {"schema", "phase", "route", "outcome"}
)
_MARGIN_ROW_DOMAIN_ALPHA_WORK_FIELDS = (
    _MARGIN_ROW_DOMAIN_ALPHA_INTEGER_FIELDS
    - {"configured_iters", "configured_eta_bits"}
)
_MARGIN_ROW_DOMAIN_ALPHA_DEFAULT_ETA_BITS = str(
    struct.unpack(">Q", struct.pack(">d", 0.25))[0]
)


def _margin_row_domain_alpha_contract_valid(fields, status):
    """Validate one frozen domain-alpha terminal and causal algebra."""
    if (
        set(fields) != _MARGIN_ROW_DOMAIN_ALPHA_EVENT_FIELDS
        or fields.get("schema") != _MARGIN_ROW_DOMAIN_ALPHA_EVENT_SCHEMA
        or status != "ran"
        or fields["phase"] != "terminal"
        or fields["route"] not in {"joint", "classwise"}
        or fields["outcome"] not in {"unsat", "unknown"}
        or any(
            fields[name] not in {"true", "false"}
            for name in _MARGIN_ROW_DOMAIN_ALPHA_BOOLEAN_FIELDS
        )
        or fields["policy_mismatch"] != "false"
    ):
        return False

    numbers = {}
    for name in _MARGIN_ROW_DOMAIN_ALPHA_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        numbers[name] = value
    value = lambda name: numbers[name]
    eta = _event_f64_from_bits(fields["configured_eta_bits"], positive=True)
    if (
        not 1 <= value("configured_iters") <= 8
        or eta is None
        or not 0.01 <= eta <= 1.0
        or value("ascent_domains") > value("movable_domains")
        or value("movable_domains") > value("gate_evals")
        or value("movable_domains") > value("movable_slots")
        or value("warm_entries_applied") > value("movable_slots")
        or value("carried_entries") > value("movable_slots")
        or value("outer_ascent_domains") > value("ascent_domains")
        or value("ascent_domains") > value("proposals")
        or value("proposals")
        > value("ascent_domains") * value("configured_iters")
        or value("proposals")
        != value("trial_passes_completed") + value("trial_pass_failures")
        or value("trial_pass_failures") > value("ascent_domains")
        or value("finite_trial_scores") > value("trial_passes_completed")
        or value("accepted_trials") > value("finite_trial_scores")
        or value("outer_finite_trial_scores") > value("finite_trial_scores")
        or value("outer_finite_trial_scores")
        > value("outer_ascent_domains") * value("configured_iters")
        or value("iters_counterfactual_domains") > value("ascent_domains")
        or value("eta_counterfactual_scored_domains")
        > value("ascent_domains")
        or value("iters_counterfactual_domains")
        > value("finite_trial_scores")
        or value("eta_counterfactual_scored_domains")
        > value("finite_trial_scores")
        or value("outer_iters_counterfactual_domains")
        > value("outer_ascent_domains")
        or value("outer_eta_counterfactual_scored_domains")
        > value("outer_ascent_domains")
        or value("outer_iters_counterfactual_domains")
        > value("outer_finite_trial_scores")
        or value("outer_eta_counterfactual_scored_domains")
        > value("outer_finite_trial_scores")
        or value("outer_iters_counterfactual_domains")
        > value("iters_counterfactual_domains")
        or value("outer_eta_counterfactual_scored_domains")
        > value("eta_counterfactual_scored_domains")
        or (
            value("configured_iters") > 2
            and (
                value("proposals")
                < 3 * value("iters_counterfactual_domains")
                or value("proposals")
                < value("ascent_domains")
                + 2 * value("iters_counterfactual_domains")
                or value("trial_passes_completed")
                < 3 * value("iters_counterfactual_domains")
                or value("finite_trial_scores")
                - value("outer_finite_trial_scores")
                > value("trial_passes_completed")
                - 3 * value("outer_iters_counterfactual_domains")
            )
        )
        or (
            fields["configured_enabled"] == "false"
            and any(value(name) != 0 for name in _MARGIN_ROW_DOMAIN_ALPHA_WORK_FIELDS)
        )
        or (
            value("configured_iters") == 2
            and any(value(name) != 0 for name in (
                "iters_counterfactual_domains",
                "outer_iters_counterfactual_domains",
            ))
        )
        or (
            fields["configured_eta_bits"]
            == _MARGIN_ROW_DOMAIN_ALPHA_DEFAULT_ETA_BITS
            and any(value(name) != 0 for name in (
                "eta_counterfactual_scored_domains",
                "outer_eta_counterfactual_scored_domains",
            ))
        )
    ):
        return False
    return True


def _margin_row_domain_alpha_terminal_evidences(row):
    """Accept exactly one strict terminal; duplicates poison the row."""
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("method") != "margin_row_domain_alpha_execution"
        ):
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii() or not reason
            or reason != reason.strip(" ") or "  " in reason
            or any(
                character.isspace() and character != " "
                for character in reason
            )
        ):
            return []
        tokens = reason.split(" ")
        if (
            not tokens
            or tokens[0] != "schema=" + _MARGIN_ROW_DOMAIN_ALPHA_EVENT_SCHEMA
        ):
            return []
        fields = {}
        for token in tokens:
            if token.count("=") != 1:
                return []
            key, token_value = token.split("=", 1)
            if not key or not token_value or key in fields:
                return []
            fields[key] = token_value
        if not _margin_row_domain_alpha_contract_valid(
            fields, event.get("status")
        ):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    return records if len(records) == 1 else []


def _margin_row_domain_alpha_configuration_matches(event, requested):
    enabled = requested.get("NY_MARGIN_ROW_DOMAIN_ALPHA")
    if enabled is not None and (
        enabled not in {"0", "1"}
        or event.get("configured_enabled")
        != ("true" if enabled == "1" else "false")
    ):
        return False
    iters = requested.get("NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS")
    if iters is not None and (
        _canonical_event_u64(iters) is None
        or event.get("configured_iters") != iters
    ):
        return False
    eta = requested.get("NY_MARGIN_ROW_DOMAIN_ALPHA_ETA")
    if eta is not None:
        if not isinstance(eta, str) or _ASCII_DECIMAL_RE.fullmatch(eta) is None:
            return False
        try:
            parsed_eta = float(eta)
            expected_bits = str(
                struct.unpack(">Q", struct.pack(">d", parsed_eta))[0]
            )
        except (OverflowError, struct.error, ValueError):
            return False
        if (
            not math.isfinite(parsed_eta)
            or parsed_eta <= 0.0
            or event.get("configured_eta_bits") != expected_bits
        ):
            return False
    return True


def _margin_row_domain_alpha_parameter_engagement(wanted, rows):
    """Credit only outer joint scoring under the exact frozen policy."""
    requested = {
        name: token for name, token in wanted
        if name in _MARGIN_ROW_DOMAIN_ALPHA_FAMILY
    }
    if not requested:
        return set()
    engaged = set()
    for row in rows:
        for event in _margin_row_domain_alpha_terminal_evidences(row):
            if (
                event["route"] != "joint"
                or event["policy_mismatch"] != "false"
                or not _margin_row_domain_alpha_configuration_matches(
                    event, requested
                )
            ):
                continue
            number = lambda name: _canonical_event_u64(event[name])
            enabled = event["configured_enabled"] == "true"
            if (
                requested.get("NY_MARGIN_ROW_DOMAIN_ALPHA") == "1"
                and enabled
                and number("outer_finite_trial_scores") > 0
            ):
                engaged.add("NY_MARGIN_ROW_DOMAIN_ALPHA")
            if (
                "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS" in requested
                and enabled
                and number("outer_iters_counterfactual_domains") > 0
            ):
                engaged.add("NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS")
            if (
                "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA" in requested
                and enabled
                and number("outer_eta_counterfactual_scored_domains") > 0
            ):
                engaged.add("NY_MARGIN_ROW_DOMAIN_ALPHA_ETA")
    return engaged


_MARGIN_ROW_ROOT_ALPHA_FAMILY = frozenset({
    "NY_MARGIN_ROW_ALPHA_OPT",
    "NY_MARGIN_ROW_ALPHA_ITERS",
    "NY_MARGIN_ROW_ALPHA_SECS",
    "NY_MARGIN_ROW_ALPHA_LR",
    "NY_MARGIN_ROW_ALPHA_COLS",
    "NY_MARGIN_ROW_ALPHA_GUARD",
    "NY_MARGIN_ROW_REBAKE",
    "NY_MARGIN_ROW_REBAKE_ROUNDS",
    "NY_MARGIN_ROW_REBAKE_SECS",
})
_MARGIN_ROW_ROOT_ALPHA_EVENT_SCHEMA = (
    "ny-cli/margin-row-root-alpha-execution/v1"
)
_MARGIN_ROW_ROOT_ALPHA_EVENT_FIELDS = frozenset({
    "schema", "phase", "outcome", "alpha_enabled", "alpha_iters",
    "alpha_secs_bits", "alpha_lr_bits", "alpha_cols", "alpha_guard",
    "rebake_enabled", "rebake_rounds", "rebake_secs_bits", "eligible",
    "standalone_suppressed", "standalone_attempted", "standalone_completed",
    "standalone_effective_secs_bits", "standalone_deadline_capped",
    "standalone_budget_exhausted", "standalone_iters_cap_exhausted",
    "standalone_fail_classes", "standalone_iters_attempted",
    "standalone_candidate_steps", "standalone_legacy_steps",
    "standalone_adam_steps", "standalone_passes",
    "standalone_max_steer_cols", "standalone_frozen_improvements",
    "standalone_accepted", "standalone_moved_neurons",
    "standalone_guard_opportunities", "standalone_guard_baseline_evals",
    "standalone_guard_candidate_evals", "standalone_guard_reversions",
    "standalone_frozen_before_bits", "standalone_frozen_best_bits",
    "standalone_published_before_bits", "standalone_published_after_bits",
    "rebake_suppressed", "rebake_attempted", "rebake_completed",
    "rebake_effective_secs_bits", "rebake_deadline_capped",
    "rebake_budget_exhausted", "rebake_round_cap_exhausted",
    "rebake_entry_unbaked_alpha", "rebake_baseline_evals",
    "rebake_rounds_started", "rebake_alpha_completed",
    "rebake_alpha_iters_attempted", "rebake_alpha_candidate_steps",
    "rebake_alpha_legacy_steps", "rebake_alpha_adam_steps",
    "rebake_alpha_passes", "rebake_alpha_max_fail_classes",
    "rebake_alpha_max_steer_cols", "rebake_alpha_moved_neurons",
    "rebake_rebuilds_completed", "rebake_rounds_scored",
    "rebake_rounds_accepted", "rebake_fixed_point_stops",
    "rebake_affordability_refusals", "rebake_aborts",
    "rebake_published_before_bits", "rebake_published_after_bits",
    "policy_mismatch",
})
_MARGIN_ROW_ROOT_ALPHA_BOOLEAN_FIELDS = frozenset({
    "alpha_enabled", "alpha_guard", "rebake_enabled", "eligible",
    "standalone_deadline_capped", "standalone_budget_exhausted",
    "standalone_iters_cap_exhausted", "standalone_accepted",
    "rebake_deadline_capped", "rebake_budget_exhausted",
    "rebake_round_cap_exhausted", "rebake_entry_unbaked_alpha",
    "policy_mismatch",
})
_MARGIN_ROW_ROOT_ALPHA_OPTIONAL_BITS_FIELDS = frozenset({
    "standalone_frozen_before_bits", "standalone_frozen_best_bits",
    "standalone_published_before_bits", "standalone_published_after_bits",
    "rebake_published_before_bits", "rebake_published_after_bits",
})
_MARGIN_ROW_ROOT_ALPHA_INTEGER_FIELDS = (
    _MARGIN_ROW_ROOT_ALPHA_EVENT_FIELDS
    - _MARGIN_ROW_ROOT_ALPHA_BOOLEAN_FIELDS
    - _MARGIN_ROW_ROOT_ALPHA_OPTIONAL_BITS_FIELDS
    - {"schema", "phase", "outcome"}
)


_MARGIN_ROW_EPOCH_T1_FAMILY = frozenset({
    "NY_EPOCH_DEPTH",
    "NY_EPOCH_ATTEMPTS",
    "NY_EPOCH_TIER0",
    "NY_EPOCH_UNIVERSE",
    "NY_EPOCH_RETAIN",
})
_MARGIN_ROW_EPOCH_CONFIGURED_FAMILY = (
    _MARGIN_ROW_EPOCH_T1_FAMILY
    | frozenset({"NY_MARGIN_ROW_EPOCH_PARENT_TOPK"})
)
_MARGIN_ROW_EPOCH_EVENT_SCHEMA = "ny-cli/margin-row-epoch-execution/v1"
_MARGIN_ROW_EPOCH_EVENT_FIELDS = frozenset({
    "schema", "phase", "route", "outcome", "configured_depth",
    "configured_attempts", "configured_tier0", "configured_universe",
    "configured_retain", "configured_parent_topk", "configured_debug",
    "policy_mismatch", "tree_entries", "configured_depth_admissions",
    "configured_depth_nested_with_stats", "adaptive_depth_admissions",
    "attempt_cap_decisions", "attempt_cap_counterfactual_differences",
    "attempt_cap_refusals", "generation_refusals", "deadline_refusals",
    "attempts_started", "attempt_failures", "attempts_closed",
    "outer_full_attempts", "outer_incremental_attempts",
    "nested_full_attempts", "nested_incremental_attempts",
    "full_rebuilds_completed", "full_rebuild_refusals",
    "parent_refreshes_started", "parent_refreshes_completed",
    "parent_refreshes_accepted", "parent_refresh_refusals",
    "parent_refresh_columns", "parent_refresh_tightened",
    "parent_refresh_stabilized", "nested_started",
    "nested_runs_with_stats", "nested_unsat", "nested_unknown",
    "nested_errors", "closures", "debug_emissions", "outer_tier0_calls",
    "outer_tier0_scored_calls", "outer_tier0_exact_binding_scored_calls",
    "outer_universe_counterfactual_scored_calls",
    "outer_retain_counterfactual_scored_calls", "nested_full_tier0_calls",
    "nested_incremental_tier0_calls",
})
_MARGIN_ROW_EPOCH_BOOLEAN_FIELDS = frozenset({
    "configured_debug", "policy_mismatch",
})
_MARGIN_ROW_EPOCH_INTEGER_FIELDS = (
    _MARGIN_ROW_EPOCH_EVENT_FIELDS
    - _MARGIN_ROW_EPOCH_BOOLEAN_FIELDS
    - {"schema", "phase", "route", "outcome"}
)


def _margin_row_epoch_contract_valid(fields, status):
    """Validate one frozen epoch terminal and its producer invariants."""
    if (
        set(fields) != _MARGIN_ROW_EPOCH_EVENT_FIELDS
        or fields.get("schema") != _MARGIN_ROW_EPOCH_EVENT_SCHEMA
        or status != "ran"
        or fields["phase"] != "terminal"
        or fields["route"] not in {"joint", "classwise"}
        or fields["outcome"] not in {"unsat", "unknown"}
        or any(
            fields[name] not in {"true", "false"}
            for name in _MARGIN_ROW_EPOCH_BOOLEAN_FIELDS
        )
        or fields["policy_mismatch"] != "false"
    ):
        return False

    numbers = {}
    for name in _MARGIN_ROW_EPOCH_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        numbers[name] = value

    value = lambda name: numbers[name]
    full_attempts = value("outer_full_attempts") + value("nested_full_attempts")
    incremental_attempts = (
        value("outer_incremental_attempts")
        + value("nested_incremental_attempts")
    )
    tier0_calls = (
        value("outer_tier0_calls")
        + value("nested_full_tier0_calls")
        + value("nested_incremental_tier0_calls")
    )
    parent_topk = value("configured_parent_topk")
    if (
        value("generation_refusals") + value("attempt_cap_decisions")
        > value("tree_entries")
        or value("attempt_cap_counterfactual_differences")
        > value("attempt_cap_decisions")
        or value("attempt_cap_refusals") > value("attempt_cap_decisions")
        or value("attempt_cap_decisions")
        != value("attempt_cap_refusals")
        + value("deadline_refusals")
        + value("attempts_started")
        or value("configured_depth_admissions")
        + value("adaptive_depth_admissions")
        != value("attempts_started")
        or value("attempt_failures") + value("attempts_closed")
        != value("attempts_started")
        or full_attempts + incremental_attempts != value("attempts_started")
        or value("full_rebuilds_completed") + value("full_rebuild_refusals")
        != full_attempts
        or value("parent_refreshes_started") != incremental_attempts
        or value("parent_refreshes_accepted")
        + value("parent_refresh_refusals")
        != value("parent_refreshes_started")
        or value("parent_refreshes_completed")
        > value("parent_refreshes_started")
        or value("parent_refreshes_accepted")
        > value("parent_refreshes_completed")
        or value("parent_refresh_tightened") > value("parent_refresh_columns")
        or value("parent_refresh_stabilized")
        > value("parent_refresh_tightened")
        or (value("parent_refreshes_accepted") == 0)
        != (value("parent_refresh_tightened") == 0)
        or (
            value("parent_refreshes_completed") == 0
            and any(value(name) != 0 for name in (
                "parent_refresh_columns", "parent_refresh_tightened",
                "parent_refresh_stabilized",
            ))
        )
        or value("nested_started")
        != value("full_rebuilds_completed")
        + value("parent_refreshes_accepted")
        or value("nested_unsat") + value("nested_unknown")
        + value("nested_errors")
        != value("nested_started")
        or value("nested_unsat") > value("nested_runs_with_stats")
        or value("nested_runs_with_stats")
        > value("nested_unsat") + value("nested_unknown")
        or value("attempts_closed") != value("nested_unsat")
        or value("closures") != value("nested_unsat")
        or value("attempt_failures")
        != value("full_rebuild_refusals")
        + value("parent_refresh_refusals")
        + value("nested_unknown")
        + value("nested_errors")
        or value("configured_depth_nested_with_stats")
        > value("configured_depth_admissions")
        or value("configured_depth_nested_with_stats")
        > value("nested_runs_with_stats")
        or (
            fields["configured_debug"] == "false"
            and value("debug_emissions") != 0
        )
        or (
            fields["configured_debug"] == "true"
            and value("debug_emissions") != value("nested_started")
        )
        or value("outer_tier0_scored_calls") > value("outer_tier0_calls")
        or any(
            value(name) > value("outer_tier0_scored_calls")
            for name in (
                "outer_tier0_exact_binding_scored_calls",
                "outer_universe_counterfactual_scored_calls",
                "outer_retain_counterfactual_scored_calls",
            )
        )
        or tier0_calls > value("tree_entries")
    ):
        return False

    if value("configured_depth") == 0:
        if (
            value("configured_depth_admissions") != 0
            or value("configured_depth_nested_with_stats") != 0
        ):
            return False
    elif value("adaptive_depth_admissions") != 0:
        return False
    if (
        value("configured_attempts") == 2
        and value("attempt_cap_counterfactual_differences") != 0
    ):
        return False
    if parent_topk == 0:
        if any(value(name) != 0 for name in (
            "generation_refusals", "outer_incremental_attempts",
            "nested_incremental_attempts", "parent_refreshes_started",
            "parent_refreshes_completed", "parent_refreshes_accepted",
            "parent_refresh_refusals", "parent_refresh_columns",
            "parent_refresh_tightened", "parent_refresh_stabilized",
            "nested_incremental_tier0_calls",
        )):
            return False
    elif any(value(name) != 0 for name in (
        "outer_full_attempts", "nested_full_attempts",
        "full_rebuilds_completed", "full_rebuild_refusals",
        "nested_full_tier0_calls",
    )):
        return False
    if value("configured_tier0") == 0 and tier0_calls != 0:
        return False
    if (
        value("configured_universe") == 64
        and value("outer_universe_counterfactual_scored_calls") != 0
    ):
        return False
    if (
        value("configured_retain") == 128
        and value("outer_retain_counterfactual_scored_calls") != 0
    ):
        return False
    return True


def _margin_row_epoch_terminal_evidences(row):
    """Accept exactly one strict epoch terminal; duplicates poison the row."""
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("method") != "margin_row_epoch_execution"
        ):
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii() or not reason
            or reason != reason.strip(" ") or "  " in reason
            or any(
                character.isspace() and character != " "
                for character in reason
            )
        ):
            return []
        tokens = reason.split(" ")
        if (
            not tokens
            or tokens[0] != "schema=" + _MARGIN_ROW_EPOCH_EVENT_SCHEMA
        ):
            return []
        fields = {}
        for token in tokens:
            if token.count("=") != 1:
                return []
            key, token_value = token.split("=", 1)
            if not key or not token_value or key in fields:
                return []
            fields[key] = token_value
        if not _margin_row_epoch_contract_valid(fields, event.get("status")):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    return records if len(records) == 1 else []


def _margin_row_epoch_configuration_matches(event, requested):
    fields = {
        "NY_EPOCH_DEPTH": "configured_depth",
        "NY_EPOCH_ATTEMPTS": "configured_attempts",
        "NY_EPOCH_TIER0": "configured_tier0",
        "NY_EPOCH_UNIVERSE": "configured_universe",
        "NY_EPOCH_RETAIN": "configured_retain",
        "NY_MARGIN_ROW_EPOCH_PARENT_TOPK": "configured_parent_topk",
    }
    for name, field in fields.items():
        if name not in requested:
            continue
        token = requested[name]
        if _canonical_event_u64(token) is None or event.get(field) != token:
            return False
    return True


def _margin_row_epoch_parameter_engagement(wanted, rows):
    """Credit only outer joint full-frame epoch decisions and scored work."""
    requested = {
        name: token for name, token in wanted
        if name in _MARGIN_ROW_EPOCH_T1_FAMILY
    }
    if not requested:
        return set()
    engaged = set()
    for row in rows:
        for event in _margin_row_epoch_terminal_evidences(row):
            if (
                event["route"] != "joint"
                or event["policy_mismatch"] != "false"
                or event["configured_parent_topk"] != "0"
                or not _margin_row_epoch_configuration_matches(event, requested)
            ):
                continue
            number = lambda name: _canonical_event_u64(event[name])
            if (
                "NY_EPOCH_DEPTH" in requested
                and number("configured_depth_nested_with_stats") > 0
                and number("full_rebuilds_completed") > 0
            ):
                engaged.add("NY_EPOCH_DEPTH")
            if (
                "NY_EPOCH_ATTEMPTS" in requested
                and number("attempt_cap_counterfactual_differences") > 0
            ):
                engaged.add("NY_EPOCH_ATTEMPTS")
            if (
                "NY_EPOCH_TIER0" in requested
                and number("outer_tier0_exact_binding_scored_calls") > 0
            ):
                engaged.add("NY_EPOCH_TIER0")
            if (
                "NY_EPOCH_UNIVERSE" in requested
                and number("outer_universe_counterfactual_scored_calls") > 0
            ):
                engaged.add("NY_EPOCH_UNIVERSE")
            if (
                "NY_EPOCH_RETAIN" in requested
                and number("outer_retain_counterfactual_scored_calls") > 0
            ):
                engaged.add("NY_EPOCH_RETAIN")
    return engaged


def _event_f64_from_bits(token, *, nonnegative=False, positive=False):
    bits = _canonical_event_u64(token)
    if bits is None or bits > (1 << 64) - 1:
        return None
    value = struct.unpack(">d", struct.pack(">Q", bits))[0]
    if (
        not math.isfinite(value)
        or (nonnegative and value < 0.0)
        or (positive and value <= 0.0)
    ):
        return None
    return value


def _optional_event_f64_from_bits(token):
    if token == "none":
        return None, True
    value = _event_f64_from_bits(token)
    return value, value is not None


def _margin_row_root_alpha_contract_valid(fields, status):
    """Validate one route-split root-alpha/rebake terminal exactly."""
    if (
        set(fields) != _MARGIN_ROW_ROOT_ALPHA_EVENT_FIELDS
        or fields.get("schema") != _MARGIN_ROW_ROOT_ALPHA_EVENT_SCHEMA
        or status != "ran"
        or fields["phase"] != "terminal"
        or fields["outcome"] not in {"unsat", "unknown"}
        or any(
            fields[name] not in {"true", "false"}
            for name in _MARGIN_ROW_ROOT_ALPHA_BOOLEAN_FIELDS
        )
        or fields["policy_mismatch"] != "false"
    ):
        return False

    integers = {}
    for name in _MARGIN_ROW_ROOT_ALPHA_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        integers[name] = value
    optional = {}
    for name in _MARGIN_ROW_ROOT_ALPHA_OPTIONAL_BITS_FIELDS:
        value, valid = _optional_event_f64_from_bits(fields[name])
        if not valid:
            return False
        optional[name] = value

    alpha_secs = _event_f64_from_bits(
        fields["alpha_secs_bits"], positive=True)
    alpha_lr = _event_f64_from_bits(
        fields["alpha_lr_bits"], nonnegative=True)
    rebake_secs = _event_f64_from_bits(
        fields["rebake_secs_bits"], nonnegative=True)
    standalone_effective = _event_f64_from_bits(
        fields["standalone_effective_secs_bits"], nonnegative=True)
    rebake_effective = _event_f64_from_bits(
        fields["rebake_effective_secs_bits"], nonnegative=True)
    if (
        alpha_secs is None
        or alpha_lr is None or alpha_lr > 1.0
        or rebake_secs is None
        or standalone_effective is None
        or rebake_effective is None
        or not 1 <= integers["alpha_iters"] <= 64
        or not 1 <= integers["alpha_cols"] <= 64
        or not 1 <= integers["rebake_rounds"] <= 8
    ):
        return False

    eligible = fields["eligible"] == "true"
    alpha_enabled = fields["alpha_enabled"] == "true"
    alpha_guard = fields["alpha_guard"] == "true"
    rebake_enabled = fields["rebake_enabled"] == "true"
    standalone_completed = integers["standalone_completed"]
    rebake_completed = integers["rebake_completed"]
    if (
        integers["standalone_suppressed"]
            != int(eligible and not alpha_enabled)
        or integers["standalone_attempted"] != int(alpha_enabled)
        or standalone_completed not in {0, 1}
        or standalone_completed > integers["standalone_attempted"]
        or integers["rebake_suppressed"]
            != int(eligible and not rebake_enabled)
        or integers["rebake_attempted"] != int(rebake_enabled)
        or rebake_completed not in {0, 1}
        or rebake_completed > integers["rebake_attempted"]
    ):
        return False

    standalone_zero_fields = {
        "standalone_fail_classes", "standalone_iters_attempted",
        "standalone_candidate_steps", "standalone_legacy_steps",
        "standalone_adam_steps", "standalone_passes",
        "standalone_max_steer_cols", "standalone_frozen_improvements",
        "standalone_moved_neurons", "standalone_guard_opportunities",
        "standalone_guard_baseline_evals",
        "standalone_guard_candidate_evals", "standalone_guard_reversions",
    }
    standalone_optional = {
        "standalone_frozen_before_bits", "standalone_frozen_best_bits",
        "standalone_published_before_bits", "standalone_published_after_bits",
    }
    if standalone_completed == 0:
        if (
            standalone_effective != 0.0
            or any(integers[name] != 0 for name in standalone_zero_fields)
            or any(optional[name] is not None for name in standalone_optional)
            or fields["standalone_deadline_capped"] != "false"
            or fields["standalone_budget_exhausted"] != "false"
            or fields["standalone_iters_cap_exhausted"] != "false"
            or fields["standalone_accepted"] != "false"
        ):
            return False
    else:
        before = optional["standalone_frozen_before_bits"]
        best = optional["standalone_frozen_best_bits"]
        fail_classes = integers["standalone_fail_classes"]
        attempted = integers["standalone_iters_attempted"]
        candidate_steps = integers["standalone_candidate_steps"]
        legacy_steps = integers["standalone_legacy_steps"]
        adam_steps = integers["standalone_adam_steps"]
        passes = integers["standalone_passes"]
        improvements = integers["standalone_frozen_improvements"]
        opportunities = integers["standalone_guard_opportunities"]
        baseline_evals = integers["standalone_guard_baseline_evals"]
        candidate_evals = integers["standalone_guard_candidate_evals"]
        reversions = integers["standalone_guard_reversions"]
        moved = integers["standalone_moved_neurons"]
        max_steer = integers["standalone_max_steer_cols"]
        published_before = optional["standalone_published_before_bits"]
        published_after = optional["standalone_published_after_bits"]
        cap_exhausted = fields["standalone_iters_cap_exhausted"] == "true"
        accepted = fields["standalone_accepted"] == "true"
        if (
            not alpha_enabled or not eligible
            or standalone_effective <= 0.0
            or standalone_effective < 0.5
            or standalone_effective > alpha_secs
            or (fields["standalone_deadline_capped"] == "true")
                != (standalone_effective < alpha_secs)
            or fail_classes == 0
            or attempted > integers["alpha_iters"]
            or candidate_steps > attempted
            or attempted - candidate_steps > 1
            or passes != candidate_steps + 1
            or legacy_steps + adam_steps != candidate_steps
            or (alpha_lr == 0.0 and adam_steps != 0)
            or (alpha_lr > 0.0 and legacy_steps != 0)
            or max_steer != min(integers["alpha_cols"], fail_classes)
            or improvements > candidate_steps
            or cap_exhausted
                != (candidate_steps == integers["alpha_iters"])
            or (
                fields["standalone_budget_exhausted"] == "true"
                and (
                    cap_exhausted
                    or attempted != candidate_steps
                )
            )
            or before is None or best is None or best < before
            or (improvements > 0) != (best > before)
            or opportunities != int(improvements > 0)
            or accepted != (moved > 0)
            or accepted and opportunities == 0
        ):
            return False
        if alpha_guard:
            if (
                baseline_evals != 1
                or published_before is None
                or candidate_evals != opportunities
                or reversions > candidate_evals
                or (
                    published_after is not None
                    and candidate_evals == 0
                )
                or accepted
                    != (
                        published_after is not None
                        and published_after > published_before
                    )
                or reversions != int(opportunities > 0 and not accepted)
            ):
                return False
        elif (
            baseline_evals != 0
            or candidate_evals != 0
            or reversions != 0
            or published_before is not None
            or published_after is not None
            or accepted != (opportunities > 0)
        ):
            return False

    rebake_zero_fields = {
        "rebake_baseline_evals", "rebake_rounds_started",
        "rebake_alpha_completed", "rebake_alpha_iters_attempted",
        "rebake_alpha_candidate_steps", "rebake_alpha_legacy_steps",
        "rebake_alpha_adam_steps", "rebake_alpha_passes",
        "rebake_alpha_max_fail_classes", "rebake_alpha_max_steer_cols",
        "rebake_alpha_moved_neurons", "rebake_rebuilds_completed",
        "rebake_rounds_scored", "rebake_rounds_accepted",
        "rebake_fixed_point_stops", "rebake_affordability_refusals",
        "rebake_aborts",
    }
    if rebake_completed == 0:
        if (
            rebake_effective != 0.0
            or any(integers[name] != 0 for name in rebake_zero_fields)
            or optional["rebake_published_before_bits"] is not None
            or optional["rebake_published_after_bits"] is not None
            or any(fields[name] != "false" for name in (
                "rebake_deadline_capped", "rebake_budget_exhausted",
                "rebake_round_cap_exhausted", "rebake_entry_unbaked_alpha",
            ))
        ):
            return False
    else:
        rounds_started = integers["rebake_rounds_started"]
        alpha_completed = integers["rebake_alpha_completed"]
        alpha_attempted = integers["rebake_alpha_iters_attempted"]
        alpha_steps = integers["rebake_alpha_candidate_steps"]
        alpha_legacy = integers["rebake_alpha_legacy_steps"]
        alpha_adam = integers["rebake_alpha_adam_steps"]
        alpha_passes = integers["rebake_alpha_passes"]
        max_fail = integers["rebake_alpha_max_fail_classes"]
        max_steer = integers["rebake_alpha_max_steer_cols"]
        rebuilds = integers["rebake_rebuilds_completed"]
        scored = integers["rebake_rounds_scored"]
        accepted_rounds = integers["rebake_rounds_accepted"]
        fixed = integers["rebake_fixed_point_stops"]
        affordability = integers["rebake_affordability_refusals"]
        aborts = integers["rebake_aborts"]
        before = optional["rebake_published_before_bits"]
        after = optional["rebake_published_after_bits"]
        if (
            not rebake_enabled or not eligible
            or rebake_effective > rebake_secs
            or (fields["rebake_deadline_capped"] == "true")
                != (rebake_effective < rebake_secs)
            or integers["rebake_baseline_evals"] != 1
            or (fields["rebake_entry_unbaked_alpha"] == "true")
                != (fields["standalone_accepted"] == "true")
            or rounds_started > integers["rebake_rounds"]
            or alpha_completed > rounds_started
            or alpha_attempted > 8 * alpha_completed
            or alpha_steps > alpha_attempted
            or alpha_attempted - alpha_steps > alpha_completed
            or alpha_passes != alpha_steps + alpha_completed
            or alpha_legacy + alpha_adam != alpha_steps
            or (alpha_lr == 0.0 and alpha_adam != 0)
            or (alpha_lr > 0.0 and alpha_legacy != 0)
            or (alpha_completed == 0) != (max_fail == 0 and max_steer == 0)
            or max_steer != min(integers["alpha_cols"], max_fail)
            or integers["rebake_alpha_moved_neurons"] > 0
                and alpha_steps == 0
            or rebuilds > rounds_started
            or scored > rebuilds
            or accepted_rounds > scored
            or scored - accepted_rounds > 1
            or fixed + affordability + aborts > 1
            or (fields["rebake_budget_exhausted"] == "true")
                != (affordability > 0)
            or (fields["rebake_round_cap_exhausted"] == "true")
                != (rounds_started == integers["rebake_rounds"])
            or affordability > 0
                and rounds_started == integers["rebake_rounds"]
            or before is None or after is None or after < before
            or (accepted_rounds > 0) != (after > before)
        ):
            return False
        if fixed > 0 and not (
            rounds_started == scored + 1
            and rebuilds == scored
            and accepted_rounds == scored
            and not (
                fields["rebake_entry_unbaked_alpha"] == "true"
                and scored == 0
            )
        ):
            return False
        if affordability > 0 and not (
            rounds_started == rebuilds == scored == accepted_rounds
        ):
            return False
        if aborts > 0 and not (
            rounds_started == scored + 1
            and rebuilds in {scored, scored + 1}
            and accepted_rounds == scored
        ):
            return False
        if fixed + affordability + aborts == 0 and not (
            rounds_started == rebuilds == scored
            and (
                accepted_rounds == scored
                and rounds_started == integers["rebake_rounds"]
                or scored > 0
                and accepted_rounds + 1 == scored
            )
        ):
            return False
        moved = integers["rebake_alpha_moved_neurons"]
        if (
            not (fields["rebake_entry_unbaked_alpha"] == "true")
            and rebuilds > 0
            and moved == 0
        ) or (
            fields["rebake_entry_unbaked_alpha"] == "true"
            and moved == 0
            and rebuilds > 1
        ):
            return False
    return True


def _margin_row_root_alpha_terminal_evidences(row):
    """Accept exactly one strict terminal; duplicates poison the row."""
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("method") != "margin_row_root_alpha_execution"
        ):
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii() or not reason
            or reason != reason.strip(" ") or "  " in reason
            or any(
                character.isspace() and character != " "
                for character in reason
            )
        ):
            return []
        tokens = reason.split(" ")
        if (
            not tokens
            or tokens[0] != "schema=" + _MARGIN_ROW_ROOT_ALPHA_EVENT_SCHEMA
        ):
            return []
        fields = {}
        for token in tokens:
            if token.count("=") != 1:
                return []
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                return []
            fields[key] = value
        if not _margin_row_root_alpha_contract_valid(
            fields, event.get("status")
        ):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    return records if len(records) == 1 else []


def _requested_f64_bits(token):
    if not isinstance(token, str):
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _margin_row_root_alpha_configuration_matches(event, requested):
    boolean_fields = {
        "NY_MARGIN_ROW_ALPHA_OPT": "alpha_enabled",
        "NY_MARGIN_ROW_ALPHA_GUARD": "alpha_guard",
        "NY_MARGIN_ROW_REBAKE": "rebake_enabled",
    }
    integer_fields = {
        "NY_MARGIN_ROW_ALPHA_ITERS": "alpha_iters",
        "NY_MARGIN_ROW_ALPHA_COLS": "alpha_cols",
        "NY_MARGIN_ROW_REBAKE_ROUNDS": "rebake_rounds",
    }
    float_fields = {
        "NY_MARGIN_ROW_ALPHA_SECS": "alpha_secs_bits",
        "NY_MARGIN_ROW_ALPHA_LR": "alpha_lr_bits",
        "NY_MARGIN_ROW_REBAKE_SECS": "rebake_secs_bits",
    }
    for name, field in boolean_fields.items():
        token = requested.get(name)
        if token is not None and (
            token not in {"0", "1"}
            or event[field] != ("true" if token == "1" else "false")
        ):
            return False
    for name, field in integer_fields.items():
        token = requested.get(name)
        if token is not None and (
            _canonical_event_u64(token) is None or event[field] != token
        ):
            return False
    for name, field in float_fields.items():
        token = requested.get(name)
        bits = _requested_f64_bits(token)
        if token is not None and (bits is None or event[field] != str(bits)):
            return False
    return True


def _root_alpha_cols_route_engaged(event, requested_cols, prefix):
    steps = _canonical_event_u64(event[prefix + "candidate_steps"])
    fail_field = (
        "standalone_fail_classes"
        if prefix == "standalone_"
        else prefix + "max_fail_classes"
    )
    fail_classes = _canonical_event_u64(event[fail_field])
    steer_cols = _canonical_event_u64(event[prefix + "max_steer_cols"])
    if steps is None or fail_classes is None or steer_cols is None or steps == 0:
        return False
    if steer_cols != min(requested_cols, fail_classes):
        return False
    if requested_cols < 8:
        return fail_classes > requested_cols
    if requested_cols > 8:
        return fail_classes > 8 and steer_cols > 8
    return True


def _margin_row_root_alpha_parameter_engagement(wanted, rows):
    """Prove value-binding standalone/rebake work without route cross-credit."""
    requested = {
        name: token for name, token in wanted
        if name in _MARGIN_ROW_ROOT_ALPHA_FAMILY
    }
    if not requested:
        return set()
    engaged = set()
    for row in rows:
        for event in _margin_row_root_alpha_terminal_evidences(row):
            if not _margin_row_root_alpha_configuration_matches(
                event, requested
            ):
                continue
            number = lambda name: _canonical_event_u64(event[name])
            standalone_steps = number("standalone_candidate_steps")
            rebake_steps = number("rebake_alpha_candidate_steps")
            standalone_live = (
                number("standalone_completed") == 1
                and standalone_steps is not None and standalone_steps > 0
            )
            rebake_live = (
                number("rebake_completed") == 1
                and rebake_steps is not None and rebake_steps > 0
            )

            alpha = requested.get("NY_MARGIN_ROW_ALPHA_OPT")
            if (
                alpha == "1" and standalone_live
            ) or (
                alpha == "0"
                and event["eligible"] == "true"
                and number("standalone_suppressed") == 1
            ):
                engaged.add("NY_MARGIN_ROW_ALPHA_OPT")
            if (
                "NY_MARGIN_ROW_ALPHA_ITERS" in requested
                and standalone_live
                and event["standalone_iters_cap_exhausted"] == "true"
            ):
                engaged.add("NY_MARGIN_ROW_ALPHA_ITERS")
            if (
                "NY_MARGIN_ROW_ALPHA_SECS" in requested
                and standalone_live
                and event["standalone_deadline_capped"] == "false"
                and event["standalone_budget_exhausted"] == "true"
            ):
                engaged.add("NY_MARGIN_ROW_ALPHA_SECS")
            lr = requested.get("NY_MARGIN_ROW_ALPHA_LR")
            if lr is not None:
                legacy = float(lr) == 0.0
                standalone_mode = number(
                    "standalone_legacy_steps" if legacy
                    else "standalone_adam_steps"
                )
                rebake_mode = number(
                    "rebake_alpha_legacy_steps" if legacy
                    else "rebake_alpha_adam_steps"
                )
                if (
                    standalone_live and standalone_mode > 0
                ) or (
                    rebake_live and rebake_mode > 0
                ):
                    engaged.add("NY_MARGIN_ROW_ALPHA_LR")
            cols = requested.get("NY_MARGIN_ROW_ALPHA_COLS")
            if cols is not None:
                requested_cols = int(cols)
                standalone_cols = _root_alpha_cols_route_engaged(
                    event, requested_cols, "standalone_")
                rebake_cols = _root_alpha_cols_route_engaged(
                    event, requested_cols, "rebake_alpha_")
                if standalone_cols or rebake_cols:
                    engaged.add("NY_MARGIN_ROW_ALPHA_COLS")
            guard = requested.get("NY_MARGIN_ROW_ALPHA_GUARD")
            if (
                guard is not None
                and standalone_live
                and number("standalone_guard_opportunities") > 0
                and (
                    guard == "0"
                    and number("standalone_guard_candidate_evals") == 0
                    or guard == "1"
                    and number("standalone_guard_candidate_evals") > 0
                    and event["standalone_published_after_bits"] != "none"
                )
            ):
                engaged.add("NY_MARGIN_ROW_ALPHA_GUARD")

            rebake = requested.get("NY_MARGIN_ROW_REBAKE")
            if (
                rebake == "1"
                and number("rebake_rebuilds_completed") > 0
                and number("rebake_rounds_scored") > 0
                and number("rebake_rounds_accepted") > 0
            ) or (
                rebake == "0"
                and event["eligible"] == "true"
                and number("rebake_suppressed") == 1
            ):
                engaged.add("NY_MARGIN_ROW_REBAKE")
            if (
                "NY_MARGIN_ROW_REBAKE_ROUNDS" in requested
                and event["rebake_round_cap_exhausted"] == "true"
                and number("rebake_rounds_started")
                    == number("rebake_rounds")
                and number("rebake_rounds_scored")
                    == number("rebake_rounds")
                and number("rebake_rounds_accepted")
                    == number("rebake_rounds")
            ):
                engaged.add("NY_MARGIN_ROW_REBAKE_ROUNDS")
            if (
                "NY_MARGIN_ROW_REBAKE_SECS" in requested
                and number("rebake_rebuilds_completed") > 0
                and event["rebake_deadline_capped"] == "false"
                and event["rebake_budget_exhausted"] == "true"
                and number("rebake_affordability_refusals") > 0
            ):
                engaged.add("NY_MARGIN_ROW_REBAKE_SECS")
    return engaged


_MARGIN_ROW_BACKWARD_INTERM_FAMILY = frozenset({
    "NY_MARGIN_ROW_BACKWARD_INTERM",
    "NY_MARGIN_ROW_BI_SECS",
    "NY_MARGIN_ROW_BI_CHUNK",
    "NY_MARGIN_ROW_BI_CHUNK_SECS",
    "NY_MARGIN_ROW_BI_TOPK",
})
_MARGIN_ROW_BACKWARD_INTERM_EVENT_SCHEMA = (
    "ny-cli/margin-row-backward-interm-execution/v1"
)
_MARGIN_ROW_BACKWARD_INTERM_EVENT_FIELDS = frozenset({
    "schema", "phase", "outcome", "enabled", "configured_secs",
    "configured_chunk", "configured_chunk_secs", "effective_chunk_secs",
    "configured_topk", "eligible_layers", "suppressed_setup_layers",
    "layers_started", "chunks_completed", "columns_completed",
    "deadline_capped", "budget_exhausted",
    "chunk_secs_capped", "chunk_counterfactual_chunks",
    "chunk_ceiling_binding_chunks", "fixed_width_chunks",
    "adaptive_width_changes", "topk_binding_layers", "policy_mismatch",
})
_MARGIN_ROW_BACKWARD_INTERM_INTEGER_FIELDS = frozenset({
    "configured_chunk", "configured_topk", "eligible_layers",
    "suppressed_setup_layers", "layers_started", "chunks_completed",
    "columns_completed",
    "chunk_counterfactual_chunks", "chunk_ceiling_binding_chunks",
    "fixed_width_chunks",
    "adaptive_width_changes", "topk_binding_layers",
})
_MARGIN_ROW_BACKWARD_INTERM_COUNTER_FIELDS = frozenset({
    "eligible_layers", "suppressed_setup_layers", "layers_started",
    "chunks_completed", "columns_completed", "chunk_counterfactual_chunks",
    "chunk_ceiling_binding_chunks", "fixed_width_chunks",
    "adaptive_width_changes", "topk_binding_layers",
})


def _canonical_event_nonnegative_decimal(token):
    """Return a canonical finite decimal token, excluding exponent spellings."""
    if not isinstance(token, str) or re.fullmatch(
        r"(?:0|[1-9][0-9]*|0\.[0-9]*[1-9]|[1-9][0-9]*\.[0-9]*[1-9])",
        token,
    ) is None:
        return None
    value = float(token)
    return token if math.isfinite(value) and value >= 0.0 else None


def _margin_row_backward_interm_contract_valid(fields, status):
    """Validate one closed backward-intermediate phase receipt."""
    if (
        set(fields) != _MARGIN_ROW_BACKWARD_INTERM_EVENT_FIELDS
        or fields.get("schema") != _MARGIN_ROW_BACKWARD_INTERM_EVENT_SCHEMA
        or status != "ran"
        or fields["phase"] != "terminal"
        or fields["outcome"] not in {"disabled", "completed", "exhausted"}
        or any(fields[name] not in {"true", "false"} for name in (
            "enabled", "deadline_capped", "budget_exhausted",
            "chunk_secs_capped", "policy_mismatch",
        ))
        or fields["policy_mismatch"] != "false"
        or _canonical_event_nonnegative_decimal(fields["configured_secs"]) is None
        or _canonical_event_nonnegative_decimal(
            fields["configured_chunk_secs"]
        ) is None
        or _canonical_event_nonnegative_decimal(
            fields["effective_chunk_secs"]
        ) is None
    ):
        return False

    integers = {}
    for name in _MARGIN_ROW_BACKWARD_INTERM_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        integers[name] = value
    if integers["configured_chunk"] == 0 or integers["configured_topk"] == 0:
        return False

    enabled = fields["enabled"] == "true"
    deadline_capped = fields["deadline_capped"] == "true"
    budget_exhausted = fields["budget_exhausted"] == "true"
    chunk_secs_capped = fields["chunk_secs_capped"] == "true"
    configured_secs = float(fields["configured_secs"])
    configured_chunk_secs = float(fields["configured_chunk_secs"])
    effective_chunk_secs = float(fields["effective_chunk_secs"])
    if configured_chunk_secs == 0.0:
        if effective_chunk_secs != 0.0 or chunk_secs_capped:
            return False
    elif (
        effective_chunk_secs <= 0.0
        or effective_chunk_secs > configured_chunk_secs
        or chunk_secs_capped != (effective_chunk_secs < configured_chunk_secs)
    ):
        return False
    if not enabled:
        return (
            fields["outcome"] == "disabled"
            and not deadline_capped
            and not budget_exhausted
            and integers["suppressed_setup_layers"]
                <= integers["eligible_layers"]
            and (
                integers["suppressed_setup_layers"] == 0
                or configured_secs > 0.0
            )
            and all(integers[name] == 0 for name in (
                "layers_started", "chunks_completed", "columns_completed",
                "chunk_counterfactual_chunks", "chunk_ceiling_binding_chunks",
                "fixed_width_chunks",
                "adaptive_width_changes", "topk_binding_layers",
            ))
        )
    if (
        fields["outcome"] == "disabled"
        or (fields["outcome"] == "exhausted") != budget_exhausted
    ):
        return False

    eligible = integers["eligible_layers"]
    suppressed_setup = integers["suppressed_setup_layers"]
    layers = integers["layers_started"]
    chunks = integers["chunks_completed"]
    columns = integers["columns_completed"]
    chunk_counterfactual = integers["chunk_counterfactual_chunks"]
    chunk_ceiling_binding = integers["chunk_ceiling_binding_chunks"]
    fixed_width = integers["fixed_width_chunks"]
    adaptive_changes = integers["adaptive_width_changes"]
    topk_binding = integers["topk_binding_layers"]
    if (
        layers > eligible
        or suppressed_setup != 0
        or chunk_counterfactual > chunks
        or chunk_ceiling_binding > chunks
        or fixed_width > chunks
        or adaptive_changes > chunks
        or topk_binding > layers
        or (chunks > 0 and (layers == 0 or columns < chunks))
        or (chunks == 0 and columns != 0)
        or (layers == 0 and (chunks or columns or topk_binding))
        or (effective_chunk_secs == 0.0 and adaptive_changes != 0)
        or (effective_chunk_secs > 0.0 and fixed_width != 0)
        or (
            integers["configured_chunk"] == 256
            and chunk_counterfactual != 0
        )
    ):
        return False
    return True


def _margin_row_backward_interm_terminal_evidences(row):
    """Strict receipts; malformed or mixed-policy terminals poison the row."""
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if not isinstance(event, dict) or event.get("method") \
                != "margin_row_backward_interm_execution":
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii()
            or not reason
            or reason != reason.strip(" ")
            or "  " in reason
            or any(character.isspace() and character != " " for character in reason)
        ):
            return []
        tokens = reason.split(" ")
        if not tokens or tokens[0] != (
            "schema=" + _MARGIN_ROW_BACKWARD_INTERM_EVENT_SCHEMA
        ):
            return []
        fields = {}
        for token in tokens:
            if token.count("=") != 1:
                return []
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                return []
            fields[key] = value
        if not _margin_row_backward_interm_contract_valid(
            fields, event.get("status")
        ):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    configurations = {
        (
            fields["enabled"],
            fields["configured_secs"],
            fields["configured_chunk"],
            fields["configured_chunk_secs"],
            fields["configured_topk"],
        )
        for fields in records
    }
    if len(configurations) > 1:
        # One scored row owns one frozen LeverSet. Concurrent/inline terminal
        # summaries may repeat that policy, but contradictory summaries prove
        # the envelope mixed configurations and cannot authenticate either.
        return []
    return records


def _margin_row_backward_interm_configuration_matches(event, requested):
    enabled = requested.get("NY_MARGIN_ROW_BACKWARD_INTERM")
    if enabled is not None and (
        enabled not in {"0", "1"}
        or event["enabled"] != ("true" if enabled == "1" else "false")
    ):
        return False

    decimal_fields = {
        "NY_MARGIN_ROW_BI_SECS": ("configured_secs", {"20", "30"}),
        "NY_MARGIN_ROW_BI_CHUNK_SECS": (
            "configured_chunk_secs", {"0", "1", "2"},
        ),
    }
    integer_fields = {
        "NY_MARGIN_ROW_BI_CHUNK": (
            "configured_chunk", {"64", "256", "1024"},
        ),
        "NY_MARGIN_ROW_BI_TOPK": (
            "configured_topk", {"256", "1024", "4096"},
        ),
    }
    for name, (field, domain) in decimal_fields.items():
        token = requested.get(name)
        if token is not None and (
            token not in domain
            or _canonical_event_nonnegative_decimal(token) is None
            or event[field] != token
        ):
            return False
    for name, (field, domain) in integer_fields.items():
        token = requested.get(name)
        if token is not None and (
            token not in domain
            or _canonical_event_u64(token) is None
            or event[field] != token
        ):
            return False
    return True


def _margin_row_backward_interm_parameter_engagement(wanted, rows):
    """Prove live, value-binding work for each requested phase parameter."""
    requested = {
        name: token for name, token in wanted
        if name in _MARGIN_ROW_BACKWARD_INTERM_FAMILY
    }
    if not requested:
        return set()
    engaged = set()
    for row in rows:
        for event in _margin_row_backward_interm_terminal_evidences(row):
            if not _margin_row_backward_interm_configuration_matches(
                event, requested
            ):
                continue
            counters = {
                name: _canonical_event_u64(event[name])
                for name in _MARGIN_ROW_BACKWARD_INTERM_COUNTER_FIELDS
            }
            enabled = event["enabled"] == "true"
            live = (
                enabled
                and counters["eligible_layers"] > 0
                and counters["layers_started"] > 0
                and counters["chunks_completed"] > 0
                and counters["columns_completed"] > 0
            )
            master = requested.get("NY_MARGIN_ROW_BACKWARD_INTERM")
            if (
                (master == "1" and live)
                or (
                    master == "0"
                    and not enabled
                    and event["outcome"] == "disabled"
                    and counters["suppressed_setup_layers"] > 0
                )
            ):
                engaged.add("NY_MARGIN_ROW_BACKWARD_INTERM")
            if (
                "NY_MARGIN_ROW_BI_SECS" in requested
                and live
                and event["deadline_capped"] == "false"
                and event["budget_exhausted"] == "true"
            ):
                engaged.add("NY_MARGIN_ROW_BI_SECS")
            chunk = requested.get("NY_MARGIN_ROW_BI_CHUNK")
            if chunk is not None and live:
                if (
                    chunk == "256"
                    and counters["chunk_ceiling_binding_chunks"] > 0
                ) or (
                    chunk != "256"
                    and counters["chunk_counterfactual_chunks"] > 0
                ):
                    engaged.add("NY_MARGIN_ROW_BI_CHUNK")
            chunk_secs = requested.get("NY_MARGIN_ROW_BI_CHUNK_SECS")
            if (
                chunk_secs == "0"
                and live
                and counters["fixed_width_chunks"] > 0
            ):
                engaged.add("NY_MARGIN_ROW_BI_CHUNK_SECS")
            elif (
                chunk_secs is not None
                and chunk_secs != "0"
                and live
                and event["chunk_secs_capped"] == "false"
                and event["effective_chunk_secs"] == chunk_secs
                and counters["adaptive_width_changes"] > 0
            ):
                engaged.add("NY_MARGIN_ROW_BI_CHUNK_SECS")
            if (
                "NY_MARGIN_ROW_BI_TOPK" in requested
                and live
                and counters["topk_binding_layers"] > 0
            ):
                engaged.add("NY_MARGIN_ROW_BI_TOPK")
    return engaged


_PGD_FAST_GRADIENT_EVENT_SCHEMA = "ny-cli/pgd-fast-gradient-execution/v1"
_PGD_FAST_GRADIENT_EVENT_FIELDS = frozenset({
    "schema", "phase", "route", "configured", "decisions", "fast_attempts",
    "fast_selected", "certified_selected", "spsa_selected",
})
_PGD_FAST_GRADIENT_INTEGER_FIELDS = frozenset({
    "decisions", "fast_attempts", "fast_selected", "certified_selected",
    "spsa_selected",
})


def _pgd_fast_gradient_contract_valid(fields, status):
    """Validate one closed sequential fast-gradient invocation in isolation."""
    if (
        set(fields) != _PGD_FAST_GRADIENT_EVENT_FIELDS
        or fields.get("schema") != _PGD_FAST_GRADIENT_EVENT_SCHEMA
        or fields["phase"] != "terminal"
        or fields["route"] not in {"conjunctive", "disjunctive"}
        or fields["configured"] not in {"true", "false"}
        or status not in {"ran", "skipped"}
    ):
        return False

    integers = {}
    for name in _PGD_FAST_GRADIENT_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        integers[name] = value

    decisions = integers["decisions"]
    attempts = integers["fast_attempts"]
    fast = integers["fast_selected"]
    certified = integers["certified_selected"]
    spsa = integers["spsa_selected"]
    configured = fields["configured"] == "true"
    if fast > attempts or fast + certified + spsa != decisions:
        return False
    if configured:
        if attempts != decisions:
            return False
    elif attempts != 0 or fast != 0:
        return False
    if decisions == 0:
        return status == "skipped" and attempts == fast == certified == spsa == 0
    return status == "ran"


def _pgd_fast_gradient_terminal_evidences(row):
    """Strict independently closed fast-gradient invocations for one row.

    Any malformed same-method event poisons the row. A later valid skipped
    invocation does not erase an earlier live decision from a separate call.
    """
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if not isinstance(event, dict) or event.get("method") != "pgd_fast_gradient_execution":
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii()
            or not reason
            or reason != reason.strip(" ")
            or "  " in reason
            or any(character.isspace() and character != " " for character in reason)
        ):
            return []
        tokens = reason.split(" ")
        if not tokens or tokens[0] != "schema=" + _PGD_FAST_GRADIENT_EVENT_SCHEMA:
            return []
        fields = {}
        for token in tokens:
            if token.count("=") != 1:
                return []
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                return []
            fields[key] = value
        if not _pgd_fast_gradient_contract_valid(fields, event.get("status")):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    if len({record["configured"] for record in records}) > 1:
        return []
    return records


def _pgd_fast_gradient_parameter_engagement(wanted, rows):
    """Whether a live sequential gradient decision consumed the requested policy."""
    requested = dict(wanted).get("NY_PGD_FAST_GRAD")
    if requested not in {"0", "1"}:
        return set()
    configured = "true" if requested == "1" else "false"
    for row in rows:
        for event in _pgd_fast_gradient_terminal_evidences(row):
            if (
                event["configured"] == configured
                and _canonical_event_u64(event["decisions"]) > 0
            ):
                return {"NY_PGD_FAST_GRAD"}
    return set()


_POSTBAB_RESERVE_EVENT_SCHEMA = "ny-cli/postbab-reserve-execution/v2"
_POSTBAB_RESERVE_EVENT_FIELDS = frozenset({
    "schema", "phase", "configured_secs", "held_secs", "verifier_before_secs",
    "verifier_after_secs", "verifier_elapsed_ms", "verifier_outcome",
    "deadline_bound", "attack_budget_ms", "work_units", "work_started",
})
_POSTBAB_RESERVE_INTEGER_FIELDS = frozenset({
    "configured_secs", "held_secs", "verifier_before_secs",
    "verifier_after_secs", "verifier_elapsed_ms", "attack_budget_ms", "work_units",
})


def _postbab_reserve_contract_valid(fields, status):
    """Validate one closed post-BaB reserve invocation in isolation."""
    if (
        set(fields) != _POSTBAB_RESERVE_EVENT_FIELDS
        or fields.get("schema") != _POSTBAB_RESERVE_EVENT_SCHEMA
        or fields["phase"] != "terminal"
        or fields["verifier_outcome"] != "timeout"
        or fields["deadline_bound"] != "true"
        or fields["work_started"] != "true"
        or status != "ran"
    ):
        return False

    integers = {}
    for name in _POSTBAB_RESERVE_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        integers[name] = value

    configured = integers["configured_secs"]
    held = integers["held_secs"]
    before = integers["verifier_before_secs"]
    after = integers["verifier_after_secs"]
    verifier_elapsed_ms = integers["verifier_elapsed_ms"]
    attack_budget_ms = integers["attack_budget_ms"]
    work_units = integers["work_units"]
    return (
        configured > 0
        and held > 0
        and before >= after
        and held == before - after
        and configured >= held
        and (configured == held or after == 10)
        and after >= 10
        and verifier_elapsed_ms >= after * 1000
        and attack_budget_ms >= 2000
        and work_units == 1
    )


def _postbab_reserve_terminal_evidences(row):
    """Strict independently closed post-BaB reserve invocations for one row.

    Any malformed same-method event poisons this row. Events are never folded
    across rows, so configured policy and live held work must be proved by the
    same independently closed terminal receipt.
    """
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if not isinstance(event, dict) or event.get("method") != "postbab_reserve_execution":
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii()
            or not reason
            or reason != reason.strip(" ")
            or "  " in reason
            or any(character.isspace() and character != " " for character in reason)
        ):
            return []
        tokens = reason.split(" ")
        if not tokens or tokens[0] != "schema=" + _POSTBAB_RESERVE_EVENT_SCHEMA:
            return []
        fields = {}
        for token in tokens:
            if token.count("=") != 1:
                return []
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                return []
            fields[key] = value
        if not _postbab_reserve_contract_valid(fields, event.get("status")):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    # The scored wrapper has exactly one reserve decision and at most one outer
    # post-BaB invocation per row. Duplicate terminals therefore indicate
    # replay/aggregation ambiguity, even when their payloads happen to match.
    if len(records) != 1:
        return []
    return records


def _postbab_reserve_parameter_engagement(wanted, rows):
    """Whether one row demonstrably converted the requested reserve into work."""
    requested = dict(wanted).get("NY_POSTBAB_RESERVE_SECS")
    expected = _canonical_event_u64(requested)
    if expected is None or expected == 0:
        return set()
    for row in rows:
        for event in _postbab_reserve_terminal_evidences(row):
            if (
                _canonical_event_u64(event["configured_secs"]) == expected
                and _canonical_event_u64(event["held_secs"]) == expected
            ):
                return {"NY_POSTBAB_RESERVE_SECS"}
    return set()


_ROOT_JOINT_INTERM_ALPHA_FAMILY = (
    _ROOT_JOINT_RECIPE_TRIGGERS | {_ROOT_JOINT_RECIPE_MASTER}
)
_ROOT_JOINT_INTERM_ALPHA_T1_FAMILY = (
    _ROOT_JOINT_INTERM_ALPHA_FAMILY
    - {"NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM"}
)
_ROOT_JOINT_INTERM_ALPHA_EVENT_SCHEMA = (
    "ny-cli/root-joint-interm-alpha-execution/v1"
)
_ROOT_JOINT_INTERM_ALPHA_OBSERVATION_SCHEMA = (
    "ny_beta_crown_root_joint_interm_alpha_observations_v1"
)
_ROOT_JOINT_INTERM_ALPHA_EVENT_FIELDS = frozenset({
    "schema", "phase", "observation_schema", "route",
    "configured_master", "configured_max_dim_override",
    "input_split_configured_max_dim", "multi_objective_configured_max_dim",
    "configured_secs",
    "configured_iters", "configured_lr_bits", "configured_deadline_ascent",
    "configured_min_remaining_secs", "configured_max_sel",
    "configured_frozen_stop", "configured_pure_chain",
    "configured_bn_gpu_extract", "max_effective_iters", "invocations",
    "input_split_invocations", "multi_objective_invocations", "target_count",
    "available_budget_ms", "effective_slice_ms",
    "configured_slice_work_invocations",
    "configured_iters_binding_invocations",
    "lowered_min_remaining_work_invocations", "work_started", "attempts",
    "raised_min_remaining_binding_refusals",
    "completed", "tightened_count", "outlier_tightened_count",
    "master_disabled",
    "target_attempts", "base_folds_completed", "ascent_iterations_attempted",
    "max_ascent_iterations_attempted", "optimizer_steps",
    "ascent_folds_completed", "max_selected_rows",
    "selection_cap_bound_targets", "frozen_stop_targets",
    "pure_chain_targets", "bn_extract_targets",
    "admission_floor_refusals", "deadline_route_refusals", "slice_refusals",
    "no_target_refusals", "capability_refusals", "execution_refusals",
})
_ROOT_JOINT_INTERM_ALPHA_INTEGER_FIELDS = frozenset({
    "configured_secs", "configured_iters",
    "configured_lr_bits", "configured_min_remaining_secs",
    "configured_max_sel", "invocations",
    "max_effective_iters",
    "input_split_invocations", "multi_objective_invocations", "target_count",
    "available_budget_ms", "effective_slice_ms",
    "configured_slice_work_invocations",
    "configured_iters_binding_invocations",
    "lowered_min_remaining_work_invocations", "work_started", "attempts",
    "raised_min_remaining_binding_refusals",
    "completed", "tightened_count", "outlier_tightened_count",
    "master_disabled",
    "target_attempts", "base_folds_completed", "ascent_iterations_attempted",
    "max_ascent_iterations_attempted", "optimizer_steps",
    "ascent_folds_completed", "max_selected_rows",
    "selection_cap_bound_targets", "frozen_stop_targets",
    "pure_chain_targets", "bn_extract_targets",
    "admission_floor_refusals", "deadline_route_refusals", "slice_refusals",
    "no_target_refusals", "capability_refusals", "execution_refusals",
})


def _root_joint_interm_alpha_contract_valid(fields, status):
    """Validate one independently closed root-alpha aggregate exactly."""
    if (
        set(fields) != _ROOT_JOINT_INTERM_ALPHA_EVENT_FIELDS
        or fields.get("schema") != _ROOT_JOINT_INTERM_ALPHA_EVENT_SCHEMA
        or fields["phase"] != "terminal"
        or fields["observation_schema"]
            != _ROOT_JOINT_INTERM_ALPHA_OBSERVATION_SCHEMA
        or fields["route"] not in {"input_split", "multi_objective", "mixed"}
        or fields["configured_master"] not in {"true", "false"}
        or fields["configured_deadline_ascent"] not in {"true", "false"}
        or fields["configured_frozen_stop"] not in {"true", "false"}
        or fields["configured_pure_chain"] not in {"true", "false"}
        or fields["configured_bn_gpu_extract"] not in {"true", "false"}
        or status not in {"ran", "skipped"}
    ):
        return False
    integers = {}
    for name in _ROOT_JOINT_INTERM_ALPHA_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        integers[name] = value
    optional_integers = {}
    for name in (
        "configured_max_dim_override",
        "input_split_configured_max_dim",
        "multi_objective_configured_max_dim",
    ):
        token = fields[name]
        value = None if token == "none" else _canonical_event_u64(token)
        if token != "none" and value is None:
            return False
        optional_integers[name] = value
    lr_bits = integers["configured_lr_bits"]
    if lr_bits > (1 << 32) - 1:
        return False
    configured_lr = struct.unpack(">f", struct.pack(">I", lr_bits))[0]
    if not math.isfinite(configured_lr):
        return False

    invocations = integers["invocations"]
    input_invocations = integers["input_split_invocations"]
    multi_invocations = integers["multi_objective_invocations"]
    attempts = integers["attempts"]
    completed = integers["completed"]
    target_attempts = integers["target_attempts"]
    base_folds = integers["base_folds_completed"]
    effective_iters = integers["max_effective_iters"]
    ascent_attempts = integers["ascent_iterations_attempted"]
    max_ascent_attempts = integers["max_ascent_iterations_attempted"]
    optimizer_steps = integers["optimizer_steps"]
    ascent_folds = integers["ascent_folds_completed"]
    configured_slice_work = integers["configured_slice_work_invocations"]
    configured_iters_binding = integers[
        "configured_iters_binding_invocations"]
    lowered_floor_work = integers[
        "lowered_min_remaining_work_invocations"]
    raised_floor_refusals = integers[
        "raised_min_remaining_binding_refusals"]
    max_selected = integers["max_selected_rows"]
    selection_cap_bound = integers["selection_cap_bound_targets"]
    frozen_stop_targets = integers["frozen_stop_targets"]
    pure_chain_targets = integers["pure_chain_targets"]
    bn_extract_targets = integers["bn_extract_targets"]
    targets = integers["target_count"]
    tightened = integers["tightened_count"]
    outlier_tightened = integers["outlier_tightened_count"]
    classified = sum(integers[name] for name in (
        "completed", "master_disabled", "admission_floor_refusals",
        "deadline_route_refusals", "slice_refusals", "no_target_refusals",
        "capability_refusals", "execution_refusals",
    ))
    route_matches = {
        "input_split": input_invocations > 0 and multi_invocations == 0,
        "multi_objective": multi_invocations > 0 and input_invocations == 0,
        "mixed": input_invocations > 0 and multi_invocations > 0,
    }[fields["route"]]
    master = fields["configured_master"] == "true"
    if (
        invocations == 0
        or input_invocations + multi_invocations != invocations
        or classified != invocations
        or not route_matches
        or (input_invocations > 0)
            != (optional_integers["input_split_configured_max_dim"] is not None)
        or (multi_invocations > 0)
            != (optional_integers["multi_objective_configured_max_dim"] is not None)
        or optional_integers["input_split_configured_max_dim"] == 0
        or optional_integers["multi_objective_configured_max_dim"] == 0
        or (optional_integers["configured_max_dim_override"] is not None and (
            optional_integers["configured_max_dim_override"] == 0
            or (optional_integers["input_split_configured_max_dim"] is not None
                and optional_integers["input_split_configured_max_dim"]
                    != optional_integers["configured_max_dim_override"])
            or (optional_integers["multi_objective_configured_max_dim"] is not None
                and optional_integers["multi_objective_configured_max_dim"]
                    != optional_integers["configured_max_dim_override"])
        ))
        or (optional_integers["configured_max_dim_override"] is None and (
            optional_integers["input_split_configured_max_dim"] not in {None, 32768}
            or optional_integers["multi_objective_configured_max_dim"] not in {
                None,
                32768 if master and fields["configured_deadline_ascent"] == "true"
                else 2048,
            }
        ))
        or attempts != completed
        or attempts > invocations
        or integers["work_started"] > invocations
        or configured_slice_work > integers["work_started"]
        or configured_iters_binding > integers["work_started"]
        or lowered_floor_work > integers["work_started"]
        or raised_floor_refusals > integers["admission_floor_refusals"]
        or (configured_slice_work > 0 and (
            integers["configured_secs"] == 0
            or integers["configured_secs"] > ((1 << 64) - 1) // 1000
            or integers["effective_slice_ms"]
                < integers["configured_secs"] * 1000
        ))
        or (configured_iters_binding > 0 and (
            integers["configured_iters"] == 0
            or effective_iters != integers["configured_iters"]
            or max_ascent_attempts != integers["configured_iters"]
        ))
        or (integers["work_started"] > 0) != (target_attempts > 0)
        or target_attempts > targets
        or base_folds > target_attempts
        or tightened > base_folds
        or (outlier_tightened > 0
            and (input_invocations == 0 or attempts == 0))
        or (target_attempts > 0
            and effective_iters > ((1 << 64) - 1) // target_attempts)
        or ascent_attempts > target_attempts * effective_iters
        or max_ascent_attempts > effective_iters
        or max_ascent_attempts > ascent_attempts
        or optimizer_steps > ascent_attempts
        or ascent_folds > optimizer_steps
        or selection_cap_bound > target_attempts
        or frozen_stop_targets > target_attempts
        or pure_chain_targets > target_attempts
        or bn_extract_targets > target_attempts
        or max_selected > integers["configured_max_sel"]
        or (selection_cap_bound > 0
            and max_selected != integers["configured_max_sel"])
        or (integers["configured_max_sel"] == 0
            and (max_selected > 0 or selection_cap_bound > 0))
        or (fields["configured_frozen_stop"] == "false"
            and frozen_stop_targets > 0)
        or (fields["configured_pure_chain"] == "false"
            and pure_chain_targets > 0)
        or (pure_chain_targets > 0
            and fields["configured_frozen_stop"] != "false")
        or (fields["configured_bn_gpu_extract"] == "false"
            and bn_extract_targets > 0)
        or (lowered_floor_work > 0 and (
            multi_invocations == 0
            or integers["configured_min_remaining_secs"] >= 240
        ))
        or (raised_floor_refusals > 0 and (
            multi_invocations == 0
            or integers["configured_min_remaining_secs"] <= 240
        ))
        or (effective_iters == 0 and (
            ascent_attempts > 0 or max_ascent_attempts > 0
            or optimizer_steps > 0 or ascent_folds > 0
        ))
        or tightened > targets
        or integers["effective_slice_ms"] > integers["available_budget_ms"]
        or (attempts > 0 and (
            not master or targets == 0 or integers["effective_slice_ms"] == 0
        ))
        or (not master and (
            target_attempts > 0 or tightened > 0 or outlier_tightened > 0
        ))
        or (master and integers["master_disabled"] > 0)
        or (not master and integers["master_disabled"] != invocations)
        or (status == "ran") != (attempts > 0)
    ):
        return False
    return True


def _root_joint_interm_alpha_terminal_evidences(row):
    """Return the one strict root-alpha event, poisoning any ambiguity."""
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if not isinstance(event, dict) or event.get("method") \
                != "root_joint_interm_alpha_execution":
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii()
            or not reason
            or reason != reason.strip(" ")
            or "  " in reason
            or any(character.isspace() and character != " " for character in reason)
        ):
            return []
        tokens = reason.split(" ")
        if not tokens or tokens[0] != (
            "schema=" + _ROOT_JOINT_INTERM_ALPHA_EVENT_SCHEMA
        ):
            return []
        fields = {}
        for token in tokens:
            if token.count("=") != 1:
                return []
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                return []
            fields[key] = value
        if not _root_joint_interm_alpha_contract_valid(fields, event.get("status")):
            return []
        fields["_status"] = event["status"]
        records.append(fields)
    # The wrapper converts one terminal verifier JSON exactly once. Replays or
    # duplicate events are attribution ambiguity, not extra evidence.
    return records if len(records) == 1 else []


def _requested_f32_bits(token):
    parsed = _parsed_ascii_f32(token)
    if parsed is None:
        return None
    return struct.unpack(">I", struct.pack(">f", parsed))[0]


def _root_joint_interm_alpha_parameter_engagement(wanted, rows):
    """Prove root-alpha parameters reached one real bounded tightener call."""
    requested = {
        name: token for name, token in wanted
        if name in _ROOT_JOINT_INTERM_ALPHA_T1_FAMILY
    }
    if not requested:
        return set()
    engaged = set()
    for row in rows:
        for event in _root_joint_interm_alpha_terminal_evidences(row):
            invocations = _canonical_event_u64(event["invocations"])
            targets = _canonical_event_u64(event["target_count"])
            attempts = _canonical_event_u64(event["attempts"])
            completed = _canonical_event_u64(event["completed"])
            target_attempts = _canonical_event_u64(event["target_attempts"])
            base_folds = _canonical_event_u64(event["base_folds_completed"])
            tightened = _canonical_event_u64(event["tightened_count"])
            outlier_tightened = _canonical_event_u64(
                event["outlier_tightened_count"])
            if invocations is None or invocations == 0 \
                    or event["configured_master"] != "true":
                continue
            live_work = (
                event["_status"] == "ran"
                and targets is not None and targets > 0
                and attempts is not None and attempts > 0
                and completed is not None and completed > 0
                and target_attempts is not None and target_attempts > 0
                and base_folds is not None and base_folds > 0
                and tightened is not None
                and outlier_tightened is not None
                and (outlier_tightened == 0 or tightened > 0)
            )

            matches = {
                "NY_ROOT_JOINT_INTERM_ALPHA": (
                    requested.get("NY_ROOT_JOINT_INTERM_ALPHA") == "1"
                    and live_work
                ),
                "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT": (
                    requested.get("NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT") == "1"
                    and live_work
                    and event["configured_deadline_ascent"] == "true"
                    and _canonical_event_u64(event["max_effective_iters"]) > 0
                    and _canonical_event_u64(
                        event["ascent_iterations_attempted"]) > 0
                ),
            }
            for name, field in (
                ("NY_ROOT_JOINT_INTERM_ALPHA_SECS", "configured_secs"),
                ("NY_ROOT_JOINT_INTERM_ALPHA_ITERS", "configured_iters"),
                ("NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL", "configured_max_sel"),
                ("NY_ROOT_JOINT_MIN_REMAINING_SECS",
                 "configured_min_remaining_secs"),
            ):
                token = requested.get(name)
                expected = _canonical_event_u64(token)
                actual = _canonical_event_u64(event[field])
                matches[name] = token is not None and expected is not None \
                    and actual == expected and (
                        live_work or name == "NY_ROOT_JOINT_MIN_REMAINING_SECS"
                    )
            lr_token = requested.get("NY_ROOT_JOINT_INTERM_ALPHA_LR")
            matches["NY_ROOT_JOINT_INTERM_ALPHA_LR"] = (
                lr_token is not None
                and live_work
                and _requested_f32_bits(lr_token)
                    == _canonical_event_u64(event["configured_lr_bits"])
                and _canonical_event_u64(event["optimizer_steps"]) > 0
            )
            for name, configured_field, effect_field in (
                ("NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP",
                 "configured_frozen_stop", "frozen_stop_targets"),
                ("NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN",
                 "configured_pure_chain", "pure_chain_targets"),
                ("NY_BN_GPU_EXTRACT",
                 "configured_bn_gpu_extract", "bn_extract_targets"),
            ):
                matches[name] = (
                    requested.get(name) == "1"
                    and live_work
                    and event[configured_field] == "true"
                    and _canonical_event_u64(event[effect_field]) > 0
                )
            if "NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN" in requested:
                matches["NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN"] &= (
                    event["configured_frozen_stop"] == "false"
                )

            if "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL" in requested:
                matches["NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL"] &= (
                    _canonical_event_u64(
                        event["selection_cap_bound_targets"]) > 0
                    and _canonical_event_u64(event["max_selected_rows"])
                        == _canonical_event_u64(event["configured_max_sel"])
                )

            if "NY_ROOT_JOINT_INTERM_ALPHA_ITERS" in requested:
                requested_iters = _canonical_event_u64(
                    requested["NY_ROOT_JOINT_INTERM_ALPHA_ITERS"])
                matches["NY_ROOT_JOINT_INTERM_ALPHA_ITERS"] &= (
                    requested_iters is not None
                    and _canonical_event_u64(
                        event["configured_iters_binding_invocations"]) > 0
                )

            # A capped slice proves work, but not that the requested seconds
            # changed that work. Only a slice actually bound by configured SECS
            # can graduate the seconds parameter itself.
            if "NY_ROOT_JOINT_INTERM_ALPHA_SECS" in requested:
                matches["NY_ROOT_JOINT_INTERM_ALPHA_SECS"] &= (
                    _canonical_event_u64(
                        event["configured_slice_work_invocations"]) > 0
                )

            # This admission floor is not consulted on input split. A lowered
            # floor earns credit only in the live window where the historical
            # 240-second policy would have refused. A raised floor earns credit
            # from the exact terminal refusal window where 240 would admit.
            if "NY_ROOT_JOINT_MIN_REMAINING_SECS" in requested:
                matches["NY_ROOT_JOINT_MIN_REMAINING_SECS"] &= (
                    (live_work and _canonical_event_u64(
                        event["lowered_min_remaining_work_invocations"]) > 0)
                    or _canonical_event_u64(
                        event["raised_min_remaining_binding_refusals"]) > 0
                )

            for name in requested:
                if matches.get(name, False):
                    engaged.add(name)
    return engaged


def _root_joint_explicit_configuration_matches(event, assignment):
    """Bind a closed root terminal to every explicit root control in its arm.

    The layered receipt separately authenticates delivery.  This check closes
    the final gap for prerequisite-only controls: telemetry saying the master
    was disabled cannot authenticate an arm whose receipt says master=1, even
    when the telemetry is otherwise a truthful zero-work refusal.
    """
    requested = dict(assignment)
    boolean_fields = {
        "NY_ROOT_JOINT_INTERM_ALPHA": "configured_master",
        "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT":
            "configured_deadline_ascent",
        "NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP": "configured_frozen_stop",
        "NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN": "configured_pure_chain",
        "NY_BN_GPU_EXTRACT": "configured_bn_gpu_extract",
    }
    integer_fields = {
        "NY_ROOT_JOINT_INTERM_ALPHA_SECS": "configured_secs",
        "NY_ROOT_JOINT_INTERM_ALPHA_ITERS": "configured_iters",
        "NY_ROOT_JOINT_MIN_REMAINING_SECS": "configured_min_remaining_secs",
        "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL": "configured_max_sel",
        "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM": "configured_max_dim_override",
    }
    for name, field in boolean_fields.items():
        if name not in requested:
            continue
        expected = {"0": "false", "1": "true"}.get(requested[name])
        if expected is None or event.get(field) != expected:
            return False
    for name, field in integer_fields.items():
        if name not in requested:
            continue
        expected = _canonical_event_u64(requested[name])
        actual = _canonical_event_u64(event.get(field))
        if expected is None or actual != expected:
            return False
    if "NY_ROOT_JOINT_INTERM_ALPHA_LR" in requested:
        if _requested_f32_bits(requested["NY_ROOT_JOINT_INTERM_ALPHA_LR"]) \
                != _canonical_event_u64(event.get("configured_lr_bits")):
            return False
    return True


_EXACT_VJP_FAMILY = frozenset({
    "NY_PGD_EXACT_BATCHED",
    "NY_PGD_VJP_BATCH",
    "NY_PGD_VJP_K",
    "NY_PGD_LEGACY_LANES",
    "NY_PGD_ODI_STEPS",
    "NY_PGD_ODI_ETA",
    "NY_PGD_VJP_STEPS",
    "NY_PGD_LANE_WINDOW",
    "NY_PGD_EXPLOIT",
    "NY_PGD_HINGE_TAU",
})

_BNN_FALSIFICATION_FAMILY = frozenset({
    "NY_BNN_SIGN_SPACE",
    "NY_BNN_SIGN_SPACE_MINIMAL_MOVE",
    "NY_BNN_SIGN_SPACE_TRUST_REGION",
    "NY_BNN_STE_PGD",
})
_BNN_FALSIFICATION_POLICY_EVENT_SCHEMA = (
    "ny-cli/bnn-falsification-policy/v1"
)
_BNN_FALSIFICATION_EXECUTION_EVENT_SCHEMA = (
    "ny-cli/bnn-falsification-execution/v1"
)
_BNN_FALSIFICATION_POLICY_EVENT_FIELDS = frozenset({
    "schema", "mip_compiled", "sign_space_enabled", "minimal_move",
    "trust_region", "ste_pgd_enabled", "lower_sign_space_enabled",
    "lower_minimal_move", "lower_trust_region", "lower_ste_pgd_enabled",
})
_BNN_FALSIFICATION_EXECUTION_EVENT_FIELDS = frozenset({
    "schema", "lane", "route", "outcome", "sign_space_enabled",
    "minimal_move", "trust_region", "ste_pgd_enabled", "work_started",
    "budget_ms", "opportunity_budget_ms", "free_units", "flips",
    "work_units",
})
_BNN_FALSIFICATION_BOOLEAN_POLICY_FIELDS = frozenset({
    "mip_compiled", "sign_space_enabled", "minimal_move", "ste_pgd_enabled",
    "lower_sign_space_enabled", "lower_minimal_move",
    "lower_ste_pgd_enabled",
})
_BNN_FALSIFICATION_TRUST_REGIONS = frozenset({
    "full_box", "box", "tight", "linf",
})
_BNN_FALSIFICATION_OUTCOMES = frozenset({
    "disarmed", "not_admitted", "core_refused", "exhausted",
    "solver_error", "candidate_reforwarded",
    "candidate_reforward_failed", "window_skipped", "budget_unavailable",
})
_BNN_FALSIFICATION_MIN_BUDGET_MS = 20_000
_BNN_FALSIFICATION_MAX_BUDGET_MS = 240_000


def _bnn_falsification_reason_fields(event, *, method, schema, fields):
    """Parse one exact canonical BNN event, or poison its containing row."""
    if not isinstance(event, dict) or event.get("method") != method:
        return None
    reason = event.get("reason")
    if (
        not isinstance(reason, str)
        or not reason.isascii()
        or not reason
        or reason != reason.strip(" ")
        or "  " in reason
        or any(character.isspace() and character != " " for character in reason)
    ):
        return False
    tokens = reason.split(" ")
    if not tokens or tokens[0] != "schema=" + schema:
        return False
    parsed = {}
    for token in tokens:
        if token.count("=") != 1:
            return False
        key, value = token.split("=", 1)
        if not key or not value or key in parsed:
            return False
        parsed[key] = value
    if set(parsed) != fields:
        return False
    parsed["_status"] = event.get("status")
    return parsed


def _bnn_falsification_policy_valid(policy):
    return (
        policy["_status"] == "ran"
        and all(
            policy[name] in {"true", "false"}
            for name in _BNN_FALSIFICATION_BOOLEAN_POLICY_FIELDS
        )
        and policy["trust_region"] in _BNN_FALSIFICATION_TRUST_REGIONS
        and policy["lower_trust_region"] in _BNN_FALSIFICATION_TRUST_REGIONS
    )


def _bnn_falsification_execution_valid(execution, policy):
    if (
        execution["lane"] not in {"sign_space", "ste_pgd"}
        or execution["route"]
        not in {"private", "committed_cap", "committed_skip", "scheduled"}
        or execution["outcome"] not in _BNN_FALSIFICATION_OUTCOMES
        or execution["sign_space_enabled"] not in {"true", "false"}
        or execution["minimal_move"] not in {"true", "false"}
        or execution["trust_region"] not in _BNN_FALSIFICATION_TRUST_REGIONS
        or execution["ste_pgd_enabled"] not in {"true", "false"}
        or execution["work_started"] not in {"true", "false"}
        or any(
            execution[name] != policy[name]
            for name in (
                "sign_space_enabled", "minimal_move", "trust_region",
                "ste_pgd_enabled",
            )
        )
    ):
        return False
    work_started = execution["work_started"] == "true"
    if execution["_status"] != ("ran" if work_started else "skipped"):
        return False
    budget = (
        None
        if execution["budget_ms"] == "none"
        else _canonical_event_u64(execution["budget_ms"])
    )
    opportunity_budget = (
        None
        if execution["opportunity_budget_ms"] == "none"
        else _canonical_event_u64(execution["opportunity_budget_ms"])
    )
    if execution["budget_ms"] != "none" and budget is None:
        return False
    if (
        execution["opportunity_budget_ms"] != "none"
        and opportunity_budget is None
    ):
        return False
    if any(
        value is not None
        and not (
            _BNN_FALSIFICATION_MIN_BUDGET_MS
            <= value
            <= _BNN_FALSIFICATION_MAX_BUDGET_MS
        )
        for value in (budget, opportunity_budget)
    ):
        return False
    numbers = {
        name: _canonical_event_u64(execution[name])
        for name in ("free_units", "flips", "work_units")
    }
    if any(value is None for value in numbers.values()):
        return False
    if numbers["flips"] > numbers["free_units"]:
        return False
    if budget is not None and budget != opportunity_budget:
        return False
    if work_started:
        if budget is None or budget <= 0:
            return False
    elif any(numbers.values()):
        return False

    outcome = execution["outcome"]
    route = execution["route"]
    lane_enabled = execution[
        "sign_space_enabled" if execution["lane"] == "sign_space"
        else "ste_pgd_enabled"
    ] == "true"
    if outcome == "disarmed":
        return (
            not work_started and not lane_enabled and budget is None
            and route in {"private", "committed_cap", "scheduled"}
            and (route != "scheduled" or opportunity_budget is None)
        )
    if outcome == "window_skipped":
        return (
            not work_started and route == "committed_skip"
            and budget is None and opportunity_budget is None
        )
    if outcome == "budget_unavailable":
        return (
            not work_started
            and lane_enabled
            and route in {"committed_cap", "scheduled"}
            and budget is None and opportunity_budget is None
        )
    if route == "committed_skip" or not lane_enabled:
        return False
    if outcome in {"core_refused", "solver_error"}:
        return work_started and not any(numbers.values())
    if outcome == "not_admitted":
        return not work_started
    return work_started


def _bnn_falsification_evidence(row):
    """Return one strict policy plus at most one terminal for each BNN lane.

    Any malformed matching event, duplicate policy, or duplicate lane terminal
    poisons the whole row. This is deliberately row-local: a disarm decision
    can never borrow sibling work from another scored instance.
    """
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return None
    policies = []
    executions = []
    for event in events:
        policy = _bnn_falsification_reason_fields(
            event,
            method="bnn_falsification_policy",
            schema=_BNN_FALSIFICATION_POLICY_EVENT_SCHEMA,
            fields=_BNN_FALSIFICATION_POLICY_EVENT_FIELDS,
        )
        if policy is False:
            return None
        if policy is not None:
            policies.append(policy)
            continue
        execution = _bnn_falsification_reason_fields(
            event,
            method="bnn_falsification_execution",
            schema=_BNN_FALSIFICATION_EXECUTION_EVENT_SCHEMA,
            fields=_BNN_FALSIFICATION_EXECUTION_EVENT_FIELDS,
        )
        if execution is False:
            return None
        if execution is not None:
            if not policies:
                return None
            executions.append(execution)
    if len(policies) != 1 or not _bnn_falsification_policy_valid(policies[0]):
        return None
    by_lane = {}
    for execution in executions:
        lane = execution["lane"]
        if lane in by_lane or not _bnn_falsification_execution_valid(
            execution, policies[0]
        ):
            return None
        by_lane[lane] = execution
    return {"policy": policies[0], "executions": by_lane}


def _bnn_falsification_configuration_matches(policy, requested):
    fields = {
        "NY_BNN_SIGN_SPACE": "sign_space_enabled",
        "NY_BNN_SIGN_SPACE_MINIMAL_MOVE": "minimal_move",
        "NY_BNN_STE_PGD": "ste_pgd_enabled",
    }
    for name, field in fields.items():
        if name not in requested:
            continue
        expected = {"0": "false", "1": "true"}.get(requested[name])
        if expected is None or policy[field] != expected:
            return False
    trust = requested.get("NY_BNN_SIGN_SPACE_TRUST_REGION")
    return trust is None or (
        trust in _BNN_FALSIFICATION_TRUST_REGIONS
        and policy["trust_region"] == trust
    )


def _bnn_falsification_substantive(execution):
    budget = _canonical_event_u64(execution.get("budget_ms"))
    work_units = _canonical_event_u64(execution.get("work_units"))
    return (
        execution.get("work_started") == "true"
        and budget is not None and budget > 0
        and work_units is not None and work_units > 0
    )


def _bnn_falsification_parameter_engagement(wanted, rows):
    """Credit only causally bound work under one immutable four-axis policy."""
    requested = {
        name: token for name, token in wanted
        if name in _BNN_FALSIFICATION_FAMILY
    }
    if not requested:
        return set()
    engaged = set()
    for row in rows:
        evidence = _bnn_falsification_evidence(row)
        if evidence is None:
            continue
        policy = evidence["policy"]
        executions = evidence["executions"]
        if (
            policy["mip_compiled"] != "true"
            or not _bnn_falsification_configuration_matches(policy, requested)
        ):
            continue

        for axis, lane, enabled_field, lower_field in (
            (
                "NY_BNN_SIGN_SPACE", "sign_space", "sign_space_enabled",
                "lower_sign_space_enabled",
            ),
            (
                "NY_BNN_STE_PGD", "ste_pgd", "ste_pgd_enabled",
                "lower_ste_pgd_enabled",
            ),
        ):
            token = requested.get(axis)
            target = executions.get(lane)
            if token == "1" and target is not None and (
                policy[enabled_field] == "true"
                and policy[lower_field] == "false"
                and _bnn_falsification_substantive(target)
            ):
                engaged.add(axis)
            if token == "0" and target is not None:
                sibling = executions.get(
                    "ste_pgd" if lane == "sign_space" else "sign_space"
                )
                if (
                    policy[enabled_field] == "false"
                    and policy[lower_field] == "true"
                    and target["outcome"] == "disarmed"
                    and (_canonical_event_u64(
                        target["opportunity_budget_ms"]
                    ) or 0) > 0
                    and sibling is not None
                    and target["route"] in {"private", "committed_cap"}
                    and sibling["route"] in {"private", "committed_cap"}
                    and _bnn_falsification_substantive(sibling)
                ):
                    engaged.add(axis)

        sign = executions.get("sign_space")
        if (
            sign is not None
            and policy["sign_space_enabled"] == "true"
            and _canonical_event_u64(sign["flips"]) > 0
            and _canonical_event_u64(sign["work_units"]) > 0
        ):
            if (
                "NY_BNN_SIGN_SPACE_MINIMAL_MOVE" in requested
                and policy["minimal_move"] != policy["lower_minimal_move"]
            ):
                engaged.add("NY_BNN_SIGN_SPACE_MINIMAL_MOVE")
            if (
                "NY_BNN_SIGN_SPACE_TRUST_REGION" in requested
                and policy["trust_region"] != policy["lower_trust_region"]
            ):
                engaged.add("NY_BNN_SIGN_SPACE_TRUST_REGION")
    return engaged


_MO_KFSB_EVENT_SCHEMA = "ny-cli/mo-kfsb-execution/v1"
_MO_KFSB_OBSERVATION_SCHEMA = "ny_beta_crown_mo_kfsb_observations_v1"
_MO_KFSB_MAX_LEAVES_PER_PARENT = 1024
_MO_KFSB_EVENT_FIELDS = frozenset({
    "schema", "phase", "observation_schema", "canonical_master",
    "legacy_childsim", "configured_master", "configured_candidates",
    "effective_candidates", "configured_reduce_op", "effective_reduce_op",
    "heuristic_eligible", "branch_opportunities", "offered_domains",
    "nonempty_domain_opportunities",
    "candidate_eligible_opportunities", "typed_select_opportunities",
    "lower_master_opportunities", "effective_master_opportunities",
    "route_eligible_opportunities", "selector_waves", "selector_domains",
    "committed_waves", "committed_parents", "committed_leaves",
    "force_on_binding_waves", "master_binding_refusals",
})
_MO_KFSB_INTEGER_FIELDS = frozenset({
    "configured_candidates", "effective_candidates", "branch_opportunities",
    "offered_domains", "nonempty_domain_opportunities",
    "candidate_eligible_opportunities", "typed_select_opportunities",
    "lower_master_opportunities", "effective_master_opportunities",
    "route_eligible_opportunities", "selector_waves", "selector_domains",
    "committed_waves", "committed_parents", "committed_leaves",
    "force_on_binding_waves", "master_binding_refusals",
})


def _mo_kfsb_execution_contract_valid(fields, status):
    """Validate one closed MO-kFSB terminal without trusting its producer."""
    if (
        set(fields) != _MO_KFSB_EVENT_FIELDS
        or fields.get("schema") != _MO_KFSB_EVENT_SCHEMA
        or fields.get("phase") != "terminal"
        or fields.get("observation_schema") != _MO_KFSB_OBSERVATION_SCHEMA
        or fields.get("canonical_master") not in {
            "absent", "exact_enable", "exact_disable", "rejected",
        }
        or fields.get("legacy_childsim") not in {"true", "false"}
        or fields.get("configured_master") not in {"true", "false"}
        or fields.get("configured_reduce_op") not in {"min", "max", "mean"}
        or fields.get("effective_reduce_op") not in {"min", "max", "mean"}
        or (
            fields.get("effective_reduce_op") == "mean"
            and fields.get("configured_reduce_op") != "mean"
        )
        or fields.get("heuristic_eligible") not in {"true", "false"}
    ):
        return False
    numbers = {
        name: _canonical_event_u64(fields.get(name))
        for name in _MO_KFSB_INTEGER_FIELDS
    }
    if any(value is None for value in numbers.values()):
        return False

    branch = numbers["branch_opportunities"]
    offered = numbers["offered_domains"]
    nonempty = numbers["nonempty_domain_opportunities"]
    candidate = numbers["candidate_eligible_opportunities"]
    typed = numbers["typed_select_opportunities"]
    lower = numbers["lower_master_opportunities"]
    effective = numbers["effective_master_opportunities"]
    route = numbers["route_eligible_opportunities"]
    selector_waves = numbers["selector_waves"]
    selector_domains = numbers["selector_domains"]
    committed_waves = numbers["committed_waves"]
    committed_parents = numbers["committed_parents"]
    committed_leaves = numbers["committed_leaves"]
    force_on = numbers["force_on_binding_waves"]
    refusals = numbers["master_binding_refusals"]
    effective_candidates = numbers["effective_candidates"]
    canonical = fields["canonical_master"]
    legacy = fields["legacy_childsim"] == "true"
    configured_master = fields["configured_master"] == "true"
    heuristic = fields["heuristic_eligible"] == "true"

    # An emitted event comes only from an observed, non-conflicting section.
    # All opportunity counters are invocation counts; domain counters are sums.
    if (
        branch == 0
        or any(value > branch for value in (
            nonempty, candidate, typed, lower, effective, route,
            selector_waves, committed_waves, force_on, refusals,
        ))
        or nonempty != branch
        or nonempty > offered
        or selector_domains > offered
        or selector_waves > selector_domains
        or committed_waves > selector_waves
        or committed_waves > route
        or committed_waves > nonempty
        or committed_waves > committed_parents
        or committed_parents > selector_domains
        or (committed_waves > 0) != (committed_parents > 0)
        or (committed_parents == 0 and committed_leaves > 0)
        or committed_parents > (
            ((1 << 64) - 1) // _MO_KFSB_MAX_LEAVES_PER_PARENT
        )
        or committed_leaves > (
            _MO_KFSB_MAX_LEAVES_PER_PARENT * committed_parents
        )
        or force_on > committed_waves
        or force_on > branch - lower
        or refusals > lower
        or refusals > route
        or (effective_candidates > 0) != (candidate == branch)
        or route > nonempty
        or route > candidate
        or typed > candidate
        or (typed > 0 and fields["effective_reduce_op"] != "min")
        or (not heuristic and (typed > 0 or route > 0 or selector_waves > 0))
        or selector_waves > effective
        or selector_waves > candidate
        or selector_waves > route
        or selector_waves < max(0, route - (branch - effective))
        or (selector_domains > 0 and selector_waves == 0)
        or (selector_domains > 0 and route == 0)
        or selector_domains > offered - (branch - selector_waves)
        or (selector_waves == branch and selector_domains != offered)
        or (
            committed_waves <= selector_waves
            and committed_parents > (
                selector_domains - (selector_waves - committed_waves)
            )
        )
        or (
            canonical == "exact_enable"
            and force_on < max(0, committed_waves - lower)
        )
        or (
            canonical == "exact_disable"
            and refusals < max(0, route - (branch - lower))
        )
        or status != ("ran" if selector_waves > 0 else "skipped")
    ):
        return False

    # The lower master is the frozen preset/typed policy before the canonical
    # environment override. A configured master is on at every opportunity;
    # otherwise only round-eligible typed Select opportunities arm it.
    expected_lower = branch if configured_master else typed
    if lower != expected_lower:
        return False

    # Canonical presence suppresses the absence-only legacy switch. Exact 1/0
    # override the lower policy in either direction; rejected input is the dark
    # declaration default and can never masquerade as an effective treatment.
    if legacy and canonical != "absent":
        return False
    expected_effective = {
        "exact_enable": branch,
        "exact_disable": 0,
        "rejected": 0,
        "absent": branch if legacy else lower,
    }[canonical]
    if effective != expected_effective:
        return False
    if canonical != "exact_enable" and force_on != 0:
        return False
    if canonical != "exact_disable" and refusals != 0:
        return False
    if effective_candidates == 0 and (
        route > 0 or selector_domains > 0 or committed_parents > 0
    ):
        return False
    return True


def _mo_kfsb_execution_evidence(row):
    """Return exactly one strict MO-kFSB terminal, or poison the row."""
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return None
    matches = []
    for event in events:
        if not isinstance(event, dict) or event.get("method") != "mo_kfsb_execution":
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii()
            or not reason
            or reason != reason.strip(" ")
            or "  " in reason
            or any(character.isspace() and character != " " for character in reason)
        ):
            return None
        fields = {}
        for token in reason.split(" "):
            if token.count("=") != 1:
                return None
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                return None
            fields[key] = value
        if not _mo_kfsb_execution_contract_valid(fields, event.get("status")):
            return None
        fields["_status"] = event["status"]
        matches.append(fields)
    return matches[0] if len(matches) == 1 else None


def _mo_kfsb_parameter_engagement(wanted, rows):
    """Credit only master/reducer changes with row-local causal witnesses."""
    requested = dict(wanted)
    requested_master = requested.get("NY_MO_KFSB")
    requested_reducer = requested.get("NY_MO_KFSB_REDUCE")
    if (
        not rows
        or (
            requested_master not in {"0", "1"}
            and requested_reducer not in {"min", "max"}
        )
    ):
        return set()
    evidence = []
    for row in rows:
        event = _mo_kfsb_execution_evidence(row)
        if event is None:
            return set()
        evidence.append(event)

    engaged = set()
    if requested_master in {"0", "1"}:
        expected_canonical = (
            "exact_enable" if requested_master == "1" else "exact_disable"
        )
        if all(
            event["canonical_master"] == expected_canonical
            for event in evidence
        ):
            if requested_master == "1" and any(
                _canonical_event_u64(event["force_on_binding_waves"]) > 0
                and _canonical_event_u64(event["selector_waves"]) > 0
                and _canonical_event_u64(event["selector_domains"]) > 0
                and _canonical_event_u64(event["committed_waves"]) > 0
                and _canonical_event_u64(event["committed_parents"]) > 0
                for event in evidence
            ):
                engaged.add("NY_MO_KFSB")
            if requested_master == "0" and any(
                _canonical_event_u64(event["lower_master_opportunities"]) > 0
                and _canonical_event_u64(
                    event["candidate_eligible_opportunities"]
                ) > 0
                and _canonical_event_u64(event["route_eligible_opportunities"]) > 0
                and _canonical_event_u64(event["master_binding_refusals"]) > 0
                for event in evidence
            ):
                engaged.add("NY_MO_KFSB")

    # The reducer is captured once with the wave policy and passed through the
    # selector. A nonempty committed parent map therefore proves that the exact
    # effective reducer ranked real simulated children. Require a real
    # counterfactual override and live work independently on every row: a
    # configured no-op, selector miss, or inactive companion row cannot borrow
    # credit from another row. The reducer is a child of the canonical master,
    # so its master authority must also match every terminal: an explicit
    # master pair is an exact environment enable, while an omitted pair means
    # the typed config was already armed and the canonical environment was
    # absent.
    explicit_master = requested_master == "1"
    typed_master = requested_master is None
    if (
        requested_reducer in {"min", "max"}
        and (explicit_master or typed_master)
        and all(
            (
                event["canonical_master"] == "exact_enable"
                if explicit_master
                else (
                    event["canonical_master"] == "absent"
                    and event["configured_master"] == "true"
                    and event["legacy_childsim"] == "false"
                )
            )
            and event["effective_reduce_op"] == requested_reducer
            and event["configured_reduce_op"] != requested_reducer
            and event["configured_reduce_op"]
            != event["effective_reduce_op"]
            and _canonical_event_u64(event["selector_waves"]) > 0
            and _canonical_event_u64(event["selector_domains"]) > 0
            and _canonical_event_u64(event["committed_waves"]) > 0
            and _canonical_event_u64(event["committed_parents"]) > 0
            for event in evidence
        )
    ):
        engaged.add("NY_MO_KFSB_REDUCE")
    return engaged


_MO_ALPHA_STEP_EVENT_SCHEMA = "ny-cli/mo-alpha-step-execution/v1"
_MO_ALPHA_STEP_OBSERVATION_SCHEMA = (
    "ny_beta_crown_mo_alpha_step_observations_v1"
)
_MO_ALPHA_STEP_SCHEDULE_PERIOD = 4
_MO_ALPHA_STEP_ROUND_REMAINDER = _MO_ALPHA_STEP_SCHEDULE_PERIOD - 1
_MO_ALPHA_STEP_EVENT_FIELD_ORDER = (
    "schema", "phase", "observation_schema", "observed",
    "effective_alpha_step", "effective_beta_spsa", "schedule_period",
    "alpha_round_remainder", "optimizer_invocations",
    "eligible_invocations", "eligible_iterations",
    "fourth_iteration_opportunities", "alpha_rounds_scheduled",
    "beta_rounds_scheduled", "alpha_rounds_completed",
    "alpha_rounds_refused", "alpha_updates_applied",
    "perturbed_coordinates",
)
_MO_ALPHA_STEP_EVENT_FIELDS = frozenset(_MO_ALPHA_STEP_EVENT_FIELD_ORDER)
_MO_ALPHA_STEP_INTEGER_FIELDS = frozenset({
    "schedule_period", "alpha_round_remainder", "optimizer_invocations",
    "eligible_invocations", "eligible_iterations",
    "fourth_iteration_opportunities", "alpha_rounds_scheduled",
    "beta_rounds_scheduled", "alpha_rounds_completed",
    "alpha_rounds_refused", "alpha_updates_applied",
    "perturbed_coordinates",
})


def _mo_alpha_step_execution_contract_valid(fields, status):
    """Validate one closed every-fourth MO-alpha terminal."""
    if (
        set(fields) != _MO_ALPHA_STEP_EVENT_FIELDS
        or fields.get("schema") != _MO_ALPHA_STEP_EVENT_SCHEMA
        or fields.get("phase") != "terminal"
        or fields.get("observation_schema")
            != _MO_ALPHA_STEP_OBSERVATION_SCHEMA
        or fields.get("observed") not in {"true", "false"}
        or fields.get("effective_alpha_step")
            not in {"true", "false", "none"}
        or fields.get("effective_beta_spsa")
            not in {"true", "false", "none"}
    ):
        return False
    numbers = {
        name: _canonical_event_u64(fields.get(name))
        for name in _MO_ALPHA_STEP_INTEGER_FIELDS
    }
    if any(value is None for value in numbers.values()):
        return False

    observed = fields["observed"] == "true"
    alpha = fields["effective_alpha_step"]
    beta = fields["effective_beta_spsa"]
    invocations = numbers["optimizer_invocations"]
    eligible_invocations = numbers["eligible_invocations"]
    iterations = numbers["eligible_iterations"]
    fourth = numbers["fourth_iteration_opportunities"]
    alpha_scheduled = numbers["alpha_rounds_scheduled"]
    beta_scheduled = numbers["beta_rounds_scheduled"]
    completed = numbers["alpha_rounds_completed"]
    refused = numbers["alpha_rounds_refused"]
    updates = numbers["alpha_updates_applied"]
    coordinates = numbers["perturbed_coordinates"]
    u64_max = (1 << 64) - 1
    scheduled_sum = alpha_scheduled + beta_scheduled
    outcome_sum = completed + refused
    fourth_floor = fourth * _MO_ALPHA_STEP_SCHEDULE_PERIOD
    fourth_ceiling = (
        fourth_floor
        + eligible_invocations * (_MO_ALPHA_STEP_SCHEDULE_PERIOD - 1)
    )

    if (
        numbers["schedule_period"] != _MO_ALPHA_STEP_SCHEDULE_PERIOD
        or numbers["alpha_round_remainder"]
            != _MO_ALPHA_STEP_ROUND_REMAINDER
        or observed != (invocations > 0)
        or (alpha != "none") != observed
        or (beta != "none") != observed
        or eligible_invocations > invocations
        or (iterations > 0 and eligible_invocations == 0)
        or fourth > iterations
        or fourth_floor > u64_max
        or fourth_ceiling > u64_max
        or iterations < fourth_floor
        or iterations > fourth_ceiling
        or scheduled_sum > u64_max
        or scheduled_sum != iterations
        or outcome_sum > u64_max
        or outcome_sum != alpha_scheduled
        or updates != completed
        or (alpha_scheduled == 0) != (coordinates == 0)
        or coordinates < alpha_scheduled
        or (
            eligible_invocations == 1
            and alpha_scheduled > 0
            and coordinates % alpha_scheduled != 0
        )
        or (
            beta == "false"
            and any(value > 0 for value in (
                eligible_invocations, iterations, fourth,
                alpha_scheduled, beta_scheduled,
            ))
        )
        or (alpha == "true" and alpha_scheduled != fourth)
        or (
            alpha == "false"
            and any(value > 0 for value in (
                alpha_scheduled, completed, refused, updates, coordinates,
            ))
        )
        or status != ("ran" if completed > 0 else "skipped")
    ):
        return False
    return True


def _mo_alpha_step_execution_evidence(row):
    """Return exactly one strict MO-alpha terminal, or poison the row."""
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return None
    matches = []
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("method") != "mo_alpha_step_execution"
        ):
            continue
        fields = _strict_ordered_event_fields(
            event,
            "mo_alpha_step_execution",
            _MO_ALPHA_STEP_EVENT_SCHEMA,
            _MO_ALPHA_STEP_EVENT_FIELD_ORDER,
        )
        if (
            fields is None
            or not _mo_alpha_step_execution_contract_valid(
                fields, event.get("status")
            )
        ):
            return None
        fields["_status"] = event["status"]
        matches.append(fields)
    return matches[0] if len(matches) == 1 else None


def _mo_alpha_step_parameter_engagement(wanted, rows):
    """Credit only an enabled master that completed real alpha updates."""
    requested = dict(wanted)
    if (
        requested.get("NY_MO_ALPHA_STEP") != "1"
        or requested.get("NY_MO_BETA_SPSA") != "1"
        or not rows
    ):
        return set()
    evidence = []
    for row in rows:
        event = _mo_alpha_step_execution_evidence(row)
        if event is None:
            return set()
        evidence.append(event)
    if all(
        event["observed"] == "true"
        and event["effective_alpha_step"] == "true"
        and event["effective_beta_spsa"] == "true"
        and _canonical_event_u64(event["schedule_period"])
            == _MO_ALPHA_STEP_SCHEDULE_PERIOD
        and _canonical_event_u64(event["alpha_round_remainder"])
            == _MO_ALPHA_STEP_ROUND_REMAINDER
        and _canonical_event_u64(event["eligible_invocations"]) > 0
        and _canonical_event_u64(event["eligible_iterations"]) > 0
        and _canonical_event_u64(
            event["fourth_iteration_opportunities"]
        ) > 0
        and _canonical_event_u64(event["alpha_rounds_scheduled"]) > 0
        and _canonical_event_u64(event["alpha_rounds_completed"]) > 0
        and _canonical_event_u64(event["alpha_updates_applied"]) > 0
        and _canonical_event_u64(event["perturbed_coordinates"]) > 0
        for event in evidence
    ):
        return {"NY_MO_ALPHA_STEP"}
    return set()

# Axes in this map have a dedicated post-run execution contract below. A
# registry/delivery receipt proves only that configuration arrived; it is not
# evidence that a verifier decision or unit of work consumed that parameter.
# Production planning and T1 validation both fail closed outside this map.
_AXIS_SPECIFIC_T1_CONTRACTS = {
    "NY_MO_KFSB": _MO_KFSB_EVENT_SCHEMA,
    "NY_MO_KFSB_REDUCE": _MO_KFSB_EVENT_SCHEMA,
    "NY_MO_ALPHA_STEP": _MO_ALPHA_STEP_EVENT_SCHEMA,
    **{
        name: (
            _BNN_FALSIFICATION_POLICY_EVENT_SCHEMA + "+"
            + _BNN_FALSIFICATION_EXECUTION_EVENT_SCHEMA
        )
        for name in _BNN_FALSIFICATION_FAMILY
    },
    **{
        name: "ny-cli/upfront-policy/v2+ny-cli/upfront-execution/v2"
        for name in {
            "NY_UPFRONT_ATTACK", "NY_UPFRONT_ATTACK_CAP",
            "NY_UPFRONT_ATTACK_AUTO_CAP", "NY_UPFRONT_ATTACK_FRAC",
            "NY_UPFRONT_ATTACK_PROBE_FIRST", "NY_BNN_ATTACK_CAP",
            "NY_BNN_ATTACK_FRAC",
        }
    },
    "NY_MARGIN_ROW_PARALLEL": "ny-cli/margin-row-frontier-execution/v1",
    **{
        name: _MARGIN_ROW_WIDTH_EVENT_SCHEMA
        for name in _MARGIN_ROW_WIDTH_FAMILY
    },
    "NY_MARGIN_ROW_CONCURRENT": (
        _MARGIN_ROW_SCHEDULE_EVENT_SCHEMA + "+"
        + _MARGIN_ROW_CONCURRENT_OVERLAP_EVENT_SCHEMA + "+"
        + _MARGIN_ROW_WIDTH_EVENT_SCHEMA
    ),
    **{
        name: _MARGIN_ROW_POLICY_EVENT_SCHEMA
        for name in _MARGIN_ROW_POLICY_FAMILY
    },
    **{
        name: _MARGIN_ROW_ROOT_PRECISION_EVENT_SCHEMA
        for name in _MARGIN_ROW_ROOT_PRECISION_FAMILY
    },
    **{
        name: _MARGIN_ROW_ROOT_ALPHA_EVENT_SCHEMA
        for name in _MARGIN_ROW_ROOT_ALPHA_FAMILY
    },
    **{
        name: _MARGIN_ROW_DOMAIN_ALPHA_EVENT_SCHEMA
        for name in _MARGIN_ROW_DOMAIN_ALPHA_FAMILY
    },
    **{
        name: _MARGIN_ROW_EPOCH_EVENT_SCHEMA
        for name in _MARGIN_ROW_EPOCH_T1_FAMILY
    },
    **{
        name: _MARGIN_ROW_BACKWARD_INTERM_EVENT_SCHEMA
        for name in _MARGIN_ROW_BACKWARD_INTERM_FAMILY
    },
    "NY_PGD_FAST_GRAD": "ny-cli/pgd-fast-gradient-execution/v1",
    "NY_POSTBAB_RESERVE_SECS": "ny-cli/postbab-reserve-execution/v2",
    **{
        name: _ROOT_JOINT_INTERM_ALPHA_EVENT_SCHEMA
        for name in _ROOT_JOINT_INTERM_ALPHA_T1_FAMILY
    },
    **{name: "ny-cli/exact-vjp-execution/v1" for name in _EXACT_VJP_FAMILY},
}


def _classify_design_evidence(design):
    """Attach the same fail-closed T1 scope used by planning and aggregation."""
    contrast_scopes = set()
    for contrasts in design["contrasts_by_assignment"].values():
        for contrast in contrasts:
            ungraduated = sorted(
                name for name in contrast["requested_axes"]
                if name not in _AXIS_SPECIFIC_T1_CONTRACTS
            )
            contrast["evidence_scope"] = (
                "delivery_only" if ungraduated else "scored"
            )
            contrast["ungraduated_axes"] = ungraduated
            contrast_scopes.add(contrast["evidence_scope"])
    return (
        next(iter(contrast_scopes)) if len(contrast_scopes) == 1
        else "mixed" if contrast_scopes else "scored"
    )


_EXACT_VJP_EVENT_SCHEMA = "ny-cli/exact-vjp-execution/v1"
_EXACT_VJP_EVENT_FIELDS = frozenset({
    "schema", "phase", "route", "outcome", "inactive_reason",
    "prefer_exact_batched", "batch_enabled", "work_started", "width_source",
    "width_configured", "width_default", "width_initial", "width_final",
    "width_max_active", "width_binding", "width_retries", "total_restarts",
    "restarts_started", "waves", "wave_steps_configured", "wave_steps_default",
    "wave_steps_effective", "vjp_steps", "odi_steps_configured", "odi_eta",
    "odi_vjp_steps", "odi_completed_phase_count", "odi_move_count",
    "legacy_lanes_configured", "legacy_lanes_used",
    "lane_window", "recycle_count", "exploit_enabled", "exploit_opportunity_count",
    "exploit_count", "hinge_tau", "hinge_extra_count",
})
_EXACT_VJP_INTEGER_FIELDS = frozenset({
    "width_default", "width_initial", "width_final", "width_max_active",
    "width_retries", "total_restarts", "restarts_started", "waves",
    "wave_steps_default", "wave_steps_effective", "vjp_steps", "odi_steps_configured",
    "odi_vjp_steps", "odi_completed_phase_count", "odi_move_count",
    "legacy_lanes_configured", "legacy_lanes_used",
    "lane_window", "recycle_count", "exploit_opportunity_count", "exploit_count",
    "hinge_extra_count",
})
_EXACT_VJP_MONOTONE_FIELDS = frozenset({
    "width_max_active", "width_retries", "restarts_started", "waves",
    "vjp_steps", "odi_vjp_steps", "odi_completed_phase_count", "odi_move_count",
    "legacy_lanes_used", "recycle_count",
    "exploit_opportunity_count", "exploit_count", "hinge_extra_count",
})
_EXACT_VJP_STATIC_FIELDS = frozenset({
    "schema", "route", "prefer_exact_batched", "batch_enabled", "width_source",
    "width_configured", "width_default", "width_initial", "total_restarts",
    "wave_steps_configured", "wave_steps_default", "wave_steps_effective",
    "odi_steps_configured", "odi_eta", "legacy_lanes_configured", "lane_window",
    "exploit_enabled", "hinge_tau",
})


def _canonical_event_u64(token):
    if (
        not isinstance(token, str)
        or re.fullmatch(r"0|[1-9][0-9]*", token) is None
    ):
        return None
    value = int(token, 10)
    return value if value <= (1 << 64) - 1 else None


def _exact_vjp_contract_valid(fields, status):
    """Validate one exact-VJP event independently of surrounding events."""
    if (
        set(fields) != _EXACT_VJP_EVENT_FIELDS
        or fields.get("schema") != _EXACT_VJP_EVENT_SCHEMA
        or status not in {"ran", "skipped"}
        or fields["phase"] not in {"progress", "terminal"}
        or fields["route"]
        not in {"conjunctive", "disjunctive", "conjunctive_generic"}
        or fields["prefer_exact_batched"] not in {"true", "false"}
        or fields["batch_enabled"] not in {"true", "false"}
        or fields["work_started"] not in {"true", "false"}
        or fields["exploit_enabled"] not in {"true", "false"}
        or fields["width_source"] not in {"contextual", "configured"}
        or fields["width_binding"]
        not in {"not_run", "requested", "restart_cap", "adapter_halved"}
    ):
        return False

    optional_integers = ("width_configured", "wave_steps_configured")
    integers = {}
    for name in _EXACT_VJP_INTEGER_FIELDS:
        value = _canonical_event_u64(fields[name])
        if value is None:
            return False
        integers[name] = value
    for name in optional_integers:
        token = fields[name]
        if token == "none":
            integers[name] = None
        else:
            value = _canonical_event_u64(token)
            if value is None:
                return False
            integers[name] = value
    odi_eta = _parsed_ascii_f32(fields["odi_eta"])
    if odi_eta is None or odi_eta <= 0.0:
        return False
    hinge_tau = _parsed_ascii_f32(fields["hinge_tau"])
    if hinge_tau is None or hinge_tau < 0.0:
        return False

    configured_width = integers["width_configured"]
    configured_wave_steps = integers["wave_steps_configured"]
    route = fields["route"]
    expected_width_default = 32 if route == "disjunctive" else 64
    expected_wave_steps = (
        configured_wave_steps
        if route == "conjunctive" and configured_wave_steps is not None
        else integers["wave_steps_default"]
    )

    if (
        (fields["width_source"] == "configured")
        != (configured_width is not None)
        or (configured_width is not None and configured_width == 0)
        or (configured_wave_steps is not None and configured_wave_steps == 0)
        or integers["width_default"] != expected_width_default
        or integers["width_initial"] == 0
        or integers["total_restarts"] == 0
        or integers["wave_steps_effective"] != expected_wave_steps
        or integers["width_initial"]
        != (
            configured_width
            if configured_width is not None
            else integers["width_default"]
        )
        or integers["restarts_started"] > integers["total_restarts"]
        or (integers["odi_vjp_steps"] > 0 and integers["odi_steps_configured"] == 0)
        or (integers["odi_move_count"] > 0 and integers["odi_vjp_steps"] == 0)
        or (
            integers["odi_completed_phase_count"] > 0
            and (
                integers["odi_steps_configured"] == 0
                or integers["odi_completed_phase_count"]
                > ((1 << 64) - 1) // integers["odi_steps_configured"]
                or integers["odi_vjp_steps"]
                < integers["odi_steps_configured"]
                * integers["odi_completed_phase_count"]
            )
        )
        or integers["odi_completed_phase_count"] > integers["waves"]
        or (integers["recycle_count"] > 0 and integers["lane_window"] == 0)
        or (
            integers["exploit_count"] > 0
            and fields["exploit_enabled"] != "true"
        )
        or (
            integers["exploit_count"] > 0
            and integers["exploit_opportunity_count"] == 0
        )
        or (
            integers["hinge_extra_count"] > 0
            and hinge_tau == 0.0
        )
    ):
        return False

    binding = fields["width_binding"]
    if binding == "not_run":
        if (
            integers["width_final"] != integers["width_initial"]
            or any(integers[name] != 0 for name in _EXACT_VJP_MONOTONE_FIELDS)
        ):
            return False
    elif (
        integers["width_max_active"] == 0
        or integers["restarts_started"] == 0
        or integers["waves"] == 0
    ):
        return False
    if binding == "requested" and (
        integers["width_final"] != integers["width_initial"]
        or integers["width_max_active"] != integers["width_initial"]
        or integers["width_retries"] != 0
    ):
        return False
    if binding == "restart_cap" and (
        integers["width_final"] >= integers["width_initial"]
        or integers["width_retries"] != 0
    ):
        return False
    if (
        any(
            integers[name] > 0
            for name in (
                "recycle_count", "exploit_opportunity_count", "exploit_count",
                "hinge_extra_count",
            )
        )
        and integers["vjp_steps"] == 0
    ):
        return False
    if route != "conjunctive" and any(
        integers[name] > 0
        for name in (
            "odi_vjp_steps", "odi_completed_phase_count", "odi_move_count",
            "legacy_lanes_used", "recycle_count", "exploit_opportunity_count",
            "exploit_count", "hinge_extra_count",
        )
    ):
        return False
    if binding == "adapter_halved" and (
        integers["width_final"] >= integers["width_initial"]
        or integers["width_retries"] == 0
    ):
        return False
    if (
        binding != "not_run"
        and integers["width_final"] == 0
    ):
        return False
    if (
        integers["width_max_active"] > integers["width_initial"]
        or integers["width_max_active"] > integers["restarts_started"]
        or integers["width_retries"] > integers["waves"]
        or (
            binding == "requested"
            and integers["width_max_active"] > integers["width_final"]
        )
    ):
        return False

    outcomes = {
        "in_progress", "candidate", "exhausted", "deadline", "stalled",
        "batch_disabled", "generic_completed", "fallback_sequential",
        "not_started", "internal_error",
    }
    inactive_reasons = {
        "none", "deadline_before_start", "batch_disabled", "gpu_unavailable",
        "empty_objective", "unsupported_plan", "forward_failure", "vjp_failure",
        "gradient_count_mismatch", "no_attack_target", "structural_ineligible",
        "insufficient_restarts", "experimental_lane_active", "generic_fallback",
        "internal_error",
    }
    outcome = fields["outcome"]
    inactive = fields["inactive_reason"]
    if outcome not in outcomes or inactive not in inactive_reasons:
        return False
    counters_prove_work = any(
        integers[name] > 0
        for name in (
            "width_max_active", "restarts_started", "waves", "vjp_steps",
            "odi_vjp_steps", "odi_move_count", "odi_completed_phase_count",
            "legacy_lanes_used", "recycle_count",
            "exploit_opportunity_count", "exploit_count", "hinge_extra_count",
        )
    )
    work_started = fields["work_started"] == "true"
    if counters_prove_work and not work_started:
        return False
    if fields["phase"] == "progress":
        return (
            route != "conjunctive_generic"
            and outcome == "in_progress" and inactive == "none"
            and status == "ran" and work_started
        )
    if outcome == "in_progress":
        return False

    if route == "conjunctive" and fields["prefer_exact_batched"] != "true":
        return False
    if route == "conjunctive_generic" and fields["prefer_exact_batched"] != "false":
        return False
    if (
        route in {"conjunctive", "disjunctive"}
        and work_started
        and fields["batch_enabled"] != "true"
    ):
        return False
    if route == "conjunctive_generic":
        return (
            outcome == "generic_completed" and status == "ran"
            and inactive == "none" and work_started
        ) or (
            outcome == "fallback_sequential"
            and status == "skipped"
            and inactive == "generic_fallback"
            and not work_started
        ) or (
            outcome == "not_started"
            and status == "skipped"
            and inactive == "deadline_before_start"
            and not work_started
        ) or (
            outcome == "internal_error"
            and inactive == "internal_error"
            and status == ("ran" if work_started else "skipped")
        )
    if outcome == "generic_completed" or inactive == "generic_fallback":
        return False
    if outcome in {"candidate", "exhausted", "deadline", "stalled"}:
        return (
            status == "ran"
            and inactive == "none"
            and work_started
        )
    if outcome == "batch_disabled":
        return (
            status == "ran"
            and inactive == "batch_disabled"
            and fields["batch_enabled"] == "false"
            and not work_started
        )
    if outcome == "fallback_sequential":
        return (
            inactive in {
                "forward_failure", "vjp_failure", "gradient_count_mismatch",
                "no_attack_target",
            }
            and status == "ran"
            and work_started
        )
    if outcome == "not_started":
        return (
            status == "skipped"
            and inactive in {
                "deadline_before_start", "gpu_unavailable", "empty_objective",
                "unsupported_plan", "structural_ineligible", "insufficient_restarts",
                "experimental_lane_active",
            }
            and not work_started
        )
    return (
        outcome == "internal_error"
        and inactive == "internal_error"
        and status == ("ran" if work_started else "skipped")
    )


def _exact_vjp_terminal_evidences(row):
    """Return every independently validated closed exact-VJP invocation.

    The method is intentionally stricter than the historical flight parsers:
    any malformed same-method event poisons the whole row, and a trailing
    progress event cannot borrow counters from an older terminal disposition.
    Counters may reset only after a terminal closes the preceding invocation.
    """
    flight = row.get("flight") if isinstance(row, dict) else None
    events = flight.get("events") if isinstance(flight, dict) else None
    if not isinstance(events, list):
        return []
    records = []
    for event in events:
        if not isinstance(event, dict) or event.get("method") != "exact_vjp_execution":
            continue
        reason = event.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.isascii()
            or not reason
            or reason != reason.strip(" ")
            or "  " in reason
            or any(character.isspace() and character != " " for character in reason)
        ):
            return []
        tokens = reason.split(" ")
        if not tokens or tokens[0] != "schema=" + _EXACT_VJP_EVENT_SCHEMA:
            return []
        fields = {}
        for token in tokens:
            if token.count("=") != 1:
                return []
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                return []
            fields[key] = value
        status = event.get("status")
        if not _exact_vjp_contract_valid(fields, status):
            return []
        fields["_status"] = status
        records.append(fields)
    if not records:
        return []

    terminals = []
    invocation = []
    for record in records:
        invocation.append(record)
        if record["phase"] != "terminal":
            continue
        terminal = record
        if any(item["phase"] != "progress" for item in invocation[:-1]):
            return []
        previous_counters = {
            name: 0 for name in _EXACT_VJP_MONOTONE_FIELDS
        }
        work_started = False
        for item in invocation:
            if any(
                item[name] != terminal[name]
                for name in _EXACT_VJP_STATIC_FIELDS
            ):
                return []
            counters = {
                name: _canonical_event_u64(item[name])
                for name in _EXACT_VJP_MONOTONE_FIELDS
            }
            if any(
                counters[name] is None
                or counters[name] < previous_counters[name]
                for name in _EXACT_VJP_MONOTONE_FIELDS
            ):
                return []
            current_work_started = item["work_started"] == "true"
            if work_started and not current_work_started:
                return []
            work_started = current_work_started
            previous_counters = counters
        terminals.append(terminal)
        invocation = []
    # Any residue is progress for an invocation that never published its
    # terminal disposition. It poisons the row instead of borrowing an older
    # invocation's already-closed evidence.
    return [] if invocation else terminals


def _exact_vjp_parameter_engagement(wanted, rows):
    """Requested exact-VJP controls consumed by at least one closed invocation."""
    requested = {name: token for name, token in wanted if name in _EXACT_VJP_FAMILY}
    if not requested:
        return set()
    terminals = [
        terminal
        for row in rows
        for terminal in _exact_vjp_terminal_evidences(row)
    ]
    engaged = set()

    def boolean_matches(event, field, token):
        return token in {"0", "1"} and event[field] == ("true" if token == "1" else "false")

    def integer_matches(event, field, token, optional=False):
        expected = _canonical_event_u64(token)
        actual = None if optional and event[field] == "none" else _canonical_event_u64(event[field])
        return expected is not None and actual == expected

    def f32_matches(event, field, token):
        expected = _parsed_ascii_f32(token)
        actual = _parsed_ascii_f32(event[field])
        if expected is None or actual is None:
            return False
        return struct.pack(">f", expected) == struct.pack(">f", actual)

    for event in terminals:
        route = event["route"]
        outcome = event["outcome"]
        exact_route = route in {"conjunctive", "disjunctive"}
        width = _canonical_event_u64(event["width_max_active"])
        restarts = _canonical_event_u64(event["restarts_started"])
        waves = _canonical_event_u64(event["waves"])
        vjp_steps = _canonical_event_u64(event["vjp_steps"])
        odi_vjp_steps = _canonical_event_u64(event["odi_vjp_steps"])
        odi_completed_phase_count = _canonical_event_u64(
            event["odi_completed_phase_count"]
        )
        odi_move_count = _canonical_event_u64(event["odi_move_count"])
        wide_work = (
            exact_route and width > 0 and restarts > 0
            and vjp_steps + odi_vjp_steps > 0
        )

        token = requested.get("NY_PGD_EXACT_BATCHED")
        if token is not None and boolean_matches(event, "prefer_exact_batched", token):
            if token == "0" and route == "conjunctive_generic" and outcome == "generic_completed":
                engaged.add("NY_PGD_EXACT_BATCHED")
            elif token == "1" and route == "conjunctive" and wide_work:
                engaged.add("NY_PGD_EXACT_BATCHED")

        token = requested.get("NY_PGD_VJP_BATCH")
        if token is not None and exact_route and boolean_matches(event, "batch_enabled", token):
            if (token == "0" and outcome == "batch_disabled") or (token == "1" and wide_work):
                engaged.add("NY_PGD_VJP_BATCH")

        token = requested.get("NY_PGD_VJP_K")
        if token is not None and exact_route and integer_matches(
            event, "width_configured", token, optional=True
        ):
            requested_width = _canonical_event_u64(token)
            if (
                wide_work
                and event["width_source"] == "configured"
                and event["width_binding"] in {"requested", "restart_cap"}
                and _canonical_event_u64(event["width_initial"]) == requested_width
                and _canonical_event_u64(event["width_default"]) != requested_width
                and width == requested_width
                and _canonical_event_u64(event["width_retries"]) == 0
            ):
                engaged.add("NY_PGD_VJP_K")

        if route != "conjunctive":
            continue

        token = requested.get("NY_PGD_LEGACY_LANES")
        if token is not None and integer_matches(event, "legacy_lanes_configured", token):
            configured = _canonical_event_u64(token)
            used = _canonical_event_u64(event["legacy_lanes_used"])
            if (
                configured == 0
                and wide_work
                and width > 1
            ) or (
                configured > 0
                and used == configured
                and vjp_steps + odi_vjp_steps > 0
            ):
                engaged.add("NY_PGD_LEGACY_LANES")

        token = requested.get("NY_PGD_ODI_STEPS")
        if token is not None and integer_matches(event, "odi_steps_configured", token):
            configured = _canonical_event_u64(token)
            if (configured == 0 and vjp_steps > 0) or (
                configured > 0
                and odi_vjp_steps >= configured
                and odi_completed_phase_count > 0
            ):
                engaged.add("NY_PGD_ODI_STEPS")

        token = requested.get("NY_PGD_ODI_ETA")
        if (
            token is not None
            and f32_matches(event, "odi_eta", token)
            and _canonical_event_u64(event["odi_steps_configured"]) > 0
            and odi_vjp_steps > 0
            and odi_move_count > 0
        ):
            engaged.add("NY_PGD_ODI_ETA")

        token = requested.get("NY_PGD_VJP_STEPS")
        if token is not None and integer_matches(
            event, "wave_steps_configured", token, optional=True
        ) and integer_matches(event, "wave_steps_effective", token):
            configured_steps = _canonical_event_u64(token)
            default_steps = _canonical_event_u64(event["wave_steps_default"])
            schedule_reached = (
                configured_steps < default_steps and vjp_steps >= configured_steps
            ) or (
                configured_steps > default_steps and vjp_steps > default_steps
            )
            if schedule_reached:
                engaged.add("NY_PGD_VJP_STEPS")

        token = requested.get("NY_PGD_LANE_WINDOW")
        if token is not None and integer_matches(event, "lane_window", token):
            configured = _canonical_event_u64(token)
            recycle_count = _canonical_event_u64(event["recycle_count"])
            if configured > 0 and recycle_count > 0:
                engaged.add("NY_PGD_LANE_WINDOW")

        token = requested.get("NY_PGD_EXPLOIT")
        if token is not None and boolean_matches(event, "exploit_enabled", token):
            opportunity_count = _canonical_event_u64(
                event["exploit_opportunity_count"]
            )
            exploit_count = _canonical_event_u64(event["exploit_count"])
            if (
                token == "0" and opportunity_count > 0 and exploit_count == 0
            ) or (token == "1" and exploit_count > 0):
                engaged.add("NY_PGD_EXPLOIT")

        token = requested.get("NY_PGD_HINGE_TAU")
        if token is not None and f32_matches(event, "hinge_tau", token):
            configured_tau = _parsed_ascii_f32(token)
            extra_count = _canonical_event_u64(event["hinge_extra_count"])
            if configured_tau > 0.0 and extra_count > 0:
                engaged.add("NY_PGD_HINGE_TAU")
    return engaged


def _upfront_parameter_engagement(wanted, rows):
    """Requested upfront controls that affected at least one measured row."""
    family = {
        "NY_UPFRONT_ATTACK",
        "NY_UPFRONT_ATTACK_CAP",
        "NY_UPFRONT_ATTACK_AUTO_CAP",
        "NY_UPFRONT_ATTACK_FRAC",
        "NY_UPFRONT_ATTACK_PROBE_FIRST",
        "NY_BNN_ATTACK_CAP",
        "NY_BNN_ATTACK_FRAC",
    }
    requested = {name: token for name, token in wanted if name in family}
    if not requested:
        return set()

    evidence = []
    for row in rows:
        for version in ("v2", "v1"):
            policies = _machine_event_fields(
                row, "upfront_attack_policy", f"ny-cli/upfront-policy/{version}"
            )
            executions = _machine_event_fields(
                row, "upfront_attack_execution", f"ny-cli/upfront-execution/{version}"
            )
            for policy in policies:
                evidence.append((policy, executions))

    engaged = set()
    enabled_routes = {
        "auto_disjunction",
        "preset_relational_conjunction",
        "forced_by_environment",
        "forced_by_preset",
    }
    def budget_effect(event):
        return (
            event.get("_status") == "ran"
            and event.get("inactive_reason") == "none"
        ) or (
            event.get("_status") == "skipped"
            and event.get("inactive_reason")
            in {"private_base_below_min", "private_effective_below_min"}
        )

    def lower_forced_budget_usable(policy, event):
        """Exact pre-Sign base gate for a lower forced route.

        This does not guess contextual/property admission.  It is used only
        when the preset itself forces admission, and mirrors Rust's checked
        three-second reserve followed by min(fractional wall, wall cap).
        """
        if policy.get("lower_route") != "forced_by_preset":
            return False
        try:
            wall_cap = float(policy["wall_cap_secs"])
            fraction = float(policy["budget_fraction"])
            remaining_token = event["remaining_secs"]
            if remaining_token == "unbounded":
                base = wall_cap
            else:
                remaining = float(remaining_token)
                if remaining < 3.0:
                    return False
                base = min((remaining - 3.0) * fraction, wall_cap)
        except (KeyError, TypeError, ValueError):
            return False
        return math.isfinite(base) and base >= 0.8

    def policy_number_matches(policy, field, token):
        try:
            actual, expected = float(policy[field]), float(token)
        except (KeyError, TypeError, ValueError):
            return False
        return math.isfinite(actual) and math.isfinite(expected) and actual == expected

    def event_number_matches(event, field, token):
        try:
            actual, expected = float(event[field]), float(token)
        except (KeyError, TypeError, ValueError):
            return False
        return math.isfinite(actual) and math.isfinite(expected) and math.isclose(
            actual, expected, rel_tol=1e-9, abs_tol=2e-6
        )

    def private_budget_lineage_matches(policy, event):
        """Mirror the v2 private budget decision from its rounded receipt.

        Frame lineage is decided earlier against a predicted clock and cannot
        be reconstructed from the later execution timestamp. Private authority
        is different: its policy, remaining wall, binding and effective budget
        all describe the same call, so disagreement is forged evidence.
        """
        if (
            event.get("schema") != "ny-cli/upfront-execution/v2"
            or event.get("budget_authority") != "private"
        ):
            return True
        try:
            wall_cap = int(policy["wall_cap_secs"])
            auto_cap = int(policy["auto_wall_cap_secs"])
            fraction = float(policy["budget_fraction"])
            bnn_cap = int(policy["bnn_wall_cap_secs"])
            bnn_fraction = float(policy["bnn_budget_fraction"])
            effective = float(event["effective_budget_secs"])
        except (KeyError, TypeError, ValueError):
            return False
        numbers = (wall_cap, auto_cap, fraction, bnn_cap, bnn_fraction, effective)
        if not all(math.isfinite(number) and number >= 0.0 for number in numbers):
            return False

        def decision_min(left, right, left_binding, right_binding):
            if left < right:
                return left, frozenset(left_binding)
            if left > right:
                return right, frozenset(right_binding)
            return left, frozenset(set(left_binding) | set(right_binding))

        wall_cap_ns = wall_cap * 1_000_000_000
        auto_cap_ns = auto_cap * 1_000_000_000
        bnn_cap_ns = bnn_cap * 1_000_000_000
        remaining_token = event.get("remaining_secs")
        if remaining_token == "unbounded":
            usable_candidates = [None]
        else:
            possible_remaining = _rust_duration_receipt_candidates(remaining_token)
            usable_candidates = [
                nanos - 3_000_000_000
                for nanos in possible_remaining
                if nanos >= 3_000_000_000
            ]
            if not usable_candidates:
                return False

        candidates = set()
        for usable_ns in usable_candidates:
            if usable_ns is None:
                base, base_binding = wall_cap_ns, frozenset({"wall_cap"})
                bnn, bnn_binding = bnn_cap_ns, frozenset({"bnn_wall_cap"})
            else:
                scaled_base = _rust_duration_mul_f64_ns(usable_ns, fraction)
                scaled_bnn = _rust_duration_mul_f64_ns(usable_ns, bnn_fraction)
                if scaled_base is None or scaled_bnn is None:
                    continue
                base, base_binding = decision_min(
                    scaled_base,
                    wall_cap_ns,
                    {"budget_fraction"},
                    {"wall_cap"},
                )
                bnn, bnn_binding = decision_min(
                    scaled_bnn,
                    bnn_cap_ns,
                    {"bnn_budget_fraction"},
                    {"bnn_wall_cap"},
                )

            reason = event.get("inactive_reason")
            if reason == "private_base_below_min":
                candidates.add((base, base_binding, "not_evaluated"))
            elif event.get("sign_net") == "true":
                if base > bnn:
                    candidates.add((base, base_binding, "ordinary_wins"))
                elif base < bnn:
                    candidates.add((bnn, bnn_binding, "bnn_wins"))
                else:
                    candidates.add((base, base_binding | bnn_binding, "tie"))
            elif event.get("sign_net") == "false":
                if event.get("admission") == "auto_disjunction":
                    budget, binding = decision_min(
                        auto_cap_ns,
                        base,
                        {"auto_wall_cap"},
                        base_binding,
                    )
                    candidates.add((budget, binding, "not_sign"))
                else:
                    candidates.add((base, base_binding, "not_sign"))
            else:
                return False

        actual_binding = frozenset(event.get("binding", "").split("+"))
        return any(
            actual_binding == binding
            and event.get("budget_competition") == competition
            and _rust_duration_format_ns(budget_ns)
                == event.get("effective_budget_secs")
            for budget_ns, binding, competition in candidates
        )

    def committed_budget_lineage_matches(policy, event):
        """Prove an exact frame source could arise on one planning clock.

        A frame is sized before the later execution receipt, so its precise
        planning `remaining` is unavailable. Numeric source lineage is still
        creditable only if some shared policy clock can produce the committed
        cap, binding and competition after both 800 ms gates. An unbounded
        execution clock is also unbounded at planning time and has no such
        existential freedom.
        """
        if (
            event.get("schema") != "ny-cli/upfront-execution/v2"
            or event.get("budget_authority") != "committed_cap"
        ):
            return True
        source_binding = set(event.get("binding", "").split("+")) - {
            "live_remaining"
        }
        numeric_names = {
            "budget_fraction", "wall_cap", "auto_wall_cap",
            "bnn_budget_fraction", "bnn_wall_cap",
        }
        if not source_binding.intersection(numeric_names):
            return True
        if not source_binding.issubset(numeric_names):
            return False
        try:
            wall_cap = int(policy["wall_cap_secs"])
            auto_cap = int(policy["auto_wall_cap_secs"])
            fraction = float(policy["budget_fraction"])
            bnn_cap = int(policy["bnn_wall_cap_secs"])
            bnn_fraction = float(policy["bnn_budget_fraction"])
            requested_cap = float(event["requested_cap_secs"])
        except (KeyError, TypeError, ValueError):
            return False
        numbers = (
            wall_cap, auto_cap, fraction, bnn_cap, bnn_fraction, requested_cap,
        )
        if not all(math.isfinite(number) and number >= 0.0 for number in numbers):
            return False
        if fraction <= 0.0 or bnn_fraction <= 0.0:
            return False

        wall_cap_ns = wall_cap * 1_000_000_000
        auto_cap_ns = auto_cap * 1_000_000_000
        bnn_cap_ns = bnn_cap * 1_000_000_000
        target_ns = _fixed_six_duration_center_ns(event.get("requested_cap_secs"))
        if target_ns is None:
            return False
        max_usable_ns = _RUST_DURATION_MAX_NS - 3_000_000_000

        def decision_min(left, right, left_binding, right_binding):
            if left < right:
                return left, frozenset(left_binding)
            if left > right:
                return right, frozenset(right_binding)
            return left, frozenset(set(left_binding) | set(right_binding))

        def evaluate(usable_ns):
            if usable_ns is None:
                base, base_binding = wall_cap_ns, frozenset({"wall_cap"})
                bnn, bnn_binding = bnn_cap_ns, frozenset({"bnn_wall_cap"})
            else:
                if usable_ns < 0 or usable_ns > max_usable_ns:
                    return None
                scaled_base = _rust_duration_mul_f64_ns(usable_ns, fraction)
                scaled_bnn = _rust_duration_mul_f64_ns(usable_ns, bnn_fraction)
                if scaled_base is None or scaled_bnn is None:
                    return None
                base, base_binding = decision_min(
                    scaled_base,
                    wall_cap_ns,
                    {"budget_fraction"},
                    {"wall_cap"},
                )
                bnn, bnn_binding = decision_min(
                    scaled_bnn,
                    bnn_cap_ns,
                    {"bnn_budget_fraction"},
                    {"bnn_wall_cap"},
                )
            if base < 800_000_000:
                return None
            if event.get("sign_net") == "true":
                if base > bnn:
                    decision = (base, base_binding, "ordinary_wins")
                elif base < bnn:
                    decision = (bnn, bnn_binding, "bnn_wins")
                else:
                    decision = (base, base_binding | bnn_binding, "tie")
            elif event.get("sign_net") == "false":
                if event.get("admission") == "auto_disjunction":
                    budget, binding = decision_min(
                        auto_cap_ns,
                        base,
                        {"auto_wall_cap"},
                        base_binding,
                    )
                    decision = (budget, binding, "not_sign")
                else:
                    decision = (base, base_binding, "not_sign")
            else:
                return None
            return decision if decision[0] >= 800_000_000 else None

        if event.get("remaining_secs") == "unbounded":
            candidates = [evaluate(None)]
        else:
            centers = {0}

            def add_center(nanoseconds, derivative=None):
                if (
                    not math.isfinite(nanoseconds)
                    or nanoseconds < 0.0
                    or nanoseconds > max_usable_ns
                ):
                    return
                center = int(round(nanoseconds))
                offsets = {-2, -1, 0, 1, 2}
                if derivative is not None and derivative > 0.0:
                    receipt_span = int(math.ceil(600.0 / derivative))
                    offsets.update({-receipt_span, receipt_span})
                # A mathematical threshold can remain an exact nanosecond tie
                # for several adjacent source Durations after `mul_f64`
                # rounding. Sample far enough along every policy slope (and
                # their outer difference) to enter each neighboring binding
                # cell while the six-decimal budget still prints identically.
                transition_slopes = {
                    fraction,
                    bnn_fraction,
                    abs(fraction - bnn_fraction),
                }
                if derivative is not None:
                    transition_slopes.add(derivative)
                for slope in transition_slopes:
                    if slope <= 0.0:
                        continue
                    transition = int(math.ceil(2.0 / slope))
                    for adjustment in range(-2, 3):
                        offsets.update({
                            -(transition + adjustment),
                            transition + adjustment,
                        })
                centers.update(
                    center + offset
                    for offset in offsets
                    if 0 <= center + offset <= max_usable_ns
                )

            for scale in (fraction, bnn_fraction):
                add_center(target_ns / scale, scale)
                add_center(800_000_000.0 / scale)
            add_center(wall_cap_ns / fraction, fraction)
            add_center(bnn_cap_ns / bnn_fraction, bnn_fraction)
            add_center(auto_cap_ns / fraction, fraction)
            add_center(wall_cap_ns / bnn_fraction, bnn_fraction)
            add_center(bnn_cap_ns / fraction, fraction)
            if centers:
                centers.add(min(max_usable_ns, max(centers) + 1_000_000_000))
            candidates = [evaluate(usable_ns) for usable_ns in centers]

        wanted_binding = frozenset(source_binding)
        wanted_competition = event.get("budget_competition")

        def same_cap_survives_execution(cap_ns):
            """Correlate frame lineage with its later live clamp exactly."""
            if (
                _rust_duration_format_ns(cap_ns)
                    != event.get("requested_cap_secs")
                or _rust_duration_format_ns(cap_ns)
                    != event.get("effective_budget_secs")
            ):
                return False
            actual_binding = frozenset(event.get("binding", "").split("+"))
            remaining_token = event.get("remaining_secs")
            if remaining_token == "unbounded":
                return actual_binding == wanted_binding
            for remaining_ns in _rust_duration_receipt_candidates(remaining_token):
                if remaining_ns < 3_000_000_000:
                    continue
                usable_ns = remaining_ns - 3_000_000_000
                if usable_ns < cap_ns:
                    continue
                expected_binding = wanted_binding
                if usable_ns == cap_ns:
                    expected_binding = expected_binding | {"live_remaining"}
                if actual_binding == expected_binding:
                    return True
            return False

        return any(
            candidate is not None
            and candidate[1] == wanted_binding
            and candidate[2] == wanted_competition
            and same_cap_survives_execution(candidate[0])
            for candidate in candidates
        )

    def budget_lineage_matches(policy, event):
        return (
            private_budget_lineage_matches(policy, event)
            and committed_budget_lineage_matches(policy, event)
        )

    master_token = requested.get("NY_UPFRONT_ATTACK")
    for policy, executions in evidence:
        if master_token == "0" and (
            policy.get("master") == "disabled_by_environment"
            and policy.get("route") == "disabled_by_environment"
            and policy.get("lower_route") == "forced_by_preset"
            and any(
                event.get("route") == "disabled_by_environment"
                and event.get("_status") == "skipped"
                and event.get("inactive_reason") == "wrapper_disabled"
                and lower_forced_budget_usable(policy, event)
                for event in executions
            )
        ):
            engaged.add("NY_UPFRONT_ATTACK")
        elif master_token == "1" and (
            policy.get("master") == "forced_by_environment"
            and policy.get("route") == "forced_by_environment"
            and policy.get("lower_route") != "forced_by_preset"
            and any(
                event.get("route") == "forced_by_environment"
                and event.get("admission") == "forced"
                and budget_effect(event)
                and budget_lineage_matches(policy, event)
                for event in executions
            )
        ):
            engaged.add("NY_UPFRONT_ATTACK")

        probe_token = requested.get("NY_UPFRONT_ATTACK_PROBE_FIRST")
        if (
            probe_token == "1"
            and policy.get("probe_first") == "true"
            and policy.get("lower_probe_first") == "false"
            and any(
                event.get("consultation") == "probe_first"
                and event.get("route") == policy.get("route")
                and event.get("budget_authority") == "probe_cap"
                and "probe_cap" in set(event.get("binding", "").split("+"))
                and event.get("_status") == "ran"
                and event.get("inactive_reason") == "none"
                for event in executions
            )
        ):
            engaged.add("NY_UPFRONT_ATTACK_PROBE_FIRST")
        elif probe_token == "0" and (
            policy.get("probe_first") == "false"
            and policy.get("lower_probe_first") == "true"
            and policy.get("route") in enabled_routes
            and any(
                event.get("consultation") == "probe_first"
                and event.get("route") == policy.get("route")
                and event.get("_status") == "skipped"
                and event.get("inactive_reason") == "probe_disarmed"
                for event in executions
            )
            and any(
                event.get("consultation") == "ordinary"
                and event.get("route") == policy.get("route")
                and event.get("_status") == "ran"
                and event.get("inactive_reason") == "none"
                and budget_lineage_matches(policy, event)
                for event in executions
            )
        ):
            engaged.add("NY_UPFRONT_ATTACK_PROBE_FIRST")

        for event in executions:
            if event.get("route") != policy.get("route"):
                continue
            binding = set(event.get("binding", "").split("+"))
            numeric_budget_effect = (
                event.get("consultation") == "ordinary"
                and event.get("budget_authority") in {"private", "committed_cap"}
                and budget_effect(event)
            )
            if not numeric_budget_effect:
                continue
            v2 = event.get("schema") == "ny-cli/upfront-execution/v2"
            if v2 and not budget_lineage_matches(policy, event):
                continue
            competition = event.get("budget_competition")
            direct_base_refusal = (
                event.get("inactive_reason") == "private_base_below_min"
            )
            ordinary_winner = direct_base_refusal or (
                event.get("sign_net") == "false"
                and (not v2 or competition == "not_sign")
            ) or (
                event.get("sign_net") == "true"
                and (
                    (v2 and competition == "ordinary_wins")
                    or (not v2 and "bnn_policy" not in binding)
                )
            )
            ordinary_cap = requested.get("NY_UPFRONT_ATTACK_CAP")
            if (
                ordinary_winner
                and ordinary_cap is not None
                and "wall_cap" in binding
                and policy_number_matches(policy, "wall_cap_secs", ordinary_cap)
                and (
                    event.get("budget_authority") != "committed_cap"
                    or event_number_matches(event, "requested_cap_secs", ordinary_cap)
                )
            ):
                engaged.add("NY_UPFRONT_ATTACK_CAP")
            ordinary_fraction = requested.get("NY_UPFRONT_ATTACK_FRAC")
            if (
                ordinary_winner
                and ordinary_fraction is not None
                and "budget_fraction" in binding
                and policy_number_matches(
                    policy, "budget_fraction", ordinary_fraction
                )
            ):
                engaged.add("NY_UPFRONT_ATTACK_FRAC")
            auto_cap = requested.get("NY_UPFRONT_ATTACK_AUTO_CAP")
            if (
                auto_cap is not None
                and "auto_wall_cap" in binding
                and event.get("admission") == "auto_disjunction"
                and event.get("sign_net") == "false"
                and (not v2 or competition == "not_sign")
                and policy_number_matches(policy, "auto_wall_cap_secs", auto_cap)
                and (
                    event.get("budget_authority") != "committed_cap"
                    or event_number_matches(event, "requested_cap_secs", auto_cap)
                )
            ):
                engaged.add("NY_UPFRONT_ATTACK_AUTO_CAP")
            if (
                v2
                and event.get("_status") == "ran"
                and event.get("inactive_reason") == "none"
                and event.get("route") in enabled_routes
                and event.get("admission")
                in {"forced", "auto_disjunction", "preset_relational_conjunction"}
                and event.get("sign_net") == "true"
                and competition == "bnn_wins"
            ):
                bnn_cap = requested.get("NY_BNN_ATTACK_CAP")
                if (
                    bnn_cap is not None
                    and "bnn_wall_cap" in binding
                    and policy_number_matches(policy, "bnn_wall_cap_secs", bnn_cap)
                    and (
                        event.get("budget_authority") != "committed_cap"
                        or event_number_matches(event, "requested_cap_secs", bnn_cap)
                    )
                ):
                    engaged.add("NY_BNN_ATTACK_CAP")
                bnn_fraction = requested.get("NY_BNN_ATTACK_FRAC")
                if (
                    bnn_fraction is not None
                    and "bnn_budget_fraction" in binding
                    and policy_number_matches(
                        policy, "bnn_budget_fraction", bnn_fraction
                    )
                ):
                    engaged.add("NY_BNN_ATTACK_FRAC")
    return engaged


def _every_row_parameter_engagement(validator, wanted, rows):
    """Intersect one axis family's causal engagement over singleton rows.

    Search results score the aggregate outcome of every measured row.  A live
    instance therefore cannot authenticate an inactive sibling: until result
    records carry row-level outcome/engagement masks, every requested axis must
    have a causal witness independently on every scored row.  Preserve the
    validator's whole-set result as a separate conjunct: several families also
    poison mixed or contradictory policies across rows, which singleton calls
    alone must not weaken.
    """
    if not rows:
        return set()
    engaged = set(validator(wanted, rows))
    for row in rows:
        engaged.intersection_update(validator(wanted, [row]))
    return engaged


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
    expected_configs=None,
    expected_preset_path=None,
    allow_ungraduated_execution_contracts=False,
    engagement_arm=None,
    require_closed_root_terminal=False,
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
    engagement = wanted if engagement_arm is None else sorted(
        (str(name), str(value)) for name, value in engagement_arm
    )
    engagement_map = dict(engagement)
    manifest_record = manifest if isinstance(manifest, dict) else {}

    def fail(check, detail):
        failures.append({"check": check, "detail": detail})

    def warn(check, detail):
        warnings.append({"check": check, "detail": detail})

    ungraduated = sorted(
        name for name in engagement_map if name not in _AXIS_SPECIFIC_T1_CONTRACTS
    )
    if ungraduated and not allow_ungraduated_execution_contracts:
        fail(
            "axis_execution_contract_graduated",
            "requested axis/axes have delivery provenance but no axis-specific T1 "
            "execution contract: %s; configuration arrival is not parameter engagement"
            % ", ".join(ungraduated),
        )

    # The native manifest seals the explicit arm independently of each row.
    # Require its canonical wire representation so a missing/legacy manifest
    # cannot borrow otherwise plausible row receipts from another request.
    raw_manifest_arm = manifest_record.get("arm")
    manifest_arm = None
    if isinstance(raw_manifest_arm, list):
        parsed = []
        for pair in raw_manifest_arm:
            if (not isinstance(pair, list) or len(pair) != 2
                    or not all(isinstance(value, str) for value in pair)):
                parsed = None
                break
            parsed.append((pair[0], pair[1]))
        if parsed is not None and parsed == sorted(parsed):
            manifest_arm = parsed
    if manifest_arm != wanted:
        fail(
            "manifest_arm_matches",
            "manifest canonical arm %r != requested %r"
            % (raw_manifest_arm, [list(pair) for pair in wanted]),
        )

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

        # T1.1 -- rejected raw resolves to the declaration default; ignored raw
        # retains a lower layer. Neither token was the requested treatment.
        status, receipt, entries = _receipt_levers(row)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if "rejected_raw" in entry:
                fail(
                    "rejected_raw",
                    "%s: %s rejected raw %r -> resolved to the default, so this row "
                    "measured the BASELINE"
                    % (_row_name(row), entry.get("name"), entry["rejected_raw"]),
                )
            if "ignored_raw" in entry:
                fail(
                    "ignored_raw",
                    "%s: %s ignored raw %r -> retained its lower layer, so this row "
                    "did not measure the requested treatment"
                    % (_row_name(row), entry.get("name"), entry["ignored_raw"]),
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
        else:
            receipt_errors = _receipt_v3_errors(receipt, wanted)
            if receipt_errors:
                detail = "%s: %s" % (_row_name(row), "; ".join(receipt_errors))
                if strict_receipt:
                    fail("receipt_v3_contract", detail)
                else:
                    warn("receipt_v3_contract", detail)

        # T1.3 -- the row's own record of the arm it was measured under.
        row_arm = sorted((str(name), str(value)) for name, value in row.get("arm", []))
        if row_arm != wanted:
            fail(
                "row_arm_matches",
                "%s: row arm %s != requested %s"
                % (_row_name(row), row_arm or "[]", wanted or "[]"),
            )

        # T1.4 -- and the environment the child actually saw. The broad flight
        # schema stayed at v3 when Rayon capture was added, so the dedicated
        # ambient sub-schema is the only proof that an empty map also checked
        # both current and deprecated Rayon worker-count variables.
        # Missing/unknown tokens are readable legacy data,
        # but cannot establish the exact context required for a measurement.
        flight = row.get("flight") if isinstance(row.get("flight"), dict) else {}
        ambient_schema = flight.get("ambient_env_schema")
        if ambient_schema != AMBIENT_ENV_SCHEMA:
            fail(
                "child_ambient_provenance",
                "%s: child ambient capture schema is %r, expected %r; a legacy/unknown "
                "capture cannot prove every Rayon worker-count variable was absent"
                % (_row_name(row), ambient_schema, AMBIENT_ENV_SCHEMA),
            )
        if expected_preset_path is not None:
            actual_preset = _flight_base_preset_path(flight)
            if not _same_resolved_path(actual_preset, expected_preset_path):
                fail(
                    "preset_matches_request",
                    "%s: child flight says base preset %r, expected exact VNN-COMP %s preset %s"
                    % (_row_name(row), actual_preset, requested_year, expected_preset_path),
                )
        ambient = flight.get("ambient_env") if isinstance(flight.get("ambient_env"), dict) else None
        if ambient is None:
            fail(
                "child_ambient_provenance",
                "%s: child flight has no ambient_env object; cleanliness is unproven"
                % _row_name(row),
            )
        else:
            for name, value in wanted:
                if ambient.get(name) != value:
                    fail(
                        "child_env_carries_arm",
                        "%s: child ambient_env has %s=%r, requested %r"
                        % (_row_name(row), name, ambient.get(name), value),
                    )
            for name, value in sorted(ambient.items()):
                if not (name.startswith("NY_") or name in RAYON_WORKER_ENV):
                    continue
                if wanted_map.get(name) != value:
                    fail(
                        "child_env_has_no_unrequested_controls",
                        "%s: child ambient_env has unrequested %s=%r"
                        % (_row_name(row), name, value),
                    )

        if row.get("category") and category and row["category"] != category:
            fail(
                "category_matches",
                "%s: row category %r != requested %r" % (_row_name(row), row["category"], category),
            )

    requested_upfront = {
        name
        for name, _token in engagement
        if name in {
            "NY_UPFRONT_ATTACK",
            "NY_UPFRONT_ATTACK_CAP",
            "NY_UPFRONT_ATTACK_AUTO_CAP",
            "NY_UPFRONT_ATTACK_FRAC",
            "NY_UPFRONT_ATTACK_PROBE_FIRST",
            "NY_BNN_ATTACK_CAP",
            "NY_BNN_ATTACK_FRAC",
        }
    }
    missing_upfront = requested_upfront.difference(
        _every_row_parameter_engagement(
            _upfront_parameter_engagement, engagement, rows
        )
    )
    if missing_upfront:
        fail(
            "upfront_parameter_engaged",
            "not every measured row independently proved effective execution for %s; "
            "probe/allocator caps, "
            "outer-max-dominated budgets, structural declines, and saturated controls are not "
            "parameter experiments"
            % ", ".join(sorted(missing_upfront)),
        )

    requested_bnn_falsification = {
        name for name, _token in engagement
        if name in _BNN_FALSIFICATION_FAMILY
    }
    missing_bnn_falsification = requested_bnn_falsification.difference(
        _every_row_parameter_engagement(
            _bnn_falsification_parameter_engagement, engagement, rows
        )
    )
    if missing_bnn_falsification:
        fail(
            "bnn_falsification_parameter_engaged",
            "not every measured row carried one singular frozen BNN policy and "
            "row-local lane terminals proving causal work for %s; MIP-off, "
            "configuration-only, zero-work, cross-row disarm stitching, "
            "duplicate, mismatched, and malformed receipts are not parameter "
            "experiments"
            % ", ".join(sorted(missing_bnn_falsification)),
        )

    requested_mo_kfsb = {
        name for name, _token in engagement
        if name in {"NY_MO_KFSB", "NY_MO_KFSB_REDUCE"}
    }
    mo_kfsb_context = [
        (name, token) for name, token in wanted
        if name in {"NY_MO_KFSB", "NY_MO_KFSB_REDUCE"}
    ]
    missing_mo_kfsb = requested_mo_kfsb.difference(
        _every_row_parameter_engagement(
            _mo_kfsb_parameter_engagement, mo_kfsb_context, rows
        )
    )
    if missing_mo_kfsb:
        fail(
            "mo_kfsb_parameter_engaged",
            "not every measured row independently carried exactly one closed MO-kFSB "
            "terminal and proved a causal master or reducer effect; force-on requires a lower-off "
            "wave with real selector work and a nonempty committed parent map, while "
            "force-off requires a lower-on, candidate/route-eligible binding refusal; "
            "reducer credit requires its exact-enable environment or absent armed typed "
            "master authority, an effective reducer differing from its configured lower "
            "value, and real selector work "
            "with a nonempty committed parent map independently on every row; "
            "absent, rejected, duplicate, missing, malformed, conflicting, overflowed, "
            "and configuration-only evidence are not parameter experiments",
        )

    requested_mo_alpha_step = {
        name for name, _token in engagement if name == "NY_MO_ALPHA_STEP"
    }
    mo_alpha_step_context = [
        (name, token) for name, token in wanted
        if name in {"NY_MO_ALPHA_STEP", "NY_MO_BETA_SPSA"}
    ]
    missing_mo_alpha_step = requested_mo_alpha_step.difference(
        _every_row_parameter_engagement(
            _mo_alpha_step_parameter_engagement,
            mo_alpha_step_context,
            rows,
        )
    )
    if missing_mo_alpha_step:
        fail(
            "mo_alpha_step_parameter_engaged",
            "not every measured row independently carried exactly one closed "
            "every-fourth MO-alpha terminal with effective alpha and beta SPSA, "
            "a structurally eligible relu-split optimizer invocation, and at "
            "least one completed alpha update over real coordinates; disabled, "
            "ineligible, deadline-refused, duplicate, missing, malformed, "
            "conflicting, overflowed, and configuration-only evidence are not "
            "parameter experiments",
        )

    requested_margin_row_frontier = {
        name for name, _token in engagement if name == "NY_MARGIN_ROW_PARALLEL"
    }
    missing_margin_row_frontier = requested_margin_row_frontier.difference(
        _every_row_parameter_engagement(
            _margin_row_frontier_parameter_engagement, engagement, rows
        )
    )
    if missing_margin_row_frontier:
        fail(
            "margin_row_frontier_parameter_engaged",
            "not every measured row carried a closed joint-tree frontier receipt proving that "
            "the requested effective width changed a live batch decision; root-only, "
            "expired, classwise, equal-context, and never-wide-enough executions are not "
            "parameter experiments",
        )

    requested_margin_row_width = {
        name for name, _token in engagement if name in _MARGIN_ROW_WIDTH_FAMILY
    }
    missing_margin_row_width = requested_margin_row_width.difference(
        _every_row_parameter_engagement(
            _margin_row_width_parameter_engagement, engagement, rows
        )
    )
    if missing_margin_row_width:
        fail(
            "margin_row_width_parameter_engaged",
            "not every measured row carried exactly one closed width terminal proving "
            "ordinary outer exact-scored counterfactual selection work for %s; "
            "configured no-ops, Tier-0 bypass, saturated or unbound adaptive thresholds, "
            "joint K/adaptive changes, mismatched effective maxima, nested-only work, "
            "cross-axis binding, duplicate/conflicting, and malformed receipts are not "
            "parameter experiments"
            % ", ".join(sorted(missing_margin_row_width)),
        )

    requested_margin_row_schedule = {
        name for name, _token in engagement
        if name in _MARGIN_ROW_SCHEDULE_FAMILY
    }
    missing_margin_row_schedule = requested_margin_row_schedule.difference(
        _every_row_parameter_engagement(
            _margin_row_schedule_parameter_engagement, engagement, rows
        )
    )
    if missing_margin_row_schedule:
        fail(
            "margin_row_schedule_parameter_engaged",
            "not every measured row carried one exact frozen scheduler allocation, "
            "one real worker/verifier overlap, and one stats-owning margin-row width "
            "terminal under a clean zero-reserve internal window; disabled, suppressed, "
            "reserved, resource-refused, duplicate, cross-row-policy, malformed, and "
            "configuration-only rows are not concurrent-lane experiments",
        )

    requested_margin_row_policy = {
        name for name, _token in engagement if name in _MARGIN_ROW_POLICY_FAMILY
    }
    missing_margin_row_policy = requested_margin_row_policy.difference(
        _every_row_parameter_engagement(
            _margin_row_policy_parameter_engagement, engagement, rows
        )
    )
    if missing_margin_row_policy:
        fail(
            "margin_row_policy_parameter_engaged",
            "not every measured row carried a closed immutable margin-row policy receipt proving "
            "live, value-binding work for %s; configuration-only, default-equivalent widths, "
            "cache-hit/single-domain prebuilds, mismatched policies, and malformed receipts "
            "are not parameter experiments"
            % ", ".join(sorted(missing_margin_row_policy)),
        )

    requested_margin_row_root_precision = {
        name
        for name, _token in engagement
        if name in _MARGIN_ROW_ROOT_PRECISION_FAMILY
    }
    missing_margin_row_root_precision = (
        requested_margin_row_root_precision.difference(
            _every_row_parameter_engagement(
                _margin_row_root_precision_parameter_engagement,
                engagement,
                rows,
            )
        )
    )
    if missing_margin_row_root_precision:
        fail(
            "margin_row_root_precision_parameter_engaged",
            "not every measured row carried a closed root-precision receipt proving "
            "live, value-binding compute or tableau-storage work for %s; "
            "parity/inward roots, zero eligible operations, refused tableau "
            "storage, mixed policies, and malformed receipts are not parameter "
            "experiments"
            % ", ".join(sorted(missing_margin_row_root_precision)),
        )

    requested_margin_row_root_alpha = {
        name for name, _token in engagement
        if name in _MARGIN_ROW_ROOT_ALPHA_FAMILY
    }
    missing_margin_row_root_alpha = requested_margin_row_root_alpha.difference(
        _every_row_parameter_engagement(
            _margin_row_root_alpha_parameter_engagement, engagement, rows
        )
    )
    if missing_margin_row_root_alpha:
        fail(
            "margin_row_root_alpha_parameter_engaged",
            "not every measured row carried one closed nine-axis root-alpha/rebake receipt "
            "proving value-binding work for %s; failed candidate passes, saturated "
            "steering widths, deadline-capped or unexhausted budgets, guard routes "
            "without a publish opportunity, fixed-point/rejected rebakes, premature "
            "round stops, mixed policies, and malformed receipts are not parameter "
            "experiments"
            % ", ".join(sorted(missing_margin_row_root_alpha)),
        )

    requested_margin_row_domain_alpha = {
        name for name, _token in engagement
        if name in _MARGIN_ROW_DOMAIN_ALPHA_FAMILY
    }
    missing_margin_row_domain_alpha = (
        requested_margin_row_domain_alpha.difference(
            _every_row_parameter_engagement(
                _margin_row_domain_alpha_parameter_engagement,
                engagement,
                rows,
            )
        )
    )
    if missing_margin_row_domain_alpha:
        fail(
            "margin_row_domain_alpha_parameter_engaged",
            "not every measured row carried one closed joint-route domain-alpha "
            "receipt proving outer value-binding scoring for %s; classwise or "
            "nested-only activity, configuration-only gates, failed or "
            "non-finite trials, default-equivalent counterfactuals, mismatched "
            "policies, and malformed receipts are not parameter experiments"
            % ", ".join(sorted(missing_margin_row_domain_alpha)),
        )

    requested_margin_row_epoch = {
        name for name, _token in engagement
        if name in _MARGIN_ROW_EPOCH_T1_FAMILY
    }
    missing_margin_row_epoch = requested_margin_row_epoch.difference(
        _every_row_parameter_engagement(
            _margin_row_epoch_parameter_engagement, engagement, rows
        )
    )
    if missing_margin_row_epoch:
        fail(
            "margin_row_epoch_parameter_engaged",
            "not every measured row carried one closed joint-route six-axis epoch "
            "receipt with the unsafe incremental parent refresh sealed dark "
            "and proving value-binding outer/full-frame work for %s; "
            "classwise-only or nested-only counters, configured-only cadence, "
            "default-equivalent attempt caps, uncompleted candidate scoring, "
            "incremental refreshes, mixed policies, and malformed receipts "
            "are not parameter experiments"
            % ", ".join(sorted(missing_margin_row_epoch)),
        )

    requested_margin_row_backward_interm = {
        name for name, _token in engagement
        if name in _MARGIN_ROW_BACKWARD_INTERM_FAMILY
    }
    missing_margin_row_backward_interm = (
        requested_margin_row_backward_interm.difference(
            _every_row_parameter_engagement(
                _margin_row_backward_interm_parameter_engagement,
                engagement,
                rows,
            )
        )
    )
    if missing_margin_row_backward_interm:
        fail(
            "margin_row_backward_interm_parameter_engaged",
            "not every measured row carried a closed backward-intermediate phase receipt "
            "proving live, value-binding work for %s; disabled/non-eligible phases, "
            "deadline-capped budgets, unexhausted seconds, non-binding fixed chunks, "
            "capped or unchanged adaptive targets, top-k selections without headroom "
            "for omitted work, mismatched "
            "policies, and malformed receipts are not parameter experiments"
            % ", ".join(sorted(missing_margin_row_backward_interm)),
        )

    requested_pgd_fast_gradient = {
        name for name, _token in engagement if name == "NY_PGD_FAST_GRAD"
    }
    missing_pgd_fast_gradient = requested_pgd_fast_gradient.difference(
        _every_row_parameter_engagement(
            _pgd_fast_gradient_parameter_engagement, engagement, rows
        )
    )
    if missing_pgd_fast_gradient:
        fail(
            "pgd_fast_gradient_parameter_engaged",
            "not every measured row carried a closed sequential gradient receipt proving a "
            "live decision consumed the requested fast-gradient policy; wide exact-VJP, "
            "direct GAMA, structural bypasses, and zero-step executions are not parameter "
            "experiments",
        )

    requested_postbab_reserve = {
        name for name, _token in engagement if name == "NY_POSTBAB_RESERVE_SECS"
    }
    missing_postbab_reserve = requested_postbab_reserve.difference(
        _every_row_parameter_engagement(
            _postbab_reserve_parameter_engagement, engagement, rows
        )
    )
    if missing_postbab_reserve:
        fail(
            "postbab_reserve_parameter_engaged",
            "not every measured row carried a closed post-BaB reserve receipt proving the "
            "requested reserve was fully held, the verifier bound on its deadline, and "
            "post-BaB attack work started; configured-only, saturated, structurally "
            "bypassed, and mismatched reserves are not parameter experiments",
        )

    requested_root_joint = {
        name for name, _token in engagement
        if name in _ROOT_JOINT_INTERM_ALPHA_T1_FAMILY
    }
    missing_root_joint = requested_root_joint.difference(
        _every_row_parameter_engagement(
            _root_joint_interm_alpha_parameter_engagement, engagement, rows
        )
    )
    if missing_root_joint:
        fail(
            "root_joint_interm_alpha_parameter_engaged",
            "not every measured row carried one closed root-joint intermediate-alpha receipt "
            "proving nonzero targets and an actual bounded tightener dispatch for %s; "
            "configuration-only, capped-away seconds, wrong-route admission floors, "
            "zero-target, refused, malformed, contradictory, and mismatched rows are "
            "not parameter experiments"
            % ", ".join(sorted(missing_root_joint)),
        )

    requested_exact_vjp = {
        name for name, _token in engagement if name in _EXACT_VJP_FAMILY
    }
    missing_exact_vjp = requested_exact_vjp.difference(
        _every_row_parameter_engagement(
            _exact_vjp_parameter_engagement, engagement, rows
        )
    )
    if missing_exact_vjp:
        fail(
            "exact_vjp_parameter_engaged",
            "not every measured row carried a closed exact-VJP execution proving actual "
            "consumption for %s; structural bypasses, inactive feature gates, width/step "
            "saturation, and requested controls that never fired are not parameter experiments"
            % ", ".join(sorted(missing_exact_vjp)),
        )

    if require_closed_root_terminal and (
        not rows
        or not all(
            any(
                _root_joint_explicit_configuration_matches(event, wanted)
                for event in _root_joint_interm_alpha_terminal_evidences(row)
            )
            for row in rows
        )
    ):
        fail(
            "root_joint_interm_alpha_terminal_closed",
            "not every measured row in the prerequisite-only comparator has an exact "
            "closed root-joint execution/refusal receipt bound to every explicit root control; "
            "delivery alone cannot authenticate a control",
        )

    # T1.5 -- the PARENT's sealed ambient set. Presence and capture version are
    # both evidence: a legacy empty map did not inspect every Rayon worker-count
    # variable and therefore cannot be promoted into a current clean seal by
    # interpretation.
    manifest_ambient_schema = manifest_record.get("ambient_env_schema")
    if manifest_ambient_schema != AMBIENT_ENV_SCHEMA:
        fail(
            "manifest_ambient_provenance",
            "manifest ambient capture schema is %r, expected %r; a legacy/unknown capture "
            "cannot prove every Rayon worker-count variable was absent"
            % (manifest_ambient_schema, AMBIENT_ENV_SCHEMA),
        )
    ambient_env = manifest_record.get("ambient_env")
    if not isinstance(ambient_env, dict):
        fail(
            "manifest_ambient_provenance",
            "manifest has no ambient_env object; parent cleanliness is unproven",
        )
    else:
        for name, value in sorted(ambient_env.items()):
            if not (name.startswith("NY_") or name in RAYON_WORKER_ENV):
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
        if expected_configs is not None:
            sealed_configs = manifest.get("configs")
            if not isinstance(sealed_configs, dict):
                fail(
                    "manifest_matches_preset",
                    "manifest seals no configuration directory, expected %s"
                    % expected_configs["canonical_path"],
                )
            elif (sealed_configs.get("canonical_path") != expected_configs["canonical_path"]
                  or sealed_configs.get("sha256") != expected_configs["sha256"]):
                fail(
                    "manifest_matches_preset",
                    "manifest configuration identity %r != requested exact-year identity %r"
                    % (sealed_configs, expected_configs),
                )

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


def validate_artifacts(run_record, arm, category, opts, *, engagement_arm=None,
                       require_closed_root_terminal=False):
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
    expected_binary = getattr(opts, "selected_binary_identity", None)
    if expected_binary is not None:
        sealed_executable = (manifest or {}).get("executable")
        sealed_build_id = (manifest or {}).get("build_id")
        if (not isinstance(sealed_executable, dict)
                or sealed_executable.get("sha256") != expected_binary.get("sha256")
                or sealed_build_id != expected_binary.get("build_id")):
            failures.append({
                "check": "manifest_matches_selected_binary",
                "detail": (
                    "manifest executable/build identity %r / %r does not match selected "
                    "--ny-bin %s / %r"
                    % (
                        sealed_executable,
                        sealed_build_id,
                        expected_binary.get("sha256"),
                        expected_binary.get("build_id"),
                    )
                ),
            })
    selection = getattr(opts, "instance_selection", None)
    limit = getattr(opts, "limit", None)
    global_selection = requested_instance_selection(
        selection, category, limit, global_intent=True)
    category_selection = requested_instance_selection(selection, category, limit)
    measured_rows, subset_failures = validate_instance_selection(
        selection, category, limit, rows, manifest)
    failures.extend(subset_failures)
    input_seals = None
    fingerprint_cache = getattr(opts, "_input_fingerprint_cache", None)
    if fingerprint_cache is None:
        fingerprint_cache = {}
        setattr(opts, "_input_fingerprint_cache", fingerprint_cache)
    try:
        input_seals = validate_selected_row_inputs(
            manifest, category, rows, fingerprint_cache=fingerprint_cache)
    except ValueError as error:
        failures.append({
            "check": "manifest_selected_inputs_match_current_bytes",
            "detail": str(error),
        })
    expected_configs, expected_preset_path = None, None
    configured = getattr(opts, "configs_dir", None)
    if configured:
        try:
            expected_configs = fingerprint_tree(configured)
            expected_preset_path = resolve_preset_path(configured, category, opts.year)
            if expected_preset_path is None:
                raise ValueError(
                    "no %s preset for category %r under %s"
                    % (opts.year, category, configured)
                )
        except (OSError, RuntimeError, ValueError) as error:
            failures.append({"check": "requested_preset_present", "detail": str(error)})
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
        expected_configs=expected_configs,
        expected_preset_path=expected_preset_path,
        allow_ungraduated_execution_contracts=getattr(
            opts, "allow_ungraduated_execution_contracts", False),
        engagement_arm=engagement_arm,
        require_closed_root_terminal=require_closed_root_terminal,
    )
    failures.extend(t1_failures)
    corpus_context = None
    if isinstance(manifest, dict):
        # The manifest's instances_csv is the FULL source corpus, not the
        # selected subset. Seal both requested selector intent and the exact
        # metadata rows so equal-sized but different subsets cannot pair.
        corpus_context = {
            field: manifest.get(field)
            for field in (
                "year", "vnnlib_version", "timeout_cap", "configs", "instances_csv",
            )
        }
        corpus_context["instance_selection"] = global_selection
        corpus_context["category_instance_selection"] = category_selection
        corpus_context["measured_row_set"] = measured_rows
        corpus_context["selected_row_inputs_schema"] = manifest.get(
            "selected_row_inputs_schema")
        corpus_context["selected_row_inputs"] = input_seals
    qualified_arm = arm if engagement_arm is None else engagement_arm
    ungraduated = sorted(
        str(name) for name, _value in qualified_arm
        if str(name) not in _AXIS_SPECIFIC_T1_CONTRACTS
    )
    evidence_scope = (
        "delivery_only"
        if ungraduated and getattr(
            opts, "allow_ungraduated_execution_contracts", False)
        else "scored"
    )
    return {
        "valid": not failures,
        "evidence_scope": evidence_scope,
        "ungraduated_axes": ungraduated,
        "failures": failures,
        "warnings": warnings,
        "counts": recomputed,
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "executable_sha256": ((manifest or {}).get("executable") or {}).get("sha256"),
        "build_id": (manifest or {}).get("build_id"),
        "build_provenance": (manifest or {}).get("build_provenance"),
        "compute_backend": (manifest or {}).get("compute_backend"),
        "host": (manifest or {}).get("host"),
        "corpus_context": corpus_context,
        "instance_selection": global_selection,
        "category_instance_selection": category_selection,
        "measured_row_set": measured_rows,
        "selected_row_inputs": input_seals,
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
                 "parents": edge.parents or None, "bound": edge.bound,
                 "token": edge.token, "site": edge.site}
                for edge in space.edges
            ],
            "unsafe": space.unsafe_names,
            "instrument_only": space.instrument_only,
            "test_only": space.test_only,
            "declarations": kinds,
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
    print("      entry point exports no NY_* variables. Its result is worth zero")
    print("      points until a typed preset key exists.")


def _fully_measured_record(record, *, structural=False):
    """A current-schema result whose rows came from this child.

    Comparator eligibility is structural: one executable assignment may also
    be a treatment whose own contrast failed parameter engagement. Treatment
    admission remains contrast-specific in `paired_repeat_aggregates`.
    """
    counts = record.get("counts")
    validity_field = "structurally_valid" if structural else "valid"
    return (
        record.get("schema") == RESULT_SCHEMA
        and record.get(validity_field) is True
        and isinstance(counts, dict)
        and counts.get("cached_rows") == 0
    )


def _measurement_context(record, *, include_repeat):
    """Blocking identity for one pair or a repeat aggregate.

    The exact repeat number belongs in pair matching but not in the grouping
    key across repeats.  Every other measurement invariant stays in both keys,
    including the complete causal-design identity.
    """
    fields = (
        "category", "repeat", "year", "executable_sha256", "build_id",
        "compute_backend", "host", "corpus_context", "search_design_schema",
        "search_design_sha256",
    )
    if any(record.get(field) is None for field in fields):
        return None
    if (
        record.get("search_design_schema") != SEARCH_DESIGN_SCHEMA
        or not isinstance(record.get("search_design_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", record["search_design_sha256"]) is None
    ):
        return None

    # Backend and host are strings today. Canonical JSON keeps this comparison
    # exact if either manifest field grows into a structured identity later.
    sealed = tuple(
        json.dumps(record[field], sort_keys=True, separators=(",", ":"))
        for field in ("compute_backend", "host", "corpus_context")
    )
    counts = record.get("counts") or {}
    context = (
        record["category"], record["year"],
        record["executable_sha256"], record["build_id"], sealed,
        counts.get("rows"), counts.get("capped_rows"), record.get("lower_bound"),
        record["search_design_schema"], record["search_design_sha256"],
    )
    return (context[:1] + (record["repeat"],) + context[1:]
            if include_repeat else context)


def _paired_context(record):
    """Blocking identity for an attributable treatment/comparator pair."""
    return _measurement_context(record, include_repeat=True)


def _aggregate_context(record):
    """All pair invariants except the repeat coordinate being summarized."""
    return _measurement_context(record, include_repeat=False)


def _record_solved(record):
    counts = record.get("counts") or {}
    sat, unsat = counts.get("sat"), counts.get("unsat")
    if not isinstance(sat, int) or not isinstance(unsat, int):
        return None
    return sat + unsat


def _canonical_record_assignment(raw):
    if not isinstance(raw, list):
        return None
    if any(
        not isinstance(pair, list)
        or len(pair) != 2
        or not isinstance(pair[0], str)
        or re.fullmatch(r"NY_[A-Z0-9_]+", pair[0]) is None
        or not isinstance(pair[1], str)
        or not pair[1]
        or not pair[1].isascii()
        or pair[1] != pair[1].strip()
        or any(character.isspace() for character in pair[1])
        for pair in raw
    ):
        return None
    pairs = tuple((pair[0], pair[1]) for pair in raw)
    names = [name for name, _value in pairs]
    return (
        pairs
        if pairs == tuple(sorted(pairs)) and len(set(names)) == len(names)
        else None
    )


def _contrast_contract_valid(contrast, treatment_assignment):
    if not isinstance(contrast, dict):
        return False
    required = {
        "schema", "requested_axes", "requested_assignment",
        "prerequisites_added", "treatment_assignment",
        "comparison_assignment", "contrast_id", "evidence_scope",
        "ungraduated_axes",
    }
    if set(contrast) != required or contrast.get("schema") != SEARCH_DESIGN_SCHEMA:
        return False
    requested = _canonical_record_assignment(contrast.get("requested_assignment"))
    prerequisites = _canonical_record_assignment(contrast.get("prerequisites_added"))
    treatment = _canonical_record_assignment(contrast.get("treatment_assignment"))
    comparison = _canonical_record_assignment(contrast.get("comparison_assignment"))
    if None in (requested, prerequisites, treatment, comparison):
        return False
    expected_ungraduated = sorted(
        name for name in contrast.get("requested_axes", [])
        if name not in _AXIS_SPECIFIC_T1_CONTRACTS
    )
    expected_scope = "delivery_only" if expected_ungraduated else "scored"
    if (
        not requested
        or treatment != treatment_assignment
        or comparison != prerequisites
        or tuple(sorted(requested + prerequisites)) != treatment
        or contrast.get("requested_axes") != sorted(name for name, _ in requested)
        or contrast.get("evidence_scope") != expected_scope
        or contrast.get("ungraduated_axes") != expected_ungraduated
    ):
        return False
    identity_payload = {
        field: contrast[field]
        for field in (
            "schema", "requested_axes", "requested_assignment",
            "prerequisites_added", "treatment_assignment",
            "comparison_assignment",
        )
    }
    expected_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return contrast.get("contrast_id") == expected_id


_CONTRAST_EVIDENCE_FIELDS = frozenset({
    "contrast_id", "evidence_scope", "valid", "failures", "warnings",
})
_RESULT_FINDING_FIELDS = frozenset({"check", "detail"})


def _result_findings_valid(findings):
    return (
        isinstance(findings, list)
        and all(
            isinstance(finding, dict)
            and set(finding) == _RESULT_FINDING_FIELDS
            and isinstance(finding["check"], str)
            and bool(finding["check"])
            and isinstance(finding["detail"], str)
            and bool(finding["detail"])
            for finding in findings
        )
    )


def _validated_result_design_roles(record, assignment):
    """Return exact per-record causal roles, or None on any ambiguity.

    Result JSONL is routinely consumed after the producing process exits.  Its
    redundant role fields therefore remain a fail-closed contract rather than
    hints: duplicate evidence, partial evidence objects, or a stale qualified
    list must not be reinterpreted as a scored contrast.
    """
    contrasts = record.get("contrasts")
    comparison_for = record.get("comparison_for")
    evidences = record.get("contrast_evidence")
    scored_qualified = record.get("scored_qualified_contrasts")
    if (
        record.get("arm_id") != arm_id(assignment)
        or not isinstance(contrasts, list)
        or not isinstance(comparison_for, list)
        or not isinstance(evidences, list)
        or not isinstance(scored_qualified, list)
        or not all(
            isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{16}", value) is not None
            for value in comparison_for
        )
        or comparison_for != sorted(set(comparison_for))
    ):
        return None

    contrast_ids = []
    by_id = {}
    for contrast in contrasts:
        if not _contrast_contract_valid(contrast, assignment):
            return None
        contrast_id = contrast["contrast_id"]
        if contrast_id in by_id:
            return None
        contrast_ids.append(contrast_id)
        by_id[contrast_id] = contrast
    if contrast_ids != sorted(contrast_ids):
        return None

    evidence_ids = []
    evidence_by_id = {}
    for evidence in evidences:
        if (
            not isinstance(evidence, dict)
            or set(evidence) != _CONTRAST_EVIDENCE_FIELDS
            or not isinstance(evidence.get("contrast_id"), str)
            or evidence.get("contrast_id") not in by_id
            or type(evidence.get("valid")) is not bool
            or evidence.get("evidence_scope")
                != by_id[evidence["contrast_id"]]["evidence_scope"]
            or not _result_findings_valid(evidence.get("failures"))
            or not _result_findings_valid(evidence.get("warnings"))
            or (evidence["valid"] and evidence["failures"])
            or (not evidence["valid"] and not evidence["failures"])
            or evidence["contrast_id"] in evidence_by_id
        ):
            return None
        evidence_ids.append(evidence["contrast_id"])
        evidence_by_id[evidence["contrast_id"]] = evidence
    if evidence_ids != contrast_ids:
        return None

    expected_scored = [
        contrast_id for contrast_id in contrast_ids
        if evidence_by_id[contrast_id]["valid"]
        and by_id[contrast_id]["evidence_scope"] == "scored"
    ]
    if scored_qualified != expected_scored:
        return None
    scopes = {contrast["evidence_scope"] for contrast in contrasts}
    expected_record_scope = (
        next(iter(scopes)) if len(scopes) == 1
        else "mixed" if scopes else "scored"
    )
    if record.get("evidence_scope") != expected_record_scope:
        return None
    if (
        type(record.get("structurally_valid")) is not bool
        or type(record.get("valid")) is not bool
        or record["valid"] != (
            record["structurally_valid"]
            and (not contrasts or any(evidence["valid"] for evidence in evidences))
        )
    ):
        return None
    return {
        "contrasts": contrasts,
        "comparison_for": comparison_for,
        "evidence_by_id": evidence_by_id,
    }


def paired_repeat_aggregates(records):
    """Aggregate each sealed causal contrast against its exact comparator.

    A pair is admitted only when both records are structurally valid, fully measured, and
    share category, repeat, year, executable, backend, host, corpus row count,
    official/lower-bound status, and search-design identity. Duplicate exact
    assignments make that block ambiguous and therefore unusable. Delivery-only
    contrasts are never scored.
    """
    measured = [
        record for record in records
        if _fully_measured_record(record, structural=True)
    ]
    assignments, roles_by_record, ambiguous = {}, {}, set()
    for record in measured:
        context = _paired_context(record)
        assignment = _canonical_record_assignment(record.get("arm"))
        if context is None or assignment is None or _record_solved(record) is None:
            continue
        roles = _validated_result_design_roles(record, assignment)
        if roles is None:
            continue
        roles_by_record[id(record)] = roles
        key = (context, assignment)
        if key in assignments:
            ambiguous.add(key)
        else:
            assignments[key] = record

    grouped = {}
    for treatment in measured:
        treatment_assignment = _canonical_record_assignment(treatment.get("arm"))
        context = _paired_context(treatment)
        aggregate_context = _aggregate_context(treatment)
        roles = roles_by_record.get(id(treatment))
        if (
            treatment_assignment is None
            or context is None
            or aggregate_context is None
            or roles is None
        ):
            continue
        treatment_solved = _record_solved(treatment)
        if treatment_solved is None:
            continue
        for contrast in roles["contrasts"]:
            evidence = roles["evidence_by_id"][contrast["contrast_id"]]
            if (
                evidence.get("valid") is not True
                or evidence.get("evidence_scope") != "scored"
                or contrast["evidence_scope"] != "scored"
            ):
                continue
            comparator_assignment = _canonical_record_assignment(
                contrast["comparison_assignment"])
            key = (context, comparator_assignment)
            comparator = None if key in ambiguous else assignments.get(key)
            comparator_solved = _record_solved(comparator or {})
            if comparator is None or comparator_solved is None:
                continue
            comparator_roles = roles_by_record.get(id(comparator))
            if (
                comparator_roles is None
                or contrast["contrast_id"] not in comparator_roles["comparison_for"]
            ):
                continue
            if (
                treatment.get("search_design_schema") != SEARCH_DESIGN_SCHEMA
                or comparator.get("search_design_schema") != SEARCH_DESIGN_SCHEMA
                or not isinstance(treatment.get("search_design_sha256"), str)
                or treatment.get("search_design_sha256")
                    != comparator.get("search_design_sha256")
            ):
                continue
            grouped.setdefault(
                (aggregate_context, contrast["contrast_id"]),
                {
                    "category": treatment["category"],
                    "contrast": contrast,
                    "pairs": [],
                },
            )["pairs"].append((treatment_solved, comparator_solved))

    aggregates = []
    for (aggregate_context, contrast_id), group in grouped.items():
        contrast = group["contrast"]
        pairs = group["pairs"]
        arm = _canonical_record_assignment(contrast["treatment_assignment"])
        comparator = _canonical_record_assignment(contrast["comparison_assignment"])
        treatment_scores = [pair[0] for pair in pairs]
        comparator_scores = [pair[1] for pair in pairs]
        deltas = [treatment - control for treatment, control in pairs]
        aggregates.append({
            "category": group["category"],
            "contrast_id": contrast_id,
            "measurement_context": aggregate_context,
            "arm": arm,
            "arm_id": arm_id(arm),
            "arm_label": arm_label(arm),
            "requested_assignment": contrast["requested_assignment"],
            "prerequisites_added": contrast["prerequisites_added"],
            "comparison_assignment": comparator,
            "comparison_arm_id": arm_id(comparator),
            "comparison_arm_label": arm_label(comparator),
            "pairs": len(pairs),
            "treatment_median": statistics.median(treatment_scores),
            "baseline_median": statistics.median(comparator_scores),
            "comparison_median": statistics.median(comparator_scores),
            "delta_median": statistics.median(deltas),
            "delta_min": min(deltas),
            "delta_max": max(deltas),
            "wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
        })
    return sorted(
        aggregates,
        key=lambda item: (
            item["category"], -item["delta_median"], -item["delta_min"],
            item["contrast_id"],
        ),
    )


def print_paired_repeat_aggregates(records):
    aggregates = paired_repeat_aggregates(records)
    print()
    print("paired fully-measured solved deltas vs each sealed exact comparator")
    print("  exact repeat + corpus/config + binary + backend + host + design; "
          "invalid, cached, delivery-only, or unmatched omitted")
    if not aggregates:
        print("  -- no attributable treatment/comparator pairs --")
        return
    category = None
    for aggregate in aggregates:
        if aggregate["category"] != category:
            category = aggregate["category"]
            print("  %s" % category)
        print(
            "    delta med %+5.1f [%+d,%+d]  pairs %2d  W/T/L %d/%d/%d  "
            "solved med %.1f vs %.1f  %-16s %s  [control %s]"
            % (
                aggregate["delta_median"], aggregate["delta_min"],
                aggregate["delta_max"], aggregate["pairs"], aggregate["wins"],
                aggregate["ties"], aggregate["losses"],
                aggregate["treatment_median"], aggregate["baseline_median"],
                aggregate["contrast_id"], aggregate["arm_label"],
                aggregate["comparison_arm_label"],
            )
        )


def print_leaderboard(records):
    valid = [
        record for record in records
        if record.get("schema") == RESULT_SCHEMA and record.get("valid") is True
    ]
    invalid = [record for record in records if record not in valid]
    delivery_only = [
        record for record in valid
        if record.get("contrasts")
        and not record.get("scored_qualified_contrasts")
    ]
    scored = [record for record in valid if record not in delivery_only]
    print()
    print("=" * 78)
    print("%d run(s): %d scored/control valid, %d delivery-only excluded, "
          "%d INVALID and excluded"
          % (len(records), len(scored), len(delivery_only), len(invalid)))
    for record in delivery_only:
        print("  DELIVERY-ONLY %-10s %-16s %s"
              % (record["arm_id"], record["category"], record["arm_label"]))
    for record in invalid:
        print("  INVALID %-10s %-16s %s"
              % (record["arm_id"], record["category"],
                 "; ".join("%s: %s" % (f["check"], f["detail"]) for f in record["failures"])[:160]))
    binaries = {
        record.get("executable_sha256") for record in scored if record.get("executable_sha256")
    }
    if len(binaries) > 1:
        print("REFUSING to rank: these runs used %d DIFFERENT ny binaries (%s). Arms measured "
              "with different executables are not comparable."
              % (len(binaries), ", ".join(sorted(sha[:8] for sha in binaries))))
        return
    by_category = {}
    for record in scored:
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
    print_paired_repeat_aggregates(records)
    # Only a run whose rows were ALL measured counts as an independent repeat.
    # T1.10 invalidates a fully replayed run, but a PARTIALLY replayed one stays
    # valid, and admission rule 2 makes that reachable without tampering: a row
    # measured under load is stored and then refused on read, so on a busy box
    # repeats 2 and 3 can be mostly byte-copies of repeat 1. Counting those
    # toward the ">= 3 runs per arm" caveat would let the variance control the
    # design doc's W6 rests on be satisfied by replaying one measurement.
    repeats, partial = {}, {}
    for record in scored:
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
    if scored and worst < 3:
        print()
        print("CAVEAT: at most %d fully-measured run(s) per (arm, category). A one- or two-row "
              "difference is NOT a result at this sample size — a single run cannot see a "
              "1-in-3 failure, and marginal rows need >= 3 runs per arm." % worst)


# --------------------------------------------------------------------------
# Self-test: arm generation and the T1 validators, against synthetic artifacts
# --------------------------------------------------------------------------


def _clean_row(arm=(), verdict="unsat", category="cifar100"):
    """A synthetic bank row with flight-v3 and the current ambient sub-schema."""
    ambient = {name: value for name, value in arm}
    ambient["OMP_NUM_THREADS"] = "1"
    receipt_entries = [
        {
            "name": "NY_SYNTHETIC_UNREQUESTED",
            "value": False,
            "source": "default",
            "bucket": "auto",
            "moat": "low",
            "provenance": "unmeasured",
        },
        *[
            {
                "name": name,
                "value": value,
                "source": "legacy_env",
                "bucket": "auto",
                "moat": "low",
                "provenance": "unmeasured",
                "env_utf8": True,
            }
            for name, value in sorted(arm)
        ],
    ]
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
            "ambient_env_schema": AMBIENT_ENV_SCHEMA,
            "ambient_env": ambient,
            "levers": {
                "status": "resolved",
                "receipt": {
                    "schema": "ny-levers/receipt/v3",
                    "lever_count": len(receipt_entries),
                    "env_present": len(arm),
                    "env_accepted": len(arm),
                    "env_rejected": 0,
                    "env_ignored": 0,
                    "levers": receipt_entries,
                },
            },
        },
        **({"witness": {"path": "witnesses/w.counterexample"}} if verdict == "sat" else {}),
    }


def _clean_manifest(year=2025, category="cifar100", arm=()):
    return {
        "schema_version": 1,
        "year": year,
        "timeout_cap": None,
        "executable": {"canonical_path": "/repo/target/release/ny", "sha256": "a" * 64},
        "build_id": SELF_TEST_BUILD_ID,
        "build_provenance": "test",
        "compute_backend": "cuda",
        "host": "gb10",
        "arm": [list(pair) for pair in sorted(arm)],
        "ambient_env_schema": AMBIENT_ENV_SCHEMA,
        "ambient_env": {},
        "instances_csv": {category: {"canonical_path": "/i.csv", "sha256": "b" * 64}},
    }


def _clean_manifest_for_rows(rows, instance_selection=None, input_artifacts=None):
    """Synthetic current native manifest with the planned-row proof populated."""
    if not rows:
        raise ValueError("synthetic selected-row manifest needs at least one row")
    category = rows[0]["category"]
    manifest = _clean_manifest(category=category, arm=rows[0].get("arm", []))
    manifest["instance_selection"] = instance_selection or {"mode": "all"}
    manifest["selected_rows"] = {category: measured_row_set_identity(rows)}
    if input_artifacts is not None:
        manifest["selected_row_inputs_schema"] = SELECTED_ROW_INPUTS_SCHEMA
        manifest["selected_row_inputs"] = {
            category: [
                {
                    "onnx": row["onnx"],
                    "vnnlib": row["vnnlib"],
                    "occurrence": row["occurrence"],
                    "instance_index": row["instance_index"],
                    "inputs": {
                        "onnx": [{
                            "artifact": input_artifacts["onnx"],
                            "external_data": input_artifacts["onnx_external_data"],
                        }],
                        "vnnlib": input_artifacts["vnnlib"],
                    },
                }
                for row in rows
            ],
        }
    return manifest


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
    kinds = lever_kinds_for_space(space)

    def synthetic_family_engagement(_wanted, singleton_rows):
        return singleton_rows[0]["engaged"]

    synthetic_family_rows = [
        {"engaged": {"NY_SYNTHETIC_A", "NY_SYNTHETIC_B"}},
        {"engaged": {"NY_SYNTHETIC_A"}},
    ]
    check(
        "all-row engagement intersects each axis of a multi-axis family",
        _every_row_parameter_engagement(
            synthetic_family_engagement,
            [("NY_SYNTHETIC_A", "1"), ("NY_SYNTHETIC_B", "1")],
            synthetic_family_rows,
        ) == {"NY_SYNTHETIC_A"}
        and not _every_row_parameter_engagement(
            synthetic_family_engagement,
            [("NY_SYNTHETIC_A", "1")],
            [*synthetic_family_rows, {"engaged": set()}],
        ),
    )

    # --- source/binary search-contract ingestion
    parsed_strings = _rust_str_list(
        '"NY_ONE", // "comment prose"\n'
        '/* outer "also prose" /* nested "still prose" */ */ "https://example.test/a", '
        '"NY_TWO"'
    )
    check("Rust string-list parsing ignores comments but not markers in strings",
          parsed_strings == ["NY_ONE", "https://example.test/a", "NY_TWO"],
          repr(parsed_strings))
    authoritative = _declarations_from_payload({
        "NY_ONE": {"kind": "F64ClosedTrimmed", "default": "0.4",
                   "min": 0.0, "max": 1.0, "members": None}
    })
    binary_space = Space([], [], [], [], [], "synthetic binary",
                         declarations=authoritative)
    check("binary declarations override source scraping without merging",
          lever_kinds_for_space(binary_space) == authoritative
          and authoritative["NY_ONE"]["default"] == "0.4")
    complete_payload = {
        "schema": SPACE_SCHEMA,
        "axes": [{"name": "NY_ONE", "domain": "bool", "values": ["0", "1"],
                  "class": "VerdictAffecting", "deliver": "env-only", "why": "test",
                  "absence_baseline": {"kind": "declaration_default"},
                  "category_prefixes": [], "preset_baseline_keys": []}],
        "edges": [],
        "unsafe": [],
        "instrument_only": [],
        "test_only": [],
        "declarations": {"NY_ONE": {"kind": "Bool", "default": "0"}},
    }
    compiled_space = _space_from_binary_payload(complete_payload, "ny-synthetic")
    check("complete versioned binary schema is accepted without source data",
          compiled_space.source.startswith("binary `ny-synthetic ")
          and compiled_space.declarations == {
              "NY_ONE": {"kind": "Bool", "default": "0"}
          })
    or_payload = dict(complete_payload)
    or_payload["axes"] = [
        {"name": name, "domain": "bool", "values": ["0", "1"],
         "class": "VerdictAffecting", "deliver": "env-only", "why": "test",
         "absence_baseline": {"kind": "declaration_default"},
         "category_prefixes": [], "preset_baseline_keys": []}
        for name in ("NY_CHILD", "NY_DIRECT", "NY_ADAPTIVE")
    ]
    or_payload["edges"] = [
        {
            "child": "NY_CHILD", "kind": "any_armed", "parent": None,
            "parents": ["NY_DIRECT", "NY_ADAPTIVE"], "bound": None,
            "token": None, "site": "test OR",
        },
        {
            "child": "NY_DIRECT", "kind": "not_equals", "parent": "NY_ADAPTIVE",
            "parents": None, "bound": None, "token": "0", "site": "test not-equals",
        },
        {
            "child": "NY_ADAPTIVE", "kind": "absent", "parent": "NY_DIRECT",
            "parents": None, "bound": None, "token": None, "site": "test absent",
        },
    ]
    or_payload["declarations"] = {
        name: {"kind": "Bool", "default": "0"}
        for name in ("NY_CHILD", "NY_DIRECT", "NY_ADAPTIVE")
    }
    compiled_or = _space_from_binary_payload(or_payload, "ny-or-synthetic")
    check("compiled schema preserves OR and source-sensitive dependencies",
          len(compiled_or.edges) == 3
          and compiled_or.edges[0].kind == "any_armed"
          and compiled_or.edges[0].parent is None
          and compiled_or.edges[0].parents == ["NY_DIRECT", "NY_ADAPTIVE"]
          and compiled_or.edges[1].kind == "not_equals"
          and compiled_or.edges[1].token == "0"
          and compiled_or.edges[2].kind == "absent")
    sentinel_payload = dict(complete_payload)
    sentinel_payload["axes"] = [
        {"name": "NY_ONE", "domain": "grid", "values": ["1", "2"],
         "class": "VerdictAffecting", "deliver": "env-only", "why": "test",
         "absence_baseline": {"kind": "declaration_default"},
         "category_prefixes": [], "preset_baseline_keys": []},
    ]
    sentinel_payload["edges"] = [{
        "child": "NY_ONE", "kind": "not_equals", "parent": "NY_ONE",
        "parents": None, "bound": None, "token": "0",
        "site": "outside-domain inert sentinel",
    }]
    sentinel_payload["declarations"] = {
        "NY_ONE": {"kind": "U64Trimmed", "default": "1"},
    }
    compiled_sentinel = _space_from_binary_payload(
        sentinel_payload, "ny-sentinel-synthetic"
    )
    plus_sentinel_payload = json.loads(json.dumps(sentinel_payload))
    plus_sentinel_payload["declarations"]["NY_ONE"]["kind"] = "U64"
    plus_sentinel_payload["edges"][0]["token"] = "+0"
    compiled_plus_sentinel = _space_from_binary_payload(
        plus_sentinel_payload, "ny-plus-sentinel-synthetic"
    )
    rejected_bad_sentinel = False
    bad_sentinel_payload = dict(sentinel_payload)
    bad_sentinel_payload["edges"] = [dict(sentinel_payload["edges"][0])]
    bad_sentinel_payload["edges"][0]["token"] = "not-a-u64"
    try:
        _space_from_binary_payload(bad_sentinel_payload, "ny-bad-sentinel-synthetic")
    except ValueError:
        rejected_bad_sentinel = True
    rejected_overflow_sentinel = False
    overflow_sentinel_payload = json.loads(json.dumps(plus_sentinel_payload))
    overflow_sentinel_payload["edges"][0]["token"] = "18446744073709551616"
    try:
        _space_from_binary_payload(
            overflow_sentinel_payload, "ny-overflow-sentinel-synthetic"
        )
    except ValueError:
        rejected_overflow_sentinel = True
    rejected_presence_repair = False
    presence_sentinel_payload = json.loads(json.dumps(sentinel_payload))
    presence_sentinel_payload["axes"][0]["values"] = ["1"]
    presence_sentinel_payload["declarations"]["NY_ONE"] = {
        "kind": "Presence", "default": "0",
    }
    try:
        _space_from_binary_payload(
            presence_sentinel_payload, "ny-presence-sentinel-synthetic"
        )
    except ValueError:
        rejected_presence_repair = True
    no_sentinel_repair = False
    no_repair_payload = dict(sentinel_payload)
    no_repair_payload["axes"] = [dict(sentinel_payload["axes"][0])]
    no_repair_payload["axes"][0]["domain"] = "enum"
    no_repair_payload["axes"][0]["values"] = ["00"]
    try:
        _space_from_binary_payload(no_repair_payload, "ny-no-repair-synthetic")
    except ValueError:
        no_sentinel_repair = True
    alias_payload = dict(no_repair_payload)
    alias_payload["axes"] = [dict(no_repair_payload["axes"][0])]
    alias_payload["axes"][0]["values"] = ["00", "1"]
    alias_space = _space_from_binary_payload(alias_payload, "ny-alias-synthetic")
    alias_is_inert = is_inert({"NY_ONE": "00"}, alias_space)
    alias_repair = _satisfying_value(
        alias_space.by_name["NY_ONE"],
        alias_space.edges[0],
        alias_space.declarations,
    )
    check("compiled NotEquals accepts a parser-valid sentinel outside the search domain",
          compiled_sentinel.edges[0].token == "0"
          and compiled_plus_sentinel.edges[0].token == "+0"
          and rejected_bad_sentinel and rejected_overflow_sentinel
          and rejected_presence_repair and no_sentinel_repair
          and alias_is_inert is not None and alias_repair == "1")
    old_version_payload = dict(complete_payload)
    old_version_payload["schema"] = "ny-levers/search-space/v5"
    rejected_old_version = False
    try:
        _space_from_binary_payload(old_version_payload, "ny-v5")
    except ValueError:
        rejected_old_version = True
    check("v5 binary schema fails closed after preset baseline chains became required",
          rejected_old_version)
    missing_absence_payload = dict(complete_payload)
    missing_absence_axis = dict(complete_payload["axes"][0])
    missing_absence_axis.pop("absence_baseline")
    missing_absence_payload["axes"] = [missing_absence_axis]
    rejected_missing_absence = False
    try:
        _space_from_binary_payload(missing_absence_payload, "ny-incomplete")
    except ValueError:
        rejected_missing_absence = True
    check("compiled schema requires typed absence metadata on every axis",
          rejected_missing_absence)
    missing_category_payload = dict(complete_payload)
    missing_category_axis = dict(complete_payload["axes"][0])
    missing_category_axis.pop("category_prefixes")
    missing_category_payload["axes"] = [missing_category_axis]
    rejected_missing_category = False
    try:
        _space_from_binary_payload(missing_category_payload, "ny-incomplete")
    except ValueError:
        rejected_missing_category = True
    check("compiled schema requires typed category applicability on every axis",
          rejected_missing_category)
    missing_preset_baseline_payload = dict(complete_payload)
    missing_preset_baseline_axis = dict(complete_payload["axes"][0])
    missing_preset_baseline_axis.pop("preset_baseline_keys")
    missing_preset_baseline_payload["axes"] = [missing_preset_baseline_axis]
    rejected_missing_preset_baseline = False
    try:
        _space_from_binary_payload(missing_preset_baseline_payload, "ny-incomplete")
    except ValueError:
        rejected_missing_preset_baseline = True
    check("compiled schema requires preset baseline keys on every axis",
          rejected_missing_preset_baseline)
    layered_payload = dict(complete_payload)
    layered_payload["axes"] = [{
        "name": "NY_LAYERED", "domain": "enum", "values": ["min", "max"],
        "class": "VerdictAffecting", "deliver": "preset:bab.branching.reduceop",
        "why": "test", "absence_baseline": {"kind": "declaration_default"},
        "category_prefixes": [],
        "preset_baseline_keys": ["bab.branching.reduceop"],
    }]
    layered_payload["declarations"] = {
        "NY_LAYERED": {"kind": "LayeredEnum", "default": "min",
                       "members": ["min", "max"]}
    }
    layered_binary_space = _space_from_binary_payload(layered_payload, "ny-layered")
    check("compiled schema accepts LayeredEnum environment members",
          layered_binary_space.declarations["NY_LAYERED"]["members"] == ["min", "max"]
          and value_admissible(layered_binary_space.declarations["NY_LAYERED"], "max")
          and not value_admissible(layered_binary_space.declarations["NY_LAYERED"], "mean"))
    with tempfile.TemporaryDirectory() as tmp:
        identity_binary = Path(tmp) / "ny-identity"
        identity_binary.write_text(
            "#!/usr/bin/env python3\n"
            "import hashlib, pathlib\n"
            "path = pathlib.Path(__file__).resolve()\n"
            "digest = hashlib.sha256(path.read_bytes()).hexdigest()\n"
            "print('NY_BUILD_ID_V1 exe=%s id=%s' % (digest, '0' * 64))\n",
            encoding="utf-8",
        )
        identity_binary.chmod(0o755)
        probed_identity = probe_binary_identity(identity_binary)
        check("selected binary identity binds readable build id to exact file bytes",
              probed_identity["sha256"] == fingerprint_file(identity_binary)["sha256"]
              and probed_identity["build_id"].startswith("NY_BUILD_ID_V1 "),
              repr(probed_identity))

        # Model an existing binary from the previous CLI generation: it can
        # publish a plausible space under an obsolete spelling, but it does not
        # implement the measured-binary contract used by this checkout.
        stale_binary = Path(tmp) / "ny-stale"
        stale_binary.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "if sys.argv[1:] == ['levers', 'space', '--json']:\n"
            "    print(json.dumps({'axes': [{'name': 'NY_STALE'}]}))\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(2)\n",
            encoding="utf-8",
        )
        stale_binary.chmod(0o755)
        rejected_stale, rejection_detail = False, ""
        try:
            load_space(stale_binary)
        except SpaceSchemaUnavailable as error:
            rejected_stale = True
            rejection_detail = str(error)
        check("an existing stale binary never inherits the checkout source schema",
              rejected_stale and "benchmarks space --json" in rejection_detail,
              rejection_detail)
        developer_space = load_space(stale_binary, allow_source_schema=True)
        check("source schema fallback requires explicit developer opt-in",
              developer_space.source.startswith("parsed ")
              and not any(axis.name == "NY_STALE" for axis in developer_space.axes),
              developer_space.source)

        incomplete_binary = Path(tmp) / "ny-incomplete"
        incomplete_payload = dict(complete_payload)
        incomplete_payload.pop("declarations")
        incomplete_binary.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "payload = %r\n"
            "if sys.argv[1:] == ['benchmarks', 'space', '--json']:\n"
            "    print(json.dumps(payload))\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(2)\n" % incomplete_payload,
            encoding="utf-8",
        )
        incomplete_binary.chmod(0o755)
        rejected_incomplete = False
        try:
            load_space(incomplete_binary)
        except SpaceSchemaUnavailable:
            rejected_incomplete = True
        check("current subcommand without complete declarations fails closed",
              rejected_incomplete)

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
    inert = is_inert({"NY_MARGIN_ROW_GPU_BATCH": "0"}, space)
    check("a child control still requires its parent",
          inert and inert.axis == "NY_MARGIN_ROW_GPU_BATCH"
          and not is_inert({"NY_MARGIN_ROW_GPU": "1",
                            "NY_MARGIN_ROW_GPU_BATCH": "0"}, space))
    inert = is_inert({"NY_MARGIN_ROW_BETA": "1",
                      "NY_MARGIN_ROW_BETA_ETA": "0.5"}, space)
    check("legacy beta eta requires an explicit Polyak selector",
          inert and inert.axis == "NY_MARGIN_ROW_BETA_ETA"
          and not is_inert({"NY_MARGIN_ROW_BETA": "1",
                            "NY_MARGIN_ROW_BETA_POLYAK": "0",
                            "NY_MARGIN_ROW_BETA_ETA": "0.5"}, space))
    eta_edges = space.edges_by_child.get("NY_MARGIN_ROW_BETA_ETA", [])
    eta_arming = next((edge for edge in eta_edges if edge.kind == "any_armed"), None)
    check("source schema preserves ordered beta OR arming paths",
          eta_arming is not None
          and eta_arming.parents == [
              "NY_MARGIN_ROW_BETA", "NY_MARGIN_ROW_CLOSURE_ADAPTIVE"])
    check("closure-adaptive beta also makes legacy eta live",
          not is_inert({"NY_MARGIN_ROW_CLOSURE_ADAPTIVE": "1",
                        "NY_MARGIN_ROW_BETA_POLYAK": "0",
                        "NY_MARGIN_ROW_BETA_ETA": "0.5"}, space))
    check("an Unsafe axis is refused",
          (is_inert({"NY_STRIP_TERMINAL_SOFTMAX": "1"}, space) or None)
          and "Unsafe" in is_inert({"NY_STRIP_TERMINAL_SOFTMAX": "1"}, space).because)

    # --- repair closes a sample constructively rather than by rejection
    repaired, added = repair({"NY_MARGIN_ROW_GPU_BATCH": "1"}, space, kinds)
    check("repair arms the prerequisite",
          repaired.get("NY_MARGIN_ROW_GPU") == "1" and added == {"NY_MARGIN_ROW_GPU"}
          and not is_inert(repaired, space), str(repaired))
    repaired, added = repair({"NY_MARGIN_ROW_GPU_BATCH": "0"}, space, kinds)
    check("repair also closes a child control under its prerequisite",
          repaired.get("NY_MARGIN_ROW_GPU") == "1"
          and repaired.get("NY_MARGIN_ROW_GPU_BATCH") == "0"
          and added == {"NY_MARGIN_ROW_GPU"} and not is_inert(repaired, space),
          str(repaired))
    repaired, added = repair({"NY_MARGIN_ROW_BETA_ETA": "0.5"}, space, kinds)
    check("repair selects the exact legacy beta mode",
          repaired.get("NY_MARGIN_ROW_BETA") == "1"
          and repaired.get("NY_MARGIN_ROW_BETA_POLYAK") == "0"
          and {"NY_MARGIN_ROW_BETA", "NY_MARGIN_ROW_BETA_POLYAK"}.issubset(added)
          and not is_inert(repaired, space), str(repaired))
    repaired, added = repair({"NY_MARGIN_ROW_CLOSURE_ADAPTIVE": "1",
                              "NY_MARGIN_ROW_BETA_ETA": "0.5"}, space, kinds)
    check("repair keeps an explicit closure satisfier without root-time beta",
          repaired.get("NY_MARGIN_ROW_CLOSURE_ADAPTIVE") == "1"
          and repaired.get("NY_MARGIN_ROW_BETA") is None
          and repaired.get("NY_MARGIN_ROW_BETA_POLYAK") == "0"
          and added == {"NY_MARGIN_ROW_BETA_POLYAK"}
          and not is_inert(repaired, space), str(repaired))
    repaired, added = repair(
        {"NY_MARGIN_ROW_BETA_ETA": "0.5"}, space, kinds,
        baseline_values={"NY_MARGIN_ROW_CLOSURE_ADAPTIVE": "1"})
    check("repair re-emits an armed closure baseline instead of adding direct beta",
          repaired.get("NY_MARGIN_ROW_CLOSURE_ADAPTIVE") == "1"
          and repaired.get("NY_MARGIN_ROW_BETA") is None
          and repaired.get("NY_MARGIN_ROW_BETA_POLYAK") == "0"
          and added == {
              "NY_MARGIN_ROW_CLOSURE_ADAPTIVE", "NY_MARGIN_ROW_BETA_POLYAK"}
          and not is_inert(repaired, space), str(repaired))
    # The comprehensive sweep can never be armed while the resident phase owns
    # the slot, so the chunks axis that depends on it has to go -- and repair
    # must NOT oscillate between arming and dropping the parent.
    repaired, _added = repair(
        {"NY_ROOT_PHASE_RESIDENT_CROWN": "1", "NY_INTERM_ROW_CHUNKS": "16"}, space, kinds)
    check("repair drops a child the owning phase suppresses",
          not is_inert(repaired, space) and "NY_INTERM_ROW_CHUNKS" not in repaired, str(repaired))
    repaired, added = repair(
        {"NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN": "1"}, space, kinds,
        baseline_values={"NY_ROOT_PHASE_RESIDENT_CROWN": "1"})
    check("repair overrides an owning preset parent to make the child reachable",
          repaired.get("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN") == "1"
          and repaired.get("NY_ROOT_PHASE_RESIDENT_CROWN") == "0"
          and "NY_ROOT_PHASE_RESIDENT_CROWN" in added
          and not is_inert(repaired, space), str(repaired))
    check("contextual upfront absence leaves ordinary budget controls measurable",
          expand({"NY_UPFRONT_ATTACK_CAP": "8"}, space)
          == [("NY_UPFRONT_ATTACK_CAP", "8")])
    try:
        expand({"NY_UPFRONT_ATTACK": "0", "NY_UPFRONT_ATTACK_CAP": "8"}, space)
        upfront_kill_refused = False
    except Inert as inert:
        upfront_kill_refused = inert.axis == "NY_UPFRONT_ATTACK_CAP"
    check("explicit upfront kill refuses an inert cap treatment", upfront_kill_refused)
    repaired, added = repair(
        {"NY_UPFRONT_ATTACK_CAP": "8"}, space, kinds,
        baseline_values={"NY_UPFRONT_ATTACK": "0"})
    check("repair overrides a typed upfront kill to make a cap treatment executable",
          repaired.get("NY_UPFRONT_ATTACK") == "1"
          and repaired.get("NY_UPFRONT_ATTACK_CAP") == "8"
          and added == {"NY_UPFRONT_ATTACK"}
          and not is_inert(repaired, space), str(repaired))
    repaired, _added = repair(
        {"NY_UPFRONT_ATTACK_AUTO_CAP": "4"}, space, kinds,
        baseline_values={"NY_UPFRONT_ATTACK": "1"})
    check("automatic cap is dropped when a typed force removes contextual admission",
          "NY_UPFRONT_ATTACK_AUTO_CAP" not in repaired, str(repaired))
    zero_cap_children = {
        "NY_UPFRONT_ATTACK_FRAC": "0.5",
        "NY_UPFRONT_ATTACK_AUTO_CAP": "1",
    }
    for child, value in zero_cap_children.items():
        arm = {"NY_UPFRONT_ATTACK_CAP": "0", child: value}
        try:
            expand(arm, space)
            zero_cap_refused = False
        except Inert as inert:
            zero_cap_refused = inert.axis == child
        repaired, _added = repair(arm, space, kinds)
        check("zero upfront cap refuses and repairs inert %s" % child,
              zero_cap_refused
              and repaired == {"NY_UPFRONT_ATTACK_CAP": "0"},
              str(repaired))

    # --- T0: every emitted value must parse, and must not equal the default
    check("Bool rejects 'true' (the parser arms on exact '1')",
          not value_admissible({"kind": "Bool"}, "true")
          and value_admissible({"kind": "Bool"}, "1"))
    check("BoolOverride accepts only exact override tokens",
          value_admissible({"kind": "BoolOverride"}, "0")
          and value_admissible({"kind": "BoolOverride"}, "1")
          and not value_admissible({"kind": "BoolOverride"}, "true")
          and not value_admissible({"kind": "BoolOverride"}, " 1"))
    check("BoolKillOverride exposes only the exact rollback token",
          value_admissible({"kind": "BoolKillOverride"}, "0")
          and not value_admissible({"kind": "BoolKillOverride"}, "1")
          and not value_admissible({"kind": "BoolKillOverride"}, "false")
          and not value_admissible({"kind": "BoolKillOverride"}, " 0"))
    truthy_trimmed = {"kind": "TruthyTrimmed"}
    check("TruthyTrimmed preserves the complete legacy UTF-8 grammar",
          all(value_admissible(truthy_trimmed, token)
              for token in ("1", "true", "on", " 1 ", "0", "false", "", "bad"))
          and tokens_semantically_equal(truthy_trimmed, "1", " true ")
          and tokens_semantically_equal(truthy_trimmed, "1", "\u20031\u2003")
          and tokens_semantically_equal(truthy_trimmed, "0", "anything-else")
          and not tokens_semantically_equal(truthy_trimmed, "on", "false")
          and not tokens_semantically_equal(truthy_trimmed, "1", "\x1c1\x1c"))
    check("F64ClosedTrimmed admits BOTH endpoints (0.0 is the disarmed control)",
          value_admissible({"kind": "F64ClosedTrimmed", "min": 0.0, "max": 1.0}, "0.0")
          and value_admissible({"kind": "F64ClosedTrimmed", "min": 0.0, "max": 1.0}, "1.0")
          and value_admissible({"kind": "F64ClosedTrimmed", "min": 0.0, "max": 1.0}, "0.35")
          and not value_admissible({"kind": "F64ClosedTrimmed", "min": 0.0, "max": 1.0}, "1.5")
          and not value_admissible({"kind": "F64ClosedTrimmed", "min": 0.0, "max": 1.0}, "nope"))
    source_kinds = parse_lever_kinds()
    upfront_master_kind = source_kinds.get("NY_UPFRONT_ATTACK")
    postbab_attack_kind = source_kinds.get("NY_POSTBAB_ATTACK")
    upfront_frac_kind = source_kinds.get("NY_UPFRONT_ATTACK_FRAC")
    exact_width_kind = source_kinds.get("NY_PGD_VJP_K")
    exact_legacy_kind = source_kinds.get("NY_PGD_LEGACY_LANES")
    exact_eta_kind = source_kinds.get("NY_PGD_ODI_ETA")
    exact_hinge_kind = source_kinds.get("NY_PGD_HINGE_TAU")
    margin_k_head_kind = source_kinds.get("NY_MARGIN_ROW_K_HEAD")
    margin_k_trunk_kind = source_kinds.get("NY_MARGIN_ROW_K_TRUNK")
    check("source schema preserves the contextual BoolOverride master",
          upfront_master_kind is not None
          and upfront_master_kind.get("kind") == "BoolOverride"
          and upfront_master_kind.get("default") is None,
          repr(upfront_master_kind))
    check("source schema preserves the post-BaB one-way kill override",
          postbab_attack_kind is not None
          and postbab_attack_kind.get("kind") == "BoolKillOverride"
          and postbab_attack_kind.get("default") == "1",
          repr(postbab_attack_kind))
    check("source schema parses the strict-positive upfront fraction bound",
          upfront_frac_kind is not None
          and upfront_frac_kind.get("min") == float.fromhex("0x0.0000000000001p-1022")
          and upfront_frac_kind.get("max") == 1.0
          and value_admissible(upfront_frac_kind, "5e-324")
          and value_admissible(upfront_frac_kind, "1")
          and not value_admissible(upfront_frac_kind, "0")
          and not value_admissible(upfront_frac_kind, "-1"),
          repr(upfront_frac_kind))
    check("source schema preserves the exact usize/f32 parser variants",
          exact_width_kind is not None
          and exact_width_kind.get("kind") == "Usize"
          and exact_width_kind.get("min") == 1
          and exact_legacy_kind is not None
          and exact_legacy_kind.get("kind") == "Usize"
          and exact_legacy_kind.get("min") == 0
          and exact_eta_kind is not None
          and exact_eta_kind.get("kind") == "F32Positive"
          and exact_hinge_kind is not None
          and exact_hinge_kind.get("kind") == "F32NonNegative",
          repr((exact_width_kind, exact_legacy_kind, exact_eta_kind, exact_hinge_kind)))
    check("source schema preserves the trimmed positive usize override pair",
          all(kind is not None
              and kind.get("kind") == "UsizeTrimmedOverride"
              and kind.get("min") == 1
              and kind.get("default") == "8"
              for kind in (margin_k_head_kind, margin_k_trunk_kind)),
          repr((margin_k_head_kind, margin_k_trunk_kind)))
    binary_exact_kinds = _declarations_from_payload({
        "NY_PGD_VJP_K": {
            "kind": "Usize", "default": None, "min": 1,
            "max": float((1 << (8 * struct.calcsize("P"))) - 1),
            "members": None,
        },
        "NY_PGD_ODI_ETA": {
            "kind": "F32Positive", "default": "1", "min": 1.401298464324817e-45,
            "max": 3.4028234663852886e38, "members": None,
        },
        "NY_MARGIN_ROW_K_HEAD": {
            "kind": "UsizeTrimmedOverride", "default": "8", "min": 1,
            "max": float((1 << (8 * struct.calcsize("P"))) - 1),
            "members": None,
        },
        "NY_VERIFIER_THREADS": {
            "kind": "UsizeTrimmedUnicodeOwned", "default": None,
            "min": None,
            "max": None,
            "members": None,
        },
        "NY_MARGIN_ROW_CONCURRENT_THREADS": {
            "kind": "UsizeOverride", "default": None, "min": 0,
            "max": float((1 << (8 * struct.calcsize("P"))) - 1),
            "members": None,
        },
    })
    check("binary schema keeps an integral Usize lower bound",
          binary_exact_kinds["NY_PGD_VJP_K"]["min"] == 1
          and isinstance(binary_exact_kinds["NY_PGD_VJP_K"]["min"], int)
          and value_admissible(binary_exact_kinds["NY_PGD_VJP_K"], "52")
          and value_admissible(binary_exact_kinds["NY_PGD_ODI_ETA"], "0.5")
          and binary_exact_kinds["NY_MARGIN_ROW_K_HEAD"]["min"] == 1
          and isinstance(binary_exact_kinds["NY_MARGIN_ROW_K_HEAD"]["min"], int)
          and value_admissible(
              binary_exact_kinds["NY_MARGIN_ROW_K_HEAD"], " 16 "),
          repr(binary_exact_kinds))
    check("binary schema preserves both scheduler usize ownership contracts",
          binary_exact_kinds["NY_VERIFIER_THREADS"]["kind"]
              == "UsizeTrimmedUnicodeOwned"
          and binary_exact_kinds["NY_MARGIN_ROW_CONCURRENT_THREADS"]["kind"]
              == "UsizeOverride"
          and binary_exact_kinds["NY_MARGIN_ROW_CONCURRENT_THREADS"]["min"] == 0
          and isinstance(
              binary_exact_kinds["NY_MARGIN_ROW_CONCURRENT_THREADS"]["min"], int)
          and value_admissible(binary_exact_kinds["NY_VERIFIER_THREADS"], " 0 ")
          and not value_admissible(
              binary_exact_kinds["NY_VERIFIER_THREADS"], "malformed")
          and value_admissible(
              binary_exact_kinds["NY_MARGIN_ROW_CONCURRENT_THREADS"], "0")
          and not value_admissible(
              binary_exact_kinds["NY_MARGIN_ROW_CONCURRENT_THREADS"], " 0 "),
          repr(binary_exact_kinds))
    per_node_floor_kind = source_kinds.get("NY_PER_NODE_FLOOR_SECS")
    check("source schema preserves f64::MAX declaration bounds",
          per_node_floor_kind is not None
          and per_node_floor_kind.get("max") == sys.float_info.max
          and value_admissible(per_node_floor_kind, "2"),
          repr(per_node_floor_kind))
    check("bounded source kinds fail closed when bounds are unreadable",
          not value_admissible({"kind": "F64ClosedTrimmed"}, "0.5")
          and not value_admissible({"kind": "F64Open"}, "0.5"))
    check("F64Open is OPEN at both ends",
          not value_admissible({"kind": "F64Open", "min": 0.0, "max": 0.9}, "0")
          and not value_admissible({"kind": "F64Open", "min": 0.0, "max": 0.9}, "25")
          and value_admissible({"kind": "F64Open", "min": 0.0, "max": 0.9}, "0.25"))
    check("numeric token admission exactly mirrors Rust FromStr",
          value_admissible({"kind": "U64"}, "+0")
          and value_admissible({"kind": "U64Trimmed"}, "  +42  ")
          and not value_admissible({"kind": "U64"}, "18446744073709551616")
          and not value_admissible({"kind": "U64"}, "\u0660")
          and not value_admissible(
              {"kind": "UsizeTrimmed"}, str(1 << (8 * struct.calcsize("P"))))
          and not value_admissible(
              {"kind": "F64Open", "min": 0.0, "max": 1.0}, " 0.5 ")
          and not value_admissible({"kind": "Secs"}, " 1 ")
          and value_admissible(
              {"kind": "F64ClosedTrimmed", "min": 0.0, "max": 1.0},
              " 0.5 "))
    pointer_max = (1 << (8 * struct.calcsize("P"))) - 1
    check("untrimmed bounded Usize mirrors the declared platform parser",
          value_admissible({"kind": "Usize", "min": 1}, "+1")
          and value_admissible({"kind": "Usize", "min": 1}, "0008")
          and value_admissible({"kind": "Usize", "min": 1}, str(pointer_max))
          and not value_admissible({"kind": "Usize", "min": 1}, "0")
          and not value_admissible({"kind": "Usize", "min": 1}, " 8 ")
          and not value_admissible({"kind": "Usize", "min": 1}, "-1")
          and not value_admissible({"kind": "Usize", "min": 1}, "\u0668")
          and not value_admissible(
              {"kind": "Usize", "min": 1}, str(pointer_max + 1)))
    check("trimmed usize override mirrors positive branch-width admission",
          value_admissible(
              {"kind": "UsizeTrimmedOverride", "min": 1}, "  +16\n")
          and value_admissible(
              {"kind": "UsizeTrimmedOverride", "min": 1}, str(pointer_max))
          and not value_admissible(
              {"kind": "UsizeTrimmedOverride", "min": 1}, "0")
          and not value_admissible(
              {"kind": "UsizeTrimmedOverride", "min": 1}, "malformed")
          and not value_admissible(
              {"kind": "UsizeTrimmedOverride", "min": 1}, "\x1c16\x1c")
          and not value_admissible(
              {"kind": "UsizeTrimmedOverride", "min": 1}, str(pointer_max + 1)))
    check("scheduler usize kinds preserve trimmed-owned and untrimmed-fallthrough grammars",
          value_admissible({"kind": "UsizeTrimmedUnicodeOwned"}, "  +0\n")
          and value_admissible(
              {"kind": "UsizeTrimmedUnicodeOwned"}, str(pointer_max))
          and not value_admissible(
              {"kind": "UsizeTrimmedUnicodeOwned"}, "malformed")
          and not value_admissible(
              {"kind": "UsizeTrimmedUnicodeOwned"}, str(pointer_max + 1))
          and value_admissible({"kind": "UsizeOverride", "min": 0}, "+0")
          and value_admissible(
              {"kind": "UsizeOverride", "min": 0}, str(pointer_max))
          and not value_admissible(
              {"kind": "UsizeOverride", "min": 0}, " 0 ")
          and not value_admissible(
              {"kind": "UsizeOverride", "min": 0}, "malformed")
          and not value_admissible(
              {"kind": "UsizeOverride", "min": 0}, str(pointer_max + 1)))
    f32_positive = {"kind": "F32Positive"}
    f32_nonnegative = {"kind": "F32NonNegative"}
    check("f32 kinds reject whitespace/overflow/underflow with Rust-like signs",
          value_admissible(f32_positive, "+1e-3")
          and value_admissible(f32_positive, "1e+3")
          and value_admissible(f32_positive, "1e-45")
          and not value_admissible(f32_positive, " 0.5 ")
          and not value_admissible(f32_positive, "3.5e38")
          and not value_admissible(f32_positive, "1e-46")
          and not value_admissible(f32_positive, "-0")
          and value_admissible(f32_nonnegative, "-0")
          and value_admissible(f32_nonnegative, "+0e-4")
          and not value_admissible(f32_nonnegative, "-1e-45"))
    check("every exact hinge/ODI float domain token survives f32 parsing",
          all(value_admissible(f32_nonnegative, token)
              for token in ("0", "0.0001", "0.001", "0.01"))
          and all(value_admissible(f32_positive, token)
                  for token in ("0.25", "0.5", "1", "1.5", "2")))
    check("numeric default equality is typed, while enum equality is exact",
          tokens_semantically_equal(
              {"kind": "F64ClosedTrimmed", "min": 0.0, "max": 1.0},
              "0.40", "0.4")
          and tokens_semantically_equal({"kind": "UsizeTrimmed"}, "064", "64")
          and tokens_semantically_equal(
              {"kind": "UsizeTrimmedOverride", "min": 1}, " 016 ", "16")
          and tokens_semantically_equal(
              {"kind": "UsizeTrimmedUnicodeOwned"}, " 016 ", "16")
          and tokens_semantically_equal(
              {"kind": "UsizeOverride", "min": 0}, "+052", "52")
          and tokens_semantically_equal({"kind": "Usize", "min": 1}, "+052", "52")
          and tokens_semantically_equal(
              f32_positive, "0.1", "0.10000000149011612")
          and tokens_semantically_equal(f32_nonnegative, "-0", "0")
          and tokens_semantically_equal({"kind": "Presence"}, "0", "1")
          and not tokens_semantically_equal({"kind": "Enum"}, "0.40", "0.4"))
    numeric_axis = Axis("NY_NUMERIC", "enum", ["0.40", "0.50"],
                        "VerdictAffecting", "env-only")
    numeric_kept, numeric_dropped = admissible_values(
        numeric_axis,
        {"NY_NUMERIC": {"kind": "F64ClosedTrimmed", "default": "0.4",
                        "min": 0.0, "max": 1.0}},
        False,
    )
    check("T0 drops a differently formatted numeric default",
          numeric_kept == ["0.50"] and any("NY_NUMERIC=0.40" in note
                                            for note in numeric_dropped),
          "%r / %r" % (numeric_kept, numeric_dropped))
    seq_axis = space.by_name["NY_SEQ_WEIGHTED_BUDGET"]
    seq_kept, seq_dropped = admissible_values(
        seq_axis, kinds, False, preset_doc={}, preset_known=True)
    check("T0 models the opt-out sequential budget's shipped-on default",
          seq_kept == ["0"] and any("NY_SEQ_WEIGHTED_BUDGET=1" in note
                                     for note in seq_dropped),
          "%r / %r" % (seq_kept, seq_dropped))

    margin_policy_expected = {
        "NY_MARGIN_ROW_BETA_PERCOL": ["1"],
        "NY_MARGIN_ROW_BETA_COLS": ["1", "4", "16", "32"],
        "NY_MARGIN_ROW_COL_RETIRE": ["1"],
        "NY_MARGIN_ROW_PARALLEL_PREBUILD": ["1"],
    }
    margin_policy_kept = {
        name: admissible_values(space.by_name[name], kinds, False)[0]
        for name in margin_policy_expected
    }
    joint_margin_policy_request = {
        "NY_MARGIN_ROW_BETA_PERCOL": "1",
        "NY_MARGIN_ROW_BETA_COLS": "4",
        "NY_MARGIN_ROW_COL_RETIRE": "1",
        "NY_MARGIN_ROW_PARALLEL_PREBUILD": "1",
    }
    joint_margin_policy_repair = _repair_preserving(
        joint_margin_policy_request,
        space,
        kinds,
        return_contrast=True,
    )
    check("margin-row policy T0 and joint repair preserve exact causal coordinates",
          margin_policy_kept == margin_policy_expected
          and joint_margin_policy_repair is not None
          and joint_margin_policy_repair[1]["requested_assignment"]
              == [[name, token]
                  for name, token in sorted(joint_margin_policy_request.items())]
          and joint_margin_policy_repair[1]["prerequisites_added"]
              == [["NY_MARGIN_ROW_BETA", "1"]]
          and joint_margin_policy_repair[1]["comparison_assignment"]
              == [["NY_MARGIN_ROW_BETA", "1"]],
          repr((margin_policy_kept, joint_margin_policy_repair)))

    # DefaultSpec::Unset is not a runtime value. These shipped-category presets
    # omit the keys below, whose reader defaults are now explicit space data.
    closure_axis = space.by_name["NY_MARGIN_ROW_CLOSURE_ADAPTIVE"]
    closure_kept, closure_dropped = admissible_values(
        closure_axis, kinds, False, preset_doc={}, preset_known=True,
        preset_expected=True,
    )
    check("safenlp absent closure policy drops the effective =0 no-op",
          closure_kept == ["1"]
          and any("NY_MARGIN_ROW_CLOSURE_ADAPTIVE=0" in note
                  for note in closure_dropped),
          "%r / %r" % (closure_kept, closure_dropped))
    reserve_axis = space.by_name["NY_BAB_RESERVE_FRAC"]
    reserve_kept, reserve_dropped = admissible_values(
        reserve_axis, kinds, False, preset_doc={}, preset_known=True,
        preset_expected=True,
    )
    check("cifar absent reserve policy drops the effective =0.0 no-op",
          "0.0" not in reserve_kept and "0.10" in reserve_kept
          and any("NY_BAB_RESERVE_FRAC=0.0" in note for note in reserve_dropped),
          "%r / %r" % (reserve_kept, reserve_dropped))
    root_secs_axis = space.by_name["NY_ROOT_COMP_GPU_INTERM_SECS"]
    root_secs_kept, root_secs_dropped = admissible_values(
        root_secs_axis, kinds, False, preset_doc={}, preset_known=True,
        preset_expected=True,
    )
    check("absent root comprehensive seconds drops the effective =20 no-op",
          root_secs_kept == ["10", "40"]
          and any("NY_ROOT_COMP_GPU_INTERM_SECS=20" in note
                  for note in root_secs_dropped),
          "%r / %r" % (root_secs_kept, root_secs_dropped))

    contextual_axis = space.by_name["NY_SWEEP_CLASS_MIB"]
    contextual_kept, contextual_notes = admissible_values(
        contextual_axis, kinds, False, preset_doc={}, preset_known=True,
        preset_expected=True,
    )
    check("an uninstrumented contextual policy is excluded fail-closed",
          contextual_kept == []
          and any("no T1 engagement contract" in note for note in contextual_notes),
          "%r / %r" % (contextual_kept, contextual_notes))
    root_joint_dim_axis = space.by_name["NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM"]
    root_joint_dim_kept, root_joint_dim_notes = admissible_values(
        root_joint_dim_axis, kinds, False, preset_doc={}, preset_known=True,
        preset_expected=True,
    )
    check("root-joint max-dim is excluded until selector binding is authenticated",
          root_joint_dim_kept == []
          and any("no T1 engagement contract" in note
                  for note in root_joint_dim_notes),
          "%r / %r" % (root_joint_dim_kept, root_joint_dim_notes))
    upfront_contextual_axis = space.by_name["NY_UPFRONT_ATTACK"]
    upfront_contextual_kept, upfront_contextual_notes = admissible_values(
        upfront_contextual_axis, kinds, False, preset_doc={}, preset_known=True,
        preset_expected=True,
    )
    check("only instrumented contextual upfront overrides remain searchable",
          upfront_contextual_kept == ["1"]
          and any("versioned T1 engagement contract" in note
                  for note in upfront_contextual_notes)
          and any("no sound contextual counterfactual" in note
                  for note in upfront_contextual_notes),
          "%r / %r" % (upfront_contextual_kept, upfront_contextual_notes))
    frontier_contextual_axis = space.by_name["NY_MARGIN_ROW_PARALLEL"]
    frontier_contextual_kept, frontier_contextual_notes = admissible_values(
        frontier_contextual_axis, kinds, False, preset_doc={}, preset_known=True,
        preset_expected=True,
    )
    check("frontier execution evidence keeps every contextual width treatment",
          frontier_contextual_kept == frontier_contextual_axis.values
          and any("versioned T1 engagement contract" in note
                  for note in frontier_contextual_notes),
          "%r / %r" % (frontier_contextual_kept, frontier_contextual_notes))
    for exact_contextual_name in ("NY_PGD_VJP_K", "NY_PGD_VJP_STEPS"):
        exact_contextual_axis = space.by_name[exact_contextual_name]
        exact_contextual_kept, exact_contextual_notes = admissible_values(
            exact_contextual_axis, kinds, False, preset_doc={}, preset_known=True,
            preset_expected=True,
        )
        check("exact execution evidence keeps all contextual %s treatments"
              % exact_contextual_name,
              exact_contextual_kept == exact_contextual_axis.values
              and any("versioned T1 engagement contract" in note
                      for note in exact_contextual_notes),
              "%r / %r" % (exact_contextual_kept, exact_contextual_notes))

    lru_axis = space.by_name["NY_MARGIN_ROW_LRU"]
    common_lru, common_lru_notes = admissible_values(
        lru_axis,
        kinds,
        False,
        preset_expected=True,
        preset_docs=[
            {"unrelated": {"category": "cifar100"}},
            {"unrelated": {"category": "tinyimagenet"}},
        ],
    )
    split_lru, split_lru_notes = admissible_values(
        lru_axis,
        kinds,
        False,
        preset_expected=True,
        preset_docs=[
            {"margin_row": {"lru": 128}},
            {"unrelated": {"category": "tinyimagenet"}},
        ],
    )
    check("different category documents still drop a per-axis common LRU baseline",
          common_lru == ["16", "32", "128"]
          and any("common effective value" in note for note in common_lru_notes)
          and split_lru == lru_axis.values
          and any("category-specific effective values" in note
                  for note in split_lru_notes),
          "%r / %r / %r / %r"
          % (common_lru, common_lru_notes, split_lru, split_lru_notes))
    mixed_parallel_docs = [
        {"margin_row": {"parallel": 0}},
        {"margin_row": {"parallel": 16}},
    ]
    mixed_parallel_repair = effective_repair_baselines(
        space, kinds, mixed_parallel_docs)
    mixed_resident_docs = [
        {"bab": {"root_phase_resident_crown": False}},
        {"bab": {"root_phase_resident_crown": True}},
    ]
    mixed_resident_repair = effective_repair_baselines(
        space, kinds, mixed_resident_docs)
    mixed_upfront_docs = [
        {},
        {"attack": {"upfront_attack": False}},
    ]
    mixed_upfront_repair = effective_repair_baselines(
        space, kinds, mixed_upfront_docs)
    unresolved_repair = effective_repair_baselines(space, kinds, [None])
    check("repair baselines preserve category values and explicit false presence",
          [profile["values"].get("NY_MARGIN_ROW_PARALLEL")
           for profile in mixed_parallel_repair] == ["0", "16"]
          and [profile["values"].get("NY_ROOT_PHASE_RESIDENT_CROWN")
               for profile in mixed_resident_repair] == ["0", "1"]
          and "NY_UPFRONT_ATTACK" not in mixed_upfront_repair[0]["present"]
          and "NY_UPFRONT_ATTACK" in mixed_upfront_repair[1]["present"]
          and unresolved_repair[0]["unknown"]
              == frozenset(axis.name for axis in space.axes),
          repr((mixed_parallel_repair, mixed_resident_repair,
                mixed_upfront_repair, unresolved_repair)))
    distinct_axis = space.by_name["NY_ALPHA_ZERO_YIELD_FRAC"]
    distinct_kept, _distinct_notes = admissible_values(
        distinct_axis, kinds, False, preset_doc={}, preset_known=True,
        preset_expected=True,
    )
    check("distinct absence keeps every explicit treatment token",
          distinct_kept == distinct_axis.values, repr(distinct_kept))

    explicit_doc = {
        "bab": {
            "root_bab_reserve_frac": 0.15,
            "root_comprehensive_gpu_interm_secs": 40,
            "sweep_class_mib": 1024,
        },
        "margin_row": {"closure_adaptive": True},
    }
    explicit_closure, _ = admissible_values(
        closure_axis, kinds, False, preset_doc=explicit_doc, preset_known=True,
        preset_expected=True,
    )
    explicit_reserve, _ = admissible_values(
        reserve_axis, kinds, False, preset_doc=explicit_doc, preset_known=True,
        preset_expected=True,
    )
    explicit_root_secs, _ = admissible_values(
        root_secs_axis, kinds, False, preset_doc=explicit_doc, preset_known=True,
        preset_expected=True,
    )
    explicit_contextual, _ = admissible_values(
        contextual_axis, kinds, False, preset_doc=explicit_doc, preset_known=True,
        preset_expected=True,
    )
    check("explicit preset values override every absence contract",
          explicit_closure == ["0"]
          and "0.0" in explicit_reserve and "0.15" not in explicit_reserve
          and "20" in explicit_root_secs and "40" not in explicit_root_secs
          and explicit_contextual == ["64", "256", "2048", "4096", "8192"],
          "%r / %r / %r / %r"
          % (explicit_closure, explicit_reserve, explicit_root_secs,
             explicit_contextual))

    absence_values = effective_baseline_values(
        space, kinds, preset_doc={}, preset_known=True, preset_expected=True)
    explicit_values = effective_baseline_values(
        space, kinds, preset_doc=explicit_doc, preset_known=True,
        preset_expected=True)
    check("effective baseline map uses absence metadata and explicit overrides",
          absence_values.get(closure_axis.name) == "0"
          and absence_values.get(reserve_axis.name) == "0.0"
          and absence_values.get(root_secs_axis.name) == "20"
          and contextual_axis.name not in absence_values
          and distinct_axis.name not in absence_values
          and explicit_values.get(closure_axis.name) == "1"
          and explicit_values.get(reserve_axis.name) == "0.15"
          and explicit_values.get(root_secs_axis.name) == "40"
          and explicit_values.get(contextual_axis.name) == "1024")

    layered_axis = space.by_name["NY_MO_KFSB_REDUCE"]
    layered_kind = kinds[layered_axis.name]
    check("LayeredEnum T0 uses only the compiled environment members",
          layered_kind.get("members") == ["min", "max"]
          and value_admissible(layered_kind, "min")
          and value_admissible(layered_kind, "max")
          and not value_admissible(layered_kind, "mean"))
    kfsb_k_axis = space.by_name["NY_MO_KFSB_K"]
    kfsb_inherited_doc = {
        "bab": {"branching": {
            "candidates": 7,
            "reduceop": "max",
            "kfsb_multi": True,
        }},
    }
    kfsb_inherited_baseline = effective_baseline_values(
        space,
        kinds,
        preset_doc=kfsb_inherited_doc,
        preset_known=True,
        preset_expected=True,
    )
    kfsb_k_inherited, kfsb_k_inherited_notes = admissible_values(
        kfsb_k_axis,
        kinds,
        False,
        preset_doc=kfsb_inherited_doc,
        preset_known=True,
        preset_expected=True,
    )
    kfsb_reduce_inherited, kfsb_reduce_inherited_notes = admissible_values(
        layered_axis,
        kinds,
        False,
        preset_doc=kfsb_inherited_doc,
        preset_known=True,
        preset_expected=True,
    )
    check("MO-kFSB T0 follows the runtime's dedicated-to-ordinary preset chain",
          kfsb_k_axis.preset_baseline_keys == [
              "bab.branching.kfsb_multi_candidates",
              "bab.branching.candidates",
          ]
          and layered_axis.preset_baseline_keys == [
              "bab.branching.kfsb_multi_reduceop",
              "bab.branching.reduceop",
          ]
          and kfsb_inherited_baseline.get(kfsb_k_axis.name) == "7"
          and kfsb_inherited_baseline.get(layered_axis.name) == "max"
          and kfsb_k_inherited == ["0", "2", "5", "8", "10"]
          and kfsb_reduce_inherited == ["min"]
          and any("runtime fallback" in note for note in kfsb_k_inherited_notes)
          and any("runtime fallback" in note for note in kfsb_reduce_inherited_notes),
          repr((kfsb_inherited_baseline, kfsb_k_inherited,
                kfsb_reduce_inherited)))
    kfsb_dedicated_doc = {
        "bab": {"branching": {
            "kfsb_multi_candidates": 5,
            "candidates": 7,
            "kfsb_multi_reduceop": "min",
            "reduceop": "max",
        }},
    }
    kfsb_k_dedicated, _ = admissible_values(
        kfsb_k_axis,
        kinds,
        False,
        preset_doc=kfsb_dedicated_doc,
        preset_known=True,
        preset_expected=True,
    )
    kfsb_reduce_dedicated, _ = admissible_values(
        layered_axis,
        kinds,
        False,
        preset_doc=kfsb_dedicated_doc,
        preset_known=True,
        preset_expected=True,
    )
    check("a dedicated MO-kFSB key takes precedence over its ordinary fallback",
          kfsb_k_dedicated == ["0", "2", "7", "8", "10"]
          and kfsb_reduce_dedicated == ["max"],
          repr((kfsb_k_dedicated, kfsb_reduce_dedicated)))
    kfsb_null_dedicated_doc = {
        "bab": {"branching": {
            "kfsb_multi_candidates": None,
            "candidates": 7,
            "kfsb_multi_reduceop": None,
            "reduceop": "max",
        }},
    }
    kfsb_null_baseline = effective_baseline_values(
        space,
        kinds,
        preset_doc=kfsb_null_dedicated_doc,
        preset_known=True,
        preset_expected=True,
    )
    null_closure_state = _effective_axis_baseline(
        closure_axis,
        kinds[closure_axis.name],
        preset_doc={"margin_row": {"closure_adaptive": None}},
        preset_known=True,
        preset_expected=True,
    )
    check("typed YAML null is absence and continues runtime preset fallback",
          kfsb_null_baseline.get(kfsb_k_axis.name) == "7"
          and kfsb_null_baseline.get(layered_axis.name) == "max"
          and null_closure_state[:2] == ("known", "0"),
          repr((kfsb_null_baseline, null_closure_state)))
    kfsb_mixed_k, kfsb_mixed_k_notes = admissible_values(
        kfsb_k_axis,
        kinds,
        False,
        preset_expected=True,
        preset_docs=[
            kfsb_inherited_doc,
            {"bab": {"branching": {"candidates": 5}}},
        ],
    )
    check("different inherited category baselines keep every kFSB treatment",
          kfsb_mixed_k == kfsb_k_axis.values
          and any("category-specific effective values" in note
                  for note in kfsb_mixed_k_notes),
          repr((kfsb_mixed_k, kfsb_mixed_k_notes)))
    v24_cifar_doc = load_preset_doc_portable(
        REPO_ROOT / "configs" / "vnncomp24" / "cifar100.yaml"
    )
    v24_cifar_k = admissible_values(
        kfsb_k_axis,
        kinds,
        False,
        preset_doc=v24_cifar_doc,
        preset_known=v24_cifar_doc is not None,
        preset_expected=True,
    )[0]
    v24_cifar_reduce = admissible_values(
        layered_axis,
        kinds,
        False,
        preset_doc=v24_cifar_doc,
        preset_known=v24_cifar_doc is not None,
        preset_expected=True,
    )[0]
    v24_baseline_values = effective_baseline_values(
        space,
        kinds,
        preset_doc=v24_cifar_doc,
        preset_known=v24_cifar_doc is not None,
        preset_expected=True,
    )
    v24_kfsb_arms, _ = generate_arms(
        space,
        [(kfsb_k_axis, v24_cifar_k), (layered_axis, v24_cifar_reduce)],
        "cover",
        0,
        0,
        1,
        kinds=kinds,
        baseline_values=v24_baseline_values,
    )
    v24_kfsb_assignments = {tuple(arm) for arm in v24_kfsb_arms}
    check("the shipped VNN-COMP 2024 CIFAR plan keeps real 8/min treatments",
          v24_cifar_doc is not None
          and v24_cifar_k == ["0", "2", "5", "8", "10"]
          and v24_cifar_reduce == ["min"]
          and (("NY_MO_KFSB", "1"), ("NY_MO_KFSB_K", "8"))
              in v24_kfsb_assignments
          and (("NY_MO_KFSB", "1"), ("NY_MO_KFSB_REDUCE", "min"))
              in v24_kfsb_assignments
          and (("NY_MO_KFSB", "1"), ("NY_MO_KFSB_K", "7"))
              not in v24_kfsb_assignments
          and (("NY_MO_KFSB", "1"), ("NY_MO_KFSB_REDUCE", "max"))
              not in v24_kfsb_assignments,
          repr((v24_cifar_k, v24_cifar_reduce, sorted(v24_kfsb_assignments))))

    cert_axis = space.by_name["NY_MO_KFSB_CERT_REUSE"]
    cert_absent_values = admissible_values(
        cert_axis,
        kinds,
        False,
        preset_doc={},
        preset_known=True,
        preset_expected=True,
    )[0]
    cert_absent_baseline = effective_baseline_values(
        space, kinds, preset_doc={}, preset_known=True, preset_expected=True)
    v25_cifar_doc = load_preset_doc_portable(
        REPO_ROOT / "configs" / "vnncomp25" / "cifar100_2024.yaml"
    )
    cert_cifar_values = admissible_values(
        cert_axis,
        kinds,
        False,
        preset_doc=v25_cifar_doc,
        preset_known=v25_cifar_doc is not None,
        preset_expected=True,
    )[0]
    cert_cifar_baseline = effective_baseline_values(
        space,
        kinds,
        preset_doc=v25_cifar_doc,
        preset_known=v25_cifar_doc is not None,
        preset_expected=True,
    )
    cert_arms, _cert_inert, cert_design = generate_arms(
        space,
        [(cert_axis, cert_absent_values)],
        "cover",
        0,
        0,
        1,
        kinds=kinds,
        baseline_values=cert_absent_baseline,
        return_design=True,
    )
    cert_plan_scope = _classify_design_evidence(cert_design)
    cert_contrasts = [
        contrast
        for contrasts in cert_design["contrasts_by_assignment"].values()
        for contrast in contrasts
    ]
    cert_contrast = cert_contrasts[0] if len(cert_contrasts) == 1 else None
    check("certificate-reuse T0 distinguishes CIFAR's true preset from absence",
          v25_cifar_doc is not None
          and cert_absent_baseline.get(cert_axis.name) == "0"
          and cert_absent_values == ["1"]
          and cert_cifar_baseline.get(cert_axis.name) == "1"
          and cert_cifar_values == ["0"],
          repr((cert_absent_baseline.get(cert_axis.name), cert_absent_values,
                cert_cifar_baseline.get(cert_axis.name), cert_cifar_values)))
    check("certificate-reuse planning stays delivery-only without a T1 contract",
          cert_axis.name not in _AXIS_SPECIFIC_T1_CONTRACTS
          and cert_plan_scope == "delivery_only"
          and cert_contrast is not None
          and cert_contrast["ungraduated_axes"] == [cert_axis.name]
          and cert_contrast["requested_assignment"] == [[cert_axis.name, "1"]]
          and cert_contrast["prerequisites_added"] == [["NY_MO_KFSB", "1"]]
          and cert_arms == [
              [],
              [("NY_MO_KFSB", "1"), (cert_axis.name, "1")],
              [("NY_MO_KFSB", "1")],
          ],
          repr((cert_plan_scope, cert_contrast, cert_arms)))
    check("decls scrape found the live kinds",
          kinds.get("NY_MARGIN_ROW_GPU", {}).get("kind") == "Bool"
          and kinds.get("NY_INTERM_ROW_CHUNKS", {}).get("kind") == "UsizeTrimmed",
          "%d levers scraped" % len(kinds))
    unclassified_unset = [
        axis.name for axis in space.axes
        if kinds.get(axis.name, {}).get("default") is None
        and axis.absence_baseline.get("kind") == "declaration_default"
    ]
    check("source parser classifies every searchable Unset declaration",
          not unclassified_unset, repr(unclassified_unset))

    pool = []
    t0_notes = []
    for axis in space.axes:
        kept, dropped = admissible_values(axis, kinds, False)
        t0_notes.extend(dropped)
        if kept:
            pool.append((axis, kept))
    check("T0 drops at least one declared value", bool(t0_notes), "; ".join(t0_notes)[:200])
    filter_opts = build_parser().parse_args(["--delivery", "preset"])
    preset_axes = [axis for axis in space.axes
                   if delivery_bucket(axis) in filter_opts.delivery_filter]
    check("--delivery preset covers the complete searchable space",
          filter_opts.delivery_filter == ["preset"] and preset_axes
          and all(axis.deliver.startswith("preset:") for axis in preset_axes)
          and len(preset_axes) == len(space.axes))
    check("configured searches default to preset-only delivery",
          effective_delivery_filter(True, []) == ["preset"]
          and effective_delivery_filter(False, []) == []
          and effective_delivery_filter(True, ["env-only"]) == ["env-only"])
    safenlp_categories = ["SafeNLP_2024"]
    safenlp_applicable = {
        axis.name for axis in space.axes
        if axis_applies_to_categories(axis, safenlp_categories)
    }
    check("safenlp T0 excludes CGAN, ResNet, and BNN axes but keeps general axes",
          "NY_CGAN_POLISH" not in safenlp_applicable
          and "NY_BAB_RESNET_WIDE" not in safenlp_applicable
          and "NY_BNN_STE_PGD" not in safenlp_applicable
          and "NY_BNN_ATTACK_CAP" not in safenlp_applicable
          and "NY_BNN_ATTACK_FRAC" not in safenlp_applicable
          and "NY_SAFENLP_MIP_CYCLE" in safenlp_applicable
          and "NY_MO_ALPHA_HEURISTIC" in safenlp_applicable
          and "category-inert" in category_applicability_finding(
              space.by_name["NY_CGAN_POLISH"], safenlp_categories),
          repr(sorted(safenlp_applicable)))
    traffic_categories = ["traffic_signs_recognition_2023"]
    check("traffic T0 admits every governed BNN axis",
          all(axis_applies_to_categories(space.by_name[name], traffic_categories)
              for name in (
                  "NY_BNN_ATTACK_CAP",
                  "NY_BNN_ATTACK_FRAC",
                  *_BNN_FALSIFICATION_FAMILY,
              )))
    traffic_doc = load_preset_doc_portable(
        REPO_ROOT / "configs" / "vnncomp25"
        / "traffic_signs_recognition_2023.yaml"
    )
    traffic_bnn_treatments = {
        name: admissible_values(
            space.by_name[name],
            kinds,
            False,
            preset_doc=traffic_doc,
            preset_known=traffic_doc is not None,
            preset_expected=True,
        )[0]
        for name in _BNN_FALSIFICATION_FAMILY
    }
    check(
        "traffic T0 compares every governed BNN falsification axis with its "
        "shipped preset baseline",
        traffic_bnn_treatments == {
            "NY_BNN_SIGN_SPACE": ["0"],
            "NY_BNN_SIGN_SPACE_MINIMAL_MOVE": ["1"],
            "NY_BNN_SIGN_SPACE_TRUST_REGION": ["box", "tight", "linf"],
            "NY_BNN_STE_PGD": ["0"],
        },
        repr(traffic_bnn_treatments),
    )
    postbab_reserve_axis = space.by_name["NY_POSTBAB_RESERVE_SECS"]
    check("post-BaB reserve T0 excludes bypassed and preset-disabled routes",
          postbab_reserve_axis.category_prefixes == ["acasxu"]
          and axis_applies_to_categories(
              postbab_reserve_axis, ["acasxu_2023"])
          and not axis_applies_to_categories(
              postbab_reserve_axis, ["yolo_2023"])
          and runtime_applicability_finding(
              postbab_reserve_axis, ["isomorphic_acasxu_2026"]
          ) is not None
          and runtime_applicability_finding(
              postbab_reserve_axis,
              ["yolo_2023"],
              preset_docs=[{
                  "bab": {"phase_budget": {"vnncomp_post_bab_attack": False}},
              }],
              preset_expected=True,
          ) is not None
          and runtime_applicability_finding(
              postbab_reserve_axis,
              ["acasxu_2023"],
              preset_docs=[{"bab": {"phase_budget": {}}}],
              preset_expected=True,
          ) is None
          and "0" not in postbab_reserve_axis.values,
          repr((postbab_reserve_axis.category_prefixes,
                postbab_reserve_axis.values)))
    concurrent_axis = space.by_name["NY_MARGIN_ROW_CONCURRENT"]
    check(
        "concurrency T0 requires an exact typed reserve-free preset context",
        runtime_applicability_finding(
            concurrent_axis,
            ["tinyimagenet_2024"],
            preset_docs=[{"margin_row": {"reserve_secs": 0}}],
            preset_expected=True,
        ) is None
        and runtime_applicability_finding(
            concurrent_axis,
            ["cifar100_2024"],
            preset_docs=[{"margin_row": {"adaptive_reserve": True}}],
            preset_expected=True,
        ) is not None
        and runtime_applicability_finding(
            concurrent_axis,
            ["tinyimagenet_2024"],
            preset_docs=[{"margin_row": {"reserve_secs": "0"}}],
            preset_expected=True,
        ) is not None
        and runtime_applicability_finding(
            concurrent_axis,
            ["control", "already_on"],
            preset_docs=[
                {"margin_row": {"reserve_secs": 0, "concurrent": False}},
                {"margin_row": {"reserve_secs": 0, "concurrent": True}},
            ],
            preset_expected=True,
        ) is not None,
    )
    resnet_axis = space.by_name["NY_BAB_RESNET_WIDE"]
    check("category scope must match every category sharing an arm",
          axis_applies_to_categories(
              resnet_axis, ["CIFAR100_2024", "TinyImageNet_2024"])
          and not axis_applies_to_categories(
              resnet_axis, ["cifar100_2024", "safenlp_2024"]))

    # A same-named category in a newer config directory must not shadow the
    # preset belonging to the requested benchmark corpus.
    with tempfile.TemporaryDirectory() as tmp:
        configs = Path(tmp)
        workspace = configs / "workspace"
        automatic_configs = workspace / "configs"
        automatic_configs.mkdir(parents=True)
        check("search mirrors the child's repo-local config auto-derivation",
              effective_configs_root(None, workspace) == automatic_configs
              and effective_configs_root(configs, workspace) == configs)
        v25 = configs / "vnncomp25"
        v26 = configs / "vnncomp26"
        v25.mkdir()
        v26.mkdir()
        p25 = v25 / "safenlp_2024.yaml"
        p26 = v26 / "safenlp_2024.yaml"
        p25.write_text("general:\n  year: 2025\n", encoding="utf-8")
        p26.write_text("general:\n  year: 2026\n", encoding="utf-8")
        nested = v25 / "a"
        nested.mkdir()
        (nested / "inside.yaml").write_text("inside\n", encoding="utf-8")
        (v25 / "a.x").write_text("sibling\n", encoding="utf-8")

        def digest_in_order(relative_names):
            digest = hashlib.sha256()
            for relative_name in relative_names:
                entry = v25 / relative_name
                encoded = relative_name.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
                body = entry.read_bytes()
                digest.update(len(body).to_bytes(8, "little"))
                digest.update(body)
            return digest.hexdigest()

        recursive_order = ["a/inside.yaml", "a.x", "safenlp_2024.yaml"]
        global_order = ["a.x", "a/inside.yaml", "safenlp_2024.yaml"]
        tree_digest = fingerprint_tree(v25)["sha256"]
        check("configuration fingerprint mirrors Rust's recursive entry order",
              tree_digest == digest_in_order(recursive_order)
              and tree_digest != digest_in_order(global_order))
        check("year-scoped preset resolution selects v25 despite duplicate v26",
              benchmark_configs_dir(configs, 2025) == v25.resolve()
              and resolve_preset_path(configs, "safenlp_2024", 2025) == p25.resolve()
              and resolve_preset_path(configs, "safenlp_2024", 2026) == p26.resolve())
        check("a differently scoped or missing year fails closed",
              benchmark_configs_dir(v26, 2025) is None
              and benchmark_configs_dir(configs, 2024) is None)

        preset_row = _clean_row(category="safenlp_2024")
        preset_row["flight"]["events"] = [
            {"method": "preset_base", "status": "ran", "reason": str(p25.resolve())},
            # The effective preset may be a plan-resolver temporary and is not
            # durable attribution evidence; T1 intentionally ignores this event.
            {"method": "preset", "status": "ran", "reason": "/tmp/effective.yaml"},
        ]
        preset_manifest = _clean_manifest(category="safenlp_2024")
        preset_manifest["configs"] = fingerprint_tree(v25)
        valid, failures, _warnings, _counts = validate_run(
            [], 2025, "safenlp_2024", [preset_row], preset_manifest,
            _summary_of([preset_row]), expected_configs=fingerprint_tree(v25),
            expected_preset_path=p25.resolve())
        check("T1 accepts the exact requested preset path and tree identity",
              valid, str(failures))

        wrong_row = json.loads(json.dumps(preset_row))
        wrong_row["flight"]["events"][0]["reason"] = str(p26.resolve())
        valid, failures, _warnings, _counts = validate_run(
            [], 2025, "safenlp_2024", [wrong_row], preset_manifest,
            _summary_of([wrong_row]), expected_configs=fingerprint_tree(v25),
            expected_preset_path=p25.resolve())
        check("T1 rejects a v26 child preset on a 2025 corpus",
              not valid and any(f["check"] == "preset_matches_request" for f in failures),
              str(failures))

        wrong_manifest = dict(preset_manifest, configs=fingerprint_tree(v26))
        valid, failures, _warnings, _counts = validate_run(
            [], 2025, "safenlp_2024", [preset_row], wrong_manifest,
            _summary_of([preset_row]), expected_configs=fingerprint_tree(v25),
            expected_preset_path=p25.resolve())
        check("T1 rejects a manifest sealed to the wrong preset tree",
              not valid and any(f["check"] == "manifest_matches_preset" for f in failures),
              str(failures))

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
            or not tokens_semantically_equal(kinds.get(name), value, kinds[name]["default"])
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
    chunks_axis = space.by_name["NY_INTERM_ROW_CHUNKS"]
    repaired_grid, repaired_grid_inert = generate_arms(
        space, [(chunks_axis, ["4"])], "grid", 0, 0, 1, kinds=kinds,
        baseline_values={"NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN": "1"})
    check("grid repairs a dependent child before deciding that it is inert",
          repaired_grid_inert == 0
          and any(dict(arm).get("NY_INTERM_ROW_CHUNKS") == "4"
                  and dict(arm).get("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN") == "1"
                  for arm in repaired_grid), repr(repaired_grid))
    parallel_prebuild_axis = space.by_name["NY_MARGIN_ROW_PARALLEL_PREBUILD"]
    mixed_parallel_values = effective_baseline_values(
        space, kinds, preset_expected=True, preset_docs=mixed_parallel_docs)
    mixed_parallel_arms, mixed_parallel_inert, mixed_parallel_design = generate_arms(
        space,
        [(parallel_prebuild_axis, ["1"])],
        "cover",
        0,
        0,
        1,
        max_arms=3,
        kinds=kinds,
        baseline_values=mixed_parallel_values,
        return_design=True,
        repair_baselines=mixed_parallel_repair,
    )
    mixed_parallel_key = (
        ("NY_MARGIN_ROW_PARALLEL", "2"),
        ("NY_MARGIN_ROW_PARALLEL_PREBUILD", "1"),
    )
    mixed_parallel_contrast = mixed_parallel_design[
        "contrasts_by_assignment"][mixed_parallel_key][0]
    check("mixed serial/parallel presets repair prebuild with an exact comparator",
          mixed_parallel_arms == [
              [],
              list(mixed_parallel_key),
              [("NY_MARGIN_ROW_PARALLEL", "2")],
          ]
          and mixed_parallel_inert == 0
          and mixed_parallel_contrast["requested_assignment"]
              == [["NY_MARGIN_ROW_PARALLEL_PREBUILD", "1"]]
          and mixed_parallel_contrast["prerequisites_added"]
              == [["NY_MARGIN_ROW_PARALLEL", "2"]]
          and mixed_parallel_contrast["comparison_assignment"]
              == [["NY_MARGIN_ROW_PARALLEL", "2"]],
          repr((mixed_parallel_arms, mixed_parallel_contrast)))

    comprehensive_axis = space.by_name[
        "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"]
    mixed_resident_values = effective_baseline_values(
        space, kinds, preset_expected=True, preset_docs=mixed_resident_docs)
    mixed_resident_arms, mixed_resident_inert, mixed_resident_design = generate_arms(
        space,
        [(comprehensive_axis, ["1"])],
        "cover",
        0,
        0,
        1,
        max_arms=3,
        kinds=kinds,
        baseline_values=mixed_resident_values,
        return_design=True,
        repair_baselines=mixed_resident_repair,
    )
    mixed_resident_key = (
        ("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN", "1"),
        ("NY_ROOT_PHASE_RESIDENT_CROWN", "0"),
    )
    mixed_resident_contrast = mixed_resident_design[
        "contrasts_by_assignment"][mixed_resident_key][0]
    check("mixed resident ownership repairs comprehensive with an exact comparator",
          mixed_resident_arms == [
              [],
              list(mixed_resident_key),
              [("NY_ROOT_PHASE_RESIDENT_CROWN", "0")],
          ]
          and mixed_resident_inert == 0
          and mixed_resident_contrast["prerequisites_added"]
              == [["NY_ROOT_PHASE_RESIDENT_CROWN", "0"]]
          and mixed_resident_contrast["comparison_assignment"]
              == [["NY_ROOT_PHASE_RESIDENT_CROWN", "0"]],
          repr((mixed_resident_arms, mixed_resident_contrast)))

    auto_cap_axis = space.by_name["NY_UPFRONT_ATTACK_AUTO_CAP"]
    mixed_upfront_values = effective_baseline_values(
        space, kinds, preset_expected=True, preset_docs=mixed_upfront_docs)
    mixed_upfront_refused = False
    try:
        generate_arms(
            space,
            [(auto_cap_axis, ["6"])],
            "cover",
            0,
            0,
            1,
            max_arms=2,
            kinds=kinds,
            baseline_values=mixed_upfront_values,
            return_design=True,
            repair_baselines=mixed_upfront_repair,
        )
    except SystemExit as error:
        mixed_upfront_refused = (
            "cover cannot exercise 1 axis/value pair(s)" in str(error)
            and "NY_UPFRONT_ATTACK_AUTO_CAP=6" in str(error)
        )
    check("mixed absent/false upfront presets refuse an impossible auto-cap arm",
          mixed_upfront_refused)

    upfront_cap_repair = _repair_preserving(
        {"NY_UPFRONT_ATTACK_CAP": "12"},
        space,
        kinds,
        return_contrast=True,
        repair_baselines=mixed_upfront_repair,
    )
    parallel_nonzero_repair = effective_repair_baselines(
        space,
        kinds,
        [{"margin_row": {"parallel": 8}},
         {"margin_row": {"parallel": 16}}],
    )
    resident_off_repair = effective_repair_baselines(
        space,
        kinds,
        [{"bab": {"root_phase_resident_crown": False}},
         {"bab": {"root_phase_resident_crown": False}}],
    )
    upfront_absent_repair = effective_repair_baselines(
        space, kinds, [{}, {"attack": {"upfront_attack": None}}])
    unresolved_parallel = _repair_preserving(
        {"NY_MARGIN_ROW_PARALLEL_PREBUILD": "1"},
        space,
        kinds,
        repair_baselines=unresolved_repair,
    )
    unresolved_resident = _repair_preserving(
        {"NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN": "1"},
        space,
        kinds,
        repair_baselines=unresolved_repair,
    )
    check("cross-category repair leaves already-compatible controls unchanged",
          upfront_cap_repair is not None
          and dict(upfront_cap_repair[0]) == {
              "NY_UPFRONT_ATTACK": "1",
              "NY_UPFRONT_ATTACK_CAP": "12",
          }
          and upfront_cap_repair[1]["comparison_assignment"]
              == [["NY_UPFRONT_ATTACK", "1"]]
          and _repair_preserving(
              {"NY_MARGIN_ROW_PARALLEL_PREBUILD": "1"},
              space,
              kinds,
              repair_baselines=parallel_nonzero_repair,
          ) == [("NY_MARGIN_ROW_PARALLEL_PREBUILD", "1")]
          and _repair_preserving(
              {"NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN": "1"},
              space,
              kinds,
              repair_baselines=resident_off_repair,
          ) == [("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN", "1")]
          and _repair_preserving(
              {"NY_UPFRONT_ATTACK_AUTO_CAP": "6"},
              space,
              kinds,
              repair_baselines=upfront_absent_repair,
          ) == [("NY_UPFRONT_ATTACK_AUTO_CAP", "6")]
          and _repair_preserving(
              {"NY_UPFRONT_ATTACK_AUTO_CAP": "6"},
              space,
              kinds,
              repair_baselines=unresolved_repair,
          ) is None
          and dict(unresolved_parallel or []) == {
              "NY_MARGIN_ROW_PARALLEL": "2",
              "NY_MARGIN_ROW_PARALLEL_PREBUILD": "1",
          }
          and dict(unresolved_resident or []) == {
              "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN": "1",
              "NY_ROOT_PHASE_RESIDENT_CROWN": "0",
          },
          repr((upfront_cap_repair, parallel_nonzero_repair,
                resident_off_repair, upfront_absent_repair,
                unresolved_repair, unresolved_parallel,
                unresolved_resident)))
    root_master_axis = space.by_name["NY_ROOT_JOINT_INTERM_ALPHA"]
    root_deadline_axis = space.by_name[
        "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT"
    ]
    root_max_sel_axis = space.by_name["NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL"]
    deduped_arms, _deduped_inert, deduped_design = generate_arms(
        space,
        [(root_deadline_axis, ["1"]), (root_max_sel_axis, ["8"])],
        "pair-cover",
        0,
        0,
        2,
        max_arms=5,
        max_candidates=4,
        kinds=kinds,
        return_design=True,
    )
    joint_assignment = (
        ("NY_BN_GPU_EXTRACT", "1"),
        ("NY_CPU_COMPUTE_ENGINE", "0"),
        ("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT", "1"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP", "1"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL", "8"),
    )
    joint_contrasts = deduped_design["contrasts_by_assignment"].get(
        joint_assignment, [])
    for contrasts in deduped_design["contrasts_by_assignment"].values():
        for contrast in contrasts:
            contrast["evidence_scope"] = "scored"
            contrast["ungraduated_axes"] = []
    design_document = search_design_manifest(
        deduped_arms,
        deduped_design,
        mode="pair-cover",
        seed=0,
        max_axes=2,
        categories=["cifar100"],
        year=2025,
        repeats=1,
        space_source=space.source,
        evidence_scope="scored",
    )
    check("dedupe preserves every intent and schedules each exact comparator",
          len(joint_contrasts) == 3
          and {tuple(map(tuple, contrast["comparison_assignment"]))
               for contrast in joint_contrasts}
              == {
                  (
                      ("NY_BN_GPU_EXTRACT", "1"),
                      ("NY_CPU_COMPUTE_ENGINE", "0"),
                      ("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
                      ("NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP", "1"),
                  ),
                  (
                      ("NY_BN_GPU_EXTRACT", "1"),
                      ("NY_CPU_COMPUTE_ENGINE", "0"),
                      ("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
                      ("NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT", "1"),
                      ("NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP", "1"),
                  ),
                  (
                      ("NY_BN_GPU_EXTRACT", "1"),
                      ("NY_CPU_COMPUTE_ENGINE", "0"),
                      ("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
                      ("NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP", "1"),
                      ("NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL", "8"),
                  ),
              }
          and [] in deduped_arms
          and len(deduped_arms) == 5
          and not [arm for arm in deduped_arms if is_inert(dict(arm), space)]
          and search_design_manifest_issues(design_document) == [],
          repr((deduped_arms, joint_contrasts,
                search_design_manifest_issues(design_document))))
    master_arms, _master_inert, master_design = generate_arms(
        space,
        [(root_master_axis, ["1"])],
        "cover",
        0,
        0,
        1,
        max_arms=2,
        kinds=kinds,
        return_design=True,
    )
    check("--no-baseline planning can detect a required default comparator",
          master_arms == [[], [("NY_ROOT_JOINT_INTERM_ALPHA", "1")]]
          and len(empty_comparator_contrast_ids(master_design)) == 1)
    root_pure_axis = space.by_name["NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN"]
    pure_arms, pure_inert, pure_design = generate_arms(
        space,
        [(root_pure_axis, ["1"])],
        "cover",
        0,
        0,
        1,
        max_arms=3,
        kinds=kinds,
        return_design=True,
    )
    pure_treatment = next(
        arm for arm in pure_arms
        if dict(arm).get("NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN") == "1"
    )
    pure_contrast = pure_design["contrasts_by_assignment"][tuple(pure_treatment)][0]
    check("the pure-chain recipe pins frozen-stop false in both exact arms",
          pure_inert == 0
          and dict(pure_treatment).get(
              "NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP") == "0"
          and dict(map(tuple, pure_contrast["comparison_assignment"])).get(
              "NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP") == "0"
          and not [arm for arm in pure_arms if is_inert(dict(arm), space)],
          repr((pure_arms, pure_contrast)))
    check("an explicit root master+child intent is refused, not miscompared",
          _repair_preserving({
              "NY_ROOT_JOINT_INTERM_ALPHA": "1",
              "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL": "8",
          }, space, kinds, return_contrast=True) is None)
    tampered_design = json.loads(json.dumps(design_document))
    next(
        item for item in tampered_design["assignments"]
        if item["comparison_for"]
    )["comparison_for"] = []
    check("tampered or unsealed contrast designs fail closed",
          bool(search_design_manifest_issues(tampered_design)))

    def reseal_design(document):
        payload = {
            key: value for key, value in document.items() if key != "identity"
        }
        document["identity"]["sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    malformed_assignment_design = json.loads(json.dumps(design_document))
    nonempty_assignment = next(
        item for item in malformed_assignment_design["assignments"]
        if item["assignment"]
    )
    nonempty_assignment["assignment"][0].append("junk")
    reseal_design(malformed_assignment_design)
    duplicate_contrast_design = json.loads(json.dumps(design_document))
    treatment_item = next(
        item for item in duplicate_contrast_design["assignments"]
        if item["contrasts"]
    )
    treatment_item["contrasts"].append(
        json.loads(json.dumps(treatment_item["contrasts"][0]))
    )
    duplicate_contrast_design["identity"]["contrast_count"] += 1
    reseal_design(duplicate_contrast_design)
    malformed_roles_design = json.loads(json.dumps(design_document))
    malformed_roles_design["assignments"][0]["comparison_for"] = None
    reseal_design(malformed_roles_design)
    scored_ungraduated_design = json.loads(json.dumps(design_document))
    scoped_contrast = next(
        contrast
        for item in scored_ungraduated_design["assignments"]
        for contrast in item["contrasts"]
    )
    scoped_contrast["ungraduated_axes"] = [scoped_contrast["requested_axes"][0]]
    reseal_design(scored_ungraduated_design)
    check("sealed malformed assignment pairs and duplicate contrast entries fail closed",
          bool(search_design_manifest_issues(malformed_assignment_design))
          and bool(search_design_manifest_issues(duplicate_contrast_design))
          and bool(search_design_manifest_issues(malformed_roles_design))
          and bool(search_design_manifest_issues(scored_ungraduated_design)))
    check("record assignments reject empty/non-NY identities and blank tokens",
          _canonical_record_assignment([["", ""]]) is None
          and _canonical_record_assignment([["NY_OK", " "]]) is None
          and _canonical_record_assignment([["NY_OK", "1"]])
              == (("NY_OK", "1"),))
    with tempfile.TemporaryDirectory() as design_temp:
        atomic_design_path = Path(design_temp) / "search_plan.json"
        _atomic_write_utf8(
            atomic_design_path,
            json.dumps(design_document, indent=2, sort_keys=True) + "\n",
        )
        written_design = json.loads(
            atomic_design_path.read_text(encoding="utf-8")
        )
        check("the sealed contrast design is atomically persisted and revalidated",
              written_design == design_document
              and search_design_manifest_issues(written_design) == [])
        output_root = Path(design_temp) / "runs"
        output_root.mkdir()
        result_path = output_root / "results.jsonl"
        current_stub = {
            "schema": RESULT_SCHEMA,
            "search_design_schema": SEARCH_DESIGN_SCHEMA,
            "search_design_sha256": design_document["identity"]["sha256"],
            "arm_id": "baseline",
            "arm": [],
            "category": "cifar100",
            "repeat": 0,
            "valid": True,
            "structurally_valid": True,
            "evidence_scope": "scored",
            "contrasts": [],
            "comparison_for": [],
            "contrast_evidence": [],
            "scored_qualified_contrasts": [],
        }
        _atomic_write_utf8(result_path, json.dumps(current_stub) + "\n")
        prior_bytes = result_path.read_bytes()
        check("reuse accepts only a result stream wholly bound to its search plan",
              existing_result_stream_issues(
                  result_path, design_document["identity"]["sha256"]) == []
              and bool(existing_result_stream_issues(result_path, "0" * 64)))
        replacement = json.dumps(dict(current_stub, completed=True)) + "\n"
        _atomic_write_utf8(result_path, replacement)
        check("completed result streams replace rather than append prior records",
              prior_bytes != result_path.read_bytes()
              and result_path.read_text(encoding="utf-8") == replacement)
        alias_issues = output_layout_issues(
            output_root,
            output_root / arm_id([]) / "cifar100.r0" / "results.jsonl",
            output_root / "search_plan.json",
            [[]],
            ["cifar100"],
            1,
        )
        check("results cannot alias a plan, selector, or per-run artifact tree",
              bool(alias_issues)
              and bool(output_layout_issues(
                  output_root, output_root / "search_plan.json",
                  output_root / "search_plan.json", [[]], ["cifar100"], 1))
              and output_layout_issues(
                  output_root, output_root / "results.jsonl",
                  output_root / "search_plan.json", [[]], ["cifar100"], 1) == [])
    wide_axis = space.by_name["NY_ROOT_WIDE_DEMANDED_INTERM_CROWN"]
    preset_grid, _preset_grid_inert = generate_arms(
        space, [(wide_axis, ["1"])], "grid", 0, 0, 1, kinds=kinds,
        baseline_values={"NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN": "1"})
    check("grid explicitly disarms a preset owner that would suppress its treatment",
          any(dict(arm).get("NY_ROOT_WIDE_DEMANDED_INTERM_CROWN") == "1"
              and dict(arm).get("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN") == "0"
              for arm in preset_grid), repr(preset_grid))
    bounded_grid_names = {
        "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN",
        "NY_ROOT_PHASE_RESIDENT_CROWN",
        "NY_ROOT_WIDE_DEMANDED_INTERM_CROWN",
        "NY_ROOT_CPU_PARALLEL_INTERM_CROWN",
        "NY_ROOT_COMP_GPU_INTERM_ROWS",
        "NY_ROOT_COMP_GPU_INTERM_SECS",
        "NY_INTERM_ROW_CHUNKS",
    }
    bounded_grid_pool = [
        (axis, values) for axis, values in pool if axis.name in bounded_grid_names
    ]
    bounded_grid_candidates = 1
    for _axis, values in bounded_grid_pool:
        bounded_grid_candidates *= len(values) + 1
    bounded_grid_arms, _bounded_grid_inert = generate_arms(
        space, bounded_grid_pool, "grid", 0, 0, len(bounded_grid_pool),
        max_arms=256, max_candidates=bounded_grid_candidates, kinds=kinds,
        baseline_values=effective_baseline_values(space, kinds))
    check("grid applies max-arms after repair and deduplication",
          bounded_grid_candidates == 768 and 1 < len(bounded_grid_arms) <= 256,
          "%d raw candidates -> %d executable arms"
          % (bounded_grid_candidates, len(bounded_grid_arms)))
    grid_arms_refused = False
    try:
        generate_arms(
            space, bounded_grid_pool, "grid", 0, 0, len(bounded_grid_pool),
            max_arms=len(bounded_grid_arms) - 1,
            max_candidates=bounded_grid_candidates, kinds=kinds,
            baseline_values=effective_baseline_values(space, kinds))
    except SystemExit as error:
        grid_arms_refused = "--max-arms" in str(error)
    check("grid refuses rather than truncating executable arms", grid_arms_refused)
    grid_candidates_refused = False
    try:
        generate_arms(
            space, bounded_grid_pool, "grid", 0, 0, len(bounded_grid_pool),
            max_arms=256, max_candidates=bounded_grid_candidates - 1, kinds=kinds)
    except SystemExit as error:
        grid_candidates_refused = "--max-candidates" in str(error)
    check("grid refuses an excessive raw enumeration before running it",
          grid_candidates_refused)
    cover_arms, cover_inert = generate_arms(
        space, pool, "cover", 0, 999, 1, max_arms=512, kinds=kinds,
        baseline_values=effective_baseline_values(space, kinds))
    requested_pairs = {
        (axis.name, value) for axis, values in pool for value in values
    }
    covered_pairs = {(name, value) for arm in cover_arms for name, value in arm}
    check("cover deterministically exercises every T0-kept axis/value",
          requested_pairs <= covered_pairs and cover_inert == 0
          and not [arm for arm in cover_arms if is_inert(dict(arm), space)],
          "%d requested pairs in %d arms" % (len(requested_pairs), len(cover_arms)))
    check("cover ignores the random seed",
          cover_arms == generate_arms(
              space, pool, "cover", 0, 1, 1, max_arms=512, kinds=kinds,
              baseline_values=effective_baseline_values(space, kinds))[0])
    cover_refused = False
    try:
        generate_arms(space, pool, "cover", 0, 0, 1, max_arms=1, kinds=kinds)
    except SystemExit as error:
        cover_refused = "--max-arms" in str(error)
    check("cover refuses rather than truncating past max-arms", cover_refused)

    pair_pool = [
        (space.by_name["NY_MARGIN_ROW_GPU_BATCH"], ["0", "1"]),
        (space.by_name["NY_MARGIN_ROW_ALPHA_ITERS"], ["4", "16"]),
    ]
    pair_arms, pair_inert = generate_arms(
        space, pair_pool, "pair-cover", 0, 999, 1, max_arms=12,
        max_candidates=9, kinds=kinds)
    expected_pair_values = {
        (("NY_MARGIN_ROW_GPU_BATCH", batch), ("NY_MARGIN_ROW_ALPHA_ITERS", iters))
        for batch in ("0", "1") for iters in ("4", "16")
    }
    covered_pair_values = {
        pair for pair in expected_pair_values
        if any(all(dict(arm).get(name) == value for name, value in pair)
               for arm in pair_arms)
    }
    check("pair-cover includes baseline, every main effect, and every requested value pair",
          [] in pair_arms and len(pair_arms) == 12 and pair_inert == 0
          and covered_pair_values == expected_pair_values
          and not [arm for arm in pair_arms if is_inert(dict(arm), space)],
          repr(pair_arms))
    check("pair-cover is deterministic and ignores the random seed",
          pair_arms == generate_arms(
              space, pair_pool, "pair-cover", 0, 1, 99, max_arms=12,
              max_candidates=9, kinds=kinds)[0])

    bi_pair_pool = [
        (space.by_name["NY_MARGIN_ROW_BI_SECS"], ["20", "30"]),
        (space.by_name["NY_MARGIN_ROW_BI_CHUNK_SECS"], ["1", "2"]),
    ]
    bi_baselines = effective_baseline_values(space, kinds)
    bi_baselines["NY_MARGIN_ROW_BACKWARD_INTERM"] = "1"
    bi_baselines["NY_MARGIN_ROW_BI_SECS"] = "40"
    bi_arms, bi_pair_inert = generate_arms(
        space, bi_pair_pool, "pair-cover", 0, 0, 1, max_arms=9,
        max_candidates=9, kinds=kinds, baseline_values=bi_baselines,
    )
    expected_bi_pairs = {
        (("NY_MARGIN_ROW_BI_SECS", secs),
         ("NY_MARGIN_ROW_BI_CHUNK_SECS", chunk_secs))
        for secs in ("20", "30") for chunk_secs in ("1", "2")
    }
    covered_bi_pairs = {
        pair for pair in expected_bi_pairs
        if any(all(dict(arm).get(name) == value for name, value in pair)
               for arm in bi_arms)
    }
    check(
        "backward-intermediate pair-cover keeps every reviewed exact interaction",
        bi_pair_inert == 0 and covered_bi_pairs == expected_bi_pairs,
        repr(bi_arms),
    )
    pair_candidates_refused = False
    try:
        generate_arms(
            space, pair_pool, "pair-cover", 0, 0, 1, max_arms=12,
            max_candidates=8, kinds=kinds)
    except SystemExit as error:
        pair_candidates_refused = "--max-candidates" in str(error)
    check("pair-cover refuses rather than truncating raw candidates",
          pair_candidates_refused)
    pair_arms_refused = False
    try:
        generate_arms(
            space, pair_pool, "pair-cover", 0, 0, 1, max_arms=11,
            max_candidates=9, kinds=kinds)
    except SystemExit as error:
        pair_arms_refused = "--max-arms" in str(error)
    check("pair-cover refuses rather than truncating executable arms", pair_arms_refused)
    incompatible_pair = _repair_preserving(
        {"NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN": "0", "NY_INTERM_ROW_CHUNKS": "4"},
        space, kinds)
    check("dependency repair never overwrites a requested interaction coordinate",
          incompatible_pair is None, repr(incompatible_pair))
    pair_coordinate_refused = False
    try:
        generate_arms(
            space,
            [(space.by_name["NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"], ["0"]),
             (chunks_axis, ["4"])],
            "pair-cover", 0, 0, 1, max_arms=8, max_candidates=4, kinds=kinds)
    except SystemExit as error:
        pair_coordinate_refused = "requested coordinate" in str(error)
    check("pair-cover refuses an unrepresentable value pair", pair_coordinate_refused)

    random_underfill_refused = False
    try:
        generate_arms(
            space, [(space.by_name["NY_MARGIN_ROW_GPU"], ["1"])],
            "random", 2, 0, 1, kinds=kinds)
    except SystemExit as error:
        random_underfill_refused = (
            "requested 2 unique treatment arm" in str(error)
            and "generated only 1" in str(error)
        )
    check("random refuses explicit unique-arm underfill", random_underfill_refused)
    check("arm ids are canonical and order-free",
          arm_id([("A", "1"), ("B", "2")]) == arm_id([("B", "2"), ("A", "1")])
          and arm_id([]) == "baseline")
    check("generation is deterministic under a seed",
          generate_arms(space, pool, "random", 20, 11, 3, kinds=kinds)[0]
          == generate_arms(space, pool, "random", 20, 11, 3, kinds=kinds)[0])
    parser_default = build_parser().parse_args([])
    parser_interaction = build_parser().parse_args(["--max-axes", "3"])
    parser_ungraduated = build_parser().parse_args([
        "--allow-ungraduated-execution-contracts",
    ])
    check("random screening defaults to one sampled treatment axis",
          parser_default.max_axes == 1 and parser_interaction.max_axes == 3)
    check("production planning defaults to the explicit T1-graduated axis set",
          not parser_default.allow_ungraduated_execution_contracts
          and parser_ungraduated.allow_ungraduated_execution_contracts
          and "NY_MARGIN_ROW_GPU" not in _AXIS_SPECIFIC_T1_CONTRACTS
          and "NY_MO_KFSB_CERT_REUSE" not in _AXIS_SPECIFIC_T1_CONTRACTS
          and not set(_AXIS_SPECIFIC_T1_CONTRACTS).difference(space.by_name),
          repr(sorted(set(_AXIS_SPECIFIC_T1_CONTRACTS).difference(space.by_name))))
    parser_pair = build_parser().parse_args([
        "--mode", "pair-cover", "--axis", "A", "--axis", "B",
    ])
    check("pair-cover requires an explicit multi-axis shortlist and a baseline",
          parser_pair.mode == "pair-cover"
          and _pair_cover_request_error(
              parser_pair.mode, parser_pair.axis_filter, parser_pair.no_baseline) is None
          and _pair_cover_request_error("pair-cover", ["A"]) is not None
          and _pair_cover_request_error("pair-cover", ["A", "A"]) is not None
          and _pair_cover_request_error("pair-cover", ["A", "B"], True) is not None)
    check("raw candidate enumeration has an independent conservative default",
          parser_default.max_candidates == DEFAULT_MAX_CANDIDATES
          and parser_default.max_candidates > parser_default.max_arms)
    check("sidecar paths mirror Rust with_extension",
          sidecar(Path("/b/cifar100.csv"), ".metadata.jsonl") == Path("/b/cifar100.metadata.jsonl"))

    # --- exact native row-subset contract
    parser_subset = build_parser().parse_args(["--instances", "rows.csv"])
    check("--instances accepts the native selector file contract",
          parser_subset.instances == "rows.csv" and parser_subset.limit is None)
    check("duplicate category invocations are refused before they can fake repeats",
          duplicate_category_error(["cifar100", "cifar100"]) is not None
          and duplicate_category_error(["cifar100", "cgan"]) is None)
    check("zero and negative first-N requests fail before dry-run planning",
          limit_request_error(0) is not None
          and limit_request_error(-1) is not None
          and limit_request_error(1) is None)
    with tempfile.TemporaryDirectory() as tmp:
        subset_root = Path(tmp)
        selector_file = subset_root / "requested.csv"
        selector_file.write_text(
            "# exact near misses\n"
            "cifar100_2024,onnx/m.onnx,vnnlib/a.vnnlib\n"
            "cgan_2023,\"onnx/a.onnx,onnx/b.onnx\",vnnlib/b.vnnlib\n",
            encoding="utf-8",
        )
        selection = load_instance_selection(selector_file)
        reordered_file = subset_root / "reordered.csv"
        reordered_file.write_text(
            "cgan_2023,\"onnx/a.onnx,onnx/b.onnx\",vnnlib/b.vnnlib\n"
            "cifar100_2024,onnx/m.onnx,vnnlib/a.vnnlib\n",
            encoding="utf-8",
        )
        reordered = load_instance_selection(reordered_file)
        check("selector parsing preserves native order and quoted commas",
              selection["ordered"] == [
                  ("cifar100_2024", "onnx/m.onnx", "vnnlib/a.vnnlib"),
                  ("cgan_2023", "onnx/a.onnx,onnx/b.onnx", "vnnlib/b.vnnlib"),
              ], repr(selection["ordered"]))
        check("selector semantic digest uses Rust's set-like matching semantics",
              selection["identity"]["sha256"] == reordered["identity"]["sha256"])
        check("selector categories must exactly match native child invocations",
              instance_request_error(
                  selection, ["cifar100_2024", "cgan_2023"], None) is None
              and instance_request_error(selection, ["cifar100_2024"], None) is not None
              and "mutually exclusive" in instance_request_error(
                  selection, ["cifar100_2024", "cgan_2023"], 1))
        malformed_refused = duplicate_refused = False
        try:
            parse_instance_selectors('cifar100,"unterminated,vnnlib/p.vnnlib')
        except ValueError:
            malformed_refused = True
        try:
            parse_instance_selectors("cifar100,m,p\ncifar100,m,p\n")
        except ValueError:
            duplicate_refused = True
        check("malformed and duplicate selectors fail closed",
              malformed_refused and duplicate_refused)

        projected = materialize_instance_selections(selection, subset_root / "artifacts")
        check("multi-category selectors are projected without changing native values",
              set(projected) == {"cifar100_2024", "cgan_2023"}
              and parse_instance_selectors(
                  Path(projected["cgan_2023"]).read_text(encoding="utf-8"))
              == [("cgan_2023", "onnx/a.onnx,onnx/b.onnx", "vnnlib/b.vnnlib")])
        cifar_identity = selection["category_identities"]["cifar100_2024"]
        selected_rows = [
            {"category": "cifar100_2024", "onnx": "onnx/m.onnx",
             "vnnlib": "vnnlib/a.vnnlib", "occurrence": 0, "instance_index": 0},
        ]
        _row_identity, scoped_failures = validate_instance_selection(
            selection,
            "cifar100_2024",
            None,
            selected_rows,
            {
                "instance_selection": {"mode": "exact", "identity": cifar_identity},
                "selected_rows": {
                    "cifar100_2024": measured_row_set_identity(selected_rows),
                },
            },
        )
        check("native selector seals compare with the per-category projection",
              not scoped_failures
              and cifar_identity["sha256"] != selection["identity"]["sha256"],
              str(scoped_failures))

        duplicate_selector_file = subset_root / "duplicate-corpus-row.csv"
        duplicate_selector_file.write_text(
            "cifar100,onnx/m.onnx,vnnlib/p.vnnlib\n", encoding="utf-8")
        duplicate_selection = load_instance_selection(duplicate_selector_file)
        duplicate_rows = [
            {"category": "cifar100", "onnx": "onnx/m.onnx",
             "vnnlib": "vnnlib/p.vnnlib", "occurrence": 0, "instance_index": 2},
            {"category": "cifar100", "onnx": "onnx/m.onnx",
             "vnnlib": "vnnlib/p.vnnlib", "occurrence": 1, "instance_index": 5},
        ]
        complete_row_identity = measured_row_set_identity(duplicate_rows)
        exact_manifest = {
            "instance_selection": {
                "mode": "exact",
                "identity": duplicate_selection["category_identities"]["cifar100"],
            },
            "selected_rows": {"cifar100": complete_row_identity},
        }
        _identity, duplicate_failures = validate_instance_selection(
            duplicate_selection, "cifar100", None, duplicate_rows[:1], exact_manifest)
        check("a partial duplicate-occurrence bank cannot satisfy an exact selector",
              any(f["check"] == "manifest_selected_rows_match_bank"
                  for f in duplicate_failures), str(duplicate_failures))

        all_manifest = {
            "instance_selection": {"mode": "all"},
            "selected_rows": {"cifar100": complete_row_identity},
        }
        _identity, all_failures = validate_instance_selection(
            None, "cifar100", None, duplicate_rows, all_manifest)
        _identity, partial_all_failures = validate_instance_selection(
            None, "cifar100", None, duplicate_rows[:1], all_manifest)
        check("all-mode reuse requires the native full planned-row seal",
              not all_failures
              and any(f["check"] == "manifest_selected_rows_match_bank"
                      for f in partial_all_failures), str(partial_all_failures))

        limit_manifest = {
            "instance_selection": {"mode": "limit", "per_category": 2},
            "selected_rows": {"cifar100": complete_row_identity},
        }
        _identity, wrong_limit_failures = validate_instance_selection(
            None, "cifar100", 1, duplicate_rows, limit_manifest)
        _identity, exact_as_all_failures = validate_instance_selection(
            None, "cifar100", None, duplicate_rows, exact_manifest)
        check("all, limit, and exact native selection modes cannot masquerade as each other",
              any(f["check"] == "manifest_subset_matches_request"
                  for f in wrong_limit_failures)
              and any(f["check"] == "manifest_subset_matches_request"
                      for f in exact_as_all_failures),
              repr((wrong_limit_failures, exact_as_all_failures)))
        _identity, legacy_failures = validate_instance_selection(
            None, "cifar100", None, duplicate_rows, {})
        check("pre-seal manifests fail closed under artifact reuse",
              {f["check"] for f in legacy_failures}
              >= {"manifest_subset_matches_request", "manifest_selected_rows_match_bank"},
              str(legacy_failures))

        command_opts = argparse.Namespace(
            year=2025, vnnlib_version=None, configs_dir=None, limit=None,
            timeout_cap=None, overwrite=False, cache="off",
            instances=str(selector_file), instances_by_category=projected,
            categories=["cifar100_2024", "cgan_2023"], keep_ambient=False,
        )
        subset_cmd = build_command(
            Path("/tmp/ny"), [], "cgan_2023", Path("/tmp/out.csv"), command_opts,
            {"--instances"})
        subset_index = subset_cmd.index("--instances")
        check("build_command forwards the exact per-category selector file",
              subset_cmd[subset_index + 1] == projected["cgan_2023"]
              and "--limit" not in subset_cmd, repr(subset_cmd))
        unsupported_refused = False
        try:
            build_command(
                Path("/tmp/ny"), [], "cgan_2023", Path("/tmp/out.csv"), command_opts,
                set())
        except ValueError as error:
            unsupported_refused = "lacks" in str(error)
        check("subset execution fails closed when the binary lacks --instances",
              unsupported_refused)
        preflight_calls = []

        def fake_preflight_runner(command, **_kwargs):
            preflight_calls.append(command)
            return argparse.Namespace(returncode=0, stdout="{}", stderr="")

        preflight_instance_selection(
            Path("/tmp/ny"), selection, command_opts,
            {"--instances", "--dry-run"}, runner=fake_preflight_runner)
        check("selector dry-run invokes the native corpus planner for every category",
              len(preflight_calls) == 2
              and all("--dry-run" in command and "--instances" in command
                      and "--limit" not in command for command in preflight_calls),
              repr(preflight_calls))
    del dropped_inert

    def synthetic_contrast(requested, prerequisites=()):
        requested = sorted(requested)
        prerequisites = sorted(prerequisites)
        treatment = sorted(requested + prerequisites)
        payload = {
            "schema": SEARCH_DESIGN_SCHEMA,
            "requested_axes": sorted(name for name, _ in requested),
            "requested_assignment": [list(pair) for pair in requested],
            "prerequisites_added": [list(pair) for pair in prerequisites],
            "treatment_assignment": [list(pair) for pair in treatment],
            "comparison_assignment": [list(pair) for pair in prerequisites],
        }
        payload["contrast_id"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        payload["evidence_scope"] = "scored"
        payload["ungraduated_axes"] = []
        return payload

    def paired_record(arm_pairs, repeat, solved, category="cifar100", valid=True,
                      cached=0, backend="cuda", host="gb10", configs_sha="c" * 64,
                      selector_sha="s" * 64, row_set_sha="r" * 64,
                      contrast=None, as_control=False, comparison_for=()):
        canonical_arm = sorted(arm_pairs)
        if contrast is None and canonical_arm and not as_control:
            contrast = synthetic_contrast(canonical_arm)
        contrasts = [contrast] if contrast is not None else []
        return {
            "schema": RESULT_SCHEMA,
            "arm_id": arm_id(canonical_arm),
            "arm": [list(pair) for pair in canonical_arm],
            "category": category,
            "repeat": repeat,
            "year": 2025,
            "valid": valid,
            "structurally_valid": valid,
            "evidence_scope": "scored",
            "contrasts": contrasts,
            "comparison_for": sorted(comparison_for),
            "contrast_evidence": ([{
                "contrast_id": contrast["contrast_id"],
                "evidence_scope": contrast["evidence_scope"],
                "valid": True,
                "failures": [],
                "warnings": [],
            }] if contrast is not None else []),
            "scored_qualified_contrasts": (
                [contrast["contrast_id"]] if contrast is not None else []
            ),
            "search_design_schema": SEARCH_DESIGN_SCHEMA,
            "search_design_sha256": "d" * 64,
            "counts": {
                "rows": 20, "sat": 0, "unsat": solved, "cached_rows": cached,
                "capped_rows": 0,
            },
            "lower_bound": False,
            "executable_sha256": "a" * 64,
            "build_id": SELF_TEST_BUILD_ID,
            "compute_backend": backend,
            "host": host,
            "corpus_context": {
                "year": 2025,
                "vnnlib_version": None,
                "timeout_cap": None,
                "configs": {"sha256": configs_sha},
                "instances_csv": {"cifar100": {"sha256": "i" * 64}},
                "instance_selection": {
                    "schema": INSTANCE_SELECTOR_SCHEMA, "count": 2,
                    "sha256": selector_sha,
                },
                "category_instance_selection": {
                    "schema": INSTANCE_SELECTOR_SCHEMA, "count": 2,
                    "sha256": selector_sha,
                },
                "measured_row_set": {
                    "schema": MEASURED_ROW_SET_SCHEMA, "count": 20,
                    "sha256": row_set_sha,
                },
            },
        }

    margin_contrast = synthetic_contrast([
        ("NY_MARGIN_ROW_PARALLEL", "32"),
    ])
    paired_records = []
    for repeat, baseline_solved, treatment_solved in (
            (0, 10, 11), (1, 9, 8), (2, 11, 13)):
        paired_records.append(paired_record(
            [], repeat, baseline_solved,
            comparison_for=[margin_contrast["contrast_id"]],
        ))
        paired_records.append(
            paired_record(
                [("NY_MARGIN_ROW_PARALLEL", "32")], repeat, treatment_solved,
                contrast=margin_contrast,
            )
        )
    # These tempting large deltas are deliberately inadmissible: cache replay,
    # invalid treatment, unmatched backend/config tree, and a category with no
    # baseline.
    paired_records.extend([
        paired_record([], 3, 10, comparison_for=[margin_contrast["contrast_id"]]),
        paired_record([("NY_MARGIN_ROW_PARALLEL", "32")], 3, 50, cached=1,
                      contrast=margin_contrast),
        paired_record([], 4, 10, comparison_for=[margin_contrast["contrast_id"]]),
        paired_record([("NY_MARGIN_ROW_PARALLEL", "32")], 4, 50, valid=False,
                      contrast=margin_contrast),
        paired_record([], 5, 10, comparison_for=[margin_contrast["contrast_id"]]),
        paired_record([("NY_MARGIN_ROW_PARALLEL", "32")], 5, 50, backend="cpu",
                      contrast=margin_contrast),
        paired_record([], 6, 10, comparison_for=[margin_contrast["contrast_id"]]),
        paired_record([("NY_MARGIN_ROW_PARALLEL", "32")], 6, 50,
                      configs_sha="d" * 64, contrast=margin_contrast),
        paired_record([("NY_MARGIN_ROW_PARALLEL", "32")], 0, 50,
                      category="safenlp", contrast=margin_contrast),
    ])
    paired = paired_repeat_aggregates(paired_records)
    check(
        "paired aggregate uses only fully measured same-context baseline repeats",
        len(paired) == 1
        and paired[0]["pairs"] == 3
        and paired[0]["delta_median"] == 1
        and paired[0]["delta_min"] == -1
        and paired[0]["delta_max"] == 2
        and (paired[0]["wins"], paired[0]["ties"], paired[0]["losses"]) == (2, 0, 1)
        and paired[0]["treatment_median"] == 11
        and paired[0]["baseline_median"] == 10,
        repr(paired),
    )
    prerequisite_contrast = synthetic_contrast(
        [("NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL", "8")],
        [("NY_ROOT_JOINT_INTERM_ALPHA", "1")],
    )
    prerequisite_pairs = paired_repeat_aggregates([
        paired_record([], 30, 100),
        paired_record(
            [("NY_ROOT_JOINT_INTERM_ALPHA", "1")],
            30,
            10,
            as_control=True,
            comparison_for=[prerequisite_contrast["contrast_id"]],
        ),
        paired_record(
            [("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
             ("NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL", "8")],
            30,
            12,
            contrast=prerequisite_contrast,
        ),
    ])
    check("a repaired child pairs against its prerequisite-only control, not baseline",
          len(prerequisite_pairs) == 1
          and prerequisite_pairs[0]["delta_median"] == 2
          and prerequisite_pairs[0]["comparison_assignment"]
              == (("NY_ROOT_JOINT_INTERM_ALPHA", "1"),),
          repr(prerequisite_pairs))

    dual_role_control = paired_record(
        [("NY_ROOT_JOINT_INTERM_ALPHA", "1")],
        32,
        10,
        contrast=synthetic_contrast(
            [("NY_ROOT_JOINT_INTERM_ALPHA", "1")]
        ),
        comparison_for=[prerequisite_contrast["contrast_id"]],
    )
    dual_role_control["valid"] = False
    dual_role_control["structurally_valid"] = True
    dual_role_control["contrast_evidence"][0]["valid"] = False
    dual_role_control["contrast_evidence"][0]["failures"] = [{
        "check": "synthetic_parameter_engagement",
        "detail": "the dual-role treatment did not engage",
    }]
    dual_role_control["scored_qualified_contrasts"] = []
    dual_role_pairs = paired_repeat_aggregates([
        paired_record([], 32, 100),
        dual_role_control,
        paired_record(
            [("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
             ("NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL", "8")],
            32,
            12,
            contrast=prerequisite_contrast,
        ),
    ])
    check("a structurally valid assignment remains an exact comparator when its own "
          "treatment contrast fails",
          len(dual_role_pairs) == 1
          and dual_role_pairs[0]["delta_median"] == 2,
          repr(dual_role_pairs))

    stale_schema_control = paired_record([], 33, 10)
    stale_schema_control["comparison_for"] = [margin_contrast["contrast_id"]]
    stale_schema_control["schema"] = "ny-search/result/v1"
    check("stale result schemas cannot enter paired evidence",
          paired_repeat_aggregates([
              stale_schema_control,
              paired_record(
                  [("NY_MARGIN_ROW_PARALLEL", "32")], 33, 12,
                  contrast=margin_contrast),
          ]) == [])

    delivery_contrast = cert_contrast
    delivery_treatment_assignment = tuple(
        tuple(pair) for pair in delivery_contrast["treatment_assignment"]
    )
    delivery_comparison_assignment = tuple(
        tuple(pair) for pair in delivery_contrast["comparison_assignment"]
    )
    forged_scored_delivery = json.loads(json.dumps(delivery_contrast))
    forged_scored_delivery["evidence_scope"] = "scored"
    delivery_treatment = paired_record(
        delivery_treatment_assignment, 31, 50, contrast=delivery_contrast)
    delivery_treatment["evidence_scope"] = "delivery_only"
    delivery_treatment["contrast_evidence"][0]["evidence_scope"] = "delivery_only"
    delivery_treatment["scored_qualified_contrasts"] = []
    check("certificate-reuse delivery evidence cannot enter scored aggregates",
          _contrast_contract_valid(
              delivery_contrast,
              delivery_treatment_assignment,
          )
          and not _contrast_contract_valid(
              forged_scored_delivery,
              delivery_treatment_assignment,
          )
          and paired_repeat_aggregates([
              paired_record(
                  delivery_comparison_assignment,
                  31,
                  10,
                  as_control=True,
                  comparison_for=[delivery_contrast["contrast_id"]],
              ),
              delivery_treatment,
          ]) == [])
    selector_mismatch = paired_repeat_aggregates([
        paired_record([], 20, 10, selector_sha="1" * 64,
                      comparison_for=[margin_contrast["contrast_id"]]),
        paired_record([("NY_MARGIN_ROW_PARALLEL", "32")], 20, 50,
                      selector_sha="2" * 64, contrast=margin_contrast),
    ])
    row_set_mismatch = paired_repeat_aggregates([
        paired_record([], 21, 10, row_set_sha="3" * 64,
                      comparison_for=[margin_contrast["contrast_id"]]),
        paired_record([("NY_MARGIN_ROW_PARALLEL", "32")], 21, 50,
                      row_set_sha="4" * 64, contrast=margin_contrast),
    ])
    check("equal-sized different requested or measured subsets never pair",
          selector_mismatch == [] and row_set_mismatch == [],
          repr((selector_mismatch, row_set_mismatch)))

    duplicate_evidence_control = paired_record(
        [], 22, 10, comparison_for=[margin_contrast["contrast_id"]])
    duplicate_evidence_treatment = paired_record(
        [("NY_MARGIN_ROW_PARALLEL", "32")], 22, 12,
        contrast=margin_contrast)
    duplicate_evidence_treatment["contrast_evidence"].append(
        dict(duplicate_evidence_treatment["contrast_evidence"][0])
    )
    missing_role_control = paired_record([], 23, 10)
    check("duplicate/partial evidence and undeclared comparator roles fail closed",
          paired_repeat_aggregates([
              duplicate_evidence_control, duplicate_evidence_treatment,
          ]) == []
          and paired_repeat_aggregates([
              missing_role_control,
              paired_record(
                  [("NY_MARGIN_ROW_PARALLEL", "32")], 23, 12,
                  contrast=margin_contrast),
          ]) == [])

    separate_context_records = []
    for repeat, configs_sha in ((24, "1" * 64), (25, "2" * 64)):
        separate_context_records.extend([
            paired_record(
                [], repeat, 10, configs_sha=configs_sha,
                comparison_for=[margin_contrast["contrast_id"]],
            ),
            paired_record(
                [("NY_MARGIN_ROW_PARALLEL", "32")], repeat, 12,
                configs_sha=configs_sha, contrast=margin_contrast,
            ),
        ])
    separate_context_aggregates = paired_repeat_aggregates(
        separate_context_records)
    check("repeat aggregation never merges distinct corpus/config/design contexts",
          len(separate_context_aggregates) == 2
          and all(item["pairs"] == 1 for item in separate_context_aggregates),
          repr(separate_context_aggregates))

    # --- T1 against synthetic artifacts
    def machine_event(method, status, **fields):
        return {
            "method": method,
            "status": status,
            "reason": " ".join("%s=%s" % item for item in fields.items()),
        }

    def mo_kfsb_event(status=None, **overrides):
        fields = {
            "schema": _MO_KFSB_EVENT_SCHEMA,
            "phase": "terminal",
            "observation_schema": _MO_KFSB_OBSERVATION_SCHEMA,
            "canonical_master": "exact_enable",
            "legacy_childsim": "false",
            "configured_master": "false",
            "configured_candidates": "2",
            "effective_candidates": "2",
            "configured_reduce_op": "max",
            "effective_reduce_op": "max",
            "heuristic_eligible": "true",
            "branch_opportunities": "1",
            "offered_domains": "3",
            "nonempty_domain_opportunities": "1",
            "candidate_eligible_opportunities": "1",
            "typed_select_opportunities": "0",
            "lower_master_opportunities": "0",
            "effective_master_opportunities": "1",
            "route_eligible_opportunities": "1",
            "selector_waves": "1",
            "selector_domains": "3",
            "committed_waves": "1",
            "committed_parents": "1",
            # A committed parent may legitimately have no feasible leaves.
            "committed_leaves": "0",
            "force_on_binding_waves": "1",
            "master_binding_refusals": "0",
        }
        fields.update(overrides)
        if status is None:
            status = "ran" if fields["selector_waves"] != "0" else "skipped"
        return machine_event("mo_kfsb_execution", status, **fields)

    def mo_kfsb_disable_event(**overrides):
        fields = {
            "canonical_master": "exact_disable",
            "configured_master": "true",
            "lower_master_opportunities": "1",
            "effective_master_opportunities": "0",
            "selector_waves": "0",
            "selector_domains": "0",
            "committed_waves": "0",
            "committed_parents": "0",
            "committed_leaves": "0",
            "force_on_binding_waves": "0",
            "master_binding_refusals": "1",
        }
        fields.update(overrides)
        return mo_kfsb_event(status="skipped", **fields)

    def synthetic_mo_kfsb_result(token, events):
        arm_pairs = [("NY_MO_KFSB", token)]
        synthetic_row = _clean_row(arm_pairs)
        synthetic_row["flight"]["events"] = list(events)
        return validate_run(
            arm_pairs,
            2025,
            "cifar100",
            [synthetic_row],
            _clean_manifest(arm=arm_pairs),
            _summary_of([synthetic_row]),
        )

    def mo_kfsb_reduce_event(reduce_op="max", **overrides):
        configured_reduce_op = "max" if reduce_op == "min" else "min"
        fields = {
            "canonical_master": "exact_enable",
            "configured_master": "false",
            "configured_reduce_op": configured_reduce_op,
            "effective_reduce_op": reduce_op,
            "lower_master_opportunities": "0",
            "effective_master_opportunities": "1",
            "force_on_binding_waves": "1",
        }
        fields.update(overrides)
        return mo_kfsb_event(**fields)

    def synthetic_mo_kfsb_reduce_result(token, events):
        arm_pairs = [
            ("NY_MO_KFSB", "1"),
            ("NY_MO_KFSB_REDUCE", token),
        ]
        engagement_pairs = [("NY_MO_KFSB_REDUCE", token)]
        synthetic_row = _clean_row(arm_pairs)
        synthetic_row["flight"]["events"] = list(events)
        return validate_run(
            arm_pairs,
            2025,
            "cifar100",
            [synthetic_row],
            _clean_manifest(arm=arm_pairs),
            _summary_of([synthetic_row]),
            engagement_arm=engagement_pairs,
        )

    def mo_alpha_step_event(status=None, **overrides):
        fields = {
            "schema": _MO_ALPHA_STEP_EVENT_SCHEMA,
            "phase": "terminal",
            "observation_schema": _MO_ALPHA_STEP_OBSERVATION_SCHEMA,
            "observed": "true",
            "effective_alpha_step": "true",
            "effective_beta_spsa": "true",
            "schedule_period": "4",
            "alpha_round_remainder": "3",
            "optimizer_invocations": "1",
            "eligible_invocations": "1",
            "eligible_iterations": "15",
            "fourth_iteration_opportunities": "3",
            "alpha_rounds_scheduled": "3",
            "beta_rounds_scheduled": "12",
            "alpha_rounds_completed": "3",
            "alpha_rounds_refused": "0",
            "alpha_updates_applied": "3",
            "perturbed_coordinates": "96",
        }
        fields.update(overrides)
        if status is None:
            status = (
                "ran" if fields["alpha_rounds_completed"] != "0"
                else "skipped"
            )
        return machine_event("mo_alpha_step_execution", status, **fields)

    def synthetic_mo_alpha_step_result(token, events):
        arm_pairs = [
            ("NY_MO_ALPHA_STEP", token),
            ("NY_MO_BETA_SPSA", "1"),
        ]
        engagement_pairs = [("NY_MO_ALPHA_STEP", token)]
        synthetic_row = _clean_row(arm_pairs)
        synthetic_row["flight"]["events"] = list(events)
        return validate_run(
            arm_pairs,
            2025,
            "cifar100",
            [synthetic_row],
            _clean_manifest(arm=arm_pairs),
            _summary_of([synthetic_row]),
            engagement_arm=engagement_pairs,
        )

    def upfront_policy_event(version="v2", **overrides):
        fields = {
            "schema": f"ny-cli/upfront-policy/{version}",
            "master": "contextual",
            "route": "auto_disjunction",
            "lower_master": "contextual",
            "lower_route": "auto_disjunction",
            "wall_cap_secs": "8",
            "auto_wall_cap_secs": "4",
            "budget_fraction": "0.08",
            "probe_first": "false",
            "lower_probe_first": "false",
        }
        if version == "v2":
            fields.update({
                "bnn_wall_cap_secs": "240",
                "bnn_budget_fraction": "0.5",
            })
        fields.update(overrides)
        return machine_event("upfront_attack_policy", "ran", **fields)

    def upfront_execution_event(status="ran", version="v2", **overrides):
        fields = {
            "schema": f"ny-cli/upfront-execution/{version}",
            "consultation": "ordinary",
            "route": "auto_disjunction",
            "admission": "auto_disjunction",
            "sign_net": "false",
            "requested_window": "private",
            "requested_cap_secs": "none",
            "budget_authority": "private",
            "binding": "auto_wall_cap",
            "remaining_secs": "100.000000",
            "effective_budget_secs": "4.000000",
            "inactive_reason": "none",
        }
        if version == "v2":
            fields["budget_competition"] = "not_sign"
        fields.update(overrides)
        return machine_event("upfront_attack_execution", status, **fields)

    def synthetic_upfront_result(arm_pairs, executions, policy=None, category="cifar100"):
        synthetic_row = _clean_row(arm_pairs, category=category)
        synthetic_row["flight"]["events"] = [
            policy or upfront_policy_event(),
            *executions,
        ]
        return validate_run(
            arm_pairs,
            2025,
            category,
            [synthetic_row],
            _clean_manifest(category=category, arm=arm_pairs),
            _summary_of([synthetic_row]),
        )

    def margin_row_frontier_event(status="ran", **overrides):
        fields = {
            "schema": _MARGIN_ROW_FRONTIER_EVENT_SCHEMA,
            "phase": "terminal",
            "route": "joint",
            "outcome": "unknown",
            "tree_started": "true",
            "configured": "1",
            "contextual": "8",
            "steps": "4",
            "counterfactual_differences": "3",
            "max_open": "8",
            "max_batch": "1",
        }
        fields.update(overrides)
        return machine_event("margin_row_frontier_execution", status, **fields)

    def synthetic_margin_row_frontier_result(arm_pairs, events, category="cifar100"):
        synthetic_row = _clean_row(arm_pairs, category=category)
        synthetic_row["flight"]["events"] = list(events)
        return validate_run(
            arm_pairs,
            2025,
            category,
            [synthetic_row],
            _clean_manifest(category=category, arm=arm_pairs),
            _summary_of([synthetic_row]),
        )

    def margin_row_frontier_contract_accepts(event):
        synthetic_row = _clean_row([])
        synthetic_row["flight"]["events"] = [event]
        return len(_margin_row_frontier_terminal_evidences(synthetic_row)) == 1

    positive_frontier_cases = [
        ("0", {}),
        ("2", {"configured": "2", "max_batch": "2"}),
        ("4", {"configured": "4", "max_batch": "4"}),
        ("8", {
            "configured": "8", "contextual": "4", "max_batch": "8",
            "counterfactual_differences": "1",
        }),
        ("16", {
            "configured": "16", "contextual": "8", "max_open": "16",
            "max_batch": "16", "counterfactual_differences": "1",
        }),
    ]
    positive_frontier_results = []
    for token, overrides in positive_frontier_cases:
        valid, failures, _warnings, _counts = synthetic_margin_row_frontier_result(
            [("NY_MARGIN_ROW_PARALLEL", token)],
            [margin_row_frontier_event(**overrides)],
        )
        positive_frontier_results.append((token, valid, failures))
    check("all canonical frontier tokens map to their actual effective width",
          all(valid for _token, valid, _failures in positive_frontier_results),
          repr(positive_frontier_results))

    root_only_frontier = margin_row_frontier_event(
        tree_started="false", steps="0", counterfactual_differences="0",
        max_open="0", max_batch="0",
    )
    no_credit_frontier_cases = [
        ("no event", []),
        ("root only", [root_only_frontier]),
        ("equal context", [margin_row_frontier_event(
            configured="2", contextual="2", counterfactual_differences="0",
            max_batch="2",
        )]),
        ("never wide", [margin_row_frontier_event(
            configured="1", contextual="8", steps="1",
            counterfactual_differences="0", max_open="1", max_batch="1",
        )]),
        ("classwise", [margin_row_frontier_event(
            route="classwise", configured="1", contextual="1",
            counterfactual_differences="0", max_batch="1",
        )]),
        ("mismatched request", [margin_row_frontier_event(
            configured="2", contextual="8", max_batch="2",
        )]),
    ]
    no_credit_frontier_results = []
    for label, events in no_credit_frontier_cases:
        valid, failures, _warnings, _counts = synthetic_margin_row_frontier_result(
            [("NY_MARGIN_ROW_PARALLEL", "0")], events,
        )
        no_credit_frontier_results.append((label, valid, failures))
    check("non-executing or non-counterfactual frontier receipts earn no credit",
          all(
              not valid
              and any(failure["check"] == "margin_row_frontier_parameter_engaged"
                      for failure in failures)
              for _label, valid, failures in no_credit_frontier_results
          ), repr(no_credit_frontier_results))

    matching_frontier_row = _clean_row(
        [("NY_MARGIN_ROW_PARALLEL", "0")], category="cifar100",
    )
    matching_frontier_row["flight"]["events"] = [margin_row_frontier_event(
        contextual="1", counterfactual_differences="0",
    )]
    differing_frontier_row = _clean_row(
        [("NY_MARGIN_ROW_PARALLEL", "0")], category="cifar100",
    )
    differing_frontier_row["flight"]["events"] = [margin_row_frontier_event(
        configured="2", max_batch="2",
    )]
    check("frontier request matching and counterfactual difference cannot cross rows",
          not _margin_row_frontier_parameter_engagement(
              [("NY_MARGIN_ROW_PARALLEL", "0")],
              [matching_frontier_row, differing_frontier_row],
          ))

    missing_frontier_field = margin_row_frontier_event()
    missing_frontier_field["reason"] = missing_frontier_field["reason"].replace(
        " max_batch=1", "",
    )
    extra_frontier_field = margin_row_frontier_event()
    extra_frontier_field["reason"] += " extra=1"
    duplicate_frontier_field = margin_row_frontier_event()
    duplicate_frontier_field["reason"] += " steps=4"
    double_space_frontier = margin_row_frontier_event()
    double_space_frontier["reason"] = double_space_frontier["reason"].replace(
        " phase=", "  phase=",
    )
    tab_frontier = margin_row_frontier_event()
    tab_frontier["reason"] = tab_frontier["reason"].replace(" phase=", "\tphase=")
    invalid_frontier_events = [
        missing_frontier_field,
        extra_frontier_field,
        duplicate_frontier_field,
        double_space_frontier,
        tab_frontier,
        margin_row_frontier_event(schema="ny-cli/margin-row-frontier-execution/v999"),
        margin_row_frontier_event(status="skipped"),
        margin_row_frontier_event(phase="progress"),
        margin_row_frontier_event(route="other"),
        margin_row_frontier_event(outcome="timeout"),
        margin_row_frontier_event(configured="01"),
        margin_row_frontier_event(configured=str(1 << 64)),
        margin_row_frontier_event(configured="0"),
        margin_row_frontier_event(contextual="0"),
        margin_row_frontier_event(steps="1", counterfactual_differences="2"),
        margin_row_frontier_event(
            tree_started="true", steps="0", counterfactual_differences="0",
            max_open="0", max_batch="0",
        ),
        margin_row_frontier_event(max_batch="2"),
        margin_row_frontier_event(
            steps="1", counterfactual_differences="1", max_open="1", max_batch="1",
        ),
        margin_row_frontier_event(counterfactual_differences="0"),
        margin_row_frontier_event(
            route="classwise", configured="2", contextual="1", max_batch="2",
            counterfactual_differences="1",
        ),
    ]
    check("frontier v1 rejects malformed grammar and impossible counter correlations",
          all(not margin_row_frontier_contract_accepts(event)
              for event in invalid_frontier_events))

    earlier_frontier_valid, earlier_frontier_failures, _warnings, _counts = (
        synthetic_margin_row_frontier_result(
            [("NY_MARGIN_ROW_PARALLEL", "0")],
            [margin_row_frontier_event(), root_only_frontier],
        )
    )
    malformed_later_frontier = margin_row_frontier_event()
    malformed_later_frontier["reason"] += " steps=4"
    poisoned_frontier_valid, poisoned_frontier_failures, _warnings, _counts = (
        synthetic_margin_row_frontier_result(
            [("NY_MARGIN_ROW_PARALLEL", "0")],
            [margin_row_frontier_event(), malformed_later_frontier],
        )
    )
    check("a later root-only frontier invocation does not erase earlier engagement",
          earlier_frontier_valid, repr(earlier_frontier_failures))
    check("a malformed same-method frontier event poisons that row's evidence",
          not poisoned_frontier_valid
          and any(failure["check"] == "margin_row_frontier_parameter_engaged"
                  for failure in poisoned_frontier_failures),
          repr(poisoned_frontier_failures))

    def margin_row_policy_event(status="ran", **overrides):
        fields = {
            "schema": _MARGIN_ROW_POLICY_EVENT_SCHEMA,
            "phase": "terminal",
            "outcome": "unknown",
            "beta_per_column": "true",
            "beta_columns": "4",
            "column_retire": "true",
            "parallel_prebuild": "true",
            "beta_per_column_evals": "3",
            "beta_columns_binding_evals": "2",
            "beta_columns_max_candidates": "7",
            "newly_retired_columns": "5",
            "inherited_retirement_evals": "4",
            "parallel_prebuild_batches": "2",
            "parallel_prebuild_domains": "4",
            "policy_mismatch": "false",
        }
        fields.update(overrides)
        return machine_event("margin_row_policy_execution", status, **fields)

    def synthetic_margin_row_policy_result(arm_pairs, events, category="cifar100"):
        synthetic_row = _clean_row(arm_pairs, category=category)
        synthetic_row["flight"]["events"] = list(events)
        return validate_run(
            arm_pairs,
            2025,
            category,
            [synthetic_row],
            _clean_manifest(category=category, arm=arm_pairs),
            _summary_of([synthetic_row]),
        )

    joint_margin_policy_arm = [
        ("NY_MARGIN_ROW_BETA_PERCOL", "1"),
        ("NY_MARGIN_ROW_BETA_COLS", "4"),
        ("NY_MARGIN_ROW_COL_RETIRE", "1"),
        ("NY_MARGIN_ROW_PARALLEL_PREBUILD", "1"),
    ]
    joint_policy_valid, joint_policy_failures, _warnings, _counts = (
        synthetic_margin_row_policy_result(
            joint_margin_policy_arm,
            [margin_row_policy_event()],
        )
    )
    check("one closed margin-row policy receipt can authenticate a same-family joint arm",
          joint_policy_valid, repr(joint_policy_failures))

    full_policy_row = _clean_row(joint_margin_policy_arm)
    full_policy_row["flight"]["events"] = [margin_row_policy_event()]
    partial_policy_row = _clean_row(joint_margin_policy_arm)
    partial_policy_row["occurrence"] = 1
    partial_policy_row["instance_index"] = 1
    partial_policy_row["flight"]["events"] = [margin_row_policy_event(
        newly_retired_columns="0", inherited_retirement_evals="0",
    )]
    policy_rows = [full_policy_row, partial_policy_row]
    policy_intersection = _every_row_parameter_engagement(
        _margin_row_policy_parameter_engagement,
        joint_margin_policy_arm,
        policy_rows,
    )
    policy_all_rows_valid, policy_all_rows_failures, _warnings, _counts = (
        validate_run(
            joint_margin_policy_arm,
            2025,
            "cifar100",
            policy_rows,
            _clean_manifest(arm=joint_margin_policy_arm),
            _summary_of(policy_rows),
        )
    )
    check(
        "multi-axis families retain only axes engaged independently on every row",
        policy_intersection == {
            "NY_MARGIN_ROW_BETA_PERCOL",
            "NY_MARGIN_ROW_BETA_COLS",
            "NY_MARGIN_ROW_PARALLEL_PREBUILD",
        }
        and not policy_all_rows_valid
        and any(
            failure["check"] == "margin_row_policy_parameter_engaged"
            and "for NY_MARGIN_ROW_COL_RETIRE;" in failure["detail"]
            for failure in policy_all_rows_failures
        ),
        repr((policy_intersection, policy_all_rows_failures)),
    )

    no_credit_policy_cases = [
        ("per-column never evaluated", {"beta_per_column_evals": "0",
                                         "beta_columns_binding_evals": "0",
                                         "beta_columns_max_candidates": "0"}),
        ("width never bound", {"beta_columns_binding_evals": "0"}),
        ("no retirement", {"newly_retired_columns": "0",
                            "inherited_retirement_evals": "0"}),
        ("retirement never consumed", {"newly_retired_columns": "5",
                                        "inherited_retirement_evals": "0"}),
        ("single-domain prebuild", {"parallel_prebuild_batches": "1",
                                     "parallel_prebuild_domains": "1"}),
        ("configured width mismatch", {"beta_columns": "8"}),
    ]
    no_credit_policy_results = []
    for label, overrides in no_credit_policy_cases:
        valid, failures, _warnings, _counts = synthetic_margin_row_policy_result(
            joint_margin_policy_arm,
            [margin_row_policy_event(**overrides)],
        )
        no_credit_policy_results.append((label, valid, failures))
    check("configuration-only and non-binding margin-row policy work earns no credit",
          all(not valid
              and any(failure["check"] == "margin_row_policy_parameter_engaged"
                      for failure in failures)
              for _label, valid, failures in no_credit_policy_results),
          repr(no_credit_policy_results))

    malformed_policy = margin_row_policy_event()
    malformed_policy["reason"] += " beta_columns=4"
    mismatch_policy = margin_row_policy_event(policy_mismatch="true")
    impossible_policy = margin_row_policy_event(
        beta_per_column_evals="0",
        beta_columns_binding_evals="0",
        beta_columns_max_candidates="7",
    )
    check("margin-row policy v1 rejects duplicate, mismatched, and impossible evidence",
          all(not _margin_row_policy_terminal_evidences({
              "flight": {"events": [event]},
          }) for event in (malformed_policy, mismatch_policy, impossible_policy)))

    poisoned_policy_valid, poisoned_policy_failures, _warnings, _counts = (
        synthetic_margin_row_policy_result(
            joint_margin_policy_arm,
            [margin_row_policy_event(), malformed_policy],
        )
    )
    check("a malformed same-method policy event poisons that row's evidence",
          not poisoned_policy_valid
          and any(failure["check"] == "margin_row_policy_parameter_engaged"
                  for failure in poisoned_policy_failures),
          repr(poisoned_policy_failures))

    def margin_row_backward_interm_event(status="ran", **overrides):
        fields = {
            "schema": _MARGIN_ROW_BACKWARD_INTERM_EVENT_SCHEMA,
            "phase": "terminal",
            "outcome": "exhausted",
            "enabled": "true",
            "configured_secs": "20",
            "configured_chunk": "64",
            "configured_chunk_secs": "2",
            "effective_chunk_secs": "2",
            "configured_topk": "1024",
            "eligible_layers": "4",
            "suppressed_setup_layers": "0",
            "layers_started": "3",
            "chunks_completed": "6",
            "columns_completed": "512",
            "deadline_capped": "false",
            "budget_exhausted": "true",
            "chunk_secs_capped": "false",
            "chunk_counterfactual_chunks": "2",
            "chunk_ceiling_binding_chunks": "2",
            "fixed_width_chunks": "0",
            "adaptive_width_changes": "2",
            "topk_binding_layers": "2",
            "policy_mismatch": "false",
        }
        fields.update(overrides)
        return machine_event(
            "margin_row_backward_interm_execution", status, **fields
        )

    def disabled_margin_row_backward_interm_event(**overrides):
        disabled = {
            "outcome": "disabled",
            "enabled": "false",
            "eligible_layers": "4",
            "suppressed_setup_layers": "4",
            "layers_started": "0",
            "chunks_completed": "0",
            "columns_completed": "0",
            "deadline_capped": "false",
            "budget_exhausted": "false",
            "chunk_counterfactual_chunks": "0",
            "chunk_ceiling_binding_chunks": "0",
            "fixed_width_chunks": "0",
            "adaptive_width_changes": "0",
            "topk_binding_layers": "0",
        }
        disabled.update(overrides)
        return margin_row_backward_interm_event(**disabled)

    def synthetic_margin_row_backward_interm_result(
        arm_pairs, events, category="cifar100"
    ):
        synthetic_row = _clean_row(arm_pairs, category=category)
        synthetic_row["flight"]["events"] = list(events)
        return validate_run(
            arm_pairs,
            2025,
            category,
            [synthetic_row],
            _clean_manifest(category=category, arm=arm_pairs),
            _summary_of([synthetic_row]),
        )

    def margin_row_backward_interm_contract_accepts(event):
        return len(_margin_row_backward_interm_terminal_evidences({
            "flight": {"events": [event]},
        })) == 1

    joint_backward_interm_arm = [
        ("NY_MARGIN_ROW_BACKWARD_INTERM", "1"),
        ("NY_MARGIN_ROW_BI_SECS", "20"),
        ("NY_MARGIN_ROW_BI_CHUNK", "64"),
        ("NY_MARGIN_ROW_BI_CHUNK_SECS", "2"),
        ("NY_MARGIN_ROW_BI_TOPK", "1024"),
    ]
    joint_bi_valid, joint_bi_failures, _warnings, _counts = (
        synthetic_margin_row_backward_interm_result(
            joint_backward_interm_arm,
            [margin_row_backward_interm_event()],
        )
    )
    check("one closed backward-intermediate receipt authenticates a joint family arm",
          joint_bi_valid, repr(joint_bi_failures))

    canonical_bi_cases = [
        (
            "NY_MARGIN_ROW_BACKWARD_INTERM", "0",
            disabled_margin_row_backward_interm_event(),
        ),
        (
            "NY_MARGIN_ROW_BACKWARD_INTERM", "1",
            margin_row_backward_interm_event(
                outcome="completed", budget_exhausted="false",
            ),
        ),
        *[
            (
                "NY_MARGIN_ROW_BI_SECS", token,
                margin_row_backward_interm_event(configured_secs=token),
            )
            for token in ("20", "30")
        ],
        (
            "NY_MARGIN_ROW_BI_CHUNK", "64",
            margin_row_backward_interm_event(configured_chunk="64"),
        ),
        (
            "NY_MARGIN_ROW_BI_CHUNK", "256",
            margin_row_backward_interm_event(
                configured_chunk="256", configured_chunk_secs="0",
                effective_chunk_secs="0", chunk_counterfactual_chunks="0",
                chunk_ceiling_binding_chunks="2",
                fixed_width_chunks="2", adaptive_width_changes="0",
            ),
        ),
        (
            "NY_MARGIN_ROW_BI_CHUNK", "1024",
            margin_row_backward_interm_event(configured_chunk="1024"),
        ),
        (
            "NY_MARGIN_ROW_BI_CHUNK_SECS", "0",
            margin_row_backward_interm_event(
                configured_chunk_secs="0", effective_chunk_secs="0",
                fixed_width_chunks="2", adaptive_width_changes="0",
            ),
        ),
        *[
            (
                "NY_MARGIN_ROW_BI_CHUNK_SECS", token,
                margin_row_backward_interm_event(
                    configured_chunk_secs=token, effective_chunk_secs=token,
                ),
            )
            for token in ("1", "2")
        ],
        *[
            (
                "NY_MARGIN_ROW_BI_TOPK", token,
                margin_row_backward_interm_event(configured_topk=token),
            )
            for token in ("256", "1024", "4096")
        ],
    ]
    canonical_bi_results = []
    for name, token, event in canonical_bi_cases:
        valid, failures, _warnings, _counts = (
            synthetic_margin_row_backward_interm_result(
                [(name, token)], [event]
            )
        )
        canonical_bi_results.append((name, token, valid, failures))
    check("every backward-intermediate search token has closed execution evidence",
          all(valid for _name, _token, valid, _failures in canonical_bi_results),
          repr(canonical_bi_results))

    narrow_default_chunk = margin_row_backward_interm_event(
        configured_chunk="256",
        configured_chunk_secs="0",
        effective_chunk_secs="0",
        chunk_counterfactual_chunks="0",
        chunk_ceiling_binding_chunks="0",
        fixed_width_chunks="2",
        adaptive_width_changes="0",
    )
    narrow_default_valid, narrow_default_failures, _warnings, _counts = (
        synthetic_margin_row_backward_interm_result(
            [("NY_MARGIN_ROW_BI_CHUNK", "256")],
            [narrow_default_chunk],
        )
    )
    check(
        "a fixed narrow workload cannot credit the default chunk ceiling",
        not narrow_default_valid
        and margin_row_backward_interm_contract_accepts(narrow_default_chunk)
        and any(
            failure["check"]
            == "margin_row_backward_interm_parameter_engaged"
            for failure in narrow_default_failures
        ),
        repr(narrow_default_failures),
    )

    disabled_without_eligible = disabled_margin_row_backward_interm_event(
        eligible_layers="0",
        suppressed_setup_layers="0",
    )
    disabled_empty_valid, disabled_empty_failures, _warnings, _counts = (
        synthetic_margin_row_backward_interm_result(
            [("NY_MARGIN_ROW_BACKWARD_INTERM", "0")],
            [disabled_without_eligible],
        )
    )
    disabled_without_budget = disabled_margin_row_backward_interm_event(
        configured_secs="0",
        effective_chunk_secs="0.000000001",
        chunk_secs_capped="true",
        suppressed_setup_layers="0",
    )
    disabled_budget_valid, disabled_budget_failures, _warnings, _counts = (
        synthetic_margin_row_backward_interm_result(
            [("NY_MARGIN_ROW_BACKWARD_INTERM", "0")],
            [disabled_without_budget],
        )
    )
    check(
        "producer-realizable disabled receipts need structural and clock admissibility",
        margin_row_backward_interm_contract_accepts(disabled_without_eligible)
        and margin_row_backward_interm_contract_accepts(disabled_without_budget)
        and not disabled_empty_valid
        and not disabled_budget_valid
        and any(
            failure["check"]
            == "margin_row_backward_interm_parameter_engaged"
            for failure in disabled_empty_failures
        )
        and any(
            failure["check"]
            == "margin_row_backward_interm_parameter_engaged"
            for failure in disabled_budget_failures
        ),
        repr((disabled_empty_failures, disabled_budget_failures)),
    )

    no_work_bi_cases = [
        ("no event", []),
        ("enabled but no eligible layer", [margin_row_backward_interm_event(
            outcome="completed", eligible_layers="0", layers_started="0",
            chunks_completed="0", columns_completed="0",
            budget_exhausted="false", chunk_counterfactual_chunks="0",
            chunk_ceiling_binding_chunks="0",
            adaptive_width_changes="0", topk_binding_layers="0",
        )]),
        ("no layer started", [margin_row_backward_interm_event(
            outcome="completed", layers_started="0", chunks_completed="0",
            columns_completed="0", budget_exhausted="false",
            chunk_counterfactual_chunks="0", adaptive_width_changes="0",
            chunk_ceiling_binding_chunks="0",
            topk_binding_layers="0",
        )]),
        ("no chunk completed", [margin_row_backward_interm_event(
            outcome="completed", layers_started="1", chunks_completed="0",
            columns_completed="0", budget_exhausted="false",
            chunk_counterfactual_chunks="0", adaptive_width_changes="0",
            chunk_ceiling_binding_chunks="0",
            topk_binding_layers="0",
        )]),
        ("deadline capped", [margin_row_backward_interm_event(
            deadline_capped="true",
        )]),
        ("budget not exhausted", [margin_row_backward_interm_event(
            outcome="completed", budget_exhausted="false",
        )]),
        ("configured chunk never changed a width", [margin_row_backward_interm_event(
            chunk_counterfactual_chunks="0",
        )]),
        ("adaptive target was capped", [margin_row_backward_interm_event(
            effective_chunk_secs="1", chunk_secs_capped="true",
        )]),
        ("adaptive width never changed", [margin_row_backward_interm_event(
            adaptive_width_changes="0",
        )]),
        ("top-k never stopped admissible work", [margin_row_backward_interm_event(
            topk_binding_layers="0",
        )]),
    ]
    no_work_bi_results = []
    for label, events in no_work_bi_cases:
        valid, failures, _warnings, _counts = (
            synthetic_margin_row_backward_interm_result(
                joint_backward_interm_arm, events
            )
        )
        no_work_bi_results.append((label, valid, failures))
    check("non-live and non-binding backward-intermediate receipts earn no credit",
          all(
              not valid
              and any(
                  failure["check"]
                  == "margin_row_backward_interm_parameter_engaged"
                  for failure in failures
              )
              for _label, valid, failures in no_work_bi_results
          ), repr(no_work_bi_results))

    mismatched_bi_events = [
        disabled_margin_row_backward_interm_event(),
        margin_row_backward_interm_event(configured_secs="30"),
        margin_row_backward_interm_event(configured_chunk="1024"),
        margin_row_backward_interm_event(
            configured_chunk_secs="1", effective_chunk_secs="1",
        ),
        margin_row_backward_interm_event(configured_topk="4096"),
    ]
    mismatched_bi_results = [
        synthetic_margin_row_backward_interm_result(
            joint_backward_interm_arm, [event]
        )[:2]
        for event in mismatched_bi_events
    ]
    check("every configured backward-intermediate family field must match its arm",
          all(
              not valid
              and any(
                  failure["check"]
                  == "margin_row_backward_interm_parameter_engaged"
                  for failure in failures
              )
              for valid, failures in mismatched_bi_results
          ), repr(mismatched_bi_results))

    partial_bi_row = _clean_row(joint_backward_interm_arm)
    partial_bi_row["flight"]["events"] = [
        margin_row_backward_interm_event(
            chunk_counterfactual_chunks="0", adaptive_width_changes="0",
            topk_binding_layers="0",
        )
    ]
    mismatched_active_bi_row = _clean_row(joint_backward_interm_arm)
    mismatched_active_bi_row["flight"]["events"] = [
        margin_row_backward_interm_event(configured_secs="30")
    ]
    check("mismatched backward-intermediate configurations cannot be stitched across rows",
          _margin_row_backward_interm_parameter_engagement(
              joint_backward_interm_arm,
              [partial_bi_row, mismatched_active_bi_row],
          ) != set(_MARGIN_ROW_BACKWARD_INTERM_FAMILY))

    missing_bi_field = margin_row_backward_interm_event()
    missing_bi_field["reason"] = missing_bi_field["reason"].replace(
        " topk_binding_layers=2", "",
    )
    extra_bi_field = margin_row_backward_interm_event()
    extra_bi_field["reason"] += " extra=1"
    duplicate_bi_field = margin_row_backward_interm_event()
    duplicate_bi_field["reason"] += " configured_chunk=256"
    double_space_bi = margin_row_backward_interm_event()
    double_space_bi["reason"] = double_space_bi["reason"].replace(
        " phase=", "  phase=",
    )
    tab_bi = margin_row_backward_interm_event()
    tab_bi["reason"] = tab_bi["reason"].replace(" phase=", "\tphase=")
    invalid_bi_events = [
        missing_bi_field,
        extra_bi_field,
        duplicate_bi_field,
        double_space_bi,
        tab_bi,
        margin_row_backward_interm_event(
            schema="ny-cli/margin-row-backward-interm-execution/v999",
        ),
        margin_row_backward_interm_event(status="skipped"),
        margin_row_backward_interm_event(phase="progress"),
        margin_row_backward_interm_event(outcome="unknown"),
        margin_row_backward_interm_event(enabled="1"),
        margin_row_backward_interm_event(policy_mismatch="true"),
        *[
            margin_row_backward_interm_event(configured_secs=token)
            for token in (
                "00", "01", "0.0", "1.0", "1e0", "nan", "inf", "-1",
                "9" * 400,
            )
        ],
        margin_row_backward_interm_event(configured_chunk="01"),
        margin_row_backward_interm_event(configured_chunk="0"),
        margin_row_backward_interm_event(configured_topk="0"),
        margin_row_backward_interm_event(configured_topk=str(1 << 64)),
        disabled_margin_row_backward_interm_event(layers_started="1"),
        disabled_margin_row_backward_interm_event(
            eligible_layers="1", suppressed_setup_layers="2",
        ),
        disabled_margin_row_backward_interm_event(configured_secs="0"),
        disabled_margin_row_backward_interm_event(deadline_capped="true"),
        disabled_margin_row_backward_interm_event(budget_exhausted="true"),
        margin_row_backward_interm_event(outcome="disabled"),
        margin_row_backward_interm_event(suppressed_setup_layers="1"),
        margin_row_backward_interm_event(
            outcome="completed", budget_exhausted="true",
        ),
        margin_row_backward_interm_event(
            outcome="exhausted", budget_exhausted="false",
        ),
        margin_row_backward_interm_event(layers_started="5"),
        margin_row_backward_interm_event(chunk_counterfactual_chunks="7"),
        margin_row_backward_interm_event(chunk_ceiling_binding_chunks="7"),
        margin_row_backward_interm_event(fixed_width_chunks="7"),
        margin_row_backward_interm_event(adaptive_width_changes="7"),
        margin_row_backward_interm_event(topk_binding_layers="4"),
        margin_row_backward_interm_event(
            layers_started="0", chunks_completed="1", columns_completed="1",
            topk_binding_layers="0",
        ),
        margin_row_backward_interm_event(
            chunks_completed="6", columns_completed="5",
        ),
        margin_row_backward_interm_event(
            chunks_completed="0", columns_completed="1",
            chunk_counterfactual_chunks="0", adaptive_width_changes="0",
        ),
        margin_row_backward_interm_event(
            configured_chunk_secs="0", effective_chunk_secs="0",
            adaptive_width_changes="1",
        ),
        margin_row_backward_interm_event(
            configured_chunk_secs="2", effective_chunk_secs="1",
            chunk_secs_capped="false",
        ),
        margin_row_backward_interm_event(
            configured_chunk_secs="2", effective_chunk_secs="2",
            chunk_secs_capped="true",
        ),
        margin_row_backward_interm_event(
            configured_chunk_secs="0", effective_chunk_secs="1",
        ),
        margin_row_backward_interm_event(
            configured_chunk="256", chunk_counterfactual_chunks="1",
        ),
        margin_row_backward_interm_event(fixed_width_chunks="1"),
    ]
    check("backward-intermediate v1 rejects malformed and impossible evidence",
          all(not margin_row_backward_interm_contract_accepts(event)
              for event in invalid_bi_events))

    poisoned_bi_valid, poisoned_bi_failures, _warnings, _counts = (
        synthetic_margin_row_backward_interm_result(
            joint_backward_interm_arm,
            [margin_row_backward_interm_event(), duplicate_bi_field],
        )
    )
    check("a malformed same-method backward-intermediate event poisons its row",
          not poisoned_bi_valid
          and any(
              failure["check"]
              == "margin_row_backward_interm_parameter_engaged"
              for failure in poisoned_bi_failures
          ), repr(poisoned_bi_failures))

    mixed_policy_bi_row = {
        "flight": {"events": [
            margin_row_backward_interm_event(configured_secs="20"),
            margin_row_backward_interm_event(configured_secs="30"),
        ]}
    }
    check("contradictory valid backward-intermediate terminals poison a row",
          _margin_row_backward_interm_terminal_evidences(mixed_policy_bi_row) == [])

    def margin_row_root_precision_event(status="ran", **overrides):
        fields = {
            "schema": _MARGIN_ROW_ROOT_PRECISION_EVENT_SCHEMA,
            "phase": "terminal",
            "outcome": "completed",
            "configured_root_f32": "true",
            "configured_tableau_f32": "true",
            "outward_eligible": "true",
            "eligible_conv_ops": "4",
            "root_f32_conv_ops": "4",
            "root_f64_conv_ops": "0",
            "eligible_store_ops": "9",
            "tableau_f32_store_ops": "9",
            "tableau_f64_store_ops": "0",
            "tableau_refused": "false",
            "policy_mismatch": "false",
        }
        fields.update(overrides)
        return machine_event(
            "margin_row_root_precision_execution", status, **fields
        )

    def root_precision_f64_event(**overrides):
        fields = {
            "configured_root_f32": "false",
            "configured_tableau_f32": "false",
            "root_f32_conv_ops": "0",
            "root_f64_conv_ops": "4",
            "tableau_f32_store_ops": "0",
            "tableau_f64_store_ops": "9",
        }
        fields.update(overrides)
        return margin_row_root_precision_event(**fields)

    def tableau_f64_event(**overrides):
        fields = {
            "configured_tableau_f32": "false",
            "tableau_f32_store_ops": "0",
            "tableau_f64_store_ops": "9",
        }
        fields.update(overrides)
        return margin_row_root_precision_event(**fields)

    def synthetic_margin_row_root_precision_result(
        arm_pairs, events, category="cifar100"
    ):
        synthetic_row = _clean_row(arm_pairs, category=category)
        synthetic_row["flight"]["events"] = list(events)
        return validate_run(
            arm_pairs,
            2025,
            category,
            [synthetic_row],
            _clean_manifest(category=category, arm=arm_pairs),
            _summary_of([synthetic_row]),
        )

    def root_precision_contract_accepts(event):
        return len(_margin_row_root_precision_terminal_evidences({
            "flight": {"events": [event]},
        })) == 1

    joint_precision_arm = [
        ("NY_MARGIN_ROW_ROOT_F32", "1"),
        ("NY_MARGIN_ROW_TABLEAU_F32", "1"),
    ]
    precision_edges = [
        edge
        for edge in space.edges
        if edge.child == "NY_MARGIN_ROW_TABLEAU_F32"
    ]
    repaired_tableau, tableau_contrast = _repair_preserving(
        {"NY_MARGIN_ROW_TABLEAU_F32": "0"},
        space,
        kinds,
        baseline_values={
            "NY_MARGIN_ROW_ROOT_F32": "1",
            "NY_MARGIN_ROW_TABLEAU_F32": "1",
        },
        return_contrast=True,
    )
    check("tableau precision treatments carry an explicit root-f32 prerequisite",
          len(precision_edges) == 1
          and precision_edges[0].kind == "armed"
          and precision_edges[0].parent == "NY_MARGIN_ROW_ROOT_F32"
          and dict(repaired_tableau or []).get("NY_MARGIN_ROW_ROOT_F32") == "1"
          and tableau_contrast is not None
          and tableau_contrast["prerequisites_added"]
          == [["NY_MARGIN_ROW_ROOT_F32", "1"]]
          and _repair_preserving(
              {
                  "NY_MARGIN_ROW_ROOT_F32": "0",
                  "NY_MARGIN_ROW_TABLEAU_F32": "1",
              },
              space,
              kinds,
          ) is None,
          repr((precision_edges, repaired_tableau, tableau_contrast)))
    joint_precision_valid, joint_precision_failures, _warnings, _counts = (
        synthetic_margin_row_root_precision_result(
            joint_precision_arm, [margin_row_root_precision_event()]
        )
    )
    root_off_valid, root_off_failures, _warnings, _counts = (
        synthetic_margin_row_root_precision_result(
            [("NY_MARGIN_ROW_ROOT_F32", "0")],
            [root_precision_f64_event()],
        )
    )
    tableau_off_valid, tableau_off_failures, _warnings, _counts = (
        synthetic_margin_row_root_precision_result(
            [("NY_MARGIN_ROW_TABLEAU_F32", "0")],
            [tableau_f64_event()],
        )
    )
    check("root precision v1 authenticates both live Boolean counterfactuals",
          joint_precision_valid and root_off_valid and tableau_off_valid,
          repr((joint_precision_failures, root_off_failures,
                tableau_off_failures)))

    refused_tableau = root_precision_f64_event(
        configured_tableau_f32="true", tableau_refused="true"
    )
    refused_valid, refused_failures, _warnings, _counts = (
        synthetic_margin_row_root_precision_result(
            [("NY_MARGIN_ROW_TABLEAU_F32", "1")], [refused_tableau]
        )
    )
    zero_work = margin_row_root_precision_event(
        eligible_conv_ops="0", root_f32_conv_ops="0",
        eligible_store_ops="0", tableau_f32_store_ops="0",
    )
    zero_work_valid, zero_work_failures, _warnings, _counts = (
        synthetic_margin_row_root_precision_result(
            joint_precision_arm, [zero_work]
        )
    )
    inward = margin_row_root_precision_event(
        configured_tableau_f32="false", outward_eligible="false",
        eligible_conv_ops="0", root_f32_conv_ops="0",
        eligible_store_ops="0", tableau_f32_store_ops="0",
        tableau_refused="false",
    )
    inward_valid, inward_failures, _warnings, _counts = (
        synthetic_margin_row_root_precision_result(
            [("NY_MARGIN_ROW_ROOT_F32", "1")], [inward]
        )
    )
    check("refused, zero-work, and inward precision paths earn no T1 credit",
          not refused_valid and not zero_work_valid and not inward_valid
          and all(
              any(
                  failure["check"]
                  == "margin_row_root_precision_parameter_engaged"
                  for failure in failures
              )
              for failures in (refused_failures, zero_work_failures,
                               inward_failures)
          ), repr((refused_failures, zero_work_failures, inward_failures)))

    missing_precision_field = margin_row_root_precision_event()
    missing_precision_field["reason"] = missing_precision_field["reason"].replace(
        " eligible_conv_ops=4", "", 1
    )
    duplicate_precision_field = margin_row_root_precision_event()
    duplicate_precision_field["reason"] += " eligible_conv_ops=4"
    invalid_precision_events = [
        missing_precision_field,
        duplicate_precision_field,
        margin_row_root_precision_event(
            schema="ny-cli/margin-row-root-precision-execution/v999"
        ),
        margin_row_root_precision_event(status="skipped"),
        margin_row_root_precision_event(phase="progress"),
        margin_row_root_precision_event(outcome="unknown"),
        margin_row_root_precision_event(configured_root_f32="1"),
        margin_row_root_precision_event(policy_mismatch="true"),
        margin_row_root_precision_event(eligible_conv_ops="04"),
        margin_row_root_precision_event(root_f64_conv_ops="1"),
        margin_row_root_precision_event(tableau_f64_store_ops="1"),
        margin_row_root_precision_event(tableau_refused="true"),
        root_precision_f64_event(
            configured_tableau_f32="true", tableau_refused="false"
        ),
        margin_row_root_precision_event(
            outward_eligible="false", tableau_refused="true"
        ),
    ]
    check("root precision v1 rejects malformed and impossible evidence",
          all(not root_precision_contract_accepts(event)
              for event in invalid_precision_events))

    poisoned_precision_valid, poisoned_precision_failures, _warnings, _counts = (
        synthetic_margin_row_root_precision_result(
            joint_precision_arm,
            [margin_row_root_precision_event(), duplicate_precision_field],
        )
    )
    mixed_precision_row = {
        "flight": {"events": [
            margin_row_root_precision_event(),
            tableau_f64_event(),
        ]}
    }
    check("malformed or mixed-policy precision terminals poison one row",
          not poisoned_precision_valid
          and any(
              failure["check"]
              == "margin_row_root_precision_parameter_engaged"
              for failure in poisoned_precision_failures
          )
          and _margin_row_root_precision_terminal_evidences(
              mixed_precision_row
          ) == [], repr(poisoned_precision_failures))

    def margin_row_root_alpha_event(status="ran", **overrides):
        bits = lambda value: str(struct.unpack(
            ">Q", struct.pack(">d", value)
        )[0])
        fields = {
            "schema": _MARGIN_ROW_ROOT_ALPHA_EVENT_SCHEMA,
            "phase": "terminal",
            "outcome": "unknown",
            "alpha_enabled": "true",
            "alpha_iters": "4",
            "alpha_secs_bits": bits(20.0),
            "alpha_lr_bits": bits(0.1),
            "alpha_cols": "4",
            "alpha_guard": "true",
            "rebake_enabled": "true",
            "rebake_rounds": "2",
            "rebake_secs_bits": bits(30.0),
            "eligible": "true",
            "standalone_suppressed": "0",
            "standalone_attempted": "1",
            "standalone_completed": "1",
            "standalone_effective_secs_bits": bits(20.0),
            "standalone_deadline_capped": "false",
            "standalone_budget_exhausted": "false",
            "standalone_iters_cap_exhausted": "true",
            "standalone_fail_classes": "9",
            "standalone_iters_attempted": "4",
            "standalone_candidate_steps": "4",
            "standalone_legacy_steps": "0",
            "standalone_adam_steps": "4",
            "standalone_passes": "5",
            "standalone_max_steer_cols": "4",
            "standalone_frozen_improvements": "1",
            "standalone_accepted": "true",
            "standalone_moved_neurons": "5",
            "standalone_guard_opportunities": "1",
            "standalone_guard_baseline_evals": "1",
            "standalone_guard_candidate_evals": "1",
            "standalone_guard_reversions": "0",
            "standalone_frozen_before_bits": bits(-1.0),
            "standalone_frozen_best_bits": bits(-0.9),
            "standalone_published_before_bits": bits(-1.0),
            "standalone_published_after_bits": bits(-0.8),
            "rebake_suppressed": "0",
            "rebake_attempted": "1",
            "rebake_completed": "1",
            "rebake_effective_secs_bits": bits(30.0),
            "rebake_deadline_capped": "false",
            "rebake_budget_exhausted": "false",
            "rebake_round_cap_exhausted": "true",
            "rebake_entry_unbaked_alpha": "true",
            "rebake_baseline_evals": "1",
            "rebake_rounds_started": "2",
            "rebake_alpha_completed": "2",
            "rebake_alpha_iters_attempted": "8",
            "rebake_alpha_candidate_steps": "8",
            "rebake_alpha_legacy_steps": "0",
            "rebake_alpha_adam_steps": "8",
            "rebake_alpha_passes": "10",
            "rebake_alpha_max_fail_classes": "9",
            "rebake_alpha_max_steer_cols": "4",
            "rebake_alpha_moved_neurons": "10",
            "rebake_rebuilds_completed": "2",
            "rebake_rounds_scored": "2",
            "rebake_rounds_accepted": "2",
            "rebake_fixed_point_stops": "0",
            "rebake_affordability_refusals": "0",
            "rebake_aborts": "0",
            "rebake_published_before_bits": bits(-1.0),
            "rebake_published_after_bits": bits(-0.5),
            "policy_mismatch": "false",
        }
        fields.update(overrides)
        return machine_event(
            "margin_row_root_alpha_execution", status, **fields
        )

    def alpha_budget_event(**overrides):
        fields = {
            "standalone_budget_exhausted": "true",
            "standalone_iters_cap_exhausted": "false",
            "standalone_iters_attempted": "2",
            "standalone_candidate_steps": "2",
            "standalone_adam_steps": "2",
            "standalone_passes": "3",
        }
        fields.update(overrides)
        return margin_row_root_alpha_event(**fields)

    def rebake_budget_event(**overrides):
        fields = {
            "rebake_budget_exhausted": "true",
            "rebake_round_cap_exhausted": "false",
            "rebake_rounds_started": "1",
            "rebake_alpha_completed": "1",
            "rebake_alpha_iters_attempted": "4",
            "rebake_alpha_candidate_steps": "4",
            "rebake_alpha_adam_steps": "4",
            "rebake_alpha_passes": "5",
            "rebake_alpha_moved_neurons": "5",
            "rebake_rebuilds_completed": "1",
            "rebake_rounds_scored": "1",
            "rebake_rounds_accepted": "1",
            "rebake_affordability_refusals": "1",
        }
        fields.update(overrides)
        return margin_row_root_alpha_event(**fields)

    def disabled_root_alpha_event(**overrides):
        fields = {
            "alpha_enabled": "false",
            "rebake_enabled": "false",
            "standalone_suppressed": "1",
            "standalone_attempted": "0",
            "standalone_completed": "0",
            "standalone_effective_secs_bits": "0",
            "standalone_iters_cap_exhausted": "false",
            "standalone_fail_classes": "0",
            "standalone_iters_attempted": "0",
            "standalone_candidate_steps": "0",
            "standalone_adam_steps": "0",
            "standalone_passes": "0",
            "standalone_max_steer_cols": "0",
            "standalone_frozen_improvements": "0",
            "standalone_accepted": "false",
            "standalone_moved_neurons": "0",
            "standalone_guard_opportunities": "0",
            "standalone_guard_baseline_evals": "0",
            "standalone_guard_candidate_evals": "0",
            "standalone_frozen_before_bits": "none",
            "standalone_frozen_best_bits": "none",
            "standalone_published_before_bits": "none",
            "standalone_published_after_bits": "none",
            "rebake_suppressed": "1",
            "rebake_attempted": "0",
            "rebake_completed": "0",
            "rebake_effective_secs_bits": "0",
            "rebake_round_cap_exhausted": "false",
            "rebake_entry_unbaked_alpha": "false",
            "rebake_baseline_evals": "0",
            "rebake_rounds_started": "0",
            "rebake_alpha_completed": "0",
            "rebake_alpha_iters_attempted": "0",
            "rebake_alpha_candidate_steps": "0",
            "rebake_alpha_adam_steps": "0",
            "rebake_alpha_passes": "0",
            "rebake_alpha_max_fail_classes": "0",
            "rebake_alpha_max_steer_cols": "0",
            "rebake_alpha_moved_neurons": "0",
            "rebake_rebuilds_completed": "0",
            "rebake_rounds_scored": "0",
            "rebake_rounds_accepted": "0",
            "rebake_published_before_bits": "none",
            "rebake_published_after_bits": "none",
        }
        fields.update(overrides)
        return margin_row_root_alpha_event(**fields)

    def synthetic_margin_row_root_alpha_result(arm_pairs, events):
        synthetic_rows = []
        for event in events:
            synthetic_row = _clean_row(arm_pairs, category="cifar100")
            synthetic_row["flight"]["events"] = [event]
            synthetic_rows.append(synthetic_row)
        return validate_run(
            arm_pairs,
            2025,
            "cifar100",
            synthetic_rows,
            _clean_manifest(category="cifar100", arm=arm_pairs),
            _summary_of(synthetic_rows),
        )

    def root_alpha_contract_accepts(event):
        return len(_margin_row_root_alpha_terminal_evidences({
            "flight": {"events": [event]},
        })) == 1

    joint_root_alpha_arm = [
        ("NY_MARGIN_ROW_ALPHA_OPT", "1"),
        ("NY_MARGIN_ROW_ALPHA_ITERS", "4"),
        ("NY_MARGIN_ROW_ALPHA_SECS", "20"),
        ("NY_MARGIN_ROW_ALPHA_LR", "0.1"),
        ("NY_MARGIN_ROW_ALPHA_COLS", "4"),
        ("NY_MARGIN_ROW_ALPHA_GUARD", "1"),
        ("NY_MARGIN_ROW_REBAKE", "1"),
        ("NY_MARGIN_ROW_REBAKE_ROUNDS", "2"),
        ("NY_MARGIN_ROW_REBAKE_SECS", "30"),
    ]
    joint_alpha_valid, joint_alpha_failures, _warnings, _counts = (
        synthetic_margin_row_root_alpha_result(
            joint_root_alpha_arm,
            [
                margin_row_root_alpha_event(),
                alpha_budget_event(),
                rebake_budget_event(),
            ],
        )
    )
    off_alpha_valid, off_alpha_failures, _warnings, _counts = (
        synthetic_margin_row_root_alpha_result(
            [
                ("NY_MARGIN_ROW_ALPHA_OPT", "0"),
                ("NY_MARGIN_ROW_REBAKE", "0"),
            ],
            [disabled_root_alpha_event()],
        )
    )
    check("root-alpha route-split controls cannot borrow engagement across rows",
          not joint_alpha_valid and off_alpha_valid
          and any(
              failure["check"] == "margin_row_root_alpha_parameter_engaged"
              and all(name in failure["detail"] for name in (
                  "NY_MARGIN_ROW_ALPHA_ITERS",
                  "NY_MARGIN_ROW_ALPHA_SECS",
                  "NY_MARGIN_ROW_REBAKE_ROUNDS",
                  "NY_MARGIN_ROW_REBAKE_SECS",
              ))
              for failure in joint_alpha_failures
          ),
          repr((joint_alpha_failures, off_alpha_failures)))

    no_candidate = margin_row_root_alpha_event(
        standalone_iters_cap_exhausted="false",
        standalone_iters_attempted="1",
        standalone_candidate_steps="0",
        standalone_adam_steps="0",
        standalone_passes="1",
        standalone_frozen_improvements="0",
        standalone_accepted="false",
        standalone_moved_neurons="0",
        standalone_guard_opportunities="0",
        standalone_guard_candidate_evals="0",
        standalone_frozen_best_bits=str(struct.unpack(
            ">Q", struct.pack(">d", -1.0)
        )[0]),
        standalone_published_after_bits="none",
        rebake_round_cap_exhausted="false",
        rebake_entry_unbaked_alpha="false",
        rebake_rounds_started="1",
        rebake_alpha_completed="1",
        rebake_alpha_iters_attempted="0",
        rebake_alpha_candidate_steps="0",
        rebake_alpha_adam_steps="0",
        rebake_alpha_passes="1",
        rebake_alpha_moved_neurons="0",
        rebake_rebuilds_completed="0",
        rebake_rounds_scored="0",
        rebake_rounds_accepted="0",
        rebake_fixed_point_stops="1",
        rebake_published_after_bits=str(struct.unpack(
            ">Q", struct.pack(">d", -1.0)
        )[0]),
    )
    saturated_cols = margin_row_root_alpha_event(
        standalone_fail_classes="4",
        rebake_alpha_max_fail_classes="4",
    )
    deadline_capped_secs = alpha_budget_event(
        standalone_effective_secs_bits=str(struct.unpack(
            ">Q", struct.pack(">d", 10.0)
        )[0]),
        standalone_deadline_capped="true",
    )
    uncapped_iters = alpha_budget_event(
        standalone_budget_exhausted="false"
    )
    guard_eval_failed = margin_row_root_alpha_event(
        standalone_accepted="false",
        standalone_moved_neurons="0",
        standalone_guard_reversions="1",
        standalone_published_after_bits="none",
        rebake_entry_unbaked_alpha="false",
    )
    fixed_rebake = margin_row_root_alpha_event(
        standalone_frozen_improvements="0",
        standalone_accepted="false",
        standalone_moved_neurons="0",
        standalone_guard_opportunities="0",
        standalone_guard_candidate_evals="0",
        standalone_frozen_best_bits=str(struct.unpack(
            ">Q", struct.pack(">d", -1.0)
        )[0]),
        standalone_published_after_bits="none",
        rebake_round_cap_exhausted="false",
        rebake_entry_unbaked_alpha="false",
        rebake_rounds_started="1",
        rebake_alpha_completed="1",
        rebake_alpha_iters_attempted="1",
        rebake_alpha_candidate_steps="1",
        rebake_alpha_adam_steps="1",
        rebake_alpha_passes="2",
        rebake_alpha_moved_neurons="0",
        rebake_rebuilds_completed="0",
        rebake_rounds_scored="0",
        rebake_rounds_accepted="0",
        rebake_fixed_point_stops="1",
        rebake_published_after_bits=str(struct.unpack(
            ">Q", struct.pack(">d", -1.0)
        )[0]),
    )
    rejected_rebake = margin_row_root_alpha_event(
        rebake_round_cap_exhausted="false",
        rebake_rounds_started="1",
        rebake_alpha_completed="1",
        rebake_alpha_iters_attempted="4",
        rebake_alpha_candidate_steps="4",
        rebake_alpha_adam_steps="4",
        rebake_alpha_passes="5",
        rebake_alpha_moved_neurons="5",
        rebake_rebuilds_completed="1",
        rebake_rounds_scored="1",
        rebake_rounds_accepted="0",
        rebake_published_after_bits=str(struct.unpack(
            ">Q", struct.pack(">d", -1.0)
        )[0]),
    )
    adversarial_alpha_cases = [
        ([("NY_MARGIN_ROW_ALPHA_LR", "0.1")], no_candidate,
         "NY_MARGIN_ROW_ALPHA_LR"),
        ([("NY_MARGIN_ROW_ALPHA_COLS", "4")], saturated_cols,
         "NY_MARGIN_ROW_ALPHA_COLS"),
        ([("NY_MARGIN_ROW_ALPHA_SECS", "20")], deadline_capped_secs,
         "NY_MARGIN_ROW_ALPHA_SECS"),
        ([("NY_MARGIN_ROW_ALPHA_ITERS", "4")], uncapped_iters,
         "NY_MARGIN_ROW_ALPHA_ITERS"),
        ([("NY_MARGIN_ROW_ALPHA_GUARD", "1")], guard_eval_failed,
         "NY_MARGIN_ROW_ALPHA_GUARD"),
        ([("NY_MARGIN_ROW_REBAKE", "1")], fixed_rebake,
         "NY_MARGIN_ROW_REBAKE"),
        ([("NY_MARGIN_ROW_REBAKE", "1")], rejected_rebake,
         "NY_MARGIN_ROW_REBAKE"),
        ([("NY_MARGIN_ROW_REBAKE_ROUNDS", "2")], fixed_rebake,
         "NY_MARGIN_ROW_REBAKE_ROUNDS"),
    ]
    check("failed, saturated, capped, and fixed routes earn no alpha credit",
          all(
              root_alpha_contract_accepts(event)
              and name not in _margin_row_root_alpha_parameter_engagement(
                  wanted, [{"flight": {"events": [event]}}]
              )
              for wanted, event, name in adversarial_alpha_cases
          ))

    guard_off = margin_row_root_alpha_event(
        alpha_guard="false",
        standalone_guard_baseline_evals="0",
        standalone_guard_candidate_evals="0",
        standalone_published_before_bits="none",
        standalone_published_after_bits="none",
    )
    check("both guard routes require an actual frozen improvement opportunity",
          "NY_MARGIN_ROW_ALPHA_GUARD"
          in _margin_row_root_alpha_parameter_engagement(
              [("NY_MARGIN_ROW_ALPHA_GUARD", "0")],
              [{"flight": {"events": [guard_off]}}],
          )
          and "NY_MARGIN_ROW_ALPHA_GUARD"
          in _margin_row_root_alpha_parameter_engagement(
              [("NY_MARGIN_ROW_ALPHA_GUARD", "1")],
              [{"flight": {"events": [margin_row_root_alpha_event()]}}],
          ))

    missing_alpha_field = margin_row_root_alpha_event()
    missing_alpha_field["reason"] = missing_alpha_field["reason"].replace(
        " standalone_candidate_steps=4", "", 1
    )
    duplicate_alpha_field = margin_row_root_alpha_event()
    duplicate_alpha_field["reason"] += " standalone_candidate_steps=4"
    extra_alpha_field = margin_row_root_alpha_event()
    extra_alpha_field["reason"] += " invented=1"
    whitespace_alpha_field = margin_row_root_alpha_event()
    whitespace_alpha_field["reason"] = whitespace_alpha_field["reason"].replace(
        " alpha_iters=4", "  alpha_iters=4", 1
    )
    invalid_alpha_events = [
        missing_alpha_field,
        duplicate_alpha_field,
        extra_alpha_field,
        whitespace_alpha_field,
        margin_row_root_alpha_event(status="skipped"),
        margin_row_root_alpha_event(phase="progress"),
        margin_row_root_alpha_event(policy_mismatch="true"),
        margin_row_root_alpha_event(alpha_iters="04"),
        margin_row_root_alpha_event(standalone_passes="4"),
        margin_row_root_alpha_event(
            standalone_iters_cap_exhausted="false",
            standalone_iters_attempted="4",
            standalone_candidate_steps="1",
            standalone_adam_steps="1",
            standalone_passes="2",
        ),
        alpha_budget_event(standalone_iters_attempted="3"),
        margin_row_root_alpha_event(rebake_alpha_passes="9"),
        margin_row_root_alpha_event(rebake_alpha_max_steer_cols="3"),
        rebake_budget_event(
            rebake_budget_exhausted="false",
            rebake_affordability_refusals="0",
        ),
    ]
    check("root-alpha v1 rejects malformed and impossible evidence",
          all(not root_alpha_contract_accepts(event)
              for event in invalid_alpha_events))

    mixed_alpha_row = {
        "flight": {"events": [
            margin_row_root_alpha_event(),
            margin_row_root_alpha_event(alpha_iters="8"),
        ]}
    }
    poisoned_alpha_arm = [("NY_MARGIN_ROW_ALPHA_OPT", "1")]
    poisoned_alpha_row = _clean_row(
        poisoned_alpha_arm, category="cifar100"
    )
    poisoned_alpha_row["flight"]["events"] = [
        margin_row_root_alpha_event(), duplicate_alpha_field
    ]
    poisoned_alpha_valid, poisoned_alpha_failures, _warnings, _counts = (
        validate_run(
            poisoned_alpha_arm,
            2025,
            "cifar100",
            [poisoned_alpha_row],
            _clean_manifest(category="cifar100", arm=poisoned_alpha_arm),
            _summary_of([poisoned_alpha_row]),
        )
    )
    duplicate_alpha_row = {
        "flight": {"events": [
            margin_row_root_alpha_event(),
            margin_row_root_alpha_event(),
        ]}
    }
    check("malformed, duplicate, or mixed-policy root-alpha terminals poison one row",
          _margin_row_root_alpha_terminal_evidences(mixed_alpha_row) == []
          and _margin_row_root_alpha_terminal_evidences(
              duplicate_alpha_row
          ) == []
          and not poisoned_alpha_valid
          and any(
              failure["check"]
              == "margin_row_root_alpha_parameter_engaged"
              for failure in poisoned_alpha_failures
          ), repr(poisoned_alpha_failures))

    def margin_row_domain_alpha_event(status="ran", **overrides):
        fields = {
            "schema": _MARGIN_ROW_DOMAIN_ALPHA_EVENT_SCHEMA,
            "phase": "terminal",
            "route": "joint",
            "outcome": "unknown",
            "configured_enabled": "true",
            "configured_iters": "4",
            "configured_eta_bits": str(
                struct.unpack(">Q", struct.pack(">d", 0.5))[0]
            ),
            "policy_mismatch": "false",
            "gate_evals": "3",
            "movable_domains": "2",
            "movable_slots": "5",
            "warm_entries_applied": "2",
            "ascent_domains": "2",
            "outer_ascent_domains": "1",
            "proposals": "4",
            "trial_passes_completed": "3",
            "trial_pass_failures": "1",
            "finite_trial_scores": "2",
            "accepted_trials": "1",
            "carried_entries": "2",
            "iters_counterfactual_domains": "1",
            "eta_counterfactual_scored_domains": "1",
            "outer_finite_trial_scores": "2",
            "outer_iters_counterfactual_domains": "1",
            "outer_eta_counterfactual_scored_domains": "1",
        }
        fields.update(overrides)
        return machine_event(
            "margin_row_domain_alpha_execution", status, **fields
        )

    domain_alpha_arm = [
        ("NY_MARGIN_ROW_DOMAIN_ALPHA", "1"),
        ("NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS", "4"),
        ("NY_MARGIN_ROW_DOMAIN_ALPHA_ETA", "0.5"),
    ]
    domain_alpha_row = {
        "flight": {"events": [margin_row_domain_alpha_event()]}
    }
    check(
        "domain-alpha v1 proves all three axes only from outer joint scoring",
        _margin_row_domain_alpha_parameter_engagement(
            domain_alpha_arm, [domain_alpha_row]
        ) == {name for name, _token in domain_alpha_arm},
    )
    nested_domain_alpha = {
        "flight": {"events": [margin_row_domain_alpha_event(
            outer_ascent_domains="0",
            outer_finite_trial_scores="0",
            outer_iters_counterfactual_domains="0",
            outer_eta_counterfactual_scored_domains="0",
        )]}
    }
    classwise_domain_alpha = {
        "flight": {"events": [margin_row_domain_alpha_event(
            route="classwise"
        )]}
    }
    check(
        "nested-only and classwise domain-alpha work earn no T1 credit",
        not _margin_row_domain_alpha_parameter_engagement(
            domain_alpha_arm, [nested_domain_alpha, classwise_domain_alpha]
        ),
    )
    malformed_domain_alpha = margin_row_domain_alpha_event(proposals="5")
    repeated_failure_domain_alpha = margin_row_domain_alpha_event(
        gate_evals="1",
        movable_domains="1",
        movable_slots="1",
        warm_entries_applied="0",
        ascent_domains="1",
        outer_ascent_domains="1",
        proposals="4",
        trial_passes_completed="2",
        trial_pass_failures="2",
        finite_trial_scores="1",
        accepted_trials="0",
        carried_entries="0",
        iters_counterfactual_domains="0",
        eta_counterfactual_scored_domains="0",
        outer_finite_trial_scores="1",
        outer_iters_counterfactual_domains="0",
        outer_eta_counterfactual_scored_domains="0",
    )
    short_high_cap_domain_alpha = margin_row_domain_alpha_event(
        proposals="2",
        trial_passes_completed="2",
        trial_pass_failures="0",
    )
    misattributed_high_cap_domain_alpha = margin_row_domain_alpha_event(
        outer_finite_trial_scores="1",
    )
    duplicate_domain_alpha = {
        "flight": {"events": [
            margin_row_domain_alpha_event(),
            margin_row_domain_alpha_event(),
        ]}
    }
    mismatched_domain_alpha = margin_row_domain_alpha_event(
        policy_mismatch="true"
    )
    check(
        "domain-alpha v1 rejects impossible, duplicate, and mismatched terminals",
        not _margin_row_domain_alpha_terminal_evidences({
            "flight": {"events": [malformed_domain_alpha]}
        })
        and not _margin_row_domain_alpha_terminal_evidences({
            "flight": {"events": [repeated_failure_domain_alpha]}
        })
        and not _margin_row_domain_alpha_terminal_evidences({
            "flight": {"events": [short_high_cap_domain_alpha]}
        })
        and not _margin_row_domain_alpha_terminal_evidences({
            "flight": {"events": [misattributed_high_cap_domain_alpha]}
        })
        and not _margin_row_domain_alpha_terminal_evidences(
            duplicate_domain_alpha
        )
        and not _margin_row_domain_alpha_terminal_evidences({
            "flight": {"events": [mismatched_domain_alpha]}
        }),
    )

    def pgd_fast_gradient_event(status="ran", **overrides):
        fields = {
            "schema": _PGD_FAST_GRADIENT_EVENT_SCHEMA,
            "phase": "terminal",
            "route": "conjunctive",
            "configured": "false",
            "decisions": "2",
            "fast_attempts": "0",
            "fast_selected": "0",
            "certified_selected": "1",
            "spsa_selected": "1",
        }
        fields.update(overrides)
        return machine_event("pgd_fast_gradient_execution", status, **fields)

    def synthetic_pgd_fast_gradient_result(arm_pairs, events, category="cifar100"):
        synthetic_row = _clean_row(arm_pairs, category=category)
        synthetic_row["flight"]["events"] = list(events)
        return validate_run(
            arm_pairs,
            2025,
            category,
            [synthetic_row],
            _clean_manifest(category=category, arm=arm_pairs),
            _summary_of([synthetic_row]),
        )

    def pgd_fast_gradient_contract_accepts(event):
        synthetic_row = _clean_row([])
        synthetic_row["flight"]["events"] = [event]
        return len(_pgd_fast_gradient_terminal_evidences(synthetic_row)) == 1

    fast_disabled_valid, fast_disabled_failures, _warnings, _counts = (
        synthetic_pgd_fast_gradient_result(
            [("NY_PGD_FAST_GRAD", "0")],
            [pgd_fast_gradient_event(route="disjunctive")],
        )
    )
    fast_enabled_valid, fast_enabled_failures, _warnings, _counts = (
        synthetic_pgd_fast_gradient_result(
            [("NY_PGD_FAST_GRAD", "1")],
            [pgd_fast_gradient_event(
                configured="true", decisions="3", fast_attempts="3",
                fast_selected="2", certified_selected="1", spsa_selected="0",
            )],
        )
    )
    check("both fast-gradient Boolean policies require and accept live decisions",
          fast_disabled_valid and fast_enabled_valid,
          repr((fast_disabled_failures, fast_enabled_failures)))

    skipped_fast_gradient = pgd_fast_gradient_event(
        status="skipped", decisions="0", fast_attempts="0", fast_selected="0",
        certified_selected="0", spsa_selected="0",
    )
    no_credit_fast_gradient_cases = [
        ("no receipt", []),
        ("wide VJP bypass", [machine_event(
            "exact_vjp_execution", "ran", schema="unrelated",
        )]),
        ("zero decisions", [skipped_fast_gradient]),
        ("mismatched policy", [pgd_fast_gradient_event(configured="true",
            fast_attempts="2", fast_selected="1", spsa_selected="0")]),
    ]
    no_credit_fast_gradient_results = []
    for label, events in no_credit_fast_gradient_cases:
        valid, failures, _warnings, _counts = synthetic_pgd_fast_gradient_result(
            [("NY_PGD_FAST_GRAD", "0")], events,
        )
        no_credit_fast_gradient_results.append((label, valid, failures))
    check("bypassed, skipped, and mismatched fast-gradient receipts earn no credit",
          all(
              not valid
              and any(failure["check"] == "pgd_fast_gradient_parameter_engaged"
                      for failure in failures)
              for _label, valid, failures in no_credit_fast_gradient_results
          ), repr(no_credit_fast_gradient_results))

    matching_fast_row = _clean_row([("NY_PGD_FAST_GRAD", "0")])
    matching_fast_row["flight"]["events"] = [skipped_fast_gradient]
    differing_fast_row = _clean_row([("NY_PGD_FAST_GRAD", "0")])
    differing_fast_row["flight"]["events"] = [pgd_fast_gradient_event(
        configured="true", fast_attempts="2", fast_selected="1", spsa_selected="0",
    )]
    check("fast-gradient policy matching and live execution cannot cross rows",
          not _pgd_fast_gradient_parameter_engagement(
              [("NY_PGD_FAST_GRAD", "0")],
              [matching_fast_row, differing_fast_row],
          ))

    missing_fast_field = pgd_fast_gradient_event()
    missing_fast_field["reason"] = missing_fast_field["reason"].replace(
        " spsa_selected=1", "",
    )
    extra_fast_field = pgd_fast_gradient_event()
    extra_fast_field["reason"] += " extra=1"
    duplicate_fast_field = pgd_fast_gradient_event()
    duplicate_fast_field["reason"] += " decisions=2"
    double_space_fast = pgd_fast_gradient_event()
    double_space_fast["reason"] = double_space_fast["reason"].replace(
        " phase=", "  phase=",
    )
    tab_fast = pgd_fast_gradient_event()
    tab_fast["reason"] = tab_fast["reason"].replace(" phase=", "\tphase=")
    invalid_fast_gradient_events = [
        missing_fast_field,
        extra_fast_field,
        duplicate_fast_field,
        double_space_fast,
        tab_fast,
        pgd_fast_gradient_event(schema="ny-cli/pgd-fast-gradient-execution/v999"),
        pgd_fast_gradient_event(status="skipped"),
        pgd_fast_gradient_event(phase="progress"),
        pgd_fast_gradient_event(route="other"),
        pgd_fast_gradient_event(configured="1"),
        pgd_fast_gradient_event(decisions="02"),
        pgd_fast_gradient_event(decisions=str(1 << 64)),
        pgd_fast_gradient_event(fast_selected="1", certified_selected="1",
                                spsa_selected="1"),
        pgd_fast_gradient_event(fast_selected="1"),
        pgd_fast_gradient_event(configured="true", fast_attempts="1",
                                fast_selected="1"),
        pgd_fast_gradient_event(fast_attempts="1", fast_selected="1",
                                certified_selected="1", spsa_selected="0"),
        pgd_fast_gradient_event(status="ran", decisions="0", fast_attempts="0",
                                fast_selected="0", certified_selected="0",
                                spsa_selected="0"),
        pgd_fast_gradient_event(status="skipped"),
    ]
    check("fast-gradient v1 rejects malformed grammar and impossible counters",
          all(not pgd_fast_gradient_contract_accepts(event)
              for event in invalid_fast_gradient_events))

    earlier_fast_valid, earlier_fast_failures, _warnings, _counts = (
        synthetic_pgd_fast_gradient_result(
            [("NY_PGD_FAST_GRAD", "0")],
            [pgd_fast_gradient_event(), skipped_fast_gradient],
        )
    )
    malformed_later_fast = pgd_fast_gradient_event()
    malformed_later_fast["reason"] += " decisions=2"
    poisoned_fast_valid, poisoned_fast_failures, _warnings, _counts = (
        synthetic_pgd_fast_gradient_result(
            [("NY_PGD_FAST_GRAD", "0")],
            [pgd_fast_gradient_event(), malformed_later_fast],
        )
    )
    check("a later skipped fast-gradient invocation preserves earlier engagement",
          earlier_fast_valid, repr(earlier_fast_failures))
    check("a malformed same-method fast-gradient event poisons that row's evidence",
          not poisoned_fast_valid
          and any(failure["check"] == "pgd_fast_gradient_parameter_engaged"
                  for failure in poisoned_fast_failures),
          repr(poisoned_fast_failures))
    contradictory_fast_valid, contradictory_fast_failures, _warnings, _counts = (
        synthetic_pgd_fast_gradient_result(
            [("NY_PGD_FAST_GRAD", "0")],
            [
                pgd_fast_gradient_event(),
                pgd_fast_gradient_event(
                    configured="true", fast_attempts="2", fast_selected="1",
                    spsa_selected="0",
                ),
            ],
        )
    )
    check("contradictory frozen fast-gradient policies poison one row's evidence",
          not contradictory_fast_valid
          and any(failure["check"] == "pgd_fast_gradient_parameter_engaged"
                  for failure in contradictory_fast_failures),
          repr(contradictory_fast_failures))

    def postbab_reserve_event(status="ran", **overrides):
        fields = {
            "schema": _POSTBAB_RESERVE_EVENT_SCHEMA,
            "phase": "terminal",
            "configured_secs": "100",
            "held_secs": "100",
            "verifier_before_secs": "110",
            "verifier_after_secs": "10",
            "verifier_elapsed_ms": "10000",
            "verifier_outcome": "timeout",
            "deadline_bound": "true",
            "attack_budget_ms": "90000",
            "work_units": "1",
            "work_started": "true",
        }
        fields.update(overrides)
        return machine_event("postbab_reserve_execution", status, **fields)

    def synthetic_postbab_reserve_result(arm_pairs, events, category="cifar100"):
        synthetic_row = _clean_row(arm_pairs, category=category)
        synthetic_row["flight"]["events"] = list(events)
        return validate_run(
            arm_pairs,
            2025,
            category,
            [synthetic_row],
            _clean_manifest(category=category, arm=arm_pairs),
            _summary_of([synthetic_row]),
        )

    def postbab_reserve_contract_accepts(event):
        synthetic_row = _clean_row([])
        synthetic_row["flight"]["events"] = [event]
        return len(_postbab_reserve_terminal_evidences(synthetic_row)) == 1

    valid_postbab_reserve, valid_postbab_failures, _warnings, _counts = (
        synthetic_postbab_reserve_result(
            [("NY_POSTBAB_RESERVE_SECS", "100")],
            [postbab_reserve_event()],
        )
    )
    check("a matching post-BaB reserve terminal proves live engagement",
          valid_postbab_reserve, repr(valid_postbab_failures))

    missing_postbab_valid, missing_postbab_failures, _warnings, _counts = (
        synthetic_postbab_reserve_result(
            [("NY_POSTBAB_RESERVE_SECS", "100")],
            [],
        )
    )
    mismatched_postbab_valid, mismatched_postbab_failures, _warnings, _counts = (
        synthetic_postbab_reserve_result(
            [("NY_POSTBAB_RESERVE_SECS", "100")],
            [postbab_reserve_event(
                configured_secs="90", held_secs="90", verifier_before_secs="100",
            )],
        )
    )
    check("missing and mismatched post-BaB reserve receipts earn no credit",
          all(
              not valid
              and any(failure["check"] == "postbab_reserve_parameter_engaged"
                      for failure in failures)
              for valid, failures in (
                  (missing_postbab_valid, missing_postbab_failures),
                  (mismatched_postbab_valid, mismatched_postbab_failures),
              )
          ), repr((missing_postbab_failures, mismatched_postbab_failures)))

    saturated_postbab_event = postbab_reserve_event(
        held_secs="90", verifier_before_secs="100",
    )
    saturated_postbab_valid, saturated_postbab_failures, _warnings, _counts = (
        synthetic_postbab_reserve_result(
            [("NY_POSTBAB_RESERVE_SECS", "100")],
            [saturated_postbab_event],
        )
    )
    check("a saturated post-BaB reserve is diagnostic but earns no treatment credit",
          postbab_reserve_contract_accepts(saturated_postbab_event)
          and not saturated_postbab_valid
          and any(failure["check"] == "postbab_reserve_parameter_engaged"
                  for failure in saturated_postbab_failures),
          repr(saturated_postbab_failures))

    missing_postbab_field = postbab_reserve_event()
    missing_postbab_field["reason"] = missing_postbab_field["reason"].replace(
        " attack_budget_ms=90000", "",
    )
    extra_postbab_field = postbab_reserve_event()
    extra_postbab_field["reason"] += " extra=1"
    duplicate_postbab_field = postbab_reserve_event()
    duplicate_postbab_field["reason"] += " held_secs=90"
    double_space_postbab = postbab_reserve_event()
    double_space_postbab["reason"] = double_space_postbab["reason"].replace(
        " phase=", "  phase=",
    )
    tab_postbab = postbab_reserve_event()
    tab_postbab["reason"] = tab_postbab["reason"].replace(" phase=", "\tphase=")
    invalid_postbab_reserve_events = [
        missing_postbab_field,
        extra_postbab_field,
        duplicate_postbab_field,
        double_space_postbab,
        tab_postbab,
        postbab_reserve_event(schema="ny-cli/postbab-reserve-execution/v1"),
        postbab_reserve_event(schema="ny-cli/postbab-reserve-execution/v999"),
        postbab_reserve_event(status="skipped"),
        postbab_reserve_event(phase="progress"),
        postbab_reserve_event(verifier_outcome="verified"),
        postbab_reserve_event(deadline_bound="false"),
        postbab_reserve_event(work_started="false"),
        postbab_reserve_event(configured_secs="0100"),
        postbab_reserve_event(held_secs=str(1 << 64)),
        postbab_reserve_event(verifier_before_secs="-1"),
        postbab_reserve_event(verifier_after_secs="10.0"),
        postbab_reserve_event(verifier_elapsed_ms="010000"),
        postbab_reserve_event(work_units=str(1 << 64)),
        postbab_reserve_event(configured_secs="0"),
        postbab_reserve_event(
            held_secs="0", verifier_before_secs="10", verifier_after_secs="10",
        ),
        postbab_reserve_event(
            held_secs="1", verifier_before_secs="10", verifier_after_secs="11",
        ),
        postbab_reserve_event(held_secs="89"),
        postbab_reserve_event(configured_secs="89"),
        postbab_reserve_event(
            held_secs="80", verifier_before_secs="100", verifier_after_secs="20",
            verifier_elapsed_ms="20000",
        ),
        postbab_reserve_event(verifier_before_secs="109", verifier_after_secs="9"),
        postbab_reserve_event(verifier_elapsed_ms="9999"),
        postbab_reserve_event(attack_budget_ms="0"),
        postbab_reserve_event(attack_budget_ms="1999"),
        postbab_reserve_event(work_units="0"),
        postbab_reserve_event(work_units="2"),
    ]
    check("post-BaB reserve v2 rejects malformed grammar, status, and invariants",
          all(not postbab_reserve_contract_accepts(event)
              for event in invalid_postbab_reserve_events))

    malformed_later_postbab = postbab_reserve_event()
    malformed_later_postbab["reason"] += " held_secs=90"
    poisoned_postbab_valid, poisoned_postbab_failures, _warnings, _counts = (
        synthetic_postbab_reserve_result(
            [("NY_POSTBAB_RESERVE_SECS", "100")],
            [postbab_reserve_event(), malformed_later_postbab],
        )
    )
    check("a later malformed post-BaB reserve event poisons that row's evidence",
          not poisoned_postbab_valid
          and any(failure["check"] == "postbab_reserve_parameter_engaged"
                  for failure in poisoned_postbab_failures),
          repr(poisoned_postbab_failures))

    duplicate_valid_postbab, duplicate_valid_failures, _warnings, _counts = (
        synthetic_postbab_reserve_result(
            [("NY_POSTBAB_RESERVE_SECS", "100")],
            [postbab_reserve_event(), postbab_reserve_event()],
        )
    )
    check("duplicate valid post-BaB reserve terminals poison one row",
          not duplicate_valid_postbab
          and any(failure["check"] == "postbab_reserve_parameter_engaged"
                  for failure in duplicate_valid_failures),
          repr(duplicate_valid_failures))

    matching_but_invalid_postbab_row = _clean_row([
        ("NY_POSTBAB_RESERVE_SECS", "100"),
    ])
    matching_but_invalid_postbab_row["flight"]["events"] = [
        postbab_reserve_event(attack_budget_ms="0"),
    ]
    live_but_mismatched_postbab_row = _clean_row([
        ("NY_POSTBAB_RESERVE_SECS", "100"),
    ])
    live_but_mismatched_postbab_row["flight"]["events"] = [
        postbab_reserve_event(
            configured_secs="90", held_secs="90", verifier_before_secs="100",
        ),
    ]
    check("post-BaB reserve configuration and live work cannot be combined across rows",
          not _postbab_reserve_parameter_engagement(
              [("NY_POSTBAB_RESERVE_SECS", "100")],
              [matching_but_invalid_postbab_row, live_but_mismatched_postbab_row],
          ))

    contradictory_postbab_row = _clean_row([
        ("NY_POSTBAB_RESERVE_SECS", "100"),
    ])
    contradictory_postbab_row["flight"]["events"] = [
        postbab_reserve_event(),
        postbab_reserve_event(
            configured_secs="90", held_secs="90", verifier_before_secs="100",
        ),
    ]
    check("contradictory post-BaB reserve configurations poison one row's evidence",
          not _postbab_reserve_parameter_engagement(
              [("NY_POSTBAB_RESERVE_SECS", "100")],
              [contradictory_postbab_row],
          ))

    def root_joint_event(status="ran", **overrides):
        fields = {
            "schema": _ROOT_JOINT_INTERM_ALPHA_EVENT_SCHEMA,
            "phase": "terminal",
            "observation_schema": _ROOT_JOINT_INTERM_ALPHA_OBSERVATION_SCHEMA,
            "route": "input_split",
            "configured_master": "true",
            "configured_max_dim_override": "32768",
            "input_split_configured_max_dim": "32768",
            "multi_objective_configured_max_dim": "none",
            "configured_secs": "30",
            "configured_iters": "50",
            "configured_lr_bits": str(_requested_f32_bits("0.05")),
            "configured_deadline_ascent": "true",
            "configured_min_remaining_secs": "240",
            "configured_max_sel": "32",
            "configured_frozen_stop": "false",
            "configured_pure_chain": "false",
            "configured_bn_gpu_extract": "false",
            "max_effective_iters": "50",
            "invocations": "1",
            "input_split_invocations": "1",
            "multi_objective_invocations": "0",
            "target_count": "3",
            "available_budget_ms": "60000",
            "effective_slice_ms": "30000",
            "configured_slice_work_invocations": "1",
            "configured_iters_binding_invocations": "1",
            "lowered_min_remaining_work_invocations": "0",
            "raised_min_remaining_binding_refusals": "0",
            "work_started": "1",
            "attempts": "1",
            "completed": "1",
            "tightened_count": "0",
            "outlier_tightened_count": "0",
            "target_attempts": "3",
            "base_folds_completed": "3",
            "ascent_iterations_attempted": "50",
            "max_ascent_iterations_attempted": "50",
            "optimizer_steps": "50",
            "ascent_folds_completed": "50",
            "max_selected_rows": "32",
            "selection_cap_bound_targets": "1",
            "frozen_stop_targets": "0",
            "pure_chain_targets": "0",
            "bn_extract_targets": "0",
            "master_disabled": "0",
            "admission_floor_refusals": "0",
            "deadline_route_refusals": "0",
            "slice_refusals": "0",
            "no_target_refusals": "0",
            "capability_refusals": "0",
            "execution_refusals": "0",
        }
        fields.update(overrides)
        return machine_event("root_joint_interm_alpha_execution", status, **fields)

    def synthetic_root_joint_result(arm_pairs, events, engagement_pairs=None,
                                    require_closed_root_terminal=False):
        synthetic_row = _clean_row(arm_pairs)
        synthetic_row["flight"]["events"] = list(events)
        return validate_run(
            arm_pairs,
            2025,
            "cifar100",
            [synthetic_row],
            _clean_manifest(arm=arm_pairs),
            _summary_of([synthetic_row]),
            engagement_arm=engagement_pairs,
            require_closed_root_terminal=require_closed_root_terminal,
        )

    root_arm = [
        ("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT", "1"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_SECS", "30"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_ITERS", "50"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_LR", "0.05"),
    ]
    valid_root, valid_root_failures, _warnings, _counts = (
        synthetic_root_joint_result(root_arm, [root_joint_event()])
    )
    check("one exact root-alpha terminal proves live family engagement",
          valid_root, repr(valid_root_failures))

    outlier_only_valid, outlier_only_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            root_arm,
            [root_joint_event(outlier_tightened_count="1")],
        )
    )
    check("an outlier-only root prepass cannot authenticate joint-alpha work",
          not outlier_only_valid
          and any(failure["check"]
                  == "root_joint_interm_alpha_parameter_engaged"
                  for failure in outlier_only_failures),
          repr(outlier_only_failures))

    no_work_root = root_joint_event(
        status="skipped", target_count="0", available_budget_ms="60000",
        effective_slice_ms="30000", work_started="0", attempts="0",
        completed="0", target_attempts="0", base_folds_completed="0",
        configured_slice_work_invocations="0",
        configured_iters_binding_invocations="0",
        max_effective_iters="0", ascent_iterations_attempted="0",
        max_ascent_iterations_attempted="0", optimizer_steps="0",
        ascent_folds_completed="0", max_selected_rows="0",
        selection_cap_bound_targets="0", frozen_stop_targets="0",
        pure_chain_targets="0", bn_extract_targets="0",
        no_target_refusals="1",
    )
    no_work_valid, no_work_failures, _warnings, _counts = (
        synthetic_root_joint_result(root_arm, [no_work_root])
    )
    mismatched_control_event = root_joint_event(
        status="skipped", configured_master="false", target_count="0",
        effective_slice_ms="0", configured_slice_work_invocations="0",
        configured_iters_binding_invocations="0", max_effective_iters="0",
        work_started="0", attempts="0", completed="0", target_attempts="0",
        base_folds_completed="0", ascent_iterations_attempted="0",
        max_ascent_iterations_attempted="0", optimizer_steps="0",
        ascent_folds_completed="0", max_selected_rows="0",
        selection_cap_bound_targets="0", frozen_stop_targets="0",
        pure_chain_targets="0", bn_extract_targets="0", master_disabled="1",
    )
    closed_control_valid, closed_control_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            [("NY_ROOT_JOINT_INTERM_ALPHA", "1")],
            [no_work_root],
            engagement_pairs=[],
            require_closed_root_terminal=True,
        )
    )
    mismatched_control_valid, mismatched_control_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            [("NY_ROOT_JOINT_INTERM_ALPHA", "1")],
            [mismatched_control_event],
            engagement_pairs=[],
            require_closed_root_terminal=True,
        )
    )
    check("root prerequisite controls accept truthful zero-work terminals but bind config",
          closed_control_valid and not mismatched_control_valid
          and any(failure["check"] == "root_joint_interm_alpha_terminal_closed"
                  for failure in mismatched_control_failures),
          repr((closed_control_failures, mismatched_control_failures)))
    matching_control_row = _clean_row([
        ("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
    ])
    matching_control_row["flight"]["events"] = [no_work_root]
    mismatched_control_row = _clean_row([
        ("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
    ])
    mismatched_control_row["occurrence"] = 1
    mismatched_control_row["instance_index"] = 1
    mismatched_control_row["flight"]["events"] = [mismatched_control_event]
    mixed_control_rows = [matching_control_row, mismatched_control_row]
    mixed_control_valid, mixed_control_failures, _warnings, _counts = validate_run(
        [("NY_ROOT_JOINT_INTERM_ALPHA", "1")],
        2025,
        "cifar100",
        mixed_control_rows,
        _clean_manifest(arm=[("NY_ROOT_JOINT_INTERM_ALPHA", "1")]),
        _summary_of(mixed_control_rows),
        engagement_arm=[],
        require_closed_root_terminal=True,
    )
    check(
        "one authenticated root prerequisite row cannot cover a mismatched sibling",
        not mixed_control_valid
        and any(
            failure["check"] == "root_joint_interm_alpha_terminal_closed"
            for failure in mixed_control_failures
        ),
        repr(mixed_control_failures),
    )
    zero_target_root = root_joint_event(target_count="0")
    zero_target_valid, zero_target_failures, _warnings, _counts = (
        synthetic_root_joint_result(root_arm, [zero_target_root])
    )
    check("configuration-only and zero-target root-alpha rows earn no T1 credit",
          not no_work_valid and not zero_target_valid
          and all(any(failure["check"]
                      == "root_joint_interm_alpha_parameter_engaged"
                      for failure in failures)
                  for failures in (no_work_failures, zero_target_failures)),
          repr((no_work_failures, zero_target_failures)))

    base_only_root = root_joint_event(
        configured_iters_binding_invocations="0",
        max_effective_iters="0", ascent_iterations_attempted="0",
        max_ascent_iterations_attempted="0", optimizer_steps="0",
        ascent_folds_completed="0",
    )
    base_only_valid, base_only_failures, _warnings, _counts = (
        synthetic_root_joint_result(root_arm, [base_only_root])
    )
    base_master_valid, base_master_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            [("NY_ROOT_JOINT_INTERM_ALPHA", "1")], [base_only_root]
        )
    )
    check("base-fold work cannot graduate ascent, iteration, or LR controls",
          not base_only_valid and base_master_valid
          and any(failure["check"]
                  == "root_joint_interm_alpha_parameter_engaged"
                  and "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT"
                      in failure["detail"]
                  and "NY_ROOT_JOINT_INTERM_ALPHA_ITERS" in failure["detail"]
                  and "NY_ROOT_JOINT_INTERM_ALPHA_LR" in failure["detail"]
                  for failure in base_only_failures),
          repr((base_only_failures, base_master_failures)))

    capped_secs_valid, capped_secs_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            root_arm,
            [root_joint_event(
                available_budget_ms="20000", effective_slice_ms="20000",
                configured_slice_work_invocations="0",
            )],
        )
    )
    check("a capped-away root-alpha SECS request is work but not SECS engagement",
          not capped_secs_valid
          and any(failure["check"]
                  == "root_joint_interm_alpha_parameter_engaged"
                  and "NY_ROOT_JOINT_INTERM_ALPHA_SECS" in failure["detail"]
                  for failure in capped_secs_failures),
          repr(capped_secs_failures))

    mismatched_root_valid, mismatched_root_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            root_arm,
            [root_joint_event(
                configured_lr_bits=str(_requested_f32_bits("0.1")),
            )],
        )
    )
    check("a mismatched root-alpha LR never borrows other optimizer work",
          not mismatched_root_valid
          and any(failure["check"]
                  == "root_joint_interm_alpha_parameter_engaged"
                  and "NY_ROOT_JOINT_INTERM_ALPHA_LR" in failure["detail"]
                  for failure in mismatched_root_failures),
          repr(mismatched_root_failures))

    floor_arm = [
        ("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
        ("NY_ROOT_JOINT_MIN_REMAINING_SECS", "120"),
    ]
    live_floor_event = root_joint_event(
        route="multi_objective", configured_min_remaining_secs="120",
        input_split_invocations="0", multi_objective_invocations="1",
        input_split_configured_max_dim="none",
        multi_objective_configured_max_dim="32768",
        available_budget_ms="180000",
        lowered_min_remaining_work_invocations="1",
    )
    floor_valid, floor_failures, _warnings, _counts = synthetic_root_joint_result(
        floor_arm, [live_floor_event])
    wrong_floor_valid, wrong_floor_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            floor_arm,
            [root_joint_event(configured_min_remaining_secs="120")],
        )
    )
    check("the admission floor credits only live multi-objective counterfactual work",
          floor_valid and not wrong_floor_valid
          and any(failure["check"]
                  == "root_joint_interm_alpha_parameter_engaged"
                  for failure in wrong_floor_failures),
          repr((floor_failures, wrong_floor_failures)))

    raised_floor_arm = [
        ("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
        ("NY_ROOT_JOINT_MIN_REMAINING_SECS", "360"),
    ]
    raised_floor_event = root_joint_event(
        status="skipped",
        route="multi_objective",
        configured_min_remaining_secs="360",
        input_split_invocations="0",
        multi_objective_invocations="1",
        input_split_configured_max_dim="none",
        multi_objective_configured_max_dim="32768",
        target_count="0",
        available_budget_ms="300000",
        effective_slice_ms="0",
        configured_slice_work_invocations="0",
        configured_iters_binding_invocations="0",
        max_effective_iters="0",
        work_started="0",
        attempts="0",
        completed="0",
        tightened_count="0",
        target_attempts="0",
        base_folds_completed="0",
        ascent_iterations_attempted="0",
        max_ascent_iterations_attempted="0",
        optimizer_steps="0",
        ascent_folds_completed="0",
        max_selected_rows="0",
        selection_cap_bound_targets="0",
        frozen_stop_targets="0",
        pure_chain_targets="0",
        bn_extract_targets="0",
        admission_floor_refusals="1",
        raised_min_remaining_binding_refusals="1",
    )
    raised_floor_valid, raised_floor_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            raised_floor_arm,
            [raised_floor_event],
            engagement_pairs=[("NY_ROOT_JOINT_MIN_REMAINING_SECS", "360")],
        )
    )
    # Rebuild rather than mutate the machine reason so duplicate tokens cannot
    # accidentally turn this into a parser-only rejection.
    unbound_raised = root_joint_event(
        status="skipped",
        route="multi_objective",
        configured_min_remaining_secs="360",
        input_split_invocations="0",
        multi_objective_invocations="1",
        input_split_configured_max_dim="none",
        multi_objective_configured_max_dim="32768",
        target_count="0",
        available_budget_ms="300000",
        effective_slice_ms="0",
        configured_slice_work_invocations="0",
        configured_iters_binding_invocations="0",
        max_effective_iters="0",
        work_started="0",
        attempts="0",
        completed="0",
        tightened_count="0",
        target_attempts="0",
        base_folds_completed="0",
        ascent_iterations_attempted="0",
        max_ascent_iterations_attempted="0",
        optimizer_steps="0",
        ascent_folds_completed="0",
        max_selected_rows="0",
        selection_cap_bound_targets="0",
        frozen_stop_targets="0",
        pure_chain_targets="0",
        bn_extract_targets="0",
        admission_floor_refusals="1",
        raised_min_remaining_binding_refusals="0",
    )
    unbound_raised_valid, unbound_raised_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            raised_floor_arm,
            [unbound_raised],
            engagement_pairs=[("NY_ROOT_JOINT_MIN_REMAINING_SECS", "360")],
        )
    )
    check("a raised admission floor credits its exact terminal refusal window",
          raised_floor_valid and not unbound_raised_valid
          and any(failure["check"]
                  == "root_joint_interm_alpha_parameter_engaged"
                  for failure in unbound_raised_failures),
          repr((raised_floor_failures, unbound_raised_failures)))

    expanded_root_arm = [
        ("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL", "8"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP", "1"),
        ("NY_BN_GPU_EXTRACT", "1"),
    ]
    expanded_valid, expanded_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            expanded_root_arm,
            [root_joint_event(
                configured_max_sel="8", max_selected_rows="8",
                configured_frozen_stop="true", frozen_stop_targets="1",
                configured_bn_gpu_extract="true", bn_extract_targets="1",
            )],
        )
    )
    inert_expanded, inert_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            expanded_root_arm,
            [root_joint_event(
                configured_max_sel="8", max_selected_rows="7",
                configured_frozen_stop="true",
                configured_bn_gpu_extract="true",
                selection_cap_bound_targets="0", frozen_stop_targets="0",
                bn_extract_targets="0",
            )],
        )
    )
    check("expanded root controls require their own authenticated runtime effects",
          expanded_valid and not inert_expanded
          and any(failure["check"]
                  == "root_joint_interm_alpha_parameter_engaged"
                  and "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL" in failure["detail"]
                  and "NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP" in failure["detail"]
                  and "NY_BN_GPU_EXTRACT" in failure["detail"]
                  for failure in inert_failures),
          repr((expanded_failures, inert_failures)))

    pure_chain_arm = [
        ("NY_ROOT_JOINT_INTERM_ALPHA", "1"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP", "0"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN", "1"),
    ]
    pure_chain_valid, pure_chain_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            pure_chain_arm,
            [root_joint_event(
                configured_pure_chain="true", pure_chain_targets="1",
            )],
            engagement_pairs=[("NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN", "1")],
        )
    )
    confounded_pure_valid, confounded_pure_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            pure_chain_arm,
            [root_joint_event(
                configured_frozen_stop="true", frozen_stop_targets="1",
                configured_pure_chain="true", pure_chain_targets="1",
            )],
            engagement_pairs=[("NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN", "1")],
        )
    )
    check("pure-chain credit requires frozen-stop to be independently disabled",
          pure_chain_valid and not confounded_pure_valid,
          repr((pure_chain_failures, confounded_pure_failures)))

    repeated_root = root_joint_event(
        invocations="2", input_split_invocations="2", target_count="6",
        available_budget_ms="120000", effective_slice_ms="60000",
        configured_slice_work_invocations="2",
        configured_iters_binding_invocations="2", work_started="2",
        attempts="2", completed="2", target_attempts="6",
        base_folds_completed="6", ascent_iterations_attempted="100",
        optimizer_steps="100", ascent_folds_completed="100",
        selection_cap_bound_targets="2",
    )
    repeated_valid, repeated_failures, _warnings, _counts = (
        synthetic_root_joint_result(root_arm, [repeated_root])
    )
    check("repeated closed root invocations retain exact parameter engagement",
          repeated_valid, repr(repeated_failures))

    malformed_root = root_joint_event()
    malformed_root["reason"] += " attempts=1"
    extra_root = root_joint_event()
    extra_root["reason"] += " invented=1"
    invalid_root_events = [
        malformed_root,
        extra_root,
        root_joint_event(
            schema="ny-cli/root-joint-interm-alpha-execution/v0"),
        root_joint_event(
            observation_schema=
                "ny_beta_crown_root_joint_interm_alpha_observations_v0"),
        root_joint_event(invocations="01"),
        root_joint_event(configured_lr_bits=str(1 << 32)),
        root_joint_event(route="multi_objective"),
        root_joint_event(completed="0"),
        root_joint_event(tightened_count="4"),
        root_joint_event(target_count="2"),
        root_joint_event(configured_max_sel="0"),
        root_joint_event(configured_frozen_stop="false", frozen_stop_targets="1"),
        root_joint_event(selection_cap_bound_targets="1", max_selected_rows="31"),
        root_joint_event(effective_slice_ms="60001"),
    ]
    check("root-alpha v1 rejects malformed grammar and contradictory aggregates",
          all(not synthetic_root_joint_result(root_arm, [event])[0]
              for event in invalid_root_events))
    duplicate_root_valid, duplicate_root_failures, _warnings, _counts = (
        synthetic_root_joint_result(
            root_arm, [root_joint_event(), root_joint_event()])
    )
    check("duplicate root-alpha terminals poison the row",
          not duplicate_root_valid
          and any(failure["check"]
                  == "root_joint_interm_alpha_parameter_engaged"
                  for failure in duplicate_root_failures),
          repr(duplicate_root_failures))

    mo_enable_valid, mo_enable_failures, _warnings, _counts = (
        synthetic_mo_kfsb_result("1", [mo_kfsb_event()])
    )
    mo_disable_valid, mo_disable_failures, _warnings, _counts = (
        synthetic_mo_kfsb_result("0", [mo_kfsb_disable_event()])
    )
    check("MO-kFSB canonical force-on credits real zero-leaf selector work",
          mo_enable_valid, repr(mo_enable_failures))
    check("MO-kFSB canonical force-off credits a live lower-on refusal",
          mo_disable_valid, repr(mo_disable_failures))

    reduce_min_valid, reduce_min_failures, _warnings, _counts = (
        synthetic_mo_kfsb_reduce_result(
            "min", [mo_kfsb_reduce_event("min")])
    )
    reduce_max_valid, reduce_max_failures, _warnings, _counts = (
        synthetic_mo_kfsb_reduce_result(
            "max", [mo_kfsb_reduce_event("max")])
    )
    check("MO-kFSB reducer credits exact Min and Max committed selector work",
          reduce_min_valid and reduce_max_valid,
          repr((reduce_min_failures, reduce_max_failures)))

    reduce_mismatch_valid, reduce_mismatch_failures, _warnings, _counts = (
        synthetic_mo_kfsb_reduce_result(
            "min", [mo_kfsb_reduce_event("max")])
    )
    reduce_no_selector = mo_kfsb_reduce_event(
        route_eligible_opportunities="0",
        selector_waves="0",
        selector_domains="0",
        committed_waves="0",
        committed_parents="0",
    )
    reduce_no_commit = mo_kfsb_reduce_event(
        committed_waves="0", committed_parents="0",
        force_on_binding_waves="0",
    )
    reduce_same_value = mo_kfsb_reduce_event(
        "max", configured_reduce_op="max",
    )
    reduce_master_mismatch = mo_kfsb_reduce_event(
        "max",
        canonical_master="absent",
        configured_master="true",
        lower_master_opportunities="1",
        force_on_binding_waves="0",
    )
    reduce_inert_results = [
        synthetic_mo_kfsb_reduce_result("max", [event])
        for event in (
            reduce_no_selector,
            reduce_no_commit,
            reduce_same_value,
            reduce_master_mismatch,
        )
    ]
    check("MO-kFSB reducer rejects mismatch, no-op, inactive, and wrong-master rows",
          not reduce_mismatch_valid
          and all(not result[0] for result in reduce_inert_results)
          and all(any(failure["check"] == "mo_kfsb_parameter_engaged"
                      for failure in result[1])
                  for result in [
                      (reduce_mismatch_valid, reduce_mismatch_failures,
                       _warnings, _counts),
                      *reduce_inert_results,
                  ]),
          repr((reduce_mismatch_failures, reduce_inert_results)))

    reduce_valid_row = _clean_row([
        ("NY_MO_KFSB", "1"), ("NY_MO_KFSB_REDUCE", "max"),
    ])
    reduce_valid_row["flight"]["events"] = [mo_kfsb_reduce_event("max")]
    reduce_mismatched_row = _clean_row([
        ("NY_MO_KFSB", "1"), ("NY_MO_KFSB_REDUCE", "max"),
    ])
    reduce_mismatched_row["flight"]["events"] = [mo_kfsb_reduce_event("min")]
    reduce_malformed_row = _clean_row([
        ("NY_MO_KFSB", "1"), ("NY_MO_KFSB_REDUCE", "max"),
    ])
    reduce_malformed_row["flight"]["events"] = [
        mo_kfsb_reduce_event("max", selector_waves="01"),
    ]
    reduce_duplicate_row = _clean_row([
        ("NY_MO_KFSB", "1"), ("NY_MO_KFSB_REDUCE", "max"),
    ])
    reduce_duplicate_row["flight"]["events"] = [
        mo_kfsb_reduce_event("max"), mo_kfsb_reduce_event("max"),
    ]
    reduce_inactive_row = _clean_row([
        ("NY_MO_KFSB", "1"), ("NY_MO_KFSB_REDUCE", "max"),
    ])
    reduce_inactive_row["flight"]["events"] = [reduce_no_selector]
    reduce_context = [
        ("NY_MO_KFSB", "1"), ("NY_MO_KFSB_REDUCE", "max"),
    ]
    check("MO-kFSB reducer requires one causal terminal on every scored row",
          "NY_MO_KFSB_REDUCE" in _mo_kfsb_parameter_engagement(
              reduce_context, [reduce_valid_row],
          )
          and "NY_MO_KFSB_REDUCE" not in _mo_kfsb_parameter_engagement(
              reduce_context, [reduce_valid_row, reduce_mismatched_row],
          )
          and "NY_MO_KFSB_REDUCE" not in _mo_kfsb_parameter_engagement(
              reduce_context, [reduce_valid_row, reduce_malformed_row],
          )
          and "NY_MO_KFSB_REDUCE" not in _mo_kfsb_parameter_engagement(
              reduce_context, [reduce_duplicate_row],
          )
          and "NY_MO_KFSB_REDUCE" not in _mo_kfsb_parameter_engagement(
              reduce_context, [reduce_valid_row, reduce_inactive_row],
          ))

    enable_lower_on = mo_kfsb_event(
        configured_master="true", lower_master_opportunities="1",
        force_on_binding_waves="0",
    )
    enable_no_commit = mo_kfsb_event(
        committed_waves="0", committed_parents="0",
        force_on_binding_waves="0",
    )
    enable_no_selector = mo_kfsb_event(
        selector_waves="0", selector_domains="0", committed_waves="0",
        committed_parents="0", force_on_binding_waves="0",
    )
    disable_lower_off = mo_kfsb_disable_event(
        configured_master="false", lower_master_opportunities="0",
        master_binding_refusals="0",
    )
    disable_no_route = mo_kfsb_disable_event(
        route_eligible_opportunities="0", master_binding_refusals="0",
    )
    disable_no_candidate = mo_kfsb_disable_event(
        effective_candidates="0", candidate_eligible_opportunities="0",
        route_eligible_opportunities="0", master_binding_refusals="0",
    )
    absent_master = mo_kfsb_event(
        canonical_master="absent", selector_waves="0", selector_domains="0",
        committed_waves="0", committed_parents="0",
        force_on_binding_waves="0", effective_master_opportunities="0",
    )
    rejected_master = mo_kfsb_event(
        canonical_master="rejected", selector_waves="0", selector_domains="0",
        committed_waves="0", committed_parents="0",
        force_on_binding_waves="0", effective_master_opportunities="0",
    )
    check("MO-kFSB configuration, lower-on, no-commit, absent, and rejected rows "
          "earn no master credit",
          all(not synthetic_mo_kfsb_result(token, [event])[0]
              for token, event in (
                  ("1", enable_lower_on),
                  ("1", enable_no_commit),
                  ("1", enable_no_selector),
                  ("0", disable_lower_off),
                  ("0", disable_no_route),
                  ("0", disable_no_candidate),
                  ("1", absent_master),
                  ("1", rejected_master),
              )))

    missing_mo_row = _clean_row([("NY_MO_KFSB", "1")])
    valid_mo_row = _clean_row([("NY_MO_KFSB", "1")])
    valid_mo_row["flight"]["events"] = [mo_kfsb_event()]
    inactive_mo_row = _clean_row([("NY_MO_KFSB", "1")])
    inactive_mo_row["occurrence"] = 1
    inactive_mo_row["instance_index"] = 1
    inactive_mo_row["flight"]["events"] = [enable_lower_on]
    malformed_mo_row = _clean_row([("NY_MO_KFSB", "1")])
    malformed_mo_row["flight"]["events"] = [
        mo_kfsb_event(force_on_binding_waves="01"),
    ]
    check("MO-kFSB requires one valid terminal on every scored row",
          not _mo_kfsb_parameter_engagement(
              [("NY_MO_KFSB", "1")], [valid_mo_row, missing_mo_row])
          and not _mo_kfsb_parameter_engagement(
              [("NY_MO_KFSB", "1")], [valid_mo_row, malformed_mo_row]))

    mo_mixed_rows = [valid_mo_row, inactive_mo_row]
    mo_mixed_valid, mo_mixed_failures, _warnings, _counts = validate_run(
        [("NY_MO_KFSB", "1")],
        2025,
        "cifar100",
        mo_mixed_rows,
        _clean_manifest(arm=[("NY_MO_KFSB", "1")]),
        _summary_of(mo_mixed_rows),
    )
    check(
        "one live MO-kFSB row cannot lend master credit to an inactive sibling",
        "NY_MO_KFSB" in _mo_kfsb_parameter_engagement(
            [("NY_MO_KFSB", "1")], mo_mixed_rows
        )
        and "NY_MO_KFSB" not in _every_row_parameter_engagement(
            _mo_kfsb_parameter_engagement,
            [("NY_MO_KFSB", "1")],
            mo_mixed_rows,
        )
        and not mo_mixed_valid
        and any(
            failure["check"] == "mo_kfsb_parameter_engaged"
            for failure in mo_mixed_failures
        ),
        repr(mo_mixed_failures),
    )

    missing_field = mo_kfsb_event()
    missing_field["reason"] = " ".join(
        token for token in missing_field["reason"].split(" ")
        if not token.startswith("offered_domains=")
    )
    missing_reduce_op = mo_kfsb_event()
    missing_reduce_op["reason"] = " ".join(
        token for token in missing_reduce_op["reason"].split(" ")
        if not token.startswith("configured_reduce_op=")
    )
    duplicate_field = mo_kfsb_event()
    duplicate_field["reason"] += " selector_waves=1"
    extra_field = mo_kfsb_event()
    extra_field["reason"] += " invented=1"
    conflict_flag = mo_kfsb_event()
    conflict_flag["reason"] += " attribution_conflict=true"
    overflow_flag = mo_kfsb_event()
    overflow_flag["reason"] += " counter_overflow=true"
    double_space = mo_kfsb_event()
    double_space["reason"] = double_space["reason"].replace(
        " phase=", "  phase=",
    )
    tab_space = mo_kfsb_event()
    tab_space["reason"] = tab_space["reason"].replace(" phase=", "\tphase=")
    non_ascii = mo_kfsb_event()
    non_ascii["reason"] = non_ascii["reason"].replace(
        "canonical_master=exact_enable", "canonical_master=exact_enablé",
    )
    full_selector_domain_omission = mo_kfsb_event(selector_domains="2")
    effective_mean_without_configured_mean = mo_kfsb_event(
        effective_reduce_op="mean",
    )
    typed_select_under_non_min = mo_kfsb_event(
        typed_select_opportunities="1",
        lower_master_opportunities="1",
        force_on_binding_waves="0",
    )
    malformed_mo_events = [
        missing_field,
        missing_reduce_op,
        duplicate_field,
        extra_field,
        conflict_flag,
        overflow_flag,
        double_space,
        tab_space,
        non_ascii,
        mo_kfsb_event(schema="ny-cli/mo-kfsb-execution/v0"),
        mo_kfsb_event(
            observation_schema="ny_beta_crown_mo_kfsb_observations_v0"),
        mo_kfsb_event(phase="progress"),
        mo_kfsb_event(status="skipped"),
        mo_kfsb_event(canonical_master="enabled"),
        mo_kfsb_event(legacy_childsim="1"),
        mo_kfsb_event(configured_master="1"),
        mo_kfsb_event(configured_reduce_op="MAX"),
        mo_kfsb_event(effective_reduce_op="median"),
        mo_kfsb_event(heuristic_eligible="1"),
        mo_kfsb_event(branch_opportunities="0"),
        mo_kfsb_event(offered_domains="0"),
        mo_kfsb_event(nonempty_domain_opportunities="0"),
        mo_kfsb_event(candidate_eligible_opportunities="0"),
        mo_kfsb_event(typed_select_opportunities="1"),
        mo_kfsb_event(lower_master_opportunities="1"),
        mo_kfsb_event(effective_master_opportunities="0"),
        mo_kfsb_event(route_eligible_opportunities="2"),
        mo_kfsb_event(selector_waves="2"),
        mo_kfsb_event(selector_domains="4"),
        mo_kfsb_event(committed_waves="0"),
        mo_kfsb_event(committed_parents="4"),
        mo_kfsb_event(committed_parents="0", committed_leaves="1"),
        mo_kfsb_event(force_on_binding_waves="2"),
        mo_kfsb_event(master_binding_refusals="1"),
        mo_kfsb_event(heuristic_eligible="false"),
        mo_kfsb_event(effective_candidates="0"),
        mo_kfsb_event(legacy_childsim="true"),
        mo_kfsb_event(force_on_binding_waves="01"),
        mo_kfsb_event(force_on_binding_waves=str(1 << 64)),
        mo_kfsb_event(
            branch_opportunities="2", offered_domains="3",
            nonempty_domain_opportunities="1",
            candidate_eligible_opportunities="2",
            effective_master_opportunities="2",
        ),
        mo_kfsb_event(
            effective_candidates="0", candidate_eligible_opportunities="0",
            typed_select_opportunities="1", lower_master_opportunities="1",
            route_eligible_opportunities="0", selector_waves="0",
            selector_domains="0", committed_waves="0", committed_parents="0",
            force_on_binding_waves="0",
        ),
        mo_kfsb_event(
            heuristic_eligible="false", typed_select_opportunities="1",
            lower_master_opportunities="1", route_eligible_opportunities="0",
            selector_waves="0", selector_domains="0", committed_waves="0",
            committed_parents="0", force_on_binding_waves="0",
        ),
        mo_kfsb_event(
            branch_opportunities="2", offered_domains="3",
            nonempty_domain_opportunities="2",
            candidate_eligible_opportunities="2",
            effective_master_opportunities="2", route_eligible_opportunities="1",
            selector_waves="2",
        ),
        mo_kfsb_event(
            selector_domains="0", committed_waves="0", committed_parents="0",
            force_on_binding_waves="0",
        ),
        mo_kfsb_event(
            branch_opportunities="2", offered_domains="3",
            nonempty_domain_opportunities="2",
            candidate_eligible_opportunities="2",
            effective_master_opportunities="2", route_eligible_opportunities="1",
            selector_waves="2", committed_waves="2", committed_parents="2",
            force_on_binding_waves="2",
        ),
        mo_kfsb_event(committed_leaves="1025"),
        mo_kfsb_event(
            offered_domains=str((1 << 64) - 1),
            selector_domains=str((1 << 64) - 1),
            committed_parents=str(((1 << 64) - 1) // 1024 + 1),
        ),
        mo_kfsb_event(
            canonical_master="absent", branch_opportunities="2",
            offered_domains="2", nonempty_domain_opportunities="2",
            candidate_eligible_opportunities="2",
            typed_select_opportunities="1", lower_master_opportunities="1",
            effective_master_opportunities="1",
            route_eligible_opportunities="2", selector_waves="0",
            selector_domains="0", committed_waves="0",
            committed_parents="0", force_on_binding_waves="0",
        ),
        mo_kfsb_disable_event(
            branch_opportunities="2", offered_domains="2",
            nonempty_domain_opportunities="2",
            candidate_eligible_opportunities="2",
            lower_master_opportunities="2",
            route_eligible_opportunities="1", master_binding_refusals="0",
        ),
        mo_kfsb_event(
            branch_opportunities="2", offered_domains="2",
            nonempty_domain_opportunities="2",
            candidate_eligible_opportunities="2",
            typed_select_opportunities="1", lower_master_opportunities="1",
            effective_master_opportunities="2",
            route_eligible_opportunities="2", selector_waves="2",
            selector_domains="2", committed_waves="2",
            committed_parents="2", force_on_binding_waves="0",
        ),
        mo_kfsb_event(
            branch_opportunities="2", offered_domains="3",
            nonempty_domain_opportunities="2",
            candidate_eligible_opportunities="2",
            effective_master_opportunities="2",
            route_eligible_opportunities="1", selector_waves="1",
            selector_domains="3", committed_waves="1",
            committed_parents="1", force_on_binding_waves="1",
        ),
        mo_kfsb_event(
            branch_opportunities="2", offered_domains="2",
            nonempty_domain_opportunities="2",
            candidate_eligible_opportunities="2",
            effective_master_opportunities="2",
            route_eligible_opportunities="2", selector_waves="2",
            selector_domains="2", committed_waves="1",
            committed_parents="2", force_on_binding_waves="1",
        ),
        full_selector_domain_omission,
        effective_mean_without_configured_mean,
        typed_select_under_non_min,
    ]
    accepted_malformed_mo = [
        index for index, event in enumerate(malformed_mo_events)
        if synthetic_mo_kfsb_result("1", [event])[0]
    ]
    check("MO-kFSB v1 rejects corrected producer-impossible aggregates",
          all(_mo_kfsb_execution_evidence({"flight": {"events": [event]}})
              is None for event in malformed_mo_events[-15:]))
    check("MO-kFSB v1 closes domains, reducer authority, and typed Select policy",
          all(_mo_kfsb_execution_evidence({"flight": {"events": [event]}})
              is None for event in (
                  full_selector_domain_omission,
                  effective_mean_without_configured_mean,
                  typed_select_under_non_min,
              )))
    check("MO-kFSB v1 rejects malformed grammar, flags, enums, and aggregates",
          not accepted_malformed_mo, repr(accepted_malformed_mo))
    duplicate_mo_valid, duplicate_mo_failures, _warnings, _counts = (
        synthetic_mo_kfsb_result("1", [mo_kfsb_event(), mo_kfsb_event()])
    )
    check("duplicate MO-kFSB terminals poison the row",
          not duplicate_mo_valid
          and any(failure["check"] == "mo_kfsb_parameter_engaged"
                  for failure in duplicate_mo_failures),
          repr(duplicate_mo_failures))
    check("only the MO-kFSB master and reducer have graduated T1 evidence",
          _AXIS_SPECIFIC_T1_CONTRACTS.get("NY_MO_KFSB")
              == _MO_KFSB_EVENT_SCHEMA
          and _AXIS_SPECIFIC_T1_CONTRACTS.get("NY_MO_KFSB_REDUCE")
              == _MO_KFSB_EVENT_SCHEMA
          and all(name not in _AXIS_SPECIFIC_T1_CONTRACTS for name in (
              "NY_MO_KFSB_K", "NY_MO_KFSB_CERT_REUSE",
          )))

    mo_alpha_valid, mo_alpha_failures, _warnings, _counts = (
        synthetic_mo_alpha_step_result("1", [mo_alpha_step_event()])
    )
    check("MO-alpha every-fourth T1 credits completed row-local updates",
          mo_alpha_valid
          and _AXIS_SPECIFIC_T1_CONTRACTS.get("NY_MO_ALPHA_STEP")
              == _MO_ALPHA_STEP_EVENT_SCHEMA,
          repr(mo_alpha_failures))
    mo_alpha_receipt_row = {
        "flight": {"events": [mo_alpha_step_event()]}
    }
    check("MO-alpha T1 requires its exact beta-SPSA prerequisite context",
          _mo_alpha_step_parameter_engagement(
              [("NY_MO_ALPHA_STEP", "1")], [mo_alpha_receipt_row]
          ) == set()
          and _mo_alpha_step_parameter_engagement(
              [("NY_MO_ALPHA_STEP", "1"), ("NY_MO_BETA_SPSA", "0")],
              [mo_alpha_receipt_row],
          ) == set(),
          repr(mo_alpha_receipt_row))

    mo_alpha_refused = mo_alpha_step_event(
        status="skipped",
        alpha_rounds_completed="0",
        alpha_rounds_refused="3",
        alpha_updates_applied="0",
    )
    mo_alpha_before_fourth = mo_alpha_step_event(
        status="skipped",
        eligible_iterations="3",
        fourth_iteration_opportunities="0",
        alpha_rounds_scheduled="0",
        beta_rounds_scheduled="3",
        alpha_rounds_completed="0",
        alpha_updates_applied="0",
        perturbed_coordinates="0",
    )
    mo_alpha_disabled = mo_alpha_step_event(
        status="skipped",
        effective_alpha_step="false",
        alpha_rounds_scheduled="0",
        beta_rounds_scheduled="15",
        alpha_rounds_completed="0",
        alpha_updates_applied="0",
        perturbed_coordinates="0",
    )
    inactive_mo_alpha_results = [
        synthetic_mo_alpha_step_result("1", [event])
        for event in (
            mo_alpha_refused,
            mo_alpha_before_fourth,
            mo_alpha_disabled,
        )
    ]
    mo_alpha_zero_result = synthetic_mo_alpha_step_result(
        "0", [mo_alpha_disabled]
    )
    check("MO-alpha configuration, pre-fourth, refused, disabled, and force-off "
          "rows earn no T1 credit",
          all(not result[0] for result in inactive_mo_alpha_results)
          and not mo_alpha_zero_result[0]
          and all(any(failure["check"] == "mo_alpha_step_parameter_engaged"
                      for failure in result[1])
                  for result in [*inactive_mo_alpha_results,
                                 mo_alpha_zero_result]),
          repr((inactive_mo_alpha_results, mo_alpha_zero_result)))

    malformed_mo_alpha_events = []
    missing_mo_alpha = mo_alpha_step_event()
    missing_mo_alpha["reason"] = " ".join(
        token for token in missing_mo_alpha["reason"].split(" ")
        if not token.startswith("eligible_iterations=")
    )
    malformed_mo_alpha_events.append(missing_mo_alpha)
    extra_mo_alpha = mo_alpha_step_event()
    extra_mo_alpha["reason"] += " invented=0"
    malformed_mo_alpha_events.append(extra_mo_alpha)
    reordered_mo_alpha = mo_alpha_step_event()
    tokens = reordered_mo_alpha["reason"].split(" ")
    tokens[8], tokens[9] = tokens[9], tokens[8]
    reordered_mo_alpha["reason"] = " ".join(tokens)
    malformed_mo_alpha_events.append(reordered_mo_alpha)
    malformed_mo_alpha_events.extend([
        mo_alpha_step_event(schema="ny-cli/mo-alpha-step-execution/v0"),
        mo_alpha_step_event(phase="progress"),
        mo_alpha_step_event(
            observation_schema=
                "ny_beta_crown_mo_alpha_step_observations_v0"),
        mo_alpha_step_event(observed="1"),
        mo_alpha_step_event(effective_alpha_step="enabled"),
        mo_alpha_step_event(effective_beta_spsa="1"),
        mo_alpha_step_event(schedule_period="3"),
        mo_alpha_step_event(alpha_round_remainder="4"),
        mo_alpha_step_event(optimizer_invocations="01"),
        mo_alpha_step_event(optimizer_invocations=str(1 << 64)),
        mo_alpha_step_event(observed="false"),
        mo_alpha_step_event(effective_alpha_step="none"),
        mo_alpha_step_event(eligible_invocations="2"),
        mo_alpha_step_event(eligible_invocations="0"),
        mo_alpha_step_event(
            eligible_iterations="11", beta_rounds_scheduled="8"),
        mo_alpha_step_event(
            eligible_iterations="16", beta_rounds_scheduled="13"),
        mo_alpha_step_event(fourth_iteration_opportunities="4"),
        mo_alpha_step_event(beta_rounds_scheduled="11"),
        mo_alpha_step_event(alpha_rounds_refused="1"),
        mo_alpha_step_event(alpha_updates_applied="2"),
        mo_alpha_step_event(perturbed_coordinates="0"),
        mo_alpha_step_event(perturbed_coordinates="2"),
        mo_alpha_step_event(perturbed_coordinates="97"),
        mo_alpha_step_event(effective_beta_spsa="false"),
        mo_alpha_step_event(effective_alpha_step="false"),
        mo_alpha_step_event(status="skipped"),
    ])
    accepted_malformed_mo_alpha = [
        index for index, event in enumerate(malformed_mo_alpha_events)
        if synthetic_mo_alpha_step_result("1", [event])[0]
    ]
    check("MO-alpha v1 rejects malformed, reordered, stale, and impossible "
          "schedule aggregates",
          not accepted_malformed_mo_alpha,
          repr(accepted_malformed_mo_alpha))

    duplicate_mo_alpha_valid, duplicate_mo_alpha_failures, _warnings, _counts = (
        synthetic_mo_alpha_step_result(
            "1", [mo_alpha_step_event(), mo_alpha_step_event()]
        )
    )
    check("duplicate MO-alpha terminals poison the row",
          not duplicate_mo_alpha_valid
          and any(failure["check"] == "mo_alpha_step_parameter_engaged"
                  for failure in duplicate_mo_alpha_failures),
          repr(duplicate_mo_alpha_failures))

    live_mo_alpha_row = _clean_row([
        ("NY_MO_ALPHA_STEP", "1"), ("NY_MO_BETA_SPSA", "1"),
    ])
    live_mo_alpha_row["flight"]["events"] = [mo_alpha_step_event()]
    refused_mo_alpha_row = _clean_row([
        ("NY_MO_ALPHA_STEP", "1"), ("NY_MO_BETA_SPSA", "1"),
    ])
    refused_mo_alpha_row["occurrence"] = 1
    refused_mo_alpha_row["instance_index"] = 1
    refused_mo_alpha_row["flight"]["events"] = [mo_alpha_refused]
    mixed_mo_alpha_valid, mixed_mo_alpha_failures, _warnings, _counts = (
        validate_run(
            [("NY_MO_ALPHA_STEP", "1"), ("NY_MO_BETA_SPSA", "1")],
            2025,
            "cifar100",
            [live_mo_alpha_row, refused_mo_alpha_row],
            _clean_manifest(arm=[
                ("NY_MO_ALPHA_STEP", "1"), ("NY_MO_BETA_SPSA", "1"),
            ]),
            _summary_of([live_mo_alpha_row, refused_mo_alpha_row]),
            engagement_arm=[("NY_MO_ALPHA_STEP", "1")],
        )
    )
    check("one live MO-alpha row cannot lend update credit to a refused sibling",
          not mixed_mo_alpha_valid
          and any(failure["check"] == "mo_alpha_step_parameter_engaged"
                  for failure in mixed_mo_alpha_failures),
          repr(mixed_mo_alpha_failures))

    cert_arm = [("NY_MO_KFSB_CERT_REUSE", "1")]
    cert_row = _clean_row(cert_arm)
    cert_default, cert_default_failures, _warnings, _counts = validate_run(
        cert_arm, 2025, "cifar100", [cert_row],
        _clean_manifest(arm=cert_arm), _summary_of([cert_row]),
    )
    cert_strict, cert_failures, _warnings, _counts = validate_run(
        cert_arm, 2025, "cifar100", [cert_row],
        _clean_manifest(arm=cert_arm), _summary_of([cert_row]),
        allow_ungraduated_execution_contracts=False,
    )
    cert_developer, _failures, _warnings, _counts = validate_run(
        cert_arm, 2025, "cifar100", [cert_row],
        _clean_manifest(arm=cert_arm), _summary_of([cert_row]),
        allow_ungraduated_execution_contracts=True,
    )
    check("certificate-reuse delivery is rejected without developer override",
          not cert_default and not cert_strict and cert_developer
          and cert_default_failures == cert_failures
          and any(failure["check"] == "axis_execution_contract_graduated"
                  and "NY_MO_KFSB_CERT_REUSE" in failure["detail"]
                  for failure in cert_failures),
          repr(cert_failures))

    def exact_vjp_event(status="ran", **overrides):
        fields = {
            "schema": _EXACT_VJP_EVENT_SCHEMA,
            "phase": "terminal",
            "route": "conjunctive",
            "outcome": "exhausted",
            "inactive_reason": "none",
            "prefer_exact_batched": "true",
            "batch_enabled": "true",
            "work_started": "true",
            "width_source": "configured",
            "width_configured": "52",
            "width_default": "64",
            "width_initial": "52",
            "width_final": "52",
            "width_max_active": "52",
            "width_binding": "requested",
            "width_retries": "0",
            "total_restarts": "104",
            "restarts_started": "104",
            "waves": "2",
            "wave_steps_configured": "250",
            "wave_steps_default": "1000",
            "wave_steps_effective": "250",
            "vjp_steps": "300",
            "odi_steps_configured": "4",
            "odi_eta": "0.5",
            "odi_vjp_steps": "4",
            "odi_completed_phase_count": "1",
            "odi_move_count": "100",
            "legacy_lanes_configured": "20",
            "legacy_lanes_used": "20",
            "lane_window": "50",
            "recycle_count": "3",
            "exploit_enabled": "true",
            "exploit_opportunity_count": "2",
            "exploit_count": "2",
            "hinge_tau": "0.001",
            "hinge_extra_count": "5",
        }
        fields.update(overrides)
        return machine_event("exact_vjp_execution", status, **fields)

    def exact_vjp_not_run_event(status="skipped", **overrides):
        fields = {
            "outcome": "not_started",
            "inactive_reason": "structural_ineligible",
            "work_started": "false",
            "width_binding": "not_run",
            "width_final": "52",
            "width_max_active": "0",
            "restarts_started": "0",
            "waves": "0",
            "vjp_steps": "0",
            "odi_vjp_steps": "0",
            "odi_completed_phase_count": "0",
            "odi_move_count": "0",
            "legacy_lanes_used": "0",
            "recycle_count": "0",
            "exploit_opportunity_count": "0",
            "exploit_count": "0",
            "hinge_extra_count": "0",
        }
        fields.update(overrides)
        return exact_vjp_event(status=status, **fields)

    def synthetic_exact_vjp_result(arm_pairs, events, category="cifar100"):
        synthetic_row = _clean_row(arm_pairs, category=category)
        synthetic_row["flight"]["events"] = list(events)
        return validate_run(
            arm_pairs,
            2025,
            category,
            [synthetic_row],
            _clean_manifest(category=category, arm=arm_pairs),
            _summary_of([synthetic_row]),
        )

    def exact_contract_accepts(event):
        synthetic_row = _clean_row([])
        synthetic_row["flight"]["events"] = [event]
        return len(_exact_vjp_terminal_evidences(synthetic_row)) == 1

    generic_deadline_event = exact_vjp_not_run_event(
        route="conjunctive_generic", prefer_exact_batched="false",
        wave_steps_effective="1000", inactive_reason="deadline_before_start",
    )
    generic_error_event = exact_vjp_not_run_event(
        route="conjunctive_generic", prefer_exact_batched="false",
        wave_steps_effective="1000", outcome="internal_error",
        inactive_reason="internal_error",
    )
    zero_step_event = exact_vjp_event(
        wave_steps_configured="none", wave_steps_default="0",
        wave_steps_effective="0", vjp_steps="0", recycle_count="0",
        exploit_opportunity_count="0", exploit_count="0", hinge_extra_count="0",
    )
    generic_deadline_valid, generic_deadline_failures, _w, _c = (
        synthetic_exact_vjp_result(
            [("NY_PGD_EXACT_BATCHED", "0")], [generic_deadline_event]
        )
    )
    check("exact-VJP v1 admits producer-realizable generic aborts and zero-step runs",
          all(exact_contract_accepts(event) for event in (
              generic_deadline_event, generic_error_event, zero_step_event,
          )))
    check("a generic no-budget terminal is truthful but earns no treatment credit",
          not generic_deadline_valid
          and any(failure["check"] == "exact_vjp_parameter_engaged"
                  for failure in generic_deadline_failures),
          repr(generic_deadline_failures))

    impossible_exact_events = [
        exact_vjp_event(width_binding="not_run"),
        exact_vjp_event(width_binding="restart_cap"),
        exact_vjp_event(width_max_active="26"),
        exact_vjp_event(vjp_steps="0"),
        exact_vjp_event(
            odi_vjp_steps="4", odi_completed_phase_count="2",
        ),
        exact_vjp_event(
            route="disjunctive", width_default="32",
            wave_steps_effective="1000",
        ),
        exact_vjp_not_run_event(total_restarts="0"),
        exact_vjp_not_run_event(
            outcome="fallback_sequential", inactive_reason="vjp_failure",
        ),
        exact_vjp_not_run_event(
            status="ran", phase="progress", route="conjunctive_generic",
            outcome="in_progress", inactive_reason="none",
            prefer_exact_batched="false", work_started="true",
            wave_steps_effective="1000",
        ),
        exact_vjp_event(
            width_final="26", width_binding="adapter_halved",
            width_retries="3",
        ),
        exact_vjp_event(
            width_max_active="0", restarts_started="0", waves="0",
            vjp_steps="0", odi_vjp_steps="0", odi_completed_phase_count="0",
            odi_move_count="0", legacy_lanes_used="0", recycle_count="0",
            exploit_opportunity_count="0", exploit_count="0",
            hinge_extra_count="0",
        ),
    ]
    check("exact-VJP v1 rejects impossible bindings, counters, and route leakage",
          all(not exact_contract_accepts(event) for event in impossible_exact_events))

    exact_positive_arm = [
        ("NY_PGD_EXACT_BATCHED", "1"),
        ("NY_PGD_VJP_BATCH", "1"),
        ("NY_PGD_VJP_K", "52"),
        ("NY_PGD_LEGACY_LANES", "20"),
        ("NY_PGD_ODI_STEPS", "4"),
        ("NY_PGD_ODI_ETA", "0.5"),
        ("NY_PGD_VJP_STEPS", "250"),
        ("NY_PGD_LANE_WINDOW", "50"),
        ("NY_PGD_EXPLOIT", "1"),
        ("NY_PGD_HINGE_TAU", "0.001"),
    ]
    exact_progress = exact_vjp_event(
        phase="progress", outcome="in_progress", restarts_started="52", waves="1",
        vjp_steps="1", odi_vjp_steps="1", odi_completed_phase_count="0",
        recycle_count="0",
        exploit_opportunity_count="0", exploit_count="0", hinge_extra_count="1",
    )
    exact_positive_valid, exact_positive_failures, _w, _c = (
        synthetic_exact_vjp_result(
            exact_positive_arm, [exact_progress, exact_vjp_event()]
        )
    )
    check("one closed exact-VJP invocation proves every actually consumed positive control",
          exact_positive_valid, repr(exact_positive_failures))

    earlier_engaged_valid, earlier_engaged_failures, _w, _c = (
        synthetic_exact_vjp_result(
            [("NY_PGD_VJP_K", "52")],
            [
                exact_vjp_event(),
                exact_vjp_not_run_event(inactive_reason="deadline_before_start"),
            ],
        )
    )
    reset_counter_row = _clean_row([("NY_PGD_VJP_K", "52")])
    reset_counter_row["flight"]["events"] = [
        exact_vjp_event(),
        exact_vjp_event(
            restarts_started="52", waves="1", vjp_steps="1", odi_vjp_steps="1",
            odi_completed_phase_count="0", recycle_count="0",
            exploit_opportunity_count="0", exploit_count="0", hinge_extra_count="1",
        ),
    ]
    reset_counter_valid, reset_counter_failures, _w, _c = validate_run(
        [("NY_PGD_VJP_K", "52")], 2025, "cifar100", [reset_counter_row],
        _clean_manifest(arm=[("NY_PGD_VJP_K", "52")]),
        _summary_of([reset_counter_row]),
    )
    check("an earlier engaged invocation survives a later no-budget terminal",
          earlier_engaged_valid, repr(earlier_engaged_failures))
    check("closed exact-VJP invocations may reset counters without mixing them",
          reset_counter_valid
          and len(_exact_vjp_terminal_evidences(reset_counter_row)) == 2,
          repr(reset_counter_failures))

    disjunctive_valid, disjunctive_failures, _w, _c = synthetic_exact_vjp_result(
        [("NY_PGD_VJP_BATCH", "1"), ("NY_PGD_VJP_K", "52")],
        [exact_vjp_event(
            route="disjunctive", width_default="32",
            wave_steps_effective="1000", vjp_steps="100", odi_vjp_steps="0",
            odi_completed_phase_count="0", odi_move_count="0",
            legacy_lanes_used="0", recycle_count="0",
            exploit_opportunity_count="0", exploit_count="0", hinge_extra_count="0",
        )],
    )
    adapter_other_control_valid, adapter_other_control_failures, _w, _c = (
        synthetic_exact_vjp_result(
            [("NY_PGD_ODI_STEPS", "4")],
            [exact_vjp_event(
                width_final="26", width_binding="adapter_halved", width_retries="1",
                waves="3", legacy_lanes_used="40",
            )],
        )
    )
    check("shared controls accept disjunctive work and adapter retry does not poison other axes",
          disjunctive_valid and adapter_other_control_valid,
          repr((disjunctive_failures, adapter_other_control_failures)))

    generic_kill_valid, generic_kill_failures, _w, _c = synthetic_exact_vjp_result(
        [("NY_PGD_EXACT_BATCHED", "0")],
        [exact_vjp_not_run_event(
            status="ran", route="conjunctive_generic", outcome="generic_completed",
            inactive_reason="none", prefer_exact_batched="false", work_started="true",
            wave_steps_effective="1000",
        )],
    )
    batch_kill_valid, batch_kill_failures, _w, _c = synthetic_exact_vjp_result(
        [("NY_PGD_VJP_BATCH", "0")],
        [exact_vjp_not_run_event(
            status="ran", outcome="batch_disabled", inactive_reason="batch_disabled",
            batch_enabled="false",
        )],
    )
    preplan_batch_kills = []
    for inactive_reason in ("gpu_unavailable", "empty_objective", "unsupported_plan"):
        preplan_valid, preplan_failures, _w, _c = synthetic_exact_vjp_result(
            [("NY_PGD_VJP_BATCH", "0")],
            [exact_vjp_not_run_event(
                batch_enabled="false", inactive_reason=inactive_reason,
            )],
        )
        preplan_batch_kills.append(
            not preplan_valid
            and any(failure["check"] == "exact_vjp_parameter_engaged"
                    for failure in preplan_failures)
        )
    exact_zero_arm = [
        ("NY_PGD_LEGACY_LANES", "0"),
        ("NY_PGD_ODI_STEPS", "0"),
        ("NY_PGD_EXPLOIT", "0"),
    ]
    zero_controls_valid, zero_controls_failures, _w, _c = synthetic_exact_vjp_result(
        exact_zero_arm,
        [exact_vjp_event(
            legacy_lanes_configured="0", legacy_lanes_used="0",
            odi_steps_configured="0", odi_vjp_steps="0", odi_move_count="0",
            odi_completed_phase_count="0",
            lane_window="0",
            recycle_count="0", exploit_enabled="false",
            exploit_opportunity_count="2", exploit_count="0", hinge_tau="0",
            hinge_extra_count="0",
        )],
    )
    check("explicit route/batch kills and exercised zero feature gates are attributable",
          generic_kill_valid and batch_kill_valid and zero_controls_valid,
          repr((generic_kill_failures, batch_kill_failures, zero_controls_failures)))
    check("batch kill credits only post-plan batch_disabled, never a capability bypass",
          all(preplan_batch_kills), repr(preplan_batch_kills))

    malformed_exact_events = []
    missing_key = exact_vjp_event()
    missing_key["reason"] = " ".join(
        token for token in missing_key["reason"].split(" ")
        if not token.startswith("wave_steps_default=")
    )
    malformed_exact_events.append(missing_key)
    missing_move_count = exact_vjp_event()
    missing_move_count["reason"] = " ".join(
        token for token in missing_move_count["reason"].split(" ")
        if not token.startswith("odi_move_count=")
    )
    malformed_exact_events.append(missing_move_count)
    missing_completed_phase = exact_vjp_event()
    missing_completed_phase["reason"] = " ".join(
        token for token in missing_completed_phase["reason"].split(" ")
        if not token.startswith("odi_completed_phase_count=")
    )
    malformed_exact_events.append(missing_completed_phase)
    extra_key = exact_vjp_event()
    extra_key["reason"] += " invented=1"
    malformed_exact_events.append(extra_key)
    duplicate_key = exact_vjp_event()
    duplicate_key["reason"] += " vjp_steps=300"
    malformed_exact_events.append(duplicate_key)
    double_space = exact_vjp_event()
    double_space["reason"] = double_space["reason"].replace(
        " phase=terminal", "  phase=terminal", 1
    )
    malformed_exact_events.append(double_space)
    tab_separated = exact_vjp_event()
    tab_separated["reason"] = tab_separated["reason"].replace(
        " phase=terminal", "\tphase=terminal", 1
    )
    malformed_exact_events.append(tab_separated)
    wrong_outcome = exact_vjp_event(outcome="plausible_but_unknown")
    malformed_exact_events.append(wrong_outcome)
    wrong_status = exact_vjp_event(status="skipped")
    malformed_exact_events.append(wrong_status)
    malformed_exact_rejected = []
    for malformed_event in malformed_exact_events:
        valid, failures, _w, _c = synthetic_exact_vjp_result(
            [("NY_PGD_VJP_K", "52")], [malformed_event]
        )
        malformed_exact_rejected.append(
            not valid
            and any(failure["check"] == "exact_vjp_parameter_engaged"
                    for failure in failures)
        )
    check("exact-VJP v1 rejects missing/extra/duplicate/whitespace/enum/status forgeries",
          all(malformed_exact_rejected), repr(malformed_exact_rejected))
    check("exact-VJP v1 rejects a missing ODI movement counter",
          malformed_exact_rejected[1], repr(malformed_exact_rejected))
    check("exact-VJP v1 rejects a missing ODI completed-phase counter",
          malformed_exact_rejected[2], repr(malformed_exact_rejected))
    poisoned_valid, poisoned_failures, _w, _c = synthetic_exact_vjp_result(
        [("NY_PGD_VJP_K", "52")], [exact_vjp_event(), missing_key]
    )
    check("a malformed later invocation poisons earlier valid exact-VJP evidence",
          not poisoned_valid
          and any(failure["check"] == "exact_vjp_parameter_engaged"
                  for failure in poisoned_failures),
          repr(poisoned_failures))

    progress_only = exact_vjp_event(phase="progress", outcome="in_progress")
    nonmonotone_progress = exact_vjp_event(
        phase="progress", outcome="in_progress", vjp_steps="301"
    )
    crossed_progress = exact_vjp_event(
        phase="progress", outcome="in_progress", odi_eta="1"
    )
    nonmonotone_move_progress = exact_vjp_event(
        phase="progress", outcome="in_progress", odi_move_count="101"
    )
    nonmonotone_completed_phase_progress = exact_vjp_event(
        phase="progress", outcome="in_progress", odi_completed_phase_count="2"
    )
    progress_forgeries = [
        [progress_only],
        [exact_vjp_event(), progress_only],
        [nonmonotone_progress, exact_vjp_event()],
        [crossed_progress, exact_vjp_event()],
        [nonmonotone_move_progress, exact_vjp_event()],
        [nonmonotone_completed_phase_progress, exact_vjp_event()],
    ]
    rejected_progress_forgeries = []
    for forged_events in progress_forgeries:
        valid, failures, _w, _c = synthetic_exact_vjp_result(
            [("NY_PGD_VJP_K", "52")], forged_events
        )
        rejected_progress_forgeries.append(
            not valid
            and any(failure["check"] == "exact_vjp_parameter_engaged"
                    for failure in failures)
        )
    check("progress cannot replace, outlive, exceed, or cross a terminal invocation",
          all(rejected_progress_forgeries), repr(rejected_progress_forgeries))
    check("an earlier terminal plus trailing progress poisons the whole row",
          rejected_progress_forgeries[1], repr(rejected_progress_forgeries))
    check("counters may not regress within one exact-VJP invocation",
          rejected_progress_forgeries[2], repr(rejected_progress_forgeries))
    check("ODI movement evidence may not regress within an invocation",
          rejected_progress_forgeries[-2], repr(rejected_progress_forgeries))
    check("ODI completed-phase evidence may not regress within an invocation",
          rejected_progress_forgeries[-1], repr(rejected_progress_forgeries))

    def exact_axis_rejected(name, token, event):
        valid, failures, _warnings, _counts = synthetic_exact_vjp_result(
            [(name, token)], [event]
        )
        return (
            not valid
            and any(failure["check"] == "exact_vjp_parameter_engaged"
                    for failure in failures)
        ), failures

    tail_width_valid, tail_width_failures, _w, _c = synthetic_exact_vjp_result(
        [("NY_PGD_VJP_K", "52")],
        [exact_vjp_event(
            total_restarts="100", restarts_started="100", width_final="48",
            width_binding="restart_cap",
        )],
    )
    inert_exact_cases = [
        ("selector clean-point candidate", *exact_axis_rejected(
            "NY_PGD_EXACT_BATCHED", "1",
            exact_vjp_not_run_event(
                status="ran", outcome="candidate", inactive_reason="none",
            ),
        )),
        ("selector structural bypass", *exact_axis_rejected(
            "NY_PGD_EXACT_BATCHED", "1", exact_vjp_not_run_event(),
        )),
        ("batch enabled without a successful wide VJP", *exact_axis_rejected(
            "NY_PGD_VJP_BATCH", "1",
            exact_vjp_event(
                vjp_steps="0", odi_vjp_steps="0", recycle_count="0",
                odi_completed_phase_count="0", odi_move_count="0",
                exploit_opportunity_count="0", exploit_count="0",
                hinge_extra_count="0", outcome="stalled",
            ),
        )),
        ("width larger than total restarts", *exact_axis_rejected(
            "NY_PGD_VJP_K", "52",
            exact_vjp_event(
                total_restarts="32", restarts_started="32", waves="1",
                width_final="32", width_max_active="32", width_binding="restart_cap",
                vjp_steps="10", exploit_opportunity_count="0", exploit_count="0",
                hinge_extra_count="0",
            ),
        )),
        ("width equal to the contextual route default", *exact_axis_rejected(
            "NY_PGD_VJP_K", "64",
            exact_vjp_event(
                width_configured="64", width_initial="64", width_final="64",
                width_max_active="64", total_restarts="128",
                restarts_started="128",
            ),
        )),
        ("adapter-halved width", *exact_axis_rejected(
            "NY_PGD_VJP_K", "52",
            exact_vjp_event(
                total_restarts="100", restarts_started="100", waves="3",
                width_final="26", width_binding="adapter_halved", width_retries="1",
            ),
        )),
        ("saturated legacy lanes", *exact_axis_rejected(
            "NY_PGD_LEGACY_LANES", "20",
            exact_vjp_event(legacy_lanes_used="7"),
        )),
        ("legacy gate without a successful VJP", *exact_axis_rejected(
            "NY_PGD_LEGACY_LANES", "20",
            exact_vjp_event(
                vjp_steps="0", odi_vjp_steps="0", recycle_count="0",
                odi_completed_phase_count="0", odi_move_count="0",
                exploit_opportunity_count="0", exploit_count="0",
                hinge_extra_count="0", outcome="stalled",
            ),
        )),
        ("zero legacy gate without a multi-lane wave", *exact_axis_rejected(
            "NY_PGD_LEGACY_LANES", "0",
            exact_vjp_event(
                width_configured="1", width_initial="1", width_final="1",
                width_max_active="1", total_restarts="2", restarts_started="2",
                waves="2", vjp_steps="10", legacy_lanes_configured="0",
                legacy_lanes_used="0",
            ),
        )),
        ("ODI step count not reached", *exact_axis_rejected(
            "NY_PGD_ODI_STEPS", "16",
            exact_vjp_event(
                odi_steps_configured="16", odi_vjp_steps="8",
                odi_completed_phase_count="0",
            ),
        )),
        ("ODI calls accumulated without a completed phase", *exact_axis_rejected(
            "NY_PGD_ODI_STEPS", "16",
            exact_vjp_event(
                odi_steps_configured="16", odi_vjp_steps="16",
                odi_completed_phase_count="0",
            ),
        )),
        ("ODI eta without an actual ODI VJP", *exact_axis_rejected(
            "NY_PGD_ODI_ETA", "0.5",
            exact_vjp_event(
                odi_vjp_steps="0", odi_completed_phase_count="0",
                odi_move_count="0",
            ),
        )),
        ("ODI eta without an actual coordinate move", *exact_axis_rejected(
            "NY_PGD_ODI_ETA", "0.5", exact_vjp_event(odi_move_count="0"),
        )),
        ("wave cap equal to contextual default", *exact_axis_rejected(
            "NY_PGD_VJP_STEPS", "1000",
            exact_vjp_event(
                wave_steps_configured="1000", wave_steps_default="1000",
                wave_steps_effective="1000", vjp_steps="1000",
            ),
        )),
        ("wave cap not reached", *exact_axis_rejected(
            "NY_PGD_VJP_STEPS", "250", exact_vjp_event(vjp_steps="249"),
        )),
        ("lane window without recycling", *exact_axis_rejected(
            "NY_PGD_LANE_WINDOW", "50", exact_vjp_event(recycle_count="0"),
        )),
        ("zero lane window without counterfactual opportunity", *exact_axis_rejected(
            "NY_PGD_LANE_WINDOW", "0",
            exact_vjp_event(lane_window="0", recycle_count="0"),
        )),
        ("enabled exploit without action", *exact_axis_rejected(
            "NY_PGD_EXPLOIT", "1",
            exact_vjp_event(exploit_opportunity_count="1", exploit_count="0"),
        )),
        ("disabled exploit without an opportunity", *exact_axis_rejected(
            "NY_PGD_EXPLOIT", "0",
            exact_vjp_event(
                exploit_enabled="false", exploit_opportunity_count="0",
                exploit_count="0",
            ),
        )),
        ("positive hinge without an added row", *exact_axis_rejected(
            "NY_PGD_HINGE_TAU", "0.001", exact_vjp_event(hinge_extra_count="0"),
        )),
        ("zero hinge without counterfactual opportunity", *exact_axis_rejected(
            "NY_PGD_HINGE_TAU", "0",
            exact_vjp_event(hinge_tau="0", hinge_extra_count="0"),
        )),
    ]
    check("tail width is live while clamped or unconsumed exact controls are rejected",
          tail_width_valid and all(rejected for _label, rejected, _failures
                                   in inert_exact_cases),
          repr((tail_width_failures, inert_exact_cases)))
    inert_exact_by_label = {
        label: rejected for label, rejected, _failures in inert_exact_cases
    }
    check("ODI eta requires a completed f32 coordinate movement",
          inert_exact_by_label["ODI eta without an actual coordinate move"])
    check("positive ODI steps require one fully completed configured phase",
          inert_exact_by_label["ODI calls accumulated without a completed phase"])
    check("zero legacy lanes require an exercised multi-lane wave",
          inert_exact_by_label["zero legacy gate without a multi-lane wave"])
    check("zero lane-window and hinge controls stay uncredited without opportunities",
          inert_exact_by_label["zero lane window without counterfactual opportunity"]
          and inert_exact_by_label["zero hinge without counterfactual opportunity"])

    arm = [("NY_MARGIN_ROW_GPU", "1")]
    rows = [_clean_row(arm), _clean_row(arm, verdict="timeout")]
    rows[1]["instance_index"] = 1
    rows[1]["occurrence"] = 1
    valid, failures, _warnings, counts = validate_run(
        arm, 2025, "cifar100", rows, _clean_manifest(arm=arm), _summary_of(rows),
        engagement_arm=[])
    check("a clean run is VALID", valid, str(failures))
    check("counts are recomputed from the bank",
          counts["unsat"] == 1 and counts["timeout"] == 1)

    malformed_receipts = []
    for mutate in (
        lambda receipt: receipt.__setitem__("schema", "ny-levers/receipt/v2"),
        lambda receipt: receipt.__setitem__("env_present", 999),
        lambda receipt: receipt.__setitem__(
            "levers",
            [entry for entry in receipt["levers"]
             if entry.get("name") != "NY_MARGIN_ROW_GPU"],
        ),
    ):
        malformed = [json.loads(json.dumps(row)) for row in rows]
        mutate(malformed[0]["flight"]["levers"]["receipt"])
        malformed_valid, malformed_failures, _w, _c = validate_run(
            arm, 2025, "cifar100", malformed, _clean_manifest(arm=arm),
            _summary_of(malformed))
        malformed_receipts.append(
            not malformed_valid
            and any(f["check"] == "receipt_v3_contract" for f in malformed_failures)
        )
    check("T1 rejects downgraded, inconsistent, or arm-incomplete receipts",
          all(malformed_receipts), repr(malformed_receipts))

    malformed_zero_events = []
    for binding, inactive_reason in [
        ("allocator_zero", "frame_skip"),
        ("allocator_zero+frame_zero", "allocator_skip"),
        ("allocator_zero+wall_cap", "allocator_skip"),
    ]:
        malformed_zero_row = _clean_row(arm)
        malformed_zero_row["flight"]["events"] = [upfront_execution_event(
            status="skipped", requested_window="skip",
            budget_authority="committed_cap", binding=binding,
            budget_competition="not_evaluated",
            effective_budget_secs="none", inactive_reason=inactive_reason,
        )]
        malformed_zero_events.append(_machine_event_fields(
            malformed_zero_row, "upfront_attack_execution",
            "ny-cli/upfront-execution/v2") == [])
    check("execution schema rejects crossed or combined zero evidence",
          all(malformed_zero_events), repr(malformed_zero_events))

    missing_v2_policy = upfront_policy_event()
    missing_v2_policy["reason"] = " ".join(
        token for token in missing_v2_policy["reason"].split()
        if not token.startswith("bnn_wall_cap_secs=")
    )
    extra_v2_execution = upfront_execution_event()
    extra_v2_execution["reason"] += " unversioned_extra=1"
    exact_schema_row = _clean_row([])
    exact_schema_row["flight"]["events"] = [missing_v2_policy, extra_v2_execution]
    check("v2 policy and execution schemas reject missing or extra fields",
          not _machine_event_fields(
              exact_schema_row, "upfront_attack_policy", "ny-cli/upfront-policy/v2")
          and not _machine_event_fields(
              exact_schema_row, "upfront_attack_execution",
              "ny-cli/upfront-execution/v2"))
    crossed_route_row = _clean_row([])
    crossed_route_row["flight"]["events"] = [
        upfront_policy_event(route="forced_by_environment"),
        upfront_execution_event(admission="forced"),
    ]
    check("policy masters and execution admissions bind to realizable routes",
          not _machine_event_fields(
              crossed_route_row, "upfront_attack_policy",
              "ny-cli/upfront-policy/v2")
          and not _machine_event_fields(
              crossed_route_row, "upfront_attack_execution",
              "ny-cli/upfront-execution/v2"))

    impossible_execution_stages = []
    for event in (
        upfront_execution_event(effective_budget_secs="none"),
        upfront_execution_event(
            status="skipped", binding="wall_cap",
            budget_competition="not_evaluated", effective_budget_secs="none",
            inactive_reason="private_base_below_min"),
        upfront_execution_event(
            admission="none", sign_net="unknown", budget_authority="none",
            binding="none", budget_competition="not_evaluated",
            effective_budget_secs="none"),
        upfront_execution_event(
            status="skipped", admission="none", sign_net="unknown",
            budget_authority="none", binding="none",
            budget_competition="not_evaluated", effective_budget_secs="none",
            inactive_reason="wrapper_disabled"),
        upfront_execution_event(
            status="skipped", route="disabled_by_preset", admission="none",
            sign_net="unknown", budget_authority="none", binding="none",
            budget_competition="not_evaluated", effective_budget_secs="none",
            inactive_reason="vnnlib_load_failed"),
        upfront_execution_event(
            status="skipped", route="forced_by_environment", admission="none",
            sign_net="unknown", budget_authority="none", binding="none",
            budget_competition="not_evaluated", effective_budget_secs="none",
            inactive_reason="admission_declined"),
        upfront_execution_event(
            status="skipped", requested_window="cap",
            requested_cap_secs="5.000000", budget_authority="committed_cap",
            binding="live_remaining", budget_competition="not_evaluated",
            remaining_secs="3.000001", effective_budget_secs="none",
            inactive_reason="committed_budget_unusable"),
        upfront_execution_event(
            status="skipped", requested_window="cap",
            requested_cap_secs="0.000000", budget_authority="committed_cap",
            binding="live_remaining", budget_competition="not_evaluated",
            remaining_secs="2.000000", effective_budget_secs="none",
            inactive_reason="committed_budget_unusable"),
        upfront_execution_event(
            status="skipped", binding="live_remaining",
            budget_competition="not_evaluated", remaining_secs="100.000000",
            effective_budget_secs="none",
            inactive_reason="private_live_budget_unusable"),
    ):
        impossible_stage_row = _clean_row([])
        impossible_stage_row["flight"]["events"] = [event]
        impossible_execution_stages.append(not _machine_event_fields(
            impossible_stage_row, "upfront_attack_execution",
            "ny-cli/upfront-execution/v2"))
    check("v2 execution stages require producer-realizable admission and budgets",
          all(impossible_execution_stages), repr(impossible_execution_stages))
    plausible_unusable_stages = []
    for event in (
        upfront_execution_event(
            status="skipped", requested_window="cap",
            requested_cap_secs="5.000000", budget_authority="committed_cap",
            binding="live_remaining", budget_competition="not_evaluated",
            remaining_secs="3.000000", effective_budget_secs="none",
            inactive_reason="committed_budget_unusable"),
        upfront_execution_event(
            status="skipped", binding="live_remaining",
            budget_competition="not_evaluated", remaining_secs="3.000000",
            effective_budget_secs="none",
            inactive_reason="private_live_budget_unusable"),
    ):
        plausible_row = _clean_row([])
        plausible_row["flight"]["events"] = [event]
        plausible_unusable_stages.append(bool(_machine_event_fields(
            plausible_row, "upfront_attack_execution",
            "ny-cli/upfront-execution/v2")))
    check("rounded three-second receipts retain a realizable unusable clock",
          all(plausible_unusable_stages), repr(plausible_unusable_stages))

    cap_arm = [("NY_UPFRONT_ATTACK_CAP", "1")]
    valid, failures, _w, _c = synthetic_upfront_result(cap_arm, [])
    check("T1 rejects an upfront cap arm with no execution evidence",
          not valid and any(f["check"] == "upfront_parameter_engaged" for f in failures),
          str(failures))
    refused_cap_evidence = []
    for event in (
        upfront_execution_event(binding="budget_fraction"),
        upfront_execution_event(
            requested_window="cap", requested_cap_secs="5.000000",
            budget_authority="committed_cap", binding="live_remaining",
        ),
    ):
        valid, failures, _w, _c = synthetic_upfront_result(cap_arm, [event])
        refused_cap_evidence.append(
            not valid
            and any(f["check"] == "upfront_parameter_engaged" for f in failures)
        )
    check("another control or live clamp cannot credit the requested cap",
          all(refused_cap_evidence), repr(refused_cap_evidence))
    valid, failures, _w, _c = synthetic_upfront_result(
        cap_arm,
        [upfront_execution_event(
            binding="wall_cap", effective_budget_secs="1.000000")],
        upfront_policy_event(wall_cap_secs="1"),
    )
    check("an ordinary private wall-cap binding credits the cap",
          valid, str(failures))
    mismatched_cap_policy, mismatched_cap_failures, _w, _c = synthetic_upfront_result(
        cap_arm,
        [upfront_execution_event(
            route="forced_by_preset", admission="forced",
            binding="wall_cap", remaining_secs="unbounded",
            effective_budget_secs="8.000000")],
        upfront_policy_event(
            master="forced_by_preset", route="forced_by_preset",
            lower_master="forced_by_preset", lower_route="forced_by_preset",
            wall_cap_secs="8"),
    )
    check("ordinary execution cannot credit a value absent from its policy receipt",
          not mismatched_cap_policy
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in mismatched_cap_failures),
          str(mismatched_cap_failures))
    valid, failures, _w, _c = synthetic_upfront_result(
        cap_arm,
        [upfront_execution_event(
            requested_window="cap", requested_cap_secs="1.000000",
            budget_authority="committed_cap", binding="wall_cap",
            effective_budget_secs="1.000000",
        )],
        upfront_policy_event(wall_cap_secs="1"),
    )
    check("a committed cap preserves and credits its upstream wall-cap binding",
          valid, str(failures))

    committed_frac_arm = [("NY_UPFRONT_ATTACK_FRAC", "0.5")]
    valid, failures, _w, _c = synthetic_upfront_result(
        committed_frac_arm,
        [upfront_execution_event(
            requested_window="cap", requested_cap_secs="4.000000",
            budget_authority="committed_cap", binding="budget_fraction+live_remaining",
            route="forced_by_preset", admission="forced",
            remaining_secs="7.000000",
        )],
        upfront_policy_event(
            master="forced_by_preset", route="forced_by_preset",
            lower_master="forced_by_preset", lower_route="forced_by_preset",
            budget_fraction="0.5"),
    )
    check("an exact committed/live tie preserves fractional lineage",
          valid, str(failures))

    tied_arm = [
        ("NY_UPFRONT_ATTACK_CAP", "1"),
        ("NY_UPFRONT_ATTACK_FRAC", "0.5"),
    ]
    valid, failures, _w, _c = synthetic_upfront_result(
        tied_arm,
        [upfront_execution_event(
            binding="budget_fraction+wall_cap", remaining_secs="5.000000",
            effective_budget_secs="1.000000")],
        upfront_policy_event(wall_cap_secs="1", budget_fraction="0.5"),
    )
    check("one exact private tie can establish both cap and fraction engagement",
          valid, str(failures))
    near_tie_fraction, near_tie_failures, _w, _c = synthetic_upfront_result(
        committed_frac_arm,
        [upfront_execution_event(
            route="forced_by_preset", admission="forced",
            binding="budget_fraction", remaining_secs="19.000000",
            effective_budget_secs="8.000000")],
        upfront_policy_event(
            master="forced_by_preset", route="forced_by_preset",
            lower_master="forced_by_preset", lower_route="forced_by_preset",
            budget_fraction="0.5"),
    )
    check("rounded private receipts retain any nanosecond-plausible exact winner",
          near_tie_fraction, str(near_tie_failures))
    rounded_frame_arm = [
        ("NY_UPFRONT_ATTACK", "1"),
        ("NY_UPFRONT_ATTACK_FRAC", "0.2"),
    ]
    rounded_frame, rounded_frame_failures, _w, _c = synthetic_upfront_result(
        rounded_frame_arm,
        [upfront_execution_event(
            route="forced_by_environment", admission="forced",
            requested_window="cap", requested_cap_secs="8.000000",
            budget_authority="committed_cap", binding="budget_fraction",
            remaining_secs="50.000000", effective_budget_secs="8.000000")],
        upfront_policy_event(
            master="forced_by_environment", route="forced_by_environment",
            lower_master="contextual", lower_route="auto_disjunction",
            wall_cap_secs="8", budget_fraction="0.2"),
    )
    check("frame planning admits rounded singleton winners beside exact ties",
          rounded_frame, str(rounded_frame_failures))
    correlated_frame_forgery, correlated_frame_failures, _w, _c = (
        synthetic_upfront_result(
            rounded_frame_arm,
            [upfront_execution_event(
                route="forced_by_environment", admission="forced",
                requested_window="cap", requested_cap_secs="8.000000",
                budget_authority="committed_cap",
                binding="budget_fraction+live_remaining",
                budget_competition="not_sign", remaining_secs="10.999999",
                effective_budget_secs="8.000000")],
            upfront_policy_event(
                master="forced_by_environment", route="forced_by_environment",
                lower_master="contextual", lower_route="auto_disjunction",
                wall_cap_secs="8", budget_fraction="0.2"),
        )
    )
    check("frame lineage and live clamping share one exact cap Duration",
          not correlated_frame_forgery
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in correlated_frame_failures),
          str(correlated_frame_failures))

    rust_float_arm = [
        ("NY_UPFRONT_ATTACK", "1"),
        ("NY_UPFRONT_ATTACK_FRAC", "1"),
    ]
    rust_float_policy = upfront_policy_event(
        master="forced_by_environment", route="forced_by_environment",
        lower_master="contextual", lower_route="auto_disjunction",
        wall_cap_secs="8", auto_wall_cap_secs="4", budget_fraction="1")
    rust_float_forgery, rust_float_failures, _w, _c = synthetic_upfront_result(
        rust_float_arm,
        [upfront_execution_event(
            route="forced_by_environment", admission="forced",
            binding="budget_fraction", budget_competition="not_sign",
            remaining_secs="3.999998", effective_budget_secs="0.999999")],
        rust_float_policy,
    )
    rust_float_positive, rust_float_positive_failures, _w, _c = (
        synthetic_upfront_result(
            rust_float_arm,
            [upfront_execution_event(
                route="forced_by_environment", admission="forced",
                binding="budget_fraction", budget_competition="not_sign",
                remaining_secs="3.999998", effective_budget_secs="0.999998")],
            rust_float_policy,
        )
    )
    check("budget lineage mirrors Rust Duration floating arithmetic",
          _rust_duration_format_ns(3_999_998_500) == "3.999999"
          and not rust_float_forgery and rust_float_positive
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in rust_float_failures),
          "%r / %r" % (rust_float_failures, rust_float_positive_failures))

    auto_arm = [("NY_UPFRONT_ATTACK_AUTO_CAP", "1")]
    valid_auto, auto_failures, _w, _c = synthetic_upfront_result(
        auto_arm,
        [upfront_execution_event(
            binding="auto_wall_cap", effective_budget_secs="1.000000")],
        upfront_policy_event(auto_wall_cap_secs="1"),
    )
    invalid_auto, invalid_auto_failures, _w, _c = synthetic_upfront_result(
        auto_arm,
        [upfront_execution_event(
            binding="bnn_budget_fraction", sign_net="true",
            budget_competition="bnn_wins", admission="auto_disjunction")],
        upfront_policy_event(auto_wall_cap_secs="1"),
    )
    committed_auto, committed_auto_failures, _w, _c = synthetic_upfront_result(
        auto_arm,
        [upfront_execution_event(
            requested_window="cap", requested_cap_secs="1.000000",
            budget_authority="committed_cap", binding="auto_wall_cap",
            effective_budget_secs="1.000000",
        )],
        upfront_policy_event(auto_wall_cap_secs="1"),
    )
    check("auto cap credits only non-Sign automatic admission with retained lineage",
          valid_auto and committed_auto and not invalid_auto
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in invalid_auto_failures),
          "%r / %r / %r"
          % (auto_failures, committed_auto_failures, invalid_auto_failures))

    bnn_category = "traffic_signs_recognition_2023"
    bnn_cap_arm = [("NY_BNN_ATTACK_CAP", "150")]
    bnn_cap_policy = upfront_policy_event(bnn_wall_cap_secs="150")
    bnn_cap_event = upfront_execution_event(
        sign_net="true", binding="bnn_wall_cap",
        budget_competition="bnn_wins", remaining_secs="400.000000",
        effective_budget_secs="150.000000",
    )
    valid_bnn_cap, bnn_cap_failures, _w, _c = synthetic_upfront_result(
        bnn_cap_arm, [bnn_cap_event], bnn_cap_policy, bnn_category)
    check("a BNN-winning exact Sign wall cap credits the governed BNN cap",
          valid_bnn_cap, str(bnn_cap_failures))

    bnn_frac_arm = [("NY_BNN_ATTACK_FRAC", "0.75")]
    bnn_frac_policy = upfront_policy_event(bnn_budget_fraction="0.75")
    bnn_frac_event = upfront_execution_event(
        sign_net="true", binding="bnn_budget_fraction",
        budget_competition="bnn_wins", effective_budget_secs="72.750000",
    )
    valid_bnn_frac, bnn_frac_failures, _w, _c = synthetic_upfront_result(
        bnn_frac_arm, [bnn_frac_event], bnn_frac_policy, bnn_category)
    check("a BNN-winning exact Sign fraction credits the governed BNN fraction",
          valid_bnn_frac, str(bnn_frac_failures))

    bnn_tied_arm = [
        ("NY_BNN_ATTACK_CAP", "150"),
        ("NY_BNN_ATTACK_FRAC", "0.75"),
    ]
    bnn_tied_policy = upfront_policy_event(
        bnn_wall_cap_secs="150", bnn_budget_fraction="0.75")
    inner_tie, inner_tie_failures, _w, _c = synthetic_upfront_result(
        bnn_tied_arm,
        [upfront_execution_event(
            sign_net="true", binding="bnn_budget_fraction+bnn_wall_cap",
            budget_competition="bnn_wins", remaining_secs="203.000000",
            effective_budget_secs="150.000000")],
        bnn_tied_policy,
        bnn_category,
    )
    outer_tie_arm = [
        ("NY_UPFRONT_ATTACK_CAP", "25"),
        ("NY_UPFRONT_ATTACK_FRAC", "0.5"),
        ("NY_BNN_ATTACK_FRAC", "0.25"),
    ]
    outer_tie_policy = upfront_policy_event(
        wall_cap_secs="25", budget_fraction="0.5", bnn_budget_fraction="0.25")
    outer_tie, outer_tie_failures, _w, _c = synthetic_upfront_result(
        outer_tie_arm,
        [upfront_execution_event(
            sign_net="true",
            binding="wall_cap+bnn_budget_fraction", remaining_secs="103.000000",
            budget_competition="tie", effective_budget_secs="25.000000")],
        outer_tie_policy,
        bnn_category,
    )
    check("an inner BNN tie credits both controls but an outer max tie credits neither",
          inner_tie and not outer_tie
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in outer_tie_failures),
          "%r / %r" % (inner_tie_failures, outer_tie_failures))

    frame_bnn, frame_bnn_failures, _w, _c = synthetic_upfront_result(
        bnn_cap_arm,
        [upfront_execution_event(
            sign_net="true", requested_window="cap",
            requested_cap_secs="150.000000", budget_authority="committed_cap",
            binding="bnn_wall_cap+live_remaining", budget_competition="bnn_wins",
            remaining_secs="153.000000", effective_budget_secs="150.000000")],
        bnn_cap_policy,
        bnn_category,
    )
    refused_bnn = []
    for event in (
        upfront_execution_event(
            sign_net="true", requested_window="cap",
            requested_cap_secs="150.000000", budget_authority="committed_cap",
            binding="committed_cap", budget_competition="not_evaluated",
            remaining_secs="400.000000", effective_budget_secs="150.000000"),
        upfront_execution_event(
            sign_net="true", requested_window="cap",
            requested_cap_secs="150.000000", budget_authority="committed_cap",
            binding="live_remaining", budget_competition="not_evaluated",
            remaining_secs="123.000000", effective_budget_secs="120.000000"),
        upfront_execution_event(
            sign_net="true", binding="bnn_budget_fraction",
            budget_competition="bnn_wins", effective_budget_secs="48.500000"),
        upfront_execution_event(
            status="skipped", sign_net="true", binding="bnn_wall_cap",
            budget_competition="bnn_wins", remaining_secs="400.000000",
            effective_budget_secs="150.000000",
            inactive_reason="trusted_forward_unavailable"),
    ):
        valid, failures, _w, _c = synthetic_upfront_result(
            bnn_cap_arm, [event], bnn_cap_policy, bnn_category)
        refused_bnn.append(
            not valid
            and any(f["check"] == "upfront_parameter_engaged" for f in failures)
        )
    check("only exact private/frame BNN-winning evidence survives allocator and live clamps",
          frame_bnn and all(refused_bnn),
          "%r / %r" % (frame_bnn_failures, refused_bnn))
    impossible_frame_lineage = []
    for remaining_token in ("100.000000", "unbounded"):
        valid, failures, _w, _c = synthetic_upfront_result(
            bnn_frac_arm,
            [upfront_execution_event(
                sign_net="true", requested_window="cap",
                requested_cap_secs="1.000000",
                budget_authority="committed_cap",
                binding="bnn_budget_fraction", budget_competition="bnn_wins",
                remaining_secs=remaining_token,
                effective_budget_secs="1.000000")],
            bnn_frac_policy,
            bnn_category,
        )
        impossible_frame_lineage.append(
            not valid
            and any(f["check"] == "upfront_parameter_engaged" for f in failures)
        )
    check("exact frames require a realizable gated planning-clock decision",
          all(impossible_frame_lineage), repr(impossible_frame_lineage))
    beyond_duration_frame, beyond_duration_failures, _w, _c = (
        synthetic_upfront_result(
            cap_arm,
            [upfront_execution_event(
                route="forced_by_preset", admission="forced",
                requested_window="cap", requested_cap_secs="1.000000",
                budget_authority="committed_cap", binding="wall_cap",
                remaining_secs="100.000000", effective_budget_secs="1.000000")],
            upfront_policy_event(
                master="forced_by_preset", route="forced_by_preset",
                lower_master="forced_by_preset", lower_route="forced_by_preset",
                wall_cap_secs="1", budget_fraction="1e-20"),
        )
    )
    check("frame lineage cannot require a planning Duration above Duration::MAX",
          not beyond_duration_frame
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in beyond_duration_failures),
          str(beyond_duration_failures))

    malformed_bnn_competitions = []
    for event in (
        upfront_execution_event(
            sign_net="true", binding="wall_cap+bnn_wall_cap",
            budget_competition="bnn_wins", remaining_secs="400.000000",
            effective_budget_secs="150.000000"),
        upfront_execution_event(
            sign_net="true", binding="bnn_wall_cap",
            budget_competition="tie", remaining_secs="400.000000",
            effective_budget_secs="150.000000"),
        upfront_execution_event(
            status="skipped", sign_net="true", binding="bnn_wall_cap",
            budget_competition="bnn_wins", remaining_secs="400.000000",
            effective_budget_secs="150.000000",
            inactive_reason="private_base_below_min"),
        upfront_execution_event(
            status="skipped", sign_net="true", binding="bnn_wall_cap",
            budget_competition="bnn_wins", remaining_secs="400.000000",
            effective_budget_secs="150.000000",
            inactive_reason="private_effective_below_min"),
        upfront_execution_event(
            sign_net="true", requested_window="cap",
            requested_cap_secs="150.000000", budget_authority="committed_cap",
            binding="bnn_wall_cap+committed_cap",
            budget_competition="bnn_wins", remaining_secs="400.000000",
            effective_budget_secs="150.000000"),
        upfront_execution_event(
            consultation="probe_first", sign_net="true", requested_window="cap",
            requested_cap_secs="150.000000", budget_authority="probe_cap",
            binding="bnn_wall_cap+probe_cap", budget_competition="bnn_wins",
            remaining_secs="400.000000", effective_budget_secs="150.000000"),
        upfront_execution_event(
            sign_net="true", binding="bnn_wall_cap+live_remaining",
            budget_competition="bnn_wins", remaining_secs="153.000000",
            effective_budget_secs="150.000000"),
    ):
        malformed_row = _clean_row(bnn_cap_arm, category=bnn_category)
        malformed_row["flight"]["events"] = [event]
        malformed_bnn_competitions.append(not _machine_event_fields(
            malformed_row, "upfront_attack_execution",
            "ny-cli/upfront-execution/v2"))
    check("v2 rejects impossible BNN winner, tie, stage, and authority bindings",
          all(malformed_bnn_competitions), repr(malformed_bnn_competitions))

    forged_private_stages = []
    for event in (
        upfront_execution_event(
            status="skipped", requested_window="cap",
            requested_cap_secs="1.000000", budget_authority="committed_cap",
            binding="wall_cap", budget_competition="not_evaluated",
            inactive_reason="private_base_below_min"),
        upfront_execution_event(
            status="skipped", requested_window="cap",
            requested_cap_secs="1.000000", budget_authority="committed_cap",
            binding="auto_wall_cap", budget_competition="not_sign",
            inactive_reason="private_effective_below_min"),
    ):
        forged_row = _clean_row(cap_arm)
        forged_row["flight"]["events"] = [event]
        forged_private_stages.append(not _machine_event_fields(
            forged_row, "upfront_attack_execution",
            "ny-cli/upfront-execution/v2"))
    check("private-stage refusals require private budget authority and window",
          all(forged_private_stages), repr(forged_private_stages))

    mismatched_bnn, mismatched_bnn_failures, _w, _c = synthetic_upfront_result(
        bnn_cap_arm,
        [upfront_execution_event(
            sign_net="true", binding="bnn_wall_cap",
            budget_competition="bnn_wins", remaining_secs="500.000000",
            effective_budget_secs="240.000000")],
        upfront_policy_event(bnn_wall_cap_secs="240"),
        bnn_category,
    )
    check("BNN execution cannot credit a requested value absent from its policy receipt",
          not mismatched_bnn
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in mismatched_bnn_failures),
          str(mismatched_bnn_failures))
    forged_bnn_lineage, forged_bnn_failures, _w, _c = synthetic_upfront_result(
        bnn_cap_arm,
        [upfront_execution_event(
            sign_net="true", binding="bnn_wall_cap",
            budget_competition="bnn_wins", remaining_secs="100.000000",
            effective_budget_secs="48.500000")],
        bnn_cap_policy,
        bnn_category,
    )
    check("private BNN binding must equal the numeric winner in the receipt",
          not forged_bnn_lineage
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in forged_bnn_failures),
          str(forged_bnn_failures))
    correlated_bnn_arm = [("NY_BNN_ATTACK_CAP", "60")]
    correlated_bnn, correlated_bnn_failures, _w, _c = synthetic_upfront_result(
        correlated_bnn_arm,
        [upfront_execution_event(
            sign_net="true", binding="bnn_wall_cap",
            budget_competition="bnn_wins", remaining_secs="103.000000",
            effective_budget_secs="60.000000")],
        upfront_policy_event(
            wall_cap_secs="60", budget_fraction="0.6",
            bnn_wall_cap_secs="60", bnn_budget_fraction="0.6"),
        bnn_category,
    )
    check("ordinary and BNN winners share one correlated remaining-wall clock",
          not correlated_bnn
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in correlated_bnn_failures),
          str(correlated_bnn_failures))

    legacy_policy = upfront_policy_event(version="v1")
    legacy_execution = upfront_execution_event(
        version="v1", sign_net="true", binding="bnn_policy",
        effective_budget_secs="48.500000")
    legacy_row = _clean_row(bnn_cap_arm, category=bnn_category)
    legacy_row["flight"]["events"] = [legacy_policy, legacy_execution]
    legacy_parses = bool(_machine_event_fields(
        legacy_row, "upfront_attack_execution", "ny-cli/upfront-execution/v1"))
    legacy_valid, legacy_failures, _w, _c = synthetic_upfront_result(
        bnn_cap_arm, [legacy_execution], legacy_policy, bnn_category)
    check("legacy v1 BNN policy remains parseable but cannot credit new exact axes",
          legacy_parses and not legacy_valid
          and any(f["check"] == "upfront_parameter_engaged" for f in legacy_failures),
          str(legacy_failures))

    legacy_cap_policy = upfront_policy_event(version="v1", wall_cap_secs="1")
    legacy_cap_execution = upfront_execution_event(
        version="v1", binding="wall_cap", effective_budget_secs="1.000000")
    legacy_cap_valid, legacy_cap_failures, _w, _c = synthetic_upfront_result(
        cap_arm, [legacy_cap_execution], legacy_cap_policy)
    mixed_version_valid, mixed_version_failures, _w, _c = synthetic_upfront_result(
        cap_arm, [legacy_cap_execution], upfront_policy_event())
    check("v1 ordinary evidence remains valid but policy/execution versions never cross-pair",
          legacy_cap_valid and not mixed_version_valid
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in mixed_version_failures),
          "%r / %r" % (legacy_cap_failures, mixed_version_failures))

    zero_cap_arm = [("NY_UPFRONT_ATTACK_CAP", "0")]
    valid, failures, _w, _c = synthetic_upfront_result(
        zero_cap_arm,
        [upfront_execution_event(
            status="skipped", binding="wall_cap", effective_budget_secs="0.000000",
            budget_competition="not_evaluated",
            inactive_reason="private_base_below_min")],
        upfront_policy_event(wall_cap_secs="0"),
    )
    check("a zero cap's directly attributable budget refusal is engagement",
          valid, str(failures))

    probe_arm = [("NY_UPFRONT_ATTACK_PROBE_FIRST", "1")]
    probe_policy = upfront_policy_event(probe_first="true", lower_probe_first="false")
    probe_started = upfront_execution_event(
        consultation="probe_first", requested_window="cap",
        requested_cap_secs="5.000000", budget_authority="probe_cap",
        binding="probe_cap", budget_competition="not_evaluated",
        effective_budget_secs="5.000000",
    )
    valid_probe, probe_failures, _w, _c = synthetic_upfront_result(
        probe_arm, [probe_started], probe_policy)
    invalid_probe, invalid_probe_failures, _w, _c = synthetic_upfront_result(
        probe_arm,
        [upfront_execution_event(
            status="skipped", consultation="probe_first", requested_window="cap",
            requested_cap_secs="5.000000", budget_authority="probe_cap",
            binding="probe_cap", budget_competition="not_evaluated",
            inactive_reason="trusted_forward_unavailable")],
        probe_policy,
    )
    mismatched_probe_policy, mismatched_probe_failures, _w, _c = (
        synthetic_upfront_result(
            probe_arm,
            [probe_started],
            upfront_policy_event(
                probe_first="false", lower_probe_first="false"),
        )
    )
    forged_probe_clamp, forged_probe_clamp_failures, _w, _c = (
        synthetic_upfront_result(
            probe_arm,
            [upfront_execution_event(
                consultation="probe_first", requested_window="cap",
                requested_cap_secs="5.000000", budget_authority="probe_cap",
                binding="probe_cap", budget_competition="not_evaluated",
                remaining_secs="7.999999", effective_budget_secs="5.000000")],
            probe_policy,
        )
    )
    check("probe-first credits only a probe-cap consultation that started model work",
          valid_probe and not invalid_probe and not mismatched_probe_policy
          and not forged_probe_clamp
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in invalid_probe_failures)
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in mismatched_probe_failures)
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in forged_probe_clamp_failures),
          "%r / %r / %r / %r"
          % (probe_failures, invalid_probe_failures, mismatched_probe_failures,
             forged_probe_clamp_failures))

    probe_off_arm = [("NY_UPFRONT_ATTACK_PROBE_FIRST", "0")]
    probe_off_policy = upfront_policy_event(
        probe_first="false", lower_probe_first="true")
    probe_disarmed = upfront_execution_event(
        status="skipped", consultation="probe_first", admission="none",
        sign_net="unknown", requested_window="skip",
        budget_authority="none", binding="none",
        budget_competition="not_evaluated",
        effective_budget_secs="none", inactive_reason="probe_disarmed",
    )
    valid_probe_off, probe_off_failures, _w, _c = synthetic_upfront_result(
        probe_off_arm,
        [probe_disarmed, upfront_execution_event()],
        probe_off_policy,
    )
    invalid_probe_off, invalid_probe_off_failures, _w, _c = synthetic_upfront_result(
        probe_off_arm, [probe_disarmed], probe_off_policy)
    forged_probe_off, forged_probe_off_failures, _w, _c = synthetic_upfront_result(
        probe_off_arm,
        [
            probe_disarmed,
            upfront_execution_event(
                admission="none", sign_net="unknown", budget_authority="none",
                binding="none", budget_competition="not_evaluated",
                effective_budget_secs="none"),
        ],
        probe_off_policy,
    )
    check("probe disarm requires proof the suppressed lane could start ordinary work",
          valid_probe_off and not invalid_probe_off and not forged_probe_off
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in invalid_probe_off_failures)
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in forged_probe_off_failures),
          "%r / %r / %r"
          % (probe_off_failures, invalid_probe_off_failures,
             forged_probe_off_failures))

    kill_arm = [("NY_UPFRONT_ATTACK", "0")]
    kill_execution = upfront_execution_event(
        status="skipped", route="disabled_by_environment", admission="none",
        sign_net="unknown", budget_authority="none", binding="none",
        budget_competition="not_evaluated",
        effective_budget_secs="none", inactive_reason="wrapper_disabled",
    )
    valid_kill, kill_failures, _w, _c = synthetic_upfront_result(
        kill_arm,
        [kill_execution],
        upfront_policy_event(
            master="disabled_by_environment", route="disabled_by_environment",
            lower_master="forced_by_preset", lower_route="forced_by_preset"),
    )
    noop_kill, noop_kill_failures, _w, _c = synthetic_upfront_result(
        kill_arm,
        [kill_execution],
        upfront_policy_event(
            master="disabled_by_environment", route="disabled_by_environment",
            lower_master="disabled_by_preset", lower_route="disabled_by_preset"),
    )
    contextual_kill, contextual_kill_failures, _w, _c = synthetic_upfront_result(
        kill_arm,
        [kill_execution],
        upfront_policy_event(
            master="disabled_by_environment", route="disabled_by_environment",
            lower_route="auto_disjunction"),
    )
    check("master kill requires a usable, forced lower counterfactual",
          valid_kill and not noop_kill
          and not contextual_kill
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in noop_kill_failures)
          and any(f["check"] == "upfront_parameter_engaged"
                  for f in contextual_kill_failures),
          "%r / %r / %r"
          % (kill_failures, noop_kill_failures, contextual_kill_failures))

    unsealed_arm_manifest = _clean_manifest(arm=arm)
    unsealed_arm_manifest.pop("arm")
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", rows, unsealed_arm_manifest, _summary_of(rows))
    check("a legacy manifest without an explicit-arm seal is INVALID",
          not valid and any(f["check"] == "manifest_arm_matches" for f in failures),
          str(failures))

    wrong_arm_manifest = _clean_manifest(arm=[("NY_MARGIN_ROW_GPU", "0")])
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", rows, wrong_arm_manifest, _summary_of(rows))
    check("the manifest explicit-arm seal must equal the requested canonical arm",
          not valid and any(f["check"] == "manifest_arm_matches" for f in failures),
          str(failures))

    unexpected_controls = []
    for name in ("NY_MARGIN_ROW_CLASSWISE", "RAYON_NUM_THREADS", "RAYON_RS_NUM_CPUS"):
        contaminated_rows = [json.loads(json.dumps(row)) for row in rows]
        contaminated_rows[0]["flight"]["ambient_env"][name] = "16"
        valid, failures, _w, _c = validate_run(
            arm, 2025, "cifar100", contaminated_rows, _clean_manifest(arm=arm),
            _summary_of(contaminated_rows))
        unexpected_controls.append(
            not valid
            and any(f["check"] == "child_env_has_no_unrequested_controls"
                    for f in failures)
        )
    check("child flight rejects unrequested NY_* and both RAYON controls but permits OMP",
          all(unexpected_controls), repr(unexpected_controls))

    legacy_rows = [json.loads(json.dumps(row)) for row in rows]
    legacy_rows[0]["flight"].pop("ambient_env_schema")
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", legacy_rows, _clean_manifest(arm=arm),
        _summary_of(legacy_rows))
    check("a readable pre-Rayon child flight cannot prove current ambient cleanliness",
          not valid and any(f["check"] == "child_ambient_provenance" for f in failures),
          str(failures))

    incomplete_rayon_rows = [json.loads(json.dumps(row)) for row in rows]
    incomplete_rayon_rows[0]["flight"]["ambient_env_schema"] = "ny-cli/ambient-env/v2"
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", incomplete_rayon_rows, _clean_manifest(arm=arm),
        _summary_of(incomplete_rayon_rows))
    check("ambient v2 child evidence cannot prove deprecated Rayon state was absent",
          not valid and any(f["check"] == "child_ambient_provenance" for f in failures),
          str(failures))

    legacy_manifest = _clean_manifest(arm=arm)
    legacy_manifest.pop("ambient_env_schema")
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", rows, legacy_manifest, _summary_of(rows))
    check("a readable pre-Rayon manifest cannot prove current ambient cleanliness",
          not valid and any(f["check"] == "manifest_ambient_provenance"
                            for f in failures), str(failures))

    incomplete_rayon_manifest = _clean_manifest(arm=arm)
    incomplete_rayon_manifest["ambient_env_schema"] = "ny-cli/ambient-env/v2"
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", rows, incomplete_rayon_manifest, _summary_of(rows))
    check("ambient v2 manifest evidence cannot prove deprecated Rayon state was absent",
          not valid and any(f["check"] == "manifest_ambient_provenance"
                            for f in failures), str(failures))

    unknown_manifest = _clean_manifest(arm=arm)
    unknown_manifest["ambient_env_schema"] = "ny-cli/ambient-env/v999"
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", rows, unknown_manifest, _summary_of(rows))
    check("an unknown ambient capture schema fails closed",
          not valid and any(f["check"] == "manifest_ambient_provenance"
                            for f in failures), str(failures))

    dirty = [json.loads(json.dumps(row)) for row in rows]
    dirty[0]["flight"]["levers"]["receipt"]["levers"][0]["rejected_raw"] = "true"
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", dirty, _clean_manifest(arm=arm), _summary_of(dirty))
    check("a rejected_raw row is INVALID",
          not valid and any(f["check"] == "rejected_raw" for f in failures))

    dirty = [json.loads(json.dumps(row)) for row in rows]
    dirty[0]["flight"]["levers"]["receipt"]["levers"][0]["ignored_raw"] = "true"
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", dirty, _clean_manifest(arm=arm), _summary_of(dirty))
    check("an ignored_raw row is INVALID",
          not valid and any(f["check"] == "ignored_raw" for f in failures))

    dirty = [json.loads(json.dumps(row)) for row in rows]
    dirty[0]["arm"] = [["NY_MARGIN_ROW_GPU", "0"]]
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", dirty, _clean_manifest(arm=arm), _summary_of(dirty))
    check("a row measured under a different arm is INVALID",
          not valid and any(f["check"] == "row_arm_matches" for f in failures))

    dirty = [json.loads(json.dumps(row)) for row in rows]
    dirty[0]["flight"]["ambient_env"] = {}
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", dirty, _clean_manifest(arm=arm), _summary_of(dirty))
    check("a child that never saw the arm is INVALID",
          not valid and any(f["check"] == "child_env_carries_arm" for f in failures))

    dirty = [json.loads(json.dumps(row)) for row in rows]
    dirty[0]["flight"]["levers"] = {"status": "not_materialized"}
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", dirty, _clean_manifest(arm=arm), _summary_of(dirty))
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
        arm, 2025, "cifar100", sat_rows, _clean_manifest(arm=arm), _summary_of(sat_rows))
    check("a sat row without a witness is INVALID",
          not valid and any(f["check"] == "sat_rows_without_witness" for f in failures))

    manifest = _clean_manifest(arm=arm)
    manifest["ambient_env"] = {"NY_BAB_RESNET_WIDE": "1"}
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", rows, manifest, _summary_of(rows))
    check("an ambient lever nobody requested is INVALID",
          not valid and any(f["check"] == "ambient_clean" for f in failures))
    manifest["ambient_env"] = {"NY_MARGIN_ROW_GPU": "1"}
    valid, _f, warnings, _c = validate_run(
        arm, 2025, "cifar100", rows, manifest, _summary_of(rows),
        engagement_arm=[])
    check("an ambient lever equal to the arm is only a WARNING",
          valid and any(w["check"] == "ambient_clean" for w in warnings))

    material_env = {
        "PATH": "/usr/bin",
        "NY_BIN": "/tmp/ny",
        "NY_MARGIN_ROW_CLASSWISE": "1",
        "OMP_NUM_THREADS": "7",
        "RAYON_NUM_THREADS": "16",
        "RAYON_RS_NUM_CPUS": "4",
    }
    scrubbed_env, scrubbed_names = child_env(False, material_env)
    check("child env scrubs NY_* and both Rayon worker controls but preserves OMP",
          scrubbed_names == ["NY_BIN", "NY_MARGIN_ROW_CLASSWISE", "RAYON_NUM_THREADS",
                             "RAYON_RS_NUM_CPUS"]
          and not (RAYON_WORKER_ENV & scrubbed_env.keys())
          and scrubbed_env.get("OMP_NUM_THREADS") == "7"
          and scrubbed_env.get("PATH") == "/usr/bin",
          repr((scrubbed_env, scrubbed_names)))
    kept_env, kept_names = child_env(True, material_env)
    check("--keep-ambient preserves both Rayon worker controls",
          kept_env == material_env and kept_names == [],
          repr((kept_env, kept_names)))

    ambient_rayon_failures = []
    for name in RAYON_WORKER_ENV:
        manifest["ambient_env"] = {name: "16"}
        valid, failures, _w, _c = validate_run(
            arm, 2025, "cifar100", rows, manifest, _summary_of(rows))
        ambient_rayon_failures.append(
            not valid and any(f["check"] == "ambient_clean" for f in failures)
        )
    check("either ambient Rayon worker override nobody requested is INVALID",
          all(ambient_rayon_failures), repr(ambient_rayon_failures))

    bad_summary = dict(_summary_of(rows), unsat=99)
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", rows, _clean_manifest(arm=arm), bad_summary)
    check("summary/metadata disagreement is INVALID",
          not valid and any(f["check"] == "summary_matches_metadata" for f in failures))

    # --- the verdict cache must not be able to fake a run (T1.10 / T1.7)
    # Every row served, so no child ever started. Note what this bank does NOT
    # trip: the arm, the receipt and the child's ambient_env are all present and
    # correct, because the served rows carry an earlier run's flight record
    # verbatim. Only `from_cache` separates it from a real measurement.
    replayed = [dict(row, from_cache=True) for row in rows]
    valid, failures, _w, counts = validate_run(
        arm, 2025, "cifar100", replayed, _clean_manifest(arm=arm),
        _summary_of(replayed))
    check("a bank whose rows were ALL served from the cache is INVALID",
          not valid and any(f["check"] == "rows_were_measured" for f in failures)
          and not any(f["check"] in ("row_arm_matches", "child_env_carries_arm")
                      for f in failures),
          str(failures))
    check("cached rows are counted from the bank", counts["cached_rows"] == 2)

    valid, _f, warnings, _c = validate_run(
        arm, 2025, "cifar100", replayed, _clean_manifest(arm=arm), _summary_of(replayed),
        allow_cached_rows=True, engagement_arm=[])
    check("--allow-cached-rows downgrades a full replay to a WARNING",
          valid and any(w["check"] == "rows_were_measured" for w in warnings))

    half = [dict(rows[0], from_cache=True), json.loads(json.dumps(rows[1]))]
    valid, _f, warnings, counts = validate_run(
        arm, 2025, "cifar100", half, _clean_manifest(arm=arm), _summary_of(half),
        engagement_arm=[])
    check("a partly served bank is VALID but reported",
          valid and counts["cached_rows"] == 1
          and any(w["check"] == "rows_were_measured" for w in warnings))

    valid, _f, warnings, counts = validate_run(
        arm, 2025, "cifar100", rows, _clean_manifest(arm=arm), _summary_of(rows),
        engagement_arm=[])
    check("an unmarked bank reads as measured, so pre-cache banks stay VALID",
          valid and counts["cached_rows"] == 0
          and not any(w["check"] == "rows_were_measured" for w in warnings))

    # A binary that serves rows without marking them -- the exact hole the
    # marker closes -- is caught by the summary it prints about itself.
    unaccounted = dict(_summary_of(rows), cache_hits=2, cache_misses=0)
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", rows, _clean_manifest(arm=arm), unaccounted)
    check("a summary claiming more cache hits than the bank marks is INVALID",
          not valid and any(f["check"] == "cache_hits_accounted" for f in failures))
    accounted = dict(_summary_of(half), cache_hits=1, cache_misses=1)
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", half, _clean_manifest(arm=arm), accounted,
        engagement_arm=[])
    check("a summary whose hits the bank accounts for is VALID", valid, str(failures))

    error_rows = [_clean_row(arm, verdict="error")]
    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", error_rows, _clean_manifest(arm=arm),
        _summary_of(error_rows))
    check("a harness-error row is INVALID (a non-measurement, not a negative result)",
          not valid and any(f["check"] == "no_error_rows" for f in failures))

    valid, failures, _w, _c = validate_run(
        arm, 2025, "cifar100", [], _clean_manifest(arm=arm), {"rows": 0})
    check("an empty bank is INVALID",
          not valid and any(f["check"] == "rows_present" for f in failures))

    valid, failures, _w, _c = validate_run(
        arm, 2026, "cifar100", rows, _clean_manifest(year=2025, arm=arm),
        _summary_of(rows))
    check("a bank from another year is INVALID",
          not valid and any(f["check"] == "manifest_matches_request" for f in failures))

    capped = [json.loads(json.dumps(row)) for row in rows]
    capped[0]["capped_from"] = 100
    valid, _f, warnings, _c = validate_run(
        arm, 2025, "cifar100", capped, _clean_manifest(arm=arm), _summary_of(capped),
        engagement_arm=[])
    check("a capped row is flagged as a LOWER BOUND, not failed",
          valid and any(w["check"] == "budget_is_official" for w in warnings))

    # --- artifact loading end to end, still with no binary and no GPU
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        output = run_dir / "cifar100.csv"
        model_path = run_dir / "model.onnx"
        property_path = run_dir / "property.vnnlib"
        # Empty bytes are a valid prost/default ModelProto with no graph and
        # therefore an affirmative empty external-data closure.
        model_path.write_bytes(b"")
        property_path.write_bytes(b"synthetic vnnlib bytes\n")
        input_artifacts = {
            "onnx": fingerprint_file(model_path),
            "onnx_external_data": [],
            "vnnlib": fingerprint_file(property_path),
        }

        def artifact_manifest(test_rows, instance_selection=None):
            return _clean_manifest_for_rows(
                test_rows, instance_selection, input_artifacts=input_artifacts)

        sidecar(output, ".metadata.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        sidecar(output, ".manifest.json").write_text(
            json.dumps(artifact_manifest(rows)), encoding="utf-8")
        opts = argparse.Namespace(
            year=2025,
            allow_unverified_receipt=False,
            allow_error_rows=False,
            allow_cached_rows=False,
            allow_ungraduated_execution_contracts=True,
            selected_binary_identity={
                "canonical_path": "/repo/target/release/ny",
                "sha256": "a" * 64,
                "build_id": SELF_TEST_BUILD_ID,
            },
        )
        verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("artifacts load and validate from disk",
              verdict["valid"]
              and verdict["selected_row_inputs"]
                  == verdict["corpus_context"]["selected_row_inputs"],
              str(verdict["failures"]))

        property_path.write_bytes(b"mutated vnnlib bytes\n")
        changed_input_verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("artifact reuse is bound to current ONNX/VNN-LIB bytes",
              not changed_input_verdict["valid"]
              and any(f["check"] == "manifest_selected_inputs_match_current_bytes"
                      for f in changed_input_verdict["failures"]),
              str(changed_input_verdict["failures"]))
        property_path.write_bytes(b"synthetic vnnlib bytes\n")

        def proto_varint(value):
            encoded = bytearray()
            while value >= 0x80:
                encoded.append((value & 0x7f) | 0x80)
                value >>= 7
            encoded.append(value)
            return bytes(encoded)

        def proto_bytes(field, payload):
            return (proto_varint((field << 3) | 2)
                    + proto_varint(len(payload)) + payload)

        def proto_int(field, value):
            return proto_varint(field << 3) + proto_varint(value)

        def external_entry(location):
            return (proto_bytes(1, b"location")
                    + proto_bytes(2, location.encode("utf-8")))

        def external_tensor(location):
            return proto_bytes(13, external_entry(location)) + proto_int(14, 1)

        def gzip_member(payload):
            compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
            return compressor.compress(payload) + compressor.flush()

        split_attribute = (
            proto_bytes(5, proto_bytes(13, external_entry("attribute.data")))
            + proto_bytes(5, proto_int(14, 1))
            + proto_int(20, 4)
        )
        split_model = (
            proto_bytes(7, proto_bytes(5, external_tensor("weights.data")))
            + proto_bytes(7, proto_bytes(1, proto_bytes(5, split_attribute)))
        )
        ignored_second_member = proto_bytes(
            7, proto_bytes(5, external_tensor("ignored.data")))
        split_gzip_path = run_dir / "split.onnx.gz"
        split_gzip_path.write_bytes(
            gzip_member(split_model)
            + gzip_member(ignored_second_member)
            + b"ignored trailing bytes"
        )
        split_locations = _current_onnx_external_locations(
            fingerprint_file(split_gzip_path), {})
        check("Python mirrors native gzip and singular-message merge semantics",
              split_locations == ["attribute.data", "weights.data"],
              repr(split_locations))
        check("Python drive-prefix rejection is ASCII like native path policy",
              _normalize_onnx_external_location(
                  "é:weights.bin", "Unicode POSIX test") == "é:weights.bin")

        # ModelProto.graph -> GraphProto.initializer plus NodeProto.attribute.t.
        # Both locations name the same file after normalization, exercising the
        # independent Python wire walk and its sorted/deduplicated closure.
        initializer = external_tensor("weights.data")
        attribute = proto_bytes(5, external_tensor("./weights.data")) + proto_int(20, 4)
        node = proto_bytes(5, attribute)
        model_path.write_bytes(proto_bytes(
            7, proto_bytes(5, initializer) + proto_bytes(1, node)))
        weights_path = run_dir / "weights.data"
        weights_path.write_bytes(b"sealed external tensor bytes\n")
        input_artifacts["onnx"] = fingerprint_file(model_path)
        input_artifacts["onnx_external_data"] = [{
            "location": "weights.data",
            "artifact": fingerprint_file(weights_path),
        }]
        sidecar(output, ".manifest.json").write_text(
            json.dumps(artifact_manifest(rows)), encoding="utf-8")
        external_data_verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("Python validates the native ONNX external-data closure",
              external_data_verdict["valid"],
              str(external_data_verdict["failures"]))

        weights_path.write_bytes(b"mutated external tensor bytes\n")
        changed_external_verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("sidecar-only mutation invalidates Python artifact reuse",
              not changed_external_verdict["valid"]
              and any(f["check"] == "manifest_selected_inputs_match_current_bytes"
                      for f in changed_external_verdict["failures"]),
              str(changed_external_verdict["failures"]))
        weights_path.write_bytes(b"sealed external tensor bytes\n")

        omitted_closure = artifact_manifest(rows)
        omitted_closure["selected_row_inputs"]["cifar100"][0]["inputs"]["onnx"][0][
            "external_data"] = []
        sidecar(output, ".manifest.json").write_text(
            json.dumps(omitted_closure), encoding="utf-8")
        omitted_closure_verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("Python fails closed when a sidecar closure is omitted",
              not omitted_closure_verdict["valid"]
              and any(f["check"] == "manifest_selected_inputs_match_current_bytes"
                      for f in omitted_closure_verdict["failures"]),
              str(omitted_closure_verdict["failures"]))

        v1_input_manifest = artifact_manifest(rows)
        v1_input_manifest["selected_row_inputs_schema"] = "ny-cli/selected-row-inputs/v1"
        sidecar(output, ".manifest.json").write_text(
            json.dumps(v1_input_manifest), encoding="utf-8")
        v1_input_verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("Python rejects container-only selected-row schema v1",
              not v1_input_verdict["valid"]
              and any(f["check"] == "manifest_selected_inputs_match_current_bytes"
                      for f in v1_input_verdict["failures"]),
              str(v1_input_verdict["failures"]))
        sidecar(output, ".manifest.json").write_text(
            json.dumps(artifact_manifest(rows)), encoding="utf-8")

        legacy_input_manifest = artifact_manifest(rows)
        legacy_input_manifest.pop("selected_row_inputs_schema")
        legacy_input_manifest.pop("selected_row_inputs")
        sidecar(output, ".manifest.json").write_text(
            json.dumps(legacy_input_manifest), encoding="utf-8")
        legacy_input_verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("pre-input-seal artifact reuse fails closed",
              not legacy_input_verdict["valid"]
              and any(f["check"] == "manifest_selected_inputs_match_current_bytes"
                      for f in legacy_input_verdict["failures"]),
              str(legacy_input_verdict["failures"]))
        sidecar(output, ".manifest.json").write_text(
            json.dumps(artifact_manifest(rows)), encoding="utf-8")

        opts.selected_binary_identity = {
            "canonical_path": "/repo/target/release/ny-new",
            "sha256": "f" * 64,
            "build_id": "NY_BUILD_ID_V1 replacement",
        }
        wrong_binary_verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("artifact reuse is bound to the currently selected binary",
              not wrong_binary_verdict["valid"]
              and any(f["check"] == "manifest_matches_selected_binary"
                      for f in wrong_binary_verdict["failures"]),
              str(wrong_binary_verdict["failures"]))
        opts.selected_binary_identity = {
            "canonical_path": "/repo/target/release/ny",
            "sha256": "a" * 64,
            "build_id": SELF_TEST_BUILD_ID,
        }

        legacy_rows = [json.loads(json.dumps(row)) for row in rows]
        for row in legacy_rows:
            row["flight"].pop("ambient_env_schema")
        legacy_manifest = artifact_manifest(legacy_rows)
        legacy_manifest.pop("ambient_env_schema")
        sidecar(output, ".metadata.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in legacy_rows), encoding="utf-8")
        sidecar(output, ".manifest.json").write_text(
            json.dumps(legacy_manifest), encoding="utf-8")
        legacy_verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(legacy_rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("artifact reuse refuses readable pre-Rayon ambient provenance",
              not legacy_verdict["valid"]
              and {f["check"] for f in legacy_verdict["failures"]}
                  >= {"child_ambient_provenance", "manifest_ambient_provenance"},
              str(legacy_verdict["failures"]))

        sidecar(output, ".metadata.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        sidecar(output, ".manifest.json").write_text(
            json.dumps(artifact_manifest(rows)), encoding="utf-8")

        selector_path = run_dir / "selected.csv"
        selector_path.write_text(
            "cifar100,onnx/m.onnx,vnnlib/p.vnnlib\n", encoding="utf-8")
        opts.instances = str(selector_path)
        opts.instance_selection = load_instance_selection(selector_path)
        exact_manifest = artifact_manifest(
            rows,
            {"mode": "exact",
             "identity": opts.instance_selection["category_identities"]["cifar100"]},
        )
        sidecar(output, ".manifest.json").write_text(
            json.dumps(exact_manifest), encoding="utf-8")
        verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("artifact validation seals requested selectors and exact duplicate occurrences",
              verdict["valid"]
              and verdict["instance_selection"]["mode"] == "exact"
              and verdict["instance_selection"]["identity"]["count"] == 1
              and verdict["measured_row_set"]["count"] == 2
              and all("entries" not in identity for identity in (
                  verdict["instance_selection"]["identity"],
                  verdict["category_instance_selection"]["identity"],
                  verdict["measured_row_set"],
              ))
              and verdict["corpus_context"]["instance_selection"]
                  == verdict["instance_selection"]
              and verdict["corpus_context"]["category_instance_selection"]
                  == verdict["category_instance_selection"]
              and verdict["corpus_context"]["measured_row_set"]
                  == verdict["measured_row_set"], str(verdict["failures"]))

        wrong_selector = run_dir / "wrong-selected.csv"
        wrong_selector.write_text(
            "cifar100,onnx/m.onnx,vnnlib/other.vnnlib\n", encoding="utf-8")
        opts.instance_selection = load_instance_selection(wrong_selector)
        wrong_verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("equal-sized metadata from a different selector is INVALID",
              not wrong_verdict["valid"]
              and any(f["check"] == "instance_subset_matches_request"
                      for f in wrong_verdict["failures"]),
              str(wrong_verdict["failures"]))

        opts.instance_selection = load_instance_selection(selector_path)
        native_manifest = artifact_manifest(rows)
        native_manifest["instance_selection"] = {
            "mode": "exact",
            "identity": {
                "schema": INSTANCE_SELECTOR_SCHEMA, "sha256": "0" * 64, "count": 1,
            },
        }
        sidecar(output, ".manifest.json").write_text(
            json.dumps(native_manifest), encoding="utf-8")
        wrong_verdict = validate_artifacts(
            {"output": str(output), "exit_code": 0, "harness_timeout": False,
             "summary": _summary_of(rows), "stderr_tail": ""},
            arm, "cifar100", opts)
        check("a native manifest selector seal is validated when exposed",
              not wrong_verdict["valid"]
              and any(f["check"] == "manifest_subset_matches_request"
                      for f in wrong_verdict["failures"]),
              str(wrong_verdict["failures"]))
        sidecar(output, ".manifest.json").write_text(
            json.dumps(exact_manifest), encoding="utf-8")
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
    parser.add_argument("--dev-source-schema", action="store_true",
                        help="developer-only source fallback for --list-space when the selected "
                             "binary lacks `benchmarks space --json`; forbidden for dry runs "
                             "and measured/configured searches")
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
    parser.add_argument("--configs-dir", default=None,
                        help="configs/ parent or exact vnncompYY preset directory; the requested "
                             "--year is selected fail-closed (default: repo-local configs/, "
                             "matching the child auto-derivation)")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N instances per category (incompatible with --instances)")
    parser.add_argument(
        "--instances", default=None, metavar="FILE",
        help="exact native row-selector CSV: one category,onnx,vnnlib triple per line; "
             "the file's categories must exactly match --category (incompatible with --limit)",
    )
    parser.add_argument("--timeout-cap", type=int, default=None,
                        help="LOWER the per-instance budget; capped runs are flagged as a "
                             "lower bound")
    parser.add_argument(
        "--mode", choices=("random", "cover", "pair-cover", "grid"), default="random",
        help="arm design: random defaults to attributable one-axis screening; cover exercises "
             "every kept axis/value; pair-cover exercises every value pair across an explicit "
             "--axis shortlist; grid screens the full interaction product",
    )
    parser.add_argument("--samples", type=int, default=8,
                        help="unique random treatment arms to generate (ignored by "
                             "cover/pair-cover/grid; explicit underfill is refused)")
    parser.add_argument(
        "--max-axes", type=int, default=1,
        help="sampled treatment axes per random arm (default: 1 for attributable main effects; "
             ">1 explicitly screens joint interactions; dependency controls may be added)",
    )
    parser.add_argument("--max-arms", type=int, default=256,
                        help="refuse a repaired, deduplicated cover/pair-cover/grid plan larger "
                             "than this")
    parser.add_argument(
        "--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES,
        help="refuse grid/pair-cover before enumerating more than this many raw candidates "
             "(default: %(default)s; independent of executable --max-arms)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axis", action="append", default=[], dest="axis_filter",
                        help="restrict the search to this axis; repeatable")
    parser.add_argument("--class", dest="class_filter", action="append", default=[],
                        choices=("VerdictAffecting", "SafeToSearch"),
                        help="restrict to axes of this class; repeatable")
    parser.add_argument("--delivery", dest="delivery_filter", action="append", default=[],
                        choices=("preset", "env-only"),
                        help="restrict to shippable preset axes or experiment-only env axes; "
                             "repeatable (configured runs default to preset only)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="runs per (arm, category); marginal rows need >= 3. NOTE: only "
                             "FULLY-MEASURED runs count as repeats, so pair this with "
                             "--cache off — under the default --cache auto a repeat can be "
                             "served from the first run and buys no variance")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip an otherwise-unneeded default arm; planning refuses this "
                             "option whenever any treatment's exact causal comparator is the "
                             "default assignment")
    parser.add_argument("--out-dir", default=None,
                        help="artifact root (default: reports/search/<timestamp>); a measured "
                             "non-reuse run requires this path not to exist")
    parser.add_argument("--results", default=None,
                        help="JSONL results file atomically replaced only after the complete "
                             "plan finishes (default: <out-dir>/results.jsonl)")
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
                        help="re-validate artifacts under an existing exact search plan; the "
                             "same-plan result stream is atomically replaced, never appended")
    parser.add_argument("--keep-ambient", action="store_true",
                        help="do NOT scrub NY_* or Rayon worker controls from the child environment "
                             "(contaminated runs then fail T1, which is the point)")
    parser.add_argument("--allow-unchecked-domains", action="store_true",
                        help="search axes whose LeverKind could not be read (fail-open; T0 off)")
    parser.add_argument(
        "--allow-ungraduated-execution-contracts",
        action="store_true",
        help="DEVELOPER/DELIVERY-ONLY: permit axes with registry provenance but no "
             "axis-specific T1 execution contract; such runs cannot establish parameter "
             "engagement and are not scored-search evidence",
    )
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
    if opts.dev_source_schema and not opts.list_space:
        print("error: --dev-source-schema is valid only with --list-space; source schemas "
              "are forbidden for dry runs and measured/configured searches", file=sys.stderr)
        return 2
    if opts.self_test:
        return self_test()
    if opts.allow_ungraduated_execution_contracts:
        print(
            "WARNING: --allow-ungraduated-execution-contracts is active; delivery-only "
            "axes may be exercised, but those runs do not establish parameter engagement "
            "and are not scored-search evidence.",
            file=sys.stderr,
        )

    selected_binary_identity = None
    source_only_listing = (
        opts.dev_source_schema and opts.list_space and not Path(opts.ny_bin).exists()
    )
    if not source_only_listing:
        try:
            selected_binary_identity = probe_binary_identity(opts.ny_bin)
        except (OSError, ValueError) as error:
            print("error: cannot seal selected --ny-bin identity: %s" % error, file=sys.stderr)
            return 2
    try:
        space = load_space(
            opts.ny_bin,
            allow_source_schema=opts.dev_source_schema,
            keep_ambient=opts.keep_ambient,
        )
    except SpaceSchemaUnavailable as error:
        print("error: %s" % error, file=sys.stderr)
        if opts.list_space:
            print("       For source-only developer inspection, add "
                  "--dev-source-schema to --list-space.", file=sys.stderr)
        else:
            print("       Build/select the checkout's ny binary before planning or running "
                  "a search.", file=sys.stderr)
        return 2
    if selected_binary_identity is not None:
        try:
            identity_after_schema = probe_binary_identity(opts.ny_bin)
        except (OSError, ValueError) as error:
            print("error: cannot revalidate selected --ny-bin after schema probe: %s"
                  % error, file=sys.stderr)
            return 2
        if identity_after_schema != selected_binary_identity:
            print("error: selected --ny-bin changed while its search schema was being read",
                  file=sys.stderr)
            return 2
        selected_binary_identity = identity_after_schema
    opts.selected_binary_identity = selected_binary_identity
    kinds = lever_kinds_for_space(space)
    if opts.list_space:
        print_space(space, kinds, opts.allow_unchecked_domains, as_json=opts.json)
        return 0

    if not opts.categories:
        print("error: --category is required (or use --list-space / --self-test)", file=sys.stderr)
        return 2
    category_error = duplicate_category_error(opts.categories)
    if category_error is not None:
        print("error: %s" % category_error, file=sys.stderr)
        return 2
    limit_error = limit_request_error(opts.limit)
    if limit_error is not None:
        print("error: %s" % limit_error, file=sys.stderr)
        return 2
    selection = None
    if opts.instances is not None:
        try:
            selection = load_instance_selection(opts.instances)
        except (OSError, ValueError) as error:
            print("error: invalid --instances: %s" % error, file=sys.stderr)
            return 2
        request_error = instance_request_error(selection, opts.categories, opts.limit)
        if request_error is not None:
            print("error: %s" % request_error, file=sys.stderr)
            return 2
        opts.instances = selection["source"]["canonical_path"]
    opts.instance_selection = selection
    opts.instances_by_category = {}

    # Subset support is a measured-binary capability. Probe it even for a dry
    # run: printing a plan that the selected executable cannot execute is not a
    # valid pre-flight.
    flags = None
    if selection is not None:
        ny_bin = Path(opts.ny_bin)
        if not ny_bin.exists():
            print("error: ny binary not found at %s" % ny_bin, file=sys.stderr)
            return 2
        flags = probe_run_flags(ny_bin, keep_ambient=opts.keep_ambient)
        if "--instances" not in flags:
            print("error: this ny has no `benchmarks run --instances`; refusing to "
                  "substitute --limit because it selects different rows", file=sys.stderr)
            return 2
    pair_cover_error = _pair_cover_request_error(
        opts.mode, opts.axis_filter, no_baseline=opts.no_baseline)
    if pair_cover_error is not None:
        print("error: %s" % pair_cover_error, file=sys.stderr)
        return 2

    # T1.1: prove the exact YEAR-SCOPED preset this run will load before spending
    # an instance budget. Native `ny vnncomp` has a deliberately yearless
    # newest-first resolver; a benchmark sweep does know its year, so letting that
    # policy leak through made a 2025 safenlp run silently load vnncomp26.
    preset_doc, preset_known = None, False
    parsed_preset_docs = []
    repair_preset_docs = []
    configured_root = effective_configs_root(opts.configs_dir)
    if opts.configs_dir is None and configured_root is not None:
        print("configs: auto-derived repo-local %s (matching `ny benchmarks run`)"
              % configured_root.resolve())
    opts.configs_dir = str(configured_root) if configured_root is not None else None
    if opts.configs_dir:
        requested_configs_dir = opts.configs_dir
        scoped_configs_dir = benchmark_configs_dir(requested_configs_dir, opts.year)
        if scoped_configs_dir is None:
            print("error: --configs-dir %s has no exact vnncomp%02d directory for "
                  "VNN-COMP %d.\n"
                  "       Pass either the configs/ parent or that exact year directory; "
                  "newer-year fallback is refused."
                  % (requested_configs_dir, opts.year % 100, opts.year), file=sys.stderr)
            return 2
        # Forward the exact directory, not its parent. The Rust sweep fingerprints
        # this path in its manifest and the native child treats an already scoped
        # vnncompYY directory as the whole search universe.
        opts.configs_dir = str(scoped_configs_dir)
        delivery_was_implicit = not opts.delivery_filter
        opts.delivery_filter = effective_delivery_filter(True, opts.delivery_filter)
        if delivery_was_implicit:
            requested_env_only = [
                axis.name for axis in space.axes
                if axis.name in opts.axis_filter and delivery_bucket(axis) == "env-only"
            ]
            if requested_env_only:
                print("error: configured competition searches default to preset-deliverable "
                      "axes, but --axis requested env-only %s.\n"
                      "       Add explicit `--delivery env-only` for a research-only arm."
                      % ", ".join(sorted(requested_env_only)), file=sys.stderr)
                return 2
            print("delivery: preset (configured competition default; use explicit "
                  "--delivery env-only for research-only axes)")
        resolved = {}
        for category in opts.categories:
            path = resolve_preset_path(opts.configs_dir, category, opts.year)
            if path is None:
                print("error: exact VNN-COMP %d directory %s has NO preset for category %r.\n"
                      "       Running anyway would measure every arm against bare defaults "
                      "and silently invalidate the comparison."
                      % (opts.year, opts.configs_dir, category), file=sys.stderr)
                return 2
            resolved[category] = path
            print("preset: %s -> %s" % (category, path))
        docs = [load_preset_doc(path) for path in resolved.values()]
        # Repair keeps unresolved documents too: their dependency facts are
        # unknown and must fail conservative rather than being mistaken for an
        # all-absent baseline. T0 still uses only a wholly parsed category set.
        repair_preset_docs = docs
        if docs and all(doc is not None for doc in docs):
            parsed_preset_docs = docs
        if docs and docs[0] is not None and all(doc == docs[0] for doc in docs[1:]):
            preset_doc, preset_known = docs[0], True
        elif any(doc is None for doc in docs):
            print("T0: preset(s) resolved but could not be parsed (PyYAML absent?) — "
                  "no preset-delivered token will be dropped")
        else:
            print("T0: %d categories with different presets — baselines will be compared "
                  "per axis, and only tokens common to every category will be dropped"
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

    if selection is not None:
        try:
            preflight_instance_selection(ny_bin, selection, opts, flags)
        except ValueError as error:
            print("error: invalid --instances: %s" % error, file=sys.stderr)
            return 2
        print("instances: native dry-run resolved %d selector(s) across %d category invocation(s)"
              % (selection["identity"]["count"], len(opts.categories)))

    pool, notes = [], []
    for axis in space.axes:
        if opts.axis_filter and axis.name not in opts.axis_filter:
            continue
        if opts.class_filter and axis.cls not in opts.class_filter:
            continue
        if opts.delivery_filter and delivery_bucket(axis) not in opts.delivery_filter:
            continue
        if axis.name in _AUTOMATED_COMPOSITE_AXES:
            notes.append(
                "%s is an admission master, not an independent treatment — "
                "excluded from automated one-axis planning; root-family tuning "
                "axes add it inside the finite-CUDA execution recipe"
                % axis.name
            )
            continue
        if (
            not opts.allow_ungraduated_execution_contracts
            and axis.name not in _AXIS_SPECIFIC_T1_CONTRACTS
        ):
            notes.append(
                "%s has delivery provenance but no axis-specific T1 execution "
                "contract — excluded fail-closed (developer-only override: "
                "--allow-ungraduated-execution-contracts)" % axis.name
            )
            continue
        applicability_finding = category_applicability_finding(axis, opts.categories)
        if applicability_finding is None:
            applicability_finding = runtime_applicability_finding(
                axis,
                opts.categories,
                preset_docs=parsed_preset_docs or None,
                preset_expected=bool(opts.configs_dir),
            )
        if applicability_finding is not None:
            notes.append(applicability_finding)
            continue
        kept, dropped = admissible_values(axis, kinds, opts.allow_unchecked_domains,
                                          preset_doc=preset_doc, preset_known=preset_known,
                                          preset_expected=bool(opts.configs_dir),
                                          preset_docs=parsed_preset_docs or None)
        notes.extend(dropped)
        if kept:
            pool.append((axis, kept))
    for note in notes:
        print("T0: %s" % note)
    missing_requested_axes = sorted(
        set(opts.axis_filter).difference(axis.name for axis, _values in pool)
    )
    if missing_requested_axes:
        print(
            "error: explicitly requested axis/axes have no T0-admissible treatment: %s"
            % ", ".join(missing_requested_axes),
            file=sys.stderr,
        )
        return 2
    if not pool and not opts.no_baseline:
        print("T0 left no searchable axis; only the baseline arm will run.")

    baseline_values = effective_baseline_values(
        space,
        kinds,
        preset_doc=preset_doc,
        preset_known=preset_known,
        preset_expected=bool(opts.configs_dir),
        preset_docs=parsed_preset_docs or None,
    )
    repair_baselines = (
        effective_repair_baselines(space, kinds, repair_preset_docs)
        if repair_preset_docs else None
    )
    arms, dropped_inert, design = generate_arms(
        space, pool, opts.mode, opts.samples, opts.seed, opts.max_axes,
        baseline=not opts.no_baseline, max_arms=opts.max_arms, kinds=kinds,
        max_candidates=opts.max_candidates, baseline_values=baseline_values,
        return_design=True, repair_baselines=repair_baselines)
    plan_evidence_scope = _classify_design_evidence(design)
    empty_comparator_contrasts = empty_comparator_contrast_ids(design)
    if opts.no_baseline and empty_comparator_contrasts:
        print(
            "error: --no-baseline cannot remove the exact default comparator required by "
            "%d planned causal contrast(s); omit --no-baseline or select only treatments "
            "with non-empty prerequisite controls" % len(empty_comparator_contrasts),
            file=sys.stderr,
        )
        return 2
    arms = select_next_arms(arms, [])

    print()
    print("space source: %s" % space.source)
    print("%d arm(s) over %d axis/axes; %d lattice-inert sample(s) never generated"
          % (len(arms), len(pool), dropped_inert))
    for arm in arms:
        contrasts, comparison_for = _assignment_design_roles(design, arm)
        roles = []
        if contrasts:
            roles.append("%d contrast(s)" % len(contrasts))
        if comparison_for:
            roles.append("control for %d" % len(comparison_for))
        print("  %-10s %-55s %s" % (
            arm_id(arm), arm_label(arm), "; ".join(roles) or "baseline/control",
        ))
    print("categories: %s | year %d | repeats %d | %d run(s) total"
          % (", ".join(opts.categories), opts.year, opts.repeats,
             len(arms) * len(opts.categories) * opts.repeats))
    if opts.dry_run:
        return 0

    ny_bin = Path(opts.ny_bin)
    if not ny_bin.exists():
        print("error: ny binary not found at %s" % ny_bin, file=sys.stderr)
        return 2
    if flags is None:
        flags = probe_run_flags(ny_bin, keep_ambient=opts.keep_ambient)
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
    try:
        identity_before_runs = probe_binary_identity(ny_bin)
    except (OSError, ValueError) as error:
        print("error: cannot revalidate selected --ny-bin before execution/reuse: %s"
              % error, file=sys.stderr)
        return 2
    if identity_before_runs != opts.selected_binary_identity:
        print("error: selected --ny-bin changed after arm planning; refusing mixed evidence",
              file=sys.stderr)
        return 2

    default_out = REPO_ROOT / "reports" / "search" / time.strftime("%Y%m%dT%H%M%S")
    out_dir = Path(opts.out_dir or default_out)
    results_path = Path(opts.results) if opts.results else out_dir / "results.jsonl"
    design_path = out_dir / "search_plan.json"
    layout_issues = output_layout_issues(
        out_dir, results_path, design_path, arms, opts.categories, opts.repeats)
    if layout_issues:
        print("error: unsafe output layout: %s" % "; ".join(layout_issues),
              file=sys.stderr)
        return 2
    if opts.reuse:
        if not out_dir.is_dir():
            print("error: --reuse requires an existing --out-dir directory",
                  file=sys.stderr)
            return 2
    else:
        if out_dir.exists():
            print(
                "error: non-reuse search requires a fresh --out-dir; refusing to mix a new "
                "plan with existing plans or run artifacts at %s" % out_dir,
                file=sys.stderr,
            )
            return 2
        if results_path.exists():
            print(
                "error: non-reuse search requires a fresh --results target; refusing to "
                "overwrite or append %s" % results_path,
                file=sys.stderr,
            )
            return 2
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            print("error: cannot create fresh --out-dir: %s" % error, file=sys.stderr)
            return 2
    design_manifest = search_design_manifest(
        arms,
        design,
        mode=opts.mode,
        seed=opts.seed,
        max_axes=opts.max_axes,
        categories=opts.categories,
        year=opts.year,
        repeats=opts.repeats,
        space_source=space.source,
        evidence_scope=plan_evidence_scope,
    )
    try:
        results_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print("error: cannot create results parent directory: %s" % error,
              file=sys.stderr)
        return 2
    try:
        if opts.reuse:
            if not design_path.is_file():
                raise ValueError(
                    "--reuse requires the existing exact search_plan.json; "
                    "artifacts without their causal contrast plan cannot be reused"
                )
        else:
            _atomic_write_utf8(
                design_path,
                json.dumps(design_manifest, indent=2, sort_keys=True) + "\n",
            )
        persisted_design = json.loads(design_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print("error: cannot persist/read sealed search design: %s" % error,
              file=sys.stderr)
        return 2
    design_issues = search_design_manifest_issues(persisted_design)
    if design_issues:
        print("error: persisted search design failed validation: %s"
              % "; ".join(design_issues), file=sys.stderr)
        return 2
    if opts.reuse and persisted_design != design_manifest:
        print(
            "error: --reuse search plan identity/payload differs from the newly requested "
            "exact contrast design; refusing to reinterpret existing artifacts",
            file=sys.stderr,
        )
        return 2
    design_manifest = persisted_design
    design_identity = design_manifest["identity"]
    if opts.reuse:
        prior_result_issues = existing_result_stream_issues(
            results_path, design_identity["sha256"])
        if prior_result_issues:
            print(
                "error: existing results stream cannot be safely replaced for reuse: %s"
                % "; ".join(prior_result_issues),
                file=sys.stderr,
            )
            return 2
    contrast_by_id = {
        contrast["contrast_id"]: contrast
        for contrasts in design["contrasts_by_assignment"].values()
        for contrast in contrasts
    }
    if selection is not None:
        try:
            opts.instances_by_category = materialize_instance_selections(selection, out_dir)
        except (OSError, ValueError) as error:
            print("error: cannot materialize exact --instances subset: %s" % error,
                  file=sys.stderr)
            return 2
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
                contrasts, comparison_for = _assignment_design_roles(design, arm)
                require_closed_root = any(
                    any(name in _ROOT_JOINT_INTERM_ALPHA_FAMILY
                        for name in contrast_by_id[contrast_id]["requested_axes"])
                    for contrast_id in comparison_for
                    if contrast_id in contrast_by_id
                )
                verdict = validate_artifacts(
                    run_record,
                    arm,
                    category,
                    opts,
                    engagement_arm=[],
                    require_closed_root_terminal=require_closed_root,
                )
                contrast_evidence = []
                for contrast in contrasts:
                    qualified = validate_artifacts(
                        run_record,
                        arm,
                        category,
                        opts,
                        engagement_arm=contrast["requested_assignment"],
                    )
                    contrast_evidence.append({
                        "contrast_id": contrast["contrast_id"],
                        "evidence_scope": contrast["evidence_scope"],
                        "valid": qualified["valid"],
                        "failures": qualified["failures"],
                        "warnings": qualified["warnings"],
                    })
                record_failures = list(verdict["failures"])
                any_contrast_valid = any(
                    evidence["valid"] for evidence in contrast_evidence
                )
                if contrasts and not any_contrast_valid:
                    record_failures.append({
                        "check": "contrast_execution_qualified",
                        "detail": "no sealed causal contrast for this assignment passed "
                                  "its requested-axis execution contract",
                    })
                record_valid = verdict["valid"] and (
                    not contrasts or any_contrast_valid
                )
                scored_qualified = [
                    evidence["contrast_id"] for evidence in contrast_evidence
                    if evidence["valid"] and evidence["evidence_scope"] == "scored"
                ]
                scopes = {evidence["evidence_scope"] for evidence in contrast_evidence}
                record_scope = (
                    next(iter(scopes)) if len(scopes) == 1
                    else "mixed" if scopes else "scored"
                )
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
                    "valid": record_valid,
                    "structurally_valid": verdict["valid"],
                    "evidence_scope": record_scope,
                    "contrasts": contrasts,
                    "comparison_for": comparison_for,
                    "contrast_evidence": contrast_evidence,
                    "scored_qualified_contrasts": scored_qualified,
                    "failures": record_failures,
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
                    "search_design_schema": SEARCH_DESIGN_SCHEMA,
                    "search_design_sha256": design_identity["sha256"],
                    "search_design_manifest": str(design_path),
                    "executable_sha256": verdict["executable_sha256"],
                    "build_id": verdict["build_id"],
                    "build_provenance": verdict["build_provenance"],
                    "compute_backend": verdict["compute_backend"],
                    "host": verdict["host"],
                    "corpus_context": verdict["corpus_context"],
                    "instance_selection": verdict["instance_selection"],
                    "category_instance_selection": verdict["category_instance_selection"],
                    "measured_row_set": verdict["measured_row_set"],
                    "output": run_record["output"],
                    "metadata": verdict["metadata"],
                    "manifest": verdict["manifest"],
                    "cmd": run_record["cmd"],
                }
                records.append(record)
                if record_valid and scored_qualified:
                    print("  VALID   solved %d (sat %d unsat %d) timeout %d"
                          % (record["solved"], counts["sat"], counts["unsat"], counts["timeout"]))
                elif record_valid and contrasts:
                    print("  DELIVERY-ONLY -- structurally valid, but no scored-search "
                          "contrast is qualified")
                elif record_valid:
                    print("  VALID CONTROL   solved %d (sat %d unsat %d) timeout %d"
                          % (record["solved"], counts["sat"], counts["unsat"], counts["timeout"]))
                else:
                    print("  INVALID -- EXCLUDED from results:")
                    for failure in record_failures:
                        print("    %s: %s" % (failure["check"], failure["detail"]))
                for warning in verdict["warnings"]:
                    print("  warning %s: %s" % (warning["check"], warning["detail"]))

    completed_stream_issues = causal_result_stream_issues(
        records, design_identity["sha256"])
    if completed_stream_issues:
        print(
            "error: refusing to replace results with a malformed/ambiguous completed "
            "stream; previous results were preserved: %s"
            % "; ".join(completed_stream_issues),
            file=sys.stderr,
        )
        return 2
    try:
        _atomic_write_utf8(
            results_path,
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        )
    except OSError as error:
        print(
            "error: cannot atomically replace completed results stream; previous results "
            "were preserved: %s" % error,
            file=sys.stderr,
        )
        return 2

    print_leaderboard(records)
    print()
    print("results: %s" % results_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
