#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Library of the SEAL PIPELINE (scripts/seal/): the system that turns BUILT capabilities into SEALED ones, so the
router (ROUTER_PROGRAM R5, "capabilities routed, not armed") can route them.

Every decision `seal_run.sh` acts on is made here, where scripts/seal/seal_selftest.py pins it; the shell only
orchestrates, and the auditor (scripts/engine_program_audit.py) calls the SAME functions to derive S5-S8 and
SR.seal from the recorded evidence. Nothing here runs cargo, an ny binary or the box.

Reused, never forked:
  * scripts/m0/m0lib.py (imported): VNN-LIB reader, witnesses, the attack-off table, the verified yaml overlay
    editor, instances/bank/field-truth alignment BY INDEX, the m0-spec/v1 runner spec and the per-run judge
    (truth contradiction, bank flip, sat-row and self-contradiction alarms, VOID rules, profile identity,
    cross-tag instance flips);
  * scripts/m0/moat_rows.py (the both-year moat row list), knife_edge.py (the knife-edge ladder), m0_judge.py
    (the run schedule), and the m0 RUNNER itself (moat_both_profiles.sh: systemd-safe, ny_box.sh measure per
    run, competition backstop, per-run raw logs, EMERGENCY stop, resume, M0_STOP_FILE);
  * scripts/seal/allonlib.py: VENDORED byte-identical from exp/all-on-preset f39aa5bd7 (VENDORED_ALLONLIB; the
    selftest pins its blob): seeded ranks, shard partition, bank walls, defect alarms, box provenance.

The ladder this pipeline derives, per capability (docs/engine_program_capabilities.yaml entry):
  S5 arrives   the capability armed ONLY through a generated yaml overlay (zero NY_*), its own observable on
               >= N real rows (else VOID), every armed key's flight lever reading `@config`;
  S7 controls  sat-finding-OFF negative controls on truly-SAT rows (field truth) with the capability engaged,
               plus knife-edge rungs where a witness makes one expressible: any unsat = EMERGENCY, stop all;
  S6 measured  a pre-registered stratified A/B (armed vs capability-absent control) at official budgets, >= 3
               reps on marginal rows, judged on ABSOLUTE conversions/losses (and absolute closes when the
               registry names a closes regex) -- never a throughput-normalised ratio;
  S8 moat      both profiles (proof + competition), both years, every banked/field-SAT row of the admitted
               categories: 0 flips, every Proof-decided row decided the same way in >= 2/3 Competition reps;
  envelope     the op-kind / spec-archetype set observed on the S7/S8 rows where the capability ENGAGED (and the
               router's own admission said admit, when its shadow line is present). SR.seal = S7 + S8 done and
               the recorded envelope inside the evidence rows' op sets.
"""

from __future__ import annotations

import collections
import copy
import datetime as _dt
import glob
import gzip
import hashlib
import json
import math
import mmap
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
M0_DIR = os.path.join(SCRIPTS, "m0")
for _p in (HERE, M0_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import m0lib as L  # noqa: E402
import m0_judge as M0J  # noqa: E402
import moat_rows as MR  # noqa: E402
import knife_edge as KE  # noqa: E402
import allonlib as A  # noqa: E402  (vendored byte-identical, see VENDORED_ALLONLIB)
import certlib as C  # noqa: E402  (the certified S8 mode)

import yaml  # noqa: E402

VENDORED_ALLONLIB = {
    "file": "scripts/seal/allonlib.py",
    "source_commit": "f39aa5bd75aef75c0af11f139c974b78deab55ba",
    "source_path": "scripts/allon/allonlib.py",
    "blob": "510c541cbb26c3ab0c967fa1c8f88cfc232707cb",
}

PLAN_SCHEMA = "seal-plan/v1"
STAGE_SCHEMA = "seal-stage/v1"
ENVELOPE_SCHEMA = "seal-envelope/v1"
FACTS_SOURCE = "offline-scan/v1"
STAGES = ("S5", "S7", "S6", "S8")               # run order
GATES = {"S5": (), "S7": ("S5",), "S6": ("S5", "S7"), "S8": ("S5", "S7")}
GATE_OK = {"S5": ("PASS",), "S7": ("PASS", "PARTIAL")}
PASS, PARTIAL, VOID, EMERGENCY = "PASS", "PARTIAL", "VOID", "EMERGENCY"
STATE_OF = {PASS: "done", PARTIAL: "partial", VOID: "open", EMERGENCY: "open"}
RC_OF = {PASS: 0, PARTIAL: 4, VOID: 5, EMERGENCY: 3}
SHARD_EMPTY_RC = 10
RANK = {"open": 0, "partial": 1, "done": 2, "na": 2}

DEFAULTS = {
    "years": (2026, 2025),
    "arrival_rows": 2,             # S5: rows that must show the observable
    "arrival_min_wall_frac": 0.2,  # S5 rows: banked unsat that needed real prover work (wall >= frac x budget)
    "negctl_per_category": 2,      # S7: truly-SAT control rows per admitted (year, category)
    "s6_gap_cap": 6,               # S6 strata caps per (year, category)
    "s6_near_cap": 6,
    "s6_sat_cap": 2,
    "s6_uniform": 4,
    "s6_marginal_reps": 3,         # ">= 3 reps on marginal rows" (targets, gap, near)
    "s6_other_reps": 1,
    "s7_reps_c": 1, "s7_reps_p": 1,
    "s8_reps_c": 3, "s8_reps_p": 1,   # section 1.7 mode identity: a Proof verdict must hold in >= 2 of 3 Competition reps
    "shards": 1,
    "s8_mode": "classic",          # or "certified" (certlib: 1 armed run per row + independent re-proof / replay)
    "checker_budget_s": 900,       # certified: `ny proof-check --budget-s` per re-proof bundle
}
NEAR_FRAC = A.NEAR_TIMEOUT_FRAC
# What a sealing stage may never be weakened below, whatever the plan options say (the judge applies these, so the
# runner, `seal record` and the auditor all see the same verdict): S8 is BOTH years and section 1.7's 3 competition +
# 1 proof reps on every row, over the un-narrowed admission scope; S5 needs at least one arrival row.
BOTH_YEARS = (2026, 2025)
S8_MIN_REPS_C, S8_MIN_REPS_P = 3, 1
S5_MIN_ROWS = 1

# the router's OpKind / SpecArchetype tokens in bit order (crates/ny-core/src/router/decl.rs; the selftest re-derives
# both lists from that source, so a reordered or renamed kind fails it rather than silently mis-decoding a receipt)
OPKIND_TOKENS = ("linear", "conv", "conv_transpose", "relu", "sigmoid", "tanh", "softmax", "bn", "residual",
                 "concat", "pool", "sign", "resize", "bilinear", "norm", "other")
SPEC_TOKENS = ("robustness", "halfspace", "disjunctive", "relational", "nonlinear")

# #admission-census: the binary's census of the router's OWN admission decisions (`ny vnncomp census`,
# crates/ny-cli/src/router/census.rs). When a plan carries one, S8's required rows ARE the census-admitted moat rows
# (both years, every scored category) and SR.seal checks the census, never the envelope's offline closure.
CENSUS_SCHEMA = "ny-admission-census/v2"
CORPUS_SCHEMA = "ny-admission-census-corpus/v1"
CENSUS_COLUMNS = ("row_id", "cap", "admit", "router", "parent_probe", "reasons", "facts", "ops", "spec", "form")
# the widest envelope a seal can record (every kind but `other`, every archetype): the census default
CENSUS_WIDEST = (tuple(t for t in OPKIND_TOKENS if t != "other"), SPEC_TOKENS)


class SealError(L.M0Error):
    """A refusal: the seal pipeline will not emit or accept an artefact it cannot vouch for."""


# ---------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def write_json_atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def write_new(path: str, text: str) -> None:
    if os.path.exists(path):
        raise SealError(f"refusing to overwrite {path} (the seal pipeline never deletes; pick a new --out)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def parse_shard(s: str | None) -> tuple[int, int]:
    try:
        return A.parse_shard(s)
    except A.AllonError as e:
        raise SealError(str(e)) from None


def shard_key(k: int, n: int) -> str:
    return f"{k}of{n}"


def tagbase(year: int, category: str) -> str:
    return f"{year % 100}{A.short_cat(category)}"


def body_of(raw: str) -> str:
    """A run's log without the `[ny-box` lines (the box echoes the command line: never evidence)."""
    return "\n".join(ln for ln in raw.splitlines() if not ln.startswith("[ny-box"))


def engaged(body: str, need: list[str]) -> bool:
    """The capability's OWN observable: every `need` regex matches the run's log (an empty need is never engaged)."""
    return bool(need) and all(re.search(n, body, re.M) for n in need)


# ---------------------------------------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------------------------------------


def load_registry(path: str) -> dict:
    try:
        text = open(path).read()
    except OSError as e:
        raise SealError(f"registry unreadable: {path}: {e}") from None
    try:
        reg = L.strict_load(text)
    except yaml.YAMLError as e:
        raise SealError(f"registry does not parse (duplicate keys refused): {path}: {str(e)[:200]}") from None
    if not isinstance(reg, dict) or not isinstance(reg.get("capabilities"), list):
        raise SealError(f"{path}: no capabilities list")
    return reg


def get_capability(reg: dict, cap_id: str) -> dict:
    hits = [c for c in reg["capabilities"] if isinstance(c, dict) and c.get("id") == cap_id]
    if len(hits) != 1:
        raise SealError(f"capability {cap_id!r} found {len(hits)} times in the registry (want exactly one)")
    return hits[0]


def _epa():
    """scripts/engine_program_audit.py: its yaml lookup and default comparison are THE definition of 'armed'
    (one implementation; imported lazily so a harness snapshot without it can still judge)."""
    if SCRIPTS not in sys.path:
        sys.path.insert(0, SCRIPTS)
    import engine_program_audit as EPA  # noqa: E402
    return EPA


def _is_bool(key: dict) -> bool:
    return str(key.get("type", "")).strip().lower().startswith("bool")


def _live_keys(cap: dict) -> dict[str, dict]:
    return {k["path"]: k for k in cap.get("keys", []) or [] if not k.get("retired")}


def m0_tier_of(cap_id: str) -> L.Tier | None:
    hits = [t for t in L.TIERS.values() if t.capability == cap_id]
    return hits[0] if len(hits) == 1 else None


def scored_categories(ny_root: str, years) -> dict[int, list[str]]:
    out = {}
    for y in years:
        banked, unbanked = MR.categories(ny_root, y)
        out[y] = sorted(set(banked) | set(unbanked))
    return out


def armed_in_configs(cap: dict, cfg: str, categories: list[str]) -> dict[str, dict]:
    """{category: {preset, keys: {path: value}}} for every category whose RESOLVED preset (the scored resolver,
    newest year first) sets one of the capability's keys to a non-default value."""
    epa = _epa()
    out: dict[str, dict] = {}
    for cat in sorted(set(categories)):
        rel = L.resolve_preset(cfg, cat)
        if not rel:
            continue
        doc = L.strict_load(open(os.path.join(cfg, rel)).read()) or {}
        hit = {}
        for path, k in _live_keys(cap).items():
            for p in [path] + list(k.get("alt_paths", []) or []):
                v = epa.Configs.lookup(doc, p)
                if v is not None and not epa.equal_default(v, k.get("default")):
                    hit[path] = v
                    break
        if hit:
            out[cat] = {"preset": rel, "keys": hit}
    return out


@dataclass
class Derived:
    """What the registry entry (plus the pin's configs) says about how to seal one capability."""

    cap_id: str
    arm: dict                       # registry key path -> value the overlay writes
    keys: dict                      # registry key path -> registry key dict
    need: list[str]                 # observable regexes (ALL must match: the capability engaged)
    markers: list[str]              # pin-strings fragments behind `need`
    arrival_markers: list[str]      # literal fragments the S5 evidence names (substrings of an engaged log)
    flight: list[str]               # lever:<NY_NAME>=<json>@config per armed key with a LeverDecl
    alarms: list[str]               # every-row alarms: the registry's seal.alarms + the defect signals
    sat_alarms: list[str]           # truly-SAT-row alarms: the m0 tier's (or seal.sat_alarms): row_alarms()
    categories: list[str]
    scope: list[str]                # the admission scope before --categories narrowed it
    years: list[int]
    routable: bool
    sources: dict = field(default_factory=dict)
    seal: dict = field(default_factory=dict)
    s6_closes: str | None = None
    s6_targets: list = field(default_factory=list)
    negctl_rows: list = field(default_factory=list)
    knife: dict = field(default_factory=dict)


def derive(cap: dict, cfg: str, scored: dict[int, list[str]], only_categories: list[str] | None = None,
           years: list[int] | None = None) -> Derived:
    """Derive the seal recipe from the registry entry. The entry's `seal:` block (a pre-registration) always wins;
    then, per field: arm = default_activation.settings, then the m0 tier that names this capability, then the
    values the pin's configs arm (else every bool key true); observable = the m0 tier, then evidence.S5 arrival
    markers, then markers[0]; categories = the m0 tier, then the live armings in the pin's configs. Refuses rather
    than guesses. `scope` keeps the admission scope BEFORE --categories narrowed it (the S8 judge refuses a
    narrowed moat as a full one)."""
    cap_id = cap["id"]
    sb = cap.get("seal") or {}
    if not isinstance(sb, dict):
        raise SealError(f"{cap_id}: seal: must be a mapping")
    keys = _live_keys(cap)
    if not keys:
        raise SealError(f"{cap_id}: keyless capability (no live typed key): there is no overlay that arms it; "
                        "seal its parent (arming.inherit) instead")
    tier = m0_tier_of(cap_id)
    src: dict[str, str] = {}
    all_cats = sorted({c for cs in scored.values() for c in cs})
    live = armed_in_configs(cap, cfg, all_cats)

    # --- arm values
    arm: dict = {}
    if sb.get("arm"):
        arm = dict(sb["arm"])
        src["arm"] = "registry seal.arm"
    elif (cap.get("default_activation") or {}).get("settings"):
        arm = dict(cap["default_activation"]["settings"])
        src["arm"] = "registry default_activation.settings"
    elif tier is not None:
        arm = dict(tier.arm)
        src["arm"] = f"m0 tier {tier.name} (its pre-registered arming)"
    else:
        for path, k in keys.items():
            found = [v["keys"][path] for v in live.values() if path in v["keys"]]
            if found:
                arm[path] = found[0]
            elif _is_bool(k):
                arm[path] = True
        src["arm"] = "derived: values the pin's configs arm, else every bool key true"
    # an arm key may be spelled by a registered key's alt path (the tier's spelling): fold it onto that key
    alt = {a: p for p, k in keys.items() for a in (k.get("alt_paths") or [])}
    arm = {alt.get(p, p): v for p, v in arm.items()}
    unknown = [p for p in arm if p not in keys]
    if unknown and src["arm"].startswith("m0 tier"):
        for p in unknown:   # a companion knob the m0 tier pre-registered (typed yaml key, no entry of its own here)
            keys[p] = {"path": p}
        src["arm"] += f"; companion key(s) outside the entry: {unknown}"
    elif unknown:
        raise SealError(f"{cap_id}: arm names key(s) that are not live keys of the entry: {unknown}")
    if not arm:
        raise SealError(f"{cap_id}: nothing to arm (no bool key and no armed value in the pin's configs): add "
                        "seal.arm to the registry entry")
    for p, v in arm.items():
        try:
            L.fmt_yaml(v)
        except L.M0Error as e:
            raise SealError(f"{cap_id}: arm value for {p}: {e}") from None

    # --- observable
    arrivals = [a.get("marker") for a in (((cap.get("evidence") or {}).get("S5") or {}).get("arrivals") or [])
                if a.get("marker")]
    cap_markers = list(cap.get("markers") or [])
    if sb.get("observable"):
        need = list(sb["observable"])
        markers = list(sb.get("markers") or cap_markers)
        arrival_markers = list(sb.get("arrival_markers") or markers)
        src["observable"] = "registry seal.observable"
    elif tier is not None:
        need = list(tier.need)
        markers = list(tier.markers)
        arrival_markers = list(tier.markers)
        src["observable"] = f"m0 tier {tier.name}"
    elif arrivals:
        need = [rx_literal(m) for m in dict.fromkeys(arrivals)]
        markers = [m for m in cap_markers if any(m in a for a in arrivals)] or cap_markers
        arrival_markers = list(dict.fromkeys(arrivals))
        src["observable"] = "registry evidence.S5 arrival markers (literal)"
    elif cap_markers:
        need = [rx_literal(cap_markers[0])]
        markers = [cap_markers[0]]
        arrival_markers = [cap_markers[0]]
        src["observable"] = "registry markers[0] (the weakest source: add seal.observable)"
    else:
        raise SealError(f"{cap_id}: no observable (no seal.observable, m0 tier, S5 arrival marker or marker)")
    if not markers:
        raise SealError(f"{cap_id}: no pin marker fragment behind the observable (a dark gate cannot be detected)")
    for n in need:
        try:
            re.compile(n)
        except re.error as e:
            raise SealError(f"{cap_id}: observable regex {n!r}: {e}") from None
        if " && " in n or n.startswith("!"):
            raise SealError(f"{cap_id}: observable regex {n!r} may not contain ' && ' or start with '!'")

    # --- flight: every armed key that has a LeverDecl must read its value from config (the yaml route arrived)
    flight = []
    for p, v in arm.items():
        decl = keys[p].get("decl")
        if decl and re.match(r"^NY_[A-Z0-9_]+$", str(decl)):
            flight.append(f"lever:{decl}={json.dumps(v)}@config")
    src["flight"] = f"{len(flight)} lever receipt(s) from the armed keys' LeverDecls"

    # --- alarms. EVERY row: the registry's seal.alarms + the defect signals (a CROSSED bound or a cross-check
    # mismatch is a defect whatever the row's truth). TRULY-SAT rows only (truth or bank sat, every NEGCTL/KNIFE row):
    # the m0 tier's alarms (or seal.sat_alarms) -- they are "the capability CLOSED this" markers (frozen-dual's
    # `verdict=all_closed`, the margin-row lane's UNSAT line, fwd-root's union_refuted == total), an emergency on a
    # sat row and the expected outcome on an unsat one; m0's own A/B arms them on sat rows only (ab_arms.py).
    # Fix (#certified-moat): they were applied to EVERY row, so S5 (banked-UNSAT arrival rows), S6 and S8 of
    # frozen-dual raised a false EMERGENCY on the first row its lane proved.
    alarms = list(sb.get("alarms") or ())
    for a in A.DEFECT_ALARMS:
        if a not in alarms:
            alarms.append(a)
    sat_alarms = [a for a in (sb.get("sat_alarms") or (tier.alarm if tier else ())) if a not in alarms]
    for a in alarms + sat_alarms:
        if " || " in a:
            raise SealError(f"{cap_id}: alarm regex {a!r} contains the separator ' || '")
        re.compile(a)

    # --- admission scope (categories)
    if sb.get("categories"):
        cats = list(sb["categories"])
        src["categories"] = "registry seal.categories"
    elif tier is not None:
        cats = list(tier.categories)
        src["categories"] = f"m0 tier {tier.name}"
    elif live:
        cats = sorted(live)
        src["categories"] = "derived: scored categories whose resolved preset arms the capability (the pin's configs)"
    else:
        raise SealError(f"{cap_id}: no admission scope: add seal.categories (the capability is armed nowhere and no "
                        "m0 tier names it)")
    full = list(cats)
    if only_categories:
        extra = [c for c in only_categories if c not in cats]
        if extra:
            raise SealError(f"{cap_id}: --categories {extra} are outside the admission scope {cats}")
        cats = [c for c in cats if c in only_categories]
        src["categories"] += f" (narrowed by --categories to {cats})"
    yrs = [int(y) for y in (years or sb.get("years") or DEFAULTS["years"])]
    scope = [c for c in full if any(c in scored.get(y, []) for y in yrs)]
    scoped = [c for c in cats if any(c in scored.get(y, []) for y in yrs)]
    if not scoped:
        raise SealError(f"{cap_id}: none of {cats} is a scored category in {yrs}")
    s6 = sb.get("s6") or {}
    return Derived(cap_id=cap_id, arm=arm, keys=keys, need=need, markers=markers, arrival_markers=arrival_markers,
                   flight=flight, alarms=alarms, sat_alarms=sat_alarms, categories=scoped, scope=scope, years=yrs,
                   routable=bool(cap.get("routable")),
                   sources=src, seal=sb, s6_closes=s6.get("closes"), s6_targets=list(s6.get("rows") or []),
                   negctl_rows=list((sb.get("negctl") or {}).get("rows") or []),
                   knife=dict(sb.get("knife") or {}))


# ---------------------------------------------------------------------------------------------------------
# configs: the PIN COMMIT's configs tree (a newer tree's yaml can carry keys the pin refuses), three overlays
# ---------------------------------------------------------------------------------------------------------


def git(repo: str, *args: str, check: bool = True) -> str:
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise SealError(f"git {' '.join(args)}: {p.stderr.strip()[:200]}")
    return p.stdout.strip()


def extract_tree(repo: str, commit: str, path: str, dst: str) -> str:
    """`git archive <commit> <path>` into dst (refused if it exists); returns the tree/blob id of commit:path."""
    if os.path.exists(dst):
        raise SealError(f"refusing: {dst} already exists")
    obj = git(repo, "rev-parse", f"{commit}:{path}")
    stage = dst + ".extract"
    os.makedirs(stage)
    arch = subprocess.run(["git", "-C", repo, "archive", "--format=tar", commit, path], capture_output=True)
    if arch.returncode != 0:
        raise SealError(f"git archive {commit} {path}: {arch.stderr.decode(errors='replace')[:200]}")
    tar = subprocess.run(["tar", "-x", "-C", stage], input=arch.stdout, capture_output=True)
    if tar.returncode != 0:
        raise SealError(f"tar -x: {tar.stderr.decode(errors='replace')[:200]}")
    os.rename(os.path.join(stage, path), dst)
    return obj


def _choose_path(doc: dict, key: dict) -> str:
    """The dotted path to write in THIS file: an existing candidate (path or alt path), else the candidate with the
    deepest existing mapping prefix, else the declared path."""
    cands = [key["path"]] + list(key.get("alt_paths", []) or [])
    for c in cands:
        if L.deep_has(doc, c):
            return c

    def depth(c: str) -> int:
        parts = c.split(".")
        for i in range(len(parts) - 1, 0, -1):
            if isinstance(L.deep_get(doc, ".".join(parts[:i])), dict):
                return i
        return 0
    return max(cands, key=lambda c: (depth(c), c == key["path"]))


def build_configs(src: str, out: str, d: Derived, router_shadow: bool, schema_path: str | None,
                  want_negctl: bool, extra_categories: list[str] | None = None) -> dict:
    """Copy the pin's configs three times and edit the categories' resolved presets:
         arm     the capability's keys set to d.arm (plus router.mode: shadow when asked)
         ctl     the capability's keys DELETED (capability absent: the control arm), same router mode
         negctl  arm + every attack key off (m0's ATTACK_TABLE, checked against the pin's preset schema)
    `extra_categories` (#admission-census: the categories of the census-admitted moat rows) are armed too, so S8 runs
    the capability wherever the router would route it. Every edit is verified yaml-aware (m0lib.apply_overlay).
    Returns the per-arm records."""
    rec = {"source": src, "arms": {}}
    if want_negctl:
        if not schema_path or not os.path.isfile(schema_path):
            raise SealError(f"preset schema source missing: {schema_path} (the attack table cannot be checked)")
        rec["attack_keys_in_scope"] = len(L.check_attack_table(schema_path))
    for arm in ("arm", "ctl") + (("negctl",) if want_negctl else ()):
        dst = os.path.join(out, "configs", arm)
        L.copy_configs(src, dst)
        files: dict[str, dict] = {}
        presets: dict[str, str] = {}
        for cat in list(d.categories) + [c for c in (extra_categories or []) if c not in d.categories]:
            rel = L.resolve_preset(dst, cat)
            if rel is None:
                raise SealError(f"{arm}: no preset resolves for category {cat} (an overlay needs a file to arm; a "
                                "census-admitted category without one cannot be moated armed)")
            presets[cat] = rel
            if rel in files:
                files[rel]["categories"].append(cat)
                continue
            path = os.path.join(dst, rel)
            base = open(path, encoding="utf-8").read()
            doc = L.strict_load(base) or {}
            sets: dict = {}
            deletes: list[str] = []
            actions: list[str] = []
            written: dict[str, str] = {}
            if arm in ("arm", "negctl"):
                for p, v in d.arm.items():
                    fp = _choose_path(doc, d.keys[p])
                    sets[fp] = v
                    written[p] = fp
                    actions.append(f"arm {fp}: {L.fmt_yaml(v)} (was {L.deep_get(doc, fp)!r})")
            else:
                for p, k in d.keys.items():
                    for fp in [p] + list(k.get("alt_paths", []) or []):
                        if L.deep_has(doc, fp):
                            if isinstance(L.deep_get(doc, fp), dict):
                                raise SealError(f"ctl: {rel}: {fp} is a mapping, not a key the control can drop")
                            deletes.append(fp)
                            actions.append(f"delete {fp} (was {L.deep_get(doc, fp)!r}): capability absent")
            if router_shadow:
                sets["router.mode"] = "shadow"
                actions.append("router.mode: shadow (admission + facts receipts; changes no verdict path)")
            if arm == "negctl":
                aset, adel, alog = L.attack_off_edits(doc)
                clash = set(aset) & set(sets)
                if clash:
                    raise SealError(f"negctl: {rel}: armed key(s) {sorted(clash)} collide with the attack table")
                sets.update(aset)
                deletes += adel
                actions += alog
            edited = L.apply_overlay(base, sets, deletes, f"seal {arm} {d.cap_id}") if (sets or deletes) else base
            if edited != base:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(edited)
            files[rel] = {"categories": [cat], "sha256_before": sha256_text(base), "sha256_after": L.sha256_file(path),
                          "actions": actions, "written_paths": written}
        rec["arms"][arm] = {"dir": dst, "files": files, "presets": presets}
    return rec


# ---------------------------------------------------------------------------------------------------------
# rows: one (year, category) unit at a time, aligned BY INDEX (m0lib), banked walls from the vendored allonlib
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Unit:
    year: int
    category: str
    inst: list
    bank: list
    walls: list
    truth: list

    def tok(self, i: int) -> str:
        t = self.truth[i]
        return t.token() if t else "-"


def load_units(ny_root: str, truth_paths: dict[int, str], d: Derived, scored: dict[int, list[str]],
               check_files: bool = True) -> tuple[list[Unit], list[str]]:
    units, notes = [], []
    for y in d.years:
        tp = truth_paths.get(y)
        if not tp or not os.path.isfile(tp):
            raise SealError(f"{y} field results missing: {tp} (a truth gate with nothing to compare is void)")
        truth = L.parse_results_txt(tp, y)
        if not truth:
            raise SealError(f"{tp}: no {y} category parsed")
        for cat in d.categories:
            if cat not in scored.get(y, []):
                notes.append(f"{y}/{cat}: not a scored category that year (skipped)")
                continue
            csvp = L.scored_instances_csv(ny_root, y, cat) if (y == 2026 or os.path.isfile(
                os.path.join(ny_root, "benchmarks/vnncomp2025/benchmarks", cat, "instances.csv"))) else None
            if csvp is None:
                notes.append(f"{y}/{cat}: no instances.csv (skipped)")
                continue
            inst = L.load_instances(csvp, check_files=check_files)
            bp = L.bank_csv(ny_root, y, cat)
            if os.path.isfile(bp):
                bank, walls, _ = A.align_bank_full(inst, A._bank_rows(bp), f"{y}/{cat} bank")
            else:
                bank, walls = ["-"] * len(inst), [None] * len(inst)
                notes.append(f"{y}/{cat}: no bank")
            if cat not in truth:
                notes.append(f"{y}/{cat}: absent from the field results (truth '-')")
            tr = L.align_truth(inst, truth.get(cat), f"{y}/{cat} truth")
            units.append(Unit(y, cat, inst, bank, walls, tr))
    if not units:
        raise SealError("no (year, category) unit to plan over")
    return units, notes


def preset_event(cfg: str, rel: str) -> str:
    return "event:preset_base:^" + re.escape(os.path.join(cfg, rel)) + "$"


def _row(u: Unit, i: int, tag: str, kind: str, cfg: str, preset: str, need: list[str], alarms: list[str],
         flight: list[str], reps_c: int, reps_p: int, group: str, note: str, truth: str | None = None) -> dict:
    r = u.inst[i]
    return L.spec_row(tag=tag, kind=kind, tier="-", year=u.year, category=u.category, onnx=r.onnx, vnnlib=r.vnnlib,
                      budget=r.budget, configs_dir=cfg, preset=preset, truth=truth or u.tok(i), bank=u.bank[i],
                      index=r.index, reps_c=reps_c, reps_p=reps_p, need=L.NEED_SEP.join(need) if need else "-",
                      alarm=L.ALARM_SEP.join(alarms) if alarms else "-",
                      flight=L.FLIGHT_SEP.join(flight) if flight else "-", group=group, note=note)


def row_alarms(d: Derived, truth: str, bank: str) -> list[str]:
    """The alarms a row carries: the every-row alarms, plus the sat-row alarms when the row is truly SAT (field truth
    or the bank says sat)."""
    return list(d.alarms) + (list(d.sat_alarms) if "sat" in (truth, bank) else [])


def round_robin(lists: list[list]) -> list:
    out, k = [], 0
    while any(k < len(x) for x in lists):
        for x in lists:
            if k < len(x):
                out.append(x[k])
        k += 1
    return out


def s5_rows(units: list[Unit], d: Derived, cfg: dict, seed: str, n: int, min_wall_frac: float) -> list[dict]:
    """Arrival rows: banked-UNSAT rows that needed real prover work (banked wall >= frac x budget), cheapest
    first, round-robin over the units; then any banked unsat; then field-unsat rows. The observable is REQUIRED."""
    per = []
    for u in units:
        cand = []
        for i, r in enumerate(u.inst):
            w = u.walls[i]
            if u.bank[i] == "unsat" and w is not None and w >= min_wall_frac * r.budget:
                cand.append((0, w, A.rank(seed, u.year, u.category, i), i))
            elif u.bank[i] == "unsat":
                cand.append((1, w if w is not None else r.budget, A.rank(seed, u.year, u.category, i), i))
            elif u.tok(i) == "unsat" and u.bank[i] not in L.DECIDED:
                cand.append((2, r.budget, A.rank(seed, u.year, u.category, i), i))
        per.append([(u, c) for c in sorted(cand)])
    picked = round_robin(per)[:n]
    if len(picked) < n:
        raise SealError(f"S5: only {len(picked)} arrival candidate rows (want {n})")
    rows = []
    for u, (tier_, w, _, i) in picked:
        preset = cfg["arm"]["presets"][u.category]
        rows.append(_row(u, i, f"S5_{tagbase(u.year, u.category)}_{i:04d}", "AB", cfg["arm"]["dir"], preset, d.need,
                         row_alarms(d, u.tok(i), u.bank[i]), d.flight + [preset_event(cfg["arm"]["dir"], preset)], 1, 0,
                         "seal/S5/arm",
                         f"arrival tier={tier_} banked_wall={w}"))
    return rows


def find_witness(ny_root: str, year: int, category: str, index: int, vnnlib: str) -> str | None:
    """A banked counterexample for this row: `<cat>_artifacts/**/<index:04d>-<vnnlib stem>.counterexample[.gz]`
    (any campaign). The ladder builder re-checks it: in the box, and satisfying a clause of THIS property."""
    base = os.path.join(ny_root, "reports/measured-2026" if year == 2026 else "reports/measured",
                        f"{category}_artifacts")
    stem = L.stem(vnnlib)
    hits = sorted(glob.glob(os.path.join(glob.escape(base), "**", f"{index:04d}-{glob.escape(stem)}.counterexample*"),
                            recursive=True))
    return hits[0] if hits else None


_RX_META = re.compile(r"([\\.^$|?*+()\[\]{}])")


def rx_literal(s: str) -> str:
    """A regex matching `s` literally, escaping only regex metacharacters (readable in logs and the registry)."""
    return _RX_META.sub(r"\\\1", s)


def s7_rows(units: list[Unit], d: Derived, cfg: dict, seed: str, per_cat: int, ny_root: str, out: str,
            reps_c: int, reps_p: int) -> tuple[list[dict], dict]:
    """Truly-SAT negative controls (field truth sat) with every attack off and the capability's observable
    REQUIRED, plus knife-edge ladders (m0 knife_edge) on the controls whose banked witness makes one expressible.
    Sources: the registry's seal.negctl.rows, else the m0 tier's rows, else seeded field-SAT rows per unit
    (witnessed rows first, so a knife-edge is possible)."""
    ncfg = cfg["negctl"]
    by_unit = {(u.year, u.category): u for u in units}
    chosen: list[tuple[Unit, int, str]] = []
    tier = m0_tier_of(d.cap_id)
    explicit = d.negctl_rows or ([dict(id=t[0], category=t[1], onnx=t[2], vnnlib=t[3], year=2026)
                                  for t in tier.rows] if tier else [])
    for e in explicit:
        u = by_unit.get((int(e.get("year", 2026)), e["category"]))
        if u is None:
            continue
        r = L.find_row(u.inst, e["onnx"], e["vnnlib"], f"negctl/{e.get('id')}")
        chosen.append((u, r.index, f"registry/{e.get('id')}"))
    # every unit the explicit rows leave without a control gets seeded field-SAT rows: the judge requires a clean
    # control in EVERY admitted unit, and an m0 tier's rows are 2026-only, so without this a both-year plan of a tier
    # capability could never reach S7 PASS (review: frozen-dual's and fwd-root-screen's 2025 units were uncovered)
    have = {(u.year, u.category) for u, _, _ in chosen}
    for u in units:
        if (u.year, u.category) in have:
            continue
        sat = [i for i in range(len(u.inst)) if u.tok(i) == "sat"]
        sat.sort(key=lambda i: (find_witness(ny_root, u.year, u.category, i, u.inst[i].vnnlib) is None,
                                A.rank(seed, u.year, u.category, i)))
        chosen += [(u, i, "field-sat" if not explicit else "field-sat (unit the explicit rows leave uncovered)")
                   for i in sat[:per_cat]]
    rows: list[dict] = []
    knife = {"ladders": [], "refused": [], "no_witness": []}
    for u, i, why in chosen:
        if u.tok(i) != "sat":
            raise SealError(f"S7: {u.year}/{u.category} row {i} is not field-SAT (truth={u.tok(i)})")
        preset = ncfg["presets"][u.category]
        flight = list(L.NEGCTL_FLIGHT) + d.flight + [preset_event(ncfg["dir"], preset)]
        row = _row(u, i, f"S7N_{tagbase(u.year, u.category)}_{i:04d}", "NEGCTL", ncfg["dir"], preset, d.need,
                   row_alarms(d, "sat", u.bank[i]), flight, reps_c, reps_p, "seal/S7/negctl",
                   f"source={why} banked={u.bank[i]}", truth="sat")
        row["bank"] = "-"   # attacks off: the bank is no comparison point (m0 negative_controls does the same)
        rows.append(row)
        wit = (d.knife.get("witnesses") or {}).get(row["tag"]) or find_witness(ny_root, u.year, u.category, i,
                                                                                row["vnnlib"])
        if not wit:
            knife["no_witness"].append(row["tag"])
            continue
        try:
            man = KE.build_ladder(row["vnnlib"], wit, os.path.join(out, "knife", row["tag"]),
                                  list(d.knife.get("deltas") or L.DEFAULT_DELTAS))
        except L.M0Error as e:
            knife["refused"].append({"tag": row["tag"], "witness": wit, "why": str(e)[:240]})
            continue
        krows = KE.spec_rows_for(man, row, reps_c, reps_p)
        for kr in krows:
            kr["group"] = "seal/S7/knife"
        rows += krows
        knife["ladders"].append({"tag": row["tag"], "witness": wit, "manifest": man["manifest_path"],
                                 "rungs": len(man["rungs"])})
    if not rows:
        raise SealError("S7: no truly-SAT control row in the admitted categories")
    knife["state"] = ("ran" if knife["ladders"] else "not-expressible" if knife["refused"] else "no-witness")
    return rows, knife


def s6_rows(units: list[Unit], d: Derived, cfg: dict, seed: str, o: dict) -> list[dict]:
    """The pre-registered stratified A/B. Strata per unit: target (registry seal.s6.rows), gap (banked undecided,
    field-decided), near (banked decided, wall >= 70% of budget), sat (banked sat), uniform (seeded rest).
    Marginal strata (target, gap, near) get >= 3 reps per arm; the rest s6_other_reps. Arm rows prove the
    overlay arrived (flight levers @config) and record engagement; control rows must NEVER engage."""
    acfg, ccfg = cfg["arm"], cfg["ctl"]
    by_unit = {(u.year, u.category): u for u in units}
    strata: dict[tuple[int, str, int], str] = {}
    for e in d.s6_targets:
        u = by_unit.get((int(e.get("year", 2026)), e["category"]))
        if u is not None:
            strata[(u.year, u.category, L.find_row(u.inst, e["onnx"], e["vnnlib"], f"s6/{e.get('id')}").index)] = "target"
    for u in units:
        def rk(i, u=u):
            return A.rank(seed, u.year, u.category, i)
        idx = range(len(u.inst))
        free = [i for i in idx if (u.year, u.category, i) not in strata]
        gap = sorted((i for i in free if u.bank[i] not in L.DECIDED and u.tok(i) in L.DECIDED), key=rk)[:o["s6_gap_cap"]]
        near = sorted((i for i in free if u.bank[i] in L.DECIDED and u.walls[i] is not None
                       and u.walls[i] >= NEAR_FRAC * u.inst[i].budget and i not in gap), key=rk)[:o["s6_near_cap"]]
        taken = set(gap) | set(near)
        sat = sorted((i for i in free if u.bank[i] == "sat" and i not in taken), key=rk)[:o["s6_sat_cap"]]
        taken |= set(sat)
        uni = sorted((i for i in free if i not in taken), key=rk)[:o["s6_uniform"]]
        for name, ids in (("gap", gap), ("near", near), ("sat", sat), ("uniform", uni)):
            for i in ids:
                strata[(u.year, u.category, i)] = name
    rows = []
    flip = 0
    for (y, cat, i), name in sorted(strata.items(), key=lambda kv: (-kv[0][0], kv[0][1], kv[0][2])):
        u = by_unit[(y, cat)]
        reps = o["s6_marginal_reps"] if name in ("target", "gap", "near") else o["s6_other_reps"]
        ap, cp = acfg["presets"][cat], ccfg["presets"][cat]
        al = row_alarms(d, u.tok(i), u.bank[i])
        a = _row(u, i, f"S6A_{tagbase(y, cat)}_{i:04d}", "AB", acfg["dir"], ap,
                 d.need if name == "target" else [], al, d.flight + [preset_event(acfg["dir"], ap)], reps, 0,
                 f"seal/S6/arm/{name}", f"stratum={name} banked_wall={u.walls[i]}")
        c = _row(u, i, f"S6C_{tagbase(y, cat)}_{i:04d}", "AB", ccfg["dir"], cp, ["!" + n for n in d.need], al,
                 [preset_event(ccfg["dir"], cp)], reps, 0, f"seal/S6/ctl/{name}",
                 f"stratum={name} banked_wall={u.walls[i]}")
        rows += [a, c] if flip % 2 == 0 else [c, a]   # the first arm alternates row by row
        flip += 1
    if not rows:
        raise SealError("S6: no A/B row")
    return rows


def smoke_row_id(r: dict) -> str:
    return f"{r['year']}/{r['category']}/{r['index']}"


def s8_rows(d: Derived, cfg: dict, ny_root: str, truth_paths: dict[int, str], reps_c: int, reps_p: int,
            admitted: set | None = None, smoke: list[str] | None = None) -> list[dict]:
    """The both-year, both-profile moat on the ARMED configs (m0 moat_rows.build: every banked sat/unsat row and
    every field-SAT row of the admitted categories), with the capability's alarms + defect signals on every row and
    the armed levers' receipts required (the moat ran what would ship). With an admission census (`admitted`: the
    census-admitted instance keys) the rows are EXACTLY the moat rows of every scored category whose instance the
    census admits -- the rows the router would route the capability on -- whatever d.categories says."""
    acfg = cfg["arm"]
    cats = None if admitted is not None else set(d.categories)
    if smoke:
        cats = {x.split("/")[1] for x in smoke}
    rows, _ = MR.build(ny_root, acfg["dir"], list(d.years), cats, truth_paths, False, reps_c, reps_p, "S8_", False)
    if admitted is not None:
        rows = [r for r in rows if L.instance_key(r) in admitted]
    if smoke:
        # a SMOKE plan (never evidence: verify_stage_record derives nothing from it) runs exactly the named moat rows
        want = list(dict.fromkeys(smoke))
        have = {smoke_row_id(r): r for r in rows}
        missing = [x for x in want if x not in have]
        if missing:
            raise SealError(f"smoke rows are not moat rows (banked sat/unsat or field-SAT) of the plan's years: {missing}")
        rows = [have[x] for x in want]
    for r in rows:
        al = row_alarms(d, r["truth"], r["bank"])
        r["alarm"] = L.ALARM_SEP.join(al) if al else "-"
        flight = d.flight + ([preset_event(acfg["dir"], r["preset"])] if r["preset"] != "-" else [])
        r["flight"] = L.FLIGHT_SEP.join(flight) if flight else "-"
        r["group"] = f"seal/S8/{r['year']}"
    if not rows:
        raise SealError("S8: the moat selected no row")
    return rows


def partition(rows: list[dict], n: int) -> dict[str, int]:
    """Greedy balance of rows over n shards by their budget ceiling (allonlib.shard_families: deterministic,
    longest first). Rows naming the same instance stay in one shard."""
    groups: dict[tuple[str, str], list[dict]] = collections.OrderedDict()
    for r in rows:
        groups.setdefault(L.instance_key(r), []).append(r)
    fams = [("|".join(k), rs) for k, rs in groups.items()]
    w = (lambda rs: sum(int(r["budget"]) * (int(r["reps_c"]) + int(r["reps_p"])) for r in rs))
    shard_of_group = A.shard_families(fams, n, w)
    out: dict[str, int] = {}
    for name, rs in fams:
        for r in rs:
            out[r["tag"]] = shard_of_group[name]
    return out


# ---------------------------------------------------------------------------------------------------------
# the plan (pre-registered once)
# ---------------------------------------------------------------------------------------------------------


def build_id(pin_copy: str) -> dict:
    p = subprocess.run([pin_copy, "--build-id"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    line = (p.stdout or "").strip()
    m = dict(re.findall(r"(\w+)=(\S+)", line))
    if p.returncode != 0 or "commit" not in m:
        raise SealError(f"{pin_copy} --build-id: rc={p.returncode} {line[:200]!r}")
    return {"line": line, "commit": m.get("commit"), "features": m.get("features"), "dirty": m.get("dirty"),
            "exe": m.get("exe")}


def pin_strings_count(pin_copy: str, fragments: list[str]) -> dict[str, int]:
    p = subprocess.run(["strings", "-a", pin_copy], capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        raise SealError(f"strings -a {pin_copy}: rc={p.returncode}")
    lines = p.stdout.split("\n")
    return {f: sum(1 for ln in lines if f in ln) for f in fragments}


def copy_pin(pin: str, dst: str) -> str:
    """A byte-identical copy under a name no one's `pgrep '^ny'` quiescence gate matches."""
    sha = L.sha256_file(pin)
    if not (os.path.isfile(dst) and L.sha256_file(dst) == sha):
        with open(pin, "rb") as a, open(dst + ".tmp", "wb") as b:
            while True:
                blk = a.read(1 << 22)
                if not blk:
                    break
                b.write(blk)
        os.chmod(dst + ".tmp", 0o755)
        os.replace(dst + ".tmp", dst)
    if L.sha256_file(dst) != sha:
        raise SealError("pin copy sha mismatch")
    return sha


def stage_spec_name(stage: str, k: int, n: int) -> str:
    return f"{stage}.tsv" if n == 1 else f"{stage}_{shard_key(k, n)}.tsv"


def plan_census(census: str, census_box: str | None, cap_id: str, pin_copy: str, pin_sha: str, out: str,
                cfg_src: str, ny_root: str, truth_paths: dict) -> tuple[dict, set, dict]:
    """#admission-census at plan time: the moat corpus -> <out>/census/corpus.tsv, the census (run on the pin copy, or
    a given file copied in) -> <out>/census/census.tsv, bound to THIS seal. Returns (the plan's census record, the
    admitted instance keys, {relpath: sha256} to pre-register)."""
    cdir = os.path.join(out, "census")
    os.makedirs(cdir, exist_ok=True)
    _, entries = census_corpus(ny_root, cfg_src, truth_paths)
    corpus = os.path.join(cdir, "corpus.tsv")
    write_new(corpus, corpus_text(entries))
    corpus_sha = L.sha256_file(corpus)
    tsv = os.path.join(cdir, "census.tsv")
    if census == "run":
        c = run_census(pin_copy, corpus, tsv, census_box)
        source = "run on the pin copy" + (f" through {census_box} build" if census_box else "")
    else:
        if not os.path.isfile(census):
            raise SealError(f"census file missing: {census}")
        with open(census, "rb") as fh:
            raw = fh.read()
        with open(tsv, "xb") as fh:
            fh.write(raw)
        c = parse_census(tsv)
        source = f"given: {os.path.abspath(census)}"
    problems = census_binding_problems(c, pin_sha, corpus_sha, cap_id)
    if problems:
        raise SealError(f"the census does not bind to this seal: {'; '.join(problems)}")
    adm = census_admitted(c, cap_id, entries)
    if not adm:
        raise SealError(f"the census admits no moat row for {cap_id}: the router would route it nowhere, so there is "
                        "nothing to moat (and nothing to seal)")
    cats = sorted({r.split("/", 2)[1] for e in adm.values() for r in e["rows"]})
    cmap = os.path.join(cdir, "corpus_map.json")
    write_new(cmap, json.dumps({e["row_id"]: {"onnx": e["onnx"], "vnnlib": e["vnnlib"], "rows": e["rows"]}
                                for e in entries.values()}, indent=0, sort_keys=True) + "\n")
    counts = census_counts(c, entries)
    rec = {"source": source, "corpus": os.path.relpath(corpus, out), "corpus_sha256": corpus_sha,
           "tsv": os.path.relpath(tsv, out), "tsv_sha256": c["sha256"], "census_sha256": c["census_sha256"],
           "map": os.path.relpath(cmap, out), "binding": c["binding"], "categories": cats,
           "admitted_keys": sorted([list(k) for k in adm]), "counts": counts[cap_id],
           "all_counts": {k: {kk: v[kk] for kk in ("admit", "router", "admit_moat_rows")} for k, v in counts.items()}}
    files = {rec["corpus"]: corpus_sha, rec["tsv"]: c["sha256"], rec["map"]: L.sha256_file(cmap)}
    return rec, set(adm), files


def build_plan(cap_id: str, pin: str, out: str, *, registry: str, repo: str, ny_root: str,
               truth_paths: dict[int, str], stages=STAGES, shards: int = 1, categories: list[str] | None = None,
               years: list[int] | None = None, router_shadow: str = "auto", seed: str | None = None,
               opts: dict | None = None, check_files: bool = True, pin_copy_name: str = "probe_seal",
               census: str | None = None, census_box: str | None = None, s8_mode: str | None = None,
               cert_ledger: str | None = None, x86_rig: dict | None = None, smoke_rows: list[str] | None = None) -> dict:
    """Write the pre-registered plan into `out` (refused if out/plan.json exists): configs from the pin's commit,
    three overlays, one m0-spec/v1 spec per stage (and shard), and plan.json + plan.json.sha256.

    #admission-census: `census` = "run" runs `ny vnncomp census` on the pin copy (through `census_box build --`) over
    the moat corpus (both years, every scored category); a path takes an existing census. Either way it must bind to
    this seal (the pin's binary, the current corpus, the capability, the widest envelope), S8's rows become exactly
    the census-admitted moat rows, and their categories are armed by the overlay.

    Certified S8 (`s8_mode="certified"`, certlib): every S8 row runs ONCE in Competition (reps 1/0); the plan
    pre-registers the certificate ledger (`cert_ledger`, copied in), the x86 replay rig (`x86_rig`, default
    certlib.DEFAULT_RIG: every file's sha256, its witness controls copied in) and the checker budget. `smoke_rows`
    (`year/category/index` moat rows) makes a SMOKE plan: S8 only, ungated, never evidence."""
    o = dict(DEFAULTS)
    o.update(opts or {})
    stages = [s for s in STAGES if s in set(stages)]
    if not stages:
        raise SealError("no stage selected")
    mode = s8_mode or o["s8_mode"]
    if mode not in (C.MODE_CLASSIC, C.MODE_CERTIFIED):
        raise SealError(f"--s8-mode {mode!r}: classic or certified")
    o["s8_mode"] = mode
    if mode == C.MODE_CERTIFIED:
        if "S8" not in stages:
            raise SealError("--s8-mode certified needs S8 in the plan")
        given = [k for k in ("s8_reps_c", "s8_reps_p") if k in (opts or {})]
        if given:
            raise SealError(f"--s8-mode certified runs every S8 row once ({given} given): its floor is the per-row "
                            "certificate / replay / classic sub-moat, never a rep count")
        o["s8_reps_c"], o["s8_reps_p"] = 1, 0
    elif cert_ledger or x86_rig:
        raise SealError("a certificate ledger or x86 rig without --s8-mode certified")
    if smoke_rows:
        if stages != ["S8"]:
            raise SealError("a smoke plan has exactly one stage, S8 (it is ungated and never evidence)")
        if census:
            raise SealError("a smoke plan names its rows; it takes no census")
        bad = [x for x in smoke_rows if not re.match(r"^(2025|2026)/[A-Za-z0-9_.-]+/[0-9]+$", x)]
        if bad:
            raise SealError(f"smoke rows must be year/category/index: {bad}")
    if int(o["arrival_rows"]) < S5_MIN_ROWS:
        raise SealError(f"--arrival-rows {o['arrival_rows']}: S5 needs at least {S5_MIN_ROWS} arrival row (an "
                        "arrival observed on zero rows proves nothing)")
    out = os.path.abspath(out)
    if os.path.exists(os.path.join(out, "plan.json")):
        raise SealError(f"{out}/plan.json exists: the plan is pre-registered once; pick a new --out")
    os.makedirs(out, exist_ok=True)
    reg = load_registry(registry)
    cap = get_capability(reg, cap_id)
    if cap.get("status") == "killed":
        raise SealError(f"{cap_id} is KILLED in the registry")
    pin = os.path.abspath(pin)
    if not os.path.isfile(pin):
        raise SealError(f"pin missing: {pin}")
    pin_copy = os.path.join(out, pin_copy_name)
    pin_sha = copy_pin(pin, pin_copy)
    bid = build_id(pin_copy)
    if bid["dirty"] != "false":
        raise SealError(f"pin {pin} is dirty ({bid['line'][:160]}): a seal is a fact about committed bytes")
    if bid.get("exe") and bid["exe"] != pin_sha:
        raise SealError(f"pin --build-id exe={bid['exe'][:16]} != sha256 {pin_sha[:16]}")
    commit = bid["commit"]
    git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    heads = cap.get("heads") or []
    missing = [h for h in heads if subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", h, commit],
                                                  capture_output=True).returncode != 0]
    if missing:
        raise SealError(f"pin commit {commit[:9]} does not contain the capability's head(s) {missing}: that binary "
                        "cannot carry it")
    cfg_src = os.path.join(out, "configs", "src")
    cfg_tree = extract_tree(repo, commit, "configs", cfg_src)
    schema_path = None
    if "S7" in stages:
        schema_path = os.path.join(out, "preset_schema_mod.rs")
        blob = git(repo, "show", f"{commit}:crates/ny-cli/src/preset/mod.rs")
        write_new(schema_path, blob + "\n")
    yrs = [int(y) for y in (years or (cap.get("seal") or {}).get("years") or o["years"])]
    scored = scored_categories(ny_root, yrs)
    d = derive(cap, cfg_src, scored, categories, yrs)
    want = (router_shadow == "on") or (router_shadow == "auto" and d.routable)
    rs_markers = ["[router] cap="] if want else []
    counts = pin_strings_count(pin_copy, d.markers + rs_markers)
    absent = [m for m, c in counts.items() if c == 0]
    if absent:
        raise SealError(f"marker(s) absent from the pin's strings: {absent} (the observable is dark in this binary)")
    census_rec = None
    admitted = None
    census_files: dict[str, str] = {}
    if census:
        census_rec, admitted, census_files = plan_census(census, census_box, cap_id, pin_copy, pin_sha, out, cfg_src,
                                                         ny_root, truth_paths)
    extra = list((census_rec or {}).get("categories") or [])
    if smoke_rows:
        extra += sorted({x.split("/")[1] for x in smoke_rows} - set(extra))
    cfg = build_configs(cfg_src, out, d, want, schema_path, "S7" in stages, extra_categories=extra or None)
    seed = seed or f"seal|{cap_id}|{pin_sha[:16]}"
    units, notes = load_units(ny_root, truth_paths, d, scored, check_files=check_files)
    stage_rows: dict[str, list[dict]] = {}
    knife = None
    if "S5" in stages:
        stage_rows["S5"] = s5_rows(units, d, cfg["arms"], seed, int(o["arrival_rows"]), float(o["arrival_min_wall_frac"]))
    if "S7" in stages:
        stage_rows["S7"], knife = s7_rows(units, d, cfg["arms"], seed, int(o["negctl_per_category"]), ny_root, out,
                                          int(o["s7_reps_c"]), int(o["s7_reps_p"]))
    if "S6" in stages:
        stage_rows["S6"] = s6_rows(units, d, cfg["arms"], seed, o)
    if "S8" in stages:
        stage_rows["S8"] = s8_rows(d, cfg["arms"], ny_root, truth_paths, int(o["s8_reps_c"]), int(o["s8_reps_p"]),
                                   admitted, smoke_rows)
    cert_block, cert_files = None, {}
    if mode == C.MODE_CERTIFIED:
        cert_block, cert_files = C.plan_block(out, cert_ledger, x86_rig, float(o["checker_budget_s"]), write_new)
        # review(certified-moat): a banked-SAT row must be replayed through ITS checker's code path (v1-2025 /
        # v1-2026 checkv1, v2-2026 the official counterexamples_v2), and a replay counts only beside a PASS and a
        # FAIL control of that same kind; a plan whose rig cannot discriminate a kind its banked-sat rows need can
        # never PASS, so it is refused here rather than discovered after phase A
        kinds = C.control_kinds(cert_block["x86"])
        need = collections.Counter(C.checker_kind(r) for r in stage_rows.get("S8", []) if r["bank"] == "sat")
        miss = sorted(k for k in need if k not in kinds)
        if miss:
            raise SealError(f"--s8-mode certified: the x86 rig has no PASS + FAIL witness control of checker kind(s) "
                            f"{miss}, which {sum(need[k] for k in miss)} banked-sat S8 row(s) need (controls cover "
                            f"{sorted(kinds)}); pass an --x86-rig whose controls cover them")
    plan_stages: dict[str, dict] = {}
    files: dict[str, str] = {}
    for st, rows in stage_rows.items():
        n = int(shards) if st in ("S6", "S8") else 1
        shard_of = partition(rows, n) if n > 1 else {r["tag"]: 1 for r in rows}
        specs = {}
        for k in range(1, n + 1):
            mine = [r for r in rows if shard_of[r["tag"]] == k]
            if not mine:
                continue
            p = os.path.join(out, "specs", stage_spec_name(st, k, n))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            L.write_spec(p, mine)
            errs = L.validate_spec(L.read_spec(p), check_files=check_files)
            if errs:
                raise SealError(f"{st} spec invalid: {errs[:5]}")
            specs[shard_key(k, n)] = os.path.relpath(p, out)
            files[os.path.relpath(p, out)] = L.sha256_file(p)
        runs = sum(int(r["reps_c"]) + int(r["reps_p"]) for r in rows)
        ceiling = sum(int(r["budget"]) * (int(r["reps_c"]) + int(r["reps_p"])) for r in rows) / 3600.0
        plan_stages[st] = {"rows": len(rows), "runs": runs, "ceiling_h": round(ceiling, 2), "shards": n,
                           "specs": specs, "shard_of": {t: shard_key(k, n) for t, k in shard_of.items()}}
    if "S5" in plan_stages:
        plan_stages["S5"]["min_rows"] = int(o["arrival_rows"])
    if "S8" in plan_stages:
        plan_stages["S8"]["mode"] = mode
    if "S7" in plan_stages:
        plan_stages["S7"]["knife"] = knife
    for arm in cfg["arms"].values():
        for rel, f in arm["files"].items():
            files[os.path.relpath(os.path.join(arm["dir"], rel), out)] = f["sha256_after"]
    files.update(census_files)
    files.update(cert_files)
    plan = {
        "schema": PLAN_SCHEMA, "capability": cap_id, "created": now(), "out": out, "seed": seed,
        "registry": {"path": os.path.abspath(registry), "sha256": L.sha256_file(registry),
                     "entry_sha256": sha256_text(canonical(cap))},
        "pin": {"path": pin, "copy": pin_copy, "sha256": pin_sha, "commit": commit, "features": bid["features"],
                "build_id": bid["line"]},
        "configs": {"source": f"git:{commit}:configs", "tree": cfg_tree, "src": cfg_src,
                    "arms": {k: {"dir": v["dir"], "presets": v["presets"],
                                 "files": {rel: dict(f) for rel, f in v["files"].items()}}
                             for k, v in cfg["arms"].items()},
                    "attack_keys_in_scope": cfg.get("attack_keys_in_scope"), "router_shadow": want},
        "arm": d.arm, "keys": {p: {kk: k.get(kk) for kk in ("decl", "default", "type", "alt_paths")}
                               for p, k in d.keys.items()},
        "observable": {"need": d.need, "markers": d.markers, "arrival_markers": d.arrival_markers,
                       "flight": d.flight, "pin_counts": counts},
        "alarms": d.alarms, "sat_alarms": d.sat_alarms, "categories": d.categories, "scope_categories": d.scope, "years": d.years,
        "routable": d.routable,
        # the entry's own pre-registration, carried verbatim so the plan names what it is measured against
        "entry": {k: cap.get(k) for k in ("milestone", "heads", "consumers", "measurement", "routable", "arming",
                                          "default_activation") if cap.get(k) is not None},
        "s6": {"closes": d.s6_closes}, "derivation": d.sources, "options": {k: (list(v) if isinstance(v, tuple) else v)
                                                                             for k, v in o.items()},
        "ny_root": ny_root, "truth": {str(k): v for k, v in truth_paths.items()}, "notes": notes,
        "units": [f"{u.year}/{u.category}" for u in units],
        "stages": plan_stages, "files": files, "vendored_allonlib": VENDORED_ALLONLIB,
    }
    if census_rec is not None:
        plan["census"] = census_rec
    if cert_block is not None:
        plan["certified"] = cert_block
    if smoke_rows:
        plan["smoke"] = {"rows": list(dict.fromkeys(smoke_rows)),
                         "note": "a SMOKE plan: S8 only, ungated; it exercises the pipeline end to end and is never "
                                 "evidence (verify_stage_record derives nothing from it, seal record refuses it)"}
    text = json.dumps(plan, indent=1, sort_keys=True, default=str) + "\n"
    write_new(os.path.join(out, "plan.json"), text)
    write_new(os.path.join(out, "plan.json.sha256"), sha256_text(text) + "\n")
    return plan


def load_plan(out: str, verify_files: bool = True) -> dict:
    """plan.json, refused unless its sha256 matches the one written with it and (verify_files) every spec and edited
    overlay file is byte-identical to what the plan pre-registered."""
    p = os.path.join(out, "plan.json")
    if not os.path.isfile(p):
        raise SealError(f"no plan: {p}")
    text = open(p).read()
    want = open(p + ".sha256").read().strip() if os.path.isfile(p + ".sha256") else None
    if want != sha256_text(text):
        raise SealError(f"{p}: sha256 differs from plan.json.sha256 (the pre-registered plan was edited)")
    plan = json.loads(text)
    if plan.get("schema") != PLAN_SCHEMA:
        raise SealError(f"{p}: schema {plan.get('schema')!r}")
    if verify_files:
        bad = []
        for rel, sha in plan["files"].items():
            f = os.path.join(out, rel)
            if not os.path.isfile(f) or L.sha256_file(f) != sha:
                bad.append(rel)
        if bad:
            raise SealError(f"pre-registered file(s) changed or missing since the plan: {bad[:6]}")
    return plan


# ---------------------------------------------------------------------------------------------------------
# facts: the op-kind / spec-archetype signature of a row (offline-scan/v1; the router's receipts when complete)
# ---------------------------------------------------------------------------------------------------------

ONNX_KIND = {
    "Gemm": "linear", "Conv": "conv", "ConvTranspose": "conv_transpose", "Relu": "relu", "Sigmoid": "sigmoid",
    "Tanh": "tanh", "Softmax": "softmax", "LogSoftmax": "softmax", "BatchNormalization": "bn", "Concat": "concat",
    "MaxPool": "pool", "AveragePool": "pool", "GlobalAveragePool": "pool", "GlobalMaxPool": "pool", "Sign": "sign",
    "Resize": "resize", "Upsample": "resize", "LayerNormalization": "norm", "GroupNormalization": "norm",
    "InstanceNormalization": "norm", "LpNormalization": "norm", "ReduceMean": "linear", "ReduceSum": "linear",
    "Neg": "linear",
}
# (both operands dynamic, one constant operand)
ONNX_BINARY = {"Add": ("residual", "linear"), "Sub": ("residual", "linear"), "Mul": ("bilinear", "linear"),
               "MatMul": ("bilinear", "linear"), "Einsum": ("bilinear", "linear")}
# layout / shape-only: no arithmetic on the activations
ONNX_LAYOUT = frozenset({"Reshape", "Flatten", "Transpose", "Squeeze", "Unsqueeze", "Identity", "Constant", "Shape",
                         "Gather", "Slice", "Cast", "Dropout", "Pad", "Split", "Expand", "ConstantOfShape", "Tile",
                         "Size", "Range"})
CONST_PRODUCERS = frozenset({"Constant", "ConstantOfShape", "Shape", "Size", "Range"})


def _pb_varint(buf, i: int) -> tuple[int, int]:
    x = shift = 0
    n = len(buf)
    while True:
        if i >= n:
            raise SealError("onnx: truncated varint")
        c = buf[i]
        i += 1
        x |= (c & 0x7F) << shift
        if not c & 0x80:
            return x, i
        shift += 7
        if shift > 63:
            raise SealError("onnx: varint overflow")


def _pb_len_fields(buf, start: int, end: int):
    """(field number, (begin, end)) for every length-delimited field of the message buf[start:end]."""
    i = start
    while i < end:
        key, i = _pb_varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            _, i = _pb_varint(buf, i)
        elif wt == 1:
            i += 8
        elif wt == 2:
            ln, i = _pb_varint(buf, i)
            if i + ln > end:
                raise SealError("onnx: field overruns its message")
            yield fn, (i, i + ln)
            i += ln
        elif wt == 5:
            i += 4
        else:
            raise SealError(f"onnx: unsupported wire type {wt}")
    if i != end:
        raise SealError("onnx: message length mismatch")


def _pb_str(buf, a: int, b: int) -> str:
    return buf[a:b].decode("utf-8", errors="replace")


def onnx_nodes(path: str) -> tuple[list[tuple[str, list[str], list[str]]], set[str]]:
    """(op_type, inputs, outputs) of every node of the main graph and its subgraphs, in file order, plus the names of
    the initializers. A minimal protobuf walk (ModelProto.graph=7, GraphProto.node=1/initializer=5/
    sparse_initializer=15, NodeProto.input=1/output=2/op_type=4/attribute=5, AttributeProto.g=6/graphs=11,
    TensorProto.name=8); no onnx package needed."""
    nodes: list[tuple[str, list[str], list[str]]] = []
    inits: set[str] = set()

    def graph(buf, a, b, depth):
        if depth > 8:
            raise SealError("onnx: subgraph nesting > 8")
        for fn, (x, y) in _pb_len_fields(buf, a, b):
            if fn == 1:
                node(buf, x, y, depth)
            elif fn == 5:
                for f2, (p, q) in _pb_len_fields(buf, x, y):
                    if f2 == 8:
                        inits.add(_pb_str(buf, p, q))
            elif fn == 15:
                for f2, (p, q) in _pb_len_fields(buf, x, y):
                    if f2 == 1:
                        for f3, (s, t) in _pb_len_fields(buf, p, q):
                            if f3 == 8:
                                inits.add(_pb_str(buf, s, t))

    def node(buf, a, b, depth):
        ins, outs, op, subs = [], [], "", []
        for fn, (x, y) in _pb_len_fields(buf, a, b):
            if fn == 1:
                ins.append(_pb_str(buf, x, y))
            elif fn == 2:
                outs.append(_pb_str(buf, x, y))
            elif fn == 4:
                op = _pb_str(buf, x, y)
            elif fn == 5:
                for f2, (p, q) in _pb_len_fields(buf, x, y):
                    if f2 in (6, 11):
                        subs.append((p, q))
        nodes.append((op, ins, outs))
        for p, q in subs:
            graph(buf, p, q, depth + 1)

    def model(buf):
        found = False
        for fn, (x, y) in _pb_len_fields(buf, 0, len(buf)):
            if fn == 7:
                graph(buf, x, y, 0)
                found = True
        if not found:
            raise SealError("onnx: no graph in the model")

    if path.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            model(fh.read())
    else:
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size == 0:
                raise SealError(f"onnx: empty file {path}")
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                model(mm)
    return nodes, inits


def classify_ops(nodes: list[tuple[str, list[str], list[str]]], inits: set[str]) -> tuple[list[str], list[str]]:
    """(op-kind tokens, unknown op_types). Constants propagate (initializers, Constant/Shape outputs, nodes whose
    every input is constant); a node whose every input is constant computes no activation and is skipped. An
    op_type outside the table is `other` and is reported: the envelope never seals `other`."""
    const = set(inits) | {""}
    kinds: set[str] = set()
    unknown: set[str] = set()
    for op, ins, outs in nodes:
        dyn = [i for i in ins if i not in const]
        if op in CONST_PRODUCERS or not dyn:
            const.update(outs)
            continue
        if op in ONNX_LAYOUT:
            continue
        if op in ONNX_KIND:
            kinds.add(ONNX_KIND[op])
        elif op in ONNX_BINARY:
            both, one = ONNX_BINARY[op]
            kinds.add(both if len(dyn) >= 2 else one)
        elif op == "Div":
            kinds.add("other" if (len(ins) > 1 and ins[1] not in const) else "linear")
        else:
            kinds.add("other")
            unknown.add(op or "<empty>")
    return sorted(kinds, key=OPKIND_TOKENS.index), sorted(unknown)


def vnnlib_spec(path: str) -> tuple[str | None, str]:
    try:
        prop = L.parse_property(L.read_text(path))
    except L.M0Error as e:
        msg = str(e)
        if "more than one declared network" in msg:
            return "relational", msg[:120]
        if "unsupported compound operand" in msg or "unsupported operand" in msg:
            return "nonlinear", msg[:120]
        return None, msg[:160]
    cl = prop.clauses
    if cl and all(len(c) == 1 and c[0].kind == "relational" for c in cl) and len({c[0].small for c in cl}) == 1:
        return "robustness", f"{len(cl)} single-atom relational clauses, common Y_{cl[0][0].small}"
    if len(cl) == 1:
        return "halfspace", f"one clause of {len(cl[0])} atom(s)"
    return "disjunctive", f"{len(cl)} clauses"


def row_facts(onnx_arg: str, vnnlib: str, cache: dict | None = None) -> dict:
    """offline-scan/v1 facts of one row: op kinds over every model member, the spec archetype, and why a part is
    unknown. Cached by (path, size, mtime)."""
    def ck(p):
        try:
            st = os.stat(p)
            return f"{p}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            return f"{p}|missing"
    key = "|".join(ck(p) for p in L.model_files(onnx_arg) + [vnnlib])
    if cache is not None and key in cache:
        return cache[key]
    ops: set[str] = set()
    unknown: set[str] = set()
    errors = []
    for m in L.model_files(onnx_arg):
        try:
            nodes, inits = onnx_nodes(m)
            k, u = classify_ops(nodes, inits)
            ops |= set(k)
            unknown |= set(u)
        except (OSError, SealError, ValueError) as e:
            errors.append(f"{os.path.basename(m)}: {str(e)[:120]}")
    spec, why = vnnlib_spec(vnnlib)
    f = {"source": FACTS_SOURCE, "ops": sorted(ops, key=OPKIND_TOKENS.index) if not errors else [],
         "unknown_ops": sorted(unknown), "spec": spec, "spec_why": why, "errors": errors}
    if cache is not None:
        cache[key] = f
    return f


def decode_opset(bits) -> list[str]:
    try:
        b = int(bits)
    except (TypeError, ValueError):
        return []
    return [t for i, t in enumerate(OPKIND_TOKENS) if b >> i & 1]


def router_facts(flight: dict | None) -> dict | None:
    """The router record's facts in a run's flight receipt (present only when router.mode was shadow/on)."""
    f = (((flight or {}).get("router") or {}).get("input") or {}).get("facts")
    if not isinstance(f, dict):
        return None
    st = f.get("status")
    known = st == "known" or (isinstance(st, str) and st.lower() == "known")
    return {"known": known, "ops": decode_opset(f.get("ops")), "ops_complete": bool(f.get("ops_complete")),
            "spec": f.get("spec") if f.get("spec") in SPEC_TOKENS else None}


def router_admit(body: str, cap_id: str) -> str | None:
    m = re.search(r"^\[router\] cap=" + re.escape(cap_id) + r" admit=(\S+)", body, re.M)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------------------------------------
# the stage judge (the runner AND the auditor call it)
# ---------------------------------------------------------------------------------------------------------


def stage_rows(out: str, plan: dict, stage: str) -> list[dict]:
    st = plan["stages"].get(stage)
    if not st:
        raise SealError(f"stage {stage} is not in this plan (stages: {sorted(plan['stages'])})")
    rows = []
    for k in sorted(st["specs"]):
        rows += L.read_spec(os.path.join(out, st["specs"][k]))
    return rows


def m0_out(out: str, stage: str, sk: str) -> str:
    return os.path.join(out, stage, f"run_{sk}")


def judged_paths(out: str, plan: dict, stage: str) -> dict[str, str]:
    return {sk: os.path.join(m0_out(out, stage, sk), "judged.tsv") for sk in plan["stages"][stage]["specs"]}


def _read(path: str) -> str | None:
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _flight(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def closes_of(body: str, rx: str | None) -> int | None:
    if not rx:
        return None
    hits = re.findall(rx, body, re.M)
    if not hits:
        return None
    last = hits[-1]
    try:
        return int(last if isinstance(last, str) else last[0])
    except (TypeError, ValueError):
        return None


def judge_stage(plan: dict, stage: str, rows: list[dict], lines: list, all_rows: dict[str, dict], all_lines: list,
                runs: dict, facts_fn=None, cert_facts: dict | None = None) -> dict:
    """One stage's verdict from its m0 judgements (`lines`), the run artefacts (`runs[(tag, profile, rep)]` =
    {raw, raw_path, flight}) and, for the cross-tag instance check, every stage's rows and judgements.

    Applied to every stage: m0's per-run judgement (EMERGENCY is sticky), m0's per-row profile rules (VERDICT-FLIP,
    PROFILE-DECIDED-SPLIT), m0's cross-tag INSTANCE-FLIP over ALL stages, and the binary binding (a run whose flight
    build_id names another commit than the plan's pin is VOID: the pin moved under the run).

    A CERTIFIED S8 (certlib) passes its phase-A rows (`S8_*`) AND its follow-up rows (`S8R_*` re-proofs, `S8F_*`
    classic sub-moats) in `rows`, and the artefact facts (certlib.gather) in `cert_facts`; certlib.decide makes the
    verdict."""
    need = plan["observable"]["need"]
    commit = plan["pin"]["commit"]
    cap_id = plan["capability"]
    by_key = L.collapse_runs(lines)
    by_tag: dict[str, list] = collections.defaultdict(list)
    for j in lines:
        by_tag[j.tag].append(j)
    emerg: list[str] = []
    notes: list[str] = []
    voids: collections.Counter = collections.Counter()
    notrun: list[str] = []
    counts: collections.Counter = collections.Counter()
    out_rows: list[dict] = []
    for r in rows:
        tag = r["tag"]
        rr = {k: r[k] for k in ("tag", "kind", "year", "category", "onnx", "vnnlib", "budget", "truth", "bank",
                                "index", "group")}
        rr["runs"] = []
        for prof, k in M0J.schedule(r):
            j = by_key.get((tag, prof, k))
            meta = runs.get((tag, prof, k)) or {}
            if j is None:
                notrun.append(f"{tag}/{prof}/r{k}")
                counts["notrun"] += 1
                rr["runs"].append({"profile": prof, "rep": k, "status": "NOT-RUN"})
                continue
            body = body_of(meta.get("raw") or "")
            status = j.status
            reasons = list(j.reasons)
            fl = meta.get("flight")
            if status != "EMERGENCY" and not (isinstance(fl, dict) and fl.get("build_id") is not None):
                status = "VOID"   # every seal row carries flight checks, so a run with no build stamp proves no binary
                reasons.append("wrong-binary(no flight build_id: the binary that ran is unproven)")
            elif status != "EMERGENCY" and f"commit={commit}" not in str(fl.get("build_id")):
                status = "VOID"
                reasons.append(f"wrong-binary(flight build_id lacks commit={commit[:12]})")
            if status != "EMERGENCY" and meta.get("raw") is None:
                status = "VOID"
                reasons.append("raw-log-missing")
            run = {"profile": prof, "rep": k, "verdict": j.verdict, "status": status, "outcome": j.outcome,
                   "wall": j.wall, "engaged": engaged(body, need), "admit": router_admit(body, cap_id),
                   "raw": meta.get("raw_path"), "reasons": reasons[:8]}
            if stage == "S6":
                run["closes"] = closes_of(body, (plan.get("s6") or {}).get("closes"))
            bf = router_facts(fl)
            if bf:
                run["binary_facts"] = bf
            counts[status] += 1
            if status == "EMERGENCY":
                emerg.append(f"{tag}/{prof}/r{k}: {';'.join(j.reasons)}")
            elif status == "VOID":
                for reason in reasons:
                    if not reason.startswith("note:"):
                        voids[reason.split("(")[0][:70]] += 1
            if j.outcome == "lost" and status == "OK":
                counts["lost"] += 1
            if j.outcome == "converted" and status == "OK":
                counts["converted"] += 1
            rr["runs"].append(run)
        for st, reason in L.judge_profiles(r, by_tag[tag]):
            (emerg if st == "EMERGENCY" else notes).append(f"{tag}: {reason}")
        ok_runs = [x for x in rr["runs"] if x["status"] == "OK"]
        rr["engaged"] = any(x["engaged"] for x in ok_runs)
        admits = {x.get("admit") for x in rr["runs"] if x.get("admit")}
        rr["admitted"] = None if not admits else ("admit" in admits)
        if rr["engaged"] and rr["admitted"] is False:
            notes.append(f"{tag}: ADMISSION-MISMATCH (the observable engaged while the router's own predicate "
                         f"declined: {sorted(admits)})")
        out_rows.append(rr)
    insts = {L.instance_key(r) for r in rows}
    for st, reason in L.judge_instances(all_rows, all_lines, insts):
        (emerg if st == "EMERGENCY" else notes).append(reason)
    if facts_fn is not None and stage in ("S7", "S8"):
        for rr in out_rows:
            if rr["engaged"]:
                rr["facts"] = facts_fn(rr["onnx"], rr["vnnlib"])
    rec = {"schema": STAGE_SCHEMA, "capability": cap_id, "stage": stage, "emergencies": emerg, "notes": notes,
           "engaged_row_count": sum(1 for x in out_rows if x["engaged"]),
           "void_reasons": dict(voids.most_common(20)), "notrun": notrun[:50], "counts": dict(counts),
           "rows": out_rows}
    reasons: list[str] = []
    sched = sum(len(M0J.schedule(r)) for r in rows)
    clean = counts["OK"] == sched
    if stage == "S5":
        eng = [x["tag"] for x in out_rows if x["engaged"]]
        need_n = int(plan["stages"]["S5"].get("min_rows", DEFAULTS["arrival_rows"]))
        rec["engaged_rows"] = eng
        if emerg:
            status = EMERGENCY
        elif need_n < S5_MIN_ROWS:
            status = VOID
            reasons.append(f"the plan pre-registered N={need_n} arrival rows (< {S5_MIN_ROWS}): an arrival on no row "
                           "proves nothing")
        elif len(eng) >= need_n:
            status = PASS
        else:
            status = VOID
            reasons.append(f"the observable arrived on {len(eng)} of the {need_n} rows required")
        rec["arrival_markers_seen"] = sorted({m for m in plan["observable"]["arrival_markers"]
                                              for x in out_rows if x["engaged"]
                                              for run in x["runs"] if run.get("raw") and run["engaged"]
                                              and m in (_read(run["raw"]) or "")})
    elif stage == "S7":
        knife = plan["stages"]["S7"].get("knife") or {}
        controls = [x for x in out_rows if x["kind"] == "NEGCTL"]
        covered = {f"{x['year']}/{x['category']}" for x in controls if x["runs"]
                   and all(run["status"] == "OK" for run in x["runs"])}
        units = set(plan.get("units") or []) | {f"{x['year']}/{x['category']}" for x in out_rows}
        uncovered = sorted(u for u in units if u not in covered)
        rec["knife"] = {"state": knife.get("state"), "ladders": len(knife.get("ladders") or []),
                        "refused": len(knife.get("refused") or []), "no_witness": len(knife.get("no_witness") or [])}
        rec["uncovered_units"] = uncovered
        if emerg:
            status = EMERGENCY
        elif not clean:
            status = VOID
            reasons.append(f"{sched - counts['OK']} of {sched} control runs are not OK (VOID or not run)")
        elif knife.get("state") in ("ran", "not-expressible") and not uncovered:
            status = PASS
        else:
            status = PARTIAL
            if knife.get("state") not in ("ran", "not-expressible"):
                reasons.append(f"knife-edge {knife.get('state')}")
            if uncovered:
                reasons.append(f"no clean control row for {uncovered}")
    elif stage == "S6":
        pairs: dict[tuple, dict] = collections.defaultdict(dict)
        for x in out_rows:
            arm = "arm" if x["tag"].startswith("S6A_") else "ctl"
            pairs[(x["year"], x["category"], x["index"])][arm] = x
        conv, lost, per = [], [], []
        closes = {"arm": 0, "ctl": 0, "runs_with_closes": 0}
        walls = {"arm": [], "ctl": []}
        arm_engaged = 0
        for key, p in sorted(pairs.items()):
            a, c = p.get("arm"), p.get("ctl")
            if not a or not c:
                continue
            n = len(a["runs"])
            need_k = math.ceil(2 * n / 3)
            ad = sum(1 for run in a["runs"] if run.get("verdict") in L.DECIDED and run["status"] == "OK")
            cd = sum(1 for run in c["runs"] if run.get("verdict") in L.DECIDED and run["status"] == "OK")
            arm_engaged += int(a["engaged"])
            row = {"unit": f"{key[0]}/{key[1]}", "index": key[2], "stratum": a["group"].rsplit("/", 1)[-1],
                   "reps": n, "arm_decided": ad, "ctl_decided": cd, "arm_engaged": a["engaged"]}
            if ad >= need_k and n - cd >= need_k:
                conv.append(row)
                row["delta"] = "converted"
            elif cd >= need_k and n - ad >= need_k:
                lost.append(row)
                row["delta"] = "lost"
            for arm, x in (("arm", a), ("ctl", c)):
                for run in x["runs"]:
                    if run.get("closes") is not None:
                        closes[arm] += run["closes"]
                        closes["runs_with_closes"] += 1
            if ad == n and cd == n:
                for arm, x in (("arm", a), ("ctl", c)):
                    walls[arm] += [float(run["wall"]) for run in x["runs"]
                                   if re.match(r"^[0-9.]+$", str(run.get("wall") or ""))]
            per.append(row)
        med = {k: (sorted(v)[len(v) // 2] if v else None) for k, v in walls.items()}
        # a conversion on a row where the armed arm never showed the observable is not the capability's: both arms
        # ran the same code there, so it is run-to-run noise (CLAUDE.md: some rows are provable ~1 in 4) and is
        # reported, never credited. A loss counts whether or not the arm engaged (the conservative direction).
        credited = [r for r in conv if r["arm_engaged"]]
        decision = "ADOPT" if credited and not lost else "REJECT" if lost else "NULL"
        rec["decision"] = {"decision": decision, "converted": conv, "lost": lost, "pairs": len(per),
                           "converted_unengaged": [r for r in conv if not r["arm_engaged"]],
                           "arm_engaged_rows": arm_engaged, "closes_absolute": closes, "median_wall_both_decided": med,
                           "rule": "converted: armed decides in >= ceil(2n/3) of n reps where control leaves it "
                                   "undecided in >= ceil(2n/3); lost: the reverse; ADOPT = >=1 converted row on which "
                                   "the arm ENGAGED and 0 lost; absolute counts only (no closure ratio)"}
        if emerg:
            status = EMERGENCY
        elif not clean:
            status = VOID
            reasons.append(f"{sched - counts['OK']} of {sched} A/B runs are not OK: re-run them (a VOID is never "
                           "counted)")
        elif arm_engaged == 0:
            status = VOID
            reasons.append("the armed arm never engaged the capability on any row: the A/B measured nothing")
        else:
            status = PASS
    elif stage == "S8" and C.is_certified(plan):
        eng_runs = sum(1 for x in out_rows for run in x["runs"] if run.get("engaged") and run["status"] == "OK")
        rec["engaged_runs"] = eng_runs
        rec["engaged_rows"] = sum(1 for x in out_rows if x["engaged"])
        phase = [r for r in rows if r["tag"].startswith(C.PHASE_A_PREFIX)]
        d = C.decide(plan, phase, lines, cert_facts or {})
        rec["certified"] = d["certified"]
        emerg.extend(d["emergencies"])
        floor = s8_floor_violations(plan, phase)
        rec["floor_violations"] = floor
        lost_n = len(d["lost"]) + int(d["lost_runs"])
        if emerg:
            status = EMERGENCY
        elif d["void"]:
            status = VOID
            reasons.append(f"{len(d['void'])} certification item(s) incomplete: {d['void'][0][:200]}")
            reasons += [v[:200] for v in d["void"][1:6]]
        elif not clean:
            status = VOID
            reasons.append(f"{sched - counts['OK']} of {sched} moat runs are not OK (VOID or not run)")
        elif eng_runs == 0:
            status = VOID
            reasons.append("the capability engaged on no moat run: this moat defends nothing it armed")
        else:
            if lost_n > C.LOST_TOLERANCE:
                reasons.append(f"{lost_n} banked verdict(s) LOST (tolerance {C.LOST_TOLERANCE}): "
                               f"{(d['lost'] or ['follow-up (re-proof / classic sub-moat) runs'])[:5]}")
            reasons += [f"below the S8 floor: {v}" for v in floor]
            status = PARTIAL if (lost_n > C.LOST_TOLERANCE or floor) else PASS
    elif stage == "S8":
        eng_runs = sum(1 for x in out_rows for run in x["runs"] if run.get("engaged") and run["status"] == "OK")
        rec["engaged_runs"] = eng_runs
        rec["engaged_rows"] = sum(1 for x in out_rows if x["engaged"])
        if emerg:
            status = EMERGENCY
        elif not clean:
            status = VOID
            reasons.append(f"{sched - counts['OK']} of {sched} moat runs are not OK (VOID or not run)")
        elif eng_runs == 0:
            status = VOID
            reasons.append("the capability engaged on no moat run: this moat defends nothing it armed")
        else:
            floor = s8_floor_violations(plan, rows)
            rec["floor_violations"] = floor
            if counts["lost"]:
                reasons.append(f"{counts['lost']} banked verdict(s) LOST (a regression to explain; not a flip)")
            reasons += [f"below the S8 floor: {v}" for v in floor]
            status = PARTIAL if (counts["lost"] or floor) else PASS
    else:
        raise SealError(f"unknown stage {stage}")
    if emerg:
        reasons.insert(0, f"{len(emerg)} EMERGENCY finding(s): {emerg[0][:200]}")
    rec.update(status=status, state=STATE_OF[status], rc=RC_OF[status], reasons=reasons, scheduled_runs=sched)
    return rec


def s8_floor_violations(plan: dict, rows: list[dict]) -> list[str]:
    """Why an S8 that ran clean is still not THE moat (PARTIAL, never PASS): fewer than section 1.7's 3 competition +
    1 proof reps on a row (with 1 competition rep the '2 of 3' rule is vacuous, with 0 proof reps there is no Proof
    verdict to reproduce), a plan that left out a year, or an admission scope narrowed by --categories."""
    v = []
    if C.is_certified(plan):
        # the certified floor: ONE armed Competition run per row (the per-row certificate / replay / classic sub-moat
        # replaces the repetition; certlib.decide refuses anything short of it)
        thin = sorted({(int(r["reps_c"]), int(r["reps_p"])) for r in rows if int(r["reps_c"]) < 1})
        if thin:
            v.append(f"reps (competition, proof) {thin}: a certified S8 row needs its armed Competition run")
    else:
        thin = sorted({(int(r["reps_c"]), int(r["reps_p"])) for r in rows
                       if int(r["reps_c"]) < S8_MIN_REPS_C or int(r["reps_p"]) < S8_MIN_REPS_P})
        if thin:
            v.append(f"reps (competition, proof) {thin} < ({S8_MIN_REPS_C}, {S8_MIN_REPS_P}) per row")
    if plan.get("smoke"):
        v.append("a SMOKE plan (named rows, ungated): never the moat")
    miss = [y for y in BOTH_YEARS if y not in {int(x) for x in plan.get("years") or []}]
    if miss:
        v.append(f"not both years: the plan leaves out {miss}")
    if plan.get("census"):
        # #admission-census: S8's scope is the census-admitted instances (both years, every scored category), which
        # --categories cannot narrow; a row set that misses one of them is not the moat.
        want = set(tuple(k) for k in plan["census"].get("admitted_keys") or [])
        have = {L.instance_key(r) for r in rows}
        if not want:
            v.append("the plan's census admits no instance for the capability (an empty moat is not a moat)")
        elif not want <= have:
            v.append(f"{len(want - have)} census-admitted instance(s) are not S8 rows")
        return v
    scope = plan.get("scope_categories")
    if scope is None:
        v.append("the plan carries no admission-scope record (scope_categories), so a narrowed scope cannot be ruled "
                 "out")
    elif sorted(plan.get("categories") or []) != sorted(scope):
        v.append(f"the admission scope was narrowed by --categories to {plan.get('categories')} of {scope}")
    return v


def collect_runs(out: str, plan: dict, stage: str, rows: list[dict]) -> dict:
    runs = {}
    for r in rows:
        sk = plan["stages"][stage]["shard_of"][r["tag"]]
        base = m0_out(out, stage, sk)
        for prof, k in M0J.schedule(r):
            raw = os.path.join(base, "raw", f"{r['tag']}.{prof}.r{k}.raw")
            res = os.path.join(base, "res", f"{r['tag']}.{prof}.r{k}.res")
            runs[(r["tag"], prof, k)] = {"raw": _read(raw), "raw_path": raw, "flight": _flight(res + ".flight.json")}
    return runs


def judge_stage_from_out(out: str, stage: str, write: bool = True, plan: dict | None = None,
                         facts: bool = True, harness: str | None = None, disk_cache: bool = True) -> dict:
    """Judge a stage from its artefacts on disk. With write=True the read AND the write happen under an exclusive
    lock on <out>/<stage>/.judge.lock, so two shards finishing together cannot leave a STALE record behind (the
    later judge always reads the earlier shard's finished judgements). disk_cache=False re-scans every fact in
    memory and never reads <out>/facts_cache.json (the auditor's path: a cache file is as editable as a record)."""
    if not write:
        return _judge_stage_from_out(out, stage, False, plan, facts, harness, disk_cache)
    import fcntl
    lock = os.path.join(os.path.abspath(out), stage, ".judge.lock")
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with open(lock, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            return _judge_stage_from_out(out, stage, True, plan, facts, harness, disk_cache)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _judge_stage_from_out(out: str, stage: str, write: bool, plan: dict | None, facts: bool,
                          harness: str | None, disk_cache: bool = True) -> dict:
    out = os.path.abspath(out)
    if harness is None:   # a manual re-judge: the newest snapshot that ran this stage (it holds the m0 runner used)
        snaps = sorted((p for p in glob.glob(os.path.join(out, f"harness_{stage}_*"))
                        if os.path.isfile(os.path.join(p, "m0", "moat_both_profiles.sh"))), key=os.path.getmtime)
        harness = snaps[-1] if snaps else None
    plan = plan or load_plan(out)
    rows = stage_rows(out, plan, stage)
    lines, judged = [], {}
    for sk, p in judged_paths(out, plan, stage).items():
        if os.path.isfile(p):
            lines += L.read_judged(p)
            judged[p] = L.sha256_file(p)
    certified = stage == "S8" and C.is_certified(plan)
    f_rows = C.followup_rows(out) if C.is_certified(plan) else []
    f_lines = []
    if C.is_certified(plan):
        for p in C.followup_judged(out).values():
            if os.path.isfile(p):
                f_lines += L.read_judged(p)
                if certified:
                    judged[p] = L.sha256_file(p)
    all_rows: dict[str, dict] = {}
    all_lines = []
    for st in plan["stages"]:
        all_rows.update({r["tag"]: r for r in stage_rows(out, plan, st)})
        for p in judged_paths(out, plan, st).values():
            if os.path.isfile(p):
                all_lines += L.read_judged(p)
    # the certified follow-ups (S8R_ re-proofs, S8F_ classic sub-moats) are runs on the same instances: every stage's
    # cross-tag INSTANCE-FLIP check sees them
    all_rows.update({r["tag"]: r for r in f_rows})
    all_lines += f_lines
    cache_path = os.path.join(out, "facts_cache.json")
    cache = ((_flight(cache_path) or {}) if disk_cache else {}) if facts else None
    fn = (lambda o, v: row_facts(o, v, cache)) if facts else None
    runs = collect_runs(out, plan, stage, rows)
    cert_facts = None
    if certified:
        runs.update(C.followup_runs(out))
        phase_res = {r["tag"]: C.res_path(m0_out(out, "S8", plan["stages"]["S8"]["shard_of"][r["tag"]]), r["tag"],
                                          "competition", 1) for r in rows}
        cert_facts = C.gather(out, plan, rows, phase_res, f_lines)
        rec = judge_stage(plan, stage, rows + f_rows, lines + f_lines, all_rows, all_lines, runs, fn, cert_facts)
    else:
        rec = judge_stage(plan, stage, rows, lines, all_rows, all_lines, runs, fn)
    rec.update(out=out, plan_sha256=sha256_text(open(os.path.join(out, "plan.json")).read()),
               pin={k: plan["pin"][k] for k in ("path", "sha256", "commit")}, judged=judged,
               specs={sk: os.path.join(out, p) for sk, p in plan["stages"][stage]["specs"].items()},
               harness=harness, created=now())
    if certified:
        rec["followup_specs"] = {ph: (C.spec_path(out, ph), L.sha256_file(C.spec_path(out, ph)))
                                 for ph in ("reproof", "classic") if os.path.isfile(C.spec_path(out, ph))}
    if plan.get("smoke"):
        rec["smoke"] = True
    if facts and cache is not None and disk_cache:
        write_json_atomic(cache_path, cache)
    if write:
        write_json_atomic(os.path.join(out, stage, "STAGE.json"), rec)
    return rec


SIGNATURE_FIELDS = ("status", "counts", "scheduled_runs", "emergencies")
# a record's fields that are not derived from its inputs (when and with which harness snapshot it was judged)
UNDERIVED_FIELDS = ("created", "harness")


def rejudge_runs(out: str, plan: dict, stage: str, rows: list[dict]) -> list[str]:
    """Every run's m0 judgement re-derived from the run's OWN artefacts (raw log, results file, flight record) with
    m0's judge_run, against the LAST judged.tsv line of that run (the m0 runner moves a re-run's old artefacts aside,
    so the last line is the current artefacts' judgement). judged.tsv is an intermediate a hand can edit and
    `seal judge-stage` will happily re-record; this is what ties a recorded OK back to what the run printed. The
    backstop rc is not stored, so it is taken as 124 exactly when the line itself says the backstop fired."""
    last: dict[tuple[str, str, int], L.Judgement] = {}
    paths = list(judged_paths(out, plan, stage).values())
    bases = {r["tag"]: m0_out(out, stage, plan["stages"][stage]["shard_of"][r["tag"]]) for r in rows}
    rows = list(rows)
    if stage == "S8" and C.is_certified(plan):
        # the certified follow-ups' runs are re-derived from THEIR artefacts the same way
        for ph in ("reproof", "classic"):
            frs = C.read_spec_or_none(C.spec_path(out, ph)) or []
            rows += frs
            bases.update({r["tag"]: C.run_dir(out, ph) for r in frs})
            paths.append(C.followup_judged(out)[ph])
    for p in paths:
        for j in L.read_judged(p):
            last[(j.tag, j.profile, j.rep)] = j
    bad = []
    for r in rows:
        base = bases[r["tag"]]
        for prof, k in M0J.schedule(r):
            j = last.get((r["tag"], prof, k))
            if j is None:
                continue
            raw_p = os.path.join(base, "raw", f"{r['tag']}.{prof}.r{k}.raw")
            res_p = os.path.join(base, "res", f"{r['tag']}.{prof}.r{k}.res")
            backstop = j.wall == "backstop" or any(x.startswith("note:killed-at-backstop") for x in j.reasons)
            again = L.judge_run(r, prof, k, _read(raw_p) or "", _flight(res_p + ".flight.json"), _read(res_p) or "",
                                L.BACKSTOP_RC if backstop else None)
            if (again.status, again.verdict, again.outcome) != (j.status, j.verdict, j.outcome):
                bad.append(f"{r['tag']}/{prof}/r{k}: judged.tsv says {j.status}/{j.verdict}/{j.outcome}, its "
                           f"artefacts say {again.status}/{again.verdict}/{again.outcome}")
    return bad


def verify_stage_record(path: str, cap_id: str | None = None, stage: str | None = None) -> tuple[str, str, dict | None]:
    """(state, note, record) for a recorded STAGE.json: the record is re-derived from its own inputs (the plan,
    sha-checked; its specs; every judged.tsv, sha-bound by the record AND re-derived run by run from the raw logs,
    results files and flight records; the facts, re-scanned in memory) with the SAME judge, and only a record that
    re-derives to the SAME RECORD counts -- every field but when/with which snapshot it was judged. The envelope and
    SR.seal read the rows' engagement and facts, so those are checked, not trusted."""
    try:
        rec = read_json(path)
    except (OSError, ValueError) as e:
        return "open", f"seal record unreadable: {path}: {e}", None
    if rec.get("schema") != STAGE_SCHEMA:
        return "open", f"seal record {path}: schema {rec.get('schema')!r}", None
    if cap_id and rec.get("capability") != cap_id:
        return "open", f"seal record {path} is for {rec.get('capability')!r}, not {cap_id!r}", None
    try:
        if load_plan(rec["out"], verify_files=False).get("smoke"):
            return "open", f"seal record {path} is from a SMOKE plan (named rows, ungated): never evidence", rec
    except Exception:  # noqa: BLE001 - a plan that does not load is refused below, with its reason
        pass
    if stage and rec.get("stage") != stage:
        return "open", f"seal record {path} is stage {rec.get('stage')}, not {stage}", None
    try:
        plan = load_plan(rec["out"])
        again = judge_stage_from_out(rec["out"], rec["stage"], write=False, plan=plan, facts=True, disk_cache=False)
        rejudged = rejudge_runs(rec["out"], plan, rec["stage"], stage_rows(rec["out"], plan, rec["stage"]))
    except Exception as e:  # noqa: BLE001 - whatever breaks the re-derivation, the record derives nothing
        return "open", f"seal record {path} does not re-derive: {str(e)[:200]}", rec
    if again["judged"] != rec.get("judged"):
        return "open", f"seal record {path}: judged.tsv changed since the record (sha mismatch)", rec
    if rejudged:
        return "open", (f"seal record {path}: {len(rejudged)} judged.tsv line(s) do not re-derive from their run "
                        f"artefacts: {rejudged[0][:200]}"), rec
    diff = [f for f in SIGNATURE_FIELDS if canonical(again.get(f)) != canonical(rec.get(f))]
    diff += sorted(f for f in set(again) | set(rec)
                   if f not in UNDERIVED_FIELDS and f not in SIGNATURE_FIELDS
                   and canonical(again.get(f)) != canonical(rec.get(f)))
    if diff:
        return "open", f"seal record {path} re-derives differently on {diff}", rec
    note = (f"seal {rec['stage']} {rec['status']} (pin {os.path.basename(rec['pin']['path'])} "
            f"sha256={rec['pin']['sha256'][:16]}; {rec['counts'].get('OK', 0)}/{rec['scheduled_runs']} runs OK"
            + (f"; {'; '.join(rec['reasons'])}" if rec.get("reasons") else "") + f") [{path}]")
    return STATE_OF.get(rec["status"], "open"), note, rec


# ---------------------------------------------------------------------------------------------------------
# envelope
# ---------------------------------------------------------------------------------------------------------


def evidence_rows(records: list[dict]) -> list[dict]:
    """S7/S8 rows on which the capability ENGAGED, not declined by the router's own predicate."""
    return [r for rec in records for r in rec.get("rows", []) if r.get("engaged") and r.get("admitted") is not False]


def envelope_of(records: list[dict]) -> dict:
    rows = evidence_rows(records)
    ops: set[str] = set()
    specs: set[str] = set()
    other_rows, disputed, unscanned = [], [], []
    binary = collections.Counter()
    for r in rows:
        f = r.get("facts") or {}
        if not f or f.get("errors"):
            unscanned.append(r["tag"])
            continue
        bf = [run["binary_facts"] for run in r.get("runs", []) if run.get("binary_facts")]
        if bf:
            binary["complete" if all(b.get("ops_complete") and b.get("spec") for b in bf) else "incomplete"] += 1
            extra = sorted({o for b in bf for o in b.get("ops", [])} - set(f.get("ops", [])))
            if extra:
                disputed.append({"tag": r["tag"], "binary_only_ops": extra})
                continue
        else:
            binary["absent"] += 1
        row_ops = set(f.get("ops", []))
        if "other" in row_ops:
            other_rows.append({"tag": r["tag"], "unknown_ops": f.get("unknown_ops", [])})
        ops |= row_ops - {"other"}
        if f.get("spec"):
            specs.add(f["spec"])
    return {"schema": ENVELOPE_SCHEMA, "ops": sorted(ops, key=OPKIND_TOKENS.index),
            "specs": sorted(specs, key=SPEC_TOKENS.index), "evidence_rows": len(rows),
            "rows_with_unsealable_other": other_rows, "disputed_rows": disputed, "unscanned_rows": unscanned,
            "facts_source": FACTS_SOURCE, "binary_facts": dict(binary)}


def envelope_covered(ops: list[str], specs: list[str], records: list[dict]) -> tuple[bool, str]:
    """SR.seal's set check: the recorded envelope inside the op sets / archetypes of the S7/S8 evidence rows."""
    env = envelope_of(records)
    eo, es = set(env["ops"]), set(env["specs"])
    bad_o = sorted(set(ops) - eo)
    bad_s = sorted(set(specs) - es)
    if not ops:
        return False, "no envelope recorded"
    if bad_o or bad_s:
        return False, f"envelope exceeds the evidence rows: ops {bad_o} specs {bad_s}"
    if env["disputed_rows"]:
        return False, f"{len(env['disputed_rows'])} evidence row(s) whose binary facts dispute the offline scan"
    return True, (f"envelope ops {sorted(ops, key=OPKIND_TOKENS.index)} specs {sorted(specs)} inside "
                  f"{env['evidence_rows']} engaged S7/S8 evidence row(s)")


def envelope_closure(ops: list[str], specs: list[str], plan: dict, covered: set[tuple[str, str]]) -> dict:
    """The moat rows an envelope ADMITS that S8 never ran. The router routes a sealed capability on any row its own
    predicate admits inside the envelope (ROUTER_PROGRAM section 0 item 3), whatever the category, so the moat must
    cover every banked sat/unsat and field-SAT row (m0 moat_rows, BOTH years, EVERY scored category) whose shape lies
    inside it -- not only the categories the plan ran. The predicate cannot be evaluated offline, so this is the
    conservative superset: a row whose op kinds (offline scan) are a subset of `ops` without `other` and whose spec
    archetype is in `specs`; a row the scan cannot read is counted as inside. ONNX scans are per model (about a
    second for every model of both years); a VNN-LIB is parsed only for rows whose ops already fit."""
    if plan.get("census"):
        # #admission-census: the plan carries the router's OWN admission decisions; the coarse op x archetype superset
        # is never the S8 scope beside it (census_closure is).
        return {"ok": False, "uncovered": {}, "checked_rows": 0,
                "why": "the plan carries an admission census: the envelope's offline closure is not its S8 scope "
                       "(refused; sr_seal checks the census)"}
    truth = {int(k): v for k, v in (plan.get("truth") or {}).items()}
    missing = [y for y in BOTH_YEARS if not truth.get(y) or not os.path.isfile(truth[y])]
    if missing:
        return {"ok": False, "uncovered": {}, "checked_rows": 0,
                "why": f"field results for {missing} unavailable: the envelope's closure cannot be computed"}
    rows, _ = MR.build(plan["ny_root"], plan["configs"]["arms"]["arm"]["dir"], list(BOTH_YEARS), None, truth, False,
                       1, 0, "C_", False)
    eo, es = set(ops), set(specs)
    mops: dict[str, set | None] = {}
    vspec: dict[str, str | None] = {}
    uncovered: collections.Counter = collections.Counter()
    for r in rows:
        if L.instance_key(r) in covered:
            continue
        got: set = set()
        unreadable = False
        for m in L.model_files(r["onnx"]):
            if m not in mops:
                try:
                    mops[m] = set(classify_ops(*onnx_nodes(m))[0])
                except Exception:  # noqa: BLE001 - an unreadable model may still be admitted by the binary
                    mops[m] = None
            if mops[m] is None:
                unreadable = True
            else:
                got |= mops[m]
        if not unreadable and ("other" in got or not got <= eo):
            continue
        v = r["vnnlib"]
        if v not in vspec:
            try:
                vspec[v] = vnnlib_spec(v)[0]
            except Exception:  # noqa: BLE001
                vspec[v] = None
        if vspec[v] is not None and vspec[v] not in es:
            continue
        uncovered[f"{r['year']}/{r['category']}"] += 1
    n = sum(uncovered.values())
    return {"ok": n == 0, "uncovered": dict(sorted(uncovered.items())), "checked_rows": len(rows),
            "why": (f"the envelope admits {n} moat row(s) S8 never ran: {dict(sorted(uncovered.items()))}"
                    if n else f"every moat row inside the envelope ({len(rows)} checked, both years) ran in S8")}


# ---------------------------------------------------------------------------------------------------------
# admission census: the router's OWN admission decisions over the moat corpus (`ny vnncomp census`)
# ---------------------------------------------------------------------------------------------------------


def census_corpus(ny_root: str, configs_dir: str, truth_paths: dict) -> tuple[list[dict], dict]:
    """The census corpus: every m0 moat row (moat_rows.build: banked sat/unsat + field-SAT) of BOTH years and EVERY
    scored category, one manifest entry per distinct instance (onnx, vnnlib) in first-seen order. Returns the moat
    rows and {instance key: {row_id, onnx, vnnlib, budget, rows: ["<year>/<category>/<index>", ...]}}. Row ids are
    opaque (`r00001`): the binary echoes them and never reads a name."""
    truth = {int(k): v for k, v in truth_paths.items()}
    missing = [y for y in BOTH_YEARS if not truth.get(y) or not os.path.isfile(truth[y])]
    if missing:
        raise SealError(f"field results for {missing} unavailable: the census corpus cannot be built")
    rows, _ = MR.build(ny_root, configs_dir, list(BOTH_YEARS), None, truth, False, 1, 0, "C_", False)
    entries: dict = collections.OrderedDict()
    for r in rows:
        k = L.instance_key(r)
        if k not in entries:
            entries[k] = {"row_id": f"r{len(entries) + 1:05d}", "onnx": r["onnx"], "vnnlib": r["vnnlib"],
                          "budget": int(r["budget"]), "rows": []}
        entries[k]["rows"].append(f"{r['year']}/{r['category']}/{r['index']}")
    for e in entries.values():
        for f in ("onnx", "vnnlib"):
            if "\t" in e[f] or "\n" in e[f]:
                raise SealError(f"census corpus: a {f} path carries a tab or newline: {e[f]!r}")
    return rows, entries


def corpus_text(entries: dict) -> str:
    """The manifest `ny vnncomp census --manifest` reads."""
    return f"# {CORPUS_SCHEMA}\n" + "".join(f"{e['row_id']}\t{e['onnx']}\t{e['vnnlib']}\t{e['budget']}\n"
                                           for e in entries.values())


def parse_census(path: str) -> dict:
    """A census TSV, verified: the trailer is the sha256 of every byte above it, the schema and columns are the ones
    this pipeline reads, and every body row has them all. Returns {binding, census_sha256, caps, rows: {row_id: {cap:
    {admit, router, parent_probe, reasons, facts, ops, form}}}}. Refuses (SealError) rather than guesses."""
    try:
        raw = open(path, "rb").read()
    except OSError as e:
        raise SealError(f"census unreadable: {path}: {e}") from None
    text = raw.decode("utf-8", errors="strict")
    body = text[:-1] if text.endswith("\n") else text
    above, _, last = body.rpartition("\n")
    if not last.startswith("# census_sha256\t"):
        raise SealError(f"{path}: no `# census_sha256` trailer (a truncated or hand-made census)")
    want = last.split("\t", 1)[1]
    got = hashlib.sha256((above + "\n").encode()).hexdigest()
    if got != want:
        raise SealError(f"{path}: census_sha256 {want[:16]} does not match its content {got[:16]} (edited)")
    lines = above.split("\n")
    if lines[0] != f"# {CENSUS_SCHEMA}":
        raise SealError(f"{path}: schema line {lines[0]!r}, expected '# {CENSUS_SCHEMA}'")
    binding: dict[str, str] = {}
    rows: dict[str, dict] = collections.OrderedDict()
    for ln in lines[1:]:
        if ln.startswith("# "):
            k, _, v = ln[2:].partition("\t")
            binding[k] = v
            continue
        f = ln.split("\t")
        if len(f) != len(CENSUS_COLUMNS):
            raise SealError(f"{path}: census row with {len(f)} fields: {ln[:120]!r}")
        cell = dict(zip(CENSUS_COLUMNS, f))
        if cell["admit"] not in ("0", "1") or cell["router"] not in ("0", "1"):
            raise SealError(f"{path}: census row with a non-0/1 decision: {ln[:120]!r}")
        rows.setdefault(cell["row_id"], {})[cell["cap"]] = cell
    if binding.get("columns", "").split("\t") != list(CENSUS_COLUMNS):
        raise SealError(f"{path}: columns {binding.get('columns')!r}")
    caps = [c for c in binding.get("caps", "").split(",") if c]
    try:
        n = int(binding.get("corpus_rows", "-1"))
    except ValueError:
        n = -1
    if n != len(rows) or any(set(v) != set(caps) for v in rows.values()):
        raise SealError(f"{path}: {len(rows)} census rows for corpus_rows={binding.get('corpus_rows')} or a row "
                        "missing a capability")
    return {"path": os.path.abspath(path), "binding": binding, "census_sha256": want, "caps": caps, "rows": rows,
            "sha256": hashlib.sha256(raw).hexdigest()}


def census_envelope(c: dict) -> tuple[set[str], set[str]]:
    """The (ops, specs) envelope the census pretended."""
    m = re.fullmatch(r"ops=(\S+) specs=(\S+)", c["binding"].get("envelope", ""))
    if not m:
        raise SealError(f"census envelope unreadable: {c['binding'].get('envelope')!r}")
    return ({t for t in m.group(1).split("+") if t and t != "-"}, {t for t in m.group(2).split("+") if t and t != "-"})


def census_binding_problems(c: dict, pin_sha256: str, corpus_sha256: str, cap_id: str,
                            ops: list[str] | None = None, specs: list[str] | None = None,
                            host: str | None = None) -> list[str]:
    """Why a census does not bind to THIS seal: another binary, another corpus, no decision for the capability, an
    envelope narrower than the one being sealed (the router's envelope check is monotone, so a census under a WIDER
    envelope covers a narrower seal, never the reverse), or -- when `host` is given (the plan's own census host) --
    another host. Review fix (#admission-census): the router's precondition reads the HOST (a device, a thread floor),
    so a census re-run on a host the capability cannot run on admits fewer rows (none, for a device capability), and
    the closure over that census passed while the plan's host routes rows S8 never ran."""
    b = c["binding"]
    v = []
    if host is not None and b.get("host") != host:
        v.append(f"made on host {b.get('host')!r}, not the plan's census host {host!r} (the router's precondition "
                 "reads the host: re-run the census where the plan's was made)")
    if b.get("binary_sha256") != pin_sha256:
        v.append(f"made by binary {str(b.get('binary_sha256'))[:16]}, not the pin {pin_sha256[:16]}")
    if b.get("corpus_sha256") != corpus_sha256:
        v.append(f"over corpus {str(b.get('corpus_sha256'))[:16]}, not the current moat corpus {corpus_sha256[:16]}")
    if cap_id not in c["caps"]:
        v.append(f"no decision for {cap_id} (caps {c['caps']})")
    eo, es = census_envelope(c)
    want_o = set(ops) if ops is not None else set(CENSUS_WIDEST[0])
    want_s = set(specs) if specs is not None else set(CENSUS_WIDEST[1])
    if not want_o <= eo or not want_s <= es:
        v.append(f"census envelope {sorted(eo)} x {sorted(es)} is narrower than {sorted(want_o)} x {sorted(want_s)}")
    return v


def census_admitted(c: dict, cap_id: str, entries: dict) -> dict:
    """{instance key: entry} the census admits for `cap_id` (the `admit` column: CR-RS's own decision with the
    capability pretended sealed and the parent lane's own admission where the census could ask it). Every corpus
    entry must have a census row (else SealError)."""
    out = {}
    for k, e in entries.items():
        row = c["rows"].get(e["row_id"])
        if row is None or cap_id not in row:
            raise SealError(f"census has no row {e['row_id']} / {cap_id} for {k}")
        if row[cap_id]["admit"] == "1":
            out[k] = e
    return out


def run_census(pin: str, manifest: str, out_tsv: str, box: str | None = None, timeout: int = 21600) -> dict:
    """Run `<pin> vnncomp census` (through `<box> build --` when given: a census is a build-class job, not a
    measurement) and return the parsed, verified census. Refuses unless the run printed exactly one
    `CENSUS_DONE rc=0` line and exited 0."""
    cmd = [pin, "vnncomp", "census", "--manifest", manifest, "--out", out_tsv]
    if box:
        cmd = [box, "build", "--"] + cmd
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SealError(f"census run failed to complete: {e}") from None
    done = [ln for ln in p.stdout.splitlines() if ln.startswith("CENSUS_DONE ")]
    if p.returncode != 0 or len(done) != 1 or not done[0].startswith("CENSUS_DONE rc=0 "):
        raise SealError(f"census run rc={p.returncode} done={done[-1:]} stderr={p.stderr.strip()[-300:]!r} (a pin "
                        "built before #admission-census has no `vnncomp census`)")
    return parse_census(out_tsv)


def census_counts(c: dict, entries: dict) -> dict:
    """{cap: {admit, router, admit_moat_rows, categories: {year/category: n}}} over the corpus (instances and the moat
    rows they stand for)."""
    out = {}
    for cap in c["caps"]:
        adm = census_admitted(c, cap, entries)
        rt = sum(1 for e in entries.values() if c["rows"][e["row_id"]][cap]["router"] == "1")
        cats: collections.Counter = collections.Counter()
        for e in adm.values():
            for r in e["rows"]:
                y, cat, _ = r.split("/", 2)
                cats[f"{y}/{cat}"] += 1
        out[cap] = {"admit": len(adm), "router": rt, "admit_moat_rows": sum(cats.values()),
                    "categories": dict(sorted(cats.items()))}
    return out


def census_closure(ops: list[str], specs: list[str], plan: dict, covered: set[tuple[str, str]],
                   rerun=None) -> dict:
    """SR.seal's closure when the plan carries a census: every moat row the census ADMITS for the capability (the
    router's own decision with the capability pretended sealed; both years, every scored category) must have run in
    S8. The census is verified first (trailer, the plan's pre-registered sha) and bound to THIS seal: made by the
    pin, over the CURRENT moat corpus (the banks and field truth today), under an envelope at least as wide as the
    one being sealed. A stale census is re-run on the pin through `rerun(plan, corpus_path) -> census` when one is
    given (`seal record` gives one); without it the closure refuses."""
    pc = plan["census"]
    cap_id = plan["capability"]
    out = plan["out"]
    try:
        c = parse_census(os.path.join(out, pc["tsv"]))
    except SealError as e:
        return {"ok": False, "why": f"the plan's census does not verify: {e}"}
    if c["sha256"] != pc["tsv_sha256"]:
        return {"ok": False, "why": "the plan's census file changed since the plan pre-registered it"}
    try:
        _, entries = census_corpus(plan["ny_root"], plan["configs"]["src"], plan.get("truth") or {})
    except Exception as e:  # noqa: BLE001 - no corpus, no closure
        return {"ok": False, "why": f"the census corpus cannot be rebuilt: {str(e)[:160]}"}
    text = corpus_text(entries)
    corpus_sha = hashlib.sha256(text.encode()).hexdigest()
    host = (pc.get("binding") or {}).get("host")
    if not host:
        return {"ok": False, "why": "the plan's census records no host binding (a census without its host cannot be "
                                    "re-checked)"}
    problems = census_binding_problems(c, plan["pin"]["sha256"], corpus_sha, cap_id, ops, specs, host)
    reran = None
    if problems:
        if rerun is None:
            return {"ok": False, "why": f"the census is stale ({'; '.join(problems)}); re-run it on the pin "
                                        f"(`seal.py census --out {out}`)"}
        path = os.path.join(out, "census", f"corpus_{corpus_sha[:16]}.tsv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.isfile(path):
            write_new(path, text)
        try:
            c = rerun(plan, path)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "why": f"the census is stale ({'; '.join(problems)}) and the re-run failed: "
                                        f"{str(e)[:200]}"}
        again = census_binding_problems(c, plan["pin"]["sha256"], corpus_sha, cap_id, ops, specs, host)
        if again:
            return {"ok": False, "why": f"the re-run census does not bind either: {'; '.join(again)}"}
        reran = c["path"]
    try:
        adm = census_admitted(c, cap_id, entries)
    except SealError as e:
        return {"ok": False, "why": str(e)}
    if not adm:
        # Review fix: an empty census is no closure. The plan refused one (plan_census, s8_floor_violations), but a
        # RE-RUN that admits nothing passed vacuously ("every census-admitted moat row (0 instance(s)) ran in S8").
        return {"ok": False, "uncovered": {}, "admitted": 0, "reran": reran,
                "why": f"the census admits no moat row for {cap_id} (an empty moat is not a moat)"
                       + (f" (census re-run on the pin: {reran})" if reran else "")}
    uncovered: collections.Counter = collections.Counter()
    for k, e in adm.items():
        if k not in covered:
            for r in e["rows"]:
                y, cat, _ = r.split("/", 2)
                uncovered[f"{y}/{cat}"] += 1
    n = sum(uncovered.values())
    tail = f" (census re-run on the pin: {reran})" if reran else ""
    return {"ok": n == 0, "uncovered": dict(sorted(uncovered.items())), "admitted": len(adm), "reran": reran,
            "why": (f"the census admits {n} moat row(s) S8 never ran: {dict(sorted(uncovered.items()))}{tail}" if n
                    else f"every census-admitted moat row ({len(adm)} instance(s), census "
                         f"{c['census_sha256'][:16]}) ran in S8{tail}")}


def census_rerun_on_pin(box: str | None = None):
    """The `rerun` `seal record` hands to census_closure: the plan's own pin copy, through `box build --`."""
    def rerun(plan: dict, corpus: str) -> dict:
        tsv = os.path.join(plan["out"], "census", f"rerun_{os.path.basename(corpus)[len('corpus_'):]}")
        if os.path.isfile(tsv):
            return parse_census(tsv)
        return run_census(plan["pin"]["copy"], corpus, tsv, box)
    return rerun


def sr_seal(ops: list[str], specs: list[str], records: dict[str, dict], rerun=None) -> tuple[bool, str]:
    """SR.seal, the ONE implementation `seal record` and the auditor both call: S7 + S8 records given, the envelope
    inside the S7/S8 evidence rows (envelope_covered), and its closure over every moat row run in S8 -- the CENSUS
    closure (census_closure: the rows the router's own admission routes the capability on) when the S8 plan carries
    an admission census, else the offline envelope closure (envelope_closure), which refuses a plan that has one.
    `rerun` re-runs a stale census on the pin (census_rerun_on_pin); without it a stale census refuses."""
    if "S7" not in records or "S8" not in records:
        return False, "S7/S8 seal records missing"
    ok, why = envelope_covered(ops, specs, [records["S7"], records["S8"]])
    if not ok:
        return False, why
    try:
        plan = load_plan(records["S8"]["out"], verify_files=False)
    except Exception as e:  # noqa: BLE001
        return False, f"the S8 plan does not load: {str(e)[:160]}"
    if plan.get("smoke"):
        return False, "the S8 record is from a SMOKE plan: never evidence"
    covered = {(r["onnx"], r["vnnlib"]) for r in records["S8"].get("rows", [])}
    try:
        if plan.get("census"):
            clo = census_closure(ops, specs, plan, covered, rerun)
        else:
            clo = envelope_closure(ops, specs, plan, covered)
    except Exception as e:  # noqa: BLE001
        return False, f"{why}; but the closure cannot be computed: {str(e)[:160]}"
    if not clo["ok"]:
        return False, f"{why}; but {clo['why']}"
    return True, f"{why}; {clo['why']}"


# ---------------------------------------------------------------------------------------------------------
# registry write-back: textual, scoped to ONE capability block, verified yaml-aware (never hand-written)
# ---------------------------------------------------------------------------------------------------------


def _q(s) -> str:
    return json.dumps(str(s))


def _flow_list(xs) -> str:
    return "[" + ", ".join(_q(x) for x in xs) + "]"


def _cap_span(lines: list[str], cap_id: str) -> tuple[int, int]:
    starts = [i for i, ln in enumerate(lines) if ln.rstrip("\n") == f"- id: {cap_id}"]
    if len(starts) != 1:
        raise SealError(f"registry text: `- id: {cap_id}` found {len(starts)} times at column 0")
    s = starts[0]
    e = s + 1
    while e < len(lines) and not re.match(r"^\S", lines[e]):
        e += 1
    while e > s + 1 and not lines[e - 1].strip():
        e -= 1
    return s, e


def _child_span(lines: list[str], a: int, b: int, indent: int, key: str) -> tuple[int, int] | None:
    rx = re.compile(r"^" + " " * indent + re.escape(key) + r":(\s|$)")
    for i in range(a, b):
        if rx.match(lines[i]):
            j = i + 1
            while j < b and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip(" ")) > indent):
                j += 1
            while j > i + 1 and not lines[j - 1].strip():
                j -= 1
            return i, j
    return None


def _mapping_end(lines: list[str], a: int, b: int, indent: int) -> int:
    """End of the block of lines (from a) belonging to a mapping whose children sit at `indent`."""
    j = a
    while j < b and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip(" ")) >= indent):
        j += 1
    while j > a and not lines[j - 1].strip():
        j -= 1
    return j


def evidence_text(st: str, ev: dict) -> list[str]:
    out = [f"    {st}:\n"]
    for k, v in ev.items():
        if k == "arrivals":
            out.append("      arrivals:\n")
            for a in v:
                out.append(f"        - {{marker: {_q(a['marker'])}, logs: {_flow_list(a['logs'])}, "
                           f"keys: {_flow_list(a['keys'])}}}\n")
        elif k == "zero_ny":
            out.append(f"      zero_ny: {{file: {_q(v['file'])}, grep: {_q(v['grep'])}}}\n")
        elif isinstance(v, list):
            out.append(f"      {k}: {_flow_list(v)}\n")
        else:
            out.append(f"      {k}: {_q(v)}\n")
    return out


def edit_capability(text: str, cap_id: str, evidence: dict[str, dict], record: dict[str, str],
                    routable: dict | None) -> str:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    s, e = _cap_span(lines, cap_id)
    # record: one flow-mapping line
    rs = _child_span(lines, s, e, 2, "record")
    if rs is None or rs[1] - rs[0] != 1:
        raise SealError(f"{cap_id}: record: must be one flow-mapping line")
    cur = yaml.safe_load(lines[rs[0]].split(":", 1)[1]) or {}
    for k, v in record.items():
        cur[k] = v
    lines[rs[0]] = "  record: {" + ", ".join(f"{k}: {v}" for k, v in cur.items()) + "}\n"
    # routable envelope
    if routable:
        s, e = _cap_span(lines, cap_id)
        rt = _child_span(lines, s, e, 2, "routable")
        if rt is None:
            raise SealError(f"{cap_id}: no routable: block to record the envelope in")
        for key in ("envelope", "envelope_specs", "envelope_evidence"):
            val = routable[key]
            new = f"    {key}: " + (_flow_list(val) if isinstance(val, list) else _q(val)) + "\n"
            cs = _child_span(lines, rt[0] + 1, rt[1], 4, key)
            if cs is not None:
                lines[cs[0]:cs[1]] = [new]
            else:
                lines.insert(rt[1], new)
            s, e = _cap_span(lines, cap_id)
            rt = _child_span(lines, s, e, 2, "routable")
    # evidence
    for st, ev in evidence.items():
        s, e = _cap_span(lines, cap_id)
        es = _child_span(lines, s, e, 2, "evidence")
        block = evidence_text(st, ev)
        if es is None:
            rs = _child_span(lines, s, e, 2, "record")
            lines[rs[1]:rs[1]] = ["  evidence:\n"] + block
            continue
        cs = _child_span(lines, es[0] + 1, es[1], 4, st)
        if cs is not None:
            lines[cs[0]:cs[1]] = block
        else:
            end = _mapping_end(lines, es[0] + 1, es[1], 4)
            lines[end:end] = block
    return "".join(lines)


def raise_floors(old: dict, derived: dict[str, str]) -> dict[str, str]:
    """The `record:` updates: a stage's floor rises to its derived state and NEVER falls (a ratchet: the auditor
    reports a derived state below the floor as a regression, so lowering it here would hide one)."""
    return {st: s for st, s in derived.items() if s in RANK and RANK[s] > RANK.get(old.get(st, "open"), 0)}


def record_into_registry(registry: str, out: str, stages=None, replace: bool = False, dry_run: bool = False,
                         census_rerun=None) -> dict:
    """Write the seal evidence of `out` into its capability's registry entry: evidence.<S5..S8> (the STAGE.json the
    auditor re-derives from, plus the S5 arrival fields its existing check reads), record floors raised to the
    derived state (never lowered), and -- when S7 and S8 are both done on a routable capability -- the envelope
    (routable.envelope / envelope_specs / envelope_evidence) and SR: done. The edit touches ONE capability block
    and is verified: the new file must parse (duplicate keys refused) to exactly the old document plus these
    changes. Returns a report; refuses (writes nothing) on any mismatch. `census_rerun` (default: the plan's pin copy
    through <ny_root>/scripts/ny_box.sh build) re-runs a stale admission census for SR.seal."""
    out = os.path.abspath(out)
    plan = load_plan(out)
    if plan.get("smoke"):
        raise SealError(f"{out} is a SMOKE plan (named rows, ungated): it is never recorded as evidence")
    if census_rerun is None and plan.get("census"):
        box = os.path.join(plan["ny_root"], "scripts", "ny_box.sh")
        census_rerun = census_rerun_on_pin(box if os.path.isfile(box) else None)
    cap_id = plan["capability"]
    if os.path.exists(os.path.join(out, "EMERGENCY")):
        raise SealError(f"{out}/EMERGENCY exists: a soundness emergency stops everything, recording included "
                        "(triage it first; nothing from this plan is written while it exists)")
    text = open(registry).read()
    reg = L.strict_load(text)
    cap = get_capability(reg, cap_id)
    stages = [s for s in STAGES if s in (stages or STAGES) and s in plan["stages"]]
    derived: dict[str, tuple[str, str, dict]] = {}
    for st in stages:
        p = os.path.join(out, st, "STAGE.json")
        if os.path.isfile(p):
            derived[st] = verify_stage_record(p, cap_id, st)
    evidence: dict[str, dict] = {}
    rec_up: dict[str, str] = {}
    old_rec = cap.get("record") or {}
    old_ev = cap.get("evidence") or {}
    for st, (state, note, r) in derived.items():
        if state not in ("done", "partial") or r is None:
            continue
        if st in old_ev and not replace:
            raise SealError(f"{cap_id}: evidence.{st} already recorded (pass --replace to supersede it; git keeps "
                            "the old one)")
        pin_note = f"{os.path.basename(r['pin']['path'])} sha256={r['pin']['sha256'][:16]}"
        if st == "S5":
            harness = r.get("harness") or ""
            runner = os.path.join(harness, "m0", "moat_both_profiles.sh") if harness else ""
            if not (runner and os.path.isfile(runner)):
                raise SealError(f"S5 record names no harness snapshot with the m0 runner ({runner!r}): the zero-NY_* "
                                "refusal the auditor checks cannot be cited")
            logs = sorted({run["raw"] for x in r["rows"] if x["engaged"] for run in x["runs"]
                           if run.get("raw") and run["engaged"] and run["status"] == "OK"})
            seen = r.get("arrival_markers_seen") or []
            if not seen:
                raise SealError("S5 record: no arrival marker literal occurs in the engaged logs")
            ev = {"seal": os.path.join(out, st, "STAGE.json"), "pin": pin_note,
                  "zero_ny": {"file": runner, "grep": "grep -vE '^NY_BOX_'"},
                  "arrivals": [{"marker": seen[0], "logs": logs, "keys": sorted(plan["arm"])}],
                  "note": f"seal S5 {r['status']}: the observable on {len(r.get('engaged_rows', []))} row(s) "
                          f"(min {plan['stages']['S5'].get('min_rows')}); overlay-only arming, zero NY_*"}
        else:
            ev = {"seal": os.path.join(out, st, "STAGE.json"), "logs": sorted(r["judged"]),
                  "note": note.split(" [", 1)[0]}
            if st == "S6":
                dd = r.get("decision") or {}
                ev["note"] += (f"; decision {dd.get('decision')}: converted {len(dd.get('converted', []))}, "
                               f"lost {len(dd.get('lost', []))} of {dd.get('pairs')} pairs")
        evidence[st] = ev
    rec_up.update(raise_floors(old_rec, {st: v[0] for st, v in derived.items() if st in evidence}))
    routable = None
    env = None
    sr_why = None
    if cap.get("routable") and derived.get("S7", ("open",))[0] == "done" and derived.get("S8", ("open",))[0] == "done":
        env = envelope_of([derived["S7"][2], derived["S8"][2]])
        # the SAME check the auditor derives SR from: never write `SR: done` (or an envelope the router would read)
        # for an envelope with disputed rows or one whose closure admits rows the moat never ran
        ok, sr_why = sr_seal(env["ops"], env["specs"], {"S7": derived["S7"][2], "S8": derived["S8"][2]},
                             census_rerun)
        if ok:
            env_path = os.path.join(out, "ENVELOPE.json")
            if not dry_run:
                write_json_atomic(env_path, dict(env, records=[os.path.join(out, s, "STAGE.json") for s in ("S7", "S8")],
                                                 sr_seal=sr_why, created=now()))
            routable = {"envelope": env["ops"], "envelope_specs": env["specs"], "envelope_evidence": env_path}
            rec_up.update(raise_floors(old_rec, {"SR": "done"}))
    if not evidence and not routable:
        return {"capability": cap_id, "written": False, "why": "no stage derives done or partial",
                "derived": {k: v[:2] for k, v in derived.items()}, "sr_seal": sr_why}
    new_text = edit_capability(text, cap_id, evidence, rec_up, routable)
    new_reg = L.strict_load(new_text)
    want = copy.deepcopy(reg)
    wc = get_capability(want, cap_id)
    wc.setdefault("record", {}).update(rec_up)
    if evidence:
        wc.setdefault("evidence", {}).update(evidence)
    if routable:
        wc["routable"].update(routable)
    if new_reg != want:
        raise SealError("registry write-back verification failed: parse(edited) != parse(old) + the seal changes")
    if not dry_run:
        tmp = f"{registry}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            fh.write(new_text)
        os.replace(tmp, registry)
    return {"capability": cap_id, "written": not dry_run, "evidence": sorted(evidence), "record": rec_up,
            "envelope": routable, "sr_seal": sr_why, "derived": {k: v[:2] for k, v in derived.items()}}


# ---------------------------------------------------------------------------------------------------------
# certified S8 (certlib): the runner's helpers -- phase-A facts, follow-up derivation, the check list, the ledger
# ---------------------------------------------------------------------------------------------------------


def cert_phase_a(out: str, plan: dict) -> tuple[list[dict], dict, dict, dict]:
    """(phase-A rows, {tag: competition r1 judgement}, {tag: ledger certificate}, {tag: results file})."""
    if not C.is_certified(plan):
        raise SealError(f"{out}: the plan's S8 is not certified (plan --s8-mode certified)")
    rows = stage_rows(out, plan, "S8")
    lines = []
    for p in judged_paths(out, plan, "S8").values():
        lines += L.read_judged(p)
    by_key = L.collapse_runs(lines)
    first = {r["tag"]: by_key.get((r["tag"], "competition", 1)) for r in rows}
    idx = C.ledger_index(C.load_ledger_of(out, plan))
    ledger = {r["tag"]: C.ledger_certificate(idx, r) for r in rows} if idx else {}
    res = {r["tag"]: C.res_path(m0_out(out, "S8", plan["stages"]["S8"]["shard_of"][r["tag"]]), r["tag"],
                                "competition", 1) for r in rows}
    return rows, first, ledger, res


def cert_reproof_outcomes(out: str, plan: dict) -> tuple[list[dict], dict]:
    """(re-proof rows on disk, {tag: certlib.reproof_outcome}) from the re-proof runs and their check artefacts."""
    reproof = C.read_spec_or_none(C.spec_path(out, "reproof"))
    if reproof is None:
        raise SealError("no re-proof spec yet (cert-derive --phase reproof first)")
    lines = L.read_judged(C.followup_judged(out)["reproof"])
    by_key = L.collapse_runs(lines)
    rows, _, _, res = cert_phase_a(out, plan)
    facts = C.gather(out, plan, rows, {}, lines)
    return reproof, {r["tag"]: C.reproof_outcome(by_key.get((r["tag"], "proof", 1)),
                                                 facts["checks"].get(r["tag"]) or {}) for r in reproof}


def cert_derive(out: str, plan: dict, phase: str) -> tuple[str, list[dict], bool]:
    """Write the follow-up spec of `phase` (write-once: an existing spec must equal the derivation, which a resume
    re-uses). Returns (spec path, rows, written now?)."""
    if phase == "reproof":
        rows, first, ledger, _ = cert_phase_a(out, plan)
        notok = [r["tag"] for r in rows if C.classify(r, first.get(r["tag"]), False) in (C.C_NOTOK, C.C_EMERGENCY)]
        if notok:
            raise SealError(f"phase A is not clean ({len(notok)} row(s) not OK, e.g. {notok[:3]}): re-run S8 first")
        want = C.derive_reproof(rows, first, ledger)
    else:
        reproof, outcomes = cert_reproof_outcomes(out, plan)
        emerg = [t for t, o in outcomes.items() if o["state"] == C.R_EMERGENCY]
        if emerg:
            raise SealError(f"re-proof EMERGENCY on {emerg[:3]}: nothing more runs")
        want = C.derive_classic(reproof, outcomes)
    path = C.spec_path(out, phase)
    if os.path.isfile(path):
        have = L.read_spec(path)
        if not C.spec_rows_equal(have, want):
            raise SealError(f"{path} exists and does not re-derive (phase A or the re-proofs changed under it)")
        return path, want, False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    L.write_spec(tmp, want)
    os.replace(tmp, path)
    return path, want, True


def cert_check_list(out: str, plan: dict) -> list[tuple[str, ...]]:
    """(tag, bundle, onnx, vnnlib, report, log) for every re-proof whose Proof run is OK and unsat, whose bundle was
    published, and whose check has not run yet."""
    reproof = C.read_spec_or_none(C.spec_path(out, "reproof")) or []
    by_key = L.collapse_runs(L.read_judged(C.followup_judged(out)["reproof"]))
    todo = []
    for r in reproof:
        j = by_key.get((r["tag"], "proof", 1))
        if j is None or j.status != "OK" or j.verdict != "unsat":
            continue
        bnd = C.bundle_path(C.res_path(C.run_dir(out, "reproof"), r["tag"], "proof", 1))
        rep, logp = C.check_paths(out, r["tag"])
        if not os.path.isfile(bnd) or C.check_log_facts(C._read(logp)).get("ran"):
            continue
        files = L.model_files(r["onnx"])
        todo.append((r["tag"], bnd, files[0], r["vnnlib"], rep, logp))
    return todo


def cert_ledger_entries(out: str) -> list[dict]:
    """Ledger entries from a certified S8 RE-DERIVED now from its artefacts (never read off a record): every row
    whose re-proof ACCEPTed. Refuses a SEAL_OUT with an EMERGENCY (file or derived)."""
    out = os.path.abspath(out)
    if os.path.exists(os.path.join(out, "EMERGENCY")):
        raise SealError(f"{out}/EMERGENCY exists: no certificate is taken from it")
    plan = load_plan(out)
    rec = judge_stage_from_out(out, "S8", write=False, plan=plan, facts=False, disk_cache=False)
    if rec["emergencies"]:
        raise SealError(f"{out}: the certified S8 derives an EMERGENCY: no certificate is taken from it")
    entries = []
    for tag in rec["certified"]["accepted"]:
        ck = rec["certified"]["rows"][tag]["reproof"]["check"]
        d = C._json(ck["report"]) or {}
        entries.append({"onnx_sha256": d["onnx"]["sha256"], "vnnlib_sha256": d["vnnlib"]["sha256"],
                        "report": ck["report"], "report_sha256": ck["report_sha256"], "bundle": ck["bundle"],
                        "bundle_sha256": ck["bundle_sha256"],
                        "checker_build": re.sub(r"\s+", " ", str(d.get("checker_build")))[:400].replace("\t", " "),
                        "source": f"{out}/S8 {tag}"})
    return entries
