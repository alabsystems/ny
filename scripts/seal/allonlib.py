#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Library of the all-on measurement runner (scripts/allon/, configs_allon/ALLON_DECISIONS.md).

Every decision the runner acts on lives here, where scripts/allon/allon_selftest.py pins it. The shell
(scripts/allon/allon_run.sh) only orchestrates. Nothing here runs cargo, an ny binary or the box.

Reused from scripts/m0/m0lib.py (imported, never copied): the VNN-LIB/witness readers, the attack-off table, the
verified yaml overlay editor, instances/bank/field-truth alignment BY INDEX with the key asserted, the m0-spec/v1
runner spec, and the per-run judge (truth contradiction, bank flip, sat-row alarms, VOID rules).

Units, families, order
  A UNIT is one scored (year, category). Units group into FAMILIES (cifar100 = {2026, 2025}; cgan = {2026 cgan2026,
  2025 cgan_2023}; ...), ordered by expected gain: the fixed head FAMILY_HEAD (the coordinator's order), then every
  other family by its count of VBS-gap rows, descending. A --shard k/n takes whole families (families that share an
  overlay yaml never straddle two boxes), balanced greedily on the estimated PHASE B hours.

The four phases (every row is judged from its own log/results file; a sat<->unsat anything is an EMERGENCY)
  A  sat-finding-OFF negative controls on truly-SAT rows under the all-on overlay (every attack key off, through
     m0's attack table), each tier's OWN engagement line required, plus the knife-edge synthetic sats. Runs through
     scripts/m0/moat_both_profiles.sh (competition + proof). Any unsat = EMERGENCY; a tier whose line is absent on
     one of its rows is VOID (never a pass) and blocks S unless explicitly accepted.
  S  the pre-registered, seeded stratified sample: (s1) VBS-gap rows (the gap TSVs, re-resolved BY INDEX against the
     CURRENT bank and field truth), (s2) a moat sample biased to risk, (s3) the phase-A rows; each row once with the
     all-on overlay and once with the shipped configs, first arm alternating per row, competition profile, m0 judge.
     Decision per unit: KEEP-CANDIDATE / NEUTRAL / REVERT / INCOMPLETE / EMERGENCY (MEASURE-ONLY-* for vit).
  B  full both-year sweep (`ny benchmarks run --configs-dir configs_allon`) of KEEP-CANDIDATE units only, one box
     hold per unit, judged index-aligned against the plan's bank snapshot and field truth, plus the campaign's own
     compare_bank.py / moat_truth.py / moat_2025.py; per-row preset_base must name the overlay file.
  C  the decision report over the B records (and S for units B did not sweep).
"""

from __future__ import annotations

import collections
import csv
import dataclasses
import glob
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
M0_DIR = os.path.join(os.path.dirname(HERE), "m0")
sys.path.insert(0, M0_DIR)
import m0lib as L  # noqa: E402

PLAN_SCHEMA = "allon-plan/v1"
DECIDED = L.DECIDED
DEFAULT_SEED = "allon-2026-09-29"
GAP_CAP = 25
UNIFORM_FRAC = 0.10
UNIFORM_MIN = 10
NEAR_TIMEOUT_FRAC = 0.70
KNIFE_MARGIN_MAX = 1e-3
S_RUN_OVERHEAD_S = 12.0     # ny_box.sh measure: two 5 s quiescence windows + process start
A_RUN_OVERHEAD_S = 12.0
# --s-budget-frac: the minimum rows each stratum keeps before the round-robin fill (s3 and s2knife are never capped)
S_BUDGET_MINIMUMS = (("s1gap", 10), ("s2near", 5), ("s2sat", 5), ("s2uni", 5))
B_ROW_OVERHEAD_S = 2.0      # one hold per unit; the sweep's own per-child start
B_UNIT_OVERHEAD_S = 60.0
ENG6_BASE = "7bada08c1"     # ALLON_DECISIONS section 0: measure with an eng6+ binary

# --- decisions
KEEP, NEUTRAL, REVERT, INCOMPLETE, EMERGENCY = "KEEP-CANDIDATE", "NEUTRAL", "REVERT", "INCOMPLETE", "EMERGENCY"
MO_KEEP, MO_NEUTRAL = "MEASURE-ONLY-CANDIDATE", "MEASURE-ONLY-NEUTRAL"
B_DIRECT = "B-DIRECT"        # pre-registered at plan time (--b-direct): no S rows, phase B decides on the full sweep
B_ELIGIBLE = (KEEP, MO_KEEP, B_DIRECT)

# --- families: the coordinator's expected-gain head, then the rest by gap count
FAMILY_HEAD = ("cifar100", "tinyimagenet", "cct", "vgg", "cgan", "vit", "ml4acopf", "cora", "malbeware",
               "relusplitter", "nn4sys", "safenlp")
FAMILY_OF = {
    "cifar100_2024": "cifar100", "tinyimagenet_2024": "tinyimagenet", "challenging_certified_training_2026": "cct",
    "vggnet16_2022": "vgg", "cgan2026": "cgan", "cgan_2023": "cgan", "vit_2023": "vit", "ml4acopf_2024": "ml4acopf",
    "cora_2024": "cora", "malbeware": "malbeware", "relusplitter": "relusplitter", "relusplitter_2026": "relusplitter",
    "nn4sys": "nn4sys", "safenlp_2024": "safenlp", "soundnessbench": "soundnessbench",
    "soundnessbench_2026": "soundnessbench",
}
# vit unsat is unbankable whatever this overlay does (ALLON_DECISIONS 3.5; the double-Infeasible silent-Verified path)
MEASURE_ONLY = frozenset({"vit_2023"})
# 2026 scored categories with no bank: the corpus version the runner sweeps (every prior vit probe ran 1.0)
UNBANKED_VERSION_2026 = {"vit_2023": "1.0"}


class AllonError(L.M0Error):
    """A refusal: the runner will not emit or accept an artefact it cannot vouch for."""


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def rank(seed: str, year: int, category: str, index: int) -> int:
    """The pre-registered seeded order of a row: a pure function of (seed, year, category, instance index)."""
    return int(hashlib.sha256(f"{seed}|{year}|{category}|{index}".encode()).hexdigest()[:16], 16)


def family_of(category: str) -> str:
    return FAMILY_OF.get(category, category)


def short_cat(category: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "", category)
    return s[:18]


# ---------------------------------------------------------------------------------------------------------
# context: every root and input path, overridable (the selftest builds a fake world)
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Ctx:
    ny: str = "<home>/ny"                         # main tree: benchmarks/ (gitignored) + scratchpad/
    wt: str = "<home>/ny-wt-allon"                 # the overlay's worktree: configs/, configs_allon/
    bank_root: str = "<home>/ny"                  # reports/measured{,-2026}/ + bank_manifest.json
    truth26: str = os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt")
    truth25: str = os.path.expanduser("~/vnncomp2025_results/SCORING-ZERO-TOL/results.txt")
    gap26: str = "<home>/ny/scratchpad/vbs/gap26.tsv"
    gap25: str = "<home>/ny/scratchpad/vbs/gap25.tsv"
    margin_audit: str = "<home>/ny/scratchpad/vbs/diag10/root_leak_bank_audit.md"
    seed: str = DEFAULT_SEED
    gap_cap: int = GAP_CAP
    uniform_frac: float = UNIFORM_FRAC
    uniform_min: int = UNIFORM_MIN
    near_frac: float = NEAR_TIMEOUT_FRAC
    knife_margin_max: float = KNIFE_MARGIN_MAX
    a_reps_c: int = 1
    a_reps_p: int = 1
    a_extra_per_cat: int = 2
    s_reps: int = 1
    s_identical_control: bool = False
    s_budget_frac: float | None = None                 # None = the literal strata; else cap S per unit at frac x B
    s_nobody_cap: int = 0                              # optional s1nobody stratum (rows NO field tool solved): off
    b_direct: tuple = ()                               # families that skip S and go straight to B (default: none)
    years: tuple = (2026, 2025)
    categories: tuple = ()                              # empty = every scored category
    a_tiers: tuple | None = None                        # None = every tier

    @property
    def cfg_allon(self) -> str:
        return os.path.join(self.wt, "configs_allon")

    @property
    def cfg_ctl(self) -> str:
        return os.path.join(self.wt, "configs")

    @property
    def bench(self) -> str:
        return self.ny

    def manifest26(self) -> dict:
        p = os.path.join(self.bank_root, "reports/measured-2026/bank_manifest.json")
        if not os.path.isfile(p):
            raise AllonError(f"bank manifest missing: {p}")
        return json.load(open(p))

    @classmethod
    def from_env(cls, **over) -> "Ctx":
        """Defaults, then ALLON_NY-derived defaults (bank root, gap TSVs, margin audit live in the main tree), then
        the explicit ALLON_* overrides, then keyword overrides."""
        c = cls()
        env = os.environ
        if env.get("ALLON_NY"):
            ny = env["ALLON_NY"]
            c.ny, c.bank_root = ny, ny
            c.gap26 = os.path.join(ny, "scratchpad/vbs/gap26.tsv")
            c.gap25 = os.path.join(ny, "scratchpad/vbs/gap25.tsv")
            c.margin_audit = os.path.join(ny, "scratchpad/vbs/diag10/root_leak_bank_audit.md")
        for f, var in (("wt", "ALLON_WT"), ("bank_root", "ALLON_BANK_ROOT"),
                       ("truth26", "ALLON_TRUTH_2026"), ("truth25", "ALLON_TRUTH_2025"), ("gap26", "ALLON_GAP_2026"),
                       ("gap25", "ALLON_GAP_2025"), ("margin_audit", "ALLON_MARGIN_AUDIT"), ("seed", "ALLON_SEED")):
            if env.get(var):
                setattr(c, f, env[var])
        for k, v in over.items():
            if v is not None:
                setattr(c, k, v)
        return c


# ---------------------------------------------------------------------------------------------------------
# presets: the scored resolver (vnncomp v1, unscoped dir) and the sweep resolver (benchmarks run --year Y)
# ---------------------------------------------------------------------------------------------------------


def scored_preset(cfg: str, category: str) -> str | None:
    """`ny vnncomp v1 --configs-dir <cfg>`: newest vnncomp* dir first (m0lib mirror); relpath or None."""
    return L.resolve_preset(cfg, category)


def sweep_preset(cfg: str, year: int, category: str) -> str | None:
    """`ny benchmarks run --year Y --configs-dir <cfg>`: capped at the corpus year (vnncomp<YY> first, then the
    older dirs newest-first: ALLON_DECISIONS 0, a 2025 sweep never loads a vnncomp26 file). Step 0 re-checks every
    unit against the pin's own dry-run."""
    lower = category.lower()
    base = re.sub(r"_20[0-9][0-9]$", "", lower)
    cands = [lower] if base == lower else [lower, base]
    dirs = sorted((p for p in glob.glob(os.path.join(cfg, "vnncomp*")) if os.path.isdir(p)
                   and re.match(r"^vnncomp\d\d$", os.path.basename(p))
                   and int(os.path.basename(p)[7:]) <= year % 100), key=os.path.basename, reverse=True)
    for d in dirs:
        for c in cands:
            p = os.path.join(d, c + ".yaml")
            if os.path.isfile(p):
                return os.path.relpath(p, cfg)
    return None


def same_file(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return os.path.isfile(a) and os.path.isfile(b) and L.sha256_file(a) == L.sha256_file(b)


# ---------------------------------------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Unit:
    year: int
    category: str
    version: str | None
    instances_csv: str
    rows: list
    bank_path: str | None
    bank_sha256: str | None
    bank: list
    walls: list
    truth: list
    scored_allon: str | None
    scored_ctl: str | None
    sweep_allon: str | None
    sweep_ctl: str | None
    identical: bool
    align_mode: str = "-"

    @property
    def uid(self) -> str:
        return f"{self.year}:{self.category}"

    @property
    def family(self) -> str:
        return family_of(self.category)

    @property
    def measure_only(self) -> bool:
        return self.category in MEASURE_ONLY

    @property
    def tagbase(self) -> str:
        return f"{self.year % 100}{short_cat(self.category)}"

    def truth_token(self, i: int) -> str:
        t = self.truth[i]
        return t.token() if t else "-"

    def solved_by_field(self, i: int) -> bool:
        t = self.truth[i]
        return bool(t and (t.fals_ok or t.ver_ok))


def _bank_rows(path: str) -> list[tuple[str, str, str, float | None]]:
    """(onnx, vnnlib, verdict, wall) per bank line; header lines skipped (reports/measured/lsnc_relu.csv has one)."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.reader(fh):
            if len(r) < 5 or r[0].strip().lower() in ("cat", "category", "benchmark"):
                continue
            verdict = (r[4] if len(r) >= 6 else r[3]).strip().lower()
            wtxt = r[5] if len(r) >= 6 else r[4]
            try:
                wall = float(wtxt)
            except ValueError:
                wall = None
            o = r[1].strip() if L.parse_model_field(r[1]) else L._rel(r[1])
            out.append((o, L._rel(r[2]), verdict, wall))
    return out


def align_bank_full(inst: list, rows: list, label: str) -> tuple[list[str], list[float | None], str]:
    """m0lib.align_bank (by index, or a unique-key permutation, else refused), carrying the banked wall too."""
    verdicts = L.align_bank(inst, [(o, v, verdict) for o, v, verdict, _ in rows], label)
    mode = L.ALIGN_MODES.get(label, "?")
    if mode == "by-index":
        return verdicts, [w for *_, w in rows], mode
    by_key = {L._row_identity(o, v): w for o, v, _, w in rows}
    return verdicts, [by_key[L._row_identity(r.onnx_rel, r.vnnlib_rel)] for r in inst], mode


def scored_categories(ctx: Ctx, year: int, truth: dict) -> list[str]:
    if year == 2026:
        return list(ctx.manifest26()["scored_categories"])
    return sorted(truth)


def instances_path(ctx: Ctx, year: int, category: str) -> tuple[str, str | None]:
    """(instances.csv, corpus version) the bank was measured on. 2026: the bank manifest's (rows checked against it);
    an unbanked 2026 category: UNBANKED_VERSION_2026; 2025: the flat corpus."""
    if year == 2026:
        b = ctx.manifest26().get("banks", {}).get(category)
        if b:
            rel = b["instances_csv"]
            m = re.search(r"/([0-9]+\.[0-9]+)/instances\.csv$", rel)
            p = os.path.join(ctx.bench, "benchmarks/vnncomp2026", rel)
            if not os.path.isfile(p):
                raise AllonError(f"instances.csv missing: {p}")
            if L.sha256_file(p) != b["instances_csv_sha256"]:
                with open(p, newline="") as fh:
                    cur = [[c.strip() for c in r[:3]] for r in csv.reader(fh) if r and r[0].strip()]
                want = [[str(c).strip() for c in r[:3]] for r in b.get("instances", [])]
                if cur != want:
                    raise AllonError(f"{p}: rows differ from the bank manifest (the bank was measured on other rows)")
            return p, (m.group(1) if m else None)
        ver = UNBANKED_VERSION_2026.get(category)
        root = os.path.join(ctx.bench, "benchmarks/vnncomp2026/benchmarks", category)
        if ver is None:
            vers = sorted(d for d in os.listdir(root) if re.match(r"^[0-9]+\.[0-9]+$", d)) if os.path.isdir(root) else []
            if len(vers) != 1:
                raise AllonError(f"2026/{category}: no bank and {len(vers)} corpus versions; add it to "
                                 "UNBANKED_VERSION_2026")
            ver = vers[0]
        p = os.path.join(root, ver, "instances.csv")
        if not os.path.isfile(p):
            raise AllonError(f"instances.csv missing: {p}")
        return p, ver
    p = os.path.join(ctx.bench, "benchmarks/vnncomp2025/benchmarks", category, "instances.csv")
    if not os.path.isfile(p):
        raise AllonError(f"instances.csv missing: {p}")
    return p, None


def bank_file(ctx: Ctx, year: int, category: str) -> str:
    return os.path.join(ctx.bank_root, "reports/measured-2026" if year == 2026 else "reports/measured",
                        f"{category}.csv")


def load_units(ctx: Ctx, check_files: bool = True) -> tuple[list[Unit], list[str]]:
    """Every scored (year, category) of ctx.years (filtered by ctx.categories), aligned by index. Returns
    (units, notes). Refuses on any misalignment."""
    notes: list[str] = []
    units: list[Unit] = []
    man = ctx.manifest26() if 2026 in ctx.years else {}
    for year in ctx.years:
        truth = L.parse_results_txt(ctx.truth26 if year == 2026 else ctx.truth25, year)
        if not truth:
            raise AllonError(f"{year}: no category parsed from the field results (a truth gate with nothing to "
                             "compare is void)")
        for cat in scored_categories(ctx, year, truth):
            if ctx.categories and cat not in ctx.categories and family_of(cat) not in ctx.categories:
                continue
            csvp, ver = instances_path(ctx, year, cat)
            inst = L.load_instances(csvp, check_files=check_files)
            bp = bank_file(ctx, year, cat)
            if year == 2026 and cat in man.get("banks", {}):
                if not os.path.isfile(bp):
                    raise AllonError(f"bank missing: {bp}")
                sha = L.sha256_file(bp)
                if sha != man["banks"][cat]["bank_csv_sha256"]:
                    raise AllonError(f"{bp}: sha256 differs from bank_manifest.json (a bank written without its "
                                     "manifest: run scripts/update_measured_2026_index.py --check)")
            if os.path.isfile(bp):
                label = f"{year}/{cat} bank"
                bank, walls, mode = align_bank_full(inst, _bank_rows(bp), label)
                bsha = L.sha256_file(bp)
            else:
                bp, bsha, mode = None, None, "no-bank"
                bank, walls = ["-"] * len(inst), [None] * len(inst)
                notes.append(f"{year}/{cat}: no bank (measured against field truth only)")
            if cat not in truth:
                notes.append(f"{year}/{cat}: absent from the field results (truth '-')")
            tr = L.align_truth(inst, truth.get(cat), f"{year}/{cat} truth")
            sa, sc = scored_preset(ctx.cfg_allon, cat), scored_preset(ctx.cfg_ctl, cat)
            wa, wc = sweep_preset(ctx.cfg_allon, year, cat), sweep_preset(ctx.cfg_ctl, year, cat)
            ident = (sa == sc and wa == wc
                     and same_file(sa and os.path.join(ctx.cfg_allon, sa), sc and os.path.join(ctx.cfg_ctl, sc))
                     and same_file(wa and os.path.join(ctx.cfg_allon, wa), wc and os.path.join(ctx.cfg_ctl, wc)))
            if sa != wa:
                notes.append(f"{year}/{cat}: scored resolver loads {sa} but the {year} sweep resolver loads {wa}")
            units.append(Unit(year, cat, ver, csvp, inst, bp, bsha, bank, walls, tr, sa, sc, wa, wc, ident, mode))
    return units, notes


def gap_counts(ctx: Ctx, units: list[Unit]) -> dict[str, int]:
    return {u.uid: len(s1_candidates(ctx, u)[0]) for u in units}


def order_families(ctx: Ctx, units: list[Unit]) -> list[tuple[str, list[Unit]]]:
    """[(family, units)] in run order: FAMILY_HEAD first, then the rest by VBS-gap rows descending, then name. Within a
    family, 2026 before 2025."""
    fams: dict[str, list[Unit]] = collections.OrderedDict()
    for u in units:
        fams.setdefault(u.family, []).append(u)
    for f in fams:
        fams[f].sort(key=lambda u: -u.year)
    gc = gap_counts(ctx, units)
    head = [f for f in FAMILY_HEAD if f in fams]
    rest = sorted((f for f in fams if f not in FAMILY_HEAD),
                  key=lambda f: (-sum(gc[u.uid] for u in fams[f]), f))
    return [(f, fams[f]) for f in head + rest]


def parse_shard(s: str | None) -> tuple[int, int]:
    if not s:
        return (1, 1)
    m = re.match(r"^([1-9][0-9]*)/([1-9][0-9]*)$", s)
    if not m or int(m.group(1)) > int(m.group(2)):
        raise AllonError(f"--shard {s!r}: want k/n with 1 <= k <= n")
    return int(m.group(1)), int(m.group(2))


def shard_families(ordered: list[tuple[str, list[Unit]]], n: int, weight) -> dict[str, int]:
    """Greedy balance (longest first) of whole families over n shards by weight(family units); deterministic.
    Returns {family: shard (1-based)}."""
    loads = [0.0] * n
    out: dict[str, int] = {}
    for f, us in sorted(ordered, key=lambda fu: (-weight(fu[1]), fu[0])):
        k = min(range(n), key=lambda i: (loads[i], i))
        loads[k] += weight(us)
        out[f] = k + 1
    return out


# ---------------------------------------------------------------------------------------------------------
# PHASE A: tiers, rows, attack-off configs
# ---------------------------------------------------------------------------------------------------------

MR_ANY_UNSAT = (r"margin-row BaB(?: \(concurrent\))?: UNSAT in |"
                r"margin-row BaB: EARLY EXIT — certified UNSAT published")

# DEFECT SIGNALS of the capabilities this overlay arms: two CERTIFIED computations that disagree. Each is an alarm on
# EVERY row the runner judges (phase A and both S arms, sat row or not): the registry's kill line for #mr-ibp-seed is
# "any CROSSED line ... -> emergency", ENGINE_PROGRAM G4 calls a frozen-dual CROSSCHECK-MISMATCH "a soundness
# finding to report", and #backward-interm's CROSSED says "this should be impossible". A defect signal stops the
# campaign exactly as an unsat on a sat row does (m0 judges an alarm hit as an EMERGENCY).
DEFECT_ALARMS = (
    r"\[mr-ibp-seed\] CROSSED layer=",                     # ibp_seed.rs: seed vs tableau enclosures disagree
    r"\[root-box\] CROSSED layer=",                        # root_box.rs: certified reference vs patch-backward
    r"\[backward-interm\] layer=[0-9]+ CROSSED x",         # backward_interm.rs: forward vs backward enclosures
    r"\[frozen-dual\] CROSSCHECK-MISMATCH",                # zono_bab.rs / wide.rs: G4 block vs scalar re-check
    r"\[frozen-dual\] armed=1 .*\bxchk_fail=[1-9]",        # the tier summary's G4 failure count
)
# pin-strings fragments behind DEFECT_ALARMS (literal runs between format placeholders; `strings` stops at the em dash)
DEFECT_MARKERS = ("[mr-ibp-seed] CROSSED layer=", "[root-box] CROSSED layer=", " CROSSED x",
                  "[frozen-dual] CROSSCHECK-MISMATCH depth=", "fd-block-eval/v1 xchk_fail=")


@dataclass(frozen=True)
class ATier:
    name: str
    capability: str
    categories: tuple
    arm: dict              # overlay keys the tier needs, verified in the resolved overlay yaml at plan time
    need: tuple            # the tier's OWN engagement line(s): all must match for the tier to count as engaged
    alarm: tuple = ()      # sat-row alarms specific to the tier (every row here is truly SAT)
    markers: tuple = ()    # pin-strings fragments behind need/alarm (a line the pin cannot print is a dark gate)
    levers: tuple = ()     # (NAME, json value): the flight lever receipt the arming should show (recorded)
    inert: bool = False    # dark BY CONSTRUCTION (ALLON_DECISIONS): reported, never gating


MR_CATS = ("cifar100_2024", "tinyimagenet_2024", "challenging_certified_training_2026", "metaroom_2023", "malbeware")

A_TIERS: dict[str, ATier] = {t.name: t for t in (
    ATier("frozen-dual", "#frozen-dual-bab V1 + #wide-fd-search M4", MR_CATS,
          {"margin_row.frozen_dual_bab": True, "margin_row.frozen_dual_bab_search": "auto"},
          (r"\[frozen-dual\] armed=1 objectives=[0-9]+ .*\bnodes=[1-9][0-9]*",),
          (r"\[frozen-dual\] armed=1 .*\bverdict=all_closed\b",),
          ("[frozen-dual] armed=1 objectives=", "fd-wide-search/v1"),
          (("NY_MARGIN_ROW_FROZEN_DUAL_BAB", True),)),
    ATier("cgan-child-zono", "#cgan-child-zono + #zono-exact-bn", ("cgan2026", "cgan_2023"),
          {"bab.alpha_crown.cgan_child_zono_refresh.enabled": True,
           "bab.alpha_crown.cgan_child_zono_refresh.max_generators": 16384},
          (r"\[cgan-child-zono\] armed=1 max_generators=[0-9]+ .*source=config key=bab\.alpha_crown\."
           r"cgan_child_zono_refresh",
           r"\[cgan-child-zono\] armed=1 children=[1-9][0-9]* installed=[1-9]"),
          (), ("[cgan-child-zono] armed=1 children=", "[cgan-child-zono] armed=1 max_generators=")),
    ATier("fwd-root-screen", "#fwd-root-screen", ("ml4acopf_2024",),
          {"bab.forward_root_screen": True, "bab.forward_root_screen_max_mb": 8192},
          (r"source=config key=bab\.forward_root_screen\b", r"\[fwd-root\] ops=\S+ admitted=1",
           r"\[fwd-root\] union screen_refuted=[0-9]+"),
          (r"\[fwd-root\] union screen_refuted=[0-9]+ precheck_refuted=\S+ union_refuted=([0-9]+) "
           r"new_by_screen=[0-9]+ total=\1\b",),
          ("[fwd-root] union screen_refuted=", " source=config key=bab.forward_root_screen")),
    ATier("joint-refresh", "#bi-window + #joint-refresh", ("challenging_certified_training_2026",),
          {"margin_row.joint_refresh": 4, "margin_row.joint_refresh_secs": 60.0, "margin_row.bi_window": True},
          (r"\[joint-refresh\] done rounds=[1-9]",), (), ("[joint-refresh] done rounds=",),
          (("NY_MARGIN_ROW_JOINT_REFRESH", 4), ("NY_MARGIN_ROW_BI_WINDOW", True))),
    ATier("root-box", "#root-box-source M5b (patch_backward)", ("metaroom_2023", "malbeware"),
          {"margin_row.root_box_source": "patch_backward"},
          (r"\[root-box\] source=patch_backward complete=",), (), ("[root-box] source=patch_backward complete=",),
          (("NY_MARGIN_ROW_ROOT_BOX_SOURCE", "patch_backward"),)),
    ATier("wave-tree", "#wave-beta-bab M6 (tree_mode: wave)", MR_CATS,
          {"margin_row.tree_mode": "wave"},
          (r"\[bbe\] phase=wave armed=1 backend=\S+ B=[0-9]+ .*\bwaves=[1-9]",), (),
          ("[bbe] phase=wave armed=1 backend=", "wave-beta-bab/v1"),
          (("NY_MARGIN_ROW_TREE_MODE", "wave"),)),
    # the registry's pass line for #root-certify-once demands "tier-engaged negative controls 0 unsat"; the overlay
    # arms it on every margin-row category, so it gates here like the program's own M0 tiers
    ATier("root-certify-once", "#root-certify-once M0(b) (+ #bbe-joint-root M5a under it)", MR_CATS,
          {"margin_row.root_certify_once": True, "margin_row.root_search": "joint_f32"},
          (r"\[root-certify-once\] armed=1 iters=",), (),
          ("[root-certify-once] armed=1 iters=", "[root-certify-once] start: iters="),
          (("NY_MARGIN_ROW_ROOT_CERTIFY_ONCE", True),)),
    # #mr-ibp-seed: its two real-ONNX soundness gates are #[ignore] and have never run (registry S3 partial); its
    # kill line is "any CROSSED line or any unsat on a field-sat row" (CROSSED is in DEFECT_ALARMS, on every row)
    ATier("ibp-seed", "#mr-ibp-seed", ("challenging_certified_training_2026",),
          {"margin_row.ibp_seed": True},
          (r"\[mr-ibp-seed\] layer=[0-9]+ neurons=", r"\[mr-ibp-seed\] done layers=[1-9]"), (),
          ("[mr-ibp-seed] layer=", "[mr-ibp-seed] done layers="),
          (("NY_MARGIN_ROW_IBP_SEED", True),)),
    ATier("vit-zono-root", "#vit-zono-root V0/I1", ("vit_2023",),
          {"bab.vit_zono_root": True},
          (r"\[vit-zono-root\] armed=1 stage=stamp model=", r"\[vit-zono-root\] armed=1 admitted=1 "), (),
          ("[vit-zono-root] armed=1 stage=stamp model=", "[vit-zono-root] armed=1 admitted=1 reason=none nodes=")),
    ATier("vit-root-alpha", "#vit-root-alpha V2 (INERT without the vit-margin-alpha lane keys)", ("vit_2023",),
          {"solver.alpha_crown.margin_alpha_search": "joint_f32"},
          (r"\[bbe\] phase=vit_root backend=",), (), ("[bbe] phase=vit_root backend=",), (), inert=True),
)}

# armed capabilities that gate nothing here but whose observables are recorded per run (arrival evidence, S5)
OBSERVABLES = collections.OrderedDict((
    ("root-certify-once", r"\[root-certify-once\] armed=1 iters="),
    ("bbe-joint-root", r"\[bbe\] phase=joint_root iters="),
    ("fd-search", r"\[bbe\] armed=1 backend="),
    ("cuda-fd-refused", r"cuda fd search unavailable:"),
    ("bi-patch", r"\[bi-patch\] done: pairs=[1-9]"),
    ("bi-window", r"\[bi-window\] done: pairs=[1-9]"),
    ("ibp-seed", r"\[mr-ibp-seed\] done layers="),
    ("mr-errlane", r"\[mr-errlane\] done mode=envelope"),
    ("mr-stack", r"\[mr-stack\] main_eval K="),
    ("root-box-declined", r"\[root-box\] source=tableau declined="),
    # the runtime twin of eng6's composition refusal: the loader should have refused the yaml first
    ("root-box-refused", r"\[root-box\] refused reason="),
    ("wave-fallback", r"\[bbe\] phase=wave armed=1 backend=\S+ refused="),
    ("fence", r"\[mo-containment-fence\] emptiness-gate verdict="),
    # vit is measure-only, so these are reported, not stopped on: the registry calls a fence CONTRADICTION "a
    # soundness finding" and disjoint_kept > 0 "VOID / alarm" for #vit-zono-root
    ("fence-contradiction", r"\[mo-containment-fence\] root-verified CONTRADICTION"),
    ("vit-zono-disjoint-kept", r"\[vit-zono-root\] stage=.*\bdisjoint_kept=[1-9]"),
    ("per-node-floor-scaled", r"\[per-node-floor\] refused .*floor_scaled=1"),
    ("mr-unsat", MR_ANY_UNSAT),
))
# pin-strings fragments behind MR_ANY_UNSAT. `strings` extracts ASCII runs only, so a fragment never spans the em dash
# of "EARLY EXIT — certified" (split in two; m0's own markers avoid it the same way)
ALARM_MARKERS = ("margin-row BaB: UNSAT in ", "margin-row BaB: EARLY EXIT ", " certified UNSAT published at ")

# the program's rows (ENGINE_PROGRAM M0 / scripts/m0 TIERS): (row id, year, category, onnx substring, vnnlib substring)
A_FIXED_ROWS = (
    ("t1831", 2026, "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "_prop_idx_1831_"),
    ("t5529", 2026, "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "_prop_idx_5529_"),
    ("t1513", 2026, "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "_prop_idx_1513_"),
    ("c1515", 2026, "cifar100_2024", "CIFAR100_resnet_large.onnx", "_prop_idx_1515_"),
    ("c8452", 2026, "cifar100_2024", "CIFAR100_resnet_large.onnx", "_prop_idx_8452_"),
    ("r4", 2026, "cgan2026", "cGAN_imgSz32_nCh_3.onnx", "cGAN_imgSz32_nCh_3_prop_0_"),
    ("r9", 2026, "cgan2026", "cGAN_imgSz64_nCh_1.onnx", "cGAN_imgSz64_nCh_1_prop_1_"),
    ("14lres_p1", 2026, "ml4acopf_2024", "/14_ieee_ml4acopf-linear-residual.onnx", "/14_ieee_prop1.vnnlib"),
    ("14lnon_p1", 2026, "ml4acopf_2024", "/14_ieee_ml4acopf-linear-nonresidual.onnx", "/14_ieee_prop1.vnnlib"),
    ("118lnon_p5", 2026, "ml4acopf_2024", "/118_ieee_ml4acopf-linear-nonresidual.onnx", "/118_ieee_prop5.vnnlib"),
    ("300base_p2", 2026, "ml4acopf_2024", "/300_ieee_ml4acopf.onnx", "/300_ieee_prop2.vnnlib"),
    ("300lres_p2", 2026, "ml4acopf_2024", "/300_ieee_ml4acopf-linear-residual.onnx", "/300_ieee_prop2.vnnlib"),
    ("8090", 2026, "challenging_certified_training_2026", "/cifar10_eps8_cnn7.onnx", "_idx8090_"),
    ("8397", 2026, "challenging_certified_training_2026", "/cifar10_eps8_wide_cnn7.onnx", "_idx8397_"),
)
# categories that get `a_extra_per_cat` seeded truly-SAT rows on top of the fixed ones ("+ vit and cct sat rows from
# the banks"; the root-box categories have no program row at all)
A_EXTRA_CATS = ("challenging_certified_training_2026", "vit_2023", "metaroom_2023", "malbeware")
# knife-edge ladders: (source row id, witness path relative to ctx.ny unless absolute)
A_KNIFE = (
    ("r4", "reports/measured-2026/cgan2026_artifacts/witnesses/"
           "0004-cGAN_imgSz32_nCh_3_prop_0_input_eps_0.070_output_eps_0.078.counterexample"),
    ("r9", "reports/measured-2026/cgan2026_artifacts/witnesses/"
           "0009-cGAN_imgSz64_nCh_1_prop_1_input_eps_0.025_output_eps_0.030.counterexample"),
    ("14lres_p1", "scratchpad/w8b/v2/FRS_14lres_p1_frs_satctl.res"),
    ("14lnon_p1", "scratchpad/w8b/v2/FRS_14lnon_p1_frs_satctl.res"),
    ("118lnon_p5", "scratchpad/w8b/v2/FRS_118lnon_p5_frs_satctl.res"),
    ("300base_p2", "scratchpad/w8b/v2/FRS_300base_p2_frs_satctl.res"),
    ("300lres_p2", "scratchpad/w8b/v2/FRS_300lres_p2_frs_satctl.res"),
)


def tiers_for(category: str, selected: tuple | None = None) -> list[ATier]:
    return [t for t in A_TIERS.values() if category in t.categories and (selected is None or t.name in selected)]


def verify_arming(ctx: Ctx, category: str, tiers: list[ATier]) -> list[str]:
    """Every tier key must be set to its value in the overlay file the scored resolver loads (fail closed: a tier
    the overlay does not arm cannot be negative-controlled)."""
    rel = scored_preset(ctx.cfg_allon, category)
    if rel is None:
        return [f"{category}: no overlay preset resolves"]
    doc = L.strict_load(open(os.path.join(ctx.cfg_allon, rel)).read()) or {}
    bad = []
    for t in tiers:
        for k, v in t.arm.items():
            got = L.deep_get(doc, k)
            if not L.same_json(got, v):
                bad.append(f"{category} ({rel}): tier {t.name} needs {k}={v!r}, overlay has {got!r}")
    return bad


def _unit(units: list[Unit], year: int, category: str) -> Unit | None:
    for u in units:
        if u.year == year and u.category == category:
            return u
    return None


def a_rows(ctx: Ctx, units: list[Unit]) -> tuple[list[dict], list[str]]:
    """The phase-A instances: [{rowid, unit, index, tiers}] (truly SAT, each with >= 1 gating or inert tier)."""
    notes: list[str] = []
    rows: list[dict] = []
    seen: set = set()
    for rowid, year, cat, osub, vsub in A_FIXED_ROWS:
        u = _unit(units, year, cat)
        if u is None:
            continue
        tiers = tiers_for(cat, ctx.a_tiers)
        if not tiers:
            continue
        r = L.find_row(u.rows, osub, vsub, f"A/{rowid}")
        if u.truth_token(r.index) != "sat":
            raise AllonError(f"A/{rowid}: not a field-SAT row (truth={u.truth_token(r.index)}); a negative control "
                             "must be truly SAT")
        rows.append({"rowid": rowid, "unit": u, "index": r.index, "tiers": tiers})
        seen.add((u.uid, r.index))
    for cat in A_EXTRA_CATS:
        u = _unit(units, 2026, cat)
        tiers = tiers_for(cat, ctx.a_tiers)
        if u is None or not tiers:
            continue
        cands = [i for i in range(len(u.rows)) if (u.uid, i) not in seen
                 and (u.truth_token(i) == "sat" or u.bank[i] == "sat") and u.truth_token(i) != "unsat"]
        # tiers need time to engage: prefer rows whose budget is >= 100 s when there are enough of them
        long_ = [i for i in cands if u.rows[i].budget >= 100]
        pool = long_ if len(long_) >= ctx.a_extra_per_cat else cands
        pick = sorted(pool, key=lambda i: rank(ctx.seed + "|A", u.year, u.category, i))[:ctx.a_extra_per_cat]
        if len(pick) < ctx.a_extra_per_cat:
            notes.append(f"A/{cat}: only {len(pick)} truly-SAT extra rows available")
        for i in sorted(pick):
            rows.append({"rowid": f"x{i}", "unit": u, "index": i, "tiers": tiers})
            seen.add((u.uid, i))
    return rows, notes


def a_need_alarm(tiers: list[ATier]) -> tuple[list[str], list[str], list[str]]:
    need, alarm, markers = [], [MR_ANY_UNSAT, *DEFECT_ALARMS], list(ALARM_MARKERS) + list(DEFECT_MARKERS)
    for t in tiers:
        if not t.inert:
            need += list(t.need)
        alarm += [a for a in t.alarm if a not in alarm]
        markers += [m for m in t.markers if m not in markers]
    return need, alarm, markers


def build_attack_off_configs(ctx: Ctx, dst: str, categories: list[str]) -> list[dict]:
    """Copy configs_allon to dst and switch every attack key off (m0 ATTACK_TABLE, checked against the worktree's
    preset schema) in the file each category resolves to. The tiers stay armed exactly as the overlay arms them."""
    schema = os.path.join(ctx.wt, "crates/ny-cli/src/preset/mod.rs")
    L.check_attack_table(schema)
    L.copy_configs(ctx.cfg_allon, dst)
    recs, done = [], set()
    for cat in categories:
        rel = scored_preset(dst, cat)
        if rel is None:
            raise AllonError(f"A: no preset resolves for {cat} under {dst}")
        if rel in done:
            continue
        done.add(rel)
        path = os.path.join(dst, rel)
        base = open(path, encoding="utf-8").read()
        doc = L.strict_load(base) or {}
        sets, deletes, log = L.attack_off_edits(doc)
        edited = L.apply_overlay(base, sets, deletes, "allon A attack-off")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(edited)
        recs.append({"category": cat, "preset": rel, "sha256_before": sha256_text(base),
                     "sha256_after": L.sha256_file(path), "actions": log})
    return recs


def build_a_spec(ctx: Ctx, units: list[Unit], outdir: str) -> dict:
    """PHASE A plan: attack-off configs, NEGCTL rows (m0-spec/v1, need='-': engagement is judged per tier by
    a_judge), and the knife-edge ladders (m0 knife_edge.py, same configs). Returns the A manifest."""
    rows, notes = a_rows(ctx, units)
    cats = sorted({r["unit"].category for r in rows})
    problems = []
    for c in cats:
        problems += verify_arming(ctx, c, tiers_for(c, ctx.a_tiers))
    if problems:
        raise AllonError("the overlay does not arm what phase A would control:\n  " + "\n  ".join(problems))
    cfg = os.path.join(outdir, "cfg_allon_attackoff")
    recs = build_attack_off_configs(ctx, cfg, cats) if cats else []
    spec_rows = []
    for r in rows:
        u, i = r["unit"], r["index"]
        inst = u.rows[i]
        pre = scored_preset(cfg, u.category)
        need, alarm, _ = a_need_alarm(r["tiers"])
        flight = list(L.NEGCTL_FLIGHT) + ["event:preset_base:^" + re.escape(os.path.join(cfg, pre)) + "$"]
        spec_rows.append(L.spec_row(
            tag=f"A_{u.tagbase}_{r['rowid']}", kind="NEGCTL", tier="-", year=u.year, category=u.category,
            onnx=inst.onnx, vnnlib=inst.vnnlib, budget=inst.budget, configs_dir=cfg, preset=pre, truth="sat",
            bank="-", index=i, reps_c=ctx.a_reps_c, reps_p=ctx.a_reps_p, need="-",
            alarm=L.ALARM_SEP.join(alarm), flight=L.FLIGHT_SEP.join(flight), group=f"A/{u.category}",
            note=f"tiers={','.join(t.name for t in r['tiers'])} gating={'yes' if need else 'no'} banked={u.bank[i]} "
                 f"unit={u.uid}"))
    manifest = {"schema": "allon-A/v1", "configs": cfg, "attack_off": recs, "notes": notes,
                "rows": [{"tag": s["tag"], "uid": r["unit"].uid, "index": r["index"],
                          "tiers": [t.name for t in r["tiers"]]} for s, r in zip(spec_rows, rows)],
                "knife": []}
    negspec = os.path.join(outdir, "a_negctl_spec.tsv")
    L.write_spec(negspec, spec_rows)
    return manifest


def plan_knife(ctx: Ctx, manifest: dict, outdir: str, negspec: str, run) -> list[dict]:
    """Knife-edge ladders through scripts/m0/knife_edge.py (`run` executes a command list and returns
    (rc, stdout+stderr)). Every source row must exist in the A spec; a missing witness is a refusal."""
    out = []
    tags = {row["tag"]: row for row in L.read_spec(negspec)}
    kspec = os.path.join(outdir, "a_knife_spec.tsv")
    for rowid, wrel in A_KNIFE:
        src = [t for t in tags if t.endswith("_" + rowid)]
        if not src:
            continue  # its category is not in this plan
        if len(src) != 1:
            raise AllonError(f"knife {rowid}: {len(src)} A rows match")
        w = wrel if os.path.isabs(wrel) else os.path.join(ctx.ny, wrel)
        if not os.path.isfile(w):
            raise AllonError(f"knife {rowid}: witness missing: {w}")
        kdir = os.path.join(outdir, "knife", rowid)
        rc, text = run([sys.executable, os.path.join(M0_DIR, "knife_edge.py"), "--from-spec", negspec, "--row",
                        src[0], "--witness", w, "--out-dir", kdir, "--spec-out", kspec,
                        "--reps-competition", str(ctx.a_reps_c), "--reps-proof", str(ctx.a_reps_p)])
        if rc != 0:
            raise AllonError(f"knife {rowid}: knife_edge.py rc={rc}: {text.strip()[-400:]}")
        out.append({"row": src[0], "witness": w, "dir": kdir})
    manifest["knife"] = out
    return out


# ---------------------------------------------------------------------------------------------------------
# PHASE S: the pre-registered stratified sample
# ---------------------------------------------------------------------------------------------------------


def parse_gap_file(path: str, status: str = "GAP") -> dict[str, list[str]]:
    """{category: [row key, ...]} of the GAP lines (a field tool solved it, ny did not bank it), or of the NOBODY
    lines (no field tool solved it) with status='NOBODY'."""
    if not os.path.isfile(path):
        raise AllonError(f"gap file missing: {path}")
    out: dict[str, list[str]] = collections.defaultdict(list)
    n = 0
    for line in open(path, errors="replace"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[0] == "GAP":
            n += 1
        if len(p) >= 3 and p[0] == status:
            out[p[1]].append(p[2])
    if n == 0:
        raise AllonError(f"{path}: no GAP line (a sample that selects nothing is void)")
    return out


_GAP_CACHE: dict[str, dict] = {}


def _gap(ctx: Ctx, year: int, status: str = "GAP") -> dict[str, list[str]]:
    p = ctx.gap26 if year == 2026 else ctx.gap25
    if (p, status) not in _GAP_CACHE:
        _GAP_CACHE[(p, status)] = parse_gap_file(p, status)
    return _GAP_CACHE[(p, status)]


def nobody_candidates(ctx: Ctx, u: Unit) -> list[int]:
    """The gap file's NOBODY keys re-resolved by index: rows no field tool solved that the bank does not decide
    (beyond the virtual best solver; a conversion there is checked by the witness screen / by nothing for unsat)."""
    keys = set(_gap(ctx, u.year, "NOBODY").get(u.category, []))
    return [r.index for r in u.rows if r.key in keys and u.bank[r.index] not in DECIDED and not u.solved_by_field(r.index)]


def s1_candidates(ctx: Ctx, u: Unit) -> tuple[list[int], dict]:
    """Gap-file keys re-resolved BY INDEX: the rows whose key is listed, that a field tool solved (score > 0), and
    that the CURRENT bank does not decide. A repeated key (safenlp's medical/ruarobot stems) resolves to every
    index that is still a gap. Returns (indices, accounting)."""
    keys = _gap(ctx, u.year).get(u.category, [])
    want = collections.Counter(keys)
    by_key: dict[str, list[int]] = collections.defaultdict(list)
    for r in u.rows:
        by_key[r.key].append(r.index)
    idx, unmatched, now_banked, not_solved = [], 0, 0, 0
    for k in want:
        hits = by_key.get(k, [])
        if not hits:
            unmatched += 1
            continue
        for i in hits:
            if u.bank[i] in DECIDED:
                now_banked += 1
            elif not u.solved_by_field(i):
                not_solved += 1
            else:
                idx.append(i)
    live = {i for i in range(len(u.rows)) if u.solved_by_field(i) and u.bank[i] not in DECIDED}
    acct = {"file_keys": len(want), "resolved": len(set(idx)), "unmatched_keys": unmatched,
            "dropped_now_banked": now_banked, "dropped_not_field_solved": not_solved,
            "live_gaps_not_in_file": len(live - set(idx))}
    return sorted(set(idx)), acct


def parse_margin_audit(path: str) -> list[dict]:
    """The per-row table of scratchpad/vbs/diag10/root_leak_bank_audit.md: {year, cat, instance, margin} with
    margin = the smallest recorded certified/sampled closing margin (an interval `[0,x)` counts as 0), or None."""
    if not os.path.isfile(path):
        raise AllonError(f"margin audit missing: {path}")
    out, on = [], False
    for line in open(path, errors="replace"):
        if line.startswith("## Appendix"):
            on = True
            continue
        if not on:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 9 or not re.match(r"^20[0-9][0-9]$", p[0]):
            continue
        text = p[6] + " " + p[8]
        vals = [float(v) for v in re.findall(r"(?<![\w.])([0-9]+(?:\.[0-9]+)?e[-+]?[0-9]+)", text)]
        if re.search(r"in \[0,", text):
            vals.append(0.0)
        out.append({"year": int(p[0]), "cat": p[1], "instance": p[2], "margin": min(vals) if vals else None})
    if not out:
        raise AllonError(f"{path}: no appendix row parsed")
    return out


AUDIT_CAT = {"cifar100": "cifar100_2024", "tinyimagenet": "tinyimagenet_2024"}


def knife_margin_rows(ctx: Ctx, u: Unit, audit: list[dict]) -> tuple[list[int], list[str]]:
    notes, idx = [], []
    for a in audit:
        if a["year"] != u.year or AUDIT_CAT.get(a["cat"]) != u.category:
            continue
        if a["margin"] is None or a["margin"] > ctx.knife_margin_max:
            continue
        hits = [r.index for r in u.rows if f"_{a['instance']}_" in "_" + L.stem(r.vnnlib_rel) + "_"]
        if len(hits) != 1:
            raise AllonError(f"margin audit {u.uid} {a['instance']}: matches {len(hits)} rows (need exactly 1)")
        if u.bank[hits[0]] not in DECIDED:
            notes.append(f"{u.uid} {a['instance']}: recorded margin {a['margin']:.2e} but not banked decided now")
            continue
        idx.append(hits[0])
    return sorted(set(idx)), notes


def s_sample(ctx: Ctx, u: Unit, a_idx: set, audit: list[dict]) -> tuple[dict[int, list[str]], dict]:
    """{index: strata} for one unit, and its accounting. Deterministic in (seed, year, category, index).

    The literal strata: (s1) gap rows, seeded, cap ctx.gap_cap; (s2) every banked row whose banked wall is
    >= near_frac x budget, every banked sat, every row with a recorded knife-edge margin <= knife_margin_max, and a
    seeded uniform uniform_frac of the remaining banked rows (min uniform_min); (s3) the phase-A rows.
    With ctx.s_budget_frac set, the unit's S seconds (all arms) are capped at s_budget_frac x its phase-B estimate:
    s3 and s2knife always stay; s1/s2near/s2sat/s2uni keep S_BUDGET_MINIMUMS each, then fill round-robin in rank
    order while the budget lasts. The caps are recorded in the accounting (taken/literal per stratum)."""
    strata: dict[int, list[str]] = collections.defaultdict(list)
    s1, acct = s1_candidates(ctx, u)
    r = lambda i: rank(ctx.seed, u.year, u.category, i)  # noqa: E731
    s1 = sorted(s1, key=r)[:ctx.gap_cap]
    banked = [i for i in range(len(u.rows)) if u.bank[i] in DECIDED]
    near = sorted((i for i in banked if u.walls[i] is not None and u.walls[i] >= ctx.near_frac * u.rows[i].budget),
                  key=r)
    sats = sorted((i for i in banked if u.bank[i] == "sat"), key=r)
    knife, knotes = knife_margin_rows(ctx, u, audit)
    risk = set(near) | set(sats) | set(knife)
    rest = sorted((i for i in banked if i not in risk), key=r)
    n_uni = min(len(rest), max(ctx.uniform_min, math.ceil(ctx.uniform_frac * len(rest)))) if rest else 0
    lit = collections.OrderedDict((("s1gap", s1), ("s2near", near), ("s2sat", sats), ("s2uni", rest[:n_uni])))
    take = {k: list(v) for k, v in lit.items()}
    if ctx.s_budget_frac is not None:
        arms = len(s_arms(ctx, u)) * ctx.s_reps
        cost = lambda i: arms * (est_row(u, i) + S_RUN_OVERHEAD_S)  # noqa: E731
        budget = ctx.s_budget_frac * est_b_unit(u)
        fixed = set(knife) | set(a_idx)
        spent = sum(cost(i) for i in fixed)
        chosen = set(fixed)
        take = {k: [] for k in lit}
        for k, n in S_BUDGET_MINIMUMS:
            for i in lit[k][:n]:
                take[k].append(i)
                if i not in chosen:
                    chosen.add(i)
                    spent += cost(i)
        cursors = {k: len(take[k]) for k in lit}
        live = [k for k in lit if cursors[k] < len(lit[k])]
        while live and spent < budget:
            for k in list(live):
                i = lit[k][cursors[k]]
                c = 0.0 if i in chosen else cost(i)
                if spent + c > budget:
                    live.remove(k)
                    continue
                take[k].append(i)
                cursors[k] += 1
                if i not in chosen:
                    chosen.add(i)
                    spent += c
                if cursors[k] >= len(lit[k]):
                    live.remove(k)
        acct["budget_s"] = round(budget, 1)
    for k, idx in take.items():
        for i in idx:
            strata[i].append(k)
    for i in knife:
        strata[i].append("s2knife")
    for i in sorted(a_idx):
        strata[i].append("s3ctl")
    nobody = sorted(nobody_candidates(ctx, u), key=r)[:ctx.s_nobody_cap] if ctx.s_nobody_cap > 0 else []
    for i in nobody:
        strata[i].append("s1nobody")
    acct["s1nobody"] = len(nobody)
    acct.update({"s1": len(take["s1gap"]), "s2near": len(take["s2near"]), "s2sat": len(take["s2sat"]),
                 "s2knife": len(knife), "s2uni": len(take["s2uni"]), "s2rest": len(rest), "s3": len(a_idx),
                 "rows": len(strata), "banked": len(banked), "knife_notes": knotes,
                 "literal": {k: len(v) for k, v in lit.items()},
                 "capped": {k: f"{len(take[k])}/{len(lit[k])}" for k in lit if len(take[k]) < len(lit[k])}})
    return dict(strata), acct


def est_row(u: Unit, i: int) -> float:
    """Expected wall of one run of row i: the banked wall of a decided row, else the full official budget."""
    w = u.walls[i]
    if u.bank[i] in DECIDED and w is not None:
        return min(float(w), float(u.rows[i].budget))
    return float(u.rows[i].budget)


def est_b_unit(u: Unit) -> float:
    s = B_UNIT_OVERHEAD_S
    for i in range(len(u.rows)):
        w = u.walls[i]
        s += (min(float(w), float(u.rows[i].budget)) if w is not None else float(u.rows[i].budget)) + B_ROW_OVERHEAD_S
    return s


def s_arms(ctx: Ctx, u: Unit) -> list[str]:
    return ["allon"] if (u.identical and not ctx.s_identical_control) else ["allon", "ctl"]


def build_s_spec_rows(ctx: Ctx, u: Unit, strata: dict[int, list[str]]) -> list[dict]:
    """Two m0 spec rows per sampled instance (all-on, control), the FIRST arm alternating row by row (no fixed-order
    load bias). An identical-overlay unit runs the all-on arm only (both arms would be the same bytes) unless
    s_identical_control."""
    out = []
    arms = s_arms(ctx, u)
    for k, i in enumerate(sorted(strata, key=lambda i: rank(ctx.seed + "|order", u.year, u.category, i))):
        inst = u.rows[i]
        order = arms if (k % 2 == 0 or len(arms) == 1) else list(reversed(arms))
        for arm in order:
            cfg = ctx.cfg_allon if arm == "allon" else ctx.cfg_ctl
            pre = u.scored_allon if arm == "allon" else u.scored_ctl
            sat_row = u.truth_token(i) == "sat" or u.bank[i] == "sat"
            # a defect signal is an alarm on every row and in both arms; the lane's UNSAT line only on a sat row
            alarms = ([MR_ANY_UNSAT] if sat_row else []) + list(DEFECT_ALARMS)
            out.append(L.spec_row(
                tag=f"S{u.tagbase}_{i:04d}_{'A' if arm == 'allon' else 'C'}", kind="AB", tier="-", year=u.year,
                category=u.category, onnx=inst.onnx, vnnlib=inst.vnnlib, budget=inst.budget, configs_dir=cfg,
                preset=pre or "-", truth=u.truth_token(i), bank=u.bank[i], index=i, reps_c=ctx.s_reps, reps_p=0,
                need="-", alarm=L.ALARM_SEP.join(alarms), flight="-", group=f"S/{u.uid}/{arm}",
                note=f"strata={'+'.join(strata[i])} bank_wall={u.walls[i]} arm={arm} unit={u.uid}"))
    return out


# ---------------------------------------------------------------------------------------------------------
# the plan (pre-registration): written once, never overwritten; every later phase reads it
# ---------------------------------------------------------------------------------------------------------


def write_json(path: str, obj) -> None:
    if os.path.exists(path):
        raise AllonError(f"refusing to overwrite {path}")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)


def build_plan(ctx: Ctx, out: str, run) -> dict:
    """Write the whole pre-registered plan under out/ (refused if out/plan.json exists)."""
    if os.path.exists(os.path.join(out, "plan.json")):
        raise AllonError(f"refusing: {out}/plan.json exists (the plan is pre-registered once; pick a new dir)")
    os.makedirs(out, exist_ok=True)
    units, notes = load_units(ctx)
    if not units:
        raise AllonError("no unit selected")
    audit = parse_margin_audit(ctx.margin_audit)
    fams = order_families(ctx, units)
    # banks snapshot (the S spec's bank column and phase B compare against exactly these bytes)
    snap = os.path.join(out, "banks")
    for u in units:
        if u.bank_path:
            d = os.path.join(snap, str(u.year))
            os.makedirs(d, exist_ok=True)
            with open(u.bank_path, "rb") as src, open(os.path.join(d, u.category + ".csv"), "wb") as dst:
                dst.write(src.read())
    # A
    adir = os.path.join(out, "A")
    os.makedirs(adir)
    aman = build_a_spec(ctx, units, adir)
    negspec = os.path.join(adir, "a_negctl_spec.tsv")
    plan_knife(ctx, aman, adir, negspec, run)
    arows = L.read_spec(negspec)
    kspec = os.path.join(adir, "a_knife_spec.tsv")
    if os.path.isfile(kspec):
        arows += L.read_spec(kspec)
    L.write_spec(os.path.join(adir, "a_spec.tsv"), arows)
    write_json(os.path.join(adir, "a_manifest.json"), aman)
    a_by_unit: dict[str, set] = collections.defaultdict(set)
    for r in aman["rows"]:
        a_by_unit[r["uid"]].add(r["index"])
    # S
    sdir = os.path.join(out, "S")
    os.makedirs(sdir)
    srows, sacct, sample_lines = [], {}, []
    unknown = set(ctx.b_direct) - {f for f, _ in fams}
    if unknown:
        raise AllonError(f"--b-direct names no planned family: {sorted(unknown)}")
    for fam, us in fams:
        for u in us:
            strata, acct = s_sample(ctx, u, a_by_unit.get(u.uid, set()), audit)
            if fam in ctx.b_direct:
                strata, acct = {}, dict(acct, rows=0, b_direct=True)
            sacct[u.uid] = acct
            srows += build_s_spec_rows(ctx, u, strata)
            for i in sorted(strata):
                sample_lines.append("\t".join([u.uid, str(i), u.rows[i].key, "+".join(strata[i]), u.bank[i],
                                               u.truth_token(i), str(u.rows[i].budget), f"{est_row(u, i):.1f}"]))
    if not srows:
        raise AllonError("the S sample is empty")
    L.write_spec(os.path.join(sdir, "s_spec.tsv"), srows)
    with open(os.path.join(sdir, "s_sample.tsv"), "w") as fh:
        fh.write("uid\tindex\tkey\tstrata\tbank\ttruth\tbudget\test_s\n")
        fh.write("\n".join(sample_lines) + "\n")
    # units
    uinfo = []
    for order, (fam, us) in enumerate(fams):
        for u in us:
            uinfo.append({
                "uid": u.uid, "year": u.year, "category": u.category, "family": fam, "family_order": order,
                "version": u.version, "instances_csv": u.instances_csv, "rows": len(u.rows),
                "bank": u.bank_path, "bank_sha256": u.bank_sha256, "align": u.align_mode,
                "banked_decided": sum(1 for v in u.bank if v in DECIDED),
                "scored_allon": u.scored_allon, "scored_ctl": u.scored_ctl, "sweep_allon": u.sweep_allon,
                "sweep_ctl": u.sweep_ctl, "identical": u.identical, "measure_only": u.measure_only,
                "overlay_sha256": (L.sha256_file(os.path.join(ctx.cfg_allon, u.sweep_allon))
                                   if u.sweep_allon else None),
                "est_b_s": est_b_unit(u), "s_accounting": sacct[u.uid], "s_arms": s_arms(ctx, u),
                "b_direct": fam in ctx.b_direct})
    inputs = {p: L.sha256_file(p) for p in (ctx.truth26 if 2026 in ctx.years else None,
                                            ctx.truth25 if 2025 in ctx.years else None,
                                            ctx.gap26 if 2026 in ctx.years else None,
                                            ctx.gap25 if 2025 in ctx.years else None, ctx.margin_audit) if p}
    plan = {"schema": PLAN_SCHEMA, "ctx": {k: (list(v) if isinstance(v, tuple) else v)
                                          for k, v in dataclasses.asdict(ctx).items()},
            "wt_head": L.git_head(ctx.wt), "units": uinfo, "notes": notes, "inputs": inputs,
            "overlay_files": {os.path.relpath(p, ctx.cfg_allon): L.sha256_file(p)
                              for p in sorted(glob.glob(os.path.join(ctx.cfg_allon, "vnncomp2*/*.yaml")))},
            "a_spec_sha256": L.sha256_file(os.path.join(adir, "a_spec.tsv")),
            "s_spec_sha256": L.sha256_file(os.path.join(sdir, "s_spec.tsv"))}
    write_json(os.path.join(out, "plan.json"), plan)
    return plan


def load_plan(out: str) -> dict:
    p = os.path.join(out, "plan.json")
    if not os.path.isfile(p):
        raise AllonError(f"no plan at {p} (run `allon_run.sh --plan` first)")
    plan = json.load(open(p))
    if plan.get("schema") != PLAN_SCHEMA:
        raise AllonError(f"{p}: schema {plan.get('schema')!r}")
    for key, rel in (("a_spec_sha256", "A/a_spec.tsv"), ("s_spec_sha256", "S/s_spec.tsv")):
        if L.sha256_file(os.path.join(out, rel)) != plan[key]:
            raise AllonError(f"{out}/{rel} changed after pre-registration (sha256 differs from plan.json)")
    return plan


def plan_ctx(plan: dict) -> Ctx:
    c = dict(plan["ctx"])
    for k in ("years", "categories", "a_tiers"):
        if c.get(k) is not None:
            c[k] = tuple(c[k])
    return Ctx(**c)


def check_overlay_unchanged(plan: dict) -> list[str]:
    ctx = plan_ctx(plan)
    bad = []
    for rel, sha in plan["overlay_files"].items():
        p = os.path.join(ctx.cfg_allon, rel)
        if not os.path.isfile(p) or L.sha256_file(p) != sha:
            bad.append(rel)
    now = {os.path.relpath(p, ctx.cfg_allon) for p in glob.glob(os.path.join(ctx.cfg_allon, "vnncomp2*/*.yaml"))}
    bad += sorted(f"{r}(new)" for r in now - set(plan["overlay_files"]))
    return bad


def unit_groups(plan: dict) -> "collections.OrderedDict[str, list[dict]]":
    """Shard groups: whole families, merged when two families load the same overlay file (a file is kept or
    reverted as a whole, so its units must be judged on one box)."""
    fams: dict[str, list[dict]] = collections.OrderedDict()
    for u in sorted(plan["units"], key=lambda u: (u["family_order"], -u["year"])):
        fams.setdefault(u["family"], []).append(u)
    parent = {f: f for f in fams}

    def find(f):
        while parent[f] != f:
            f = parent[f]
        return f
    owner: dict[str, str] = {}
    for f, us in fams.items():
        for u in us:
            for rel in (u["sweep_allon"], u["scored_allon"]):
                if rel is None:
                    continue
                if rel in owner and find(owner[rel]) != find(f):
                    parent[find(f)] = find(owner[rel])
                owner.setdefault(rel, f)
    groups: dict[str, list[dict]] = collections.OrderedDict()
    for f, us in fams.items():
        groups.setdefault(find(f), []).extend(us)
    return groups


def unit_shards(plan: dict, n: int) -> dict[str, int]:
    """{uid: shard} for n shards: whole shard groups, balanced on the estimated phase-B seconds."""
    groups = unit_groups(plan)
    by_g = shard_families(list(groups.items()), n, lambda us: sum(u["est_b_s"] for u in us))
    return {u["uid"]: by_g[g] for g, us in groups.items() for u in us}


def shard_units(plan: dict, shard: tuple[int, int]) -> list[dict]:
    k, n = shard
    sh = unit_shards(plan, n)
    return [u for u in sorted(plan["units"], key=lambda u: (u["family_order"], -u["year"])) if sh[u["uid"]] == k]


def spec_unit(row: dict) -> str:
    m = re.search(r"\bunit=(\S+)", row["note"])
    return m.group(1) if m else f"{row['year']}:{row['category']}"


# ---------------------------------------------------------------------------------------------------------
# estimates
# ---------------------------------------------------------------------------------------------------------


def estimates(plan: dict, out: str) -> dict:
    """Box-hour estimates: A = sum budget x runs (attacks off: SAT rows run to the wall); S = per sampled row and arm,
    the banked wall of a decided row else the full budget, + per-run box overhead; B = per unit sum of banked walls
    (budget where none) + per-row overhead. Per unit and per family."""
    arows = L.read_spec(os.path.join(out, "A/a_spec.tsv"))
    a_s = sum((int(r["reps_c"]) + int(r["reps_p"])) * (int(r["budget"]) + A_RUN_OVERHEAD_S) for r in arows)
    srows = L.read_spec(os.path.join(out, "S/s_spec.tsv"))
    per_unit_s: dict[str, float] = collections.Counter()
    sample = {}
    for line in open(os.path.join(out, "S/s_sample.tsv")):
        p = line.rstrip("\n").split("\t")
        if p[0] != "uid":
            sample[(p[0], int(p[1]))] = float(p[7])
    for r in srows:
        uid = spec_unit(r)
        per_unit_s[uid] += int(r["reps_c"]) * (sample[(uid, int(r["index"]))] + S_RUN_OVERHEAD_S)
    per_unit = {}
    for u in plan["units"]:
        per_unit[u["uid"]] = {"family": u["family"], "S_h": per_unit_s.get(u["uid"], 0.0) / 3600,
                              "B_h": u["est_b_s"] / 3600, "s_rows": u["s_accounting"]["rows"],
                              "rows": u["rows"], "identical": u["identical"]}
    fam: dict[str, dict] = collections.OrderedDict()
    for u in sorted(plan["units"], key=lambda u: (u["family_order"], -u["year"])):
        f = fam.setdefault(u["family"], {"S_h": 0.0, "B_h": 0.0, "units": []})
        f["S_h"] += per_unit[u["uid"]]["S_h"]
        f["B_h"] += per_unit[u["uid"]]["B_h"]
        f["units"].append(u["uid"])
    return {"A_h": a_s / 3600, "A_runs": sum(int(r["reps_c"]) + int(r["reps_p"]) for r in arows),
            "S_h": sum(v["S_h"] for v in per_unit.values()), "S_runs": sum(int(r["reps_c"]) for r in srows),
            "B_h": sum(v["B_h"] for v in per_unit.values()), "units": per_unit, "families": fam}


# ---------------------------------------------------------------------------------------------------------
# judging helpers shared by A and S: raw log body, engagement, box accounting
# ---------------------------------------------------------------------------------------------------------

PROV_RE = re.compile(r"^\[ny-box-provenance\] mode=\S+ rc=(-?\d+) elapsed=(\d+) load_before=(\S+) load_after=(\S+) "
                     r"lock_wait=(\d+)", re.M)


def raw_body(raw: str) -> str:
    return "\n".join(ln for ln in raw.splitlines() if not ln.startswith("[ny-box"))


def box_facts(raw: str) -> dict:
    m = PROV_RE.findall(raw)
    if not m:
        return {"elapsed": None, "load_before": None, "lock_wait": None}
    rc, el, lb, _la, lw = m[-1]
    return {"elapsed": int(el), "load_before": lb, "lock_wait": int(lw), "box_rc": int(rc)}


def observed(body: str) -> list[str]:
    return [name for name, ere in OBSERVABLES.items() if re.search(ere, body, re.M)]


def tier_engaged(t: ATier, body: str) -> bool:
    return all(re.search(ere, body, re.M) for ere in t.need)


def lever_receipt(flight: dict | None, name: str):
    for lv in (((flight or {}).get("levers") or {}).get("receipt") or {}).get("levers", []):
        if lv.get("name") == name:
            return lv.get("value"), lv.get("source")
    return None


HARNESS_VOID = ("wrong-or-missing-instance-line", "preset:", "profile-unproven", "ambient-env", "flight-",
                "multiple-Result-tokens", "unknown-verdict", "dark-violation", "need-absent")


def harness_void(reasons: list[str]) -> bool:
    """A VOID the harness caused (the run cannot be tied to what was meant to run), as opposed to a run that ran as
    intended and failed to state a clean verdict (no Result line, results file empty/mismatched): the latter is
    what competition would score, and counts against a banked row."""
    return any(r.startswith(HARNESS_VOID) for r in reasons)


def run_paths(m0_out: str, tag: str, profile: str, rep: int) -> tuple[str, str]:
    return (os.path.join(m0_out, "raw", f"{tag}.{profile}.r{rep}.raw"),
            os.path.join(m0_out, "res", f"{tag}.{profile}.r{rep}.res"))


def read_file(p: str) -> str:
    return open(p, errors="replace").read() if os.path.isfile(p) else ""


def read_flight(res: str) -> dict | None:
    p = res + ".flight.json"
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p))
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------------------------
# PHASE A judge
# ---------------------------------------------------------------------------------------------------------


def a_judge(out: str, m0_out: str, accept_void: set[str] | None = None) -> tuple[int, list[str], dict]:
    """Per (tier, row) engagement over every scheduled run. Returns (rc, report lines, record).
    rc 3: any EMERGENCY (m0 judgement or cross-run); 5: any gating (tier,row) not engaged in every run, a VOID/not-run
    run, or a knife rung VOID -- unless every such tier is in accept_void (then rc 0, recorded); 0: PASS."""
    spec = {r["tag"]: r for r in L.read_spec(os.path.join(out, "A/a_spec.tsv"))}
    aman = json.load(open(os.path.join(out, "A/a_manifest.json")))
    row_tiers = {r["tag"]: r["tiers"] for r in aman["rows"]}
    judged_path = os.path.join(m0_out, "judged.tsv")
    lines = L.read_judged(judged_path)
    runs = L.collapse_runs(lines)
    emerg = [f"{j.tag}/{j.profile}/r{j.rep}: {';'.join(j.reasons)}" for j in lines if j.status == "EMERGENCY"]
    by_tag = collections.defaultdict(list)
    for j in lines:
        by_tag[j.tag].append(j)
    for t, r in spec.items():
        for status, reason in L.judge_profiles(r, by_tag[t]):
            if status == "EMERGENCY":
                emerg.append(f"{t}: {reason}")
    for status, reason in L.judge_instances(spec, lines):
        if status == "EMERGENCY":
            emerg.append(reason)
    rep: list[str] = []
    void_tiers: set[str] = set()
    # a control that never ran (NOT-RUN) or cannot be tied to what was meant to run (a harness VOID: wrong instance or
    # preset, attacks not confirmed off, no Result line) is NOT a dark tier: nothing was controlled. ALLON_ACCEPT_A_VOID
    # may accept a tier that RAN and stayed dark; it can never accept a control that did not happen, or phase A could
    # be "passed" without being run
    unrun: set[str] = set()
    per: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    obs_count: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for tag, r in spec.items():
        m = re.search(r"\bsource_tag=(\S+)", r["note"])
        src = tag if tag in row_tiers else (m.group(1) if m else "")
        if src not in row_tiers:
            raise AllonError(f"A spec row {tag}: no tier record (source {src or '-'})")
        tiers = [A_TIERS[n] for n in row_tiers[src]]
        for prof, k in ((p, i) for i in range(1, max(int(r["reps_c"]), int(r["reps_p"])) + 1)
                        for p in ("competition", "proof")
                        if i <= int(r["reps_c" if p == "competition" else "reps_p"])):
            j = runs.get((tag, prof, k))
            raw, res = run_paths(m0_out, tag, prof, k)
            body = raw_body(read_file(raw))
            flight = read_flight(res)
            for o in observed(body):
                obs_count[r["category"]][o] += 1
            for t in tiers:
                if j is None:
                    state = "NOT-RUN"
                elif j.status == "EMERGENCY":
                    state = "EMERGENCY"
                elif j.status == "VOID":
                    state = "VOID"
                elif tier_engaged(t, body):
                    state = "ENGAGED"
                else:
                    state = "INERT" if t.inert else "DARK"
                lev = ",".join(f"{n}={json.dumps(lever_receipt(flight, n))}" for n, _ in t.levers) or "-"
                per[(t.name, tag)].append(f"{prof[0]}{k}={state}")
                if not t.inert and state != "ENGAGED":
                    void_tiers.add(t.name)
                if state in ("NOT-RUN", "VOID"):
                    unrun.add(f"{tag}/{prof}/r{k}={state}")
                rep.append(f"  A-RUN {tag} {prof} r{k} tier={t.name} state={state} "
                           f"verdict={j.verdict if j else '-'} levers={lev}")
    rep.append("  -- per tier and row (c=competition p=proof)")
    for (tn, tag), states in sorted(per.items()):
        rep.append(f"    {tn:18s} {tag:40s} {' '.join(states)}")
    for cat, cnt in sorted(obs_count.items()):
        rep.append(f"  OBSERVED {cat}: " + " ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    for e in emerg:
        rep.append(f"  EMERGENCY {e}")
    # a tier the overlay arms for a planned category with NO phase-A row is UNCONTROLLED: never silently a pass
    plan = json.load(open(os.path.join(out, "plan.json")))
    planned = {u["category"] for u in plan["units"]}
    have = {n for tl in row_tiers.values() for n in tl}
    auto_ok = []
    for t in A_TIERS.values():
        cats = [c for c in t.categories if c in planned]
        if t.inert or not cats or t.name in have:
            continue
        if plan["ctx"].get("a_tiers") is not None and t.name not in plan["ctx"]["a_tiers"]:
            continue
        if all(c in MEASURE_ONLY for c in cats):
            auto_ok.append(t.name)
            rep.append(f"  UNCONTROLLED {t.name}: no truly-SAT row exists for {','.join(cats)}; auto-accepted because "
                       "every such category is measure-only (its unsat is never banked)")
        else:
            void_tiers.add(t.name)
            rep.append(f"  UNCONTROLLED {t.name}: armed for {','.join(cats)} but no phase-A row controls it")
    accepted = sorted(void_tiers & (accept_void or set()))
    blocking = sorted(void_tiers - (accept_void or set()))
    for x in sorted(unrun)[:40]:
        rep.append(f"  UNRUN {x} (never acceptable: re-run it; ALLON_ACCEPT_A_VOID covers only tiers that ran dark)")
    if emerg:
        rc, status = 3, "EMERGENCY"
    elif blocking or unrun or not runs:
        rc, status = 5, "VOID"
    else:
        rc, status = 0, "PASS" if not accepted else "PASS-WITH-ACCEPTED-VOID"
    box = box_hours(m0_out)
    # the pin phase A controlled (allon_run.sh writes OUT/PIN_LOCK before the A run): S and B must run the same bytes
    lock = os.path.join(out, "PIN_LOCK")
    pin_lock = open(lock).read().split() if os.path.isfile(lock) else []
    record = {"schema": "allon-A-record/v1", "status": status, "rc": rc, "emergencies": emerg,
              "void_tiers": sorted(void_tiers), "accepted_void": accepted, "blocking_void": blocking,
              "unrun": sorted(unrun), "uncontrolled_auto_accepted": auto_ok,
              "judged_sha256": L.sha256_file(judged_path) if os.path.isfile(judged_path) else None,
              "runs": len(runs), "box": box, "pin_sha256": pin_lock[0] if pin_lock else None}
    rep.append(f"  A BOX-HOURS measured={box['box_hours']} h over {box['runs_with_provenance']} runs "
               f"(lock waits {box['lock_wait_hours']} h)")
    rep.append(f"ALLON_A_RESULT status={status} rc={rc} emergencies={len(emerg)} void_tiers={','.join(sorted(void_tiers)) or '-'}"
               f" accepted={','.join(accepted) or '-'} unrun={len(unrun)} runs={len(runs)} box_h={box['box_hours']}")
    return rc, rep, record


def box_hours(m0_out: str) -> dict:
    el = lw = n = 0
    for p in glob.glob(os.path.join(m0_out, "raw", "*.raw")):
        f = box_facts(read_file(p))
        if f["elapsed"] is not None:
            el += f["elapsed"]
            lw += f["lock_wait"] or 0
            n += 1
    return {"runs_with_provenance": n, "box_hours": round(el / 3600, 3), "lock_wait_hours": round(lw / 3600, 3)}


# ---------------------------------------------------------------------------------------------------------
# PHASE S decision
# ---------------------------------------------------------------------------------------------------------


@dataclass
class ArmTally:
    converted: list = field(default_factory=list)   # (index, verdict)
    lost: list = field(default_factory=list)        # (index, bank, verdict)
    held: int = 0
    void: list = field(default_factory=list)        # (index, reasons)
    notrun: list = field(default_factory=list)
    emergency: list = field(default_factory=list)
    walls: list = field(default_factory=list)
    outcome: dict = field(default_factory=dict)      # index -> held/lost/converted/open/void (attribution)


def scored_outcome(j: L.Judgement, bank: str, truth: str) -> str:
    """held / lost / converted / flip / open / wrong for one judged run (its scored verdict = the results file)."""
    v = j.verdict
    if bank in DECIDED:
        if v in DECIDED and v != bank:
            return "flip"
        return "held" if v == bank else "lost"
    if v in DECIDED:
        if truth in DECIDED and v != truth:
            return "wrong"
        return "converted"
    return "open"


def s_decide(plan: dict, out: str, m0_out: str, units: list[dict]) -> tuple[int, list[str], list[dict]]:
    """Per unit decision from the S judgements. Returns (rc, report lines, decision records)."""
    spec = L.read_spec(os.path.join(out, "S/s_spec.tsv"))
    lines = L.read_judged(os.path.join(m0_out, "judged.tsv"))
    runs = L.collapse_runs(lines)
    want = {u["uid"] for u in units}
    rows_by_tag = {r["tag"]: r for r in spec}
    cross = []
    by_tag = collections.defaultdict(list)
    for j in lines:
        by_tag[j.tag].append(j)
    for t, r in rows_by_tag.items():
        if spec_unit(r) in want:
            for status, reason in L.judge_profiles(r, by_tag[t]):
                if status == "EMERGENCY":
                    cross.append((spec_unit(r), f"{t}: {reason}"))
    for status, reason in L.judge_instances(rows_by_tag, lines):
        if status == "EMERGENCY":
            m = re.search(r"INSTANCE-FLIP\(([^:]+):", reason)
            hits = sorted({spec_unit(r) for r in spec if m and os.path.basename(r["vnnlib"]) == m.group(1)})
            for h in hits or ["?"]:
                cross.append((h, reason))
    tallies: dict[str, dict[str, ArmTally]] = {u: {"allon": ArmTally(), "ctl": ArmTally()} for u in want}
    # engagement (CLAUDE.md: demand the lever's own observable): per unit and arm, the runs on which each gating
    # tier's own line and each recorded observable appeared
    eng: dict[str, dict[str, collections.Counter]] = {u: {"allon": collections.Counter(), "ctl": collections.Counter(),
                                                          "allon_obs": collections.Counter(), "runs_allon": collections.Counter()}
                                                      for u in want}
    loads = []
    for r in spec:
        uid = spec_unit(r)
        if uid not in want:
            continue
        arm = "allon" if r["tag"].endswith("_A") else "ctl"
        t = tallies[uid][arm]
        i = int(r["index"])
        for k in range(1, int(r["reps_c"]) + 1):
            j = runs.get((r["tag"], "competition", k))
            raw, _ = run_paths(m0_out, r["tag"], "competition", k)
            text = read_file(raw)
            bf = box_facts(text)
            if j is not None:
                body = raw_body(text)
                for tier in tiers_for(r["category"]):
                    if not tier.inert and tier_engaged(tier, body):
                        eng[uid][arm][tier.name] += 1
                if arm == "allon":
                    eng[uid]["runs_allon"]["n"] += 1
                    for o in observed(body):
                        eng[uid]["allon_obs"][o] += 1
            if bf["load_before"] is not None:
                loads.append(f"    {r['tag']} r{k} load_before={bf['load_before']} elapsed={bf['elapsed']}")
            if j is None:
                t.notrun.append(i)
                continue
            if j.status == "EMERGENCY":
                t.emergency.append((i, ";".join(j.reasons)))
            if j.status == "VOID" and harness_void(j.reasons):
                t.void.append((i, ";".join(x for x in j.reasons if not x.startswith("note:"))[:160]))
                t.outcome[i] = "void"
                continue
            o = scored_outcome(j, r["bank"], r["truth"])
            t.outcome[i] = o
            if o in ("flip", "wrong"):
                t.emergency.append((i, f"{o}: bank={r['bank']} truth={r['truth']} now={j.verdict}"))
            elif o == "converted":
                t.converted.append((i, j.verdict))
            elif o == "lost":
                t.lost.append((i, r["bank"], j.verdict))
            elif o == "held":
                t.held += 1
            try:
                t.walls.append(float(j.wall))
            except ValueError:
                pass
    uinfo = {u["uid"]: u for u in plan["units"]}
    recs, rep = [], []
    worst = 0
    for u in units:
        uid = u["uid"]
        a, c = tallies[uid]["allon"], tallies[uid]["ctl"]
        emerg = [f"{i}: {why}" for i, why in a.emergency + c.emergency] + [why for x, why in cross if x in (uid, "?")]
        ctl_conv = {i for i, _ in c.converted}
        if emerg:
            dec = EMERGENCY
        elif u.get("b_direct"):
            dec = B_DIRECT
        elif a.void or a.notrun:
            dec = INCOMPLETE
        elif a.lost:
            dec = REVERT
        elif a.converted:
            dec = MO_KEEP if u["measure_only"] else KEEP
        else:
            dec = MO_NEUTRAL if u["measure_only"] else NEUTRAL
        rc = 3 if dec == EMERGENCY else 5 if dec == INCOMPLETE else 0
        worst = max(worst, rc)
        gating = [t.name for t in tiers_for(u["category"]) if not t.inert]
        e = eng[uid]
        dark = [t for t in gating if e["allon"][t] == 0] if e["runs_allon"]["n"] else list(gating)
        contaminated = [t for t in gating if e["ctl"][t] > 0]

        def attrib(i: int) -> str:
            if u["identical"]:
                return "pin(identical-overlay)"
            o = c.outcome.get(i)
            return "pin" if o == "lost" else "overlay" if o == "held" else f"?(ctl={o or 'not-run'})"
        lost_rows = [f"{i}:{b}->{v}({attrib(i)})" for i, b, v in a.lost]
        conv_rows = [f"{i}:{v}({'also-ctl' if i in ctl_conv else 'allon-only'})" for i, v in a.converted]
        rec = {"uid": uid, "decision": dec, "identical": u["identical"], "measure_only": u["measure_only"],
               "converted": len(a.converted), "converted_unsat_unbankable": sum(
                   1 for _, v in a.converted if v == "unsat") if u["measure_only"] else 0,
               "lost": len(a.lost), "held": a.held, "void": len(a.void), "notrun": len(a.notrun),
               "emergency": emerg, "ctl_converted": len(c.converted), "ctl_lost": len(c.lost),
               "lost_rows": lost_rows, "converted_rows": conv_rows, "void_rows": a.void[:20],
               "overlay_preset": u["sweep_allon"], "scored_preset": u["scored_allon"],
               "engaged_allon": dict(e["allon"]), "observed_allon": dict(e["allon_obs"]),
               "allon_runs": e["runs_allon"]["n"], "dark_tiers": dark, "ctl_contaminated": contaminated}
        recs.append(rec)
        rep.append(f"ALLON_S_DECISION {uid} decision={dec} converted={len(a.converted)} lost={len(a.lost)} "
                   f"held={a.held} void={len(a.void)} notrun={len(a.notrun)} emergency={len(emerg)} "
                   f"ctl_converted={len(c.converted)} ctl_lost={len(c.lost)} identical={int(u['identical'])}"
                   + (f" lost_rows={','.join(lost_rows)}" if lost_rows else "")
                   + (f" converted_rows={','.join(conv_rows)}" if conv_rows else ""))
        if gating or e["allon_obs"]:
            rep.append(f"  ENGAGEMENT {uid} allon_runs={e['runs_allon']['n']} "
                       + " ".join(f"{t}={e['allon'][t]}" for t in gating)
                       + (f" DARK={','.join(dark)} (a null is VOID for these tiers, never negative)" if dark else "")
                       + (f" CONTROL-SHOWS={','.join(contaminated)} (the control ran an overlay tier: read its preset)"
                          if contaminated else "")
                       + " | observed: " + " ".join(f"{k}={v}" for k, v in sorted(e["allon_obs"].items())))
        for x in emerg:
            rep.append(f"  EMERGENCY {uid} {x}")
        for i, why in a.void[:5]:
            rep.append(f"  VOID {uid} {i}: {why}")
    # an overlay file is kept or reverted as a whole: a KEEP-CANDIDATE unit whose overlay file another unit
    # REVERTs (or could not decide) is not eligible for phase B
    by_file = collections.defaultdict(list)
    for rec in recs:
        by_file[rec["overlay_preset"]].append(rec)
    for rec in recs:
        sib = [s for s in by_file[rec["overlay_preset"]] if s is not rec and rec["overlay_preset"]]
        blockers = [f"{s['uid']}={s['decision']}" for s in sib if s["decision"] in (REVERT, INCOMPLETE, EMERGENCY)]
        if not rec["identical"] and rec["scored_preset"] != rec["overlay_preset"]:
            blockers.append(f"resolver-split(S measured {rec['scored_preset']}, the sweep would load "
                            f"{rec['overlay_preset']})")
        rec["b_eligible"] = rec["decision"] in B_ELIGIBLE and not blockers
        rec["b_blockers"] = blockers
        if rec["decision"] in B_ELIGIBLE and blockers:
            rep.append(f"  NOTE {rec['uid']}: {rec['decision']} but its overlay file {rec['overlay_preset']} is shared "
                       f"with {', '.join(blockers)}: not eligible for phase B")
    rep.append("  -- load_before per run (interleaved arm order)")
    rep += loads
    return worst, rep, recs


# ---------------------------------------------------------------------------------------------------------
# PHASE B judge (a sweep CSV vs the bank snapshot and field truth, index-aligned)
# ---------------------------------------------------------------------------------------------------------


def read_sweep_csv(path: str) -> list[tuple[str, str, str, str, float | None]]:
    out = []
    with open(path, newline="") as fh:
        for r in csv.reader(fh):
            if len(r) < 5:
                continue
            v = (r[4] if len(r) >= 6 else r[3]).strip().lower()
            try:
                w = float(r[5]) if len(r) >= 6 else None
            except ValueError:
                w = None
            out.append((r[0].strip(), r[1].strip(), r[2].strip(), v, w))
    return out


def read_metadata(path: str) -> dict[int, dict]:
    """instance_index -> the LAST metadata row for it (a resumed sweep appends)."""
    out: dict[int, dict] = {}
    if not os.path.isfile(path):
        return out
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and isinstance(r.get("instance_index"), int):
            out[r["instance_index"]] = r
    return out


@dataclass
class BResult:
    uid: str
    status: str                 # COMPLETE | INCOMPLETE | EMERGENCY
    rows: int = 0
    expected: int = 0
    held: int = 0
    converted: list = field(default_factory=list)
    lost: list = field(default_factory=list)
    flips: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)
    unbankable: list = field(default_factory=list)
    void_rows: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    tools: dict = field(default_factory=dict)

    @property
    def emergency(self) -> bool:
        return bool(self.flips or self.contradictions or self.tools.get("emergency"))

    def line(self) -> str:
        return (f"ALLON_B_UNIT {self.uid} status={self.status} rows={self.rows}/{self.expected} held={self.held} "
                f"converted={len(self.converted)} lost={len(self.lost)} flips={len(self.flips)} "
                f"contradictions={len(self.contradictions)} void_rows={len(self.void_rows)} "
                f"unbankable={len(self.unbankable)} problems={len(self.problems)}")


def b_judge(plan: dict, out: str, uid: str, csv_path: str, pin_sha: str | None, pin_commit: str | None,
            check_manifest: bool = True) -> BResult:
    """Soundness checks run on EVERY row present (a partial sweep can still flip). COMPLETE needs every instance row,
    index-aligned, each with its own flight record naming the overlay file (preset_base) and a clean ambient env."""
    ctx = plan_ctx(plan)
    u = next((x for x in plan["units"] if x["uid"] == uid), None)
    if u is None:
        raise AllonError(f"{uid} not in the plan")
    year, cat = u["year"], u["category"]
    inst = L.load_instances(u["instances_csv"], check_files=False)
    res = BResult(uid, "INCOMPLETE", expected=len(inst))
    snap = os.path.join(out, "banks", str(year), cat + ".csv")
    if u["bank"] and os.path.isfile(snap):
        bank, _, _ = align_bank_full(inst, _bank_rows(snap), f"B {uid} bank")
    else:
        bank = ["-"] * len(inst)
    truth = L.align_truth(inst, L.parse_results_txt(ctx.truth26 if year == 2026 else ctx.truth25, year).get(cat),
                          f"B {uid} truth")
    if not os.path.isfile(csv_path):
        res.problems.append("no-csv")
        return res
    rows = read_sweep_csv(csv_path)
    res.rows = len(rows)
    meta = read_metadata(csv_path[:-4] + ".metadata.jsonl" if csv_path.endswith(".csv") else csv_path + ".metadata.jsonl")
    want_pre = os.path.realpath(os.path.join(ctx.cfg_allon, u["sweep_allon"])) if u["sweep_allon"] else None
    allon_real = os.path.realpath(ctx.cfg_allon)
    for i, (c, o, v, verdict, _w) in enumerate(rows):
        if i >= len(inst):
            res.problems.append(f"extra-row:{i}")
            continue
        r = inst[i]
        if c != cat or L._row_identity(o, v) != L._row_identity(r.onnx_rel, r.vnnlib_rel):
            res.problems.append(f"misaligned:{i}:{o},{v}")
            continue
        b, t = bank[i], (truth[i].token() if truth[i] else "-")
        if verdict in DECIDED and t in DECIDED and verdict != t:
            res.contradictions.append(f"{i}:{r.key}:ny={verdict}:truth={t}")
        if b in DECIDED and verdict in DECIDED and verdict != b:
            res.flips.append(f"{i}:{r.key}:bank={b}->now={verdict}")
        elif b in DECIDED and verdict == b:
            res.held += 1
        elif b in DECIDED:
            res.lost.append(f"{i}:{r.key}:bank={b}->now={verdict}")
        elif verdict in DECIDED and not (t in DECIDED and verdict != t):
            if u["measure_only"] and verdict == "unsat":
                res.unbankable.append(f"{i}:{r.key}:unsat(measure-only)")
            else:
                res.converted.append(f"{i}:{r.key}:{b}->now={verdict}")
        # per-row provenance (the sweep's own observables: CLAUDE.md, `ny benchmarks run` discards stderr)
        m = meta.get(i)
        if m is None:
            res.void_rows.append(f"{i}:no-metadata")
            continue
        fl = m.get("flight") or {}
        pb = [e for e in fl.get("events") or [] if isinstance(e, dict) and e.get("method") == "preset_base"]
        if want_pre:
            if not pb or os.path.realpath(str(pb[-1].get("reason"))) != want_pre:
                res.void_rows.append(f"{i}:preset_base={pb[-1].get('reason') if pb else 'none'}")
        elif not pb or "AUTO DEFAULTS" not in str(pb[-1].get("reason")) \
                or allon_real not in str(pb[-1].get("reason")):
            res.void_rows.append(f"{i}:preset_base={pb[-1].get('reason') if pb else 'none'}(want AUTO under overlay)")
        amb = fl.get("ambient_env")
        if isinstance(amb, dict):
            extra = sorted(k for k, val in amb.items() if (k, val) != ("OMP_NUM_THREADS", "1"))
            if extra:
                res.void_rows.append(f"{i}:ambient-env:{','.join(extra)[:80]}")
        if m.get("from_cache"):
            res.void_rows.append(f"{i}:from_cache")
        mv = str(m.get("verdict", "")).lower()
        if mv and mv != verdict:
            res.problems.append(f"csv-metadata-verdict-mismatch:{i}:{verdict}!={mv}")
    if check_manifest:
        mp = csv_path[:-4] + ".manifest.json" if csv_path.endswith(".csv") else csv_path + ".manifest.json"
        try:
            man = json.load(open(mp))
        except (OSError, ValueError):
            man = None
            res.problems.append("manifest-missing")
        if man:
            exe = (man.get("executable") or {}).get("sha256")
            if pin_sha and exe != pin_sha:
                res.problems.append(f"manifest-exe:{str(exe)[:16]}!={pin_sha[:16]}")
            if pin_commit and f"commit={pin_commit}" not in str(man.get("build_id", "")):
                res.problems.append("manifest-build-id-commit")
            cp = str((man.get("configs") or {}).get("canonical_path", ""))
            if not cp.startswith(os.path.realpath(ctx.cfg_allon)):
                res.problems.append(f"manifest-configs:{cp}")
            if man.get("year") != year:
                res.problems.append(f"manifest-year:{man.get('year')}")
            if u["version"] and man.get("vnnlib_version") != u["version"]:
                res.problems.append(f"manifest-vnnlib-version:{man.get('vnnlib_version')}!={u['version']}")
            if man.get("timeout_cap") is not None:
                res.problems.append(f"manifest-timeout-cap:{man.get('timeout_cap')}(not official budgets)")
    if res.emergency:
        res.status = "EMERGENCY"
    elif res.rows == len(inst) and not res.problems and not res.void_rows:
        res.status = "COMPLETE"
    return res


def parse_tool_outputs(compare: tuple[int, str] | None, truth_tool: tuple[int, str] | None,
                       moat25: tuple[int, str] | None, category: str) -> dict:
    """The campaign's own checkers, parsed fail-closed: a tool that ran but printed no summary line checked
    nothing (the empty-grep trap), which is a problem, not a pass."""
    d: dict = {"emergency": [], "problems": []}
    if compare is not None:
        rc, text = compare
        m = re.search(r"BANKCMP \S+ rows=(\d+) held=(\d+) converted=(\d+) lost=(\d+) emergency=(\d+)", text)
        d["compare_bank"] = m.group(0) if m else None
        if rc == 3 or (m and int(m.group(5)) > 0):
            d["emergency"].append(f"compare_bank: {m.group(0) if m else 'rc=3'}")
        elif rc != 0 or not m:
            d["problems"].append(f"compare_bank rc={rc} line={'yes' if m else 'none'}")
    if truth_tool is not None:
        rc, text = truth_tool
        m = re.search(r"TRUTHCMP \S+ rows=\d+ agree=\d+ open=\d+ undecided_truth=\d+ contradictions=(\d+)", text)
        d["moat_truth"] = m.group(0) if m else None
        if rc == 3 or (m and int(m.group(1)) > 0):
            d["emergency"].append(f"moat_truth: {m.group(0) if m else 'rc=3'}")
        elif rc != 0 or not m:
            d["problems"].append(f"moat_truth rc={rc} line={'yes' if m else 'none'}")
    if moat25 is not None:
        rc, text = moat25
        m = re.search(rf"^{re.escape(category)}\s+rows=\d+ matched=\d+ flags=(\d+)", text, re.M)
        d["moat_2025"] = m.group(0) if m else None
        if m and int(m.group(1)) > 0:
            d["emergency"].append(f"moat_2025: {m.group(0)}")
        elif rc != 0 or not m:
            d["problems"].append(f"moat_2025 rc={rc} line={'yes' if m else 'none (category not checked)'}")
    return d


# ---------------------------------------------------------------------------------------------------------
# PHASE C: the decision report
# ---------------------------------------------------------------------------------------------------------


def c_report(plan: dict, out: str, units: list[dict]) -> tuple[int, list[str], list[dict]]:
    """keep-all-on iff the unit's B record is COMPLETE with lost == 0 and no emergency (converted >= 0); REVERT with the
    lost rows for bisection otherwise; units B did not sweep carry their S decision. An overlay file is kept only
    when every unit that loads it keeps."""
    sdec = {}
    for p in glob.glob(os.path.join(out, "S", "run_*", "decisions.json")):
        for r in json.load(open(p)):
            sdec[r["uid"]] = r
    recs, rep = [], []
    emerg = lost = conv = swept = 0
    worst = 0
    for u in units:
        uid = u["uid"]
        bp = os.path.join(out, "B", "records", uid.replace(":", "_") + ".json")
        b = json.load(open(bp)) if os.path.isfile(bp) else None
        s = sdec.get(uid)
        if b is not None:
            swept += 1
            emerg += int(b["status"] == "EMERGENCY")
            lost += len(b["lost"])
            conv += len(b["converted"])
            if b["status"] == "EMERGENCY":
                dec = "EMERGENCY"
            elif u["measure_only"]:
                dec = "MEASURE-ONLY"
            elif b["status"] != "COMPLETE":
                dec = "INCOMPLETE"
            elif b["lost"]:
                dec = "REVERT"
            else:
                dec = "KEEP-ALL-ON"
            why = f"B: converted={len(b['converted'])} lost={len(b['lost'])} held={b['held']}"
            bis = b["lost"]
        else:
            dec = f"NOT-SWEPT({s['decision'] if s else 'no S decision'})"
            why = (f"S: converted={s['converted']} lost={s['lost']}" if s else "S: -")
            bis = s["lost_rows"] if s else []
        if u["identical"] and dec == "KEEP-ALL-ON":
            dec = "KEEP-ALL-ON(identical-overlay: a pin sweep)"
        rc = 3 if dec == "EMERGENCY" else 5 if dec == "INCOMPLETE" else 0
        worst = max(worst, rc)
        recs.append({"uid": uid, "decision": dec, "why": why, "bisect": bis, "overlay_preset": u["sweep_allon"],
                     "identical": u["identical"]})
    by_file = collections.defaultdict(list)
    for r in recs:
        by_file[r["overlay_preset"]].append(r)
    for r in recs:
        sib = [x for x in by_file[r["overlay_preset"]] if x is not r and r["overlay_preset"]]
        r["file_keep"] = r["decision"].startswith("KEEP-ALL-ON") and all(x["decision"].startswith("KEEP-ALL-ON")
                                                                        for x in sib)
        rep.append(f"ALLON_C {r['uid']} decision={r['decision']} file={r['overlay_preset']} "
                   f"file_keep={int(r['file_keep'])} {r['why']}" + (f" bisect={';'.join(r['bisect'][:40])}"
                                                                    if r["bisect"] else ""))
    return worst, rep + [f"ALLON_C_TOTAL swept={swept} converted={conv} lost={lost} emergency={emerg}"], recs
