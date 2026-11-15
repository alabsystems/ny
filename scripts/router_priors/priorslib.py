#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Offline priors v1 (#router-r7, ROUTER_PROGRAM section 5.5): the COST-MODEL prior pipeline.

Tooling only: pure python3 standard library. No cargo, no ny binary, no box.

WHAT IS LEARNED (and what is never learned)
    A cost model per PHASE: log2(seconds) = a_cell + b * log2(work), where `work` is a structural count the
    router has at its start (`params` = summed initializer element counts, the plan resolver's scan) and
    `cell` is a COARSE key of the operator signature (`conv` x `nonpwl`, two bits). One exponent `b` is shared
    by every cell of a phase, so the continuous size enters ONLY through one global monotone map, never as a
    lookup key. A cost model predicts TIME; it never predicts a verdict, a category, a route or success.

    The anti-fingerprint lesson (generalization audit 2026-09-28: `n_params` alone isolates 24/24 scored
    categories) is why the KEY is two operator bits and nothing else, and why every gate below exists.

GATES (each recorded in the table and re-checked by `crates/ny-core/tests/router_priors.rs`)
    G1 key vocabulary: cells are `*` or `conv=<0|1>;nonpwl=<0|1>`; work is `params`. Never n_params as a key,
       never a budget, a file size bucket, an input dim or an op-signature id.
    G2 every shipped cell spans >= 2 FAMILY-level leakage groups (GENERALIZATION_DESIGN 9.1 + K4.3 merges)
       and its dominant family carries <= 0.6 of the cell's family-balanced weight; a failing cell collapses
       to the root `*` (which must itself span >= 2 families, or the phase does not ship).
    G3 identifiability: a per-cell-majority classifier (the Bayes classifier over the discrete key: no tree
       on these keys can beat it) trained on one model-hash fold and scored on the other, family-balanced,
       may not beat the global-majority baseline by more than 0.20.
    G4 isolation: no cell contains the units of exactly one family (implied by G2, recorded separately).
    G5 shared exponent: 0 <= b <= 2 for every phase, and every cell of a phase carries the SAME b.
    G6 leave-one-family-out (LOGO) overfit gap: median over held-out families of (held-out error - full-fit
       error), in log2 units, <= 0.5; the held-out error against a CONSTANT null model is reported with an
       exact sign test (wins / losses / p) and a phase whose LOGO loses to the null more often than it wins
       does not ship.
    G7 ship stability: the phase must pass G2-G6 again in EVERY leave-one-family-out refit (one level: the
       refit's own gates, not its own G7). A model whose pass depends on one family's rows is that family's
       model, whatever its other numbers say; it does not ship.

TRAINING SET
    Every `<results>.flight.json` receipt under the scanned roots whose identity is recoverable from a raw
    log's `Running VNN-COMP instance:` line, measured on the declared host class (GB10: 20 logical cores,
    ~122 GiB), with 1-min load <= LOAD_MAX at the receipt's start, joined BY MODEL to the static instance
    features table. Phase observations are right-censored where a cap could have ended them (see
    `phase_observations`); censored observations are DROPPED, never imputed. Units are (family, model):
    every row of one model is collapsed to its median first (rows are clustered), and every family carries
    total weight 1 (a 300-row family cannot outvote a 3-row one).

NAME-BLINDNESS
    Family labels are needed to hold families out; they never reach ny-core. The compiled table carries no
    family, no category and no path, and the committed fixture (`crates/ny-core/tests/data/
    priors_v1_rows.tsv`) carries OPAQUE family ids (F01..), assigned in sorted order of the family label.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass

SCHEMA = "ny-router-priors/v1"
VERSION = "cost-v1"
HOST_CLASS = "gb10"
LOAD_MAX = 6.0

# Gate constants (mirrored by `ny_core::router::priors`; the Rust test asserts equality).
MIN_FAMILIES_PER_CELL = 2
PURITY_MAX = 0.6
FINGERPRINT_EXCESS_MAX = 0.20
SLOPE_MIN = 0.0
SLOPE_MAX = 2.0
OVERFIT_GAP_MAX_LOG2 = 0.5
SPREAD_K = 1.0
# A cell's spread (weighted median |residual|, log2) is floored here: with two families a weighted median can
# land exactly on one unit and report 0, which would make `upper == median` (overconfident).
SPREAD_FLOOR_LOG2 = 0.5
MIN_UNITS_PER_PHASE = 4

ROOT_CELL = "*"
WORK_VARS = ("params",)

# The phases v1 fits. Each is a (start, end) pair of flight events with its own censoring rule.
PHASES = ("attack.upfront", "verifier.setup", "root.bound")
PHASE_UNIT = {"attack.upfront": "secs", "verifier.setup": "secs", "root.bound": "secs_per_clause"}

# Which capability's FIRST RUNG (setup + one progress window, ROUTER_PROGRAM 3.2 C6) each phase prices.
# A capability is priced only when EVERY phase of its binding ships; otherwise it stays unpriced (today's cap).
BINDINGS = (
    ("attack.upfront_apgd", ("attack.upfront",)),
    ("relu_split.graph", ("verifier.setup", "root.bound")),
    ("input_split", ("verifier.setup", "root.bound")),
    ("lowdim", ("verifier.setup", "root.bound")),
    ("mip.ay", ("verifier.setup", "root.bound")),
)

# Family-level leakage groups (GENERALIZATION_DESIGN 9.1: shared model/spec md5; plus the K4.3 merges:
# relusplitter pair, the resnet certified-training convnets, the yolo pair, the MNIST/CIFAR MLPs).
FAMILY_MERGES = {
    "acasxu_2023": "acasxu",
    "isomorphic_acasxu_2026": "acasxu",
    "monotonic_acasxu_2026": "acasxu",
    "cgan2026": "cgan",
    "cgan_2023": "cgan",
    "soundnessbench": "soundnessbench",
    "soundnessbench_2026": "soundnessbench",
    "cifar100_2024": "resnet_ct",
    "tinyimagenet_2024": "resnet_ct",
    "challenging_certified_training_2026": "resnet_ct",
    "yolo_2023": "yolo",
    "cctsdb_yolo_2023": "yolo",
    "cora_2024": "image_mlp",
    "relusplitter": "image_mlp",
    "relusplitter_2026": "image_mlp",
}

INSTANCE_RE = re.compile(
    r"Running VNN-COMP instance: category='(?P<cat>[^']+)' onnx='(?P<onnx>[^']+)' "
    r"vnnlib='(?P<vnnlib>[^']+)' results='(?P<res>[^']+)'")
BENCH_RE = re.compile(r"/benchmarks/vnncomp(?P<year>20\d\d)/benchmarks/(?P<cat>[^/]+)/(?:(?:1\.0|2\.0)/)?(?P<rel>.+)$")


class PriorsError(Exception):
    """A refusal: the pipeline never emits an artefact it cannot vouch for."""


def family_of(category: str) -> str:
    return FAMILY_MERGES.get(category, category)


def opaque_family_ids(families) -> dict:
    """F01.. in sorted order of the family label (deterministic; the mapping stays in tooling)."""
    return {f: f"F{i + 1:02d}" for i, f in enumerate(sorted(set(families)))}


# ---- the key (mirrors ny_core::router::priors::CostKey::from_facts) --------------------------------------

NONPWL_FLAGS = ("has_sigmoid", "has_tanh", "has_softmax", "has_layernorm")


def key_bits(feat: dict) -> tuple[int, int]:
    """(conv, nonpwl) from a static instance-features row.

    conv   = the model has a Conv or ConvTranspose node            (OpKind::Conv | OpKind::ConvTranspose)
    nonpwl = Sigmoid / Tanh / Softmax / a var x var product / a normalisation layer
                                                                  (OpKind::Sigmoid | Tanh | Softmax | Bilinear | Norm)
    """
    conv = int(_int(feat.get("has_conv")) > 0 or _int(feat.get("has_convT")) > 0)
    nonpwl = int(any(_int(feat.get(k)) > 0 for k in NONPWL_FLAGS) or _int(feat.get("n_bilinear")) > 0)
    return conv, nonpwl


def cell_token(conv: int, nonpwl: int) -> str:
    return f"conv={conv};nonpwl={nonpwl}"


def _int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


# ---- collection ------------------------------------------------------------------------------------------

def scan_identities(roots, exts=(".raw", ".log", ".out")) -> dict:
    """results path -> (category, onnx path, vnnlib path), from every raw log under `roots`.

    A results path claimed by two different identities is DROPPED (ambiguous provenance)."""
    seen: dict = {}
    clash: set = set()
    for root in roots:
        if not os.path.isdir(root):
            raise PriorsError(f"scan root does not resolve: {root}")
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if not name.endswith(exts):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, errors="replace") as fh:
                        for line in fh:
                            if "Running VNN-COMP instance:" not in line:
                                continue
                            m = INSTANCE_RE.search(line)
                            if not m:
                                continue
                            ident = (m["cat"], m["onnx"], m["vnnlib"])
                            res = m["res"]
                            if res in seen and seen[res] != ident:
                                clash.add(res)
                            seen.setdefault(res, ident)
                except OSError:
                    continue
    for res in clash:
        seen.pop(res, None)
    return seen


def load_features(path: str) -> dict:
    """(year, category, onnx_rel) -> model-level feature row (the first instance row of that model)."""
    if not os.path.isfile(path):
        raise PriorsError(f"instance features table does not resolve: {path}")
    out: dict = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            k = (r["year"], r["category"], _norm_rel(r["onnx"]))
            out.setdefault(k, r)
    if not out:
        raise PriorsError(f"{path}: zero feature rows")
    return out


def _norm_rel(p: str) -> str:
    p = p.strip()
    if p.startswith("./"):
        p = p[2:]
    if p.endswith(".gz"):
        p = p[:-3]
    return p


def host_matches(flight: dict) -> bool:
    h = flight.get("host") or {}
    cores = h.get("logical_cores")
    ram = (h.get("ram_bytes") or 0) / 2**30
    return cores == 20 and 100 <= ram <= 140


def _num(text: str, key: str):
    m = re.search(rf"(?:^|\s){re.escape(key)}=([-0-9.eE+]+)", text or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def phase_observations(flight: dict) -> list[tuple[str, float, str]]:
    """Uncensored (phase, value, note) observations from one receipt's events.

    attack.upfront   seconds from the ordinary upfront execution start to its `upfront_attack` close, kept only
                     when the lane ended with no confirmed candidate (a success ends it early: censored) and
                     before 0.9 x its own effective budget (a cap ends it: censored).
    verifier.setup   seconds from the margin-row schedule allocation (the hand-off) to the verifier's first
                     internal event (`mip_reserve_demand`, emitted after the graph is built).
    root.bound       crown precheck seconds per clause, kept when every clause verified (the pass ran to the
                     end) or the pass ended before a quarter of the budget (no cap can have ended it).
    """
    ev = flight.get("events") or []
    budget = float(flight.get("budget_secs") or 0.0)
    out: list = []
    # attack.upfront
    for i, e in enumerate(ev):
        if e.get("method") != "upfront_attack_execution" or e.get("status") != "ran":
            continue
        reason = e.get("reason") or ""
        if "consultation=ordinary" not in reason:
            continue
        cap = _num(reason, "effective_budget_secs")
        close = next((x for x in ev[i + 1:] if x.get("method") == "upfront_attack" and x.get("status") == "ran"),
                     None)
        if close is None or cap is None or cap <= 0:
            break
        dur = float(close["at_secs"]) - float(e["at_secs"])
        if "no confirmed candidate" in (close.get("reason") or "") and 0 < dur < 0.9 * cap:
            out.append(("attack.upfront", dur, f"cap={cap:g}"))
        break
    # verifier.setup
    alloc = [e for e in ev if e.get("method") == "margin_row_schedule_execution"
             and "phase=allocation" in (e.get("reason") or "")]
    if alloc:
        a = alloc[-1]
        b = next((x for x in ev if x.get("method") == "mip_reserve_demand"
                  and float(x["at_secs"]) >= float(a["at_secs"])), None)
        if b is not None:
            dur = float(b["at_secs"]) - float(a["at_secs"])
            if dur > 0:
                out.append(("verifier.setup", dur, "handoff->verifier"))
    # root.bound
    begin = next((e for e in ev if e.get("method") == "crown_precheck_begin"), None)
    end = next((e for e in ev if e.get("method") == "crown_precheck_end"), None)
    if begin is not None and end is not None:
        k = _num(begin.get("reason"), "clauses")
        v = _num(end.get("reason"), "verified")
        t = _num(end.get("reason"), "elapsed_s")
        if k and k > 0 and v is not None and t is not None and t > 0:
            if v >= k or (budget > 0 and t < 0.25 * budget):
                out.append(("root.bound", t / k, f"clauses={k:g}"))
    return out


@dataclass
class Row:
    phase: str
    family: str
    category: str
    model: str  # model md5 (cluster)
    conv: int
    nonpwl: int
    params: float
    value: float
    receipt: str


def collect(roots, features_path: str) -> tuple[list[Row], dict]:
    """Training rows + a census of what was dropped and why (nothing silent)."""
    idents = scan_identities(roots)
    feats = load_features(features_path)
    rows: list[Row] = []
    census = {"identities": len(idents), "no_receipt": 0, "bad_receipt": 0, "category_mismatch": 0,
              "other_host": 0, "loaded": 0, "unresolved_path": 0, "paired_model": 0, "no_features": 0,
              "no_params": 0, "receipts_used": 0, "observations": 0}
    for res, (cat, onnx, _vnnlib) in sorted(idents.items()):
        fpath = res + ".flight.json"
        if not os.path.isfile(fpath):
            census["no_receipt"] += 1
            continue
        try:
            flight = json.load(open(fpath))
        except (OSError, ValueError):
            census["bad_receipt"] += 1
            continue
        if flight.get("category") != cat:
            census["category_mismatch"] += 1
            continue
        if not host_matches(flight):
            census["other_host"] += 1
            continue
        load = (flight.get("load_avg_at_begin") or [None])[0]
        if load is None or float(load) > LOAD_MAX:
            census["loaded"] += 1
            continue
        if onnx.startswith("["):
            census["paired_model"] += 1
            continue
        m = BENCH_RE.search(onnx)
        if not m:
            census["unresolved_path"] += 1
            continue
        feat = feats.get((m["year"], cat, _norm_rel(m["rel"])))
        if feat is None:
            census["no_features"] += 1
            continue
        params = float(_int(feat.get("n_params_init")))
        if params <= 0:
            census["no_params"] += 1
            continue
        conv, nonpwl = key_bits(feat)
        obs = phase_observations(flight)
        if obs:
            census["receipts_used"] += 1
        for phase, value, _note in obs:
            census["observations"] += 1
            rows.append(Row(phase, family_of(cat), cat, feat["model_md5"], conv, nonpwl, params, value,
                            os.path.relpath(fpath, "<home>/ny") if fpath.startswith("<home>/ny/")
                            else fpath))
    return rows, census


# ---- units, weights, fit ---------------------------------------------------------------------------------

@dataclass
class Unit:
    phase: str
    family: str
    model: str
    cell: str
    x: float  # log2(work)
    y: float  # log2(value)
    n_rows: int
    weight: float = 0.0


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise PriorsError("median of nothing")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _r6(v: float) -> float:
    return float(f"{v:.6f}")


def units_of(rows, phase: str) -> list[Unit]:
    """Collapse rows to (family, model) units (median value), family-balanced weights."""
    by: dict = {}
    for r in rows:
        if r.phase != phase:
            continue
        by.setdefault((r.family, r.model), []).append(r)
    units = []
    for (fam, model), rs in sorted(by.items()):
        first = rs[0]
        # Rounded to the fixture's 6 decimals BEFORE fitting, so the Rust re-derivation
        # (crates/ny-core/tests/router_priors.rs) fits bit-identical inputs.
        units.append(Unit(phase, fam, model, cell_token(first.conv, first.nonpwl),
                          _r6(math.log2(first.params)), _r6(math.log2(_median([r.value for r in rs]))),
                          len(rs)))
    per_fam: dict = {}
    for u in units:
        per_fam[u.family] = per_fam.get(u.family, 0) + 1
    for u in units:
        u.weight = 1.0 / per_fam[u.family]
    return units


def cell_census(units) -> dict:
    """cell -> {families, share_max, weight}."""
    out: dict = {}
    for u in units:
        c = out.setdefault(u.cell, {"fam_w": {}, "weight": 0.0, "units": 0})
        c["fam_w"][u.family] = c["fam_w"].get(u.family, 0.0) + u.weight
        c["weight"] += u.weight
        c["units"] += 1
    for c in out.values():
        c["families"] = len(c["fam_w"])
        c["share_max"] = max(c["fam_w"].values()) / c["weight"] if c["weight"] > 0 else 1.0
    return out


def passing_cells(units) -> set:
    return {cell for cell, c in cell_census(units).items()
            if c["families"] >= MIN_FAMILIES_PER_CELL and c["share_max"] <= PURITY_MAX}


@dataclass
class PhaseModel:
    phase: str
    slope: float
    intercepts: dict  # cell -> a (ROOT_CELL always present)
    spreads: dict  # cell -> weighted MAD of residuals (log2)
    cells_census: dict
    n_units: int
    n_families: int
    root_share_max: float

    def predict(self, cell: str, x: float) -> float:
        a = self.intercepts.get(cell, self.intercepts[ROOT_CELL])
        return a + self.slope * x


def _wls_line(units):
    """Weighted least squares y = a + b x."""
    sw = sum(u.weight for u in units)
    if sw <= 0:
        raise PriorsError("no weight")
    mx = sum(u.weight * u.x for u in units) / sw
    my = sum(u.weight * u.y for u in units) / sw
    sxx = sum(u.weight * (u.x - mx) ** 2 for u in units)
    sxy = sum(u.weight * (u.x - mx) * (u.y - my) for u in units)
    b = sxy / sxx if sxx > 1e-12 else 0.0
    return my - b * mx, b


def _wmedian(pairs):
    """Weighted median of (value, weight) pairs."""
    s = sorted(pairs)
    tot = sum(w for _, w in s)
    acc = 0.0
    for v, w in s:
        acc += w
        if acc >= 0.5 * tot:
            return v
    return s[-1][0]


def fit_phase(units) -> PhaseModel:
    if len(units) < MIN_UNITS_PER_PHASE:
        raise PriorsError(f"{len(units)} units < {MIN_UNITS_PER_PHASE}")
    fams = {u.family for u in units}
    if len(fams) < MIN_FAMILIES_PER_CELL:
        raise PriorsError(f"root spans {len(fams)} family(ies) < {MIN_FAMILIES_PER_CELL}")
    _, b = _wls_line(units)
    if not (SLOPE_MIN <= b <= SLOPE_MAX):
        raise PriorsError(f"G5 shared exponent {b:.3f} outside [{SLOPE_MIN}, {SLOPE_MAX}]")
    cells = passing_cells(units)
    intercepts = {ROOT_CELL: _wmedian([(u.y - b * u.x, u.weight) for u in units])}
    for cell in sorted(cells):
        intercepts[cell] = _wmedian([(u.y - b * u.x, u.weight) for u in units if u.cell == cell])
    spreads = {}
    for cell in intercepts:
        members = units if cell == ROOT_CELL else [u for u in units if u.cell == cell]
        a = intercepts[cell]
        spreads[cell] = max(SPREAD_FLOOR_LOG2,
                            _wmedian([(abs(u.y - (a + b * u.x)), u.weight) for u in members]))
    fam_w: dict = {}
    for u in units:
        fam_w[u.family] = fam_w.get(u.family, 0.0) + u.weight
    root_share = max(fam_w.values()) / sum(fam_w.values())
    return PhaseModel(units[0].phase, b, intercepts, spreads, cell_census(units), len(units), len(fams),
                      root_share)


# ---- LOGO and the gates ----------------------------------------------------------------------------------

def logo(units) -> dict:
    """Leave-one-family-out: refit without family G (cells re-gated), predict G's units.

    Returns per-family {heldout, full, null, gap} (median |log2 error| over G's units) and the summary."""
    full = fit_phase(units)
    fams = sorted({u.family for u in units})
    per: dict = {}
    for g in fams:
        train = [u for u in units if u.family != g]
        test = [u for u in units if u.family == g]
        try:
            held = fit_phase(_reweighted(train))
        except PriorsError as exc:
            per[g] = {"skipped": str(exc)}
            continue
        null = _wmedian([(u.y, u.weight) for u in _reweighted(train)])
        per[g] = {
            "heldout": _median([abs(u.y - held.predict(u.cell, u.x)) for u in test]),
            "full": _median([abs(u.y - full.predict(u.cell, u.x)) for u in test]),
            "null": _median([abs(u.y - null) for u in test]),
            "units": len(test),
        }
        per[g]["gap"] = per[g]["heldout"] - per[g]["full"]
    scored = [v for v in per.values() if "skipped" not in v]
    wins = sum(1 for v in scored if v["heldout"] < v["null"])
    losses = sum(1 for v in scored if v["heldout"] > v["null"])
    return {
        "per_family": per,
        "families": len(scored),
        "gap_median": _median([v["gap"] for v in scored]) if scored else None,
        "heldout_median": _median([v["heldout"] for v in scored]) if scored else None,
        "null_median": _median([v["null"] for v in scored]) if scored else None,
        "wins": wins,
        "losses": losses,
        "sign_p": sign_test_p(wins, losses),
    }


def _reweighted(units):
    per_fam: dict = {}
    for u in units:
        per_fam[u.family] = per_fam.get(u.family, 0) + 1
    out = []
    for u in units:
        out.append(Unit(u.phase, u.family, u.model, u.cell, u.x, u.y, u.n_rows, 1.0 / per_fam[u.family]))
    return out


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def fold_of(model: str) -> int:
    """Deterministic 2-fold split by model hash (held-out models, never held-out rows of one model)."""
    return hashlib.sha256(model.encode()).digest()[0] & 1


def fingerprint_gate(units, shipped_cells: set) -> dict:
    """G3: can the prior's INPUT KEY recover family identity on held-out models?

    Classifier = per-cell majority family (family-balanced weights), falling back to the global majority for a
    cell unseen in training; trained on one model-hash fold, scored on the other, both directions; metric =
    family-balanced accuracy. Baseline = the global-majority classifier under the same protocol. The key is
    the SHIPPED key: a cell that did not pass G2 maps to `*`."""
    def key(u):
        return u.cell if u.cell in shipped_cells else ROOT_CELL

    accs, bases = [], []
    for test_fold in (0, 1):
        train = _reweighted([u for u in units if fold_of(u.model) != test_fold])
        test = [u for u in units if fold_of(u.model) == test_fold]
        if not train or not test:
            continue
        glob_w: dict = {}
        cell_w: dict = {}
        for u in train:
            glob_w[u.family] = glob_w.get(u.family, 0.0) + u.weight
            cw = cell_w.setdefault(key(u), {})
            cw[u.family] = cw.get(u.family, 0.0) + u.weight
        gmaj = max(sorted(glob_w), key=lambda f: glob_w[f])
        cmaj = {c: max(sorted(w), key=lambda f: w[f]) for c, w in cell_w.items()}
        fam_hits: dict = {}
        fam_base: dict = {}
        for u in test:
            pred = cmaj.get(key(u), gmaj)
            fam_hits.setdefault(u.family, []).append(1.0 if pred == u.family else 0.0)
            fam_base.setdefault(u.family, []).append(1.0 if gmaj == u.family else 0.0)
        accs.append(sum(sum(v) / len(v) for v in fam_hits.values()) / len(fam_hits))
        bases.append(sum(sum(v) / len(v) for v in fam_base.values()) / len(fam_base))
    if not accs:
        raise PriorsError("fingerprint gate: a fold is empty")
    acc = sum(accs) / len(accs)
    base = sum(bases) / len(bases)
    return {"accuracy": acc, "baseline": base, "excess": acc - base, "bound": FINGERPRINT_EXCESS_MAX}


def gates_pass(units) -> bool:
    """G2-G6 on `units` (no G7): the ship decision a refit would make."""
    try:
        model = fit_phase(units)
    except PriorsError:
        return False
    shipped = set(model.intercepts) - {ROOT_CELL}
    lg = logo(units)
    if lg["gap_median"] is None or lg["gap_median"] > OVERFIT_GAP_MAX_LOG2 or lg["losses"] > lg["wins"]:
        return False
    if fingerprint_gate(units, shipped)["excess"] > FINGERPRINT_EXCESS_MAX:
        return False
    return isolation(units, shipped) == 0


def stability(units) -> tuple[int, int]:
    """G7: (folds whose leave-one-family-out refit still passes G2-G6, families)."""
    fams = sorted({u.family for u in units})
    ok = sum(1 for f in fams if gates_pass(_reweighted([u for u in units if u.family != f])))
    return ok, len(fams)


def isolation(units, shipped_cells: set) -> int:
    """G4: shipped cells that contain the units of exactly one family."""
    cen = cell_census([u for u in units if u.cell in shipped_cells])
    return sum(1 for c in cen.values() if c["families"] < 2)


# ---- the table -------------------------------------------------------------------------------------------

def build(rows) -> dict:
    """Fit every phase, run every gate, decide what ships. Pure."""
    phases = {}
    for phase in PHASES:
        units = units_of(rows, phase)
        rec: dict = {"units": len(units), "families": len({u.family for u in units})}
        try:
            model = fit_phase(units)
        except PriorsError as exc:
            rec["ships"] = False
            rec["refused"] = str(exc)
            phases[phase] = rec
            continue
        shipped = set(model.intercepts) - {ROOT_CELL}
        lg = logo(units)
        fp = fingerprint_gate(units, shipped)
        iso = isolation(units, shipped)
        refusals = []
        if lg["gap_median"] is None:
            refusals.append("G6 no family could be held out")
        elif lg["gap_median"] > OVERFIT_GAP_MAX_LOG2:
            refusals.append(f"G6 overfit gap {lg['gap_median']:.3f} > {OVERFIT_GAP_MAX_LOG2}")
        if lg["losses"] > lg["wins"]:
            refusals.append(f"G6 LOGO loses to the constant null ({lg['wins']} wins vs {lg['losses']} losses)")
        if fp["excess"] > FINGERPRINT_EXCESS_MAX:
            refusals.append(f"G3 fingerprint excess {fp['excess']:.3f} > {FINGERPRINT_EXCESS_MAX}")
        if iso:
            refusals.append(f"G4 {iso} isolating cell(s)")
        stable = stability(units)
        if stable[0] < stable[1]:
            refusals.append(f"G7 ship decision unstable under LOGO ({stable[0]} of {stable[1]} refits pass)")
        rec.update({"model": model, "logo": lg, "fingerprint": fp, "isolation": iso, "stable": stable,
                    "ships": not refusals, "refused": "; ".join(refusals) or None, "units_list": units})
        phases[phase] = rec
    return phases


def _f(v: float) -> str:
    return f"{v:.6f}"


def render_table(phases: dict, trainset_sha: str, census: dict) -> str:
    """The compiled table text. Deterministic; carries no family, category or path."""
    lines = [
        "# ny-router priors table: cost models only (time per phase), never a verdict or a route.",
        "# GENERATED by scripts/router_priors/build_priors.py fit -- do not hand-edit.",
        "# model: log2(value) = intercept + slope*log2(work); upper = + spread_k*spread.",
        "# Cells: the coarse operator key (conv x nonpwl) or the root '*'. Work: params.",
        f"schema\t{SCHEMA}",
        f"version\t{VERSION}",
        f"host\t{HOST_CLASS}",
        f"trainset_sha256\t{trainset_sha}",
        f"receipts\t{census.get('receipts_used', 0)}",
        f"spread_k\t{_f(SPREAD_K)}",
        f"spread_floor_log2\t{_f(SPREAD_FLOOR_LOG2)}",
        "layer_b\tuniform\tsuccess curves fit only on generic-board rows; none recorded",
        "gate\tmin_families_per_cell\t2",
        f"gate\tpurity_max\t{_f(PURITY_MAX)}",
        f"gate\tfingerprint_excess_max\t{_f(FINGERPRINT_EXCESS_MAX)}",
        f"gate\tslope_min\t{_f(SLOPE_MIN)}",
        f"gate\tslope_max\t{_f(SLOPE_MAX)}",
        f"gate\toverfit_gap_max_log2\t{_f(OVERFIT_GAP_MAX_LOG2)}",
        "# phase\tunit\twork\tcell\tintercept\tslope\tspread\tunits\tfamilies\tshare_max",
    ]
    for phase in PHASES:
        rec = phases[phase]
        if not rec.get("ships"):
            lines.append(f"refused\t{phase}\t{_sanitize(rec.get('refused') or 'no fit')}")
            continue
        m: PhaseModel = rec["model"]
        lg, fp = rec["logo"], rec["fingerprint"]
        lines.append(
            f"evidence\t{phase}\tlogo_families={lg['families']}\tgap_median={_f(lg['gap_median'])}\t"
            f"heldout_median={_f(lg['heldout_median'])}\tnull_median={_f(lg['null_median'])}\t"
            f"wins={lg['wins']}\tlosses={lg['losses']}\tsign_p={_f(lg['sign_p'])}\t"
            f"fp_accuracy={_f(fp['accuracy'])}\tfp_baseline={_f(fp['baseline'])}\t"
            f"fp_excess={_f(fp['excess'])}\tisolating={rec['isolation']}\t"
            f"stable_folds={rec['stable'][0]}\tstable_of={rec['stable'][1]}")
        for cell in sorted(m.intercepts, key=lambda c: (c != ROOT_CELL, c)):
            if cell == ROOT_CELL:
                units, fams, share = m.n_units, m.n_families, m.root_share_max
            else:
                c = m.cells_census[cell]
                units, fams, share = c["units"], c["families"], c["share_max"]
            lines.append(f"model\t{phase}\t{PHASE_UNIT[phase]}\tparams\t{cell}\t{_f(m.intercepts[cell])}\t"
                         f"{_f(m.slope)}\t{_f(m.spreads[cell])}\t{units}\t{fams}\t{_f(share)}")
    for cap, phs in BINDINGS:
        lines.append(f"bind\t{cap}\t{'+'.join(phs)}")
    return "\n".join(lines) + "\n"


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9 .:,;=<>()/_+-]", "_", text).replace("\t", " ")


def render_fixture(rows, phases: dict) -> str:
    """The anonymised training manifest for the Rust gates: one line per (phase, unit)."""
    fams = opaque_family_ids(r.family for r in rows)
    lines = ["# phase\tfamily\tmodel\tcell\tlog2_work\tlog2_value\trows",
             "# family = an OPAQUE leakage-group id; model = sha256(model md5)[:12]."]
    for phase in PHASES:
        for u in units_of(rows, phase):
            mid = hashlib.sha256(u.model.encode()).hexdigest()[:12]
            lines.append(f"{phase}\t{fams[u.family]}\t{mid}\t{u.cell}\t{_f(u.x)}\t{_f(u.y)}\t{u.n_rows}")
    return "\n".join(lines) + "\n"


def trainset_sha(rows) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: (r.phase, r.receipt, r.value)):
        h.update(f"{r.phase}\t{r.receipt}\t{r.model}\t{r.value:.9f}\n".encode())
    return h.hexdigest()


def write_rows(path: str, rows) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["phase", "family", "category", "model_md5", "conv", "nonpwl", "params", "value", "receipt"])
        for r in sorted(rows, key=lambda r: (r.phase, r.category, r.receipt)):
            w.writerow([r.phase, r.family, r.category, r.model, r.conv, r.nonpwl, int(r.params),
                        f"{r.value:.9f}", r.receipt])


def read_rows(path: str) -> list[Row]:
    if not os.path.isfile(path):
        raise PriorsError(f"rows file does not resolve: {path}")
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out.append(Row(r["phase"], r["family"], r["category"], r["model_md5"], int(r["conv"]),
                           int(r["nonpwl"]), float(r["params"]), float(r["value"]), r["receipt"]))
    if not out:
        raise PriorsError(f"{path}: zero rows")
    return out
