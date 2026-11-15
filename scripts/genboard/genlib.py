#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""GEN-R generic-board harness library (#router-r7, ROUTER_PROGRAM section 9, step R7).

Tooling only: python3 + PyYAML (through scripts/m0/m0lib.py). Nothing here runs cargo; the runner drives a
PINNED binary through `scripts/ny_box.sh measure` and never `ny benchmarks run` (which discards the `[router]`
observables that distinguish VOID from negative, CLAUDE.md).

THE BOARD
    GEN-R  = the pinned binary with a configs dir holding ONLY the generic default (one byte-identical yaml per
             category, so the preset resolver finds it; its only keys are router.mode / optimizer.mode /
             optimizer.priors) -- no category yaml, no per-category arming.
    TUNED  = the SAME pin with the shipped configs dir. A bank produced by another pin is accepted only under
             an explicit flag and is labelled NOT-SAME-PIN in every report line.
    Headline (section 9.8): generality ratio GEN-R / TUNED per year, rows gained / lost, and the per-family
             sign test; target GEN-R >= 0.95 x TUNED.

PROTOCOL (section 9.7-9.9, GENERALIZATION_DESIGN K4)
    * screen first: a seeded random order within each category (never instances.csv order), a fraction F;
    * rule-existence LOGO: in the fold that holds out family G, every rule whose evidence families are a
      subset of {G} is DISABLED (its ablation key) and G's rows run under that fold; a prior refit without G
      that changes a planning DECISION on G's rows needs a held-out binary (reported, not hidden);
    * clustered statistics: exact McNemar per category with the discordant rows collapsed to their model
      cluster, and an exact sign test across family groups (>= 17 of 23 scored 2026 groups for p < 0.05);
    * sealed external holdout: a file list hashed BEFORE anyone looks, a spend ledger, and a refusal to score
      holdout rows without the seal;
    * stop-loss: a category's GEN-R arm is aborted after 10 seeded rows if it decided <= 50% of what TUNED
      decided on the same rows;
    * any sat <-> unsat contradiction between GEN-R and TUNED or the field is a SOUNDNESS EMERGENCY.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "m0"))
sys.path.insert(0, os.path.join(HERE, "..", "router_priors"))

import m0lib as L  # noqa: E402
import priorslib as P  # noqa: E402

SCHEMA = "genboard/v1"
DECIDED = ("sat", "unsat")
STOPLOSS_ROWS = 10
STOPLOSS_FRAC = 0.5
TARGET_RATIO = 0.95
ALPHA = 0.05

# The ONLY keys the generic default may carry. Anything else is per-category arming in disguise.
GENERIC_KEYS = ("router.mode", "optimizer.mode", "optimizer.priors")
YEAR_DIRS = {2025: "vnncomp25", 2026: "vnncomp26"}


class GenError(Exception):
    """A refusal: the harness never emits an artefact it cannot vouch for."""


def family_of(category: str) -> str:
    return P.family_of(category)


# ---- statistics ------------------------------------------------------------------------------------------

def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial(n, 1/2) p for an outcome as extreme as k."""
    if n <= 0:
        return 1.0
    lo = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, lo + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def sign_test(wins: int, losses: int) -> float:
    return binom_two_sided(wins, wins + losses)


def sign_threshold(n: int, alpha: float = ALPHA) -> int | None:
    """Smallest k such that k of n same-direction groups is significant (two-sided p < alpha)."""
    for k in range(n // 2, n + 1):
        if binom_two_sided(k, n) < alpha:
            return k
    return None


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar: b = gen decided / tuned not, c = tuned decided / gen not."""
    return binom_two_sided(b, b + c)


def clustered_mcnemar(pairs) -> dict:
    """pairs: iterable of (cluster, gen_decided: bool, tuned_decided: bool).

    Each cluster contributes ONE signed vote: the sign of (gained - lost) inside it (rows of one model are not
    independent, GENERALIZATION_DESIGN K4.2). Returns the row-level counts (reported, anti-conservative) and the
    clustered exact sign test (the statistic the protocol uses)."""
    per: dict = {}
    b = c = 0
    for cluster, gen, tuned in pairs:
        d = per.setdefault(cluster, [0, 0])
        if gen and not tuned:
            d[0] += 1
            b += 1
        elif tuned and not gen:
            d[1] += 1
            c += 1
    up = sum(1 for g, lo in per.values() if g > lo)
    down = sum(1 for g, lo in per.values() if lo > g)
    return {"gained_rows": b, "lost_rows": c, "row_p": mcnemar_exact(b, c),
            "clusters_up": up, "clusters_down": down, "clustered_p": sign_test(up, down)}


# ---- the generic default configs dir ---------------------------------------------------------------------

def generic_yaml(router: str = "on", optimizer: str = "on", priors: bool = True) -> str:
    for token, what in ((router, "router.mode"), (optimizer, "optimizer.mode")):
        if token not in ("off", "shadow", "on"):
            raise GenError(f"{what}={token!r} is not off|shadow|on")
    # `priors` is OMITTED (not `false`) when off, so the same generic dir also runs on a pin that predates
    # the key (its preset schema denies unknown fields).
    return ("# GEN-R generic default (scripts/genboard): the ONLY preset content of this configs dir.\n"
            "# Byte-identical for every category; no category-specific key may appear here.\n"
            f"router:\n  mode: \"{router}\"\n"
            f"optimizer:\n  mode: \"{optimizer}\"\n" + ("  priors: true\n" if priors else ""))


def flat_keys(doc, prefix="") -> list[str]:
    out = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.extend(flat_keys(v, key + "."))
            else:
                out.append(key)
    return out


def is_ablation_key(key: str) -> bool:
    """A LOGO fold's ablation key: one rule switch (`router.rules.*` / `optimizer.rules.*`) or the prior switch.
    Rule switches are generic by construction (every category of a fold gets the same bytes)."""
    return key == "optimizer.priors" or (
        key.count(".") == 2 and key.startswith(("router.rules.", "optimizer.rules.")))


def check_generic_text(text: str, fold: bool = False) -> list[str]:
    """Violations of the generic-default contract (empty = ok). `fold` (review fix #router-r7): a LOGO fold's
    text is the generic default PLUS ablation keys set to `false` -- the runner must accept exactly that (and
    nothing else), or the folds `logo` writes can never be run."""
    bad = []
    try:
        doc = L.strict_load(text) or {}
    except Exception as exc:  # noqa: BLE001 - any parse failure is a refusal
        return [f"generic default does not parse: {exc}"]
    keys = flat_keys(doc)
    for k in keys:
        if k in GENERIC_KEYS:
            continue
        if fold and is_ablation_key(k) and L.deep_get(doc, k) is False:
            continue
        bad.append(f"generic default carries a non-generic key {k!r}"
                   + (" (a fold may only add rule switches set to false)" if fold else ""))
    if L.deep_get(doc, "router.mode") not in ("off", "shadow", "on"):
        bad.append("router.mode missing or invalid")
    return bad


def priors_armed(text: str) -> bool:
    """Whether a generic (or fold) text asks the router to read the offline prior."""
    try:
        return L.deep_get(L.strict_load(text) or {}, "optimizer.priors") is True
    except Exception:  # noqa: BLE001 - an unparsable text is refused elsewhere
        return False


def write_configs(out: str, categories, years=(2025, 2026), text: str | None = None) -> dict:
    """Create the GEN-R configs dir: one byte-identical generic yaml per (year dir, category).

    Refuses an existing destination (never deletes). Returns the manifest written beside the files."""
    if os.path.exists(out):
        raise GenError(f"refusing: {out} already exists (this harness never deletes; pick a new --out)")
    text = text if text is not None else generic_yaml()
    bad = check_generic_text(text)
    if bad:
        raise GenError("; ".join(bad))
    files = []
    for year in years:
        yd = os.path.join(out, YEAR_DIRS[year])
        os.makedirs(yd)
        for cat in sorted(set(categories)):
            p = os.path.join(yd, f"{cat.lower()}.yaml")
            with open(p, "w") as fh:
                fh.write(text)
            files.append(os.path.relpath(p, out))
    sha = hashlib.sha256(text.encode()).hexdigest()
    manifest = {"schema": SCHEMA, "kind": "gen-r-configs", "generic_sha256": sha, "files": files,
                "keys": flat_keys(L.strict_load(text))}
    with open(os.path.join(out, "GENERIC_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return manifest


def verify_configs(configs_dir: str, categories, fold: bool = False) -> dict:
    """The configs dir is GENERIC: every yaml under it is byte-identical to every other, carries only generic
    keys (plus, for a LOGO `fold`, rule switches set to false), and every category RESOLVES to one of them
    through the binary's resolver mirror. Returns the generic sha and whether the text arms the prior; raises
    on any violation (all violations reported at once)."""
    if not os.path.isdir(configs_dir):
        raise GenError(f"configs dir does not resolve: {configs_dir}")
    texts = {}
    for dirpath, _dirs, files in os.walk(configs_dir):
        for name in files:
            if name.endswith((".yaml", ".yml")):
                p = os.path.join(dirpath, name)
                texts[os.path.relpath(p, configs_dir)] = open(p).read()
    bad = []
    if not texts:
        bad.append("no yaml under the configs dir")
    shas = {hashlib.sha256(t.encode()).hexdigest() for t in texts.values()}
    if len(shas) > 1:
        bad.append(f"{len(shas)} distinct yaml texts: a generic board has exactly one")
    for rel, t in sorted(texts.items()):
        for v in check_generic_text(t, fold=fold):
            bad.append(f"{rel}: {v}")
    for cat in sorted(set(categories)):
        if L.resolve_preset(configs_dir, cat) is None:
            bad.append(f"category {cat} resolves to NO preset (AUTO DEFAULTS would run, not GEN-R)")
    if bad:
        raise GenError("generic configs check failed: " + "; ".join(bad))
    return {"generic_sha256": next(iter(shas)), "files": len(texts),
            "priors": priors_armed(next(iter(texts.values())))}


# ---- the screen plan -------------------------------------------------------------------------------------

@dataclass
class PlanRow:
    year: int
    category: str
    family: str
    index: int
    onnx: str
    vnnlib: str
    budget: int
    model: str
    order: str
    holdout: bool = False

    def key(self) -> tuple:
        return (self.year, self.category, self.index)


PLAN_COLUMNS = ("year", "category", "family", "index", "onnx", "vnnlib", "budget", "model", "order", "holdout")


def seeded_order(seed: str, year: int, category: str, index: int) -> str:
    return hashlib.sha256(f"{seed}|{year}|{category}|{index}".encode()).hexdigest()


def screen(rows: list[PlanRow], frac: float) -> list[PlanRow]:
    """Per category: seeded order, keep ceil(frac * n) (at least 1). Never instances.csv order."""
    if not (0 < frac <= 1):
        raise GenError(f"screen fraction {frac} not in (0, 1]")
    by: dict = {}
    for r in rows:
        by.setdefault((r.year, r.category), []).append(r)
    out = []
    for key in sorted(by):
        rs = sorted(by[key], key=lambda r: r.order)
        out.extend(rs[: max(1, math.ceil(frac * len(rs)))])
    return out


def build_plan(ny_root: str, year: int, categories, seed: str, frac: float, check_files: bool = True,
               sealed: set | None = None) -> list[PlanRow]:
    rows = []
    for cat in categories:
        inst = L.load_instances(L.scored_instances_csv(ny_root, year, cat), check_files=check_files)
        for r in inst:
            rows.append(PlanRow(year, cat, family_of(cat), r.index, r.onnx, r.vnnlib, r.budget,
                                L.model_key(r.onnx_rel), seeded_order(seed, year, cat, r.index),
                                holdout=bool(sealed and (year, cat, r.index) in sealed)))
    return screen([r for r in rows if not r.holdout], frac)


def write_plan(path: str, rows: list[PlanRow]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(PLAN_COLUMNS)
        for r in rows:
            w.writerow([r.year, r.category, r.family, r.index, r.onnx, r.vnnlib, r.budget, r.model, r.order,
                        int(r.holdout)])


def read_plan(path: str) -> list[PlanRow]:
    if not os.path.isfile(path):
        raise GenError(f"plan does not resolve: {path}")
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out.append(PlanRow(int(r["year"]), r["category"], r["family"], int(r["index"]), r["onnx"],
                               r["vnnlib"], int(r["budget"]), r["model"], r["order"], r["holdout"] == "1"))
    if not out:
        raise GenError(f"{path}: empty plan (an empty board must never look like a score)")
    return out


# ---- run results and judging -----------------------------------------------------------------------------

@dataclass
class Outcome:
    key: tuple
    verdict: str  # sat | unsat | unknown | timeout | error | void
    wall: float | None
    void_reason: str | None = None
    router_done: bool = False
    router_banner: bool = False
    # Review fix (#router-r7): what the binary PUBLISHED for THIS row, even when the board VOIDs the row for a
    # missing observable. A VOID makes a row unscorable; it must never hide a wrong verdict (the moat is
    # absolute). `None` when there is no result token or the log is not this row's (identity VOID).
    published: str | None = None
    # The offline prior on this row: `read` (priced on the table's host class), `host_gated` (the row's host
    # class is not the table's: the prior priced nothing), `unavailable`, or `-` (no `[router] priors` line).
    prior: str = "-"
    # The `[router] v1` banner's host class (`-` when no banner).
    host: str = "-"


RESULT_RE = re.compile(r"^\s*(sat|unsat|unknown|timeout|error)\b", re.I)
IDENTITY_VOIDS = ("no result file token", "the log's INSTANCE line is not this row")


def published_verdict(o: "Outcome | None") -> str | None:
    """The verdict a contradiction check must read: the judged verdict, or -- for a row VOIDed for a missing
    observable -- what the binary published anyway. Identity VOIDs (no result / another row's log) carry
    nothing attributable to this row."""
    if o is None:
        return None
    if o.verdict != "void":
        return o.verdict
    return o.published


def _strip_gz(path: str) -> str:
    return path[:-3] if path.endswith(".gz") else path


def same_model(logged_onnx: str, row_onnx: str) -> bool:
    """The log's INSTANCE onnx names THIS row's network (review fix: a vnnlib basename alone is shared by many
    rows, e.g. one ACAS Xu property across 45 networks, so a wrong-network run passed the identity check)."""
    files = L.model_files(row_onnx)
    if len(files) > 1:
        return all(os.path.basename(_strip_gz(f)) in logged_onnx for f in files)
    return os.path.basename(_strip_gz(logged_onnx)) == os.path.basename(_strip_gz(row_onnx))


def prior_state(raw_text: str) -> str:
    """`read` / `host_gated` / `unavailable` / `-` from the row's `[router] priors` line (review fix: under a
    cgroup cap below the table's host-class floor -- e.g. the MemoryMax=96G the protocol recommends for GEN arms
    on the GB10, whose class needs >= 100 GiB -- the host class becomes `other` and the prior silently prices
    nothing; the board must see that, not assume the prior was read)."""
    for line in raw_text.splitlines():
        if line.startswith("[router] priors unavailable"):
            return "unavailable"
        if line.startswith("[router] priors v="):
            return "host_gated" if re.search(r"[:=,]host:", line) else "read"
    return "-"


def banner_host(raw_text: str) -> str:
    m = re.search(r"^\[router\] v1 .*? host=(\S+)", raw_text, re.M)
    return m.group(1) if m else "-"


def parse_result_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    head = open(path, errors="replace").read(256)
    m = RESULT_RE.match(head)
    return m.group(1).lower() if m else None


def preset_base_of(flight: dict | None, raw_text: str) -> str | None:
    """The DURABLE preset the run started from: the flight sidecar's `preset_base` event (the effective yaml
    may be a temporary materialisation), else the log's `Loading preset:` line."""
    for e in (flight or {}).get("events") or []:
        if e.get("method") == "preset_base" and e.get("status") == "ran":
            return (e.get("reason") or "").strip() or None
    m = re.search(r"Loading preset: (\S+)", raw_text)
    return m.group(1) if m else None


def judge_run(row: PlanRow, raw_text: str, result_token: str | None, wall: float | None, router_on: bool,
              configs_dir: str | None = None, preset_path: str | None = None,
              priors_on: bool = False) -> Outcome:
    """One run -> an Outcome. VOID when the run cannot be trusted to be what the board says it is:
    no result, no `Running VNN-COMP instance` line for THIS row (category, network AND property), a preset not
    loaded from the board's own configs dir (the arm would be another arm), the router required but its banner
    or its `[router] done` missing (CLAUDE.md: a router run without `[router] done` is VOID), or the prior
    armed (`priors_on`) but no `[router] priors` line (the key is dark: another arm). A VOID for a missing
    observable still records what the binary PUBLISHED, for the contradiction check."""
    key = row.key()
    banner = "[router] v1 mode=" in raw_text
    done = "[router] done faults=" in raw_text
    prior = prior_state(raw_text)
    host = banner_host(raw_text)
    inst = re.search(r"Running VNN-COMP instance: category='([^']+)' onnx='([^']+)' vnnlib='([^']+)'", raw_text)

    def void(why: str, published: str | None) -> Outcome:
        return Outcome(key, "void", wall, why, done, banner, published, prior, host)

    if result_token is None:
        return void(IDENTITY_VOIDS[0], None)
    if (inst is None or inst.group(1) != row.category or not same_model(inst.group(2), row.onnx)
            or os.path.basename(inst.group(3)) != os.path.basename(row.vnnlib)):
        return void(IDENTITY_VOIDS[1], None)
    if configs_dir is not None:
        root = os.path.realpath(configs_dir)
        if preset_path is None or not os.path.realpath(preset_path).startswith(root + os.sep):
            return void("the preset was not loaded from this arm's configs dir", result_token)
    if router_on and not banner:
        return void("router required but no [router] v1 banner (arm dark)", result_token)
    if router_on and not done:
        return void("router required but no [router] done line", result_token)
    if priors_on and prior == "-":
        return void("optimizer.priors armed but no [router] priors line (the key is dark)", result_token)
    return Outcome(key, result_token, wall, None, done, banner, result_token, prior, host)


def stop_loss(gen_rows: list[Outcome], tuned_decided: dict) -> bool:
    """After STOPLOSS_ROWS seeded rows of one category: abort if GEN-R decided <= 50% of what TUNED decided."""
    if len(gen_rows) < STOPLOSS_ROWS:
        return False
    first = gen_rows[:STOPLOSS_ROWS]
    t = sum(1 for o in first if tuned_decided.get(o.key) in DECIDED)
    g = sum(1 for o in first if o.verdict in DECIDED)
    return t > 0 and g <= STOPLOSS_FRAC * t


def dark_arm(gen_rows: list[Outcome]) -> bool:
    """Review fix (#router-r7): after STOPLOSS_ROWS seeded rows, an arm whose every row is VOID is DARK -- the
    board could not see it at all. That is a stop with no capability verdict, never a stop-loss (which would
    read a VOID arm as a GEN-R negative)."""
    first = gen_rows[:STOPLOSS_ROWS]
    return len(first) >= STOPLOSS_ROWS and all(o.verdict == "void" for o in first)


def contradiction(a: str | None, b: str | None) -> bool:
    return a in DECIDED and b in DECIDED and a != b


def write_results(path: str, rows: list[PlanRow], outcomes: dict) -> None:
    """Bank-format CSV (category,onnx,vnnlib,prepare,verdict,wall) + a VOID column-free sibling .judged.tsv."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        for r in rows:
            o = outcomes.get(r.key())
            if o is None:
                continue
            w.writerow([r.category, r.onnx, r.vnnlib, "0.0", o.verdict, "" if o.wall is None else f"{o.wall:.1f}"])
    with open(path + ".judged.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["year", "category", "index", "verdict", "wall", "void_reason", "router_banner", "router_done",
                    "published", "prior", "host"])
        for r in rows:
            o = outcomes.get(r.key())
            if o is None:
                continue
            w.writerow([r.year, r.category, r.index, o.verdict, "" if o.wall is None else f"{o.wall:.1f}",
                        o.void_reason or "", int(o.router_banner), int(o.router_done), o.published or "",
                        o.prior, o.host])


def read_judged(path: str) -> dict:
    if not os.path.isfile(path):
        raise GenError(f"judged table does not resolve: {path}")
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            key = (int(r["year"]), r["category"], int(r["index"]))
            # A table written before the `published` column: a decided verdict IS its published verdict; a
            # VOID row's publication is unknown (None).
            published = r.get("published")
            if published is None:
                published = r["verdict"] if r["verdict"] != "void" else ""
            out[key] = Outcome(key, r["verdict"], float(r["wall"]) if r["wall"] else None, r["void_reason"] or None,
                               r["router_done"] == "1", r["router_banner"] == "1", published or None,
                               r.get("prior") or "-", r.get("host") or "-")
    return out


# ---- scoring ---------------------------------------------------------------------------------------------

@dataclass
class CategoryScore:
    year: int
    category: str
    family: str
    rows: int
    gen_decided: int
    tuned_decided: int
    gen_void: int
    contradictions: int
    truth_contradictions: int
    gen_raw: int
    tuned_raw: int
    gained: int
    lost: int
    mcnemar: dict = field(default_factory=dict)
    stopped: bool = False

    @property
    def ratio(self):
        return None if self.tuned_raw <= 0 else self.gen_raw / self.tuned_raw


def score(plan: list[PlanRow], gen: dict, tuned: dict, truth: dict | None = None) -> dict:
    """GEN-R vs TUNED on the SAME rows. gen / tuned: key -> Outcome (tuned may come from a bank: key -> verdict
    string wrapped as an Outcome). truth: key -> 'sat' | 'unsat' | '-'.

    raw = 10 x decided - 150 x contradicted (a decided GEN-R verdict that contradicts TUNED or the field is
    counted WRONG provisionally and raises the EMERGENCY flag: one of the two answers is wrong)."""
    by: dict = {}
    for r in plan:
        by.setdefault((r.year, r.category), []).append(r)
    cats = []
    emergencies = []
    for (year, cat), rows in sorted(by.items()):
        gd = td = void = contra = tcontra = wrong = gained = lost = 0
        pairs = []
        for r in rows:
            g = gen.get(r.key())
            t = tuned.get(r.key())
            gv = g.verdict if g else None
            tv = t.verdict if t else None
            # Review fix (#router-r7): contradictions read what each arm PUBLISHED, so a row the board VOIDs
            # for a missing observable can never hide a wrong verdict; and TUNED (the same pin) is checked
            # against the field too.
            gp = published_verdict(g)
            tp = published_verdict(t)
            tr = (truth or {}).get(r.key())
            if gv == "void":
                void += 1
            vs_tuned = contradiction(gp, tp)
            vs_field = tr in DECIDED and contradiction(gp, tr)
            if vs_tuned:
                contra += 1
                emergencies.append(f"{year} {cat} #{r.index}: GEN-R {gp}{' (VOID row)' if gv == 'void' else ''} "
                                   f"vs TUNED {tp}")
            if vs_field:
                tcontra += 1
                emergencies.append(f"{year} {cat} #{r.index}: GEN-R {gp}{' (VOID row)' if gv == 'void' else ''} "
                                   f"vs FIELD {tr}")
            if tr in DECIDED and contradiction(tp, tr):
                emergencies.append(f"{year} {cat} #{r.index}: TUNED {tp} vs FIELD {tr}")
            wrong += int(vs_tuned or vs_field)
            gdec = gv in DECIDED and not (vs_tuned or vs_field)
            tdec = tv in DECIDED
            gd += int(gdec)
            td += int(tdec)
            gained += int(gdec and not tdec)
            lost += int(tdec and not gdec)
            pairs.append((r.model, gdec, tdec))
        cats.append(CategoryScore(year, cat, family_of(cat), len(rows), gd, td, void, contra, tcontra,
                                  10 * gd - 150 * wrong, 10 * td, gained, lost, clustered_mcnemar(pairs)))
    # family sign test: one vote per family (sum of its categories' raw difference)
    fam: dict = {}
    for c in cats:
        fam.setdefault((c.year, c.family), 0)
        fam[(c.year, c.family)] += c.gen_raw - c.tuned_raw
    # Review fix (#router-r7): the sign test and the headline ratio are PER YEAR (section 9.8). The twin year
    # is reproducibility only -- 20 of 22 shared categories load the same yaml -- so pooling a two-year plan
    # would count one family twice and inflate n.
    per_year: dict = {}
    for year in sorted({c.year for c in cats}):
        votes = [v for (y, _f), v in fam.items() if y == year]
        y_up = sum(1 for v in votes if v > 0)
        y_down = sum(1 for v in votes if v < 0)
        g = sum(c.gen_raw for c in cats if c.year == year)
        t = sum(c.tuned_raw for c in cats if c.year == year)
        y_ratio = None if t <= 0 else g / t
        per_year[year] = {"family_up": y_up, "family_down": y_down, "family_sign_p": sign_test(y_up, y_down),
                          "family_threshold": sign_threshold(len(votes)), "families": len(votes),
                          "gen_raw": g, "tuned_raw": t, "ratio": y_ratio,
                          "meets_target": y_ratio is not None and y_ratio >= TARGET_RATIO}
    only = per_year[next(iter(per_year))] if len(per_year) == 1 else None
    tot_g = sum(c.gen_raw for c in cats)
    tot_t = sum(c.tuned_raw for c in cats)
    ratio = None if tot_t <= 0 else tot_g / tot_t
    return {
        "categories": cats,
        "families": {f"{y}:{f}": v for (y, f), v in sorted(fam.items())},
        "per_year": per_year,
        # The top-level sign test is the single year's; a multi-year plan has NO pooled sign test.
        "family_up": only["family_up"] if only else None,
        "family_down": only["family_down"] if only else None,
        "family_sign_p": only["family_sign_p"] if only else None,
        "family_threshold": only["family_threshold"] if only else None,
        "gen_raw": tot_g,
        "tuned_raw": tot_t,
        "ratio": ratio,
        "meets_target": (all(y["meets_target"] for y in per_year.values()) if per_year else False),
        "emergencies": emergencies,
        "voids": sum(c.gen_void for c in cats),
    }


def render_score(s: dict, same_pin: bool) -> str:
    lines = []
    tag = "" if same_pin else " [NOT-SAME-PIN: TUNED from a bank]"
    lines.append(f"{'year':<5} {'category':<40} {'fam':<16} {'rows':>4} {'gen':>4} {'tun':>4} {'void':>4} "
                 f"{'+':>3} {'-':>3} {'ratio':>6} {'clus_p':>7}{tag}")
    for c in s["categories"]:
        ratio = "-" if c.ratio is None else f"{c.ratio:.3f}"
        lines.append(f"{c.year:<5} {c.category:<40} {c.family:<16} {c.rows:>4} {c.gen_decided:>4} "
                     f"{c.tuned_decided:>4} {c.gen_void:>4} {c.gained:>3} {c.lost:>3} {ratio:>6} "
                     f"{c.mcnemar.get('clustered_p', 1.0):>7.3f}")
    for year, y in sorted(s["per_year"].items()):
        ratio = "-" if y["ratio"] is None else f"{y['ratio']:.4f}"
        lines.append(f"{year}: GEN-R/TUNED raw {y['gen_raw']}/{y['tuned_raw']} = {ratio} (target >= {TARGET_RATIO}: "
                     f"{'MET' if y['meets_target'] else 'NOT MET'}){tag}")
        lines.append(f"{year}: family sign test: up={y['family_up']} down={y['family_down']} "
                     f"p={y['family_sign_p']:.4f} (significance needs >= {y['family_threshold']} of "
                     f"{y['families']} same-direction groups)")
    if len(s["per_year"]) > 1:
        lines.append("years are NOT pooled: the twin year is reproducibility only (section 9.8)")
    for e in s["emergencies"]:
        lines.append(f"EMERGENCY {e}")
    return "\n".join(lines)


# ---- rule-existence LOGO ---------------------------------------------------------------------------------

def read_rule_evidence(path: str) -> list[dict]:
    """rule_evidence.tsv: rule, ablation key, evidence families (comma list, or `structural`)."""
    if not os.path.isfile(path):
        raise GenError(f"rule evidence table does not resolve: {path}")
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader((line for line in fh if not line.startswith("#")), delimiter="\t"):
            fams = r["evidence"].strip()
            # `structural` (no fitted constant) and `multi` (evidence spans many families by construction,
            # e.g. the priors' G2) are never disabled by a fold.
            out.append({"rule": r["rule"], "key": r["ablation_key"],
                        "evidence": None if fams in ("structural", "multi") else
                        {f.strip() for f in fams.split(",") if f.strip()},
                        "note": r.get("note", "")})
    if not out:
        raise GenError(f"{path}: no rules")
    return out


def logo_folds(plan: list[PlanRow], rules: list[dict]) -> dict:
    """family -> the ablation keys its held-out fold sets: every rule whose evidence families are a SUBSET of
    {family} is disabled there (GENERALIZATION_DESIGN K4.1). Structural rules (no fitted constant) are never
    disabled. A family with no such rule runs under the plain GEN-R configs (its GEN-R row IS its held-out
    row) -- and the report says so explicitly instead of hiding a zero gap."""
    fams = sorted({r.family for r in plan})
    out = {}
    for f in fams:
        keys = sorted(rule["key"] for rule in rules
                      if rule["evidence"] is not None and rule["evidence"] and rule["evidence"] <= {f})
        out[f] = keys
    return out


def fold_yaml(base_text: str, ablations: list[str]) -> str:
    """The generic default plus `key: false` for each ablation key (dotted path)."""
    doc = L.strict_load(base_text) or {}
    for key in ablations:
        L.deep_set(doc, key, False)
    import yaml
    return ("# GEN-R LOGO fold: the generic default with the rules whose evidence is this family disabled.\n"
            + yaml.safe_dump(doc, sort_keys=True))


def prior_decision_changes(plan: list[PlanRow], trainset_rows, features: dict, join_margin_ms: int = 2000,
                           build=None, clauses: dict | None = None) -> dict:
    """family -> plan rows of that family whose PLANNING DECISION changes when the cost-model prior is refit
    WITHOUT the family: a bound capability's rung fits the window under one fit and not under the other
    (CR-C / CR-A). Value-only changes (a different contract number with the same fit/decline) are counted
    separately: they change logs, not behaviour. Rows needing a held-out binary are the decision changes.

    `clauses`: (year, category, index) -> the row's clause count, for `secs_per_clause` phases (review fix: the
    mirror summed a per-clause time as if it were the phase's time). A rung that cannot be computed on one
    side (a per-clause phase without a clause count, or a model without features) is UNDECIDABLE here and is
    counted as a decision change when the other side prices it (fail closed: it needs a held-out binary).
    The comparison is per binding against the whole window; the Rust plan's sequential CR-C `left` can make
    a rung that fits the window decline after earlier contracts, so this report is a lower bound on the
    decisions a held-out binary could change only through that order (it never hides a fit/decline flip)."""
    build = build or P.build
    phases_full = build(trainset_rows)
    out = {}
    for fam in sorted({r.family for r in plan}):
        held_rows = [t for t in trainset_rows if t.family != fam]
        try:
            phases_held = build(held_rows)
        except P.PriorsError:
            phases_held = {}
        dec = val = undecidable = 0
        for r in (x for x in plan if x.family == fam):
            feat = features.get((r.year, r.category, r.model))
            if feat is None:
                continue
            k = (clauses or {}).get(r.key())
            window = r.budget * 1000 - max(5000, r.budget * 1000 // 20) - join_margin_ms
            for cap, phs in P.BINDINGS:
                a = _rung(phases_full, phs, feat, k)
                b = _rung(phases_held, phs, feat, k)
                if a is None and b is None:
                    continue
                if UNDECIDABLE in (a, b):
                    undecidable += 1
                    dec += 1
                elif (a is None) != (b is None) or ((a <= window) != (b <= window)):
                    dec += 1
                elif a != b:
                    val += 1
        out[fam] = {"decision_changes": dec, "value_changes": val}
        if undecidable:
            out[fam]["undecidable"] = undecidable
    return out


# The Rust reader's caps (`ny_core::router::priors`).
PREDICTION_CEILING_MS = 86_400_000
UNDECIDABLE = "undecidable"


def _rung(phases: dict, phs, feat, clauses: int | None = None):
    """Sum of upper predictions in ms (the Rust reader's arithmetic: per phase ceil(ms) >= 1 with upper >=
    median, the unit's scale -- `secs_per_clause` x clauses -- and the one-day ceiling), None when any phase
    does not ship, UNDECIDABLE when a per-clause phase ships but the row's clause count is unknown."""
    total = 0
    conv, nonpwl, params = feat
    cell = P.cell_token(conv, nonpwl)
    for ph in phs:
        rec = phases.get(ph)
        if not rec or not rec.get("ships"):
            return None
        m = rec["model"]
        unit = P.PHASE_UNIT.get(ph, "secs")
        if unit == "secs":
            scale = 1.0
        elif unit == "secs_per_clause":
            if not clauses or clauses <= 0:
                return UNDECIDABLE
            scale = float(clauses)
        else:
            return UNDECIDABLE
        a = m.intercepts.get(cell, m.intercepts[P.ROOT_CELL])
        spread = m.spreads.get(cell if cell in m.intercepts else P.ROOT_CELL)
        med = a + m.slope * math.log2(params)
        up = med + P.SPREAD_K * spread

        def ms(log2_secs: float) -> int:
            v = math.ceil(2 ** log2_secs * scale * 1000)
            return PREDICTION_CEILING_MS if v >= PREDICTION_CEILING_MS else max(1, v)

        total += max(ms(up), ms(med))
    return min(total, PREDICTION_CEILING_MS)


# ---- sealed holdout --------------------------------------------------------------------------------------

def seal_list(entries) -> str:
    """sha256 over the sorted, de-duplicated holdout entries (each a 'year|category|index' or a file path)."""
    h = hashlib.sha256()
    for e in sorted(set(entries)):
        h.update(e.encode() + b"\n")
    return h.hexdigest()


def write_seal(path: str, entries, note: str) -> dict:
    if os.path.exists(path):
        raise GenError(f"refusing: {path} exists (a seal is written once; never overwritten)")
    rec = {"schema": SCHEMA, "kind": "holdout-seal", "sha256": seal_list(entries), "count": len(set(entries)),
           "note": note}
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return rec


def verify_seal(path: str, entries) -> dict:
    if not os.path.isfile(path):
        raise GenError(f"seal does not resolve: {path}")
    rec = json.load(open(path))
    got = seal_list(entries)
    if rec.get("sha256") != got or rec.get("count") != len(set(entries)):
        raise GenError(f"holdout list does not match its seal ({got[:12]} vs {str(rec.get('sha256'))[:12]}): "
                       "the holdout was edited after sealing")
    return rec


def spend_holdout(ledger: str, seal: dict, purpose: str, allow: bool) -> int:
    """Append one spend line; refuse a second spend unless explicitly allowed (each look costs the holdout)."""
    lines = open(ledger).read().splitlines() if os.path.isfile(ledger) else []
    spent = [x for x in lines if x and not x.startswith("#") and x.split("\t")[0] == seal["sha256"]]
    if spent and not allow:
        raise GenError(f"the sealed holdout {seal['sha256'][:12]} was already spent {len(spent)} time(s); "
                       "pass --allow-respend and record why")
    with open(ledger, "a") as fh:
        if not lines:
            fh.write("# seal_sha256\tspend_no\tpurpose\n")
        fh.write(f"{seal['sha256']}\t{len(spent) + 1}\t{purpose}\n")
    return len(spent) + 1
