#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""#prepared-root-route: the pre-registered S8 moat plan of a KEYLESS, DEFAULT-ON routable capability.

`seallib.derive` refuses a keyless entry ("no overlay arms it"), and rightly for a capability the router would ARM: a
default-on capability is already armed by the shipped code, so its moat runs the SHIPPED configs (no overlay, no
control arm) on exactly the rows its own admission admits. This driver writes that plan from an admission census
directory (`seal.py census --pin <pin> --out <dir>`: corpus.tsv, corpus_map.json, census.tsv, counts.json) with the
pipeline's own functions only -- `census_corpus` (the current moat corpus; the census must bind to it and to the pin),
`census_admitted` (the `admit` column) and `s8_rows` (m0 moat rows, both years, every scored category, the defect
alarms on every row) -- and runs nothing.

  plan        --cap <id> --census-dir <dir> --pin <bin> --out <dir>   write <out>/plan.json (+ .sha256), the m0 specs
              (specs/S8.tsv = the census admits; specs/S8T.tsv = the superset tail, below) and configs/src (the pin
              commit's configs, as `seal.py plan` extracts them). Refuses an existing plan, an unbound census, a
              capability with a live key, or a census that admits no moat row.

The superset tail: rows the census refused ONLY for reasons that do not bound where a default-on capability runs --
`envelope:ops` (an operator kind no seal records) and `admit:unknown:*` (facts the router could not read at start) --
except rows of the relational route (facts `unknown:relational-route`), which returns before any other lane runs.
CR-RS would never route there, but a default-on capability is not routed: it runs wherever its own admission admits.
review(prepared-root-route): a tail row the capability's OWN property gates refuse is left out too
(`LANE_PROPERTY_GATES`, each fact pinned by a named Rust test): a spec read the lane provably refuses (prepared-root:
`unknown:spec>16MiB`, above its 1 MiB staged ceiling) and, where the lane's property acceptance is contained in its
served archetypes, a KNOWN archetype outside them (a facts-unknown refusal short-circuits the census's own
`spec-not-served` decline, so the census does not say it). Census g3's tail was 158 rows, every one of them such a
row (147 `spec>16MiB` + 11 half-space / disjunctive): a stage the judge could only call VOID (no run can engage).

The plan is CERTIFIED-MODE (ENGINE_PROGRAM §1.7 / B2): every run of the Proof profile that publishes `unsat` from the
capability must replay `verified` through `ny proof-check` on its own `<results>.proof.jsonl`; the replay command and
its pass rule are pre-registered in plan.json (`certified_mode`).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seallib as S  # noqa: E402
from seallib import L, A  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
PLAN_SCHEMA = "seal-keyless-moat-plan/v1"
TAIL_REASONS = ("envelope:ops", "admit:unknown:")
# review(prepared-root-route): per capability, the census `spec` reads on which the lane's OWN property gates refuse
# before any work, each pinned by a named Rust test (ny-cli, `prepared_root_route_tests.rs`):
#   refused_spec_reads -- the router reads a property up to 16 MiB (spec_shape::SPEC_READ_MAX_BYTES) and the route
#     refuses one above its 1 MiB staged ceiling (original_property_census::MAX_STAGED_PROPERTY_BYTES) by the file
#     length, before any parse: `prepared_root_property_ceiling_is_below_the_router_spec_read_ceiling`;
#   served_archetypes_contain_the_lane -- every property the route accepts is one the router reads as robustness, so a
#     known archetype outside the served set is a lane refusal:
#     `prepared_root_property_admission_is_inside_the_routers_robustness_archetype`.
LANE_PROPERTY_GATES = {
    "prepared-root": {
        "refused_spec_reads": ("unknown:spec>16MiB",),
        "served_archetypes_contain_the_lane": True,
        "tests": ("prepared_root_property_ceiling_is_below_the_router_spec_read_ceiling",
                  "prepared_root_property_admission_is_inside_the_routers_robustness_archetype"),
    },
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lane_refuses_spec(cell: dict, gates: dict, served: tuple) -> str | None:
    """Why the capability's OWN property gates refuse this row's property (`None`: they may not), from the census
    `spec` cell (the router's start-time read: an archetype token and its shape, or `unknown:<why>`, or `-`)."""
    read = (cell.get("spec") or "-").split(" ", 1)[0]
    if read in gates.get("refused_spec_reads", ()):
        return "spec_read_refused"
    if gates.get("served_archetypes_contain_the_lane") and served and read in S.SPEC_TOKENS and read not in served:
        return "archetype_not_served"
    return None


def tail_only(cell: dict, gates: dict | None = None, served: tuple = ()) -> bool:
    """A census refusal whose every reason is in TAIL_REASONS (and at least one reason), on a row the scored entry
    point does not hand to the relational route first (that route returns before any other lane runs), and whose
    property the capability's own gates (`LANE_PROPERTY_GATES`) do not provably refuse."""
    if cell.get("facts") == "unknown:relational-route":
        return False
    parts = [p for p in cell["reasons"].split(",") if p and p != "-"]
    if not (bool(parts) and all(any(p.startswith(t) for t in TAIL_REASONS) for p in parts)):
        return False
    return lane_refuses_spec(cell, gates or {}, served) is None


def per_category(entries: dict) -> dict:
    cats: collections.Counter = collections.Counter()
    for e in entries.values():
        for r in e["rows"]:
            y, cat, _ = r.split("/", 2)
            cats[f"{y}/{cat}"] += 1
    return dict(sorted(cats.items()))


def keyless_derived(cap_id: str, years: list[int]) -> "S.Derived":
    """What s8_rows reads for a default-on capability: no armed key (no lever receipt to require), the defect
    alarms on every row, both years; categories come from the census (`admitted`). No truly-SAT-row alarm
    (`sat_alarms`, #certified-moat: an m0 tier's "the capability CLOSED this" marker): a keyless capability names
    none, so every row carries exactly the defect alarms, as before the field existed (merge eng16)."""
    return S.Derived(cap_id=cap_id, arm={}, keys={}, need=[], markers=[], arrival_markers=[], flight=[],
                     alarms=list(A.DEFECT_ALARMS), sat_alarms=[], categories=[], scope=[], years=list(years),
                     routable=True)


def cmd_plan(a) -> int:
    out = os.path.abspath(a.out)
    if os.path.exists(os.path.join(out, "plan.json")):
        raise S.SealError(f"{out}/plan.json exists: a plan is pre-registered once")
    reg = S.load_registry(a.registry)
    cap = S.get_capability(reg, a.cap)
    live = [k.get("path") for k in cap.get("keys") or [] if not k.get("retired")]
    if live:
        raise S.SealError(f"{a.cap} has live key(s) {live}: it is armed by an overlay; use `seal.py plan`")
    if not cap.get("routable"):
        raise S.SealError(f"{a.cap} is not routable: the census has no decision for it")
    pin = os.path.abspath(a.pin)
    pin_sha = sha256_file(pin)
    bid = S.build_id(pin)
    if bid["dirty"] != "false":
        raise S.SealError(f"pin {pin} is dirty ({bid['line'][:160]}): a plan binds committed bytes")
    commit = bid["commit"]
    S.git(a.repo, "cat-file", "-e", f"{commit}^{{commit}}")
    cdir = os.path.abspath(a.census_dir)
    c = S.parse_census(os.path.join(cdir, "census.tsv"))
    truth = {2026: a.truth_2026, 2025: a.truth_2025}
    os.makedirs(out, exist_ok=True)
    cfg_src = os.path.join(out, "configs", "src")
    cfg_tree = S.extract_tree(a.repo, commit, "configs", cfg_src)
    _, entries = S.census_corpus(a.ny_root, cfg_src, truth)
    corpus = S.corpus_text(entries)
    corpus_sha = S.sha256_text(corpus)
    problems = S.census_binding_problems(c, pin_sha, corpus_sha, a.cap)
    if problems:
        raise S.SealError(f"the census does not bind to this plan: {'; '.join(problems)}")
    admitted = S.census_admitted(c, a.cap, entries)
    if not admitted:
        raise S.SealError(f"the census admits no moat row for {a.cap}: an empty moat is not a moat")
    gates = LANE_PROPERTY_GATES.get(a.cap, {})
    served = tuple((cap["routable"].get("precondition") or {}).get("specs") or ())
    tail = {k: e for k, e in entries.items()
            if k not in admitted and tail_only(c["rows"][e["row_id"]][a.cap], gates, served)}
    # the rows the census refused only for tail reasons that the lane's own property gates refuse (left out of S8T)
    gated: collections.Counter = collections.Counter()
    for k, e in entries.items():
        cell = c["rows"][e["row_id"]][a.cap]
        if k not in admitted and k not in tail and tail_only(cell):
            gated[lane_refuses_spec(cell, gates, served) or "?"] += len(e["rows"])
    d = keyless_derived(a.cap, list(S.BOTH_YEARS))
    cfg = {"arm": {"dir": cfg_src}}
    specs = {}
    stages = {}
    for st, keys in (("S8", set(admitted)), ("S8T", set(tail))):
        if not keys:
            stages[st] = {"rows": 0, "runs": 0, "ceiling_h": 0.0, "spec": None}
            continue
        rows = S.s8_rows(d, cfg, a.ny_root, truth, a.reps_c, a.reps_p, keys)
        for r in rows:
            r["group"] = f"seal/{st}/{r['year']}"
            if st == "S8T":
                r["tag"] = "S8T_" + r["tag"][len("S8_"):]
        p = os.path.join(out, "specs", f"{st}.tsv")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        L.write_spec(p, rows)
        errs = L.validate_spec(L.read_spec(p), check_files=True)
        if errs:
            raise S.SealError(f"{st} spec invalid: {errs[:5]}")
        specs[st] = os.path.relpath(p, out)
        runs = sum(int(r["reps_c"]) + int(r["reps_p"]) for r in rows)
        ceiling = sum(int(r["budget"]) * (int(r["reps_c"]) + int(r["reps_p"])) for r in rows) / 3600.0
        cats = collections.Counter(f"{r['year']}/{r['category']}" for r in rows)
        stages[st] = {"rows": len(rows), "runs": runs, "ceiling_h": round(ceiling, 2), "spec": specs[st],
                      "spec_sha256": L.sha256_file(p), "per_category": dict(sorted(cats.items())),
                      "truth": dict(collections.Counter(r["truth"] for r in rows)),
                      "bank": dict(collections.Counter(r["bank"] for r in rows))}
    counts = S.census_counts(c, entries).get(a.cap, {})
    plan = {
        "schema": PLAN_SCHEMA, "capability": a.cap, "created": S.now(), "out": out,
        "registry": {"path": os.path.relpath(os.path.abspath(a.registry), REPO), "sha256": L.sha256_file(a.registry),
                     "entry_sha256": S.sha256_text(S.canonical(cap))},
        "pin": {"path": pin, "sha256": pin_sha, "commit": commit, "features": bid["features"], "build_id": bid["line"]},
        "configs": {"source": f"git:{commit}:configs", "tree": cfg_tree, "src": os.path.relpath(cfg_src, out),
                    "overlay": "none: the capability is default-on in the shipped code (no key arms it)"},
        "census": {"dir": cdir, "tsv_sha256": c["sha256"], "census_sha256": c["census_sha256"],
                   "corpus_sha256": corpus_sha, "binding": c["binding"], "instances": len(entries),
                   "moat_rows": sum(len(e["rows"]) for e in entries.values()),
                   "admit_instances": len(admitted), "admit_moat_rows": sum(len(e["rows"]) for e in admitted.values()),
                   "router_instances": counts.get("router"), "admit_per_category": per_category(admitted),
                   "tail_instances": len(tail), "tail_moat_rows": sum(len(e["rows"]) for e in tail.values()),
                   "tail_per_category": per_category(tail), "tail_reasons": list(TAIL_REASONS),
                   "tail_lane_gates": {"gates": gates, "served": list(served),
                                       "excluded_moat_rows": dict(sorted(gated.items()))}},
        "reps": {"competition": a.reps_c, "proof": a.reps_p},
        "stages": stages,
        "judge": {
            "per_run": "m0 judge_run (scripts/m0): truth contradiction, bank flip, defect alarm = EMERGENCY; wrong "
                       "instance / preset / profile or no Result line = VOID (never a pass)",
            "s8": "0 flips and 0 truth contradictions across both profiles; a Proof verdict reproduced in >= 2 of 3 "
                  "Competition reps (PROFILE-DECIDED-SPLIT otherwise = EMERGENCY); a lost banked verdict = PARTIAL; "
                  "cross-tag INSTANCE-FLIP over S8 + S8T = EMERGENCY",
            "engagement": "per row, the flight event `prepared_root` (method) of every run: Ran with status "
                          "RootProven / TreeProven / Open, or Skipped with the admission's reason; the stage is VOID "
                          "if no run of it shows status RootProven, TreeProven or Open",
            "stop": "any EMERGENCY stops everything (CLAUDE.md: a banked verdict that flips is a soundness emergency)",
        },
        "certified_mode": {
            "rule": "every Proof-profile run whose Result is unsat AND whose flight `prepared_root` event reports "
                    "RootProven or TreeProven (the route published) must replay through the offline checker on its "
                    "own bundle: PROOF_CHECK_DONE rc=0 verdict=verified; any other checker verdict on such a run is "
                    "an EMERGENCY (unconfirmed/invalid/malformed included: the route claims B2 holds)",
            "command": "<pin> proof-check <results>.proof.jsonl --onnx <onnx> --vnnlib <vnnlib>",
            "match": "^PROOF_CHECK_DONE rc=0 verdict=verified ",
        },
        "files": {v: stages[k]["spec_sha256"] for k, v in specs.items()},
    }
    text = json.dumps(plan, indent=1, sort_keys=True, default=str) + "\n"
    S.write_new(os.path.join(out, "plan.json"), text)
    S.write_new(os.path.join(out, "plan.json.sha256"), S.sha256_text(text) + "\n")
    print(f"=== keyless moat plan {a.cap} ({out}) census_sha256={c['census_sha256'][:16]} pin={pin_sha[:16]}")
    for st, v in stages.items():
        print(f"  stage {st} rows={v['rows']} runs={v['runs']} ceiling={v['ceiling_h']}h")
    print(f"KEYLESS_PLAN_DONE rc=0 capability={a.cap} admit={len(admitted)}/{plan['census']['admit_moat_rows']} "
          f"tail={len(tail)}/{plan['census']['tail_moat_rows']} s8_rows={stages['S8']['rows']} "
          f"s8t_rows={stages['S8T']['rows']} out={out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--cap", required=True)
    p.add_argument("--census-dir", required=True)
    p.add_argument("--pin", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--registry", default=os.path.join(REPO, "docs", "engine_program_capabilities.yaml"))
    p.add_argument("--repo", default=REPO)
    p.add_argument("--ny-root", default="<home>/ny")
    p.add_argument("--truth-2026", default=os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt"))
    p.add_argument("--truth-2025", default=os.path.expanduser("~/vnncomp2025_results/SCORING-ZERO-TOL/results.txt"))
    p.add_argument("--reps-c", type=int, default=S.S8_MIN_REPS_C)
    p.add_argument("--reps-p", type=int, default=S.S8_MIN_REPS_P)
    a = ap.parse_args(argv)
    if a.reps_c < S.S8_MIN_REPS_C or a.reps_p < S.S8_MIN_REPS_P:
        raise S.SealError(f"reps (competition, proof) ({a.reps_c}, {a.reps_p}) below the S8 floor "
                          f"({S.S8_MIN_REPS_C}, {S.S8_MIN_REPS_P})")
    try:
        return cmd_plan(a)
    except (S.SealError, L.M0Error, subprocess.CalledProcessError) as e:
        print(f"KEYLESS_PLAN_DONE rc=2 error={e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
