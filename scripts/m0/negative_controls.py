#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""M0 item 2: tier-engaged sat NEGATIVE CONTROLS (ENGINE_PROGRAM section 2, M0).

For each named tier, copy a configs tree and edit the tier's category yaml(s) -- the file the preset resolver
actually loads (vnncomp26 shadows vnncomp25) -- so that:

  * the tier is armed through its typed yaml key(s) (never NY_*), and
  * every attack key is off: the masters `attack.pgd_order: skip`, `attack.upfront_attack: false`,
    `bab.phase_budget.vnncomp_post_bab_attack: false` + `post_bab_pgd_fraction: 0.0` (the tail goes to proof),
    zero upfront/disjunctive/extension PGD slices, and every independent falsification lane switched off.
    The table in m0lib.ATTACK_TABLE must classify EVERY field of the preset schema's AttackPreset (and the
    attack/PGD fields of PhaseBudgetPreset / InputSplitPreset), or this tool refuses.

and write a runner spec (m0-spec/v1) with one row per truly-SAT control row. Each row REQUIRES, for the run to
count at all: the tier's own engagement line (e.g. `[frozen-dual] armed=1 ... nodes>0`), the attack masters
confirmed off in the flight record, the exact preset and instance, and the `[proof-profile]` line. Any unsat
(or a tier alarm such as `verdict=all_closed` on a sat row) is an EMERGENCY. Every control so far ended at the
attack with 0 tier lines, which is why engagement is required rather than assumed.

No cargo, no ny binary, no box. Nothing is deleted; an existing output directory is refused.

    python3 scripts/m0/negative_controls.py --configs <home>/ny-wt-<pin>/configs \
        --out <home>/ny/scratchpad/m0/negctl_<pin> --tier frozen-dual --tier cgan-child-zono \
        --tier fwd-root-screen --tier bi-window-joint-refresh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m0lib as L  # noqa: E402

TAG_SHORT = {"frozen-dual": "fd", "cgan-child-zono": "cz", "fwd-root-screen": "frs",
             "bi-window-joint-refresh": "bwjr"}


def overlay_tier(tier: L.Tier, dst: str) -> list[dict]:
    """Edit every category yaml of the tier inside the copied tree `dst`; returns per-file records."""
    records = []
    for cat in tier.categories:
        rel = L.resolve_preset(dst, cat)
        if rel is None:
            raise L.M0Error(f"{tier.name}: no preset resolves for category {cat} under {dst}")
        path = os.path.join(dst, rel)
        base = open(path, encoding="utf-8").read()
        doc = L.strict_load(base) or {}
        sets, deletes, log = L.attack_off_edits(doc)
        overlap = set(sets) & set(tier.arm)
        if overlap:
            raise L.M0Error(f"{tier.name}: tier key(s) {sorted(overlap)} collide with the attack table")
        for k, v in tier.arm.items():
            sets[k] = v
            log.append(f"arm {k}: {L.fmt_yaml(v)} (was {L.deep_get(doc, k)!r}) -- tier {tier.name}")
        edited = L.apply_overlay(base, sets, deletes, f"m0 negctl {tier.name}")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(edited)
        records.append({"category": cat, "preset": rel, "sha256_before": hashlib.sha256(base.encode()).hexdigest(),
                        "sha256_after": L.sha256_file(path), "actions": log})
    return records


def control_rows(tier: L.Tier, dst: str, ny_root: str, truth26: dict, witnesses: dict[str, str],
                 reps_c: int, reps_p: int, records: list[dict]) -> list[dict]:
    preset_of = {r["category"]: r["preset"] for r in records}
    rows = []
    for rowid, cat, onnx_sub, vnnlib_sub in tier.rows:
        csv_path = L.scored_instances_csv(ny_root, 2026, cat)
        inst = L.load_instances(csv_path)
        r = L.find_row(inst, onnx_sub, vnnlib_sub, f"{tier.name}/{rowid}")
        if cat not in truth26:
            raise L.M0Error(f"{tier.name}/{rowid}: category {cat} absent from the 2026 field results")
        t = L.align_truth(inst, truth26[cat], f"{tier.name}/{rowid}")[r.index]
        if t is None or t.token() != "sat":
            raise L.M0Error(f"{tier.name}/{rowid}: not a field-SAT row (truth={t.token() if t else 'none'}); "
                            "a negative control must be truly SAT")
        bank = "-"
        bank_path = L.bank_csv(ny_root, 2026, cat)
        if os.path.isfile(bank_path):
            bank = L.align_bank(inst, L.load_bank(bank_path), f"bank {cat}")[r.index]
        alarm = list(tier.alarm)
        wnote = "-"
        if rowid in witnesses:
            if tier.class_alarm is None:
                raise L.M0Error(f"{tier.name}: --witness given but the tier has no per-class alarm")
            prop = L.parse_property(L.read_text(r.vnnlib))
            w = L.load_witness(witnesses[rowid], prop.in_name, prop.out_name)
            L.witness_in_box(prop, w.x)
            classes = L.witness_classes(prop, w.y)
            if not classes:
                raise L.M0Error(f"{tier.name}/{rowid}: the witness satisfies no single-atom relational clause")
            alarm += [tier.class_alarm.format(j=j) for j in classes]
            wnote = f"witness_classes={','.join(map(str, classes))} witness_sha256={w.sha256[:16]}"
        preset = preset_of[cat]
        flight = list(L.NEGCTL_FLIGHT) + list(tier.flight) + [
            "event:preset_base:^" + re.escape(os.path.join(dst, preset)) + "$"]
        rows.append(L.spec_row(
            tag=f"NEG_{TAG_SHORT[tier.name]}_{rowid}", kind="NEGCTL", tier=tier.name, year=2026, category=cat,
            onnx=r.onnx, vnnlib=r.vnnlib, budget=r.budget, configs_dir=dst, preset=preset, truth="sat",
            bank="-", index=r.index, reps_c=reps_c, reps_p=reps_p, need=L.NEED_SEP.join(tier.need),
            alarm=L.ALARM_SEP.join(alarm) if alarm else "-", flight=L.FLIGHT_SEP.join(flight),
            group=f"negctl/{tier.name}",
            note=f"banked={bank} field_fals_ok={len(t.fals_ok)} {wnote}"))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", required=True, help="source configs dir (the arming pin's worktree configs/)")
    ap.add_argument("--out", required=True, help="new output dir (refused if it exists)")
    ap.add_argument("--tier", action="append", required=True, choices=sorted(L.TIERS))
    ap.add_argument("--ny-root", default="<home>/ny", help="main tree (benchmarks/ and reports/ live there)")
    ap.add_argument("--schema", help="preset schema source (default <configs>/../crates/ny-cli/src/preset/mod.rs)")
    ap.add_argument("--truth-2026", default=os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt"))
    ap.add_argument("--witness", action="append", default=[], metavar="ROWID=PATH",
                    help="a sat witness for a classification row: adds a per-class 'never closed' alarm")
    ap.add_argument("--reps-competition", type=int, default=1)
    ap.add_argument("--reps-proof", type=int, default=1)
    a = ap.parse_args(argv)
    try:
        configs = os.path.abspath(a.configs)
        out = os.path.abspath(a.out)
        schema = a.schema or os.path.join(os.path.dirname(configs), "crates/ny-cli/src/preset/mod.rs")
        in_scope = L.check_attack_table(schema)
        if os.path.exists(out):
            raise L.M0Error(f"refusing: {out} already exists (this harness never deletes; pick a new --out)")
        witnesses = {}
        for w in a.witness:
            if "=" not in w:
                raise L.M0Error(f"--witness {w!r}: want ROWID=PATH")
            k, v = w.split("=", 1)
            witnesses[k] = os.path.abspath(v)
        truth26 = L.parse_results_txt(a.truth_2026, 2026)
        if not truth26:
            raise L.M0Error(f"{a.truth_2026}: no 2026 category parsed (a truth gate with nothing to compare is void)")
        os.makedirs(out)
        manifest = {"schema": "m0-negctl/v1", "source_configs": configs, "source_git_head": L.git_head(configs),
                    "preset_schema": schema, "preset_schema_sha256": L.sha256_file(schema),
                    "attack_keys_in_scope": len(in_scope), "tiers": {}}
        rows: list[dict] = []
        for name in a.tier:
            tier = L.TIERS[name]
            dst = os.path.join(out, f"cfg_negctl_{name}")
            L.copy_configs(configs, dst)
            recs = overlay_tier(tier, dst)
            rows += control_rows(tier, dst, a.ny_root, truth26, witnesses, a.reps_competition, a.reps_proof, recs)
            manifest["tiers"][name] = {"configs_dir": dst, "capability": tier.capability, "files": recs,
                                       "rows": [r["tag"] for r in rows if r["tier"] == name]}
        unused = set(witnesses) - {r["tag"].split("_", 2)[2] for r in rows}
        if unused:
            raise L.M0Error(f"--witness row ids match no control row: {sorted(unused)}")
        L.write_spec(os.path.join(out, "negctl_spec.tsv"), rows)
        with open(os.path.join(out, "negctl_arms.json"), "w") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)
        for name, t in manifest["tiers"].items():
            print(f"  tier {name}: {t['configs_dir']} rows={len(t['rows'])}")
            for f in t["files"]:
                print(f"    {f['preset']}: {len(f['actions'])} edits")
        print(f"NEGCTL_DONE rc=0 rows={len(rows)} spec={os.path.join(out, 'negctl_spec.tsv')}")
        return 0
    except L.M0Error as e:
        print(f"NEGCTL_REFUSED: {e}", file=sys.stderr)
        print("NEGCTL_DONE rc=2 rows=0")
        return 2


if __name__ == "__main__":
    sys.exit(main())
