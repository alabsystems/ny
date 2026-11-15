#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""M0 item 4: the three PRE-REGISTERED zero-build A/Bs (ENGINE_PROGRAM section 2, M0; judge_impact F1/F3/F4).

Emits one configs copy per arm (typed yaml keys only, verified overlays) and a runner spec (kind AB, 3
Competition reps per arm and row, 0 Proof reps: an A/B measures what ships). Each arm row REQUIRES its own
observable (VOID otherwise), and every row carries the bank and the field truth, so a flip or a contradiction
is an EMERGENCY exactly as in the moat. The decision rule for each A/B is in scripts/m0/README.md; this tool
only makes the arms exact.

  (a) tier share   ctl   = margin_row.frozen_dual_bab: true                       (the W10B FDB arm)
                   share = ctl + margin_row.frozen_dual_bab_secs: 1e9             (capped at the 0.75 share)
  (b) certify-once ctl   = the shipped yaml
                   rco   = margin_row.alpha_opt: false + margin_row.root_certify_once: true
                           (certify-once runs AFTER alpha_opt; arming it alone adds a phase, judge_impact F3)
  (c) upfront      ctl   = the shipped yaml (shared with (b): one control arm, rows = the union)
                   after = attack.pgd_order: after (attack.ny_pgd_order_compat deleted): the attack moves
                           behind BaB (ResolvedInitialPgdSchedule::Deferred); the upfront route reads
                           disabled_by_preset
                   cap1  = attack.upfront_attack_auto_cap_secs: 1 (the binding cap on these rows: the flight
                           record reads admission=auto_disjunction binding=auto_wall_cap, auto_wall_cap_secs=4)
                   rows add tiny 1831 and 5529 (field-sat, BANKED timeout: (c) can only gain them) and the
                   banked-sat rows t1513, c1515, c8452 (a lost sat there is the cost side of (c)).

Rows: FDB6 = cifar100 c3494, c9585; tiny t6230, t5652, t4863, t8853; + t2712 (positive control); 2026, 100 s.

    python3 scripts/m0/ab_arms.py --configs <home>/ny-wt-<pin>/configs --out <home>/ny/scratchpad/m0/ab_<pin>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m0lib as L  # noqa: E402

CIFAR, TINY = "cifar100_2024", "tinyimagenet_2024"
FDB6 = (("c3494", CIFAR, "CIFAR100_resnet_medium.onnx", "_prop_idx_3494_"),
        ("c9585", CIFAR, "CIFAR100_resnet_medium.onnx", "_prop_idx_9585_"),
        ("t6230", TINY, "TinyImageNet_resnet_medium.onnx", "_prop_idx_6230_"),
        ("t5652", TINY, "TinyImageNet_resnet_medium.onnx", "_prop_idx_5652_"),
        ("t4863", TINY, "TinyImageNet_resnet_medium.onnx", "_prop_idx_4863_"),
        ("t8853", TINY, "TinyImageNet_resnet_medium.onnx", "_prop_idx_8853_"),
        ("t2712", TINY, "TinyImageNet_resnet_medium.onnx", "_prop_idx_2712_"))
# field-SAT rows for (c): 1831/5529 as the program names them (both BANKED timeout today, so (c) can only gain
# them) plus the three banked-SAT rows of the two families, which are what (c) can actually lose.
SAT_ROWS = (("t1831", TINY, "TinyImageNet_resnet_medium.onnx", "_prop_idx_1831_"),
            ("t5529", TINY, "TinyImageNet_resnet_medium.onnx", "_prop_idx_5529_"),
            ("t1513", TINY, "TinyImageNet_resnet_medium.onnx", "_prop_idx_1513_"),
            ("c1515", CIFAR, "CIFAR100_resnet_large.onnx", "_prop_idx_1515_"),
            ("c8452", CIFAR, "CIFAR100_resnet_large.onnx", "_prop_idx_8452_"))

FD_NEED = r"\[frozen-dual\] armed=1 objectives="
# arm -> (A/B ids it serves, sets, deletes, need terms, flight checks, rows)
ARMS: dict[str, dict] = {
    "a_ctl": {"ab": "a", "sets": {"margin_row.frozen_dual_bab": True}, "deletes": [],
              "need": [FD_NEED], "flight": ["lever:NY_MARGIN_ROW_FROZEN_DUAL_BAB=true@config",
                                            "lever:NY_MARGIN_ROW_FROZEN_DUAL_BAB_SECS=0.0@default"],
              "rows": FDB6},
    "a_share": {"ab": "a", "sets": {"margin_row.frozen_dual_bab": True, "margin_row.frozen_dual_bab_secs": 1e9},
                "deletes": [], "need": [FD_NEED],
                "flight": ["lever:NY_MARGIN_ROW_FROZEN_DUAL_BAB=true@config",
                           "lever:NY_MARGIN_ROW_FROZEN_DUAL_BAB_SECS=1000000000.0@config"],
                "rows": FDB6},
    "bc_ctl": {"ab": "bc", "sets": {}, "deletes": [],
               "need": [r"\[alpha-opt\] (done|refuse)", r"!\[root-certify-once\] armed=1"],
               "flight": ["lever:NY_MARGIN_ROW_ROOT_CERTIFY_ONCE=false@*",
                          "event:upfront_attack_policy:master=contextual route=auto_disjunction "],
               "rows": FDB6 + SAT_ROWS},
    "b_rco": {"ab": "b", "sets": {"margin_row.alpha_opt": False, "margin_row.root_certify_once": True},
              "deletes": [], "need": [r"\[root-certify-once\] armed=1 iters=", r"!\[alpha-opt\] done iters="],
              "flight": ["lever:NY_MARGIN_ROW_ALPHA_OPT=false@config",
                         "lever:NY_MARGIN_ROW_ROOT_CERTIFY_ONCE=true@config"],
              "rows": FDB6},
    "c_after": {"ab": "c", "sets": {"attack.pgd_order": "after"}, "deletes": ["attack.ny_pgd_order_compat"],
                "need": [], "flight": ["event:upfront_attack_policy:master=contextual route=disabled_by_preset "],
                "rows": FDB6 + SAT_ROWS},
    "c_cap1": {"ab": "c", "sets": {"attack.upfront_attack_auto_cap_secs": 1}, "deletes": [],
               "need": [], "flight": ["lever:NY_UPFRONT_ATTACK_AUTO_CAP=1@config",
                                      "event:upfront_attack_policy:route=auto_disjunction .*auto_wall_cap_secs=1 "],
               "rows": FDB6 + SAT_ROWS},
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ab", default="a,b,c", help="subset of a,b,c")
    ap.add_argument("--ny-root", default="<home>/ny")
    ap.add_argument("--truth-2026", default=os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt"))
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    try:
        want = {x.strip() for x in a.ab.split(",") if x.strip()}
        if not want or want - {"a", "b", "c"}:
            raise L.M0Error(f"--ab {a.ab!r}: want a subset of a,b,c")
        configs, out = os.path.abspath(a.configs), os.path.abspath(a.out)
        if os.path.exists(out):
            raise L.M0Error(f"refusing: {out} already exists")
        truth = L.parse_results_txt(a.truth_2026, 2026)
        arms = {k: v for k, v in ARMS.items() if set(v["ab"]) & want}
        os.makedirs(out)
        inst_cache: dict[str, list] = {}
        rows, manifest = [], {"schema": "m0-ab/v1", "source_configs": configs, "source_git_head": L.git_head(configs),
                              "arms": {}}
        for name, arm in arms.items():
            dst = os.path.join(out, f"cfg_ab_{name}")
            L.copy_configs(configs, dst)
            files = []
            for cat in (CIFAR, TINY):
                rel = L.resolve_preset(dst, cat)
                if rel is None:
                    raise L.M0Error(f"{name}: no preset for {cat}")
                if arm["sets"] or arm["deletes"]:
                    path = os.path.join(dst, rel)
                    base = open(path, encoding="utf-8").read()
                    doc = L.strict_load(base) or {}
                    missing = [k for k in arm["deletes"] if not L.deep_has(doc, k)]
                    if missing:
                        raise L.M0Error(f"{name}: {rel} lacks {missing} (the arm would not be the pre-registered one)")
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(L.apply_overlay(base, arm["sets"], arm["deletes"], f"m0 ab {name}"))
                    files.append({"preset": rel, "sha256_after": L.sha256_file(path)})
                else:
                    files.append({"preset": rel, "unchanged": True})
            manifest["arms"][name] = {"configs_dir": dst, "sets": arm["sets"], "deletes": arm["deletes"], "files": files}
            for rowid, cat, onnx_sub, vnnlib_sub in arm["rows"]:
                if cat not in inst_cache:
                    inst_cache[cat] = L.load_instances(L.scored_instances_csv(a.ny_root, 2026, cat))
                inst = inst_cache[cat]
                r = L.find_row(inst, onnx_sub, vnnlib_sub, f"{name}/{rowid}")
                bank = L.align_bank(inst, L.load_bank(L.bank_csv(a.ny_root, 2026, cat)), f"bank {cat}")[r.index]
                t = L.align_truth(inst, truth.get(cat), f"truth {cat}")[r.index]
                rows.append(L.spec_row(
                    tag=f"AB_{name}_{rowid}", kind="AB", tier="frozen-dual" if name.startswith("a_") else "-",
                    year=2026, category=cat, onnx=r.onnx, vnnlib=r.vnnlib, budget=r.budget, configs_dir=dst,
                    preset=L.resolve_preset(dst, cat), truth=t.token() if t else "-", bank=bank, index=r.index,
                    reps_c=a.reps, reps_p=0, need=L.NEED_SEP.join(arm["need"]) if arm["need"] else "-",
                    alarm=L.MR_UNSAT if (t and t.token() == "sat") else "-",
                    flight=L.FLIGHT_SEP.join(arm["flight"]) if arm["flight"] else "-",
                    group=f"ab_{arm['ab']}/{name}", note=f"row={rowid}"))
        L.write_spec(os.path.join(out, "ab_spec.tsv"), rows)
        with open(os.path.join(out, "ab_arms.json"), "w") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)
        runs = sum(int(r["reps_c"]) for r in rows)
        print(f"AB_DONE rc=0 arms={','.join(arms)} rows={len(rows)} runs={runs} spec={os.path.join(out, 'ab_spec.tsv')}")
        return 0
    except L.M0Error as e:
        print(f"AB_REFUSED: {e}", file=sys.stderr)
        print("AB_DONE rc=2 rows=0")
        return 2


if __name__ == "__main__":
    sys.exit(main())
