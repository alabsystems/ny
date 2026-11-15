#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed selftests for scripts/costcensus/costcensus.py on synthetic fixtures (no ny binary, no box).

Prints one `test_<name> ... ok|FAILED` line per test and ends with
`COSTCENSUS_SELFTEST_DONE rc=<0|1> passed=<n> failed=<n>`.
"""

from __future__ import annotations

import copy
import io
import json
import os
import sys
import tempfile
import traceback
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import costcensus as cc  # noqa: E402

COMMIT = "a" * 40
SHA = "b" * 64


def phase(name, wall, enters=1, units=0, thread=None, self_s=None, unit="domains", inflight=1):
    return {"phase": name, "wall_s": wall, "thread_s": wall if thread is None else thread,
            "self_s": wall if self_s is None else self_s, "enters": enters, "reentries": 0, "units": units,
            "unit": unit, "max_inflight": inflight, "open": 0}


def cost(phases, edges=None, counters=None, level="coarse", schema=cc.SCHEMA):
    if edges is None:
        edges = [{"parent": "root", "child": p["phase"], "thread_s": p["thread_s"]} for p in phases]
    return {"schema": schema, "level": level, "elapsed_s": 1.0, "phases": phases, "edges": edges,
            "counters": counters or {}, "rss_samples": 3, "rss_peak_kb": 2048, "process_hwm_kb": 4096,
            "unbalanced": 0, "stack_overflow": 0}


def flight(category, wall, cost_rec, verdict="unsat", commit=COMMIT, ambient=None, partial=None):
    rec = {
        "schema_version": 3,
        "build_id": f"NY_BUILD_STAMP_V1 version=0.0.0 features=mip,cuda commit={commit} dirty=false stamp=x",
        "category": category,
        "budget_secs": 100,
        "ambient_env": {"OMP_NUM_THREADS": "1"} if ambient is None else ambient,
        "events": [{"method": "preset", "status": "ran", "at_secs": 0.1}],
    }
    if partial is not None:
        rec["record_status"] = {"status": "partial_hard_stop", "reason": "stop", "flushed_at_secs": partial}
        rec["events"].append({"method": "run_incomplete", "status": "incomplete", "reason": "stop",
                              "at_secs": partial})
    else:
        rec["events"].append({"method": "run_complete", "status": "complete", "reason": verdict, "at_secs": wall})
    if cost_rec is not None:
        rec["phase_cost"] = cost_rec
    return rec


def row(cat, idx, budget=100, bank="unsat", stratum="solved"):
    return {"tag": f"{cat}__i{idx}", "year": 2026, "category": cat, "index": idx,
            "onnx": f"/b/{cat}/onnx/m.onnx", "vnnlib": f"/b/{cat}/vnnlib/p{idx}.vnnlib", "budget": budget,
            "bank_verdict": bank, "bank_wall": 10.0, "stratum": stratum}


def log(r, verdict="unsat", cost_line=True, result=True, vnnlib=None, profile=True):
    lines = [f"Running VNN-COMP instance: category='{r['category']}' onnx='{r['onnx']}' "
             f"vnnlib='{vnnlib or r['vnnlib']}' results='x' timeout={r['budget']}s"]
    if profile:
        lines.append(cc.PROFILE_LINE)
    lines.append(f"Loading preset: /cfg/vnncomp26/{r['category']}.yaml")
    if cost_line:
        lines.append("[phase-cost] level=coarse elapsed_s=1.00 phases=3 enters=9 top=x domains_explored=1 "
                     "domains_verified=1 rss_peak_mb=2")
    if result:
        lines.append(f"Result: {verdict}")
    return "\n".join(lines) + "\n"


def judge_ok(r, wall, cost_rec, verdict="unsat"):
    out = cc.judge_run(r, flight(r["category"], wall, cost_rec, verdict), log(r, verdict), verdict + "\n", COMMIT)
    assert out["status"] == "OK", out["why"]
    return out


def bab_cost(wall, select, bound, extra=None, verified=10, explored=40):
    ph = [phase("bab_loop", 0.9 * wall), phase("branch_select", select * wall, enters=7, units=70, unit="domains"),
          phase("child_bound", bound * wall, enters=7, units=140), phase("root_bounds", 0.05 * wall)]
    edges = [{"parent": "root", "child": "bab_loop", "thread_s": 0.9 * wall},
             {"parent": "root", "child": "root_bounds", "thread_s": 0.05 * wall},
             {"parent": "bab_loop", "child": "branch_select", "thread_s": select * wall},
             {"parent": "bab_loop", "child": "child_bound", "thread_s": bound * wall}]
    for p in extra or []:
        ph.append(p)
        edges.append({"parent": "bab_loop", "child": p["phase"], "thread_s": p["thread_s"]})
    return cost(ph, edges, {"domains_explored": explored, "domains_verified": verified})


# ------------------------------------------------------------------------------------------------------------------


def test_select_rows_is_deterministic_and_stratified():
    # sat rows are fast (an attack) so the unsat-only median differs from the all-solved median
    rows = [dict(index=i, onnx="o", vnnlib=f"v{i}", budget=100,
                 bank_verdict=["unsat", "timeout", "sat", "unsat", "unknown"][i % 5],
                 bank_wall=1.0 if i % 5 == 2 else float(10 * i))
            for i in range(20)]
    a = cc.select_rows(rows, 2, 120, 1800)
    b = cc.select_rows(list(reversed(rows)), 2, 120, 1800)
    assert [r["index"] for r in a] == [r["index"] for r in b], "pick must not depend on input order"
    assert [r["stratum"] for r in a] == ["solved", "open"], a
    solved = sorted((r for r in rows if r["bank_verdict"] == "unsat"), key=lambda r: (r["bank_wall"], r["index"]))
    assert a[0]["index"] == solved[(len(solved) - 1) // 2]["index"] == 8, "solved pick is the median banked unsat wall"
    sat_only = [dict(r, bank_verdict="sat") if r["bank_verdict"] == "unsat" else r for r in rows]
    s = cc.select_rows(sat_only, 2, 120, 1800)
    assert s[0]["bank_verdict"] == "sat" and s[0]["stratum"] == "solved", "no unsat: the solved stratum is a sat"
    assert a[1]["index"] == 1, "open pick is the lowest-index open row of the smallest budget"


def test_select_rows_respects_budget_caps_and_one_row_for_heavy_categories():
    heavy = [dict(index=i, onnx="o", vnnlib="v", budget=900, bank_verdict="unsat" if i else "timeout",
                  bank_wall=5.0 * i) for i in range(6)]
    picks = cc.select_rows(heavy, 2, 120, 1800)
    assert len(picks) == 1 and picks[0]["stratum"] == "solved", picks
    assert cc.select_rows([dict(r, budget=3600) for r in heavy], 2, 120, 1800) == [], "over max-budget: none"
    only_open = [dict(index=i, onnx="o", vnnlib="v", budget=20, bank_verdict="timeout", bank_wall=None)
                 for i in range(4)]
    picks = cc.select_rows(only_open, 2, 120, 1800)
    assert [p["stratum"] for p in picks] == ["open", "fill"] and [p["index"] for p in picks] == [0, 1], picks


def test_spec_roundtrip_and_refuses_a_foreign_spec():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plan.tsv")
        spec = [row("cat_a", 3), row("cat_b", 7, budget=300, bank="timeout", stratum="open")]
        cc.write_spec(p, spec)
        back = cc.read_spec(p)
        assert [(r["tag"], r["budget"], r["bank_verdict"]) for r in back] == \
               [("cat_a__i3", 100, "unsat"), ("cat_b__i7", 300, "timeout")]
        with open(p, "w") as fh:
            fh.write("tag\tyear\n")
        try:
            cc.read_spec(p)
        except cc.CensusError:
            pass
        else:
            raise AssertionError("a foreign spec must be refused")
        cc.write_spec(p, spec + [row("cat_a", 3)])
        try:
            cc.read_spec(p)
        except cc.CensusError:
            pass
        else:
            raise AssertionError("duplicate tags must be refused")


def test_every_void_reason_fires_and_ok_needs_all_of_them():
    r = row("cat_a", 1)
    good = bab_cost(10.0, 0.1, 0.5)
    judge_ok(r, 10.0, good)

    def why(fl, lg=None, res="unsat\n"):
        out = cc.judge_run(r, fl, log(r) if lg is None else lg, res, COMMIT)
        assert out["status"] == "VOID", out
        return " ".join(out["why"])

    assert "no flight record" in why(None)
    assert "no phase_cost" in why(flight("cat_a", 10.0, None))
    assert "schema" in why(flight("cat_a", 10.0, cost([phase("load", 1.0)], schema="other/v9")))
    assert "level=off" in why(flight("cat_a", 10.0, cost([phase("load", 1.0)], level="off")))
    assert "build_id" in why(flight("cat_a", 10.0, good, commit="c" * 40))
    dirty = flight("cat_a", 10.0, good)
    dirty["build_id"] = dirty["build_id"].replace("dirty=false", "dirty=true")
    assert "dirty" in why(dirty)
    assert "NY_" in why(flight("cat_a", 10.0, good, ambient={"OMP_NUM_THREADS": "1", "NY_FOO": "1"}))
    assert "OMP_NUM_THREADS=1" in why(flight("cat_a", 10.0, good, ambient={}))
    assert "[phase-cost] level=" in why(flight("cat_a", 10.0, good), log(r, cost_line=False))
    assert "Result:" in why(flight("cat_a", 10.0, good), log(r, result=False))
    assert "names another row" in why(flight("cat_a", 10.0, good), log(r, vnnlib="/b/cat_a/vnnlib/p999.vnnlib"))
    assert "proof-profile" in why(flight("cat_a", 10.0, good), log(r, profile=False))
    assert "category" in why(flight("cat_b", 10.0, good))
    assert "results file" in why(flight("cat_a", 10.0, good), res="timeout\n")
    assert "no run log" in " ".join(cc.judge_run(r, flight("cat_a", 10.0, good), None, "unsat\n", COMMIT)["why"])


def test_a_bank_flip_is_an_emergency_never_averaged_in():
    r = row("cat_a", 1, bank="unsat")
    out = cc.judge_run(r, flight("cat_a", 10.0, bab_cost(10.0, 0.1, 0.5), "sat"), log(r, "sat"), "sat\n", COMMIT)
    assert out["status"] == "EMERGENCY" and "BANK-FLIP" in out["why"][0], out
    agg = cc.aggregate([out])
    assert agg["ok"] == [] and agg["per_category"] == {}
    md = cc.render(agg, [out], {})
    assert "SOUNDNESS EMERGENCY" in md and "1 EMERGENCY" in md
    # a timeout/unknown on a banked row is a loss, not an emergency
    t = cc.judge_run(r, flight("cat_a", 10.0, bab_cost(10.0, 0.1, 0.5), "timeout"), log(r, "timeout"), "timeout\n",
                     COMMIT)
    assert t["status"] == "OK", t


def test_a_row_the_bank_holds_open_that_the_run_decided_is_surfaced_never_silent():
    # review(cost-census): census3 decided smart_turn i0/i1 `sat` (banked unknown: the x86-parity demotion class)
    # and cifar100 i1 `unsat` (banked timeout); only the cifar row was mentioned, by hand. The judge must surface
    # every such row itself, OK or not, and a decided bank row never carries the marker.
    sat_row = row("smart", 0, bank="unknown", stratum="open")
    out = cc.judge_run(sat_row, flight("smart", 52.0, bab_cost(52.0, 0.1, 0.5), "sat"), log(sat_row, "sat"), "sat\n",
                       COMMIT)
    assert out["status"] == "OK" and out["new_decision"] == "sat", out
    uns_row = row("cifar", 1, bank="timeout", stratum="open")
    out2 = judge_ok(uns_row, 24.4, bab_cost(24.4, 0.1, 0.5))
    assert out2["new_decision"] == "unsat", out2
    same = judge_ok(row("acas", 0), 10.0, bab_cost(10.0, 0.1, 0.5))
    assert "new_decision" not in same, same
    still_open = cc.judge_run(uns_row, flight("cifar", 100.0, bab_cost(100.0, 0.1, 0.5), "timeout"),
                              log(uns_row, "timeout"), "timeout\n", COMMIT)
    assert "new_decision" not in still_open, still_open
    md = cc.render(cc.aggregate([out, out2, same]), [out, out2, same], {})
    assert "DECIDED (2)" in md and "`smart__i0` banked unknown -> sat" in md and "x86" in md, md[:1500]
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "res"))
        os.makedirs(os.path.join(d, "raw"))
        cc.write_spec(os.path.join(d, "plan.tsv"), [sat_row])
        with open(os.path.join(d, "res", "smart__i0.txt"), "w") as fh:
            fh.write("sat\n")
        with open(os.path.join(d, "res", "smart__i0.txt.flight.json"), "w") as fh:
            json.dump(flight("smart", 52.0, bab_cost(52.0, 0.1, 0.5), "sat"), fh)
        with open(os.path.join(d, "raw", "smart__i0.log"), "w") as fh:
            fh.write(log(sat_row, "sat"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cc.main(["aggregate", "--out-dir", d, "--expect-commit", COMMIT])
        assert rc == 0 and "newly_decided=1" in buf.getvalue(), buf.getvalue()


def test_a_hard_stopped_row_uses_its_flush_wall_and_is_marked_partial():
    r = row("cat_a", 1, bank="timeout", stratum="open")
    fl = flight("cat_a", 0.0, bab_cost(110.0, 0.2, 0.6), partial=110.0)
    out = cc.judge_run(r, fl, log(r, "timeout"), "timeout\n", COMMIT)
    assert out["status"] == "OK" and out["partial"] and abs(out["wall_s"] - 110.0) < 1e-9, out
    md = cc.render(cc.aggregate([out]), [out], {})
    assert "(partial)" in md
    # the envelope's flushed_at is authoritative even when the terminal event is missing
    fl["events"] = [e for e in fl["events"] if e["method"] != "run_incomplete"]
    out = cc.judge_run(r, fl, log(r, "timeout"), "timeout\n", COMMIT)
    assert out["status"] == "OK" and abs(out["wall_s"] - 110.0) < 1e-9, out


def test_shares_units_and_coverage_are_derived_exactly():
    r = row("cat_a", 1)
    c = bab_cost(20.0, 0.25, 0.5, verified=4, explored=10)
    out = judge_ok(r, 20.0, c)
    m = cc.row_metrics(out)
    assert abs(m["share"]["branch_select"] - 0.25) < 1e-12 and abs(m["share"]["child_bound"] - 0.5) < 1e-12
    # umbrellas are not coverage; their direct children are
    assert m["top"] == ["branch_select", "child_bound", "root_bounds"], m["top"]
    assert abs(m["coverage"] - 0.8) < 1e-12, m["coverage"]
    assert abs(m["share"][cc.UNATTRIBUTED] - 0.2) < 1e-12
    v = cost([phase("verify", 18.0), phase("mip", 2.0)],
             [{"parent": "root", "child": "verify", "thread_s": 18.0},
              {"parent": "verify", "child": "mip", "thread_s": 2.0}])
    mv = cc.row_metrics(judge_ok(r, 20.0, v))
    assert mv["top"] == ["mip"] and abs(mv["share"][cc.UNATTRIBUTED] - 0.9) < 1e-12, mv
    assert abs(m["s_per_verified"] - 5.0) < 1e-12 and abs(m["s_per_explored"] - 2.0) < 1e-12
    over = cost([phase("attack", 50.0)])
    assert cc.row_metrics(judge_ok(r, 20.0, over))["share"]["attack"] == 1.0, "a share is clamped at 1"


def _planted_world():
    """Five ordinary categories (select ~0.10 of the row, bound ~0.55) and one kFSB-shaped category (select 0.66,
    bound 0.20), plus a VOID row that would dominate everything if it were counted."""
    runs = []
    for k, cat in enumerate(["acas", "cifar", "cora", "nn4sys", "tll"]):
        for i in range(2):
            r = row(cat, i)
            runs.append(judge_ok(r, 30.0 + k, bab_cost(30.0 + k, 0.10 + 0.01 * i, 0.55)))
    for i in range(2):
        r = row("yolo", i, budget=300)
        sim = phase("branch_sim", 0.5 * 80.0, enters=12, units=480, unit="simulations")
        runs.append(judge_ok(r, 80.0, bab_cost(80.0, 0.66, 0.20, extra=[sim])))
    bad = row("tll", 9)
    void = cc.judge_run(bad, flight("tll", 30.0, bab_cost(30.0, 0.99, 0.0), commit="d" * 40), log(bad), "unsat\n",
                        COMMIT)
    assert void["status"] == "VOID"
    runs.append(void)
    return runs


def test_a_planted_kfsb_shape_ranks_first_with_its_evidence():
    runs = _planted_world()
    agg = cc.aggregate(runs)
    top = agg["findings"][0]
    assert (top["category"], top["phase"], top["kind"]) == ("yolo", "branch_select", "share"), agg["findings"][:3]
    assert abs(top["share"] - 0.66) < 1e-9 and abs(top["median"] - 0.105) < 0.02, top
    assert top["evidence"][0]["tag"].startswith("yolo__i") and top["evidence"][0]["enters"] == 7
    kinds = {(f["category"], f["phase"], f["kind"]) for f in agg["findings"]}
    assert ("yolo", "branch_select", "select>=bound") in kinds
    assert ("yolo", "branch_sim", "share") in kinds, kinds
    assert not any(f["phase"] in cc.UMBRELLAS for f in agg["findings"]), "an umbrella is never a finding"
    assert not any(f["category"] != "yolo" for f in agg["findings"]), agg["findings"]
    # the VOID tll row (select 0.99) did not move tll's mean
    assert abs(agg["per_category"]["tll"]["mean_share"]["branch_select"] - 0.105) < 1e-9
    assert agg["per_category"]["tll"]["rows"] == ["tll__i0", "tll__i1"]


def test_report_names_the_rule_the_evidence_and_every_void_reason():
    runs = _planted_world()
    md = cc.render(cc.aggregate(runs), runs, {"pin": "/pins/x", "commit": COMMIT})
    assert "## Ranked bottlenecks" in md and "| 1 | yolo | branch_select | share |" in md, md[:2000]
    assert "`yolo__i0`" in md and "s/simulation" in md
    assert "build_id does not name commit" in md and "| VOID |" in md
    assert "**median**" in md and f"`{COMMIT}`" in md


def test_aggregate_cli_end_to_end_and_its_rc():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "res"))
        os.makedirs(os.path.join(d, "raw"))
        spec = [row("acas", 0), row("acas", 1, bank="timeout", stratum="open"), row("yolo", 0, budget=300)]
        cc.write_spec(os.path.join(d, "plan.tsv"), spec)

        def put(r, fl, lg, res):
            with open(os.path.join(d, "res", f"{r['tag']}.txt"), "w") as fh:
                fh.write(res)
            with open(os.path.join(d, "res", f"{r['tag']}.txt.flight.json"), "w") as fh:
                json.dump(fl, fh)
            with open(os.path.join(d, "raw", f"{r['tag']}.log"), "w") as fh:
                fh.write(lg)

        put(spec[0], flight("acas", 10.0, bab_cost(10.0, 0.1, 0.5)), log(spec[0]), "unsat\n")
        put(spec[1], flight("acas", 116.0, bab_cost(116.0, 0.1, 0.6), "timeout"), log(spec[1], "timeout"),
            "timeout\n")
        put(spec[2], flight("yolo", 80.0, bab_cost(80.0, 0.66, 0.2)), log(spec[2]), "unsat\n")
        rep, js = os.path.join(d, "r.md"), os.path.join(d, "a.json")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cc.main(["aggregate", "--out-dir", d, "--expect-commit", COMMIT, "--report", rep, "--json", js])
        assert rc == 0 and "COSTCENSUS_AGG_DONE rc=0 rows=3 ok=3 void=0 emergency=0" in buf.getvalue(), buf.getvalue()
        data = json.load(open(js))
        assert data["schema"] == "costcensus-aggregate/v1" and data["findings"][0]["category"] == "yolo"
        # a missing flight record makes the run VOID and the rc 5 (never a silent pass)
        os.rename(os.path.join(d, "res", "acas__i0.txt.flight.json"), os.path.join(d, "res", "moved.json"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cc.main(["aggregate", "--out-dir", d, "--expect-commit", COMMIT, "--report", rep])
        assert rc == 5 and "void=1" in buf.getvalue(), buf.getvalue()
        # an emergency outranks everything: rc 3
        put(spec[0], flight("acas", 10.0, bab_cost(10.0, 0.1, 0.5), "sat"), log(spec[0], "sat"), "sat\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cc.main(["aggregate", "--out-dir", d, "--expect-commit", COMMIT, "--report", rep])
        assert rc == 3 and "emergency=1" in buf.getvalue(), buf.getvalue()
        # a plan that is not a costcensus spec is refused (rc 2), not aggregated as empty
        with open(os.path.join(d, "plan.tsv"), "w") as fh:
            fh.write("garbage\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cc.main(["aggregate", "--out-dir", d, "--expect-commit", COMMIT])
        assert rc == 2 and "refused" in buf.getvalue(), buf.getvalue()


def test_an_umbrella_phase_is_never_a_finding_even_when_anomalous():
    runs = []
    for cat in ["a", "b", "c", "d"]:
        c = cost([phase("bab_loop", 3.0), phase("child_bound", 2.0)])
        runs.append(judge_ok(row(cat, 0), 10.0, c))
    c = cost([phase("bab_loop", 9.8), phase("child_bound", 2.0)])
    runs.append(judge_ok(row("e", 0), 10.0, c))
    agg = cc.aggregate(runs)
    assert agg["per_category"]["e"]["mean_share"]["bab_loop"] > 0.9
    assert not agg["findings"], agg["findings"]


def test_anomaly_needs_ratio_absolute_and_floor_together():
    runs = []
    # every category spends ~0.30 in attack: nothing anomalous even though 0.30 is large
    for cat in ["a", "b", "c", "d"]:
        r = row(cat, 0)
        c = bab_cost(10.0, 0.1, 0.5, extra=[phase("attack", 3.0)])
        runs.append(judge_ok(r, 10.0, c))
    agg = cc.aggregate(runs)
    assert not [f for f in agg["findings"] if f["phase"] == "attack"], agg["findings"]
    # one category at 0.12 vs a 0.0 median: under ANOM_ABS, not flagged
    r = row("e", 0)
    runs.append(judge_ok(r, 10.0, bab_cost(10.0, 0.1, 0.5, extra=[phase("mip", 1.2)])))
    agg = cc.aggregate(runs)
    assert not [f for f in agg["findings"] if f["phase"] == "mip"], agg["findings"]
    r = row("f", 0)
    runs.append(judge_ok(r, 10.0, bab_cost(10.0, 0.1, 0.5, extra=[phase("mip", 4.0)])))
    agg = cc.aggregate(runs)
    assert [f for f in agg["findings"] if f["phase"] == "mip" and f["category"] == "f"], agg["findings"]


def test_bounding_nested_inside_selection_is_netted_out_of_child_bound():
    # kFSB shape: child_bound 0.7 of the wall (> select 0.6), but 0.5 of it is simulated children under branch_sim
    wall = 10.0
    ph = [phase("bab_loop", 9.0), phase("branch_select", 6.0), phase("branch_sim", 5.5),
          phase("child_bound", 7.0), phase("beta_opt", 4.0)]
    edges = [{"parent": "root", "child": "bab_loop", "thread_s": 9.0},
             {"parent": "bab_loop", "child": "branch_select", "thread_s": 6.0},
             {"parent": "branch_select", "child": "branch_sim", "thread_s": 5.5},
             {"parent": "branch_sim", "child": "child_bound", "thread_s": 5.0},
             {"parent": "bab_loop", "child": "child_bound", "thread_s": 2.0},
             {"parent": "child_bound", "child": "beta_opt", "thread_s": 4.0}]
    out = judge_ok(row("yolo", 0, budget=300), wall, cost(ph, edges))
    m = cc.row_metrics(out)
    assert abs(m["bound_net_share"] - 0.2) < 1e-12, m["bound_net_share"]
    assert abs(m["in_select_share"] - 0.5) < 1e-12 and m["in_select"] == {"child_bound": 5.0}, m["in_select"]
    agg = cc.aggregate([out])
    sig = [f for f in agg["findings"] if f["kind"] == "select>=bound"]
    assert sig and "nested inside selection" in sig[0]["note"], agg["findings"]
    # the gross share alone (0.7 > select 0.6) would have hidden the signature
    assert agg["per_category"]["yolo"]["mean_share"]["child_bound"] > agg["per_category"]["yolo"]["mean_share"]["branch_select"]


def test_unattributed_wall_is_a_ranked_pseudo_phase():
    runs = []
    for cat in ["a", "b", "c", "d"]:   # well covered: 0.95 of the wall attributed
        runs.append(judge_ok(row(cat, 0), 10.0, bab_cost(10.0, 0.1, 0.5)))
    c = cost([phase("root_bounds", 1.0), phase("attack", 2.0)])   # 0.7 of the wall nowhere
    runs.append(judge_ok(row("e", 0), 10.0, c))
    agg = cc.aggregate(runs)
    assert abs(agg["per_category"]["e"]["mean_share"][cc.UNATTRIBUTED] - 0.7) < 1e-12
    hit = [f for f in agg["findings"] if f["phase"] == cc.UNATTRIBUTED]
    assert hit and hit[0]["category"] == "e", agg["findings"]


def test_the_longest_uninstrumented_stretch_is_located_between_phases():
    # mip 0.05..2.05, then nothing until lowdim at 44.45 (the acasxu shape), all inside a verify umbrella
    ph = [phase("verify", 44.4), phase("mip", 2.0), phase("lowdim", 0.01), phase("load", 0.08)]
    ph[0].update(first_enter_s=0.02, last_exit_s=44.46)
    ph[1].update(first_enter_s=0.05, last_exit_s=2.05)
    ph[2].update(first_enter_s=44.45, last_exit_s=44.46)
    ph[3].update(first_enter_s=0.0, last_exit_s=0.1)
    m = cc.row_metrics(judge_ok(row("acas", 1), 44.46, cost(ph)))
    assert abs(m["max_gap_s"] - 42.4) < 1e-9 and (m["gap_after"], m["gap_before"]) == ("mip", "lowdim"), m
    out = judge_ok(row("acas", 1), 44.46, cost(ph))
    agg = cc.aggregate([out])
    assert "gap 42.4 s (mip -> lowdim)" in cc.render(agg, [out], {})
    # a span CONTAINED in an earlier (dense) one must not shrink the covered frontier
    ph[1].update(last_exit_s=44.0, wall_s=40.0, thread_s=40.0, self_s=40.0)
    ph.append(phase("exact", 0.1))
    ph[-1].update(first_enter_s=0.1, last_exit_s=0.2)
    m = cc.row_metrics(judge_ok(row("acas", 1), 44.46, cost(ph)))
    assert abs(m["max_gap_s"] - 0.45) < 1e-9 and m["gap_after"] == "mip", m


def test_a_sparse_span_never_hides_a_stretch_and_is_charged_back():
    # review(cost-census): the REAL record shape -- `publish` entered for the placeholder at 0.01 s and for the
    # verdict at the end, so its [first, last] span covers the whole row while it occupied 0.2 ms. Counted as
    # occupancy it hid the 42.4 s acasxu stretch (gap 0.00 on every census row).
    ph = [phase("verify", 44.4), phase("mip", 2.0), phase("lowdim", 0.01), phase("load", 0.08),
          phase("publish", 0.0002, enters=2)]
    ph[0].update(first_enter_s=0.02, last_exit_s=44.46)
    ph[1].update(first_enter_s=0.05, last_exit_s=2.05)
    ph[2].update(first_enter_s=44.45, last_exit_s=44.46)
    ph[3].update(first_enter_s=0.0, last_exit_s=0.1)
    ph[4].update(first_enter_s=0.01, last_exit_s=44.461)
    m = cc.row_metrics(judge_ok(row("acas", 1), 44.461, cost(ph)))
    assert abs(m["max_gap_s"] - (42.4 - 0.0002)) < 1e-9, m
    assert (m["gap_after"], m["gap_before"]) == ("mip", "lowdim"), m
    # a sparse phase's occupied wall is charged against every stretch its span overlaps: an attack that ran 30 s
    # somewhere inside a 70 s span leaves at most 40 s of that stretch certainly uninstrumented
    ph2 = [phase("load", 0.1), phase("attack", 30.0, enters=3), phase("lowdim", 1.0)]
    ph2[0].update(first_enter_s=0.0, last_exit_s=0.1)
    ph2[1].update(first_enter_s=0.1, last_exit_s=70.1)
    ph2[2].update(first_enter_s=70.1, last_exit_s=71.1)
    m2 = cc.row_metrics(judge_ok(row("acas", 2), 71.1, cost(ph2)))
    assert abs(m2["max_gap_s"] - 40.0) < 1e-9 and (m2["gap_after"], m2["gap_before"]) == ("load", "lowdim"), m2


def test_a_phase_open_at_the_snapshot_occupies_the_row_to_its_end():
    # a phase still OPEN when the record was written (a concurrent lane published first) has no last exit after
    # its true end: its span runs to the snapshot, so no false `-> end` stretch appears
    ph = [phase("load", 0.1), phase("root_bounds", 20.0, thread=7.7)]
    ph[0].update(first_enter_s=0.0, last_exit_s=0.1)
    ph[1].update(first_enter_s=0.1, last_exit_s=18.0, open=1)
    c = cost(ph)
    c["elapsed_s"] = 20.1
    m = cc.row_metrics(judge_ok(row("cifar", 1), 20.1, c))
    assert m["max_gap_s"] < 1e-9, m
    ph[1]["open"] = 0   # closed at 18.0: the last 2.1 s are a real stretch
    m = cc.row_metrics(judge_ok(row("cifar", 1), 20.1, cost(ph)))
    assert abs(m["max_gap_s"] - 2.1) < 1e-9 and m["gap_before"] == "end", m


def test_nested_selection_bounding_is_netted_in_wall_units_not_thread_sums():
    # review(cost-census): selection on 4 threads at once, each bounding its simulated children: 8 s of child_bound
    # THREAD time nested under branch_select inside 6 s of occupied child_bound wall whose thread time is 24 s.
    # Subtracting thread seconds from wall seconds netted child bounding to 0 and flagged `select>=bound` on a row
    # whose net bounding (2/3 of its child_bound) is far larger than its selection.
    wall = 10.0
    ph = [phase("bab_loop", 9.0), phase("branch_select", 1.5, thread=6.0, inflight=4),
          phase("child_bound", 6.0, thread=24.0, inflight=4)]
    edges = [{"parent": "root", "child": "bab_loop", "thread_s": 9.0},
             {"parent": "bab_loop", "child": "branch_select", "thread_s": 2.0},
             {"parent": "root", "child": "branch_select", "thread_s": 4.0},
             {"parent": "branch_select", "child": "child_bound", "thread_s": 8.0},
             {"parent": "bab_loop", "child": "child_bound", "thread_s": 4.0},
             {"parent": "root", "child": "child_bound", "thread_s": 12.0}]
    out = judge_ok(row("cora", 0), wall, cost(ph, edges))
    m = cc.row_metrics(out)
    assert abs(m["bound_net_share"] - 0.4) < 1e-12, m["bound_net_share"]       # 6 s x (1 - 8/24) / 10 s
    assert abs(m["in_select_share"] - 0.2) < 1e-12, m["in_select_share"]       # 6 s x 8/24 / 10 s
    agg = cc.aggregate([out])
    assert not [f for f in agg["findings"] if f["kind"] == "select>=bound"], agg["findings"]


def test_a_phase_expensive_everywhere_is_reported_as_systemic_and_attack_on_unsat_is_priced():
    runs = []
    for k, cat in enumerate(["a", "b", "c", "d"]):   # attack 0.6 of the wall in every category: no median anomaly
        c = bab_cost(10.0, 0.1, 0.2, extra=[phase("attack", 6.0)])
        runs.append(judge_ok(row(cat, 0), 10.0, c, verdict="unsat"))
    agg = cc.aggregate(runs)
    assert not [f for f in agg["findings"] if f["phase"] == "attack"], agg["findings"]
    sysm = {x["phase"]: x for x in agg["systemic"]}
    assert "attack" in sysm and abs(sysm["attack"]["median_share"] - 0.6) < 1e-12, agg["systemic"]
    assert sysm["attack"]["categories_over_0p1"] == ["a", "b", "c", "d"]
    au = agg["attack_on_unsat"]
    assert au["rows"] == 4 and abs(au["attack_s"] - 24.0) < 1e-9 and abs(au["wall_s"] - 40.0) < 1e-9, au
    # top-level shares 0.1 + 0.2 + 0.05 + 0.6 = 0.95: the attack could have run serially
    assert au["worst"][0]["concurrent"] is False
    over = judge_ok(row("e", 0), 10.0, bab_cost(10.0, 0.1, 0.2, extra=[phase("attack", 9.0)]), verdict="unsat")
    au2 = cc.aggregate(runs + [over])["attack_on_unsat"]
    assert au2["worst"][0]["tag"] == "e__i0" and au2["worst"][0]["concurrent"] is True, au2   # 1.25 > 1.05
    md = cc.render(agg, runs, {})
    assert "## Systemic cost centres" in md and "Attack on rows the run proved unsat" in md
    to = judge_ok(row("f", 0, bank="timeout", stratum="open"), 10.0,
                  bab_cost(10.0, 0.1, 0.2, extra=[phase("attack", 3.0)]), verdict="timeout")
    ao = cc.aggregate(runs + [to])["attack_on_open"]
    assert ao["rows"] == 1 and abs(ao["attack_s"] - 3.0) < 1e-9 and ao["worst"][0]["tag"] == "f__i0", ao


def test_a_relational_row_without_an_instance_line_is_judged_by_its_route():
    r = row("iso", 1)
    fl = flight("iso", 10.0, bab_cost(10.0, 0.1, 0.5))
    lg = "\n".join(l for l in log(r).splitlines() if not l.startswith("Running VNN-COMP instance:")) + "\n"
    out = cc.judge_run(r, fl, lg, "unsat\n", COMMIT)
    assert out["status"] == "VOID" and "missing `Running VNN-COMP instance:`" in " ".join(out["why"]), out
    fl["events"].insert(0, {"method": "relational_route", "status": "ran", "reason": "dual-network"})
    out = cc.judge_run(r, fl, lg, "unsat\n", COMMIT)
    assert out["status"] == "OK" and out["relational"], out
    # a run line naming ANOTHER row is VOID even on the relational route
    out = cc.judge_run(r, fl, log(r, vnnlib="/b/iso/vnnlib/p9.vnnlib"), "unsat\n", COMMIT)
    assert out["status"] == "VOID" and "names another row" in " ".join(out["why"]), out


def test_overhead_bound_prices_every_enter_and_sample():
    r = row("cat_a", 1)
    c = cost([phase("bab_loop", 5.0, enters=1000), phase("child_bound", 4.0, enters=99000)])
    c["rss_samples"] = 100
    out = judge_ok(r, 10.0, c)
    got = cc.overhead_bound(out, guard_ns=100.0, sample_us=20.0)
    assert abs(got - (100000 * 100e-9 + 100 * 20e-6) / 10.0) < 1e-12, got
    agg = cc.aggregate([out], guard_ns=100.0, sample_us=20.0)
    assert agg["overhead"]["worst_tag"] == "cat_a__i1" and agg["overhead"]["rows_over_0p5pct"] == 0
    md = cc.render(agg, [out], {})
    assert "census overhead bound" in md and "overhead<=" in md
    assert cc.aggregate([out])["overhead"] is None, "no guard cost given: no bound claimed"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main() -> int:
    passed = failed = 0
    for t in TESTS:
        try:
            t()
        except Exception:  # noqa: BLE001 - every failure is reported, none is fatal to the run
            failed += 1
            print(f"test_{t.__name__[5:]} ... FAILED")
            traceback.print_exc()
        else:
            passed += 1
            print(f"test_{t.__name__[5:]} ... ok")
    rc = 0 if failed == 0 and passed > 0 else 1
    print(f"COSTCENSUS_SELFTEST_DONE rc={rc} passed={passed} failed={failed}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
