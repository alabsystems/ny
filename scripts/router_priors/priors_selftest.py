#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Named selftests of the offline-priors pipeline (scripts/router_priors). Synthetic data only, plus a
drift check of the committed table against its committed training set.

    python3 scripts/router_priors/priors_selftest.py   # last line: PRIORS_SELFTEST_DONE rc=0 ran=N passed=N failed=0
"""

from __future__ import annotations

import math
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import priorslib as P  # noqa: E402

ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def check(cond, why):
    if not cond:
        raise AssertionError(why)


def ev(method, status, at, reason=""):
    return {"method": method, "status": status, "at_secs": at, "reason": reason}


@test
def key_bits_follow_the_documented_truth_table():
    check(P.key_bits({"has_conv": "1"}) == (1, 0), "conv")
    check(P.key_bits({"has_convT": "1"}) == (1, 0), "convT is conv")
    check(P.key_bits({"has_sigmoid": "1"}) == (0, 1), "sigmoid is non-PWL")
    check(P.key_bits({"n_bilinear": "3"}) == (0, 1), "a var x var product is non-PWL")
    check(P.key_bits({"has_layernorm": "1", "has_conv": "1"}) == (1, 1), "both bits")
    check(P.key_bits({"has_relu": "1", "has_bn": "1", "has_maxpool": "1"}) == (0, 0),
          "ReLU, BN and pooling are piecewise-linear")


@test
def censoring_drops_capped_and_successful_attacks():
    base = [ev("upfront_attack_execution", "ran", 0.1, "consultation=ordinary effective_budget_secs=4.000000")]
    done = P.phase_observations({"budget_secs": 100, "events": base + [
        ev("upfront_attack", "ran", 2.1, "consulted; no confirmed candidate")]})
    check(done == [("attack.upfront", 2.0, "cap=4")], f"an exhausted attack is observed: {done}")
    capped = P.phase_observations({"budget_secs": 100, "events": base + [
        ev("upfront_attack", "ran", 3.9, "consulted; no confirmed candidate")]})
    check(capped == [], f"an attack that ran into its cap is right-censored: {capped}")
    found = P.phase_observations({"budget_secs": 100, "events": base + [
        ev("upfront_attack", "ran", 1.0, "confirmed candidate")]})
    check(found == [], f"an attack ended by success is censored: {found}")


@test
def root_bound_is_kept_only_when_no_cap_could_end_it():
    def flight(verified, total, elapsed, budget=100):
        return {"budget_secs": budget, "events": [
            ev("crown_precheck_begin", "ran", 0.3, f"ordinal=0 clauses={total} precheck_fraction=0.850"),
            ev("crown_precheck_end", "ran", 0.3 + elapsed,
               f"ordinal=0 verified={verified} total={total} elapsed_s={elapsed}")]}
    check(P.phase_observations(flight(2, 2, 60.0)) == [("root.bound", 30.0, "clauses=2")], "all verified")
    check(P.phase_observations(flight(1, 2, 0.03))[0][0] == "root.bound", "short and unverified: kept")
    check(P.phase_observations(flight(210, 1000, 14.7, budget=20)) == [], "cut by the budget: censored")


@test
def units_collapse_models_and_balance_families():
    rows = [P.Row("verifier.setup", "fa", "ca", "m1", 0, 0, 1024.0, v, f"r{i}") for i, v in enumerate((1, 2, 4))]
    rows += [P.Row("verifier.setup", "fb", "cb", f"m{k}", 0, 0, 4096.0, 1.0, f"s{k}") for k in range(2, 6)]
    units = P.units_of(rows, "verifier.setup")
    check(len(units) == 5, f"one unit per (family, model): {len(units)}")
    u1 = [u for u in units if u.model == "m1"][0]
    check(u1.y == 1.0 and u1.n_rows == 3, f"the median of the model's rows (2.0 -> log2 1.0): {u1}")
    wa = sum(u.weight for u in units if u.family == "fa")
    wb = sum(u.weight for u in units if u.family == "fb")
    check(abs(wa - 1.0) < 1e-12 and abs(wb - 1.0) < 1e-12, f"each family weighs 1: {wa} {wb}")


def synth(families, per_model=2, models=3, cell_of=None, slope=0.3, icpt=-6.0, noise=0.05, off=None):
    rows = []
    for fi, fam in enumerate(families):
        for m in range(models):
            params = 2.0 ** (10 + 2 * m + fi % 3)
            conv, nonpwl = cell_of(fam) if cell_of else (fi % 2, 0)
            for k in range(per_model):
                y = icpt + slope * math.log2(params) + (noise if (m + k) % 2 else -noise)
                if off and fam in off:
                    y += off[fam]
                rows.append(P.Row("verifier.setup", fam, fam, f"{fam}-m{m}", conv, nonpwl, params, 2 ** y,
                                  f"{fam}-{m}-{k}"))
    return rows


@test
def a_shared_structure_ships_with_every_gate_recorded():
    # Ten families: with F families a two-cell key can at best name 2 of F (excess 1/F), so G3 needs F > 5
    # even in every leave-one-out refit -- few-family phases are refused by construction.
    rows = synth([f"f{i}" for i in range(10)])
    rec = P.build(rows)["verifier.setup"]
    check(rec["ships"], f"a clean shared power law ships: {rec.get('refused')}")
    check(abs(rec["model"].slope - 0.3) < 0.05, f"the shared exponent is recovered: {rec['model'].slope}")
    check(rec["stable"][0] == rec["stable"][1] == 10, f"G7 holds in every refit: {rec['stable']}")
    text = P.render_table({"verifier.setup": rec, "attack.upfront": {"ships": False, "refused": "x"},
                           "root.bound": {"ships": False, "refused": "y"}}, "0" * 64, {"receipts_used": 1})
    check("stable_folds=10\tstable_of=10" in text, "the evidence line carries G7")


@test
def a_per_family_key_is_refused_by_purity():
    fams = [f"f{i}" for i in range(4)]
    cell = {f: (i % 2, i // 2) for i, f in enumerate(fams)}  # every family alone in its cell
    rows = synth(fams, cell_of=lambda f: cell[f])
    units = P.units_of(rows, "verifier.setup")
    check(P.passing_cells(units) == set(), "a cell holding one family never ships (G2)")
    rec = P.build(rows)["verifier.setup"]
    check(set(rec["model"].intercepts) == {P.ROOT_CELL}, "every cell collapses to the root")


@test
def a_fingerprinting_key_fails_g3_on_held_out_models():
    fams = [f"f{i}" for i in range(6)]
    units = []
    for i, f in enumerate(fams):
        for m in range(4):
            units.append(P.Unit("verifier.setup", f, f"{f}-model-{m}", f"cell{i}", 10.0, 1.0, 1))
    units = P._reweighted(units)
    fp = P.fingerprint_gate(units, {f"cell{i}" for i in range(6)})
    check(fp["excess"] > P.FINGERPRINT_EXCESS_MAX, f"one family per key value is a lookup: {fp}")
    coarse = P.fingerprint_gate(units, set())
    check(coarse["excess"] <= 1e-12, f"the root key carries nothing: {coarse}")


@test
def a_pass_that_depends_on_one_family_is_refused_by_g7():
    """G7 wiring: when ONE leave-one-family-out refit fails G2-G6, the phase does not ship (and says how many
    refits passed). The refit's decision is forced here so the test is decisive about the wiring."""
    rows = synth([f"f{i}" for i in range(10)])
    real = P.gates_pass
    try:
        P.gates_pass = lambda units: "f3" in {u.family for u in units}  # dropping f3 breaks the pass
        rec = P.build(rows)["verifier.setup"]
    finally:
        P.gates_pass = real
    check(not rec["ships"], "a pass that needs f3 is f3's model")
    check(rec["stable"] == (9, 10), f"nine refits keep f3 and pass, one drops it: {rec['stable']}")
    check("G7 ship decision unstable under LOGO (9 of 10 refits pass)" in rec["refused"], rec["refused"])
    # ... and the real committed data exhibits exactly this on attack.upfront.
    trained = P.build(P.read_rows(os.path.join(HERE, "data", "priors_v1_trainset.tsv")))
    check(not trained["attack.upfront"]["ships"] and "G7" in trained["attack.upfront"]["refused"],
          str(trained["attack.upfront"].get("refused")))


@test
def the_fixture_is_anonymous_and_the_committed_table_has_no_drift():
    rows = P.read_rows(os.path.join(HERE, "data", "priors_v1_trainset.tsv"))
    phases = P.build(rows)
    fixture = P.render_fixture(rows, phases)
    for cat in {r.category for r in rows}:
        check(cat not in fixture, f"the fixture names {cat}")
    table = P.render_table(phases, P.trainset_sha(rows), {"receipts_used": len({r.receipt for r in rows})})
    committed = open(os.path.join(ROOT, "crates/ny-core/src/router/priors_v1.tsv")).read()
    check(table == committed, "the committed table is not what the pipeline generates from its training set")
    committed_fx = open(os.path.join(ROOT, "crates/ny-core/tests/data/priors_v1_rows.tsv")).read()
    check(fixture == committed_fx, "the committed fixture drifted")
    for cat in {r.category for r in rows}:
        check(cat not in committed, f"the compiled table names {cat}")


@test
def sign_test_is_exact():
    check(P.sign_test_p(0, 0) == 1.0, "empty")
    check(abs(P.sign_test_p(7, 2) - 0.1796875) < 1e-12, str(P.sign_test_p(7, 2)))
    check(P.sign_test_p(10, 0) < 0.002, str(P.sign_test_p(10, 0)))


def main() -> int:
    ran = passed = 0
    for fn in TESTS:
        ran += 1
        try:
            fn()
            passed += 1
            print(f"TEST {fn.__name__} ... ok")
        except Exception as exc:  # noqa: BLE001
            print(f"TEST {fn.__name__} ... FAILED: {exc}")
            traceback.print_exc(limit=3)
    failed = ran - passed
    rc = 0 if failed == 0 else 1
    print(f"PRIORS_SELFTEST_DONE rc={rc} ran={ran} passed={passed} failed={failed}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
