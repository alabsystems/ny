#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the M0 arming harness (scripts/m0/). No cargo, no ny binary, no box.

Every test is a unittest method, so the run prints one `test_<name> (...) ... ok` line per test and a gate can
assert each BY NAME (CLAUDE.md: a filter that matched nothing must not read as green). The last line is
`M0_SELFTEST_DONE rc=<0|1> ran=<n> failures=<n> errors=<n> skipped=<n> lane=<default|real-data> bash=<path>
bash_version=<v> timeout=<path|stand-in>`; a run that ran nothing exits 1.

The runner tests drive scripts/m0/moat_both_profiles.sh end to end against a STUB binary and a STUB
`scripts/ny_box.sh` under a temporary fake root (M0_NY), so the shell's exit codes, its EMERGENCY stop, its
VOID accounting, its zero-NY_* refusal and its resume are pinned without touching the box.

`--bash PATH` selects the shell that runs the runner (default: `bash` on PATH). `--bash /bin/bash` is the stock
shell, bash 3.2 on macOS, and ny-test-utils' `python_tooling_black_box` target (its `m0_` tests) runs the default
lane exactly that way, so a bash-4-only construct in the runner is a red, not a host quirk. (On Linux /bin/bash is
bash 5; there the lexical scan in ny-test-utils' `source_policy` target is what refuses such a construct.) The
runner's own `python3` is this interpreter (a wrapper first on the runner's PATH), so the judge it calls has the
same packages as the selftest.
On a host with no `timeout` on PATH (macOS ships none; GNU coreutils provides it on Linux) the runner tests put a
stand-in first on PATH that keeps GNU timeout's contract -- the command runs in timeout's own process group,
expiry sends that group SIGTERM and exits 124, otherwise the command's status (128+N for signal N), 126/127
when it cannot run -- because what they pin is the RUNNER's use of the backstop (`timeout budget+10` around the
pin, rc 124 judged from the results file), not coreutils. The DONE line names which one ran.

The three real-data tests (class `RealData`) read the main tree's benchmarks/, reports/ and the field results.
They are a separate lane, absent from the default run, and selecting it makes a missing input a FAILURE:

    python3 scripts/m0/m0_selftest.py              # default lane
    python3 scripts/m0/m0_selftest.py --real-data  # real-data lane (main tree + field results required)
    python3 scripts/m0/m0_selftest.py --bash /bin/bash  # default lane, runner under the stock shell
"""

from __future__ import annotations

import contextlib
import gzip
import io
import glob
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from fractions import Fraction
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m0lib as L  # noqa: E402
import knife_edge as KE  # noqa: E402
import negative_controls as NC  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
# The shell that runs moat_both_profiles.sh in the Runner tests; main() sets it from --bash.
RUNNER_BASH = "bash"
# The runner's timestamp, as GNU `date -Is` prints it (the runner builds it portably: now_iso).
ISO_STAMP = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d"
NY_ROOT = "<home>/ny"
TRUTH26 = os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt")
# The real-data lane's shared inputs (main-tree benchmarks, bank manifest, field results).
REAL_INPUTS = (os.path.join(NY_ROOT, "benchmarks/vnncomp2026/benchmarks"), TRUTH26,
               os.path.join(NY_ROOT, "reports/measured-2026/bank_manifest.json"))

V1_PROP = textwrap.dedent("""\
    ; toy property
    (declare-const X_0 Real)
    (declare-const X_1 Real)
    (declare-const Y_0 Real)
    (declare-const Y_1 Real)
    (assert (<= X_0 1.0))
    (assert (>= X_0 -1.0))
    (assert (<= X_1 0.5))
    (assert (>= X_1 0.0))
    (assert (or
        (and (>= Y_0 0.25))
        (and (<= Y_0 -0.25))
        (and (>= Y_1 Y_0) (<= Y_1 3.0))
    ))
    """)

V2_PROP = textwrap.dedent("""\
    (vnnlib-version <2.0>)
    (declare-network N
        (declare-input  X float32 [1, 2, 2])
        (declare-output Y float32 [1, 3])
    )
    (assert (>= X[0,0,0] 0.0))
    (assert (<= X[0,0,0] 1.0))
    (assert (>= X[0,0,1] 0.0))
    (assert (<= X[0,0,1] 1.0))
    (assert (>= X[0,1,0] 0.0))
    (assert (<= X[0,1,0] 1.0))
    (assert (>= X[0,1,1] 0.0))
    (assert (<= X[0,1,1] 1.0))
    (assert (or
      (and (<= Y[0,1] Y[0,0]))
      (and (<= Y[0,1] Y[0,2]))
    ))
    """)


def write(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return path


# ---------------------------------------------------------------------------------------------------------
class VnnlibAndWitness(unittest.TestCase):
    def test_parse_v1_threshold_disjunction(self):
        p = L.parse_property(V1_PROP)
        self.assertEqual((p.n_x, p.n_y, len(p.clauses)), (2, 2, 3))
        self.assertEqual([a.kind for a in p.clauses[2]], ["relational", "threshold"])
        self.assertEqual((p.lower[1], p.upper[1]), (0.0, 0.5))
        self.assertEqual(p.clauses[1][0].direction, "le")

    def test_parse_v2_tensor_syntax_flattens_row_major(self):
        p = L.parse_property(V2_PROP)
        self.assertEqual((p.version, p.n_x, p.n_y), ("2.0", 4, 3))
        self.assertEqual(sorted(p.lower), [0, 1, 2, 3])
        a = p.clauses[1][0]
        self.assertEqual((a.kind, a.big, a.small), ("relational", 2, 1))

    def test_parse_refuses_mixed_arithmetic_and_disjunctive_inputs(self):
        for bad in ("(declare-const X_0 Real)(declare-const Y_0 Real)(assert (and (<= X_0 1) (<= Y_0 1)))",
                    "(declare-const X_0 Real)(declare-const Y_0 Real)(declare-const Y_1 Real)"
                    "(assert (<= X_0 1))(assert (>= X_0 0))(assert (<= (- Y_0 Y_1) 1))",
                    "(declare-const X_0 Real)(declare-const Y_0 Real)(assert (or (<= X_0 1) (>= X_0 2)))"
                    "(assert (<= Y_0 1))",
                    "(declare-const X_0 Real)(declare-const Y_0 Real)(assert (<= X_0 1))(assert (<= Y_0 1))"):
            with self.assertRaises(L.M0Error, msg=bad):
                L.parse_property(bad)

    def test_witness_v1_with_sat_header_and_gz(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.counterexample.gz")
            with gzip.open(p, "wt") as fh:
                fh.write("sat\n((X_0 0.5)\n(X_1 0.25)\n(Y_0 0.75)\n(Y_1 -1.0))\n")
            w = L.load_witness(p)
            self.assertEqual((w.x, w.y), ({0: 0.5, 1: 0.25}, {0: 0.75, 1: -1.0}))

    def test_witness_v2_tensor_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(os.path.join(d, "w"), "X float32 [1, 2, 2]\n0.1\n0.2\n0.3\n0.4\nY float32 [1, 3]\n1\n2\n3\n")
            w = L.load_witness(p)
            self.assertEqual((w.x[3], w.y[2]), (0.4, 3.0))
            write(p, "X float32 [1, 2, 2]\n0.1\n0.2\n0.3\nY float32 [1, 3]\n1\n2\n3\n")
            with self.assertRaises(L.M0Error):
                L.load_witness(p)

    def test_witness_refuses_non_sat_results_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            for body in ("unsat\n", "timeout\n", "((X_0 1)(X_0 2)(Y_0 1))", "((X_0 1))", "((X_0 nan)(Y_0 1))"):
                p = write(os.path.join(d, "w"), body)
                with self.assertRaises(L.M0Error, msg=body):
                    L.load_witness(p)

    def test_witness_outside_the_box_is_refused(self):
        p = L.parse_property(V1_PROP)
        self.assertEqual(L.witness_in_box(p, {0: 1.0, 1: 0.25}), 0)
        with self.assertRaises(L.M0Error):
            L.witness_in_box(p, {0: 1.0000000000000002, 1: 0.25})
        with self.assertRaises(L.M0Error):
            L.witness_in_box(p, {0: 0.0})


# ---------------------------------------------------------------------------------------------------------
class KnifeEdge(unittest.TestCase):
    def test_f64_floor_and_ceil_are_exact(self):
        rng = random.Random(7)
        for _ in range(2000):
            q = Fraction(rng.randint(-10**12, 10**12), rng.randint(1, 10**9)) + Fraction(1, 10**15)
            lo, hi = L.f64_floor(q), L.f64_ceil(q)
            self.assertLessEqual(Fraction(lo), q)
            self.assertGreaterEqual(Fraction(hi), q)
            self.assertGreater(Fraction(math.nextafter(lo, math.inf)), q)
            self.assertLess(Fraction(math.nextafter(hi, -math.inf)), q)

    def test_plain_decimal_round_trips_without_exponent(self):
        for x in (1e-12, -3.5e-9, 1e16, 0.5161725878705514, -0.0, 123456789.125, 5e-324):
            s = L.plain_decimal(x)
            self.assertNotIn("e", s.lower())
            self.assertEqual(float(s), x)

    def test_knife_edge_slack_is_at_least_delta_exactly(self):
        rng = random.Random(11)
        p = L.parse_property(V1_PROP)
        for _ in range(400):
            y0 = rng.choice([1, -1]) * rng.uniform(0.3, 50.0)
            y = {0: y0, 1: y0 - 10.0}
            for d in L.DEFAULT_DELTAS:
                _, rep = L.knife_edge_rewrite(p, y, d)
                for r in rep["rewrites"]:
                    slack = Fraction(y[r["output"]]) - Fraction(r["new"]) if r["direction"] == "ge" \
                        else Fraction(r["new"]) - Fraction(y[r["output"]])
                    dd = Fraction(d)
                    self.assertGreaterEqual(slack, dd)
                    self.assertLess(slack - dd, Fraction(2 * math.ulp(abs(r["new"]) + abs(y[r["output"]]))))
                    self.assertEqual(float(r["new_text"]), r["new"])

    def test_knife_edge_rewrites_only_the_threshold_token(self):
        p = L.parse_property(V1_PROP)
        text, rep = L.knife_edge_rewrite(p, {0: 0.9, 1: -5.0}, "1e-6")
        self.assertEqual(rep["clauses_rewritten"], [0])
        self.assertTrue(rep["knife_edge_exact"])
        self.assertEqual(text.replace("0.899999", "0.25", 1).count("0.25"), 2)
        a, b = V1_PROP.split("0.25", 1)
        self.assertTrue(text.startswith(a))
        self.assertTrue(text.endswith(b))
        self.assertIn("(and (<= Y_0 -0.25))", text)
        q = L.parse_property(text)
        self.assertEqual(len(q.clauses), 3)

    def test_knife_edge_rewrites_every_witnessed_clause(self):
        p = L.parse_property(V1_PROP)
        _, rep = L.knife_edge_rewrite(p, {0: 1.0, 1: 2.0}, "1e-9")  # clause 0 (Y0>=.25) and clause 2 hold
        self.assertEqual(rep["clauses_rewritten"], [0, 2])
        self.assertEqual({r["output"] for r in rep["rewrites"]}, {0, 1})

    def test_knife_edge_refuses_a_relational_only_witnessed_clause(self):
        p = L.parse_property(V2_PROP)
        with self.assertRaises(L.M0Error) as cm:
            L.knife_edge_rewrite(p, {0: 0.0, 1: -1.0, 2: 0.5}, "1e-6")
        self.assertIn("relational atom cannot take an offset", str(cm.exception))
        _, rep = L.knife_edge_rewrite(p, {0: 0.0, 1: -1.0, 2: 0.5}, "1e-6", allow_partial=True)
        self.assertFalse(rep["knife_edge_exact"])
        self.assertEqual(rep["clauses_relational_only_at_natural_slack"], [0, 1])

    def test_knife_edge_refuses_a_non_witness_unless_asked(self):
        p = L.parse_property(V1_PROP)
        y = {0: 0.0, 1: -5.0}  # no clause holds
        with self.assertRaises(L.M0Error):
            L.knife_edge_rewrite(p, y, "1e-3")
        text, rep = L.knife_edge_rewrite(p, y, "1e-3", clause=0, allow_non_witness=True)
        q = L.parse_property(text)
        self.assertTrue(all(a.holds({k: Fraction(v) for k, v in y.items()}) for a in q.clauses[0]))
        with self.assertRaises(L.M0Error):  # a violated relational atom cannot be fixed by a threshold
            L.knife_edge_rewrite(p, y, "1e-3", clause=2, allow_non_witness=True)

    def test_knife_edge_ladder_files_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            v = write(os.path.join(d, "p.vnnlib"), V1_PROP)
            w = write(os.path.join(d, "w.counterexample"), "((X_0 0.5)\n(X_1 0.5)\n(Y_0 -3.0)\n(Y_1 0.0))\n")
            man = KE.build_ladder(v, w, os.path.join(d, "out"), list(L.DEFAULT_DELTAS))
            self.assertEqual(len(man["rungs"]), 4)
            for rung in man["rungs"]:
                body = open(rung["file"]).read()
                self.assertIn("; m0 knife-edge rung delta=", body)
                self.assertEqual(rung["sha256"], L.sha256_file(rung["file"]))
            self.assertTrue(man["rungs"][-1]["sub_f32_resolution"])
            self.assertFalse(man["rungs"][0]["sub_f32_resolution"])
            with self.assertRaises(L.M0Error):
                KE.build_ladder(v, w, os.path.join(d, "out"), ["1e-3"])



# ---------------------------------------------------------------------------------------------------------
YAML = textwrap.dedent("""\
    # a category preset
    attack:
      pgd_order: middle
      ny_pgd_order_compat: upfront   # historical
      pgd_restarts: 10
      falsify_portfolio: true
      upfront_attack_probe_first: true
    bab:
      phase_budget:
        disjunctive_pgd_fraction: 0.4
        disjunctive_pgd_min_secs: 20
        post_bab_pgd_fraction: 0.10
    margin_row:
      concurrent: true
    """)


class Overlays(unittest.TestCase):
    def test_overlay_sets_deletes_and_keeps_comments(self):
        out = L.apply_overlay(YAML, {"attack.pgd_order": "skip", "margin_row.frozen_dual_bab": True,
                                     "new_section.key": 1.5}, ["attack.ny_pgd_order_compat"], "t")
        self.assertIn("# a category preset", out)
        doc = L.strict_load(out)
        self.assertEqual(doc["attack"]["pgd_order"], "skip")
        self.assertNotIn("ny_pgd_order_compat", doc["attack"])
        self.assertIs(doc["margin_row"]["frozen_dual_bab"], True)
        self.assertEqual(doc["new_section"]["key"], 1.5)

    def test_overlay_float_spelling_is_typed_in_both_yaml_dialects(self):
        out = L.apply_overlay(YAML, {"margin_row.frozen_dual_bab_secs": 1e9}, [], "t")
        self.assertIn("frozen_dual_bab_secs: 1000000000.0", out)
        self.assertIsInstance(L.strict_load(out)["margin_row"]["frozen_dual_bab_secs"], float)

    def test_overlay_refuses_duplicates_and_mapping_deletes(self):
        with self.assertRaises(Exception):
            L.strict_load("a: 1\na: 2\n")
        with self.assertRaises(L.M0Error):
            L.apply_overlay(YAML, {}, ["bab.phase_budget"], "t")
        with self.assertRaises(L.M0Error):
            L.apply_overlay(YAML, {"bab": 1}, [], "t")

    def test_attack_off_edits_switch_every_lane_off(self):
        sets, deletes, _ = L.attack_off_edits(L.strict_load(YAML))
        out = L.strict_load(L.apply_overlay(YAML, sets, deletes, "t"))
        a, pb = out["attack"], out["bab"]["phase_budget"]
        self.assertEqual((a["pgd_order"], a["upfront_attack"], a["falsify_portfolio"]), ("skip", False, False))
        self.assertNotIn("ny_pgd_order_compat", a)
        self.assertNotIn("upfront_attack_probe_first", a)
        self.assertEqual(a["pgd_restarts"], 10)  # a parameter of a lane the masters switch off: kept
        self.assertEqual((pb["disjunctive_pgd_fraction"], pb["post_bab_pgd_fraction"], pb["upfront_pgd_fraction"],
                          pb["attack_extension_fraction"], pb["vnncomp_post_bab_attack"]), (0.0, 0.0, 0.0, 0.0, False))
        self.assertNotIn("disjunctive_pgd_min_secs", pb)

    def test_attack_table_covers_the_preset_schema(self):
        schema = os.path.join(REPO, "crates/ny-cli/src/preset/mod.rs")
        self.assertTrue(os.path.isfile(schema), f"{schema} is tracked; its absence is repository drift")
        keys = L.check_attack_table(schema)
        self.assertGreater(len(keys), 60)
        self.assertIn("attack.upfront_attack", keys)
        self.assertIn("bab.phase_budget.vnncomp_post_bab_attack", keys)

    def test_attack_table_refuses_unclassified_and_stale_keys(self):
        with tempfile.TemporaryDirectory() as d:
            fields = "".join(f"    pub(crate) {k.split('.')[-1]}: Option<bool>,\n"
                             for k in L.ATTACK_TABLE if k.startswith("attack."))
            pb = "".join(f"    pub(crate) {k.split('.')[-1]}: Option<bool>,\n"
                         for k in L.ATTACK_TABLE if k.startswith("bab.phase_budget."))
            ins = "    pub(crate) lowdim_leaf_attack_every: Option<u64>,\n"
            src = (f"pub(crate) struct AttackPreset {{\n{fields}}}\n"
                   f"pub(crate) struct PhaseBudgetPreset {{\n{pb}}}\n"
                   f"pub(crate) struct InputSplitPreset {{\n{ins}}}\n")
            ok = write(os.path.join(d, "ok.rs"), src)
            L.check_attack_table(ok)
            bad = write(os.path.join(d, "bad.rs"), src.replace(
                "pub(crate) struct AttackPreset {\n", "pub(crate) struct AttackPreset {\n    pub(crate) new_lane: Option<bool>,\n"))
            with self.assertRaises(L.M0Error) as cm:
                L.check_attack_table(bad)
            self.assertIn("UNCLASSIFIED attack key attack.new_lane", str(cm.exception))
            stale = write(os.path.join(d, "stale.rs"), src.replace("    pub(crate) falsify_portfolio: Option<bool>,\n", ""))
            with self.assertRaises(L.M0Error) as cm:
                L.check_attack_table(stale)
            self.assertIn("STALE table key attack.falsify_portfolio", str(cm.exception))

    def test_resolve_preset_mirrors_the_resolver(self):
        with tempfile.TemporaryDirectory() as d:
            for rel in ("vnncomp24/cifar100.yaml", "vnncomp25/cifar100_2024.yaml", "vnncomp25/cgan2026.yaml",
                        "vnncomp26/cgan2026.yaml", "vnncomp25/acasxu.yaml"):
                write(os.path.join(d, rel), "a: 1\n")
            self.assertEqual(L.resolve_preset(d, "cifar100_2024"), "vnncomp25/cifar100_2024.yaml")
            self.assertEqual(L.resolve_preset(d, "cgan2026"), "vnncomp26/cgan2026.yaml")
            self.assertEqual(L.resolve_preset(d, "Acasxu_2023"), "vnncomp25/acasxu.yaml")
            self.assertIsNone(L.resolve_preset(d, "vit_2023"))



# ---------------------------------------------------------------------------------------------------------
class Alignment(unittest.TestCase):
    def _inst(self, d, rows):
        write(os.path.join(d, "instances.csv"), "".join(f"{o},{v},100\n" for o, v in rows))
        for o, v in rows:
            write(os.path.join(d, o), "x")
            write(os.path.join(d, v), "x")
        return L.load_instances(os.path.join(d, "instances.csv"))

    def test_align_bank_by_index_by_permutation_and_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            inst = self._inst(d, [("onnx/a.onnx", "vnnlib/p0.vnnlib"), ("onnx/a.onnx", "vnnlib/p1.vnnlib")])
            same = [("onnx/a.onnx", "vnnlib/p0.vnnlib", "sat"), ("./onnx/a.onnx", "./vnnlib/p1.vnnlib", "unsat")]
            self.assertEqual(L.align_bank(inst, same, "t1"), ["sat", "unsat"])
            self.assertEqual(L.ALIGN_MODES["t1"], "by-index")
            perm = [same[1], same[0]]
            self.assertEqual(L.align_bank(inst, perm, "t2"), ["sat", "unsat"])
            self.assertEqual(L.ALIGN_MODES["t2"], "by-key(permutation)")
            with self.assertRaises(L.M0Error):
                L.align_bank(inst, [same[0], ("onnx/a.onnx", "vnnlib/p9.vnnlib", "sat")], "t3")
            dup = self._inst(os.path.join(d, "dup"), [("onnx/a.onnx", "vnnlib/p0.vnnlib")] * 2 +
                             [("onnx/a.onnx", "vnnlib/p1.vnnlib")])
            with self.assertRaises(L.M0Error):  # repeated keys and a different order: no guess
                L.align_bank(dup, [same[1], same[0], same[0]], "t4")

    def test_results_txt_parse_and_key_asserted_alignment(self):
        txt = textwrap.dedent("""\
            Category 2026_toy:
            Checking counterexamplesfor index 0. Violated: 1
            Row: ['a-p0', '1.0 (v)']
            True Result: sat
            0: toolA score: 10, is_ver: False, is_fals: True, is_fastest: False
            Row: ['a-p1', '1.0 (h)']
            True Result: unsat
            1: toolA score: 10, is_ver: True, is_fals: False, is_fastest: False
            1: toolB score: -150, is_ver: False, is_fals: True, is_fastest: False
            Category 2026_toy:
            toolA: 20 (100.0%)
            """)
        with tempfile.TemporaryDirectory() as d:
            t = L.parse_results_txt(write(os.path.join(d, "results.txt"), txt), 2026)
            self.assertEqual((t["toy"][0].token(), t["toy"][1].token()), ("sat", "unsat"))
            inst = self._inst(d, [("onnx/a.onnx", "vnnlib/p0.vnnlib"), ("onnx/a.onnx", "vnnlib/p1.vnnlib")])
            self.assertEqual([x.token() for x in L.align_truth(inst, t["toy"], "toy")], ["sat", "unsat"])
            swapped = self._inst(os.path.join(d, "s"), [("onnx/a.onnx", "vnnlib/p1.vnnlib"),
                                                        ("onnx/a.onnx", "vnnlib/p0.vnnlib")])
            with self.assertRaises(L.M0Error):
                L.align_truth(swapped, t["toy"], "toy")

    def test_moat_selects_field_sat_rows_whatever_the_bank(self):
        import moat_rows as MR
        self.assertTrue(MR.select_row("unsat", "-"))
        self.assertTrue(MR.select_row("sat", "unsat"))
        self.assertTrue(MR.select_row("timeout", "sat"))       # the row a newly armed tier can falsely close
        self.assertTrue(MR.select_row("unknown", "sat"))
        self.assertFalse(MR.select_row("timeout", "unsat"))
        self.assertFalse(MR.select_row("timeout", "-"))
        self.assertFalse(MR.select_row("timeout", "sat", banked_only=True))
        self.assertTrue(MR.select_row("timeout", "-", all_rows=True))

    def test_paired_model_field_becomes_an_absolute_literal(self):
        with tempfile.TemporaryDirectory() as d:
            write(os.path.join(d, "onnx/f.onnx"), "x")
            write(os.path.join(d, "onnx/g.onnx"), "x")
            write(os.path.join(d, "vnnlib/i0.vnnlib"), "x")
            write(os.path.join(d, "instances.csv"),
                  "\"[('f', 'onnx/f.onnx'), ('g', 'onnx/g.onnx')]\",./vnnlib/i0.vnnlib,100\n")
            r = L.load_instances(os.path.join(d, "instances.csv"))[0]
            self.assertEqual(r.key, "f_g-i0")
            self.assertEqual(L.model_files(r.onnx), [os.path.realpath(os.path.join(d, "onnx/f.onnx")),
                                                     os.path.realpath(os.path.join(d, "onnx/g.onnx"))])
            with self.assertRaises(L.M0Error):
                L.parse_model_field("[__import__('os')]")


# ---------------------------------------------------------------------------------------------------------
def negctl_row(tmp: str, **kw) -> dict:
    cfg = os.path.join(tmp, "cfg")
    write(os.path.join(cfg, "vnncomp25/toy.yaml"), "a: 1\n")
    onnx = write(os.path.join(tmp, "m.onnx"), "x")
    vnn = write(os.path.join(tmp, "p.vnnlib"), "x")
    base = dict(tag="NEG_t_1", kind="NEGCTL", tier="frozen-dual", year=2026, category="toy", onnx=onnx, vnnlib=vnn,
                budget=100, configs_dir=cfg, preset="vnncomp25/toy.yaml", truth="sat", bank="-", index=0, reps_c=1,
                reps_p=1, need=L.NEED_SEP.join(L.TIERS["frozen-dual"].need),
                alarm=L.ALARM_SEP.join(L.TIERS["frozen-dual"].alarm + (r"\[frozen-dual\] class=7 .*\bverdict=closed\b",)),
                flight=L.FLIGHT_SEP.join(("lever:NY_UPFRONT_ATTACK=false@config", "lever:NY_POSTBAB_RESTARTS=0@*",
                                          L.NEGCTL_FLIGHT[-2], L.NEGCTL_FLIGHT[-1])),
                group="negctl/frozen-dual")
    base.update(kw)
    return L.spec_row(**base)


def raw_log(row: dict, profile: str, verdict: str, tier: bool = True, extra: str = "") -> str:
    lines = [f"[ny-box] cmd: env probe vnncomp v1 --proof-profile {profile} [proof-profile] mode={profile} source=cli",
             f"Running VNN-COMP instance: category='{row['category']}' onnx='{row['onnx']}' vnnlib='{row['vnnlib']}'",
             f"Loading preset: {os.path.join(row['configs_dir'], row['preset'])}",
             f"[proof-profile] mode={profile} source=cli"]
    if tier:
        lines.append("[frozen-dual] class=3 state_gens=9 nodes=12 verdict=open reason=cap secs=0.1")
        lines.append("[frozen-dual] armed=1 objectives=1 closed=0 handed_back=1 refused=0 nodes=12 depth_max=3 "
                     "secs=0.1 budget_secs=40.0 node_cap=2000000 peak_rss_mb=10 verdict=none_closed fd-block-eval/v1")
    lines += [extra, f"Result: {verdict}", f"[stop] verdict={verdict} wall_s=12.5 budget_s=100"]
    return "\n".join(lines) + "\n"


def flight(upfront="false", source="config", restarts=0, route="proof_tail_by_preset") -> dict:
    return {"levers": {"receipt": {"levers": [
        {"name": "NY_UPFRONT_ATTACK", "source": source, "value": json.loads(upfront)},
        {"name": "NY_POSTBAB_RESTARTS", "source": "default", "value": restarts}]}},
        "events": [
            {"method": "upfront_attack_policy", "reason": "schema=ny-cli/upfront-policy/v2 master=disabled_by_preset "
                                                          "route=disabled_by_preset lower_master=x"},
            {"method": "postbab_wrapper_policy", "reason": f"schema=x disabled_by_preset=1 route={route}"}]}


class Judge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.row = negctl_row(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_judge_ok_on_an_engaged_clean_control(self):
        j = L.judge_run(self.row, "competition", 1, raw_log(self.row, "competition", "timeout"), flight())
        self.assertEqual((j.status, j.verdict, j.wall), ("OK", "timeout", "12.5"), j.reasons)

    def test_judge_unsat_on_a_sat_row_is_an_emergency_even_when_the_tier_is_dark(self):
        j = L.judge_run(self.row, "proof", 1, raw_log(self.row, "proof", "unsat", tier=False), None)
        self.assertEqual(j.status, "EMERGENCY")
        self.assertTrue(any(r.startswith("TRUTH-CONTRADICTION") for r in j.reasons))

    def test_judge_void_reasons_are_specific(self):
        cases = {
            "need-absent": (raw_log(self.row, "competition", "timeout", tier=False), flight()),
            "profile-unproven": (raw_log(self.row, "competition", "timeout").replace(
                "\n[proof-profile] mode=competition source=cli", ""), flight()),
            "preset:": (raw_log(self.row, "competition", "timeout").replace("vnncomp25/toy", "vnncomp26/toy"), flight()),
            "flight-lever-source:NY_UPFRONT_ATTACK": (raw_log(self.row, "competition", "timeout"),
                                                      flight(source="default")),
            "flight-lever:NY_POSTBAB_RESTARTS": (raw_log(self.row, "competition", "timeout"), flight(restarts=2)),
            "flight-lever:NY_UPFRONT_ATTACK": (raw_log(self.row, "competition", "timeout"), flight(upfront="0")),
            "flight-event-absent:postbab_wrapper_policy": (raw_log(self.row, "competition", "timeout"),
                                                           flight(route="attack_enabled")),
            "flight-record-missing": (raw_log(self.row, "competition", "timeout"), None),
            "no-Result-line": (raw_log(self.row, "competition", "timeout").replace("Result: timeout", ""), flight()),
            "wrong-or-missing-instance-line": (raw_log(self.row, "competition", "timeout").replace("p.vnnlib", "q.vnnlib"),
                                               flight()),
        }
        for want, (raw, fl) in cases.items():
            j = L.judge_run(self.row, "competition", 1, raw, fl)
            self.assertEqual(j.status, "VOID", want)
            self.assertTrue(any(r.startswith(want) for r in j.reasons), (want, j.reasons))

    def test_judge_ny_box_cmd_line_cannot_satisfy_a_check(self):
        raw = raw_log(self.row, "proof", "timeout").replace("\n[proof-profile] mode=proof source=cli", "")
        j = L.judge_run(self.row, "proof", 1, raw, flight())
        self.assertEqual(j.status, "VOID")

    def test_judge_tier_alarms(self):
        closed = "[frozen-dual] armed=1 objectives=3 closed=3 handed_back=0 refused=0 nodes=9 verdict=all_closed x"
        j = L.judge_run(self.row, "competition", 1, raw_log(self.row, "competition", "timeout", extra=closed), flight())
        self.assertEqual(j.status, "EMERGENCY")
        cls = "[frozen-dual] class=7 state_gens=9 nodes=4 verdict=closed reason=none secs=0.1"
        j = L.judge_run(self.row, "competition", 1, raw_log(self.row, "competition", "timeout", extra=cls), flight())
        self.assertEqual(j.status, "EMERGENCY")
        other = cls.replace("class=7", "class=8")
        j = L.judge_run(self.row, "competition", 1, raw_log(self.row, "competition", "timeout", extra=other), flight())
        self.assertEqual(j.status, "OK", j.reasons)
        mr = "margin-row BaB (concurrent): UNSAT in 3.2s"
        j = L.judge_run(self.row, "competition", 1, raw_log(self.row, "competition", "timeout", extra=mr), flight())
        self.assertEqual(j.status, "EMERGENCY")

    def test_fwd_root_alarm_fires_only_when_every_clause_is_refuted(self):
        ere = L.TIERS["fwd-root-screen"].alarm[0]
        self.assertTrue(re.search(ere, "[fwd-root] union screen_refuted=56 precheck_refuted=skipped union_refuted=56 "
                                       "new_by_screen=56 total=56"))
        self.assertFalse(re.search(ere, "[fwd-root] union screen_refuted=2 precheck_refuted=0 union_refuted=2 "
                                        "new_by_screen=2 total=56"))
        self.assertFalse(re.search(ere, "[fwd-root] union screen_refuted=5 precheck_refuted=0 union_refuted=5 "
                                        "new_by_screen=5 total=56"))

    def test_judge_bank_flip_lost_and_converted(self):
        row = dict(self.row, kind="MOAT", truth="-", bank="unsat", need="-", alarm="-", flight="-", tier="-")
        self.assertEqual(L.judge_run(row, "competition", 1, raw_log(row, "competition", "sat", False), None).status,
                         "EMERGENCY")
        j = L.judge_run(row, "competition", 1, raw_log(row, "competition", "timeout", False), None)
        self.assertEqual((j.status, j.outcome), ("OK", "lost"))
        row2 = dict(row, bank="timeout")
        j = L.judge_run(row2, "competition", 1, raw_log(row, "competition", "unsat", False), None)
        self.assertEqual((j.status, j.outcome), ("OK", "converted"))

    def test_profiles_flip_split_and_proof_slower(self):
        def J(p, k, v, s="OK"):
            return L.Judgement(self.row["tag"], p, k, v, s, "n/a", "1", [])
        st = lambda runs: [s for s, _ in L.judge_profiles(self.row, runs)]  # noqa: E731
        self.assertEqual(st([J("competition", 1, "unsat"), J("proof", 1, "unsat"), J("competition", 2, "unsat"),
                             J("competition", 3, "timeout")]), [])
        self.assertIn("EMERGENCY", st([J("competition", 1, "unsat"), J("proof", 1, "sat")]))
        self.assertIn("EMERGENCY", st([J("competition", 1, "unsat"), J("competition", 2, "sat")]))
        self.assertIn("EMERGENCY", st([J("proof", 1, "unsat"), J("competition", 1, "timeout"),
                                       J("competition", 2, "unsat"), J("competition", 3, "timeout")]))
        self.assertEqual(st([J("proof", 1, "timeout"), J("competition", 1, "unsat")]), ["NOTE"])
        self.assertEqual(st([J("proof", 1, "unsat"), J("competition", 1, "unsat", "VOID")]), ["NOTE"])

    def _moat(self, **kw):
        return dict(self.row, **dict(dict(kind="MOAT", truth="-", bank="-", need="-", alarm="-", flight="-",
                                          tier="-"), **kw))

    def test_judge_reads_the_scored_results_file_as_the_verdict_of_record(self):
        # the log says timeout, the scored file says unsat: on a SAT row that is an emergency, not a timeout
        raw = raw_log(self.row, "competition", "timeout")
        j = L.judge_run(self.row, "competition", 1, raw, flight(), "unsat\n", 0)
        self.assertEqual((j.status, j.verdict), ("EMERGENCY", "unsat"), j.reasons)
        self.assertTrue(any(r.startswith("TRUTH-CONTRADICTION(unsat") for r in j.reasons), j.reasons)
        self.assertFalse(any(r.startswith("SELF-CONTRADICTION") for r in j.reasons), j.reasons)
        # agreement is OK; a non-contradicting disagreement is VOID (the log cannot vouch for the scored file)
        self.assertEqual(L.judge_run(self.row, "competition", 1, raw, flight(), "timeout\n", 0).status, "OK")
        j = L.judge_run(self.row, "competition", 1, raw, flight(), "unknown\n", 0)
        self.assertEqual(j.status, "VOID")
        self.assertTrue(any(r.startswith("res-log-mismatch(res=unknown,log=timeout)") for r in j.reasons), j.reasons)
        # a missing results file is VOID -- and a decided log line still counts as a claim
        j = L.judge_run(self.row, "competition", 1, raw, flight(), "", 0)
        self.assertEqual(j.status, "VOID")
        self.assertIn("results-file-missing-or-empty", j.reasons)
        j = L.judge_run(self.row, "competition", 1, raw_log(self.row, "competition", "unsat"), flight(), "", 0)
        self.assertEqual(j.status, "EMERGENCY", j.reasons)

    def test_judge_sat_and_unsat_in_one_run_is_a_self_contradiction(self):
        row = self._moat()  # no truth, no bank: only the run itself can show it
        raw = raw_log(row, "competition", "sat", False)
        j = L.judge_run(row, "competition", 1, raw, None, "unsat\n", 0)
        self.assertEqual(j.status, "EMERGENCY")
        self.assertTrue(any(r.startswith("SELF-CONTRADICTION") for r in j.reasons), j.reasons)
        raw2 = raw.replace("Result: sat", "Result: unsat\nResult: sat")
        self.assertEqual(L.judge_run(row, "competition", 1, raw2, None).status, "EMERGENCY")
        # two distinct non-contradicting tokens: VOID, never read as the last one
        raw3 = raw.replace("Result: sat", "Result: unknown\nResult: timeout")
        j = L.judge_run(row, "competition", 1, raw3, None)
        self.assertEqual(j.status, "VOID")
        self.assertTrue(any(r.startswith("multiple-Result-tokens") for r in j.reasons), j.reasons)

    def test_judge_backstop_kill_scores_the_results_file(self):
        row = self._moat(bank="unsat")
        raw = raw_log(row, "competition", "timeout", False)
        raw = raw.replace("Result: timeout\n", "").replace("[stop] verdict=timeout wall_s=12.5 budget_s=100\n", "")
        j = L.judge_run(row, "competition", 1, raw, None, "unknown\n", L.BACKSTOP_RC)
        self.assertEqual((j.status, j.verdict, j.outcome, j.wall), ("OK", "unknown", "lost", "backstop"), j.reasons)
        self.assertIn("note:killed-at-backstop(scored=unknown)", j.reasons)
        j = L.judge_run(row, "competition", 1, raw, None, "unsat\n", L.BACKSTOP_RC)
        self.assertEqual((j.status, j.outcome), ("OK", "held"))
        # the same missing Result line WITHOUT a backstop kill (a crash) proves nothing
        j = L.judge_run(row, "competition", 1, raw, None, "unsat\n", 139)
        self.assertEqual(j.status, "VOID")
        self.assertTrue(any(r.startswith("no-Result-line") for r in j.reasons), j.reasons)

    def test_judge_voids_a_run_whose_own_flight_saw_a_verdict_env(self):
        fl = flight()
        fl["ambient_env"] = {"OMP_NUM_THREADS": "1"}
        self.assertEqual(L.judge_run(self.row, "competition", 1, raw_log(self.row, "competition", "timeout"),
                                     fl).status, "OK")
        for amb in ({"OMP_NUM_THREADS": "1", "NY_MARGIN_ROW_FROZEN_DUAL_BAB": "1"},
                    {"OMP_NUM_THREADS": "1", "RAYON_NUM_THREADS": "4"}, {"OMP_NUM_THREADS": "8"}):
            fl["ambient_env"] = amb
            j = L.judge_run(self.row, "competition", 1, raw_log(self.row, "competition", "timeout"), fl)
            self.assertEqual(j.status, "VOID", amb)
            self.assertTrue(any(r.startswith("ambient-env:") for r in j.reasons), j.reasons)

    def test_judge_self_contradiction_alarms_fire_on_every_row(self):
        row = self._moat()
        for line in ("margin-row BaB: EARLY EXIT REFUSED — this instance holds an ONNX-Runtime-reconfirmed "
                     "counterexample, and a constructive witness outranks a bound-based proof",
                     "vnncomp: REFUSING to publish `unsat` — this instance holds an ONNX-Runtime-reconfirmed "
                     "counterexample whose witness could not be published (#sat-precedence)"):
            j = L.judge_run(row, "competition", 1, raw_log(row, "competition", "unknown", False, extra=line), None)
            self.assertEqual(j.status, "EMERGENCY", line)
        fl = {"events": [{"method": "sat_precedence",
                          "reason": "lane UNSAT at 3.0s REFUSED publication: a confirmed counterexample already exists"}]}
        j = L.judge_run(row, "competition", 1, raw_log(row, "competition", "sat", False), fl)
        self.assertEqual(j.status, "EMERGENCY", j.reasons)
        fl["events"][0]["reason"] = "lane UNSAT at 3.0s DEFERRED: a counterexample candidate was under adjudication"
        self.assertEqual(L.judge_run(row, "competition", 1, raw_log(row, "competition", "sat", False), fl).status, "OK")

    def test_judge_sat_row_alarms_apply_to_moat_rows(self):
        lane = "margin-row BaB (concurrent): UNSAT in 3.1s (root_bound=0.1)"
        for kw in (dict(truth="sat"), dict(bank="sat")):
            row = self._moat(**kw)
            j = L.judge_run(row, "competition", 1, raw_log(row, "competition", "timeout", False, extra=lane), None)
            self.assertEqual(j.status, "EMERGENCY", (kw, j.reasons))
        row = self._moat(truth="unsat")
        j = L.judge_run(row, "competition", 1, raw_log(row, "competition", "timeout", False, extra=lane), None)
        self.assertEqual(j.status, "OK", j.reasons)

    def test_profiles_flip_counts_void_runs_that_name_the_instance(self):
        def J(p, k, v, s="OK", reasons=()):
            return L.Judgement(self.row["tag"], p, k, v, s, "n/a", "1", list(reasons))
        st = lambda runs: [s for s, _ in L.judge_profiles(self.row, runs)]  # noqa: E731
        self.assertIn("EMERGENCY", st([J("competition", 1, "unsat"),
                                       J("proof", 1, "sat", "VOID", ["need-absent:x"])]))
        self.assertIn("EMERGENCY", st([J("competition", 1, "sat", "VOID", ["flight-record-missing"]),
                                       J("competition", 2, "unsat", "VOID", ["preset:x"])]))
        # a run that cannot be tied to the instance is not a claim about it
        self.assertNotIn("EMERGENCY", st([J("competition", 1, "unsat"),
                                          J("proof", 1, "sat", "VOID", ["wrong-or-missing-instance-line"])]))
        # a superseded line still stated what it stated
        self.assertIn("EMERGENCY", st([J("competition", 1, "sat", "VOID", ["need-absent:x"]),
                                       J("competition", 1, "unsat")]))

    def test_instances_flip_across_tags_is_an_emergency(self):
        a = self.row
        b = dict(self.row, tag="AB_x", kind="AB", configs_dir=self.row["configs_dir"])
        c = dict(self.row, tag="KNIFE_x", kind="KNIFE", vnnlib=write(os.path.join(self.tmp.name, "k.vnnlib"), "x"))
        rows = {r["tag"]: r for r in (a, b, c)}
        J = lambda t, v, s="OK": L.Judgement(t, "competition", 1, v, s, "n/a", "1", [])  # noqa: E731
        out = L.judge_instances(rows, [J(a["tag"], "sat"), J("AB_x", "unsat", "VOID")])
        self.assertEqual([s for s, _ in out], ["EMERGENCY"])
        self.assertIn("INSTANCE-FLIP", out[0][1])
        self.assertEqual(L.judge_instances(rows, [J(a["tag"], "sat"), J("KNIFE_x", "unsat")]), [])  # other property
        self.assertEqual(L.judge_instances(rows, [J(a["tag"], "sat"), J("AB_x", "sat")]), [])

    def test_summary_keeps_an_emergency_sticky_over_a_rejudge(self):
        e = L.Judgement(self.row["tag"], "competition", 1, "unsat", "EMERGENCY", "n/a", "1", ["TRUTH-CONTRADICTION(x)"])
        ok = L.Judgement(self.row["tag"], "competition", 1, "timeout", "OK", "n/a", "1", [])
        self.assertEqual(L.collapse_runs([e, ok])[(self.row["tag"], "competition", 1)].status, "EMERGENCY")
        self.assertEqual(L.collapse_runs([ok, e])[(self.row["tag"], "competition", 1)].status, "EMERGENCY")
        v = L.Judgement(self.row["tag"], "competition", 1, "timeout", "VOID", "n/a", "1", ["need-absent:x"])
        self.assertEqual(L.collapse_runs([v, ok])[(self.row["tag"], "competition", 1)].status, "OK")
        spec = os.path.join(self.tmp.name, "s.tsv")
        L.write_spec(spec, [dict(self.row, reps_p="0")])
        judged = write(os.path.join(self.tmp.name, "judged.tsv"),
                       "\t".join(L.JUDGED_COLUMNS) + "\n" + e.line() + "\n" + ok.line() + "\n")
        import m0_judge as JG
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = JG.main(["summary", "--spec", spec, "--judged", judged])
        self.assertEqual(rc, 3, out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(JG.main(["prior", "--spec", spec, "--judged", judged]), 3)

    def test_spec_round_trip_and_validation(self):
        p = os.path.join(self.tmp.name, "s.tsv")
        L.write_spec(p, [self.row])
        self.assertEqual(L.read_spec(p), [self.row])
        with self.assertRaises(L.M0Error):
            L.write_spec(p, [self.row])
        with self.assertRaises(L.M0Error):
            L.write_spec(p, [self.row], append=True)
        bad = [dict(self.row, truth="unsat"), dict(self.row, tag="x y"), dict(self.row, tag="B", need="(unclosed"),
               dict(self.row, tag="C", flight="lever:NY_X=notjson@config"), dict(self.row, tag="D", budget="0")]
        errs = L.validate_spec(bad)
        for frag in ("truly-SAT", "tag must be", "need regex", "flight check", "budget"):
            self.assertTrue(any(frag in e for e in errs), (frag, errs))


# ---------------------------------------------------------------------------------------------------------
FAKE_BIN = r'''#!/usr/bin/env bash
# stub pin for m0_selftest. Markers: Running VNN-COMP instance Loading preset: Result:  [stop] verdict=
# [proof-profile] mode= [frozen-dual] armed=1 objectives= [frozen-dual] class= margin-row BaB (concurrent): UNSAT in 1s
# EARLY EXIT REFUSED / REFUSING to publish `unsat` / [fwd-root] union screen_refuted=
if [ "${1:-}" = --build-id ]; then
  echo "ny-build commit=__COMMIT__ features=mip,cuda dirty=false exe=$(sha256sum "$0" | cut -d' ' -f1)"; exit 0
fi
if [ "$1 $2" = "vnncomp plan" ]; then
  [ -n "${M0TEST_PLANLOG:-}" ] && echo "$3|$8" >> "$M0TEST_PLANLOG"
  echo "preset: $8/__PRESET__"; echo "vnnlib: $5"; echo "scored_budget_secs = $6"; exit 0
fi
[ "$1 $2" = "vnncomp v1" ] || exit 2
C=$3; O=$4; V=$5; RES=$6; B=$7; CFG=$9; P=${11}
mode=${M0TEST_MODE:-ok}
[ "$mode" = split ] && { [ "$P" = proof ] && mode=unsat_proof || mode=ok; }
echo "$(ps -o args= -p $PPID)" >> "$(dirname "$RES")/../parents.log"
env | grep -E '^(NY_|RAYON_|GPU_AVAILABLE)' >> "$(dirname "$RES")/../verifier_env.log"
echo unknown > "$RES"   # the scored placeholder ny writes before any work
echo "Running VNN-COMP instance: category='$C' onnx='$O' vnnlib='$V' results='$RES' timeout=${B}s"
echo "Loading preset: $CFG/__PRESET__"
echo "[proof-profile] mode=$P source=cli" >&2
if [ "$mode" = hang ]; then echo "$C $P" >> "$(dirname "$RES")/../calls.log"; sleep 60; exit 0; fi
[ "$mode" = dark ] || echo "[frozen-dual] armed=1 objectives=1 closed=0 handed_back=1 refused=0 nodes=40 depth_max=3 verdict=none_closed" >&2
v=timeout; [ "$mode" = unsat ] && v=unsat; [ "$mode" = unsat_proof ] && v=unsat
echo "Result: $v"; echo "[stop] verdict=$v wall_s=1.5 budget_s=$B" >&2
if [ "$mode" = res_unsat ]; then echo unsat > "$RES"; else echo "$v" > "$RES"; fi
[ "$mode" = noflight ] && { echo "$C $P" >> "$(dirname "$RES")/../calls.log"; exit 0; }
cat > "$RES.flight.json" <<EOF
{"ambient_env": {"OMP_NUM_THREADS": "$OMP_NUM_THREADS"},
 "levers": {"receipt": {"levers": [{"name": "NY_UPFRONT_ATTACK", "source": "config", "value": false}]}},
 "events": [{"method": "upfront_attack_policy", "reason": "master=disabled_by_preset route=disabled_by_preset lower=x"}]}
EOF
echo "$C $P" >> "$(dirname "$RES")/../calls.log"
'''

FAKE_BOX = r'''#!/usr/bin/env bash
# stub ny_box.sh: `measure -- cmd...` runs cmd; prints the header line the real one prints
[ "$1" = measure ] || exit 2
shift; while [ "$#" -gt 0 ] && [ "$1" != -- ]; do shift; done; shift
echo "[ny-box] cmd: $*"
"$@"; rc=$?
echo "[ny-box-provenance] mode=measure rc=$rc elapsed=0"
exit $rc
'''

# GNU coreutils `timeout DURATION COMMAND...` contract, for hosts without `timeout` (see the module docstring).
TIMEOUT_STAND_IN = r'''
import os, signal, subprocess, sys


def main(argv):
    if len(argv) < 2:
        print("timeout (m0 selftest stand-in): usage: timeout DURATION COMMAND [ARG]...", file=sys.stderr)
        return 125
    try:
        secs = float(argv[0])
    except ValueError:
        print(f"timeout (m0 selftest stand-in): invalid duration {argv[0]!r}", file=sys.stderr)
        return 125
    try:
        os.setpgid(0, 0)  # GNU timeout without --foreground: the command runs in timeout's own process group
    except OSError:
        pass
    try:
        child = subprocess.Popen(argv[1:])
    except FileNotFoundError:
        return 127
    except PermissionError:
        return 126
    try:
        rc = child.wait(timeout=secs)
    except subprocess.TimeoutExpired:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.killpg(0, signal.SIGTERM)
        child.wait()
        return 124
    return rc if rc >= 0 else 128 - rc


sys.exit(main(sys.argv[1:]))
'''


def host_timeout() -> str | None:
    return shutil.which("timeout")


def runner_tool_dir(root: str) -> str:
    """bin/ for the runner's PATH: `python3` is this interpreter; `timeout` is the stand-in only when the host has
    none. Returns the directory to put first on the runner's PATH."""
    bindir = os.path.join(root, "bin")
    os.makedirs(bindir, exist_ok=True)
    wrapper = write(os.path.join(bindir, "python3"), f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n")
    os.chmod(wrapper, 0o755)
    if host_timeout() is None:
        src = write(os.path.join(root, "standin", "timeout"), TIMEOUT_STAND_IN)
        tmo = write(os.path.join(bindir, "timeout"),
                    f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(src)} \"$@\"\n")
        os.chmod(tmo, 0o755)
    return bindir


def cap_address_space() -> None:
    """preexec: a finite RLIMIT_AS. 64 GiB where the kernel takes it; macOS refuses (EINVAL) any cap below the
    address space the process already reserves, so take the first larger power of two it accepts."""
    import resource
    for cap in (64 << 30, 1 << 40, 1 << 44):
        try:
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            return
        except (ValueError, OSError):
            continue
    raise OSError("no finite RLIMIT_AS cap was accepted by this kernel")


class Runner(unittest.TestCase):
    """moat_both_profiles.sh --run against a stub pin and a stub ny_box.sh under a fake M0_NY."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = os.path.join(self.tmp.name, "ny")
        os.makedirs(os.path.join(root, "scripts"))
        os.makedirs(os.path.join(root, "scratchpad"))
        box = write(os.path.join(root, "scripts/ny_box.sh"), FAKE_BOX)
        os.chmod(box, 0o755)
        self.commit = "c" * 40
        self.pin = write(os.path.join(self.tmp.name, "pin"),
                         FAKE_BIN.replace("__COMMIT__", self.commit).replace("__PRESET__", "vnncomp25/toy.yaml"))
        os.chmod(self.pin, 0o755)
        self.out = os.path.join(root, "scratchpad/m0out")
        self.bindir = runner_tool_dir(self.tmp.name)
        self.planlog = os.path.join(self.tmp.name, "plans.log")
        row = negctl_row(self.tmp.name, flight=L.FLIGHT_SEP.join(("lever:NY_UPFRONT_ATTACK=false@config",
                                                                   L.NEGCTL_FLIGHT[-2])))
        self.spec = os.path.join(self.tmp.name, "spec.tsv")
        L.write_spec(self.spec, [row])

    def tearDown(self):
        self.tmp.cleanup()

    def run_runner(self, mode="ok", extra_env=None, spec=None, preexec=None):
        # The fixture owns a clean baseline; extra_env below still exercises
        # the real runner's refusal of explicitly supplied verdict settings.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("NY_", "NYCERT_", "RAYON_", "AY_MILP_", "PROBE_"))
               and k not in ("GPU_AVAILABLE", "CUDA_VISIBLE_DEVICES")}
        env.update({"M0_NY": self.root, "M0_PIN": self.pin, "M0_PIN_SHA16": L.sha256_file(self.pin)[:16],
                    "M0_PIN_COMMIT": self.commit, "M0_OUT": self.out, "M0TEST_MODE": mode,
                    "M0TEST_PLANLOG": self.planlog,
                    "PATH": self.bindir + os.pathsep + os.environ.get("PATH", os.defpath)})
        env.update(extra_env or {})
        p = subprocess.run([RUNNER_BASH, os.path.join(HERE, "moat_both_profiles.sh"), "--run", spec or self.spec],
                           env=env, capture_output=True, text=True, timeout=120, preexec_fn=preexec)
        log = open(os.path.join(self.out, "m0moat.log")).read() if os.path.isfile(os.path.join(self.out, "m0moat.log")) else ""
        done = [ln for ln in (p.stdout + log).splitlines() if ln.startswith("M0MOAT_DONE ")]
        return p.returncode, done[-1] if done else "", log

    def calls(self):
        p = os.path.join(self.out, "calls.log")
        return open(p).read().splitlines() if os.path.isfile(p) else []

    def test_runner_clean_run_exits_zero_and_done_propagates_rc(self):
        rc, done, log = self.run_runner("ok")
        self.assertEqual(rc, 0, log[-2000:])
        self.assertTrue(done.startswith("M0MOAT_DONE rc=0 WORST=0 "), done)
        self.assertEqual(self.calls(), ["toy competition", "toy proof"])
        self.assertIn("M0_SUMMARY rc=0", log)
        # the runner's timestamps keep GNU `date -Is` format on every host (BSD date has no -Is; its %z has no colon)
        self.assertRegex(done, " " + ISO_STAMP + "$")
        self.assertRegex(log, r"(?m)^=== M0MOAT heartbeat start " + ISO_STAMP + " pid=")
        self.assertRegex(log, r"(?m)^  START NEG_t_1 competition r1 \[toy\] budget=\d+s backstop=\d+s " + ISO_STAMP + "$")

    def test_runner_fixture_isolates_inherited_verdict_env(self):
        inherited = {"NY_UPFRONT_ATTACK": "true", "NYCERT_TIGHT": "1",
                     "RAYON_NUM_THREADS": "6", "AY_MILP_NO_CUTS": "1",
                     "PROBE_TEST": "1", "GPU_AVAILABLE": "1", "CUDA_VISIBLE_DEVICES": "0"}
        with mock.patch.dict(os.environ, inherited):
            rc, done, log = self.run_runner("ok")
        self.assertEqual(rc, 0, log[-2000:])
        self.assertTrue(done.startswith("M0MOAT_DONE rc=0 WORST=0 "), done)
        self.assertEqual(self.calls(), ["toy competition", "toy proof"])

    def test_runner_stops_before_a_run_while_its_stop_file_exists(self):
        stop = os.path.join(self.tmp.name, "EMERGENCY")
        rc, done, log = self.run_runner("ok", {"M0_STOP_FILE": stop})     # absent: the runner runs everything
        self.assertEqual(rc, 0, log[-2000:])
        self.assertEqual(self.calls(), ["toy competition", "toy proof"])
        write(stop, "an emergency recorded elsewhere\n")
        for f in glob.glob(os.path.join(self.out, "raw", "*.judged")):
            os.remove(f)                                                  # would re-run both, but the file exists
        rc, done, log = self.run_runner("ok", {"M0_STOP_FILE": stop})
        self.assertEqual(rc, 3, log[-2000:])
        self.assertTrue(done.startswith("M0MOAT_DONE rc=3 WORST=3 STOPPED_AT=NEG_t_1/competition/r1 reason=stop-file "),
                        done)
        self.assertRegex(done, " reason=stop-file " + ISO_STAMP + "$")    # a stamp, not an empty `date -Is` on BSD
        self.assertEqual(self.calls(), ["toy competition", "toy proof"])  # nothing new ran

    def test_runner_out_dir_must_live_under_its_root_and_the_root_is_overridable(self):
        outside = os.path.join(self.tmp.name, "elsewhere", "out")
        rc, done, _ = self.run_runner("ok", {"M0_OUT": outside})          # default root: $M0_NY/scratchpad
        self.assertEqual(rc, 2, done)
        self.assertIn("M0_OUT must live under", done)
        self.assertEqual(self.calls(), [])
        rc, done, _ = self.run_runner("ok", {"M0_OUT": outside, "M0_OUT_ROOT": "relative/root"})
        self.assertEqual(rc, 2, done)                                     # a relative root is refused
        self.assertIn("M0_OUT_ROOT must be absolute", done)
        self.out = outside
        rc, done, log = self.run_runner("ok", {"M0_OUT": outside,
                                               "M0_OUT_ROOT": os.path.join(self.tmp.name, "elsewhere")})
        self.assertEqual(rc, 0, log[-2000:])
        self.assertEqual(self.calls(), ["toy competition", "toy proof"])

    def test_runner_stops_on_an_emergency_with_rc3(self):
        rc, done, log = self.run_runner("unsat")
        self.assertEqual(rc, 3, log[-2000:])
        self.assertTrue(done.startswith("M0MOAT_DONE rc=3 WORST=3 STOPPED_AT=NEG_t_1/competition/r1"), done)
        self.assertEqual(self.calls(), ["toy competition"])  # stopped before the proof run

    def test_runner_void_run_is_not_a_pass(self):
        rc, done, log = self.run_runner("dark")
        self.assertEqual(rc, 5, log[-2000:])
        self.assertIn("void=2", done)

    def test_runner_profile_split_is_an_emergency(self):
        row = dict(L.read_spec(self.spec)[0], tag="MOAT_x", kind="MOAT", truth="-", bank="-", need="-", alarm="-",
                   flight="-", tier="-")
        spec2 = os.path.join(self.tmp.name, "spec2.tsv")
        L.write_spec(spec2, [row])
        rc, done, log = self.run_runner("split", spec=spec2)
        self.assertEqual(rc, 3, log[-2000:])
        self.assertIn("reason=profile-emergency", done)
        self.assertIn("PROFILE-DECIDED-SPLIT", log)

    def test_runner_refuses_ny_env_before_measuring(self):
        rc, done, _ = self.run_runner("ok", {"NY_SOME_LEVER": "1"})
        self.assertEqual(rc, 95)
        self.assertEqual(self.calls(), [])

    def test_runner_refuses_a_wrong_pin(self):
        rc, done, _ = self.run_runner("ok", {"M0_PIN_SHA16": "0" * 16})
        self.assertEqual(rc, 91)
        rc, done, _ = self.run_runner("ok", {"M0_PIN_COMMIT": "d" * 40})
        self.assertEqual(rc, 91)
        self.assertEqual(self.calls(), [])

    def test_runner_resume_skips_judged_runs(self):
        rc, _, _ = self.run_runner("ok")
        self.assertEqual(rc, 0)
        rc, done, _ = self.run_runner("ok")
        self.assertEqual(rc, 0)
        self.assertIn("ran=0", done)
        self.assertEqual(len(self.calls()), 2)

    def test_runner_runs_under_the_competition_backstop(self):
        rc, _, log = self.run_runner("ok")
        self.assertEqual(rc, 0, log[-2000:])
        parents = open(os.path.join(self.out, "parents.log")).read().splitlines()
        self.assertEqual(len(parents), 2, parents)
        # run_instance.sh: `timeout $((budget + 10))`, budget 100 here, with the pin's copy as its command. Where
        # the host has `timeout`, it IS the measured process's parent and the line starts with it, as this test has
        # always required. Only where the stand-in runs is the parent the interpreter running it, and then the
        # line is exactly `<interpreter> <this test's stand-in> 110 <pin> vnncomp v1 ...`. A command line that
        # merely contains `timeout 110` (a wrapper around something else) is neither.
        command = "110 " + os.path.join(self.out, "probe_m0") + " vnncomp v1 "
        standin = os.path.join(self.tmp.name, "standin", "timeout")
        for ln in parents:
            if host_timeout() is not None:
                self.assertTrue(ln.startswith("timeout " + command), ln)
            else:
                interpreter, found, rest = ln.partition(" " + standin + " ")
                self.assertTrue(found and rest.startswith(command), ln)
                self.assertTrue(os.path.isfile(interpreter) and os.access(interpreter, os.X_OK), ln)

    def test_runner_backstop_kill_is_judged_from_the_results_file(self):
        row = dict(L.read_spec(self.spec)[0], tag="MOAT_hang", kind="MOAT", truth="-", bank="unsat", need="-",
                   alarm="-", flight="-", tier="-", budget="1", reps_p="0")
        spec2 = os.path.join(self.tmp.name, "spec_hang.tsv")
        L.write_spec(spec2, [row])
        rc, done, log = self.run_runner("hang", spec=spec2)  # killed at 1 + 10 s, with only the placeholder written
        self.assertEqual(rc, 4, log[-2000:])  # the banked unsat is LOST, not VOID and not held
        self.assertIn("box_rc=124", log)
        self.assertIn("note:killed-at-backstop(scored=unknown)", log)

    def test_runner_results_file_unsat_is_an_emergency_when_the_log_says_timeout(self):
        rc, done, log = self.run_runner("res_unsat")
        self.assertEqual(rc, 3, log[-2000:])
        self.assertIn("TRUTH-CONTRADICTION(unsat", log)

    def test_runner_refuses_to_relaunch_after_an_emergency(self):
        rc, _, _ = self.run_runner("unsat")
        self.assertEqual(rc, 3)
        rc, done, log = self.run_runner("ok")
        self.assertEqual(rc, 3, log[-2000:])
        self.assertIn("reason=prior-emergency", done)
        self.assertEqual(self.calls(), ["toy competition"])  # nothing measured after the emergency

    def test_runner_refuses_verdict_affecting_non_ny_env(self):
        for var in ("GPU_AVAILABLE", "RAYON_NUM_THREADS", "AY_MILP_NO_CUTS", "NYCERT_TIGHT", "CUDA_VISIBLE_DEVICES", "PROBE_TEST"):
            rc, done, _ = self.run_runner("ok", {var: "1"})
            self.assertEqual(rc, 95, var)
        self.assertEqual(self.calls(), [])
        rc, _, log = self.run_runner("ok", {"NY_BOX_LOAD_MAX": "9"})  # the box's own knobs are not the verifier's
        self.assertEqual(rc, 0, log[-2000:])
        seen = open(os.path.join(self.out, "verifier_env.log")).read() \
            if os.path.isfile(os.path.join(self.out, "verifier_env.log")) else ""
        self.assertEqual(seen, "", "the measured process saw: " + seen)  # every NY_BOX_* stripped, not only three

    def test_runner_refuses_an_address_space_rlimit(self):
        # any finite cap: on Linux ny's memory governor reads it; the runner refuses it on every host
        rc, done, _ = self.run_runner("ok", preexec=cap_address_space)
        self.assertEqual(rc, 86, done)
        self.assertIn("rlimit-as-set", done)
        self.assertEqual(self.calls(), [])

    def test_runner_resolver_checks_each_category_configs_preset_key_once(self):
        # step 0 resolver-checks one row per (category, configs dir, preset). Two rows share a key and one does
        # not: exactly two plan calls, and every row still runs. This is the dedup bash 3.2 cannot do with an
        # associative array (DEFECT_REGISTER D1: `declare -A` collapses to an indexed array there).
        base = L.read_spec(self.spec)[0]
        moat = dict(base, kind="MOAT", truth="-", bank="-", need="-", alarm="-", flight="-", tier="-", reps_p="0")
        rows = [dict(moat, tag="MOAT_a"), dict(moat, tag="MOAT_b", index="1"), dict(moat, tag="MOAT_c", category="toy2")]
        spec3 = os.path.join(self.tmp.name, "spec_keys.tsv")
        L.write_spec(spec3, rows)
        rc, done, log = self.run_runner("ok", spec=spec3)
        self.assertEqual(rc, 0, log[-2000:])
        cfg = base["configs_dir"]
        self.assertEqual(open(self.planlog).read().splitlines(), [f"toy|{cfg}", f"toy2|{cfg}"])
        self.assertIn("plans ok (2 resolver checks over (category, configs dir)", log)
        self.assertEqual(self.calls(), ["toy competition", "toy competition", "toy2 competition"])

    def test_runner_moves_a_stale_flight_record_aside(self):
        # an interrupted earlier attempt left a PASSING flight record at this run's path; this attempt writes none
        res = os.path.join(self.out, "res", "NEG_t_1.competition.r1.res")
        write(res + ".flight.json", json.dumps(flight()))
        rc, done, log = self.run_runner("noflight")
        self.assertEqual(rc, 5, log[-2000:])
        self.assertIn("flight-record-missing", log)
        stale = [f for f in os.listdir(os.path.join(self.out, "res")) if ".flight.json.stale." in f]
        self.assertEqual(len(stale), 1, stale)


class RealData(unittest.TestCase):
    """The `--real-data` lane: tests that read the main tree's benchmarks/, reports/ and the field results.

    They are absent from the default run rather than skipped in it (TEST_CONFORMANCE.md; the source policy
    forbids skips), and selecting the lane makes every missing input a FAILURE that names the path.
    """

    def require(self, *paths):
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            self.fail("--real-data lane: required input(s) absent (VOID, never a pass): " + ", ".join(missing))

    def test_knife_edge_on_the_real_cgan_r4_witness(self):
        v = os.path.join(NY_ROOT, "benchmarks/vnncomp2026/benchmarks/cgan2026/1.0/vnnlib/"
                                  "cGAN_imgSz32_nCh_3_prop_0_input_eps_0.070_output_eps_0.078.vnnlib")
        w = os.path.join(REPO, "reports/measured-2026/cgan2026_artifacts/witnesses/"
                               "0004-cGAN_imgSz32_nCh_3_prop_0_input_eps_0.070_output_eps_0.078.counterexample")
        self.require(v, w)
        with tempfile.TemporaryDirectory() as d:
            man = KE.build_ladder(v, w, d, list(L.DEFAULT_DELTAS))
            for rung in man["rungs"]:
                self.assertTrue(rung["knife_edge_exact"])
                self.assertEqual(rung["clauses_rewritten"], [0])

    def test_negative_controls_end_to_end_on_the_repo_configs(self):
        self.require(*REAL_INPUTS)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "negctl")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = NC.main(["--configs", os.path.join(REPO, "configs"), "--out", out, "--ny-root", NY_ROOT,
                              "--truth-2026", TRUTH26] + sum((["--tier", t] for t in L.TIERS), []))
            self.assertEqual(rc, 0)
            rows = L.read_spec(os.path.join(out, "negctl_spec.tsv"))
            self.assertEqual(len(rows), 14)
            self.assertEqual(L.validate_spec(rows), [])
            self.assertTrue(all(r["truth"] == "sat" and r["kind"] == "NEGCTL" for r in rows))
            cif = L.strict_load(open(os.path.join(out, "cfg_negctl_frozen-dual/vnncomp25/cifar100_2024.yaml")).read())
            self.assertEqual((cif["attack"]["pgd_order"], cif["attack"]["upfront_attack"]), ("skip", False))
            self.assertIs(cif["margin_row"]["frozen_dual_bab"], True)
            self.assertNotIn("disjunctive_pgd_min_secs", cif["bab"]["phase_budget"])
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(NC.main(["--configs", os.path.join(REPO, "configs"), "--out", out,
                                          "--tier", "frozen-dual", "--truth-2026", TRUTH26]), 2)

    def test_moat_default_carries_tiny_1831_and_5529_on_the_real_banks(self):
        self.require(*REAL_INPUTS)
        import moat_rows as MR
        truth = {2026: TRUTH26, 2025: TRUTH26}
        rows, _ = MR.build(NY_ROOT, os.path.join(REPO, "configs"), [2026], {"tinyimagenet_2024"}, truth, False, 1, 0,
                           "MOAT")
        picked = {os.path.basename(r["vnnlib"]).split("_prop_idx_")[1].split("_")[0]: r for r in rows}
        for idx in ("1831", "5529"):
            self.assertIn(idx, picked)
            self.assertEqual((picked[idx]["truth"], picked[idx]["bank"]), ("sat", "timeout"))
        rows_b, _ = MR.build(NY_ROOT, os.path.join(REPO, "configs"), [2026], {"tinyimagenet_2024"}, truth, False, 1, 0,
                             "MOAT", banked_only=True)
        self.assertEqual(len(rows) - len(rows_b), 2)


def main(argv=None) -> int:
    global RUNNER_BASH
    args = sys.argv[1:] if argv is None else list(argv)
    real_data = False
    while args:
        if args[0] == "--real-data":
            real_data, args = True, args[1:]
        elif args[0] == "--bash" and len(args) >= 2:
            RUNNER_BASH, args = args[1], args[2:]
        else:
            print("usage: m0_selftest.py [--real-data] [--bash PATH]", file=sys.stderr)
            return 2
    bash = shutil.which(RUNNER_BASH)
    if bash is None:
        print(f"m0_selftest.py: --bash {RUNNER_BASH}: no such executable (the runner tests need it)", file=sys.stderr)
        return 2
    RUNNER_BASH = bash
    bash_version = subprocess.run([bash, "-c", 'printf %s "$BASH_VERSION"'], capture_output=True, text=True,
                                  timeout=30).stdout.strip() or "?"
    lane = "real-data" if real_data else "default"
    loader = unittest.defaultTestLoader
    if real_data:
        suite = loader.loadTestsFromTestCase(RealData)
    else:
        cases = [case for case in vars(sys.modules[__name__]).values()
                 if isinstance(case, type) and issubclass(case, unittest.TestCase) and case is not RealData]
        suite = unittest.TestSuite(loader.loadTestsFromTestCase(case) for case in cases)
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    res = runner.run(suite)
    sys.stdout.write(stream.getvalue())
    rc = 0 if res.wasSuccessful() and res.testsRun > 0 else 1
    print(f"M0_SELFTEST_DONE rc={rc} ran={res.testsRun} failures={len(res.failures)} errors={len(res.errors)} "
          f"skipped={len(res.skipped)} lane={lane} bash={bash} bash_version={bash_version} "
          f"timeout={host_timeout() or 'stand-in'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
