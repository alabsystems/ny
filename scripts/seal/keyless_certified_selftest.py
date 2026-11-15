#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of keyless_certified's AMENDMENT 2 (the stratified sample): allocation, the zero-event bounds, the
sample record's locks and bindings, the effective plan, and the global cross-tag check's coverage. Synthetic plan
directories only (no box, no pin, no benchmark files), created under $TMPDIR and never deleted (the pipeline never
deletes; point TMPDIR at a scratch directory):

  TMPDIR=<scratch> python3 scripts/seal/keyless_certified_selftest.py -v
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import keyless_certified as KC  # noqa: E402
from seallib import L  # noqa: E402


def wr(path: str, text: str, mode: str = "w") -> None:
    with open(path, mode) as fh:
        fh.write(text)


def rd(path: str) -> str:
    with open(path) as fh:
        return fh.read()


def jload(path: str):
    with open(path) as fh:
        return json.load(fh)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a, **k)
    return rc, buf.getvalue()


class Allocation(unittest.TestCase):
    def test_floor_judged_and_capacity(self):
        sizes = {"2026/a/sat-replay": 100, "2026/b/sat-replay": 2, "2026/c/sat-replay": 10}
        judged = {"2026/a/sat-replay": 5, "2026/c/sat-replay": 3}
        w = {h: 1.0 for h in sizes}
        new = KC.allocate(sizes, judged, w, 3, 20)
        self.assertEqual(sum(new.values()), 20)
        self.assertEqual(new["2026/b/sat-replay"], 2)                  # capped at the stratum size (< floor)
        for h in sizes:
            self.assertGreaterEqual(judged.get(h, 0) + new[h], min(3, sizes[h]))
            self.assertLessEqual(judged.get(h, 0) + new[h], sizes[h])

    def test_floor_wins_over_target(self):
        sizes = {f"2025/c{i}/undecided": 50 for i in range(10)}
        new = KC.allocate(sizes, {}, {h: 1.0 for h in sizes}, 3, 12)
        self.assertEqual(set(new.values()), {3})                        # 30 > 12: the floor is kept, reported

    def test_engaged_weight_triples_the_share(self):
        sizes = {"2025/x/unsat-reprove": 1000, "2025/y/unsat-reprove": 1000}
        w = {"2025/x/unsat-reprove": 3.0, "2025/y/unsat-reprove": 1.0}
        new = KC.allocate(sizes, {}, w, 0, 400)
        self.assertEqual(sum(new.values()), 400)
        self.assertAlmostEqual(new["2025/x/unsat-reprove"] / new["2025/y/unsat-reprove"], 3.0, delta=0.05)

    def test_judged_rows_count_toward_the_share(self):
        sizes = {"2025/x/sat-replay": 500, "2025/y/sat-replay": 500}
        judged = {"2025/x/sat-replay": 40}
        new = KC.allocate(sizes, judged, {h: 1.0 for h in sizes}, 0, 60)
        self.assertEqual(judged["2025/x/sat-replay"] + new["2025/x/sat-replay"], 50)
        self.assertEqual(new["2025/y/sat-replay"], 50)

    def test_everything_exhausted_stops(self):
        sizes = {"2025/x/sat-replay": 4}
        new = KC.allocate(sizes, {"2025/x/sat-replay": 1}, {"2025/x/sat-replay": 1.0}, 3, 100)
        self.assertEqual(new["2025/x/sat-replay"], 3)

    def test_engaged_weight_never_on_sat(self):
        self.assertEqual(KC.stratum_weight("2025/u/sat-replay", {"2025/u"}, 3), 1.0)
        self.assertEqual(KC.stratum_weight("2025/u/unsat-reprove", {"2025/u"}, 3), 3.0)
        self.assertEqual(KC.stratum_weight("2025/u/undecided", {"2025/u"}, 3), 3.0)
        self.assertEqual(KC.stratum_weight("2025/v/undecided", {"2025/u"}, 3), 1.0)

    def test_predicted_class(self):
        self.assertEqual(KC.predicted_class({"bank": "unsat"}), "unsat-reprove")
        self.assertEqual(KC.predicted_class({"bank": "sat"}), "sat-replay")
        for b in ("timeout", "unknown", "-"):
            self.assertEqual(KC.predicted_class({"bank": b}), "undecided")


class Bounds(unittest.TestCase):
    def test_single_stratum_is_clopper_pearson(self):
        for n in (3, 30, 340):
            self.assertAlmostEqual(KC.stratified_zero_event_bound([(1.0, n, False)]), KC.zero_event_bound(n),
                                   places=9)
        self.assertAlmostEqual(KC.zero_event_bound(300), 1 - 0.05 ** (1 / 300))
        self.assertLess(abs(KC.zero_event_bound(300) - 3 / 300), 0.0003)

    def test_exhaustive_and_unsampled(self):
        self.assertEqual(KC.stratified_zero_event_bound([(0.5, 10, True), (0.5, 0, False)]), 0.5)
        b = KC.stratified_zero_event_bound([(0.5, 10, True), (0.5, 100, False)])
        self.assertAlmostEqual(b, 0.5 * KC.zero_event_bound(100), places=9)

    def test_worst_case_is_at_least_every_single_stratum_placement(self):
        st = [(0.6, 30, False), (0.3, 5, False), (0.1, 3, False)]
        b = KC.stratified_zero_event_bound(st)
        for w, n, _ in st:
            self.assertGreaterEqual(b + 1e-12, w * KC.zero_event_bound(n))
        self.assertLessEqual(b, 1.0)

    def test_more_rows_tighter(self):
        a = KC.stratified_zero_event_bound([(0.7, 20, False), (0.3, 5, False)])
        b = KC.stratified_zero_event_bound([(0.7, 40, False), (0.3, 10, False)])
        self.assertLess(b, a)

    def test_claim_with_an_event_has_no_bound(self):
        c = KC.sample_claim({"2025/a/sat-replay": 10}, {"2025/a/sat-replay": 5}, 1)
        self.assertNotIn("bound_rows", c)
        self.assertIn("NO bound", c["text"])


ROWS = ([("2025", "alpha", "sat", "sat")] * 30 + [("2025", "alpha", "unsat", "unsat")] * 20
        + [("2025", "alpha", "timeout", "unsat")] * 10 + [("2026", "beta", "unsat", "unsat")] * 5
        + [("2026", "beta", "sat", "sat")] * 2)


class SampleEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kc_selftest_")
        self.out = os.path.join(self.tmp, "kc")
        os.makedirs(os.path.join(self.out, "specs", "1of1"))
        cfg = os.path.join(self.tmp, "configs")
        os.makedirs(cfg)
        model = os.path.join(self.tmp, "m.onnx")
        wr(model, "x")
        rows = []
        for i, (y, cat, bank, truth) in enumerate(ROWS):
            v = os.path.join(self.tmp, f"p{i}.vnnlib")
            wr(v, str(i))
            rows.append(L.spec_row(tag=f"S8_{int(y) % 100}_{cat}_{i:04d}", kind="MOAT", tier="-", year=y,
                                   category=cat, onnx=model, vnnlib=v, budget=20, configs_dir=cfg, preset="-",
                                   truth=truth, bank=bank, index=i, reps_c=1, reps_p=0, need="-", alarm="-",
                                   flight="-", group=f"seal/S8/{y}", note="-"))
        self.rows = rows
        batches, files = [], {}
        for k in range(0, len(rows), 10):
            bid = f"b{k // 10 + 1:04d}"
            rel = os.path.join("specs", "1of1", f"{bid}.tsv")
            L.write_spec(os.path.join(self.out, rel), rows[k:k + 10])
            files[rel] = L.sha256_file(os.path.join(self.out, rel))
            batches.append({"id": bid, "spec": rel, "rows": len(rows[k:k + 10]), "stages": {"S8": 10}})
        self.plan = {"schema": KC.PLAN_SCHEMA, "capability": "prepared-root", "out": self.out,
                     "pin": {"path": "/nonexistent/pin", "sha256": "ab" * 32, "commit": "c" * 40,
                             "features": "mip,cuda"},
                     "shards": {"1of1": {"rows": len(rows), "batches": batches}},
                     "rows_index": {r["tag"]: {"stage": "S8", "shard": "1of1"} for r in rows}, "files": files}
        text = json.dumps(self.plan, indent=1, sort_keys=True) + "\n"
        wr(os.path.join(self.out, "plan.json"), text)
        wr(os.path.join(self.out, "plan.json.sha256"), hashlib.sha256(text.encode()).hexdigest() + "\n")
        # b0001 judged (final, PASS) with one engaged row on 2025/alpha
        b1 = rows[:10]
        rec = {"schema": KC.BATCH_SCHEMA, "batch": "b0001", "status": "PASS", "rc": 0, "final": True,
               "emergencies": [], "void": [], "lost": [],
               "rows": [{"tag": r["tag"], "year": r["year"], "category": r["category"], "bank": r["bank"],
                         "truth": r["truth"], "verdict": r["bank"], "status": "OK", "outcome": "held",
                         "route": {"status": "ran" if j == 0 else "absent", "route": "Open" if j == 0 else "-"}}
                        for j, r in enumerate(b1)]}
        self.b1 = os.path.join(self.out, "run", "1of1", "b0001", "BATCH.json")
        os.makedirs(os.path.dirname(self.b1))
        wr(self.b1, json.dumps(rec))
        self.sample = os.path.join(self.out, "sample1.json")

    def draw(self, *extra):
        return quiet(KC.main, ["sample", "--out", self.out, "--target-new", "20", "--batch-rows", "8",
                               "--ny-root", self.tmp, *extra])

    def test_draw_lock_and_effective_plan(self):
        rc, txt = self.draw()
        self.assertEqual(rc, 0, txt)
        self.assertIn("KC_SAMPLE_DONE rc=0", txt)
        s = jload(self.sample)
        self.assertEqual(s["engaged"]["units"], ["2025/alpha"])
        self.assertEqual(s["shards"]["1of1"]["carried"], ["b0001"])
        self.assertEqual(sum(v["new"] for v in s["strata"].values()), 20)
        for h, v in s["strata"].items():
            self.assertGreaterEqual(v["n"], min(3, v["N"]), h)
            self.assertEqual(v["weight"], 1.0 if h.endswith("sat-replay") or not h.startswith("2025/alpha") else 3.0)
        plan = KC.load_plan(self.out, sample=self.sample)
        ids = [b["id"] for b in plan["shards"]["1of1"]["batches"]]
        self.assertEqual(ids, ["b0001", "s0001", "s0002", "s0003"])
        self.assertEqual(len(plan["base_shards"]["1of1"]["batches"]), 7)
        self.assertEqual(KC.verify_sample_rows(self.out, plan, "1of1"), (None, 20))
        drawn = set(s["rows"])
        self.assertFalse(drawn & {r["tag"] for r in self.rows[:10]})        # never a carried row
        self.assertTrue(drawn <= {r["tag"] for r in self.rows})
        # the runner's batch list is the effective one
        rc, txt = quiet(KC.main, ["batches", "--out", self.out, "--sample", self.sample])
        self.assertEqual(rc, 0)
        self.assertEqual([ln.split("\t")[0] for ln in txt.splitlines()], ids)
        # the global cross-tag check still reads EVERY plan row (a row neither carried nor drawn included)
        rows_all, _ = KC.all_runs(self.out, plan)
        self.assertEqual(set(rows_all), {r["tag"] for r in self.rows})
        # Batch resolves a sample batch and a carried one
        self.assertEqual(len(KC.Batch(self.out, plan, "1of1", "s0001").rows), 8)
        self.assertEqual(len(KC.Batch(self.out, plan, "1of1", "b0001").rows), 10)
        with self.assertRaises(KC.KCError):
            KC.Batch(self.out, plan, "1of1", "b0002")                       # not carried, not in the sample
        # without --sample the full plan is untouched
        self.assertEqual(len(KC.load_plan(self.out)["shards"]["1of1"]["batches"]), 7)

    def test_judged_rows_counted_and_never_redrawn(self):
        rc, txt = self.draw("--target-new", "57")                      # every row outside b0001
        self.assertEqual(rc, 0, txt)
        s = jload(self.sample)
        self.assertEqual(s["strata"]["2025/alpha/sat-replay"]["judged"], 10)
        for h, v in s["strata"].items():
            self.assertEqual(v["n"], v["judged"] + v["new"], h)
            self.assertEqual(v["n"], v["N"], h)                        # exhausted: judged + drawn = the stratum
        self.assertEqual(set(s["rows"]), {r["tag"] for r in self.rows[10:]})

    def test_a_carried_row_in_a_sample_batch_is_refused(self):
        self.assertEqual(self.draw()[0], 0)
        plan = KC.load_plan(self.out, sample=self.sample)
        spec = os.path.join(self.out, plan["sample"]["shards"]["1of1"]["batches"][0]["spec"])
        rows = L.read_spec(spec)
        rows[0] = self.rows[3]                                         # a b0001 (carried) row
        os.replace(spec, spec + ".orig")
        L.write_spec(spec, rows)
        why, _ = KC.verify_sample_rows(self.out, plan, "1of1")
        self.assertIn("is in carried batch b0001", why or "")

    def test_a_row_drawn_twice_is_refused(self):
        self.assertEqual(self.draw()[0], 0)
        plan = KC.load_plan(self.out, sample=self.sample)
        b = plan["sample"]["shards"]["1of1"]["batches"]
        first = L.read_spec(os.path.join(self.out, b[0]["spec"]))
        spec = os.path.join(self.out, b[1]["spec"])
        rows = L.read_spec(spec)
        rows[0] = first[0]                                             # the same row in two sample batches
        os.replace(spec, spec + ".orig")
        L.write_spec(spec, rows)
        why, _ = KC.verify_sample_rows(self.out, plan, "1of1")
        self.assertIn("drawn twice", why or "")

    def test_runs_of_a_started_unjudged_batch_are_forced_in(self):
        # b0002 was started by the full plan's runner and measured two rows before the re-plan stopped it
        os.makedirs(os.path.join(self.out, "run", "1of1", "b0002"))
        os.makedirs(os.path.join(self.out, "run", "1of1", "A"))
        line = L.Judgement(self.rows[11]["tag"], "competition", 1, "sat", "OK", "held", "1.00", [])
        wr(os.path.join(self.out, "run", "1of1", "A", "judged.tsv"), line.line() + "\n")
        self.assertEqual(len(L.read_judged(os.path.join(self.out, "run", "1of1", "A", "judged.tsv"))), 1)
        rc, txt = self.draw("--target-new", "3")
        self.assertEqual(rc, 0, txt)
        s = jload(self.sample)
        self.assertEqual(s["forced"], [self.rows[11]["tag"]])
        self.assertIn(self.rows[11]["tag"], s["rows"])
        self.assertEqual(s["started_not_carried"][0]["id"], "b0002")

    def test_deterministic_and_seeded(self):
        rc, a = self.draw("--dry-run")
        rc2, b = self.draw("--dry-run")
        self.assertEqual((rc, rc2), (0, 0))
        self.assertEqual(a.replace(a.split("\n")[0], ""), b.replace(b.split("\n")[0], ""))
        rc, txt = self.draw()
        s1 = jload(self.sample)
        rc, txt = self.draw("--name", "sample2", "--prefix", "t", "--seed", "another")
        self.assertEqual(rc, 0, txt)
        s2 = jload(os.path.join(self.out, "sample2.json"))
        self.assertEqual({h: v["new"] for h, v in s1["strata"].items()}, {h: v["new"] for h, v in s2["strata"].items()})
        self.assertNotEqual(set(s1["rows"]), set(s2["rows"]))
        self.assertFalse(os.path.exists(os.path.join(self.out, "sample3.json")))

    def test_dry_run_writes_nothing(self):
        before = sorted(os.listdir(self.out)), sorted(os.listdir(os.path.join(self.out, "specs", "1of1")))
        rc, txt = self.draw("--dry-run")
        self.assertEqual(rc, 0, txt)
        after = sorted(os.listdir(self.out)), sorted(os.listdir(os.path.join(self.out, "specs", "1of1")))
        self.assertEqual(before, after)

    def test_write_once(self):
        self.assertEqual(self.draw()[0], 0)
        rc, txt = self.draw()
        self.assertEqual(rc, 2)
        self.assertIn("drawn once", txt)

    def test_tampering_refused(self):
        self.assertEqual(self.draw()[0], 0)
        s = jload(self.sample)
        spec = os.path.join(self.out, s["shards"]["1of1"]["batches"][0]["spec"])
        # an edited sample spec: refused by the verified load (step 0)
        orig = rd(spec)
        wr(spec, "".join(orig.splitlines(True)[:-1]))
        with self.assertRaises(KC.KCError):
            KC.load_plan(self.out, sample=self.sample)
        wr(spec, orig)
        KC.load_plan(self.out, sample=self.sample)
        # an edited sample record: refused by its sha lock
        text = rd(self.sample)
        wr(self.sample, text.replace('"floor": 3', '"floor": 4'))
        with self.assertRaises(KC.KCError):
            KC.load_plan(self.out, verify_files=False, sample=self.sample)
        wr(self.sample, text)
        # a sample drawn against other plan bytes: refused
        s2 = dict(s, base_plan_sha256="0" * 64)
        t2 = json.dumps(s2, indent=1, sort_keys=True) + "\n"
        other = os.path.join(self.out, "other.json")
        wr(other, t2)
        wr(other + ".sha256", hashlib.sha256(t2.encode()).hexdigest() + "\n")
        with self.assertRaises(KC.KCError):
            KC.load_plan(self.out, verify_files=False, sample=other)
        # a re-judged carried final batch: refused by step 0's row check
        plan = KC.load_plan(self.out, sample=self.sample)
        rec = jload(self.b1)
        rec["judged"] = "later"
        wr(self.b1, json.dumps(rec))
        why, _ = KC.verify_sample_rows(self.out, plan, "1of1")
        self.assertIn("BATCH.json changed", why or "")

    def test_a_drawn_row_that_differs_from_the_plan_is_refused(self):
        self.assertEqual(self.draw()[0], 0)
        plan = KC.load_plan(self.out, sample=self.sample)
        s = plan["sample"]
        spec = os.path.join(self.out, s["shards"]["1of1"]["batches"][0]["spec"])
        rows = L.read_spec(spec)
        rows[0] = dict(rows[0], budget="21")
        os.replace(spec, spec + ".orig")
        L.write_spec(spec, rows)
        why, _ = KC.verify_sample_rows(self.out, plan, "1of1")
        self.assertIn("differs from the plan's row", why or "")

    def test_carried_emergency_refuses_a_replan(self):
        rec = jload(self.b1)
        rec["emergencies"] = ["S8_25_alpha_0000: phase A: truth contradiction"]
        wr(self.b1, json.dumps(rec))
        rc, txt = self.draw()
        self.assertEqual(rc, 2)
        self.assertIn("EMERGENCY", txt)
        wr(os.path.join(self.out, "EMERGENCY"), "x\n")
        rc, txt = self.draw("--dry-run")
        self.assertEqual(rc, 2)

    def test_summary_reports_the_sample_claim(self):
        self.assertEqual(self.draw()[0], 0)
        rc, txt = quiet(KC.main, ["summary", "--out", self.out, "--sample", self.sample])
        self.assertEqual(rc, 0, txt)
        last = [ln for ln in txt.splitlines() if ln.startswith("KC_SUMMARY ")][-1]
        self.assertIn("sample=sample1 sample_rows=10 ", last)
        self.assertIn(f"bound95_rows={KC.zero_event_bound(10):.4f}", last)


if __name__ == "__main__":
    unittest.main()
