#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Named selftests of the GEN-R board harness (scripts/genboard). No cargo, no ny binary, no box: every decision
rule runs on synthetic data, and the runner runs end to end in a STUB world (a stub pin and a stub box script).

    python3 scripts/genboard/genboard_selftest.py     # last line: GENBOARD_SELFTEST_DONE rc=0 ran=N passed=N failed=0

Each test prints `TEST <name> ... ok` or `TEST <name> ... FAILED: <why>`; the DONE line propagates the real rc.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import tempfile
import traceback
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import genboard as B  # noqa: E402
import genlib as G  # noqa: E402

L = G.L
P = G.P
ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def check(cond, why):
    if not cond:
        raise AssertionError(why)


def raises(fn, exc, needle=""):
    try:
        fn()
    except exc as e:  # noqa: BLE001
        check(needle in str(e), f"refused, but for another reason: {e}")
        return
    raise AssertionError(f"expected {exc.__name__} ({needle!r})")


def tmp():
    return tempfile.mkdtemp(prefix="genboard_selftest_")


# ---- statistics ------------------------------------------------------------------------------------------

@test
def stats_sign_thresholds_match_the_protocol():
    check(G.sign_threshold(23) == 17, f"23 groups need 17, got {G.sign_threshold(23)}")
    check(G.sign_threshold(29) == 21, f"29 groups need 21, got {G.sign_threshold(29)}")
    check(G.binom_two_sided(0, 0) == 1.0, "empty test is p=1")
    check(abs(G.sign_test(17, 6) - 0.0347) < 1e-3, f"17/23: {G.sign_test(17, 6)}")
    check(G.sign_test(16, 7) > 0.05, "16/23 is not significant")
    check(G.mcnemar_exact(0, 0) == 1.0, "no discordant rows")


@test
def clustered_mcnemar_votes_once_per_model():
    pairs = [("m0", True, False)] * 10 + [("m1", False, True), ("m2", False, True), ("m3", False, True)]
    r = G.clustered_mcnemar(pairs)
    check(r["gained_rows"] == 10 and r["lost_rows"] == 3, str(r))
    check(r["clusters_up"] == 1 and r["clusters_down"] == 3, f"one model is one vote: {r}")
    check(r["row_p"] < 0.1 and r["clustered_p"] > 0.5, f"rows look significant, clusters do not: {r}")


# ---- the generic configs dir -------------------------------------------------------------------------------

CATS = ["alpha_net_2026", "beta_net", "gamma_2025"]


@test
def generic_configs_are_byte_identical_and_resolve():
    out = os.path.join(tmp(), "cfg")
    man = G.write_configs(out, CATS)
    info = G.verify_configs(out, CATS)
    check(info["generic_sha256"] == man["generic_sha256"], "manifest and check disagree")
    check(info["files"] == len(CATS) * 2, f"one file per (year, category): {info}")
    for c in CATS:
        rel = L.resolve_preset(out, c)
        check(rel is not None and rel.startswith("vnncomp26/"), f"{c} resolves newest-first: {rel}")
    texts = {open(os.path.join(out, f)).read() for f in man["files"]}
    check(len(texts) == 1, "every file is the same bytes")
    check(set(man["keys"]) == set(G.GENERIC_KEYS), f"only generic keys: {man['keys']}")
    check(L.strict_load(open(os.path.join(out, man["files"][0])).read())["optimizer"]["priors"] is True,
          "the default GEN-R arm reads the prior")


@test
def generic_configs_refuse_category_arming_and_drift():
    bad_text = G.generic_yaml() + "margin_row:\n  frozen_dual_bab: true\n"
    raises(lambda: G.write_configs(os.path.join(tmp(), "cfg"), CATS, text=bad_text), G.GenError, "non-generic key")
    out = os.path.join(tmp(), "cfg")
    man = G.write_configs(out, CATS)
    with open(os.path.join(out, man["files"][0]), "a") as fh:
        fh.write("# drift\n")
    raises(lambda: G.verify_configs(out, CATS), G.GenError, "distinct yaml texts")
    out2 = os.path.join(tmp(), "cfg")
    G.write_configs(out2, CATS[:1])
    raises(lambda: G.verify_configs(out2, CATS), G.GenError, "resolves to NO preset")
    raises(lambda: G.write_configs(out2, CATS), G.GenError, "never deletes")
    raises(lambda: G.generic_yaml(router="maybe"), G.GenError, "off|shadow|on")
    check("priors" not in G.generic_yaml(priors=False), "an off prior is omitted, not written false")


# ---- the screen plan ---------------------------------------------------------------------------------------

def synthetic_rows(seed="s0", n=40):
    rows = []
    for cat in ("c_one", "c_two"):
        for i in range(n):
            rows.append(G.PlanRow(2026, cat, G.family_of(cat), i, f"/m/{cat}_{i % 3}.onnx", f"/p/{i}.vnnlib", 100,
                                  f"{cat}_{i % 3}", G.seeded_order(seed, 2026, cat, i)))
    return rows


@test
def screen_is_seeded_and_never_csv_order():
    a = G.screen(synthetic_rows("s0"), 0.25)
    b = G.screen(synthetic_rows("s0"), 0.25)
    c = G.screen(synthetic_rows("s1"), 0.25)
    check([r.key() for r in a] == [r.key() for r in b], "same seed, same screen")
    check([r.key() for r in a] != [r.key() for r in c], "another seed, another screen")
    per = {}
    for r in a:
        per.setdefault(r.category, []).append(r.index)
    check(all(len(v) == 10 for v in per.values()), f"ceil(0.25 x 40) per category: {per}")
    check(any(sorted(v) != list(range(10)) for v in per.values()), "not the first rows of instances.csv")
    raises(lambda: G.screen(synthetic_rows(), 0.0), G.GenError, "screen fraction")


# ---- judging, stop-loss, scoring ---------------------------------------------------------------------------

def row(i=0, cat="c_one"):
    return G.PlanRow(2026, cat, cat, i, "/m/a.onnx", f"/p/{i}.vnnlib", 100, "a", "0")


def log(r, router=True, done=True):
    text = f"Running VNN-COMP instance: category='{r.category}' onnx='{r.onnx}' vnnlib='{r.vnnlib}' results='x'\n"
    if router:
        text += "[router] v1 mode=on effective=acting\n"
    if done:
        text += "[router] done faults=0 bus_pushed=0\n"
    return text


@test
def judge_voids_what_the_board_cannot_vouch_for():
    r = row()
    cfg = tmp()
    inside = os.path.join(cfg, "vnncomp26", "c_one.yaml")
    ok = G.judge_run(r, log(r), "unsat", 3.0, True, cfg, inside)
    check(ok.verdict == "unsat" and ok.void_reason is None, str(ok))
    check(G.judge_run(r, log(r), None, 3.0, True).verdict == "void", "no result token")
    other = row(1)
    check(G.judge_run(r, log(other), "unsat", 1.0, True).void_reason.startswith("the log's INSTANCE"),
          "a wrong-row log is VOID")
    check("no [router] done" in G.judge_run(r, log(r, done=False), "unsat", 1.0, True).void_reason,
          "a router run without done is VOID")
    check("banner" in G.judge_run(r, log(r, router=False, done=False), "unsat", 1.0, True).void_reason,
          "a dark arm is VOID")
    check(G.judge_run(r, log(r, router=False, done=False), "sat", 1.0, False).verdict == "sat",
          "a router-off arm needs no router lines")
    check("configs dir" in G.judge_run(r, log(r), "unsat", 1.0, True, cfg, "/elsewhere/c_one.yaml").void_reason,
          "a preset from another dir is another arm")
    flight = {"events": [{"method": "preset_base", "status": "ran", "reason": inside}]}
    check(G.preset_base_of(flight, "Loading preset: /tmp/eff.yaml") == inside, "the durable base wins")


@test
def stop_loss_fires_at_half_of_tuned():
    keys = [(2026, "c", i) for i in range(10)]
    tuned = {k: ("unsat" if i < 8 else "timeout") for i, k in enumerate(keys)}
    gen4 = [G.Outcome(k, "unsat" if i < 4 else "timeout", 1.0) for i, k in enumerate(keys)]
    gen5 = [G.Outcome(k, "unsat" if i < 5 else "timeout", 1.0) for i, k in enumerate(keys)]
    check(G.stop_loss(gen4, tuned), "4 <= 0.5 x 8 stops")
    check(not G.stop_loss(gen5, tuned), "5 > 0.5 x 8 continues")
    check(not G.stop_loss(gen4[:9], tuned), "never before 10 rows")


@test
def score_counts_contradictions_as_wrong_and_emergencies():
    plan = [row(i) for i in range(4)]
    gen = {plan[0].key(): G.Outcome(plan[0].key(), "unsat", 1.0),
           plan[1].key(): G.Outcome(plan[1].key(), "sat", 1.0),
           plan[2].key(): G.Outcome(plan[2].key(), "timeout", 1.0),
           plan[3].key(): G.Outcome(plan[3].key(), "unsat", 1.0)}
    tuned = {plan[0].key(): G.Outcome(plan[0].key(), "unsat", 1.0),
             plan[1].key(): G.Outcome(plan[1].key(), "unsat", 1.0),
             plan[2].key(): G.Outcome(plan[2].key(), "sat", 1.0),
             plan[3].key(): G.Outcome(plan[3].key(), "timeout", 1.0)}
    truth = {plan[3].key(): "sat"}
    s = G.score(plan, gen, tuned, truth)
    c = s["categories"][0]
    check(len(s["emergencies"]) == 2, f"GEN sat vs TUNED unsat, GEN unsat vs FIELD sat: {s['emergencies']}")
    check(c.gen_decided == 1 and c.tuned_decided == 3, str(c))
    check(c.gen_raw == 10 * 1 - 150 * 2, f"two wrong rows at -150: {c.gen_raw}")
    check(c.gained == 0 and c.lost == 2, f"{c.gained}/{c.lost}")


@test
def score_ratio_target_and_family_sign():
    plan = []
    gen, tuned = {}, {}
    for f, (g, t) in enumerate([(9, 10), (10, 10), (8, 10)]):
        for i in range(10):
            r = G.PlanRow(2026, f"cat{f}", f"fam{f}", i, "/m.onnx", f"/{i}.vnnlib", 100, f"m{f}", "0")
            plan.append(r)
            gen[r.key()] = G.Outcome(r.key(), "unsat" if i < g else "timeout", 1.0)
            tuned[r.key()] = G.Outcome(r.key(), "unsat" if i < t else "timeout", 1.0)
    s = G.score(plan, gen, tuned)
    check(abs(s["ratio"] - 27 / 30) < 1e-12, str(s["ratio"]))
    check(not s["meets_target"], "0.9 < 0.95")
    check(s["family_up"] == 0 and s["family_down"] == 2, f"{s['family_up']}/{s['family_down']}")
    text = G.render_score(s, same_pin=False)
    check("NOT-SAME-PIN" in text and "NOT MET" in text, text)


# ---- rule-existence LOGO -----------------------------------------------------------------------------------

@test
def logo_folds_disable_exactly_the_single_family_rules():
    rules = G.read_rule_evidence(os.path.join(HERE, "rule_evidence.tsv"))
    plan = [G.PlanRow(2026, c, G.family_of(c), 0, "/m", "/p", 100, "m", "0")
            for c in ("cifar100_2024", "tinyimagenet_2024", "ml4acopf_2024", "acasxu_2023")]
    folds = G.logo_folds(plan, rules)
    check(set(folds) == {"resnet_ct", "ml4acopf_2024", "acasxu"}, str(folds))
    check("optimizer.rules.governor" in folds["resnet_ct"] and "router.rules.far_hand_on" in folds["resnet_ct"],
          str(folds["resnet_ct"]))
    check(folds["ml4acopf_2024"] == ["router.rules.derived_caps"], str(folds["ml4acopf_2024"]))
    check(folds["acasxu"] == [], "a family that is nobody's only evidence runs the plain board")
    for keys in folds.values():
        check("router.rules.cap_authority" not in keys and "optimizer.priors" not in keys,
              "structural / multi rules are never disabled")
    text = G.fold_yaml(G.generic_yaml(), folds["resnet_ct"])
    doc = L.strict_load(text)
    check(L.deep_get(doc, "optimizer.rules.governor") is False and L.deep_get(doc, "router.mode") == "on", text)


@test
def rule_evidence_names_real_typed_keys_and_known_families():
    space = open(os.path.join(ROOT, "crates/ny-levers/src/space.rs")).read()
    rules = G.read_rule_evidence(os.path.join(HERE, "rule_evidence.tsv"))
    known = set(P.FAMILY_MERGES.values()) | set(P.FAMILY_MERGES) | {"ml4acopf_2024", "collins_rul_cnn_2022"}
    for r in rules:
        check(f'Deliver::PresetKey("{r["key"]}")' in space, f"{r['rule']}: {r['key']} is not a delivered typed key")
        for f in r["evidence"] or ():
            check(f in known, f"{r['rule']}: unknown family {f}")
    # review(eng12): "every acting rule is listed" is DERIVED from the delivered rule keys, never a count floor. The
    # floor (`len(rules) >= 16`) stayed green when the ENG12 merge put R6's four O-TRIAL rules beside R7's table:
    # four acting rules with no row, so no LOGO fold could ever disable them.
    delivered = set(re.findall(
        r'Deliver::PresetKey\("((?:router|optimizer)\.rules\.[a-z_]+|optimizer\.priors)"\)', space))
    check("router.rules.route" in delivered and "optimizer.priors" in delivered,
          f"the delivered rule-key scan of space.rs matched nothing it must (VOID): {sorted(delivered)}")
    missing = sorted(delivered - {r["key"] for r in rules})
    check(not missing, f"acting rules with no rule-evidence row: {missing}")


@test
def logo_prior_refit_reports_decision_changes_per_family():
    """On the committed training set the shipped table prices no binding (attack.upfront fails G7, root.bound
    G6) and no leave-one-family-out refit made by the SAME pipeline prices one either: the GEN-R rows ARE the
    held-out rows for the prior (0 decision changes, 0 value changes)."""
    rows = P.read_rows(os.path.join(ROOT, "scripts/router_priors/data/priors_v1_trainset.tsv"))
    plan = ([G.PlanRow(2026, "safenlp_2024", "safenlp_2024", i, "/m", "/p", 20, "sn", "0") for i in range(2)]
            + [G.PlanRow(2026, "acasxu_2023", "acasxu", i, "/m", "/p", 116, "ac", "0") for i in range(2)])
    feats = {(2026, "safenlp_2024", "sn"): (0, 0, 4231.0), (2026, "acasxu_2023", "ac"): (0, 0, 13310.0)}
    out = G.prior_decision_changes(plan, rows, feats)
    check(set(out) == {"safenlp_2024", "acasxu"}, str(out))
    for fam, rec in out.items():
        check(rec == {"decision_changes": 0, "value_changes": 0}, f"{fam}: {rec}")


@test
def logo_prior_decision_change_positive_control():
    """POSITIVE CONTROL: a pipeline whose full fit REFUSES the attack phase but whose fit without family `fx`
    SHIPS it makes fx's rows priced-vs-unpriced: a decision change (needs a held-out binary); a value-only
    difference is counted apart; a rung past the window is a decision change too."""
    model = P.PhaseModel("attack.upfront", 0.25, {P.ROOT_CELL: -4.0}, {P.ROOT_CELL: 0.5}, {}, 9, 3, 0.3)
    big = P.PhaseModel("attack.upfront", 0.25, {P.ROOT_CELL: 30.0}, {P.ROOT_CELL: 0.5}, {}, 9, 3, 0.3)
    small = P.PhaseModel("attack.upfront", 0.25, {P.ROOT_CELL: -3.0}, {P.ROOT_CELL: 0.5}, {}, 9, 3, 0.3)

    def fake_build(rows):
        fams = {r.family for r in rows}
        if "fx" not in fams:
            return {"attack.upfront": {"ships": True, "model": model}}
        if "fy" not in fams:
            return {"attack.upfront": {"ships": True, "model": big}}
        if "fz" not in fams:
            return {"attack.upfront": {"ships": True, "model": small}}
        return {"attack.upfront": {"ships": False}} if "fw" in fams else {"attack.upfront": {"ships": True,
                                                                                             "model": model}}

    rows = [P.Row("attack.upfront", f, f, f"m{f}", 0, 0, 1000.0, 1.0, "r") for f in ("fx", "fy", "fz", "fw")]
    plan = [G.PlanRow(2026, f, f, 0, "/m", "/p", 100, "m", "0") for f in ("fx", "fy", "fz", "fw")]
    feats = {(2026, f, "m"): (0, 0, 2.0 ** 16) for f in ("fx", "fy", "fz", "fw")}

    # full fit = rows of every family: fx, fy, fz, fw all present -> refused (unpriced)
    out = G.prior_decision_changes(plan, rows, feats, build=fake_build)
    check(out["fx"]["decision_changes"] == 1, f"unpriced in full, priced held-out: {out['fx']}")
    check(out["fy"]["decision_changes"] == 1, f"priced held-out at a rung past the window: {out['fy']}")
    check(out["fw"]["decision_changes"] == 1, f"{out['fw']}")
    # value-only: compare two SHIPPING fits that both fit the window
    def ship_build(rows):
        fams = {r.family for r in rows}
        return {"attack.upfront": {"ships": True, "model": model if "fz" in fams else small}}
    out2 = G.prior_decision_changes(plan, rows, feats, build=ship_build)
    check(out2["fz"] == {"decision_changes": 0, "value_changes": 1}, f"value-only change: {out2['fz']}")
    check(out2["fx"] == {"decision_changes": 0, "value_changes": 0}, str(out2["fx"]))


# ---- sealed holdout ----------------------------------------------------------------------------------------

@test
def sealed_holdout_refuses_edits_and_a_second_look():
    d = tmp()
    entries = ["2022|mnist_fc|0", "2022|mnist_fc|1", "2021|oval21|3"]
    seal = G.write_seal(os.path.join(d, "seal.json"), entries, "selftest")
    check(G.verify_seal(os.path.join(d, "seal.json"), list(reversed(entries)))["sha256"] == seal["sha256"],
          "order does not matter")
    raises(lambda: G.verify_seal(os.path.join(d, "seal.json"), entries + ["2022|mnist_fc|2"]), G.GenError,
           "edited after sealing")
    raises(lambda: G.write_seal(os.path.join(d, "seal.json"), entries, "again"), G.GenError, "written once")
    led = os.path.join(d, "ledger.tsv")
    check(G.spend_holdout(led, seal, "first board", False) == 1, "first spend")
    raises(lambda: G.spend_holdout(led, seal, "peek", False), G.GenError, "already spent")
    check(G.spend_holdout(led, seal, "pre-registered v2", True) == 2, "an explicit, recorded respend")


# ---- the runner, end to end in a stub world ----------------------------------------------------------------

STUB_PIN = r"""#!/usr/bin/env bash
# stub ny: prints the lines a real run prints and writes a result + sidecar
cat="$3"; onnx="$4"; vnnlib="$5"; res="$6"; cfg="$9"
echo "Running VNN-COMP instance: category='$cat' onnx='$onnx' vnnlib='$vnnlib' results='$res'"
echo "Loading preset: $cfg/vnncomp26/$cat.yaml"
if [ "${STUB_ROUTER:-1}" = 1 ]; then
  echo "[router] v1 mode=on effective=acting optimizer=on host=${STUB_HOST:-gb10} threads=20" >&2
  if [ "${STUB_PRIORS:-1}" = 1 ]; then
    if [ "${STUB_HOST:-gb10}" = gb10 ]; then
      echo "[router] priors v=cost-v1+deadbeef host=gb10 cell=conv=0;nonpwl=0 priced=- unpriced=attack.upfront_apgd:attack.upfront:refused predict=verifier.setup:12/20ms" >&2
    else
      echo "[router] priors v=cost-v1+deadbeef host=gb10 cell=conv=0;nonpwl=0 priced=- unpriced=attack.upfront_apgd:host:${STUB_HOST} predict=-" >&2
    fi
  fi
  if [ "${STUB_DONE:-1}" = 1 ]; then
    echo "[router] done faults=0 bus_pushed=0" >&2
  fi
fi
echo "${STUB_VERDICT:-unsat}" > "$res"
printf '{"events":[{"method":"preset_base","status":"ran","reason":"%s"}]}' "$cfg/vnncomp26/$cat.yaml" > "$res.flight.json"
exit 0
"""

STUB_BOX = r"""#!/usr/bin/env bash
# stub ny_box.sh: measure [--budget N] -- cmd...
while [ "$1" != "--" ]; do shift; done; shift
echo "[ny-box] stub measure" >&2
"$@"
"""


def stub_world():
    d = tmp()
    pin = os.path.join(d, "ny_stub")
    box = os.path.join(d, "ny_box_stub.sh")
    for path, text in ((pin, STUB_PIN), (box, STUB_BOX)):
        with open(path, "w") as fh:
            fh.write(text)
        os.chmod(path, 0o755)
    files = os.path.join(d, "files")
    os.makedirs(files)
    plan_rows = []
    for i in range(12):
        onnx = os.path.join(files, "net.onnx")
        vnn = os.path.join(files, f"p{i}.vnnlib")
        for f in (onnx, vnn):
            open(f, "a").close()
        plan_rows.append(G.PlanRow(2026, "stub_cat", "stub_cat", i, onnx, vnn, 5, "net",
                                   G.seeded_order("st", 2026, "stub_cat", i)))
    plan = os.path.join(d, "plan.tsv")
    G.write_plan(plan, plan_rows)
    cfg = os.path.join(d, "cfg")
    G.write_configs(cfg, ["stub_cat"])
    judged = os.path.join(d, "tuned.judged.tsv")
    with open(judged, "w") as fh:
        fh.write("year\tcategory\tindex\tverdict\twall\tvoid_reason\trouter_banner\trouter_done\n")
        for r in plan_rows:
            fh.write(f"2026\tstub_cat\t{r.index}\tunsat\t1.0\t\t0\t0\n")
    return d, pin, box, plan, cfg, judged


def run_cli(argv, box, env_extra=None):
    old_box = B.BOX
    saved = dict(os.environ)
    try:
        B.BOX = box
        for k in list(os.environ):
            if k.startswith("NY_") and not k.startswith("NY_BOX_"):
                del os.environ[k]
        os.environ.update(env_extra or {})
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = B.main(argv)
        return rc, buf.getvalue()
    finally:
        B.BOX = old_box
        os.environ.clear()
        os.environ.update(saved)


@test
def runner_end_to_end_in_a_stub_world():
    d, pin, box, plan, cfg, judged = stub_world()
    out = os.path.join(d, "gen")
    rc, text = run_cli(["--data-root", d, "run", "--plan", plan, "--configs", cfg, "--pin", pin, "--out", out,
                        "--arm", "gen-r", "--tuned-judged", judged], box)
    check(rc == 0 and "GENBOARD_RUN_DONE rc=0 holds=2" in text, text)
    check("prior_read=12 prior_host_gated=0" in text and "hosts=gb10:12" in text, text)
    res = G.read_judged(os.path.join(out, "results.csv.judged.tsv"))
    check(len(res) == 12 and all(o.verdict == "unsat" for o in res.values()), str(res))
    check(all(o.router_done for o in res.values()), "every armed row printed [router] done")
    meta = json.load(open(os.path.join(out, "RUN_META.json")))
    check(len(meta["pin_sha256"]) == 64 and meta["arm"] == "gen-r", str(meta))
    rc, text = run_cli(["--data-root", d, "score", "--plan", plan, "--gen", out, "--tuned", out,
                        "--truth-2026", "", "--truth-2025", ""], box)
    check(rc == 0 and "ratio=1.0" in text, text)
    # Resume: a second run skips rows already captured (no hold).
    rc, text = run_cli(["--data-root", d, "run", "--plan", plan, "--configs", cfg, "--pin", pin, "--out", out,
                        "--arm", "gen-r", "--tuned-judged", judged], box)
    check(rc == 0 and "holds=0" in text, text)


@test
def runner_refuses_before_any_hold():
    d, pin, box, plan, cfg, judged = stub_world()
    rc, text = run_cli(["--data-root", d, "run", "--plan", plan, "--configs", cfg, "--pin", pin,
                        "--out", os.path.join(d, "o1"), "--arm", "gen-r"], box, {"NY_ROUTER_MODE": "on"})
    check(rc == 2 and "zero NY_*" in text, text)
    rc, text = run_cli(["--data-root", d, "run", "--plan", plan, "--configs", cfg, "--pin",
                        os.path.join(d, "missing"), "--out", os.path.join(d, "o2"), "--arm", "gen-r"], box)
    check(rc == 2 and "not an executable" in text, text)
    shipped_like = os.path.join(d, "shipped")
    os.makedirs(os.path.join(shipped_like, "vnncomp26"))
    with open(os.path.join(shipped_like, "vnncomp26", "stub_cat.yaml"), "w") as fh:
        fh.write("bab:\n  max_depth: 7\n")
    rc, text = run_cli(["--data-root", d, "run", "--plan", plan, "--configs", shipped_like, "--pin", pin,
                        "--out", os.path.join(d, "o3"), "--arm", "gen-r"], box)
    check(rc == 2 and "non-generic key" in text, f"a tuned dir is not a GEN-R dir: {text}")


@test
def runner_stops_on_a_contradiction_and_on_a_dark_arm():
    d, pin, box, plan, cfg, judged = stub_world()
    out = os.path.join(d, "gen_sat")
    rc, text = run_cli(["--data-root", d, "run", "--plan", plan, "--configs", cfg, "--pin", pin, "--out", out,
                        "--arm", "gen-r", "--tuned-judged", judged], box, {"STUB_VERDICT": "sat"})
    check(rc == 3 and os.path.isfile(os.path.join(out, "EMERGENCY")), f"sat vs TUNED unsat is an emergency: {text}")
    out2 = os.path.join(d, "gen_dark")
    rc, text = run_cli(["--data-root", d, "run", "--plan", plan, "--configs", cfg, "--pin", pin, "--out", out2,
                        "--arm", "gen-r", "--tuned-judged", judged], box, {"STUB_ROUTER": "0"})
    check(rc == 0 and " dark=1 " in text, text)
    res = G.read_judged(os.path.join(out2, "results.csv.judged.tsv"))
    check(all(o.verdict == "void" for o in res.values()), "a dark router arm is VOID, never negative")
    # Review fix (#router-r7): a dark arm stops the category with NO capability verdict -- it is not a
    # stop-loss (which would read a VOID arm as a GEN-R negative).
    check(os.path.isfile(os.path.join(out2, "DARK")), "the dark arm is recorded as dark")
    check(not os.path.isfile(os.path.join(out2, "STOPLOSS")), "and never as a stop-loss")


# ---- review fixes (#router-r7) -----------------------------------------------------------------------------

@test
def a_void_row_never_hides_a_contradiction():
    """A row VOIDed for a missing observable still PUBLISHED a verdict: a wrong one is an emergency (the moat is
    absolute), at `score` and in the runner. Identity VOIDs (no result / another row's log) carry nothing."""
    r = row()
    o = G.judge_run(r, log(r, done=False), "sat", 1.0, True)
    check(o.verdict == "void" and o.published == "sat", str(o))
    check(G.published_verdict(o) == "sat", "the publication survives the VOID")
    s = G.score([r], {r.key(): o}, {r.key(): G.Outcome(r.key(), "unsat", 1.0, published="unsat")})
    check(len(s["emergencies"]) == 1 and "VOID row" in s["emergencies"][0], str(s["emergencies"]))
    check(s["categories"][0].gen_raw == -150, "a contradicting VOID row is provisionally wrong")
    other = row(1)
    ident = G.judge_run(r, log(other), "sat", 1.0, True)
    check(ident.published is None and G.published_verdict(ident) is None, "another row's log publishes nothing")
    # TUNED (same pin) against the field is an emergency too.
    s2 = G.score([r], {r.key(): G.Outcome(r.key(), "timeout", 1.0)},
                 {r.key(): G.Outcome(r.key(), "sat", 1.0, published="sat")}, {r.key(): "unsat"})
    check(any("TUNED sat vs FIELD unsat" in e for e in s2["emergencies"]), str(s2["emergencies"]))
    # The runner: a published sat with no `[router] done` against TUNED unsat stops the board.
    d, pin, box, plan, cfg, judged = stub_world()
    out = os.path.join(d, "gen_void_sat")
    rc, text = run_cli(["--data-root", d, "run", "--plan", plan, "--configs", cfg, "--pin", pin, "--out", out,
                        "--arm", "gen-r", "--tuned-judged", judged], box, {"STUB_VERDICT": "sat", "STUB_DONE": "0"})
    check(rc == 3 and "VOID row" in open(os.path.join(out, "EMERGENCY")).read(), text)


@test
def a_field_contradiction_stops_the_runner():
    d, pin, box, plan, cfg, judged = stub_world()
    rows = G.read_plan(plan)
    old = B.truth_table
    try:
        B.truth_table = lambda a, p: {rows[0].key(): "sat"}
        out = os.path.join(d, "gen_field")
        rc, text = run_cli(["--data-root", d, "run", "--plan", plan, "--configs", cfg, "--pin", pin, "--out", out,
                            "--arm", "gen-r"], box)
    finally:
        B.truth_table = old
    check(rc == 3 and "vs FIELD sat" in open(os.path.join(out, "EMERGENCY")).read(), text)


@test
def the_instance_line_must_name_this_rows_network():
    r = G.PlanRow(2026, "c_one", "c_one", 0, "/b/onnx/net_1_1.onnx", "/b/vnnlib/prop_1.vnnlib", 100, "n", "0")
    same_prop_other_net = G.PlanRow(2026, "c_one", "c_one", 1, "/b/onnx/net_4_5.onnx", "/b/vnnlib/prop_1.vnnlib",
                                    100, "n", "0")
    o = G.judge_run(r, log(same_prop_other_net), "unsat", 1.0, True)
    check(o.verdict == "void" and o.void_reason == G.IDENTITY_VOIDS[1], f"wrong network, same property: {o}")
    gz = G.PlanRow(2026, "c_one", "c_one", 0, "/b/onnx/net_1_1.onnx.gz", "/b/vnnlib/prop_1.vnnlib", 100, "n", "0")
    check(G.judge_run(gz, log(r), "unsat", 1.0, True).verdict == "unsat", "a .gz row matches its decompressed log")
    paired = "[('f', '/x/a.onnx'), ('g', '/x/b.onnx')]"
    check(G.same_model(paired, paired), "a paired row matches its own literal")
    check(not G.same_model("[('f', '/x/a.onnx'), ('g', '/x/c.onnx')]", paired), "a paired row with another member")


@test
def logo_folds_run_under_the_fold_arm_and_nothing_else_slips_in():
    rules = G.read_rule_evidence(os.path.join(HERE, "rule_evidence.tsv"))
    plan = [G.PlanRow(2026, "cifar100_2024", "resnet_ct", 0, "/m", "/p", 100, "m", "0")]
    keys = G.logo_folds(plan, rules)["resnet_ct"]
    text = G.fold_yaml(G.generic_yaml(), keys)
    d = tmp()
    fold = os.path.join(d, "fold")
    B._write_fold(fold, ["cifar100_2024"], text)
    raises(lambda: G.verify_configs(fold, ["cifar100_2024"]), G.GenError, "non-generic key")
    info = G.verify_configs(fold, ["cifar100_2024"], fold=True)
    check(info["priors"] is True, "the fold keeps the generic prior switch")
    for bad in (G.generic_yaml() + "margin_row:\n  frozen_dual_bab: true\n",
                G.fold_yaml(G.generic_yaml(), []) + "router:\n  rules:\n    governor: true\n"):
        check(G.check_generic_text(bad, fold=True), f"a fold may only switch rules OFF: {bad}")
    armed = G.fold_yaml(G.generic_yaml(), ["router.rules.publish_grace"]).replace("publish_grace: false",
                                                                                  "publish_grace: true")
    check(G.check_generic_text(armed, fold=True), "an ablation key set true is arming, not ablation")


@test
def logo_refuses_an_armed_prior_without_its_refit_report():
    d = tmp()
    cfg = os.path.join(d, "cfg")
    G.write_configs(cfg, ["acasxu_2023"])
    plan = os.path.join(d, "plan.tsv")
    G.write_plan(plan, [G.PlanRow(2026, "acasxu_2023", "acasxu", 0, "/m", "/p", 100, "m", "0")])
    rc, text = run_cli(["logo", "--plan", plan, "--base-configs", cfg, "--out", os.path.join(d, "lg")], "/bin/false")
    check(rc == 2 and "needs --trainset and --features" in text, text)
    cfg2 = os.path.join(d, "cfg_noprior")
    G.write_configs(cfg2, ["acasxu_2023"], text=G.generic_yaml(priors=False))
    rc, text = run_cli(["logo", "--plan", plan, "--base-configs", cfg2, "--out", os.path.join(d, "lg2")],
                       "/bin/false")
    check(rc == 0 and "GENBOARD_LOGO_DONE rc=0" in text, f"no prior armed, no refit needed: {text}")


@test
def the_prior_rung_mirror_honours_units_and_the_ceiling():
    """`_rung` is the Rust reader's arithmetic: secs_per_clause x clauses, per-phase ceil >= 1 with upper >=
    median, the one-day ceiling; a per-clause phase without a clause count is UNDECIDABLE and counts as a
    decision change (fail closed)."""
    setup = P.PhaseModel("verifier.setup", 0.0, {P.ROOT_CELL: -2.0}, {P.ROOT_CELL: 0.5}, {}, 9, 3, 0.3)
    root = P.PhaseModel("root.bound", 0.0, {P.ROOT_CELL: -4.0}, {P.ROOT_CELL: 0.5}, {}, 9, 3, 0.3)
    phases = {"verifier.setup": {"ships": True, "model": setup}, "root.bound": {"ships": True, "model": root}}
    feat = (0, 0, 2.0 ** 10)
    one = G._rung(phases, ("verifier.setup", "root.bound"), feat, 1)
    ten = G._rung(phases, ("verifier.setup", "root.bound"), feat, 10)
    up_setup = math.ceil(2 ** -1.5 * 1000)
    check(one == up_setup + math.ceil(2 ** -3.5 * 1000), f"one clause: {one}")
    check(ten == up_setup + math.ceil(2 ** -3.5 * 10 * 1000), f"ten clauses scale the per-clause phase: {ten}")
    check(G._rung(phases, ("verifier.setup", "root.bound"), feat, None) == G.UNDECIDABLE, "no clause count")
    huge = {"verifier.setup": {"ships": True, "model": P.PhaseModel("verifier.setup", 0.0, {P.ROOT_CELL: 40.0},
                                                                    {P.ROOT_CELL: 0.5}, {}, 9, 3, 0.3)}}
    check(G._rung(huge, ("verifier.setup",), feat) == G.PREDICTION_CEILING_MS, "clamped at one day")

    def build(rows):
        return phases if any(r.family == "fz" for r in rows) else {"verifier.setup": {"ships": False}}

    rows = [P.Row("verifier.setup", f, f, f"m{f}", 0, 0, 1024.0, 1.0, "r") for f in ("fz", "fy")]
    plan = [G.PlanRow(2026, "fz", "fz", 0, "/m", "/p", 100, "m", "0")]
    feats = {(2026, "fz", "m"): feat}
    no_k = G.prior_decision_changes(plan, rows, feats, build=build)
    check(no_k["fz"]["decision_changes"] >= 1 and no_k["fz"].get("undecidable", 0) >= 1, str(no_k))


@test
def a_dark_or_host_gated_prior_is_visible_never_assumed():
    r = row()
    no_line = G.judge_run(r, log(r), "unsat", 1.0, True, priors_on=True)
    check(no_line.verdict == "void" and "priors" in no_line.void_reason and no_line.published == "unsat",
          f"the key armed but no [router] priors line is another arm: {no_line}")
    gated = log(r) + ("[router] priors v=cost-v1+deadbeef host=gb10 cell=conv=0;nonpwl=0 priced=- "
                      "unpriced=attack.upfront_apgd:host:other,input_split:host:other predict=-\n")
    o = G.judge_run(r, gated, "unsat", 1.0, True, priors_on=True)
    check(o.verdict == "unsat" and o.prior == "host_gated", str(o))
    read = log(r) + ("[router] priors v=cost-v1+deadbeef host=gb10 cell=conv=0;nonpwl=0 priced=- "
                     "unpriced=attack.upfront_apgd:attack.upfront:refused predict=verifier.setup:12/20ms\n")
    check(G.judge_run(r, read, "unsat", 1.0, True, priors_on=True).prior == "read", "priced on its host")
    check(G.prior_state("[router] priors unavailable why=x\n") == "unavailable", "unavailable")
    d, pin, box, plan, cfg, judged = stub_world()
    out = os.path.join(d, "gen_gated")
    rc, text = run_cli(["--data-root", d, "run", "--plan", plan, "--configs", cfg, "--pin", pin, "--out", out,
                        "--arm", "gen-r", "--tuned-judged", judged], box, {"STUB_HOST": "other"})
    check(rc == 0 and "prior_host_gated=12" in text and "hosts=other:12" in text, text)
    out2 = os.path.join(d, "gen_noprior")
    rc, text = run_cli(["--data-root", d, "run", "--plan", plan, "--configs", cfg, "--pin", pin, "--out", out2,
                        "--arm", "gen-r", "--tuned-judged", judged], box, {"STUB_PRIORS": "0"})
    check(rc == 0 and " dark=1 " in text, f"a prior-armed board with no prior line is dark: {text}")


@test
def score_is_per_year_and_never_pools_the_twin_year():
    plan, gen, tuned = [], {}, {}
    for year in (2025, 2026):
        for f in range(3):
            r = G.PlanRow(year, f"cat{f}", f"fam{f}", 0, "/m.onnx", "/0.vnnlib", 100, f"m{f}", "0")
            plan.append(r)
            gen[r.key()] = G.Outcome(r.key(), "unsat", 1.0)
            tuned[r.key()] = G.Outcome(r.key(), "timeout", 1.0)
    s = G.score(plan, gen, tuned)
    check(set(s["per_year"]) == {2025, 2026}, str(s["per_year"]))
    check(all(y["families"] == 3 and y["family_up"] == 3 for y in s["per_year"].values()), str(s["per_year"]))
    check(s["family_sign_p"] is None, "no pooled sign test across years")
    check("NOT pooled" in G.render_score(s, True), G.render_score(s, True))


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
    print(f"GENBOARD_SELFTEST_DONE rc={rc} ran={ran} passed={passed} failed={failed}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
