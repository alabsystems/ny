#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""seal: the pipeline that turns BUILT capabilities into SEALED ones (scripts/seal/README.md).

    seal.py plan <capability-id> --pin <bin> --out <dir> [--shards N | --shard k/n] [--stages S5,S7,S6,S8]
                                      pre-register: configs from the pin's commit, three verified overlays (arm /
                                      ctl / negctl), one m0-spec/v1 spec per stage (and shard), plan.json (+sha256)
    seal.py run --out <dir> --stage S [--shard k/n] [--check|--snapshot|--launch]
                                      execute through scripts/seal/seal_run.sh (systemd-run --user, MemoryMax=96G,
                                      ny_box.sh measure per run via the m0 runner, step 0 before any wait,
                                      heartbeat, SEAL_DONE line, EMERGENCY file stops every shard)
    seal.py judge-stage --out <dir> --stage S [--no-write]
                                      the stage verdict (PASS / PARTIAL / VOID / EMERGENCY) -> <out>/<S>/STAGE.json
    seal.py envelope --out <dir>      the op-kind / spec-archetype envelope of the S7 + S8 evidence rows
    seal.py record --out <dir> [--registry R] [--replace] [--dry-run]
                                      write the evidence into the registry entry the auditor reads (verified edit)
    seal.py verify-record <STAGE.json> [--cap ID] [--stage S]
                                      what the auditor does: re-derive a record from its own inputs
    seal.py show --out <dir>          the plan, readable
    seal.py census --pin <bin> --out <dir> [--box scripts/ny_box.sh]
                                      #admission-census: the router's own admission decisions over the moat corpus
                                      (both years, every scored category); `plan --census run|<tsv>` makes S8
                                      exactly the census-admitted rows and SR.seal checks the census
    seal.py plan ... --s8-mode certified [--cert-ledger <tsv>] [--x86-rig <json>] [--checker-budget-s N]
                                      #certified-moat (certlib): every S8 row ONCE in Competition, then per row an
                                      independent re-proof (`ny proof-check` ACCEPT) of every armed unsat, an x86
                                      replay of every armed sat, the classic sub-moat where the checker cannot ACCEPT
    seal.py run --out <dir> --stage S8C   the certified follow-ups (replays, re-proofs, checks, classic sub-moats)
    seal.py cert-ledger --ledger <tsv> <SEAL_OUT>...
                                      the ACCEPTed certificates of certified S8s (re-derived now) as a ledger a later
                                      plan takes with --cert-ledger
    seal.py plan ... --stages S8 --smoke-rows y/cat/idx,...
                                      a SMOKE plan: exactly those moat rows, S8 ungated, never evidence

Runner-only helpers (called by seal_run.sh): step0, pin-info, markers, loader-list, loader-bad, resolver-control,
cert-step0, cert-derive, cert-check-list, cert-scan.
Exit codes: 0 ok/PASS; 3 EMERGENCY; 4 PARTIAL; 5 VOID; 10 empty shard; 2 usage or refusal; 97 a plan/gate refusal.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seallib as S  # noqa: E402
from seallib import L  # noqa: E402
from seallib import C  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))


def _out(a) -> str:
    return os.path.abspath(a.out)


def cmd_plan(a) -> int:
    shards = a.shards
    if a.shard:
        k, n = S.parse_shard(a.shard)
        shards = n
    opts = {k: v for k, v in (("arrival_rows", a.arrival_rows), ("negctl_per_category", a.negctl_per_category),
                              ("s6_uniform", a.s6_uniform), ("s8_reps_c", a.s8_reps_c), ("s8_reps_p", a.s8_reps_p),
                              ("arrival_min_wall_frac", a.arrival_min_wall_frac),
                              ("checker_budget_s", a.checker_budget_s)) if v is not None}
    plan = S.build_plan(
        a.capability, a.pin, a.out, registry=a.registry, repo=a.repo, ny_root=a.ny_root,
        truth_paths={2026: a.truth_2026, 2025: a.truth_2025},
        stages=[s.strip() for s in a.stages.split(",") if s.strip()], shards=shards,
        categories=[c.strip() for c in a.categories.split(",")] if a.categories else None,
        years=[int(y) for y in a.years.split(",")] if a.years else None, router_shadow=a.router_shadow,
        seed=a.seed, opts=opts, check_files=not a.no_check_files, census=a.census, census_box=a.census_box,
        s8_mode=a.s8_mode, cert_ledger=a.cert_ledger,
        x86_rig=(S.read_json(a.x86_rig) if a.x86_rig else None),
        smoke_rows=[x.strip() for x in a.smoke_rows.split(",") if x.strip()] if a.smoke_rows else None)
    print(render_plan(plan))
    st = plan["stages"]
    print(f"SEAL_PLAN_DONE rc=0 capability={plan['capability']} stages={','.join(s for s in S.STAGES if s in st)} "
          f"rows={sum(v['rows'] for v in st.values())} runs={sum(v['runs'] for v in st.values())} "
          f"ceiling_h={sum(v['ceiling_h'] for v in st.values()):.2f} out={plan['out']}")
    return 0


def render_plan(plan: dict) -> str:
    o = []
    o.append(f"=== seal plan: {plan['capability']}  ({plan['out']})")
    o.append(f"  pin      {plan['pin']['path']} sha256={plan['pin']['sha256'][:16]} commit={plan['pin']['commit'][:12]} "
             f"features={plan['pin']['features']}")
    o.append(f"  configs  {plan['configs']['source']} (tree {plan['configs']['tree'][:12]}); router shadow: "
             f"{'on' if plan['configs']['router_shadow'] else 'off'}")
    o.append(f"  arm      " + ", ".join(f"{k}={json.dumps(v)}" for k, v in plan["arm"].items()))
    o.append(f"  need     " + " && ".join(plan["observable"]["need"]))
    o.append(f"  markers  " + "; ".join(f"{m!r}={c}" for m, c in plan["observable"]["pin_counts"].items()))
    o.append(f"  flight   " + (" && ".join(plan["observable"]["flight"]) or "(no LeverDecl on the armed keys)"))
    o.append(f"  scope    categories={plan['categories']} years={plan['years']} units={plan.get('units')}")
    if plan.get("census"):
        c = plan["census"]
        o.append(f"  census   {c['tsv']} census_sha256={c['census_sha256'][:16]} ({c['source']}): admits "
                 f"{c['counts']['admit']} instance(s) = {c['counts']['admit_moat_rows']} moat row(s) in "
                 f"{len(c['categories'])} categories (router alone {c['counts']['router']}); S8 = exactly those rows")
    meas = (plan.get("entry") or {}).get("measurement") or {}
    for k in ("targets", "metric", "kill"):
        if meas.get(k):
            o.append(f"  {k:<8} {str(meas[k])[:200]}")
    for k, v in plan["derivation"].items():
        o.append(f"  from     {k}: {v}")
    for arm, a in plan["configs"]["arms"].items():
        for rel, f in a["files"].items():
            o.append(f"  overlay  {arm}/{rel} ({', '.join(f['categories'])}): {len(f['actions'])} edit(s)")
    if plan.get("certified"):
        c = plan["certified"]
        led = c.get("ledger") or {}
        o.append(f"  certified S8: 1 armed Competition run per row; re-proof (ny proof-check ACCEPT, budget "
                 f"{c['checker']['budget_s']:.0f}s) of every armed unsat not covered by the ledger "
                 f"({led.get('entries', 0)} certificate(s)); x86 replay ({c['x86']['cpu']}, ORT {c['x86']['onnxruntime']}, "
                 f"{len(c['x86']['controls'])} controls) of every armed sat; classic 3c+1p sub-moat where the checker "
                 f"cannot ACCEPT; lost tolerance {c['lost_tolerance']}")
    if plan.get("smoke"):
        o.append(f"  SMOKE    rows {plan['smoke']['rows']}: S8 ungated, never evidence")
    for st in S.STAGES:
        if st in plan["stages"]:
            v = plan["stages"][st]
            extra = ""
            if st == "S5":
                extra = f" min_rows={v.get('min_rows')}"
            if st == "S7":
                kn = v.get("knife") or {}
                extra = (f" knife={kn.get('state')} (ladders={len(kn.get('ladders', []))} "
                         f"refused={len(kn.get('refused', []))} no_witness={len(kn.get('no_witness', []))})")
            o.append(f"  stage {st}  rows={v['rows']} runs={v['runs']} ceiling={v['ceiling_h']}h shards={v['shards']} "
                     f"specs={sorted(v['specs'].values())}{extra}")
    for n in plan.get("notes", []):
        o.append(f"  note     {n}")
    return "\n".join(o)


def cmd_census(a) -> int:
    """The admission census over the moat corpus, outside any plan: build the corpus (both years, every scored
    category), run `<pin> vnncomp census` on it (through `--box build --` when given), and print each routable
    capability's admitted instances / moat rows / categories (`--json` for the full record)."""
    out = _out(a)
    os.makedirs(out, exist_ok=True)
    _, entries = S.census_corpus(a.ny_root, a.configs, {2026: a.truth_2026, 2025: a.truth_2025})
    corpus = os.path.join(out, "corpus.tsv")
    if os.path.exists(corpus):
        raise S.SealError(f"{corpus} exists: a census directory is written once")
    S.write_new(corpus, S.corpus_text(entries))
    S.write_new(os.path.join(out, "corpus_map.json"),
                json.dumps({e["row_id"]: {"onnx": e["onnx"], "vnnlib": e["vnnlib"], "rows": e["rows"]}
                            for e in entries.values()}, indent=0, sort_keys=True) + "\n")
    c = S.run_census(os.path.abspath(a.pin), corpus, os.path.join(out, "census.tsv"), a.box)
    counts = S.census_counts(c, entries)
    rows = sum(len(e["rows"]) for e in entries.values())
    S.write_new(os.path.join(out, "counts.json"), json.dumps(counts, indent=1, sort_keys=True) + "\n")
    if a.json:
        print(json.dumps({"binding": c["binding"], "counts": counts}, indent=1, sort_keys=True))
    print(f"=== admission census {c['path']} census_sha256={c['census_sha256'][:16]}: {len(entries)} instance(s), "
          f"{rows} moat row(s), both years")
    for cap, v in counts.items():
        print(f"  {cap:28s} admit={v['admit']:5d} router={v['router']:5d} moat_rows={v['admit_moat_rows']:5d} "
              f"categories={len(v['categories'])} {sorted(v['categories'])[:6]}{' ...' if len(v['categories']) > 6 else ''}")
    print(f"SEAL_CENSUS_DONE rc=0 instances={len(entries)} moat_rows={rows} "
          + " ".join(f"{cap}={v['admit']}/{v['admit_moat_rows']}" for cap, v in counts.items())
          + f" census_sha256={c['census_sha256']} out={out}")
    return 0


def cmd_show(a) -> int:
    print(render_plan(S.load_plan(_out(a), verify_files=False)))
    return 0


def cmd_step0(a) -> int:
    out = _out(a)
    if os.path.exists(os.path.join(out, "EMERGENCY")):
        print(f"  EMERGENCY record present: {out}/EMERGENCY (nothing runs while it exists)")
        return 3
    plan = S.load_plan(out)
    st = plan["stages"].get(a.stage)
    if not st:
        print(f"SEAL_STEP0_REFUSED stage {a.stage} is not in the plan ({sorted(plan['stages'])})")
        return 97
    k, n = S.parse_shard(a.shard)
    if (st["shards"] == 1 and (k, n) != (1, 1)) or (st["shards"] > 1 and n != st["shards"]):
        print(f"SEAL_STEP0_REFUSED stage {a.stage} was planned with {st['shards']} shard(s), not {a.shard}")
        return 97
    gates = S.GATES[a.stage]
    if plan.get("smoke") and a.stage == "S8":
        print("  SMOKE plan: S8 runs ungated (named rows; verify_stage_record derives nothing from it)")
        gates = ()
    for g in gates:
        if g not in plan["stages"]:
            print(f"SEAL_STEP0_REFUSED gate {g} is not in the plan: {a.stage} cannot run ungated")
            return 97
        gp = os.path.join(out, g, "STAGE.json")
        if not os.path.isfile(gp):
            print(f"SEAL_STEP0_REFUSED gate {g} has no STAGE.json (run it first)")
            return 97
        state, note, rec = S.verify_stage_record(gp, plan["capability"], g)
        if (rec is None or state not in ("done", "partial") or rec.get("status") not in S.GATE_OK[g]
                or rec["pin"]["sha256"] != plan["pin"]["sha256"]):
            print(f"SEAL_STEP0_REFUSED gate {g}: {note}")
            return 97
        print(f"  gate {g}: {rec['status']} ({note[:160]})")
    sk = S.shard_key(k, n)
    if sk not in st["specs"]:
        print(f"  shard {sk} of {a.stage} holds no rows")
        return S.SHARD_EMPTY_RC
    spec = os.path.join(out, st["specs"][sk])
    rows = L.read_spec(spec)
    print(f"SPEC={spec}")
    print(f"M0OUT={S.m0_out(out, a.stage, sk)}")
    print(f"ROWS={len(rows)}")
    print(f"RUNS={sum(int(r['reps_c']) + int(r['reps_p']) for r in rows)}")
    return 0


def cmd_pin_info(a) -> int:
    plan = S.load_plan(_out(a), verify_files=False)
    p = plan["pin"]
    print(f"PIN_PATH={p['path']}")
    print(f"PIN_COPY={p['copy']}")
    print(f"PIN_SHA={p['sha256']}")
    print(f"PIN_SHA16={p['sha256'][:16]}")
    print(f"PIN_COMMIT={p['commit']}")
    print(f"PIN_FEATURES={p['features']}")
    print(f"CAPABILITY={plan['capability']}")
    print(f"ROUTER_SHADOW={1 if plan['configs']['router_shadow'] else 0}")
    if plan.get("certified"):
        ck = plan["certified"]["checker"]
        print(f"CERT_BUDGET_S={int(ck['budget_s'])}")
        print(f"CERT_TIMEOUT_S={int(ck['timeout_s'])}")
    return 0


def cmd_markers(a) -> int:
    plan = S.load_plan(_out(a), verify_files=False)
    seen = []
    for m in list(plan["observable"]["markers"]) + (["[router] cap="] if plan["configs"]["router_shadow"] else []):
        if m not in seen:
            seen.append(m)
    print("\n".join(seen))
    return 0


def _stage_presets(out: str, plan: dict, stage: str) -> list[str]:
    seen = []
    for r in S.stage_rows(out, plan, stage):
        if r["preset"] != "-":
            p = os.path.join(r["configs_dir"], r["preset"])
            if p not in seen:
                seen.append(p)
    return seen


def cmd_loader_list(a) -> int:
    out = _out(a)
    print("\n".join(_stage_presets(out, S.load_plan(out, verify_files=False), a.stage)))
    return 0


def cmd_loader_bad(a) -> int:
    """A NEGATIVE control for the loader probe: a copy of the stage's first overlay yaml with an unknown key next to
    the first armed key. The pin's preset loader must REFUSE it and name that key (deny_unknown_fields), which proves
    the probe validates fields -- so its acceptance of the overlay proves every armed key is a known typed key."""
    out = _out(a)
    plan = S.load_plan(out, verify_files=False)
    src = _stage_presets(out, plan, a.stage)[0]
    first = next(iter(plan["arm"]))
    files = [f for arm in plan["configs"]["arms"].values() for rel, f in arm["files"].items()
             if os.path.join(arm["dir"], rel) == src]
    written = (files[0].get("written_paths") or {}).get(first, first) if files else first
    parent = written.rsplit(".", 1)[0] if "." in written else ""
    bogus = "seal_negative_control_unknown_key"
    dotted = f"{parent}.{bogus}" if parent else bogus
    text = L.apply_overlay(open(src).read(), {dotted: True}, [], "seal loader negative control")
    os.makedirs(a.dest_dir, exist_ok=True)
    dst = os.path.join(a.dest_dir, "loader_negative_control.yaml")
    with open(dst, "w") as fh:
        fh.write(text)
    print(f"{dst}\t{bogus}")
    return 0


def cmd_resolver_control(a) -> int:
    """One row per (category, configs dir) of the stage for the resolver checks, and an EMPTY configs dir: the pin's
    own `vnncomp plan` must resolve the planned preset from the overlay dir (m0's step 0) and must NOT resolve it from
    the empty dir (this negative control proves the resolver check discriminates)."""
    out = _out(a)
    plan = S.load_plan(out, verify_files=False)
    empty = os.path.join(a.dest_dir, "empty_configs")
    os.makedirs(os.path.join(empty, "vnncomp26"), exist_ok=True)
    r = S.stage_rows(out, plan, a.stage)[0]
    print("\t".join([r["category"], r["onnx"], r["vnnlib"], r["budget"],
                     os.path.join(r["configs_dir"], r["preset"]) if r["preset"] != "-" else "-", empty]))
    return 0


def cmd_judge_stage(a) -> int:
    rec = S.judge_stage_from_out(_out(a), a.stage, write=not a.no_write, harness=a.harness)
    c = rec["counts"]
    print(f"=== seal {rec['stage']} {rec['status']} capability={rec['capability']} out={rec['out']}")
    for e in rec["emergencies"]:
        print(f"  EMERGENCY {e}")
    for n in rec["notes"]:
        print(f"  NOTE {n}")
    for r, k in rec["void_reasons"].items():
        print(f"  VOID x{k} {r}")
    for r in rec["reasons"]:
        print(f"  REASON {r}")
    if rec["stage"] == "S6":
        d = rec["decision"]
        print(f"  DECISION {d['decision']} converted={len(d['converted'])} lost={len(d['lost'])} pairs={d['pairs']} "
              f"arm_engaged_rows={d['arm_engaged_rows']} closes={d['closes_absolute']}")
    cert = ""
    if rec.get("certified"):
        cc = rec["certified"]
        print(f"  CERTIFIED accepted={len(cc['accepted'])} classic_covered={len(cc['classic_covered'])} "
              f"replayed={len(cc['replayed'])} ledger_covered={len(cc['ledger_covered'])} lost={len(cc['lost'])} "
              f"counts={json.dumps(cc['counts'], sort_keys=True)}")
        cert = (f" cert_accepted={len(cc['accepted'])} cert_classic={len(cc['classic_covered'])} "
                f"cert_replayed={len(cc['replayed'])}")
    eng_n = rec.get("engaged_row_count", 0)
    print(f"SEAL_STAGE status={rec['status']} stage={rec['stage']} scheduled={rec['scheduled_runs']} "
          f"ok={c.get('OK', 0)} void={c.get('VOID', 0)} notrun={c.get('notrun', 0)} "
          f"emergency={len(rec['emergencies'])} lost={c.get('lost', 0)} converted={c.get('converted', 0)} "
          f"engaged_rows={eng_n}{cert}")
    return rec["rc"]


def cmd_envelope(a) -> int:
    out = _out(a)
    recs = []
    for st in ("S7", "S8"):
        p = os.path.join(out, st, "STAGE.json")
        if os.path.isfile(p):
            recs.append(S.read_json(p))
    env = S.envelope_of(recs)
    print(json.dumps(env, indent=1, sort_keys=True))
    return 0 if env["ops"] else 5


def cmd_record(a) -> int:
    rep = S.record_into_registry(a.registry, _out(a), [s.strip() for s in a.stages.split(",")] if a.stages else None,
                                 replace=a.replace, dry_run=a.dry_run)
    print(json.dumps(rep, indent=1, sort_keys=True, default=str))
    print(f"SEAL_RECORD_DONE rc=0 written={int(bool(rep.get('written')))} capability={rep['capability']} "
          f"evidence={','.join(rep.get('evidence') or []) or '-'} registry={os.path.abspath(a.registry)}")
    return 0


def cmd_verify_record(a) -> int:
    state, note, rec = S.verify_stage_record(a.path, a.cap, a.stage)
    print(f"SEAL_VERIFY state={state} {note}")
    return 0 if state == "done" else (4 if state == "partial" else 5)


def cmd_cert_step0(a) -> int:
    """Step 0 of the certified follow-ups (no box): the plan is certified; phase A is CLEAN (every S8 row's armed
    Competition run judged OK, no emergency in it); the ledger parses; the x86 rig is the plan's (every file's
    sha256, the driver's constants, the control witnesses). Prints the work: re-proofs, replays, pending checks."""
    out = _out(a)
    plan = S.load_plan(out)
    rows, first, ledger, res = S.cert_phase_a(out, plan)
    cls = {r["tag"]: C.classify(r, first.get(r["tag"]), (ledger.get(r["tag"]) or {}).get("ok", False)) for r in rows}
    bad = [t for t, c in cls.items() if c in (C.C_NOTOK, C.C_EMERGENCY)]
    if bad:
        print(f"SEAL_STEP0_REFUSED phase A is not clean: {len(bad)} row(s) not OK or EMERGENCY (e.g. {bad[:3]}); "
              "re-run S8 (or triage) first")
        return 97
    import x86_replay as X
    probs = X.rig_check(plan["certified"]["x86"])
    sats = sum(1 for c in cls.values() if c == C.C_SAT)
    if probs and sats:
        for p in probs:
            print(f"  RIG-BAD {p}")
        print(f"SEAL_STEP0_REFUSED the x86 replay rig is not the plan's ({len(probs)} problem(s)) and {sats} armed-sat "
              "row(s) need it")
        return 97
    # review(certified-moat): every armed-sat row's checker kind needs a PASS and a FAIL control of that kind in the
    # plan's rig (certlib.controls_problem): refuse before any box wait rather than replay into INCONCLUSIVE
    kinds = C.control_kinds(plan["certified"]["x86"])
    unc = sorted({C.checker_kind(r) for r in rows if cls[r["tag"]] == C.C_SAT} - kinds)
    if unc:
        print(f"SEAL_STEP0_REFUSED the plan's x86 rig has no PASS + FAIL witness control of checker kind(s) {unc}, "
              f"which armed-sat row(s) need (controls cover {sorted(kinds)}): a replay that cannot be shown to fail on "
              "that code path proves nothing; re-plan with an --x86-rig whose controls cover them")
        return 97
    n = collections.Counter(cls.values())
    print("  phase A clean: " + " ".join(f"{k}={v}" for k, v in sorted(n.items())))
    print(f"  ledger: {sum(1 for v in ledger.values() if v['ok'])} row(s) covered by a re-verified certificate; "
          f"rig: {'ok' if not probs else 'NOT OK (no sat row needs it)'}")
    print(f"REPROOF={n.get(C.C_UNSAT_REPROVE, 0)}")
    print(f"REPLAY={sats}")
    return 0


def cmd_cert_derive(a) -> int:
    out = _out(a)
    plan = S.load_plan(out)
    path, rows, new = S.cert_derive(out, plan, a.phase)
    print(f"SPEC={path}")
    print(f"ROWS={len(rows)}")
    print(f"RUNS={sum(int(r['reps_c']) + int(r['reps_p']) for r in rows)}")
    print(f"M0OUT={C.run_dir(out, a.phase)}")
    print(f"WRITTEN={1 if new else 0}")
    return 0


def cmd_cert_check_list(a) -> int:
    out = _out(a)
    for t in S.cert_check_list(out, S.load_plan(out)):
        print("\t".join(t))
    return 0


def cmd_cert_scan(a) -> int:
    """The re-proof outcomes after the checks: exit 3 when any is an EMERGENCY (refuted, or a Proof sat)."""
    out = _out(a)
    reproof, outcomes = S.cert_reproof_outcomes(out, S.load_plan(out))
    n = collections.Counter(o["state"] for o in outcomes.values())
    for t, o in outcomes.items():
        if o["state"] in (C.R_EMERGENCY, C.R_PENDING):
            print(f"  {o['state'].upper()} {t}: {o['why'][:200]}")
    print("CERT_SCAN " + " ".join(f"{k}={n.get(k, 0)}" for k in (C.R_ACCEPT, C.R_FALLBACK, C.R_PENDING, C.R_EMERGENCY)))
    return 3 if n.get(C.R_EMERGENCY) else 0


def cmd_cert_ledger(a) -> int:
    entries = []
    for o in a.outs:
        e = S.cert_ledger_entries(o)
        print(f"  {o}: {len(e)} ACCEPTed certificate(s)")
        entries += e
    if os.path.exists(a.ledger):
        raise S.SealError(f"{a.ledger} exists (a ledger is written once; pick a new path)")
    S.write_new(a.ledger, C.ledger_text(entries))
    print(f"SEAL_LEDGER_DONE rc=0 entries={len(entries)} ledger={os.path.abspath(a.ledger)}")
    return 0


def cmd_run(a) -> int:
    mode = "--check" if a.check else "--snapshot" if a.snapshot else "--launch"
    env = dict(os.environ, SEAL_OUT=_out(a))
    args = ["bash", os.path.join(HERE, "seal_run.sh"), mode, "--stage", a.stage, "--shard", a.shard]
    sys.stdout.flush()
    os.execvpe("bash", args, env)
    return 0  # unreachable


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("capability")
    p.add_argument("--pin", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--registry", default=os.path.join(REPO, "docs", "engine_program_capabilities.yaml"))
    p.add_argument("--repo", default=REPO, help="git repository holding the pin's commit (configs are taken from it)")
    p.add_argument("--ny-root", default="<home>/ny", help="main tree: benchmarks/ and reports/ (gitignored)")
    p.add_argument("--truth-2026", default=os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt"))
    p.add_argument("--truth-2025", default=os.path.expanduser("~/vnncomp2025_results/SCORING-ZERO-TOL/results.txt"))
    p.add_argument("--stages", default=",".join(S.STAGES))
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", help="k/n: plan n shards (k is only a convenience for the printout)")
    p.add_argument("--categories")
    p.add_argument("--years")
    p.add_argument("--router-shadow", default="auto", choices=("auto", "on", "off"))
    p.add_argument("--seed")
    p.add_argument("--arrival-rows", type=int)
    p.add_argument("--arrival-min-wall-frac", type=float)
    p.add_argument("--negctl-per-category", type=int)
    p.add_argument("--s6-uniform", type=int)
    p.add_argument("--s8-reps-c", type=int)
    p.add_argument("--s8-reps-p", type=int)
    p.add_argument("--no-check-files", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--census", help="#admission-census: `run` (run the pin's `vnncomp census` over the moat corpus) or "
                                    "a census TSV made by the pin; S8 becomes exactly the census-admitted moat rows")
    p.add_argument("--census-box", help="run the census through `<this> build --` (scripts/ny_box.sh)")
    p.add_argument("--s8-mode", choices=("classic", "certified"),
                   help="#certified-moat: certified = every S8 row once in Competition + per-row independent evidence")
    p.add_argument("--cert-ledger", help="certified: earlier ACCEPTed certificates (seal.py cert-ledger)")
    p.add_argument("--x86-rig", help="certified: a JSON rig description (default certlib.DEFAULT_RIG)")
    p.add_argument("--checker-budget-s", type=float, help="certified: ny proof-check --budget-s")
    p.add_argument("--smoke-rows", help="year/category/index,...: a SMOKE plan of exactly these moat rows (S8 only)")
    p = sub.add_parser("census")
    p.add_argument("--pin", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--configs", default=os.path.join(REPO, "configs"),
                   help="configs dir moat_rows resolves presets from (rows do not depend on it)")
    p.add_argument("--box", help="run the census through `<this> build --` (scripts/ny_box.sh)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--ny-root", default="<home>/ny", help="main tree: benchmarks/ and reports/ (gitignored)")
    p.add_argument("--truth-2026", default=os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt"))
    p.add_argument("--truth-2025", default=os.path.expanduser("~/vnncomp2025_results/SCORING-ZERO-TOL/results.txt"))
    p = sub.add_parser("show")
    p.add_argument("--out", required=True)
    for name in ("step0", "judge-stage", "loader-list", "loader-bad", "resolver-control"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--stage", required=True, choices=S.STAGES)
        p.add_argument("--shard", default="1/1")
        if name == "judge-stage":
            p.add_argument("--no-write", action="store_true")
            p.add_argument("--harness")
        if name in ("loader-bad", "resolver-control"):
            p.add_argument("--dest-dir", required=True)
    for name in ("pin-info", "markers", "envelope"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
    p = sub.add_parser("record")
    p.add_argument("--out", required=True)
    p.add_argument("--registry", default=os.path.join(REPO, "docs", "engine_program_capabilities.yaml"))
    p.add_argument("--stages")
    p.add_argument("--replace", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("verify-record")
    p.add_argument("path")
    p.add_argument("--cap")
    p.add_argument("--stage")
    for name in ("cert-step0", "cert-check-list", "cert-scan"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
    p = sub.add_parser("cert-derive")
    p.add_argument("--out", required=True)
    p.add_argument("--phase", required=True, choices=("reproof", "classic"))
    p = sub.add_parser("cert-ledger")
    p.add_argument("--ledger", required=True)
    p.add_argument("outs", nargs="+")
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--stage", required=True, choices=S.STAGES + ("S8C",))
    p.add_argument("--shard", default="1/1")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--snapshot", action="store_true")
    g.add_argument("--launch", action="store_true")
    a = ap.parse_args(argv)
    fn = {"plan": cmd_plan, "show": cmd_show, "step0": cmd_step0, "pin-info": cmd_pin_info, "markers": cmd_markers,
          "loader-list": cmd_loader_list, "loader-bad": cmd_loader_bad, "resolver-control": cmd_resolver_control,
          "judge-stage": cmd_judge_stage, "envelope": cmd_envelope, "record": cmd_record,
          "verify-record": cmd_verify_record, "run": cmd_run, "census": cmd_census, "cert-step0": cmd_cert_step0,
          "cert-derive": cmd_cert_derive, "cert-check-list": cmd_cert_check_list, "cert-scan": cmd_cert_scan,
          "cert-ledger": cmd_cert_ledger}[a.cmd]
    try:
        return fn(a)
    except L.M0Error as e:
        print(f"SEAL_REFUSED ({a.cmd}): {e}", file=sys.stderr)
        print(f"SEAL_REFUSED rc=2 cmd={a.cmd} reason={str(e)[:200]!r}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
