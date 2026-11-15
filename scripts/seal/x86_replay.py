#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""x86_replay: the certified S8's WRONG-SAT check (scripts/seal/README.md, "Certified S8").

Every armed-sat row's witness (its phase-A results file) is replayed on the organisers' OWN checker code under
emulated x86-64 (qemu-user `-cpu Haswell-v4`, ORT 1.26.0: the rig measured in scratchpad/x86replay and
scratchpad/x86parity, which reproduces the organisers' graded witnesses 100/100 and 67/67), exactly as
scratchpad/x86parity/inner_x86parity_v1.sh does it:

    v1-2025 / v1-2026   checkv1.py (the year's counterexamples.get_ce_diff, unmodified) under qemu, dumping feeds
    v2-2026             x86replay/driver.py --mode check --dump-feeds (the official counterexamples_v2)
    then                x86replay/driver.py --mode ort on the same feeds, bitwise-compared (cmp_npz.py)

and each batch first runs the plan's witness CONTROLS through the same code path: a known-good witness must PASS and a
known-bad one must FAIL, or no row of that batch is judged (a replay that cannot fail proves nothing).

    x86_replay.py batch --out <SEAL_OUT>      every phase-A sat row without a final record (run under
                                              `scripts/ny_box.sh build --`); last line
                                              X86_REPLAY_DONE rc=<0|3|5> rows= passed= failed= inconclusive= controls=
    x86_replay.py rig-check --out <SEAL_OUT>  step 0: the rig's files are the plan's, its driver's constants are the
                                              plan's paths (no box, no emulation)

rc 3 = a witness the checker rejects on the certified platform (a WRONG SAT: EMERGENCY, stop at once); 5 = a row
or the controls inconclusive; 0 = every row PASSED. The judge re-verifies each record (certlib.verify_replay_record).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certlib as C  # noqa: E402


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _set_aside(path: str) -> None:
    """A re-run never deletes: the previous file is renamed beside it."""
    if os.path.exists(path):
        os.replace(path, f"{path}.stale.{_dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')}")


def _run(cmd: list[str], cwd: str, log: str, timeout: int, env_extra: dict | None = None) -> int:
    env = dict(os.environ)
    env.update(env_extra or {})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        try:
            p = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, env=env,
                               stdin=subprocess.DEVNULL)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 124
        except OSError as e:
            fh.write(f"OSError: {e}\n")
            rc = 127
        fh.write(f"\nX86_STEP_RC rc={rc}\n")
    return rc


def benchmark_dir(onnx: str) -> str:
    parts = os.path.normpath(onnx).split(os.sep)
    if "onnx" not in parts:
        raise C.CertError(f"{onnx}: no `onnx` directory component (the driver resolves models below it)")
    i = len(parts) - 1 - parts[::-1].index("onnx")
    return os.sep.join(parts[:i]) or os.sep


def replay_core(rig: dict, kind: str, onnx: str, vnnlib: str, ce: str, dest: str, rid: str) -> dict:
    """One witness through the rig: the official check (v1 or v2 path) and the ORT replay of its feeds.
    Returns {result, check: {record, record_sha256, rc, log}, ort: {state, cmp, rc, log}}."""
    safe = rid.replace(":", "__")
    work, xr, cpu = rig["work"], rig["xr_dir"], rig["cpu"]
    timeout = int(rig.get("timeout_s", 1500))
    feeds = os.path.join(work, "feeds", f"{safe}.npz")
    chk_out = os.path.join(work, "out_sealchk", f"{safe}.npz")
    ort_out = os.path.join(work, "out_sealort", f"{safe}.npz")
    for p in (feeds, chk_out, ort_out):
        _set_aside(p)
    manifest = os.path.join(dest, f"{safe}.manifest.json")
    os.makedirs(dest, exist_ok=True)
    job = {"id": rid, "category": "seal", "tool": "ny-seal", "benchmark_dir": benchmark_dir(onnx),
           "network_field": onnx, "property_field": vnnlib, "ce_path": ce, "dump_feeds": feeds}
    _write_json(manifest, [job])
    record = os.path.join(dest, "check.json")
    _set_aside(record)
    chk_log = os.path.join(dest, "check.log")
    if kind == "v2-2026":
        rc = _run([sys.executable, rig["driver"], "--manifest", manifest, "--platform", f"x86:{cpu}", "--mode", "check",
                   "--dump-feeds", "--tag", "sealchk", "--ids", rid, "--timeout", str(timeout)], xr, chk_log, timeout + 120)
        res_file = os.path.join(xr, "results", f"sealchk__{safe}.manifest__check.jsonl")
        last = None
        try:
            for ln in open(res_file):
                try:
                    d = json.loads(ln)
                except ValueError:
                    continue
                if d.get("id") == rid:
                    last = d
        except OSError:
            pass
        if last is not None and rc == 0 and not last.get("error"):
            _write_json(record, last)
    elif kind in ("v1-2025", "v1-2026"):
        scoring = rig["scoring"]["2025" if kind == "v1-2025" else "2026"]
        rc = _run([rig["qemu"], "-cpu", cpu, "-L", rig["sysroot"], rig["x86py"], rig["checkv1"], "--scoring", scoring,
                   "--onnx", onnx, "--vnnlib", vnnlib, "--ce", ce, "--feeds", feeds, "--outputs", chk_out,
                   "--json-out", record], work, chk_log, timeout, {"QEMU_CPU_MODEL": cpu})
        if rc != 0:
            _set_aside(record)
    else:
        raise C.CertError(f"unknown checker kind {kind!r}")
    rec = C._json(record) if os.path.isfile(record) else None
    result = rec.get("result") if isinstance(rec, dict) and isinstance(rec.get("result"), str) else "rig_error"
    out = {"result": result, "check": {"record": record, "record_sha256": C.sha256_file(record), "rc": rc,
                                       "log": chk_log}}
    ort_log = os.path.join(dest, "ort.log")
    if not os.path.isfile(feeds):
        out["ort"] = {"state": "err", "cmp": "no feeds dumped by the check", "rc": None, "log": ort_log}
        return out
    orc = _run([sys.executable, rig["driver"], "--manifest", manifest, "--platform", f"x86:{cpu}", "--mode", "ort",
                "--tag", "sealort", "--ids", rid, "--timeout", str(timeout)], xr, ort_log, timeout + 120)
    cmp_log = os.path.join(dest, "cmp.log")
    crc = _run([rig["armpy"], rig["cmp"], chk_out, ort_out], dest, cmp_log, 600)
    line = ([ln for ln in (C._read(cmp_log) or "").splitlines() if ln.startswith("CMP ")] or ["CMP err none"])[-1]
    state = "match" if (orc == 0 and crc == 0 and line.startswith("CMP match")) else (
        "diff" if line.startswith("CMP diff") else "err")
    out["ort"] = {"state": state, "cmp": line, "rc": orc, "cmp_rc": crc, "log": ort_log}
    return out


def run_controls(rig: dict, dest_root: str, batch: str) -> str:
    """The batch's witness controls through replay_core: every `expect: pass` control must PASS (correct and ORT
    match), every `expect: fail` control must FAIL (an official non-correct result). Returns the controls file."""
    ctl = []
    for c in rig.get("controls") or []:
        dest = os.path.join(dest_root, "controls", batch, c["name"])
        cid = f"sealctl:{batch}:{c['name']}"
        r = replay_core(rig, c["kind"], c["onnx"], c["vnnlib"], c["ce"], dest, cid)
        if r["result"] in C.WITNESS_PASS and r["ort"]["state"] == "match":
            got = "pass"
        elif r["result"] in C.WITNESS_FAIL:
            got = "fail"
        else:
            got = "error"
        ctl.append({"name": c["name"], "id": cid, "kind": c["kind"], "expect": c["expect"], "got": got,
                    "result": r["result"],
                    "ort": r["ort"]["state"], "record": r["check"]["record"],
                    "record_sha256": r["check"]["record_sha256"], "ce_sha256": C.sha256_file(c["ce"])})
    doc = {"schema": C.CONTROLS_SCHEMA, "batch": batch, "created": now(), "cpu": rig["cpu"],
           "ok": bool(ctl) and all(x["got"] == x["expect"] for x in ctl), "controls": ctl}
    path = os.path.join(dest_root, "controls", f"{batch}.json")
    _write_json(path, doc)
    return path


def replay_row(rig: dict, row: dict, witness_res: str, dest: str, controls: str, plan_sha16: str) -> dict:
    """Replay one row's phase-A witness and write its record (dest/replay.json)."""
    os.makedirs(dest, exist_ok=True)
    ce = os.path.join(dest, "witness.counterexample")
    _set_aside(ce)
    with open(ce, "w") as fh:
        fh.write(C.witness_text(C._read(witness_res) or ""))
    kind = C.checker_kind(row)
    rid = f"seal:{plan_sha16}:{row['tag']}"
    models = C.L.model_files(row["onnx"])
    if len(models) != 1:
        # a paired-model (relational) row: the rig replays one network; never a pass, never a fail (inconclusive)
        r = {"result": "rig_unsupported_multimodel", "check": {"record": None, "record_sha256": None, "rc": None,
                                                               "log": None},
             "ort": {"state": "err", "cmp": f"{len(models)} models: the rig replays one network", "rc": None}}
    else:
        r = replay_core(rig, kind, models[0], row["vnnlib"], ce, dest, rid)
    rec = {"schema": C.REPLAY_SCHEMA, "tag": row["tag"], "id": rid, "onnx": row["onnx"], "vnnlib": row["vnnlib"],
           "year": row["year"], "category": row["category"], "checker": kind,
           "witness_res": witness_res, "witness_res_sha256": C.sha256_file(witness_res),
           "ce": ce, "ce_sha256": C.sha256_file(ce), "result": r["result"], "check": r["check"], "ort": r["ort"],
           "controls": {"path": controls, "sha256": C.sha256_file(controls)},
           "rig": {"cpu": rig["cpu"], "onnxruntime": rig["onnxruntime"], "files_sha256": C.rig_files_digest(rig)},
           "created": now()}
    path = os.path.join(dest, "replay.json")
    _set_aside(path)
    _write_json(path, rec)
    return rec


def rig_check(rig: dict) -> list[str]:
    """The rig's files are the plan's (sha256), and its driver's hard-coded constants are the plan's paths."""
    bad = C.rig_problems(rig)
    probe = ("import json, sys; sys.path.insert(0, sys.argv[1]); import driver; print('RIGCONST ' + json.dumps("
             "{'qemu': str(driver.QEMU), 'sysroot': str(driver.SYSROOT), 'x86py': str(driver.X86PY), "
             "'work': str(driver.WORK), 'replay': str(driver.REPLAY)}))")
    try:
        p = subprocess.run([sys.executable, "-c", probe, rig["xr_dir"]], capture_output=True, text=True, timeout=60,
                           stdin=subprocess.DEVNULL)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("RIGCONST ")]
        got = json.loads(line[-1][len("RIGCONST "):]) if line else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        got = None
    if got is None:
        bad.append(f"the rig's driver.py does not import from {rig['xr_dir']}")
    else:
        for k in ("qemu", "sysroot", "x86py", "work", "replay"):
            if os.path.normpath(got[k]) != os.path.normpath(rig[k]):
                bad.append(f"driver.py's {k} is {got[k]}, the plan's rig says {rig[k]}")
    return bad


def _plan_and_rows(out: str):
    import seallib as S
    plan = S.load_plan(out)
    if not C.is_certified(plan):
        raise C.CertError(f"{out}: the plan's S8 is not certified")
    return S, plan


def sat_rows(S, out: str, plan: dict) -> list[tuple[dict, str]]:
    """(phase-A row, its competition r1 results file) for every phase-A row whose OK judgement says sat."""
    rows = S.stage_rows(out, plan, "S8")
    lines = []
    for p in S.judged_paths(out, plan, "S8").values():
        lines += C.L.read_judged(p)
    by_key = C.L.collapse_runs(lines)
    picked = []
    for r in rows:
        j = by_key.get((r["tag"], "competition", 1))
        if C.classify(r, j, False) == C.C_SAT:
            base = S.m0_out(out, "S8", plan["stages"]["S8"]["shard_of"][r["tag"]])
            picked.append((r, C.res_path(base, r["tag"], "competition", 1)))
    return picked


def cmd_batch(a) -> int:
    out = os.path.abspath(a.out)
    S, plan = _plan_and_rows(out)
    rig = plan["certified"]["x86"]
    todo, kept_fail = [], []
    for r, res in sat_rows(S, out, plan):
        st = C.verify_replay_record(C.replay_record_path(out, r["tag"]), r, res, rig)["state"]
        if st in (C.X_PASS, C.X_FAIL):
            print(f"  replay {r['tag']}: {st} (record kept)")
            if st == C.X_FAIL:
                kept_fail.append(r["tag"])
            continue
        todo.append((r, res))
    if kept_fail:
        # review(certified-moat): a KEPT failing record is still a WRONG SAT -- a resumed batch must not report rc=0
        # (and let the runner spend re-proofs and sub-moats) because the failure was recorded by an earlier batch
        print(f"  ###### WRONG SAT (recorded earlier): {kept_fail[:5]} -- STOPPING ######")
        print(f"X86_REPLAY_DONE rc=3 rows={len(todo)} passed=0 failed={len(kept_fail)} inconclusive=0 controls=kept "
              f"stopped_at={kept_fail[0]}")
        return 3
    print(f"=== x86 replay batch: {len(todo)} row(s) to replay out={out} {now()}")
    if not todo:
        print("X86_REPLAY_DONE rc=0 rows=0 passed=0 failed=0 inconclusive=0 controls=skipped")
        return 0
    bad = rig_check(rig)
    if bad:
        for b in bad:
            print(f"  RIG-BAD {b}")
        print(f"X86_REPLAY_DONE rc=5 rows={len(todo)} passed=0 failed=0 inconclusive={len(todo)} controls=rig-bad")
        return 5
    batch = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    dest_root = os.path.join(C.cert_dir(out), "x86")
    controls = run_controls(rig, dest_root, batch)
    cdoc = C._json(controls) or {}
    print(f"  controls {controls}: ok={cdoc.get('ok')} " + " ".join(f"{c['name']}={c['got']}(want {c['expect']}, "
                                                                     f"{c['result']})" for c in cdoc.get("controls", [])))
    if cdoc.get("ok") is not True:
        print(f"X86_REPLAY_DONE rc=5 rows={len(todo)} passed=0 failed=0 inconclusive={len(todo)} controls=fail")
        return 5
    n = {C.X_PASS: 0, C.X_FAIL: 0, C.X_INCONCLUSIVE: 0, C.X_PENDING: 0}
    sha16 = S.sha256_text(open(os.path.join(out, "plan.json")).read())[:16]
    for r, res in todo:
        replay_row(rig, r, res, os.path.dirname(C.replay_record_path(out, r["tag"])), controls, sha16)
        v = C.verify_replay_record(C.replay_record_path(out, r["tag"]), r, res, rig)
        n[v["state"]] += 1
        print(f"  replay {r['tag']} [{C.checker_kind(r)}]: {v['state']} result={v.get('result')} ({v['why'][:160]})")
        if v["state"] == C.X_FAIL:
            print(f"  ###### WRONG SAT: {r['tag']} -- the organisers' checker rejects the witness on x86: STOPPING ######")
            print(f"X86_REPLAY_DONE rc=3 rows={len(todo)} passed={n[C.X_PASS]} failed={n[C.X_FAIL]} "
                  f"inconclusive={n[C.X_INCONCLUSIVE]} controls=ok stopped_at={r['tag']}")
            return 3
    rc = 0 if n[C.X_PASS] == len(todo) else 5
    print(f"X86_REPLAY_DONE rc={rc} rows={len(todo)} passed={n[C.X_PASS]} failed={n[C.X_FAIL]} "
          f"inconclusive={n[C.X_INCONCLUSIVE] + n[C.X_PENDING]} controls=ok")
    return rc


def cmd_rig_check(a) -> int:
    out = os.path.abspath(a.out)
    _, plan = _plan_and_rows(out)
    bad = rig_check(plan["certified"]["x86"])
    for b in bad:
        print(f"  RIG-BAD {b}")
    print(f"X86_RIG_CHECK rc={1 if bad else 0} problems={len(bad)}")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("batch", "rig-check"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        return {"batch": cmd_batch, "rig-check": cmd_rig_check}[a.cmd](a)
    except C.L.M0Error as e:
        print(f"X86_REPLAY_DONE rc=2 refused={str(e)[:200]!r}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
