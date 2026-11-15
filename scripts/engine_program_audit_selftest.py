#!/usr/bin/env python3
"""Fail-closed self-test for scripts/engine_program_audit.py.

Each case builds a small synthetic input in a temporary directory and asserts that the auditor REFUSES
to count it, or errors, rather than passing it. These are the traps CLAUDE.md records: an empty grep
counted as absence, a match that exists only in a comment or in test code, a `test ... ok` that belongs
to stderr, a leg file overwritten by a re-run, a dirty gate, a device test that printed SKIP=VOID, a
yaml arming that no scored category loads, and a missing pin / gate log / gate script / config year /
benchmark root. No cargo, no binary, no box. Exit 0 = every case held; 1 = a case failed.

    python3 scripts/engine_program_audit_selftest.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_program_audit as A  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        FAILS.append(what)


def write(path: str, text: str, mtime: float | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


RUST = '''//! A module doc that mentions [marker] in a comment only.
/* block comment with a brace { and [marker] */
pub fn live() {
    eprintln!("[live-marker] armed=1 x={}", 1);
    let s = "a string with a { brace";
    helper();
}

fn helper() {}

pub fn defined_only() {}

pub fn called_in_tests_only() {}

#[cfg(test)]
mod tests {
    #[test]
    fn t() {
        eprintln!("[test-marker] only here");
        super::called_in_tests_only();
    }
}
'''


def case_source(tmp: str) -> None:
    print("source scanning")
    write(os.path.join(tmp, "crates/x/src/lib.rs"), RUST + "#[cfg(test)]\nmod extra_checks;\n")
    write(os.path.join(tmp, "crates/x/src/extra_checks.rs"), "fn uses() { super::defined_only(); }\n")
    write(os.path.join(tmp, "crates/x/Cargo.toml"), "# depends on nothing-real\n[dependencies]\nreal = { path = \"../real\" }\n")
    src = A.SourceIndex(tmp)
    sf = src.file("crates/x/src/lib.rs")
    check(A.find_code_line(sf, "[marker]") is None, "a marker that exists only in comments is absent")
    check(A.find_code_line(sf, "[live-marker] armed=1") == 4, "a marker in a production string literal is found")
    check(A.find_code_line(sf, "[test-marker]") is None, "a marker inside #[cfg(test)] mod tests is absent")
    check(sf.in_test(sf.line_of(sf.text.index("fn t()"))), "the #[cfg(test)] module is a test region")
    check("crates/x/src/extra_checks.rs" in src.test_only(), "a file reached only by #[cfg(test)] mod x; is test-only")
    check(src.prod_uses("helper") == ["crates/x/src/lib.rs:6"], "a production call is found")
    check(src.prod_uses("called_in_tests_only") == [], "a call only from test code is not a production caller")
    check(src.prod_uses("defined_only") == [], "a call only from a test-only FILE is not a production caller")
    toml = src.file("crates/x/Cargo.toml")
    check(A.find_code_line(toml, "nothing-real") is None, "a Cargo.toml `#` comment is not a dependency line")
    check(A.find_code_line(toml, "real = {") == 3, "a real Cargo.toml dependency line is found")


def case_verdicts() -> None:
    print("test verdict parsing")
    lines = ["test a::b ... [stderr] status ok", "FAILED", "test c ... ok"]
    check(A.verdict_from(" [stderr] status ok", lines, 0) == "FAILED", "a stderr tail ending in ' ok' is not a verdict")
    lines = ["test a::b ... [bbe] armed", "  noise ok", "ok", "test c ... ok"]
    check(A.verdict_from(" [bbe] armed", lines, 0) == "ok", "the bare `ok` line after interleaved stderr is the verdict")


def case_retired() -> None:
    print("retired keys are audited in reverse")
    path = "margin_row.k"
    delivered = {"deliver": "PresetKey", "key": path, "line": 1}
    withheld = {"deliver": "EnvOnly", "key": None, "line": 1}
    clean = A.retired_key_findings(path, withheld, "margin_row.k is RETIRED", 42, None, False)
    check(clean == [], f"a refused, undelivered, unbridged, unledgered retired key holds ({clean})")
    check(any("still a space.rs Deliver::PresetKey" in f
              for f in A.retired_key_findings(path, delivered, "margin_row.k is RETIRED", 42, None, False)),
          "a retired key that is still a PresetKey axis is NOT retired")
    check(any("no production refusal" in f
              for f in A.retired_key_findings(path, withheld, "margin_row.k is RETIRED", None, None, False)),
          "a retired key that preset validation does not refuse is NOT retired")
    check(any("records no `retired.refusal`" in f
              for f in A.retired_key_findings(path, withheld, None, None, None, False)),
          "a retired key with no recorded refusal string cannot pass")
    check(any("still bridged" in f
              for f in A.retired_key_findings(path, withheld, "margin_row.k is RETIRED", 42, 7, False)),
          "a retired key the bridge still writes is NOT retired")
    check(any("unarmed-promotion" in f or "preset_unarmed_promotions.tsv" in f
              for f in A.retired_key_findings(path, withheld, "margin_row.k is RETIRED", 42, None, True)),
          "a retired key with a promotions-ledger row is NOT retired")
    # Fail closed over EVERY Axis literal: the auditor's first-literal-per-name map would let an early EnvOnly
    # literal (UNSAFE_AXES comes first in space.rs) mask a later one that still delivers the key.
    renamed = {"deliver": "PresetKey", "key": "margin_row.k_v2", "line": 9}  # gitleaks:allow (an axis key name, not a credential)
    other_lever = {"deliver": "PresetKey", "key": path, "line": 11, "name": "NY_OTHER"}
    for what, axes in [("an EnvOnly literal first and a delivering literal after it", [withheld, delivered]),
                       ("a re-promotion of the same lever under another yaml spelling", [withheld, renamed]),
                       ("another lever's axis delivering the retired yaml path", [withheld, other_lever])]:
        check(any("still a space.rs Deliver::PresetKey" in f
                  for f in A.retired_key_findings(path, axes, "margin_row.k is RETIRED", 42, None, False)),
              f"a retired key is NOT retired with {what}")
    check(A.retired_key_findings(path, [withheld], "margin_row.k is RETIRED", 42, None, False) == [],
          "a retired key whose only literal is EnvOnly holds (list form)")


def gate_block(tag: str, ts: str, head: str, prefix: str, dirty: str = "0") -> str:
    return (f"=== {tag} heartbeat start {ts} pid=1\n  HEAD: {head} subject dirty={dirty}\n"
            f"  [{prefix}_prop] passed=1 failed=0 binaries=1\n{tag}_DONE rc=0 WORST=0\n")


def case_gates(tmp: str) -> None:
    print("gate runs and legs")
    d = os.path.join(tmp, "gates")
    now = time.time()
    ts_old = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - 7200)) + "-07:00"
    ts_new = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - 3600)) + "-07:00"
    ts_future = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now + 3600)) + "-07:00"
    # two blocks with the same prefix: the leg file belongs to the newer one only
    write(os.path.join(d, "g.log"), gate_block("G", ts_old, "aaaaaaaaa", "g1") + gate_block("G", ts_new, "bbbbbbbbb", "g1"))
    write(os.path.join(d, "g1_prop.txt"), "[ny-box] cmd: cargo test\ntest m::x ... ok\n", mtime=now)
    # a dirty block, and a block whose leg predates it
    write(os.path.join(d, "h.log"), gate_block("H", ts_new, "ccccccccc", "h1", dirty="1"))
    write(os.path.join(d, "h1_prop.txt"), "test m::y ... ok\n", mtime=now)
    write(os.path.join(d, "k.log"), gate_block("K", ts_future, "ddddddddd", "k1"))
    write(os.path.join(d, "k1_prop.txt"), "test m::z ... ok\n", mtime=now)
    # device tests: captured output (no --nocapture) and a visible SKIP=VOID
    write(os.path.join(d, "v.log"), gate_block("V", ts_old, "eeeeeeeee", "v1") + gate_block("W", ts_old, "fffffffff", "w1"))
    write(os.path.join(d, "v1_prop.txt"), "[ny-box] cmd: cargo test -p ny-cuda\ntest s::dev_t ... ok\n", mtime=now)
    write(os.path.join(d, "w1_prop.txt"), "[ny-box] cmd: cargo test -- --nocapture\n"
          "[cuda-fd-kat] dev_t: SKIP=VOID (libcuda=false) — no CUDA\ntest s::dev_t ... ok\n"
          "test s::dev_u ... ok\n", mtime=now)
    pending: list[dict] = []
    runs = []
    for log in ("g.log", "h.log", "k.log", "v.log"):
        runs += A.parse_gate_log(os.path.join(d, log), pending)
    A.attach_legs(runs)
    by = {(r.prefix, r.head): r for r in runs}
    check(by[("g1", "aaaaaaaaa")].excluded is not None and not by[("g1", "aaaaaaaaa")].status,
          "an older block whose leg files a re-run overwrote owns no test results")
    check(by[("g1", "bbbbbbbbb")].status.get("x") == "ok", "the newest block owns the leg files")
    check(by[("h1", "ccccccccc")].excluded is not None, "a dirty gate run is excluded")
    check("z" not in by[("k1", "ddddddddd")].status, "a leg file older than its block is stale and ignored")
    v, w = by[("v1", "eeeeeeeee")], by[("w1", "fffffffff")]
    check(v.status.get("dev_t") == "ok" and not v.device_ok.get("dev_t"),
          "a device test ok from a leg that captured its output is unproven")
    check(not w.device_ok.get("dev_t"), "a device test that printed SKIP=VOID is unproven")
    check(w.device_ok.get("dev_u") is True, "a device test ok with output visible and no SKIP is proven")


def case_configs(tmp: str) -> None:
    print("preset resolution")
    c = os.path.join(tmp, "cfgtree/configs")
    write(os.path.join(c, "vnncomp26/cgan2026.yaml"), "bab: {x: 1}\n")
    write(os.path.join(c, "vnncomp25/cifar100_2024.yaml"), "margin_row: {domain_stack: false}\n")
    write(os.path.join(c, "vnncomp24/cifar100.yaml"), "margin_row: {domain_stack: true}\n")
    write(os.path.join(c, "vnncomp24/orphan_bench.yaml"), "margin_row: {domain_stack: true}\n")
    write(os.path.join(c, "vnncomp26/cgan2026_x_canary.yaml"), "margin_row: {domain_stack: true}\n")
    write(os.path.join(c, "vnncomp25/broken.yaml"), "margin_row: [unclosed\n")
    errs: list[str] = []
    cfg = A.Configs(os.path.join(tmp, "cfgtree"), ["cifar100_2024", "cgan2026"], errs)
    key = {"path": "margin_row.domain_stack", "default": False}
    hits = {h["file"]: h for h in cfg.arming(key)}
    h24 = hits["configs/vnncomp24/cifar100.yaml"]
    check(h24["armed"] and h24["shadowed_by"] and "cifar100_2024.yaml" in h24["shadowed_by"],
          "vnncomp24/cifar100.yaml is shadowed by vnncomp25/cifar100_2024.yaml for category cifar100_2024")
    check(bool(hits["configs/vnncomp24/orphan_bench.yaml"]["shadowed_by"]), "a yaml no scored category loads is a placebo")
    check(hits["configs/vnncomp26/cgan2026_x_canary.yaml"]["canary"], "a canary is listed as a canary")
    check(any("broken.yaml does not parse" in e for e in errs), "an unparseable yaml is an ERROR, not an empty config")
    errs2: list[str] = []
    A.Configs(os.path.join(tmp, "no-such-tree"), [], errs2)
    check(sum("missing or empty" in e for e in errs2) == 3, "missing config year directories are ERRORS")


def git(cwd: str, *args: str) -> None:
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t"})


def case_inputs(tmp: str) -> None:
    print("missing inputs fail closed")
    tree = os.path.join(tmp, "tree")
    write(os.path.join(tree, "crates/x/src/lib.rs"), RUST)
    write(os.path.join(tree, "configs/vnncomp26/a.yaml"), "{}\n")
    write(os.path.join(tree, "configs/vnncomp25/a.yaml"), "{}\n")
    write(os.path.join(tree, "configs/vnncomp24/a.yaml"), "{}\n")
    git(tree, "init", "-q")
    git(tree, "add", "-A")
    git(tree, "commit", "-q", "-m", "init")
    git(tree, "update-ref", "refs/remotes/origin/main", "HEAD")
    scratch = os.path.join(tmp, "scratch")
    os.makedirs(scratch, exist_ok=True)
    reg = os.path.join(tmp, "reg.yaml")
    write(reg, f"""scratch_root: {scratch}
defaults:
  main_ref: origin/main
  pins: {{latest: /nonexistent/ny_x_abc1234, shipped: /nonexistent/s, x86: /nonexistent/x}}
  gate_logs: [no_such_gate.log]
  gate_script: no_such_gate.sh
  benchmark_roots: [/nonexistent/benchmarks]
capabilities:
  - id: c
    family: f
    name: c
    heads: []
    consumers:
      - {{file: crates/x/src/lib.rs, grep: "[marker]"}}
    callers:
      - {{what: renamed symbol, regex: "no_such_fn\\\\(", min: 1, severity: open, control: "fn no_such_fn"}}
    record: {{}}
""")
    out = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_program_audit.py"),
                          "--registry", reg, "--tree", tree, "--details"], capture_output=True, text=True)
    txt = out.stdout + out.stderr
    check(out.returncode != 0, f"an audit with unreadable inputs exits non-zero (rc={out.returncode})")
    for needle, what in (("latest pin unreadable", "a missing latest pin"),
                         ("shipped pin unreadable", "a missing shipped pin"),
                         ("x86 pin unreadable", "a missing x86 pin"),
                         ("gate log missing", "a missing gate log"),
                         ("gate script missing", "a missing gate script"),
                         ("benchmark root missing", "a missing benchmark root"),
                         ("positive control", "a callers count whose control matches nothing")):
        check(needle in txt, f"{what} is an ERROR")
    check("consumer MISSING crates/x/src/lib.rs /[marker]/: no production (non-comment, non-test) line matches" in txt,
          "a consumer that matches only a comment is not a consumer")
    check("renamed symbol: 0 (control at None)" in txt, "the failed control is reported next to its zero")


def case_routable() -> None:
    """SR (ROUTER_PROGRAM 3.5): n/a without a routable: block; never done without S7 + S8 from seal records, whatever
    the registry's envelope says (the end-to-end derivation from real seal records is pinned by
    scripts/seal/seal_selftest.py test_runner_seals_a_capability_end_to_end_and_the_record_derives)."""
    import types
    print("SR (routable) derivation")
    fake = types.SimpleNamespace(_seal_records={}, errors=[], path=lambda p: p)
    S = {s: A.StageResult("open") for s in A.STAGES}
    check(A.Auditor.audit_routable(fake, {"id": "x"}, S).state == "na", "a capability with no routable: block is n/a")
    cap = {"id": "x", "routable": {"kind": "tier", "envelope": ["conv", "relu"]}}
    check(A.Auditor.audit_routable(fake, cap, S).state == "partial",
          "a recorded envelope without S7 + S8 is partial, never sealed")
    done = dict(S, S7=A.StageResult("done"), S8=A.StageResult("done"))
    r = A.Auditor.audit_routable(fake, cap, done)
    check(r.state == "partial" and "not from seal records" in r.note,
          "S7 + S8 done without seal records cannot check the envelope against evidence rows")
    check(A.STAGES.index("SR") == A.STAGES.index("S8") + 1, "SR sits right after S8 on the ladder")


def case_sr_ladder() -> None:
    """review(eng11): the merged tree carries TWO readers of `record.SR` — the router's SR ladder (#router-r1 /
    #router-r5: decl / obs / seal; `derive_sr`, and router_r5_registry_ladders_match_the_capability_registry, which
    compiles S7/S8/obs from the record) and the seal pipeline's SR.seal stage (RANK states). A ladder level must not
    be an auditor error, and the router's ladder must never reach seal where SR.seal does not derive."""
    import types
    print("SR: the router's ladder vs the seal pipeline's SR.seal")
    env_cap = {"kind": "tier", "envelope": ["conv", "relu"]}

    def finish(record: dict, derived: dict, routable=env_cap):
        fake = types.SimpleNamespace(errors=[])
        cap = {"id": "x", "record": record, "routable": routable}
        stages = {s: A.StageResult("open") for s in A.STAGES}
        stages.update({s: A.StageResult(v) for s, v in derived.items()})
        res = {"flags": [], "stages": stages, "arming": {}}
        A.Auditor._finish(fake, cap, res)
        return fake.errors, [f for f in res["flags"] if f.startswith("REGRESSION")], res

    errs, reg, _ = finish({"SR": "obs"}, {})
    check(not errs and not reg, "a recorded SR.obs (the router ladder's own level) is neither an error nor a regression")
    errs, reg, _ = finish({"SR": "decl"}, {})
    check(not errs and not reg, "a recorded SR.decl is neither an error nor a regression")
    errs, reg, _ = finish({"SR": "seal"}, {"SR": "partial"})
    check(any(f.startswith("REGRESSION SR: recorded seal") for f in reg), "a recorded seal that does not derive regresses")
    errs, reg, res = finish({"SR": "obs"}, {"SR": "done"})
    check(not reg and any(f.startswith("improved SR") for f in res["flags"]), "a derived SR.seal over a recorded obs is improved")
    errs, reg, _ = finish({"SR": "sealed"}, {})
    check(any("bad recorded state SR=sealed" in e for e in errs), "an SR value in neither vocabulary is an error")
    errs, reg, _ = finish({"S7": "done", "S8": "done", "SR": "open"}, {"S7": "done", "S8": "done", "SR": "partial"})
    check(any("router's ladder reaches seal" in f for f in reg),
          "record S7 + S8 done + an envelope (the router's compiled ladder routes) without SR.seal derived regresses")
    errs, reg, _ = finish({"S7": "done", "S8": "done", "SR": "seal"}, {"S7": "done", "S8": "done", "SR": "done"})
    check(not reg and not errs, "the same ladder with SR.seal derived from seal records is clean (control)")
    errs, reg, _ = finish({"S7": "done", "S8": "done", "SR": "open"}, {"S7": "done", "S8": "done", "SR": "partial"},
                          routable={"kind": "tier", "envelope": []})
    check(not any("router's ladder reaches seal" in f for f in reg), "no envelope: the router's ladder is not at seal (control)")
    done = {s: "done" for s in A.STAGES[: A.STAGES.index("S8") + 1]}
    _, _, res = finish(dict(done, SR="seal"), dict(done, SR="done"))
    check(res["recorded_ladder"] == "SR", "a recorded SR: seal counts as SR done on the recorded ladder")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="epa-selftest-") as tmp:
        case_routable()
        case_sr_ladder()
        case_source(tmp)
        case_verdicts()
        case_retired()
        case_gates(tmp)
        case_configs(tmp)
        case_inputs(tmp)
    print(f"SELFTEST_DONE rc={1 if FAILS else 0} failed={len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
