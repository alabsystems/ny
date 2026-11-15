#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""The CERTIFIED S8 mode of the seal pipeline (scripts/seal/README.md, "Certified S8: the argument").

A classic S8 runs every census-admitted moat row 3x Competition + 1x Proof. A certified S8 runs every row ONCE in
Competition with the capability armed (phase A) and then discharges each of the three things the moat exists to catch
by the cheapest INDEPENDENT evidence that covers it, per row:

  (1) a FALSE UNSAT     every armed-unsat row not already covered by a re-verified certificate (a banked-unsat row
                        whose instance has an ACCEPTed `ny proof-check` report in the plan's ledger) gets ONE
                        Proof-profile run (tag S8R_*) and `ny proof-check` on its bundle. ACCEPT (verdict=verified,
                        exit 0, every cover row claimed AND confirmed, the report, bundle, onnx and vnnlib hashes
                        bound to the row and the pin) certifies the INSTANCE unsat, so every unsat any run states on it
                        is correct. Anything short of ACCEPT -- unconfirmed (opaque claims), unsupported, invalid,
                        malformed, a checker fault, no published bundle, a Proof run that does not reproduce the unsat
                        -- falls back to the CLASSIC sub-moat for that row (tag S8F_*: 3 Competition + 1 Proof reps,
                        m0's per-row profile rules). `refuted` (a rigorous counterexample) is an EMERGENCY.
  (2) a WRONG SAT       every armed-sat row's witness is replayed on the organisers' checker code under emulated x86
                        (qemu Haswell-v4, ORT 1.26.0: scripts/seal/x86_replay.py, the scratchpad/x86replay rig).
                        A checker verdict other than correct / correct_up_to_tolerance on the certified platform is
                        an EMERGENCY; an unusable replay (rig error, ORT replay mismatch, controls that do not
                        discriminate) is INCONCLUSIVE, which blocks the seal.
  (3) a LOST verdict    every banked sat/unsat the armed run does not reproduce is LOST: reported, never a soundness
                        failure, but it blocks PASS above LOST_TOLERANCE (0).

Every decision is a function of the run artefacts, the pre-registered plan and the files on disk; `seallib`'s stage
judge calls `gather` (artefacts -> plain facts) and `decide` (facts -> verdict), and the auditor re-derives the whole
record through the same two functions (`seallib.verify_stage_record`). Nothing here runs a binary or the box.
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M0_DIR = os.path.join(os.path.dirname(HERE), "m0")
for _p in (HERE, M0_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import m0lib as L  # noqa: E402
import m0_judge as M0J  # noqa: E402

MODE_CLASSIC, MODE_CERTIFIED = "classic", "certified"
CERT_SCHEMA = "seal-certified/v1"
LEDGER_SCHEMA = "seal-cert-ledger/v1"
LEDGER_COLUMNS = ("onnx_sha256", "vnnlib_sha256", "report", "report_sha256", "bundle", "bundle_sha256",
                  "checker_build", "source")
REPLAY_SCHEMA = "seal-x86-replay/v1"
CONTROLS_SCHEMA = "seal-x86-controls/v1"
REPORT_SCHEMA = "ny-proof-check-report/v1"
BUNDLE_SCHEMA = "ny-proof-bundle/v2"
# a declared tolerance of lost banked verdicts: 0, a constant (a plan cannot raise it; the plan records it)
LOST_TOLERANCE = 0
CERT_DIR = "S8C"
PHASE_A_PREFIX, REPROOF_PREFIX, CLASSIC_PREFIX = "S8_", "S8R_", "S8F_"
CLASSIC_REPS = (3, 1)               # the classic S8 floor (seallib.S8_MIN_REPS_C / S8_MIN_REPS_P)
# a re-proof or classic row is an armed-UNSAT row: a sat stated on it by any later run is a flip, stopped at once
FLIP_ALARM = r"^Result: sat$"
CHECK_RAN_RE = re.compile(r"^CERT_CHECK_RAN rc=(-?[0-9]+)(?: |$)", re.M)
CHECK_DONE_RE = re.compile(r"^PROOF_CHECK_DONE rc=([0-9]+) verdict=(\S+) .*? report=(\S+) report_sha256=([0-9a-f]{64})\s*$",
                           re.M)
CHECKER_VERDICTS = ("verified", "unconfirmed", "invalid", "refuted", "malformed", "unsupported", "checker_fault")
WITNESS_PASS = ("correct", "correct_up_to_tolerance")
# every result the organisers' checkers can return that is NOT a correct witness (2025 + 2026 CounterexampleResult),
# plus checkv1's own `checker_exception` (the official checker crashing on our witness is not a pass)
WITNESS_FAIL = ("no_ce", "exec_doesnt_match", "spec_not_violated", "wrong_shape", "malformed_ce", "unsupported",
                "checker_exception")
X86_MACHINE = "x86_64"

# row classes after phase A
C_EMERGENCY, C_NOTOK = "emergency", "phase-a-not-ok"
C_UNSAT_LEDGER, C_UNSAT_REPROVE = "unsat-ledger", "unsat-reprove"
C_SAT, C_LOST, C_UNDECIDED = "sat-replay", "lost", "undecided"
# re-proof outcomes
R_ACCEPT, R_FALLBACK, R_PENDING, R_EMERGENCY = "accepted", "fallback", "pending", "emergency"
# replay outcomes
X_PASS, X_FAIL, X_INCONCLUSIVE, X_PENDING = "passed", "failed", "inconclusive", "pending"


class CertError(L.M0Error):
    """A refusal of the certified mode (an artefact it cannot vouch for)."""


# ---------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------


_SHA_MEMO: dict[tuple, str] = {}


_MEMO_MIN_BYTES = 1 << 20


def sha256_file(path: str) -> str | None:
    """sha256 of a file, None when it is unreadable. Files of 1 MiB and more (the models a 678-row moat names hundreds
    of times) are memoised on (path, size, mtime_ns, ctime_ns, inode); every smaller file -- a report, a bundle, a
    log, a results file, a replay record: what an edit would target -- is hashed afresh on every call, so a same-size
    rewrite inside one timestamp tick can never be served a stale hash."""
    try:
        st = os.stat(path)
        if st.st_size < _MEMO_MIN_BYTES:
            return L.sha256_file(path)
        key = (os.path.abspath(path), st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)
        if key not in _SHA_MEMO:
            _SHA_MEMO[key] = L.sha256_file(path)
        return _SHA_MEMO[key]
    except (OSError, TypeError):
        return None


def _read(path: str) -> str | None:
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _json(path: str):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def is_certified(plan: dict) -> bool:
    return ((plan.get("stages") or {}).get("S8") or {}).get("mode") == MODE_CERTIFIED


def cert_dir(out: str) -> str:
    return os.path.join(out, CERT_DIR)


def spec_path(out: str, phase: str) -> str:
    if phase not in ("reproof", "classic"):
        raise CertError(f"unknown certified phase {phase!r}")
    return os.path.join(cert_dir(out), "specs", f"{phase}.tsv")


def run_dir(out: str, phase: str) -> str:
    return os.path.join(cert_dir(out), f"run_{phase}")


def res_path(base: str, tag: str, prof: str, rep: int) -> str:
    return os.path.join(base, "res", f"{tag}.{prof}.r{rep}.res")


def raw_path(base: str, tag: str, prof: str, rep: int) -> str:
    return os.path.join(base, "raw", f"{tag}.{prof}.r{rep}.raw")


def bundle_path(res: str) -> str:
    """crates/ny-cli/src/commands/proof_bundle.rs bundle_path: `<results>.proof.jsonl`."""
    return res + ".proof.jsonl"


def check_paths(out: str, tag: str) -> tuple[str, str]:
    d = os.path.join(cert_dir(out), "check")
    return os.path.join(d, f"{tag}.check.json"), os.path.join(d, f"{tag}.check.log")


def replay_record_path(out: str, tag: str) -> str:
    return os.path.join(cert_dir(out), "x86", tag, "replay.json")


def followup_tag(prefix: str, tag: str) -> str:
    if not tag.startswith(PHASE_A_PREFIX):
        raise CertError(f"{tag}: a certified S8 row tag must start {PHASE_A_PREFIX!r}")
    return prefix + tag[len(PHASE_A_PREFIX):]


def phase_a_tag(tag: str) -> str:
    for p in (REPROOF_PREFIX, CLASSIC_PREFIX):
        if tag.startswith(p):
            return PHASE_A_PREFIX + tag[len(p):]
    return tag


def row_shas(row: dict) -> tuple[str | None, str | None, str]:
    """(onnx sha256, vnnlib sha256, why-not): a certificate binds ONE model file and one property file."""
    files = L.model_files(row["onnx"])
    if len(files) != 1:
        return None, None, f"a multi-model row ({len(files)} models): the checker reads one ONNX model"
    o, v = sha256_file(files[0]), sha256_file(row["vnnlib"])
    if o is None or v is None:
        return o, v, "a row file is unreadable"
    return o, v, ""


# ---------------------------------------------------------------------------------------------------------
# the certificate: one `ny proof-check` report over one bundle, verified against the row's own files
# ---------------------------------------------------------------------------------------------------------


def bundle_header(bundle: str) -> dict | None:
    try:
        with open(bundle) as fh:
            first = fh.readline()
        d = json.loads(first)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("kind") == "header" else None


def verify_certificate(report: str, bundle: str, onnx_sha: str, vnnlib_sha: str, commit: str | None = None,
                       report_sha: str | None = None) -> dict:
    """A `ny-proof-check-report/v1` file judged as the certificate of ONE instance. Returns
    {state: accept|refuted|reject, why, verdict, report_sha256, bundle_sha256, nodes, checker_build}.

    ACCEPT needs ALL of: the report parses with the checker's schema; its onnx and vnnlib sha256 are the row's files;
    its bundle is THIS bundle (path and sha256 of the file on disk), and the bundle's own header names the same input
    hashes and the Proof profile; (with `commit`) the checker build AND the bundle's build name the pin's commit;
    (with `report_sha`) the file is byte-identical to the one the checker's DONE line hashed; verdict `verified` with
    exit code 0; and a non-empty cover whose every row is claimed AND confirmed. `refuted` with the report's onnx/vnnlib
    hashes the row's files is the checker's rigorous counterexample in the INSTANCE: the caller makes it an EMERGENCY,
    whatever the bundle/pin/DONE-line bindings say (review(certified-moat): an alarm is never downgraded by a
    binding). Everything else -- unconfirmed (an opaque or budget-cut claim), invalid, malformed, unsupported, a
    checker fault, any binding failure -- is REJECT: no certificate, never evidence for the unsat."""
    ev = {"state": "reject", "why": "", "verdict": None, "report_sha256": sha256_file(report),
          "bundle_sha256": sha256_file(bundle), "nodes": None, "checker_build": None}

    def rej(why: str) -> dict:
        ev["why"] = why
        return ev
    if ev["report_sha256"] is None:
        return rej("no report file")
    d = _json(report)
    if not isinstance(d, dict):
        return rej("the report does not parse")
    if d.get("schema") != REPORT_SCHEMA:
        return rej(f"report schema {d.get('schema')!r}")
    ev["verdict"] = d.get("verdict")
    ev["nodes"] = d.get("nodes") if isinstance(d.get("nodes"), dict) else None
    ev["checker_build"] = d.get("checker_build")
    if ev["verdict"] not in CHECKER_VERDICTS:
        return rej(f"unknown checker verdict {ev['verdict']!r}")
    if (d.get("onnx") or {}).get("sha256") != onnx_sha or (d.get("vnnlib") or {}).get("sha256") != vnnlib_sha:
        return rej("the report's onnx/vnnlib sha256 are not the row's files")
    if ev["verdict"] == "refuted":
        # review(certified-moat): a rigorous counterexample in THIS instance's input set (the report's onnx/vnnlib
        # are the row's files) makes every unsat stated on the instance false, whichever bundle the checker read and
        # whichever build wrote it -- so `refuted` is the EMERGENCY before any bundle/pin binding is asked, never a
        # quiet REJECT that sends the row to a classic sub-moat that cannot see a consistent false unsat.
        ev["state"] = "refuted"
        ev["why"] = "the checker found a rigorous counterexample in the input set"
        return ev
    if report_sha is not None and ev["report_sha256"] != report_sha:
        return rej("the report changed after the checker hashed it (DONE line report_sha256 differs)")
    b = d.get("bundle") or {}
    if ev["bundle_sha256"] is None:
        return rej("the bundle file is missing")
    if os.path.normpath(str(b.get("path"))) != os.path.normpath(bundle) or b.get("sha256") != ev["bundle_sha256"]:
        return rej("the report checked another bundle (path or sha256 differs from the bundle on disk)")
    if b.get("schema") != BUNDLE_SCHEMA:
        return rej(f"bundle schema {b.get('schema')!r}")
    h = bundle_header(bundle)
    if h is None:
        return rej("the bundle has no header line")
    if (h.get("onnx") or {}).get("sha256") != onnx_sha or (h.get("vnnlib") or {}).get("sha256") != vnnlib_sha:
        return rej("the bundle header's input hashes are not the row's files")
    if h.get("profile") != "proof":
        return rej(f"the bundle was written under profile {h.get('profile')!r}, not proof")
    if commit is not None:
        if f"commit={commit} " not in f"{d.get('checker_build')} " or f"commit={commit} " not in f"{h.get('build')} ":
            return rej(f"the checker or the bundle is not the pin's build (commit={commit[:12]})")
    if ev["verdict"] != "verified" or d.get("exit_code") != 0:
        return rej(f"checker verdict {ev['verdict']} (exit {d.get('exit_code')}): no certificate")
    cover = d.get("cover")
    if not isinstance(cover, list) or not cover:
        return rej("the report carries no cover rows")
    bad = [c for c in cover if not (isinstance(c, dict) and c.get("claimed_ok") is True and c.get("confirmed_ok") is True)]
    if bad:
        return rej(f"{len(bad)} cover row(s) not claimed AND confirmed")
    ev["state"] = "accept"
    ev["why"] = (f"verified: {len(cover)} cover row(s) confirmed"
                 + (f", nodes {ev['nodes'].get('confirmed')}/{ev['nodes'].get('total')}" if ev["nodes"] else ""))
    return ev


def check_certificate(log: dict, report: str, bundle: str, onnx_sha: str, vnnlib_sha: str, commit: str) -> dict:
    """The certificate of one re-proof check: the report the checker's DONE line names (and hashed), verified against
    the row and the pin (`verify_certificate`). review(certified-moat): an ACCEPT also needs the checker's OWN
    process to say so -- its DONE line `rc=0 verdict=verified` and the runner's `CERT_CHECK_RAN rc=0` -- not only a
    report file that says verified. A `refuted` report stays `refuted` whatever the log says."""
    if log.get("done_report") is None or os.path.normpath(log["done_report"]) != os.path.normpath(report):
        c = verify_certificate(report, bundle, onnx_sha, vnnlib_sha, commit)
        if c["state"] == "refuted":
            return c
        return {"state": "reject", "why": "the checker's DONE line names another report (or none)"}
    c = verify_certificate(report, bundle, onnx_sha, vnnlib_sha, commit, log.get("done_sha"))
    if c["state"] == "accept" and not (log.get("done_rc") == 0 and log.get("ran_rc") == 0
                                       and log.get("done_verdict") == "verified"):
        c = dict(c, state="reject", why=f"the checker's DONE line / exit (rc={log.get('done_rc')} "
                                        f"verdict={log.get('done_verdict')} exit={log.get('ran_rc')}) is not "
                                        "rc=0 verdict=verified")
    return c


def check_log_facts(log: str | None) -> dict:
    """The runner's check log: `CERT_CHECK_RAN rc=<r>` (the runner's own line: the check ran) and the checker's
    `PROOF_CHECK_DONE rc=<r> verdict=<v> ... report=<p> report_sha256=<s>` line."""
    if log is None:
        return {"ran": False}
    ran = CHECK_RAN_RE.findall(log)
    done = CHECK_DONE_RE.findall(log)
    out = {"ran": bool(ran), "ran_rc": int(ran[-1]) if ran else None}
    if done:
        rc, verdict, rep, sha = done[-1]
        out.update(done_rc=int(rc), done_verdict=verdict, done_report=rep, done_sha=sha)
    return out


# ---------------------------------------------------------------------------------------------------------
# the ledger: certificates from earlier certified S8s (an instance certified unsat stays unsat)
# ---------------------------------------------------------------------------------------------------------


def parse_ledger(path: str) -> list[dict]:
    text = _read(path)
    if text is None:
        raise CertError(f"ledger unreadable: {path}")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "# " + LEDGER_SCHEMA:
        raise CertError(f"{path}: not a {LEDGER_SCHEMA} ledger (first line)")
    if len(lines) < 2 or lines[1] != "\t".join(LEDGER_COLUMNS):
        raise CertError(f"{path}: ledger columns differ from {LEDGER_COLUMNS}")
    out = []
    for n, ln in enumerate(lines[2:], start=3):
        if not ln.strip() or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) != len(LEDGER_COLUMNS):
            raise CertError(f"{path}:{n}: {len(parts)} columns, want {len(LEDGER_COLUMNS)}")
        e = dict(zip(LEDGER_COLUMNS, parts))
        for k in ("onnx_sha256", "vnnlib_sha256", "report_sha256", "bundle_sha256"):
            if not re.match(r"^[0-9a-f]{64}$", e[k]):
                raise CertError(f"{path}:{n}: {k} is not a sha256")
        out.append(e)
    return out


def ledger_text(entries: list[dict]) -> str:
    rows = sorted({tuple(e[c] for c in LEDGER_COLUMNS) for e in entries})
    return "# " + LEDGER_SCHEMA + "\n" + "\t".join(LEDGER_COLUMNS) + "\n" + "".join("\t".join(r) + "\n" for r in rows)


def ledger_index(entries: list[dict]) -> dict[tuple[str, str], list[dict]]:
    idx: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for e in entries:
        idx[(e["onnx_sha256"], e["vnnlib_sha256"])].append(e)
    return idx


def ledger_certificate(idx: dict, row: dict) -> dict:
    """{ok, why, entry}: a ledger certificate for the row's instance that RE-VERIFIES now (the report and bundle on
    disk, byte-identical to the ledger's hashes, ACCEPT against the row's current files). The pin is not required:
    a checker ACCEPT is a fact about the instance, not about the binary that produced the bundle."""
    o, v, why = row_shas(row)
    if why:
        return {"ok": False, "why": why, "entry": None}
    hits = idx.get((o, v)) or []
    if not hits:
        return {"ok": False, "why": "no ledger certificate for this instance", "entry": None}
    last = "?"
    for e in hits:
        if sha256_file(e["report"]) != e["report_sha256"] or sha256_file(e["bundle"]) != e["bundle_sha256"]:
            last = f"ledger certificate {e['report']}: report or bundle changed since the ledger recorded it"
            continue
        c = verify_certificate(e["report"], e["bundle"], o, v)
        if c["state"] == "accept":
            # review(certified-moat): the ledger DECLARES which checker vouched (its checker_build column) and the
            # report must agree: a report whose checker names no commit, or another commit than the ledger
            # declared, is no certificate -- and the per-row evidence names that build, so the S8 record says which
            # checker each ledger-covered row rests on
            m = re.search(r"(?:^|\s)commit=([0-9a-f]{7,64})(?:\s|$)", str(c.get("checker_build") or ""))
            if m is None or f"commit={m.group(1)}" not in str(e.get("checker_build") or "").split():
                last = (f"ledger certificate {e['report']}: the report's checker build "
                        f"({'commit=' + m.group(1)[:12] if m else 'no commit'}) is not the one the ledger declares")
                continue
            return {"ok": True, "why": f"ledger certificate {e['report']} ({c['why']})",
                    "entry": {"report": e["report"], "report_sha256": e["report_sha256"], "source": e["source"],
                              "checker_commit": m.group(1)}}
        last = f"ledger certificate {e['report']}: {c['why']}"
    return {"ok": False, "why": last, "entry": None}


# ---------------------------------------------------------------------------------------------------------
# follow-up rows: a pure function of the phase-A rows + judgements (+ ledger, + the re-proof outcomes)
# ---------------------------------------------------------------------------------------------------------


def classify(row: dict, j, ledger_ok: bool) -> str:
    """The phase-A class of one S8 row from its (single) competition judgement."""
    if j is None or j.status == "VOID":
        return C_NOTOK
    if j.status == "EMERGENCY":
        return C_EMERGENCY
    if j.status != "OK":
        return C_NOTOK
    v = j.verdict
    if v == "unsat":
        return C_UNSAT_LEDGER if (row["bank"] == "unsat" and ledger_ok) else C_UNSAT_REPROVE
    if v == "sat":
        return C_SAT
    if row["bank"] in L.DECIDED:
        return C_LOST
    return C_UNDECIDED


def _followup(row: dict, prefix: str, reps_c: int, reps_p: int, group: str, note: str) -> dict:
    r = dict(row)
    alarms = L.split_terms(row["alarm"], L.ALARM_SEP)
    if FLIP_ALARM not in alarms:
        alarms.append(FLIP_ALARM)
    r.update(tag=followup_tag(prefix, row["tag"]), reps_c=str(reps_c), reps_p=str(reps_p),
             alarm=L.ALARM_SEP.join(alarms), group=group, note=note)
    return r


def reproof_row(row: dict) -> dict:
    return _followup(row, REPROOF_PREFIX, 0, 1, "seal/S8C/reproof",
                     f"certified re-proof of {row['tag']}: armed competition unsat, one Proof run + ny proof-check")


def classic_row(row: dict) -> dict:
    return _followup(row, CLASSIC_PREFIX, CLASSIC_REPS[0], CLASSIC_REPS[1], "seal/S8C/classic",
                     f"classic sub-moat of {row['tag']}: the re-proof did not ACCEPT")


def derive_reproof(rows: list[dict], first: dict, ledger: dict) -> list[dict]:
    """`first[tag]` = the phase-A competition r1 judgement; `ledger[tag]` = ledger_certificate(...)."""
    return [reproof_row(r) for r in rows
            if classify(r, first.get(r["tag"]), (ledger.get(r["tag"]) or {}).get("ok", False)) == C_UNSAT_REPROVE]


def derive_classic(reproof: list[dict], outcomes: dict) -> list[dict]:
    """Every re-proof row whose outcome is FALLBACK. Refuses while any outcome is still pending (the classic set is
    only defined once every re-proof is final)."""
    pend = [r["tag"] for r in reproof if (outcomes.get(r["tag"]) or {}).get("state") == R_PENDING
            or r["tag"] not in outcomes]
    if pend:
        raise CertError(f"{len(pend)} re-proof(s) still pending ({pend[:3]}): the classic set is not defined yet")
    return [classic_row(_phase_row(r)) for r in reproof if outcomes[r["tag"]]["state"] == R_FALLBACK]


def _phase_row(followup: dict) -> dict:
    """The phase-A row a follow-up row was derived from (its alarm without the flip alarm, its own group/note)."""
    r = dict(followup)
    alarms = [a for a in L.split_terms(followup["alarm"], L.ALARM_SEP) if a != FLIP_ALARM]
    r.update(tag=phase_a_tag(followup["tag"]), reps_c="1", reps_p="0",
             alarm=L.ALARM_SEP.join(alarms) if alarms else "-")
    return r


def spec_rows_equal(a: list[dict], b: list[dict]) -> bool:
    return [tuple(r[c] for c in L.SPEC_COLUMNS) for r in a] == [tuple(r[c] for c in L.SPEC_COLUMNS) for r in b]


# ---------------------------------------------------------------------------------------------------------
# outcomes: a re-proof (Proof verdict + the check's facts) and a replay (its record)
# ---------------------------------------------------------------------------------------------------------


def reproof_outcome(j, facts: dict) -> dict:
    """The state of one re-proof from the Proof run's judgement and the gathered artefact facts
    (bundle_exists, log facts, certificate)."""
    if j is None or j.status == "VOID":
        return {"state": R_PENDING, "why": "the Proof re-run is not OK or did not run (re-run it)"}
    if j.status == "EMERGENCY":
        return {"state": R_EMERGENCY, "why": f"the Proof re-run is an EMERGENCY: {';'.join(j.reasons)[:200]}"}
    if j.verdict == "sat":
        return {"state": R_EMERGENCY, "why": "the Proof profile states sat on a row the armed Competition run called unsat"}
    if j.verdict != "unsat":
        return {"state": R_FALLBACK, "why": f"profile divergence: the Proof profile did not reproduce the unsat "
                                            f"({j.verdict})"}
    if "bundle_exists" not in facts:
        return {"state": R_PENDING, "why": "no artefact facts gathered for this re-proof"}
    if not facts.get("bundle_exists"):
        return {"state": R_FALLBACK, "why": "the Proof run published no bundle (the route streams no checkable proof)"}
    log = facts.get("log") or {}
    if not log.get("ran"):
        return {"state": R_PENDING, "why": "ny proof-check has not run on the bundle"}
    cert = facts.get("cert") or {}
    # review(certified-moat): `refuted` -- in the report (bound to the row's files) OR on the checker's own DONE line /
    # exit code -- is the EMERGENCY before anything else is asked: a missing DONE line or a binding failure must
    # never turn a counterexample into a classic-fallback row
    if cert.get("state") == "refuted":
        return {"state": R_EMERGENCY, "why": f"ny proof-check REFUTED the armed unsat: {cert.get('why')}", "cert": cert}
    if log.get("done_verdict") == "refuted" or log.get("done_rc") == 21 or log.get("ran_rc") == 21:
        return {"state": R_EMERGENCY, "why": "ny proof-check REFUTED the armed unsat (its DONE line / exit code says "
                                             f"refuted: rc={log.get('done_rc')}/{log.get('ran_rc')})", "cert": cert}
    if log.get("done_rc") is None:
        return {"state": R_FALLBACK, "why": f"the checker printed no PROOF_CHECK_DONE line (rc={log.get('ran_rc')}): "
                                            "no certificate", "cert": cert}
    if cert.get("state") == "accept":
        return {"state": R_ACCEPT, "why": cert.get("why"), "cert": cert}
    return {"state": R_FALLBACK, "why": f"no certificate: {cert.get('why')}", "cert": cert}


def verify_replay_record(rec_path: str, row: dict, witness_res: str, rig: dict) -> dict:
    """One x86 replay record (x86_replay.py) judged against the row, the phase-A results file it replays and the
    plan's rig. {state: passed|failed|inconclusive|pending, why, result, record_sha256}."""
    out = {"state": X_PENDING, "why": "no replay record", "result": None, "record_sha256": sha256_file(rec_path)}
    if out["record_sha256"] is None:
        return out
    d = _json(rec_path)

    def inc(why: str) -> dict:
        out.update(state=X_INCONCLUSIVE, why=why)
        return out
    if not isinstance(d, dict) or d.get("schema") != REPLAY_SCHEMA:
        return inc("the replay record does not parse")
    out["result"] = d.get("result")
    if d.get("tag") != row["tag"] or d.get("onnx") != row["onnx"] or d.get("vnnlib") != row["vnnlib"]:
        return inc("the replay record names another row")
    if d.get("witness_res_sha256") != sha256_file(witness_res) or os.path.normpath(str(d.get("witness_res"))) != \
            os.path.normpath(witness_res):
        return inc("the replayed witness is not the phase-A results file (path or sha256)")
    ce, res_text = d.get("ce"), _read(witness_res) or ""
    if sha256_file(str(ce)) != d.get("ce_sha256") or _read(str(ce)) != witness_text(res_text):
        return inc("the counterexample file is not the results file's witness")
    if d.get("checker") != checker_kind(row):
        return inc(f"replayed with checker {d.get('checker')!r}, the row needs {checker_kind(row)!r}")
    if (d.get("rig") or {}).get("files_sha256") != rig_files_digest(rig):
        return inc("the replay ran on another rig than the plan's (rig files digest differs)")
    ck = d.get("check") or {}
    if sha256_file(str(ck.get("record"))) != ck.get("record_sha256"):
        return inc("the checker's own result record is missing or changed")
    # review(certified-moat): the checker's record must be THIS replay's -- beside replay.json, and naming this
    # witness (v1: its sha256, model, property and the year's scoring code; v2: this replay's job id) -- or a record
    # of another row, or of the batch's PASSING control, could stand in for it
    if os.path.dirname(os.path.abspath(str(ck.get("record")))) != os.path.dirname(os.path.abspath(rec_path)):
        return inc("the checker's result record is not this replay's own (it lives outside the replay directory)")
    raw = _json(str(ck.get("record")))
    if not isinstance(raw, dict) or raw.get("result") != d.get("result"):
        return inc("the replay's result is not the checker record's result")
    rid = str(d.get("id") or "")
    if checker_kind(row).startswith("v2") and not (rid.startswith("seal:") and rid.endswith(":" + row["tag"])):
        return inc(f"the replay's job id {rid!r} is not this row's")
    why = raw_binding_problem(raw, checker_kind(row), d.get("ce_sha256"), L.model_files(row["onnx"])[0],
                              row["vnnlib"], rid, rig)
    if why:
        return inc(f"the checker's result record is not this witness's: {why}")
    why = platform_problem(raw, rig)
    if why:
        return inc(why)
    why = controls_problem(d.get("controls") or {}, rig, checker_kind(row))
    if why:
        return inc(f"the batch's witness controls did not discriminate for {checker_kind(row)} (the rig must PASS a "
                   f"good witness and FAIL a bad one of the row's own checker kind, in this batch's unedited controls "
                   f"file): {why}")
    res = d.get("result")
    if res in WITNESS_FAIL:
        out.update(state=X_FAIL, why=f"the organisers' checker says {res} on the certified x86 platform")
        return out
    if res not in WITNESS_PASS:
        return inc(f"checker result {res!r} (rig error)")
    if (d.get("ort") or {}).get("state") != "match":
        return inc(f"the ORT replay of the same feeds does not match the checker's outputs ({(d.get('ort') or {}).get('cmp')})")
    out.update(state=X_PASS, why=f"{res} on {X86_MACHINE}/{rig.get('cpu')} ORT {rig.get('onnxruntime')}; ORT replay match")
    return out


def rig_files_digest(rig: dict) -> str:
    """The digest x86_replay.replay_row stamps into a record: sha256 of the rig's pre-registered {file: sha256}."""
    import hashlib
    return hashlib.sha256(json.dumps(rig.get("files") or {}, sort_keys=True).encode()).hexdigest()


def platform_problem(raw: dict, rig: dict) -> str:
    env = raw.get("env") or {}
    if (env.get("machine") == X86_MACHINE and env.get("qemu_cpu") == rig.get("cpu")
            and str(env.get("onnxruntime")) == str(rig.get("onnxruntime"))):
        return ""
    return (f"not the certified platform (machine={env.get('machine')} cpu={env.get('qemu_cpu')} "
            f"ort={env.get('onnxruntime')}; want {X86_MACHINE}/{rig.get('cpu')}/{rig.get('onnxruntime')})")


def _same_path(a, b) -> bool:
    return a is not None and b is not None and os.path.realpath(str(a)) == os.path.realpath(str(b))


def raw_binding_problem(raw: dict, kind: str, ce_sha: str | None, onnx: str, vnnlib: str, rid: str, rig: dict) -> str:
    """'' when the organisers'-checker record `raw` is the check of THIS witness on THIS instance: v1 (checkv1.py)
    names the witness sha256, the model, the property and the year's scoring directory; v2 (driver.py --mode check,
    replay.py check-one) names the job id, the x86 platform, and no injected outputs. Else why not."""
    if kind in ("v1-2025", "v1-2026"):
        scoring = (rig.get("scoring") or {}).get("2025" if kind == "v1-2025" else "2026")
        if raw.get("mode") != "checkv1":
            return f"mode {raw.get('mode')!r} is not checkv1"
        if not ce_sha or raw.get("ce_sha256") != ce_sha:
            return "it checked another witness (ce_sha256)"
        if not _same_path(raw.get("onnx"), onnx) or not _same_path(raw.get("vnnlib"), vnnlib):
            return "it checked another model or property"
        if not _same_path(raw.get("scoring"), scoring):
            return f"it ran another scoring directory ({raw.get('scoring')}, want {scoring})"
        return ""
    if kind == "v2-2026":
        if raw.get("mode") != "check" or raw.get("error") or raw.get("injected_from"):
            return f"not an official v2 check (mode={raw.get('mode')!r} error={raw.get('error')!r} " \
                   f"injected={raw.get('injected_from')!r})"
        if not rid or raw.get("id") != rid:
            return f"it is job {raw.get('id')!r}, not {rid!r}"
        if raw.get("platform") != f"x86:{rig.get('cpu')}":
            return f"platform {raw.get('platform')!r}"
        return ""
    return f"unknown checker kind {kind!r}"


def controls_problem(ctl: dict, rig: dict, kind: str) -> str:
    """'' when the batch's controls file (`ctl` = {path, sha256} from a replay record) RE-DERIVES as discriminating
    for checker `kind`: byte-identical, exactly the plan's controls (name, kind, expect, witness sha256), each
    control's own checker record bound to its witness/instance on the certified platform with the result it reports,
    each `got` re-derived from that record and equal to its `expect` -- and the row's kind has at least one PASS
    and one FAIL control (review(certified-moat): a v1 control proves nothing about the v2 code path)."""
    path = str(ctl.get("path"))
    cdoc = _json(path)
    if sha256_file(path) != ctl.get("sha256") or not isinstance(cdoc, dict) or cdoc.get("schema") != CONTROLS_SCHEMA:
        return "the controls file is missing, changed, or not a controls file"
    if cdoc.get("ok") is not True or cdoc.get("cpu") != rig.get("cpu"):
        return f"the controls file says ok={cdoc.get('ok')} cpu={cdoc.get('cpu')}"
    plan_ctl = {c.get("name"): c for c in rig.get("controls") or []}
    got = cdoc.get("controls") or []
    if not isinstance(got, list) or sorted(str(c.get("name")) for c in got) != sorted(str(n) for n in plan_ctl):
        return "the controls file does not run exactly the plan's controls"
    for c in got:
        p = plan_ctl[c.get("name")]
        if (c.get("kind"), c.get("expect"), c.get("ce_sha256")) != (p.get("kind"), p.get("expect"), p.get("ce_sha256")):
            return f"control {c.get('name')}: not the plan's control (kind / expect / witness sha256)"
        rec = str(c.get("record"))
        if sha256_file(rec) != c.get("record_sha256"):
            return f"control {c.get('name')}: its checker record is missing or changed"
        raw = _json(rec)
        if not isinstance(raw, dict) or raw.get("result") != c.get("result"):
            return f"control {c.get('name')}: its checker record's result is not the one the file reports"
        why = raw_binding_problem(raw, p["kind"], p.get("ce_sha256"), p.get("onnx"), p.get("vnnlib"),
                                  str(c.get("id") or ""), rig) or platform_problem(raw, rig)
        if why:
            return f"control {c.get('name')}: {why}"
        derived = ("pass" if raw.get("result") in WITNESS_PASS and c.get("ort") == "match"
                   else "fail" if raw.get("result") in WITNESS_FAIL else "error")
        if derived != c.get("got") or derived != p.get("expect"):
            return (f"control {c.get('name')}: re-derived {derived}, the file says {c.get('got')}, "
                    f"want {p.get('expect')}")
    have = {c.get("expect") for c in got if c.get("kind") == kind}
    if have != {"pass", "fail"}:
        return f"no PASS and FAIL control of checker kind {kind} (the plan's rig has {sorted(have) or 'none'})"
    return ""


def control_kinds(rig: dict) -> set[str]:
    """The checker kinds the rig's controls can discriminate (a PASS and a FAIL control of that kind)."""
    by = collections.defaultdict(set)
    for c in rig.get("controls") or []:
        by[c.get("kind")].add(c.get("expect"))
    return {k for k, v in by.items() if v >= {"pass", "fail"}}


def witness_text(res_text: str) -> str:
    """The counterexample a results file carries: the file minus its first `sat` line (scratchpad/x86parity's rule,
    validated with the organisers' code on positive and negative controls)."""
    lines = res_text.splitlines(keepends=True)
    if lines and lines[0].strip() == "sat":
        return "".join(lines[1:])
    return res_text


def checker_kind(row: dict) -> str:
    """Which organisers' checker scores the row: 2025 -> the v1 path (SCORING-ZERO-TOL); 2026 -> v2 when the row's
    benchmark version directory is `2.0`, else v1 (scratchpad/x86parity select_rows_v1.py)."""
    if int(row["year"]) == 2025:
        return "v1-2025"
    parts = os.path.normpath(L.model_files(row["onnx"])[0]).split(os.sep)
    ver = parts[parts.index("onnx") - 1] if "onnx" in parts and parts.index("onnx") > 0 else ""
    return "v2-2026" if ver == "2.0" else "v1-2026"


# ---------------------------------------------------------------------------------------------------------
# gather (artefacts -> facts) and decide (facts -> verdict)
# ---------------------------------------------------------------------------------------------------------


def read_spec_or_none(path: str) -> list[dict] | None:
    return L.read_spec(path) if os.path.isfile(path) else None


def followup_rows(out: str) -> list[dict]:
    rows = []
    for ph in ("reproof", "classic"):
        rs = read_spec_or_none(spec_path(out, ph))
        rows += rs or []
    return rows


def followup_judged(out: str) -> dict[str, str]:
    return {ph: os.path.join(run_dir(out, ph), "judged.tsv") for ph in ("reproof", "classic")}


def followup_runs(out: str) -> dict:
    """{(tag, profile, rep): {raw, raw_path, flight}} for every follow-up run (the stage judge's `runs`)."""
    runs = {}
    for ph in ("reproof", "classic"):
        rows = read_spec_or_none(spec_path(out, ph)) or []
        base = run_dir(out, ph)
        for r in rows:
            for prof, k in M0J.schedule(r):
                raw = raw_path(base, r["tag"], prof, k)
                res = res_path(base, r["tag"], prof, k)
                runs[(r["tag"], prof, k)] = {"raw": _read(raw), "raw_path": raw, "flight": _json(res + ".flight.json")}
    return runs


def load_ledger_of(out: str, plan: dict) -> list[dict]:
    led = (plan.get("certified") or {}).get("ledger")
    if not led:
        return []
    return parse_ledger(os.path.join(out, led["path"]))


def gather(out: str, plan: dict, rows: list[dict], phase_res: dict[str, str], followup_lines: list) -> dict:
    """Every artefact fact the certified decision reads, as plain data: the ledger certificates of the phase-A rows,
    the on-disk follow-up specs, each re-proof's bundle / check log / certificate, and each phase-A row's replay
    record (verified against the rig the plan pre-registered). `phase_res[tag]` = the phase-A competition r1
    results file."""
    cplan = plan.get("certified") or {}
    try:
        idx = ledger_index(load_ledger_of(out, plan))
        ledger_err = None
    except CertError as e:
        idx, ledger_err = {}, str(e)
    ledger = {r["tag"]: ledger_certificate(idx, r) for r in rows} if idx else {}
    reproof = read_spec_or_none(spec_path(out, "reproof"))
    classic = read_spec_or_none(spec_path(out, "classic"))
    commit = plan["pin"]["commit"]
    checks = {}
    for r in reproof or []:
        res = res_path(run_dir(out, "reproof"), r["tag"], "proof", 1)
        bnd = bundle_path(res)
        rep, logp = check_paths(out, r["tag"])
        log = check_log_facts(_read(logp))
        f = {"bundle": bnd, "bundle_exists": os.path.isfile(bnd), "report": rep, "log_path": logp, "log": log}
        o, v, why = row_shas(r)
        if why:
            f["cert"] = {"state": "reject", "why": why}
        elif log.get("ran"):
            f["cert"] = check_certificate(log, rep, bnd, o, v, commit)
        checks[r["tag"]] = f
    rig = cplan.get("x86") or {}
    replays = {}
    for r in rows:
        if r["tag"] in phase_res:
            replays[r["tag"]] = verify_replay_record(replay_record_path(out, r["tag"]), r, phase_res[r["tag"]], rig)
    return {"ledger": ledger, "ledger_error": ledger_err, "reproof_spec": reproof, "classic_spec": classic,
            "checks": checks, "replays": replays}


def decide(plan: dict, rows: list[dict], lines: list, facts: dict) -> dict:
    """The certified S8 decision. `rows` = the phase-A S8 rows; `lines` = every judgement of phase A AND the
    follow-ups (collapsed per run, EMERGENCY sticky); `facts` = gather(...). Returns {status_inputs, certified}:
    `emergencies`, `void` (why the record is incomplete), `lost`, and the per-row evidence the record carries."""
    by_key = L.collapse_runs(lines)
    first = {r["tag"]: by_key.get((r["tag"], "competition", 1)) for r in rows}
    ledger = facts.get("ledger") or {}
    emerg: list[str] = []
    void: list[str] = []
    per: dict[str, dict] = {}
    counts = collections.Counter()
    if facts.get("ledger_error"):
        void.append(f"the plan's ledger does not parse: {facts['ledger_error']}")
    for r in rows:
        j = first.get(r["tag"])
        led = ledger.get(r["tag"]) or {}
        c = classify(r, j, led.get("ok", False))
        counts[c] += 1
        per[r["tag"]] = {"class": c, "verdict": j.verdict if j else None, "bank": r["bank"], "truth": r["truth"]}
        if c == C_UNSAT_LEDGER:
            per[r["tag"]]["ledger"] = led.get("entry")
        if c == C_NOTOK:
            void.append(f"{r['tag']}: phase-A run not OK or not run")
    # the re-proof set must be EXACTLY the derivation from phase A (+ ledger)
    want_reproof = derive_reproof(rows, first, ledger)
    have_reproof = facts.get("reproof_spec")
    outcomes: dict[str, dict] = {}
    if want_reproof and have_reproof is None:
        void.append(f"certification pending: {len(want_reproof)} armed-unsat row(s) need a re-proof "
                    "(run `seal.py run --stage S8C`)")
    elif have_reproof is not None and not spec_rows_equal(have_reproof, want_reproof):
        void.append("the re-proof spec on disk does not re-derive from phase A (edited, or phase A changed under it)")
    elif have_reproof is not None:
        for rr in have_reproof:
            jp = by_key.get((rr["tag"], "proof", 1))
            o = reproof_outcome(jp, (facts.get("checks") or {}).get(rr["tag"]) or {})
            outcomes[rr["tag"]] = o
            pa = phase_a_tag(rr["tag"])
            ent = {"tag": rr["tag"], "state": o["state"], "why": o["why"],
                   "proof_verdict": jp.verdict if jp else None}
            cf = (facts.get("checks") or {}).get(rr["tag"]) or {}
            cert = o.get("cert") or cf.get("cert") or {}
            if cert:
                ent["check"] = {"report": cf.get("report"), "report_sha256": cert.get("report_sha256"),
                                "bundle": cf.get("bundle"), "bundle_sha256": cert.get("bundle_sha256"),
                                "verdict": cert.get("verdict"), "certificate": cert.get("state"),
                                "nodes": cert.get("nodes")}
            per[pa]["reproof"] = ent
            counts[f"reproof_{o['state']}"] += 1
            # review(certified-moat): a Proof re-run that does not reproduce a BANKED verdict is a lost run, exactly
            # as the classic S8 counts its Proof rep (any lost run blocks PASS at tolerance 0) -- the fallback it also
            # triggers must not make the certified mode more lenient than the classic one
            if jp is not None and jp.status == "OK" and jp.outcome == "lost":
                counts["reproof_lost_runs"] += 1
                ent["lost_run"] = True
            if o["state"] == R_EMERGENCY:
                emerg.append(f"{rr['tag']}: {o['why']}")
            elif o["state"] == R_PENDING:
                void.append(f"{rr['tag']}: {o['why']}")
    # the classic set must be EXACTLY the re-proofs that fell back (defined only once every re-proof is final)
    have_classic = facts.get("classic_spec")
    reproof_ok = have_reproof is not None and spec_rows_equal(have_reproof, want_reproof)
    if have_classic is not None and not reproof_ok:
        void.append("a classic spec exists without a re-proof set that re-derives")
    elif reproof_ok:
        final = all(o["state"] in (R_ACCEPT, R_FALLBACK, R_EMERGENCY) for o in outcomes.values())
        want_classic = ([classic_row(_phase_row(rr)) for rr in have_reproof
                         if outcomes[rr["tag"]]["state"] == R_FALLBACK] if final else None)
        if want_classic is None:
            if have_classic is not None:
                void.append("a classic spec exists while re-proofs are still pending")
        elif want_classic and have_classic is None:
            void.append(f"certification pending: {len(want_classic)} re-proof(s) fell back to the classic sub-moat "
                        "(run `seal.py run --stage S8C` again)")
        elif have_classic is not None and not spec_rows_equal(have_classic, want_classic):
            void.append("the classic spec on disk does not re-derive from the re-proof outcomes")
        elif have_classic is not None:
            for cr in have_classic:
                runs = [by_key.get((cr["tag"], p, k)) for p, k in M0J.schedule(cr)]
                pa = phase_a_tag(cr["tag"])
                if any(x is not None and x.status == "EMERGENCY" for x in runs):
                    state = "emergency"     # the stage judge's per-run / profile rules carry the reason
                elif any(x is None or x.status != "OK" for x in runs):
                    void.append(f"{cr['tag']}: classic sub-moat runs not OK or not run")
                    state = "pending"
                else:
                    state = "covered"
                lost_runs = sum(1 for x in runs if x is not None and x.status == "OK" and x.outcome == "lost")
                if lost_runs:
                    counts["classic_lost_runs"] += lost_runs
                per[pa]["classic"] = {"tag": cr["tag"], "state": state, "lost_runs": lost_runs,
                                      "verdicts": [f"{p}/r{k}={x.verdict if x else '-'}"
                                                   for (p, k), x in zip(M0J.schedule(cr), runs)]}
                counts[f"classic_{state}"] += 1
    # replays: every armed-sat row
    for r in rows:
        if per[r["tag"]]["class"] != C_SAT:
            continue
        x = (facts.get("replays") or {}).get(r["tag"]) or {"state": X_PENDING, "why": "no replay record"}
        per[r["tag"]]["replay"] = {"state": x["state"], "why": x["why"], "result": x.get("result"),
                                   "record_sha256": x.get("record_sha256")}
        counts[f"replay_{x['state']}"] += 1
        if x["state"] == X_FAIL:
            emerg.append(f"{r['tag']}: WRONG SAT: {x['why']}")
        elif x["state"] in (X_PENDING, X_INCONCLUSIVE):
            void.append(f"{r['tag']}: x86 replay {x['state']}: {x['why']}")
    lost = sorted(t for t, p in per.items() if p["class"] == C_LOST)
    accepted = sorted(t for t, p in per.items() if (p.get("reproof") or {}).get("state") == R_ACCEPT)
    covered = sorted(t for t, p in per.items() if (p.get("classic") or {}).get("state") == "covered")
    replayed = sorted(t for t, p in per.items() if (p.get("replay") or {}).get("state") == X_PASS)
    return {"emergencies": emerg, "void": void, "lost": lost, "lost_runs": counts.get("classic_lost_runs", 0) + counts.get("reproof_lost_runs", 0),
            "certified": {"schema": CERT_SCHEMA, "lost_tolerance": LOST_TOLERANCE, "counts": dict(sorted(counts.items())),
                          "accepted": accepted, "classic_covered": covered, "replayed": replayed, "lost": lost,
                          "ledger_covered": sorted(t for t, p in per.items() if p["class"] == C_UNSAT_LEDGER),
                          "rows": per}}


# ---------------------------------------------------------------------------------------------------------
# the plan's certified block
# ---------------------------------------------------------------------------------------------------------

# The x86 replay rig as measured 2026-09-26/28 (scratchpad/x86replay/README.md, scratchpad/x86parity/): qemu-user
# 10.0 `-cpu Haswell-v4` + the noble amd64 sysroot + python-build-standalone 3.12.13 + the organisers' pinned
# requirements reproduce the organisers' graded witnesses 100/100 (smart_turn) and 67/67 (ACC).
DEFAULT_RIG = {
    "xr_dir": "<home>/ny/scratchpad/x86replay",
    "driver": "<home>/ny/scratchpad/x86replay/driver.py",
    "replay": "<home>/ny/scratchpad/x86replay/replay.py",
    "checkv1": "<home>/ny/scratchpad/x86parity/checkv1.py",
    "cmp": "<home>/ny/scratchpad/x86parity/cmp_npz.py",
    "qemu": "<home>/x86emu/qemu10u/usr/bin/qemu-x86_64",
    "sysroot": "<home>/x86emu/sysroot",
    "x86py": "<home>/x86emu/py312x86/python/bin/python3.12",
    "armpy": "<home>/x86emu/venv-arm/bin/python",
    "work": "<home>/x86emu/work",
    "cpu": "Haswell-v4",
    "onnxruntime": "1.26.0",
    "scoring": {"2025": "<home>/vnncomp2025_results/SCORING-ZERO-TOL",
                "2026": "<home>/vnncomp2026_results/SCORING"},
    "timeout_s": 1500,
    "controls": [
        {"name": "pos2025", "kind": "v1-2025", "expect": "pass",
         "onnx": "<home>/ny/benchmarks/vnncomp2025/benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_2_batch_2000.onnx",
         "vnnlib": "<home>/ny/benchmarks/vnncomp2025/benchmarks/acasxu_2023/vnnlib/prop_2.vnnlib",
         "ce": "<home>/ny/scratchpad/x86parity/controls/pos2025_acas12p2.counterexample"},
        {"name": "neg2025", "kind": "v1-2025", "expect": "fail",
         "onnx": "<home>/ny/benchmarks/vnncomp2025/benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_2_batch_2000.onnx",
         "vnnlib": "<home>/ny/benchmarks/vnncomp2025/benchmarks/acasxu_2023/vnnlib/prop_2.vnnlib",
         "ce": "<home>/ny/scratchpad/x86parity/controls/neg2025_acas12p2.counterexample"},
        {"name": "pos2026", "kind": "v1-2026", "expect": "pass",
         "onnx": "<home>/ny/benchmarks/vnncomp2026/benchmarks/acasxu_2023/1.0/onnx/ACASXU_run2a_1_2_batch_2000.onnx",
         "vnnlib": "<home>/ny/benchmarks/vnncomp2026/benchmarks/acasxu_2023/1.0/vnnlib/prop_2.vnnlib",
         "ce": "<home>/ny/scratchpad/x86parity/controls/pos2026_acas12p2.counterexample"},
        {"name": "neg2026", "kind": "v1-2026", "expect": "fail",
         "onnx": "<home>/ny/benchmarks/vnncomp2026/benchmarks/acasxu_2023/1.0/onnx/ACASXU_run2a_1_2_batch_2000.onnx",
         "vnnlib": "<home>/ny/benchmarks/vnncomp2026/benchmarks/acasxu_2023/1.0/vnnlib/prop_2.vnnlib",
         "ce": "<home>/ny/scratchpad/x86parity/controls/neg2026_acas12p2.counterexample"},
        # review(certified-moat): the v2-2026 path (driver.py --mode check: the official counterexamples_v2), which
        # scores the 13 VNN-LIB 2.0 categories of 2026 (cifar100_2024 among them), needs its OWN discriminating pair;
        # measured on this rig 2026-10-08 (see scripts/seal/README.md, "Review")
        {"name": "pos2026v2", "kind": "v2-2026", "expect": "pass",
         "onnx": "<home>/ny/benchmarks/vnncomp2026/benchmarks/acasxu_2023/2.0/onnx/ACASXU_run2a_1_2_batch_2000.onnx",
         "vnnlib": "<home>/ny/benchmarks/vnncomp2026/benchmarks/acasxu_2023/2.0/vnnlib/prop_2.vnnlib",
         "ce": os.path.join(HERE, "x86_controls", "pos2026v2_acas12p2.counterexample")},
        {"name": "neg2026v2", "kind": "v2-2026", "expect": "fail",
         "onnx": "<home>/ny/benchmarks/vnncomp2026/benchmarks/acasxu_2023/2.0/onnx/ACASXU_run2a_1_2_batch_2000.onnx",
         "vnnlib": "<home>/ny/benchmarks/vnncomp2026/benchmarks/acasxu_2023/2.0/vnnlib/prop_2.vnnlib",
         "ce": os.path.join(HERE, "x86_controls", "neg2026v2_acas12p2.counterexample")},
    ],
}
RIG_FILE_KEYS = ("driver", "replay", "checkv1", "cmp", "qemu", "x86py", "armpy")
RIG_DIR_KEYS = ("xr_dir", "sysroot", "work")


def rig_files(rig: dict) -> dict[str, str]:
    """{path: sha256} of every rig file the replay executes (and the organisers' checker modules)."""
    files = [rig[k] for k in RIG_FILE_KEYS]
    for y, d in sorted((rig.get("scoring") or {}).items()):
        for name in ("counterexamples.py", "counterexamples_v2.py", "vnnlib.py", "vnnlib_v1.py", "settings.py"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                files.append(p)
    out = {}
    for f in files:
        s = sha256_file(f)
        if s is None:
            raise CertError(f"x86 rig file missing: {f}")
        out[f] = s
    return out


def rig_problems(rig: dict) -> list[str]:
    """Step 0's rig check: every pre-registered file byte-identical, every directory present, every control CE."""
    bad = []
    for f, sha in (rig.get("files") or {}).items():
        if sha256_file(f) != sha:
            bad.append(f"rig file changed or missing: {f}")
    for k in RIG_DIR_KEYS:
        if not os.path.isdir(str(rig.get(k))):
            bad.append(f"rig directory missing: {k}={rig.get(k)}")
    if not rig.get("controls"):
        bad.append("the rig has no witness controls (a replay that cannot be shown to FAIL proves nothing)")
    exp = {c.get("expect") for c in rig.get("controls") or []}
    if rig.get("controls") and exp != {"pass", "fail"}:
        bad.append("the controls need at least one expected PASS and one expected FAIL witness")
    for c in rig.get("controls") or []:
        for k in ("onnx", "vnnlib"):
            if not os.path.isfile(str(c.get(k))):
                bad.append(f"control {c.get('name')}: {k} missing: {c.get(k)}")
        if sha256_file(str(c.get("ce"))) != c.get("ce_sha256"):
            bad.append(f"control {c.get('name')}: counterexample changed or missing: {c.get('ce')}")
    return bad


def plan_block(out: str, ledger: str | None, rig: dict | None, checker_budget_s: float, write_new) -> tuple[dict, dict]:
    """The plan's `certified` block and the {relpath: sha256} files it pre-registers: the ledger copy (re-verified
    at every judge), the x86 rig (every file's sha256) with its control witnesses COPIED into the plan dir, and the
    checker settings."""
    files: dict[str, str] = {}
    led = None
    if ledger:
        entries = parse_ledger(ledger)
        rel = os.path.join("certified", "ledger.tsv")
        write_new(os.path.join(out, rel), ledger_text(entries))
        files[rel] = L.sha256_file(os.path.join(out, rel))
        led = {"path": rel, "source": os.path.abspath(ledger), "sha256": files[rel], "entries": len(entries)}
    r = json.loads(json.dumps(rig if rig is not None else DEFAULT_RIG))
    r["files"] = rig_files(r)
    ctl = []
    for c in r.get("controls") or []:
        src = c["ce"]
        if not os.path.isfile(src):
            raise CertError(f"control {c.get('name')}: counterexample missing: {src}")
        rel = os.path.join("certified", "controls", f"{c['name']}.counterexample")
        write_new(os.path.join(out, rel), open(src).read())
        files[rel] = L.sha256_file(os.path.join(out, rel))
        ctl.append(dict(c, ce=os.path.join(out, rel), ce_sha256=files[rel], source=src))
    r["controls"] = ctl
    bad = rig_problems(r)
    if bad:
        raise CertError("the x86 rig is not usable: " + "; ".join(bad[:4]))
    block = {"schema": CERT_SCHEMA, "mode": MODE_CERTIFIED, "lost_tolerance": LOST_TOLERANCE, "ledger": led,
             "checker": {"budget_s": float(checker_budget_s), "timeout_s": int(checker_budget_s) + 300},
             "x86": r}
    return block, files
