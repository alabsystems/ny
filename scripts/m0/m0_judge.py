#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""The decision half of scripts/m0/moat_both_profiles.sh: every verdict the runner acts on is computed here,
in python, where scripts/m0/m0_selftest.py can pin it. The shell only orchestrates.

    m0_judge.py validate --spec S [--no-files]      spec well-formed (+ files/configs/presets exist)
    m0_judge.py merge --out S --spec A --spec B      concatenate specs (unique tags)
    m0_judge.py markers --spec S                     pin marker fragments the spec's rows need, one per line
    m0_judge.py rows --spec S                        tab rows for the runner: tag cat onnx vnnlib budget cfg preset reps_c reps_p
    m0_judge.py judge --spec S --tag T --profile P --rep K --raw R [--flight F] [--res RES --box-rc N] --judged J
                                                     append one judgement; exit 0 OK, 3 EMERGENCY, 10 VOID
    m0_judge.py profiles --spec S --judged J [--tag T]   cross-run checks (the row's runs, and every tag that
                                                     shares its instance); exit 3 on EMERGENCY
    m0_judge.py prior --spec S --judged J            exit 3 when the record already holds an EMERGENCY (a
                                                     relaunch must not keep measuring past one)
    m0_judge.py summary --spec S --judged J          report + `M0_SUMMARY rc=...`; exit rc

Exit codes of `summary` (and so of the runner): 3 any EMERGENCY; 5 any VOID or not-run run; 4 any LOST
banked verdict; 0 clean. VOID is never a pass: a run that cannot prove what it ran proves nothing.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m0lib as L  # noqa: E402

EXIT_OK, EXIT_EMERGENCY, EXIT_LOST, EXIT_VOID = 0, 3, 4, 5
JUDGE_VOID_RC = 10


def load(spec: str) -> dict[str, dict]:
    return {r["tag"]: r for r in L.read_spec(spec)}


def schedule(row: dict) -> list[tuple[str, int]]:
    """Run order for one row: competition r1, proof r1, competition r2, ... (a flip shows up early)."""
    rc, rp = int(row["reps_c"]), int(row["reps_p"])
    out = []
    for k in range(1, max(rc, rp) + 1):
        if k <= rc:
            out.append(("competition", k))
        if k <= rp:
            out.append(("proof", k))
    return out


def cmd_validate(a) -> int:
    rows = L.read_spec(a.spec)
    errs = L.validate_spec(rows, check_files=not a.no_files)
    for e in errs:
        print(f"  SPEC-BAD {e}")
    print(f"SPEC_VALIDATE rc={1 if errs else 0} rows={len(rows)} errors={len(errs)}")
    return 1 if errs else 0


def cmd_merge(a) -> int:
    rows: list[dict] = []
    for s in a.spec:
        rows += L.read_spec(s)
    L.write_spec(a.out, rows)
    print(f"SPEC_MERGE rc=0 rows={len(rows)} out={a.out}")
    return 0


def cmd_markers(a) -> int:
    seen: list[str] = []
    for m in L.BASE_MARKERS + L.SELF_CONTRADICTION_MARKERS + L.SAT_ROW_MARKERS:
        if m not in seen:
            seen.append(m)
    for r in L.read_spec(a.spec):
        if r["tier"] != "-":
            for m in L.TIERS[r["tier"]].markers:
                if m not in seen:
                    seen.append(m)
    print("\n".join(seen))
    return 0


def cmd_rows(a) -> int:
    for r in L.read_spec(a.spec):
        print("\t".join(r[c] for c in ("tag", "category", "onnx", "vnnlib", "budget", "configs_dir", "preset",
                                       "reps_c", "reps_p")))
    return 0


def cmd_judge(a) -> int:
    rows = load(a.spec)
    if a.tag not in rows:
        raise L.M0Error(f"tag {a.tag} not in {a.spec}")
    raw = open(a.raw, errors="replace").read() if os.path.isfile(a.raw) else ""
    flight = None
    if a.flight and os.path.isfile(a.flight):
        try:
            flight = json.load(open(a.flight))
        except ValueError:
            flight = None
    res_text = None
    if a.res is not None:  # supplied: the scored file is the verdict of record ('' when ny never wrote it)
        res_text = open(a.res, errors="replace").read() if os.path.isfile(a.res) else ""
    j = L.judge_run(rows[a.tag], a.profile, a.rep, raw, flight, res_text, a.box_rc)
    new = not os.path.isfile(a.judged)
    with open(a.judged, "a") as fh:
        if new:
            fh.write("\t".join(L.JUDGED_COLUMNS) + "\n")
        fh.write(j.line() + "\n")
    print(j.line())
    return {"OK": 0, "EMERGENCY": EXIT_EMERGENCY}.get(j.status, JUDGE_VOID_RC)


def cross_run(rows: dict[str, dict], lines: list, tags: list[str]) -> list[tuple[str, str, str]]:
    """(status, tag, reason) for the given tags' own runs and for every instance they share with other tags."""
    by_tag: dict[str, list] = collections.defaultdict(list)
    for j in lines:
        by_tag[j.tag].append(j)
    out = [(status, t, reason) for t in tags for status, reason in L.judge_profiles(rows[t], by_tag[t])]
    insts = {L.instance_key(rows[t]) for t in tags}
    out += [(status, "-", reason) for status, reason in L.judge_instances(rows, lines, insts)]
    return out


def cmd_profiles(a) -> int:
    rows = load(a.spec)
    if a.tag and a.tag not in rows:
        raise L.M0Error(f"tag {a.tag} not in {a.spec}")
    lines = L.read_judged(a.judged)  # every line: a superseded judgement still stated what it stated
    rc = 0
    for status, t, reason in cross_run(rows, lines, [a.tag] if a.tag else list(rows)):
        print(f"  {status} {t} {reason}")
        if status == "EMERGENCY":
            rc = EXIT_EMERGENCY
    return rc


def cmd_prior(a) -> int:
    rows = load(a.spec)
    lines = L.read_judged(a.judged)
    hits = [f"{j.tag}/{j.profile}/r{j.rep}: {';'.join(j.reasons)}" for j in lines if j.status == "EMERGENCY"]
    hits += [f"{t}: {reason}" for status, t, reason in cross_run(rows, lines, [t for t in rows])
             if status == "EMERGENCY"]
    for h in hits:
        print(f"  PRIOR-EMERGENCY {h}")
    print(f"M0_PRIOR rc={EXIT_EMERGENCY if hits else 0} emergencies={len(hits)} judged_lines={len(lines)}")
    return EXIT_EMERGENCY if hits else 0


def cmd_summary(a) -> int:
    rows = load(a.spec)
    lines = L.read_judged(a.judged)
    by_key = L.collapse_runs(lines)  # a re-judged run keeps its last line, but an EMERGENCY is sticky
    by_tag: dict[str, list] = collections.defaultdict(list)
    for j in lines:
        by_tag[j.tag].append(j)
    counts = collections.Counter()
    groups: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    voids = collections.Counter()
    emerg, lost, notes, notrun = [], [], [], []
    per_row: dict[str, list[str]] = collections.defaultdict(list)
    for t, r in rows.items():
        for prof, k in schedule(r):
            j = by_key.get((t, prof, k))
            if j is None:
                notrun.append(f"{t}/{prof}/r{k}")
                counts["notrun"] += 1
                continue
            counts[j.status] += 1
            groups[r["group"]][f"{j.status}:{j.verdict}"] += 1
            per_row[t].append(f"{prof[0]}{k}={j.verdict}{'' if j.status == 'OK' else '(' + j.status + ')'}")
            if j.status == "EMERGENCY":
                emerg.append(f"{t}/{prof}/r{k}: {';'.join(j.reasons)}")
            elif j.status == "VOID":
                for reason in j.reasons:
                    if not reason.startswith("note:"):
                        voids[reason.split("(")[0][:70]] += 1
            if j.outcome == "lost" and j.status != "VOID":
                lost.append(f"{t}/{prof}/r{k}: bank={r['bank']} now={j.verdict}")
                counts["lost"] += 1
            if j.outcome == "converted" and j.status == "OK":
                counts["converted"] += 1
        for status, reason in L.judge_profiles(r, by_tag[t]):
            (emerg if status == "EMERGENCY" else notes).append(f"{t}: {reason}")
    for status, reason in L.judge_instances(rows, lines):
        (emerg if status == "EMERGENCY" else notes).append(reason)
    print(f"=== M0 summary: spec={a.spec} judged={a.judged}")
    for g in sorted(groups):
        print(f"  {g:44s} " + " ".join(f"{k}={v}" for k, v in sorted(groups[g].items())))
    print("  -- per row (c=competition p=proof, run order)")
    for t in rows:
        if per_row.get(t):
            print(f"    {t:44s} bank={rows[t]['bank']:7s} truth={rows[t]['truth']:5s} {' '.join(per_row[t])}")
    for e in emerg:
        print(f"  EMERGENCY {e}")
    for n in notes:
        print(f"  NOTE {n}")
    for s in lost:
        print(f"  LOST {s}")
    for reason, n in voids.most_common(12):
        print(f"  VOID x{n} {reason}")
    if notrun:
        print(f"  NOT-RUN {len(notrun)}: {' '.join(notrun[:12])}{' ...' if len(notrun) > 12 else ''}")
    rc = EXIT_EMERGENCY if emerg else EXIT_VOID if (counts["VOID"] or notrun) else EXIT_LOST if lost else EXIT_OK
    print(f"M0_SUMMARY rc={rc} runs={sum(len(schedule(r)) for r in rows.values())} ok={counts['OK']} "
          f"void={counts['VOID']} notrun={len(notrun)} emergency={len(emerg)} lost={len(lost)} "
          f"converted={counts['converted']} notes={len(notes)}")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate")
    p.add_argument("--spec", required=True)
    p.add_argument("--no-files", action="store_true")
    p = sub.add_parser("merge")
    p.add_argument("--out", required=True)
    p.add_argument("--spec", action="append", required=True)
    for name in ("markers", "rows"):
        p = sub.add_parser(name)
        p.add_argument("--spec", required=True)
    p = sub.add_parser("judge")
    for k in ("--spec", "--tag", "--raw", "--judged"):
        p.add_argument(k, required=True)
    p.add_argument("--profile", required=True, choices=("competition", "proof"))
    p.add_argument("--rep", required=True, type=int)
    p.add_argument("--flight")
    p.add_argument("--res", help="the run's results file (the scored verdict; '' when it was never written)")
    p.add_argument("--box-rc", type=int, help="exit status of the measured command (124 = the backstop fired)")
    p = sub.add_parser("profiles")
    p.add_argument("--spec", required=True)
    p.add_argument("--judged", required=True)
    p.add_argument("--tag")
    p = sub.add_parser("prior")
    p.add_argument("--spec", required=True)
    p.add_argument("--judged", required=True)
    p = sub.add_parser("summary")
    p.add_argument("--spec", required=True)
    p.add_argument("--judged", required=True)
    a = ap.parse_args(argv)
    try:
        return {"validate": cmd_validate, "merge": cmd_merge, "markers": cmd_markers, "rows": cmd_rows,
                "judge": cmd_judge, "profiles": cmd_profiles, "prior": cmd_prior,
                "summary": cmd_summary}[a.cmd](a)
    except L.M0Error as e:
        print(f"M0_JUDGE_ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
