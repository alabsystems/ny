#!/usr/bin/env python3
"""Detect drift between configs/preset_score_at_risk.yaml and reality.

WHY THIS EXISTS. `banked_normalized` prices what a preset model-load refusal
would delete, so the model-load guard can say "this deletes N banked points".
vit_2023 carried 16.8 for SIXTEEN DAYS after a proof retraction took its bank to
zero solved rows. Nothing noticed, because the only consumer
(crates/ny-cli/src/preset/model_load_smoke_tests.rs) asserts merely
`0.0 <= banked_normalized <= 100.0` — which 16.8 satisfied the whole time.

WHAT IT CAN AND CANNOT CHECK. Recomputing a member's normalized value needs the
WINNER denominators, which live in external_tools/vnncomp2025_results (gitignored,
absent from this box). So this checker deliberately does NOT attempt that. Every
check below is DENOMINATOR-FREE and would have caught the vit drift on day one.

WHICH BANK. The yaml prices the 2025 retroactive scorecard, so a benchmark's
bank is read ONLY from the directory that scorer reads for that row's track:
its `--measured` default for a REGULAR benchmark, its `--ext-measured` default
for an EXTENDED one — taken from the scorer's own argparse, the same anchor as
the benchmark lists. A bank for the same benchmark NAME anywhere else is not
evidence about this stake and is listed as a note, never as drift:
  * another field's ledger (reports/measured-2026 is the 2026 competition, a
    different instance set): yolo_2023's 2026 bank shares 10 of its 72
    instances with the 2025 set, and reading it here reported a 2025
    "recovered category" that the 2025 scorer cannot credit;
  * the scorer's OTHER track directory: seven extended benchmarks also have a
    reports/measured/<name>.csv, which the extended scorecard never reads
    (lsnc_relu: 80 solved there, 11 in reports/measured-ext).
The yaml's recorded `command` is CHECKED against that anchor, never followed:
if it passes `--measured` / `--ext-measured`, each must name the scorer's
default directory. Following it would anchor the check to the file under
test — a mistyped directory in the yaml silently narrowed check 3 from 24
banks to 8 and passed a zero-solved bank priced at 96.7 (planted 2026-10-08).
A scored member priced above 0.0 with NO bank in its directory is drift too:
the scorer credits an unmeasured benchmark 0.

Exit 0 = no drift. Exit 1 = drift. Exit 2 = the check could not run (VOID).
"""
from __future__ import annotations
import collections, csv, pathlib, re, shlex, sys, typing

ROOT = pathlib.Path(__file__).resolve().parent.parent
YAML = ROOT / "configs" / "preset_score_at_risk.yaml"
SCORER = ROOT / "scripts" / "ny_retroactive_scorecard.py"
# Members are stored to 1 dp; a 16-term sum accumulates rounding.
TOLERANCE = 0.2


def void(message: str) -> "typing.NoReturn":
    """The check could not run: say why and exit 2, the code this file documents.

    These used to be `raise SystemExit("VOID: ...")`, which exits 1 — the DRIFT
    code — so "could not look" and "looked and found drift" left with the same
    status.
    """
    print(f"VOID: {message}")
    raise SystemExit(2)


def scorer_lists() -> tuple[list[str], list[str]]:
    """The scorer's OWN benchmark lists — the anchor against a mis-scoped census.

    Anchoring here rather than to the yaml's own rows is the point: if a
    benchmark is dropped from the yaml, comparing the yaml to itself still
    passes. Comparing it to the scorer makes the omission fail closed.
    """
    src = SCORER.read_text()
    out = []
    for name in ("REGULAR", "EXTENDED"):
        m = re.search(rf"^{name}\s*=\s*\[(.*?)\]", src, re.S | re.M)
        if not m:
            void(f"cannot find `{name}` in {SCORER}")
        names = [a or b for a, b in re.findall(r'"([^"]+)"|\'([^\']+)\'', m.group(1))]
        if not names:
            void(f"`{name}` parsed as EMPTY in {SCORER}")
        out.append(names)
    return out[0], out[1]


def load_yaml():
    try:
        import yaml
    except ImportError:
        void("pyyaml unavailable; cannot check the score-at-risk file")
    doc = yaml.safe_load(YAML.read_text())
    # A duplicated top-level key is silently last-wins in YAML, which is how a
    # provenance stamp can be re-added without anyone noticing.
    seen, dupes = set(), set()
    for line in YAML.read_text().splitlines():
        m = re.match(r"^([a-z_]+):", line)
        if m:
            if m.group(1) in seen:
                dupes.add(m.group(1))
            seen.add(m.group(1))
    return doc, sorted(dupes)


VERDICTS = {"sat", "unsat", "timeout", "unknown", "error", "prepared"}
SOLVED = {"sat", "unsat"}


# The scorer's bank-directory flag for each scored track.
TRACK_FLAGS = (("regular", "--measured"), ("extended", "--ext-measured"))


def bank_dirs(doc: dict) -> tuple[dict[str, str], list[str]]:
    """({track: bank directory}, drift findings about the yaml's recorded command).

    The directories are the SCORER's own argparse defaults. The yaml's recorded
    `command` is compared against them and never followed: it is part of the
    file under test, so reading the bank from wherever it points lets one wrong
    token in the yaml choose what the yaml is checked against. A recorded
    command that names another directory is reported as drift (a stake priced
    from a ledger the scorer does not read by default is not one this check
    can vouch for); a default directory that does not exist is VOID.
    """
    src = SCORER.read_text()
    dirs: dict[str, str] = {}
    for track, flag in TRACK_FLAGS:
        m = re.search(rf'add_argument\(\s*"{flag}",\s*default="([^"]+)"', src, re.S)
        if not m:
            void(f"cannot find the scorer's `{flag}` default in {SCORER}")
        dirs[track] = m.group(1)
        if not (ROOT / dirs[track]).is_dir():
            void(f"bank directory `{dirs[track]}` (the scorer's `{flag}` default) does not "
                 "exist; check 3 would read nothing from it")

    failures: list[str] = []
    command = str(doc.get("command") or "")
    try:
        tokens = shlex.split(command)
    except ValueError as err:
        failures.append(f"recorded `command` cannot be tokenised ({err}): {command!r}")
        tokens = []
    for track, flag in TRACK_FLAGS:
        named = [t.split("=", 1)[1] for t in tokens if t.startswith(flag + "=")]
        for i, token in enumerate(tokens):
            if token == flag:
                named.append(tokens[i + 1] if i + 1 < len(tokens) else "")
        for value in named:
            if not value or (ROOT / value).resolve() != (ROOT / dirs[track]).resolve():
                failures.append(
                    f"recorded `command` passes `{flag} {value or '<nothing>'}` but the scorer "
                    f"reads `{dirs[track]}` by default — this check reads the scorer's "
                    "directory and cannot vouch for a stake priced from another ledger; "
                    "correct the recorded command, or re-price the yaml from the scorer's bank")
    return dirs, failures


def bank_files(bench: str, track: str, dirs: dict[str, str]) -> list[pathlib.Path]:
    """The files the scorer reads as `bench`'s bank on `track`, those that exist.

    Regular: `<--measured>/<bench>.csv`. Extended: `<--ext-measured>/<bench>.csv`
    or `ext_<bench>_results.csv` (both spellings are `load_ny_ext`'s).
    """
    names = [f"{bench}.csv"]
    if track == "extended":
        names.append(f"ext_{bench}_results.csv")
    return [path for path in (ROOT / dirs[track] / name for name in names) if path.is_file()]


def banks_elsewhere(bench: str, track: str, dirs: dict[str, str]) -> list[str]:
    """reports/*/<bench>.csv files the scorer does NOT read for this track's row."""
    own = (ROOT / dirs[track]).resolve()
    return sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "reports").glob(f"*/{bench}.csv")
        if path.parent.resolve() != own
    )


def solved_rows(bench: str, track: str, dirs: dict[str, str]) -> tuple[int, int] | None:
    """(solved, rows_with_a_recognisable_verdict), or None if there is no bank.

    The verdict COLUMN IS NOT FIXED across the bank, and hard-coding one is how
    the first version of this checker produced 21 confident, specific, WRONG
    findings: reports/measured-ext/vit_2023.csv has 5 fields with the verdict at
    index 3, while reports/measured/acasxu_2023.csv has 6-7 with the verdict at
    index 4 (index 3 is the literal "prepared"). Reading index 3 there yields
    "prepared", which is not in SOLVED, so 138 unsat + 46 sat rows read as ZERO.

    So: scan the row for a field that IS a verdict token, rather than trusting a
    position. A file where NO row yields a recognisable verdict returns 0 parsed
    rows, and the caller treats that as VOID rather than as "zero solved" — the
    two are opposite conclusions and must never be conflated.
    """
    files = bank_files(bench, track, dirs)
    if not files:
        return None
    solved = parsed = 0
    for path in files:
        with path.open() as fh:
            for row in csv.reader(fh):
                tokens = [c.strip() for c in row if c.strip() in VERDICTS]
                # "prepared" is a status column, not a verdict; require a real one.
                real = [t for t in tokens if t != "prepared"]
                if not real:
                    continue
                parsed += 1
                if real[-1] in SOLVED:
                    solved += 1
    return solved, parsed


def main() -> int:
    doc, dupes = load_yaml()
    reg, ext = scorer_lists()
    dirs, command_drift = bank_dirs(doc)
    notes: list[str] = []
    roots = doc.get("roots") or []
    totals = doc.get("totals") or {}
    if not roots or not totals:
        print("VOID: yaml has no roots/totals"); return 2

    failures: list[str] = list(command_drift)

    if dupes:
        failures.append(
            f"duplicated top-level key(s) {dupes}: YAML silently takes the LAST, "
            "so a re-added provenance stamp would override the real one")

    # -- values per benchmark, checking duplicate roots agree ------------------
    vals: dict[str, set[float]] = collections.defaultdict(set)
    for r in roots:
        vals[r["benchmark"]].add(float(r["banked_normalized"]))
    for bench, vs in sorted(vals.items()):
        if len(vs) > 1:
            failures.append(
                f"{bench}: duplicate roots disagree {sorted(vs)} — the alternate-year "
                "rows must carry the same stake")

    # -- CHECK 1: census anchored to the scorer, not to the yaml itself --------
    for label, names in (("REGULAR", reg), ("EXTENDED", ext)):
        missing = [n for n in names if n not in vals]
        if missing:
            failures.append(
                f"{label}: scored benchmark(s) {missing} are absent from the yaml — "
                "a refusal there would be priced as UNRECORDED")

    # -- CHECK 2: totals are the sum of their members (denominator-free) -------
    d_reg = sum(next(iter(vals[n])) for n in reg if n in vals)
    d_ext = sum(next(iter(vals[n])) for n in ext if n in vals)
    for key, derived in (("regular_16", d_reg), ("extended_9", d_ext),
                         ("full_field_25", d_reg + d_ext)):
        stated = float(totals.get(key, float("nan")))
        if abs(stated - derived) > TOLERANCE:
            failures.append(
                f"totals.{key}: stated {stated} but members sum to {derived:.1f} "
                f"(delta {derived - stated:+.1f}, tolerance +/-{TOLERANCE})")

    # -- CHECK 3: a bank with zero solved rows must be priced at 0.0 ----------
    # raw = 0 gives normalized 0.0 under BOTH branches of the scorer's
    # `meas_norm` (win_raw>0 -> max(0, 0); win_raw=0 -> denom 0 -> else 0.0),
    # so this needs no denominator. It is exactly the vit_2023 failure.
    # Scoped to SCORED benchmarks only. An `unscored` root (e.g. the 2026
    # benchmarks, outside the 2025 scored field) is legitimately priced 0.0 even
    # with a populated bank, because it contributes to no total. Checking those
    # too produced two false positives on the first run.
    # Each row is read from the ONE directory the scorer reads for its track,
    # and a scored member priced above 0.0 with no bank there is drift: the
    # scorer credits an unmeasured benchmark 0 ("unmeasured benchmarks score ny
    # 0"), so nothing in its inputs backs the stake.
    checked = 0
    for track, names in (("regular", reg), ("extended", ext)):
        for bench in sorted(b for b in names if b in vals):
            stake = next(iter(vals[bench]))
            got = solved_rows(bench, track, dirs)
            if got is None:
                elsewhere = banks_elsewhere(bench, track, dirs)
                if stake != 0.0:
                    failures.append(
                        f"{bench}: priced at {stake} but has no bank under {dirs[track]} — the "
                        f"scorer reads only that directory for the {track} track and credits "
                        "an unmeasured benchmark 0"
                        + (f" (not read for this row: {', '.join(elsewhere)})" if elsewhere else ""))
                elif elsewhere:
                    notes.append(
                        f"{bench}: no bank under {dirs[track]}; {', '.join(elsewhere)} is not "
                        f"an input of the scorer for this {track} row")
                continue
            n, parsed = got
            if parsed == 0:
                failures.append(
                    f"{bench}: bank file has NO recognisable verdict column — VOID, "
                    "not evidence of zero solved rows")
                continue
            checked += 1
            if n == 0 and stake != 0.0:
                failures.append(
                    f"{bench}: bank has ZERO sat/unsat rows but is priced at {stake} — "
                    "raw=0 forces normalized 0.0; this is the vit_2023 drift")
            if n > 0 and stake == 0.0:
                failures.append(
                    f"{bench}: bank has {n} solved rows but is priced at 0.0 — "
                    "a recovered category whose stake was never restored")
    if checked == 0:
        print("VOID: no benchmark bank files were found; check 3 verified NOTHING")
        for f in failures:
            print(f"  - {f}")
        return 2

    print(f"score-at-risk drift check: {len(vals)} benchmarks, {checked} with banks "
          f"(read from {' + '.join(dirs[track] for track, _ in TRACK_FLAGS)}), "
          f"totals regular={d_reg:.1f} extended={d_ext:.1f} full={d_reg + d_ext:.1f}")
    for note in notes:
        print(f"  note: {note}")
    if failures:
        print("\nDRIFT DETECTED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: yaml agrees with the bank and with the scorer's own benchmark lists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
