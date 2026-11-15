#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""keyless_certified: the CERTIFIED S8 moat of a KEYLESS, DEFAULT-ON routable capability (prepared-root) over
EVERY scored row its own admission admits, both years -- not only the m0 moat corpus.

Why a separate driver. `seallib.derive` refuses a keyless capability (no overlay arms it) and `seal_run.sh` needs an
overlay, markers and S5/S7 gates that a default-on capability does not have; `keyless_moat_plan.py` writes the
classic 3c+1p plan of the census-admitted MOAT rows (14364 runs, 303 h) and runs nothing. This driver writes and runs
the certified variant (scripts/seal/README.md "Certified S8") of the same capability, with the pipeline's own
functions (`census_corpus`, `census_admitted`, `s8_rows`, `keyless_moat_plan.tail_only` / `LANE_PROPERTY_GATES`,
certlib's classify / reproof / classic / certificate / replay checks, x86_replay's rig) and the m0 runner for every
measured run:

  S8   the census-admitted MOAT rows (m0 moat rows: banked sat/unsat + field-SAT), both years, every category
  S8X  the census-admitted scored rows OUTSIDE the moat corpus (banked undecided with no field SAT, unbanked
       categories): a false unsat there is invisible to the truth judge, so only the certificate can catch it
  S8T  the superset tail (keyless_moat_plan): rows the census refused ONLY for envelope:ops / admit:unknown:*,
       minus relational-route rows and minus rows the route's OWN property gates refuse (LANE_PROPERTY_GATES, each
       pinned by a named Rust test)

Every row runs ONCE in Competition on the SHIPPED configs (no overlay: the capability is already on), zero NY_*,
the official per-row budget, one `ny_box.sh measure` hold per run (the m0 runner). The rows are processed in
BATCHES (default 20), and each batch is certified before the next starts, so results are judged as they arrive:

  A        phase A: every batch row once in Competition (m0 runner; truth contradiction / bank flip / alarm =
           EMERGENCY at once)
  replay   every phase-A sat: the witness through the organisers' checker under emulated x86 (x86_replay's rig,
           the batch's own PASS + FAIL controls per checker kind); a rejected witness = WRONG SAT = EMERGENCY
  reproof  every phase-A unsat (unless a ledger certificate re-verifies): ONE Proof-profile run (tag S8R_)
  check    `<pin> proof-check` on the re-proof's bundle (one ny_box.sh build hold each)
  classic  every re-proof the checker did not ACCEPT: 3 Competition + 1 Proof (tag S8F_, m0 profile rules)
  judge    the batch verdict + the GLOBAL cross-tag INSTANCE-FLIP over every batch judged so far

The keyless plan's pre-registered certified-mode rule is applied on top of certlib's: a Proof re-run whose flight
`prepared_root` event reports the route PUBLISHED (status RootProven / TreeProven) must replay `verified`; any
other outcome on such a run (refuted, unconfirmed, invalid, malformed, no bundle) is an EMERGENCY ("the route
claims B2 holds").

  xcensus   --pin <bin> --out <dir> [--box ny_box.sh]     the out-of-moat corpus (every scored row of both years whose
            instance is NOT in the moat corpus; one entry per instance) and `<pin> vnncomp census` on it
  plan      --cap <id> --census-dir <moat census> --xcensus-dir <x census> --pin <bin> --out <dir>
            [--batch-rows 20] [--shards 1] [--checker-budget-s 900] [--cert-ledger <tsv>]
  sample    --out <plan dir> [--target-new 240] [--floor 3] [--engaged-weight 3] [--seed S] [--dry-run]
            AMENDMENT 2: a STRATIFIED SAMPLE of the plan's rows not yet in a judged batch (the full plan is ~4 box-
            months on the shared box). Strata (year, category, bank-predicted phase-A class); proportional allocation
            with a floor per stratum, engaged strata (an observed `prepared_root` Ran in a judged run of that unit, on a
            non-sat class) weighted --engaged-weight; rows already judged count toward their strata; a seeded order
            inside a stratum. Writes <out>/sample.json + its sha256 + the new batch specs (rows byte-identical to the
            plan's); plan.json and every pre-registered file stay untouched. Every runner helper takes --sample
            <sample.json>: the batch list becomes the carried (judged) batches + the sample's, while every judge,
            EMERGENCY and global cross-tag rule is unchanged (the global check still reads EVERY plan row).
  (runner helpers, called by keyless_certified_run.sh) step0, batches, reproof, replay, check-list, scan, judge-batch,
  summary
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seallib as S  # noqa: E402
from seallib import L, A, C, MR  # noqa: E402
import keyless_moat_plan as K  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
PLAN_SCHEMA = "seal-keyless-certified-plan/v1"
XCORPUS_PREFIX = "x"
STAGE_ORDER = ("S8", "S8X", "S8T")
ROUTE_EVENT = "prepared_root"
ROUTE_PUBLISHED = ("RootProven", "TreeProven")
ROUTE_STATUS_RE = re.compile(r"\bstatus: (\w+)")
ROUTE_REASON_RE = re.compile(r'\breason: "([^"]*)"')
BATCH_SCHEMA = "seal-keyless-certified-batch/v1"
SAMPLE_SCHEMA = "seal-keyless-certified-sample/v1"


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


class KCError(S.SealError):
    pass


# ---------------------------------------------------------------------------------------------------------
# every scored row of both years
# ---------------------------------------------------------------------------------------------------------


def field_categories(truth_path: str, year: int) -> set[str]:
    return set(L.parse_results_txt(truth_path, year))


def scored_rows(ny_root: str, cfg: str, truth_paths: dict) -> tuple[list[dict], list[str]]:
    """Every SCORED row of both years as an m0 spec row (tag S8_<yy>_<cat>_<idx>, reps 1/0, truth, bank). A category
    is scored when it is in the year's field results (2025: SCORING-ZERO-TOL; 2026: SCORING and the bank manifest's
    scored list). Banked categories: m0 moat_rows.build(all_rows=True) (the instances.csv the bank was measured on).
    Unbanked scored categories: their instances.csv (2026: EVERY version directory present -- the bank manifest names
    no version for them -- tagged `<cat>-v<ver>`; the category column stays the real name for the resolver)."""
    rows, notes = [], []
    for y in S.BOTH_YEARS:
        fc = field_categories(truth_paths[y], y)
        banked, unbanked = MR.categories(ny_root, y)
        b_in = sorted(c for c in banked if c in fc)
        built, _ = MR.build(ny_root, cfg, [y], set(b_in), truth_paths, True, 1, 0, "S8_", False)
        rows += built
        for c in sorted(set(banked) - fc):
            notes.append(f"{y}/{c}: banked but absent from the {y} field results (not scored): skipped")
        truth = L.parse_results_txt(truth_paths[y], y)
        for cat in sorted(c for c in unbanked if c in fc):
            if y == 2025:
                csvs = [(None, os.path.join(ny_root, "benchmarks/vnncomp2025/benchmarks", cat, "instances.csv"))]
            else:
                base = os.path.join(ny_root, "benchmarks/vnncomp2026/benchmarks", cat)
                csvs = [(v, os.path.join(base, v, "instances.csv")) for v in sorted(os.listdir(base))
                        if os.path.isfile(os.path.join(base, v, "instances.csv"))]
            if not csvs or not all(os.path.isfile(p) for _, p in csvs):
                raise KCError(f"{y}/{cat}: scored (field results) but no instances.csv")
            preset = L.resolve_preset(cfg, cat)
            for ver, p in csvs:
                inst = L.load_instances(p)
                try:
                    tr = L.align_truth(inst, truth.get(cat), f"{y}/{cat}{'@' + ver if ver else ''} truth")
                except L.M0Error as e:
                    notes.append(f"{y}/{cat}@{ver}: field rows do not align ({e}); truth '-' on this version")
                    tr = [None] * len(inst)
                tcat = f"{cat}-v{ver}" if ver else cat
                for r, t in zip(inst, tr):
                    tok = t.token() if t else "-"
                    rows.append(L.spec_row(
                        tag=f"S8_{y % 100}_{tcat}_{r.index:04d}", kind="MOAT", tier="-", year=y, category=cat,
                        onnx=r.onnx, vnnlib=r.vnnlib, budget=r.budget, configs_dir=cfg, preset=preset or "-",
                        truth=tok, bank="-", index=r.index, reps_c=1, reps_p=0, need="-", alarm="-", flight="-",
                        group=f"moat/{y}", note=f"unbanked{'@' + ver if ver else ''} "
                                                f"true_result={t.true_result if t else '-'}"))
    tags = collections.Counter(r["tag"] for r in rows)
    dup = [t for t, n in tags.items() if n > 1]
    if dup:
        raise KCError(f"scored rows: duplicate tags {dup[:5]}")
    return rows, notes


def row_id_of(r: dict) -> str:
    """`<year>/<category>[@<ver>]/<index>` (the corpus map's row names)."""
    ver = re.search(r"-v([0-9.]+)_[0-9]{4}$", r["tag"])
    return f"{r['year']}/{r['category']}{'@' + ver.group(1) if ver else ''}/{r['index']}"


def xcorpus(ny_root: str, cfg: str, truth_paths: dict) -> tuple[dict, dict, list[dict], list[str]]:
    """(moat corpus entries, out-of-moat corpus entries, every scored row, notes): the x corpus holds one entry per
    DISTINCT instance (onnx, vnnlib) of every scored row whose instance is NOT in the moat corpus."""
    _, moat = S.census_corpus(ny_root, cfg, truth_paths)
    rows, notes = scored_rows(ny_root, cfg, truth_paths)
    ent: dict = collections.OrderedDict()
    for r in rows:
        k = L.instance_key(r)
        if k in moat:
            continue
        if k not in ent:
            ent[k] = {"row_id": f"{XCORPUS_PREFIX}{len(ent) + 1:05d}", "onnx": r["onnx"], "vnnlib": r["vnnlib"],
                      "budget": int(r["budget"]), "rows": []}
        ent[k]["rows"].append(row_id_of(r))
    return moat, ent, rows, notes


def cmd_xcensus(a) -> int:
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    truth = {2026: a.truth_2026, 2025: a.truth_2025}
    moat, ent, rows, notes = xcorpus(a.ny_root, a.configs, truth)
    corpus = os.path.join(out, "corpus.tsv")
    if os.path.exists(corpus):
        raise KCError(f"{corpus} exists: a census directory is written once")
    S.write_new(corpus, S.corpus_text(ent))
    S.write_new(os.path.join(out, "corpus_map.json"),
                json.dumps({e["row_id"]: {"onnx": e["onnx"], "vnnlib": e["vnnlib"], "budget": e["budget"],
                                          "rows": e["rows"]} for e in ent.values()}, indent=0, sort_keys=True) + "\n")
    c = S.run_census(os.path.abspath(a.pin), corpus, os.path.join(out, "census.tsv"), a.box)
    counts = S.census_counts(c, ent)
    S.write_new(os.path.join(out, "counts.json"), json.dumps(counts, indent=1, sort_keys=True) + "\n")
    for n in notes:
        print(f"  note {n}")
    nrows = sum(len(e["rows"]) for e in ent.values())
    print(f"=== out-of-moat census {c['path']} census_sha256={c['census_sha256'][:16]}: {len(ent)} instance(s) = "
          f"{nrows} scored row(s) outside the moat corpus ({len(moat)} moat instances; {len(rows)} scored rows total)")
    for cap, v in counts.items():
        print(f"  {cap:28s} admit={v['admit']:5d} router={v['router']:5d} rows={v['admit_moat_rows']:5d}")
    print(f"KC_XCENSUS_DONE rc=0 instances={len(ent)} rows={nrows} scored_rows={len(rows)} "
          + " ".join(f"{cap}={v['admit']}/{v['admit_moat_rows']}" for cap, v in counts.items())
          + f" census_sha256={c['census_sha256']} out={out}")
    return 0


# ---------------------------------------------------------------------------------------------------------
# the plan (pre-registered once)
# ---------------------------------------------------------------------------------------------------------


def _census_cells(c: dict, entries: dict, cap: str) -> dict:
    out = {}
    for k, e in entries.items():
        row = c["rows"].get(e["row_id"])
        if row is None or cap not in row:
            raise KCError(f"census {c['path']} has no row {e['row_id']} / {cap} for {k}")
        out[k] = row[cap]
    return out


def keyed_rows(rows: list[dict]) -> dict:
    by: dict = collections.defaultdict(list)
    for r in rows:
        by[L.instance_key(r)].append(r)
    return by


def order_rows(rows: list[dict], seed: str) -> list[dict]:
    """Round-robin over (year, category) units -- so the first batches cover every unit and an early stop still
    samples each -- with a seeded order inside a unit (sha256(seed|tag): index order groups model / property
    families in several categories, so the first rows of a unit would not be representative)."""
    units: dict = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: (int(r["year"]) * -1, r["category"])):
        units.setdefault((r["year"], r["category"]), []).append(r)
    lists = [sorted(v, key=lambda r: hashlib.sha256(f"{seed}|{r['tag']}".encode()).hexdigest())
             for v in units.values()]
    return S.round_robin(lists)


def walls_of(ny_root: str, rows: list[dict]) -> dict[str, float | None]:
    """The banked wall per row tag (None when unbanked or the bank carries none)."""
    out: dict[str, float | None] = {}
    cache: dict = {}
    for r in rows:
        y, cat = int(r["year"]), r["category"]
        if (y, cat) not in cache:
            bp = L.bank_csv(ny_root, y, cat)
            if not os.path.isfile(bp):
                cache[(y, cat)] = None
            else:
                try:
                    inst = L.load_instances(L.scored_instances_csv(ny_root, y, cat), check_files=False)
                    _, walls, _ = A.align_bank_full(inst, A._bank_rows(bp), f"{y}/{cat} bank")
                    cache[(y, cat)] = walls
                except (L.M0Error, S.SealError):
                    cache[(y, cat)] = None
        walls = cache[(y, cat)]
        idx = int(r["index"])
        out[r["tag"]] = walls[idx] if walls is not None and idx < len(walls) else None
    return out


RUN_OVERHEAD_S = A.A_RUN_OVERHEAD_S     # ny_box.sh measure: quiescence windows + process start


def estimate(rows: list[dict], walls: dict) -> dict:
    """Box-hour estimates of the certified moat. ceiling: every run uses its full budget (+ checks at their budget);
    expected: phase A at the banked wall (budget when unbanked, timeout or unknown), a Proof re-run of every row
    expected to state unsat (bank unsat, else field truth unsat) at ~1.2x its banked wall, a check per re-proof
    (~min(60 s, budget)), 12 s box overhead per run; and the two bounds on the classic sub-moat (none of the
    re-proofs fall back / all of them do: 3 Competition + 1 Proof each)."""
    ceil_a = sum(int(r["budget"]) for r in rows)
    exp_a = exp_r = exp_chk = exp_f = 0.0
    n_unsat = n_sat = 0
    for r in rows:
        b = int(r["budget"])
        w = walls.get(r["tag"])
        bank = r["bank"]
        wa = min(float(w), b) if (w is not None and bank in L.DECIDED) else float(b)
        exp_a += wa + RUN_OVERHEAD_S
        expect_unsat = bank == "unsat" or (bank not in L.DECIDED and r["truth"] == "unsat")
        if expect_unsat:
            n_unsat += 1
            wr = min(1.2 * wa, b) if bank == "unsat" else float(b)
            exp_r += wr + RUN_OVERHEAD_S
            exp_chk += min(60.0, b)
            exp_f += 3 * (wa + RUN_OVERHEAD_S) + (wr + RUN_OVERHEAD_S)
        if bank == "sat" or r["truth"] == "sat":
            n_sat += 1
    ceil_r = sum(int(r["budget"]) for r in rows if r["bank"] == "unsat" or r["truth"] == "unsat")
    return {"rows": len(rows), "expected_unsat_rows": n_unsat, "expected_sat_rows": n_sat,
            "ceiling_h": {"phase_a": round(ceil_a / 3600, 1), "reproof": round(ceil_r / 3600, 1),
                          "classic_all_fallback": round(4 * ceil_r / 3600, 1)},
            "expected_h": {"phase_a": round(exp_a / 3600, 1), "reproof": round(exp_r / 3600, 1),
                           "checks": round(exp_chk / 3600, 1), "replays": round(n_sat * 45 / 3600, 1),
                           "classic_if_none_fall_back": 0.0, "classic_if_all_fall_back": round(exp_f / 3600, 1)},
            "expected_total_h": {"low": round((exp_a + exp_r + exp_chk + n_sat * 45) / 3600, 1),
                                 "high": round((exp_a + exp_r + exp_chk + n_sat * 45 + exp_f) / 3600, 1)}}


def cmd_plan(a) -> int:
    out = os.path.abspath(a.out)
    if os.path.exists(os.path.join(out, "plan.json")):
        raise KCError(f"{out}/plan.json exists: a plan is pre-registered once")
    if a.batch_rows < 1:
        raise KCError("--batch-rows must be >= 1")
    reg = S.load_registry(a.registry)
    cap = S.get_capability(reg, a.cap)
    live = [k.get("path") for k in cap.get("keys") or [] if not k.get("retired")]
    if live:
        raise KCError(f"{a.cap} has live key(s) {live}: it is armed by an overlay; use `seal.py plan --s8-mode certified`")
    if not cap.get("routable"):
        raise KCError(f"{a.cap} is not routable: the census has no decision for it")
    pin = os.path.abspath(a.pin)
    pin_sha = K.sha256_file(pin)
    bid = S.build_id(pin)
    if bid["dirty"] != "false":
        raise KCError(f"pin {pin} is dirty ({bid['line'][:160]}): a plan binds committed bytes")
    if bid.get("exe") and bid["exe"] != pin_sha:
        raise KCError(f"pin --build-id exe={bid['exe'][:16]} != sha256 {pin_sha[:16]}")
    commit = bid["commit"]
    S.git(a.repo, "cat-file", "-e", f"{commit}^{{commit}}")
    heads = cap.get("heads") or []
    missing = [h for h in heads if subprocess.run(["git", "-C", a.repo, "merge-base", "--is-ancestor", h, commit],
                                                  capture_output=True).returncode != 0]
    if missing:
        raise KCError(f"pin commit {commit[:9]} does not contain the capability's head(s) {missing}")
    truth = {2026: a.truth_2026, 2025: a.truth_2025}
    os.makedirs(out, exist_ok=True)
    cfg_src = os.path.join(out, "configs", "src")
    cfg_tree = S.extract_tree(a.repo, commit, "configs", cfg_src)
    # the two censuses: each must bind to THIS pin and to the corpus re-derived now
    moat, xent, scored, notes = xcorpus(a.ny_root, cfg_src, truth)
    cm = S.parse_census(os.path.join(os.path.abspath(a.census_dir), "census.tsv"))
    cx = S.parse_census(os.path.join(os.path.abspath(a.xcensus_dir), "census.tsv"))
    pm = S.census_binding_problems(cm, pin_sha, S.sha256_text(S.corpus_text(moat)), a.cap)
    px = S.census_binding_problems(cx, pin_sha, S.sha256_text(S.corpus_text(xent)), a.cap)
    if pm or px:
        raise KCError(f"a census does not bind to this plan: moat {pm}; out-of-moat {px}")
    if cm["binding"].get("host") != cx["binding"].get("host"):
        raise KCError(f"the two censuses were made on different hosts ({cm['binding'].get('host')} / "
                      f"{cx['binding'].get('host')})")
    gates = K.LANE_PROPERTY_GATES.get(a.cap, {})
    served = tuple((cap["routable"].get("precondition") or {}).get("specs") or ())
    cells = {**_census_cells(cm, moat, a.cap), **_census_cells(cx, xent, a.cap)}
    adm = {k for k, c in cells.items() if c["admit"] == "1"}
    tail = {k for k, c in cells.items() if c["admit"] != "1" and K.tail_only(c, gates, served)}
    gated = collections.Counter()
    gated_rows: dict = collections.defaultdict(list)
    by_inst = keyed_rows(scored)
    for k, c in cells.items():
        if k not in adm and k not in tail and K.tail_only(c):
            why = K.lane_refuses_spec(c, gates, served) or "?"
            for r in by_inst.get(k, []):
                gated[why] += 1
                gated_rows[why].append(row_id_of(r))
    # the rows: S8 = the census-admitted MOAT rows (s8_rows, exactly as keyless_moat_plan, reps 1/0), S8X = every
    # other scored row of an admitted instance, S8T = every scored row of a tail instance
    d = K.keyless_derived(a.cap, list(S.BOTH_YEARS))
    s8 = S.s8_rows(d, {"arm": {"dir": cfg_src}}, a.ny_root, truth, 1, 0, {k for k in adm if k in moat})
    s8_tags = {r["tag"] for r in s8}
    stage_rows: dict[str, list[dict]] = {"S8": s8, "S8X": [], "S8T": []}
    for r in scored:
        if r["tag"] in s8_tags:
            continue
        k = L.instance_key(r)
        st = "S8X" if k in adm else "S8T" if k in tail else None
        if st is None:
            continue
        al = S.row_alarms(d, r["truth"], r["bank"])
        r = dict(r, alarm=L.ALARM_SEP.join(al) if al else "-",
                 flight=S.preset_event(cfg_src, r["preset"]) if r["preset"] != "-" else "-")
        stage_rows[st].append(r)
    # a MOAT row (m0 select_row: banked sat/unsat or field-SAT) of an admitted instance belongs to S8, never S8X/S8T
    leak = [r["tag"] for r in stage_rows["S8X"] + stage_rows["S8T"]
            if r["bank"] != "-" and MR.select_row(r["bank"], r["truth"]) and L.instance_key(r) in adm]
    if leak:
        raise KCError(f"a moat row escaped S8: {leak[:5]}")
    for st, rows in stage_rows.items():
        for r in rows:
            r["group"] = f"seal/{st}/{r['year']}"
            r["note"] = (r.get("note") or "-")[:200]
    tags = collections.Counter(r["tag"] for rows in stage_rows.values() for r in rows)
    if any(n > 1 for n in tags.values()):
        raise KCError("a row is in two stages")
    seed = a.seed or f"kc|{a.cap}|{pin_sha[:16]}"
    ordered = order_rows(stage_rows["S8"] + stage_rows["S8X"], seed) + order_rows(stage_rows["S8T"], seed)
    smoke = [t.strip() for t in (a.smoke_tags or "").split(",") if t.strip()]
    if smoke:
        # a SMOKE plan: exactly the named rows (they must be rows of the full plan), never evidence
        have = {r["tag"]: r for r in ordered}
        miss = [t for t in smoke if t not in have]
        if miss:
            raise KCError(f"smoke tags are not rows of the plan: {miss}")
        ordered = [have[t] for t in smoke]
        stage_rows = {st: [r for r in rows if r["tag"] in set(smoke)] for st, rows in stage_rows.items()}
    stage_of = {r["tag"]: st for st, rows in stage_rows.items() for r in rows}
    # shards: greedy budget balance keeping an instance in one shard (seallib.partition); a shard keeps the global
    # order and is cut into batches
    n = int(a.shards)
    shard_of = S.partition(ordered, n) if n > 1 else {r["tag"]: 1 for r in ordered}
    cblock, cfiles = C.plan_block(out, a.cert_ledger, None, float(a.checker_budget_s), S.write_new)
    kinds = C.control_kinds(cblock["x86"])
    need = collections.Counter(C.checker_kind(r) for r in ordered if "sat" in (r["bank"], r["truth"]))
    miss = sorted(k for k in need if k not in kinds)
    if miss:
        raise KCError(f"the x86 rig has no PASS + FAIL control of kind(s) {miss} ({sum(need[k] for k in miss)} rows)")
    files: dict[str, str] = dict(cfiles)
    shards: dict = {}
    for k in range(1, n + 1):
        sk = S.shard_key(k, n)
        mine = [r for r in ordered if shard_of[r["tag"]] == k]
        batches = []
        for i in range(0, len(mine), a.batch_rows):
            b = f"b{i // a.batch_rows + 1:04d}"
            p = os.path.join(out, "specs", sk, f"{b}.tsv")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            chunk = mine[i:i + a.batch_rows]
            L.write_spec(p, chunk)
            errs = L.validate_spec(L.read_spec(p), check_files=True)
            if errs:
                raise KCError(f"{p} invalid: {errs[:5]}")
            rel = os.path.relpath(p, out)
            files[rel] = L.sha256_file(p)
            batches.append({"id": b, "spec": rel, "rows": len(chunk),
                            "stages": dict(collections.Counter(stage_of[r["tag"]] for r in chunk))})
        shards[sk] = {"rows": len(mine), "batches": batches}
    walls = walls_of(a.ny_root, ordered)
    stages = {}
    for st in STAGE_ORDER:
        rows = stage_rows[st]
        stages[st] = {"rows": len(rows), "runs_phase_a": len(rows),
                      "per_category": dict(sorted(collections.Counter(f"{r['year']}/{r['category']}"
                                                                      for r in rows).items())),
                      "truth": dict(collections.Counter(r["truth"] for r in rows)),
                      "bank": dict(collections.Counter(r["bank"] for r in rows)),
                      "estimate": estimate(rows, walls)}
    plan = {
        "schema": PLAN_SCHEMA, "capability": a.cap, "created": now(), "out": out, "seed": seed,
        "registry": {"path": os.path.relpath(os.path.abspath(a.registry), REPO), "sha256": L.sha256_file(a.registry),
                     "entry_sha256": S.sha256_text(S.canonical(cap))},
        "pin": {"path": pin, "sha256": pin_sha, "commit": commit, "features": bid["features"], "build_id": bid["line"]},
        "configs": {"source": f"git:{commit}:configs", "tree": cfg_tree, "src": cfg_src,
                    "overlay": "none: the capability is default-on in the shipped code (no key arms it)"},
        "census": {
            "moat": {"dir": os.path.abspath(a.census_dir), "tsv_sha256": cm["sha256"],
                     "census_sha256": cm["census_sha256"], "binding": cm["binding"], "instances": len(moat),
                     "rows": sum(len(e["rows"]) for e in moat.values())},
            "out_of_moat": {"dir": os.path.abspath(a.xcensus_dir), "tsv_sha256": cx["sha256"],
                            "census_sha256": cx["census_sha256"], "binding": cx["binding"], "instances": len(xent),
                            "rows": sum(len(e["rows"]) for e in xent.values())},
            "scored_rows": len(scored), "admit_instances": len(adm), "tail_instances": len(tail),
            "tail_reasons": list(K.TAIL_REASONS),
            "tail_lane_gates": {"gates": gates, "served": list(served), "excluded_rows": dict(sorted(gated.items())),
                                "excluded_row_ids": {k: v for k, v in sorted(gated_rows.items())}},
            "notes": notes},
        "stages": stages,
        "order": {"rule": "S8 + S8X round-robin over (year, category) units, seeded order inside a unit; then S8T",
                  "batch_rows": a.batch_rows, "shards": n},
        "shards": shards,
        "rows_index": {r["tag"]: {"stage": stage_of[r["tag"]], "shard": S.shard_key(shard_of[r["tag"]], n)}
                       for r in ordered},
        "certified": cblock,
        "rules": {
            "phase_a": "m0 judge_run per run (scripts/m0): truth contradiction, bank flip, defect alarm, self-"
                       "contradiction = EMERGENCY at once (m0 exit 3); wrong instance / preset / profile, no Result "
                       "line, ambient NY_* = VOID (never a pass)",
            "replay": "every phase-A sat: x86_replay.replay_row under the plan's rig after the batch's own controls "
                      "(certlib.verify_replay_record); WITNESS_FAIL = WRONG SAT = EMERGENCY; inconclusive = VOID",
            "reproof": "every phase-A unsat not covered by a re-verified ledger certificate: one Proof run (S8R_, "
                       "+ `^Result: sat$` alarm); a Proof sat = EMERGENCY; certlib.reproof_outcome",
            "check": "<pin> proof-check <bundle> --onnx <row onnx> --vnnlib <row vnnlib> --report <r> --budget-s "
                     f"{float(a.checker_budget_s):.0f}; refuted = EMERGENCY (certlib.check_certificate)",
            "route_b2": "keyless plan certified_mode: a Proof re-run whose flight `prepared_root` event is Ran with "
                        "status RootProven/TreeProven must ACCEPT (verdict=verified); any final non-ACCEPT on such a "
                        "run (fallback or no bundle) is an EMERGENCY",
            "classic": "every re-proof that did not ACCEPT: 3 Competition + 1 Proof (S8F_), m0 profile rules "
                       "(VERDICT-FLIP / PROFILE-DECIDED-SPLIT = EMERGENCY)",
            "global": "cross-tag INSTANCE-FLIP (m0 judge_instances) over every judged run of every batch so far = "
                      "EMERGENCY",
            "lost": "a banked sat/unsat the phase-A run (or a Proof re-run, or a classic run) does not reproduce "
                    "is LOST: reported per row, PARTIAL above tolerance 0",
            "engagement": "per run, the flight `prepared_root` event (Ran: RootProven/TreeProven/Open; Skipped: "
                          "the admission's or the route's refusal reason; absent: the run returned before the route, "
                          "e.g. an upfront-attack sat)",
            "stop": "any EMERGENCY writes <out>/EMERGENCY and stops every shard before its next run"},
        "files": files,
    }
    if smoke:
        plan["smoke"] = {"rows": smoke, "note": "a SMOKE plan: the pipeline end to end on named rows; never evidence"}
    text = json.dumps(plan, indent=1, sort_keys=True, default=str) + "\n"
    S.write_new(os.path.join(out, "plan.json"), text)
    S.write_new(os.path.join(out, "plan.json.sha256"), S.sha256_text(text) + "\n")
    print(render(plan))
    tot = sum(v["rows"] for v in stages.values())
    print(f"KC_PLAN_DONE rc=0 capability={a.cap} rows={tot} " + " ".join(f"{k}={v['rows']}" for k, v in stages.items())
          + f" shards={n} batches={sum(len(v['batches']) for v in shards.values())} out={out}")
    return 0


def render(plan: dict) -> str:
    o = [f"=== keyless certified plan: {plan['capability']} ({plan['out']})",
         f"  pin      {plan['pin']['path']} sha256={plan['pin']['sha256'][:16]} commit={plan['pin']['commit'][:12]} "
         f"features={plan['pin']['features']}",
         f"  configs  {plan['configs']['source']} (tree {plan['configs']['tree'][:12]}); {plan['configs']['overlay']}"]
    c = plan["census"]
    o.append(f"  census   moat {c['moat']['census_sha256'][:16]} ({c['moat']['instances']} inst) + out-of-moat "
             f"{c['out_of_moat']['census_sha256'][:16]} ({c['out_of_moat']['instances']} inst): {c['scored_rows']} "
             f"scored rows; admit {c['admit_instances']} inst; tail {c['tail_instances']} inst; gated out "
             f"{c['tail_lane_gates']['excluded_rows']}")
    for st, v in plan["stages"].items():
        e = v["estimate"]
        o.append(f"  stage {st:4s} rows={v['rows']:5d} truth={v['truth']} bank={v['bank']}")
        o.append(f"             ceiling_h={e['ceiling_h']} expected_h={e['expected_h']} total={e['expected_total_h']}")
    for sk, v in plan["shards"].items():
        o.append(f"  shard {sk}: {v['rows']} rows in {len(v['batches'])} batches of <= {plan['order']['batch_rows']}")
    cb = plan["certified"]
    o.append(f"  certified: checker budget {cb['checker']['budget_s']:.0f}s; x86 {cb['x86']['cpu']} ORT "
             f"{cb['x86']['onnxruntime']} controls {sorted({x['kind'] for x in cb['x86']['controls']})}; ledger "
             f"{(cb.get('ledger') or {}).get('entries', 0)}")
    return "\n".join(o)


def load_plan(out: str, verify_files: bool = True, sample: str | None = None) -> dict:
    """The sha-locked plan; with `sample`, the EFFECTIVE plan of a sample (AMENDMENT 2): the shard batch lists become
    the sample's carried batches + its new batches, `base_shards` keeps the full plan's lists (the global cross-tag
    check reads every plan row), and `sample` holds the sample's record. The sample must bind to THIS plan's bytes."""
    p = os.path.join(out, "plan.json")
    text = open(p).read()
    want = open(p + ".sha256").read().strip() if os.path.isfile(p + ".sha256") else None
    if want != S.sha256_text(text):
        raise KCError(f"{p}: sha256 differs from plan.json.sha256 (the pre-registered plan was edited)")
    plan = json.loads(text)
    if plan.get("schema") != PLAN_SCHEMA:
        raise KCError(f"{p}: schema {plan.get('schema')!r}")
    if verify_files:
        bad = [rel for rel, sha in plan["files"].items()
               if not os.path.isfile(os.path.join(out, rel)) or L.sha256_file(os.path.join(out, rel)) != sha]
        if bad:
            raise KCError(f"pre-registered file(s) changed or missing since the plan: {bad[:6]}")
    if sample:
        plan = apply_sample(plan, load_sample(out, S.sha256_text(text), sample, verify_files))
    return plan


def load_sample(out: str, plan_sha: str, path: str, verify_files: bool = True) -> dict:
    """A sample record: its own sha256 lock, its schema, bound to the plan's sha256 and to this plan directory, and
    (verify_files) every spec it wrote unchanged."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise KCError(f"sample {path} does not exist")
    text = open(path).read()
    want = open(path + ".sha256").read().strip() if os.path.isfile(path + ".sha256") else None
    if want != S.sha256_text(text):
        raise KCError(f"{path}: sha256 differs from {os.path.basename(path)}.sha256 (the sample was edited)")
    s = json.loads(text)
    if s.get("schema") != SAMPLE_SCHEMA:
        raise KCError(f"{path}: schema {s.get('schema')!r}")
    if s.get("base_plan_sha256") != plan_sha:
        raise KCError(f"{path}: drawn from plan sha256 {str(s.get('base_plan_sha256'))[:16]}, this plan is "
                      f"{plan_sha[:16]}")
    if os.path.normpath(s.get("out") or "") != os.path.normpath(out):
        raise KCError(f"{path}: drawn for plan directory {s.get('out')}, not {out}")
    if verify_files:
        bad = [rel for rel, sha in s["files"].items()
               if not os.path.isfile(os.path.join(out, rel)) or L.sha256_file(os.path.join(out, rel)) != sha]
        if bad:
            raise KCError(f"sample file(s) changed or missing since the sample was drawn: {bad[:6]}")
    s["_path"] = path
    s["_sha256"] = S.sha256_text(text)
    return s


def apply_sample(plan: dict, s: dict) -> dict:
    p = dict(plan)
    p["base_shards"] = plan["shards"]
    shards = {}
    for sk, sh in plan["shards"].items():
        ss = s["shards"].get(sk)
        if ss is None:
            shards[sk] = {"rows": 0, "batches": []}
            continue
        base_ids = [b["id"] for b in sh["batches"]]
        miss = [b for b in ss["carried"] if b not in base_ids]
        if miss:
            raise KCError(f"sample carries batch(es) {miss} that are not in plan shard {sk}")
        clash = [b["id"] for b in ss["batches"] if b["id"] in base_ids]
        if clash:
            raise KCError(f"sample batch id(s) {clash} collide with plan batch ids")
        bat = [b for b in sh["batches"] if b["id"] in set(ss["carried"])] + list(ss["batches"])
        shards[sk] = {"rows": sum(b["rows"] for b in bat), "batches": bat}
    p["shards"] = shards
    p["files"] = {**plan["files"], **s["files"]}
    p["sample"] = s
    return p


# ---------------------------------------------------------------------------------------------------------
# a batch: its paths, phase-A facts, the follow-up derivations
# ---------------------------------------------------------------------------------------------------------


# AMENDMENT 1 (2026-10-09, after batch b0001; recorded in <out>/AMENDMENTS): the m0 judge VOIDs a run whose
# `Loading preset:` line names another file than <configs_dir>/<preset>. The scored entry point's plan resolver
# MERGES model / budget / backend-adaptive defaults into a temporary effective preset (`/tmp/ny-plan-*.yaml`,
# vnncomp.rs `Plan resolver: merged effective preset P (base: B)`) on some rows -- shipped behaviour, what
# run_instance.sh runs -- so the m0 rule reads those runs as "wrong preset" although they loaded the plan's preset.
# Measured: S8_25_traffic_signs_recognition_2023_0012 (base vnncomp25/traffic_signs_recognition_2023.yaml). Such a run
# is accepted (status OK, note `preset-merged`) ONLY when every non-note VOID reason is that preset mismatch, the
# wanted file is the row's own preset, the raw log names the SAME temporary file on both its `Plan resolver: merged
# effective preset P (base: B)` and `Loading preset: P` lines with B the row's preset, and the flight `preset_base`
# event is B. The m0 soundness checks (truth contradiction, bank flip, alarms) run before any VOID and are untouched.
MERGED_VOID_RE = re.compile(r"^preset:(/tmp/ny-plan-[A-Za-z0-9_]+\.yaml)\(want (.+)\)$")


def merged_preset_ok(row: dict, j, d: str) -> str | None:
    """The base preset when the VOID judgement `j` (run dir `d`) is ONLY the merged-preset false VOID, else None."""
    if j is None or j.status != "VOID" or row is None or row["preset"] == "-":
        return None
    want = os.path.normpath(os.path.join(row["configs_dir"], row["preset"]))
    reasons = [r for r in j.reasons if not r.startswith("note:")]
    if not reasons:
        return None
    tmp = None
    for r in reasons:
        m = MERGED_VOID_RE.match(r)
        if not m or os.path.normpath(m.group(2)) != want or (tmp is not None and m.group(1) != tmp):
            return None
        tmp = m.group(1)
    raw = C._read(C.raw_path(d, j.tag, j.profile, j.rep)) or ""
    body = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("[ny-box"))
    merged = re.findall(r"^Plan resolver: merged effective preset (\S+) \(base: (.+)\)$", body, re.M)
    loads = re.findall(r"^Loading preset: (\S+)", body, re.M)
    if merged != [(tmp, want)] or loads != [tmp]:
        return None
    fl = C._json(C.res_path(d, j.tag, j.profile, j.rep) + ".flight.json") or {}
    bases = [e.get("reason") for e in fl.get("events") or [] if isinstance(e, dict) and e.get("method") == "preset_base"]
    if not bases or any(os.path.normpath(str(b)) != want for b in bases):
        return None
    return want


def effective(lines: list, rows: dict, d: str) -> list:
    """The judged lines with AMENDMENT 1 applied (a verified merged-preset false VOID becomes OK, noted)."""
    out = []
    for j in lines:
        base = merged_preset_ok(rows.get(j.tag), j, d)
        if base is not None:
            j = L.Judgement(j.tag, j.profile, j.rep, j.verdict, "OK", j.outcome, j.wall,
                            [r for r in j.reasons if r.startswith("note:")] + [f"note:preset-merged(base={base})"])
        out.append(j)
    return out


def shard_dir(out: str, sk: str, phase: str) -> str:
    return os.path.join(out, "run", sk, phase)


class Batch:
    def __init__(self, out: str, plan: dict, sk: str, bid: str):
        sh = plan["shards"].get(sk)
        if sh is None:
            raise KCError(f"shard {sk} is not in the plan ({sorted(plan['shards'])})")
        ent = [b for b in sh["batches"] if b["id"] == bid]
        if not ent:
            raise KCError(f"batch {bid} is not in shard {sk}")
        self.out, self.plan, self.sk, self.id = out, plan, sk, bid
        self.spec = os.path.join(out, ent[0]["spec"])
        self.base = os.path.join(out, "run", sk, bid)
        # the m0 output dirs are SHARED by the shard's batches (one pin copy, one judged.tsv per phase: the m0
        # runner keys every run by its unique tag, resumes by its own markers, and its `prior` check refuses a
        # relaunch past ANY recorded emergency); the follow-up specs and evidence are per batch
        self.a_dir, self.r_dir, self.f_dir = (shard_dir(out, sk, p) for p in ("A", "R", "F"))
        self.r_spec = os.path.join(self.base, "R.tsv")
        self.f_spec = os.path.join(self.base, "F.tsv")
        self.check_dir = os.path.join(self.base, "check")
        self.x86_dir = os.path.join(self.base, "x86")
        self.record = os.path.join(self.base, "BATCH.json")
        self.rows = L.read_spec(self.spec)

    def judged(self, d: str) -> list:
        rows = {r["tag"]: r for r in self.rows}
        for p in (self.r_spec, self.f_spec):
            for r in C.read_spec_or_none(p) or []:
                rows[r["tag"]] = r
        return effective(L.read_judged(os.path.join(d, "judged.tsv")), rows, d)

    def first(self) -> dict:
        by = L.collapse_runs(self.judged(self.a_dir))
        return {r["tag"]: by.get((r["tag"], "competition", 1)) for r in self.rows}

    def res(self, tag: str) -> str:
        return C.res_path(self.a_dir, tag, "competition", 1)

    def check_paths(self, tag: str) -> tuple[str, str]:
        return os.path.join(self.check_dir, f"{tag}.check.json"), os.path.join(self.check_dir, f"{tag}.check.log")

    def replay_record(self, tag: str) -> str:
        return os.path.join(self.x86_dir, tag, "replay.json")


def ledger_of(out: str, plan: dict, rows: list[dict]) -> tuple[dict, str | None]:
    led = (plan.get("certified") or {}).get("ledger")
    if not led:
        return {}, None
    try:
        idx = C.ledger_index(C.parse_ledger(os.path.join(out, led["path"])))
    except C.CertError as e:
        return {}, str(e)
    return {r["tag"]: C.ledger_certificate(idx, r) for r in rows}, None


def route_event(flight: dict | None) -> dict:
    """The run's `prepared_root` flight note: {status: ran|skipped|absent|no-flight, route: RootProven|...|-, reason}."""
    evs = [e for e in ((flight or {}).get("events") or []) if isinstance(e, dict) and e.get("method") == ROUTE_EVENT]
    if flight is None:
        return {"status": "no-flight", "route": "-", "reason": "-"}
    if not evs:
        return {"status": "absent", "route": "-", "reason": "-"}
    e = evs[-1]
    txt = str(e.get("reason") or "")
    m = ROUTE_STATUS_RE.search(txt)
    rr = ROUTE_REASON_RE.search(txt)
    # the flight record spells FlightStatus in lower case (`ran`, `skipped`): measured on the eng16 pin
    return {"status": str(e.get("status") or "?").lower(), "route": m.group(1) if m else "-",
            "reason": rr.group(1) if rr else txt[:80]}


def published(ev: dict) -> bool:
    return str(ev.get("status")).lower() == "ran" and ev.get("route") in ROUTE_PUBLISHED


def _write_spec_once(path: str, want: list[dict]) -> bool:
    if os.path.isfile(path):
        if not C.spec_rows_equal(L.read_spec(path), want):
            raise KCError(f"{path} exists and does not re-derive (phase A or the re-proofs changed under it)")
        return False
    tmp = path + f".tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if want:
        L.write_spec(tmp, want)
    else:
        with open(tmp, "w") as fh:
            fh.write("#" + L.SCHEMA + "\t" + "\t".join(L.SPEC_COLUMNS) + "\n")
    os.replace(tmp, path)
    return True


def cmd_reproof(a) -> int:
    """Classify the batch's phase-A rows and write its re-proof spec (write-once; a resume must re-derive it)."""
    out = os.path.abspath(a.out)
    plan = load_plan(out, verify_files=False, sample=a.sample)
    b = Batch(out, plan, a.shard, a.batch)
    first = b.first()
    ledger, lerr = ledger_of(out, plan, b.rows)
    if lerr:
        raise KCError(f"the plan's ledger does not parse: {lerr}")
    cls = collections.Counter(C.classify(r, first.get(r["tag"]), (ledger.get(r["tag"]) or {}).get("ok", False))
                              for r in b.rows)
    want = C.derive_reproof(b.rows, first, ledger)
    _write_spec_once(b.r_spec, want)
    print("  phase A classes: " + " ".join(f"{k}={v}" for k, v in sorted(cls.items())))
    print(f"SAT={cls.get(C.C_SAT, 0)}")
    print(f"NOTOK={cls.get(C.C_NOTOK, 0)}")
    print(f"EMERG={cls.get(C.C_EMERGENCY, 0)}")
    print(f"SPEC={b.r_spec}")
    print(f"ROWS={len(want)}")
    print(f"M0OUT={b.r_dir}")
    return 3 if cls.get(C.C_EMERGENCY) else 0


def sat_rows(b: Batch) -> list[tuple[dict, str]]:
    first = b.first()
    return [(r, b.res(r["tag"])) for r in b.rows if C.classify(r, first.get(r["tag"]), False) == C.C_SAT]


def cmd_replay(a) -> int:
    """Every phase-A sat of the batch through the x86 rig (run under `ny_box.sh build --`): the batch's own controls
    first (a PASS and a FAIL per checker kind), then each witness; X86_REPLAY_DONE rc=0|3|5."""
    import x86_replay as X
    out = os.path.abspath(a.out)
    plan = load_plan(out, verify_files=False, sample=a.sample)
    b = Batch(out, plan, a.shard, a.batch)
    rig = plan["certified"]["x86"]
    todo, kept_fail = [], []
    for r, res in sat_rows(b):
        st = C.verify_replay_record(b.replay_record(r["tag"]), r, res, rig)["state"]
        if st in (C.X_PASS, C.X_FAIL):
            print(f"  replay {r['tag']}: {st} (record kept)")
            if st == C.X_FAIL:
                kept_fail.append(r["tag"])
            continue
        todo.append((r, res))
    if kept_fail:
        print(f"X86_REPLAY_DONE rc=3 rows={len(todo)} passed=0 failed={len(kept_fail)} inconclusive=0 controls=kept "
              f"stopped_at={kept_fail[0]}")
        return 3
    print(f"=== x86 replay {b.sk}/{b.id}: {len(todo)} row(s) {now()}")
    if not todo:
        print("X86_REPLAY_DONE rc=0 rows=0 passed=0 failed=0 inconclusive=0 controls=skipped")
        return 0
    kinds = C.control_kinds(rig)
    unc = sorted({C.checker_kind(r) for r, _ in todo} - kinds)
    bad = X.rig_check(rig) + [f"no PASS+FAIL control of kind {k}" for k in unc]
    if bad:
        for x in bad:
            print(f"  RIG-BAD {x}")
        print(f"X86_REPLAY_DONE rc=5 rows={len(todo)} passed=0 failed=0 inconclusive={len(todo)} controls=rig-bad")
        return 5
    batch = f"{b.id}-{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    controls = X.run_controls(rig, b.x86_dir, batch)
    cdoc = C._json(controls) or {}
    print(f"  controls {controls}: ok={cdoc.get('ok')} " + " ".join(
        f"{c['name']}={c['got']}(want {c['expect']}, {c['result']})" for c in cdoc.get("controls", [])))
    if cdoc.get("ok") is not True:
        print(f"X86_REPLAY_DONE rc=5 rows={len(todo)} passed=0 failed=0 inconclusive={len(todo)} controls=fail")
        return 5
    n = collections.Counter()
    sha16 = S.sha256_text(open(os.path.join(out, "plan.json")).read())[:16]
    for r, res in todo:
        X.replay_row(rig, r, res, os.path.dirname(b.replay_record(r["tag"])), controls, sha16)
        v = C.verify_replay_record(b.replay_record(r["tag"]), r, res, rig)
        n[v["state"]] += 1
        print(f"  replay {r['tag']} [{C.checker_kind(r)}]: {v['state']} result={v.get('result')} ({v['why'][:160]})")
        if v["state"] == C.X_FAIL:
            print(f"  ###### WRONG SAT: {r['tag']} -- the organisers' checker rejects the witness on x86 ######")
            print(f"X86_REPLAY_DONE rc=3 rows={len(todo)} passed={n[C.X_PASS]} failed={n[C.X_FAIL]} "
                  f"inconclusive={n[C.X_INCONCLUSIVE]} controls=ok stopped_at={r['tag']}")
            return 3
    rc = 0 if n[C.X_PASS] == len(todo) else 5
    print(f"X86_REPLAY_DONE rc={rc} rows={len(todo)} passed={n[C.X_PASS]} failed={n[C.X_FAIL]} "
          f"inconclusive={n[C.X_INCONCLUSIVE] + n[C.X_PENDING]} controls=ok")
    return rc


def reproof_rows(b: Batch) -> list[dict]:
    return C.read_spec_or_none(b.r_spec) or []


def cmd_check_list(a) -> int:
    """(tag, bundle, onnx, vnnlib, report, log) per re-proof whose Proof run is OK + unsat, with a bundle, unchecked."""
    out = os.path.abspath(a.out)
    plan = load_plan(out, verify_files=False, sample=a.sample)
    b = Batch(out, plan, a.shard, a.batch)
    by = L.collapse_runs(b.judged(b.r_dir))
    for r in reproof_rows(b):
        j = by.get((r["tag"], "proof", 1))
        if j is None or j.status != "OK" or j.verdict != "unsat":
            continue
        bnd = C.bundle_path(C.res_path(b.r_dir, r["tag"], "proof", 1))
        rep, logp = b.check_paths(r["tag"])
        if not os.path.isfile(bnd) or C.check_log_facts(C._read(logp)).get("ran"):
            continue
        print("\t".join((r["tag"], bnd, L.model_files(r["onnx"])[0], r["vnnlib"], rep, logp)))
    return 0


def reproof_outcomes(b: Batch, commit: str) -> dict[str, dict]:
    """{S8R_ tag: certlib.reproof_outcome + the route-B2 rule + facts} for every re-proof row on disk."""
    by = L.collapse_runs(b.judged(b.r_dir))
    out = {}
    for r in reproof_rows(b):
        jp = by.get((r["tag"], "proof", 1))
        res = C.res_path(b.r_dir, r["tag"], "proof", 1)
        bnd = C.bundle_path(res)
        rep, logp = b.check_paths(r["tag"])
        log = C.check_log_facts(C._read(logp))
        f = {"bundle": bnd, "bundle_exists": os.path.isfile(bnd), "report": rep, "log_path": logp, "log": log}
        o_sha, v_sha, why = C.row_shas(r)
        if why:
            f["cert"] = {"state": "reject", "why": why}
        elif log.get("ran"):
            f["cert"] = C.check_certificate(log, rep, bnd, o_sha, v_sha, commit)
        o = dict(C.reproof_outcome(jp, f))
        ev = route_event(C._json(res + ".flight.json")) if jp is not None else {"status": "not-run"}
        o["route"] = ev
        o["facts"] = {k: v for k, v in f.items() if k != "cert"}
        o["cert_facts"] = f.get("cert")
        o["proof_verdict"] = jp.verdict if jp else None
        o["proof_status"] = jp.status if jp else None
        if published(ev) and o["state"] in (C.R_FALLBACK,):
            o = dict(o, state=C.R_EMERGENCY, why=f"ROUTE-B2: the Proof run's prepared_root PUBLISHED "
                                                 f"({ev.get('route')}) but the certificate did not ACCEPT: {o['why']}")
        out[r["tag"]] = o
    return out


def cmd_scan(a) -> int:
    """The batch's re-proof outcomes after the checks; writes the classic spec once every outcome is final.
    Exit 3 on an EMERGENCY (refuted, a Proof sat, a Proof-run emergency, a route-B2 breach)."""
    out = os.path.abspath(a.out)
    plan = load_plan(out, verify_files=False, sample=a.sample)
    b = Batch(out, plan, a.shard, a.batch)
    rr = reproof_rows(b)
    oc = reproof_outcomes(b, plan["pin"]["commit"])
    n = collections.Counter(o["state"] for o in oc.values())
    for t, o in oc.items():
        print(f"  {o['state'].upper():9s} {t}: proof={o.get('proof_verdict')} route={o['route'].get('status')}/"
              f"{o['route'].get('route')} {o['why'][:200]}")
    print("CERT_SCAN " + " ".join(f"{k}={n.get(k, 0)}" for k in (C.R_ACCEPT, C.R_FALLBACK, C.R_PENDING, C.R_EMERGENCY)))
    if n.get(C.R_EMERGENCY):
        return 3
    if n.get(C.R_PENDING):
        print(f"FSPEC={b.f_spec}")
        print("FROWS=pending")
        return 5
    want = C.derive_classic(rr, {t: {"state": o["state"]} for t, o in oc.items()})
    _write_spec_once(b.f_spec, want)
    print(f"FSPEC={b.f_spec}")
    print(f"FROWS={len(want)}")
    print(f"FM0OUT={b.f_dir}")
    return 0


# ---------------------------------------------------------------------------------------------------------
# judging: one batch, and the cumulative summary
# ---------------------------------------------------------------------------------------------------------


def all_runs(out: str, plan: dict) -> tuple[dict, list]:
    """(every spec row of every batch's phase A + follow-ups, every judged line) on disk: the global cross-tag check."""
    rows: dict = {}
    lines: list = []
    for sk, sh in plan["shards"].items():
        for ph in ("A", "R", "F"):
            lines += L.read_judged(os.path.join(shard_dir(out, sk, ph), "judged.tsv"))
        # a sample's effective plan: EVERY batch of the full plan (a run of a row the sample did not draw, e.g. one
        # measured before the re-plan, must still meet its instance's other tags) + the sample's own batches
        seen, every = set(), []
        for bb in list((plan.get("base_shards") or {}).get(sk, {}).get("batches") or []) + list(sh["batches"]):
            if bb["id"] not in seen:
                seen.add(bb["id"])
                every.append(bb)
        for bb in every:
            base = os.path.join(out, "run", sk, bb["id"])
            for spec in (os.path.join(out, bb["spec"]), os.path.join(base, "R.tsv"), os.path.join(base, "F.tsv")):
                if os.path.isfile(spec):
                    for r in L.read_spec(spec):
                        rows[r["tag"]] = r
    return rows, lines


def judge_batch(out: str, plan: dict, sk: str, bid: str) -> dict:
    b = Batch(out, plan, sk, bid)
    rig = plan["certified"]["x86"]
    commit = plan["pin"]["commit"]
    a_lines, r_lines, f_lines = b.judged(b.a_dir), b.judged(b.r_dir), b.judged(b.f_dir)
    by = L.collapse_runs(a_lines + r_lines + f_lines)
    first = {r["tag"]: by.get((r["tag"], "competition", 1)) for r in b.rows}
    ledger, lerr = ledger_of(out, plan, b.rows)
    emerg, void, lost = [], [], []
    if lerr:
        void.append(f"ledger: {lerr}")
    rr_spec = C.read_spec_or_none(b.r_spec)
    f_spec = C.read_spec_or_none(b.f_spec)
    want_r = C.derive_reproof(b.rows, first, ledger)
    if rr_spec is not None and not C.spec_rows_equal(rr_spec, want_r):
        void.append("the re-proof spec does not re-derive from phase A")
    oc = reproof_outcomes(b, commit) if rr_spec is not None else {}
    stage_of = plan["rows_index"]
    per = []
    counts = collections.Counter()
    for r in b.rows:
        t = r["tag"]
        j = first.get(t)
        led = ledger.get(t) or {}
        cls = C.classify(r, j, led.get("ok", False))
        counts[cls] += 1
        ev = route_event(C._json(b.res(t) + ".flight.json")) if j is not None else {"status": "not-run",
                                                                                  "route": "-", "reason": "-"}
        rec = {"tag": t, "stage": stage_of.get(t, {}).get("stage"), "year": r["year"], "category": r["category"],
               "index": r["index"], "budget": r["budget"], "truth": r["truth"], "bank": r["bank"], "class": cls,
               "verdict": j.verdict if j else None, "status": j.status if j else "not-run",
               "outcome": j.outcome if j else None, "wall": j.wall if j else None,
               "reasons": j.reasons if j else [], "route": ev}
        counts[f"route_{ev['status']}{'_' + ev['route'] if ev.get('route') not in (None, '-') else ''}"] += 1
        if j is None or j.status == "VOID":
            void.append(f"{t}: phase-A run {'VOID: ' + ';'.join(j.reasons)[:160] if j else 'not run'}")
        if j is not None and j.status == "EMERGENCY":
            emerg.append(f"{t}: phase A: {';'.join(j.reasons)[:240]}")
        if cls == C.C_LOST:
            lost.append(f"{t}: bank={r['bank']} now={j.verdict if j else '-'}")
        if cls == C.C_UNSAT_LEDGER:
            rec["ledger"] = led.get("entry")
        if cls == C.C_UNSAT_REPROVE:
            rt = C.followup_tag(C.REPROOF_PREFIX, t)
            o = oc.get(rt)
            if o is None:
                void.append(f"{t}: re-proof not derived / not run")
                rec["reproof"] = {"state": "pending", "why": "not derived"}
            else:
                cf = o.get("cert_facts") or {}
                rec["reproof"] = {"tag": rt, "state": o["state"], "why": o["why"][:300],
                                  "proof_verdict": o.get("proof_verdict"), "proof_status": o.get("proof_status"),
                                  "route": o.get("route"), "checker_verdict": cf.get("verdict"),
                                  "certificate": cf.get("state"), "nodes": cf.get("nodes"),
                                  "report": o["facts"].get("report") if o["facts"].get("log", {}).get("ran") else None,
                                  "bundle_exists": o["facts"].get("bundle_exists")}
                counts[f"reproof_{o['state']}"] += 1
                jp = by.get((rt, "proof", 1))
                if jp is not None and jp.status == "OK" and jp.outcome == "lost":
                    lost.append(f"{rt}: Proof re-run lost banked {r['bank']} ({jp.verdict})")
                if o["state"] == C.R_EMERGENCY:
                    emerg.append(f"{rt}: {o['why'][:300]}")
                elif o["state"] == C.R_PENDING:
                    void.append(f"{rt}: {o['why'][:160]}")
                elif o["state"] == C.R_FALLBACK:
                    ft = C.followup_tag(C.CLASSIC_PREFIX, t)
                    fr = [x for x in (f_spec or []) if x["tag"] == ft]
                    if not fr:
                        void.append(f"{ft}: classic sub-moat not derived / not run")
                        rec["classic"] = {"state": "pending"}
                    else:
                        runs = [by.get((ft, p, k)) for p, k in L_schedule(fr[0])]
                        st = ("emergency" if any(x is not None and x.status == "EMERGENCY" for x in runs) else
                              "pending" if any(x is None or x.status != "OK" for x in runs) else "covered")
                        if st == "pending":
                            void.append(f"{ft}: classic sub-moat runs not OK or not run")
                        if st == "emergency":
                            emerg.append(f"{ft}: classic sub-moat EMERGENCY")
                        for x in runs:
                            if x is not None and x.status == "OK" and x.outcome == "lost":
                                lost.append(f"{ft}/{x.profile}/r{x.rep}: lost banked {r['bank']} ({x.verdict})")
                        for status, reason in L.judge_profiles(fr[0], [x for x in f_lines if x.tag == ft]):
                            if status == "EMERGENCY":
                                emerg.append(f"{ft}: {reason}")
                        rec["classic"] = {"tag": ft, "state": st, "verdicts": [
                            f"{p}/r{k}={x.verdict if x else '-'}{'' if x is None or x.status == 'OK' else '(' + x.status + ')'}"
                            for (p, k), x in zip(L_schedule(fr[0]), runs)]}
                        counts[f"classic_{st}"] += 1
        if cls == C.C_SAT:
            x = C.verify_replay_record(b.replay_record(t), r, b.res(t), rig)
            rec["replay"] = {"state": x["state"], "why": x["why"][:300], "result": x.get("result"),
                             "checker": C.checker_kind(r)}
            counts[f"replay_{x['state']}"] += 1
            if x["state"] == C.X_FAIL:
                emerg.append(f"{t}: WRONG SAT: {x['why'][:200]}")
            elif x["state"] in (C.X_PENDING, C.X_INCONCLUSIVE):
                void.append(f"{t}: x86 replay {x['state']}: {x['why'][:160]}")
        per.append(rec)
    if f_spec is not None and rr_spec is not None:
        if all(o["state"] in (C.R_ACCEPT, C.R_FALLBACK) for o in oc.values()):
            want_f = C.derive_classic(rr_spec, {t: {"state": o["state"]} for t, o in oc.items()})
            if not C.spec_rows_equal(f_spec, want_f):
                void.append("the classic spec does not re-derive from the re-proof outcomes")
        elif not any(o["state"] == C.R_EMERGENCY for o in oc.values()):
            void.append("a classic spec exists while re-proofs are still pending")
    # m0's own per-spec cross-run rules (phase A rows have one run each; the follow-ups carry the profile rules)
    for spec_rows, lines in ((b.rows, a_lines), (rr_spec or [], r_lines)):
        for r in spec_rows:
            for status, reason in L.judge_profiles(r, [x for x in lines if x.tag == r["tag"]]):
                if status == "EMERGENCY":
                    emerg.append(f"{r['tag']}: {reason}")
    # the GLOBAL cross-tag instance check over every batch judged so far (an instance can sit in two batches: a
    # 2025 and a 2026 row, a moat and an out-of-moat row, phase A and its follow-ups)
    rows_all, lines_all = all_runs(out, plan)
    for status, reason in L.judge_instances(rows_all, lines_all):
        if status == "EMERGENCY":
            emerg.append(f"GLOBAL {reason}")
    emerg = list(dict.fromkeys(emerg))
    status = "EMERGENCY" if emerg else "VOID" if void else "PARTIAL" if lost else "PASS"
    rc = {"PASS": 0, "PARTIAL": 4, "VOID": 5, "EMERGENCY": 3}[status]
    return {"schema": BATCH_SCHEMA, "batch": bid, "shard": sk, "judged": now(), "status": status, "rc": rc,
            "final": not void, "counts": dict(sorted(counts.items())), "emergencies": emerg, "void": void,
            "lost": lost, "rows": per,
            "spec_sha256": L.sha256_file(b.spec)}


def L_schedule(row: dict) -> list[tuple[str, int]]:
    return C.M0J.schedule(row)


def cmd_judge_batch(a) -> int:
    out = os.path.abspath(a.out)
    plan = load_plan(out, verify_files=False, sample=a.sample)
    rec = judge_batch(out, plan, a.shard, a.batch)
    b = Batch(out, plan, a.shard, a.batch)
    S.write_json_atomic(b.record, rec)
    for e in rec["emergencies"]:
        print(f"  EMERGENCY {e}")
    for v in rec["void"][:20]:
        print(f"  VOID {v}")
    for x in rec["lost"]:
        print(f"  LOST {x}")
    for r in rec["rows"]:
        extra = ""
        if "reproof" in r:
            rp = r["reproof"]
            extra += f" reproof={rp.get('state')}(proof={rp.get('proof_verdict')} checker={rp.get('checker_verdict')})"
        if "classic" in r:
            extra += f" classic={r['classic'].get('state')}{r['classic'].get('verdicts', '')}"
        if "replay" in r:
            extra += f" replay={r['replay'].get('state')}({r['replay'].get('result')})"
        print(f"    {r['tag']:44s} {r['stage']:4s} bank={r['bank']:7s} truth={r['truth']:5s} A={r['verdict']}"
              f"({r['status']},{r['outcome']},wall={r['wall']}) route={r['route'].get('status')}/"
              f"{r['route'].get('route')}{extra}")
    c = rec["counts"]
    print(f"BATCH_DONE rc={rec['rc']} status={rec['status']} batch={a.batch} shard={a.shard} rows={len(rec['rows'])} "
          f"emergency={len(rec['emergencies'])} void={len(rec['void'])} lost={len(rec['lost'])} "
          f"unsat={c.get(C.C_UNSAT_REPROVE, 0) + c.get(C.C_UNSAT_LEDGER, 0)} sat={c.get(C.C_SAT, 0)} "
          f"accepted={c.get('reproof_accepted', 0)} fallback={c.get('reproof_fallback', 0)} "
          f"replayed={c.get('replay_passed', 0)} route_proven="
          f"{sum(v for k, v in c.items() if k in ('route_ran_RootProven', 'route_ran_TreeProven'))}")
    return rec["rc"]


def cmd_summary(a) -> int:
    out = os.path.abspath(a.out)
    plan = load_plan(out, verify_files=False, sample=a.sample)
    tot = collections.Counter()
    emerg, lost, new_unsat, sats, voids, routes = [], [], [], [], [], collections.Counter()
    nb = 0
    for sk, sh in plan["shards"].items():
        for bb in sh["batches"]:
            p = os.path.join(out, "run", sk, bb["id"], "BATCH.json")
            if not os.path.isfile(p):
                continue
            rec = json.load(open(p))
            nb += 1
            tot[rec["status"]] += 1
            emerg += [f"{bb['id']}: {e}" for e in rec["emergencies"]]
            lost += [f"{bb['id']}: {x}" for x in rec["lost"]]
            voids += [f"{bb['id']}: {x}" for x in rec["void"]]
            for r in rec["rows"]:
                tot["rows"] += 1
                tot[f"A_{r['status']}"] += 1
                routes[f"{r['route'].get('status')}/{r['route'].get('route')}"] += 1
                if r["verdict"] == "unsat" and r["bank"] != "unsat":
                    new_unsat.append(r)
                if r["verdict"] == "sat":
                    sats.append(r)
                if r["verdict"] in ("sat", "unsat") and r["status"] == "OK":
                    tot[f"verdict_{r['verdict']}"] += 1
                tot[f"outcome_{r['outcome']}"] += 1
                if "reproof" in r:
                    tot[f"reproof_{r['reproof']['state']}"] += 1
                    tot[f"checker_{r['reproof'].get('checker_verdict')}"] += 1
                if "replay" in r:
                    tot[f"replay_{r['replay']['state']}"] += 1
    print(f"=== keyless certified summary {out}: {nb} batch(es) judged, {tot['rows']} row(s)")
    print("  batches: " + " ".join(f"{k}={tot[k]}" for k in ("PASS", "PARTIAL", "VOID", "EMERGENCY")))
    print("  counts:  " + " ".join(f"{k}={v}" for k, v in sorted(tot.items()) if k not in
                                  ("PASS", "PARTIAL", "VOID", "EMERGENCY", "rows")))
    print("  route:   " + " ".join(f"{k}={v}" for k, v in routes.most_common()))
    for r in new_unsat:
        rp = r.get("reproof") or {}
        print(f"  NEW-UNSAT {r['tag']} bank={r['bank']} truth={r['truth']} route={r['route'].get('route')} "
              f"reproof={rp.get('state')} proof={rp.get('proof_verdict')} checker={rp.get('checker_verdict')} "
              f"classic={(r.get('classic') or {}).get('state')}")
    for r in sats:
        rp = r.get("replay") or {}
        print(f"  SAT {r['tag']} bank={r['bank']} truth={r['truth']} replay={rp.get('state')} result={rp.get('result')} "
              f"[{rp.get('checker')}]")
    for e in emerg:
        print(f"  EMERGENCY {e}")
    for x in lost:
        print(f"  LOST {x}")
    for v in voids[:30]:
        print(f"  VOID {v}")
    rc = 3 if emerg else 5 if voids else 4 if lost else 0
    so, cl = sample_report(out, plan)
    for x in so:
        print(x)
    extra = "" if cl is None else (f" sample={plan['sample']['name']} sample_rows={cl['rows_judged']} "
                                   f"strata_covered={cl['strata_covered']}/{cl['strata']}"
                                   + (f" bound95_rows={cl['bound_rows']:.4f} bound95_census={cl['bound_census_minimax']:.4f}"
                                      if cl["events"] == 0 else " bound95=none"))
    print(f"KC_SUMMARY rc={rc} batches={nb} rows={tot['rows']} emergency={len(emerg)} void={len(voids)} "
          f"lost={len(lost)} new_unsat={len(new_unsat)} sat={len(sats)} accepted={tot['reproof_accepted']} "
          f"fallback={tot['reproof_fallback']} replayed={tot['replay_passed']}{extra}")
    return rc


# ---------------------------------------------------------------------------------------------------------
# AMENDMENT 2: the stratified SAMPLE of the rows not yet judged
# ---------------------------------------------------------------------------------------------------------


def predicted_class(r: dict) -> str:
    """The phase-A class a row is EXPECTED to take, fixed before it runs (its stratum): the bank's decided verdict
    (unsat -> unsat-reprove, sat -> sat-replay), else undecided (banked timeout / unknown, or unbanked). A judged row
    stays in this PRE-RUN stratum whatever it then returns (the strata must partition the census independently of
    the outcome); its observed class is reported beside it."""
    return {"unsat": C.C_UNSAT_REPROVE, "sat": C.C_SAT}.get(r["bank"], C.C_UNDECIDED)


def stratum_of(r: dict) -> str:
    return f"{r['year']}/{r['category']}/{predicted_class(r)}"


def unit_of_stratum(h: str) -> str:
    return h.rsplit("/", 1)[0]


def stratum_sort_key(h: str) -> tuple:
    y, cat, cls = h.split("/")
    return (-int(y), cat, cls)


def plan_rows(out: str, plan: dict, sk: str) -> list[tuple[str, dict]]:
    """(plan batch id, spec row) for every row of the FULL plan's shard, in plan order."""
    shards = plan.get("base_shards") or plan["shards"]
    if sk not in shards:
        raise KCError(f"shard {sk} is not in the plan ({sorted(shards)})")
    return [(bb["id"], r) for bb in shards[sk]["batches"] for r in L.read_spec(os.path.join(out, bb["spec"]))]


def engagement_evidence(out: str, sk: str, batch_ids: list[str]) -> dict[str, list[str]]:
    """{unit `year/category`: [evidence]}: every judged row of the given batches whose phase-A run or Proof re-run
    flight `prepared_root` event is Ran (any route outcome) -- the route actually engaged on that unit."""
    ev: dict = collections.defaultdict(list)
    for bid in batch_ids:
        p = os.path.join(out, "run", sk, bid, "BATCH.json")
        if not os.path.isfile(p):
            continue
        rec = json.load(open(p))
        for r in rec.get("rows") or []:
            for what, route in (("A", r.get("route") or {}), ("R", (r.get("reproof") or {}).get("route") or {})):
                if str(route.get("status")).lower() == "ran":
                    ev[f"{r['year']}/{r['category']}"].append(f"{bid}:{r['tag']}:{what}=ran/{route.get('route')}")
    return {k: v for k, v in sorted(ev.items())}


def stratum_weight(h: str, engaged_units: set, engaged_weight: float) -> float:
    """engaged_weight on a stratum of an engaged unit whose class can reach the route; 1 otherwise. A sat-replay
    stratum is never weighted: its rows are decided by the upfront attack before the route (measured: every sat row
    of the first judged batches has `prepared_root` absent or skipped)."""
    return float(engaged_weight) if unit_of_stratum(h) in engaged_units and not h.endswith("/" + C.C_SAT) else 1.0


def allocate(sizes: dict, judged: dict, weights: dict, floor: int, target_new: int) -> dict:
    """{stratum: NEW rows to draw}. First every stratum is topped up to min(floor, its size) (rows already judged
    count; the floor wins even past target_new); the rest of target_new is handed out one row at a time by the
    Sainte-Lague divisor rule on the weighted population (size x weight) with the judged + already-allocated rows
    pre-seated, so each stratum's final total tracks its weighted proportional share of (judged + target_new) and no
    stratum exceeds its size. Deterministic (ties: the stratum key)."""
    keys = sorted(sizes, key=stratum_sort_key)
    cap = {h: sizes[h] - judged.get(h, 0) for h in keys}
    bad = [h for h in keys if cap[h] < 0]
    if bad:
        raise KCError(f"allocate: judged rows exceed the stratum size in {bad[:4]}")
    new = {h: min(cap[h], max(0, min(floor, sizes[h]) - judged.get(h, 0))) for h in keys}
    seats = {h: judged.get(h, 0) + new[h] for h in keys}
    for _ in range(max(0, target_new - sum(new.values()))):
        room = [h for h in keys if new[h] < cap[h]]
        if not room:
            break
        h = max(room, key=lambda h: sizes[h] * weights[h] / (2 * seats[h] + 1))  # first max in key order
        new[h] += 1
        seats[h] += 1
    return new


def zero_event_bound(n: int, alpha: float = 0.05) -> float:
    """The exact one-sided (1-alpha) upper bound on a per-row event rate after 0 events in n rows (Clopper-Pearson;
    ~3/n at alpha=0.05)."""
    return 1.0 - alpha ** (1.0 / n) if n > 0 else 1.0


def stratified_zero_event_bound(strata: list[tuple[float, int, bool]], alpha: float = 0.05) -> float:
    """(W_h census weight, n_h judged rows, exhaustive) per stratum, 0 events everywhere -> the largest census-
    weighted per-row event rate p = sum W_h p_h for which SOME placement of the events across the strata still leaves
    P(0 events in the sample) = prod (1-p_h)^n_h >= alpha. That worst case puts the events where the sample is
    thinnest (the KKT solution p_h = max(0, 1 - n_h / (kappa W_h))), so the bound holds whatever the strata's true
    rates are. Binomial (with replacement): conservative for the without-replacement draw. An exhaustively judged
    stratum is known exactly (p_h = 0); an unsampled one can hide events at no cost (p_h = 1)."""
    import math
    base = sum(w for w, n, ex in strata if not ex and n == 0)
    free = [(w, n) for w, n, ex in strata if not ex and n > 0 and w > 0]
    if not free:
        return min(1.0, base)

    def at(kappa: float) -> tuple[float, float]:
        p = lp = 0.0
        for w, n in free:
            ph = max(0.0, 1.0 - n / (kappa * w))
            p += w * ph
            if ph > 0:
                lp += n * math.log(1.0 - ph)
        return p, lp

    la = math.log(alpha)
    lo = min(n / w for w, n in free)
    hi = lo * 2.0
    while at(hi)[1] > la:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if at(mid)[1] > la:
            lo = mid
        else:
            hi = mid
    return min(1.0, base + at(hi)[0])


def kish_n_eff(strata: list[tuple[float, int, bool]]) -> float:
    s = sum(w * w / n for w, n, ex in strata if not ex and n > 0)
    return 1.0 / s if s > 0 else float("inf")


def sample_claim(sizes: dict, n_by: dict, events: int, alpha: float = 0.05) -> dict:
    """The statistical statement a judged sample supports (0 events = no EMERGENCY on any judged row)."""
    N = sum(sizes.values())
    strata = [(sizes[h] / N, n_by.get(h, 0), n_by.get(h, 0) >= sizes[h]) for h in sizes]
    n = sum(n_by.values())
    covered = sum(1 for h in sizes if n_by.get(h, 0) > 0)
    out = {"rows_judged": n, "events": events, "strata": len(sizes), "strata_covered": covered,
           "census_rows": N, "alpha": alpha}
    if events == 0:
        out.update(bound_rows=zero_event_bound(n, alpha), bound_census_minimax=stratified_zero_event_bound(strata, alpha),
                   n_eff_kish=kish_n_eff(strata))
        out["text"] = (f"0 soundness events (EMERGENCY) in {n} judged rows: the per-row event rate of the judged-row "
                       f"distribution is < {out['bound_rows']:.4f} (exact one-sided {1 - alpha:.0%}, ~3/n); the "
                       f"census-weighted rate over all {N} plan rows is < {out['bound_census_minimax']:.4f} at "
                       f"{1 - alpha:.0%} whatever the per-stratum rates (worst-case placement across the "
                       f"{len(sizes)} strata; Kish n_eff {out['n_eff_kish']:.0f}), i.e. fewer than "
                       f"{out['bound_census_minimax'] * N:.0f} wrong rows expected in the census")
    else:
        out["text"] = f"{events} soundness event(s) in {n} judged rows: NO bound -- every event is an EMERGENCY"
    return out


def cmd_sample(a) -> int:
    """Draw the stratified sample (AMENDMENT 2) of the plan rows outside the judged batches and write it once."""
    out = os.path.abspath(a.out)
    plan = load_plan(out)
    plan_sha = S.sha256_text(open(os.path.join(out, "plan.json")).read())
    sk = a.shard
    if a.floor < 0 or a.target_new < 0 or a.engaged_weight < 1 or a.batch_rows < 1:
        raise KCError("--floor/--target-new must be >= 0, --engaged-weight and --batch-rows >= 1")
    spath = os.path.join(out, f"{a.name}.json")
    if os.path.exists(spath) and not a.dry_run:
        raise KCError(f"{spath} exists: a sample is drawn once (pick another --name for a second one)")
    if os.path.exists(os.path.join(out, "EMERGENCY")):
        raise KCError(f"{out}/EMERGENCY exists: no re-plan past a recorded emergency")
    rows = plan_rows(out, plan, sk)
    base_batches = plan["shards"][sk]["batches"]
    carried, started = [], []
    for bb in base_batches:
        bj = os.path.join(out, "run", sk, bb["id"], "BATCH.json")
        if os.path.isfile(bj):
            rec = json.load(open(bj))
            carried.append({"id": bb["id"], "status": rec.get("status"), "final": bool(rec.get("final")),
                            "rc": rec.get("rc"), "rows": bb["rows"], "batch_json_sha256": L.sha256_file(bj),
                            "emergencies": len(rec.get("emergencies") or []), "lost": rec.get("lost") or []})
        elif os.path.isdir(os.path.join(out, "run", sk, bb["id"])):
            started.append(bb)
    bad = [c["id"] for c in carried if c["emergencies"]]
    if bad:
        raise KCError(f"carried batch(es) {bad} record an EMERGENCY: no re-plan")
    # a batch the full plan's runner started but never judged (stopped for the re-plan): its rows stay in the pool,
    # and any run it measured stays in the shared m0 dirs, read by the global cross-tag check like every other run
    judged_tags = collections.Counter()
    for ph in ("A", "R", "F"):
        for j in L.read_judged(os.path.join(shard_dir(out, sk, ph), "judged.tsv")):
            judged_tags[j.tag] += 1
    started = [{"id": bb["id"], "runs_judged": {r["tag"]: judged_tags[r["tag"]] for r in
                                                L.read_spec(os.path.join(out, bb["spec"])) if judged_tags[r["tag"]]}}
               for bb in started]
    cids = {c["id"] for c in carried}
    judged_rows = [r for bid, r in rows if bid in cids]
    remaining = [r for bid, r in rows if bid not in cids]
    sizes = collections.Counter(stratum_of(r) for _, r in rows)
    judged = collections.Counter(stratum_of(r) for r in judged_rows)
    ev = engagement_evidence(out, sk, [c["id"] for c in carried])
    weights = {h: stratum_weight(h, set(ev), a.engaged_weight) for h in sizes}
    new = allocate(dict(sizes), dict(judged), weights, a.floor, a.target_new)
    seed = a.seed or f"kcs|{plan['capability']}|{plan['pin']['sha256'][:16]}|{a.name}"
    pool: dict = collections.defaultdict(list)
    for r in remaining:
        pool[stratum_of(r)].append(r)
    # fail closed on runs measured before the re-plan: a row of a started-but-unjudged batch with a judged phase-A
    # run is FORCED into the sample (first in its stratum, even past its allocation), so a measured sat is replayed
    # and a measured unsat re-proved instead of sitting unjudged in the shared m0 dirs
    forced = {t for st in started for t in st["runs_judged"]}
    drawn: dict = {}
    for h in sorted(sizes, key=stratum_sort_key):
        def rank(r):
            return (r["tag"] not in forced, hashlib.sha256(f"{seed}|{r['tag']}".encode()).hexdigest())
        ranked = sorted(pool.get(h, []), key=rank)
        nf = sum(1 for r in ranked if r["tag"] in forced)
        new[h] = max(new[h], nf)
        drawn[h] = ranked[:new[h]]
        if len(drawn[h]) != new[h]:
            raise KCError(f"stratum {h}: allocated {new[h]} but only {len(ranked)} rows remain")
    ordered = S.round_robin([drawn[h] for h in sorted(drawn, key=stratum_sort_key)])
    # the m0 runner keys every run by tag: a drawn row must be the plan's row byte for byte
    by_tag = {r["tag"]: r for _, r in rows}
    if len({r["tag"] for r in ordered}) != len(ordered) or any(not C.spec_rows_equal([r], [by_tag[r["tag"]]])
                                                               for r in ordered):
        raise KCError("a drawn row is duplicated or differs from the plan's row")
    stage_of = plan["rows_index"]
    batches, files = [], {}
    for i in range(0, len(ordered), a.batch_rows):
        bid = f"{a.prefix}{i // a.batch_rows + 1:04d}"
        if any(bid == bb["id"] for bb in base_batches):
            raise KCError(f"sample batch id {bid} collides with a plan batch id (pick another --prefix)")
        rel = os.path.join("specs", sk, f"{a.name}_{bid}.tsv")
        chunk = ordered[i:i + a.batch_rows]
        if os.path.exists(os.path.join(out, rel)) or os.path.exists(os.path.join(out, "run", sk, bid)):
            if not a.dry_run:
                raise KCError(f"{rel} or run/{sk}/{bid} exists: refusing to reuse a batch name")
        batches.append({"id": bid, "spec": rel, "rows": len(chunk),
                        "stages": dict(collections.Counter(stage_of[r["tag"]]["stage"] for r in chunk)),
                        "strata": dict(collections.Counter(stratum_of(r) for r in chunk)), "_rows": chunk})
    N = sum(sizes.values())
    strata = {}
    for h in sorted(sizes, key=stratum_sort_key):
        n_h = judged.get(h, 0) + new[h]
        strata[h] = {"N": sizes[h], "W": round(sizes[h] / N, 6), "weight": weights[h],
                     "engaged": unit_of_stratum(h) in ev, "judged": judged.get(h, 0), "new": new[h], "n": n_h,
                     "bound95": round(zero_event_bound(n_h), 4) if n_h < sizes[h] else 0.0}
    planned = sample_claim(dict(sizes), {h: v["n"] for h, v in strata.items()}, 0)
    walls = walls_of(a.ny_root, ordered)
    rec = {
        "schema": SAMPLE_SCHEMA, "name": a.name, "created": now(), "out": out, "capability": plan["capability"],
        "base_plan_sha256": plan_sha, "pin": {k: plan["pin"][k] for k in ("path", "sha256", "commit", "features")},
        "seed": seed,
        "params": {"target_new": a.target_new, "floor": a.floor, "engaged_weight": a.engaged_weight,
                   "batch_rows": a.batch_rows, "prefix": a.prefix,
                   "allocation": "floor top-up first (judged rows count), then Sainte-Lague divisor rule on size x "
                                 "weight with the judged rows pre-seated; capped by stratum size",
                   "order": "round-robin over strata (year desc, category, class), seeded order inside a stratum "
                            "(sha256(seed|tag)): an early stop still covers every stratum"},
        "stratum_rule": "(year, category, bank-predicted phase-A class: bank unsat -> unsat-reprove, bank sat -> "
                        "sat-replay, otherwise undecided)",
        "engaged": {"rule": "a unit (year, category) is engaged when a judged phase-A run or Proof re-run of a "
                            "carried batch has its flight `prepared_root` event Ran; its non-sat strata get weight "
                            f"{a.engaged_weight}", "units": sorted(ev), "evidence": ev},
        "carried": carried, "started_not_carried": started, "forced": sorted(forced),
        "strata": strata,
        "shards": {sk: {"carried": [c["id"] for c in carried],
                        "batches": [{k: v for k, v in b.items() if k != "_rows"} for b in batches]}},
        "rows": {r["tag"]: {"stratum": stratum_of(r), "batch": b["id"]} for b in batches for r in b["_rows"]},
        "estimate_new_rows": estimate(ordered, walls),
        "claim_if_zero_events": planned,
        "files": {},
    }
    if not a.dry_run:
        for b in batches:
            p = os.path.join(out, b["spec"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            L.write_spec(p, b["_rows"])
            errs = L.validate_spec(L.read_spec(p), check_files=True)
            if errs:
                raise KCError(f"{p} invalid: {errs[:5]}")
            if not C.spec_rows_equal(L.read_spec(p), b["_rows"]):
                raise KCError(f"{p} does not read back as written")
            files[b["spec"]] = L.sha256_file(p)
        rec["files"] = files
        text = json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n"
        S.write_new(spath, text)
        S.write_new(spath + ".sha256", S.sha256_text(text) + "\n")
    print(render_sample(rec))
    print(f"KC_SAMPLE_DONE rc=0 name={a.name} dry_run={int(a.dry_run)} carried={len(carried)} "
          f"carried_rows={len(judged_rows)} new_rows={len(ordered)} batches={len(batches)} strata={len(strata)} "
          f"engaged_units={len(ev)} seed={seed} bound95_rows={planned['bound_rows']:.4f} "
          f"bound95_census={planned['bound_census_minimax']:.4f}"
          + ("" if a.dry_run else f" sample={spath} sha256={S.sha256_text(text)}"))
    return 0


def render_sample(rec: dict) -> str:
    o = [f"=== keyless certified SAMPLE {rec['name']} of {rec['out']}/plan.json (sha256 {rec['base_plan_sha256'][:16]})",
         f"  seed {rec['seed']}; params {json.dumps({k: v for k, v in rec['params'].items() if k in ('target_new', 'floor', 'engaged_weight', 'batch_rows')})}",
         "  carried " + " ".join(f"{c['id']}={c['status']}" for c in rec["carried"])
         + (f"; started, not carried: {rec['started_not_carried']}" if rec["started_not_carried"] else ""),
         f"  engaged units {rec['engaged']['units']}"]
    o.append(f"  {'stratum':58s} {'N':>5s} {'W':>7s} {'wt':>3s} {'judged':>6s} {'new':>4s} {'n':>4s} {'b95':>6s}")
    for h, v in rec["strata"].items():
        o.append(f"  {h:58s} {v['N']:5d} {v['W']:7.4f} {v['weight']:3.0f} {v['judged']:6d} {v['new']:4d} {v['n']:4d} "
                 f"{v['bound95']:6.3f}")
    tot = {k: sum(v[k] for v in rec["strata"].values()) for k in ("N", "judged", "new", "n")}
    o.append(f"  {'TOTAL':58s} {tot['N']:5d} {'':7s} {'':3s} {tot['judged']:6d} {tot['new']:4d} {tot['n']:4d}")
    for sk, sh in rec["shards"].items():
        o.append(f"  shard {sk}: carried {sh['carried']} + {len(sh['batches'])} sample batch(es) "
                 f"({', '.join(b['id'] + ':' + str(b['rows']) for b in sh['batches'])})")
    e = rec["estimate_new_rows"]
    o.append(f"  estimate (new rows, box time excluding waits): expected_h={e['expected_total_h']} ceiling_h={e['ceiling_h']}")
    o.append(f"  claim if 0 events: {rec['claim_if_zero_events']['text']}")
    return "\n".join(o)


def verify_sample_rows(out: str, plan: dict, sk: str) -> tuple[str | None, int]:
    """(refusal or None, drawn rows) for a sample's effective plan: every sample row is the plan's row byte for byte,
    drawn once, from outside the carried batches; every carried FINAL batch's record is the one the sample was drawn
    against (a re-judged carried batch would change the strata the sample was allocated on)."""
    smp = plan["sample"]
    if sk not in smp["shards"]:
        return f"shard {sk} is not in the sample", 0
    base = {r["tag"]: (bid, r) for bid, r in plan_rows(out, plan, sk)}
    carried = set(smp["shards"][sk]["carried"])
    seen: set = set()
    for bb in smp["shards"][sk]["batches"]:
        for r in L.read_spec(os.path.join(out, bb["spec"])):
            b0 = base.get(r["tag"])
            if b0 is None:
                return f"sample row {r['tag']} is not a plan row", 0
            if not C.spec_rows_equal([r], [b0[1]]):
                return f"sample row {r['tag']} differs from the plan's row", 0
            if r["tag"] in seen:
                return f"sample row {r['tag']} is drawn twice", 0
            if b0[0] in carried:
                return f"sample row {r['tag']} is in carried batch {b0[0]}", 0
            seen.add(r["tag"])
    for c in smp["carried"]:
        bj = os.path.join(out, "run", sk, c["id"], "BATCH.json")
        if c["final"] and (not os.path.isfile(bj) or L.sha256_file(bj) != c["batch_json_sha256"]):
            return f"carried final batch {c['id']}: BATCH.json changed since the sample was drawn", 0
    return None, len(seen)


def sample_report(out: str, plan: dict) -> tuple[list[str], dict | None]:
    """The realized coverage + claim of a sample's effective plan from its FINAL batch records."""
    s = plan.get("sample")
    if not s:
        return [], None
    sizes = {h: v["N"] for h, v in s["strata"].items()}
    n_by: collections.Counter = collections.Counter()
    events = 0
    for sk, sh in plan["shards"].items():
        for bb in sh["batches"]:
            p = os.path.join(out, "run", sk, bb["id"], "BATCH.json")
            if not os.path.isfile(p):
                continue
            rec = json.load(open(p))
            events += len(rec.get("emergencies") or [])
            if rec.get("final"):
                for r in rec["rows"]:
                    n_by[stratum_of(r)] += 1
    cl = sample_claim(sizes, dict(n_by), events)
    o = [f"  sample {s['name']} (sha256 {s['_sha256'][:16]}, seed {s['seed']}): judged n per stratum (planned):"]
    for h, v in s["strata"].items():
        o.append(f"    {h:58s} N={v['N']:5d} n={n_by.get(h, 0):4d} ({v['n']:4d})")
    o.append(f"  claim: {cl['text']}")
    return o, cl


def cmd_batches(a) -> int:
    out = os.path.abspath(a.out)
    plan = load_plan(out, verify_files=False, sample=a.sample)
    sh = plan["shards"].get(a.shard)
    if sh is None:
        raise KCError(f"shard {a.shard} is not in the plan")
    for bb in sh["batches"]:
        print(f"{bb['id']}\t{os.path.join(out, bb['spec'])}\t{bb['rows']}")
    return 0


def cmd_step0(a) -> int:
    """Fail-closed preconditions (no box): the plan and every pre-registered file, the pin, both censuses
    re-verified (trailer, binding to the pin, the corpora RE-DERIVED from the benchmark files now), the shard's spec
    union written for the m0 step 0 (which runs the pin's own resolver per category), the x86 rig."""
    import x86_replay as X
    out = os.path.abspath(a.out)
    if os.path.exists(os.path.join(out, "EMERGENCY")):
        print(f"KC_STEP0_REFUSED {out}/EMERGENCY exists")
        return 3
    plan = load_plan(out, sample=a.sample)
    sh = plan["shards"].get(a.shard)
    if sh is None:
        print(f"KC_STEP0_REFUSED shard {a.shard} is not in the plan ({sorted(plan['shards'])})")
        return 97
    if plan.get("sample"):
        smp = plan["sample"]
        why, nrows = verify_sample_rows(out, plan, a.shard)
        if why:
            print(f"KC_STEP0_REFUSED {why}")
            return 97
        print(f"  sample ok: {smp['_path']} sha256={smp['_sha256'][:16]} seed={smp['seed']} carried="
              f"{smp['shards'][a.shard]['carried']} + {len(smp['shards'][a.shard]['batches'])} batch(es), "
              f"{nrows} drawn row(s), every one a plan row")
        print(f"SAMPLE={smp['_path']}")
    if K.sha256_file(plan["pin"]["path"]) != plan["pin"]["sha256"]:
        print("KC_STEP0_REFUSED the pin's sha256 differs from the plan")
        return 91
    truth = {2026: os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt"),
             2025: os.path.expanduser("~/vnncomp2025_results/SCORING-ZERO-TOL/results.txt")}
    moat, xent, scored, _ = xcorpus("<home>/ny", plan["configs"]["src"], truth)
    for name, ent in (("moat", moat), ("out_of_moat", xent)):
        c = plan["census"][name]
        cc = S.parse_census(os.path.join(c["dir"], "census.tsv"))
        if cc["sha256"] != c["tsv_sha256"]:
            print(f"KC_STEP0_REFUSED the {name} census file changed since the plan")
            return 97
        probs = S.census_binding_problems(cc, plan["pin"]["sha256"], S.sha256_text(S.corpus_text(ent)),
                                          plan["capability"])
        if probs:
            print(f"KC_STEP0_REFUSED the {name} census no longer binds: {probs}")
            return 97
    print(f"  census ok: both bind to the pin and to the corpora re-derived now ({len(moat)} + {len(xent)} "
          f"instances, {len(scored)} scored rows)")
    rows = []
    for bb in sh["batches"]:
        rows += L.read_spec(os.path.join(out, bb["spec"]))
    union = os.path.join(out, f"step0_{a.shard}_union.tsv")
    if os.path.exists(union):
        os.replace(union, union + f".stale.{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}")
    L.write_spec(union, rows)
    rig = plan["certified"]["x86"]
    bad = X.rig_check(rig)
    if bad:
        for x in bad:
            print(f"  RIG-BAD {x}")
        print("KC_STEP0_REFUSED the x86 rig is not the plan's")
        return 97
    print(f"  x86 rig ok ({rig['cpu']}, ORT {rig['onnxruntime']}, controls {sorted(C.control_kinds(rig))})")
    # a negative control for the resolver check: the first row with a preset, against an EMPTY configs dir, must
    # NOT resolve that preset (the m0 step 0 then checks the real dir for every category)
    neg = [r for r in rows if r["preset"] != "-"][:1]
    if neg:
        r = neg[0]
        ed = os.path.join(out, f"step0_{a.shard}_empty_configs")
        os.makedirs(ed, exist_ok=True)
        print(f"RESOLVER_NEG={r['category']}\t{r['onnx']}\t{r['vnnlib']}\t{r['budget']}\t{ed}\t"
              f"{os.path.join(r['configs_dir'], r['preset'])}")
    print(f"UNION={union}")
    print(f"BATCHES={len(sh['batches'])}")
    print(f"ROWS={len(rows)}")
    print(f"PIN={plan['pin']['path']}")
    print(f"PIN_SHA={plan['pin']['sha256']}")
    print(f"PIN_COMMIT={plan['pin']['commit']}")
    print(f"PIN_FEATURES={plan['pin']['features']}")
    print(f"CERT_BUDGET_S={int(plan['certified']['checker']['budget_s'])}")
    print(f"CERT_TIMEOUT_S={int(plan['certified']['checker']['timeout_s'])}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("xcensus")
    p.add_argument("--pin", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--box")
    p.add_argument("--configs", default=os.path.join(REPO, "configs"))
    p.add_argument("--ny-root", default="<home>/ny")
    p.add_argument("--truth-2026", default=os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt"))
    p.add_argument("--truth-2025", default=os.path.expanduser("~/vnncomp2025_results/SCORING-ZERO-TOL/results.txt"))
    p = sub.add_parser("plan")
    p.add_argument("--cap", required=True)
    p.add_argument("--census-dir", required=True)
    p.add_argument("--xcensus-dir", required=True)
    p.add_argument("--pin", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-rows", type=int, default=20)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--checker-budget-s", type=float, default=S.DEFAULTS["checker_budget_s"])
    p.add_argument("--cert-ledger")
    p.add_argument("--seed")
    p.add_argument("--smoke-tags", help="a SMOKE plan: exactly these row tags (never evidence)")
    p.add_argument("--registry", default=os.path.join(REPO, "docs", "engine_program_capabilities.yaml"))
    p.add_argument("--repo", default=REPO)
    p.add_argument("--ny-root", default="<home>/ny")
    p.add_argument("--truth-2026", default=os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt"))
    p.add_argument("--truth-2025", default=os.path.expanduser("~/vnncomp2025_results/SCORING-ZERO-TOL/results.txt"))
    p = sub.add_parser("sample")
    p.add_argument("--out", required=True)
    p.add_argument("--shard", default="1of1")
    p.add_argument("--name", default="sample1")
    p.add_argument("--prefix", default="s", help="the sample batch ids' prefix (s0001, ...)")
    p.add_argument("--target-new", type=int, default=240)
    p.add_argument("--floor", type=int, default=3)
    p.add_argument("--engaged-weight", type=float, default=3.0)
    p.add_argument("--batch-rows", type=int, default=20)
    p.add_argument("--seed")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ny-root", default="<home>/ny")
    sample_help = "AMENDMENT 2: the sample record (<out>/<name>.json) whose batches replace the plan's"
    for name in ("step0", "batches"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--shard", default="1of1")
        p.add_argument("--sample", help=sample_help)
    for name in ("reproof", "replay", "check-list", "scan", "judge-batch"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--shard", default="1of1")
        p.add_argument("--batch", required=True)
        p.add_argument("--sample", help=sample_help)
    p = sub.add_parser("summary")
    p.add_argument("--out", required=True)
    p.add_argument("--sample", help=sample_help)
    a = ap.parse_args(argv)
    try:
        return {"xcensus": cmd_xcensus, "plan": cmd_plan, "step0": cmd_step0, "batches": cmd_batches,
                "reproof": cmd_reproof, "replay": cmd_replay, "check-list": cmd_check_list, "scan": cmd_scan,
                "judge-batch": cmd_judge_batch, "summary": cmd_summary, "sample": cmd_sample}[a.cmd](a)
    except (S.SealError, L.M0Error, subprocess.CalledProcessError) as e:
        print(f"KC_DONE rc=2 cmd={a.cmd} error={e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
