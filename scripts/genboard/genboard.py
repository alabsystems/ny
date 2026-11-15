#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""GEN-R generic-board harness (#router-r7, ROUTER_PROGRAM section 9). See genlib.py for the protocol.

    genboard.py configs  --out DIR [--categories a,b | --all] [--router on] [--optimizer on] [--no-priors]
    genboard.py plan     --year Y [--categories a,b] --seed S --frac F --out plan.tsv
                         [--holdout-entries FILE --seal seal.json]
    genboard.py run      --plan plan.tsv --configs DIR --pin BIN --out OUTDIR --arm gen-r|fold|tuned
                         [--categories a,b] [--tuned-judged FILE] [--truth-2026 F] [--truth-2025 F] [--dry-run]
    genboard.py judge    --plan plan.tsv --out OUTDIR --arm gen-r|fold|tuned --configs DIR
    genboard.py score    --plan plan.tsv --gen OUTDIR [--tuned OUTDIR | --tuned-bank] [--truth-2026 F]
                         [--truth-2025 F] [--report score.json]
    genboard.py logo     --plan plan.tsv --rules rule_evidence.tsv --base-configs DIR --out DIR
                         [--trainset T --features F]
    genboard.py seal     --entries FILE --out seal.json --note TEXT
    genboard.py verify-seal --entries FILE --seal seal.json
    genboard.py spend    --seal seal.json --ledger LEDGER --purpose TEXT [--allow-respend]

`run` refuses (before any lock wait) when: any NY_* other than NY_BOX_* is set; the pin is missing or not
executable; a GEN-R configs dir is not generic (one byte-identical yaml carrying only router/optimizer keys,
resolving for every category; a LOGO `fold` dir may add only rule switches set to false); a plan row's files
are missing; OUTDIR exists with a different plan. Each category runs in TWO `ny_box.sh measure` holds: its
first 10 seeded rows (then the dark-arm and stop-loss checks), then the rest. Every row is driven directly
(`<pin> vnncomp v1 ...`, stdout+stderr captured per row), never through `ny benchmarks run`. A sat <-> unsat
contradiction -- what the binary PUBLISHED, VOID row or not, against TUNED or the field -- writes
OUTDIR/EMERGENCY and stops (exit 3).

Every command ends with one `GENBOARD_<CMD>_DONE rc=<rc> ...` line that propagates the real rc.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import genlib as G  # noqa: E402

L = G.L
P = G.P
NY_ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))
MAIN_ROOT = "<home>/ny"  # benchmarks/ and reports/ live in the main tree (gitignored in worktrees)
BOX = os.path.join(MAIN_ROOT, "scripts", "ny_box.sh")


def data_root(arg: str | None) -> str:
    return arg or (NY_ROOT if os.path.isdir(os.path.join(NY_ROOT, "benchmarks")) else MAIN_ROOT)


def benchmark_categories(root: str, year: int) -> list[str]:
    d = os.path.join(root, f"benchmarks/vnncomp{year}/benchmarks")
    if not os.path.isdir(d):
        raise G.GenError(f"benchmark dir does not resolve: {d}")
    return sorted(c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))


def scored_categories(root: str, year: int) -> list[str]:
    if year == 2026:
        man = os.path.join(root, "reports/measured-2026/bank_manifest.json")
        return sorted(json.load(open(man)).get("banks", {}))
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(root, "reports/measured/*.csv"))
                  if os.path.isdir(os.path.join(root, "benchmarks/vnncomp2025/benchmarks",
                                                os.path.basename(p)[:-4])))


def _cats(arg: str | None):
    return [c for c in (arg or "").split(",") if c]


# ---- configs / plan ----------------------------------------------------------------------------------------

def cmd_configs(a) -> int:
    root = data_root(a.data_root)
    cats = _cats(a.categories) or sorted(set(benchmark_categories(root, 2025)) | set(benchmark_categories(root, 2026)))
    text = G.generic_yaml(a.router, a.optimizer, not a.no_priors)
    man = G.write_configs(a.out, cats, text=text)
    G.verify_configs(a.out, cats)
    print(f"GENBOARD_CONFIGS_DONE rc=0 files={len(man['files'])} generic_sha256={man['generic_sha256'][:12]} "
          f"keys={','.join(man['keys'])}")
    return 0


def read_entries(path: str) -> list[str]:
    return [x.strip() for x in open(path) if x.strip() and not x.startswith("#")]


def cmd_plan(a) -> int:
    root = data_root(a.data_root)
    cats = _cats(a.categories) or scored_categories(root, a.year)
    sealed = None
    if a.holdout_entries:
        if not a.seal:
            raise G.GenError("--holdout-entries needs --seal: an unsealed holdout is not a holdout")
        entries = read_entries(a.holdout_entries)
        G.verify_seal(a.seal, entries)
        sealed = set()
        for e in entries:
            y, c, i = e.split("|")
            sealed.add((int(y), c, int(i)))
    rows = G.build_plan(root, a.year, cats, a.seed, a.frac, check_files=not a.no_file_check, sealed=sealed)
    if os.path.exists(a.out):
        raise G.GenError(f"refusing: {a.out} exists (plans are pre-registered once)")
    G.write_plan(a.out, rows)
    per = {}
    for r in rows:
        per[r.category] = per.get(r.category, 0) + 1
    print(f"GENBOARD_PLAN_DONE rc=0 rows={len(rows)} categories={len(per)} seed={a.seed} frac={a.frac}")
    return 0


# ---- run ---------------------------------------------------------------------------------------------------

def preconditions(a, plan) -> dict:
    """Fail-closed checks that exercise the SAME paths the run uses, before any lock wait."""
    bad = []
    leaked = sorted(k for k in os.environ if k.startswith("NY_") and not k.startswith("NY_BOX_"))
    if leaked:
        bad.append(f"NY_* in the environment ({', '.join(leaked)}): a board runs with zero NY_*")
    pin = os.path.realpath(a.pin)
    if not (os.path.isfile(pin) and os.access(pin, os.X_OK)):
        bad.append(f"pin is not an executable file: {a.pin}")
    cats = sorted({r.category for r in plan})
    info = {}
    try:
        if a.arm in ROUTER_ARMS:
            info = G.verify_configs(a.configs, cats, fold=a.arm == "fold")
        else:
            missing = [c for c in cats if L.resolve_preset(a.configs, c) is None]
            if missing:
                bad.append(f"tuned configs resolve no preset for {missing}")
    except G.GenError as exc:
        bad.append(str(exc))
    for r in plan:
        for f in L.model_files(r.onnx) + [r.vnnlib]:
            if not os.path.isfile(f):
                bad.append(f"row {r.category}#{r.index}: missing {f}")
                break
    if not a.dry_run and not os.path.isfile(BOX):
        bad.append(f"the box script does not resolve: {BOX}")
    if bad:
        raise G.GenError("preconditions failed: " + " | ".join(bad))
    sha = hashlib.sha256(open(pin, "rb").read()).hexdigest() if os.path.isfile(pin) else "-"
    return {"pin": pin, "pin_sha256": sha, **info}


ROUTER_ARMS = ("gen-r", "fold")


def configs_priors(configs: str) -> bool:
    """Whether a generic / fold configs dir arms the offline prior (its one text, byte-identical everywhere)."""
    for p in sorted(glob.glob(os.path.join(configs, "**", "*.yaml"), recursive=True)):
        return G.priors_armed(open(p).read())
    return False


def row_paths(out: str, r) -> tuple[str, str]:
    stem = f"{r.year}_{r.category}_{r.index:05d}"
    return os.path.join(out, "res", stem + ".res"), os.path.join(out, "raw", stem + ".raw")


def category_script(rows, a, pin: str, out: str) -> str:
    lines = ["#!/usr/bin/env bash", "# generated by scripts/genboard/genboard.py run -- one box hold",
             "set -uo pipefail", f"cd {shlex.quote(data_root(a.data_root))} || exit 90", "WORST=0"]
    for r in rows:
        res, raw = row_paths(out, r)
        cmd = [pin, "vnncomp", "v1", r.category, r.onnx, r.vnnlib, res, str(r.budget), "--configs-dir",
               os.path.realpath(a.configs)]
        lines.append(f"T0=$(date +%s%N); timeout -k 10 {r.budget + 90} {' '.join(shlex.quote(c) for c in cmd)} "
                     f"> {shlex.quote(raw)} 2>&1; RC=$?; T1=$(date +%s%N); "
                     f"echo \"GENBOARD_ROW rc=$RC wall_ns=$((T1 - T0))\" >> {shlex.quote(raw)}; "
                     "[ $RC -gt $WORST ] && WORST=$RC")
    lines.append('echo "GENBOARD_HOLD_DONE rc=0 WORST=$WORST"')
    lines.append("exit 0")
    return "\n".join(lines) + "\n"


def judge_rows(rows, a, out: str) -> dict:
    outcomes = {}
    router_on = a.arm in ROUTER_ARMS
    priors_on = router_on and configs_priors(a.configs)
    for r in rows:
        res, raw = row_paths(out, r)
        if not os.path.isfile(raw):
            continue
        text = open(raw, errors="replace").read()
        wall = None
        for line in text.splitlines():
            if line.startswith("GENBOARD_ROW "):
                for tok in line.split():
                    if tok.startswith("wall_ns="):
                        try:
                            wall = int(tok[8:]) / 1e9
                        except ValueError:
                            wall = None
        flight = None
        side = res + ".flight.json"
        if os.path.isfile(side):
            try:
                flight = json.load(open(side))
            except ValueError:
                flight = None
        outcomes[r.key()] = G.judge_run(r, text, G.parse_result_file(res), wall, router_on,
                                        configs_dir=a.configs, preset_path=G.preset_base_of(flight, text),
                                        priors_on=priors_on)
    return outcomes


def tuned_reference(a, plan, strict: bool = False) -> dict:
    """Reference verdicts for contradiction checks: a same-pin judged table, else the bank (by index, key
    asserted). A category with no bank has no reference (reported); `strict` refuses that instead."""
    if getattr(a, "tuned_judged", None):
        return G.read_judged(a.tuned_judged)
    ref = {}
    root = data_root(a.data_root)
    for (year, cat) in sorted({(r.year, r.category) for r in plan}):
        try:
            inst = L.load_instances(L.scored_instances_csv(root, year, cat), check_files=False)
            bank = L.align_bank(inst, L.load_bank(L.bank_csv(root, year, cat)), f"{year} {cat}")
        except L.M0Error as exc:
            if strict:
                raise
            print(f"[genboard] no bank reference for {year} {cat} ({exc}): no contradiction check there")
            continue
        for r, v in zip(inst, bank):
            ref[(year, cat, r.index)] = G.Outcome((year, cat, r.index), v, None)
    return ref


def cmd_run(a) -> int:
    plan = G.read_plan(a.plan)
    if a.categories:
        keep = set(_cats(a.categories))
        plan = [r for r in plan if r.category in keep]
    if any(r.holdout for r in plan):
        raise G.GenError("the plan carries sealed holdout rows: score them only through `spend` + a sealed run")
    info = preconditions(a, plan)
    out = os.path.realpath(a.out)
    os.makedirs(os.path.join(out, "res"), exist_ok=True)
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(out, "holds"), exist_ok=True)
    meta = os.path.join(out, "RUN_META.json")
    plan_sha = hashlib.sha256(open(a.plan, "rb").read()).hexdigest()
    if os.path.isfile(meta):
        old = json.load(open(meta))
        if old.get("plan_sha256") != plan_sha or old.get("pin_sha256") != info["pin_sha256"]:
            raise G.GenError("OUTDIR belongs to another plan or pin: pick a new --out")
    else:
        with open(meta, "w") as fh:
            json.dump({"schema": G.SCHEMA, "arm": a.arm, "plan_sha256": plan_sha, **info,
                       "configs": os.path.realpath(a.configs)}, fh, indent=1, sort_keys=True)
            fh.write("\n")
    # Both arms are checked against the reference: a TUNED same-pin verdict that contradicts the bank is an
    # emergency too. Review fix (#router-r7): and against the FIELD, here and not only at `score`, so a field
    # contradiction stops the board before another row runs (the README's promise).
    tuned = tuned_reference(a, plan)
    truth = truth_table(a, plan)
    by: dict = {}
    for r in plan:
        by.setdefault((r.year, r.category), []).append(r)
    holds = stopped = emergencies = dark = 0
    for (year, cat), rows in sorted(by.items()):
        rows = sorted(rows, key=lambda r: r.order)
        stages = [rows[: G.STOPLOSS_ROWS], rows[G.STOPLOSS_ROWS:]]
        for stage_no, stage in enumerate(stages):
            if not stage:
                continue
            todo = [r for r in stage if not os.path.isfile(row_paths(out, r)[1])]
            if todo:
                script = os.path.join(out, "holds", f"{year}_{cat}_{stage_no}.sh")
                with open(script, "w") as fh:
                    fh.write(category_script(todo, a, info["pin"], out))
                os.chmod(script, 0o755)
                if a.dry_run:
                    print(f"[genboard] dry-run hold {script} rows={len(todo)}")
                else:
                    env = {k: v for k, v in os.environ.items() if not k.startswith("NY_") or k.startswith("NY_BOX_")}
                    env.setdefault("NY_BOX_WAIT_LOCK", "21600")
                    env.setdefault("NY_BOX_WAIT_QUIET", "21600")
                    budget = sum(r.budget + 90 for r in todo)
                    rc = subprocess.call([BOX, "measure", "--budget", str(budget), "--", "bash", script],
                                         env=env, cwd=data_root(a.data_root))
                    holds += 1
                    if rc != 0:
                        print(f"[genboard] hold {script} rc={rc} (refused or failed): stopping")
                        print(f"GENBOARD_RUN_DONE rc={rc} holds={holds} stopped={stopped} emergencies=0")
                        return rc
            if a.dry_run:
                continue
            judged = judge_rows(stage, a, out)
            for r in stage:
                o = judged.get(r.key())
                pub = G.published_verdict(o)
                t = G.published_verdict(tuned.get(r.key()))
                tr = truth.get(r.key())
                tag = " (VOID row)" if o is not None and o.verdict == "void" else ""
                for ref, what in ((t, "TUNED"), (tr, "FIELD")):
                    if G.contradiction(pub, ref):
                        emergencies += 1
                        with open(os.path.join(out, "EMERGENCY"), "a") as fh:
                            fh.write(f"{year} {cat} #{r.index}: {a.arm} {pub}{tag} vs {what} {ref}\n")
            if emergencies:
                print(f"[genboard] SOUNDNESS EMERGENCY: see {out}/EMERGENCY")
                print(f"GENBOARD_RUN_DONE rc=3 holds={holds} stopped={stopped} emergencies={emergencies}")
                return 3
            if stage_no == 0 and a.arm in ROUTER_ARMS:
                firsts = [judged[r.key()] for r in stage if r.key() in judged]
                td = {k: v.verdict for k, v in tuned.items()}
                if G.dark_arm(firsts):
                    dark += 1
                    with open(os.path.join(out, "DARK"), "a") as fh:
                        fh.write(f"{year} {cat}: the first {G.STOPLOSS_ROWS} seeded rows are all VOID "
                                 f"({firsts[0].void_reason}): stopped with NO capability verdict\n")
                    break
                if G.stop_loss(firsts, td):
                    stopped += 1
                    with open(os.path.join(out, "STOPLOSS"), "a") as fh:
                        fh.write(f"{year} {cat}: decided <= {G.STOPLOSS_FRAC} x TUNED on the first "
                                 f"{G.STOPLOSS_ROWS} seeded rows\n")
                    break
    census = ""
    if not a.dry_run:
        all_judged = judge_rows(plan, a, out)
        G.write_results(os.path.join(out, "results.csv"), plan, all_judged)
        census = " " + census_tokens(all_judged.values())
    print(f"GENBOARD_RUN_DONE rc=0 holds={holds} stopped={stopped} dark={dark} emergencies={emergencies} "
          f"dry_run={int(a.dry_run)}{census}")
    return 0


def census_tokens(outcomes) -> str:
    """`void=N prior_read=N prior_host_gated=N prior_unavailable=N hosts=a:n,b:m` -- so a board whose prior
    was host-gated (e.g. a cgroup cap below the table's host-class floor) says so in its DONE line."""
    outcomes = list(outcomes)
    hosts: dict = {}
    for o in outcomes:
        hosts[o.host] = hosts.get(o.host, 0) + 1
    count = lambda s: sum(1 for o in outcomes if o.prior == s)  # noqa: E731
    return (f"void={sum(1 for o in outcomes if o.verdict == 'void')} prior_read={count('read')} "
            f"prior_host_gated={count('host_gated')} prior_unavailable={count('unavailable')} "
            f"hosts={','.join(f'{h}:{n}' for h, n in sorted(hosts.items()))}")


def cmd_judge(a) -> int:
    plan = G.read_plan(a.plan)
    out = os.path.realpath(a.out)
    judged = judge_rows(plan, a, out)
    G.write_results(os.path.join(out, "results.csv"), plan, judged)
    print(f"GENBOARD_JUDGE_DONE rc=0 rows={len(judged)} {census_tokens(judged.values())}")
    return 0


# ---- score -------------------------------------------------------------------------------------------------

def truth_table(a, plan) -> dict:
    out = {}
    root = data_root(a.data_root)
    for year, path in ((2026, a.truth_2026), (2025, a.truth_2025)):
        if not path or not any(r.year == year for r in plan):
            continue
        cats = L.parse_results_txt(path, year)
        for cat in sorted({r.category for r in plan if r.year == year}):
            try:
                inst = L.load_instances(L.scored_instances_csv(root, year, cat), check_files=False)
            except L.M0Error as exc:
                print(f"[genboard] no field truth for {year} {cat} ({exc})")
                continue
            for r, t in zip(inst, L.align_truth(inst, cats.get(cat), f"{year} {cat}")):
                if t is not None:
                    out[(year, cat, r.index)] = t.token()
    return out


def cmd_score(a) -> int:
    plan = G.read_plan(a.plan)
    gen = G.read_judged(os.path.join(a.gen, "results.csv.judged.tsv"))
    if a.tuned:
        tuned = G.read_judged(os.path.join(a.tuned, "results.csv.judged.tsv"))
        same_pin = json.load(open(os.path.join(a.tuned, "RUN_META.json"))).get("pin_sha256") == json.load(
            open(os.path.join(a.gen, "RUN_META.json"))).get("pin_sha256")
    elif a.tuned_bank:
        tuned = tuned_reference(a, plan, strict=True)
        same_pin = False
    else:
        raise G.GenError("score needs --tuned OUTDIR (same pin) or an explicit --tuned-bank")
    s = G.score(plan, gen, tuned, truth_table(a, plan))
    print(G.render_score(s, same_pin))
    if a.report:
        rep = dict(s)
        rep["categories"] = [dict(c.__dict__, ratio=c.ratio) for c in s["categories"]]
        rep["same_pin"] = same_pin
        with open(a.report, "w") as fh:
            json.dump(rep, fh, indent=1, sort_keys=True, default=str)
            fh.write("\n")
    rc = 3 if s["emergencies"] else 0
    print(f"GENBOARD_SCORE_DONE rc={rc} ratio={'-' if s['ratio'] is None else round(s['ratio'], 4)} "
          f"emergencies={len(s['emergencies'])} voids={s['voids']} same_pin={int(same_pin)}")
    return rc


# ---- LOGO --------------------------------------------------------------------------------------------------

def cmd_logo(a) -> int:
    plan = G.read_plan(a.plan)
    rules = G.read_rule_evidence(a.rules)
    folds = G.logo_folds(plan, rules)
    base_files = [p for p in glob.glob(os.path.join(a.base_configs, "**", "*.yaml"), recursive=True)]
    if not base_files:
        raise G.GenError(f"no generic yaml under {a.base_configs}")
    base_text = open(base_files[0]).read()
    G.verify_configs(a.base_configs, sorted({r.category for r in plan}))
    if os.path.exists(a.out):
        raise G.GenError(f"refusing: {a.out} exists")
    os.makedirs(a.out)
    report = {"schema": G.SCHEMA, "kind": "logo", "folds": {}}
    prior = {}
    # Review fix (#router-r7): the GEN-R binary compiles the FULL table (every family's receipts). When the
    # generic text arms the prior, a fold's rows are held out only if the prior refit without the family
    # makes the same planning decisions -- so without the refit report a fold would pass the full-table
    # binary's rows as held-out. Fail closed.
    if G.priors_armed(base_text) and not (a.trainset and a.features):
        raise G.GenError("the generic default arms optimizer.priors: `logo` needs --trainset and --features to "
                         "report the prior's held-out decisions (else a fold would treat the full-table binary's "
                         "rows as held-out)")
    if a.trainset and a.features:
        feats = P.load_features(a.features)
        by_model = {}
        for (year, cat, onnx_rel), f in feats.items():
            params = float(P._int(f.get("n_params_init")))
            if params > 0:
                by_model[(int(year), cat, L.model_key(onnx_rel))] = (*P.key_bits(f), params)
        prior = G.prior_decision_changes(plan, P.read_rows(a.trainset), by_model,
                                         clauses=instance_clauses(a.features))
    for fam, keys in sorted(folds.items()):
        rec = {"ablations": keys, "rows": sum(1 for r in plan if r.family == fam), "prior": prior.get(fam)}
        if keys:
            cats = sorted({r.category for r in plan if r.family == fam})
            fold_dir = os.path.join(a.out, f"fold_{fam}")
            text = G.fold_yaml(base_text, keys)
            _write_fold(fold_dir, cats, text)
            rec["configs"] = fold_dir
            rec["note"] = ("run this family's rows under its fold configs with `run --arm fold` (rules with "
                           "evidence only here are off)")
        else:
            rec["note"] = ("no rule's evidence is this family alone: its GEN-R row IS its held-out row "
                           "(explicit, not a hidden zero gap)")
        if rec["prior"] and rec["prior"]["decision_changes"]:
            rec["needs_heldout_binary"] = True
        report["folds"][fam] = rec
    with open(os.path.join(a.out, "LOGO.json"), "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
        fh.write("\n")
    n_fold = sum(1 for r in report["folds"].values() if r["ablations"])
    n_bin = sum(1 for r in report["folds"].values() if r.get("needs_heldout_binary"))
    print(f"GENBOARD_LOGO_DONE rc=0 families={len(folds)} ablated_folds={n_fold} heldout_binaries={n_bin}")
    return 0


def instance_clauses(path: str) -> dict:
    """(year, category, inst_idx) -> the property's clause count from the instance-features table (the
    per-clause phases' scale; `L.load_features` keeps one row per model, clauses are per instance)."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                k = int(float(r.get("n_clauses") or 0))
                if k > 0:
                    out[(int(r["year"]), r["category"], int(r["inst_idx"]))] = k
            except (TypeError, ValueError, KeyError):
                continue
    return out


def _write_fold(out: str, cats, text: str) -> None:
    """A fold configs dir: like the generic board, but the fold's text carries ablation keys (rule switches,
    which are generic by construction: every category in the fold gets the same bytes)."""
    if os.path.exists(out):
        raise G.GenError(f"refusing: {out} exists")
    for year in (2025, 2026):
        yd = os.path.join(out, G.YEAR_DIRS[year])
        os.makedirs(yd)
        for cat in cats:
            with open(os.path.join(yd, f"{cat.lower()}.yaml"), "w") as fh:
                fh.write(text)


# ---- sealed holdout ----------------------------------------------------------------------------------------

def cmd_seal(a) -> int:
    rec = G.write_seal(a.out, read_entries(a.entries), a.note)
    print(f"GENBOARD_SEAL_DONE rc=0 sha256={rec['sha256'][:12]} count={rec['count']}")
    return 0


def cmd_verify_seal(a) -> int:
    rec = G.verify_seal(a.seal, read_entries(a.entries))
    print(f"GENBOARD_VERIFY_SEAL_DONE rc=0 sha256={rec['sha256'][:12]} count={rec['count']}")
    return 0


def cmd_spend(a) -> int:
    seal = json.load(open(a.seal))
    n = G.spend_holdout(a.ledger, seal, a.purpose, a.allow_respend)
    print(f"GENBOARD_SPEND_DONE rc=0 sha256={seal['sha256'][:12]} spend_no={n}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", help="tree holding benchmarks/ and reports/ (default: this tree, else main)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("configs")
    c.add_argument("--out", required=True)
    c.add_argument("--categories")
    c.add_argument("--router", default="on")
    c.add_argument("--optimizer", default="on")
    c.add_argument("--no-priors", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("--year", type=int, required=True, choices=(2025, 2026))
    p.add_argument("--categories")
    p.add_argument("--seed", required=True)
    p.add_argument("--frac", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--holdout-entries")
    p.add_argument("--seal")
    p.add_argument("--no-file-check", action="store_true")
    for name in ("run", "judge"):
        r = sub.add_parser(name)
        r.add_argument("--plan", required=True)
        r.add_argument("--configs", required=True)
        r.add_argument("--out", required=True)
        r.add_argument("--arm", required=True, choices=("gen-r", "fold", "tuned"))
        if name == "run":
            r.add_argument("--pin", required=True)
            r.add_argument("--categories")
            r.add_argument("--tuned-judged")
            r.add_argument("--dry-run", action="store_true")
            r.add_argument("--truth-2026",
                           default=os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt"))
            r.add_argument("--truth-2025",
                           default=os.path.expanduser("~/vnncomp2025_results/SCORING-ZERO-TOL/results.txt"))
    s = sub.add_parser("score")
    s.add_argument("--plan", required=True)
    s.add_argument("--gen", required=True)
    s.add_argument("--tuned")
    s.add_argument("--tuned-bank", action="store_true")
    s.add_argument("--truth-2026", default=os.path.expanduser("~/vnncomp2026_results/SCORING/results.txt"))
    s.add_argument("--truth-2025", default=os.path.expanduser("~/vnncomp2025_results/SCORING-ZERO-TOL/results.txt"))
    s.add_argument("--report")
    lg = sub.add_parser("logo")
    lg.add_argument("--plan", required=True)
    lg.add_argument("--rules", default=os.path.join(HERE, "rule_evidence.tsv"))
    lg.add_argument("--base-configs", required=True)
    lg.add_argument("--out", required=True)
    lg.add_argument("--trainset", default=os.path.join(HERE, "..", "router_priors", "data", "priors_v1_trainset.tsv"))
    lg.add_argument("--features")
    se = sub.add_parser("seal")
    se.add_argument("--entries", required=True)
    se.add_argument("--out", required=True)
    se.add_argument("--note", required=True)
    vs = sub.add_parser("verify-seal")
    vs.add_argument("--entries", required=True)
    vs.add_argument("--seal", required=True)
    sp = sub.add_parser("spend")
    sp.add_argument("--seal", required=True)
    sp.add_argument("--ledger", required=True)
    sp.add_argument("--purpose", required=True)
    sp.add_argument("--allow-respend", action="store_true")
    a = ap.parse_args(argv)
    fn = {"configs": cmd_configs, "plan": cmd_plan, "run": cmd_run, "judge": cmd_judge, "score": cmd_score,
          "logo": cmd_logo, "seal": cmd_seal, "verify-seal": cmd_verify_seal, "spend": cmd_spend}[a.cmd]
    try:
        return fn(a)
    except (G.GenError, L.M0Error, P.PriorsError) as exc:
        print(f"[genboard] REFUSED {exc}")
        print(f"GENBOARD_{a.cmd.upper().replace('-', '_')}_DONE rc=2 refused=1")
        return 2


if __name__ == "__main__":
    sys.exit(main())
