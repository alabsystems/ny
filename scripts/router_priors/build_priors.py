#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Offline priors v1 pipeline driver (#router-r7). See priorslib.py for the model and the gates.

    build_priors.py collect --roots DIR... --features instance_features.tsv --out trainset.tsv [--census c.json]
    build_priors.py fit     --rows trainset.tsv --table priors_v1.tsv --fixture priors_v1_rows.tsv [--report r.json]
    build_priors.py check   --rows trainset.tsv --table priors_v1.tsv --fixture priors_v1_rows.tsv

`fit` is the ONLY writer of the compiled table (`crates/ny-core/src/router/priors_v1.tsv`) and of its
anonymised fixture (`crates/ny-core/tests/data/priors_v1_rows.tsv`). `check` regenerates both in memory and
exits 1 on any byte of drift, so a hand edit of either file cannot survive a gate that runs it.

Every command ends with one DONE line that carries its own counts: `PRIORS_<CMD>_DONE rc=<rc> ...`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import priorslib as P  # noqa: E402


def _report(phases: dict) -> dict:
    out = {}
    for phase, rec in phases.items():
        r = {k: rec.get(k) for k in ("units", "families", "ships", "refused", "isolation")}
        if "logo" in rec:
            r["logo"] = {k: v for k, v in rec["logo"].items()}
            r["fingerprint"] = rec["fingerprint"]
            m = rec["model"]
            r["model"] = {"slope": m.slope, "intercepts": m.intercepts, "spreads": m.spreads,
                          "cells": {c: {k: v for k, v in cen.items() if k != "fam_w"}
                                    for c, cen in m.cells_census.items()}}
        out[phase] = r
    return out


def cmd_collect(a) -> int:
    rows, census = P.collect(a.roots, a.features)
    if not rows:
        print("PRIORS_COLLECT_DONE rc=2 rows=0 (an empty training set must never look like a fit)")
        return 2
    P.write_rows(a.out, rows)
    if a.census:
        with open(a.census, "w") as fh:
            json.dump(census, fh, indent=1, sort_keys=True)
            fh.write("\n")
    by_phase = {ph: sum(1 for r in rows if r.phase == ph) for ph in P.PHASES}
    fams = {ph: len({r.family for r in rows if r.phase == ph}) for ph in P.PHASES}
    print(json.dumps(census, sort_keys=True))
    print("PRIORS_COLLECT_DONE rc=0 rows={} receipts={} {}".format(
        len(rows), census["receipts_used"],
        " ".join(f"{ph}={by_phase[ph]}/{fams[ph]}fam" for ph in P.PHASES)))
    return 0


def _render(rows):
    phases = P.build(rows)
    census = {"receipts_used": len({r.receipt for r in rows})}
    table = P.render_table(phases, P.trainset_sha(rows), census)
    fixture = P.render_fixture(rows, phases)
    return phases, table, fixture


def cmd_fit(a) -> int:
    rows = P.read_rows(a.rows)
    phases, table, fixture = _render(rows)
    for path, text in ((a.table, table), (a.fixture, fixture)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
    if a.report:
        with open(a.report, "w") as fh:
            json.dump(_report(phases), fh, indent=1, sort_keys=True, default=str)
            fh.write("\n")
    shipped = [ph for ph, rec in phases.items() if rec.get("ships")]
    refused = [ph for ph, rec in phases.items() if not rec.get("ships")]
    for ph in refused:
        print(f"[priors] refused phase={ph} why={phases[ph].get('refused')}")
    print(f"PRIORS_FIT_DONE rc=0 rows={len(rows)} shipped={','.join(shipped) or '-'} "
          f"refused={','.join(refused) or '-'}")
    return 0


def cmd_check(a) -> int:
    rows = P.read_rows(a.rows)
    _, table, fixture = _render(rows)
    drift = []
    for path, text in ((a.table, table), (a.fixture, fixture)):
        if not os.path.isfile(path):
            drift.append(f"{path}: missing")
            continue
        if open(path).read() != text:
            drift.append(f"{path}: differs from what `fit` generates from {a.rows}")
    for d in drift:
        print(f"[priors] DRIFT {d}")
    rc = 1 if drift else 0
    print(f"PRIORS_CHECK_DONE rc={rc} drift={len(drift)}")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--roots", nargs="+", required=True)
    c.add_argument("--features", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--census")
    f = sub.add_parser("fit")
    f.add_argument("--rows", required=True)
    f.add_argument("--table", required=True)
    f.add_argument("--fixture", required=True)
    f.add_argument("--report")
    k = sub.add_parser("check")
    k.add_argument("--rows", required=True)
    k.add_argument("--table", required=True)
    k.add_argument("--fixture", required=True)
    a = ap.parse_args(argv)
    try:
        return {"collect": cmd_collect, "fit": cmd_fit, "check": cmd_check}[a.cmd](a)
    except P.PriorsError as exc:
        print(f"[priors] REFUSED {exc}")
        print(f"PRIORS_{a.cmd.upper()}_DONE rc=3 refused=1")
        return 3


if __name__ == "__main__":
    sys.exit(main())
