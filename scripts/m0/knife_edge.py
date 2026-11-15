#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""M0 item 3: SYNTHETIC SATS and the KNIFE-EDGE LADDER (ENGINE_PROGRAM section 2, M0; judge_soundness section 6).

Given a VNN-LIB property and a known witness x* (a bank artefact `reports/measured-2026/<cat>_artifacts/
witnesses/*.counterexample[.gz]`, or a `sat` result file), rewrite the property's threshold(s) to
`f(x*) - delta` (for `Y >= c`; `f(x*) + delta` for `Y <= c`) for delta in {1e-3, 1e-6, 1e-9, 1e-12}, so that
x* satisfies the rewritten property with an exact slack of delta. Each rung is a truly-SAT instance: a sound
verifier must NEVER prove it unsat, and an unsat on any rung is a soundness EMERGENCY.

What is rewritten, exactly:
  * every clause that holds at x* (its threshold atoms move to slack delta; its relational atoms keep their
    own slack); clauses that do not hold at x* are left byte-identical;
  * only the numeric token of each rewritten atom changes (offsets are kept), so the property's structure --
    and hence ny's routing -- is unchanged; a provenance comment is APPENDED at the end of the file;
  * the new threshold is the exact f64 floor of y* - delta (ceil of y* + delta), so the slack is >= delta
    exactly, and it is spelled as a plain decimal that round-trips to the same f64.

Refused (M0Error, exit 2), never guessed:
  * x* outside the input box, x* satisfying no clause, a missing output value;
  * a witnessed clause with NO threshold atom (relational-only, i.e. every classification-robustness row:
    cifar100 / tinyimagenet / cct). ny's parser refuses an offset on a relational atom
    (crates/ny-onnx/src/vnnlib/normalize/convert.rs: "Relational constraint must have zero constant"), so the
    knife-edge is not expressible in VNN-LIB for ny there. --allow-partial emits the rung anyway with those
    clauses at their natural slack and records knife_edge_exact=false.

Semantics note, recorded per rung: y* in a witness file is ny's ONNX Runtime f32 forward. A rung whose delta is
below the f32 spacing of y* (`sub_f32_resolution`) is anchored to a value that the real-valued forward may not
reproduce to that precision; pass --y-values with a higher-precision forward at x* to anchor the ladder in
real semantics. The kill line is unchanged either way: an unsat on any rung stops everything.

    python3 scripts/m0/knife_edge.py --from-spec <negctl>/negctl_spec.tsv --row NEG_cz_r4 \
        --witness reports/measured-2026/cgan2026_artifacts/witnesses/0004-cGAN_imgSz32_nCh_3_prop_0_....counterexample \
        --out-dir <home>/ny/scratchpad/m0/knife_r4 --spec-out <home>/ny/scratchpad/m0/knife_spec.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m0lib as L  # noqa: E402


def delta_tag(d: str) -> str:
    return "d" + d.replace("-", "m").replace("+", "").replace(".", "p")


def build_ladder(vnnlib: str, witness: str, out_dir: str, deltas: list[str], y_values: str | None = None,
                 clause: str = "auto", allow_partial: bool = False, allow_non_witness: bool = False) -> dict:
    """Write one rewritten property per delta into out_dir; returns the manifest (also written as JSON)."""
    text = L.read_text(vnnlib)
    prop = L.parse_property(text)
    w = L.load_witness(witness, prop.in_name, prop.out_name)
    box_margin = L.witness_in_box(prop, w.x)
    y = dict(w.y)
    y_source = "witness"
    y_diff = None
    if y_values:
        yv = L.load_y_values(y_values)
        common = sorted(set(yv) & set(y))
        if not common:
            raise L.M0Error("--y-values shares no output with the witness")
        y_diff = max(abs(yv[k] - y[k]) for k in common)
        if any(abs(yv[k] - y[k]) > 1e-2 * max(1.0, abs(y[k])) for k in common):
            raise L.M0Error(f"--y-values disagree with the witness outputs by {y_diff:g}: not the same point?")
        y.update(yv)
        y_source = f"y-values:{os.path.abspath(y_values)}"
    stem = L.stem(vnnlib)
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "schema": "m0-knife-edge/v1", "source_vnnlib": os.path.abspath(vnnlib), "source_sha256": L.sha256_file(vnnlib),
        "vnnlib_version": prop.version, "witness": os.path.abspath(witness), "witness_sha256": w.sha256,
        "witness_min_distance_to_box_face": float(box_margin), "y_source": y_source,
        "y_values_max_abs_diff_vs_witness": y_diff, "clause_selector": clause, "rungs": [],
    }
    planned = []
    for d in deltas:
        new_text, rep = L.knife_edge_rewrite(prop, y, d, clause, allow_partial, allow_non_witness)
        path = os.path.join(out_dir, f"{stem}__ke_{delta_tag(d)}.vnnlib")
        if os.path.exists(path):
            raise L.M0Error(f"refusing to overwrite {path}")
        # the rewritten property must still parse, and x* must satisfy it with slack >= delta
        check = L.parse_property(new_text)
        if len(check.clauses) != len(prop.clauses):
            raise L.M0Error("internal: the rewrite changed the clause structure")
        if not any(all(a.holds({k: Fraction(v) for k, v in y.items()}) for a in cl) for cl in check.clauses):
            raise L.M0Error("internal: x* does not satisfy the rewritten property")
        trailer = (f"\n; m0 knife-edge rung delta={d} source={os.path.basename(vnnlib)} "
                   f"sha256={manifest['source_sha256'][:16]} witness={os.path.basename(witness)} "
                   f"sha256={w.sha256[:16]} y_source={y_source.split(':', 1)[0]} exact={int(rep['knife_edge_exact'])}\n")
        planned.append((path, new_text + trailer, rep))
    for path, body, rep in planned:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        rep = dict(rep, file=path, sha256=L.sha256_file(path))
        manifest["rungs"].append(rep)
    man_path = os.path.join(out_dir, f"{stem}__knife_edge.json")
    if os.path.exists(man_path):
        raise L.M0Error(f"refusing to overwrite {man_path}")
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    manifest["manifest_path"] = man_path
    return manifest


def spec_rows_for(manifest: dict, src: dict, reps_c: int, reps_p: int) -> list[dict]:
    rows = []
    for rung in manifest["rungs"]:
        tag = f"KNIFE_{src['tag']}_{delta_tag(rung['delta'])}"
        note = (f"delta={rung['delta']} exact={int(rung['knife_edge_exact'])} "
                f"sub_f32={int(rung['sub_f32_resolution'])} clauses_rewritten={len(rung['clauses_rewritten'])} "
                f"y_source={manifest['y_source'].split(':', 1)[0]} source_tag={src['tag']}")
        rows.append(L.spec_row(
            tag=tag, kind="KNIFE", tier=src["tier"], year=src["year"], category=src["category"], onnx=src["onnx"],
            vnnlib=rung["file"], budget=src["budget"], configs_dir=src["configs_dir"], preset=src["preset"],
            truth="sat", bank="-", index=src["index"], reps_c=reps_c, reps_p=reps_p, need=src["need"],
            alarm=src["alarm"], flight=src["flight"], group=f"knife/{src['tier']}", note=note))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--witness", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vnnlib", help="source property (default: the --row's vnnlib)")
    ap.add_argument("--from-spec", help="negative-control spec whose --row supplies model, budget, configs, tier")
    ap.add_argument("--row", help="tag of the source row in --from-spec")
    ap.add_argument("--spec-out", help="append KNIFE rows here (created if absent)")
    ap.add_argument("--deltas", default=",".join(L.DEFAULT_DELTAS))
    ap.add_argument("--y-values", help="higher-precision outputs at x* in the witness format")
    ap.add_argument("--clause", default="auto")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--allow-non-witness", action="store_true",
                    help="synthetic sat on a point that satisfies no clause (needs --clause K and --y-values)")
    ap.add_argument("--reps-competition", type=int, default=1)
    ap.add_argument("--reps-proof", type=int, default=1)
    a = ap.parse_args(argv)
    try:
        src = None
        if a.from_spec or a.row:
            if not (a.from_spec and a.row):
                raise L.M0Error("--from-spec and --row go together")
            hits = [r for r in L.read_spec(a.from_spec) if r["tag"] == a.row]
            if len(hits) != 1:
                raise L.M0Error(f"row {a.row} not found exactly once in {a.from_spec}")
            src = hits[0]
            if src["kind"] != "NEGCTL":
                raise L.M0Error(f"row {a.row} is {src['kind']}: knife-edge rows inherit a NEGCTL row's "
                                "sat-finding-OFF configs (the attack would otherwise answer before the tier)")
            if a.vnnlib and os.path.abspath(a.vnnlib) != src["vnnlib"]:
                raise L.M0Error("--vnnlib differs from the row's vnnlib")
        vnnlib = a.vnnlib or (src["vnnlib"] if src else None)
        if not vnnlib:
            raise L.M0Error("need --vnnlib or --from-spec/--row")
        if a.spec_out and not src:
            raise L.M0Error("--spec-out needs --from-spec/--row (a KNIFE row inherits the control's configs)")
        if a.allow_non_witness and (a.clause == "auto" or not a.y_values):
            raise L.M0Error("--allow-non-witness needs --clause K and --y-values")
        deltas = [d.strip() for d in a.deltas.split(",") if d.strip()]
        man = build_ladder(vnnlib, a.witness, a.out_dir, deltas, a.y_values, a.clause, a.allow_partial,
                           a.allow_non_witness)
        for r in man["rungs"]:
            rw = r["rewrites"]
            print(f"  rung delta={r['delta']} exact={int(r['knife_edge_exact'])} sub_f32={int(r['sub_f32_resolution'])} "
                  f"clauses={r['clauses_rewritten']} atoms={len(rw)} first_slack={rw[0]['slack'] if rw else '-'} "
                  f"file={r['file']}")
        n = 0
        if a.spec_out:
            rows = spec_rows_for(man, src, a.reps_competition, a.reps_proof)
            L.write_spec(a.spec_out, rows, append=True)
            n = len(rows)
        print(f"KNIFE_DONE rc=0 rungs={len(man['rungs'])} spec_rows={n} manifest={man['manifest_path']}")
        return 0
    except L.M0Error as e:
        print(f"KNIFE_REFUSED: {e}", file=sys.stderr)
        print("KNIFE_DONE rc=2 rungs=0")
        return 2


if __name__ == "__main__":
    sys.exit(main())
