#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Self-tests of the seal pipeline (scripts/seal/). No cargo, no ny binary, no box.

Every test is a unittest method, so the run prints one `test_<name> (...) ... ok` line per test and a gate can assert
each BY NAME (CLAUDE.md: a filter that matched nothing must not read as green). The last line is
`SEAL_SELFTEST_DONE rc=<0|1> ran=<n> failures=<n> errors=<n> skipped=<n>`.

A STUB WORLD stands in for everything real: a git repository whose commit carries configs/ and the real preset
schema source, a benchmark tree with a hand-encoded ONNX model and VNN-LIB properties, a bank, a bank manifest, field
results, a capability registry, a stub pin (named stubpin_*, never ny*: another agent's `pgrep '^ny'` quiescence gate
must not see it) and a stub scripts/ny_box.sh. The runner tests drive scripts/seal/seal_run.sh --run end to end
through the REAL m0 runner against that world.

    python3 scripts/seal/seal_selftest.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seallib as S  # noqa: E402
from seallib import L  # noqa: E402
from seallib import C  # noqa: E402

SCRIPTS = os.path.dirname(HERE)
REPO = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
import engine_program_audit as EPA  # noqa: E402

CAT = "toy_2026"
SCHEMA_SRC = os.path.join(REPO, "crates/ny-cli/src/preset/mod.rs")
DECL_SRC = os.path.join(REPO, "crates/ny-core/src/router/decl.rs")
NEED = r"\[toy-tier\] armed=1 nodes=[1-9]"
CLOSES = r"\[toy-tier\] armed=1 nodes=[0-9]+ closed=([0-9]+)"


def write(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return path


# ---------------------------------------------------------------------------------------------------------
# a hand-encoded ONNX model (protobuf), so the facts scanner is tested without the onnx package
# ---------------------------------------------------------------------------------------------------------


def _vi(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _ld(fn: int, payload: bytes) -> bytes:
    return _vi(fn << 3 | 2) + _vi(len(payload)) + payload


def _s(fn: int, s: str) -> bytes:
    return _ld(fn, s.encode())


def onnx_node(op: str, ins: list[str], outs: list[str], sub: bytes | None = None) -> bytes:
    b = b"".join(_s(1, i) for i in ins) + b"".join(_s(2, o) for o in outs) + _s(4, op)
    if sub is not None:
        b += _ld(5, _s(1, "body") + _ld(6, sub))    # AttributeProto{name, g}
    return b


def onnx_graph(nodes: list[bytes], inits: list[str]) -> bytes:
    return (b"".join(_ld(1, n) for n in nodes) + b"".join(_ld(5, _s(8, t) + _vi(1 << 3) + _vi(2)) for t in inits)
            + _s(2, "g"))


def onnx_model(nodes: list[bytes], inits: list[str]) -> bytes:
    return _vi(1 << 3) + _vi(7) + _ld(7, onnx_graph(nodes, inits))


TOY_MODEL = onnx_model([onnx_node("Gemm", ["X", "W1", "B1"], ["h1"]), onnx_node("Relu", ["h1"], ["h2"]),
                        onnx_node("Reshape", ["h2", "shape"], ["h3"]), onnx_node("Gemm", ["h3", "W2", "B2"], ["Y"])],
                       ["W1", "B1", "W2", "B2", "shape"])


def toy_prop(comment: str, threshold: bool = False) -> str:
    head = f"; sealtest {comment}\n" + "".join(f"(declare-const X_{i} Real)\n" for i in range(2)) + \
        "".join(f"(declare-const Y_{i} Real)\n" for i in range(3)) + \
        "(assert (>= X_0 0.0))\n(assert (<= X_0 1.0))\n(assert (>= X_1 0.0))\n(assert (<= X_1 1.0))\n"
    if threshold:
        return head + "(assert (or (and (>= Y_0 0.5))))\n"
    return head + "(assert (or (and (>= Y_1 Y_0)) (and (>= Y_2 Y_0))))\n"


# (bank verdict, banked wall, field truth, per-arm verdicts the stub pin states, threshold property?)
TOY_ROWS = [
    ("unsat", 5.0, "unsat", "arm=unsat ctl=unsat negctl=timeout", False),     # p0: S5 arrival (cheapest)
    ("unsat", 8.0, "unsat", "arm=unsat ctl=timeout negctl=timeout", False),   # p1: near; the arm converts it
    ("sat", 1.0, "sat", "arm=sat ctl=sat negctl=timeout", False),             # p2: banked + field sat
    ("timeout", 10.0, "sat", "arm=timeout ctl=timeout negctl=timeout", True),  # p3: field-SAT, witnessed (knife)
    ("unsat", 9.5, "unsat", "arm=unsat ctl=unsat negctl=timeout", False),     # p4: near
    ("timeout", 10.0, "unsat", "arm=timeout ctl=timeout negctl=timeout", False),  # p5: gap
]

STUB_PIN = r'''#!/usr/bin/env bash
# stub pin for seal_selftest. Markers: Running VNN-COMP instance Loading preset: Result:  [stop] verdict=
# [proof-profile] mode= [frozen-dual] armed=1 objectives= [frozen-dual] class= margin-row BaB (concurrent): UNSAT in 1s
# EARLY EXIT REFUSED / REFUSING to publish `unsat` / [fwd-root] union screen_refuted= / [toy-tier] armed=1 nodes=
# [router] cap=
SELF=$(sha256sum "$0" | cut -d' ' -f1)
if [ "${1:-}" = --build-id ]; then
  echo "NY_BUILD_ID_V1 version=0.0.1 features=mip,cuda backend=cpu commit=__COMMIT__ dirty=${SEALTEST_DIRTY:-false} revision=1 exe=$SELF id=x"; exit 0
fi
if [ "${1:-} ${2:-}" = "beta-crown /nonexistent_seal.onnx" ]; then
  Y=$4
  if [ "${SEALTEST_LOADER:-strict}" = strict ] && grep -q seal_negative_control_unknown_key "$Y"; then
    echo "Error: Failed to parse preset YAML: $Y"
    echo "    margin_row: unknown field \`seal_negative_control_unknown_key\`, expected one of x"; exit 1
  fi
  echo "Error: Model loading failed: File not found: /nonexistent_seal.onnx"; exit 1
fi
if [ "${1:-} ${2:-}" = "vnncomp plan" ]; then
  CFG=$8
  if [ -n "${SEALTEST_ALWAYS_PRESET:-}" ]; then echo "preset:   $SEALTEST_ALWAYS_PRESET"
  elif [ -f "$CFG/vnncomp26/$3.yaml" ]; then echo "preset:   $CFG/vnncomp26/$3.yaml"
  else echo "preset:   none (built-in defaults)"; fi
  echo "vnnlib:   $5"; echo "  scored_budget_secs  = $6"; exit 0
fi
if [ "${1:-}" = proof-check ]; then
  # #certified-moat stub of `ny proof-check`: the bundle header's stub_check token decides the verdict; the report
  # carries the checker's real fields (schema, builds, input/bundle hashes, cover, nodes) and the DONE line its sha256
  python3 - "$@" <<'PCEOF'
import hashlib, json, os, sys
a = sys.argv[2:]
if a and a[0] == "--help":
    print("Check the proof bundle (stub)\n\nUsage: stubpin proof-check [OPTIONS] <BUNDLE>")
    sys.exit(0)
bundle = a[0]
def opt(k, d=None):
    return a[a.index(k) + 1] if k in a else d
onnx, vnn, rep = opt("--onnx"), opt("--vnnlib"), opt("--report", bundle + ".check.json")
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
hdr = json.loads(open(bundle).readline())
ck = hdr.get("stub_check", "verified")
if ck == "crash":
    print("Error: proof-check: stub crash", file=sys.stderr)
    sys.exit(4)
code = {"verified": 0, "unconfirmed": 10, "invalid": 20, "refuted": 21, "malformed": 30, "unsupported": 31,
        "checker_fault": 40}[ck]
ok = ck == "verified"
d = {"schema": "ny-proof-check-report/v1", "checker_build": "NY_BUILD_STAMP_V1 commit=__COMMIT__ dirty=false",
     "bundle": {"path": bundle, "sha256": h(bundle), "schema": hdr.get("schema"), "lines": 3,
                "header_build": hdr.get("build")},
     "onnx": {"path": onnx, "sha256": h(onnx)}, "vnnlib": {"path": vnn, "sha256": h(vnn)},
     "verdict": ck, "exit_code": code, "reasons": [],
     "cover": [{"region": 0, "disjunct": 0, "claimed_ok": True, "confirmed_ok": ok}],
     "nodes": {"total": 1, "confirmed": int(ok), "unconfirmed": int(not ok), "opaque": int(ck == "unconfirmed")}}
text = json.dumps(d, sort_keys=True, separators=(",", ":")) + "\n"
open(rep, "w").write(text)
sha = hashlib.sha256(text.encode()).hexdigest()
done = (f"PROOF_CHECK_DONE rc={code} verdict={ck} nodes=1 confirmed={int(ok)} unconfirmed={int(not ok)} cover=ok "
        f"refuted={int(ck == 'refuted')} wall_s=0.010 report={rep} report_sha256={sha}")
print(f"[proof-check] verdict={ck} exit={code}", file=sys.stderr)
print(done, file=sys.stderr)
print(done)
sys.exit(code)
PCEOF
  exit $?
fi
if [ "${1:-} ${2:-}" = "vnncomp census" ]; then
  # #admission-census stub: the binary's census format; SEALTEST_CENSUS_ADMIT is a regex on the property path
  python3 - "$SELF" "$@" <<'CENSUSEOF'
import hashlib, os, re, sys
selfsha, args = sys.argv[1], sys.argv[2:]
man, out = args[args.index("--manifest") + 1], args[args.index("--out") + 1]
if os.path.exists(out):
    sys.exit("refusing to overwrite " + out)
raw = open(man, "rb").read()
rx = re.compile(os.environ.get("SEALTEST_CENSUS_ADMIT", "."))
rows = [l.split("\t") for l in raw.decode().splitlines()[1:] if l and not l.startswith("#")]
ops = "linear+conv+conv_transpose+relu+sigmoid+tanh+softmax+bn+residual+concat+pool+sign+resize+bilinear+norm"
specs = "robustness+halfspace+disjunctive+relational+nonlinear"
cols = ["row_id", "cap", "admit", "router", "parent_probe", "reasons", "facts", "ops", "spec", "form"]
head = ["# ny-admission-census/v2", "# binary_sha256\t" + selfsha, "# build\tcommit=stub dirty=false features=stub",
        "# router_schema\tny-router/v1", "# r5_schema\tny-router-r5/v1", "# registry_version\t0123abcd",
        "# corpus_schema\tny-admission-census-corpus/v1", "# corpus_sha256\t" + hashlib.sha256(raw).hexdigest(),
        "# corpus_rows\t%d" % len(rows),
        "# host\t" + os.environ.get("SEALTEST_CENSUS_HOST", "class=other threads=1 device=none declared=true"),
        "# envelope\tops=%s specs=%s" % (ops, specs), "# probes\tmargin-row=structural/v1", "# caps\ttoy-tier",
        "# columns\t" + "\t".join(cols)]
body, n = [], 0
for rid, onnx, vnnlib, budget in rows:
    a = 1 if rx.search(vnnlib) else 0
    n += a
    body.append("\t".join([rid, "toy-tier", str(a), "1", "admit" if a else "decline:stub",
                           "-" if a else "parent:margin-row:decline:stub", "known/complete", "linear+relu",
                           "halfspace c=1 a=1 bx=0 -", "plain"]))
text = "\n".join(head + body) + "\n"
sha = hashlib.sha256(text.encode()).hexdigest()
open(out, "w").write(text + "# census_sha256\t" + sha + "\n")
print("CENSUS_DONE rc=0 rows=%d caps=1 admitted=toy-tier:%d router_admitted=toy-tier:%d census_sha256=%s out=%s"
      % (len(rows), n, len(rows), sha, out))
CENSUSEOF
  exit $?
fi
[ "${1:-} ${2:-}" = "vnncomp v1" ] || exit 2
C=$3; O=$4; V=$5; RES=$6; B=$7; CFG=$9; P=${11}
Y=$CFG/vnncomp26/$C.yaml
[ -f "$Y" ] || Y=$CFG/vnncomp26/toy_2026.yaml
D=$(dirname "$RES")/..
arm=$(basename "$CFG")
echo "$(basename "$V") $arm $P" >> "$D/calls.log"
env | grep -E '^(NY_|RAYON_|GPU_AVAILABLE)' >> "$D/verifier_env.log"
echo unknown > "$RES"
echo "Running VNN-COMP instance: category='$C' onnx='$O' vnnlib='$V' results='$RES' timeout=${B}s"
echo "Loading preset: $Y"
echo "[proof-profile] mode=$P source=cli" >&2
mode=${SEALTEST_MODE:-ok}
tier=0; grep -qE '^ *toy_tier: true' "$Y" && tier=1
[ "$mode" = dark ] && tier=0
RT=0
if grep -qE '^ *mode: shadow' "$Y"; then echo "[router] cap=toy-tier admit=admit ledger=open rung=unknown" >&2; RT=1; fi
[ $tier = 1 ] && echo "[toy-tier] armed=1 nodes=12 closed=$([ "$arm" = arm ] && echo 5 || echo 3)" >&2
v=$(sed -n "s/^; sealtest.* $arm=\([a-z]*\).*/\1/p" "$V" | head -1); v=${v:-timeout}
case "$mode" in
  unsat_negctl) [ "$arm" = negctl ] && v=unsat ;;
  flip) [ "$arm" = arm ] && [ "$v" = unsat ] && v=sat ;;
  stopfile) touch "$SEALTEST_STOPFILE" ;;
esac
# #certified-moat: `proofv=<v>` in the property comment is the armed Proof profile's verdict (profile divergence)
pv=$(sed -n "s/^; sealtest.* proofv=\([a-z]*\).*/\1/p" "$V" | head -1)
[ "$P" = proof ] && [ "$arm" = arm ] && [ -n "$pv" ] && v=$pv
echo "Result: $v"; echo "[stop] verdict=$v wall_s=1.5 budget_s=$B" >&2
echo "$v" > "$RES"
if [ "$v" = sat ]; then   # a witness the x86 replay can read (STUB_FAIL: one the organisers' checker rejects)
  printf '((X_0 0.25)\n (X_1 0.5))\n' >> "$RES"
  [ "$mode" = wrongsat ] && echo "; STUB_FAIL" >> "$RES"
fi
if [ "$P" = proof ] && [ "$v" = unsat ] && ! grep -q ' bundle=no' "$V"; then
  ck=$(sed -n "s/^; sealtest.* check=\([a-z_]*\).*/\1/p" "$V" | head -1)
  python3 - "$O" "$V" "$RES.proof.jsonl" "${ck:-verified}" <<'BEOF'
import hashlib, json, sys
o, v, out, ck = sys.argv[1:]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
hdr = {"kind": "header", "schema": "ny-proof-bundle/v2", "profile": "proof", "profile_source": "cli",
       "build": "NY_BUILD_STAMP_V1 commit=__COMMIT__ dirty=false", "onnx": {"path": o, "sha256": h(o)},
       "vnnlib": {"path": v, "sha256": h(v)}, "stub_check": ck}
lines = [hdr, {"kind": "verdict", "nodes": 1, "required": [0]}, {"kind": "end", "lines": 3}]
open(out, "w").write("".join(json.dumps(x) + "\n" for x in lines))
BEOF
  echo "[proof-artifact] artifact=written path=$RES.proof.jsonl nodes=1 schema=ny-proof-bundle/v2" >&2
fi
commit=__COMMIT__; [ "$mode" = wrongbin ] && commit=0000000000000000000000000000000000000000
tv=false; ts=default; grep -qE '^ *toy_tier: true' "$Y" && { tv=true; ts=config; }
lv=0; ls=default; grep -qE '^ *toy_level: 4' "$Y" && { lv=4; ls=config; }
ua=true; us=default; grep -qE '^ *upfront_attack: false' "$Y" && { ua=false; us=config; }
pa=true; ps=default; grep -qE '^ *vnncomp_post_bab_attack: false' "$Y" && { pa=false; ps=config; }
router=""
[ $RT = 1 ] && router=', "router": {"input": {"facts": {"status": "known", "ops": 9, "ops_complete": false, "spec": null}}}'
cat > "$RES.flight.json" <<EOF
{"build_id": "NY_BUILD_STAMP_V1 commit=$commit dirty=false",
 "ambient_env": {"OMP_NUM_THREADS": "$OMP_NUM_THREADS"},
 "levers": {"receipt": {"levers": [
   {"name": "NY_TOY_TIER", "source": "$ts", "value": $tv},
   {"name": "NY_TOY_LEVEL", "source": "$ls", "value": $lv},
   {"name": "NY_UPFRONT_ATTACK", "source": "$us", "value": $ua},
   {"name": "NY_POSTBAB_ATTACK", "source": "$ps", "value": $pa},
   {"name": "NY_UPFRONT_ATTACK_PROBE_FIRST", "source": "default", "value": false},
   {"name": "NY_FALSIFY_PORTFOLIO", "source": "default", "value": false},
   {"name": "NY_BNN_SIGN_SPACE", "source": "default", "value": false},
   {"name": "NY_BNN_STE_PGD", "source": "default", "value": false},
   {"name": "NY_ATTACK_POINT_FOLDED_TWIN", "source": "default", "value": false},
   {"name": "NY_POSTBAB_RESTARTS", "source": "default", "value": 0},
   {"name": "NY_PGD_DISJ_ODI", "source": "default", "value": false},
   {"name": "NY_PGD_DISJ_FOCUS_LANES", "source": "default", "value": 0},
   {"name": "NY_LOWDIM_LEAF_ATTACK_EVERY", "source": "default", "value": 0}]}},
 "events": [{"method": "preset_base", "status": "ran", "reason": "$Y"},
  {"method": "upfront_attack_policy", "reason": "schema=x master=disabled_by_preset route=disabled_by_preset lower=x"},
  {"method": "postbab_wrapper_policy", "reason": "schema=x disabled_by_preset=1 route=proof_tail_by_preset"}]$router}
EOF
'''

STUB_BOX = r'''#!/usr/bin/env bash
# stub ny_box.sh: `measure|build -- cmd...` runs cmd; prints the header line the real one prints
m=$1
[ "$m" = measure ] || [ "$m" = build ] || exit 2
shift; while [ "$#" -gt 0 ] && [ "$1" != -- ]; do shift; done; shift
echo "[ny-box] cmd: $*" >&2
if [ -n "${SEALTEST_BOX_REFUSE:-}" ] && [[ "$*" == *"$SEALTEST_BOX_REFUSE"* ]]; then
  echo "[ny-box] REFUSING TO MEASURE: stub lock wait expired" >&2; exit 75
fi
echo "$m $*" >> "${SEALTEST_BOXLOG:-/dev/null}"
"$@"; rc=$?
echo "[ny-box-provenance] mode=$m rc=$rc elapsed=0 load_before=0.1 load_after=0.1" >&2
exit $rc
'''

REGISTRY = '''# toy registry
schema: 1
scratch_root: __SCRATCH__
stages: {S1: built}
capabilities:

- id: toy-tier
  family: toy
  name: "toy tier"
  heads: [__HEAD__]
  keys:
    - {path: margin_row.toy_tier, type: bool, default: false, decl: NY_TOY_TIER, route: cell}
    - {path: margin_row.toy_level, type: "usize", default: 0, decl: NY_TOY_LEVEL, route: cell}
  markers: ["[toy-tier] armed=1 nodes="]
  seal:
    observable: ['__NEED__']
    s6: {closes: '__CLOSES__'}
  routable:
    kind: tier
    parent: margin-row
    envelope: []
    ledger: open
  record: {S1: done, S2: done, S3: done, S4: done, S5: open, S6: open, S7: open, S8: open, S9: open, S10: open, S11: open, SR: open}
  evidence:
    S6:
      logs: ["old/ab.log"]
      note: "an older hand-recorded A/B"
  gaps:
    - "toy"

# ======= another family
- id: other-cap
  family: other
  name: "untouched"
  keys:
    - {path: bab.other, type: bool, default: false}
  markers: ["[other] x="]
  record: {S1: done, S2: open}
'''


class World:
    """The stub world (see the module docstring)."""

    def __init__(self, tmp: str, two_commits: bool = False, extra_cats: dict[str, bool] | None = None,
                 unarmed_presets: bool = False, cert_tokens: dict[int, str] | None = None):
        """extra_cats: {category: armed in the pin's configs?} -- more 2026 categories carrying the SAME model and
        properties as CAT (the envelope's closure and the S7 unit coverage need a second unit). unarmed_presets: an
        unarmed extra category still has a preset (one that does not arm the capability), so an overlay can arm it
        (#admission-census: a census-admitted category the configs never armed). cert_tokens: {row: " check=<v>
        proofv=<v> bundle=no"} appended to a property's comment (#certified-moat: the stub pin's Proof bundle, its
        `proof-check` verdict and the armed Proof profile's verdict)."""
        self.tmp = tmp
        self.ny = os.path.join(tmp, "ny")
        self.repo = os.path.join(tmp, "repo")
        os.makedirs(os.path.join(self.ny, "scripts"))
        box = write(os.path.join(self.ny, "scripts/ny_box.sh"), STUB_BOX)
        os.chmod(box, 0o755)
        extra_cats = dict(extra_cats or {})
        man = {"schema": 1, "scored_categories": [], "banks": {}}
        truth = []
        for cat in [CAT] + list(extra_cats):
            bdir = os.path.join(self.ny, "benchmarks/vnncomp2026/benchmarks", cat, "1.0")
            os.makedirs(os.path.join(bdir, "onnx"))
            with open(os.path.join(bdir, "onnx/m.onnx"), "wb") as fh:
                fh.write(TOY_MODEL)
            inst, bank = [], []
            truth.append(f"Category 2026_{cat}:")
            for i, (bv, wall, tv, comment, thr) in enumerate(TOY_ROWS):
                write(os.path.join(bdir, f"vnnlib/p{i}.vnnlib"), toy_prop(comment + (cert_tokens or {}).get(i, ""), thr))
                inst.append(f"onnx/m.onnx,vnnlib/p{i}.vnnlib,10\n")
                bank.append(f"{cat},onnx/m.onnx,vnnlib/p{i}.vnnlib,0.0,{bv},{wall}\n")
                truth += [f"Row: ['m-p{i}', '1.0 (x)']", f"True Result: {tv}",
                          f"{i}: toolA score: 10, is_ver: {tv == 'unsat'}, is_fals: {tv == 'sat'}, is_fastest: False"]
            csvp = write(os.path.join(bdir, "instances.csv"), "".join(inst))
            bankp = write(os.path.join(self.ny, "reports/measured-2026", f"{cat}.csv"), "".join(bank))
            man["scored_categories"].append(cat)
            man["banks"][cat] = {
                "instances_csv": f"benchmarks/{cat}/1.0/instances.csv", "instances_csv_sha256": L.sha256_file(csvp),
                "bank_csv_sha256": L.sha256_file(bankp), "instances": [ln.strip().split(",") for ln in inst]}
        write(os.path.join(self.ny, "reports/measured-2026/bank_manifest.json"), json.dumps(man))
        os.makedirs(os.path.join(self.ny, "benchmarks/vnncomp2025/benchmarks"))
        write(os.path.join(self.ny, f"reports/measured-2026/{CAT}_artifacts/camp/witnesses/{CAT}/0003-p3.counterexample"),
              "sat\n((X_0 0.25)\n (X_1 0.5)\n (Y_0 0.75)\n (Y_1 0.125)\n (Y_2 0.0))\n")
        self.truth = write(os.path.join(tmp, "results26.txt"), "\n".join(truth) + "\n")
        # 2025 has no scored toy category, but a both-year plan (the S8 floor) needs its field results to parse
        self.truth25 = write(os.path.join(tmp, "results25.txt"), "Category 2025_unrelated_2025:\n")
        # the git repository: the pin's commit carries configs/ and the real preset schema source
        os.makedirs(self.repo)
        self.git("init", "-q")
        self.git("config", "user.email", "t@t")
        self.git("config", "user.name", "t")
        yaml_text = ("margin_row:\n  toy_tier: true\n  toy_level: 4\n  reserve_secs: 1.0\nattack:\n  pgd_order: middle\n"
                     "  ny_pgd_order_compat: upfront\nbab:\n  phase_budget:\n    post_bab_pgd_fraction: 0.1\n")
        write(os.path.join(self.repo, "configs/vnncomp26", f"{CAT}.yaml"), yaml_text)
        for cat, armed in extra_cats.items():
            if armed:
                write(os.path.join(self.repo, "configs/vnncomp26", f"{cat}.yaml"), yaml_text)
            elif unarmed_presets:
                write(os.path.join(self.repo, "configs/vnncomp26", f"{cat}.yaml"),
                      "margin_row:\n  reserve_secs: 1.0\n")
        write(os.path.join(self.repo, "configs/vnncomp25/other.yaml"), "a: 1\n")
        write(os.path.join(self.repo, "crates/ny-cli/src/preset/mod.rs"), open(SCHEMA_SRC).read())
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "pin commit")
        self.commit = self.git("rev-parse", "HEAD")
        if two_commits:
            write(os.path.join(self.repo, "configs/vnncomp26", f"{CAT}.yaml"), yaml_text + "  newer_key_the_pin_refuses: 1\n")
            self.git("commit", "-qam", "newer tree")
        self.pin = write(os.path.join(tmp, "stubpin_toy"), STUB_PIN.replace("__COMMIT__", self.commit))
        os.chmod(self.pin, 0o755)
        self.scratch = os.path.join(tmp, "scratch")
        os.makedirs(self.scratch)
        self.registry = write(os.path.join(tmp, "registry.yaml"), REGISTRY.replace("__HEAD__", self.commit)
                              .replace("__NEED__", NEED).replace("__CLOSES__", CLOSES).replace("__SCRATCH__", self.scratch))

    def git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", self.repo, *args], capture_output=True, text=True, check=True).stdout.strip()

    def plan(self, out: str = "out", **kw) -> dict:
        kw.setdefault("stages", S.STAGES)
        kw.setdefault("years", [2026, 2025])
        kw.setdefault("registry", self.registry)
        return S.build_plan("toy-tier", self.pin, os.path.join(self.tmp, out), repo=self.repo, ny_root=self.ny,
                            truth_paths={2026: self.truth, 2025: self.truth25}, **kw)

    def run(self, out: str, stage: str, shard: str = "1/1", mode: str = "ok", extra: dict | None = None,
            runner: str = "--run"):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("NY_", "SEAL", "M0_"))}
        env.update({"SEAL_OUT": os.path.join(self.tmp, out), "SEAL_NY": self.ny, "SEALTEST_MODE": mode})
        env.update(extra or {})
        p = subprocess.run(["bash", os.path.join(HERE, "seal_run.sh"), runner, "--stage", stage, "--shard", shard],
                           env=env, capture_output=True, text=True, timeout=600)
        tag = stage if shard == "1/1" else f"{stage}_{shard.replace('/', 'of')}"
        logp = os.path.join(self.tmp, out, f"seal_{tag}.log")
        log = open(logp).read() if os.path.isfile(logp) else ""
        done = [ln for ln in (p.stdout + log).splitlines() if ln.startswith("SEAL_DONE ")]
        return p.returncode, (done[-1] if done else ""), log + p.stdout

    def calls(self, out: str) -> list[str]:
        lines = []
        for p in sorted(__import__("glob").glob(os.path.join(self.tmp, out, "S*", "run_*", "calls.log"))):
            lines += open(p).read().splitlines()
        return lines


def toy_cap(**over) -> dict:
    reg = L.strict_load(REGISTRY.replace("__HEAD__", "c" * 40).replace("__NEED__", NEED).replace("__CLOSES__", CLOSES)
                        .replace("__SCRATCH__", "/tmp"))
    cap = reg["capabilities"][0]
    cap.update(over)
    return cap


# ---------------------------------------------------------------------------------------------------------
class PlanDerivation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = os.path.join(self.tmp.name, "cfg")
        write(os.path.join(self.cfg, "vnncomp26", f"{CAT}.yaml"), "margin_row:\n  toy_tier: true\n  toy_level: 4\n")
        write(os.path.join(self.cfg, "vnncomp26", "dark_2026.yaml"), "margin_row:\n  toy_tier: false\n")
        self.scored = {2026: [CAT, "dark_2026"]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_derive_arm_values_come_from_the_seal_block_then_the_configs_then_bool_keys(self):
        d = S.derive(toy_cap(), self.cfg, self.scored)
        self.assertEqual(d.arm, {"margin_row.toy_tier": True, "margin_row.toy_level": 4})
        self.assertIn("configs arm", d.sources["arm"])
        d = S.derive(toy_cap(seal={"observable": [NEED], "arm": {"margin_row.toy_level": 7}}), self.cfg, self.scored)
        self.assertEqual(d.arm, {"margin_row.toy_level": 7})
        write(os.path.join(self.cfg, "vnncomp26", f"{CAT}.yaml"), "margin_row:\n  toy_tier: false\n")
        d = S.derive(toy_cap(seal={"observable": [NEED], "categories": [CAT]}), self.cfg, self.scored)
        self.assertEqual(d.arm, {"margin_row.toy_tier": True})       # nothing armed in configs: every bool key true
        with self.assertRaises(S.SealError):
            S.derive(toy_cap(seal={"observable": [NEED], "arm": {"margin_row.not_a_key": 1}}), self.cfg, self.scored)
        with self.assertRaisesRegex(S.SealError, "keyless"):
            S.derive(toy_cap(keys=[]), self.cfg, self.scored)       # keyless: nothing an overlay can arm

    def test_tier_closing_alarms_apply_to_truly_sat_rows_only(self):
        # #certified-moat fix: an m0 tier's alarms are "the capability CLOSED this" markers (frozen-dual's
        # verdict=all_closed, the lane's UNSAT line): an emergency on a SAT row, the expected outcome on a banked-unsat
        # one. They were applied to EVERY row, so frozen-dual's S5 / S6 / S8 raised a false EMERGENCY on its first proof.
        reg = S.load_registry(os.path.join(REPO, "docs", "engine_program_capabilities.yaml"))
        d = S.derive(S.get_capability(reg, "frozen-dual-bab"), os.path.join(REPO, "configs"),
                     {2026: ["cifar100_2024", "tinyimagenet_2024"], 2025: ["cifar100_2024", "tinyimagenet_2024"]})
        self.assertIn(L.MR_UNSAT, d.sat_alarms)
        self.assertNotIn(L.MR_UNSAT, d.alarms)
        self.assertTrue(all(a in d.alarms for a in S.A.DEFECT_ALARMS))
        self.assertEqual(S.row_alarms(d, "unsat", "unsat"), d.alarms)
        self.assertEqual(S.row_alarms(d, "-", "timeout"), d.alarms)
        for truth, bank in (("sat", "-"), ("-", "sat"), ("sat", "sat")):
            self.assertIn(L.MR_UNSAT, S.row_alarms(d, truth, bank))
        with tempfile.TemporaryDirectory() as tmp:
            w = World(tmp)
            text = open(w.registry).read().replace("    s6: {closes: ", "    sat_alarms: ['toy_closed_on_a_sat_row']\n"
                                                                     "    s6: {closes: ", 1)
            write(w.registry, text)
            plan = w.plan(stages=["S5", "S7", "S8"])
            for st in ("S5", "S7", "S8"):
                for r in S.stage_rows(plan["out"], plan, st):
                    sat = r["kind"] in ("NEGCTL", "KNIFE") or "sat" in (r["truth"], r["bank"])
                    self.assertEqual("toy_closed_on_a_sat_row" in r["alarm"].split(L.ALARM_SEP), sat, (st, r["tag"]))
            self.assertEqual(plan["sat_alarms"], ["toy_closed_on_a_sat_row"])

    def test_derive_observable_precedence_flight_and_refusals(self):
        d = S.derive(toy_cap(), self.cfg, self.scored)
        self.assertEqual(d.need, [NEED])
        self.assertEqual(d.flight, ["lever:NY_TOY_TIER=true@config", "lever:NY_TOY_LEVEL=4@config"])
        self.assertTrue(all(a in d.alarms for a in S.A.DEFECT_ALARMS), "defect signals alarm on every row")
        cap = toy_cap(seal={}, evidence={"S5": {"arrivals": [{"marker": "[toy-tier] armed=1 nodes=3 (x)"}]}})
        d = S.derive(cap, self.cfg, self.scored)
        self.assertEqual(d.need, [r"\[toy-tier\] armed=1 nodes=3 \(x\)"])
        self.assertTrue(re.search(d.need[0], "[toy-tier] armed=1 nodes=3 (x)"))
        self.assertIn("arrival markers", d.sources["observable"])
        d = S.derive(toy_cap(seal={}), self.cfg, self.scored)
        self.assertIn("markers[0]", d.sources["observable"])
        with self.assertRaises(S.SealError):
            S.derive(toy_cap(seal={}, markers=[]), self.cfg, self.scored)
        with self.assertRaises(S.SealError):
            S.derive(toy_cap(seal={"observable": ["(unclosed"]}), self.cfg, self.scored)

    def test_derive_scope_is_the_live_armings_and_narrowing_cannot_widen_it(self):
        d = S.derive(toy_cap(), self.cfg, self.scored)
        self.assertEqual(d.categories, [CAT])                       # dark_2026 sets the key at its default
        with self.assertRaises(S.SealError):
            S.derive(toy_cap(), self.cfg, self.scored, only_categories=["dark_2026"])
        d = S.derive(toy_cap(seal={"observable": [NEED], "categories": [CAT, "dark_2026"]}), self.cfg, self.scored,
                     only_categories=["dark_2026"])
        self.assertEqual(d.categories, ["dark_2026"])
        with self.assertRaises(S.SealError):                        # armed nowhere and no scope given
            write(os.path.join(self.cfg, "vnncomp26", f"{CAT}.yaml"), "margin_row:\n  toy_tier: false\n")
            S.derive(toy_cap(), self.cfg, self.scored)

    def test_derive_takes_the_m0_tier_arming_and_folds_alt_paths(self):
        reg = S.load_registry(os.path.join(REPO, "docs/engine_program_capabilities.yaml"))
        cz = S.get_capability(reg, "cgan-child-zono")
        write(os.path.join(self.cfg, "vnncomp26", "cgan2026.yaml"), "bab:\n  alpha_crown:\n    iterations: 2\n")
        d = S.derive(cz, self.cfg, {2026: ["cgan2026"]})
        self.assertIn("m0 tier", d.sources["arm"])
        self.assertEqual(d.arm, {"solver.alpha_crown.cgan_child_zono_refresh.enabled": True,
                                 "solver.alpha_crown.cgan_child_zono_refresh.max_generators": 16384})
        doc = L.strict_load(open(os.path.join(self.cfg, "vnncomp26", "cgan2026.yaml")).read())
        self.assertEqual(S._choose_path(doc, d.keys["solver.alpha_crown.cgan_child_zono_refresh.enabled"]),
                         "bab.alpha_crown.cgan_child_zono_refresh.enabled")    # the file's own spelling
        bw = S.get_capability(reg, "bi-window-joint-refresh")
        write(os.path.join(self.cfg, "vnncomp26", "challenging_certified_training_2026.yaml"), "margin_row:\n  k: 1\n")
        d = S.derive(bw, self.cfg, {2026: ["challenging_certified_training_2026"]})
        self.assertEqual(d.arm["margin_row.bi_topk"], 8192)                 # the tier's companion knob
        self.assertIn("companion", d.sources["arm"])
        self.assertIn("lever:NY_MARGIN_ROW_JOINT_REFRESH=4@config", d.flight)  # both observables can engage

    def test_overlay_choose_path_prefers_the_file_s_existing_spelling(self):
        key = {"path": "solver.alpha_crown.k", "alt_paths": ["bab.alpha_crown.k"]}
        self.assertEqual(S._choose_path({"bab": {"alpha_crown": {"x": 1}}}, key), "bab.alpha_crown.k")
        self.assertEqual(S._choose_path({"solver": {"alpha_crown": {"k": 2}}, "bab": {}}, key), "solver.alpha_crown.k")
        self.assertEqual(S._choose_path({}, key), "solver.alpha_crown.k")


# ---------------------------------------------------------------------------------------------------------
class PlanEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan_writes_specs_overlays_and_a_sha_bound_plan(self):
        w = World(self.tmp.name)
        plan = w.plan(shards=2)
        out = plan["out"]
        self.assertEqual(plan["configs"]["source"], f"git:{w.commit}:configs")
        self.assertTrue(plan["configs"]["router_shadow"], "a routable capability gets the router's admission receipt")
        s5 = L.read_spec(os.path.join(out, plan["stages"]["S5"]["specs"]["1of1"]))
        self.assertEqual([r["index"] for r in s5], ["0", "1"])     # banked unsat, wall >= 0.2 x budget, cheapest
        for r in s5:
            self.assertEqual(r["need"], NEED)
            self.assertIn("lever:NY_TOY_TIER=true@config", r["flight"])
            self.assertIn("event:preset_base:^" + re.escape(os.path.join(out, "configs/arm")), r["flight"])
            self.assertEqual((r["reps_c"], r["reps_p"]), ("1", "0"))
        arm = open(os.path.join(out, "configs/arm/vnncomp26", f"{CAT}.yaml")).read()
        ctl = L.strict_load(open(os.path.join(out, "configs/ctl/vnncomp26", f"{CAT}.yaml")).read())
        neg = L.strict_load(open(os.path.join(out, "configs/negctl/vnncomp26", f"{CAT}.yaml")).read())
        self.assertIn("mode: shadow", arm)
        self.assertFalse(L.deep_has(ctl, "margin_row.toy_tier"), "the control arm drops the capability's keys")
        self.assertEqual((L.deep_get(neg, "attack.pgd_order"), L.deep_get(neg, "attack.upfront_attack"),
                          L.deep_get(neg, "margin_row.toy_tier")), ("skip", False, True))
        s7 = S.stage_rows(out, plan, "S7")
        self.assertTrue(all(r["truth"] == "sat" for r in s7))
        self.assertEqual(sorted({r["kind"] for r in s7}), ["KNIFE", "NEGCTL"])
        self.assertEqual(plan["stages"]["S7"]["knife"]["state"], "ran")
        s6 = S.stage_rows(out, plan, "S6")
        ctl_rows = [r for r in s6 if r["tag"].startswith("S6C_")]
        self.assertTrue(ctl_rows and all(r["need"] == "!" + NEED for r in ctl_rows), "control must never engage")
        reps = {r["group"].rsplit("/", 1)[-1]: r["reps_c"] for r in s6 if r["tag"].startswith("S6A_")}
        self.assertEqual((reps["gap"], reps["near"], reps["sat"], reps["uniform"]), ("3", "3", "1", "1"))
        s8 = S.stage_rows(out, plan, "S8")
        self.assertEqual(sorted(int(r["index"]) for r in s8), [0, 1, 2, 3, 4])   # banked sat/unsat + field-SAT p3
        self.assertTrue(all((r["reps_c"], r["reps_p"]) == ("3", "1") for r in s8))
        self.assertEqual(plan["stages"]["S8"]["shards"], 2)
        self.assertEqual(open(os.path.join(out, "plan.json.sha256")).read().strip(),
                         S.sha256_text(open(os.path.join(out, "plan.json")).read()))
        with self.assertRaises(S.SealError):
            w.plan(shards=2)                                         # pre-registered once

    def test_plan_takes_configs_from_the_pin_commit_not_the_newer_tree(self):
        w = World(self.tmp.name, two_commits=True)
        plan = w.plan(stages=["S5"])
        src = open(os.path.join(plan["configs"]["src"], "vnncomp26", f"{CAT}.yaml")).read()
        self.assertNotIn("newer_key_the_pin_refuses", src)
        self.assertEqual(plan["configs"]["tree"], w.git("rev-parse", f"{w.commit}:configs"))

    def test_plan_refuses_a_dirty_pin_a_missing_head_and_a_dark_marker(self):
        w = World(self.tmp.name)
        dirty = write(os.path.join(self.tmp.name, "stubpin_dirty"),
                      open(w.pin).read().replace("dirty=${SEALTEST_DIRTY:-false}", "dirty=true"))
        os.chmod(dirty, 0o755)
        with self.assertRaisesRegex(S.SealError, "dirty"):
            S.build_plan("toy-tier", dirty, os.path.join(self.tmp.name, "o1"), registry=w.registry, repo=w.repo,
                         ny_root=w.ny, truth_paths={2026: w.truth}, years=[2026], stages=["S5"])
        write(os.path.join(w.repo, "later.txt"), "x")
        w.git("add", "-A")
        w.git("commit", "-qm", "later")
        reg2 = write(os.path.join(self.tmp.name, "reg2.yaml"),
                     open(w.registry).read().replace(f"heads: [{w.commit}]", f"heads: [{w.git('rev-parse', 'HEAD')}]"))
        with self.assertRaisesRegex(S.SealError, "does not contain the capability"):
            S.build_plan("toy-tier", w.pin, os.path.join(self.tmp.name, "o2"), registry=reg2, repo=w.repo,
                         ny_root=w.ny, truth_paths={2026: w.truth}, years=[2026], stages=["S5"])
        dark = write(os.path.join(self.tmp.name, "stubpin_dark"), open(w.pin).read().replace("[toy-tier] armed=1", "x"))
        os.chmod(dark, 0o755)
        with self.assertRaisesRegex(S.SealError, "absent from the pin"):
            S.build_plan("toy-tier", dark, os.path.join(self.tmp.name, "o3"), registry=w.registry, repo=w.repo,
                         ny_root=w.ny, truth_paths={2026: w.truth}, years=[2026], stages=["S5"])

    def test_load_plan_refuses_an_edited_plan_or_overlay(self):
        w = World(self.tmp.name)
        plan = w.plan(stages=["S5"])
        out = plan["out"]
        S.load_plan(out)
        y = os.path.join(out, "configs/arm/vnncomp26", f"{CAT}.yaml")
        orig = open(y).read()
        write(y, orig + "# edited\n")
        with self.assertRaisesRegex(S.SealError, "changed or missing"):
            S.load_plan(out)
        write(y, orig)
        p = os.path.join(out, "plan.json")
        write(p, open(p).read().replace('"min_rows": 2', '"min_rows": 1'))
        with self.assertRaisesRegex(S.SealError, "was edited"):
            S.load_plan(out)

    def test_s7_controls_cover_every_admitted_unit_even_when_explicit_rows_name_one(self):
        # review: explicit control rows (an m0 tier's, all 2026) replaced the seeded field-SAT rows outright, so every
        # other unit stayed uncovered and a both-year tier capability (frozen-dual, fwd-root-screen) could never reach
        # S7 PASS
        w = World(self.tmp.name, extra_cats={"toy2_2026": True})
        reg = write(os.path.join(self.tmp.name, "reg_negctl.yaml"), open(w.registry).read().replace(
            "    s6: {closes:", "    negctl: {rows: [{id: n1, category: toy_2026, onnx: m.onnx, vnnlib: /p3.vnnlib, "
            "year: 2026}]}\n    s6: {closes:"))
        plan = w.plan(stages=["S7"], registry=reg)
        self.assertEqual(sorted(plan["categories"]), sorted([CAT, "toy2_2026"]))
        neg = [r for r in S.stage_rows(plan["out"], plan, "S7") if r["kind"] == "NEGCTL"]
        self.assertEqual(sorted({r["category"] for r in neg}), sorted([CAT, "toy2_2026"]))
        self.assertTrue(any("registry/n1" in r["note"] for r in neg))
        self.assertTrue(any("uncovered" in r["note"] for r in neg if r["category"] == "toy2_2026"))


# ---------------------------------------------------------------------------------------------------------
class Shards(unittest.TestCase):
    def test_shard_partition_is_deterministic_balanced_and_keeps_an_instance_together(self):
        rows = []
        for i in range(12):
            for arm in ("A", "C"):
                rows.append(L.spec_row(tag=f"S6{arm}_x_{i:04d}", kind="AB", tier="-", year=2026, category="c",
                                       onnx="/m.onnx", vnnlib=f"/p{i}.vnnlib", budget=10 * (i + 1),
                                       configs_dir="/c", preset="p.yaml", truth="-", bank="-", index=i, reps_c=3,
                                       reps_p=0, need="-", alarm="-", flight="-", group="g", note="-"))
        a = S.partition(rows, 3)
        self.assertEqual(a, S.partition(list(rows), 3))
        for i in range(12):
            self.assertEqual(a[f"S6A_x_{i:04d}"], a[f"S6C_x_{i:04d}"], "one instance, one shard")
        load = {k: sum(int(r["budget"]) * 3 for r in rows if a[r["tag"]] == k) for k in (1, 2, 3)}
        self.assertLessEqual(max(load.values()) - min(load.values()), 2 * 120 * 3)
        self.assertEqual(sorted(set(a.values())), [1, 2, 3])
        with self.assertRaises(S.SealError):
            S.parse_shard("3/2")


# ---------------------------------------------------------------------------------------------------------
def jrow(tag: str, kind: str = "AB", truth: str = "-", bank: str = "-", index: int = 0, reps_c: int = 1,
         reps_p: int = 0, vnnlib: str = "/p0.vnnlib", group: str = "seal/S5/arm") -> dict:
    return L.spec_row(tag=tag, kind=kind, tier="-", year=2026, category="toy", onnx="/m.onnx", vnnlib=vnnlib,
                      budget=10, configs_dir="/c", preset="p.yaml", truth=truth, bank=bank, index=index,
                      reps_c=reps_c, reps_p=reps_p, need="-", alarm="-", flight="-", group=group, note="-")


def J(tag, prof, rep, verdict, status="OK", outcome="n/a", reasons=()):
    return L.Judgement(tag, prof, rep, verdict, status, outcome, "1.5", list(reasons))


def runs_for(rows, engaged=True, commit="c" * 40, closes=None):
    raw = "Result: x\n" + ("[toy-tier] armed=1 nodes=12 closed=%d\n" % (closes or 0) if engaged else "")
    out = {}
    for r in rows:
        for prof, k in S.M0J.schedule(r):
            out[(r["tag"], prof, k)] = {"raw": raw, "raw_path": f"/raw/{r['tag']}.{prof}.r{k}.raw",
                                        "flight": {"build_id": f"NY_BUILD_STAMP_V1 commit={commit}"}}
    return out


def toy_plan(**over) -> dict:
    p = {"capability": "toy", "observable": {"need": [NEED], "arrival_markers": ["[toy-tier] armed=1 nodes="]},
         "pin": {"commit": "c" * 40}, "stages": {"S5": {"min_rows": 2}, "S7": {"knife": {"state": "ran"}}},
         "years": [2026, 2025], "categories": ["toy"], "scope_categories": ["toy"], "units": ["2026/toy"],
         "s6": {"closes": CLOSES}}
    p.update(over)
    return p


def judge(plan, stage, rows, lines, runs, other_rows=None, other_lines=None):
    all_rows = {r["tag"]: r for r in rows + (other_rows or [])}
    return S.judge_stage(plan, stage, rows, lines, all_rows, lines + (other_lines or []), runs)


class StageJudge(unittest.TestCase):
    def test_s5_passes_only_with_the_observable_on_n_rows_and_is_void_otherwise(self):
        rows = [jrow("S5_a", index=0, vnnlib="/p0.vnnlib"), jrow("S5_b", index=1, vnnlib="/p1.vnnlib")]
        lines = [J("S5_a", "competition", 1, "unsat"), J("S5_b", "competition", 1, "unsat")]
        rec = judge(toy_plan(), "S5", rows, lines, runs_for(rows))
        self.assertEqual((rec["status"], rec["rc"], rec["state"]), ("PASS", 0, "done"))
        rec = judge(toy_plan(), "S5", rows, lines, runs_for(rows, engaged=False))
        self.assertEqual((rec["status"], rec["rc"]), ("VOID", 5))
        self.assertIn("0 of the 2 rows", rec["reasons"][0])
        rec = judge(toy_plan(), "S5", rows, lines[:1], runs_for(rows))       # the second run never ran
        self.assertEqual(rec["status"], "VOID")
        self.assertEqual(rec["counts"]["notrun"], 1)

    def test_a_run_whose_flight_names_another_binary_is_void(self):
        rows = [jrow("S5_a"), jrow("S5_b", index=1, vnnlib="/p1.vnnlib")]
        lines = [J("S5_a", "competition", 1, "unsat"), J("S5_b", "competition", 1, "unsat")]
        rec = judge(toy_plan(), "S5", rows, lines, runs_for(rows, commit="d" * 40))
        self.assertEqual(rec["status"], "VOID")
        self.assertIn("wrong-binary", " ".join(rec["void_reasons"]))
        runs = runs_for(rows)
        for m in runs.values():
            m["flight"] = {"levers": {}}                                  # review: a flight with no build stamp
        rec = judge(toy_plan(), "S5", rows, lines, runs)
        self.assertEqual(rec["status"], "VOID")
        self.assertIn("no flight build_id", " ".join(r for x in rec["rows"] for run in x["runs"] for r in run["reasons"]))

    def test_s7_any_unsat_on_a_control_is_an_emergency_even_when_dark(self):
        rows = [jrow("S7N_a", kind="NEGCTL", truth="sat", reps_p=1, group="seal/S7/negctl")]
        lines = [J("S7N_a", "competition", 1, "timeout"),
                 J("S7N_a", "proof", 1, "unsat", "EMERGENCY", reasons=["TRUTH-CONTRADICTION(unsat on a sat row)"])]
        rec = judge(toy_plan(), "S7", rows, lines, runs_for(rows, engaged=False))
        self.assertEqual((rec["status"], rec["rc"]), ("EMERGENCY", 3))
        self.assertIn("TRUTH-CONTRADICTION", rec["emergencies"][0])

    def test_s7_is_partial_without_knife_evidence_or_a_controlled_unit(self):
        rows = [jrow("S7N_a", kind="NEGCTL", truth="sat", reps_p=1, group="seal/S7/negctl")]
        lines = [J("S7N_a", "competition", 1, "timeout"), J("S7N_a", "proof", 1, "timeout")]
        self.assertEqual(judge(toy_plan(), "S7", rows, lines, runs_for(rows))["status"], "PASS")
        plan = toy_plan(stages={"S7": {"knife": {"state": "no-witness"}}})
        rec = judge(plan, "S7", rows, lines, runs_for(rows))
        self.assertEqual((rec["status"], rec["rc"]), ("PARTIAL", 4))
        plan = toy_plan(units=["2026/toy", "2025/toy"])
        rec = judge(plan, "S7", rows, lines, runs_for(rows))
        self.assertEqual(rec["status"], "PARTIAL")
        self.assertEqual(rec["uncovered_units"], ["2025/toy"])
        rec = judge(toy_plan(), "S7", rows, lines[:1] + [J("S7N_a", "proof", 1, "timeout", "VOID",
                                                         reasons=["need-absent:x"])], runs_for(rows))
        self.assertEqual(rec["status"], "VOID")

    def test_s6_counts_absolute_conversions_and_losses_by_two_of_three(self):
        rows, lines = [], []
        plan = toy_plan()
        verdicts = {0: (["unsat", "unsat", "timeout"], ["timeout"] * 3),     # converted (2/3 vs 0/3)
                    1: (["unsat", "timeout", "timeout"], ["timeout"] * 3),   # 1/3: not a conversion
                    2: (["timeout"] * 3, ["unsat"] * 3)}                     # lost
        for i, (av, cv) in verdicts.items():
            for arm, vs in (("A", av), ("C", cv)):
                t = f"S6{arm}_toy_{i:04d}"
                rows.append(jrow(t, index=i, vnnlib=f"/p{i}.vnnlib", reps_c=3,
                                 group=f"seal/S6/{'arm' if arm == 'A' else 'ctl'}/gap"))
                lines += [J(t, "competition", k + 1, v) for k, v in enumerate(vs)]
        runs = runs_for([r for r in rows if r["tag"].startswith("S6A")], closes=5)
        runs.update(runs_for([r for r in rows if r["tag"].startswith("S6C")], engaged=False))
        rec = judge(plan, "S6", rows, lines, runs)
        d = rec["decision"]
        self.assertEqual(rec["status"], "PASS")
        self.assertEqual(([x["index"] for x in d["converted"]], [x["index"] for x in d["lost"]]), (["0"], ["2"]))
        self.assertEqual(d["decision"], "REJECT")
        self.assertEqual(d["closes_absolute"]["arm"], 5 * 9)
        self.assertNotIn("ratio", json.dumps(d).replace("closure ratio", ""))
        rec = judge(plan, "S6", rows, lines, runs_for(rows, engaged=False))
        self.assertEqual(rec["status"], "VOID")                       # the armed arm never engaged: measured nothing

    def test_s6_credits_a_conversion_only_where_the_arm_engaged(self):
        # review: a conversion on a row where the armed arm never showed the observable ran the same code as the
        # control -- run-to-run noise -- and still made the decision ADOPT
        rows, lines = [], []
        for i, (av, cv) in {0: (["unsat"] * 3, ["timeout"] * 3), 1: (["unsat"] * 3, ["unsat"] * 3)}.items():
            for arm, vs in (("A", av), ("C", cv)):
                t = f"S6{arm}_toy_{i:04d}"
                rows.append(jrow(t, index=i, vnnlib=f"/p{i}.vnnlib", reps_c=3,
                                 group=f"seal/S6/{'arm' if arm == 'A' else 'ctl'}/gap"))
                lines += [J(t, "competition", k + 1, v) for k, v in enumerate(vs)]
        runs = runs_for([r for r in rows if r["tag"] == "S6A_toy_0001"])               # engaged only on row 1
        runs.update(runs_for([r for r in rows if r["tag"] != "S6A_toy_0001"], engaged=False))
        rec = judge(toy_plan(), "S6", rows, lines, runs)
        d = rec["decision"]
        self.assertEqual(rec["status"], "PASS")
        self.assertEqual([x["index"] for x in d["converted"]], ["0"])
        self.assertEqual([x["index"] for x in d["converted_unengaged"]], ["0"])
        self.assertEqual(d["decision"], "NULL", "an unengaged conversion is noise, never an ADOPT")
        runs.update(runs_for([r for r in rows if r["tag"] == "S6A_toy_0000"]))        # now it engaged there
        self.assertEqual(judge(toy_plan(), "S6", rows, lines, runs)["decision"]["decision"], "ADOPT")

    def test_s8_flip_and_profile_split_are_emergencies_lost_is_partial_dark_is_void(self):
        r = jrow("S8_a", kind="MOAT", truth="unsat", bank="unsat", reps_c=3, reps_p=1, group="seal/S8/2026")
        ok = [J("S8_a", "competition", k, "unsat", outcome="held") for k in (1, 2, 3)] + \
             [J("S8_a", "proof", 1, "unsat", outcome="held")]
        self.assertEqual(judge(toy_plan(), "S8", [r], ok, runs_for([r]))["status"], "PASS")
        self.assertEqual(judge(toy_plan(), "S8", [r], ok, runs_for([r], engaged=False))["status"], "VOID")
        split = [J("S8_a", "competition", 1, "unsat", outcome="held")] + \
                [J("S8_a", "competition", k, "timeout", outcome="lost") for k in (2, 3)] + \
                [J("S8_a", "proof", 1, "unsat", outcome="held")]
        rec = judge(toy_plan(), "S8", [r], split, runs_for([r]))
        self.assertEqual(rec["status"], "EMERGENCY")
        self.assertIn("PROFILE-DECIDED-SPLIT", " ".join(rec["emergencies"]))
        flip = ok[:3] + [J("S8_a", "proof", 1, "sat", "EMERGENCY", "flip", ["BANK-FLIP(banked unsat, now sat)"])]
        self.assertEqual(judge(toy_plan(), "S8", [r], flip, runs_for([r]))["status"], "EMERGENCY")
        r2 = dict(r, truth="-")
        lost = [J("S8_a", "competition", k, "timeout", outcome="lost") for k in (1, 2, 3)] + \
               [J("S8_a", "proof", 1, "timeout", outcome="lost")]
        rec = judge(toy_plan(), "S8", [r2], lost, runs_for([r2]))
        self.assertEqual((rec["status"], rec["rc"]), ("PARTIAL", 4))

    def test_a_sat_unsat_pair_across_stages_is_an_instance_flip_emergency(self):
        a = jrow("S5_a", truth="-", bank="-")
        b = jrow("S8_a", kind="MOAT", reps_c=1, group="seal/S8/2026")
        la = [J("S5_a", "competition", 1, "unsat")]
        lb = [J("S8_a", "competition", 1, "sat")]
        rec = judge(toy_plan(stages={"S5": {"min_rows": 1}}), "S5", [a], la, runs_for([a]), [b], lb)
        self.assertEqual(rec["status"], "EMERGENCY")
        self.assertIn("INSTANCE-FLIP", rec["emergencies"][0])

    def test_s8_below_its_floor_is_partial_never_pass(self):
        # review: --s8-reps-c 1 made '2 of 3' vacuous, --s8-reps-p 0 left no Proof verdict to reproduce, --years 2026
        # dropped a year and --categories narrowed the scope -- and each still judged PASS, and so sealed
        def s8(reps_c=3, reps_p=1):
            r = jrow("S8_a", kind="MOAT", truth="unsat", bank="unsat", reps_c=reps_c, reps_p=reps_p,
                     group="seal/S8/2026")
            ok = [J("S8_a", "competition", k, "unsat", outcome="held") for k in range(1, reps_c + 1)] + \
                 [J("S8_a", "proof", k, "unsat", outcome="held") for k in range(1, reps_p + 1)]
            return r, ok
        r, ok = s8()
        self.assertEqual(judge(toy_plan(), "S8", [r], ok, runs_for([r]))["status"], "PASS")   # the floor itself
        for (rc_, rp_), plan, why in (((1, 1), toy_plan(), "reps"), ((3, 0), toy_plan(), "reps"),
                                      ((3, 1), toy_plan(years=[2026]), "not both years"),
                                      ((3, 1), toy_plan(scope_categories=["toy", "toy2"]), "narrowed"),
                                      ((3, 1), {k: v for k, v in toy_plan().items() if k != "scope_categories"},
                                       "no admission-scope record")):
            r, ok = s8(rc_, rp_)
            rec = judge(plan, "S8", [r], ok, runs_for([r]))
            self.assertEqual((rec["status"], rec["rc"], rec["state"]), ("PARTIAL", 4, "partial"), (rc_, rp_, why))
            self.assertIn(why, " ".join(rec["reasons"]))

    def test_a_zero_row_arrival_is_refused_at_plan_and_void_at_judge(self):
        # review: --arrival-rows 0 planned an S5 with no row, and the judge's `0 >= 0` called it PASS
        rec = judge(toy_plan(stages={"S5": {"min_rows": 0}}), "S5", [], [], {})
        self.assertEqual((rec["status"], rec["rc"]), ("VOID", 5))
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            with self.assertRaisesRegex(S.SealError, "at least 1 arrival row"):
                w.plan(stages=["S5"], opts={"arrival_rows": 0})


# ---------------------------------------------------------------------------------------------------------
class Facts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_onnx_scan_classifies_ops_with_constant_propagation(self):
        p = os.path.join(self.tmp.name, "m.onnx")
        with open(p, "wb") as fh:
            fh.write(TOY_MODEL)
        nodes, inits = S.onnx_nodes(p)
        self.assertEqual([n[0] for n in nodes], ["Gemm", "Relu", "Reshape", "Gemm"])
        self.assertEqual(S.classify_ops(nodes, inits), (["linear", "relu"], []))
        sub = onnx_graph([onnx_node("Sigmoid", ["a"], ["b"])], [])
        m2 = onnx_model([onnx_node("Conv", ["X", "W"], ["c"]), onnx_node("BatchNormalization", ["c", "s", "b"], ["d"]),
                         onnx_node("Add", ["d", "X"], ["e"]),                    # two dynamic operands: residual
                         onnx_node("Add", ["e", "bias"], ["f"]),                 # constant operand: linear
                         onnx_node("Shape", ["f"], ["sh"]), onnx_node("Mul", ["sh", "two"], ["sh2"]),  # constant math
                         onnx_node("MatMul", ["f", "f"], ["g"]),                 # bilinear
                         onnx_node("Erf", ["g"], ["h"]),                         # unknown: other
                         onnx_node("If", ["h"], ["i"], sub)], ["W", "s", "b", "bias", "two"])
        p2 = os.path.join(self.tmp.name, "m2.onnx")
        with open(p2, "wb") as fh:
            fh.write(m2)
        kinds, unknown = S.classify_ops(*S.onnx_nodes(p2))
        self.assertEqual(kinds, ["linear", "conv", "sigmoid", "bn", "residual", "bilinear", "other"])
        self.assertEqual(unknown, ["Erf", "If"])
        with open(os.path.join(self.tmp.name, "bad.onnx"), "wb") as fh:
            fh.write(b"\x3a\xff")
        with self.assertRaises(S.SealError):
            S.onnx_nodes(os.path.join(self.tmp.name, "bad.onnx"))

    def test_vnnlib_spec_archetypes(self):
        d = self.tmp.name
        self.assertEqual(S.vnnlib_spec(write(os.path.join(d, "r.vnnlib"), toy_prop("x")))[0], "robustness")
        self.assertEqual(S.vnnlib_spec(write(os.path.join(d, "h.vnnlib"), toy_prop("x", threshold=True)))[0],
                         "halfspace")
        disj = toy_prop("x", threshold=True).replace("(and (>= Y_0 0.5))", "(and (>= Y_0 0.5)) (and (<= Y_2 0.1))")
        self.assertEqual(S.vnnlib_spec(write(os.path.join(d, "d.vnnlib"), disj))[0], "disjunctive")
        two = ("(vnnlib-version <2.0>)\n(declare-network f (declare-input X float32 [1]) (declare-output Y float32 [1]))\n"
               "(declare-network g (declare-input A float32 [1]) (declare-output B float32 [1]))\n(assert (>= X[0] 0.0))\n")
        self.assertEqual(S.vnnlib_spec(write(os.path.join(d, "two.vnnlib"), two))[0], "relational")

    def test_opkind_and_spec_tokens_match_the_router_source(self):
        # A tracked file of this repository: its absence is repository drift, and a skip would let the two token
        # tables go unchecked without a word (TEST_CONFORMANCE.md: no skip constructs; ny-test-utils source_policy).
        self.assertTrue(os.path.isfile(DECL_SRC), f"router source absent at {DECL_SRC}: the OpKind and spec tokens "
                                                  "cannot be checked against it")
        src = open(DECL_SRC).read()
        order = re.search(r"pub const ALL: \[Self; 16\] = \[(.*?)\];", src, re.S).group(1)
        names = re.findall(r"Self::(\w+)", order)
        body = src[src.index("impl OpKind"):]
        tok = dict(re.findall(r"Self::(\w+) => \"(\w+)\"", body[:body.index("pub struct OpSet")]))
        self.assertEqual(tuple(tok[n] for n in names), S.OPKIND_TOKENS)
        sb = src[src.index("impl SpecArchetype"):]
        stok = re.findall(r"Self::\w+ => \"(\w+)\"", sb[:sb.index("pub struct SpecSet")])
        self.assertEqual(tuple(stok), S.SPEC_TOKENS)
        self.assertEqual(S.decode_opset(9), ["linear", "relu"])

    def test_envelope_is_the_engaged_rows_op_union_and_never_seals_other(self):
        def row(tag, ops, spec, eng=True, admitted=None, bops=None):
            runs = [{"binary_facts": {"ops": bops, "ops_complete": False, "spec": None}}] if bops else []
            return {"tag": tag, "engaged": eng, "admitted": admitted, "runs": runs,
                    "facts": {"ops": ops, "spec": spec, "unknown_ops": ["Erf"] if "other" in ops else [], "errors": []}}
        recs = [{"rows": [row("a", ["linear", "relu"], "robustness"), row("b", ["conv", "relu", "other"], "robustness"),
                          row("c", ["sigmoid"], "halfspace", eng=False), row("d", ["tanh"], "halfspace", admitted=False)]},
                {"rows": [row("e", ["linear", "bn"], "robustness", bops=["linear"])]}]
        env = S.envelope_of(recs)
        self.assertEqual((env["ops"], env["specs"]), (["linear", "conv", "relu", "bn"], ["robustness"]))
        self.assertEqual([x["tag"] for x in env["rows_with_unsealable_other"]], ["b"])
        self.assertTrue(S.envelope_covered(["conv", "relu"], ["robustness"], recs)[0])
        ok, why = S.envelope_covered(["conv", "sigmoid"], ["robustness"], recs)
        self.assertFalse(ok)
        self.assertIn("sigmoid", why)
        self.assertFalse(S.envelope_covered([], [], recs)[0])
        recs[1]["rows"][0]["runs"][0]["binary_facts"]["ops"] = ["linear", "softmax"]   # the binary saw more
        self.assertFalse(S.envelope_covered(["linear"], ["robustness"], recs)[0])


# ---------------------------------------------------------------------------------------------------------
class Runner(unittest.TestCase):
    """seal_run.sh --run end to end through the REAL m0 runner, against the stub world."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.w = World(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_runner_seals_a_capability_end_to_end_and_the_record_derives(self):
        w = self.w
        plan = w.plan(shards=2)
        rc, done, log = w.run("out", "S6", "1/2")
        self.assertEqual(rc, 97, done)                                  # S6 is gated on S5 + S7
        self.assertIn("gate", log)
        self.assertEqual(w.calls("out"), [])
        rc, done, log = w.run("out", "S5")
        self.assertEqual(rc, 0, log[-3000:])
        self.assertTrue(done.startswith("SEAL_DONE rc=0 WORST=0 stage=S5 shard=1/1 status=PASS "), done)
        self.assertEqual(done.count(" rc="), 1, "a DONE line carries exactly one rc=")
        self.assertIn("loader ok", log)
        self.assertIn("resolver negative control ok", log)
        self.assertTrue(os.path.isfile(os.path.join(plan["out"], "PIN_LOCK")))
        rc, done, log = w.run("out", "S7")
        self.assertEqual(rc, 0, log[-3000:])
        for k in ("1/2", "2/2"):
            rc, done, log = w.run("out", "S6", k)
            self.assertIn(rc, (0, 5), log[-3000:])
        s6 = S.read_json(os.path.join(plan["out"], "S6", "STAGE.json"))
        self.assertEqual(s6["status"], "PASS", s6["reasons"])
        self.assertEqual(s6["decision"]["decision"], "ADOPT")
        self.assertEqual([x["index"] for x in s6["decision"]["converted"]], ["1"])
        for k in ("1/2", "2/2"):
            rc, done, log = w.run("out", "S8", k)
        self.assertEqual(rc, 0, log[-3000:])
        s8 = S.read_json(os.path.join(plan["out"], "S8", "STAGE.json"))
        self.assertEqual(s8["status"], "PASS", s8["reasons"])
        before = open(w.registry).read()
        with self.assertRaisesRegex(S.SealError, "already recorded"):
            S.record_into_registry(w.registry, plan["out"])            # the hand-recorded S6 is never overwritten silently
        rep = S.record_into_registry(w.registry, plan["out"], replace=True)
        after = open(w.registry).read()
        self.assertEqual(rep["record"], {"S5": "done", "S6": "done", "S7": "done", "S8": "done", "SR": "done"})
        self.assertEqual(rep["envelope"]["envelope"], ["linear", "relu"])
        self.assertEqual(rep["envelope"]["envelope_specs"], ["robustness", "halfspace"])
        b_lines, a_lines = before.splitlines(), after.splitlines()
        cut_b = b_lines.index("# ======= another family")
        cut_a = a_lines.index("# ======= another family")
        self.assertEqual(b_lines[cut_b:], a_lines[cut_a:], "only the capability's own block may change")
        reg = L.strict_load(after)
        cap = reg["capabilities"][0]
        self.assertTrue(cap["evidence"]["S5"]["seal"].endswith("S5/STAGE.json"))
        # the auditor's own functions derive S5..S8 and SR.seal from the records
        fake = types.SimpleNamespace(path=lambda p: p, errors=[], _seal_records={},
                                     git=types.SimpleNamespace(is_ancestor=lambda a, b: a == w.commit and b == w.commit))
        stages = {}
        for st in ("S6", "S7", "S8"):
            stages[st] = EPA.Auditor.audit_seal_stage(fake, cap, st, cap["evidence"][st])
            self.assertEqual(stages[st].state, "done", stages[st].note)
        sr = EPA.Auditor.audit_routable(fake, cap, stages)
        self.assertEqual(sr.state, "done", sr.note)
        cap_bad = dict(cap, routable=dict(cap["routable"], envelope=["linear", "relu", "conv"]))
        self.assertEqual(EPA.Auditor.audit_routable(fake, cap_bad, stages).state, "partial")
        fake2 = types.SimpleNamespace(path=lambda p: p, errors=[], _seal_records={},
                                      git=types.SimpleNamespace(is_ancestor=lambda a, b: False))
        self.assertEqual(EPA.Auditor.audit_seal_stage(fake2, cap, "S8", cap["evidence"]["S8"]).state, "open",
                         "a record from a pin that lacks the capability's heads derives nothing")
        # a tampered judged.tsv no longer re-derives
        jp = sorted(S.read_json(os.path.join(plan["out"], "S8", "STAGE.json"))["judged"])[0]
        with open(jp, "a") as fh:
            fh.write("S8_26_toy_2026_0000\tproof\t9\tunsat\tOK\theld\t1.0\t-\n")
        state, note, _ = S.verify_stage_record(os.path.join(plan["out"], "S8", "STAGE.json"), "toy-tier", "S8")
        self.assertEqual(state, "open", note)

    def test_runner_emergency_writes_the_emergency_file_and_every_stage_stops(self):
        w = self.w
        plan = w.plan(stages=["S5", "S7", "S8"], shards=2)
        rc, done, log = w.run("out", "S5")
        self.assertEqual(rc, 0, log[-3000:])
        rc, done, log = w.run("out", "S7", mode="unsat_negctl")
        self.assertEqual(rc, 3, log[-3000:])
        self.assertTrue(done.startswith("SEAL_DONE rc=3 WORST=3 stage=S7 "), done)
        self.assertIn("status=EMERGENCY", done)
        self.assertTrue(os.path.isfile(os.path.join(plan["out"], "EMERGENCY")))
        n = len(w.calls("out"))
        for st, sh in (("S5", "1/1"), ("S8", "1/2"), ("S8", "2/2")):
            rc, done, _ = w.run("out", st, sh)
            self.assertEqual(rc, 3, done)
            self.assertIn("prior-emergency", done)
        self.assertEqual(len(w.calls("out")), n, "nothing measured after the emergency")
        before = open(w.registry).read()
        with self.assertRaisesRegex(S.SealError, "EMERGENCY exists"):
            S.record_into_registry(w.registry, plan["out"])              # nor recorded (the S5 PASS included)
        self.assertEqual(open(w.registry).read(), before)

    def test_runner_stops_before_its_next_run_when_another_shard_records_an_emergency(self):
        w = self.w
        plan = w.plan(stages=["S5"], opts={"arrival_rows": 3})
        stop = os.path.join(plan["out"], "EMERGENCY")
        rc, done, log = w.run("out", "S5", mode="stopfile", extra={"SEALTEST_STOPFILE": stop})
        self.assertEqual(rc, 3, log[-3000:])
        self.assertIn("stopped_by=", done)
        self.assertEqual(len(w.calls("out")), 1, "the m0 runner re-reads the EMERGENCY file before every run")

    def test_runner_void_and_wrong_binary_are_never_a_pass(self):
        w = self.w
        w.plan(stages=["S5"])
        rc, done, log = w.run("out", "S5", mode="dark")
        self.assertEqual(rc, 5, log[-3000:])
        self.assertIn("status=VOID", done)
        w.plan("out2", stages=["S5"])
        rc, done, log = w.run("out2", "S5", mode="wrongbin")
        self.assertEqual(rc, 5, log[-3000:])
        self.assertIn("wrong-binary", log)

    def test_runner_gates_every_later_stage_on_a_void_arrival(self):
        w = self.w
        w.plan(stages=["S5", "S7", "S8"])
        rc, done, _ = w.run("out", "S5", mode="dark")
        self.assertEqual(rc, 5, done)
        n = len(w.calls("out"))
        for st in ("S7", "S8"):
            rc, done, log = w.run("out", st)
            self.assertEqual(rc, 97, done)
            self.assertIn("gate S5", log)
        self.assertEqual(len(w.calls("out")), n, "a stage gated on a VOID arrival measures nothing")

    def test_runner_refuses_ny_env_a_wrong_pin_and_a_lax_loader_before_measuring(self):
        w = self.w
        plan = w.plan(stages=["S5"])
        rc, done, _ = w.run("out", "S5", extra={"NY_SOME_LEVER": "1"})
        self.assertEqual(rc, 95, done)
        rc, done, _ = w.run("out", "S5", extra={"SEALTEST_LOADER": "lax"})
        self.assertEqual(rc, 94, done)
        self.assertIn("loader-negative-control-PASSED", done)
        rc, done, _ = w.run("out", "S5", extra={"SEALTEST_ALWAYS_PRESET": os.path.join(
            plan["out"], "configs/arm/vnncomp26", f"{CAT}.yaml")})
        self.assertEqual(rc, 94, done)
        self.assertIn("resolver-negative-control", done)
        with open(w.pin, "a") as fh:
            fh.write("# the pin changed after the plan\n")
        rc, done, _ = w.run("out", "S5")
        self.assertEqual(rc, 91, done)
        self.assertEqual(w.calls("out"), [])

    def _seal_all(self, out: str = "out", shards: int = 2) -> None:
        w = self.w
        for st, sh in [("S5", "1/1"), ("S7", "1/1")] + [(st, f"{k}/{shards}") for st in ("S6", "S8")
                                                         for k in range(1, shards + 1)]:
            rc, done, log = w.run(out, st, sh)
            self.assertIn(rc, (0, 5, 10), log[-3000:])
        for st in S.STAGES:
            rec = S.read_json(os.path.join(self.tmp.name, out, st, "STAGE.json"))
            self.assertEqual(rec["status"], "PASS", (st, rec["reasons"]))

    def test_runner_records_the_emergency_before_judging_so_a_failing_judge_still_stops_every_shard(self):
        # review: after the m0 runner stopped on an emergency (exit 3) a judge that then failed made the runner die 96
        # with NO EMERGENCY file, so the stage's other shards kept measuring past a soundness emergency
        w = self.w
        plan = w.plan(stages=["S5", "S7", "S8"], shards=2)
        for st in ("S5", "S7"):
            rc, done, log = w.run("out", st)
            self.assertEqual(rc, 0, log[-3000:])
        shard = next(k for k in (1, 2) if any(r["bank"] == "unsat" for r in L.read_spec(
            os.path.join(plan["out"], plan["stages"]["S8"]["specs"][f"{k}of2"]))))
        os.makedirs(os.path.join(plan["out"], "S8", "STAGE.json"))     # the judge's write fails (a directory)
        rc, done, log = w.run("out", "S8", f"{shard}/2", mode="flip")   # a banked unsat comes back sat: BANK-FLIP
        self.assertEqual(rc, 3, log[-3000:])
        self.assertTrue(done.startswith("SEAL_DONE rc=3 WORST=3 stage=S8 "), done)
        self.assertIn("judge=failed", done)
        self.assertEqual(done.count(" rc="), 1, done)
        self.assertTrue(os.path.isfile(os.path.join(plan["out"], "EMERGENCY")), "recorded before the judge ran")
        n = len(w.calls("out"))
        rc, done, _ = w.run("out", "S8", f"{3 - shard}/2")               # the stage's other shard
        self.assertEqual(rc, 3, done)
        self.assertIn("prior-emergency", done)
        self.assertEqual(len(w.calls("out")), n, "the other shard measured nothing after the emergency")

    def test_a_hand_edited_record_or_judged_tsv_derives_nothing(self):
        # review: the auditor re-judged a record only on status/counts/scheduled/emergencies, so a STAGE.json whose
        # rows' facts or engagement were edited (the envelope and SR.seal read exactly those) still derived done; and
        # judged.tsv was trusted, so an edited verdict re-recorded with `seal judge-stage` derived done too
        w = self.w
        plan = w.plan(shards=2)
        self._seal_all()
        out = plan["out"]
        for st in S.STAGES:
            state, note, _ = S.verify_stage_record(os.path.join(out, st, "STAGE.json"), "toy-tier", st)
            self.assertEqual(state, "done", note)                        # positive control: untouched records derive
        p8 = os.path.join(out, "S8", "STAGE.json")
        orig = open(p8).read()
        rec = json.loads(orig)
        row = next(r for r in rec["rows"] if r["engaged"])
        row["facts"]["ops"] = row["facts"]["ops"] + ["softmax"]          # widen the envelope by hand
        S.write_json_atomic(p8, rec)
        state, note, _ = S.verify_stage_record(p8, "toy-tier", "S8")
        self.assertEqual(state, "open", note)
        self.assertIn("rows", note)
        reg = L.strict_load(open(w.registry).read())
        cap = dict(reg["capabilities"][0], evidence={st: {"seal": os.path.join(out, st, "STAGE.json")}
                                                    for st in S.STAGES})
        cap["routable"] = dict(cap["routable"], envelope=["linear", "relu", "softmax"],
                               envelope_specs=["robustness", "halfspace"])
        fake = types.SimpleNamespace(path=lambda q: q, errors=[], _seal_records={},
                                     git=types.SimpleNamespace(is_ancestor=lambda a, b: True))
        stages = {st: EPA.Auditor.audit_seal_stage(fake, cap, st, cap["evidence"][st]) for st in ("S7", "S8")}
        self.assertEqual(stages["S8"].state, "open", stages["S8"].note)
        self.assertNotEqual(EPA.Auditor.audit_routable(fake, cap, stages).state, "done")
        write(p8, orig)
        rec = json.loads(orig)
        rec["rows"][0]["engaged"] = not rec["rows"][0]["engaged"]
        S.write_json_atomic(p8, rec)
        self.assertEqual(S.verify_stage_record(p8, "toy-tier", "S8")[0], "open", "an edited engagement flag")
        write(p8, orig)
        # judged.tsv: a control's decided verdict rewritten as undecided fabricates a conversion; re-recorded with the
        # pipeline's own judge-stage it carries a fresh sha, so only the artefacts can refuse it
        s6 = S.read_json(os.path.join(out, "S6", "STAGE.json"))
        self.assertEqual(len(s6["decision"]["converted"]), 1)
        edited = 0
        for jp in sorted(s6["judged"]):
            text = open(jp).read()
            new = re.sub(r"^(S6C_26toy2026_0004\t\w+\t\d+\t)unsat\tOK\theld\t", r"\1timeout\tOK\tlost\t", text,
                         flags=re.M)
            edited += text != new
            write(jp, new)
        self.assertEqual(edited, 1)
        again = S.judge_stage_from_out(out, "S6")
        self.assertEqual(len(again["decision"]["converted"]), 2, "the edit did fabricate a conversion")
        state, note, _ = S.verify_stage_record(os.path.join(out, "S6", "STAGE.json"), "toy-tier", "S6")
        self.assertEqual(state, "open", note)
        self.assertIn("do not re-derive from their run artefacts", note)

    def test_sr_seal_refuses_an_envelope_whose_closure_admits_rows_the_moat_never_ran(self):
        # review: S8 ran the capability's categories while the router routes a sealed capability on ANY row inside its
        # envelope, so a moat over toy_2026 sealed an envelope that also admits toy2_2026's (never moated) rows
        self.tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp2.cleanup)
        self.w = w = World(self.tmp2.name, extra_cats={"toy2_2026": False})   # same shape, not armed: out of scope
        self.tmp = self.tmp2
        plan = w.plan(shards=2)
        self.assertEqual(plan["categories"], [CAT])
        self._seal_all()
        recs = {st: S.verify_stage_record(os.path.join(plan["out"], st, "STAGE.json"), "toy-tier", st)
                for st in S.STAGES}
        self.assertTrue(all(v[0] == "done" for v in recs.values()), {k: v[1] for k, v in recs.items()})
        rep = S.record_into_registry(w.registry, plan["out"], replace=True)
        self.assertEqual(rep["record"], {"S5": "done", "S6": "done", "S7": "done", "S8": "done"}, rep)   # no SR
        self.assertIsNone(rep["envelope"], "no envelope is written for the router to read")
        self.assertIn("toy2_2026", rep["sr_seal"])
        cap = L.strict_load(open(w.registry).read())["capabilities"][0]
        self.assertEqual(cap["routable"]["envelope"], [])
        env = S.envelope_of([recs["S7"][2], recs["S8"][2]])
        ok, why = S.envelope_covered(env["ops"], env["specs"], [recs["S7"][2], recs["S8"][2]])
        self.assertTrue(ok, why)                                          # inside its evidence rows ...
        ok, why = S.sr_seal(env["ops"], env["specs"], {"S7": recs["S7"][2], "S8": recs["S8"][2]})
        self.assertFalse(ok, why)                                         # ... but its closure is not moated
        self.assertIn("2026/toy2_2026", why)
        fake = types.SimpleNamespace(path=lambda q: q, errors=[], _seal_records={},
                                     git=types.SimpleNamespace(is_ancestor=lambda a, b: True))
        stages = {st: EPA.Auditor.audit_seal_stage(fake, cap, st, cap["evidence"][st]) for st in ("S7", "S8")}
        cap2 = dict(cap, routable=dict(cap["routable"], envelope=env["ops"], envelope_specs=env["specs"]))
        sr = EPA.Auditor.audit_routable(fake, cap2, stages)
        self.assertEqual(sr.state, "partial", sr.note)
        self.assertIn("S8 never ran", sr.note)

    def test_runner_refuses_box_knobs_that_relax_quiescence_or_the_lock(self):
        # review: step 0 excluded every NY_BOX_* from its env refusal, so a raised NY_BOX_LOAD_MAX (or another lock
        # dir, a test /proc/stat, an inherited fcntl lock) reached ny_box.sh and confounded every arm
        w = self.w
        w.plan(stages=["S5"])
        for knob in ("NY_BOX_LOAD_MAX=50", "NY_BOX_LOCK_DIR=/tmp/elsewhere", "NY_BOX_PROC_STAT=/dev/null",
                     "NY_BOX_FCNTL_LOCKED=1"):
            k, v = knob.split("=", 1)
            rc, done, _ = w.run("out", "S5", extra={k: v})
            self.assertEqual(rc, 95, (knob, done))
            self.assertIn("box-env", done)
        self.assertEqual(w.calls("out"), [])
        rc, done, log = w.run("out", "S5", extra={"NY_BOX_WAIT_LOCK": "60", "NY_BOX_WAIT_QUIET": "60"})
        self.assertEqual(rc, 0, log[-3000:])                              # the waits themselves are the runner's

    def test_runner_binds_one_memory_cap_per_seal_out(self):
        # review: nothing bound the cgroup cap across launches, so an A/B or moat could straddle two caps (a relaunch
        # with another SEAL_MEMORY_MAX, a --run outside the unit) -- CLAUDE.md: the cap's VALUE changes verdicts
        w = self.w
        plan = w.plan(stages=["S5", "S7"])
        rc, done, log = w.run("out", "S5")
        self.assertEqual(rc, 0, log[-3000:])
        lock = os.path.join(plan["out"], "MEM_LOCK")
        have = open(lock).read().split()[0]
        self.assertIn("memory cap bound", log)
        write(lock, ("103079215104" if have == "max" else "max") + " 96G earlier-launch\n")
        n = len(w.calls("out"))
        rc, done, _ = w.run("out", "S7")
        self.assertEqual(rc, 86, done)
        self.assertIn("mem-lock", done)
        self.assertEqual(len(w.calls("out")), n)

    def test_runner_refuses_a_shard_the_plan_did_not_make_and_an_empty_shard(self):
        w = self.w
        w.plan(stages=["S5", "S8"], shards=8)
        rc, done, _ = w.run("out", "S5", "1/2")
        self.assertEqual(rc, 97, done)
        empty = [k for k in range(1, 9) if not os.path.isfile(os.path.join(self.tmp.name, "out", "specs",
                                                                           f"S8_{k}of8.tsv"))]
        self.assertTrue(empty)


# ---------------------------------------------------------------------------------------------------------
class RegistryWriteBack(unittest.TestCase):
    def test_record_floors_rise_and_never_fall(self):
        old = {"S5": "done", "S6": "partial", "S7": "open", "SR": "open"}
        self.assertEqual(S.raise_floors(old, {"S5": "partial", "S6": "done", "S7": "partial", "S8": "open",
                                              "SR": "done"}), {"S6": "done", "S7": "partial", "SR": "done"})

    def test_capability_edit_touches_only_its_block_and_reparses_to_the_intended_change(self):
        text = REGISTRY.replace("__HEAD__", "c" * 40).replace("__NEED__", NEED).replace("__CLOSES__", CLOSES) \
            .replace("__SCRATCH__", "/tmp")
        ev = {"S7": {"seal": "/o/S7/STAGE.json", "logs": ["/o/S7/run_1of1/judged.tsv"], "note": "it's \"quoted\""},
              "S6": {"seal": "/o/S6/STAGE.json", "logs": [], "note": "replaced"}}
        new = S.edit_capability(text, "toy-tier", ev, {"S7": "done", "SR": "done"},
                                {"envelope": ["linear", "relu"], "envelope_specs": ["robustness"],
                                 "envelope_evidence": "/o/ENVELOPE.json"})
        old_doc, new_doc = L.strict_load(text), L.strict_load(new)
        cap = new_doc["capabilities"][0]
        self.assertEqual(cap["evidence"]["S7"], ev["S7"])
        self.assertEqual(cap["evidence"]["S6"], ev["S6"])
        self.assertEqual((cap["record"]["S7"], cap["record"]["SR"], cap["record"]["S1"]), ("done", "done", "done"))
        self.assertEqual(cap["routable"]["envelope"], ["linear", "relu"])
        self.assertEqual(new_doc["capabilities"][1], old_doc["capabilities"][1])
        self.assertEqual({k: v for k, v in new_doc.items() if k != "capabilities"},
                         {k: v for k, v in old_doc.items() if k != "capabilities"})
        new2 = S.edit_capability(text.replace("  evidence:\n    S6:\n      logs: [\"old/ab.log\"]\n"
                                              "      note: \"an older hand-recorded A/B\"\n", ""), "toy-tier", ev, {}, None)
        self.assertEqual(L.strict_load(new2)["capabilities"][0]["evidence"], ev)   # an evidence block is created
        with self.assertRaises(S.SealError):
            S.edit_capability(text, "no-such-cap", ev, {}, None)

    def test_record_refuses_and_writes_nothing_when_the_edit_reparses_differently(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            plan = w.plan(stages=["S5"])
            rc, done, log = w.run("out", "S5")
            self.assertEqual(rc, 0, log[-2000:])
            before = open(w.registry).read()
            real = S.edit_capability
            try:
                S.edit_capability = lambda *a, **k: real(*a, **k).replace("name: \"untouched\"", "name: \"touched\"")
                with self.assertRaisesRegex(S.SealError, "verification failed"):
                    S.record_into_registry(w.registry, plan["out"])
            finally:
                S.edit_capability = real
            self.assertEqual(open(w.registry).read(), before)
            rep = S.record_into_registry(w.registry, plan["out"], dry_run=True)
            self.assertEqual((rep["written"], rep["record"]), (False, {"S5": "done"}))
            self.assertEqual(open(w.registry).read(), before, "a dry run writes nothing")


# ---------------------------------------------------------------------------------------------------------
CENSUS_SRC = os.path.join(REPO, "crates/ny-cli/src/router/census.rs")
ROUTE_SRC = os.path.join(REPO, "crates/ny-core/src/router/route.rs")


def census_text(entries: dict, admit, binary_sha: str, corpus_sha: str, caps=("toy-tier",), envelope=None) -> str:
    """A census in the binary's format (crates/ny-cli/src/router/census.rs), `admit(entry)` deciding each row."""
    ops, specs = envelope or ("+".join(S.CENSUS_WIDEST[0]), "+".join(S.CENSUS_WIDEST[1]))
    head = [f"# {S.CENSUS_SCHEMA}", f"# binary_sha256\t{binary_sha}", "# build\tcommit=test dirty=false features=-",
            "# router_schema\tny-router/v1", "# r5_schema\tny-router-r5/v1", "# registry_version\t0123abcd",
            f"# corpus_schema\t{S.CORPUS_SCHEMA}", f"# corpus_sha256\t{corpus_sha}", f"# corpus_rows\t{len(entries)}",
            "# host\tclass=other threads=1 device=none declared=true", f"# envelope\tops={ops} specs={specs}",
            "# probes\tmargin-row=structural/v1", "# caps\t" + ",".join(caps),
            "# columns\t" + "\t".join(S.CENSUS_COLUMNS)]
    body = []
    for e in entries.values():
        for cap in caps:
            a = int(bool(admit(e)))
            body.append("\t".join([e["row_id"], cap, str(a), "1", "admit" if a else "decline:x",
                                   "-" if a else "parent:margin-row:decline:x", "known/complete", "linear+relu",
                                   "halfspace c=1 a=1 bx=0 -", "plain"]))
    text = "\n".join(head + body) + "\n"
    return text + f"# census_sha256\t{hashlib.sha256(text.encode()).hexdigest()}\n"


class Census(unittest.TestCase):
    """#admission-census: S8's rows are the router's own admission decisions (the binary's census), and SR.seal
    checks the census, never the offline envelope closure, when a plan carries one."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.w = World(self.tmp.name, extra_cats={"toy2_2026": False}, unarmed_presets=True)
        self.truth = {2026: self.w.truth, 2025: self.w.truth25}

    def tearDown(self):
        self.tmp.cleanup()

    def entries(self):
        return S.census_corpus(self.w.ny, os.path.join(self.w.repo, "configs"), self.truth)

    def given(self, name: str, admit, binary_sha: str | None = None, corpus_sha: str | None = None, **kw) -> str:
        _, entries = self.entries()
        text = census_text(entries, admit, binary_sha or L.sha256_file(self.w.pin),
                           corpus_sha or hashlib.sha256(S.corpus_text(entries).encode()).hexdigest(), **kw)
        return write(os.path.join(self.tmp.name, name), text)

    def test_census_schema_and_columns_match_the_binary_source(self):
        src = open(CENSUS_SRC).read()
        self.assertIn(f'pub(crate) const CORPUS_SCHEMA: &str = "{S.CORPUS_SCHEMA}";', src)
        cols = re.search(r"pub\(crate\) const COLUMNS: \[&str; (\d+)\] = \[(.*?)\];", src, re.S)
        self.assertIsNotNone(cols, "COLUMNS not found in census.rs: the check would be VOID")
        self.assertEqual(tuple(re.findall(r'"([a-z_]+)"', cols.group(2))), S.CENSUS_COLUMNS)
        self.assertEqual(int(cols.group(1)), len(S.CENSUS_COLUMNS))
        self.assertIn(f'pub const CENSUS_SCHEMA: &str = "{S.CENSUS_SCHEMA}";', open(ROUTE_SRC).read())
        self.assertNotIn("other", S.CENSUS_WIDEST[0])
        self.assertEqual(set(S.CENSUS_WIDEST[0]) | {"other"}, set(S.OPKIND_TOKENS))

    def test_census_corpus_is_every_moat_row_of_every_category_once(self):
        rows, entries = self.entries()
        self.assertEqual(len(rows), 10, "5 moat rows (banked sat/unsat + field-SAT) per category, both categories")
        self.assertEqual(len(entries), 10)
        self.assertEqual([e["row_id"] for e in entries.values()], [f"r{i:05d}" for i in range(1, 11)])
        cats = {e["rows"][0].split("/")[1] for e in entries.values()}
        self.assertEqual(cats, {CAT, "toy2_2026"}, "every scored category, armed or not")
        text = S.corpus_text(entries)
        self.assertTrue(text.startswith(f"# {S.CORPUS_SCHEMA}\n"))
        self.assertEqual(len(text.strip().splitlines()), 11)
        self.assertNotIn(CAT, "".join(e["row_id"] for e in entries.values()), "row ids are opaque")

    def test_parse_census_verifies_its_trailer_and_shape(self):
        path = self.given("c.tsv", lambda e: True)
        c = S.parse_census(path)
        self.assertEqual(c["caps"], ["toy-tier"])
        self.assertEqual(len(c["rows"]), 10)
        text = open(path).read()
        edited = write(os.path.join(self.tmp.name, "edited.tsv"),
                       text.replace("\ttoy-tier\t1\t", "\ttoy-tier\t0\t", 1))
        with self.assertRaisesRegex(S.SealError, "edited"):
            S.parse_census(edited)
        lines = text.splitlines()
        body = [ln for ln in lines[:-1] if not ln.startswith("# ")]
        short = "\n".join([ln for ln in lines[:-1] if ln.startswith("# ")] + body[1:]) + "\n"
        short += f"# census_sha256\t{hashlib.sha256(short.encode()).hexdigest()}\n"
        with self.assertRaisesRegex(S.SealError, "corpus_rows"):
            S.parse_census(write(os.path.join(self.tmp.name, "short.tsv"), short))
        with self.assertRaisesRegex(S.SealError, "trailer"):
            S.parse_census(write(os.path.join(self.tmp.name, "cut.tsv"), "\n".join(lines[:-1]) + "\n"))

    def test_plan_with_a_census_moats_exactly_the_census_admitted_rows(self):
        admitted = re.compile(rf"{CAT}/1.0/vnnlib/p[012]\.|toy2_2026/1.0/vnnlib/p3\.")
        path = self.given("c.tsv", lambda e: admitted.search(e["vnnlib"]))
        plan = self.w.plan(census=path)
        c = plan["census"]
        self.assertEqual(c["categories"], sorted([CAT, "toy2_2026"]))
        self.assertEqual(c["counts"]["admit"], 4)
        self.assertEqual(plan["categories"], [CAT], "S5-S7 keep the capability's own scope")
        s8 = [r for k in plan["stages"]["S8"]["specs"].values() for r in L.read_spec(os.path.join(plan["out"], k))]
        got = sorted(os.path.relpath(r["vnnlib"], self.w.ny) for r in s8)
        self.assertEqual(got, sorted(f"benchmarks/vnncomp2026/benchmarks/{c_}/1.0/vnnlib/p{i}.vnnlib"
                                     for c_, i in ((CAT, 0), (CAT, 1), (CAT, 2), ("toy2_2026", 3))))
        armed = open(os.path.join(plan["out"], "configs/arm/vnncomp26/toy2_2026.yaml")).read()
        self.assertIn("toy_tier: true", armed, "a census-admitted category is armed by the overlay")
        for rel, sha in plan["files"].items():
            if rel.startswith("census/"):
                self.assertEqual(L.sha256_file(os.path.join(plan["out"], rel)), sha)
        self.assertEqual(set(plan["files"]) & {c["tsv"], c["corpus"], c["map"]}, {c["tsv"], c["corpus"], c["map"]})
        # the same plan through the pin's own census (the stub's); a narrowed --categories cannot shrink S8
        os.environ["SEALTEST_CENSUS_ADMIT"] = admitted.pattern
        try:
            plan2 = self.w.plan("out_run", census="run", categories=[CAT])
        finally:
            os.environ.pop("SEALTEST_CENSUS_ADMIT", None)
        s8b = [r for k in plan2["stages"]["S8"]["specs"].values() for r in L.read_spec(os.path.join(plan2["out"], k))]
        self.assertEqual(sorted(os.path.relpath(r["vnnlib"], self.w.ny) for r in s8b), got)
        self.assertIn("run on the pin copy", plan2["census"]["source"])
        self.assertEqual(S.s8_floor_violations(plan2, s8b), [], "the census is S8's scope, not --categories")
        self.assertTrue(S.s8_floor_violations(plan2, s8b[1:]), "a row set missing an admitted instance is not THE moat")

    def test_plan_refuses_a_census_that_does_not_bind(self):
        for name, kw, why in (("bin.tsv", {"binary_sha": "0" * 64}, "not the pin"),
                              ("corpus.tsv", {"corpus_sha": "0" * 64}, "not the current moat corpus"),
                              ("narrow.tsv", {"envelope": ("linear+relu", "robustness")}, "narrower"),
                              ("caps.tsv", {"caps": ("other-cap",)}, "no decision")):
            path = self.given(name, lambda e: True, **kw)
            with self.assertRaisesRegex(S.SealError, why):
                self.w.plan(f"o_{name}", census=path)
        with self.assertRaisesRegex(S.SealError, "admits no moat row"):
            self.w.plan("o_none", census=self.given("none.tsv", lambda e: False))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        w2 = World(tmp.name, extra_cats={"toy2_2026": False})          # toy2 has no preset at all
        _, entries = S.census_corpus(w2.ny, os.path.join(w2.repo, "configs"), {2026: w2.truth, 2025: w2.truth25})
        text = census_text(entries, lambda e: True, L.sha256_file(w2.pin),
                           hashlib.sha256(S.corpus_text(entries).encode()).hexdigest())
        with self.assertRaisesRegex(S.SealError, "no preset resolves for category toy2_2026"):
            w2.plan(census=write(os.path.join(tmp.name, "c.tsv"), text))

    def test_sr_seal_checks_the_census_reruns_a_stale_one_and_refuses_the_envelope_path(self):
        w = self.w
        plan = w.plan(shards=2, census=self.given("c.tsv", lambda e: CAT in e["vnnlib"] or "/p0." in e["vnnlib"]))
        self.assertEqual(plan["census"]["categories"], sorted([CAT, "toy2_2026"]))
        Runner._seal_all(self)                                              # S5, S7, S6, S8 all PASS
        recs = {st: S.verify_stage_record(os.path.join(plan["out"], st, "STAGE.json"), "toy-tier", st)
                for st in ("S7", "S8")}
        self.assertTrue(all(v[0] == "done" for v in recs.values()), {k: v[1] for k, v in recs.items()})
        records = {st: v[2] for st, v in recs.items()}
        env = S.envelope_of(list(records.values()))
        ok, why = S.sr_seal(env["ops"], env["specs"], records)
        self.assertTrue(ok, why)
        self.assertIn("every census-admitted moat row", why)
        # the offline envelope closure refuses a plan that carries a census
        clo = S.envelope_closure(env["ops"], env["specs"], S.load_plan(plan["out"], verify_files=False), set())
        self.assertFalse(clo["ok"])
        self.assertIn("admission census", clo["why"])
        # `seal record` writes SR through the census, and the auditor derives it the same way
        rep = S.record_into_registry(w.registry, plan["out"], replace=True)
        self.assertEqual(rep["record"].get("SR"), "done", rep)
        cap = L.strict_load(open(w.registry).read())["capabilities"][0]
        fake = types.SimpleNamespace(path=lambda q: q, errors=[], _seal_records={},
                                     git=types.SimpleNamespace(is_ancestor=lambda a, b: True))
        stages = {st: EPA.Auditor.audit_seal_stage(fake, cap, st, cap["evidence"][st]) for st in ("S7", "S8")}
        self.assertEqual(EPA.Auditor.audit_routable(fake, cap, stages).state, "done")
        # a STALE census: the moat corpus moved (toy2's p5 is now a banked unsat: a new moat row exists)
        bank = os.path.join(w.ny, "reports/measured-2026/toy2_2026.csv")
        text = open(bank).read()
        man = os.path.join(w.ny, "reports/measured-2026/bank_manifest.json")
        md = json.load(open(man))

        def rebank(p5: str, p4: str = "unsat") -> None:
            write(bank, text.replace("vnnlib/p5.vnnlib,0.0,timeout,", f"vnnlib/p5.vnnlib,0.0,{p5},")
                  .replace("vnnlib/p4.vnnlib,0.0,unsat,", f"vnnlib/p4.vnnlib,0.0,{p4},"))
            md["banks"]["toy2_2026"]["bank_csv_sha256"] = L.sha256_file(bank)
            write(man, json.dumps(md))

        rebank("unsat")
        ok, why = S.sr_seal(env["ops"], env["specs"], records)
        self.assertFalse(ok, why)
        self.assertIn("stale", why)
        self.assertEqual(EPA.Auditor.audit_routable(fake, cap, stages).state, "partial")
        # re-run on the pin: the fresh census admits toy2's new p5 row, which S8 never ran
        os.environ["SEALTEST_CENSUS_ADMIT"] = rf"{CAT}/|/p0\.|toy2_2026/1.0/vnnlib/p5\."
        try:
            ok, why = S.sr_seal(env["ops"], env["specs"], records, S.census_rerun_on_pin(None))
        finally:
            os.environ.pop("SEALTEST_CENSUS_ADMIT", None)
        self.assertFalse(ok, why)
        self.assertIn("S8 never ran", why)
        self.assertIn("2026/toy2_2026", why)
        self.assertIn("re-run on the pin", why)
        # ... and a re-run over another corpus (p4 leaves the moat, p5 stays) that admits nothing S8 did not run
        rebank("unsat", p4="timeout")
        os.environ["SEALTEST_CENSUS_ADMIT"] = rf"{CAT}/|/p0\."
        try:
            ok, why = S.sr_seal(env["ops"], env["specs"], records, S.census_rerun_on_pin(None))
        finally:
            os.environ.pop("SEALTEST_CENSUS_ADMIT", None)
        self.assertTrue(ok, why)
        self.assertIn("census re-run on the pin", why)
        # a census edited after the plan derives nothing
        tsv = os.path.join(plan["out"], plan["census"]["tsv"])
        with open(tsv, "a") as fh:
            fh.write("# tampered\n")
        ok, why = S.sr_seal(env["ops"], env["specs"], records, S.census_rerun_on_pin(None))
        self.assertFalse(ok, why)
        self.assertIn("does not verify", why)


    def test_sr_seal_refuses_a_rerun_on_another_host_and_an_empty_census(self):
        """Review (#admission-census): a stale census re-run on ANOTHER host (the router's precondition reads the
        host: a device, a thread floor) does not bind, and a re-run census that admits NOTHING is no closure. Before
        the review fix both passed: the first because the host was recorded and never compared, the second
        vacuously ("every census-admitted moat row (0 instance(s)) ran in S8")."""
        w = self.w
        plan = w.plan(shards=2, census=self.given("c.tsv", lambda e: CAT in e["vnnlib"] or "/p0." in e["vnnlib"]))
        Runner._seal_all(self)
        recs = {st: S.verify_stage_record(os.path.join(plan["out"], st, "STAGE.json"), "toy-tier", st)
                for st in ("S7", "S8")}
        records = {st: v[2] for st, v in recs.items()}
        env = S.envelope_of(list(records.values()))
        ok, why = S.sr_seal(env["ops"], env["specs"], records)
        self.assertTrue(ok, why)
        bank = os.path.join(w.ny, "reports/measured-2026/toy2_2026.csv")
        text = open(bank).read()
        man = os.path.join(w.ny, "reports/measured-2026/bank_manifest.json")
        md = json.load(open(man))

        def rebank(p5: str, p4: str = "unsat") -> None:
            write(bank, text.replace("vnnlib/p5.vnnlib,0.0,timeout,", f"vnnlib/p5.vnnlib,0.0,{p5},")
                  .replace("vnnlib/p4.vnnlib,0.0,unsat,", f"vnnlib/p4.vnnlib,0.0,{p4},"))
            md["banks"]["toy2_2026"]["bank_csv_sha256"] = L.sha256_file(bank)
            write(man, json.dumps(md))

        def rerun(admit: str, host: str | None = None) -> tuple[bool, str]:
            os.environ["SEALTEST_CENSUS_ADMIT"] = admit
            if host is not None:
                os.environ["SEALTEST_CENSUS_HOST"] = host
            try:
                return S.sr_seal(env["ops"], env["specs"], records, S.census_rerun_on_pin(None))
            finally:
                os.environ.pop("SEALTEST_CENSUS_ADMIT", None)
                os.environ.pop("SEALTEST_CENSUS_HOST", None)

        # the corpus moved; the re-run on ANOTHER host admits only rows S8 ran: refused on the host, not passed
        rebank("unsat")
        ok, why = rerun(rf"{CAT}/|/p0\.", host="class=gb10 threads=20 device=search declared=false")
        self.assertFalse(ok, why)
        self.assertIn("not the plan's census host", why)
        # the corpus moved again; the re-run (same host) admits NOTHING: no closure, never a vacuous pass
        rebank("unsat", p4="timeout")
        ok, why = rerun(r"(?!)")
        self.assertFalse(ok, why)
        self.assertIn("admits no moat row", why)
        # positive control: the same corpus state with a census that admits only S8's rows, on the plan's host
        rebank("timeout", p4="timeout")
        ok, why = rerun(rf"{CAT}/|/p0\.")
        self.assertTrue(ok, why)
        self.assertIn("census re-run on the pin", why)

# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# #certified-moat: a stub x86 replay rig (the scratchpad/x86replay + x86parity tools' interfaces, no emulation)
# ---------------------------------------------------------------------------------------------------------

STUB_QEMU = r"""#!/usr/bin/env bash
# stub qemu-x86_64: drop -cpu/-L, run the rest natively
while [ $# -gt 0 ]; do case "$1" in -cpu|-L) shift 2;; *) break;; esac; done
exec "$@"
"""

STUB_CHECKV1 = r"""import argparse, hashlib, json, os, sys
ap = argparse.ArgumentParser()
for k in ("--scoring", "--onnx", "--vnnlib", "--ce", "--feeds", "--outputs", "--json-out"):
    ap.add_argument(k)
a = ap.parse_args()
ce = open(a.ce).read()
if "STUB_RIGERR" in ce:
    sys.exit(1)
res = "spec_not_violated" if "STUB_FAIL" in ce else "correct"
feed = hashlib.sha256(ce.encode()).hexdigest()
for p, t in ((a.feeds, feed), (a.outputs, "Y:" + feed)):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(t)
rec = {"mode": "checkv1", "result": res, "ce": a.ce, "ce_sha256": hashlib.sha256(open(a.ce, "rb").read()).hexdigest(),
       "onnx": a.onnx, "vnnlib": a.vnnlib, "scoring": os.path.realpath(a.scoring),
       "env": {"machine": os.environ.get("STUB_MACHINE", "x86_64"), "onnxruntime": "1.26.0",
               "qemu_cpu": os.environ.get("QEMU_CPU_MODEL", "")}}
open(a.json_out, "w").write(json.dumps(rec))
print("RESULT_JSON " + json.dumps(rec))
"""

STUB_DRIVER = r"""import argparse, hashlib, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
QEMU = "__QEMU__"
SYSROOT = "__SYSROOT__"
X86PY = "__X86PY__"
WORK = Path("__WORK__")
REPLAY = str(HERE / "replay.py")


def main():
    ap = argparse.ArgumentParser()
    for k in ("--manifest", "--platform", "--mode", "--tag", "--ids", "--timeout"):
        ap.add_argument(k)
    ap.add_argument("--dump-feeds", action="store_true")
    a = ap.parse_args()
    jobs = [j for j in json.load(open(a.manifest)) if j["id"] in a.ids.split(",")]
    out_file = HERE / "results" / f"{a.tag}__{Path(a.manifest).stem}__{a.mode}.jsonl"
    out_file.parent.mkdir(exist_ok=True)
    cpu = a.platform.split(":", 1)[1]
    for job in jobs:
        safe = job["id"].replace(":", "__")
        outp = WORK / f"out_{a.tag}" / f"{safe}.npz"
        outp.parent.mkdir(parents=True, exist_ok=True)
        if a.mode == "check":
            ce = open(job["ce_path"]).read()
            res = "spec_not_violated" if "STUB_FAIL" in ce else "correct"
            feed = hashlib.sha256(ce.encode()).hexdigest()
            Path(job["dump_feeds"]).parent.mkdir(parents=True, exist_ok=True)
            open(job["dump_feeds"], "w").write(feed)
            open(outp, "w").write("Y:" + feed)
            rec = {"id": job["id"], "mode": "check-one", "result": res, "rc": 0, "injected_from": None,
                   "env": {"machine": "x86_64", "onnxruntime": "1.26.0", "qemu_cpu": cpu}}
        else:
            feed = open(WORK / "feeds" / f"{safe}.npz").read()
            open(outp, "w").write("Y:diff" if os.environ.get("STUB_ORT_DIFF") else "Y:" + feed)
            rec = {"id": job["id"], "mode": "ort-one", "rc": 0}
        # the real driver.py stamps every record the same way (rec.update: id, tag, platform, mode, rc)
        rec.update({"id": job["id"], "tag": a.tag, "platform": a.platform, "mode": a.mode, "rc": 0})
        with open(out_file, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    print(f"DRIVER_DONE rc=0 WORST=0 ok={len(jobs)} fail=0 jobs={len(jobs)} tag={a.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""

STUB_CMP = r"""import sys
try:
    a, b = open(sys.argv[1]).read(), open(sys.argv[2]).read()
except OSError as e:
    print(f"CMP err {e}")
    sys.exit(2)
if a == b:
    print("CMP match keys=y")
    sys.exit(0)
print("CMP diff y:maxabs=1.0")
sys.exit(1)
"""


def make_rig(tmp: str, onnx: str, vnnlib: str) -> dict:
    """A stub x86 rig with the real rig's interfaces (driver.py check/ort, checkv1.py, cmp_npz.py, qemu-x86_64) and
    two witness controls on the toy instance: a good witness (must PASS) and a STUB_FAIL one (must FAIL)."""
    r = os.path.join(tmp, "rig")
    xr = os.path.join(r, "xr")
    for d in ("sysroot", "work", "xr/results", "sc25", "sc26", "controls"):
        os.makedirs(os.path.join(r, d), exist_ok=True)
    qemu = write(os.path.join(r, "qemu-x86_64"), STUB_QEMU)
    os.chmod(qemu, 0o755)
    write(os.path.join(xr, "driver.py"), STUB_DRIVER.replace("__QEMU__", qemu)
          .replace("__SYSROOT__", os.path.join(r, "sysroot")).replace("__X86PY__", sys.executable)
          .replace("__WORK__", os.path.join(r, "work")))
    write(os.path.join(xr, "replay.py"), "# stub replay.py (the driver's REPLAY constant)\n")
    write(os.path.join(r, "checkv1.py"), STUB_CHECKV1)
    write(os.path.join(r, "cmp_npz.py"), STUB_CMP)
    for y in ("sc25", "sc26"):
        write(os.path.join(r, y, "counterexamples.py"), "# stub organisers' checker\n")
    pos = write(os.path.join(r, "controls", "pos.counterexample"), "((X_0 0.1)\n (X_1 0.2))\n")
    neg = write(os.path.join(r, "controls", "neg.counterexample"), "((X_0 0.1))\n; STUB_FAIL\n")
    return {"xr_dir": xr, "driver": os.path.join(xr, "driver.py"), "replay": os.path.join(xr, "replay.py"),
            "checkv1": os.path.join(r, "checkv1.py"), "cmp": os.path.join(r, "cmp_npz.py"), "qemu": qemu,
            "sysroot": os.path.join(r, "sysroot"), "x86py": sys.executable, "armpy": sys.executable,
            "work": os.path.join(r, "work"), "cpu": "Haswell-v4", "onnxruntime": "1.26.0",
            "scoring": {"2025": os.path.join(r, "sc25"), "2026": os.path.join(r, "sc26")}, "timeout_s": 120,
            "controls": [{"name": "pos", "kind": "v1-2026", "expect": "pass", "onnx": onnx, "vnnlib": vnnlib, "ce": pos},
                         {"name": "neg", "kind": "v1-2026", "expect": "fail", "onnx": onnx, "vnnlib": vnnlib,
                          "ce": neg}]}


# the certified world's rows (TOY_ROWS): p0, p1, p4 banked unsat and armed unsat (re-proofs), p2 banked sat (an
# x86 replay), p3 field-SAT the arm leaves timeout (undecided); p5 is not a moat row
CERT_TOKENS = {0: " check=verified", 1: " check=unconfirmed", 4: " check=verified"}


def cert_world(tmp: str, tokens: dict[int, str] | None = None) -> tuple[World, dict]:
    w = World(tmp, cert_tokens={**CERT_TOKENS, **(tokens or {})})
    bdir = os.path.join(w.ny, "benchmarks/vnncomp2026/benchmarks", CAT, "1.0")
    return w, make_rig(tmp, os.path.join(bdir, "onnx/m.onnx"), os.path.join(bdir, "vnnlib/p0.vnnlib"))


def certify(w: World, rig: dict, out: str = "out", mode: str = "ok", ledger: str | None = None,
            extra: dict | None = None) -> tuple[dict, list]:
    """Plan S5 + S7 + certified S8, run S5, S7 and S8 phase A; returns (plan, [(rc, done, log) of S8])."""
    plan = w.plan(out, stages=["S5", "S7", "S8"], s8_mode="certified", x86_rig=rig, cert_ledger=ledger)
    res = []
    for st in ("S5", "S7", "S8"):
        rc, done, log = w.run(out, st, mode=mode if st == "S8" else "ok", extra=extra)
        res.append((rc, done, log))
        if st != "S8" and rc != 0:
            raise AssertionError(f"{st} rc={rc}: {log[-2000:]}")
    return plan, res


class CertifiedUnit(unittest.TestCase):
    """certlib's pure decisions: classes, follow-up rows, re-proof outcomes, the decision."""

    def test_certified_classifies_every_phase_a_row(self):
        r = jrow("S8_26_toy_0000", kind="MOAT", truth="unsat", bank="unsat", group="seal/S8/2026")
        self.assertEqual(C.classify(r, J(r["tag"], "competition", 1, "unsat"), False), C.C_UNSAT_REPROVE)
        self.assertEqual(C.classify(r, J(r["tag"], "competition", 1, "unsat"), True), C.C_UNSAT_LEDGER)
        new = dict(r, bank="timeout", truth="-")
        self.assertEqual(C.classify(new, J(r["tag"], "competition", 1, "unsat"), True), C.C_UNSAT_REPROVE,
                         "a ledger certificate skips the re-proof only on a BANKED-unsat row")
        self.assertEqual(C.classify(r, J(r["tag"], "competition", 1, "sat"), False), C.C_SAT)
        self.assertEqual(C.classify(r, J(r["tag"], "competition", 1, "timeout"), False), C.C_LOST)
        self.assertEqual(C.classify(new, J(r["tag"], "competition", 1, "timeout"), False), C.C_UNDECIDED)
        self.assertEqual(C.classify(r, None, False), C.C_NOTOK)
        self.assertEqual(C.classify(r, J(r["tag"], "competition", 1, "unsat", "VOID"), False), C.C_NOTOK)
        self.assertEqual(C.classify(r, J(r["tag"], "competition", 1, "sat", "EMERGENCY"), False), C.C_EMERGENCY)

    def test_certified_followup_rows_rederive_and_carry_the_flip_alarm(self):
        rows = [jrow(f"S8_26_toy_{i:04d}", kind="MOAT", truth="-", bank=b, index=i, vnnlib=f"/p{i}.vnnlib",
                     group="seal/S8/2026") for i, b in enumerate(("unsat", "sat", "timeout", "unsat"))]
        first = {rows[0]["tag"]: J(rows[0]["tag"], "competition", 1, "unsat"),
                 rows[1]["tag"]: J(rows[1]["tag"], "competition", 1, "sat"),
                 rows[2]["tag"]: J(rows[2]["tag"], "competition", 1, "unsat"),
                 rows[3]["tag"]: J(rows[3]["tag"], "competition", 1, "unsat")}
        rp = C.derive_reproof(rows, first, {rows[3]["tag"]: {"ok": True}})
        self.assertEqual([r["tag"] for r in rp], ["S8R_26_toy_0000", "S8R_26_toy_0002"])
        self.assertEqual([(r["reps_c"], r["reps_p"]) for r in rp], [("0", "1")] * 2)
        self.assertIn(C.FLIP_ALARM, rp[0]["alarm"].split(L.ALARM_SEP))
        self.assertEqual({k: v for k, v in rp[0].items() if k not in ("tag", "reps_c", "reps_p", "alarm", "group",
                                                                      "note")},
                         {k: v for k, v in rows[0].items() if k not in ("tag", "reps_c", "reps_p", "alarm", "group",
                                                                        "note")}, "the same instance and arming")
        out = {rp[0]["tag"]: {"state": C.R_ACCEPT}, rp[1]["tag"]: {"state": C.R_FALLBACK}}
        cl = C.derive_classic(rp, out)
        self.assertEqual([(r["tag"], r["reps_c"], r["reps_p"]) for r in cl], [("S8F_26_toy_0002", "3", "1")])
        self.assertIn(C.FLIP_ALARM, cl[0]["alarm"].split(L.ALARM_SEP))
        with self.assertRaisesRegex(C.CertError, "pending"):
            C.derive_classic(rp, {rp[0]["tag"]: {"state": C.R_ACCEPT}, rp[1]["tag"]: {"state": C.R_PENDING}})
        with self.assertRaises(C.CertError):
            C.followup_tag(C.REPROOF_PREFIX, "S5_26_toy_0000")

    def test_reproof_outcome_accepts_only_a_verified_certificate(self):
        J1 = lambda v, st="OK": J("S8R_x", "proof", 1, v, st)
        ran = {"bundle_exists": True, "log": {"ran": True, "done_rc": 0}}
        self.assertEqual(C.reproof_outcome(J1("unsat"), dict(ran, cert={"state": "accept", "why": "v"}))["state"],
                         C.R_ACCEPT)
        for cert in ({"state": "reject", "why": "checker verdict unconfirmed"}, {"state": "reject", "why": "hash"}):
            self.assertEqual(C.reproof_outcome(J1("unsat"), dict(ran, cert=cert))["state"], C.R_FALLBACK)
        self.assertEqual(C.reproof_outcome(J1("unsat"), dict(ran, cert={"state": "refuted", "why": "cex"}))["state"],
                         C.R_EMERGENCY)
        self.assertEqual(C.reproof_outcome(J1("sat"), ran)["state"], C.R_EMERGENCY, "a Proof sat is a flip")
        o = C.reproof_outcome(J1("timeout"), ran)
        self.assertEqual(o["state"], C.R_FALLBACK)
        self.assertIn("profile divergence", o["why"])
        self.assertEqual(C.reproof_outcome(J1("unsat"), {"bundle_exists": False})["state"], C.R_FALLBACK)
        self.assertEqual(C.reproof_outcome(J1("unsat"), {"bundle_exists": True, "log": {"ran": False}})["state"],
                         C.R_PENDING)
        self.assertEqual(C.reproof_outcome(J1("unsat"), {"bundle_exists": True, "log": {"ran": True, "ran_rc": 4}})
                         ["state"], C.R_FALLBACK, "a checker that printed no DONE line certifies nothing")
        self.assertEqual(C.reproof_outcome(None, ran)["state"], C.R_PENDING)
        self.assertEqual(C.reproof_outcome(J1("unsat", "VOID"), ran)["state"], C.R_PENDING)

    def _decide_world(self):
        rows = [jrow(f"S8_26_toy_{i:04d}", kind="MOAT", truth=t, bank=b, index=i, vnnlib=f"/p{i}.vnnlib",
                     group="seal/S8/2026") for i, (b, t) in enumerate((("unsat", "unsat"), ("sat", "sat"),
                                                                       ("unsat", "-")))]
        lines = [J(rows[0]["tag"], "competition", 1, "unsat", outcome="held"),
                 J(rows[1]["tag"], "competition", 1, "sat", outcome="held"),
                 J(rows[2]["tag"], "competition", 1, "unsat", outcome="held")]
        first = {r["tag"]: j for r, j in zip(rows, lines)}
        rp = C.derive_reproof(rows, first, {})
        return rows, lines, rp

    def test_certified_decision_pass_needs_every_item_and_names_its_evidence(self):
        rows, lines, rp = self._decide_world()
        acc = {"bundle_exists": True, "log": {"ran": True, "done_rc": 0}, "report": "/r", "bundle": "/b",
               "cert": {"state": "accept", "why": "verified", "report_sha256": "a" * 64, "bundle_sha256": "b" * 64,
                        "verdict": "verified"}}
        unc = dict(acc, cert={"state": "reject", "why": "checker verdict unconfirmed", "verdict": "unconfirmed"})
        cl = C.derive_classic(rp, {rp[0]["tag"]: {"state": C.R_ACCEPT}, rp[1]["tag"]: {"state": C.R_FALLBACK}})
        plines = lines + [J(rp[0]["tag"], "proof", 1, "unsat"), J(rp[1]["tag"], "proof", 1, "unsat")]
        clines = plines + [J(cl[0]["tag"], p, k, "unsat", outcome="n/a") for p, k in S.M0J.schedule(cl[0])]
        facts = {"reproof_spec": rp, "classic_spec": cl, "checks": {rp[0]["tag"]: acc, rp[1]["tag"]: unc},
                 "replays": {rows[1]["tag"]: {"state": C.X_PASS, "why": "correct", "result": "correct"}}}
        d = C.decide(toy_plan(), rows, clines, facts)
        self.assertEqual((d["emergencies"], d["void"], d["lost"]), ([], [], []))
        c = d["certified"]
        self.assertEqual(c["accepted"], [rows[0]["tag"]])
        self.assertEqual(c["classic_covered"], [rows[2]["tag"]])
        self.assertEqual(c["replayed"], [rows[1]["tag"]])
        self.assertEqual(c["rows"][rows[0]["tag"]]["reproof"]["check"]["report_sha256"], "a" * 64)
        # every incomplete item is VOID, never a pass
        for k, v in (("classic_spec", None), ("reproof_spec", None),
                     ("replays", {rows[1]["tag"]: {"state": C.X_INCONCLUSIVE, "why": "rig"}}),
                     ("replays", {}), ("checks", {rp[0]["tag"]: acc})):
            dd = C.decide(toy_plan(), rows, clines, dict(facts, **{k: v}))
            self.assertTrue(dd["void"], (k, v))
            self.assertFalse(dd["emergencies"])
        # a re-proof spec that does not re-derive (a row dropped, or an edited alarm) is VOID, by its own check
        for spec in (rp[:1], [dict(rp[0], alarm="-")] + rp[1:]):
            dd = C.decide(toy_plan(), rows, clines, dict(facts, reproof_spec=spec, classic_spec=None))
            self.assertTrue(any("re-proof spec on disk does not re-derive" in v for v in dd["void"]), dd["void"])
        # a classic spec that does not re-derive from the re-proof outcomes is VOID
        dd = C.decide(toy_plan(), rows, clines, dict(facts, classic_spec=[dict(cl[0], reps_c="1")]))
        self.assertTrue(any("classic spec on disk does not re-derive" in v for v in dd["void"]), dd["void"])
        # a classic sub-moat whose runs are not all OK is VOID; one that split is an EMERGENCY via the stage judge
        dd = C.decide(toy_plan(), rows, plines + [J(cl[0]["tag"], "competition", 1, "unsat")], facts)
        self.assertTrue(any("classic sub-moat runs" in v for v in dd["void"]))

    def test_certified_emergencies_refuted_check_proof_sat_and_wrong_sat(self):
        rows, lines, rp = self._decide_world()
        ref = {"bundle_exists": True, "log": {"ran": True, "done_rc": 21},
               "cert": {"state": "refuted", "why": "cex", "verdict": "refuted"}}
        facts = {"reproof_spec": rp, "classic_spec": None, "checks": {rp[0]["tag"]: ref, rp[1]["tag"]: ref},
                 "replays": {rows[1]["tag"]: {"state": C.X_FAIL, "why": "spec_not_violated", "result": "spec_not_violated"}}}
        plines = lines + [J(rp[0]["tag"], "proof", 1, "unsat"), J(rp[1]["tag"], "proof", 1, "sat")]
        d = C.decide(toy_plan(), rows, plines, facts)
        em = " ".join(d["emergencies"])
        self.assertIn("REFUTED", em)
        self.assertIn("Proof profile states sat", em)
        self.assertIn("WRONG SAT", em)

    def test_certified_s8_judge_lost_is_partial_pending_is_void_floor_is_one_rep(self):
        plan = toy_plan(stages={"S8": {"mode": "certified"}})
        r = jrow("S8_26_toy_0000", kind="MOAT", truth="-", bank="unsat", reps_c=1, group="seal/S8/2026")
        lost = [J(r["tag"], "competition", 1, "timeout", outcome="lost")]
        rec = judge(plan, "S8", [r], lost, runs_for([r]))
        self.assertEqual((rec["status"], rec["rc"]), ("PARTIAL", 4))
        self.assertIn("LOST (tolerance 0)", " ".join(rec["reasons"]))
        held = [J(r["tag"], "competition", 1, "unsat", outcome="held")]
        rec = judge(plan, "S8", [r], held, runs_for([r]))
        self.assertEqual((rec["status"], rec["rc"]), ("VOID", 5))
        self.assertIn("certification pending", " ".join(rec["reasons"]))
        und = dict(r, bank="-")
        rec = judge(plan, "S8", [und], [J(r["tag"], "competition", 1, "timeout")], runs_for([und]))
        self.assertEqual(rec["status"], "PASS", rec["reasons"])
        self.assertEqual(rec["certified"]["counts"], {C.C_UNDECIDED: 1})
        # the classic floor still applies to a CLASSIC plan with one rep
        rec = judge(toy_plan(), "S8", [und], [J(r["tag"], "competition", 1, "timeout")], runs_for([und]))
        self.assertEqual(rec["status"], "PARTIAL")


def _cert_files(tmp: str, verdict: str = "verified", commit: str = "c" * 40, profile: str = "proof",
                confirmed: bool = True) -> dict:
    """A real-format bundle + checker report + its check log over a toy instance."""
    onnx = write(os.path.join(tmp, "b/onnx/m.onnx"), "onnx-bytes")
    vnn = write(os.path.join(tmp, "b/vnnlib/p.vnnlib"), "(assert (<= X_0 1))\n")
    sh = lambda p: L.sha256_file(p)
    bundle = os.path.join(tmp, "res.proof.jsonl")
    hdr = {"kind": "header", "schema": C.BUNDLE_SCHEMA, "profile": profile, "build": f"NY_BUILD_STAMP_V1 commit={commit} x",
           "onnx": {"path": onnx, "sha256": sh(onnx)}, "vnnlib": {"path": vnn, "sha256": sh(vnn)}}
    write(bundle, json.dumps(hdr) + "\n" + json.dumps({"kind": "end"}) + "\n")
    rep = os.path.join(tmp, "check", "T.check.json")
    d = {"schema": C.REPORT_SCHEMA, "checker_build": f"NY_BUILD_STAMP_V1 commit={commit} x", "verdict": verdict,
         "exit_code": {"verified": 0, "unconfirmed": 10, "refuted": 21, "invalid": 20}.get(verdict, 31),
         "bundle": {"path": bundle, "sha256": sh(bundle), "schema": C.BUNDLE_SCHEMA},
         "onnx": {"path": onnx, "sha256": sh(onnx)}, "vnnlib": {"path": vnn, "sha256": sh(vnn)},
         "cover": [{"region": 0, "disjunct": 0, "claimed_ok": True, "confirmed_ok": confirmed}],
         "nodes": {"total": 3, "confirmed": 3, "unconfirmed": 0, "opaque": 0}}
    write(rep, json.dumps(d) + "\n")
    log = write(os.path.join(tmp, "check", "T.check.log"),
                f"PROOF_CHECK_DONE rc={d['exit_code']} verdict={verdict} nodes=3 confirmed=3 unconfirmed=0 cover=ok "
                f"refuted=0 wall_s=1.0 report={rep} report_sha256={sh(rep)}\nCERT_CHECK_RAN rc={d['exit_code']}\n")
    return {"onnx": onnx, "vnnlib": vnn, "bundle": bundle, "report": rep, "log": log, "doc": d,
            "onnx_sha": sh(onnx), "vnn_sha": sh(vnn)}


class CertifiedArtefacts(unittest.TestCase):
    """The certificate, the replay record and the ledger, verified against real files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_check_report_accept_requires_verified_cover_hashes_bundle_and_pin(self):
        f = _cert_files(self.tmp.name)
        lf = C.check_log_facts(open(f["log"]).read())
        self.assertEqual((lf["ran"], lf["done_rc"], lf["done_verdict"]), (True, 0, "verified"))
        ok = C.verify_certificate(f["report"], f["bundle"], f["onnx_sha"], f["vnn_sha"], "c" * 40, lf["done_sha"])
        self.assertEqual(ok["state"], "accept", ok["why"])

        def mutate(fn, why):
            d = json.loads(open(f["report"]).read())
            fn(d)
            write(f["report"], json.dumps(d) + "\n")
            got = C.verify_certificate(f["report"], f["bundle"], f["onnx_sha"], f["vnn_sha"], "c" * 40)
            write(f["report"], json.dumps(f["doc"]) + "\n")
            self.assertEqual(got["state"], "reject", why)
            return got
        mutate(lambda d: d.update(verdict="unconfirmed", exit_code=10), "an opaque / unconfirmed claim")
        mutate(lambda d: d.update(exit_code=10), "exit code")
        mutate(lambda d: d["cover"][0].update(confirmed_ok=False), "a cover row claimed but not confirmed")
        mutate(lambda d: d.update(cover=[]), "no cover")
        mutate(lambda d: d["onnx"].update(sha256="0" * 64), "another model")
        mutate(lambda d: d["vnnlib"].update(sha256="0" * 64), "another property")
        mutate(lambda d: d["bundle"].update(sha256="0" * 64), "another bundle")
        mutate(lambda d: d["bundle"].update(path="/elsewhere.proof.jsonl"), "another bundle path")
        mutate(lambda d: d.update(checker_build="NY_BUILD_STAMP_V1 commit=" + "d" * 40 + " x"), "another checker")
        mutate(lambda d: d.update(schema="x"), "schema")
        d = dict(f["doc"], verdict="refuted")
        write(f["report"], json.dumps(d) + "\n")
        self.assertEqual(C.verify_certificate(f["report"], f["bundle"], f["onnx_sha"], f["vnn_sha"], "c" * 40)["state"],
                         "refuted", "a refuted verdict bound to the row is the emergency, whatever its exit code says")
        write(f["report"], json.dumps(f["doc"]) + "\n")
        # the DONE line's sha binds the file: an edit after the check rejects
        write(f["report"], json.dumps(f["doc"]) + " \n")
        got = C.verify_certificate(f["report"], f["bundle"], f["onnx_sha"], f["vnn_sha"], "c" * 40, lf["done_sha"])
        self.assertEqual(got["state"], "reject")
        # a bundle written under the competition profile, or by another build, is no Proof certificate
        g = _cert_files(os.path.join(self.tmp.name, "g"), profile="competition")
        self.assertEqual(C.verify_certificate(g["report"], g["bundle"], g["onnx_sha"], g["vnn_sha"], "c" * 40)["state"],
                         "reject")
        h = _cert_files(os.path.join(self.tmp.name, "h"), commit="e" * 40)
        self.assertEqual(C.verify_certificate(h["report"], h["bundle"], h["onnx_sha"], h["vnn_sha"], "c" * 40)["state"],
                         "reject")
        self.assertEqual(C.verify_certificate(h["report"], h["bundle"], h["onnx_sha"], h["vnn_sha"])["state"], "accept",
                         "without a pin (a ledger certificate) the build is not required")
        r = _cert_files(os.path.join(self.tmp.name, "r"), verdict="refuted")
        self.assertEqual(C.verify_certificate(r["report"], r["bundle"], r["onnx_sha"], r["vnn_sha"], "c" * 40)["state"],
                         "refuted")
        for v in ("unconfirmed", "invalid", "unsupported"):
            u = _cert_files(os.path.join(self.tmp.name, v), verdict=v)
            self.assertEqual(C.verify_certificate(u["report"], u["bundle"], u["onnx_sha"], u["vnn_sha"], "c" * 40)
                             ["state"], "reject", v)

    def test_ledger_certificate_skips_reproof_only_when_it_reverifies(self):
        f = _cert_files(self.tmp.name)
        e = {"onnx_sha256": f["onnx_sha"], "vnnlib_sha256": f["vnn_sha"], "report": f["report"],
             "report_sha256": L.sha256_file(f["report"]), "bundle": f["bundle"],
             "bundle_sha256": L.sha256_file(f["bundle"]), "checker_build": f"NY_BUILD_STAMP_V1 commit={'c' * 40} x",
             "source": "test"}
        led = write(os.path.join(self.tmp.name, "ledger.tsv"), C.ledger_text([e]))
        self.assertEqual(C.parse_ledger(led), [e])
        idx = C.ledger_index(C.parse_ledger(led))
        row = jrow("S8_26_toy_0000", kind="MOAT", bank="unsat")
        row.update(onnx=f["onnx"], vnnlib=f["vnnlib"])
        got = C.ledger_certificate(idx, row)
        self.assertTrue(got["ok"], got["why"])
        self.assertEqual(got["entry"]["checker_commit"], "c" * 40, "the record names the checker a ledger row rests on")
        # review(certified-moat): the ledger DECLARES its checker; a report from another checker build is no
        # certificate, and neither is a ledger that declares none
        for col in (f"NY_BUILD_STAMP_V1 commit={'d' * 40} x", "b", f"commit={'c' * 40}x"):
            bad = C.ledger_index([dict(e, checker_build=col)])
            self.assertFalse(C.ledger_certificate(bad, row)["ok"], col)
            self.assertIn("is not the one the ledger declares", C.ledger_certificate(bad, row)["why"])
        other = dict(row, vnnlib=write(os.path.join(self.tmp.name, "b/vnnlib/q.vnnlib"), "(assert (<= X_0 2))\n"))
        self.assertFalse(C.ledger_certificate(idx, other)["ok"], "another instance")
        # a report rewritten since the ledger recorded it -- still a verified report -- is no certificate
        d = json.loads(open(f["report"]).read())
        d["rewritten"] = 1
        write(f["report"], json.dumps(d) + "\n")
        self.assertTrue(C.verify_certificate(f["report"], f["bundle"], f["onnx_sha"], f["vnn_sha"])["state"] == "accept")
        got = C.ledger_certificate(idx, row)
        self.assertFalse(got["ok"])
        self.assertIn("changed since the ledger recorded it", got["why"])
        write(f["report"], json.dumps(f["doc"]) + "\n")
        self.assertTrue(C.ledger_certificate(idx, row)["ok"])
        d["verdict"] = "unconfirmed"
        write(f["report"], json.dumps(d) + "\n")
        self.assertFalse(C.ledger_certificate(idx, row)["ok"], "a report changed since the ledger recorded it")
        with self.assertRaises(C.CertError):
            C.parse_ledger(write(os.path.join(self.tmp.name, "bad.tsv"), "# other/v1\n"))

    def test_x86_replay_pass_needs_controls_ort_match_and_the_x86_platform_fail_is_a_wrong_sat(self):
        import x86_replay as X
        w, rig = cert_world(self.tmp.name)
        rig = json.loads(json.dumps(rig))
        rig["files"] = C.rig_files(rig)
        for c in rig["controls"]:
            c["ce_sha256"] = L.sha256_file(c["ce"])
        self.assertEqual(C.rig_problems(rig), [])
        self.assertEqual(X.rig_check(rig), [])
        dest = os.path.join(self.tmp.name, "x86")
        ctl = X.run_controls(rig, dest, "b1")
        cdoc = json.load(open(ctl))
        self.assertTrue(cdoc["ok"], cdoc)
        self.assertEqual([(c["name"], c["got"]) for c in cdoc["controls"]], [("pos", "pass"), ("neg", "fail")])
        bdir = os.path.join(w.ny, "benchmarks/vnncomp2026/benchmarks", CAT, "1.0")
        row = jrow("S8_26_toy_0002", kind="MOAT", bank="sat", truth="sat", index=2)
        row.update(onnx=os.path.join(bdir, "onnx/m.onnx"), vnnlib=os.path.join(bdir, "vnnlib/p2.vnnlib"), year="2026")
        self.assertEqual(C.checker_kind(row), "v1-2026")
        self.assertEqual(C.checker_kind(dict(row, onnx=row["onnx"].replace("/1.0/", "/2.0/"))), "v2-2026")
        self.assertEqual(C.checker_kind(dict(row, year="2025")), "v1-2025")
        res = write(os.path.join(self.tmp.name, "w.res"), "sat\n((X_0 0.25)\n (X_1 0.5))\n")
        rdir = os.path.join(dest, row["tag"])
        X.replay_row(rig, row, res, rdir, ctl, "0" * 16)
        rec = os.path.join(rdir, "replay.json")
        v = C.verify_replay_record(rec, row, res, rig)
        self.assertEqual(v["state"], C.X_PASS, v["why"])
        # the replayed witness must be THIS results file
        write(res, "sat\n((X_0 0.75))\n")
        self.assertEqual(C.verify_replay_record(rec, row, res, rig)["state"], C.X_INCONCLUSIVE)
        # a witness the organisers' checker rejects: FAIL (the caller's EMERGENCY)
        write(res, "sat\n((X_0 0.25))\n; STUB_FAIL\n")
        X.replay_row(rig, row, res, rdir, ctl, "0" * 16)
        self.assertEqual(C.verify_replay_record(rec, row, res, rig)["state"], C.X_FAIL)
        # ... but only on the certified platform: an arm64 result is inconclusive, never a pass or a fail
        os.environ["STUB_MACHINE"] = "aarch64"
        try:
            X.replay_row(rig, row, res, rdir, ctl, "0" * 16)
        finally:
            del os.environ["STUB_MACHINE"]
        self.assertEqual(C.verify_replay_record(rec, row, res, rig)["state"], C.X_INCONCLUSIVE)
        # an ORT replay that does not reproduce the checker's outputs, a rig error: inconclusive
        write(res, "sat\n((X_0 0.25))\n")
        os.environ["STUB_ORT_DIFF"] = "1"
        try:
            X.replay_row(rig, row, res, rdir, ctl, "0" * 16)
        finally:
            del os.environ["STUB_ORT_DIFF"]
        self.assertEqual(C.verify_replay_record(rec, row, res, rig)["state"], C.X_INCONCLUSIVE)
        write(res, "sat\n; STUB_RIGERR\n")
        X.replay_row(rig, row, res, rdir, ctl, "0" * 16)
        self.assertEqual(C.verify_replay_record(rec, row, res, rig)["state"], C.X_INCONCLUSIVE)
        # an edited controls file (after the replay) voids the replay
        write(res, "sat\n((X_0 0.25))\n")
        X.replay_row(rig, row, res, rdir, ctl, "0" * 16)
        self.assertEqual(C.verify_replay_record(rec, row, res, rig)["state"], C.X_PASS)
        corig = open(ctl).read()
        write(ctl, corig.replace('"ok": true', '"ok": true, "edited": 1'))
        self.assertEqual(C.verify_replay_record(rec, row, res, rig)["state"], C.X_INCONCLUSIVE)
        write(ctl, corig)
        # controls that do not discriminate void the batch: a FAIL control that passes
        write(rig["controls"][1]["ce"], "((X_0 0.1))\n")
        bad = X.run_controls(rig, dest, "b2")
        self.assertFalse(json.load(open(bad))["ok"])
        write(res, "sat\n((X_0 0.25))\n")
        X.replay_row(rig, row, res, rdir, bad, "0" * 16)
        self.assertEqual(C.verify_replay_record(rec, row, res, rig)["state"], C.X_INCONCLUSIVE)
        # the v2 path (driver --mode check --dump-feeds, then --mode ort)
        r2 = X.replay_core(rig, "v2-2026", row["onnx"], row["vnnlib"], rig["controls"][0]["ce"],
                           os.path.join(dest, "v2"), "seal:test:v2")
        self.assertEqual((r2["result"], r2["ort"]["state"]), ("correct", "match"))


class CertifiedRunner(unittest.TestCase):
    """seal_run.sh S8 (phase A) + S8C end to end through the REAL m0 runner, the stub pin's bundles and checker, and
    the stub x86 rig."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_certified_plan_runs_each_row_once_and_preregisters_ledger_rig_and_controls(self):
        w, rig = cert_world(self.tmp.name)
        plan = w.plan(stages=["S5", "S7", "S8"], s8_mode="certified", x86_rig=rig)
        rows = S.stage_rows(plan["out"], plan, "S8")
        self.assertEqual({(r["reps_c"], r["reps_p"]) for r in rows}, {("1", "0")})
        self.assertEqual(plan["stages"]["S8"]["mode"], "certified")
        c = plan["certified"]
        self.assertEqual(c["lost_tolerance"], 0)
        self.assertIn(rig["checkv1"], c["x86"]["files"])
        self.assertTrue(all(x["ce"].startswith(plan["out"]) for x in c["x86"]["controls"]), "controls copied in")
        self.assertTrue(any(k.startswith("certified/controls/") for k in plan["files"]))
        with self.assertRaisesRegex(S.SealError, "never a rep count"):
            w.plan("o2", stages=["S5", "S7", "S8"], s8_mode="certified", x86_rig=rig, opts={"s8_reps_c": 3})
        with self.assertRaisesRegex(S.SealError, "without --s8-mode certified"):
            w.plan("o3", stages=["S5", "S7", "S8"], x86_rig=rig)
        bad = dict(rig, controls=[rig["controls"][0]])
        with self.assertRaisesRegex(C.CertError, "expected PASS and one expected FAIL"):
            w.plan("o4", stages=["S5", "S7", "S8"], s8_mode="certified", x86_rig=bad)
        # an edited control witness is refused like any pre-registered file
        p = os.path.join(plan["out"], "certified/controls/neg.counterexample")
        with open(p, "a") as fh:
            fh.write("edited\n")
        with self.assertRaisesRegex(S.SealError, "pre-registered"):
            S.load_plan(plan["out"])

    def test_runner_certified_s8_end_to_end_and_the_record_derives(self):
        w, rig = cert_world(self.tmp.name)
        boxlog = os.path.join(self.tmp.name, "box.log")
        plan, res = certify(w, rig, extra={"SEALTEST_BOXLOG": boxlog})
        rc, done, log = res[-1]
        self.assertEqual(rc, 5, log[-3000:])
        self.assertIn("status=VOID", done)
        s8 = S.read_json(os.path.join(plan["out"], "S8", "STAGE.json"))
        self.assertIn("certification pending", " ".join(s8["reasons"]))
        rc, done, log = w.run("out", "S8C", extra={"SEALTEST_BOXLOG": boxlog})
        self.assertEqual(rc, 0, log[-4000:])
        self.assertTrue(done.startswith("SEAL_DONE rc=0 WORST=0 stage=S8C shard=1/1 status=PASS "), done)
        self.assertEqual(done.count(" rc="), 1)
        self.assertIn("cert_accepted=2 cert_classic=1 cert_replayed=1", done)
        s8 = S.read_json(os.path.join(plan["out"], "S8", "STAGE.json"))
        c = s8["certified"]
        self.assertEqual([t[-4:] for t in c["accepted"]], ["0000", "0004"])
        self.assertEqual([t[-4:] for t in c["classic_covered"]], ["0001"])
        self.assertEqual([t[-4:] for t in c["replayed"]], ["0002"])
        self.assertEqual(c["counts"][C.C_UNDECIDED], 1)
        # the runs: phase A once per row; one Proof re-run per armed unsat; 3c+1p only where the checker said no
        def calls_of(d):
            p = os.path.join(plan["out"], d, "calls.log")
            return sorted(open(p).read().splitlines()) if os.path.isfile(p) else []
        self.assertEqual(calls_of("S8/run_1of1"), sorted(f"p{i}.vnnlib arm competition" for i in (0, 1, 2, 3, 4)))
        self.assertEqual(calls_of("S8C/run_reproof"), sorted(f"p{i}.vnnlib arm proof" for i in (0, 1, 4)))
        self.assertEqual(calls_of("S8C/run_classic"), ["p1.vnnlib arm competition"] * 3 + ["p1.vnnlib arm proof"])
        calls = w.calls("out")
        box = open(boxlog).read()
        self.assertEqual(box.count("proof-check"), 2 + 1, "one check per published bundle (p0, p1, p4)")
        self.assertEqual(box.count("x86_replay.py batch"), 1)
        reproof = L.read_spec(C.spec_path(plan["out"], "reproof"))
        classic = L.read_spec(C.spec_path(plan["out"], "classic"))
        self.assertEqual([r["tag"][-4:] for r in reproof], ["0000", "0001", "0004"])
        self.assertEqual([(r["tag"][-4:], r["reps_c"], r["reps_p"]) for r in classic], [("0001", "3", "1")])
        # a second S8C re-runs nothing and re-derives the same verdict
        n_calls, n_box = len(calls), open(boxlog).read().count("proof-check")
        rc, done, log = w.run("out", "S8C", extra={"SEALTEST_BOXLOG": boxlog})
        self.assertEqual(rc, 0, log[-3000:])
        self.assertEqual((len(w.calls("out")), open(boxlog).read().count("proof-check")), (n_calls, n_box))
        # the record derives, the registry records it, and the auditor derives S8 and SR.seal from it
        p8 = os.path.join(plan["out"], "S8", "STAGE.json")
        state, note, _ = S.verify_stage_record(p8, "toy-tier", "S8")
        self.assertEqual(state, "done", note)
        rep = S.record_into_registry(w.registry, plan["out"])
        self.assertEqual(rep["record"].get("S8"), "done")
        self.assertEqual(rep["record"].get("SR"), "done", rep["sr_seal"])
        cap = L.strict_load(open(w.registry).read())["capabilities"][0]
        fake = types.SimpleNamespace(path=lambda p: p, errors=[], _seal_records={},
                                     git=types.SimpleNamespace(is_ancestor=lambda a, b: a == w.commit and b == w.commit))
        stages = {st: EPA.Auditor.audit_seal_stage(fake, cap, st, cap["evidence"][st]) for st in ("S7", "S8")}
        self.assertEqual(stages["S8"].state, "done", stages["S8"].note)
        self.assertEqual(EPA.Auditor.audit_routable(fake, cap, stages).state, "done")
        # the record re-verifies the checker reports and the replays: any edit derives nothing
        acc = c["rows"][c["accepted"][0]]["reproof"]["check"]["report"]
        orig = open(acc).read()
        for edit, what in ((orig.replace('"verdict":"verified"', '"verdict":"unconfirmed"'), "a report's verdict"),
                           (orig.replace('"confirmed_ok":true', '"confirmed_ok":false'), "a report's cover")):
            write(acc, edit)
            state, note, _ = S.verify_stage_record(p8, "toy-tier", "S8")
            self.assertEqual(state, "open", what)
            write(acc, orig)
        self.assertEqual(S.verify_stage_record(p8, "toy-tier", "S8")[0], "done")
        _, logp = C.check_paths(plan["out"], reproof[0]["tag"])
        lorig = open(logp).read()
        write(logp, lorig.replace("CERT_CHECK_RAN", "CERT_CHECK_X"))
        self.assertEqual(S.verify_stage_record(p8, "toy-tier", "S8")[0], "open", "a check that did not run")
        write(logp, lorig)
        rp = C.replay_record_path(plan["out"], c["replayed"][0])
        rorig = open(rp).read()
        write(rp, rorig.replace('"result": "correct"', '"result": "correct_up_to_tolerance"'))
        self.assertEqual(S.verify_stage_record(p8, "toy-tier", "S8")[0], "open", "an edited replay record")
        write(rp, rorig)
        self.assertEqual(S.verify_stage_record(p8, "toy-tier", "S8")[0], "done")
        # a follow-up judged.tsv edited AND re-recorded (`judge-stage` writes a fresh sha) is caught by re-deriving the
        # run from its artefacts: the classic sub-moat's competition r2 rewritten as LOST
        jp = os.path.join(C.run_dir(plan["out"], "classic"), "judged.tsv")
        jorig = open(jp).read()
        lines = jorig.splitlines(keepends=True)
        k = next(i for i, ln in enumerate(lines) if "\tcompetition\t2\t" in ln)
        lines[k] = lines[k].replace("\theld\t", "\tlost\t")
        write(jp, "".join(lines))
        S.judge_stage_from_out(plan["out"], "S8")
        state, note, _ = S.verify_stage_record(p8, "toy-tier", "S8")
        self.assertEqual(state, "open", note)
        self.assertIn("do not re-derive from their run artefacts", note)
        write(jp, jorig)
        S.judge_stage_from_out(plan["out"], "S8")
        self.assertEqual(S.verify_stage_record(p8, "toy-tier", "S8")[0], "done")
        # the ACCEPTed certificates become a ledger a later plan takes: their banked-unsat rows skip the re-proof
        led = os.path.join(self.tmp.name, "ledger.tsv")
        self.assertEqual(len(S.cert_ledger_entries(plan["out"])), 2)
        S.write_new(led, C.ledger_text(S.cert_ledger_entries(plan["out"])))
        plan2, res2 = certify(w, rig, "out2", ledger=led)
        rc, done, log = w.run("out2", "S8C")
        self.assertEqual(rc, 0, log[-3000:])
        s82 = S.read_json(os.path.join(plan2["out"], "S8", "STAGE.json"))
        self.assertEqual([t[-4:] for t in s82["certified"]["ledger_covered"]], ["0000", "0004"])
        self.assertEqual([r["tag"][-4:] for r in L.read_spec(C.spec_path(plan2["out"], "reproof"))], ["0001"])

    def test_runner_certified_refuted_check_is_an_emergency(self):
        w, rig = cert_world(self.tmp.name, {0: " check=refuted"})
        plan, _ = certify(w, rig)
        rc, done, log = w.run("out", "S8C")
        self.assertEqual(rc, 3, log[-3000:])
        self.assertIn("status=EMERGENCY", done)
        self.assertIn("phase=check", done)
        self.assertTrue(os.path.isfile(os.path.join(plan["out"], "EMERGENCY")))
        self.assertFalse(os.path.isfile(C.spec_path(plan["out"], "classic")), "nothing more runs")
        rc, done, _ = w.run("out", "S8C")
        self.assertEqual(rc, 3, done)
        self.assertIn("prior-emergency", done)

    def test_runner_certified_proof_profile_sat_stops_at_once(self):
        w, rig = cert_world(self.tmp.name, {0: " check=verified proofv=sat"})
        plan, _ = certify(w, rig)
        rc, done, log = w.run("out", "S8C")
        self.assertEqual(rc, 3, log[-3000:])
        self.assertIn("phase=reproof", done)
        self.assertIn("ALARM[Result: sat]", open(os.path.join(C.run_dir(plan["out"], "reproof"), "judged.tsv")).read())
        self.assertTrue(os.path.isfile(os.path.join(plan["out"], "EMERGENCY")))

    def test_runner_certified_wrong_sat_is_an_emergency_before_any_reproof(self):
        w, rig = cert_world(self.tmp.name)
        plan, _ = certify(w, rig, mode="wrongsat")
        rc, done, log = w.run("out", "S8C")
        self.assertEqual(rc, 3, log[-3000:])
        self.assertIn("phase=replay", done)
        self.assertTrue(os.path.isfile(os.path.join(plan["out"], "EMERGENCY")))
        self.assertFalse(os.path.isfile(C.spec_path(plan["out"], "reproof")), "no re-proof box time after a wrong sat")

    def test_runner_certified_fallbacks_profile_divergence_and_no_bundle_take_the_classic_submoat(self):
        w, rig = cert_world(self.tmp.name, {0: " check=verified proofv=timeout", 4: " bundle=no",
                                            1: " check=crash"})
        plan, _ = certify(w, rig)
        rc, done, log = w.run("out", "S8C")
        # p0's armed Proof profile never decides it, so its classic sub-moat's Proof rep LOSES the banked unsat: the
        # classic rule (any lost run is PARTIAL) holds inside the certified mode too
        self.assertEqual(rc, 4, log[-3000:])
        s8 = S.read_json(os.path.join(plan["out"], "S8", "STAGE.json"))
        self.assertEqual(s8["status"], "PARTIAL")
        self.assertIn("LOST (tolerance 0)", " ".join(s8["reasons"]))
        rows = s8["certified"]["rows"]
        self.assertEqual(rows["S8_26_toy_2026_0000"]["classic"]["lost_runs"], 1)
        self.assertIn("profile divergence", rows["S8_26_toy_2026_0000"]["reproof"]["why"])
        self.assertIn("no bundle", rows["S8_26_toy_2026_0004"]["reproof"]["why"])
        self.assertIn("PROOF_CHECK_DONE", rows["S8_26_toy_2026_0001"]["reproof"]["why"])
        self.assertEqual(s8["certified"]["accepted"], [])
        self.assertEqual([t[-4:] for t in s8["certified"]["classic_covered"]], ["0000", "0001", "0004"])

    def test_runner_certified_box_refused_check_stays_pending_and_resumes(self):
        # a check the box refused never ran: it must stay PENDING (S8C re-runs it), never fall back to a classic
        # sub-moat the checker never asked for
        w, rig = cert_world(self.tmp.name)
        plan, _ = certify(w, rig)
        rc, done, log = w.run("out", "S8C", extra={"SEALTEST_BOX_REFUSE": "proof-check"})
        self.assertEqual(rc, 5, log[-3000:])
        self.assertIn("the check stays pending", log)
        self.assertFalse(os.path.isfile(C.spec_path(plan["out"], "classic")))
        s8 = S.read_json(os.path.join(plan["out"], "S8", "STAGE.json"))
        self.assertEqual(s8["status"], "VOID")
        self.assertEqual(s8["certified"]["counts"].get("reproof_pending"), 3)
        rc, done, log = w.run("out", "S8C")
        self.assertEqual(rc, 0, log[-3000:])
        self.assertIn("cert_accepted=2 cert_classic=1 cert_replayed=1", done)

    def test_runner_certified_refuses_s8c_before_phase_a_is_clean(self):
        w, rig = cert_world(self.tmp.name)
        w.plan(stages=["S5", "S7", "S8"], s8_mode="certified", x86_rig=rig)
        for st in ("S5", "S7"):
            self.assertEqual(w.run("out", st)[0], 0)
        rc, done, log = w.run("out", "S8C")
        self.assertEqual(rc, 97, log[-2000:])
        self.assertIn("phase A is not clean", log)
        self.assertEqual([c for c in w.calls("out") if " proof" in c and "S8" in c], [])

    def test_smoke_plan_runs_named_rows_ungated_and_is_never_evidence(self):
        w, rig = cert_world(self.tmp.name)
        plan = w.plan(stages=["S8"], s8_mode="certified", x86_rig=rig,
                      smoke_rows=[f"2026/{CAT}/0", f"2026/{CAT}/2", f"2026/{CAT}/1"])
        self.assertEqual([r["index"] for r in S.stage_rows(plan["out"], plan, "S8")], ["0", "2", "1"])
        rc, done, log = w.run("out", "S8")
        self.assertEqual(rc, 5, log[-3000:])
        self.assertIn("SMOKE plan: S8 runs ungated", log)
        rc, done, log = w.run("out", "S8C")
        self.assertEqual(rc, 4, log[-3000:])
        s8 = S.read_json(os.path.join(plan["out"], "S8", "STAGE.json"))
        self.assertEqual(s8["status"], "PARTIAL")
        self.assertEqual([r for r in s8["reasons"]], ["below the S8 floor: a SMOKE plan (named rows, ungated): never the moat"])
        self.assertEqual(S.verify_stage_record(os.path.join(plan["out"], "S8", "STAGE.json"), "toy-tier", "S8")[0],
                         "open")
        with self.assertRaisesRegex(S.SealError, "SMOKE"):
            S.record_into_registry(w.registry, plan["out"])
        with self.assertRaisesRegex(S.SealError, "smoke rows are not moat rows"):
            w.plan("o2", stages=["S8"], smoke_rows=[f"2026/{CAT}/5"])
        with self.assertRaisesRegex(S.SealError, "exactly one stage"):
            w.plan("o3", stages=["S5", "S8"], smoke_rows=[f"2026/{CAT}/0"])


class CertifiedReview(unittest.TestCase):
    """review(certified-moat): the bindings an adversarial review found missing -- a replay counted on a record that
    is not its own, v2 rows judged by v1 controls, a control's `got` taken on its word, a `refuted` check quietly
    downgraded by a binding, an ACCEPT the checker's own DONE line / exit does not state, a ledger certificate from
    an undeclared checker, a resumed replay batch that forgets an earlier WRONG SAT, a plan that can never PASS."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _rig(rig: dict, extra=()) -> dict:
        rig = json.loads(json.dumps(rig))
        rig["controls"] = rig["controls"] + list(extra)
        rig["files"] = C.rig_files(rig)
        for c in rig["controls"]:
            c["ce_sha256"] = L.sha256_file(c["ce"])
        return rig

    @staticmethod
    def _edit_json(path: str, fn) -> None:
        d = json.load(open(path))
        fn(d)
        write(path, json.dumps(d, indent=1, sort_keys=True) + "\n")

    def _row(self, w: "World", ver: str = "1.0") -> dict:
        b1 = os.path.join(w.ny, "benchmarks/vnncomp2026/benchmarks", CAT, "1.0")
        b = os.path.join(w.ny, "benchmarks/vnncomp2026/benchmarks", CAT, ver)
        if ver != "1.0":
            for sub in ("onnx/m.onnx", "vnnlib/p2.vnnlib"):
                os.makedirs(os.path.dirname(os.path.join(b, sub)), exist_ok=True)
                with open(os.path.join(b1, sub), "rb") as src, open(os.path.join(b, sub), "wb") as dst:
                    dst.write(src.read())
        row = jrow("S8_26_toy_0002", kind="MOAT", bank="sat", truth="sat", index=2)
        row.update(onnx=os.path.join(b, "onnx/m.onnx"), vnnlib=os.path.join(b, "vnnlib/p2.vnnlib"), year="2026")
        return row

    def test_review_replay_counts_only_with_its_own_checker_record(self):
        import x86_replay as X
        w, rig0 = cert_world(self.tmp.name)
        rig = self._rig(rig0)
        dest = os.path.join(self.tmp.name, "x86")
        ctl = X.run_controls(rig, dest, "b1")
        row = self._row(w)
        res = write(os.path.join(self.tmp.name, "w.res"), "sat\n((X_0 0.25)\n (X_1 0.5))\n")
        rdir = os.path.join(dest, row["tag"])
        X.replay_row(rig, row, res, rdir, ctl, "0" * 16)
        rec, own = os.path.join(rdir, "replay.json"), os.path.join(rdir, "check.json")
        self.assertEqual(C.verify_replay_record(rec, row, res, rig)["state"], C.X_PASS)
        orig, own_orig = open(rec).read(), open(own).read()
        pos = json.load(open(ctl))["controls"][0]   # the batch's PASSING control (another witness, another property)
        # (a) the passing control's record copied in as this row's own
        write(own, open(pos["record"]).read())
        self._edit_json(rec, lambda d: d["check"].update(record_sha256=L.sha256_file(own)))
        v = C.verify_replay_record(rec, row, res, rig)
        self.assertEqual(v["state"], C.X_INCONCLUSIVE, v["why"])
        self.assertIn("not this witness's", v["why"])
        write(own, own_orig)
        write(rec, orig)
        # (b) the replay record repointed at the control's record where it lies
        self._edit_json(rec, lambda d: d["check"].update(record=pos["record"], record_sha256=pos["record_sha256"]))
        v = C.verify_replay_record(rec, row, res, rig)
        self.assertEqual(v["state"], C.X_INCONCLUSIVE, v["why"])
        self.assertIn("outside the replay directory", v["why"])
        write(rec, orig)
        # (c) another rig than the plan's
        self._edit_json(rec, lambda d: d["rig"].update(files_sha256="0" * 64))
        self.assertIn("another rig", C.verify_replay_record(rec, row, res, rig)["why"])
        write(rec, orig)
        # (d) this witness, but checked with another year's scoring code
        self._edit_json(own, lambda d: d.update(scoring=rig["scoring"]["2025"]))
        self._edit_json(rec, lambda d: d["check"].update(record_sha256=L.sha256_file(own)))
        self.assertIn("another scoring directory", C.verify_replay_record(rec, row, res, rig)["why"])
        write(own, own_orig)
        write(rec, orig)
        self.assertEqual(C.verify_replay_record(rec, row, res, rig)["state"], C.X_PASS, "restored: passes again")

    def test_review_v2_rows_need_v2_controls_and_their_own_job(self):
        import x86_replay as X
        w, rig0 = cert_world(self.tmp.name)
        row = self._row(w, "2.0")
        self.assertEqual(C.checker_kind(row), "v2-2026")
        res = write(os.path.join(self.tmp.name, "w.res"), "sat\n((X_0 0.25)\n (X_1 0.5))\n")
        dest = os.path.join(self.tmp.name, "x86")
        rig = self._rig(rig0)
        self.assertEqual(C.control_kinds(rig), {"v1-2026"})
        ctl = X.run_controls(rig, dest, "b1")
        self.assertTrue(json.load(open(ctl))["ok"])
        rdir = os.path.join(dest, row["tag"])
        X.replay_row(rig, row, res, rdir, ctl, "0" * 16)
        rec, own = os.path.join(rdir, "replay.json"), os.path.join(rdir, "check.json")
        v = C.verify_replay_record(rec, row, res, rig)
        self.assertEqual(v["state"], C.X_INCONCLUSIVE, "v1 controls prove nothing about the v2 code path")
        self.assertIn("no PASS and FAIL control of checker kind v2-2026", v["why"])
        # with a PASS and a FAIL control of its own kind (through the v2 path), the same witness passes
        v2c = [dict(c, name=c["name"] + "v2", kind="v2-2026", onnx=row["onnx"], vnnlib=row["vnnlib"])
               for c in rig0["controls"]]
        rig2 = self._rig(rig0, v2c)
        self.assertEqual(C.control_kinds(rig2), {"v1-2026", "v2-2026"})
        ctl2 = X.run_controls(rig2, dest, "b2")
        self.assertTrue(json.load(open(ctl2))["ok"], open(ctl2).read())
        X.replay_row(rig2, row, res, rdir, ctl2, "0" * 16)
        v = C.verify_replay_record(rec, row, res, rig2)
        self.assertEqual(v["state"], C.X_PASS, v["why"])
        orig, own_orig = open(rec).read(), open(own).read()
        # a v2 check record of another job, or one whose outputs were injected, is not this replay's
        for edit, want in ((lambda d: d.update(id="seal:0000000000000000:S8_26_toy_0009"), "is job"),
                           (lambda d: d.update(injected_from="/elsewhere.npz"), "not an official v2 check"),
                           (lambda d: d.update(platform="arm"), "platform")):
            self._edit_json(own, edit)
            self._edit_json(rec, lambda d: d["check"].update(record_sha256=L.sha256_file(own)))
            v = C.verify_replay_record(rec, row, res, rig2)
            self.assertEqual(v["state"], C.X_INCONCLUSIVE, want)
            self.assertIn(want, v["why"])
            write(own, own_orig)
            write(rec, orig)
        self._edit_json(rec, lambda d: d.update(id="seal:0000000000000000:S8_26_toy_0009"))
        self.assertIn("job id", C.verify_replay_record(rec, row, res, rig2)["why"])
        write(rec, orig)
        self.assertEqual(C.verify_replay_record(rec, row, res, rig2)["state"], C.X_PASS)

    def test_review_controls_rederive_from_their_own_records(self):
        import x86_replay as X
        w, rig0 = cert_world(self.tmp.name)
        rig = self._rig(rig0)
        ctl = X.run_controls(rig, os.path.join(self.tmp.name, "x86"), "b1")
        self.assertEqual(C.controls_problem({"path": ctl, "sha256": L.sha256_file(ctl)}, rig, "v1-2026"), "")
        self.assertIn("no PASS and FAIL control of checker kind v1-2025",
                      C.controls_problem({"path": ctl, "sha256": L.sha256_file(ctl)}, rig, "v1-2025"))
        orig = open(ctl).read()

        def edited(fn) -> str:
            self._edit_json(ctl, fn)
            why = C.controls_problem({"path": ctl, "sha256": L.sha256_file(ctl)}, rig, "v1-2026")
            write(ctl, orig)
            return why
        # the FAIL control's OWN record says the rig passed the bad witness: the file's `got` cannot hide it
        neg, pos = json.loads(orig)["controls"][1], json.loads(orig)["controls"][0]
        nrec = open(neg["record"]).read()
        self._edit_json(neg["record"], lambda d: d.update(result="correct"))
        self.assertIn("re-derived pass", edited(lambda d: d["controls"][1].update(
            record_sha256=L.sha256_file(neg["record"]), result="correct")))
        write(neg["record"], nrec)
        # the PASS control's record standing in for the FAIL one; a file that runs only the PASS control; a control
        # with another witness; another cpu; an edit without re-hashing
        self.assertIn("another witness", edited(lambda d: d["controls"][1].update(
            record=pos["record"], record_sha256=pos["record_sha256"], result=pos["result"])))
        self.assertIn("exactly the plan's controls", edited(lambda d: d.update(controls=d["controls"][:1])))
        self.assertIn("not the plan's control", edited(lambda d: d["controls"][1].update(ce_sha256="0" * 64)))
        self.assertIn("cpu", edited(lambda d: d.update(cpu="Nehalem")))
        self.assertIn("missing, changed", C.controls_problem({"path": ctl, "sha256": "0" * 64}, rig, "v1-2026"))
        self.assertEqual(C.controls_problem({"path": ctl, "sha256": L.sha256_file(ctl)}, rig, "v1-2026"), "")

    def test_review_refuted_is_an_emergency_whatever_the_bindings_say(self):
        r = _cert_files(os.path.join(self.tmp.name, "r"), verdict="refuted", commit="e" * 40)
        self.assertEqual(C.verify_certificate(r["report"], r["bundle"], r["onnx_sha"], r["vnn_sha"], "c" * 40,
                                              "0" * 64)["state"], "refuted",
                         "another build and another DONE sha: still a counterexample in THIS instance")
        self.assertEqual(C.verify_certificate(r["report"], r["bundle"], "0" * 64, r["vnn_sha"], "c" * 40)["state"],
                         "reject", "a refutation of ANOTHER instance is not this row's")
        lf = C.check_log_facts(open(r["log"]).read())
        self.assertEqual(C.check_certificate(dict(lf, done_report="/elsewhere.json"), r["report"], r["bundle"],
                                             r["onnx_sha"], r["vnn_sha"], "c" * 40)["state"], "refuted")

        def J1(v):
            return J("S8R_26_toy_0000", "proof", 1, v)
        for log, cert in ((dict(ran=True, ran_rc=21, done_rc=21, done_verdict="refuted"),
                           {"state": "reject", "why": "x"}),
                          (dict(ran=True, ran_rc=21), {}),
                          (dict(ran=True, ran_rc=124), {"state": "refuted", "why": "cex"})):
            o = C.reproof_outcome(J1("unsat"), {"bundle_exists": True, "log": log, "cert": cert})
            self.assertEqual(o["state"], C.R_EMERGENCY, (log, cert, o))
        self.assertEqual(C.reproof_outcome(J1("unsat"), {"bundle_exists": True, "log": dict(ran=True, ran_rc=124),
                                                         "cert": {}})["state"], C.R_FALLBACK)

    def test_review_accept_needs_the_checkers_own_done_line_and_exit(self):
        f = _cert_files(self.tmp.name)
        lf = C.check_log_facts(open(f["log"]).read())
        args = (f["report"], f["bundle"], f["onnx_sha"], f["vnn_sha"], "c" * 40)
        self.assertEqual(C.check_certificate(lf, *args)["state"], "accept")
        for k, v in (("ran_rc", 4), ("ran_rc", None), ("done_rc", 10), ("done_verdict", "unconfirmed"),
                     ("done_report", "/elsewhere.check.json"), ("done_report", None), ("done_sha", "0" * 64)):
            got = C.check_certificate(dict(lf, **{k: v}), *args)
            self.assertEqual(got["state"], "reject", (k, v, got))

    def test_review_a_kept_failing_replay_stops_a_resumed_s8c(self):
        w, rig = cert_world(self.tmp.name)
        plan, _ = certify(w, rig, mode="wrongsat")
        rc, done, log = w.run("out", "S8C")
        self.assertEqual(rc, 3, log[-3000:])
        # resumed with the EMERGENCY file moved aside (a triage): the earlier WRONG SAT still stops the replay phase
        os.replace(os.path.join(plan["out"], "EMERGENCY"), os.path.join(plan["out"], "EMERGENCY.triaged"))
        rc, done, log = w.run("out", "S8C")
        self.assertEqual(rc, 3, log[-3000:])
        self.assertIn("phase=replay", done)
        self.assertIn("X86_REPLAY_DONE rc=3 ", log)
        self.assertFalse(os.path.isfile(C.spec_path(plan["out"], "reproof")), "no re-proof box time after a wrong sat")

    def test_review_a_lost_proof_rerun_counts_as_lost_like_the_classic_proof_rep(self):
        rows, lines, rp = CertifiedUnit._decide_world(self)
        acc = {"bundle_exists": True, "log": {"ran": True, "done_rc": 0}, "report": "/r", "bundle": "/b",
               "cert": {"state": "accept", "why": "verified", "report_sha256": "a" * 64, "bundle_sha256": "b" * 64,
                        "verdict": "verified"}}
        # rp[0] is a BANKED-unsat row: its Proof re-run times out (outcome lost) and so falls back; the classic
        # sub-moat then holds the verdict in every rep -- the lost Proof re-run still counts, as the classic S8's
        # lost Proof rep would
        cl = C.derive_classic(rp, {rp[0]["tag"]: {"state": C.R_FALLBACK}, rp[1]["tag"]: {"state": C.R_ACCEPT}})
        plines = lines + [J(rp[0]["tag"], "proof", 1, "timeout", outcome="lost"),
                          J(rp[1]["tag"], "proof", 1, "unsat", outcome="held")]
        clines = plines + [J(cl[0]["tag"], p, k, "unsat", outcome="held") for p, k in S.M0J.schedule(cl[0])]
        facts = {"reproof_spec": rp, "classic_spec": cl, "checks": {rp[0]["tag"]: acc, rp[1]["tag"]: acc},
                 "replays": {rows[1]["tag"]: {"state": C.X_PASS, "why": "correct", "result": "correct"}}}
        d = C.decide(toy_plan(), rows, clines, facts)
        self.assertEqual((d["emergencies"], d["void"]), ([], []))
        self.assertEqual(d["certified"]["rows"][rows[0]["tag"]]["classic"]["state"], "covered")
        self.assertEqual(d["lost_runs"], 1, d["certified"]["counts"])
        self.assertTrue(d["certified"]["rows"][rows[0]["tag"]]["reproof"]["lost_run"])
        # and the stage judge makes it PARTIAL, never PASS
        plan = toy_plan(stages={"S8": {"mode": "certified"}})
        rec = S.judge_stage(plan, "S8", rows + rp + cl, clines, {}, [], runs_for(rows + rp + cl), None, facts)
        self.assertEqual(rec["status"], "PARTIAL", rec["reasons"])
        self.assertIn("LOST (tolerance 0)", " ".join(rec["reasons"]))

    def test_review_default_rig_discriminates_every_checker_kind(self):
        # a certified plan with the default rig must be able to replay a banked sat of every scored kind: v1-2025 and
        # v1-2026 (checkv1) AND v2-2026 (the 13 VNN-LIB 2.0 categories of 2026); the v2 pair lives in the repository,
        # byte-identical to the witnesses measured on the real rig 2026-10-08 (pos -> correct + ORT match, neg ->
        # spec_not_violated, scratchpad certified-moat-review/v2ctl)
        self.assertEqual(C.control_kinds(C.DEFAULT_RIG), {"v1-2025", "v1-2026", "v2-2026"})
        got = {c["name"]: (c["expect"], L.sha256_file(c["ce"])) for c in C.DEFAULT_RIG["controls"]
               if c["kind"] == "v2-2026"}
        self.assertEqual(got, {
            "pos2026v2": ("pass", "12f2ba6b7020bf9bd48077779cf50afd94390e179293e32aff22f5d2d70494ef"),
            "neg2026v2": ("fail", "e3c5b039bf24847a1beddfda4e36ba0a59cd4e087fe2251bf0c086ea24f551d9")})

    def test_review_plan_and_step0_refuse_a_rig_without_controls_of_a_needed_kind(self):
        import argparse
        import contextlib
        from unittest import mock
        import seal as SP
        w, rig = cert_world(self.tmp.name)
        r25 = dict(rig, controls=[dict(c, kind="v1-2025") for c in rig["controls"]])
        with self.assertRaisesRegex(S.SealError, r"no PASS \+ FAIL witness control of checker kind\(s\) \['v1-2026'\]"):
            w.plan("o1", stages=["S5", "S7", "S8"], s8_mode="certified", x86_rig=r25)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "o1", "plan.json")))
        plan, _ = certify(w, rig)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(SP.cmd_cert_step0(argparse.Namespace(out=plan["out"])), 0, buf.getvalue())
        # an armed-sat row whose kind the plan's controls cannot discriminate: refused before any box wait
        buf = io.StringIO()
        with mock.patch.object(C, "control_kinds", lambda r: set()), contextlib.redirect_stdout(buf):
            rc = SP.cmd_cert_step0(argparse.Namespace(out=plan["out"]))
        self.assertEqual(rc, 97, buf.getvalue())
        self.assertIn("no PASS + FAIL witness control of checker kind(s) ['v1-2026']", buf.getvalue())


class Vendoring(unittest.TestCase):
    """The vendored allonlib, in two contracts.

    The HERMETIC one runs with every selftest: the file's bytes are the pinned blob, and it judges with the same
    m0lib. The EXTERNAL one re-derives that blob from the commit it was vendored from; it needs that commit's
    objects, which live on `exp/all-on-preset` and in no ref of the shared remote, so most clones cannot run it.
    It used to be the tail of the hermetic test behind `self.skipTest`, which the repository forbids (a test either
    runs in the selected lane or is not part of it; TEST_CONFORMANCE.md, ny-test-utils source_policy) and which
    also skipped the m0lib identity check below on every clone without the commit. It is now the selftest's
    EXTERNAL lane, like the `external_*` pytest contracts: selected explicitly, and it FAILS when the commit is
    absent:

        python3 scripts/seal/seal_selftest.py --external

    `--external` runs every `external_*` method of this module and nothing else. Nothing selects it for you (no
    default run, no gate lane): run it on a host that holds the source commit WHENEVER `VENDORED_ALLONLIB` or
    `allonlib.py` changes. That is the one moment its answer can change, because it compares three constants.
    """

    def test_vendored_allonlib_is_byte_identical_to_its_source_commit(self):
        v = S.VENDORED_ALLONLIB
        blob = subprocess.run(["git", "hash-object", os.path.join(REPO, v["file"])], capture_output=True,
                              text=True).stdout.strip()
        self.assertEqual(blob, v["blob"], "scripts/seal/allonlib.py was edited: re-vendor, never fork")
        self.assertIs(S.A.L, S.L, "allonlib must judge with the same m0lib")

    def external_vendored_allonlib_matches_its_source_commit(self):
        v = S.VENDORED_ALLONLIB
        src = subprocess.run(["git", "-C", REPO, "rev-parse", f"{v['source_commit']}:{v['source_path']}"],
                             capture_output=True, text=True)
        self.assertEqual(src.returncode, 0,
                         f"the vendoring source commit {v['source_commit']} is not in this repository (fetch the "
                         f"branch that carries it, exp/all-on-preset): {src.stderr.strip()}")
        self.assertEqual(src.stdout.strip(), v["blob"])


class KeylessTail(unittest.TestCase):
    """review(prepared-root-route): the keyless moat plan's superset tail keeps a census refusal whose reasons do not
    bound where a default-on capability runs, and leaves out a row its own property gates provably refuse."""

    def setUp(self):
        import keyless_moat_plan as KM
        self.KM = KM
        self.gates = KM.LANE_PROPERTY_GATES["prepared-root"]
        self.served = ("robustness",)

    def cell(self, reasons: str, facts: str = "known/complete", spec: str = "robustness c=2 a=2 bx=0 or+rel"):
        return {"reasons": reasons, "facts": facts, "spec": spec}

    def test_keyless_tail_keeps_rows_the_lane_may_run_and_drops_rows_its_gates_refuse(self):
        KM, g, sv = self.KM, self.gates, self.served
        keep = [self.cell("envelope:ops"), self.cell("admit:unknown:facts-unknown", "unknown:scan>256MiB"),
                self.cell("envelope:ops", spec="unknown:spec-unparsed"),
                self.cell("admit:unknown:facts-unknown", "unknown:scan>256MiB", "unknown:spec-parse-panic")]
        for c in keep:
            self.assertTrue(KM.tail_only(c, g, sv), c)
            self.assertIsNone(KM.lane_refuses_spec(c, g, sv), c)
        drop = {
            "spec_read_refused": self.cell("envelope:ops", spec="unknown:spec>16MiB"),
            "archetype_not_served": self.cell("admit:unknown:facts-unknown", "unknown:scan>256MiB",
                                              "halfspace c=1 a=1 bx=0 -"),
        }
        for why, c in drop.items():
            self.assertTrue(KM.tail_only(c), f"{why}: a tail row by its reasons alone (positive control)")
            self.assertFalse(KM.tail_only(c, g, sv), c)
            self.assertEqual(KM.lane_refuses_spec(c, g, sv), why)
        for c in (self.cell("admit:unknown:facts-unknown", "unknown:scan>256MiB", "disjunctive c=2 a=2 bx=0 or"),
                  self.cell("envelope:ops", spec="nonlinear")):
            self.assertFalse(KM.tail_only(c, g, sv), c)
        # the reasons rule and the relational rule are unchanged
        self.assertFalse(KM.tail_only(self.cell("admit:decline:spec-not-served"), g, sv))
        self.assertFalse(KM.tail_only(self.cell("admit:decline:spec-not-served,envelope:ops"), g, sv))
        self.assertFalse(KM.tail_only(self.cell("admit:unknown:facts-unknown", "unknown:relational-route", "-"),
                                      g, sv))
        self.assertFalse(KM.tail_only(self.cell("-"), g, sv))
        # a capability with no recorded lane gate keeps every reasons-only tail row (the gates are opt-in facts)
        self.assertTrue(KM.tail_only(drop["spec_read_refused"], {}, sv))
        self.assertTrue(KM.tail_only(drop["archetype_not_served"], {}, sv))
        # every gate cites the named Rust tests that pin it
        self.assertEqual(set(g["tests"]), {
            "prepared_root_property_ceiling_is_below_the_router_spec_read_ceiling",
            "prepared_root_property_admission_is_inside_the_routers_robustness_archetype"})

    def test_keyless_derived_builds_against_the_merged_derived_and_alarms_every_row_alike(self):
        """merge eng16: #certified-moat made `sat_alarms` a required Derived field; the keyless plan's own Derived
        must construct (a TypeError here is a plan that cannot be written) and carry the defect alarms on every
        row, sat or unsat, with no sat-row alarm (a keyless capability names no closing marker)."""
        d = self.KM.keyless_derived("prepared-root", list(S.BOTH_YEARS))
        self.assertIsInstance(d, S.Derived)
        self.assertEqual(d.sat_alarms, [])
        self.assertEqual(d.alarms, list(S.A.DEFECT_ALARMS))
        self.assertTrue(d.routable)
        self.assertEqual((d.arm, d.keys, d.flight), ({}, {}, []))
        for truth, bank in (("sat", "sat"), ("unsat", "unsat"), ("sat", "-"), ("unsat", "sat")):
            self.assertEqual(S.row_alarms(d, truth, bank), list(S.A.DEFECT_ALARMS), (truth, bank))


def main() -> int:
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    names = sys.argv[1:]
    if names == ["--external"]:
        # The external lane: every `external_*` contract, each of which fails where its prerequisite is absent.
        loader = unittest.TestLoader()
        loader.testMethodPrefix = "external_"
        suite = loader.loadTestsFromModule(sys.modules[__name__])
    else:
        suite = (unittest.defaultTestLoader.loadTestsFromNames(names, sys.modules[__name__]) if names else
                 unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    res = runner.run(suite)
    sys.stdout.write(stream.getvalue())
    # A selection that ran nothing is not a pass (`--external` with no external contract left to run).
    rc = 0 if res.wasSuccessful() and res.testsRun > 0 else 1
    print(f"SEAL_SELFTEST_DONE rc={rc} ran={res.testsRun} failures={len(res.failures)} errors={len(res.errors)} "
          f"skipped={len(res.skipped)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
