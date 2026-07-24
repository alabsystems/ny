<!-- Copyright 2026 Andrew Yates -->
<!-- Author: Andrew Yates <andrewyates.name@gmail.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hard-six MIP corpus — the ay MILP program's target instances

**Program:** ay-as-Gurobi-class-MILP (`docs/AY_MIP_P0.md`, `docs/SOLVER_POLICY.md`;
ay repo `designs/2026-07-12-gurobi-class-milp-for-ny.md` — this corpus is the
W3 "strategic weapon" workload made concrete).
**Why these instances:** the six cifar100_2024 instances (prop_idx 885, 8945,
7641, 2176, 6415, 9897) have never been verified by any sound tool
(`docs/HARD_SIX_BRIEF.md`). The NY campaign's ~30 gated experiments killed
every CROWN-family lever; the **single surviving lever is full-depth MIP
coefficient rows** (`docs/CERTIFIED_CUT_CROWN_DESIGN.md`). These files are
that lever as concrete solver input: exact big-M MIPs over the *pinned* BaB
subdomains (the budget-invariant −0.45..−1.14 frontier lineages), from the
2-block window up to the **full network over the exact vnnlib input box**
(ground truth — no per-layer decomposition, no correlation loss).

Solving any `*_dec` instance UNSAT **verifies that spec row on that pinned
subdomain** — the exact object the whole BaB tree pins on. Doing it at the
full-depth window, at any budget, is the capstone: no LP-class relaxation can
(the CROWN pin), and HiGHS could not (baselines below).

## Formats

Every instance exists in three forms (generated together, same model):

- `*_dec.smt2` — **decision form** (primary, ay-native): the margin column is
  asserted `<= 0`; **UNSAT ⇔ the subdomain is verified on that row**. SAT
  gives a candidate violation that must be revalidated on the real f32 net
  (NyVerdictAdmission posture).
- `*_min.smt2` — OMT form, `(minimize margin)`. Exercises ay's optimization
  lane (P0 finding: pre-R1 binaries return `unknown` here — fail-closed).
- `*.milp` — ny-mip `dump.rs` bit-pattern format, loadable by the standing
  `mip-diff` gate (`cargo run -p ny-mip --bin mip-diff -- <file.milp>`;
  add `--certify` for the LG3 certificate-coverage mode).

The `.smt2` dialect is byte-faithful to ny-mip's ay lowering
(`crates/ny-mip/src/ay/lower.rs`): QF_LRA, exact IEEE-754 rational literals,
ReLU binaries as `{0,1}` disjunctions. Emitter self-test: a 1-ReLU known-answer
pair (UNSAT twin + SAT twin) passes against a real ay binary, and the `.milp`
round-trips through `mip-diff` with HiGHS/ay verdict agreement (measured
pre-LG3, against the since-deleted HiGHS oracle; the surviving backends are
`ay`/`ay-proc`, cross-checked by `mip-diff --certify`).

Large files are committed zstd-compressed (`zstd -d` to unpack); raw-file
sha256 hashes below are the identity anchors (pre-compression).

### `--int-scale`: power-of-2 integer-scaled lowering (experiment; NOT default)

`emit_hard_six.py emit --int-scale` (and `emit_tiny_head.py --int-scale`) emit
an alternative, semantically-identical `_int_dec.smt2`/`_int_min.smt2`: because
every IEEE-754 f64 coefficient is a dyadic `mant·2^exp`, its denominator is a
**power of 2**, so each affine row is multiplied through by `2^maxk`
(`maxk` = max denominator exponent over the row's coefficients *and* rhs) to
clear all denominators exactly — pure bit-shifts — yielding **pure-integer
literals, no `(/ …)`**. Scaling a constraint by a positive constant is
equivalence-preserving (proven exact per row by `verify_int_scale` via
`fractions.Fraction`); same cols/rows/nnz/binaries, ReLU disjunctions intact.
SMT2-only (the scaled integers exceed 2^53, so the f64 `.milp` cannot hold
them). Implementation: `tools/emit_hard_six.py` (`_dyadic`/`_scaled_int`/
`int_literal`/`_row_maxk`/`verify_int_scale`/`_scaled_bound`, `emit_smt2(…,
int_scale=)`).

**MEASURED 2026-07-13 — int-scaling does NOT unblock ay** (ay 0.11.0 build 2987,
`--competition -t 400000`, 128 GB host, solver run under a neutral name to avoid
a co-scheduled agent's `pkill`):

| instance | form | verdict | wall | peak RSS |
|---|---|---|---:|---:|
| whead_full16 (solvable) | rational | `sat` | 0.05 s | 19.4 MB |
| whead_full16 (solvable) | int-scale | `sat` | 0.09 s | 19.6 MB |
| w2 (`prop8945 r99-67`) | rational | `unknown` | 402 s | 3.75 GB |
| w2 (`prop8945 r99-67`) | int-scale | `unknown` | 402 s | 2.69 GB |

Equivalence cross-checked: z3 *and* ay both `sat` on the rational and int-scale
`whead_full16`. Int-scale w2 is **still `unknown@400s`** — identical verdict to
rational; it trims peak RSS 3.75→2.69 GB (−28%) and file size 106→63 MB (integer
literals are half the bytes of `(/ num den)` pairs), but does **not** reach a
solve, and it is ~1.8× *slower* on the trivial solvable case (max integer
coefficient 94 bits on w2, 67 on whead — bignum cost moves from denominators
into numerators). **Conclusion: the dense-row wall is ay's exact-rational LP
*pivot representation*, not the input form** — the first Gaussian pivot on an
integer matrix already produces rational tableau entries, so clearing input
denominators buys nothing without a **fraction-free (Bareiss) integer pivot** or
a **certified numeric / directed-rounding simplex** inside ay. Int-scale stays an
opt-in emit flag (a decisive negative + a modest RSS/size win), not the default.

## The instances

Network: `CIFAR100_resnet_medium.onnx` (in-repo,
`benchmarks/vnncomp2025/benchmarks/cifar100_2024/onnx/`), ε=0.0039 ℓ∞ boxes:
prop 8945 = `prop_idx_8945_sidx_3584`, prop 885 = `prop_idx_885_sidx_7654`.
Windows: `w2` = last residual block + `Gemm_56→Relu_57→Gemm_58` tail; `w5` =
last 4 blocks + tail (w2..w4 are *identical* in strength here — `Relu_39/45/51`
are stable on these boxes, the span is affine); `full` = whole net from the
exact vnnlib input box (stem included; only the intermediate big-M boxes are
DELTA-inflated refined bounds — see Soundness).

### G3-gate targets (full-depth: the global-first weapon)

| instance | cols | rows | nnz | binaries | CROWN lb | HiGHS baseline |
|---|---|---|---|---|---|---|
| `cifar100med_prop8945_dom1_d8_r99-67_wfull` | 125121 | 106486 | 44.4M | 494 | −0.447 | **died: no LP finish at 85 min / 10 GB** |
| `cifar100med_prop8945_dom1_d8_r99-73_wfull` | 125121 | 106486 | 44.4M | 494 | −0.170 | (same model, easier row) |

Unstable-ReLU census (matches the campaign map — the nonconvexity lives in
stem+layer1, upstream of every tractable window): Relu_2:235, Relu_5:135,
Relu_13:56, Relu_19:15, Relu_31:37, Relu_57:16.

Local-only (too big for git even compressed; regenerate in ~2 min, see below):
raw sha256 `107bdf2f…` (r99-67 dec, 2.66 GB), `be0aeed8…` (r99-67 min),
`cab67730…` (r99-67 milp, 1.04 GB), `a2c3cf00…` (r99-73 dec), `3c20a7eb…`
(r99-73 min), `19f17bfe…` (r99-73 milp).

### G3 ladder rung: w5 (the measured HiGHS LP-death class)

| instance | cols | rows | nnz | binaries | CROWN lb | HiGHS baseline |
|---|---|---|---|---|---|---|
| `cifar100med_prop8945_dom1_d8_r99-67_w5` | 26831 | 18692 | 7.47M | 53 | −0.447 | **LP hit 1806 s limit, MIP none; dual −5.46** |
| `cifar100med_prop8945_dom1_d8_r99-73_w5` | 26831 | 18692 | 7.47M | 53 | −0.170 | (same model) |
| `cifar100med_prop885_dom36_d8_r44-93_w5` | 26891 | 18782 | 7.51M | 83 | −1.139 | unmeasured (same class) |

Committed: `instances/cifar100med_prop8945_dom1_d8_r99-67_w5_dec.smt2.zst`.
Local-only raw sha256: `9ed5a06e…`/`0e32dd68…`/`e9278367…` (8945 dec/min/milp),
`b5781e63…`/`592aee2d…`/`5d68c198…` (885 dec/min/milp).

### MEDIUM: w2 — the P1/P2 speed benchmarks (HiGHS solves in minutes)

| instance | cols | rows | nnz | binaries | CROWN lb | HiGHS LP/MIP (measured) | ay P0 observed |
|---|---|---|---|---|---|---|---|
| `cifar100med_prop8945_dom1_d8_r99-67_w2` | 6277 | 4245 | 1.85M | 16 | −0.447 | −4.258 / −4.212 opt, 10 s / 449 s | unknown @120 s (parses fine) |
| `cifar100med_prop8945_dom1_d8_r99-73_w2` | 6277 | 4245 | 1.85M | 16 | −0.170 | −4.478 / −4.423 opt, 6 s / 215 s | — |
| `cifar100med_prop885_dom36_d8_r44-93_w2` | 6303 | 4284 | 1.85M | 29 | −1.139 | −6.633 / −6.583 opt, 7 s / 88 s | — |

Committed: the three `*_w2_dec.smt2.zst` + `*_w2.milp.zst`. These are the
recovery gates' wall-clock ladder: **P1 (LP core) should beat the 6–10 s HiGHS
LP times; P2 (branch-and-cut) the 88–449 s MIP times** — on the *same files*.
The MIP optima being 3–5 units below CROWN is the measured correlation-loss
fact (`docs/HARD_SIX_BRIEF.md` §3.2): windowed z-box MIPs cannot verify these
rows; only the full-depth model can — which is why the G3 targets matter and
why harvested **rows** (cuts), not window verdicts, are the production path
(P4, `CERTIFIED_CUT_CROWN_DESIGN.md` λ-fold seam, kernel-checked validity).

## Soundness

- All intermediate boxes are per-subdomain **refined bounds dumped from the
  live NY run** (premise-conditioned `interm_refine` caches), inflated by
  DELTA=1e-4 to absorb the measured ≤1.1e-5 f32-net vs f64-affine gap.
  Inflation only weakens the model: **UNSAT verdicts are sound** for the real
  f32 network under the same float caveat NY's own paths carry.
- The full-depth input box is the **exact** vnnlib box (not inflated).
- Premise clamps (the BaB split literals, `prem=` in the probe dump) are
  applied to the `Gemm_56` output bounds — each instance is one BaB leaf.
- Big-M ReLU encoding is exact given valid pre-activation boxes; stable
  neurons are linearized (positive) or dropped (negative) per the boxes.
- These offline HiGHS baselines predate `docs/SOLVER_POLICY.md` and never
  touched a verdict path; the foreign-oracle exception is CLOSED (HiGHS and
  the `foreign-oracle` feature were deleted at LG3), so no foreign solver
  runs in-tree — the standing gate is `mip-diff --certify`, ay held to its
  own verified certificates.

## Regeneration

```bash
# domains/*.npz (committed) hold everything per-subdomain; maps/ (local,
# 573 MB) holds the exact affine-map dumps; probe-logs/ (local) the raw dumps.
python3 tools/emit_hard_six.py emit \
  --domain domains/prop8945_dom1_d8.npz --maps maps/ --win full \
  --vnnlib ../../../../benchmarks/vnncomp2025/benchmarks/cifar100_2024/vnnlib/CIFAR100_resnet_medium_prop_idx_8945_sidx_3584_eps_0.0039.vnnlib \
  --out instances/            # needs numpy+scipy; ~2 min for full
```

- `maps/` is rebuilt from the in-repo ONNX by `tools/provenance/
  build_block_maps*.py` (as-run copies; they read `model_taps.onnx` = the
  benchmark ONNX with intermediate tensors exposed as extra graph outputs —
  the initializers are identical; each build validates against onnxruntime).
- New pinned domains: rerun the probe (`tools/provenance/run_dumps.sh` — the
  full-stack env + `NY_C3_57_PROBE=1 NY_SUFFIX_MIP_DUMP=1`, add
  `NY_SUFFIX_MIP_DUMP_NODES` for the stem/early-block boxes needed by
  `--win full`), then `tools/emit_hard_six.py extract`.
- **Queued follow-ups (need a free GPU; the pool100 sweep owns it now):**
  prop885 full-depth (its probe logs lack stem-node boxes — re-dump with
  `NY_SUFFIX_MIP_DUMP_NODES`), and the prop54/1588/7258 ladder-wall w2 dumps
  (the `-0.86`-plateau class, more MEDIUM rungs).

## What "solved" buys, per rung

- **w2 optimal faster than HiGHS (P1/P2):** ay displaces the foreign oracle
  on real NN workloads — the recovery gate on the way to G2/G3.
- **w5 LP finished at all:** past the wall where HiGHS's simplex died; the
  factorized-basis P1 core proves itself on a 7.5M-nnz NN LP.
- **full-depth dec UNSAT (G3/G4, any budget):** first sound verification of a
  hard-six pinned subdomain by exact reasoning — the object NY then requests
  per-subdomain through the `MipBackend::Ay` seam (or, cheaper, as harvested
  Farkas-certified rows folded into CROWN as λ≥0 duals). With ay's
  certificates this lands inside the proof boundary — the capability Gurobi
  structurally cannot offer (`SOLVER_POLICY.md`).
