<!-- Copyright 2026 Andrew Yates -->
<!-- Author: Andrew Yates <andrewyates.name@gmail.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hard-six MIP corpus

Exact MILP instances for the six `cifar100_2024` properties (prop_idx 885,
8945, 7641, 2176, 6415, 9897) that no sound verifier has proved. They are the
target workload for making `ay` a competitive MILP solver for neural-network
verification.

Branch-and-bound with CROWN-family bounds stalls on these properties. Each file
is an exact big-M MIP over one pinned branch-and-bound subdomain (the lineages
whose bounds stay between −0.45 and −1.14 at any budget), from the `w2` window
up to the **full network over the exact VNN-LIB input box**, with no per-layer
decomposition. Solving a `*_dec` instance UNSAT **verifies that spec row on
that pinned subdomain**. Only the full-depth model can: windowed optima lose
the correlation between layers (see w2 below).

## Formats

Every instance is generated in three forms from the same model:

- `*_dec.smt2` — **decision form** (primary): the margin column is asserted
  `<= 0`; **UNSAT ⇔ the subdomain is verified on that row**. SAT is a candidate
  violation that must be revalidated on the real f32 network.
- `*_min.smt2` — OMT form, `(minimize margin)`, for ay's optimization lane.
- `*.milp` — ny-mip `dump.rs` bit-pattern format (`milp v1`), loadable by
  `mip-diff` (`targo --unverified run -p ny-mip --bin mip-diff -- <file.milp>`
  diffs the `ay` and `ay-proc` backends; add `--certify` to check ay against
  its own certificates).

The `.smt2` dialect is byte-faithful to ny-mip's ay lowering
(`crates/ny-mip/src/ay/lower.rs`): QF_LRA, exact IEEE-754 rational literals,
ReLU binaries as `{0,1}` disjunctions.

Large files are committed zstd-compressed (`zstd -d` to unpack). The sha256
prefixes below identify the raw (uncompressed) files; `emit` prints each
output's full sha256.

### `--int-scale`: integer-scaled lowering (opt-in, not default)

`tools/emit_hard_six.py emit --int-scale` (and `tools/emit_tiny_head.py
--int-scale`) writes `_int_dec.smt2`/`_int_min.smt2`. Every f64 coefficient is
dyadic, so multiplying each affine row by `2^maxk` (`maxk` = the largest
denominator exponent over the row's coefficients and rhs) clears all
denominators exactly and leaves pure-integer literals with no `(/ …)`.
`verify_int_scale` spot-checks exactness with `fractions.Fraction` on a sample
of rows (the 48 densest plus an even stride; every row when there are at most
96); cols/rows/nnz/binaries are unchanged. SMT2 only: the scaled integers
exceed 2^53, so no `.milp` is written.

Measured on ay 0.11.0 (`--competition -t 400000`): int-scale did not change the
w2 (`prop8945 r99-67`) verdict (`unknown` at 402 s), cut its peak RSS
3.75→2.69 GB and its file 106→63 MB, and ran ~1.8× slower on the trivially
solvable `whead_full16` head rung from `emit_tiny_head.py` (z3 and ay both
`sat` in both forms). The bottleneck was ay's exact-rational LP pivot
representation, not the input form: the first pivot reintroduces fractions, so
clearing input denominators needs a fraction-free (Bareiss) pivot or a
certified numeric simplex in ay to pay off. Re-measure against the current ay
pin before changing the default.

## The instances

Network: `CIFAR100_resnet_medium.onnx` (gitignored; fetched by
`benchmarks/download_benchmarks.sh` into
`benchmarks/vnncomp2025/benchmarks/cifar100_2024/onnx/`), ε=0.0039 ℓ∞ boxes:
prop 8945 = `prop_idx_8945_sidx_3584`, prop 885 = `prop_idx_885_sidx_7654`.
Windows (`--win 2|3|5|full`, file suffix `_w2`/`_w3`/`_w5`/`_wfull`): `w2` =
last residual block + `Gemm_56→Relu_57→Gemm_58` tail; `w5` = last 4 blocks +
tail (`Relu_39/45/51` are stable on these boxes, so narrower windows such as
`w3` are no stronger than w2); `full` = whole net from the exact vnnlib input
box (only the intermediate big-M boxes are DELTA-inflated; see Soundness).

### G3-gate targets (full depth)

| instance | cols | rows | nnz | binaries | CROWN lb | HiGHS baseline |
|---|---|---|---|---|---|---|
| `cifar100med_prop8945_dom1_d8_r99-67_wfull` | 125121 | 106486 | 44.4M | 494 | −0.447 | **died: no LP finish at 85 min / 10 GB** |
| `cifar100med_prop8945_dom1_d8_r99-73_wfull` | 125121 | 106486 | 44.4M | 494 | −0.170 | (same model, easier row) |

Unstable-ReLU census (the nonconvexity lives in the stem and first blocks,
upstream of every tractable window): Relu_2:235, Relu_5:135, Relu_13:56,
Relu_19:15, Relu_31:37, Relu_57:16.

Local only (too big for git even compressed; regenerate in ~2 min, see below):
raw sha256 `107bdf2f…` (r99-67 dec, 2.66 GB), `be0aeed8…` (r99-67 min),
`cab67730…` (r99-67 milp, 1.04 GB), `a2c3cf00…` (r99-73 dec), `3c20a7eb…`
(r99-73 min), `19f17bfe…` (r99-73 milp).

### G3 ladder rung: w5 (where HiGHS's LP died)

| instance | cols | rows | nnz | binaries | CROWN lb | HiGHS baseline |
|---|---|---|---|---|---|---|
| `cifar100med_prop8945_dom1_d8_r99-67_w5` | 26831 | 18692 | 7.47M | 53 | −0.447 | **LP hit 1806 s limit, MIP none; dual −5.46** |
| `cifar100med_prop8945_dom1_d8_r99-73_w5` | 26831 | 18692 | 7.47M | 53 | −0.170 | (same model) |
| `cifar100med_prop885_dom36_d8_r44-93_w5` | 26891 | 18782 | 7.51M | 83 | −1.139 | unmeasured (same class) |

Committed: the three `*_w5.milp.zst` and
`cifar100med_prop8945_dom1_d8_r99-67_w5_dec.smt2.zst`; the other w5 smt2 forms
are local only. Raw sha256: r99-67 `9ed5a06e…`/`0e32dd68…`/`e9278367…`
(dec/min/milp), r99-73 `2bcc8fa3…` (milp), 885 `b5781e63…`/`592aee2d…`/
`5d68c198…` (dec/min/milp).

### MEDIUM: w2, the P1/P2 speed benchmarks (HiGHS solves in minutes)

| instance | cols | rows | nnz | binaries | CROWN lb | HiGHS LP/MIP (measured) |
|---|---|---|---|---|---|---|
| `cifar100med_prop8945_dom1_d8_r99-67_w2` | 6277 | 4245 | 1.85M | 16 | −0.447 | −4.258 / −4.212 opt, 10 s / 449 s |
| `cifar100med_prop8945_dom1_d8_r99-73_w2` | 6277 | 4245 | 1.85M | 16 | −0.170 | −4.478 / −4.423 opt, 6 s / 215 s |
| `cifar100med_prop885_dom36_d8_r44-93_w2` | 6303 | 4284 | 1.85M | 29 | −1.139 | −6.633 / −6.583 opt, 7 s / 88 s |

Committed: the three `*_w2_dec.smt2.zst` and `*_w2.milp.zst`. These set the
wall-clock ladder on the same files: **P1 (LP core) should beat the 6–10 s HiGHS
LP times; P2 (branch-and-cut) the 88–449 s MIP times**. The MIP optima are 3–5
units below CROWN: windowed MIPs lose the correlation between layers and cannot
verify these rows, which is why the full-depth targets matter and why the
production path harvests certified rows (cuts) rather than window verdicts.

## Soundness

- All intermediate boxes are per-subdomain **refined bounds dumped from the
  live NY run** (premise-conditioned `interm_refine` caches), inflated by
  DELTA=1e-4 to absorb the measured ≤1.1e-5 f32-net vs f64-affine gap.
  Inflation only weakens the model: **UNSAT verdicts are sound** for the real
  f32 network under the same float caveat NY's own paths carry.
- The full-depth input box is the **exact** vnnlib box (not inflated).
- Premise clamps (the BaB split literals, `prem=` in the probe dump) are
  applied to the `Gemm_56` output bounds: each instance is one BaB leaf.
- Big-M ReLU encoding is exact given valid pre-activation boxes; stable
  neurons are linearized (positive) or dropped (negative) per the boxes.
- The HiGHS baselines are offline reference timings that never touched a
  verdict path. HiGHS is no longer in the tree; the standing check is
  `mip-diff --certify`, which holds ay to its own certificates.

## Regeneration

```bash
# domains/*.npz (committed) hold everything per-subdomain; maps/ (local,
# 573 MB) holds the exact affine-map dumps; probe-logs/ (local) the raw dumps.
python3 tools/emit_hard_six.py emit \
  --domain domains/prop8945_dom1_d8.npz --maps maps/ --win full \
  --vnnlib ../../../../benchmarks/vnncomp2025/benchmarks/cifar100_2024/vnnlib/CIFAR100_resnet_medium_prop_idx_8945_sidx_3584_eps_0.0039.vnnlib \
  --out instances/            # needs numpy+scipy; ~2 min for full
```

`emit` writes every spec row with a negative CROWN lb; `--row 99-67` selects
one.

- `maps/` is rebuilt from the benchmark ONNX by
  `tools/provenance/build_block_maps*.py`. They read `model_taps.onnx` (the
  benchmark ONNX with intermediate tensors exposed as extra graph outputs;
  initializers identical) and validate against onnxruntime. They are as-run
  copies: edit their hardcoded scratchpad paths before running.
- New pinned domains: rerun the probe with `tools/provenance/run_dumps.sh` (an
  as-run copy with hardcoded paths; it sets the full-stack env plus
  `NY_C3_57_PROBE=1 NY_SUFFIX_MIP_DUMP=1`; add `NY_SUFFIX_MIP_DUMP_NODES` for
  the stem/early-block boxes `--win full` needs), then
  `tools/emit_hard_six.py extract`.
- **Open follow-ups:** prop885 full depth (its probe logs lack stem-node
  boxes; re-dump with `NY_SUFFIX_MIP_DUMP_NODES`), and the prop54/1588/7258
  w2 dumps (the `-0.86`-plateau class, more MEDIUM rungs).

## What "solved" buys, per rung

- **w2 optimal faster than HiGHS (P1/P2):** ay is competitive on real NN
  workloads.
- **w5 LP finished at all:** ay's LP core gets past the wall where HiGHS's
  simplex died, on a 7.5M-nnz NN LP.
- **full-depth dec UNSAT (G3/G4, any budget):** the first sound verification
  of a hard-six pinned subdomain by exact reasoning. NY can then request it
  per subdomain through `MipBackend::Ay`, or more cheaply fold harvested
  Farkas-certified rows into CROWN as λ≥0 duals; ay's certificates keep it
  inside the proof boundary.
