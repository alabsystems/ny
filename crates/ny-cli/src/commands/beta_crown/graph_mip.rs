// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// DAG-aware MIP lowering (increments 1-4: Linear / ReLU / Flatten / Reshape /
// BatchNorm / residual Add / Conv2d — the full cifar100 resnet topology). Walks
// a `GraphNetwork` in topological order and builds a solver-neutral
// `ny_mip::MilpProblem`, assigning a block of MIP columns to each node's output
// tensor. MaxPool / everything else is still rejected (later increments).
//
// SOUNDNESS: every emitted row is an EXACT encoding of the f64-affine op — no
// relaxation is introduced beyond the big-M ReLU relaxation
// `ny_mip::encoder::encode_relu` already uses (keyed on the sound per-node
// pre-activation bounds):
//   * Linear   `y_i - Σ W_ij x_j = b_i`            (exact affine equality)
//   * Conv2d   im2col → the SAME Linear equality rows (increment 4, exact —
//              reuses `mip_preprocess::unfold_conv2d_to_linear`)
//   * BatchNorm `y_i - a_c·x_i = b_c`   (increment 2, exact per-channel affine)
//   * Add      `out_i - A_i - B_i = 0`  (increment 3, exact element-wise sum)
//
// DELTA box inflation (increment 4 — THE cifar100 soundness mechanism). Every
// f64-affine equality above uses ny's f32-rounded weights/biases as exact
// rational coefficients, so the MIP models the f64 semantics of the *f32* net.
// Per `ny-mip/corpus/hard-six/tools/emit_hard_six.py` (the ground-truth cifar100
// MIP generator), the measured f32-net vs f64-affine gap is ≤ 1.1e-5 at the
// Gemm_56 output. We absorb it by inflating EVERY per-node intermediate box the
// caller supplies by `DELTA = 1e-4` (`lo-DELTA, hi+DELTA`) before it becomes a
// column bound or a ReLU big-M `[pl, pu]`. Box inflation only WEAKENS the model:
// the inflated feasible set ⊇ (real f32 reachable set ∩ input box ∩ subdomain),
// so an ay `unsat` still certifies the real f32 network under the same float
// caveat ny's own bound-propagation paths carry (emit_hard_six.py header
// "Soundness posture"). The network INPUT box is used exactly (NOT inflated),
// matching emit_hard_six.py's `xl, xu = xin  # exact, NOT inflated`. This box
// inflation is what lets BatchNorm keep its EXACT f32-folded equality row while
// remaining sound against the real (pre-fold) affine — it absorbs the per-channel
// `scale_err`/`bias_err` fold error, so the increment-1..3 fail-closed BN gate is
// removed (a nonzero fold error ≤ DELTA no longer needs a range row).
//
// On a Linear+ReLU chain with `delta == 0.0` the encoding stays byte-for-byte
// equivalent to `ny_mip::encode_feedforward` (the increment-1 invariant); the
// production path uses `delta == DELTA`. BatchNorm, Add, and Conv2d only add
// equality rows / columns, so an ay `unsat` still certifies the subdomain (spec
// §"Soundness contract"). BatchNorm is encoded standalone (one affine row per
// element) rather than folded into an adjacent Linear/Conv — a standalone affine
// row is already exact and much simpler; folding is a future zero-column
// optimization, not a soundness requirement.
//
// Increment 5 — nn4sys mscn coverage (mscn_128d / mscn_128d_dual, all-UNSAT
// cardinality instances). Empirically (see `mscn_introspect_layer_variants`),
// ny's loader lowers those models to: Slice (incl. Split → per-output Slice),
// Linear (MatMul-by-initializer and Gemm; row-batched: `[R, in] -> [R, out]`),
// AddConstant, ReLU, MulBinary (hidden × input-derived mask), ReduceSum,
// Div (masked-sum / mask-count), Concat (3-ary), Sigmoid (final node; Sub of
// two Sigmoids for the dual). New EXACT encodings (no new relaxations):
//   * Slice / Gather / Concat / Squeeze / Unsqueeze — pure index plumbing:
//     output columns ARE the input columns (aliasing), using each layer's
//     exact forward index math (validated against the loader's declared
//     shapes; fail-closed on any inconsistency).
//   * Linear (row-batched), AddConstant / SubConstant / MulConstant /
//     DivConstant (broadcast-exact), ReduceSum, Sub — exact affine rows.
//     DivConstant emits `x - c·y = 0` (NOT `y = (1/c)·x`): multiplying by the
//     divisor keeps the row exactly rational with no reciprocal rounding.
//   * MulBinary / Div with a VARIABLE second operand are bilinear (not
//     affine) in general → fail-closed, EXCEPT when one operand is PINNED by
//     the constraint system: a column whose bounds are `[v, v]` (the exact,
//     never-inflated instance input box fixes it — mscn's masks), or a
//     ReduceSum of pinned columns whose f64 sum is verified EXACT. On the MIP
//     feasible set such an operand ≡ v, so `y = v·x` (Mul) / `a - β·y = 0`
//     (Div, β = pinned denominator ≠ 0) have EXACTLY the same feasible set as
//     the bilinear original — an exact encoding, not a relaxation.
//   * Sigmoid is NEVER encoded. `encode_graph_peel_final_sigmoid` peels a
//     Sigmoid OUTPUT node (sigmoid is a strict monotone bijection, so a const
//     band on the output is a const band on the logit) and the caller maps
//     thresholds through `logit_upper_threshold` / `logit_lower_threshold`,
//     which round the f64 inverse OUTWARD so the transformed constraint is
//     never stricter than the original. Sigmoid anywhere else fails closed
//     (this is what still blocks mscn_128d_dual: its output is
//     Sub(Sigmoid, Sigmoid), so the two Sigmoids are not final).
//
// Production wiring lives in `graph_mip_escalate`: the dispatch fallback's
// "no sequential form" arm uses its strict, resource-gated planner by default;
// `NY_GRAPH_MIP=0` disables the whole-net escalation. The per-subdomain LEAF
// lane is also default-on (`NY_GRAPH_MIP_LEAF=0` disables it; see
// `docs/GRAPH_MIP_LEAF_SOLVER.md`) and reuses this encoder with premise pinning
// via `binary_keys`.
//
// Spec: docs/GRAPH_MIP_ENCODER_FOR_CIFAR100.md (increment 1).
//
// Increment 5b introduced the older direct escalation entry
// ([`try_graph_mip_escalation`]). It remains for tests/reference, but dispatch
// no longer calls it after the strict planner declines: doing so would bypass
// the strict path's pre-encode NNZ envelope. This legacy entry is UNSAT-ONLY BY
// CONSTRUCTION: its success value ([`GraphMipCertifiedUnsat`]) cannot carry a
// counterexample, so it either certifies every violation clause or returns
// `None`.
//
// Spec: docs/GRAPH_MIP_ENCODER_FOR_CIFAR100.md (increment 1).
//
// Some helpers (e.g. `encode_graph_with_delta`) are still test-only consumers;
// keep the allow until the cifar100 full-depth call site lands.
#![allow(dead_code)]

use std::collections::HashMap;
use std::mem::size_of;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use ndarray::{Array1, Array2, ArrayD};
use ny_core::Bound;
use ny_mip::ir::{Col, MilpProblem};
use ny_mip::{
    certify_linear_lower_bound_at_with_ay_admission, certify_linear_lower_bound_with_ay_admission,
    CertifiedLinearLowerBound, CertifiedLinearLowerBoundConfig, CertifiedLinearLowerDecisionConfig,
    CertifiedLinearLowerWorkerAdmission, MipBackend, MipConfig, MipParts, MipResult, MipSolver,
    CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES,
};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_onnx::{load_onnx_with_config, CompoundNodePolicy, GraphNetworkOptions, OnnxLoadConfig};
use ny_propagate::imb::{
    AyTailAffineReachabilityEnvelope, AyTailSharedInputReachabilityEnvelope,
    AY_TAIL_AFFINE_REACHABILITY_ROWS,
};
use ny_propagate::layers::{
    BatchNormLayer, Conv2dLayer, GatherLayer, LinearLayer, ReduceSumLayer, SliceLayer,
};
use ny_propagate::shape::broadcast_shapes;
use ny_propagate::types::BoundsProvenance;
use ny_propagate::{AlphaCrownConfig, GraphNetwork, Layer, Network, Verifier, NETWORK_INPUT};
use tracing::info;

use super::mip_preprocess::unfold_conv2d_to_linear;

/// f32-net vs f64-affine semantic-gap absorber for the cifar100 resnet MIP.
///
/// Every per-node intermediate box the caller supplies is inflated by `±DELTA`
/// before it becomes a column bound or a ReLU big-M `[pl, pu]`. The value
/// (`1e-4`) is copied verbatim from `ny-mip/corpus/hard-six/tools/emit_hard_six.py`
/// (`DELTA = 1e-4`), which measured the real gap at ≤ `1.1e-5`. Inflation only
/// WEAKENS the model, so an ay `unsat` on the inflated MIP still certifies the
/// real f32 network (module header §"DELTA box inflation"). The network INPUT
/// box is used EXACTLY (not inflated), matching emit_hard_six.py.
pub(crate) const DELTA: f64 = 1e-4;

fn graph_mip_enabled_from_value(value: Option<&str>) -> bool {
    value != Some("0")
}

/// Whether the DAG-aware MIP escalation is enabled at its call site
/// (`dispatch::run_bab_with_fallback`, the graph-only `Err` reload arm).
///
/// DEFAULT-ON (2026-07-21): the whole-net Graph-MIP escalation is verdict-SOUND
/// (certified-UNSAT / graph-forward-revalidated-SAT only; any other outcome
/// keeps the BaB verdict) and now declines CHEAPLY on over-scale graph
/// instances via the memory-safe encode-nnz cap (5M, was 100M) + the ay-solver
/// binary ladder (`graph_mip_max_binaries`, default 1024). Instances beyond
/// either cap decline before encoding. `NY_GRAPH_MIP=0` restores the old off
/// path (byte-identical BaB). Every enabled value, including unset, uses the
/// same enable predicate as the phase ledger; the ledger additionally requires
/// the active phase policy to request a nonzero MIP slice before BaB, and Auto
/// escalation declines a zero-reservation policy before any bounds work.
/// Explicit MIP remains an override. This is not a claim of finite-budget
/// score monotonicity: any eligible escalation still needs A/B measurement.
/// Raise the caps only with a requalified solver and memory envelope (the north
/// star: ay-milp > Gurobi).
pub(crate) fn graph_mip_enabled() -> bool {
    graph_mip_enabled_from_value(std::env::var("NY_GRAPH_MIP").ok().as_deref())
}

// The post-seam cGAN proof is deliberately much smaller than a whole-network
// Graph-MIP, but the affine convolution rows can still be wide. These immutable
// caps decline before AY exact replay becomes an unpriced memory workload.
const IMB_AY_TAIL_MAX_NODES: usize = 32;
const IMB_AY_TAIL_MAX_INPUTS: usize = 65_536;
const IMB_AY_TAIL_MAX_OUTPUTS: usize = 1_024;
// The sealed cGAN row-5 tail at `Relu_17` has 112 unstable ReLUs after the
// mandatory seam-box-only IBP replay.  A 64-binary ceiling therefore rejected
// the exact authority lane before AY saw an otherwise small 7,716-column,
// 4,945-row encoding.  Admit that measured shape with a narrow power-of-two
// ceiling; the independent column, row, and 20M-nnz caps still bound encoder
// memory, while AY's 45-second solve deadline remains fail-closed.
const IMB_AY_TAIL_MAX_BINARIES: usize = 128;
const IMB_AY_TAIL_MAX_COLS: usize = 131_072;
const IMB_AY_TAIL_MAX_ROWS: usize = 131_072;
const IMB_AY_TAIL_MAX_NNZ: usize = 20_000_000;
/// The relational prefix image is intentionally tiny in its latent dimension:
/// sealed cGAN has five original inputs (four free). Decline before allocating
/// an unexpectedly wide auxiliary block.
const IMB_AY_TAIL_AFFINE_MAX_LATENT_INPUTS: usize = 16;
/// K=2 emits one lower and one upper row per support direction.
const IMB_AY_TAIL_AFFINE_ROWS: usize = 2 * AY_TAIL_AFFINE_REACHABILITY_ROWS;
/// The sealed 2×2048 support bank costs about 8.2k nonzeros. Keep a narrow,
/// immutable envelope around that measured shape in addition to the global
/// 20M-nnz model cap.
const IMB_AY_TAIL_AFFINE_MAX_NNZ: usize = 16_384;
/// The opt-in shared-input bank admits only the measured deterministic
/// K=16→K=8→K=4 ladder. No environment value may widen this closed set.
const IMB_AY_TAIL_SHARED_ALLOWED_SUPPORTS: [usize; 3] = [4, 8, 16];
const IMB_AY_TAIL_SHARED_MAX_SUPPORTS: usize = 16;
const IMB_AY_TAIL_SHARED_MAX_LATENT_INPUTS: usize = 16;
const IMB_AY_TAIL_SHARED_MAX_ROWS: usize = 32;
/// K=16 dense rows at the sealed 2,048-coordinate seam cost 65,696 added
/// nonzeros. This independent 2x headroom remains far below the whole-model
/// cap and rejects an accidentally widened bank before model construction.
const IMB_AY_TAIL_SHARED_MAX_NNZ: usize = 131_072;
/// Immutable dense bank payload only: directions, affine coefficients, biases,
/// and support identities. Root/region boxes and the seam-node string are
/// validated and request-bound separately; this byte cap does not claim to
/// charge their owned storage.
const IMB_AY_TAIL_SHARED_MAX_BANK_BYTES: usize = 256 * 1024;
const IMB_AY_TAIL_SOLVE_CAP_SECS: f64 = 45.0;
const IMB_AY_TAIL_DEADLINE_RESERVE_SECS: f64 = 0.05;
const IMB_AY_TAIL_MIN_SOLVE_SECS: f64 = 0.25;

fn imb_ay_tail_encoding_within_caps(cols: usize, rows: usize, binaries: usize) -> bool {
    cols <= IMB_AY_TAIL_MAX_COLS
        && rows <= IMB_AY_TAIL_MAX_ROWS
        && binaries <= IMB_AY_TAIL_MAX_BINARIES
}

fn imb_ay_tail_input_within_caps(inputs: usize) -> bool {
    inputs != 0 && inputs <= IMB_AY_TAIL_MAX_INPUTS
}

fn next_up_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Directed-up maximum of `p·y` over a finite binary32 seam box.
///
/// Every f32 product is exact in f64; widening each accumulated addition by one
/// f64 ULP therefore bounds the exact real dot product from above.
fn imb_affine_box_upper(p: &[f32], seam_box: &ny_tensor::BoundedTensor) -> Option<f64> {
    let flat = seam_box.flatten();
    let lower = flat.lower().as_slice()?;
    let upper = flat.upper().as_slice()?;
    if p.len() != lower.len()
        || lower.len() != upper.len()
        || p.iter()
            .zip(lower)
            .zip(upper)
            .any(|((&coefficient, &lo), &hi)| {
                !coefficient.is_finite() || !lo.is_finite() || !hi.is_finite() || lo > hi
            })
    {
        return None;
    }
    let mut total = 0.0_f64;
    for ((&coefficient, &lo), &hi) in p.iter().zip(lower).zip(upper) {
        if coefficient == 0.0 {
            continue;
        }
        let endpoint = if coefficient > 0.0 { hi } else { lo };
        total = next_up_f64(total + f64::from(coefficient) * f64::from(endpoint));
    }
    total.is_finite().then_some(total)
}

/// Serialize this opt-in exact lane. AY's configuration environment is
/// process-global, and parallel region callers must never multiply a
/// potentially 20M-nnz exact replay into an RSS spike.
static IMB_AY_TAIL_LOCK: Mutex<()> = Mutex::new(());

/// Attach the CLI's proof-carrying Graph-MIP implementation to ny-propagate.
///
/// The private authority bridge independently exact-gates registration on
/// `NY_IMB_TAIL_CERT_AY=1`; gate-off startup never mutates the oracle cell.
pub(crate) fn install_imb_ay_tail_certificate_oracle() {
    let _ = super::ay_tail_authority::install_verified_oracle_if_requested();
}

#[derive(Clone, Copy)]
enum ImbAyTailProofRequest<'a> {
    Residual {
        p: &'a [f32],
        requested_lower: Option<f32>,
    },
    Reachability {
        p: &'a [f32],
        prefix_lower: f32,
        requested_lower: f32,
    },
    AffineReachability {
        envelope: &'a AyTailAffineReachabilityEnvelope,
        requested_lower: f32,
    },
    SharedInputReachability {
        envelope: &'a AyTailSharedInputReachabilityEnvelope,
        requested_lower: f32,
    },
}

#[allow(clippy::too_many_arguments)]
pub(super) fn imb_ay_tail_certificate_exact_result(
    tail: &GraphNetwork,
    seam_box: &ny_tensor::BoundedTensor,
    _supplied_node_bounds: &HashMap<String, ny_tensor::BoundedTensor>,
    objective: &[f32],
    p: &[f32],
    requested_lower: Option<f32>,
    deadline: Instant,
) -> Option<CertifiedLinearLowerBound> {
    imb_ay_tail_certificate_exact_result_impl(
        tail,
        seam_box,
        objective,
        ImbAyTailProofRequest::Residual { p, requested_lower },
        deadline,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn imb_ay_tail_reachability_certificate_exact_result(
    tail: &GraphNetwork,
    seam_box: &ny_tensor::BoundedTensor,
    objective: &[f32],
    p: &[f32],
    prefix_lower: f32,
    requested_lower: f32,
    deadline: Instant,
) -> Option<CertifiedLinearLowerBound> {
    imb_ay_tail_certificate_exact_result_impl(
        tail,
        seam_box,
        objective,
        ImbAyTailProofRequest::Reachability {
            p,
            prefix_lower,
            requested_lower,
        },
        deadline,
    )
}

pub(super) fn imb_ay_tail_affine_reachability_certificate_exact_result(
    tail: &GraphNetwork,
    seam_box: &ny_tensor::BoundedTensor,
    objective: &[f32],
    envelope: &AyTailAffineReachabilityEnvelope,
    requested_lower: f32,
    deadline: Instant,
) -> Option<CertifiedLinearLowerBound> {
    imb_ay_tail_certificate_exact_result_impl(
        tail,
        seam_box,
        objective,
        ImbAyTailProofRequest::AffineReachability {
            envelope,
            requested_lower,
        },
        deadline,
    )
}

pub(super) fn imb_ay_tail_shared_input_reachability_certificate_exact_result(
    tail: &GraphNetwork,
    seam_box: &ny_tensor::BoundedTensor,
    objective: &[f32],
    envelope: &AyTailSharedInputReachabilityEnvelope,
    requested_lower: f32,
    deadline: Instant,
) -> Option<CertifiedLinearLowerBound> {
    imb_ay_tail_certificate_exact_result_impl(
        tail,
        seam_box,
        objective,
        ImbAyTailProofRequest::SharedInputReachability {
            envelope,
            requested_lower,
        },
        deadline,
    )
}

fn affine_reachability_added_nnz(envelope: &AyTailAffineReachabilityEnvelope) -> Option<usize> {
    if envelope.directions().nrows() != AY_TAIL_AFFINE_REACHABILITY_ROWS
        || envelope.lower_a().nrows() != AY_TAIL_AFFINE_REACHABILITY_ROWS
        || envelope.upper_a().nrows() != AY_TAIL_AFFINE_REACHABILITY_ROWS
    {
        return None;
    }
    let mut total = 0usize;
    for row in 0..AY_TAIL_AFFINE_REACHABILITY_ROWS {
        let direction_nnz = envelope
            .directions()
            .row(row)
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        let lower_input_nnz = envelope
            .lower_a()
            .row(row)
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        let upper_input_nnz = envelope
            .upper_a()
            .row(row)
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        total = total
            .checked_add(direction_nnz.checked_mul(2)?)?
            .checked_add(lower_input_nnz)?
            .checked_add(upper_input_nnz)?;
    }
    Some(total)
}

/// Add the exact f32→f64 relational rows
///
/// ```text
/// P_j y - A^-_j x >= b^-_j
/// P_j y - A^+_j x <= b^+_j
/// ```
///
/// for both support rows against one shared regional-prefix input block `x`.
/// Returns `(latent_cols, added_nnz)`.
fn add_affine_reachability_envelope(
    problem: &mut MilpProblem,
    seam_vars: &[Col],
    envelope: &AyTailAffineReachabilityEnvelope,
) -> Option<(Vec<Col>, usize)> {
    let flat_region = envelope.region_input().flatten();
    let region_lower = flat_region.lower().as_slice()?;
    let region_upper = flat_region.upper().as_slice()?;
    add_affine_reachability_rows(
        problem,
        seam_vars,
        region_lower,
        region_upper,
        envelope.directions(),
        envelope.lower_a(),
        envelope.lower_b(),
        envelope.upper_a(),
        envelope.upper_b(),
    )
}

#[allow(clippy::too_many_arguments)]
fn add_affine_reachability_rows(
    problem: &mut MilpProblem,
    seam_vars: &[Col],
    region_lower: &[f32],
    region_upper: &[f32],
    directions: &Array2<f32>,
    lower_a: &Array2<f32>,
    lower_b: &Array1<f32>,
    upper_a: &Array2<f32>,
    upper_b: &Array1<f32>,
) -> Option<(Vec<Col>, usize)> {
    let input_dim = region_lower.len();
    if region_upper.len() != input_dim
        || seam_vars.len() != directions.ncols()
        || directions.nrows() != AY_TAIL_AFFINE_REACHABILITY_ROWS
        || lower_a.shape() != [AY_TAIL_AFFINE_REACHABILITY_ROWS, input_dim]
        || upper_a.shape() != [AY_TAIL_AFFINE_REACHABILITY_ROWS, input_dim]
        || lower_b.len() != AY_TAIL_AFFINE_REACHABILITY_ROWS
        || upper_b.len() != AY_TAIL_AFFINE_REACHABILITY_ROWS
        || directions.iter().any(|value| !value.is_finite())
        || lower_a.iter().any(|value| !value.is_finite())
        || upper_a.iter().any(|value| !value.is_finite())
        || lower_b.iter().any(|value| !value.is_finite())
        || upper_b.iter().any(|value| !value.is_finite())
        || region_lower
            .iter()
            .zip(region_upper)
            .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
    {
        return None;
    }

    let latent_cols: Vec<Col> = region_lower
        .iter()
        .zip(region_upper)
        .map(|(&lower, &upper)| problem.add_col(0.0, f64::from(lower), f64::from(upper)))
        .collect();
    let mut added_nnz = 0usize;
    for row in 0..AY_TAIL_AFFINE_REACHABILITY_ROWS {
        let direction = directions.row(row);

        let mut lower_terms = Vec::new();
        for (&col, &coefficient) in seam_vars.iter().zip(direction.iter()) {
            if coefficient != 0.0 {
                lower_terms.push((col, f64::from(coefficient)));
            }
        }
        for (&col, &coefficient) in latent_cols.iter().zip(lower_a.row(row).iter()) {
            if coefficient != 0.0 {
                lower_terms.push((col, -f64::from(coefficient)));
            }
        }
        if lower_terms.is_empty() {
            return None;
        }
        added_nnz = added_nnz.checked_add(lower_terms.len())?;
        problem.add_row(f64::from(lower_b[row]), f64::INFINITY, lower_terms);

        let mut upper_terms = Vec::new();
        for (&col, &coefficient) in seam_vars.iter().zip(direction.iter()) {
            if coefficient != 0.0 {
                upper_terms.push((col, f64::from(coefficient)));
            }
        }
        for (&col, &coefficient) in latent_cols.iter().zip(upper_a.row(row).iter()) {
            if coefficient != 0.0 {
                upper_terms.push((col, -f64::from(coefficient)));
            }
        }
        if upper_terms.is_empty() {
            return None;
        }
        added_nnz = added_nnz.checked_add(upper_terms.len())?;
        problem.add_row(f64::NEG_INFINITY, f64::from(upper_b[row]), upper_terms);
    }
    Some((latent_cols, added_nnz))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SharedInputReachabilityStats {
    supports: usize,
    latent_inputs: usize,
    rows: usize,
    added_nnz: usize,
    bank_bytes: usize,
}

fn shared_input_reachability_added_nnz(
    directions: &Array2<f32>,
    lower_a: &Array2<f32>,
    upper_a: &Array2<f32>,
) -> Option<usize> {
    if directions.nrows() != lower_a.nrows() || directions.nrows() != upper_a.nrows() {
        return None;
    }
    let mut total = 0usize;
    for row in 0..directions.nrows() {
        let direction_nnz = directions
            .row(row)
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        if direction_nnz == 0 {
            return None;
        }
        let lower_input_nnz = lower_a
            .row(row)
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        let upper_input_nnz = upper_a
            .row(row)
            .iter()
            .filter(|&&value| value != 0.0)
            .count();
        total = total
            .checked_add(direction_nnz.checked_mul(2)?)?
            .checked_add(lower_input_nnz)?
            .checked_add(upper_input_nnz)?;
    }
    Some(total)
}

fn shared_input_reachability_stats(
    envelope: &AyTailSharedInputReachabilityEnvelope,
    seam_dim: usize,
) -> Option<SharedInputReachabilityStats> {
    let supports = envelope.directions().nrows();
    let root = envelope.certified_root_input();
    let region = envelope.region_input();
    let latent_inputs = root.flatten().len();
    let rows = supports.checked_mul(2)?;
    if !IMB_AY_TAIL_SHARED_ALLOWED_SUPPORTS.contains(&supports)
        || supports > IMB_AY_TAIL_SHARED_MAX_SUPPORTS
        || rows > IMB_AY_TAIL_SHARED_MAX_ROWS
        || latent_inputs == 0
        || latent_inputs > IMB_AY_TAIL_SHARED_MAX_LATENT_INPUTS
        || root.shape() != region.shape()
        || region.flatten().len() != latent_inputs
        || root.has_l2_constraint()
        || region.has_l2_constraint()
        || envelope.support_indices().len() != supports
        || envelope
            .support_indices()
            .iter()
            .enumerate()
            .any(|(index, value)| envelope.support_indices()[..index].contains(value))
        || envelope.directions().ncols() != seam_dim
        || envelope.lower_a().shape() != [supports, latent_inputs]
        || envelope.upper_a().shape() != [supports, latent_inputs]
        || envelope.lower_b().len() != supports
        || envelope.upper_b().len() != supports
        || envelope.directions().iter().any(|value| !value.is_finite())
        || envelope.lower_a().iter().any(|value| !value.is_finite())
        || envelope.upper_a().iter().any(|value| !value.is_finite())
        || envelope.lower_b().iter().any(|value| !value.is_finite())
        || envelope.upper_b().iter().any(|value| !value.is_finite())
    {
        return None;
    }

    // The regional latent box is the only prefix input block added to this
    // region's exact tail model. Every one of its points must lie inside the
    // exact root box over which the shared CROWN bank was certified.
    if !shared_input_region_is_inside(root, region) {
        return None;
    }

    let added_nnz = shared_input_reachability_added_nnz(
        envelope.directions(),
        envelope.lower_a(),
        envelope.upper_a(),
    )?;
    if added_nnz > IMB_AY_TAIL_SHARED_MAX_NNZ {
        return None;
    }

    let bank_f32_values = envelope
        .directions()
        .len()
        .checked_add(envelope.lower_a().len())?
        .checked_add(envelope.lower_b().len())?
        .checked_add(envelope.upper_a().len())?
        .checked_add(envelope.upper_b().len())?;
    let bank_bytes = bank_f32_values.checked_mul(size_of::<f32>())?.checked_add(
        envelope
            .support_indices()
            .len()
            .checked_mul(size_of::<usize>())?,
    )?;
    if bank_bytes != envelope.bank_bytes() || bank_bytes > IMB_AY_TAIL_SHARED_MAX_BANK_BYTES {
        return None;
    }
    Some(SharedInputReachabilityStats {
        supports,
        latent_inputs,
        rows,
        added_nnz,
        bank_bytes,
    })
}

fn shared_input_region_is_inside(
    root: &ny_tensor::BoundedTensor,
    region: &ny_tensor::BoundedTensor,
) -> bool {
    root.shape() == region.shape()
        && !root.has_l2_constraint()
        && !region.has_l2_constraint()
        && root
            .lower()
            .iter()
            .zip(root.upper())
            .zip(region.lower().iter().zip(region.upper()))
            .all(
                |((&root_lower, &root_upper), (&region_lower, &region_upper))| {
                    root_lower.is_finite()
                        && root_upper.is_finite()
                        && region_lower.is_finite()
                        && region_upper.is_finite()
                        && root_lower <= root_upper
                        && region_lower <= region_upper
                        && region_lower >= root_lower
                        && region_upper <= root_upper
                },
            )
}

/// Add a variable-width shared-input relational bank:
///
/// ```text
/// P_j y - A^-_j x >= b^-_j
/// P_j y - A^+_j x <= b^+_j.
/// ```
///
/// All row-local shapes, finite values, closed support counts, and added-nnz
/// budget are validated before the first column is added. Thus rejection by
/// this helper's local preflight cannot leave a partially mutated model; the
/// caller still performs whole-model caps after graph and bank encoding.
#[allow(clippy::too_many_arguments)]
fn add_shared_input_reachability_rows(
    problem: &mut MilpProblem,
    seam_vars: &[Col],
    region_lower: &[f32],
    region_upper: &[f32],
    directions: &Array2<f32>,
    lower_a: &Array2<f32>,
    lower_b: &Array1<f32>,
    upper_a: &Array2<f32>,
    upper_b: &Array1<f32>,
) -> Option<(Vec<Col>, usize)> {
    let supports = directions.nrows();
    let input_dim = region_lower.len();
    let rows = supports.checked_mul(2)?;
    let added_nnz = shared_input_reachability_added_nnz(directions, lower_a, upper_a)?;
    if !IMB_AY_TAIL_SHARED_ALLOWED_SUPPORTS.contains(&supports)
        || supports > IMB_AY_TAIL_SHARED_MAX_SUPPORTS
        || rows > IMB_AY_TAIL_SHARED_MAX_ROWS
        || input_dim == 0
        || input_dim > IMB_AY_TAIL_SHARED_MAX_LATENT_INPUTS
        || region_upper.len() != input_dim
        || seam_vars.len() != directions.ncols()
        || lower_a.shape() != [supports, input_dim]
        || upper_a.shape() != [supports, input_dim]
        || lower_b.len() != supports
        || upper_b.len() != supports
        || added_nnz > IMB_AY_TAIL_SHARED_MAX_NNZ
        || directions.iter().any(|value| !value.is_finite())
        || lower_a.iter().any(|value| !value.is_finite())
        || upper_a.iter().any(|value| !value.is_finite())
        || lower_b.iter().any(|value| !value.is_finite())
        || upper_b.iter().any(|value| !value.is_finite())
        || region_lower
            .iter()
            .zip(region_upper)
            .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
    {
        return None;
    }

    let latent_cols: Vec<Col> = region_lower
        .iter()
        .zip(region_upper)
        .map(|(&lower, &upper)| problem.add_col(0.0, f64::from(lower), f64::from(upper)))
        .collect();
    for row in 0..supports {
        let direction = directions.row(row);
        let lower_terms: Vec<_> = seam_vars
            .iter()
            .copied()
            .zip(direction.iter().copied())
            .filter_map(|(col, coefficient)| {
                (coefficient != 0.0).then_some((col, f64::from(coefficient)))
            })
            .chain(
                latent_cols
                    .iter()
                    .copied()
                    .zip(lower_a.row(row).iter().copied())
                    .filter_map(|(col, coefficient)| {
                        (coefficient != 0.0).then_some((col, -f64::from(coefficient)))
                    }),
            )
            .collect();
        problem.add_row(f64::from(lower_b[row]), f64::INFINITY, lower_terms);

        let upper_terms: Vec<_> = seam_vars
            .iter()
            .copied()
            .zip(direction.iter().copied())
            .filter_map(|(col, coefficient)| {
                (coefficient != 0.0).then_some((col, f64::from(coefficient)))
            })
            .chain(
                latent_cols
                    .iter()
                    .copied()
                    .zip(upper_a.row(row).iter().copied())
                    .filter_map(|(col, coefficient)| {
                        (coefficient != 0.0).then_some((col, -f64::from(coefficient)))
                    }),
            )
            .collect();
        problem.add_row(f64::NEG_INFINITY, f64::from(upper_b[row]), upper_terms);
    }
    Some((latent_cols, added_nnz))
}

fn add_shared_input_reachability_envelope(
    problem: &mut MilpProblem,
    seam_vars: &[Col],
    envelope: &AyTailSharedInputReachabilityEnvelope,
) -> Option<(Vec<Col>, SharedInputReachabilityStats)> {
    let stats = shared_input_reachability_stats(envelope, seam_vars.len())?;
    let flat_region = envelope.region_input().flatten();
    let region_lower = flat_region.lower().as_slice()?;
    let region_upper = flat_region.upper().as_slice()?;
    let (latent_cols, added_nnz) = add_shared_input_reachability_rows(
        problem,
        seam_vars,
        region_lower,
        region_upper,
        envelope.directions(),
        envelope.lower_a(),
        envelope.lower_b(),
        envelope.upper_a(),
        envelope.upper_b(),
    )?;
    if added_nnz != stats.added_nnz || latent_cols.len() != stats.latent_inputs {
        return None;
    }
    Some((latent_cols, stats))
}

#[cfg(test)]
mod shared_input_reachability_tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn box1(lower: &[f32], upper: &[f32]) -> ny_tensor::BoundedTensor {
        ny_tensor::BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn shared_rows_have_exact_signs_bounds_and_one_latent_block() {
        let mut problem = MilpProblem::new();
        let seam = vec![
            problem.add_col(0.0, -10.0, 10.0),
            problem.add_col(0.0, -20.0, 20.0),
        ];
        let directions =
            Array2::from_shape_vec((4, 2), vec![1.0, -2.0, 0.5, 0.0, 0.0, 3.0, -1.0, 0.0]).unwrap();
        let lower_a =
            Array2::from_shape_vec((4, 2), vec![4.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let upper_a =
            Array2::from_shape_vec((4, 2), vec![-9.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let lower_b = Array1::from_vec(vec![7.0, -8.0, 9.0, -10.0]);
        let upper_b = Array1::from_vec(vec![12.0, -13.0, 14.0, -15.0]);

        let (latent, added_nnz) = add_shared_input_reachability_rows(
            &mut problem,
            &seam,
            &[-1.0, -2.0],
            &[3.0, 4.0],
            &directions,
            &lower_a,
            &lower_b,
            &upper_a,
            &upper_b,
        )
        .expect("well-formed K=4 shared bank");

        assert_eq!(latent, vec![Col(2), Col(3)]);
        assert_eq!(problem.num_cols(), 4);
        assert_eq!(problem.num_rows(), 8);
        assert_eq!(added_nnz, 14);
        assert_eq!(problem.cols()[2].lb, -1.0);
        assert_eq!(problem.cols()[2].ub, 3.0);
        assert_eq!(problem.cols()[3].lb, -2.0);
        assert_eq!(problem.cols()[3].ub, 4.0);
        assert_eq!(problem.rows()[0].lb, 7.0);
        assert_eq!(problem.rows()[0].ub, f64::INFINITY);
        assert_eq!(
            problem.rows()[0].coeffs,
            vec![(0, 1.0), (1, -2.0), (2, -4.0), (3, 5.0)]
        );
        assert_eq!(problem.rows()[1].lb, f64::NEG_INFINITY);
        assert_eq!(problem.rows()[1].ub, 12.0);
        assert_eq!(
            problem.rows()[1].coeffs,
            vec![(0, 1.0), (1, -2.0), (2, 9.0), (3, -10.0)]
        );
        // Every later support row must reference the same two latent columns;
        // no per-support copy of x is permitted.
        assert_eq!(problem.rows()[2].coeffs, vec![(0, 0.5)]);
        assert_eq!(problem.rows()[4].coeffs, vec![(1, 3.0)]);
    }

    #[test]
    fn shared_rows_fail_closed_without_partial_mutation() {
        let mut problem = MilpProblem::new();
        let seam = vec![problem.add_col(0.0, -1.0, 1.0)];
        let malformed = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, f32::NAN, 4.0]).unwrap();
        let a = Array2::zeros((4, 1));
        let b = Array1::zeros(4);
        let before = (problem.num_cols(), problem.num_rows());
        assert!(add_shared_input_reachability_rows(
            &mut problem,
            &seam,
            &[-1.0],
            &[1.0],
            &malformed,
            &a,
            &b,
            &a,
            &b,
        )
        .is_none());
        assert_eq!((problem.num_cols(), problem.num_rows()), before);

        let unsupported = Array2::ones((3, 1));
        let unsupported_a = Array2::zeros((3, 1));
        let unsupported_b = Array1::zeros(3);
        assert!(add_shared_input_reachability_rows(
            &mut problem,
            &seam,
            &[-1.0],
            &[1.0],
            &unsupported,
            &unsupported_a,
            &unsupported_b,
            &unsupported_a,
            &unsupported_b,
        )
        .is_none());
        assert_eq!((problem.num_cols(), problem.num_rows()), before);
    }

    #[test]
    fn shared_caps_cover_measured_k16_and_reject_an_nnz_overrun() {
        assert_eq!(IMB_AY_TAIL_SHARED_ALLOWED_SUPPORTS, [4, 8, 16]);
        assert_eq!(IMB_AY_TAIL_SHARED_MAX_SUPPORTS, 16);
        assert_eq!(IMB_AY_TAIL_SHARED_MAX_ROWS, 32);
        assert_eq!(IMB_AY_TAIL_SHARED_MAX_LATENT_INPUTS, 16);
        assert_eq!(IMB_AY_TAIL_SHARED_MAX_NNZ, 131_072);
        assert_eq!(IMB_AY_TAIL_SHARED_MAX_BANK_BYTES, 256 * 1024);
        let measured_dense_nnz = 2 * 16 * 2_048 + 2 * 16 * 5;
        assert_eq!(measured_dense_nnz, 65_696);
        assert!(measured_dense_nnz <= IMB_AY_TAIL_SHARED_MAX_NNZ);
        let measured_bank_bytes =
            (16 * 2_048 + 2 * 16 * 5 + 2 * 16) * size_of::<f32>() + 16 * size_of::<usize>();
        assert!(measured_bank_bytes <= IMB_AY_TAIL_SHARED_MAX_BANK_BYTES);

        let mut problem = MilpProblem::new();
        let seam: Vec<_> = (0..4_097)
            .map(|_| problem.add_col(0.0, -1.0, 1.0))
            .collect();
        let directions = Array2::ones((16, 4_097));
        let a = Array2::zeros((16, 1));
        let b = Array1::zeros(16);
        let before = (problem.num_cols(), problem.num_rows());
        assert!(add_shared_input_reachability_rows(
            &mut problem,
            &seam,
            &[-1.0],
            &[1.0],
            &directions,
            &a,
            &b,
            &a,
            &b,
        )
        .is_none());
        assert_eq!(
            (problem.num_cols(), problem.num_rows()),
            before,
            "nnz-cap rejection must precede latent/row mutation"
        );
    }

    #[test]
    fn shared_region_must_be_a_finite_exact_shape_subset_of_root() {
        let root = box1(&[-1.0, -2.0], &[3.0, 4.0]);
        let inside = box1(&[-0.5, -2.0], &[2.0, 1.0]);
        let outside_lower = box1(&[-1.5, -2.0], &[2.0, 1.0]);
        let outside_upper = box1(&[-0.5, -2.0], &[3.5, 1.0]);
        let wrong_shape = ny_tensor::BoundedTensor::new(
            ArrayD::zeros(IxDyn(&[1, 2])),
            ArrayD::ones(IxDyn(&[1, 2])),
        )
        .unwrap();
        assert!(shared_input_region_is_inside(&root, &inside));
        assert!(!shared_input_region_is_inside(&root, &outside_lower));
        assert!(!shared_input_region_is_inside(&root, &outside_upper));
        assert!(!shared_input_region_is_inside(&root, &wrong_shape));
    }
}

fn imb_ay_tail_certificate_exact_result_impl(
    tail: &GraphNetwork,
    seam_box: &ny_tensor::BoundedTensor,
    objective: &[f32],
    request: ImbAyTailProofRequest<'_>,
    deadline: Instant,
) -> Option<CertifiedLinearLowerBound> {
    let (
        p,
        affine_envelope,
        shared_envelope,
        requested_lower,
        prefix_premise,
        proof_mode,
        residual_mode,
    ) = match request {
        ImbAyTailProofRequest::Residual { p, requested_lower } => {
            (Some(p), None, None, requested_lower, None, "residual", true)
        }
        ImbAyTailProofRequest::Reachability {
            p,
            prefix_lower,
            requested_lower,
        } => (
            Some(p),
            None,
            None,
            Some(requested_lower),
            Some(prefix_lower),
            "reachability",
            false,
        ),
        ImbAyTailProofRequest::AffineReachability {
            envelope,
            requested_lower,
        } => (
            None,
            Some(envelope),
            None,
            Some(requested_lower),
            None,
            "affine-reachability",
            false,
        ),
        ImbAyTailProofRequest::SharedInputReachability {
            envelope,
            requested_lower,
        } => (
            None,
            None,
            Some(envelope),
            Some(requested_lower),
            None,
            "shared-input-reachability",
            false,
        ),
    };
    if Instant::now() >= deadline {
        return None;
    }
    let _serial = match IMB_AY_TAIL_LOCK.try_lock() {
        Ok(guard) => guard,
        Err(std::sync::TryLockError::WouldBlock) => {
            eprintln!("[imb] AY-TAIL-CERT declined: another tail encoder is active");
            return None;
        }
        Err(std::sync::TryLockError::Poisoned(_)) => {
            eprintln!("[imb] AY-TAIL-CERT declined: tail encoder admission is poisoned");
            return None;
        }
    };
    if Instant::now() >= deadline
        || tail.num_nodes() == 0
        || tail.num_nodes() > IMB_AY_TAIL_MAX_NODES
        || objective.is_empty()
        || objective.len() > IMB_AY_TAIL_MAX_OUTPUTS
        || p.is_some_and(|values| {
            values.is_empty()
                || values.len() > IMB_AY_TAIL_MAX_INPUTS
                || values.iter().any(|value| !value.is_finite())
        })
        || objective.iter().any(|value| !value.is_finite())
        || requested_lower.is_some_and(|value| !value.is_finite())
        || prefix_premise.is_some_and(|value| !value.is_finite())
    {
        return None;
    }
    if let Some(envelope) = affine_envelope {
        let region_dim = envelope.region_input().flatten().len();
        let added_nnz = affine_reachability_added_nnz(envelope)?;
        if region_dim == 0
            || region_dim > IMB_AY_TAIL_AFFINE_MAX_LATENT_INPUTS
            || added_nnz > IMB_AY_TAIL_AFFINE_MAX_NNZ
            || envelope.directions().nrows() != AY_TAIL_AFFINE_REACHABILITY_ROWS
            || envelope.directions().ncols() != seam_box.flatten().len()
            || envelope.lower_a().shape() != [AY_TAIL_AFFINE_REACHABILITY_ROWS, region_dim]
            || envelope.upper_a().shape() != [AY_TAIL_AFFINE_REACHABILITY_ROWS, region_dim]
        {
            eprintln!(
                "[imb] AY-TAIL-CERT affine reachability declined before admission: \
                 seam={} latent={} rows={} nnz={added_nnz}",
                envelope.directions().ncols(),
                region_dim,
                envelope.directions().nrows(),
            );
            return None;
        }
    }
    let shared_stats = if let Some(envelope) = shared_envelope {
        let Some(stats) = shared_input_reachability_stats(envelope, seam_box.flatten().len())
        else {
            eprintln!(
                "[imb] AY-TAIL-CERT shared-input reachability declined before admission: \
                 malformed, over-cap, or region outside certified root"
            );
            return None;
        };
        Some(stats)
    } else {
        None
    };
    // Reserve the process-wide exact worker *before* tail-local bounds and the
    // potentially 20M-nnz Graph-MIP are materialized. A worker detached at a
    // prior hard deadline retains this admission until its actual exit, so a
    // retry sheds both solver threads and encoder/RSS work immediately.
    let worker_admission = match CertifiedLinearLowerWorkerAdmission::try_acquire() {
        Some(admission) => admission,
        None => {
            eprintln!(
                "[imb] AY-TAIL-CERT declined before encode: a prior exact AY worker is active"
            );
            return None;
        }
    };

    let input_bounds = super::mip_preprocess::bounded_tensor_to_bounds(seam_box).ok()?;
    if !imb_ay_tail_input_within_caps(input_bounds.len()) {
        eprintln!(
            "[imb] AY-TAIL-CERT declined before node bounds: seam input {} outside 1..={}",
            input_bounds.len(),
            IMB_AY_TAIL_MAX_INPUTS,
        );
        return None;
    }
    if p.is_some_and(|values| input_bounds.len() != values.len()) {
        eprintln!(
            "[imb] AY-TAIL-CERT declined: seam input {} != p dimension {}",
            input_bounds.len(),
            p.map_or(0, <[f32]>::len)
        );
        return None;
    }
    if let Some(prefix_lower) = prefix_premise {
        let p = p?;
        let Some(box_upper) = imb_affine_box_upper(p, seam_box) else {
            eprintln!(
                "[imb] AY-TAIL-CERT reachability premise declined: malformed affine seam box"
            );
            return None;
        };
        if f64::from(prefix_lower) > box_upper {
            eprintln!(
                "[imb] AY-TAIL-CERT reachability premise declined: prefix_lower \
                 {prefix_lower:.9} exceeds seam-box affine maximum {box_upper:.9}"
            );
            return None;
        }
    }

    // Build every downstream bound solely from the seam box. The residual mode
    // remains universal over every point in that box. Reachability mode adds
    // only its explicitly certified affine prefix premise below; a
    // caller-supplied full-graph node box is never smuggled into either model.
    let node_bounds = tail
        .collect_node_bounds_with_engine_and_deadline(seam_box, None, Some(deadline))
        .ok()?;
    let flat_bounds = ibp_boxes_for_encoder(tail, &node_bounds);
    let estimated_nnz = super::graph_mip_leaf::estimate_encode_nnz(tail, &flat_bounds)?;
    let p_nnz = p.map_or(0, |values| {
        values.iter().filter(|&&value| value != 0.0).count()
    });
    let objective_nnz = objective.iter().filter(|&&value| value != 0.0).count();
    let premise_nnz = prefix_premise.map_or(0, |_| p_nnz);
    let affine_nnz = affine_envelope.map_or(Some(0), affine_reachability_added_nnz)?;
    let shared_nnz = shared_stats.map_or(0, |stats| stats.added_nnz);
    let decision_nnz = objective_nnz.checked_add(if residual_mode { p_nnz } else { 0 })?;
    let projected_nnz = estimated_nnz
        .checked_add(premise_nnz)?
        .checked_add(affine_nnz)?
        .checked_add(shared_nnz)?
        .checked_add(decision_nnz)?;
    if projected_nnz > IMB_AY_TAIL_MAX_NNZ {
        eprintln!(
            "[imb] AY-TAIL-CERT declined before encode: estimated nnz \
             {projected_nnz} > {IMB_AY_TAIL_MAX_NNZ}"
        );
        return None;
    }

    let mut encoding = encode_graph_with_delta_and_deadline(
        tail,
        &input_bounds,
        &flat_bounds,
        DELTA,
        Some(deadline),
    )
    .ok()?;
    if let Some(prefix_lower) = prefix_premise {
        let p = p?;
        let premise_terms: Vec<_> = encoding
            .input_vars
            .iter()
            .copied()
            .zip(p.iter().copied())
            .filter_map(|(col, coefficient)| {
                (coefficient != 0.0).then_some((col, f64::from(coefficient)))
            })
            .collect();
        if premise_terms.is_empty() {
            eprintln!("[imb] AY-TAIL-CERT reachability premise declined: p is identically zero");
            return None;
        }
        // Exact f32→f64 transport: the independently certified regional prefix
        // fact is `p·h(x) >= prefix_lower`. Every reachable seam point in that
        // region therefore remains feasible after this one-sided row is added.
        encoding
            .problem
            .add_row(f64::from(prefix_lower), f64::INFINITY, premise_terms);
    }
    let affine_latent_cols = if let Some(envelope) = affine_envelope {
        let (latent_cols, added_nnz) = add_affine_reachability_envelope(
            &mut encoding.problem,
            &encoding.input_vars,
            envelope,
        )?;
        if latent_cols.len() > IMB_AY_TAIL_AFFINE_MAX_LATENT_INPUTS
            || added_nnz > IMB_AY_TAIL_AFFINE_MAX_NNZ
        {
            return None;
        }
        eprintln!(
            "[imb] AY-TAIL-CERT affine envelope encoded: supports={} latent={} \
             rows={} nnz={added_nnz}",
            AY_TAIL_AFFINE_REACHABILITY_ROWS,
            latent_cols.len(),
            IMB_AY_TAIL_AFFINE_ROWS,
        );
        Some(latent_cols)
    } else {
        None
    };
    let shared_latent_cols = if let Some(envelope) = shared_envelope {
        let (latent_cols, stats) = add_shared_input_reachability_envelope(
            &mut encoding.problem,
            &encoding.input_vars,
            envelope,
        )?;
        if stats != shared_stats?
            || latent_cols.len() > IMB_AY_TAIL_SHARED_MAX_LATENT_INPUTS
            || stats.rows > IMB_AY_TAIL_SHARED_MAX_ROWS
            || stats.added_nnz > IMB_AY_TAIL_SHARED_MAX_NNZ
            || stats.bank_bytes > IMB_AY_TAIL_SHARED_MAX_BANK_BYTES
        {
            return None;
        }
        eprintln!(
            "[imb] AY-TAIL-CERT shared-input envelope encoded: supports={} latent={} \
             rows={} nnz={} bank_bytes={}",
            stats.supports,
            latent_cols.len(),
            stats.rows,
            stats.added_nnz,
            stats.bank_bytes,
        );
        Some(latent_cols)
    } else {
        None
    };
    let projected_rows = encoding.problem.num_rows().checked_add(1)?;
    if p.is_some_and(|values| encoding.input_vars.len() != values.len())
        || encoding.output_vars.len() != objective.len()
        || affine_latent_cols
            .as_ref()
            .is_some_and(|cols| cols.len() > IMB_AY_TAIL_AFFINE_MAX_LATENT_INPUTS)
        || shared_latent_cols
            .as_ref()
            .is_some_and(|cols| cols.len() > IMB_AY_TAIL_SHARED_MAX_LATENT_INPUTS)
        || !imb_ay_tail_encoding_within_caps(
            encoding.problem.num_cols(),
            projected_rows,
            encoding.binary_vars.len(),
        )
    {
        eprintln!(
            "[imb] AY-TAIL-CERT declined after encode: cols={} encoded_rows={} \
             projected_rows={} binaries={} input={} output={}",
            encoding.problem.num_cols(),
            encoding.problem.num_rows(),
            projected_rows,
            encoding.binary_vars.len(),
            encoding.input_vars.len(),
            encoding.output_vars.len()
        );
        return None;
    }
    let actual_nnz = encoding
        .problem
        .rows()
        .iter()
        .try_fold(0usize, |total, row| total.checked_add(row.coeffs.len()))?;
    if actual_nnz > IMB_AY_TAIL_MAX_NNZ {
        return None;
    }

    // Residual mode proves objective·tail(y) - p·y. Reachability mode instead
    // proves the original objective directly under the certified `p·y >= L`
    // row above, avoiding the lossy sum of two independently minimized terms.
    // Both routes verify AY's exact relaxation entailment or root/tree proof
    // and independently reconstruct every linear obligation through ny-cert.
    let mut terms = Vec::with_capacity(objective.len() + p.map_or(0, <[f32]>::len));
    for (&col, &coefficient) in encoding.output_vars.iter().zip(objective) {
        if coefficient != 0.0 {
            terms.push((col, f64::from(coefficient)));
        }
    }
    if residual_mode {
        let p = p?;
        for (&col, &coefficient) in encoding.input_vars.iter().zip(p) {
            if coefficient != 0.0 {
                terms.push((col, -f64::from(coefficient)));
            }
        }
    }
    if terms.is_empty() {
        return None;
    }
    let proof_nnz = actual_nnz.checked_add(terms.len())?;
    if proof_nnz > IMB_AY_TAIL_MAX_NNZ {
        eprintln!(
            "[imb] AY-TAIL-CERT declined before proof: encoded+decision nnz \
             {proof_nnz} > {IMB_AY_TAIL_MAX_NNZ}"
        );
        return None;
    }

    let remaining = deadline
        .checked_duration_since(Instant::now())?
        .as_secs_f64()
        - IMB_AY_TAIL_DEADLINE_RESERVE_SECS;
    let total = remaining.min(IMB_AY_TAIL_SOLVE_CAP_SECS);
    if !total.is_finite() || total < IMB_AY_TAIL_MIN_SOLVE_SECS {
        return None;
    }
    let proof_t0 = Instant::now();
    let proof_result = if let Some(lower) = requested_lower {
        certify_linear_lower_bound_at_with_ay_admission(
            &encoding.problem,
            &terms,
            lower,
            CertifiedLinearLowerDecisionConfig {
                proof_timeout_secs: total,
                max_tree_leaves: CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES,
            },
            worker_admission,
        )
    } else {
        if total < 2.0 * IMB_AY_TAIL_MIN_SOLVE_SECS {
            return None;
        }
        let proposal_timeout_secs = (0.35 * total).max(IMB_AY_TAIL_MIN_SOLVE_SECS);
        let proof_timeout_secs = total - proposal_timeout_secs;
        certify_linear_lower_bound_with_ay_admission(
            &encoding.problem,
            &terms,
            CertifiedLinearLowerBoundConfig {
                proposal_timeout_secs,
                proof_timeout_secs,
                max_tree_leaves: CERTIFIED_LINEAR_LOWER_HARD_MAX_TREE_LEAVES,
            },
            worker_admission,
        )
    };
    let certified = match proof_result {
        Ok(Some(certificate)) => certificate,
        Ok(None) => {
            eprintln!(
                "[imb] AY-TAIL-CERT proof inconclusive: mode={proof_mode} \
                 decision_only={} elapsed={:.3}s",
                requested_lower.is_some(),
                proof_t0.elapsed().as_secs_f64(),
            );
            return None;
        }
        Err(error) => {
            eprintln!(
                "[imb] AY-TAIL-CERT proof error: mode={proof_mode} decision_only={} \
                 elapsed={:.3}s error={error}",
                requested_lower.is_some(),
                proof_t0.elapsed().as_secs_f64(),
            );
            return None;
        }
    };
    if Instant::now() >= deadline {
        return None;
    }
    eprintln!(
        "[imb] AY-TAIL-CERT proof accepted: mode={proof_mode} cols={} rows={} nnz={} \
         binaries={} decision_only={} lower={:.9} elapsed={:.3}s tree_leaves={} \
         ny_cert_replays={} proof_route={:?}",
        encoding.problem.num_cols(),
        encoding.problem.num_rows(),
        proof_nnz,
        encoding.binary_vars.len(),
        requested_lower.is_some(),
        certified.lower,
        proof_t0.elapsed().as_secs_f64(),
        certified.ay_tree_leaves,
        certified.ny_cert_farkas_replays,
        certified.proof_route
    );
    Some(certified)
}

/// inc5d — minimum per-clause budget slice (seconds) to attempt CROWN-IBP
/// tightening of the per-node boxes. CROWN backward passes cost more than the
/// plain IBP sweep; on 1,000+-clause instances the slices are milliseconds and
/// must stay on the cheap IBP path (those clauses are out of solver reach
/// anyway — spending their slice on bounds would only starve the solver).
const GRAPH_MIP_CROWN_MIN_SLICE_SECS: f64 = 2.0;

/// inc5d — fraction of the clause's budget slice the CROWN-IBP collection may
/// consume. The remainder is reserved for ay: tighter big-M is worthless if
/// the bound collection eats the solve budget (the `solver_secs <= 0.05`
/// fail-closed guard would then void the clause).
const GRAPH_MIP_CROWN_SLICE_FRACTION: f64 = 0.25;

/// inc5d — per-node floor override for the escalation's CROWN-IBP collection.
///
/// The collector's default policy skips a node to IBP when its share of the
/// remaining deadline falls below a 2 s floor (`MIN_PER_NODE_BUDGET_SECS`,
/// sized for deep conv backwards). The mscn-class graphs this escalation
/// targets have ~12 tiny tightening candidates that each finish in ~10 ms, so
/// under a 2.5 s collection deadline the 2 s floor skipped 10/12 of them
/// (measured: `PerNodeDeadlineExceeded x10`, leaving 42 loose big-Ms). The
/// override is a pure time-vs-tightness policy on OUR freshly reloaded graph
/// object — any value is sound (`set_crown_ibp_per_node_time_budget` doc): a
/// too-small share just degrades that node to its IBP bound.
const GRAPH_MIP_CROWN_PER_NODE_FLOOR_SECS: f64 = 0.05;

/// inc5e — minimum per-clause budget slice (seconds) to attempt the α-CROWN
/// output-bound tightening pass. Mirrors the inc5d CROWN gate: on
/// 1,000+-clause instances the slices are milliseconds and must not spend
/// anything on α optimization (those clauses are out of solver reach anyway).
const GRAPH_MIP_ALPHA_MIN_SLICE_SECS: f64 = 5.0;

/// inc5e — fraction of the clause's budget slice the α-CROWN output-bound
/// pass may consume. Like `GRAPH_MIP_CROWN_SLICE_FRACTION`, the remainder is
/// reserved for ay (a tight output row is worthless if computing it eats the
/// solve budget).
const GRAPH_MIP_ALPHA_SLICE_FRACTION: f64 = 0.30;

/// inc5e — hard cap (seconds) on the α-CROWN output-bound pass. The measured
/// mscn α convergence is sub-second; anything longer is a stall the deadline
/// should cut so ay keeps its slice.
const GRAPH_MIP_ALPHA_SLICE_CAP_SECS: f64 = 5.0;

/// The result of encoding a `GraphNetwork` into a `MilpProblem`.
///
/// Mirrors the shape of `ny_mip::MipParts` (problem + input/output/binary column
/// handles) but is produced by the DAG walk rather than the chain encoder.
#[derive(Clone, Debug)]
pub(crate) struct GraphMipEncoding {
    /// The solver-neutral MILP formulation.
    pub problem: MilpProblem,
    /// Columns for the network input tensor (the `_input` sentinel), in order.
    pub input_vars: Vec<Col>,
    /// Columns for the graph's output node's tensor, in order.
    pub output_vars: Vec<Col>,
    /// Binary (ReLU indicator) columns, in creation order.
    pub binary_vars: Vec<Col>,
    /// Pre-activation width `u - l` per unstable ReLU neuron, aligned with
    /// `binary_vars` (drives phase-split branch selection downstream).
    pub binary_widths: Vec<f64>,
    /// `(relu_node_name, flat neuron index)` per binary, aligned with
    /// `binary_vars` — the identity the LEAF pin step (increment 6,
    /// `docs/GRAPH_MIP_LEAF_SOLVER.md`) uses to fix a split premise's
    /// indicator column. ReLU is the ONLY binary-emitting op (the pinned-mask
    /// MulBinary / Div lanes are exact affine rows), so this stays complete
    /// across the mscn op coverage.
    pub binary_keys: Vec<(String, usize)>,
    /// Node name -> that node's output columns (includes the `_input`
    /// sentinel). Introspection for tests (the point-eval parity oracle walks
    /// it) and for future call sites; a peeled final Sigmoid has no entry.
    pub node_cols: HashMap<String, Vec<Col>>,
}

/// Whether Graph-MIP should mark its decision row for AY's experimental
/// objective reframe.
///
/// Default-OFF is deliberate.  A sealed real isomorphic-ACAS A/B on
/// 2026-07-21 found that the reframe turned a verified case-split result into
/// bare UNSAT, which NY must reject, while taking about 2.5x as long.  Exact
/// `NY_AY_MARGIN_REFRAME=1` is the research force-on; every other spelling is
/// plain feasibility.  AY's independent `AY_MILP_NO_MARGIN_REFRAME` emergency
/// kill switch still wins if both are set.  NY reads but never mutates either
/// process variable.
fn ay_margin_reframe_enabled_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

fn ay_margin_reframe_enabled() -> bool {
    ay_margin_reframe_enabled_from_value(std::env::var("NY_AY_MARGIN_REFRAME").ok().as_deref())
}

impl GraphMipEncoding {
    /// Convert into [`ny_mip::MipParts`] — the same shape `MipEncoder::into_parts`
    /// produces — so `MipSolver`'s phase-split racing / warm-start / certificate
    /// machinery (`check_feasibility_with_warm_start`) is REUSED, not rebuilt.
    /// Field-for-field move; `num_cols` is read off the problem for warm-start
    /// vector sizing.
    pub(crate) fn into_parts(self) -> MipParts {
        let num_cols = self.problem.num_cols();
        MipParts {
            problem: self.problem,
            input_vars: self.input_vars,
            output_vars: self.output_vars,
            binary_vars: self.binary_vars,
            binary_widths: self.binary_widths,
            num_cols,
        }
    }

    /// Assert one VNN-LIB output constraint over this encoding's output columns.
    ///
    /// Emits EXACTLY the rows `MipEncoder::constrain_output_{leq,geq,leq_const,
    /// geq_const}` emit (same bounds, same coefficient order), so a Linear+ReLU
    /// chain encoded either way stays byte-identical after constraint stamping
    /// (the increment-1 invariant, extended to the spec surface). Strict and
    /// non-strict comparisons encode identically (LP relaxations have no strict
    /// inequality; the strictness is re-checked at witness revalidation).
    /// Assert the DECISION-form violation row for one ny-internal objective:
    /// `Σ_i coeffs[i] · y_i <= threshold` over this encoding's output columns
    /// (zero coefficients skipped — sparse row). The MIP is then infeasible
    /// iff `objective · y > threshold` on the whole encoded set, i.e. iff the
    /// row is VERIFIED there — emit_hard_six.py's `*_dec` form (margin `<= 0`
    /// modulo the constant fold). With exact `NY_AY_MARGIN_REFRAME=1`, the
    /// append also carries this row's typed identity through
    /// `MilpProblem`/`MipParts` to activate AY's experimental reframe. The safe
    /// default and all ordinary output rows remain unmarked. Used by the LEAF
    /// oracle (increment 6).
    pub(crate) fn add_violation_row(&mut self, coeffs: &[f32], threshold: f64) -> Result<()> {
        self.add_violation_row_with_margin_reframe(coeffs, threshold, ay_margin_reframe_enabled())
    }

    /// Append the exact same decision row under an explicit marker policy.
    /// The argument exists so unset/disabled/forced behavior can be tested
    /// without mutating process-global environment state.
    fn add_violation_row_with_margin_reframe(
        &mut self,
        coeffs: &[f32],
        threshold: f64,
        mark_for_ay: bool,
    ) -> Result<()> {
        if coeffs.len() != self.output_vars.len() {
            bail!(
                "violation row length {} != {} output columns",
                coeffs.len(),
                self.output_vars.len()
            );
        }
        let sparse: Vec<(Col, f64)> = coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0.0)
            .map(|(i, c)| (self.output_vars[i], *c as f64))
            .collect();
        if sparse.is_empty() {
            bail!("violation row has no nonzero coefficients");
        }
        if mark_for_ay {
            self.problem
                .add_margin_row(f64::NEG_INFINITY, threshold, sparse)
                .map_err(|e| {
                    anyhow!(
                        "cannot add the graph violation row as the unique AY decision margin: {e}"
                    )
                })?;
        } else {
            self.problem.add_row(f64::NEG_INFINITY, threshold, sparse);
        }
        Ok(())
    }

    pub(crate) fn add_output_constraint(&mut self, constraint: &OutputConstraint) -> Result<()> {
        let var = |i: usize| -> Result<Col> {
            self.output_vars.get(i).copied().ok_or_else(|| {
                anyhow!(
                    "output index {i} out of range ({} output columns)",
                    self.output_vars.len()
                )
            })
        };
        match constraint {
            OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                let (yi, yj) = (var(*i)?, var(*j)?);
                self.problem
                    .add_row(f64::NEG_INFINITY, 0.0, [(yi, 1.0), (yj, -1.0)]);
            }
            OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                let (yi, yj) = (var(*i)?, var(*j)?);
                self.problem
                    .add_row(0.0, f64::INFINITY, [(yi, 1.0), (yj, -1.0)]);
            }
            OutputConstraint::LessEqConst(i, c) | OutputConstraint::LessThanConst(i, c) => {
                let yi = var(*i)?;
                self.problem.add_row(f64::NEG_INFINITY, *c, [(yi, 1.0)]);
            }
            OutputConstraint::GreaterEqConst(i, c) | OutputConstraint::GreaterThanConst(i, c) => {
                let yi = var(*i)?;
                self.problem.add_row(*c, f64::INFINITY, [(yi, 1.0)]);
            }
            other => bail!("unsupported OutputConstraint variant for graph MIP: {other:?}"),
        }
        Ok(())
    }
}

/// Encode a `GraphNetwork` (Linear / Conv2d / ReLU / Flatten / Reshape /
/// BatchNorm / residual Add — the full cifar100 resnet topology) into a
/// `MilpProblem`, walking the graph in topological order.
///
/// Production entry point: applies the [`DELTA`] box inflation (the cifar100
/// soundness mechanism). Use [`encode_graph_with_delta`] with `delta == 0.0`
/// only to reproduce the byte-identical-to-`encode_feedforward` invariant.
pub(crate) fn encode_graph(
    graph: &GraphNetwork,
    input_bounds: &[Bound],
    node_bounds: &HashMap<String, Vec<Bound>>,
) -> Result<GraphMipEncoding> {
    encode_graph_with_delta_and_deadline(graph, input_bounds, node_bounds, DELTA, None)
}

/// [`encode_graph`] with a wall-clock deadline: the DAG walk checks it at every
/// node and BAILS CLEANLY when it expires mid-encode (the measured whole-net
/// failure: a ~44M-nnz encode + exact-rational conversion ate the entire MIP
/// slice + the watchdog grace with no interior check). The per-node granularity
/// bounds overrun to one node's unfold. An `Err` here degrades soundly (the
/// caller keeps its inconclusive BaB verdict) — never a wrong verdict.
pub(crate) fn encode_graph_with_deadline(
    graph: &GraphNetwork,
    input_bounds: &[Bound],
    node_bounds: &HashMap<String, Vec<Bound>>,
    deadline: Option<Instant>,
) -> Result<GraphMipEncoding> {
    encode_graph_with_delta_and_deadline(graph, input_bounds, node_bounds, DELTA, deadline)
}

/// [`encode_graph`] with an explicit box-inflation `delta`.
///
/// # Arguments
/// * `graph` — the DAG to encode. Nodes must be Linear / Conv2d / ReLU /
///   Flatten / Reshape / BatchNorm / Add; `MaxPool` and everything else are
///   later increments and are rejected here.
/// * `input_bounds` — bounds on the network input elements (the `_input`
///   sentinel's columns), in flattened order. Used EXACTLY (never inflated),
///   matching emit_hard_six.py's exact input box.
/// * `node_bounds` — per-node intermediate boxes keyed by node name, the sound
///   BaB/CROWN boxes on each node's OUTPUT tensor (flattened). Two consumers:
///   (a) an AFFINE node (Linear / Conv2d / BatchNorm / Add) present in the map
///   has its output columns bounded by its inflated box `[lo-delta, hi+delta]`
///   (absent ⇒ free `±inf` columns — sound, just looser); (b) a ReLU reads its
///   pre-activation box (its own name first, else its input node's box) and
///   inflates it for the big-M `[pl, pu]`.
/// * `delta` — box-inflation half-width. `DELTA` (1e-4) in production; `0.0`
///   reproduces the byte-identical `encode_feedforward` invariant.
///
/// On a Linear+ReLU chain with `delta == 0.0` and `node_bounds` keyed only by
/// the ReLU nodes (pre-activation), the column and row insertion order is
/// identical to `ny_mip::encode_feedforward`, so the two produce byte-identical
/// `MilpProblem`s (the increment-1 invariant). Conv2d, BatchNorm, and residual
/// Add extend the walk with exact affine equality rows (increments 2-4): a
/// residual `Add` looks up BOTH parents' column vectors from the map, which is
/// why the encoder must be DAG-aware (the two operands are distinct upstream
/// blocks); a Conv2d is unfolded (im2col) into an equivalent dense Linear and
/// encoded through the SAME `encode_linear_node` rows.
pub(crate) fn encode_graph_with_delta(
    graph: &GraphNetwork,
    input_bounds: &[Bound],
    node_bounds: &HashMap<String, Vec<Bound>>,
    delta: f64,
) -> Result<GraphMipEncoding> {
    encode_graph_with_delta_and_deadline(graph, input_bounds, node_bounds, delta, None)
}

/// [`encode_graph_with_delta`] + the per-node deadline check (see
/// [`encode_graph_with_deadline`]). The final-Sigmoid peel stays disabled on
/// this lane.
fn encode_graph_with_delta_and_deadline(
    graph: &GraphNetwork,
    input_bounds: &[Bound],
    node_bounds: &HashMap<String, Vec<Bound>>,
    delta: f64,
    deadline: Option<Instant>,
) -> Result<GraphMipEncoding> {
    let (encoding, peeled) =
        encode_graph_impl(graph, input_bounds, node_bounds, delta, false, deadline)?;
    debug_assert!(!peeled, "peel disabled");
    Ok(encoding)
}

/// [`encode_graph`] that additionally PEELS a final Sigmoid output node
/// (increment 5 — the nn4sys mscn output rewrite).
///
/// If the graph's output node is a single-input `Sigmoid`, the returned
/// encoding stops at the Sigmoid's INPUT (the logit): `output_vars` are the
/// logit's columns and the second return value is `true`. The caller must then
/// map its sigmoid-space output thresholds through [`logit_upper_threshold`] /
/// [`logit_lower_threshold`] (`sigmoid(z) <= t  ⟺  z <= logit(t)` — sigmoid is
/// a strictly increasing bijection, so the rewrite is exact; the helpers round
/// the f64 logit OUTWARD so the transformed constraint is never stricter).
/// If the output node is NOT a Sigmoid, this behaves exactly like
/// [`encode_graph`] and returns `false` (thresholds stay in output space).
///
/// Sigmoid anywhere OTHER than the (peeled) output node still fails closed —
/// `encode_graph` itself stays sigmoid-free.
pub(crate) fn encode_graph_peel_final_sigmoid(
    graph: &GraphNetwork,
    input_bounds: &[Bound],
    node_bounds: &HashMap<String, Vec<Bound>>,
) -> Result<(GraphMipEncoding, bool)> {
    encode_graph_impl(graph, input_bounds, node_bounds, DELTA, true, None)
}

/// Shared implementation behind [`encode_graph_with_delta_and_deadline`]
/// (peel = false) and [`encode_graph_peel_final_sigmoid`] (peel = true).
/// Returns the encoding and whether a final Sigmoid was peeled. The optional
/// `deadline` is checked once per node in the DAG walk (see
/// [`encode_graph_with_deadline`]).
fn encode_graph_impl(
    graph: &GraphNetwork,
    input_bounds: &[Bound],
    node_bounds: &HashMap<String, Vec<Bound>>,
    delta: f64,
    peel_final_sigmoid: bool,
    deadline: Option<Instant>,
) -> Result<(GraphMipEncoding, bool)> {
    if !delta.is_finite() || delta < 0.0 {
        bail!("encode_graph: delta must be finite and non-negative, got {delta}");
    }
    let node_pre_activation_bounds = node_bounds;
    let mut problem = MilpProblem::new();

    // node name -> its output MIP columns.
    let mut cols_of: HashMap<String, Vec<Col>> = HashMap::new();

    // Columns PINNED to a single value on the MIP feasible set (increment 5).
    // Seeded from DEGENERATE input-box entries (`lb == ub`, finite — the input
    // box is exact, never inflated, so the column constraint `x ∈ [v, v]`
    // forces `x ≡ v` at every feasible point) and propagated ONLY through
    // (a) index plumbing (aliases share the Col, so pinnedness is automatic)
    // and (b) ReduceSum of all-pinned columns whose f64 sum is verified EXACT
    // (`exact_f64_sum`). Consumed by MulBinary / Div to encode the otherwise
    // bilinear op as an exact affine row (module header §increment 5).
    let mut pinned: HashMap<Col, f64> = HashMap::new();

    // Input node: one column per input element, bounds = the input box.
    // Order and creation mirror `MipEncoder::new` exactly.
    let mut input_vars = Vec::with_capacity(input_bounds.len());
    for (i, b) in input_bounds.iter().enumerate() {
        if b.lower().is_nan() || b.upper().is_nan() {
            bail!("NaN bound for input variable {i}");
        }
        let col = problem.add_col(0.0, b.lower() as f64, b.upper() as f64);
        if b.lower() == b.upper() && b.lower().is_finite() {
            pinned.insert(col, b.lower() as f64);
        }
        input_vars.push(col);
    }
    cols_of.insert(NETWORK_INPUT.to_string(), input_vars.clone());

    let mut binary_vars: Vec<Col> = Vec::new();
    let mut binary_widths: Vec<f64> = Vec::new();
    let mut binary_keys: Vec<(String, usize)> = Vec::new();

    let exec_order = graph
        .exec_order()
        .map_err(|e| anyhow!("graph exec_order failed: {e}"))?;

    // Increment 5 — final-Sigmoid peel target: `Some(sigmoid_node_name)` when
    // peeling is requested AND the output node is a single-input Sigmoid. The
    // peeled node is SKIPPED by the walk (never encoded) and the encoding's
    // output becomes its input (the logit). Any other Sigmoid fails closed.
    let peeled_sigmoid: Option<(String, String)> = if peel_final_sigmoid {
        match graph.node(graph.output_name()) {
            Some(out_node) if matches!(out_node.layer(), Layer::Sigmoid(_)) => {
                let logit = out_node
                    .require_unary_input()
                    .map_err(|e| anyhow!("final Sigmoid node: {e}"))?
                    .to_string();
                Some((out_node.name().to_string(), logit))
            }
            _ => None,
        }
    } else {
        None
    };

    for name in exec_order {
        // Deadline check per node: the conv im2col unfolds dominate the encode
        // cost, so one check per node bounds the overrun to a single unfold.
        if deadline.is_some_and(|d| Instant::now() >= d) {
            bail!("graph MIP encode deadline expired at node '{name}' (degrading cleanly)");
        }
        let node = graph
            .node(name)
            .ok_or_else(|| anyhow!("exec_order references unknown node '{name}'"))?;
        let inputs = node.inputs();

        // Inflated per-node output box for AFFINE nodes (Linear / Conv2d /
        // BatchNorm / Add): `Some([lo-delta, hi+delta]…)` when the caller
        // supplied this node's box, else `None` ⇒ free `±inf` columns. ReLU does
        // NOT consume this (its output columns follow the big-M, not a box).
        let node_box = node_bounds.get(name).map(Vec::as_slice);

        match node.layer() {
            Layer::Linear(lin) => {
                let in_cols = lookup_single(&cols_of, inputs, name)?;
                let out_cols =
                    encode_linear_node(&mut problem, lin, &in_cols, node_box, delta, name)?;
                cols_of.insert(name.clone(), out_cols);
            }
            // Increment 4 — Conv2d: unfold (im2col) into an equivalent dense
            // Linear and emit the SAME Linear equality rows. Reuses
            // `mip_preprocess::unfold_conv2d_to_linear` (which also carries the
            // fail-closed groups/dilation and nnz/memory caps → an over-large or
            // unsupported conv `bail!`s rather than OOM-aborting).
            Layer::Conv2d(conv) => {
                let in_cols = lookup_single(&cols_of, inputs, name)?;
                let (ih, iw) = conv_input_spatial(graph, conv, &inputs[0], in_cols.len(), name)?;
                let lin = unfold_conv_node(conv, ih, iw, name)?;
                let out_cols =
                    encode_linear_node(&mut problem, &lin, &in_cols, node_box, delta, name)?;
                cols_of.insert(name.clone(), out_cols);
            }
            Layer::ReLU(_) => {
                let in_cols = lookup_single(&cols_of, inputs, name)?;
                // Pre-activation box: the INPUT node's OUTPUT box FIRST (the
                // affine producer — emit_hard_six.py keys the box there, and
                // every live caller's map stores per-node OUTPUT boxes), with
                // the ReLU's own entry as the legacy fallback (the chain-test
                // convention keys pre boxes under relu names ONLY).
                //
                // SOUNDNESS (#relational-bab): the order MATTERS. A live map
                // carries the relu's POST-activation box under the relu name
                // (lower >= 0 always); reading it as the pre box makes every
                // unstable neuron encode as stable-active pass-through
                // `y = x` — NOT an over-approximation (the true (x<0, y=0)
                // points are excluded), so a certified-UNSAT on that model
                // could be WRONG. The input node's entry is the true
                // pre-activation in every such map. Inflated by `delta` for
                // the big-M.
                let pre = node_pre_activation_bounds
                    .get(&inputs[0])
                    .or_else(|| node_pre_activation_bounds.get(name))
                    .ok_or_else(|| {
                        anyhow!(
                            "ReLU node '{name}': no pre-activation bounds supplied for it or its \
                             input '{}' (needed for big-M)",
                            inputs[0]
                        )
                    })?;
                let out_cols = encode_relu_node(
                    &mut problem,
                    name,
                    &in_cols,
                    pre,
                    delta,
                    &mut binary_vars,
                    &mut binary_widths,
                    &mut binary_keys,
                )?;
                cols_of.insert(name.clone(), out_cols);
            }
            // Flatten / Reshape / Squeeze / Unsqueeze are pure index remaps that
            // PRESERVE row-major flat order: the output columns ARE the input
            // columns. No new columns or rows.
            Layer::Flatten(_) | Layer::Reshape(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => {
                let in_cols = lookup_single(&cols_of, inputs, name)?;
                cols_of.insert(name.clone(), in_cols);
            }
            // Increment 5 — Slice: pure index plumbing (each output element IS
            // one input element); the output columns alias the input columns
            // through the layer's exact forward index map.
            Layer::Slice(slice) => {
                let in_cols = lookup_single(&cols_of, inputs, name)?;
                let out_cols = alias_slice_cols(graph, slice, name, &inputs[0], &in_cols)?;
                cols_of.insert(name.clone(), out_cols);
            }
            // Increment 5 — Gather with embedded constant indices: pure index
            // plumbing (repeated indices alias the same column repeatedly).
            Layer::Gather(gather) => {
                let in_cols = lookup_single(&cols_of, inputs, name)?;
                let out_cols = alias_gather_cols(graph, gather, name, &inputs[0], &in_cols)?;
                cols_of.insert(name.clone(), out_cols);
            }
            // Increment 5 — Concat (N-ary): pure index plumbing across the
            // parents' column blocks, using the exact axis interleaving.
            Layer::Concat(concat) => {
                let out_cols = encode_concat_node(graph, concat, name, inputs, &cols_of)?;
                cols_of.insert(name.clone(), out_cols);
            }
            // Increment 5 — AddConstant / SubConstant / MulConstant /
            // DivConstant: exact per-element affine rows against the layer's
            // embedded constant. `const_op_layout` resolves each layer's exact
            // broadcast (NumPy trailing-axis alignment; AddConstant's CNN-bias
            // `[C]`→`[C,1,1]` special case mirrored) and fails closed when the
            // broadcast cannot be validated against declared shapes.
            Layer::AddConstant(ac) => {
                let in_cols = lookup_single(&cols_of, inputs, name)?;
                let (xs, cs) =
                    const_op_layout(graph, ac.constant(), &inputs[0], &in_cols, name, true)?;
                let out_cols = encode_const_affine_node(
                    &mut problem,
                    ConstAffineOp::Add,
                    &xs,
                    &cs,
                    node_box,
                    delta,
                    name,
                )?;
                cols_of.insert(name.clone(), out_cols);
            }
            Layer::SubConstant(sc) => {
                let in_cols = lookup_single(&cols_of, inputs, name)?;
                let (xs, cs) =
                    const_op_layout(graph, sc.constant(), &inputs[0], &in_cols, name, false)?;
                let op = if sc.reverse {
                    ConstAffineOp::SubFromConst
                } else {
                    ConstAffineOp::SubConst
                };
                let out_cols =
                    encode_const_affine_node(&mut problem, op, &xs, &cs, node_box, delta, name)?;
                cols_of.insert(name.clone(), out_cols);
            }
            Layer::MulConstant(mc) => {
                let in_cols = lookup_single(&cols_of, inputs, name)?;
                let (xs, cs) =
                    const_op_layout(graph, mc.constant(), &inputs[0], &in_cols, name, false)?;
                let out_cols = encode_const_affine_node(
                    &mut problem,
                    ConstAffineOp::Mul,
                    &xs,
                    &cs,
                    node_box,
                    delta,
                    name,
                )?;
                cols_of.insert(name.clone(), out_cols);
            }
            Layer::DivConstant(dc) => {
                let in_cols = lookup_single(&cols_of, inputs, name)?;
                let (xs, cs) =
                    const_op_layout(graph, dc.constant(), &inputs[0], &in_cols, name, false)?;
                let out_cols = encode_const_affine_node(
                    &mut problem,
                    ConstAffineOp::Div,
                    &xs,
                    &cs,
                    node_box,
                    delta,
                    name,
                )?;
                cols_of.insert(name.clone(), out_cols);
            }
            // Increment 5 — ReduceSum: one exact ones-row per output element
            // over the reduced axes.
            Layer::ReduceSum(rs) => {
                let in_cols = lookup_single(&cols_of, inputs, name)?;
                let in_shape = required_shape(graph, &inputs[0], in_cols.len(), name)?;
                let out_cols = encode_reduce_sum_node(
                    &mut problem,
                    rs,
                    &in_cols,
                    &in_shape,
                    graph.declared_shape(name),
                    node_box,
                    delta,
                    name,
                    &mut pinned,
                )?;
                cols_of.insert(name.clone(), out_cols);
            }
            // Increment 5 — element-wise Sub `out = A - B` (exact rows; equal
            // lengths only, mirroring the residual-Add posture).
            Layer::Sub(_) => {
                let (a_cols, b_cols) = lookup_pair(&cols_of, inputs, name)?;
                let out_cols =
                    encode_sub_node(&mut problem, &a_cols, &b_cols, node_box, delta, name)?;
                cols_of.insert(name.clone(), out_cols);
            }
            // Increment 5 — MulBinary: bilinear in general (fail-closed), but
            // EXACT affine when one operand is pinned by the instance box
            // (mscn's input-derived masks; module header §increment 5).
            Layer::MulBinary(_) => {
                let (a_cols, b_cols) = lookup_pair(&cols_of, inputs, name)?;
                let out_cols = encode_mul_binary_node(
                    &mut problem,
                    graph,
                    (a_cols.as_slice(), inputs[0].as_str()),
                    (b_cols.as_slice(), inputs[1].as_str()),
                    node_box,
                    delta,
                    name,
                    &pinned,
                )?;
                cols_of.insert(name.clone(), out_cols);
            }
            // Increment 5 — Div: affine ONLY when the denominator is pinned
            // (mscn's mask-count); `a - β·y = 0` avoids reciprocal rounding.
            Layer::Div(_) => {
                let (a_cols, b_cols) = lookup_pair(&cols_of, inputs, name)?;
                let out_cols = encode_div_node(
                    &mut problem,
                    graph,
                    (a_cols.as_slice(), inputs[0].as_str()),
                    (b_cols.as_slice(), inputs[1].as_str()),
                    node_box,
                    delta,
                    name,
                    &pinned,
                )?;
                cols_of.insert(name.clone(), out_cols);
            }
            // Increment 5 — final-Sigmoid peel: the peeled node is skipped
            // entirely (the caller rewrites output thresholds through the
            // logit); ANY other Sigmoid is nonlinear and fails closed.
            Layer::Sigmoid(_) => match &peeled_sigmoid {
                Some((sigmoid_name, _)) if sigmoid_name == name => continue,
                _ => bail!(
                    "encode_graph: Sigmoid node '{name}' is only supported as the peeled FINAL \
                     output node (see encode_graph_peel_final_sigmoid); encoding sigmoid \
                     mid-graph would require a relaxation, which this exact encoder refuses"
                ),
            },
            // Increment 2 — BatchNorm: exact per-channel affine (inference-time
            // `y_c = a_c·x_c + b_c` with the baked `scale`/`bias`).
            Layer::BatchNorm(bn) => {
                let in_cols = lookup_single(&cols_of, inputs, name)?;
                let out_cols =
                    encode_batchnorm_node(&mut problem, bn, &in_cols, node_box, delta, name)?;
                cols_of.insert(name.clone(), out_cols);
            }
            // Increment 3 — residual Add: the DAG piece. Both parents' column
            // vectors are looked up from the map (distinct upstream blocks).
            Layer::Add(_) => {
                let (a_cols, b_cols) = lookup_pair(&cols_of, inputs, name)?;
                let out_cols =
                    encode_add_node(&mut problem, &a_cols, &b_cols, node_box, delta, name)?;
                cols_of.insert(name.clone(), out_cols);
            }
            other => bail!(
                "encode_graph supports only Linear / Conv2d / ReLU / Flatten / Reshape / Squeeze / \
                 Unsqueeze / BatchNorm / Add / Sub / Slice / Gather / Concat / AddConstant / \
                 SubConstant / MulConstant / DivConstant / ReduceSum / MulBinary / Div (+ a peeled \
                 final Sigmoid); node '{name}' is {}",
                other.layer_type()
            ),
        }
    }

    // The peeled final Sigmoid re-targets the encoding's output to its INPUT
    // node (the logit); otherwise the graph output is used unchanged.
    let output_name = match &peeled_sigmoid {
        Some((_, logit_name)) => logit_name.as_str(),
        None => graph.output_name(),
    };
    let output_vars = cols_of
        .get(output_name)
        .ok_or_else(|| anyhow!("output node '{output_name}' produced no columns"))?
        .clone();

    Ok((
        GraphMipEncoding {
            problem,
            input_vars,
            output_vars,
            binary_vars,
            binary_widths,
            binary_keys,
            node_cols: cols_of,
        },
        peeled_sigmoid.is_some(),
    ))
}

/// Resolve the single input node of a chain node to its (owned) output columns.
///
/// Returns an owned `Vec` so the immutable borrow of `cols_of` ends before the
/// caller re-inserts this node's output columns.
fn lookup_single(
    cols_of: &HashMap<String, Vec<Col>>,
    inputs: &[String],
    node_name: &str,
) -> Result<Vec<Col>> {
    if inputs.len() != 1 {
        bail!(
            "chain node '{node_name}' expects exactly 1 input, got {}",
            inputs.len()
        );
    }
    cols_of.get(&inputs[0]).cloned().ok_or_else(|| {
        anyhow!(
            "node '{node_name}': input '{}' has no encoded columns",
            inputs[0]
        )
    })
}

/// Resolve the TWO input nodes of a binary node (e.g. residual `Add`) to their
/// (owned) output column vectors, in `node.inputs()` order — `(A, B)`.
///
/// This is the multi-input generalization of [`lookup_single`]: `A` and `B` are
/// two DIFFERENT upstream blocks (the skip connection's two operands), which is
/// exactly why the encoder must walk the graph rather than a linear chain.
fn lookup_pair(
    cols_of: &HashMap<String, Vec<Col>>,
    inputs: &[String],
    node_name: &str,
) -> Result<(Vec<Col>, Vec<Col>)> {
    if inputs.len() != 2 {
        bail!(
            "binary node '{node_name}' expects exactly 2 inputs, got {}",
            inputs.len()
        );
    }
    let lookup = |which: usize| -> Result<Vec<Col>> {
        cols_of.get(&inputs[which]).cloned().ok_or_else(|| {
            anyhow!(
                "node '{node_name}': input '{}' has no encoded columns",
                inputs[which]
            )
        })
    };
    Ok((lookup(0)?, lookup(1)?))
}

/// Column `[lo, hi]` bounds for output element `i` of an affine node: the
/// caller's per-node CROWN/BaB box inflated by `±delta` when supplied, else free
/// `±inf`.
///
/// Box inflation only WEAKENS the model (module header §"DELTA box inflation"),
/// so this is sound for any `delta >= 0` and any box that soundly bounds the
/// node's reachable output; the free `±inf` fallback is looser still. A
/// `delta == 0.0` with no box reproduces the exact free-column encoding of
/// `encode_feedforward` (the increment-1 invariant).
#[inline]
fn out_col_bounds(node_box: Option<&[Bound]>, i: usize, delta: f64) -> (f64, f64) {
    match node_box {
        Some(b) => (b[i].lower() as f64 - delta, b[i].upper() as f64 + delta),
        None => (f64::NEG_INFINITY, f64::INFINITY),
    }
}

/// Validate that a supplied per-node output box has exactly one entry per output
/// element (fail-closed: a mismatched box would silently mis-bound columns).
fn check_box_len(node_box: Option<&[Bound]>, expected: usize, node_name: &str) -> Result<()> {
    if let Some(b) = node_box {
        if b.len() != expected {
            bail!(
                "node '{node_name}': supplied intermediate box has {} entries but the node's \
                 output tensor has {expected} elements",
                b.len()
            );
        }
    }
    Ok(())
}

/// Encode a Linear node `y = Wx + b` as one equality row per output.
///
/// Mirrors `ny_mip::encoder::MipEncoder::encode_linear` EXACTLY when `node_box`
/// is `None`, `delta == 0.0`, and the input is a plain vector: each output is a
/// free (`±inf`) column, the row is `sum_j(W_ij x_j) - y_i = -b_i` with zero
/// weights skipped, and the `-y_i` coefficient appended last. When the caller
/// supplies this node's intermediate box, the output columns are additionally
/// bounded by the DELTA-inflated box (sound; see [`out_col_bounds`]) — the
/// exact equality row is unchanged.
///
/// Increment 5 — ROW-BATCHED inputs. `LinearLayer`'s forward contracts the
/// LAST axis: `[..., in_features] -> [..., out_features]` (linear/ibp.rs). Any
/// valid input tensor therefore flattens row-major to `[R, in_features]` with
/// `R = in_cols.len() / in_features` (the leading axes flatten into R), and the
/// output flattens to `[R, out_features]`. We emit the same exact per-output
/// equality row per (r, i), iterating r outer / i inner to match the row-major
/// output flat order. `R == 1` reproduces the increment-1 byte-identical
/// encoding. A non-divisible column count fails closed (the real forward would
/// reject such an input too).
fn encode_linear_node(
    problem: &mut MilpProblem,
    lin: &LinearLayer,
    in_cols: &[Col],
    node_box: Option<&[Bound]>,
    delta: f64,
    node_name: &str,
) -> Result<Vec<Col>> {
    let out_features = lin.out_features();
    let in_features = lin.in_features();
    if in_features == 0 || out_features == 0 {
        bail!("Linear node '{node_name}': zero in/out features");
    }
    if !in_cols.len().is_multiple_of(in_features) {
        bail!(
            "Linear node '{node_name}': input column count {} is not a multiple of in_features {}",
            in_cols.len(),
            in_features
        );
    }
    let batch_rows = in_cols.len() / in_features;
    check_box_len(node_box, batch_rows * out_features, node_name)?;
    let weight = &lin.weight; // Array2<f32>, shape [out_features, in_features]

    let mut out_cols = Vec::with_capacity(batch_rows * out_features);
    for r in 0..batch_rows {
        let row_cols = &in_cols[r * in_features..(r + 1) * in_features];
        for i in 0..out_features {
            // y_{r,i}: free (`±inf`) unless the caller pinned this node's
            // inflated box (row-major flat index r*out_features + i).
            let (lo, hi) = out_col_bounds(node_box, r * out_features + i, delta);
            let y = problem.add_col(0.0, lo, hi);

            let mut coeffs: Vec<(Col, f64)> = Vec::with_capacity(in_features + 1);
            for (j, &x) in row_cols.iter().enumerate() {
                let w = weight[[i, j]] as f64;
                if w != 0.0 {
                    coeffs.push((x, w));
                }
            }
            coeffs.push((y, -1.0));

            // Equality: sum(W_ij x_j) - y_i = -b_i. Absent bias => b_i = 0.
            let neg_b = match &lin.bias {
                Some(b) => -(b[i] as f64),
                None => 0.0,
            };
            problem.add_row(neg_b, neg_b, coeffs);

            out_cols.push(y);
        }
    }
    Ok(out_cols)
}

/// Determine a Conv2d node's input spatial dimensions `(in_h, in_w)` so it can be
/// im2col-unfolded into a dense Linear (increment 4).
///
/// The unfold needs `in_channels * in_h * in_w == in_cols.len()`. Sources, in
/// order of preference:
///   1. `conv.input_shape` — set by ny's load-time / forward-linear shape pass
///      (a real cifar100 conv carries it, since Conv2d CROWN backward requires
///      it); validated against the input column count.
///   2. `graph.declared_shape(input_node)` — the load-time shape-inferred output
///      shape of the producer node, in ny's unbatched `[…, C, H, W]` convention;
///      the last two dims are `(in_h, in_w)`, validated against the column count.
///   3. Square fallback: if `in_cols.len() / in_channels` is a perfect square,
///      assume `in_h == in_w`.
///
/// Fails closed (`bail!`) when none yields a consistent shape — the caller then
/// degrades to the already-sound bound-propagation verdict rather than building
/// a wrong im2col matrix.
fn conv_input_spatial(
    graph: &GraphNetwork,
    conv: &Conv2dLayer,
    input_node: &str,
    in_cols_len: usize,
    node_name: &str,
) -> Result<(usize, usize)> {
    let ic = conv.in_channels();
    if ic == 0 {
        bail!("Conv2d node '{node_name}': zero input channels");
    }
    let consistent = |ih: usize, iw: usize| -> bool {
        ic.checked_mul(ih)
            .and_then(|v| v.checked_mul(iw))
            .is_some_and(|v| v == in_cols_len)
    };

    // 1. The conv's own recorded input spatial shape.
    if let Some((ih, iw)) = conv.input_shape {
        if consistent(ih, iw) {
            return Ok((ih, iw));
        }
        bail!(
            "Conv2d node '{node_name}': recorded input_shape ({ih},{iw}) with {ic} in-channels = \
             {} elements != {in_cols_len} input columns",
            ic.saturating_mul(ih).saturating_mul(iw)
        );
    }

    // 2. The producer node's declared (load-time shape-inferred) output shape.
    if let Some(shape) = graph.declared_shape(input_node) {
        if shape.len() >= 2 {
            let ih = shape[shape.len() - 2];
            let iw = shape[shape.len() - 1];
            if consistent(ih, iw) {
                return Ok((ih, iw));
            }
        }
    }

    // 3. Square fallback: infer ih == iw from the flat input count.
    if in_cols_len.is_multiple_of(ic) {
        let spatial = in_cols_len / ic;
        let side = (spatial as f64).sqrt().round() as usize;
        if side > 0 && side * side == spatial {
            return Ok((side, side));
        }
    }

    bail!(
        "Conv2d node '{node_name}': cannot determine input spatial shape. \
         conv.input_shape is unset, producer '{input_node}' has no usable declared shape, and \
         {in_cols_len} input columns / {ic} in-channels is not a perfect square. \
         The MIP encoding degrades to the inconclusive bound-propagation verdict."
    )
}

/// Unfold a single Conv2d node into its equivalent dense Linear (im2col) by
/// reusing [`unfold_conv2d_to_linear`] on a one-layer sequential `Network`.
///
/// The reused helper carries the fail-closed soundness gates for free: it
/// `bail!`s on grouped/dilated convs (whose dense im2col would index out of
/// bounds / build the wrong matrix) and on unfoldings past a ~1 GB nnz cap
/// (which would OOM-abort) — both degrade the MIP escalation to the already-sound
/// bound-propagation verdict rather than crashing or emitting a wrong result.
fn unfold_conv_node(
    conv: &Conv2dLayer,
    in_h: usize,
    in_w: usize,
    node_name: &str,
) -> Result<LinearLayer> {
    let mut conv = conv.clone();
    conv.set_input_shape(in_h, in_w);
    let ic = conv.in_channels();

    let mut net = Network::new();
    net.add_layer(Layer::Conv2d(conv));
    let unfolded = unfold_conv2d_to_linear(&net, &[ic, in_h, in_w])
        .map_err(|e| anyhow!("Conv2d node '{node_name}': im2col unfold failed: {e}"))?;

    match unfolded.layers() {
        [Layer::Linear(lin)] => Ok(lin.clone()),
        other => bail!(
            "Conv2d node '{node_name}': im2col unfold produced {} layer(s), expected exactly one \
             Linear",
            other.len()
        ),
    }
}

/// Encode a BatchNorm node as an EXACT per-channel affine (increment 2).
///
/// At inference BatchNorm is `y_c = a_c·x_c + b_c` where the layer already bakes
/// `a_c = γ_c/√(σ²_c+ε)` into [`BatchNormLayer::scale`] and
/// `b_c = β_c − γ_c·μ_c/√(σ²_c+ε)` into [`BatchNormLayer::bias`] (both per
/// channel). We emit ONE free output column and ONE equality row per element:
/// `out_i − a_c·in_i = b_c`, i.e. `add_row(b_c, b_c, [(out_i, 1.0), (in_i, -a_c)])`.
///
/// This is standalone (not folded into an adjacent Linear/Conv) — an affine
/// equality is already exact, and folding is a future zero-column optimization,
/// not a soundness requirement. Using the baked f32 `scale`/`bias` (cast to f64)
/// as exact rational coefficients is the SAME posture as `encode_linear_node`
/// (which uses the f32 `weight`/`bias` as exact coefficients): the f32-folded
/// affine IS the layer ny evaluates at inference.
///
/// SOUNDNESS (increment 4 — the fail-closed `scale_err`/`bias_err` gate is
/// REMOVED). `scale`/`bias` are ny's f32-ROUNDED folded coefficients; the layer
/// carries per-channel certified `scale_err`/`bias_err` bounding the gap to the
/// pre-fold REAL affine. An exact equality row `out = scale·in + bias` alone
/// could be TIGHTER than the real net by up to `scale_err·|in| + bias_err` per
/// element (a false-UNSAT risk). That gap is now absorbed by the DELTA box
/// inflation applied to EVERY downstream intermediate box (module header
/// §"DELTA box inflation"; emit_hard_six.py measured the fold-inclusive f32↔f64
/// gap ≤ 1.1e-5 < DELTA = 1e-4 and folds Conv+BN into the block affine under the
/// same posture). Concretely: the inflated feasible set of this BN's consumers
/// ⊇ the real reachable set, so an ay `unsat` still certifies the subdomain. We
/// therefore keep the EXACT equality row and rely on the caller's inflated boxes,
/// rather than fail-closing on nonzero fold error. (A `±inf` scale from a
/// degenerate var+ε→0 channel is still refused — it would poison the row.)
///
/// Channel mapping: columns are the tensor flattened row-major with the channel
/// as the slowest-varying axis after a batch of 1 (`[C, spatial…]` → flat index
/// `i` has channel `i / elements_per_channel`), matching
/// `BatchNormInputLayout::channel_for_flat_index` in the f64-cell path. For a
/// BatchNorm on a plain vector (`elements_per_channel == 1`) each element is its
/// own channel.
fn encode_batchnorm_node(
    problem: &mut MilpProblem,
    bn: &BatchNormLayer,
    in_cols: &[Col],
    node_box: Option<&[Bound]>,
    delta: f64,
    node_name: &str,
) -> Result<Vec<Col>> {
    let num_channels = bn.num_channels;
    let n = in_cols.len();
    if num_channels == 0 {
        bail!("BatchNorm node '{node_name}': zero channels");
    }
    if !n.is_multiple_of(num_channels) {
        bail!(
            "BatchNorm node '{node_name}': input element count {n} not divisible by channel \
             count {num_channels}"
        );
    }
    let elements_per_channel = n / num_channels;
    check_box_len(node_box, n, node_name)?;

    // Row-major flatten of the per-channel affine coefficients (channel order).
    let scale: Vec<f32> = bn.scale.iter().copied().collect();
    let bias: Vec<f32> = bn.bias.iter().copied().collect();
    if scale.len() != num_channels || bias.len() != num_channels {
        bail!(
            "BatchNorm node '{node_name}': scale/bias length ({}, {}) != num_channels {num_channels}",
            scale.len(),
            bias.len()
        );
    }

    // NOTE: the former fail-closed `scale_err`/`bias_err` gate is REMOVED
    // (increment 4). The f32-fold error is now absorbed by the DELTA box
    // inflation on every downstream intermediate box (see the fn-level SOUNDNESS
    // doc and the module header). We keep the EXACT f32-folded equality row.

    let mut out_cols = Vec::with_capacity(n);
    for (i, &x) in in_cols.iter().enumerate() {
        let c = i / elements_per_channel; // channel of flat position i (batch = 1).
        let a_c = scale[c] as f64;
        let b_c = bias[c] as f64;
        // A non-finite coefficient (degenerate var+eps→0 channel → ±inf scale)
        // would poison the row / solver; fail closed rather than emit garbage.
        if !a_c.is_finite() || !b_c.is_finite() {
            bail!(
                "BatchNorm node '{node_name}' channel {c}: non-finite affine coefficient \
                 (scale={a_c}, bias={b_c})"
            );
        }
        // Output column: free (±inf) unless the caller pinned this node's
        // DELTA-inflated intermediate box.
        let (lo, hi) = out_col_bounds(node_box, i, delta);
        let y = problem.add_col(0.0, lo, hi);
        // Exact affine equality: out_i − a_c·in_i = b_c.
        problem.add_row(b_c, b_c, [(y, 1.0), (x, -a_c)]);
        out_cols.push(y);
    }
    Ok(out_cols)
}

/// Encode a residual `Add` node `out = A + B` (element-wise) as EXACT equality
/// rows (increment 3 — the DAG piece).
///
/// One output column and one equality row per element:
/// `out_i − A_i − B_i = 0`, i.e. `add_row(0.0, 0.0, [(out_i, 1.0), (A_i, -1.0),
/// (B_i, -1.0)])`. `A` and `B` are the two parents resolved by [`lookup_pair`]
/// (distinct upstream var-blocks — the skip connection). The output column is
/// free (`±inf`) unless the caller pinned this node's DELTA-inflated box.
///
/// Only equal-length (non-broadcasting) Add is supported: a residual skip sums
/// two tensors of identical shape, so the column vectors must have equal length.
/// A broadcasting Add (rank/shape mismatch) is rejected — replicating columns
/// per NumPy broadcast is a later concern and not needed for conv-resnet skips.
fn encode_add_node(
    problem: &mut MilpProblem,
    a_cols: &[Col],
    b_cols: &[Col],
    node_box: Option<&[Bound]>,
    delta: f64,
    node_name: &str,
) -> Result<Vec<Col>> {
    if a_cols.len() != b_cols.len() {
        bail!(
            "Add node '{node_name}': parent column counts differ ({} vs {}); broadcasting Add is \
             not supported (residual skip operands must have equal length)",
            a_cols.len(),
            b_cols.len()
        );
    }
    check_box_len(node_box, a_cols.len(), node_name)?;
    let mut out_cols = Vec::with_capacity(a_cols.len());
    for (i, (&a, &b)) in a_cols.iter().zip(b_cols.iter()).enumerate() {
        // Output column: free (±inf) unless the caller pinned this node's box.
        let (lo, hi) = out_col_bounds(node_box, i, delta);
        let y = problem.add_col(0.0, lo, hi);
        // Exact element-wise sum: out_i − A_i − B_i = 0.
        problem.add_row(0.0, 0.0, [(y, 1.0), (a, -1.0), (b, -1.0)]);
        out_cols.push(y);
    }
    Ok(out_cols)
}

/// Encode a ReLU node using the big-M formulation, keyed on per-neuron
/// pre-activation bounds `[l, u]` INFLATED by `±delta`.
///
/// Mirrors `ny_mip::encoder::MipEncoder::encode_relu` EXACTLY (when `delta ==
/// 0.0`), including the stable-neuron special-casing (`l >= 0` pass-through,
/// `u <= 0` fixed zero) and the exact row shapes / argument order for the three
/// unstable-neuron rows. The `delta` inflation (`l-delta, u+delta`) is the
/// cifar100 soundness mechanism: it can only DEMOTE a stable neuron to unstable
/// (adding a big-M relaxation ⇒ more feasible points) and widens the big-M
/// constant `u` and the output column ub — all strictly WEAKENING, hence sound
/// (module header §"DELTA box inflation"). emit_hard_six.py applies the SAME
/// `pl, pu = infl(box)` to its `add_relu`.
fn encode_relu_node(
    problem: &mut MilpProblem,
    relu_name: &str,
    in_cols: &[Col],
    pre_activation_bounds: &[Bound],
    delta: f64,
    binary_vars: &mut Vec<Col>,
    binary_widths: &mut Vec<f64>,
    binary_keys: &mut Vec<(String, usize)>,
) -> Result<Vec<Col>> {
    if pre_activation_bounds.len() != in_cols.len() {
        bail!(
            "ReLU bounds dimension mismatch: {} cols, {} bounds",
            in_cols.len(),
            pre_activation_bounds.len()
        );
    }

    let mut out_cols = Vec::with_capacity(in_cols.len());
    for (i, bound) in pre_activation_bounds.iter().enumerate() {
        let x = in_cols[i];
        // DELTA-inflated pre-activation box (l-delta, u+delta): sound (only
        // weakens the relaxation). Matches emit_hard_six.py's `infl(box)`.
        let lb = bound.lower() as f64 - delta;
        let ub = bound.upper() as f64 + delta;

        if lb >= 0.0 {
            // Always active: y = x (no new variable).
            out_cols.push(x);
        } else if ub <= 0.0 {
            // Always inactive: y = 0 (fixed variable).
            let y = problem.add_col(0.0, 0.0, 0.0);
            out_cols.push(y);
        } else {
            // Unstable neuron: big-M encoding with tight bounds.
            let y = problem.add_col(0.0, 0.0, ub);
            let z = problem.add_integer_col(0.0, 0.0, 1.0);
            binary_vars.push(z);
            binary_widths.push(ub - lb);
            binary_keys.push((relu_name.to_string(), i));

            // y >= x  =>  y - x >= 0
            problem.add_row(0.0, f64::INFINITY, [(y, 1.0), (x, -1.0)]);
            // y <= x - l*(1-z) = x + l*z - l  =>  y - x - l*z <= -l
            problem.add_row(f64::NEG_INFINITY, -lb, [(y, 1.0), (x, -1.0), (z, -lb)]);
            // y <= u*z  =>  y - u*z <= 0
            problem.add_row(f64::NEG_INFINITY, 0.0, [(y, 1.0), (z, -ub)]);

            out_cols.push(y);
        }
    }
    Ok(out_cols)
}

// ── increment 5 helpers: shapes / index math ────────────────────────────────

/// Row-major strides for `shape` (empty shape = scalar → no strides).
fn strides_for(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Overflow-checked element count of `shape` (scalar/empty shape → 1).
fn checked_product(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// Resolve a possibly negative ONNX-style axis against rank `ndim`
/// (mirrors `layers::common::resolve_axis`: negative axes add `ndim`).
fn resolve_axis_signed(axis: i64, ndim: usize, node_name: &str) -> Result<usize> {
    let resolved = if axis < 0 { axis + ndim as i64 } else { axis };
    if resolved < 0 || resolved >= ndim as i64 {
        bail!("node '{node_name}': axis {axis} out of range for rank {ndim}");
    }
    Ok(resolved as usize)
}

/// The graph's load-time declared (shape-inferred) shape for `name`, REQUIRED
/// and validated against the encoded column count. Fail-closed on a missing or
/// contradicting shape — a wrong shape would mis-wire index plumbing /
/// broadcasting, which is a wrong-verdict risk, so we refuse to guess.
fn required_shape(
    graph: &GraphNetwork,
    name: &str,
    n_elems: usize,
    ctx: &str,
) -> Result<Vec<usize>> {
    let Some(shape) = graph.declared_shape(name) else {
        bail!(
            "node '{ctx}': no declared shape for '{name}' (needed for exact index/broadcast \
             math; fail closed)"
        );
    };
    let prod = checked_product(shape)
        .ok_or_else(|| anyhow!("node '{ctx}': declared shape {shape:?} of '{name}' overflows"))?;
    if prod != n_elems {
        bail!(
            "node '{ctx}': declared shape {shape:?} of '{name}' has {prod} elements but \
             {n_elems} columns are encoded (fail closed)"
        );
    }
    Ok(shape.to_vec())
}

/// [`required_shape`] that tolerates a missing declared shape ONLY for a
/// single-element operand (shape `[1]` — a scalar broadcasts identically
/// regardless of its true rank, so no guess is involved).
fn operand_shape(
    graph: &GraphNetwork,
    name: &str,
    n_elems: usize,
    ctx: &str,
) -> Result<Vec<usize>> {
    if graph.declared_shape(name).is_none() && n_elems == 1 {
        return Ok(vec![1]);
    }
    required_shape(graph, name, n_elems, ctx)
}

/// Flat index into `src_shape` for broadcast source lookup: NumPy trailing-axis
/// alignment (src dim `d` aligns with out dim `d + (out_ndim - src_ndim)`;
/// size-1 src dims pin their coordinate to 0). Caller must have validated
/// broadcast compatibility via [`broadcast_shapes`].
fn broadcast_src_flat(out_multi: &[usize], src_shape: &[usize]) -> usize {
    let offset = out_multi.len() - src_shape.len();
    let mut flat = 0usize;
    for (d, &s) in src_shape.iter().enumerate() {
        let coord = if s == 1 { 0 } else { out_multi[d + offset] };
        flat = flat * s + coord;
    }
    flat
}

/// Decompose `flat` into a multi-index over `shape` (row-major).
fn unflatten_index(flat: usize, strides: &[usize], out: &mut Vec<usize>) {
    out.clear();
    let mut rem = flat;
    for &s in strides {
        out.push(rem / s);
        rem %= s;
    }
}

// ── increment 5: index-plumbing encodings (Slice / Gather / Concat) ─────────

/// Alias a Slice node's output columns onto its input columns using the
/// layer's EXACT forward index map (mirrors `SliceLayer`'s out-flat → in-flat
/// mapping in `propagate_linear_batched`, including ONNX start/end clamping via
/// the layer's own `compute_output_shape`).
///
/// Input shape: the producer's declared shape when present; otherwise (e.g.
/// the producer is the `_input` sentinel, which carries no declared shape) it
/// is RECONSTRUCTED from this node's own declared OUTPUT shape — a slice keeps
/// every dim except the sliced axis, whose length is uniquely fixed by the
/// input column count. Both paths are validated (`compute_output_shape` must
/// reproduce the declared output shape); any inconsistency fails closed.
fn alias_slice_cols(
    graph: &GraphNetwork,
    slice: &SliceLayer,
    node_name: &str,
    input_name: &str,
    in_cols: &[Col],
) -> Result<Vec<Col>> {
    let in_shape: Vec<usize> = if graph.declared_shape(input_name).is_some() {
        required_shape(graph, input_name, in_cols.len(), node_name)?
    } else if let Some(out_decl) = graph.declared_shape(node_name) {
        let ndim = out_decl.len();
        let axis = resolve_axis_signed(slice.axis as i64, ndim, node_name)?;
        let rest = checked_product(
            &out_decl
                .iter()
                .enumerate()
                .filter_map(|(d, &v)| (d != axis).then_some(v))
                .collect::<Vec<_>>(),
        )
        .ok_or_else(|| anyhow!("Slice node '{node_name}': shape product overflow"))?;
        if rest == 0 || !in_cols.len().is_multiple_of(rest) {
            bail!(
                "Slice node '{node_name}': cannot reconstruct input shape from output shape \
                 {out_decl:?} and {} input columns (fail closed)",
                in_cols.len()
            );
        }
        let mut s = out_decl.to_vec();
        s[axis] = in_cols.len() / rest;
        s
    } else {
        bail!(
            "Slice node '{node_name}': no declared shape for it or its input '{input_name}' — \
             cannot derive the exact index map (fail closed)"
        );
    };

    // The layer's own ONNX-clamping output-shape logic (validates the range).
    let out_shape = slice
        .compute_output_shape(&in_shape)
        .map_err(|e| anyhow!("Slice node '{node_name}': {e}"))?;
    if let Some(decl) = graph.declared_shape(node_name) {
        if decl != out_shape.as_slice() {
            bail!(
                "Slice node '{node_name}': computed output shape {out_shape:?} contradicts the \
                 declared shape {decl:?} (fail closed)"
            );
        }
    }

    let ndim = in_shape.len();
    let axis = resolve_axis_signed(slice.axis as i64, ndim, node_name)?;
    // ONNX clamps start to the axis length (mirrors SliceLayer::validate_range).
    let start = slice.start.min(in_shape[axis]);
    let out_size = checked_product(&out_shape)
        .ok_or_else(|| anyhow!("Slice node '{node_name}': output shape overflow"))?;
    let in_strides = strides_for(&in_shape);
    let out_strides = strides_for(&out_shape);

    let mut out_cols = Vec::with_capacity(out_size);
    let mut multi = Vec::with_capacity(ndim);
    for out_flat in 0..out_size {
        unflatten_index(out_flat, &out_strides, &mut multi);
        multi[axis] += start;
        let in_flat: usize = multi.iter().zip(&in_strides).map(|(&c, &s)| c * s).sum();
        out_cols.push(in_cols[in_flat]);
    }
    Ok(out_cols)
}

/// Alias a Gather node's output columns onto its input columns (ONNX Gather:
/// output shape replaces the gathered axis with the indices shape;
/// `out[outer, idx, inner] = in[outer, indices[idx], inner]`). Only the
/// standard mode with EMBEDDED CONSTANT indices is exact index plumbing;
/// dynamic indices and the runtime shape-query mode fail closed. Negative
/// indices resolve ONNX-style (`+ axis_len`) and are bounds-checked.
fn alias_gather_cols(
    graph: &GraphNetwork,
    gather: &GatherLayer,
    node_name: &str,
    input_name: &str,
    in_cols: &[Col],
) -> Result<Vec<Col>> {
    if gather.is_runtime_last_axis_len() {
        bail!(
            "Gather node '{node_name}': runtime shape-query gather is not index plumbing \
             (fail closed)"
        );
    }
    let Some(indices) = gather.constant_indices() else {
        bail!(
            "Gather node '{node_name}': dynamic (non-constant) indices are not exact index \
             plumbing (fail closed)"
        );
    };
    let in_shape = required_shape(graph, input_name, in_cols.len(), node_name)?;
    let ndim = in_shape.len();
    let axis = resolve_axis_signed(gather.axis_raw(), ndim, node_name)?;
    let axis_len = in_shape[axis];
    let outer = checked_product(&in_shape[..axis])
        .ok_or_else(|| anyhow!("Gather node '{node_name}': shape overflow"))?;
    let inner = checked_product(&in_shape[axis + 1..])
        .ok_or_else(|| anyhow!("Gather node '{node_name}': shape overflow"))?;

    // ONNX output shape: in[..axis] ++ indices.shape ++ in[axis+1..].
    let mut out_shape: Vec<usize> = in_shape[..axis].to_vec();
    out_shape.extend_from_slice(indices.shape());
    out_shape.extend_from_slice(&in_shape[axis + 1..]);
    if let Some(decl) = graph.declared_shape(node_name) {
        if decl != out_shape.as_slice() {
            bail!(
                "Gather node '{node_name}': computed output shape {out_shape:?} contradicts the \
                 declared shape {decl:?} (fail closed)"
            );
        }
    }

    let resolved_indices: Vec<usize> = indices
        .iter()
        .map(|&raw| {
            let j = if raw < 0 { raw + axis_len as i64 } else { raw };
            if j < 0 || j >= axis_len as i64 {
                bail!(
                    "Gather node '{node_name}': index {raw} out of range for axis length \
                     {axis_len}"
                );
            }
            Ok(j as usize)
        })
        .collect::<Result<_>>()?;

    let mut out_cols = Vec::with_capacity(outer * resolved_indices.len() * inner);
    for o in 0..outer {
        for &j in &resolved_indices {
            let base = (o * axis_len + j) * inner;
            for i in 0..inner {
                out_cols.push(in_cols[base + i]);
            }
        }
    }
    Ok(out_cols)
}

/// Alias a Concat node's output columns onto its parents' columns (N-ary;
/// exact axis interleaving). All operands must share rank and non-axis dims;
/// the mapping walks each output element to the parent block segment its axis
/// coordinate falls in. Declared shapes are REQUIRED for every operand (the
/// interleaving depends on them) and validated; embedded constant operands
/// fail closed (they have no columns to alias).
fn encode_concat_node(
    graph: &GraphNetwork,
    concat: &ny_propagate::layers::ConcatLayer,
    node_name: &str,
    inputs: &[String],
    cols_of: &HashMap<String, Vec<Col>>,
) -> Result<Vec<Col>> {
    if concat.constant_inputs.is_some() {
        bail!(
            "Concat node '{node_name}': embedded constant operands are not supported \
             (fail closed)"
        );
    }
    if inputs.is_empty() {
        bail!("Concat node '{node_name}': no inputs");
    }
    let blocks: Vec<Vec<Col>> = inputs
        .iter()
        .map(|inp| {
            cols_of
                .get(inp)
                .cloned()
                .ok_or_else(|| anyhow!("node '{node_name}': input '{inp}' has no encoded columns"))
        })
        .collect::<Result<_>>()?;
    let shapes: Vec<Vec<usize>> = inputs
        .iter()
        .zip(&blocks)
        .map(|(inp, block)| required_shape(graph, inp, block.len(), node_name))
        .collect::<Result<_>>()?;

    let ndim = shapes[0].len();
    if shapes.iter().any(|s| s.len() != ndim) {
        bail!("Concat node '{node_name}': operand ranks differ (fail closed)");
    }
    let axis = resolve_axis_signed(concat.axis, ndim, node_name)?;
    for s in &shapes[1..] {
        for d in 0..ndim {
            if d != axis && s[d] != shapes[0][d] {
                bail!(
                    "Concat node '{node_name}': operand shapes {:?} vs {s:?} differ off the \
                     concat axis {axis} (fail closed)",
                    shapes[0]
                );
            }
        }
    }

    let mut out_shape = shapes[0].clone();
    out_shape[axis] = shapes.iter().map(|s| s[axis]).sum();
    if let Some(decl) = graph.declared_shape(node_name) {
        if decl != out_shape.as_slice() {
            bail!(
                "Concat node '{node_name}': computed output shape {out_shape:?} contradicts the \
                 declared shape {decl:?} (fail closed)"
            );
        }
    }

    // Cumulative axis offsets: segment k covers [offsets[k], offsets[k+1]).
    let mut offsets = Vec::with_capacity(shapes.len() + 1);
    offsets.push(0usize);
    for s in &shapes {
        offsets.push(offsets.last().unwrap() + s[axis]);
    }

    let out_size = checked_product(&out_shape)
        .ok_or_else(|| anyhow!("Concat node '{node_name}': output shape overflow"))?;
    let out_strides = strides_for(&out_shape);
    let mut out_cols = Vec::with_capacity(out_size);
    let mut multi = Vec::with_capacity(ndim);
    for out_flat in 0..out_size {
        unflatten_index(out_flat, &out_strides, &mut multi);
        let c = multi[axis];
        // Linear scan over the (few) operands for the owning segment.
        let k = (0..shapes.len())
            .find(|&k| c >= offsets[k] && c < offsets[k + 1])
            .expect("concat coordinate must fall in one segment");
        let local_strides = strides_for(&shapes[k]);
        let mut in_flat = 0usize;
        for d in 0..ndim {
            let coord = if d == axis { c - offsets[k] } else { multi[d] };
            in_flat += coord * local_strides[d];
        }
        out_cols.push(blocks[k][in_flat]);
    }
    Ok(out_cols)
}

// ── increment 5: constant-operand affine ops ────────────────────────────────

/// Which `*Constant` affine op a row encodes (all exact; see
/// [`encode_const_affine_node`]).
#[derive(Clone, Copy, Debug)]
enum ConstAffineOp {
    /// `y = x + c`
    Add,
    /// `y = x - c`
    SubConst,
    /// `y = c - x`
    SubFromConst,
    /// `y = c · x`
    Mul,
    /// `y = x / c` — encoded as `x - c·y = 0` (multiplying through by the
    /// divisor keeps the row exactly rational: no `1/c` reciprocal rounding).
    Div,
}

/// Resolve a `*Constant` layer's embedded constant against its input into
/// per-OUTPUT-element `(input column, constant value)` pairs, mirroring the
/// layer's forward broadcast EXACTLY:
///   * scalar constant (1 element) — applies to every input element;
///   * `cnn_bias_special_case` (AddConstant only, arithmetic/add_constant.rs):
///     a 1-D `[C]` constant with a 3-D `[C, H, W]` input and matching `C` is
///     treated as `[C, 1, 1]` (per-channel bias) BEFORE NumPy broadcasting;
///   * otherwise full NumPy trailing-axis broadcasting of BOTH operands to the
///     broadcast output shape (the output may be LARGER than the input — e.g.
///     constant `[4]` × input `[1]` → output `[4]`), exactly as the layers'
///     `propagate_ibp` does.
///
/// The input shape comes from the graph's validated declared shape; a missing
/// shape is only tolerated for a scalar constant (fail closed otherwise).
fn const_op_layout(
    graph: &GraphNetwork,
    constant: &ArrayD<f32>,
    input_name: &str,
    in_cols: &[Col],
    node_name: &str,
    cnn_bias_special_case: bool,
) -> Result<(Vec<Col>, Vec<f64>)> {
    let cvals: Vec<f64> = constant.iter().map(|&v| v as f64).collect();
    if cvals.iter().any(|v| !v.is_finite()) {
        // The layer constructors validate finiteness; defense in depth.
        bail!("node '{node_name}': non-finite constant entry (fail closed)");
    }
    if cvals.len() == 1 {
        return Ok((in_cols.to_vec(), vec![cvals[0]; in_cols.len()]));
    }

    let in_shape = required_shape(graph, input_name, in_cols.len(), node_name)?;
    // AddConstant's CNN-bias special case: [C] against 3-D [C, H, W] → [C,1,1].
    let const_shape: Vec<usize> = if cnn_bias_special_case
        && constant.ndim() == 1
        && in_shape.len() == 3
        && constant.len() == in_shape[0]
    {
        vec![in_shape[0], 1, 1]
    } else {
        constant.shape().to_vec()
    };

    let Some(out_shape) = broadcast_shapes(&in_shape, &const_shape) else {
        bail!(
            "node '{node_name}': constant shape {const_shape:?} does not broadcast against \
             input shape {in_shape:?} (fail closed)"
        );
    };
    if let Some(decl) = graph.declared_shape(node_name) {
        let decl_prod = checked_product(decl)
            .ok_or_else(|| anyhow!("node '{node_name}': declared shape overflow"))?;
        let out_prod = checked_product(&out_shape)
            .ok_or_else(|| anyhow!("node '{node_name}': broadcast shape overflow"))?;
        if decl_prod != out_prod {
            bail!(
                "node '{node_name}': broadcast output shape {out_shape:?} contradicts the \
                 declared shape {decl:?} (fail closed)"
            );
        }
    }

    let out_size = checked_product(&out_shape)
        .ok_or_else(|| anyhow!("node '{node_name}': broadcast shape overflow"))?;
    let out_strides = strides_for(&out_shape);
    let mut xs = Vec::with_capacity(out_size);
    let mut cs = Vec::with_capacity(out_size);
    let mut multi = Vec::with_capacity(out_shape.len());
    for out_flat in 0..out_size {
        unflatten_index(out_flat, &out_strides, &mut multi);
        xs.push(in_cols[broadcast_src_flat(&multi, &in_shape)]);
        cs.push(cvals[broadcast_src_flat(&multi, &const_shape)]);
    }
    Ok((xs, cs))
}

/// Encode a `*Constant` affine op as one EXACT equality row per output element
/// over the pre-resolved `(x, c)` layout from [`const_op_layout`]:
///   * Add          `x - y = -c`
///   * SubConst     `x - y =  c`
///   * SubFromConst `x + y =  c`
///   * Mul          `c·x - y = 0` (zero constants skip the x coefficient,
///     mirroring the Linear zero-skip)
///   * Div          `x - c·y = 0` (exact; `c` must be finite and nonzero —
///     bail otherwise, matching DivConstantLayer's own divisor validation)
///
/// Output columns are free (`±inf`) unless the caller pinned this node's
/// DELTA-inflated box (same posture as every other affine node here).
fn encode_const_affine_node(
    problem: &mut MilpProblem,
    op: ConstAffineOp,
    xs: &[Col],
    cs: &[f64],
    node_box: Option<&[Bound]>,
    delta: f64,
    node_name: &str,
) -> Result<Vec<Col>> {
    debug_assert_eq!(xs.len(), cs.len());
    check_box_len(node_box, xs.len(), node_name)?;
    let mut out_cols = Vec::with_capacity(xs.len());
    for (i, (&x, &c)) in xs.iter().zip(cs.iter()).enumerate() {
        let (lo, hi) = out_col_bounds(node_box, i, delta);
        let y = problem.add_col(0.0, lo, hi);
        match op {
            ConstAffineOp::Add => {
                problem.add_row(-c, -c, [(x, 1.0), (y, -1.0)]);
            }
            ConstAffineOp::SubConst => {
                problem.add_row(c, c, [(x, 1.0), (y, -1.0)]);
            }
            ConstAffineOp::SubFromConst => {
                problem.add_row(c, c, [(x, 1.0), (y, 1.0)]);
            }
            ConstAffineOp::Mul => {
                if c != 0.0 {
                    problem.add_row(0.0, 0.0, [(x, c), (y, -1.0)]);
                } else {
                    // y = 0·x = 0 exactly; skip the zero coefficient.
                    problem.add_row(0.0, 0.0, [(y, -1.0)]);
                }
            }
            ConstAffineOp::Div => {
                if c == 0.0 || !c.is_finite() {
                    bail!(
                        "node '{node_name}': division by zero/non-finite constant {c} \
                         (fail closed)"
                    );
                }
                problem.add_row(0.0, 0.0, [(x, 1.0), (y, -c)]);
            }
        }
        out_cols.push(y);
    }
    Ok(out_cols)
}

// ── increment 5: ReduceSum ──────────────────────────────────────────────────

/// TwoSum-verified EXACT f64 summation: returns `Some(sum)` only when EVERY
/// accumulation step is exactly representable (zero TwoSum error term), i.e.
/// the returned f64 equals the REAL sum of the inputs. Any rounding → `None`
/// (callers fail closed). Used to propagate PINNED values through ReduceSum:
/// mscn's mask counts are sums of 0/1 values, which are always exact.
fn exact_f64_sum(vals: &[f64]) -> Option<f64> {
    let mut s = 0.0f64;
    for &v in vals {
        if !v.is_finite() {
            return None;
        }
        let t = s + v;
        // Knuth TwoSum: err is EXACTLY the rounding error of `s + v`.
        let vp = t - s;
        let sp = t - vp;
        let err = (s - sp) + (v - vp);
        if err != 0.0 || !t.is_finite() {
            return None;
        }
        s = t;
    }
    Some(s)
}

/// Encode a ReduceSum node as one EXACT ones-row per output element:
/// `sum(x_g) - y = 0` over the reduced-axes group `g` (mirrors
/// `ReduceSumLayer`'s axes/keepdims semantics: negative axes resolve `+ndim`,
/// empty axes reduce everything, duplicate axes are rejected — see
/// `reduction/mod.rs::resolve_reduction_axes`). `keepdims` only changes the
/// output SHAPE metadata (size-1 dims don't affect row-major flat order), so
/// the emitted rows are identical either way.
///
/// PINNED propagation: an output whose entire group is pinned AND whose f64
/// sum is verified exact ([`exact_f64_sum`]) becomes pinned itself — this is
/// what lets a downstream Div-by-mask-count encode exactly (module header
/// §increment 5). Aliased duplicate columns in a group (possible via Gather
/// repeats upstream) are coalesced into one coefficient (`count`).
#[allow(clippy::too_many_arguments)]
fn encode_reduce_sum_node(
    problem: &mut MilpProblem,
    rs: &ReduceSumLayer,
    in_cols: &[Col],
    in_shape: &[usize],
    declared_out: Option<&[usize]>,
    node_box: Option<&[Bound]>,
    delta: f64,
    node_name: &str,
    pinned: &mut HashMap<Col, f64>,
) -> Result<Vec<Col>> {
    let ndim = in_shape.len();
    let axes: Vec<usize> = if rs.axes.is_empty() {
        (0..ndim).collect()
    } else {
        rs.axes
            .iter()
            .map(|&a| resolve_axis_signed(a, ndim, node_name))
            .collect::<Result<_>>()?
    };
    let mut reduced = vec![false; ndim];
    for &ax in &axes {
        if reduced[ax] {
            bail!("ReduceSum node '{node_name}': duplicate reduction axis {ax} (fail closed)");
        }
        reduced[ax] = true;
    }

    // Output shape: reduced dims dropped (or kept as 1 with keepdims). Size-1
    // dims never affect row-major flat order, so only the KEPT dims drive the
    // out-flat index below.
    let mut out_shape: Vec<usize> = Vec::new();
    for d in 0..ndim {
        if reduced[d] {
            if rs.keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(in_shape[d]);
        }
    }
    let out_size = checked_product(&out_shape)
        .ok_or_else(|| anyhow!("ReduceSum node '{node_name}': output shape overflow"))?;
    if let Some(decl) = declared_out {
        let decl_prod = checked_product(decl)
            .ok_or_else(|| anyhow!("ReduceSum node '{node_name}': declared shape overflow"))?;
        if decl_prod != out_size {
            bail!(
                "ReduceSum node '{node_name}': computed output shape {out_shape:?} contradicts \
                 the declared shape {decl:?} (fail closed)"
            );
        }
    }
    check_box_len(node_box, out_size, node_name)?;

    // Group each input element under its output element (row-major over the
    // kept dims — reduced coordinates are simply dropped).
    let in_strides = strides_for(in_shape);
    let mut groups: Vec<Vec<Col>> = vec![Vec::new(); out_size];
    for (in_flat, &col) in in_cols.iter().enumerate() {
        let mut rem = in_flat;
        let mut out_flat = 0usize;
        for d in 0..ndim {
            let coord = rem / in_strides[d];
            rem %= in_strides[d];
            if !reduced[d] {
                out_flat = out_flat * in_shape[d] + coord;
            }
        }
        groups[out_flat].push(col);
    }

    let mut out_cols = Vec::with_capacity(out_size);
    for (i, group) in groups.iter().enumerate() {
        let (lo, hi) = out_col_bounds(node_box, i, delta);
        let y = problem.add_col(0.0, lo, hi);

        // Coalesce duplicate columns (aliasing upstream) into integer counts —
        // duplicate sparse entries in one row would be solver-hostile.
        let mut coeffs: Vec<(Col, f64)> = Vec::with_capacity(group.len() + 1);
        for &c in group {
            match coeffs.iter_mut().find(|(cc, _)| *cc == c) {
                Some(entry) => entry.1 += 1.0,
                None => coeffs.push((c, 1.0)),
            }
        }
        // Pinned propagation BEFORE pushing y (values follow group order; the
        // real sum is order-independent, and exactness of any order certifies
        // the exact value).
        let pinned_vals: Option<Vec<f64>> = group.iter().map(|c| pinned.get(c).copied()).collect();
        if let Some(vals) = pinned_vals {
            if let Some(sum) = exact_f64_sum(&vals) {
                pinned.insert(y, sum);
            }
        }

        coeffs.push((y, -1.0));
        // Exact equality: sum(x_g) - y = 0.
        problem.add_row(0.0, 0.0, coeffs);
        out_cols.push(y);
    }
    Ok(out_cols)
}

// ── increment 5: Sub / MulBinary / Div ──────────────────────────────────────

/// Encode an element-wise Sub node `out = A - B` as EXACT equality rows
/// `A_i - B_i - out_i = 0`. Equal-length (non-broadcasting) operands only,
/// mirroring the residual-Add posture; `A == B` per element degenerates to
/// `out_i = 0` (coefficients coalesced — no duplicate sparse entries).
fn encode_sub_node(
    problem: &mut MilpProblem,
    a_cols: &[Col],
    b_cols: &[Col],
    node_box: Option<&[Bound]>,
    delta: f64,
    node_name: &str,
) -> Result<Vec<Col>> {
    if a_cols.len() != b_cols.len() {
        bail!(
            "Sub node '{node_name}': parent column counts differ ({} vs {}); broadcasting Sub \
             is not supported (fail closed)",
            a_cols.len(),
            b_cols.len()
        );
    }
    check_box_len(node_box, a_cols.len(), node_name)?;
    let mut out_cols = Vec::with_capacity(a_cols.len());
    for (i, (&a, &b)) in a_cols.iter().zip(b_cols.iter()).enumerate() {
        let (lo, hi) = out_col_bounds(node_box, i, delta);
        let y = problem.add_col(0.0, lo, hi);
        if a == b {
            // A - A = 0 exactly.
            problem.add_row(0.0, 0.0, [(y, -1.0)]);
        } else {
            problem.add_row(0.0, 0.0, [(a, 1.0), (b, -1.0), (y, -1.0)]);
        }
        out_cols.push(y);
    }
    Ok(out_cols)
}

/// All-pinned lookup: `Some(values)` iff EVERY column of the operand is pinned
/// to a single value on the MIP feasible set (see the `pinned` map in
/// [`encode_graph_impl`]).
fn pinned_values(cols: &[Col], pinned: &HashMap<Col, f64>) -> Option<Vec<f64>> {
    cols.iter().map(|c| pinned.get(c).copied()).collect()
}

/// Encode a MulBinary node `out = A ⊙ B` (element-wise with NumPy
/// broadcasting, mirroring `MulBinaryLayer`/`elementwise` semantics).
///
/// Bilinear in general → fail-closed. EXACT affine encoding when one operand
/// is PINNED: if every column of (say) B is pinned to `v` on the feasible set,
/// then `out = v ⊙ A` holds at EVERY feasible point, so the rows
/// `v_i·A_i - out_i = 0` have exactly the same feasible set as the bilinear
/// original — an exact encoding, not a relaxation (module header §increment 5;
/// mscn's masks are Split slices of the instance-fixed input box).
#[allow(clippy::too_many_arguments)]
fn encode_mul_binary_node(
    problem: &mut MilpProblem,
    graph: &GraphNetwork,
    (a_cols, a_name): (&[Col], &str),
    (b_cols, b_name): (&[Col], &str),
    node_box: Option<&[Bound]>,
    delta: f64,
    node_name: &str,
    pinned: &HashMap<Col, f64>,
) -> Result<Vec<Col>> {
    let a_shape = operand_shape(graph, a_name, a_cols.len(), node_name)?;
    let b_shape = operand_shape(graph, b_name, b_cols.len(), node_name)?;
    let Some(out_shape) = broadcast_shapes(&a_shape, &b_shape) else {
        bail!(
            "MulBinary node '{node_name}': operand shapes {a_shape:?} × {b_shape:?} do not \
             broadcast (fail closed)"
        );
    };
    if let Some(decl) = graph.declared_shape(node_name) {
        let decl_prod = checked_product(decl)
            .ok_or_else(|| anyhow!("MulBinary node '{node_name}': declared shape overflow"))?;
        let out_prod = checked_product(&out_shape)
            .ok_or_else(|| anyhow!("MulBinary node '{node_name}': broadcast shape overflow"))?;
        if decl_prod != out_prod {
            bail!(
                "MulBinary node '{node_name}': broadcast output shape {out_shape:?} contradicts \
                 the declared shape {decl:?} (fail closed)"
            );
        }
    }
    let out_size = checked_product(&out_shape)
        .ok_or_else(|| anyhow!("MulBinary node '{node_name}': broadcast shape overflow"))?;
    check_box_len(node_box, out_size, node_name)?;

    // Pick the pinned operand (either side); fail closed when neither is.
    let (var_cols, var_shape, pin_vals, pin_shape) = if let Some(bv) = pinned_values(b_cols, pinned)
    {
        (a_cols, &a_shape, bv, &b_shape)
    } else if let Some(av) = pinned_values(a_cols, pinned) {
        (b_cols, &b_shape, av, &a_shape)
    } else {
        bail!(
            "MulBinary node '{node_name}': neither operand is pinned to a constant by the \
                 instance box — a bilinear Mul is not affine (fail closed)"
        );
    };

    let out_strides = strides_for(&out_shape);
    let mut out_cols = Vec::with_capacity(out_size);
    let mut multi = Vec::with_capacity(out_shape.len());
    for out_flat in 0..out_size {
        unflatten_index(out_flat, &out_strides, &mut multi);
        let x = var_cols[broadcast_src_flat(&multi, var_shape)];
        let c = pin_vals[broadcast_src_flat(&multi, pin_shape)];
        if !c.is_finite() {
            bail!("MulBinary node '{node_name}': non-finite pinned factor {c} (fail closed)");
        }
        let (lo, hi) = out_col_bounds(node_box, out_flat, delta);
        let y = problem.add_col(0.0, lo, hi);
        if c != 0.0 {
            // Exact: c·x - y = 0 (y = c·x at every feasible point).
            problem.add_row(0.0, 0.0, [(x, c), (y, -1.0)]);
        } else {
            // y = 0·x = 0 exactly; skip the zero coefficient.
            problem.add_row(0.0, 0.0, [(y, -1.0)]);
        }
        out_cols.push(y);
    }
    Ok(out_cols)
}

/// Encode a Div node `out = A / B` (element-wise with NumPy broadcasting,
/// mirroring `DivLayer` semantics).
///
/// Non-affine in general → fail-closed. EXACT affine encoding ONLY when the
/// DENOMINATOR is pinned: with `B ≡ β` (β finite, nonzero) at every feasible
/// point, `out = A / β` is encoded as `A_i - β·out_i = 0` — multiplying
/// through by β keeps the row exactly rational (NO `1/β` reciprocal rounding),
/// so the feasible set equals the original's exactly. A pinned NUMERATOR does
/// not help (`c / B` is still nonlinear in B) and also fails closed.
#[allow(clippy::too_many_arguments)]
fn encode_div_node(
    problem: &mut MilpProblem,
    graph: &GraphNetwork,
    (a_cols, a_name): (&[Col], &str),
    (b_cols, b_name): (&[Col], &str),
    node_box: Option<&[Bound]>,
    delta: f64,
    node_name: &str,
    pinned: &HashMap<Col, f64>,
) -> Result<Vec<Col>> {
    let a_shape = operand_shape(graph, a_name, a_cols.len(), node_name)?;
    let b_shape = operand_shape(graph, b_name, b_cols.len(), node_name)?;
    let Some(out_shape) = broadcast_shapes(&a_shape, &b_shape) else {
        bail!(
            "Div node '{node_name}': operand shapes {a_shape:?} / {b_shape:?} do not broadcast \
             (fail closed)"
        );
    };
    if let Some(decl) = graph.declared_shape(node_name) {
        let decl_prod = checked_product(decl)
            .ok_or_else(|| anyhow!("Div node '{node_name}': declared shape overflow"))?;
        let out_prod = checked_product(&out_shape)
            .ok_or_else(|| anyhow!("Div node '{node_name}': broadcast shape overflow"))?;
        if decl_prod != out_prod {
            bail!(
                "Div node '{node_name}': broadcast output shape {out_shape:?} contradicts the \
                 declared shape {decl:?} (fail closed)"
            );
        }
    }
    let out_size = checked_product(&out_shape)
        .ok_or_else(|| anyhow!("Div node '{node_name}': broadcast shape overflow"))?;
    check_box_len(node_box, out_size, node_name)?;

    let Some(betas) = pinned_values(b_cols, pinned) else {
        bail!(
            "Div node '{node_name}': denominator is not pinned to a constant by the instance \
             box — a variable division is not affine (fail closed)"
        );
    };

    let out_strides = strides_for(&out_shape);
    let mut out_cols = Vec::with_capacity(out_size);
    let mut multi = Vec::with_capacity(out_shape.len());
    for out_flat in 0..out_size {
        unflatten_index(out_flat, &out_strides, &mut multi);
        let a = a_cols[broadcast_src_flat(&multi, &a_shape)];
        let beta = betas[broadcast_src_flat(&multi, &b_shape)];
        if beta == 0.0 || !beta.is_finite() {
            bail!(
                "Div node '{node_name}': pinned denominator is {beta} — division undefined \
                 (fail closed)"
            );
        }
        let (lo, hi) = out_col_bounds(node_box, out_flat, delta);
        let y = problem.add_col(0.0, lo, hi);
        // Exact: A - β·y = 0 ⟺ y = A/β over the reals.
        problem.add_row(0.0, 0.0, [(a, 1.0), (y, -beta)]);
        out_cols.push(y);
    }
    Ok(out_cols)
}

// ── increment 5: final-Sigmoid threshold rewrite ────────────────────────────

/// A sigmoid-space output threshold transformed into logit space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LogitThreshold {
    /// The transformed finite bound on the logit `z` (already rounded OUTWARD
    /// so the z-space constraint is never stricter than the σ-space original).
    Bound(f64),
    /// The σ-space constraint holds for EVERY real logit (e.g. `σ(z) <= t`
    /// with `t >= 1`): drop the constraint.
    Vacuous,
    /// The σ-space constraint holds for NO real logit (e.g. `σ(z) <= t` with
    /// `t <= 0`, since `σ(z) ∈ (0, 1)` for all finite z): the constrained
    /// problem is infeasible without solving.
    Infeasible,
}

/// One-directional f64 ULP steps (outward rounding helpers).
fn step_up(x: f64, n: u32) -> f64 {
    let mut v = x;
    for _ in 0..n {
        v = v.next_up();
    }
    v
}

fn step_down(x: f64, n: u32) -> f64 {
    let mut v = x;
    for _ in 0..n {
        v = v.next_down();
    }
    v
}

/// Certified f64 enclosure `[lo, hi] ∋ logit(t) = ln(t / (1 - t))` for
/// `t ∈ (0, 1)`, built with directed rounding:
///   * `1 - t` and the division are IEEE correctly rounded (≤ 0.5 ulp), so one
///     ULP outward on each endpoint encloses the exact intermediate; the
///     quotient is evaluated at the OPPOSITE denominator endpoint (a/b is
///     monotone decreasing in b > 0);
///   * `ln` is monotone increasing but libm's `f64::ln` is only faithfully
///     rounded, not correctly rounded — glibc's documented worst-case known
///     error for `log` is ≤ 3 ulp, so 4 ULPs outward on each endpoint covers
///     it with margin. (Endpoint degeneracies stay sound: `ln(0) = -inf`,
///     `ln(+inf) = +inf` only ever WIDEN the enclosure.)
///
/// Rounding the enclosure outward at every step means `lo <= logit(t) <= hi`
/// even though no step is exact — which is exactly what the never-stricter
/// threshold contract needs.
fn logit_enclosure(t: f64) -> (f64, f64) {
    debug_assert!(t > 0.0 && t < 1.0);
    let om = 1.0 - t;
    let om_lo = om.next_down().max(0.0); // 1 - t > 0 for t < 1
    let om_hi = om.next_up();
    let r_lo = (t / om_hi).next_down().max(0.0); // t / (1-t) > 0 for t > 0
    let r_hi = (t / om_lo).next_up();
    (step_down(r_lo.ln(), 4), step_up(r_hi.ln(), 4))
}

/// Transform a σ-space UPPER threshold: `σ(z) <= t ⟺ z <= logit(t)` (σ is a
/// strictly increasing bijection onto (0, 1)).
///
/// DIRECTED ROUNDING: the returned bound is the UPPER end of the certified
/// [`logit_enclosure`], i.e. rounded UP (toward +inf). Rounding up can only
/// ADMIT extra z (weaker constraint), never exclude a z whose σ(z) <= t — so
/// an `unsat` of the transformed problem still certifies the original, while a
/// down-rounded bound could wrongly exclude boundary logits (a false-UNSAT
/// risk). Real-arithmetic edge cases are handled exactly: `t <= 0` is
/// infeasible (σ > 0), `t >= 1` is vacuous (σ < 1). NaN fails closed.
///
/// NOTE (float caveat, same posture as the rest of this encoder): the rewrite
/// is exact for the REAL-valued sigmoid; the f32-forward gap of the logit
/// itself is the caller's concern (the standard DELTA output-slack discipline
/// this encoder already documents for every output constraint).
pub(crate) fn logit_upper_threshold(t: f64) -> Result<LogitThreshold> {
    if t.is_nan() {
        bail!("sigmoid threshold is NaN (fail closed)");
    }
    if t <= 0.0 {
        return Ok(LogitThreshold::Infeasible);
    }
    if t >= 1.0 {
        return Ok(LogitThreshold::Vacuous);
    }
    Ok(LogitThreshold::Bound(logit_enclosure(t).1))
}

/// Transform a σ-space LOWER threshold: `σ(z) >= t ⟺ z >= logit(t)`.
///
/// DIRECTED ROUNDING: the returned bound is the LOWER end of the certified
/// [`logit_enclosure`], i.e. rounded DOWN (toward -inf) — for a `z >= s`
/// constraint the WEAKER (never-stricter) direction is a smaller `s`.
/// Real-arithmetic edges: `t <= 0` is vacuous (σ > 0 ≥ t), `t >= 1` is
/// infeasible (σ < 1). NaN fails closed.
pub(crate) fn logit_lower_threshold(t: f64) -> Result<LogitThreshold> {
    if t.is_nan() {
        bail!("sigmoid threshold is NaN (fail closed)");
    }
    if t <= 0.0 {
        return Ok(LogitThreshold::Vacuous);
    }
    if t >= 1.0 {
        return Ok(LogitThreshold::Infeasible);
    }
    Ok(LogitThreshold::Bound(logit_enclosure(t).0))
}

// ── increment 5b: the BaB→MIP escalation call site (nn4sys mscn) ───────────

/// Marker for a successful graph-MIP escalation: EVERY violation clause of the
/// spec was proven infeasible by the exact MILP backend WITH a verified
/// certificate, so the property holds (`unsat`).
///
/// UNSAT-ONLY BY CONSTRUCTION (V1 soundness posture): this is the only value
/// [`try_graph_mip_escalation`] can return and it cannot carry a
/// counterexample or output values — the caller is structurally unable to emit
/// `sat` from this path (falsification stays the attack's job).
#[derive(Debug)]
pub(crate) struct GraphMipCertifiedUnsat {
    /// `VnnLibSpec::num_outputs`, so the caller can shape the (unbounded)
    /// output box of the reported verdict.
    pub num_outputs: usize,
}

/// Outcome of transforming ONE output constraint into a MIP row (see
/// [`add_violation_constraint`]).
#[derive(Debug, PartialEq, Eq)]
enum ClauseConstraintOutcome {
    /// A (weakened-outward) linear row was added to the problem.
    Added,
    /// The constraint holds for EVERY output value (e.g. `σ(z) <= t` with
    /// `t >= 1`): dropped. Dropping only ENLARGES the violation system's
    /// feasible set, so a subsequent `infeasible` still certifies the clause.
    DroppedVacuous,
}

/// Directed f64→f32 conversion of one vnnlib input bound `[l, u]` for the
/// graph-MIP clause box.
///
/// * Degenerate (`l == u`, the mscn mask/pin case): pin to the nearest f32
///   `v = l as f32` — the box constrains f32 witness values, so if `l` is
///   f32-representable `{v}` is EXACT, and if it is not then NO f32 point lies
///   in `[l, l]` (the true feasible set is empty) and `{v}` is a superset of
///   the empty set — sound for proving infeasibility either way. Pinning (vs
///   outward widening) is what keeps the encoder's exact MulBinary/Div
///   pinned-operand encodings available.
/// * Non-degenerate: round each endpoint OUTWARD (lower toward −inf, upper
///   toward +inf, preserving exactly-representable endpoints), enclosing every
///   real point of `[l, u]` — the same posture as
///   `VnnLibSpec::split_input_bounds_f32`.
///
/// `None` = fail closed (NaN, inverted, or a degenerate bound outside f32
/// range).
fn clause_bound_to_f32(l: f64, u: f64) -> Option<Bound> {
    if l.is_nan() || u.is_nan() || l > u {
        return None;
    }
    if l == u {
        let v = l as f32;
        if !v.is_finite() {
            return None;
        }
        return Some(Bound::new(v, v));
    }
    let lf = {
        let v = l as f32;
        if v == f32::INFINITY {
            // l is finite but above f32 range: widen DOWN to the largest finite f32.
            f32::MAX
        } else if v == f32::NEG_INFINITY || (v as f64) <= l {
            v
        } else {
            ny_tensor::next_down_f32(v)
        }
    };
    let uf = {
        let v = u as f32;
        if v == f32::NEG_INFINITY {
            // u is finite but below f32 range: widen UP to the smallest finite f32.
            -f32::MAX
        } else if v == f32::INFINITY || (v as f64) >= u {
            v
        } else {
            ny_tensor::next_up_f32(v)
        }
    };
    if lf > uf {
        return None;
    }
    Some(Bound::new_allow_infinite(lf, uf))
}

/// Extra z-space slack absorbing the f32 sigmoid's EVALUATION error when a
/// σ-space threshold `t` is rewritten onto the logit (increment 5b).
///
/// The violation the organizer scores is on the f32 sigmoid OUTPUT:
/// `σ_f32(z) <= t` implies only `σ(z) <= t + ε` for the REAL sigmoid, where
/// `ε` bounds the f32 sigmoid's absolute evaluation error (`SIGMA_EVAL_ERR =
/// 1e-6`, ≥ 8 f32 ULPs at output scale ≤ 1 — comfortably above the ≤ ~2 ULP
/// error of any faithful implementation). Inverting through the logit
/// amplifies that band by `1/σ'`, and `σ'(z) = y(1−y) ≥ t(1−t) − ε` on the
/// band `[t−ε, t+ε]` (|d(y(1−y))/dy| ≤ 1). We return `2·ε / (t(1−t) − ε)` —
/// the factor 2 swallows all second-order slop. Thresholds within `T_GUARD`
/// of the sigmoid's range boundary fail closed (`Err`): there the f32 sigmoid
/// saturates (it CAN output exactly 0.0 / 1.0, which the real σ never does),
/// so no finite z-space rewrite is trustworthy.
fn sigmoid_quantization_slack(t: f64) -> Result<f64> {
    const SIGMA_EVAL_ERR: f64 = 1e-6;
    const T_GUARD: f64 = 1e-4;
    if !(T_GUARD..=1.0 - T_GUARD).contains(&t) {
        bail!(
            "σ-space threshold {t} is within {T_GUARD} of the sigmoid range boundary; the f32 \
             sigmoid saturates there, so the logit rewrite fails closed"
        );
    }
    let density = t * (1.0 - t) - SIGMA_EVAL_ERR;
    if density <= 0.0 {
        bail!("σ-space threshold {t}: non-positive certified density; fail closed");
    }
    Ok(2.0 * SIGMA_EVAL_ERR / density)
}

/// Add ONE vnnlib output constraint of a violation clause to the encoded MIP,
/// transformed for the (possibly sigmoid-peeled) output columns.
///
/// Accepted shapes — ONLY const bounds on a single output (the V1 contract;
/// anything else fails closed with `Err`):
///   * `Y_i <= t` / `Y_i < t`  → `out_col_i <= rhs` (strictness dropped —
///     `<=` only ENLARGES the violation system, sound for unsat);
///   * `Y_i >= t` / `Y_i > t`  → `out_col_i >= rhs`.
///
/// When the final Sigmoid was PEELED, `t` is rewritten into logit space via
/// [`logit_upper_threshold`] / [`logit_lower_threshold`]:
///   * `Bound(z_t)` — the row is added on the logit column with the OUTWARD
///     slack `DELTA + sigmoid_quantization_slack(t)` (the f32↔f64 forward gap
///     of the logit plus the f32 sigmoid's evaluation band, both in the
///     weakening direction);
///   * `Vacuous` — the constraint is dropped ([`ClauseConstraintOutcome::
///     DroppedVacuous`]; dropping only weakens the system → sound);
///   * `Infeasible` — FAIL CLOSED (`Err`). The REAL sigmoid never reaches the
///     threshold, but the f32 sigmoid saturates to exactly 0.0/1.0, so
///     declaring the clause impossible here would be a false-UNSAT risk.
///
/// Without a peel the constraint lands on the raw output column with the
/// standard `DELTA` output slack.
fn add_violation_constraint(
    enc: &mut GraphMipEncoding,
    constraint: &OutputConstraint,
    peeled: bool,
) -> Result<ClauseConstraintOutcome> {
    use OutputConstraint as OC;
    let (idx, t, is_upper) = match constraint {
        OC::LessEqConst(i, t) | OC::LessThanConst(i, t) => (*i, *t, true),
        OC::GreaterEqConst(i, t) | OC::GreaterThanConst(i, t) => (*i, *t, false),
        other => bail!(
            "graph-MIP supports only const bounds on a single output (V1); got {other:?} (fail \
             closed)"
        ),
    };
    let col = *enc.output_vars.get(idx).ok_or_else(|| {
        anyhow!(
            "output constraint references Y_{idx} but the encoding has {} output column(s)",
            enc.output_vars.len()
        )
    })?;
    let (rhs, slack) = if peeled {
        let transformed = if is_upper {
            logit_upper_threshold(t)?
        } else {
            logit_lower_threshold(t)?
        };
        match transformed {
            LogitThreshold::Vacuous => return Ok(ClauseConstraintOutcome::DroppedVacuous),
            LogitThreshold::Infeasible => bail!(
                "σ-space threshold {t} is infeasible for the REAL sigmoid, but the f32 sigmoid \
                 can saturate to exactly 0.0/1.0 — declaring the clause impossible would risk a \
                 false unsat (fail closed)"
            ),
            LogitThreshold::Bound(z_t) => (z_t, DELTA + sigmoid_quantization_slack(t)?),
        }
    } else {
        (t, DELTA)
    };
    if !rhs.is_finite() || !slack.is_finite() {
        bail!("non-finite transformed threshold (rhs {rhs}, slack {slack}); fail closed");
    }
    if is_upper {
        enc.problem
            .add_row(f64::NEG_INFINITY, rhs + slack, [(col, 1.0)]);
    } else {
        enc.problem
            .add_row(rhs - slack, f64::INFINITY, [(col, 1.0)]);
    }
    Ok(ClauseConstraintOutcome::Added)
}

/// Per-node boxes for the encoder from a graph IBP pass over `tensor`.
///
/// ⚠ THE RELU-NAME TRAP (mirrors `ibp_node_boxes` in graph_mip_tests): the
/// encoder reads a ReLU-named entry as that ReLU's PRE-activation box, but the
/// IBP map stores POST-activation OUTPUT bounds under ReLU names — so ReLU
/// entries are REMOVED here and the encoder falls back to the ReLU's input
/// node's (true pre-activation) box. Entries containing NaN are dropped too
/// (the encoder then uses free ±inf columns — looser, still sound); the
/// `_input` sentinel is skipped (the encoder receives the input box
/// explicitly).
fn ibp_boxes_for_encoder(
    graph: &GraphNetwork,
    node_bt: &HashMap<String, ny_tensor::BoundedTensor>,
) -> HashMap<String, Vec<Bound>> {
    let mut node_bounds: HashMap<String, Vec<Bound>> = HashMap::new();
    for (name, bt) in node_bt {
        let Some(node) = graph.node(name) else {
            continue; // the `_input` sentinel or an unknown key
        };
        if matches!(node.layer(), Layer::ReLU(_)) {
            continue; // see doc comment: OUTPUT box ≠ the encoder's pre-activation key
        }
        let lower = bt.lower();
        let upper = bt.upper();
        let mut bounds = Vec::with_capacity(lower.len());
        let mut has_nan = false;
        for (&l, &u) in lower.iter().zip(upper.iter()) {
            if l.is_nan() || u.is_nan() {
                has_nan = true;
                break;
            }
            bounds.push(Bound::new_allow_infinite(l, u));
        }
        if !has_nan {
            node_bounds.insert(name.clone(), bounds);
        }
    }
    node_bounds
}

/// inc5d — CROWN-IBP tightening of the escalation's per-clause node boxes.
///
/// Runs ny's deadline-capped CROWN-IBP collector (the SAME sound bound source
/// the BaB precheck trusts: `shared/init.rs` bootstraps graph BaB from
/// `collect_crown_ibp_bounds_dag_*`) over the clause box, seeded with the
/// already-computed plain-IBP map so no budget is wasted re-running the IBP
/// forward pass. Tighter boxes shrink the ReLU big-M constants, which is what
/// lets ay prove clause infeasibility at the ROOT LP (verified Farkas witness
/// → `Unsat { certified: true }`) instead of via uncertified case-splits.
///
/// SOUNDNESS: this function NEVER synthesizes or tightens a bound itself — it
/// only relays the collector's output. The collector intersects each node's
/// CROWN bound with its IBP bound (sound enclosure by construction) and falls
/// back to that node's IBP bound on any per-node failure or deadline; nodes
/// the deadline starves keep plain IBP. Fail-closed posture on top:
///   * collector `Err` ⇒ the untouched IBP map is returned (never absent
///     bounds);
///   * any returned entry containing NaN or an inverted pair (`l > u`) is
///     REJECTED and that node keeps its IBP entry (a malformed box handed to
///     the encoder could exclude the true trajectory — the false-UNSAT
///     mechanism — or panic in `Bound::new_allow_infinite`).
///
/// Returns the (per-node) bounds map plus `Some(n)` = collection succeeded
/// with `n` nodes CROWN-tightened, or `None` = collector error, IBP map
/// returned unchanged.
fn crown_tightened_node_bounds(
    graph: &GraphNetwork,
    tensor: &ny_tensor::BoundedTensor,
    ibp: HashMap<String, ny_tensor::BoundedTensor>,
    deadline: Instant,
) -> (HashMap<String, ny_tensor::BoundedTensor>, Option<usize>) {
    let result = match graph.collect_crown_ibp_bounds_dag_with_precomputed_ibp(
        tensor,
        ibp.clone(),
        Some(deadline),
    ) {
        Ok(result) => result,
        Err(e) => {
            info!(
                "graph-MIP escalation: CROWN-IBP tightening failed ({e}); keeping the plain IBP \
                 boxes (sound, looser)"
            );
            return (ibp, None);
        }
    };
    let crown_nodes = result
        .provenance
        .values()
        .filter(|p| matches!(p, BoundsProvenance::Crown))
        .count();
    // Attribution (inc5d measurement): which fallback reasons kept nodes on
    // plain IBP — this is what decides whether the big-M shrink can make a
    // clause root-LP-certifiable. Info-level so probe runs can read it.
    if !result.fallback_events.is_empty() {
        let mut by_reason: HashMap<String, usize> = HashMap::new();
        for event in &result.fallback_events {
            *by_reason.entry(format!("{:?}", event.reason)).or_insert(0) += 1;
        }
        let mut parts: Vec<String> = by_reason
            .into_iter()
            .map(|(reason, n)| format!("{reason} x{n}"))
            .collect();
        parts.sort();
        info!(
            "graph-MIP escalation: CROWN-IBP tightened {crown_nodes} node(s); IBP fallbacks: {}",
            parts.join(", ")
        );
    }
    // Overlay the collector's map onto the IBP map so every IBP-covered node
    // keeps a bound even if the collector omitted it, rejecting malformed
    // entries per node (see doc comment). Rejection is a pure WEAKENING back
    // to the plain-IBP enclosure — never a synthesized bound.
    let mut merged = ibp;
    let mut rejected = 0usize;
    for (name, bt) in result.bounds {
        let well_formed = bt
            .lower()
            .iter()
            .zip(bt.upper().iter())
            .all(|(&l, &u)| !l.is_nan() && !u.is_nan() && l <= u);
        if well_formed {
            merged.insert(name, bt);
        } else {
            rejected += 1;
        }
    }
    if rejected > 0 {
        info!(
            "graph-MIP escalation: CROWN-IBP tightening returned {rejected} malformed node \
             bound(s) (NaN or inverted); those nodes keep their plain IBP boxes"
        );
    }
    (merged, Some(crown_nodes))
}

/// inc5e — the node whose columns the encoding exposes as `output_vars`: the
/// peeled logit when the graph's output is a single-input final Sigmoid
/// (mirroring `encode_graph_impl`'s peel test EXACTLY — same `Layer::Sigmoid`
/// match, same `require_unary_input`), else the graph output node itself.
///
/// `None` = the output shape is not decidable here (missing node / non-unary
/// Sigmoid); the encoder will fail closed on the same shape, so the α pass is
/// simply skipped.
fn alpha_output_target(graph: &GraphNetwork) -> Option<String> {
    let out = graph.node(graph.output_name())?;
    if matches!(out.layer(), Layer::Sigmoid(_)) {
        out.require_unary_input().ok().map(str::to_string)
    } else {
        Some(graph.output_name().to_string())
    }
}

/// inc5e — α-CROWN-OPTIMIZED sound enclosure of the encoding's output node
/// (the peeled logit) over the clause's input box.
///
/// `alpha_graph` is a private clone of the escalation's graph retargeted with
/// `set_output` to [`alpha_output_target`], so the MAIN verification path's
/// deadline-capped DAG α-CROWN entry
/// (`GraphNetwork::propagate_alpha_crown_with_config_and_engine`) optimizes
/// and returns the bound of exactly the node whose columns are
/// `output_vars`.
///
/// SOUNDNESS: this function NEVER synthesizes or tightens a bound itself — it
/// only relays the collector's output. That collector is sound by
/// construction: α parameters only STEER the CROWN relaxation (any α ∈ [0,1]
/// yields a valid relaxation), every iterate is a sound CROWN enclosure of
/// the target node's reachable set, and the machinery keeps the
/// elementwise-best across iterates with NaN/inversion fallbacks to the plain
/// CROWN bound — the same contract the escalation already trusts for its
/// CROWN-IBP boxes. Fail-open posture on top (`None` ⇒ the caller keeps
/// today's output bound row, sound and looser):
///   * collector `Err` (unsupported op / deadline / memory cap / anything);
///   * a returned entry containing NaN or an inverted pair (`l > u`).
fn alpha_output_bound(
    alpha_graph: &GraphNetwork,
    tensor: &ny_tensor::BoundedTensor,
    deadline: Instant,
    clause_idx: usize,
) -> Option<ny_tensor::BoundedTensor> {
    let config = AlphaCrownConfig {
        deadline: Some(deadline),
        // CROWN-IBP (not plain-IBP) intermediates: the escalation's whole
        // premise is that IBP boxes are too loose here, and the α pass must
        // START from at least the CROWN-IBP quality the encoder's own boxes
        // have (measured: with IBP intermediates the α logit bound came back
        // LOOSER than the encoder's CROWN-IBP row and tightened nothing).
        // Sound: a pure reference-bounds quality choice inside the collector.
        fix_interm_bounds: false,
        // SPSA gradients: the default AnalyticChain cannot build its A-matrix
        // chain through the mscn Concat/MulBinary/Div topology (#1937 "missing
        // both A matrix and pre-ReLU bounds" for every ReLU) so α never moves;
        // SPSA only needs bound evaluations. Gradient choice is
        // non-soundness-critical — any α ∈ [0,1] yields a valid relaxation,
        // gradients only steer.
        gradient_method: ny_propagate::GradientMethod::Spsa,
        ..AlphaCrownConfig::default()
    };
    let started = Instant::now();
    let bound =
        match alpha_graph.propagate_alpha_crown_with_config_and_engine(tensor, &config, None) {
            Ok(bound) => bound,
            Err(e) => {
                info!(
                "graph-MIP escalation: clause {} α-CROWN output bound failed ({e}); keeping the \
                 existing output bound row (sound, looser)",
                clause_idx + 1
            );
                return None;
            }
        };
    let well_formed = bound
        .lower()
        .iter()
        .zip(bound.upper().iter())
        .all(|(&l, &u)| !l.is_nan() && !u.is_nan() && l <= u);
    if !well_formed {
        info!(
            "graph-MIP escalation: clause {} α-CROWN output bound is malformed (NaN or inverted); \
             keeping the existing output bound row",
            clause_idx + 1
        );
        return None;
    }
    info!(
        "graph-MIP escalation: clause {} α-CROWN output bound {:?} for node '{}' in {:.2}s",
        clause_idx + 1,
        bound
            .lower()
            .iter()
            .zip(bound.upper().iter())
            .map(|(&l, &u)| format!("[{l:.6}, {u:.6}]"))
            .collect::<Vec<_>>(),
        alpha_graph.output_name(),
        started.elapsed().as_secs_f64()
    );
    Some(bound)
}

/// inc5e — the α output rows actually applied to a clause encoding.
struct AlphaOutputRows {
    /// Effective per-output-column `[lo, hi]` — the α row intersected with the
    /// existing column bound — parallel to `output_vars`. Used by the
    /// refutability diagnostic; equals the column bound where α added nothing.
    effective: Vec<(f64, f64)>,
    /// Number of bound rows actually appended (only strictly-tightening rows
    /// are materialized).
    rows_added: usize,
}

/// inc5e — intersect the α-CROWN output enclosure into the encoding as
/// explicit single-variable bound rows on the output column(s).
///
/// SOUNDNESS (the anti-false-UNSAT contract of this whole increment):
///   * the α value is only RELAYED (never tightened) and gets the SAME
///     `±DELTA` outward inflation as every other collector box that becomes a
///     column bound (`out_col_bounds`) — it must absorb the identical f32-net
///     vs f64-affine semantic gap because the row constrains the MIP's exact
///     f64-affine logit — plus one f64 ULP outward for the inflation
///     arithmetic's own rounding (`step_down`/`step_up`);
///   * the row is INTERSECTED with (never replaces) the existing column
///     bound: `new_lo = max(col.lb, α_lo⁻)`, `new_hi = min(col.ub, α_hi⁺)` —
///     intersecting two sound enclosures is sound;
///   * an EMPTY intersection means two allegedly-sound enclosures are
///     disjoint — evidence of a wrong bound or wrong node mapping upstream —
///     so NOTHING is added for the whole encoding (fail-open, `None`);
///   * a length mismatch between the α tensor and `output_vars` is likewise
///     uninterpretable → `None`.
///
/// Rows that would not strictly tighten the column bound are skipped (adding
/// a redundant row changes nothing but solver noise).
fn add_alpha_output_rows(
    enc: &mut GraphMipEncoding,
    alpha: &ny_tensor::BoundedTensor,
    clause_idx: usize,
) -> Option<AlphaOutputRows> {
    if alpha.len() != enc.output_vars.len() {
        info!(
            "graph-MIP escalation: clause {} α output bound has {} elements but the encoding has \
             {} output columns; skipping the α row (fail-open)",
            clause_idx + 1,
            alpha.len(),
            enc.output_vars.len()
        );
        return None;
    }
    // First pass: compute every intersected row and reject the WHOLE batch on
    // any empty intersection (see doc comment).
    let mut rows: Vec<(Col, f64, f64)> = Vec::with_capacity(enc.output_vars.len());
    for (i, (&l, &u)) in alpha.lower().iter().zip(alpha.upper().iter()).enumerate() {
        let col = enc.output_vars[i];
        let spec = &enc.problem.cols()[col.0];
        // The identical DELTA discipline as `out_col_bounds`, then one ULP
        // outward for the f64 add/sub rounding itself.
        let alpha_lo = step_down((l as f64) - DELTA, 1);
        let alpha_hi = step_up((u as f64) + DELTA, 1);
        let new_lo = spec.lb.max(alpha_lo);
        let new_hi = spec.ub.min(alpha_hi);
        if new_lo > new_hi {
            info!(
                "graph-MIP escalation: clause {} α output bound [{alpha_lo:.6}, {alpha_hi:.6}] is \
                 DISJOINT from the existing column bound [{:.6}, {:.6}] on output col {i} — \
                 upstream bound or node mapping is suspect; adding NO α rows (fail-open)",
                clause_idx + 1,
                spec.lb,
                spec.ub
            );
            return None;
        }
        rows.push((col, new_lo, new_hi));
    }
    let mut effective = Vec::with_capacity(rows.len());
    let mut rows_added = 0usize;
    for (col, lo, hi) in rows {
        let spec = &enc.problem.cols()[col.0];
        if lo > spec.lb || hi < spec.ub {
            enc.problem.add_row(lo, hi, [(col, 1.0)]);
            rows_added += 1;
        }
        effective.push((lo, hi));
    }
    Some(AlphaOutputRows {
        effective,
        rows_added,
    })
}

/// The graph-MIP escalation (task #38 increment 5b): decide an inconclusive
/// BaB verdict as `unsat` for graph-only models (nn4sys mscn) by proving EVERY
/// violation clause infeasible with the exact DAG-aware MILP encoding.
///
/// Called by `dispatch::run_bab_with_fallback` from the arm where the
/// sequential MIP reload failed ("model has no sequential form"). The caller
/// has already established: `#[cfg(feature = "mip")]`, policy allows
/// escalation (`CompleteVerifierArg::Mip | Auto`), BaB was inconclusive, and
/// `mip_timeout_secs >= 5`; the gates are still re-checked here (defense in
/// depth).
///
/// Returns `Some(GraphMipCertifiedUnsat)` ONLY when every clause's violation
/// system (per-clause input box ∧ exact network encoding ∧ outward-weakened
/// violation constraints) is `Unsat { certified: true }` — i.e. the backend's
/// verdict carries an independently VERIFIED exact certificate (Farkas /
/// case-split at the LG3 seam). EVERY other outcome — feasible, timeout,
/// uncertified unsat, solver error, encoding failure, unsupported spec shape,
/// budget exhausted — returns `None` and the caller keeps the (sound,
/// inconclusive) BaB verdict. All decisions are `info!`-logged for
/// diagnosability.
pub(crate) fn try_graph_mip_escalation(
    model_path: &Path,
    onnx_load_config: &OnnxLoadConfig,
    input_shape: &[usize],
    vnnlib: Option<&VnnLibSpec>,
    backend: MipBackend,
    mip_timeout_secs: u64,
) -> Option<GraphMipCertifiedUnsat> {
    // Gate: default-on, with an exact-zero kill switch. This legacy entry is
    // retained for tests/reference; production dispatch uses the stricter
    // planner in `graph_mip_escalate` and never falls through to this encoder.
    if !graph_mip_enabled() {
        return None;
    }
    // Gate: minimum budget, mirroring the sequential escalation arm.
    if mip_timeout_secs < 5 {
        info!("graph-MIP escalation: budget {mip_timeout_secs}s < 5s; not escalating");
        return None;
    }
    // Gate: a vnnlib spec is required — the violation clauses ARE the property.
    let Some(spec) = vnnlib else {
        info!("graph-MIP escalation: no vnnlib spec available; fail closed");
        return None;
    };
    let flat_inputs: usize = input_shape.iter().product();
    if spec.input_bounds.len() != flat_inputs || spec.input_bounds.is_empty() {
        info!(
            "graph-MIP escalation: vnnlib has {} input bounds but the model input has {} \
             elements; fail closed",
            spec.input_bounds.len(),
            flat_inputs
        );
        return None;
    }

    // Violation clauses. `output_constraint_clauses` when present (each clause
    // is one conjunctive violation system; requiring EVERY clause infeasible is
    // exact for a disjunction and conservative — never unsound — for a
    // conjunction), else the flat conjunctive `output_constraints` as a single
    // clause (mirrors mip_highs::conjunctive_constraints_owned).
    let clauses: Vec<&[OutputConstraint]> = if !spec.output_constraint_clauses.is_empty() {
        spec.output_constraint_clauses
            .iter()
            .map(Vec::as_slice)
            .collect()
    } else if !spec.output_constraints.is_empty() {
        vec![spec.output_constraints.as_slice()]
    } else {
        info!("graph-MIP escalation: spec has no output constraints; fail closed");
        return None;
    };
    // Per-clause input boxes are parallel to `output_constraint_clauses`; any
    // other shape is uninterpretable → fail closed.
    if !spec.per_clause_input_bounds.is_empty()
        && spec.per_clause_input_bounds.len() != clauses.len()
    {
        info!(
            "graph-MIP escalation: {} per-clause input boxes vs {} clauses; fail closed",
            spec.per_clause_input_bounds.len(),
            clauses.len()
        );
        return None;
    }

    // Reload the model as a GraphNetwork — same reload posture as the
    // sequential arm (fresh load; BaB may have mutated the in-memory model)
    // and the same graph options as the main load path (model_load.rs).
    let overall_start = Instant::now();
    let overall_budget = Duration::from_secs(mip_timeout_secs);
    let onnx = match load_onnx_with_config(model_path, onnx_load_config) {
        Ok(model) => model,
        Err(e) => {
            info!("graph-MIP escalation: model reload failed ({e}); fail closed");
            return None;
        }
    };
    let graph_options = GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    };
    let mut graph = match onnx.to_graph_network_with_options(graph_options) {
        Ok(graph) => graph,
        Err(e) => {
            info!("graph-MIP escalation: graph conversion failed ({e}); fail closed");
            return None;
        }
    };
    // inc5d — per-node budget floor for the CROWN-IBP collection (see the
    // constant's doc; a pure time-vs-tightness policy on this private reload,
    // sound for any value).
    graph.set_crown_ibp_per_node_time_budget(ny_propagate::types::CrownIbpPerNodeTimeBudget {
        floor_secs: Some(GRAPH_MIP_CROWN_PER_NODE_FLOOR_SECS),
        cap_secs: None,
    });
    // inc5e — ONE private clone retargeted (`set_output`) to the node whose
    // columns are the encoding's `output_vars` (the peeled logit for a final
    // Sigmoid), shared by every clause's α-CROWN output-bound pass. Cloned
    // AFTER the budget override so the α pass inherits the same per-node
    // CROWN-IBP policy; the encoder keeps walking the UNTOUCHED `graph`.
    let alpha_graph: Option<GraphNetwork> = alpha_output_target(&graph).map(|target| {
        let mut retargeted = graph.clone();
        retargeted.set_output(target);
        retargeted
    });
    let n_clauses = clauses.len();
    info!(
        "graph-MIP escalation: {n_clauses} violation clause(s), {mip_timeout_secs}s budget, \
         {backend:?} backend (UNSAT-only; any non-certified-infeasible clause falls back to the \
         BaB verdict)"
    );

    for (clause_idx, clause) in clauses.iter().enumerate() {
        // Adaptive per-clause slice: remaining budget / remaining clauses
        // (mirrors mip_highs::solve_disjunctive round 1; no retry rounds —
        // ANY undecided clause already dooms the escalation, so spending the
        // tail budget on retries cannot change the outcome).
        let elapsed = overall_start.elapsed();
        if elapsed >= overall_budget {
            info!(
                "graph-MIP escalation: budget exhausted at clause {}/{n_clauses}; fail closed",
                clause_idx + 1
            );
            return None;
        }
        let clause_slice = overall_budget
            .checked_sub(elapsed)
            .expect("guarded above: elapsed < overall_budget")
            .as_secs_f64()
            / (n_clauses - clause_idx) as f64;
        let clause_start = Instant::now();
        let clause_deadline = clause_start + Duration::from_secs_f64(clause_slice);

        // Per-clause input box: the global vnnlib box overridden by this
        // clause's own bounds (mscn clauses pin most inputs), converted f64→f32
        // with the pin-preserving directed rounding of `clause_bound_to_f32`.
        let mut box64 = spec.input_bounds.clone();
        if let Some(overrides) = spec.per_clause_input_bounds.get(clause_idx) {
            for (&idx, &(l, u)) in overrides {
                if idx >= box64.len() {
                    info!(
                        "graph-MIP escalation: clause {} overrides X_{idx} beyond the input \
                         dimension; fail closed",
                        clause_idx + 1
                    );
                    return None;
                }
                box64[idx] = (l, u);
            }
        }
        let clause_box: Option<Vec<Bound>> = box64
            .iter()
            .map(|&(l, u)| clause_bound_to_f32(l, u))
            .collect();
        let Some(clause_box) = clause_box else {
            info!(
                "graph-MIP escalation: clause {} has an unusable input bound (NaN/inverted/out \
                 of f32 range); fail closed",
                clause_idx + 1
            );
            return None;
        };
        let tensor = match Verifier::bounds_to_tensor(&clause_box, Some(input_shape)) {
            Ok(tensor) => tensor,
            Err(e) => {
                info!(
                    "graph-MIP escalation: clause {} box→tensor failed ({e}); fail closed",
                    clause_idx + 1
                );
                return None;
            }
        };

        // Sound per-node IBP boxes over THIS clause's box (deadline-capped).
        let node_bt = match graph.collect_node_bounds_with_engine_and_deadline(
            &tensor,
            None,
            Some(clause_deadline),
        ) {
            Ok(map) => map,
            Err(e) => {
                info!(
                    "graph-MIP escalation: clause {} node-bound IBP failed ({e}); fail closed",
                    clause_idx + 1
                );
                return None;
            }
        };
        // inc5d — CROWN-IBP tightening of the per-node boxes when this
        // clause's slice can afford it. IBP boxes are LOOSE ⇒ loose big-M ⇒ a
        // weak LP relaxation ⇒ ay must branch ⇒ `certified: false` ⇒ the
        // escalation fail-closes (measured inc5c: a folded mscn clause decided
        // only via ~250s case-split, uncertified). CROWN∩IBP boxes shrink the
        // big-M so the ROOT LP can already be infeasible, which is the only
        // outcome that carries a verified Farkas certificate. Sound: the
        // helper only relays the collector's enclosures and falls back to the
        // plain IBP map per node / wholesale on any failure.
        let (node_bt, crown_nodes) = if clause_slice >= GRAPH_MIP_CROWN_MIN_SLICE_SECS {
            let crown_deadline = (Instant::now()
                + Duration::from_secs_f64(clause_slice * GRAPH_MIP_CROWN_SLICE_FRACTION))
            .min(clause_deadline);
            crown_tightened_node_bounds(&graph, &tensor, node_bt, crown_deadline)
        } else {
            (node_bt, None)
        };
        let mut bounds_source = match crown_nodes {
            Some(n) => format!("crown-ibp, {n} node(s) tightened"),
            None => "ibp".to_string(),
        };
        let node_bounds = ibp_boxes_for_encoder(&graph, &node_bt);

        // inc5e — α-CROWN-optimized sound bound of the OUTPUT node (the
        // peeled logit) over THIS clause's box, when the slice affords it
        // (mirrors the inc5d CROWN gate). Measured motivation: the CROWN-IBP
        // output row can straddle the violation threshold (mscn_128d clause 1:
        // [0.9301, 1.1404] vs 0.9479) forcing ay into an uncertified
        // case-split, while α-CROWN bounds the same logit past the threshold —
        // making the clause ROOT-LP infeasible (verified Farkas certificate)
        // in milliseconds. Sound: `alpha_output_bound` only relays the main
        // path's α-CROWN enclosure and fails open to `None`.
        let alpha_bound = match &alpha_graph {
            Some(alpha_graph) if clause_slice >= GRAPH_MIP_ALPHA_MIN_SLICE_SECS => {
                let alpha_deadline = (Instant::now()
                    + Duration::from_secs_f64(
                        (clause_slice * GRAPH_MIP_ALPHA_SLICE_FRACTION)
                            .min(GRAPH_MIP_ALPHA_SLICE_CAP_SECS),
                    ))
                .min(clause_deadline);
                alpha_output_bound(alpha_graph, &tensor, alpha_deadline, clause_idx)
            }
            _ => None,
        };

        // Exact DAG encoding (peeling a final Sigmoid when present).
        let (mut enc, peeled) =
            match encode_graph_peel_final_sigmoid(&graph, &clause_box, &node_bounds) {
                Ok(pair) => pair,
                Err(e) => {
                    info!(
                        "graph-MIP escalation: clause {} encoding failed ({e:#}); fail closed",
                        clause_idx + 1
                    );
                    return None;
                }
            };

        // inc5e — apply the α output rows, but ONLY when the encoder's actual
        // output choice matches the node the α pass bounded: `peeled` must
        // agree with the retargeted clone's output differing from the graph
        // output. Both derive from the same deterministic peel test, so a
        // mismatch is structurally impossible — this is defense in depth
        // against the two ever diverging (an α bound applied to the WRONG
        // node's columns is the false-UNSAT mechanism).
        let alpha_rows = match (&alpha_bound, &alpha_graph) {
            (Some(bound), Some(alpha_graph)) => {
                let target_is_graph_output = alpha_graph.output_name() == graph.output_name();
                if peeled == target_is_graph_output {
                    info!(
                        "graph-MIP escalation: clause {} α target '{}' does not match the \
                         encoder's output choice (peeled: {peeled}); skipping the α row \
                         (fail-open)",
                        clause_idx + 1,
                        alpha_graph.output_name()
                    );
                    None
                } else {
                    add_alpha_output_rows(&mut enc, bound, clause_idx)
                }
            }
            _ => None,
        };
        if let Some(rows) = &alpha_rows {
            if rows.rows_added > 0 {
                bounds_source.push_str(", α output row");
            }
        }

        // The clause's violation constraints (outward-weakened).
        let mut rows_added = 0usize;
        for constraint in clause.iter() {
            match add_violation_constraint(&mut enc, constraint, peeled) {
                Ok(ClauseConstraintOutcome::Added) => rows_added += 1,
                Ok(ClauseConstraintOutcome::DroppedVacuous) => {}
                Err(e) => {
                    info!(
                        "graph-MIP escalation: clause {} constraint rejected ({e:#}); fail closed",
                        clause_idx + 1
                    );
                    return None;
                }
            }
        }
        if rows_added == 0 {
            // Every constraint was vacuous: the violation system is just the
            // box ∧ network, which is trivially feasible — nothing to certify.
            info!(
                "graph-MIP escalation: clause {} has only vacuous output constraints (violation \
                 trivially reachable); fail closed",
                clause_idx + 1
            );
            return None;
        }
        if clause_idx == 0 {
            // One-time encoding-size diagnostic (the clause encodings only
            // differ in bounds, not in shape).
            info!(
                "graph-MIP escalation: clause encoding has {} cols, {} rows, {} ReLU binaries \
                 (peeled sigmoid: {peeled})",
                enc.problem.num_cols(),
                enc.problem.num_rows(),
                enc.binary_vars.len()
            );
        }
        // inc5d root-LP-gap attribution (diagnostic only, no verdict effect):
        // the output column's bound is the bound-collector-quality enclosure of
        // the (peeled) logit under this clause's box; the violation rows just
        // added constrain the same column(s). When the column bound alone
        // already contradicts a violation row, the root LP is trivially
        // infeasible; otherwise this logs how far the collector's bound is
        // from single-handedly certifying the clause.
        for (i, &col) in enc.output_vars.iter().enumerate() {
            let spec = &enc.problem.cols()[col.0];
            let viol: Vec<(f64, f64)> = enc.problem.rows()[enc.problem.num_rows() - rows_added..]
                .iter()
                .filter(|row| row.coeffs.len() == 1 && row.coeffs[0].0 == col.0)
                .map(|row| (row.lb, row.ub))
                .collect();
            let viol_str = viol
                .iter()
                .map(|(l, u)| format!("[{l:.4}, {u:.4}]"))
                .collect::<Vec<_>>()
                .join(" ");
            match alpha_rows.as_ref().and_then(|rows| rows.effective.get(i)) {
                Some(&(eff_lo, eff_hi)) => {
                    // inc5e — bound-row-refutable: some violation row's band is
                    // DISJOINT from the α-tightened column interval, so the
                    // root LP is trivially infeasible (the outcome that carries
                    // the verified Farkas certificate).
                    let refutable = viol.iter().any(|&(rl, ru)| eff_lo > ru || eff_hi < rl);
                    info!(
                        "graph-MIP escalation: clause {} output col bound [{:.4}, {:.4}] + α row \
                         → [{eff_lo:.4}, {eff_hi:.4}] vs violation row(s) {viol_str} — \
                         bound-row-refutable: {refutable} (bounds: {bounds_source})",
                        clause_idx + 1,
                        spec.lb,
                        spec.ub
                    );
                }
                None => {
                    info!(
                        "graph-MIP escalation: clause {} output col bound [{:.4}, {:.4}] vs \
                         violation row(s) {viol_str} (bounds: {bounds_source})",
                        clause_idx + 1,
                        spec.lb,
                        spec.ub
                    );
                }
            }
        }

        // Increment 5c — FOLD provably-pinned columns into the row constants
        // (graph_mip_fold.rs: exact rational substitution, outward-only
        // rounding, fail-closed on anything inexact). The mscn clause boxes
        // pin ~95% of the inputs and pinnedness propagates through the exact
        // affine rows, so this shrinks the ~10K-column LP by 1-2 orders of
        // magnitude before ay sees it. `None` = the folder proved a constant
        // row violated → fail closed for the WHOLE escalation: the folder's
        // infeasibility bypasses ay's verified-certificate gate (LG3), so it
        // is never allowed to become a verdict.
        let enc = super::graph_mip_fold::fold_for_escalation(enc, clause_idx)?;

        // Solve. `Unsat { certified: true }` is the ONLY acceptable outcome.
        let num_cols = enc.problem.num_cols();
        let GraphMipEncoding {
            problem,
            input_vars,
            output_vars,
            binary_vars,
            binary_widths,
            binary_keys: _,
            node_cols: _,
        } = enc;
        let parts = MipParts {
            problem,
            input_vars,
            output_vars,
            binary_vars,
            binary_widths,
            num_cols,
        };
        let solver_secs = clause_deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64();
        if solver_secs <= 0.05 {
            info!(
                "graph-MIP escalation: clause {}/{n_clauses} has no solver budget left after \
                 IBP+encode; fail closed",
                clause_idx + 1
            );
            return None;
        }
        // inc5d diagnostic (default-off, measurement only): `NY_GRAPH_MIP_SERIAL=1`
        // disables ny's 2^k phase-split racing (`parallel_split: 1` = serial) so
        // certificate attribution can be read off ONE ay solve instead of an
        // AND over 2^k raced subproblems. Sound either way: the serial and split
        // lanes share the same all-Unsat / any-Sat / else-Timeout contract.
        let parallel_split = if std::env::var("NY_GRAPH_MIP_SERIAL").ok().as_deref() == Some("1") {
            1
        } else {
            MipConfig::default().parallel_split
        };
        let config = MipConfig {
            backend,
            timeout_secs: solver_secs,
            parallel_split,
            ..MipConfig::default()
        };
        match MipSolver::new(parts, config).check_feasibility() {
            Ok(MipResult::Unsat { certified: true }) => {
                // Clause certified impossible; continue with the next one.
                // inc5d attribution: `certified: true` is minted ONLY for
                // root-LP infeasibility with a verified Farkas witness
                // (ny-mip/src/ay_lib.rs) — so this line distinguishes
                // root-certified from branched-uncertified per clause.
                info!(
                    "graph-MIP escalation: clause {}/{n_clauses} ROOT-CERTIFIED infeasible \
                     (verified Farkas) in {:.2}s (bounds: {bounds_source})",
                    clause_idx + 1,
                    clause_start.elapsed().as_secs_f64()
                );
            }
            Ok(MipResult::Unsat { certified: false }) => {
                info!(
                    "graph-MIP escalation: clause {}/{n_clauses} infeasible only via BRANCHED \
                     search (case-split, no root-LP Farkas certificate) after {:.2}s (bounds: \
                     {bounds_source}); fail closed (certified evidence is required)",
                    clause_idx + 1,
                    clause_start.elapsed().as_secs_f64()
                );
                return None;
            }
            Ok(MipResult::Sat { .. }) => {
                // UNSAT-ONLY: a feasible violation point is NOT turned into a
                // counterexample here (mscn is an all-unsat class; sat is the
                // attack's job) — fall back to the inconclusive BaB verdict.
                info!(
                    "graph-MIP escalation: clause {}/{n_clauses} violation system is feasible; \
                     UNSAT-only escalation does not emit sat — keeping the BaB verdict",
                    clause_idx + 1
                );
                return None;
            }
            Ok(MipResult::Timeout) => {
                info!(
                    "graph-MIP escalation: clause {}/{n_clauses} timed out ({:.1}s solver slice, \
                     bounds: {bounds_source}); fail closed",
                    clause_idx + 1,
                    solver_secs
                );
                return None;
            }
            Ok(MipResult::Error(msg)) => {
                info!(
                    "graph-MIP escalation: clause {}/{n_clauses} solver error ({msg}); fail \
                     closed",
                    clause_idx + 1
                );
                return None;
            }
            Err(e) => {
                info!(
                    "graph-MIP escalation: clause {}/{n_clauses} solve failed ({e}); fail closed",
                    clause_idx + 1
                );
                return None;
            }
        }
    }

    info!(
        "graph-MIP escalation: ALL {n_clauses} violation clause(s) certified infeasible in \
         {:.1}s → property UNSAT",
        overall_start.elapsed().as_secs_f64()
    );
    Some(GraphMipCertifiedUnsat {
        num_outputs: spec.num_outputs,
    })
}

#[cfg(test)]
#[path = "graph_mip_tests.rs"]
mod tests;
