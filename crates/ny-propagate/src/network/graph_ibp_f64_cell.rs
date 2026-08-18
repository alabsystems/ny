// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound f64 interval forward over a [`GraphNetwork`] for (near-)point inputs.
//!
//! # Why f64 (#cctsdb Phase C)
//!
//! The cctsdb_yolo_2023 cell-enumeration driver fixes every input dimension to
//! a point and needs a decisive enclosure of `Y_0` against the 0.5 threshold.
//! Plain f32 interval propagation amplifies interval width by roughly x2-7 per
//! convolution through the 30-conv backbone (measured ~2e10 end-to-end), so
//! even 1-ulp f32 input widening blows up to +/-1e4 at the heads — undecidable.
//! In f64 the rounding seeds are ~1e-16 relative, so the amplified output width
//! stays ~1e-4: decisively below the property margin.
//!
//! # Second consumer: box-refinement f64 leaf escalation (#f64-leaf)
//!
//! The nn4sys mscn band clauses have margins (~1e-5) BELOW the sound f32
//! forward floor: the disjunctive box-refinement screen refines clause boxes
//! down to f32 resolution and the f32 outward rounding still straddles the
//! threshold. Its terminal-failure leaves are re-evaluated here (gated by
//! [`GraphNetwork::supports_ibp_f64_cell`], kill-switch `NY_F64_LEAF=0`),
//! which motivated the Sigmoid / ReduceSum / live-MatMul / N-D-Linear ops.
//!
//! # Soundness contract
//!
//! For every node the computed interval ENCLOSES the exact real-arithmetic
//! image of the input box:
//! - data-movement ops (slice/reshape/gather/concat/transpose/resize/maxpool/
//!   relu/clip/min/max/trunc) are EXACT in f64 — endpoints are copied or
//!   compared, never rounded;
//! - element-wise arithmetic (+, -, *, /) computes endpoint candidates in f64
//!   round-to-nearest and widens the result OUTWARD by 1 ulp per endpoint
//!   (a nearest-rounded f64 op is within 0.5 ulp of the real value);
//! - dot products (Conv2d / Linear / MatMul) accumulate naively in f64 and
//!   widen by the standard Higham bound `gamma_n * sum(|terms|)` plus an
//!   absolute `ops * 2^-1074` floor. The relative term covers normal-range
//!   products and sums (Higham, *Accuracy and Stability of Numerical
//!   Algorithms*, sec. 3.5); the absolute term conservatively charges every
//!   multiply, add, magnitude reduction, bias/assembly op, including
//!   subnormal results for which the relative model alone is insufficient;
//! - index ops (ArgMax) return the sound candidate-set hull; comparisons
//!   (Equal) return {0}, {1}, or [0,1] exactly as the f32 CompareTensor.
//!
//! Unsupported layers FAIL CLOSED with `UnsupportedOp`: the caller must treat
//! the cell as undecided. f32 weights convert to f64 exactly (no rounding).
//! Every proof entry also probes the active thread's binary64 environment and
//! refuses the walk unless round-to-nearest/ties-to-even and gradual underflow
//! are active: the widening arguments do not hold under directed rounding or
//! when FTZ/DAZ silently replaces a subnormal operand or result with zero.

use std::collections::{HashMap, HashSet};

use ndarray::{ArrayD, ArrayView2, Ix2, IxDyn};
pub(super) use ny_core::require_f64_interval_proof_environment;
use ny_core::{NyError, Result};

use crate::bounds::safe_math::f32_to_f64_exact_for_bounds;
use crate::layers::misc::CompareOp;
use crate::layers::Layer;

use super::core::graph::{GraphNetwork, NETWORK_INPUT};
use super::graph_ibp_f64_gemm::{
    fast_gemm_enabled, rump_interval_matmul, FAST_GEMM_MIN_ROWS, FAST_GEMM_MIN_VOLUME,
};

/// Check both the caller and every worker in Rayon’s current pool.
///
/// Batched f64 walks perform outward arithmetic inside Rayon closures. FP
/// control state is thread-local on common targets, so checking only the
/// caller would not qualify those worker computations.
pub(super) fn require_f64_interval_proof_environment_rayon() -> Result<()> {
    require_f64_interval_proof_environment()?;
    for result in rayon::broadcast(|_| require_f64_interval_proof_environment()) {
        result?;
    }
    Ok(())
}

/// f64 interval tensor: elementwise `lower <= upper`.
#[derive(Debug, Clone)]
pub struct Interval64 {
    pub lower: ArrayD<f64>,
    pub upper: ArrayD<f64>,
}

impl Interval64 {
    pub fn point(values: ArrayD<f64>) -> Self {
        Self {
            lower: values.clone(),
            upper: values,
        }
    }

    /// Exact f32 -> f64 widening of an interval (every f32 is representable
    /// in f64, so no rounding occurs).
    pub fn from_f32(lower: &ArrayD<f32>, upper: &ArrayD<f32>) -> Self {
        Self {
            lower: lower.mapv(f32_to_f64_exact_for_bounds),
            upper: upper.mapv(f32_to_f64_exact_for_bounds),
        }
    }

    fn shape(&self) -> &[usize] {
        self.lower.shape()
    }

    fn reshaped(&self, shape: &[usize]) -> Result<Self> {
        let reshape = |arr: &ArrayD<f64>| -> Result<ArrayD<f64>> {
            arr.clone()
                .into_shape_with_order(IxDyn(shape))
                .map_err(|e| NyError::InvalidSpec(format!("f64 cell reshape failed: {e}")))
        };
        Ok(Self {
            lower: reshape(&self.lower)?,
            upper: reshape(&self.upper)?,
        })
    }
}

/// Unit roundoff for f64.
const U64_EPS: f64 = 2.220446049250313e-16 / 2.0; // 2^-53

/// Higham gamma_n = n*u / (1 - n*u); infinite/invalid for n*u >= 1.
pub(super) fn gamma_n(n: usize) -> Result<f64> {
    let nu = n as f64 * U64_EPS;
    if nu >= 1.0 {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: dot product too long for Higham bound".to_string(),
        ));
    }
    Ok(nu / (1.0 - nu))
}

/// Widen an interval endpoint pair outward by one ulp each.
#[inline]
pub(super) fn widen1(lo: f64, hi: f64) -> (f64, f64) {
    (lo.next_down(), hi.next_up())
}

/// Step a value `n` ulps toward -inf.
#[inline]
pub(super) fn widen_down_n(x: f64, n: u32) -> f64 {
    let mut v = x;
    for _ in 0..n {
        v = v.next_down();
    }
    v
}

/// Step a value `n` ulps toward +inf.
#[inline]
pub(super) fn widen_up_n(x: f64, n: u32) -> f64 {
    let mut v = x;
    for _ in 0..n {
        v = v.next_up();
    }
    v
}

/// Conservative absolute error floor for `operations` binary64 operations.
///
/// Under round-to-nearest with gradual underflow, one finite elementary
/// result can lose at most half a minimum subnormal when rounded; charging a
/// whole `2^-1074` per multiply/add/subtract is deliberately conservative.
/// Callers include magnitude and bias/assembly arithmetic too, so a relative
/// error term cannot hide a subnormal result that rounded to zero. `next_up`
/// covers the
/// integer-to-f64 count conversion and multiplication used to build the
/// floor. Callers have already qualified the floating-point environment.
pub(super) fn operation_underflow_floor(operations: usize) -> Result<f64> {
    if operations == 0 {
        return Ok(0.0);
    }
    let floor = operations as f64 * f64::from_bits(1);
    if !floor.is_finite() || floor <= 0.0 {
        return Err(NyError::SoundnessRefusal(
            "f64 interval proof could not construct its operation-underflow floor".to_string(),
        ));
    }
    Ok(floor.next_up())
}

/// Checked conservative elementary-operation count for one reduction.
fn reduction_operation_count(
    terms: usize,
    operations_per_term: usize,
    fixed_operations: usize,
    operation: &str,
) -> Result<usize> {
    terms
        .checked_mul(operations_per_term)
        .and_then(|count| count.checked_add(fixed_operations))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "f64 cell eval: {operation} operation count overflow"
            ))
        })
}

/// Largest integer magnitude every binary64 value represents EXACTLY: every
/// integer `k` with `|k| <= 2^53` is a binary64 value, so an arithmetic result
/// that is mathematically an integer in that range is returned with NO rounding
/// error, under ANY rounding mode.
const EXACT_INTEGER_MAGNITUDE: i128 = 1i128 << 53;

/// `Some(|v|)` when `v` is an integer-valued finite f64 inside
/// [`EXACT_INTEGER_MAGNITUDE`]; `None` otherwise (non-integer, non-finite, or
/// too large for the exactness argument below).
#[inline]
fn exact_integer_magnitude(v: f64) -> Option<i128> {
    if !v.is_finite() || v != v.trunc() {
        return None;
    }
    let m = v.abs();
    if m > EXACT_INTEGER_MAGNITUDE as f64 {
        return None;
    }
    // `m` is an integer <= 2^53, so the cast is exact.
    Some(m as i128)
}

/// Max integer magnitude over `values`, or `None` as soon as any entry fails
/// [`exact_integer_magnitude`]. Bails on the FIRST offender, so on ordinary
/// (non-integral) weights and activations this costs O(1).
#[inline]
fn max_exact_integer_magnitude<I: IntoIterator<Item = f64>>(values: I) -> Option<i128> {
    let mut max = 0i128;
    for v in values {
        max = max.max(exact_integer_magnitude(v)?);
    }
    Some(max)
}

/// EXACTNESS CERTIFICATE for one Linear layer's interval reductions
/// (#sat-relu-zero-margin).
///
/// # What it decides
///
/// `true` means: for EVERY output row of this layer, every product `w * x`,
/// every partial sum in ANY summation order, and the optional bias add are
/// mathematically integers of magnitude at most `2^53`, hence EXACTLY
/// representable in binary64. Therefore the naive `interval_dot_f32w`
/// accumulation returns the EXACT real-arithmetic endpoints, the Higham
/// `gamma_n * Sum|terms|` relative term is provably ZERO, and the subnormal
/// operation floor is provably ZERO (no result is subnormal — every nonzero
/// result has magnitude at least 1). The caller may then emit `[lo, hi]`
/// with no outward widening at all, and a point input yields a WIDTH-ZERO
/// output interval that is still a valid enclosure.
///
/// # Why the bound is sufficient
///
/// Let `S = in_dim * max|w| * max|x| + max|b|` over this layer, computed in
/// exact integer arithmetic. Every term `w_j * x_j` is an integer with
/// `|w_j * x_j| <= max|w| * max|x| <= S`, so each product is exact. Every
/// partial sum of any subset of terms is an integer bounded by
/// `Sum_j |w_j x_j| <= in_dim * max|w| * max|x| <= S`, so every add is exact —
/// including the 4-lane unrolled order, the lane recombination, and the bias
/// add (`|lo| + |b| <= S`). `interval_dot_f32w`'s `min`/`max` selection and
/// `abs` are exact for finite operands. `S <= 2^53` is therefore sufficient.
///
/// # Why it does not need round-to-nearest
///
/// The argument is about results that are exactly representable, which no
/// rounding mode perturbs — so this path is strictly weaker than the walk's
/// standing round-to-nearest/gradual-underflow precondition, never stronger.
///
/// # Why this matters
///
/// `sat_relu` compiles k-SAT into `Gemm -> ReLU -> Gemm` with `w in {-1,1,2}`
/// and integer biases; its unsafe region `Y_0 >= 1 AND Y_1 <= 0` is attained
/// with EXACTLY zero margin at boolean corners. Charging `gamma_n` there
/// straddles the boundary (measured `Y_0 in [1 - 9.8e-15, 1 + 1.0e-14]`) and no
/// enclosure-based rule can certify the counterexample. With the operands
/// integral the rounding contribution is not "small", it is ZERO, and the
/// enclosure collapses onto the exact value `Y = [1, 0]`.
fn integer_exact_linear_reduction(
    weight: &[f32],
    bias: Option<&ndarray::Array1<f32>>,
    x_lo: &[f64],
    x_hi: &[f64],
    in_dim: usize,
) -> bool {
    // Cheapest discriminator first: ordinary nets fail on their first weight,
    // so they never reach the environment lookup below.
    let Some(w_max) =
        max_exact_integer_magnitude(weight.iter().map(|&w| f32_to_f64_exact_for_bounds(w)))
    else {
        return false;
    };
    // Kill switch (`NY_F64_EXACT_INTEGER=0`) restores the unconditional Higham
    // widening, so the before/after is measurable in one binary. Disabling only
    // LOOSENS the enclosure — it can never make an unsound bound sound.
    if std::env::var("NY_F64_EXACT_INTEGER").is_ok_and(|v| v == "0") {
        return false;
    }
    let Some(x_max) = max_exact_integer_magnitude(x_lo.iter().copied())
        .and_then(|lo| max_exact_integer_magnitude(x_hi.iter().copied()).map(|hi| lo.max(hi)))
    else {
        return false;
    };
    let b_max = match bias {
        None => 0,
        Some(b) => {
            match max_exact_integer_magnitude(b.iter().map(|&v| f32_to_f64_exact_for_bounds(v))) {
                Some(m) => m,
                None => return false,
            }
        }
    };
    (in_dim as i128)
        .checked_mul(w_max)
        .and_then(|v| v.checked_mul(x_max))
        .and_then(|v| v.checked_add(b_max))
        .is_some_and(|bound| bound <= EXACT_INTEGER_MAGNITUDE)
}

/// Outward upper bound for a product reduction's total rounding error.
fn product_sum_error_bound(gamma: f64, magnitude: f64, operations: usize) -> Result<f64> {
    if !gamma.is_finite() || !magnitude.is_finite() || gamma < 0.0 || magnitude < 0.0 {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: non-finite product error magnitude".to_string(),
        ));
    }
    let relative = widen_up_n(gamma * magnitude, 2);
    let absolute = operation_underflow_floor(operations)?;
    let total = widen_up_n(relative + absolute, 2);
    if !total.is_finite() || total < 0.0 {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: product error bound overflow".to_string(),
        ));
    }
    Ok(total)
}

/// Numerically stable ordinary sigmoid used only as a concrete test oracle.
/// It deliberately has no proof authority: Rust does not specify a global
/// error bound for the platform `exp`.  Certified walks use
/// [`certified_sigmoid_f64`] below.
#[inline]
#[cfg(test)]
pub(super) fn stable_sigmoid_f64(x: f64) -> f64 {
    if x <= 0.0 {
        let t = x.exp();
        t / (1.0 + t)
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Rigorous enclosure of `exp(-a)` for one finite binary64 `a >= 0`.
///
/// No platform-libm accuracy assumption enters this path.  Range reduction
/// divides by an exact power of two until `y <= 1/2`; the alternating series
///
/// `exp(-y) = 1 - y + y^2/2! - ...`
///
/// has decreasing non-negative terms, so every odd partial sum is a lower
/// bound and every even partial sum is an upper bound.  Directed interval
/// arithmetic encloses those partial sums, then monotone interval squaring
/// reverses the exact power-of-two reduction.  Thirty-two terms put the
/// analytic remainder far below binary64 resolution at `y <= 1/2`; the proof
/// itself relies only on the alternating-series ordering, not that estimate.
fn exp_neg_certified(a: f64) -> (f64, f64) {
    debug_assert!(a >= 0.0 || a.is_nan());
    if a.is_nan() {
        return (0.0, 1.0);
    }
    if a == 0.0 {
        return (1.0, 1.0);
    }
    if a == f64::INFINITY {
        return (0.0, 0.0);
    }
    // e > 2 (already the first two positive terms of its power series), so
    // a >= 1075 implies exp(-a) < 2^-1075 < the minimum positive binary64.
    // Keeping one minimum subnormal as the upper endpoint is conservative.
    if a >= 1075.0 {
        return (0.0, f64::from_bits(1));
    }

    let mut y = a;
    let mut squarings = 0_u32;
    // Range reduction by exact halving: `0 < a < 1075` here (NaN/0/inf and the
    // `a >= 1075` tail returned above), so the loop terminates in at most 12
    // steps. The comparison against 1/2 is the reduction invariant the proof
    // above relies on, so it is left exactly as written.
    #[allow(clippy::while_float)]
    while y > 0.5 {
        // Multiplication by 2^-1 is exact throughout this bounded range.
        y *= 0.5;
        squarings += 1;
    }

    let mut term_lo = 1.0_f64;
    let mut term_hi = 1.0_f64;
    let mut sum_lo = 1.0_f64;
    let mut sum_hi = 1.0_f64;
    let mut lower = 0.0_f64;
    let mut upper = 1.0_f64;
    for k in 1_u32..=32 {
        let divisor = f64::from(k);
        let product_lo = (term_lo * y).next_down().max(0.0);
        let product_hi = (term_hi * y).next_up();
        term_lo = (product_lo / divisor).next_down().max(0.0);
        term_hi = (product_hi / divisor).next_up();

        if k % 2 == 1 {
            sum_lo = (sum_lo - term_hi).next_down();
            sum_hi = (sum_hi - term_lo).next_up();
            // The exact odd partial sum is <= exp(-y).
            lower = sum_lo.max(0.0);
        } else {
            sum_lo = (sum_lo + term_lo).next_down();
            sum_hi = (sum_hi + term_hi).next_up();
            // The exact even partial sum is >= exp(-y).
            upper = sum_hi.min(1.0);
        }
    }

    for _ in 0..squarings {
        lower = (lower * lower).next_down().max(0.0);
        upper = (upper * upper).next_up().min(1.0);
    }
    (lower, upper)
}

/// Rigorous binary64 interval containing the exact-real logistic sigmoid at
/// one binary64 point.  This is the proof-authority entry; the ordinary
/// [`stable_sigmoid_f64`] remains only a concrete/test oracle.
pub(super) fn certified_sigmoid_f64(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (0.0, 1.0);
    }
    if x == f64::NEG_INFINITY {
        return (0.0, 0.0);
    }
    if x == f64::INFINITY {
        return (1.0, 1.0);
    }
    if x == 0.0 {
        return (0.5, 0.5);
    }

    let (t_lo, t_hi) = exp_neg_certified(x.abs());
    let (lower, upper) = if x < 0.0 {
        // sigma(x) = t/(1+t), monotone increasing in t = exp(x).
        let denominator_up = (1.0 + t_lo).next_up();
        let denominator_down = (1.0 + t_hi).next_down().max(t_hi);
        (
            (t_lo / denominator_up).next_down().max(0.0),
            (t_hi / denominator_down).next_up().min(1.0),
        )
    } else {
        // sigma(x) = 1/(1+t), monotone decreasing in t = exp(-x).
        let denominator_up = (1.0 + t_hi).next_up();
        let denominator_down = (1.0 + t_lo).next_down().max(t_lo);
        (
            (1.0 / denominator_up).next_down().max(0.0),
            (1.0 / denominator_down).next_up().min(1.0),
        )
    };
    (lower, upper)
}

/// Elementwise binary interval op with NumPy broadcasting.
///
/// `combine(alo, ahi, blo, bhi)` must return a sound REAL-arithmetic interval
/// for the op; the result is widened 1 ulp outward per endpoint to absorb the
/// f64 rounding of the endpoint computation itself (skipped when `exact`).
///
/// The four broadcast views are materialized to standard layout (a free
/// borrow for already-contiguous inputs — the dominant same-shape case — and
/// a small copy for stride-0 broadcast axes) so the element loop runs on
/// FLAT slices: the previous `indexed_iter` + per-element `IxDyn` clones and
/// indexed lookups dominated the tiny-tensor mscn walks (#f64-flat-elemwise,
/// ~13% whole-run `Baseiter` self time). VALUE-IDENTICAL: flat element `e`
/// of a standard-layout copy of a broadcast view is exactly the value
/// `indexed_iter` yielded at logical (row-major) index `e`, and `combine` is
/// applied in the same row-major order with the same endpoint pairs
/// (bit-identity gate: `broadcast_binary_flat_matches_indexed_reference`).
pub(super) fn broadcast_binary(
    a: &Interval64,
    b: &Interval64,
    exact: bool,
    combine: impl Fn(f64, f64, f64, f64) -> Result<(f64, f64)>,
) -> Result<Interval64> {
    let out_shape = crate::shape::broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| {
        NyError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        }
    })?;
    let bc_err = || NyError::InvalidSpec("f64 cell eval: broadcast failed".to_string());
    let alo = a.lower.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;
    let ahi = a.upper.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;
    let blo = b.lower.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;
    let bhi = b.upper.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;

    let alo_std = alo.as_standard_layout();
    let ahi_std = ahi.as_standard_layout();
    let blo_std = blo.as_standard_layout();
    let bhi_std = bhi.as_standard_layout();
    let (alo_s, ahi_s, blo_s, bhi_s) = match (
        alo_std.as_slice(),
        ahi_std.as_slice(),
        blo_std.as_slice(),
        bhi_std.as_slice(),
    ) {
        (Some(al), Some(ah), Some(bl), Some(bh)) => (al, ah, bl, bh),
        _ => return Err(bc_err()),
    };

    let n: usize = out_shape.iter().product();
    let mut out_lo = Vec::with_capacity(n);
    let mut out_hi = Vec::with_capacity(n);
    for i in 0..n {
        let (mut lo, mut hi) = combine(alo_s[i], ahi_s[i], blo_s[i], bhi_s[i])?;
        if !exact {
            (lo, hi) = widen1(lo, hi);
        }
        out_lo.push(lo);
        out_hi.push(hi);
    }
    Ok(Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&out_shape), out_lo)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: broadcast out: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&out_shape), out_hi)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: broadcast out: {e}")))?,
    })
}

/// Sound real interval product of `[al, ah] * [bl, bh]` (endpoint candidates).
pub(super) fn interval_mul(al: f64, ah: f64, bl: f64, bh: f64) -> (f64, f64) {
    let candidates = [al * bl, al * bh, ah * bl, ah * bh];
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in candidates {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    (lo, hi)
}

impl GraphNetwork {
    /// Sound f64 interval forward, restricted to the ancestors of the output
    /// node. Fails closed (`UnsupportedOp`/`InvalidSpec`) on anything outside
    /// the supported op set — the caller must treat that as "cell undecided".
    pub fn propagate_ibp_f64_cell(&self, input: &Interval64) -> Result<Interval64> {
        require_f64_interval_proof_environment()?;

        // Restrict evaluation to ancestors of the output: dead branches (e.g.
        // cctsdb's OpaqueSkip'd Slice_34/38 whose consumers all const-folded
        // away) must not fail-closed a decidable cell.
        let needed = self.output_ancestors()?;

        let mut cache: HashMap<&str, Interval64> = HashMap::new();
        let exec_order = self.exec_order()?;
        for node_name in exec_order {
            if !needed.contains(node_name.as_str()) {
                continue;
            }
            let node = self.node(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("f64 cell eval: missing node '{node_name}'"))
            })?;
            let resolve = |name: &str| -> Result<Interval64> {
                if name == NETWORK_INPUT {
                    return Ok(input.clone());
                }
                cache.get(name).cloned().ok_or_else(|| {
                    NyError::InvalidSpec(format!("f64 cell eval: '{name}' not computed"))
                })
            };
            let out = eval_node(node.layer(), node, &resolve)?;
            cache.insert(node.name(), out);
        }
        cache.get(self.output_name()).cloned().ok_or_else(|| {
            NyError::InvalidSpec("f64 cell eval: output node not computed".to_string())
        })
    }

    /// Whether every output-ancestor node of this graph is in the op set
    /// supported by [`Self::propagate_ibp_f64_cell`]. Callers that escalate
    /// many boxes (the box-refinement f64 leaf escalation, #f64-leaf) should
    /// gate on this once instead of paying a failed walk per box. Runtime
    /// conditions (e.g. a Div divisor interval containing zero for a specific
    /// box) can still fail an individual walk — always fail-closed on `Err`.
    pub fn supports_ibp_f64_cell(&self) -> bool {
        match self.output_ancestors() {
            Ok(needed) => needed.iter().all(|name| {
                self.node(name)
                    .is_some_and(|node| cell_supports_layer(node.layer()))
            }),
            Err(_) => false,
        }
    }

    /// Names of all ancestor nodes of the output node (inclusive).
    pub(super) fn output_ancestors(&self) -> Result<HashSet<&str>> {
        let mut needed: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = vec![self.output_name()];
        while let Some(name) = stack.pop() {
            if name == NETWORK_INPUT || !needed.insert(name) {
                continue;
            }
            let node = self.node(name).ok_or_else(|| {
                NyError::InvalidSpec(format!("f64 cell eval: missing ancestor '{name}'"))
            })?;
            for input in node.inputs() {
                stack.push(input);
            }
        }
        Ok(needed)
    }
}

/// Resolve this node's inputs in ONNX order (interleaving Concat constants).
fn node_inputs(
    node: &super::core::graph::GraphNode,
    resolve: &dyn Fn(&str) -> Result<Interval64>,
) -> Result<Vec<Interval64>> {
    if let Layer::Concat(concat) = node.layer() {
        if let Some(ref constant_inputs) = concat.constant_inputs {
            let mut graph_idx = 0usize;
            return constant_inputs
                .iter()
                .map(|slot| match slot {
                    Some(constant) => Ok(Interval64::from_f32(constant.lower(), constant.upper())),
                    None => {
                        let name = node.inputs().get(graph_idx).ok_or_else(|| {
                            NyError::InvalidSpec(
                                "f64 cell eval: Concat ran out of graph inputs".to_string(),
                            )
                        })?;
                        graph_idx += 1;
                        resolve(name)
                    }
                })
                .collect();
        }
    }
    node.inputs().iter().map(|name| resolve(name)).collect()
}

/// Static layer-support predicate for [`GraphNetwork::supports_ibp_f64_cell`].
///
/// KEEP IN SYNC with the `eval_node` dispatch below: every arm that can
/// succeed must be listed here (including its static preconditions — e.g.
/// Gather needs constant indices); anything else must return `false` so the
/// escalation gate fails closed. Shape/rank/runtime conditions (Div divisor
/// sign, MatMul rank) are still checked per walk.
fn cell_supports_layer(layer: &Layer) -> bool {
    match layer {
        Layer::Reshape(_)
        | Layer::Flatten(_)
        | Layer::Slice(_)
        | Layer::Squeeze(_)
        | Layer::Unsqueeze(_)
        | Layer::Transpose(_)
        | Layer::Concat(_)
        | Layer::Resize(_)
        | Layer::MaxPool2d(_)
        | Layer::ReLU(_)
        | Layer::Clip(_)
        | Layer::Trunc(_)
        | Layer::Sigmoid(_)
        | Layer::MinBinary(_)
        | Layer::MaxBinary(_)
        | Layer::ArgMax(_)
        | Layer::Add(_)
        | Layer::Sub(_)
        | Layer::MulBinary(_)
        | Layer::Div(_)
        | Layer::AddConstant(_)
        | Layer::SubConstant(_)
        | Layer::MulConstant(_)
        | Layer::ReduceSum(_)
        | Layer::Linear(_)
        | Layer::MatMul(_)
        | Layer::Conv2d(_) => true,
        // Geometry is a STATIC property of the layer (stride/dilation >= 1,
        // output_padding < stride, rank-4 kernel with nonzero extents), so the
        // gate can qualify it here; only the input rank is per-walk.
        Layer::ConvTranspose2d(conv) => conv.validate_geometry().is_ok(),
        // A degenerate channel (var + eps -> 0) carries a +/-inf `scale`, whose
        // product with a zero-width endpoint is NaN rather than an enclosure, so
        // non-finite coefficients fail the gate STATICALLY (they are load-time
        // constants) as well as per walk.
        Layer::BatchNorm(bn) => {
            bn.scale.iter().all(|v| v.is_finite()) && bn.bias.iter().all(|v| v.is_finite())
        }
        Layer::Gather(gather) => gather.constant_indices().is_some(),
        Layer::CompareTensor(compare) => compare.op == CompareOp::Eq,
        Layer::ScatterNd(scatter) => {
            scatter.data_constant().is_some()
                && scatter.updates_constant().is_some()
                && !scatter.has_static_indices()
        }
        _ => false,
    }
}

pub(super) fn eval_node(
    layer: &Layer,
    node: &super::core::graph::GraphNode,
    resolve: &dyn Fn(&str) -> Result<Interval64>,
) -> Result<Interval64> {
    let inputs = node_inputs(node, resolve)?;
    let unary = || -> Result<&Interval64> {
        inputs
            .first()
            .ok_or_else(|| NyError::InvalidSpec("f64 cell eval: missing unary input".to_string()))
    };
    let binary = || -> Result<(&Interval64, &Interval64)> {
        match (inputs.first(), inputs.get(1)) {
            (Some(a), Some(b)) => Ok((a, b)),
            _ => Err(NyError::InvalidSpec(
                "f64 cell eval: missing binary inputs".to_string(),
            )),
        }
    };

    match layer {
        // ---- exact data movement -------------------------------------------------
        Layer::Reshape(reshape) => {
            let x = unary()?;
            let out_shape = reshape.compute_output_shape(x.shape())?;
            x.reshaped(&out_shape)
        }
        // Exact data movement like Reshape: ONNX Flatten collapses the shape to
        // 2-D around `axis`; endpoints are copied verbatim (row-major), never
        // rounded. Unblocks the exact-f64 witness gate on soundnessbench
        // (Conv x6 -> Flatten -> Gemm x2), whose whole-net escalation previously
        // failed closed here (#sb-rebank lever 2).
        Layer::Flatten(flatten) => {
            let x = unary()?;
            let out_shape = flatten.compute_output_shape(x.shape())?;
            x.reshaped(&out_shape)
        }
        Layer::Slice(slice) => {
            let x = unary()?;
            let (axis, start, end) = slice.resolved_range(x.shape())?;
            let take = |arr: &ArrayD<f64>| {
                arr.slice_axis(ndarray::Axis(axis), ndarray::Slice::from(start..end))
                    .to_owned()
            };
            Ok(Interval64 {
                lower: take(&x.lower),
                upper: take(&x.upper),
            })
        }
        Layer::Squeeze(squeeze) => {
            let x = unary()?;
            let ndim = x.shape().len();
            let axis = resolve_axis_i64(squeeze.axis as i64, ndim, "Squeeze")?;
            if x.shape()[axis] != 1 {
                return Err(NyError::InvalidSpec(format!(
                    "f64 cell eval: Squeeze axis {axis} has size {}",
                    x.shape()[axis]
                )));
            }
            let mut shape = x.shape().to_vec();
            shape.remove(axis);
            x.reshaped(&shape)
        }
        Layer::Unsqueeze(unsqueeze) => {
            let x = unary()?;
            let ndim = x.shape().len();
            let axis = resolve_axis_i64(unsqueeze.axis as i64, ndim + 1, "Unsqueeze")?;
            let mut shape = x.shape().to_vec();
            shape.insert(axis, 1);
            x.reshaped(&shape)
        }
        Layer::Transpose(transpose) => {
            let x = unary()?;
            if transpose.axes.len() != x.shape().len() {
                return Err(NyError::InvalidSpec(
                    "f64 cell eval: Transpose perm rank mismatch".to_string(),
                ));
            }
            let perm = transpose.axes.clone();
            Ok(Interval64 {
                lower: x
                    .lower
                    .clone()
                    .permuted_axes(IxDyn(&perm))
                    .as_standard_layout()
                    .to_owned(),
                upper: x
                    .upper
                    .clone()
                    .permuted_axes(IxDyn(&perm))
                    .as_standard_layout()
                    .to_owned(),
            })
        }
        Layer::Gather(gather) => eval_gather(gather, unary()?),
        Layer::Concat(concat) => eval_concat(concat, &inputs),
        Layer::Resize(resize) => eval_resize(resize, unary()?),
        Layer::MaxPool2d(pool) => eval_maxpool(pool, unary()?),

        // ---- exact monotone / selection ops ---------------------------------------
        Layer::ReLU(_) => {
            let x = unary()?;
            Ok(Interval64 {
                lower: x.lower.mapv(|v| v.max(0.0)),
                upper: x.upper.mapv(|v| v.max(0.0)),
            })
        }
        Layer::Clip(clip) => {
            let x = unary()?;
            let (min, max) = (
                f32_to_f64_exact_for_bounds(clip.min),
                f32_to_f64_exact_for_bounds(clip.max),
            );
            Ok(Interval64 {
                lower: x.lower.mapv(|v| v.clamp(min, max)),
                upper: x.upper.mapv(|v| v.clamp(min, max)),
            })
        }
        Layer::Trunc(trunc) => {
            // trunc (round-toward-zero) is monotone non-decreasing and exact in
            // f64. A Trunc originating from FLOAT32->INT32/INT64 Cast carries
            // the additional ONNX destination-domain certificate; this direct
            // evaluator bypasses BoundPropagation, so enforce it explicitly.
            let x = unary()?;
            trunc.validate_f64_domain(&x.lower, &x.upper)?;
            Ok(Interval64 {
                lower: x.lower.mapv(f64::trunc),
                upper: x.upper.mapv(f64::trunc),
            })
        }
        Layer::Sigmoid(_) => {
            // Monotone increasing: evaluate the lower endpoint's certified
            // lower bound and the upper endpoint's certified upper bound.
            // The certificate uses a directed alternating series and contains
            // no platform-libm accuracy assumption. nn4sys mscn output head.
            let x = unary()?;
            Ok(Interval64 {
                lower: x.lower.mapv(|value| certified_sigmoid_f64(value).0),
                upper: x.upper.mapv(|value| certified_sigmoid_f64(value).1),
            })
        }
        Layer::MinBinary(_) => {
            let (a, b) = binary()?;
            broadcast_binary(a, b, true, |al, ah, bl, bh| Ok((al.min(bl), ah.min(bh))))
        }
        Layer::MaxBinary(_) => {
            let (a, b) = binary()?;
            broadcast_binary(a, b, true, |al, ah, bl, bh| Ok((al.max(bl), ah.max(bh))))
        }
        Layer::ArgMax(argmax) => eval_argmax(argmax, unary()?),
        Layer::CompareTensor(compare) => {
            let (a, b) = binary()?;
            if compare.op != CompareOp::Eq {
                return Err(NyError::UnsupportedOp(
                    "f64 cell eval: only Eq CompareTensor supported".to_string(),
                ));
            }
            broadcast_binary(a, b, true, |al, ah, bl, bh| {
                Ok(if al == ah && bl == bh && al == bl {
                    (1.0, 1.0)
                } else if al > bh || ah < bl {
                    (0.0, 0.0)
                } else {
                    (0.0, 1.0)
                })
            })
        }

        // ---- element-wise arithmetic (1-ulp outward) -------------------------------
        Layer::Add(_) => {
            let (a, b) = binary()?;
            broadcast_binary(a, b, false, |al, ah, bl, bh| Ok((al + bl, ah + bh)))
        }
        Layer::Sub(_) => {
            let (a, b) = binary()?;
            broadcast_binary(a, b, false, |al, ah, bl, bh| Ok((al - bh, ah - bl)))
        }
        Layer::MulBinary(_) => {
            let (a, b) = binary()?;
            broadcast_binary(a, b, false, |al, ah, bl, bh| {
                Ok(interval_mul(al, ah, bl, bh))
            })
        }
        Layer::Div(_) => {
            let (a, b) = binary()?;
            broadcast_binary(a, b, false, |al, ah, bl, bh| {
                if bl <= 0.0 && bh >= 0.0 {
                    return Err(NyError::InvalidSpec(
                        "f64 cell eval: Div divisor interval contains zero".to_string(),
                    ));
                }
                let candidates = [al / bl, al / bh, ah / bl, ah / bh];
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for c in candidates {
                    lo = lo.min(c);
                    hi = hi.max(c);
                }
                Ok((lo, hi))
            })
        }
        Layer::AddConstant(add) => {
            let x = unary()?;
            let c = Interval64::from_f32(add.constant(), add.constant());
            broadcast_binary(x, &c, false, |al, ah, bl, bh| Ok((al + bl, ah + bh)))
        }
        // `Sub` against a CONSTANT operand. The whole acasxu class opens with
        // `Sub(input, input_AvgImg)` on an initializer, which the loader folds
        // to `SubConstant`, and its absence here fail-closed every acasxu
        // point-box walk (W0.2 reachability gap (a)). The constant is a POINT
        // interval (f32 -> f64 is exact), so the ordinary monotone interval
        // difference applies:
        //   forward  y = x - c  =>  [x_lo - c, x_hi - c]
        //   reverse  y = c - x  =>  [c - x_hi, c - x_lo]
        // `exact = false` widens 1 ulp per endpoint for the f64 subtraction
        // itself, exactly as `AddConstant` does for its addition. Since IEEE-754
        // gives `x - c == x + (-c)` bit-for-bit and negation is exact, this arm
        // is bit-identical to the `AddConstant` arm on the `Sub(x,c) -> Add(x,-c)`
        // rewrite W0.2 measured acasxu on.
        Layer::SubConstant(sub) => {
            let x = unary()?;
            let c = Interval64::from_f32(sub.constant(), sub.constant());
            let reverse = sub.reverse;
            broadcast_binary(x, &c, false, |xl, xh, cl, ch| {
                if reverse {
                    Ok((cl - xh, ch - xl))
                } else {
                    Ok((xl - ch, xh - cl))
                }
            })
        }
        Layer::MulConstant(mul) => {
            let x = unary()?;
            let c = Interval64::from_f32(mul.constant(), mul.constant());
            broadcast_binary(x, &c, false, |al, ah, bl, bh| {
                Ok(interval_mul(al, ah, bl, bh))
            })
        }

        // ---- reductions (Higham widening) -------------------------------------------
        Layer::ReduceSum(reduce) => eval_reduce_sum(reduce, unary()?),

        // ---- dot products (Higham widening) ----------------------------------------
        Layer::Linear(linear) => eval_linear(linear, unary()?),
        Layer::MatMul(matmul) => {
            let (a, b) = binary()?;
            eval_matmul(matmul, a, b)
        }
        Layer::Conv2d(conv) => eval_conv2d(conv, unary()?),
        Layer::ConvTranspose2d(conv) => eval_conv_transpose2d(conv, unary()?),

        // ---- per-channel affine ----------------------------------------------------
        Layer::BatchNorm(bn) => eval_batch_norm(bn, unary()?),

        // ---- scatter ---------------------------------------------------------------
        Layer::ScatterNd(scatter) => eval_scatter_nd(scatter, &inputs),

        other => Err(NyError::UnsupportedOp(format!(
            "f64 cell eval: unsupported layer {}",
            other.layer_type()
        ))),
    }
}

fn resolve_axis_i64(axis: i64, ndim: usize, op: &str) -> Result<usize> {
    let resolved = if axis < 0 { axis + ndim as i64 } else { axis };
    if resolved < 0 || resolved >= ndim as i64 {
        return Err(NyError::InvalidSpec(format!(
            "f64 cell eval: {op} axis {axis} out of range for rank {ndim}"
        )));
    }
    Ok(resolved as usize)
}

fn eval_gather(gather: &crate::layers::GatherLayer, x: &Interval64) -> Result<Interval64> {
    let indices = gather.constant_indices().ok_or_else(|| {
        NyError::UnsupportedOp("f64 cell eval: Gather requires constant indices".to_string())
    })?;
    let axis = resolve_axis_i64(gather.axis_raw(), x.shape().len(), "Gather")?;
    let axis_len = x.shape()[axis] as i64;
    // Output shape: data[..axis] ++ indices.shape ++ data[axis+1..]
    let mut out_shape: Vec<usize> = x.shape()[..axis].to_vec();
    out_shape.extend_from_slice(indices.shape());
    out_shape.extend_from_slice(&x.shape()[axis + 1..]);

    let take = |arr: &ArrayD<f64>| -> Result<ArrayD<f64>> {
        // Select each index along `axis` in indices order, then reshape the
        // gathered axis to the indices shape.
        let mut slices: Vec<ArrayD<f64>> = Vec::with_capacity(indices.len().max(1));
        for &raw in indices.iter() {
            let idx = if raw < 0 { raw + axis_len } else { raw };
            if idx < 0 || idx >= axis_len {
                return Err(NyError::InvalidSpec(format!(
                    "f64 cell eval: Gather index {raw} out of range for axis len {axis_len}"
                )));
            }
            slices.push(
                arr.index_axis(ndarray::Axis(axis), idx as usize)
                    .insert_axis(ndarray::Axis(axis))
                    .to_owned(),
            );
        }
        let views: Vec<_> = slices.iter().map(|s| s.view()).collect();
        let stacked = if views.is_empty() {
            // Empty indices: empty gathered axis.
            arr.slice_axis(ndarray::Axis(axis), ndarray::Slice::from(0..0))
                .to_owned()
        } else {
            ndarray::concatenate(ndarray::Axis(axis), &views)
                .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: Gather concat: {e}")))?
        };
        stacked
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: Gather reshape: {e}")))
    };
    Ok(Interval64 {
        lower: take(&x.lower)?,
        upper: take(&x.upper)?,
    })
}

pub(super) fn eval_concat(
    concat: &crate::layers::ConcatLayer,
    inputs: &[Interval64],
) -> Result<Interval64> {
    let first = inputs.first().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: Concat needs at least one input".to_string())
    })?;
    let axis = concat.normalize_axis(first.shape().len())?;
    let lo_views: Vec<_> = inputs.iter().map(|i| i.lower.view()).collect();
    let hi_views: Vec<_> = inputs.iter().map(|i| i.upper.view()).collect();
    let lower = ndarray::concatenate(ndarray::Axis(axis), &lo_views)
        .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: Concat: {e}")))?;
    let upper = ndarray::concatenate(ndarray::Axis(axis), &hi_views)
        .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: Concat: {e}")))?;
    Ok(Interval64 { lower, upper })
}

fn eval_resize(resize: &crate::layers::ResizeLayer, x: &Interval64) -> Result<Interval64> {
    // Nearest-neighbor integer upsample of the last two axes (floor mapping):
    // out[..., i, j] = in[..., i / sh, j / sw] — a pure copy, exact.
    let ndim = x.shape().len();
    if ndim < 2 {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: Resize needs rank >= 2".to_string(),
        ));
    }
    let (sh, sw) = (resize.scale_h, resize.scale_w);
    let mut out_shape = x.shape().to_vec();
    out_shape[ndim - 2] *= sh;
    out_shape[ndim - 1] *= sw;
    let take = |arr: &ArrayD<f64>| -> ArrayD<f64> {
        ArrayD::from_shape_fn(IxDyn(&out_shape), |idx| {
            let mut src: Vec<usize> = (0..ndim).map(|d| idx[d]).collect();
            src[ndim - 2] /= sh;
            src[ndim - 1] /= sw;
            arr[IxDyn(&src)]
        })
    };
    Ok(Interval64 {
        lower: take(&x.lower),
        upper: take(&x.upper),
    })
}

fn eval_maxpool(pool: &crate::layers::MaxPool2dLayer, x: &Interval64) -> Result<Interval64> {
    let shape = x.shape().to_vec();
    let ndim = shape.len();
    if !(3..=4).contains(&ndim) {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: MaxPool2d needs rank 3 or 4".to_string(),
        ));
    }
    let (in_h, in_w) = (shape[ndim - 2], shape[ndim - 1]);
    let (kh, kw) = pool.kernel_size;
    let (sh, sw) = pool.stride;
    let (ph, pw) = pool.padding;
    let out_h = (in_h + 2 * ph).saturating_sub(kh) / sh + 1;
    let out_w = (in_w + 2 * pw).saturating_sub(kw) / sw + 1;
    let mut out_shape = shape;
    out_shape[ndim - 2] = out_h;
    out_shape[ndim - 1] = out_w;

    let take = |arr: &ArrayD<f64>| -> ArrayD<f64> {
        ArrayD::from_shape_fn(IxDyn(&out_shape), |idx| {
            let idx_vec: Vec<usize> = (0..ndim).map(|d| idx[d]).collect();
            let (oh, ow) = (idx_vec[ndim - 2], idx_vec[ndim - 1]);
            let mut best = f64::NEG_INFINITY;
            let mut any_valid = false;
            for dh in 0..kh {
                for dw in 0..kw {
                    let ih = (oh * sh + dh) as isize - ph as isize;
                    let iw = (ow * sw + dw) as isize - pw as isize;
                    if ih < 0 || iw < 0 || ih >= in_h as isize || iw >= in_w as isize {
                        continue; // padded region: valid-only pooling
                    }
                    let mut src = idx_vec.clone();
                    src[ndim - 2] = ih as usize;
                    src[ndim - 1] = iw as usize;
                    best = best.max(arr[IxDyn(&src)]);
                    any_valid = true;
                }
            }
            if any_valid {
                best
            } else {
                0.0
            }
        })
    };
    Ok(Interval64 {
        lower: take(&x.lower),
        upper: take(&x.upper),
    })
}

fn eval_argmax(argmax: &crate::layers::ArgMaxLayer, x: &Interval64) -> Result<Interval64> {
    let ndim = x.shape().len();
    let axis = resolve_axis_i64(argmax.axis, ndim, "ArgMax")?;
    let axis_len = x.shape()[axis];
    if axis_len == 0 {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: ArgMax over empty axis".to_string(),
        ));
    }
    let mut out_shape = x.shape().to_vec();
    if argmax.keepdims {
        out_shape[axis] = 1;
    } else {
        out_shape.remove(axis);
    }

    let mut out_lo = ArrayD::zeros(IxDyn(&out_shape));
    let mut out_hi = ArrayD::zeros(IxDyn(&out_shape));
    // Iterate lanes along `axis`.
    let lanes_lo = x.lower.lanes(ndarray::Axis(axis));
    let lanes_hi = x.upper.lanes(ndarray::Axis(axis));
    for ((lane_lo, lane_hi), (slot_lo, slot_hi)) in lanes_lo
        .into_iter()
        .zip(lanes_hi)
        .zip(out_lo.iter_mut().zip(out_hi.iter_mut()))
    {
        // Candidate set: i is a possible argmax iff upper_i >= max_j lower_j.
        let best_lower = lane_lo.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut min_cand = usize::MAX;
        let mut max_cand = 0usize;
        for (i, &hi) in lane_hi.iter().enumerate() {
            if hi >= best_lower {
                min_cand = min_cand.min(i);
                max_cand = max_cand.max(i);
            }
        }
        if min_cand == usize::MAX {
            return Err(NyError::InvalidSpec(
                "f64 cell eval: ArgMax found no candidates".to_string(),
            ));
        }
        if (max_cand as u128) > (1_u128 << 53) {
            return Err(NyError::SoundnessRefusal(
                "f64 cell eval: ArgMax index is not exactly representable in binary64".to_string(),
            ));
        }
        *slot_lo = min_cand as f64;
        *slot_hi = max_cand as f64;
    }
    Ok(Interval64 {
        lower: out_lo,
        upper: out_hi,
    })
}

fn eval_linear(linear: &crate::layers::LinearLayer, x: &Interval64) -> Result<Interval64> {
    eval_linear_with_bias(linear, x, true)
}

/// Linear with an explicit bias switch: the mean-value derivative channel
/// (#f64-mvf) maps derivatives through the SAME Higham-widened `W·d` but the
/// bias (a constant) contributes zero derivative, so it must be excluded.
pub(super) fn eval_linear_with_bias(
    linear: &crate::layers::LinearLayer,
    x: &Interval64,
    include_bias: bool,
) -> Result<Interval64> {
    eval_linear_with_bias_inner(linear, x, include_bias, true)
}

/// Scalar-only Linear — the reference path the fast kernel wiring is
/// soundness-tested against.
#[cfg(test)]
pub(super) fn eval_linear_with_bias_scalar(
    linear: &crate::layers::LinearLayer,
    x: &Interval64,
    include_bias: bool,
) -> Result<Interval64> {
    eval_linear_with_bias_inner(linear, x, include_bias, false)
}

fn eval_linear_with_bias_inner(
    linear: &crate::layers::LinearLayer,
    x: &Interval64,
    include_bias: bool,
    allow_fast: bool,
) -> Result<Interval64> {
    // N-D input, weight applied to the LAST axis (matches the f32 Linear IBP
    // convention `[...batch, in] -> [...batch, out]`). Rank-1 (the original
    // cctsdb cell case) is the batch=1 special case. nn4sys mscn applies its
    // set-MLPs row-wise over [rows, in] inputs (#f64-leaf).
    let shape = x.shape().to_vec();
    if shape.is_empty() {
        return Err(NyError::UnsupportedOp(
            "f64 cell eval: Linear on rank-0 input".to_string(),
        ));
    }
    let (out_dim, in_dim) = linear.weight.dim();
    if shape[shape.len() - 1] != in_dim {
        return Err(NyError::ShapeMismatch {
            expected: vec![in_dim],
            got: shape,
        });
    }
    let batch: usize = shape[..shape.len() - 1].iter().product();

    let x_lo_owned = x.lower.as_standard_layout();
    let x_hi_owned = x.upper.as_standard_layout();
    let x_lo = x_lo_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: linear input not contiguous".to_string())
    })?;
    let x_hi = x_hi_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: linear input not contiguous".to_string())
    })?;

    let mut out_shape = shape[..shape.len() - 1].to_vec();
    out_shape.push(out_dim);

    let weight_owned = linear.weight.as_standard_layout();
    let weight = weight_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: linear weight not contiguous".to_string())
    })?;

    // EXACTNESS path (#sat-relu-zero-margin): integral operands whose reduction
    // bound stays inside binary64's exact-integer range make BOTH widening terms
    // provably zero (argument in `integer_exact_linear_reduction`). The check
    // bails on the first non-integral weight or activation, so ordinary nets pay
    // O(1) for it. Exact layers skip the Rump kernel too — that kernel is
    // correct but charges its own `gamma` term, which would reintroduce the
    // width this path exists to remove.
    let exact = integer_exact_linear_reduction(
        weight,
        if include_bias { linear.bias() } else { None },
        x_lo,
        x_hi,
        in_dim,
    );

    // Fast path (#f64-blas-gemm): Rump midpoint-radius interval GEMM on
    // probed Rayon workers for FAT batches; the constant weight is a point
    // operand so it costs 2 plain GEMMs. Thin batches take the unrolled scalar
    // loop below. `None` (unsafe worker FP mode, non-finite input, layout
    // surprise, gamma overflow) also falls back.
    if allow_fast
        && !exact
        && fast_gemm_enabled()
        && batch >= FAST_GEMM_MIN_ROWS
        && batch
            .checked_mul(out_dim)
            .and_then(|v| v.checked_mul(in_dim))
            .is_some_and(|v| v > FAST_GEMM_MIN_VOLUME)
    {
        if let Some(out) =
            try_eval_linear_fast(linear, x_lo, x_hi, batch, in_dim, include_bias, &out_shape)
        {
            return Ok(out);
        }
    }

    // Per term the unrolled interval dot performs two endpoint products and
    // three reductions (lo/hi/magnitude). Six leaves margin for selection;
    // the fixed charge covers lane assembly, optional bias, and error/end
    // point assembly. Both are unused on the exact path (the error IS zero),
    // and `gamma_n` is not even consulted there so an exact reduction can never
    // be refused for length.
    let widening = if exact {
        None
    } else {
        let gamma_terms = in_dim.checked_add(2).ok_or_else(|| {
            NyError::InvalidSpec("f64 cell eval: linear term count overflow".to_string())
        })?;
        Some((
            gamma_n(gamma_terms)?,
            reduction_operation_count(in_dim, 6, 32, "Linear")?,
        ))
    };
    let mut out_lo = vec![0.0f64; batch * out_dim];
    let mut out_hi = vec![0.0f64; batch * out_dim];
    for row in 0..batch {
        let x_base = row * in_dim;
        let xl_row = &x_lo[x_base..x_base + in_dim];
        let xu_row = &x_hi[x_base..x_base + in_dim];
        for o in 0..out_dim {
            let w_row = &weight[o * in_dim..(o + 1) * in_dim];
            let (mut lo, mut hi, mut abs) = interval_dot_f32w(w_row, xl_row, xu_row);
            if include_bias {
                if let Some(bias) = linear.bias.as_ref() {
                    let b = f32_to_f64_exact_for_bounds(bias[o]);
                    lo += b;
                    hi += b;
                    abs += b.abs();
                }
            }
            match widening {
                // Exact reduction: `lo`/`hi` ARE the real-arithmetic endpoints.
                None => {
                    out_lo[row * out_dim + o] = lo;
                    out_hi[row * out_dim + o] = hi;
                }
                Some((gamma, underflow_operations)) => {
                    let err = product_sum_error_bound(gamma, abs, underflow_operations)?;
                    out_lo[row * out_dim + o] = (lo - err).next_down();
                    out_hi[row * out_dim + o] = (hi + err).next_up();
                }
            }
        }
    }
    Ok(Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&out_shape), out_lo)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: linear out: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&out_shape), out_hi)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: linear out: {e}")))?,
    })
}

/// Interval dot product of one f32 weight row against an f64 interval
/// vector: returns `(lo, hi, abs)` with `lo = Σ min(w·xl, w·xu)`,
/// `hi = Σ max(w·xl, w·xu)`, `abs = Σ max(|pl|, |pu|)` — exactly the
/// quantities the Higham-widened scalar Linear needs.
///
/// 4-way unrolled with independent accumulator lanes (breaks the FP add
/// dependency chain: measured ~3x on the mscn m<=6, k=n=2048 shapes, where
/// this loop IS the f64 cell's per-node cost). SOUNDNESS: re-associating
/// the sum changes only WHICH nearest-rounded partial sums occur; the
/// Higham bound `gamma_n * Σ|terms|` applied by the caller covers EVERY
/// summation order (each term passes through <= n roundings), so the
/// widened interval still encloses the exact value. The branchless
/// `min/max` selects the same two products as the old `w >= 0.0` branch
/// for every non-NaN input.
#[inline]
fn interval_dot_f32w(w_row: &[f32], xl: &[f64], xu: &[f64]) -> (f64, f64, f64) {
    const LANES: usize = 4;
    let mut lo = [0.0f64; LANES];
    let mut hi = [0.0f64; LANES];
    let mut abs = [0.0f64; LANES];
    let (wc, w_rem) = w_row.as_chunks::<LANES>();
    let (xlc, xl_rem) = xl.as_chunks::<LANES>();
    let (xuc, xu_rem) = xu.as_chunks::<LANES>();
    for ((w4, l4), u4) in wc.iter().zip(xlc).zip(xuc) {
        for lane in 0..LANES {
            let w = f32_to_f64_exact_for_bounds(w4[lane]);
            let a = w * l4[lane];
            let b = w * u4[lane];
            let (pl, pu) = if a < b { (a, b) } else { (b, a) };
            lo[lane] += pl;
            hi[lane] += pu;
            abs[lane] += pl.abs().max(pu.abs());
        }
    }
    for ((&w, &l), &u) in w_rem.iter().zip(xl_rem.iter()).zip(xu_rem.iter()) {
        let w = f32_to_f64_exact_for_bounds(w);
        let a = w * l;
        let b = w * u;
        let (pl, pu) = if a < b { (a, b) } else { (b, a) };
        lo[0] += pl;
        hi[0] += pu;
        abs[0] += pl.abs().max(pu.abs());
    }
    (
        (lo[0] + lo[1]) + (lo[2] + lo[3]),
        (hi[0] + hi[1]) + (hi[2] + hi[3]),
        (abs[0] + abs[1]) + (abs[2] + abs[3]),
    )
}

/// Fast Linear via the Rump midpoint-radius GEMM kernel
/// (`super::graph_ibp_f64_gemm`, soundness argument there): `[x] @ W^T` with
/// the exact f64 weight as a POINT operand, bias folded afterwards with 1-ulp
/// outward widening per endpoint (`fl(lo + b)` is within 0.5 ulp of the real
/// sum, and `b` converts f32 -> f64 exactly). Returns `None` to fall back to
/// the scalar Higham loop.
fn try_eval_linear_fast(
    linear: &crate::layers::LinearLayer,
    x_lo: &[f64],
    x_hi: &[f64],
    batch: usize,
    in_dim: usize,
    include_bias: bool,
    out_shape: &[usize],
) -> Option<Interval64> {
    if std::env::var("NY_F64_BLAS_LOG").is_ok_and(|v| v == "1") {
        let (out_dim, _) = linear.weight.dim();
        eprintln!("NY_F64_BLAS linear m={batch} k={in_dim} n={out_dim}");
    }
    let a_lo = ArrayView2::from_shape((batch, in_dim), x_lo).ok()?;
    let a_hi = ArrayView2::from_shape((batch, in_dim), x_hi).ok()?;
    // Exact f32 -> f64 transpose of the weight: a point interval [wt, wt].
    let wt = linear.weight.t().map(|&w| f32_to_f64_exact_for_bounds(w));
    let (mut lo, mut hi) = rump_interval_matmul(a_lo, a_hi, wt.view(), wt.view())?;
    if include_bias {
        if let Some(bias) = linear.bias.as_ref() {
            for mut row_lo in lo.rows_mut() {
                for (l, &b) in row_lo.iter_mut().zip(bias.iter()) {
                    *l = (*l + f32_to_f64_exact_for_bounds(b)).next_down();
                }
            }
            for mut row_hi in hi.rows_mut() {
                for (h, &b) in row_hi.iter_mut().zip(bias.iter()) {
                    *h = (*h + f32_to_f64_exact_for_bounds(b)).next_up();
                }
            }
        }
    }
    Some(Interval64 {
        lower: lo.into_dyn().into_shape_with_order(IxDyn(out_shape)).ok()?,
        upper: hi.into_dyn().into_shape_with_order(IxDyn(out_shape)).ok()?,
    })
}

/// ReduceSum over resolved axes: each axis sum accumulates in f64
/// round-to-nearest and widens by the Higham bound `gamma_n * sum(|terms|)`
/// plus 1 ulp per endpoint — enclosing every rounding of the summation.
pub(super) fn eval_reduce_sum(
    reduce: &crate::layers::ReduceSumLayer,
    x: &Interval64,
) -> Result<Interval64> {
    let ndim = x.shape().len();
    let mut axes: Vec<usize> = if reduce.axes.is_empty() {
        (0..ndim).collect()
    } else {
        reduce
            .axes
            .iter()
            .map(|&axis| resolve_axis_i64(axis, ndim, "ReduceSum"))
            .collect::<Result<Vec<_>>>()?
    };
    axes.sort_unstable();
    let n_before = axes.len();
    axes.dedup();
    if axes.len() != n_before {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: duplicate ReduceSum axes".to_string(),
        ));
    }

    let mut lo = x.lower.clone();
    let mut hi = x.upper.clone();
    // Descending so earlier reductions don't shift later axis indices.
    for &axis in axes.iter().rev() {
        let n = lo.shape()[axis];
        let gamma_terms = n.checked_add(1).ok_or_else(|| {
            NyError::InvalidSpec("f64 cell eval: ReduceSum term count overflow".to_string())
        })?;
        let gamma = gamma_n(gamma_terms)?;
        // Each term is added both to the value and its magnitude reduction.
        // The fixed charge covers construction and endpoint assembly.
        let underflow_operations = reduction_operation_count(n, 2, 8, "ReduceSum")?;
        let underflow_floor = operation_underflow_floor(underflow_operations)?;
        let sum_widened = |arr: &ArrayD<f64>, is_lower: bool| -> ArrayD<f64> {
            arr.map_axis(ndarray::Axis(axis), |lane| {
                let mut s = 0.0f64;
                let mut abs = 0.0f64;
                for &v in lane.iter() {
                    s += v;
                    abs += v.abs();
                }
                let relative = widen_up_n(gamma * abs, 2);
                let err = widen_up_n(relative + underflow_floor, 2);
                if is_lower {
                    (s - err).next_down()
                } else {
                    (s + err).next_up()
                }
            })
        };
        let new_lo = sum_widened(&lo, true);
        let new_hi = sum_widened(&hi, false);
        if reduce.keepdims {
            let mut shape = new_lo.shape().to_vec();
            shape.insert(axis, 1);
            lo = new_lo.into_shape_with_order(IxDyn(&shape)).map_err(|e| {
                NyError::InvalidSpec(format!("f64 cell eval: ReduceSum keepdims: {e}"))
            })?;
            hi = new_hi.into_shape_with_order(IxDyn(&shape)).map_err(|e| {
                NyError::InvalidSpec(format!("f64 cell eval: ReduceSum keepdims: {e}"))
            })?;
        } else {
            lo = new_lo;
            hi = new_hi;
        }
    }
    Ok(Interval64 {
        lower: lo,
        upper: hi,
    })
}

/// Live-x-live rank-2 interval matmul `[m,k] @ [k,n]` (honoring the layer's
/// optional B-transpose and scale). Large shapes take the Rump
/// midpoint-radius probed-thread kernel (#f64-blas-gemm, kill-switch
/// `NY_F64_BLAS=0`); everything else — and every case the fast kernel declines
/// — takes the scalar corner-product path. Both ENCLOSE the true interval
/// product.
pub(super) fn eval_matmul(
    layer: &crate::layers::MatMulLayer,
    a: &Interval64,
    b: &Interval64,
) -> Result<Interval64> {
    if let Some(out) = try_eval_matmul_fast(layer, a, b) {
        return Ok(out);
    }
    eval_matmul_scalar(layer, a, b)
}

/// Fast rank-2 MatMul via the Rump midpoint-radius GEMM kernel; the optional
/// scale applies as one interval product widened 1 ulp (same tail as the
/// scalar path). `None` = use the scalar path (small shape, kill-switch,
/// shape problem — reported by the scalar path — or non-finite input).
fn try_eval_matmul_fast(
    layer: &crate::layers::MatMulLayer,
    a: &Interval64,
    b: &Interval64,
) -> Option<Interval64> {
    if !fast_gemm_enabled() || a.shape().len() != 2 || b.shape().len() != 2 {
        return None;
    }
    let transpose_b = layer.transpose_b();
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = if transpose_b {
        (b.shape()[1], b.shape()[0])
    } else {
        (b.shape()[0], b.shape()[1])
    };
    if k != k2
        || m < FAST_GEMM_MIN_ROWS
        || m.checked_mul(n)
            .and_then(|v| v.checked_mul(k))
            .is_none_or(|v| v <= FAST_GEMM_MIN_VOLUME)
    {
        return None;
    }
    fn view2(arr: &ArrayD<f64>) -> Option<ArrayView2<'_, f64>> {
        arr.view().into_dimensionality::<Ix2>().ok()
    }
    let (a_lo, a_hi) = (view2(&a.lower)?, view2(&a.upper)?);
    let (mut b_lo, mut b_hi) = (view2(&b.lower)?, view2(&b.upper)?);
    if transpose_b {
        b_lo = b_lo.reversed_axes();
        b_hi = b_hi.reversed_axes();
    }
    let (mut lo, mut hi) = rump_interval_matmul(a_lo, a_hi, b_lo, b_hi)?;
    if let Some(s) = layer.scale().map(f64::from) {
        for (l, h) in lo.iter_mut().zip(hi.iter_mut()) {
            let (nl, nh) = interval_mul(*l, *h, s, s);
            let (nl, nh) = widen1(nl, nh);
            *l = nl;
            *h = nh;
        }
    }
    Some(Interval64 {
        lower: lo.into_dyn(),
        upper: hi.into_dyn(),
    })
}

/// Scalar rank-2 interval matmul: per output element, k interval corner
/// products accumulate in f64 and widen by the Higham bound like
/// `eval_linear`; the optional scale applies as one more interval product
/// widened 1 ulp. Fails closed on other ranks. This is the reference path
/// the fast kernel is soundness-tested against.
pub(super) fn eval_matmul_scalar(
    layer: &crate::layers::MatMulLayer,
    a: &Interval64,
    b: &Interval64,
) -> Result<Interval64> {
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(NyError::UnsupportedOp(format!(
            "f64 cell eval: MatMul supports rank-2 x rank-2 only (got {:?} x {:?})",
            a.shape(),
            b.shape()
        )));
    }
    let transpose_b = layer.transpose_b();
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = if transpose_b {
        (b.shape()[1], b.shape()[0])
    } else {
        (b.shape()[0], b.shape()[1])
    };
    if k != k2 {
        return Err(NyError::ShapeMismatch {
            expected: vec![k],
            got: vec![k2],
        });
    }
    let scale = layer.scale().map(f64::from);

    let contiguous = |arr: &ArrayD<f64>| -> Result<Vec<f64>> { Ok(arr.iter().copied().collect()) };
    let a_lo = contiguous(&a.lower)?;
    let a_hi = contiguous(&a.upper)?;
    let b_lo = contiguous(&b.lower)?;
    let b_hi = contiguous(&b.upper)?;
    // Flat index of B[l, j] honoring the optional transpose.
    let bidx = |l: usize, j: usize| if transpose_b { j * k + l } else { l * n + j };

    let gamma_terms = k.checked_add(2).ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: MatMul term count overflow".to_string())
    })?;
    let gamma = gamma_n(gamma_terms)?;
    // `interval_mul` evaluates four endpoint products per term; lo, hi, and
    // magnitude each perform a reduction add. Eight operations per term plus
    // fixed endpoint/error assembly is conservative.
    let underflow_operations = reduction_operation_count(k, 8, 32, "MatMul")?;
    let mut out_lo = vec![0.0f64; m * n];
    let mut out_hi = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut lo = 0.0f64;
            let mut hi = 0.0f64;
            let mut abs = 0.0f64;
            for l in 0..k {
                let (pl, pu) = interval_mul(
                    a_lo[i * k + l],
                    a_hi[i * k + l],
                    b_lo[bidx(l, j)],
                    b_hi[bidx(l, j)],
                );
                lo += pl;
                hi += pu;
                abs += pl.abs().max(pu.abs());
            }
            let err = product_sum_error_bound(gamma, abs, underflow_operations)?;
            let (mut lo, mut hi) = ((lo - err).next_down(), (hi + err).next_up());
            if let Some(s) = scale {
                (lo, hi) = interval_mul(lo, hi, s, s);
                (lo, hi) = widen1(lo, hi);
            }
            out_lo[i * n + j] = lo;
            out_hi[i * n + j] = hi;
        }
    }
    Ok(Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&[m, n]), out_lo)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: matmul out: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&[m, n]), out_hi)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: matmul out: {e}")))?,
    })
}

fn eval_conv2d(conv: &crate::layers::Conv2dLayer, x: &Interval64) -> Result<Interval64> {
    let shape = x.shape().to_vec();
    let ndim = shape.len();
    if !(3..=4).contains(&ndim) {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: Conv2d needs rank 3 or 4".to_string(),
        ));
    }
    let has_batch = ndim == 4;
    let batch = if has_batch { shape[0] } else { 1 };
    let (in_c, in_h, in_w) = if has_batch {
        (shape[1], shape[2], shape[3])
    } else {
        (shape[0], shape[1], shape[2])
    };
    let kshape = conv.kernel.shape();
    let (out_c, kc, kh, kw) = (kshape[0], kshape[1], kshape[2], kshape[3]);
    let groups = conv.groups;
    if in_c != kc * groups || out_c % groups != 0 {
        return Err(NyError::ShapeMismatch {
            expected: vec![kc * groups],
            got: vec![in_c],
        });
    }
    let (sh, sw) = conv.stride;
    let (ph, pw) = conv.padding;
    let (dh, dw) = conv.dilation;
    let eff_kh = (kh - 1) * dh + 1;
    let eff_kw = (kw - 1) * dw + 1;
    let out_h = (in_h + 2 * ph)
        .checked_sub(eff_kh)
        .ok_or_else(|| NyError::InvalidSpec("f64 cell eval: conv kernel too large".to_string()))?
        / sh
        + 1;
    let out_w = (in_w + 2 * pw)
        .checked_sub(eff_kw)
        .ok_or_else(|| NyError::InvalidSpec("f64 cell eval: conv kernel too large".to_string()))?
        / sw
        + 1;
    let out_c_per_group = out_c / groups;

    let product_terms = kc
        .checked_mul(kh)
        .and_then(|terms| terms.checked_mul(kw))
        .ok_or_else(|| {
            NyError::InvalidSpec("f64 cell eval: Conv2d term count overflow".to_string())
        })?;
    let gamma_terms = product_terms.checked_add(2).ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: Conv2d term count overflow".to_string())
    })?;
    let gamma = gamma_n(gamma_terms)?;
    // Two endpoint products and three reductions occur per live kernel term;
    // six operations plus a fixed bias/assembly charge also covers padded or
    // zero-weight terms without depending on data-dependent loop counts.
    let underflow_operations = reduction_operation_count(product_terms, 6, 32, "Conv2d")?;

    // Flat standard-layout copies for fast indexing.
    let x_lo_owned = x.lower.as_standard_layout();
    let x_hi_owned = x.upper.as_standard_layout();
    let x_lo = x_lo_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: conv input not contiguous".to_string())
    })?;
    let x_hi = x_hi_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: conv input not contiguous".to_string())
    })?;
    let kernel_owned = conv.kernel.as_standard_layout();
    let kernel = kernel_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: conv kernel not contiguous".to_string())
    })?;

    let out_shape: Vec<usize> = if has_batch {
        vec![batch, out_c, out_h, out_w]
    } else {
        vec![out_c, out_h, out_w]
    };
    let out_len = batch * out_c * out_h * out_w;
    let mut out_lo = vec![0.0f64; out_len];
    let mut out_hi = vec![0.0f64; out_len];

    let in_hw = in_h * in_w;
    for b in 0..batch {
        let in_base_b = b * in_c * in_hw;
        for oc in 0..out_c {
            let group = oc / out_c_per_group;
            let ic_base = group * kc;
            let k_base_oc = oc * kc * kh * kw;
            let bias = conv
                .bias
                .as_ref()
                .map(|bias| f32_to_f64_exact_for_bounds(bias[oc]))
                .unwrap_or(0.0);
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut lo = bias;
                    let mut hi = bias;
                    let mut abs = bias.abs();
                    for ic_off in 0..kc {
                        let ic = ic_base + ic_off;
                        let in_base = in_base_b + ic * in_hw;
                        let k_base = k_base_oc + ic_off * kh * kw;
                        for r in 0..kh {
                            let ih = (oh * sh + r * dh) as isize - ph as isize;
                            if ih < 0 || ih >= in_h as isize {
                                continue;
                            }
                            let row_base = in_base + ih as usize * in_w;
                            let k_row = k_base + r * kw;
                            for c in 0..kw {
                                let iw = (ow * sw + c * dw) as isize - pw as isize;
                                if iw < 0 || iw >= in_w as isize {
                                    continue;
                                }
                                let w = f32_to_f64_exact_for_bounds(kernel[k_row + c]);
                                if w == 0.0 {
                                    continue;
                                }
                                let xi = row_base + iw as usize;
                                let (xl, xu) = (x_lo[xi], x_hi[xi]);
                                let (pl, pu) = if w >= 0.0 {
                                    (w * xl, w * xu)
                                } else {
                                    (w * xu, w * xl)
                                };
                                lo += pl;
                                hi += pu;
                                abs += pl.abs().max(pu.abs());
                            }
                        }
                    }
                    let err = product_sum_error_bound(gamma, abs, underflow_operations)?;
                    let out_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
                    out_lo[out_idx] = (lo - err).next_down();
                    out_hi[out_idx] = (hi + err).next_up();
                }
            }
        }
    }
    Ok(Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&out_shape), out_lo)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: conv out: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&out_shape), out_hi)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: conv out: {e}")))?,
    })
}

/// Sound f64 interval ConvTranspose2d (ONNX ConvTranspose), Higham-widened.
///
/// Closes W0.2 reachability gap (b1): every cgan sat row runs 4-5
/// `ConvTranspose` nodes, so the whole class fail-closed here.
///
/// Geometry is the SCATTER form of [`conv2d_transpose_forward`] verbatim —
/// `out[oc, ih*sh + ki*dh - ph, iw*sw + kj*dw - pw] += x[ic, ih, iw] *
/// K[ic, oc, ki, kj]`, with the same `< out_h` / `< out_w` guard and the same
/// `out_h = (in_h-1)*sh + eff_kh + oph - 2*ph` output extent (`output_padding`
/// cells receive only bias). Scatter rather than gather because the gather form
/// would iterate `out_h*out_w >= sh*sw * in_h*in_w` output positions and reject
/// most `(ki, kj)` taps on the stride residue.
///
/// SOUNDNESS. Per output cell the accumulation is a dot product of at most
/// `in_c*kh*kw` interval-times-scalar terms, so it is charged exactly as
/// [`eval_conv2d`] charges its `kc*kh*kw`: monotone endpoint products by the
/// sign of the (exactly f64-representable) weight, naive f64 endpoint sums, and
/// an outward `gamma_n * sum|terms| + ops*2^-1074` widening. `in_c*kh*kw` is an
/// UPPER bound on the live tap count (the stride residue only removes taps) and
/// `gamma_n` is increasing in `n`, so over-counting is conservative. Unlike the
/// f32 forward this arm never uses GEMM: the certificate must not depend on a
/// summation order, and the Higham term is order-independent by construction.
fn eval_conv_transpose2d(
    conv: &crate::layers::ConvTranspose2dLayer,
    x: &Interval64,
) -> Result<Interval64> {
    let (in_c_kernel, out_c) = conv.validate_geometry()?;
    let shape = x.shape().to_vec();
    let ndim = shape.len();
    if !(3..=4).contains(&ndim) {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: ConvTranspose2d needs rank 3 or 4".to_string(),
        ));
    }
    let has_batch = ndim == 4;
    let batch = if has_batch { shape[0] } else { 1 };
    let (in_c, in_h, in_w) = if has_batch {
        (shape[1], shape[2], shape[3])
    } else {
        (shape[0], shape[1], shape[2])
    };
    if in_c != in_c_kernel {
        return Err(NyError::ShapeMismatch {
            expected: vec![in_c_kernel],
            got: vec![in_c],
        });
    }
    let kshape = conv.kernel.shape();
    let (kh, kw) = (kshape[2], kshape[3]);
    let (sh, sw) = conv.stride;
    let (ph, pw) = conv.padding;
    let (dh, dw) = conv.dilation;
    let (oph, opw) = conv.output_padding;
    // Checked throughout, as in `conv2d_transpose_forward`: a hostile ONNX can
    // pick pads/dilations that overflow or underflow these extents, and a
    // wrapped extent would silently mis-size the output rather than fail closed.
    let effective = |k: usize, d: usize| -> Result<usize> {
        k.checked_sub(1)
            .and_then(|extent| extent.checked_mul(d))
            .and_then(|extent| extent.checked_add(1))
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "f64 cell eval: ConvTranspose2d effective kernel overflow".to_string(),
                )
            })
    };
    let eff_kh = effective(kh, dh)?;
    let eff_kw = effective(kw, dw)?;

    let dim_out = |n: usize, s: usize, eff: usize, op: usize, p: usize| -> Result<usize> {
        n.checked_sub(1)
            .and_then(|v| v.checked_mul(s))
            .and_then(|v| v.checked_add(eff))
            .and_then(|v| v.checked_add(op))
            .and_then(|v| p.checked_mul(2).and_then(|double| v.checked_sub(double)))
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "f64 cell eval: ConvTranspose2d output extent underflow".to_string(),
                )
            })
    };
    let out_h = dim_out(in_h, sh, eff_kh, oph, ph)?;
    let out_w = dim_out(in_w, sw, eff_kw, opw, pw)?;

    // Conservative UPPER bound on the taps summed into one output cell.
    let product_terms = in_c
        .checked_mul(kh)
        .and_then(|terms| terms.checked_mul(kw))
        .ok_or_else(|| {
            NyError::InvalidSpec("f64 cell eval: ConvTranspose2d term count overflow".to_string())
        })?;
    let gamma_terms = product_terms.checked_add(2).ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: ConvTranspose2d term count overflow".to_string())
    })?;
    let gamma = gamma_n(gamma_terms)?;
    // Same charge model as Conv2d: two endpoint products and three reductions
    // per tap, plus a fixed bias/assembly allowance.
    let underflow_operations = reduction_operation_count(product_terms, 6, 32, "ConvTranspose2d")?;

    let x_lo_owned = x.lower.as_standard_layout();
    let x_hi_owned = x.upper.as_standard_layout();
    let x_lo = x_lo_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: convT input not contiguous".to_string())
    })?;
    let x_hi = x_hi_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: convT input not contiguous".to_string())
    })?;
    let kernel_owned = conv.kernel.as_standard_layout();
    let kernel = kernel_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: convT kernel not contiguous".to_string())
    })?;

    let out_shape: Vec<usize> = if has_batch {
        vec![batch, out_c, out_h, out_w]
    } else {
        vec![out_c, out_h, out_w]
    };
    let out_spatial = out_h.checked_mul(out_w).ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: ConvTranspose2d output spatial overflow".to_string())
    })?;
    let out_len = batch
        .checked_mul(out_c)
        .and_then(|v| v.checked_mul(out_spatial))
        .ok_or_else(|| {
            NyError::InvalidSpec("f64 cell eval: ConvTranspose2d output size overflow".to_string())
        })?;

    // Bias seeds every output cell (including the output_padding cells, which
    // receive no input contribution) and also seeds the magnitude accumulator.
    let mut out_lo = vec![0.0f64; out_len];
    let mut out_hi = vec![0.0f64; out_len];
    let mut out_abs = vec![0.0f64; out_len];
    if let Some(bias) = conv.bias.as_ref() {
        if bias.len() != out_c {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_c],
                got: vec![bias.len()],
            });
        }
        for b in 0..batch {
            for oc in 0..out_c {
                let bv = f32_to_f64_exact_for_bounds(bias[oc]);
                let base = (b * out_c + oc) * out_spatial;
                for cell in 0..out_spatial {
                    out_lo[base + cell] = bv;
                    out_hi[base + cell] = bv;
                    out_abs[base + cell] = bv.abs();
                }
            }
        }
    }

    // The (kernel tap -> output position) map depends only on the spatial input
    // index, never on the channel, so resolve it once per row/column instead of
    // re-deriving it inside the channel loops. `taps_h[ih]` lists the (ki, oh)
    // pairs the scatter loop would have accepted; the same `checked_sub` +
    // `< out_h` guard decides membership, so the visited (tap, cell) set is
    // identical to the f32 forward's.
    let taps_h: Vec<Vec<(usize, usize)>> = (0..in_h)
        .map(|ih| {
            (0..kh)
                .filter_map(|ki| {
                    (ih * sh + ki * dh)
                        .checked_sub(ph)
                        .filter(|&index| index < out_h)
                        .map(|oh| (ki, oh))
                })
                .collect()
        })
        .collect();
    let taps_w: Vec<Vec<(usize, usize)>> = (0..in_w)
        .map(|iw| {
            (0..kw)
                .filter_map(|kj| {
                    (iw * sw + kj * dw)
                        .checked_sub(pw)
                        .filter(|&index| index < out_w)
                        .map(|ow| (kj, ow))
                })
                .collect()
        })
        .collect();

    let in_hw = in_h * in_w;
    let kernel_spatial = kh * kw;
    for b in 0..batch {
        let in_base_b = b * in_c * in_hw;
        let out_base_b = b * out_c * out_spatial;
        for ic in 0..in_c {
            let in_base = in_base_b + ic * in_hw;
            let k_base_ic = ic * out_c * kernel_spatial;
            for ih in 0..in_h {
                for iw in 0..in_w {
                    let xi = in_base + ih * in_w + iw;
                    let (xl, xu) = (x_lo[xi], x_hi[xi]);
                    for oc in 0..out_c {
                        let k_base = k_base_ic + oc * kernel_spatial;
                        let out_base_oc = out_base_b + oc * out_spatial;
                        for &(ki, oh) in &taps_h[ih] {
                            let out_row = out_base_oc + oh * out_w;
                            let k_row = k_base + ki * kw;
                            for &(kj, ow) in &taps_w[iw] {
                                let w = f32_to_f64_exact_for_bounds(kernel[k_row + kj]);
                                if w == 0.0 {
                                    continue;
                                }
                                let (pl, pu) = if w >= 0.0 {
                                    (w * xl, w * xu)
                                } else {
                                    (w * xu, w * xl)
                                };
                                let oi = out_row + ow;
                                out_lo[oi] += pl;
                                out_hi[oi] += pu;
                                out_abs[oi] += pl.abs().max(pu.abs());
                            }
                        }
                    }
                }
            }
        }
    }

    for cell in 0..out_len {
        let err = product_sum_error_bound(gamma, out_abs[cell], underflow_operations)?;
        out_lo[cell] = (out_lo[cell] - err).next_down();
        out_hi[cell] = (out_hi[cell] + err).next_up();
    }

    Ok(Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&out_shape), out_lo)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: convT out: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&out_shape), out_hi)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: convT out: {e}")))?,
    })
}

/// Sound f64 interval BatchNorm (inference form), per channel.
///
/// Closes W0.2 reachability gap (b2): cgan carries 7-9 `BatchNormalization`
/// nodes that survive load (`batch_norm_fold.rs` only fuses the Conv+BN pairs),
/// so the class fail-closed here even with `ConvTranspose2d` in place.
///
/// The loaded layer IS the affine `y = x * scale + bias` over its stored f32
/// per-channel coefficients; `expanded_affine_parameters` is the single
/// shape-aware source of truth for the channel-axis heuristic (duplicating it
/// here would mistake a squeezed `[C, H, W]` map's leading axis for a batch
/// axis). f32 -> f64 is exact, so the coefficients enter as POINTS and the
/// evaluation is the monotone endpoint pairing by `sign(scale)`, widened
/// outward by 4 ulps: 1 for the product, 1 for the `+ bias`, and 2 of slack.
///
/// SCOPE, stated plainly (W0.2 section 3 / section 7). Like every other arm in
/// this file, this one encloses the exact real-arithmetic image of the layer
/// **as NY loaded it** — here the f32-precomputed `scale = ny/sqrt(var+eps)` and
/// `bias = beta - mean*scale`. It does NOT enclose the raw-ONNX BatchNorm, which
/// computes that affine from the unrounded statistics; the layer carries
/// `scale_err`/`bias_err` bounds for exactly that gap (the f32 IBP path folds
/// them), and a graph-fidelity gate that wants the raw-ONNX claim must fold them
/// here too. They are deliberately NOT folded in: they are ~`ulp(scale)`, i.e.
/// ~6e-8 relative, which is ~2 orders of magnitude WIDER than the whole measured
/// cgan enclosure (1.66e-6 relative against 4.6e2 headroom), so folding them
/// silently converts a tight certificate into a useless one; and
/// `next_up_f32(0)` leaves them incomplete against the exact real coefficient
/// anyway, so folding them would not buy a provable raw-ONNX claim.
fn eval_batch_norm(bn: &crate::layers::BatchNormLayer, x: &Interval64) -> Result<Interval64> {
    let shape = x.shape().to_vec();
    let total: usize = shape.iter().product();
    let (scale, bias, _scale_err, _bias_err) = bn.expanded_affine_parameters(&shape, total)?;
    if scale.len() != total || bias.len() != total {
        return Err(NyError::ShapeMismatch {
            expected: vec![total],
            got: vec![scale.len()],
        });
    }

    let x_lo_owned = x.lower.as_standard_layout();
    let x_hi_owned = x.upper.as_standard_layout();
    let x_lo = x_lo_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: BatchNorm input not contiguous".to_string())
    })?;
    let x_hi = x_hi_owned.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: BatchNorm input not contiguous".to_string())
    })?;

    let mut out_lo = Vec::with_capacity(total);
    let mut out_hi = Vec::with_capacity(total);
    for i in 0..total {
        let s = f32_to_f64_exact_for_bounds(scale[i]);
        let b = f32_to_f64_exact_for_bounds(bias[i]);
        if !s.is_finite() || !b.is_finite() {
            return Err(NyError::UnsupportedOp(
                "f64 cell eval: BatchNorm coefficient not finite".to_string(),
            ));
        }
        let (xl, xh) = (x_lo[i], x_hi[i]);
        let (lo, hi) = if s >= 0.0 {
            (s * xl + b, s * xh + b)
        } else {
            (s * xh + b, s * xl + b)
        };
        out_lo.push(widen_down_n(lo, 4));
        out_hi.push(widen_up_n(hi, 4));
    }

    Ok(Interval64 {
        lower: ArrayD::from_shape_vec(IxDyn(&shape), out_lo)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: batchnorm out: {e}")))?,
        upper: ArrayD::from_shape_vec(IxDyn(&shape), out_hi)
            .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: batchnorm out: {e}")))?,
    })
}

/// ScatterND with embedded constant data/updates and a dynamic indices input
/// whose per-cell interval is SINGLETON integers (the cctsdb cell case).
///
/// Rows with any coordinate out of range are SKIPPED — exactly matching the
/// clamped-window semantics of the source graph (design B4: the unclamped
/// static-max window emits sentinel rows the true graph never writes).
/// Non-singleton coordinates fail closed.
fn eval_scatter_nd(
    scatter: &crate::layers::ScatterNdLayer,
    inputs: &[Interval64],
) -> Result<Interval64> {
    let data = scatter.data_constant().ok_or_else(|| {
        NyError::UnsupportedOp("f64 cell eval: ScatterND requires constant data".to_string())
    })?;
    let updates = scatter.updates_constant().ok_or_else(|| {
        NyError::UnsupportedOp("f64 cell eval: ScatterND requires constant updates".to_string())
    })?;
    if scatter.has_static_indices() || inputs.len() != 1 {
        return Err(NyError::UnsupportedOp(
            "f64 cell eval: ScatterND expects exactly the dynamic indices input".to_string(),
        ));
    }
    let indices = &inputs[0];

    let data_shape = data.shape().to_vec();
    let idx_shape = indices.shape().to_vec();
    let index_depth = *idx_shape.last().ok_or_else(|| {
        NyError::InvalidSpec("f64 cell eval: ScatterND indices rank 0".to_string())
    })?;
    if index_depth == 0 || index_depth > data_shape.len() {
        return Err(NyError::InvalidSpec(
            "f64 cell eval: ScatterND index depth out of range".to_string(),
        ));
    }
    let prefix_elems: usize = idx_shape[..idx_shape.len() - 1].iter().product();
    let remainder_shape = &data_shape[index_depth..];
    let slice_len: usize = remainder_shape.iter().product();
    let expected_updates: Vec<usize> = idx_shape[..idx_shape.len() - 1]
        .iter()
        .copied()
        .chain(remainder_shape.iter().copied())
        .collect();
    if updates.shape() != expected_updates.as_slice() {
        return Err(NyError::ShapeMismatch {
            expected: expected_updates,
            got: updates.shape().to_vec(),
        });
    }

    // Strides of data.
    let mut strides = vec![1usize; data_shape.len()];
    for i in (1..data_shape.len()).rev() {
        strides[i - 1] = strides[i] * data_shape[i];
    }

    let idx_lo_owned = indices.lower.as_standard_layout();
    let idx_hi_owned = indices.upper.as_standard_layout();
    let idx_lo = idx_lo_owned
        .as_slice()
        .ok_or_else(|| NyError::InvalidSpec("f64 cell eval: indices not contiguous".to_string()))?;
    let idx_hi = idx_hi_owned
        .as_slice()
        .ok_or_else(|| NyError::InvalidSpec("f64 cell eval: indices not contiguous".to_string()))?;
    let updates_flat: Vec<f64> = updates
        .iter()
        .map(|&v| f32_to_f64_exact_for_bounds(v))
        .collect();

    let mut out: Vec<f64> = data
        .iter()
        .map(|&v| f32_to_f64_exact_for_bounds(v))
        .collect();
    'rows: for row in 0..prefix_elems {
        let mut base = 0usize;
        for axis in 0..index_depth {
            let flat = row * index_depth + axis;
            let (lo, hi) = (idx_lo[flat], idx_hi[flat]);
            if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
                return Err(NyError::UnsupportedOp(
                    "f64 cell eval: ScatterND index interval not finite".to_string(),
                ));
            }
            // The TRUE coordinate is an integer (ONNX ScatterND indices are
            // int64; the source graph builds them from Cast-to-int outputs and
            // integer arithmetic) lying inside this sound enclosure. Per cell
            // the enclosure is only a few ulps wide (outward widening of the
            // affine coordinate chain), so its integer hull normally has
            // EXACTLY one member — that member IS the true coordinate.
            // Anything ambiguous fails closed.
            let ilo = lo.ceil();
            let ihi = hi.floor();
            if ilo != ihi {
                return Err(NyError::UnsupportedOp(
                    "f64 cell eval: ScatterND index interval not a unique integer".to_string(),
                ));
            }
            let axis_len = data_shape[axis] as i64;
            let mut coord = ilo as i64;
            if coord < 0 {
                coord += axis_len;
            }
            if coord < 0 || coord >= axis_len {
                // Out-of-range sentinel row (clamped window edge): no write.
                continue 'rows;
            }
            base += coord as usize * strides[axis];
        }
        for off in 0..slice_len {
            out[base + off] = updates_flat[row * slice_len + off];
        }
    }

    let arr = ArrayD::from_shape_vec(IxDyn(&data_shape), out)
        .map_err(|e| NyError::InvalidSpec(format!("f64 cell eval: scatter out: {e}")))?;
    Ok(Interval64 {
        lower: arr.clone(),
        upper: arr,
    })
}

#[cfg(test)]
mod tests {
    use super::super::core::graph::GraphNode;
    use super::*;
    use crate::layers::{
        AddConstantLayer, AddLayer, DivLayer, LinearLayer, MatMulLayer, MulBinaryLayer, ReLULayer,
        ReduceSumLayer, SigmoidLayer, SliceLayer, SubLayer, TanhLayer,
    };
    use ndarray::{arr1, arr2};
    use num_bigint::BigInt;
    use num_rational::BigRational;

    #[test]
    fn certified_exp_negative_contains_exact_alternating_brackets() {
        for a in [0.125_f64, 0.5, 1.0, 2.0, 10.0] {
            let mut y = a;
            let mut squarings = 0_u32;
            // Mirrors the range reduction inside `exp_neg_certified` bit for bit:
            // every `a` above is finite and in (0, 1075), so the exact halving
            // terminates. Restructuring the float condition here would stop the
            // oracle from reproducing the reduction it is checking.
            #[allow(clippy::while_float)]
            while y > 0.5 {
                y *= 0.5;
                squarings += 1;
            }
            let y = BigRational::from_float(y).expect("finite dyadic");
            let one = BigRational::from_integer(BigInt::from(1));
            let mut term = one.clone();
            let mut sum = one;
            let mut exact_lower = BigRational::from_integer(BigInt::from(0));
            let mut exact_upper = sum.clone();
            for k in 1_u32..=32 {
                term = term * &y / BigRational::from_integer(BigInt::from(k));
                if k % 2 == 1 {
                    sum -= &term;
                    exact_lower = sum.clone();
                } else {
                    sum += &term;
                    exact_upper = sum.clone();
                }
            }
            for _ in 0..squarings {
                exact_lower = &exact_lower * &exact_lower;
                exact_upper = &exact_upper * &exact_upper;
            }

            let (lower, upper) = exp_neg_certified(a);
            let lower = BigRational::from_float(lower).expect("finite lower");
            let upper = BigRational::from_float(upper).expect("finite upper");
            assert!(lower <= exact_lower, "a={a}: lower missed odd bracket");
            assert!(upper >= exact_upper, "a={a}: upper missed even bracket");
        }
    }

    #[test]
    fn certified_sigmoid_is_ordered_symmetric_and_contains_oracle() {
        for x in [
            -1075.0_f64,
            f64::NEG_INFINITY,
            -1000.0,
            -745.0,
            -710.0,
            -100.0,
            -20.0,
            -2.0,
            -0.5,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            0.5,
            2.0,
            20.0,
            100.0,
            710.0,
            745.0,
            1000.0,
            1075.0,
            f64::INFINITY,
        ] {
            let (lower, upper) = certified_sigmoid_f64(x);
            assert!(
                0.0 <= lower && lower <= upper && upper <= 1.0,
                "x={x}: [{lower}, {upper}]"
            );
            let oracle = stable_sigmoid_f64(x);
            assert!(
                lower <= oracle && oracle <= upper,
                "x={x}: oracle {oracle} escapes [{lower}, {upper}]"
            );

            let (negative_lower, negative_upper) = certified_sigmoid_f64(-x);
            let symmetric_lower = (1.0 - upper).next_down().max(0.0);
            let symmetric_upper = (1.0 - lower).next_up().min(1.0);
            assert!(
                negative_lower <= symmetric_upper && symmetric_lower <= negative_upper,
                "x={x}: reflected intervals do not overlap"
            );
        }

        let mut previous = certified_sigmoid_f64(-1000.0);
        for quarter in -3999_i32..=4000 {
            let x = f64::from(quarter) * 0.25;
            let current = certified_sigmoid_f64(x);
            let oracle = stable_sigmoid_f64(x);
            assert!(current.0 <= oracle && oracle <= current.1, "x={x}");
            assert!(
                previous.0 <= current.0 && previous.1 <= current.1,
                "certified sigmoid lost monotonicity at x={x}: {previous:?} -> {current:?}"
            );
            previous = current;
        }
    }

    #[test]
    fn interval64_decodes_binary32_subnormals_without_hardware_conversion() {
        let tiny = f32::from_bits(1);
        let interval = Interval64::from_f32(
            &arr1(&[tiny, -tiny]).into_dyn(),
            &arr1(&[tiny, -tiny]).into_dyn(),
        );
        let expected = f64::from_bits((u64::from(1023_u16 - 149)) << 52);
        assert_eq!(interval.lower[IxDyn(&[0])].to_bits(), expected.to_bits());
        assert_eq!(interval.lower[IxDyn(&[1])].to_bits(), (-expected).to_bits());
    }

    #[test]
    fn interval_mul_covers_sign_cases() {
        let (lo, hi) = interval_mul(-2.0, 3.0, -5.0, 7.0);
        assert_eq!(lo, -15.0); // 3 * -5
        assert_eq!(hi, 21.0); // 3 * 7
    }

    #[test]
    fn gamma_grows_with_terms() {
        let g10 = gamma_n(10).unwrap();
        let g1000 = gamma_n(1000).unwrap();
        assert!(g10 > 0.0 && g1000 > g10);
        assert!(g1000 < 1e-12);
    }

    #[test]
    fn scalar_linear_covers_multiple_products_that_each_underflow_to_zero() {
        let weight = f32::from_bits(1); // 2^-149
        let input_value = f64::from_bits(u64::from(1023_u16 - 926) << 52); // 2^-926
        let rounded_product = std::hint::black_box(f32_to_f64_exact_for_bounds(weight))
            * std::hint::black_box(input_value);
        assert_eq!(
            rounded_product.to_bits(),
            0,
            "each exact 2^-1075 product must tie to zero under round-to-nearest"
        );

        let linear =
            LinearLayer::new(arr2(&[[weight, weight, weight, weight]]), None).expect("linear");
        let input = Interval64::point(
            arr1(&[input_value, input_value, input_value, input_value]).into_dyn(),
        );
        let output = eval_linear_with_bias_scalar(&linear, &input, false).expect("scalar interval");
        let exact_sum = f64::from_bits(2); // 4 * 2^-1075 = 2^-1073
        assert!(
            output.lower[IxDyn(&[0])] <= exact_sum && exact_sum <= output.upper[IxDyn(&[0])],
            "absolute product floor must enclose {exact_sum:e} in [{:e}, {:e}]",
            output.lower[IxDyn(&[0])],
            output.upper[IxDyn(&[0])]
        );
    }

    #[test]
    fn scalar_linear_covers_subnormal_products_with_cancellation() {
        let tiny_weight = f32::from_bits(1); // 2^-149
        let input_value = f64::from_bits(u64::from(1023_u16 - 926) << 52); // 2^-926
        let weights = [
            tiny_weight,
            -tiny_weight,
            tiny_weight,
            tiny_weight,
            tiny_weight,
            tiny_weight,
        ];
        let linear = LinearLayer::new(arr2(&[weights]), None).expect("linear");
        let input = Interval64::point(
            arr1(&[
                input_value,
                input_value,
                input_value,
                input_value,
                input_value,
                input_value,
            ])
            .into_dyn(),
        );

        // Every exact product has magnitude 2^-1075 and rounds to signed
        // zero. Their real sum includes cancellation but is still
        // 4 * 2^-1075 = 2^-1073. A relative-only reduction bound sees zero;
        // the absolute multiply/add/assembly charge must carry the proof.
        let output = eval_linear_with_bias_scalar(&linear, &input, false).expect("scalar interval");
        let exact_sum = f64::from_bits(2);
        assert!(
            output.lower[IxDyn(&[0])] <= exact_sum && exact_sum <= output.upper[IxDyn(&[0])],
            "operation floor must enclose cancellation result {exact_sum:e} in [{:e}, {:e}]",
            output.lower[IxDyn(&[0])],
            output.upper[IxDyn(&[0])]
        );

        let operations = reduction_operation_count(weights.len(), 6, 32, "test").unwrap();
        assert!(
            operations >= 2 * weights.len() + 3,
            "the absolute charge must include products, reduction adds, and assembly"
        );
    }

    #[test]
    fn widen1_moves_outward() {
        let (lo, hi) = widen1(1.0, 1.0);
        assert!(lo < 1.0 && hi > 1.0);
    }

    // -----------------------------------------------------------------------
    // Integer-exactness path (#sat-relu-zero-margin)
    // -----------------------------------------------------------------------

    /// The certificate accepts integral operands inside the exact-integer range
    /// and REFUSES on every way out: a fractional weight, a fractional
    /// activation, a fractional bias, and a magnitude bound above 2^53.
    #[test]
    fn integer_exactness_certificate_accepts_only_provably_exact_reductions() {
        let ones = [1.0f64, 1.0, 1.0, 1.0];
        assert!(
            integer_exact_linear_reduction(&[1.0f32, -1.0, 2.0, -2.0], None, &ones, &ones, 4),
            "small integer weights x integer activations must certify"
        );
        assert!(
            !integer_exact_linear_reduction(&[1.0f32, -1.0, 2.5, -2.0], None, &ones, &ones, 4),
            "a fractional weight must refuse"
        );
        let frac = [1.0f64, 1.5, 1.0, 1.0];
        assert!(
            !integer_exact_linear_reduction(&[1.0f32; 4], None, &frac, &frac, 4),
            "a fractional activation must refuse"
        );
        assert!(
            !integer_exact_linear_reduction(&[1.0f32; 4], Some(&arr1(&[0.5f32])), &ones, &ones, 4),
            "a fractional bias must refuse"
        );
        // 4 * 2^52 * 1 = 2^54 > 2^53: integral, but partial sums leave the
        // exactly-representable range, so the certificate must refuse.
        let big = [(1u64 << 52) as f64; 4];
        assert!(
            !integer_exact_linear_reduction(&[1.0f32; 4], None, &big, &big, 4),
            "an over-2^53 reduction bound must refuse"
        );
        // Non-finite endpoints refuse rather than certify.
        let inf = [f64::INFINITY, 1.0, 1.0, 1.0];
        assert!(!integer_exact_linear_reduction(
            &[1.0f32; 4],
            None,
            &inf,
            &inf,
            4
        ));
    }

    /// A sat_relu-shaped gadget (`Gemm -> ReLU -> Gemm`, weights in {-1,1,2},
    /// integer biases) at a boolean corner: the enclosure must collapse to the
    /// EXACT integer value with width ZERO, so the spec's non-strict thresholds
    /// `Y_0 >= 1` / `Y_1 <= 0` are met with margin exactly 0 instead of the
    /// measured `-9.8e-15` the Higham widening produced.
    #[test]
    fn sat_relu_shaped_gadget_certifies_at_exactly_zero_margin() {
        // One clause row `h = ReLU(-x0 - x1 + 1)` (clause `x0 OR x1`), the
        // identity block `ReLU(x_j)` and the Booleanization block
        // `ReLU(2 x_j - 1)`, exactly the sat_relu compilation shape.
        let w1 = arr2(&[
            [-1.0f32, -1.0], // clause row
            [1.0, 0.0],      // identity x0
            [0.0, 1.0],      // identity x1
            [2.0, 0.0],      // boolean x0
            [0.0, 2.0],      // boolean x1
        ]);
        let b1 = arr1(&[1.0f32, 0.0, 0.0, -1.0, -1.0]);
        // Y_0 = 1 - sum(clauses); Y_1 = sum(x_j) - sum(ReLU(2 x_j - 1)).
        let w2 = arr2(&[[-1.0f32, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, -1.0, -1.0]]);
        let b2 = arr1(&[1.0f32, 0.0]);

        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "g0",
            Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
        ));
        g.add_node(GraphNode::new(
            "r",
            Layer::ReLU(ReLULayer),
            vec!["g0".to_string()],
        ));
        g.add_node(GraphNode::new(
            "g2",
            Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
            vec!["r".to_string()],
        ));
        g.set_output("g2");
        assert!(g.supports_ibp_f64_cell());

        for corner in [[1.0f64, 0.0], [0.0, 1.0], [1.0, 1.0]] {
            let out = g
                .propagate_ibp_f64_cell(&Interval64::point(arr1(&corner).into_dyn()))
                .expect("cell walk");
            let (y0_lo, y0_hi) = (out.lower[IxDyn(&[0])], out.upper[IxDyn(&[0])]);
            let (y1_lo, y1_hi) = (out.lower[IxDyn(&[1])], out.upper[IxDyn(&[1])]);
            assert_eq!(
                (y0_lo, y0_hi),
                (1.0, 1.0),
                "corner {corner:?}: Y_0 enclosure must be the exact point 1"
            );
            assert_eq!(
                (y1_lo, y1_hi),
                (0.0, 0.0),
                "corner {corner:?}: Y_1 enclosure must be the exact point 0"
            );
            // The spec margins (`Y_0 >= 1` worst side, `Y_1 <= 0` worst side).
            assert_eq!(y0_lo - 1.0, 0.0);
            assert_eq!(0.0 - y1_hi, 0.0);
        }

        // NEGATIVE control: a non-boolean (fractional) input leaves the exact
        // path, so the enclosure widens again — the exactness is a property of
        // the OPERANDS, never an unconditional tightening.
        let mid = g
            .propagate_ibp_f64_cell(&Interval64::point(arr1(&[0.5f64, 0.5]).into_dyn()))
            .expect("cell walk");
        assert!(
            mid.upper[IxDyn(&[1])] - mid.lower[IxDyn(&[1])] > 0.0,
            "fractional input must keep the Higham widening"
        );
    }

    /// The exact path must agree with an INDEPENDENT integer reference over the
    /// whole gadget-shaped operand space (random integer weights, biases and
    /// activations within the bound), and must still produce a valid enclosure.
    #[test]
    fn integer_exact_linear_matches_an_independent_integer_reference() {
        let mut rng = Rng(0x5a7_7e10);
        let (out_dim, in_dim) = (5usize, 7usize);
        for _ in 0..200 {
            let mut w = vec![0.0f32; out_dim * in_dim];
            for v in w.iter_mut() {
                *v = (rng.next_unit() * 5.0).floor() as f32 - 2.0; // -2..=2
            }
            let bias: Vec<f32> = (0..out_dim)
                .map(|_| (rng.next_unit() * 9.0).floor() as f32 - 4.0)
                .collect();
            let x: Vec<f64> = (0..in_dim)
                .map(|_| (rng.next_unit() * 7.0).floor())
                .collect();
            let linear = LinearLayer::new(
                ndarray::Array2::from_shape_vec((out_dim, in_dim), w.clone()).unwrap(),
                Some(arr1(&bias)),
            )
            .unwrap();
            let out = eval_linear_with_bias(
                &linear,
                &Interval64::point(ndarray::Array1::from(x.clone()).into_dyn()),
                true,
            )
            .expect("linear");
            for o in 0..out_dim {
                // i128 reference — no floating point involved.
                let mut acc: i128 = bias[o] as i128;
                for j in 0..in_dim {
                    acc += w[o * in_dim + j] as i128 * x[j] as i128;
                }
                let expect = acc as f64;
                assert_eq!(
                    (out.lower[IxDyn(&[o])], out.upper[IxDyn(&[o])]),
                    (expect, expect),
                    "row {o}: exact path must reproduce the integer reference {acc} as a point"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // f64 leaf-escalation op set (#f64-leaf): Sigmoid/ReduceSum/MatMul/N-D
    // Linear over an mscn-shaped DAG.
    // -----------------------------------------------------------------------

    /// Deterministic pseudo-random stream (xorshift) — no extra dev-dep.
    struct Rng(u64);
    impl Rng {
        fn next_unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn box64(lo: [f32; 4], hi: [f32; 4]) -> Interval64 {
        Interval64::from_f32(&arr1(&lo).into_dyn(), &arr1(&hi).into_dyn())
    }

    /// mscn-shaped DAG: input [4] --Linear(4->3)--> ReLU --Mul(slice of
    /// input)--> Add(slice) --ReduceSum--> Div(by 2.5 + ReduceSum(relu)) -->
    /// Sigmoid. Exercises Linear (N-D convention on rank-1), ReLU, MulBinary,
    /// Add, ReduceSum, Div, Sigmoid, Slice.
    fn build_mscn_like_graph() -> GraphNetwork {
        let w = arr2(&[
            [0.5f32, -1.25, 2.0, 0.75],
            [-0.375, 1.5, -0.625, 1.0],
            [1.125, 0.25, -1.75, -0.5],
        ]);
        let b = arr1(&[0.125f32, -0.25, 0.5]);
        let linear = LinearLayer::new(w, Some(b)).unwrap();
        let slice = SliceLayer::new(0, 0, 3);

        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input("lin", Layer::Linear(linear)));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["lin".to_string()],
        ));
        g.add_node(GraphNode::from_input("head", Layer::Slice(slice)));
        g.add_node(GraphNode::binary(
            "mul",
            Layer::MulBinary(MulBinaryLayer),
            "relu",
            "head",
        ));
        g.add_node(GraphNode::binary(
            "add",
            Layer::Add(AddLayer),
            "mul",
            "head",
        ));
        g.add_node(GraphNode::new(
            "sum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
            vec!["add".to_string()],
        ));
        // Divisor: 2.5 + sum(relu) >= 2.5 > 0 (sign-definite).
        g.add_node(GraphNode::new(
            "relu_sum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
            vec!["relu".to_string()],
        ));
        g.add_node(GraphNode::new(
            "denom",
            Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(
                IxDyn(&[1]),
                2.5f32,
            ))),
            vec!["relu_sum".to_string()],
        ));
        g.add_node(GraphNode::binary(
            "div",
            Layer::Div(DivLayer),
            "sum",
            "denom",
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["div".to_string()],
        ));
        g.set_output("out");
        g
    }

    /// Plain f64 (round-to-nearest) concrete forward of the same DAG — the
    /// sample oracle for the enclosure property test.
    fn mscn_like_concrete(x: &[f64; 4]) -> f64 {
        let w = [
            [0.5f64, -1.25, 2.0, 0.75],
            [-0.375, 1.5, -0.625, 1.0],
            [1.125, 0.25, -1.75, -0.5],
        ];
        let b = [0.125f64, -0.25, 0.5];
        let mut relu = [0.0f64; 3];
        for o in 0..3 {
            let mut s = b[o];
            for j in 0..4 {
                s += w[o][j] * x[j];
            }
            relu[o] = s.max(0.0);
        }
        let head = [x[0], x[1], x[2]];
        let mut sum = 0.0f64;
        let mut relu_sum = 0.0f64;
        for i in 0..3 {
            sum += relu[i] * head[i] + head[i];
            relu_sum += relu[i];
        }
        let div = sum / (2.5 + relu_sum);
        stable_sigmoid_f64(div)
    }

    /// ENCLOSURE: every sampled concrete forward (and every box corner) lies
    /// inside the f64 interval on the mscn-shaped DAG over a wide box.
    #[test]
    fn f64_cell_encloses_sampled_forwards_mscn_dag() {
        let g = build_mscn_like_graph();
        assert!(g.supports_ibp_f64_cell(), "test DAG must be supported");
        let lo = [-0.5f32, 0.25, -1.0, 0.125];
        let hi = [0.75f32, 1.5, -0.25, 0.875];
        let out = g.propagate_ibp_f64_cell(&box64(lo, hi)).unwrap();
        let (out_l, out_u) = (out.lower[[0]], out.upper[[0]]);
        assert!(out_l <= out_u);

        let mut rng = Rng(0x9E3779B97F4A7C15);
        for _ in 0..2000 {
            let mut x = [0.0f64; 4];
            for i in 0..4 {
                let (l, h) = (f64::from(lo[i]), f64::from(hi[i]));
                x[i] = l + (h - l) * rng.next_unit();
            }
            let y = mscn_like_concrete(&x);
            assert!(
                out_l <= y && y <= out_u,
                "sample {y} escapes f64 interval [{out_l}, {out_u}] at x={x:?}"
            );
        }
        for mask in 0..16u32 {
            let mut x = [0.0f64; 4];
            for i in 0..4 {
                x[i] = if mask & (1 << i) != 0 {
                    f64::from(hi[i])
                } else {
                    f64::from(lo[i])
                };
            }
            let y = mscn_like_concrete(&x);
            assert!(
                out_l <= y && y <= out_u,
                "corner {y} escapes f64 interval [{out_l}, {out_u}]"
            );
        }
    }

    /// ENCLOSURE on a MatMul/Mul/Add/ReLU DAG with a live rank-2 MatMul
    /// (both operands perturbed).
    #[test]
    fn f64_cell_encloses_matmul_dag() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::new(
            "mm",
            Layer::MatMul(MatMulLayer::new(false, None)),
            vec![NETWORK_INPUT.to_string(), NETWORK_INPUT.to_string()],
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["mm".to_string()],
        ));
        g.add_node(GraphNode::new(
            "mul",
            Layer::MulBinary(MulBinaryLayer),
            vec!["relu".to_string(), NETWORK_INPUT.to_string()],
        ));
        g.add_node(GraphNode::new(
            "out",
            Layer::Add(AddLayer),
            vec!["mul".to_string(), NETWORK_INPUT.to_string()],
        ));
        g.set_output("out");
        assert!(g.supports_ibp_f64_cell());

        let lo = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.5f32, 0.25, -0.75, 0.5]).unwrap();
        let hi = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5f32, 1.0, 0.25, 1.25]).unwrap();
        let input = Interval64::from_f32(&lo, &hi);
        let out = g.propagate_ibp_f64_cell(&input).unwrap();

        let mut rng = Rng(0xDEADBEEFCAFEF00D);
        for _ in 0..2000 {
            let mut x = [[0.0f64; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    let l = f64::from(lo[[i, j]]);
                    let h = f64::from(hi[[i, j]]);
                    x[i][j] = l + (h - l) * rng.next_unit();
                }
            }
            for i in 0..2 {
                for j in 0..2 {
                    let mm = x[i][0] * x[0][j] + x[i][1] * x[1][j];
                    let y = mm.max(0.0) * x[i][j] + x[i][j];
                    let (l, u) = (out.lower[[i, j]], out.upper[[i, j]]);
                    assert!(
                        l <= y && y <= u,
                        "sample {y} escapes [{l}, {u}] at [{i},{j}]"
                    );
                }
            }
        }
    }

    /// TIGHTNESS: on a degenerate (point) box the interval must be within a
    /// few ulps of the concrete forward — the whole point of the f64 leaf
    /// escalation is that this width sits far below f32's ~1e-7 floor.
    #[test]
    fn f64_cell_point_box_is_ulp_tight() {
        let g = build_mscn_like_graph();
        let p = [0.375f32, 0.625, -0.5, 0.25];
        let out = g.propagate_ibp_f64_cell(&box64(p, p)).unwrap();
        let (l, u) = (out.lower[[0]], out.upper[[0]]);
        let y = mscn_like_concrete(&[
            f64::from(p[0]),
            f64::from(p[1]),
            f64::from(p[2]),
            f64::from(p[3]),
        ]);
        assert!(l <= y && y <= u, "concrete {y} outside [{l}, {u}]");
        let width = u - l;
        assert!(
            width < 1e-12,
            "point-box width {width} not ulp-tight (f32 floor is ~1e-7)"
        );
        assert!(width < f64::from(f32::EPSILON) * y.abs().max(1.0));
    }

    /// Div with a divisor interval straddling zero fails closed.
    #[test]
    fn f64_cell_div_straddling_zero_fails_closed() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::new(
            "d",
            Layer::Div(DivLayer),
            vec![NETWORK_INPUT.to_string(), NETWORK_INPUT.to_string()],
        ));
        g.set_output("d");
        assert!(g.supports_ibp_f64_cell()); // statically supported...
        let input = box64([-1.0; 4], [1.0; 4]);
        // ...but the runtime divisor-sign check fails closed for this box.
        assert!(g.propagate_ibp_f64_cell(&input).is_err());
    }

    /// Sub interval direction: [a]-[b] uses (a_l - b_u, a_u - b_l).
    #[test]
    fn f64_cell_sub_interval_direction() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "head",
            Layer::Slice(SliceLayer::new(0, 0, 2)),
        ));
        g.add_node(GraphNode::from_input(
            "tail",
            Layer::Slice(SliceLayer::new(0, 2, 4)),
        ));
        g.add_node(GraphNode::binary(
            "sub",
            Layer::Sub(SubLayer),
            "head",
            "tail",
        ));
        g.set_output("sub");
        let input = box64([1.0, 2.0, 0.25, 0.5], [1.5, 3.0, 0.75, 1.0]);
        let out = g.propagate_ibp_f64_cell(&input).unwrap();
        // head[0]-tail[0]: [1-0.75, 1.5-0.25] = [0.25, 1.25] (±1 ulp outward)
        assert!(out.lower[[0]] <= 0.25 && out.lower[[0]] > 0.25 - 1e-12);
        assert!(out.upper[[0]] >= 1.25 && out.upper[[0]] < 1.25 + 1e-12);
    }

    /// Unsupported op (Tanh) is rejected by BOTH the static gate and the walk.
    #[test]
    fn f64_cell_fails_closed_on_unsupported_op() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input("t", Layer::Tanh(TanhLayer)));
        g.set_output("t");
        assert!(!g.supports_ibp_f64_cell());
        let input = box64([0.0; 4], [1.0; 4]);
        assert!(g.propagate_ibp_f64_cell(&input).is_err());
    }

    /// Sigmoid saturation regression: at x = -710 the naive 1/(1+e^-x)
    /// overflows exp and collapses to 0.0 with a non-enclosing widened upper;
    /// the stable form must still enclose the true value ~4.47e-309.
    #[test]
    fn f64_cell_sigmoid_extreme_negative_still_encloses() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "s",
            Layer::Sigmoid(SigmoidLayer::new()),
        ));
        g.set_output("s");
        let x = -710.0f32;
        let input = Interval64::from_f32(&arr1(&[x]).into_dyn(), &arr1(&[x]).into_dyn());
        let out = g.propagate_ibp_f64_cell(&input).unwrap();
        let truth = f64::from(x).exp(); // σ(x) ≈ e^x for very negative x
        assert!(truth > 0.0, "test needs a representable subnormal truth");
        assert!(
            out.lower[[0]] <= truth && truth <= out.upper[[0]],
            "σ(-710)≈{truth:e} escapes [{:e}, {:e}]",
            out.lower[[0]],
            out.upper[[0]]
        );
        assert!(out.lower[[0]] >= 0.0);
    }

    /// ReduceSum keepdims=false and multi-axis resolution stay sound/exact
    /// on a point box.
    #[test]
    fn f64_cell_reduce_sum_axes_and_keepdims() {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "sum",
            Layer::ReduceSum(ReduceSumLayer::new(vec![0], false)),
        ));
        g.set_output("sum");
        let vals = arr1(&[1.5f32, -2.25, 4.0, 0.125]).into_dyn();
        let input = Interval64::from_f32(&vals, &vals);
        let out = g.propagate_ibp_f64_cell(&input).unwrap();
        assert_eq!(out.lower.shape(), &[] as &[usize]);
        let truth = 1.5 - 2.25 + 4.0 + 0.125;
        let l = *out.lower.iter().next().unwrap();
        let u = *out.upper.iter().next().unwrap();
        assert!(l <= truth && truth <= u);
        assert!(u - l < 1e-13);
    }

    // -----------------------------------------------------------------------
    // Fast Rump-GEMM wiring (#f64-blas-gemm): above-threshold Linear/MatMul
    // take the probed-thread midpoint-radius kernel. The layer results must ENCLOSE
    // the scalar-path results AND sampled true products. (The assertions
    // also hold if NY_F64_BLAS=0 forces the scalar path — enclosure is
    // path-independent; the kernel itself is unit-gated in
    // graph_ibp_f64_gemm.)
    // -----------------------------------------------------------------------

    fn random_interval64(rng: &mut Rng, shape: &[usize], width: f64) -> Interval64 {
        let n: usize = shape.iter().product();
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        for _ in 0..n {
            let c = rng.next_unit() * 2.0 - 1.0;
            let w = rng.next_unit() * width;
            lo.push(c - w);
            hi.push(c + w);
        }
        Interval64 {
            lower: ArrayD::from_shape_vec(IxDyn(shape), lo).unwrap(),
            upper: ArrayD::from_shape_vec(IxDyn(shape), hi).unwrap(),
        }
    }

    fn sample_inside64(rng: &mut Rng, x: &Interval64) -> ArrayD<f64> {
        let mut out = x.lower.clone();
        for (o, (&l, &h)) in out.iter_mut().zip(x.lower.iter().zip(x.upper.iter())) {
            *o = l + (h - l) * rng.next_unit();
        }
        out
    }

    /// Rank-3 Linear above the fast-path threshold: fast wiring ⊇ scalar
    /// path ⊇ sampled x@W^T + b (both bias and no-bias, the MVF case).
    #[test]
    fn linear_fast_path_encloses_scalar_and_samples() {
        let (d0, d1, in_dim, out_dim) = (4usize, 12usize, 96usize, 64usize);
        assert!(
            d0 * d1 * in_dim * out_dim > 32_768,
            "test must cross the threshold"
        );
        let mut rng = Rng(0xC0FFEE0DDF00DBAD);
        // Heap-allocated: a [[f32; 96]; 64] literal is a 24 KiB stack array.
        let mut w = ndarray::Array2::<f32>::zeros((64, 96));
        for v in w.iter_mut() {
            *v = (rng.next_unit() * 2.0 - 1.0) as f32;
        }
        let mut b = arr1(&[0f32; 64]);
        for v in b.iter_mut() {
            *v = (rng.next_unit() * 2.0 - 1.0) as f32;
        }
        let linear = LinearLayer::new(w.clone(), Some(b.clone())).unwrap();
        let x = random_interval64(&mut rng, &[d0, d1, in_dim], 0.25);

        for include_bias in [true, false] {
            let fast = eval_linear_with_bias(&linear, &x, include_bias).unwrap();
            let scalar = eval_linear_with_bias_scalar(&linear, &x, include_bias).unwrap();
            assert_eq!(fast.lower.shape(), &[d0, d1, out_dim]);
            for (f, s) in fast.lower.iter().zip(scalar.lower.iter()) {
                assert!(f <= s, "fast lower {f} above scalar lower {s}");
            }
            for (f, s) in fast.upper.iter().zip(scalar.upper.iter()) {
                assert!(f >= s, "fast upper {f} below scalar upper {s}");
            }
            for _ in 0..50 {
                let xs = sample_inside64(&mut rng, &x);
                for i0 in 0..d0 {
                    for i1 in 0..d1 {
                        for o in 0..out_dim {
                            let mut y = if include_bias { f64::from(b[o]) } else { 0.0 };
                            for i in 0..in_dim {
                                y += f64::from(w[[o, i]]) * xs[[i0, i1, i]];
                            }
                            assert!(
                                fast.lower[[i0, i1, o]] <= y && y <= fast.upper[[i0, i1, o]],
                                "sample {y} escapes fast Linear interval"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Above-threshold MatMul (plain / transpose_b / scaled): fast wiring
    /// ⊇ scalar path ⊇ sampled true products.
    #[test]
    fn matmul_fast_path_encloses_scalar_and_samples() {
        let (m, k, n) = (40usize, 64usize, 40usize);
        assert!(m * k * n > 32_768, "test must cross the threshold");
        let mut rng = Rng(0x0DDBA11FADEDFACE);
        for (transpose_b, scale) in [(false, None), (true, None), (false, Some(0.125f32))] {
            let layer = MatMulLayer::new(transpose_b, scale);
            let a = random_interval64(&mut rng, &[m, k], 0.5);
            let b_shape = if transpose_b { [n, k] } else { [k, n] };
            let b = random_interval64(&mut rng, &b_shape, 0.5);

            let fast = eval_matmul(&layer, &a, &b).unwrap();
            let scalar = eval_matmul_scalar(&layer, &a, &b).unwrap();
            for (f, s) in fast.lower.iter().zip(scalar.lower.iter()) {
                assert!(f <= s, "fast lower {f} above scalar lower {s}");
            }
            for (f, s) in fast.upper.iter().zip(scalar.upper.iter()) {
                assert!(f >= s, "fast upper {f} below scalar upper {s}");
            }
            let s64 = scale.map_or(1.0, f64::from);
            for _ in 0..50 {
                let a_s = sample_inside64(&mut rng, &a);
                let b_s = sample_inside64(&mut rng, &b);
                for i in 0..m {
                    for j in 0..n {
                        let mut y = 0.0f64;
                        for l in 0..k {
                            let bv = if transpose_b {
                                b_s[[j, l]]
                            } else {
                                b_s[[l, j]]
                            };
                            y += a_s[[i, l]] * bv;
                        }
                        y *= s64;
                        assert!(
                            fast.lower[[i, j]] <= y && y <= fast.upper[[i, j]],
                            "sample {y} escapes fast MatMul interval at [{i},{j}]"
                        );
                    }
                }
            }
        }
    }
    /// Reference `broadcast_binary` (the pre-flat implementation, verbatim):
    /// `indexed_iter` over the broadcast views with per-element indexed
    /// lookups. The flat production path must be BIT-IDENTICAL to this on
    /// every input — same values, same error behavior.
    fn broadcast_binary_indexed_reference(
        a: &Interval64,
        b: &Interval64,
        exact: bool,
        combine: impl Fn(f64, f64, f64, f64) -> Result<(f64, f64)>,
    ) -> Result<Interval64> {
        let out_shape = crate::shape::broadcast_shapes(a.lower.shape(), b.lower.shape())
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: a.lower.shape().to_vec(),
                got: b.lower.shape().to_vec(),
            })?;
        let bc_err = || NyError::InvalidSpec("f64 cell eval: broadcast failed".to_string());
        let alo = a.lower.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;
        let ahi = a.upper.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;
        let blo = b.lower.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;
        let bhi = b.upper.broadcast(IxDyn(&out_shape)).ok_or_else(bc_err)?;

        let mut out_lo = ArrayD::zeros(IxDyn(&out_shape));
        let mut out_hi = ArrayD::zeros(IxDyn(&out_shape));
        for (idx, &al) in alo.indexed_iter() {
            let (mut lo, mut hi) =
                combine(al, ahi[idx.clone()], blo[idx.clone()], bhi[idx.clone()])?;
            if !exact {
                (lo, hi) = widen1(lo, hi);
            }
            out_lo[idx.clone()] = lo;
            out_hi[idx] = hi;
        }
        Ok(Interval64 {
            lower: out_lo,
            upper: out_hi,
        })
    }

    /// Bit-identity gate for the flat `broadcast_binary` (#f64-flat-elemwise):
    /// across same-shape, scalar-broadcast, axis-broadcast, non-contiguous
    /// (transposed view), and empty operands, the flat path must produce
    /// bit-identical outputs to the indexed reference for every elementwise
    /// rule used by `eval_node` (add / sub / interval mul / min-exact), and
    /// identical error behavior for a failing `combine`.
    #[test]
    fn broadcast_binary_flat_matches_indexed_reference() {
        let mut rng = Rng(0x0DDB_1773_57A7_E001);
        let mk = |rng: &mut Rng, shape: &[usize], width: f64| -> Interval64 {
            let n: usize = shape.iter().product();
            let mut lo = Vec::with_capacity(n);
            let mut hi = Vec::with_capacity(n);
            for _ in 0..n {
                let c = rng.next_unit() * 4.0 - 2.0;
                let w = rng.next_unit() * width;
                lo.push(c - w);
                hi.push(c + w);
            }
            Interval64 {
                lower: ArrayD::from_shape_vec(IxDyn(shape), lo).unwrap(),
                upper: ArrayD::from_shape_vec(IxDyn(shape), hi).unwrap(),
            }
        };
        // Transposed (non-standard-layout) variant of an interval.
        let transpose2 = |x: &Interval64| -> Interval64 {
            Interval64 {
                lower: x.lower.clone().permuted_axes(IxDyn(&[1, 0])),
                upper: x.upper.clone().permuted_axes(IxDyn(&[1, 0])),
            }
        };

        let shapes: &[(&[usize], &[usize])] = &[
            (&[3, 128], &[3, 128]), // same shape (dominant mscn case)
            (&[3, 128], &[128]),    // trailing-axis broadcast
            (&[3, 1], &[1, 4]),     // two-sided broadcast
            (&[1], &[2, 3]),        // scalar-vs-tensor
            (&[2, 3, 4], &[3, 1]),  // rank + axis broadcast
            (&[0, 4], &[4]),        // empty output
        ];
        type Rule = (bool, fn(f64, f64, f64, f64) -> Result<(f64, f64)>);
        let rules: &[Rule] = &[
            (false, |al, ah, bl, bh| Ok((al + bl, ah + bh))),
            (false, |al, ah, bl, bh| Ok((al - bh, ah - bl))),
            (false, |al, ah, bl, bh| Ok(interval_mul(al, ah, bl, bh))),
            (true, |al, ah, bl, bh| Ok((al.min(bl), ah.min(bh)))),
        ];
        for &(sa, sb) in shapes {
            let a = mk(&mut rng, sa, 0.5);
            let b = mk(&mut rng, sb, 0.5);
            for (r, &(exact, rule)) in rules.iter().enumerate() {
                let flat = broadcast_binary(&a, &b, exact, rule).unwrap();
                let reference = broadcast_binary_indexed_reference(&a, &b, exact, rule).unwrap();
                let bits = |x: &ArrayD<f64>| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(
                    bits(&flat.lower),
                    bits(&reference.lower),
                    "rule {r} {sa:?}x{sb:?}: flat lower diverged"
                );
                assert_eq!(
                    bits(&flat.upper),
                    bits(&reference.upper),
                    "rule {r} {sa:?}x{sb:?}: flat upper diverged"
                );
            }
        }

        // Non-contiguous operands (transposed views) still agree bitwise.
        let a = mk(&mut rng, &[4, 6], 0.3);
        let b = mk(&mut rng, &[6, 4], 0.3);
        let bt = transpose2(&b);
        let flat = broadcast_binary(&a, &bt, false, |al, ah, bl, bh| {
            Ok(interval_mul(al, ah, bl, bh))
        })
        .unwrap();
        let reference = broadcast_binary_indexed_reference(&a, &bt, false, |al, ah, bl, bh| {
            Ok(interval_mul(al, ah, bl, bh))
        })
        .unwrap();
        assert_eq!(
            flat.lower.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference
                .lower
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            flat.upper.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference
                .upper
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );

        // Error parity: a combine that fails on a specific element (Div-style
        // divisor check) errors on both paths with the same message.
        let a = mk(&mut rng, &[2, 3], 0.5);
        let mut b = mk(&mut rng, &[2, 3], 0.0);
        b.lower[[1, 1]] = -1.0;
        b.upper[[1, 1]] = 1.0;
        let div_rule = |al: f64, ah: f64, bl: f64, bh: f64| -> Result<(f64, f64)> {
            if bl <= 0.0 && bh >= 0.0 {
                return Err(NyError::InvalidSpec("divisor straddles zero".to_string()));
            }
            Ok((al / bl.max(bh), ah / bl.min(bh)))
        };
        let flat_err = broadcast_binary(&a, &b, false, div_rule).unwrap_err();
        let ref_err = broadcast_binary_indexed_reference(&a, &b, false, div_rule).unwrap_err();
        assert_eq!(flat_err.to_string(), ref_err.to_string());

        // Shape-mismatch parity.
        let a = mk(&mut rng, &[2, 3], 0.1);
        let b = mk(&mut rng, &[2, 4], 0.1);
        assert!(broadcast_binary(&a, &b, true, |al, ah, _, _| Ok((al, ah))).is_err());
        assert!(
            broadcast_binary_indexed_reference(&a, &b, true, |al, ah, _, _| Ok((al, ah))).is_err()
        );
    }

    // -----------------------------------------------------------------------
    // Flatten (#sb-rebank lever 2): exact data movement, whitelist coverage.
    // -----------------------------------------------------------------------

    /// Flatten is EXACT data movement: bit-identical endpoints, row-major order
    /// preserved, ONNX 2-D collapse semantics for axis 0 / mid / negative /
    /// rank axes. This is the arm that unblocks `supports_ibp_f64_cell` on the
    /// soundnessbench net (Conv x6 -> Flatten -> Gemm x2).
    #[test]
    fn flatten_is_exact_data_movement_and_supported() {
        use crate::layers::FlattenLayer;

        // Whitelist coverage: the support predicate and the eval arm must stay
        // in sync (the escalation gate fails closed otherwise).
        assert!(cell_supports_layer(&Layer::Flatten(FlattenLayer::new(1))));

        let mut rng = Rng(0x5b_f1a7);
        let vals: Vec<f64> = (0..24).map(|_| rng.next_unit() * 2.0 - 1.0).collect();
        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), vals).unwrap();
        let upper = lower.mapv(|v| v + 0.5);
        let x = Interval64 {
            lower: lower.clone(),
            upper: upper.clone(),
        };

        for (axis, want_shape) in [
            (0i32, vec![1usize, 24]),
            (1, vec![2, 12]),
            (2, vec![6, 4]),
            (3, vec![24, 1]),
            (-1, vec![6, 4]),
        ] {
            let mut g = GraphNetwork::new();
            g.add_node(GraphNode::from_input(
                "flat",
                Layer::Flatten(FlattenLayer::new(axis)),
            ));
            g.set_output("flat");
            assert!(g.supports_ibp_f64_cell(), "axis {axis}: gate must open");
            let out = g.propagate_ibp_f64_cell(&x).unwrap();
            assert_eq!(out.lower.shape(), &want_shape[..], "axis {axis}");
            // Bit-exact row-major copy: no widening, no reordering.
            let got_lo: Vec<u64> = out.lower.iter().map(|v| v.to_bits()).collect();
            let want_lo: Vec<u64> = lower.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got_lo, want_lo, "axis {axis}: lower endpoints moved");
            let got_hi: Vec<u64> = out.upper.iter().map(|v| v.to_bits()).collect();
            let want_hi: Vec<u64> = upper.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got_hi, want_hi, "axis {axis}: upper endpoints moved");
        }
    }

    /// Flatten feeding a Linear (the soundnessbench Conv->Flatten->Gemm shape):
    /// the cell walk through Flatten must agree with manually pre-flattened
    /// input on the SAME Linear — bit-identical bounds.
    #[test]
    fn flatten_then_linear_matches_preflattened() {
        use crate::layers::FlattenLayer;

        let w = arr2(&[
            [0.5f32, -1.25, 2.0, 0.75, -0.375, 1.5],
            [1.125, 0.25, -1.75, -0.5, 0.625, -1.0],
        ]);
        let b = arr1(&[0.125f32, -0.25]);

        // Graph A: rank-3 input [1,2,3] -> Flatten(1) -> Linear(6->2).
        let mut ga = GraphNetwork::new();
        ga.add_node(GraphNode::from_input(
            "flat",
            Layer::Flatten(FlattenLayer::new(1)),
        ));
        ga.add_node(GraphNode::new(
            "lin",
            Layer::Linear(LinearLayer::new(w.clone(), Some(b.clone())).unwrap()),
            vec!["flat".to_string()],
        ));
        ga.set_output("lin");

        // Graph B: the identical Linear on the already-flat [1,6] input.
        let mut gb = GraphNetwork::new();
        gb.add_node(GraphNode::from_input(
            "lin",
            Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()),
        ));
        gb.set_output("lin");

        let mut rng = Rng(0xf1a77e4);
        for _ in 0..8 {
            let vals: Vec<f64> = (0..6).map(|_| rng.next_unit() * 4.0 - 2.0).collect();
            let point3 = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vals.clone()).unwrap();
            let point2 = ArrayD::from_shape_vec(IxDyn(&[1, 6]), vals).unwrap();
            let oa = ga
                .propagate_ibp_f64_cell(&Interval64::point(point3))
                .unwrap();
            let ob = gb
                .propagate_ibp_f64_cell(&Interval64::point(point2))
                .unwrap();
            let bits = |a: &ArrayD<f64>| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&oa.lower), bits(&ob.lower));
            assert_eq!(bits(&oa.upper), bits(&ob.upper));
        }
    }

    // ================= W0.2 reachability arms (SubConstant / ConvTranspose2d /
    // BatchNorm) ==============================================================

    /// acasxu-shaped chain: `SubConstant(avg)` -> Linear -> ReLU -> Linear.
    /// W0.2 could only measure this class after rewriting the ONNX
    /// `Sub(x, avg)` to `Add(x, -avg)`, because `SubConstant` had no arm. The
    /// rewrite is bit-exact (negation is a sign-bit flip; IEEE-754 `x - c` and
    /// `x + (-c)` return the same bit pattern), so the in-tree `SubConstant`
    /// arm must reproduce the rewrite's endpoints EXACTLY — this is the
    /// in-tree restatement of that measurement.
    #[test]
    fn sub_constant_endpoints_match_the_add_constant_rewrite_bitwise() {
        use crate::layers::SubConstantLayer;

        // Stand-in for acasxu's `input_AvgImg` (5 means), deliberately not
        // dyadic so the subtraction actually rounds.
        let avg = arr1(&[1.9791091e4f32, 0.0, 0.0, 650.0, 600.0]).into_dyn();
        let neg_avg = avg.mapv(|v| -v);
        let w1 = arr2(&[
            [0.5f32, -1.25, 2.0, 0.75, -0.375],
            [1.125, 0.25, -1.75, -0.5, 0.625],
            [-1.0, 0.875, 0.125, 1.375, -2.25],
            [0.3125, -0.6875, 1.0625, -0.15625, 0.9375],
        ]);
        let b1 = arr1(&[0.125f32, -0.25, 0.5, -0.0625]);
        let w2 = arr2(&[
            [0.75f32, -0.5, 1.25, 0.375],
            [-1.125, 0.625, -0.25, 1.0],
            [0.5, 0.875, -1.375, -0.75],
        ]);
        let b2 = arr1(&[0.0625f32, -0.125, 0.25]);

        let build = |first: Layer| -> GraphNetwork {
            let mut g = GraphNetwork::new();
            g.add_node(GraphNode::from_input("norm", first));
            g.add_node(GraphNode::new(
                "lin1",
                Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
                vec!["norm".to_string()],
            ));
            g.add_node(GraphNode::new(
                "relu",
                Layer::ReLU(ReLULayer),
                vec!["lin1".to_string()],
            ));
            g.add_node(GraphNode::new(
                "lin2",
                Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
                vec!["relu".to_string()],
            ));
            g.set_output("lin2");
            g
        };
        let g_sub = build(Layer::SubConstant(SubConstantLayer::new(avg)));
        let g_add = build(Layer::AddConstant(AddConstantLayer::new(neg_avg)));
        assert!(
            g_sub.supports_ibp_f64_cell(),
            "SubConstant must now pass the escalation gate"
        );
        assert!(g_add.supports_ibp_f64_cell());

        let mut rng = Rng(0xACA5_0000_1234_5678);
        for _ in 0..64 {
            // acasxu-like raw input magnitudes (pre-normalization).
            let vals: Vec<f64> = (0..5)
                .map(|i| {
                    let scale = if i == 0 { 6.0e4 } else { 1.0e3 };
                    rng.next_unit() * scale
                })
                .collect();
            let point = Interval64::point(ArrayD::from_shape_vec(IxDyn(&[5]), vals).unwrap());
            let os = g_sub.propagate_ibp_f64_cell(&point).unwrap();
            let oa = g_add.propagate_ibp_f64_cell(&point).unwrap();
            let bits = |a: &ArrayD<f64>| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&os.lower),
                bits(&oa.lower),
                "SubConstant lower endpoints differ from the Add(x,-c) rewrite"
            );
            assert_eq!(
                bits(&os.upper),
                bits(&oa.upper),
                "SubConstant upper endpoints differ from the Add(x,-c) rewrite"
            );
        }
    }

    /// `SubConstant` in REVERSE form (`y = c - x`, the LayerNorm mean-subtract
    /// shape): endpoints must swap with the input and still enclose the exact
    /// real difference over a wide box.
    #[test]
    fn sub_constant_reverse_swaps_endpoints_and_encloses() {
        use crate::layers::SubConstantLayer;

        let c = arr1(&[10.0f32, -0.5, 0.1]).into_dyn();
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "rev",
            Layer::SubConstant(SubConstantLayer::new_reverse(c.clone())),
        ));
        g.set_output("rev");
        assert!(g.supports_ibp_f64_cell());

        let lo = arr1(&[-1.0f32, 0.25, -3.5]).into_dyn();
        let hi = arr1(&[2.0f32, 4.75, 0.5]).into_dyn();
        let out = g
            .propagate_ibp_f64_cell(&Interval64::from_f32(&lo, &hi))
            .unwrap();
        for i in 0..3 {
            let (cl, xl, xh) = (f64::from(c[[i]]), f64::from(lo[[i]]), f64::from(hi[[i]]));
            assert!(
                out.lower[[i]] <= cl - xh,
                "reverse lower not outward at {i}"
            );
            assert!(
                out.upper[[i]] >= cl - xl,
                "reverse upper not outward at {i}"
            );
            // 1-ulp widening only: the interval must not be blown up.
            assert!(out.upper[[i]] - out.lower[[i]] <= (xh - xl) + 1e-12);
        }
    }

    /// Naive GATHER-form f64 reference for ConvTranspose2d, derived
    /// independently of the SCATTER form the arm implements: for each output
    /// cell it inverts `oh = ih*sh + ki*dh - ph` (requiring an exact stride
    /// residue) instead of pushing forward from the input.
    #[allow(clippy::too_many_arguments)]
    fn convt_reference_f64(
        x: &[f64],
        in_c: usize,
        in_h: usize,
        in_w: usize,
        kernel: &ArrayD<f32>,
        bias: Option<&ndarray::Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        out_h: usize,
        out_w: usize,
    ) -> Vec<f64> {
        let out_c = kernel.shape()[1];
        let (kh, kw) = (kernel.shape()[2], kernel.shape()[3]);
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let (dh, dw) = dilation;
        let mut out = vec![0.0f64; out_c * out_h * out_w];
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias.map(|b| f64::from(b[oc])).unwrap_or(0.0);
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            let th = oh as isize + ph as isize - (ki * dh) as isize;
                            if th < 0 || th % sh as isize != 0 {
                                continue;
                            }
                            let ih = (th / sh as isize) as usize;
                            if ih >= in_h {
                                continue;
                            }
                            for kj in 0..kw {
                                let tw = ow as isize + pw as isize - (kj * dw) as isize;
                                if tw < 0 || tw % sw as isize != 0 {
                                    continue;
                                }
                                let iw = (tw / sw as isize) as usize;
                                if iw >= in_w {
                                    continue;
                                }
                                acc += x[(ic * in_h + ih) * in_w + iw]
                                    * f64::from(kernel[[ic, oc, ki, kj]]);
                            }
                        }
                    }
                    out[(oc * out_h + oh) * out_w + ow] = acc;
                }
            }
        }
        out
    }

    /// ConvTranspose2d (cgan's gap): the f64 interval arm must (a) agree with
    /// NY's own f32 forward `conv2d_transpose_forward` on EXACT dyadic data,
    /// (b) enclose an independently derived gather-form f64 reference on random
    /// data, over stride/padding/dilation/output_padding configurations.
    #[test]
    fn conv_transpose2d_encloses_independent_reference_across_geometries() {
        use crate::layers::convolution::conv2d::conv2d_transpose_forward;
        use crate::layers::ConvTranspose2dLayer;

        let (in_c, out_c, kh, kw) = (2usize, 3usize, 3usize, 3usize);
        let (in_h, in_w) = (4usize, 5usize);
        let geometries: &[(
            (usize, usize),
            (usize, usize),
            (usize, usize),
            (usize, usize),
        )] = &[
            ((1, 1), (0, 0), (1, 1), (0, 0)),
            ((2, 2), (1, 1), (1, 1), (0, 0)),
            ((2, 3), (1, 2), (1, 1), (1, 2)),
            ((1, 1), (0, 0), (2, 2), (0, 0)),
            ((2, 2), (0, 1), (2, 1), (1, 0)),
        ];

        let mut rng = Rng(0xC0FF_EE00_D15E_A5E5);
        for &(stride, padding, dilation, output_padding) in geometries {
            // (a) DYADIC data: every product and partial sum is exactly
            //     representable in binary32, so the f32 forward IS the exact
            //     real value and must sit inside the f64 enclosure.
            let dyadic = [0.5f32, -1.0, 2.0, -0.25, 1.0, -0.5];
            let kernel = ArrayD::from_shape_fn(IxDyn(&[in_c, out_c, kh, kw]), |ix| {
                dyadic[(ix[0] * 7 + ix[1] * 5 + ix[2] * 3 + ix[3]) % dyadic.len()]
            });
            let bias = ndarray::Array1::from_shape_fn(out_c, |o| 0.25f32 * (o as f32) - 0.5);
            let layer = ConvTranspose2dLayer::new_full(
                kernel.clone(),
                Some(bias.clone()),
                stride,
                padding,
                dilation,
                output_padding,
            )
            .unwrap();
            let mut g = GraphNetwork::new();
            g.add_node(GraphNode::from_input(
                "convt",
                Layer::ConvTranspose2d(layer),
            ));
            g.set_output("convt");
            assert!(
                g.supports_ibp_f64_cell(),
                "ConvTranspose2d must pass the escalation gate for {stride:?}"
            );

            let x32 = ArrayD::from_shape_fn(IxDyn(&[in_c, in_h, in_w]), |ix| {
                dyadic[(ix[0] * 3 + ix[1] * 5 + ix[2]) % dyadic.len()]
            });
            let f32_out =
                conv2d_transpose_forward(&x32, &kernel, stride, padding, dilation, output_padding)
                    .unwrap();
            let point = Interval64::point(x32.mapv(f64::from));
            let out = g.propagate_ibp_f64_cell(&point).unwrap();
            assert_eq!(
                out.lower.shape(),
                f32_out.shape(),
                "output shape disagrees with the f32 forward for {stride:?}"
            );
            // The f32 forward carries no bias for the output_padding cells; NY's
            // f32 op omits bias entirely (the layer applies it separately), so
            // compare against the reference with bias = None.
            let (out_h, out_w) = (f32_out.shape()[1], f32_out.shape()[2]);
            let xs: Vec<f64> = x32.iter().map(|&v| f64::from(v)).collect();
            let no_bias = convt_reference_f64(
                &xs, in_c, in_h, in_w, &kernel, None, stride, padding, dilation, out_h, out_w,
            );
            for (idx, (&r, f)) in no_bias.iter().zip(f32_out.iter()).enumerate() {
                assert_eq!(
                    r,
                    f64::from(*f),
                    "dyadic setup is not exact at {idx} for {stride:?}"
                );
            }
            let with_bias = convt_reference_f64(
                &xs,
                in_c,
                in_h,
                in_w,
                &kernel,
                Some(&bias),
                stride,
                padding,
                dilation,
                out_h,
                out_w,
            );
            for (i, &r) in with_bias.iter().enumerate() {
                let (l, u) = (
                    out.lower.as_slice().unwrap()[i],
                    out.upper.as_slice().unwrap()[i],
                );
                assert!(
                    l <= r && r <= u,
                    "exact dyadic value {r} escapes [{l}, {u}] at {i} for {stride:?}"
                );
                assert!(u - l < 1e-12, "enclosure too wide at {i}: {}", u - l);
            }

            // (b) RANDOM data: enclosure of the independent gather reference.
            let rker = ArrayD::from_shape_fn(IxDyn(&[in_c, out_c, kh, kw]), |_| {
                (rng.next_unit() * 2.0 - 1.0) as f32
            });
            let rbias = ndarray::Array1::from_shape_fn(out_c, |_| (rng.next_unit() - 0.5) as f32);
            let rlayer = ConvTranspose2dLayer::new_full(
                rker.clone(),
                Some(rbias.clone()),
                stride,
                padding,
                dilation,
                output_padding,
            )
            .unwrap();
            let mut gr = GraphNetwork::new();
            gr.add_node(GraphNode::from_input(
                "convt",
                Layer::ConvTranspose2d(rlayer),
            ));
            gr.set_output("convt");
            let rx: Vec<f64> = (0..in_c * in_h * in_w)
                .map(|_| rng.next_unit() * 4.0 - 2.0)
                .collect();
            let rpoint = Interval64::point(
                ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), rx.clone()).unwrap(),
            );
            let rout = gr.propagate_ibp_f64_cell(&rpoint).unwrap();
            let reference = convt_reference_f64(
                &rx,
                in_c,
                in_h,
                in_w,
                &rker,
                Some(&rbias),
                stride,
                padding,
                dilation,
                rout.lower.shape()[1],
                rout.lower.shape()[2],
            );
            for (i, &r) in reference.iter().enumerate() {
                let (l, u) = (
                    rout.lower.as_slice().unwrap()[i],
                    rout.upper.as_slice().unwrap()[i],
                );
                assert!(
                    l <= r && r <= u,
                    "reference {r} escapes [{l}, {u}] at {i} for {stride:?}"
                );
            }
        }
    }

    /// A batched (rank-4) ConvTranspose2d must evaluate each batch row
    /// identically to the unbatched rank-3 walk on that row.
    #[test]
    fn conv_transpose2d_batched_matches_unbatched_rows() {
        use crate::layers::ConvTranspose2dLayer;

        let (in_c, out_c, kh, kw, in_h, in_w) = (2usize, 2usize, 2usize, 2usize, 3usize, 3usize);
        let mut rng = Rng(0x5EED_B47C_4ED0_0001);
        let kernel = ArrayD::from_shape_fn(IxDyn(&[in_c, out_c, kh, kw]), |_| {
            (rng.next_unit() * 2.0 - 1.0) as f32
        });
        let mk = || {
            let layer = ConvTranspose2dLayer::new_full(
                kernel.clone(),
                None,
                (2, 2),
                (1, 1),
                (1, 1),
                (1, 1),
            )
            .unwrap();
            let mut g = GraphNetwork::new();
            g.add_node(GraphNode::from_input(
                "convt",
                Layer::ConvTranspose2d(layer),
            ));
            g.set_output("convt");
            g
        };
        let g = mk();
        let per_row: Vec<Vec<f64>> = (0..2)
            .map(|_| {
                (0..in_c * in_h * in_w)
                    .map(|_| rng.next_unit() * 2.0 - 1.0)
                    .collect()
            })
            .collect();
        let mut flat = per_row[0].clone();
        flat.extend_from_slice(&per_row[1]);
        let batched = g
            .propagate_ibp_f64_cell(&Interval64::point(
                ArrayD::from_shape_vec(IxDyn(&[2, in_c, in_h, in_w]), flat).unwrap(),
            ))
            .unwrap();
        let per_out = batched.lower.len() / 2;
        for (row, values) in per_row.iter().enumerate() {
            let single = g
                .propagate_ibp_f64_cell(&Interval64::point(
                    ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), values.clone()).unwrap(),
                ))
                .unwrap();
            for i in 0..per_out {
                assert_eq!(
                    batched.lower.as_slice().unwrap()[row * per_out + i].to_bits(),
                    single.lower.as_slice().unwrap()[i].to_bits(),
                    "batched row {row} lower differs at {i}"
                );
                assert_eq!(
                    batched.upper.as_slice().unwrap()[row * per_out + i].to_bits(),
                    single.upper.as_slice().unwrap()[i].to_bits(),
                    "batched row {row} upper differs at {i}"
                );
            }
        }
    }

    /// BatchNorm (cgan's second gap): the arm encloses the exact real affine of
    /// the LOADED f32 coefficients (see `eval_batch_norm`'s scope note), keeps
    /// the enclosure a few ulps wide, and orders endpoints correctly for
    /// negative scales.
    #[test]
    fn batch_norm_encloses_loaded_affine_and_stays_tight() {
        use crate::layers::BatchNormLayer;

        let scale = arr1(&[2.5f32, -1.25, 0.0625]).into_dyn();
        let bias = arr1(&[-0.75f32, 0.5, 3.0]).into_dyn();
        let bn = BatchNormLayer::from_scale_bias(scale.clone(), bias.clone()).unwrap();
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input("bn", Layer::BatchNorm(bn)));
        g.set_output("bn");
        assert!(
            g.supports_ibp_f64_cell(),
            "finite-coefficient BatchNorm must pass the escalation gate"
        );

        // [1, C, H, W]: channel axis 1.
        let (c, h, w) = (3usize, 2usize, 2usize);
        let mut rng = Rng(0xB47C_4074_0001_0001);
        let lo: Vec<f64> = (0..c * h * w)
            .map(|_| rng.next_unit() * 4.0 - 2.0)
            .collect();
        let hi: Vec<f64> = lo.iter().map(|v| v + rng.next_unit() * 0.5).collect();
        let input = Interval64 {
            lower: ArrayD::from_shape_vec(IxDyn(&[1, c, h, w]), lo.clone()).unwrap(),
            upper: ArrayD::from_shape_vec(IxDyn(&[1, c, h, w]), hi.clone()).unwrap(),
        };
        let out = g.propagate_ibp_f64_cell(&input).unwrap();
        for i in 0..c * h * w {
            let ch = i / (h * w);
            let s = f64::from(scale[[ch]]);
            let b = f64::from(bias[[ch]]);
            let (e1, e2) = (s * lo[i] + b, s * hi[i] + b);
            let (want_lo, want_hi) = (e1.min(e2), e1.max(e2));
            let (l, u) = (
                out.lower.as_slice().unwrap()[i],
                out.upper.as_slice().unwrap()[i],
            );
            assert!(
                l <= want_lo && u >= want_hi,
                "BatchNorm interval [{l}, {u}] does not enclose [{want_lo}, {want_hi}] at {i}"
            );
            let slack = (want_lo - l).max(u - want_hi);
            assert!(
                slack <= 8.0 * f64::EPSILON * (1.0 + want_hi.abs()),
                "BatchNorm widening {slack} is not ulp-scale at {i}"
            );
        }
    }

    /// A degenerate BatchNorm channel (var + eps -> 0) has no finite affine
    /// interpretation and must be rejected before it can enter any graph.
    #[test]
    fn batch_norm_with_zero_denominator_is_rejected_at_construction() {
        use crate::layers::BatchNormLayer;

        let err = BatchNormLayer::new(
            &arr1(&[1.0f32, 1.0]).into_dyn(),
            &arr1(&[0.0f32, 0.0]).into_dyn(),
            &arr1(&[0.0f32, 0.0]).into_dyn(),
            &arr1(&[1.0f32, 0.0]).into_dyn(),
            0.0,
        )
        .expect_err("zero BatchNorm denominator must fail closed");
        assert!(
            matches!(err, NyError::InvalidSpec(ref message) if message.contains("positive")),
            "expected a strictly-positive denominator rejection, got {err:?}"
        );
    }

    #[test]
    fn guarded_integer_cast_domain_is_enforced_by_the_direct_f64_cell_evaluator() {
        use crate::layers::TruncLayer;

        for (layer, exponent) in [
            (TruncLayer::for_int32_cast(), 31),
            (TruncLayer::for_int64_cast(), 63),
        ] {
            let mut graph = GraphNetwork::new();
            graph.add_node(GraphNode::from_input("cast", Layer::Trunc(layer)));
            graph.set_output("cast");
            assert!(
                graph.supports_ibp_f64_cell(),
                "a guarded Cast is supported when its runtime domain certificate passes"
            );

            let limit = 2.0_f64.powi(exponent);
            let in_range = Interval64 {
                lower: arr1(&[-limit]).into_dyn(),
                upper: arr1(&[ny_core::dd::next_down_f64(limit)]).into_dyn(),
            };
            graph
                .propagate_ibp_f64_cell(&in_range)
                .expect("the exact-f64 cell evaluator must accept the in-range domain");

            for invalid in [
                Interval64::point(arr1(&[limit]).into_dyn()),
                Interval64::point(arr1(&[f64::INFINITY]).into_dyn()),
                Interval64::point(arr1(&[f64::NAN]).into_dyn()),
                Interval64::point(arr1(&[ny_core::dd::next_down_f64(-limit)]).into_dyn()),
            ] {
                assert!(
                    graph.propagate_ibp_f64_cell(&invalid).is_err(),
                    "NaN, infinity, and either out-of-range Cast edge must fail closed"
                );
            }
        }
    }
}
