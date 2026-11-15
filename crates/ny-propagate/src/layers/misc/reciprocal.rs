// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Reciprocal layer (y = 1/x) with tangent-line CROWN relaxation.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{
    cast_f64_to_f32_down, cast_f64_to_f32_up, next_down_f32, next_up_f32, BoundedTensor,
    RepairStrategy,
};
use std::borrow::Cow;
use tracing::debug;

use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{
    compose::{compose_lower, compose_upper, log_nonfinite_fallback, precompute_relaxations},
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, BoundPropagation,
};
use crate::layers::linear::bias::{add_f64_down, add_f64_up};
use crate::{BatchedLinearBounds, LinearBounds};

/// Reciprocal layer: y = 1/x.
///
/// Requires x != 0. For interval bounds containing zero, returns conservative [-inf, inf].
#[derive(Debug, Clone, Default)]
pub struct ReciprocalLayer;

impl ReciprocalLayer {
    /// Create a new Reciprocal layer.
    pub fn new() -> Self {
        Self
    }
}

impl BoundPropagation for ReciprocalLayer {
    /// IBP for Reciprocal: y = 1/x
    ///
    /// Reciprocal is monotonically decreasing for x > 0 and x < 0.
    /// For x in [lb, ub] where lb > 0: y in [1/ub, 1/lb]
    /// For x in [lb, ub] where ub < 0: y in [1/ub, 1/lb]
    /// For intervals containing zero: y in [-inf, inf] (conservative)
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut out_lower = ArrayD::zeros(IxDyn(input.shape()));
        let mut out_upper = ArrayD::zeros(IxDyn(input.shape()));

        for (idx, &lb) in input.lower().indexed_iter() {
            let ub = input.upper()[idx.clone()];

            let (r_lb, r_ub) = if lb > 0.0 {
                // Entire interval is positive: 1/x is decreasing, so flip bounds.
                // Directed rounding: compute in f64, round lower down, upper up.
                // IEEE 754 division is correctly rounded (max 0.5 ULP), but f64→f32
                // cast may round the wrong direction.
                //
                // SATURATION IS THE HAZARD THE PLAIN `as f32` STEP MISSES, and
                // the old comment above named only the cast direction. When
                // `1/ub` exceeds f32 range the inner cast yields +inf, and
                // `next_down_f32` deliberately returns ±Inf UNCHANGED as an
                // infeasible-interval sentinel (rounding.rs:47-52) — so the
                // LOWER bound was rounded UP, from a large finite real all the
                // way to +inf. `cast_f64_to_f32_down` handles exactly this and
                // says so in its own docs (rounding.rs:203-210).
                //
                // Witness: lb = f32::from_bits(0x00100000) = 1.4693679e-39,
                // ub = f32::from_bits(0x00200000) = 2.9387359e-39. The exact
                // range of 1/x there is [3.4028236692093846e38,
                // 6.8056473384187693e38]; `1.0/f64::from(ub)` is finite, but
                // `as f32` overflows and the emitted lower bound was +inf —
                // certifying `1/x >= +inf` where the true infimum is
                // 3.4028236692093846e38. `BoundedTensor::new_repaired` with
                // `RepairStrategy::Conservative` does NOT catch it: it replaces
                // only NaN and preserves ±Inf by design, and `fix_inverted`
                // swaps only when lower > upper, which +inf > +inf is not. So
                // [+inf, +inf] shipped.
                //
                // Enumerating all 8,388,607 positive subnormal f32 values of
                // `ub`, exactly 2^21 = 2,097,152 of them triggered this, and
                // `cast_f64_to_f32_down` is sound on every one of them (checked
                // against exact rationals, not floats). The outer
                // `next_down_f32` is kept so the result is <= today's value at
                // EVERY input: over those 8.4M subnormals plus 208,192 normal
                // `ub` including the top and bottom of the normal range, the new
                // expression is tighter than the old one in 0 cases. Strictly
                // widening, so no banked bound can move.
                (
                    next_down_f32(cast_f64_to_f32_down(1.0_f64 / f64::from(ub))),
                    next_up_f32((1.0_f64 / lb as f64) as f32),
                )
            } else if ub < 0.0 {
                // Entire interval is negative: 1/x is decreasing, so flip bounds.
                // Same directed rounding pattern.
                //
                // Exact mirror of the positive branch above, on the UPPER side:
                // when `1/lb` underflows past -f32::MAX the cast yields -inf and
                // `next_up_f32` returns it unchanged, so the UPPER bound was
                // rounded DOWN to -inf. `cast_f64_to_f32_up` documents this
                // mirror case at rounding.rs:224-226.
                //
                // Witness: lb = -2.9387359e-39, ub = -1.4693679e-39, where the
                // exact range of 1/x is [-6.8056473384187693e38,
                // -3.4028236692093846e38] and the emitted upper bound was -inf,
                // certifying `1/x <= -inf` against a true supremum of
                // -3.4028236692093846e38. Same 2^21 magnitudes as above, by the
                // symmetry 1/(-x) = -(1/x). Note the branch geometry forces both
                // endpoints to collapse together here: lb <= ub < 0 gives
                // |lb| >= |ub|, so whenever 1/lb overflows, 1/ub already has.
                (
                    next_down_f32((1.0_f64 / ub as f64) as f32),
                    next_up_f32(cast_f64_to_f32_up(1.0_f64 / f64::from(lb))),
                )
            } else {
                // Interval contains zero: reciprocal is undefined at 0
                // Return conservative bounds
                (f32::NEG_INFINITY, f32::INFINITY)
            };

            out_lower[idx.clone()] = r_lb;
            out_upper[idx] = r_ub;
        }

        // Centralized NaN/Inf repair at constructor.
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    /// CROWN requires pre-activation bounds for nonlinear Reciprocal.
    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Reciprocal is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        ReciprocalLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

/// Linear relaxation for reciprocal on interval [l, u].
/// Reciprocal f(x) = 1/x is:
/// - Monotonically decreasing for both x > 0 and x < 0
/// - Convex for x > 0 (f''(x) = 2/x³ > 0)
/// - Concave for x < 0 (f''(x) = 2/x³ < 0)
///
/// For convex regions (x > 0): secant is upper bound, tangent is lower bound.
/// For concave regions (x < 0): secant is lower bound, tangent is upper bound.
// Bit-identical tangent anchor: f32::midpoint rounds differently at overflow/subnormal edges.
#[allow(clippy::manual_midpoint)]
pub(crate) fn reciprocal_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    reciprocal_linear_relaxation_with_alpha(l, u, 0.5 * (l + u))
}

/// Alpha-aware linear relaxation for reciprocal on interval [l, u].
///
/// For the positive domain (convex): the tangent at `mid` is the lower bound
/// (optimizable), and the secant/chord is the upper bound (fixed).
/// For the negative domain (concave): the tangent at `mid` is the upper bound
/// (optimizable), and the secant/chord is the lower bound (fixed).
///
/// `mid` is the optimizable tangent point (alpha parameter). When
/// `mid = 0.5 * (l + u)` this reproduces the fixed midpoint tangent
/// from `reciprocal_linear_relaxation`.
///
/// Reference: alpha-beta-CROWN `operators/convex_concave.py:66-196`
pub(crate) fn reciprocal_linear_relaxation_with_alpha(
    l: f32,
    u: f32,
    mid: f32,
) -> LinearRelaxation {
    // Guard: NaN bounds → return conservative intercepts so downstream
    // propagation widens to ±inf instead of silently absorbing NaN.
    // Pattern: every other *_linear_relaxation function has this guard.
    if l.is_nan() || u.is_nan() {
        return LinearRelaxation::nan_fallback();
    }

    #[inline]
    fn round_line_to_f32(
        slope: f64,
        intercept: f64,
        l: f64,
        u: f64,
        interval_positive: bool,
        is_lower: bool,
    ) -> (f32, f32) {
        let rounded_slope = if is_lower {
            if interval_positive {
                // x >= 0: smaller slope lowers the lower line.
                next_down_f32(slope as f32)
            } else {
                // x <= 0: larger slope lowers the lower line.
                next_up_f32(slope as f32)
            }
        } else if interval_positive {
            // x >= 0: larger slope raises the upper line.
            next_up_f32(slope as f32)
        } else {
            // x <= 0: smaller slope raises the upper line.
            next_down_f32(slope as f32)
        };

        // After slope rounding, choose the tightest intercept that still keeps
        // the rounded line conservative for both interval endpoints.
        //
        // If L(x)=s*x+b is exact and R(x)=s_r*x+b_r is rounded, then:
        // R(x) <= L(x) (lower)  iff b_r <= (s - s_r)*x + b
        // R(x) >= L(x) (upper)  iff b_r >= (s - s_r)*x + b
        // The RHS is linear in x, so endpoint extrema are sufficient.
        let rounded_slope_f64 = f64::from(rounded_slope);
        let intercept_at_l = (slope - rounded_slope_f64) * l + intercept;
        let intercept_at_u = (slope - rounded_slope_f64) * u + intercept;
        let rounded_intercept = if is_lower {
            next_down_f32(intercept_at_l.min(intercept_at_u) as f32)
        } else {
            next_up_f32(intercept_at_l.max(intercept_at_u) as f32)
        };

        (rounded_slope, rounded_intercept)
    }

    // Handle intervals that cross zero - reciprocal is undefined.
    if l <= 0.0 && u >= 0.0 {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }

    // Use f64 intermediates to prevent catastrophic cancellation.
    // Reciprocal slopes grow as 1/x² near zero (e.g., l=0.01 → slope=-10000),
    // making intercept subtraction (1/l - slope*l) prone to cancellation in f32.
    // Same pattern as Exp fix.
    let l64 = l as f64;
    let u64 = u as f64;

    // Degenerate case: zero-width interval (l == u in f64).
    // Use tangent-at-l for both bounds since there's only one point.
    if l64 == u64 {
        let slope = -1.0 / (l64 * l64);
        let intercept = 1.0 / l64 - slope * l64;
        let interval_positive = l > 0.0;
        let (lower_slope, lower_intercept) =
            round_line_to_f32(slope, intercept, l64, u64, interval_positive, true);
        let (upper_slope, upper_intercept) =
            round_line_to_f32(slope, intercept, l64, u64, interval_positive, false);
        return LinearRelaxation::new(lower_slope, lower_intercept, upper_slope, upper_intercept);
    }

    let recip_l = 1.0 / l64;

    // Secant (chord) through endpoints: slope = (1/u - 1/l) / (u - l) = -1/(l*u).
    let secant_slope = -1.0 / (l64 * u64);
    let secant_intercept = recip_l - secant_slope * l64;

    // Tangent at optimizable point d (clamped to [l, u]).
    let d = (mid as f64).clamp(l64, u64);
    let tangent_slope = -1.0 / (d * d);
    let tangent_intercept = 2.0 / d;

    if l > 0.0 {
        // Convex: secant is upper, tangent is lower.
        let (lower_slope, lower_intercept) =
            round_line_to_f32(tangent_slope, tangent_intercept, l64, u64, true, true);
        let (upper_slope, upper_intercept) =
            round_line_to_f32(secant_slope, secant_intercept, l64, u64, true, false);
        LinearRelaxation::new(lower_slope, lower_intercept, upper_slope, upper_intercept)
    } else {
        // Concave: secant is lower, tangent is upper.
        let (lower_slope, lower_intercept) =
            round_line_to_f32(secant_slope, secant_intercept, l64, u64, false, true);
        let (upper_slope, upper_intercept) =
            round_line_to_f32(tangent_slope, tangent_intercept, l64, u64, false, false);
        LinearRelaxation::new(lower_slope, lower_intercept, upper_slope, upper_intercept)
    }
}

/// #vit-collector-throughput P4: is `[l, u]` the one unbounded shape the armed
/// kernel relaxes — a finite, strictly positive lower end and `u = +inf`?
///
/// This is exactly the pre-activation of the fixed-index softmax's
/// `Reciprocal(ReduceSum(Exp(d)))` once a saturated `Exp` has pushed the sum's
/// IBP upper to `+inf`: the `j*` term is exactly 1 and the rest are `>= 0`, so
/// `l >= 1` is structural (`network/core/graph/softmax_fixed_index.rs`).
#[inline]
fn is_unbounded_positive(l: f32, u: f32) -> bool {
    l > 0.0 && l.is_finite() && u == f32::INFINITY
}

/// #vit-collector-throughput P4: a sound linear relaxation of `y = 1/x` over
/// `x ∈ [l, +inf)` with `l > 0` finite (the caller checks
/// [`is_unbounded_positive`]; any other `l` gets the trivially-true
/// `[-inf, +inf]` pair).
///
/// * Upper: `y <= 0·x + c` with `c = next_up(cast_up(1/l))`. `1/x` is
///   decreasing on `x > 0`, so its supremum over `[l, +inf)` is `1/l`; this is
///   the exact limit of the chord as `u -> +inf` (slope `-1/(l u) -> 0`,
///   intercept `1/l + 1/u -> 1/l`), tight at `x = l`. `1.0 / l` is correctly
///   rounded in f64 (within half an f64 ULP of the real quotient), the directed
///   cast rounds that UP into f32, and `next_up_f32` adds a full f32 ULP, which
///   dominates the f64 division error — so `c >= 1/l` exactly. A quotient past
///   `f32::MAX` casts to `+inf`, a vacuous but sound intercept.
/// * Lower: `y >= 0·x + 0`. Exact: `x > 0` makes `1/x > 0`. It is the tangent
///   "at `t = +inf`"; a finite-`t` tangent (`2/t - x/t²`, valid for all `x > 0`
///   by convexity) would put a negative slope on the saturated input, whose
///   concretization against `+inf` is vacuous, so it is not offered here.
///
/// The generic chord/tangent formula must NOT be used for `u = +inf`: its
/// slopes evaluate to `-0.0`, the directed slope rounding in
/// `round_line_to_f32` moves them to `±1.4e-45`, and against `x = +inf` that
/// concretizes to `±inf` — which is why the dark path refuses the shape.
pub(crate) fn reciprocal_unbounded_upper_relaxation(l: f32) -> LinearRelaxation {
    if !(l > 0.0 && l.is_finite()) {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }
    let upper_intercept = next_up_f32(cast_f64_to_f32_up(1.0_f64 / f64::from(l)));
    LinearRelaxation::new(0.0, 0.0, 0.0, upper_intercept)
}

/// #vit-collector-throughput P4: the armed relaxation — the unbounded branch on
/// `[l, +inf)`, the historical midpoint relaxation everywhere else (so an armed
/// call with no unbounded element is identical to the dark one).
fn reciprocal_linear_relaxation_unbounded_aware(l: f32, u: f32) -> LinearRelaxation {
    if is_unbounded_positive(l, u) {
        reciprocal_unbounded_upper_relaxation(l)
    } else {
        reciprocal_linear_relaxation(l, u)
    }
}

/// The pre-activation domain guard shared by the four CROWN entries.
///
/// Dark (`armed == false`): byte-for-byte the historical guard — any
/// non-finite endpoint or an interval touching zero refuses with
/// `NyError::InvalidSpec(message)`.
///
/// Armed (`bab.reciprocal_unbounded_upper`): `[l, +inf)` with `l > 0` finite is
/// admitted (and counted), and every remaining refusal (NaN, an interval
/// touching zero, any other infinity) is `NyError::UnsupportedOp(message)` —
/// the class the Exp domain guard already uses and every CROWN collector
/// already maps to "keep this target's certified IBP box". `InvalidSpec` is
/// outside those fallback sets, so on vit_2023's pgd rows it aborted the whole
/// verification (`unknown@9.71s`, scratchpad/vbs/diag6/vit.md §4). Changing
/// the class cannot tighten any bound: the refused walk still publishes
/// nothing, and the fallback it now reaches is the sound IBP enclosure.
///
/// Returns the number of admitted unbounded elements.
fn reciprocal_domain_guard(
    pre_activation: &BoundedTensor,
    armed: bool,
    message: &str,
) -> Result<usize> {
    let mut unbounded = 0usize;
    let mut refused = false;
    for (&l, &u) in pre_activation
        .lower()
        .iter()
        .zip(pre_activation.upper().iter())
    {
        if armed && is_unbounded_positive(l, u) {
            unbounded += 1;
            continue;
        }
        if !l.is_finite() || !u.is_finite() || (l <= 0.0 && u >= 0.0) {
            refused = true;
            break;
        }
    }
    if refused {
        if armed {
            crate::layers::collector_kernels::record_reciprocal_refusal();
            return Err(NyError::UnsupportedOp(message.to_string()));
        }
        return Err(NyError::InvalidSpec(message.to_string()));
    }
    if unbounded > 0 {
        crate::layers::collector_kernels::record_reciprocal_unbounded(unbounded);
    }
    Ok(unbounded)
}

impl ReciprocalLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        self.propagate_linear_with_bounds_policy(
            bounds,
            pre_activation,
            crate::layers::collector_kernels::reciprocal_unbounded_armed(),
        )
    }

    /// [`Self::propagate_linear_with_bounds`] with the typed
    /// `bab.reciprocal_unbounded_upper` choice explicit. `armed == false` is
    /// the historical entry byte-for-byte; production passes the latch, tests
    /// pass both values.
    pub(crate) fn propagate_linear_with_bounds_policy(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        armed: bool,
    ) -> Result<LinearBounds> {
        debug!("Reciprocal layer CROWN backward propagation with pre-activation bounds");
        let unbounded = reciprocal_domain_guard(
            pre_activation,
            armed,
            "Reciprocal CROWN requires finite pre-activation bounds that do not cross zero",
        )?;
        if unbounded > 0 {
            return crown_elementwise_backward(
                bounds,
                pre_activation,
                reciprocal_linear_relaxation_unbounded_aware,
            );
        }
        crown_elementwise_backward(bounds, pre_activation, reciprocal_linear_relaxation)
    }

    /// Alpha-parameterized CROWN backward propagation.
    ///
    /// `alpha` contains the tangent point for each neuron on the lower-path
    /// (for positive domain: tangent is the lower bound).
    /// `alpha_upper` optionally provides separate tangent points for the upper path.
    /// Mirrors `SqrtLayer::propagate_linear_with_alpha` structure.
    pub(crate) fn propagate_linear_with_alpha(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
        alpha_upper: Option<&Array1<f32>>,
    ) -> Result<LinearBounds> {
        self.propagate_linear_with_alpha_policy(
            bounds,
            pre_activation,
            alpha,
            alpha_upper,
            crate::layers::collector_kernels::reciprocal_unbounded_armed(),
        )
    }

    /// [`Self::propagate_linear_with_alpha`] with the typed
    /// `bab.reciprocal_unbounded_upper` choice explicit (see
    /// [`reciprocal_domain_guard`]). An unbounded element takes the constant
    /// relaxation of [`reciprocal_unbounded_upper_relaxation`] on BOTH paths,
    /// ignoring its tangent parameter; every other element keeps its
    /// alpha-tangent relaxation exactly.
    pub(crate) fn propagate_linear_with_alpha_policy(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &Array1<f32>,
        alpha_upper: Option<&Array1<f32>>,
        armed: bool,
    ) -> Result<LinearBounds> {
        debug!("Reciprocal layer alpha-CROWN backward propagation");

        // DISCHARGE THE INCOMING COEFFICIENT ERROR BEFORE COMPOSING.
        //
        // The alpha composition below publishes through `LinearBounds::new_or_conservative`,
        // which hardcodes `lower_a_err: None, upper_a_err: None` -- it ERASES the error
        // plane rather than leaving it stale. `bounds/concretize.rs` reads exactly those
        // fields to compute `coefficient_penalty_l/u` and charges ZERO when they are None,
        // and neither production route discharges first, so the error simply vanished.
        // The dropped quantity is the upstream Linear's full Higham charge (gamma_n * S),
        // NOT an ULP -- `linear/crown_single.rs` always publishes an error plane, so any
        // Reciprocal node below a Linear/Conv in the backward walk receives an err-bearing
        // carrier.
        //
        // Fold it into the bias over the OUTPUT box: the incoming carrier's COLUMNS index
        // THIS node's output, not its input, so folding over `pre_activation` would charge
        // the wrong magnitudes. `fold_coeff_err_into_bias` discharges via
        // `certified_affine_sum_f32` and falls back to the conservative row-degrade if the
        // box cannot be mapped onto the columns, so it fails closed either way.
        //
        // PARTIAL: this discharges the INCOMING error (the unbounded-magnitude half). It
        // does not yet publish a FRESH gap for this relaxation's own directed-rounding
        // error, which is ULP-scale. The certified sibling
        // `crown_elementwise_backward_dual_indexed` (trigonometric/s_shaped/shared.rs:563)
        // is the shape to lift for that.
        let discharged_bounds;
        let bounds = if bounds.has_coeff_err() {
            let out_box = self.propagate_ibp(pre_activation)?;
            let out_flat = out_box.flatten();
            let out_l: Vec<f32> = out_flat.lower().iter().copied().collect();
            let out_u: Vec<f32> = out_flat.upper().iter().copied().collect();
            let mut owned = bounds.clone();
            owned.fold_coeff_err_into_bias(&out_l, &out_u);
            discharged_bounds = owned;
            &discharged_bounds
        } else {
            bounds
        };
        let unbounded = reciprocal_domain_guard(
            pre_activation,
            armed,
            "Reciprocal alpha-CROWN requires finite pre-activation bounds that do not cross zero",
        )?;

        let pre_flat = pre_activation.flatten();
        let pre_lower = pre_flat
            .lower()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.lower().shape().to_vec(),
            })?;
        let pre_upper = pre_flat
            .upper()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![pre_flat.len()],
                got: pre_flat.upper().shape().to_vec(),
            })?;
        let num_neurons = pre_lower.len();
        if bounds.num_inputs() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }
        if alpha.len() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![alpha.len()],
            });
        }
        if let Some(alpha_upper) = alpha_upper {
            if alpha_upper.len() != num_neurons {
                return Err(NyError::ShapeMismatch {
                    expected: vec![num_neurons],
                    got: vec![alpha_upper.len()],
                });
            }
        }

        let pre_lower_cow = crate::contiguous_flat_slice(&pre_lower);
        let pre_upper_cow = crate::contiguous_flat_slice(&pre_upper);
        let pre_lower_slice = pre_lower_cow.as_ref();
        let pre_upper_slice = pre_upper_cow.as_ref();
        let lower_path_relaxations =
            precompute_relaxations(pre_lower_slice, pre_upper_slice, &|l, u, idx| {
                if unbounded > 0 && is_unbounded_positive(l, u) {
                    return reciprocal_unbounded_upper_relaxation(l);
                }
                reciprocal_linear_relaxation_with_alpha(l, u, alpha[idx])
            });
        let upper_path_relaxations =
            precompute_relaxations(pre_lower_slice, pre_upper_slice, &|l, u, idx| {
                if unbounded > 0 && is_unbounded_positive(l, u) {
                    return reciprocal_unbounded_upper_relaxation(l);
                }
                reciprocal_linear_relaxation_with_alpha(
                    l,
                    u,
                    alpha_upper.map_or(alpha[idx], |upper| upper[idx]),
                )
            });

        let num_outputs = bounds.num_outputs();
        let mut new_lower_a = ndarray::Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
        let mut new_upper_a = ndarray::Array2::<f32>::zeros((num_outputs, num_neurons));
        let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);
        let mut lower_nonfinite_rows = vec![false; num_outputs];
        let mut upper_nonfinite_rows = vec![false; num_outputs];

        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let lr = compose_lower(bounds.lower_a()[[j, i]], &lower_path_relaxations[i]);
                new_lower_a[[j, i]] = lr.new_coeff;
                new_lower_b_f64[j] = add_f64_down(new_lower_b_f64[j], lr.intercept_contrib);
                lower_nonfinite_rows[j] |= lr.nonfinite;

                let ur = compose_upper(bounds.upper_a()[[j, i]], &upper_path_relaxations[i]);
                new_upper_a[[j, i]] = ur.new_coeff;
                new_upper_b_f64[j] = add_f64_up(new_upper_b_f64[j], ur.intercept_contrib);
                upper_nonfinite_rows[j] |= ur.nonfinite;
            }
        }

        let lower_affected = lower_nonfinite_rows.iter().filter(|&&row| row).count();
        let upper_affected = upper_nonfinite_rows.iter().filter(|&&row| row).count();
        log_nonfinite_fallback(
            "Reciprocal-alpha",
            lower_affected,
            upper_affected,
            num_outputs,
        );

        let mut new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
        let mut new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));
        for j in 0..num_outputs {
            if lower_nonfinite_rows[j] {
                for i in 0..num_neurons {
                    new_lower_a[[j, i]] = 0.0;
                }
                new_lower_b[j] = f32::NEG_INFINITY;
            }
            if upper_nonfinite_rows[j] {
                for i in 0..num_neurons {
                    new_upper_a[[j, i]] = 0.0;
                }
                new_upper_b[j] = f32::INFINITY;
            }
        }

        LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("Reciprocal layer batched CROWN backward propagation");
        let unbounded = reciprocal_domain_guard(
            pre_activation,
            crate::layers::collector_kernels::reciprocal_unbounded_armed(),
            "Reciprocal batched CROWN requires finite pre-activation bounds that do not cross zero",
        )?;
        if unbounded > 0 {
            return crown_elementwise_backward_batched(
                bounds,
                pre_activation,
                reciprocal_linear_relaxation_unbounded_aware,
            );
        }
        crown_elementwise_backward_batched(bounds, pre_activation, reciprocal_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        let unbounded = reciprocal_domain_guard(
            pre_activation,
            crate::layers::collector_kernels::reciprocal_unbounded_armed(),
            "Reciprocal Patches CROWN requires finite pre-activation bounds that do not cross zero",
        )?;
        if unbounded > 0 {
            return crown_elementwise_backward_patches(
                bounds,
                pre_activation,
                reciprocal_linear_relaxation_unbounded_aware,
            );
        }
        crown_elementwise_backward_patches(bounds, pre_activation, reciprocal_linear_relaxation)
    }
}

/// #vit-collector-throughput P4 soundness pins: the `[l, +inf)` relaxation,
/// the armed CROWN entries end to end, the refusal class, and the dark path.
///
/// Every containment check here is in EXACT rational arithmetic (every finite
/// f32 is a dyadic rational), so no floating tolerance can hide a violation.
#[cfg(test)]
mod unbounded_upper_tests {
    use super::*;
    use ndarray::Array2;
    use num_rational::BigRational;
    use num_traits::{One, Signed, Zero};

    /// SplitMix64: deterministic, dependency-free, bit-reproducible.
    struct SplitMix(u64);

    impl SplitMix {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in `[0, 1)`.
        fn unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
        }

        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            (f64::from(lo) + (f64::from(hi) - f64::from(lo)) * self.unit()) as f32
        }
    }

    fn rat(value: f32) -> BigRational {
        BigRational::from_float(f64::from(value)).expect("finite f32 is an exact rational")
    }

    /// The lower ends the brief names, plus the edges of the f32 range: the
    /// smallest normal (`1/l` still finite in f32) and a subnormal (`1/l`
    /// overflows f32, so the upper intercept must saturate to `+inf`).
    fn lower_ends() -> Vec<f32> {
        vec![
            1.0,
            next_up_f32(1.0),
            next_down_f32(1.0),
            3.7,
            0.25,
            1e30,
            f32::MIN_POSITIVE,
            f32::from_bits(0x0000_1000),
        ]
    }

    /// A point of `[l, f32::MAX]`, log-uniform, as an f32 inside the range.
    fn sample_half_line(rng: &mut SplitMix, l: f32) -> f32 {
        let lo = f64::from(l).log2();
        let hi = f64::from(f32::MAX).log2();
        let t = lo + (hi - lo) * rng.unit();
        (t.exp2() as f32).clamp(l, f32::MAX)
    }

    fn box1(lower: f32, upper: f32) -> BoundedTensor {
        BoundedTensor::new_allow_infinite(
            ArrayD::from_elem(IxDyn(&[1]), lower),
            ArrayD::from_elem(IxDyn(&[1]), upper),
        )
        .expect("well-formed box")
    }

    /// `c` is a certified upper bound on `1/x` at `x` (exactly: `c·x >= 1`).
    fn upper_covers(c: f32, x: f32) -> bool {
        c == f32::INFINITY || rat(c) * rat(x) >= BigRational::one()
    }

    /// (1) The relaxation itself: slopes exactly 0, lower intercept exactly 0,
    /// and the upper intercept `c >= 1/x` for every `x` in `[l, f32::MAX]` —
    /// checked at `x = l` (where it is tight) and at log-uniform samples — in
    /// exact rationals. A finite `c` also sits within a few f32 ULPs of `1/l`
    /// (tightness), and it is `+inf` only when `1/l` exceeds `f32::MAX`.
    #[test]
    fn reciprocal_unbounded_upper_relaxation_contains_samples() {
        let mut rng = SplitMix(0x5EED_4EC1_2026_0926);
        for l in lower_ends() {
            assert!(is_unbounded_positive(l, f32::INFINITY), "l = {l:e}");
            let relax = reciprocal_unbounded_upper_relaxation(l);
            assert_eq!(relax.lower_slope.to_bits(), 0.0_f32.to_bits(), "l = {l:e}");
            assert_eq!(relax.upper_slope.to_bits(), 0.0_f32.to_bits(), "l = {l:e}");
            assert_eq!(
                relax.lower_intercept.to_bits(),
                0.0_f32.to_bits(),
                "l = {l:e}"
            );
            let c = relax.upper_intercept;
            assert!(!c.is_nan(), "l = {l:e}");
            if c.is_finite() {
                assert!(upper_covers(c, l), "c = {c:e} < 1/l at l = {l:e}");
                // Tightness: at most 3 f32 ULPs above the exact 1/l.
                let slack = rat(next_up_f32(next_up_f32(next_up_f32(
                    (1.0_f64 / f64::from(l)) as f32,
                ))));
                assert!(rat(c) <= slack, "c = {c:e} is loose at l = {l:e}");
            } else {
                assert_eq!(c, f32::INFINITY);
                assert!(
                    rat(l) * rat(f32::MAX) < BigRational::one(),
                    "c saturated to +inf although 1/l = {} fits in f32",
                    1.0 / f64::from(l)
                );
            }
            for sample in 0..256 {
                let x = match sample {
                    0 => l,
                    1 => f32::MAX,
                    _ => sample_half_line(&mut rng, l),
                };
                assert!(x >= l && x.is_finite());
                // Lower: 0 <= 1/x, exact since x > 0.
                assert!(rat(x) > BigRational::zero());
                assert!(
                    upper_covers(c, x),
                    "1/x > c = {c:e} at x = {x:e}, l = {l:e}"
                );
            }
        }
        // Outside the admitted shape the helper answers the trivially-true
        // pair rather than anything that could be read as a bound.
        for l in [0.0_f32, -1.0, f32::INFINITY, f32::NAN] {
            let relax = reciprocal_unbounded_upper_relaxation(l);
            assert_eq!(relax.lower_intercept, f32::NEG_INFINITY);
            assert_eq!(relax.upper_intercept, f32::INFINITY);
        }
    }

    /// (2) The relaxation agrees with IBP on the output range: both contain
    /// `[0, 1/l]`, the CROWN upper intercept is never below the IBP upper
    /// end and at most one f32 ULP above it, and the CROWN lower (0) is never
    /// below the IBP lower end.
    #[test]
    fn reciprocal_unbounded_ibp_matches_crown_range() {
        let layer = ReciprocalLayer::new();
        for l in lower_ends() {
            let ibp = layer
                .propagate_ibp(&box1(l, f32::INFINITY))
                .expect("IBP accepts [l, +inf)");
            let ibp_lower = ibp.lower().iter().copied().next().expect("one element");
            let ibp_upper = ibp.upper().iter().copied().next().expect("one element");
            let relax = reciprocal_unbounded_upper_relaxation(l);
            assert!(ibp_lower <= 0.0, "IBP lower {ibp_lower:e} at l = {l:e}");
            assert!(relax.lower_intercept >= ibp_lower);
            if ibp_upper.is_finite() {
                assert!(
                    relax.upper_intercept >= ibp_upper
                        && relax.upper_intercept <= next_up_f32(ibp_upper),
                    "CROWN upper {:e} vs IBP upper {ibp_upper:e} at l = {l:e}",
                    relax.upper_intercept
                );
            } else {
                assert_eq!(relax.upper_intercept, f32::INFINITY, "l = {l:e}");
            }
        }
    }

    /// Exact check of one published relation pair against `C·(1/x) + d` at a
    /// finite point, charging each certified coefficient error against `|x|`
    /// in the pessimistic direction (what concretization does).
    fn assert_relation_contains(
        out: &LinearBounds,
        coeffs: &Array2<f32>,
        bias: &[f32],
        x: &[f32],
        label: &str,
    ) {
        let rows = coeffs.nrows();
        for r in 0..rows {
            let mut truth = rat(bias[r]);
            for (j, &xj) in x.iter().enumerate() {
                if coeffs[[r, j]] != 0.0 {
                    truth += rat(coeffs[[r, j]]) / rat(xj);
                }
            }
            let lb = out.lower_b()[r];
            assert!(
                lb != f32::INFINITY && !lb.is_nan(),
                "{label}: row {r} lower bias {lb}"
            );
            if lb.is_finite() {
                let mut low = rat(lb);
                for (j, &xj) in x.iter().enumerate() {
                    low += rat(out.lower_a()[[r, j]]) * rat(xj);
                    if let Some(err) = out.lower_a_err() {
                        low -= rat(err[[r, j]]) * rat(xj).abs();
                    }
                }
                assert!(
                    low <= truth,
                    "{label}: row {r} lower relation above 1/x at {x:?}"
                );
            }
            let ub = out.upper_b()[r];
            assert!(
                ub != f32::NEG_INFINITY && !ub.is_nan(),
                "{label}: row {r} upper bias {ub}"
            );
            if ub.is_finite() {
                let mut up = rat(ub);
                for (j, &xj) in x.iter().enumerate() {
                    up += rat(out.upper_a()[[r, j]]) * rat(xj);
                    if let Some(err) = out.upper_a_err() {
                        up += rat(err[[r, j]]) * rat(xj).abs();
                    }
                }
                assert!(
                    truth <= up,
                    "{label}: row {r} upper relation below 1/x at {x:?}"
                );
            }
        }
    }

    /// (3) End to end through BOTH armed CROWN entries (the midpoint entry and
    /// the alpha entry): a pre-activation mixing `[l, +inf)` elements with
    /// bounded positive, bounded negative and point elements is RELAXED (the
    /// dark entry refuses it), and the published relation contains the exact
    /// `C·(1/x) + d` at sampled finite points, box corners and `x = l`.
    #[test]
    fn reciprocal_unbounded_crown_backward_contains_samples() {
        let mut rng = SplitMix(0x0DD5_EED5_4EC1_0001);
        let layer = ReciprocalLayer::new();
        let ends = lower_ends();
        let n = 5usize;
        let rows = 3usize;
        for case in 0..48 {
            let mut lower = vec![0.0f32; n];
            let mut upper = vec![0.0f32; n];
            // Elements 0 and 1: [l, +inf) (the saturated denominator shape).
            for j in 0..2 {
                lower[j] = ends[(case + 3 * j) % ends.len()];
                upper[j] = f32::INFINITY;
            }
            // Element 2: bounded positive; 3: bounded negative; 4: a point.
            let a = rng.range(0.1, 10.0);
            let w = rng.range(0.0, 5.0);
            lower[2] = a;
            upper[2] = a + w;
            lower[3] = -(a + w);
            upper[3] = -a;
            let p = rng.range(0.5, 2.0);
            lower[4] = p;
            upper[4] = p;
            let pre = BoundedTensor::new_allow_infinite(
                ArrayD::from_shape_vec(IxDyn(&[n]), lower.clone()).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[n]), upper.clone()).unwrap(),
            )
            .unwrap();
            let coeffs = Array2::from_shape_fn((rows, n), |_| {
                if rng.unit() < 0.2 {
                    0.0
                } else {
                    rng.range(-2.0, 2.0)
                }
            });
            let bias: Vec<f32> = (0..rows).map(|_| rng.range(-1.0, 1.0)).collect();
            let incoming = LinearBounds::new(
                coeffs.clone(),
                Array1::from_vec(bias.clone()),
                coeffs.clone(),
                Array1::from_vec(bias.clone()),
            )
            .unwrap();
            let alpha = Array1::from_shape_fn(n, |j| {
                if upper[j].is_finite() {
                    lower[j] + (upper[j] - lower[j]) * rng.unit() as f32
                } else {
                    lower[j] * 2.0
                }
            });

            // The dark entries keep the historical refusal (the vit pgd abort).
            assert!(matches!(
                layer.propagate_linear_with_bounds_policy(&incoming, &pre, false),
                Err(NyError::InvalidSpec(_))
            ));
            assert!(matches!(
                layer.propagate_linear_with_alpha_policy(&incoming, &pre, &alpha, None, false),
                Err(NyError::InvalidSpec(_))
            ));

            let midpoint = layer
                .propagate_linear_with_bounds_policy(&incoming, &pre, true)
                .expect("the armed midpoint entry relaxes [l, +inf)");
            let with_alpha = layer
                .propagate_linear_with_alpha_policy(&incoming, &pre, &alpha, None, true)
                .expect("the armed alpha entry relaxes [l, +inf)");

            let mut points: Vec<Vec<f32>> = Vec::new();
            // Tight corner of every element: x = lower end.
            points.push(lower.clone());
            for _ in 0..24 {
                points.push(
                    (0..n)
                        .map(|j| {
                            if upper[j].is_finite() {
                                (lower[j] + (upper[j] - lower[j]) * rng.unit() as f32)
                                    .clamp(lower[j], upper[j])
                            } else {
                                sample_half_line(&mut rng, lower[j])
                            }
                        })
                        .collect(),
                );
            }
            for x in &points {
                assert_relation_contains(&midpoint, &coeffs, &bias, x, "midpoint entry");
                assert_relation_contains(&with_alpha, &coeffs, &bias, x, "alpha entry");
            }
        }
    }

    /// (4) The refusal CLASS. Dark: every out-of-domain pre-activation is the
    /// historical `InvalidSpec`, on both entries. Armed: the same inputs are
    /// `UnsupportedOp` — the class every CROWN collector maps to "keep this
    /// target's IBP box" — including a call where one element IS admissible
    /// (`[1, +inf)`) and another is not (the whole call is refused).
    #[test]
    fn reciprocal_domain_refusal_is_unsupported_op_not_invalid_spec() {
        let layer = ReciprocalLayer::new();
        let refused = [
            (-1.0_f32, 2.0_f32),
            (0.0, f32::INFINITY),
            (-2.0, 0.0),
            (0.0, 0.0),
            (f32::NEG_INFINITY, -1.0),
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::INFINITY, f32::INFINITY),
        ];
        let incoming = LinearBounds::identity(1);
        let alpha = Array1::from_elem(1, 1.0_f32);
        for (l, u) in refused {
            let pre = box1(l, u);
            assert!(
                matches!(
                    layer.propagate_linear_with_bounds_policy(&incoming, &pre, false),
                    Err(NyError::InvalidSpec(_))
                ),
                "dark midpoint entry at [{l}, {u}]"
            );
            assert!(
                matches!(
                    layer.propagate_linear_with_alpha_policy(&incoming, &pre, &alpha, None, false),
                    Err(NyError::InvalidSpec(_))
                ),
                "dark alpha entry at [{l}, {u}]"
            );
            assert!(
                matches!(
                    layer.propagate_linear_with_bounds_policy(&incoming, &pre, true),
                    Err(NyError::UnsupportedOp(_))
                ),
                "armed midpoint entry at [{l}, {u}]"
            );
            assert!(
                matches!(
                    layer.propagate_linear_with_alpha_policy(&incoming, &pre, &alpha, None, true),
                    Err(NyError::UnsupportedOp(_))
                ),
                "armed alpha entry at [{l}, {u}]"
            );
        }

        // Mixed: one admissible unbounded element, one crossing zero.
        let mixed = BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, -1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 1.0]).unwrap(),
        )
        .unwrap();
        let incoming2 = LinearBounds::identity(2);
        assert!(matches!(
            layer.propagate_linear_with_bounds_policy(&incoming2, &mixed, true),
            Err(NyError::UnsupportedOp(_))
        ));
        assert!(matches!(
            layer.propagate_linear_with_bounds_policy(&incoming2, &mixed, false),
            Err(NyError::InvalidSpec(_))
        ));
    }

    /// (5) DARK BY CONSTRUCTION where it matters: with no `[l, +inf)` element
    /// the armed entries are BIT-IDENTICAL to the dark ones (same relaxation,
    /// same composition), with and without an incoming coefficient-error plane.
    #[test]
    fn reciprocal_armed_without_unbounded_elements_is_bit_identical_to_dark() {
        let mut rng = SplitMix(0xB17_1DE7_1CA1_0005);
        let layer = ReciprocalLayer::new();
        let n = 6usize;
        let rows = 4usize;
        for case in 0..32 {
            let mut lower = vec![0.0f32; n];
            let mut upper = vec![0.0f32; n];
            for j in 0..n {
                let a = rng.range(0.05, 20.0);
                let w = rng.range(0.0, 10.0);
                if (j + case) % 2 == 0 {
                    lower[j] = a;
                    upper[j] = a + w;
                } else {
                    lower[j] = -(a + w);
                    upper[j] = -a;
                }
            }
            let pre = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[n]), lower.clone()).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[n]), upper.clone()).unwrap(),
            )
            .unwrap();
            let la = Array2::from_shape_fn((rows, n), |_| rng.range(-3.0, 3.0));
            let ua = Array2::from_shape_fn((rows, n), |_| rng.range(-3.0, 3.0));
            let lb = Array1::from_shape_fn(rows, |_| rng.range(-1.0, 1.0));
            let ub = Array1::from_shape_fn(rows, |_| rng.range(-1.0, 1.0));
            let incoming = if case % 2 == 0 {
                LinearBounds::new(la, lb, ua, ub).unwrap()
            } else {
                let le = Array2::from_shape_fn((rows, n), |_| rng.range(0.0, 1e-4));
                let ue = Array2::from_shape_fn((rows, n), |_| rng.range(0.0, 1e-4));
                LinearBounds::new_or_conservative_with_err(la, lb, ua, ub, le, ue).unwrap()
            };
            let alpha =
                Array1::from_shape_fn(n, |j| lower[j] + (upper[j] - lower[j]) * rng.unit() as f32);

            let bits = |b: &LinearBounds| -> Vec<u32> {
                let mut v: Vec<u32> = b.lower_a().iter().map(|x| x.to_bits()).collect();
                v.extend(b.upper_a().iter().map(|x| x.to_bits()));
                v.extend(b.lower_b().iter().map(|x| x.to_bits()));
                v.extend(b.upper_b().iter().map(|x| x.to_bits()));
                for err in [b.lower_a_err(), b.upper_a_err()].into_iter().flatten() {
                    v.extend(err.iter().map(|x| x.to_bits()));
                }
                v
            };
            let dark = layer
                .propagate_linear_with_bounds_policy(&incoming, &pre, false)
                .unwrap();
            let armed = layer
                .propagate_linear_with_bounds_policy(&incoming, &pre, true)
                .unwrap();
            assert_eq!(bits(&dark), bits(&armed), "midpoint entry, case {case}");
            assert_eq!(dark.lower_a_err().is_some(), armed.lower_a_err().is_some());

            let dark = layer
                .propagate_linear_with_alpha_policy(&incoming, &pre, &alpha, None, false)
                .unwrap();
            let armed = layer
                .propagate_linear_with_alpha_policy(&incoming, &pre, &alpha, None, true)
                .unwrap();
            assert_eq!(bits(&dark), bits(&armed), "alpha entry, case {case}");
        }
    }
}
