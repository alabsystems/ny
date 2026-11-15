// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{checked_dim_product, checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::shape::broadcast_shapes;
use crate::{contiguous_flat_slice, BatchedLinearBounds, LinearBounds, MulBinaryRelaxationMode};

/// Element-wise multiplication layer for two bounded tensors (e.g., SwiGLU gating).
///
/// For z = x * y where both x and y are bounded, uses IBP-style interval arithmetic:
/// - z_l = min(x_l*y_l, x_l*y_u, x_u*y_l, x_u*y_u)
/// - z_u = max(x_l*y_l, x_l*y_u, x_u*y_l, x_u*y_u)
#[derive(Debug, Clone)]
pub struct MulBinaryLayer;

/// Bound direction for McCormick envelope plane selection.
///
/// Shared by `mul` (element-wise) and `matmul` (batched bilinear CROWN).
#[derive(Clone, Copy)]
pub(super) enum BoundDir {
    Lower,
    Upper,
}

/// Select the tightest McCormick envelope plane for bilinear z = x·y.
///
/// Returns `(coeff_x, coeff_y, const_term)` for the affine relaxation plane.
///
/// McCormick envelope for z = x·y over box [lx, ux] × [ly, uy]:
/// - Lower planes: L1: ly·x + lx·y − lx·ly,  L2: uy·x + ux·y − ux·uy
/// - Upper planes: U1: uy·x + lx·y − lx·uy,  U2: ly·x + ux·y − ux·ly
///
/// Selection depends on the incoming CROWN weight `w` and bound direction:
/// positive weight wants tight same-direction plane, negative weight flips.
///
/// Reference: McCormick (1976), "Computability of global solutions to factorable
/// nonconvex programs". Used in auto_LiRPA/operators/bivariate.py.
#[inline]
// Justification: McCormick envelope selection requires variable bounds (lx, ux, ly, uy),
// evaluation point (x0, y0), incoming weight (w), and bound direction — 8 parameters
// that map directly to the mathematical formulation (McCormick 1976).
#[allow(clippy::too_many_arguments)]
pub(super) fn select_mccormick_plane(
    lx: f32,
    ux: f32,
    ly: f32,
    uy: f32,
    x0: f32,
    y0: f32,
    w: f32,
    dir: BoundDir,
) -> (f32, f32, f32) {
    // Non-finite guard: if any input bound or evaluation point is NaN or
    // infinite, return conservative trivial bounds rather than propagating
    // NaN coefficients. Infinity is included because McCormick products like
    // `-lx * ly` can produce NaN via `0 * inf` when one bound is zero and
    // another is infinite.
    // Lower → z >= -∞ (trivially true), Upper → z <= +∞ (trivially true).
    // Reference: auto_LiRPA/operators/bivariate.py:161-186 (softmax-only
    // output-level NaN guard sets similar trivial bounds).
    if !lx.is_finite()
        || !ux.is_finite()
        || !ly.is_finite()
        || !uy.is_finite()
        || !x0.is_finite()
        || !y0.is_finite()
    {
        return match dir {
            BoundDir::Lower => (0.0, 0.0, f32::NEG_INFINITY),
            BoundDir::Upper => (0.0, 0.0, f32::INFINITY),
        };
    }

    let l1 = (ly, lx, -lx * ly, lx * y0 + ly * x0 - lx * ly);
    let l2 = (uy, ux, -ux * uy, ux * y0 + uy * x0 - ux * uy);
    let u1 = (uy, lx, -lx * uy, lx * y0 + uy * x0 - lx * uy);
    let u2 = (ly, ux, -ux * ly, ux * y0 + ly * x0 - ux * ly);

    match dir {
        BoundDir::Lower => {
            if w >= 0.0 {
                // w * lower(z): choose the larger lower plane at reference point
                if l1.3 >= l2.3 {
                    (l1.0, l1.1, l1.2)
                } else {
                    (l2.0, l2.1, l2.2)
                }
            } else {
                // w * upper(z) for lower bound: choose the smaller upper plane
                if u1.3 <= u2.3 {
                    (u1.0, u1.1, u1.2)
                } else {
                    (u2.0, u2.1, u2.2)
                }
            }
        }
        BoundDir::Upper => {
            if w >= 0.0 {
                // w * upper(z): choose the smaller upper plane at reference point
                if u1.3 <= u2.3 {
                    (u1.0, u1.1, u1.2)
                } else {
                    (u2.0, u2.1, u2.2)
                }
            } else {
                // w * lower(z) for upper bound: choose the larger lower plane
                if l1.3 >= l2.3 {
                    (l1.0, l1.1, l1.2)
                } else {
                    (l2.0, l2.1, l2.2)
                }
            }
        }
    }
}

/// Widen one accumulated McCormick/envelope CONSTANT outward by a certified
/// bound on its own floating-point error (#mul-mccormick-const).
///
/// # The defect this closes
///
/// Every `propagate_linear_*` driver below folds the envelope constant as
/// `const += w * c` in PLAIN round-to-nearest f64 and published it with no error
/// term at all — only the final directed `f64 -> f32` cast. That cast covers the
/// NARROWING and nothing else, so two distinct errors reached the verdict path
/// uncharged, in the very functions whose COEFFICIENT half is already certified
/// through `new_or_conservative_with_err` / `set_coeff_err`:
///
/// 1. **f64 accumulation.** Recursive summation of `adds + 1` summands satisfies
///    `|fl(S) - S| <= gamma^f64_{adds} * sum|summand|` (Higham, *Accuracy and
///    Stability*, Thm 3.1). The charge scales with the MAGNITUDE sum, not with
///    `|S|`, so it does NOT vanish under cancellation — which is exactly when a
///    result-scaled 1-ULP cast allowance under-counts it. Concretely, terms
///    `+2^60`, `4096 x -1`, `-2^60` fold to EXACTLY `0.0` in f64 (`ulp(2^60) =
///    256` swallows every `-1`) where the true constant is `-4096`; the old
///    publication emitted `next_down_f32(0.0) = -1.4e-45`, a lower plane sitting
///    `4096` ABOVE the true one.
/// 2. **The plane constant itself.** In `McCormick` mode the `c` reaching this
///    fold is `fl32(-lx*ly)` (or one of the three sibling corner products from
///    [`select_mccormick_plane`]) — ONE round-to-nearest f32 multiply away from
///    the exact McCormick constant, with NO outward margin. The exact lower plane
///    `L1` satisfies `x*y - (ly*x + lx*y - lx*ly) = (x-lx)(y-ly) >= 0` with ZERO
///    slack at the corner `(lx, ly)`, so a `c` that rounded UP by even half an
///    ULP lifts the "lower" plane strictly ABOVE the bilinear surface there.
///    (The `Middle`/alpha modes route through
///    `bilinear::mccormick::certify_plane_constant`, which already rounds each
///    plane constant OUTWARD, so for them this second charge is redundant slack
///    rather than a repair. It is still applied: one uniform certificate cannot
///    drift out of sync with the mode dispatch above it.)
///
/// # The certificate
///
/// ```text
/// S_safe        = magnitude_sum      / (1 - gamma^f64_{adds+1})
/// S_terms_safe  = term_magnitude_sum / (1 - gamma^f64_{adds+1})
/// err           = gamma^f64_{adds+1} * S_safe  +  gamma^f32_3 * S_terms_safe
/// out           = down(accumulated - err)   /   up(accumulated + err)
/// ```
///
/// The `1/(1 - gamma)` inflation mirrors `conv2d::bound` and the certified twin
/// `MatMulLayer::propagate_linear_binary` (`binary_ops/matmul/crown_dense.rs`,
/// #matmul-dense-mccormick-const), which folds the SAME `sum w*c` over planes
/// from the SAME [`select_mccormick_plane`] and must certify the same thing:
/// `magnitude_sum` is itself an f64 round-to-nearest sum of non-negative values
/// and can fall SHORT of the true magnitude sum by its own accumulation deficit,
/// so charging against the raw figure could under-bound the very error it is
/// meant to cover. The final directed f64 step absorbs the rounding of the
/// widening subtract/add.
///
/// # Direction
///
/// `err >= 0` always (`gamma >= 0`, both magnitude sums are sums of `abs`
/// values, `inflate >= 1`), so a `Lower` constant only ever moves DOWN and an
/// `Upper` constant only ever moves UP. `adds == 0` returns the input
/// bit-identically: no f64 addition ran and no plane was consumed, so there is
/// genuinely nothing to charge. A NaN anywhere fails CLOSED to the infinite
/// constant, which is strictly safer than the previous behaviour (a NaN bias
/// tripped `validate_no_nan` and degraded the WHOLE relation, both directions,
/// to `conservative`).
fn widen_mccormick_constant(
    accumulated: f64,
    magnitude_sum: f64,
    term_magnitude_sum: f64,
    adds: usize,
    dir: BoundDir,
) -> f64 {
    let trivial = match dir {
        BoundDir::Lower => f64::NEG_INFINITY,
        BoundDir::Upper => f64::INFINITY,
    };
    if adds == 0 {
        // `accumulated` is the exact f64 image of the incoming f32 bias.
        return accumulated;
    }
    let gamma_acc = crate::layers::linear::crown_single_gamma_n_f64(adds.saturating_add(1));
    let inflate = if gamma_acc < 1.0 {
        1.0 / (1.0 - gamma_acc)
    } else {
        f64::INFINITY
    };
    // Two f32 roundings per term are possible (the plane's own corner product,
    // and any f32 scaling folded into it) plus one step of slack.
    let gamma_plane = crate::layers::linear::crown_single_gamma_n_f32(3);
    let err = gamma_acc * (magnitude_sum * inflate) + gamma_plane * (term_magnitude_sum * inflate);
    if err.is_nan() {
        return trivial;
    }
    if err <= 0.0 {
        return accumulated;
    }
    let widened = match dir {
        BoundDir::Lower => accumulated - err,
        BoundDir::Upper => accumulated + err,
    };
    if widened.is_nan() {
        return trivial;
    }
    match dir {
        BoundDir::Lower => widened.next_down(),
        BoundDir::Upper => widened.next_up(),
    }
}

/// Running McCormick/envelope CONSTANT fold that carries its own Higham
/// certificate (#mul-mccormick-const).
///
/// Replaces the bare `const_lower += w as f64 * c as f64` chains in all three
/// CROWN-backward drivers below. Every driver folds the same shape — an incoming
/// f32 bias plus one `w * plane_const` product per inner-loop step — so the
/// certificate lives in ONE place and cannot drift between the scalar, alpha and
/// batched arms the way the uncertified `+=` chains did.
///
/// `magnitude_sum` carries the incoming bias (the f64 accumulation charge is
/// against EVERY summand); `term_magnitude_sum` excludes it (the per-term f32
/// plane-rounding charge applies only to the plane terms). `adds` counts the f64
/// additions actually performed, so a term-free fold charges exactly zero and is
/// published bit-identically to the historical directed-cast result.
struct McCormickConstFold {
    accumulated: f64,
    magnitude_sum: f64,
    term_magnitude_sum: f64,
    adds: usize,
    dir: BoundDir,
}

impl McCormickConstFold {
    /// Seed the fold with the incoming relation's f32 bias (exact in f64).
    #[inline]
    fn new(bias: f32, dir: BoundDir) -> Self {
        let accumulated = f64::from(bias);
        Self {
            accumulated,
            magnitude_sum: accumulated.abs(),
            term_magnitude_sum: 0.0,
            adds: 0,
            dir,
        }
    }

    /// Fold in one `w * plane_const` term and record its magnitude.
    ///
    /// The product is EXACT in f64 (an f32 x f32 product needs <= 48 < 53
    /// significand bits), so the ONLY rounding introduced here is the f64 `+=`
    /// — which is precisely what [`widen_mccormick_constant`] charges for.
    #[inline]
    fn add_term(&mut self, w: f32, plane_const: f32) {
        let term = f64::from(w) * f64::from(plane_const);
        self.accumulated += term;
        let magnitude = term.abs();
        self.magnitude_sum += magnitude;
        self.term_magnitude_sum += magnitude;
        self.adds += 1;
    }

    /// Collapse the fold to its trivially-true constant (`-inf` lower, `+inf`
    /// upper) after a non-finite input box, and poison the magnitude sums so the
    /// certificate can never certify a finite value out of a poisoned fold.
    #[inline]
    fn force_trivial(&mut self) {
        self.accumulated = match self.dir {
            BoundDir::Lower => f64::NEG_INFINITY,
            BoundDir::Upper => f64::INFINITY,
        };
        self.magnitude_sum = f64::INFINITY;
        self.term_magnitude_sum = f64::INFINITY;
        self.adds = self.adds.saturating_add(1);
    }

    /// Publish the certified-outward f64 constant.
    #[inline]
    fn finish(&self) -> f64 {
        widen_mccormick_constant(
            self.accumulated,
            self.magnitude_sum,
            self.term_magnitude_sum,
            self.adds,
            self.dir,
        )
    }
}

impl MulBinaryLayer {
    /// Propagate IBP bounds through element-wise multiplication with broadcasting.
    ///
    /// Delegates to `BoundedTensor::mul` for the core interval arithmetic:
    /// `[a,b]*[c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]`.
    /// This ensures consistent NaN/Inf semantics matching the alpha-beta-CROWN reference
    /// (`auto_LiRPA/operators/bivariate.py:419-421`).
    ///
    /// Supports NumPy-style broadcasting (e.g., [1, 11, 128] * [1, 11, 1]).
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // Fast path: same shape, delegate directly
        if input_a.shape() == input_b.shape() {
            return input_a.mul(input_b);
        }

        // Broadcast to common shape
        let target_shape = broadcast_shapes(input_a.shape(), input_b.shape()).ok_or_else(|| {
            NyError::ShapeMismatch {
                expected: input_a.shape().to_vec(),
                got: input_b.shape().to_vec(),
            }
        })?;

        let a_lower = input_a
            .lower()
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.clone(),
                got: input_a.shape().to_vec(),
            })?
            .to_owned();
        let a_upper = input_a
            .upper()
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.clone(),
                got: input_a.shape().to_vec(),
            })?
            .to_owned();
        let b_lower = input_b
            .lower()
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.clone(),
                got: input_b.shape().to_vec(),
            })?
            .to_owned();
        let b_upper = input_b
            .upper()
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.clone(),
                got: input_b.shape().to_vec(),
            })?
            .to_owned();

        // Construct broadcast BoundedTensors (unchecked: intermediate bounds may be infinite).
        // NaN guard: broadcasting cannot introduce NaN, but if upstream bounds
        // contain NaN (from a prior new_unchecked path), catch it here.
        if a_lower.iter().any(|v| v.is_nan())
            || a_upper.iter().any(|v| v.is_nan())
            || b_lower.iter().any(|v| v.is_nan())
            || b_upper.iter().any(|v| v.is_nan())
        {
            return Err(NyError::NumericalInstability(
                "MulBinaryLayer::propagate_ibp_binary: NaN in broadcast bounds".to_string(),
            ));
        }
        let a_broadcast = BoundedTensor::new_allow_infinite(a_lower, a_upper)?;
        let b_broadcast = BoundedTensor::new_allow_infinite(b_lower, b_upper)?;

        a_broadcast.mul(&b_broadcast)
    }

    /// Compute middle relaxation coefficients for MulBinary (z = x * y).
    ///
    /// Uses auto_LiRPA's fixed-middle formulas with interpolation parameter 0.5.
    /// Returns (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) where:
    ///   z >= alpha_l * x + beta_l * y + ny_l  (lower bound)
    ///   z <= alpha_u * x + beta_u * y + ny_u  (upper bound)
    ///
    /// Reference: auto_LiRPA/operators/bivariate.py:MulHelper.interpolated_relaxation
    #[inline]
    fn compute_middle_coefficients(
        x_l: f32,
        x_u: f32,
        y_l: f32,
        y_u: f32,
    ) -> (f32, f32, f32, f32, f32, f32) {
        // auto_LiRPA middle interpolation formulas with r=0.5:
        // alpha_l = (y_l - y_u) * 0.5 + y_u = y_mid
        // beta_l  = (x_l - x_u) * 0.5 + x_u = x_mid
        // ny_l = (y_u * x_u - y_l * x_l) * 0.5 - y_u * x_u
        // alpha_u = (y_u - y_l) * 0.5 + y_l = y_mid
        // beta_u  = (x_l - x_u) * 0.5 + x_u = x_mid
        // ny_u = (y_l * x_u - y_u * x_l) * 0.5 - y_l * x_u

        Self::compute_interpolated_coefficients(x_l, x_u, y_l, y_u, 0.5, 0.5)
    }

    /// Compute interpolated McCormick coefficients for MulBinary (z = x * y)
    /// with learnable interpolation parameters r_l and r_u in [0, 1].
    ///
    /// Generalizes `compute_middle_coefficients` (which uses r=0.5).
    /// At r_l=0: L2 facet (xu, yu corner). At r_l=1: L1 facet (xl, yl corner).
    /// At r_u=0: U2 facet (xu, yl corner). At r_u=1: U1 facet (xl, yu corner).
    ///
    /// Reference: auto_LiRPA operators/bivariate.py:40-75 (Xu et al., Appendix C,
    /// https://openreview.net/pdf?id=BJxwPJHFwS).
    #[inline]
    fn compute_interpolated_coefficients(
        x_l: f32,
        x_u: f32,
        y_l: f32,
        y_u: f32,
        r_l: f32,
        r_u: f32,
    ) -> (f32, f32, f32, f32, f32, f32) {
        // Delegates to the single canonical CERTIFIED implementation
        // (#mulbinary-mccormick-f32). Constructing these planes in f32 is
        // catastrophically cancelling and yields NON-ENCLOSING planes; see
        // `bilinear::mccormick::interpolated_mccormick` for the derivation and
        // `docs/MULBINARY_MCCORMICK_F32_CANCELLATION_2026-07-28.md` for the
        // measurements. Kept as one implementation so the two cannot drift.
        super::bilinear::mccormick::interpolated_mccormick(x_l, x_u, y_l, y_u, r_l, r_u)
    }

    /// CROWN backward propagation for Mul (z = x * y) using McCormick or Middle relaxation.
    ///
    /// For element-wise multiplication `z[i] = x[i] * y[i]`, McCormick envelope provides
    /// sound linear bounds:
    ///
    /// Lower bounds (take max):
    ///   z ≥ x_l*y + x*y_l - x_l*y_l
    ///   z ≥ x_u*y + x*y_u - x_u*y_u
    ///
    /// Upper bounds (take min):
    ///   z ≤ x_l*y + x*y_u - x_l*y_u
    ///   z ≤ x_u*y + x*y_l - x_u*y_l
    ///
    /// Each bound is linear in (x, y), enabling CROWN backward propagation.
    /// Returns (bounds_for_a, bounds_for_b).
    ///
    /// # Arguments
    /// * `relaxation_mode` - Selects between McCormick (default, selects among 4 facets)
    ///   and Middle (fixed coefficients with interpolation parameter 0.5, matches auto_LiRPA).
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        relaxation_mode: MulBinaryRelaxationMode,
    ) -> Result<(LinearBounds, LinearBounds)> {
        let n = bounds.num_inputs();
        let num_outputs = bounds.num_outputs();
        let n_a = input_a_bounds.len();
        let n_b = input_b_bounds.len();

        // Verify shapes are broadcastable to the output dimension.
        // The forward pass uses NumPy-style broadcasting: [512,5] * [512,1] → [512,5].
        // CROWN backward must reduce coefficients for broadcast inputs by summing
        // along the broadcast dimensions (alpha-beta-CROWN: reduce_broadcast_dims).
        let output_shape = broadcast_shapes(input_a_bounds.shape(), input_b_bounds.shape())
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: input_a_bounds.shape().to_vec(),
                got: input_b_bounds.shape().to_vec(),
            })?;
        let broadcast_n: usize = checked_shape_product(&output_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MulBinary broadcast shape product overflows: {output_shape:?}"
            ))
        })?;
        if broadcast_n != n {
            return Err(NyError::ShapeMismatch {
                expected: vec![n],
                got: vec![broadcast_n],
            });
        }

        // Build broadcast index maps: for each output flat index, which input
        // flat index provides the value? Identity when no broadcast occurs.
        let a_idx_map =
            crate::shape::broadcast_flat_index_map(&output_shape, input_a_bounds.shape());
        let b_idx_map =
            crate::shape::broadcast_flat_index_map(&output_shape, input_b_bounds.shape());

        super::validate_mccormick_inputs(input_a_bounds, input_b_bounds, "MulBinary")?;

        // Flatten bounds for element-wise indexing.
        // Defensive contiguity: reshape/broadcast can produce non-contiguous
        // views where .as_slice() returns None. Copy only when necessary.
        let a_lower_flat = contiguous_flat_slice(input_a_bounds.lower());
        let a_upper_flat = contiguous_flat_slice(input_a_bounds.upper());
        let b_lower_flat = contiguous_flat_slice(input_b_bounds.lower());
        let b_upper_flat = contiguous_flat_slice(input_b_bounds.upper());

        // Coefficient matrices sized to each input's actual dimension.
        // When broadcast, multiple output positions map to the same input element,
        // so coefficients accumulate with += (not =).
        let mut lower_a_a = Array2::<f32>::zeros((num_outputs, n_a));
        let mut lower_a_b = Array2::<f32>::zeros((num_outputs, n_b));
        let mut upper_a_a = Array2::<f32>::zeros((num_outputs, n_a));
        let mut upper_a_b = Array2::<f32>::zeros((num_outputs, n_b));
        // Certified-error accumulators mirroring crown_dense.rs (#matmul-dense-mccormick):
        // each coefficient cell is f32-accumulated with += over the inner `j` axis
        // (broadcast can map every output position to one input cell), so it carries
        // the Higham f32 accumulation error gamma_n * S that MUST reach concretize.
        // S = Sum|w * coeff| is accumulated exactly in f64 (f32*f32 is exact in f64).
        // #vnncomp-aw-soundness.
        let mut lower_s_a = Array2::<f64>::zeros((num_outputs, n_a));
        let mut lower_s_b = Array2::<f64>::zeros((num_outputs, n_b));
        let mut upper_s_a = Array2::<f64>::zeros((num_outputs, n_a));
        let mut upper_s_b = Array2::<f64>::zeros((num_outputs, n_b));
        // Bias accumulation uses f64 to prevent catastrophic cancellation:
        // McCormick constant terms involve products of input bounds (e.g., -lx*ly, -ux*uy).
        // When accumulated across large inner dimensions (4096+ for SwiGLU), f32 loses
        // significant bits from alternating-sign cancellation.
        // Same pattern as crown_dense.rs and common/mod.rs.
        // f64 REDUCES that cancellation; it does not ELIMINATE it, and the residual
        // was published uncharged until #mul-mccormick-const. The fold now runs
        // through `McCormickConstFold`, which carries the Higham magnitude sum
        // alongside the running total and discharges it OUTWARD at `finish()`.
        let mut lower_b_total = Array1::<f64>::zeros(num_outputs);
        let mut upper_b_total = Array1::<f64>::zeros(num_outputs);

        // Process each output dimension
        for out_idx in 0..num_outputs {
            let mut const_lower =
                McCormickConstFold::new(bounds.lower_b()[out_idx], BoundDir::Lower);
            let mut const_upper =
                McCormickConstFold::new(bounds.upper_b()[out_idx], BoundDir::Upper);

            // For each output element position (element-wise: z[j] = x[a_j] * y[b_j]
            // where a_j/b_j follow broadcast mapping).
            for j in 0..n {
                let w_lower = bounds.lower_a()[[out_idx, j]];
                let w_upper = bounds.upper_a()[[out_idx, j]];

                // Broadcast-aware indexing: a_idx/b_idx map the output
                // flat index j to the corresponding input element. For non-broadcast
                // inputs these are identity (a_idx == j); for broadcast inputs
                // (e.g., SE block [512,1] → [512,5]) multiple j values map to
                // the same input element.
                let a_idx = a_idx_map[j];
                let b_idx = b_idx_map[j];
                let lx = a_lower_flat[a_idx];
                let ux = a_upper_flat[a_idx];
                let ly = b_lower_flat[b_idx];
                let uy = b_upper_flat[b_idx];

                match relaxation_mode {
                    MulBinaryRelaxationMode::McCormick => {
                        // Bit-identical McCormick anchors: f32::midpoint rounds differently at overflow/subnormal edges.
                        #[allow(clippy::manual_midpoint)]
                        let x0 = (lx + ux) * 0.5;
                        #[allow(clippy::manual_midpoint)]
                        let y0 = (ly + uy) * 0.5;

                        // Select McCormick plane for lower bound computation
                        let (ax_l, ay_l, c_l) = select_mccormick_plane(
                            lx,
                            ux,
                            ly,
                            uy,
                            x0,
                            y0,
                            w_lower,
                            BoundDir::Lower,
                        );
                        // Accumulate with += for broadcast: multiple output positions
                        // map to the same input element, coefficients sum.
                        lower_a_a[[out_idx, a_idx]] += w_lower * ax_l;
                        lower_a_b[[out_idx, b_idx]] += w_lower * ay_l;
                        lower_s_a[[out_idx, a_idx]] += (w_lower as f64).abs() * (ax_l as f64).abs();
                        lower_s_b[[out_idx, b_idx]] += (w_lower as f64).abs() * (ay_l as f64).abs();
                        const_lower.add_term(w_lower, c_l);

                        // Select McCormick plane for upper bound computation
                        let (ax_u, ay_u, c_u) = select_mccormick_plane(
                            lx,
                            ux,
                            ly,
                            uy,
                            x0,
                            y0,
                            w_upper,
                            BoundDir::Upper,
                        );
                        upper_a_a[[out_idx, a_idx]] += w_upper * ax_u;
                        upper_a_b[[out_idx, b_idx]] += w_upper * ay_u;
                        upper_s_a[[out_idx, a_idx]] += (w_upper as f64).abs() * (ax_u as f64).abs();
                        upper_s_b[[out_idx, b_idx]] += (w_upper as f64).abs() * (ay_u as f64).abs();
                        const_upper.add_term(w_upper, c_u);
                    }
                    MulBinaryRelaxationMode::Middle => {
                        // Middle relaxation: fixed coefficients with interpolation r=0.5
                        // z_lower >= alpha_l * x + beta_l * y + ny_l
                        // z_upper <= alpha_u * x + beta_u * y + ny_u
                        let (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) =
                            Self::compute_middle_coefficients(lx, ux, ly, uy);

                        // For lower bound accumulation, select based on weight sign:
                        // w >= 0: use lower bound of z -> alpha_l, beta_l, ny_l
                        // w < 0:  use upper bound of z -> alpha_u, beta_u, ny_u
                        if w_lower >= 0.0 {
                            lower_a_a[[out_idx, a_idx]] += w_lower * alpha_l;
                            lower_a_b[[out_idx, b_idx]] += w_lower * beta_l;
                            lower_s_a[[out_idx, a_idx]] +=
                                (w_lower as f64).abs() * (alpha_l as f64).abs();
                            lower_s_b[[out_idx, b_idx]] +=
                                (w_lower as f64).abs() * (beta_l as f64).abs();
                            const_lower.add_term(w_lower, ny_l);
                        } else {
                            lower_a_a[[out_idx, a_idx]] += w_lower * alpha_u;
                            lower_a_b[[out_idx, b_idx]] += w_lower * beta_u;
                            lower_s_a[[out_idx, a_idx]] +=
                                (w_lower as f64).abs() * (alpha_u as f64).abs();
                            lower_s_b[[out_idx, b_idx]] +=
                                (w_lower as f64).abs() * (beta_u as f64).abs();
                            const_lower.add_term(w_lower, ny_u);
                        }

                        // For upper bound accumulation, select based on weight sign:
                        // w >= 0: use upper bound of z -> alpha_u, beta_u, ny_u
                        // w < 0:  use lower bound of z -> alpha_l, beta_l, ny_l
                        if w_upper >= 0.0 {
                            upper_a_a[[out_idx, a_idx]] += w_upper * alpha_u;
                            upper_a_b[[out_idx, b_idx]] += w_upper * beta_u;
                            upper_s_a[[out_idx, a_idx]] +=
                                (w_upper as f64).abs() * (alpha_u as f64).abs();
                            upper_s_b[[out_idx, b_idx]] +=
                                (w_upper as f64).abs() * (beta_u as f64).abs();
                            const_upper.add_term(w_upper, ny_u);
                        } else {
                            upper_a_a[[out_idx, a_idx]] += w_upper * alpha_l;
                            upper_a_b[[out_idx, b_idx]] += w_upper * beta_l;
                            upper_s_a[[out_idx, a_idx]] +=
                                (w_upper as f64).abs() * (alpha_l as f64).abs();
                            upper_s_b[[out_idx, b_idx]] +=
                                (w_upper as f64).abs() * (beta_l as f64).abs();
                            const_upper.add_term(w_upper, ny_l);
                        }
                    }
                }
            }

            // Discharge the constant's OWN accumulation + plane-rounding error
            // outward before it leaves f64 (#mul-mccormick-const).
            lower_b_total[out_idx] = const_lower.finish();
            upper_b_total[out_idx] = const_upper.finish();
        }

        // Apply directed rounding on f64→f32 downcast:
        // lower bounds round toward -inf, upper bounds round toward +inf.
        // CORRECTED CLAIM (#mul-mccormick-const): this cast covers the NARROWING
        // and NOTHING else. The line that used to stand here — "this ensures
        // stored f32 bounds always contain the true mathematical value" — was
        // FALSE, and it is the false justification that let the constant half ship
        // uncertified beside an already-certified coefficient half: the cast
        // allows ~2^-24·|result| where the f64 fold's own error is
        // gamma^f64_n·sum|term|, unbounded relative to a cancelling result, and it
        // allows nothing at all for the round-to-nearest plane constant returned
        // by `select_mccormick_plane`. Both are charged upstream by
        // `McCormickConstFold::finish`, so the values reaching this line are
        // ALREADY widened outward.
        let lower_b_f32 = lower_b_total.mapv(|v| next_down_f32(v as f32));
        let upper_b_f32 = upper_b_total.mapv(|v| next_up_f32(v as f32));

        // Certified coefficient error mirroring crown_dense.rs (#matmul-dense-mccormick).
        // Every coefficient cell `[out_idx, idx]` is f32-accumulated with += only inside
        // the inner `for j in 0..n` loop (out_idx is fixed by the outer loop). Each j does
        // at most one += into a given cell, and broadcast index maps can land every one of
        // the n iterations in the SAME cell, so n is a sound UPPER BOUND on the per-cell
        // f32-addition depth for BOTH the a-side and b-side. |stored - exact| <=
        // gamma_n_f32(n) * S (Higham f32 unit roundoff), rounded UP to a sound f32 and
        // carried so concretize penalizes outward.
        let gamma = crate::layers::linear::crown_single_gamma_n_f32(n);
        let lower_a_a_err = lower_s_a.mapv(|s| next_up_f32((gamma * s) as f32));
        let lower_a_b_err = lower_s_b.mapv(|s| next_up_f32((gamma * s) as f32));
        let upper_a_a_err = upper_s_a.mapv(|s| next_up_f32((gamma * s) as f32));
        let upper_a_b_err = upper_s_b.mapv(|s| next_up_f32((gamma * s) as f32));

        // MulBinary returns two affine forms (for x and y) but only one shared
        // McCormick constant term. Keep the full bias on bounds_a and zero
        // bounds_b bias so DAG accumulation counts the constant exactly once.
        let bounds_a = LinearBounds::new_or_conservative_with_err(
            lower_a_a,
            lower_b_f32,
            upper_a_a,
            upper_b_f32,
            lower_a_a_err,
            upper_a_a_err,
        )?;

        let bounds_b = LinearBounds::new_or_conservative_with_err(
            lower_a_b,
            Array1::zeros(num_outputs),
            upper_a_b,
            Array1::zeros(num_outputs),
            lower_a_b_err,
            upper_a_b_err,
        )?;

        Ok((bounds_a, bounds_b))
    }

    /// CROWN backward for MulBinary with alpha-parameterized McCormick interpolation.
    ///
    /// When `alphas` is `Some`, uses per-element interpolation parameters
    /// `r_l = alphas[[0, j]]`, `r_u = alphas[[1, j]]` (shape `[2, n]`) to select
    /// between McCormick envelope facets. When `None`, delegates to fixed McCormick.
    ///
    /// Reference: auto_LiRPA operators/bivariate.py:40-75 (Xu et al., Appendix C).
    pub fn propagate_linear_binary_with_alpha(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        alphas: Option<&Array2<f32>>,
    ) -> Result<(LinearBounds, LinearBounds)> {
        let alphas = match alphas {
            Some(a) => a,
            None => {
                return self.propagate_linear_binary(
                    bounds,
                    input_a_bounds,
                    input_b_bounds,
                    MulBinaryRelaxationMode::McCormick,
                );
            }
        };

        let n = bounds.num_inputs();
        let num_outputs = bounds.num_outputs();
        let n_a = input_a_bounds.len();
        let n_b = input_b_bounds.len();

        if alphas.shape() != [2, n] {
            return Err(NyError::ShapeMismatch {
                expected: vec![2, n],
                got: alphas.shape().to_vec(),
            });
        }

        let output_shape = broadcast_shapes(input_a_bounds.shape(), input_b_bounds.shape())
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: input_a_bounds.shape().to_vec(),
                got: input_b_bounds.shape().to_vec(),
            })?;
        let broadcast_n: usize = checked_shape_product(&output_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MulBinary alpha broadcast shape product overflows: {output_shape:?}"
            ))
        })?;
        if broadcast_n != n {
            return Err(NyError::ShapeMismatch {
                expected: vec![n],
                got: vec![broadcast_n],
            });
        }

        let a_idx_map =
            crate::shape::broadcast_flat_index_map(&output_shape, input_a_bounds.shape());
        let b_idx_map =
            crate::shape::broadcast_flat_index_map(&output_shape, input_b_bounds.shape());

        super::validate_mccormick_inputs(input_a_bounds, input_b_bounds, "MulBinary alpha")?;

        let a_lower_flat = contiguous_flat_slice(input_a_bounds.lower());
        let a_upper_flat = contiguous_flat_slice(input_a_bounds.upper());
        let b_lower_flat = contiguous_flat_slice(input_b_bounds.lower());
        let b_upper_flat = contiguous_flat_slice(input_b_bounds.upper());

        let mut lower_a_a = Array2::<f32>::zeros((num_outputs, n_a));
        let mut lower_a_b = Array2::<f32>::zeros((num_outputs, n_b));
        let mut upper_a_a = Array2::<f32>::zeros((num_outputs, n_a));
        let mut upper_a_b = Array2::<f32>::zeros((num_outputs, n_b));
        let mut lower_b_total = Array1::<f64>::zeros(num_outputs);
        let mut upper_b_total = Array1::<f64>::zeros(num_outputs);

        // Certified-error abssum accumulators (#mul-mccormick-alpha), mirroring
        // crown_dense.rs: each coefficient cell is f32-accumulated over the j axis
        // (the only inner loop), so it carries the Higham f32 accumulation error
        // gamma_depth * S that MUST reach concretize. S = Sum|w * coeff| is summed
        // exactly in f64 (f32*f32 is exact in f64). #vnncomp-aw-soundness.
        let mut lower_s_a = Array2::<f64>::zeros((num_outputs, n_a));
        let mut lower_s_b = Array2::<f64>::zeros((num_outputs, n_b));
        let mut upper_s_a = Array2::<f64>::zeros((num_outputs, n_a));
        let mut upper_s_b = Array2::<f64>::zeros((num_outputs, n_b));

        for out_idx in 0..num_outputs {
            // Certified constant fold (#mul-mccormick-const): the alpha planes come
            // from `interpolated_mccormick`, whose per-plane constant IS already
            // rounded outward by `certify_plane_constant` — but that margin certifies
            // ONE plane, not the f64 SUM of `n` of them, whose Higham residual
            // gamma^f64_n·sum|w·ny| was published uncharged.
            let mut const_lower =
                McCormickConstFold::new(bounds.lower_b()[out_idx], BoundDir::Lower);
            let mut const_upper =
                McCormickConstFold::new(bounds.upper_b()[out_idx], BoundDir::Upper);

            for j in 0..n {
                let w_lower = bounds.lower_a()[[out_idx, j]];
                let w_upper = bounds.upper_a()[[out_idx, j]];

                let a_idx = a_idx_map[j];
                let b_idx = b_idx_map[j];

                let lx = a_lower_flat[a_idx];
                let ux = a_upper_flat[a_idx];
                let ly = b_lower_flat[b_idx];
                let uy = b_upper_flat[b_idx];

                // Non-finite guard: same as select_mccormick_plane.
                if !lx.is_finite() || !ux.is_finite() || !ly.is_finite() || !uy.is_finite() {
                    // Trivial bounds: lower → -inf, upper → +inf.
                    if w_lower != 0.0 {
                        const_lower.force_trivial();
                    }
                    if w_upper != 0.0 {
                        const_upper.force_trivial();
                    }
                    continue;
                }

                let r_l = alphas[[0, j]].clamp(0.0, 1.0);
                let r_u = alphas[[1, j]].clamp(0.0, 1.0);

                let (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) =
                    Self::compute_interpolated_coefficients(lx, ux, ly, uy, r_l, r_u);

                // Weight-sign splitting: w >= 0 uses same-direction relaxation,
                // w < 0 uses opposite-direction relaxation.
                if w_lower >= 0.0 {
                    lower_a_a[[out_idx, a_idx]] += w_lower * alpha_l;
                    lower_a_b[[out_idx, b_idx]] += w_lower * beta_l;
                    lower_s_a[[out_idx, a_idx]] += (w_lower as f64).abs() * (alpha_l as f64).abs();
                    lower_s_b[[out_idx, b_idx]] += (w_lower as f64).abs() * (beta_l as f64).abs();
                    const_lower.add_term(w_lower, ny_l);
                } else {
                    lower_a_a[[out_idx, a_idx]] += w_lower * alpha_u;
                    lower_a_b[[out_idx, b_idx]] += w_lower * beta_u;
                    lower_s_a[[out_idx, a_idx]] += (w_lower as f64).abs() * (alpha_u as f64).abs();
                    lower_s_b[[out_idx, b_idx]] += (w_lower as f64).abs() * (beta_u as f64).abs();
                    const_lower.add_term(w_lower, ny_u);
                }

                if w_upper >= 0.0 {
                    upper_a_a[[out_idx, a_idx]] += w_upper * alpha_u;
                    upper_a_b[[out_idx, b_idx]] += w_upper * beta_u;
                    upper_s_a[[out_idx, a_idx]] += (w_upper as f64).abs() * (alpha_u as f64).abs();
                    upper_s_b[[out_idx, b_idx]] += (w_upper as f64).abs() * (beta_u as f64).abs();
                    const_upper.add_term(w_upper, ny_u);
                } else {
                    upper_a_a[[out_idx, a_idx]] += w_upper * alpha_l;
                    upper_a_b[[out_idx, b_idx]] += w_upper * beta_l;
                    upper_s_a[[out_idx, a_idx]] += (w_upper as f64).abs() * (alpha_l as f64).abs();
                    upper_s_b[[out_idx, b_idx]] += (w_upper as f64).abs() * (beta_l as f64).abs();
                    const_upper.add_term(w_upper, ny_l);
                }
            }

            // Discharge the constant's OWN accumulation error outward before it
            // leaves f64 (#mul-mccormick-const).
            lower_b_total[out_idx] = const_lower.finish();
            upper_b_total[out_idx] = const_upper.finish();
        }

        // Directed f64 -> f32 narrowing ONLY. It is NOT a certificate for
        // the fold above — see `widen_mccormick_constant`; the totals reaching this
        // line are already widened outward (#mul-mccormick-const).
        let lower_b_f32 = lower_b_total.mapv(|v| next_down_f32(v as f32));
        let upper_b_f32 = upper_b_total.mapv(|v| next_up_f32(v as f32));

        // Certified coefficient error (#mul-mccormick-alpha), mirroring crown_dense.rs.
        // The only inner loop is `for j in 0..n`; the destination cell column is
        // a_idx = a_idx_map[j] (A side) / b_idx = b_idx_map[j] (B side), with out_idx
        // fixed by the outer loop. The number of f32 += landing in one A-side cell is
        // the broadcast multiplicity of that input element, which NumPy broadcasting
        // replicates uniformly = n / n_a; n.div_ceil(n_a) is a conservative UPPER bound
        // (== n/n_a when n_a | n, never an under-count). B side: n.div_ceil(n_b).
        // |stored - exact| <= gamma_depth_f32 * S (Higham f32 unit roundoff), with
        // S = Sum|w * coeff| accumulated exactly in f64, rounded UP to a sound f32 and
        // carried so concretize penalizes outward. #vnncomp-aw-soundness.
        let depth_a = n.div_ceil(n_a.max(1)).max(1);
        let depth_b = n.div_ceil(n_b.max(1)).max(1);
        let gamma_a = crate::layers::linear::crown_single_gamma_n_f32(depth_a);
        let gamma_b = crate::layers::linear::crown_single_gamma_n_f32(depth_b);
        let lower_a_a_err = lower_s_a.mapv(|s| next_up_f32((gamma_a * s) as f32));
        let upper_a_a_err = upper_s_a.mapv(|s| next_up_f32((gamma_a * s) as f32));
        let lower_a_b_err = lower_s_b.mapv(|s| next_up_f32((gamma_b * s) as f32));
        let upper_a_b_err = upper_s_b.mapv(|s| next_up_f32((gamma_b * s) as f32));

        let bounds_a = LinearBounds::new_or_conservative_with_err(
            lower_a_a,
            lower_b_f32,
            upper_a_a,
            upper_b_f32,
            lower_a_a_err,
            upper_a_a_err,
        )?;

        let bounds_b = LinearBounds::new_or_conservative_with_err(
            lower_a_b,
            Array1::zeros(num_outputs),
            upper_a_b,
            Array1::zeros(num_outputs),
            lower_a_b_err,
            upper_a_b_err,
        )?;

        Ok((bounds_a, bounds_b))
    }

    /// Batched CROWN backward propagation for MulBinary (z = x * y) with McCormick or Middle relaxation.
    ///
    /// Same as `propagate_linear_binary` but operates on N-D batched bounds,
    /// preserving batch structure [...batch, dim].
    ///
    /// For element-wise `z[i] = x[i] * y[i]`, uses McCormick relaxation:
    ///   Lower bounds (take max):
    ///     z ≥ x_l·y + x·y_l - x_l·y_l
    ///     z ≥ x_u·y + x·y_u - x_u·y_u
    ///   Upper bounds (take min):
    ///     z ≤ x_l·y + x·y_u - x_l·y_u
    ///     z ≤ x_u·y + x·y_l - x_u·y_l
    ///
    /// Returns (bounds_for_a, bounds_for_b).
    ///
    /// # Arguments
    /// * `relaxation_mode` - Selects between McCormick (default, selects among 4 facets)
    ///   and Middle (fixed coefficients with interpolation parameter 0.5, matches auto_LiRPA).
    pub fn propagate_linear_batched_binary(
        &self,
        bounds: &BatchedLinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        relaxation_mode: MulBinaryRelaxationMode,
    ) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
        // Get shapes
        let a_shape = bounds.lower_a.shape();
        let ndim = a_shape.len();

        // For element-wise multiplication, the last dimension is the "n" (features)
        // BatchedLinearBounds shape: [...batch, out_dim, n]
        // where out_dim == n for element-wise operations (identity CROWN)
        if ndim < 2 {
            return Err(NyError::ShapeMismatch {
                expected: vec![2], // at least 2D
                got: a_shape.to_vec(),
            });
        }

        let n = a_shape[ndim - 1]; // last dimension is input features
        let out_dim = a_shape[ndim - 2]; // second-to-last is output dimension

        // Verify input bounds shapes match
        if input_a_bounds.len() != input_b_bounds.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![input_a_bounds.len()],
                got: vec![input_b_bounds.len()],
            });
        }

        super::validate_mccormick_inputs(input_a_bounds, input_b_bounds, "MulBinary batched")?;

        // Flatten bounds for element-wise indexing.
        // Defensive contiguity: reshape/broadcast can produce non-contiguous
        // views where .as_slice() returns None. Copy only when necessary.
        let a_lower_flat = contiguous_flat_slice(input_a_bounds.lower());
        let a_upper_flat = contiguous_flat_slice(input_a_bounds.upper());
        let b_lower_flat = contiguous_flat_slice(input_b_bounds.lower());
        let b_upper_flat = contiguous_flat_slice(input_b_bounds.upper());

        // Calculate batch size (all dimensions except last two)
        let batch_dims = &a_shape[..ndim - 2];
        let batch_size: usize = checked_shape_product(batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "MulBinary batched CROWN: batch dimensions {batch_dims:?} overflow usize",
                ))
            })?
            .max(1);

        // Flatten input arrays to work with them more easily
        let lower_a_flat = contiguous_flat_slice(&bounds.lower_a);
        let upper_a_flat = contiguous_flat_slice(&bounds.upper_a);
        let lower_b_flat = contiguous_flat_slice(&bounds.lower_b);
        let upper_b_flat = contiguous_flat_slice(&bounds.upper_b);

        // Output arrays
        let coeff_len = checked_dim_product(&[batch_size, out_dim, n], "MulBinary batched CROWN")?;
        let bias_len = checked_dim_product(
            &[batch_size, out_dim],
            "MulBinary batched CROWN bias buffers",
        )?;
        let mut lower_a_a = vec![0.0_f32; coeff_len];
        let mut lower_a_b = vec![0.0_f32; coeff_len];
        let mut upper_a_a = vec![0.0_f32; coeff_len];
        let mut upper_a_b = vec![0.0_f32; coeff_len];
        // Certified coefficient-error abssum accumulators (#mul-batched-mccormick).
        // Unlike the broadcast non-batched path (which `+=`-accumulates via index maps),
        // every coeff cell here is written EXACTLY ONCE with `=`: the dest index
        // a_flat_idx = batch_idx*out_dim*n + out_idx*n + j is a bijection of the three
        // enclosing loops (batch_idx, out_idx, j), so no inner reduction axis collides
        // into a cell and the per-cell f32-rounding depth is exactly 1 (a single
        // f32*f32 multiply stored to f32, which still rounds-to-nearest and can land
        // TIGHTER than the true real product). S = |w * coeff| is exact in f64
        // (f32*f32 is exact in f64); err = gamma_1 * S is carried via set_coeff_err so
        // concretize penalizes outward, mirroring crown_dense.rs / crown_batched.rs.
        // #vnncomp-aw-soundness.
        let mut s_lower_a_a = vec![0.0_f64; coeff_len];
        let mut s_lower_a_b = vec![0.0_f64; coeff_len];
        let mut s_upper_a_a = vec![0.0_f64; coeff_len];
        let mut s_upper_a_b = vec![0.0_f64; coeff_len];
        // Bias accumulation in f64 to prevent catastrophic cancellation.
        // Same rationale as non-batched path above and crown_dense.rs —
        // including its correction: f64 REDUCES the cancellation but leaves a
        // Higham residual that was published uncharged until #mul-mccormick-const.
        // Each entry is folded through `McCormickConstFold` and widened OUTWARD.
        let mut lower_b_out = vec![0.0_f64; bias_len];
        let mut upper_b_out = vec![0.0_f64; bias_len];

        // Input bounds may or may not have a batch dimension.
        // If input_len == n, the same bounds are broadcast over all batches.
        // If input_len == batch_size * n, each batch has its own bounds slice.
        let input_len = a_lower_flat.len();
        let input_is_batched = input_len != n;
        if input_is_batched && input_len != batch_size * n {
            return Err(NyError::ShapeMismatch {
                expected: vec![batch_size * n],
                got: vec![input_len],
            });
        }

        // Process each batch position
        for batch_idx in 0..batch_size {
            for out_idx in 0..out_dim {
                let out_flat_idx = batch_idx * out_dim + out_idx;
                let mut const_lower =
                    McCormickConstFold::new(lower_b_flat[out_flat_idx], BoundDir::Lower);
                let mut const_upper =
                    McCormickConstFold::new(upper_b_flat[out_flat_idx], BoundDir::Upper);

                for j in 0..n {
                    let a_flat_idx = batch_idx * out_dim * n + out_idx * n + j;
                    let w_lower = lower_a_flat[a_flat_idx];
                    let w_upper = upper_a_flat[a_flat_idx];

                    // Index into input bounds: broadcast (reuse) or batch-offset
                    let input_j = if input_is_batched {
                        batch_idx * n + j
                    } else {
                        j
                    };

                    let lx = a_lower_flat[input_j];
                    let ux = a_upper_flat[input_j];
                    let ly = b_lower_flat[input_j];
                    let uy = b_upper_flat[input_j];

                    match relaxation_mode {
                        MulBinaryRelaxationMode::McCormick => {
                            // Bit-identical McCormick anchors: f32::midpoint rounds differently at overflow/subnormal edges.
                            #[allow(clippy::manual_midpoint)]
                            let x0 = (lx + ux) * 0.5;
                            #[allow(clippy::manual_midpoint)]
                            let y0 = (ly + uy) * 0.5;

                            // Select McCormick plane for lower bound
                            let (ax_l, ay_l, c_l) = select_mccormick_plane(
                                lx,
                                ux,
                                ly,
                                uy,
                                x0,
                                y0,
                                w_lower,
                                BoundDir::Lower,
                            );
                            lower_a_a[a_flat_idx] = w_lower * ax_l;
                            lower_a_b[a_flat_idx] = w_lower * ay_l;
                            s_lower_a_a[a_flat_idx] = (w_lower as f64).abs() * (ax_l as f64).abs();
                            s_lower_a_b[a_flat_idx] = (w_lower as f64).abs() * (ay_l as f64).abs();
                            const_lower.add_term(w_lower, c_l);

                            // Select McCormick plane for upper bound
                            let (ax_u, ay_u, c_u) = select_mccormick_plane(
                                lx,
                                ux,
                                ly,
                                uy,
                                x0,
                                y0,
                                w_upper,
                                BoundDir::Upper,
                            );
                            upper_a_a[a_flat_idx] = w_upper * ax_u;
                            upper_a_b[a_flat_idx] = w_upper * ay_u;
                            s_upper_a_a[a_flat_idx] = (w_upper as f64).abs() * (ax_u as f64).abs();
                            s_upper_a_b[a_flat_idx] = (w_upper as f64).abs() * (ay_u as f64).abs();
                            const_upper.add_term(w_upper, c_u);
                        }
                        MulBinaryRelaxationMode::Middle => {
                            // Middle relaxation: fixed coefficients with interpolation r=0.5
                            let (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) =
                                Self::compute_middle_coefficients(lx, ux, ly, uy);

                            // For lower bound accumulation, select based on weight sign
                            if w_lower >= 0.0 {
                                lower_a_a[a_flat_idx] = w_lower * alpha_l;
                                lower_a_b[a_flat_idx] = w_lower * beta_l;
                                s_lower_a_a[a_flat_idx] =
                                    (w_lower as f64).abs() * (alpha_l as f64).abs();
                                s_lower_a_b[a_flat_idx] =
                                    (w_lower as f64).abs() * (beta_l as f64).abs();
                                const_lower.add_term(w_lower, ny_l);
                            } else {
                                lower_a_a[a_flat_idx] = w_lower * alpha_u;
                                lower_a_b[a_flat_idx] = w_lower * beta_u;
                                s_lower_a_a[a_flat_idx] =
                                    (w_lower as f64).abs() * (alpha_u as f64).abs();
                                s_lower_a_b[a_flat_idx] =
                                    (w_lower as f64).abs() * (beta_u as f64).abs();
                                const_lower.add_term(w_lower, ny_u);
                            }

                            // For upper bound accumulation, select based on weight sign
                            if w_upper >= 0.0 {
                                upper_a_a[a_flat_idx] = w_upper * alpha_u;
                                upper_a_b[a_flat_idx] = w_upper * beta_u;
                                s_upper_a_a[a_flat_idx] =
                                    (w_upper as f64).abs() * (alpha_u as f64).abs();
                                s_upper_a_b[a_flat_idx] =
                                    (w_upper as f64).abs() * (beta_u as f64).abs();
                                const_upper.add_term(w_upper, ny_u);
                            } else {
                                upper_a_a[a_flat_idx] = w_upper * alpha_l;
                                upper_a_b[a_flat_idx] = w_upper * beta_l;
                                s_upper_a_a[a_flat_idx] =
                                    (w_upper as f64).abs() * (alpha_l as f64).abs();
                                s_upper_a_b[a_flat_idx] =
                                    (w_upper as f64).abs() * (beta_l as f64).abs();
                                const_upper.add_term(w_upper, ny_l);
                            }
                        }
                    }
                }

                // Discharge the constant's OWN accumulation + plane-rounding error
                // outward before it leaves f64 (#mul-mccormick-const).
                lower_b_out[out_flat_idx] = const_lower.finish();
                upper_b_out[out_flat_idx] = const_upper.finish();
            }
        }

        // Apply directed rounding on f64→f32 downcast:
        // lower bounds round toward -inf, upper bounds round toward +inf.
        // This covers the NARROWING ONLY. The f64 fold's own Higham residual and
        // the round-to-nearest plane constant from `select_mccormick_plane` are
        // charged upstream by `McCormickConstFold::finish`, so the values reaching
        // this line are ALREADY widened outward (#mul-mccormick-const). The
        // batched arm shares the scalar arm's defect verbatim — same fold, same
        // planes — so it must share the certificate.
        let lower_b_f32: Vec<f32> = lower_b_out
            .iter()
            .map(|&v| next_down_f32(v as f32))
            .collect();
        let upper_b_f32: Vec<f32> = upper_b_out.iter().map(|&v| next_up_f32(v as f32)).collect();

        // Certified coefficient error (#mul-batched-mccormick). Per-cell f32-rounding
        // depth is exactly 1: each coeff cell is written with a single `=` multiply and
        // a_flat_idx = batch_idx*out_dim*n + out_idx*n + j bijects the three enclosing
        // loops (no reduction axis holds a_flat_idx fixed), so gamma_1 upper-bounds the
        // round-to-nearest error of the stored f32 product. err = gamma_1 * S rounded UP,
        // carried via set_coeff_err so concretize penalizes outward (mirrors
        // crown_dense.rs / crown_batched.rs). #vnncomp-aw-soundness.
        let gamma1 = crate::layers::linear::crown_single_gamma_n_f32(1);
        let mk_err = |s: &[f64]| -> Result<ArrayD<f32>> {
            let v: Vec<f32> = s
                .iter()
                .map(|&sv| next_up_f32((gamma1 * sv) as f32))
                .collect();
            let len = v.len();
            ArrayD::from_shape_vec(IxDyn(a_shape), v).map_err(|_| NyError::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: vec![len],
            })
        };
        let lower_a_a_err = mk_err(&s_lower_a_a)?;
        let upper_a_a_err = mk_err(&s_upper_a_a)?;
        let lower_a_b_err = mk_err(&s_lower_a_b)?;
        let upper_a_b_err = mk_err(&s_upper_a_b)?;

        // Reshape outputs to batched form
        let lower_b_bias_shape = IxDyn(bounds.lower_b.shape());
        let upper_b_bias_shape = IxDyn(bounds.upper_b.shape());
        // Phase 4 audit: per-layer McCormick output — catches NaN from upstream via McCormick.
        // Capture vec lengths before from_shape_vec moves them.
        let lower_a_a_len = lower_a_a.len();
        let lower_b_f32_len = lower_b_f32.len();
        let upper_a_a_len = upper_a_a.len();
        let upper_b_f32_len = upper_b_f32.len();
        let lower_a_b_len = lower_a_b.len();
        let upper_a_b_len = upper_a_b.len();
        let mut bounds_a = BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(a_shape), lower_a_a).map_err(|_| {
                NyError::ShapeMismatch {
                    expected: a_shape.to_vec(),
                    got: vec![lower_a_a_len],
                }
            })?,
            ArrayD::from_shape_vec(lower_b_bias_shape.clone(), lower_b_f32).map_err(|_| {
                NyError::ShapeMismatch {
                    expected: bounds.lower_b.shape().to_vec(),
                    got: vec![lower_b_f32_len],
                }
            })?,
            ArrayD::from_shape_vec(IxDyn(a_shape), upper_a_a).map_err(|_| {
                NyError::ShapeMismatch {
                    expected: a_shape.to_vec(),
                    got: vec![upper_a_a_len],
                }
            })?,
            ArrayD::from_shape_vec(upper_b_bias_shape.clone(), upper_b_f32).map_err(|_| {
                NyError::ShapeMismatch {
                    expected: bounds.upper_b.shape().to_vec(),
                    got: vec![upper_b_f32_len],
                }
            })?,
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )?;
        bounds_a.set_coeff_err(lower_a_a_err, upper_a_a_err);

        let mut bounds_b = BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(a_shape), lower_a_b).map_err(|_| {
                NyError::ShapeMismatch {
                    expected: a_shape.to_vec(),
                    got: vec![lower_a_b_len],
                }
            })?,
            ArrayD::zeros(lower_b_bias_shape),
            ArrayD::from_shape_vec(IxDyn(a_shape), upper_a_b).map_err(|_| {
                NyError::ShapeMismatch {
                    expected: a_shape.to_vec(),
                    got: vec![upper_a_b_len],
                }
            })?,
            ArrayD::zeros(upper_b_bias_shape),
            bounds.input_shape.clone(),
            bounds.output_shape.clone(),
        )?;
        bounds_b.set_coeff_err(lower_a_b_err, upper_a_b_err);

        Ok((bounds_a, bounds_b))
    }
}

#[cfg(test)]
mod mccormick_constant_certificate_tests {
    use super::*;

    /// Inner width of the counterexample: two `2^60`-scale taps plus 4096 unit
    /// taps that the f64 running sum swallows whole.
    const K: usize = 4098;
    /// `2^30`; `BIG * BIG == 2^60` is exact in f32 and in f64.
    const BIG: f32 = 1_073_741_824.0;

    /// Degenerate (`lower == upper`) taps for both operands.
    ///
    /// `x = [2^30, 1, ..., 1, 2^30]`, `y = [-2^30, 1, ..., 1, 2^30]`, so the
    /// McCormick constants `-x_j*y_j` are `+2^60`, then 4096 x `-1`, then `-2^60`.
    fn degenerate_taps() -> (Vec<f32>, Vec<f32>) {
        let mut xs = vec![1.0_f32; K];
        let mut ys = vec![1.0_f32; K];
        xs[0] = BIG;
        ys[0] = -BIG;
        xs[K - 1] = BIG;
        ys[K - 1] = BIG;
        (xs, ys)
    }

    /// `sum_j x_j*y_j`, computed exactly in i128.
    fn exact_objective(xs: &[f32], ys: &[f32]) -> i128 {
        (0..xs.len())
            .map(|l| (f64::from(xs[l]) * f64::from(ys[l])) as i128)
            .sum()
    }

    /// Exactly what the production loop folds: `w = 1`, `c = fl32(-x*y)`, product
    /// exact in f64, plain round-to-nearest `+=`. This is the PRE-FIX published
    /// value (up to the final directed f32 cast).
    fn naive_f64_fold(xs: &[f32], ys: &[f32]) -> f64 {
        let mut folded = 0.0_f64;
        for l in 0..xs.len() {
            folded += f64::from(-xs[l] * ys[l]);
        }
        folded
    }

    fn degenerate_input(values: &[f32]) -> BoundedTensor {
        let array = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec())
            .expect("degenerate input shape");
        BoundedTensor::new(array.clone(), array).expect("degenerate BoundedTensor")
    }

    /// COUNTEREXAMPLE — the accumulated McCormick envelope CONSTANT must be
    /// published with its own certified error term (#mul-mccormick-const).
    ///
    /// This is a WIRING test: it drives the public CROWN-backward entry point and
    /// reads the PUBLISHED relation, so it fails if the certificate exists but is
    /// not plumbed into `lower_b_total`.
    ///
    /// Both operands are DEGENERATE boxes (`lower == upper`), so all four
    /// McCormick planes coincide, plane selection is irrelevant, and the
    /// relaxation is exact at the single feasible point `(X, Y)`. Nothing is
    /// broadcast, so each coefficient cell receives exactly ONE exact term and the
    /// containment condition reduces to a statement about the constant alone:
    ///
    /// ```text
    /// crown_lower(x, y) = sum_j (w*Y_j)*x_j + sum_j (w*X_j)*y_j + const
    ///                   = 2*sum_j X_j*Y_j + const               (w = 1)
    /// objective         =   sum_j X_j*Y_j  =  T
    /// sound  <=>  const <= -T
    /// ```
    ///
    /// The taps make the f64 fold catastrophically wrong: `+2^60`, then 4096 x
    /// `-1`, then `-2^60`. `ulp(2^60) = 256`, so every `-1` rounds away and the
    /// accumulator returns EXACTLY `0.0` while the true constant is `-4096`. The
    /// pre-fix code published that `0.0` protected by nothing but the f64 -> f32
    /// cast — `next_down_f32(0.0) = -1.4e-45` — i.e. a "lower" plane sitting
    /// `4096` ABOVE the bilinear surface at a FEASIBLE point, which is a false
    /// `unsat` available directly from the relation. This assertion FAILS on the
    /// pre-fix code.
    #[ntest::timeout(60000)]
    #[test]
    fn scalar_envelope_constant_carries_its_higham_charge() {
        let (xs, ys) = degenerate_taps();
        let objective = exact_objective(&xs, &ys);
        assert_eq!(objective, 4096, "true objective sum_j X_j*Y_j");
        let folded = naive_f64_fold(&xs, &ys);
        assert_eq!(
            folded, 0.0,
            "premise: the plain f64 constant fold must cancel to 0, otherwise this \
             test no longer exercises the defect"
        );
        assert!(
            next_down_f32(folded as f32) > -(objective as f64) as f32,
            "premise: the PRE-FIX publication must be unsound, otherwise this test \
             cannot fail on the pre-fix code"
        );

        let input_a = degenerate_input(&xs);
        let input_b = degenerate_input(&ys);
        let bounds = LinearBounds::new(
            Array2::from_elem((1, K), 1.0_f32),
            Array1::zeros(1),
            Array2::from_elem((1, K), 1.0_f32),
            Array1::zeros(1),
        )
        .expect("incoming all-ones relation");

        let layer = MulBinaryLayer;
        let (bounds_a, bounds_b) = layer
            .propagate_linear_binary(
                &bounds,
                &input_a,
                &input_b,
                MulBinaryRelaxationMode::McCormick,
            )
            .expect("scalar McCormick CROWN backward");

        // Premise of the reduction above: every coefficient cell holds exactly one
        // exact term, so the whole error budget lives in the constant.
        for &l in &[0_usize, 1, K - 1] {
            assert_eq!(
                bounds_a.lower_a[[0, l]],
                ys[l],
                "A-side coefficient cell {l} must be the single exact term w*ly"
            );
            assert_eq!(
                bounds_b.lower_a[[0, l]],
                xs[l],
                "B-side coefficient cell {l} must be the single exact term w*lx"
            );
        }

        let published = f64::from(bounds_a.lower_b[0]) + f64::from(bounds_b.lower_b[0]);
        assert!(
            published <= -(objective as f64),
            "published McCormick lower constant {published} must be <= {} — otherwise \
             the lower relaxation sits ABOVE the bilinear surface at a feasible point",
            -(objective as f64)
        );
    }

    /// The batched twin shares the scalar arm's fold verbatim, so it must share
    /// the certificate. Same counterexample through
    /// [`MulBinaryLayer::propagate_linear_batched_binary`]; also FAILS pre-fix.
    #[ntest::timeout(60000)]
    #[test]
    fn batched_envelope_constant_carries_its_higham_charge() {
        let (xs, ys) = degenerate_taps();
        let objective = exact_objective(&xs, &ys);
        assert_eq!(objective, 4096, "true objective sum_j X_j*Y_j");
        assert_eq!(
            naive_f64_fold(&xs, &ys),
            0.0,
            "premise: the plain f64 constant fold must cancel to 0"
        );

        let input_a = degenerate_input(&xs);
        let input_b = degenerate_input(&ys);
        let ones = ArrayD::from_elem(IxDyn(&[1, K]), 1.0_f32);
        let zeros = ArrayD::from_elem(IxDyn(&[1]), 0.0_f32);
        let bounds =
            BatchedLinearBounds::new(ones.clone(), zeros.clone(), ones, zeros, vec![K], vec![1])
                .expect("incoming batched all-ones relation");

        let layer = MulBinaryLayer;
        let (bounds_a, bounds_b) = layer
            .propagate_linear_batched_binary(
                &bounds,
                &input_a,
                &input_b,
                MulBinaryRelaxationMode::McCormick,
            )
            .expect("batched McCormick CROWN backward");

        let coeff_a = bounds_a
            .lower_a
            .as_slice()
            .expect("batched A-side coefficients are contiguous");
        let coeff_b = bounds_b
            .lower_a
            .as_slice()
            .expect("batched B-side coefficients are contiguous");
        for &l in &[0_usize, 1, K - 1] {
            assert_eq!(coeff_a[l], ys[l], "A-side cell {l} must be the exact w*ly");
            assert_eq!(coeff_b[l], xs[l], "B-side cell {l} must be the exact w*lx");
        }

        let bias_a = *bounds_a
            .lower_b
            .iter()
            .next()
            .expect("batched A-side bias present");
        let bias_b = *bounds_b
            .lower_b
            .iter()
            .next()
            .expect("batched B-side bias present");
        let published = f64::from(bias_a) + f64::from(bias_b);
        assert!(
            published <= -(objective as f64),
            "published batched McCormick lower constant {published} must be <= {}",
            -(objective as f64)
        );
    }

    /// Direction and fail-closed contract of the certificate itself.
    ///
    /// A term-free fold must be published BIT-IDENTICALLY (no f64 addition ran and
    /// no plane was consumed, so there is nothing to charge); any fold with at
    /// least one term may only move OUTWARD; a NaN anywhere must fail CLOSED to
    /// the trivially-true constant rather than to a finite number.
    #[test]
    fn constant_certificate_is_outward_only_and_fails_closed() {
        for &seed in &[0.0_f64, -1.5, 7.25, 1.0e18, f64::MIN_POSITIVE] {
            assert_eq!(
                widen_mccormick_constant(seed, seed.abs(), 0.0, 0, BoundDir::Lower).to_bits(),
                seed.to_bits(),
                "adds == 0 must be the identity (Lower)"
            );
            assert_eq!(
                widen_mccormick_constant(seed, seed.abs(), 0.0, 0, BoundDir::Upper).to_bits(),
                seed.to_bits(),
                "adds == 0 must be the identity (Upper)"
            );
        }

        for &(accumulated, s) in &[(0.0_f64, 1.0e18_f64), (-3.5, 12.0), (1.0e9, 1.0e9)] {
            let lower = widen_mccormick_constant(accumulated, s, s, 4, BoundDir::Lower);
            let upper = widen_mccormick_constant(accumulated, s, s, 4, BoundDir::Upper);
            assert!(
                lower < accumulated,
                "Lower constant must move DOWN: {lower} !< {accumulated}"
            );
            assert!(
                upper > accumulated,
                "Upper constant must move UP: {upper} !> {accumulated}"
            );
        }

        assert_eq!(
            widen_mccormick_constant(f64::NAN, 1.0, 1.0, 1, BoundDir::Lower),
            f64::NEG_INFINITY
        );
        assert_eq!(
            widen_mccormick_constant(f64::NAN, 1.0, 1.0, 1, BoundDir::Upper),
            f64::INFINITY
        );
        // A poisoned magnitude sum (the `force_trivial` state) can never certify a
        // finite constant.
        let mut poisoned = McCormickConstFold::new(0.0, BoundDir::Lower);
        poisoned.force_trivial();
        assert_eq!(poisoned.finish(), f64::NEG_INFINITY);
        let mut poisoned_up = McCormickConstFold::new(0.0, BoundDir::Upper);
        poisoned_up.force_trivial();
        assert_eq!(poisoned_up.finish(), f64::INFINITY);
    }
}

#[cfg(test)]
mod tests;
