// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward propagation for MatMul using McCormick envelopes.

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::checked_shape_product;
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::debug;

use super::super::validate_mccormick_inputs;
use super::shape::{decode_batch_index_into, parse_matmul_dims};
use super::{select_mccormick_plane, BatchedLinearBounds, BoundDir, MatMulLayer, NyError, Result};

/// Widen one accumulated McCormick ENVELOPE CONSTANT outward by a certified bound
/// on its own floating-point error (#matmul-batched-mccormick-const).
///
/// BATCHED TWIN of `crown_dense::widen_mccormick_constant`
/// (#matmul-dense-mccormick-const), kept as a private copy for the same reason
/// that one is a private copy of `binary_ops::mul`'s: the original is private to
/// its module and the certificate is small enough that duplicating it beats
/// widening anything's visibility. All FOUR copies (`binary_ops::mul`,
/// `bilinear::linear`, `matmul::crown_dense` and this one) MUST stay in step — a
/// divergence would mean the same fold is charged differently depending only on
/// which entry point the caller happened to take.
///
/// # The defect this closes
///
/// `propagate_linear_batched_binary` folded the envelope constant as
/// `new_lower_b[[bb, d]] += w * c` in PLAIN round-to-nearest f64 and published it
/// with no error term at all — only the final f64 -> f32 directed cast in the
/// halving step. That cast covers the narrowing and nothing else, so two distinct
/// errors reached the verdict path uncharged:
///
/// 1. **f64 accumulation.** Recursive summation of `adds + 1` summands satisfies
///    `|fl(S) - S| <= gamma^f64_{adds} * sum|summand|` (Higham, *Accuracy and
///    Stability*, Thm 3.1). The charge scales with the MAGNITUDE sum, not with
///    `|S|`, so it does NOT vanish under cancellation — which is exactly when a
///    result-scaled 1-ULP cast allowance under-counts it. This arm's per-cell fold
///    depth is `batch_size * m * n * k`, i.e. 1e4-1e6 on attention-shaped nodes.
///    Worked case: terms `+2^60`, 4096 x `-1`, `-2^60` fold to EXACTLY `0.0` in f64
///    (`ulp(2^60) = 256` swallows every `-1`) where the truth is `-4096`; one ULP of
///    `0.0` is `1.4e-45`, so the directed cast allows `1.4e-45` against a real error
///    of `4096`. The magnitude-sum charge covers it; a result-scaled ULP cannot.
/// 2. **The plane constant itself.** The `c` reaching this fold is
///    `fl32(-lx*ly)` (or one of the three sibling corner products from
///    [`select_mccormick_plane`]) — ONE round-to-nearest f32 multiply away from the
///    exact McCormick constant — and `w` is `fl32(lower_a * scale)`, one more.
///    Either rounding can move the plane the UNSOUND way: a lower plane whose
///    constant rounded UP no longer under-estimates the bilinear surface, so the
///    relaxation stops containing `x*y`. Both are relative to `|w*c|`, hence a
///    charge at f32 unit roundoff against the TERM magnitude sum.
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
/// The `1/(1 - gamma)` inflation mirrors `conv2d::bound`: `magnitude_sum` is itself
/// an f64 round-to-nearest sum of non-negative values and can fall SHORT of the true
/// magnitude sum by its own accumulation deficit, so charging against the raw figure
/// could under-bound the very error it is meant to cover. The final directed f64 step
/// absorbs the rounding of the widening subtract/add.
///
/// # Ordering
///
/// This runs on the f64 TOTAL, BEFORE the `* 0.5` bias split and before the f32
/// cast. `0.5` is a power of two, so the halving is exact and direction-preserving;
/// widening after the cast would charge against an already-narrowed value.
///
/// # Direction
///
/// `err >= 0` always (`gamma >= 0`, both magnitude sums are sums of `abs` values,
/// `inflate >= 1`), so a `Lower` constant only ever moves DOWN and an `Upper`
/// constant only ever moves UP. `adds == 0` returns the input bit-identically: no
/// f64 addition ran and no McCormick plane was consumed, so there is genuinely
/// nothing to charge — that is what keeps the term-free halving contract
/// (`rounding_tests::batched_crown_bias_halving_uses_directed_rounding_2173`)
/// byte-for-byte intact. A NaN anywhere fails CLOSED to the infinite constant.
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
    // 2 f32 roundings per term (`* scale`, and the corner product inside
    // `select_mccormick_plane`) plus one step of slack.
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

impl MatMulLayer {
    /// Batched CROWN backward propagation for MatMul (C = A @ B or A @ B^T).
    ///
    /// Uses McCormick envelope relaxation for the bilinear terms.
    /// Supports N-D batched inputs: A has shape [..., M, K], B has shape [..., K, N] or [..., N, K].
    ///
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_batched_binary(
        &self,
        bounds: &BatchedLinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
        debug!("MatMul batched CROWN backward propagation");

        let dims = parse_matmul_dims(
            self.transpose_b,
            input_a_bounds.shape(),
            input_b_bounds.shape(),
        )?;

        validate_mccormick_inputs(input_a_bounds, input_b_bounds, "MatMul batched")?;

        let batch_size = dims.batch_size()?;
        let c_size = dims.c_size_per_batch()?;
        let a_size = dims.a_size_per_batch()?;
        let b_size = dims.b_size_per_batch;

        // Get the bounds shape
        let bounds_a_shape = bounds.lower_a.shape();
        if bounds_a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = bounds_a_shape[bounds_a_shape.len() - 2];
        let mid_dim = bounds_a_shape[bounds_a_shape.len() - 1];

        if mid_dim != c_size {
            return Err(NyError::UnsupportedOp(format!(
                "MatMul batched CROWN expects flattened output dim m*n = {} (got in_dim = {}); consider reshaping MatMul outputs or fall back to IBP",
                c_size, mid_dim
            )));
        }

        let bounds_batch_dims = &bounds_a_shape[..bounds_a_shape.len() - 2];
        let total_bounds_batch: usize = checked_shape_product(bounds_batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "MatMul batched CROWN: bounds batch dimensions {bounds_batch_dims:?} overflow usize",
                ))
            })?
            .max(1);

        // Output shapes
        let mut out_a_shape: Vec<usize> = bounds_batch_dims.to_vec();
        out_a_shape.push(out_dim);
        out_a_shape.push(a_size);

        let mut out_b_shape: Vec<usize> = bounds_batch_dims.to_vec();
        out_b_shape.push(out_dim);
        out_b_shape.push(b_size);

        let mut out_bias_shape: Vec<usize> = bounds_batch_dims.to_vec();
        out_bias_shape.push(out_dim);

        let scale = self.scale.unwrap_or(1.0);

        // Reshape bounds to [total_batch, out_dim, c_size]
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_bounds_batch, out_dim, c_size))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_bounds_batch, out_dim, c_size))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;
        let lower_b_2d = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_bounds_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
        let upper_b_2d = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_bounds_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

        // Allocate output coefficient matrices
        let mut new_lower_a_a = Array2::zeros((total_bounds_batch * out_dim, a_size));
        let mut new_upper_a_a = Array2::zeros((total_bounds_batch * out_dim, a_size));
        let mut new_lower_a_b = Array2::zeros((total_bounds_batch * out_dim, b_size));
        let mut new_upper_a_b = Array2::zeros((total_bounds_batch * out_dim, b_size));
        // Certified-error abssum accumulators (#matmul-batched-mccormick): each coeff
        // cell is summed in round-to-nearest f32 over the (batch_idx, j) pairs for the
        // A side and (batch_idx, i) for the B side — the cell index a_flat=i*k+l /
        // b_flat omits batch_idx, so batch_size*dims.n (A) / batch_size*dims.m (B)
        // terms collide into one cell. Carry the Higham f32 accumulation error
        // gamma_depth * S (S = Sum|w*a| in exact f64) so it reaches concretize.
        // #vnncomp-aw-soundness.
        let mut s_lower_a_a = Array2::<f64>::zeros((total_bounds_batch * out_dim, a_size));
        let mut s_upper_a_a = Array2::<f64>::zeros((total_bounds_batch * out_dim, a_size));
        let mut s_lower_a_b = Array2::<f64>::zeros((total_bounds_batch * out_dim, b_size));
        let mut s_upper_a_b = Array2::<f64>::zeros((total_bounds_batch * out_dim, b_size));
        let mut new_lower_b = Array2::<f64>::zeros((total_bounds_batch, out_dim));
        let mut new_upper_b = Array2::<f64>::zeros((total_bounds_batch, out_dim));
        let batch_index_len = dims.batch_dims.len();

        // Process each batch position and output dimension
        for bb in 0..total_bounds_batch {
            let mut batch_indices = Vec::with_capacity(batch_index_len);
            let mut a_idx = Vec::with_capacity(batch_index_len + 2);
            let mut b_idx = Vec::with_capacity(batch_index_len + 2);

            for d in 0..out_dim {
                let row_idx = bb * out_dim + d;

                // PER-CELL envelope-constant fold state (#matmul-batched-mccormick-const).
                // Every `[[bb, d]]` cell is its OWN independent fold: the
                // (batch_idx, i, j, l) loops below run to completion for this `d`
                // before the next one, and these locals are re-seeded here, so the
                // certificate charges exactly this cell's own summands — never one
                // cell's magnitude against another's (which would be simultaneously
                // loose here and wrong there), and never a shared count that
                // UNDER-charges a cell whose incoming row is denser than average.
                //
                // `const_*` is the running f64 sum, seeded with the incoming bias;
                // `const_*_s` is the Higham magnitude sum Sum|summand| INCLUDING that
                // bias (the f64 accumulation charge is against every summand);
                // `const_*_terms_s` EXCLUDES it (the per-term f32 plane/scale rounding
                // charge applies only to the McCormick terms); `const_*_adds` counts
                // the f64 additions actually performed, so a term-free cell charges
                // exactly zero. See `widen_mccormick_constant`.
                let mut const_lower = lower_b_2d[[bb, d]] as f64;
                let mut const_upper = upper_b_2d[[bb, d]] as f64;
                let mut const_lower_s = const_lower.abs();
                let mut const_upper_s = const_upper.abs();
                let mut const_lower_terms_s = 0.0_f64;
                let mut const_upper_terms_s = 0.0_f64;
                let mut const_lower_adds = 0_usize;
                let mut const_upper_adds = 0_usize;

                for batch_idx in 0..batch_size {
                    decode_batch_index_into(batch_idx, &dims.batch_dims, &mut batch_indices)?;
                    a_idx.clear();
                    a_idx.extend_from_slice(&batch_indices);
                    a_idx.resize(batch_index_len + 2, 0);
                    b_idx.clear();
                    b_idx.extend_from_slice(&batch_indices);
                    b_idx.resize(batch_index_len + 2, 0);

                    for i in 0..dims.m {
                        a_idx[batch_index_len] = i;
                        for j in 0..dims.n {
                            let c_flat = i * dims.n + j;

                            let w_lower = lower_a_3d[[bb, d, c_flat]] * scale;
                            let w_upper = upper_a_3d[[bb, d, c_flat]] * scale;

                            if w_lower == 0.0 && w_upper == 0.0 {
                                continue;
                            }

                            if self.transpose_b {
                                b_idx[batch_index_len] = j;
                            } else {
                                b_idx[batch_index_len + 1] = j;
                            }

                            for l in 0..dims.k {
                                // Get A[batch..., i, l]
                                a_idx[batch_index_len + 1] = l;
                                let lx = input_a_bounds.lower()[a_idx.as_slice()];
                                let ux = input_a_bounds.upper()[a_idx.as_slice()];
                                // Bit-identical McCormick anchor: f32::midpoint rounds differently at overflow/subnormal edges.
                                #[allow(clippy::manual_midpoint)]
                                let x0 = (lx + ux) * 0.5;

                                let a_flat = i * dims.k + l;

                                // Get B[batch..., l, j] or B[batch..., j, l] if transposed
                                let b_flat = if self.transpose_b {
                                    b_idx[batch_index_len + 1] = l;
                                    j * dims.k + l
                                } else {
                                    b_idx[batch_index_len] = l;
                                    l * dims.n + j
                                };

                                let ly = input_b_bounds.lower()[b_idx.as_slice()];
                                let uy = input_b_bounds.upper()[b_idx.as_slice()];
                                // Bit-identical McCormick anchor: f32::midpoint rounds differently at overflow/subnormal edges.
                                #[allow(clippy::manual_midpoint)]
                                let y0 = (ly + uy) * 0.5;

                                if w_lower != 0.0 {
                                    let (ax, ay, c) = select_mccormick_plane(
                                        lx,
                                        ux,
                                        ly,
                                        uy,
                                        x0,
                                        y0,
                                        w_lower,
                                        BoundDir::Lower,
                                    );
                                    new_lower_a_a[[row_idx, a_flat]] += w_lower * ax;
                                    new_lower_a_b[[row_idx, b_flat]] += w_lower * ay;
                                    s_lower_a_a[[row_idx, a_flat]] +=
                                        (w_lower as f64).abs() * (ax as f64).abs();
                                    s_lower_a_b[[row_idx, b_flat]] +=
                                        (w_lower as f64).abs() * (ay as f64).abs();
                                    // f32 x f32 is EXACT in f64 (48 < 53 significand
                                    // bits), so only this ADDITION rounds — that is
                                    // the error `widen_mccormick_constant` charges for.
                                    let term_lower = w_lower as f64 * c as f64;
                                    const_lower += term_lower;
                                    const_lower_s += term_lower.abs();
                                    const_lower_terms_s += term_lower.abs();
                                    const_lower_adds += 1;
                                }

                                if w_upper != 0.0 {
                                    let (ax, ay, c) = select_mccormick_plane(
                                        lx,
                                        ux,
                                        ly,
                                        uy,
                                        x0,
                                        y0,
                                        w_upper,
                                        BoundDir::Upper,
                                    );
                                    new_upper_a_a[[row_idx, a_flat]] += w_upper * ax;
                                    new_upper_a_b[[row_idx, b_flat]] += w_upper * ay;
                                    s_upper_a_a[[row_idx, a_flat]] +=
                                        (w_upper as f64).abs() * (ax as f64).abs();
                                    s_upper_a_b[[row_idx, b_flat]] +=
                                        (w_upper as f64).abs() * (ay as f64).abs();
                                    let term_upper = w_upper as f64 * c as f64;
                                    const_upper += term_upper;
                                    const_upper_s += term_upper.abs();
                                    const_upper_terms_s += term_upper.abs();
                                    const_upper_adds += 1;
                                }
                            }
                        }
                    }
                }

                // Publish this cell's f64 TOTAL already widened outward for its own
                // accumulation and plane-rounding error, BEFORE the `* 0.5` split and
                // the directed f32 cast below (#matmul-batched-mccormick-const).
                new_lower_b[[bb, d]] = widen_mccormick_constant(
                    const_lower,
                    const_lower_s,
                    const_lower_terms_s,
                    const_lower_adds,
                    BoundDir::Lower,
                );
                new_upper_b[[bb, d]] = widen_mccormick_constant(
                    const_upper,
                    const_upper_s,
                    const_upper_terms_s,
                    const_upper_adds,
                    BoundDir::Upper,
                );
            }
        }

        // Reshape back to output shape
        let (new_lower_a_a_vec, _) = new_lower_a_a.into_raw_vec_and_offset();
        let (new_upper_a_a_vec, _) = new_upper_a_a.into_raw_vec_and_offset();
        let (new_lower_a_b_vec, _) = new_lower_a_b.into_raw_vec_and_offset();
        let (new_upper_a_b_vec, _) = new_upper_a_b.into_raw_vec_and_offset();
        // Split constant terms across both inputs. Halve in f64, then apply directed
        // rounding on the final f32 cast. The f64 totals reaching this point
        // have ALREADY been widened outward for their own accumulation and
        // plane-rounding error by `widen_mccormick_constant`; the directed cast below
        // only covers the f64 -> f32 narrowing, which is NOT sufficient on its own
        // (#matmul-batched-mccormick-const). `0.5` is a power of two, so the halving
        // is exact and cannot walk either half back inward.
        let new_lower_b_f32 = new_lower_b.mapv(|v| next_down_f32((v * 0.5) as f32));
        let new_upper_b_f32 = new_upper_b.mapv(|v| next_up_f32((v * 0.5) as f32));
        let (new_lower_b_vec, _) = new_lower_b_f32.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b_f32.into_raw_vec_and_offset();

        let new_lower_a_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a_a".to_string()))?;
        let new_upper_a_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a_a".to_string()))?;
        let new_lower_a_b = ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_a_b_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a_b".to_string()))?;
        let new_upper_a_b = ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_a_b_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a_b".to_string()))?;
        let lower_b_half = ArrayD::from_shape_vec(IxDyn(&out_bias_shape), new_lower_b_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?;
        let upper_b_half = ArrayD::from_shape_vec(IxDyn(&out_bias_shape), new_upper_b_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?;

        // Update input shapes to flattened representations (preserve batch dims).
        let a_batch = &dims.batch_dims;
        let b_batch = &dims.batch_dims;
        let mut new_input_shape_a = a_batch.clone();
        new_input_shape_a.push(a_size);
        let mut new_input_shape_b = b_batch.clone();
        new_input_shape_b.push(b_size);

        // Certified coefficient error (#matmul-batched-mccormick). Conservative depth:
        // a cell collides over at most batch_size*dims.n (A) / batch_size*dims.m (B)
        // f32 += operations; gamma_n_f32 of that upper-bounds the true Higham factor
        // (sound whether or not batch_idx actually collides). err = gamma*S rounded UP,
        // carried via set_coeff_err so concretize penalizes outward. #vnncomp-aw-soundness.
        //
        // CORRECTED CHARGE (#matmul-batched-mccormick-const), taken from the dense
        // twin `crown_dense.rs`, whose charge was corrected this session while this
        // one was left a step short: `gamma_n_f32(count)` covers only the
        // `count - 1` ADDITIONS. Each summand also carries TWO further f32 roundings
        // BEFORE it is added -- `w = fl32(lower_a * scale)` and `fl32(w * ax)` -- so
        // the honest Higham factor is `gamma_{count+1}`; `count + 2` buys one more
        // step, which also absorbs the (far smaller) f64 accumulation deficit of S
        // itself. This matters most at `count == 1`, where the old charge paid
        // `gamma_1 ~ u` for a cell that took zero additions and two roundings.
        // `gamma_n_f32` is monotone in `n`, so this is strictly LOOSER than the
        // previous charge at every cell, in both directions and on both operands.
        let depth_a = batch_size.saturating_mul(dims.n).max(1).saturating_add(2);
        let depth_b = batch_size.saturating_mul(dims.m).max(1).saturating_add(2);
        let gamma_a = crate::layers::linear::crown_single_gamma_n_f32(depth_a);
        let gamma_b = crate::layers::linear::crown_single_gamma_n_f32(depth_b);
        let mk_err = |s: Array2<f64>, gamma: f64, shape: &[usize]| -> Result<ArrayD<f32>> {
            let e = s.mapv(|sv| next_up_f32((gamma * sv) as f32));
            let (v, _) = e.into_raw_vec_and_offset();
            ArrayD::from_shape_vec(IxDyn(shape), v)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape coeff err".to_string()))
        };
        let lower_a_a_err = mk_err(s_lower_a_a, gamma_a, &out_a_shape)?;
        let upper_a_a_err = mk_err(s_upper_a_a, gamma_a, &out_a_shape)?;
        let lower_a_b_err = mk_err(s_lower_a_b, gamma_b, &out_b_shape)?;
        let upper_a_b_err = mk_err(s_upper_a_b, gamma_b, &out_b_shape)?;

        // Phase 4 audit: per-layer MatMul McCormick output — catches NaN from McCormick.
        let mut bounds_a = BatchedLinearBounds::new_or_conservative(
            new_lower_a_a,
            lower_b_half.clone(),
            new_upper_a_a,
            upper_b_half.clone(),
            new_input_shape_a,
            bounds.output_shape.clone(),
        )?;
        bounds_a.set_coeff_err(lower_a_a_err, upper_a_a_err);

        let mut bounds_b = BatchedLinearBounds::new_or_conservative(
            new_lower_a_b,
            lower_b_half,
            new_upper_a_b,
            upper_b_half,
            new_input_shape_b,
            bounds.output_shape.clone(),
        )?;
        bounds_b.set_coeff_err(lower_a_b_err, upper_a_b_err);

        Ok((bounds_a, bounds_b))
    }
}

#[cfg(test)]
mod batched_mccormick_constant_certificate_tests {
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    use super::MatMulLayer;
    use crate::BatchedLinearBounds;

    /// The accumulated McCormick envelope CONSTANT of the BATCHED arm must be
    /// published with its own certified error term, in BOTH directions
    /// (#matmul-batched-mccormick-const).
    ///
    /// COUNTEREXAMPLE. `A` is `[1, K]`, `B` is `[K, 1]`, both DEGENERATE boxes
    /// (`lower == upper`), so all four McCormick planes coincide, plane selection
    /// is irrelevant, and the relaxation is exact at the single feasible point.
    /// With `m = n = 1` each coefficient cell receives exactly ONE term, so no
    /// coefficient accumulation error exists and the containment condition reduces
    /// to a statement about the constant alone.
    ///
    /// The incoming relation carries TWO output rows over the single flattened
    /// MatMul output (`c_size = m*n = 1`): row 0 with weight `w = +1`, row 1 with
    /// `w = -1`. Both rows fold the SAME plane constants `c = -X_l*Y_l`, so with
    /// `S = sum_l X_l*Y_l`:
    ///
    /// ```text
    /// row 0:  crown(X, Y) =  2*S + const_0,  true output =  S
    ///         lower sound <=> const_0_lower <= -S
    ///         upper sound <=> const_0_upper >= -S
    /// row 1:  crown(X, Y) = -2*S + const_1,  true output = -S
    ///         lower sound <=> const_1_lower <=  S
    ///         upper sound <=> const_1_upper >=  S
    /// ```
    ///
    /// The taps make the f64 fold catastrophically wrong in a way that is
    /// SIGN-SYMMETRIC, which is why one call has teeth in both directions. Row 0
    /// folds `+2^60`, 4096 x `-1`, `-2^60`; row 1 folds the exact negation. Every
    /// `2^60 -+ 1` rounds straight back to `2^60` (the f64 ULP there is 256), so
    /// BOTH accumulators return EXACTLY `0` while the true constants are `-4096`
    /// and `+4096`. Publishing those `0`s with only the f64 -> f32 halving cast to
    /// protect them puts row 0's LOWER plane `4096` too HIGH (it sits strictly
    /// ABOVE the bilinear surface at a feasible point) and row 1's UPPER plane
    /// `4096` too LOW. A ULP of `0.0` is `1.4e-45`; the real error is `4096`.
    ///
    /// So the row-0 lower assertion and the row-1 upper assertion BOTH FAIL on the
    /// pre-fix code, and a single-direction fix cannot pass. The two remaining
    /// directions are already loose on this fixture and are pinned so the
    /// certificate cannot be applied INWARD by mistake.
    ///
    /// The term-free (`adds == 0`) contract — the constant staying bit-identical
    /// to the historical directed halving — is covered by
    /// `super::super::rounding_tests::batched_crown_bias_halving_uses_directed_rounding_2173`.
    #[ntest::timeout(30000)]
    #[test]
    fn batched_envelope_constant_carries_its_higham_charge_in_both_directions() {
        const K: usize = 4098;
        const BIG: f32 = 1_073_741_824.0; // 2^30; BIG * BIG = 2^60, exact in f32.

        let mut xs = vec![1.0_f32; K];
        let mut ys = vec![1.0_f32; K];
        xs[0] = BIG;
        ys[0] = -BIG;
        xs[K - 1] = BIG;
        ys[K - 1] = BIG;

        // Preconditions of the counterexample, pinned so a future change cannot
        // silently turn this into a test of nothing.
        let objective_exact: i128 = (0..K).map(|l| (xs[l] as f64 * ys[l] as f64) as i128).sum();
        assert_eq!(objective_exact, 4096, "true objective sum_l X_l*Y_l");
        let objective = objective_exact as f64;

        // Exactly what the production loop accumulates per row: w = +-1,
        // c = fl32(-X*Y), and f32 x f32 is exact in f64.
        let mut folded_row0 = 0.0_f64;
        let mut folded_row1 = 0.0_f64;
        for l in 0..K {
            let term = -(xs[l] as f64) * (ys[l] as f64);
            folded_row0 += term;
            folded_row1 += -term;
        }
        assert_eq!(
            folded_row0, 0.0,
            "the row-0 f64 constant fold must cancel to 0 for this to exercise the defect"
        );
        assert_eq!(
            folded_row1, 0.0,
            "the row-1 f64 constant fold must cancel to 0 for this to exercise the defect"
        );

        let a = ArrayD::from_shape_vec(IxDyn(&[1, K]), xs.clone()).expect("a shape");
        let b = ArrayD::from_shape_vec(IxDyn(&[K, 1]), ys.clone()).expect("b shape");
        let input_a = BoundedTensor::new(a.clone(), a).expect("input_a");
        let input_b = BoundedTensor::new(b.clone(), b).expect("input_b");

        let incoming = BatchedLinearBounds::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0_f32, -1.0]).expect("incoming lower_a"),
            ArrayD::from_elem(IxDyn(&[2]), 0.0_f32),
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0_f32, -1.0]).expect("incoming upper_a"),
            ArrayD::from_elem(IxDyn(&[2]), 0.0_f32),
            vec![1],
            vec![2],
        )
        .expect("incoming bounds");

        let matmul = MatMulLayer::new(false, None);
        let (bounds_a, bounds_b) = matmul
            .propagate_linear_batched_binary(&incoming, &input_a, &input_b)
            .expect("batched McCormick CROWN backward");

        // Premise of the reduction above: every coefficient cell holds exactly one
        // exact term, so the whole error budget lives in the constant.
        for &l in &[0_usize, 1, K - 1] {
            assert_eq!(
                bounds_a.lower_a[[0, l]],
                ys[l],
                "row 0 A-side lower coefficient {l} must be the single exact term w*ly"
            );
            assert_eq!(
                bounds_a.upper_a[[0, l]],
                ys[l],
                "row 0 A-side upper coefficient {l} must be the single exact term w*uy"
            );
            assert_eq!(
                bounds_b.lower_a[[0, l]],
                xs[l],
                "row 0 B-side lower coefficient {l} must be the single exact term w*lx"
            );
            assert_eq!(
                bounds_b.upper_a[[0, l]],
                xs[l],
                "row 0 B-side upper coefficient {l} must be the single exact term w*ux"
            );
            assert_eq!(
                bounds_a.lower_a[[1, l]],
                -ys[l],
                "row 1 A-side lower coefficient {l} must be the single exact term w*ly"
            );
            assert_eq!(
                bounds_a.upper_a[[1, l]],
                -ys[l],
                "row 1 A-side upper coefficient {l} must be the single exact term w*uy"
            );
            assert_eq!(
                bounds_b.lower_a[[1, l]],
                -xs[l],
                "row 1 B-side lower coefficient {l} must be the single exact term w*lx"
            );
            assert_eq!(
                bounds_b.upper_a[[1, l]],
                -xs[l],
                "row 1 B-side upper coefficient {l} must be the single exact term w*ux"
            );
        }

        // The constant is split across both operand branches; the relation the
        // caller evaluates sums the two halves.
        let lower_0 = bounds_a.lower_b[[0]] as f64 + bounds_b.lower_b[[0]] as f64;
        let upper_0 = bounds_a.upper_b[[0]] as f64 + bounds_b.upper_b[[0]] as f64;
        let lower_1 = bounds_a.lower_b[[1]] as f64 + bounds_b.lower_b[[1]] as f64;
        let upper_1 = bounds_a.upper_b[[1]] as f64 + bounds_b.upper_b[[1]] as f64;

        // FAILS PRE-FIX: the bare fold publishes ~0, which is 4096 too HIGH.
        assert!(
            lower_0 <= -objective,
            "published row-0 McCormick LOWER constant {lower_0} must be <= {} — otherwise \
             the lower relaxation sits ABOVE the bilinear surface at a feasible point",
            -objective
        );
        // FAILS PRE-FIX: the bare fold publishes ~0, which is 4096 too LOW.
        assert!(
            upper_1 >= objective,
            "published row-1 McCormick UPPER constant {upper_1} must be >= {objective} — \
             otherwise the upper relaxation sits BELOW the bilinear surface at a feasible point"
        );
        // Already loose on this fixture; pinned so the certificate can never be
        // applied INWARD.
        assert!(
            upper_0 >= -objective,
            "published row-0 McCormick UPPER constant {upper_0} must stay >= {}",
            -objective
        );
        assert!(
            lower_1 <= objective,
            "published row-1 McCormick LOWER constant {lower_1} must stay <= {objective}"
        );
    }
}
