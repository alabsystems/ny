// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::super::matmul::{
    decode_batch_index_into_buf, mccormick_gemm_backward, parse_matmul_dims, GemmSchedule,
    MatMulLayer, McCormickOperands, McCormickPlaneTable, McCormickRowChunk, McCormickRowparBuffers,
    McCormickRowparShape,
};
use super::super::mul::BoundDir;
use super::super::validate_mccormick_inputs;
use super::{interpolated_mccormick, BilinearCrownLayer};
use crate::layers::collector_kernels::BilinearKernelChoice;

/// Widen one accumulated McCormick ENVELOPE CONSTANT outward by a certified bound
/// on its own floating-point error (#bilinear-alpha-mccormick-const).
///
/// Byte-for-byte the certificate of the two twins that fold the SAME shape —
/// `MatMulLayer::propagate_linear_binary` (`binary_ops/matmul/crown_dense.rs`,
/// #matmul-dense-mccormick-const) and `MulBinaryLayer` (`binary_ops/mul/mod.rs`,
/// #mul-mccormick-const, where it is wrapped as `McCormickConstFold`). It is
/// duplicated here rather than shared because the `mul` copy is private to that
/// module; the symbol names are kept IDENTICAL so a census grep for
/// `McCormickConstFold\|widen_mccormick_constant` finds all three.
///
/// # The defect this closes
///
/// [`BilinearCrownLayer::propagate_linear_binary_with_alpha`] folded the envelope
/// constant as `const += w * c` in PLAIN round-to-nearest f64 and published it
/// with no error term at all — only the final f64 -> f32 directed cast in the
/// halving step. That cast covers the narrowing and nothing else, so two distinct
/// errors reached the verdict path uncharged:
///
/// 1. **f64 accumulation.** Recursive summation of `adds + 1` summands satisfies
///    `|fl(S) - S| <= gamma^f64_{adds} * sum|summand|` (Higham, *Accuracy and
///    Stability*, Thm 3.1). The charge scales with the MAGNITUDE sum, not with
///    `|S|`, so it does NOT vanish under cancellation — which is exactly when a
///    result-scaled 1-ULP cast allowance under-counts it. This arm's fold depth is
///    `batch_size * m * n * k`, i.e. 1e4-1e6 on attention-shaped nodes, so
///    `gamma^f64_d` reaches ~1e-10 while a directed f32 cast on a cancelling
///    constant allows only ~2^-24 of a result that has collapsed toward zero.
///    Worked case: terms `+2^60`, 4096 x `-1`, `-2^60` fold to exactly `0.0` in
///    f64 where the truth is `-4096`; the magnitude-sum charge covers it, a
///    result-scaled ULP does not.
/// 2. **The plane constant and the weight.** `w` is `fl32(lower_a * scale)` — one
///    round-to-nearest f32 multiply — and `c` is the interpolated-McCormick plane
///    constant from [`interpolated_mccormick`]. Either rounding can move the plane
///    the UNSOUND way: a lower plane whose constant rounded UP no longer
///    under-estimates the bilinear surface, so the relaxation stops containing
///    `x*y`. Both are relative to `|w*c|`, hence a charge at f32 unit roundoff
///    against the TERM magnitude sum. (`interpolated_mccormick` routes both
///    constants through `bilinear::mccormick::certify_plane_constant`, which
///    already rounds each plane constant OUTWARD, so that half of this charge is
///    redundant slack rather than a repair. It is still applied: one uniform
///    certificate cannot drift out of sync with its two twins. The `* scale`
///    rounding is a genuine repair.)
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
/// The `1/(1 - gamma)` inflation mirrors `conv2d::bound`: `magnitude_sum` is
/// itself an f64 round-to-nearest sum of non-negative values and can fall SHORT of
/// the true magnitude sum by its own accumulation deficit, so charging against the
/// raw figure could under-bound the very error it is meant to cover. The final
/// directed f64 step absorbs the rounding of the widening subtract/add.
///
/// # Direction
///
/// `err >= 0` always (`gamma >= 0`, both magnitude sums are sums of `abs` values,
/// `inflate >= 1`), so a `Lower` constant only ever moves DOWN and an `Upper`
/// constant only ever moves UP. `adds == 0` returns the input bit-identically: no
/// f64 addition ran and no McCormick plane was consumed, so there is genuinely
/// nothing to charge. A NaN anywhere fails CLOSED to the infinite constant.
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
    // 2 f32 roundings per term (`* scale`, and the plane constant returned by
    // `interpolated_mccormick`) plus one step of slack.
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

/// Running McCormick envelope CONSTANT fold that carries its own Higham
/// certificate (#bilinear-alpha-mccormick-const).
///
/// Replaces the bare `const_lower += w as f64 * c as f64` chain in
/// [`BilinearCrownLayer::propagate_linear_binary_with_alpha`]. The `alphas ==
/// None` sibling of that arm delegates to `MatMulLayer::propagate_linear_binary`,
/// which already carries this certificate, so leaving the alpha arm bare made the
/// proof object STRICTLY WEAKER the moment the DAG alpha-CROWN loop supplied
/// alphas — which it does for every `BilinearCrown` node.
///
/// `magnitude_sum` carries the incoming bias (the f64 accumulation charge is
/// against EVERY summand); `term_magnitude_sum` excludes it (the per-term f32
/// plane/scale rounding charge applies only to the plane terms). `adds` counts the
/// f64 additions actually performed, so a term-free fold charges exactly zero and
/// is published bit-identically to the historical directed-cast result.
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

impl BilinearCrownLayer {
    /// CROWN backward propagation for LinearBounds (flattened graph networks).
    ///
    /// Delegates to MatMulLayer's McCormick relaxation implementation.
    /// This enables BilinearCrown to work with DAG-CROWN graph networks
    /// that use `LinearBounds` instead of `BatchedLinearBounds`.
    ///
    /// # Arguments
    /// * `bounds` - Incoming LinearBounds on the output C
    /// * `input_a_bounds` - IBP bounds on input A (Q in attention)
    /// * `input_b_bounds` - IBP bounds on input B (K in attention)
    ///
    /// # Returns
    /// Two sets of linear bounds: one for A input, one for B input.
    pub fn propagate_linear_binary(
        &self,
        bounds: &crate::LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(crate::LinearBounds, crate::LinearBounds)> {
        let matmul = MatMulLayer::new(self.transpose_b, self.scale);
        matmul.propagate_linear_binary(bounds, input_a_bounds, input_b_bounds)
    }

    /// CROWN backward propagation with alpha-parameterized McCormick interpolation.
    ///
    /// Like `propagate_linear_binary`, but uses `interpolated_mccormick` with
    /// optimizable `r_l, r_u ∈ [0,1]` per-element instead of a fixed midpoint
    /// heuristic. This enables joint ReLU + bilinear alpha optimization in
    /// the DAG α-CROWN loop.
    ///
    /// Alpha shape: `[4, m, n, k]` where m, n, k are the matmul dimensions.
    /// - `[0, i, j, l]`: r_l for lower bound computation
    /// - `[1, i, j, l]`: r_l for upper bound computation (direction-dependent)
    /// - `[2, i, j, l]`: r_u for lower bound computation
    /// - `[3, i, j, l]`: r_u for upper bound computation (direction-dependent)
    ///
    /// When `alphas` is None, delegates to the fixed `propagate_linear_binary`.
    ///
    /// # Certified coefficient error (#vnncomp-aw-soundness)
    ///
    /// This arm publishes the SAME Higham certificate its `alphas == None`
    /// sibling does. That sibling delegates to
    /// `MatMulLayer::propagate_linear_binary` (`binary_ops/matmul/crown_dense.rs`),
    /// which accumulates `S = Σ |w| · |plane_coeff|` in f64 and publishes
    /// `|stored − exact| ≤ γ^{f32}_depth · S` via
    /// [`crate::LinearBounds::new_or_conservative_with_err`]. The in-file loop
    /// below performs the identical f32 `+=` composition, so it carries the
    /// identical error; publishing it through `new_or_conservative` (error
    /// `None`) asserted the coefficients were EXACT and silently DOWNGRADED the
    /// proof object the moment the DAG α-CROWN loop supplied alphas — which it
    /// does for every `BilinearCrown` node
    /// (`network/graph_alpha/propagate_dag/init.rs`, `Array4::ones`). Under
    /// cancellation the deviation is unbounded relative to the result, so the
    /// missing charge is a false-`unsat` mechanism, not a tightness detail.
    ///
    /// Accumulation depth: an A-side cell `(batch, i, l)` receives one `+=` per
    /// `j ∈ 0..dims.n`; a B-side cell `(batch, l, j)` receives one per
    /// `i ∈ 0..dims.m`. Those extents are UPPER bounds on the true depth (the
    /// `w != 0.0` guards can only skip iterations), and the charge is
    /// `γ_{extent+2}`, byte-for-byte `crown_dense.rs`'s corrected charge (the
    /// two extra steps cover the `fl32(lower_a · scale)` rounding and the f64
    /// deficit of `S`), so it is outward.
    ///
    /// # Certified envelope CONSTANT (#bilinear-alpha-mccormick-const)
    ///
    /// The coefficient certificate above is only HALF the relaxation. The other
    /// half — the McCormick plane constants — was folded as
    /// `const += w as f64 * c as f64` in plain round-to-nearest f64 over a
    /// `batch_size × m × n × k` nest and published through a directed f32 cast
    /// alone. That cast is scaled by the RESULT, so it collapses toward a
    /// subnormal exactly when cancellation makes the accumulation error largest,
    /// while `γ^{f64}_d` at `d ≈ 1e4–1e6` is ~1e-10 of the MAGNITUDE sum. Both
    /// constants now fold through [`McCormickConstFold`], which publishes the same
    /// Higham certificate its two twins do (`matmul/crown_dense.rs`,
    /// `mul/mod.rs`) — the `alphas == None` sibling of this very function already
    /// carried it, so the alpha arm was strictly the weaker proof object.
    ///
    /// # The two returned relations are SHARES (#concretize-vacuity)
    ///
    /// Like its `alphas == None` sibling, this returns one relation per operand
    /// with the certified constant halved onto each; only their sum over
    /// `[A; B]` encloses the incoming relation. A half concretized on its own
    /// can come back `[-inf, +inf]` from `concretize_sound`'s inversion repair
    /// with every coefficient finite — that is the repair doing its job on a
    /// non-enclosure, not a defect of this arm. See
    /// `matmul_operand_half_inverts_alone_and_only_the_summed_relation_encloses`.
    ///
    /// # Reference
    /// auto_LiRPA operators/bivariate.py:MulHelper.interpolated_relaxation (r_l, r_u)
    pub fn propagate_linear_binary_with_alpha(
        &self,
        bounds: &crate::LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        alphas: Option<&ndarray::Array4<f32>>,
    ) -> Result<(crate::LinearBounds, crate::LinearBounds)> {
        self.propagate_linear_binary_with_alpha_kernels(
            bounds,
            input_a_bounds,
            input_b_bounds,
            alphas,
            BilinearKernelChoice::armed(),
        )
    }

    /// [`Self::propagate_linear_binary_with_alpha`] with the kernel choice
    /// explicit (#vit-collector-throughput P2 step 1).
    ///
    /// `rowpar == false` is the historical serial nest, byte-for-byte, and the
    /// `alphas == None` delegation is the historical midpoint arm with the SAME
    /// kernel choice. `rowpar == true` first offers the problem to the
    /// bit-identical row-parallel twin (`propagate_linear_binary_with_alpha_rowpar`),
    /// which declines shapes it does not handle. Never tries the GEMM-shaped
    /// kernel (see [`Self::propagate_linear_binary_with_alpha_kernels`]).
    /// Test-only since #vit-batched-bilinear routed production through
    /// `_with_alpha_kernels`.
    #[cfg(test)]
    pub(crate) fn propagate_linear_binary_with_alpha_kernel(
        &self,
        bounds: &crate::LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        alphas: Option<&ndarray::Array4<f32>>,
        rowpar: bool,
    ) -> Result<(crate::LinearBounds, crate::LinearBounds)> {
        self.propagate_linear_binary_with_alpha_kernels(
            bounds,
            input_a_bounds,
            input_b_bounds,
            alphas,
            BilinearKernelChoice {
                gemm: false,
                rowpar,
            },
        )
    }

    /// [`Self::propagate_linear_binary_with_alpha`] with every kernel choice
    /// explicit (#vit-batched-bilinear, ENGINE_PROGRAM V1).
    ///
    /// Tried in order, each only when chosen: the GEMM-shaped kernel
    /// (`propagate_linear_binary_with_alpha_gemm`, the dense arm's
    /// `crown_gemm.rs` machinery), the row-parallel kernel, then the serial
    /// nest. Both kernels are BIT-IDENTICAL to the nest and decline shapes they
    /// do not handle. The `alphas == None` delegation is the historical
    /// midpoint arm with the SAME choice. `BilinearKernelChoice::SERIAL` is the
    /// serial nest byte-for-byte. Production passes the typed latches
    /// (`bab.bilinear_gemm_kernel`, `bab.bilinear_rowpar_kernel`).
    pub(crate) fn propagate_linear_binary_with_alpha_kernels(
        &self,
        bounds: &crate::LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        alphas: Option<&ndarray::Array4<f32>>,
        kernels: BilinearKernelChoice,
    ) -> Result<(crate::LinearBounds, crate::LinearBounds)> {
        let alphas = match alphas {
            Some(a) => a,
            None => {
                return MatMulLayer::new(self.transpose_b, self.scale)
                    .propagate_linear_binary_with_kernels(
                        bounds,
                        input_a_bounds,
                        input_b_bounds,
                        kernels,
                    )
            }
        };

        let dims = parse_matmul_dims(
            self.transpose_b,
            input_a_bounds.shape(),
            input_b_bounds.shape(),
        )?;

        // Validate alpha shape matches [4, m, n, k]
        if alphas.shape() != [4, dims.m, dims.n, dims.k] {
            return Err(NyError::ShapeMismatch {
                expected: vec![4, dims.m, dims.n, dims.k],
                got: alphas.shape().to_vec(),
            });
        }

        validate_mccormick_inputs(input_a_bounds, input_b_bounds, "BilinearCrown")?;

        let batch_size = dims.batch_size()?;
        let c_size_per_batch = dims.c_size_per_batch()?;
        let a_size_per_batch = dims.a_size_per_batch()?;

        let total_c_size = batch_size.checked_mul(c_size_per_batch).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BilinearCrown: total_c_size overflow: {batch_size} * {c_size_per_batch}",
            ))
        })?;
        let total_a_size = batch_size.checked_mul(a_size_per_batch).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BilinearCrown: total_a_size overflow: {batch_size} * {a_size_per_batch}",
            ))
        })?;
        let total_b_size = batch_size
            .checked_mul(dims.b_size_per_batch)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "BilinearCrown: total_b_size overflow: {batch_size} * {}",
                    dims.b_size_per_batch,
                ))
            })?;

        if bounds.num_inputs() != total_c_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_c_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        let scale = self.scale.unwrap_or(1.0);

        // #vit-batched-bilinear (typed `bab.bilinear_gemm_kernel`) and
        // #vit-collector-throughput P2 step 1 (typed `bab.bilinear_rowpar_kernel`),
        // both dark by default: the bit-identical GEMM-shaped and row-parallel
        // twins of the nest below. `None` = declined shape; the next choice
        // (finally the serial nest) then runs unchanged (and the nest raises its
        // own errors, e.g. a zero-sized batch dimension or more batch dims than
        // its stack index buffer holds).
        if kernels.gemm || kernels.rowpar {
            let degenerate = dims.batch_dims.contains(&0) || dims.batch_dims.len() + 2 > 8;
            let shape = McCormickRowparShape {
                batch_size,
                m: dims.m,
                n: dims.n,
                k: dims.k,
                c_size_per_batch,
                a_size_per_batch,
                b_size_per_batch: dims.b_size_per_batch,
                total_a_size,
                total_b_size,
                transpose_b: self.transpose_b,
            };
            if kernels.gemm {
                let published = if degenerate {
                    None
                } else {
                    self.propagate_linear_binary_with_alpha_gemm(
                        bounds,
                        input_a_bounds,
                        input_b_bounds,
                        alphas,
                        &shape,
                        scale,
                        None,
                    )?
                };
                match published {
                    Some(published) => return Ok(published),
                    None => crate::layers::collector_kernels::record_bilinear_gemm_declined(),
                }
            }
            if kernels.rowpar {
                let published = if degenerate {
                    None
                } else {
                    self.propagate_linear_binary_with_alpha_rowpar(
                        bounds,
                        input_a_bounds,
                        input_b_bounds,
                        alphas,
                        &shape,
                        scale,
                    )?
                };
                match published {
                    Some(published) => return Ok(published),
                    None => crate::layers::collector_kernels::record_bilinear_declined(),
                }
            }
        }

        let mut lower_a_a = Array2::<f32>::zeros((num_outputs, total_a_size));
        let mut lower_a_b = Array2::<f32>::zeros((num_outputs, total_b_size));
        let mut upper_a_a = Array2::<f32>::zeros((num_outputs, total_a_size));
        let mut upper_a_b = Array2::<f32>::zeros((num_outputs, total_b_size));

        // Certified-error accumulators, byte-for-byte the construction in
        // `matmul/crown_dense.rs` (#matmul-dense-mccormick): each coefficient cell
        // is summed in round-to-nearest f32 over the j axis (A side, depth
        // `dims.n`) / i axis (B side, depth `dims.m`), so it carries the Higham f32
        // accumulation error `γ_n · S` that MUST reach concretize. `S = Σ|w·a|` is
        // accumulated exactly in f64 (f32×f32 fits in 48 < 53 bits).
        let mut lower_s_a = Array2::<f64>::zeros((num_outputs, total_a_size));
        let mut lower_s_b = Array2::<f64>::zeros((num_outputs, total_b_size));
        let mut upper_s_a = Array2::<f64>::zeros((num_outputs, total_a_size));
        let mut upper_s_b = Array2::<f64>::zeros((num_outputs, total_b_size));

        let mut lower_b_total = Array1::<f64>::zeros(num_outputs);
        let mut upper_b_total = Array1::<f64>::zeros(num_outputs);

        for out_idx in 0..num_outputs {
            // Certified ENVELOPE-CONSTANT folds (#bilinear-alpha-mccormick-const).
            // The bare `+= w as f64 * c as f64` chain these replace published a
            // round-to-nearest f64 sum of up to `batch_size * m * n * k` terms with
            // NO error term — only the directed f32 cast in the halving step below,
            // which is scaled by the RESULT and therefore collapses exactly when
            // cancellation makes the accumulation error matter. See
            // `widen_mccormick_constant`.
            let mut const_lower =
                McCormickConstFold::new(bounds.lower_b()[out_idx], BoundDir::Lower);
            let mut const_upper =
                McCormickConstFold::new(bounds.upper_b()[out_idx], BoundDir::Upper);
            let batch_index_len = dims.batch_dims.len();
            // Stack-allocated index buffers.
            assert!(
                batch_index_len + 2 <= 8,
                "BilinearCrown: batch_index_len + 2 exceeds stack buffer"
            );
            let mut a_idx = [0usize; 8];
            let mut b_idx = [0usize; 8];
            let idx_len = batch_index_len + 2;

            for batch_idx in 0..batch_size {
                decode_batch_index_into_buf(
                    batch_idx,
                    &dims.batch_dims,
                    &mut a_idx[..batch_index_len],
                )?;
                b_idx[..batch_index_len].copy_from_slice(&a_idx[..batch_index_len]);
                a_idx[batch_index_len + 1] = 0;
                b_idx[batch_index_len] = 0;
                b_idx[batch_index_len + 1] = 0;

                for i in 0..dims.m {
                    a_idx[batch_index_len] = i;
                    for j in 0..dims.n {
                        let c_flat = batch_idx * c_size_per_batch + i * dims.n + j;
                        let w_lower = bounds.lower_a()[[out_idx, c_flat]] * scale;
                        let w_upper = bounds.upper_a()[[out_idx, c_flat]] * scale;
                        if self.transpose_b {
                            b_idx[batch_index_len] = j;
                        } else {
                            b_idx[batch_index_len + 1] = j;
                        }

                        for l in 0..dims.k {
                            // Get A[batch..., i, l]
                            a_idx[batch_index_len + 1] = l;
                            let lx = input_a_bounds.lower()[&a_idx[..idx_len]];
                            let ux = input_a_bounds.upper()[&a_idx[..idx_len]];

                            let a_flat = batch_idx * a_size_per_batch + i * dims.k + l;

                            // Get B[batch..., l, j] or B[batch..., j, l] if transposed
                            let b_flat = if self.transpose_b {
                                b_idx[batch_index_len + 1] = l;
                                batch_idx * dims.b_size_per_batch + j * dims.k + l
                            } else {
                                b_idx[batch_index_len] = l;
                                batch_idx * dims.b_size_per_batch + l * dims.n + j
                            };
                            let ly = input_b_bounds.lower()[&b_idx[..idx_len]];
                            let uy = input_b_bounds.upper()[&b_idx[..idx_len]];

                            // Non-finite guard: if any bound is non-finite, use
                            // conservative trivial relaxation (same as
                            // select_mccormick_plane NaN guard).
                            // `force_trivial` assigns -inf/+inf directly; do NOT
                            // multiply by weight, because negative weights would flip
                            // the sign. It ALSO poisons the fold's magnitude
                            // sums to +inf, so a later `add_term` can never let the
                            // certificate publish a finite constant out of a box that
                            // was already known non-finite
                            // (#bilinear-alpha-mccormick-const).
                            if !lx.is_finite()
                                || !ux.is_finite()
                                || !ly.is_finite()
                                || !uy.is_finite()
                            {
                                if w_lower != 0.0 {
                                    const_lower.force_trivial();
                                }
                                if w_upper != 0.0 {
                                    const_upper.force_trivial();
                                }
                                continue;
                            }

                            // Get interpolation parameters from alpha state.
                            // [0]: r_l for lower, [1]: r_l for upper,
                            // [2]: r_u for lower, [3]: r_u for upper.
                            let r_l_lower = alphas[[0, i, j, l]].clamp(0.0, 1.0);
                            let r_u_lower = alphas[[2, i, j, l]].clamp(0.0, 1.0);
                            let r_l_upper = alphas[[1, i, j, l]].clamp(0.0, 1.0);
                            let r_u_upper = alphas[[3, i, j, l]].clamp(0.0, 1.0);

                            // Compute interpolated McCormick for lower bound direction
                            let (al_l, bl_l, gl_l, au_l, bu_l, gu_l) =
                                interpolated_mccormick(lx, ux, ly, uy, r_l_lower, r_u_lower);

                            // Compute interpolated McCormick for upper bound direction
                            let (al_u, bl_u, gl_u, au_u, bu_u, gu_u) =
                                interpolated_mccormick(lx, ux, ly, uy, r_l_upper, r_u_upper);

                            if w_lower != 0.0 {
                                // For lower bound: w >= 0 uses lower relaxation,
                                // w < 0 uses upper relaxation.
                                let (ax, ay, c) = if w_lower >= 0.0 {
                                    (al_l, bl_l, gl_l)
                                } else {
                                    (au_l, bu_l, gu_l)
                                };
                                lower_a_a[[out_idx, a_flat]] += w_lower * ax;
                                lower_a_b[[out_idx, b_flat]] += w_lower * ay;
                                lower_s_a[[out_idx, a_flat]] +=
                                    (w_lower as f64).abs() * (ax as f64).abs();
                                lower_s_b[[out_idx, b_flat]] +=
                                    (w_lower as f64).abs() * (ay as f64).abs();
                                const_lower.add_term(w_lower, c);
                            }

                            if w_upper != 0.0 {
                                // For upper bound: w >= 0 uses upper relaxation,
                                // w < 0 uses lower relaxation.
                                let (ax, ay, c) = if w_upper >= 0.0 {
                                    (au_u, bu_u, gu_u)
                                } else {
                                    (al_u, bl_u, gl_u)
                                };
                                upper_a_a[[out_idx, a_flat]] += w_upper * ax;
                                upper_a_b[[out_idx, b_flat]] += w_upper * ay;
                                upper_s_a[[out_idx, a_flat]] +=
                                    (w_upper as f64).abs() * (ax as f64).abs();
                                upper_s_b[[out_idx, b_flat]] +=
                                    (w_upper as f64).abs() * (ay as f64).abs();
                                const_upper.add_term(w_upper, c);
                            }
                        }
                    }
                }
            }

            lower_b_total[out_idx] = const_lower.finish();
            upper_b_total[out_idx] = const_upper.finish();
        }

        // Split constant terms across both inputs for GraphNetwork accumulation.
        // Halve in f64, then apply directed rounding on final f32 cast. The
        // f64 totals reaching this point have ALREADY been widened outward for their
        // own accumulation and plane/scale-rounding error by `widen_mccormick_constant`
        // (via `McCormickConstFold::finish`); the directed cast below covers ONLY the
        // f64 -> f32 narrowing, which is NOT sufficient on its own
        // (#bilinear-alpha-mccormick-const, and see the identical note in
        // `matmul/crown_dense.rs`). Ordering matters: `finish()` runs BEFORE this
        // `* 0.5`, and 0.5 is a power of two, so the halving is EXACT and cannot
        // undo the direction the certificate established.
        let lower_b_half = lower_b_total.mapv(|v| next_down_f32((v * 0.5) as f32));
        let upper_b_half = upper_b_total.mapv(|v| next_up_f32((v * 0.5) as f32));

        // Certified coefficient error (#matmul-dense-mccormick), identical to the
        // alphas == None arm: A-side cells accumulate over `dims.n` terms in f32,
        // B-side cells over `dims.m`, and `|stored − exact| ≤ γ^{f32}_{count+2} · S`
        // (Higham, f32 unit roundoff), rounded UP to a sound f32 and carried so
        // concretize penalizes OUTWARD.
        //
        // #vit-margin-alpha review (2026-09-24): this arm used to charge
        // `γ_count`, which covers the `count − 1` ADDITIONS plus the `fl32(w·ax)`
        // product rounding only. Each summand ALSO carries `w = fl32(lower_a ·
        // scale)`, which rounds whenever `scale` is not a power of two, so the
        // honest factor is `γ_{count+1}` — exactly the deficit the `None`
        // sibling (`matmul/crown_dense.rs`, #matmul-dense-mccormick-const) was
        // corrected for, while this arm kept the old charge. It became live for
        // more than `BilinearCrown` when #vit-margin-alpha routed var×var
        // `MatMul` nodes (which may carry a `scale` attribute) through this arm.
        // `count + 2` is now byte-for-byte the `None` arm's charge: one step
        // beyond the honest `γ_{count+1}`, which also absorbs the (far smaller)
        // f64 accumulation deficit of `S` itself. Monotone in `count`, so this is
        // strictly looser (outward) than the previous charge at every cell.
        //
        // `next_up_f32` adds a full f32 ULP (relative ≥ 2^-24), covering the
        // ≤ 0.5-ULP f64→f32 cast-down and the γ evaluation. A non-finite `γ`
        // (`crown_single_gamma_n_f32` returns +∞ once `n·2^-24 ≥ 1`) gives a
        // non-finite err, which `new_or_conservative_with_err` sanitizes to `+∞`
        // and concretize degrades the row on — fail-closed.
        let gamma_a = crate::layers::linear::crown_single_gamma_n_f32(dims.n.saturating_add(2));
        let gamma_b = crate::layers::linear::crown_single_gamma_n_f32(dims.m.saturating_add(2));
        let lower_a_a_err = lower_s_a.mapv(|s| next_up_f32((gamma_a * s) as f32));
        let upper_a_a_err = upper_s_a.mapv(|s| next_up_f32((gamma_a * s) as f32));
        let lower_a_b_err = lower_s_b.mapv(|s| next_up_f32((gamma_b * s) as f32));
        let upper_a_b_err = upper_s_b.mapv(|s| next_up_f32((gamma_b * s) as f32));

        let bounds_a = crate::LinearBounds::new_or_conservative_with_err(
            lower_a_a,
            lower_b_half.clone(),
            upper_a_a,
            upper_b_half.clone(),
            lower_a_a_err,
            upper_a_a_err,
        )?;

        let bounds_b = crate::LinearBounds::new_or_conservative_with_err(
            lower_a_b,
            lower_b_half,
            upper_a_b,
            upper_b_half,
            lower_a_b_err,
            upper_a_b_err,
        )?;

        Ok((bounds_a, bounds_b))
    }

    /// #vit-margin-alpha (B4): steering gradient of ONE lower-bound objective
    /// row with respect to this node's LOWER-computation interpolation
    /// parameters, evaluated at a concrete operand point. Accumulates into
    /// `grad` (shape `[4, m, n, k]`, ascent convention: `+` raises the lower
    /// bound) and returns the number of product terms that contributed.
    ///
    /// [`Self::propagate_linear_binary_with_alpha`] relaxes each product term
    /// `w·x·y` (`w = coefficient·scale`, the incoming LOWER coefficient) by the
    /// interpolated plane `L2 + r·(L1 − L2)` with `r = alphas[0,i,j,l]` when
    /// `w ≥ 0`, and by `U2 + r·(U1 − U2)` with `r = alphas[2,i,j,l]` when
    /// `w < 0`. For a fixed downstream relaxation the lower bound is a
    /// minimum over the input box of an expression affine in `r`, so by the
    /// envelope (Danskin) theorem its derivative is `w·(L1 − L2)(x*, y*)`
    /// (resp. `w·(U1 − U2)(x*, y*)`) at the row's minimiser — see
    /// [`super::mccormick::interpolated_plane_derivative`]. `a_point` /
    /// `b_point` carry the concrete operand values at the row's input argmin:
    /// exact when the operands are affine in the network input, and the same
    /// documented point-forward surrogate `backward/gradients.rs` uses for
    /// ReLU α deeper in the graph.
    ///
    /// Slots 1 and 3 (the UPPER-bound computation) are never touched. The
    /// α tensor is shared over batch dimensions, so contributions from every
    /// batch index sum into the same `[slot, i, j, l]` cell — the exact chain
    /// rule for a parameter shared across those uses.
    ///
    /// Row-major flat indexing of the operands matches the relaxation's own
    /// (`parse_matmul_dims` requires identical batch dims on both sides).
    ///
    /// SOUNDNESS: none at stake. This value only steers which parameter the
    /// optimizer tries next; every parameter it can reach is clamped into
    /// `[0, 1]`, where `interpolated_mccormick` returns a certified plane.
    ///
    /// # Errors
    ///
    /// Shape mismatches between the operands, the coefficient row, the
    /// points, and `grad`.
    pub(crate) fn margin_alpha_gradient_at_point(
        &self,
        coefficient_row: ndarray::ArrayView1<'_, f32>,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        a_point: &[f32],
        b_point: &[f32],
        grad: &mut ndarray::Array4<f32>,
    ) -> Result<usize> {
        let dims = parse_matmul_dims(
            self.transpose_b,
            input_a_bounds.shape(),
            input_b_bounds.shape(),
        )?;
        if grad.shape() != [4, dims.m, dims.n, dims.k] {
            return Err(NyError::ShapeMismatch {
                expected: vec![4, dims.m, dims.n, dims.k],
                got: grad.shape().to_vec(),
            });
        }
        let batch_size = dims.batch_size()?;
        let c_size_per_batch = dims.c_size_per_batch()?;
        let a_size_per_batch = dims.a_size_per_batch()?;
        let overflow =
            || NyError::InvalidSpec("BilinearCrown margin gradient: size overflow".into());
        let total_c = batch_size
            .checked_mul(c_size_per_batch)
            .ok_or_else(overflow)?;
        let total_a = batch_size
            .checked_mul(a_size_per_batch)
            .ok_or_else(overflow)?;
        let total_b = batch_size
            .checked_mul(dims.b_size_per_batch)
            .ok_or_else(overflow)?;
        let a_lower: Vec<f32> = input_a_bounds.lower().iter().copied().collect();
        let a_upper: Vec<f32> = input_a_bounds.upper().iter().copied().collect();
        let b_lower: Vec<f32> = input_b_bounds.lower().iter().copied().collect();
        let b_upper: Vec<f32> = input_b_bounds.upper().iter().copied().collect();
        if coefficient_row.len() != total_c
            || a_point.len() != total_a
            || b_point.len() != total_b
            || a_lower.len() != total_a
            || b_lower.len() != total_b
        {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_c, total_a, total_b],
                got: vec![coefficient_row.len(), a_point.len(), b_point.len()],
            });
        }
        let scale = self.scale.unwrap_or(1.0);
        let mut contributions = 0usize;
        for batch_idx in 0..batch_size {
            for i in 0..dims.m {
                for j in 0..dims.n {
                    let c_flat = batch_idx * c_size_per_batch + i * dims.n + j;
                    // The SAME f32 product the relaxation forms.
                    let w = coefficient_row[c_flat] * scale;
                    if w == 0.0 || !w.is_finite() {
                        continue;
                    }
                    let lower_plane = w > 0.0;
                    let slot = if lower_plane { 0 } else { 2 };
                    for l in 0..dims.k {
                        let a_flat = batch_idx * a_size_per_batch + i * dims.k + l;
                        let b_flat = if self.transpose_b {
                            batch_idx * dims.b_size_per_batch + j * dims.k + l
                        } else {
                            batch_idx * dims.b_size_per_batch + l * dims.n + j
                        };
                        let values = [
                            a_lower[a_flat],
                            a_upper[a_flat],
                            b_lower[b_flat],
                            b_upper[b_flat],
                            a_point[a_flat],
                            b_point[b_flat],
                        ];
                        if values.iter().any(|v| !v.is_finite()) {
                            continue;
                        }
                        let derivative = super::mccormick::interpolated_plane_derivative(
                            f64::from(values[0]),
                            f64::from(values[1]),
                            f64::from(values[2]),
                            f64::from(values[3]),
                            f64::from(values[4]),
                            f64::from(values[5]),
                            lower_plane,
                        );
                        let contribution = f64::from(w) * derivative;
                        if contribution.is_finite() {
                            grad[[slot, i, j, l]] += contribution as f32;
                            contributions += 1;
                        }
                    }
                }
            }
        }
        Ok(contributions)
    }

    /// Compute output shape for alpha parameters.
    ///
    /// Returns (m, n, k) where alphas should have shape [4, m, n, k].
    /// Inputs must be at least 2D (matmul operands).
    pub fn alpha_shape(
        &self,
        input_a_shape: &[usize],
        input_b_shape: &[usize],
    ) -> Result<(usize, usize, usize)> {
        let a_ndim = input_a_shape.len();
        let b_ndim = input_b_shape.len();
        if a_ndim < 2 || b_ndim < 2 {
            return Err(NyError::InvalidSpec(format!(
                "BilinearCrown alpha_shape: inputs must be >= 2D, got {a_ndim}D and {b_ndim}D"
            )));
        }
        let m = input_a_shape[a_ndim - 2];
        let k = input_a_shape[a_ndim - 1];
        let n = if self.transpose_b {
            input_b_shape[b_ndim - 2]
        } else {
            input_b_shape[b_ndim - 1]
        };
        Ok((m, n, k))
    }

    /// Initialize alpha parameters with default values.
    ///
    /// Default initialization uses r=1.0 (matching auto_LiRPA's torch.ones() default),
    /// which starts optimization from the L1/U1 McCormick planes.
    ///
    /// Returns [4, m, n, k] with direction-dependent layout.
    pub fn init_alpha(
        &self,
        input_a_shape: &[usize],
        input_b_shape: &[usize],
    ) -> Result<ndarray::Array4<f32>> {
        let (m, n, k) = self.alpha_shape(input_a_shape, input_b_shape)?;
        Ok(ndarray::Array4::ones((4, m, n, k)))
    }
}

/// One (b, i, j, l) element's hoisted interpolated-McCormick planes for the
/// row-parallel alpha arm: whether all four operand bounds are finite (the
/// nest's non-finite guard), and — only when they are — the six-tuple
/// `interpolated_mccormick` returns for the LOWER-computation parameters
/// (`alphas[0]`, `alphas[2]`) and for the UPPER-computation ones (`alphas[1]`,
/// `alphas[3]`). None of it depends on the row or on the incoming weight.
#[derive(Clone, Copy, Default)]
struct AlphaSlabPlanes {
    finite: bool,
    lower_dir: (f32, f32, f32, f32, f32, f32),
    upper_dir: (f32, f32, f32, f32, f32, f32),
}

impl BilinearCrownLayer {
    /// #vit-collector-throughput P2 step 1: the row-parallel, plane-hoisted
    /// twin of the serial alpha nest in
    /// [`Self::propagate_linear_binary_with_alpha_kernels`] — BIT-IDENTICAL to it.
    ///
    /// The argument is the dense arm's (`matmul/crown_dense.rs`, the block
    /// comment above `McCormickRowparShape`), with this arm's differences:
    ///
    /// * the planes come from `interpolated_mccormick` with the element's
    ///   clamped parameters — a pure function of `(lx, ux, ly, uy, r_l, r_u)`,
    ///   none of which depends on the row, so computing it once per chunk
    ///   instead of once per row changes no bit;
    /// * the weight selects between the two relaxations of each direction with
    ///   the nest's own predicate (`w >= 0.0`);
    /// * a non-finite element does exactly what the nest does, in the same
    ///   position of the row's sequence: `force_trivial` on each constant fold
    ///   whose weight is nonzero, and no coefficient update;
    /// * the constant folds are the nest's own `McCormickConstFold`, fed the
    ///   same terms in the same order, and published by the same `finish`.
    ///
    /// Returns `Ok(None)` (the caller then runs the serial nest) for a shape it
    /// does not handle.
    #[allow(clippy::too_many_arguments)]
    fn propagate_linear_binary_with_alpha_rowpar(
        &self,
        bounds: &crate::LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        alphas: &ndarray::Array4<f32>,
        shape: &McCormickRowparShape,
        scale: f32,
    ) -> Result<Option<(crate::LinearBounds, crate::LinearBounds)>> {
        let started = std::time::Instant::now();
        let num_outputs = bounds.num_outputs();
        let Some(mut buffers) =
            McCormickRowparBuffers::new(num_outputs, shape.total_a_size, shape.total_b_size)
        else {
            return Ok(None);
        };
        let total_c = bounds.num_inputs();
        let a_lo = crate::contiguous_flat_slice(input_a_bounds.lower());
        let a_hi = crate::contiguous_flat_slice(input_a_bounds.upper());
        let b_lo = crate::contiguous_flat_slice(input_b_bounds.lower());
        let b_hi = crate::contiguous_flat_slice(input_b_bounds.upper());
        let w_lo = crate::contiguous_flat_slice(bounds.lower_a());
        let w_hi = crate::contiguous_flat_slice(bounds.upper_a());
        let bias_lo = crate::contiguous_flat_slice(bounds.lower_b());
        let bias_hi = crate::contiguous_flat_slice(bounds.upper_b());
        let lengths_agree = a_lo.len() == shape.total_a_size
            && a_hi.len() == shape.total_a_size
            && b_lo.len() == shape.total_b_size
            && b_hi.len() == shape.total_b_size
            && Some(w_lo.len()) == num_outputs.checked_mul(total_c)
            && w_hi.len() == w_lo.len()
            && bias_lo.len() == num_outputs
            && bias_hi.len() == num_outputs
            && Some(total_c) == shape.batch_size.checked_mul(shape.c_size_per_batch)
            && alphas.shape() == [4, shape.m, shape.n, shape.k];
        if !lengths_agree {
            return Ok(None);
        }
        let (a_lo, a_hi, b_lo, b_hi) = (&*a_lo, &*a_hi, &*b_lo, &*b_hi);
        let (w_lo, w_hi, bias_lo, bias_hi) = (&*w_lo, &*w_hi, &*bias_lo, &*bias_hi);
        let total_a = shape.total_a_size;
        let total_b = shape.total_b_size;
        let k = shape.k;

        let parallel = buffers.run(|chunk: &mut McCormickRowChunk<'_>| {
            let mut folds: Vec<(McCormickConstFold, McCormickConstFold)> = (0..chunk.rows)
                .map(|r| {
                    let row = chunk.row_start + r;
                    (
                        McCormickConstFold::new(bias_lo[row], BoundDir::Lower),
                        McCormickConstFold::new(bias_hi[row], BoundDir::Upper),
                    )
                })
                .collect();
            let mut slab: Vec<AlphaSlabPlanes> = vec![AlphaSlabPlanes::default(); k];
            for batch_idx in 0..shape.batch_size {
                for i in 0..shape.m {
                    for j in 0..shape.n {
                        for (l, slot) in slab.iter_mut().enumerate() {
                            let a_flat = shape.a_flat(batch_idx, i, l);
                            let lx = a_lo[a_flat];
                            let ux = a_hi[a_flat];
                            let b_flat = shape.b_flat(batch_idx, j, l);
                            let ly = b_lo[b_flat];
                            let uy = b_hi[b_flat];
                            if !lx.is_finite()
                                || !ux.is_finite()
                                || !ly.is_finite()
                                || !uy.is_finite()
                            {
                                *slot = AlphaSlabPlanes::default();
                                continue;
                            }
                            let r_l_lower = alphas[[0, i, j, l]].clamp(0.0, 1.0);
                            let r_u_lower = alphas[[2, i, j, l]].clamp(0.0, 1.0);
                            let r_l_upper = alphas[[1, i, j, l]].clamp(0.0, 1.0);
                            let r_u_upper = alphas[[3, i, j, l]].clamp(0.0, 1.0);
                            *slot = AlphaSlabPlanes {
                                finite: true,
                                lower_dir: interpolated_mccormick(
                                    lx, ux, ly, uy, r_l_lower, r_u_lower,
                                ),
                                upper_dir: interpolated_mccormick(
                                    lx, ux, ly, uy, r_l_upper, r_u_upper,
                                ),
                            };
                        }
                        let c_flat = shape.c_flat(batch_idx, i, j);
                        for (r, (const_lower, const_upper)) in folds.iter_mut().enumerate() {
                            let row = chunk.row_start + r;
                            let w_lower = w_lo[row * total_c + c_flat] * scale;
                            let w_upper = w_hi[row * total_c + c_flat] * scale;
                            // Every action below is guarded by `w != 0.0`, in the
                            // nest as here, so a row with both weights zero is a
                            // no-op for this slab.
                            if w_lower == 0.0 && w_upper == 0.0 {
                                continue;
                            }
                            let la_a = &mut chunk.lower_a_a[r * total_a..(r + 1) * total_a];
                            let la_b = &mut chunk.lower_a_b[r * total_b..(r + 1) * total_b];
                            let ua_a = &mut chunk.upper_a_a[r * total_a..(r + 1) * total_a];
                            let ua_b = &mut chunk.upper_a_b[r * total_b..(r + 1) * total_b];
                            let ls_a = &mut chunk.lower_s_a[r * total_a..(r + 1) * total_a];
                            let ls_b = &mut chunk.lower_s_b[r * total_b..(r + 1) * total_b];
                            let us_a = &mut chunk.upper_s_a[r * total_a..(r + 1) * total_a];
                            let us_b = &mut chunk.upper_s_b[r * total_b..(r + 1) * total_b];
                            for (l, planes) in slab.iter().enumerate() {
                                if !planes.finite {
                                    if w_lower != 0.0 {
                                        const_lower.force_trivial();
                                    }
                                    if w_upper != 0.0 {
                                        const_upper.force_trivial();
                                    }
                                    continue;
                                }
                                let a_flat = shape.a_flat(batch_idx, i, l);
                                let b_flat = shape.b_flat(batch_idx, j, l);
                                let (al_l, bl_l, gl_l, au_l, bu_l, gu_l) = planes.lower_dir;
                                let (al_u, bl_u, gl_u, au_u, bu_u, gu_u) = planes.upper_dir;
                                if w_lower != 0.0 {
                                    let (ax, ay, c) = if w_lower >= 0.0 {
                                        (al_l, bl_l, gl_l)
                                    } else {
                                        (au_l, bu_l, gu_l)
                                    };
                                    la_a[a_flat] += w_lower * ax;
                                    la_b[b_flat] += w_lower * ay;
                                    ls_a[a_flat] += (w_lower as f64).abs() * (ax as f64).abs();
                                    ls_b[b_flat] += (w_lower as f64).abs() * (ay as f64).abs();
                                    const_lower.add_term(w_lower, c);
                                }
                                if w_upper != 0.0 {
                                    let (ax, ay, c) = if w_upper >= 0.0 {
                                        (au_u, bu_u, gu_u)
                                    } else {
                                        (al_u, bl_u, gl_u)
                                    };
                                    ua_a[a_flat] += w_upper * ax;
                                    ua_b[b_flat] += w_upper * ay;
                                    us_a[a_flat] += (w_upper as f64).abs() * (ax as f64).abs();
                                    us_b[b_flat] += (w_upper as f64).abs() * (ay as f64).abs();
                                    const_upper.add_term(w_upper, c);
                                }
                            }
                        }
                    }
                }
            }
            for (r, (const_lower, const_upper)) in folds.iter().enumerate() {
                chunk.lower_b_total[r] = const_lower.finish();
                chunk.upper_b_total[r] = const_upper.finish();
            }
        });
        let published = buffers.publish(shape.n, shape.m)?;
        crate::layers::collector_kernels::record_bilinear_rowpar(
            true,
            num_outputs,
            parallel,
            started.elapsed(),
        );
        Ok(Some(published))
    }

    /// #vit-batched-bilinear (ENGINE_PROGRAM V1): the GEMM-shaped twin of the
    /// serial alpha nest in [`Self::propagate_linear_binary_with_alpha_kernels`]
    /// — BIT-IDENTICAL to it.
    ///
    /// The argument is the dense arm's (`matmul/crown_gemm.rs` module docs),
    /// with this arm's differences, each exactly the nest's:
    ///
    /// * the table holds, per element, the two `interpolated_mccormick`
    ///   results of its CLAMPED parameters (`alphas[0]`/`alphas[2]` for the
    ///   lower computation, `alphas[1]`/`alphas[3]` for the upper) — a pure
    ///   function of `(lx, ux, ly, uy, r_l, r_u)`, none of which depends on the
    ///   row — laid out in selection order: lower computation with `w >= 0`
    ///   uses the LOWER plane, with `w < 0` the UPPER plane; upper computation
    ///   with `w >= 0` the UPPER plane, with `w < 0` the LOWER plane;
    /// * an element whose operand box is non-finite is marked non-finite: the
    ///   kernel skips its coefficient update and forces every nonzero-weight
    ///   constant fold trivial at that point of the row's sequence (the nest's
    ///   `force_trivial` + `continue`);
    /// * the constant folds are finished by THIS file's
    ///   `widen_mccormick_constant` (what `McCormickConstFold::finish` calls).
    ///
    /// `schedule == None` is the production schedule; tests pass explicit
    /// ones. Returns `Ok(None)` (the caller's next choice then runs) for a
    /// shape it does not handle.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propagate_linear_binary_with_alpha_gemm(
        &self,
        bounds: &crate::LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        alphas: &ndarray::Array4<f32>,
        shape: &McCormickRowparShape,
        scale: f32,
        schedule: Option<GemmSchedule>,
    ) -> Result<Option<(crate::LinearBounds, crate::LinearBounds)>> {
        let started = std::time::Instant::now();
        if alphas.shape() != [4, shape.m, shape.n, shape.k] {
            return Ok(None);
        }
        let Some(operands) =
            McCormickOperands::gather(bounds, input_a_bounds, input_b_bounds, shape)
        else {
            return Ok(None);
        };
        let num_outputs = bounds.num_outputs();
        let schedule = schedule.unwrap_or_else(|| GemmSchedule::production(num_outputs));
        let (a_lo, a_hi) = (&*operands.a_lo, &*operands.a_hi);
        let (b_lo, b_hi) = (&*operands.b_lo, &*operands.b_hi);
        let Some(table) = McCormickPlaneTable::build(shape, schedule.parallel, |batch, i, j, l| {
            let a_flat = shape.a_flat(batch, i, l);
            let b_flat = shape.b_flat(batch, j, l);
            let (lx, ux) = (a_lo[a_flat], a_hi[a_flat]);
            let (ly, uy) = (b_lo[b_flat], b_hi[b_flat]);
            if !lx.is_finite() || !ux.is_finite() || !ly.is_finite() || !uy.is_finite() {
                return None;
            }
            let r_l_lower = alphas[[0, i, j, l]].clamp(0.0, 1.0);
            let r_u_lower = alphas[[2, i, j, l]].clamp(0.0, 1.0);
            let r_l_upper = alphas[[1, i, j, l]].clamp(0.0, 1.0);
            let r_u_upper = alphas[[3, i, j, l]].clamp(0.0, 1.0);
            let (al_l, bl_l, gl_l, au_l, bu_l, gu_l) =
                interpolated_mccormick(lx, ux, ly, uy, r_l_lower, r_u_lower);
            let (al_u, bl_u, gl_u, au_u, bu_u, gu_u) =
                interpolated_mccormick(lx, ux, ly, uy, r_l_upper, r_u_upper);
            // Selection order: [lower w>=0, lower w<0, upper w>=0, upper w<0].
            Some([
                (al_l, bl_l, gl_l),
                (au_l, bu_l, gu_l),
                (au_u, bu_u, gu_u),
                (al_u, bl_u, gl_u),
            ])
        }) else {
            return Ok(None);
        };
        let inputs = operands.inputs(shape, &table, scale);
        let Some((published, parallel)) =
            mccormick_gemm_backward(&inputs, num_outputs, schedule, widen_mccormick_constant)?
        else {
            return Ok(None);
        };
        crate::layers::collector_kernels::record_bilinear_gemm(
            true,
            num_outputs,
            parallel,
            started.elapsed(),
            table.build_elapsed,
        );
        Ok(Some(published))
    }
}

impl MatMulLayer {
    /// #vit-margin-alpha (B4): CROWN backward of a var×var `MatMul` with
    /// OPTIONAL interpolated-McCormick parameters `[4, m, n, k]`.
    ///
    /// `None` is the historical midpoint arm, byte-identical
    /// ([`Self::propagate_linear_binary`]). `Some` routes through the SAME
    /// certified interpolated arm `BilinearCrown` uses
    /// ([`BilinearCrownLayer::propagate_linear_binary_with_alpha`]): identical
    /// shape contract (`parse_matmul_dims`), identical Higham coefficient
    /// certificate, identical `McCormickConstFold` envelope-constant
    /// certificate, and every parameter clamped into `[0, 1]` before
    /// `interpolated_mccormick` builds its outward-certified plane. Only the
    /// certified margin ascent (`propagate_dag/margin_alpha.rs`) ever supplies
    /// parameters for a `MatMul` node; every other walk passes `None`.
    ///
    /// Kept as a method so the shared backward dispatcher's match arm names
    /// no second layer type (the dispatch-coverage parser counts every
    /// `Layer`-path spelling inside that match).
    ///
    /// # Errors
    ///
    /// A non-finite `scale` (refused by `BilinearCrownLayer::try_new`), and
    /// every error of the two delegated arms.
    pub(crate) fn propagate_linear_binary_with_interpolation(
        &self,
        bounds: &crate::LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        alphas: Option<&ndarray::Array4<f32>>,
    ) -> Result<(crate::LinearBounds, crate::LinearBounds)> {
        match alphas {
            None => self.propagate_linear_binary(bounds, input_a_bounds, input_b_bounds),
            Some(alphas) => BilinearCrownLayer::try_new(self.transpose_b, self.scale)?
                .propagate_linear_binary_with_alpha(
                    bounds,
                    input_a_bounds,
                    input_b_bounds,
                    Some(alphas),
                ),
        }
    }
}

#[cfg(test)]
mod coefficient_certificate_tests {
    use super::*;
    use ndarray::{Array4, ArrayD, IxDyn};

    fn bt(shape: &[usize], lower: &[f32], upper: &[f32]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    /// COUNTEREXAMPLE for the alpha-parameterized BilinearCrown arm publishing
    /// uncertified coefficients (the `alphas == None` arm delegates to
    /// `crown_dense.rs`, which DOES certify; the two must agree).
    ///
    /// Construction, chosen so every intermediate is exact and hand-checkable:
    /// * `A` is `[1, 1]` with `x ∈ [1, 1000]`; `B` is `[1, 3]` with the three
    ///   entries pinned to the POINTS `y = [-2^24, -1, +2^24]`.
    /// * `alphas = Array4::ones([4, 1, 3, 1])` — exactly what the DAG α-CROWN
    ///   loop installs for every BilinearCrown node
    ///   (`graph_alpha/propagate_dag/init.rs`), so this is the production arm.
    /// * With `y_l == y_u`, `interpolated_mccormick` returns
    ///   `alpha = y_l`, `beta = x_l = 1`, `ny = -y_l·x_l` for BOTH directions and
    ///   any `r`, and `certify_plane_constant` finds zero corner violation.
    /// * incoming spec weights are all `1`.
    ///
    /// The A-side coefficient cell therefore accumulates
    /// `fl(fl(0 + (-2^24)) + (-1)) + 2^24 = 0`, while the exact coefficient is
    /// `-2^24 - 1 + 2^24 = -1`. The op's true value is
    /// `Σ_j 1·(x·y_j) = -x ∈ [-1000, -1]`, minimum `-1000`. Published with no
    /// error matrix the relation concretizes to about `-3` — it EXCLUDES the true
    /// minimum by ~997, i.e. a false `unsat` is available from it. Post-fix the
    /// `γ^{f32}_{dims.n+2} · S` charge (`S = 2^24 + 1 + 2^24`, `dims.n = 3` ⇒
    /// `err ≈ 10.0`) discharges as `10.0 · 1000 = 10000` and the bound encloses.
    #[test]
    fn alpha_arm_cancelling_coefficient_stays_enclosed() {
        const P24: f32 = 16_777_216.0; // 2^24

        let layer = BilinearCrownLayer::new(false, None);
        let a = bt(&[1, 1], &[1.0], &[1000.0]);
        let y = [-P24, -1.0f32, P24];
        let b = bt(&[1, 3], &y, &y);

        let spec = crate::LinearBounds::new(
            Array2::<f32>::ones((1, 3)),
            Array1::zeros(1),
            Array2::<f32>::ones((1, 3)),
            Array1::zeros(1),
        )
        .unwrap();
        let alphas = Array4::<f32>::ones((4, 1, 3, 1));

        let (bounds_a, bounds_b) = layer
            .propagate_linear_binary_with_alpha(&spec, &a, &b, Some(&alphas))
            .unwrap();

        // Precondition of the counterexample: the f32 += chain really did cancel.
        assert_eq!(
            bounds_a.lower_a()[[0, 0]],
            0.0,
            "counterexample precondition: the f32 += chain must cancel to 0"
        );

        // The alpha arm must publish the certificate its alpha-free sibling does.
        let err_lo = bounds_a
            .lower_a_err()
            .expect("BilinearCrown alpha arm must publish a coefficient certificate");
        assert!(
            f64::from(err_lo[[0, 0]]) >= 1.0,
            "certificate {} does not cover |stored 0 - exact -1| = 1",
            err_lo[[0, 0]]
        );
        assert!(
            bounds_b.lower_a_err().is_some() && bounds_b.upper_a_err().is_some(),
            "BilinearCrown alpha arm published bounds_b without a certificate"
        );

        // End-to-end: the published relation must lower-bound the true minimum.
        // The bias is split half onto each side, so summing the two
        // concretizations reproduces exactly what `backward_dispatch` composes.
        let lo =
            bounds_a.concretize_sound(&a).lower()[0] + bounds_b.concretize_sound(&b).lower()[0];
        assert!(
            lo <= -1000.0,
            "published CROWN lower bound {lo} is ABOVE the true minimum -1000 of \
             sum_j (x * y_j) = -x over x in [1, 1000]; the alpha arm's f32 \
             coefficient accumulation error is uncertified"
        );
    }

    /// The alpha arm and its `alphas == None` sibling must make the same KIND of
    /// claim about their coefficients: one carrying a certificate and the other
    /// asserting exactness is the defect, independent of any numeric gap.
    #[test]
    fn alpha_and_alpha_free_arms_both_certify() {
        let layer = BilinearCrownLayer::new(false, None);
        let a = bt(&[1, 2], &[-1.0, 0.5], &[2.0, 3.0]);
        let b = bt(&[2, 2], &[-2.0, 0.0, 1.0, -1.0], &[1.0, 2.0, 4.0, 0.5]);

        let spec = crate::LinearBounds::new(
            Array2::<f32>::ones((1, 2)),
            Array1::zeros(1),
            Array2::<f32>::ones((1, 2)),
            Array1::zeros(1),
        )
        .unwrap();

        let (free_a, free_b) = layer
            .propagate_linear_binary_with_alpha(&spec, &a, &b, None)
            .unwrap();
        let alphas = Array4::<f32>::ones((4, 1, 2, 2));
        let (alpha_a, alpha_b) = layer
            .propagate_linear_binary_with_alpha(&spec, &a, &b, Some(&alphas))
            .unwrap();

        for (name, bounds) in [
            ("alpha-free A", &free_a),
            ("alpha-free B", &free_b),
            ("alpha A", &alpha_a),
            ("alpha B", &alpha_b),
        ] {
            assert!(
                bounds.lower_a_err().is_some() && bounds.upper_a_err().is_some(),
                "{name} published without a coefficient certificate"
            );
        }
    }

    /// COUNTEREXAMPLE for the CONSTANT half of the alpha arm
    /// (#bilinear-alpha-mccormick-const). The session that added the coefficient
    /// certificate above left `const += w as f64 * c as f64` folding in plain
    /// round-to-nearest f64, published through `next_down_f32((v * 0.5) as f32)`
    /// alone — while its own `alphas == None` sibling, which delegates to
    /// `MatMulLayer::propagate_linear_binary`, DID get the constant certificate in
    /// the same pass. This asserts the two arms now make the same claim.
    ///
    /// # Construction
    ///
    /// `m = n = 1, k = 4097`, so a single incoming weight `w = 1` drives one
    /// `add_term` per `l`, in `l` order — the classic Higham cancellation:
    ///
    /// * every box is SYMMETRIC (`x ∈ [-A, A]`, `y ∈ [-B, B]`) and every alpha is
    ///   `0.5`, which makes `interpolated_mccormick` return `alpha = beta = 0`
    ///   exactly, so ALL plane coefficients are zero. `S = 0`, the coefficient
    ///   certificate is identically zero, and it therefore cannot be what saves
    ///   the bound — the constant is on its own.
    /// * the incoming bias is `+2^55` (an exact f32, carrying no relaxation
    ///   slack); the first 4096 columns are the unit box, whose lower plane
    ///   constant is `≈ -1`, small enough (`ulp(2^55) = 4`) to be swallowed WHOLE
    ///   by the running sum; column 4096 is the box `A = 2^27, B = 2^28`, whose
    ///   lower plane constant is `≈ -2^55` and cancels the bias.
    ///
    /// The fold therefore lands near zero having silently dropped ~4096, and the
    /// f32 ULP the directed cast allows at that magnitude is only ~512. The
    /// `γ^{f64}_{adds+1} · Σ|summand|` charge scales with the MAGNITUDE sum
    /// (`≈ 2^56`), not with the cancelled result, so it covers the drop with room
    /// to spare.
    ///
    /// Both directions are asserted in the same call — the LOWER constant must
    /// end up at or below the exact fold and the UPPER at or above it — because
    /// certifying one direction and not the other is the same half-a-relaxation
    /// defect this test exists to close.
    ///
    /// This asserts the CERTIFICATE'S OWN CONTRACT (published constant vs. the
    /// exact sum of the terms actually folded) rather than an end-to-end false
    /// `unsat`. That is deliberate and is the strongest available claim here: in
    /// McCormick geometry `|c| = |α|·|β|`, so a plane constant large enough to
    /// drive an f64 accumulation deficit necessarily carries its own f32 ULP of
    /// `certify_plane_constant` slack, ~2^29 times larger than the deficit. The
    /// relaxation's looseness masks the defect end-to-end; the contract does not.
    #[test]
    fn alpha_arm_cancelling_constant_is_certified_outward() {
        const BIAS: f32 = 36_028_797_018_963_968.0; // 2^55
        const P27: f32 = 134_217_728.0; // 2^27
        const P28: f32 = 268_435_456.0; // 2^28
        const N: usize = 4096;
        let k = N + 1;

        // Read the ACTUAL plane constants the driver will fold, from the same
        // entry point at the same interpolation parameters.
        let (al_s, bl_s, c_small_lo, au_s, bu_s, c_small_hi) =
            interpolated_mccormick(-1.0, 1.0, -1.0, 1.0, 0.5, 0.5);
        let (al_b, bl_b, c_big_lo, au_b, bu_b, c_big_hi) =
            interpolated_mccormick(-P27, P27, -P28, P28, 0.5, 0.5);
        for (name, coeff) in [
            ("al_s", al_s),
            ("bl_s", bl_s),
            ("au_s", au_s),
            ("bu_s", bu_s),
            ("al_b", al_b),
            ("bl_b", bl_b),
            ("au_b", au_b),
            ("bu_b", bu_b),
        ] {
            assert_eq!(
                coeff, 0.0,
                "precondition: symmetric box at r = 0.5 must give a zero {name}, \
                 so the COEFFICIENT certificate is identically zero here"
            );
        }

        // Preconditions, under the same f64 semantics the driver's fold uses.
        let mut naive_lo = f64::from(BIAS);
        let mut naive_hi = -f64::from(BIAS);
        for _ in 0..N {
            naive_lo += f64::from(c_small_lo);
            naive_hi += f64::from(c_small_hi);
        }
        naive_lo += f64::from(c_big_lo);
        naive_hi += f64::from(c_big_hi);
        // Grouped so that each addition is exact: (bias + big) is a difference of
        // f32-exact quantities, and N is a power of two.
        let exact_lo = (f64::from(BIAS) + f64::from(c_big_lo)) + (N as f64) * f64::from(c_small_lo);
        let exact_hi =
            (-f64::from(BIAS) + f64::from(c_big_hi)) + (N as f64) * f64::from(c_small_hi);
        assert!(
            naive_lo - exact_lo > 4000.0,
            "precondition: the sequential f64 lower fold must have dropped ~4096 \
             (naive {naive_lo}, exact {exact_lo})"
        );
        assert!(
            exact_hi - naive_hi > 4000.0,
            "precondition: the sequential f64 upper fold must have dropped ~4096 \
             (naive {naive_hi}, exact {exact_hi})"
        );

        let mut a_lo = vec![-1.0f32; k];
        let mut a_hi = vec![1.0f32; k];
        a_lo[k - 1] = -P27;
        a_hi[k - 1] = P27;
        let mut b_lo = vec![-1.0f32; k];
        let mut b_hi = vec![1.0f32; k];
        b_lo[k - 1] = -P28;
        b_hi[k - 1] = P28;
        let a = bt(&[1, k], &a_lo, &a_hi);
        let b = bt(&[k, 1], &b_lo, &b_hi);

        let layer = BilinearCrownLayer::new(false, None);
        let spec = crate::LinearBounds::new(
            Array2::<f32>::ones((1, 1)),
            Array1::from_elem(1, BIAS),
            Array2::<f32>::ones((1, 1)),
            Array1::from_elem(1, -BIAS),
        )
        .unwrap();
        let alphas = Array4::<f32>::from_elem((4, 1, 1, k), 0.5);

        let (bounds_a, bounds_b) = layer
            .propagate_linear_binary_with_alpha(&spec, &a, &b, Some(&alphas))
            .unwrap();

        // Precondition: every published coefficient really is zero.
        for l in 0..k {
            assert_eq!(bounds_a.lower_a()[[0, l]], 0.0);
            assert_eq!(bounds_a.upper_a()[[0, l]], 0.0);
            assert_eq!(bounds_b.lower_a()[[0, l]], 0.0);
            assert_eq!(bounds_b.upper_a()[[0, l]], 0.0);
        }

        // The constant is HALVED onto each side, so the two biases sum back to the
        // published total — exactly what `backward_dispatch` composes.
        let published_lo = f64::from(bounds_a.lower_b()[0]) + f64::from(bounds_b.lower_b()[0]);
        let published_hi = f64::from(bounds_a.upper_b()[0]) + f64::from(bounds_b.upper_b()[0]);
        assert!(
            published_lo <= exact_lo,
            "published lower constant {published_lo} is ABOVE the exact fold \
             {exact_lo}; the alpha arm's McCormick-CONSTANT accumulation error is \
             uncertified (the naive f64 fold was {naive_lo}, and the directed f32 \
             cast allows only one ULP of THAT)"
        );
        assert!(
            published_hi >= exact_hi,
            "published upper constant {published_hi} is BELOW the exact fold \
             {exact_hi}; certifying only the LOWER constant is the same \
             half-a-relaxation defect (the naive f64 fold was {naive_hi})"
        );
    }
}

/// #vit-margin-alpha (B4) soundness pins for the interpolated McCormick planes
/// the margin ascent optimizes: the planes themselves (exact rational check at
/// corners and interior points), and the `MatMul` α arm end to end.
#[cfg(test)]
mod matmul_interpolation_containment_tests {
    use super::*;
    use crate::LinearBounds;
    use ndarray::{Array4, ArrayD, Axis, IxDyn};
    use num_rational::BigRational;

    /// SplitMix64: deterministic, dependency-free, and bit-reproducible.
    struct SplitMix(u64);

    impl SplitMix {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in `[0, 1)`, exactly representable (24 random bits).
        fn unit(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1_u64 << 24) as f32
        }

        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }

        fn pick(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    /// Exact value of an f32 (every finite f32 is a dyadic rational).
    fn rat(value: f32) -> BigRational {
        BigRational::from_float(f64::from(value)).expect("finite f32 is an exact rational")
    }

    /// A point of `[lo, hi]` as an f32 that is guaranteed to lie inside it.
    fn inside(lo: f32, hi: f32, t: f32) -> f32 {
        (lo + (hi - lo) * t).clamp(lo, hi)
    }

    /// One random endpoint pair, drawn from the shapes that stress the plane
    /// certificate: mixed sign, a tiny endpoint beside a large one (the
    /// cancellation shape of `(y_u·x_u − y_l·x_l)·r − y_u·x_u`), a point
    /// interval, and a same-sign interval far from zero.
    fn interval(rng: &mut SplitMix) -> (f32, f32) {
        let (a, b) = match rng.pick(5) {
            0 => (rng.range(-4.0, 4.0), rng.range(-4.0, 4.0)),
            1 => {
                let tiny = rng.range(-1.0, 1.0) * 10_f32.powi(-(rng.pick(12) as i32) - 4);
                (tiny, rng.range(0.5, 1000.0))
            }
            2 => {
                let v = rng.range(-3.0, 3.0);
                (v, v)
            }
            3 => {
                let base = rng.range(10.0, 1000.0);
                (base, base + rng.range(0.0, 50.0))
            }
            _ => {
                let scale = 10_f32.powi(rng.pick(7) as i32 - 3);
                (rng.range(-1.0, 1.0) * scale, rng.range(-1.0, 1.0) * scale)
            }
        };
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    }

    /// Exact `z = C · (scale · A @ B)` for one row, flat row-major operands.
    #[allow(clippy::too_many_arguments)]
    fn exact_row_value(
        row: &[f32],
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
        scale: f32,
    ) -> BigRational {
        let mut total = rat(0.0);
        for bi in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let c = row[bi * m * n + i * n + j];
                    if c == 0.0 {
                        continue;
                    }
                    let mut dot = rat(0.0);
                    for l in 0..k {
                        let av = a[bi * m * k + i * k + l];
                        let bv = if transpose_b {
                            b[bi * n * k + j * k + l]
                        } else {
                            b[bi * k * n + l * n + j]
                        };
                        dot += rat(av) * rat(bv);
                    }
                    total += rat(c) * rat(scale) * dot;
                }
            }
        }
        total
    }

    fn bits(
        values: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, impl ndarray::Dimension>,
    ) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// B4 SOUNDNESS. (1) Every interpolated McCormick plane pair
    /// `interpolated_mccormick` returns for `r ∈ [0, 1]` — f32-rounded
    /// coefficients, outward-certified constant — contains the EXACT product
    /// at all four box corners and at interior points, checked in exact
    /// rational arithmetic (no floating tolerance anywhere). (2) The var×var
    /// `MatMul` α arm (`propagate_linear_binary_with_interpolation`, the only
    /// route the margin ascent uses for a `MatMul` node) publishes linear
    /// bounds whose outward concretization at sampled operand points contains
    /// the exact `C·(s·A@B)`, for parameters drawn from `[-0.25, 1.25]` (the
    /// arm must clamp them), with batch dims and both `transpose_b`
    /// settings; the `None` arm is bit-identical to the historical midpoint
    /// arm, and the `Some` arm to the `BilinearCrown` arm it delegates to.
    #[test]
    fn matmul_interpolated_mccormick_alpha_contains_product() {
        let mut rng = SplitMix(0x5EED_0B4E_A1FA_2026);

        // (1) Plane level.
        let mut planes_checked = 0_usize;
        for case in 0..3000 {
            let (x_l, x_u) = interval(&mut rng);
            let (y_l, y_u) = interval(&mut rng);
            let (r_l, r_u) = match case % 4 {
                0 => (0.0, 0.0),
                1 => (1.0, 1.0),
                _ => (rng.unit(), rng.unit()),
            };
            let (al, bl, cl, au, bu, cu) = interpolated_mccormick(x_l, x_u, y_l, y_u, r_l, r_u);
            let mut points = vec![(x_l, y_l), (x_l, y_u), (x_u, y_l), (x_u, y_u)];
            for _ in 0..4 {
                let tx = rng.unit();
                let ty = rng.unit();
                points.push((inside(x_l, x_u, tx), inside(y_l, y_u, ty)));
            }
            for (x, y) in points {
                let product = rat(x) * rat(y);
                let lower = rat(al) * rat(x) + rat(bl) * rat(y) + rat(cl);
                let upper = rat(au) * rat(x) + rat(bu) * rat(y) + rat(cu);
                assert!(
                    lower <= product,
                    "LOWER plane above x*y: box x=[{x_l:e},{x_u:e}] y=[{y_l:e},{y_u:e}] \
                     r_l={r_l} at ({x:e},{y:e}); plane=({al:e},{bl:e},{cl:e}) — a false-proof \
                     mechanism"
                );
                assert!(
                    upper >= product,
                    "UPPER plane below x*y: box x=[{x_l:e},{x_u:e}] y=[{y_l:e},{y_u:e}] \
                     r_u={r_u} at ({x:e},{y:e}); plane=({au:e},{bu:e},{cu:e})"
                );
                planes_checked += 1;
            }
        }
        assert_eq!(planes_checked, 3000 * 8);

        // (2) Layer level: the MatMul α arm.
        //
        // WHAT IS CHECKED, AND WHY NOT PER OPERAND (#concretize-vacuity). The arm
        // returns TWO relations — one over `A`, one over `B` — with the McCormick
        // constant HALVED onto each. Neither half encloses anything on its own; only
        // their SUM does (`lower_A(a) + lower_B(b) <= C·(s·A@B) <= upper_A(a) +
        // upper_B(b)`). Evaluated at a point, one half's "lower" routinely sits ABOVE
        // its own "upper" (see
        // `matmul_operand_half_inverts_alone_and_only_the_summed_relation_encloses`),
        // and `concretize_sound`, which publishes ONE interval per row, repairs any
        // inverted row to `[-inf, +inf]` (`f64_to_bounded_tensor_with_deadline`,
        // `bounds/concretize.rs`) — the correct fail-closed for its contract. The
        // first version of this test concretized each half separately, so 992 of its
        // 1440 α-arm rows (and 1117 None-arm rows) came back vacuous with every
        // coefficient, bias and error finite (int/wave3 v4 gate). Never BOTH halves of
        // one row: that would invert the sum, i.e. break the enclosure.
        //
        // Production never concretizes a half: `backward_dispatch` moves both halved
        // constants into ONE bias channel, and each half's coefficient block is
        // accumulated at its operand and chained back to the network input, where
        // the contributions are SUMMED before the single concretization. So this test
        // checks what production relies on — the SUMMED relation over `[A; B]` —
        // twice: (i) evaluated EXACTLY at the point (no floating rounding anywhere,
        // certified error charged `∓ err·|p|` like `concretize_sound` does), and
        // (ii) through the production concretizer, which must return a FINITE row (a
        // valid enclosure evaluated at a point cannot invert, so a vacuous row there
        // would be a violation hidden behind the repair) containing the exact value.
        // Both arms are held to it, with no vacuity allowance.
        let (batch, m, n, k) = (2_usize, 3_usize, 2_usize, 4_usize);
        let rows = 3_usize;
        let mut points_checked = 0_usize;
        // [arm][operand]: rows where that half ALONE is inverted at the point.
        let mut half_inversions = [[0_usize; 2]; 2];
        for case in 0..48 {
            let transpose_b = case % 2 == 1;
            let scale = if case % 3 == 0 { None } else { Some(0.375_f32) };
            let a_shape = [batch, m, k];
            let b_shape = if transpose_b {
                [batch, n, k]
            } else {
                [batch, k, n]
            };
            let draw_box = |rng: &mut SplitMix, len: usize| -> (Vec<f32>, Vec<f32>) {
                (0..len).map(|_| interval(rng)).unzip()
            };
            let (a_lo, a_hi) = draw_box(&mut rng, batch * m * k);
            let (b_lo, b_hi) = draw_box(&mut rng, batch * k * n);
            let a_box = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&a_shape), a_lo.clone()).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&a_shape), a_hi.clone()).unwrap(),
            )
            .unwrap();
            let b_box = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&b_shape), b_lo.clone()).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&b_shape), b_hi.clone()).unwrap(),
            )
            .unwrap();
            let total_c = batch * m * n;
            let spec_values: Vec<f32> = (0..rows * total_c)
                .map(|_| match rng.pick(4) {
                    0 => 0.0,
                    _ => rng.range(-2.0, 2.0),
                })
                .collect();
            let spec = LinearBounds::from_spec_matrix(
                Array2::from_shape_vec((rows, total_c), spec_values.clone()).unwrap(),
            )
            .unwrap();
            let alphas = Array4::from_shape_fn((4, m, n, k), |_| match rng.pick(6) {
                0 => 0.0,
                1 => 1.0,
                _ => rng.range(-0.25, 1.25),
            });

            let matmul = MatMulLayer::new(transpose_b, scale);
            // `None` is the historical arm, bit for bit.
            let (none_a, none_b) = matmul
                .propagate_linear_binary_with_interpolation(&spec, &a_box, &b_box, None)
                .unwrap();
            let (hist_a, hist_b) = matmul
                .propagate_linear_binary(&spec, &a_box, &b_box)
                .unwrap();
            for (got, want) in [(&none_a, &hist_a), (&none_b, &hist_b)] {
                assert_eq!(bits(got.lower_a()), bits(want.lower_a()));
                assert_eq!(bits(got.upper_a()), bits(want.upper_a()));
                assert_eq!(bits(got.lower_b()), bits(want.lower_b()));
                assert_eq!(bits(got.upper_b()), bits(want.upper_b()));
            }
            // `Some` is the BilinearCrown α arm, bit for bit.
            let (la, lb) = matmul
                .propagate_linear_binary_with_interpolation(&spec, &a_box, &b_box, Some(&alphas))
                .unwrap();
            let (ba, bb) = BilinearCrownLayer::new(transpose_b, scale)
                .propagate_linear_binary_with_alpha(&spec, &a_box, &b_box, Some(&alphas))
                .unwrap();
            for (got, want) in [(&la, &ba), (&lb, &bb)] {
                assert_eq!(bits(got.lower_a()), bits(want.lower_a()));
                assert_eq!(bits(got.upper_a()), bits(want.upper_a()));
                assert_eq!(bits(got.lower_b()), bits(want.lower_b()));
                assert_eq!(bits(got.upper_b()), bits(want.upper_b()));
            }

            let arms = [
                ("alpha", operand_pair_relation(&la, &lb), &la, &lb),
                (
                    "none",
                    operand_pair_relation(&none_a, &none_b),
                    &none_a,
                    &none_b,
                ),
            ];

            // Containment at vertices (each coordinate at an endpoint) and at
            // interior points of both operand boxes.
            let effective_scale = scale.unwrap_or(1.0);
            for sample in 0..10 {
                let vertex = sample < 5;
                let mut draw_point = |lo: &[f32], hi: &[f32]| -> Vec<f32> {
                    lo.iter()
                        .zip(hi)
                        .map(|(&l, &h)| {
                            if vertex {
                                if rng.pick(2) == 0 {
                                    l
                                } else {
                                    h
                                }
                            } else {
                                inside(l, h, rng.unit())
                            }
                        })
                        .collect()
                };
                let a_pt = draw_point(&a_lo, &a_hi);
                let b_pt = draw_point(&b_lo, &b_hi);
                // The operand point laid out as the summed relation's input: `[A; B]`.
                let joint: Vec<f32> = a_pt.iter().chain(&b_pt).copied().collect();
                let joint_point = BoundedTensor::concrete(
                    ArrayD::from_shape_vec(IxDyn(&[joint.len()]), joint.clone()).unwrap(),
                )
                .unwrap();
                let published: Vec<BoundedTensor> = arms
                    .iter()
                    .map(|(_, whole, _, _)| whole.concretize_sound(&joint_point))
                    .collect();
                for r in 0..rows {
                    let exact = exact_row_value(
                        &spec_values[r * total_c..(r + 1) * total_c],
                        &a_pt,
                        &b_pt,
                        batch,
                        m,
                        n,
                        k,
                        transpose_b,
                        effective_scale,
                    );
                    for (arm_idx, (arm, whole, half_a, half_b)) in arms.iter().enumerate() {
                        // (i) The relation itself, exactly: the soundness claim.
                        let (rel_lo, rel_hi) = relation_at_point(whole, r, &joint);
                        assert!(
                            rel_lo <= exact,
                            "case {case} sample {sample} row {r} ({arm} arm): the summed \
                             MatMul relation's LOWER side exceeds the exact product value \
                             (transpose_b={transpose_b}, scale={scale:?}) — a false-proof \
                             mechanism"
                        );
                        assert!(
                            rel_hi >= exact,
                            "case {case} sample {sample} row {r} ({arm} arm): the summed \
                             MatMul relation's UPPER side is below the exact product value \
                             (transpose_b={transpose_b}, scale={scale:?})"
                        );

                        // Observable only: which half, ALONE, is inverted here.
                        let (a_lo_v, a_hi_v) = relation_at_point(half_a, r, &a_pt);
                        let (b_lo_v, b_hi_v) = relation_at_point(half_b, r, &b_pt);
                        let inverted_a = a_lo_v > a_hi_v;
                        let inverted_b = b_lo_v > b_hi_v;
                        assert!(
                            !(inverted_a && inverted_b),
                            "case {case} sample {sample} row {r} ({arm} arm): BOTH operand \
                             halves are inverted, so their sum is — the relation is not an \
                             enclosure"
                        );
                        half_inversions[arm_idx][0] += usize::from(inverted_a);
                        half_inversions[arm_idx][1] += usize::from(inverted_b);

                        // (ii) The production concretizer on the summed relation.
                        let lo = published[arm_idx].lower()[r];
                        let hi = published[arm_idx].upper()[r];
                        assert!(
                            lo.is_finite() && hi.is_finite(),
                            "case {case} sample {sample} row {r} ({arm} arm): the SUMMED \
                             relation concretized to [{lo:e}, {hi:e}] at a finite point \
                             (transpose_b={transpose_b}, scale={scale:?}); an enclosure \
                             evaluated at a point cannot invert, so this vacuity hides a \
                             violation behind the inversion repair"
                        );
                        assert!(
                            rat(lo) <= exact,
                            "case {case} sample {sample} row {r} ({arm} arm): certified \
                             MatMul LOWER bound exceeds the exact product value \
                             (transpose_b={transpose_b}, scale={scale:?}) — a false-proof \
                             mechanism"
                        );
                        assert!(
                            rat(hi) >= exact,
                            "case {case} sample {sample} row {r} ({arm} arm): certified \
                             MatMul UPPER bound is below the exact product value \
                             (transpose_b={transpose_b}, scale={scale:?})"
                        );
                    }
                    points_checked += 1;
                }
            }
        }
        assert_eq!(points_checked, 48 * 10 * rows);
        eprintln!(
            "[matmul-alpha-test] summed-relation rows checked: {points_checked} per arm, all \
             finite and containing; operand halves inverted ALONE (not a defect — a half is \
             not an enclosure): alpha A={} B={}, none A={} B={}",
            half_inversions[0][0],
            half_inversions[0][1],
            half_inversions[1][0],
            half_inversions[1][1],
        );
    }

    /// The relation production actually concretizes for one var×var product,
    /// over the concatenated operand vector `[A; B]`: both halves' coefficient
    /// blocks side by side and their constants SUMMED — exactly what
    /// `backward_dispatch` composes (`lb_a.lower_b() + lb_b.lower_b()` into one
    /// bias channel) before the accumulator adds the blocks at the network input.
    ///
    /// Both halves carry the SAME halved constant (one `lower_b_half` cloned onto
    /// each side), which is what makes that f32 addition exact (`h + h = 2h`, a
    /// power-of-two scaling); it is pinned here rather than assumed.
    fn operand_pair_relation(half_a: &LinearBounds, half_b: &LinearBounds) -> LinearBounds {
        assert_eq!(
            bits(half_a.lower_b()),
            bits(half_b.lower_b()),
            "the two halves must carry the identical halved LOWER constant"
        );
        assert_eq!(
            bits(half_a.upper_b()),
            bits(half_b.upper_b()),
            "the two halves must carry the identical halved UPPER constant"
        );
        let side_by_side = |left: &Array2<f32>, right: &Array2<f32>| -> Array2<f32> {
            ndarray::concatenate(Axis(1), &[left.view(), right.view()])
                .expect("both halves have one row per output")
        };
        let err_or_zero = |err: Option<&Array2<f32>>, like: &Array2<f32>| -> Array2<f32> {
            err.cloned()
                .unwrap_or_else(|| Array2::zeros(like.raw_dim()))
        };
        let mut whole = LinearBounds::new(
            side_by_side(half_a.lower_a(), half_b.lower_a()),
            half_a.lower_b() + half_b.lower_b(),
            side_by_side(half_a.upper_a(), half_b.upper_a()),
            half_a.upper_b() + half_b.upper_b(),
        )
        .expect("finite halves compose into a finite relation");
        whole.set_coeff_err(
            side_by_side(
                &err_or_zero(half_a.lower_a_err(), half_a.lower_a()),
                &err_or_zero(half_b.lower_a_err(), half_b.lower_a()),
            ),
            side_by_side(
                &err_or_zero(half_a.upper_a_err(), half_a.upper_a()),
                &err_or_zero(half_b.upper_a_err(), half_b.upper_a()),
            ),
        );
        whole
    }

    /// One row of a relation evaluated EXACTLY at a point: `(lower, upper)`,
    /// with the certified coefficient error charged outward (`∓ err·|p|`, the
    /// charge `concretize_sound` applies at a degenerate box) and no floating
    /// rounding anywhere.
    fn relation_at_point(
        relation: &LinearBounds,
        row: usize,
        point: &[f32],
    ) -> (BigRational, BigRational) {
        assert_eq!(relation.num_inputs(), point.len(), "relation / point width");
        let mut lower = rat(relation.lower_b()[row]);
        let mut upper = rat(relation.upper_b()[row]);
        for (j, &p) in point.iter().enumerate() {
            let value = rat(p);
            let magnitude = rat(p.abs());
            lower += rat(relation.lower_a()[[row, j]]) * value.clone();
            upper += rat(relation.upper_a()[[row, j]]) * value;
            if let Some(err) = relation.lower_a_err() {
                lower -= rat(err[[row, j]]) * magnitude.clone();
            }
            if let Some(err) = relation.upper_a_err() {
                upper += rat(err[[row, j]]) * magnitude;
            }
        }
        (lower, upper)
    }

    /// #concretize-vacuity: WHY concretizing one operand half of the var×var
    /// McCormick split at a point comes back `[-inf, +inf]`, pinned on a
    /// hand-checkable box so the diagnosis cannot silently rot.
    ///
    /// `x ∈ [1, 2]`, `y ∈ [0, 1]`, one product, weight `1`. At `r = 1` — and on
    /// the `None` arm, whose midpoint selection ties and keeps `L1`/`U1` — the
    /// planes are `L1 = y` and `U1 = x + y − 1`, so with the constant halved onto
    /// each side:
    ///
    /// ```text
    /// A half:  lower ≈ 0·x + 0      upper ≈ 1·x − 1/2
    /// B half:  lower ≈ 1·y + 0      upper ≈ 1·y − 1/2
    /// ```
    ///
    /// The B half's lower sits ~`1/2` ABOVE its upper at EVERY `y`: it is not an
    /// enclosure of anything, so at the point `y = 0.5` `concretize_sound`'s
    /// per-row inversion repair publishes it as `[-inf, +inf]` — with every
    /// coefficient, bias and certified error finite, so no NaN / degrade /
    /// `CROWN_COEFF_MAX` guard can be what fired. Over the whole B BOX the same
    /// half happens not to invert (`[~0, ~0.5]`). The SUMMED relation `[y, x + y
    /// − 1]` is a valid McCormick enclosure of `x·y`: at `(1.5, 0.5)` it
    /// concretizes finite (`≈ [0.5, 1.0]`) and contains `0.75` exactly.
    #[test]
    fn matmul_operand_half_inverts_alone_and_only_the_summed_relation_encloses() {
        let bt = |lo: f32, hi: f32| {
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![lo]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![hi]).unwrap(),
            )
            .unwrap()
        };
        let point = |values: Vec<f32>| {
            BoundedTensor::concrete(ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).unwrap())
                .unwrap()
        };
        let a = bt(1.0, 2.0);
        let b = bt(0.0, 1.0);
        let spec = LinearBounds::from_spec_matrix(Array2::from_elem((1, 1), 1.0_f32)).unwrap();
        let alphas = Array4::<f32>::ones((4, 1, 1, 1));
        let (x, y) = (1.5_f32, 0.5_f32);
        let matmul = MatMulLayer::new(false, None);

        for (arm, parameters) in [("alpha", Some(&alphas)), ("none", None)] {
            let (half_a, half_b) = matmul
                .propagate_linear_binary_with_interpolation(&spec, &a, &b, parameters)
                .unwrap();

            // Precondition: nothing is non-finite, so the ONLY concretize guard
            // that can vacate a row here is the inversion repair.
            for half in [&half_a, &half_b] {
                let lower_err = half
                    .lower_a_err()
                    .expect("the arm certifies its coefficients");
                let upper_err = half
                    .upper_a_err()
                    .expect("the arm certifies its coefficients");
                let all_finite = half
                    .lower_a()
                    .iter()
                    .chain(half.upper_a())
                    .chain(half.lower_b())
                    .chain(half.upper_b())
                    .chain(lower_err)
                    .chain(upper_err)
                    .all(|v| v.is_finite());
                assert!(
                    all_finite,
                    "{arm}: precondition — every coefficient, bias and error is finite"
                );
            }

            // The B half ALONE is inverted at the point, exactly (no rounding) ...
            let (b_lower, b_upper) = relation_at_point(&half_b, 0, &[y]);
            assert!(
                b_lower.clone() - b_upper.clone() >= rat(0.25),
                "{arm}: the B half's lower ({b_lower}) must sit ~1/2 above its own upper \
                 ({b_upper}) at y = {y} — it is a SHARE of an enclosure, not one"
            );
            // ... so the production concretizer repairs that row to [-inf, +inf].
            let alone = half_b.concretize_sound(&point(vec![y]));
            assert_eq!(
                (alone.lower()[0], alone.upper()[0]),
                (f32::NEG_INFINITY, f32::INFINITY),
                "{arm}: concretize_sound must fail an inverted row closed to [-inf, +inf]"
            );
            // The same half over its whole BOX is not inverted.
            let over_box = half_b.concretize_sound(&b);
            assert!(
                over_box.lower()[0].is_finite() && over_box.upper()[0].is_finite(),
                "{arm}: the B half over y in [0, 1] is [~0, ~0.5], not inverted"
            );
            // At most one half inverts: the A half is fine at its point.
            let (a_lower, a_upper) = relation_at_point(&half_a, 0, &[x]);
            assert!(
                a_lower <= a_upper,
                "{arm}: the A half is not inverted at x = {x}"
            );

            // The SUMMED relation encloses x·y, exactly and through the concretizer.
            let whole = operand_pair_relation(&half_a, &half_b);
            let exact = rat(x) * rat(y);
            let (whole_lower, whole_upper) = relation_at_point(&whole, 0, &[x, y]);
            assert!(
                whole_lower <= exact && exact <= whole_upper,
                "{arm}: the summed relation [{whole_lower}, {whole_upper}] must contain \
                 x·y = {exact}"
            );
            let published = whole.concretize_sound(&point(vec![x, y]));
            let (lo, hi) = (published.lower()[0], published.upper()[0]);
            assert!(
                lo.is_finite() && hi.is_finite(),
                "{arm}: the summed relation concretized vacuous: [{lo:e}, {hi:e}]"
            );
            assert!(
                rat(lo) <= exact && exact <= rat(hi),
                "{arm}: published [{lo:e}, {hi:e}] must contain x·y = 0.75"
            );
            // Hand-derived: [y, x + y − 1] = [0.5, 1.0], up to certified slack.
            assert!(
                lo >= 0.49 && hi <= 1.01,
                "{arm}: published [{lo:e}, {hi:e}] should be [0.5, 1.0] up to rounding slack"
            );
        }
    }

    /// One row of a relation concretized EXACTLY over a box: the same corner
    /// selection (`lo` where the coefficient is non-negative, `hi` otherwise,
    /// mirrored for the upper side) and the same `∓ err·max(|lo|, |hi|)` charge as
    /// `concretize_sound`, with no floating rounding.
    fn relation_over_box(
        relation: &LinearBounds,
        row: usize,
        lo: &[f32],
        hi: &[f32],
    ) -> (BigRational, BigRational) {
        assert_eq!(relation.num_inputs(), lo.len(), "relation / box width");
        assert_eq!(lo.len(), hi.len(), "box endpoints");
        let mut lower = rat(relation.lower_b()[row]);
        let mut upper = rat(relation.upper_b()[row]);
        for (j, (&l, &h)) in lo.iter().zip(hi).enumerate() {
            let magnitude = rat(l.abs().max(h.abs()));
            let coefficient_lower = relation.lower_a()[[row, j]];
            let coefficient_upper = relation.upper_a()[[row, j]];
            let at_lower = if coefficient_lower >= 0.0 { l } else { h };
            let at_upper = if coefficient_upper >= 0.0 { h } else { l };
            lower += rat(coefficient_lower) * rat(at_lower);
            upper += rat(coefficient_upper) * rat(at_upper);
            if let Some(err) = relation.lower_a_err() {
                lower -= rat(err[[row, j]]) * magnitude.clone();
            }
            if let Some(err) = relation.upper_a_err() {
                upper += rat(err[[row, j]]) * magnitude;
            }
        }
        (lower, upper)
    }

    /// HAZARD PIN (#concretize-vacuity) — this pins an existing LOOSENESS, not a
    /// desired behaviour. The point-level vacuity above is a test artefact, but
    /// the same inversion exists over a GENUINE box, and several production
    /// paths DO concretize a node's share of a relation at that node's box with
    /// `concretize_sound`: the graph-CROWN per-node IBP fallback and the Div
    /// `ConcretizeCurrentNode` route (`concretized_node_bias_with_deadline`), the
    /// graph-α cut segment, and the truncated-frontier collapse. On those paths a
    /// MatMul operand share that inverts is published as `[-inf, +inf]`, which
    /// vacates the whole spec row — sound, but its two endpoints were usable.
    ///
    /// `x ∈ [0.1, 0.12]`, `y ∈ [1, 2]`, weight `1`, every interpolation
    /// parameter `0` (the `L2`/`U2` planes, which the optimizer can reach):
    /// the A share carries `2·x` below and `1·x` above, so over its box its
    /// minimum `≈ 0.2` sits above its maximum `≈ 0.12`. The shares are formed
    /// exactly as `backward_dispatch` leaves them (constants summed into one bias
    /// channel, zero-bias blocks). The pin also shows what is lost: summing the
    /// shares' INDEPENDENT endpoints with the bias channel encloses `x·y ∈ [0.1,
    /// 0.24]` as `≈ [0.08, 0.24]`. If per-share concretization is ever made
    /// endpoint-wise, flip the vacated assertion — after a moat run.
    #[test]
    fn matmul_operand_share_can_invert_over_a_genuine_box_so_per_share_concretization_vacates() {
        let (x_l, x_u, y_l, y_u) = (0.1_f32, 0.12_f32, 1.0_f32, 2.0_f32);
        let bt = |lo: f32, hi: f32| {
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![lo]).unwrap(),
                ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![hi]).unwrap(),
            )
            .unwrap()
        };
        let a = bt(x_l, x_u);
        let b = bt(y_l, y_u);
        let spec = LinearBounds::from_spec_matrix(Array2::from_elem((1, 1), 1.0_f32)).unwrap();
        let alphas = Array4::<f32>::zeros((4, 1, 1, 1));
        let (mut share_a, mut share_b) = MatMulLayer::new(false, None)
            .propagate_linear_binary_with_interpolation(&spec, &a, &b, Some(&alphas))
            .unwrap();
        assert_eq!(
            (share_a.lower_a()[[0, 0]], share_a.upper_a()[[0, 0]]),
            (y_u, y_l),
            "precondition: r = 0 selects L2 (x-coefficient y_u) and U2 (x-coefficient y_l)"
        );

        // Exactly what `backward_dispatch` does: one bias channel, zero-bias shares.
        let bias_lower = share_a.lower_b() + share_b.lower_b();
        let bias_upper = share_a.upper_b() + share_b.upper_b();
        for share in [&mut share_a, &mut share_b] {
            share.lower_b_mut().fill(0.0);
            share.upper_b_mut().fill(0.0);
        }

        // The A share alone is inverted over its GENUINE box, exactly ...
        let (a_lower, a_upper) = relation_over_box(&share_a, 0, &[x_l], &[x_u]);
        assert!(
            a_lower.clone() - a_upper.clone() >= rat(0.05),
            "the A share's minimum ({a_lower}) must sit ~0.08 above its maximum ({a_upper}) \
             over x in [{x_l}, {x_u}]"
        );
        // ... so concretizing it at its node box, as the per-node paths do, vacates it.
        let vacated = share_a.concretize_sound(&a);
        assert_eq!(
            (vacated.lower()[0], vacated.upper()[0]),
            (f32::NEG_INFINITY, f32::INFINITY),
            "an inverted share is repaired to [-inf, +inf] by concretize_sound"
        );

        // Yet its endpoints ARE valid contributions: with the B share's endpoints and
        // the bias channel they enclose x·y over the box, and tightly.
        let (b_lower, b_upper) = relation_over_box(&share_b, 0, &[y_l], &[y_u]);
        assert!(
            b_lower <= b_upper,
            "the B share is not inverted over its box"
        );
        let lower = a_lower + b_lower + rat(bias_lower[0]);
        let upper = a_upper + b_upper + rat(bias_upper[0]);
        let min_product = rat(x_l) * rat(y_l);
        let max_product = rat(x_u) * rat(y_u);
        assert!(
            lower <= min_product && max_product <= upper,
            "endpoint-wise shares + bias [{lower}, {upper}] must enclose x·y over the box \
             [{min_product}, {max_product}]"
        );
        assert!(
            lower >= rat(0.07) && upper <= rat(0.25),
            "endpoint-wise shares + bias [{lower}, {upper}] should be ~[0.08, 0.24]"
        );
    }
}

/// #vit-collector-throughput P2 step 1: the row-parallel, plane-hoisted
/// McCormick kernels are BIT-IDENTICAL to the serial nests — every published
/// coefficient, constant and certified-error cell, on both operand shares —
/// for the midpoint (`MatMul`) arm and the interpolated alpha (`BilinearCrown`)
/// arm, on vit_2023's attention shapes and on random boxes with batch dims,
/// both `transpose_b` settings, non-power-of-two scales, zero weights and
/// single-row / multi-chunk row counts.
///
/// The row-parallel kernels are called DIRECTLY and must accept every shape
/// here (`Some`): a decline would fall back to the serial nest and make the
/// comparison vacuous. The routed entries (`rowpar == true`) are compared too,
/// so the gate that production takes is pinned as well.
#[cfg(test)]
mod rowpar_bit_identity_tests {
    use super::*;
    use crate::LinearBounds;
    use ndarray::{Array4, ArrayD, IxDyn};

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

        fn unit(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1_u64 << 24) as f32
        }

        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }

        fn pick(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    /// One endpoint pair: mixed sign, a point interval, a tiny endpoint beside
    /// a large one, or a same-sign interval away from zero.
    fn interval(rng: &mut SplitMix) -> (f32, f32) {
        let (a, b) = match rng.pick(4) {
            0 => (rng.range(-4.0, 4.0), rng.range(-4.0, 4.0)),
            1 => {
                let v = rng.range(-3.0, 3.0);
                (v, v)
            }
            2 => (rng.range(-1e-3, 1e-3), rng.range(0.5, 500.0)),
            _ => {
                let base = rng.range(1.0, 100.0);
                (base, base + rng.range(0.0, 20.0))
            }
        };
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    }

    fn random_box(rng: &mut SplitMix, shape: &[usize]) -> BoundedTensor {
        let len: usize = shape.iter().product();
        let (lower, upper): (Vec<f32>, Vec<f32>) = (0..len).map(|_| interval(rng)).unzip();
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(shape), lower).unwrap(),
            ArrayD::from_shape_vec(IxDyn(shape), upper).unwrap(),
        )
        .unwrap()
    }

    /// Incoming relation: independent lower/upper coefficients with ~20% exact
    /// zeros (the nest's `w != 0.0` guards) and a few negative zeros.
    fn random_bounds(rng: &mut SplitMix, rows: usize, cols: usize) -> LinearBounds {
        let coeff = |rng: &mut SplitMix| match rng.pick(10) {
            0 | 1 => 0.0,
            2 => -0.0,
            _ => rng.range(-2.0, 2.0),
        };
        let lower_a = Array2::from_shape_fn((rows, cols), |_| coeff(rng));
        let upper_a = Array2::from_shape_fn((rows, cols), |_| coeff(rng));
        let lower_b = Array1::from_shape_fn(rows, |_| rng.range(-1.0, 1.0));
        let upper_b = Array1::from_shape_fn(rows, |_| rng.range(-1.0, 1.0));
        LinearBounds::new(lower_a, lower_b, upper_a, upper_b).unwrap()
    }

    fn shape_for(transpose_b: bool, a: &[usize], b: &[usize]) -> McCormickRowparShape {
        let dims = parse_matmul_dims(transpose_b, a, b).unwrap();
        let batch_size = dims.batch_size().unwrap();
        let c_size_per_batch = dims.c_size_per_batch().unwrap();
        let a_size_per_batch = dims.a_size_per_batch().unwrap();
        McCormickRowparShape {
            batch_size,
            m: dims.m,
            n: dims.n,
            k: dims.k,
            c_size_per_batch,
            a_size_per_batch,
            b_size_per_batch: dims.b_size_per_batch,
            total_a_size: batch_size * a_size_per_batch,
            total_b_size: batch_size * dims.b_size_per_batch,
            transpose_b,
        }
    }

    /// Every published bit of one share, certified-error planes included.
    fn bits(bounds: &LinearBounds) -> Vec<u32> {
        let mut out: Vec<u32> = bounds.lower_a().iter().map(|v| v.to_bits()).collect();
        out.extend(bounds.upper_a().iter().map(|v| v.to_bits()));
        out.extend(bounds.lower_b().iter().map(|v| v.to_bits()));
        out.extend(bounds.upper_b().iter().map(|v| v.to_bits()));
        out.push(u32::from(bounds.lower_a_err().is_some()));
        out.push(u32::from(bounds.upper_a_err().is_some()));
        for err in [bounds.lower_a_err(), bounds.upper_a_err()]
            .into_iter()
            .flatten()
        {
            out.extend(err.iter().map(|v| v.to_bits()));
        }
        out
    }

    fn assert_same(
        serial: &(LinearBounds, LinearBounds),
        other: &(LinearBounds, LinearBounds),
        label: &str,
    ) {
        assert_eq!(bits(&serial.0), bits(&other.0), "{label}: A share differs");
        assert_eq!(bits(&serial.1), bits(&other.1), "{label}: B share differs");
    }

    #[test]
    fn bilinear_rowpar_bit_identical_to_serial() {
        let mut rng = SplitMix(0x0B11_1EA5_2026_0926);
        // (transpose_b, A shape, B shape, scale, incoming rows)
        let cases: Vec<(bool, Vec<usize>, Vec<usize>, Option<f32>, usize)> = vec![
            // vit_2023 attn·V shape named by the brief: [3,17,16]·[3,16,17].
            (false, vec![3, 17, 16], vec![3, 16, 17], None, 37),
            // vit_2023 Q·Kᵀ shape, with a non-power-of-two attention scale.
            (true, vec![3, 17, 16], vec![3, 17, 16], Some(0.25), 29),
            (true, vec![3, 17, 16], vec![3, 17, 16], Some(0.37), 5),
            // Two batch dims.
            (false, vec![2, 3, 5, 4], vec![2, 3, 4, 6], Some(0.37), 9),
            // Plain 2-D, one row (single chunk, sequential driver).
            (true, vec![5, 7], vec![3, 7], None, 1),
            // k = 1 and many rows (many chunks).
            (false, vec![4, 1], vec![1, 3], Some(1.5), 64),
        ];
        for (case, (transpose_b, a_shape, b_shape, scale, rows)) in cases.iter().enumerate() {
            let shape = shape_for(*transpose_b, a_shape, b_shape);
            let total_c = shape.batch_size * shape.c_size_per_batch;
            let raw_scale = scale.unwrap_or(1.0);
            for rep in 0..3 {
                let label = format!("case {case} rep {rep}");
                let a = random_box(&mut rng, a_shape);
                let b = random_box(&mut rng, b_shape);
                let incoming = random_bounds(&mut rng, *rows, total_c);

                // Midpoint (alpha-free) MatMul arm.
                let matmul = MatMulLayer::new(*transpose_b, *scale);
                let serial = matmul
                    .propagate_linear_binary_with_kernel(&incoming, &a, &b, false)
                    .unwrap();
                let rowpar = matmul
                    .propagate_linear_binary_rowpar(&incoming, &a, &b, &shape, raw_scale)
                    .unwrap()
                    .expect("the row-parallel midpoint kernel must accept this shape");
                assert_same(&serial, &rowpar, &format!("{label} midpoint direct"));
                let routed = matmul
                    .propagate_linear_binary_with_kernel(&incoming, &a, &b, true)
                    .unwrap();
                assert_same(&serial, &routed, &format!("{label} midpoint routed"));

                // Interpolated alpha BilinearCrown arm; parameters outside
                // [0, 1] exercise the per-element clamp.
                let bilinear = BilinearCrownLayer::new(*transpose_b, *scale);
                let alphas = Array4::from_shape_fn((4, shape.m, shape.n, shape.k), |_| {
                    rng.range(-0.25, 1.25)
                });
                let serial = bilinear
                    .propagate_linear_binary_with_alpha_kernel(
                        &incoming,
                        &a,
                        &b,
                        Some(&alphas),
                        false,
                    )
                    .unwrap();
                let rowpar = bilinear
                    .propagate_linear_binary_with_alpha_rowpar(
                        &incoming, &a, &b, &alphas, &shape, raw_scale,
                    )
                    .unwrap()
                    .expect("the row-parallel alpha kernel must accept this shape");
                assert_same(&serial, &rowpar, &format!("{label} alpha direct"));
                let routed = bilinear
                    .propagate_linear_binary_with_alpha_kernel(
                        &incoming,
                        &a,
                        &b,
                        Some(&alphas),
                        true,
                    )
                    .unwrap();
                assert_same(&serial, &routed, &format!("{label} alpha routed"));

                // The alpha entry's `None` arm delegates to the midpoint arm
                // with the SAME kernel choice.
                let none_serial = bilinear
                    .propagate_linear_binary_with_alpha_kernel(&incoming, &a, &b, None, false)
                    .unwrap();
                let none_rowpar = bilinear
                    .propagate_linear_binary_with_alpha_kernel(&incoming, &a, &b, None, true)
                    .unwrap();
                assert_same(&none_serial, &none_rowpar, &format!("{label} alpha None"));
            }
        }
    }
}
