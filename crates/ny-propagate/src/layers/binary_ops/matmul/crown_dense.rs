// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Non-batched CROWN backward propagation for MatMul using McCormick envelopes.

use ndarray::{Array1, Array2};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::super::validate_mccormick_inputs;
use super::shape::{decode_batch_index_into, parse_matmul_dims};
use super::{select_mccormick_plane, BoundDir, LinearBounds, MatMulLayer, NyError, Result};
use crate::layers::collector_kernels::BilinearKernelChoice;

/// Widen one accumulated McCormick ENVELOPE CONSTANT outward by a certified bound
/// on its own floating-point error (#matmul-dense-mccormick-const).
///
/// # The defect this closes
///
/// `propagate_linear_binary` folds the envelope constant as
/// `const += w * c` in PLAIN round-to-nearest f64 and then published it with no
/// error term at all — only the final f64 -> f32 directed cast in the halving step.
/// That cast covers the narrowing and nothing else, so two distinct errors reached
/// the verdict path uncharged:
///
/// 1. **f64 accumulation.** Recursive summation of `adds + 1` summands satisfies
///    `|fl(S) - S| <= gamma^f64_{adds} * sum|summand|` (Higham, *Accuracy and
///    Stability*, Thm 3.1). The charge scales with the MAGNITUDE sum, not with
///    `|S|`, so it does NOT vanish under cancellation — which is exactly when a
///    result-scaled 1-ULP cast allowance under-counts it. `adds * 2^-53 * S` beats
///    one f32 ULP of the result (`2^-24 * |S|`) as soon as
///    `S / |result| > 2^29 / adds`, i.e. a cancellation ratio of ~500 at a million
///    terms — ordinary for an attention-sized bilinear fold.
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
/// The `1/(1 - gamma)` inflation mirrors `conv2d::bound` : `magnitude_sum` is
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
///
/// `pub(super)` so the GEMM-shaped twin (`crown_gemm.rs`, #vit-batched-bilinear)
/// finishes its folds with THIS function, not a copy of it.
pub(super) fn widen_mccormick_constant(
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
    /// CROWN backward propagation for MatMul (C = A @ B or A @ B^T).
    ///
    /// Uses McCormick envelope relaxation for the bilinear terms.
    /// Supports batched N-D inputs: A has shape [..., M, K], B has shape [..., K, N] or [..., N, K].
    ///
    /// Returns (bounds_for_a, bounds_for_b).
    ///
    /// Neither returned relation is an enclosure on its own: the McCormick
    /// constant is HALVED onto each, and only the SUM over `[A; B]` bounds the
    /// incoming relation. One half's lower can sit above its own upper (at a
    /// point, routinely), so never `concretize_sound` a half by itself — its
    /// inversion repair returns `[-inf, +inf]`. Callers sum the halves
    /// (`backward_dispatch` moves both constants into one bias channel) before
    /// concretizing (#concretize-vacuity).
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
    ) -> Result<(LinearBounds, LinearBounds)> {
        self.propagate_linear_binary_with_kernels(
            bounds,
            input_a_bounds,
            input_b_bounds,
            BilinearKernelChoice::armed(),
        )
    }

    /// [`Self::propagate_linear_binary`] with the kernel choice explicit
    /// (#vit-collector-throughput P2 step 1).
    ///
    /// `rowpar == false` is the historical serial nest, byte-for-byte.
    /// `rowpar == true` first offers the problem to
    /// [`Self::propagate_linear_binary_rowpar`], which is BIT-IDENTICAL to the
    /// serial nest (see its docs) and declines shapes it does not handle, in
    /// which case the serial nest runs exactly as before. Never tries the
    /// GEMM-shaped kernel (see [`Self::propagate_linear_binary_with_kernels`]).
    /// Test-only since #vit-batched-bilinear routed production through
    /// `_with_kernels`.
    #[cfg(test)]
    pub(crate) fn propagate_linear_binary_with_kernel(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        rowpar: bool,
    ) -> Result<(LinearBounds, LinearBounds)> {
        self.propagate_linear_binary_with_kernels(
            bounds,
            input_a_bounds,
            input_b_bounds,
            BilinearKernelChoice {
                gemm: false,
                rowpar,
            },
        )
    }

    /// [`Self::propagate_linear_binary`] with every kernel choice explicit
    /// (#vit-batched-bilinear, ENGINE_PROGRAM V1).
    ///
    /// Tried in order, each only when chosen: the GEMM-shaped kernel
    /// ([`Self::propagate_linear_binary_gemm`], `crown_gemm.rs`), the
    /// row-parallel kernel ([`Self::propagate_linear_binary_rowpar`]), then the
    /// historical serial nest. Both kernels are BIT-IDENTICAL to the nest and
    /// decline shapes they do not handle, so every choice publishes the same
    /// bits. `BilinearKernelChoice::SERIAL` is the serial nest byte-for-byte.
    /// Production passes the typed latches (`bab.bilinear_gemm_kernel`,
    /// `bab.bilinear_rowpar_kernel`); tests pass every combination.
    pub(crate) fn propagate_linear_binary_with_kernels(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        kernels: BilinearKernelChoice,
    ) -> Result<(LinearBounds, LinearBounds)> {
        let dims = parse_matmul_dims(
            self.transpose_b,
            input_a_bounds.shape(),
            input_b_bounds.shape(),
        )?;

        validate_mccormick_inputs(input_a_bounds, input_b_bounds, "MatMul")?;

        let batch_size = dims.batch_size()?;
        let c_size_per_batch = dims.c_size_per_batch()?;
        let a_size_per_batch = dims.a_size_per_batch()?;

        let total_c_size = batch_size.checked_mul(c_size_per_batch).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MatMul: total_c_size overflow: {batch_size} * {c_size_per_batch}",
            ))
        })?;
        let total_a_size = batch_size.checked_mul(a_size_per_batch).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MatMul: total_a_size overflow: {batch_size} * {a_size_per_batch}",
            ))
        })?;
        let total_b_size = batch_size
            .checked_mul(dims.b_size_per_batch)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "MatMul: total_b_size overflow: {batch_size} * {}",
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
        // (finally the serial nest) then runs unchanged.
        if kernels.gemm || kernels.rowpar {
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
            // A zero-sized batch dimension makes the serial nest's index decode
            // fail; decline so it fails there, with its own error.
            let degenerate_batch = dims.batch_dims.contains(&0);
            if kernels.gemm {
                let published = if degenerate_batch {
                    None
                } else {
                    self.propagate_linear_binary_gemm(
                        bounds,
                        input_a_bounds,
                        input_b_bounds,
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
                let published = if degenerate_batch {
                    None
                } else {
                    self.propagate_linear_binary_rowpar(
                        bounds,
                        input_a_bounds,
                        input_b_bounds,
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

        // Certified-error accumulators (#matmul-dense-mccormick): each coefficient
        // cell is summed in round-to-nearest f32 over the j axis (A side, depth
        // dims.n) / i axis (B side, depth dims.m), so it carries the Higham f32
        // accumulation error gamma_n * S that MUST reach concretize. S = Sum|w*a| is
        // accumulated exactly in f64 (f32*f32 is exact in f64). #vnncomp-aw-soundness.
        let mut lower_s_a = Array2::<f64>::zeros((num_outputs, total_a_size));
        let mut lower_s_b = Array2::<f64>::zeros((num_outputs, total_b_size));
        let mut upper_s_a = Array2::<f64>::zeros((num_outputs, total_a_size));
        let mut upper_s_b = Array2::<f64>::zeros((num_outputs, total_b_size));

        let mut lower_b_total = Array1::<f64>::zeros(num_outputs);
        let mut upper_b_total = Array1::<f64>::zeros(num_outputs);

        for out_idx in 0..num_outputs {
            let mut const_lower = bounds.lower_b()[out_idx] as f64;
            let mut const_upper = bounds.upper_b()[out_idx] as f64;
            // Certified-error accumulators for the ENVELOPE CONSTANT
            // (#matmul-dense-mccormick-const). `*_s` is the Higham magnitude sum
            // Sum|summand| INCLUDING the incoming bias (the f64 accumulation charge
            // is against every summand); `*_terms_s` excludes it (the per-term f32
            // plane/scale rounding charge applies only to the McCormick terms).
            // `*_adds` counts the f64 additions actually performed, so the
            // term-free case charges exactly zero. See `widen_mccormick_constant`.
            let mut const_lower_s = const_lower.abs();
            let mut const_upper_s = const_upper.abs();
            let mut const_lower_terms_s = 0.0_f64;
            let mut const_upper_terms_s = 0.0_f64;
            let mut const_lower_adds = 0_usize;
            let mut const_upper_adds = 0_usize;
            let batch_index_len = dims.batch_dims.len();
            let mut batch_indices = Vec::with_capacity(batch_index_len);
            let mut a_idx = Vec::with_capacity(batch_index_len + 2);
            let mut b_idx = Vec::with_capacity(batch_index_len + 2);

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
                            let lx = input_a_bounds.lower()[a_idx.as_slice()];
                            let ux = input_a_bounds.upper()[a_idx.as_slice()];
                            // Bit-identical McCormick anchor: f32::midpoint rounds differently at overflow/subnormal edges.
                            #[allow(clippy::manual_midpoint)]
                            let x0 = (lx + ux) * 0.5;

                            let a_flat = batch_idx * a_size_per_batch + i * dims.k + l;

                            // Get B[batch..., l, j] or B[batch..., j, l] if transposed
                            let b_flat = if self.transpose_b {
                                b_idx[batch_index_len + 1] = l;
                                batch_idx * dims.b_size_per_batch + j * dims.k + l
                            } else {
                                b_idx[batch_index_len] = l;
                                batch_idx * dims.b_size_per_batch + l * dims.n + j
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
                                lower_a_a[[out_idx, a_flat]] += w_lower * ax;
                                lower_a_b[[out_idx, b_flat]] += w_lower * ay;
                                lower_s_a[[out_idx, a_flat]] +=
                                    (w_lower as f64).abs() * (ax as f64).abs();
                                lower_s_b[[out_idx, b_flat]] +=
                                    (w_lower as f64).abs() * (ay as f64).abs();
                                // f32 x f32 is EXACT in f64 (48 < 53 significand
                                // bits), so only this ADDITION rounds — that is the
                                // error `widen_mccormick_constant` charges for.
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
                                upper_a_a[[out_idx, a_flat]] += w_upper * ax;
                                upper_a_b[[out_idx, b_flat]] += w_upper * ay;
                                upper_s_a[[out_idx, a_flat]] +=
                                    (w_upper as f64).abs() * (ax as f64).abs();
                                upper_s_b[[out_idx, b_flat]] +=
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

            lower_b_total[out_idx] = widen_mccormick_constant(
                const_lower,
                const_lower_s,
                const_lower_terms_s,
                const_lower_adds,
                BoundDir::Lower,
            );
            upper_b_total[out_idx] = widen_mccormick_constant(
                const_upper,
                const_upper_s,
                const_upper_terms_s,
                const_upper_adds,
                BoundDir::Upper,
            );
        }

        // Split constant terms across both inputs so that GraphNetwork accumulation
        // sums correctly. Halve in f64, then apply directed rounding on final f32
        // cast. The f64 totals reaching this point have ALREADY been widened
        // outward for their own accumulation and plane-rounding error by
        // `widen_mccormick_constant`; the directed cast below only covers the
        // f64 -> f32 narrowing, which is NOT sufficient on its own (#matmul-dense-
        // mccormick-const).
        let lower_b_half = lower_b_total.mapv(|v| next_down_f32((v * 0.5) as f32));
        let upper_b_half = upper_b_total.mapv(|v| next_up_f32((v * 0.5) as f32));

        // Certified coefficient error (#matmul-dense-mccormick): A-side cells
        // accumulate over dims.n terms in f32, B-side cells over dims.m terms.
        // CORRECTED CHARGE (#matmul-dense-mccormick-const): `gamma_n_f32(count)`
        // covers only the `count - 1` ADDITIONS. Each summand also carries TWO
        // further f32 roundings before it is added -- `w = fl32(lower_a * scale)`
        // and `fl32(w * ax)` -- so the honest Higham factor is `gamma_{count+1}`;
        // `count + 2` buys one more step, which also absorbs the (far smaller) f64
        // accumulation deficit of S itself. |stored - exact| <=
        // gamma_n_f32(count + 2) * S, rounded UP to a sound f32 and carried so
        // concretize penalizes outward. Monotone in `count`, so this is strictly
        // looser than the previous charge at every cell.
        let gamma_a = crate::layers::linear::crown_single_gamma_n_f32(dims.n.saturating_add(2));
        let gamma_b = crate::layers::linear::crown_single_gamma_n_f32(dims.m.saturating_add(2));
        let lower_a_a_err = lower_s_a.mapv(|s| next_up_f32((gamma_a * s) as f32));
        let upper_a_a_err = upper_s_a.mapv(|s| next_up_f32((gamma_a * s) as f32));
        let lower_a_b_err = lower_s_b.mapv(|s| next_up_f32((gamma_b * s) as f32));
        let upper_a_b_err = upper_s_b.mapv(|s| next_up_f32((gamma_b * s) as f32));

        let bounds_a = LinearBounds::new_or_conservative_with_err(
            lower_a_a,
            lower_b_half.clone(),
            upper_a_a,
            upper_b_half.clone(),
            lower_a_a_err,
            upper_a_a_err,
        )?;

        let bounds_b = LinearBounds::new_or_conservative_with_err(
            lower_a_b,
            lower_b_half,
            upper_a_b,
            upper_b_half,
            lower_a_b_err,
            upper_a_b_err,
        )?;

        Ok((bounds_a, bounds_b))
    }
}

// ---------------------------------------------------------------------------
// #vit-collector-throughput P2 step 1: the row-parallel, plane-hoisted
// McCormick backward (typed `bab.bilinear_rowpar_kernel`, dark by default).
// ---------------------------------------------------------------------------
//
// WHY. The serial nest above is `rows x batch x m x n x k` scalar iterations on
// ONE thread, with four `IxDyn` lookups and two `select_mccormick_plane` calls
// per iteration — and the plane it selects depends only on the (b, i, j, l)
// element's operand bounds and the SIGN of the incoming weight, never on the
// row. On vit_2023's attention products that is 13,872 iterations per row per
// crossing, recomputing the same planes for every one of up to 1,632 rows, and
// it is why no intermediate collection target completes inside its share
// (scratchpad/vbs/diag6/vit.md §2).
//
// WHAT CHANGES. (1) The operand bounds are read from flat row-major slices
// instead of `IxDyn` indexing. (2) For one (b, i, j) slab the four candidate
// planes of every `l` — {Lower, Upper} x {w >= 0, w < 0} — are computed ONCE
// per chunk of rows instead of once per row. (3) Chunks of rows run on rayon.
//
// WHY IT IS BIT-IDENTICAL (the soundness argument; nothing here is a new
// relaxation):
//
// * Rows are independent. Every accumulator the nest writes is indexed by the
//   row (`[out_idx, ..]` coefficient and S cells, the per-row constant folds),
//   and nothing reads another row's accumulator. Running rows on different
//   threads therefore changes no value.
// * Within one row the operation SEQUENCE is unchanged: the same (b, i, j, l)
//   visiting order, the same `w * ax` f32 products added in the same order to
//   the same cells, the same exact-in-f64 `|w|·|a|` S terms, the same f64
//   constant fold (`+= w as f64 * c as f64`, magnitude sums, add count) and
//   the same `widen_mccormick_constant` publication. Rust does not contract
//   `a * b + c` into an FMA, so an identical source expression is an identical
//   IEEE operation.
// * `select_mccormick_plane` is a pure function of its eight arguments and
//   reads the weight ONLY through `w >= 0.0`. The table stores the result for
//   `w = 1.0` and `w = -1.0` in each direction, and the row loop picks with the
//   SAME predicate `w >= 0.0` — so a NaN weight (predicate false) and `-0.0`
//   (skipped by `w != 0.0` in both) behave exactly as in the nest, and the
//   non-finite guard (which ignores `w`) returns the same trivial plane.
// * The flat slices are the operands' logical row-major order
//   (`contiguous_flat_slice` borrows a standard-layout array or copies it in
//   iteration order), and `a_flat` / `b_flat` are the nest's own formulas,
//   which ARE the row-major offsets of the nest's `IxDyn` index vectors
//   (`decode_batch_index_into` is the row-major decode of `batch_idx`).
// * The publication tail (`McCormickRowparBuffers::publish`) is the nest's tail
//   verbatim: exact halving, directed f32 casts, `gamma_{depth+2}` charges.
//
// `bilinear_rowpar_bit_identical_to_serial` pins all of this on vit shapes and
// random boxes, for this arm and for the `BilinearCrown` alpha arm.

/// Shape facts the row-parallel kernels need, computed once by the caller from
/// the SAME `parse_matmul_dims` result (and overflow checks) as the nest.
pub(in crate::layers::binary_ops) struct McCormickRowparShape {
    pub(in crate::layers::binary_ops) batch_size: usize,
    pub(in crate::layers::binary_ops) m: usize,
    pub(in crate::layers::binary_ops) n: usize,
    pub(in crate::layers::binary_ops) k: usize,
    pub(in crate::layers::binary_ops) c_size_per_batch: usize,
    pub(in crate::layers::binary_ops) a_size_per_batch: usize,
    pub(in crate::layers::binary_ops) b_size_per_batch: usize,
    pub(in crate::layers::binary_ops) total_a_size: usize,
    pub(in crate::layers::binary_ops) total_b_size: usize,
    pub(in crate::layers::binary_ops) transpose_b: bool,
}

impl McCormickRowparShape {
    /// Row-major offset of `A[batch, i, l]` — the nest's `a_flat`.
    #[inline]
    pub(in crate::layers::binary_ops) fn a_flat(&self, batch: usize, i: usize, l: usize) -> usize {
        batch * self.a_size_per_batch + i * self.k + l
    }

    /// Row-major offset of `B[batch, l, j]` (or `B[batch, j, l]` when
    /// transposed) — the nest's `b_flat`.
    #[inline]
    pub(in crate::layers::binary_ops) fn b_flat(&self, batch: usize, j: usize, l: usize) -> usize {
        if self.transpose_b {
            batch * self.b_size_per_batch + j * self.k + l
        } else {
            batch * self.b_size_per_batch + l * self.n + j
        }
    }

    /// Row-major offset of `C[batch, i, j]` — the nest's `c_flat`.
    #[inline]
    pub(in crate::layers::binary_ops) fn c_flat(&self, batch: usize, i: usize, j: usize) -> usize {
        batch * self.c_size_per_batch + i * self.n + j
    }
}

/// Rows per chunk. Every chunk recomputes the slab planes once, so a chunk
/// should hold enough rows to amortize that, and there should be a few chunks
/// per worker for balance. Neither choice can change a value (rows are
/// independent), only wall time. (Also the GEMM-shaped twin's default,
/// rounded up to its row tile.)
pub(super) fn mccormick_rowpar_chunk_rows(num_outputs: usize) -> usize {
    let workers = rayon::current_num_threads().max(1);
    num_outputs
        .div_ceil(workers.saturating_mul(4))
        .clamp(4, 256)
}

/// One chunk of consecutive output rows and the disjoint slices of every
/// accumulator those rows own. Row `r` of the chunk is output row
/// `row_start + r`; its coefficient cells are `[r * total, (r + 1) * total)`.
pub(in crate::layers::binary_ops) struct McCormickRowChunk<'a> {
    pub(in crate::layers::binary_ops) row_start: usize,
    pub(in crate::layers::binary_ops) rows: usize,
    pub(in crate::layers::binary_ops) lower_a_a: &'a mut [f32],
    pub(in crate::layers::binary_ops) lower_a_b: &'a mut [f32],
    pub(in crate::layers::binary_ops) upper_a_a: &'a mut [f32],
    pub(in crate::layers::binary_ops) upper_a_b: &'a mut [f32],
    pub(in crate::layers::binary_ops) lower_s_a: &'a mut [f64],
    pub(in crate::layers::binary_ops) lower_s_b: &'a mut [f64],
    pub(in crate::layers::binary_ops) upper_s_a: &'a mut [f64],
    pub(in crate::layers::binary_ops) upper_s_b: &'a mut [f64],
    pub(in crate::layers::binary_ops) lower_b_total: &'a mut [f64],
    pub(in crate::layers::binary_ops) upper_b_total: &'a mut [f64],
}

/// The row-parallel kernels' accumulators: the SAME ten arrays the serial nest
/// allocates (zero-initialized, row-major), held as flat vectors so chunks of
/// rows can be handed to workers as disjoint mutable slices.
pub(in crate::layers::binary_ops) struct McCormickRowparBuffers {
    num_outputs: usize,
    total_a_size: usize,
    total_b_size: usize,
    lower_a_a: Vec<f32>,
    lower_a_b: Vec<f32>,
    upper_a_a: Vec<f32>,
    upper_a_b: Vec<f32>,
    lower_s_a: Vec<f64>,
    lower_s_b: Vec<f64>,
    upper_s_a: Vec<f64>,
    upper_s_b: Vec<f64>,
    lower_b_total: Vec<f64>,
    upper_b_total: Vec<f64>,
}

impl McCormickRowparBuffers {
    /// Zeroed accumulators, or `None` for a degenerate shape (no rows, or an
    /// empty operand) that the serial nest handles on its own.
    pub(in crate::layers::binary_ops) fn new(
        num_outputs: usize,
        total_a_size: usize,
        total_b_size: usize,
    ) -> Option<Self> {
        if num_outputs == 0 || total_a_size == 0 || total_b_size == 0 {
            return None;
        }
        let cells_a = num_outputs.checked_mul(total_a_size)?;
        let cells_b = num_outputs.checked_mul(total_b_size)?;
        Some(Self {
            num_outputs,
            total_a_size,
            total_b_size,
            lower_a_a: vec![0.0; cells_a],
            lower_a_b: vec![0.0; cells_b],
            upper_a_a: vec![0.0; cells_a],
            upper_a_b: vec![0.0; cells_b],
            lower_s_a: vec![0.0; cells_a],
            lower_s_b: vec![0.0; cells_b],
            upper_s_a: vec![0.0; cells_a],
            upper_s_b: vec![0.0; cells_b],
            lower_b_total: vec![0.0; num_outputs],
            upper_b_total: vec![0.0; num_outputs],
        })
    }

    /// Run `body` once per chunk of rows — on rayon when nesting permits
    /// (`faer_parallelism::nested_rayon_fanout_permitted`), else sequentially
    /// over the SAME chunks. Returns whether the chunks ran in parallel
    /// (telemetry only; the values are identical either way).
    pub(in crate::layers::binary_ops) fn run<F>(&mut self, body: F) -> bool
    where
        F: Fn(&mut McCormickRowChunk<'_>) + Sync + Send,
    {
        self.run_scheduled(
            mccormick_rowpar_chunk_rows(self.num_outputs),
            crate::faer_parallelism::nested_rayon_fanout_permitted(),
            body,
        )
    }

    /// [`Self::run`] with the schedule explicit (#vit-batched-bilinear):
    /// `chunk_rows` rows per chunk (at least one), and the chunks run on rayon
    /// only when `allow_parallel` and there is more than one chunk. Rows are
    /// independent, so neither knob can change a value; the thread-count
    /// independence test drives both directly.
    pub(in crate::layers::binary_ops) fn run_scheduled<F>(
        &mut self,
        chunk_rows: usize,
        allow_parallel: bool,
        body: F,
    ) -> bool
    where
        F: Fn(&mut McCormickRowChunk<'_>) + Sync + Send,
    {
        let chunk_rows = chunk_rows.max(1);
        let total_a = self.total_a_size;
        let total_b = self.total_b_size;
        let mut lower_a_a: &mut [f32] = &mut self.lower_a_a;
        let mut lower_a_b: &mut [f32] = &mut self.lower_a_b;
        let mut upper_a_a: &mut [f32] = &mut self.upper_a_a;
        let mut upper_a_b: &mut [f32] = &mut self.upper_a_b;
        let mut lower_s_a: &mut [f64] = &mut self.lower_s_a;
        let mut lower_s_b: &mut [f64] = &mut self.lower_s_b;
        let mut upper_s_a: &mut [f64] = &mut self.upper_s_a;
        let mut upper_s_b: &mut [f64] = &mut self.upper_s_b;
        let mut lower_b_total: &mut [f64] = &mut self.lower_b_total;
        let mut upper_b_total: &mut [f64] = &mut self.upper_b_total;

        // Every buffer holds `num_outputs` rows, so each split below takes
        // exactly the next chunk's rows and never runs short.
        const ROW_SLAB: &str = "McCormick buffers hold num_outputs rows";
        let mut chunks: Vec<McCormickRowChunk<'_>> =
            Vec::with_capacity(self.num_outputs.div_ceil(chunk_rows));
        let mut row_start = 0usize;
        while row_start < self.num_outputs {
            let rows = chunk_rows.min(self.num_outputs - row_start);
            chunks.push(McCormickRowChunk {
                row_start,
                rows,
                lower_a_a: lower_a_a.split_off_mut(..rows * total_a).expect(ROW_SLAB),
                lower_a_b: lower_a_b.split_off_mut(..rows * total_b).expect(ROW_SLAB),
                upper_a_a: upper_a_a.split_off_mut(..rows * total_a).expect(ROW_SLAB),
                upper_a_b: upper_a_b.split_off_mut(..rows * total_b).expect(ROW_SLAB),
                lower_s_a: lower_s_a.split_off_mut(..rows * total_a).expect(ROW_SLAB),
                lower_s_b: lower_s_b.split_off_mut(..rows * total_b).expect(ROW_SLAB),
                upper_s_a: upper_s_a.split_off_mut(..rows * total_a).expect(ROW_SLAB),
                upper_s_b: upper_s_b.split_off_mut(..rows * total_b).expect(ROW_SLAB),
                lower_b_total: lower_b_total.split_off_mut(..rows).expect(ROW_SLAB),
                upper_b_total: upper_b_total.split_off_mut(..rows).expect(ROW_SLAB),
            });
            row_start += rows;
        }

        let parallel = chunks.len() > 1 && allow_parallel;
        if parallel {
            use rayon::prelude::*;
            chunks.par_iter_mut().for_each(&body);
        } else {
            for chunk in &mut chunks {
                body(chunk);
            }
        }
        parallel
    }

    /// Publish the two operand SHARES exactly as the serial nest's tail does:
    /// halve each widened f64 constant (exact) and cast it outward, charge
    /// `gamma_{depth+2} * S` on every coefficient cell (A side over `depth_a`
    /// = `n`, B side over `depth_b` = `m`), and build both relations through
    /// `new_or_conservative_with_err`.
    pub(in crate::layers::binary_ops) fn publish(
        self,
        depth_a: usize,
        depth_b: usize,
    ) -> Result<(LinearBounds, LinearBounds)> {
        let n_out = self.num_outputs;
        let shape_err = |what: &str| {
            NyError::InternalError(format!(
                "McCormick row-parallel {what}: buffer shape mismatch"
            ))
        };
        let lower_a_a = Array2::from_shape_vec((n_out, self.total_a_size), self.lower_a_a)
            .map_err(|_| shape_err("lower_a_a"))?;
        let lower_a_b = Array2::from_shape_vec((n_out, self.total_b_size), self.lower_a_b)
            .map_err(|_| shape_err("lower_a_b"))?;
        let upper_a_a = Array2::from_shape_vec((n_out, self.total_a_size), self.upper_a_a)
            .map_err(|_| shape_err("upper_a_a"))?;
        let upper_a_b = Array2::from_shape_vec((n_out, self.total_b_size), self.upper_a_b)
            .map_err(|_| shape_err("upper_a_b"))?;
        let lower_s_a = Array2::from_shape_vec((n_out, self.total_a_size), self.lower_s_a)
            .map_err(|_| shape_err("lower_s_a"))?;
        let lower_s_b = Array2::from_shape_vec((n_out, self.total_b_size), self.lower_s_b)
            .map_err(|_| shape_err("lower_s_b"))?;
        let upper_s_a = Array2::from_shape_vec((n_out, self.total_a_size), self.upper_s_a)
            .map_err(|_| shape_err("upper_s_a"))?;
        let upper_s_b = Array2::from_shape_vec((n_out, self.total_b_size), self.upper_s_b)
            .map_err(|_| shape_err("upper_s_b"))?;
        let lower_b_total = Array1::from_vec(self.lower_b_total);
        let upper_b_total = Array1::from_vec(self.upper_b_total);

        // Verbatim the serial tail (see the nest above for the argument).
        let lower_b_half = lower_b_total.mapv(|v| next_down_f32((v * 0.5) as f32));
        let upper_b_half = upper_b_total.mapv(|v| next_up_f32((v * 0.5) as f32));
        let gamma_a = crate::layers::linear::crown_single_gamma_n_f32(depth_a.saturating_add(2));
        let gamma_b = crate::layers::linear::crown_single_gamma_n_f32(depth_b.saturating_add(2));
        let lower_a_a_err = lower_s_a.mapv(|s| next_up_f32((gamma_a * s) as f32));
        let upper_a_a_err = upper_s_a.mapv(|s| next_up_f32((gamma_a * s) as f32));
        let lower_a_b_err = lower_s_b.mapv(|s| next_up_f32((gamma_b * s) as f32));
        let upper_a_b_err = upper_s_b.mapv(|s| next_up_f32((gamma_b * s) as f32));

        let bounds_a = LinearBounds::new_or_conservative_with_err(
            lower_a_a,
            lower_b_half.clone(),
            upper_a_a,
            upper_b_half.clone(),
            lower_a_a_err,
            upper_a_a_err,
        )?;
        let bounds_b = LinearBounds::new_or_conservative_with_err(
            lower_a_b,
            lower_b_half,
            upper_a_b,
            upper_b_half,
            lower_a_b_err,
            upper_a_b_err,
        )?;
        Ok((bounds_a, bounds_b))
    }
}

/// The four planes `select_mccormick_plane` can return for one (b, i, j, l)
/// element of the midpoint arm — `[Lower w>=0, Lower w<0, Upper w>=0,
/// Upper w<0]` — from the nest's own midpoint anchors. The plane reads the
/// weight only through `w >= 0.0`, so `1.0` / `-1.0` stand for every weight of
/// that sign (see the block comment above `McCormickRowparShape`). Also the
/// GEMM-shaped twin's plane-table source (`crown_gemm.rs`).
#[inline]
pub(super) fn hoisted_midpoint_planes(lx: f32, ux: f32, ly: f32, uy: f32) -> [(f32, f32, f32); 4] {
    // Bit-identical McCormick anchors (the nest's own expressions).
    #[allow(clippy::manual_midpoint)]
    let x0 = (lx + ux) * 0.5;
    #[allow(clippy::manual_midpoint)]
    let y0 = (ly + uy) * 0.5;
    [
        select_mccormick_plane(lx, ux, ly, uy, x0, y0, 1.0, BoundDir::Lower),
        select_mccormick_plane(lx, ux, ly, uy, x0, y0, -1.0, BoundDir::Lower),
        select_mccormick_plane(lx, ux, ly, uy, x0, y0, 1.0, BoundDir::Upper),
        select_mccormick_plane(lx, ux, ly, uy, x0, y0, -1.0, BoundDir::Upper),
    ]
}

/// The serial nest's per-row constant-fold state, field for field.
struct DenseConstFold {
    lower: f64,
    upper: f64,
    lower_s: f64,
    upper_s: f64,
    lower_terms_s: f64,
    upper_terms_s: f64,
    lower_adds: usize,
    upper_adds: usize,
}

impl DenseConstFold {
    fn new(lower_bias: f32, upper_bias: f32) -> Self {
        let lower = lower_bias as f64;
        let upper = upper_bias as f64;
        Self {
            lower,
            upper,
            lower_s: lower.abs(),
            upper_s: upper.abs(),
            lower_terms_s: 0.0,
            upper_terms_s: 0.0,
            lower_adds: 0,
            upper_adds: 0,
        }
    }
}

impl MatMulLayer {
    /// The row-parallel, plane-hoisted twin of the serial nest in
    /// [`Self::propagate_linear_binary_with_kernels`] — BIT-IDENTICAL to it (see
    /// the block comment above `McCormickRowparShape`).
    ///
    /// Returns `Ok(None)` (the caller then runs the serial nest) for any shape
    /// it does not handle: no rows, an empty operand, or an operand/coefficient
    /// buffer whose length disagrees with the parsed dims.
    pub(in crate::layers::binary_ops) fn propagate_linear_binary_rowpar(
        &self,
        bounds: &LinearBounds,
        input_a_bounds: &BoundedTensor,
        input_b_bounds: &BoundedTensor,
        shape: &McCormickRowparShape,
        scale: f32,
    ) -> Result<Option<(LinearBounds, LinearBounds)>> {
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
            && Some(total_c) == shape.batch_size.checked_mul(shape.c_size_per_batch);
        if !lengths_agree {
            return Ok(None);
        }
        let (a_lo, a_hi, b_lo, b_hi) = (&*a_lo, &*a_hi, &*b_lo, &*b_hi);
        let (w_lo, w_hi, bias_lo, bias_hi) = (&*w_lo, &*w_hi, &*bias_lo, &*bias_hi);
        let total_a = shape.total_a_size;
        let total_b = shape.total_b_size;
        let k = shape.k;

        let parallel = buffers.run(|chunk: &mut McCormickRowChunk<'_>| {
            let mut folds: Vec<DenseConstFold> = (0..chunk.rows)
                .map(|r| {
                    DenseConstFold::new(bias_lo[chunk.row_start + r], bias_hi[chunk.row_start + r])
                })
                .collect();
            // planes[l] = [Lower w>=0, Lower w<0, Upper w>=0, Upper w<0].
            let mut planes: Vec<[(f32, f32, f32); 4]> = vec![[(0.0, 0.0, 0.0); 4]; k];
            for batch_idx in 0..shape.batch_size {
                for i in 0..shape.m {
                    for j in 0..shape.n {
                        for (l, slot) in planes.iter_mut().enumerate() {
                            let a_flat = shape.a_flat(batch_idx, i, l);
                            let b_flat = shape.b_flat(batch_idx, j, l);
                            *slot = hoisted_midpoint_planes(
                                a_lo[a_flat],
                                a_hi[a_flat],
                                b_lo[b_flat],
                                b_hi[b_flat],
                            );
                        }
                        let c_flat = shape.c_flat(batch_idx, i, j);
                        for (r, fold) in folds.iter_mut().enumerate() {
                            let row = chunk.row_start + r;
                            let w_lower = w_lo[row * total_c + c_flat] * scale;
                            let w_upper = w_hi[row * total_c + c_flat] * scale;
                            // Both guards below are `w != 0.0`; a row with both
                            // weights zero touches nothing in the nest either.
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
                            for (l, plane) in planes.iter().enumerate() {
                                let a_flat = shape.a_flat(batch_idx, i, l);
                                let b_flat = shape.b_flat(batch_idx, j, l);
                                if w_lower != 0.0 {
                                    let (ax, ay, c) =
                                        if w_lower >= 0.0 { plane[0] } else { plane[1] };
                                    la_a[a_flat] += w_lower * ax;
                                    la_b[b_flat] += w_lower * ay;
                                    ls_a[a_flat] += (w_lower as f64).abs() * (ax as f64).abs();
                                    ls_b[b_flat] += (w_lower as f64).abs() * (ay as f64).abs();
                                    let term_lower = w_lower as f64 * c as f64;
                                    fold.lower += term_lower;
                                    fold.lower_s += term_lower.abs();
                                    fold.lower_terms_s += term_lower.abs();
                                    fold.lower_adds += 1;
                                }
                                if w_upper != 0.0 {
                                    let (ax, ay, c) =
                                        if w_upper >= 0.0 { plane[2] } else { plane[3] };
                                    ua_a[a_flat] += w_upper * ax;
                                    ua_b[b_flat] += w_upper * ay;
                                    us_a[a_flat] += (w_upper as f64).abs() * (ax as f64).abs();
                                    us_b[b_flat] += (w_upper as f64).abs() * (ay as f64).abs();
                                    let term_upper = w_upper as f64 * c as f64;
                                    fold.upper += term_upper;
                                    fold.upper_s += term_upper.abs();
                                    fold.upper_terms_s += term_upper.abs();
                                    fold.upper_adds += 1;
                                }
                            }
                        }
                    }
                }
            }
            for (r, fold) in folds.iter().enumerate() {
                chunk.lower_b_total[r] = widen_mccormick_constant(
                    fold.lower,
                    fold.lower_s,
                    fold.lower_terms_s,
                    fold.lower_adds,
                    BoundDir::Lower,
                );
                chunk.upper_b_total[r] = widen_mccormick_constant(
                    fold.upper,
                    fold.upper_s,
                    fold.upper_terms_s,
                    fold.upper_adds,
                    BoundDir::Upper,
                );
            }
        });
        let published = buffers.publish(shape.n, shape.m)?;
        crate::layers::collector_kernels::record_bilinear_rowpar(
            false,
            num_outputs,
            parallel,
            started.elapsed(),
        );
        Ok(Some(published))
    }
}

#[cfg(test)]
mod mccormick_constant_certificate_tests {
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

    use super::MatMulLayer;
    use crate::LinearBounds;

    /// The accumulated McCormick envelope CONSTANT must be published with its own
    /// certified error term (#matmul-dense-mccormick-const).
    ///
    /// COUNTEREXAMPLE. `A` is `[1, K]`, `B` is `[K, 1]`, both DEGENERATE boxes
    /// (`lower == upper`), so all four McCormick planes coincide, plane selection is
    /// irrelevant, and the relaxation is exact at the single feasible point. With
    /// `m = n = 1` each coefficient cell receives exactly ONE term, so no
    /// coefficient accumulation error exists and the containment condition reduces
    /// to a statement about the constant alone:
    ///
    /// ```text
    /// crown_lower(X, Y) = sum_l (w*Y_l)*X_l + sum_l (w*X_l)*Y_l + const
    ///                   = 2*sum_l X_l*Y_l + const          (w = 1)
    /// objective         =   sum_l X_l*Y_l
    /// sound  <=>  const <= -sum_l X_l*Y_l
    /// ```
    ///
    /// The taps are chosen so the f64 fold the production code performs is
    /// catastrophically wrong: terms `+2^60`, then 4096 x `-1`, then `-2^60`. Every
    /// `2^60 - 1` rounds back to `2^60` (f64 ULP there is 256), so the accumulator
    /// returns EXACTLY 0 while the true constant is `-4096`. The pre-fix code
    /// published that 0 with only the f64 -> f32 halving cast to protect it, i.e. a
    /// constant `4096` too HIGH — a lower relaxation that sits strictly ABOVE the
    /// bilinear surface at a feasible point. This assertion FAILS on the pre-fix
    /// code.
    #[ntest::timeout(30000)]
    #[test]
    fn accumulated_envelope_constant_carries_its_higham_charge() {
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
        let objective: i128 = (0..K).map(|l| (xs[l] as f64 * ys[l] as f64) as i128).sum();
        assert_eq!(objective, 4096, "true objective sum_l X_l*Y_l");
        let mut folded = 0.0_f64;
        for l in 0..K {
            // Exactly what the production loop accumulates: w = 1, c = fl32(-X*Y),
            // and f32 x f32 is exact in f64.
            folded += -(xs[l] as f64) * (ys[l] as f64);
        }
        assert_eq!(
            folded, 0.0,
            "the f64 constant fold must cancel to 0 for this to exercise the defect"
        );

        let a = ArrayD::from_shape_vec(IxDyn(&[1, K]), xs.clone()).expect("a shape");
        let b = ArrayD::from_shape_vec(IxDyn(&[K, 1]), ys.clone()).expect("b shape");
        let input_a = BoundedTensor::new(a.clone(), a).expect("input_a");
        let input_b = BoundedTensor::new(b.clone(), b).expect("input_b");

        let matmul = MatMulLayer::new(false, None);
        let (bounds_a, bounds_b) = matmul
            .propagate_linear_binary(&LinearBounds::identity(1), &input_a, &input_b)
            .expect("dense McCormick CROWN backward");

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

        let published = bounds_a.lower_b[0] as f64 + bounds_b.lower_b[0] as f64;
        assert!(
            published <= -(objective as f64),
            "published McCormick lower constant {published} must be <= {} — otherwise \
             the lower relaxation sits ABOVE the bilinear surface at a feasible point",
            -(objective as f64)
        );
    }

    /// A fold with NO McCormick term performs NO f64 addition and consumes NO plane,
    /// so there is genuinely nothing to charge: the constant must stay bit-identical
    /// to the historical directed-halving result.
    #[ntest::timeout(30000)]
    #[test]
    fn term_free_constant_fold_stays_bit_identical() {
        let matmul = MatMulLayer::new(false, None);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), 0.5_f32),
            ArrayD::from_elem(IxDyn(&[1, 1]), 1.5_f32),
        )
        .expect("input");

        let mut bounds = LinearBounds::identity(1);
        bounds.lower_a.fill(0.0);
        bounds.upper_a.fill(0.0);
        // Subnormal seeds: "round then halve" differs from "halve in f64 then
        // directed-round" by an ULP here, so the assertion has teeth.
        let lower_seed = f32::from_bits(0x004a_e545);
        let upper_seed = f32::from_bits(0x0063_e42f);
        bounds.lower_b[0] = lower_seed;
        bounds.upper_b[0] = upper_seed;

        let (bounds_a, bounds_b) = matmul
            .propagate_linear_binary(&bounds, &input, &input)
            .expect("dense McCormick CROWN backward");

        let expected_lower = next_down_f32(((lower_seed as f64) * 0.5) as f32);
        let expected_upper = next_up_f32(((upper_seed as f64) * 0.5) as f32);
        assert_eq!(bounds_a.lower_b[0].to_bits(), expected_lower.to_bits());
        assert_eq!(bounds_b.lower_b[0].to_bits(), expected_lower.to_bits());
        assert_eq!(bounds_a.upper_b[0].to_bits(), expected_upper.to_bits());
        assert_eq!(bounds_b.upper_b[0].to_bits(), expected_upper.to_bits());
    }
}
