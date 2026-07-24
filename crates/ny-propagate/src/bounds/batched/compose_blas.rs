// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! BLAS-accelerated composition of batched linear bounds for CROWN backward.
//!
//! Uses positive/negative coefficient split + ndarray `.dot()` → BLAS SGEMM
//! (Accelerate on macOS) to replace the O(batch × out × in × k) scalar loop.
//!
//! Matching alpha-beta-CROWN backward_bound.py:
//!   lA_new = clamp(lA, min=0) @ lA_next + clamp(lA, max=0) @ uA_next
//!   uA_new = clamp(uA, min=0) @ uA_next + clamp(uA, max=0) @ lA_next
//!
//! Part of #2220 Packet C.

use super::BatchedLinearBounds;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use ny_core::{Result, CROWN_COEFF_MAX};
use ny_tensor::{next_down_f32, next_up_f32};

impl BatchedLinearBounds {
    /// Check if all coefficient and bias arrays are finite (no NaN/Inf).
    /// When true, BLAS SGEMM is safe (no 0*inf=NaN risk).
    pub(super) fn all_finite_for_compose(
        a2_lower: &ArrayView3<f32>,
        a2_upper: &ArrayView3<f32>,
        a1_lower: &ArrayView3<f32>,
        a1_upper: &ArrayView3<f32>,
        b1_lower: &ArrayView2<f32>,
        b1_upper: &ArrayView2<f32>,
        b2_lower: &ArrayView2<f32>,
        b2_upper: &ArrayView2<f32>,
    ) -> bool {
        let coeff_ok = |v: &f32| v.is_finite() && v.abs() <= CROWN_COEFF_MAX;
        a2_lower.iter().all(coeff_ok)
            && a2_upper.iter().all(coeff_ok)
            && a1_lower.iter().all(coeff_ok)
            && a1_upper.iter().all(coeff_ok)
            && b1_lower.iter().all(|v| v.is_finite())
            && b1_upper.iter().all(|v| v.is_finite())
            && b2_lower.iter().all(|v| v.is_finite())
            && b2_upper.iter().all(|v| v.is_finite())
    }

    /// BLAS-accelerated compose using positive/negative coefficient split.
    ///
    /// For each batch b:
    ///   lower_a_new[b] = pos(a2_l[b]) @ a1_l[b] + neg(a2_l[b]) @ a1_u[b]
    ///   upper_a_new[b] = pos(a2_u[b]) @ a1_u[b] + neg(a2_u[b]) @ a1_l[b]
    ///   lower_b_new[b] = pos(a2_l[b]) @ b1_l[b] + neg(a2_l[b]) @ b1_u[b] + b2_l[b]
    ///   upper_b_new[b] = pos(a2_u[b]) @ b1_u[b] + neg(a2_u[b]) @ b1_l[b] + b2_u[b]
    ///
    /// Where pos(x) = max(x, 0), neg(x) = min(x, 0).
    ///
    /// # SOUNDNESS — VERDICT-SAFE (measured-divergence + `γ·S` certified error)
    ///
    /// The coefficient SGEMM (`a2_l_pos.dot(&a1_l_b)`, etc.) accumulates the
    /// `k`-term dot product in **f32**, which can lose multiple ULPs of accumulated
    /// rounding error for `k > 2` — far more than the 1 ULP a single directed cast
    /// could cover. We keep the fast f32 SGEMM as the NOMINAL coefficient `stored`
    /// and attach a certified per-coefficient error matrix that soundly bounds
    /// `|stored − exact_real_product|`:
    ///
    /// ```text
    ///   err[i,j] = |stored − D_f64|  +  γ_{k+1}^{f64} · S[i,j]
    ///   D_f64[i,j] = Σ_k (exact f32→f64 products of the SAME sign-split terms)
    ///   S[i,j]     = Σ_k |c2[i,k]|·|c1[k,j]|       (absolute-product sum, f64)
    /// ```
    ///
    /// By the triangle inequality `|stored − exact| ≤ |stored − D_f64| + |D_f64 −
    /// exact|`: the first term is the EXACTLY-measured divergence between the f32
    /// SGEMM and an independent f64 re-accumulation of the same products (both
    /// finite), and the second is Higham's f64 dot bound `γ_k^{f64}·S` on the f64
    /// reference's own sub-ULP summation error (each f32→f64 product is exact). This
    /// is **order-independent** — it makes NO assumption about the opaque BLAS f32
    /// accumulation order (blocking / FMA / pairwise), so it cannot be defeated by a
    /// different SGEMM summation strategy the way a tight f32 `γ_k·S` could. It is the
    /// same explicit `cast_err + γ·S` construction `crown_batched` uses for its
    /// f64-accumulated `A·W`. The error is rounded UP to a sound f32, returned, and
    /// attached via [`set_coeff_err`](BatchedLinearBounds::set_coeff_err); concretize
    /// then penalizes the result OUTWARD by `Σ_j max(|x_l|,|x_u|)·err[i,j]`, so the
    /// effective certified coefficient interval `[stored−err, stored+err]` encloses
    /// the exact real product `(A2·A1)[i,j]` for every coefficient.
    ///
    /// The bias composition accumulates in **f64** directly (mirroring
    /// `compose_scalar`), so its only rounding is the final f32 cast, covered by
    /// the directed `next_down_f32`/`next_up_f32` — no bias error term is needed.
    ///
    /// Returns `(lower_a, upper_a, lower_b, upper_b, lower_a_err, upper_a_err)`.
    #[allow(clippy::type_complexity)]
    pub(super) fn compose_blas(
        a2_lower: &ArrayView3<f32>,
        a2_upper: &ArrayView3<f32>,
        a1_lower: &ArrayView3<f32>,
        a1_upper: &ArrayView3<f32>,
        b1_lower: &ArrayView2<f32>,
        b1_upper: &ArrayView2<f32>,
        b2_lower: &ArrayView2<f32>,
        b2_upper: &ArrayView2<f32>,
        batch_size: usize,
        other_out_dim: usize,
        self_in_dim: usize,
    ) -> Result<(
        Array3<f32>,
        Array3<f32>,
        Array2<f32>,
        Array2<f32>,
        Array3<f32>,
        Array3<f32>,
    )> {
        let mut composed_lower_a = Array3::<f32>::zeros((batch_size, other_out_dim, self_in_dim));
        let mut composed_upper_a = Array3::<f32>::zeros((batch_size, other_out_dim, self_in_dim));
        let mut composed_lower_b = Array2::<f32>::zeros((batch_size, other_out_dim));
        let mut composed_upper_b = Array2::<f32>::zeros((batch_size, other_out_dim));
        let mut composed_lower_a_err =
            Array3::<f32>::zeros((batch_size, other_out_dim, self_in_dim));
        let mut composed_upper_a_err =
            Array3::<f32>::zeros((batch_size, other_out_dim, self_in_dim));

        // Contraction length is `other_in_dim` (== `self_out_dim`), i.e. the shared
        // axis k of A2 [out, k] · A1 [k, in]. We re-accumulate the SAME sign-split dot
        // in f64 (`compose_dot_f64_with_abssum`) and bound the certified error by the
        // MEASURED divergence between the f32 SGEMM and that f64 reference, PLUS the
        // f64 reference's own (tiny) Higham accumulation error `γ_k^{f64}·S` — see the
        // error loop below. The f64 factor is the right one for that residual since
        // the reference is f64-accumulated. `γ_{k+1}` would NOT be a sound bound on
        // the OPAQUE BLAS f32 accumulation order; measuring the divergence directly
        // is order-independent and bulletproof.
        let contraction = a1_lower.shape()[1];
        let gamma_k =
            crate::layers::linear::crown_single_gamma_n_f64(contraction.saturating_add(1));

        for b in 0..batch_size {
            // Extract batch slices: [out_dim, k] and [k, in_dim]
            let a2_l_b = a2_lower.index_axis(ndarray::Axis(0), b);
            let a2_u_b = a2_upper.index_axis(ndarray::Axis(0), b);
            let a1_l_b = a1_lower.index_axis(ndarray::Axis(0), b);
            let a1_u_b = a1_upper.index_axis(ndarray::Axis(0), b);

            // Pos/neg split: max(x, 0) and min(x, 0)
            let a2_l_pos = a2_l_b.mapv(|v| v.max(0.0));
            let a2_l_neg = a2_l_b.mapv(|v| v.min(0.0));
            let a2_u_pos = a2_u_b.mapv(|v| v.max(0.0));
            let a2_u_neg = a2_u_b.mapv(|v| v.min(0.0));

            // BLAS SGEMM: coefficient composition (2 calls per bound direction)
            // lower_a = pos(a2_l) @ a1_l + neg(a2_l) @ a1_u
            let lower_a_b = a2_l_pos.dot(&a1_l_b) + a2_l_neg.dot(&a1_u_b);
            // upper_a = pos(a2_u) @ a1_u + neg(a2_u) @ a1_l
            let upper_a_b = a2_u_pos.dot(&a1_u_b) + a2_u_neg.dot(&a1_l_b);

            // f64 reference dots `D` and certified absolute-product sums `S`. For (i,j):
            //   D_lower[i,j] = Σ_k pos(a2_l[i,k])·a1_l[k,j] + neg(a2_l[i,k])·a1_u[k,j]  (f64)
            //   S_lower[i,j] = Σ_k |a2_l[i,k]|·|(a1_l or a1_u)[k,j]|                     (f64)
            // selecting the SAME a1 operand the sign-split SGEMM summed (pos(a2)
            // multiplies a1_l for lower / a1_u for upper, neg(a2) the opposite). Since
            // pos(a2[i,k]) and neg(a2[i,k]) are disjoint (one is 0), exactly one a1
            // operand is selected per k and |term| = |a2[i,k]|·|a1_sel[k,j]|.
            let (lower_d, lower_s, upper_d, upper_s) = Self::compose_dot_f64_with_abssum(
                &a2_l_b,
                &a2_u_b,
                &a1_l_b,
                &a1_u_b,
                other_out_dim,
                self_in_dim,
                contraction,
            );

            // Store the round-to-nearest f32 SGEMM result DIRECTLY as the nominal
            // coefficient (NO `next_down_f32`/`next_up_f32` nudge) and make the
            // certified error
            //
            //   err[i,j] = |fl_f32(SGEMM) − D_f64|  +  γ_{k+1}^{f64}·S
            //
            // a SOUND bound on `|stored − exact_real|`, by the triangle inequality:
            //   |stored − exact| ≤ |stored − D_f64| + |D_f64 − exact|,
            // where the first term is computed EXACTLY (f32 SGEMM vs f64 reference,
            // both finite) and the second is Higham's f64 dot bound `γ_k^{f64}·S`
            // (the f64 reference's only error is its sub-ULP f64 summation). This is
            // order-independent — it makes NO assumption about the opaque BLAS f32
            // accumulation order, so it cannot be defeated by a different SGEMM
            // summation/FMA strategy (the failure mode a tight f32 `γ_k` would risk).
            // Mirrors `crown_batched`'s explicit `l_cast_err + γ·S` construction.
            // NOTE: applying `next_up`/`next_down` to the stored coefficient would be
            // UNSOUND here — it shifts `stored` away from `fl(SGEMM)` and the symmetric
            // `±err` interval concretize applies is measured around `stored`.
            for i in 0..other_out_dim {
                for j in 0..self_in_dim {
                    let s_l = lower_a_b[[i, j]];
                    let s_u = upper_a_b[[i, j]];
                    composed_lower_a[[b, i, j]] = s_l;
                    composed_upper_a[[b, i, j]] = s_u;
                    let div_l = (s_l as f64 - lower_d[[i, j]]).abs();
                    let div_u = (s_u as f64 - upper_d[[i, j]]).abs();
                    composed_lower_a_err[[b, i, j]] =
                        next_up_f32((div_l + gamma_k * lower_s[[i, j]]) as f32);
                    composed_upper_a_err[[b, i, j]] =
                        next_up_f32((div_u + gamma_k * upper_s[[i, j]]) as f32);
                }
            }

            // Bias composition: accumulate the k-term dot in f64 (exact f32 products,
            // only the f64 sum rounds sub-ULP), add b2, directed rounding on final
            // cast. Mirrors compose_scalar — no separate bias error term needed.
            let b1_l_b = b1_lower.index_axis(ndarray::Axis(0), b);
            let b1_u_b = b1_upper.index_axis(ndarray::Axis(0), b);

            for i in 0..other_out_dim {
                let mut sum_l = b2_lower[[b, i]] as f64;
                let mut sum_u = b2_upper[[b, i]] as f64;
                for k in 0..contraction {
                    let a2l = a2_l_b[[i, k]] as f64;
                    let a2u = a2_u_b[[i, k]] as f64;
                    // lower: pos(a2_l)·b1_l + neg(a2_l)·b1_u
                    sum_l += if a2l >= 0.0 {
                        a2l * b1_l_b[k] as f64
                    } else {
                        a2l * b1_u_b[k] as f64
                    };
                    // upper: pos(a2_u)·b1_u + neg(a2_u)·b1_l
                    sum_u += if a2u >= 0.0 {
                        a2u * b1_u_b[k] as f64
                    } else {
                        a2u * b1_l_b[k] as f64
                    };
                }
                composed_lower_b[[b, i]] = next_down_f32(sum_l as f32);
                composed_upper_b[[b, i]] = next_up_f32(sum_u as f32);
            }
        }

        Ok((
            composed_lower_a,
            composed_upper_a,
            composed_lower_b,
            composed_upper_b,
            composed_lower_a_err,
            composed_upper_a_err,
        ))
    }

    /// Per-coefficient f64 reference dot `D[i,j]` and absolute-product sum
    /// `S[i,j] = Σ_k |a2[i,k]|·|a1_sel[k,j]|` for the lower and upper sign-split
    /// coefficient SGEMMs. `a1_sel` is the SAME a1 operand the SGEMM summed for that
    /// sign: lower uses `a1_l` where `a2_l[i,k] >= 0` and `a1_u` otherwise; upper uses
    /// `a1_u` where `a2_u[i,k] >= 0` and `a1_l` otherwise. Each f32→f64 product is
    /// EXACT (24+24 bits < 53), so the f64 dot `D` differs from the exact real product
    /// by at most `γ_k^{f64}·S` (Higham), and `|a2|·|a1|` is the magnitude of every
    /// term, so `γ_n·S` upper-bounds the residual regardless of cancellation. Returns
    /// `(lower_d, lower_s, upper_d, upper_s)`.
    #[allow(clippy::too_many_arguments)]
    fn compose_dot_f64_with_abssum(
        a2_l_b: &ArrayView2<f32>,
        a2_u_b: &ArrayView2<f32>,
        a1_l_b: &ArrayView2<f32>,
        a1_u_b: &ArrayView2<f32>,
        other_out_dim: usize,
        self_in_dim: usize,
        contraction: usize,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let mut lower_d = Array2::<f64>::zeros((other_out_dim, self_in_dim));
        let mut lower_s = Array2::<f64>::zeros((other_out_dim, self_in_dim));
        let mut upper_d = Array2::<f64>::zeros((other_out_dim, self_in_dim));
        let mut upper_s = Array2::<f64>::zeros((other_out_dim, self_in_dim));
        for i in 0..other_out_dim {
            for j in 0..self_in_dim {
                let mut dl = 0.0f64;
                let mut sl = 0.0f64;
                let mut du = 0.0f64;
                let mut su = 0.0f64;
                for k in 0..contraction {
                    let a2l = a2_l_b[[i, k]] as f64;
                    let a2u = a2_u_b[[i, k]] as f64;
                    // lower: pos(a2_l)·a1_l + neg(a2_l)·a1_u — select a1_l if a2_l>=0.
                    let a1_for_l = if a2l >= 0.0 {
                        a1_l_b[[k, j]] as f64
                    } else {
                        a1_u_b[[k, j]] as f64
                    };
                    dl += a2l * a1_for_l;
                    sl += a2l.abs() * a1_for_l.abs();
                    // upper: pos(a2_u)·a1_u + neg(a2_u)·a1_l — select a1_u if a2_u>=0.
                    let a1_for_u = if a2u >= 0.0 {
                        a1_u_b[[k, j]] as f64
                    } else {
                        a1_l_b[[k, j]] as f64
                    };
                    du += a2u * a1_for_u;
                    su += a2u.abs() * a1_for_u.abs();
                }
                lower_d[[i, j]] = dl;
                lower_s[[i, j]] = sl;
                upper_d[[i, j]] = du;
                upper_s[[i, j]] = su;
            }
        }
        (lower_d, lower_s, upper_d, upper_s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, ArrayD, IxDyn};

    /// Helper to create BatchedLinearBounds from 2D arrays (batch_size=1).
    fn make_bounds(
        la: Array2<f32>,
        lb: ndarray::Array1<f32>,
        ua: Array2<f32>,
        ub: ndarray::Array1<f32>,
        in_shape: Vec<usize>,
        out_shape: Vec<usize>,
    ) -> BatchedLinearBounds {
        let m = la.nrows();
        let n = la.ncols();
        let (la_vec, _) = la.into_raw_vec_and_offset();
        let (lb_vec, _) = lb.into_raw_vec_and_offset();
        let (ua_vec, _) = ua.into_raw_vec_and_offset();
        let (ub_vec, _) = ub.into_raw_vec_and_offset();
        BatchedLinearBounds::from_parts_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[m, n]), la_vec).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[m]), lb_vec).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[m, n]), ua_vec).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[m]), ub_vec).unwrap(),
            in_shape,
            out_shape,
        )
    }

    #[test]
    fn test_compose_blas_matches_scalar_identity() {
        // A2 = identity → composed should equal A1
        let a1 = make_bounds(
            array![[1.0, 2.0], [3.0, 4.0]],
            array![0.5, 0.6],
            array![[1.0, 2.0], [3.0, 4.0]],
            array![0.5, 0.6],
            vec![2],
            vec![2],
        );
        let a2 = make_bounds(
            array![[1.0, 0.0], [0.0, 1.0]],
            array![0.0, 0.0],
            array![[1.0, 0.0], [0.0, 1.0]],
            array![0.0, 0.0],
            vec![2],
            vec![2],
        );
        let result = a1.compose(&a2).unwrap();
        // Result should be close to a1 (with 1-ULP directed rounding)
        for (&r, &e) in result.lower_a().iter().zip(a1.lower_a().iter()) {
            assert!((r - e).abs() < 1e-6, "lower_a mismatch: {r} vs {e}");
        }
        for (&r, &e) in result.lower_b().iter().zip(a1.lower_b().iter()) {
            assert!((r - e).abs() < 1e-6, "lower_b mismatch: {r} vs {e}");
        }
    }

    #[test]
    fn test_compose_blas_mixed_sign_coefficients() {
        // Test with mixed positive and negative coefficients
        let a1 = make_bounds(
            array![[1.0, -2.0], [3.0, -4.0]],
            array![0.1, 0.2],
            array![[2.0, -1.0], [4.0, -3.0]],
            array![0.3, 0.4],
            vec![2],
            vec![2],
        );
        let a2 = make_bounds(
            array![[1.0, -1.0], [0.5, 0.5]],
            array![0.0, 0.0],
            array![[2.0, -0.5], [1.0, 1.0]],
            array![0.0, 0.0],
            vec![2],
            vec![2],
        );
        // Both BLAS and scalar should produce results — just verify no panic/NaN
        let result = a1.compose(&a2).unwrap();
        assert!(result.lower_a().iter().all(|v| !v.is_nan()));
        assert!(result.upper_a().iter().all(|v| !v.is_nan()));
        assert!(result.lower_b().iter().all(|v| !v.is_nan()));
        assert!(result.upper_b().iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn test_compose_blas_soundness_lower_le_upper() {
        // Lower bounds should always be <= upper bounds element-wise
        let a1 = make_bounds(
            array![[1.0, -2.0, 0.5], [3.0, -4.0, 1.0]],
            array![0.1, 0.2],
            array![[2.0, -1.0, 1.5], [4.0, -3.0, 2.0]],
            array![0.3, 0.4],
            vec![3],
            vec![2],
        );
        let a2 = make_bounds(
            array![[-0.5, 1.0], [0.5, -1.0], [1.0, 0.0]],
            array![0.0, 0.0, 0.0],
            array![[0.5, 2.0], [1.5, 0.0], [2.0, 1.0]],
            array![0.0, 0.0, 0.0],
            vec![2],
            vec![3],
        );
        let result = a1.compose(&a2).unwrap();
        // Coefficient matrices don't have strict lower <= upper ordering
        // (they're independent linear functions), but biases should be ordered.
        for (&l, &u) in result.lower_b().iter().zip(result.upper_b().iter()) {
            assert!(
                l <= u || l.is_nan() || u.is_nan(),
                "lower_b {l} > upper_b {u}"
            );
        }
    }
}
