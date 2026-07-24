// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Matrix multiplication, dot product, and elementwise operations for MultiNormBounds.

use ndarray::{s, Array2, Array3, Array4, Ix3};
use ny_core::{nan_propagating_max_zero, nan_propagating_min_zero, NyError, Result};

use super::MultiNormBounds;

impl MultiNormBounds {
    /// Linear layer: bounds @ weight (shape: in_features x out_features).
    pub fn matmul(&self, weight: &Array2<f32>) -> Result<Self> {
        let (batch, length, dim_in, dim_out) = self.lw.dim();
        if dim_out != weight.shape()[0] {
            return Err(NyError::shape_mismatch(
                vec![dim_out, weight.shape()[1]],
                weight.shape().to_vec(),
            ));
        }
        let out_features = weight.shape()[1];
        // NaN-propagating split: IEEE 754 v.max(0.0) silently zeroes NaN weights,
        // producing unsound (too-tight) bounds. See ny-core::nan_math, #2415, #3116.
        let w_pos = weight.mapv(nan_propagating_max_zero);
        let w_neg = weight.mapv(nan_propagating_min_zero);

        let lw_flat = self
            .lw
            .view()
            .into_shape_with_order((batch * length * dim_in, dim_out))
            .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
        let uw_flat = self
            .uw
            .view()
            .into_shape_with_order((batch * length * dim_in, dim_out))
            .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
        let lb_flat = self
            .lb
            .view()
            .into_shape_with_order((batch * length, dim_out))
            .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
        let ub_flat = self
            .ub
            .view()
            .into_shape_with_order((batch * length, dim_out))
            .map_err(|e| NyError::InvalidSpec(e.to_string()))?;

        let lw_new: Array2<f32> = lw_flat.dot(&w_pos) + uw_flat.dot(&w_neg);
        let uw_new: Array2<f32> = lw_flat.dot(&w_neg) + uw_flat.dot(&w_pos);
        let lb_new: Array2<f32> = lb_flat.dot(&w_pos) + ub_flat.dot(&w_neg);
        let ub_new: Array2<f32> = lb_flat.dot(&w_neg) + ub_flat.dot(&w_pos);

        let lw_new = lw_new
            .into_shape_with_order((batch, length, dim_in, out_features))
            .map_err(|e: ndarray::ShapeError| NyError::InvalidSpec(e.to_string()))?;
        let uw_new = uw_new
            .into_shape_with_order((batch, length, dim_in, out_features))
            .map_err(|e: ndarray::ShapeError| NyError::InvalidSpec(e.to_string()))?;
        let lb_new = lb_new
            .into_shape_with_order((batch, length, out_features))
            .map_err(|e: ndarray::ShapeError| NyError::InvalidSpec(e.to_string()))?;
        let ub_new = ub_new
            .into_shape_with_order((batch, length, out_features))
            .map_err(|e: ndarray::ShapeError| NyError::InvalidSpec(e.to_string()))?;

        Ok(Self {
            p: self.p,
            q: self.q,
            eps: self.eps,
            perturbed_words: self.perturbed_words,
            lw: lw_new,
            lb: lb_new,
            uw: uw_new,
            ub: ub_new,
        })
    }

    /// Batched linear layer: per-batch weight matrices (batch, dim_out, out_features).
    pub fn matmul_batched(&self, weight: &Array3<f32>) -> Result<Self> {
        let (batch, length, dim_in, dim_out) = self.lw.dim();
        if weight.dim().0 != batch || weight.dim().1 != dim_out {
            return Err(NyError::shape_mismatch(
                vec![batch, dim_out, weight.dim().2],
                weight.shape().to_vec(),
            ));
        }
        let out_features = weight.dim().2;
        let mut lw_new = Array4::<f32>::zeros((batch, length, dim_in, out_features));
        let mut uw_new = Array4::<f32>::zeros((batch, length, dim_in, out_features));
        let mut lb_new = Array3::<f32>::zeros((batch, length, out_features));
        let mut ub_new = Array3::<f32>::zeros((batch, length, out_features));

        for b in 0..batch {
            let w = weight.slice(s![b, .., ..]);
            // NaN-propagating split: see matmul() comment, #3116.
            let w_pos = w.mapv(nan_propagating_max_zero);
            let w_neg = w.mapv(nan_propagating_min_zero);

            // Use as_standard_layout() to borrow when data is contiguous
            // (the common case), avoiding heap allocation per batch iteration.
            // Ref: #3330 Finding 1 — eliminates ~577MB/batch of unnecessary churn.
            let lw_view = self.lw.slice(s![b, .., .., ..]);
            let lw_flat = lw_view
                .as_standard_layout()
                .into_shape_with_order((length * dim_in, dim_out))
                .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
            let uw_view = self.uw.slice(s![b, .., .., ..]);
            let uw_flat = uw_view
                .as_standard_layout()
                .into_shape_with_order((length * dim_in, dim_out))
                .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
            let lb_view = self.lb.slice(s![b, .., ..]);
            let lb_flat = lb_view
                .as_standard_layout()
                .into_shape_with_order((length, dim_out))
                .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
            let ub_view = self.ub.slice(s![b, .., ..]);
            let ub_flat = ub_view
                .as_standard_layout()
                .into_shape_with_order((length, dim_out))
                .map_err(|e| NyError::InvalidSpec(e.to_string()))?;

            let lw_out: Array2<f32> = lw_flat.dot(&w_pos) + uw_flat.dot(&w_neg);
            let uw_out: Array2<f32> = lw_flat.dot(&w_neg) + uw_flat.dot(&w_pos);
            let lb_out: Array2<f32> = lb_flat.dot(&w_pos) + ub_flat.dot(&w_neg);
            let ub_out: Array2<f32> = lb_flat.dot(&w_neg) + ub_flat.dot(&w_pos);

            let lw_out = lw_out
                .into_shape_with_order((length, dim_in, out_features))
                .map_err(|e: ndarray::ShapeError| NyError::InvalidSpec(e.to_string()))?;
            let uw_out = uw_out
                .into_shape_with_order((length, dim_in, out_features))
                .map_err(|e: ndarray::ShapeError| NyError::InvalidSpec(e.to_string()))?;

            lw_new.slice_mut(s![b, .., .., ..]).assign(&lw_out);
            uw_new.slice_mut(s![b, .., .., ..]).assign(&uw_out);
            lb_new.slice_mut(s![b, .., ..]).assign(&lb_out);
            ub_new.slice_mut(s![b, .., ..]).assign(&ub_out);
        }

        Ok(Self {
            p: self.p,
            q: self.q,
            eps: self.eps,
            perturbed_words: self.perturbed_words,
            lw: lw_new,
            lb: lb_new,
            uw: uw_new,
            ub: ub_new,
        })
    }

    /// Dot product for multi-head attention (DeepT-style).
    ///
    /// Returns bounds for the matrix product self @ other^T, where self has
    /// shape (batch, len_a, dim_in, dim_out) and other has shape
    /// (batch, len_b, dim_in, dim_out). The output bounds have shape
    /// (batch, len_a, dim_in, len_b) with biases (batch, len_a, len_b).
    pub fn dot_product(&self, other: &Self) -> Result<Self> {
        self.ensure_compatible(other)?;

        let (batch, length, dim_in, dim_out) = self.lw.dim();
        let (other_batch, other_length, other_dim_in, other_dim_out) = other.lw.dim();
        if batch != other_batch || dim_in != other_dim_in || dim_out != other_dim_out {
            return Err(NyError::shape_mismatch(
                vec![batch, length, dim_in, dim_out],
                vec![other_batch, other_length, other_dim_in, other_dim_out],
            ));
        }

        // Sound concretization for dot product intermediate bounds (#2239).
        let concretized_a = self.concretize_sound()?;
        let concretized_b = other.concretize_sound()?;
        let (l_a, u_a) = concretized_a.into_parts();
        let (l_b, u_b) = concretized_b.into_parts();
        let l_a = l_a
            .into_dimensionality::<Ix3>()
            .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
        let u_a = u_a
            .into_dimensionality::<Ix3>()
            .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
        let l_b = l_b
            .into_dimensionality::<Ix3>()
            .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
        let u_b = u_b
            .into_dimensionality::<Ix3>()
            .map_err(|e| NyError::InvalidSpec(e.to_string()))?;

        let mut lw = Array4::<f32>::zeros((batch, length, dim_in, other_length));
        let mut uw = Array4::<f32>::zeros((batch, length, dim_in, other_length));
        let mut lb = Array3::<f32>::zeros((batch, length, other_length));
        let mut ub = Array3::<f32>::zeros((batch, length, other_length));

        if dim_in == 1 {
            for t in 0..batch {
                for i in 0..length {
                    for j in 0..other_length {
                        let mut l_sum = 0.0f32;
                        let mut u_sum = 0.0f32;
                        for k in 0..dim_out {
                            let l1 = l_a[[t, i, k]];
                            let u1 = u_a[[t, i, k]];
                            let l2 = l_b[[t, j, k]];
                            let u2 = u_b[[t, j, k]];
                            let p1 = l1 * l2;
                            let p2 = l1 * u2;
                            let p3 = u1 * l2;
                            let p4 = u1 * u2;
                            let l = p1.min(p2).min(p3).min(p4);
                            let u = p1.max(p2).max(p3).max(p4);
                            l_sum += l;
                            u_sum += u;
                        }
                        lb[[t, i, j]] = l_sum;
                        ub[[t, i, j]] = u_sum;
                    }
                }
            }

            return Self::new(self.p, self.eps, self.perturbed_words, lw, lb, uw, ub);
        }

        for t in 0..batch {
            for i in 0..length {
                for j in 0..other_length {
                    let mut lb_acc = 0.0f32;
                    let mut ub_acc = 0.0f32;
                    let mut lb_alpha = 0.0f32;
                    let mut lb_beta = 0.0f32;
                    let mut ub_alpha = 0.0f32;
                    let mut ub_beta = 0.0f32;

                    for k in 0..dim_out {
                        let l_x = l_a[[t, i, k]];
                        let l_y = l_b[[t, j, k]];
                        let u_y = u_b[[t, j, k]];

                        let alpha_l = l_y;
                        let beta_l = l_x;
                        let ny_l = -alpha_l * beta_l;
                        let alpha_u = u_y;
                        let beta_u = l_x;
                        let ny_u = -alpha_u * beta_u;

                        lb_acc += ny_l;
                        ub_acc += ny_u;

                        let alpha_l_pos = if alpha_l > 0.0 { alpha_l } else { 0.0 };
                        let alpha_l_neg = if alpha_l < 0.0 { alpha_l } else { 0.0 };
                        let beta_l_pos = if beta_l > 0.0 { beta_l } else { 0.0 };
                        let beta_l_neg = if beta_l < 0.0 { beta_l } else { 0.0 };
                        let alpha_u_pos = if alpha_u > 0.0 { alpha_u } else { 0.0 };
                        let alpha_u_neg = if alpha_u < 0.0 { alpha_u } else { 0.0 };
                        let beta_u_pos = if beta_u > 0.0 { beta_u } else { 0.0 };
                        let beta_u_neg = if beta_u < 0.0 { beta_u } else { 0.0 };

                        lb_alpha += if alpha_l >= 0.0 {
                            self.lb[[t, i, k]] * alpha_l
                        } else {
                            self.ub[[t, i, k]] * alpha_l
                        };
                        lb_beta += if beta_l >= 0.0 {
                            other.lb[[t, j, k]] * beta_l
                        } else {
                            other.ub[[t, j, k]] * beta_l
                        };
                        ub_alpha += if alpha_u >= 0.0 {
                            self.ub[[t, i, k]] * alpha_u
                        } else {
                            self.lb[[t, i, k]] * alpha_u
                        };
                        ub_beta += if beta_u >= 0.0 {
                            other.ub[[t, j, k]] * beta_u
                        } else {
                            other.lb[[t, j, k]] * beta_u
                        };

                        for m in 0..dim_in {
                            lw[[t, i, m, j]] += alpha_l_pos * self.lw[[t, i, m, k]]
                                + alpha_l_neg * self.uw[[t, i, m, k]];
                            lw[[t, i, m, j]] += beta_l_pos * other.lw[[t, j, m, k]]
                                + beta_l_neg * other.uw[[t, j, m, k]];
                            uw[[t, i, m, j]] += alpha_u_pos * self.uw[[t, i, m, k]]
                                + alpha_u_neg * self.lw[[t, i, m, k]];
                            uw[[t, i, m, j]] += beta_u_pos * other.uw[[t, j, m, k]]
                                + beta_u_neg * other.lw[[t, j, m, k]];
                        }
                    }

                    lb[[t, i, j]] = lb_acc + lb_alpha + lb_beta;
                    ub[[t, i, j]] = ub_acc + ub_alpha + ub_beta;
                }
            }
        }

        Self::new(self.p, self.eps, self.perturbed_words, lw, lb, uw, ub)
    }

    /// Elementwise multiplication with linear relaxations (DeepT-style).
    /// Uses DeepT's get_bounds_xy coefficients (alpha_l=l_y, beta_l=l_x, alpha_u=u_y, beta_u=l_x).
    pub fn mul_elementwise(&self, other: &Self) -> Result<Self> {
        self.ensure_compatible(other)?;
        // Sound concretization for elementwise mul intermediate bounds (#2239).
        let concretized_a = self.concretize_sound()?;
        let concretized_b = other.concretize_sound()?;
        let (l_a, _u_a) = concretized_a.into_parts();
        let (l_b, u_b) = concretized_b.into_parts();
        let l_a = l_a
            .into_dimensionality::<Ix3>()
            .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
        let l_b = l_b
            .into_dimensionality::<Ix3>()
            .map_err(|e| NyError::InvalidSpec(e.to_string()))?;
        let u_b = u_b
            .into_dimensionality::<Ix3>()
            .map_err(|e| NyError::InvalidSpec(e.to_string()))?;

        let alpha_l = l_b;
        let beta_l = l_a.clone();
        let ny_l = &alpha_l * &beta_l * -1.0;
        let alpha_u = u_b;
        let beta_u = l_a;
        let ny_u = &alpha_u * &beta_u * -1.0;

        let mut lw = Array4::<f32>::zeros(self.lw.dim());
        let mut uw = Array4::<f32>::zeros(self.uw.dim());
        let mut lb = Array3::<f32>::zeros(self.lb.dim());
        let mut ub = Array3::<f32>::zeros(self.ub.dim());

        Self::add_linear(&mut lw, &mut lb, self, &alpha_l, Some(&ny_l), true);
        Self::add_linear(&mut lw, &mut lb, other, &beta_l, None, true);
        Self::add_linear(&mut uw, &mut ub, self, &alpha_u, Some(&ny_u), false);
        Self::add_linear(&mut uw, &mut ub, other, &beta_u, None, false);

        Self::new(self.p, self.eps, self.perturbed_words, lw, lb, uw, ub)
    }

    fn add_linear(
        w_out: &mut Array4<f32>,
        b_out: &mut Array3<f32>,
        src: &Self,
        k: &Array3<f32>,
        ny: Option<&Array3<f32>>,
        is_lower: bool,
    ) {
        let (batch, length, dim_in, dim_out) = src.lw.dim();
        for b in 0..batch {
            for l in 0..length {
                for o in 0..dim_out {
                    let k_val = k[[b, l, o]];
                    let ny_val = ny.map_or(0.0, |g| g[[b, l, o]]);
                    if is_lower {
                        if k_val >= 0.0 {
                            for i in 0..dim_in {
                                w_out[[b, l, i, o]] += src.lw[[b, l, i, o]] * k_val;
                            }
                            b_out[[b, l, o]] += src.lb[[b, l, o]] * k_val + ny_val;
                        } else {
                            for i in 0..dim_in {
                                w_out[[b, l, i, o]] += src.uw[[b, l, i, o]] * k_val;
                            }
                            b_out[[b, l, o]] += src.ub[[b, l, o]] * k_val + ny_val;
                        }
                    } else if k_val >= 0.0 {
                        for i in 0..dim_in {
                            w_out[[b, l, i, o]] += src.uw[[b, l, i, o]] * k_val;
                        }
                        b_out[[b, l, o]] += src.ub[[b, l, o]] * k_val + ny_val;
                    } else {
                        for i in 0..dim_in {
                            w_out[[b, l, i, o]] += src.lw[[b, l, i, o]] * k_val;
                        }
                        b_out[[b, l, o]] += src.lb[[b, l, o]] * k_val + ny_val;
                    }
                }
            }
        }
    }
}
