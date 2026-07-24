// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Interval concretization and Lq dual norm computation for MultiNormBounds.

use ndarray::{s, Array3, ArrayView4};
use ny_core::Result;

use crate::rounding::{next_down_f32, next_up_f32};
use crate::{BoundedTensor, RepairStrategy};

use super::MultiNormBounds;

impl MultiNormBounds {
    /// Concretize to interval bounds via the dual norm.
    ///
    /// Uses f32 arithmetic throughout. For sound bounds with directed rounding,
    /// use `concretize_sound` instead.
    pub fn concretize(&self) -> Result<BoundedTensor> {
        let (_, _, dim_in, _) = self.lw.dim();
        let dim_per_word = dim_in / self.perturbed_words;
        let mut lower = self.lb.clone();
        let mut upper = self.ub.clone();
        for word in 0..self.perturbed_words {
            let start = word * dim_per_word;
            let end = start + dim_per_word;
            let lw_slice = self.lw.slice(s![.., .., start..end, ..]);
            let uw_slice = self.uw.slice(s![.., .., start..end, ..]);
            let norm_l = Self::norm_q(lw_slice, self.q);
            let norm_u = Self::norm_q(uw_slice, self.q);
            lower.zip_mut_with(&norm_l, |l, &n| {
                *l -= self.eps * n;
            });
            upper.zip_mut_with(&norm_u, |u, &n| {
                *u += self.eps * n;
            });
        }
        BoundedTensor::new(lower.into_dyn(), upper.into_dyn())
    }

    /// Concretize with f64 accumulators and directed rounding for soundness (#2239, #2364).
    ///
    /// Uses f64 accumulators in `norm_q_f64()` and f64 arithmetic for the
    /// `eps * norm` subtraction/addition, then casts to f32 with directed
    /// rounding (`next_down_f32` for lower, `next_up_f32` for upper).
    ///
    /// With f64 accumulators the norm computation has negligible rounding error
    /// (eps_f64 ~ 1.1e-16), so the only significant rounding event is the
    /// f64→f32 cast, which is ≤0.5 ULP — well within the 1-ULP directed
    /// rounding budget.
    ///
    /// Reference: same pattern as `BatchedLinearBounds::concretize_sound`
    /// (`bounds/batched/mod.rs:276`). Prior version used f32 accumulators in
    /// `norm_q()`, which had O(dim_in * eps_f32) error — insufficient for
    /// dim_in ≥ 768 (see #2364 error analysis).
    pub fn concretize_sound(&self) -> Result<BoundedTensor> {
        let (_, _, dim_in, _) = self.lw.dim();
        let dim_per_word = dim_in / self.perturbed_words;
        let eps = self.eps as f64;
        let mut lower = self.lb.mapv(|v| v as f64);
        let mut upper = self.ub.mapv(|v| v as f64);
        for word in 0..self.perturbed_words {
            let start = word * dim_per_word;
            let end = start + dim_per_word;
            let lw_slice = self.lw.slice(s![.., .., start..end, ..]);
            let uw_slice = self.uw.slice(s![.., .., start..end, ..]);
            let norm_l = Self::norm_q_f64(lw_slice, self.q);
            let norm_u = Self::norm_q_f64(uw_slice, self.q);
            lower.zip_mut_with(&norm_l, |l, &n| {
                *l -= eps * n;
            });
            upper.zip_mut_with(&norm_u, |u, &n| {
                *u += eps * n;
            });
        }
        // Cast f64→f32 with directed rounding: lower toward -inf, upper toward +inf.
        let lower_f32 = lower.mapv(|v| next_down_f32(v as f32));
        let upper_f32 = upper.mapv(|v| next_up_f32(v as f32));
        // Repair NaN/Inf at the type boundary (#3423). Widen strategy replaces NaN
        // with ±inf and fixes inversions.
        BoundedTensor::new_repaired(
            lower_f32.into_dyn(),
            upper_f32.into_dyn(),
            RepairStrategy::Widen,
        )
    }

    /// Concretize lower bounds for a specific coefficient slice.
    pub fn concretize_lower(&self, lw: ArrayView4<'_, f32>) -> Array3<f32> {
        let norm = Self::norm_q(lw, self.q);
        norm.mapv(|v| -self.eps * v)
    }

    /// Concretize upper bounds for a specific coefficient slice.
    pub fn concretize_upper(&self, uw: ArrayView4<'_, f32>) -> Array3<f32> {
        let norm = Self::norm_q(uw, self.q);
        norm.mapv(|v| self.eps * v)
    }

    pub(crate) fn norm_q(lw: ArrayView4<'_, f32>, q: f32) -> Array3<f32> {
        let (batch, length, dim_in, dim_out) = lw.dim();
        let mut out = Array3::<f32>::zeros((batch, length, dim_out));
        if q.is_infinite() {
            for b in 0..batch {
                for l in 0..length {
                    for o in 0..dim_out {
                        let mut max_val = 0.0f32;
                        for i in 0..dim_in {
                            let v = lw[[b, l, i, o]].abs();
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        out[[b, l, o]] = max_val;
                    }
                }
            }
            return out;
        }
        let use_l1 = (q - 1.0).abs() < 1e-6;
        for b in 0..batch {
            for l in 0..length {
                for o in 0..dim_out {
                    let mut acc = 0.0f32;
                    for i in 0..dim_in {
                        let v = lw[[b, l, i, o]].abs();
                        if use_l1 {
                            acc += v;
                        } else {
                            acc += v.powf(q);
                        }
                    }
                    out[[b, l, o]] = if use_l1 { acc } else { acc.powf(1.0 / q) };
                }
            }
        }
        out
    }

    /// Compute Lq dual norm with f64 accumulators for soundness (#2364).
    ///
    /// Same algorithm as `norm_q()` but accumulates in f64 to avoid
    /// O(dim_in * eps_f32) rounding error. Each f32 element is promoted to f64
    /// before accumulation, so the accumulated error is only O(dim_in * eps_f64)
    /// which is negligible for any practical dimension.
    fn norm_q_f64(lw: ArrayView4<'_, f32>, q: f32) -> Array3<f64> {
        let (batch, length, dim_in, dim_out) = lw.dim();
        let mut out = Array3::<f64>::zeros((batch, length, dim_out));
        if q.is_infinite() {
            // L-inf: max absolute value. No accumulation error.
            for b in 0..batch {
                for l in 0..length {
                    for o in 0..dim_out {
                        let mut max_val = 0.0f64;
                        for i in 0..dim_in {
                            let v = (lw[[b, l, i, o]] as f64).abs();
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        out[[b, l, o]] = max_val;
                    }
                }
            }
            return out;
        }
        let q_f64 = q as f64;
        let use_l1 = (q - 1.0).abs() < 1e-6;
        for b in 0..batch {
            for l in 0..length {
                for o in 0..dim_out {
                    let mut acc = 0.0f64;
                    for i in 0..dim_in {
                        let v = (lw[[b, l, i, o]] as f64).abs();
                        if use_l1 {
                            acc += v;
                        } else {
                            acc += v.powf(q_f64);
                        }
                    }
                    out[[b, l, o]] = if use_l1 { acc } else { acc.powf(1.0 / q_f64) };
                }
            }
        }
        out
    }
}
