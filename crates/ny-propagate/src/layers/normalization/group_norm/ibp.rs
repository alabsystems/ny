// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Interval bound propagation for GroupNorm (conservative mode).
//!
//! GroupNorm: y[c, t] = ny[c] * (x[c, t] - mean_g) / sqrt(var_g + eps) + beta[c]
//!
//! where mean_g and var_g are computed over all (C/G)*T elements in group g.
//!
//! IBP strategy (per group):
//! 1. Bound mean_g via interval arithmetic over all elements in the group
//! 2. Bound var_g from interval on centered values
//! 3. Bound std_g = sqrt(var_g + eps)
//! 4. For each element: bound y using sign analysis and interval division
//!    with per-element ny/beta
//!
//! Forward-mode IBP lives in `ibp_forward.rs`.
//!
//! Part of #3205.

use std::borrow::Cow;

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_dim_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use super::super::math_common::{compute_batch_prefix, square_interval_bounds};

use super::types::GroupNormLayer;
use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

impl GroupNormLayer {
    /// Validate IBP input and return (num_channels, time_len).
    pub(super) fn validate_ibp_input(&self, input: &BoundedTensor) -> Result<(usize, usize)> {
        let shape = input.shape();
        let ndim = shape.len();

        if ndim < 2 {
            return Err(NyError::InvalidSpec(
                "GroupNorm requires at least 2D input [C, T]".to_string(),
            ));
        }
        if shape.contains(&0) {
            return Err(NyError::InvalidSpec(
                "GroupNorm: zero-valued dimension in input shape".to_string(),
            ));
        }
        if input.lower().iter().any(|x| !x.is_finite())
            || input.upper().iter().any(|x| !x.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "GroupNorm IBP: non-finite input bounds".to_string(),
            ));
        }

        let num_channels = shape[ndim - 2];
        let time_len = shape[ndim - 1];

        if time_len > (1 << 24) {
            return Err(NyError::InternalError(format!(
                "GroupNorm time dimension {time_len} exceeds f32 exact integer range"
            )));
        }
        let cpg = self.channels_per_group();
        let group_size = cpg.checked_mul(time_len).ok_or_else(|| {
            NyError::InternalError(format!(
                "GroupNorm group size overflow (cpg={cpg} * time_len={time_len})"
            ))
        })?;
        if group_size > (1 << 24) {
            return Err(NyError::InternalError(format!(
                "GroupNorm group size {group_size} (cpg={cpg} * time_len={time_len}) exceeds f32 exact integer range",
            )));
        }
        if self.num_channels() != num_channels {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.num_channels()],
                got: vec![num_channels],
            });
        }

        Ok((num_channels, time_len))
    }

    /// Fallback output bounds when input bounds are non-finite.
    pub(super) fn fallback_output_bounds(&self, shape: &[usize]) -> Result<BoundedTensor> {
        let ndim = shape.len();
        let num_channels = shape[ndim - 2];
        let time_len = shape[ndim - 1];
        let group_size = self.channels_per_group() * time_len;

        if group_size == 0 {
            return BoundedTensor::new(ArrayD::zeros(IxDyn(shape)), ArrayD::zeros(IxDyn(shape)));
        }

        // next_up: z_max is a range bound → round UP for soundness (#3344).
        let z_max = if group_size <= 1 {
            0.0
        } else {
            next_up_f32(((group_size as f32) - 1.0).sqrt())
        };

        let batch_size: usize =
            checked_dim_product(&shape[..ndim - 2], "GroupNorm IBP batch dimensions")?;

        let mut out_lower = ArrayD::<f32>::zeros(IxDyn(shape));
        let mut out_upper = ArrayD::<f32>::zeros(IxDyn(shape));

        for c in 0..num_channels {
            let ny = self.ny[c];
            let beta = self.beta[c];
            // Directed rounding on fallback bounds (#3344).
            let (ch_lower, ch_upper) = if !ny.is_finite() || !beta.is_finite() {
                (f32::NEG_INFINITY, f32::INFINITY)
            } else if ny >= 0.0 {
                (
                    next_down_f32(beta - ny * z_max),
                    next_up_f32(beta + ny * z_max),
                )
            } else {
                (
                    next_down_f32(beta + ny * z_max),
                    next_up_f32(beta - ny * z_max),
                )
            };
            for batch_idx in 0..batch_size.max(1) {
                let batch_prefix = compute_batch_prefix(shape, ndim, batch_idx);
                for t in 0..time_len {
                    let mut idx = batch_prefix.clone();
                    idx.push(c);
                    idx.push(t);
                    out_lower[idx.as_slice()] = ch_lower;
                    out_upper[idx.as_slice()] = ch_upper;
                }
            }
        }

        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    /// Conservative IBP for one group within a batch position.
    ///
    /// `x_lowers`/`x_uppers` have length cpg*time_len. `time_len` derived internally.
    fn ibp_conservative_group(
        &self,
        x_lowers: &[f32],
        x_uppers: &[f32],
        group_idx: usize,
        out_lower: &mut [f32],
        out_upper: &mut [f32],
    ) {
        let cpg = self.channels_per_group();
        let n = x_lowers.len();
        let time_len = n / cpg;
        // f64 accumulation + directed rounding for mean/variance (#3327, pattern from #3324)
        let nf_64 = n as f64;
        let group_start_ch = group_idx * cpg;

        // next_down/next_up: enclose true mean from below/above
        let mean_lower: f32 =
            next_down_f32((x_lowers.iter().map(|&x| x as f64).sum::<f64>() / nf_64) as f32);
        let mean_upper: f32 =
            next_up_f32((x_uppers.iter().map(|&x| x as f64).sum::<f64>() / nf_64) as f32);

        // Mean overflow guard
        if !mean_lower.is_finite() || !mean_upper.is_finite() {
            self.fill_overflow_bounds(out_lower, out_upper, group_start_ch, cpg, time_len, n);
            return;
        }

        // Bound variance and std
        let (std_lower, std_upper) =
            Self::bound_group_std(x_lowers, x_uppers, mean_lower, mean_upper, self.eps);

        // Sound f32 float-error margin for the sqrt(n-1) z-clamp (#3196 leak fix).
        //
        // GroupNorm is a MEAN-SUBTRACT (centered) norm: z_i = (x_i - mean)/std with
        // the n group elements satisfying sum_i (x_i - mean) = 0. In exact reals this
        // zero-sum / unit-population-variance constraint gives the tight Cauchy-Schwarz
        // ceiling |z_i| <= sqrt(n-1). But the DOWNSTREAM f32 inference program the
        // verifier must soundly cover evaluates z in finite precision, and catastrophic
        // cancellation in fl(x_i - mean) for large |x_i| can push the realized |z_i|
        // slightly ABOVE sqrt(n-1). Clamping the IBP interval at next_up(sqrt(n-1))
        // (only ~1 ULP above the real bound) could then chop a reachable output =>
        // UNSOUND. We add a margin that bounds that f32 overshoot.
        //
        // Worst-case f32 evaluation error of z_i = fl(fl(x_i - mean)/std):
        //   (a) a spec-compliant naive f32 group mean is a sequential sum of n
        //       terms of magnitude ~M=max|x|, accumulating ~(n-1)*0.5*ulp(n*M)
        //       error; after /n that is ~0.5*(n-1)*ulp(M), and the (x_i - mean)
        //       subtraction adds ~ulp(M). The cumulative numerator absolute error
        //       is therefore ~0.5*(n+1)*ulp(M), NOT ulp(M): a bare ulp(max|x|)
        //       (no n factor) UNDER-counts it and is UNSOUND (validated
        //       counterexample at n=10 realized numerator error 0.14375 vs budget
        //       ulp(max)=0.0625). The (n+2) factor below dominates this.
        //   (b) the final divide rounds <= 0.5*ulp(|z|) <= 0.5*ulp(sqrt(n-1)); the
        //       divide's relative error at |z|<=base is covered by base*EPSILON.
        // Dividing the numerator error by the smallest sound std (std_lower, already
        // rounded DOWN so 1/std_lower is over-estimated) gives the amplified term. Hence
        //   delta = (n+2)*ulp(max_abs_x)/std_lower + base*EPSILON + 0.5*ulp(sqrt(n-1)),
        // and |z_i| <= sqrt(n-1) + delta over the reachable f32 set. (The mean-subtract
        // (n+2)*ulp/std numerator term is what distinguishes centered norms from RMSNorm,
        // which has no such numerator cancellation and uses a denominator-relative margin.)
        //
        // Every step rounds strictly OUTWARD: std_lower is rounded DOWN (over-estimates
        // 1/std), ulp_x and ulp_b are >= the true ULPs, their sum is next_up'd, and the
        // final base+delta is next_up'd. So max_norm >= the true f32 ceiling and the
        // clamped interval [-max_norm, max_norm] is a sound SUPERSET of the reachable set,
        // while still tighter than the unclamped 4-corner interval in the
        // high-magnitude/low-variance regime (retains the #3196 tightening).
        //
        // Non-finite guard: if max_abs_x is not finite (Inf inputs slipped past the
        // finite-input check, or NaN), fall through to max_norm = +INF so the unclamped
        // interval is kept and repaired downstream, never tightened.
        // next_up: max_norm is a range bound → round UP (#3344).
        let max_norm = if n > 1 {
            let max_abs_x = x_lowers
                .iter()
                .zip(x_uppers.iter())
                .map(|(&l, &u)| l.abs().max(u.abs()))
                .fold(0.0_f32, f32::max);
            let base = (((n as f64) - 1.0).sqrt()) as f32;
            if max_abs_x.is_finite() && std_lower > 0.0 && std_lower.is_finite() {
                let ulp_x = next_up_f32(max_abs_x) - max_abs_x;
                let ulp_b = next_up_f32(base) - base;
                // (n+2) ULP(max|x|): a spec-compliant naive f32 group mean is a
                // sequential sum of n=group_size terms of magnitude ~M, accumulating
                // ~(n-1)*0.5*ulp(n*M) error; after /n that is ~0.5*(n-1)*ulp(M),
                // plus the (x_i - mean) subtraction (~ulp(M)). A bare ulp(max|x|)
                // (no n factor) UNDER-counts this and is unsound; (n+2) dominates
                // the (n-1)-term summation + subtraction. base*EPSILON covers the
                // final divide's relative error at |z|<=base; 0.5*ulp(base) sqrt.
                let delta = next_up_f32(
                    (n as f32 + 2.0) * ulp_x / std_lower + base * f32::EPSILON + 0.5 * ulp_b,
                );
                next_up_f32(base + delta)
            } else {
                // Degenerate std/inputs: keep the unclamped interval (sound, never tighter).
                f32::INFINITY
            }
        } else {
            0.0
        };

        // Bound each element y = ny * (x - mean) / std + beta
        for i in 0..n {
            let c = group_start_ch + (i / time_len);
            let g = self.ny[c];
            let b = self.beta[c];

            // Directed rounding on intermediate deviation: if centered_lower
            // rounds UP, the 4-corner domain is too narrow → unsound. Part of #3344.
            let centered_lower = next_down_f32(x_lowers[i] - mean_upper);
            let centered_upper = next_up_f32(x_uppers[i] - mean_lower);

            let corners = [
                centered_lower / std_upper,
                centered_lower / std_lower,
                centered_upper / std_upper,
                centered_upper / std_lower,
            ];

            let ratio_l = nan_propagating_max(
                corners
                    .iter()
                    .fold(f32::INFINITY, |a, &v| nan_propagating_min(a, v)),
                -max_norm,
            );
            let ratio_u = nan_propagating_min(
                corners
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &v| nan_propagating_max(a, v)),
                max_norm,
            );

            // Directed rounding on final bounds: bare f32 mul+add from
            // corner division results. Part of #3344.
            let (y_l, y_u) = if g >= 0.0 {
                (next_down_f32(g * ratio_l + b), next_up_f32(g * ratio_u + b))
            } else {
                (next_down_f32(g * ratio_u + b), next_up_f32(g * ratio_l + b))
            };
            out_lower[i] = y_l;
            out_upper[i] = y_u;
        }
    }

    /// Bound std from lower/upper bounds on group elements.
    fn bound_group_std(
        x_lowers: &[f32],
        x_uppers: &[f32],
        mean_lower: f32,
        mean_upper: f32,
        eps: f32,
    ) -> (f32, f32) {
        let n = x_lowers.len();
        // f64 accumulation for variance bounds (#3327, pattern from #3324)
        let nf_64 = n as f64;
        let eps_64 = eps as f64;
        let mut sq_sum_lower_f64 = 0.0_f64;
        let mut sq_sum_upper_f64 = 0.0_f64;
        for i in 0..n {
            // Directed rounding on intermediate deviation: feeds variance via
            // square_interval_bounds — too-narrow domain → too-narrow variance → unsound.
            // Part of #3344.
            let centered_lower = next_down_f32(x_lowers[i] - mean_upper);
            let centered_upper = next_up_f32(x_uppers[i] - mean_lower);
            let (sq_l, sq_u) = square_interval_bounds(centered_lower, centered_upper);
            sq_sum_lower_f64 += sq_l as f64;
            sq_sum_upper_f64 += sq_u as f64;
        }
        // next_down/next_up: var_lower rounds DOWN (denominator), var_upper rounds UP
        let var_lower = next_down_f32((sq_sum_lower_f64 / nf_64) as f32);
        let var_upper = next_up_f32((sq_sum_upper_f64 / nf_64) as f32);
        // Directed rounding on sqrt: std_lower is denominator → round DOWN,
        // std_upper rounds UP (wider interval = sound). #3327.
        (
            next_down_f32(((var_lower as f64 + eps_64).sqrt()) as f32),
            next_up_f32(((var_upper as f64 + eps_64).sqrt()) as f32),
        )
    }

    /// Fill output with overflow fallback bounds (sqrt(n-1) z-bound).
    fn fill_overflow_bounds(
        &self,
        out_lower: &mut [f32],
        out_upper: &mut [f32],
        group_start_ch: usize,
        cpg: usize,
        time_len: usize,
        n: usize,
    ) {
        // next_up: z_max is a range bound → round UP for soundness (#3344).
        let z_max = if n > 1 {
            next_up_f32(((n as f32) - 1.0).sqrt())
        } else {
            0.0
        };
        for c_offset in 0..cpg {
            let c = group_start_ch + c_offset;
            let g = self.ny[c];
            let b = self.beta[c];
            let (ch_lower, ch_upper) = if !g.is_finite() || !b.is_finite() {
                (f32::NEG_INFINITY, f32::INFINITY)
            } else if g >= 0.0 {
                // Directed rounding on overflow fallback bounds (#3344).
                (next_down_f32(b - g * z_max), next_up_f32(b + g * z_max))
            } else {
                (next_down_f32(b + g * z_max), next_up_f32(b - g * z_max))
            };
            for t in 0..time_len {
                let idx = c_offset * time_len + t;
                out_lower[idx] = ch_lower;
                out_upper[idx] = ch_upper;
            }
        }
    }

    /// Conservative IBP over all batch positions and groups.
    fn propagate_ibp_conservative(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();
        let (_num_channels, time_len) = self.validate_ibp_input(input)?;
        let cpg = self.channels_per_group();
        let group_size = cpg * time_len;

        let mut out_lower = input.lower().clone();
        let mut out_upper = input.upper().clone();
        let batch_size: usize =
            checked_dim_product(&shape[..ndim - 2], "GroupNorm IBP batch dimensions")?;

        for batch_idx in 0..batch_size.max(1) {
            let batch_prefix = compute_batch_prefix(shape, ndim, batch_idx);

            for g_idx in 0..self.num_groups {
                let group_start_ch = g_idx * cpg;

                let mut x_lowers = Vec::with_capacity(group_size);
                let mut x_uppers = Vec::with_capacity(group_size);
                for c_offset in 0..cpg {
                    let c = group_start_ch + c_offset;
                    for t in 0..time_len {
                        let mut idx = batch_prefix.clone();
                        idx.push(c);
                        idx.push(t);
                        x_lowers.push(input.lower()[idx.as_slice()]);
                        x_uppers.push(input.upper()[idx.as_slice()]);
                    }
                }

                let mut grp_lower = vec![0.0_f32; group_size];
                let mut grp_upper = vec![0.0_f32; group_size];
                self.ibp_conservative_group(
                    &x_lowers,
                    &x_uppers,
                    g_idx,
                    &mut grp_lower,
                    &mut grp_upper,
                );

                for c_offset in 0..cpg {
                    let c = group_start_ch + c_offset;
                    for t in 0..time_len {
                        let mut idx = batch_prefix.clone();
                        idx.push(c);
                        idx.push(t);
                        let local = c_offset * time_len + t;
                        out_lower[idx.as_slice()] = grp_lower[local];
                        out_upper[idx.as_slice()] = grp_upper[local];
                    }
                }
            }
        }

        if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "GroupNorm IBP: NaN in computed bounds".to_string(),
            ));
        }

        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }
}

impl BoundPropagation for GroupNormLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.validate_ibp_input(input)?;

        if self.forward_mode {
            self.propagate_ibp_forward_mode(input)
        } else {
            self.propagate_ibp_conservative(input)
        }
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "GroupNorm is nonlinear -- use propagate_linear_with_bounds".to_string(),
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
        GroupNormLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}
