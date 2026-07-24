// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Max pooling layer for bound propagation.

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};
use std::borrow::Cow;

use crate::bounds::nan_propagating_max;
use crate::contiguous_flat_slice;
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

use super::checked_pool_output_size;

/// 2D max pooling layer: y = max(x over pooling window)
///
/// For interval propagation:
/// - lower_y = max(lower_x over window) - maximum of lower bounds
/// - upper_y = max(upper_x over window) - maximum of upper bounds
///
/// This is exact for IBP because max is monotonically increasing.
#[derive(Debug, Clone)]
pub struct MaxPool2dLayer {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Padding mode: true = use -inf for padding, false = only pool valid region
    pub use_negative_inf_padding: bool,
}

impl MaxPool2dLayer {
    /// Create a new max pool layer.
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            use_negative_inf_padding: true,
        }
    }

    /// Create max pool with explicit padding mode.
    pub fn with_padding_mode(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        use_negative_inf_padding: bool,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            use_negative_inf_padding,
        }
    }

    /// Compute output spatial dimensions.
    ///
    /// Requires `padding < kernel_size` per dimension: `padding >= kernel`
    /// admits pooling windows made entirely of padding, whose max (over an
    /// empty set of inputs) is -inf — no finite bound on such an output can
    /// be sound. PyTorch and ONNX Runtime both reject this geometry.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> Result<(usize, usize)> {
        let (kh, kw) = self.kernel_size;
        let (ph, pw) = self.padding;
        if ph >= kh || pw >= kw {
            return Err(NyError::InvalidSpec(format!(
                "MaxPool2d padding ({ph},{pw}) must be < kernel ({kh},{kw}) per dimension: \
                 an all-padding window has no defined max, so no bound on it is sound"
            )));
        }
        checked_pool_output_size(
            "MaxPool2d",
            input_h,
            input_w,
            self.kernel_size,
            self.stride,
            self.padding,
        )
    }
}

impl BoundPropagation for MaxPool2dLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Validate input shape: expect (C,H,W) or (B,C,H,W).
        let input_shape = input.lower().shape();
        let ndim = input_shape.len();
        if !(3..=4).contains(&ndim) {
            return Err(NyError::InvalidSpec(format!(
                "MaxPool2d IBP requires 3D or 4D input, got {}D",
                ndim
            )));
        }

        let (batch_size, channels, in_h, in_w) = if ndim == 4 {
            (
                Some(input_shape[0]),
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            (None, input_shape[0], input_shape[1], input_shape[2])
        };

        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        // Create output arrays
        let out_shape = if let Some(b) = batch_size {
            vec![b, channels, out_h, out_w]
        } else {
            vec![channels, out_h, out_w]
        };
        let mut out_lower = ArrayD::from_elem(IxDyn(&out_shape), f32::NEG_INFINITY);
        let mut out_upper = ArrayD::from_elem(IxDyn(&out_shape), f32::NEG_INFINITY);

        // Apply max pooling for 3D case
        if batch_size.is_none() {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih_start = oh * sh;
                        let iw_start = ow * sw;

                        let mut max_lower = f32::NEG_INFINITY;
                        let mut max_upper = f32::NEG_INFINITY;

                        for kh_off in 0..kh {
                            for kw_off in 0..kw {
                                let ih = (ih_start + kh_off) as isize - ph as isize;
                                let iw = (iw_start + kw_off) as isize - pw as isize;

                                // SAFETY(as usize): ih/iw are isize, guard ensures >= 0 and < in_h/in_w.
                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    max_lower =
                                        nan_propagating_max(max_lower, input.lower()[[c, ih, iw]]);
                                    max_upper =
                                        nan_propagating_max(max_upper, input.upper()[[c, ih, iw]]);
                                } else if !self.use_negative_inf_padding {
                                    // If not using -inf padding, skip this position
                                    // (max over fewer elements)
                                }
                                // If using -inf padding, the -inf won't affect max
                            }
                        }

                        out_lower[[c, oh, ow]] = max_lower;
                        out_upper[[c, oh, ow]] = max_upper;
                    }
                }
            }
        }

        // Handle 4D batch case
        if let Some(b) = batch_size {
            for batch_idx in 0..b {
                for c in 0..channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let ih_start = oh * sh;
                            let iw_start = ow * sw;

                            let mut max_lower = f32::NEG_INFINITY;
                            let mut max_upper = f32::NEG_INFINITY;

                            for kh_off in 0..kh {
                                for kw_off in 0..kw {
                                    let ih = (ih_start + kh_off) as isize - ph as isize;
                                    let iw = (iw_start + kw_off) as isize - pw as isize;

                                    // SAFETY(as usize): ih/iw are isize, guard ensures >= 0 and < in_h/in_w.
                                    if ih >= 0
                                        && ih < in_h as isize
                                        && iw >= 0
                                        && iw < in_w as isize
                                    {
                                        let ih = ih as usize;
                                        let iw = iw as usize;
                                        max_lower = nan_propagating_max(
                                            max_lower,
                                            input.lower()[[batch_idx, c, ih, iw]],
                                        );
                                        max_upper = nan_propagating_max(
                                            max_upper,
                                            input.upper()[[batch_idx, c, ih, iw]],
                                        );
                                    }
                                }
                            }

                            out_lower[[batch_idx, c, oh, ow]] = max_lower;
                            out_upper[[batch_idx, c, oh, ow]] = max_upper;
                        }
                    }
                }
            }
        }

        // NaN in output indicates NaN in input (data corruption, not overflow).
        // Must check before new_repaired which would silently swallow NaN (#2812, #3423).
        if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "MaxPool2d IBP: NaN in bounds (from NaN input)".into(),
            ));
        }
        // Repair non-finite outputs (Inf→fallback) for consistency with linear IBP (#3030).
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Max pooling CROWN requires tracking which element achieves the max,
        // which depends on the input intervals. Use propagate_linear_with_bounds instead.
        Err(NyError::UnsupportedOp(
            "MaxPool2d linear propagation requires bounds - use propagate_linear_with_bounds"
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
        MaxPool2dLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl MaxPool2dLayer {
    /// CROWN backward propagation for MaxPool2d with pre-activation bounds.
    ///
    /// MaxPool is a piecewise-linear operation (max of k inputs). The CROWN relaxation:
    /// - If one input definitely dominates (l_i > max_{j!=i}(u_j)), route gradient through it
    /// - Otherwise, use constant IBP-style bounds
    ///
    /// This is similar to ReLU relaxation but for multi-input max instead of single-input clamp.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let input_shape = pre_activation.shape();
        let ndim = input_shape.len();

        // Expect 3D (C, H, W) or 4D (B, C, H, W)
        if !(3..=4).contains(&ndim) {
            return Err(NyError::InvalidSpec(format!(
                "MaxPool2d CROWN requires 3D or 4D input, got {}D",
                ndim
            )));
        }

        let (batch_size, channels, in_h, in_w) = if ndim == 4 {
            (
                Some(input_shape[0]),
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            (None, input_shape[0], input_shape[1], input_shape[2])
        };

        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let input_size = if let Some(b) = batch_size {
            checked_shape_product(&[b, channels, in_h, in_w])
        } else {
            checked_shape_product(&[channels, in_h, in_w])
        }
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MaxPool2d CROWN: input size product overflows: batch={batch_size:?} ch={channels} h={in_h} w={in_w}"
            ))
        })?;

        let output_size = if let Some(b) = batch_size {
            checked_shape_product(&[b, channels, out_h, out_w])
        } else {
            checked_shape_product(&[channels, out_h, out_w])
        }
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "MaxPool2d CROWN: output size product overflows: batch={batch_size:?} ch={channels} h={out_h} w={out_w}"
            ))
        })?;

        if bounds.num_inputs() != output_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![output_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Build the backward propagation coefficients.
        //
        // SOUNDNESS (#maxpool-dense-winner-coeff-accum): a single input column can
        // be the definite winner (or the i* lower-witness) for MULTIPLE overlapping
        // output windows (stride < kernel), so `new_*_a[[out_idx, x_flat]]` receives
        // several `+=` of f32 incoming coefficients. An f32 running sum of n terms
        // carries an accumulation error up to γ_n^f32·Σ|term| that the OLD path
        // (round-to-nearest f32 += then `new_or_conservative`, NO certified error)
        // dropped entirely → the concretized bound could be tighter than the true
        // reachable value = FALSE PROOF on overlapping-window MaxPool.
        //
        // Fix (mirrors the proven conv f64-accumulate fix, becc501): accumulate the
        // coefficient in f64 (f32→f64 widening is exact; the summands are f32 so each
        // is exact in f64; only the f64 sum rounds → bounded by γ_n^f64·S), track
        // S = Σ|term| per column, store the round-to-nearest f32, and certify
        //   err = |f64_sum − stored_f32| (cast) + γ_n^f64·S (accumulation).
        // This is the tight f64 route (no γ_n^f32 stopgap needed). Incoming coeff
        // error need not be re-propagated here: MaxPool is NOT a `propagates_coeff_err`
        // carrier, so the backward dispatcher folds/discharges any incoming error
        // into the bias OUTWARD before this method runs.
        let mut lower_a_f64 = Array2::<f64>::zeros((num_outputs, input_size));
        let mut upper_a_f64 = Array2::<f64>::zeros((num_outputs, input_size));
        // S = Σ|term| and term count per coefficient (drives γ_n^f64·S).
        let mut lower_s = Array2::<f64>::zeros((num_outputs, input_size));
        let mut upper_s = Array2::<f64>::zeros((num_outputs, input_size));
        let mut lower_cnt = Array2::<u32>::zeros((num_outputs, input_size));
        let mut upper_cnt = Array2::<u32>::zeros((num_outputs, input_size));
        let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
        let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);

        // Helper to compute flat indices
        let in_flat = |b_opt: Option<usize>, c: usize, h: usize, w: usize| -> usize {
            if let Some(b) = b_opt {
                b * channels * in_h * in_w + c * in_h * in_w + h * in_w + w
            } else {
                c * in_h * in_w + h * in_w + w
            }
        };

        let out_flat = |b_opt: Option<usize>, c: usize, oh: usize, ow: usize| -> usize {
            if let Some(b) = b_opt {
                b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow
            } else {
                c * out_h * out_w + oh * out_w + ow
            }
        };

        // Flatten pre-activation bounds for easy access
        let pre_lower = contiguous_flat_slice(pre_activation.lower());
        let pre_upper = contiguous_flat_slice(pre_activation.upper());

        // Iterate over output positions
        let batch_count = batch_size.unwrap_or(1);

        for b in 0..batch_count {
            let b_opt = if batch_size.is_some() { Some(b) } else { None };

            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let y_flat = out_flat(b_opt, c, oh, ow);
                        let ih_start = oh * sh;
                        let iw_start = ow * sw;

                        // Collect valid inputs in this pooling window
                        let mut window_inputs: Vec<(usize, f32, f32, f32)> =
                            Vec::with_capacity(kh * kw);

                        for kh_off in 0..kh {
                            for kw_off in 0..kw {
                                let ih = (ih_start + kh_off) as isize - ph as isize;
                                let iw = (iw_start + kw_off) as isize - pw as isize;

                                // SAFETY(as usize): ih/iw are isize, guard ensures >= 0 and < in_h/in_w.
                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let x_flat = in_flat(b_opt, c, ih, iw);
                                    let l = pre_lower[x_flat];
                                    let u = pre_upper[x_flat];
                                    // Bit-identical window anchor: f32::midpoint rounds differently at overflow/subnormal edges.
                                    #[allow(clippy::manual_midpoint)]
                                    let mid = (l + u) * 0.5;
                                    window_inputs.push((x_flat, l, u, mid));
                                }
                            }
                        }

                        if window_inputs.is_empty() {
                            // All positions are padding: the output is max over an
                            // empty set (-inf), which no finite linear row can bound.
                            // output_size() already rejects the padding >= kernel
                            // geometry that creates such windows; refuse here too
                            // rather than emit an unexamined row.
                            return Err(NyError::InvalidSpec(format!(
                                "MaxPool2d CROWN: pooling window at output ({oh},{ow}) \
                                 contains no input positions: kernel=({kh},{kw}), \
                                 padding=({ph},{pw})"
                            )));
                        }

                        // Find the maximum lower bound and upper bound across inputs
                        // NaN-propagating folds: NaN in input bounds must propagate — see #2577.
                        let max_lower = window_inputs
                            .iter()
                            .map(|&(_, l, _, _)| l)
                            .fold(f32::NEG_INFINITY, nan_propagating_max);
                        let max_upper = window_inputs
                            .iter()
                            .map(|&(_, _, u, _)| u)
                            .fold(f32::NEG_INFINITY, nan_propagating_max);

                        // Check if there's a definite winner (one input whose lower bound >= all other upper bounds)
                        // We need to exclude self-comparison (comparing against own upper bound)
                        let definite_winner = window_inputs.iter().find(|&&(idx, l, _, _)| {
                            window_inputs
                                .iter()
                                .all(|&(other_idx, _, other_u, _)| idx == other_idx || l >= other_u)
                        });

                        if let Some(&(winner_flat, _, _, _)) = definite_winner {
                            // Single definite winner - gradient flows through it entirely (like identity)
                            for out_idx in 0..num_outputs {
                                let la = bounds.lower_a()[[out_idx, y_flat]];
                                let ua = bounds.upper_a()[[out_idx, y_flat]];
                                lower_a_f64[[out_idx, winner_flat]] += la as f64;
                                lower_s[[out_idx, winner_flat]] += (la as f64).abs();
                                lower_cnt[[out_idx, winner_flat]] += 1;
                                upper_a_f64[[out_idx, winner_flat]] += ua as f64;
                                upper_s[[out_idx, winner_flat]] += (ua as f64).abs();
                                upper_cnt[[out_idx, winner_flat]] += 1;
                            }
                        } else {
                            // Multiple candidates - no single input definitely dominates.
                            //
                            // SOUND LOWER RELAXATION (dense path only): the standard
                            // alpha-beta-CROWN maxpool lower bound is the linear function
                            //   y = max(x_1..x_k) >= x_{i*},   where i* = argmax_i l_i.
                            // This holds pointwise over the entire box (no convexity
                            // subtlety): x_{i*} is one of the maxed inputs, so the max is
                            // always at least x_{i*}. Routing the lower row through x_{i*}
                            // instead of the constant max_lower = l_{i*} is strictly tighter
                            // (x_{i*} >= l_{i*}) and sound:
                            // - la > 0:  la*y >= la*x_{i*}  → add la to new_lower_a[out, i*]
                            // - ua < 0:  ua*y <= ua*x_{i*}  → add ua to new_upper_a[out, i*]
                            //
                            // The la<0 / ua>0 arms must STAY CONSTANT: x_{i*} is NOT an
                            // upper bound on y (y >= x_{i*}), so routing those arms through
                            // x_{i*} would be UNSOUND. They keep the constant max_upper:
                            // - la < 0:  la*y >= la*max_upper  (la*y minimized at y=max_upper)
                            // - ua > 0:  ua*y <= ua*max_upper  (ua*y maximized at y=max_upper)
                            //
                            // NOTE: This linear-lower-bound change is DELIBERATELY NOT applied
                            // to the patches path (max_patches.rs): that path uses a single
                            // shared winner_d slope map for BOTH the lower and upper rows, so
                            // a slope-1 lower bound would also feed the upper row where
                            // y <= x_{i*} is FALSE → unsound. See max_patches.rs.

                            // i* = argmax_i l_i (the input achieving max_lower). The
                            // definite-winner search already failed, so this is purely the
                            // lower-bound witness, not a dominating winner.
                            // NaN-safe: window_inputs is non-empty (checked above); if any l
                            // is NaN the IBP path already errors out upstream.
                            let istar_flat = window_inputs
                                .iter()
                                .max_by(|a, b| {
                                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|&(idx, _, _, _)| idx);

                            for out_idx in 0..num_outputs {
                                let la = bounds.lower_a()[[out_idx, y_flat]];
                                let ua = bounds.upper_a()[[out_idx, y_flat]];

                                // Lower bound row: la * y.
                                // Guard: skip zero coefficients to avoid 0*inf NaN (#1739).
                                if la > 0.0 {
                                    // SOUND tighter: route through x_{i*} (y >= x_{i*}).
                                    if let Some(istar) = istar_flat {
                                        lower_a_f64[[out_idx, istar]] += la as f64;
                                        lower_s[[out_idx, istar]] += (la as f64).abs();
                                        lower_cnt[[out_idx, istar]] += 1;
                                    } else {
                                        new_lower_b_f64[out_idx] += la as f64 * max_lower as f64;
                                    }
                                } else if la < 0.0 {
                                    // UNCHANGED: x_{i*} is not an upper bound on y → constant.
                                    new_lower_b_f64[out_idx] += la as f64 * max_upper as f64;
                                }

                                // Upper bound row: ua * y.
                                if ua > 0.0 {
                                    // UNCHANGED: x_{i*} is not an upper bound on y → constant.
                                    new_upper_b_f64[out_idx] += ua as f64 * max_upper as f64;
                                } else if ua < 0.0 {
                                    // SOUND tighter: route through x_{i*} (y >= x_{i*}).
                                    if let Some(istar) = istar_flat {
                                        upper_a_f64[[out_idx, istar]] += ua as f64;
                                        upper_s[[out_idx, istar]] += (ua as f64).abs();
                                        upper_cnt[[out_idx, istar]] += 1;
                                    } else {
                                        new_upper_b_f64[out_idx] += ua as f64 * max_lower as f64;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Finalize: directed-round each f64 coefficient to f32 and certify the
        // per-coefficient error = |f64_sum − stored_f32| (cast) + γ_n^f64·S
        // (f64-accumulation error, Higham Thm 3.1). `n` is this coefficient's own
        // term count (exact, the tightest factor). A coefficient written exactly
        // once still gets a (zero-cast, near-zero γ_1·S) certificate — harmless.
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_lower_a_err = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a_err = Array2::<f32>::zeros((num_outputs, input_size));
        for i in 0..num_outputs {
            for j in 0..input_size {
                // Lower coefficient.
                let lv = lower_a_f64[[i, j]];
                let lf = lv as f32;
                new_lower_a[[i, j]] = lf;
                let l_cast = (lv - lf as f64).abs();
                let l_gamma =
                    crate::layers::linear::crown_single_gamma_n_f64(lower_cnt[[i, j]] as usize);
                new_lower_a_err[[i, j]] = next_up_f32((l_cast + l_gamma * lower_s[[i, j]]) as f32);
                // Upper coefficient.
                let uv = upper_a_f64[[i, j]];
                let uf = uv as f32;
                new_upper_a[[i, j]] = uf;
                let u_cast = (uv - uf as f64).abs();
                let u_gamma =
                    crate::layers::linear::crown_single_gamma_n_f64(upper_cnt[[i, j]] as usize);
                new_upper_a_err[[i, j]] = next_up_f32((u_cast + u_gamma * upper_s[[i, j]]) as f32);
            }
        }

        LinearBounds::new_or_conservative_with_err(
            new_lower_a,
            new_lower_b_f64.mapv(|x| next_down_f32(x as f32)),
            new_upper_a,
            new_upper_b_f64.mapv(|x| next_up_f32(x as f32)),
            new_lower_a_err,
            new_upper_a_err,
        )
    }
}
