// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Average pooling layer for bound propagation.

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};
use std::borrow::Cow;

use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

use super::checked_pool_output_size;

/// Average pooling layer: applies average pooling over spatial dimensions.
///
/// For 2D input (channels, height, width), applies kernel_size x kernel_size
/// average pooling with given stride and padding.
///
/// Average pooling is a linear operation with positive weights, so in REAL
/// arithmetic IBP bounds are exact: y_lower = avg_pool(x_lower),
/// y_upper = avg_pool(x_upper). In floating point they are NOT exact: the
/// window sum rounds. `propagate_ibp` accumulates each window sum in f64 and
/// directed-rounds the single f64→f32 store 1 ULP outward, which certifies the
/// cast but NOT the f64 accumulation residual (up to `γ⁶⁴_{k−1}·Σ|x_i|`, which
/// under ≥2^29 cancellation exceeds 1 f32 ULP of the result). The certified
/// enclosure for verdict paths is [`AveragePoolLayer::propagate_ibp_sound`]
/// (#vnncomp-aw-soundness, avgpool 1-ULP-arm item).
#[derive(Debug, Clone)]
pub struct AveragePoolLayer {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Whether to count padding zeros in divisor
    pub count_include_pad: bool,
}

impl AveragePoolLayer {
    /// Create a new average pool layer.
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        count_include_pad: bool,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            count_include_pad,
        }
    }

    /// Check if this is a global average pool operation.
    /// Global pooling uses kernel_size (0, 0) as a sentinel value.
    pub fn is_global(&self) -> bool {
        self.kernel_size == (0, 0)
    }

    /// Compute output spatial dimensions.
    ///
    /// Requires `padding < kernel_size` per dimension (the global-pooling
    /// kernel `(0, 0)` sentinel excepted): `padding >= kernel` admits pooling
    /// windows made entirely of padding, which average zero inputs — `0/0`
    /// under `count_include_pad=false` — so any value emitted for them is
    /// fabricated. ONNX Runtime rejects this geometry.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> Result<(usize, usize)> {
        // Global pooling: kernel_size (0, 0) means pool entire spatial dims
        if self.is_global() {
            return Ok((1, 1));
        }
        let (kh, kw) = self.kernel_size;
        let (ph, pw) = self.padding;
        if ph >= kh || pw >= kw {
            return Err(NyError::InvalidSpec(format!(
                "AveragePool padding ({ph},{pw}) must be < kernel ({kh},{kw}) per dimension: \
                 an all-padding window averages no inputs, so any value for it is fabricated"
            )));
        }
        checked_pool_output_size(
            "AveragePool",
            input_h,
            input_w,
            self.kernel_size,
            self.stride,
            self.padding,
        )
    }
}

impl BoundPropagation for AveragePoolLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Average pooling is linear with positive weights: y = (1/k) * sum(x_i),
        // so in REAL arithmetic bounds would be exact: y_l = avg_pool(x_l),
        // y_u = avg_pool(x_u). In floating point they are NOT: the window sum
        // below is accumulated in f64 and only the final f64→f32 store is
        // directed-rounded 1 ULP outward — that certifies the CAST, not the f64
        // accumulation residual (up to γ⁶⁴_{k−1}·Σ|x_i|, which under ≥2^29
        // cancellation exceeds 1 f32 ULP of the result, e.g. window
        // [2^30, 2^-30, -2^30]). Verdict paths must use `propagate_ibp_sound`,
        // which folds that residual outward as a certified Higham term.
        // Reference: alpha-beta-CROWN BoundAveragePool sets use_default_ibp=True
        // and uses F.avg_pool2d for forward over NCHW inputs
        // (auto_LiRPA/operators/pooling.py:522-527).

        // Validate input shape: expect (channels, height, width) or
        // (batch, channels, height, width).
        let input_shape = input.lower().shape();
        let ndim = input_shape.len();
        if !(3..=4).contains(&ndim) {
            return Err(NyError::InvalidSpec(format!(
                "AveragePool IBP requires 3D or 4D input, got {}D",
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
            // 3D input: (channels, height, width)
            (None, input_shape[0], input_shape[1], input_shape[2])
        };

        // Handle global average pooling: kernel_size (0, 0) means pool entire spatial dims
        if self.is_global() {
            let out_shape = if let Some(b) = batch_size {
                vec![b, channels, 1, 1]
            } else {
                vec![channels, 1, 1]
            };
            let mut out_lower = ArrayD::zeros(IxDyn(&out_shape));
            let mut out_upper = ArrayD::zeros(IxDyn(&out_shape));
            // Guard zero spatial dimensions to prevent 0/0 = NaN (#2924).
            // Matches the non-global path's count.max(1) pattern.
            let divisor = (in_h * in_w).max(1) as f32;

            let divisor64 = divisor as f64;

            if let Some(b) = batch_size {
                for batch_idx in 0..b {
                    for c in 0..channels {
                        // SOUND: accumulate the window sum in f64 so the running
                        // sum never rounds INWARD (each f32 `+=` rounds to nearest,
                        // which can pull the lower sum up / the upper sum down →
                        // uncertified). f32→f64 widening is exact, so only the f64
                        // sum rounds (a sub-ULP residual), and the single f64→f32
                        // store is then directed-rounded OUTWARD (#3338,
                        // #vnncomp-aw-soundness — mirrors conv becc501 / IBP
                        // batched_interval_matvec_finite).
                        let mut sum_lower = 0.0f64;
                        let mut sum_upper = 0.0f64;
                        for ih in 0..in_h {
                            for iw in 0..in_w {
                                sum_lower += input.lower()[[batch_idx, c, ih, iw]] as f64;
                                sum_upper += input.upper()[[batch_idx, c, ih, iw]] as f64;
                            }
                        }
                        // Directed rounding of the f64→f32 store: lower→-∞,
                        // upper→+∞ (#3338).
                        out_lower[[batch_idx, c, 0, 0]] =
                            next_down_f32((sum_lower / divisor64) as f32);
                        out_upper[[batch_idx, c, 0, 0]] =
                            next_up_f32((sum_upper / divisor64) as f32);
                    }
                }
            } else {
                for c in 0..channels {
                    // SOUND: f64 window-sum accumulation (see batch path above).
                    let mut sum_lower = 0.0f64;
                    let mut sum_upper = 0.0f64;
                    for ih in 0..in_h {
                        for iw in 0..in_w {
                            sum_lower += input.lower()[[c, ih, iw]] as f64;
                            sum_upper += input.upper()[[c, ih, iw]] as f64;
                        }
                    }
                    // Directed rounding of the f64→f32 store: lower→-∞, upper→+∞ (#3338).
                    out_lower[[c, 0, 0]] = next_down_f32((sum_lower / divisor64) as f32);
                    out_upper[[c, 0, 0]] = next_up_f32((sum_upper / divisor64) as f32);
                }
            }

            // NaN in output indicates NaN in input (data corruption, not overflow).
            // Must check before new_repaired which would silently swallow NaN (#2812, #3423).
            if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
                return Err(NyError::NumericalInstability(
                    "AveragePool IBP (global): NaN in bounds (from NaN input)".into(),
                ));
            }
            // Repair non-finite outputs (Inf→fallback) for consistency with linear IBP (#3030).
            return BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative);
        }

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
        let mut out_lower = ArrayD::zeros(IxDyn(&out_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(&out_shape));

        if let Some(b) = batch_size {
            // Handle 4D batch case.
            for batch_idx in 0..b {
                for c in 0..channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let ih_start = oh * sh;
                            let iw_start = ow * sw;

                            // SOUND: f64 window-sum accumulation (the f32 running
                            // sum could round INWARD → uncertified). See the global
                            // path for the full argument (#vnncomp-aw-soundness).
                            let mut sum_lower = 0.0f64;
                            let mut sum_upper = 0.0f64;
                            let mut count = 0usize;

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
                                        sum_lower += input.lower()[[batch_idx, c, ih, iw]] as f64;
                                        sum_upper += input.upper()[[batch_idx, c, ih, iw]] as f64;
                                        count += 1;
                                    } else if self.count_include_pad {
                                        count += 1;
                                    }
                                }
                            }

                            let divisor = if self.count_include_pad {
                                (kh * kw) as f64
                            } else {
                                count.max(1) as f64
                            };

                            // Directed rounding of the f64→f32 store: lower→-∞, upper→+∞ (#3338).
                            out_lower[[batch_idx, c, oh, ow]] =
                                next_down_f32((sum_lower / divisor) as f32);
                            out_upper[[batch_idx, c, oh, ow]] =
                                next_up_f32((sum_upper / divisor) as f32);
                        }
                    }
                }
            }
        } else {
            // Handle 3D (C, H, W) case.
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih_start = oh * sh;
                        let iw_start = ow * sw;

                        // SOUND: f64 window-sum accumulation (the f32 running sum
                        // could round INWARD → uncertified). See the global path
                        // for the full argument (#vnncomp-aw-soundness).
                        let mut sum_lower = 0.0f64;
                        let mut sum_upper = 0.0f64;
                        let mut count = 0usize;

                        for kh_off in 0..kh {
                            for kw_off in 0..kw {
                                let ih = (ih_start + kh_off) as isize - ph as isize;
                                let iw = (iw_start + kw_off) as isize - pw as isize;

                                // SAFETY(as usize): ih/iw are isize, guard ensures >= 0 and < in_h/in_w.
                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    sum_lower += input.lower()[[c, ih, iw]] as f64;
                                    sum_upper += input.upper()[[c, ih, iw]] as f64;
                                    count += 1;
                                } else if self.count_include_pad {
                                    count += 1;
                                }
                            }
                        }

                        let divisor = if self.count_include_pad {
                            (kh * kw) as f64
                        } else {
                            count.max(1) as f64
                        };

                        // Directed rounding of the f64→f32 store: lower→-∞, upper→+∞ (#3338).
                        out_lower[[c, oh, ow]] = next_down_f32((sum_lower / divisor) as f32);
                        out_upper[[c, oh, ow]] = next_up_f32((sum_upper / divisor) as f32);
                    }
                }
            }
        }

        // NaN in output indicates NaN in input (data corruption, not overflow).
        // Must check before new_repaired which would silently swallow NaN (#2812, #3423).
        if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "AveragePool IBP: NaN in bounds (from NaN input)".into(),
            ));
        }
        // Repair non-finite outputs (Inf→fallback) for consistency with linear IBP (#3030).
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // AveragePool CROWN backward is not identity: coefficients must be distributed
        // from pooled outputs back to input spatial positions.
        // Reference: alpha-beta-CROWN BoundAveragePool.bound_backward uses
        // interpolate/conv_transpose to expand A coefficients
        // (auto_LiRPA/operators/pooling.py:529-569).
        Err(NyError::UnsupportedOp(
            "AveragePool CROWN propagation requires bounds - use propagate_linear_with_bounds"
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
        AveragePoolLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl AveragePoolLayer {
    /// SOUND IBP forward — the AveragePool analogue of
    /// `Conv2dLayer::propagate_ibp_sound_with_engine` (#vnncomp-aw-soundness,
    /// avgpool 1-ULP-arm item).
    ///
    /// [`Self::propagate_ibp`] accumulates each window sum in **f64** (each f32
    /// input widens exactly) and directed-rounds only the final f64→f32 store
    /// outward by 1 ULP. That 1 ULP certifies the CAST, but NOT the f64
    /// accumulation residual: the f64 sum of `k` f32 terms deviates from the true
    /// sum by up to `γ⁶⁴_{k−1}·Σ|x_i|` (Higham Thm 3.1, u₆₄ = 2^−53), and under
    /// catastrophic cancellation (`Σ|x_i|/|y| ≳ 2^29`) that residual EXCEEDS
    /// 1 f32 ULP of the result. Example: window `[2^30, 2^-30, -2^30]` sums to
    /// exactly 0.0 in f64 (the 2^-30 is below half-ULP at magnitude 2^30) where
    /// the true sum is 2^-30, and `next_up(0) ≈ 1.4e-45` cannot reach the true
    /// average `2^-30/3 ≈ 3.1e-10` — a "sound" box EXCLUDING the true output on
    /// the verdict / intermediate-bound path.
    ///
    /// # Certified error derivation (uniform weights `+1/d`)
    ///
    /// Per output window with `c ≤ k` in-bounds terms and divisor `d`:
    /// `avg₆₄ = fl₆₄( fl₆₄(Σ x_i) / d )` — `c−1` adds plus 1 divide, i.e. ≤ `c`
    /// roundings each `(1+δ)`, `|δ| ≤ u₆₄`. Higham Thm 3.1 gives
    ///
    /// ```text
    /// |avg₆₄ − (Σ x_i)/d|  ≤  γ⁶⁴_c · (Σ|x_i|)/d  ≤  γ⁶⁴_{k+1} · S_safe
    /// ```
    ///
    /// where `S/d = Σ|x_i|/d` is computed by running the SAME pooling forward on
    /// the degenerate `max(|l|,|u|)` box (all inputs ≥ 0, weights `+1/d` > 0 ⇒
    /// its UPPER endpoint over-estimates the f64 value; identical window
    /// geometry and divisors, so exclude-pad edge windows match exactly — this
    /// mirrors conv's `|kernel|` run and handles 3D/4D/global uniformly), then
    /// inflated by `1/(1 − γ⁶⁴_k) ≥` its own accumulation deficit so
    /// `S_safe ≥ S_true/d` (mirrors conv's `s_inflate`). `k` is the MAX window
    /// term count (`kh·kw`; `in_h·in_w` for global pooling) — γ is monotone in
    /// its index, so `γ_{k+1} ≥ γ_{c}` covers every window; charging `k+1`
    /// (one unit over the counted `c ≤ k` roundings) keeps a documented safety
    /// margin, mirroring conv's `γ_{K+2}`. The final f64→f32 cast is already
    /// covered by `propagate_ibp`'s outward 1-ULP store, and the fold's own f32
    /// subtract/add below is covered by its `next_down`/`next_up` (a nearest
    /// rounding is within half-ULP; the directed step moves a full ULP).
    ///
    /// The f64 gamma is the honest constant here BECAUSE the accumulation is
    /// f64; an f32-accumulated sum would need `γ³²` (2^29× larger — see
    /// `crown_single_gamma_n_f32`'s warning). SOUND: the returned box strictly
    /// encloses the true average-pool output; looser only ⇒ Timeout, never a
    /// false proof.
    pub fn propagate_ibp_sound(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let y = self.propagate_ibp(input)?;

        // S_o/d_o: the SAME pooling forward on the degenerate |x| box.
        let mut xmax = input.lower().mapv(f32::abs);
        ndarray::Zip::from(&mut xmax)
            .and(input.upper())
            .for_each(|m, &u| *m = m.max(u.abs()));
        let s_bt = self.propagate_ibp(&BoundedTensor::concrete(xmax)?)?;
        let s = s_bt.upper(); // xmax ≥ 0, weights +1/d ⇒ upper ≥ f64 value of S/d

        // k = max #terms in one window sum. propagate_ibp above validated
        // ndim ∈ {3, 4} with spatial dims last, so indexing is in range.
        let shape = input.shape();
        let (in_h, in_w) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        let k = if self.is_global() {
            in_h.saturating_mul(in_w).max(1)
        } else {
            self.kernel_size.0.saturating_mul(self.kernel_size.1).max(1)
        };
        // γ⁶⁴ machinery shared with the Linear/conv certificates; saturates to
        // +inf (⇒ ±inf bounds below) for pathologically wide windows.
        let gamma = crate::layers::linear::crown_single_gamma_n_f64(k.saturating_add(1));
        let gamma_k = crate::layers::linear::crown_single_gamma_n_f64(k);
        let s_inflate = if gamma_k < 1.0 {
            1.0 / (1.0 - gamma_k)
        } else {
            f64::INFINITY
        };

        let mut lower = y.lower().to_owned();
        let mut upper = y.upper().to_owned();
        ndarray::Zip::from(&mut lower)
            .and(&mut upper)
            .and(s)
            .for_each(|lo_o, up_o, &s_o| {
                let s_safe = s_o as f64 * s_inflate;
                let err = next_up_f32((gamma * s_safe) as f32);
                if err.is_finite() {
                    *lo_o = next_down_f32(*lo_o - err);
                    *up_o = next_up_f32(*up_o + err);
                } else {
                    *lo_o = f32::NEG_INFINITY;
                    *up_o = f32::INFINITY;
                }
            });
        BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
    }

    /// CROWN backward propagation for AveragePool with pre-activation bounds.
    ///
    /// AveragePool is a linear operation, so the Jacobian is constant.
    /// For each output y[c, oh, ow], it averages inputs x[c, ih, iw] in the pooling window.
    /// The Jacobian entry J[y_flat, x_flat] = 1/k where k is the number of inputs pooled.
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
                "AveragePool CROWN requires 3D or 4D input, got {}D",
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

        // Compute output dimensions
        let (out_h, out_w) = if self.is_global() {
            (1, 1)
        } else {
            self.output_size(in_h, in_w)?
        };

        let (kh, kw) = if self.is_global() {
            (in_h, in_w)
        } else {
            self.kernel_size
        };
        let (sh, sw) = if self.is_global() {
            (1, 1)
        } else {
            self.stride
        };
        let (ph, pw) = if self.is_global() {
            (0, 0)
        } else {
            self.padding
        };

        let input_size = if let Some(b) = batch_size {
            checked_shape_product(&[b, channels, in_h, in_w])
        } else {
            checked_shape_product(&[channels, in_h, in_w])
        }
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "AvgPool2d CROWN: input size product overflows: batch={batch_size:?} ch={channels} h={in_h} w={in_w}"
            ))
        })?;

        let output_size = if let Some(b) = batch_size {
            checked_shape_product(&[b, channels, out_h, out_w])
        } else {
            checked_shape_product(&[channels, out_h, out_w])
        }
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "AvgPool2d CROWN: output size product overflows: batch={batch_size:?} ch={channels} h={out_h} w={out_w}"
            ))
        })?;

        if bounds.num_inputs() != output_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![output_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();

        // Build the transposed Jacobian coefficients for backward propagation
        // A'[out_idx, x_flat] = sum over y_flat of A[out_idx, y_flat] * J[y_flat, x_flat]
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));

        // EXACT per-column composition of any INCOMING certified coefficient error
        // (defensive; #cgan-conv-err-compose generalization). AvgPool backward is an
        // EXACT positive-weight scatter, so the true coefficient interval [A−e, A+e]
        // maps to `new_a ± Σ_w e[y_w]·weight` — the SAME scatter applied to |e|. This
        // replaces the previous LOOSE `row_max(e,i)·weight_l1` uniform over-bound with
        // the tightest sound per-column value (Σ_w e·weight ≤ n·max_w e), so a
        // non-empty incoming err is never under-counted and never wildly over-counted.
        //
        // Accumulated in f64 (each summand is an exact f32→f64 widening; only the
        // running sum rounds) and rounded OUTWARD at assembly. Allocated only when an
        // incoming error is actually present, so the common no-error case is unchanged.
        //
        // NOTE: AvgPool is DELIBERATELY NOT a `propagates_coeff_err` carrier
        // (query.rs). Because it is a SCATTER (many output errors sum into one input
        // column), propagating incurs a triangle-inequality loss and discharging over
        // AvgPool's own CROWN-tightened output box is provably tighter — measured 1.10×
        // (2×2/s2) to 1.26× (3×3/s1) tighter than even this exact composition
        // (test_avgpool_carried_coeff_err_encloses_and_ab_width). So the dispatcher
        // folds the incoming err into the bias BEFORE this runs and this term
        // contributes 0 in production; it is the tight defensive fallback if err ever
        // reaches here via a direct backward call.
        let in_lower_err = bounds.lower_a_err();
        let in_upper_err = bounds.upper_a_err();
        let mut prop_lower = in_lower_err.map(|_| Array2::<f64>::zeros((num_outputs, input_size)));
        let mut prop_upper = in_upper_err.map(|_| Array2::<f64>::zeros((num_outputs, input_size)));

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

        // Iterate over output positions and propagate coefficients backward
        let batch_count = batch_size.unwrap_or(1);

        for b in 0..batch_count {
            let b_opt = if batch_size.is_some() { Some(b) } else { None };

            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let y_flat = out_flat(b_opt, c, oh, ow);

                        // Count valid inputs in this window
                        let ih_start = oh * sh;
                        let iw_start = ow * sw;
                        let mut count = 0usize;

                        for kh_off in 0..kh {
                            for kw_off in 0..kw {
                                let ih = (ih_start + kh_off) as isize - ph as isize;
                                let iw = (iw_start + kw_off) as isize - pw as isize;

                                if (ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize)
                                    || self.count_include_pad
                                {
                                    count += 1;
                                }
                            }
                        }

                        let divisor = if self.count_include_pad {
                            (kh * kw) as f32
                        } else {
                            count.max(1) as f32
                        };
                        let weight = 1.0 / divisor;

                        // Distribute coefficients to input positions
                        for kh_off in 0..kh {
                            for kw_off in 0..kw {
                                let ih = (ih_start + kh_off) as isize - ph as isize;
                                let iw = (iw_start + kw_off) as isize - pw as isize;

                                // SAFETY(as usize): ih/iw are isize, guard ensures >= 0 and < in_h/in_w.
                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let x_flat = in_flat(b_opt, c, ih, iw);

                                    // A'[out_idx, x_flat] += A[out_idx, y_flat] * weight
                                    let w64 = weight as f64;
                                    for out_idx in 0..num_outputs {
                                        new_lower_a[[out_idx, x_flat]] +=
                                            bounds.lower_a()[[out_idx, y_flat]] * weight;
                                        new_upper_a[[out_idx, x_flat]] +=
                                            bounds.upper_a()[[out_idx, y_flat]] * weight;
                                        // EXACT per-column err composition: the SAME
                                        // positive-weight scatter applied to |e_in|.
                                        if let (Some(pl), Some(e)) =
                                            (prop_lower.as_mut(), in_lower_err)
                                        {
                                            pl[[out_idx, x_flat]] +=
                                                (e[[out_idx, y_flat]] as f64).abs() * w64;
                                        }
                                        if let (Some(pu), Some(e)) =
                                            (prop_upper.as_mut(), in_upper_err)
                                        {
                                            pu[[out_idx, x_flat]] +=
                                                (e[[out_idx, y_flat]] as f64).abs() * w64;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // SOUND AvgPool coefficient error (#vnncomp-aw-soundness). The backward
        // coefficient A'[i,x] = Σ_{windows w covering x} A[i,y_w]·weight_w is a
        // contraction computed in round-to-nearest f32 with no directed rounding —
        // the SAME class of bug as Conv2d transpose-conv (conv2d/bound.rs:308) and
        // dense A·W (crown_single.rs). Without a certified error term the fresh
        // rounding error leaks uncertified into concretize, so a bound sitting a few
        // ULP from the threshold could be reported Verified when it is not. Certify
        // it as γ_n·S, mirroring Conv2d's over-bound:
        //   n = max #output windows covering one input pixel = ⌈kh/sh⌉·⌈kw/sw⌉,
        //       capped by the total window count (so global pooling is n = 1).
        //   S[i,x] = Σ_w |A[i,y_w]|·weight_w ≤ row_max(A,i)·(Σ_w weight_w)
        //          ≤ row_max(A,i)·n,  since every weight_w = 1/divisor_w ≤ 1.
        // Per row i this is constant over input pixels. Any INCOMING coeff error is
        // normally discharged into the bias by the dispatcher before this runs (AvgPool
        // is NOT a `propagates_coeff_err` carrier — see query.rs). If it ever reaches
        // here (direct backward call), it is composed EXACTLY per-column into
        // `prop_lower`/`prop_upper` above (same positive-weight scatter) and ADDED to
        // this fresh term below — the tight defensive fallback.
        let nw_h = kh.div_ceil(sh.max(1));
        let nw_w = kw.div_ceil(sw.max(1));
        let n_contraction = nw_h
            .saturating_mul(nw_w)
            .min(out_h.saturating_mul(out_w))
            .max(1);
        // SOUND: the backward coefficient A'[i,x] (lines ~465-468) is accumulated
        // in round-to-nearest f32 (`new_*_a[[..]] += a[..]·weight`), so its error
        // is bounded by the **f32** Higham growth factor γ_n^f32·S (≈ n·2^-24·S),
        // ~2^29× the f64 factor. The previous γ_n^f64 UNDER-counted the true f32
        // accumulation error → false-proof risk (#vnncomp-aw-soundness, same class
        // as conv becc501). The stored coefficient IS the f32 sum, so no cast term
        // is needed (matches conv_coeff_err_matrix's `coeff_f64 = None` mode).
        let gamma = crate::layers::linear::crown_single_gamma_n_f32(n_contraction);
        let weight_l1 = n_contraction as f64; // Σ weights ≤ n · max_weight, max_weight ≤ 1
        let row_max = |a: &Array2<f32>, i: usize| -> f64 {
            let mut m = 0.0f64;
            for k in 0..a.ncols() {
                let v = (a[[i, k]] as f64).abs();
                if v > m {
                    m = v;
                }
            }
            m
        };
        let mut lower_err = Array2::<f32>::zeros(new_lower_a.raw_dim());
        let mut upper_err = Array2::<f32>::zeros(new_upper_a.raw_dim());
        for i in 0..num_outputs {
            // Fresh f32-accumulation error of the coefficient scatter (per row).
            let s_l = gamma * row_max(bounds.lower_a(), i) * weight_l1;
            let s_u = gamma * row_max(bounds.upper_a(), i) * weight_l1;
            for p in 0..lower_err.ncols() {
                // ADD the EXACT per-column propagated incoming error (0.0 when no
                // incoming err). This is strictly ≤ the previous uniform
                // row_max·weight_l1 over-bound (Σ_w e·weight ≤ n·max_w e), and it
                // discharges at a later/tighter box than AvgPool's own output box.
                let pl = prop_lower.as_ref().map_or(0.0, |m| m[[i, p]]);
                let pu = prop_upper.as_ref().map_or(0.0, |m| m[[i, p]]);
                lower_err[[i, p]] = next_up_f32((s_l + pl) as f32);
                upper_err[[i, p]] = next_up_f32((s_u + pu) as f32);
            }
        }

        // Bias passes through unchanged (linear operation has no bias)
        LinearBounds::new_or_conservative_with_err(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
            lower_err,
            upper_err,
        )
    }
}
