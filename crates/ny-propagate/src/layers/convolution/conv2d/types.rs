// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{s, Array1, Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, is_crown_coeff_safe, GemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};
use tracing::debug;

use super::super::crown_helpers::batched_conv_coeff_err;
use super::{
    conv2d_forward_backward_coeff_f64, conv2d_forward_batched_gemm, conv2d_single,
    conv2d_transpose_backward_coeff_f64, conv2d_transpose_batched_gemm_grouped,
};
use crate::{contiguous_flat_slice_mut, BatchedLinearBounds};
#[derive(Debug, Clone)]
pub struct Conv2dLayer {
    /// Convolution kernel of shape (out_channels, in_channels/groups, kernel_h, kernel_w)
    pub kernel: ArrayD<f32>,
    /// Optional bias of shape (out_channels,)
    pub bias: Option<Array1<f32>>,
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Dilation (height, width), spacing between kernel elements. Default (1, 1).
    /// Reference: PyTorch `torch.nn.Conv2d`, ONNX Conv `dilations` attribute.
    pub dilation: (usize, usize),
    /// Groups (number of blocked connections from input to output channels). Default 1.
    /// Reference: PyTorch `torch.nn.Conv2d`, ONNX Conv `group` attribute.
    pub groups: usize,
    /// Input spatial dimensions (height, width) - required for CROWN backward pass
    pub input_shape: Option<(usize, usize)>,
}

impl Conv2dLayer {
    /// Create a new Conv2d layer with groups=1.
    pub fn new(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        Self::new_full(kernel, bias, stride, padding, 1)
    }

    /// Create a new Conv2d layer with explicit groups and dilation=(1, 1).
    ///
    /// Kernel shape: `(out_channels, in_channels/groups, kernel_h, kernel_w)`.
    /// Reference: PyTorch `torch.nn.Conv2d`, ONNX Conv `group` attribute.
    pub fn new_full(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Result<Self> {
        Self::new_dilated(kernel, bias, stride, padding, (1, 1), groups)
    }

    /// Create a new Conv2d layer with explicit dilation and groups.
    ///
    /// Kernel shape: `(out_channels, in_channels/groups, kernel_h, kernel_w)`.
    /// Reference: PyTorch `torch.nn.Conv2d`, ONNX Conv `dilations`/`group` attributes.
    pub fn new_dilated(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Self> {
        if stride.0 == 0 || stride.1 == 0 {
            return Err(NyError::InvalidSpec(format!(
                "Conv2d stride must be >= 1 in both dimensions, got stride=({},{})",
                stride.0, stride.1
            )));
        }
        if dilation.0 == 0 || dilation.1 == 0 {
            return Err(NyError::InvalidSpec(format!(
                "Conv2d dilation must be >= 1 in both dimensions, got dilation=({},{})",
                dilation.0, dilation.1
            )));
        }
        if groups == 0 {
            return Err(NyError::InvalidSpec(
                "Conv2d groups must be >= 1, got 0".to_string(),
            ));
        }
        if kernel.ndim() != 4 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0, 0, 0], // 4D expected
                got: kernel.shape().to_vec(),
            });
        }
        let out_channels = kernel.shape()[0];
        if !out_channels.is_multiple_of(groups) {
            return Err(NyError::InvalidSpec(format!(
                "Conv2d out_channels ({out_channels}) must be divisible by groups ({groups})"
            )));
        }
        if let Some(ref b) = bias {
            if b.len() != out_channels {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_channels],
                    got: vec![b.len()],
                });
            }
        }
        Ok(Self {
            kernel,
            bias,
            stride,
            padding,
            dilation,
            groups,
            input_shape: None,
        })
    }

    /// Create a new Conv2d layer with known input spatial dimensions.
    /// Required for CROWN backward propagation.
    pub fn with_input_shape(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        input_height: usize,
        input_width: usize,
    ) -> Result<Self> {
        let mut layer = Self::new(kernel, bias, stride, padding)?;
        layer.input_shape = Some((input_height, input_width));
        Ok(layer)
    }

    /// Create a new Conv2d layer with groups and known input spatial dimensions.
    pub fn with_input_shape_full(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
        input_height: usize,
        input_width: usize,
    ) -> Result<Self> {
        let mut layer = Self::new_full(kernel, bias, stride, padding, groups)?;
        layer.input_shape = Some((input_height, input_width));
        Ok(layer)
    }

    /// Set the input spatial dimensions. Required for CROWN backward propagation.
    pub fn set_input_shape(&mut self, height: usize, width: usize) {
        self.input_shape = Some((height, width));
    }

    /// Output channels.
    pub fn out_channels(&self) -> usize {
        self.kernel.shape()[0]
    }

    /// Input channels (total, accounting for groups).
    ///
    /// Kernel shape is `(out_c, in_c/groups, kh, kw)`, so total input channels
    /// is `kernel.shape()[1] * groups`.
    pub fn in_channels(&self) -> usize {
        self.kernel.shape()[1] * self.groups
    }

    /// Kernel size (height, width).
    pub fn kernel_size(&self) -> (usize, usize) {
        (self.kernel.shape()[2], self.kernel.shape()[3])
    }

    /// Compute output spatial dimensions.
    ///
    /// Returns an error if the padded input is smaller than the kernel.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> Result<(usize, usize)> {
        let (kh, kw) = self.kernel_size();
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (dh, dw) = self.dilation;
        // Effective (dilated) kernel span: dilation*(kernel-1) + 1.
        let eff_kh = dh * (kh - 1) + 1;
        let eff_kw = dw * (kw - 1) + 1;
        let padded_h = input_h
            .checked_add(2 * ph)
            .ok_or_else(|| NyError::InvalidSpec("Conv2d padded height overflow".to_string()))?;
        let padded_w = input_w
            .checked_add(2 * pw)
            .ok_or_else(|| NyError::InvalidSpec("Conv2d padded width overflow".to_string()))?;
        if padded_h < eff_kh || padded_w < eff_kw {
            return Err(NyError::InvalidSpec(format!(
                "Conv2d effective kernel larger than padded input: input=({input_h},{input_w}), \
                 padding=({ph},{pw}), kernel=({kh},{kw}), dilation=({dh},{dw})"
            )));
        }
        Ok(((padded_h - eff_kh) / sh + 1, (padded_w - eff_kw) / sw + 1))
    }

    /// Batched CROWN backward propagation through Conv2d layer.
    ///
    /// For a conv layer y = conv2d(x, W) + b, with batched linear bounds A @ y + c:
    /// - The backward pass through conv is a transposed convolution
    /// - new_A = conv_transpose(A_reshaped, W)
    /// - new_b = A @ b + c (where b is broadcast across spatial positions)
    ///
    /// BatchedLinearBounds:
    /// - lower_a shape: [...batch, out_dim, out_c * out_h * out_w]
    /// - Reshapes to [...batch, out_dim, out_c, out_h, out_w] for conv_transpose
    /// - Output: [...batch, out_dim, in_c * in_h * in_w]
    ///
    /// Requires `input_shape` to be set for proper shape computation.
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BatchedLinearBounds> {
        debug!("Conv2d layer batched CROWN backward propagation");

        // Get input spatial dimensions (required for CROWN)
        let (in_h, in_w) = self.input_shape.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "Conv2d CROWN requires input_shape to be set. Use with_input_shape() or set_input_shape()."
                    .to_string(),
            )
        })?;

        let in_c = self.in_channels();
        let out_c = self.out_channels();
        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let conv_in_size = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d CROWN: input dims product overflows: {in_c} * {in_h} * {in_w}"
            ))
        })?;
        let conv_out_size = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d CROWN: output dims product overflows: {out_c} * {out_h} * {out_w}"
            ))
        })?;

        let a_shape = bounds.lower_a.shape();
        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let mid_dim = a_shape[a_shape.len() - 1];

        if mid_dim != conv_out_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_dim, conv_out_size],
                got: vec![out_dim, mid_dim],
            });
        }

        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch = checked_shape_product(batch_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d CROWN: batch dims product overflows: {:?}",
                batch_dims
            ))
        })?;
        let total_batch = total_batch.max(1);

        // Output A shape: [...batch, out_dim, in_c * in_h * in_w]
        let mut out_a_shape: Vec<usize> = batch_dims.to_vec();
        out_a_shape.push(out_dim);
        out_a_shape.push(conv_in_size);

        // Output b shape: [...batch, out_dim]
        let mut out_b_shape: Vec<usize> = batch_dims.to_vec();
        out_b_shape.push(out_dim);

        // Reshape A to [total_batch, out_dim, mid_dim] for computation
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;

        // Flatten (total_batch, out_dim, mid_dim) → (total_batch * out_dim, mid_dim)
        // for batched GEMM. Same approach as conv2d/bound.rs propagate_linear (#3382).
        let total_rows = total_batch.checked_mul(out_dim).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d CROWN: total_batch * out_dim overflows: {total_batch} * {out_dim}"
            ))
        })?;
        let lower_a_flat = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_rows, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot flatten lower_a for GEMM".to_string()))?;
        let upper_a_flat = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_rows, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot flatten upper_a for GEMM".to_string()))?;

        // Per-group batched GEMM replaces N per-row conv2d_transpose calls (#3382).
        // #3399: engine parameter enables GPU acceleration via GemmEngine.
        // #3770: groups parameter enables depthwise/grouped convolutions.
        let lower_a_2d = lower_a_flat.to_owned();
        let upper_a_2d = upper_a_flat.to_owned();
        let mut new_lower_a = conv2d_transpose_batched_gemm_grouped(
            &lower_a_2d,
            &self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            (in_h, in_w),
            (out_h, out_w),
            out_c,
            self.groups,
            2, // new_lower_a remains live while new_upper_a is built
            engine,
        )?;
        let mut new_upper_a = conv2d_transpose_batched_gemm_grouped(
            &upper_a_2d,
            &self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            (in_h, in_w),
            (out_h, out_w),
            out_c,
            self.groups,
            2, // both f32 result buffers are retained by the caller
            engine,
        )?;

        // SOUND coefficient (#vnncomp-aw-soundness — conv f32-accumulation bug on
        // the BATCHED β-CROWN/BaB verdict path). This path historically used the
        // f32 GEMM coefficient with NO certified error, yet Conv2d is declared
        // `propagates_coeff_err = true` (the dispatcher trusts it to carry the
        // error) — an UNSOUND false-proof risk identical to the scalar path. Fix
        // it the same way: f64-recompute the SAME contraction, store the directed
        // f32, and attach `cast_err + γ_n^f64·S + prop` (built below). The flattened
        // (total_rows, mid_dim) layout matches `lower_a_2d`/`upper_a_2d`.
        // Small-contraction fast path: skip the f64 recompute and certify the f32
        // GEMM coefficient with `γ_n^f32·S` (sound; tight for small n) — see
        // crown_helpers::conv_should_f64_recompute.
        let (kh_e, kw_e) = self.kernel_size();
        let n_contraction = out_c.saturating_mul(kh_e).saturating_mul(kw_e);
        let want_recompute =
            crate::layers::convolution::crown_helpers::conv_should_f64_recompute(n_contraction);
        let coeff_f64 = want_recompute
            .then(|| {
                conv2d_transpose_backward_coeff_f64(
                    &lower_a_2d,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (in_h, in_w),
                    (out_h, out_w),
                    out_c,
                    self.groups,
                    2, // coeff_f64 remains live through the upper recompute
                )
                .ok()
            })
            .flatten();
        let coeff_f64_u = want_recompute
            .then(|| {
                conv2d_transpose_backward_coeff_f64(
                    &upper_a_2d,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (in_h, in_w),
                    (out_h, out_w),
                    out_c,
                    self.groups,
                    2, // both f64 result buffers are retained below
                )
                .ok()
            })
            .flatten();
        let lower_recompute_ok = coeff_f64
            .as_ref()
            .is_some_and(|c| c.raw_dim() == new_lower_a.raw_dim());
        let upper_recompute_ok = coeff_f64_u
            .as_ref()
            .is_some_and(|c| c.raw_dim() == new_upper_a.raw_dim());
        // Degrade ONLY when a recompute was WANTED (wide conv) but failed — the f32
        // coefficient cannot then be soundly certified with the f64 gamma. On the
        // small-n fast path (`!want_recompute`) the f32 coefficient is kept and
        // certified with `γ_n^f32` (sound), so no degrade.
        let lower_recompute_failed = want_recompute && !lower_recompute_ok;
        let upper_recompute_failed = want_recompute && !upper_recompute_ok;
        if let Some(ref c64) = coeff_f64 {
            if lower_recompute_ok {
                for i in 0..new_lower_a.nrows() {
                    for p in 0..new_lower_a.ncols() {
                        new_lower_a[[i, p]] = c64[[i, p]] as f32;
                    }
                }
            }
        }
        if let Some(ref c64) = coeff_f64_u {
            if upper_recompute_ok {
                for i in 0..new_upper_a.nrows() {
                    for p in 0..new_upper_a.ncols() {
                        new_upper_a[[i, p]] = c64[[i, p]] as f32;
                    }
                }
            }
        }
        // Certified error matrices (`cast + γ·S + prop`, flattened), reshaped to
        // out_a_shape below.
        let kernel_l1: f64 = self.kernel.iter().map(|&v| (v as f64).abs()).sum();
        // Flatten any incoming error to (total_rows, mid_dim) for propagation.
        let in_lower_err_2d = bounds.lower_a_err.as_ref().and_then(|e| {
            e.view()
                .into_shape_with_order((total_rows, mid_dim))
                .ok()
                .map(|v| v.to_owned())
        });
        let in_upper_err_2d = bounds.upper_a_err.as_ref().and_then(|e| {
            e.view()
                .into_shape_with_order((total_rows, mid_dim))
                .ok()
                .map(|v| v.to_owned())
        });
        let mut lower_err_2d = batched_conv_coeff_err(
            &lower_a_2d,
            in_lower_err_2d.as_ref(),
            &new_lower_a,
            coeff_f64.as_ref().filter(|_| lower_recompute_ok),
            kernel_l1,
            n_contraction,
            None,
        );
        let mut upper_err_2d = batched_conv_coeff_err(
            &upper_a_2d,
            in_upper_err_2d.as_ref(),
            &new_upper_a,
            coeff_f64_u.as_ref().filter(|_| upper_recompute_ok),
            kernel_l1,
            n_contraction,
            None,
        );
        // A WANTED-but-failed recompute degrades to ±inf bias (the f32 coefficient
        // cannot be soundly certified with the f64 gamma in that case).
        if lower_recompute_failed {
            new_lower_a.fill(0.0);
            lower_err_2d.fill(0.0);
        }
        if upper_recompute_failed {
            new_upper_a.fill(0.0);
            upper_err_2d.fill(0.0);
        }

        // Track which output rows have unsafe coefficients (#3256, #2812, #3228).
        // Scan GEMM output for non-finite or near-overflow values (same pattern as bound.rs).
        // Rows whose f64 recompute failed are forced unsafe so they degrade to the
        // maximally-loose ±inf bias (the f32-GEMM coefficient cannot be soundly
        // certified with the f64 gamma).
        let mut lower_nonfinite_rows = vec![lower_recompute_failed; total_rows];
        let mut upper_nonfinite_rows = vec![upper_recompute_failed; total_rows];
        for row_idx in 0..total_rows {
            if new_lower_a
                .row(row_idx)
                .iter()
                .any(|v| !is_crown_coeff_safe(*v))
            {
                lower_nonfinite_rows[row_idx] = true;
            }
            if new_upper_a
                .row(row_idx)
                .iter()
                .any(|v| !is_crown_coeff_safe(*v))
            {
                upper_nonfinite_rows[row_idx] = true;
            }
        }

        // #3256: For rows with non-finite A-matrix coefficients, zero the entire row.
        // Bias override happens after bias computation below.
        // Reference: conv2d/bound.rs:328-355 (scalar path).
        let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
        let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
        if lower_affected > 0 || upper_affected > 0 {
            debug!(
                "Conv2d batched CROWN backward: non-finite A-matrix overflow in {}/{} lower rows, \
                 {}/{} upper rows — falling back to ±inf bias for affected rows",
                lower_affected, total_rows, upper_affected, total_rows
            );
            for i in 0..total_rows {
                if lower_nonfinite_rows[i] {
                    for j in 0..conv_in_size {
                        new_lower_a[[i, j]] = 0.0;
                        lower_err_2d[[i, j]] = 0.0;
                    }
                }
                if upper_nonfinite_rows[i] {
                    for j in 0..conv_in_size {
                        new_upper_a[[i, j]] = 0.0;
                        upper_err_2d[[i, j]] = 0.0;
                    }
                }
            }
        }

        // Reshape back to [...batch, out_dim, in_c * in_h * in_w]
        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let new_lower_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?;
        let new_upper_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?;
        // Reshape the certified error to the same out_a_shape for attachment.
        let (lower_err_vec, _) = lower_err_2d.into_raw_vec_and_offset();
        let (upper_err_vec, _) = upper_err_2d.into_raw_vec_and_offset();
        let new_lower_a_err = ArrayD::from_shape_vec(IxDyn(&out_a_shape), lower_err_vec).ok();
        let new_upper_a_err = ArrayD::from_shape_vec(IxDyn(&out_a_shape), upper_err_vec).ok();

        // Compute bias contribution
        let (new_lower_b, new_upper_b) = if let Some(ref bias) = self.bias {
            // For each batch position and output dim: compute sum over spatial positions weighted by bias
            // bias_contrib = sum over (c, h, w) of A[c*out_h*out_w + h*out_w + w] * bias[c]
            let lower_b_3d = bounds
                .lower_b
                .view()
                .into_shape_with_order((total_batch, out_dim))
                .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
            let upper_b_3d = bounds
                .upper_b
                .view()
                .into_shape_with_order((total_batch, out_dim))
                .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

            let mut new_lower_b = Array2::<f64>::zeros((total_batch, out_dim));
            let mut new_upper_b = Array2::<f64>::zeros((total_batch, out_dim));

            for b in 0..total_batch {
                for d in 0..out_dim {
                    let mut lower_sum = 0.0_f64;
                    let mut upper_sum = 0.0_f64;

                    for c in 0..out_c {
                        // Sum all spatial positions for this channel
                        let spatial_start = c * out_h * out_w;
                        let spatial_end = spatial_start + out_h * out_w;

                        let lower_spatial_sum: f64 = lower_a_3d
                            .slice(s![b, d, spatial_start..spatial_end])
                            .iter()
                            .map(|&v| v as f64)
                            .sum();
                        let upper_spatial_sum: f64 = upper_a_3d
                            .slice(s![b, d, spatial_start..spatial_end])
                            .iter()
                            .map(|&v| v as f64)
                            .sum();

                        lower_sum += lower_spatial_sum * bias[c] as f64;
                        upper_sum += upper_spatial_sum * bias[c] as f64;
                    }

                    // NaN guard: inf + (-inf) → conservative bounds.
                    // Same pattern as BatchedLinearBounds::compose().
                    let lb_sum = lower_b_3d[[b, d]] as f64 + lower_sum;
                    let ub_sum = upper_b_3d[[b, d]] as f64 + upper_sum;
                    new_lower_b[[b, d]] = if lb_sum.is_nan() {
                        f64::NEG_INFINITY
                    } else {
                        lb_sum
                    };
                    new_upper_b[[b, d]] = if ub_sum.is_nan() {
                        f64::INFINITY
                    } else {
                        ub_sum
                    };
                }
            }

            // #3256: Override bias for non-finite A-matrix rows before f32 conversion.
            // Setting bias to ±inf with zeroed A-row produces sound [-inf,+inf] bounds.
            for b in 0..total_batch {
                for d in 0..out_dim {
                    let row_idx = b * out_dim + d;
                    if lower_nonfinite_rows[row_idx] {
                        new_lower_b[[b, d]] = f64::NEG_INFINITY;
                    }
                    if upper_nonfinite_rows[row_idx] {
                        new_upper_b[[b, d]] = f64::INFINITY;
                    }
                }
            }

            let new_lower_b_f32 = new_lower_b.mapv(|v| next_down_f32(v as f32));
            let new_upper_b_f32 = new_upper_b.mapv(|v| next_up_f32(v as f32));
            let (new_lower_b_vec, _) = new_lower_b_f32.into_raw_vec_and_offset();
            let (new_upper_b_vec, _) = new_upper_b_f32.into_raw_vec_and_offset();
            (
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
                    .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
                    .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            )
        } else {
            // #3256: Even without conv bias, override for non-finite A-matrix rows.
            let mut lb = bounds.lower_b.clone();
            let mut ub = bounds.upper_b.clone();
            if lower_affected > 0 || upper_affected > 0 {
                let lb_flat = contiguous_flat_slice_mut(&mut lb)?;
                let ub_flat = contiguous_flat_slice_mut(&mut ub)?;
                for i in 0..total_rows {
                    if lower_nonfinite_rows[i] {
                        lb_flat[i] = f32::NEG_INFINITY;
                    }
                    if upper_nonfinite_rows[i] {
                        ub_flat[i] = f32::INFINITY;
                    }
                }
            }
            (lb, ub)
        };

        // Update input shape to reflect the conv layer's input dimensions
        let new_input_shape = if bounds.input_shape.is_empty() {
            vec![conv_in_size]
        } else if bounds.input_shape.len() >= 3 {
            // Update last three dims from [out_c, out_h, out_w] to [in_c, in_h, in_w].
            let mut shape = bounds.input_shape.clone();
            let len = shape.len();
            shape[len - 3] = in_c;
            shape[len - 2] = in_h;
            shape[len - 1] = in_w;
            shape
        } else {
            // Preserve batch dims while updating flattened feature size.
            let mut shape = bounds.input_shape[..bounds.input_shape.len() - 1].to_vec();
            shape.push(conv_in_size);
            shape
        };

        // CROWN backward NaN firewall (#2812): conservative fallback instead of hard error.
        let mut out = BatchedLinearBounds::new_or_conservative(
            new_lower_a,
            new_lower_b,
            new_upper_a,
            new_upper_b,
            new_input_shape,
            bounds.output_shape.clone(),
        )?;
        // Attach the certified coefficient error (#vnncomp-aw-soundness). Conv2d is
        // `propagates_coeff_err = true`, so concretize_sound MUST see this penalty.
        if let (Some(le), Some(ue)) = (new_lower_a_err, new_upper_a_err) {
            out.set_coeff_err(le, ue);
        }
        Ok(out)
    }
}

/// Perform 2D convolution on a single (channels, height, width) input.
///
/// This is a straightforward implementation for correctness testing.
/// For production, use optimized backends (ONNX Runtime, Metal, etc.)

#[derive(Debug, Clone)]
pub struct ConvTranspose2dLayer {
    /// Transposed convolution kernel of shape (in_channels, out_channels, kernel_h, kernel_w)
    pub kernel: ArrayD<f32>,
    /// Optional bias of shape (out_channels,)
    pub bias: Option<Array1<f32>>,
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Dilation (height, width), spacing between kernel elements. Default (1, 1).
    /// Reference: PyTorch `torch.nn.ConvTranspose2d`, ONNX ConvTranspose `dilations`.
    pub dilation: (usize, usize),
    /// Output padding (height, width): extra size added to one side of each
    /// output spatial dimension. Default (0, 0). These extra positions receive
    /// no input contribution (only bias). Reference: ONNX ConvTranspose
    /// `output_padding`, PyTorch `torch.nn.ConvTranspose2d`.
    pub output_padding: (usize, usize),
    /// Input spatial dimensions (height, width) - required for CROWN backward pass
    pub input_shape: Option<(usize, usize)>,
}

impl ConvTranspose2dLayer {
    /// Create a new ConvTranspose2d layer with dilation=(1, 1), output_padding=(0, 0).
    pub fn new(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        Self::new_full(kernel, bias, stride, padding, (1, 1), (0, 0))
    }

    /// Create a new ConvTranspose2d layer with explicit dilation and output_padding.
    ///
    /// `output_padding` must be strictly less than `stride` in each dimension.
    /// This matches the PyTorch/ONNX uniqueness constraint and, critically,
    /// guarantees that the CROWN backward pass (a regular strided convolution
    /// over the padded output grid) recovers the exact input size: the extra
    /// `output_padding` cells at the high end receive no input contribution and
    /// `floor(output_padding/stride) == 0` keeps the recovered input dimension
    /// equal to the true input dimension. Reference: ONNX ConvTranspose.
    pub fn new_full(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        output_padding: (usize, usize),
    ) -> Result<Self> {
        if stride.0 == 0 || stride.1 == 0 {
            return Err(NyError::InvalidSpec(format!(
                "ConvTranspose2d stride must be >= 1 in both dimensions, got stride=({},{})",
                stride.0, stride.1
            )));
        }
        if dilation.0 == 0 || dilation.1 == 0 {
            return Err(NyError::InvalidSpec(format!(
                "ConvTranspose2d dilation must be >= 1 in both dimensions, got dilation=({},{})",
                dilation.0, dilation.1
            )));
        }
        // Require output_padding < stride per dimension (see doc comment): this
        // is what keeps the CROWN backward conv's recovered input size exact.
        if output_padding.0 >= stride.0 || output_padding.1 >= stride.1 {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose2d output_padding ({},{}) must be < stride ({},{}) per dimension \
                 for sound bound propagation",
                output_padding.0, output_padding.1, stride.0, stride.1
            )));
        }
        if kernel.ndim() != 4 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: kernel.shape().to_vec(),
            });
        }
        let out_channels = kernel.shape()[1];
        if let Some(ref b) = bias {
            if b.len() != out_channels {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_channels],
                    got: vec![b.len()],
                });
            }
        }
        Ok(Self {
            kernel,
            bias,
            stride,
            padding,
            dilation,
            output_padding,
            input_shape: None,
        })
    }

    /// Create a new ConvTranspose2d layer with known input spatial dimensions.
    pub fn with_input_shape(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        input_height: usize,
        input_width: usize,
    ) -> Result<Self> {
        let mut layer = Self::new(kernel, bias, stride, padding)?;
        layer.input_shape = Some((input_height, input_width));
        Ok(layer)
    }

    /// Set the input spatial dimensions. Required for CROWN backward propagation.
    pub fn set_input_shape(&mut self, height: usize, width: usize) {
        self.input_shape = Some((height, width));
    }

    /// Output channels.
    pub fn out_channels(&self) -> usize {
        self.kernel.shape()[1]
    }

    /// Input channels.
    pub fn in_channels(&self) -> usize {
        self.kernel.shape()[0]
    }

    /// Kernel size (height, width).
    pub fn kernel_size(&self) -> (usize, usize) {
        (self.kernel.shape()[2], self.kernel.shape()[3])
    }

    /// Compute output spatial dimensions.
    ///
    /// Returns an error if the arithmetic would underflow.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> Result<(usize, usize)> {
        let (kh, kw) = self.kernel_size();
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (dh, dw) = self.dilation;
        let (oph, opw) = self.output_padding;
        // ConvTranspose output:
        //   stride*(in-1) + dilation*(kernel-1) + 1 - 2*pad + output_padding
        let eff_kh = dh * (kh - 1) + 1;
        let eff_kw = dw * (kw - 1) + 1;
        let expanded_h = (input_h.saturating_sub(1))
            .checked_mul(sh)
            .and_then(|v| v.checked_add(eff_kh))
            .and_then(|v| v.checked_add(oph))
            .ok_or_else(|| {
                NyError::InvalidSpec("ConvTranspose2d output height overflow".to_string())
            })?;
        let expanded_w = (input_w.saturating_sub(1))
            .checked_mul(sw)
            .and_then(|v| v.checked_add(eff_kw))
            .and_then(|v| v.checked_add(opw))
            .ok_or_else(|| {
                NyError::InvalidSpec("ConvTranspose2d output width overflow".to_string())
            })?;
        let double_ph = 2 * ph;
        let double_pw = 2 * pw;
        if expanded_h < double_ph || expanded_w < double_pw {
            return Err(NyError::InvalidSpec(format!(
                "ConvTranspose2d output size underflow: \
                 input=({input_h},{input_w}), stride=({sh},{sw}), kernel=({kh},{kw}), \
                 dilation=({dh},{dw}), padding=({ph},{pw}), output_padding=({oph},{opw})"
            )));
        }
        Ok((expanded_h - double_ph, expanded_w - double_pw))
    }

    /// Batched CROWN backward propagation through ConvTranspose2d layer.
    ///
    /// For a transposed conv layer y = conv_transpose(x, W) + b, with batched linear bounds A @ y + c:
    /// - The backward pass through conv_transpose is a regular convolution
    /// - new_A = conv(A_reshaped, W)
    /// - new_b = A @ b + c (where b is broadcast across spatial positions)
    ///
    /// Requires `input_shape` to be set for proper shape computation.
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        self.propagate_linear_batched_maybe_engine(bounds, None)
    }

    /// Batched CROWN backward propagation through ConvTranspose2d with optional GemmEngine.
    pub(crate) fn propagate_linear_batched_maybe_engine(
        &self,
        bounds: &BatchedLinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BatchedLinearBounds> {
        debug!("ConvTranspose2d layer batched CROWN backward propagation");

        let (in_h, in_w) = self.input_shape.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "ConvTranspose2d CROWN requires input_shape to be set. Use with_input_shape() or set_input_shape()."
                    .to_string(),
            )
        })?;

        let in_c = self.in_channels();
        let out_c = self.out_channels();
        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let conv_in_size = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d CROWN: input dims product overflows: {in_c} * {in_h} * {in_w}"
            ))
        })?;
        let conv_out_size = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d CROWN: output dims product overflows: {out_c} * {out_h} * {out_w}"
            ))
        })?;

        let a_shape = bounds.lower_a.shape();
        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let mid_dim = a_shape[a_shape.len() - 1];

        if mid_dim != conv_out_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![out_dim, conv_out_size],
                got: vec![out_dim, mid_dim],
            });
        }

        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch = checked_shape_product(batch_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d CROWN: batch dims product overflows: {:?}",
                batch_dims
            ))
        })?;
        let total_batch = total_batch.max(1);

        // Output A shape: [...batch, out_dim, in_c * in_h * in_w]
        let mut out_a_shape: Vec<usize> = batch_dims.to_vec();
        out_a_shape.push(out_dim);
        out_a_shape.push(conv_in_size);

        // Output b shape: [...batch, out_dim]
        let mut out_b_shape: Vec<usize> = batch_dims.to_vec();
        out_b_shape.push(out_dim);

        // Reshape A to [total_batch, out_dim, mid_dim] for computation
        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((total_batch, out_dim, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((total_batch, out_dim, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;

        let total_rows = total_batch.checked_mul(out_dim).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d CROWN: total_batch * out_dim overflows: {total_batch} * {out_dim}"
            ))
        })?;
        let mut new_lower_a = Array2::zeros((total_rows, conv_in_size));
        let mut new_upper_a = Array2::zeros((total_rows, conv_in_size));
        let mut lower_nonfinite_rows = vec![false; total_rows];
        let mut upper_nonfinite_rows = vec![false; total_rows];

        if engine.is_some() {
            let lower_a_flat = bounds
                .lower_a
                .view()
                .into_shape_with_order((total_rows, mid_dim))
                .map_err(|_| NyError::InvalidSpec("Cannot flatten lower_a for GEMM".to_string()))?
                .to_owned();
            let upper_a_flat = bounds
                .upper_a
                .view()
                .into_shape_with_order((total_rows, mid_dim))
                .map_err(|_| NyError::InvalidSpec("Cannot flatten upper_a for GEMM".to_string()))?
                .to_owned();

            new_lower_a = conv2d_forward_batched_gemm(
                &lower_a_flat,
                &self.kernel,
                self.stride,
                self.padding,
                self.dilation,
                (out_h, out_w),
                engine,
            )?;
            new_upper_a = conv2d_forward_batched_gemm(
                &upper_a_flat,
                &self.kernel,
                self.stride,
                self.padding,
                self.dilation,
                (out_h, out_w),
                engine,
            )?;

            for row_idx in 0..total_rows {
                for col_idx in 0..conv_in_size {
                    if !is_crown_coeff_safe(new_lower_a[[row_idx, col_idx]]) {
                        lower_nonfinite_rows[row_idx] = true;
                        new_lower_a[[row_idx, col_idx]] = 0.0;
                    }
                    if !is_crown_coeff_safe(new_upper_a[[row_idx, col_idx]]) {
                        upper_nonfinite_rows[row_idx] = true;
                        new_upper_a[[row_idx, col_idx]] = 0.0;
                    }
                }
            }
        } else {
            for b in 0..total_batch {
                for d in 0..out_dim {
                    let lower_row = lower_a_3d.slice(s![b, d, ..]);
                    let upper_row = upper_a_3d.slice(s![b, d, ..]);

                    let lower_3d =
                        ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w]), lower_row.to_vec())
                            .map_err(|_| NyError::ShapeMismatch {
                            expected: vec![out_c, out_h, out_w],
                            got: vec![lower_row.len()],
                        })?;

                    let upper_3d =
                        ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w]), upper_row.to_vec())
                            .map_err(|_| NyError::ShapeMismatch {
                            expected: vec![out_c, out_h, out_w],
                            got: vec![upper_row.len()],
                        })?;

                    let lower_conv = conv2d_single(
                        &lower_3d,
                        &self.kernel,
                        self.stride,
                        self.padding,
                        self.dilation,
                    )?;
                    let upper_conv = conv2d_single(
                        &upper_3d,
                        &self.kernel,
                        self.stride,
                        self.padding,
                        self.dilation,
                    )?;

                    let row_idx = b * out_dim + d;
                    for (i, &val) in lower_conv.iter().enumerate() {
                        if is_crown_coeff_safe(val) {
                            new_lower_a[[row_idx, i]] = val;
                        } else {
                            lower_nonfinite_rows[row_idx] = true;
                        }
                    }
                    for (i, &val) in upper_conv.iter().enumerate() {
                        if is_crown_coeff_safe(val) {
                            new_upper_a[[row_idx, i]] = val;
                        } else {
                            upper_nonfinite_rows[row_idx] = true;
                        }
                    }
                }
            }
        }

        // SOUND coefficient (#vnncomp-aw-soundness — conv f32-accumulation bug on
        // the BATCHED β-CROWN/BaB verdict path). ConvTranspose2d is
        // `propagates_coeff_err = true` but stored the f32 forward-conv coefficient
        // with NO certified error → unsound. f64-recompute the SAME forward-conv
        // contraction (conv input = ConvTranspose2d output domain), store the
        // directed f32, attach `cast_err + γ_n^f64·S + prop`.
        let lower_a_2d = lower_a_3d
            .view()
            .into_shape_with_order((total_rows, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot flatten lower_a for f64".to_string()))?
            .to_owned();
        let upper_a_2d = upper_a_3d
            .view()
            .into_shape_with_order((total_rows, mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot flatten upper_a for f64".to_string()))?
            .to_owned();
        let (kh_e, kw_e) = self.kernel_size();
        let in_c_per_group = self.kernel.shape()[1];
        let n_contraction = in_c_per_group.saturating_mul(kh_e).saturating_mul(kw_e);
        let want_recompute =
            crate::layers::convolution::crown_helpers::conv_should_f64_recompute(n_contraction);
        let coeff_f64 = want_recompute
            .then(|| {
                conv2d_forward_backward_coeff_f64(
                    &lower_a_2d,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (out_h, out_w),
                )
                .ok()
            })
            .flatten();
        let coeff_f64_u = want_recompute
            .then(|| {
                conv2d_forward_backward_coeff_f64(
                    &upper_a_2d,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (out_h, out_w),
                )
                .ok()
            })
            .flatten();
        let lower_recompute_ok = coeff_f64
            .as_ref()
            .is_some_and(|c| c.dim() == (total_rows, conv_in_size));
        let upper_recompute_ok = coeff_f64_u
            .as_ref()
            .is_some_and(|c| c.dim() == (total_rows, conv_in_size));
        let lower_recompute_failed = want_recompute && !lower_recompute_ok;
        let upper_recompute_failed = want_recompute && !upper_recompute_ok;
        if let Some(ref c64) = coeff_f64 {
            if lower_recompute_ok {
                for i in 0..total_rows {
                    for p in 0..conv_in_size {
                        new_lower_a[[i, p]] = c64[[i, p]] as f32;
                    }
                }
            }
        }
        if let Some(ref c64) = coeff_f64_u {
            if upper_recompute_ok {
                for i in 0..total_rows {
                    for p in 0..conv_in_size {
                        new_upper_a[[i, p]] = c64[[i, p]] as f32;
                    }
                }
            }
        }
        let kernel_l1: f64 = self.kernel.iter().map(|&v| (v as f64).abs()).sum();
        let in_lower_err_2d = bounds.lower_a_err.as_ref().and_then(|e| {
            e.view()
                .into_shape_with_order((total_rows, mid_dim))
                .ok()
                .map(|v| v.to_owned())
        });
        let in_upper_err_2d = bounds.upper_a_err.as_ref().and_then(|e| {
            e.view()
                .into_shape_with_order((total_rows, mid_dim))
                .ok()
                .map(|v| v.to_owned())
        });
        let mut lower_err_2d = batched_conv_coeff_err(
            &lower_a_2d,
            in_lower_err_2d.as_ref(),
            &new_lower_a,
            coeff_f64.as_ref().filter(|_| lower_recompute_ok),
            kernel_l1,
            n_contraction,
            None,
        );
        let mut upper_err_2d = batched_conv_coeff_err(
            &upper_a_2d,
            in_upper_err_2d.as_ref(),
            &new_upper_a,
            coeff_f64_u.as_ref().filter(|_| upper_recompute_ok),
            kernel_l1,
            n_contraction,
            None,
        );
        if lower_recompute_failed {
            new_lower_a.fill(0.0);
            lower_err_2d.fill(0.0);
            for r in lower_nonfinite_rows.iter_mut() {
                *r = true;
            }
        }
        if upper_recompute_failed {
            new_upper_a.fill(0.0);
            upper_err_2d.fill(0.0);
            for r in upper_nonfinite_rows.iter_mut() {
                *r = true;
            }
        }

        // #3256: Zero rows with unsafe A-matrix coefficients.
        let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
        let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
        if lower_affected > 0 || upper_affected > 0 {
            debug!(
                "ConvTranspose2d batched CROWN backward: non-finite A-matrix in {}/{} lower, \
                 {}/{} upper rows — ±inf bias fallback",
                lower_affected, total_rows, upper_affected, total_rows
            );
            for i in 0..total_rows {
                if lower_nonfinite_rows[i] {
                    for j in 0..conv_in_size {
                        new_lower_a[[i, j]] = 0.0;
                        lower_err_2d[[i, j]] = 0.0;
                    }
                }
                if upper_nonfinite_rows[i] {
                    for j in 0..conv_in_size {
                        new_upper_a[[i, j]] = 0.0;
                        upper_err_2d[[i, j]] = 0.0;
                    }
                }
            }
        }

        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let new_lower_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?;
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let new_upper_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?;
        let (lower_err_vec, _) = lower_err_2d.into_raw_vec_and_offset();
        let (upper_err_vec, _) = upper_err_2d.into_raw_vec_and_offset();
        let new_lower_a_err = ArrayD::from_shape_vec(IxDyn(&out_a_shape), lower_err_vec).ok();
        let new_upper_a_err = ArrayD::from_shape_vec(IxDyn(&out_a_shape), upper_err_vec).ok();

        // Compute bias contribution
        let (new_lower_b, new_upper_b) = if let Some(ref bias) = self.bias {
            let lower_b_3d = bounds
                .lower_b
                .view()
                .into_shape_with_order((total_batch, out_dim))
                .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
            let upper_b_3d = bounds
                .upper_b
                .view()
                .into_shape_with_order((total_batch, out_dim))
                .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

            let mut new_lower_b = Array2::<f64>::zeros((total_batch, out_dim));
            let mut new_upper_b = Array2::<f64>::zeros((total_batch, out_dim));

            for b in 0..total_batch {
                for d in 0..out_dim {
                    let mut lower_sum = 0.0_f64;
                    let mut upper_sum = 0.0_f64;

                    for c in 0..out_c {
                        let spatial_start = c * out_h * out_w;
                        let spatial_end = spatial_start + out_h * out_w;

                        let lower_spatial_sum: f64 = lower_a_3d
                            .slice(s![b, d, spatial_start..spatial_end])
                            .iter()
                            .map(|&v| v as f64)
                            .sum();
                        let upper_spatial_sum: f64 = upper_a_3d
                            .slice(s![b, d, spatial_start..spatial_end])
                            .iter()
                            .map(|&v| v as f64)
                            .sum();

                        lower_sum += lower_spatial_sum * bias[c] as f64;
                        upper_sum += upper_spatial_sum * bias[c] as f64;
                    }

                    // NaN guard for transposed conv2d bias accumulation.
                    let lb_sum = lower_b_3d[[b, d]] as f64 + lower_sum;
                    let ub_sum = upper_b_3d[[b, d]] as f64 + upper_sum;
                    new_lower_b[[b, d]] = if lb_sum.is_nan() {
                        f64::NEG_INFINITY
                    } else {
                        lb_sum
                    };
                    new_upper_b[[b, d]] = if ub_sum.is_nan() {
                        f64::INFINITY
                    } else {
                        ub_sum
                    };
                }
            }

            // #3256: Override bias for non-finite A-matrix rows.
            for b in 0..total_batch {
                for d in 0..out_dim {
                    let row_idx = b * out_dim + d;
                    if lower_nonfinite_rows[row_idx] {
                        new_lower_b[[b, d]] = f64::NEG_INFINITY;
                    }
                    if upper_nonfinite_rows[row_idx] {
                        new_upper_b[[b, d]] = f64::INFINITY;
                    }
                }
            }

            let new_lower_b_f32 = new_lower_b.mapv(|v| next_down_f32(v as f32));
            let new_upper_b_f32 = new_upper_b.mapv(|v| next_up_f32(v as f32));
            let (new_lower_b_vec, _) = new_lower_b_f32.into_raw_vec_and_offset();
            let (new_upper_b_vec, _) = new_upper_b_f32.into_raw_vec_and_offset();
            (
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
                    .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
                ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
                    .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            )
        } else {
            // #3256: Even without conv bias, override for non-finite A-matrix rows.
            let mut lb = bounds.lower_b.clone();
            let mut ub = bounds.upper_b.clone();
            if lower_affected > 0 || upper_affected > 0 {
                let lb_flat = contiguous_flat_slice_mut(&mut lb)?;
                let ub_flat = contiguous_flat_slice_mut(&mut ub)?;
                for i in 0..total_rows {
                    if lower_nonfinite_rows[i] {
                        lb_flat[i] = f32::NEG_INFINITY;
                    }
                    if upper_nonfinite_rows[i] {
                        ub_flat[i] = f32::INFINITY;
                    }
                }
            }
            (lb, ub)
        };

        let new_input_shape = if bounds.input_shape.is_empty() {
            vec![conv_in_size]
        } else if bounds.input_shape.len() >= 3 {
            let mut shape = bounds.input_shape.clone();
            let len = shape.len();
            shape[len - 3] = in_c;
            shape[len - 2] = in_h;
            shape[len - 1] = in_w;
            shape
        } else {
            let mut shape = bounds.input_shape[..bounds.input_shape.len() - 1].to_vec();
            shape.push(conv_in_size);
            shape
        };

        // CROWN backward NaN firewall (#2812): conservative fallback instead of hard error.
        let mut out = BatchedLinearBounds::new_or_conservative(
            new_lower_a,
            new_lower_b,
            new_upper_a,
            new_upper_b,
            new_input_shape,
            bounds.output_shape.clone(),
        )?;
        // Attach the certified coefficient error (#vnncomp-aw-soundness).
        if let (Some(le), Some(ue)) = (new_lower_a_err, new_upper_a_err) {
            out.set_coeff_err(le, ue);
        }
        Ok(out)
    }
}
