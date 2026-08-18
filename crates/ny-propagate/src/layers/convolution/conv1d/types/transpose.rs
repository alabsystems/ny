// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ConvTranspose1dLayer` — transposed 1D convolution with batched CROWN backward propagation.

use ndarray::{s, Array1, Array2, ArrayD, IxDyn};
use ny_core::{is_crown_coeff_safe, GemmEngine, NyError, Result};
use tracing::debug;

use super::common::{
    build_backward_batch_context, finalize_bias_bounds, flattened_input_shape, zero_nonfinite_rows,
};
use crate::layers::convolution::conv1d::{
    conv1d_forward_backward_coeff_f64, conv1d_forward_batched_gemm, conv1d_single,
};
use crate::layers::convolution::crown_helpers::batched_conv_coeff_err;
use crate::BatchedLinearBounds;

/// Transposed 1D convolution for CROWN backward pass.
///
/// Given gradient at conv output, compute gradient at conv input.
/// This is the inverse operation of conv1d in the gradient sense.
#[derive(Debug, Clone)]
pub struct ConvTranspose1dLayer {
    /// Transposed convolution kernel of shape (in_channels, out_channels/groups, kernel_size)
    pub kernel: ArrayD<f32>,
    /// Optional bias of shape (out_channels,)
    pub bias: Option<Array1<f32>>,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Dilation (spacing between kernel elements). Default 1.
    pub dilation: usize,
    /// Groups (number of blocked channel groups). Default 1.
    pub groups: usize,
    /// Input length (required for CROWN backward propagation)
    pub input_length: Option<usize>,
}

impl ConvTranspose1dLayer {
    /// Create a new ConvTranspose1d layer with dilation=1, groups=1.
    pub fn new(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        Self::new_full(kernel, bias, stride, padding, 1, 1)
    }

    /// Create a new ConvTranspose1d layer with explicit dilation and groups.
    ///
    /// Kernel shape: `(in_channels, out_channels/groups, kernel_size)`.
    /// Reference: PyTorch `torch.nn.ConvTranspose1d` documentation.
    pub fn new_full(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        if stride == 0 {
            return Err(NyError::InvalidSpec(
                "ConvTranspose1d stride must be >= 1, got 0".to_string(),
            ));
        }
        if dilation == 0 {
            return Err(NyError::InvalidSpec(
                "ConvTranspose1d dilation must be >= 1, got 0".to_string(),
            ));
        }
        if groups == 0 {
            return Err(NyError::InvalidSpec(
                "ConvTranspose1d groups must be >= 1, got 0".to_string(),
            ));
        }
        if kernel.ndim() != 3 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: kernel.shape().to_vec(),
            });
        }
        let in_channels = kernel.shape()[0];
        let output_channels_per_group = kernel.shape()[1];
        let kernel_size = kernel.shape()[2];
        if in_channels == 0 || output_channels_per_group == 0 || kernel_size == 0 {
            return Err(NyError::InvalidSpec(format!(
                "ConvTranspose1d kernel dimensions must be nonzero, got {:?}",
                kernel.shape()
            )));
        }
        if !in_channels.is_multiple_of(groups) {
            return Err(NyError::InvalidSpec(format!(
                "ConvTranspose1d in_channels ({in_channels}) must be divisible by groups ({groups})"
            )));
        }
        let out_channels = output_channels_per_group
            .checked_mul(groups)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                "ConvTranspose1d output channels overflow: {output_channels_per_group} * {groups}"
            ))
            })?;
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
            input_length: None,
        })
    }

    /// Create a new ConvTranspose1d layer with input length specified.
    pub fn with_input_length(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: usize,
        padding: usize,
        input_length: usize,
    ) -> Result<Self> {
        let mut layer = Self::new(kernel, bias, stride, padding)?;
        layer.input_length = Some(input_length);
        Ok(layer)
    }

    /// Create a new ConvTranspose1d layer with all parameters including dilation and groups.
    pub fn with_input_length_full(
        kernel: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        input_length: usize,
    ) -> Result<Self> {
        let mut layer = Self::new_full(kernel, bias, stride, padding, dilation, groups)?;
        layer.input_length = Some(input_length);
        Ok(layer)
    }

    /// Set the input length (required for CROWN backward propagation).
    pub fn set_input_length(&mut self, input_length: usize) {
        self.input_length = Some(input_length);
    }

    /// Output channels.
    pub fn out_channels(&self) -> usize {
        self.kernel.shape()[1] * self.groups
    }

    /// Input channels.
    pub fn in_channels(&self) -> usize {
        self.kernel.shape()[0]
    }

    /// Kernel size.
    pub fn kernel_size(&self) -> usize {
        self.kernel.shape()[2]
    }

    /// Compute output length.
    ///
    /// Returns an error if the arithmetic would underflow (e.g. kernel < 2*padding).
    pub fn output_length(&self, input_len: usize) -> Result<usize> {
        let k = self.kernel_size();
        if self.stride == 0 || self.dilation == 0 || k == 0 {
            return Err(NyError::InvalidSpec(
                "ConvTranspose1d output length requires nonzero stride, dilation, and kernel \
                 length"
                    .to_string(),
            ));
        }
        let effective_k = k
            .checked_sub(1)
            .and_then(|extent| extent.checked_mul(self.dilation))
            .and_then(|extent| extent.checked_add(1))
            .ok_or_else(|| {
                NyError::InvalidSpec("ConvTranspose1d effective kernel overflow".to_string())
            })?;
        let expanded = input_len
            .checked_sub(1)
            .and_then(|extent| extent.checked_mul(self.stride))
            .and_then(|v| v.checked_add(effective_k))
            .ok_or_else(|| {
                NyError::InvalidSpec("ConvTranspose1d output length overflow".to_string())
            })?;
        let double_pad = self
            .padding
            .checked_mul(2)
            .ok_or_else(|| NyError::InvalidSpec("ConvTranspose1d padding overflow".to_string()))?;
        if expanded < double_pad {
            return Err(NyError::InvalidSpec(format!(
                "ConvTranspose1d output length underflow: \
                 (input_len={input_len} - 1) * stride={} + effective_kernel={} < 2 * padding={}",
                self.stride, effective_k, self.padding
            )));
        }
        Ok(expanded - double_pad)
    }

    /// Batched CROWN backward propagation through ConvTranspose1d layer.
    ///
    /// For a transposed conv layer y = conv_transpose1d(x, W) + b, with batched linear bounds A @ y + c:
    /// - The backward pass through conv_transpose is a regular convolution
    /// - new_A = conv(A_reshaped, W)
    /// - new_b = A @ b + c (where b is broadcast across spatial positions)
    ///
    /// Requires `input_length` to be set for proper shape computation.
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        self.propagate_linear_batched_maybe_engine(bounds, None)
    }

    /// Batched CROWN backward propagation through ConvTranspose1d with optional GemmEngine.
    pub(crate) fn propagate_linear_batched_maybe_engine(
        &self,
        bounds: &BatchedLinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BatchedLinearBounds> {
        debug!("ConvTranspose1d layer batched CROWN backward propagation");

        let in_len = self.input_length.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "ConvTranspose1d CROWN requires input_length to be set. Use with_input_length() or set_input_length()."
                    .to_string(),
            )
        })?;

        let in_c = self.in_channels();
        let out_c = self.out_channels();
        let out_len = self.output_length(in_len)?;
        let conv_in_size = in_c * in_len;
        let conv_out_size = out_c * out_len;

        let ctx =
            build_backward_batch_context(bounds, conv_out_size, conv_in_size, "ConvTranspose1d")?;

        let lower_a_3d = bounds
            .lower_a
            .view()
            .into_shape_with_order((ctx.total_batch, ctx.out_dim, ctx.mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
        let upper_a_3d = bounds
            .upper_a
            .view()
            .into_shape_with_order((ctx.total_batch, ctx.out_dim, ctx.mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;

        let mut new_lower_a = Array2::zeros((ctx.total_rows, conv_in_size));
        let mut new_upper_a = Array2::zeros((ctx.total_rows, conv_in_size));
        let mut lower_nonfinite_rows = vec![false; ctx.total_rows];
        let mut upper_nonfinite_rows = vec![false; ctx.total_rows];

        if engine.is_some() {
            let lower_a_flat = bounds
                .lower_a
                .view()
                .into_shape_with_order((ctx.total_rows, ctx.mid_dim))
                .map_err(|_| NyError::InvalidSpec("Cannot flatten lower_a for GEMM".to_string()))?
                .to_owned();
            let upper_a_flat = bounds
                .upper_a
                .view()
                .into_shape_with_order((ctx.total_rows, ctx.mid_dim))
                .map_err(|_| NyError::InvalidSpec("Cannot flatten upper_a for GEMM".to_string()))?
                .to_owned();

            new_lower_a = conv1d_forward_batched_gemm(
                &lower_a_flat,
                &self.kernel,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                out_c,
                out_len,
                engine,
            )?;
            new_upper_a = conv1d_forward_batched_gemm(
                &upper_a_flat,
                &self.kernel,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                out_c,
                out_len,
                engine,
            )?;

            for row_idx in 0..ctx.total_rows {
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
            for b in 0..ctx.total_batch {
                for d in 0..ctx.out_dim {
                    let lower_row = lower_a_3d.slice(s![b, d, ..]);
                    let upper_row = upper_a_3d.slice(s![b, d, ..]);

                    let lower_2d =
                        ArrayD::from_shape_vec(IxDyn(&[out_c, out_len]), lower_row.to_vec())
                            .map_err(|_| NyError::ShapeMismatch {
                                expected: vec![out_c, out_len],
                                got: vec![lower_row.len()],
                            })?;

                    let upper_2d =
                        ArrayD::from_shape_vec(IxDyn(&[out_c, out_len]), upper_row.to_vec())
                            .map_err(|_| NyError::ShapeMismatch {
                                expected: vec![out_c, out_len],
                                got: vec![upper_row.len()],
                            })?;

                    let lower_conv = conv1d_single(
                        &lower_2d,
                        &self.kernel,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )?;
                    let upper_conv = conv1d_single(
                        &upper_2d,
                        &self.kernel,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )?;

                    let row_idx = b * ctx.out_dim + d;
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
        // the BATCHED β-CROWN/BaB verdict path). ConvTranspose1d is
        // `propagates_coeff_err = true`, but this path stored the f32 forward-conv
        // coefficient with NO certified error → unsound. f64-recompute the SAME
        // forward-conv contraction (conv input = ConvTranspose1d output domain:
        // conv_in_channels = out_c, conv_in_len = out_len, matching the f32 GEMM
        // call), store the directed f32, and attach `cast_err + γ_n^f64·S + prop`.
        let lower_a_2d = lower_a_3d
            .view()
            .into_shape_with_order((ctx.total_rows, ctx.mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot flatten lower_a for f64".to_string()))?
            .to_owned();
        let upper_a_2d = upper_a_3d
            .view()
            .into_shape_with_order((ctx.total_rows, ctx.mid_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot flatten upper_a for f64".to_string()))?
            .to_owned();
        let n_contraction = self.kernel.shape()[1].saturating_mul(self.kernel.shape()[2]);
        let want_recompute =
            crate::layers::convolution::crown_helpers::conv_should_f64_recompute(n_contraction);
        let coeff_f64 = want_recompute
            .then(|| {
                conv1d_forward_backward_coeff_f64(
                    &lower_a_2d,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    out_c,
                    out_len,
                )
                .ok()
            })
            .flatten();
        let coeff_f64_u = want_recompute
            .then(|| {
                conv1d_forward_backward_coeff_f64(
                    &upper_a_2d,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    out_c,
                    out_len,
                )
                .ok()
            })
            .flatten();
        let lower_recompute_ok = coeff_f64
            .as_ref()
            .is_some_and(|c| c.dim() == (ctx.total_rows, conv_in_size));
        let upper_recompute_ok = coeff_f64_u
            .as_ref()
            .is_some_and(|c| c.dim() == (ctx.total_rows, conv_in_size));
        let lower_recompute_failed = want_recompute && !lower_recompute_ok;
        let upper_recompute_failed = want_recompute && !upper_recompute_ok;
        if let Some(ref c64) = coeff_f64 {
            if lower_recompute_ok {
                for i in 0..ctx.total_rows {
                    for p in 0..conv_in_size {
                        new_lower_a[[i, p]] = c64[[i, p]] as f32;
                    }
                }
            }
        }
        if let Some(ref c64) = coeff_f64_u {
            if upper_recompute_ok {
                for i in 0..ctx.total_rows {
                    for p in 0..conv_in_size {
                        new_upper_a[[i, p]] = c64[[i, p]] as f32;
                    }
                }
            }
        }
        let kernel_l1: f64 = self.kernel.iter().map(|&v| (v as f64).abs()).sum();
        let in_lower_err_2d = bounds.lower_a_err.as_ref().and_then(|e| {
            e.view()
                .into_shape_with_order((ctx.total_rows, ctx.mid_dim))
                .ok()
                .map(|v| v.to_owned())
        });
        let in_upper_err_2d = bounds.upper_a_err.as_ref().and_then(|e| {
            e.view()
                .into_shape_with_order((ctx.total_rows, ctx.mid_dim))
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
        zero_nonfinite_rows(
            &mut new_lower_a,
            &mut new_upper_a,
            &lower_nonfinite_rows,
            &upper_nonfinite_rows,
            conv_in_size,
            ctx.total_rows,
            "ConvTranspose1d",
        );
        for i in 0..ctx.total_rows {
            if lower_nonfinite_rows[i] {
                for p in 0..conv_in_size {
                    lower_err_2d[[i, p]] = 0.0;
                }
            }
            if upper_nonfinite_rows[i] {
                for p in 0..conv_in_size {
                    upper_err_2d[[i, p]] = 0.0;
                }
            }
        }

        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let new_lower_a = ArrayD::from_shape_vec(IxDyn(&ctx.out_a_shape), new_lower_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?;
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let new_upper_a = ArrayD::from_shape_vec(IxDyn(&ctx.out_a_shape), new_upper_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?;
        let (lower_err_vec, _) = lower_err_2d.into_raw_vec_and_offset();
        let (upper_err_vec, _) = upper_err_2d.into_raw_vec_and_offset();
        let new_lower_a_err = ArrayD::from_shape_vec(IxDyn(&ctx.out_a_shape), lower_err_vec).ok();
        let new_upper_a_err = ArrayD::from_shape_vec(IxDyn(&ctx.out_a_shape), upper_err_vec).ok();

        let (new_lower_b, new_upper_b) = finalize_bias_bounds(
            bounds,
            &ctx,
            out_c,
            out_len,
            lower_a_3d,
            upper_a_3d,
            self.bias.as_ref(),
            &lower_nonfinite_rows,
            &upper_nonfinite_rows,
        )?;

        let new_input_shape = flattened_input_shape(bounds, conv_in_size);

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
