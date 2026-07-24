// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, Axis};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};
use std::borrow::Cow;
use tracing::debug;

use super::super::crown_helpers::{
    compute_conv_bias_f64, detect_and_fix_nonfinite_rows, guard_nan_weights,
};
use super::{
    conv1d_forward_backward_coeff_f64, conv1d_forward_batched_gemm, conv1d_single,
    conv1d_transpose_backward_coeff_f64, conv1d_transpose_batched_gemm, conv1d_transpose_forward,
    Conv1dLayer, ConvTranspose1dLayer,
};
use crate::bounds::{nan_propagating_max_zero, nan_propagating_min_zero};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;
impl BoundPropagation for Conv1dLayer {
    /// IBP for Conv1d layer: y = conv1d(x, W) + b
    ///
    /// For x in [l, u], compute y bounds:
    /// - W+ = max(W, 0), W- = min(W, 0)
    /// - lower_y = conv1d(l, W+) + conv1d(u, W-) + b
    /// - upper_y = conv1d(u, W+) + conv1d(l, W-) + b
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let in_c = self.in_channels();

        match input.lower().ndim() {
            2 => {
                // Input shape: (in_channels, length)
                if input.lower().shape()[0] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![in_c],
                        got: vec![input.lower().shape()[0]],
                    });
                }

                // Split kernel into positive and negative parts
                let kernel_pos = self.kernel.mapv(nan_propagating_max_zero);
                let kernel_neg = self.kernel.mapv(nan_propagating_min_zero);

                // Compute bounds using W+/W- splitting
                let lower_from_pos = conv1d_single(
                    input.lower(),
                    &kernel_pos,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )?;
                let lower_from_neg = conv1d_single(
                    input.upper(),
                    &kernel_neg,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )?;
                let mut lower_y = lower_from_pos + lower_from_neg;

                let upper_from_pos = conv1d_single(
                    input.upper(),
                    &kernel_pos,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )?;
                let upper_from_neg = conv1d_single(
                    input.lower(),
                    &kernel_neg,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )?;
                let mut upper_y = upper_from_pos + upper_from_neg;

                // Add bias if present (broadcast over length dimension)
                if let Some(ref b) = self.bias {
                    let out_c = self.out_channels();
                    let out_len = lower_y.shape()[1];

                    for oc in 0..out_c {
                        for ol in 0..out_len {
                            lower_y[[oc, ol]] += b[oc];
                            upper_y[[oc, ol]] += b[oc];
                        }
                    }
                }

                // Repair non-finite outputs for consistency with linear IBP (#3030).
                BoundedTensor::new_repaired(lower_y, upper_y, RepairStrategy::Conservative)
            }
            3 => {
                // Input shape: (batch, in_channels, length)
                if input.lower().shape()[1] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![0, in_c, 0],
                        got: input.lower().shape().to_vec(),
                    });
                }

                // Split kernel into positive and negative parts
                let kernel_pos = self.kernel.mapv(nan_propagating_max_zero);
                let kernel_neg = self.kernel.mapv(nan_propagating_min_zero);

                let batch = input.lower().shape()[0];
                let input_len = input.lower().shape()[2];
                let out_len = self.output_length(input_len)?;
                let out_c = self.out_channels();

                let mut lower_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_len]));
                let mut upper_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_len]));

                for b in 0..batch {
                    let lower_b = input.lower().index_axis(Axis(0), b).to_owned().into_dyn();
                    let upper_b = input.upper().index_axis(Axis(0), b).to_owned().into_dyn();

                    let lower_from_pos = conv1d_single(
                        &lower_b,
                        &kernel_pos,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )?;
                    let lower_from_neg = conv1d_single(
                        &upper_b,
                        &kernel_neg,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )?;
                    let lower_batch = lower_from_pos + lower_from_neg;

                    let upper_from_pos = conv1d_single(
                        &upper_b,
                        &kernel_pos,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )?;
                    let upper_from_neg = conv1d_single(
                        &lower_b,
                        &kernel_neg,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )?;
                    let upper_batch = upper_from_pos + upper_from_neg;

                    for oc in 0..out_c {
                        for ol in 0..out_len {
                            lower_y[[b, oc, ol]] = lower_batch[[oc, ol]];
                            upper_y[[b, oc, ol]] = upper_batch[[oc, ol]];
                        }
                    }
                }

                // Add bias if present (broadcast over batch/length dimension)
                if let Some(ref bias) = self.bias {
                    for b in 0..batch {
                        for oc in 0..out_c {
                            for ol in 0..out_len {
                                lower_y[[b, oc, ol]] += bias[oc];
                                upper_y[[b, oc, ol]] += bias[oc];
                            }
                        }
                    }
                }

                // Repair non-finite outputs for consistency with linear IBP (#3030).
                BoundedTensor::new_repaired(lower_y, upper_y, RepairStrategy::Conservative)
            }
            _ => Err(NyError::ShapeMismatch {
                expected: vec![in_c, 0],
                got: input.lower().shape().to_vec(),
            }),
        }
    }

    /// CROWN backward propagation through Conv1d layer (CPU path).
    ///
    /// Delegates to `propagate_linear_with_engine` with `engine: None`.
    /// For GPU-accelerated path, use `Conv1dLayer::propagate_linear_with_engine`.
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        self.propagate_linear_with_engine(bounds, None)
    }
}

impl Conv1dLayer {
    /// IBP propagation with optional GEMM-engine acceleration for PGD.
    ///
    /// The engine-aware path batches the four W+/W- convolution evaluations
    /// through the existing forward GEMM helper, then falls back to the scalar
    /// CPU implementation if the helper path rejects the input or engine.
    pub fn propagate_ibp_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let Some(engine) = engine else {
            return self.propagate_ibp(input);
        };

        match propagate_ibp_via_gemm(self, input, engine) {
            Ok(bounds) => Ok(bounds),
            Err(err) => {
                debug!("Conv1d IBP GemmEngine path failed, falling back to CPU: {err}");
                self.propagate_ibp(input)
            }
        }
    }

    /// SOUND IBP forward — the Conv1d analogue of `Conv2dLayer::propagate_ibp_sound_with_engine`
    /// (#vnncomp-aw-soundness). The plain forward accumulates each output over
    /// `K = (in_c/groups)·kw` products in round-to-nearest f32 (no f64, no directed rounding),
    /// so under cancellation it can EXCLUDE the true value — unsound as an intermediate /
    /// verdict node bound. This adds the certified Higham error `up(γ_{K+2}·S + 2u·|y|)` with
    /// `S = Σ_k |W_ok|·max(|x_l_k|,|x_u_k|)` (run the SAME forward on `|kernel|` and the
    /// degenerate `max(|l|,|u|)` box), rounded outward. SOUND: strictly encloses; looser only.
    pub fn propagate_ibp_sound_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let y = self.propagate_ibp_with_engine(input, engine)?;
        let mut xmax = input.lower().mapv(f32::abs);
        ndarray::Zip::from(&mut xmax)
            .and(input.upper())
            .for_each(|m, &u| *m = m.max(u.abs()));
        let abs_kernel = self.kernel.mapv(f32::abs);
        let abs_layer = Conv1dLayer::new_full(
            abs_kernel,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )?;
        let s_bt = abs_layer.propagate_ibp_with_engine(&BoundedTensor::concrete(xmax)?, engine)?;
        // Conv1d kernel is (out_c, in_c/groups, kw): shape[1] is already in_c/groups.
        let macs = self.kernel.shape()[1].saturating_mul(self.kernel.shape()[2]);
        super::super::crown_helpers::higham_widen_ibp(&y, s_bt.lower(), macs)
    }

    /// Single forward pass for a concrete (point) input.
    ///
    /// Skips IBP's 4x W+/W- splitting which is unnecessary when lower == upper.
    /// Used by the graph builder's constant pre-evaluation to avoid the 4x overhead
    /// of splitting kernel into positive/negative parts and running four convolutions.
    ///
    /// For concrete x: conv1d(x, W) + b = conv1d(x, W+) + conv1d(x, W-) + b,
    /// so a single conv1d with the full kernel gives the exact result.
    pub fn forward_concrete(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let in_c = self.in_channels();
        match input.ndim() {
            2 => {
                if input.shape()[0] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![in_c],
                        got: vec![input.shape()[0]],
                    });
                }
                let mut output = conv1d_single(
                    input,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )?;
                if let Some(ref b) = self.bias {
                    let out_c = self.out_channels();
                    let out_len = output.shape()[1];
                    for oc in 0..out_c {
                        for ol in 0..out_len {
                            output[[oc, ol]] += b[oc];
                        }
                    }
                }
                Ok(output)
            }
            _ => {
                // Fall back to IBP for batched or unexpected shapes
                let concrete = BoundedTensor::concrete(input.clone())?;
                let out = self.propagate_ibp(&concrete)?;
                Ok(out.lower().clone())
            }
        }
    }

    /// CROWN backward propagation through Conv1d layer with optional GemmEngine (#3598).
    ///
    /// For a conv layer y = conv1d(x, W) + b, and current linear bounds A @ y + c:
    /// - The backward pass through conv is a transposed convolution
    /// - new_A = conv_transpose(A_reshaped, W)
    /// - new_b = A @ b + c (where b is broadcast across spatial positions)
    ///
    /// When `engine` is `Some`, dispatches the transposed convolution to GPU via
    /// batched GEMM. Falls back to faer CPU GEMM when engine is None.
    ///
    /// Requires `input_length` to be set for proper shape computation.
    pub fn propagate_linear_with_engine<'a>(
        &self,
        bounds: &'a LinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Cow<'a, LinearBounds>> {
        debug!("Conv1d layer CROWN backward propagation");

        guard_nan_weights(&self.kernel, self.bias.as_ref(), "Conv1d")?;

        // Get input length (required for CROWN)
        let in_len = self.input_length.ok_or_else(|| NyError::UnsupportedConfiguration(
            "Conv1d CROWN requires input_length to be set. Use with_input_length() or set_input_length().".to_string()
        ))?;

        let in_c = self.in_channels();
        let out_c = self.out_channels();
        let out_len = self.output_length(in_len)?;

        // Verify that bounds dimensions match expected conv output
        let expected_conv_out = out_c * out_len;
        if bounds.num_inputs() != expected_conv_out {
            return Err(NyError::ShapeMismatch {
                expected: vec![expected_conv_out],
                got: vec![bounds.num_inputs()],
            });
        }

        let conv_in_size = in_c * in_len;

        // Batched GEMM: process all objectives in a single GEMM + col2im scatter (#3598).
        // Replaces the per-row conv1d_transpose loop. Dispatches to GPU when engine
        // is available. Reference: Conv2d pattern at conv2d/bound.rs.
        let mut new_lower_a = conv1d_transpose_batched_gemm(
            bounds.lower_a(),
            &self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            out_c,
            out_len,
            in_len,
            engine,
        )?;
        let mut new_upper_a = conv1d_transpose_batched_gemm(
            bounds.upper_a(),
            &self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            out_c,
            out_len,
            in_len,
            engine,
        )?;

        // SOUND coefficient (#vnncomp-aw-soundness — conv f32-accumulation bug):
        // on wide contractions re-accumulate the SAME transpose-conv contraction in
        // f64 + certify `cast_err + γ_n^f64·S`; on small contractions keep the f32
        // GEMM coefficient + certify `γ_n^f32·S` (both sound). n = out_c·k.
        let k = self.kernel.shape()[2];
        let n_contraction = out_c.saturating_mul(k);
        let want_recompute = super::super::crown_helpers::conv_should_f64_recompute(n_contraction);
        let coeff_f64 = want_recompute
            .then(|| {
                conv1d_transpose_backward_coeff_f64(
                    bounds.lower_a(),
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    out_c,
                    out_len,
                    in_len,
                )
                .ok()
            })
            .flatten();
        let coeff_f64_u = want_recompute
            .then(|| {
                conv1d_transpose_backward_coeff_f64(
                    bounds.upper_a(),
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    out_c,
                    out_len,
                    in_len,
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

        let (mut new_lower_b, mut new_upper_b) =
            compute_conv_bias_f64(bounds, self.bias.as_ref(), out_c, out_len);

        // Certified coefficient error `cast + γ·S + prop` (shared helper).
        let kernel_l1: f64 = self.kernel.iter().map(|&v| (v as f64).abs()).sum();
        let mut lower_err = super::super::crown_helpers::conv_coeff_err_matrix(
            bounds.lower_a(),
            bounds.lower_a_err(),
            &new_lower_a,
            coeff_f64.as_ref().filter(|_| lower_recompute_ok),
            kernel_l1,
            n_contraction,
            None,
        );
        let mut upper_err = super::super::crown_helpers::conv_coeff_err_matrix(
            bounds.upper_a(),
            bounds.upper_a_err(),
            &new_upper_a,
            coeff_f64_u.as_ref().filter(|_| upper_recompute_ok),
            kernel_l1,
            n_contraction,
            None,
        );
        let nrows = new_lower_a.nrows();
        // A WANTED-but-failed recompute degrades the row to ±inf bias.
        if lower_recompute_failed {
            for i in 0..nrows {
                for p in 0..new_lower_a.ncols() {
                    new_lower_a[[i, p]] = 0.0;
                    lower_err[[i, p]] = 0.0;
                }
                new_lower_b[i] = f32::NEG_INFINITY;
            }
        }
        if upper_recompute_failed {
            for i in 0..nrows {
                for p in 0..new_upper_a.ncols() {
                    new_upper_a[[i, p]] = 0.0;
                    upper_err[[i, p]] = 0.0;
                }
                new_upper_b[i] = f32::INFINITY;
            }
        }

        detect_and_fix_nonfinite_rows(
            &mut new_lower_a,
            &mut new_upper_a,
            &mut new_lower_b,
            &mut new_upper_b,
            conv_in_size,
            "Conv1d",
        );
        // Zero error on any row that detect_and_fix degraded to ±inf bias (already
        // maximally loose); also keeps shapes consistent for the err attach.
        for i in 0..new_lower_a.nrows() {
            if !new_lower_b[i].is_finite() {
                for p in 0..lower_err.ncols() {
                    lower_err[[i, p]] = 0.0;
                }
            }
            if !new_upper_b[i].is_finite() {
                for p in 0..upper_err.ncols() {
                    upper_err[[i, p]] = 0.0;
                }
            }
        }

        // CROWN backward NaN firewall (#2812): conservative fallback instead of hard error.
        Ok(Cow::Owned(LinearBounds::new_or_conservative_with_err(
            new_lower_a,
            new_lower_b,
            new_upper_a,
            new_upper_b,
            lower_err,
            upper_err,
        )?))
    }
}

fn propagate_ibp_via_gemm(
    layer: &Conv1dLayer,
    input: &BoundedTensor,
    engine: &dyn GemmEngine,
) -> Result<BoundedTensor> {
    let in_c = layer.in_channels();
    let (batch, input_len, squeeze_batch) = match input.lower().ndim() {
        2 => {
            if input.lower().shape()[0] != in_c {
                return Err(NyError::ShapeMismatch {
                    expected: vec![in_c],
                    got: vec![input.lower().shape()[0]],
                });
            }
            (1, input.lower().shape()[1], true)
        }
        3 => {
            if input.lower().shape()[1] != in_c {
                return Err(NyError::ShapeMismatch {
                    expected: vec![0, in_c, 0],
                    got: input.lower().shape().to_vec(),
                });
            }
            (input.lower().shape()[0], input.lower().shape()[2], false)
        }
        _ => {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_c, 0],
                got: input.lower().shape().to_vec(),
            });
        }
    };

    let out_c = layer.out_channels();
    let out_len = layer.output_length(input_len)?;
    let kernel_pos = layer.kernel.mapv(nan_propagating_max_zero);
    let kernel_neg = layer.kernel.mapv(nan_propagating_min_zero);
    let flat_dim = in_c
        .checked_mul(input_len)
        .ok_or_else(|| NyError::InvalidSpec("Conv1d IBP: flat input dims overflow".to_string()))?;

    let lower_flat = input
        .lower()
        .view()
        .into_shape_with_order((batch, flat_dim))
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![batch, flat_dim],
            got: input.lower().shape().to_vec(),
        })?
        .to_owned();
    let upper_flat = input
        .upper()
        .view()
        .into_shape_with_order((batch, flat_dim))
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![batch, flat_dim],
            got: input.upper().shape().to_vec(),
        })?
        .to_owned();

    let lower_from_pos = conv1d_forward_batched_gemm(
        &lower_flat,
        &kernel_pos,
        layer.stride,
        layer.padding,
        layer.dilation,
        layer.groups,
        in_c,
        input_len,
        Some(engine),
    )?;
    let lower_from_neg = conv1d_forward_batched_gemm(
        &upper_flat,
        &kernel_neg,
        layer.stride,
        layer.padding,
        layer.dilation,
        layer.groups,
        in_c,
        input_len,
        Some(engine),
    )?;
    let upper_from_pos = conv1d_forward_batched_gemm(
        &upper_flat,
        &kernel_pos,
        layer.stride,
        layer.padding,
        layer.dilation,
        layer.groups,
        in_c,
        input_len,
        Some(engine),
    )?;
    let upper_from_neg = conv1d_forward_batched_gemm(
        &lower_flat,
        &kernel_neg,
        layer.stride,
        layer.padding,
        layer.dilation,
        layer.groups,
        in_c,
        input_len,
        Some(engine),
    )?;

    let lower_rows = lower_from_pos + lower_from_neg;
    let upper_rows = upper_from_pos + upper_from_neg;
    let mut lower_y = if squeeze_batch {
        lower_rows
            .index_axis(Axis(0), 0)
            .to_owned()
            .into_shape_with_order((out_c, out_len))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![out_c, out_len],
                got: vec![out_c * out_len],
            })?
            .into_dyn()
    } else {
        lower_rows
            .into_shape_with_order((batch, out_c, out_len))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![batch, out_c, out_len],
                got: vec![batch, out_c * out_len],
            })?
            .into_dyn()
    };
    let mut upper_y = if squeeze_batch {
        upper_rows
            .index_axis(Axis(0), 0)
            .to_owned()
            .into_shape_with_order((out_c, out_len))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![out_c, out_len],
                got: vec![out_c * out_len],
            })?
            .into_dyn()
    } else {
        upper_rows
            .into_shape_with_order((batch, out_c, out_len))
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![batch, out_c, out_len],
                got: vec![batch, out_c * out_len],
            })?
            .into_dyn()
    };

    if let Some(bias) = &layer.bias {
        if squeeze_batch {
            for oc in 0..out_c {
                for ol in 0..out_len {
                    lower_y[[oc, ol]] += bias[oc];
                    upper_y[[oc, ol]] += bias[oc];
                }
            }
        } else {
            for b in 0..batch {
                for oc in 0..out_c {
                    for ol in 0..out_len {
                        lower_y[[b, oc, ol]] += bias[oc];
                        upper_y[[b, oc, ol]] += bias[oc];
                    }
                }
            }
        }
    }

    BoundedTensor::new_repaired(lower_y, upper_y, RepairStrategy::Conservative)
}

/// A 1D transposed convolution layer: y = conv_transpose1d(x, W) + b
///
/// Input shape: (batch, in_channels, length) or (in_channels, length)
/// Kernel shape: (in_channels, out_channels, kernel_size) (ONNX ConvTranspose layout)
/// Output shape: (batch, out_channels, out_len) or (out_channels, out_len)
impl BoundPropagation for ConvTranspose1dLayer {
    /// IBP for ConvTranspose1d layer: y = conv_transpose1d(x, W) + b
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let in_c = self.in_channels();

        match input.lower().ndim() {
            2 => {
                // Input shape: (in_channels, length)
                if input.lower().shape()[0] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![in_c],
                        got: vec![input.lower().shape()[0]],
                    });
                }

                let kernel_pos = self.kernel.mapv(nan_propagating_max_zero);
                let kernel_neg = self.kernel.mapv(nan_propagating_min_zero);

                let lower_from_pos = conv1d_transpose_forward(
                    input.lower(),
                    &kernel_pos,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )?;
                let lower_from_neg = conv1d_transpose_forward(
                    input.upper(),
                    &kernel_neg,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )?;
                let mut lower_y = lower_from_pos + lower_from_neg;

                let upper_from_pos = conv1d_transpose_forward(
                    input.upper(),
                    &kernel_pos,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )?;
                let upper_from_neg = conv1d_transpose_forward(
                    input.lower(),
                    &kernel_neg,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )?;
                let mut upper_y = upper_from_pos + upper_from_neg;

                if let Some(ref b) = self.bias {
                    let out_c = self.out_channels();
                    let out_len = lower_y.shape()[1];
                    for oc in 0..out_c {
                        for i in 0..out_len {
                            lower_y[[oc, i]] += b[oc];
                            upper_y[[oc, i]] += b[oc];
                        }
                    }
                }

                // Repair non-finite outputs for consistency with linear IBP (#3030).
                BoundedTensor::new_repaired(lower_y, upper_y, RepairStrategy::Conservative)
            }
            3 => {
                // Input shape: (batch, in_channels, length)
                if input.lower().shape()[1] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![0, in_c, 0],
                        got: input.lower().shape().to_vec(),
                    });
                }

                let kernel_pos = self.kernel.mapv(nan_propagating_max_zero);
                let kernel_neg = self.kernel.mapv(nan_propagating_min_zero);

                let batch = input.lower().shape()[0];
                let input_len = input.lower().shape()[2];
                let out_len = self.output_length(input_len)?;
                let out_c = self.out_channels();

                let mut lower_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_len]));
                let mut upper_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_len]));

                for b in 0..batch {
                    let lower_b = input.lower().index_axis(Axis(0), b).to_owned().into_dyn();
                    let upper_b = input.upper().index_axis(Axis(0), b).to_owned().into_dyn();

                    let lower_from_pos = conv1d_transpose_forward(
                        &lower_b,
                        &kernel_pos,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )?;
                    let lower_from_neg = conv1d_transpose_forward(
                        &upper_b,
                        &kernel_neg,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )?;
                    let lower_batch = lower_from_pos + lower_from_neg;

                    let upper_from_pos = conv1d_transpose_forward(
                        &upper_b,
                        &kernel_pos,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )?;
                    let upper_from_neg = conv1d_transpose_forward(
                        &lower_b,
                        &kernel_neg,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )?;
                    let upper_batch = upper_from_pos + upper_from_neg;

                    for oc in 0..out_c {
                        for i in 0..out_len {
                            lower_y[[b, oc, i]] = lower_batch[[oc, i]];
                            upper_y[[b, oc, i]] = upper_batch[[oc, i]];
                        }
                    }
                }

                if let Some(ref bias) = self.bias {
                    for b in 0..batch {
                        for oc in 0..out_c {
                            for i in 0..out_len {
                                lower_y[[b, oc, i]] += bias[oc];
                                upper_y[[b, oc, i]] += bias[oc];
                            }
                        }
                    }
                }

                // Repair non-finite outputs for consistency with linear IBP (#3030).
                BoundedTensor::new_repaired(lower_y, upper_y, RepairStrategy::Conservative)
            }
            _ => Err(NyError::ShapeMismatch {
                expected: vec![in_c, 0],
                got: input.lower().shape().to_vec(),
            }),
        }
    }

    /// CROWN backward propagation through ConvTranspose1d layer (CPU path).
    ///
    /// Delegates to `propagate_linear_with_engine` with `engine: None`.
    /// For GPU-accelerated path, use `ConvTranspose1dLayer::propagate_linear_with_engine`.
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        self.propagate_linear_with_engine(bounds, None)
    }
}

impl ConvTranspose1dLayer {
    /// IBP propagation with optional GEMM-engine acceleration.
    ///
    /// ConvTranspose1d IBP does not yet have a GEMM-accelerated path; this
    /// delegates to the CPU implementation regardless of engine presence.
    /// Exists for dispatch-site consistency with Conv1d/Conv2d.
    pub fn propagate_ibp_with_engine(
        &self,
        input: &BoundedTensor,
        _engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_ibp(input)
    }

    /// SOUND IBP forward (#vnncomp-aw-soundness) — same Higham construction as
    /// `Conv1dLayer::propagate_ibp_sound_with_engine`, for the transposed conv. The plain
    /// forward f32-accumulates each output over at most `K = (in_c/groups)·kw` scattered
    /// products, so it can EXCLUDE the true value under cancellation; this folds the certified
    /// `up(γ_{K+2}·S + 2u·|y|)` outward (`S = |kernel| transpose-forward on max(|l|,|u|)`).
    pub fn propagate_ibp_sound_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let y = self.propagate_ibp_with_engine(input, engine)?;
        let mut xmax = input.lower().mapv(f32::abs);
        ndarray::Zip::from(&mut xmax)
            .and(input.upper())
            .for_each(|m, &u| *m = m.max(u.abs()));
        let abs_kernel = self.kernel.mapv(f32::abs);
        let abs_layer = ConvTranspose1dLayer::new_full(
            abs_kernel,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )?;
        let s_bt = abs_layer.propagate_ibp_with_engine(&BoundedTensor::concrete(xmax)?, engine)?;
        // Transpose kernel is (in_c, out_c/groups, kw); per-output fan-in <= (in_c/groups)·kw.
        let macs =
            (self.kernel.shape()[0] / self.groups.max(1)).saturating_mul(self.kernel.shape()[2]);
        super::super::crown_helpers::higham_widen_ibp(&y, s_bt.lower(), macs)
    }

    /// Single forward pass for a concrete (point) input.
    ///
    /// Same rationale as `Conv1dLayer::forward_concrete`: avoids the 4x IBP
    /// W+/W- overhead for point inputs in the graph builder's constant chain.
    pub fn forward_concrete(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let in_c = self.in_channels();
        match input.ndim() {
            2 => {
                if input.shape()[0] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![in_c],
                        got: vec![input.shape()[0]],
                    });
                }
                let mut output = conv1d_transpose_forward(
                    input,
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )?;
                if let Some(ref b) = self.bias {
                    let out_c = self.out_channels();
                    let out_len = output.shape()[1];
                    for oc in 0..out_c {
                        for i in 0..out_len {
                            output[[oc, i]] += b[oc];
                        }
                    }
                }
                Ok(output)
            }
            _ => {
                let concrete = BoundedTensor::concrete(input.clone())?;
                let out = self.propagate_ibp(&concrete)?;
                Ok(out.lower().clone())
            }
        }
    }

    /// CROWN backward propagation through ConvTranspose1d layer with optional GemmEngine (#3598).
    ///
    /// For a transposed conv layer y = conv_transpose1d(x, W) + b, and current linear bounds A @ y + c:
    /// - The backward pass through conv_transpose is a regular convolution
    /// - new_A = conv(A_reshaped, W)
    /// - new_b = A @ b + c (where b is broadcast across spatial positions)
    ///
    /// When `engine` is `Some`, dispatches to GPU via batched im2col + GEMM.
    /// Falls back to faer CPU GEMM when engine is None.
    ///
    /// Requires `input_length` to be set for proper shape computation.
    pub fn propagate_linear_with_engine<'a>(
        &self,
        bounds: &'a LinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Cow<'a, LinearBounds>> {
        debug!("ConvTranspose1d layer CROWN backward propagation");

        guard_nan_weights(&self.kernel, self.bias.as_ref(), "ConvTranspose1d")?;

        let in_len = self.input_length.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "ConvTranspose1d CROWN requires input_length to be set. Use with_input_length() or set_input_length()."
                    .to_string(),
            )
        })?;

        let in_c = self.in_channels();
        let out_c = self.out_channels();
        let out_len = self.output_length(in_len)?;

        let expected_conv_out = out_c * out_len;
        if bounds.num_inputs() != expected_conv_out {
            return Err(NyError::ShapeMismatch {
                expected: vec![expected_conv_out],
                got: vec![bounds.num_inputs()],
            });
        }

        let conv_in_size = in_c * in_len;

        // Batched im2col + GEMM: process all objectives at once (#3598).
        // ConvTranspose1d backward = forward conv. The kernel is in ONNX ConvTranspose
        // layout (in_c, out_c/groups, k); conv1d_forward_batched_gemm treats it as
        // grouped Conv1d layout with out_c_conv=kernel[0] and in_c_per_group=kernel[1].
        let mut new_lower_a = conv1d_forward_batched_gemm(
            bounds.lower_a(),
            &self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            out_c,
            out_len,
            engine,
        )?;
        let mut new_upper_a = conv1d_forward_batched_gemm(
            bounds.upper_a(),
            &self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            out_c,
            out_len,
            engine,
        )?;

        // SOUND coefficient (#vnncomp-aw-soundness — conv f32-accumulation bug):
        // re-accumulate the SAME forward-conv contraction in f64, store the
        // directed f32 of it, and certify a per-coefficient `cast_err` plus
        // `γ_n^f64·S`. The forward-conv's INPUT is the ConvTranspose1d output
        // domain: `conv_in_channels = out_c`, `conv_in_len = out_len` — matching
        // the f32 `conv1d_forward_batched_gemm` call above EXACTLY (the kernel is
        // ONNX ConvTranspose layout `(in_c, out_c/groups, k)`, treated as grouped
        // conv with `in_c_per_group = kernel[1] = out_c/groups`, so
        // `in_c_per_group·groups = out_c`). See conv2d/bound.rs for the rationale.
        // On wide contractions re-accumulate in f64 + certify `cast_err + γ_n^f64·S`;
        // on small contractions keep the f32 GEMM coefficient + certify `γ_n^f32·S`
        // (both sound). The forward conv's INPUT is the ConvTranspose1d output
        // domain (`conv_in_channels = out_c`, `conv_in_len = out_len` — matching the
        // f32 call). The contracted width is `kernel[1]·k = in_c_per_group·k`.
        let in_c_per_group = self.kernel.shape()[1];
        let k = self.kernel.shape()[2];
        let n_contraction = in_c_per_group.saturating_mul(k);
        let want_recompute = super::super::crown_helpers::conv_should_f64_recompute(n_contraction);
        let coeff_f64 = want_recompute
            .then(|| {
                conv1d_forward_backward_coeff_f64(
                    bounds.lower_a(),
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
                    bounds.upper_a(),
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
            .is_some_and(|c| c.raw_dim() == new_lower_a.raw_dim());
        let upper_recompute_ok = coeff_f64_u
            .as_ref()
            .is_some_and(|c| c.raw_dim() == new_upper_a.raw_dim());
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

        let (mut new_lower_b, mut new_upper_b) =
            compute_conv_bias_f64(bounds, self.bias.as_ref(), out_c, out_len);

        // Certified coefficient error `cast + γ·S + prop` (shared helper).
        let kernel_l1: f64 = self.kernel.iter().map(|&v| (v as f64).abs()).sum();
        let mut lower_err = super::super::crown_helpers::conv_coeff_err_matrix(
            bounds.lower_a(),
            bounds.lower_a_err(),
            &new_lower_a,
            coeff_f64.as_ref().filter(|_| lower_recompute_ok),
            kernel_l1,
            n_contraction,
            None,
        );
        let mut upper_err = super::super::crown_helpers::conv_coeff_err_matrix(
            bounds.upper_a(),
            bounds.upper_a_err(),
            &new_upper_a,
            coeff_f64_u.as_ref().filter(|_| upper_recompute_ok),
            kernel_l1,
            n_contraction,
            None,
        );
        let nrows = new_lower_a.nrows();
        // A WANTED-but-failed recompute degrades the row to ±inf bias.
        if lower_recompute_failed {
            for i in 0..nrows {
                for p in 0..new_lower_a.ncols() {
                    new_lower_a[[i, p]] = 0.0;
                    lower_err[[i, p]] = 0.0;
                }
                new_lower_b[i] = f32::NEG_INFINITY;
            }
        }
        if upper_recompute_failed {
            for i in 0..nrows {
                for p in 0..new_upper_a.ncols() {
                    new_upper_a[[i, p]] = 0.0;
                    upper_err[[i, p]] = 0.0;
                }
                new_upper_b[i] = f32::INFINITY;
            }
        }

        detect_and_fix_nonfinite_rows(
            &mut new_lower_a,
            &mut new_upper_a,
            &mut new_lower_b,
            &mut new_upper_b,
            conv_in_size,
            "ConvTranspose1d",
        );
        for i in 0..new_lower_a.nrows() {
            if !new_lower_b[i].is_finite() {
                for p in 0..lower_err.ncols() {
                    lower_err[[i, p]] = 0.0;
                }
            }
            if !new_upper_b[i].is_finite() {
                for p in 0..upper_err.ncols() {
                    upper_err[[i, p]] = 0.0;
                }
            }
        }

        // CROWN backward NaN firewall (#2812): conservative fallback instead of hard error.
        Ok(Cow::Owned(LinearBounds::new_or_conservative_with_err(
            new_lower_a,
            new_lower_b,
            new_upper_a,
            new_upper_b,
            lower_err,
            upper_err,
        )?))
    }
}
