// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn as DynamicIndex};
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};
use std::borrow::Cow;
use std::time::Instant as DeadlineInstant;
use tracing::debug;

use super::super::crown_helpers::{
    compute_conv_bias_f64, detect_and_fix_nonfinite_rows, guard_nan_weights,
};
use super::{conv2d_transpose_forward, ConvTranspose2dLayer};
use crate::bounds::{nan_propagating_max_zero, nan_propagating_min_zero};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

const DEADLINE_CONV_TRANSPOSE_IBP_POLL_OPS: usize = 4_096;
const DEADLINE_CONV_TRANSPOSE_IBP_MAX_OUTPUT_ELEMENTS: usize = 4 * 1024 * 1024;

#[inline]
fn contains_binary32_subnormal(values: &ArrayD<f32>) -> bool {
    values.iter().any(|value| {
        let magnitude = value.to_bits() & 0x7fff_ffff;
        magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits()
    })
}

#[inline]
fn is_binary32_subnormal(value: f32) -> bool {
    let magnitude = value.to_bits() & 0x7fff_ffff;
    magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits()
}

fn convtranspose2d_deadline_contains_subnormal<I>(
    values: I,
    deadline: DeadlineInstant,
    stage: &str,
) -> Result<bool>
where
    I: IntoIterator<Item = f32>,
{
    for (index, value) in values.into_iter().enumerate() {
        if index.is_multiple_of(DEADLINE_CONV_TRANSPOSE_IBP_POLL_OPS) {
            check_convtranspose2d_ibp_deadline(deadline, stage)?;
        }
        if is_binary32_subnormal(value) {
            return Ok(true);
        }
    }
    check_convtranspose2d_ibp_deadline(deadline, stage)?;
    Ok(false)
}

#[inline]
fn check_convtranspose2d_ibp_deadline(deadline: DeadlineInstant, stage: &str) -> Result<()> {
    if DeadlineInstant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(format!(
            "ConvTranspose2d IBP forward: deadline exceeded {stage}"
        )));
    }
    Ok(())
}

struct ConvTranspose2dDeadlineGeometry {
    batched: bool,
    batch: usize,
    in_c: usize,
    out_c: usize,
    input_h: usize,
    input_w: usize,
    out_h: usize,
    out_w: usize,
    output_elements: usize,
    output_shape: Vec<usize>,
}

fn convtranspose2d_deadline_geometry(
    layer: &ConvTranspose2dLayer,
    input: &BoundedTensor,
) -> Result<ConvTranspose2dDeadlineGeometry> {
    let (in_c, out_c) = layer.validate_geometry()?;
    let (batched, batch, input_h, input_w) = match input.lower().ndim() {
        3 => {
            if input.lower().shape()[0] != in_c {
                return Err(NyError::ShapeMismatch {
                    expected: vec![in_c],
                    got: vec![input.lower().shape()[0]],
                });
            }
            (false, 1, input.lower().shape()[1], input.lower().shape()[2])
        }
        4 => {
            if input.lower().shape()[1] != in_c {
                return Err(NyError::ShapeMismatch {
                    expected: vec![0, in_c, 0, 0],
                    got: input.lower().shape().to_vec(),
                });
            }
            (
                true,
                input.lower().shape()[0],
                input.lower().shape()[2],
                input.lower().shape()[3],
            )
        }
        _ => {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_c, 0, 0],
                got: input.lower().shape().to_vec(),
            });
        }
    };
    let (out_h, out_w) = layer.output_size(input_h, input_w)?;
    let input_elements = checked_shape_product(input.shape()).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ConvTranspose2d finite-deadline IBP input dimensions overflow: {:?}",
            input.shape()
        ))
    })?;
    if input_elements > DEADLINE_CONV_TRANSPOSE_IBP_MAX_OUTPUT_ELEMENTS {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: input_elements.saturating_mul(2 * size_of::<f32>()),
            budget_bytes: DEADLINE_CONV_TRANSPOSE_IBP_MAX_OUTPUT_ELEMENTS * 2 * size_of::<f32>(),
            site: "ConvTranspose2d finite-deadline IBP input scan",
        });
    }
    let output_shape = if batched {
        vec![batch, out_c, out_h, out_w]
    } else {
        vec![out_c, out_h, out_w]
    };
    let output_elements = checked_shape_product(&output_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "ConvTranspose2d finite-deadline IBP output dimensions overflow: {output_shape:?}"
        ))
    })?;
    if output_elements > DEADLINE_CONV_TRANSPOSE_IBP_MAX_OUTPUT_ELEMENTS {
        let bytes_per_element = 2 * size_of::<f64>() + 2 * size_of::<f32>();
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: output_elements.saturating_mul(bytes_per_element),
            budget_bytes: DEADLINE_CONV_TRANSPOSE_IBP_MAX_OUTPUT_ELEMENTS * bytes_per_element,
            site: "ConvTranspose2d finite-deadline IBP output buffers",
        });
    }
    Ok(ConvTranspose2dDeadlineGeometry {
        batched,
        batch,
        in_c,
        out_c,
        input_h,
        input_w,
        out_h,
        out_w,
        output_elements,
        output_shape,
    })
}

fn reserve_convtranspose2d_deadline_vec<T>(
    len: usize,
    deadline: DeadlineInstant,
    name: &str,
) -> Result<Vec<T>> {
    check_convtranspose2d_ibp_deadline(deadline, "before bounded CPU allocation")?;
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NyError::InvalidSpec(format!(
            "ConvTranspose2d finite-deadline IBP {name} allocation failed \
             for {len} elements: {error}"
        ))
    })?;
    check_convtranspose2d_ibp_deadline(deadline, "after bounded CPU allocation")?;
    Ok(values)
}

#[inline]
fn convtranspose2d_f64_to_f32_down(value: f64) -> f32 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value == f64::INFINITY {
        return f32::MAX;
    }
    if value.abs() < f64::from(f32::MIN_POSITIVE) {
        return if value.is_sign_negative() {
            -f32::MIN_POSITIVE
        } else {
            0.0
        };
    }
    ny_tensor::next_down_f32(value as f32)
}

#[inline]
fn convtranspose2d_f64_to_f32_up(value: f64) -> f32 {
    if value.is_nan() || value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == f64::NEG_INFINITY {
        return f32::MIN;
    }
    if value.abs() < f64::from(f32::MIN_POSITIVE) {
        return if value.is_sign_negative() {
            0.0
        } else {
            f32::MIN_POSITIVE
        };
    }
    ny_tensor::next_up_f32(value as f32)
}

fn convtranspose2d_deadline_universal(
    geometry: &ConvTranspose2dDeadlineGeometry,
    deadline: DeadlineInstant,
) -> Result<BoundedTensor> {
    let mut lower = reserve_convtranspose2d_deadline_vec(
        geometry.output_elements,
        deadline,
        "universal lower output",
    )?;
    let mut upper = reserve_convtranspose2d_deadline_vec(
        geometry.output_elements,
        deadline,
        "universal upper output",
    )?;
    while lower.len() < geometry.output_elements {
        let chunk =
            (geometry.output_elements - lower.len()).min(DEADLINE_CONV_TRANSPOSE_IBP_POLL_OPS);
        lower.extend(std::iter::repeat_n(f32::NEG_INFINITY, chunk));
        upper.extend(std::iter::repeat_n(f32::INFINITY, chunk));
        check_convtranspose2d_ibp_deadline(deadline, "while initializing universal output")?;
    }
    let lower =
        ArrayD::from_shape_vec(ndarray::IxDyn(&geometry.output_shape), lower).map_err(|error| {
            NyError::InternalError(format!(
                "ConvTranspose2d finite-deadline IBP universal lower reshape: {error}"
            ))
        })?;
    let upper =
        ArrayD::from_shape_vec(ndarray::IxDyn(&geometry.output_shape), upper).map_err(|error| {
            NyError::InternalError(format!(
                "ConvTranspose2d finite-deadline IBP universal upper reshape: {error}"
            ))
        })?;
    let result =
        BoundedTensor::new_repaired_with_poll(lower, upper, RepairStrategy::Conservative, || {
            check_convtranspose2d_ibp_deadline(deadline, "during universal output repair")
        })?;
    check_convtranspose2d_ibp_deadline(deadline, "immediately before publishing universal output")?;
    Ok(result)
}

/// A 2D transposed convolution layer: y = conv_transpose(x, W) + b
///
/// Input shape: (batch, in_channels, height, width) or (in_channels, height, width)
/// Kernel shape: (in_channels, out_channels, kernel_h, kernel_w) (ONNX ConvTranspose layout)
/// Output shape: (batch, out_channels, out_h, out_w) or (out_channels, out_h, out_w)
impl BoundPropagation for ConvTranspose2dLayer {
    /// IBP for ConvTranspose2d layer: y = conv_transpose(x, W) + b
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let (in_c, _) = self.validate_geometry()?;

        match input.lower().ndim() {
            3 => {
                if input.lower().shape()[0] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![in_c],
                        got: vec![input.lower().shape()[0]],
                    });
                }
                self.propagate_ibp_unbatched(input)
            }
            4 => {
                if input.lower().shape()[1] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![0, in_c, 0, 0],
                        got: input.lower().shape().to_vec(),
                    });
                }
                self.propagate_ibp_batched(input)
            }
            _ => Err(NyError::ShapeMismatch {
                expected: vec![in_c, 0, 0],
                got: input.lower().shape().to_vec(),
            }),
        }
    }

    /// CROWN backward propagation through ConvTranspose2d layer.
    ///
    /// For a transposed conv layer y = conv_transpose(x, W) + b, and current linear bounds A @ y + c:
    /// - The backward pass through conv_transpose is a regular convolution
    /// - new_A = conv(A_reshaped, W)
    /// - new_b = A @ b + c (where b is broadcast across spatial positions)
    ///
    /// Requires `input_shape` to be set for proper shape computation.
    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        self.propagate_linear_with_engine(bounds, None)
    }
}

impl ConvTranspose2dLayer {
    /// IBP propagation with optional GEMM-engine acceleration.
    ///
    /// ConvTranspose2d IBP does not yet have a GEMM-accelerated path; this
    /// delegates to the CPU implementation regardless of engine presence.
    /// Exists for dispatch-site consistency with Conv1d/Conv2d.
    pub fn propagate_ibp_with_engine(
        &self,
        input: &BoundedTensor,
        _engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_ibp(input)
    }

    /// Deadline-authoritative ConvTranspose2d interval forward.
    ///
    /// `deadline: None` preserves [`Self::propagate_ibp_with_engine`] exactly.
    /// A finite authority refuses the opaque caller engine and uses a capped,
    /// directed-f64 CPU scatter with bounded polling quanta. The finite result
    /// is already a certified enclosure, so it is also used by the sound graph
    /// route below.
    pub fn propagate_ibp_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<DeadlineInstant>,
    ) -> Result<BoundedTensor> {
        let Some(deadline) = deadline else {
            return self.propagate_ibp_with_engine(input, engine);
        };
        check_convtranspose2d_ibp_deadline(deadline, "before entry")?;
        self.propagate_ibp_pollable_f64(input, deadline)
    }

    /// SOUND IBP forward (#vnncomp-aw-soundness) — same Higham construction as
    /// `Conv2dLayer::propagate_ibp_sound_with_engine`, for the transposed conv. The plain
    /// forward f32-accumulates each output over at most `K = in_c·kh·kw` scattered products
    /// (groups=1), so under cancellation it can EXCLUDE the true value — unsound as a node
    /// bound. This folds the certified `up(γ_{K+2}·S + 2u·|y|)` outward
    /// (`S = |kernel| transpose-forward on max(|l|,|u|)`).
    pub fn propagate_ibp_sound_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.validate_geometry()?;
        // DAZ can erase a subnormal source operand before multiplication. Its
        // lost contribution is not bounded by the binary32 underflow floor:
        // the other (normal) operand can amplify it up to O(1). The Higham
        // widening below therefore cannot repair this case after the fact.
        // Detect it by bits (not floating-point classification) and fail open
        // to the correctly shaped universal interval.
        if contains_binary32_subnormal(&self.kernel)
            || contains_binary32_subnormal(input.lower())
            || contains_binary32_subnormal(input.upper())
        {
            debug!(
                "ConvTranspose2d certified IBP: subnormal kernel/input endpoint; \
                 returning universal bounds for DAZ independence"
            );
            return self.conservative_ibp_for_daz_source(input);
        }

        let y = self.propagate_ibp_with_engine(input, engine)?;
        let mut xmax = input.lower().mapv(f32::abs);
        ndarray::Zip::from(&mut xmax)
            .and(input.upper())
            .for_each(|m, &u| *m = m.max(u.abs()));
        let abs_kernel = self.kernel.mapv(f32::abs);
        let abs_layer = ConvTranspose2dLayer::new_full(
            abs_kernel,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
        )?;
        let s_bt = abs_layer.propagate_ibp_with_engine(&BoundedTensor::concrete(xmax)?, engine)?;
        // Transpose kernel is (in_c, out_c, kh, kw), groups=1: per-output fan-in <= in_c·kh·kw.
        let macs = self.kernel.shape()[0]
            .saturating_mul(self.kernel.shape()[2])
            .saturating_mul(self.kernel.shape()[3]);
        super::super::crown_helpers::higham_widen_ibp(&y, s_bt.lower(), macs)
    }

    /// Deadline-authoritative certified ConvTranspose2d interval forward.
    ///
    /// The deadline-free branch delegates to the historical Higham path
    /// exactly. The finite branch uses the directed-f64 implementation above,
    /// which is itself a sound enclosure and never enters the unpolled legacy
    /// transpose forward or caller engine.
    pub fn propagate_ibp_sound_with_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<DeadlineInstant>,
    ) -> Result<BoundedTensor> {
        let Some(deadline) = deadline else {
            return self.propagate_ibp_sound_with_engine(input, engine);
        };
        check_convtranspose2d_ibp_deadline(deadline, "before certified propagation")?;
        self.propagate_ibp_pollable_f64(input, deadline)
    }

    fn propagate_ibp_pollable_f64(
        &self,
        input: &BoundedTensor,
        deadline: DeadlineInstant,
    ) -> Result<BoundedTensor> {
        let geometry = convtranspose2d_deadline_geometry(self, input)?;
        check_convtranspose2d_ibp_deadline(deadline, "before directed CPU contraction")?;

        // A DAZ-enabled host can erase subnormal source operands before f64
        // conversion. Unlike rounding residual, that loss is not recoverable
        // after multiplication, so fail open to a correctly shaped universal
        // interval.
        let kernel_has_subnormal = convtranspose2d_deadline_contains_subnormal(
            self.kernel.iter().copied(),
            deadline,
            "while scanning the kernel for subnormal operands",
        )?;
        let bias_has_subnormal = if let Some(bias) = &self.bias {
            convtranspose2d_deadline_contains_subnormal(
                bias.iter().copied(),
                deadline,
                "while scanning bias for subnormal operands",
            )?
        } else {
            false
        };
        let lower_has_subnormal = convtranspose2d_deadline_contains_subnormal(
            input.lower().iter().copied(),
            deadline,
            "while scanning lower input for subnormal operands",
        )?;
        let upper_has_subnormal = convtranspose2d_deadline_contains_subnormal(
            input.upper().iter().copied(),
            deadline,
            "while scanning upper input for subnormal operands",
        )?;
        if kernel_has_subnormal || bias_has_subnormal || lower_has_subnormal || upper_has_subnormal
        {
            debug!(
                "ConvTranspose2d finite-deadline IBP: subnormal source operand; \
                 returning universal bounds for DAZ independence"
            );
            return convtranspose2d_deadline_universal(&geometry, deadline);
        }

        let mut lower_f64 = reserve_convtranspose2d_deadline_vec(
            geometry.output_elements,
            deadline,
            "lower accumulator",
        )?;
        let mut upper_f64 = reserve_convtranspose2d_deadline_vec(
            geometry.output_elements,
            deadline,
            "upper accumulator",
        )?;
        let spatial = geometry.out_h.checked_mul(geometry.out_w).ok_or_else(|| {
            NyError::InvalidSpec(
                "ConvTranspose2d finite-deadline IBP output spatial size overflows".to_string(),
            )
        })?;
        let mut initialized = 0usize;
        for _batch_index in 0..geometry.batch {
            for output_channel in 0..geometry.out_c {
                let bias = self
                    .bias
                    .as_ref()
                    .map_or(0.0_f64, |values| f64::from(values[output_channel]));
                for _ in 0..spatial {
                    lower_f64.push(bias);
                    upper_f64.push(bias);
                    initialized += 1;
                    if initialized.is_multiple_of(DEADLINE_CONV_TRANSPOSE_IBP_POLL_OPS) {
                        check_convtranspose2d_ibp_deadline(
                            deadline,
                            "while initializing directed accumulators",
                        )?;
                    }
                }
            }
        }
        if lower_f64.len() != geometry.output_elements
            || upper_f64.len() != geometry.output_elements
        {
            return Err(NyError::InternalError(format!(
                "ConvTranspose2d finite-deadline IBP initialized {} elements, expected {}",
                lower_f64.len(),
                geometry.output_elements
            )));
        }

        let (stride_h, stride_w) = self.stride;
        let (padding_h, padding_w) = self.padding;
        let (dilation_h, dilation_w) = self.dilation;
        let kernel_h = self.kernel.shape()[2];
        let kernel_w = self.kernel.shape()[3];
        let mut operations = 0usize;
        let mut input_positions = 0usize;

        for batch_index in 0..geometry.batch {
            for input_channel in 0..geometry.in_c {
                for input_y in 0..geometry.input_h {
                    for input_x in 0..geometry.input_w {
                        input_positions += 1;
                        if input_positions.is_multiple_of(DEADLINE_CONV_TRANSPOSE_IBP_POLL_OPS) {
                            check_convtranspose2d_ibp_deadline(
                                deadline,
                                "while traversing input positions",
                            )?;
                        }
                        let input_index = if geometry.batched {
                            DynamicIndex(&[batch_index, input_channel, input_y, input_x])
                        } else {
                            DynamicIndex(&[input_channel, input_y, input_x])
                        };
                        let input_lower = input.lower()[input_index.clone()];
                        let input_upper = input.upper()[input_index];
                        for output_channel in 0..geometry.out_c {
                            for kernel_y in 0..kernel_h {
                                let padded_y = input_y
                                    .checked_mul(stride_h)
                                    .and_then(|base| {
                                        kernel_y
                                            .checked_mul(dilation_h)
                                            .and_then(|offset| base.checked_add(offset))
                                    })
                                    .ok_or_else(|| {
                                        NyError::InvalidSpec(
                                            "ConvTranspose2d finite-deadline IBP \
                                             height coordinate overflows"
                                                .to_string(),
                                        )
                                    })?;
                                let Some(output_y) = padded_y.checked_sub(padding_h) else {
                                    operations = operations.saturating_add(kernel_w);
                                    if operations >= DEADLINE_CONV_TRANSPOSE_IBP_POLL_OPS {
                                        check_convtranspose2d_ibp_deadline(
                                            deadline,
                                            "during directed CPU contraction",
                                        )?;
                                        operations = 0;
                                    }
                                    continue;
                                };
                                if output_y >= geometry.out_h {
                                    operations = operations.saturating_add(kernel_w);
                                    if operations >= DEADLINE_CONV_TRANSPOSE_IBP_POLL_OPS {
                                        check_convtranspose2d_ibp_deadline(
                                            deadline,
                                            "during directed CPU contraction",
                                        )?;
                                        operations = 0;
                                    }
                                    continue;
                                }
                                for kernel_x in 0..kernel_w {
                                    operations += 1;
                                    if operations == DEADLINE_CONV_TRANSPOSE_IBP_POLL_OPS {
                                        check_convtranspose2d_ibp_deadline(
                                            deadline,
                                            "during directed CPU contraction",
                                        )?;
                                        operations = 0;
                                    }
                                    let padded_x = input_x
                                        .checked_mul(stride_w)
                                        .and_then(|base| {
                                            kernel_x
                                                .checked_mul(dilation_w)
                                                .and_then(|offset| base.checked_add(offset))
                                        })
                                        .ok_or_else(|| {
                                            NyError::InvalidSpec(
                                                "ConvTranspose2d finite-deadline IBP \
                                                 width coordinate overflows"
                                                    .to_string(),
                                            )
                                        })?;
                                    let Some(output_x) = padded_x.checked_sub(padding_w) else {
                                        continue;
                                    };
                                    if output_x >= geometry.out_w {
                                        continue;
                                    }

                                    let weight = f64::from(
                                        self.kernel
                                            [[input_channel, output_channel, kernel_y, kernel_x]],
                                    );
                                    let (lower_factor, upper_factor) = if weight >= 0.0 {
                                        (f64::from(input_lower), f64::from(input_upper))
                                    } else {
                                        (f64::from(input_upper), f64::from(input_lower))
                                    };
                                    let output_index = ((batch_index * geometry.out_c
                                        + output_channel)
                                        * geometry.out_h
                                        + output_y)
                                        * geometry.out_w
                                        + output_x;
                                    lower_f64[output_index] = ny_core::dd::next_down_f64(
                                        lower_f64[output_index] + weight * lower_factor,
                                    );
                                    upper_f64[output_index] = ny_core::dd::next_up_f64(
                                        upper_f64[output_index] + weight * upper_factor,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        check_convtranspose2d_ibp_deadline(deadline, "after directed CPU contraction")?;

        let mut lower = reserve_convtranspose2d_deadline_vec(
            geometry.output_elements,
            deadline,
            "published lower output",
        )?;
        let mut upper = reserve_convtranspose2d_deadline_vec(
            geometry.output_elements,
            deadline,
            "published upper output",
        )?;
        for (index, (&lower_value, &upper_value)) in
            lower_f64.iter().zip(upper_f64.iter()).enumerate()
        {
            if index.is_multiple_of(DEADLINE_CONV_TRANSPOSE_IBP_POLL_OPS) {
                check_convtranspose2d_ibp_deadline(deadline, "while publishing directed bounds")?;
            }
            lower.push(convtranspose2d_f64_to_f32_down(lower_value));
            upper.push(convtranspose2d_f64_to_f32_up(upper_value));
        }
        drop(lower_f64);
        drop(upper_f64);

        let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&geometry.output_shape), lower).map_err(
            |error| {
                NyError::InternalError(format!(
                    "ConvTranspose2d finite-deadline IBP lower reshape: {error}"
                ))
            },
        )?;
        let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&geometry.output_shape), upper).map_err(
            |error| {
                NyError::InternalError(format!(
                    "ConvTranspose2d finite-deadline IBP upper reshape: {error}"
                ))
            },
        )?;
        let result = BoundedTensor::new_repaired_with_poll(
            lower,
            upper,
            RepairStrategy::Conservative,
            || check_convtranspose2d_ibp_deadline(deadline, "during result repair"),
        )?;
        check_convtranspose2d_ibp_deadline(deadline, "immediately before publishing result")?;
        Ok(result)
    }

    fn conservative_ibp_for_daz_source(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let (in_c, out_c) = self.validate_geometry()?;
        let (batch, input_h, input_w) = match input.lower().ndim() {
            3 => {
                if input.lower().shape()[0] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![in_c],
                        got: vec![input.lower().shape()[0]],
                    });
                }
                (None, input.lower().shape()[1], input.lower().shape()[2])
            }
            4 => {
                if input.lower().shape()[1] != in_c {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![0, in_c, 0, 0],
                        got: input.lower().shape().to_vec(),
                    });
                }
                (
                    Some(input.lower().shape()[0]),
                    input.lower().shape()[2],
                    input.lower().shape()[3],
                )
            }
            _ => {
                return Err(NyError::ShapeMismatch {
                    expected: vec![in_c, 0, 0],
                    got: input.lower().shape().to_vec(),
                });
            }
        };
        let (out_h, out_w) = self.output_size(input_h, input_w)?;
        let shape = if let Some(batch) = batch {
            vec![batch, out_c, out_h, out_w]
        } else {
            vec![out_c, out_h, out_w]
        };
        let lower = ArrayD::from_elem(ndarray::IxDyn(&shape), f32::NEG_INFINITY);
        let upper = ArrayD::from_elem(ndarray::IxDyn(&shape), f32::INFINITY);
        BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
    }

    fn propagate_ibp_unbatched(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let kernel_pos = self.kernel.mapv(nan_propagating_max_zero);
        let kernel_neg = self.kernel.mapv(nan_propagating_min_zero);

        let lower_from_pos = conv2d_transpose_forward(
            input.lower(),
            &kernel_pos,
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
        )?;
        let lower_from_neg = conv2d_transpose_forward(
            input.upper(),
            &kernel_neg,
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
        )?;
        let mut lower_y = lower_from_pos + lower_from_neg;

        let upper_from_pos = conv2d_transpose_forward(
            input.upper(),
            &kernel_pos,
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
        )?;
        let upper_from_neg = conv2d_transpose_forward(
            input.lower(),
            &kernel_neg,
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
        )?;
        let mut upper_y = upper_from_pos + upper_from_neg;

        if let Some(ref bias) = self.bias {
            let out_c = self.out_channels();
            let out_h = lower_y.shape()[1];
            let out_w = lower_y.shape()[2];
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        lower_y[[oc, oh, ow]] += bias[oc];
                        upper_y[[oc, oh, ow]] += bias[oc];
                    }
                }
            }
        }

        BoundedTensor::new_repaired(lower_y, upper_y, RepairStrategy::Conservative)
    }

    fn propagate_ibp_batched(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let kernel_pos = self.kernel.mapv(nan_propagating_max_zero);
        let kernel_neg = self.kernel.mapv(nan_propagating_min_zero);

        let batch = input.lower().shape()[0];
        let input_h = input.lower().shape()[2];
        let input_w = input.lower().shape()[3];
        let (out_h, out_w) = self.output_size(input_h, input_w)?;
        let out_c = self.out_channels();

        let mut lower_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_h, out_w]));
        let mut upper_y = ArrayD::zeros(ndarray::IxDyn(&[batch, out_c, out_h, out_w]));

        for batch_idx in 0..batch {
            let lower_b = input
                .lower()
                .index_axis(ndarray::Axis(0), batch_idx)
                .to_owned()
                .into_dyn();
            let upper_b = input
                .upper()
                .index_axis(ndarray::Axis(0), batch_idx)
                .to_owned()
                .into_dyn();

            let lower_from_pos = conv2d_transpose_forward(
                &lower_b,
                &kernel_pos,
                self.stride,
                self.padding,
                self.dilation,
                self.output_padding,
            )?;
            let lower_from_neg = conv2d_transpose_forward(
                &upper_b,
                &kernel_neg,
                self.stride,
                self.padding,
                self.dilation,
                self.output_padding,
            )?;
            let lower_batch = lower_from_pos + lower_from_neg;

            let upper_from_pos = conv2d_transpose_forward(
                &upper_b,
                &kernel_pos,
                self.stride,
                self.padding,
                self.dilation,
                self.output_padding,
            )?;
            let upper_from_neg = conv2d_transpose_forward(
                &lower_b,
                &kernel_neg,
                self.stride,
                self.padding,
                self.dilation,
                self.output_padding,
            )?;
            let upper_batch = upper_from_pos + upper_from_neg;

            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        lower_y[[batch_idx, oc, oh, ow]] = lower_batch[[oc, oh, ow]];
                        upper_y[[batch_idx, oc, oh, ow]] = upper_batch[[oc, oh, ow]];
                    }
                }
            }
        }

        if let Some(ref bias) = self.bias {
            for batch_idx in 0..batch {
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            lower_y[[batch_idx, oc, oh, ow]] += bias[oc];
                            upper_y[[batch_idx, oc, oh, ow]] += bias[oc];
                        }
                    }
                }
            }
        }

        BoundedTensor::new_repaired(lower_y, upper_y, RepairStrategy::Conservative)
    }

    /// CROWN backward through ConvTranspose2d with optional GemmEngine.
    ///
    /// Uses batched im2col+GEMM via `conv2d_forward_batched_gemm` for GPU
    /// acceleration (#3598). Falls back to CPU faer GEMM when engine is None.
    pub fn propagate_linear_with_engine<'a>(
        &self,
        bounds: &'a LinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Cow<'a, LinearBounds>> {
        self.propagate_linear_with_engine_and_deadline(bounds, engine, None)
    }

    /// Deadline-bearing CROWN backward through ConvTranspose2d
    /// (#wall-deadwork ConvTranspose port). `deadline: None` is byte-identical
    /// to [`Self::propagate_linear_with_engine`]. With a deadline, the skip
    /// path checks it up front and the dominant f64 recompute polls it between
    /// objective blocks; expiry surfaces as `DeadlineExceeded`, which the
    /// graph collector already maps to its sound reference-bounds fallback.
    pub fn propagate_linear_with_engine_and_deadline<'a>(
        &self,
        bounds: &'a LinearBounds,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<Cow<'a, LinearBounds>> {
        let input_shape = self.input_shape.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "ConvTranspose2d CROWN requires input_shape to be set. Use with_input_shape() or set_input_shape()."
                    .to_string(),
            )
        })?;
        self.propagate_linear_with_engine_and_deadline_for_input_shape(
            bounds,
            engine,
            deadline,
            input_shape,
        )
    }

    /// Borrowing variant for dispatchers that already authenticate the
    /// current pre-activation shape. Avoids an O(kernel+bias) layer clone just
    /// to update spatial metadata under finite authority.
    pub(crate) fn propagate_linear_with_engine_and_deadline_for_input_shape<'a>(
        &self,
        bounds: &'a LinearBounds,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
        (in_h, in_w): (usize, usize),
    ) -> Result<Cow<'a, LinearBounds>> {
        debug!("ConvTranspose2d layer CROWN backward propagation");
        let (in_c, out_c) = self.validate_geometry()?;

        guard_nan_weights(&self.kernel, self.bias.as_ref(), "ConvTranspose2d")?;

        let (out_h, out_w) = self.output_size(in_h, in_w)?;

        let expected_conv_out = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d CROWN: output dims product overflows: {out_c} * {out_h} * {out_w}"
            ))
        })?;
        if bounds.num_inputs() != expected_conv_out {
            return Err(NyError::ShapeMismatch {
                expected: vec![expected_conv_out],
                got: vec![bounds.num_inputs()],
            });
        }

        let conv_in_size = checked_shape_product(&[in_c, in_h, in_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ConvTranspose2d CROWN: input dims product overflows: {in_c} * {in_h} * {in_w}"
            ))
        })?;

        // SOUND coefficient (#vnncomp-aw-soundness — conv f32-accumulation bug):
        // on wide contractions re-accumulate the SAME forward-conv contraction in
        // f64, store the directed f32, and certify `cast_err + γ_n^f64·S`; on small
        // contractions keep the f32 GEMM coefficient and certify `γ_n^f32·S` (both
        // sound). See conv2d/bound.rs for the full rationale. The kernel is ONNX
        // ConvTranspose layout `(in_c, out_c/groups, kh, kw)`; the backward forward
        // conv contracts over `kernel[1]·kh·kw`, so n = in_c_per_group·kh·kw.
        let (kh, kw) = self.kernel_size();
        let in_c_per_group = self.kernel.shape()[1];
        let n_contraction = in_c_per_group.saturating_mul(kh).saturating_mul(kw);
        let want_recompute = super::super::crown_helpers::conv_should_f64_recompute(n_contraction);
        // #wall-deadwork port (ConvTranspose; default-on, `NY_CONV_SKIP_DEAD_F32=0`
        // kill-switch): under `want_recompute` the f32 pair's A-values are
        // discarded on BOTH paths below (success → overwritten with the rounded
        // f64 recompute; failure → row degraded to ±inf bias and the err matrix
        // zeroed), so the pair contributes only the buffers. Skip it, allocating
        // directly; the geometry fixes the pair's output shape at
        // `(nrows, conv_in_size)` (validated by the bitwise-identity oracle).
        // Unlike conv2d/bound.rs there is no memory guard here because the pair
        // (`conv2d_forward_batched_gemm`) never enforced one — adding a new
        // refusal would not be value-identical.
        let skip_dead_f32 =
            want_recompute && super::super::crown_helpers::conv_skip_dead_f32_enabled();
        let (mut new_lower_a, mut new_upper_a) = if skip_dead_f32 {
            if deadline.is_some_and(|dl| std::time::Instant::now() >= dl) {
                return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
            }
            (
                ndarray::Array2::<f32>::zeros((bounds.lower_a().nrows(), conv_in_size)),
                ndarray::Array2::<f32>::zeros((bounds.upper_a().nrows(), conv_in_size)),
            )
        } else {
            (
                super::conv2d_forward_batched_gemm_with_deadline(
                    bounds.lower_a(),
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (out_h, out_w),
                    engine,
                    deadline,
                )?,
                super::conv2d_forward_batched_gemm_with_deadline(
                    bounds.upper_a(),
                    &self.kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (out_h, out_w),
                    engine,
                    deadline,
                )?,
            )
        };

        // Deadline expiry inside the f64 recompute propagates as
        // `DeadlineExceeded` (collector reference-bounds fallback) instead of
        // degrading the side's rows to ±inf bias; every other recompute error
        // keeps the shipped failed-recompute degrade path. With
        // `deadline: None` (all pre-existing callers) this is byte-identical
        // to the old `.ok()` handling — no deadline error can occur.
        let recompute_side = |a: &ndarray::Array2<f32>| -> Result<Option<ndarray::Array2<f64>>> {
            match super::conv2d_forward_backward_coeff_f64_with_deadline(
                a,
                &self.kernel,
                self.stride,
                self.padding,
                self.dilation,
                (out_h, out_w),
                deadline,
            ) {
                Ok(c) => Ok(Some(c)),
                Err(e @ NyError::DeadlineExceeded(_)) => Err(e),
                Err(_) => Ok(None),
            }
        };
        let recompute_pair = || -> Result<Option<(ndarray::Array2<f64>, ndarray::Array2<f64>)>> {
            match super::conv2d_forward_backward_coeff_f64_pair_with_deadline(
                bounds.lower_a(),
                bounds.upper_a(),
                &self.kernel,
                self.stride,
                self.padding,
                self.dilation,
                (out_h, out_w),
                deadline,
            ) {
                Ok(pair) => Ok(pair),
                Err(e @ NyError::DeadlineExceeded(_)) => Err(e),
                // The pair route is an optimization only. Any other error
                // falls through to the exact prior per-side recomputes.
                Err(_) => Ok(None),
            }
        };
        // Under the skip-dead-f32 recompute gate the two independent f64
        // recomputes run concurrently. Independently, exact opt-in
        // `NY_CONVTRANSPOSE_SOUND_F64_GPU=1` lets large internal GEMM blocks use
        // a direct-row-major lower/upper pair through the process-global
        // sound-f64 engine (not the caller's possibly f32-only `engine`).
        // Unset/malformed values, unavailable engines, rejected geometry,
        // allocation failure, or malformed pair output fall through to the
        // exact prior rayon/faer route below. The IEEE-f64 reduction order is
        // covered by the same summation-order-independent certified error
        // channel. Turning off the skip-dead-f32 recompute gate keeps the
        // shipped serial order untouched.
        let (coeff_f64, coeff_f64_u) = if !want_recompute {
            (None, None)
        } else if skip_dead_f32 {
            if let Some((lower, upper)) = recompute_pair()? {
                (Some(lower), Some(upper))
            } else {
                let (l, u) = rayon::join(
                    || recompute_side(bounds.lower_a()),
                    || recompute_side(bounds.upper_a()),
                );
                (l?, u?)
            }
        } else {
            (
                recompute_side(bounds.lower_a())?,
                recompute_side(bounds.upper_a())?,
            )
        };
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
            compute_conv_bias_f64(bounds, self.bias.as_ref(), out_c, out_h * out_w)?;

        // Certified coefficient error `cast + γ·S + prop` (shared helper).
        //
        // #cgan-conv-err-compose: when incoming certified error is present,
        // compose it EXACTLY through the same backward transform with |kernel|
        // (`prop[i,p] = Σ_j err_in[i,j]·|K_{j→p}|`, one extra f32 GEMM on
        // non-negative data) instead of the row-constant
        // `row_max(err_in)·‖kernel‖_1` over-bound, which amplified the carried
        // error by ~‖kernel‖_1/column-L1 (≥100×) per conv layer and collapsed
        // cGAN's per-target CROWN to near-IBP after the discharge at the next
        // BatchNorm. The exact composition is the first-order enclosure
        // |Σ_j (a±e)_j·K − Σ_j a_j·K| ≤ Σ_j e_j·|K|; its own f32 rounding is
        // covered inside `conv_coeff_err_matrix` by the (1+γ) inflation.
        let abs_kernel = (bounds.lower_a_err().is_some() || bounds.upper_a_err().is_some())
            .then(|| self.kernel.mapv(f32::abs));
        let compose_err =
            |err_in: Option<&ndarray::Array2<f32>>| -> Result<Option<ndarray::Array2<f32>>> {
                let Some(err_in) = err_in else {
                    return Ok(None);
                };
                let Some(abs_kernel) = abs_kernel.as_ref() else {
                    return Ok(None);
                };
                match super::conv2d_forward_batched_gemm_with_deadline(
                    err_in,
                    abs_kernel,
                    self.stride,
                    self.padding,
                    self.dilation,
                    (out_h, out_w),
                    engine,
                    deadline,
                ) {
                    Ok(composed) => Ok(Some(composed)),
                    Err(error @ NyError::DeadlineExceeded(_)) => Err(error),
                    Err(_) => Ok(None),
                }
            };
        let prop_l = compose_err(bounds.lower_a_err())?;
        let prop_u = compose_err(bounds.upper_a_err())?;
        let kernel_l1: f64 = self.kernel.iter().map(|&v| (v as f64).abs()).sum();
        let mut lower_err = super::super::crown_helpers::conv_coeff_err_matrix(
            bounds.lower_a(),
            bounds.lower_a_err(),
            &new_lower_a,
            coeff_f64.as_ref().filter(|_| lower_recompute_ok),
            kernel_l1,
            n_contraction,
            prop_l.as_ref(),
            None,
        );
        let mut upper_err = super::super::crown_helpers::conv_coeff_err_matrix(
            bounds.upper_a(),
            bounds.upper_a_err(),
            &new_upper_a,
            coeff_f64_u.as_ref().filter(|_| upper_recompute_ok),
            kernel_l1,
            n_contraction,
            prop_u.as_ref(),
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
            "ConvTranspose2d",
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

        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
            return Err(super::ops_transpose_gemm::per_node_deadline_exceeded());
        }
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
