// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Deadline-bounded true joint-α adjoint for the CUDA backend.
//!
//! The structural forward/reverse walk is host orchestrated, while every
//! affine contraction is issued through the caller's `GemmEngine`. Production
//! passes `DeadlineCrownGemm`, whose `gemm_f64` is the ATS-only, tiled cuBLAS
//! primitive. Thus this is a real CUDA path (linear and convolution work reaches
//! cuBLAS) without falsely advertising the backend-global CROWN deadline flag.

use ny_core::{GemmEngine, GpuCrownLayer, GpuResnetSegment, NyError, Result};
use std::mem::size_of;

const HOST_POLL_STRIDE: usize = 4096;

struct ReluRecord {
    a_pre: Vec<f64>,
    sigma: Vec<f64>,
    tau: Vec<f64>,
    width: usize,
}

#[inline]
fn poll<E: GemmEngine + ?Sized>(engine: &E) -> Result<()> {
    engine.poll_crown_backward_deadline()
}

#[inline]
fn poll_index<E: GemmEngine + ?Sized>(engine: &E, index: usize) -> Result<()> {
    if index.is_multiple_of(HOST_POLL_STRIDE) {
        poll(engine)?;
    }
    Ok(())
}

/// Account for one constant-size innermost host work unit.
///
/// `completed` is monotonic for the complete surrounding loop. Polling before
/// units 0, 4096, ... ensures no more than 4096 elementary operations can run
/// between cooperative deadline checks, irrespective of tensor geometry.
#[inline]
fn poll_host_work<E: GemmEngine + ?Sized>(engine: &E, completed: &mut usize) -> Result<()> {
    if completed.is_multiple_of(HOST_POLL_STRIDE) {
        poll(engine)?;
    }
    *completed = completed.checked_add(1).ok_or_else(|| {
        NyError::InvalidSpec("cuda joint alpha: host work counter overflow".into())
    })?;
    Ok(())
}

fn reserve_vec<T, E: GemmEngine + ?Sized>(
    engine: &E,
    len: usize,
    site: &'static str,
) -> Result<Vec<T>> {
    poll(engine)?;
    let required_bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
        NyError::InvalidSpec("cuda joint alpha: host allocation bytes overflow".into())
    })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes: usize::MAX,
            site,
        })?;
    poll(engine)?;
    Ok(values)
}

/// Allocate and initialize f64 storage in poll-bounded chunks.
fn zeroed_f64<E: GemmEngine + ?Sized>(
    engine: &E,
    len: usize,
    site: &'static str,
) -> Result<Vec<f64>> {
    let mut values = reserve_vec(engine, len, site)?;
    while values.len() < len {
        poll(engine)?;
        let end = values.len().saturating_add(HOST_POLL_STRIDE).min(len);
        values.resize(end, 0.0);
    }
    poll(engine)?;
    Ok(values)
}

fn clone_f64<E: GemmEngine + ?Sized>(
    engine: &E,
    values: &[f64],
    site: &'static str,
) -> Result<Vec<f64>> {
    let mut out = reserve_vec(engine, values.len(), site)?;
    for (index, &value) in values.iter().enumerate() {
        poll_index(engine, index)?;
        out.push(value);
    }
    poll(engine)?;
    Ok(out)
}

fn product(parts: &[usize], label: &str) -> Result<usize> {
    parts.iter().try_fold(1usize, |value, &part| {
        value
            .checked_mul(part)
            .ok_or_else(|| NyError::InvalidSpec(format!("cuda joint alpha: {label} overflow")))
    })
}

fn shape(expected: usize, got: usize) -> NyError {
    NyError::shape_mismatch(vec![expected], vec![got])
}

fn add_vectors<E: GemmEngine + ?Sized>(
    engine: &E,
    left: &[f64],
    right: &[f64],
) -> Result<Vec<f64>> {
    if left.len() != right.len() {
        return Err(shape(left.len(), right.len()));
    }
    let mut out = reserve_vec(engine, left.len(), "cuda::joint_alpha/add_vectors")?;
    for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
        poll_index(engine, index)?;
        out.push(a + b);
    }
    poll(engine)?;
    Ok(out)
}

fn widen_f32<E: GemmEngine + ?Sized>(engine: &E, values: &[f32]) -> Result<Vec<f64>> {
    let mut out = reserve_vec(engine, values.len(), "cuda::joint_alpha/widen_f32")?;
    for (index, &value) in values.iter().enumerate() {
        poll_index(engine, index)?;
        out.push(f64::from(value));
    }
    poll(engine)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn crown_joint_alpha_gradient_with_deadline_impl<E: GemmEngine + ?Sized>(
    engine: &E,
    segments: &[GpuResnetSegment],
    seed_lower_a: &[f32],
    num_specs: usize,
    output_dim: usize,
    input_lower: &[f32],
    input_upper: &[f32],
) -> Result<Vec<Vec<f32>>> {
    poll(engine)?;
    if num_specs == 0 || output_dim == 0 {
        return Err(NyError::InvalidSpec(
            "cuda joint alpha: empty specification/output".into(),
        ));
    }
    let seed_len = product(&[num_specs, output_dim], "seed size")?;
    if seed_lower_a.len() != seed_len {
        return Err(shape(seed_len, seed_lower_a.len()));
    }
    if input_lower.is_empty() || input_lower.len() != input_upper.len() {
        return Err(shape(input_lower.len(), input_upper.len()));
    }
    if segments.is_empty() {
        return Err(NyError::InvalidSpec(
            "cuda joint alpha: empty segment list".into(),
        ));
    }

    let mut relus = Vec::new();
    let mut coefficients = widen_f32(engine, seed_lower_a)?;
    let mut dim = output_dim;
    for segment in segments {
        poll(engine)?;
        (coefficients, dim) =
            forward_segment(engine, segment, coefficients, num_specs, dim, &mut relus)?;
        poll(engine)?;
    }
    let input_dim = input_lower.len();
    if dim != input_dim {
        return Err(shape(input_dim, dim));
    }

    let adjoint_len = product(&[num_specs, input_dim], "adjoint seed")?;
    let mut adjoint = reserve_vec(engine, adjoint_len, "cuda::joint_alpha/adjoint_seed")?;
    for index in 0..adjoint_len {
        poll_index(engine, index)?;
        let input_index = index % input_dim;
        adjoint.push(if coefficients[index] >= 0.0 {
            f64::from(input_lower[input_index])
        } else {
            f64::from(input_upper[input_index])
        });
    }
    poll(engine)?;

    let mut gradients = reserve_vec(engine, relus.len(), "cuda::joint_alpha/gradient_rows")?;
    for record in &relus {
        gradients.push(zeroed_f64(
            engine,
            record.width,
            "cuda::joint_alpha/gradient_row",
        )?);
    }
    let mut cursor = relus.len();
    let (_adjoint_out, _output_width) = adjoint_segments(
        engine,
        segments,
        adjoint,
        num_specs,
        input_dim,
        &relus,
        &mut cursor,
        &mut gradients,
    )?;
    if cursor != 0 {
        return Err(NyError::InvalidSpec(format!(
            "cuda joint alpha: {cursor} ReLU records remained after adjoint"
        )));
    }

    let mut result = reserve_vec(
        engine,
        gradients.len(),
        "cuda::joint_alpha/f32_gradient_rows",
    )?;
    for gradient in gradients {
        poll(engine)?;
        let mut row = reserve_vec(engine, gradient.len(), "cuda::joint_alpha/f32_gradient_row")?;
        for (index, value) in gradient.into_iter().enumerate() {
            poll_index(engine, index)?;
            let value = value as f32;
            if !value.is_finite() {
                return Err(NyError::NumericalInstability(
                    "cuda joint alpha: non-finite f32 gradient".into(),
                ));
            }
            row.push(value);
        }
        result.push(row);
    }
    poll(engine)?;
    Ok(result)
}

fn forward_segment<E: GemmEngine + ?Sized>(
    engine: &E,
    segment: &GpuResnetSegment,
    coefficients: Vec<f64>,
    num_specs: usize,
    dim: usize,
    relus: &mut Vec<ReluRecord>,
) -> Result<(Vec<f64>, usize)> {
    poll(engine)?;
    match segment {
        GpuResnetSegment::Chain(layers) => {
            forward_chain(engine, layers, coefficients, num_specs, dim, relus)
        }
        GpuResnetSegment::Residual(branch) => {
            let skip = clone_f64(
                engine,
                &coefficients,
                "cuda::joint_alpha/forward_residual_skip",
            )?;
            let (branch_coefficients, branch_dim) =
                forward_chain(engine, branch, coefficients, num_specs, dim, relus)?;
            if branch_dim != dim {
                return Err(shape(dim, branch_dim));
            }
            Ok((
                add_vectors(engine, &skip, &branch_coefficients)?,
                branch_dim,
            ))
        }
        GpuResnetSegment::ResidualProj(main, projection) => {
            let projection_input = clone_f64(
                engine,
                &coefficients,
                "cuda::joint_alpha/forward_projection_input",
            )?;
            let (main_coefficients, main_dim) =
                forward_chain(engine, main, coefficients, num_specs, dim, relus)?;
            let (projection_coefficients, projection_dim) =
                forward_chain(engine, projection, projection_input, num_specs, dim, relus)?;
            if main_dim != projection_dim {
                return Err(shape(main_dim, projection_dim));
            }
            Ok((
                add_vectors(engine, &main_coefficients, &projection_coefficients)?,
                main_dim,
            ))
        }
    }
}

fn forward_chain<E: GemmEngine + ?Sized>(
    engine: &E,
    layers: &[GpuCrownLayer],
    mut coefficients: Vec<f64>,
    num_specs: usize,
    mut dim: usize,
    relus: &mut Vec<ReluRecord>,
) -> Result<(Vec<f64>, usize)> {
    for layer in layers {
        poll(engine)?;
        (coefficients, dim) = forward_layer(engine, layer, coefficients, num_specs, dim, relus)?;
        poll(engine)?;
    }
    Ok((coefficients, dim))
}

fn forward_layer<E: GemmEngine + ?Sized>(
    engine: &E,
    layer: &GpuCrownLayer,
    coefficients: Vec<f64>,
    num_specs: usize,
    dim: usize,
    relus: &mut Vec<ReluRecord>,
) -> Result<(Vec<f64>, usize)> {
    poll(engine)?;
    match layer {
        GpuCrownLayer::Linear {
            weight,
            out_features,
            in_features,
            ..
        } => {
            if *out_features != dim {
                return Err(shape(*out_features, dim));
            }
            let weight_len = product(&[*out_features, *in_features], "linear weight")?;
            if weight.len() != weight_len {
                return Err(shape(weight_len, weight.len()));
            }
            let weight = widen_f32(engine, weight)?;
            let result = engine.gemm_f64(
                num_specs,
                *out_features,
                *in_features,
                &coefficients,
                &weight,
            )?;
            poll(engine)?;
            Ok((result, *in_features))
        }
        GpuCrownLayer::Conv2d {
            weight_col,
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            in_h,
            in_w,
            ..
        } => {
            let geometry = ConvGeometry {
                out_channels: *out_channels,
                in_channels: *in_channels,
                kernel_h: *kernel_h,
                kernel_w: *kernel_w,
                stride_h: *stride_h,
                stride_w: *stride_w,
                pad_h: *pad_h,
                pad_w: *pad_w,
                out_h: *out_h,
                out_w: *out_w,
                in_h: *in_h,
                in_w: *in_w,
            };
            geometry.validate(dim, weight_col.len())?;
            let result = forward_conv(engine, &coefficients, num_specs, weight_col, geometry)?;
            Ok((result, geometry.input_dim()?))
        }
        GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            num_neurons,
        } => {
            let width = *num_neurons;
            if width != dim
                || lower_slope.len() != width
                || upper_slope.len() != width
                || lower_intercept.len() != width
                || upper_intercept.len() != width
            {
                return Err(shape(width, dim));
            }
            let expected = product(&[num_specs, width], "ReLU coefficient")?;
            if coefficients.len() != expected {
                return Err(shape(expected, coefficients.len()));
            }
            let mut transformed =
                reserve_vec(engine, expected, "cuda::joint_alpha/relu_transformed")?;
            let mut sigma = reserve_vec(engine, expected, "cuda::joint_alpha/relu_sigma")?;
            let mut tau = reserve_vec(engine, expected, "cuda::joint_alpha/relu_tau")?;
            for (index, &coefficient) in coefficients.iter().enumerate() {
                poll_index(engine, index)?;
                let neuron = index % width;
                let (selected_slope, selected_intercept) = if coefficient >= 0.0 {
                    (lower_slope[neuron], lower_intercept[neuron])
                } else {
                    (upper_slope[neuron], upper_intercept[neuron])
                };
                let selected_slope = f64::from(selected_slope);
                transformed.push(coefficient * selected_slope);
                sigma.push(selected_slope);
                tau.push(f64::from(selected_intercept));
            }
            relus.push(ReluRecord {
                a_pre: coefficients,
                sigma,
                tau,
                width,
            });
            poll(engine)?;
            Ok((transformed, width))
        }
        GpuCrownLayer::ActivationReluDualAlpha { .. } | GpuCrownLayer::MaxPool2d { .. } => Err(
            NyError::UnsupportedOp("cuda joint alpha: dual-alpha/maxpool is unsupported".into()),
        ),
    }
}

#[derive(Clone, Copy)]
struct ConvGeometry {
    out_channels: usize,
    in_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    out_h: usize,
    out_w: usize,
    in_h: usize,
    in_w: usize,
}

impl ConvGeometry {
    fn output_spatial(self) -> Result<usize> {
        product(&[self.out_h, self.out_w], "conv output spatial")
    }

    fn output_dim(self) -> Result<usize> {
        product(
            &[self.out_channels, self.out_h, self.out_w],
            "conv output dimension",
        )
    }

    fn input_dim(self) -> Result<usize> {
        product(
            &[self.in_channels, self.in_h, self.in_w],
            "conv input dimension",
        )
    }

    fn kernel_columns(self) -> Result<usize> {
        product(
            &[self.in_channels, self.kernel_h, self.kernel_w],
            "conv kernel columns",
        )
    }

    fn validate(self, dim: usize, weight_len: usize) -> Result<()> {
        if self.stride_h == 0 || self.stride_w == 0 {
            return Err(NyError::InvalidSpec(
                "cuda joint alpha: zero convolution stride".into(),
            ));
        }
        if [
            self.out_channels,
            self.in_channels,
            self.kernel_h,
            self.kernel_w,
            self.out_h,
            self.out_w,
            self.in_h,
            self.in_w,
        ]
        .contains(&0)
        {
            return Err(NyError::InvalidSpec(
                "cuda joint alpha: zero convolution geometry".into(),
            ));
        }
        let output_dim = self.output_dim()?;
        if dim != output_dim {
            return Err(shape(output_dim, dim));
        }
        let expected_weight = product(
            &[
                self.out_channels,
                self.in_channels,
                self.kernel_h,
                self.kernel_w,
            ],
            "conv weight",
        )?;
        if weight_len != expected_weight {
            return Err(shape(expected_weight, weight_len));
        }
        Ok(())
    }
}

fn forward_conv<E: GemmEngine + ?Sized>(
    engine: &E,
    coefficients: &[f64],
    num_specs: usize,
    weight_col: &[f32],
    geometry: ConvGeometry,
) -> Result<Vec<f64>> {
    poll(engine)?;
    let output_spatial = geometry.output_spatial()?;
    let output_dim = geometry.output_dim()?;
    let input_dim = geometry.input_dim()?;
    let kernel_columns = geometry.kernel_columns()?;
    let expected = product(&[num_specs, output_dim], "conv forward input")?;
    if coefficients.len() != expected {
        return Err(shape(expected, coefficients.len()));
    }

    let rows = product(&[num_specs, output_spatial], "conv forward rows")?;
    let reshaped_len = product(&[rows, geometry.out_channels], "conv reshape")?;
    let mut reshaped = reserve_vec(
        engine,
        reshaped_len,
        "cuda::joint_alpha/conv_forward_reshape",
    )?;
    let mut reshape_work = 0usize;
    for spec in 0..num_specs {
        for spatial in 0..output_spatial {
            for channel in 0..geometry.out_channels {
                poll_host_work(engine, &mut reshape_work)?;
                reshaped.push(coefficients[spec * output_dim + channel * output_spatial + spatial]);
            }
        }
    }
    poll(engine)?;
    let weight = widen_f32(engine, weight_col)?;
    let columns = engine.gemm_f64(
        rows,
        geometry.out_channels,
        kernel_columns,
        &reshaped,
        &weight,
    )?;
    poll(engine)?;

    let mut result = zeroed_f64(
        engine,
        product(&[num_specs, input_dim], "conv result")?,
        "cuda::joint_alpha/conv_forward_result",
    )?;
    let mut scatter_work = 0usize;
    for spec in 0..num_specs {
        for output_y in 0..geometry.out_h {
            for output_x in 0..geometry.out_w {
                let row =
                    (spec * output_spatial + output_y * geometry.out_w + output_x) * kernel_columns;
                for input_channel in 0..geometry.in_channels {
                    for kernel_y in 0..geometry.kernel_h {
                        for kernel_x in 0..geometry.kernel_w {
                            poll_host_work(engine, &mut scatter_work)?;
                            let padded_y = output_y
                                .checked_mul(geometry.stride_h)
                                .and_then(|value| value.checked_add(kernel_y))
                                .ok_or_else(|| {
                                    NyError::InvalidSpec(
                                        "cuda joint alpha: conv input-y overflow".into(),
                                    )
                                })?;
                            if padded_y < geometry.pad_h {
                                continue;
                            }
                            let input_y = padded_y - geometry.pad_h;
                            if input_y >= geometry.in_h {
                                continue;
                            }
                            let padded_x = output_x
                                .checked_mul(geometry.stride_w)
                                .and_then(|value| value.checked_add(kernel_x))
                                .ok_or_else(|| {
                                    NyError::InvalidSpec(
                                        "cuda joint alpha: conv input-x overflow".into(),
                                    )
                                })?;
                            let input_x = padded_x;
                            if input_x < geometry.pad_w {
                                continue;
                            }
                            let input_x = input_x - geometry.pad_w;
                            if input_x >= geometry.in_w {
                                continue;
                            }
                            let column = input_channel * geometry.kernel_h * geometry.kernel_w
                                + kernel_y * geometry.kernel_w
                                + kernel_x;
                            let destination = spec * input_dim
                                + (input_channel * geometry.in_h + input_y) * geometry.in_w
                                + input_x;
                            result[destination] += columns[row + column];
                        }
                    }
                }
            }
        }
    }
    poll(engine)?;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn adjoint_segments<E: GemmEngine + ?Sized>(
    engine: &E,
    segments: &[GpuResnetSegment],
    mut adjoint: Vec<f64>,
    num_specs: usize,
    mut dim: usize,
    relus: &[ReluRecord],
    cursor: &mut usize,
    gradients: &mut [Vec<f64>],
) -> Result<(Vec<f64>, usize)> {
    for segment in segments.iter().rev() {
        poll(engine)?;
        (adjoint, dim) = adjoint_segment(
            engine, segment, adjoint, num_specs, dim, relus, cursor, gradients,
        )?;
        poll(engine)?;
    }
    Ok((adjoint, dim))
}

#[allow(clippy::too_many_arguments)]
fn adjoint_segment<E: GemmEngine + ?Sized>(
    engine: &E,
    segment: &GpuResnetSegment,
    adjoint: Vec<f64>,
    num_specs: usize,
    dim: usize,
    relus: &[ReluRecord],
    cursor: &mut usize,
    gradients: &mut [Vec<f64>],
) -> Result<(Vec<f64>, usize)> {
    poll(engine)?;
    match segment {
        GpuResnetSegment::Chain(layers) => adjoint_chain(
            engine, layers, adjoint, num_specs, dim, relus, cursor, gradients,
        ),
        GpuResnetSegment::Residual(branch) => {
            let skip = clone_f64(engine, &adjoint, "cuda::joint_alpha/adjoint_residual_skip")?;
            let (branch_adjoint, branch_dim) = adjoint_chain(
                engine, branch, adjoint, num_specs, dim, relus, cursor, gradients,
            )?;
            if branch_dim != dim {
                return Err(shape(dim, branch_dim));
            }
            Ok((add_vectors(engine, &skip, &branch_adjoint)?, dim))
        }
        GpuResnetSegment::ResidualProj(main, projection) => {
            // Reverse of the forward F-then-P record order: consume P first.
            let projection_input = clone_f64(
                engine,
                &adjoint,
                "cuda::joint_alpha/adjoint_projection_input",
            )?;
            let (projection_adjoint, projection_dim) = adjoint_chain(
                engine,
                projection,
                projection_input,
                num_specs,
                dim,
                relus,
                cursor,
                gradients,
            )?;
            let (main_adjoint, main_dim) = adjoint_chain(
                engine, main, adjoint, num_specs, dim, relus, cursor, gradients,
            )?;
            if main_dim != projection_dim {
                return Err(shape(main_dim, projection_dim));
            }
            Ok((
                add_vectors(engine, &main_adjoint, &projection_adjoint)?,
                main_dim,
            ))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn adjoint_chain<E: GemmEngine + ?Sized>(
    engine: &E,
    layers: &[GpuCrownLayer],
    mut adjoint: Vec<f64>,
    num_specs: usize,
    mut dim: usize,
    relus: &[ReluRecord],
    cursor: &mut usize,
    gradients: &mut [Vec<f64>],
) -> Result<(Vec<f64>, usize)> {
    for layer in layers.iter().rev() {
        poll(engine)?;
        (adjoint, dim) = adjoint_layer(
            engine, layer, adjoint, num_specs, dim, relus, cursor, gradients,
        )?;
        poll(engine)?;
    }
    Ok((adjoint, dim))
}

#[allow(clippy::too_many_arguments)]
fn adjoint_layer<E: GemmEngine + ?Sized>(
    engine: &E,
    layer: &GpuCrownLayer,
    adjoint: Vec<f64>,
    num_specs: usize,
    dim: usize,
    relus: &[ReluRecord],
    cursor: &mut usize,
    gradients: &mut [Vec<f64>],
) -> Result<(Vec<f64>, usize)> {
    poll(engine)?;
    match layer {
        GpuCrownLayer::Linear {
            weight,
            bias,
            out_features,
            in_features,
            ..
        } => {
            if *in_features != dim {
                return Err(shape(*in_features, dim));
            }
            let weight_len = product(&[*out_features, *in_features], "adjoint linear weight")?;
            if weight.len() != weight_len {
                return Err(shape(weight_len, weight.len()));
            }
            let mut transposed =
                reserve_vec(engine, weight_len, "cuda::joint_alpha/linear_transpose")?;
            let mut transpose_work = 0usize;
            for input in 0..*in_features {
                for output in 0..*out_features {
                    poll_host_work(engine, &mut transpose_work)?;
                    transposed.push(f64::from(weight[output * *in_features + input]));
                }
            }
            poll(engine)?;
            let mut result = engine.gemm_f64(
                num_specs,
                *in_features,
                *out_features,
                &adjoint,
                &transposed,
            )?;
            poll(engine)?;
            if let Some(bias) = bias {
                if bias.len() != *out_features {
                    return Err(shape(*out_features, bias.len()));
                }
                let mut bias_work = 0usize;
                for spec in 0..num_specs {
                    for output in 0..*out_features {
                        poll_host_work(engine, &mut bias_work)?;
                        result[spec * *out_features + output] += f64::from(bias[output]);
                    }
                }
                poll(engine)?;
            }
            poll(engine)?;
            Ok((result, *out_features))
        }
        GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            in_h,
            in_w,
            ..
        } => {
            let geometry = ConvGeometry {
                out_channels: *out_channels,
                in_channels: *in_channels,
                kernel_h: *kernel_h,
                kernel_w: *kernel_w,
                stride_h: *stride_h,
                stride_w: *stride_w,
                pad_h: *pad_h,
                pad_w: *pad_w,
                out_h: *out_h,
                out_w: *out_w,
                in_h: *in_h,
                in_w: *in_w,
            };
            geometry.validate(geometry.output_dim()?, weight_col.len())?;
            if dim != geometry.input_dim()? {
                return Err(shape(geometry.input_dim()?, dim));
            }
            let result = adjoint_conv(
                engine,
                &adjoint,
                num_specs,
                weight_col,
                bias_expanded.as_deref(),
                geometry,
            )?;
            Ok((result, geometry.output_dim()?))
        }
        GpuCrownLayer::Activation { num_neurons, .. } => {
            let width = *num_neurons;
            if width != dim || *cursor == 0 {
                return Err(shape(width, dim));
            }
            *cursor -= 1;
            let record = &relus[*cursor];
            if record.width != width {
                return Err(shape(record.width, width));
            }
            let expected = product(&[num_specs, width], "ReLU adjoint")?;
            if adjoint.len() != expected
                || record.a_pre.len() != expected
                || record.sigma.len() != expected
                || record.tau.len() != expected
            {
                return Err(shape(expected, adjoint.len()));
            }
            let gradient = gradients.get_mut(*cursor).ok_or_else(|| {
                NyError::InvalidSpec("cuda joint alpha: missing gradient row".into())
            })?;
            let mut result =
                reserve_vec(engine, expected, "cuda::joint_alpha/relu_adjoint_result")?;
            let mut relu_work = 0usize;
            for spec in 0..num_specs {
                for neuron in 0..width {
                    poll_host_work(engine, &mut relu_work)?;
                    let index = spec * width + neuron;
                    gradient[neuron] += adjoint[index] * record.a_pre[index].max(0.0);
                    result.push(adjoint[index] * record.sigma[index] + record.tau[index]);
                }
            }
            poll(engine)?;
            Ok((result, width))
        }
        GpuCrownLayer::ActivationReluDualAlpha { .. } | GpuCrownLayer::MaxPool2d { .. } => Err(
            NyError::UnsupportedOp("cuda joint alpha: dual-alpha/maxpool is unsupported".into()),
        ),
    }
}

fn adjoint_conv<E: GemmEngine + ?Sized>(
    engine: &E,
    adjoint: &[f64],
    num_specs: usize,
    weight_col: &[f32],
    bias_expanded: Option<&[f32]>,
    geometry: ConvGeometry,
) -> Result<Vec<f64>> {
    poll(engine)?;
    let input_dim = geometry.input_dim()?;
    let output_dim = geometry.output_dim()?;
    let output_spatial = geometry.output_spatial()?;
    let kernel_columns = geometry.kernel_columns()?;
    let expected = product(&[num_specs, input_dim], "conv adjoint input")?;
    if adjoint.len() != expected {
        return Err(shape(expected, adjoint.len()));
    }

    // Gather each output position's receptive input patch. Padding entries stay 0.
    let rows = product(&[num_specs, output_spatial], "conv adjoint rows")?;
    let mut patches = zeroed_f64(
        engine,
        product(&[rows, kernel_columns], "conv adjoint patches")?,
        "cuda::joint_alpha/conv_adjoint_patches",
    )?;
    let mut gather_work = 0usize;
    for spec in 0..num_specs {
        for output_y in 0..geometry.out_h {
            for output_x in 0..geometry.out_w {
                let row =
                    (spec * output_spatial + output_y * geometry.out_w + output_x) * kernel_columns;
                for input_channel in 0..geometry.in_channels {
                    for kernel_y in 0..geometry.kernel_h {
                        for kernel_x in 0..geometry.kernel_w {
                            poll_host_work(engine, &mut gather_work)?;
                            let padded_y = output_y
                                .checked_mul(geometry.stride_h)
                                .and_then(|value| value.checked_add(kernel_y))
                                .ok_or_else(|| {
                                    NyError::InvalidSpec(
                                        "cuda joint alpha: adjoint conv input-y overflow".into(),
                                    )
                                })?;
                            if padded_y < geometry.pad_h {
                                continue;
                            }
                            let input_y = padded_y - geometry.pad_h;
                            if input_y >= geometry.in_h {
                                continue;
                            }
                            let padded_x = output_x
                                .checked_mul(geometry.stride_w)
                                .and_then(|value| value.checked_add(kernel_x))
                                .ok_or_else(|| {
                                    NyError::InvalidSpec(
                                        "cuda joint alpha: adjoint conv input-x overflow".into(),
                                    )
                                })?;
                            if padded_x < geometry.pad_w {
                                continue;
                            }
                            let input_x = padded_x - geometry.pad_w;
                            if input_x >= geometry.in_w {
                                continue;
                            }
                            let column = input_channel * geometry.kernel_h * geometry.kernel_w
                                + kernel_y * geometry.kernel_w
                                + kernel_x;
                            patches[row + column] = adjoint[spec * input_dim
                                + (input_channel * geometry.in_h + input_y) * geometry.in_w
                                + input_x];
                        }
                    }
                }
            }
        }
    }
    poll(engine)?;

    let transposed_len = product(
        &[kernel_columns, geometry.out_channels],
        "conv weight transpose",
    )?;
    let mut weight_transposed = reserve_vec(
        engine,
        transposed_len,
        "cuda::joint_alpha/conv_weight_transpose",
    )?;
    let mut transpose_work = 0usize;
    for column in 0..kernel_columns {
        for output_channel in 0..geometry.out_channels {
            poll_host_work(engine, &mut transpose_work)?;
            weight_transposed.push(f64::from(
                weight_col[output_channel * kernel_columns + column],
            ));
        }
    }
    poll(engine)?;
    let contracted = engine.gemm_f64(
        rows,
        kernel_columns,
        geometry.out_channels,
        &patches,
        &weight_transposed,
    )?;
    poll(engine)?;

    if let Some(bias) = bias_expanded {
        if bias.len() != output_dim {
            return Err(shape(output_dim, bias.len()));
        }
    }
    let mut result = zeroed_f64(
        engine,
        product(&[num_specs, output_dim], "conv adjoint result")?,
        "cuda::joint_alpha/conv_adjoint_result",
    )?;
    let mut result_work = 0usize;
    for spec in 0..num_specs {
        for spatial in 0..output_spatial {
            for output_channel in 0..geometry.out_channels {
                poll_host_work(engine, &mut result_work)?;
                let destination = spec * output_dim + output_channel * output_spatial + spatial;
                let mut value = contracted
                    [(spec * output_spatial + spatial) * geometry.out_channels + output_channel];
                if let Some(bias) = bias_expanded {
                    value += f64::from(bias[output_channel * output_spatial + spatial]);
                }
                result[destination] = value;
            }
        }
    }
    poll(engine)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Default)]
    struct RecordingGemm {
        launches: AtomicUsize,
    }

    impl GemmEngine for RecordingGemm {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Err(NyError::UnsupportedOp("test uses f64 only".into()))
        }

        fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
            self.launches.fetch_add(1, Ordering::SeqCst);
            let mut out = vec![0.0; m * n];
            for row in 0..m {
                for contraction in 0..k {
                    for column in 0..n {
                        out[row * n + column] +=
                            a[row * k + contraction] * b[contraction * n + column];
                    }
                }
            }
            Ok(out)
        }
    }

    #[derive(Default)]
    struct DeadlineOnSecondPoll {
        polls: AtomicUsize,
    }

    impl GemmEngine for DeadlineOnSecondPoll {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Err(NyError::UnsupportedOp("poll test performs no GEMM".into()))
        }

        fn poll_crown_backward_deadline(&self) -> Result<()> {
            let poll_number = self.polls.fetch_add(1, Ordering::SeqCst) + 1;
            if poll_number >= 2 {
                Err(NyError::DeadlineExceeded(
                    "scripted CUDA host-work deadline".into(),
                ))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn host_work_poll_cancels_before_unexecuted_tail() {
        let engine = DeadlineOnSecondPoll::default();
        let mut completed = 0usize;
        let mut executed = Vec::new();
        let result: Result<()> = (|| {
            for unit in 0..HOST_POLL_STRIDE + 4 {
                poll_host_work(&engine, &mut completed)?;
                executed.push(unit);
            }
            Ok(())
        })();

        assert!(matches!(result, Err(NyError::DeadlineExceeded(_))));
        assert_eq!(executed.len(), HOST_POLL_STRIDE);
        assert_eq!(executed.last(), Some(&(HOST_POLL_STRIDE - 1)));
        assert!(
            !executed.contains(&HOST_POLL_STRIDE),
            "the poll at the next bounded unit must leave the complete tail unexecuted"
        );
    }

    fn relu(alpha: [f32; 2]) -> GpuCrownLayer {
        GpuCrownLayer::Activation {
            lower_slope: alpha.to_vec(),
            upper_slope: vec![0.6, 0.7],
            lower_intercept: vec![0.0, 0.0],
            upper_intercept: vec![0.3, 0.2],
            num_neurons: 2,
        }
    }

    fn assert_gradient_parity(got: &[Vec<f32>], expected: &[Vec<f32>]) {
        assert_eq!(got.len(), expected.len(), "ReLU row count");
        for (row, (got, expected)) in got.iter().zip(expected).enumerate() {
            assert_eq!(got.len(), expected.len(), "ReLU row {row} width");
            for (neuron, (got, expected)) in got.iter().zip(expected).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-5,
                    "row={row} neuron={neuron}: CUDA-orchestrated={got} CPU-oracle={expected}"
                );
            }
        }
    }

    #[test]
    fn true_joint_linear_path_dispatches_gemm_and_matches_cpu_oracle() {
        let engine = RecordingGemm::default();
        let segments = vec![GpuResnetSegment::Chain(vec![
            GpuCrownLayer::Linear {
                weight: Arc::from(vec![0.7, -0.2, 0.4, 0.9]),
                bias: Some(Arc::from(vec![0.1, -0.3])),
                out_features: 2,
                in_features: 2,
                cert_err: Default::default(),
            },
            relu([0.25, 0.8]),
            GpuCrownLayer::Linear {
                weight: Arc::from(vec![0.5, -0.4, 0.6, 0.2]),
                bias: Some(Arc::from(vec![0.05, 0.15])),
                out_features: 2,
                in_features: 2,
                cert_err: Default::default(),
            },
        ])];
        let seed = [1.0f32, -0.75];
        let input_lower = [-1.0f32, -0.5];
        let input_upper = [0.8f32, 1.2];
        let got = crown_joint_alpha_gradient_with_deadline_impl(
            &engine,
            &segments,
            &seed,
            1,
            2,
            &input_lower,
            &input_upper,
        )
        .expect("deadline joint gradient");
        let expected = ny_core::joint_alpha_grad::joint_alpha_gradient(
            &segments,
            &seed,
            &[0.0],
            1,
            2,
            &input_lower,
            &input_upper,
            ny_core::joint_alpha_grad::JointGradConfig::default(),
        )
        .expect("CPU oracle");
        assert!(
            engine.launches.load(Ordering::SeqCst) >= 4,
            "forward and reverse affine maps must dispatch through GEMM"
        );
        assert_gradient_parity(&got, &expected);
    }

    #[test]
    fn true_joint_conv_forward_and_adjoint_dispatch_and_match_cpu_oracle() {
        let engine = RecordingGemm::default();
        let segments = vec![GpuResnetSegment::Chain(vec![
            GpuCrownLayer::Linear {
                weight: Arc::from(vec![0.7, -0.2, 0.4, 0.9, -0.1, 0.8, 0.3, -0.6]),
                bias: Some(Arc::from(vec![0.1, -0.3])),
                out_features: 2,
                in_features: 4,
                cert_err: Default::default(),
            },
            GpuCrownLayer::Activation {
                lower_slope: vec![0.25, 0.8, 0.4, 0.65],
                upper_slope: vec![0.6, 0.7, 0.55, 0.75],
                lower_intercept: vec![0.0; 4],
                upper_intercept: vec![0.3, 0.2, 0.1, 0.4],
                num_neurons: 4,
            },
            GpuCrownLayer::Conv2d {
                weight_col: Arc::from(vec![1.25]),
                bias_expanded: Some(Arc::from(vec![0.05, -0.1, 0.2, 0.15])),
                out_channels: 1,
                in_channels: 1,
                kernel_h: 1,
                kernel_w: 1,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
                out_h: 2,
                out_w: 2,
                in_h: 2,
                in_w: 2,
                cert_err: Default::default(),
            },
        ])];
        let seed = [1.0f32, -0.75];
        let input_lower = [-1.0f32, -0.5, -0.25, -0.8];
        let input_upper = [0.8f32, 1.2, 0.9, 0.6];
        let got = crown_joint_alpha_gradient_with_deadline_impl(
            &engine,
            &segments,
            &seed,
            1,
            2,
            &input_lower,
            &input_upper,
        )
        .expect("deadline joint gradient");
        let expected = ny_core::joint_alpha_grad::joint_alpha_gradient(
            &segments,
            &seed,
            &[0.0],
            1,
            2,
            &input_lower,
            &input_upper,
            ny_core::joint_alpha_grad::JointGradConfig::default(),
        )
        .expect("CPU oracle");
        assert_eq!(
            engine.launches.load(Ordering::SeqCst),
            4,
            "linear+conv forward and conv+linear adjoint each use one GEMM"
        );
        assert_gradient_parity(&got, &expected);
    }

    #[test]
    fn multichannel_strided_residual_projection_multispec_matches_cpu_oracle() {
        let engine = RecordingGemm::default();
        let activation = |width: usize, phase: f32| GpuCrownLayer::Activation {
            lower_slope: (0..width)
                .map(|index| 0.15 + ((index % 7) as f32) * 0.08 + phase)
                .collect(),
            upper_slope: (0..width)
                .map(|index| 0.45 + ((index % 5) as f32) * 0.06)
                .collect(),
            lower_intercept: vec![0.0; width],
            upper_intercept: (0..width)
                .map(|index| 0.03 + ((index % 4) as f32) * 0.02)
                .collect(),
            num_neurons: width,
        };
        let weights = |len: usize, scale: f32| -> Arc<[f32]> {
            Arc::from(
                (0..len)
                    .map(|index| ((index % 11) as f32 - 5.0) * scale)
                    .collect::<Vec<_>>(),
            )
        };
        let segments = vec![
            GpuResnetSegment::Chain(vec![GpuCrownLayer::Linear {
                weight: weights(2 * 8, 0.07),
                bias: Some(Arc::from(vec![0.12, -0.08])),
                out_features: 2,
                in_features: 8,
                cert_err: Default::default(),
            }]),
            GpuResnetSegment::ResidualProj(
                vec![
                    activation(8, 0.01),
                    GpuCrownLayer::Conv2d {
                        weight_col: weights(2 * 2 * 2 * 2, 0.05),
                        bias_expanded: Some(Arc::from(
                            (0..8)
                                .map(|index| (index as f32 - 3.0) * 0.015)
                                .collect::<Vec<_>>(),
                        )),
                        out_channels: 2,
                        in_channels: 2,
                        kernel_h: 2,
                        kernel_w: 2,
                        stride_h: 2,
                        stride_w: 2,
                        pad_h: 1,
                        pad_w: 1,
                        out_h: 2,
                        out_w: 2,
                        in_h: 3,
                        in_w: 3,
                        cert_err: Default::default(),
                    },
                ],
                vec![
                    activation(8, 0.02),
                    GpuCrownLayer::Linear {
                        weight: weights(8 * 18, 0.0125),
                        bias: Some(Arc::from(
                            (0..8)
                                .map(|index| (index as f32 - 4.0) * 0.01)
                                .collect::<Vec<_>>(),
                        )),
                        out_features: 8,
                        in_features: 18,
                        cert_err: Default::default(),
                    },
                ],
            ),
            GpuResnetSegment::Residual(vec![
                activation(18, 0.0),
                GpuCrownLayer::Linear {
                    weight: weights(18 * 18, 0.006),
                    bias: Some(Arc::from(
                        (0..18)
                            .map(|index| (index as f32 - 9.0) * 0.004)
                            .collect::<Vec<_>>(),
                    )),
                    out_features: 18,
                    in_features: 18,
                    cert_err: Default::default(),
                },
            ]),
        ];
        let num_specs = 3;
        let seed = [1.0f32, -0.75, -0.2, 0.9, 0.6, 0.35];
        let input_lower: Vec<f32> = (0..18)
            .map(|index| -1.2 + (index % 5) as f32 * 0.08)
            .collect();
        let input_upper: Vec<f32> = (0..18)
            .map(|index| 0.7 + (index % 7) as f32 * 0.06)
            .collect();

        let got = crown_joint_alpha_gradient_with_deadline_impl(
            &engine,
            &segments,
            &seed,
            num_specs,
            2,
            &input_lower,
            &input_upper,
        )
        .expect("deadline joint gradient");
        let expected = ny_core::joint_alpha_grad::joint_alpha_gradient(
            &segments,
            &seed,
            &[0.0; 3],
            num_specs,
            2,
            &input_lower,
            &input_upper,
            ny_core::joint_alpha_grad::JointGradConfig::default(),
        )
        .expect("CPU oracle");

        assert_eq!(
            engine.launches.load(Ordering::SeqCst),
            8,
            "four forward and four reverse affine contractions"
        );
        assert_gradient_parity(&got, &expected);
    }
}
