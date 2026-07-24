// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Resource-bounded, block-generator convolution experiment for [`Star`].
//!
//! This module is deliberately **unwired**.  The ordinary [`Star::conv2d`] remains
//! the scalar-row reference and no verifier, verdict path, or scored configuration
//! calls the method below.

use std::mem::size_of;

use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2, Array4, ArrayD, ArrayView2, ArrayView4, ArrayViewMut2, Ix4, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};

use super::Star;
use crate::zonotope::ZonotopeTensor;

const OUTPUT_SITE: &str = "Star::conv2d_blocked_unwired return materialization";
const WORKSPACE_SITE: &str = "Star::conv2d_blocked_unwired explicit workspace";
const PEAK_SITE: &str = "Star::conv2d_blocked_unwired peak owned allocation";

/// Explicit resource limits for [`Star::conv2d_blocked_unwired`].
///
/// There is intentionally no `Default`: an experimental caller must choose every
/// cap.  `max_workspace_bytes` covers the owned kernel matrix plus the reusable
/// im2col and GEMM buffers. `max_return_bytes` covers the output coefficients and
/// cloned predicate. `max_peak_owned_bytes` caps the modeled lifetime overlap of
/// those f32 backing buffers. Array headers, allocator bookkeeping, and
/// backend-private GEMM packing buffers are not observable through `ndarray` and
/// are therefore not included; this is one reason the primitive remains unwired.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StarConv2dBlockLimits {
    /// Maximum center/generator rows in one im2col/GEMM block; must be non-zero.
    pub block_rows: usize,
    /// Hard cap on explicitly owned reusable workspace bytes.
    pub max_workspace_bytes: usize,
    /// Hard cap on bytes retained by the returned star.
    pub max_return_bytes: usize,
    /// Hard cap on the peak overlap of return storage and explicit workspace.
    pub max_peak_owned_bytes: usize,
    /// Hard cap on scalar multiply-accumulate terms in the real contraction.
    pub max_multiply_accumulates: usize,
}

/// Checked operation and memory model for a block-generator convolution.
///
/// Counts exclude the input star (already resident) and backend-private GEMM
/// packing. All public byte/operation fields were computed with checked integer
/// arithmetic before an allocation is attempted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StarConv2dBlockPlan {
    /// Center plus generator rows.
    pub rows: usize,
    /// Effective rows per block (`min(limits.block_rows, rows)`).
    pub block_rows: usize,
    /// Number of GEMM calls (`ceil(rows / block_rows)`).
    pub gemm_calls: usize,
    /// Input channels.
    pub input_channels: usize,
    /// Input height.
    pub input_height: usize,
    /// Input width.
    pub input_width: usize,
    /// Output channels.
    pub output_channels: usize,
    /// Output height.
    pub output_height: usize,
    /// Output width.
    pub output_width: usize,
    /// Kernel height.
    pub kernel_height: usize,
    /// Kernel width.
    pub kernel_width: usize,
    /// GEMM reduction width (`input_channels * kernel_height * kernel_width`).
    pub patch_width: usize,
    /// Output coefficient elements, including center and all generators.
    pub coefficient_output_elements: usize,
    /// Output coefficient bytes.
    pub coefficient_output_bytes: usize,
    /// Bytes needed to preserve `A` and `b` in the returned star.
    pub predicate_clone_bytes: usize,
    /// Total retained bytes of the returned star.
    pub return_bytes: usize,
    /// Owned, flattened kernel bytes.
    pub kernel_bytes: usize,
    /// Maximum reusable block-im2col bytes.
    pub unfold_block_bytes: usize,
    /// Maximum reusable block-GEMM-output bytes.
    pub gemm_block_bytes: usize,
    /// Sum of explicitly owned temporary buffers.
    pub workspace_bytes: usize,
    /// Peak modeled f32 backing-buffer bytes under the implementation's lifetimes.
    pub peak_owned_bytes: usize,
    /// Number of real multiply-accumulate terms (padding zeros included by GEMM).
    pub multiply_accumulates: usize,
}

impl StarConv2dBlockPlan {
    /// Build a checked plan without materializing a star.
    ///
    /// `rows` includes the center, so the predicate width is `rows - 1`.
    /// This estimator is useful for architecture-level models such as Metaroom;
    /// the executable method calls the same planner with its actual dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate(
        rows: usize,
        predicate_rows: usize,
        input_shape: [usize; 3],
        weight_shape: [usize; 4],
        stride: (usize, usize),
        padding: (usize, usize),
        limits: StarConv2dBlockLimits,
    ) -> Result<Self> {
        if rows == 0 {
            return Err(NyError::InvalidSpec(
                "StarConv2dBlockPlan::estimate requires at least the center row".to_string(),
            ));
        }
        if limits.block_rows == 0 {
            return Err(NyError::InvalidConfig(
                "Star::conv2d_blocked_unwired block_rows must be non-zero".to_string(),
            ));
        }

        let [input_channels, input_height, input_width] = input_shape;
        let [output_channels, weight_input_channels, kernel_height, kernel_width] = weight_shape;
        if input_channels != weight_input_channels {
            return Err(NyError::shape_mismatch(
                vec![output_channels, input_channels, kernel_height, kernel_width],
                weight_shape.to_vec(),
            ));
        }
        if input_channels == 0 || input_height == 0 || input_width == 0 {
            return Err(NyError::InvalidSpec(format!(
                "Star::conv2d_blocked_unwired requires non-empty input, got {input_shape:?}"
            )));
        }
        if output_channels == 0 || kernel_height == 0 || kernel_width == 0 {
            return Err(NyError::InvalidSpec(format!(
                "Star::conv2d_blocked_unwired requires non-empty weights, got {weight_shape:?}"
            )));
        }
        let (stride_height, stride_width) = stride;
        if stride_height == 0 || stride_width == 0 {
            return Err(NyError::InvalidSpec(format!(
                "Star::conv2d_blocked_unwired requires non-zero stride, got {stride:?}"
            )));
        }

        let (pad_height, pad_width) = padding;
        let double_pad_height = pad_height.checked_mul(2).ok_or_else(|| {
            NyError::InvalidSpec(
                "Star::conv2d_blocked_unwired vertical padding overflows usize".to_string(),
            )
        })?;
        let double_pad_width = pad_width.checked_mul(2).ok_or_else(|| {
            NyError::InvalidSpec(
                "Star::conv2d_blocked_unwired horizontal padding overflows usize".to_string(),
            )
        })?;
        let padded_height = input_height.checked_add(double_pad_height).ok_or_else(|| {
            NyError::InvalidSpec(
                "Star::conv2d_blocked_unwired padded height overflows usize".to_string(),
            )
        })?;
        let padded_width = input_width.checked_add(double_pad_width).ok_or_else(|| {
            NyError::InvalidSpec(
                "Star::conv2d_blocked_unwired padded width overflows usize".to_string(),
            )
        })?;
        if padded_height < kernel_height || padded_width < kernel_width {
            return Err(NyError::ShapeMismatch {
                expected: vec![kernel_height, kernel_width],
                got: vec![padded_height, padded_width],
            });
        }
        let output_height = (padded_height - kernel_height) / stride_height + 1;
        let output_width = (padded_width - kernel_width) / stride_width + 1;
        let output_spatial = checked_product(
            &[output_height, output_width],
            "Star::conv2d_blocked_unwired output spatial shape",
        )?;
        let patch_width = checked_product(
            &[input_channels, kernel_height, kernel_width],
            "Star::conv2d_blocked_unwired patch width",
        )?;
        let coefficient_output_elements = checked_product(
            &[rows, output_channels, output_spatial],
            "Star::conv2d_blocked_unwired output coefficient elements",
        )?;
        let coefficient_output_bytes = checked_bytes(
            coefficient_output_elements,
            "Star::conv2d_blocked_unwired output coefficient bytes",
        )?;

        let alpha_dim = rows - 1;
        let predicate_matrix_elements = checked_product(
            &[predicate_rows, alpha_dim],
            "Star::conv2d_blocked_unwired predicate matrix elements",
        )?;
        let predicate_elements = predicate_matrix_elements
            .checked_add(predicate_rows)
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "Star::conv2d_blocked_unwired predicate elements overflow usize".to_string(),
                )
            })?;
        let predicate_clone_bytes = checked_bytes(
            predicate_elements,
            "Star::conv2d_blocked_unwired predicate clone bytes",
        )?;
        let return_bytes = coefficient_output_bytes
            .checked_add(predicate_clone_bytes)
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "Star::conv2d_blocked_unwired return bytes overflow usize".to_string(),
                )
            })?;

        let block_rows = limits.block_rows.min(rows);
        let gemm_calls = rows.div_ceil(block_rows);
        let block_spatial = checked_product(
            &[block_rows, output_spatial],
            "Star::conv2d_blocked_unwired block spatial rows",
        )?;
        let kernel_elements = checked_product(
            &[output_channels, patch_width],
            "Star::conv2d_blocked_unwired kernel elements",
        )?;
        let unfold_elements = checked_product(
            &[block_spatial, patch_width],
            "Star::conv2d_blocked_unwired block unfold elements",
        )?;
        let gemm_elements = checked_product(
            &[output_channels, block_spatial],
            "Star::conv2d_blocked_unwired block GEMM elements",
        )?;
        let kernel_bytes =
            checked_bytes(kernel_elements, "Star::conv2d_blocked_unwired kernel bytes")?;
        let unfold_block_bytes =
            checked_bytes(unfold_elements, "Star::conv2d_blocked_unwired unfold bytes")?;
        let gemm_block_bytes =
            checked_bytes(gemm_elements, "Star::conv2d_blocked_unwired GEMM bytes")?;
        let workspace_bytes = kernel_bytes
            .checked_add(unfold_block_bytes)
            .and_then(|value| value.checked_add(gemm_block_bytes))
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "Star::conv2d_blocked_unwired workspace bytes overflow usize".to_string(),
                )
            })?;
        // Predicate cloning happens only after the GEMM workspace is dropped.
        let output_plus_workspace = coefficient_output_bytes
            .checked_add(workspace_bytes)
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "Star::conv2d_blocked_unwired peak bytes overflow usize".to_string(),
                )
            })?;
        let peak_owned_bytes = output_plus_workspace.max(return_bytes);
        let multiply_accumulates = coefficient_output_elements
            .checked_mul(patch_width)
            .ok_or_else(|| {
                NyError::InvalidSpec(
                    "Star::conv2d_blocked_unwired operation count overflows usize".to_string(),
                )
            })?;

        enforce_memory_limit(return_bytes, limits.max_return_bytes, OUTPUT_SITE)?;
        enforce_memory_limit(workspace_bytes, limits.max_workspace_bytes, WORKSPACE_SITE)?;
        enforce_memory_limit(peak_owned_bytes, limits.max_peak_owned_bytes, PEAK_SITE)?;
        if multiply_accumulates > limits.max_multiply_accumulates {
            return Err(NyError::InvalidConfig(format!(
                "Star::conv2d_blocked_unwired requires {multiply_accumulates} multiply-accumulates, cap is {}",
                limits.max_multiply_accumulates
            )));
        }

        Ok(Self {
            rows,
            block_rows,
            gemm_calls,
            input_channels,
            input_height,
            input_width,
            output_channels,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            patch_width,
            coefficient_output_elements,
            coefficient_output_bytes,
            predicate_clone_bytes,
            return_bytes,
            kernel_bytes,
            unfold_block_bytes,
            gemm_block_bytes,
            workspace_bytes,
            peak_owned_bytes,
            multiply_accumulates,
        })
    }
}

impl Star {
    /// Plan the experimental blocked convolution using this star's true dimensions.
    pub fn plan_conv2d_blocked_unwired(
        &self,
        weight: &Array4<f32>,
        stride: (usize, usize),
        padding: (usize, usize),
        limits: StarConv2dBlockLimits,
    ) -> Result<StarConv2dBlockPlan> {
        let shape = self.shape();
        if shape.len() != 3 {
            return Err(NyError::InvalidSpec(format!(
                "Star::conv2d_blocked_unwired expects (C, H, W) value shape, got {shape:?}"
            )));
        }

        // `ZonotopeTensor::new` historically accepts a zero-length leading axis and
        // derives `n_error_terms` with `saturating_sub`. Other crate-private builders
        // can also construct metadata/backing mismatches. Do not infer an executable
        // row from that metadata: validate the actual backing first, derive `rows`
        // from it, and fail closed before any planner allocation or executor index.
        let backing_shape = self.zono.coeffs().shape();
        if backing_shape.len() != 4 {
            return Err(NyError::InvalidSpec(format!(
                "Star::conv2d_blocked_unwired coefficient backing must have shape \
                 [1 + alpha_dim, C, H, W], got {backing_shape:?}"
            )));
        }
        if backing_shape[0] == 0 {
            return Err(NyError::InvalidSpec(format!(
                "Star::conv2d_blocked_unwired coefficient backing requires at least one \
                 center row, got {backing_shape:?}"
            )));
        }
        let rows = backing_shape[0];
        let backing_alpha_dim = rows - 1;
        let expected_rows = self.alpha_dim().checked_add(1).ok_or_else(|| {
            NyError::InvalidSpec(
                "Star::conv2d_blocked_unwired row count overflows usize".to_string(),
            )
        })?;
        let expected_backing_shape = [expected_rows, shape[0], shape[1], shape[2]];
        if backing_shape != expected_backing_shape {
            return Err(NyError::InvalidSpec(format!(
                "Star::conv2d_blocked_unwired coefficient backing must be exactly \
                 [1 + alpha_dim, C, H, W] = {expected_backing_shape:?}, got \
                 {backing_shape:?}"
            )));
        }
        if self.a.nrows() != self.b.len() || self.a.ncols() != backing_alpha_dim {
            return Err(NyError::InvalidSpec(format!(
                "Star::conv2d_blocked_unwired predicate backing must be ({}, {}), got {:?}",
                self.b.len(),
                backing_alpha_dim,
                self.a.shape()
            )));
        }
        let weight_shape = weight.shape();
        StarConv2dBlockPlan::estimate(
            rows,
            self.num_constraints(),
            [shape[0], shape[1], shape[2]],
            [
                weight_shape[0],
                weight_shape[1],
                weight_shape[2],
                weight_shape[3],
            ],
            stride,
            padding,
            limits,
        )
    }

    /// Experimental block-generator `Conv2d`; **unwired and not verdict-safe**.
    ///
    /// This computes the same real cross-correlation as [`Star::conv2d`] while
    /// replacing one im2col allocation and one GEMM per center/generator with one
    /// reusable block im2col buffer and `ceil(rows / block_rows)` GEMMs. Bias is
    /// still added only to the center. Predicate rows are copied unchanged.
    ///
    /// # Floating-point contract
    ///
    /// The reduction axis has the same `(channel, kernel_h, kernel_w)` order as
    /// [`Star::conv2d`], but enlarging GEMM's column dimension can change backend
    /// tiling, FMA use, or reduction grouping (and macOS may route to Accelerate).
    /// Results are therefore **not promised bit-identical** to the scalar-row
    /// reference. This method supplies no directed-rounding enclosure or certified
    /// remainder for that difference. It must not authorize SAT/UNSAT, pruning, or
    /// scored bounds until a separate f64/error-budget layer certifies the output.
    ///
    /// # Resource contract
    ///
    /// The coefficient backing is first checked to be exactly
    /// `[1 + alpha_dim, C, H, W]`. Every shape product, byte count, and
    /// source/destination index domain is then checked before allocation. Explicit
    /// allocations use `try_reserve_exact` and are gated by all limits. The peak
    /// model covers only named f32 backing buffers; array headers, allocator
    /// bookkeeping, and GEMM-private packing are excluded. Non-finite
    /// inputs/weights/biases fail closed. The unmodelled backend scratch is another
    /// activation gate.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_blocked_unwired(
        &self,
        weight: &Array4<f32>,
        bias: Option<&Array1<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
        limits: StarConv2dBlockLimits,
    ) -> Result<Self> {
        let plan = self.plan_conv2d_blocked_unwired(weight, stride, padding, limits)?;
        if let Some(bias) = bias {
            if bias.len() != plan.output_channels {
                return Err(NyError::shape_mismatch(
                    vec![plan.output_channels],
                    vec![bias.len()],
                ));
            }
            reject_non_finite("bias", bias.iter().copied())?;
        }
        reject_non_finite("weight", weight.iter().copied())?;
        reject_non_finite("star coefficients", self.zono.coeffs().iter().copied())?;
        reject_non_finite("predicate A", self.a.iter().copied())?;
        reject_non_finite("predicate b", self.b.iter().copied())?;

        let coeffs4 = self
            .zono
            .coeffs()
            .view()
            .into_dimensionality::<Ix4>()
            .map_err(|error| {
                NyError::InternalError(format!(
                    "Star::conv2d_blocked_unwired coefficient rank invariant: {error}"
                ))
            })?;

        // Scope all GEMM temporaries so predicate cloning cannot overlap them.
        let output_data = execute_blocks(coeffs4, weight, bias, stride, padding, &plan)?;
        let output = ArrayD::from_shape_vec(
            IxDyn(&[
                plan.rows,
                plan.output_channels,
                plan.output_height,
                plan.output_width,
            ]),
            output_data,
        )
        .map_err(|error| {
            NyError::InternalError(format!(
                "Star::conv2d_blocked_unwired output shape invariant: {error}"
            ))
        })?;
        let zono = ZonotopeTensor::new(output)?;
        let a = try_clone_array2(&self.a, limits.max_return_bytes)?;
        let b = try_clone_array1(&self.b, limits.max_return_bytes)?;
        Self::new(zono, a, b)
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_blocks(
    coeffs: ArrayView4<'_, f32>,
    weight: &Array4<f32>,
    bias: Option<&Array1<f32>>,
    stride: (usize, usize),
    padding: (usize, usize),
    plan: &StarConv2dBlockPlan,
) -> Result<Vec<f32>> {
    let output_spatial = checked_product(
        &[plan.output_height, plan.output_width],
        "Star::conv2d_blocked_unwired execution output spatial",
    )?;
    let max_block_spatial = checked_product(
        &[plan.block_rows, output_spatial],
        "Star::conv2d_blocked_unwired execution block spatial",
    )?;
    let max_unfold_elements = checked_product(
        &[max_block_spatial, plan.patch_width],
        "Star::conv2d_blocked_unwired execution unfold elements",
    )?;
    let max_gemm_elements = checked_product(
        &[plan.output_channels, max_block_spatial],
        "Star::conv2d_blocked_unwired execution GEMM elements",
    )?;

    let mut output_data = try_zeroed_f32(
        plan.coefficient_output_elements,
        plan.coefficient_output_bytes,
        OUTPUT_SITE,
    )?;
    let kernel_elements = checked_product(
        &[plan.output_channels, plan.patch_width],
        "Star::conv2d_blocked_unwired execution kernel elements",
    )?;
    let mut kernel_data = try_zeroed_f32(kernel_elements, plan.kernel_bytes, WORKSPACE_SITE)?;
    flatten_kernel(weight, &mut kernel_data, plan);
    let kernel = ArrayView2::from_shape(
        (plan.output_channels, plan.patch_width),
        kernel_data.as_slice(),
    )
    .map_err(|error| {
        NyError::InternalError(format!(
            "Star::conv2d_blocked_unwired kernel shape invariant: {error}"
        ))
    })?;
    let mut unfold_data =
        try_zeroed_f32(max_unfold_elements, plan.unfold_block_bytes, WORKSPACE_SITE)?;
    let mut gemm_data = try_zeroed_f32(max_gemm_elements, plan.gemm_block_bytes, WORKSPACE_SITE)?;

    let mut row_start = 0usize;
    while row_start < plan.rows {
        let active_rows = plan.block_rows.min(plan.rows - row_start);
        let active_spatial = checked_product(
            &[active_rows, output_spatial],
            "Star::conv2d_blocked_unwired active block spatial",
        )?;
        let active_unfold_elements = checked_product(
            &[active_spatial, plan.patch_width],
            "Star::conv2d_blocked_unwired active unfold elements",
        )?;
        let active_gemm_elements = checked_product(
            &[plan.output_channels, active_spatial],
            "Star::conv2d_blocked_unwired active GEMM elements",
        )?;
        fill_unfold_block(
            coeffs,
            row_start,
            active_rows,
            stride,
            padding,
            plan,
            &mut unfold_data[..active_unfold_elements],
        );
        let unfolded = ArrayView2::from_shape(
            (active_spatial, plan.patch_width),
            &unfold_data[..active_unfold_elements],
        )
        .map_err(|error| {
            NyError::InternalError(format!(
                "Star::conv2d_blocked_unwired unfold shape invariant: {error}"
            ))
        })?;
        let mut gemm = ArrayViewMut2::from_shape(
            (plan.output_channels, active_spatial),
            &mut gemm_data[..active_gemm_elements],
        )
        .map_err(|error| {
            NyError::InternalError(format!(
                "Star::conv2d_blocked_unwired GEMM shape invariant: {error}"
            ))
        })?;
        general_mat_mul(1.0_f32, &kernel, &unfolded.t(), 0.0_f32, &mut gemm);

        for local_row in 0..active_rows {
            let destination_row = row_start + local_row;
            for output_channel in 0..plan.output_channels {
                let source_start = output_channel * active_spatial + local_row * output_spatial;
                let destination_start =
                    (destination_row * plan.output_channels + output_channel) * output_spatial;
                output_data[destination_start..destination_start + output_spatial]
                    .copy_from_slice(&gemm_data[source_start..source_start + output_spatial]);
            }
        }
        row_start += active_rows;
    }

    if let Some(bias) = bias {
        for output_channel in 0..plan.output_channels {
            let start = output_channel * output_spatial;
            let channel_bias = bias[output_channel];
            for value in &mut output_data[start..start + output_spatial] {
                *value += channel_bias;
            }
        }
    }
    reject_non_finite("convolution output", output_data.iter().copied())?;
    Ok(output_data)
}

fn flatten_kernel(weight: &Array4<f32>, destination: &mut [f32], plan: &StarConv2dBlockPlan) {
    let kernel_height = weight.shape()[2];
    let kernel_width = weight.shape()[3];
    for output_channel in 0..plan.output_channels {
        for input_channel in 0..plan.input_channels {
            for kernel_row in 0..kernel_height {
                for kernel_column in 0..kernel_width {
                    let patch_column =
                        (input_channel * kernel_height + kernel_row) * kernel_width + kernel_column;
                    destination[output_channel * plan.patch_width + patch_column] =
                        weight[[output_channel, input_channel, kernel_row, kernel_column]];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_unfold_block(
    coeffs: ArrayView4<'_, f32>,
    row_start: usize,
    active_rows: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    plan: &StarConv2dBlockPlan,
    destination: &mut [f32],
) {
    let kernel_height = plan.kernel_height;
    let kernel_width = plan.kernel_width;
    let output_spatial = plan.output_height * plan.output_width;
    let (stride_height, stride_width) = stride;
    let (pad_height, pad_width) = padding;
    for local_row in 0..active_rows {
        let source_row = row_start + local_row;
        for output_row in 0..plan.output_height {
            for output_column in 0..plan.output_width {
                let patch_row =
                    local_row * output_spatial + output_row * plan.output_width + output_column;
                for input_channel in 0..plan.input_channels {
                    for kernel_row in 0..kernel_height {
                        for kernel_column in 0..kernel_width {
                            let padded_input_row = output_row * stride_height + kernel_row;
                            let padded_input_column = output_column * stride_width + kernel_column;
                            let value = if padded_input_row >= pad_height
                                && padded_input_column >= pad_width
                            {
                                let input_row = padded_input_row - pad_height;
                                let input_column = padded_input_column - pad_width;
                                if input_row < plan.input_height && input_column < plan.input_width
                                {
                                    coeffs[[source_row, input_channel, input_row, input_column]]
                                } else {
                                    0.0
                                }
                            } else {
                                0.0
                            };
                            let patch_column = (input_channel * kernel_height + kernel_row)
                                * kernel_width
                                + kernel_column;
                            destination[patch_row * plan.patch_width + patch_column] = value;
                        }
                    }
                }
            }
        }
    }
}

fn checked_product(values: &[usize], context: &str) -> Result<usize> {
    checked_shape_product(values)
        .ok_or_else(|| NyError::InvalidSpec(format!("{context} overflows usize: {values:?}")))
}

fn checked_bytes(elements: usize, context: &str) -> Result<usize> {
    elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| NyError::InvalidSpec(format!("{context} overflows usize")))
}

fn enforce_memory_limit(required: usize, budget: usize, site: &'static str) -> Result<()> {
    if required > budget {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: required,
            budget_bytes: budget,
            site,
        });
    }
    Ok(())
}

fn try_zeroed_f32(elements: usize, required_bytes: usize, site: &'static str) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|error| {
        NyError::InternalError(format!(
            "{site}: allocator refused {required_bytes} bytes after resource validation: {error}"
        ))
    })?;
    values.resize(elements, 0.0);
    Ok(values)
}

fn try_clone_array2(source: &Array2<f32>, budget: usize) -> Result<Array2<f32>> {
    let elements = checked_product(
        &[source.nrows(), source.ncols()],
        "Star::conv2d_blocked_unwired predicate A clone elements",
    )?;
    let bytes = checked_bytes(
        elements,
        "Star::conv2d_blocked_unwired predicate A clone bytes",
    )?;
    enforce_memory_limit(bytes, budget, OUTPUT_SITE)?;
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|error| {
        NyError::InternalError(format!(
            "Star::conv2d_blocked_unwired predicate A allocation failed: {error}"
        ))
    })?;
    for row in 0..source.nrows() {
        for column in 0..source.ncols() {
            values.push(source[[row, column]]);
        }
    }
    Array2::from_shape_vec((source.nrows(), source.ncols()), values).map_err(|error| {
        NyError::InternalError(format!(
            "Star::conv2d_blocked_unwired predicate A shape invariant: {error}"
        ))
    })
}

fn try_clone_array1(source: &Array1<f32>, budget: usize) -> Result<Array1<f32>> {
    let bytes = checked_bytes(
        source.len(),
        "Star::conv2d_blocked_unwired predicate b clone bytes",
    )?;
    enforce_memory_limit(bytes, budget, OUTPUT_SITE)?;
    let mut values = Vec::new();
    values.try_reserve_exact(source.len()).map_err(|error| {
        NyError::InternalError(format!(
            "Star::conv2d_blocked_unwired predicate b allocation failed: {error}"
        ))
    })?;
    values.extend(source.iter().copied());
    Ok(Array1::from_vec(values))
}

fn reject_non_finite(name: &str, values: impl Iterator<Item = f32>) -> Result<()> {
    if values.into_iter().any(|value| !value.is_finite()) {
        return Err(NyError::NumericalInstability(format!(
            "Star::conv2d_blocked_unwired rejects non-finite {name}"
        )));
    }
    Ok(())
}
