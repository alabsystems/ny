// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Resize layer: nearest-neighbor spatial upsample over the last two dims.
//!
//! The implementation follows alpha-beta-CROWN's nearest-only ONNX Resize path:
//! - IBP is exact because nearest-neighbor upsample is monotone replication.
//! - CROWN backward sums each `scale_h x scale_w` output block back to the
//!   corresponding input cell, matching the reference `avg_pool2d(...,
//!   divisor_override=1)` formulation.
//!
//! Reference:
//! `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/resize.py:27-82`

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_up_f32, BoundedTensor};
use std::borrow::Cow;

use super::super::common::BoundPropagation;
use crate::{contiguous_flat_slice, contiguous_flat_slice_mut, BatchedLinearBounds, LinearBounds};

/// Nearest-neighbor spatial resize over the last two tensor dimensions.
#[derive(Debug, Clone)]
pub struct ResizeLayer {
    /// Positive integer scale factor for the penultimate (height) axis.
    pub scale_h: usize,
    /// Positive integer scale factor for the last (width) axis.
    pub scale_w: usize,
}

impl ResizeLayer {
    /// Create a new nearest-neighbor resize layer.
    pub fn new(scale_h: usize, scale_w: usize) -> Self {
        Self { scale_h, scale_w }
    }

    fn validate_input_shape(&self, input_shape: &[usize]) -> Result<()> {
        if self.scale_h == 0 || self.scale_w == 0 {
            return Err(NyError::InvalidSpec(
                "Resize scale factors must be positive".to_string(),
            ));
        }
        if input_shape.len() < 2 {
            return Err(NyError::InvalidSpec(format!(
                "Resize expects at least 2D input, got shape {:?}",
                input_shape
            )));
        }
        Ok(())
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        self.validate_input_shape(input_shape)?;

        let mut output_shape = input_shape.to_vec();
        let h_idx = output_shape.len() - 2;
        let w_idx = output_shape.len() - 1;
        output_shape[h_idx] = output_shape[h_idx]
            .checked_mul(self.scale_h)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Resize output height overflow: {} * {}",
                    input_shape[h_idx], self.scale_h
                ))
            })?;
        output_shape[w_idx] = output_shape[w_idx]
            .checked_mul(self.scale_w)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Resize output width overflow: {} * {}",
                    input_shape[w_idx], self.scale_w
                ))
            })?;
        Ok(output_shape)
    }

    fn compute_strides(shape: &[usize]) -> Result<Vec<usize>> {
        if shape.is_empty() {
            return Ok(Vec::new());
        }

        let mut strides = vec![1usize; shape.len()];
        for idx in (0..shape.len() - 1).rev() {
            strides[idx] = strides[idx + 1]
                .checked_mul(shape[idx + 1])
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!("Resize stride overflow for shape {:?}", shape))
                })?;
        }
        Ok(strides)
    }

    /// Map each flattened output position to the flattened input position it reads.
    fn build_output_to_input_map(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        let output_shape = self.output_shape(input_shape)?;
        let output_size = checked_shape_product(&output_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Resize output shape product overflows usize: {:?}",
                output_shape
            ))
        })?;
        let input_strides = Self::compute_strides(input_shape)?;
        let output_strides = Self::compute_strides(&output_shape)?;

        let mut mapping = Vec::with_capacity(output_size);
        for out_flat in 0..output_size {
            let mut out_index = vec![0usize; output_shape.len()];
            let mut remainder = out_flat;
            for (axis, &stride) in output_strides.iter().enumerate() {
                out_index[axis] = remainder / stride;
                remainder %= stride;
            }

            let h_idx = out_index.len() - 2;
            let w_idx = out_index.len() - 1;
            out_index[h_idx] /= self.scale_h;
            out_index[w_idx] /= self.scale_w;

            let in_flat = out_index
                .iter()
                .zip(input_strides.iter())
                .try_fold(0usize, |acc, (&idx, &stride)| {
                    acc.checked_add(idx.checked_mul(stride)?)
                })
                .ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Resize input index overflow for shape {:?}",
                        input_shape
                    ))
                })?;
            mapping.push(in_flat);
        }

        Ok(mapping)
    }

    fn upsample_last_two(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let input_shape = input.shape().to_vec();
        let output_shape = self.output_shape(&input_shape)?;
        let mapping = self.build_output_to_input_map(&input_shape)?;

        let flat_input = contiguous_flat_slice(input);
        let output = mapping
            .iter()
            .map(|&in_flat| flat_input[in_flat])
            .collect::<Vec<_>>();

        ArrayD::from_shape_vec(IxDyn(&output_shape), output)
            .map_err(|e| NyError::InvalidSpec(format!("Resize output reshape failed: {}", e)))
    }

    fn propagate_linear_for_shape(
        &self,
        bounds: &LinearBounds,
        input_shape: &[usize],
    ) -> Result<LinearBounds> {
        let input_size = checked_shape_product(input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Resize input shape product overflows usize: {:?}",
                input_shape
            ))
        })?;
        let mapping = self.build_output_to_input_map(input_shape)?;
        let output_size = mapping.len();

        if bounds.num_inputs() != output_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![output_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));

        // Inverse map: nearest-neighbor upsample sends scale_h*scale_w DUPLICATE output
        // columns into each input column, so the backward f32-accumulates a fan-in > 1 sum
        // that rounds (gather-class). Certify it via the shared gather_backward_coeff_err
        // (gamma_k*S + prop), else new_or_conservative trusts the rounded coeff as exact —
        // a false-proof under cancellation. (#vnncomp-aw-soundness self-audit.)
        let mut in_to_outs: Vec<Vec<usize>> = vec![Vec::new(); input_size];
        for (out_flat, &in_flat) in mapping.iter().enumerate() {
            in_to_outs[in_flat].push(out_flat);
            for row in 0..num_outputs {
                new_lower_a[[row, in_flat]] += bounds.lower_a()[[row, out_flat]];
                new_upper_a[[row, in_flat]] += bounds.upper_a()[[row, out_flat]];
            }
        }

        let has_duplicate = in_to_outs.iter().any(|outs| outs.len() >= 2);
        let has_incoming_err = bounds.lower_a_err().is_some() || bounds.upper_a_err().is_some();
        if has_duplicate || has_incoming_err {
            let (lower_err, upper_err) = super::gather::gather_backward_coeff_err(
                bounds.lower_a(),
                bounds.upper_a(),
                bounds.lower_a_err(),
                bounds.upper_a_err(),
                input_size,
                &in_to_outs,
            );
            return LinearBounds::new_or_conservative_with_err(
                new_lower_a,
                bounds.lower_b().clone(),
                new_upper_a,
                bounds.upper_b().clone(),
                lower_err,
                upper_err,
            );
        }

        LinearBounds::new_or_conservative(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
        )
    }

    /// Exact dense CROWN backward for nearest-neighbor Resize.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        self.propagate_linear_for_shape(bounds, pre_activation.shape())
    }

    /// Batched CROWN backward for flattened A-matrices.
    ///
    /// This matches Tile/Slice: it supports the common flattened-column case
    /// where the last A dimension equals the full resized tensor size. If the
    /// caller keeps only the last logical tensor dim in A, this returns a shape
    /// mismatch so the graph engine can fall back to another path.
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let input_shape = pre_activation.shape().to_vec();
        let input_size = checked_shape_product(&input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Resize batched CROWN input shape product overflows usize: {:?}",
                input_shape
            ))
        })?;
        let mapping = self.build_output_to_input_map(&input_shape)?;
        let output_size = mapping.len();

        let a_shape = bounds.lower_a().shape();
        let a_ndim = a_shape.len();
        if a_ndim < 2 {
            return Err(NyError::InvalidSpec(
                "Resize batched CROWN: A matrices must have at least 2 dimensions".to_string(),
            ));
        }
        if a_shape[a_ndim - 1] != output_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![output_size],
                got: vec![a_shape[a_ndim - 1]],
            });
        }

        let mut new_a_shape = a_shape.to_vec();
        new_a_shape[a_ndim - 1] = input_size;
        let mut new_lower_a = ArrayD::<f32>::zeros(IxDyn(&new_a_shape));
        let mut new_upper_a = ArrayD::<f32>::zeros(IxDyn(&new_a_shape));

        let outer_size = checked_shape_product(&a_shape[..a_ndim - 1]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Resize batched CROWN outer shape product overflows: {:?}",
                &a_shape[..a_ndim - 1],
            ))
        })?;

        let flat_lower = contiguous_flat_slice(bounds.lower_a());
        let flat_upper = contiguous_flat_slice(bounds.upper_a());
        // Certified scatter-add error (#vnncomp-aw-soundness), same as the dense path: each
        // input cell with fan-in k = #(output cols mapped to it) f32-accumulates k coeffs.
        let flat_lower_err = bounds
            .lower_a_err
            .as_ref()
            .and_then(|e| e.as_slice_memory_order());
        let flat_upper_err = bounds
            .upper_a_err
            .as_ref()
            .and_then(|e| e.as_slice_memory_order());
        let mut fan_in = vec![0usize; input_size];
        for &in_col in mapping.iter() {
            fan_in[in_col] += 1;
        }
        let new_len = outer_size * input_size;
        let mut s_lower = vec![0.0f64; new_len];
        let mut s_upper = vec![0.0f64; new_len];
        let mut p_lower = vec![0.0f64; new_len];
        let mut p_upper = vec![0.0f64; new_len];

        {
            let new_lower_flat = contiguous_flat_slice_mut(&mut new_lower_a)?;
            let new_upper_flat = contiguous_flat_slice_mut(&mut new_upper_a)?;
            for row in 0..outer_size {
                let old_base = row * output_size;
                let new_base = row * input_size;
                for (out_col, &in_col) in mapping.iter().enumerate() {
                    let slot = new_base + in_col;
                    new_lower_flat[slot] += flat_lower[old_base + out_col];
                    new_upper_flat[slot] += flat_upper[old_base + out_col];
                    s_lower[slot] += (flat_lower[old_base + out_col] as f64).abs();
                    s_upper[slot] += (flat_upper[old_base + out_col] as f64).abs();
                    if let Some(e) = flat_lower_err {
                        p_lower[slot] += (e[old_base + out_col] as f64).abs();
                    }
                    if let Some(e) = flat_upper_err {
                        p_upper[slot] += (e[old_base + out_col] as f64).abs();
                    }
                }
            }
        }

        let mut out = BatchedLinearBounds::new_or_conservative(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
            input_shape,
            bounds.output_shape().to_vec(),
        )?;
        if fan_in.iter().any(|&k| k >= 2) || flat_lower_err.is_some() || flat_upper_err.is_some() {
            let mut le = vec![0.0f32; new_len];
            let mut ue = vec![0.0f32; new_len];
            for row in 0..outer_size {
                let new_base = row * input_size;
                for (in_col, &k) in fan_in.iter().enumerate() {
                    let slot = new_base + in_col;
                    let gamma = if k >= 2 {
                        crate::layers::linear::crown_single_gamma_n_f32(k)
                    } else {
                        0.0
                    };
                    le[slot] = next_up_f32((gamma * s_lower[slot] + p_lower[slot]) as f32);
                    ue[slot] = next_up_f32((gamma * s_upper[slot] + p_upper[slot]) as f32);
                }
            }
            let lower_err = ArrayD::from_shape_vec(IxDyn(&new_a_shape), le)
                .map_err(|_| NyError::InvalidSpec("Resize coeff err reshape".to_string()))?;
            let upper_err = ArrayD::from_shape_vec(IxDyn(&new_a_shape), ue)
                .map_err(|_| NyError::InvalidSpec("Resize coeff err reshape".to_string()))?;
            out.set_coeff_err(lower_err, upper_err);
        }
        Ok(out)
    }
}

impl BoundPropagation for ResizeLayer {
    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        ResizeLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }

    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let lower = self.upsample_last_two(input.lower())?;
        let upper = self.upsample_last_two(input.upper())?;
        BoundedTensor::new(lower, upper)
    }

    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedConfiguration(
            "Resize CROWN backward requires pre-activation bounds".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn make_linear_bounds(values: Vec<f32>) -> LinearBounds {
        LinearBounds::new_or_conservative(
            Array2::from_shape_vec((1, values.len()), values.clone()).unwrap(),
            ndarray::Array1::from_vec(vec![0.0_f32]),
            Array2::from_shape_vec((1, values.len()), values).unwrap(),
            ndarray::Array1::from_vec(vec![0.0_f32]),
        )
        .unwrap()
    }

    /// #vnncomp-aw-soundness self-audit regression: nearest Resize backward scatter-adds
    /// scale_h*scale_w DUPLICATE output coeffs into one input cell; under f32 cancellation a
    /// stored coeff is tighter than the true value, so it MUST carry a certified error.
    /// 2x2 upsample of a 1x1x1 input → 4 outputs map to the 1 cell; coeffs [2^24,1,1,-2^24]
    /// f32-accumulate to 0 while the true coeff is 2.
    #[ntest::timeout(5000)]
    #[test]
    fn resize_crown_backward_carries_cancellation_coeff_error() {
        let two24 = 16_777_216.0_f32; // 2^24
        let layer = ResizeLayer::new(2, 2);
        let pre = BoundedTensor::new(
            ArrayD::zeros(IxDyn(&[1, 1, 1])),
            ArrayD::zeros(IxDyn(&[1, 1, 1])),
        )
        .unwrap();
        let bounds = make_linear_bounds(vec![two24, 1.0, 1.0, -two24]);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
        assert_eq!(result.lower_a.ncols(), 1);
        let stored = result.lower_a[[0, 0]] as f64;
        let err = result
            .lower_a_err()
            .expect("resize duplicate-fan-in backward must carry a certified coeff error")[[0, 0]]
            as f64;
        let true_coeff = 2.0_f64; // 2^24 + 1 + 1 - 2^24
        assert!(
            stored < true_coeff,
            "f32 scatter-add drops the units: stored {stored}"
        );
        assert!(
            stored + err >= true_coeff,
            "certified err must enclose true: stored {stored} + err {err} < {true_coeff}"
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn resize_ibp_duplicates_spatial_cells() {
        let layer = ResizeLayer::new(2, 2);
        let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.5, 2.5, 3.5, 4.5]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        assert_eq!(output.shape(), &[1, 4, 4]);
        assert_eq!(
            output.lower().as_slice().unwrap(),
            &[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0]
        );
        assert_eq!(
            output.upper().as_slice().unwrap(),
            &[1.5, 1.5, 2.5, 2.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5, 3.5, 3.5, 4.5, 4.5]
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn resize_crown_backward_sums_upsampled_blocks() {
        let layer = ResizeLayer::new(2, 2);
        let pre = BoundedTensor::new(
            ArrayD::zeros(IxDyn(&[1, 2, 2])),
            ArrayD::zeros(IxDyn(&[1, 2, 2])),
        )
        .unwrap();
        let bounds = make_linear_bounds((1..=16).map(|v| v as f32).collect());

        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
        assert_eq!(result.lower_a().shape(), &[1, 4]);
        assert_eq!(result.upper_a().shape(), &[1, 4]);
        assert_eq!(
            result.lower_a().as_slice().unwrap(),
            &[14.0, 22.0, 46.0, 54.0]
        );
        assert_eq!(
            result.upper_a().as_slice().unwrap(),
            &[14.0, 22.0, 46.0, 54.0]
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn resize_batched_crown_reduces_flattened_columns() {
        let layer = ResizeLayer::new(2, 2);
        let pre = BoundedTensor::new(
            ArrayD::zeros(IxDyn(&[1, 2, 2])),
            ArrayD::zeros(IxDyn(&[1, 2, 2])),
        )
        .unwrap();
        let values = (1..=16).map(|v| v as f32).collect::<Vec<_>>();
        let bounds = BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(&[1, 16]), values.clone()).unwrap(),
            ArrayD::zeros(IxDyn(&[1])),
            ArrayD::from_shape_vec(IxDyn(&[1, 16]), values).unwrap(),
            ArrayD::zeros(IxDyn(&[1])),
            vec![16],
            vec![1],
        )
        .unwrap();

        let result = layer.propagate_linear_batched(&bounds, &pre).unwrap();
        assert_eq!(result.lower_a().shape(), &[1, 4]);
        assert_eq!(result.upper_a().shape(), &[1, 4]);
        assert_eq!(
            result.lower_a().as_slice().unwrap(),
            &[14.0, 22.0, 46.0, 54.0]
        );
        assert_eq!(
            result.upper_a().as_slice().unwrap(),
            &[14.0, 22.0, 46.0, 54.0]
        );
    }
}
