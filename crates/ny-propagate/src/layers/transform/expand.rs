// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Narrow ONNX Expand lowering for activation-path `[... ,1] -> [... ,T]`.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_up_f32, BoundedTensor, RepairStrategy};

use crate::{BatchedLinearBounds, LinearBounds};

/// Expand a singleton last axis to match a live reference tensor's last axis.
///
/// This is the narrow runtime contract needed for the avoice speaker encoder's
/// attentive statistics pooling path:
///
/// - source: `[prefix..., 1]`
/// - reference: `[prefix..., T]`
/// - output: `[prefix..., T]`
///
/// The output depends only on the source values. The reference tensor is used
/// solely to supply the runtime width `T`.
#[derive(Debug, Clone, Default)]
pub struct ExpandLikeLastAxisLayer;

impl ExpandLikeLastAxisLayer {
    /// Create a new narrow Expand layer.
    pub fn new() -> Self {
        Self
    }

    fn validate_contract(
        &self,
        source_shape: &[usize],
        reference_shape: &[usize],
        label: &str,
    ) -> Result<usize> {
        if source_shape.is_empty() || reference_shape.is_empty() {
            return Err(NyError::InvalidSpec(format!(
                "{label}: ExpandLikeLastAxis requires non-empty source/reference shapes"
            )));
        }
        if source_shape.len() != reference_shape.len() {
            return Err(NyError::ShapeMismatch {
                expected: source_shape.to_vec(),
                got: reference_shape.to_vec(),
            });
        }
        let rank = source_shape.len();
        if source_shape[..rank - 1] != reference_shape[..rank - 1] {
            return Err(NyError::ShapeMismatch {
                expected: source_shape[..rank - 1].to_vec(),
                got: reference_shape[..rank - 1].to_vec(),
            });
        }
        if source_shape[rank - 1] != 1 {
            return Err(NyError::UnsupportedConfiguration(format!(
                "{label}: ExpandLikeLastAxis requires source last axis = 1, got {:?}",
                source_shape
            )));
        }
        if reference_shape[rank - 1] == 0 {
            return Err(NyError::InvalidSpec(format!(
                "{label}: reference last axis must be non-zero, got {:?}",
                reference_shape
            )));
        }
        Ok(reference_shape[rank - 1])
    }

    /// IBP through `Expand(source, shape(reference))`.
    pub fn propagate_ibp_binary(
        &self,
        source: &BoundedTensor,
        reference: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        self.validate_contract(source.shape(), reference.shape(), "IBP")?;

        let out_lower = source
            .lower()
            .broadcast(IxDyn(reference.shape()))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: reference.shape().to_vec(),
                got: source.shape().to_vec(),
            })?
            .to_owned();
        let out_upper = source
            .upper()
            .broadcast(IxDyn(reference.shape()))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: reference.shape().to_vec(),
                got: source.shape().to_vec(),
            })?
            .to_owned();

        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    /// Dense CROWN backward split.
    ///
    /// The reference input contributes zero coefficients because the output does not
    /// depend on its values, only on its runtime shape.
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
        source: &BoundedTensor,
        reference: &BoundedTensor,
    ) -> Result<(LinearBounds, LinearBounds)> {
        let target_width =
            self.validate_contract(source.shape(), reference.shape(), "CROWN backward")?;
        let input_size = source.len();
        let output_size = checked_shape_product(reference.shape()).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ExpandLikeLastAxis: output shape product overflows usize: {:?}",
                reference.shape()
            ))
        })?;
        if output_size != input_size * target_width {
            return Err(NyError::ShapeMismatch {
                expected: vec![input_size * target_width],
                got: vec![output_size],
            });
        }

        let num_outputs = bounds.num_outputs();
        if bounds.num_inputs() != output_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_outputs, output_size],
                got: vec![num_outputs, bounds.num_inputs()],
            });
        }

        let mut lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut upper_a = Array2::<f32>::zeros((num_outputs, input_size));
        // Certified scatter-add error (#vnncomp-aw-soundness): broadcasting sums
        // `target_width` incoming coeffs into ONE source cell in round-to-nearest f32,
        // a depth-`target_width` accumulation (fan-in can be large) that rounds — the
        // gather-class duplicate-fan-in bug. Carry gamma_{target_width}*S + prop (S =
        // abs-sum over the summed outputs, cancellation-safe; prop = re-summed incoming
        // err), rounded OUTWARD, via new_or_conservative_with_err.
        let in_lower_err = bounds.lower_a_err();
        let in_upper_err = bounds.upper_a_err();
        let mut s_lower = Array2::<f64>::zeros((num_outputs, input_size));
        let mut s_upper = Array2::<f64>::zeros((num_outputs, input_size));
        let mut p_lower = Array2::<f64>::zeros((num_outputs, input_size));
        let mut p_upper = Array2::<f64>::zeros((num_outputs, input_size));

        for input_idx in 0..input_size {
            let base = input_idx * target_width;
            for output_offset in 0..target_width {
                let output_idx = base + output_offset;
                for row in 0..num_outputs {
                    lower_a[[row, input_idx]] += bounds.lower_a()[[row, output_idx]];
                    upper_a[[row, input_idx]] += bounds.upper_a()[[row, output_idx]];
                    s_lower[[row, input_idx]] += (bounds.lower_a()[[row, output_idx]] as f64).abs();
                    s_upper[[row, input_idx]] += (bounds.upper_a()[[row, output_idx]] as f64).abs();
                    if let Some(e) = in_lower_err {
                        p_lower[[row, input_idx]] += (e[[row, output_idx]] as f64).abs();
                    }
                    if let Some(e) = in_upper_err {
                        p_upper[[row, input_idx]] += (e[[row, output_idx]] as f64).abs();
                    }
                }
            }
        }

        let source_bounds = if target_width >= 2 || in_lower_err.is_some() || in_upper_err.is_some()
        {
            let gamma = if target_width >= 2 {
                crate::layers::linear::crown_single_gamma_n_f32(target_width)
            } else {
                0.0
            };
            let lower_err = ndarray::Zip::from(&s_lower)
                .and(&p_lower)
                .map_collect(|&s, &p| next_up_f32((gamma * s + p) as f32));
            let upper_err = ndarray::Zip::from(&s_upper)
                .and(&p_upper)
                .map_collect(|&s, &p| next_up_f32((gamma * s + p) as f32));
            LinearBounds::new_or_conservative_with_err(
                lower_a,
                bounds.lower_b().clone(),
                upper_a,
                bounds.upper_b().clone(),
                lower_err,
                upper_err,
            )?
        } else {
            LinearBounds::new_or_conservative(
                lower_a,
                bounds.lower_b().clone(),
                upper_a,
                bounds.upper_b().clone(),
            )?
        };

        let zero_bias = Array1::<f32>::zeros(num_outputs);
        let reference_bounds = LinearBounds::new_or_conservative(
            Array2::<f32>::zeros((num_outputs, reference.len())),
            zero_bias.clone(),
            Array2::<f32>::zeros((num_outputs, reference.len())),
            zero_bias,
        )?;

        Ok((source_bounds, reference_bounds))
    }

    /// Batched CROWN backward split.
    ///
    /// Supports both standard batched A-matrices (`[..., out_dim, in_dim]`,
    /// where `in_dim` is just the last logical axis) and flattened-column
    /// bounds used by a few graph paths (`[..., out_dim, flat_input]`).
    pub fn propagate_linear_batched_binary(
        &self,
        bounds: &BatchedLinearBounds,
        source: &BoundedTensor,
        reference: &BoundedTensor,
    ) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
        let target_width =
            self.validate_contract(source.shape(), reference.shape(), "batched CROWN backward")?;
        let flat_input_size = source.len();
        let flat_output_size = checked_shape_product(reference.shape()).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ExpandLikeLastAxis batched CROWN: output shape product overflows usize: {:?}",
                reference.shape()
            ))
        })?;
        if flat_output_size != flat_input_size * target_width {
            return Err(NyError::ShapeMismatch {
                expected: vec![flat_input_size * target_width],
                got: vec![flat_output_size],
            });
        }
        let source_feature_dim = source.shape().last().copied().unwrap_or(1);
        let reference_feature_dim = reference.shape().last().copied().unwrap_or(1);

        let a_shape = bounds.lower_a().shape();
        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "ExpandLikeLastAxis batched CROWN requires A matrices with ndim >= 2".to_string(),
            ));
        }
        let in_dim = a_shape[a_shape.len() - 1];
        let (input_size, output_size, reference_input_dim) = if in_dim == reference_feature_dim {
            (
                source_feature_dim,
                reference_feature_dim,
                reference_feature_dim,
            )
        } else if in_dim == flat_output_size {
            (flat_input_size, flat_output_size, reference.len())
        } else {
            return Err(NyError::InvalidSpec(format!(
                "ExpandLikeLastAxis batched CROWN expects last A dim {} (per-position) or {} (flattened), got {}",
                reference_feature_dim, flat_output_size, in_dim
            )));
        };

        let mut new_a_shape = a_shape.to_vec();
        new_a_shape[a_shape.len() - 1] = input_size;
        let mut lower_a = ArrayD::<f32>::zeros(IxDyn(&new_a_shape));
        let mut upper_a = ArrayD::<f32>::zeros(IxDyn(&new_a_shape));

        let outer_size = checked_shape_product(&a_shape[..a_shape.len() - 1]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ExpandLikeLastAxis batched CROWN: outer shape product overflows: {:?}",
                &a_shape[..a_shape.len() - 1],
            ))
        })?;
        let src_lower = bounds.lower_a().as_slice_memory_order().ok_or_else(|| {
            NyError::InvalidSpec(
                "ExpandLikeLastAxis batched CROWN: lower_a is not contiguous".to_string(),
            )
        })?;
        let src_upper = bounds.upper_a().as_slice_memory_order().ok_or_else(|| {
            NyError::InvalidSpec(
                "ExpandLikeLastAxis batched CROWN: upper_a is not contiguous".to_string(),
            )
        })?;
        let dst_lower = lower_a.as_slice_memory_order_mut().ok_or_else(|| {
            NyError::InvalidSpec(
                "ExpandLikeLastAxis batched CROWN: new lower_a is not contiguous".to_string(),
            )
        })?;
        let dst_upper = upper_a.as_slice_memory_order_mut().ok_or_else(|| {
            NyError::InvalidSpec(
                "ExpandLikeLastAxis batched CROWN: new upper_a is not contiguous".to_string(),
            )
        })?;

        // Certified scatter-add error (#vnncomp-aw-soundness), same as the dense path:
        // each source cell sums `target_width` incoming coeffs in f32, carry
        // gamma_{target_width}*S + prop (abs-sums over the summed outputs) outward.
        let dst_len = dst_lower.len();
        let src_lower_err = bounds
            .lower_a_err
            .as_ref()
            .and_then(|e| e.as_slice_memory_order());
        let src_upper_err = bounds
            .upper_a_err
            .as_ref()
            .and_then(|e| e.as_slice_memory_order());
        let mut s_lower_flat = vec![0.0f64; dst_len];
        let mut s_upper_flat = vec![0.0f64; dst_len];
        let mut p_lower_flat = vec![0.0f64; dst_len];
        let mut p_upper_flat = vec![0.0f64; dst_len];

        for outer in 0..outer_size {
            let src_base = outer * output_size;
            let dst_base = outer * input_size;
            for input_idx in 0..input_size {
                let source_offset = src_base + input_idx * target_width;
                let target_slot = dst_base + input_idx;
                for output_offset in 0..target_width {
                    let src_idx = source_offset + output_offset;
                    dst_lower[target_slot] += src_lower[src_idx];
                    dst_upper[target_slot] += src_upper[src_idx];
                    s_lower_flat[target_slot] += (src_lower[src_idx] as f64).abs();
                    s_upper_flat[target_slot] += (src_upper[src_idx] as f64).abs();
                    if let Some(e) = src_lower_err {
                        p_lower_flat[target_slot] += (e[src_idx] as f64).abs();
                    }
                    if let Some(e) = src_upper_err {
                        p_upper_flat[target_slot] += (e[src_idx] as f64).abs();
                    }
                }
            }
        }

        let mut source_bounds = BatchedLinearBounds::new_or_conservative(
            lower_a,
            bounds.lower_b().clone(),
            upper_a,
            bounds.upper_b().clone(),
            source.shape().to_vec(),
            bounds.output_shape().to_vec(),
        )?;
        if target_width >= 2 || src_lower_err.is_some() || src_upper_err.is_some() {
            let gamma = if target_width >= 2 {
                crate::layers::linear::crown_single_gamma_n_f32(target_width)
            } else {
                0.0
            };
            let le: Vec<f32> = s_lower_flat
                .iter()
                .zip(&p_lower_flat)
                .map(|(&s, &p)| next_up_f32((gamma * s + p) as f32))
                .collect();
            let ue: Vec<f32> = s_upper_flat
                .iter()
                .zip(&p_upper_flat)
                .map(|(&s, &p)| next_up_f32((gamma * s + p) as f32))
                .collect();
            let lower_err = ArrayD::from_shape_vec(IxDyn(&new_a_shape), le)
                .map_err(|_| NyError::InvalidSpec("Expand coeff err reshape".to_string()))?;
            let upper_err = ArrayD::from_shape_vec(IxDyn(&new_a_shape), ue)
                .map_err(|_| NyError::InvalidSpec("Expand coeff err reshape".to_string()))?;
            source_bounds.set_coeff_err(lower_err, upper_err);
        }

        let bias_shape = bounds.lower_b().raw_dim();
        let mut reference_a_shape = a_shape.to_vec();
        reference_a_shape[a_shape.len() - 1] = reference_input_dim;
        let reference_bounds = BatchedLinearBounds::new_or_conservative(
            ArrayD::<f32>::zeros(IxDyn(&reference_a_shape)),
            ArrayD::<f32>::zeros(bias_shape.clone()),
            ArrayD::<f32>::zeros(IxDyn(&reference_a_shape)),
            ArrayD::<f32>::zeros(bias_shape),
            reference.shape().to_vec(),
            bounds.output_shape().to_vec(),
        )?;

        Ok((source_bounds, reference_bounds))
    }
}
