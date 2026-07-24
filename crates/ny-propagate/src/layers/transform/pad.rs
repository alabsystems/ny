// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pad layer: explicit ONNX padding over unbatched activation tensors.

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;

use super::super::common::BoundPropagation;
use crate::{contiguous_flat_slice, contiguous_flat_slice_mut, BatchedLinearBounds, LinearBounds};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PadMode {
    Constant(f32),
    Reflect,
}

#[derive(Debug, Clone, PartialEq)]
enum PadSource {
    Input(usize),
    Constant(f32),
}

/// Explicit pad over each tensor axis.
#[derive(Debug, Clone)]
pub struct PadLayer {
    /// Per-axis `(pad_before, pad_after)` after batch stripping.
    pub pads: Vec<(usize, usize)>,
    pub mode: PadMode,
}

impl PadLayer {
    pub fn new(pads: Vec<(usize, usize)>, mode: PadMode) -> Self {
        Self { pads, mode }
    }

    fn validate_input_shape(&self, input_shape: &[usize]) -> Result<()> {
        if input_shape.len() != self.pads.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.pads.len()],
                got: vec![input_shape.len()],
            });
        }
        if matches!(self.mode, PadMode::Reflect) {
            for (axis, (&dim, &(pad_before, pad_after))) in
                input_shape.iter().zip(self.pads.iter()).enumerate()
            {
                if dim == 0 {
                    return Err(NyError::InvalidSpec(format!(
                        "Pad reflect mode requires non-empty axis {axis}, got shape {:?}",
                        input_shape
                    )));
                }
                if (pad_before > 0 || pad_after > 0) && dim < 2 {
                    return Err(NyError::UnsupportedConfiguration(format!(
                        "Pad reflect mode requires axis {axis} size >= 2 when padding is non-zero, got {}",
                        dim
                    )));
                }
                if pad_before >= dim || pad_after >= dim {
                    return Err(NyError::UnsupportedConfiguration(format!(
                        "Pad reflect mode requires pad < dim on axis {axis}, got pads ({pad_before}, {pad_after}) for dim {}",
                        dim
                    )));
                }
            }
        }
        Ok(())
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        self.validate_input_shape(input_shape)?;
        input_shape
            .iter()
            .zip(self.pads.iter())
            .map(|(&dim, &(pad_before, pad_after))| {
                dim.checked_add(pad_before)
                    .and_then(|value| value.checked_add(pad_after))
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "Pad output shape overflow for dim {} with pads ({}, {})",
                            dim, pad_before, pad_after
                        ))
                    })
            })
            .collect()
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
                    NyError::InvalidSpec(format!("Pad stride overflow for shape {:?}", shape))
                })?;
        }
        Ok(strides)
    }

    fn reflect_index(coord: usize, pad_before: usize, dim: usize) -> Result<usize> {
        let shifted = coord as isize - pad_before as isize;
        if shifted < 0 {
            return usize::try_from(-shifted).map_err(|_| {
                NyError::InvalidSpec("Pad reflect index overflow on left border".to_string())
            });
        }
        if shifted >= dim as isize {
            let mirrored = 2 * dim as isize - shifted - 2;
            return usize::try_from(mirrored).map_err(|_| {
                NyError::InvalidSpec("Pad reflect index overflow on right border".to_string())
            });
        }
        Ok(shifted as usize)
    }

    fn map_axis_coord(&self, axis: usize, coord: usize, dim: usize) -> Result<PadSource> {
        let (pad_before, _) = self.pads[axis];
        if coord >= pad_before && coord < pad_before + dim {
            return Ok(PadSource::Input(coord - pad_before));
        }

        match self.mode {
            PadMode::Constant(value) => Ok(PadSource::Constant(value)),
            PadMode::Reflect => Ok(PadSource::Input(Self::reflect_index(
                coord, pad_before, dim,
            )?)),
        }
    }

    fn build_output_sources(&self, input_shape: &[usize]) -> Result<Vec<PadSource>> {
        let output_shape = self.output_shape(input_shape)?;
        let output_size = checked_shape_product(&output_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Pad output shape product overflows usize: {:?}",
                output_shape
            ))
        })?;
        let input_strides = Self::compute_strides(input_shape)?;
        let output_strides = Self::compute_strides(&output_shape)?;

        let mut sources = Vec::with_capacity(output_size);
        for out_flat in 0..output_size {
            let mut out_index = vec![0usize; output_shape.len()];
            let mut remainder = out_flat;
            for (axis, &stride) in output_strides.iter().enumerate() {
                out_index[axis] = remainder / stride;
                remainder %= stride;
            }

            let mut input_flat = 0usize;
            let mut constant = None;
            for (axis, (&coord, &dim)) in out_index.iter().zip(input_shape.iter()).enumerate() {
                match self.map_axis_coord(axis, coord, dim)? {
                    PadSource::Input(mapped) => {
                        input_flat = input_flat
                            .checked_add(mapped.checked_mul(input_strides[axis]).ok_or_else(
                                || {
                                    NyError::InvalidSpec(format!(
                                        "Pad input index overflow on axis {}",
                                        axis
                                    ))
                                },
                            )?)
                            .ok_or_else(|| {
                                NyError::InvalidSpec(
                                    "Pad flattened input index overflow".to_string(),
                                )
                            })?;
                    }
                    PadSource::Constant(value) => {
                        constant = Some(value);
                        break;
                    }
                }
            }

            if let Some(value) = constant {
                sources.push(PadSource::Constant(value));
            } else {
                sources.push(PadSource::Input(input_flat));
            }
        }

        Ok(sources)
    }

    fn pad_array(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let input_shape = input.shape().to_vec();
        let output_shape = self.output_shape(&input_shape)?;
        let sources = self.build_output_sources(&input_shape)?;
        let flat_input = contiguous_flat_slice(input);

        let output = sources
            .iter()
            .map(|source| match source {
                PadSource::Input(in_flat) => flat_input[*in_flat],
                PadSource::Constant(value) => *value,
            })
            .collect::<Vec<_>>();

        ArrayD::from_shape_vec(IxDyn(&output_shape), output)
            .map_err(|e| NyError::InvalidSpec(format!("Pad output reshape failed: {}", e)))
    }

    fn propagate_linear_for_shape(
        &self,
        bounds: &LinearBounds,
        input_shape: &[usize],
    ) -> Result<LinearBounds> {
        let input_size = checked_shape_product(input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Pad input shape product overflows usize: {:?}",
                input_shape
            ))
        })?;
        let sources = self.build_output_sources(input_shape)?;
        let output_size = sources.len();

        if bounds.num_inputs() != output_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![output_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_lower_b = bounds.lower_b().clone();
        let mut new_upper_b = bounds.upper_b().clone();

        // Reflect mode sends MULTIPLE output columns to the SAME border input column, so the
        // backward f32-accumulates a duplicate-fan-in sum that rounds (gather-class). Constant
        // mode folds `coeff * value` into the bias in f32 (round-to-nearest, false-tight under
        // cancellation, and silently dropping incoming coeff err). Certify both OUTWARD: the
        // Input scatter via the shared gather_backward_coeff_err, the Constant fold via f64
        // accumulation + incoming-err fold + directed cast. (#vnncomp-aw-soundness self-audit.)
        let in_lo_err = bounds.lower_a_err();
        let in_up_err = bounds.upper_a_err();
        let mut in_to_outs: Vec<Vec<usize>> = vec![Vec::new(); input_size];
        let mut const_lo = vec![0.0f64; num_outputs];
        let mut const_up = vec![0.0f64; num_outputs];
        let mut const_lo_err = vec![0.0f64; num_outputs];
        let mut const_up_err = vec![0.0f64; num_outputs];
        for (out_flat, source) in sources.iter().enumerate() {
            match source {
                PadSource::Input(in_flat) => {
                    in_to_outs[*in_flat].push(out_flat);
                    for row in 0..num_outputs {
                        new_lower_a[[row, *in_flat]] += bounds.lower_a()[[row, out_flat]];
                        new_upper_a[[row, *in_flat]] += bounds.upper_a()[[row, out_flat]];
                    }
                }
                PadSource::Constant(value) => {
                    let v = *value as f64;
                    for row in 0..num_outputs {
                        const_lo[row] += (bounds.lower_a()[[row, out_flat]] as f64) * v;
                        const_up[row] += (bounds.upper_a()[[row, out_flat]] as f64) * v;
                        if let Some(e) = in_lo_err {
                            const_lo_err[row] += (e[[row, out_flat]] as f64).abs() * v.abs();
                        }
                        if let Some(e) = in_up_err {
                            const_up_err[row] += (e[[row, out_flat]] as f64).abs() * v.abs();
                        }
                    }
                }
            }
        }
        for row in 0..num_outputs {
            // Only touch the bias when there is an actual Constant contribution, so the common
            // pad-with-0 case (and exact folds) are not needlessly widened by the directed cast.
            if const_lo[row] != 0.0 || const_lo_err[row] != 0.0 {
                new_lower_b[row] = next_down_f32(
                    ((new_lower_b[row] as f64) + const_lo[row] - const_lo_err[row]) as f32,
                );
            }
            if const_up[row] != 0.0 || const_up_err[row] != 0.0 {
                new_upper_b[row] = next_up_f32(
                    ((new_upper_b[row] as f64) + const_up[row] + const_up_err[row]) as f32,
                );
            }
        }

        let has_duplicate = in_to_outs.iter().any(|outs| outs.len() >= 2);
        if has_duplicate || in_lo_err.is_some() || in_up_err.is_some() {
            let (lower_err, upper_err) = super::gather::gather_backward_coeff_err(
                bounds.lower_a(),
                bounds.upper_a(),
                in_lo_err,
                in_up_err,
                input_size,
                &in_to_outs,
            );
            return LinearBounds::new_or_conservative_with_err(
                new_lower_a,
                new_lower_b,
                new_upper_a,
                new_upper_b,
                lower_err,
                upper_err,
            );
        }

        LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
    }

    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        self.propagate_linear_for_shape(bounds, pre_activation.shape())
    }

    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let input_shape = pre_activation.shape().to_vec();
        let input_size = checked_shape_product(&input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Pad batched CROWN input shape product overflows usize: {:?}",
                input_shape
            ))
        })?;
        let sources = self.build_output_sources(&input_shape)?;
        let output_size = sources.len();

        let a_shape = bounds.lower_a().shape();
        let a_ndim = a_shape.len();
        if a_ndim < 2 {
            return Err(NyError::InvalidSpec(
                "Pad batched CROWN: A matrices must have at least 2 dimensions".to_string(),
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
        let mut new_lower_b = bounds.lower_b().clone();
        let mut new_upper_b = bounds.upper_b().clone();

        let outer_size = checked_shape_product(&a_shape[..a_ndim - 1]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Pad batched CROWN outer shape product overflows: {:?}",
                &a_shape[..a_ndim - 1],
            ))
        })?;

        // Certified scatter-add + constant-fold error (#vnncomp-aw-soundness), batched twin of
        // the dense path. Reflect duplicate-fan-in: gamma_k*S + prop (defensive — Pad is a
        // batched carrier, so incoming err is usually stripped and the carrier adds it).
        // Constant fold: f64 accumulate + directed cast + defensive incoming-err fold.
        let flat_lower = contiguous_flat_slice(bounds.lower_a());
        let flat_upper = contiguous_flat_slice(bounds.upper_a());
        let flat_lo_err = bounds
            .lower_a_err
            .as_ref()
            .and_then(|e| e.as_slice_memory_order());
        let flat_up_err = bounds
            .upper_a_err
            .as_ref()
            .and_then(|e| e.as_slice_memory_order());
        let mut fan_in = vec![0usize; input_size];
        for source in sources.iter() {
            if let PadSource::Input(in_col) = source {
                fan_in[*in_col] += 1;
            }
        }
        let new_len = outer_size * input_size;
        let mut s_lower = vec![0.0f64; new_len];
        let mut s_upper = vec![0.0f64; new_len];
        let mut p_lower = vec![0.0f64; new_len];
        let mut p_upper = vec![0.0f64; new_len];
        let mut const_lo = vec![0.0f64; outer_size];
        let mut const_up = vec![0.0f64; outer_size];
        let mut const_lo_err = vec![0.0f64; outer_size];
        let mut const_up_err = vec![0.0f64; outer_size];

        {
            let new_lower_flat = contiguous_flat_slice_mut(&mut new_lower_a)?;
            let new_upper_flat = contiguous_flat_slice_mut(&mut new_upper_a)?;
            for row in 0..outer_size {
                let old_base = row * output_size;
                let new_base = row * input_size;
                for (out_col, source) in sources.iter().enumerate() {
                    match source {
                        PadSource::Input(in_col) => {
                            let slot = new_base + *in_col;
                            new_lower_flat[slot] += flat_lower[old_base + out_col];
                            new_upper_flat[slot] += flat_upper[old_base + out_col];
                            s_lower[slot] += (flat_lower[old_base + out_col] as f64).abs();
                            s_upper[slot] += (flat_upper[old_base + out_col] as f64).abs();
                            if let Some(e) = flat_lo_err {
                                p_lower[slot] += (e[old_base + out_col] as f64).abs();
                            }
                            if let Some(e) = flat_up_err {
                                p_upper[slot] += (e[old_base + out_col] as f64).abs();
                            }
                        }
                        PadSource::Constant(value) => {
                            let v = *value as f64;
                            const_lo[row] += (flat_lower[old_base + out_col] as f64) * v;
                            const_up[row] += (flat_upper[old_base + out_col] as f64) * v;
                            if let Some(e) = flat_lo_err {
                                const_lo_err[row] += (e[old_base + out_col] as f64).abs() * v.abs();
                            }
                            if let Some(e) = flat_up_err {
                                const_up_err[row] += (e[old_base + out_col] as f64).abs() * v.abs();
                            }
                        }
                    }
                }
            }
            let new_lower_b_flat = contiguous_flat_slice_mut(&mut new_lower_b)?;
            let new_upper_b_flat = contiguous_flat_slice_mut(&mut new_upper_b)?;
            for row in 0..outer_size {
                if const_lo[row] != 0.0 || const_lo_err[row] != 0.0 {
                    new_lower_b_flat[row] = next_down_f32(
                        ((new_lower_b_flat[row] as f64) + const_lo[row] - const_lo_err[row]) as f32,
                    );
                }
                if const_up[row] != 0.0 || const_up_err[row] != 0.0 {
                    new_upper_b_flat[row] = next_up_f32(
                        ((new_upper_b_flat[row] as f64) + const_up[row] + const_up_err[row]) as f32,
                    );
                }
            }
        }

        let mut out = BatchedLinearBounds::new_or_conservative(
            new_lower_a,
            new_lower_b,
            new_upper_a,
            new_upper_b,
            input_shape,
            bounds.output_shape().to_vec(),
        )?;
        if fan_in.iter().any(|&k| k >= 2) || flat_lo_err.is_some() || flat_up_err.is_some() {
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
                .map_err(|_| NyError::InvalidSpec("Pad coeff err reshape".to_string()))?;
            let upper_err = ArrayD::from_shape_vec(IxDyn(&new_a_shape), ue)
                .map_err(|_| NyError::InvalidSpec("Pad coeff err reshape".to_string()))?;
            out.set_coeff_err(lower_err, upper_err);
        }
        Ok(out)
    }
}

impl BoundPropagation for PadLayer {
    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        PadLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }

    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let lower = self.pad_array(input.lower())?;
        let upper = self.pad_array(input.upper())?;
        BoundedTensor::new(lower, upper)
    }

    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedConfiguration(
            "Pad CROWN backward requires pre-activation bounds".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_linear_bounds(values: Vec<f32>) -> LinearBounds {
        LinearBounds::new_or_conservative(
            Array2::from_shape_vec((1, values.len()), values.clone()).unwrap(),
            Array1::from_vec(vec![0.0]),
            Array2::from_shape_vec((1, values.len()), values).unwrap(),
            Array1::from_vec(vec![0.0]),
        )
        .unwrap()
    }

    #[ntest::timeout(5000)]
    #[test]
    fn reflect_pad_ibp_mirrors_last_axis() {
        let layer = PadLayer::new(vec![(0, 0), (2, 2)], PadMode::Reflect);
        let lower = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.5, 2.5, 3.5, 4.5]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        assert_eq!(output.shape(), &[1, 8]);
        assert_eq!(
            output.lower().as_slice().unwrap(),
            &[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
        assert_eq!(
            output.upper().as_slice().unwrap(),
            &[3.5, 2.5, 1.5, 2.5, 3.5, 4.5, 3.5, 2.5]
        );
    }

    /// #vnncomp-aw-soundness self-audit regression: Reflect-mode Pad backward scatter-adds
    /// DUPLICATE output coeffs into interior input cells (fan-in > 1), so under f32 cancellation
    /// a stored coeff is tighter than the true value and MUST carry a certified error. Pre-fix
    /// Pad was an `is_exact_linear_coeff_err_carrier` with no fresh err (false-proof).
    #[ntest::timeout(5000)]
    #[test]
    fn pad_reflect_crown_carries_duplicate_fanin_coeff_error() {
        // input [3], pad (2,2) reflect -> 7 outputs, interior cells get fan-in >= 2.
        let layer = PadLayer::new(vec![(2, 2)], PadMode::Reflect);
        let pre =
            BoundedTensor::new(ArrayD::zeros(IxDyn(&[3])), ArrayD::zeros(IxDyn(&[3]))).unwrap();
        // Large-magnitude coeffs so an f32 fan-in sum drops low-order terms.
        let coeffs = vec![1.0, 1.0, 1e8, 1.0, -1e8, 1.0, 1.0];
        let bounds = make_linear_bounds(coeffs);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
        let err = result
            .lower_a_err()
            .expect("reflect-pad duplicate fan-in backward must carry a certified coeff error");
        let max_err = err.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            max_err >= 1.0,
            "fresh gamma_k*S err must cover the dropped low-order coeff, max err = {max_err}"
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn constant_pad_crown_backward_adds_bias_for_border_cells() {
        let layer = PadLayer::new(vec![(1, 1)], PadMode::Constant(5.0));
        let pre =
            BoundedTensor::new(ArrayD::zeros(IxDyn(&[2])), ArrayD::zeros(IxDyn(&[2]))).unwrap();
        let bounds = make_linear_bounds(vec![1.0, 2.0, 3.0, 4.0]);

        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
        assert_eq!(result.lower_a().shape(), &[1, 2]);
        assert_eq!(result.lower_a().as_slice().unwrap(), &[2.0, 3.0]);
        assert_eq!(result.upper_a().as_slice().unwrap(), &[2.0, 3.0]);
        // Border bias = (coeff[0] + coeff[3]) * 5 = (1+4)*5 = 25, now folded via f64 + directed
        // cast so it ENCLOSES 25 (sound widening of the constant fold, #vnncomp-aw-soundness).
        assert!(
            result.lower_b()[[0]] <= 25.0 && (result.lower_b()[[0]] - 25.0).abs() < 1e-4,
            "lower bias must enclose 25, got {}",
            result.lower_b()[[0]]
        );
        assert!(
            result.upper_b()[[0]] >= 25.0 && (result.upper_b()[[0]] - 25.0).abs() < 1e-4,
            "upper bias must enclose 25, got {}",
            result.upper_b()[[0]]
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn reflect_pad_batched_crown_sums_mirrored_columns() {
        let layer = PadLayer::new(vec![(0, 0), (1, 1)], PadMode::Reflect);
        let pre = BoundedTensor::new(ArrayD::zeros(IxDyn(&[1, 3])), ArrayD::zeros(IxDyn(&[1, 3])))
            .unwrap();
        let lower_a =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let upper_a = lower_a.clone();
        let lower_b = ArrayD::zeros(IxDyn(&[1, 1]));
        let upper_b = ArrayD::zeros(IxDyn(&[1, 1]));
        let bounds = BatchedLinearBounds::new_or_conservative(
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            vec![1, 5],
            vec![1, 1],
        )
        .unwrap();

        let result = layer.propagate_linear_batched(&bounds, &pre).unwrap();
        assert_eq!(result.lower_a().shape(), &[1, 1, 3]);
        assert_eq!(result.lower_a().as_slice().unwrap(), &[2.0, 9.0, 4.0]);
        assert_eq!(result.upper_a().as_slice().unwrap(), &[2.0, 9.0, 4.0]);
    }
}
