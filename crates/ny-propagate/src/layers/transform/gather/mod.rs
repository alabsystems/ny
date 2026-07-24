// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Gather layer: index/select elements along an axis using an indices tensor.

use ndarray::{Array2, ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;

use super::super::common::BoundPropagation;
use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::LinearBounds;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GatherMode {
    Standard,
    RuntimeLastAxisLen,
}

/// Gather layer: index/select elements along an axis using an indices tensor.
///
/// ONNX semantics: output shape replaces the axis dimension with indices shape.
/// For example, data shape [a, b, c], axis=1, indices shape [i, j] -> output [a, i, j, c].
///
/// If indices are not available (dynamic), IBP falls back to conservative
/// min/max bounds across the gather axis.
#[derive(Debug, Clone)]
pub struct GatherLayer {
    /// Axis along which to gather (supports negative indexing).
    axis: i64,
    /// Whether this is a normal data gather or a shape-query gather.
    mode: GatherMode,
    /// Optional constant indices tensor.
    indices: Option<ArrayD<i64>>,
    /// Shape of the indices tensor (used when indices are dynamic).
    indices_shape: Vec<usize>,
    /// Input shape (required for CROWN backward propagation).
    input_shape: Option<Vec<usize>>,
}

impl GatherLayer {
    /// Create a new Gather layer.
    pub fn new(axis: i64, indices: Option<ArrayD<i64>>, indices_shape: Vec<usize>) -> Self {
        let indices_shape = if indices_shape.is_empty() {
            indices
                .as_ref()
                .map(|arr| arr.shape().to_vec())
                .unwrap_or_default()
        } else {
            indices_shape
        };
        Self {
            axis,
            mode: GatherMode::Standard,
            indices,
            indices_shape,
            input_shape: None,
        }
    }

    /// Create a narrow runtime shape query that returns the input's last-axis length.
    pub fn runtime_last_axis_len(indices_shape: Vec<usize>) -> Self {
        Self {
            axis: -1,
            mode: GatherMode::RuntimeLastAxisLen,
            indices: None,
            indices_shape,
            input_shape: None,
        }
    }

    pub fn is_runtime_last_axis_len(&self) -> bool {
        matches!(self.mode, GatherMode::RuntimeLastAxisLen)
    }

    /// The gather axis (raw, may be negative). In-crate introspection for the
    /// f64 cell evaluator and the cell-enumeration structural trigger.
    pub fn axis_raw(&self) -> i64 {
        self.axis
    }

    /// The embedded constant indices, when static.
    pub fn constant_indices(&self) -> Option<&ArrayD<i64>> {
        self.indices.as_ref()
    }

    /// Set the input shape for CROWN backward propagation.
    pub fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shape = Some(shape);
    }

    fn resolve_axis(&self, ndim: usize) -> Result<usize> {
        super::super::common::resolve_axis(self.axis, ndim, "Gather")
    }

    fn runtime_query_output_shape(&self) -> Vec<usize> {
        self.indices_shape.clone()
    }

    fn output_shape(&self, input_shape: &[usize], axis: usize) -> Vec<usize> {
        let mut output_shape =
            Vec::with_capacity(input_shape.len().saturating_sub(1) + self.indices_shape.len());
        output_shape.extend_from_slice(&input_shape[..axis]);
        output_shape.extend_from_slice(&self.indices_shape);
        output_shape.extend_from_slice(&input_shape[axis + 1..]);
        output_shape
    }

    fn normalize_indices(&self, indices: &[i64], axis_len: i64) -> Result<Vec<usize>> {
        let mut normalized = Vec::with_capacity(indices.len());
        for &idx in indices {
            let adj = if idx < 0 { axis_len + idx } else { idx };
            if adj < 0 || adj >= axis_len {
                return Err(NyError::InvalidSpec(format!(
                    "Gather index {} out of bounds for axis length {}",
                    idx, axis_len
                )));
            }
            // SAFETY(as usize): adj is i64, guard above ensures 0 <= adj < axis_len.
            normalized.push(adj as usize);
        }
        Ok(normalized)
    }

    fn gather_with_indices(
        &self,
        input: &ArrayD<f32>,
        axis: usize,
        indices: &ArrayD<i64>,
    ) -> Result<ArrayD<f32>> {
        let input_shape = input.shape();
        let axis_len = input_shape[axis] as i64;
        let indices_flat: Vec<i64> = indices.iter().copied().collect();

        if indices_flat.is_empty() {
            let output_shape = self.output_shape(input_shape, axis);
            return Ok(ArrayD::zeros(IxDyn(&output_shape)));
        }

        let indices_norm = self.normalize_indices(&indices_flat, axis_len)?;
        let indices_shape = indices.shape();

        if indices_shape.is_empty() {
            if indices_norm.len() != 1 {
                return Err(NyError::InvalidSpec(format!(
                    "Gather scalar indices expected 1 element, got {}",
                    indices_norm.len()
                )));
            }
            let index = indices_norm[0];
            return Ok(input.index_axis(Axis(axis), index).to_owned());
        }

        let selected = input.select(Axis(axis), &indices_norm);
        let mut output_shape = Vec::with_capacity(input_shape.len() - 1 + indices_shape.len());
        output_shape.extend_from_slice(&input_shape[..axis]);
        output_shape.extend_from_slice(indices_shape);
        output_shape.extend_from_slice(&input_shape[axis + 1..]);

        // ndarray::select on non-first axes produces non-standard layout.
        // Convert to standard (C-contiguous) layout before reshaping.
        selected
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order(IxDyn(&output_shape))
            .map_err(|e| NyError::InvalidSpec(format!("Gather reshape failed: {}", e)))
    }

    fn broadcast_reduced(
        &self,
        reduced: ArrayD<f32>,
        axis: usize,
        output_shape: &[usize],
    ) -> Result<ArrayD<f32>> {
        let mut expanded = reduced;
        for _ in 0..self.indices_shape.len() {
            expanded = expanded.insert_axis(Axis(axis));
        }
        expanded
            .broadcast(IxDyn(output_shape))
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Gather broadcast failed for output shape {:?}",
                    output_shape
                ))
            })
            .map(|view| view.to_owned())
    }
}

/// Certified coefficient error for the Gather backward scatter-add
/// (#vnncomp-aw-soundness) — the Gather analogue of Tile's `tile_backward_coeff_err`.
///
/// On DUPLICATE indices, `k = in_to_outs[i].len()` output columns scatter-add (`+=`)
/// into input column `i`. That f32 accumulation of `k` terms has a rounding error
/// bounded by the **f32** Higham growth factor `gamma_k^f32 ~= k*2^-24` (NOT the f64
/// factor, which would UNDER-count the real f32 error). The certified error per
/// `(row, i)` is `gamma_k*S + prop` with the EXACT abs-sums over the SAME `k` summed
/// source columns:
///   S[row,i]    = sum_{out in in_to_outs[i]} |A[row, out]|
///   prop[row,i] = sum_{out in in_to_outs[i]} |err_in[row, out]|
/// rounded OUTWARD. Using the abs-sum `S` (not `|sum|`) makes the bound hold under
/// arbitrary sign CANCELLATION among the summed coefficients. Columns with fan-in `< 2`
/// sum a single `0 + a` exactly (no fresh `gamma` term) but STILL propagate any
/// incoming `prop` for their one source column.
pub(crate) fn gather_backward_coeff_err(
    lower_a: &Array2<f32>,
    upper_a: &Array2<f32>,
    lower_a_err: Option<&Array2<f32>>,
    upper_a_err: Option<&Array2<f32>>,
    input_size: usize,
    in_to_outs: &[Vec<usize>],
) -> (Array2<f32>, Array2<f32>) {
    let num_outputs = lower_a.nrows();
    let mut lower_err = Array2::<f32>::zeros((num_outputs, input_size));
    let mut upper_err = Array2::<f32>::zeros((num_outputs, input_size));
    for (i, outs) in in_to_outs.iter().enumerate() {
        // fan-in < 2 introduces no summation rounding -> no fresh gamma term.
        let gamma = if outs.len() >= 2 {
            crate::layers::linear::crown_single_gamma_n_f32(outs.len())
        } else {
            0.0
        };
        // Nothing to certify for this column when there is neither a gamma term nor any
        // incoming err to propagate (leaves the err at the initialized zero).
        if gamma == 0.0 && lower_a_err.is_none() && upper_a_err.is_none() {
            continue;
        }
        for row in 0..num_outputs {
            let mut s_l = 0.0f64;
            let mut s_u = 0.0f64;
            let mut prop_l = 0.0f64;
            let mut prop_u = 0.0f64;
            for &out in outs {
                s_l += (lower_a[[row, out]] as f64).abs();
                s_u += (upper_a[[row, out]] as f64).abs();
                if let Some(e) = lower_a_err {
                    prop_l += (e[[row, out]] as f64).abs();
                }
                if let Some(e) = upper_a_err {
                    prop_u += (e[[row, out]] as f64).abs();
                }
            }
            lower_err[[row, i]] = ny_tensor::next_up_f32((gamma * s_l + prop_l) as f32);
            upper_err[[row, i]] = ny_tensor::next_up_f32((gamma * s_u + prop_u) as f32);
        }
    }
    (lower_err, upper_err)
}

impl BoundPropagation for GatherLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        if self.is_runtime_last_axis_len() {
            let last_axis_len = *input.shape().last().ok_or_else(|| {
                NyError::InvalidSpec(
                    "Gather runtime last-axis query requires non-empty input shape".to_string(),
                )
            })?;
            let output = ArrayD::from_elem(
                IxDyn(&self.runtime_query_output_shape()),
                last_axis_len as f32,
            );
            return BoundedTensor::new(output.clone(), output);
        }

        let input_shape = input.shape();
        let axis = self.resolve_axis(input_shape.len())?;

        if let Some(indices) = &self.indices {
            let lower = self.gather_with_indices(input.lower(), axis, indices)?;
            let upper = self.gather_with_indices(input.upper(), axis, indices)?;
            // Static-index gather is a pure layout op (selects existing values),
            // so infinite bounds pass through soundly. Allow `±inf` to flow
            // without tripping the NaN firewall; NaN is still rejected.
            return BoundedTensor::new_allow_infinite(lower, upper);
        }

        // Dynamic indices: conservatively take min(lower) and max(upper) across axis.
        // Guard NaN/Inf from unchecked callers: f32::min/max skip NaN, which can
        // silently narrow bounds. Fall back to maximally loose sound bounds.
        let has_non_finite_input = input.lower().iter().any(|&v| !v.is_finite())
            || input.upper().iter().any(|&v| !v.is_finite());
        if has_non_finite_input {
            let output_shape = self.output_shape(input_shape, axis);
            let lower = ArrayD::from_elem(IxDyn(&output_shape), f32::NEG_INFINITY);
            let upper = ArrayD::from_elem(IxDyn(&output_shape), f32::INFINITY);
            return BoundedTensor::new_allow_infinite(lower, upper);
        }

        // NaN-propagating folds: NaN in input bounds must propagate — see #2577.
        let lower_min = input
            .lower()
            .map_axis(Axis(axis), |lane| {
                lane.iter()
                    .fold(f32::INFINITY, |acc, &v| nan_propagating_min(acc, v))
            })
            .into_dyn();
        let upper_max = input
            .upper()
            .map_axis(Axis(axis), |lane| {
                lane.iter()
                    .fold(f32::NEG_INFINITY, |acc, &v| nan_propagating_max(acc, v))
            })
            .into_dyn();

        let output_shape = self.output_shape(input_shape, axis);
        let lower = self.broadcast_reduced(lower_min, axis, &output_shape)?;
        let upper = self.broadcast_reduced(upper_max, axis, &output_shape)?;
        BoundedTensor::new(lower, upper)
    }

    /// CROWN backward propagation for Gather.
    ///
    /// Gather is a linear operation (selection matrix): the backward pass scatters
    /// incoming A-matrix columns back to their original positions in the input
    /// dimension, with zeros at non-gathered positions.
    ///
    /// Requires `input_shape` to be set (via `set_input_shape`) and static `indices`.
    ///
    /// Reference: alpha-beta-CROWN `BoundGather.bound_backward` at
    /// `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/operators/indexing.py:47-82`
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        if self.is_runtime_last_axis_len() {
            return self.propagate_linear_runtime_query(bounds);
        }

        let indices = self.indices.as_ref().ok_or_else(|| {
            NyError::UnsupportedOp("Gather CROWN backward requires static indices".into())
        })?;
        let input_shape = self.input_shape.as_ref().ok_or_else(|| {
            NyError::UnsupportedOp(
                "Gather CROWN backward requires input_shape (call set_input_shape first)".into(),
            )
        })?;
        let axis = self.resolve_axis(input_shape.len())?;
        let output_shape = self.output_shape(input_shape, axis);
        let input_size = checked_shape_product(input_shape)
            .ok_or_else(|| NyError::InvalidSpec("Gather: input shape overflow".into()))?;
        let output_size = checked_shape_product(&output_shape)
            .ok_or_else(|| NyError::InvalidSpec("Gather: output shape overflow".into()))?;

        if bounds.num_inputs() != output_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![output_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let out_to_in =
            self.build_output_to_input_map(input_shape, &output_shape, axis, indices)?;
        let num_outputs = bounds.num_outputs();
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));

        // Inverse map: for each input column, the output columns that scatter-add into
        // it. DUPLICATE indices give an input column fan-in > 1, so its f32 `+=`
        // accumulation rounds — an error that MUST be certified (#vnncomp-aw-soundness).
        let mut in_to_outs: Vec<Vec<usize>> = vec![Vec::new(); input_size];

        // Scatter: accumulate (+=) because duplicate indices map multiple
        // output positions to the same input position.
        for (out_flat, &in_flat) in out_to_in.iter().enumerate() {
            in_to_outs[in_flat].push(out_flat);
            for row in 0..num_outputs {
                new_lower_a[[row, in_flat]] += bounds.lower_a()[[row, out_flat]];
                new_upper_a[[row, in_flat]] += bounds.upper_a()[[row, out_flat]];
            }
        }

        // SOUND Gather scatter-add coefficient error (#vnncomp-aw-soundness): on a
        // duplicated input column with fan-in `k`, the f32 sum of `k` source coeffs
        // rounds by at most gamma_k*S; any incoming err is re-propagated via `prop`.
        // Gather lives in `propagates_coeff_err` (query.rs), so the dispatcher hands
        // the err-carrying bounds straight here (NOT through the carrier path whose
        // `attach_err_from_carried` would OVERWRITE this fresh gamma_k*S err). Attach
        // the error only when it can be nonzero (duplicates exist OR incoming err is
        // present) so pure-permutation gathers stay exactly error-free.
        let has_duplicate = in_to_outs.iter().any(|outs| outs.len() >= 2);
        let has_incoming_err = bounds.lower_a_err().is_some() || bounds.upper_a_err().is_some();
        if has_duplicate || has_incoming_err {
            let (lower_err, upper_err) = gather_backward_coeff_err(
                bounds.lower_a(),
                bounds.upper_a(),
                bounds.lower_a_err(),
                bounds.upper_a_err(),
                input_size,
                &in_to_outs,
            );
            return Ok(Cow::Owned(LinearBounds::new_or_conservative_with_err(
                new_lower_a,
                bounds.lower_b().clone(),
                new_upper_a,
                bounds.upper_b().clone(),
                lower_err,
                upper_err,
            )?));
        }

        Ok(Cow::Owned(LinearBounds::new_or_conservative(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
        )?))
    }
}

impl GatherLayer {
    /// CROWN backward for runtime last-axis length queries.
    ///
    /// The output is a constant (the input's last-axis length), so the backward
    /// A-matrix is all zeros and the bias absorbs the constant contribution.
    fn propagate_linear_runtime_query<'a>(
        &self,
        bounds: &'a LinearBounds,
    ) -> Result<Cow<'a, LinearBounds>> {
        let input_shape = self.input_shape.as_ref().ok_or_else(|| {
            NyError::UnsupportedOp(
                "Gather runtime last-axis query requires input_shape (call set_input_shape first)"
                    .into(),
            )
        })?;
        let input_size = checked_shape_product(input_shape).ok_or_else(|| {
            NyError::InvalidSpec("Gather runtime last-axis query: input shape overflow".into())
        })?;
        let output_shape = self.runtime_query_output_shape();
        let output_size = if output_shape.is_empty() {
            1
        } else {
            checked_shape_product(&output_shape).ok_or_else(|| {
                NyError::InvalidSpec("Gather runtime last-axis query: output shape overflow".into())
            })?
        };
        if bounds.num_inputs() != output_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![output_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let queried_len = *input_shape.last().ok_or_else(|| {
            NyError::InvalidSpec(
                "Gather runtime last-axis query requires non-empty input shape".into(),
            )
        })? as f32;
        let num_outputs = bounds.num_outputs();
        let mut lower_bias = bounds.lower_b().clone();
        let mut upper_bias = bounds.upper_b().clone();
        // Fold each coeff * queried_len into the bias (the output A is all-zero). Accumulate
        // in f64 and directed-cast OUTWARD, and fold the INCOMING coeff error into the bias
        // too — the old f32 `.sum()` rounded round-to-nearest (false-tight under cancellation)
        // AND silently dropped lower_a_err/upper_a_err (#vnncomp-aw-soundness self-audit).
        // queried_len is an integer dim length (<= 2^24, exact in f32/f64).
        let ql = queried_len as f64;
        let in_lower_err = bounds.lower_a_err();
        let in_upper_err = bounds.upper_a_err();
        for row in 0..num_outputs {
            let mut lower_const = 0.0f64;
            let mut upper_const = 0.0f64;
            let mut lower_err = 0.0f64;
            let mut upper_err = 0.0f64;
            for col in 0..output_size {
                lower_const += (bounds.lower_a()[[row, col]] as f64) * ql;
                upper_const += (bounds.upper_a()[[row, col]] as f64) * ql;
                if let Some(e) = in_lower_err {
                    lower_err += (e[[row, col]] as f64).abs() * ql;
                }
                if let Some(e) = in_upper_err {
                    upper_err += (e[[row, col]] as f64).abs() * ql;
                }
            }
            lower_bias[row] = ny_tensor::next_down_f32(
                ((lower_bias[row] as f64) + lower_const - lower_err) as f32,
            );
            upper_bias[row] =
                ny_tensor::next_up_f32(((upper_bias[row] as f64) + upper_const + upper_err) as f32);
        }

        Ok(Cow::Owned(LinearBounds::new_or_conservative(
            Array2::zeros((num_outputs, input_size)),
            lower_bias,
            Array2::zeros((num_outputs, input_size)),
            upper_bias,
        )?))
    }

    /// Build a mapping from each flat output index to its corresponding flat input index.
    ///
    /// For each output position, decomposes into multi-dim coords, replaces the
    /// gather-axis portion with the gathered index from `indices`, and converts
    /// back to a flat input position.
    fn build_output_to_input_map(
        &self,
        input_shape: &[usize],
        output_shape: &[usize],
        axis: usize,
        indices: &ArrayD<i64>,
    ) -> Result<Vec<usize>> {
        let axis_len = input_shape[axis] as i64;
        let indices_flat: Vec<i64> = indices.iter().copied().collect();
        let indices_norm = self.normalize_indices(&indices_flat, axis_len)?;

        let ndim = input_shape.len();
        let out_ndim = output_shape.len();
        let output_size = checked_shape_product(output_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Gather output shape product overflows: {:?}",
                output_shape
            ))
        })?;
        let idx_ndim = self.indices_shape.len();

        let output_strides = Self::compute_strides(output_shape);
        let input_strides = Self::compute_strides(input_shape);

        let mut map = Vec::with_capacity(output_size);
        for out_flat in 0..output_size {
            let out_multi = Self::unflatten(out_flat, &output_strides, out_ndim);
            let idx_flat =
                Self::flatten_range(&out_multi[axis..axis + idx_ndim], &self.indices_shape);
            let gathered_idx = indices_norm[idx_flat];

            let mut in_multi = Vec::with_capacity(ndim);
            in_multi.extend_from_slice(&out_multi[..axis]);
            in_multi.push(gathered_idx);
            in_multi.extend_from_slice(&out_multi[axis + idx_ndim..]);

            let in_flat: usize = in_multi
                .iter()
                .zip(&input_strides)
                .map(|(i, s)| i * s)
                .sum();
            map.push(in_flat);
        }
        Ok(map)
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let n = shape.len();
        let mut strides = vec![1usize; n];
        for i in (0..n.saturating_sub(1)).rev() {
            // Safety: callers verify the total shape product via checked_shape_product
            // before reaching here, so all partial stride products fit in usize.
            strides[i] = strides[i + 1]
                .checked_mul(shape[i + 1])
                .expect("invariant: total shape product was checked before compute_strides");
        }
        strides
    }

    fn unflatten(flat: usize, strides: &[usize], ndim: usize) -> Vec<usize> {
        let mut multi = vec![0usize; ndim];
        let mut remaining = flat;
        for i in 0..ndim {
            multi[i] = remaining / strides[i];
            remaining %= strides[i];
        }
        multi
    }

    fn flatten_range(coords: &[usize], shape: &[usize]) -> usize {
        let mut flat = 0usize;
        let mut stride = 1usize;
        for i in (0..coords.len()).rev() {
            flat += coords[i] * stride;
            if i > 0 {
                stride *= shape[i];
            }
        }
        flat
    }
}

#[cfg(test)]
mod tests;
