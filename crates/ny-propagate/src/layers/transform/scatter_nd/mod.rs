// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ScatterND layer: scatter update values into a copy of a data tensor.
//!
//! ONNX `ScatterND` writes `updates` into `data` at positions described by
//! `indices`. For verification we support:
//! - static indices: exact IBP overwrite semantics with duplicate-target union
//! - dynamic indices: conservative IBP hull over all update placements
//!
//! alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) has no
//! `ScatterND` implementation (`rg ScatterND` over a reference checkout, surveyed
//! 2026-03-10), so CROWN backward remains an explicit `UnsupportedOp` fallback here.

use ndarray::{Array1, Array2, ArrayD, IxDyn, Zip};
use ny_core::{checked_dim_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;

use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

/// Which operand of ScatterND is the bounded/variable activation input for the
/// constant-index CROWN backward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VariableOperand {
    /// `data` is variable, `updates` constant. Output keeps `data` at unwritten
    /// positions (identity Jacobian) and a constant at written positions.
    Data,
    /// `updates` is variable, `data` constant. Each updates element maps 1:1 to a
    /// written output position (selection Jacobian); `data` shifts the bias.
    Updates,
}

#[derive(Debug, Clone)]
pub struct ScatterNdLayer {
    data_constant: Option<ArrayD<f32>>,
    indices: Option<ArrayD<i64>>,
    updates_constant: Option<ArrayD<f32>>,
}

#[derive(Debug, Clone)]
enum ResolvedIndices {
    Static(Box<ArrayD<i64>>),
    /// Variable indices with their propagated interval bounds (#cctsdb B4).
    /// The bounds drive the definitely/possibly-written classification; when
    /// they are unusable (non-finite, wrong shape, over budget) the layer
    /// degrades to the global min/max hull.
    Dynamic(Box<BoundedTensor>),
}

#[derive(Debug, Clone)]
struct ResolvedInputs {
    data: BoundedTensor,
    indices: ResolvedIndices,
    updates: BoundedTensor,
}

impl ScatterNdLayer {
    pub fn new(
        data_constant: Option<ArrayD<f32>>,
        indices: Option<ArrayD<i64>>,
        updates_constant: Option<ArrayD<f32>>,
    ) -> Self {
        Self {
            data_constant,
            indices,
            updates_constant,
        }
    }

    pub fn has_static_indices(&self) -> bool {
        self.indices.is_some()
    }

    /// Embedded constant data operand, when static (f64 cell evaluator).
    pub fn data_constant(&self) -> Option<&ArrayD<f32>> {
        self.data_constant.as_ref()
    }

    /// Embedded constant updates operand, when static (f64 cell evaluator).
    pub fn updates_constant(&self) -> Option<&ArrayD<f32>> {
        self.updates_constant.as_ref()
    }

    pub fn activation_input_count(&self) -> usize {
        usize::from(self.data_constant.is_none())
            + usize::from(self.indices.is_none())
            + usize::from(self.updates_constant.is_none())
    }

    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        self.propagate_ibp_with_inputs(&[input_a, input_b])
    }

    pub fn propagate_ibp_ternary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
        input_c: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        self.propagate_ibp_with_inputs(&[input_a, input_b, input_c])
    }

    fn propagate_ibp_with_inputs(
        &self,
        activation_inputs: &[&BoundedTensor],
    ) -> Result<BoundedTensor> {
        let resolved = self.resolve_inputs(activation_inputs)?;
        match resolved.indices {
            ResolvedIndices::Static(indices) => {
                let lower = self.scatter_tensor(
                    resolved.data.lower(),
                    resolved.updates.lower(),
                    &indices,
                    nan_propagating_min,
                )?;
                let upper = self.scatter_tensor(
                    resolved.data.upper(),
                    resolved.updates.upper(),
                    &indices,
                    nan_propagating_max,
                )?;
                BoundedTensor::new(lower, upper)
            }
            ResolvedIndices::Dynamic(indices_bounds) => {
                // Bounded-index path first (#cctsdb B4): classify output
                // elements as definitely / possibly / never written using the
                // indices' interval bounds. Falls back to the global hull when
                // the bounds are unusable.
                if let Some(result) = self.propagate_bounded_dynamic_indices(
                    &resolved.data,
                    &indices_bounds,
                    &resolved.updates,
                )? {
                    return Ok(result);
                }
                self.propagate_dynamic_indices(&resolved.data, &resolved.updates)
            }
        }
    }

    fn resolve_inputs(&self, activation_inputs: &[&BoundedTensor]) -> Result<ResolvedInputs> {
        let expected = self.activation_input_count();
        if activation_inputs.len() != expected {
            return Err(NyError::InvalidSpec(format!(
                "ScatterND expects {} activation input(s), got {}",
                expected,
                activation_inputs.len()
            )));
        }

        let mut next_input = 0usize;

        let data = if let Some(data) = &self.data_constant {
            BoundedTensor::concrete(data.clone())?
        } else {
            let input = activation_inputs
                .get(next_input)
                .ok_or_else(|| NyError::InvalidSpec("ScatterND missing data input".to_string()))?;
            next_input += 1;
            (*input).clone()
        };

        let indices = if let Some(indices) = &self.indices {
            ResolvedIndices::Static(Box::new(indices.clone()))
        } else {
            let input = activation_inputs.get(next_input).ok_or_else(|| {
                NyError::InvalidSpec("ScatterND missing indices input".to_string())
            })?;
            next_input += 1;
            // Keep the indices' interval bounds (#cctsdb B4) — they drive the
            // definitely/possibly-written classification instead of being
            // discarded for a position-blind global hull.
            ResolvedIndices::Dynamic(Box::new((*input).clone()))
        };

        let updates = if let Some(updates) = &self.updates_constant {
            BoundedTensor::concrete(updates.clone())?
        } else {
            let input = activation_inputs.get(next_input).ok_or_else(|| {
                NyError::InvalidSpec("ScatterND missing updates input".to_string())
            })?;
            (*input).clone()
        };

        Ok(ResolvedInputs {
            data,
            indices,
            updates,
        })
    }

    /// Bounded-index dynamic ScatterND (#cctsdb B4).
    ///
    /// For each indices prefix row, its k-th coordinate interval `[lo, hi]`
    /// constrains the true (integer) index to `[ceil(lo), floor(hi)]` — the
    /// true index IS an integer inside the sound f32 interval, so the integer
    /// hull is exact. Per output element:
    /// - DEFINITELY written: some row has all-singleton in-range coordinates
    ///   hitting it — the element is certainly overwritten, so its bounds are
    ///   the hull of updates over every row that may write it (data dropped).
    /// - POSSIBLY written: it lies in some row's coordinate box — bounds are
    ///   the elementwise hull of data and those rows' updates.
    /// - Untouched: data passes through exactly.
    ///
    /// Out-of-range possible coordinates are dropped (they never widen data);
    /// a fully out-of-range row is skipped — this keeps static-max-shape
    /// windows exact when the true graph clamps at the edge (patch-3): the
    /// rejected sentinel row matches the row the clamped graph never writes.
    ///
    /// Soundness of duplicates: when several rows may hit an element, ONNX
    /// leaves the result order-dependent; the hull over ALL candidate writers
    /// contains every resolution.
    ///
    /// Returns `Ok(None)` to fall back to the position-blind global hull when
    /// the interval bounds are unusable (non-finite indices, shape mismatch,
    /// enumeration over budget, non-finite data/updates).
    fn propagate_bounded_dynamic_indices(
        &self,
        data: &BoundedTensor,
        indices: &BoundedTensor,
        updates: &BoundedTensor,
    ) -> Result<Option<BoundedTensor>> {
        // Enumeration budget: total coordinate-box cells (x slice_len) that the
        // classification loop may visit before degrading to the global hull.
        const BOX_CELL_BUDGET: usize = 8_000_000;

        if updates.lower().is_empty() {
            return Ok(Some(data.clone()));
        }
        // Degrade gracefully on non-finite indices (design B4) and keep the
        // legacy full-unbounded behavior for non-finite data/updates.
        let all_finite = |t: &BoundedTensor| {
            t.lower().iter().all(|v| v.is_finite()) && t.upper().iter().all(|v| v.is_finite())
        };
        if !all_finite(indices) || !all_finite(data) || !all_finite(updates) {
            return Ok(None);
        }

        let data_shape = data.shape().to_vec();
        if indices.lower().ndim() == 0 {
            return Ok(None);
        }
        let Some(&index_depth) = indices.shape().last() else {
            return Ok(None);
        };
        if index_depth == 0 || index_depth > data_shape.len() {
            return Ok(None);
        }
        let prefix_shape = &indices.shape()[..indices.lower().ndim() - 1];
        let prefix_elems = shape_product(prefix_shape)?;
        let remainder_shape = &data_shape[index_depth..];
        let slice_len = shape_product(remainder_shape)?;
        let expected_updates_shape = prefix_shape
            .iter()
            .copied()
            .chain(remainder_shape.iter().copied())
            .collect::<Vec<_>>();
        if updates.shape() != expected_updates_shape.as_slice() {
            return Ok(None);
        }
        if prefix_elems == 0 || slice_len == 0 {
            return Ok(Some(data.clone()));
        }

        let data_size = shape_product(&data_shape)?;
        let data_strides = compute_strides(&data_shape);
        let remainder_offsets = compute_remainder_offsets(&data_shape, index_depth)?;
        let idx_lower: Vec<f32> = indices.lower().iter().copied().collect();
        let idx_upper: Vec<f32> = indices.upper().iter().copied().collect();
        let upd_lower: Vec<f32> = updates.lower().iter().copied().collect();
        let upd_upper: Vec<f32> = updates.upper().iter().copied().collect();

        // Per-element classification accumulators.
        let mut definitely = vec![false; data_size];
        let mut possible = vec![false; data_size];
        let mut acc_lo = vec![f32::INFINITY; data_size];
        let mut acc_up = vec![f32::NEG_INFINITY; data_size];
        let mut budget_used = 0usize;

        for prefix_idx in 0..prefix_elems {
            // Per-axis sets of possible normalized (non-negative) coordinates,
            // as flat-stride offsets contributions.
            let mut axis_offsets: Vec<Vec<usize>> = Vec::with_capacity(index_depth);
            let mut row_singleton = true;
            let mut row_feasible = true;
            for axis in 0..index_depth {
                let flat = prefix_idx * index_depth + axis;
                let axis_len = data_shape[axis];
                let (coords, singleton) =
                    match possible_normalized_coords(idx_lower[flat], idx_upper[flat], axis_len) {
                        Some(v) => v,
                        None => {
                            // No integer in the interval, or every candidate is
                            // out of range: this row cannot perform a write.
                            row_feasible = false;
                            break;
                        }
                    };
                row_singleton &= singleton;
                axis_offsets.push(coords);
            }
            if !row_feasible {
                continue;
            }

            let box_cells: usize = axis_offsets.iter().map(Vec::len).product();
            budget_used = budget_used.saturating_add(box_cells.saturating_mul(slice_len));
            if budget_used > BOX_CELL_BUDGET {
                return Ok(None);
            }

            // Enumerate the cartesian product of per-axis coordinates via an
            // odometer over axis_offsets.
            let mut odometer = vec![0usize; index_depth];
            loop {
                let mut target_base = 0usize;
                for axis in 0..index_depth {
                    target_base += axis_offsets[axis][odometer[axis]] * data_strides[axis];
                }
                let updates_start = prefix_idx * slice_len;
                for (offset_idx, rel_offset) in remainder_offsets.iter().copied().enumerate() {
                    let target = target_base + rel_offset;
                    let update_lo = upd_lower[updates_start + offset_idx];
                    let update_up = upd_upper[updates_start + offset_idx];
                    possible[target] = true;
                    acc_lo[target] = nan_propagating_min(acc_lo[target], update_lo);
                    acc_up[target] = nan_propagating_max(acc_up[target], update_up);
                    if row_singleton {
                        definitely[target] = true;
                    }
                }
                // Advance odometer.
                let mut axis = index_depth;
                loop {
                    if axis == 0 {
                        break;
                    }
                    axis -= 1;
                    odometer[axis] += 1;
                    if odometer[axis] < axis_offsets[axis].len() {
                        break;
                    }
                    odometer[axis] = 0;
                    if axis == 0 {
                        // Wrapped around completely: done with this row.
                        break;
                    }
                }
                if odometer.iter().all(|&c| c == 0) {
                    break;
                }
            }
        }

        let mut out_lower = data.lower().clone();
        let mut out_upper = data.upper().clone();
        {
            let (Some(lo_slice), Some(up_slice)) =
                (out_lower.as_slice_mut(), out_upper.as_slice_mut())
            else {
                return Ok(None);
            };
            for element in 0..data_size {
                if definitely[element] {
                    // Certainly overwritten by some row: hull of all candidate
                    // writers' updates; data is dropped.
                    lo_slice[element] = acc_lo[element];
                    up_slice[element] = acc_up[element];
                } else if possible[element] {
                    lo_slice[element] = nan_propagating_min(lo_slice[element], acc_lo[element]);
                    up_slice[element] = nan_propagating_max(up_slice[element], acc_up[element]);
                }
            }
        }
        BoundedTensor::new(out_lower, out_upper).map(Some)
    }

    fn propagate_dynamic_indices(
        &self,
        data: &BoundedTensor,
        updates: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        if updates.lower().is_empty() {
            return Ok(data.clone());
        }

        let has_non_finite = data.lower().iter().any(|&v| !v.is_finite())
            || data.upper().iter().any(|&v| !v.is_finite())
            || updates.lower().iter().any(|&v| !v.is_finite())
            || updates.upper().iter().any(|&v| !v.is_finite());
        if has_non_finite {
            let lower = ArrayD::from_elem(IxDyn(data.shape()), f32::NEG_INFINITY);
            let upper = ArrayD::from_elem(IxDyn(data.shape()), f32::INFINITY);
            return BoundedTensor::new_allow_infinite(lower, upper);
        }

        let global_lower = updates
            .lower()
            .iter()
            .fold(f32::INFINITY, |acc, &v| nan_propagating_min(acc, v));
        let global_upper = updates
            .upper()
            .iter()
            .fold(f32::NEG_INFINITY, |acc, &v| nan_propagating_max(acc, v));

        let lower = Zip::from(data.lower()).map_collect(|&v| nan_propagating_min(v, global_lower));
        let upper = Zip::from(data.upper()).map_collect(|&v| nan_propagating_max(v, global_upper));
        BoundedTensor::new(lower, upper)
    }

    fn scatter_tensor(
        &self,
        data: &ArrayD<f32>,
        updates: &ArrayD<f32>,
        indices: &ArrayD<i64>,
        combine_duplicate: fn(f32, f32) -> f32,
    ) -> Result<ArrayD<f32>> {
        let data_shape = data.shape().to_vec();
        if indices.ndim() == 0 {
            return Err(NyError::InvalidSpec(
                "ScatterND indices rank must be at least 1".to_string(),
            ));
        }

        let Some(&index_depth) = indices.shape().last() else {
            return Err(NyError::InvalidSpec(
                "ScatterND indices missing last dimension".to_string(),
            ));
        };
        if index_depth == 0 || index_depth > data_shape.len() {
            return Err(NyError::InvalidSpec(format!(
                "ScatterND index depth {} out of range for data rank {}",
                index_depth,
                data_shape.len()
            )));
        }

        let prefix_shape = &indices.shape()[..indices.ndim() - 1];
        let prefix_elems = shape_product(prefix_shape)?;
        if prefix_elems == 0 {
            return Ok(data.clone());
        }

        let remainder_shape = &data_shape[index_depth..];
        let expected_updates_shape = prefix_shape
            .iter()
            .copied()
            .chain(remainder_shape.iter().copied())
            .collect::<Vec<_>>();
        if updates.shape() != expected_updates_shape.as_slice() {
            return Err(NyError::ShapeMismatch {
                expected: expected_updates_shape,
                got: updates.shape().to_vec(),
            });
        }

        let slice_len = shape_product(remainder_shape)?;
        let data_size = shape_product(&data_shape)?;
        let mut output = data.iter().copied().collect::<Vec<_>>();
        let mut written = vec![false; data_size];
        let indices_flat = indices.iter().copied().collect::<Vec<_>>();
        let updates_flat = updates.iter().copied().collect::<Vec<_>>();
        let data_strides = compute_strides(&data_shape);
        let remainder_offsets = compute_remainder_offsets(&data_shape, index_depth)?;

        for prefix_idx in 0..prefix_elems {
            let index_start = prefix_idx * index_depth;
            let mut target_base = 0usize;
            for axis in 0..index_depth {
                let raw = *indices_flat.get(index_start + axis).ok_or_else(|| {
                    NyError::InvalidSpec(
                        "ScatterND indices flattened layout shorter than expected".to_string(),
                    )
                })?;
                let axis_len = data_shape[axis];
                let normalized = normalize_index(raw, axis_len)?;
                target_base += normalized * data_strides[axis];
            }

            if slice_len == 0 {
                continue;
            }

            let updates_start = prefix_idx * slice_len;
            for (offset_idx, rel_offset) in remainder_offsets.iter().copied().enumerate() {
                let update_idx = updates_start + offset_idx;
                let target_idx = target_base + rel_offset;
                let update_val = *updates_flat.get(update_idx).ok_or_else(|| {
                    NyError::InvalidSpec(
                        "ScatterND updates flattened layout shorter than expected".to_string(),
                    )
                })?;
                if !written[target_idx] {
                    output[target_idx] = update_val;
                    written[target_idx] = true;
                } else {
                    output[target_idx] = combine_duplicate(output[target_idx], update_val);
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&data_shape), output)
            .map_err(|e| NyError::InvalidSpec(format!("ScatterND output reshape failed: {}", e)))
    }
}

impl ScatterNdLayer {
    /// Identify the single variable operand for constant-index CROWN backward.
    fn crown_variable_operand(&self) -> Result<VariableOperand> {
        if self.indices.is_none() {
            return Err(NyError::UnsupportedOp(
                "ScatterND CROWN backward requires static (constant) indices".to_string(),
            ));
        }
        match (
            self.data_constant.is_none(),
            self.updates_constant.is_none(),
        ) {
            (true, false) => Ok(VariableOperand::Data),
            (false, true) => Ok(VariableOperand::Updates),
            _ => Err(NyError::UnsupportedOp(
                "ScatterND CROWN backward requires exactly one variable operand".to_string(),
            )),
        }
    }

    /// Compute, for the given data shape, the write map of ScatterND:
    /// `out_to_update[out_flat] = Some(update_flat)` if output position `out_flat`
    /// is overwritten by `updates[update_flat]`, else `None`. Returns an error if
    /// any output position is written more than once (last-write/union semantics
    /// are not an exact linear map), so the caller can fall back to IBP.
    fn build_write_map(
        &self,
        data_shape: &[usize],
        updates_shape: &[usize],
        indices: &ArrayD<i64>,
    ) -> Result<Vec<Option<usize>>> {
        if indices.ndim() == 0 {
            return Err(NyError::InvalidSpec(
                "ScatterND indices rank must be at least 1".to_string(),
            ));
        }
        let Some(&index_depth) = indices.shape().last() else {
            return Err(NyError::InvalidSpec(
                "ScatterND indices missing last dimension".to_string(),
            ));
        };
        if index_depth == 0 || index_depth > data_shape.len() {
            return Err(NyError::InvalidSpec(format!(
                "ScatterND index depth {} out of range for data rank {}",
                index_depth,
                data_shape.len()
            )));
        }

        let prefix_shape = &indices.shape()[..indices.ndim() - 1];
        let prefix_elems = shape_product(prefix_shape)?;
        let remainder_shape = &data_shape[index_depth..];
        let slice_len = shape_product(remainder_shape)?;
        let expected_updates_shape = prefix_shape
            .iter()
            .copied()
            .chain(remainder_shape.iter().copied())
            .collect::<Vec<_>>();
        if updates_shape != expected_updates_shape.as_slice() {
            return Err(NyError::ShapeMismatch {
                expected: expected_updates_shape,
                got: updates_shape.to_vec(),
            });
        }

        let data_size = shape_product(data_shape)?;
        let mut map: Vec<Option<usize>> = vec![None; data_size];
        if prefix_elems == 0 || slice_len == 0 {
            return Ok(map);
        }

        let indices_flat = indices.iter().copied().collect::<Vec<_>>();
        let data_strides = compute_strides(data_shape);
        let remainder_offsets = compute_remainder_offsets(data_shape, index_depth)?;

        for prefix_idx in 0..prefix_elems {
            let index_start = prefix_idx * index_depth;
            let mut target_base = 0usize;
            for axis in 0..index_depth {
                let raw = *indices_flat.get(index_start + axis).ok_or_else(|| {
                    NyError::InvalidSpec(
                        "ScatterND indices flattened layout shorter than expected".to_string(),
                    )
                })?;
                let normalized = normalize_index(raw, data_shape[axis])?;
                target_base += normalized * data_strides[axis];
            }
            let updates_start = prefix_idx * slice_len;
            for (offset_idx, rel_offset) in remainder_offsets.iter().copied().enumerate() {
                let update_idx = updates_start + offset_idx;
                let target_idx = target_base + rel_offset;
                if map[target_idx].is_some() {
                    // Duplicate target: overwrite/union semantics are not an exact
                    // linear map. Bail so the caller uses the sound IBP fallback.
                    return Err(NyError::UnsupportedOp(
                        "ScatterND CROWN backward: duplicate write targets are not exactly \
                         linear"
                            .to_string(),
                    ));
                }
                map[target_idx] = Some(update_idx);
            }
        }
        Ok(map)
    }

    /// Exact CROWN backward for ScatterND with constant indices and a single
    /// variable operand. Overwrite semantics: written positions take the update
    /// (constant when data is variable), unwritten positions keep data.
    pub fn crown_backward(&self, node_lb: &LinearBounds) -> Result<LinearBounds> {
        let variable = self.crown_variable_operand()?;
        let indices = self.indices.as_ref().expect("static indices checked above");
        let output_size = node_lb.num_inputs();
        let num_outputs = node_lb.num_outputs();
        let lower_a = node_lb.lower_a();
        let upper_a = node_lb.upper_a();

        match variable {
            VariableOperand::Data => {
                let updates = self.updates_constant.as_ref().ok_or_else(|| {
                    NyError::UnsupportedOp(
                        "ScatterND CROWN (data-variable) requires constant updates".to_string(),
                    )
                })?;
                // data shape == output shape; only the flat size is available, so
                // we require 1-D data (the common feature-vector overwrite case).
                if index_depth_of(indices)? != 1 {
                    return Err(NyError::UnsupportedOp(
                        "ScatterND CROWN (data-variable) supports 1-D data only without \
                         explicit data shape"
                            .to_string(),
                    ));
                }
                let data_shape = vec![output_size];
                let write_map = self.build_write_map(&data_shape, updates.shape(), indices)?;
                let updates_flat: Vec<f32> = updates.iter().copied().collect();

                let mut new_lower_a = lower_a.clone();
                let mut new_upper_a = upper_a.clone();
                let mut new_lower_b = node_lb.lower_b().clone();
                let mut new_upper_b = node_lb.upper_b().clone();
                // Fold each overwritten cell's `coeff * c` into the bias in f64 + directed
                // cast (#vnncomp-aw-soundness): the f32 multiply + multi-output accumulation
                // rounds and can be tighter than the true value under cancellation.
                let in_lo_err = node_lb.lower_a_err();
                let in_up_err = node_lb.upper_a_err();
                let mut const_lo = vec![0.0f64; num_outputs];
                let mut const_up = vec![0.0f64; num_outputs];
                let mut const_lo_err = vec![0.0f64; num_outputs];
                let mut const_up_err = vec![0.0f64; num_outputs];
                for (out_i, &written) in write_map.iter().enumerate() {
                    if let Some(update_idx) = written {
                        let c = updates_flat[update_idx] as f64;
                        for row in 0..num_outputs {
                            const_lo[row] += (lower_a[[row, out_i]] as f64) * c;
                            const_up[row] += (upper_a[[row, out_i]] as f64) * c;
                            if let Some(e) = in_lo_err {
                                const_lo_err[row] += (e[[row, out_i]] as f64).abs() * c.abs();
                            }
                            if let Some(e) = in_up_err {
                                const_up_err[row] += (e[[row, out_i]] as f64).abs() * c.abs();
                            }
                            new_lower_a[[row, out_i]] = 0.0;
                            new_upper_a[[row, out_i]] = 0.0;
                        }
                    }
                }
                fold_const_into_bias(
                    &mut new_lower_b,
                    &mut new_upper_b,
                    &const_lo,
                    &const_up,
                    &const_lo_err,
                    &const_up_err,
                );
                LinearBounds::new_or_conservative(
                    new_lower_a,
                    new_lower_b,
                    new_upper_a,
                    new_upper_b,
                )
            }
            VariableOperand::Updates => {
                let data = self.data_constant.as_ref().ok_or_else(|| {
                    NyError::UnsupportedOp(
                        "ScatterND CROWN (updates-variable) requires constant data".to_string(),
                    )
                })?;
                if data.len() != output_size {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![output_size],
                        got: vec![data.len()],
                    });
                }
                let data_shape = data.shape().to_vec();
                // updates shape = prefix(indices) ++ data_shape[index_depth..].
                let updates_shape = scatter_nd_updates_shape(&data_shape, indices)?;
                let updates_size = shape_product(&updates_shape)?;
                let write_map = self.build_write_map(&data_shape, &updates_shape, indices)?;
                let data_flat: Vec<f32> = data.iter().copied().collect();

                let mut new_lower_a = Array2::<f32>::zeros((num_outputs, updates_size));
                let mut new_upper_a = Array2::<f32>::zeros((num_outputs, updates_size));
                let mut new_lower_b = node_lb.lower_b().clone();
                let mut new_upper_b = node_lb.upper_b().clone();
                // The routing scatter-add below is EXACT — `build_write_map` rejects duplicate
                // write targets (returns UnsupportedOp → sound IBP fallback), so each
                // `update_idx` cell receives exactly one column (a bijection, no accumulation).
                // The CONSTANT data fold, however, f32-accumulates `coeff * c` over many `out_i`
                // into one row's bias, so fold it OUTWARD in f64 + directed (#vnncomp-aw-soundness).
                let in_lo_err = node_lb.lower_a_err();
                let in_up_err = node_lb.upper_a_err();
                let mut const_lo = vec![0.0f64; num_outputs];
                let mut const_up = vec![0.0f64; num_outputs];
                let mut const_lo_err = vec![0.0f64; num_outputs];
                let mut const_up_err = vec![0.0f64; num_outputs];
                for (out_i, &written) in write_map.iter().enumerate() {
                    match written {
                        Some(update_idx) => {
                            // out_i comes from updates[update_idx]: route the column (exact).
                            for row in 0..num_outputs {
                                new_lower_a[[row, update_idx]] += lower_a[[row, out_i]];
                                new_upper_a[[row, update_idx]] += upper_a[[row, out_i]];
                            }
                        }
                        None => {
                            let c = data_flat[out_i] as f64;
                            for row in 0..num_outputs {
                                const_lo[row] += (lower_a[[row, out_i]] as f64) * c;
                                const_up[row] += (upper_a[[row, out_i]] as f64) * c;
                                if let Some(e) = in_lo_err {
                                    const_lo_err[row] += (e[[row, out_i]] as f64).abs() * c.abs();
                                }
                                if let Some(e) = in_up_err {
                                    const_up_err[row] += (e[[row, out_i]] as f64).abs() * c.abs();
                                }
                            }
                        }
                    }
                }
                fold_const_into_bias(
                    &mut new_lower_b,
                    &mut new_upper_b,
                    &const_lo,
                    &const_up,
                    &const_lo_err,
                    &const_up_err,
                );
                LinearBounds::new_or_conservative(
                    new_lower_a,
                    new_lower_b,
                    new_upper_a,
                    new_upper_b,
                )
            }
        }
    }
}

/// Fold the f64-accumulated `coeff * c` constant contributions OUTWARD into the bias
/// (#vnncomp-aw-soundness): the f32 multiply + multi-output accumulation rounds and can be
/// tighter than the true real value under cancellation, so each row's contribution is
/// accumulated in f64, the incoming coeff err re-folded outward (subtract from lower / add to
/// upper), and the result directed-cast (next_down / next_up). A row with no real contribution
/// (`const == 0` and no err) is left untouched so the common no-fold case is never widened.
fn fold_const_into_bias(
    lower_b: &mut Array1<f32>,
    upper_b: &mut Array1<f32>,
    const_lo: &[f64],
    const_up: &[f64],
    const_lo_err: &[f64],
    const_up_err: &[f64],
) {
    for row in 0..lower_b.len() {
        if const_lo[row] != 0.0 || const_lo_err[row] != 0.0 {
            lower_b[row] =
                next_down_f32(((lower_b[row] as f64) + const_lo[row] - const_lo_err[row]) as f32);
        }
        if const_up[row] != 0.0 || const_up_err[row] != 0.0 {
            upper_b[row] =
                next_up_f32(((upper_b[row] as f64) + const_up[row] + const_up_err[row]) as f32);
        }
    }
}

/// The last dimension (index depth) of a ScatterND indices tensor.
fn index_depth_of(indices: &ArrayD<i64>) -> Result<usize> {
    indices
        .shape()
        .last()
        .copied()
        .ok_or_else(|| NyError::InvalidSpec("ScatterND indices missing last dimension".to_string()))
}

/// Compute the ScatterND updates shape from the data shape and indices:
/// `prefix(indices) ++ data_shape[index_depth..]`.
fn scatter_nd_updates_shape(data_shape: &[usize], indices: &ArrayD<i64>) -> Result<Vec<usize>> {
    let index_depth = index_depth_of(indices)?;
    if index_depth == 0 || index_depth > data_shape.len() {
        return Err(NyError::InvalidSpec(format!(
            "ScatterND index depth {} out of range for data rank {}",
            index_depth,
            data_shape.len()
        )));
    }
    let prefix_shape = &indices.shape()[..indices.ndim() - 1];
    let mut shape = prefix_shape.to_vec();
    shape.extend_from_slice(&data_shape[index_depth..]);
    Ok(shape)
}

impl BoundPropagation for ScatterNdLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_ibp_with_inputs(&[input])
    }

    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        self.crown_backward(bounds).map(Cow::Owned)
    }
}

/// Possible normalized coordinates for one bounded index component (#cctsdb B4).
///
/// The true index is an INTEGER inside the sound interval `[lo, hi]`, so its
/// integer hull is `[ceil(lo), floor(hi)]` (f32 ceil/floor are exact; the
/// `as i64` casts saturate, and the subsequent clamps to `[-len, len-1]` make
/// saturation harmless). ONNX allows negative indices (normalized by adding
/// `axis_len`); candidates outside `[-len, len-1]` are dropped (undefined
/// behavior in ONNX — they can never widen data).
///
/// Returns `None` when the row cannot perform a write through this component
/// (no integer in the interval, or all candidates out of range). The `bool` is
/// `true` when the true index is uniquely determined (singleton interval) —
/// the "definitely written" precondition.
fn possible_normalized_coords(lo: f32, hi: f32, axis_len: usize) -> Option<(Vec<usize>, bool)> {
    let ilo_f = lo.ceil();
    let ihi_f = hi.floor();
    if ilo_f > ihi_f {
        return None;
    }
    let ilo = ilo_f as i64; // saturating cast
    let ihi = ihi_f as i64;
    let len = i64::try_from(axis_len).ok()?;
    if len == 0 {
        return None;
    }

    let mut coords: Vec<usize> = Vec::new();
    // Negative candidates map to len + i.
    let neg_start = ilo.max(-len);
    let neg_end = ihi.min(-1);
    for i in neg_start..=neg_end {
        coords.push((i + len) as usize);
    }
    // Non-negative candidates map to themselves.
    let pos_start = ilo.max(0);
    let pos_end = ihi.min(len - 1);
    for i in pos_start..=pos_end {
        coords.push(i as usize);
    }
    if coords.is_empty() {
        return None;
    }
    coords.sort_unstable();
    coords.dedup();
    Some((coords, ilo == ihi))
}

fn normalize_index(index: i64, axis_len: usize) -> Result<usize> {
    let axis_len_i64 = i64::try_from(axis_len).map_err(|_| {
        NyError::InvalidSpec(format!("ScatterND axis length {} exceeds i64", axis_len))
    })?;
    let adjusted = if index < 0 {
        axis_len_i64 + index
    } else {
        index
    };
    if adjusted < 0 || adjusted >= axis_len_i64 {
        return Err(NyError::InvalidSpec(format!(
            "ScatterND index {} out of bounds for axis length {}",
            index, axis_len
        )));
    }
    Ok(adjusted as usize)
}

fn shape_product(shape: &[usize]) -> Result<usize> {
    checked_dim_product(shape, "ScatterND shape product")
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    // Row-major strides: walk axes from last-1 down to 0. The previous
    // `(0..len).rev().skip(1)` form included idx=0 and underflowed on
    // `idx - 1` for any rank >= 2 (latent: all prior callers used rank-1
    // data); exposed by the bounded-index path on rank-3 masks (#cctsdb B4).
    for idx in (1..shape.len()).rev() {
        strides[idx - 1] = strides[idx] * shape[idx];
    }
    strides
}

fn compute_remainder_offsets(data_shape: &[usize], index_depth: usize) -> Result<Vec<usize>> {
    let remainder_shape = &data_shape[index_depth..];
    let slice_len = shape_product(remainder_shape)?;
    if remainder_shape.is_empty() {
        return Ok(vec![0]);
    }

    let remainder_strides = compute_strides(remainder_shape);
    let data_strides = compute_strides(data_shape);
    let mut offsets = Vec::with_capacity(slice_len);
    for flat_idx in 0..slice_len {
        let mut rem = flat_idx;
        let mut offset = 0usize;
        for axis in 0..remainder_shape.len() {
            let stride = remainder_strides[axis];
            let coord = rem / stride;
            rem %= stride;
            offset += coord * data_strides[index_depth + axis];
        }
        offsets.push(offset);
    }
    Ok(offsets)
}

#[cfg(test)]
mod tests;
