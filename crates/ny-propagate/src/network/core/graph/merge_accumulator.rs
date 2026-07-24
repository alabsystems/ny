// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ndarray::{Array, Dimension, Zip};
use ny_tensor::{next_down_f32, next_up_f32};
use tracing::warn;

use crate::bounds::patches::CrownBounds;
use crate::bounds::{LinearBounds, LinearBounds64};
use ny_core::{NyError, Result};

/// Vec-backed indexed storage for the hot backward loop.
/// When present, all operations use O(1) indexed access instead of HashMap lookups.
struct IndexedStorage {
    name_to_idx: HashMap<String, usize>,
    pending: Vec<Option<CrownBounds>>,
    merged_dense: Vec<Option<LinearBounds64>>,
    /// Reverse map from index to name, for drain().
    idx_to_name: Vec<String>,
}

/// Keeps single-parent nodes in their original carrier while merge points
/// accumulate dense bounds in f64 until the node is consumed.
///
/// Supports an optional indexed mode (`new_indexed`) that replaces HashMap
/// lookups with Vec index operations for the graph backward hot loop.
#[derive(Default)]
pub(crate) struct CrownMergeAccumulator {
    pending: HashMap<String, CrownBounds>,
    merged_dense: HashMap<String, LinearBounds64>,
    /// When Some, all operations use Vec-backed indexed storage.
    indexed: Option<IndexedStorage>,
}

impl CrownMergeAccumulator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Create an indexed accumulator for the graph backward hot loop.
    ///
    /// All node names in `exec_order` plus `NETWORK_INPUT` get assigned
    /// sequential indices. Operations use O(1) Vec access instead of
    /// HashMap lookups.
    pub(crate) fn new_indexed(exec_order: &[String]) -> Self {
        use super::NETWORK_INPUT;
        let capacity = exec_order.len() + 1; // +1 for NETWORK_INPUT
        let mut name_to_idx = HashMap::with_capacity(capacity);
        let mut idx_to_name = Vec::with_capacity(capacity);

        for (i, name) in exec_order.iter().enumerate() {
            name_to_idx.insert(name.clone(), i);
            idx_to_name.push(name.clone());
        }
        let ni_idx = exec_order.len();
        name_to_idx.insert(NETWORK_INPUT.to_string(), ni_idx);
        idx_to_name.push(NETWORK_INPUT.to_string());

        Self {
            pending: HashMap::new(),
            merged_dense: HashMap::new(),
            indexed: Some(IndexedStorage {
                name_to_idx,
                pending: vec![None; capacity],
                merged_dense: vec![None; capacity],
                idx_to_name,
            }),
        }
    }

    pub(crate) fn insert(&mut self, key: String, bounds: CrownBounds) {
        if let Some(ref mut idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(&key) {
                debug_assert!(
                    idx_store.pending[i].is_none() && idx_store.merged_dense[i].is_none(),
                    "duplicate CrownMergeAccumulator insert for key {key}",
                );
                idx_store.pending[i] = Some(bounds);
                return;
            }
            // Fall through to HashMap for keys not in exec_order (shouldn't happen normally)
        }
        debug_assert!(
            !self.pending.contains_key(&key) && !self.merged_dense.contains_key(&key),
            "duplicate CrownMergeAccumulator insert for key {key}",
        );
        self.pending.insert(key, bounds);
    }

    pub(crate) fn contains_key(&self, key: &str) -> bool {
        if let Some(ref idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(key) {
                return idx_store.pending[i].is_some() || idx_store.merged_dense[i].is_some();
            }
        }
        self.pending.contains_key(key) || self.merged_dense.contains_key(key)
    }

    pub(crate) fn is_empty(&self) -> bool {
        if let Some(ref idx_store) = self.indexed {
            return idx_store.pending.iter().all(Option::is_none)
                && idx_store.merged_dense.iter().all(Option::is_none)
                && self.pending.is_empty()
                && self.merged_dense.is_empty();
        }
        self.pending.is_empty() && self.merged_dense.is_empty()
    }

    pub(crate) fn has_only_key(&self, key: &str) -> bool {
        if let Some(ref idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(key) {
                let has_this =
                    idx_store.pending[i].is_some() || idx_store.merged_dense[i].is_some();
                if !has_this {
                    return false;
                }
                let total_indexed = idx_store.pending.iter().filter(|x| x.is_some()).count()
                    + idx_store
                        .merged_dense
                        .iter()
                        .filter(|x| x.is_some())
                        .count();
                let total_hash = self.pending.len() + self.merged_dense.len();
                return total_indexed + total_hash == 1;
            }
        }
        self.pending.len() + self.merged_dense.len() == 1 && self.contains_key(key)
    }

    pub(crate) fn take(&mut self, key: &str) -> Result<Option<CrownBounds>> {
        if let Some(ref mut idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(key) {
                return Self::take_from_vecs(
                    &mut idx_store.pending,
                    &mut idx_store.merged_dense,
                    i,
                );
            }
        }
        let pending = self.pending.remove(key);
        let merged = self.merged_dense.remove(key);
        debug_assert!(
            pending.is_none() || merged.is_none(),
            "CrownMergeAccumulator key {key} existed in both stores",
        );
        if let Some(bounds) = pending {
            return Ok(Some(bounds));
        }
        Ok(merged.map(|dense| CrownBounds::Dense(Self::downcast_dense(dense))))
    }

    /// Direct-index take for the hot loop where the caller already knows the index.
    /// Avoids the name_to_idx HashMap lookup.
    #[inline]
    pub(crate) fn take_by_idx(&mut self, idx: usize) -> Result<Option<CrownBounds>> {
        if let Some(ref mut idx_store) = self.indexed {
            return Self::take_from_vecs(&mut idx_store.pending, &mut idx_store.merged_dense, idx);
        }
        Err(NyError::InvalidSpec(
            "take_by_idx called on non-indexed CrownMergeAccumulator".to_string(),
        ))
    }

    fn take_from_vecs(
        pending: &mut [Option<CrownBounds>],
        merged_dense: &mut [Option<LinearBounds64>],
        i: usize,
    ) -> Result<Option<CrownBounds>> {
        let p = pending[i].take();
        let m = merged_dense[i].take();
        debug_assert!(
            p.is_none() || m.is_none(),
            "CrownMergeAccumulator indexed slot {i} existed in both stores",
        );
        if let Some(bounds) = p {
            return Ok(Some(bounds));
        }
        Ok(m.map(|dense| CrownBounds::Dense(Self::downcast_dense(dense))))
    }

    pub(crate) fn drain(&mut self) -> Vec<(String, CrownBounds)> {
        let mut drained = Vec::new();
        if let Some(ref mut idx_store) = self.indexed {
            for (i, bounds) in idx_store.pending.iter_mut().enumerate() {
                if let Some(b) = bounds.take() {
                    drained.push((idx_store.idx_to_name[i].clone(), b));
                }
            }
            for (i, dense) in idx_store.merged_dense.iter_mut().enumerate() {
                if let Some(d) = dense.take() {
                    drained.push((
                        idx_store.idx_to_name[i].clone(),
                        CrownBounds::Dense(Self::downcast_dense(d)),
                    ));
                }
            }
        }
        drained.extend(self.pending.drain());
        drained.extend(
            self.merged_dense
                .drain()
                .map(|(key, dense)| (key, CrownBounds::Dense(Self::downcast_dense(dense)))),
        );
        drained
    }

    /// Merge a CrownBounds contribution, preserving patches when compatible.
    ///
    /// Policy:
    /// - Patches + Patches (compatible): merge in-place in `pending`
    /// - Patches + Patches (incompatible): convert both to dense, route to f64
    /// - Dense + Dense: route to existing `merge_dense`
    /// - Mixed Dense/Patches: convert both to dense, route to f64
    ///
    /// Part of #4382: patches-native residual merge for CNN DAGs.
    pub(crate) fn merge_crown(&mut self, key: &str, new_bounds: CrownBounds) -> Result<()> {
        if let CrownBounds::Patches(ref new_pb) = new_bounds {
            if self.try_patches_merge(key, new_pb)? {
                return Ok(());
            }
        }
        let new_lb = new_bounds.into_dense()?;
        self.merge_dense(key, new_lb)
    }

    fn try_patches_merge(
        &mut self,
        key: &str,
        new_pb: &crate::bounds::patches::PatchesLinearBounds,
    ) -> Result<bool> {
        if let Some(ref mut idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(key) {
                if let Some(CrownBounds::Patches(ref mut existing)) = idx_store.pending[i] {
                    if existing.try_merge_inplace(new_pb)? {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
        }
        if let Some(CrownBounds::Patches(ref mut existing)) = self.pending.get_mut(key) {
            if existing.try_merge_inplace(new_pb)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn merge_dense(&mut self, key: &str, new_bounds: LinearBounds) -> Result<()> {
        if let Some(ref mut idx_store) = self.indexed {
            if let Some(&i) = idx_store.name_to_idx.get(key) {
                if let Some(ref mut existing) = idx_store.merged_dense[i] {
                    Self::accumulate_linear_bounds64(existing, &new_bounds);
                    return Ok(());
                }
                if let Some(existing_bounds) = idx_store.pending[i].take() {
                    let mut accumulator = LinearBounds64::from_f32(&existing_bounds.into_dense()?);
                    Self::accumulate_linear_bounds64(&mut accumulator, &new_bounds);
                    idx_store.merged_dense[i] = Some(accumulator);
                    return Ok(());
                }
                let _ = new_bounds;
                return Err(NyError::InvalidSpec(format!(
                    "CrownMergeAccumulator merge expected existing entry for key {key}",
                )));
            }
        }

        if let Some(existing) = self.merged_dense.get_mut(key) {
            Self::accumulate_linear_bounds64(existing, &new_bounds);
            return Ok(());
        }

        if let Some(existing_bounds) = self.pending.remove(key) {
            let mut accumulator = LinearBounds64::from_f32(&existing_bounds.into_dense()?);
            Self::accumulate_linear_bounds64(&mut accumulator, &new_bounds);
            self.merged_dense.insert(key.to_string(), accumulator);
            return Ok(());
        }

        let _ = new_bounds;
        Err(NyError::InvalidSpec(format!(
            "CrownMergeAccumulator merge expected existing entry for key {key}",
        )))
    }

    fn downcast_dense(bounds: LinearBounds64) -> LinearBounds {
        // Read the carried certified coefficient error (f64) BEFORE consuming the
        // struct (#vnncomp-aw-soundness). `into_parts` only returns the 4 numeric
        // components; the error matrices must be captured here or the merged
        // DAG-residual error would be silently dropped at the f64→f32 downcast.
        let has_err = bounds.lower_a_err().is_some() || bounds.upper_a_err().is_some();
        let lower_err_f64 = bounds.lower_a_err().cloned();
        let upper_err_f64 = bounds.upper_a_err().cloned();
        let (lower_a, lower_b, upper_a, upper_b) = bounds.into_parts();
        let (num_outputs, num_inputs) = (lower_a.nrows(), lower_a.ncols());
        let mut dense = LinearBounds::conservative(num_outputs, num_inputs);

        // Per-coefficient output error: carried f64 error + the f64→f32 cast gap,
        // rounded UP to a sound f32. Only allocated when there is error to carry.
        let mut out_lower_err =
            has_err.then(|| ndarray::Array2::<f32>::zeros((num_outputs, num_inputs)));
        let mut out_upper_err = out_lower_err.clone();

        for row in 0..num_outputs {
            let Some((lower_row, lower_bias, upper_row, upper_bias)) =
                Self::try_downcast_row(&lower_a, &lower_b, &upper_a, &upper_b, row, num_inputs)
            else {
                warn!(
                    row,
                    "CrownMergeAccumulator f64->f32 row downcast failed; returning conservative row"
                );
                // A degraded row: mark the error as +inf so concretize widens it
                // (the conservative row already has ±inf bias, but keep the error
                // consistent in case a downstream op reads it before concretize).
                if let Some(le) = out_lower_err.as_mut() {
                    for col in 0..num_inputs {
                        le[[row, col]] = f32::INFINITY;
                    }
                }
                if let Some(ue) = out_upper_err.as_mut() {
                    for col in 0..num_inputs {
                        ue[[row, col]] = f32::INFINITY;
                    }
                }
                continue;
            };

            for (col, value) in lower_row.into_iter().enumerate() {
                if let Some(le) = out_lower_err.as_mut() {
                    let carried = lower_err_f64.as_ref().map(|e| e[[row, col]]).unwrap_or(0.0);
                    let cast_gap = (value as f64 - lower_a[[row, col]]).abs();
                    le[[row, col]] = Self::err_to_f32(carried + cast_gap);
                }
                dense.lower_a_mut()[[row, col]] = value;
            }
            dense.lower_b_mut()[row] = lower_bias;
            for (col, value) in upper_row.into_iter().enumerate() {
                if let Some(ue) = out_upper_err.as_mut() {
                    let carried = upper_err_f64.as_ref().map(|e| e[[row, col]]).unwrap_or(0.0);
                    let cast_gap = (value as f64 - upper_a[[row, col]]).abs();
                    ue[[row, col]] = Self::err_to_f32(carried + cast_gap);
                }
                dense.upper_a_mut()[[row, col]] = value;
            }
            dense.upper_b_mut()[row] = upper_bias;
        }

        if let (Some(le), Some(ue)) = (out_lower_err, out_upper_err) {
            dense.set_coeff_err(le, ue);
        }
        dense
    }

    /// Round a non-negative f64 error magnitude UP to a sound f32 error
    /// (over-approximation is always sound; under-approximation is not).
    /// A non-finite or negative value becomes `f32::INFINITY` so the affected
    /// row degrades to `[-inf, +inf]` at concretize.
    #[inline]
    fn err_to_f32(e: f64) -> f32 {
        if !e.is_finite() || e < 0.0 {
            return f32::INFINITY;
        }
        let cast = e as f32;
        // `as f32` rounds to nearest, which may round the magnitude DOWN; widen
        // outward so the stored f32 error is never below the true f64 magnitude.
        let up = next_up_f32(cast);
        if up.is_finite() {
            up
        } else {
            f32::INFINITY
        }
    }

    fn try_downcast_row(
        lower_a: &ndarray::Array2<f64>,
        lower_b: &ndarray::Array1<f64>,
        upper_a: &ndarray::Array2<f64>,
        upper_b: &ndarray::Array1<f64>,
        row: usize,
        num_inputs: usize,
    ) -> Option<(Vec<f32>, f32, Vec<f32>, f32)> {
        let lower_bias = Self::downcast_lower_bias(lower_b[row])?;
        let upper_bias = Self::downcast_upper_bias(upper_b[row])?;
        let mut lower_row = Vec::with_capacity(num_inputs);
        let mut upper_row = Vec::with_capacity(num_inputs);

        for col in 0..num_inputs {
            lower_row.push(Self::downcast_lower_coeff(lower_a[[row, col]])?);
            upper_row.push(Self::downcast_upper_coeff(upper_a[[row, col]])?);
        }

        Some((lower_row, lower_bias, upper_row, upper_bias))
    }

    fn downcast_lower_coeff(value: f64) -> Option<f32> {
        Self::downcast_coeff(value, true)
    }

    fn downcast_upper_coeff(value: f64) -> Option<f32> {
        Self::downcast_coeff(value, false)
    }

    fn downcast_coeff(value: f64, is_lower: bool) -> Option<f32> {
        if !value.is_finite() {
            return None;
        }

        let cast = value as f32;
        if !cast.is_finite() {
            return None;
        }

        Some(if is_lower {
            next_down_f32(cast)
        } else {
            next_up_f32(cast)
        })
    }

    fn downcast_lower_bias(value: f64) -> Option<f32> {
        if value == f64::NEG_INFINITY {
            return Some(f32::NEG_INFINITY);
        }
        Self::downcast_coeff(value, true)
    }

    fn downcast_upper_bias(value: f64) -> Option<f32> {
        if value == f64::INFINITY {
            return Some(f32::INFINITY);
        }
        Self::downcast_coeff(value, false)
    }

    fn accumulate_linear_bounds64(existing: &mut LinearBounds64, new_bounds: &LinearBounds) {
        if existing.num_outputs() != new_bounds.num_outputs()
            || existing.num_inputs() != new_bounds.num_inputs()
            || existing.lower_b().len() != new_bounds.lower_b().len()
            || existing.upper_b().len() != new_bounds.upper_b().len()
        {
            warn!(
                existing_shape = ?existing.lower_a().shape(),
                new_shape = ?new_bounds.lower_a().shape(),
                existing_lower_bias = existing.lower_b().len(),
                new_lower_bias = new_bounds.lower_b().len(),
                existing_upper_bias = existing.upper_b().len(),
                new_upper_bias = new_bounds.upper_b().len(),
                "CrownMergeAccumulator shape mismatch; widening accumulator to infinities"
            );
            Self::widen_to_infinities(existing);
            return;
        }

        // Accumulate the coefficients in f64, capturing the per-element f64
        // accumulation roundoff so we can fold it into the certified error
        // (#vnncomp-aw-soundness). f32→f64 widening of `new` is exact, so the only
        // new error introduced by `existing + new` is the single f64 add's
        // roundoff, bounded by 2^-53·|sum|. We add that OUTWARD into the error
        // accumulator below, alongside the incoming contribution's certified
        // coefficient error — neither may be silently dropped at a DAG merge.
        let lower_roundoff =
            Self::accumulate_coeff_array(existing.lower_a_mut(), new_bounds.lower_a());
        let upper_roundoff =
            Self::accumulate_coeff_array(existing.upper_a_mut(), new_bounds.upper_a());
        Self::accumulate_array(existing.lower_b_mut(), new_bounds.lower_b(), true);
        Self::accumulate_array(existing.upper_b_mut(), new_bounds.upper_b(), false);

        // Carry the certified coefficient error: existing_err + new_err + roundoff,
        // all accumulated in f64 (non-negative, so summation rounds harmlessly; we
        // add a final cast widening at downcast time). The incoming f32 error is
        // widened to f64 exactly. If either side carries no error it contributes 0.
        let n_out = existing.num_outputs();
        let n_in = existing.num_inputs();
        Self::accumulate_err(
            &mut existing.lower_a_err,
            new_bounds.lower_a_err(),
            &lower_roundoff,
            n_out,
            n_in,
        );
        Self::accumulate_err(
            &mut existing.upper_a_err,
            new_bounds.upper_a_err(),
            &upper_roundoff,
            n_out,
            n_in,
        );
    }

    /// Accumulate `existing += new` (f32→f64 exact widening) element-wise with the
    /// NaN→±inf firewall, returning the per-element f64 add roundoff bound
    /// `2^-53·|result|` (0 where the result is non-finite — those rows degrade via
    /// the bias/err). This is the coefficient-array analogue of
    /// [`accumulate_array`] that additionally reports the introduced roundoff so
    /// it can be folded into the certified coefficient error.
    fn accumulate_coeff_array(
        existing: &mut Array<f64, ndarray::Ix2>,
        new: &Array<f32, ndarray::Ix2>,
    ) -> Array<f64, ndarray::Ix2> {
        // Unit roundoff for f64 round-to-nearest.
        const U_F64: f64 = 1.1102230246251565e-16; // 2^-53
        let mut roundoff = Array::<f64, _>::zeros(existing.raw_dim());
        Zip::from(existing).and(new).and(&mut roundoff).for_each(
            |existing_value, &new_value, ro| {
                if existing_value.is_nan() || new_value.is_nan() {
                    // A NaN coefficient is never sound; widen the accumulator
                    // coefficient and let concretize degrade the row.
                    *existing_value = f64::NAN;
                    *ro = 0.0;
                    return;
                }
                let sum = *existing_value + new_value as f64;
                if sum.is_finite() {
                    // |fl(a+b) - (a+b)| <= u·|fl(a+b)| (round-to-nearest).
                    *ro = U_F64 * sum.abs();
                    *existing_value = sum;
                } else {
                    // Inf coefficient: row will degrade at concretize; no finite
                    // roundoff to carry.
                    *ro = 0.0;
                    *existing_value = sum;
                }
            },
        );
        roundoff
    }

    /// Accumulate the certified coefficient error in f64:
    /// `existing_err += new_err + roundoff`. `new_err` is the f32 incoming error
    /// (exact f32→f64 widening); `roundoff` is the f64 add roundoff bound from
    /// [`accumulate_coeff_array`]. The result is allocated lazily: if there is no
    /// error to carry (both sides None and roundoff all-zero) `existing_err` stays
    /// `None` (exact). All entries stay non-negative; a non-finite entry marks the
    /// row for degradation at downcast/concretize.
    fn accumulate_err(
        existing_err: &mut Option<ndarray::Array2<f64>>,
        new_err: Option<&ndarray::Array2<f32>>,
        roundoff: &Array<f64, ndarray::Ix2>,
        n_out: usize,
        n_in: usize,
    ) {
        let new_has = new_err.is_some();
        let roundoff_has = roundoff.iter().any(|&v| v != 0.0);
        if existing_err.is_none() && !new_has && !roundoff_has {
            // Nothing to carry; keep exact.
            return;
        }
        let acc = existing_err.get_or_insert_with(|| ndarray::Array2::<f64>::zeros((n_out, n_in)));
        if acc.shape() != [n_out, n_in] {
            // Shape drift (should not happen): degrade to a fully-degraded error so
            // concretize widens every row rather than under-counting.
            *acc = ndarray::Array2::<f64>::from_elem((n_out, n_in), f64::INFINITY);
            return;
        }
        Zip::from(acc).and(roundoff).for_each(|a, &ro| {
            let s = *a + ro;
            *a = if s.is_nan() { f64::INFINITY } else { s };
        });
        if let Some(ne) = new_err {
            if ne.shape() == [n_out, n_in] {
                let acc = existing_err.as_mut().expect("allocated above");
                Zip::from(acc).and(ne).for_each(|a, &e| {
                    let e = e as f64;
                    let s = *a + e;
                    *a = if s.is_nan() || !e.is_finite() {
                        f64::INFINITY
                    } else {
                        s
                    };
                });
            } else {
                // Incoming error shape mismatch: cannot map it soundly; degrade.
                *existing_err = Some(ndarray::Array2::<f64>::from_elem(
                    (n_out, n_in),
                    f64::INFINITY,
                ));
            }
        }
    }

    fn widen_to_infinities(existing: &mut LinearBounds64) {
        *existing.lower_a_mut() = Array::from_elem(existing.lower_a().raw_dim(), f64::NEG_INFINITY);
        *existing.lower_b_mut() = Array::from_elem(existing.lower_b().raw_dim(), f64::NEG_INFINITY);
        *existing.upper_a_mut() = Array::from_elem(existing.upper_a().raw_dim(), f64::INFINITY);
        *existing.upper_b_mut() = Array::from_elem(existing.upper_b().raw_dim(), f64::INFINITY);
    }

    fn accumulate_array<D: Dimension>(
        existing: &mut Array<f64, D>,
        new: &Array<f32, D>,
        is_lower: bool,
    ) {
        let nan_fallback = if is_lower {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        Zip::from(existing)
            .and(new)
            .for_each(|existing_value, &new_value| {
                if existing_value.is_nan() || new_value.is_nan() {
                    *existing_value = nan_fallback;
                    return;
                }
                let sum = *existing_value + new_value as f64;
                *existing_value = if sum.is_nan() { nan_fallback } else { sum };
            });
    }
}

#[cfg(test)]
#[path = "merge_accumulator_tests.rs"]
mod tests;
