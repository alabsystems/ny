// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-layer ny variables and INVPROP optimization state.

use ndarray::Array3;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::BTreeMap;

use super::OutputConstraints;

/// Per-layer ny variables for output constraint dualization.
///
/// In INVPROP, each layer selected by `apply_output_constraints_to` has
/// associated ny dual variables that are optimized to tighten bounds.
///
/// # Shape Convention
///
/// Following alpha,beta-CROWN/auto_LiRPA convention:
/// - Default: `(2, num_constraints, num_neurons)` - per-neuron gammas
/// - With `share_gammas`: `(2, num_constraints, 1)` - shared across neurons
///
/// The leading dimension `2` represents `(lower_bound_gammas, upper_bound_gammas)`.
///
/// # Constraint
///
/// Gammas must be non-negative: `ny >= 0`. This is enforced by clamping after
/// each optimization step via [`LayerGammas::clip`].
#[derive(Debug, Clone)]
pub struct LayerGammas {
    /// Ny values with shape `(2, num_constraints, num_neurons)` or
    /// `(2, num_constraints, 1)` when sharing.
    ///
    /// - `gammas[[0, c, n]]` = lower bound gamma for constraint c, neuron n
    /// - `gammas[[1, c, n]]` = upper bound gamma for constraint c, neuron n
    pub gammas: Array3<f32>,

    /// Whether gammas are active (being optimized) for this layer.
    pub active: bool,

    /// Whether gammas are shared across neurons (memory optimization).
    pub shared: bool,
}

impl LayerGammas {
    /// Create new layer gammas.
    ///
    /// # Arguments
    /// * `num_constraints` - Number of output constraints
    /// * `num_neurons` - Number of neurons in this layer
    /// * `share_gammas` - If true, share gammas across neurons
    ///
    /// Gammas are initialized to zero (inactive initially).
    pub fn new(num_constraints: usize, num_neurons: usize, share_gammas: bool) -> Self {
        let neuron_dim = if share_gammas { 1 } else { num_neurons };
        let gammas = Array3::zeros((2, num_constraints, neuron_dim));
        Self {
            gammas,
            active: true,
            shared: share_gammas,
        }
    }

    /// Create inactive (disabled) layer gammas.
    pub fn inactive() -> Self {
        Self {
            gammas: Array3::zeros((2, 0, 0)),
            active: false,
            shared: false,
        }
    }

    /// Clip gammas to enforce non-negativity: gamma >= 0
    ///
    /// Must be called after each optimization step.
    pub fn clip(&mut self) {
        self.gammas.mapv_inplace(|v| v.max(0.0));
    }

    /// Get the number of constraints.
    #[must_use]
    pub fn num_constraints(&self) -> usize {
        self.gammas.shape()[1]
    }

    /// Get the number of neurons (or 1 if shared).
    #[must_use]
    pub fn num_neurons(&self) -> usize {
        self.gammas.shape()[2]
    }

    /// Return the lower/upper gamma matrices when the public tensor still
    /// satisfies the `(2, constraints, neurons)` shape contract.
    ///
    /// `gammas` is public for optimizer integration, so callers can replace it
    /// with an arbitrary `Array3`. Production backward paths use this checked
    /// accessor and fail closed instead of indexing axis 0 at `1` and panicking
    /// on malformed external state.
    #[must_use]
    pub fn checked_bound_gammas(
        &self,
    ) -> Option<(ndarray::ArrayView2<'_, f32>, ndarray::ArrayView2<'_, f32>)> {
        if self.gammas.shape()[0] != 2 {
            return None;
        }
        Some((
            self.gammas.index_axis(ndarray::Axis(0), 0),
            self.gammas.index_axis(ndarray::Axis(0), 1),
        ))
    }

    /// Get lower bound gammas for all constraints and neurons.
    /// Shape: `[num_constraints, num_neurons]`
    pub fn lower_gammas(&self) -> ndarray::ArrayView2<'_, f32> {
        self.gammas.slice(ndarray::s![0, .., ..])
    }

    /// Get upper bound gammas for all constraints and neurons.
    /// Shape: `[num_constraints, num_neurons]`
    pub fn upper_gammas(&self) -> ndarray::ArrayView2<'_, f32> {
        self.gammas.slice(ndarray::s![1, .., ..])
    }

    /// Get mutable lower bound gammas.
    pub fn lower_gammas_mut(&mut self) -> ndarray::ArrayViewMut2<'_, f32> {
        self.gammas.slice_mut(ndarray::s![0, .., ..])
    }

    /// Get mutable upper bound gammas.
    pub fn upper_gammas_mut(&mut self) -> ndarray::ArrayViewMut2<'_, f32> {
        self.gammas.slice_mut(ndarray::s![1, .., ..])
    }

    /// Expand shared gammas to full neuron dimension.
    ///
    /// If gammas are shared (`num_neurons = 1`), this broadcasts them to
    /// `[num_constraints, target_neurons]`. If already per-neuron, returns as-is.
    #[must_use]
    pub fn expand_to(&self, target_neurons: usize) -> Array3<f32> {
        if !self.shared || self.gammas.shape()[2] == target_neurons {
            return self.gammas.clone();
        }
        // Broadcast from (2, num_constraints, 1) to (2, num_constraints, target_neurons)
        let (bound_dim, num_constraints, _) = self.gammas.dim();
        let mut expanded = Array3::zeros((bound_dim, num_constraints, target_neurons));
        for b in 0..bound_dim {
            for c in 0..num_constraints {
                let shared_val = self.gammas[[b, c, 0]];
                for n in 0..target_neurons {
                    expanded[[b, c, n]] = shared_val;
                }
            }
        }
        expanded
    }
}

/// State for INVPROP optimization across the network.
///
/// This structure holds all the ny variables for layers where INVPROP is applied,
/// along with the output constraints being propagated.
#[derive(Debug, Clone)]
pub struct InvpropState {
    /// Output constraints being propagated backward.
    pub constraints: OutputConstraints,

    /// Per-layer ny variables, keyed by layer name.
    /// Only layers in `apply_output_constraints_to` have entries.
    /// BTreeMap for O(log n) lookup by name and deterministic iteration order
    /// (was Vec with O(n) linear search).
    pub layer_gammas: BTreeMap<String, LayerGammas>,

    /// Per-batch infeasibility mask.
    /// `true` if the corresponding batch element is infeasible (lb > ub).
    pub infeasible_mask: Vec<bool>,
}

impl InvpropState {
    /// Create new INVPROP state from output constraints.
    pub fn new(constraints: OutputConstraints, batch_size: usize) -> Self {
        Self {
            constraints,
            layer_gammas: BTreeMap::new(),
            infeasible_mask: vec![false; batch_size],
        }
    }

    /// Add gammas for a layer.
    pub fn add_layer_gammas(&mut self, layer_name: String, gammas: LayerGammas) {
        self.layer_gammas.insert(layer_name, gammas);
    }

    /// Gammas for a layer by name. O(log n) via BTreeMap lookup.
    pub fn layer_gammas(&self, layer_name: &str) -> Option<&LayerGammas> {
        self.layer_gammas.get(layer_name)
    }

    /// Mutable gammas for a layer by name. O(log n) via BTreeMap lookup.
    pub fn layer_gammas_mut(&mut self, layer_name: &str) -> Option<&mut LayerGammas> {
        self.layer_gammas.get_mut(layer_name)
    }

    /// Clip all gammas to enforce non-negativity.
    pub fn clip_all_gammas(&mut self) {
        for gammas in self.layer_gammas.values_mut() {
            gammas.clip();
        }
    }

    /// Mark a batch element as infeasible.
    ///
    /// Returns `Err` if `batch_idx >= batch_size`.
    pub fn mark_infeasible(&mut self, batch_idx: usize) -> Result<()> {
        let batch_size = self.infeasible_mask.len();
        let mask = self.infeasible_mask.get_mut(batch_idx).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "batch_idx {} out of bounds for batch_size {}",
                batch_idx, batch_size
            ))
        })?;
        *mask = true;
        Ok(())
    }

    /// Check if a batch element is infeasible.
    #[must_use]
    pub fn is_infeasible(&self, batch_idx: usize) -> bool {
        self.infeasible_mask
            .get(batch_idx)
            .copied()
            .unwrap_or(false)
    }

    /// Apply the infeasible mask to bounds by setting lb=+inf and ub=-inf.
    ///
    /// When the bounds include a batch dimension matching the mask length, the
    /// mask is applied per batch element. Otherwise, any infeasible entry
    /// marks the entire bounds as infeasible.
    pub fn apply_infeasible_mask(&self, bounds: &mut BoundedTensor) {
        if !self.infeasible_mask.iter().any(|&v| v) {
            return;
        }

        let mask_len = self.infeasible_mask.len();
        if mask_len == 0 {
            return;
        }

        if mask_len == 1 {
            if self.infeasible_mask[0] {
                bounds.mark_infeasible_all();
            }
            return;
        }

        let batch_dim = bounds.lower().shape().first().copied();
        if batch_dim == Some(mask_len) {
            for (batch_idx, &infeasible) in self.infeasible_mask.iter().enumerate() {
                if infeasible {
                    if let Err(e) = bounds.mark_infeasible_at(0, batch_idx) {
                        tracing::warn!(
                            batch_idx,
                            mask_len,
                            error = %e,
                            "apply_infeasible_mask: mark_infeasible_at failed despite validated index; marking all infeasible"
                        );
                        bounds.mark_infeasible_all();
                        return;
                    }
                }
            }
        } else {
            bounds.mark_infeasible_all();
        }
    }

    /// Get all ny parameters as a flat vector for optimization.
    ///
    /// Returns `(values, indices)` where indices map back to layer/position.
    #[must_use]
    pub fn all_ny_params(&self) -> Vec<f32> {
        let mut params = Vec::new();
        for gammas in self.layer_gammas.values() {
            if gammas.active {
                params.extend(gammas.gammas.iter().copied());
            }
        }
        params
    }

    /// Update ny parameters from a flat vector (after optimization step).
    ///
    /// Returns `Err` if `params.len()` doesn't match total ny count from
    /// `all_ny_params`. (#2712: converted from assert_eq! to Result)
    pub fn update_ny_params(&mut self, params: &[f32]) -> Result<()> {
        let expected_len: usize = self
            .layer_gammas
            .values()
            .filter(|g| g.active)
            .map(|g| g.gammas.len())
            .sum();
        if params.len() != expected_len {
            return Err(NyError::InvalidSpec(format!(
                "update_ny_params: params length {} doesn't match expected ny count {}",
                params.len(),
                expected_len
            )));
        }

        let mut offset = 0;
        for gammas in self.layer_gammas.values_mut() {
            if gammas.active {
                let size = gammas.gammas.len();
                for (i, val) in gammas.gammas.iter_mut().enumerate() {
                    *val = params[offset + i];
                }
                offset += size;
            }
        }
        Ok(())
    }
}
