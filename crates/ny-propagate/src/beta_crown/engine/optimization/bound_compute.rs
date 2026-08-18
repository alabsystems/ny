// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bound-computation methods used by β-CROWN optimization.

use super::super::BetaCrownVerifier;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use std::sync::Arc;

use crate::beta_crown::bab_cuts::CutPool;
use crate::beta_crown::branching::SplitHistory;
use crate::beta_crown::domain::IntermediateLinearBounds;
use crate::beta_crown::state::{BetaState, DomainAlphaState};
use crate::{LinearBounds, Network};

impl BetaCrownVerifier {
    /// Compute output bounds incorporating split constraints.
    ///
    /// This is the core of β-CROWN: it modifies the CROWN backward pass
    /// to incorporate the β parameters for constrained neurons.
    #[cfg(test)]
    pub(crate) fn compute_bounds_with_constraints(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &BetaState,
    ) -> Result<BoundedTensor> {
        if network.layers.is_empty() {
            return Ok(input.clone());
        }

        self.validate_layer_bounds_len(network, layer_bounds)?;
        self.validate_split_history(network, input, layer_bounds, history)?;

        // Build constraint lookup: layer_idx -> neuron_idx -> is_active
        let mut constraints: std::collections::HashMap<
            usize,
            std::collections::HashMap<usize, bool>,
        > = std::collections::HashMap::new();
        for c in &history.constraints {
            constraints
                .entry(c.layer_idx)
                .or_default()
                .insert(c.neuron_idx, c.is_active);
        }

        // Start with identity linear bounds for output
        let output_dim =
            self.output_dim_from_layer_bounds(layer_bounds, "compute_bounds_with_constraints")?;
        let mut lin_bounds = LinearBounds::identity(output_dim);

        // Backward pass through layers
        for (layer_idx, layer) in network.layers.iter().enumerate().rev() {
            // Use references instead of clones (Arc derefs to inner BoundedTensor)
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                input
            } else {
                layer_bounds[layer_idx - 1].as_ref()
            };

            // Check if this layer has constraints
            let layer_constraints = constraints.get(&layer_idx);

            lin_bounds = self.propagate_layer_backward_with_beta(
                layer,
                &lin_bounds,
                pre_bounds,
                layer_constraints,
                beta_state,
                layer_idx,
                None,
            )?;
        }

        // Concretize with input bounds (#2239: directed rounding for soundness).
        Ok(lin_bounds.concretize_sound(input))
    }

    /// Compute output bounds using α, β, and λ (cut) parameters.
    ///
    /// This extends `compute_bounds_with_constraints` to use optimizable α values
    /// for unstable neurons instead of the heuristic.
    // Justification: β-CROWN bound computation requires network, input, split history,
    // layer bounds, and alpha/beta/cut state — the full BaB verification context.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_bounds_with_alpha_beta(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &BetaState,
        alpha_state: &DomainAlphaState,
        cut_pool: &CutPool,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        if network.layers.is_empty() {
            return Ok(input.clone());
        }

        self.validate_layer_bounds_len(network, layer_bounds)?;
        self.validate_split_history(network, input, layer_bounds, history)?;

        // Build constraint lookup: layer_idx -> neuron_idx -> is_active
        let mut constraints: std::collections::HashMap<
            usize,
            std::collections::HashMap<usize, bool>,
        > = std::collections::HashMap::new();
        for c in &history.constraints {
            constraints
                .entry(c.layer_idx)
                .or_default()
                .insert(c.neuron_idx, c.is_active);
        }

        // Start with identity linear bounds for output
        let output_dim =
            self.output_dim_from_layer_bounds(layer_bounds, "compute_bounds_with_alpha_beta")?;
        let mut lin_bounds = LinearBounds::identity(output_dim);

        // CUT QUARANTINE: `cut_pool` is deliberately NOT consulted. Both folds
        // that once read it are gone — the post-hoc scalar contribution (deleted
        // in 28d1fbeb) and the `arelu_cut` backward integration (deleted here).
        // The parameter is retained on purpose: it is the entry point the
        // `test_quarantined_cut_authority_does_not_modify_*_2422` regressions
        // push a POPULATED pool through to assert the bounds are unchanged.
        // Do NOT consume it without a proof-producing, outward-rounded fold —
        // see `BetaCrownConfig::cut_proof_authority_enabled()`.
        let _ = cut_pool;

        // Backward pass through layers
        for (layer_idx, layer) in network.layers.iter().enumerate().rev() {
            // Use references instead of clones (Arc derefs to inner BoundedTensor)
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                input
            } else {
                layer_bounds[layer_idx - 1].as_ref()
            };

            // Check if this layer has constraints
            let layer_constraints = constraints.get(&layer_idx);

            lin_bounds = self.propagate_layer_backward_with_alpha_beta(
                layer,
                &lin_bounds,
                pre_bounds,
                layer_constraints,
                beta_state,
                alpha_state,
                layer_idx,
                engine,
            )?;
        }

        // Concretize with input bounds (#2239: directed rounding for soundness).
        //
        // No post-hoc cut term is added here. The legacy GCP-CROWN "scalar
        // contribution after concretization" fold was deleted: it was never a
        // certified GCP-CROWN fold (it was applied outside the backward
        // relaxation) and `BetaCrownConfig::cut_proof_authority_enabled()` had
        // already made it statically unreachable.
        Ok(lin_bounds.concretize_sound(input))
    }

    /// Compute output bounds while capturing intermediate linear bounds.
    ///
    /// This is identical to `compute_bounds_with_alpha_beta` but additionally
    /// returns the LinearBounds at each layer during the backward pass. These
    /// intermediate bounds enable efficient bound transfer to child domains.
    ///
    /// Returns:
    /// - Output bounds (concrete BoundedTensor)
    /// - Intermediate linear bounds at each layer (for transfer to children)
    // Justification: Same parameter set as compute_bounds_with_alpha_beta — network,
    // input, history, layer bounds, alpha/beta/cut state, engine — plus intermediate capture.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_bounds_capturing_intermediate(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &BetaState,
        alpha_state: &DomainAlphaState,
        cut_pool: &CutPool,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(BoundedTensor, IntermediateLinearBounds)> {
        self.compute_bounds_capturing_intermediate_inner(
            network,
            input,
            history,
            layer_bounds,
            beta_state,
            alpha_state,
            cut_pool,
            engine,
        )
    }

    /// Inner implementation for beta-CROWN bound computation with intermediate capture.
    ///
    /// Always uses `concretize_sound()` for directed rounding on f64→f32 (#2239).
    // Justification: 9 parameters is the minimum for the beta-CROWN backward pass
    // (network, input, history, layer bounds, beta/alpha/cut state, engine).
    #[allow(clippy::too_many_arguments)]
    fn compute_bounds_capturing_intermediate_inner(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &BetaState,
        alpha_state: &DomainAlphaState,
        cut_pool: &CutPool,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(BoundedTensor, IntermediateLinearBounds)> {
        if network.layers.is_empty() {
            return Ok((input.clone(), IntermediateLinearBounds::empty()));
        }

        self.validate_layer_bounds_len(network, layer_bounds)?;
        self.validate_split_history(network, input, layer_bounds, history)?;

        // Build constraint lookup: layer_idx -> neuron_idx -> is_active
        let mut constraints: std::collections::HashMap<
            usize,
            std::collections::HashMap<usize, bool>,
        > = std::collections::HashMap::new();
        for c in &history.constraints {
            constraints
                .entry(c.layer_idx)
                .or_default()
                .insert(c.neuron_idx, c.is_active);
        }

        // Start with identity linear bounds for output
        let output_dim = self
            .output_dim_from_layer_bounds(layer_bounds, "compute_bounds_capturing_intermediate")?;
        let mut lin_bounds = LinearBounds::identity(output_dim);
        let num_layers = network.layers.len();

        // Storage for intermediate bounds at each layer.
        // bounds_at_layer[i] holds the LinearBounds BEFORE processing layer i in the backward pass.
        // For example:
        //   bounds_at_layer[num_layers-1] = identity (initial state at output)
        //   bounds_at_layer[i] = state after processing layers num_layers-1 down to i+1
        // This allows partial backward pass: to recompute from layer L, start with
        // bounds_at_layer[L] and process layers L down to 0.
        let mut intermediate_bounds: Vec<Arc<LinearBounds>> = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            intermediate_bounds.push(Arc::new(lin_bounds.clone())); // Initialize with identity
        }

        // CUT QUARANTINE: `cut_pool` is deliberately NOT consulted.
        // See `compute_bounds_with_alpha_beta` for the full rationale.
        let _ = cut_pool;

        // Backward pass through layers
        for (layer_idx, layer) in network.layers.iter().enumerate().rev() {
            // Store bounds BEFORE processing this layer (i.e., the current lin_bounds)
            intermediate_bounds[layer_idx] = Arc::new(lin_bounds.clone());

            // Use references instead of clones (Arc derefs to inner BoundedTensor)
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                input
            } else {
                layer_bounds[layer_idx - 1].as_ref()
            };

            // Check if this layer has constraints
            let layer_constraints = constraints.get(&layer_idx);

            lin_bounds = self.propagate_layer_backward_with_alpha_beta(
                layer,
                &lin_bounds,
                pre_bounds,
                layer_constraints,
                beta_state,
                alpha_state,
                layer_idx,
                engine,
            )?;
        }

        // #2239: Always use directed rounding on f64→f32 for soundness.
        // Matches alpha-beta-CROWN's __double2float_rd/__double2float_ru.
        // (No post-hoc cut term — see `compute_bounds_with_alpha_beta`.)
        let output_bounds = lin_bounds.concretize_sound(input);

        let intermediate = IntermediateLinearBounds {
            bounds_at_layer: intermediate_bounds,
            start_layer: num_layers - 1,
        };

        Ok((output_bounds, intermediate))
    }

    /// Compute output bounds starting from a given layer using parent's intermediate bounds.
    ///
    /// Uses sound directed rounding (`concretize_sound`) for the final f64→f32 cast.
    /// All concretization in the inner implementation is sound since #2239.
    ///
    /// This is the key optimization for intermediate bound transfer: instead of running
    /// a full backward pass from the output layer, we start from `start_layer` using
    /// the parent domain's intermediate bounds for layers after `start_layer`.
    ///
    /// When splitting at layer L, we can:
    /// - Reuse parent's intermediate bounds for layers L+1 to num_layers-1
    /// - Only propagate backward from layer L to input
    /// - This saves (num_layers - L - 1) layer propagations
    ///
    /// # Arguments
    /// - `start_layer`: The layer index to start backward propagation from
    /// - `parent_intermediate`: Parent domain's intermediate bounds (bounds_at_layer[L] gives
    ///   the LinearBounds BEFORE processing layer L)
    ///
    /// # Returns
    /// - Output bounds (concrete BoundedTensor)
    /// - New intermediate linear bounds (with layers > start_layer copied from parent)
    // Justification: Incremental bound computation from a split layer needs the full
    // verification context plus start_layer and parent intermediate bounds.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_bounds_from_layer(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &BetaState,
        alpha_state: &DomainAlphaState,
        cut_pool: &CutPool,
        start_layer: usize,
        parent_intermediate: &IntermediateLinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(BoundedTensor, IntermediateLinearBounds)> {
        self.compute_bounds_from_layer_inner(
            network,
            input,
            history,
            layer_bounds,
            beta_state,
            alpha_state,
            cut_pool,
            start_layer,
            parent_intermediate,
            engine,
        )
    }

    /// Inner implementation for `compute_bounds_from_layer`.
    // Justification: Shared implementation avoids duplicating the backward
    // pass, intermediate transfer, and GCP-CROWN logic.
    #[allow(clippy::too_many_arguments)]
    fn compute_bounds_from_layer_inner(
        &self,
        network: &Network,
        input: &BoundedTensor,
        history: &SplitHistory,
        layer_bounds: &[Arc<BoundedTensor>],
        beta_state: &BetaState,
        alpha_state: &DomainAlphaState,
        cut_pool: &CutPool,
        start_layer: usize,
        parent_intermediate: &IntermediateLinearBounds,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<(BoundedTensor, IntermediateLinearBounds)> {
        let num_layers = network.layers.len();

        // Validate start_layer
        if start_layer >= num_layers {
            return Err(ny_core::NyError::InvalidSpec(format!(
                "start_layer {} >= num_layers {}",
                start_layer, num_layers
            )));
        }

        // Fall back to full computation if parent intermediates are unusable for
        // transfer (empty/output start/shape mismatch).
        if parent_intermediate.is_empty()
            || start_layer >= num_layers - 1
            || parent_intermediate.bounds_at_layer.len() != num_layers
        {
            return self.compute_bounds_capturing_intermediate_inner(
                network,
                input,
                history,
                layer_bounds,
                beta_state,
                alpha_state,
                cut_pool,
                engine,
            );
        }

        self.validate_layer_bounds_len(network, layer_bounds)?;
        self.validate_split_history(network, input, layer_bounds, history)?;

        // Build constraint lookup: layer_idx -> neuron_idx -> is_active
        let mut constraints: std::collections::HashMap<
            usize,
            std::collections::HashMap<usize, bool>,
        > = std::collections::HashMap::new();
        for c in &history.constraints {
            constraints
                .entry(c.layer_idx)
                .or_default()
                .insert(c.neuron_idx, c.is_active);
        }

        // Initialize intermediate bounds storage
        // Copy parent's bounds for layers > start_layer (they won't be recomputed)
        let mut intermediate_bounds: Vec<Arc<LinearBounds>> = Vec::with_capacity(num_layers);
        let output_dim = self.output_dim_from_layer_bounds(
            layer_bounds,
            "compute_bounds_from_layer(placeholder initialization)",
        )?;
        for layer_idx in 0..num_layers {
            if layer_idx > start_layer {
                // Copy from parent - these layers are unchanged
                intermediate_bounds
                    .push(Arc::clone(&parent_intermediate.bounds_at_layer[layer_idx]));
            } else {
                // Will be computed below - use placeholder for now
                intermediate_bounds.push(Arc::new(LinearBounds::identity(output_dim)));
            }
        }

        // Start from parent's intermediate bounds at start_layer
        // bounds_at_layer[L] = LinearBounds BEFORE processing layer L
        // So we start with the state that was used to process layer L in parent
        let mut lin_bounds = match parent_intermediate.get(start_layer) {
            Some(lb) => lb.clone(),
            None => {
                // Parent doesn't have bounds at start_layer, fall back to full computation
                return self.compute_bounds_capturing_intermediate_inner(
                    network,
                    input,
                    history,
                    layer_bounds,
                    beta_state,
                    alpha_state,
                    cut_pool,
                    engine,
                );
            }
        };

        // CUT QUARANTINE: `cut_pool` is deliberately NOT consulted here either;
        // it is only forwarded to the full-recompute fallbacks above.
        // See `compute_bounds_with_alpha_beta` for the full rationale.

        // Backward pass through layers 0..=start_layer only
        // We process in reverse order: start_layer, start_layer-1, ..., 0
        for layer_idx in (0..=start_layer).rev() {
            let layer = &network.layers[layer_idx];

            // Store bounds BEFORE processing this layer
            intermediate_bounds[layer_idx] = Arc::new(lin_bounds.clone());

            // Use references instead of clones
            let pre_bounds: &BoundedTensor = if layer_idx == 0 {
                input
            } else {
                layer_bounds[layer_idx - 1].as_ref()
            };

            // Check if this layer has constraints
            let layer_constraints = constraints.get(&layer_idx);

            lin_bounds = self.propagate_layer_backward_with_alpha_beta(
                layer,
                &lin_bounds,
                pre_bounds,
                layer_constraints,
                beta_state,
                alpha_state,
                layer_idx,
                engine,
            )?;
        }

        // #2239: Always use directed rounding on f64→f32 for soundness.
        // (No post-hoc cut term — see `compute_bounds_with_alpha_beta`.)
        let output_bounds = lin_bounds.concretize_sound(input);

        let intermediate = IntermediateLinearBounds {
            bounds_at_layer: intermediate_bounds,
            start_layer,
        };

        Ok((output_bounds, intermediate))
    }
}
