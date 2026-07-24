// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential α-CROWN propagation for `GraphNetwork`.
//!
//! Contains the main α-CROWN entry points for sequential (linear chain) graphs:
//! - [`GraphNetwork::propagate_alpha_crown`]
//! - [`GraphNetwork::propagate_alpha_crown_with_config`]
//! - [`GraphNetwork::propagate_alpha_crown_with_config_and_engine`]

mod backward;
mod collection;
mod optimization;
mod orchestration;

use crate::bounds::{AlphaCrownConfig, AlphaState, LinearBounds};
use crate::invprop::{InvpropConfig, OutputConstraints};

use ndarray::Array1;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::Instant;
use tracing::instrument;

use super::reference_bounds::GraphAlphaReferenceBounds;
use crate::network::core::GraphNetwork;

/// Result of a sequential backward pass: linear bounds + optional lower/upper gradients.
type BackwardPassResult = (
    LinearBounds,
    Option<Vec<Array1<f32>>>,
    Option<Vec<Array1<f32>>>,
);

#[cfg(test)]
type SequentialReferenceModeResult = (
    BoundedTensor,
    HashMap<String, BoundedTensor>,
    Vec<String>,
    usize,
    usize,
);

struct SequentialAlphaOptimizationResult {
    bounds: BoundedTensor,
    #[cfg(test)]
    reference_bounds: HashMap<String, BoundedTensor>,
    #[cfg(test)]
    reference_targets: Vec<String>,
    #[cfg(test)]
    reference_refresh_attempts: usize,
    #[cfg(test)]
    reference_tightened_targets_total: usize,
}

impl SequentialAlphaOptimizationResult {
    fn from_bounds(bounds: BoundedTensor) -> Self {
        Self {
            bounds,
            #[cfg(test)]
            reference_bounds: HashMap::new(),
            #[cfg(test)]
            reference_targets: Vec::new(),
            #[cfg(test)]
            reference_refresh_attempts: 0,
            #[cfg(test)]
            reference_tightened_targets_total: 0,
        }
    }

    fn with_reference_bounds(
        bounds: BoundedTensor,
        _reference_bounds: HashMap<String, BoundedTensor>,
        _reference_targets: Vec<String>,
        _reference_refresh_attempts: usize,
        _reference_tightened_targets_total: usize,
    ) -> Self {
        Self {
            bounds,
            #[cfg(test)]
            reference_bounds: _reference_bounds,
            #[cfg(test)]
            reference_targets: _reference_targets,
            #[cfg(test)]
            reference_refresh_attempts: _reference_refresh_attempts,
            #[cfg(test)]
            reference_tightened_targets_total: _reference_tightened_targets_total,
        }
    }
}

struct SequentialAlphaOptimizationContext<'a> {
    input: &'a BoundedTensor,
    config: &'a AlphaCrownConfig,
    engine: Option<&'a dyn GemmEngine>,
    reference_bounds: &'a mut GraphAlphaReferenceBounds,
    alpha_state: &'a mut AlphaState,
    exec_order: &'a [String],
    output_dim: usize,
    relu_name_to_idx: &'a HashMap<String, usize>,
    invprop_enabled: bool,
    carry_forward_reference_bounds: bool,
}

#[derive(Clone, Copy)]
struct SequentialBackwardPassContext<'a> {
    input: &'a BoundedTensor,
    node_bounds: &'a HashMap<String, BoundedTensor>,
    exec_order: &'a [String],
    output_dim: usize,
    relu_name_to_idx: &'a HashMap<String, usize>,
    alpha_state: &'a AlphaState,
    engine: Option<&'a dyn GemmEngine>,
}

struct SequentialBackwardPassRequest<'a> {
    context: SequentialBackwardPassContext<'a>,
    invprop_config: Option<&'a InvpropConfig>,
    output_constraints: Option<&'a OutputConstraints>,
    collect_gradients: bool,
    bounds_without_oc: Option<&'a mut Option<LinearBounds>>,
}

#[derive(Clone, Copy)]
pub(super) struct SequentialSinglePassRequest<'a> {
    pub(super) input: &'a BoundedTensor,
    pub(super) node_bounds: &'a HashMap<String, BoundedTensor>,
    pub(super) exec_order: &'a [String],
    pub(super) output_dim: usize,
    pub(super) relu_name_to_idx: &'a HashMap<String, usize>,
    pub(super) alpha_state: &'a AlphaState,
    pub(super) engine: Option<&'a dyn GemmEngine>,
    pub(super) deadline: Option<Instant>,
}

impl GraphNetwork {
    /// Propagate bounds through the graph using α-CROWN with optimized parameters.
    ///
    /// α-CROWN extends CROWN by making the lower bound slope (α) for unstable ReLUs
    /// learnable and optimizing it via gradient descent to tighten bounds.
    ///
    /// Algorithm:
    /// 1. Run CROWN-IBP to collect tighter pre-activation bounds at each node
    /// 2. Identify ReLU nodes and initialize α state
    /// 3. For each optimization iteration:
    ///    a. Run CROWN backward with current α values
    ///    b. Concretize to get bounds
    ///    c. Compute gradients ∂bounds/∂α
    ///    d. Update α via gradient descent
    /// 4. Return the tightest bounds found
    #[inline]
    #[instrument(skip(self, input), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_alpha_crown(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_engine(input, None)
    }

    /// α-CROWN with optional GEMM acceleration engine.
    #[inline]
    #[instrument(skip(self, input, engine), fields(num_nodes = self.nodes.len(), input_shape = ?input.shape()))]
    pub fn propagate_alpha_crown_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_config_and_engine(
            input,
            &AlphaCrownConfig::default(),
            engine,
        )
    }

    /// α-CROWN with custom configuration (no acceleration engine).
    #[instrument(skip(self, input, config), fields(num_nodes = self.nodes.len(), iterations = config.iterations))]
    pub fn propagate_alpha_crown_with_config(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
    ) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_config_and_engine(input, config, None)
    }

    /// α-CROWN with custom configuration and optional GEMM acceleration engine.
    #[instrument(skip(self, input, config, engine), fields(num_nodes = self.nodes.len(), iterations = config.iterations))]
    pub fn propagate_alpha_crown_with_config_and_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_config_and_engine_impl(input, config, engine, true)
            .map(|result| result.bounds)
    }

    #[cfg(test)]
    pub(crate) fn propagate_alpha_crown_with_reference_mode_for_test(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        carry_forward_reference_bounds: bool,
    ) -> Result<SequentialReferenceModeResult> {
        let result = self.propagate_alpha_crown_with_config_and_engine_impl(
            input,
            config,
            None,
            carry_forward_reference_bounds,
        )?;
        Ok((
            result.bounds,
            result.reference_bounds,
            result.reference_targets,
            result.reference_refresh_attempts,
            result.reference_tightened_targets_total,
        ))
    }

    /// GraphNetwork α-CROWN with directed rounding for soundness.
    ///
    /// Same as `propagate_alpha_crown` but applies 1-ULP widening to the final bounds.
    pub fn propagate_alpha_crown_sound(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_sound_with_engine(input, None)
    }

    /// GraphNetwork α-CROWN with directed rounding for soundness and optional GEMM acceleration.
    pub fn propagate_alpha_crown_sound_with_engine(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        let bounds = self.propagate_alpha_crown_with_engine(input, engine)?;
        Ok(bounds.round_for_soundness())
    }
}

#[cfg(test)]
thread_local! {
    static SEQUENTIAL_REFERENCE_REFRESH_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SEQUENTIAL_REFERENCE_TIGHTENED_TARGETS_TOTAL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Number of sequential-orchestration root intermediate-bound collection
    /// episodes (IBP or deep-override CROWN-IBP) started by
    /// `propagate_alpha_crown_with_config_and_engine_impl`. Graphs that route
    /// to the DAG delegate must never start one (#dedup-root-collections
    /// Fix A: the routing scan runs before the collection).
    static SEQUENTIAL_ROOT_COLLECTION_EPISODES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
