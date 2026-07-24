// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::Instant;

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;
use ndarray::Array2;

use crate::beta_crown::config::BetaCrownConfig;
use crate::bounds::GraphAlphaState;
use crate::GraphNetwork;

/// Shared immutable bound-computation context for graph input-split BaB.
#[derive(Clone, Copy)]
pub(crate) struct InputSplitCrownContext<'a> {
    pub(crate) graph: &'a GraphNetwork,
    pub(crate) spec_matrix: &'a Array2<f32>,
    pub(crate) engine: Option<&'a dyn GemmEngine>,
    pub(crate) alpha_node_bounds: Option<&'a HashMap<String, BoundedTensor>>,
    pub(crate) alpha_state: Option<&'a GraphAlphaState>,
    pub(crate) mul_binary_alphas: Option<&'a HashMap<String, Array2<f32>>>,
    pub(crate) deadline: Option<Instant>,
    pub(crate) crown_backward_layers: Option<usize>,
    pub(crate) config: &'a BetaCrownConfig,
}
