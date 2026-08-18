// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for β-CROWN verifier.
//!
//! Extracted from engine/tests.rs for maintainability.
//! Part of #531.

mod bab_core_loop;
mod bab_frontier;
mod backward;
mod batched;
mod beta_optimization;
mod beta_state;
mod bilinear_bab;
mod branch_stem;
mod branching;
mod clause_learning;
mod clip_interm_domain_gate;
mod clipping_proptests;
mod complete_clipping_error_guards;
mod core_engine_identity;
mod crown_ibp;
mod cutting_planes;
mod deadline_explicit_caps;
mod deadline_initial_bound_stall;
mod deadline_none_parity;
mod domain_processing;
mod from_sequential_multi_objective_4355;
mod genbab;
mod gpu_bab;
mod gradients;
mod graph_beta_gradients;
mod graph_beta_lookup_paths;
mod graph_clause_learning;
mod graph_cuts;
mod graph_forward_mode_roots_4354;
mod graph_input_split_guards;
mod graph_multi_objective_relaxed_clipping;
mod graph_transitions;
mod lookahead;
mod lr_scheduler;
mod mul_binary_bab;
mod multi_objective_gpu_parity;
mod multi_relu_gradients;
mod norm_genbab;
mod optimization;
mod optimizer_bench;
mod pgd;
mod prelude;
mod relaxed_clip_helpers;
mod relaxed_clipping;
mod relu_directed_rounding;
mod relu_split_bounds;
mod transpose_backward;
mod transpose_backward_errors;

use crate::{Layer, LinearLayer, Network, ReLULayer};

use ndarray::arr2;

fn simple_network() -> Network {
    // Layer 1: Linear 2 -> 2
    let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    // Layer 2: ReLU
    // Layer 3: Linear 2 -> 1
    let w2 = arr2(&[[1.0, 1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network
}
