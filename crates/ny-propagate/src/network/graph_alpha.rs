// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alpha-CROWN methods for GraphNetwork.

mod alpha_projection;
mod backward;
mod bounds;
pub(crate) mod invprop_backward;
mod propagate_dag;
pub(crate) mod propagate_helpers;
mod propagate_sequential;
mod reference_bounds;
// pub(crate) so the network module can re-export the resnet decomposition for the
// beta_crown BaB engine's per-domain GPU beta backward (#unsat-keystone step 4).
pub(crate) mod resnet_decompose;
pub(crate) mod resnet_skeleton;
mod runtime_state;
mod sequential_gradients;
mod spsa;
mod spsa_accumulate;
mod zonotope;

pub(crate) use bounds::budget_policy;
pub(crate) use reference_bounds::merge_reference_bound_maps;
