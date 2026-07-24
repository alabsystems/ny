// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward propagation methods for `GraphNetwork`.
//!
//! This module contains CROWN (Convex Relaxation based perturbatiON analysis
//! of neural Networks) propagation methods extracted from `GraphNetwork`'s impl.
//!
//! ## Design
//!
//! Uses the extension trait pattern established by `graph_ibp.rs`:
//! - Trait defines `_impl` methods with full implementation
//! - `GraphNetwork` methods in `core/graph.rs` delegate to trait methods
//! - This keeps the public API in one place while splitting implementation
//!
//! ## Module map
//! - `backward_node_dispatch`: shared per-node backward dispatch helpers (#3935)
//! - `helpers`: graph-pattern helpers (ex: softmax decomposition detection)
//! - `propagation`: core CROWN backward pass (spec-free)
//! - `point_vjp`: fast f32 point back-prop (exact point-Jacobian) for PGD attack
//! - `point_vjp_batched`: batched (K-restart) point-VJP plan + CPU mask capture
//!   for the ONE-wide-GPU-pass exact-gradient attack (#batched-vjp)
//! - `spec_propagation`: spec-guided CROWN plus linear-bound extraction/fallback
//! - `utils`: GraphNetwork helpers for accumulating/sanitizing bounds

mod backward_node_dispatch;
mod helpers;
mod point_vjp;
pub(crate) mod point_vjp_batched;
pub(crate) mod point_vjp_batched_resnet;
mod propagation;
pub(crate) mod spec_propagation;
mod utils;

// ATTACK-only soft-sign β (sharpness) ramp control for the point-VJP Sign arm.
pub use point_vjp::{
    attack_sign_beta, set_attack_sign_beta, AttackSignBetaGuard, DEFAULT_ATTACK_SIGN_BETA,
};
pub use point_vjp_batched::{point_vjp_forward_masks, PointVjpBatchPlan};
pub use point_vjp_batched_resnet::{
    point_vjp_resnet_forward_masks, PointVjpResnetPlan, PointVjpWavePlan,
};

#[cfg(test)]
#[path = "backward_node_dispatch_tests.rs"]
mod backward_node_dispatch_tests;

#[cfg(test)]
#[path = "propagation_tests.rs"]
mod propagation_tests;

pub(crate) use backward_node_dispatch::{backward_div_to_numerator, DivBackwardResult};
pub(crate) use propagation::GraphNetworkCrownExt;
