// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Softmax layers for bound propagation.

mod bounds;
mod causal_softmax;
mod checked_lse;
mod gelu;
mod ibp;
mod layer;
mod linear;
mod logsoftmax;
mod logsumexp;
mod simplex_v;
// Justification: The `softmax` submodule contains the SoftmaxLayer type, re-exported from
// this parent `softmax` module. Renaming would break the natural module hierarchy where
// layers/softmax/ groups all softmax-related types.
#[allow(clippy::module_inception)]
pub mod softmax;
mod utils;
pub(crate) mod weighted_support;

pub(crate) use simplex_v::{simplex_lp_max, simplex_lp_min, tighten_softmax_v_ibp};

pub use causal_softmax::CausalSoftmaxLayer;
// Un-gated since #qdq-crown-followup: the shared-input staircase tightening
// (`crate::qdq_staircase`) reuses the reviewed 1-D GELU interval in
// production, not only in tests.
pub(crate) use gelu::gelu_bound_interval;
pub use gelu::{GELULayer, GeluApproximation, RelaxationMode};
pub use logsoftmax::LogSoftmaxLayer;
pub use logsumexp::LogSumExpLayer;
pub use softmax::SoftmaxLayer;

// Re-exports for tests: gelu relaxation functions used by tests via
// `crate::layers::*`.
pub use gelu::{
    adaptive_gelu_linear_relaxation, gelu_eval, gelu_linear_relaxation,
    gelu_sound_linear_relaxation, gelu_tanh_inflection_point, gelu_tanh_sound_linear_relaxation,
};
