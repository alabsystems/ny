// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Constraint-aware CROWN propagation regression tests.
//!
//! Split by regression family from the original monolithic `tests.rs` (#4255).

pub(super) const TOL: f32 = 1e-6;

mod binary_soundness_audit;
mod clip_alpha;
mod concat;
mod cone_delta;
mod deadline;
mod error_mapping;
mod forward_basic;
mod genbab;
mod operator_dispatch;
mod patches;
mod runtime_guards;
mod seeded_gpu_suffix;
mod support;
mod two_layer;
mod two_neuron;
mod upstream_cache;
