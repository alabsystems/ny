// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph-level input-split BaB verifier for `BetaCrownVerifier`.
//!
//! Split from the original monolithic `input_split.rs` into a directory module
//! following the conv1d/types shell-first pattern (#3973).
//!
//! - `shared` — Domain types, priority ordering, helper functions
//! - `adv_check` — Lightweight PGD probe for early SAT detection
//! - `batching` — Reordered BaB batch helpers for single-objective input split
//! - `loop_batch_size` — Runtime batch-size clamp for reordered input split
//! - `mul_binary` — MulBinary SPSA alpha optimization
//! - `root_bounds` — Root intermediate-bound warmup selection
//! - `single_objective` — Single-objective input-split verifier loop
//! - `multi_objective` — Multi-objective conjunctive verifier loop
//! - `disjunctive_multi_clause` — Multi-clause disjunctive (OR-of-AND) verifier loop (#3740)

pub(crate) mod adv_check;
pub(super) mod batched_clip;
pub(crate) mod batching;
pub(crate) mod bounds_eval;
pub(crate) mod build_batches;
mod disjunctive_multi_clause;
/// Lane hooks for the lsnc certified f64 tail pass
/// (docs/LSNC_F64_TAIL_DESIGN.md; gate `NY_F64_TAIL=1`, default OFF).
mod f64_tail;
/// Default-dark exact-domain Clip-and-Verify route.  The bound-producing
/// callback is invoked on one domain at a time and the returned planes stay
/// paired with that exact source box, preventing parent-plane reuse.
mod fresh_domain_clip;
pub(crate) mod grouped_semantics;
pub(crate) mod ibp_prescreen;
mod ibp_prescreen_flat;
pub(crate) mod loop_batch_size;
pub(crate) mod metrics;
pub(crate) mod mul_binary;
mod multi_obj_domain;
mod multi_objective;
pub(crate) mod parent_clip;
pub(crate) mod root_bounds;
/// Saturation-Escape Branching (SEB) advisory input-split scorer
/// (docs/SATURATION_ESCAPE_BRANCHING_DESIGN.md; preset
/// `bab.branching.input_split.sat_escape_branch`, env `NY_SAT_ESCAPE_BRANCH`
/// override, default OFF). Advisory only — reorders which dims to split,
/// never a bound.
pub(crate) mod sat_escape;
pub(crate) mod shared;
pub(crate) mod shared_specs;
mod single_objective;

#[cfg(test)]
mod tests;
