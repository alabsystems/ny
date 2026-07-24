// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! NY-owned trace bridge.
//!
//! This crate owns the contract between an ML framework's traced computation
//! graph and NY's verifier graph (`ny_build::GraphModel`). Historically the
//! translation logic lived inside the consuming framework (NN's `nn-verify`,
//! ~11k lines across 27 files) where it drifted independently from the
//! propagation engine. Moving it here makes the op→layer mapping, axis
//! conventions, and soundness-coverage classification a single NY-owned source
//! of truth that every intake path shares.
//!
//! - [`schema`] — NY-owned, `serde`-serializable mirror of a traced computation
//!   graph (`ComputationGraph`/`TraceNode`/`TraceOp`/`SegmentedGraph`). This is
//!   the stable cross-repo wire format; consumers serialize their trace to it.
//! - [`translate`] — lowers a [`schema::ComputationGraph`] into an
//!   `ny_build::GraphModel`. Sound by construction: any op not yet handled
//!   returns an error rather than emitting a vacuous/incorrect layer.
//! - [`coverage`] — build-time soundness-coverage gate: an exhaustive
//!   classification of every `TraceOp` so unsupported ops fail loudly.

pub mod coverage;
pub mod schema;
pub mod translate;
