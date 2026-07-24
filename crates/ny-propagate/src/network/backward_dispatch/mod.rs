// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared backward CROWN layer dispatch core.
//!
//! This module provides a single canonical dispatch function for backward
//! CROWN propagation through individual graph nodes. All dispatch sites
//! (graph_crown::propagation, graph_alpha::backward, graph_alpha::bounds,
//! constraints::backward, spec_propagation, batched::backward_core) should route their layer match
//! through this module to eliminate the drift risk documented in #1949.
//!
//! The dispatch function handles the per-layer logic (shape setup, trait
//! dispatch, error mapping) and returns a [`BackwardDispatchResult`] that
//! tells the caller what to do with the resulting linear bounds. The caller
//! retains control over accumulation strategy, fallback behavior (IBP vs
//! error propagation), and mode-specific hooks (alpha/beta for ReLU).
//!
//! # Design
//!
//! Step B of the #1949 local-maximum unblock plan:
//! `designs/archive/2026-02-11-backward-dispatch-local-maximum-unblock.md`

mod concat;
mod dispatch;
mod helpers;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_dispatch_arms;
#[cfg(test)]
mod tests_engine;
mod types;

// Re-export public API items at the module level for unchanged import paths.
pub(crate) use dispatch::dispatch_backward_layer;
pub(crate) use types::{BackwardDispatchResult, DispatchContext};
