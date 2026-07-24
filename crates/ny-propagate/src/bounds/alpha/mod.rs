// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alpha-CROWN optimization state.
//!
//! Contains learnable alpha parameters for both sequential (`AlphaState`) and
//! graph/DAG (`GraphAlphaState`) verification pipelines.

mod graph;
mod graph_channel_only;
mod sequential;
mod shared;

// Re-export config types so existing `crate::bounds::alpha::X` paths continue to work.
pub use super::alpha_config::{
    AdamParams, AlphaCrownConfig, AlphaCrownIntermediate, AlphaSpecEarlyExit, GradientMethod,
    GraphAlphaCrownIntermediate, MultiSpecKeep, Optimizer,
};
pub use graph::GraphAlphaState;
pub use sequential::AlphaState;

#[cfg(test)]
mod tests;
