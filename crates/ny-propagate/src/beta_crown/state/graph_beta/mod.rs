// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph network β state for DAG-based CROWN propagation.

mod core;
mod entry;
mod gradients;
mod init;
mod lookup;
mod optimize;

pub use core::GraphBetaState;
pub use entry::GraphBetaEntry;

#[cfg(test)]
#[path = "../graph_beta_tests.rs"]
mod tests;
