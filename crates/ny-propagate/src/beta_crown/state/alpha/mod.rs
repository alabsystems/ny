// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain-specific α state for joint α-β optimization.

mod domain;
mod graph_init;
mod graph_runtime;
mod neuron;
mod packed;

pub use domain::DomainAlphaState;
pub use graph_init::GraphDomainAlphaState;
pub use neuron::AlphaNeuronState;
pub(crate) use packed::PackedGraphDomainAlphaState;
pub use packed::{
    GraphAlphaStateByteCensus, GraphAlphaStateRepresentation, PACKED_GRAPH_ALPHA_FORMAT_VERSION,
};

#[cfg(test)]
mod tests;
