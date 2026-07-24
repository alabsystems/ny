// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Branching heuristics and neuron constraint types for beta-CROWN.

mod general_history;
mod graph_constraints;
mod graph_history;
mod heuristic;
mod sequential;
mod split;

pub use general_history::GeneralSplitHistory;
pub use graph_constraints::{
    GenBabConstraint, GraphConstraint, GraphNeuronConstraint, NormInvRmsConstraint,
};
pub use graph_history::GraphSplitHistory;
pub use heuristic::BranchingHeuristic;
pub use sequential::{NeuronConstraint, SplitHistory};
pub use split::{LayerRef, NeuronSplit};

#[cfg(test)]
mod tests;
