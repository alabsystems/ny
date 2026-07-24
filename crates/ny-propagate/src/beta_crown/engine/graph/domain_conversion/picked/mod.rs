// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain extraction from PickedDomains batches and root domain creation.
//!
//! Functions for creating the initial root domain and extracting/branching
//! individual domains from batched `PickedDomains`.

mod graph_domain;
mod input_split;
mod relu_branch;
mod root;
mod shared;

pub use graph_domain::graph_domain_from_picked;
pub use input_split::branch_input_split_from_picked;
pub use relu_branch::branch_relu_from_picked;
pub use root::create_root_processed_domain;

#[cfg(test)]
pub use input_split::select_input_split_dimension;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_branching;
#[cfg(test)]
mod tests_root_graph;
