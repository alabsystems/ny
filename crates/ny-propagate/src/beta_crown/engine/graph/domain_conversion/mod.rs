// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conversion utilities between GraphBabDomain and DomainList/ProcessedDomains.
//!
//! These functions enable interoperability between the per-domain BaB representation
//! (`GraphBabDomain`) and the batched tensor storage (`DomainList`/`ProcessedDomains`).
//!
//! # Submodules
//!
//! - [`history`] — Serialization/deserialization between GraphSplitHistory and ConstraintTuple
//! - [`picked`] — Domain extraction from PickedDomains batches and root domain creation
//! - [`processed`] — Conversion from GraphBabDomains to ProcessedDomains

mod history;
mod picked;
mod processed;

// Re-export public items to maintain existing import paths.
pub use picked::{
    branch_input_split_from_picked, branch_relu_from_picked, create_root_processed_domain,
    graph_domain_from_picked,
};
pub use processed::{processed_from_backward_results, processed_from_graph_domains_with_la};

// Re-exports used only in test code.
#[cfg(test)]
pub use history::history_from_constraints;
#[cfg(test)]
pub use picked::select_input_split_dimension;
#[cfg(test)]
pub use processed::processed_from_graph_domains_direct;
