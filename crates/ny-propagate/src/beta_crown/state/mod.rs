// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! β and α state management for CROWN optimization.
//!
//! Contains:
//! - `GraphBetaEntry`/`GraphBetaState`: β state for graph networks (DAG)
//! - `BetaEntry`/`BetaState`: β state for sequential networks
//! - `DomainAlphaState`: Per-domain α parameters for joint optimization

mod alpha;
mod beta;
mod beta_lookup;
mod graph_beta;

pub(crate) use alpha::PackedGraphDomainAlphaState;
pub use alpha::{
    AlphaNeuronState, DomainAlphaState, GraphAlphaStateByteCensus, GraphAlphaStateRepresentation,
    GraphDomainAlphaState, PACKED_GRAPH_ALPHA_FORMAT_VERSION,
};
pub use beta::{BetaEntry, BetaState};
pub use graph_beta::{GraphBetaEntry, GraphBetaState};

#[cfg(test)]
mod api_tests {
    //! Compile-level API compatibility guard for the state module split.
    //!
    //! These tests verify that both import path families remain functional
    //! after the state.rs → state/ directory migration (#1698).
    //! Required by: designs/2026-02-08-beta-crown-state-api-compat-contract.md

    use std::mem::size_of;

    /// Verify that all core state types are accessible via `crate::beta_crown::state::*`.
    #[test]
    fn state_api_paths_compile_via_state_module() {
        // These imports must compile — if any re-export is missing from
        // state/mod.rs, this test fails at compile time.
        use crate::beta_crown::state::AlphaNeuronState;
        use crate::beta_crown::state::BetaEntry;
        use crate::beta_crown::state::BetaState;
        use crate::beta_crown::state::DomainAlphaState;
        use crate::beta_crown::state::GraphBetaEntry;
        use crate::beta_crown::state::GraphBetaState;
        use crate::beta_crown::state::GraphDomainAlphaState;

        // Instantiate to prevent dead-code elimination from optimizing away the imports
        let _ = BetaState::empty();
        let _ = DomainAlphaState::empty();
        let _ = GraphBetaState::empty();
        let _ = GraphDomainAlphaState::empty();
        let _ = size_of::<BetaEntry>();
        let _ = size_of::<GraphBetaEntry>();
        let _ = size_of::<AlphaNeuronState>();
    }

    /// Verify that all core state types are accessible via `crate::beta_crown::*`
    /// (root-level re-exports from beta_crown/mod.rs).
    #[test]
    fn state_api_paths_compile_via_root_reexports() {
        // These imports must compile — if any re-export is missing from
        // beta_crown/mod.rs pub use state::{...}, this test fails at compile time.
        use crate::beta_crown::AlphaNeuronState;
        use crate::beta_crown::BetaEntry;
        use crate::beta_crown::BetaState;
        use crate::beta_crown::DomainAlphaState;
        use crate::beta_crown::GraphBetaEntry;
        use crate::beta_crown::GraphBetaState;
        use crate::beta_crown::GraphDomainAlphaState;

        let _ = BetaState::empty();
        let _ = DomainAlphaState::empty();
        let _ = GraphBetaState::empty();
        let _ = GraphDomainAlphaState::empty();
        let _ = size_of::<BetaEntry>();
        let _ = size_of::<GraphBetaEntry>();
        let _ = size_of::<AlphaNeuronState>();
    }
}
