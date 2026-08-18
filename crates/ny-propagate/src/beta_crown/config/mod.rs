// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Configuration types for β-CROWN branch-and-bound search.

mod beta_config;
mod cut_config;
pub(crate) mod defaults;
mod optimization;
mod phase_budget;
mod relu_split;
#[cfg(test)]
mod tests;

// Re-export all public types to preserve `crate::beta_crown::config::*` paths.
pub(crate) use beta_config::AUTO_ENLARGE_BATCH_CAP;
pub use beta_config::{
    BetaCrownConfig, ConvMode, DepthTwoBranchLookaheadConfig, DepthTwoBranchLookaheadMode,
    InputClipType, KfsbReduceOp, VerificationArtifactAuthority,
    ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS, DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES,
    DEPTH_TWO_LOOKAHEAD_MAX_ROUNDS,
};
pub use cut_config::{CutEvictionPolicy, CutScoreWeights};
pub(crate) use optimization::radam_rectification_factor;
pub use optimization::{AdaptiveOptConfig, LRScheduler, LookaheadConfig, PerLayerLR};
pub use phase_budget::PhaseBudgetConfig;
