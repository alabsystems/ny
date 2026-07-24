// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Result types and configuration for bound propagation.
//!
//! This module contains the data structures used to configure propagation
//! and report verification results.

mod block_bounds;
mod checkpoint;
mod config;
mod crown_ibp;
mod helpers;
mod node_bounds;
mod progress;

// Re-export all public types at module root for backward compatibility
pub use block_bounds::{BlockBoundsInfo, BlockWiseResult};
pub use checkpoint::VerificationCheckpoint;
pub use config::{MulBinaryRelaxationMode, PropagationConfig, PropagationMethod};
pub use crown_ibp::{
    BoundsProvenance, CrownBackwardResult, CrownIbpBoundsResult, CrownIbpFallbackEvent,
    CrownIbpFallbackReason, CrownIbpPerNodeTimeBudget, GraphCrownIbpBoundsResult,
};
pub use helpers::{compute_model_hash, truncate_name};
pub use node_bounds::{LayerByLayerResult, NodeBoundsInfo};
pub use progress::{BlockProgress, LayerProgress};
