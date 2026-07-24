// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Final result assembly for multi-objective graph BaB verification.
//!
//! Thin adapter over `GraphBabLifecycle::build_final_result()` that adds
//! the multi-objective-specific "could not verify all objectives" fallback.
//!
//! Part of #1860 (graph BaB service convergence, Packet A).

use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};

/// Assemble the terminal result after queue exhaustion/termination checks.
///
/// If any domains were unresolved (depth, no-branch, violated-drop, or
/// propagation failure), returns Unknown with cause-specific reasons.
/// If the queue is empty and domains were verified, returns Verified.
/// Otherwise returns Unknown "could not verify all objectives".
///
/// #1861/#1866: Check for unresolved domains before claiming Verified.
pub(super) fn finalize_multi_objective_result(
    lifecycle: &GraphBabLifecycle,
    queue_is_empty: bool,
) -> BetaCrownResult {
    if lifecycle.has_unresolved() {
        return lifecycle.build_result(BabVerificationStatus::Unknown {
            reason: lifecycle.unresolved_reason(),
        });
    }

    if lifecycle.domains_verified > 0 && queue_is_empty {
        lifecycle.build_result(BabVerificationStatus::Verified)
    } else {
        lifecycle.build_result(BabVerificationStatus::Unknown {
            reason: "Could not verify all objectives in explored domains".to_string(),
        })
    }
}
