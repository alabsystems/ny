// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches fast-path flow control for spec-guided CROWN backward.
//!
//! This module handles the Patches-mode backward dispatch attempt and the
//! `ensure_dense()` downgrade when patches dispatch fails. It is pure flow
//! control around the already-shared patches helper surface — no operator math
//! belongs here. Split from `core.rs` as part of #3960.

use crate::bounds::patches::CrownBounds;
use crate::layers::Layer;
use crate::network::core::{crown_backward_step_patches, CrownStepResult};
use crate::types::CrownIbpFallbackReason;

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::debug;

/// Result of the patches fast-path dispatch attempt.
pub(super) enum PatchesDispatchOutcome {
    /// Patches dispatch succeeded; caller should accumulate `node_cb` to input
    /// and continue to the next node.
    AccumulateToInput,
    /// Full IBP fallback needed for the given reason.
    IbpFallback(CrownIbpFallbackReason),
    /// Patches did not apply or failed; `node_cb` has been ensured dense.
    /// Caller should proceed with dense dispatch.
    FallThroughDense,
}

/// Attempt patches-mode backward dispatch with `ensure_dense()` downgrade.
///
/// If the node's `CrownBounds` are in Patches mode, attempts the patches
/// backward step. On success, returns `AccumulateToInput`. On recoverable
/// failure (patches dispatch error), downgrades to Dense mode and returns
/// `FallThroughDense`. On irrecoverable failure (`ensure_dense` fails),
/// returns `IbpFallback`.
pub(super) fn dispatch_patches_or_fallback(
    node_cb: &mut CrownBounds,
    layer: &Layer,
    pre_activation: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    node_deadline: Option<Instant>,
    node_name: &str,
    layer_type: &str,
) -> PatchesDispatchOutcome {
    match crown_backward_step_patches(
        layer,
        node_cb,
        pre_activation,
        engine,
        0,
        "SPEC-CROWN",
        node_deadline,
    ) {
        Ok(CrownStepResult::Continue) => return PatchesDispatchOutcome::AccumulateToInput,
        Ok(CrownStepResult::IbpFallback(fallback)) => {
            return PatchesDispatchOutcome::IbpFallback(fallback.reason)
        }
        Err(err) => {
            debug!(
                "Spec-guided CROWN: Patches dispatch failed at {} ({}): {}, falling back to Dense dispatch",
                node_name, layer_type, err
            );
        }
    }
    if matches!(node_cb, CrownBounds::Patches(_)) {
        match node_cb.ensure_dense() {
            Ok(_) => {}
            Err(err) => {
                debug!(
                    "Spec-guided CROWN: ensure_dense failed at {}: {}, falling back to IBP",
                    node_name, err
                );
                return PatchesDispatchOutcome::IbpFallback(
                    CrownIbpFallbackReason::CrownPropagationError,
                );
            }
        }
    }
    PatchesDispatchOutcome::FallThroughDense
}
