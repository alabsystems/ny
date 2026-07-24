// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fallback heuristics for batched graph PGD shape-compatibility failures.

use ny_core::NyError;
use std::time::Instant;

pub(super) fn graph_pgd_batching_error_should_skip(err: &anyhow::Error) -> bool {
    matches!(
        err.downcast_ref::<NyError>(),
        Some(NyError::ShapeMismatch { .. })
    )
}

pub(super) fn skip_incompatible_batched_graph_pgd(
    err: &anyhow::Error,
    json: bool,
    pgd_start: Instant,
    engine_label: &str,
) {
    super::super::graph_pgd::emit_graph_pgd_status(
        json,
        format_args!(
            "  Graph PGD: batching hit a shape mismatch; falling back to sequential upfront attack ({:.2}s, engine={})",
            pgd_start.elapsed().as_secs_f64(),
            engine_label
        ),
    );
    tracing::warn!(
        error = %err,
        "Graph PGD batched path incompatible with this graph; falling back to sequential upfront attack"
    );
}

pub(super) fn should_retry_with_folded_batch(input_shape: &[usize], err: &anyhow::Error) -> bool {
    if input_shape.len() < 2 {
        return false;
    }

    match err.downcast_ref::<NyError>() {
        Some(NyError::InvalidSpec(message)) => {
            message.contains(&format!("got {}D", input_shape.len() + 1))
        }
        _ => false,
    }
}
