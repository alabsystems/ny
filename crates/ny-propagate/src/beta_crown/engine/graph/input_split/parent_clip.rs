// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Single-objective child clipping with precomputed parent linear bounds.
//!
//! Extracted from `shared.rs` to keep file sizes under 500 lines.
//! Part of #3870 Gap B.

use std::collections::HashMap;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::config::InputClipType;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::bounds::LinearBounds;
use crate::GraphNetwork;

/// Result of single-objective clipping with parent linear bounds.
/// Part of #3870 Gap B.
pub(crate) struct ParentLinearClipResult {
    pub(crate) bounds: BoundedTensor,
    pub(crate) verified: bool,
    /// Node bounds override from complete clipping for the deferred child CROWN pass.
    pub(crate) complete_clip_node_bounds: Option<HashMap<String, BoundedTensor>>,
}

/// Clip a single-objective child domain using precomputed parent linear bounds.
///
/// When the parent's CROWN linear bounds are available, they remain valid for
/// the child box (a strict subset of the parent). This avoids a redundant child
/// CROWN backward pass just for clipping, matching alpha-beta-CROWN's contract
/// where `input_split_and_repeat()` duplicates parent `lA`/`lbias` into split
/// children and `clip_domains()` consumes them directly.
///
/// When parent linear bounds are absent, falls back to a fresh graph CROWN pass
/// via `apply_relaxed_clipping_graph` / `apply_complete_clipping_graph`.
///
/// Source: `batch_branch_and_bound.py:151-169`, `clip.py:174-232`.
/// Part of #3870 Gap B.
#[allow(clippy::too_many_arguments)]
pub(crate) fn clip_child_with_parent_linear(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    child_input: &BoundedTensor,
    shape: &[usize],
    objective: &[f32],
    threshold: f32,
    parent_linear: Option<&LinearBounds>,
    engine: Option<&dyn GemmEngine>,
) -> Result<ParentLinearClipResult> {
    if let Some(parent_lb) = parent_linear {
        let clip_outcome = match verifier.config.input_clip_type {
            InputClipType::Relaxed => verifier.clip_with_precomputed_linear(
                child_input,
                shape,
                parent_lb,
                0,
                threshold,
            )?,
            InputClipType::Complete => verifier.complete_clip_with_precomputed_specs(
                child_input,
                shape,
                parent_lb,
                &[threshold],
            )?,
        };

        let mut complete_clip_node_bounds =
            if matches!(verifier.config.input_clip_type, InputClipType::Complete)
                && !clip_outcome.verified
            {
                match super::super::clip_complete::build_graph_complete_clip_node_bounds(
                    graph,
                    &clip_outcome.bounds,
                    parent_lb,
                    &[threshold],
                    verifier.config.verify_upper_bound,
                    verifier.config.clip_neuron_selection_ratio,
                    engine,
                ) {
                    Ok(node_bounds) => node_bounds,
                    Err(err) => {
                        tracing::trace!(
                            "graph complete clip from parent linear: \
                             skipping hidden-layer tightening: {}",
                            err
                        );
                        None
                    }
                }
            } else {
                None
            };

        // #3870: When the Lagrangian path has no active constraints (the spec
        // constraint is fully covered at the input level), fall back to IBP
        // bounds for the clipped child. These are at least as tight as what the
        // deferred CROWN pass would recompute and match the alpha-beta-CROWN
        // contract where clipped domains carry their node bounds through the
        // branching queue.
        if complete_clip_node_bounds.is_none()
            && matches!(verifier.config.input_clip_type, InputClipType::Complete)
            && !clip_outcome.verified
        {
            complete_clip_node_bounds =
                Some(graph.collect_node_bounds_with_engine(&clip_outcome.bounds, engine)?);
        }

        return Ok(ParentLinearClipResult {
            bounds: clip_outcome.bounds,
            verified: clip_outcome.verified,
            complete_clip_node_bounds,
        });
    }

    // Fallback: no parent linear bounds — run a fresh graph CROWN clip pass.
    let clip_outcome = match verifier.config.input_clip_type {
        InputClipType::Relaxed => verifier.apply_relaxed_clipping_graph(
            graph,
            child_input,
            shape,
            objective,
            threshold,
            engine,
        )?,
        InputClipType::Complete => verifier.apply_complete_clipping_graph(
            graph,
            child_input,
            shape,
            objective,
            threshold,
            engine,
        )?,
    };

    // #3870: carry IBP bounds for the clipped child input. When parent linear
    // bounds are absent and the fallback clip path runs, there's no Lagrangian
    // hidden-layer tightening, but the deferred CROWN pass still benefits from
    // receiving IBP bounds for the clipped domain.
    let complete_clip_node_bounds =
        if matches!(verifier.config.input_clip_type, InputClipType::Complete)
            && !clip_outcome.verified
        {
            Some(graph.collect_node_bounds_with_engine(&clip_outcome.bounds, engine)?)
        } else {
            None
        };

    Ok(ParentLinearClipResult {
        bounds: clip_outcome.bounds,
        verified: clip_outcome.verified,
        complete_clip_node_bounds,
    })
}
