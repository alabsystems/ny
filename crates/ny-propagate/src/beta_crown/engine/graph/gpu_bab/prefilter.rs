// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Pre-filtering of picked domains before branching.
//!
//! Separates already-verified, violated, and max-depth domains from those
//! that need further splitting. This was previously inlined in the main BaB
//! loop (gpu_bab.rs lines 416-464).

use crate::batched_domain::DomainMetadata;

use super::check::{check_domain_bounds, BabLoopState, DomainCheckResult};

/// Result of pre-filtering a batch of picked domains.
pub(crate) struct PrefilterResult {
    /// Indices (into the picked batch) of domains that need further splitting.
    pub processable_indices: Vec<usize>,
    /// Whether a violation was found in the batch, requiring immediate return.
    pub violation: bool,
}

/// Filter a batch of picked domain metadata, separating verified/violated/max-depth
/// domains from those needing further splitting.
///
/// Updates `state` counters (domains_verified, max_depth_reached, unresolved flags)
/// as a side-effect.
///
/// # Returns
/// A `PrefilterResult` with processable indices and a violation flag.
pub(crate) fn prefilter_picked_domains(
    metadata: &[DomainMetadata],
    threshold: f32,
    verify_upper_bound: bool,
    max_depth: usize,
    state: &mut BabLoopState,
) -> PrefilterResult {
    let mut processable_indices = Vec::new();

    for (idx, meta) in metadata.iter().enumerate() {
        state.max_depth_reached = state.max_depth_reached.max(meta.depth);

        // Guard: NaN/Inf bounds from upstream propagation failure (#2933).
        // Without this, NaN domains fall through check_domain_bounds to Undecided,
        // enter the branching pipeline, and produce NaN children indefinitely.
        if !meta.lower_bound.is_finite() || !meta.upper_bound.is_finite() {
            tracing::warn!(
                idx,
                depth = meta.depth,
                lower = meta.lower_bound,
                upper = meta.upper_bound,
                "GPU BaB prefilter: domain dropped — non-finite bounds"
            );
            state.unresolved_due_to_propagation_failure = true;
            continue;
        }

        match check_domain_bounds(
            meta.lower_bound,
            meta.upper_bound,
            threshold,
            verify_upper_bound,
        ) {
            DomainCheckResult::Verified => {
                state.domains_verified += 1;
                continue;
            }
            DomainCheckResult::Violation => {
                return PrefilterResult {
                    processable_indices,
                    violation: true,
                };
            }
            DomainCheckResult::Undecided => {}
        }

        // Check max depth — domain is unresolved, not verified
        if meta.depth >= max_depth {
            tracing::info!(
                "GPU BaB: domain at max depth {} dropped (unresolved)",
                meta.depth
            );
            state.unresolved_due_to_depth = true;
            continue;
        }

        processable_indices.push(idx);
    }

    PrefilterResult {
        processable_indices,
        violation: false,
    }
}
