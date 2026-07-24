// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use ny_propagate::PhaseBudgetConfig;
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};

/// Compute MIP CROWN-IBP budget from a `PhaseBudgetConfig`.
///
/// Centralizes the constants that were previously hardcoded as module-level
/// constants. The default `PhaseBudgetConfig` values produce identical behavior.
/// Source: `PhaseBudgetConfig.mip_crown_ibp_fraction` / `mip_crown_ibp_min_secs`
/// / `mip_crown_ibp_max_secs`.
pub(super) fn mip_crown_ibp_budget_secs(timeout_secs: f64, policy: &PhaseBudgetConfig) -> f64 {
    (timeout_secs * policy.mip_crown_ibp_fraction)
        .clamp(policy.mip_crown_ibp_min_secs, policy.mip_crown_ibp_max_secs)
}

/// Compute intermediate bounds for MIP encoding using budgeted CROWN-IBP (#3817).
///
/// The refinement pass gets a small slice of the MIP timeout budget; if the
/// deadline expires, the underlying CROWN-IBP helper falls back to the
/// precomputed IBP bounds for the remaining layers. This preserves a tight
/// solver budget on short verifier runs while still exploiting cheap CROWN-IBP
/// wins when available.
///
/// Accepts the caller's `PhaseBudgetConfig` so the CROWN-IBP budget is derived
/// from the shared policy rather than `PhaseBudgetConfig::default()` (#2206).
pub(crate) fn collect_mip_intermediate_bounds(
    network: &ny_propagate::Network,
    input: &BoundedTensor,
    timeout_secs: f64,
    policy: &PhaseBudgetConfig,
) -> Result<Vec<BoundedTensor>> {
    let crown_budget_secs = mip_crown_ibp_budget_secs(timeout_secs, policy);
    let deadline = Some(Instant::now() + Duration::from_secs_f64(crown_budget_secs));
    collect_mip_intermediate_bounds_with_deadline(network, input, deadline)
}

pub(crate) fn collect_mip_intermediate_bounds_with_deadline(
    network: &ny_propagate::Network,
    input: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<Vec<BoundedTensor>> {
    let ibp_bounds = network
        .collect_ibp_bounds(input)
        .map_err(|e| anyhow::anyhow!("IBP failed: {}", e))?;
    network
        .collect_crown_ibp_bounds_with_precomputed_ibp(input, ibp_bounds, None, deadline)
        .map_err(|e| anyhow::anyhow!("CROWN-IBP failed: {}", e))
}
