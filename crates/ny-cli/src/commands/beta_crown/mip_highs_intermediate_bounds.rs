// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use ny_propagate::PhaseBudgetConfig;
use ny_tensor::BoundedTensor;
use std::time::Instant;

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

/// Compute intermediate bounds for MIP encoding under an absolute deadline.
///
/// If the deadline expires, the underlying CROWN-IBP helper falls back to the
/// precomputed IBP bounds for remaining layers. The caller derives this
/// deadline from its shared phase policy and caps it at the overall attempt
/// deadline, so no local clock can extend the verification budget.
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
