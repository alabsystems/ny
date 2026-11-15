// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use ny_propagate::PhaseBudgetConfig;
use ny_tensor::BoundedTensor;
use std::time::Instant;

use super::super::mip_sound_intermediate_bounds::SoundIntermediateBounds;

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
///
/// SOUNDNESS (#mip-conv-sound-enclosure): the IBP half MUST come from
/// [`Network::collect_ibp_bounds_sound`], never the plain collector. These
/// boxes become the Big-M constants of `encode_relu` verbatim (no inflation),
/// the CROWN-IBP intersection is a per-element `max(lower)`/`min(upper)` that
/// cannot repair an inward endpoint, and the AY Farkas/tree certificate is
/// checked against the ENCODED model, not the network — so an inward
/// pre-activation box turns a feasible instance into a certified `Unsat`.
/// The plain collector's Conv2d arm is round-to-nearest f32 im2col GEMM plus an
/// f32 bias add with no directed rounding and no Higham widening
/// (`conv2d/bound.rs`, "SOUND IBP forward" doc); under cancellation it excludes
/// the true value. The sound collector routes Conv2d through the certified
/// Higham arm, Linear through `propagate_ibp_sound`, and widens every other
/// layer by 1 ULP. Strictly wider-or-equal: it can only turn a proof into a
/// timeout, never mint one.
///
///
/// The return type is the TYPE-ENFORCED seam (`mip_sound_intermediate_bounds.rs`):
/// [`SoundIntermediateBounds`] can only be built from the certified collector
/// and refined by CROWN over that certified vector, and
/// `convert_intermediate_bounds` accepts nothing else, so a plain-collector
/// substitution here no longer compiles.
///
/// [`Network::collect_ibp_bounds_sound`]: ny_propagate::Network::collect_ibp_bounds_sound
pub(crate) fn collect_mip_intermediate_bounds_with_deadline(
    network: &ny_propagate::Network,
    input: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<SoundIntermediateBounds> {
    let ibp_bounds = SoundIntermediateBounds::collect_sound_ibp(network, input)?;
    tracing::info!(
        "[mip-sound-ibp] sequential MIP intermediate bounds: certified IBP enclosure over {} layers (CROWN-IBP intersection follows)",
        ibp_bounds.as_slice().len()
    );
    ibp_bounds.refine_with_crown_ibp(network, input, deadline)
}
