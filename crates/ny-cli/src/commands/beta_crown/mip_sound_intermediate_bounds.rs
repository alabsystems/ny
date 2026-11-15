// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mip-conv-sound-enclosure — the TYPE-ENFORCED seam between the IBP
//! collectors and the sequential MIP encoder.
//!
//! The Big-M constants of `encode_relu` are the pre-activation boxes handed to
//! `convert_intermediate_bounds` VERBATIM, and the AY Farkas/tree certificate
//! is checked against the ENCODED model, not the network. So an inward box is
//! a certified `Unsat` on a feasible instance — a minted proof. The first
//! repair (bbe9e9657) substituted `collect_ibp_bounds_sound` for the plain
//! collector at the two MIP collectors, which is a one-token change that a
//! merge can silently undo (CLAUDE.md records exactly that shape being
//! "silently clobbered in a merge"). This newtype makes the substitution a
//! property of the TYPE SYSTEM instead:
//!
//! - [`SoundIntermediateBounds`] has a PRIVATE field, so the only constructors
//!   are the two below, and both are fed exclusively by certified enclosures:
//!   [`SoundIntermediateBounds::collect_sound_ibp`] wraps
//!   `Network::collect_ibp_bounds_sound`, and
//!   [`SoundIntermediateBounds::refine_with_crown_ibp`] consumes an
//!   already-sound value and intersects it with the sound CROWN half
//!   (`collect_crown_ibp_bounds_with_precomputed_ibp`, whose per-element
//!   `max(lower)`/`min(upper)` of two enclosures is itself an enclosure).
//! - `convert_intermediate_bounds` (`mip_preprocess.rs`) accepts ONLY this
//!   type. A caller holding a plain `Vec<BoundedTensor>` from the plain
//!   collector cannot reach the encoder without a compile error.
//!
//! Both routes are strictly wider-or-equal than the plain box: they can only
//! turn a proof into a timeout, never mint one.

use anyhow::Result;
use ny_propagate::Network;
use ny_tensor::BoundedTensor;
use std::time::Instant;

/// Per-layer intermediate boxes CERTIFIED to enclose the true pre-activations
/// of the ORIGINAL (un-canonicalized) sequential network, one entry per layer
/// in network order.
///
/// The field is private on purpose: see the module doc. There is no
/// `From<Vec<BoundedTensor>>`, no `new`, and no mutable accessor.
#[derive(Debug, Clone)]
pub(super) struct SoundIntermediateBounds(Vec<BoundedTensor>);

impl SoundIntermediateBounds {
    /// Constructor 1: the certified sequential IBP collector.
    ///
    /// Linear goes through `propagate_ibp_sound` (Higham/abssum widening plus
    /// directed casts), the Conv family through
    /// `propagate_ibp_sound_with_engine` (certified `γ_{K+2}·S + 2u·|y|`
    /// term, rounded outward), AveragePool through its certified window-sum
    /// residual, and every other layer through its plain arm plus a 1-ULP
    /// outward margin (sound because every layer the MIP admits has a directed
    /// plain arm — pinned by `mip_admitted_layer_ratchet_tests.rs`).
    pub(super) fn collect_sound_ibp(network: &Network, input: &BoundedTensor) -> Result<Self> {
        let bounds = network
            .collect_ibp_bounds_sound(input)
            .map_err(|e| anyhow::anyhow!("IBP (sound) failed: {}", e))?;
        Ok(Self(bounds))
    }

    /// Constructor 2: CROWN-IBP over an already-sound precomputed vector.
    ///
    /// Consumes `self` so the CROWN half can only ever be intersected with a
    /// certified IBP half. On an already-expired `deadline` the precomputed
    /// vector is returned verbatim (the bounds-only API moves it back), so the
    /// starved-CROWN fallback is the certified box, never a plain one.
    pub(super) fn refine_with_crown_ibp(
        self,
        network: &Network,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<Self> {
        let refined = network
            .collect_crown_ibp_bounds_with_precomputed_ibp(input, self.0, None, deadline)
            .map_err(|e| anyhow::anyhow!("CROWN-IBP failed: {}", e))?;
        Ok(Self(refined))
    }

    /// Read-only view, one `BoundedTensor` per layer of the original network.
    pub(super) fn as_slice(&self) -> &[BoundedTensor] {
        &self.0
    }
}
