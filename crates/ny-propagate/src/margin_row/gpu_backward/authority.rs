// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The verdict-authority capability chain — the M1 bit gate made BLOCKING
//! (adversarial-review item 2) and the DenormPreserve admission check
//! (item 3), enforced by the TYPE SYSTEM instead of protocol discipline:
//!
//! ```text
//! on-device probe readback ──▶ DeviceParityProof::qualify ──▶ VerdictAuthority
//!        (implementation           (the ONLY constructor;        (the ONLY way
//!         session wires the         refuses unless the           run_transaction
//!         dispatch)                 resolved denorm-preserve     — the only
//!                                   channel is ON, the probe     PassOut producer
//!                                   covers >= PARITY_MIN_LANES,  in this module —
//!                                   and every lane is            can be called)
//!                                   BIT-IDENTICAL; any mismatch
//!                                   latches the channel DEAD)
//! ```
//!
//! Both structs have private fields and no `Default`/`Clone`/`Copy`, and they
//! live in this child module, so code in `gpu_backward::*` — including a
//! prematurely-wired call site — CANNOT fabricate them: the compiler, not a
//! comment, guarantees that no pass reaches verdict math before the running
//! device/driver has reproduced the host ds twin bit-for-bit. This is the
//! pattern `ny_core::eft`'s module docs mandate for the GPU twin ("runs the
//! same probes ON DEVICE and refuses the channel", eft.rs:28–30), which the
//! original skeleton had dropped: WGSL does not guarantee `fma` is fused, so
//! a driver/naga update after a one-time M3 qualification could de-fuse it,
//! make `eft_two_prod`'s residual identically zero, and silently collapse the
//! value path to plain f32 (~2^-24) while the host still charges
//! `u_ds = 2^-44` — a ~1e6x UNDERCHARGE, i.e. a false-VERIFY generator. The
//! once-per-process re-qualification (`super::device_authority`, cached like
//! `ny_core::eft::eft_available`) catches that at the next process start.
//!
//! TRUST BOUNDARY (stated, not enforceable in types): [`DeviceParityProof::qualify`]
//! cannot verify that the `device` slice IS the readback of the probe
//! dispatch on the live adapter. Wiring the real dispatch into
//! `super::run_device_self_check` is an implementation-session review
//! obligation (design section 6), and the M2 shadow-enclosure asserts
//! cross-check the whole channel end-to-end before any authority A/B.

#![allow(dead_code)] // consumed by the implementation session's transaction

use super::ds::{bit_compare_streams, Ds};
use super::Refusal;

/// Minimum probe coverage for qualification: the banked adversarial lane
/// count of `ny-gpu ops/double_single_probe.rs` on this GB10 — 509 fma
/// TwoProduct lanes + 307 fma-barrier TwoSum lanes. A shorter stream refuses:
/// a token self-check must not be able to mint authority.
pub(crate) const PARITY_MIN_LANES: usize = 816;

/// Proof that THIS process's live device/driver reproduced the host ds twin
/// BIT-FOR-BIT over the adversarial probe lanes, on a denorm-preserving
/// channel. Unforgeable outside this module (private field, sole constructor
/// [`Self::qualify`]).
pub(crate) struct DeviceParityProof {
    lanes: usize,
}

impl DeviceParityProof {
    /// The ONLY constructor. `denorm_preserve_enabled` must be the device's
    /// RESOLVED word-channel state (`WgpuDevice::denorm_preserve_enabled`):
    /// `DenormPreservePolicy::Auto` resolves to `passthrough_supported`
    /// (`shader_loading.rs:100`), i.e. silently OFF on an unsupported
    /// adapter, where FTZ flushes subnormal TwoSum residuals to zero and
    /// voids the exact-residual identity in the UNDERCHARGE direction
    /// (review item 3) — so OFF refuses, unconditionally.
    ///
    /// `host` is the ds twin's stream over the probe lanes; `device` the
    /// readback of the same lanes dispatched through `ds_primitives.wgsl`.
    /// ANY bit divergence latches the lane's channel DEAD for the process
    /// (review item 2; there is deliberately no tolerance and no retry — a
    /// tolerance here is a soundness hole, design section 7 M1).
    pub(crate) fn qualify(
        denorm_preserve_enabled: bool,
        host: &[Ds],
        device: &[(f32, f32)],
    ) -> Result<Self, Refusal> {
        if !denorm_preserve_enabled {
            return Err(Refusal::Unmappable(
                "denorm-preserve channel resolved OFF on this adapter",
            ));
        }
        if host.len() < PARITY_MIN_LANES {
            return Err(Refusal::Unmappable(
                "parity self-check under-covers the probe lanes",
            ));
        }
        match bit_compare_streams(host, device) {
            Ok(()) => Ok(Self { lanes: host.len() }),
            Err(index) => {
                // Soundness-adjacent: print the diverging lane, then latch.
                eprintln!("[margin-row-gpu-eft] M1 parity mismatch at probe lane {index}");
                super::mark_channel_dead("M1 device/host EFT bit mismatch");
                Err(Refusal::ChannelDead)
            }
        }
    }

    /// Number of bit-compared lanes (>= [`PARITY_MIN_LANES`]).
    pub(crate) fn lanes(&self) -> usize {
        self.lanes
    }
}

/// The capability every verdict-touching pass must hold. Its ONLY constructor
/// consumes a [`DeviceParityProof`]; `super::run_transaction` — the only
/// function in this module that can produce a `PassOut` — takes it by
/// reference. Design section 6 "admit"; review item 2 resolution.
pub(crate) struct VerdictAuthority {
    proof: DeviceParityProof,
}

impl VerdictAuthority {
    /// Grant verdict authority by consuming the parity proof.
    pub(crate) fn grant(proof: DeviceParityProof) -> Self {
        Self { proof }
    }

    /// Probe coverage this authority was granted on (telemetry).
    pub(crate) fn parity_lanes(&self) -> usize {
        self.proof.lanes()
    }
}
