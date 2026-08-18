// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified Cut-CROWN increment C2 (resident-lane gate): dark fold of
//! multi-neuron cuts into the SOUND GPU-resident resnet CROWN backward
//! (`docs/CERTIFIED_CUT_CROWN_DESIGN.md` §C2).
//!
//! The fold ENTRY (`ResidentCutFold`), its global registry, and the dark gate
//! now live in `ny-core` (`ny_core::resident_cut_fold`) so that `ny-propagate`
//! — which drives the resident backward but depends on `ny-gpu` only as a
//! dev-dependency (cycle avoidance) — can WRITE the fold through the shared
//! `ny-core` dependency, while THIS crate's resident backward READS it. The
//! registry symbols are re-exported here unchanged for raw research
//! construction. The proof-path reader is hard-quarantined, so environment
//! requests cannot make those entries influence certificate-bearing bounds.
//!
//! What stays local to `ny-gpu`: the read-only C2b frontier CAPTURE channel and
//! the applied-fold counter (experiment observability the resident backward
//! populates; `ny-propagate` does not need them).
//!
//! A registered fold contributes `Σ_j λ_j·(Σ_{i∈G_j} cc_i·relu(ẑ_i) − B_j)` to
//! the LOWER objective at the network's FIRST ReLU. Since every cut satisfies
//! `Σ cc·relu(ẑ) − B ≤ 0` on the box and `λ_j ≥ 0`, the folded objective
//! lower-bounds the true one (`cuts_fold_lower_bound` in the Lean schema).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

// Shared registry (writer = ny-propagate, reader = ny-gpu backward). Re-exported
// so `ny_gpu::wgpu_device::{ResidentCutFold, set/clear/active/enabled}` resolve
// exactly as before the ny-core move.
pub use ny_core::resident_cut_fold::{
    active_resident_cut_fold, clear_resident_cut_fold, head_resident_retarget_enabled,
    resident_cut_fold_enabled, set_resident_cut_fold, ResidentCutFold,
};

/// C2b capture: the LOWER-side coefficient frontier at the fold ReLU (the
/// incoming per-spec-row coefficients `A` over the target ReLU's
/// post-activation, taken BEFORE the fold's `+λ·cc` is added). This is the
/// signal the objective-signed group selection needs: the fold only helps
/// where it cancels NEGATIVE coefficient mass (upper-chord intercept
/// payments); everywhere else it pays `λ·B` for nothing (the measured C2
/// negative — `docs/CERTIFIED_CUT_CROWN_DESIGN.md`).
#[derive(Debug, Clone, Default)]
pub struct ResidentCutFoldCapture {
    /// Number of spec rows captured.
    pub num_specs: usize,
    /// Frontier dimension = the target ReLU layer's flat neuron count.
    pub dim: usize,
    /// Row-major `num_specs x dim` lower-side coefficients.
    pub lower_a: Vec<f32>,
}

fn capture_slot() -> &'static RwLock<Option<ResidentCutFoldCapture>> {
    static SLOT: OnceLock<RwLock<Option<ResidentCutFoldCapture>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(None))
}

/// Legacy capture request. The proof-path fold branch is hard-quarantined, so
/// setting this environment variable cannot currently populate a capture
/// through certificate-bearing resident CROWN.
pub fn resident_cut_fold_capture_enabled() -> bool {
    matches!(
        std::env::var("NY_CUT_FOLD_CAPTURE").ok().as_deref(),
        Some("1")
    )
}

/// Take (and clear) the captured fold-frontier coefficients, if any.
pub fn take_resident_cut_fold_capture() -> Option<ResidentCutFoldCapture> {
    capture_slot().write().ok()?.take()
}

/// Store a capture (called by the fold site; last write wins across
/// explosion-fallback re-runs — the surviving run's frontier is the one that
/// produced the returned bounds).
pub(crate) fn store_resident_cut_fold_capture(capture: ResidentCutFoldCapture) {
    if let Ok(mut guard) = capture_slot().write() {
        *guard = Some(capture);
    }
}

/// Number of times the fold was actually applied to a resident backward pass
/// (each explosion-fallback re-run counts separately) — lets the experiment
/// harness assert the fold site was really exercised.
static APPLIED: AtomicU64 = AtomicU64::new(0);

/// How many resident backward passes have had the fold applied since reset.
pub fn resident_cut_fold_applied_count() -> u64 {
    APPLIED.load(Ordering::Relaxed)
}

/// Reset the applied-fold counter.
pub fn reset_resident_cut_fold_applied_count() {
    APPLIED.store(0, Ordering::Relaxed);
}

/// Bump the applied-fold counter (called by the fold site on application).
pub(crate) fn note_resident_cut_fold_applied() {
    APPLIED.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single sequential test: env-var + global-registry manipulation must not
    /// race with itself across parallel test threads. The gate/registry
    /// semantics themselves are covered in `ny_core::resident_cut_fold`; this
    /// asserts the ny-gpu re-export quarantine + the local diagnostic counter.
    #[test]
    fn reexport_and_applied_counter() {
        // Serialized env scope (clippy env wall); pre-test state restored on
        // exit.
        ny_test_utils::env::with_env_edits(|env| {
            env.remove("NY_CUT_FOLD_RESIDENT");
            env.remove("NY_MULTINEURON_STEM");
            set_resident_cut_fold(ResidentCutFold {
                coeffs: vec![(3, 0.5)],
                bias_shift: -1.25,
                pre_coeffs: vec![(2, 0.75)],
                sound_round: true,
            });
            assert!(active_resident_cut_fold().is_none());

            env.set("NY_CUT_FOLD_RESIDENT", "1");
            assert!(
                active_resident_cut_fold().is_none(),
                "legacy env plus public registry must not grant resident proof authority"
            );

            reset_resident_cut_fold_applied_count();
            assert_eq!(resident_cut_fold_applied_count(), 0);
            note_resident_cut_fold_applied();
            assert_eq!(resident_cut_fold_applied_count(), 1);
            reset_resident_cut_fold_applied_count();

            clear_resident_cut_fold();
            assert!(active_resident_cut_fold().is_none());
        });
    }
}
