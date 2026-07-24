// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified Cut-CROWN resident-lane cut-fold REGISTRY (shared seam).
//!
//! The fold ENTRY + its global registry + the dark gate live here in `ny-core`
//! so that BOTH the writer (`ny-propagate`, which drives the resident GPU
//! backward but — to avoid a dependency cycle — depends on `ny-gpu` only as a
//! dev-dependency) and the reader (`ny-gpu`'s sound resident CROWN backward) can
//! reach the same registry through their shared `ny-core` dependency. `ny-gpu`
//! re-exports these symbols from `ny_gpu::wgpu_device` for the experiment
//! harnesses (`NY_CUT_FOLD_RESIDENT`), so existing call sites are unchanged.
//!
//! A registered fold contributes, to the LOWER objective at the target ReLU:
//!   * `coeffs`     — `+Σλ·cc_i` on the ReLU-OUTPUT frontier (pre-transform);
//!   * `pre_coeffs` — `+Σβ·a_i`  on the ReLU-INPUT frontier (post-transform);
//!   * `bias_shift` — `−Σλ·B`    on the lower bias.
//! Every cut satisfies `Σ cc·relu(ẑ) − B ≤ 0` on the box with `λ ≥ 0`, so the
//! folded objective lower-bounds the true one (Lean `cuts_fold_lower_bound`).
//!
//! DARK GATE: inert unless `NY_CUT_FOLD_RESIDENT=1` (legacy research lane) AND
//! an entry is registered. The proposed stem/head production authorities are
//! hard-quarantined below; the default path is byte-identical (no branch split,
//! no host work).

use std::sync::{OnceLock, RwLock};

/// The resident-lane cut fold: summed per-neuron coefficients at the TARGET
/// ReLU (the last `Activation` in resident fold order = the network's first
/// ReLU), plus the summed lower-bias shift.
#[derive(Debug, Clone, Default)]
pub struct ResidentCutFold {
    /// `(flat neuron index within the target ReLU layer, Σ_j λ_j·cc_i(j))`,
    /// added to the LOWER-side post-activation coefficient of every spec row
    /// (the ReLU-OUTPUT frontier, BEFORE the target Activation's transform).
    pub coeffs: Vec<(u32, f32)>,
    /// `−Σ_j λ_j·B_j`, added to every spec row's lower bias.
    pub bias_shift: f32,
    /// `(flat input-neuron index, Σ_j β_j·a_i(j))` — added to the LOWER-side
    /// PRE-activation (ReLU-INPUT) frontier column AFTER the target Activation
    /// transform (the `+β·a_i` channel the legacy C2 fold lacked; realizes the
    /// §2.2 pre-activation term on the resident lane). Empty ⇒ the fold site
    /// runs the single-part backward exactly as the legacy C2 path (no sub-
    /// split), so `NY_CUT_FOLD_RESIDENT` behaviour is untouched.
    pub pre_coeffs: Vec<(u32, f32)>,
    /// When true, every coefficient/bias add rounds OUTWARD and widens the
    /// certified `lower_err`/`lower_b_err` (production soundness — the
    /// proposed `NY_MULTINEURON_STEM` stem lever). `false` = the legacy
    /// experiment-grade
    /// plain-f32 add (byte-identical to today's `NY_CUT_FOLD_RESIDENT` path).
    pub sound_round: bool,
}

fn registry() -> &'static RwLock<Option<ResidentCutFold>> {
    static REGISTRY: OnceLock<RwLock<Option<ResidentCutFold>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(None))
}

/// The dark gate: resident cut folding is active only under the legacy research
/// harness `NY_CUT_FOLD_RESIDENT=1`. Neither proposed production environment
/// request (`NY_MULTINEURON_STEM` nor `NY_MN_HEAD_RESIDENT`) grants authority.
/// The research gate unset ⇒ `active_resident_cut_fold()` is `None` ⇒ the fold
/// site is never entered ⇒ byte-identical to today.
pub fn resident_cut_fold_enabled() -> bool {
    matches!(
        std::env::var("NY_CUT_FOLD_RESIDENT").ok().as_deref(),
        Some("1")
    )
}

/// Production authority for the proposed HEAD-RESIDENT retarget.
///
/// Hard-quarantined even when `NY_MN_HEAD_RESIDENT=1` is present. The current
/// producer reduces an exact facet multiplier to independent f32 post/pre/bias
/// values without carrying the reduction error that the certified f64 head
/// path records. Its runtime true-margin check is finite sampling: useful for
/// falsification, but not an enclosure proof, because an unseen input may have a
/// smaller true margin. Either gap can make a verdict-authoritative lower bound
/// too high. Returning `false` here also prevents mixed research/production env
/// requests from retargeting a fold onto the wrong activation.
///
/// Re-authorize only after the resident entry carries a checker-backed facet
/// certificate (including coefficient/bias build error) and the target identity
/// is bound into that certificate. Research implementation and tests remain in
/// tree, but an environment variable cannot grant production authority.
pub fn head_resident_retarget_enabled() -> bool {
    false
}

/// Register (or replace) the resident cut fold entry.
pub fn set_resident_cut_fold(entry: ResidentCutFold) {
    if let Ok(mut guard) = registry().write() {
        *guard = Some(entry);
    }
}

/// Clear the resident fold registry (back to the byte-identical default path).
pub fn clear_resident_cut_fold() {
    if let Ok(mut guard) = registry().write() {
        *guard = None;
    }
}

/// The active fold, or `None` (zero-cost untouched path) unless the gate is set
/// AND an entry is registered.
pub fn active_resident_cut_fold() -> Option<ResidentCutFold> {
    if !resident_cut_fold_enabled() {
        return None;
    }
    registry().read().ok()?.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_and_registry_semantics() {
        // Serialized env scope (clippy env wall); pre-test state restored on
        // exit.
        ny_test_utils::env::with_env_edits(|env| {
            env.remove("NY_CUT_FOLD_RESIDENT");
            env.remove("NY_MULTINEURON_STEM");
            env.remove("NY_MN_HEAD_RESIDENT");
            set_resident_cut_fold(ResidentCutFold {
                coeffs: vec![(3, 0.5)],
                bias_shift: -1.25,
                pre_coeffs: vec![(2, 0.75)],
                sound_round: true,
            });
            // Default (research gate unset) ⇒ inert even with an entry registered.
            assert!(active_resident_cut_fold().is_none());

            // Legacy gate arms it.
            env.set("NY_CUT_FOLD_RESIDENT", "1");
            let fold = active_resident_cut_fold().expect("entry must be active");
            assert_eq!(fold.coeffs, vec![(3, 0.5)]);
            assert_eq!(fold.bias_shift, -1.25);
            assert_eq!(fold.pre_coeffs, vec![(2, 0.75)]);
            assert!(fold.sound_round);
            env.remove("NY_CUT_FOLD_RESIDENT");

            // The proposed stem gate is authority-quarantined too.
            env.set("NY_MULTINEURON_STEM", "1");
            assert!(active_resident_cut_fold().is_none());
            env.remove("NY_MULTINEURON_STEM");
            assert!(active_resident_cut_fold().is_none());

            // The proposed HEAD-resident environment request is authority-
            // quarantined: it neither arms the registry nor retargets a stem
            // fold. Sampling is not a proof and the f32 reduction currently
            // carries no build-error certificate.
            assert!(!head_resident_retarget_enabled());
            env.set("NY_MN_HEAD_RESIDENT", "1");
            assert!(!head_resident_retarget_enabled());
            assert!(active_resident_cut_fold().is_none());
            env.set("NY_MULTINEURON_STEM", "1");
            assert!(active_resident_cut_fold().is_none());
            assert!(
                !head_resident_retarget_enabled(),
                "a head request must never retarget any registered fold"
            );
            env.remove("NY_MULTINEURON_STEM");
            env.remove("NY_MN_HEAD_RESIDENT");
            assert!(!head_resident_retarget_enabled());
            assert!(active_resident_cut_fold().is_none());

            clear_resident_cut_fold();
            env.set("NY_CUT_FOLD_RESIDENT", "1");
            assert!(active_resident_cut_fold().is_none());
        });
    }
}
