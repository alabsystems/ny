// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Narrow CLI-owned bridge for post-seam AY verdict authority.
//!
//! `ny-propagate` cannot independently rebuild the CLI's Graph-MIP encoding, so
//! its cross-crate oracle registration and certificate constructor are unsafe:
//! safe library callers must not be able to install a forged proof result. This
//! module is the only CLI exception to the crate-wide unsafe-code denial. It
//! contains no memory-unsafe operation; each unsafe call attests a
//! verdict-soundness invariant established by the exact AY + ny-cert pipeline.

use std::collections::HashMap;
use std::time::Instant;

use ny_propagate::imb::{
    AyTailAffineReachabilityCertificate, AyTailAffineReachabilityEnvelope, AyTailCertificate,
    AyTailReachabilityCertificate, AyTailSharedInputReachabilityCertificate,
    AyTailSharedInputReachabilityEnvelope,
};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;

#[allow(unsafe_code)]
fn verified_oracle(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    objective: &[f32],
    p: &[f32],
    requested_lower: Option<f32>,
    deadline: Instant,
) -> Option<AyTailCertificate> {
    let certified = super::graph_mip::imb_ay_tail_certificate_exact_result(
        tail,
        seam_box,
        node_bounds,
        objective,
        p,
        requested_lower,
        deadline,
    )?;
    // SAFETY: the exact-result function returns only the opaque result of
    // ny-mip's proposal+proof or decision-only exact lower-bound API for the
    // just-encoded residual and the exact request arguments forwarded above.
    // Both APIs verify AY's exact relaxation entailment or root/tree
    // certificate against the original model and independently replay every
    // linear obligation through ny-cert before returning.
    Some(unsafe {
        AyTailCertificate::from_independently_verified_parts(
            certified.lower,
            certified.ay_tree_leaves,
            certified.ny_cert_farkas_replays,
        )
    })
}

#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)]
fn verified_reachability_oracle(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    objective: &[f32],
    p: &[f32],
    prefix_lower: f32,
    requested_lower: f32,
    deadline: Instant,
) -> Option<AyTailReachabilityCertificate> {
    let certified = super::graph_mip::imb_ay_tail_reachability_certificate_exact_result(
        tail,
        seam_box,
        objective,
        p,
        prefix_lower,
        requested_lower,
        deadline,
    )?;
    // SAFETY: the exact-result function adds the exact regional prefix premise
    // to the just-encoded tail model, proves the exact requested original
    // objective decision threshold with AY, verifies AY's exact relaxation
    // entailment or root/tree certificate, and independently replays every
    // linear obligation through ny-cert.
    Some(unsafe {
        AyTailReachabilityCertificate::from_independently_verified_parts(
            certified.lower,
            prefix_lower,
            certified.ay_tree_leaves,
            certified.ny_cert_farkas_replays,
        )
    })
}

#[allow(unsafe_code)]
fn verified_affine_reachability_oracle(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    objective: &[f32],
    envelope: &AyTailAffineReachabilityEnvelope,
    requested_lower: f32,
    deadline: Instant,
) -> Option<AyTailAffineReachabilityCertificate> {
    let certified = super::graph_mip::imb_ay_tail_affine_reachability_certificate_exact_result(
        tail,
        seam_box,
        objective,
        envelope,
        requested_lower,
        deadline,
    )?;
    // SAFETY: the exact-result function transports the complete immutable K=2
    // envelope into one tail model, proves the exact requested original
    // objective row with AY, verifies AY's exact relaxation entailment or
    // root/tree certificate, and independently replays every linear
    // obligation through ny-cert.
    Some(unsafe {
        AyTailAffineReachabilityCertificate::from_independently_verified_parts(
            tail,
            seam_box,
            objective,
            envelope,
            requested_lower,
            deadline,
            certified.lower,
            certified.ay_tree_leaves,
            certified.ny_cert_farkas_replays,
        )
    })
}

#[allow(unsafe_code)]
fn verified_shared_input_reachability_oracle(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    objective: &[f32],
    envelope: &AyTailSharedInputReachabilityEnvelope,
    requested_lower: f32,
    deadline: Instant,
) -> Option<AyTailSharedInputReachabilityCertificate> {
    let certified =
        super::graph_mip::imb_ay_tail_shared_input_reachability_certificate_exact_result(
            tail,
            seam_box,
            objective,
            envelope,
            requested_lower,
            deadline,
        )?;
    // SAFETY: the exact-result function independently validates the complete
    // root-valid shared bank and regional latent subset, transports every row
    // into the just-encoded tail model, proves the exact requested objective
    // threshold with AY, and replays every linear obligation through ny-cert.
    Some(unsafe {
        AyTailSharedInputReachabilityCertificate::from_independently_verified_parts(
            tail,
            seam_box,
            objective,
            envelope,
            requested_lower,
            deadline,
            certified.lower,
            certified.ay_tree_leaves,
            certified.ny_cert_farkas_replays,
        )
    })
}

fn requested(value: Option<&str>) -> bool {
    value == Some("1")
}

#[allow(unsafe_code)]
pub(super) fn install_verified_oracle_if_requested() -> bool {
    // Keep the exact gate at the unsafe boundary itself, so no future safe
    // in-crate caller can accidentally bypass it.
    if !requested(std::env::var("NY_IMB_TAIL_CERT_AY").ok().as_deref()) {
        return false;
    }
    // SAFETY: `verified_oracle` forwards every request argument unchanged and
    // has no result-construction path except the exact solver adapter above.
    let residual =
        unsafe { ny_propagate::imb::install_ay_tail_certificate_oracle(verified_oracle) };
    // SAFETY: `verified_reachability_oracle` forwards every request argument
    // unchanged and has no result-construction path except the conditional
    // exact solver adapter above.
    let reachability = unsafe {
        ny_propagate::imb::install_ay_tail_reachability_certificate_oracle(
            verified_reachability_oracle,
        )
    };
    // SAFETY: `verified_affine_reachability_oracle` has no result-construction
    // path except the exact augmented-model adapter above.
    let affine_reachability = unsafe {
        ny_propagate::imb::install_ay_tail_affine_reachability_certificate_oracle(
            verified_affine_reachability_oracle,
        )
    };
    // SAFETY: `verified_shared_input_reachability_oracle` has no
    // result-construction path except the exact augmented-model adapter above.
    let shared_input_reachability = unsafe {
        ny_propagate::imb::install_ay_tail_shared_input_reachability_certificate_oracle(
            verified_shared_input_reachability_oracle,
        )
    };
    residual && reachability && affine_reachability && shared_input_reachability
}

#[cfg(test)]
mod tests {
    #[test]
    fn oracle_install_boundary_is_exact_gated() {
        assert!(!super::requested(None));
        for malformed in ["", "0", "true", " 1", "1 ", "01", "１"] {
            assert!(!super::requested(Some(malformed)));
        }
        assert!(super::requested(Some("1")));
    }

    #[test]
    fn unsafe_authority_exception_is_confined_to_this_module() {
        let crate_root = include_str!("../../main.rs");
        assert!(crate_root.contains("#![deny(unsafe_code)]"));
        assert!(!crate_root.contains("#![forbid(unsafe_code)]"));

        let bridge = include_str!("ay_tail_authority.rs");
        let allow_marker = ["#![allow(", "unsafe_code", ")]"].concat();
        assert_eq!(
            bridge.matches(&allow_marker).count(),
            0,
            "the bridge must not grant a module-wide unsafe-code exception"
        );
        let narrow_allow_marker = ["#[allow(", "unsafe_code", ")]"].concat();
        assert_eq!(
            bridge.matches(&narrow_allow_marker).count(),
            5,
            "only the five authority-boundary functions may allow unsafe code"
        );
    }
}
