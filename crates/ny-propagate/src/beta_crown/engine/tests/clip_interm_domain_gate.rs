// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! TEETH for the sequential `clip_interm_domain` capability gate.
//!
//! A preset may REQUEST `bab.clip.interm_domain` on the sequential engine, where
//! the feature is quarantined (see
//! `crate::beta_crown::engine::domain::clip::sequential_clip_interm_domain_supported`).
//! The contract is that the request is SKIPPED: BaB must branch exactly as if the
//! flag were off.
//!
//! The bug these tests exist to prevent was measured on
//! `configs/vnncomp25/relusplitter.yaml`, which sets `bab.clip.interm_domain: true`.
//! The quarantined adapter used to return `NyError::SoundnessRefusal`, and
//! `create_child_domain` propagates errors with `?`. Because the gate fires for
//! every child at `history.depth() > 0`, i.e. EVERY child, the sequential search
//! collapsed at the root on all 220 instances of that benchmark: no domain was ever
//! branched, every one was recorded as a `PropagationFailure`, and provable
//! instances came back `unknown`.
//!
//! Both halves of the fix are covered here, and each is independently load-bearing:
//!   * the call-site capability gate in `domain::child::create_child_domain`, and
//!   * the identity fallback inside `apply_clip_interm_domain` itself.
//!
//! Reverting EITHER one alone still fails `sequential_bab_branches_when_preset_requests_clip_interm_domain`,
//! because removing the call-site gate re-enters the adapter and removing the
//! identity fallback makes that re-entry fatal.

use super::prelude::*;

/// The falsifiable-property configuration proven to force branching by
/// `branching::test_babsr_branching_heuristic`: `simple_network` computes
/// `y = relu(x1-x2) + relu(-x1+x2) = |x1-x2|`, whose true minimum over
/// `[-1,1]^2` is 0, so the CROWN root lower bound (0) sits below the 0.5
/// threshold and BaB is forced into the branching loop.
///
/// The network has exactly two ReLU neurons, so the split tree is finite and the
/// search terminates by exhaustion rather than on the clock. That makes
/// `domains_explored` and `max_depth_reached` deterministic and therefore safe to
/// compare across arms on a contended host — no timing artifact can move them.
const BRANCH_FORCING_THRESHOLD: f32 = 0.5;

fn branch_forcing_config(enable_clip_interm_domain: bool) -> BetaCrownConfig {
    BetaCrownConfig {
        max_domains: 64,
        timeout: Duration::from_mins(1),
        branching_heuristic: BranchingHeuristic::BoundImpact,
        enable_clip_interm_domain,
        ..Default::default()
    }
}

fn branch_forcing_input() -> BoundedTensor {
    BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap()
}

/// TEETH. With `enable_clip_interm_domain: true` — exactly what the relusplitter
/// preset sets — the sequential engine must still create children and descend
/// past the root.
///
/// Revert either half of the fix and this test fails hard: every child at
/// `depth > 0` errors out, `create_child_domain` yields no children,
/// `domains_explored` stays at 1 (the root) and `max_depth_reached` stays 0.
#[ntest::timeout(120000)]
#[test]
fn sequential_bab_branches_when_preset_requests_clip_interm_domain() {
    let network = simple_network();
    let input = branch_forcing_input();
    let verifier = BetaCrownVerifier::new(branch_forcing_config(true));

    let result = verifier
        .verify(&network, &input, BRANCH_FORCING_THRESHOLD)
        .expect("requesting a quarantined optional tightening must not fail verification");

    assert!(
        result.domains_explored >= 2,
        "BaB must branch with clip_interm_domain REQUESTED but quarantined; \
         domains_explored={} (1 means the root was processed and every child was \
         rejected — the collapse this gate exists to prevent)",
        result.domains_explored,
    );
    assert!(
        result.max_depth_reached >= 1,
        "BaB must descend past the root with clip_interm_domain REQUESTED but \
         quarantined; max_depth_reached={} (0 means no child at depth > 0 ever \
         survived, which is precisely the reverted-fix signature)",
        result.max_depth_reached,
    );
}

/// TEETH + SOUNDNESS STATEMENT. While the capability is quarantined, requesting it
/// must be a pure no-op: the verdict and the entire search trace must be identical
/// to running with the flag off.
///
/// This is the strongest available machine-checkable statement that the fix cannot
/// change a verdict. The flag does not tighten a bound, does not prune a domain and
/// does not reorder the search — so it cannot turn an `unknown` into an `unsat`.
/// It also pins the fix against a "helpful" future refactor that makes the skip
/// path do something other than nothing.
#[ntest::timeout(120000)]
#[test]
fn requested_clip_interm_domain_is_verdict_identical_to_disabled() {
    let network = simple_network();
    let input = branch_forcing_input();

    let disabled = BetaCrownVerifier::new(branch_forcing_config(false))
        .verify(&network, &input, BRANCH_FORCING_THRESHOLD)
        .expect("baseline arm must verify");
    let requested = BetaCrownVerifier::new(branch_forcing_config(true))
        .verify(&network, &input, BRANCH_FORCING_THRESHOLD)
        .expect("requested arm must verify");

    assert_eq!(
        requested.result, disabled.result,
        "a quarantined clip_interm_domain request must not change the VERDICT",
    );
    assert_eq!(
        requested.domains_explored, disabled.domains_explored,
        "a quarantined clip_interm_domain request must not change the search trace \
         (domains_explored)",
    );
    assert_eq!(
        requested.max_depth_reached, disabled.max_depth_reached,
        "a quarantined clip_interm_domain request must not change the search trace \
         (max_depth_reached)",
    );

    // Negative control: the arms are only meaningfully equal because BaB actually
    // ran. If both collapsed to a single root domain the equality above would hold
    // vacuously, so pin the shared trace to a branching one.
    assert!(
        disabled.domains_explored >= 2 && requested.domains_explored >= 2,
        "both arms must actually branch for the parity assertion to have content; \
         disabled={} requested={}",
        disabled.domains_explored,
        requested.domains_explored,
    );
}

/// The verdict must never be `Verified` here: `y = |x1 - x2|` attains 0 at
/// `x1 = x2`, which is inside the box, so no sound verifier may claim
/// `y > 0.5` everywhere.
///
/// This is the moat assertion in miniature. It is what makes the two tests above
/// safe to keep green by "just making BaB run": if unblocking the search ever
/// started minting unsound `Verified` verdicts on a property with a known
/// counterexample, this fails.
#[ntest::timeout(120000)]
#[test]
fn unblocked_bab_never_verifies_a_falsifiable_property() {
    let network = simple_network();
    let input = branch_forcing_input();

    for enable_clip_interm_domain in [false, true] {
        let result = BetaCrownVerifier::new(branch_forcing_config(enable_clip_interm_domain))
            .verify(&network, &input, BRANCH_FORCING_THRESHOLD)
            .expect("verification must not error");
        assert_ne!(
            result.result,
            BabVerificationStatus::Verified,
            "y = |x1-x2| attains 0 at x1=x2 inside the box, so threshold 0.5 is \
             unreachable and Verified would be a FALSE PROOF \
             (enable_clip_interm_domain={enable_clip_interm_domain})",
        );
    }
}
