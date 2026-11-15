// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::commands::beta_crown::resolve_overall_deadline;
use std::time::{Duration, Instant};

#[test]
fn historical_delayed_verifier_entry_keeps_the_original_absolute_solver_grant() {
    // Retained row 27 has budget550 and a27s original flush reserve. Simulate
    // loading/wrapper delays without sleeping or constructing a verifier.
    let start = Instant::now();
    let instance = start + Duration::from_secs(550);
    let decision = internal_timeout_decision("ordinary_fixture", 550, None);
    assert_eq!(decision.route, InternalTimeoutRoute::Historical);
    assert_eq!(decision.solver_timeout_secs, 523);
    assert_eq!(decision.effective_flush_reserve_secs, 27);
    let route = beta_phase_deadline(decision, instance);
    assert_eq!(route, None, "historical tail scheduling stays unchanged");
    let original_solver = solver_phase_deadline(decision, instance);
    let authority = beta_solver_authority_deadline(route, Some(original_solver));
    assert_eq!(authority, Some(start + Duration::from_secs(523)));
    for delay in [0, 10, 65, 522, 530] {
        let entry = start + Duration::from_secs(delay);
        let configured = Duration::from_secs(decision.solver_timeout_secs);
        // This is the production beta resolver, not a duplicate of its math.
        assert_eq!(
            resolve_overall_deadline(entry, configured, authority).unwrap(),
            Some(original_solver),
            "setup delay {delay} must not add time to the original grant"
        );
        assert_eq!(
            resolve_overall_deadline(entry, configured, None).unwrap(),
            Some(entry + configured),
            "the regression really distinguishes the previous restart"
        );
    }
}

#[test]
fn original_authority_never_extends_a_shorter_configured_reserve_window() {
    let start = Instant::now();
    let original_solver = start + Duration::from_secs(523);
    let authority = beta_solver_authority_deadline(None, Some(original_solver));
    // Any tighter inner grant (fixed margin-row/post-BaB reserves included)
    // still wins through the existing production min(configured,authority).
    let entry = start + Duration::from_secs(65);
    let shorter = Duration::from_secs(100);
    assert_eq!(
        resolve_overall_deadline(entry, shorter, authority).unwrap(),
        Some(entry + shorter)
    );
    let near_end = start + Duration::from_secs(500);
    assert_eq!(
        resolve_overall_deadline(near_end, shorter, authority).unwrap(),
        Some(original_solver)
    );
}

#[test]
fn historical_authority_preserves_expiry_even_for_an_unbounded_inner_timeout() {
    let start = Instant::now();
    let original_solver = start + Duration::from_secs(15);
    let after_expiry = start + Duration::from_secs(16);
    let authority = beta_solver_authority_deadline(None, Some(original_solver));
    for configured in [Duration::ZERO, Duration::from_secs(15)] {
        let resolved = resolve_overall_deadline(after_expiry, configured, authority)
            .unwrap()
            .unwrap();
        assert_eq!(resolved, original_solver);
        assert!(
            resolved <= after_expiry,
            "expiry cannot become a fresh grant"
        );
        assert_eq!(
            resolved.saturating_duration_since(after_expiry),
            Duration::ZERO
        );
    }
    assert_eq!(
        resolve_overall_deadline(after_expiry, Duration::ZERO, None).unwrap(),
        None,
        "interactive unbounded calls retain their existing contract"
    );
}

#[test]
fn explicit_short_grace_flush_and_proof_tail_authorities_keep_their_existing_routes() {
    let start = Instant::now();
    let short = internal_timeout_decision("safenlp_2024", 20, Some(OsStr::new("1")));
    let proof = apply_postbab_wrapper_timeout_policy(
        internal_timeout_decision("ordinary_fixture", 900, None),
        900,
        PostBabWrapperRoute::ProofTailByPreset,
    );
    let flush = apply_results_flush_grace_policy(
        internal_timeout_decision("ordinary_fixture", 900, None),
        900,
        Some(2),
    );
    for (decision, budget) in [(short, 20), (proof, 900), (flush, 900)] {
        let instance = start + Duration::from_secs(budget);
        let route = beta_phase_deadline(decision, instance).unwrap();
        let authority = beta_solver_authority_deadline(
            Some(route),
            Some(solver_phase_deadline(decision, instance)),
        );
        assert_eq!(authority, Some(route), "an explicit route is unchanged");
        let entry = route + Duration::from_secs(1);
        assert_eq!(
            resolve_overall_deadline(
                entry,
                Duration::from_secs(decision.solver_timeout_secs),
                authority
            )
            .unwrap(),
            Some(route),
            "an expired explicit route also stays expired"
        );
    }
    // The short route deliberately owns its scored boundary. The new
    // historical fallback must not replace20 by its nominal19s solver grant.
    let instance = start + Duration::from_secs(20);
    assert_ne!(
        beta_phase_deadline(short, instance),
        Some(solver_phase_deadline(short, instance))
    );
}

#[test]
fn absent_authority_and_zero_scored_budget_have_no_new_time_grant() {
    assert_eq!(beta_solver_authority_deadline(None, None), None);
    let start = Instant::now();
    let zero = internal_timeout_decision("ordinary_fixture", 0, None);
    assert_eq!(zero.solver_timeout_secs, 0);
    let authority = beta_solver_authority_deadline(
        beta_phase_deadline(zero, start),
        Some(solver_phase_deadline(zero, start)),
    );
    assert_eq!(authority, Some(start));
    assert_eq!(
        resolve_overall_deadline(start, Duration::ZERO, authority).unwrap(),
        Some(start),
        "a scored zero grant cannot become interactive unbounded work"
    );
}

#[test]
fn unrepresentable_configured_budget_is_refused_without_overflow_or_extension() {
    let entry = Instant::now();
    let authority = Some(entry);
    assert!(entry.checked_add(Duration::MAX).is_none());
    assert!(resolve_overall_deadline(entry, Duration::MAX, authority).is_err());
}
