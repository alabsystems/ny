// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Structural aggregation equivalence; bound arithmetic belongs to the
//! original certified producers and is not reinterpreted here.
use super::*;
use crate::certify::margin_row;
use crate::margin_row::RoundMode;

fn root(class: usize, margin: f64) -> ClassCover {
    margin_row::root_class(RoundMode::Outward, class, margin).unwrap()
}
fn reference(
    required: &[usize],
    covers: &[ClassCover],
) -> Result<(Vec<usize>, f64), AuthorizeRefusal> {
    if required.is_empty() {
        return Err(AuthorizeRefusal::NoClasses);
    }
    let mut minimum = f64::INFINITY;
    for &class in required {
        let mut best: Option<f64> = None;
        for cover in covers {
            if let Some(margin) = cover.covers(class) {
                best = Some(best.map_or(margin, |old| old.max(margin)));
            }
        }
        let best = best.ok_or(AuthorizeRefusal::Uncovered { class })?;
        minimum = minimum.min(best);
    }
    let mut classes = required.to_vec();
    classes.sort_unstable();
    classes.dedup();
    Ok((classes, minimum))
}
fn equivalent(required: &[usize], covers: &[ClassCover]) {
    let actual = authorize_unsat(TokenScope::Instance, required, covers, &GuardReport::new());
    match (reference(required, covers), actual) {
        (Ok((classes, margin)), Ok(token)) => {
            assert_eq!(token.classes(), classes);
            assert_eq!(token.min_margin().to_bits(), margin.to_bits());
            assert_eq!(token.source(), UnsatSource::Authorized);
        }
        (Err(expected), Err(actual)) => assert_eq!(actual, expected),
        (expected, actual) => {
            panic!("aggregation differs: expected={expected:?} actual={actual:?}")
        }
    }
}

#[test]
fn token_aggregation_original_sparse_sorted_classes_have_identical_authority_and_margin_bits() {
    let required = [0, 3, 11, usize::MAX - 1, usize::MAX];
    let covers: Vec<_> = required
        .iter()
        .enumerate()
        .map(|(i, &c)| root(c, 0.125 + (i as f64) * 0.25))
        .collect();
    assert!(aligned_single_class_covers(&required, &covers));
    equivalent(&required, &covers);
}
#[test]
fn token_aggregation_duplicate_reordered_and_extra_discharge_keep_original_best_cover() {
    for required in [vec![1, 3], vec![3, 1], vec![1, 1, 3]] {
        let covers = vec![root(1, 0.125), root(3, 0.5), root(1, 0.75)];
        assert!(!aligned_single_class_covers(&required, &covers));
        equivalent(&required, &covers);
    }
    let covers = [root(3, 0.5), root(1, 0.75)];
    assert!(!aligned_single_class_covers(&[1, 3], &covers));
    equivalent(&[1, 3], &covers);
}
#[test]
fn token_aggregation_zero_negative_nonfinite_and_missing_original_class_cannot_authorize() {
    for bad in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY] {
        let Some(bound) = margin_row::outward_bound(RoundMode::Outward, bad) else {
            // The original adapter cannot create a nonfinite payload at all.
            assert!(margin_row::root_class(RoundMode::Outward, 3, bad).is_none());
            equivalent(&[1, 3], &[root(1, 0.25)]);
            continue;
        };
        let covers = [root(1, 0.25), ClassCover::Root { class: 3, bound }];
        assert!(aligned_single_class_covers(&[1, 3], &covers));
        equivalent(&[1, 3], &covers);
        assert!(
            authorize_unsat(TokenScope::Instance, &[1, 3], &covers, &GuardReport::new()).is_err()
        );
    }
    equivalent(&[1, 3, 5], &[root(1, 0.5), root(5, 0.75)]);
    equivalent(&[], &[]);
}
#[test]
fn token_aggregation_class_part_tree_scope_and_guard_order_are_preserved() {
    let part = authorize_unsat(
        TokenScope::ClassPart,
        &[1, 3],
        &[root(1, 0.25), root(3, 0.5)],
        &GuardReport::new(),
    )
    .unwrap();
    let covers = [ClassCover::Part(part)];
    assert!(!aligned_single_class_covers(&[1, 3], &covers));
    equivalent(&[1, 3], &covers);
    let subdomain = authorize_unsat(
        TokenScope::Subdomain,
        &[1],
        &[root(1, 0.25)],
        &GuardReport::new(),
    )
    .unwrap();
    assert!(!subdomain.publishable());
    equivalent(&[1], &[ClassCover::Part(subdomain)]);
    let mut guards = GuardReport::new();
    guards.trip(GuardId::SpecAgreement, "original objective disagrees");
    for required in [&[][..], &[1][..]] {
        assert!(matches!(
            authorize_unsat(TokenScope::Instance, required, &[root(1, 0.25)], &guards),
            Err(AuthorizeRefusal::GuardTripped(_))
        ));
    }
}
#[test]
fn token_aggregation_full_vocabulary_sized_synthetic_cover_uses_aligned_scan() {
    let required: Vec<usize> = (0..50_303).map(|c| c * 2 + 1).collect();
    let covers: Vec<_> = required
        .iter()
        .map(|&class| root(class, 0.5 + ((class % 17) as f64) * 0.03125))
        .collect();
    assert!(aligned_single_class_covers(&required, &covers));
    let token = authorize_unsat(
        TokenScope::Instance,
        &required,
        &covers,
        &GuardReport::new(),
    )
    .unwrap();
    assert_eq!(token.classes(), required);
    assert_eq!(token.min_margin().to_bits(), 0.5_f64.to_bits());
    assert!(token.covers_all(&required));
    assert!(token.covers_all(&required.iter().rev().copied().collect::<Vec<_>>()));
    assert!(!token.covers_all(&[0]));
    assert!(!token.covers_all(&[usize::MAX]));
}
#[test]
fn token_aggregation_sorted_and_arbitrary_membership_match_independent_contains() {
    let token = authorize_unsat(
        TokenScope::Instance,
        &[0, 3, 9, usize::MAX],
        &[
            root(0, 0.25),
            root(3, 0.5),
            root(9, 0.75),
            root(usize::MAX, 1.0),
        ],
        &GuardReport::new(),
    )
    .unwrap();
    for classes in [
        vec![],
        vec![0, 0, 3, 9],
        vec![9, 3, 0, 9],
        vec![0, 1, 9],
        vec![usize::MAX, 0],
        vec![usize::MAX],
    ] {
        assert_eq!(
            token.covers_all(&classes),
            classes.iter().all(|class| token.classes().contains(class))
        );
    }
    let legacy = legacy_uncovered_unsat(LegacyUnsatAuthority::TestFixture);
    assert!(legacy.covers_all(&[]));
    assert!(!legacy.covers_all(&[0]));
}

fn tree_cover(classes: &[usize], margin: f64, empty: bool) -> ClassCover {
    let mut ledger = crate::certify::CoverLedger::new();
    for positive in [false, true] {
        let path = crate::certify::LeafPath::from_literals(vec![crate::certify::CoverLiteral {
            site: crate::certify::CoverSite::Head { index: 5 },
            positive,
        }])
        .unwrap();
        let evidence = if empty {
            margin_row::empty_evidence(RoundMode::Outward, margin)
        } else {
            margin_row::bound_evidence(RoundMode::Outward, margin)
        };
        ledger.leaf(Some(path), evidence);
    }
    ClassCover::Tree(ledger.prove(classes).unwrap())
}
#[test]
fn token_aggregation_canonical_single_real_tree_matches_original_membership_and_margin_bits() {
    let required = [0, 3, 4097, usize::MAX];
    let covers = [tree_cover(&required, 0.75, false)];
    assert!(canonical_single_tree_cover(&required, &covers));
    for scope in [
        TokenScope::Instance,
        TokenScope::ClassPart,
        TokenScope::Subdomain,
    ] {
        let token = authorize_unsat(scope, &required, &covers, &GuardReport::new()).unwrap();
        assert_eq!(token.scope(), scope);
        assert_eq!(token.classes(), required);
        assert_eq!(token.min_margin().to_bits(), 0.75_f64.to_bits());
    }
    equivalent(&required, &covers);
}
#[test]
fn token_aggregation_single_tree_missing_extra_duplicate_and_reordered_classes_keep_fallback() {
    let covers = [tree_cover(&[1, 3, 9], 0.75, false)];
    for required in [
        vec![1, 3],
        vec![1, 3, 9, 11],
        vec![3, 1, 9],
        vec![1, 1, 3, 9],
        vec![],
    ] {
        assert!(!canonical_single_tree_cover(&required, &covers));
        equivalent(&required, &covers);
    }
}
#[test]
fn token_aggregation_tree_plus_other_cover_preserves_original_best_per_class() {
    let covers = [
        tree_cover(&[1, 3], 0.25, false),
        root(1, 0.75),
        root(3, 0.5),
    ];
    assert!(!canonical_single_tree_cover(&[1, 3], &covers));
    equivalent(&[1, 3], &covers);
    let token =
        authorize_unsat(TokenScope::Instance, &[1, 3], &covers, &GuardReport::new()).unwrap();
    assert_eq!(token.min_margin().to_bits(), 0.5_f64.to_bits());
}
#[test]
fn token_aggregation_canonical_empty_leaf_tree_preserves_generic_infinite_minimum_semantics() {
    let covers = [tree_cover(&[1, 3], 0.5, true)];
    assert!(canonical_single_tree_cover(&[1, 3], &covers));
    equivalent(&[1, 3], &covers);
    let token =
        authorize_unsat(TokenScope::Subdomain, &[1, 3], &covers, &GuardReport::new()).unwrap();
    assert_eq!(token.min_margin(), f64::INFINITY);
    // Actual prepared original-box publication separately requires a finite
    // selected bound; this structural optimization changes no cover policy.
}
#[test]
fn token_aggregation_canonical_tree_downgrade_guard_order_and_no_classes_are_unchanged() {
    let covers = [tree_cover(&[1, 3], 0.5, false)];
    let mut guards = GuardReport::new();
    guards.trip(GuardId::Containment, "original root replay mismatch");
    for required in [&[][..], &[1, 3][..], &[1, 3, 9][..]] {
        assert!(matches!(
            authorize_unsat(TokenScope::Instance, required, &covers, &guards),
            Err(AuthorizeRefusal::GuardTripped(_))
        ));
    }
    assert_eq!(
        authorize_unsat(TokenScope::Instance, &[], &covers, &GuardReport::new()),
        Err(AuthorizeRefusal::NoClasses)
    );
}
#[test]
fn token_aggregation_canonical_complete_tree_full50303_vocabulary_keeps_exact_sorted_usize_ids() {
    let required: Vec<usize> = (0..50_303).map(|class| class * 2 + 1).collect();
    let covers = [tree_cover(&required, 0.75, false)];
    assert!(canonical_single_tree_cover(&required, &covers));
    let token = authorize_unsat(
        TokenScope::Instance,
        &required,
        &covers,
        &GuardReport::new(),
    )
    .unwrap();
    assert_eq!(token.classes(), required);
    assert_eq!(token.min_margin().to_bits(), 0.75_f64.to_bits());
    assert!(token.covers_all(&required));
    assert!(!token.covers_all(&[0]));
    assert!(!token.covers_all(&[usize::MAX]));
}
