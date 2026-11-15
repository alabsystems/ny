// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::scope::{BoxRegion, NumericalSemantics, ScopeIdentity, SCOPE_SCHEMA_VERSION};
use crate::certify::work::{LeaseClass, WorkBudget, WorkRequest};
use num_rational::BigRational;
use num_traits::Zero;
use std::time::{Duration, Instant};

fn budget() -> WorkBudget {
    WorkBudget::new(None, 16 << 20, 16 << 20, 100_000_000, 1000, Duration::ZERO).unwrap()
}

fn scope(lo: &[f64], hi: &[f64], property: Digest) -> CertificateScope {
    let digest = Digest::of_bytes(b"atlas-test", b"original-model");
    CertificateScope::new(
        ScopeIdentity {
            schema_version: SCOPE_SCHEMA_VERSION,
            model: digest,
            lowering: digest,
            coordinates: digest,
            assumptions: digest,
            numerical: NumericalSemantics::ExactReal,
        },
        BoxRegion::new(lo.to_vec(), hi.to_vec()).unwrap(),
        property,
    )
    .unwrap()
}

fn q(coefficients: &[f64], constant: f64) -> AffineObjective<'_> {
    AffineObjective {
        coefficients,
        constant,
    }
}

#[test]
fn geometry_conflict_transfers_across_objectives_only_with_exact_containment() {
    let work = budget();
    let a = q(&[1.0], 0.0);
    let b = q(&[-1.0], 0.0);
    let parent = scope(&[-2.0], &[2.0], a.digest());
    let child = scope(&[-1.0], &[1.0], b.digest());
    let rows = [
        AffineInequality {
            coefficients: &[1.0],
            upper: -1.0,
        },
        AffineInequality {
            coefficients: &[-1.0],
            upper: -1.0,
        },
    ];
    let proof = replay_conflict(
        AffineGeometry {
            scope: &parent,
            inequalities: &rows,
        },
        &[1.0, 1.0],
        ReplayOptions::default(),
        &work,
    )
    .unwrap();
    assert!(proof.slack_lower() > 1.999999999 && proof.slack_lower() <= 2.0);
    let mut atlas = RegionAtlas::new(2, &work).unwrap();
    atlas.insert_replayed(AtlasEntry::GeometryConflict(proof));
    assert!(atlas
        .match_antecedent(
            AffineGeometry {
                scope: &child,
                inequalities: &rows
            },
            ReplayOptions::default(),
            &work
        )
        .unwrap()
        .is_some());
    let expanded = scope(&[-2.0000000000000004], &[1.0], b.digest());
    assert!(atlas
        .match_antecedent(
            AffineGeometry {
                scope: &expanded,
                inequalities: &rows
            },
            ReplayOptions::default(),
            &work
        )
        .unwrap()
        .is_none());
    assert_eq!(atlas.stats().hits, 1);
}

#[test]
fn class_safe_margin_never_becomes_geometry_or_other_objective_exclusion() {
    let work = budget();
    let a = q(&[1.0], 2.0);
    let b = q(&[-1.0], -0.25);
    let sa = scope(&[-1.0], &[1.0], a.digest());
    let sb = scope(&[-1.0], &[1.0], b.digest());
    let ga = AffineGeometry {
        scope: &sa,
        inequalities: &[],
    };
    let gb = AffineGeometry {
        scope: &sb,
        inequalities: &[],
    };
    let safe = replay_margin(ga, a, &[], ReplayOptions::default(), &work).unwrap();
    assert!(safe.lower() > 0.0);
    let mut atlas = RegionAtlas::new(2, &work).unwrap();
    atlas.insert_replayed(AtlasEntry::ObjectiveExclusion(safe.exclusion().unwrap()));
    assert!(atlas
        .find_exclusion(ga, a, ReplayOptions::default(), &work)
        .unwrap()
        .is_some());
    assert!(atlas
        .find_exclusion(gb, b, ReplayOptions::default(), &work)
        .unwrap()
        .is_none());
    assert!(atlas
        .match_antecedent(gb, ReplayOptions::default(), &work)
        .unwrap()
        .is_none());
    // x=0 is an explicit unsafe witness for B.
    assert!(b.constant <= 0.0);
    let unsafe_margin = replay_margin(gb, b, &[], ReplayOptions::default(), &work).unwrap();
    assert!(unsafe_margin.exclusion().is_none());
    let tie = q(&[0.0], 0.0);
    let st = scope(&[-1.0], &[1.0], tie.digest());
    let tied = replay_margin(
        AffineGeometry {
            scope: &st,
            inequalities: &[],
        },
        tie,
        &[],
        ReplayOptions::default(),
        &work,
    )
    .unwrap();
    assert_eq!(tied.lower(), 0.0);
    assert!(tied.exclusion().is_none());
}

#[test]
fn deleting_or_changing_original_antecedents_requires_fresh_replay() {
    let work = budget();
    let s = scope(&[-2.0], &[2.0], q(&[1.0], 0.0).digest());
    let rows = [
        AffineInequality {
            coefficients: &[1.0],
            upper: -1.0,
        },
        AffineInequality {
            coefficients: &[-1.0],
            upper: -1.0,
        },
    ];
    let g = AffineGeometry {
        scope: &s,
        inequalities: &rows,
    };
    let proof = replay_conflict(g, &[1.0, 1.0], ReplayOptions::default(), &work).unwrap();
    let mut atlas = RegionAtlas::new(1, &work).unwrap();
    atlas.insert_replayed(AtlasEntry::GeometryConflict(proof));
    let weakened = AffineGeometry {
        scope: &s,
        inequalities: &rows[..1],
    };
    assert!(atlas
        .match_antecedent(weakened, ReplayOptions::default(), &work)
        .unwrap()
        .is_none());
    assert!(matches!(
        replay_conflict(weakened, &[1.0], ReplayOptions::default(), &work),
        Err(AtlasRefusal::NoContradiction)
    ));
    for field in 0..4 {
        let mut identity = s.identity().clone();
        let changed = Digest::of_bytes(b"atlas-test", b"different");
        match field {
            0 => identity.model = changed,
            1 => identity.lowering = changed,
            2 => identity.coordinates = changed,
            _ => identity.assumptions = changed,
        }
        let other = CertificateScope::new(identity, s.region().clone(), s.property()).unwrap();
        assert!(atlas
            .match_antecedent(
                AffineGeometry {
                    scope: &other,
                    inequalities: &rows
                },
                ReplayOptions::default(),
                &work
            )
            .unwrap()
            .is_none());
    }
    let mut identity = s.identity().clone();
    identity.numerical = NumericalSemantics::EnclosedExecution {
        contract: s.property(),
    };
    let other = CertificateScope::new(identity, s.region().clone(), s.property()).unwrap();
    assert!(matches!(
        atlas.match_antecedent(
            AffineGeometry {
                scope: &other,
                inequalities: &rows
            },
            ReplayOptions::default(),
            &work
        ),
        Err(AtlasRefusal::UnsupportedSemantics)
    ));
}

#[test]
fn conic_margin_improves_bound_from_a_checked_original_constraint() {
    let work = budget();
    let objective = q(&[1.0], 0.25);
    let s = scope(&[-1.0], &[1.0], objective.digest());
    let rows = [AffineInequality {
        coefficients: &[-1.0],
        upper: 0.0,
    }];
    let g = AffineGeometry {
        scope: &s,
        inequalities: &rows,
    };
    let baseline = replay_margin(g, objective, &[0.0], ReplayOptions::default(), &work).unwrap();
    let improved = replay_margin(g, objective, &[1.0], ReplayOptions::default(), &work).unwrap();
    assert!(baseline.lower() < 0.0);
    assert_eq!(improved.lower(), 0.25);
    assert!(improved.exclusion().is_some());
    // Its x>=0 antecedent is feasible; margin evidence is not geometry empty.
    assert!(matches!(
        replay_conflict(g, &[1.0], ReplayOptions::default(), &work),
        Err(AtlasRefusal::NoContradiction)
    ));
}

#[test]
fn facet_replay_checks_full_residual_and_never_trusts_target_score() {
    let work = budget();
    let s = scope(&[-1.0], &[1.0], q(&[1.0], 0.0).digest());
    let rows = [AffineInequality {
        coefficients: &[1.0],
        upper: 0.0,
    }];
    let g = AffineGeometry {
        scope: &s,
        inequalities: &rows,
    };
    let exact = replay_facet(g, rows[0], &[1.0], ReplayOptions::default(), &work).unwrap();
    assert_eq!(exact.inequality().upper, 0.0);
    let weaker = AffineInequality {
        coefficients: &[0.5],
        upper: 0.75,
    };
    assert!(replay_facet(g, weaker, &[1.0], ReplayOptions::default(), &work).is_ok());
    let invalid = AffineInequality {
        coefficients: &[1.0],
        upper: -0.25,
    };
    assert!(matches!(
        replay_facet(g, invalid, &[1.0], ReplayOptions::default(), &work),
        Err(AtlasRefusal::InvalidFacet)
    ));
    let mut atlas = RegionAtlas::new(1, &work).unwrap();
    atlas.insert_replayed(AtlasEntry::Facet(exact));
    assert!(atlas
        .find_facet(g, rows[0], ReplayOptions::default(), &work)
        .unwrap()
        .is_some());
    assert!(atlas
        .find_facet(g, invalid, ReplayOptions::default(), &work)
        .unwrap()
        .is_none());
}

#[test]
fn conic_transfer_charges_constants_residuals_and_every_source_scope() {
    let work = budget();
    let a = q(&[1.0], 2.0);
    let b = q(&[-1.0], 2.0);
    let sa = scope(&[-1.0], &[1.0], a.digest());
    let sb = scope(&[-1.0], &[1.0], b.digest());
    let pa = replay_margin(
        AffineGeometry {
            scope: &sa,
            inequalities: &[],
        },
        a,
        &[],
        ReplayOptions::default(),
        &work,
    )
    .unwrap();
    let pb = replay_margin(
        AffineGeometry {
            scope: &sb,
            inequalities: &[],
        },
        b,
        &[],
        ReplayOptions::default(),
        &work,
    )
    .unwrap();
    let target = q(&[0.125], 4.5);
    let st = scope(&[-1.0], &[1.0], target.digest());
    let g = AffineGeometry {
        scope: &st,
        inequalities: &[],
    };
    let proof = replay_conic_transfer(
        g,
        target,
        &[(&pa, 1.0), (&pb, 1.0)],
        ReplayOptions::default(),
        &work,
    )
    .unwrap();
    assert!(proof.lower() <= 2.375 && proof.lower() > 2.374);
    assert!(proof.exclusion().is_some());
    assert_eq!(proof.dependency_count(), 2);
    assert!(matches!(
        replay_conic_transfer(g, target, &[(&pa, -1.0)], ReplayOptions::default(), &work),
        Err(AtlasRefusal::InvalidMultipliers)
    ));
    let expanded = scope(&[-1.0000000000000002], &[1.0], target.digest());
    assert!(matches!(
        replay_conic_transfer(
            AffineGeometry {
                scope: &expanded,
                inequalities: &[]
            },
            target,
            &[(&pa, 1.0)],
            ReplayOptions::default(),
            &work
        ),
        Err(AtlasRefusal::ScopeMismatch)
    ));
}

#[test]
fn rounded_farkas_cancellation_retains_residual_and_feasible_witness() {
    let work = budget();
    let s = scope(&[1.0], &[1.0], q(&[1.0], 0.0).digest());
    let rows = [
        AffineInequality {
            coefficients: &[1e16],
            upper: 1e16,
        },
        AffineInequality {
            coefficients: &[1.0],
            upper: 1.0,
        },
        AffineInequality {
            coefficients: &[-1e16],
            upper: -1e16,
        },
    ];
    // Exact combined coefficient is1 even though a naive RN reduction yields0.
    let exact = rows.iter().fold(BigRational::zero(), |sum, row| {
        sum + BigRational::from_float(row.coefficients[0]).unwrap()
    });
    assert_eq!(exact, BigRational::from_integer(1.into()));
    for row in rows {
        assert!(row.coefficients[0] <= row.upper);
    }
    assert!(matches!(
        replay_conflict(
            AffineGeometry {
                scope: &s,
                inequalities: &rows
            },
            &[1.0, 1.0, 1.0],
            ReplayOptions::default(),
            &work
        ),
        Err(AtlasRefusal::NoContradiction)
    ));
}

#[test]
fn directed_interval_operations_enclose_independent_exact_rational_oracle() {
    use super::numeric::Interval;
    let values = [
        0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        0.1,
        -0.3,
        1.0 + f64::EPSILON,
        1e100,
        -1e100,
    ];
    for &a in &values {
        for &b in &values {
            let qa = BigRational::from_float(a).unwrap();
            let qb = BigRational::from_float(b).unwrap();
            for (actual, exact) in [
                (
                    Interval::point(a).add(Interval::point(b)).unwrap(),
                    &qa + &qb,
                ),
                (
                    Interval::point(a).mul(Interval::point(b)).unwrap(),
                    &qa * &qb,
                ),
            ] {
                assert!(BigRational::from_float(actual.lo).unwrap() <= exact);
                assert!(exact <= BigRational::from_float(actual.hi).unwrap());
            }
        }
    }
    assert!(Interval::point(f64::MAX).mul(Interval::point(2.0)).is_err());
}

#[test]
fn margin_lower_bounds_enclose_exact_affine_corner_oracles() {
    let work = budget();
    let lo = [-1.0, -0.125, 0.0];
    let hi = [0.75, 0.5, 1.25];
    for seed in 0..32 {
        let coefficients = [0.1 * seed as f64 - 1.5, -0.3, 0.7];
        let objective = q(&coefficients, 0.1);
        let s = scope(&lo, &hi, objective.digest());
        let proof = replay_margin(
            AffineGeometry {
                scope: &s,
                inequalities: &[],
            },
            objective,
            &[],
            ReplayOptions::default(),
            &work,
        )
        .unwrap();
        for corner in 0..8 {
            let exact = (0..3).fold(
                BigRational::from_float(objective.constant).unwrap(),
                |sum, j| {
                    sum + BigRational::from_float(coefficients[j]).unwrap()
                        * BigRational::from_float(if corner & (1 << j) == 0 {
                            lo[j]
                        } else {
                            hi[j]
                        })
                        .unwrap()
                },
            );
            assert!(BigRational::from_float(proof.lower()).unwrap() <= exact);
        }
    }
}

#[test]
fn replay_refuses_negative_nonfinite_and_misbound_property_proposals() {
    let work = budget();
    let objective = q(&[1.0], 2.0);
    let s = scope(&[-1.0], &[1.0], objective.digest());
    let rows = [AffineInequality {
        coefficients: &[1.0],
        upper: 1.0,
    }];
    let g = AffineGeometry {
        scope: &s,
        inequalities: &rows,
    };
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            replay_conflict(g, &[bad], ReplayOptions::default(), &work),
            Err(AtlasRefusal::InvalidMultipliers)
        ));
    }
    assert!(matches!(
        replay_margin(g, q(&[-1.0], 2.0), &[0.0], ReplayOptions::default(), &work),
        Err(AtlasRefusal::PropertyMismatch)
    ));
    assert!(matches!(
        replay_margin(
            g,
            q(&[1.0, 0.0], 2.0),
            &[0.0],
            ReplayOptions::default(),
            &work
        ),
        Err(AtlasRefusal::Shape)
    ));
    let rows = [AffineInequality {
        coefficients: &[f64::NAN],
        upper: 0.0,
    }];
    assert!(matches!(
        replay_conflict(
            AffineGeometry {
                scope: &s,
                inequalities: &rows
            },
            &[1.0],
            ReplayOptions::default(),
            &work
        ),
        Err(AtlasRefusal::NonFinite)
    ));
}

#[test]
fn pool_eviction_releases_only_its_owner_and_preserves_checked_proof() {
    let work = budget();
    let objective = q(&[1.0], 2.0);
    let s = scope(&[-1.0], &[1.0], objective.digest());
    let g = AffineGeometry {
        scope: &s,
        inequalities: &[],
    };
    let first = replay_margin(g, objective, &[], ReplayOptions::default(), &work).unwrap();
    let second = replay_margin(g, objective, &[], ReplayOptions::default(), &work).unwrap();
    let mut atlas = RegionAtlas::new(1, &work).unwrap();
    atlas.insert_replayed(AtlasEntry::RegionMargin(first.clone()));
    let before = work.receipt().unwrap().live_persistent_bytes;
    atlas.insert_replayed(AtlasEntry::RegionMargin(second));
    assert_eq!(atlas.len(), 1);
    assert_eq!(atlas.stats().evicted, 1);
    assert_eq!(work.receipt().unwrap().live_persistent_bytes, before);
    drop(atlas);
    assert_eq!(
        work.receipt().unwrap().live_persistent_bytes,
        first.reserved_bytes()
    );
    assert!(first.exclusion().is_some());
    drop(first);
    assert_eq!(work.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn replay_uses_certificate_reserve_but_cannot_cross_final_deadline_or_caps() {
    let objective = q(&[1.0], 2.0);
    let s = scope(&[-1.0], &[1.0], objective.digest());
    let g = AffineGeometry {
        scope: &s,
        inequalities: &[],
    };
    let reserve = WorkBudget::new(
        Some(Instant::now() + Duration::from_secs(30)),
        1 << 20,
        1 << 20,
        100_000,
        10,
        Duration::from_secs(60),
    )
    .unwrap();
    assert!(matches!(
        reserve.lease(WorkRequest::default()),
        Err(WorkRefusal::Deadline)
    ));
    assert!(reserve.checkpoint_for(LeaseClass::Replay).is_ok());
    assert!(replay_margin(g, objective, &[], ReplayOptions::default(), &reserve).is_ok());
    let expired = WorkBudget::new(
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
        1 << 20,
        1 << 20,
        100_000,
        10,
        Duration::ZERO,
    )
    .unwrap();
    assert!(matches!(
        replay_margin(g, objective, &[], ReplayOptions::default(), &expired),
        Err(AtlasRefusal::Budget(WorkRefusal::Deadline))
    ));
    let starved = WorkBudget::new(None, 1, 1 << 20, 100_000, 10, Duration::ZERO).unwrap();
    assert!(matches!(
        replay_margin(g, objective, &[], ReplayOptions::default(), &starved),
        Err(AtlasRefusal::Budget(WorkRefusal::ScratchLimit))
    ));
    assert_eq!(starved.receipt().unwrap().admitted, 0);
    let opts = ReplayOptions {
        max_dimensions: 0,
        ..ReplayOptions::default()
    };
    assert!(matches!(
        replay_margin(g, objective, &[], opts, &budget()),
        Err(AtlasRefusal::ResourceLimit)
    ));
}

#[test]
fn conic_dependencies_remain_charged_after_source_and_pool_eviction() {
    let work = budget();
    let objective = q(&[1.0], 2.0);
    let s = scope(&[-1.0], &[1.0], objective.digest());
    let g = AffineGeometry {
        scope: &s,
        inequalities: &[],
    };
    let source = replay_margin(g, objective, &[], ReplayOptions::default(), &work).unwrap();
    let source_bytes = source.reserved_bytes();
    let target = replay_conic_transfer(
        g,
        objective,
        &[(&source, 1.0)],
        ReplayOptions::default(),
        &work,
    )
    .unwrap();
    assert_eq!(target.dependency_count(), 1);
    drop(source);
    assert_eq!(
        work.receipt().unwrap().live_persistent_bytes,
        target.reserved_bytes() + source_bytes
    );
    let mut atlas = RegionAtlas::new(1, &work).unwrap();
    atlas.insert_replayed(AtlasEntry::RegionMargin(target.clone()));
    let replacement = replay_margin(g, objective, &[], ReplayOptions::default(), &work).unwrap();
    atlas.insert_replayed(AtlasEntry::RegionMargin(replacement));
    drop(atlas);
    assert_eq!(
        work.receipt().unwrap().live_persistent_bytes,
        target.reserved_bytes() + source_bytes
    );
    assert!(target.exclusion().is_some());
    drop(target);
    assert_eq!(work.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn conic_dependency_depth_is_bounded_and_immutable_no_cycle_can_be_supplied() {
    let work = budget();
    let objective = q(&[1.0], 2.0);
    let s = scope(&[-1.0], &[1.0], objective.digest());
    let g = AffineGeometry {
        scope: &s,
        inequalities: &[],
    };
    let mut proof = replay_margin(g, objective, &[], ReplayOptions::default(), &work).unwrap();
    for _ in 0..64 {
        proof = replay_conic_transfer(
            g,
            objective,
            &[(&proof, 1.0)],
            ReplayOptions::default(),
            &work,
        )
        .unwrap();
    }
    assert!(matches!(
        replay_conic_transfer(
            g,
            objective,
            &[(&proof, 1.0)],
            ReplayOptions::default(),
            &work
        ),
        Err(AtlasRefusal::ResourceLimit)
    ));
    drop(proof);
    assert_eq!(work.receipt().unwrap().live_persistent_bytes, 0);
}
