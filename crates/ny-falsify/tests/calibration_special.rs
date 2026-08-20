// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! REQUIRED TEST (b), part 1: `special` reproduces its calibration win.
//!
//! The calibration
//! (`reports/falsification_audit/selftest_calibration.json`, 60 s, 100
//! known-SAT rows) attributes **34 of the 75 refutations** to `special`, and on
//! every single one of them the effort record is `"strategies_run":
//! {"special": 8}` at 0.016-0.686 s wall. So the win has a very sharp
//! signature: eight points, one batch, no search. That signature is what these
//! tests pin.

#[path = "fixtures/special_golden.rs"]
mod golden;
#[path = "fixtures/support.rs"]
mod support;

use ny_falsify::strategies::{SpecialPoints, SPECIAL_PATTERNS};
use ny_falsify::{
    Budget, ParamSpace, Proposal, Score, SearchBox, SearchState, StallRule, Strategy, StrategyName,
    WorkUnit,
};
use std::time::{Duration, Instant};
use support::PredicateOracle;

fn budget(batch: usize) -> Budget {
    Budget {
        deadline: Instant::now() + Duration::from_secs(30),
        batch,
        params: ParamSpace {
            free_dims_ceiling: usize::MAX,
            max_points: 1 << 20,
            max_restarts: 1,
        },
        stall_rule: StallRule::new(WorkUnit::BatchesWithoutNewBest, 1),
    }
}

/// Bit-for-bit, not approximately. A candidate that differs from the scored
/// point by one ULP after float32 rounding is a different point.
fn assert_bits(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{what}: coordinate {index} is {a:?} ({:#x}), expected {e:?} ({:#x})",
            a.to_bits(),
            e.to_bits()
        );
    }
}

#[test]
fn the_eight_patterns_are_bit_identical_to_the_shipped_python_portfolio() {
    let domain = SearchBox::new(&golden::LOW, &golden::HIGH).unwrap();

    assert_eq!(domain.free_dims(), golden::FREE_INDICES.len());
    assert_eq!(domain.pinned_dims(), 1, "index 3 is pinned at 2.0 == 2.0");
    assert_bits(
        domain.centre_free(),
        &golden::RAW_FREE_PATTERNS[2],
        "the snapped centre",
    );

    let patterns = SpecialPoints::patterns(&domain);
    assert_eq!(patterns.len(), 8);
    assert_eq!(patterns.len(), SPECIAL_PATTERNS.len());

    for (index, (pattern, expected)) in patterns.iter().zip(&golden::RAW_FREE_PATTERNS).enumerate()
    {
        assert_bits(
            pattern,
            expected,
            &format!("raw pattern {index} ({})", SPECIAL_PATTERNS[index]),
        );
        assert_bits(
            &domain.materialise(pattern),
            &golden::MATERIALISED_POINTS[index],
            &format!("materialised point {index} ({})", SPECIAL_PATTERNS[index]),
        );
    }
}

#[test]
fn the_pinned_coordinate_keeps_its_exact_constant_and_free_ones_land_on_the_f32_grid() {
    // This is what makes cctsdb-style equality-pinned specs falsifiable at all:
    // a pinned coordinate whose constant is not float32-representable must
    // reach the oracle as the exact f64 the assertions were checked against.
    for point in &golden::MATERIALISED_POINTS {
        assert_eq!(point[3].to_bits(), 2.0f64.to_bits(), "pinned coordinate");
        for (index, &value) in point.iter().enumerate() {
            if index == 3 {
                continue;
            }
            assert_eq!(
                f64::from(value as f32).to_bits(),
                value.to_bits(),
                "free coordinate {index} = {value:?} is not on the float32 grid"
            );
            assert!(
                value >= golden::LOW[index] && value <= golden::HIGH[index],
                "coordinate {index} = {value:?} left the declared box"
            );
        }
    }
}

#[test]
fn special_wins_in_eight_points_and_one_batch() {
    // The calibration signature, reproduced: the violating point is one of the
    // eight declared patterns, and the strategy spends eight points finding it.
    // Pattern 4 (`parity_even_high`) is deliberately not the first pattern, so
    // "it found it" cannot be an accident of ordering.
    let domain = SearchBox::new(&golden::LOW, &golden::HIGH).unwrap();
    let target = golden::MATERIALISED_POINTS[4];
    let mut oracle = PredicateOracle::new(move |point: &[f64]| {
        let hit = point
            .iter()
            .zip(&target)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        // A two-level objective: this is the traffic_signs shape, where the
        // margin is exactly -1.0 everywhere and nothing hill-climbs.
        Score {
            steer: if hit { 0.0 } else { -1.0 },
            holds: hit,
        }
    });

    let mut state = SearchState::at_centre(&domain);
    let proposal = SpecialPoints.search(&domain, &mut oracle, &budget(256), &mut state);

    let Proposal::Candidate(candidate) = proposal else {
        panic!("special did not find the declared pattern: {proposal:?}");
    };
    assert_eq!(candidate.found_by(), StrategyName::SpecialPoints);
    assert_bits(candidate.inputs(), &target, "the proposed input vector");
    assert_eq!(
        candidate.effort().points,
        8,
        "the calibration's `strategies_run: {{special: 8}}`"
    );
    assert_eq!(candidate.effort().batches, 1, "one oracle call");
    assert_eq!(oracle.calls, 1);
    assert_eq!(oracle.points, 8);
}

#[test]
fn special_wins_at_a_dimension_where_nys_corner_lane_refuses_outright() {
    // `collins_rul` `if_then_7levels_w20`, a calibration `special` win, has
    // 200 free inputs. ny's nearest lane, `low_dim_ort_corner_falsify`, caps at
    // UPFRONT_CORNER_MAX_VARIABLE_DIMS = 5 and returns an empty seed vector
    // above it, so on that row ny has no corner search at all -- and exhaustive
    // enumeration is not an option either, at 2^200 vertices.
    const N: usize = 200;
    const NY_CORNER_LANE_CAP: usize = 5;
    const _: () = assert!(N > NY_CORNER_LANE_CAP);

    let lo = vec![-1.0f64; N];
    let hi = vec![1.0f64; N];
    let domain = SearchBox::new(&lo, &hi).unwrap();

    // Violating region: the all-high vertex. Reachable by pattern 1.
    let mut oracle = PredicateOracle::new(|point: &[f64]| {
        let hit = point.iter().all(|&v| v == 1.0);
        Score {
            steer: if hit { 0.0 } else { -1.0 },
            holds: hit,
        }
    });
    let mut state = SearchState::at_centre(&domain);
    let proposal = SpecialPoints.search(&domain, &mut oracle, &budget(256), &mut state);

    let Proposal::Candidate(candidate) = proposal else {
        panic!("special missed the all-high vertex at {N} free dims: {proposal:?}");
    };
    assert_eq!(candidate.effort().points, 8);
    assert!(candidate.inputs().iter().all(|&v| v == 1.0));
}

#[test]
fn half_the_patterns_are_not_vertices_which_is_why_a_corner_lane_cannot_stand_in() {
    // centre, low_centre_midpoint, high_centre_midpoint have NO coordinate at a
    // bound. A vertex enumerator reaches none of them at any budget.
    let lo = vec![-1.0f64, -1.0, -1.0, -1.0];
    let hi = vec![1.0f64, 1.0, 1.0, 1.0];
    let domain = SearchBox::new(&lo, &hi).unwrap();
    let patterns = SpecialPoints::patterns(&domain);

    for &index in &[2usize, 6, 7] {
        let interior = patterns[index]
            .iter()
            .enumerate()
            .all(|(i, &v)| v > lo[i] && v < hi[i]);
        assert!(
            interior,
            "pattern {index} ({}) should be strictly interior, got {:?}",
            SPECIAL_PATTERNS[index], patterns[index]
        );
    }
    for &index in &[0usize, 1, 3, 4, 5] {
        let vertex = patterns[index]
            .iter()
            .enumerate()
            .all(|(i, &v)| v == lo[i] || v == hi[i]);
        assert!(
            vertex,
            "pattern {index} ({}) should be a vertex",
            SPECIAL_PATTERNS[index]
        );
    }
}
