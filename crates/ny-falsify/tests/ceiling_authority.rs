// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #falsify-ceiling-authority: the threaded free-dimension ceiling BINDS.
//!
//! `ParamSpace::free_dims_ceiling` is documented "**Per strategy, never
//! shared**", and the reason it gives is ny's own MEASURED regression: letting
//! the LP lane inherit the STE lane's 32768 free-unit cap took
//! `model_30_idx_1703_eps_15` from 1.3 s to 139.4 s for 0 accepted flips.
//!
//! The value was computed in `admit`, threaded through `AdmissionProfile` ->
//! `Slice` -> `Budget`, handed to `Strategy::search` -- and read by NOBODY.
//! What actually enforced the ceiling was each strategy's private constant,
//! re-checked inside its own `admit`. A slice carrying a NARROWED ceiling was
//! silently ignored. That is the same "state faithfully computed and then never
//! consulted" shape as the measured instance the doc cites.
//!
//! These tests pin both halves:
//!   * a narrowed per-slice ceiling now refuses the walk at zero oracle cost;
//!   * the production ceiling is unchanged, so nothing that ships moves.

#[path = "fixtures/support.rs"]
mod support;

use ny_falsify::strategies::{
    SpecialPoints, Square, SPECIAL_MAX_FREE_DIMS, SQUARE_MAX_FREE_DIMS, SQUARE_STALL_BATCHES,
};
use ny_falsify::{
    Admission, AdmissionContext, Arming, Budget, Decline, ObjectiveQuality, ParamSpace, Proposal,
    Score, SearchBox, SearchState, StallRule, Strategy, WorkUnit,
};
use std::time::{Duration, Instant};
use support::{box_spec, PredicateOracle};

/// Free dimensions in the box handed to `search`.
const N: usize = 64;
/// A ceiling a hypothetical allocator narrowed to below `N`.
const NARROWED: usize = 8;

fn domain() -> SearchBox {
    SearchBox::new(&vec![0.0f64; N], &vec![1.0f64; N]).expect("box")
}

/// Never satisfied, so nothing can short-circuit on a lucky hit: the only way
/// out is the ceiling refusal or a spent budget.
fn never(_point: &[f64]) -> Score {
    Score {
        steer: -1.0,
        holds: false,
    }
}

fn budget_with_ceiling(free_dims_ceiling: usize) -> Budget {
    Budget {
        // Generous: the point is that the ceiling refuses BEFORE the clock does.
        deadline: Instant::now() + Duration::from_secs(30),
        batch: 32,
        params: ParamSpace {
            free_dims_ceiling,
            max_points: 1 << 22,
            max_restarts: 64,
        },
        stall_rule: StallRule::new(
            WorkUnit::BlockBatchesWithoutImprovement,
            SQUARE_STALL_BATCHES,
        ),
    }
}

#[test]
fn square_honours_a_narrowed_per_slice_ceiling_at_zero_oracle_cost() {
    let domain = domain();
    let mut oracle = PredicateOracle::new(never);
    let mut state = SearchState::at_centre(&domain);
    let mut square = Square::default();

    let proposal = square.search(
        &domain,
        &mut oracle,
        &budget_with_ceiling(NARROWED),
        &mut state,
    );

    let Proposal::Exhausted(effort) = proposal else {
        panic!("a slice whose ceiling is below the box must not produce a candidate");
    };
    assert_eq!(
        effort.points, 0,
        "the ceiling must refuse before any point is constructed"
    );
    assert_eq!(
        oracle.calls, 0,
        "the ceiling must refuse before the oracle is touched -- this is the \
         1.3 s -> 139.4 s shape: {N} free dims walked under a ceiling of {NARROWED}"
    );
}

#[test]
fn special_points_honours_a_narrowed_per_slice_ceiling_at_zero_oracle_cost() {
    let domain = domain();
    let mut oracle = PredicateOracle::new(never);
    let mut state = SearchState::at_centre(&domain);
    let mut special = SpecialPoints;

    let proposal = special.search(
        &domain,
        &mut oracle,
        &budget_with_ceiling(NARROWED),
        &mut state,
    );

    let Proposal::Exhausted(effort) = proposal else {
        panic!("a slice whose ceiling is below the box must not produce a candidate");
    };
    assert_eq!(effort.points, 0);
    assert_eq!(
        oracle.calls, 0,
        "corner enumeration must not spend scored oracle forwards under a \
         ceiling it exceeds"
    );
}

/// The other half of the claim: nothing that SHIPS moves.
///
/// Production never narrows the ceiling -- `admit` writes the strategy's own
/// constant into `params.free_dims_ceiling` and refuses anything above it, and
/// `SpecFacts::free_dims` is `domain.free_dims()` off the very box passed to
/// `search`. So the new branch is unreachable on every production path and the
/// walk is byte-identical.
#[test]
fn the_shipped_ceiling_is_the_strategys_own_constant_and_the_walk_is_unchanged() {
    for (name, admitted_ceiling, expected) in [
        (
            "square",
            admit_ceiling(&Square::default(), N),
            SQUARE_MAX_FREE_DIMS,
        ),
        (
            "special",
            admit_ceiling(&SpecialPoints, N),
            SPECIAL_MAX_FREE_DIMS,
        ),
    ] {
        assert_eq!(
            admitted_ceiling,
            Some(expected),
            "{name} must publish its OWN constant as the slice ceiling"
        );
        assert!(
            N <= expected,
            "{name}: an admitted box is never above the ceiling it was admitted under"
        );
    }

    // And with the shipped ceiling in the budget the walk actually runs.
    let domain = domain();
    let mut oracle = PredicateOracle::new(never);
    let mut state = SearchState::at_centre(&domain);
    let mut square = Square::default();
    let mut shipped = budget_with_ceiling(SQUARE_MAX_FREE_DIMS);
    shipped.deadline = Instant::now() + Duration::from_millis(200);
    let proposal = square.search(&domain, &mut oracle, &shipped, &mut state);
    let Proposal::Exhausted(effort) = proposal else {
        panic!("the surrogate is never satisfied");
    };
    assert!(
        effort.points > 0 && oracle.calls > 0,
        "under the shipped ceiling the walk must be untouched"
    );
}

/// `admit`'s published ceiling for a box of `free` dimensions, if admitted.
fn admit_ceiling(strategy: &dyn Strategy, free: usize) -> Option<usize> {
    let spec = box_spec(free);
    let ctx = AdmissionContext::new(
        &spec,
        Duration::from_secs(30),
        Arming::Armed,
        ObjectiveQuality::Flat,
    );
    match strategy.admit(&ctx) {
        Admission::Admitted(profile) => Some(profile.params.free_dims_ceiling),
        Admission::Declined(d) => {
            let _: Decline = d;
            None
        }
    }
}
