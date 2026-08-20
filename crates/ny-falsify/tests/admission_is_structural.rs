// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! REQUIRED TEST (a): admission refuses STRUCTURALLY and CHEAPLY.
//!
//! "Cheaply" is asserted two ways, because a stopwatch alone is worthless on a
//! fast machine and worse on a throttling one: the expensive ladder stages must
//! not be ENTERED (counter assertions), and the elapsed time must stay inside
//! the 0.00-0.05 s band the sign-space lane already measures for its own
//! structural refusals.

#[path = "fixtures/support.rs"]
mod support;

use ny_falsify::strategies::{
    SpecialPoints, Square, SPECIAL_MAX_FREE_DIMS, SQUARE_MAX_FREE_DIMS, SQUARE_MIN_FREE_DIMS,
};
use ny_falsify::{
    Admission, Decline, ObjectiveQuality, Registry, SpecFacts, SpecShape, StrategyName,
};
use std::time::Duration;
use support::{box_spec, CountingLadder};

/// The measured band for a structural refusal in the shipped sign-space lane.
const REFUSAL_CEILING: Duration = Duration::from_millis(50);

fn registry() -> Registry {
    Registry::new()
        .with(Box::new(SpecialPoints))
        .with(Box::new(Square::default()))
        .armed()
}

fn decline_of(receipts: &[ny_falsify::AdmissionReceipt], who: StrategyName) -> Decline {
    match &receipts
        .iter()
        .find(|r| r.strategy == who)
        .unwrap()
        .admission
    {
        Admission::Declined(decline) => decline.clone(),
        Admission::Admitted(profile) => panic!("{who} was admitted: {profile:?}"),
    }
}

#[test]
fn a_non_box_spec_is_refused_without_touching_the_graph_or_the_model() {
    let spec = SpecFacts {
        free_dims: 3072,
        pinned_dims: 0,
        // The exact refusal the 2026 corpus produces today for every cora_2024
        // row, because the shipped validator's variable regex is legacy `X_0`
        // only and cannot read VNN-LIB 2.0's `X[0,i]`.
        shape: SpecShape::NonBoxInputAssertions {
            non_box_assertions: 6144,
        },
        disjunct_targets: 1,
        has_equality_atoms: false,
    };
    let mut ladder = CountingLadder::new(spec);
    let registry = registry();

    let started = std::time::Instant::now();
    let receipts = registry
        .admit(
            &mut ladder,
            Duration::from_mins(5),
            ObjectiveQuality::Informative,
        )
        .expect("the ladder itself did not fail");
    let elapsed = started.elapsed();

    assert_eq!(receipts.len(), 2, "one receipt per registered strategy");
    for receipt in &receipts {
        assert!(
            matches!(
                receipt.admission,
                Admission::Declined(Decline::SpecShapeUnsupported { .. })
            ),
            "{} must decline on SPEC SHAPE, got {:?}",
            receipt.strategy,
            receipt.admission
        );
        assert!(
            receipt.elapsed < REFUSAL_CEILING,
            "{} refusal took {:?}, above the measured 0.00-0.05 s band",
            receipt.strategy,
            receipt.elapsed
        );
    }

    // THE LOAD-BEARING ASSERTION. Not "it was fast" -- "it never went there".
    assert_eq!(ladder.spec_calls, 1, "the spec is parsed exactly once");
    assert_eq!(ladder.graph_calls, 0, "no strategy scanned the ONNX graph");
    assert_eq!(ladder.model_calls, 0, "no strategy loaded the model");
    assert!(
        elapsed < REFUSAL_CEILING,
        "whole admission pass took {elapsed:?}"
    );
}

#[test]
fn the_two_strategies_have_independent_ceilings_and_refuse_on_their_own() {
    // Design §5 rule 3: ceilings are PER STRATEGY. A shared ceiling is a
    // measured regression, not a simplification -- letting the LP lane inherit
    // the STE lane's 32768 free-unit cap took model_30_idx_1703_eps_15 from
    // 1.3 s to 139.4 s for zero accepted flips.
    assert_ne!(
        SPECIAL_MAX_FREE_DIMS, SQUARE_MAX_FREE_DIMS,
        "the ported strategies must not share a free-dimension ceiling"
    );
    const _: () = assert!(SPECIAL_MAX_FREE_DIMS > SQUARE_MAX_FREE_DIMS);

    // One free dimension: `special` still works (eight points is eight points),
    // `square` declines on its own floor because a "block" would be a single
    // coordinate and corner enumeration -- which ny already ships -- dominates.
    let mut ladder = CountingLadder::new(box_spec(1));
    let receipts = registry()
        .admit(
            &mut ladder,
            Duration::from_mins(1),
            ObjectiveQuality::Informative,
        )
        .unwrap();
    assert!(matches!(
        receipts
            .iter()
            .find(|r| r.strategy == StrategyName::SpecialPoints)
            .unwrap()
            .admission,
        Admission::Admitted(_)
    ));
    assert_eq!(
        decline_of(&receipts, StrategyName::Square),
        Decline::FreeDimsBelowFloor {
            free: 1,
            floor: SQUARE_MIN_FREE_DIMS
        }
    );

    // Above `square`'s ceiling but far below `special`'s: exactly the case a
    // shared ceiling would get wrong in one direction or the other.
    let wide = SQUARE_MAX_FREE_DIMS + 1;
    let mut ladder = CountingLadder::new(box_spec(wide));
    let receipts = registry()
        .admit(
            &mut ladder,
            Duration::from_mins(1),
            ObjectiveQuality::Informative,
        )
        .unwrap();
    assert!(matches!(
        receipts
            .iter()
            .find(|r| r.strategy == StrategyName::SpecialPoints)
            .unwrap()
            .admission,
        Admission::Admitted(_),
    ));
    assert_eq!(
        decline_of(&receipts, StrategyName::Square),
        Decline::FreeDimsAboveCeiling {
            free: wide,
            ceiling: SQUARE_MAX_FREE_DIMS
        }
    );
    assert_eq!(ladder.graph_calls, 0);
    assert_eq!(ladder.model_calls, 0);
}

#[test]
fn both_measured_calibration_regimes_are_inside_the_ceilings() {
    // soundnessbench model_36 at 128 free inputs and
    // traffic_signs model_48_idx_10495_eps_10 at 6912 -- the two rows `square`
    // actually won -- plus collins_rul if_then_7levels_w20 at 200, a `special`
    // win at a dimension where ny's corner lane (cap 5) refuses outright.
    for free in [2usize, 5, 128, 200, 6912] {
        let mut ladder = CountingLadder::new(box_spec(free));
        let receipts = registry()
            .admit(&mut ladder, Duration::from_mins(1), ObjectiveQuality::Flat)
            .unwrap();
        for receipt in &receipts {
            assert!(
                matches!(receipt.admission, Admission::Admitted(_)),
                "{} refused at {free} free dims: {:?}",
                receipt.strategy,
                receipt.admission
            );
        }
    }
}

#[test]
fn a_flat_objective_does_not_decline_either_ported_strategy() {
    // Both are ValueOnly. `square` in particular exists FOR the flat case: on
    // traffic_signs the margin is exactly -1.0 at every in-box point, over
    // exactly two distinct output values, so every gradient-shaped strategy is
    // blind there and declining `square` too would leave nothing at all.
    let mut ladder = CountingLadder::new(box_spec(6912));
    let receipts = registry()
        .admit(&mut ladder, Duration::from_mins(8), ObjectiveQuality::Flat)
        .unwrap();
    for receipt in receipts {
        match receipt.admission {
            Admission::Admitted(profile) => assert_eq!(
                profile.objective,
                ny_falsify::ObjectiveRequirement::ValueOnly,
                "{} must not require gradient signal",
                receipt.strategy
            ),
            other => panic!(
                "{} declined on a flat objective: {other:?}",
                receipt.strategy
            ),
        }
    }
}

#[test]
fn a_decline_is_a_receipt_and_disarmed_is_not_a_structural_claim() {
    // Design §5 rule 4. A structural decline licenses "this family is out of
    // reach"; `Disarmed` licenses nothing, and the type says which is which.
    assert!(Decline::SpecShapeUnsupported {
        want: SpecShape::BoxInputs,
        got: SpecShape::NonBoxInputAssertions {
            non_box_assertions: 1
        }
    }
    .is_structural());
    assert!(Decline::FreeDimsAboveCeiling {
        free: 9,
        ceiling: 8
    }
    .is_structural());
    assert!(!Decline::Disarmed.is_structural());
}

#[test]
fn admission_cannot_read_a_path_a_category_or_a_preset() {
    // Design §5 rule 1 and rule 5, enforced on the SOURCE rather than by
    // convention: `run_sign_space_lane` already claims "Nothing here looks at a
    // filename, a directory or a benchmark category" in a doc comment. This
    // turns that sentence into a build failure.
    //
    // Comments are stripped first -- the module doc necessarily uses these
    // words to say that they are forbidden.
    let source = include_str!("../src/admission.rs");
    let code: String = source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase();

    for forbidden in [
        "path",
        "filename",
        "file_name",
        "category",
        "preset",
        "benchmark",
        "instance_name",
        "directory",
        "onnx_path",
        "vnnlib_path",
    ] {
        assert!(
            !code.contains(forbidden),
            "`{forbidden}` appears in admission.rs outside a comment: admission must be a \
             pure function of (parsed spec, graph structure, remaining budget)"
        );
    }
}
