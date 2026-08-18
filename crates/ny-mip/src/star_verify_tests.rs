// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Candidate-search tests. The load-bearing ones compare internal-model evidence against
//! brute force over a dense input grid; they are consistency checks, not concrete authority.

use std::time::{Duration, Instant};

use ndarray::{array, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::zonotope::{Star, ZonotopeTensor};

use super::{
    few_unstable_remaining, star_search_unsafe_conjunction, star_search_with_clock, star_verify,
    unsafe_closure_report_outcome, unsafe_closure_report_outcome_at, LeafOutcome, SearchProperty,
    StarBudget, StarLayer, StarSpec, StarUnsafeCandidateVerdict, StarUnsafeConjunction,
    StarUnsafeRow, StarVerdict,
};
use crate::star_lp::StarLpReport;

fn budget() -> StarBudget {
    StarBudget::new(20_000, 64, Instant::now() + Duration::from_mins(1))
}

/// Star over the box `center ± eps` with one symbol per input.
fn input_star(center: &[f32], eps: f32) -> Star {
    let values = ArrayD::from_shape_vec(IxDyn(&[center.len()]), center.to_vec()).expect("shape");
    Star::from_input_box(&values, eps)
}

#[test]
fn exact_split_switch_counts_only_the_current_network_suffix() {
    let layers = vec![
        StarLayer::Gemm {
            weight: Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            // The first two coordinates straddle zero; the third is stable.
            bias: Some(array![0.0, 0.0, 2.0]),
        },
        StarLayer::Relu,
        StarLayer::Gemm {
            weight: Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 1.0]).unwrap(),
            bias: Some(array![0.0]),
        },
    ];
    let at_relu = match &layers[0] {
        StarLayer::Gemm { weight, bias } => input_star(&[0.0, 0.0], 1.0)
            .gemm(weight, bias.as_ref())
            .expect("first affine layer"),
        StarLayer::Relu => unreachable!(),
    };
    let mut b = budget();

    // There are two unstable coordinates in the whole current ReLU, so the
    // strict "below 3" switch must select exact splitting. The old helper
    // replayed layers[0] and dimension-mismatched this already-3-wide star
    // against the affine's 2-wide input, then incorrectly returned false.
    b.exact_below_unstable = 3;
    assert!(few_unstable_remaining(&at_relu, &layers, 1, 0, &b));

    // At coordinate 1 only one unstable coordinate remains in this ReLU.
    b.exact_below_unstable = 2;
    assert!(few_unstable_remaining(&at_relu, &layers, 1, 1, &b));
    assert!(!few_unstable_remaining(&at_relu, &layers, 1, 0, &b));
}

/// Evaluate the concrete network at a point.
fn eval(layers: &[StarLayer], x: &[f32]) -> Vec<f32> {
    let mut v = Array1::from(x.to_vec());
    for l in layers {
        match l {
            StarLayer::Gemm { weight, bias } => {
                let mut out = weight.dot(&v);
                if let Some(b) = bias {
                    out += b;
                }
                v = out;
            }
            StarLayer::Relu => v = v.mapv(|t| t.max(0.0)),
        }
    }
    v.to_vec()
}

/// Does every point of a dense grid over the input box satisfy every spec row?
fn brute_force_holds(
    layers: &[StarLayer],
    center: &[f32],
    eps: f32,
    spec: &StarSpec,
    steps: usize,
) -> bool {
    let n = center.len();
    let mut idx = vec![0usize; n];
    loop {
        let x: Vec<f32> = (0..n)
            .map(|i| center[i] - eps + 2.0 * eps * (idx[i] as f32) / ((steps - 1) as f32))
            .collect();
        let y = eval(layers, &x);
        for (coeffs, t) in &spec.rows {
            let m: f64 = coeffs
                .iter()
                .zip(&y)
                .map(|(c, yi)| c * f64::from(*yi))
                .sum();
            if m <= *t {
                return false;
            }
        }
        // odometer
        let mut k = 0;
        loop {
            if k == n {
                return true;
            }
            idx[k] += 1;
            if idx[k] < steps {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

/// 2 -> 2 -> 1 ReLU net used by several tests.
fn small_net() -> Vec<StarLayer> {
    vec![
        StarLayer::Gemm {
            weight: Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, -1.0]).unwrap(),
            bias: Some(array![0.0, 0.0]),
        },
        StarLayer::Relu,
        StarLayer::Gemm {
            weight: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
            bias: Some(array![0.0]),
        },
    ]
}

#[test]
fn verifies_a_property_that_actually_holds() {
    // y = relu(x0+x1) + relu(x0-x1) >= |x0| >= 0, and around center (3,0) it is >= 2.
    let layers = small_net();
    let spec = StarSpec {
        rows: vec![(vec![1.0], 2.0)],
    };
    let (verdict, stats) =
        star_verify(&layers, &input_star(&[3.0, 0.0], 0.5), &spec, &budget()).expect("search");
    assert_eq!(verdict, StarVerdict::CandidateVerified, "stats: {stats:?}");
    assert!(
        brute_force_holds(&layers, &[3.0, 0.0], 0.5, &spec, 15),
        "brute force must agree the property holds"
    );
}

#[test]
fn finds_that_a_violated_property_has_a_counterexample() {
    // Same net, but demand y > 100, which is false everywhere near the origin.
    let layers = small_net();
    let spec = StarSpec {
        rows: vec![(vec![1.0], 100.0)],
    };
    let (verdict, _stats) =
        star_verify(&layers, &input_star(&[0.0, 0.0], 1.0), &spec, &budget()).expect("search");
    assert_eq!(verdict, StarVerdict::CandidateCounterexampleExists);
    assert!(
        !brute_force_holds(&layers, &[0.0, 0.0], 1.0, &spec, 15),
        "brute force must agree the property fails"
    );
}

#[test]
fn agrees_with_brute_force_across_a_threshold_sweep() {
    // Sweep the threshold across the sampled minimum. Candidate evidence from the internal
    // f32 star model should agree with this dense-grid consistency check.
    let layers = small_net();
    let (center, eps) = ([1.0f32, 0.25f32], 0.5f32);
    for t in [-5.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0] {
        let spec = StarSpec {
            rows: vec![(vec![1.0], t)],
        };
        let (verdict, stats) =
            star_verify(&layers, &input_star(&center, eps), &spec, &budget()).expect("search");
        let truth = brute_force_holds(&layers, &center, eps, &spec, 25);
        match verdict {
            StarVerdict::CandidateVerified => assert!(
                truth,
                "candidate-safe at t={t} but brute force found a violation; stats {stats:?}"
            ),
            StarVerdict::CandidateCounterexampleExists => assert!(
                !truth,
                "candidate-violated at t={t} but brute force found none; stats {stats:?}"
            ),
            StarVerdict::Unknown(ref why) => panic!("unexpected Unknown at t={t}: {why}"),
        }
    }
}

/// #star-bisect REGRESSION: input bisection DELEGATES a node to its two halves, so the
/// parent must not be advanced as well.
///
/// The bisection arm used to fall through into the "layer resolved without branching"
/// arm below it, which re-queued the PARENT at `layer + 1` with the ReLU never applied
/// to the unstable coordinate — nor to any coordinate after it, because the scan was
/// short-circuited. `relu(a) >= a` pointwise, so that parent evaluates a strictly
/// SMALLER function than the network on part of the box: not a relaxation of the true
/// image but a different function, and one that can manufacture a counterexample for a
/// property that actually holds. It was also re-queued at the ORIGINAL depth, so the
/// lineage could bisect forever without ever charging `max_depth`.
///
/// The invariant pinned here is the one that survives any future search change:
/// `prefer_input_split` picks a SEARCH STRATEGY, so it must never move the verdict.
#[test]
fn input_bisection_never_changes_the_verdict() {
    // y = relu(x0 + 2·x1) + relu(x0 − x1) over x0 ∈ [1,3], x1 ∈ [−1,1]. This net is
    // chosen so the bisection arm is actually REACHED on a property that HOLDS:
    //   * neuron b = x0 − x1 ∈ [0,4] is stable-active; a = x0 + 2·x1 ∈ [−1,5] straddles,
    //     so the layer scan reaches the unstable arm;
    //   * relu(a) ≥ 0 and b ≥ 0, so the interval relaxation only yields y ≥ 0 and cannot
    //     discharge y > 1.25 — the search is forced to branch rather than exit early;
    //   * the truth is y ≥ 1.5: for a ≤ 0, y = b = x0 − x1 ≥ 1.5·x0; for a > 0,
    //     y = a + b = 2·x0 + x1 > 1.5·x0. Both regimes bottom out at 1.5.
    //   * dropping the ReLUs — exactly what the fall-through did to the parent — leaves
    //     a + b = 2·x0 + x1, whose minimum is 1.0 < 1.25. So the un-ReLU'd parent
    //     violates a row the real network satisfies everywhere on the box.
    let layers = vec![
        StarLayer::Gemm {
            weight: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 1.0, -1.0]).unwrap(),
            bias: Some(array![0.0, 0.0]),
        },
        StarLayer::Relu,
        StarLayer::Gemm {
            weight: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
            bias: Some(array![0.0]),
        },
    ];
    let (center, eps) = ([2.0f32, 0.0f32], 1.0f32);
    let mut saw_bisection = false;
    for t in [-1.0, 0.5, 1.25, 1.45, 1.55, 2.5] {
        let spec = StarSpec {
            rows: vec![(vec![1.0], t)],
        };
        let mut with_split = budget();
        with_split.prefer_input_split = true;
        let (split_verdict, stats) =
            star_verify(&layers, &input_star(&center, eps), &spec, &with_split).expect("search");
        let (neuron_verdict, _) =
            star_verify(&layers, &input_star(&center, eps), &spec, &budget()).expect("search");
        saw_bisection |= stats.input_bisections > 0;

        assert_eq!(
            split_verdict, neuron_verdict,
            "prefer_input_split moved the verdict at t={t}; stats {stats:?}"
        );
        let truth = brute_force_holds(&layers, &center, eps, &spec, 25);
        match split_verdict {
            StarVerdict::CandidateVerified => assert!(
                truth,
                "bisecting search called t={t} safe but brute force found a violation; \
                 stats {stats:?}"
            ),
            StarVerdict::CandidateCounterexampleExists => assert!(
                !truth,
                "bisecting search called t={t} violated but brute force found none; \
                 stats {stats:?}"
            ),
            StarVerdict::Unknown(ref why) => panic!("unexpected Unknown at t={t}: {why}"),
        }
    }
    assert!(
        saw_bisection,
        "this sweep never reached the bisection arm, so it cannot regress the fall-through"
    );

    // The verdict alone does not pin the fall-through, because leaf refinement chases the
    // bogus lineage down and discards it rather than concluding from it. What it cannot
    // hide is the COST: re-exploring the delegated parent took 46 pops here where
    // delegation takes 9. A star budget between the two is the observable.
    let spec = StarSpec {
        rows: vec![(vec![1.0], 1.25)],
    };
    let mut tight = StarBudget::new(24, 64, Instant::now() + Duration::from_mins(1));
    tight.prefer_input_split = true;
    let (verdict, stats) =
        star_verify(&layers, &input_star(&center, eps), &spec, &tight).expect("search");
    assert_eq!(
        verdict,
        StarVerdict::CandidateVerified,
        "delegating the node must keep this inside a 24-star budget; re-queueing the \
         parent as well needed 46. stats {stats:?}"
    );
}

#[test]
fn a_stable_network_needs_no_splits() {
    // Far from the kink both ReLUs are firmly active, so the search must discharge the
    // property without branching at all.
    let layers = small_net();
    let spec = StarSpec {
        rows: vec![(vec![1.0], 0.0)],
    };
    let (verdict, stats) =
        star_verify(&layers, &input_star(&[10.0, 0.0], 0.1), &spec, &budget()).expect("search");
    assert_eq!(verdict, StarVerdict::CandidateVerified);
    assert_eq!(stats.splits, 0, "no unstable neuron here; stats {stats:?}");
}

#[test]
fn budget_exhaustion_yields_unknown_never_a_claim() {
    let layers = small_net();
    let spec = StarSpec {
        rows: vec![(vec![1.0], 2.0)],
    };
    // One star only: the search cannot possibly finish.
    let tight = StarBudget::new(1, 64, Instant::now() + Duration::from_secs(30));
    let (verdict, _) =
        star_verify(&layers, &input_star(&[0.0, 0.0], 5.0), &spec, &tight).expect("search");
    assert!(
        matches!(verdict, StarVerdict::Unknown(_)),
        "an exhausted budget must never produce a positive claim, got {verdict:?}"
    );

    let shallow = StarBudget::new(20_000, 0, Instant::now() + Duration::from_secs(30));
    let (verdict, _) =
        star_verify(&layers, &input_star(&[0.0, 0.0], 5.0), &spec, &shallow).expect("search");
    assert!(
        matches!(verdict, StarVerdict::Unknown(_)),
        "got {verdict:?}"
    );
}

#[test]
fn a_conjunctive_spec_requires_every_row() {
    let layers = small_net();
    // Row 1 holds comfortably; row 2 cannot. The conjunction must fail.
    let spec = StarSpec {
        rows: vec![(vec![1.0], -10.0), (vec![1.0], 1000.0)],
    };
    let (verdict, _) =
        star_verify(&layers, &input_star(&[1.0, 0.0], 0.5), &spec, &budget()).expect("search");
    assert_eq!(verdict, StarVerdict::CandidateCounterexampleExists);
}

#[test]
fn empty_spec_is_rejected() {
    let layers = small_net();
    let spec = StarSpec { rows: vec![] };
    assert!(star_verify(&layers, &input_star(&[0.0, 0.0], 1.0), &spec, &budget()).is_err());
}

#[test]
fn a_wrong_width_spec_row_fails_closed() {
    let layers = small_net();
    let spec = StarSpec {
        rows: vec![(vec![1.0, 1.0, 1.0], 0.0)],
    };
    assert!(star_verify(&layers, &input_star(&[0.0, 0.0], 1.0), &spec, &budget()).is_err());
}

#[test]
fn unsafe_conjunction_is_checked_jointly_not_atom_by_atom() {
    // Every atom is individually feasible on x in [-1,1], but they cannot hold at the
    // SAME point: x <= -0.5 AND x >= 0.5. This is the ACAS prop_2 semantic shape.
    let unsafe_region = StarUnsafeConjunction {
        rows: vec![
            StarUnsafeRow {
                coefficients: vec![1.0],
                threshold: -0.5,
                strict: false,
            },
            StarUnsafeRow {
                coefficients: vec![-1.0],
                threshold: -0.5,
                strict: false,
            },
        ],
    };
    let (verdict, stats) =
        star_search_unsafe_conjunction(&[], &input_star(&[0.0], 1.0), &unsafe_region, &budget())
            .expect("candidate search");
    assert_eq!(
        verdict,
        StarUnsafeCandidateVerdict::CandidateUnsafeClosureEmpty,
        "joint LP must see the contradiction; stats: {stats:?}"
    );
}

#[test]
fn unsafe_conjunction_not_proved_empty_is_only_candidate_evidence() {
    // x <= 0 AND x >= 0 has the feasible point x=0. The candidate API reports only
    // that the completed query did not prove the closure empty; it does not claim LP
    // feasibility or manufacture a validated counterexample.
    let unsafe_region = StarUnsafeConjunction {
        rows: vec![
            StarUnsafeRow {
                coefficients: vec![1.0],
                threshold: 0.0,
                strict: false,
            },
            StarUnsafeRow {
                coefficients: vec![-1.0],
                threshold: 0.0,
                strict: false,
            },
        ],
    };
    let (verdict, _) =
        star_search_unsafe_conjunction(&[], &input_star(&[0.0], 1.0), &unsafe_region, &budget())
            .expect("candidate search");
    assert_eq!(
        verdict,
        StarUnsafeCandidateVerdict::CandidateUnsafeClosureNotProvedEmpty
    );
}

#[test]
fn strict_boundary_feasibility_never_becomes_a_sat_claim() {
    // x < 0 AND x >= 0 has EMPTY strict semantics, but its closure contains x=0.
    // The candidate search must retain that ambiguity explicitly.
    let unsafe_region = StarUnsafeConjunction {
        rows: vec![
            StarUnsafeRow {
                coefficients: vec![1.0],
                threshold: 0.0,
                strict: true,
            },
            StarUnsafeRow {
                coefficients: vec![-1.0],
                threshold: 0.0,
                strict: false,
            },
        ],
    };
    let (verdict, _) =
        star_search_unsafe_conjunction(&[], &input_star(&[0.0], 1.0), &unsafe_region, &budget())
            .expect("candidate search");
    assert_eq!(
        verdict,
        StarUnsafeCandidateVerdict::CandidateUnsafeClosureNotProvedEmpty
    );
}

#[test]
fn unresolved_unsafe_leaf_report_is_undecided_not_feasible() {
    let future = Instant::now() + Duration::from_mins(1);
    let partial = StarLpReport {
        lp_bounds: vec![(f64::NEG_INFINITY, f64::INFINITY)],
        infeasible: false,
    };
    assert!(matches!(
        unsafe_closure_report_outcome(&partial, future),
        LeafOutcome::Undecided
    ));

    let completed = StarLpReport {
        lp_bounds: vec![(-0.0, 0.0)],
        infeasible: false,
    };
    assert!(matches!(
        unsafe_closure_report_outcome(&completed, future),
        LeafOutcome::Violated
    ));

    // Even a finite partial result cannot become positive candidate evidence after
    // the caller's wall authority has expired.
    assert!(matches!(
        unsafe_closure_report_outcome(&completed, Instant::now()),
        LeafOutcome::Undecided
    ));

    // A completed infeasibility proof is mathematically useful, but an expired bounded
    // search may not publish it as a completed candidate result.
    let proved_empty = StarLpReport {
        lp_bounds: vec![(f64::NEG_INFINITY, f64::INFINITY)],
        infeasible: true,
    };
    assert!(matches!(
        unsafe_closure_report_outcome(&proved_empty, Instant::now()),
        LeafOutcome::Undecided
    ));
}

#[test]
fn late_exact_infeasible_report_is_undecided() {
    let start = Instant::now();
    let deadline = start + Duration::from_secs(1);
    let proved_empty = StarLpReport {
        lp_bounds: vec![(f64::NEG_INFINITY, f64::INFINITY)],
        infeasible: true,
    };

    assert!(matches!(
        unsafe_closure_report_outcome_at(&proved_empty, deadline, start),
        LeafOutcome::Safe
    ));
    assert!(matches!(
        unsafe_closure_report_outcome_at(
            &proved_empty,
            deadline,
            deadline + Duration::from_nanos(1)
        ),
        LeafOutcome::Undecided
    ));
}

#[test]
fn late_last_item_fallthrough_is_unknown() {
    let start = Instant::now();
    let deadline = start + Duration::from_secs(1);
    let concrete = Star::from_zonotope(ZonotopeTensor::concrete(array![1.0].into_dyn()));
    let unsafe_region = StarUnsafeConjunction {
        rows: vec![StarUnsafeRow {
            coefficients: vec![1.0],
            threshold: 0.0,
            strict: false,
        }],
    };
    let search_budget = StarBudget::new(20_000, 64, deadline);

    let (verdict, stats) = star_search_with_clock(
        &[],
        &concrete,
        SearchProperty::UnsafeConjunction(&unsafe_region),
        &search_budget,
        |checkpoint| {
            if checkpoint == "star search-completion publication" {
                deadline
            } else {
                start
            }
        },
    )
    .expect("candidate search");

    assert_eq!(
        verdict,
        StarVerdict::Unknown("deadline exceeded".into()),
        "last-item fallthrough must not publish a late candidate; stats: {stats:?}"
    );
}

#[test]
fn unsafe_tail_discharge_needs_only_one_impossible_atom() {
    let layers = vec![StarLayer::Relu];
    let unsafe_region = StarUnsafeConjunction {
        rows: vec![
            StarUnsafeRow {
                // y <= 0 is impossible for y in [1.9,2.1].
                coefficients: vec![1.0],
                threshold: 0.0,
                strict: false,
            },
            StarUnsafeRow {
                // Deliberately feasible atom: the conjunction is still impossible.
                coefficients: vec![-1.0],
                threshold: 100.0,
                strict: false,
            },
        ],
    };
    let (verdict, stats) = star_search_unsafe_conjunction(
        &layers,
        &input_star(&[2.0], 0.1),
        &unsafe_region,
        &budget(),
    )
    .expect("candidate search");
    assert_eq!(
        verdict,
        StarUnsafeCandidateVerdict::CandidateUnsafeClosureEmpty
    );
    assert_eq!(stats.discharged_by_overapprox, 1, "stats: {stats:?}");
    assert_eq!(stats.splits, 0, "stats: {stats:?}");
}

#[test]
fn exact_lp_empty_node_is_dropped_instead_of_advanced() {
    // Deliberately hide the contradiction from the zero-iteration dual so the exact
    // per-neuron session is the component that discovers it.
    let base = input_star(&[0.0], 1.0);
    let star = base
        .with_constraint(&array![1.0], -0.9)
        .unwrap()
        .with_constraint(&array![-1.0], -0.9)
        .unwrap();
    let layers = vec![StarLayer::Relu];
    let spec = StarSpec {
        rows: vec![(vec![1.0], 100.0)],
    };
    let mut search_budget = budget();
    search_budget.dual_iters = 0;
    let (verdict, stats) = star_verify(&layers, &star, &spec, &search_budget).expect("search");
    assert_eq!(
        verdict,
        StarVerdict::CandidateVerified,
        "the input star is empty"
    );
    assert_eq!(
        stats.popped, 1,
        "an exact-LP-empty node must not be pushed to the next layer; stats: {stats:?}"
    );
    assert_eq!(stats.pruned_infeasible, 1, "stats: {stats:?}");
    assert_eq!(stats.leaves_verified, 0, "stats: {stats:?}");
}

#[test]
fn zonotope_input_without_a_predicate_is_accepted() {
    // A plain zonotope (k = 0) is a valid star; the driver must not require constraints.
    let z = ZonotopeTensor::from_input_elementwise(
        &ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0f32, 0.5]).unwrap(),
        0.25,
    );
    let star = Star::from_zonotope(z);
    assert_eq!(star.num_constraints(), 0);
    let layers = small_net();
    let spec = StarSpec {
        rows: vec![(vec![1.0], -100.0)],
    };
    let (verdict, _) = star_verify(&layers, &star, &spec, &budget()).expect("search");
    assert_eq!(verdict, StarVerdict::CandidateVerified);
}

#[test]
fn concrete_zero_alpha_input_does_not_require_an_lp_model() {
    let concrete = Star::from_zonotope(ZonotopeTensor::concrete(array![1.0, 0.0].into_dyn()));
    assert_eq!(concrete.alpha_dim(), 0);
    let layers = small_net();
    let holds = StarSpec {
        rows: vec![(vec![1.0], 0.5)],
    };
    let fails = StarSpec {
        rows: vec![(vec![1.0], 100.0)],
    };
    assert_eq!(
        star_verify(&layers, &concrete, &holds, &budget())
            .expect("concrete safe search")
            .0,
        StarVerdict::CandidateVerified
    );
    assert_eq!(
        star_verify(&layers, &concrete, &fails, &budget())
            .expect("concrete violated search")
            .0,
        StarVerdict::CandidateCounterexampleExists
    );
}

/// Deterministic pseudo-random weight in [-1, 1).
fn prand(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 30) as f32) - 1.0
}

fn dense(rows: usize, cols: usize, seed: &mut u64, scale: f32) -> Array2<f32> {
    let v: Vec<f32> = (0..rows * cols).map(|_| prand(seed) * scale).collect();
    Array2::from_shape_vec((rows, cols), v).expect("shape")
}

fn full_measurement_mode() -> bool {
    std::env::var("NY_FULL_MEASUREMENTS").as_deref() == Ok("1")
}

/// Bounded scaling correctness smoke. `NY_FULL_MEASUREMENTS=1` restores the
/// full width sweep while retaining the same hard assertions.
#[test]
fn scaling_smoke_verifies_each_requested_width() {
    let widths: &[usize] = if full_measurement_mode() {
        &[8, 16, 32, 50]
    } else {
        &[8]
    };
    for &width in widths {
        let mut seed = 0x5eed_1234u64;
        let layers = vec![
            StarLayer::Gemm {
                weight: dense(width, 5, &mut seed, 1.0),
                bias: Some(Array1::zeros(width)),
            },
            StarLayer::Relu,
            StarLayer::Gemm {
                weight: dense(width, width, &mut seed, 1.0 / (width as f32).sqrt()),
                bias: Some(Array1::zeros(width)),
            },
            StarLayer::Relu,
            StarLayer::Gemm {
                weight: dense(1, width, &mut seed, 1.0 / (width as f32).sqrt()),
                bias: Some(array![0.0]),
            },
        ];
        let spec = StarSpec {
            rows: vec![(vec![1.0], -1e6)],
        };
        let b = StarBudget::new(200_000, 512, Instant::now() + Duration::from_mins(2));
        let t0 = Instant::now();
        let out = star_verify(
            &layers,
            &input_star(&[0.0, 0.0, 0.0, 0.0, 0.0], 0.1),
            &spec,
            &b,
        );
        let dt = t0.elapsed();
        match out {
            Ok((v, s)) => {
                assert_eq!(
                    v,
                    StarVerdict::CandidateVerified,
                    "the deliberately easy scaling fixture must verify at width {width}"
                );
                assert!(s.popped > 0, "the search must process a root node");
                println!(
                "width {width:3} ({} relus): {:?} in {:?} | popped {} splits {} pruned {} lp-stable {}",
                2 * width, v, dt, s.popped, s.splits, s.pruned_infeasible, s.lp_reclaimed_stable
                );
            }
            Err(e) => panic!("width {width:3}: scaling search failed: {e}"),
        }
    }
}

/// HARD SCALING PROBE — thresholds placed near the true minimum, so the over-approximation
/// CANNOT discharge the node and the search must actually refine. This is the measurement
/// that says whether the star path is usable. The default lane runs a bounded
/// width-8 smoke; `NY_FULL_MEASUREMENTS=1` restores the full sweep.
#[test]
fn hard_scaling_probe_forces_real_refinement() {
    let full = full_measurement_mode();
    let widths: &[usize] = if full { &[8, 16, 32, 50] } else { &[8] };
    for &width in widths {
        let mut seed = 0x5eed_1234u64;
        let layers = vec![
            StarLayer::Gemm {
                weight: dense(width, 5, &mut seed, 1.0),
                // Generic biases. Zero biases put every ReLU kink at EXACTLY 0, where any
                // outward-rounded bound reads negative and no sound stability test can fire —
                // an artifact of the benchmark, not of real networks.
                bias: Some(Array1::from_shape_fn(width, |_| prand(&mut seed) * 0.3)),
            },
            StarLayer::Relu,
            StarLayer::Gemm {
                weight: dense(width, width, &mut seed, 1.0 / (width as f32).sqrt()),
                bias: Some(Array1::from_shape_fn(width, |_| prand(&mut seed) * 0.3)),
            },
            StarLayer::Relu,
            StarLayer::Gemm {
                weight: dense(1, width, &mut seed, 1.0 / (width as f32).sqrt()),
                bias: Some(array![0.05]),
            },
        ];
        let center = [0.0f32; 5];
        let eps = 0.5f32;

        // Empirical minimum over a coarse grid, then verify a threshold just BELOW it:
        // true, but only just — the relaxation cannot prove it without refinement.
        let mut lo = f32::INFINITY;
        let sample_count: u32 = if full { 5_000 } else { 512 };
        for i in 0..sample_count {
            let mut s2 = 0x1234_5678u64 ^ u64::from(i);
            let x: Vec<f32> = (0..5).map(|k| center[k] + prand(&mut s2) * eps).collect();
            lo = lo.min(eval(&layers, &x)[0]);
        }
        let t = f64::from(lo) - 0.05;

        let spec = StarSpec {
            rows: vec![(vec![1.0], t)],
        };
        let mut b = if full {
            StarBudget::new(200_000, 512, Instant::now() + Duration::from_mins(1))
        } else {
            StarBudget::new(10_000, 64, Instant::now() + Duration::from_secs(3))
        };
        b.dual_iters = std::env::var("PROBE_DUAL_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        let t0 = Instant::now();
        let out = star_verify(&layers, &input_star(&center, eps), &spec, &b);
        let dt = t0.elapsed();
        match out {
            Ok((v, st)) => {
                assert!(
                    st.popped > 1 && st.splits > 0,
                    "the hard fixture must enter real branch refinement at width {width}: popped={} splits={}",
                    st.popped,
                    st.splits
                );
                println!(
                    "width {width:3} ({} relus): {:?} elapsed={dt:?} popped {} splits {} lp-stable {} | LP {}ms/{} calls ({} declined, {} float-verified)  dual {}ms  tail {}ms  star {}ms",
                    2 * width, v, st.popped, st.splits, st.lp_reclaimed_stable,
                    st.ns_exact_lp / 1_000_000, st.exact_lp_calls,
                    st.ns_declined, st.verified_float_hits,
                    st.ns_dual / 1_000_000, st.ns_tail / 1_000_000, st.ns_star / 1_000_000
                );
            }
            Err(e) => panic!("width {width:3}: hard scaling search failed: {e}"),
        }
    }
}

/// Exercise real split predicates and pin that the float lane either returns a
/// sound enclosure or explicitly declines before the rigorous query runs.
#[test]
fn capture_float_decline() {
    use crate::star_lp::{StarLpRequest, StarLpSession};
    use ny_tensor::zonotope::StarReluSplit;

    let mut seed = 0x5eed_1234u64;
    let width = 16usize;
    let layers = vec![
        StarLayer::Gemm {
            weight: dense(width, 5, &mut seed, 1.0),
            bias: Some(Array1::zeros(width)),
        },
        StarLayer::Relu,
    ];
    // Walk one branch, splitting greedily, and probe each predicate as it grows.
    let mut star = input_star(&[0.0f32; 5], 0.5);
    for l in &layers {
        if let StarLayer::Gemm { weight, bias } = l {
            star = star.gemm(weight, bias.as_ref()).expect("gemm");
        }
    }
    let mut dumped = 0;
    let mut declined = 0;
    for idx in 0..width {
        if dumped >= 3 {
            break;
        }
        if let StarReluSplit::Split { active, .. } = star.relu_split(idx).expect("split") {
            star = *active;
            let (a, b) = star.constraints();
            let a_rows: Vec<Vec<f64>> = (0..a.nrows())
                .map(|r| (0..a.ncols()).map(|c| f64::from(a[[r, c]])).collect())
                .collect();
            let b_v: Vec<f64> = b.iter().map(|v| f64::from(*v)).collect();
            let (c_i, g_i) = star.coordinate_form(idx.min(width - 1)).expect("coord");
            let g: Vec<f64> = g_i.iter().map(|v| f64::from(*v)).collect();
            let req = StarLpRequest {
                alpha_dim: star.alpha_dim(),
                a_rows: a_rows.clone(),
                b: b_v.clone(),
                targets: vec![],
            };
            let mut sess = StarLpSession::new_alpha_only(
                &req,
                Duration::from_secs(5),
                Instant::now() + Duration::from_secs(30),
            )
            .expect("sess");
            let got = sess
                .verified_float_bounds(f64::from(c_i), &g, &a_rows, &b_v)
                .expect("q");
            declined += usize::from(got.is_none());
            let tight = sess.expr_bounds(f64::from(c_i), &g).expect("tight");
            println!(
                "k={} float={got:?} tight=({:.6}, {:.6})",
                a_rows.len(),
                tight.0,
                tight.1
            );
            dumped += 1;
        }
    }
    assert!(
        dumped > 0,
        "fixture must exercise at least one split predicate"
    );
    println!("observed {declined} verified-float declines across {dumped} predicates");
}

/// CAPABILITY probe: how close to the true minimum can the driver actually PROVE, inside a
/// fixed budget? Throughput is not the goal — provable margin is. Reports the gap between
/// the empirical minimum and the tightest threshold verified. The default is a
/// bounded smoke; `NY_FULL_MEASUREMENTS=1` restores the long capability sweep.
#[test]
fn capability_probe() {
    let full = full_measurement_mode();
    let widths: &[usize] = if full { &[8, 16, 32, 50] } else { &[8] };
    for &width in widths {
        let mut seed = 0x5eed_1234u64;
        let layers = vec![
            StarLayer::Gemm {
                weight: dense(width, 5, &mut seed, 1.0),
                bias: Some(Array1::from_shape_fn(width, |_| prand(&mut seed) * 0.3)),
            },
            StarLayer::Relu,
            StarLayer::Gemm {
                weight: dense(width, width, &mut seed, 1.0 / (width as f32).sqrt()),
                bias: Some(Array1::from_shape_fn(width, |_| prand(&mut seed) * 0.3)),
            },
            StarLayer::Relu,
            StarLayer::Gemm {
                weight: dense(1, width, &mut seed, 1.0 / (width as f32).sqrt()),
                bias: Some(array![0.05]),
            },
        ];
        let (center, eps) = ([0.0f32; 5], 0.5f32);

        // Empirical minimum by sampling.
        let mut lo = f32::INFINITY;
        let sample_count: u32 = if full { 20_000 } else { 512 };
        for i in 0..sample_count {
            let mut s2 = 0x1234_5678u64 ^ u64::from(i);
            let x: Vec<f32> = (0..5).map(|k| center[k] + prand(&mut s2) * eps).collect();
            lo = lo.min(eval(&layers, &x)[0]);
        }
        let true_min = f64::from(lo);

        // Tightest threshold provable in 10s, at a given split depth. depth = 0 disables
        // splitting entirely, so it measures the plain OVER-APPROXIMATION — the baseline the
        // exact search has to beat to be worth anything.
        let tightest_mode = |max_depth: usize, input_split: bool| -> f64 {
            let (mut easy, mut hard) = (true_min - 20.0, true_min);
            let mut best = f64::NEG_INFINITY;
            let rounds = if full { 9 } else { 2 };
            for _ in 0..rounds {
                let t = 0.5 * (easy + hard);
                let spec = StarSpec {
                    rows: vec![(vec![1.0], t)],
                };
                let mut b = StarBudget::new(
                    if full { 2_000_000 } else { 10_000 },
                    if full { max_depth } else { max_depth.min(64) },
                    Instant::now() + Duration::from_secs(if full { 10 } else { 1 }),
                );
                b.dual_iters = 0;
                b.prefer_input_split = input_split;
                match star_verify(&layers, &input_star(&center, eps), &spec, &b) {
                    Ok((StarVerdict::CandidateVerified, _)) => {
                        best = best.max(t);
                        easy = t;
                    }
                    _ => hard = t,
                }
            }
            best
        };
        let relax = tightest_mode(0, false);
        let neuron = tightest_mode(4096, false);
        let inputs = tightest_mode(4096, true);
        assert!(
            [relax, neuron, inputs]
                .into_iter()
                .all(|bound| bound.is_finite() && bound <= true_min),
            "every capability arm must publish a finite certified threshold no greater than the sampled minimum: true_min={true_min}, relax={relax}, neuron={neuron}, inputs={inputs}"
        );
        println!(
            "width {width:3} ({} relus): true_min {true_min:+.4} | relax {relax:+.4} (gap {:.4}) | NEURON-split {neuron:+.4} (gap {:.4}, closes {:.4}) | INPUT-split {inputs:+.4} (gap {:.4}, closes {:.4})",
            2 * width,
            true_min - relax,
            true_min - neuron, neuron - relax,
            true_min - inputs, inputs - relax
        );
    }
}
