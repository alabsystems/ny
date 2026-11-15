// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Controls of the actual public entry and existing native worker route.
//! No supplied Plan, multipliers, activity, receipt, table or native admission.
use super::*;
use crate::single_hidden_hull::{bab::HullBabOptions, HullMarginProblem};
use ny_core::Bound;
use std::time::{Duration, Instant};

struct Fixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    fn new() -> Self {
        let e = 1. / 1024.;
        Self {
            weights: vec![
                vec![
                    1.,
                    0.,
                    0.,
                    0.,
                    0.,
                    1.,
                    0.,
                    0.,
                    1.,
                    0.,
                    e,
                    0.,
                    1.,
                    1.,
                    2. * e,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                ],
                vec![2., 1., -1., -1., -e, 0., 0., 0., 0., 0.],
            ],
            // d=0 supplies genuine original unsafe points. This control makes
            // no safe-verdict or strict numerical-gain assertion.
            biases: vec![vec![0., 0., 0., 0., -1.], vec![0., 0.]],
            bounds: vec![vec![
                Bound::new(-1., 1.),
                Bound::new(-1., 1.),
                Bound::new(-(1. + e) as f32, (1. + e) as f32),
                Bound::new(-(2. + 2. * e) as f32, (2. + 2. * e) as f32),
                Bound::new(-1., -1.),
            ]],
        }
    }
    fn source(&self) -> HullMarginProblem {
        HullMarginProblem::new(
            self.weights[0].clone(),
            self.biases[0].clone(),
            4,
            &self.weights[1],
            &self.biases[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; 4],
            vec![1.; 4],
        )
        .unwrap()
    }
    fn original(&self) -> MipParts {
        let mut encoder = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &[4, 5, 2],
            &[Bound::new(-1., 1.); 4],
            &self.bounds,
        )
        .unwrap();
        encoder.constrain_output_leq_row(0, 1).unwrap();
        encoder.into_parts()
    }
    fn encoding(&self) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.weights,
            biases: &self.biases,
            dimensions: &[4, 5, 2],
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(self.original().problem.num_rows() - 1),
        }
    }
}
fn automatic(p: &HullMarginProblem, end: Instant) -> HullRootMipCuts<'_> {
    let attempt =
        propose_root_mip_cuts_with_mass_portfolio_phase_cube_native_attempt(p, end, Duration::ZERO);
    assert!(attempt.primal.same_call && attempt.primal.decoded && attempt.primal.banked);
    let cuts = attempt
        .cuts
        .expect("actual new public entry produces a checked owner");
    assert_eq!(cuts.proposal_lp_calls, 1);
    let owner = cuts.signed.as_ref().expect("actual signed owner");
    assert!(owner.budget.phase_all_enabled && owner.budget.phase_cube_enabled);
    let counts = owner.budget.state.lock().unwrap();
    assert!(counts.native_rows.reservation.is_some());
    assert!(counts.native_rows.admission.is_some());
    drop(counts);
    cuts
}
fn publish(cuts: &HullRootMipCuts<'_>, f: &Fixture, end: Instant) -> Box<MipParts> {
    let mut parts = Box::new(f.original());
    let prefix = parts.problem.clone();
    let binary = parts.binary_vars.clone();
    let report = cuts.append_to(parts.as_mut(), &f.encoding(), end).unwrap();
    assert_eq!(report.proposal_lp_calls, 1);
    let stats = report.signed.as_ref().unwrap();
    assert_eq!(stats.source_replays, 1);
    assert!(stats.component_rows.phase.cube.appended_rows >= 1);
    assert!(
        stats.component_rows.phase.appended_rows > stats.component_rows.phase.cube.appended_rows,
        "actual automatic unary publication"
    );
    assert!(stats.work_units <= MAX_WORK && stats.admitted_bytes <= MAX_BYTES);
    assert_eq!(parts.problem.cols(), prefix.cols());
    assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
    assert_eq!(parts.binary_vars, binary);
    assert_eq!(parts.problem.num_rows() - prefix.num_rows(), report.cuts);
    let counts = cuts.signed.as_ref().unwrap().budget.state.lock().unwrap();
    let pair = counts
        .native_rows
        .published
        .as_ref()
        .expect("actual committed two-row license");
    assert!(pair.planes().is_some());
    parts
}

fn collect_production_cache(cuts: &HullRootMipCuts<'_>, end: Instant) -> f64 {
    assert!(cuts.signed_child_cache.get().is_none());
    // The CLI makes this second whole-Source replay after append. The unsafe
    // fixture has no positive root proof, but must retain its complete cache.
    assert!(cuts.best_signed_root_proof(end).is_none());
    let cache = cuts
        .signed_child_cache
        .get()
        .expect("production second replay collects the signed cache");
    let stats = cache.stats();
    assert_eq!(stats.source_replays, 2);
    assert_eq!(stats.collection_attempts, 1);
    assert!(stats.cached_masks > 0 && stats.cache_bytes > 0);
    assert_eq!(
        stats.bind_attempts, 0,
        "continuation owns the only cache bind"
    );
    // Replay checks this scalar bit-for-bit before installing the cache. Do
    // not bind it here: that would spend credit absent from the CLI route.
    let before = cuts.checked_lower.max(cuts.signed.as_ref().unwrap().lower);
    assert!(before.is_finite() && before <= -3. / 1024.);
    before
}

#[test]
fn native_public_entry_publishes_automatic_pair_and_engages_existing_child_lp() {
    let _serial = tests::native_test_guard();
    let f = Fixture::new();
    let p = f.source();
    let e = 1. / 1024.;
    let margin = |x: &[f64; 4]| {
        f.biases[1][0] - f.biases[1][1]
            + (0..5)
                .map(|j| {
                    let h = f.biases[0][j]
                        + f.weights[0][4 * j..4 * (j + 1)]
                            .iter()
                            .zip(x)
                            .map(|(a, x)| a * x)
                            .sum::<f64>();
                    (f.weights[1][j] - f.weights[1][5 + j]) * h.max(0.)
                })
                .sum::<f64>()
    };
    // Exact raw original-output dyadic witnesses in both first children.
    // The downward-rounded Source margin is deliberately NOT used as equality.
    assert_eq!(margin(&[0., 0., 1., 0.]), -3. * e);
    assert_eq!(margin(&[-1.5 * e, 1., 1., 0.]), -0.5 * e);
    assert_eq!(margin(&[0.25 * e, -3. * e, 1., 0.]), -0.75 * e);
    let end = Instant::now() + Duration::from_secs(30);
    let cuts = automatic(&p, end);
    let _parts = publish(&cuts, &f, end);
    let before = collect_production_cache(&cuts, end);
    assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
    let result = cuts
        .continue_hull_bab_with_native_rows(
            end,
            HullBabOptions {
                workers: 1,
                max_nodes: 3,
            },
        )
        .expect("actual reused-root continuation");
    let tree = result.report();
    assert!(tree.root_before_native.unwrap() >= before);
    let child = result.signed_child_report().unwrap();
    assert_eq!(child.source_replays, 2);
    assert_eq!(child.bind_attempts, 1);
    assert_eq!(tree.root_lp_calls, 1);
    assert!((2..=3).contains(&tree.proposal_lp_calls));
    assert!(tree.proposal_lp_calls > tree.root_lp_calls);
    assert!((2..=3).contains(&tree.nodes));
    let native = cuts.native_rows_report();
    assert!((1..=2).contains(&native.attempted));
    assert!((1..=native.attempted).contains(&native.committed));
    assert!(
        native.committed > tree.root_lp_calls,
        "a child must use the published rows too"
    );
    assert!((1..=2 * native.committed).contains(&native.priced));
    assert_eq!(cuts.proposal_lp_calls, 1);
    // The root consumes one of the two optional augmentation credits. This
    // can add one total LP; unaugmented child fallbacks remain ordinary calls.
    // Engagement is not evidence of a winning scalar or solver verdict.
}

#[test]
fn native_public_entry_prices_original_root_before_any_child() {
    let _serial = tests::native_test_guard();
    let f = Fixture::new();
    let p = f.source();
    let end = Instant::now() + Duration::from_secs(30);
    let cuts = automatic(&p, end);
    let _parts = publish(&cuts, &f, end);
    let before = collect_production_cache(&cuts, end);
    let result = cuts
        .continue_hull_bab_with_native_rows(
            end,
            HullBabOptions {
                workers: 1,
                max_nodes: 1,
            },
        )
        .expect("actual root-only continuation");
    let tree = result.report();
    let child = result.signed_child_report().unwrap();
    assert_eq!(child.source_replays, 2);
    assert_eq!(child.bind_attempts, 1);
    assert_eq!(tree.nodes, 1, "no child may explain native engagement");
    assert_eq!(tree.root_lp_calls, 1);
    assert_eq!(tree.proposal_lp_calls, 1);
    let original_floor = tree.root_before_native.expect("actual root/cache floor");
    assert!(original_floor.is_finite() && original_floor >= before);
    let effective_after = tree
        .root_certified
        .expect("independently checked root")
        .max(original_floor);
    assert!(effective_after >= original_floor);
    eprintln!(
        "[native-root-control] before={original_floor:?} after={effective_after:?} improved={}",
        effective_after > original_floor,
    );
    // x=[0,0,1,0] is an original feasible point with exact margin -3/1024.
    // A certified lower bound must stay below it, even after adding both rows.
    assert!(effective_after <= -3. / 1024.);
    let native = cuts.native_rows_report();
    assert_eq!(native.attempted, 1);
    assert_eq!(native.committed, 1);
    assert!((1..=2).contains(&native.priced));
    assert_eq!(
        cuts.proposal_lp_calls, 1,
        "the original proposal was reused"
    );
    // This unsafe fixture checks automatic engagement and non-regression.
    // Its old K2 family can attain the exact -3/1024 minimum: mask bits a,b
    // admit alpha_x=(a+b)/2, alpha_y=b and floors [0,-e,-2e,-3e]. It therefore
    // cannot establish a substantive strict gain over the complete old cache.
}

#[test]
fn native_public_entry_strictly_improves_original_root_over_complete_cache() {
    let _serial = tests::native_test_guard();
    // Inputs x,y,z,t. Distinct stable-dead positive heads occupy the
    // component's remaining slots; head 16 remains outside that component.
    let mut w1 = vec![1., 0., 0., 0., 0., 1., 0., 0.];
    let mut b1 = vec![0., 0.];
    let mut c = vec![2., 1.];
    let mut hidden = vec![Bound::new(-1., 1.), Bound::new(-1., 1.)];
    for i in 0..14 {
        let m = 9. / 16. + i as f64 / 4096.;
        w1.extend_from_slice(&[m, 0., 0., 0.]);
        b1.push(-m - 1. / 4.);
        c.push(1.);
        hidden.push(Bound::new((-2. * m - 1. / 4.) as f32, -0.25));
    }
    w1.extend_from_slice(&[
        0.,
        0.,
        1.,
        0., // C = R(z - 1/2)
        1.,
        0.,
        1. / 4.,
        0., // N0 = R(x + z/4)
        1.,
        1.,
        1. / 2.,
        0., // N1 = R(x + y + z/2)
    ]);
    b1.extend_from_slice(&[-1. / 2., 0., 0.]);
    c.extend_from_slice(&[3. / 4., -1., -1.]);
    hidden.extend_from_slice(&[
        Bound::new(-1.5, 0.5),
        Bound::new(-1.25, 1.25),
        Bound::new(-2.5, 2.5),
    ]);
    c.resize(38, 0.);
    let f = Fixture {
        weights: vec![w1, c],
        biases: vec![b1, vec![0., 0.]],
        bounds: vec![hidden],
    };
    let p = f.source();
    assert_eq!(p.input_dim, 4);
    assert_eq!(p.hidden_dim, 19);
    // Exact raw original graph witness, independent of rounded Source c/d.
    let witness = [0., 0., 1., 0.];
    let raw_upper = (0..19)
        .map(|j| {
            let h = f.biases[0][j]
                + f.weights[0][4 * j..4 * (j + 1)]
                    .iter()
                    .zip(&witness)
                    .map(|(a, x)| a * x)
                    .sum::<f64>();
            (f.weights[1][j] - f.weights[1][19 + j]) * h.max(0.)
        })
        .sum::<f64>();
    assert_eq!(raw_upper, -3. / 8.);
    let end = Instant::now() + Duration::from_secs(30);
    let cuts = automatic(&p, end);
    let owner = cuts.signed.as_ref().unwrap();
    eprintln!(
        "[native-root-gain-discovery] alpha_c={:?} lower={:?} plan={:?}",
        cuts.multipliers.alpha[16], owner.lower, owner.plan,
    );
    assert_eq!(cuts.multipliers.alpha[16], 0.);
    assert_eq!(owner.plan.components.len(), 1);
    assert_eq!(owner.plan.leftover.as_slice(), &[16]);
    let component = &owner.plan.components[0];
    assert_eq!(component.positive.len(), 16);
    assert!((0..16).all(|j| component.positive.contains(&j)));
    assert_eq!(component.negative.len(), 2);
    assert!(component.negative.contains(&17) && component.negative.contains(&18));
    let dimensions = [4, 19, 2];
    let mut encoder = crate::encode_feedforward(
        &f.weights,
        &f.biases,
        &dimensions,
        &[Bound::new(-1., 1.); 4],
        &f.bounds,
    )
    .unwrap();
    encoder.constrain_output_leq_row(0, 1).unwrap();
    let mut parts = Box::new(encoder.into_parts());
    let prefix = parts.problem.clone();
    let binary = parts.binary_vars.clone();
    let encoding = HullMipEncoding {
        weights: &f.weights,
        biases: &f.biases,
        dimensions: &dimensions,
        hidden_bounds: &f.bounds,
        first_output: 0,
        second_output: 1,
        reverse: false,
        unsafe_row: Row(prefix.num_rows() - 1),
    };
    let report = cuts.append_to(parts.as_mut(), &encoding, end).unwrap();
    assert_eq!(report.proposal_lp_calls, 1);
    let stats = report.signed.as_ref().unwrap();
    assert_eq!(stats.source_replays, 1);
    assert_eq!(stats.component_rows.appended_rows, 1);
    assert_eq!(stats.component_rows.phase.cube.appended_rows, 1);
    assert_eq!(stats.component_rows.phase.cube.appended_selector_terms, 2);
    assert!(stats.work_units <= MAX_WORK);
    assert!(stats.admitted_bytes <= MAX_BYTES);
    assert_eq!(parts.problem.cols(), prefix.cols());
    assert_eq!(&parts.problem.rows()[..prefix.num_rows()], prefix.rows());
    assert_eq!(parts.binary_vars, binary);
    assert_eq!(parts.problem.num_rows() - prefix.num_rows(), report.cuts);
    {
        let counts = owner.budget.state.lock().unwrap();
        let pair = counts.native_rows.published.as_ref().unwrap();
        assert_eq!(pair.payload.context.component, 0);
        let planes = pair.planes().unwrap();
        for (index, plane) in planes.iter().enumerate() {
            let terms: Vec<_> = plane.terms().iter().map(|t| t.semantic()).collect();
            eprintln!(
                "[native-root-gain-plane] index={index} lower={:?} terms={terms:?}",
                plane.lower(),
            );
        }
    }
    let before = collect_production_cache(&cuts, end);
    assert!(before <= -3. / 4., "unexpected old floor: {before:?}");
    assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
    let result = cuts
        .continue_hull_bab_with_native_rows(
            end,
            HullBabOptions {
                workers: 1,
                max_nodes: 1,
            },
        )
        .expect("actual root-only native continuation");
    let tree = result.report();
    let child = result.signed_child_report().unwrap();
    assert_eq!(child.source_replays, 2);
    assert_eq!(child.bind_attempts, 1);
    assert_eq!(tree.nodes, 1, "no child may explain the gain");
    assert_eq!(tree.root_lp_calls, 1);
    assert_eq!(tree.proposal_lp_calls, 1);
    let b0 = tree.root_before_native.unwrap();
    assert!(b0.is_finite() && b0 >= before);
    assert!(b0 <= -3. / 4., "unexpected old root/cache floor: {b0:?}");
    let b1 = tree.root_certified.unwrap().max(b0);
    eprintln!(
        "[native-root-strict-gain] before={b0:?} after={b1:?} gain={:?}",
        b1 - b0,
    );
    assert!(
        b1 > b0,
        "automatic returned dual did not improve the old floor"
    );
    assert!(b1 <= raw_upper, "bound exceeds an original feasible point");
    let native = cuts.native_rows_report();
    assert_eq!(native.attempted, 1);
    assert_eq!(native.committed, 1);
    assert!((1..=2).contains(&native.priced));
    assert_eq!(cuts.proposal_lp_calls, 1);
}

#[test]
fn native_short_publication_clips_optional_clock_and_expired_bind_keeps_old_tree() {
    let _serial = tests::native_test_guard();
    let f = Fixture::new();
    let p = f.source();
    let end = Instant::now() + Duration::from_secs(30);
    let cuts = automatic(&p, end);
    let append_end = Instant::now() + Duration::from_secs(5);
    let _parts = publish(&cuts, &f, append_end);
    let owner = cuts.signed.as_ref().unwrap();
    let published_end = publication_end(owner).unwrap();
    assert_eq!(
        published_end,
        append_end.min(owner.budget.authority).min(cuts.deadline)
    );
    assert!(published_end < end);
    let bound = owner
        .bind_native_rows(&cuts, end, &mut || true)
        .expect("later caller is clipped, not rejected or renewed");
    assert_eq!(bound.end(), published_end);
    assert!(std::ptr::eq(bound.source, std::ptr::from_ref(&p)));
    assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
    let splits = [bab::HullSplit {
        neuron: 2,
        active: true,
    }];
    let (_, _, _, calls) = bab::native_rows_test_node(&p, &splits, &bound, end);
    assert_eq!(
        calls, 1,
        "one existing LP under min(requested, published) end"
    );
    let native = cuts.native_rows_report();
    assert_eq!(native.attempted, 1);
    assert_eq!(native.committed, 1);
    assert!((1..=2).contains(&native.priced));
    drop(bound);

    // Genuine expired requested bind; no token fabrication or wall-clock sleep.
    // The consumed optional once-latch must not suppress the old continuation.
    let end = Instant::now() + Duration::from_secs(30);
    let cuts = automatic(&p, end);
    let _parts = publish(&cuts, &f, end);
    let owner = cuts.signed.as_ref().unwrap();
    assert!(bind(owner, &cuts, Instant::now(), &mut || true).is_none());
    assert!(owner.budget.state.lock().unwrap().native_rows.bound);
    let result = cuts
        .continue_hull_bab_with_native_rows(
            end,
            HullBabOptions {
                workers: 1,
                max_nodes: 3,
            },
        )
        .expect("expired optional grant retains the live old continuation");
    assert_eq!(result.report().root_lp_calls, 0);
    assert!((1..=2).contains(&result.report().proposal_lp_calls));
    assert_eq!(cuts.native_rows_report(), SignedNativeRowsReport::default());
    assert_eq!(cuts.proposal_lp_calls, 1);
}
