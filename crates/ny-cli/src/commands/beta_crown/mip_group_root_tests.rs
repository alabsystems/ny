// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Actual helper/LP controls; no original benchmark or full ONNX route claim.
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;
use ny_onnx::vnnlib::OutputConstraint;

struct Fixture {
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    dims: Vec<usize>,
    input: Vec<Bound>,
    bounds: Vec<Vec<Bound>>,
    spec: VnnLibSpec,
}
impl Fixture {
    fn new(bias: f64, permuted: bool) -> Self {
        // Authored original 30-input model. The remaining 23 coordinates
        // are independent zeros in this graph, not altered runtime inputs.
        let d = 30;
        let h = 7;
        let mut first = vec![0.; d * h];
        for j in 0..h {
            let edge = if permuted { h - 1 - j } else { j };
            let a = if permuted { h - 1 - edge } else { edge };
            let b = if permuted {
                h - 1 - (edge + 1) % h
            } else {
                (edge + 1) % h
            };
            first[j * d + a] = 1.;
            first[j * d + b] = -1.;
        }
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = d;
        spec.num_outputs = 2;
        spec.input_bounds = vec![(-1., 1.); d];
        spec.output_constraints = vec![OutputConstraint::LessEq(0, 1)];
        Self {
            w: vec![first, vec![-1.; h].into_iter().chain(vec![0.; h]).collect()],
            b: vec![vec![0.; h], vec![bias, 0.]],
            dims: vec![d, h, 2],
            input: vec![Bound::new(-1., 1.); d],
            bounds: vec![vec![Bound::new(-2., 2.); h]],
            spec,
        }
    }
    fn parts(&self) -> (MipParts, Vec<Row>) {
        let mut encoder =
            encode_feedforward(&self.w, &self.b, &self.dims, &self.input, &self.bounds).unwrap();
        let rows = add_vnnlib_constraints(&mut encoder, &self.spec).unwrap();
        (encoder.into_parts(), rows)
    }
    fn run(&self, groups: bool, allowed: bool) -> (MipParts, StrengtheningOutcome) {
        let (mut parts, rows) = self.parts();
        let before = parts.problem.clone();
        let args_end = Instant::now() + Duration::from_secs(30);
        let outcome = if groups {
            strengthen_with_groups(
                &mut parts,
                true,
                &self.w,
                &self.b,
                &self.dims,
                &self.bounds,
                &self.input,
                &self.spec,
                &rows,
                args_end,
                allowed,
            )
        } else {
            strengthen_with_short_proof(
                &mut parts,
                true,
                &self.w,
                &self.b,
                &self.dims,
                &self.bounds,
                &self.input,
                &self.spec,
                &rows,
                args_end,
                allowed,
            )
        }
        .unwrap();
        assert_eq!(&parts.problem.rows()[..before.num_rows()], before.rows());
        assert_eq!(parts.problem.cols(), before.cols());
        (parts, outcome)
    }
}

#[test]
fn group_root_cli_default_odd_cycle_closes_from_one_lp_and_preserves_original_prefix() {
    for permuted in [false, true] {
        let (_, out) = Fixture::new(6.5, permuted).run(true, true);
        assert_eq!(
            out.cuts.proposal_lp_calls, 1,
            "the real attaining LP is shared"
        );
        assert!(out.cuts.checked_hull_lower < -0.499 && out.cuts.checked_hull_lower > -0.501);
        assert_eq!(out.cuts.individual_cuts, 7);
        assert_eq!(out.cuts.compact_cuts, 1);
        assert!(out
            .cuts
            .compact_root_lower
            .is_some_and(|x| x > 0.499 && x <= 0.5));
        let proof = out.aggregate.unwrap();
        assert!(proof.verified());
        assert!(proof.lower() > 0.499 && proof.lower() <= 0.5);
        assert!(
            out.continuation.is_none(),
            "a positive group floor needs no child/root LP"
        );
    }
}

#[test]
fn group_root_cli_pair_only_aggregate_is_open_without_claiming_old_tree_failure() {
    let (_, out) = Fixture::new(6.5, false).run(false, false);
    assert_eq!(out.cuts.proposal_lp_calls, 1);
    assert!(out.cuts.checked_hull_lower < 0.);
    assert!(out.cuts.compact_root_lower.is_none_or(|x| x <= -0.499));
    assert!(out.aggregate.is_none() && out.continuation.is_none());
    // A real phase tree may already close this authored family. This only
    // compares the pair aggregate to the new group aggregate.
}

#[test]
fn group_root_cli_original_unsafe_and_tie_never_mint_positive_native_proof() {
    for (bias, ceiling) in [(5.5, -0.5), (6., 0.)] {
        let (_, out) = Fixture::new(bias, false).run(true, true);
        assert_eq!(out.cuts.proposal_lp_calls, 1);
        assert!(out.aggregate.is_none());
        assert!(out.cuts.compact_root_lower.is_some_and(|x| x <= ceiling));
        assert!(out.continuation.as_ref().is_none_or(|p| !p.verified()));
    }
}

#[test]
fn group_root_cli_required_authorities_retain_geometry_without_minting_native_token() {
    let (_, out) = Fixture::new(6.5, false).run(true, false);
    assert_eq!(out.cuts.proposal_lp_calls, 1);
    assert_eq!(out.cuts.compact_cuts, 1);
    assert!(out.cuts.compact_root_lower.is_some_and(|x| x > 0.499));
    assert!(out.aggregate.is_none() && out.continuation.is_none());
    for shared in [false, true] {
        for canary in [false, true] {
            for bridge in [false, true] {
                assert_eq!(
                    short_proof_allowed(shared, canary, bridge),
                    !shared && !canary && !bridge
                );
            }
        }
    }
}

#[test]
fn group_root_cli_foreign_original_layout_refuses_before_any_group_append() {
    let f = Fixture::new(6.5, false);
    let (mut parts, rows) = f.parts();
    parts.output_vars.swap(0, 1);
    let before = parts.problem.clone();
    let binary = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    assert!(strengthen_with_groups(
        &mut parts,
        true,
        &f.w,
        &f.b,
        &f.dims,
        &f.bounds,
        &f.input,
        &f.spec,
        &rows,
        Instant::now() + Duration::from_secs(30),
        true
    )
    .is_none());
    assert_eq!(parts.problem, before);
    assert_eq!(parts.binary_vars, binary);
    assert_eq!(parts.binary_widths, widths);
}

#[test]
fn group_root_cli_open_default_fallback_keeps_actual_ay_original_unsafe_witness() {
    let (parts, out) = Fixture::new(5.5, false).run(true, true);
    assert!(out.aggregate.is_none());
    assert!(out.continuation.as_ref().is_none_or(|p| !p.verified()));
    let solver = MipSolver::new(
        parts,
        MipConfig {
            backend: MipBackend::Ay,
            parallel_split: 1,
            lp_tighten: false,
            timeout_secs: 10.,
            ay_hard_deadline: Some(Instant::now() + Duration::from_secs(30)),
            ..Default::default()
        },
    );
    let result = solver.check_feasibility().unwrap();
    let MipResult::Sat { input_values, .. } = result else {
        panic!("unsafe original excluded: {result:?}")
    };
    assert_eq!(input_values.len(), 30);
    let x: Vec<_> = input_values
        .iter()
        .map(|v| Q::from_float(*v).unwrap())
        .collect();
    let minus_one = Q::from_integer((-1).into());
    let one = Q::from_integer(1.into());
    assert!(x.iter().all(|v| *v >= minus_one && *v <= one));
    let mut original = Q::from_float(5.5).unwrap();
    for j in 0..7 {
        original -= (&x[j] - &x[(j + 1) % 7]).max(Q::zero());
    }
    assert!(
        original <= Q::zero(),
        "returned input fails unchanged original rational graph"
    );
}
