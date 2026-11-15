// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Authored original helper graphs, not corpus or optimized-performance claims.
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
    fn cycle(k: usize, bias: f64, permuted: bool) -> Self {
        let d = 30;
        let mut first = vec![0.; d * k];
        for j in 0..k {
            let edge = if permuted { k - 1 - j } else { j };
            let a = if permuted { k - 1 - edge } else { edge };
            let b = if permuted {
                k - 1 - (edge + 1) % k
            } else {
                (edge + 1) % k
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
            w: vec![first, vec![-1.; k].into_iter().chain(vec![0.; k]).collect()],
            b: vec![vec![0.; k], vec![bias, 0.]],
            dims: vec![d, k, 2],
            input: vec![Bound::new(-1., 1.); d],
            bounds: vec![vec![Bound::new(-2., 2.); k]],
            spec,
        }
    }
    fn parts(&self) -> (MipParts, Vec<Row>) {
        let mut e =
            encode_feedforward(&self.w, &self.b, &self.dims, &self.input, &self.bounds).unwrap();
        let rows = add_vnnlib_constraints(&mut e, &self.spec).unwrap();
        (e.into_parts(), rows)
    }
    fn attempt(
        &self,
        parts: &mut MipParts,
        rows: &[Row],
        global: bool,
        allowed: bool,
        end: Instant,
    ) -> Option<StrengtheningOutcome> {
        if global {
            strengthen_with_global_sdp(
                parts,
                true,
                &self.w,
                &self.b,
                &self.dims,
                &self.bounds,
                &self.input,
                &self.spec,
                rows,
                end,
                allowed,
            )
        } else {
            strengthen_with_groups(
                parts,
                true,
                &self.w,
                &self.b,
                &self.dims,
                &self.bounds,
                &self.input,
                &self.spec,
                rows,
                end,
                allowed,
            )
        }
    }
    fn run(&self, global: bool, allowed: bool) -> (MipParts, StrengtheningOutcome) {
        let (mut parts, rows) = self.parts();
        let before = parts.problem.clone();
        let binary = parts.binary_vars.clone();
        let widths = parts.binary_widths.clone();
        let out = self
            .attempt(
                &mut parts,
                &rows,
                global,
                allowed,
                Instant::now() + Duration::from_secs(30),
            )
            .unwrap();
        assert_eq!(&parts.problem.rows()[..before.num_rows()], before.rows());
        assert_eq!(parts.problem.cols(), before.cols());
        assert_eq!(parts.binary_vars, binary);
        assert_eq!(parts.binary_widths, widths);
        (parts, out)
    }
}

#[test]
fn sdp_root_cli_real_optimizer_closes_odd9_both_permutations_from_one_lp() {
    for permuted in [false, true] {
        let (_, out) = Fixture::cycle(9, 285. / 32., permuted).run(true, true);
        assert_eq!(out.cuts.proposal_lp_calls, 1);
        assert_eq!(out.cuts.individual_cuts, 9);
        assert!(out.cuts.checked_hull_lower < 0.);
        assert!(out.cuts.compact_root_lower.is_none_or(|v| v < 0.));
        let sdp = out
            .cuts
            .global_sdp
            .as_ref()
            .expect("real optimizer must admit a global row");
        assert_eq!(sdp.dimension, 40);
        assert!(sdp.active_dimension >= 18 && sdp.active_dimension <= sdp.dimension);
        assert!(sdp.optimizer_iterations > 0 && sdp.optimizer_iterations <= 48);
        assert!(sdp.factorizations > 0 && sdp.checked_candidates > 0);
        assert!(sdp.source_replays > 0);
        assert!(sdp.work_units <= 256_000_000 && sdp.admitted_bytes <= 2 * 1024 * 1024);
        assert!(sdp.correction.is_finite() && sdp.correction >= 0.);
        assert!(sdp.lower > 0.005 && sdp.lower <= 29. / 32.);
        assert_eq!(
            out.cuts.cuts,
            out.cuts.individual_cuts + out.cuts.compact_cuts + 1
        );
        let proof = out.aggregate.unwrap();
        assert_eq!(proof.route(), "global-sdp");
        assert!(proof.verified() && proof.lower() > 0.005 && proof.lower() <= 29. / 32.);
        assert!(
            out.continuation.is_none(),
            "a positive root needs no native tree"
        );
    }
}

#[test]
fn sdp_root_cli_old_local_groups_stay_open_without_claiming_old_tree_failure() {
    let (_, out) = Fixture::cycle(9, 285. / 32., false).run(false, false);
    assert_eq!(out.cuts.proposal_lp_calls, 1);
    assert!(out.cuts.global_sdp.is_none());
    assert!(out.cuts.compact_root_lower.is_none_or(|v| v <= -3. / 32.));
    assert!(out.aggregate.is_none() && out.continuation.is_none());
    // This compares unconditional local geometry. A native phase tree or a
    // different complete verifier can already solve this authored graph.
}

#[test]
fn sdp_root_cli_unsafe_and_tie_never_mint_a_positive_native_token() {
    for (bias, ceiling) in [(31. / 4., -0.25), (8., 0.)] {
        let (_, out) = Fixture::cycle(9, bias, false).run(true, true);
        assert_eq!(out.cuts.proposal_lp_calls, 1);
        let sdp = out
            .cuts
            .global_sdp
            .as_ref()
            .expect("unsafe geometry still exercises the optimizer");
        assert!(sdp.optimizer_iterations > 0 && sdp.factorizations > 0);
        assert!(sdp.lower <= ceiling);
        assert!(out.aggregate.is_none());
        assert!(out.continuation.as_ref().is_none_or(|p| !p.verified()));
    }
}

#[test]
fn sdp_root_cli_publication_exclusions_keep_geometry_without_native_authority() {
    let (_, out) = Fixture::cycle(9, 285. / 32., false).run(true, false);
    assert!(out
        .cuts
        .global_sdp
        .as_ref()
        .is_some_and(|s| s.lower > 0.005));
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
fn sdp_root_cli_foreign_original_model_refuses_before_atomic_append() {
    for kind in 0..3 {
        let mut f = Fixture::cycle(9, 285. / 32., false);
        let (mut parts, rows) = f.parts();
        if kind == 0 {
            parts.output_vars.swap(0, 1);
        }
        if kind == 1 {
            f.w[0][0] = 1.25;
        }
        if kind == 2 {
            f.input[0] = Bound::new(-0.5, 1.);
        }
        let before = parts.problem.clone();
        let binary = parts.binary_vars.clone();
        let widths = parts.binary_widths.clone();
        assert!(f
            .attempt(
                &mut parts,
                &rows,
                true,
                true,
                Instant::now() + Duration::from_secs(30)
            )
            .is_none());
        assert_eq!(parts.problem, before);
        assert_eq!(parts.binary_vars, binary);
        assert_eq!(parts.binary_widths, widths);
    }
}

#[test]
fn sdp_root_cli_expired_original_clock_cannot_be_renewed_or_append_rows() {
    let f = Fixture::cycle(9, 285. / 32., false);
    let (mut parts, rows) = f.parts();
    let before = parts.problem.clone();
    assert!(f
        .attempt(
            &mut parts,
            &rows,
            true,
            true,
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("test clock supports an expired deadline")
        )
        .is_none());
    assert_eq!(parts.problem, before);
}

#[test]
fn sdp_root_cli_old_positive_group_skips_optional_optimizer_and_keeps_its_proof() {
    let (_, out) = Fixture::cycle(7, 6.5, false).run(true, true);
    assert_eq!(out.cuts.proposal_lp_calls, 1);
    assert!(out.cuts.global_sdp.is_none());
    let proof = out.aggregate.unwrap();
    assert_eq!(proof.route(), "compact");
    assert!(proof.verified() && proof.lower() > 0.499 && proof.lower() <= 0.5);
    assert!(out.continuation.is_none());
}

#[test]
fn sdp_root_cli_actual_ay_keeps_an_unchanged_original_unsafe_witness() {
    let (parts, out) = Fixture::cycle(9, 31. / 4., false).run(true, false);
    assert!(out.aggregate.is_none());
    assert!(out
        .cuts
        .global_sdp
        .as_ref()
        .is_some_and(|s| s.optimizer_iterations > 0));
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
        panic!("global geometry excluded the original unsafe graph: {result:?}")
    };
    assert_eq!(input_values.len(), 30);
    let x: Vec<_> = input_values
        .iter()
        .map(|v| Q::from_float(*v).unwrap())
        .collect();
    let minus_one = Q::from_integer((-1).into());
    let one = Q::from_integer(1.into());
    assert!(x.iter().all(|v| *v >= minus_one && *v <= one));
    let mut margin = Q::from_float(31. / 4.).unwrap();
    for j in 0..9 {
        margin -= (&x[j] - &x[(j + 1) % 9]).max(Q::zero());
    }
    assert!(
        margin <= Q::zero(),
        "input fails unchanged original rational replay"
    );
}

#[test]
fn sdp_root_cli_rendered_native_proof_keeps_native_hull_label() {
    struct Capture;
    impl Drop for Capture {
        fn drop(&mut self) {
            crate::commands::beta_crown::output::end_capture();
        }
    }
    let (_, out) = Fixture::cycle(9, 285. / 32., false).run(true, true);
    let proof = out.aggregate.unwrap();
    assert_eq!(proof.route(), "global-sdp");
    assert!(proof.verified());
    crate::commands::beta_crown::output::begin_capture();
    let _capture = Capture;
    let result = VerificationResult::Verified {
        provenance: Default::default(),
        output_bounds: vec![Bound::new(0.005, 1.)],
        proof: None,
        actual_method: Some(ny_core::MethodUsed::BetaCrown),
    };
    let refused = print_result_with_native_hull(
        &result,
        None,
        None,
        0.,
        0.,
        Duration::ZERO,
        MipBackend::Ay,
        true,
        None,
        true,
    )
    .unwrap();
    assert!(!refused);
    let rendered = crate::commands::beta_crown::output::take_captured_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&rendered).unwrap();
    assert_eq!(value["status"], "verified");
    assert_eq!(value["method"], "native-hull");
}
