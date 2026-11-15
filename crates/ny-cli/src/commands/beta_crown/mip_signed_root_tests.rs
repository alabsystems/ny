// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Authored original helper models; no corpus or optimized-performance claim.
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
    fn twins(bias: f64, permuted: bool) -> Self {
        let d = 30;
        // These are independently authored positive/negative graph roles.
        // Two nonzero first biases and the final input are part of the model.
        let order = if permuted { [3, 0, 2, 1] } else { [0, 1, 2, 3] };
        let mut first = vec![0.; 4 * d];
        let mut first_bias = vec![0.; 4];
        let mut c = vec![0.; 4];
        for (j, role) in order.into_iter().enumerate() {
            let base = role % 2;
            let terms = if base == 0 {
                [(0, 1.), (29, 0.5)]
            } else {
                [(1, 1.), (28, -0.25)]
            };
            for (i, v) in terms {
                let col = if permuted { (7 * i + 3) % d } else { i };
                first[j * d + col] = v;
            }
            first_bias[j] = if base == 0 { 0.125 } else { -0.125 };
            c[j] = if role < 2 { 1. } else { -1. };
        }
        Self::from_arrays(first, first_bias, c, bias, d)
    }
    fn cycle(bias: f64) -> Self {
        let d = 30;
        let h = 7;
        let mut w = vec![0.; d * (h + 1)];
        for j in 0..h {
            w[j * d + j] = 1.;
            w[j * d + (j + 1) % h] = -1.;
        }
        // A genuine nonzero-gradient positive role prevents a no-positive
        // eligibility refusal from masquerading as old-positive proof reuse.
        w[h * d + 29] = 0.125;
        let mut first_bias = vec![0.; h];
        first_bias.push(2.);
        let mut c = vec![-1.; h];
        c.push(1.);
        let mut f = Self::from_arrays(w, first_bias, c, bias - 1.875, d);
        f.bounds[0][h] = Bound::new(1.875, 2.125);
        f
    }
    fn from_arrays(
        first: Vec<f64>,
        first_bias: Vec<f64>,
        c: Vec<f64>,
        bias: f64,
        d: usize,
    ) -> Self {
        let h = c.len();
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = d;
        spec.num_outputs = 2;
        spec.input_bounds = vec![(-1., 1.); d];
        spec.output_constraints = vec![OutputConstraint::LessEq(0, 1)];
        Self {
            w: vec![first, c.into_iter().chain(vec![0.; h]).collect()],
            b: vec![first_bias, vec![bias, 0.]],
            dims: vec![d, h, 2],
            input: vec![Bound::new(-1., 1.); d],
            bounds: vec![vec![Bound::new(-2., 2.); h]],
            spec,
        }
    }
    fn parts(&self) -> (MipParts, Vec<Row>) {
        let mut e =
            encode_feedforward(&self.w, &self.b, &self.dims, &self.input, &self.bounds).unwrap();
        let rows = add_vnnlib_constraints(&mut e, &self.spec).unwrap();
        let mut parts = e.into_parts();
        // Use the same checked sign relaxation as actual eligible production.
        let r = convex_binary::relax(
            &mut parts,
            true,
            &self.w,
            &self.b,
            &self.dims,
            &self.bounds,
            &self.spec,
            &rows,
            Instant::now() + Duration::from_secs(30),
        )
        .unwrap();
        let unstable = |j: usize| self.bounds[0][j].lower() < 0. && self.bounds[0][j].upper() > 0.;
        let positive = (0..self.dims[1])
            .filter(|&j| self.w[1][j] > 0. && unstable(j))
            .count();
        let negative = (0..self.dims[1])
            .filter(|&j| self.w[1][j] < 0. && unstable(j))
            .count();
        assert_eq!(r.relaxed_nonnegative, positive);
        assert_eq!(r.negative_binaries, negative);
        (parts, rows)
    }
    fn attempt(
        &self,
        parts: &mut MipParts,
        rows: &[Row],
        signed: bool,
        allowed: bool,
        end: Instant,
    ) -> Option<StrengtheningOutcome> {
        if signed {
            strengthen_with_signed_refinement(
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
        }
    }
    fn run(&self, signed: bool, allowed: bool) -> (MipParts, StrengtheningOutcome) {
        let (mut parts, rows) = self.parts();
        let before = parts.problem.clone();
        let binaries = parts.binary_vars.clone();
        let widths = parts.binary_widths.clone();
        let out = self
            .attempt(
                &mut parts,
                &rows,
                signed,
                allowed,
                Instant::now() + Duration::from_secs(30),
            )
            .unwrap();
        assert_eq!(&parts.problem.rows()[..before.num_rows()], before.rows());
        assert_eq!(parts.problem.cols(), before.cols());
        assert_eq!(parts.binary_vars, binaries);
        assert_eq!(parts.binary_widths, widths);
        (parts, out)
    }
    fn rational_margin(&self, x: &[Q]) -> Q {
        assert_eq!(x.len(), self.dims[0]);
        let h = self.dims[1];
        let d = self.dims[0];
        let mut margin = Q::from_float(self.b[1][0] - self.b[1][1]).unwrap();
        for j in 0..h {
            let z = Q::from_float(self.b[0][j]).unwrap()
                + (0..d)
                    .map(|i| Q::from_float(self.w[0][j * d + i]).unwrap() * &x[i])
                    .sum::<Q>();
            let c = Q::from_float(self.w[1][j]).unwrap() - Q::from_float(self.w[1][h + j]).unwrap();
            margin += c * z.max(Q::zero());
        }
        margin
    }
}

#[test]
fn signed_root_cli_real_coordinate_optimizer_closes_biased_twins_both_permutations() {
    for permuted in [false, true] {
        let (_, out) = Fixture::twins(0.25, permuted).run(true, true);
        assert_eq!(out.cuts.proposal_lp_calls, 1);
        assert_eq!(out.cuts.individual_cuts, 2);
        assert!(out.cuts.checked_hull_lower < 0.);
        let s = out
            .cuts
            .signed
            .as_ref()
            .expect("new mode keeps actual attempt evidence");
        assert!(s.attempted && s.admitted && s.refined_components > 0);
        assert!(s.assigned_positive_heads > 0 && s.negative_heads == 2);
        assert!(s.coordinate_proposals > 0 && s.completed_masks > 0 && s.checked_masks > 0);
        assert!(s.source_replays > 0 && s.work_units <= 64_000_000 && s.admitted_bytes > 0);
        assert!(s.lower.is_some_and(|v| v > 0.249 && v <= 0.25));
        assert_eq!(s.components, 2);
        assert_eq!(s.component_rows.appended_rows, s.components);
        assert_eq!(
            out.cuts.cuts,
            out.cuts.individual_cuts
                + out.cuts.compact_cuts
                + usize::from(out.cuts.global_sdp.is_some())
                + 1
                + s.component_rows.appended_rows
                + s.component_rows.phase.appended_rows
                + s.component_rows.joint.appended_rows
                + s.component_rows.domination.appended_rows
        );
        let proof = out.aggregate.unwrap();
        assert_eq!(proof.route(), "signed-coordinate");
        assert!(proof.verified() && proof.lower() > 0.249 && proof.lower() <= 0.25);
        assert!(
            out.continuation.is_none(),
            "a positive checked root skips the optional tree"
        );
    }
}

#[test]
fn signed_root_cli_negative_only_root_comparator_stays_open_without_full_solver_claim() {
    let (_, out) = Fixture::twins(0.25, false).run(false, false);
    assert_eq!(out.cuts.proposal_lp_calls, 1);
    assert!(out.cuts.signed.is_none());
    assert!(out.cuts.checked_hull_lower < 0.);
    assert!(out.cuts.compact_root_lower.is_none_or(|v| v <= 0.));
    assert!(out.cuts.global_sdp.as_ref().is_none_or(|s| s.lower <= 0.));
    assert!(out.aggregate.is_none() && out.continuation.is_none());
    // The old complete phase tree can already solve these twins. This only
    // distinguishes the named unconditional negative-only root family.
}

#[test]
fn signed_root_cli_unsafe_and_tie_keep_original_margin_nonpositive() {
    for permuted in [false, true] {
        for bias in [-0.25, 0.] {
            let f = Fixture::twins(bias, permuted);
            let (_, out) = f.run(true, true);
            let s = out.cuts.signed.as_ref().unwrap();
            assert!(s.admitted && s.coordinate_proposals > 0 && s.checked_masks > 0);
            assert!(s.lower.is_some_and(|v| v <= bias));
            assert!(out.aggregate.is_none());
            assert!(out.continuation.as_ref().is_none_or(|p| !p.verified()));
            let zero = vec![Q::zero(); 30];
            assert_eq!(f.rational_margin(&zero), Q::from_float(bias).unwrap());
        }
    }
}

#[test]
fn signed_root_cli_all_ingress_exclusions_keep_geometry_without_native_authority() {
    let (_, out) = Fixture::twins(0.25, false).run(true, false);
    assert!(out
        .cuts
        .signed
        .as_ref()
        .is_some_and(|s| s.admitted && s.lower.is_some_and(|v| v > 0.249)));
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
fn signed_root_cli_foreign_original_model_refuses_before_atomic_append() {
    for kind in 0..3 {
        let mut f = Fixture::twins(0.25, false);
        let (mut parts, rows) = f.parts();
        if kind == 0 {
            parts.output_vars.swap(0, 1);
        }
        if kind == 1 {
            f.w[0][29] = 0.75;
        }
        if kind == 2 {
            f.input[29] = Bound::new(-0.5, 1.);
        }
        let before = parts.problem.clone();
        let binaries = parts.binary_vars.clone();
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
        assert_eq!(parts.binary_vars, binaries);
        assert_eq!(parts.binary_widths, widths);
    }
}

#[test]
fn signed_root_cli_expired_original_clock_does_not_append_or_renew() {
    let f = Fixture::twins(0.25, false);
    let (mut parts, rows) = f.parts();
    let before = parts.problem.clone();
    let binaries = parts.binary_vars.clone();
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
    assert_eq!(parts.binary_vars, binaries);
}

#[test]
fn signed_root_cli_old_positive_group_skips_search_and_retains_attempt_receipt() {
    let (_, out) = Fixture::cycle(6.5).run(true, true);
    assert_eq!(out.cuts.proposal_lp_calls, 1);
    assert!(out.cuts.global_sdp.is_none());
    let s = out
        .cuts
        .signed
        .as_ref()
        .expect("new mode keeps no-row scheduling evidence");
    assert!(!s.admitted && s.coordinate_proposals == 0 && s.completed_masks == 0);
    let proof = out.aggregate.unwrap();
    assert_eq!(proof.route(), "compact");
    assert!(proof.verified() && proof.lower() > 0.499 && proof.lower() <= 0.5);
    assert!(out.continuation.is_none());
}

#[test]
fn signed_root_cli_actual_ay_retains_original_unsafe_witness_and_relaxed_metadata() {
    let f = Fixture::twins(-0.25, true);
    let (parts, out) = f.run(true, false);
    assert_eq!(parts.binary_vars.len(), 2);
    assert!(out
        .cuts
        .signed
        .as_ref()
        .is_some_and(|s| s.admitted && s.source_replays > 0));
    assert!(out.aggregate.is_none());
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
        panic!("signed root geometry excluded the original unsafe graph: {result:?}")
    };
    assert_eq!(input_values.len(), 30);
    let x: Vec<_> = input_values
        .iter()
        .map(|v| Q::from_float(*v).unwrap())
        .collect();
    let one = Q::from_integer(1.into());
    assert!(x.iter().all(|v| *v >= -&one && *v <= one));
    assert!(
        f.rational_margin(&x) <= Q::zero(),
        "unchanged original rational unsafe replay"
    );
}

#[test]
fn signed_root_cli_rendered_positive_root_remains_native_hull() {
    struct Capture;
    impl Drop for Capture {
        fn drop(&mut self) {
            crate::commands::beta_crown::output::end_capture();
        }
    }
    let (_, out) = Fixture::twins(0.25, false).run(true, true);
    let proof = out.aggregate.unwrap();
    assert_eq!(proof.route(), "signed-coordinate");
    assert!(proof.verified());
    crate::commands::beta_crown::output::begin_capture();
    let _capture = Capture;
    let result = VerificationResult::Verified {
        provenance: Default::default(),
        output_bounds: vec![Bound::new(0.249, 1.)],
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
    let raw = crate::commands::beta_crown::output::take_captured_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(value["status"], "verified");
    assert_eq!(value["method"], "native-hull");
}
