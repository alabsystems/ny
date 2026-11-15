// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_mip::{MipBackend, MipConfig, MipResult, MipSolver};
use ny_onnx::vnnlib::OutputConstraint;

fn end() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

struct Fixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
    input: Vec<Bound>,
    spec: VnnLibSpec,
}

impl Fixture {
    // C-r(x)+r(2x)-r(-x)+r(-2x) = C+|x|. The separate-head
    // ideal root is C-1, so native completion must actually split.
    fn abs(constant: f64) -> Self {
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = 1;
        spec.num_outputs = 2;
        spec.input_bounds = vec![(-1.0, 1.0)];
        spec.output_constraints = vec![OutputConstraint::LessEq(0, 1)];
        Self {
            weights: vec![
                vec![1.0, 2.0, -1.0, -2.0],
                vec![-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            biases: vec![vec![0.0; 4], vec![constant, 0.0]],
            dims: vec![1, 4, 2],
            bounds: vec![vec![
                Bound::new(-1.0, 1.0),
                Bound::new(-2.0, 2.0),
                Bound::new(-1.0, 1.0),
                Bound::new(-2.0, 2.0),
            ]],
            input: vec![Bound::new(-1.0, 1.0)],
            spec,
        }
    }

    fn parts(&self) -> (MipParts, Vec<Row>) {
        let mut encoder = encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dims,
            &self.input,
            &self.bounds,
        )
        .unwrap();
        let rows = add_vnnlib_constraints(&mut encoder, &self.spec).unwrap();
        (encoder.into_parts(), rows)
    }

    fn strengthen(
        &self,
        parts: &mut MipParts,
        rows: &[Row],
        allow: bool,
    ) -> Option<StrengtheningOutcome> {
        strengthen_with_short_proof(
            parts,
            true,
            &self.weights,
            &self.biases,
            &self.dims,
            &self.bounds,
            &self.input,
            &self.spec,
            rows,
            end(),
            allow,
        )
    }
}

fn assert_global_prefix(parts: &MipParts, before: &ny_mip::ir::MilpProblem, cuts: usize) {
    assert_eq!(parts.problem.cols(), before.cols());
    assert_eq!(&parts.problem.rows()[..before.num_rows()], before.rows());
    assert_eq!(parts.problem.margin_row(), before.margin_row());
    assert_eq!(parts.problem.num_rows(), before.num_rows() + cuts);
}

#[test]
fn short_hull_phase_end_is_eighth_remaining_two_second_cap_and_original_stage() {
    let now = Instant::now();
    assert_eq!(
        continuation_end(now, Duration::from_secs(20), now + Duration::from_secs(4)),
        Some(now + Duration::from_secs(2))
    );
    assert_eq!(
        continuation_end(now, Duration::from_secs(8), now + Duration::from_secs(4)),
        Some(now + Duration::from_secs(1))
    );
    assert_eq!(
        continuation_end(
            now,
            Duration::from_secs(8),
            now + Duration::from_millis(400)
        ),
        Some(now + Duration::from_millis(400))
    );
    assert_eq!(
        continuation_end(now, Duration::from_nanos(8), now + Duration::from_secs(1)),
        Some(now + Duration::from_nanos(1))
    );
    assert_eq!(
        continuation_end(now, Duration::ZERO, now + Duration::from_secs(1)),
        None
    );
}

#[test]
fn short_hull_phase_end_cannot_renew_an_expired_or_nearly_spent_stage() {
    let now = Instant::now();
    let stage = now + Duration::from_secs(4);
    assert_eq!(
        continuation_end(
            now + Duration::from_millis(3999),
            Duration::from_secs(20),
            stage
        ),
        Some(stage)
    );
    assert_eq!(
        continuation_end(stage, Duration::from_secs(20), stage),
        None
    );
    assert_eq!(
        continuation_end(
            stage + Duration::from_secs(1),
            Duration::from_secs(20),
            stage
        ),
        None
    );
}

#[test]
fn short_hull_cli_reuses_one_actual_lp_and_completes_the_root_open_safe_abs_tree() {
    let f = Fixture::abs(0.25);
    let (mut parts, rows) = f.parts();
    let original = parts.problem.clone();
    let variables = parts.binary_vars.clone();
    let widths = parts.binary_widths.clone();
    let outcome = f
        .strengthen(&mut parts, &rows, true)
        .expect("global LP/cut admission");
    assert_eq!(outcome.cuts.proposal_lp_calls, 1);
    assert!(outcome.cuts.checked_hull_lower > -0.751 && outcome.cuts.checked_hull_lower <= -0.749);
    assert_global_prefix(&parts, &original, outcome.cuts.cuts);
    assert_eq!(parts.binary_vars, variables);
    assert_eq!(parts.binary_widths, widths);
    let proof = outcome.continuation.expect("same-root native continuation");
    let report = proof.report();
    assert!(proof.verified(), "{report:?}");
    assert!(report.verified());
    assert_eq!(
        report.root_lp_calls, 0,
        "no second root LP may be charged or constructed"
    );
    assert!(report.root_certified.is_some_and(|value| value <= 0.0));
    assert_eq!((report.neg_unstable, report.pos_unstable), (2, 2));
    assert!((3..=31).contains(&report.nodes), "{report:?}");
    assert_eq!(report.open, 0);
    assert_eq!(report.worst, None);
    assert!(report.closed > 0 && report.closed <= report.nodes);
}

#[test]
fn short_hull_cli_unsafe_and_equality_abs_cases_remain_open_with_all_global_rows() {
    for constant in [-0.25, 0.0] {
        let f = Fixture::abs(constant);
        let (mut parts, rows) = f.parts();
        let original = parts.problem.clone();
        let variables = parts.binary_vars.clone();
        let widths = parts.binary_widths.clone();
        let outcome = f.strengthen(&mut parts, &rows, true).unwrap();
        assert_eq!(outcome.cuts.proposal_lp_calls, 1);
        let proof = outcome
            .continuation
            .expect("unsafe/tie continuation is observed honestly");
        let report = proof.report();
        assert!(!proof.verified(), "C={constant}: {report:?}");
        assert!(!report.verified());
        assert_eq!(report.root_lp_calls, 0);
        assert!(report.open > 0 && report.worst.is_some_and(|value| value <= constant));
        assert_global_prefix(&parts, &original, outcome.cuts.cuts);
        assert_eq!(parts.binary_vars, variables);
        assert_eq!(parts.binary_widths, widths);
        // The actual original graph point x=0 has margin C<=0. In this
        // encoder all preactivations/heads/phases are zero at that point.
        let mut values = vec![0.0; parts.num_cols];
        values[parts.output_vars[0].0] = constant;
        for row in parts.problem.rows() {
            let lhs: f64 = row
                .coeffs
                .iter()
                .map(|(i, coefficient)| coefficient * values[*i])
                .sum();
            assert!(
                lhs >= row.lb && lhs <= row.ub,
                "global row excludes original x=0: {row:?}"
            );
        }
    }
}

#[test]
fn short_hull_cli_disabled_and_legacy_metadata_api_never_mint_a_continuation() {
    let f = Fixture::abs(0.25);
    let (mut parts, rows) = f.parts();
    let before = parts.problem.clone();
    let outcome = f.strengthen(&mut parts, &rows, false).unwrap();
    assert!(outcome.continuation.is_none());
    assert_eq!(outcome.cuts.proposal_lp_calls, 1);
    assert_global_prefix(&parts, &before, outcome.cuts.cuts);
    let (mut legacy, rows) = f.parts();
    let before = legacy.problem.clone();
    let report = strengthen(
        &mut legacy,
        true,
        &f.weights,
        &f.biases,
        &f.dims,
        &f.bounds,
        &f.input,
        &f.spec,
        &rows,
        end(),
    )
    .unwrap();
    assert_eq!(report.proposal_lp_calls, 1);
    assert_global_prefix(&legacy, &before, report.cuts);
}

#[test]
fn short_hull_cli_foreign_model_and_expiry_refuse_without_rows_or_proof() {
    for change in 0..4 {
        let mut f = Fixture::abs(0.25);
        let (mut parts, rows) = f.parts();
        if change == 0 {
            f.weights[0][0] = 0.5;
        }
        if change == 1 {
            f.input[0] = Bound::new(-0.5, 1.0);
        }
        if change == 2 {
            parts.output_vars.swap(0, 1);
        }
        let before = parts.problem.clone();
        let variables = parts.binary_vars.clone();
        let widths = parts.binary_widths.clone();
        let deadline = if change == 3 {
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("test clock supports an expired deadline")
        } else {
            end()
        };
        assert!(strengthen_with_short_proof(
            &mut parts, true, &f.weights, &f.biases, &f.dims, &f.bounds, &f.input, &f.spec, &rows,
            deadline, true
        )
        .is_none());
        assert_eq!(parts.problem, before);
        assert_eq!(parts.binary_vars, variables);
        assert_eq!(parts.binary_widths, widths);
    }
}

#[test]
fn short_hull_cli_open_fallback_still_allows_actual_ay_original_unsafe_witness() {
    let f = Fixture::abs(-0.25);
    let (mut parts, rows) = f.parts();
    let outcome = f.strengthen(&mut parts, &rows, true).unwrap();
    assert!(!outcome
        .continuation
        .as_ref()
        .is_some_and(HullMipContinuation::verified));
    let solver = MipSolver::new(
        parts,
        MipConfig {
            backend: MipBackend::Ay,
            parallel_split: 1,
            lp_tighten: false,
            timeout_secs: 10.0,
            ay_hard_deadline: Some(end()),
            ..Default::default()
        },
    );
    let result = solver.check_feasibility().unwrap();
    let MipResult::Sat { input_values, .. } = result else {
        panic!("original unsafe case: {result:?}")
    };
    assert_eq!(input_values.len(), 1);
    let x = input_values[0];
    assert!(x.is_finite() && (-1.0..=1.0).contains(&x));
    assert!(
        -0.25 + x.abs() <= 0.0,
        "returned input is not unsafe in the original graph"
    );
}

#[test]
fn short_hull_gate_preserves_required_shared_canary_and_original_bridge_authority() {
    for required_shared in [false, true] {
        for canary in [false, true] {
            for bridge_capture in [false, true] {
                let allowed = short_proof_allowed(required_shared, canary, bridge_capture);
                assert_eq!(
                    allowed,
                    (required_shared, canary, bridge_capture) == (false, false, false),
                    "required_shared={required_shared} canary={canary} bridge_capture={bridge_capture}"
                );
            }
        }
    }
}
