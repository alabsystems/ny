// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::{add_vnnlib_constraints, map_mip_result_revalidated};
use super::*;
use ndarray::{arr1, arr2};
use ny_core::VerificationResult;
use ny_mip::{encode_feedforward, MipBackend, MipConfig, MipResult, MipSolver};
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{Layer, Network};
use ny_tensor::BoundedTensor;
use std::time::Duration;

struct Fixture {
    parts: MipParts,
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
    spec: VnnLibSpec,
    rows: Vec<Row>,
}

impl Fixture {
    fn mixed() -> Self {
        Self::build(
            vec![1.0; 5],
            vec![0.0, 0.0, 0.0, 2.0, -2.0],
            vec![1.0, -1.0, 0.0, 2.0, -2.0],
            0.25,
        )
    }

    fn duplicate(bias: f64) -> Self {
        Self::build(vec![1.0; 2], vec![0.0; 2], vec![1.0, -1.0], bias)
    }

    fn build(w1: Vec<f64>, b1: Vec<f64>, q: Vec<f64>, bias: f64) -> Self {
        let h = b1.len();
        let mut output = q;
        output.extend(vec![0.0; h]);
        let weights = vec![w1, output];
        let biases = vec![b1.clone(), vec![bias, 0.0]];
        let dims = vec![1, h, 2];
        let bounds = vec![b1
            .iter()
            .map(|b| Bound::new((b - 1.0) as f32, (b + 1.0) as f32))
            .collect()];
        let mut spec = VnnLibSpec::new();
        spec.num_inputs = 1;
        spec.num_outputs = 2;
        spec.input_bounds = vec![(-1.0, 1.0)];
        spec.output_constraints = vec![OutputConstraint::LessEq(0, 1)];
        let mut encoder =
            encode_feedforward(&weights, &biases, &dims, &[Bound::new(-1.0, 1.0)], &bounds)
                .unwrap();
        let rows = add_vnnlib_constraints(&mut encoder, &spec).unwrap();
        Self {
            parts: encoder.into_parts(),
            weights,
            biases,
            dims,
            bounds,
            spec,
            rows,
        }
    }

    fn relax(&mut self) -> Option<RelaxationReport> {
        relax(
            &mut self.parts,
            true,
            &self.weights,
            &self.biases,
            &self.dims,
            &self.bounds,
            &self.spec,
            &self.rows,
            Instant::now() + Duration::from_secs(5),
        )
    }

    fn assert_unchanged_refusal(&mut self) {
        let before = self.parts.problem.clone();
        let vars = self.parts.binary_vars.clone();
        let widths = self.parts.binary_widths.clone();
        assert!(self.relax().is_none());
        assert_eq!(self.parts.problem, before);
        assert_eq!(self.parts.binary_vars, vars);
        assert_eq!(self.parts.binary_widths, widths);
    }
}

fn original_duplicate(bias: f32) -> (Network, BoundedTensor) {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0], [1.0]]), None).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, -1.0], [0.0, 0.0]]), Some(arr1(&[bias, 0.0]))).unwrap(),
    ));
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    (network, input)
}

#[test]
fn signed_binary_mapping_keeps_negative_only_and_preserves_exact_model_rows() {
    let mut f = Fixture::mixed();
    f.parts.problem.mark_margin_row(f.rows[0]).unwrap();
    let before = f.parts.problem.clone();
    let original_vars = f.parts.binary_vars.clone();
    let report = f.relax().unwrap();
    assert_eq!(
        report,
        RelaxationReport {
            unstable_before: 3,
            relaxed_nonnegative: 2,
            negative_binaries: 1,
        }
    );
    assert_eq!(f.parts.binary_vars, vec![original_vars[1]]);
    assert_eq!(f.parts.binary_widths, vec![2.0]);
    assert_eq!(f.parts.problem.rows(), before.rows());
    assert_eq!(f.parts.problem.margin_row(), before.margin_row());
    let mut expected = before.cols().to_vec();
    expected[original_vars[0].0].integer = false;
    expected[original_vars[2].0].integer = false;
    assert_eq!(f.parts.problem.cols(), expected);
}

#[test]
fn signed_binary_reverse_property_reverses_convex_role() {
    let mut f = Fixture::duplicate(0.25);
    let original = f.parts.binary_vars.clone();
    f.spec.output_constraints = vec![OutputConstraint::GreaterEq(0, 1)];
    let mut encoder = encode_feedforward(
        &f.weights,
        &f.biases,
        &f.dims,
        &[Bound::new(-1.0, 1.0)],
        &f.bounds,
    )
    .unwrap();
    f.rows = add_vnnlib_constraints(&mut encoder, &f.spec).unwrap();
    f.parts = encoder.into_parts();
    assert_eq!(f.relax().unwrap().negative_binaries, 1);
    assert_eq!(f.parts.binary_vars, vec![original[0]]);
}

#[test]
fn signed_binary_refuses_multirow_or_disjunctive_properties() {
    let mut f = Fixture::mixed();
    f.spec
        .output_constraints
        .push(OutputConstraint::GreaterEq(0, 1));
    f.assert_unchanged_refusal();
    let mut f = Fixture::mixed();
    f.spec.is_disjunction = true;
    f.assert_unchanged_refusal();
}

#[test]
fn signed_binary_refuses_foreign_reordered_or_duplicate_binary_handles() {
    for variant in 0..3 {
        let mut f = Fixture::mixed();
        match variant {
            0 => f.parts.binary_vars[0] = f.parts.input_vars[0],
            1 => f.parts.binary_vars.swap(0, 1),
            _ => f.parts.binary_vars[1] = f.parts.binary_vars[0],
        }
        f.assert_unchanged_refusal();
    }
}

#[test]
fn signed_binary_refuses_width_or_output_column_misalignment() {
    let mut f = Fixture::mixed();
    f.parts.binary_widths[2] = f64::from_bits(2.0f64.to_bits() + 1);
    f.assert_unchanged_refusal();
    let mut f = Fixture::mixed();
    f.parts.output_vars.swap(0, 1);
    f.assert_unchanged_refusal();
}

#[test]
fn signed_binary_refuses_shape_nonfinite_and_missing_bound_vectors() {
    let mut f = Fixture::mixed();
    f.weights[1][0] = f64::NAN;
    f.assert_unchanged_refusal();
    let mut f = Fixture::mixed();
    f.bounds[0].pop();
    f.assert_unchanged_refusal();
    let mut f = Fixture::mixed();
    f.dims.push(2);
    f.assert_unchanged_refusal();
}

#[test]
fn signed_binary_refuses_wrong_unsafe_row_identity_transactionally() {
    let mut f = Fixture::mixed();
    f.rows = vec![Row(0)];
    f.assert_unchanged_refusal();
    let mut f = Fixture::mixed();
    f.rows.push(f.rows[0]);
    f.assert_unchanged_refusal();
}

#[test]
fn signed_binary_expired_or_non_single_hidden_route_is_inert() {
    for single_hidden in [false, true] {
        let mut f = Fixture::mixed();
        let before = f.parts.problem.clone();
        let end = if single_hidden {
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("test clock supports an expired deadline")
        } else {
            Instant::now() + Duration::from_secs(5)
        };
        assert!(relax(
            &mut f.parts,
            single_hidden,
            &f.weights,
            &f.biases,
            &f.dims,
            &f.bounds,
            &f.spec,
            &f.rows,
            end
        )
        .is_none());
        assert_eq!(f.parts.problem, before);
    }
}

#[test]
fn signed_binary_all_original_graph_grid_points_remain_feasible() {
    let mut f = Fixture::mixed();
    // Test the network region alone; unsafe exclusion is tested independently.
    f.relax().unwrap();
    for x in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
        let point = super::super::warm_start::build_warm_start_vector(
            &arr1(&[x]).into_dyn(),
            &f.weights,
            &f.biases,
            &f.dims,
            &f.bounds,
            f.parts.num_cols,
        )
        .unwrap();
        for (col, spec) in point.iter().zip(f.parts.problem.cols()) {
            assert!(*col >= spec.lb && *col <= spec.ub);
            if spec.integer {
                assert!(*col == 0.0 || *col == 1.0);
            }
        }
        for row in &f.parts.problem.rows()[..f.rows[0].0] {
            let lhs: f64 = row.coeffs.iter().map(|(i, c)| point[*i] * c).sum();
            assert!(lhs >= row.lb && lhs <= row.ub);
        }
    }
}

#[test]
fn signed_binary_fractional_outputs_cannot_publish_a_false_counterexample() {
    let (network, input) = original_duplicate(0.25);
    let result = map_mip_result_revalidated(
        MipResult::Sat {
            objective: -1.0,
            output_values: vec![-1.0, 0.0],
            input_values: vec![0.0],
            dual_bound: None,
        },
        &network,
        &input,
        &[OutputConstraint::LessEq(0, 1)],
        2,
    );
    assert!(matches!(result, VerificationResult::Unknown { .. }));
}

#[test]
fn signed_binary_original_unsafe_witness_still_passes_existing_replay() {
    let (network, input) = original_duplicate(-0.25);
    let result = map_mip_result_revalidated(
        MipResult::Sat {
            objective: 0.0,
            output_values: vec![123.0, -123.0],
            input_values: vec![0.0],
            dual_bound: None,
        },
        &network,
        &input,
        &[OutputConstraint::LessEq(0, 1)],
        2,
    );
    assert!(matches!(result, VerificationResult::Violated { .. }));
}

#[test]
fn signed_binary_timeout_and_unchecked_unsat_never_publish_verified() {
    let (network, input) = original_duplicate(0.25);
    for candidate in [MipResult::Timeout, MipResult::Unsat { certified: false }] {
        let mapped = map_mip_result_revalidated(
            candidate,
            &network,
            &input,
            &[OutputConstraint::LessEq(0, 1)],
            2,
        );
        assert!(!matches!(mapped, VerificationResult::Verified { .. }));
    }
}

#[test]
fn signed_binary_infeasibility_requires_actual_ay_model_bound_certificate() {
    let mut f = Fixture::duplicate(0.25);
    assert_eq!(f.relax().unwrap().negative_binaries, 1);
    // Production's typed margin cycle marks this same original unsafe row.
    // This selects the model-bound threshold proof ingress, without changing
    // any constraint, bound or objective.
    f.parts.problem.mark_margin_row(f.rows[0]).unwrap();
    let end = Instant::now() + Duration::from_secs(5);
    let solver = MipSolver::new(
        f.parts,
        MipConfig {
            backend: MipBackend::Ay,
            parallel_split: 1,
            lp_tighten: false,
            timeout_secs: 5.0,
            ay_hard_deadline: Some(end),
            ..Default::default()
        },
    );
    let actual = solver.check_feasibility().unwrap();
    assert!(
        matches!(actual, MipResult::Unsat { certified: true }),
        "actual marked AY outcome: {actual:?}"
    );
}

#[test]
fn signed_binary_unmarked_actual_ay_cannot_publish_an_unchecked_verdict() {
    let mut f = Fixture::duplicate(0.25);
    assert_eq!(f.relax().unwrap().negative_binaries, 1);
    let solver = MipSolver::new(
        f.parts,
        MipConfig {
            backend: MipBackend::Ay,
            parallel_split: 1,
            lp_tighten: false,
            timeout_secs: 5.0,
            ay_hard_deadline: Some(Instant::now() + Duration::from_secs(5)),
            ..Default::default()
        },
    );
    let actual = solver.check_feasibility().unwrap();
    assert!(!matches!(actual, MipResult::Sat { .. }), "{actual:?}");
    let checked = matches!(actual, MipResult::Unsat { certified: true });
    let (network, input) = original_duplicate(0.25);
    let mapped = map_mip_result_revalidated(
        actual,
        &network,
        &input,
        &[OutputConstraint::LessEq(0, 1)],
        2,
    );
    if matches!(mapped, VerificationResult::Verified { .. }) {
        assert!(checked, "unmarked numerical outcome gained proof authority");
    }
}
