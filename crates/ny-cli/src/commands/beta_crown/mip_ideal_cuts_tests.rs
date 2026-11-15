// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_onnx::vnnlib::OutputConstraint;

fn fixture() -> (
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<usize>,
    Vec<Vec<Bound>>,
    VnnLibSpec,
) {
    let weights = vec![
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        vec![-1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ];
    let biases = vec![vec![0.0, 2.0, 0.0], vec![-0.75, 0.0]];
    let dims = vec![2, 3, 2];
    let bounds = vec![vec![
        Bound::new(-2.0, 2.0),
        Bound::new(1.0, 3.0),
        Bound::new(-1.0, 1.0),
    ]];
    let mut spec = VnnLibSpec::new();
    spec.num_inputs = 2;
    spec.num_outputs = 2;
    spec.input_bounds = vec![(-1.0, 1.0); 2];
    spec.output_constraints = vec![OutputConstraint::LessEq(0, 1)];
    (weights, biases, dims, bounds, spec)
}

#[test]
fn ideal_mip_cut_stage_uses_quarter_of_original_time_without_a_floor() {
    let now = Instant::now();
    assert_eq!(stage_end(now, now), None);
    assert_eq!(
        stage_end(now, now + Duration::from_secs(20)),
        Some(now + Duration::from_secs(4))
    );
    assert_eq!(
        stage_end(now, now + Duration::from_secs(2)),
        Some(now + Duration::from_millis(500))
    );
    assert_eq!(
        stage_end(now, now + Duration::from_nanos(4)),
        Some(now + Duration::from_nanos(1))
    );
}

#[test]
fn ideal_mip_cut_cli_route_refuses_unrelated_topology_property_and_expiry() {
    let (w, b, d, bounds, mut spec) = fixture();
    let input = [Bound::new(-1.0, 1.0); 2];
    let mut encoder = encode_feedforward(&w, &b, &d, &input, &bounds).unwrap();
    let rows = add_vnnlib_constraints(&mut encoder, &spec).unwrap();
    let mut parts = encoder.into_parts();
    let before = parts.problem.clone();
    assert!(strengthen(
        &mut parts,
        false,
        &w,
        &b,
        &d,
        &bounds,
        &input,
        &spec,
        &rows,
        Instant::now() + Duration::from_secs(30)
    )
    .is_none());
    spec.output_constraints
        .push(OutputConstraint::GreaterEq(0, 1));
    assert!(strengthen(
        &mut parts,
        true,
        &w,
        &b,
        &d,
        &bounds,
        &input,
        &spec,
        &rows,
        Instant::now() + Duration::from_secs(30)
    )
    .is_none());
    spec.output_constraints.pop();
    assert!(strengthen(
        &mut parts,
        true,
        &w,
        &b,
        &d,
        &bounds,
        &input,
        &spec,
        &rows,
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("test clock supports an expired deadline")
    )
    .is_none());
    assert_eq!(parts.problem, before);
}

#[test]
fn ideal_mip_cut_cli_actual_lp_survives_previous_convex_binary_relaxation() {
    let (w, b, d, bounds, spec) = fixture();
    let input = [Bound::new(-1.0, 1.0); 2];
    let mut encoder = encode_feedforward(&w, &b, &d, &input, &bounds).unwrap();
    let rows = add_vnnlib_constraints(&mut encoder, &spec).unwrap();
    let mut parts = encoder.into_parts();
    let end = Instant::now() + Duration::from_secs(30);
    let relaxed =
        convex_binary::relax(&mut parts, true, &w, &b, &d, &bounds, &spec, &rows, end).unwrap();
    assert_eq!((relaxed.unstable_before, relaxed.negative_binaries), (2, 1));
    let before = parts.problem.clone();
    let variables = parts.binary_vars.clone();
    let report = strengthen(
        &mut parts, true, &w, &b, &d, &bounds, &input, &spec, &rows, end,
    )
    .unwrap();
    assert!(report.cuts >= 1 && report.input_terms >= 1);
    assert!(report.checked_hull_lower > 0.249 && report.checked_hull_lower <= 0.25);
    assert_eq!(parts.problem.cols(), before.cols());
    assert_eq!(&parts.problem.rows()[..before.num_rows()], before.rows());
    assert_eq!(parts.binary_vars, variables);
}
