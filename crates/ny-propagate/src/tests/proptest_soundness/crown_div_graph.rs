// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest soundness coverage for spec-guided graph Div CROWN (#3626).
//!
//! These tests exercise the `spec_propagation.rs` Div handler directly through
//! `GraphNetwork::propagate_crown_with_specs_and_engine`, covering both:
//! - element-wise Div where each output has its own denominator
//! - broadcast Div where a row shares one denominator across multiple outputs

use crate::*;
use ndarray::{Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::sample_points;

const DIV_SOUNDNESS_TOLERANCE: f32 = 1e-4;
const DIV_TIGHTEN_TOLERANCE: f32 = 1e-3;

fn bounded_nd(shape: &[usize], lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
    let lo = ArrayD::from_shape_vec(IxDyn(shape), lower).expect("lower shape");
    let hi = ArrayD::from_shape_vec(IxDyn(shape), upper).expect("upper shape");
    BoundedTensor::new(lo, hi).expect("valid bounds")
}

fn valid_interval_vec(
    len: usize,
    min_value: f32,
    max_value: f32,
) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    proptest::collection::vec((min_value..=max_value, min_value..=max_value), len).prop_map(
        move |pairs| {
            let mut lower = Vec::with_capacity(len);
            let mut upper = Vec::with_capacity(len);
            for (a, b) in pairs {
                lower.push(a.min(b));
                upper.push(a.max(b));
            }
            (lower, upper)
        },
    )
}

fn build_elementwise_div_graph_3626() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let shift_const = ArrayD::from_shape_vec(IxDyn(&[2]), vec![4.0_f32, 5.0_f32]).unwrap();
    graph.add_node(GraphNode::from_input(
        "den",
        Layer::AddConstant(AddConstantLayer::new(shift_const)),
    ));
    graph.add_node(GraphNode::binary(
        "div",
        Layer::Div(DivLayer),
        NETWORK_INPUT,
        "den",
    ));
    graph.set_output("div");
    graph
}

fn eval_elementwise_div_graph_3626(x: &[f32; 2]) -> [f32; 2] {
    [x[0] / (x[0] + 4.0), x[1] / (x[1] + 5.0)]
}

fn build_rowwise_broadcast_div_graph_3626() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "reduce",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
    ));
    let shift_const = ArrayD::from_elem(IxDyn(&[1]), 4.0_f32);
    graph.add_node(GraphNode::new(
        "shift",
        Layer::AddConstant(AddConstantLayer::new(shift_const)),
        vec!["reduce".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "div",
        Layer::Div(DivLayer),
        NETWORK_INPUT,
        "shift",
    ));
    graph.set_output("div");
    graph
}

fn eval_rowwise_broadcast_div_graph_3626(x: &[f32; 4]) -> [f32; 4] {
    let denom0 = x[0] + x[1] + 4.0;
    let denom1 = x[2] + x[3] + 4.0;
    [x[0] / denom0, x[1] / denom0, x[2] / denom1, x[3] / denom1]
}

fn soundness_tol(true_value: f32, lower: f32, upper: f32) -> f32 {
    DIV_SOUNDNESS_TOLERANCE * true_value.abs().max(lower.abs()).max(upper.abs()).max(1.0)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Spec-guided graph CROWN soundness for element-wise Div.
    ///
    /// Graph: `y = x / (x + c)` with `c = [4, 5]`.
    /// The positive denominator keeps the reciprocal well-defined while still
    /// allowing mixed-sign numerators, which is the failure mode from #3626.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_div_spec_guided_crown_elementwise(
        (lower, upper) in valid_interval_vec(2, -3.0, 2.0),
    ) {
        let graph = build_elementwise_div_graph_3626();
        let input = bounded_nd(&[2], lower.clone(), upper.clone());

        let ibp = graph.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(format!("IBP failed: {e}")))?;
        let crown = graph
            .propagate_crown_with_specs_and_engine(&input, &Array2::eye(2), None)
            .map_err(|e| TestCaseError::fail(format!("CROWN failed: {e}")))?;

        prop_assert_eq!(ibp.shape(), &[2], "IBP shape mismatch");
        prop_assert_eq!(crown.len(), 2, "CROWN flat output size mismatch");

        let ibp_lower: Vec<f32> = ibp.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp.upper().iter().copied().collect();

        for dim in 0..2 {
            let crown_lo = crown.lower()[[dim]];
            let crown_hi = crown.upper()[[dim]];
            prop_assert!(crown_lo.is_finite(), "dim {dim}: non-finite lower {crown_lo}");
            prop_assert!(crown_hi.is_finite(), "dim {dim}: non-finite upper {crown_hi}");
            prop_assert!(crown_lo <= crown_hi, "dim {dim}: invalid bounds {crown_lo} > {crown_hi}");
            prop_assert!(
                crown_lo >= ibp_lower[dim] - DIV_TIGHTEN_TOLERANCE,
                "dim {dim}: CROWN lower {crown_lo} looser than IBP lower {}",
                ibp_lower[dim]
            );
            prop_assert!(
                crown_hi <= ibp_upper[dim] + DIV_TIGHTEN_TOLERANCE,
                "dim {dim}: CROWN upper {crown_hi} looser than IBP upper {}",
                ibp_upper[dim]
            );
        }

        for x0 in sample_points(lower[0], upper[0], 5) {
            for x1 in sample_points(lower[1], upper[1], 5) {
                let output = eval_elementwise_div_graph_3626(&[x0, x1]);
                for (dim, &true_value) in output.iter().enumerate() {
                    let crown_lo = crown.lower()[[dim]];
                    let crown_hi = crown.upper()[[dim]];
                    let tol = soundness_tol(true_value, crown_lo, crown_hi);
                    prop_assert!(
                        crown_lo - tol <= true_value,
                        "dim {dim}: lower unsound for x=[{x0}, {x1}]: \
                         lower={crown_lo}, true={true_value}, tol={tol}"
                    );
                    prop_assert!(
                        true_value <= crown_hi + tol,
                        "dim {dim}: upper unsound for x=[{x0}, {x1}]: \
                         true={true_value}, upper={crown_hi}, tol={tol}"
                    );
                }
            }
        }
    }

    /// Spec-guided graph CROWN soundness for broadcast Div.
    ///
    /// Graph: `y[row, col] = x[row, col] / (sum_row(x) + 4)`.
    /// This exercises the grouped midpoint+bias path where multiple outputs
    /// share the same reciprocal interval.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_div_spec_guided_crown_rowwise_broadcast(
        (lower, upper) in valid_interval_vec(4, -1.5, 1.5),
    ) {
        let graph = build_rowwise_broadcast_div_graph_3626();
        let input = bounded_nd(&[2, 2], lower.clone(), upper.clone());

        let ibp = graph.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(format!("IBP failed: {e}")))?;
        let crown = graph
            .propagate_crown_with_specs_and_engine(&input, &Array2::eye(4), None)
            .map_err(|e| TestCaseError::fail(format!("CROWN failed: {e}")))?;

        prop_assert_eq!(ibp.shape(), &[2, 2], "IBP shape mismatch");
        prop_assert_eq!(crown.len(), 4, "CROWN flat output size mismatch");

        let ibp_lower: Vec<f32> = ibp.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp.upper().iter().copied().collect();

        for dim in 0..4 {
            let crown_lo = crown.lower()[[dim]];
            let crown_hi = crown.upper()[[dim]];
            prop_assert!(crown_lo.is_finite(), "dim {dim}: non-finite lower {crown_lo}");
            prop_assert!(crown_hi.is_finite(), "dim {dim}: non-finite upper {crown_hi}");
            prop_assert!(crown_lo <= crown_hi, "dim {dim}: invalid bounds {crown_lo} > {crown_hi}");
            prop_assert!(
                crown_lo >= ibp_lower[dim] - DIV_TIGHTEN_TOLERANCE,
                "dim {dim}: CROWN lower {crown_lo} looser than IBP lower {}",
                ibp_lower[dim]
            );
            prop_assert!(
                crown_hi <= ibp_upper[dim] + DIV_TIGHTEN_TOLERANCE,
                "dim {dim}: CROWN upper {crown_hi} looser than IBP upper {}",
                ibp_upper[dim]
            );
        }

        for x0 in sample_points(lower[0], upper[0], 3) {
            for x1 in sample_points(lower[1], upper[1], 3) {
                for x2 in sample_points(lower[2], upper[2], 3) {
                    for x3 in sample_points(lower[3], upper[3], 3) {
                        let output = eval_rowwise_broadcast_div_graph_3626(&[x0, x1, x2, x3]);
                        for (dim, &true_value) in output.iter().enumerate() {
                            let crown_lo = crown.lower()[[dim]];
                            let crown_hi = crown.upper()[[dim]];
                            let tol = soundness_tol(true_value, crown_lo, crown_hi);
                            prop_assert!(
                                crown_lo - tol <= true_value,
                                "dim {dim}: lower unsound for x=[{x0}, {x1}, {x2}, {x3}]: \
                                 lower={crown_lo}, true={true_value}, tol={tol}"
                            );
                            prop_assert!(
                                true_value <= crown_hi + tol,
                                "dim {dim}: upper unsound for x=[{x0}, {x1}, {x2}, {x3}]: \
                                 true={true_value}, upper={crown_hi}, tol={tol}"
                            );
                        }
                    }
                }
            }
        }
    }
}
