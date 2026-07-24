// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::vnnlib::{load_vnnlib, OutputConstraint};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_propagate::layers::LinearLayer;
use ny_propagate::{Layer, Network as PropNetwork};
use ny_tensor::BoundedTensor;
use ny_test_utils::workspace_root;
use std::path::PathBuf;

fn acasxu_1923_paths() -> (PathBuf, PathBuf) {
    let root = workspace_root();
    (
        root.join("benchmarks/vnncomp2023/benchmarks/acasxu/onnx/ACASXU_run2a_4_2_batch_2000.onnx"),
        root.join("benchmarks/vnncomp2023/benchmarks/acasxu/vnnlib/prop_2.vnnlib"),
    )
}

fn acasxu_1923_case() -> Option<(OnnxModel, BoundedTensor, Vec<Vec<f32>>)> {
    let (model_path, property_path) = acasxu_1923_paths();
    if !model_path.exists() || !property_path.exists() {
        eprintln!(
            "SKIP: optional ACAS-Xu benchmark assets are unavailable (model={}, property={}); \
             download them with benchmarks/download_benchmarks.sh",
            model_path.display(),
            property_path.display()
        );
        return None;
    }
    let model = load_onnx(&model_path).expect("load ACAS-Xu 4_2 ONNX model");
    let vnnlib = load_vnnlib(&property_path).expect("load ACAS-Xu prop_2 VNNLIB");
    let (lower_bounds, upper_bounds) = vnnlib.split_input_bounds_f32();
    let input_shape: Vec<usize> = model
        .network
        .inputs
        .first()
        .expect("ACAS-Xu model should have one input")
        .shape
        .iter()
        .map(|&dim| if dim <= 0 { 1 } else { dim as usize })
        .collect();
    let lower =
        ArrayD::from_shape_vec(IxDyn(&input_shape), lower_bounds).expect("ACAS-Xu lower shape");
    let upper =
        ArrayD::from_shape_vec(IxDyn(&input_shape), upper_bounds).expect("ACAS-Xu upper shape");
    let input = BoundedTensor::new(lower, upper).expect("ACAS-Xu input bounds");
    let objectives = vnnlib
        .output_constraints
        .iter()
        .map(|constraint| objective_from_constraint(constraint, vnnlib.num_outputs))
        .collect::<Vec<_>>();
    Some((model, input, objectives))
}

fn objective_from_constraint(constraint: &OutputConstraint, num_outputs: usize) -> Vec<f32> {
    let mut coeffs = vec![0.0_f32; num_outputs];
    match constraint {
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
            coeffs[*i] = 1.0;
            coeffs[*j] = -1.0;
        }
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            coeffs[*j] = 1.0;
            coeffs[*i] = -1.0;
        }
        other => panic!("expected relational ACAS-Xu prop_2 constraint, got {other:?}"),
    }
    coeffs
}

fn scalar_objective_network(network: &PropNetwork, coeffs: &[f32]) -> PropNetwork {
    let mut augmented = network.clone();
    let objective =
        Array2::from_shape_vec((1, coeffs.len()), coeffs.to_vec()).expect("objective shape");
    let bias = Array1::zeros(1);
    let projection = LinearLayer::new(objective, Some(bias)).expect("objective projection");
    augmented.add_layer(Layer::Linear(projection));
    augmented
}

fn single_row_spec(coeffs: &[f32]) -> Array2<f32> {
    Array2::from_shape_vec((1, coeffs.len()), coeffs.to_vec()).expect("spec shape")
}

fn scalar_bounds(bounds: &BoundedTensor) -> (f32, f32) {
    assert_eq!(
        bounds.len(),
        1,
        "expected scalar bounds, got {}",
        bounds.len()
    );
    let lower = *bounds.lower().iter().next().expect("scalar lower");
    let upper = *bounds.upper().iter().next().expect("scalar upper");
    (lower, upper)
}

fn split_input_dimension(
    input: &BoundedTensor,
    flat_dim: usize,
    take_left_child: bool,
) -> BoundedTensor {
    let lower_vec: Vec<f32> = input.lower().iter().copied().collect();
    let upper_vec: Vec<f32> = input.upper().iter().copied().collect();
    let mut child_lower = lower_vec.clone();
    let mut child_upper = upper_vec.clone();
    let midpoint = f32::midpoint(lower_vec[flat_dim], upper_vec[flat_dim]);
    if take_left_child {
        child_upper[flat_dim] = midpoint;
    } else {
        child_lower[flat_dim] = midpoint;
    }

    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(input.shape()), child_lower).expect("child lower shape"),
        ArrayD::from_shape_vec(IxDyn(input.shape()), child_upper).expect("child upper shape"),
    )
    .expect("child bounds remain valid")
}

fn assert_scalar_parity(
    seq_bounds: &BoundedTensor,
    graph_bounds: &BoundedTensor,
    context: &str,
    tolerance: f32,
) {
    let (seq_lower, seq_upper) = scalar_bounds(seq_bounds);
    let (graph_lower, graph_upper) = scalar_bounds(graph_bounds);
    let lower_diff = (seq_lower - graph_lower).abs();
    let upper_diff = (seq_upper - graph_upper).abs();
    eprintln!(
        "{context}: seq=[{seq_lower:.6}, {seq_upper:.6}] graph=[{graph_lower:.6}, {graph_upper:.6}] \
         diff=({lower_diff:.6}, {upper_diff:.6})"
    );
    assert!(
        lower_diff <= tolerance,
        "{context}: lower diff {lower_diff:.6} exceeded tolerance {tolerance:.6}"
    );
    assert!(
        upper_diff <= tolerance,
        "{context}: upper diff {upper_diff:.6} exceeded tolerance {tolerance:.6}"
    );
}

#[ntest::timeout(120000)]
#[test]
fn test_acasxu_4_2_prop_2_root_spec_guided_parity_1923() {
    let Some((model, input, objectives)) = acasxu_1923_case() else {
        return;
    };
    let network = model
        .to_propagate_network()
        .expect("convert ACAS-Xu 4_2 to sequential network");
    let graph = model
        .to_graph_network()
        .expect("convert ACAS-Xu 4_2 to graph network");

    for (objective_idx, coeffs) in objectives.iter().enumerate() {
        let augmented = scalar_objective_network(&network, coeffs);
        let seq_bounds = augmented
            .propagate_crown(&input)
            .expect("sequential scalar-objective CROWN");
        let graph_bounds = graph
            .propagate_crown_with_specs_and_engine(&input, &single_row_spec(coeffs), None)
            .expect("graph spec-guided CROWN");
        assert_scalar_parity(
            &seq_bounds,
            &graph_bounds,
            &format!("root objective[{objective_idx}]"),
            1e-3,
        );
    }
}

#[ntest::timeout(300000)]
#[test]
fn test_acasxu_4_2_prop_2_child_spec_guided_parity_1923() {
    let Some((model, input, objectives)) = acasxu_1923_case() else {
        return;
    };
    let network = model
        .to_propagate_network()
        .expect("convert ACAS-Xu 4_2 to sequential network");
    let graph = model
        .to_graph_network()
        .expect("convert ACAS-Xu 4_2 to graph network");

    for flat_dim in 0..input.len() {
        let lower = input
            .lower()
            .iter()
            .nth(flat_dim)
            .expect("input lower value");
        let upper = input
            .upper()
            .iter()
            .nth(flat_dim)
            .expect("input upper value");
        if !lower.is_finite() || !upper.is_finite() || upper <= lower {
            continue;
        }

        for (take_left_child, label) in [(true, "left"), (false, "right")] {
            let child = split_input_dimension(&input, flat_dim, take_left_child);
            for (objective_idx, coeffs) in objectives.iter().enumerate() {
                let augmented = scalar_objective_network(&network, coeffs);
                let seq_bounds = augmented
                    .propagate_crown(&child)
                    .expect("sequential child scalar-objective CROWN");
                let graph_bounds = graph
                    .propagate_crown_with_specs_and_engine(&child, &single_row_spec(coeffs), None)
                    .expect("graph child spec-guided CROWN");
                assert_scalar_parity(
                    &seq_bounds,
                    &graph_bounds,
                    &format!("child dim={flat_dim} {label} objective[{objective_idx}]"),
                    1e-3,
                );
            }
        }
    }
}
