// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork Min/Max binary IBP tests.
//!
//! Reference: alpha-beta-CROWN `auto_LiRPA/operators/minmax.py`

use crate::*;
use ndarray::Array1;

#[ntest::timeout(10000)]
#[test]
fn test_minbinary_ibp_soundness() {
    // Test that MinBinary IBP produces sound bounds.
    // For min(x, y) where x ∈ [x_l, x_u] and y ∈ [y_l, y_u]:
    //   lower = min(x_l, y_l)
    //   upper = min(x_u, y_u)
    let min = MinBinaryLayer;

    // Test case 1: [1, 3] min [2, 4] = [min(1,2), min(3,4)] = [1, 3]
    let input_a = BoundedTensor::new(
        Array1::from_vec(vec![1.0_f32]).into_dyn(),
        Array1::from_vec(vec![3.0_f32]).into_dyn(),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        Array1::from_vec(vec![2.0_f32]).into_dyn(),
        Array1::from_vec(vec![4.0_f32]).into_dyn(),
    )
    .unwrap();

    let result = min.propagate_ibp_binary(&input_a, &input_b).unwrap();
    assert!(
        (result.lower()[0] - 1.0).abs() < 1e-5,
        "lower should be 1.0, got {}",
        result.lower()[0]
    );
    assert!(
        (result.upper()[0] - 3.0).abs() < 1e-5,
        "upper should be 3.0, got {}",
        result.upper()[0]
    );

    // Test case 2: [5, 10] min [2, 4] = [min(5,2), min(10,4)] = [2, 4]
    let input_a2 = BoundedTensor::new(
        Array1::from_vec(vec![5.0_f32]).into_dyn(),
        Array1::from_vec(vec![10.0_f32]).into_dyn(),
    )
    .unwrap();
    let input_b2 = BoundedTensor::new(
        Array1::from_vec(vec![2.0_f32]).into_dyn(),
        Array1::from_vec(vec![4.0_f32]).into_dyn(),
    )
    .unwrap();

    let result2 = min.propagate_ibp_binary(&input_a2, &input_b2).unwrap();
    assert!(
        (result2.lower()[0] - 2.0).abs() < 1e-5,
        "lower should be 2.0, got {}",
        result2.lower()[0]
    );
    assert!(
        (result2.upper()[0] - 4.0).abs() < 1e-5,
        "upper should be 4.0, got {}",
        result2.upper()[0]
    );

    // Test case 3: Negative bounds [-3, -1] min [-2, 2] = [min(-3,-2), min(-1,2)] = [-3, -1]
    let input_a3 = BoundedTensor::new(
        Array1::from_vec(vec![-3.0_f32]).into_dyn(),
        Array1::from_vec(vec![-1.0_f32]).into_dyn(),
    )
    .unwrap();
    let input_b3 = BoundedTensor::new(
        Array1::from_vec(vec![-2.0_f32]).into_dyn(),
        Array1::from_vec(vec![2.0_f32]).into_dyn(),
    )
    .unwrap();

    let result3 = min.propagate_ibp_binary(&input_a3, &input_b3).unwrap();
    assert!(
        (result3.lower()[0] - (-3.0)).abs() < 1e-5,
        "lower should be -3.0, got {}",
        result3.lower()[0]
    );
    assert!(
        (result3.upper()[0] - (-1.0)).abs() < 1e-5,
        "upper should be -1.0, got {}",
        result3.upper()[0]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_maxbinary_ibp_soundness() {
    // Test that MaxBinary IBP produces sound bounds.
    // For max(x, y) where x ∈ [x_l, x_u] and y ∈ [y_l, y_u]:
    //   lower = max(x_l, y_l)
    //   upper = max(x_u, y_u)
    let max = MaxBinaryLayer;

    // Test case 1: [1, 3] max [2, 4] = [max(1,2), max(3,4)] = [2, 4]
    let input_a = BoundedTensor::new(
        Array1::from_vec(vec![1.0_f32]).into_dyn(),
        Array1::from_vec(vec![3.0_f32]).into_dyn(),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        Array1::from_vec(vec![2.0_f32]).into_dyn(),
        Array1::from_vec(vec![4.0_f32]).into_dyn(),
    )
    .unwrap();

    let result = max.propagate_ibp_binary(&input_a, &input_b).unwrap();
    assert!(
        (result.lower()[0] - 2.0).abs() < 1e-5,
        "lower should be 2.0, got {}",
        result.lower()[0]
    );
    assert!(
        (result.upper()[0] - 4.0).abs() < 1e-5,
        "upper should be 4.0, got {}",
        result.upper()[0]
    );

    // Test case 2: [5, 10] max [2, 4] = [max(5,2), max(10,4)] = [5, 10]
    let input_a2 = BoundedTensor::new(
        Array1::from_vec(vec![5.0_f32]).into_dyn(),
        Array1::from_vec(vec![10.0_f32]).into_dyn(),
    )
    .unwrap();
    let input_b2 = BoundedTensor::new(
        Array1::from_vec(vec![2.0_f32]).into_dyn(),
        Array1::from_vec(vec![4.0_f32]).into_dyn(),
    )
    .unwrap();

    let result2 = max.propagate_ibp_binary(&input_a2, &input_b2).unwrap();
    assert!(
        (result2.lower()[0] - 5.0).abs() < 1e-5,
        "lower should be 5.0, got {}",
        result2.lower()[0]
    );
    assert!(
        (result2.upper()[0] - 10.0).abs() < 1e-5,
        "upper should be 10.0, got {}",
        result2.upper()[0]
    );

    // Test case 3: Negative bounds [-3, -1] max [-2, 2] = [max(-3,-2), max(-1,2)] = [-2, 2]
    let input_a3 = BoundedTensor::new(
        Array1::from_vec(vec![-3.0_f32]).into_dyn(),
        Array1::from_vec(vec![-1.0_f32]).into_dyn(),
    )
    .unwrap();
    let input_b3 = BoundedTensor::new(
        Array1::from_vec(vec![-2.0_f32]).into_dyn(),
        Array1::from_vec(vec![2.0_f32]).into_dyn(),
    )
    .unwrap();

    let result3 = max.propagate_ibp_binary(&input_a3, &input_b3).unwrap();
    assert!(
        (result3.lower()[0] - (-2.0)).abs() < 1e-5,
        "lower should be -2.0, got {}",
        result3.lower()[0]
    );
    assert!(
        (result3.upper()[0] - 2.0).abs() < 1e-5,
        "upper should be 2.0, got {}",
        result3.upper()[0]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_minmax_ibp_broadcasting() {
    // Test that Min/Max handle broadcasting correctly
    let min = MinBinaryLayer;
    let max = MaxBinaryLayer;

    // 2-element tensor min/max scalar
    let input_a = BoundedTensor::new(
        Array1::from_vec(vec![1.0_f32, 5.0]).into_dyn(),
        Array1::from_vec(vec![3.0_f32, 7.0]).into_dyn(),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        Array1::from_vec(vec![2.0_f32]).into_dyn(),
        Array1::from_vec(vec![4.0_f32]).into_dyn(),
    )
    .unwrap();

    // min([1,5], [2]) with broadcast = min([1,5], [2,2]) = [min(1,2), min(5,2)] = [1, 2]
    // upper: min([3,7], [4,4]) = [min(3,4), min(7,4)] = [3, 4]
    let min_result = min.propagate_ibp_binary(&input_a, &input_b).unwrap();
    assert_eq!(min_result.shape(), &[2]);
    assert!(
        (min_result.lower()[0] - 1.0).abs() < 1e-5,
        "min lower[0] should be 1.0"
    );
    assert!(
        (min_result.lower()[1] - 2.0).abs() < 1e-5,
        "min lower[1] should be 2.0"
    );
    assert!(
        (min_result.upper()[0] - 3.0).abs() < 1e-5,
        "min upper[0] should be 3.0"
    );
    assert!(
        (min_result.upper()[1] - 4.0).abs() < 1e-5,
        "min upper[1] should be 4.0"
    );

    // max([1,5], [2,2]) = [max(1,2), max(5,2)] = [2, 5]
    // upper: max([3,7], [4,4]) = [max(3,4), max(7,4)] = [4, 7]
    let max_result = max.propagate_ibp_binary(&input_a, &input_b).unwrap();
    assert_eq!(max_result.shape(), &[2]);
    assert!(
        (max_result.lower()[0] - 2.0).abs() < 1e-5,
        "max lower[0] should be 2.0"
    );
    assert!(
        (max_result.lower()[1] - 5.0).abs() < 1e-5,
        "max lower[1] should be 5.0"
    );
    assert!(
        (max_result.upper()[0] - 4.0).abs() < 1e-5,
        "max upper[0] should be 4.0"
    );
    assert!(
        (max_result.upper()[1] - 7.0).abs() < 1e-5,
        "max upper[1] should be 7.0"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_minmax_in_graph_network_ibp() {
    // Test Min/Max in a GraphNetwork with IBP propagation
    use ndarray::Array2;

    let mut graph = GraphNetwork::new();
    let hidden = 4;

    // Two linear branches from input
    let linear_a = LinearLayer::new(
        Array2::<f32>::from_elem((hidden, hidden), 0.5),
        Some(Array1::<f32>::zeros(hidden)),
    )
    .unwrap();
    graph.add_node(GraphNode::new(
        "branch_a",
        Layer::Linear(linear_a),
        vec!["_input".to_string()],
    ));

    let linear_b = LinearLayer::new(
        Array2::<f32>::from_elem((hidden, hidden), 0.3),
        Some(Array1::<f32>::zeros(hidden)),
    )
    .unwrap();
    graph.add_node(GraphNode::new(
        "branch_b",
        Layer::Linear(linear_b),
        vec!["_input".to_string()],
    ));

    // Element-wise minimum of the two branches
    graph.add_node(GraphNode::binary(
        "min_out",
        Layer::MinBinary(MinBinaryLayer),
        "branch_a",
        "branch_b",
    ));

    graph.set_output("min_out");

    // Create bounded input
    let input = BoundedTensor::new(
        Array2::<f32>::from_elem((1, hidden), -1.0).into_dyn(),
        Array2::<f32>::from_elem((1, hidden), 1.0).into_dyn(),
    )
    .unwrap();

    // Run IBP propagation
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify bounds are finite and reasonable
    for i in 0..hidden {
        assert!(
            ibp_bounds.lower()[[0, i]].is_finite(),
            "IBP lower must be finite at position {}",
            i
        );
        assert!(
            ibp_bounds.upper()[[0, i]].is_finite(),
            "IBP upper must be finite at position {}",
            i
        );
        assert!(
            ibp_bounds.lower()[[0, i]] <= ibp_bounds.upper()[[0, i]],
            "IBP lower must be <= upper at position {}",
            i
        );
    }

    // Verify that min actually produces smaller values than max would
    // by constructing a max network and comparing
    let mut graph_max = GraphNetwork::new();
    let linear_a2 = LinearLayer::new(
        Array2::<f32>::from_elem((hidden, hidden), 0.5),
        Some(Array1::<f32>::zeros(hidden)),
    )
    .unwrap();
    graph_max.add_node(GraphNode::new(
        "branch_a",
        Layer::Linear(linear_a2),
        vec!["_input".to_string()],
    ));

    let linear_b2 = LinearLayer::new(
        Array2::<f32>::from_elem((hidden, hidden), 0.3),
        Some(Array1::<f32>::zeros(hidden)),
    )
    .unwrap();
    graph_max.add_node(GraphNode::new(
        "branch_b",
        Layer::Linear(linear_b2),
        vec!["_input".to_string()],
    ));

    graph_max.add_node(GraphNode::binary(
        "max_out",
        Layer::MaxBinary(MaxBinaryLayer),
        "branch_a",
        "branch_b",
    ));
    graph_max.set_output("max_out");

    let max_bounds = graph_max.propagate_ibp(&input).unwrap();

    // For these weights, min bounds should be <= max bounds
    for i in 0..hidden {
        assert!(
            ibp_bounds.lower()[[0, i]] <= max_bounds.upper()[[0, i]] + 1e-4,
            "min lower should be <= max upper at position {}",
            i
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_minmax_in_graph_network_crown_fallback() {
    // Test that CROWN propagation correctly falls back to IBP for MinBinary/MaxBinary.
    // CROWN doesn't have linear relaxation for Min/Max, so it should return IBP bounds.
    use ndarray::Array2;

    let mut graph = GraphNetwork::new();
    let hidden = 4;

    // Two linear branches from input
    let linear_a = LinearLayer::new(
        Array2::<f32>::from_elem((hidden, hidden), 0.5),
        Some(Array1::<f32>::zeros(hidden)),
    )
    .unwrap();
    graph.add_node(GraphNode::new(
        "branch_a",
        Layer::Linear(linear_a),
        vec!["_input".to_string()],
    ));

    let linear_b = LinearLayer::new(
        Array2::<f32>::from_elem((hidden, hidden), 0.3),
        Some(Array1::<f32>::zeros(hidden)),
    )
    .unwrap();
    graph.add_node(GraphNode::new(
        "branch_b",
        Layer::Linear(linear_b),
        vec!["_input".to_string()],
    ));

    // Element-wise minimum of the two branches
    graph.add_node(GraphNode::binary(
        "min_out",
        Layer::MinBinary(MinBinaryLayer),
        "branch_a",
        "branch_b",
    ));

    graph.set_output("min_out");

    // Create bounded input
    let input = BoundedTensor::new(
        Array2::<f32>::from_elem((1, hidden), -1.0).into_dyn(),
        Array2::<f32>::from_elem((1, hidden), 1.0).into_dyn(),
    )
    .unwrap();

    // Get IBP bounds for reference
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Run CROWN propagation - should fall back to IBP for MinBinary
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    // CROWN with unsupported ops falls back to IBP, so bounds should match
    for i in 0..hidden {
        assert!(
            crown_bounds.lower()[[0, i]].is_finite(),
            "CROWN lower must be finite at position {}",
            i
        );
        assert!(
            crown_bounds.upper()[[0, i]].is_finite(),
            "CROWN upper must be finite at position {}",
            i
        );
        assert!(
            crown_bounds.lower()[[0, i]] <= crown_bounds.upper()[[0, i]],
            "CROWN lower must be <= upper at position {}",
            i
        );
        // CROWN should produce the same bounds as IBP for MinBinary (fallback)
        assert!(
            (crown_bounds.lower()[[0, i]] - ibp_bounds.lower()[[0, i]]).abs() < 1e-4,
            "CROWN lower should match IBP lower at position {} (CROWN: {}, IBP: {})",
            i,
            crown_bounds.lower()[[0, i]],
            ibp_bounds.lower()[[0, i]]
        );
        assert!(
            (crown_bounds.upper()[[0, i]] - ibp_bounds.upper()[[0, i]]).abs() < 1e-4,
            "CROWN upper should match IBP upper at position {} (CROWN: {}, IBP: {})",
            i,
            crown_bounds.upper()[[0, i]],
            ibp_bounds.upper()[[0, i]]
        );
    }

    // Test MaxBinary as well
    let mut graph_max = GraphNetwork::new();
    let linear_a2 = LinearLayer::new(
        Array2::<f32>::from_elem((hidden, hidden), 0.5),
        Some(Array1::<f32>::zeros(hidden)),
    )
    .unwrap();
    graph_max.add_node(GraphNode::new(
        "branch_a",
        Layer::Linear(linear_a2),
        vec!["_input".to_string()],
    ));

    let linear_b2 = LinearLayer::new(
        Array2::<f32>::from_elem((hidden, hidden), 0.3),
        Some(Array1::<f32>::zeros(hidden)),
    )
    .unwrap();
    graph_max.add_node(GraphNode::new(
        "branch_b",
        Layer::Linear(linear_b2),
        vec!["_input".to_string()],
    ));

    graph_max.add_node(GraphNode::binary(
        "max_out",
        Layer::MaxBinary(MaxBinaryLayer),
        "branch_a",
        "branch_b",
    ));
    graph_max.set_output("max_out");

    let ibp_max_bounds = graph_max.propagate_ibp(&input).unwrap();
    let crown_max_bounds = graph_max.propagate_crown(&input).unwrap();

    // Verify MaxBinary CROWN fallback matches IBP
    for i in 0..hidden {
        assert!(
            (crown_max_bounds.lower()[[0, i]] - ibp_max_bounds.lower()[[0, i]]).abs() < 1e-4,
            "Max CROWN lower should match IBP lower at position {}",
            i
        );
        assert!(
            (crown_max_bounds.upper()[[0, i]] - ibp_max_bounds.upper()[[0, i]]).abs() < 1e-4,
            "Max CROWN upper should match IBP upper at position {}",
            i
        );
    }
}
