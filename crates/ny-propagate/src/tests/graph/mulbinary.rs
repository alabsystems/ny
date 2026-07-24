// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork MulBinary CROWN tests.
use crate::*;
use ndarray::{Array1, Array2};

/// Assert CROWN bounds are finite, ordered, and overlap with IBP at a given batch index.
/// CROWN is NOT guaranteed tighter than IBP per-element due to linearization error.
fn assert_crown_ibp_overlap(
    crown: &BoundedTensor,
    ibp: &BoundedTensor,
    batch_idx: &[usize],
    n: usize,
) {
    for i in 0..n {
        let idx: Vec<usize> = batch_idx
            .iter()
            .copied()
            .chain(std::iter::once(i))
            .collect();
        let cl = crown.lower()[idx.as_slice()];
        let cu = crown.upper()[idx.as_slice()];
        let il = ibp.lower()[idx.as_slice()];
        let iu = ibp.upper()[idx.as_slice()];
        assert!(cl.is_finite() && cu.is_finite(), "non-finite at {:?}", idx);
        assert!(cl <= cu + 1e-4, "lower > upper at {:?}: {cl}, {cu}", idx);
        assert!(cl <= iu + 1e-4, "crown_l > ibp_u at {:?}: {cl}, {iu}", idx);
        assert!(cu >= il - 1e-4, "crown_u < ibp_l at {:?}: {cu}, {il}", idx);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_mulbinary_mccormick_crown_soundness() {
    // Test that MulBinary McCormick CROWN produces sound bounds
    // z = x * y where x and y are bounded
    use ndarray::Array1;

    let mul = MulBinaryLayer;

    // Test case 1: Positive bounds [1, 2] * [3, 5] = [3, 10]
    let input_a = BoundedTensor::new(
        Array1::from_vec(vec![1.0_f32]).into_dyn(),
        Array1::from_vec(vec![2.0_f32]).into_dyn(),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        Array1::from_vec(vec![3.0_f32]).into_dyn(),
        Array1::from_vec(vec![5.0_f32]).into_dyn(),
    )
    .unwrap();

    // IBP bounds - exact for bilinear operations
    let ibp_result = mul.propagate_ibp_binary(&input_a, &input_b).unwrap();
    assert!(
        (ibp_result.lower()[0] - 3.0).abs() < 1e-5,
        "IBP lower should be 3.0"
    );
    assert!(
        (ibp_result.upper()[0] - 10.0).abs() < 1e-5,
        "IBP upper should be 10.0"
    );

    // CROWN with identity bounds (out = z)
    let identity = LinearBounds::identity(1);
    let (bounds_a, bounds_b) = mul
        .propagate_linear_binary(
            &identity,
            &input_a,
            &input_b,
            MulBinaryRelaxationMode::McCormick,
        )
        .unwrap();

    // For z = x * y with identity output, McCormick gives:
    // z_lower ≥ a_x * x + a_y * y + c (where coeffs depend on bounds)
    // To compute the final bound, we need to combine both contributions

    // For the lower bound:
    // - bounds_a.lower_a[0,0] is the coefficient for x
    // - bounds_b.lower_a[0,0] is the coefficient for y
    // - bounds_a.lower_b[0] = bounds_b.lower_b[0] is the constant c (not split for MulBinary)

    // Manually compute the concretized lower bound
    let a_x_lower = bounds_a.lower_a[[0, 0]];
    let a_y_lower = bounds_b.lower_a[[0, 0]];
    let c_lower = bounds_a.lower_b[0]; // Same as bounds_b.lower_b[0]

    // Concretize: use x_l if coeff >= 0, else x_u (for lower bound minimization)
    let x_contrib_lower = if a_x_lower >= 0.0 {
        a_x_lower * input_a.lower()[0]
    } else {
        a_x_lower * input_a.upper()[0]
    };
    let y_contrib_lower = if a_y_lower >= 0.0 {
        a_y_lower * input_b.lower()[0]
    } else {
        a_y_lower * input_b.upper()[0]
    };
    let crown_lower = x_contrib_lower + y_contrib_lower + c_lower;

    // Do the same for upper bound
    let a_x_upper = bounds_a.upper_a[[0, 0]];
    let a_y_upper = bounds_b.upper_a[[0, 0]];
    let c_upper = bounds_a.upper_b[0];

    // Concretize: use x_u if coeff >= 0, else x_l (for upper bound maximization)
    let x_contrib_upper = if a_x_upper >= 0.0 {
        a_x_upper * input_a.upper()[0]
    } else {
        a_x_upper * input_a.lower()[0]
    };
    let y_contrib_upper = if a_y_upper >= 0.0 {
        a_y_upper * input_b.upper()[0]
    } else {
        a_y_upper * input_b.lower()[0]
    };
    let crown_upper = x_contrib_upper + y_contrib_upper + c_upper;

    // McCormick bounds should be sound: contain the true range [3, 10]
    assert!(
        crown_lower <= 3.0 + 1e-4,
        "CROWN lower {} must be <= IBP min 3.0",
        crown_lower
    );
    assert!(
        crown_upper >= 10.0 - 1e-4,
        "CROWN upper {} must be >= IBP max 10.0",
        crown_upper
    );

    // Test case 2: Mixed signs [-1, 2] * [-3, 4] = [-6, 8]
    let input_a2 = BoundedTensor::new(
        Array1::from_vec(vec![-1.0_f32]).into_dyn(),
        Array1::from_vec(vec![2.0_f32]).into_dyn(),
    )
    .unwrap();
    let input_b2 = BoundedTensor::new(
        Array1::from_vec(vec![-3.0_f32]).into_dyn(),
        Array1::from_vec(vec![4.0_f32]).into_dyn(),
    )
    .unwrap();

    let ibp_result2 = mul.propagate_ibp_binary(&input_a2, &input_b2).unwrap();
    // True range: min of (-1)*(-3)=3, (-1)*4=-4, 2*(-3)=-6, 2*4=8 -> [-6, 8]
    assert!(
        (ibp_result2.lower()[0] - (-6.0)).abs() < 1e-5,
        "IBP lower should be -6.0"
    );
    assert!(
        (ibp_result2.upper()[0] - 8.0).abs() < 1e-5,
        "IBP upper should be 8.0"
    );

    let (bounds_a2, bounds_b2) = mul
        .propagate_linear_binary(
            &identity,
            &input_a2,
            &input_b2,
            MulBinaryRelaxationMode::McCormick,
        )
        .unwrap();

    // Manually compute concretized bounds for mixed signs case
    let a_x_lower2 = bounds_a2.lower_a[[0, 0]];
    let a_y_lower2 = bounds_b2.lower_a[[0, 0]];
    let c_lower2 = bounds_a2.lower_b[0];

    let x_contrib_lower2 = if a_x_lower2 >= 0.0 {
        a_x_lower2 * input_a2.lower()[0]
    } else {
        a_x_lower2 * input_a2.upper()[0]
    };
    let y_contrib_lower2 = if a_y_lower2 >= 0.0 {
        a_y_lower2 * input_b2.lower()[0]
    } else {
        a_y_lower2 * input_b2.upper()[0]
    };
    let crown_lower2 = x_contrib_lower2 + y_contrib_lower2 + c_lower2;

    let a_x_upper2 = bounds_a2.upper_a[[0, 0]];
    let a_y_upper2 = bounds_b2.upper_a[[0, 0]];
    let c_upper2 = bounds_a2.upper_b[0];

    let x_contrib_upper2 = if a_x_upper2 >= 0.0 {
        a_x_upper2 * input_a2.upper()[0]
    } else {
        a_x_upper2 * input_a2.lower()[0]
    };
    let y_contrib_upper2 = if a_y_upper2 >= 0.0 {
        a_y_upper2 * input_b2.upper()[0]
    } else {
        a_y_upper2 * input_b2.lower()[0]
    };
    let crown_upper2 = x_contrib_upper2 + y_contrib_upper2 + c_upper2;

    // McCormick must be sound (may be looser than IBP for mixed signs)
    assert!(
        crown_lower2 <= -6.0 + 1e-4,
        "CROWN lower {} must be sound for mixed signs",
        crown_lower2
    );
    assert!(
        crown_upper2 >= 8.0 - 1e-4,
        "CROWN upper {} must be sound for mixed signs",
        crown_upper2
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mulbinary_crown_in_graph_network() {
    // Test MulBinary with McCormick CROWN in a graph network (SwiGLU pattern)
    // SwiGLU: up(x) * silu(gate(x)) where silu ≈ x * sigmoid(x)

    let mut graph = GraphNetwork::new();

    // Two branches from input that will be multiplied
    let hidden = 4;

    // Linear for "up" branch
    let up_weights = Array2::<f32>::from_elem((hidden, hidden), 0.5);
    let up_bias = Array1::<f32>::zeros(hidden);
    let up_linear = LinearLayer::new(up_weights, Some(up_bias)).unwrap();
    graph.add_node(GraphNode::new(
        "up",
        Layer::Linear(up_linear),
        vec!["_input".to_string()],
    ));

    // Linear for "gate" branch
    let gate_weights = Array2::<f32>::from_elem((hidden, hidden), 0.3);
    let gate_bias = Array1::<f32>::zeros(hidden);
    let gate_linear = LinearLayer::new(gate_weights, Some(gate_bias)).unwrap();
    graph.add_node(GraphNode::new(
        "gate",
        Layer::Linear(gate_linear),
        vec!["_input".to_string()],
    ));

    // Apply sigmoid to gate (approximates silu when combined with gate*sigmoid(gate))
    graph.add_node(GraphNode::new(
        "gate_sigmoid",
        Layer::Sigmoid(SigmoidLayer),
        vec!["gate".to_string()],
    ));

    // Element-wise multiplication (the SwiGLU gating)
    graph.add_node(GraphNode::binary(
        "swiglu_mul",
        Layer::MulBinary(MulBinaryLayer),
        "up",
        "gate_sigmoid",
    ));

    graph.set_output("swiglu_mul");

    // Create bounded input
    let input = BoundedTensor::new(
        Array2::<f32>::from_elem((1, hidden), -0.5).into_dyn(),
        Array2::<f32>::from_elem((1, hidden), 0.5).into_dyn(),
    )
    .unwrap();

    // Run CROWN propagation (should use McCormick for MulBinary)
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    // Run IBP for comparison
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    assert_crown_ibp_overlap(&crown_bounds, &ibp_bounds, &[0], hidden);
}

#[ntest::timeout(10000)]
#[test]
fn test_mulbinary_batched_crown_in_graph_network() {
    // Test MulBinary with McCormick CROWN in batched mode (SwiGLU pattern)
    // This tests the propagate_linear_batched_binary method

    let mut graph = GraphNetwork::new();

    // Two branches from input that will be multiplied
    let hidden = 4;

    // Linear for "up" branch
    let up_weights = Array2::<f32>::from_elem((hidden, hidden), 0.5);
    let up_bias = Array1::<f32>::zeros(hidden);
    let up_linear = LinearLayer::new(up_weights, Some(up_bias)).unwrap();
    graph.add_node(GraphNode::new(
        "up",
        Layer::Linear(up_linear),
        vec!["_input".to_string()],
    ));

    // Linear for "gate" branch
    let gate_weights = Array2::<f32>::from_elem((hidden, hidden), 0.3);
    let gate_bias = Array1::<f32>::zeros(hidden);
    let gate_linear = LinearLayer::new(gate_weights, Some(gate_bias)).unwrap();
    graph.add_node(GraphNode::new(
        "gate",
        Layer::Linear(gate_linear),
        vec!["_input".to_string()],
    ));

    // Apply sigmoid to gate
    graph.add_node(GraphNode::new(
        "gate_sigmoid",
        Layer::Sigmoid(SigmoidLayer),
        vec!["gate".to_string()],
    ));

    // Element-wise multiplication (the SwiGLU gating)
    graph.add_node(GraphNode::binary(
        "swiglu_mul",
        Layer::MulBinary(MulBinaryLayer),
        "up",
        "gate_sigmoid",
    ));

    graph.set_output("swiglu_mul");

    // Create bounded input with batch dimension
    let batch = 2;
    let input = BoundedTensor::new(
        Array2::<f32>::from_elem((batch, hidden), -0.5).into_dyn(),
        Array2::<f32>::from_elem((batch, hidden), 0.5).into_dyn(),
    )
    .unwrap();

    // Run batched CROWN propagation
    let crown_bounds = graph.propagate_crown_batched(&input).unwrap();

    // Run IBP for comparison
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    for b in 0..batch {
        assert_crown_ibp_overlap(&crown_bounds, &ibp_bounds, &[b], hidden);
    }
}

// =========================================================================
// Middle Relaxation Tests
// =========================================================================

/// Test middle relaxation mode matches auto_LiRPA's `mul.middle` formulas.
///
/// Reference: auto_LiRPA/operators/bivariate.py:MulHelper.interpolated_relaxation
/// With interpolation parameter r=0.5:
///   alpha_l = (y_l - y_u) * 0.5 + y_u
///   beta_l  = (x_l - x_u) * 0.5 + x_u
///   ny_l = (y_u * x_u - y_l * x_l) * 0.5 - y_u * x_u
///   alpha_u = (y_u - y_l) * 0.5 + y_l
///   beta_u  = (x_l - x_u) * 0.5 + x_u
///   ny_u = (y_l * x_u - y_u * x_l) * 0.5 - y_l * x_u
#[ntest::timeout(10000)]
#[test]
fn test_mulbinary_middle_relaxation_coefficients() {
    let mul = MulBinaryLayer;

    // Test case: x in [1, 3], y in [2, 5]
    let x_l = 1.0_f32;
    let x_u = 3.0_f32;
    let y_l = 2.0_f32;
    let y_u = 5.0_f32;

    let input_a = BoundedTensor::new(
        Array1::from_vec(vec![x_l]).into_dyn(),
        Array1::from_vec(vec![x_u]).into_dyn(),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        Array1::from_vec(vec![y_l]).into_dyn(),
        Array1::from_vec(vec![y_u]).into_dyn(),
    )
    .unwrap();

    // Identity output bounds
    let identity = LinearBounds::identity(1);

    // Compute expected middle coefficients (auto_LiRPA formulas with r=0.5)
    let expected_alpha_l = (y_l - y_u) * 0.5 + y_u; // y_mid = 3.5
    let expected_beta_l = (x_l - x_u) * 0.5 + x_u; // x_mid = 2.0
    let expected_ny_l = (y_u * x_u - y_l * x_l) * 0.5 - y_u * x_u; // (15 - 2) * 0.5 - 15 = -8.5

    let expected_alpha_u = (y_u - y_l) * 0.5 + y_l; // y_mid = 3.5
    let expected_beta_u = (x_l - x_u) * 0.5 + x_u; // x_mid = 2.0
    let expected_ny_u = (y_l * x_u - y_u * x_l) * 0.5 - y_l * x_u; // (6 - 5) * 0.5 - 6 = -5.5

    // Get middle relaxation coefficients
    let (bounds_a, bounds_b) = mul
        .propagate_linear_binary(
            &identity,
            &input_a,
            &input_b,
            MulBinaryRelaxationMode::Middle,
        )
        .unwrap();

    // With identity bounds and positive weights (w=1), the middle relaxation should use:
    // For lower bound: alpha_l * x + beta_l * y + ny_l (w >= 0 case)
    // For upper bound: alpha_u * x + beta_u * y + ny_u (w >= 0 case)
    //
    // bounds_a.lower_a[0,0] is the coefficient for x (alpha_l)
    // bounds_b.lower_a[0,0] is the coefficient for y (beta_l)
    // bounds_a.lower_b[0] contains the constant ny_l

    let actual_alpha_l = bounds_a.lower_a[[0, 0]]; // coeff for x in lower bound
    let actual_beta_l = bounds_b.lower_a[[0, 0]]; // coeff for y in lower bound
    let actual_ny_l = bounds_a.lower_b[0]; // constant for lower bound

    let actual_alpha_u = bounds_a.upper_a[[0, 0]]; // coeff for x in upper bound
    let actual_beta_u = bounds_b.upper_a[[0, 0]]; // coeff for y in upper bound
    let actual_ny_u = bounds_a.upper_b[0]; // constant for upper bound

    // Verify lower bound coefficients match auto_LiRPA formulas
    assert!(
        (actual_alpha_l - expected_alpha_l).abs() < 1e-5,
        "alpha_l mismatch: actual={}, expected={}",
        actual_alpha_l,
        expected_alpha_l
    );
    assert!(
        (actual_beta_l - expected_beta_l).abs() < 1e-5,
        "beta_l mismatch: actual={}, expected={}",
        actual_beta_l,
        expected_beta_l
    );
    assert!(
        (actual_ny_l - expected_ny_l).abs() < 1e-5,
        "ny_l mismatch: actual={}, expected={}",
        actual_ny_l,
        expected_ny_l
    );

    // Verify upper bound coefficients match auto_LiRPA formulas
    assert!(
        (actual_alpha_u - expected_alpha_u).abs() < 1e-5,
        "alpha_u mismatch: actual={}, expected={}",
        actual_alpha_u,
        expected_alpha_u
    );
    assert!(
        (actual_beta_u - expected_beta_u).abs() < 1e-5,
        "beta_u mismatch: actual={}, expected={}",
        actual_beta_u,
        expected_beta_u
    );
    assert!(
        (actual_ny_u - expected_ny_u).abs() < 1e-5,
        "ny_u mismatch: actual={}, expected={}",
        actual_ny_u,
        expected_ny_u
    );

    // Verify soundness: computed bounds should contain the true range
    // True range: x*y for x in [1,3], y in [2,5] is [2, 15]
    let true_min = 2.0_f32;
    let true_max = 15.0_f32;

    // Concretize lower bound: min over x,y of alpha_l*x + beta_l*y + ny_l
    let lower_bound = actual_alpha_l * x_l + actual_beta_l * y_l + actual_ny_l;
    assert!(
        lower_bound <= true_min + 1e-4,
        "Middle lower bound {} must be <= true min {}",
        lower_bound,
        true_min
    );

    // Concretize upper bound: max over x,y of alpha_u*x + beta_u*y + ny_u
    let upper_bound = actual_alpha_u * x_u + actual_beta_u * y_u + actual_ny_u;
    assert!(
        upper_bound >= true_max - 1e-4,
        "Middle upper bound {} must be >= true max {}",
        upper_bound,
        true_max
    );
}

/// Test that default McCormick mode remains unchanged when config is not set to Middle.
#[ntest::timeout(10000)]
#[test]
fn test_mulbinary_default_is_mccormick() {
    let mul = MulBinaryLayer;

    // Same test case as above: x in [1, 3], y in [2, 5]
    let input_a = BoundedTensor::new(
        Array1::from_vec(vec![1.0_f32]).into_dyn(),
        Array1::from_vec(vec![3.0_f32]).into_dyn(),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        Array1::from_vec(vec![2.0_f32]).into_dyn(),
        Array1::from_vec(vec![5.0_f32]).into_dyn(),
    )
    .unwrap();

    let identity = LinearBounds::identity(1);

    // Get McCormick coefficients (default)
    let (mccormick_a, mccormick_b) = mul
        .propagate_linear_binary(
            &identity,
            &input_a,
            &input_b,
            MulBinaryRelaxationMode::McCormick,
        )
        .unwrap();

    // Get Middle coefficients
    let (middle_a, _middle_b) = mul
        .propagate_linear_binary(
            &identity,
            &input_a,
            &input_b,
            MulBinaryRelaxationMode::Middle,
        )
        .unwrap();

    // McCormick and Middle should produce different coefficients
    // (McCormick selects planes, Middle uses fixed interpolation)
    let mccormick_lower_coeff = mccormick_a.lower_a[[0, 0]];
    let middle_lower_coeff = middle_a.lower_a[[0, 0]];

    // For positive inputs, McCormick typically chooses different plane than middle
    // The coefficients shouldn't match exactly
    let coeffs_differ = (mccormick_lower_coeff - middle_lower_coeff).abs() > 1e-6
        || (mccormick_a.lower_b[0] - middle_a.lower_b[0]).abs() > 1e-6;

    assert!(
        coeffs_differ,
        "McCormick and Middle should produce different coefficients"
    );

    // Both should be sound (contain true range [2, 15])
    let true_min = 2.0_f32;
    let true_max = 15.0_f32;

    // Concretize McCormick bounds
    let mc_coeff_x = mccormick_a.lower_a[[0, 0]];
    let mc_coeff_y = mccormick_b.lower_a[[0, 0]];
    let mc_const = mccormick_a.lower_b[0];
    let mc_lower = mc_coeff_x * 1.0 + mc_coeff_y * 2.0 + mc_const;

    assert!(
        mc_lower <= true_min + 1e-4,
        "McCormick lower bound {} must be <= true min {}",
        mc_lower,
        true_min
    );

    let mc_coeff_x_u = mccormick_a.upper_a[[0, 0]];
    let mc_coeff_y_u = mccormick_b.upper_a[[0, 0]];
    let mc_const_u = mccormick_a.upper_b[0];
    let mc_upper = mc_coeff_x_u * 3.0 + mc_coeff_y_u * 5.0 + mc_const_u;

    assert!(
        mc_upper >= true_max - 1e-4,
        "McCormick upper bound {} must be >= true max {}",
        mc_upper,
        true_max
    );
}

// =========================================================================
// IBP interval multiplication regression tests (#1816)
// =========================================================================

/// Verify IBP multiplication matches BoundedTensor::mul for same-shape inputs.
/// This confirms the deduplication: propagate_ibp_binary delegates to BoundedTensor::mul.
///
/// Reference: auto_LiRPA/operators/bivariate.py:419-421
#[ntest::timeout(10000)]
#[test]
fn test_mulbinary_ibp_matches_bounded_tensor_mul() {
    let mul = MulBinaryLayer;

    // Mixed signs: [-2, 3] * [-1, 4]
    let a = BoundedTensor::new(
        Array1::from_vec(vec![-2.0_f32, 0.0, 1.0]).into_dyn(),
        Array1::from_vec(vec![3.0_f32, 5.0, 2.0]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        Array1::from_vec(vec![-1.0_f32, -3.0, 4.0]).into_dyn(),
        Array1::from_vec(vec![4.0_f32, 2.0, 6.0]).into_dyn(),
    )
    .unwrap();

    let ibp_result = mul.propagate_ibp_binary(&a, &b).unwrap();
    let direct_result = a.mul(&b).unwrap();

    for i in 0..3 {
        assert_eq!(
            ibp_result.lower()[i],
            direct_result.lower()[i],
            "Lower mismatch at {}: ibp={}, direct={}",
            i,
            ibp_result.lower()[i],
            direct_result.lower()[i]
        );
        assert_eq!(
            ibp_result.upper()[i],
            direct_result.upper()[i],
            "Upper mismatch at {}: ibp={}, direct={}",
            i,
            ibp_result.upper()[i],
            direct_result.upper()[i]
        );
    }
}

/// Verify IBP multiplication with broadcasting produces correct bounds.
/// E.g., [1, 3, 1] * [1, 1, 4] should broadcast to [1, 3, 4].
#[ntest::timeout(10000)]
#[test]
fn test_mulbinary_ibp_broadcasting() {
    use ndarray::ArrayD;

    let mul = MulBinaryLayer;

    // a: shape [3], b: shape [1] → broadcast to [3]
    let a = BoundedTensor::new(
        Array1::from_vec(vec![1.0_f32, -2.0, 3.0]).into_dyn(),
        Array1::from_vec(vec![2.0_f32, -1.0, 4.0]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(vec![1], vec![2.0_f32]).unwrap(),
        ArrayD::from_shape_vec(vec![1], vec![3.0_f32]).unwrap(),
    )
    .unwrap();

    let result = mul.propagate_ibp_binary(&a, &b).unwrap();
    assert_eq!(result.shape(), &[3]);

    // [1,2]*[2,3] = [2,6]
    assert!((result.lower()[0] - 2.0).abs() < 1e-5);
    assert!((result.upper()[0] - 6.0).abs() < 1e-5);

    // [-2,-1]*[2,3] = [-6,-2]
    assert!((result.lower()[1] - (-6.0)).abs() < 1e-5);
    assert!((result.upper()[1] - (-2.0)).abs() < 1e-5);

    // [3,4]*[2,3] = [6,12]
    assert!((result.lower()[2] - 6.0).abs() < 1e-5);
    assert!((result.upper()[2] - 12.0).abs() < 1e-5);
}

/// Corner products with negative × negative should produce positive upper bounds.
/// Regression: ensures min/max over all 4 corner products is correct.
#[ntest::timeout(10000)]
#[test]
fn test_mulbinary_ibp_all_negative() {
    let mul = MulBinaryLayer;

    // [-4, -1] * [-3, -2] = [2, 12]
    let a = BoundedTensor::new(
        Array1::from_vec(vec![-4.0_f32]).into_dyn(),
        Array1::from_vec(vec![-1.0_f32]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        Array1::from_vec(vec![-3.0_f32]).into_dyn(),
        Array1::from_vec(vec![-2.0_f32]).into_dyn(),
    )
    .unwrap();

    let result = mul.propagate_ibp_binary(&a, &b).unwrap();
    // Corner products: (-4)*(-3)=12, (-4)*(-2)=8, (-1)*(-3)=3, (-1)*(-2)=2
    assert!(
        (result.lower()[0] - 2.0).abs() < 1e-5,
        "lower={}",
        result.lower()[0]
    );
    assert!(
        (result.upper()[0] - 12.0).abs() < 1e-5,
        "upper={}",
        result.upper()[0]
    );
}

/// Infinite bounds must not be clamped away in MulBinary IBP.
/// Regression for #1816: old implementation clamped infinities to finite values.
#[ntest::timeout(10000)]
#[test]
fn test_mulbinary_ibp_infinite_bounds_match_shared_kernel() {
    let mul = MulBinaryLayer;

    // Use unchecked constructor to exercise overflow/unbounded intermediate behavior.
    let a = BoundedTensor::new_unchecked(
        Array1::from_vec(vec![0.0_f32, -2.0, 1.0]).into_dyn(),
        Array1::from_vec(vec![0.0_f32, -1.0, 2.0]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new_unchecked(
        Array1::from_vec(vec![f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY]).into_dyn(),
        Array1::from_vec(vec![f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY]).into_dyn(),
    )
    .unwrap();

    let ibp_result = mul.propagate_ibp_binary(&a, &b).unwrap();
    let direct_result = a.mul(&b).unwrap();

    for i in 0..3 {
        let ibp_l = ibp_result.lower()[i];
        let direct_l = direct_result.lower()[i];
        assert_eq!(
            ibp_l, direct_l,
            "Lower mismatch at {}: ibp={}, direct={}",
            i, ibp_l, direct_l
        );

        let ibp_u = ibp_result.upper()[i];
        let direct_u = direct_result.upper()[i];
        assert_eq!(
            ibp_u, direct_u,
            "Upper mismatch at {}: ibp={}, direct={}",
            i, ibp_u, direct_u
        );
    }

    // [0,0] * [+inf,+inf] yields NaN corner products; use conservative widening.
    assert_eq!(ibp_result.lower()[0], f32::NEG_INFINITY);
    assert_eq!(ibp_result.upper()[0], f32::INFINITY);
    // Negative finite interval times +inf should stay -inf (no finite clamping).
    assert_eq!(ibp_result.lower()[1], f32::NEG_INFINITY);
    assert_eq!(ibp_result.upper()[1], f32::NEG_INFINITY);
}

/// Broadcasting path must preserve infinite bounds (no MAX_BOUND clamping).
#[ntest::timeout(10000)]
#[test]
fn test_mulbinary_ibp_broadcasting_with_infinite_scalar() {
    use ndarray::ArrayD;

    let mul = MulBinaryLayer;

    let a = BoundedTensor::new(
        Array1::from_vec(vec![1.0_f32, -3.0]).into_dyn(),
        Array1::from_vec(vec![2.0_f32, -1.0]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(vec![1], vec![f32::INFINITY]).unwrap(),
        ArrayD::from_shape_vec(vec![1], vec![f32::INFINITY]).unwrap(),
    )
    .unwrap();

    let result = mul.propagate_ibp_binary(&a, &b).unwrap();
    assert_eq!(result.shape(), &[2]);

    // Positive finite interval * +inf -> +inf
    assert_eq!(result.lower()[0], f32::INFINITY);
    assert_eq!(result.upper()[0], f32::INFINITY);
    // Negative finite interval * +inf -> -inf
    assert_eq!(result.lower()[1], f32::NEG_INFINITY);
    assert_eq!(result.upper()[1], f32::NEG_INFINITY);
}

// =========================================================================
// Domain Clipping Integration Tests
// =========================================================================
