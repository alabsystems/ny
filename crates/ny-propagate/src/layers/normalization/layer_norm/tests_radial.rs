// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use num_rational::BigRational;
use ny_tensor::BoundedTensor;

use super::LayerNormLayer;
use crate::certify::work::WorkBudget;
use crate::{LinearBounds, NyError};

fn boxed(shape: &[usize], lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

fn direction(c: &[f32], bias: f32) -> LinearBounds {
    let a = Array2::from_shape_vec((1, c.len()), c.to_vec()).unwrap();
    LinearBounds::new(a.clone(), arr1(&[bias]), a, arr1(&[bias])).unwrap()
}

fn at(bounds: &LinearBounds, x: &[f32]) -> (f64, f64) {
    let lower = bounds.lower_b()[0] as f64
        + bounds
            .lower_a()
            .row(0)
            .iter()
            .zip(x)
            .map(|(&a, &x)| a as f64 * x as f64)
            .sum::<f64>();
    let upper = bounds.upper_b()[0] as f64
        + bounds
            .upper_a()
            .row(0)
            .iter()
            .zip(x)
            .map(|(&a, &x)| a as f64 * x as f64)
            .sum::<f64>();
    (lower, upper)
}

fn rational(x: f32) -> BigRational {
    BigRational::from_float(x as f64).unwrap()
}

#[test]
fn default_preserves_exact_layernorm_sum_invariant_across_wide_boxes() {
    let layer = LayerNormLayer::new_default(3, 1e-5).unwrap();
    let pre = boxed(&[3], &[-10.0, -2.0, -3.0], &[7.0, 11.0, 4.0]);
    let result = layer
        .propagate_linear_with_bounds(&direction(&[1.0; 3], 0.25), &pre)
        .unwrap();
    assert!(result
        .lower_a()
        .iter()
        .chain(result.upper_a().iter())
        .all(|&a| a == 0.0));
    assert_eq!(result.lower_b()[0], 0.25);
    assert_eq!(result.upper_b()[0], 0.25);
}

#[test]
fn transformed_uniform_direction_cancels_signed_gamma_and_beta_exactly() {
    let layer =
        LayerNormLayer::new(arr1(&[2.0, -1.0, 0.5]), arr1(&[0.125, 0.25, -0.5]), 1e-5).unwrap();
    let pre = boxed(&[3], &[-1e10, -2e10, -3e10], &[4e10, 5e10, 6e10]);
    let result = layer
        .propagate_radial_linear_with_bounds(&direction(&[0.5, -1.0, 2.0], 0.25), &pre, None)
        .unwrap();
    let truth = rational(0.25) + rational(0.5) * rational(0.125) - rational(0.25)
        + rational(2.0) * rational(-0.5);
    assert!(rational(result.lower_b()[0]) <= truth && truth <= rational(result.upper_b()[0]));
    assert!(result
        .lower_a()
        .iter()
        .chain(result.upper_a().iter())
        .all(|&a| a == 0.0));
    assert!(result.upper_b()[0] - result.lower_b()[0] < 1e-6);
}

#[test]
fn point_cast_and_center_accounting_enclose_exact_rational_norm() {
    // mean=0, variance=1, epsilon=3 => denominator exactly 2.
    let layer = LayerNormLayer::new(arr1(&[2.0, -1.0]), arr1(&[0.125, -0.25]), 3.0).unwrap();
    let pre = boxed(&[2], &[-1.0, 1.0], &[-1.0, 1.0]);
    let c = [1e20_f32, -1e20_f32];
    let result = layer
        .propagate_radial_linear_with_bounds(&direction(&c, 0.0), &pre, None)
        .unwrap();
    let truth = rational(c[0]) * (rational(-1.0) + rational(0.125))
        + rational(c[1]) * (rational(-0.5) + rational(-0.25));
    let evaluate = |a: &Array2<f32>, b: &Array1<f32>| {
        rational(b[0]) + rational(a[[0, 0]]) * rational(-1.0) + rational(a[[0, 1]]) * rational(1.0)
    };
    assert!(evaluate(result.lower_a(), result.lower_b()) <= truth);
    assert!(truth <= evaluate(result.upper_a(), result.upper_b()));
}

#[test]
fn radial_encloses_every_vertex_and_interior_for_signed_directions() {
    let gamma = [2.0_f32, -0.75, 0.5];
    let beta = [0.125_f32, -0.25, 0.5];
    let c = [-1.5_f32, 0.25, 2.0];
    let layer = LayerNormLayer::new(arr1(&gamma), arr1(&beta), 0.125).unwrap();
    let lower = [-1.2_f32, 0.3, 1.5];
    let upper = [-0.8_f32, 0.5, 1.9];
    let pre = boxed(&[3], &lower, &upper);
    let result = layer
        .propagate_radial_linear_with_bounds(&direction(&c, -0.25), &pre, None)
        .unwrap();
    for i in 0..125 {
        let x = std::array::from_fn::<_, 3, _>(|j| {
            let t = ((i / 5usize.pow(j as u32)) % 5) as f32 / 4.0;
            lower[j] * (1.0 - t) + upper[j] * t
        });
        let mean = x.iter().map(|&x| x as f64).sum::<f64>() / 3.0;
        let var = x.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / 3.0;
        let denominator = (var + layer.eps as f64).sqrt();
        let truth = -0.25
            + (0..3)
                .map(|j| {
                    c[j] as f64
                        * (gamma[j] as f64 * (x[j] as f64 - mean) / denominator + beta[j] as f64)
                })
                .sum::<f64>();
        let (lo, hi) = at(&result, &x);
        assert!(
            lo <= truth && truth <= hi,
            "{x:?}: [{lo},{hi}] excludes {truth}"
        );
    }
}

#[test]
fn dimension_one_and_tiny_epsilon_variance_crossing_remain_finite() {
    let one = LayerNormLayer::new(arr1(&[-2.0]), arr1(&[0.25]), 1e-12).unwrap();
    let result = one
        .propagate_radial_linear_with_bounds(
            &direction(&[3.0], 0.5),
            &boxed(&[1], &[-1e20], &[1e20]),
            None,
        )
        .unwrap();
    assert_eq!(result.lower_a()[[0, 0]], 0.0);
    assert!(result.lower_b()[0] <= 1.25 && result.upper_b()[0] >= 1.25);
    assert!(result.upper_b()[0] - result.lower_b()[0] < 5e-7);
    let two = LayerNormLayer::new_default(2, 1e-12).unwrap();
    let result = two
        .propagate_radial_linear_with_bounds(
            &direction(&[1.0, -1.0], 0.0),
            &boxed(&[2], &[-1e-6; 2], &[1e-6; 2]),
            None,
        )
        .unwrap();
    let (lo, hi) = at(&result, &[-1e-6, 1e-6]);
    let truth = -2e-6_f64 / (1e-12_f64 + two.eps as f64).sqrt();
    assert!(lo <= truth && truth <= hi);
    assert!(lo.is_finite() && hi.is_finite());
}

#[test]
fn original_coefficient_uncertainty_survives_scalar_and_multi_group_bridges() {
    let layer = LayerNormLayer::new_default(2, 3.0).unwrap();
    for groups in [1usize, 2] {
        let n = groups * 2;
        let mut bounds = direction(&vec![1.0; n], 0.0);
        bounds.lower_a_err = Some(Array2::from_elem((1, n), 0.25));
        bounds.upper_a_err = bounds.lower_a_err.clone();
        let x: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let shape = if groups == 1 {
            vec![2]
        } else {
            vec![groups, 2]
        };
        let pre = boxed(&shape, &x, &x);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
        let (lo, hi) = at(&result, &x);
        // Actual normalized coordinates [-.5,.5]. Permitted independent
        // coefficient errors attain +/- .25 per normalization group.
        assert!(
            lo <= -0.25 * groups as f64 && hi >= 0.25 * groups as f64,
            "group={groups}: [{lo},{hi}]"
        );
    }
}

#[test]
fn radial_budget_refuses_before_allocating_and_completes_a_shared_lease() {
    use std::time::{Duration, Instant};
    let layer = LayerNormLayer::new_default(3, 0.125).unwrap();
    let pre = boxed(&[3], &[-1.0, 0.0, 1.0], &[-0.9, 0.1, 1.1]);
    let bounds = direction(&[1.0, -1.0, 0.5], 0.0);
    let expired = WorkBudget::new(
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
        1 << 20,
        0,
        u64::MAX,
        10,
        Duration::ZERO,
    )
    .unwrap();
    assert!(matches!(
        layer.propagate_radial_linear_with_bounds(&bounds, &pre, Some(&expired)),
        Err(NyError::DeadlineExceeded(_))
    ));
    for budget in [
        WorkBudget::new(None, 1, 0, u64::MAX, 10, Duration::ZERO).unwrap(),
        WorkBudget::new(None, 1 << 20, 0, 0, 10, Duration::ZERO).unwrap(),
    ] {
        assert!(layer
            .propagate_radial_linear_with_bounds(&bounds, &pre, Some(&budget))
            .is_err());
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(budget.receipt().unwrap().admitted, 0);
    }
    let budget = WorkBudget::new(None, 1 << 20, 0, u64::MAX, 10, Duration::ZERO).unwrap();
    let _ = layer
        .propagate_radial_linear_with_bounds(&bounds, &pre, Some(&budget))
        .unwrap();
    let receipt = budget.receipt().unwrap();
    // Scalar bridge, mandatory demand inspection, and active refinement all
    // complete on the original ledger; inspection spends no proposal.
    assert_eq!(receipt.admitted, 3);
    assert_eq!(receipt.completed, receipt.admitted);
    assert_eq!(receipt.proposals_reserved, 1);
    assert!(receipt.work_reserved > bounds.lower_a.len() as u64);
    assert_eq!(receipt.live_scratch_bytes, 0);
    assert_eq!(receipt.live_persistent_bytes, 0);
    assert!(receipt.peak_leased_bytes < 1 << 20);
}

#[test]
fn direct_reference_contract_refuses_nonfinite_and_forward_mode() {
    let layer = LayerNormLayer::new_default(2, 1e-5).unwrap();
    let unbounded = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY, 0.0]).into_dyn(),
        arr1(&[f32::INFINITY, 1.0]).into_dyn(),
    )
    .unwrap();
    assert!(layer
        .propagate_radial_linear_with_bounds(&direction(&[1.0, 0.0], 0.0), &unbounded, None)
        .is_err());
    let forward = layer.with_forward_mode(true);
    assert!(forward
        .propagate_radial_linear_with_bounds(
            &direction(&[1.0, 0.0], 0.0),
            &boxed(&[2], &[0.0; 2], &[1.0; 2]),
            None
        )
        .is_err());
}

#[test]
fn default_norm_causal_attention_residual_norm_graph_retains_the_exact_margin() {
    use crate::layers::{AddLayer, AttentionMask, LinearLayer, ReshapeLayer, SelfAttentionLayer};
    use crate::network::{GraphNetwork, GraphNetworkCrownExt, GraphNode};
    use crate::{Layer, MulBinaryRelaxationMode};
    let _env_lock = ny_test_utils::env::lock_env();
    let _gate = ny_test_utils::env::ScopedEnvVar::set("NY_ATTN_CROWN_TERNARY", "1");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "tokens",
        Layer::Reshape(ReshapeLayer::new(vec![2, 3])),
    ));
    graph.add_node(GraphNode::new(
        "pre_norm",
        Layer::LayerNorm(LayerNormLayer::new_default(3, 0.125).unwrap()),
        vec!["tokens".into()],
    ));
    graph.add_node(GraphNode::new(
        "causal_attn",
        Layer::SelfAttention(SelfAttentionLayer::new(AttentionMask::Causal, None)),
        vec!["pre_norm".into(), "pre_norm".into(), "pre_norm".into()],
    ));
    graph.add_node(GraphNode::new(
        "residual",
        Layer::Add(AddLayer),
        vec!["pre_norm".into(), "causal_attn".into()],
    ));
    graph.add_node(GraphNode::new(
        "post_norm",
        Layer::LayerNorm(LayerNormLayer::new_default(3, 0.125).unwrap()),
        vec!["residual".into()],
    ));
    graph.add_node(GraphNode::new(
        "flat",
        Layer::Reshape(ReshapeLayer::new(vec![6])),
        vec!["post_norm".into()],
    ));
    graph.add_node(GraphNode::new(
        "margin",
        Layer::Linear(LinearLayer::new(Array2::ones((1, 6)), Some(arr1(&[0.125]))).unwrap()),
        vec!["flat".into()],
    ));
    graph.set_output("margin");
    let input = boxed(
        &[6],
        &[-0.3, 0.4, 1.1, -0.8, 0.2, 0.7],
        &[-0.2, 0.5, 1.2, -0.7, 0.3, 0.8],
    );
    let intervals = graph.collect_node_bounds(&input).unwrap();
    let at_norm = direction(&[1.0; 6], 0.125);
    let norm = LayerNormLayer::new_default(3, 0.125)
        .unwrap()
        .propagate_linear_with_bounds(&at_norm, intervals.get("residual").unwrap())
        .unwrap();
    assert!(norm
        .lower_a()
        .iter()
        .chain(norm.upper_a().iter())
        .all(|&a| a == 0.0));
    let result = GraphNetworkCrownExt::crown_backward_with_relaxation(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
    )
    .unwrap();
    let lower = result.lower().iter().next().copied().unwrap();
    let upper = result.upper().iter().next().copied().unwrap();
    assert!(lower <= 0.125 && upper >= 0.125);
    assert!(
        lower > 0.1249 && upper < 0.1251,
        "checked invariant lost in decoder-shaped graph: [{lower},{upper}]"
    );
}
