// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::domain_list::{DomainMetadata, PickedDomains};
use ndarray::{ArrayD, IxDyn};
use ny_core::NyError;
use std::collections::HashMap;

fn metadata(lower_bound: f32, upper_bound: f32, depth: usize) -> DomainMetadata {
    DomainMetadata {
        lower_bound,
        upper_bound,
        depth,
        constraints: vec![],
        cached_la: None,
        needs_bounding: false,
        node_bounds_override: None,
        alpha_state: None,
    }
}

fn make_non_contiguous_batch_bounds_2x4(values: [[f32; 4]; 2]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(
        IxDyn(&[4, 2]),
        vec![
            values[0][0],
            values[1][0],
            values[0][1],
            values[1][1],
            values[0][2],
            values[1][2],
            values[0][3],
            values[1][3],
        ],
    )
    .expect("shape should be valid")
    .view()
    .reversed_axes()
    .to_owned()
}

// ============================================================================
// Tests for batched unstable neuron detection and branch selection
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn test_picked_domains_find_unstable_neurons_batched_empty() {
    let picked = PickedDomains {
        batch_size: 0,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::zeros(IxDyn(&[0])),
        input_uppers: ArrayD::zeros(IxDyn(&[0])),
        global_lbs: Vec::new(),
        global_ubs: Vec::new(),
        metadata: Vec::new(),
    };

    let relu_pre_map: HashMap<String, String> = HashMap::new();
    let unstable = picked
        .find_unstable_neurons_batched(&relu_pre_map)
        .expect("find_unstable_neurons_batched should handle empty batch");
    assert!(unstable.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_picked_domains_find_unstable_neurons_batched_basic() {
    // Create a batch of 2 domains with 4 neurons in linear0 layer
    // Domain 0: neurons 0,1 are unstable (l<0 && u>0), neurons 2,3 are stable
    // Domain 1: neurons 1,2 are unstable, neurons 0,3 are stable
    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![
                -1.0, -0.5, 0.1, 0.2, // Domain 0: neurons 0,1 have l<0
                0.1, -0.3, -0.4, 0.0, // Domain 1: neurons 1,2 have l<0
            ],
        )
        .unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![
                0.5, 0.3, 0.5, 0.5, // Domain 0: neurons 0,1 have u>0, but 2,3 too (but l>=0)
                0.3, 0.2, 0.1, -0.1, // Domain 1: neurons 0,1,2 have u>0 but 3 has u<0
            ],
        )
        .unwrap(),
    );

    let picked = PickedDomains {
        batch_size: 2,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::zeros(IxDyn(&[2, 2])),
        input_uppers: ArrayD::ones(IxDyn(&[2, 2])),
        global_lbs: vec![-1.0, -0.5],
        global_ubs: vec![1.0, 0.5],
        metadata: vec![metadata(-1.0, 1.0, 0), metadata(-0.5, 0.5, 1)],
    };

    // Map relu0 -> linear0 (relu's pre-activation layer)
    let relu_pre_map: HashMap<String, String> = [("relu0".to_string(), "linear0".to_string())]
        .into_iter()
        .collect();

    let unstable = picked
        .find_unstable_neurons_batched(&relu_pre_map)
        .expect("find_unstable_neurons_batched should compute unstable neurons");
    assert_eq!(unstable.len(), 2);

    // Domain 0 should have neurons 0,1 unstable (l<0 && u>0)
    assert!(unstable[0].contains(&("relu0".to_string(), 0)));
    assert!(unstable[0].contains(&("relu0".to_string(), 1)));
    assert_eq!(unstable[0].len(), 2);

    // Domain 1 should have neurons 1,2 unstable (l<0 && u>0)
    // Neuron 0 has l=0.1>=0, neuron 3 has u=-0.1<0
    assert!(unstable[1].contains(&("relu0".to_string(), 1)));
    assert!(unstable[1].contains(&("relu0".to_string(), 2)));
    assert_eq!(unstable[1].len(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_picked_domains_find_unstable_respects_constraints() {
    // Domain with 2 unstable neurons but one is already constrained
    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![-1.0, -0.5, 0.1]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 0.5, 0.5]).unwrap(),
    );

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::zeros(IxDyn(&[1, 2])),
        input_uppers: ArrayD::ones(IxDyn(&[1, 2])),
        global_lbs: vec![-1.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            constraints: vec![("relu0".to_string(), 0, true, None)], // Neuron 0 already constrained
            ..metadata(-1.0, 1.0, 1)
        }],
    };

    let relu_pre_map: HashMap<String, String> = [("relu0".to_string(), "linear0".to_string())]
        .into_iter()
        .collect();

    let unstable = picked
        .find_unstable_neurons_batched(&relu_pre_map)
        .expect("find_unstable_neurons_batched should respect constraints");
    assert_eq!(unstable.len(), 1);

    // Only neuron 1 should be returned (0 is constrained, 2 has l>=0)
    assert_eq!(unstable[0].len(), 1);
    assert!(unstable[0].contains(&("relu0".to_string(), 1)));
}

#[ntest::timeout(10000)]
#[test]
fn test_picked_domains_select_branch_batched_intercept_scoring() {
    // Two domains, each with 2 unstable neurons.
    // Test that the neuron with higher intercept score is selected.
    // intercept = (-l * u) / (u - l)
    //
    // Domain 0:
    //   neuron 0: l=-1.0, u=1.0 -> intercept = 1.0 / 2.0 = 0.5
    //   neuron 1: l=-0.5, u=2.0 -> intercept = 1.0 / 2.5 = 0.4
    //   -> select neuron 0 (higher intercept)
    //
    // Domain 1:
    //   neuron 0: l=-0.1, u=0.2 -> intercept = 0.02 / 0.3 = 0.067
    //   neuron 1: l=-0.8, u=0.4 -> intercept = 0.32 / 1.2 = 0.267
    //   -> select neuron 1 (higher intercept)

    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, -0.5, -0.1, -0.8]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 0.2, 0.4]).unwrap(),
    );

    let picked = PickedDomains {
        batch_size: 2,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::zeros(IxDyn(&[2, 2])),
        input_uppers: ArrayD::ones(IxDyn(&[2, 2])),
        global_lbs: vec![-1.0, -0.5],
        global_ubs: vec![1.0, 0.5],
        metadata: vec![metadata(-1.0, 1.0, 0), metadata(-0.5, 0.5, 0)],
    };

    let relu_pre_map: HashMap<String, String> = [("relu0".to_string(), "linear0".to_string())]
        .into_iter()
        .collect();

    let unstable = picked
        .find_unstable_neurons_batched(&relu_pre_map)
        .expect("find_unstable_neurons_batched should find candidates");
    let branches = picked
        .select_branch_batched(&unstable, &relu_pre_map)
        .expect("select_branch_batched should score candidates");

    assert_eq!(branches.len(), 2);

    // Domain 0: should select neuron 0 (intercept 0.5 > 0.4)
    let branch0 = branches[0].as_ref().unwrap();
    assert_eq!(branch0.0, "relu0");
    assert_eq!(branch0.1, 0);
    assert!((branch0.2 - 0.5).abs() < 1e-6);

    // Domain 1: should select neuron 1 (intercept 0.267 > 0.067)
    let branch1 = branches[1].as_ref().unwrap();
    assert_eq!(branch1.0, "relu0");
    assert_eq!(branch1.1, 1);
    assert!((branch1.2 - 0.2667).abs() < 0.01); // ~0.267
}

#[ntest::timeout(10000)]
#[test]
fn test_picked_domains_branching_accepts_non_contiguous_bounds_4250() {
    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "linear0".to_string(),
        make_non_contiguous_batch_bounds_2x4([[-1.0, -0.5, 0.1, 0.2], [0.1, -0.3, -0.4, 0.0]]),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "linear0".to_string(),
        make_non_contiguous_batch_bounds_2x4([[0.5, 0.3, 0.5, 0.5], [0.3, 0.2, 0.1, -0.1]]),
    );

    assert!(
        layer_lowers["linear0"].as_slice().is_none(),
        "test setup: lower bounds should be non-contiguous"
    );
    assert!(
        layer_uppers["linear0"].as_slice().is_none(),
        "test setup: upper bounds should be non-contiguous"
    );

    let picked = PickedDomains {
        batch_size: 2,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::zeros(IxDyn(&[2, 2])),
        input_uppers: ArrayD::ones(IxDyn(&[2, 2])),
        global_lbs: vec![-1.0, -0.5],
        global_ubs: vec![1.0, 0.5],
        metadata: vec![metadata(-1.0, 1.0, 0), metadata(-0.5, 0.5, 0)],
    };

    let relu_pre_map: HashMap<String, String> = [("relu0".to_string(), "linear0".to_string())]
        .into_iter()
        .collect();

    let unstable = picked
        .find_unstable_neurons_batched(&relu_pre_map)
        .expect("non-contiguous bounds should still yield unstable neurons");
    assert_eq!(unstable.len(), 2);
    assert_eq!(unstable[0].len(), 2);
    assert!(unstable[0].contains(&("relu0".to_string(), 0)));
    assert!(unstable[0].contains(&("relu0".to_string(), 1)));
    assert_eq!(unstable[1].len(), 2);
    assert!(unstable[1].contains(&("relu0".to_string(), 1)));
    assert!(unstable[1].contains(&("relu0".to_string(), 2)));

    let branches = picked
        .select_branch_batched(&unstable, &relu_pre_map)
        .expect("non-contiguous bounds should still yield intercept scores");
    let branch0 = branches[0].as_ref().expect("domain 0 should branch");
    let branch1 = branches[1].as_ref().expect("domain 1 should branch");
    // Domain 0 bounds: lower [-1.0, -0.5, 0.1, 0.2], upper [0.5, 0.3, 0.5, 0.5].
    //   neuron 0: l=-1.0, u=0.5 -> intercept = 0.5 / 1.5  = 0.3333
    //   neuron 1: l=-0.5, u=0.3 -> intercept = 0.15 / 0.8 = 0.1875
    //   -> select neuron 0.
    assert_eq!(branch0.1, 0);
    assert!((branch0.2 - 0.3333).abs() < 0.01);
    // Domain 1 bounds: lower [0.1, -0.3, -0.4, 0.0], upper [0.3, 0.2, 0.1, -0.1].
    //   neuron 1: l=-0.3, u=0.2 -> intercept = 0.06 / 0.5 = 0.12
    //   neuron 2: l=-0.4, u=0.1 -> intercept = 0.04 / 0.5 = 0.08
    //   -> select neuron 1.
    assert_eq!(branch1.1, 1);
    assert!((branch1.2 - 0.12).abs() < 0.01);
}

#[ntest::timeout(10000)]
#[test]
fn test_picked_domains_select_branch_batched_no_unstable() {
    // Domain with no unstable neurons (all stable)
    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.1, 0.2]).unwrap(), // all l >= 0
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.5, 0.6]).unwrap(),
    );

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::zeros(IxDyn(&[1, 2])),
        input_uppers: ArrayD::ones(IxDyn(&[1, 2])),
        global_lbs: vec![-1.0],
        global_ubs: vec![1.0],
        metadata: vec![metadata(-1.0, 1.0, 0)],
    };

    let relu_pre_map: HashMap<String, String> = [("relu0".to_string(), "linear0".to_string())]
        .into_iter()
        .collect();

    let unstable = picked
        .find_unstable_neurons_batched(&relu_pre_map)
        .expect("find_unstable_neurons_batched should produce empty unstable set");
    let branches = picked
        .select_branch_batched(&unstable, &relu_pre_map)
        .expect("select_branch_batched should handle empty unstable set");

    assert_eq!(branches.len(), 1);
    assert!(branches[0].is_none()); // No branching decision when all stable
}

#[ntest::timeout(10000)]
#[test]
fn test_picked_domains_select_branch_batched_stale_neuron_returns_error_2998() {
    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-1.0, -0.5]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 0.5]).unwrap(),
    );

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::zeros(IxDyn(&[1, 2])),
        input_uppers: ArrayD::ones(IxDyn(&[1, 2])),
        global_lbs: vec![-1.0],
        global_ubs: vec![1.0],
        metadata: vec![metadata(-1.0, 1.0, 0)],
    };

    let relu_pre_map: HashMap<String, String> = [("relu0".to_string(), "linear0".to_string())]
        .into_iter()
        .collect();
    let stale_unstable = vec![vec![("relu0".to_string(), 7)]];

    let err = picked
        .select_branch_batched(&stale_unstable, &relu_pre_map)
        .expect_err("stale branch candidate should return NyError");
    match err {
        NyError::InternalError(msg) => {
            assert!(
                msg.contains("select_branch_batched: bound index"),
                "expected checked bound lookup context, got: {msg}"
            );
        }
        other => panic!("expected InternalError, got: {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_picked_domains_find_unstable_neurons_batched_multi_layer() {
    // Test with 2 ReLU layers: relu0 (3 neurons) and relu1 (2 neurons)
    // Domain 0: relu0 has 1 unstable, relu1 has 2 unstable
    // Domain 1: relu0 has 2 unstable, relu1 has 0 unstable
    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![
                -1.0, 0.5, 0.5, // Domain 0: only neuron 0 has l<0
                -0.5, -0.3, 0.1, // Domain 1: neurons 0,1 have l<0
            ],
        )
        .unwrap(),
    );
    layer_lowers.insert(
        "linear1".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![
                -0.2, -0.4, // Domain 0: both have l<0
                0.1, 0.2, // Domain 1: neither has l<0
            ],
        )
        .unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "linear0".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![
                0.5, 1.0, 1.0, // Domain 0: all have u>0
                0.5, 0.3, 0.5, // Domain 1: all have u>0
            ],
        )
        .unwrap(),
    );
    layer_uppers.insert(
        "linear1".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![
                0.3, 0.5, // Domain 0: both have u>0
                0.3, 0.5, // Domain 1: both have u>0
            ],
        )
        .unwrap(),
    );

    let picked = PickedDomains {
        batch_size: 2,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::zeros(IxDyn(&[2, 2])),
        input_uppers: ArrayD::ones(IxDyn(&[2, 2])),
        global_lbs: vec![-1.0, -0.5],
        global_ubs: vec![1.0, 0.5],
        metadata: vec![metadata(-1.0, 1.0, 0), metadata(-0.5, 0.5, 0)],
    };

    let relu_pre_map: HashMap<String, String> = [
        ("relu0".to_string(), "linear0".to_string()),
        ("relu1".to_string(), "linear1".to_string()),
    ]
    .into_iter()
    .collect();

    let unstable = picked
        .find_unstable_neurons_batched(&relu_pre_map)
        .expect("find_unstable_neurons_batched should aggregate per-layer unstable neurons");
    assert_eq!(unstable.len(), 2);

    // Domain 0: relu0 neuron 0 + relu1 neurons 0,1 = 3 total
    assert!(unstable[0].contains(&("relu0".to_string(), 0)));
    assert!(unstable[0].contains(&("relu1".to_string(), 0)));
    assert!(unstable[0].contains(&("relu1".to_string(), 1)));
    assert_eq!(unstable[0].len(), 3);

    // Domain 1: relu0 neurons 0,1 = 2 total (relu1 has no unstable)
    assert!(unstable[1].contains(&("relu0".to_string(), 0)));
    assert!(unstable[1].contains(&("relu0".to_string(), 1)));
    assert_eq!(unstable[1].len(), 2);
}
