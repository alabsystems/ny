// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for #1924: non-contiguous tensor silent fallback
//! in `find_unstable_neurons_batched` and `select_branch_batched`.

use super::*;

fn metadata(lower_bound: f32, upper_bound: f32) -> DomainMetadata {
    DomainMetadata {
        lower_bound,
        upper_bound,
        depth: 0,
        constraints: vec![],
        cached_la: None,
        needs_bounding: false,
        node_bounds_override: None,
        alpha_state: None,
    }
}

/// Create a non-contiguous ArrayD with the given logical values.
/// Uses Fortran (column-major) memory order so `as_slice()` returns `None`,
/// which is exactly the condition that triggered the #1924 bug.
fn make_non_contiguous(data: &[f32], shape: &[usize]) -> ArrayD<f32> {
    use ndarray::ShapeBuilder;
    let total: usize = shape.iter().product();
    assert_eq!(data.len(), total, "data length must match shape product");
    // Fortran order stores column-major; as_slice() requires C-contiguous.
    // We must reorder data into Fortran layout so logical indexing matches.
    let c_arr = ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap();
    let mut f_arr = ArrayD::zeros(IxDyn(shape).f());
    f_arr.assign(&c_arr);
    assert!(
        f_arr.as_slice().is_none(),
        "test helper: expected non-contiguous (Fortran-order) array \
         but as_slice() succeeded; ndarray version may have changed"
    );
    f_arr
}

#[ntest::timeout(10000)]
#[test]
fn test_find_unstable_non_contiguous_1924() {
    // Set up a PickedDomains with non-contiguous pre-activation arrays.
    // 2 domains, 3 neurons each:
    //   Domain 0: lower=[-1, -0.5, 0.5], upper=[1, 0.3, 2.0]
    //     → neuron 0 unstable (l<0, u>0), neuron 1 unstable, neuron 2 stable (l>0)
    //   Domain 1: lower=[-2, 0.1, -0.5], upper=[0.5, 1.0, -0.1]
    //     → neuron 0 unstable, neuron 1 stable (l>0), neuron 2 stable (u<0)
    let lower_data = vec![-1.0, -0.5, 0.5, -2.0, 0.1, -0.5];
    let upper_data = vec![1.0, 0.3, 2.0, 0.5, 1.0, -0.1];

    let lower = make_non_contiguous(&lower_data, &[2, 3]);
    let upper = make_non_contiguous(&upper_data, &[2, 3]);

    // Verify values are correct despite non-contiguous layout
    assert_eq!(lower[[0, 0]], -1.0);
    assert_eq!(lower[[1, 2]], -0.5);
    assert_eq!(upper[[0, 0]], 1.0);

    let mut layer_lowers = HashMap::new();
    layer_lowers.insert("linear0".to_string(), lower);
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert("linear0".to_string(), upper);

    let picked = PickedDomains {
        batch_size: 2,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::zeros(IxDyn(&[2, 2])),
        input_uppers: ArrayD::ones(IxDyn(&[2, 2])),
        global_lbs: vec![-1.0, -2.0],
        global_ubs: vec![2.0, 1.0],
        metadata: vec![metadata(-1.0, 2.0), metadata(-2.0, 1.0)],
    };

    let relu_pre_map: HashMap<String, String> = [("relu0".to_string(), "linear0".to_string())]
        .into_iter()
        .collect();

    let unstable = picked
        .find_unstable_neurons_batched(&relu_pre_map)
        .expect("find_unstable_neurons_batched should support non-contiguous inputs");

    // Domain 0: neurons 0,1 unstable (l<0 && u>0)
    assert_eq!(unstable.len(), 2);
    let d0: Vec<usize> = unstable[0].iter().map(|(_, idx)| *idx).collect();
    assert!(
        d0.contains(&0) && d0.contains(&1),
        "domain 0 should have neurons 0,1 unstable, got {:?}",
        unstable[0]
    );
    assert!(
        !d0.contains(&2),
        "domain 0 neuron 2 (l=0.5>0) should be stable"
    );

    // Domain 1: only neuron 0 unstable
    let d1: Vec<usize> = unstable[1].iter().map(|(_, idx)| *idx).collect();
    assert!(
        d1.contains(&0),
        "domain 1 neuron 0 should be unstable, got {:?}",
        unstable[1]
    );
    assert!(
        !d1.contains(&1) && !d1.contains(&2),
        "domain 1 neurons 1,2 should be stable, got {:?}",
        unstable[1]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_select_branch_non_contiguous_1924() {
    // Exercise select_branch_batched with non-contiguous pre-activation arrays.
    // 2 domains, 3 neurons each (need rows!=cols for Fortran-order non-contiguity):
    //   Domain 0: lower=[-1, -3, -0.5], upper=[2, 1, 0.8]
    //     neuron 0: intercept = (1*2)/(2+1) = 2/3 ≈ 0.667
    //     neuron 1: intercept = (3*1)/(1+3) = 3/4 = 0.75
    //     neuron 2: intercept = (0.5*0.8)/(0.8+0.5) = 0.4/1.3 ≈ 0.308
    //     → should select neuron 1 (highest intercept)
    //   Domain 1: lower=[-2, -1, 0.1], upper=[4, 3, 1.0]
    //     neuron 0: intercept = (2*4)/(4+2) = 8/6 ≈ 1.333
    //     neuron 1: intercept = (1*3)/(3+1) = 3/4 = 0.75
    //     neuron 2: stable (l>0), not in unstable list
    //     → should select neuron 0 (highest intercept)
    let lower = make_non_contiguous(&[-1.0, -3.0, -0.5, -2.0, -1.0, 0.1], &[2, 3]);
    let upper = make_non_contiguous(&[2.0, 1.0, 0.8, 4.0, 3.0, 1.0], &[2, 3]);

    let mut layer_lowers = HashMap::new();
    layer_lowers.insert("linear0".to_string(), lower);
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert("linear0".to_string(), upper);

    let picked = PickedDomains {
        batch_size: 2,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::zeros(IxDyn(&[2, 2])),
        input_uppers: ArrayD::ones(IxDyn(&[2, 2])),
        global_lbs: vec![-1.0, -2.0],
        global_ubs: vec![2.0, 4.0],
        metadata: vec![metadata(-1.0, 2.0), metadata(-2.0, 4.0)],
    };

    let relu_pre_map: HashMap<String, String> = [("relu0".to_string(), "linear0".to_string())]
        .into_iter()
        .collect();

    // Unstable neurons per domain
    let unstable = vec![
        vec![
            ("relu0".to_string(), 0),
            ("relu0".to_string(), 1),
            ("relu0".to_string(), 2),
        ],
        vec![("relu0".to_string(), 0), ("relu0".to_string(), 1)],
    ];

    let branches = picked
        .select_branch_batched(&unstable, &relu_pre_map)
        .expect("select_branch_batched should support non-contiguous inputs");
    assert_eq!(branches.len(), 2);

    // Domain 0: neuron 1 has highest intercept (0.75)
    let (name0, idx0, score0) = branches[0]
        .as_ref()
        .expect("domain 0 should select a branch for non-contiguous input");
    assert_eq!(name0, "relu0");
    assert_eq!(
        *idx0, 1,
        "domain 0 should pick neuron 1 (highest intercept)"
    );
    assert!(
        (*score0 - 0.75).abs() < 1e-5,
        "domain 0 intercept should be 0.75, got {}",
        score0
    );

    // Domain 1: neuron 0 has highest intercept (8/6 ≈ 1.333)
    let (name1, idx1, score1) = branches[1]
        .as_ref()
        .expect("domain 1 should select a branch for non-contiguous input");
    assert_eq!(name1, "relu0");
    assert_eq!(
        *idx1, 0,
        "domain 1 should pick neuron 0 (highest intercept)"
    );
    assert!(
        (*score1 - 8.0 / 6.0).abs() < 1e-5,
        "domain 1 intercept should be ~1.333, got {}",
        score1
    );
}
