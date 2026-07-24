// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dense low-effective-dimension sweep regression tests (#dense-sweep,
//! `attack: dense_low_dim_sweep`).
//!
//! The sweep is a pre-PGD phase for boxes where only a handful of input dims
//! have nonzero width (cctsdb_yolo: 2 of 39). A deterministic grid plus top-k
//! refinement covers the whole 2-D box at a resolution random restarts cannot
//! match. Attack-only: a hit is a candidate for the normal witness path.

use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::layers::*;
use crate::pgd_attack::attacker::PgdAttacker;
use crate::pgd_attack::config::PgdConfig;
use crate::Network;

/// out = 1 − 50·|x0 − 0.4| − 50·|x1 − 0.6| (pyramid peak at (0.4, 0.6)).
/// out ≥ 0.5 requires L1 distance ≤ 0.01 from the peak — a ~5·10⁻⁵ fraction
/// of the [-1, 1]² box, far below the initial grid resolution, so only the
/// top-k refinement rounds can land it.
fn pyramid_pocket_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[50.0_f32, 0.0], [-50.0, 0.0], [0.0, 50.0], [0.0, -50.0]]),
            Some(arr1(&[-20.0, 20.0, -30.0, 30.0])),
        )
        .unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[-1.0_f32, -1.0, -1.0, -1.0]]), Some(arr1(&[1.0]))).unwrap(),
    ));
    network
}

fn unit_box_2d() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 1.0]).unwrap(),
    )
    .unwrap()
}

/// num_restarts: 0 isolates the sweep — PGD proper contributes nothing.
fn sweep_config(dense_low_dim_sweep: bool) -> PgdConfig {
    PgdConfig {
        num_restarts: 0,
        num_steps: 1,
        parallel: false,
        seed: 42,
        dense_low_dim_sweep,
        dense_sweep_points: 4096,
        ..PgdConfig::default()
    }
}

/// The sweep's grid + refinement finds the narrow violating pocket that zero
/// PGD restarts (and the coarse initial grid alone) cannot reach.
#[ntest::timeout(60000)]
#[test]
fn test_dense_sweep_finds_narrow_pocket_2d() {
    let network = pyramid_pocket_network();
    let bounds = unit_box_2d();

    // Without the sweep (and zero restarts) nothing can find the pocket.
    let off = PgdAttacker::new(sweep_config(false));
    let result = off.attack(&network, &bounds, 0, 0.5, true).unwrap();
    assert!(
        !result.found_counterexample,
        "with the sweep off and zero restarts, no counterexample should be found"
    );

    let on = PgdAttacker::new(sweep_config(true));
    let result = on.attack(&network, &bounds, 0, 0.5, true).unwrap();
    assert!(
        result.found_counterexample,
        "dense sweep should find the pyramid pocket (best={}, evals={})",
        result.best_output_value, result.total_evaluations
    );
    assert!(
        result.best_output_value >= 0.5,
        "violation must hold at the returned point, got {}",
        result.best_output_value
    );
    assert!(
        result.total_evaluations <= 4096 + 1,
        "sweep must respect its point budget, used {}",
        result.total_evaluations
    );
    let ce = result.counterexample.expect("counterexample present");
    assert_eq!(ce.len(), 2);
    for (i, v) in ce.iter().enumerate() {
        assert!(
            (-1.0..=1.0).contains(v),
            "counterexample dim {i} out of box: {v}"
        );
    }
    let l1 = (ce[[0]] - 0.4).abs() + (ce[[1]] - 0.6).abs();
    assert!(
        l1 <= 0.01 + 1e-6,
        "counterexample must sit inside the L1-0.01 pocket, distance {l1}"
    );
}

/// Effective-dimension gate: when more dims vary than `dense_sweep_max_dims`,
/// the sweep must decline and leave the attack to the normal PGD schedule.
#[ntest::timeout(10000)]
#[test]
fn test_dense_sweep_gated_off_above_max_dims() {
    let network = pyramid_pocket_network();
    let bounds = unit_box_2d();

    let attacker = PgdAttacker::new(PgdConfig {
        dense_sweep_max_dims: 1, // 2 varying dims > 1 → gate closed
        ..sweep_config(true)
    });
    let result = attacker.attack(&network, &bounds, 0, 0.5, true).unwrap();
    assert!(
        !result.found_counterexample,
        "sweep must be gated off when varying dims exceed dense_sweep_max_dims"
    );
    assert_eq!(
        result.total_evaluations, 0,
        "gated-off sweep must not spend evaluations"
    );
}

/// Pinned dims (width 0) stay at their exact bound value in every probe: the
/// third input dim is pinned at 0.25 and the network forwards it to the
/// output, so the returned witness must carry it verbatim.
#[ntest::timeout(30000)]
#[test]
fn test_dense_sweep_keeps_pinned_dims_exact() {
    // out0 = 1 − 50|x0−0.4| − 50|x1−0.6| (as above, ignoring x2);
    // the pinned x2 participates via out0 += 0·x2 and must stay 0.25 in the CE.
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[
                [50.0_f32, 0.0, 0.0],
                [-50.0, 0.0, 0.0],
                [0.0, 50.0, 0.0],
                [0.0, -50.0, 0.0],
            ]),
            Some(arr1(&[-20.0, 20.0, -30.0, 30.0])),
        )
        .unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[-1.0_f32, -1.0, -1.0, -1.0]]), Some(arr1(&[1.0]))).unwrap(),
    ));

    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0_f32, -1.0, 0.25]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0_f32, 1.0, 0.25]).unwrap(),
    )
    .unwrap();

    let attacker = PgdAttacker::new(sweep_config(true));
    let result = attacker.attack(&network, &bounds, 0, 0.5, true).unwrap();
    assert!(
        result.found_counterexample,
        "sweep should still fire with 2 varying + 1 pinned dim (best={})",
        result.best_output_value
    );
    let ce = result.counterexample.expect("counterexample present");
    assert_eq!(
        ce[[2]],
        0.25,
        "pinned dim must be carried verbatim in the witness"
    );
}
