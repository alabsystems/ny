// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! NY ext 3 acceptance: permutation / finite-rotation invariance via
//! difference networks. Constructed invariant nets verify; asymmetric nets
//! falsify with a concrete witness (ny-groundtruth witness-search pattern).

use ndarray::{arr1, arr2, Array1};
use ny_api::graph::{GraphNetwork, GraphNode};
use ny_api::layers::{Layer, LinearLayer, ReLULayer};
use ny_api::symmetry::{
    block_permutation, octahedral_rotations, verify_permutation_invariance,
    verify_rotation_invariance_finite, SymmetryOutcome,
};
use ny_api::Bound;
use ny_tensor::BoundedTensor;

/// Build `relu(W x + b)` as a GraphNetwork (output = the ReLU node).
fn relu_net(weight: ndarray::Array2<f32>, bias: Array1<f32>) -> GraphNetwork {
    let mut f = GraphNetwork::new();
    f.add_node(GraphNode::from_input(
        "lin",
        Layer::Linear(LinearLayer::new(weight, Some(bias)).expect("valid linear")),
    ));
    f.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["lin".to_string()],
    ));
    f.set_output("relu");
    f
}

/// Evaluate a network at a concrete point via zero-width IBP (sound
/// enclosure; exact here because all test weights are small integers).
fn eval_at(net: &GraphNetwork, point: &[f32]) -> (f32, f32) {
    let arr = Array1::from(point.to_vec()).into_dyn();
    let t = BoundedTensor::new(arr.clone(), arr).expect("valid point");
    let out = net.propagate_ibp(&t).expect("evaluation succeeds");
    (out.lower()[0], out.upper()[0])
}

fn unit_box(dim: usize) -> Vec<Bound> {
    vec![Bound::new(0.0, 1.0); dim]
}

fn symmetric_box(dim: usize) -> Vec<Bound> {
    vec![Bound::new(-1.0, 1.0); dim]
}

/// A symmetric sum net f(x) = relu(x0 + x1 + x2 + 1) is permutation
/// invariant; on [0, 1]³ the ReLU is stably active, so CROWN cancels the two
/// branches up to sound directed rounding, proving |f(Px) − f(x)| ≤ 1e-4.
#[test]
fn symmetric_sum_net_verifies_permutation_invariance() {
    let f = relu_net(arr2(&[[1.0_f32, 1.0, 1.0]]), arr1(&[1.0_f32]));
    let outcome =
        verify_permutation_invariance(&f, &[1, 2, 0], &unit_box(3), 1e-4).expect("query runs");
    match outcome {
        SymmetryOutcome::Verified { difference_bounds } => {
            assert_eq!(difference_bounds.len(), 1);
            assert!(difference_bounds[0].lower() >= -1e-4 && difference_bounds[0].upper() <= 1e-4);
        }
        other => panic!("expected Verified, got {other:?}"),
    }
}

/// DeepSets-style net: shared per-point encoder (block-diagonal Linear) +
/// sum pooling is invariant under permutations of the point blocks; the
/// blockwise permutation comes from `block_permutation`.
#[test]
fn deep_sets_net_verifies_block_permutation_invariance() {
    // 3 points × 2 dims; shared encoder W = [[1, 2], [3, 4]], bias (1, 1);
    // all inputs in [0, 1] ⇒ pre-activations ≥ 1 ⇒ stable ReLUs ⇒ exact.
    let mut enc = ndarray::Array2::<f32>::zeros((6, 6));
    for p in 0..3 {
        enc[[2 * p, 2 * p]] = 1.0;
        enc[[2 * p, 2 * p + 1]] = 2.0;
        enc[[2 * p + 1, 2 * p]] = 3.0;
        enc[[2 * p + 1, 2 * p + 1]] = 4.0;
    }
    let mut f = GraphNetwork::new();
    f.add_node(GraphNode::from_input(
        "enc",
        Layer::Linear(LinearLayer::new(enc, Some(arr1(&[1.0_f32; 6]))).expect("valid encoder")),
    ));
    f.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["enc".to_string()],
    ));
    // Sum-pool the two features across the three points.
    f.add_node(GraphNode::new(
        "pool",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[
                    [1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                ]),
                None,
            )
            .expect("valid pool"),
        ),
        vec!["relu".to_string()],
    ));
    f.set_output("pool");

    let perm = block_permutation(&[2, 0, 1], 2).expect("valid point permutation");
    assert_eq!(perm, vec![4, 5, 0, 1, 2, 3]);
    let outcome = verify_permutation_invariance(&f, &perm, &unit_box(6), 1e-4).expect("query runs");
    assert!(
        outcome.is_verified(),
        "DeepSets net must verify: {outcome:?}"
    );
}

/// An asymmetric net is falsified with a concrete witness, and the witness
/// checks out against the ORIGINAL network: |f(Px*) − f(x*)| > ε.
#[test]
fn asymmetric_net_is_falsified_with_witness() {
    // f(x) = relu(x0 + 2·x1 + 3·x2 + 10): stably active on [0, 1]³, so
    // f(Px) − f(x) = x1 − x0 for the swap P; range [−1, 1] ⊄ [−0.5, 0.5].
    let f = relu_net(arr2(&[[1.0_f32, 2.0, 3.0]]), arr1(&[10.0_f32]));
    let permutation = [1_usize, 0, 2];
    let outcome =
        verify_permutation_invariance(&f, &permutation, &unit_box(3), 0.5).expect("query runs");
    match outcome {
        SymmetryOutcome::Falsified {
            witness,
            difference,
        } => {
            assert_eq!(witness.len(), 3);
            assert!(
                difference.lower() > 0.5 || difference.upper() < -0.5,
                "enclosure must certainly violate ε: {difference:?}"
            );
            // Cross-check on the original network: evaluate f at Px* and x*.
            let permuted: Vec<f32> = permutation.iter().map(|&i| witness[i]).collect();
            let (f_perm_lo, f_perm_hi) = eval_at(&f, &permuted);
            let (f_lo, f_hi) = eval_at(&f, &witness);
            let diff_lo = f_perm_lo - f_hi;
            let diff_hi = f_perm_hi - f_lo;
            assert!(
                diff_lo > 0.5 || diff_hi < -0.5,
                "witness must violate on the original net: diff in [{diff_lo}, {diff_hi}]"
            );
        }
        other => panic!("expected Falsified with witness, got {other:?}"),
    }
}

/// The verified box must be setwise invariant under the permutation;
/// otherwise the property is ill-posed and rejected.
#[test]
fn non_symmetric_box_is_rejected() {
    let f = relu_net(arr2(&[[1.0_f32, 1.0]]), arr1(&[1.0_f32]));
    let bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 2.0)];
    let err = verify_permutation_invariance(&f, &[1, 0], &bounds, 1e-3)
        .expect_err("asymmetric box must be rejected");
    assert!(err.to_string().contains("not invariant"), "{err}");
}

#[test]
fn invalid_permutations_are_rejected() {
    let f = relu_net(arr2(&[[1.0_f32, 1.0]]), arr1(&[1.0_f32]));
    let b = unit_box(2);
    assert!(verify_permutation_invariance(&f, &[0, 0], &b, 1e-3).is_err());
    assert!(verify_permutation_invariance(&f, &[0, 2], &b, 1e-3).is_err());
    assert!(verify_permutation_invariance(&f, &[0], &b, 1e-3).is_err());
    assert!(verify_permutation_invariance(&f, &[1, 0], &b, 0.0).is_err());
}

/// f(x, y, z) = relu(z + 2) depends only on z, so it is invariant under all
/// four 90° rotations about the z-axis; the ReLU is stably active on
/// [−1, 1]³, so every rotation verifies with a tight ε (1e-4, absorbing sound rounding).
#[test]
fn z_axis_rotation_invariant_net_verifies() {
    let f = relu_net(arr2(&[[0.0_f32, 0.0, 1.0]]), arr1(&[2.0_f32]));
    let rotations = vec![
        arr2(&[[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), // I
        arr2(&[[0.0_f32, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), // Rz(90°)
        arr2(&[[-1.0_f32, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), // Rz(180°)
        arr2(&[[0.0_f32, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), // Rz(270°)
    ];
    let outcome = verify_rotation_invariance_finite(&f, &rotations, &symmetric_box(3), 1e-4)
        .expect("query runs");
    assert_eq!(outcome.per_rotation.len(), 4);
    assert!(outcome.all_verified(), "all four z-rotations must verify");
    assert!(outcome.first_falsified().is_none());
}

/// A net reading x is NOT invariant under Rz(90°): h = f(Rx) − f(x) = −y − x
/// reaches −2 on the corner (1, 1, ·), certainly violating ε = 0.5 — the
/// witness search must find and certify it.
#[test]
fn rotation_sensitive_net_is_falsified_with_witness() {
    let f = relu_net(arr2(&[[1.0_f32, 0.0, 0.0]]), arr1(&[3.0_f32]));
    let rz90 = arr2(&[[0.0_f32, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let outcome =
        verify_rotation_invariance_finite(&f, &[rz90], &symmetric_box(3), 0.5).expect("query runs");
    assert!(!outcome.all_verified());
    let (idx, falsified) = outcome
        .first_falsified()
        .expect("must produce a concrete counterexample");
    assert_eq!(idx, 0);
    match falsified {
        SymmetryOutcome::Falsified { witness, .. } => {
            // |−y − x| > 0.5 at the witness.
            assert!(
                (witness[0] + witness[1]).abs() > 0.5,
                "witness must violate: {witness:?}"
            );
        }
        other => panic!("expected Falsified, got {other:?}"),
    }
}

/// A constant network is invariant under the full 24-element octahedral
/// rotation set (exercising validation + verification of every element).
#[test]
fn constant_net_verifies_all_octahedral_rotations() {
    let f = relu_net(arr2(&[[0.0_f32, 0.0, 0.0]]), arr1(&[5.0_f32]));
    let rotations = octahedral_rotations();
    assert_eq!(rotations.len(), 24);
    let outcome = verify_rotation_invariance_finite(&f, &rotations, &symmetric_box(3), 1e-4)
        .expect("query runs");
    assert!(
        outcome.all_verified(),
        "constant net is invariant under all 24"
    );
}

/// Reflections, continuous rotations, mismatched dimensions, empty sets, and
/// non-invariant boxes are all rejected up front.
#[test]
fn rotation_preconditions_are_enforced() {
    let f = relu_net(arr2(&[[0.0_f32, 0.0, 1.0]]), arr1(&[2.0_f32]));
    let b3 = symmetric_box(3);

    // Reflection (det = −1).
    let refl = arr2(&[[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
    let err = verify_rotation_invariance_finite(&f, &[refl], &b3, 1e-3).unwrap_err();
    assert!(err.to_string().contains("reflection"), "{err}");

    // Continuous SO(3) rotation: out of scope in v1.
    let c = std::f32::consts::FRAC_1_SQRT_2;
    let cont = arr2(&[[c, -c, 0.0], [c, c, 0.0], [0.0, 0.0, 1.0]]);
    let err = verify_rotation_invariance_finite(&f, &[cont], &b3, 1e-3).unwrap_err();
    assert!(err.to_string().contains("out of scope"), "{err}");

    // Input dimension not a multiple of the rotation dimension.
    let rz90 = arr2(&[[0.0_f32, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    assert!(verify_rotation_invariance_finite(&f, &[rz90], &symmetric_box(4), 1e-3).is_err());

    // Empty rotation set.
    assert!(verify_rotation_invariance_finite(&f, &[], &b3, 1e-3).is_err());

    // Box not invariant: z ∈ [0, 2] but a rotation about x maps y ↔ z.
    let rx90 = arr2(&[[1.0_f32, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]);
    let skewed = vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
        Bound::new(0.0, 2.0),
    ];
    let err = verify_rotation_invariance_finite(&f, &[rx90], &skewed, 1e-3).unwrap_err();
    assert!(err.to_string().contains("not invariant"), "{err}");
}
