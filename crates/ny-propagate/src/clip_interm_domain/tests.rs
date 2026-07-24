// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::beta_crown::constraint_store::{
    ConstraintOrigin, ConstraintSense, DomainConstraintStore,
};
use ndarray::{array, s, Array1, Array2};

#[ntest::timeout(10000)]
#[test]
fn split_build_and_preprocess_poll_inside_dense_work() {
    use crate::beta_crown::branching::GraphNeuronConstraint;

    let dim = 8192usize;
    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu_1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut polls = 0usize;
    let mut expire = || {
        polls += 1;
        polls >= 5
    };
    let err = build_split_constraints_with_deadline_check(
        &history,
        |_name, _idx, deadline: &mut _| {
            let mut lower = Array1::<f32>::zeros(dim);
            let mut upper = Array1::<f32>::zeros(dim);
            for j in 0..dim {
                if j.is_multiple_of(1024) && deadline() {
                    return None;
                }
                lower[j] = 1.0;
                upper[j] = 1.0;
            }
            Some((lower, 0.0, upper, 0.0))
        },
        dim,
        &mut expire,
    )
    .expect_err("expiry while sourcing a split row must refuse the whole set");
    assert!(matches!(err, ny_core::NyError::DeadlineExceeded(_)));

    let n = 32_768usize;
    let constraints = SplitConstraints {
        a_matrix: Array2::from_elem((n, 1), 1.0f32),
        b_vector: Array1::from_elem(n, -0.5f32),
        num_constraints: n,
    };
    let lo = array![-1.0f32];
    let hi = array![1.0f32];
    let mut polls = 0usize;
    let mut expire = || {
        polls += 1;
        polls >= 8
    };
    let err = sort_out_constraints_with_deadline_check(&constraints, &lo, &hi, &mut expire)
        .expect_err("X=1 preprocessing must poll across constraint rows");
    assert!(matches!(err, ny_core::NyError::DeadlineExceeded(_)));
    assert!(err.to_string().contains("preprocessing"));
}

#[ntest::timeout(10000)]
#[test]
fn test_build_split_constraints_empty() {
    let history = GraphSplitHistory::new();
    let constraints =
        build_split_constraints(&history, |_name, _idx| None, 4).expect("finite constraint");

    assert!(constraints.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_split_constraints_from_store_normalizes_ge() {
    let mut store = DomainConstraintStore::new();
    store
        .delta_mut()
        .add_constraint(
            &[0, 1],
            &[1.0, -2.0],
            0.5,
            ConstraintSense::Ge,
            ConstraintOrigin::Split,
        )
        .expect("finite constraint");

    let constraints = split_constraints_from_store(&store, 2).expect("finite constraint");

    assert_eq!(constraints.a_matrix.shape(), &[1, 2]);
    assert_eq!(constraints.b_vector.shape(), &[1]);
    assert!((constraints.a_matrix[[0, 0]] - -1.0).abs() < 1e-6);
    assert!((constraints.a_matrix[[0, 1]] - 2.0).abs() < 1e-6);
    assert!((constraints.b_vector[0] - -0.5).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_build_split_constraints_relu_active() {
    use crate::beta_crown::branching::GraphNeuronConstraint;

    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu_1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0, // z >= 0
    });

    // Mock linear bounds: z = 2*x1 + 3*x2 - 1 (lower bound)
    //                     z = 2*x1 + 3*x2 + 1 (upper bound)
    let constraints = build_split_constraints(
        &history,
        |name, idx| {
            if name == "relu_1" && idx == 0 {
                Some((
                    array![2.0, 3.0], // lA
                    -1.0,             // lbias
                    array![2.0, 3.0], // uA
                    1.0,              // ubias
                ))
            } else {
                None
            }
        },
        2,
    )
    .expect("finite constraint");

    assert_eq!(constraints.num_constraints, 1);

    // Active branch (z >= 0) uses upper bound:
    // -uA·x + (-ubias + 0) <= 0
    // => -2*x1 - 3*x2 - 1 <= 0
    let a_row = constraints.a_matrix.row(0);
    assert!((a_row[0] - (-2.0)).abs() < 1e-6);
    assert!((a_row[1] - (-3.0)).abs() < 1e-6);
    assert!((constraints.b_vector[0] - (-1.0)).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_build_split_constraints_relu_inactive() {
    use crate::beta_crown::branching::GraphNeuronConstraint;

    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu_1".to_string(),
        neuron_idx: 0,
        is_active: false,
        score: 0.0, // z <= 0
    });

    // Mock linear bounds: z = 2*x1 + 3*x2 - 1 (lower bound)
    let constraints = build_split_constraints(
        &history,
        |name, idx| {
            if name == "relu_1" && idx == 0 {
                Some((
                    array![2.0, 3.0], // lA
                    -1.0,             // lbias
                    array![2.0, 3.0], // uA
                    1.0,              // ubias
                ))
            } else {
                None
            }
        },
        2,
    )
    .expect("finite constraint");

    assert_eq!(constraints.num_constraints, 1);

    // Inactive branch (z <= 0) uses lower bound:
    // lA·x + (lbias - 0) <= 0
    // => 2*x1 + 3*x2 - 1 <= 0
    let a_row = constraints.a_matrix.row(0);
    assert!((a_row[0] - 2.0).abs() < 1e-6);
    assert!((a_row[1] - 3.0).abs() < 1e-6);
    assert!((constraints.b_vector[0] - (-1.0)).abs() < 1e-6);
}

/// #clip-interm DIRECTION PIN (constraints.rs:98-112): the half-space each split
/// premise builds must be a SOUND NECESSARY CONDITION of the true ReLU sign — i.e.
/// EVERY input satisfying the true sign must also satisfy the derived half-space
/// `A·x + b ≤ 0`. A flipped direction (active using the LOWER bound instead of the
/// UPPER, or vice-versa) would exclude truly-feasible points ⇒ the clip's feasible
/// region would be a SUBSET of the true one ⇒ a too-tight (false-VERIFY) bound.
///
/// The neuron's TRUE pre-activation is an exact affine `z = a·x + b`; the CROWN
/// linear bounds fed to `build_split_constraints` are a LOOSER enclosure
/// (`a·x + b − δ ≤ z ≤ a·x + b + δ`, δ>0), so the built constraint is a genuine
/// relaxation (necessary, not sufficient). 2 input dims, an ACTIVE and an INACTIVE
/// premise (both direction arms), dense boundary sampling.
#[ntest::timeout(10000)]
#[test]
fn direction_relaxation_is_necessary_condition_2neuron_sampling() {
    use crate::beta_crown::branching::GraphNeuronConstraint;

    const DELTA: f32 = 0.3; // enclosure slack: bounds are looser than the true z

    // Build the single-premise constraint (a_row, b_val) for neuron with true
    // affine z = a·x + b, split at 0 with the given direction.
    let build = |a: [f32; 2], b: f32, is_active: bool| -> (Vec<f32>, f32) {
        let mut history = GraphSplitHistory::new();
        history.add_constraint(GraphNeuronConstraint {
            node_name: "relu_1".to_string(),
            neuron_idx: 0,
            is_active,
            score: 0.0,
        });
        let c = build_split_constraints(
            &history,
            |name, idx| {
                if name == "relu_1" && idx == 0 {
                    // Looser-but-valid enclosure of z = a·x + b over ANY box.
                    Some((array![a[0], a[1]], b - DELTA, array![a[0], a[1]], b + DELTA))
                } else {
                    None
                }
            },
            2,
        )
        .expect("finite constraint");
        assert_eq!(c.num_constraints, 1);
        (c.a_matrix.row(0).to_vec(), c.b_vector[0])
    };

    // Two neurons: A split ACTIVE (z ≥ 0), B split INACTIVE (z ≤ 0).
    let (a_coef, a_bias) = ([1.0f32, -0.5], 0.2f32);
    let (b_coef, b_bias) = ([-0.7f32, 0.4], -0.1f32);
    let (ha, hab) = build(a_coef, a_bias, true); // active necessary condition
    let (hb, hbb) = build(b_coef, b_bias, false); // inactive necessary condition

    let z = |c: [f32; 2], bias: f32, x: [f32; 2]| c[0] * x[0] + c[1] * x[1] + bias;
    let half = |a: &[f32], b: f32, x: [f32; 2]| a[0] * x[0] + a[1] * x[1] + b;

    // Dense grid over [-1,1]^2 (fine enough to sit near the z=0 boundary band).
    let n = 200i32;
    let (mut active_feasible, mut inactive_feasible) = (0u32, 0u32);
    let (mut active_relaxed_extra, mut inactive_relaxed_extra) = (0u32, 0u32);
    let tol = 1e-5f32;
    for i in 0..=n {
        for j in 0..=n {
            let x = [
                -1.0 + 2.0 * (i as f32) / (n as f32),
                -1.0 + 2.0 * (j as f32) / (n as f32),
            ];
            let za = z(a_coef, a_bias, x);
            let zb = z(b_coef, b_bias, x);

            // NECESSARY CONDITION: true sign ⇒ half-space holds (A·x+b ≤ 0).
            if za >= 0.0 {
                active_feasible += 1;
                assert!(
                    half(&ha, hab, x) <= tol,
                    "ACTIVE premise: truly-active x={x:?} (z_A={za}) violates its \
                     necessary half-space {} — direction is UNSOUND",
                    half(&ha, hab, x)
                );
            }
            if zb <= 0.0 {
                inactive_feasible += 1;
                assert!(
                    half(&hb, hbb, x) <= tol,
                    "INACTIVE premise: truly-inactive x={x:?} (z_B={zb}) violates its \
                     necessary half-space {} — direction is UNSOUND",
                    half(&hb, hbb, x)
                );
            }
            // RELAXATION (superset, not exact): with δ>0 there exist points that
            // satisfy the half-space but have the WRONG true sign (a·x+b in the
            // [ -δ, 0 ) band for active). Count them to prove the built constraint
            // is genuinely looser than the exact sign set (so this is a real
            // necessary-condition test, not a degenerate exact-affine one).
            if half(&ha, hab, x) <= 0.0 && za < 0.0 {
                active_relaxed_extra += 1;
            }
            if half(&hb, hbb, x) <= 0.0 && zb > 0.0 {
                inactive_relaxed_extra += 1;
            }
        }
    }
    assert!(
        active_feasible > 100 && inactive_feasible > 100,
        "sampling vacuous: active_feasible={active_feasible} inactive_feasible={inactive_feasible}"
    );
    assert!(
        active_relaxed_extra > 0 && inactive_relaxed_extra > 0,
        "the built half-space must be a genuine RELAXATION (superset): \
         active_extra={active_relaxed_extra} inactive_extra={inactive_relaxed_extra}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn split_offset_subtraction_rounds_toward_weaker_halfspace() {
    use crate::beta_crown::branching::GenBabConstraint;

    let split = 2.0f32.powi(-25);
    let mut history = GraphSplitHistory::new();
    history.add_genbab_constraint(
        GenBabConstraint::new("relu".to_string(), 0, split, false, 0.0).expect("finite split"),
    );
    let constraints = build_split_constraints(
        &history,
        |_, _| Some((array![0.0f32], 1.0, array![0.0f32], 1.0)),
        1,
    )
    .expect("constraint");
    // Exact b = 1 - 2^-25 lies halfway between 1 and next_down(1). Round-to-
    // nearest would choose 1, which is a stricter half-space. The stored offset
    // must instead be <= the exact expression.
    assert_eq!(constraints.b_vector[0], ny_tensor::next_down_f32(1.0));
    assert!(f64::from(constraints.b_vector[0]) <= 1.0 - f64::from(split));
}

#[ntest::timeout(10000)]
#[test]
fn split_offsets_are_outward_for_both_signs_subnormals_and_overflow() {
    use crate::beta_crown::branching::GenBabConstraint;

    // Upper branch: exact b = -u_bias+s = 1+2^-25, halfway between
    // adjacent f32s. Directed-down storage must choose 1, never next_up(1).
    let split = 2.0f32.powi(-25);
    let mut upper = GraphSplitHistory::new();
    upper.add_genbab_constraint(GenBabConstraint::new("relu".into(), 0, split, true, 0.0).unwrap());
    let c = build_split_constraints(
        &upper,
        |_, _| Some((array![0.0], 0.0, array![0.0], -1.0)),
        1,
    )
    .unwrap();
    assert_eq!(c.b_vector[0], 1.0);
    assert!(f64::from(c.b_vector[0]) <= 1.0 + f64::from(split));

    // A min-subnormal add is lost even in f64 at this exponent. The explicit
    // f64 interval widening still makes the eventual f32 no greater than the
    // exact-real expression.
    let tiny = f32::from_bits(1);
    let mut lower = GraphSplitHistory::new();
    lower
        .add_genbab_constraint(GenBabConstraint::new("relu".into(), 0, -tiny, false, 0.0).unwrap());
    let c = build_split_constraints(&lower, |_, _| Some((array![0.0], 1.0, array![0.0], 0.0)), 1)
        .unwrap();
    assert!(c.b_vector[0] <= 1.0);

    // Finite f32 sources can have an exact sum outside the finite f32 range.
    // The weak/downward side saturates at MAX and remains usable.
    let mut overflow = GraphSplitHistory::new();
    overflow.add_genbab_constraint(
        GenBabConstraint::new("relu".into(), 0, f32::MAX, true, 0.0).unwrap(),
    );
    let c = build_split_constraints(
        &overflow,
        |_, _| Some((array![0.0], 0.0, array![0.0], -f32::MAX)),
        1,
    )
    .unwrap();
    assert_eq!(c.b_vector.as_slice(), Some(&[f32::MAX][..]));

    // Negative overflow rounds to -inf and is dropped. Dropping a necessary
    // condition weakens the relaxation and is the fail-closed authority result.
    let mut negative_overflow = GraphSplitHistory::new();
    negative_overflow.add_genbab_constraint(
        GenBabConstraint::new("relu".into(), 0, f32::MAX, false, 0.0).unwrap(),
    );
    let c = build_split_constraints(
        &negative_overflow,
        |_, _| Some((array![0.0], -f32::MAX, array![0.0], 0.0)),
        1,
    )
    .unwrap();
    assert!(c.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_build_split_constraints_relu_inactive_non_contiguous_coeffs_4250() {
    use crate::beta_crown::branching::GraphNeuronConstraint;

    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu_1".to_string(),
        neuron_idx: 0,
        is_active: false,
        score: 0.0,
    });

    let constraints = build_split_constraints(
        &history,
        |name, idx| {
            if name == "relu_1" && idx == 0 {
                let lower = array![2.0_f32, 99.0, 3.0, 101.0].slice_move(s![..;2]);
                let upper = array![2.0_f32, 98.0, 3.0, 100.0].slice_move(s![..;2]);
                assert!(
                    lower.as_slice().is_none(),
                    "test setup: lower coefficients should be non-contiguous"
                );
                Some((lower, -1.0, upper, 1.0))
            } else {
                None
            }
        },
        2,
    )
    .expect("non-contiguous coefficients should still build constraints");

    assert_eq!(constraints.num_constraints, 1);
    let a_row = constraints.a_matrix.row(0);
    assert!((a_row[0] - 2.0).abs() < 1e-6);
    assert!((a_row[1] - 3.0).abs() < 1e-6);
    assert!((constraints.b_vector[0] - (-1.0)).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_sort_out_constraints_filtering() {
    let constraints = SplitConstraints {
        a_matrix: array![
            [1.0, 0.0],  // x1 + 0.5 <= 0 (never satisfied on [0,1]^2)
            [1.0, 0.0],  // x1 - 0.5 <= 0 (active on [0,1]^2)
            [-1.0, 0.0], // -x1 + 0.5 <= 0 (always satisfied on [0,1]^2 for x1 >= 0.5)
        ],
        b_vector: array![0.5, -0.5, -1.5], // Last one: -x1 - 1.5 <= 0, always true
        num_constraints: 3,
    };

    let x_l = array![0.0, 0.0];
    let x_u = array![1.0, 1.0];

    let preprocessed = sort_out_constraints(&constraints, &x_l, &x_u).expect("finite constraint");

    // First constraint: x1 + 0.5 <= 0
    // d = 0.5*1 + 0.5 = 1.0, max_violation = |1|*0.5 = 0.5
    // d - max_violation = 0.5 > 0, infeasible
    assert!(preprocessed.infeasible_mask[0]);

    // Second constraint: x1 - 0.5 <= 0
    // d = 0.5*1 - 0.5 = 0, max_violation = 0.5
    // d - max_violation = -0.5 <= 0 (feasible)
    // d + max_violation = 0.5 > 0 (not fully covered)
    // => active
    assert!(!preprocessed.infeasible_mask[1]);
    assert!(!preprocessed.fully_covered_mask[1]);
    assert_eq!(preprocessed.b_active, array![-0.5f32]);
}

#[ntest::timeout(10000)]
#[test]
fn certified_dual_is_structural_even_when_legacy_gate_is_zero() {
    ny_test_utils::env::with_serialized_env_vars(&[("NY_CLIP_DUAL_CERTIFY", "0")], || {
        // x in [0,1], constraint -x+0.5 <= 0 (x >= 0.5).
        let constraints = PreprocessedConstraints {
            a_active: array![[-1.0f32]],
            b_active: array![0.5f32],
            // d = A*x0+b = -0.5+0.5 = 0.
            d_active: array![0.0f32],
            infeasible_mask: vec![false],
            fully_covered_mask: vec![false],
        };
        let (lower, upper) = tighten_with_constraints(
            &constraints,
            &array![[1.0f32]],
            &array![0.0f32],
            &array![[1.0f32]],
            &array![0.0f32],
            &array![0.0f32],
            &array![1.0f32],
        )
        .expect("structurally certified clip");
        assert!(lower[0] <= 0.5 && lower[0] > 0.499_9, "lower={}", lower[0]);
        assert!(upper[0] >= 1.0, "upper={}", upper[0]);
    });
}

#[ntest::timeout(10000)]
#[test]
fn preprocessing_retains_original_b_when_centered_roundtrip_is_stricter() {
    let original_b = 5.275_492f32;
    let constraints = SplitConstraints {
        a_matrix: array![[-73_127_152.0f32]],
        b_vector: array![original_b],
        num_constraints: 1,
    };
    let x_l = array![-0.610_265f32];
    let x_u = array![3.389_735f32];
    let preprocessed = sort_out_constraints(&constraints, &x_l, &x_u).expect("active row");
    assert_eq!(preprocessed.a_active.nrows(), 1);
    assert_eq!(preprocessed.b_active[0], original_b);

    // The legacy centered round-trip loses the small offset to cancellation
    // and reconstructs 8.0 > original_b, which makes A*x+b<=0 strictly harder.
    let x0 = (&x_l + &x_u) / 2.0;
    let ax0: f32 = preprocessed
        .a_active
        .row(0)
        .iter()
        .zip(x0.iter())
        .map(|(a, x)| a * x)
        .sum();
    let reconstructed = preprocessed.d_active[0] - ax0;
    assert_eq!(reconstructed, 8.0);
    assert!(reconstructed > preprocessed.b_active[0]);
}

#[ntest::timeout(10000)]
#[test]
fn test_select_objective_neurons() {
    let lower = array![-1.0, 0.5, -2.0, -0.5, 1.0];
    let upper = array![1.0, 1.0, 0.5, 0.5, 2.0];
    let coeff_mag = array![1.0, 1.0, 1.0, 2.0, 1.0];

    // Unstable neurons: 0 (l=-1, u=1), 2 (l=-2, u=0.5), 3 (l=-0.5, u=0.5)
    // 1 and 4 are stable (l >= 0 or u <= 0 is NOT satisfied for 1, but u <= 0 for neuron 2)
    // Actually: neuron 2 has u=0.5 > 0, so it's unstable too

    let selected = select_objective_neurons(&lower, &upper, &coeff_mag, 2);

    // Should select 2 highest-scoring unstable neurons
    assert!(selected.len() <= 2);
    // All selected should be unstable
    for &idx in &selected {
        assert!(lower[idx] < 0.0 && upper[idx] > 0.0);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_select_objective_neurons_uses_coefficients() {
    let lower = array![-1.0, -1.0, -1.0];
    let upper = array![1.0, 1.0, 1.0];
    let coeff_mag = array![1.0, 10.0, 2.0];

    // All neurons have the same intercept, so coefficient magnitudes should decide.
    let selected = select_objective_neurons(&lower, &upper, &coeff_mag, 1);

    assert_eq!(selected.len(), 1);
    assert_eq!(selected, vec![1]);
}

#[ntest::timeout(10000)]
#[test]
fn test_merge_bounds_no_inversion() {
    let original_lower = array![-1.0, -2.0, -3.0];
    let original_upper = array![1.0, 2.0, 3.0];
    let tightened_lower = array![-0.5, -1.5]; // For indices [0, 2]
    let tightened_upper = array![0.5, 2.5];
    let selected = vec![0, 2];

    let (lower, upper) = merge_bounds(
        &original_lower,
        &original_upper,
        &tightened_lower,
        &tightened_upper,
        &selected,
    );

    // Index 0: tightened from [-1, 1] to [-0.5, 0.5]
    assert!((lower[0] - (-0.5)).abs() < 1e-6);
    assert!((upper[0] - 0.5).abs() < 1e-6);

    // Index 1: unchanged
    assert!((lower[1] - (-2.0)).abs() < 1e-6);
    assert!((upper[1] - 2.0).abs() < 1e-6);

    // Index 2: tightened from [-3, 3] to [-1.5, 2.5]
    assert!((lower[2] - (-1.5)).abs() < 1e-6);
    assert!((upper[2] - 2.5).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_merge_bounds_with_inversion_fallback() {
    let original_lower = array![-1.0];
    let original_upper = array![1.0];
    let tightened_lower = array![0.5]; // Would invert
    let tightened_upper = array![0.2]; // 0.5 > 0.2, inverted
    let selected = vec![0];

    let (lower, upper) = merge_bounds(
        &original_lower,
        &original_upper,
        &tightened_lower,
        &tightened_upper,
        &selected,
    );

    // Should keep original bounds due to inversion
    assert!((lower[0] - (-1.0)).abs() < 1e-6);
    assert!((upper[0] - 1.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_unconstrained_bounds() {
    // Test the unconstrained (fallback) bound computation
    let obj_lower_a = array![[1.0, 1.0]]; // z_l = x1 + x2
    let obj_lower_bias = array![0.0];
    let obj_upper_a = array![[1.0, 1.0]]; // z_u = x1 + x2
    let obj_upper_bias = array![0.0];
    let x_l = array![0.0, 0.0];
    let x_u = array![1.0, 1.0];

    let (lower, upper) = compute_unconstrained_bounds(
        &obj_lower_a,
        &obj_lower_bias,
        &obj_upper_a,
        &obj_upper_bias,
        &x_l,
        &x_u,
    )
    .expect("finite constraint");

    // For z = x1 + x2 on [0,1]^2:
    // x0 = [0.5, 0.5], eps = [0.5, 0.5]
    // lower = 0.5 + 0.5 - (0.5 + 0.5) = 0
    // upper = 0.5 + 0.5 + (0.5 + 0.5) = 2
    assert!((lower[0] - 0.0).abs() < 1e-6);
    assert!((upper[0] - 2.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_build_split_constraints_genbab_upper_branch() {
    // Test GenBaB constraint with non-zero split point (upper branch: z >= s)
    use crate::beta_crown::branching::GenBabConstraint;

    let mut history = GraphSplitHistory::new();
    history.add_genbab_constraint(
        GenBabConstraint::new(
            "gelu_1".to_string(),
            0,
            0.5,  // split at 0.5
            true, // upper branch (z >= 0.5)
            0.0,
        )
        .expect("finite constraint"),
    );

    // Mock linear bounds: z = 2*x1 + 3*x2 + 1 (upper bound)
    let constraints = build_split_constraints(
        &history,
        |name, idx| {
            if name == "gelu_1" && idx == 0 {
                Some((
                    array![2.0, 3.0], // lA
                    0.0,              // lbias
                    array![2.0, 3.0], // uA
                    1.0,              // ubias
                ))
            } else {
                None
            }
        },
        2,
    )
    .expect("finite constraint");

    assert_eq!(constraints.num_constraints, 1);

    // Upper branch (z >= 0.5) uses upper bound:
    // -uA·x + (-ubias + s) <= 0
    // => -2*x1 - 3*x2 - 1 + 0.5 <= 0
    // => -2*x1 - 3*x2 - 0.5 <= 0
    let a_row = constraints.a_matrix.row(0);
    assert!((a_row[0] - (-2.0)).abs() < 1e-6);
    assert!((a_row[1] - (-3.0)).abs() < 1e-6);
    assert!((constraints.b_vector[0] - (-0.5)).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_build_split_constraints_genbab_lower_branch() {
    // Test GenBaB constraint with non-zero split point (lower branch: z <= s)
    use crate::beta_crown::branching::GenBabConstraint;

    let mut history = GraphSplitHistory::new();
    history.add_genbab_constraint(
        GenBabConstraint::new(
            "gelu_1".to_string(),
            0,
            0.5,   // split at 0.5
            false, // lower branch (z <= 0.5)
            0.0,
        )
        .expect("finite constraint"),
    );

    // Mock linear bounds: z = 2*x1 + 3*x2 (lower bound)
    let constraints = build_split_constraints(
        &history,
        |name, idx| {
            if name == "gelu_1" && idx == 0 {
                Some((
                    array![2.0, 3.0], // lA
                    0.0,              // lbias
                    array![2.0, 3.0], // uA
                    1.0,              // ubias
                ))
            } else {
                None
            }
        },
        2,
    )
    .expect("finite constraint");

    assert_eq!(constraints.num_constraints, 1);

    // Lower branch (z <= 0.5) uses lower bound:
    // lA·x + (lbias - s) <= 0
    // => 2*x1 + 3*x2 + 0 - 0.5 <= 0
    // => 2*x1 + 3*x2 - 0.5 <= 0
    let a_row = constraints.a_matrix.row(0);
    assert!((a_row[0] - 2.0).abs() < 1e-6);
    assert!((a_row[1] - 3.0).abs() < 1e-6);
    assert!((constraints.b_vector[0] - (-0.5)).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_sort_out_constraints_fully_covered() {
    // Test that fully covered constraints are detected correctly
    let constraints = SplitConstraints {
        a_matrix: array![
            [-1.0, 0.0], // -x1 - 2 <= 0, always true for x1 >= 0
        ],
        b_vector: array![-2.0], // d = -0.5*1 - 2 = -2.5, max_violation = 0.5
        num_constraints: 1,
    };

    let x_l = array![0.0, 0.0];
    let x_u = array![1.0, 1.0];

    let preprocessed = sort_out_constraints(&constraints, &x_l, &x_u).expect("finite constraint");

    // -x1 - 2 <= 0 => -x1 <= 2, always true for x1 in [0, 1]
    // d = -0.5 - 2 = -2.5, max_violation = |−1|*0.5 = 0.5
    // d + max_violation = -2.5 + 0.5 = -2 < 0 => fully covered
    assert!(preprocessed.fully_covered_mask[0]);
    assert!(!preprocessed.infeasible_mask[0]);
}

#[ntest::timeout(10000)]
#[test]
fn test_sort_out_constraints_active() {
    // Test that active (non-trivial) constraints are detected
    let constraints = SplitConstraints {
        a_matrix: array![
            [1.0, 0.0], // x1 <= 0.5
        ],
        b_vector: array![-0.5],
        num_constraints: 1,
    };

    let x_l = array![0.0, 0.0];
    let x_u = array![1.0, 1.0];

    let preprocessed = sort_out_constraints(&constraints, &x_l, &x_u).expect("finite constraint");

    // x1 - 0.5 <= 0
    // d = 0.5 - 0.5 = 0, max_violation = 0.5
    // d - max_violation = -0.5 <= 0 (not infeasible)
    // d + max_violation = 0.5 > 0 (not fully covered)
    // => active constraint
    assert!(!preprocessed.infeasible_mask[0]);
    assert!(!preprocessed.fully_covered_mask[0]);
}

#[ntest::timeout(10000)]
#[test]
fn test_tighten_with_constraints_no_active() {
    // Test with no active constraints
    let constraints = PreprocessedConstraints {
        a_active: Array2::zeros((0, 2)),
        b_active: Array1::zeros(0),
        d_active: Array1::zeros(0),
        infeasible_mask: vec![],
        fully_covered_mask: vec![],
    };

    let obj_lower_a = array![[1.0, 1.0]]; // z = x1 + x2
    let obj_lower_bias = array![0.0];
    let obj_upper_a = array![[1.0, 1.0]];
    let obj_upper_bias = array![0.0];
    let x_l = array![0.0, 0.0];
    let x_u = array![1.0, 1.0];

    let (lower, upper) = tighten_with_constraints(
        &constraints,
        &obj_lower_a,
        &obj_lower_bias,
        &obj_upper_a,
        &obj_upper_bias,
        &x_l,
        &x_u,
    )
    .expect("finite constraint");

    // With no active constraints, should return unconstrained bounds
    // min z = x1 + x2 on [0, 1]^2 = 0
    // max z = x1 + x2 on [0, 1]^2 = 2
    assert!(
        (lower[0] - 0.0).abs() < 1e-5,
        "Expected lower ~0, got {}",
        lower[0]
    );
    assert!(
        (upper[0] - 2.0).abs() < 1e-5,
        "Expected upper ~2, got {}",
        upper[0]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_select_objective_neurons_coefficient_weighting() {
    // Test that coefficient magnitudes influence neuron selection order
    // All neurons have same intercept score, so coefficients determine ranking

    // Two unstable neurons with identical intercept characteristics
    let lower = array![-1.0, -1.0];
    let upper = array![1.0, 1.0];

    // With uniform coefficients, both neurons have equal score
    let uniform_coeffs = array![1.0, 1.0];
    let selected_uniform = select_objective_neurons(&lower, &upper, &uniform_coeffs, 2);
    assert_eq!(
        selected_uniform.len(),
        2,
        "Should select both with uniform coefficients"
    );

    // With weighted coefficients, neuron 1 should be ranked higher
    let weighted_coeffs = array![1.0, 10.0];
    let selected_weighted = select_objective_neurons(&lower, &upper, &weighted_coeffs, 1);
    assert_eq!(selected_weighted.len(), 1);
    assert_eq!(
        selected_weighted[0], 1,
        "Neuron with higher coefficient magnitude should be selected first"
    );

    // Verify the opposite: if neuron 0 has higher coefficient, it should be selected
    let reverse_coeffs = array![10.0, 1.0];
    let selected_reverse = select_objective_neurons(&lower, &upper, &reverse_coeffs, 1);
    assert_eq!(
        selected_reverse[0], 0,
        "Neuron 0 should be selected with higher coefficient"
    );
}
