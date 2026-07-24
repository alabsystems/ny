// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for the arena-based constraint store.

use super::*;
use crate::beta_crown::branching::{LayerRef, NeuronSplit, SplitHistory};
use crate::beta_crown::NeuronConstraint;
use crate::LinearBounds;
use ndarray::{Array1, Array2};

#[ntest::timeout(5000)]
#[test]
fn test_arena_constraint_store_basic() {
    let mut store = ArenaConstraintStore::new();
    assert!(store.is_empty(), "new store should be empty");

    // Add a constraint: x[0] + 2*x[1] <= 3.0
    store
        .add_constraint(
            &[0, 1],
            &[1.0, 2.0],
            3.0,
            ConstraintSense::Le,
            ConstraintOrigin::Split,
        )
        .unwrap();

    assert_eq!(store.len(), 1);
    assert_eq!(store.total_terms(), 2);

    let c = store.get(0).unwrap();
    assert_eq!(c.indices, &[0, 1]);
    assert_eq!(c.coeffs, &[1.0, 2.0]);
    assert_eq!(c.bias, 3.0);
    assert_eq!(c.sense, ConstraintSense::Le);
    assert_eq!(c.origin, ConstraintOrigin::Split);
}

#[ntest::timeout(5000)]
#[test]
fn test_arena_push_pop_scope() {
    let mut store = ArenaConstraintStore::new();

    // Add initial constraint
    store
        .add_constraint(
            &[0],
            &[1.0],
            0.0,
            ConstraintSense::Le,
            ConstraintOrigin::Split,
        )
        .unwrap();
    assert_eq!(store.len(), 1);

    // Push scope
    store.push_scope();
    assert_eq!(store.scope_depth(), 1);

    // Add more constraints
    store
        .add_constraint(
            &[1],
            &[1.0],
            0.0,
            ConstraintSense::Le,
            ConstraintOrigin::Split,
        )
        .unwrap();
    store
        .add_constraint(
            &[2],
            &[-1.0],
            0.0,
            ConstraintSense::Le,
            ConstraintOrigin::Split,
        )
        .unwrap();
    assert_eq!(store.len(), 3);

    // Pop scope - should remove last 2 constraints
    assert!(store.pop_scope(), "pop_scope should succeed at depth 1");
    assert_eq!(store.len(), 1);
    assert_eq!(store.scope_depth(), 0);

    // Pop again - nothing to pop
    assert!(
        !store.pop_scope(),
        "pop_scope should return false at depth 0"
    );
    assert_eq!(store.len(), 1);
}

#[ntest::timeout(5000)]
#[test]
fn test_relu_split_encoding() {
    let mut store = ArenaConstraintStore::new();

    // Active split: z >= 0 → -z <= 0
    store.add_relu_split(5, true).unwrap();
    let c = store.get(0).unwrap();
    assert_eq!(c.indices, &[5]);
    assert_eq!(c.coeffs, &[-1.0]);
    assert_eq!(c.bias, 0.0);

    // Inactive split: z <= 0
    store.add_relu_split(7, false).unwrap();
    let c = store.get(1).unwrap();
    assert_eq!(c.indices, &[7]);
    assert_eq!(c.coeffs, &[1.0]);
    assert_eq!(c.bias, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_genbab_split_encoding() {
    let mut store = ArenaConstraintStore::new();

    // Lower branch: z <= 0.5
    store.add_genbab_split(3, 0.5, false).unwrap();
    let c = store.get(0).unwrap();
    assert_eq!(c.indices, &[3]);
    assert_eq!(c.coeffs, &[1.0]);
    assert_eq!(c.bias, 0.5);

    // Upper branch: z >= 0.5 → -z <= -0.5
    store.add_genbab_split(4, 0.5, true).unwrap();
    let c = store.get(1).unwrap();
    assert_eq!(c.indices, &[4]);
    assert_eq!(c.coeffs, &[-1.0]);
    assert_eq!(c.bias, -0.5);
}

#[ntest::timeout(5000)]
#[test]
fn test_domain_constraint_store_child() {
    let mut parent = DomainConstraintStore::new();

    // Add base constraints
    parent
        .delta_mut()
        .add_constraint(
            &[0],
            &[1.0],
            0.0,
            ConstraintSense::Le,
            ConstraintOrigin::Output,
        )
        .unwrap();
    parent.delta_mut().add_relu_split(1, true).unwrap();
    assert_eq!(parent.len(), 2);

    // Create child
    let child = parent.create_child().unwrap();
    assert_eq!(child.base_len(), 2); // Inherited from parent
    assert_eq!(child.delta_len(), 0); // Empty delta
    assert_eq!(child.len(), 2);

    // Parent unchanged
    assert_eq!(parent.len(), 2);
}

#[ntest::timeout(5000)]
#[test]
fn test_convert_from_split_history() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 5,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 3,
        is_active: false,
        score: 0.0,
    });

    let mut store = ArenaConstraintStore::new();

    // Simple mapping: layer * 100 + neuron
    let count = store
        .add_from_split_history(&history, |layer, neuron| (layer * 100 + neuron) as u32)
        .unwrap();

    assert_eq!(count, 2);
    assert_eq!(store.len(), 2);

    // First constraint: active split on var 5 (0*100+5)
    let c0 = store.get(0).unwrap();
    assert_eq!(c0.indices, &[5]);
    assert_eq!(c0.coeffs, &[-1.0]); // Active: -z <= 0

    // Second constraint: inactive split on var 103 (1*100+3)
    let c1 = store.get(1).unwrap();
    assert_eq!(c1.indices, &[103]);
    assert_eq!(c1.coeffs, &[1.0]); // Inactive: z <= 0
}

#[ntest::timeout(5000)]
#[test]
fn test_header_size() {
    // Verify header is 16 bytes for GPU alignment
    assert_eq!(size_of::<ConstraintHeader>(), 16);
}

#[ntest::timeout(5000)]
#[test]
fn test_invalid_sense_byte_returns_error() {
    // A corrupted sense byte (e.g., from GPU buffer corruption) must be
    // detected rather than silently reinterpreted (#2261).
    let mut header = ConstraintHeader::new(0, 0, 0.0, ConstraintSense::Le, ConstraintOrigin::Split);
    header.sense = 5; // Corrupt the sense byte
    let err = header.sense().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("invalid sense byte"),
        "expected sense byte error, got: {msg}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_invalid_origin_byte_returns_error() {
    let mut header = ConstraintHeader::new(0, 0, 0.0, ConstraintSense::Le, ConstraintOrigin::Split);
    header.origin = 99; // Corrupt the origin byte
    let err = header.origin().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("invalid origin byte"),
        "expected origin byte error, got: {msg}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_get_returns_none_on_corrupted_data_range() {
    // A header with data_start/data_len pointing beyond the arena should
    // return None rather than panicking on out-of-bounds slice (#2261).
    let mut bad_store = ArenaConstraintStore::new();
    bad_store
        .add_constraint(
            &[0],
            &[1.0],
            0.0,
            ConstraintSense::Le,
            ConstraintOrigin::Split,
        )
        .expect("add_constraint should succeed");

    assert_eq!(bad_store.len(), 1);
    assert!(
        bad_store.get(0).is_some(),
        "get(0) should return Some before corruption"
    );

    // Corrupt the existing header's data range to point beyond the arenas.
    // Uses safe #[cfg(test)] helper instead of UB raw-pointer cast (#2754).
    bad_store.corrupt_header_data_range(0, 10, 5);

    // Corrupted header is rejected by indexed lookup.
    assert!(
        bad_store.get(0).is_none(),
        "get(0) should return None after data range corruption"
    );

    // Iterator should skip the corrupted row rather than panic.
    let rows: Vec<_> = bad_store.iter().collect();
    assert!(
        rows.is_empty(),
        "iter() should skip corrupted data range row"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_iterator() {
    let mut store = ArenaConstraintStore::new();
    store
        .add_constraint(
            &[0],
            &[1.0],
            1.0,
            ConstraintSense::Le,
            ConstraintOrigin::Split,
        )
        .unwrap();
    store
        .add_constraint(
            &[1, 2],
            &[2.0, 3.0],
            2.0,
            ConstraintSense::Ge,
            ConstraintOrigin::Output,
        )
        .unwrap();
    store
        .add_constraint(
            &[3],
            &[4.0],
            3.0,
            ConstraintSense::Le,
            ConstraintOrigin::BoundProp,
        )
        .unwrap();

    let constraints: Vec<_> = store.iter().collect();
    assert_eq!(constraints.len(), 3);

    assert_eq!(constraints[0].bias, 1.0);
    assert_eq!(constraints[1].bias, 2.0);
    assert_eq!(constraints[1].indices.len(), 2);
    assert_eq!(constraints[2].origin, ConstraintOrigin::BoundProp);
}

// =========================================================================
// Tests for add_split_with_crown_bounds
// Design doc: designs/2026-01-29-genbab-split-to-linear.md
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_relu_active() {
    // Example 1 from design doc: ReLU Active Split (z >= 0)
    // CROWN lower bound: z >= [0.5, -0.3]·x + 0.1
    // Expected: a = [-0.5, 0.3], b = -0.1
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::relu_active(LayerRef::Index(1), 0);
    let crown_bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![0.5, -0.3]).unwrap(),
        lower_b: Array1::from_vec(vec![0.1]),
        upper_a: Array2::from_shape_vec((1, 2), vec![0.7, 0.2]).unwrap(),
        upper_b: Array1::from_vec(vec![0.4]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let count = store
        .add_split_with_crown_bounds(&split, &crown_bounds, 0)
        .unwrap();
    assert_eq!(count, 1);
    assert_eq!(store.len(), 1);

    let c = store.get(0).unwrap();
    assert_eq!(c.indices, &[0, 1]);
    assert_eq!(c.coeffs, &[-0.5, 0.3]); // Negated: -lA
    assert!(
        (c.bias - (-0.1)).abs() < 1e-6,
        "relu active bias: expected -0.1, got {}",
        c.bias
    ); // s - lb = 0 - 0.1 = -0.1
    assert_eq!(c.sense, ConstraintSense::Le);
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_relu_inactive() {
    // Example 2 from design doc: ReLU Inactive Split (z <= 0)
    // CROWN upper bound: z <= [0.7, 0.2]·x + 0.4
    // Expected: a = [0.7, 0.2], b = 0.4
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::relu_inactive(LayerRef::Index(1), 0);
    let crown_bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![0.5, -0.3]).unwrap(),
        lower_b: Array1::from_vec(vec![0.1]),
        upper_a: Array2::from_shape_vec((1, 2), vec![0.7, 0.2]).unwrap(),
        upper_b: Array1::from_vec(vec![0.4]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let count = store
        .add_split_with_crown_bounds(&split, &crown_bounds, 0)
        .unwrap();
    assert_eq!(count, 1);
    assert_eq!(store.len(), 1);

    let c = store.get(0).unwrap();
    assert_eq!(c.indices, &[0, 1]);
    assert_eq!(c.coeffs, &[0.7, 0.2]); // Direct: uA
    assert!(
        (c.bias - 0.4).abs() < 1e-6,
        "relu inactive bias: expected 0.4, got {}",
        c.bias
    ); // ub - s = 0.4 - 0 = 0.4
    assert_eq!(c.sense, ConstraintSense::Le);
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_genbab_lower() {
    // Example 3 from design doc: GenBaB Lower branch (z <= 0.5)
    // CROWN upper bound: z <= [0.4, -0.6]·x + 0.8
    // Expected: a = [0.4, -0.6], b = 0.3
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::at_point(LayerRef::Index(1), 0, 0.5, false).unwrap(); // lower branch
    let crown_bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![0.3, 0.1]).unwrap(),
        lower_b: Array1::from_vec(vec![0.2]),
        upper_a: Array2::from_shape_vec((1, 2), vec![0.4, -0.6]).unwrap(),
        upper_b: Array1::from_vec(vec![0.8]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let count = store
        .add_split_with_crown_bounds(&split, &crown_bounds, 0)
        .unwrap();
    assert_eq!(count, 1);
    assert_eq!(store.len(), 1);

    let c = store.get(0).unwrap();
    assert_eq!(c.indices, &[0, 1]);
    assert_eq!(c.coeffs, &[0.4, -0.6]); // Direct: uA
    assert!(
        (c.bias - 0.3).abs() < 1e-6,
        "genbab lower bias: expected 0.3, got {}",
        c.bias
    ); // ub - s = 0.8 - 0.5 = 0.3
    assert_eq!(c.sense, ConstraintSense::Le);
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_genbab_upper() {
    // Example 3 from design doc: GenBaB Upper branch (z >= 0.5)
    // CROWN lower bound: z >= [0.3, 0.1]·x + 0.2
    // Expected: a = [-0.3, -0.1], b = 0.3
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::at_point(LayerRef::Index(1), 0, 0.5, true).unwrap(); // upper branch
    let crown_bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![0.3, 0.1]).unwrap(),
        lower_b: Array1::from_vec(vec![0.2]),
        upper_a: Array2::from_shape_vec((1, 2), vec![0.4, -0.6]).unwrap(),
        upper_b: Array1::from_vec(vec![0.8]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let count = store
        .add_split_with_crown_bounds(&split, &crown_bounds, 0)
        .unwrap();
    assert_eq!(count, 1);
    assert_eq!(store.len(), 1);

    let c = store.get(0).unwrap();
    assert_eq!(c.indices, &[0, 1]);
    assert_eq!(c.coeffs, &[-0.3, -0.1]); // Negated: -lA
    assert!(
        (c.bias - 0.3).abs() < 1e-6,
        "genbab upper bias: expected 0.3, got {}",
        c.bias
    ); // s - lb = 0.5 - 0.2 = 0.3
    assert_eq!(c.sense, ConstraintSense::Le);
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_both_bounds() {
    // Split with both lower and upper bounds (e.g., 0.3 <= z <= 0.7)
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit {
        layer: LayerRef::Index(1),
        neuron_idx: 0,
        lower_bound: Some(0.3),
        upper_bound: Some(0.7),
        score: 0.0,
        input_index: None,
        norm_inv_rms_window: None,
    };
    let crown_bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap(),
        lower_b: Array1::from_vec(vec![0.1]),
        upper_a: Array2::from_shape_vec((1, 2), vec![1.5, 2.5]).unwrap(),
        upper_b: Array1::from_vec(vec![0.9]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let count = store
        .add_split_with_crown_bounds(&split, &crown_bounds, 0)
        .unwrap();
    assert_eq!(count, 2); // Both constraints added
    assert_eq!(store.len(), 2);

    // Lower bound constraint: z >= 0.3 → -lA·x + (0.3 - 0.1) ≤ 0
    let c0 = store.get(0).unwrap();
    assert_eq!(c0.coeffs, &[-1.0, -2.0]);
    assert!(
        (c0.bias - 0.2).abs() < 1e-6,
        "both-bounds lower bias: expected 0.2, got {}",
        c0.bias
    );

    // Upper bound constraint: z <= 0.7 → uA·x + (0.9 - 0.7) ≤ 0
    let c1 = store.get(1).unwrap();
    assert_eq!(c1.coeffs, &[1.5, 2.5]);
    assert!(
        (c1.bias - 0.2).abs() < 1e-6,
        "both-bounds upper bias: expected 0.2, got {}",
        c1.bias
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_all_zero_coeffs_infeasible_returns_error() {
    // When all CROWN coefficients are near-zero and the bias makes the
    // constraint infeasible, the code should return an error (#2260).
    // Example: CROWN says z >= 0*x - 1.0, split says z >= 0.0.
    // Constraint: 0*x + (0 - (-1)) <= 0 => 1 <= 0 (infeasible).
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::relu_active(LayerRef::Index(1), 0); // z >= 0
    let crown_bounds = LinearBounds {
        // All-zero coefficients and positive residual bias: bias = s - lb = 1.0 > 0.
        lower_a: Array2::from_shape_vec((1, 2), vec![1e-15, 1e-15]).expect("shape ok"),
        lower_b: Array1::from_vec(vec![-1.0]), // lb = -1.0, s = 0.0, bias = 0 - (-1) = 1.0
        upper_a: Array2::from_shape_vec((1, 2), vec![0.7, 0.2]).expect("shape ok"),
        upper_b: Array1::from_vec(vec![0.4]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = store.add_split_with_crown_bounds(&split, &crown_bounds, 0);
    assert!(
        result.is_err(),
        "All-zero coefficients with positive bias should be infeasible"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("infeasible"),
        "Error should mention infeasibility: {}",
        err_msg
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_all_zero_coeffs_trivially_satisfied_drops_safely() {
    // When all CROWN coefficients are near-zero and the bias is ≤ 0,
    // the constraint is trivially satisfied and should be safely dropped.
    // Example: CROWN says z >= 0*x + 1.0, split says z >= 0.
    // bias = 0 - 1.0 = -1.0 ≤ 0 → trivially satisfied.
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::relu_active(LayerRef::Index(1), 0); // z >= 0
    let crown_bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![1e-15, 1e-15]).expect("shape ok"),
        lower_b: Array1::from_vec(vec![1.0]), // lb = 1.0, s = 0.0, bias = 0 - 1 = -1.0 ≤ 0
        upper_a: Array2::from_shape_vec((1, 2), vec![0.7, 0.2]).expect("shape ok"),
        upper_b: Array1::from_vec(vec![0.4]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let count = store
        .add_split_with_crown_bounds(&split, &crown_bounds, 0)
        .expect("trivially satisfied constraint should not error");
    assert_eq!(count, 0, "Trivially satisfied constraint should be dropped");
    assert!(store.is_empty(), "No constraints should be stored");
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_upper_all_zero_coeffs_infeasible_returns_error() {
    // Upper-bound branch (z <= s): if all uA coefficients are near-zero and
    // ub - s > 0, then the constraint reduces to an infeasible constant.
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::relu_inactive(LayerRef::Index(1), 0); // z <= 0
    let crown_bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![0.3, -0.2]).expect("shape ok"),
        lower_b: Array1::from_vec(vec![0.1]),
        upper_a: Array2::from_shape_vec((1, 2), vec![1e-15, -1e-15]).expect("shape ok"),
        upper_b: Array1::from_vec(vec![1.0]), // ub = 1.0, s = 0.0, bias = 1.0 > 0
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = store.add_split_with_crown_bounds(&split, &crown_bounds, 0);
    assert!(
        result.is_err(),
        "All-zero upper coefficients with positive bias should be infeasible"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("infeasible"),
        "Error should mention infeasibility: {}",
        err_msg
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_upper_all_zero_coeffs_trivially_satisfied_drops_safely() {
    // Upper-bound branch (z <= s): if all uA coefficients are near-zero and
    // ub - s <= 0, the constraint is trivially satisfied and can be dropped.
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::relu_inactive(LayerRef::Index(1), 0); // z <= 0
    let crown_bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![0.3, -0.2]).expect("shape ok"),
        lower_b: Array1::from_vec(vec![0.1]),
        upper_a: Array2::from_shape_vec((1, 2), vec![1e-15, -1e-15]).expect("shape ok"),
        upper_b: Array1::from_vec(vec![-1.0]), // ub = -1.0, s = 0.0, bias = -1.0 <= 0
        lower_a_err: None,
        upper_a_err: None,
    };

    let count = store
        .add_split_with_crown_bounds(&split, &crown_bounds, 0)
        .expect("trivially satisfied upper-bound constraint should not error");
    assert_eq!(
        count, 0,
        "Trivially satisfied upper-bound constraint should be dropped"
    );
    assert!(store.is_empty(), "No constraints should be stored");
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_sparse_filtering() {
    // Test that near-zero coefficients are filtered out
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::relu_active(LayerRef::Index(1), 0);
    let crown_bounds = LinearBounds {
        // Only x[1] has significant coefficient, x[0] is ~0
        lower_a: Array2::from_shape_vec((1, 3), vec![1e-12, 0.5, 1e-15]).unwrap(),
        lower_b: Array1::from_vec(vec![0.1]),
        upper_a: Array2::from_shape_vec((1, 3), vec![0.7, 1e-11, 0.2]).unwrap(),
        upper_b: Array1::from_vec(vec![0.4]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let count = store
        .add_split_with_crown_bounds(&split, &crown_bounds, 0)
        .unwrap();
    assert_eq!(count, 1);

    let c = store.get(0).unwrap();
    // Only index 1 should be present (others filtered)
    assert_eq!(c.indices, &[1]);
    assert_eq!(c.coeffs, &[-0.5]); // Negated
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_multi_neuron() {
    // Test selecting specific neuron row from multi-neuron bounds
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::relu_active(LayerRef::Index(1), 2);
    let crown_bounds = LinearBounds {
        // 3 neurons, 2 inputs
        lower_a: Array2::from_shape_vec(
            (3, 2),
            vec![
                0.1, 0.2, // neuron 0
                0.3, 0.4, // neuron 1
                0.5, 0.6, // neuron 2 (this one)
            ],
        )
        .unwrap(),
        lower_b: Array1::from_vec(vec![0.01, 0.02, 0.03]),
        upper_a: Array2::from_shape_vec((3, 2), vec![0.7, 0.8, 0.9, 1.0, 1.1, 1.2]).unwrap(),
        upper_b: Array1::from_vec(vec![0.07, 0.08, 0.09]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let count = store
        .add_split_with_crown_bounds(&split, &crown_bounds, 2)
        .unwrap();
    assert_eq!(count, 1);

    let c = store.get(0).unwrap();
    assert_eq!(c.coeffs, &[-0.5, -0.6]); // Row 2 negated
    assert!(
        (c.bias - (-0.03)).abs() < 1e-6,
        "multi-neuron row 2 bias: expected -0.03, got {}",
        c.bias
    ); // s - lb = 0 - 0.03
}

// =========================================================================
// NaN/Inf rejection tests (#2259)
// ArenaConstraintStore must reject NaN/Inf in bias, coefficients, and
// split points to prevent meaningless constraints from corrupting BaB.
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_add_constraint_rejects_nan_bias() {
    let mut store = ArenaConstraintStore::new();
    let result = store.add_constraint(
        &[0, 1],
        &[1.0, -1.0],
        f32::NAN,
        ConstraintSense::Le,
        ConstraintOrigin::Split,
    );
    assert!(result.is_err(), "NaN bias should be rejected");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("NaN") || err.contains("bias"),
        "Error should mention NaN or bias: {}",
        err
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_add_constraint_rejects_inf_bias() {
    let mut store = ArenaConstraintStore::new();
    let result = store.add_constraint(
        &[0],
        &[1.0],
        f32::INFINITY,
        ConstraintSense::Le,
        ConstraintOrigin::Split,
    );
    assert!(result.is_err(), "Inf bias should be rejected");

    // Also test negative infinity
    let result2 = store.add_constraint(
        &[0],
        &[1.0],
        f32::NEG_INFINITY,
        ConstraintSense::Le,
        ConstraintOrigin::Split,
    );
    assert!(result2.is_err(), "Negative Inf bias should be rejected");
}

#[ntest::timeout(5000)]
#[test]
fn test_add_constraint_rejects_nan_coefficient() {
    let mut store = ArenaConstraintStore::new();
    let result = store.add_constraint(
        &[0, 1, 2],
        &[1.0, f32::NAN, -0.5],
        0.0,
        ConstraintSense::Le,
        ConstraintOrigin::Split,
    );
    assert!(result.is_err(), "NaN coefficient should be rejected");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("coefficient"),
        "Error should mention coefficient: {}",
        err
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_add_constraint_rejects_inf_coefficient() {
    let mut store = ArenaConstraintStore::new();
    let result = store.add_constraint(
        &[0, 1],
        &[f32::INFINITY, 1.0],
        0.0,
        ConstraintSense::Le,
        ConstraintOrigin::Split,
    );
    assert!(result.is_err(), "Inf coefficient should be rejected");
}

#[ntest::timeout(5000)]
#[test]
fn test_genbab_split_rejects_nan_split_point() {
    let mut store = ArenaConstraintStore::new();
    let result = store.add_genbab_split(5, f32::NAN, true);
    assert!(result.is_err(), "NaN split_point should be rejected");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("split_point") || err.contains("NaN"),
        "Error should mention split_point or NaN: {}",
        err
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_genbab_split_rejects_inf_split_point() {
    let mut store = ArenaConstraintStore::new();
    let result = store.add_genbab_split(3, f32::INFINITY, false);
    assert!(result.is_err(), "Inf split_point should be rejected");

    let result2 = store.add_genbab_split(3, f32::NEG_INFINITY, true);
    assert!(
        result2.is_err(),
        "Negative Inf split_point should be rejected"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_rejects_nan_lower_bound() {
    // CROWN lower_b containing NaN should be caught before constraint creation.
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::relu_active(LayerRef::Index(1), 0); // z >= 0
    let crown_bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![0.5, -0.3]).expect("shape ok"),
        lower_b: Array1::from_vec(vec![f32::NAN]), // NaN in CROWN lower bound
        upper_a: Array2::from_shape_vec((1, 2), vec![0.7, 0.2]).expect("shape ok"),
        upper_b: Array1::from_vec(vec![0.4]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = store.add_split_with_crown_bounds(&split, &crown_bounds, 0);
    assert!(result.is_err(), "NaN in CROWN lower_b should be rejected");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("lower_b") || err.contains("NaN"),
        "Error should mention lower_b or NaN: {}",
        err
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_rejects_nan_upper_bound() {
    // CROWN upper_b containing NaN should be caught before constraint creation.
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::relu_inactive(LayerRef::Index(1), 0); // z <= 0
    let crown_bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![0.5, -0.3]).expect("shape ok"),
        lower_b: Array1::from_vec(vec![0.1]),
        upper_a: Array2::from_shape_vec((1, 2), vec![0.7, 0.2]).expect("shape ok"),
        upper_b: Array1::from_vec(vec![f32::NAN]), // NaN in CROWN upper bound
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = store.add_split_with_crown_bounds(&split, &crown_bounds, 0);
    assert!(result.is_err(), "NaN in CROWN upper_b should be rejected");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("upper_b") || err.contains("NaN"),
        "Error should mention upper_b or NaN: {}",
        err
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_split_crown_rejects_inf_crown_coefficient() {
    // Inf in CROWN coefficient row should be caught by add_constraint().
    // Inf coefficients pass the abs() > EPSILON filter and reach add_constraint().
    let mut store = ArenaConstraintStore::new();

    let split = NeuronSplit::relu_active(LayerRef::Index(1), 0);
    let crown_bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![f32::INFINITY, -0.3]).expect("shape ok"),
        lower_b: Array1::from_vec(vec![0.1]),
        upper_a: Array2::from_shape_vec((1, 2), vec![0.7, 0.2]).expect("shape ok"),
        upper_b: Array1::from_vec(vec![0.4]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = store.add_split_with_crown_bounds(&split, &crown_bounds, 0);
    assert!(
        result.is_err(),
        "Inf in CROWN coefficient should be rejected"
    );
}

/// Regression test for #2981 Slice 2: corrupted sense byte causes get() to
/// return None (constraint silently dropped). Before the fix, `.ok()?` silently
/// converted the decode error to None. After the fix, a warn!() is emitted
/// (not tested here since tracing subscriber setup is complex), but the
/// functional behavior remains: get() returns None for corrupted constraints.
#[ntest::timeout(5000)]
#[test]
fn test_get_returns_none_on_corrupted_sense_byte() {
    let mut store = ArenaConstraintStore::new();
    store
        .add_constraint(
            &[0, 1],
            &[1.0, -1.0],
            0.5,
            ConstraintSense::Le,
            ConstraintOrigin::Split,
        )
        .expect("add_constraint should succeed");

    // Valid before corruption
    assert!(
        store.get(0).is_some(),
        "get(0) should return Some before sense corruption"
    );

    // Corrupt the sense byte to an invalid value
    store.corrupt_header_sense(0, 42);

    // get() should return None for the corrupted constraint
    assert!(
        store.get(0).is_none(),
        "corrupted sense byte should cause get() to return None"
    );

    // iter() should skip the corrupted constraint
    let rows: Vec<_> = store.iter().collect();
    assert!(
        rows.is_empty(),
        "corrupted sense byte should cause iter() to skip constraint"
    );
}

/// Regression test for #2981 Slice 2: corrupted origin byte.
#[ntest::timeout(5000)]
#[test]
fn test_get_returns_none_on_corrupted_origin_byte() {
    let mut store = ArenaConstraintStore::new();
    store
        .add_constraint(
            &[0],
            &[1.0],
            0.0,
            ConstraintSense::Ge,
            ConstraintOrigin::Output,
        )
        .expect("add_constraint should succeed");

    assert!(
        store.get(0).is_some(),
        "get(0) should return Some before origin corruption"
    );

    // Corrupt the origin byte
    store.corrupt_header_origin(0, 255);

    assert!(
        store.get(0).is_none(),
        "corrupted origin byte should cause get() to return None"
    );

    let rows: Vec<_> = store.iter().collect();
    assert!(
        rows.is_empty(),
        "corrupted origin byte should cause iter() to skip constraint"
    );
}

/// Regression test for #2981 Slice 2: iter() correctly skips only corrupted
/// constraints while still returning valid ones.
#[ntest::timeout(5000)]
#[test]
fn test_iter_skips_only_corrupted_constraints() {
    let mut store = ArenaConstraintStore::new();

    // Add 3 valid constraints
    store
        .add_constraint(
            &[0],
            &[1.0],
            1.0,
            ConstraintSense::Le,
            ConstraintOrigin::Split,
        )
        .unwrap();
    store
        .add_constraint(
            &[1],
            &[2.0],
            2.0,
            ConstraintSense::Ge,
            ConstraintOrigin::Output,
        )
        .unwrap();
    store
        .add_constraint(
            &[2],
            &[3.0],
            3.0,
            ConstraintSense::Le,
            ConstraintOrigin::BoundProp,
        )
        .unwrap();

    assert_eq!(store.iter().count(), 3);

    // Corrupt only the middle constraint's sense byte
    store.corrupt_header_sense(1, 99);

    let rows: Vec<_> = store.iter().collect();
    assert_eq!(rows.len(), 2, "should skip only the corrupted constraint");
    assert_eq!(rows[0].bias, 1.0, "first valid constraint preserved");
    assert_eq!(rows[1].bias, 3.0, "third valid constraint preserved");
}
