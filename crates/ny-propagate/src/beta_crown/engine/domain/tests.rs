// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for domain processing and split handling.

use std::sync::Arc;

use ndarray::{Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::clip::merge_domain_bounds;
use crate::beta_crown::branching::{NeuronConstraint, SplitHistory};
use crate::beta_crown::engine::BetaCrownVerifier;

/// Regression test (#2858, #2876): Exercises the production
/// `merge_domain_bounds` function — the same code path used by
/// `apply_clip_interm_domain` in `domain/clip.rs` — to ensure NaN in
/// either old or new bounds propagates through the merge.
///
/// Before #2858 fix, the production merge used `f32::max`/`f32::min`
/// (IEEE 754), which returns the non-NaN operand, silently absorbing
/// NaN corruption into finite (unsound) bounds.
#[test]
fn test_domain_merge_nan_propagation() {
    // NaN in old lower bound: max(NaN, 3.0) must be NaN, not 3.0.
    let old_lower = Array1::from_vec(vec![1.0_f32, f32::NAN]);
    let new_lower = Array1::from_vec(vec![2.0_f32, 3.0]);
    let old_upper = Array1::from_vec(vec![5.0_f32, 4.0]);
    let new_upper = Array1::from_vec(vec![3.0_f32, f32::NAN]);

    let (merged_lower, merged_upper) =
        merge_domain_bounds(&old_lower, &new_lower, &old_upper, &new_upper);

    // Lower: max(1.0, 2.0) = 2.0; max(NaN, 3.0) = NaN (sound)
    assert_eq!(merged_lower[0], 2.0);
    assert!(
        merged_lower[1].is_nan(),
        "NaN in old_lower must propagate through production merge_domain_bounds"
    );

    // Upper: min(5.0, 3.0) = 3.0; min(4.0, NaN) = NaN (sound)
    assert_eq!(merged_upper[0], 3.0);
    assert!(
        merged_upper[1].is_nan(),
        "NaN in new_upper must propagate through production merge_domain_bounds"
    );

    // Negative control: f32::max/min would silently absorb NaN
    assert_eq!(f32::NAN.max(3.0), 3.0, "f32::max absorbs NaN — unsound");
    assert_eq!(4.0_f32.min(f32::NAN), 4.0, "f32::min absorbs NaN — unsound");
}

/// Regression test (#2858): NaN-aware change detection. IEEE 754 comparisons
/// like `NaN > x` return false, so NaN-only changes would be classified as
/// "unchanged" without explicit NaN checks. The fix adds `is_nan()` checks
/// to the change detection loop.
#[test]
fn test_domain_change_detection_with_nan() {
    // Simulate: old = [1.0, 2.0], new = [1.0, NaN]
    // Without NaN check: NaN > 2.0 is false → "unchanged" (WRONG)
    // With NaN check: NaN detected → "changed" (CORRECT)
    let old_lower = [1.0_f32, 2.0];
    let new_lower = [1.0_f32, f32::NAN];
    let old_upper = [3.0_f32, 4.0];
    let new_upper = [3.0_f32, 4.0];

    let mut changed = false;
    for i in 0..old_lower.len() {
        if new_lower[i] > old_lower[i]
            || new_upper[i] < old_upper[i]
            || new_lower[i].is_nan()
            || new_upper[i].is_nan()
            || old_lower[i].is_nan()
            || old_upper[i].is_nan()
        {
            changed = true;
            break;
        }
    }
    assert!(changed, "NaN in new_lower must be detected as a change");

    // Without the NaN checks, this would falsely report no change:
    let mut changed_without_nan_check = false;
    for i in 0..old_lower.len() {
        if new_lower[i] > old_lower[i] || new_upper[i] < old_upper[i] {
            changed_without_nan_check = true;
            break;
        }
    }
    assert!(
        !changed_without_nan_check,
        "Negative control: without NaN checks, NaN change is invisible"
    );
}

fn make_layer_bounds(lower: &[f32], upper: &[f32]) -> Arc<BoundedTensor> {
    let n = lower.len();
    Arc::new(
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
        )
        .unwrap(),
    )
}

/// Regression test for #2710: apply_history_constraints must skip
/// constraints with layer_idx == 0 (no pre-activation bounds for input layer).
/// Skipping is sound — it produces looser bounds (original bounds unchanged).
#[test]
fn test_apply_history_constraints_skip_layer_idx_zero() {
    let verifier = BetaCrownVerifier::default();
    let base_bounds = vec![
        make_layer_bounds(&[-1.0, -2.0], &[1.0, 2.0]),
        make_layer_bounds(&[-3.0, -4.0], &[3.0, 4.0]),
    ];

    let mut history = SplitHistory::new();
    // layer_idx=0 is invalid (layers are 1-indexed for pre-activation bounds)
    history.add_constraint(NeuronConstraint::new(0, 0, true, 1.0).unwrap());

    let result = verifier
        .apply_history_constraints(&base_bounds, &history)
        .unwrap();
    let bounds = result.expect("should return Some (not infeasible)");

    // Bounds should be unchanged — the constraint was skipped
    assert_eq!(bounds[0].lower().len(), 2);
    assert_eq!(bounds[0].lower()[[0]], -1.0);
    assert_eq!(bounds[0].upper()[[0]], 1.0);
    assert_eq!(bounds[1].lower()[[0]], -3.0);
    assert_eq!(bounds[1].upper()[[0]], 3.0);
}

/// Regression test for #2710: apply_history_constraints must skip
/// constraints with layer_idx > layer_bounds.len().
#[test]
fn test_apply_history_constraints_skip_layer_idx_too_large() {
    let verifier = BetaCrownVerifier::default();
    let base_bounds = vec![make_layer_bounds(&[-1.0, -2.0], &[1.0, 2.0])];

    let mut history = SplitHistory::new();
    // layer_idx=5 exceeds bounds (len=1, valid range 1..=1)
    history.add_constraint(NeuronConstraint::new(5, 0, true, 1.0).unwrap());

    let result = verifier
        .apply_history_constraints(&base_bounds, &history)
        .unwrap();
    let bounds = result.expect("should return Some (not infeasible)");

    // Bounds unchanged — constraint skipped
    assert_eq!(bounds[0].lower()[[0]], -1.0);
    assert_eq!(bounds[0].upper()[[0]], 1.0);
}

/// Regression test for #2710: apply_history_constraints must skip
/// constraints with neuron_idx >= neuron count in the layer.
#[test]
fn test_apply_history_constraints_skip_neuron_idx_out_of_bounds() {
    let verifier = BetaCrownVerifier::default();
    let base_bounds = vec![make_layer_bounds(&[-1.0, -2.0], &[1.0, 2.0])];

    let mut history = SplitHistory::new();
    // layer_idx=1 is valid, but neuron_idx=5 exceeds neuron count (2)
    history.add_constraint(NeuronConstraint::new(1, 5, true, 1.0).unwrap());

    let result = verifier
        .apply_history_constraints(&base_bounds, &history)
        .unwrap();
    let bounds = result.expect("should return Some (not infeasible)");

    // Bounds unchanged — constraint skipped
    assert_eq!(bounds[0].lower()[[0]], -1.0);
    assert_eq!(bounds[0].lower()[[1]], -2.0);
    assert_eq!(bounds[0].upper()[[0]], 1.0);
    assert_eq!(bounds[0].upper()[[1]], 2.0);
}

/// Verify that valid constraints DO tighten bounds (positive control).
/// A valid active constraint (x >= 0) should raise the lower bound to 0.
#[test]
fn test_apply_history_constraints_valid_constraint_tightens() {
    let verifier = BetaCrownVerifier::default();
    // Layer bounds: neuron 0 has [-1, 1], neuron 1 has [-2, 2]
    let base_bounds = vec![make_layer_bounds(&[-1.0, -2.0], &[1.0, 2.0])];

    let mut history = SplitHistory::new();
    // Valid: layer_idx=1, neuron_idx=0, is_active=true → lower clamped to 0
    history.add_constraint(NeuronConstraint::new(1, 0, true, 1.0).unwrap());

    let result = verifier
        .apply_history_constraints(&base_bounds, &history)
        .unwrap();
    let bounds = result.expect("should return Some (not infeasible)");

    // Neuron 0: lower raised from -1.0 to 0.0 (active constraint)
    assert_eq!(bounds[0].lower()[[0]], 0.0);
    // Neuron 0: upper unchanged
    assert_eq!(bounds[0].upper()[[0]], 1.0);
    // Neuron 1: unchanged (no constraint)
    assert_eq!(bounds[0].lower()[[1]], -2.0);
    assert_eq!(bounds[0].upper()[[1]], 2.0);
}

/// Verify that skipping out-of-bounds constraints produces LOOSER
/// bounds than applying valid constraints (soundness property).
#[test]
fn test_apply_history_constraints_skip_produces_looser_bounds() {
    let verifier = BetaCrownVerifier::default();
    let base_bounds = vec![make_layer_bounds(&[-1.0, -2.0], &[1.0, 2.0])];

    // History with valid constraint
    let mut valid_history = SplitHistory::new();
    valid_history.add_constraint(NeuronConstraint::new(1, 0, true, 1.0).unwrap());

    // History with out-of-bounds constraint (should be skipped)
    let mut oob_history = SplitHistory::new();
    oob_history.add_constraint(NeuronConstraint::new(0, 0, true, 1.0).unwrap());

    let valid_bounds = verifier
        .apply_history_constraints(&base_bounds, &valid_history)
        .unwrap()
        .unwrap();
    let oob_bounds = verifier
        .apply_history_constraints(&base_bounds, &oob_history)
        .unwrap()
        .unwrap();

    // oob_bounds should be at least as loose as valid_bounds for all neurons
    for i in 0..2 {
        assert!(
            oob_bounds[0].lower()[[i]] <= valid_bounds[0].lower()[[i]],
            "neuron {i}: oob lower {} should be <= valid lower {} (looser)",
            oob_bounds[0].lower()[[i]],
            valid_bounds[0].lower()[[i]]
        );
        assert!(
            oob_bounds[0].upper()[[i]] >= valid_bounds[0].upper()[[i]],
            "neuron {i}: oob upper {} should be >= valid upper {} (looser)",
            oob_bounds[0].upper()[[i]],
            valid_bounds[0].upper()[[i]]
        );
    }
}
