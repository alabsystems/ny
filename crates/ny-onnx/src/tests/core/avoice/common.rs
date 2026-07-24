// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for avoice model-family test modules.

use super::*;
use ny_core::{Bound, VerificationSpec};
use std::sync::{Mutex, MutexGuard, OnceLock};

pub(super) fn assert_finite_and_ordered(bounds: &BoundedTensor, label: &str) {
    assert!(
        bounds.lower().iter().all(|value| value.is_finite()),
        "{label}: lower bounds contain non-finite values"
    );
    assert!(
        bounds.upper().iter().all(|value| value.is_finite()),
        "{label}: upper bounds contain non-finite values"
    );
    for (idx, (&lower, &upper)) in bounds.lower().iter().zip(bounds.upper().iter()).enumerate() {
        assert!(
            lower <= upper,
            "{label}: inverted bounds at dim {idx}: lower={lower}, upper={upper}"
        );
    }
}

pub(super) fn input_spec_by_name<'a>(model: &'a OnnxModel, name: &str) -> &'a TensorSpec {
    model
        .network
        .inputs
        .iter()
        .find(|spec| spec.name == name)
        .unwrap_or_else(|| {
            panic!(
                "expected input spec '{name}', got {:?}",
                model.network.inputs
            )
        })
}

pub(super) fn unbatched_shape_from_input_spec(
    input_spec: &TensorSpec,
    dynamic_size: usize,
    label: &str,
) -> Vec<usize> {
    assert!(
        input_spec.shape.len() >= 2,
        "{label} input should include a batch axis, got {:?}",
        input_spec.shape
    );
    input_spec.shape[1..]
        .iter()
        .map(|&dim| if dim > 0 { dim as usize } else { dynamic_size })
        .collect()
}

pub(super) fn node_names_by_layer_type(graph: &GraphNetwork, layer_type: &str) -> Vec<String> {
    graph
        .node_names()
        .iter()
        .filter_map(|name| {
            graph
                .node(name)
                .filter(|node| node.layer().layer_type() == layer_type)
                .map(|_| name.clone())
        })
        .collect()
}

pub(super) fn node_name_hits(graph: &GraphNetwork, needle: &str) -> Vec<String> {
    let needle = needle.to_ascii_lowercase();
    graph
        .node_names()
        .iter()
        .filter(|name| name.to_ascii_lowercase().contains(&needle))
        .take(8)
        .cloned()
        .collect()
}

pub(super) fn node_names_by_layer_types(graph: &GraphNetwork, layer_types: &[&str]) -> Vec<String> {
    layer_types
        .iter()
        .flat_map(|layer_type| node_names_by_layer_type(graph, layer_type))
        .collect()
}

pub(super) fn assert_node_bounds_finite_and_ordered(
    node_bounds: &HashMap<String, BoundedTensor>,
    node_names: &[String],
    label: &str,
) {
    for node_name in node_names {
        let bounds = node_bounds
            .get(node_name)
            .unwrap_or_else(|| panic!("{label}: missing node bounds for {node_name}"));
        assert_finite_and_ordered(bounds, &format!("{label} ({node_name})"));
    }
}

pub(super) fn lock_heavy_avoice_round_trip() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("heavy avoice round-trip test mutex should not be poisoned")
}

/// Evaluate a graph at the center of an input epsilon ball and return the
/// concrete output. Uses IBP with a zero-width BoundedTensor (lower == upper)
/// so the output is the exact function value at the center point (up to FP
/// rounding).
pub(super) fn evaluate_graph_at_center(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    label: &str,
) -> BoundedTensor {
    let center = input_bounds.center();
    let point_input = BoundedTensor::new(center.clone(), center)
        .unwrap_or_else(|e| panic!("{label}: center point BoundedTensor failed: {e}"));
    graph
        .propagate_ibp(&point_input)
        .unwrap_or_else(|e| panic!("{label}: concrete evaluation at center failed: {e}"))
}

/// Assert that every element of `concrete` (a zero-width or near-zero-width
/// BoundedTensor from a point evaluation) falls within `bounds`.
///
/// Tolerance is relative: `1e-5 * max(|value|, |lower|, |upper|, 1.0)`.
/// The scale factor absorbs ordinary f32 rounding in deterministic center-point
/// evaluations without masking materially unsound bounds.
pub(super) fn assert_concrete_contained_in_bounds(
    concrete: &BoundedTensor,
    bounds: &BoundedTensor,
    label: &str,
) {
    let concrete_flat = concrete.flatten();
    let bounds_flat = bounds.flatten();
    assert_eq!(
        concrete_flat.lower().len(),
        bounds_flat.lower().len(),
        "{label}: concrete output length {} != bounds length {}",
        concrete_flat.lower().len(),
        bounds_flat.lower().len()
    );

    let mut contained_count = 0usize;
    for (idx, ((&val, &lo), &hi)) in concrete_flat
        .lower()
        .iter()
        .zip(bounds_flat.lower().iter())
        .zip(bounds_flat.upper().iter())
        .enumerate()
    {
        let scale = val.abs().max(lo.abs()).max(hi.abs()).max(1.0);
        let tol = 1e-5 * scale;
        assert!(
            val >= lo - tol,
            "{label}[{idx}]: concrete={val:.6e} below lower={lo:.6e} (tol={tol:.6e})"
        );
        assert!(
            val <= hi + tol,
            "{label}[{idx}]: concrete={val:.6e} above upper={hi:.6e} (tol={tol:.6e})"
        );
        contained_count += 1;
    }
    eprintln!("{label}: {contained_count} elements contained in bounds");
}

/// Wall-clock verifier budget, asserted only under `--release` (see the
/// wall-clock budget policy in the `avoice` module docs).
///
/// The release budget is passed through unchanged so `Timeout` verdicts (and
/// the `panic!("timed out")` arms downstream) keep their meaning on release
/// hardware. Debug builds substitute an effectively unbounded 24h budget: a
/// debug-mode `Timeout` verdict would only measure the build profile, and it
/// would also skip the verdict/bounds/center correctness assertions that are
/// the actual point of these tests.
pub(super) const fn release_budget_ms(release_ms: u64) -> u64 {
    if cfg!(debug_assertions) {
        86_400_000 // 24h: completes-or-hangs watchdog, not a perf assertion
    } else {
        release_ms
    }
}

/// Build a `VerificationSpec` from a `BoundedTensor` input and output bound
/// constraints. Shared across all avoice verifier-smoke tests.
///
/// Consolidates the previously duplicated `verifier_spec_from_bounded_input`
/// in talker_attention, speaker_encoder, and duration_predictor (#3950).
pub(super) fn verifier_spec_from_bounded_input(
    input: &BoundedTensor,
    output_bounds: Vec<Bound>,
    timeout_ms: u64,
) -> VerificationSpec {
    let flat = input.flatten();
    let input_bounds: Vec<Bound> = flat
        .lower()
        .iter()
        .zip(flat.upper().iter())
        .map(|(&lo, &hi)| Bound::new(lo, hi))
        .collect();
    VerificationSpec::from_parts(
        input_bounds,
        output_bounds,
        Some(timeout_ms),
        Some(input.shape().to_vec()),
    )
    .expect("verifier spec should be valid")
}

/// Assert that bounds from a `VerificationResult::Unknown` result are
/// structurally sound: cardinality-preserving, non-empty, finite, and
/// well-ordered.
///
/// Without this, the `Unknown` arm silently passes tests, masking NaN
/// propagation, inverted bounds, and broken propagation engines.
///
/// Consolidates the previously duplicated `assert_unknown_bounds_sound`
/// in talker_attention and duration_predictor (#3950).
pub(super) fn assert_unknown_verifier_bounds_sound(
    bounds: &[Bound],
    expected_len: usize,
    label: &str,
) {
    assert_eq!(
        bounds.len(),
        expected_len,
        "{label}: Unknown result should preserve output count ({expected_len}), got {}",
        bounds.len()
    );
    assert!(
        !bounds.is_empty(),
        "{label}: Unknown result with empty bounds — propagation returned no output"
    );
    for (idx, bound) in bounds.iter().enumerate() {
        assert!(
            bound.lower().is_finite(),
            "{label}: Unknown result bounds[{idx}] has non-finite lower: {}",
            bound.lower()
        );
        assert!(
            bound.upper().is_finite(),
            "{label}: Unknown result bounds[{idx}] has non-finite upper: {}",
            bound.upper()
        );
        assert!(
            bound.lower() <= bound.upper(),
            "{label}: Unknown result bounds[{idx}] inverted: lower={} > upper={}",
            bound.lower(),
            bound.upper()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::assert_unknown_verifier_bounds_sound;
    use ny_core::Bound;

    #[test]
    fn test_assert_unknown_verifier_bounds_sound_accepts_exact_length_4062() {
        let bounds = vec![
            Bound::new(0.0, 1.0),
            Bound::new(0.25, 1.25),
            Bound::new(0.5, 1.5),
        ];
        let result = std::panic::catch_unwind(|| {
            assert_unknown_verifier_bounds_sound(
                &bounds,
                bounds.len(),
                "exact unknown cardinality",
            );
        });
        assert!(
            result.is_ok(),
            "exact-length Unknown bounds should satisfy the helper contract"
        );
    }

    #[test]
    fn test_assert_unknown_verifier_bounds_sound_rejects_truncated_vector_4062() {
        let bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
        let result = std::panic::catch_unwind(|| {
            assert_unknown_verifier_bounds_sound(&bounds, 3, "vector unknown cardinality");
        });
        assert!(
            result.is_err(),
            "truncated vector Unknown bounds should panic on cardinality mismatch"
        );
    }

    #[test]
    fn test_assert_unknown_verifier_bounds_sound_rejects_non_scalar_unknown_4062() {
        let bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
        let result = std::panic::catch_unwind(|| {
            assert_unknown_verifier_bounds_sound(&bounds, 1, "scalar unknown cardinality");
        });
        assert!(
            result.is_err(),
            "non-scalar Unknown bounds should panic when scalar output is expected"
        );
    }
}
