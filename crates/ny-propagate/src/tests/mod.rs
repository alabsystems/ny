// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// Test suites.
mod activations;
mod attention_monotonicity;
mod bab_soundness_stress;
mod backward_dispatch;
mod bilinear;
mod checkpoint;
mod conv;
pub(crate) mod crown;
mod crown_profile;
mod doc_imports;
mod flatten;
mod gelu_sound;
mod graph;
mod invprop_coverage;
mod matmul;
mod proptest_soundness;
mod reciprocal;
mod sdp_crown;
mod soundness_audit_wave3;
mod speaker_embedding_distance;
mod streaming_boundary;
mod streaming_boundary_seam;
mod sub_backward_enclosure;
mod tile;
mod transformer;
mod verifier;
mod verifier_certification;

// Proof coverage tests (Prover-authored).
mod proof_coverage_branching;

// Shared helpers/fixtures.
mod bounds;
mod ibp;
mod sequential_ibp_nan_firewall;

// Re-export crate items so test submodules can share a single import surface.
pub use super::*;
use crate::layers::activations::LinearRelaxation;
#[allow(unused_imports)]
pub use ndarray::{arr1, arr2};

// Serialize tests that mutate shared env vars consumed by CROWN code.
//
// Delegates to the workspace-blessed env-mutation choke point
// (`ny_test_utils::env`) — the only place raw `std::env::set_var`/`remove_var`
// is permitted under the clippy env wall (root `clippy.toml`).
#[allow(unused_imports)]
pub(crate) use ny_test_utils::env::{
    with_env_edits, with_serialized_env_vars, with_serialized_env_vars_removed,
};

/// Serialize tests that mutate the shared CROWN dense budget env var.
#[allow(dead_code)]
pub(crate) fn with_crown_dense_budget_mb<T>(value: &str, f: impl FnOnce() -> T) -> T {
    with_serialized_env_vars(&[("NY_DENSE_BUDGET_MB", value)], f)
}

/// Serialize tests that mutate the Conv2d CROWN-backward memory cap env var
/// (`NY_CROWN_MEM_CAP_MB`, #conv-crown-oom).
#[allow(dead_code)]
pub(crate) fn with_crown_mem_cap_mb<T>(value: &str, f: impl FnOnce() -> T) -> T {
    with_serialized_env_vars(&[("NY_CROWN_MEM_CAP_MB", value)], f)
}

/// Serialize tests that mutate the shared patches budget env var.
#[allow(dead_code)]
pub(crate) fn with_patches_tightening_budget_secs<T>(value: &str, f: impl FnOnce() -> T) -> T {
    with_serialized_env_vars(&[("NY_PATCHES_BUDGET_SECS", value)], f)
}

/// Assert two arrays are element-wise close within `tol`.
///
/// Centralised from duplicate definitions in `bilinear.rs` and
/// `activations/add_constant.rs` (issue #1716).
#[allow(dead_code)] // Shared test utility; not all test modules import it in every build.
fn floats_close_or_equal(actual: f32, expected: f32, tol: f32) -> bool {
    actual == expected || (actual - expected).abs() <= tol
}

#[allow(dead_code)] // Shared test utility; not all test modules import it in every build.
pub fn assert_all_close(
    actual: &ndarray::ArrayD<f32>,
    expected: &ndarray::ArrayD<f32>,
    tol: f32,
    label: &str,
) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label} shape mismatch: actual={:?}, expected={:?}",
        actual.shape(),
        expected.shape()
    );
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            floats_close_or_equal(a, e, tol),
            "{label} mismatch at flat index {idx}: actual={a}, expected={e}, tol={tol}"
        );
    }
}

/// Assert two scalar f32 values are close within `tol`.
///
/// Consolidated from 5 duplicate `assert_close` definitions in layers/
/// test modules (issue #2496). Each call site passes its own tolerance
/// instead of hiding it inside the helper.
#[allow(dead_code)] // Shared test utility; not all test modules import it in every build.
pub fn assert_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        floats_close_or_equal(actual, expected, tol),
        "expected {expected}, got {actual} (diff={:.2e}, tol={tol})",
        (actual - expected).abs()
    );
}

/// Assert two `LinearBounds` match elementwise within `tol`.
///
/// Consolidated from 4 duplicate definitions across beta_crown/engine/tests,
/// layers/linear/tests, and tests/crown/ modules (issue #3685).
#[allow(dead_code)] // Shared test utility; not all test modules import it in every build.
pub fn assert_linear_bounds_close(
    actual: &LinearBounds,
    expected: &LinearBounds,
    tol: f32,
    label: &str,
) {
    assert_eq!(
        actual.lower_a().shape(),
        expected.lower_a().shape(),
        "{label}: lower_a shape mismatch {:?} vs {:?}",
        actual.lower_a().shape(),
        expected.lower_a().shape()
    );
    assert_eq!(
        actual.upper_a().shape(),
        expected.upper_a().shape(),
        "{label}: upper_a shape mismatch {:?} vs {:?}",
        actual.upper_a().shape(),
        expected.upper_a().shape()
    );
    assert_eq!(
        actual.lower_b().shape(),
        expected.lower_b().shape(),
        "{label}: lower_b shape mismatch {:?} vs {:?}",
        actual.lower_b().shape(),
        expected.lower_b().shape()
    );
    assert_eq!(
        actual.upper_b().shape(),
        expected.upper_b().shape(),
        "{label}: upper_b shape mismatch {:?} vs {:?}",
        actual.upper_b().shape(),
        expected.upper_b().shape()
    );
    for (idx, (&a, &e)) in actual
        .lower_a()
        .iter()
        .zip(expected.lower_a().iter())
        .enumerate()
    {
        assert!(
            floats_close_or_equal(a, e, tol),
            "{label}: lower_a[{idx}] actual={a} expected={e} diff={:.2e} tol={tol}",
            (a - e).abs()
        );
    }
    for (idx, (&a, &e)) in actual
        .upper_a()
        .iter()
        .zip(expected.upper_a().iter())
        .enumerate()
    {
        assert!(
            floats_close_or_equal(a, e, tol),
            "{label}: upper_a[{idx}] actual={a} expected={e} diff={:.2e} tol={tol}",
            (a - e).abs()
        );
    }
    for (idx, (&a, &e)) in actual
        .lower_b()
        .iter()
        .zip(expected.lower_b().iter())
        .enumerate()
    {
        assert!(
            floats_close_or_equal(a, e, tol),
            "{label}: lower_b[{idx}] actual={a} expected={e} diff={:.2e} tol={tol}",
            (a - e).abs()
        );
    }
    for (idx, (&a, &e)) in actual
        .upper_b()
        .iter()
        .zip(expected.upper_b().iter())
        .enumerate()
    {
        assert!(
            floats_close_or_equal(a, e, tol),
            "{label}: upper_b[{idx}] actual={a} expected={e} diff={:.2e} tol={tol}",
            (a - e).abs()
        );
    }
}

/// Assert two `BatchedLinearBounds` match elementwise within `tol`.
///
/// Consolidated from 3 duplicate definitions in layers/convolution/ and
/// layers/binary_ops/bilinear/ test modules (issue #3685).
#[allow(dead_code)] // Shared test utility; not all test modules import it in every build.
pub fn assert_batched_bounds_close(
    actual: &BatchedLinearBounds,
    expected: &BatchedLinearBounds,
    tol: f32,
    label: &str,
) {
    assert_eq!(
        actual.input_shape(),
        expected.input_shape(),
        "{label}: input_shape mismatch"
    );
    assert_eq!(
        actual.output_shape(),
        expected.output_shape(),
        "{label}: output_shape mismatch"
    );
    assert_eq!(
        actual.lower_a().shape(),
        expected.lower_a().shape(),
        "{label}: lower_a shape mismatch"
    );
    assert_eq!(
        actual.upper_a().shape(),
        expected.upper_a().shape(),
        "{label}: upper_a shape mismatch"
    );
    assert_eq!(
        actual.lower_b().shape(),
        expected.lower_b().shape(),
        "{label}: lower_b shape mismatch"
    );
    assert_eq!(
        actual.upper_b().shape(),
        expected.upper_b().shape(),
        "{label}: upper_b shape mismatch"
    );
    let fields: &[(&str, &ndarray::ArrayD<f32>, &ndarray::ArrayD<f32>)] = &[
        ("lower_a", actual.lower_a(), expected.lower_a()),
        ("upper_a", actual.upper_a(), expected.upper_a()),
        ("lower_b", actual.lower_b(), expected.lower_b()),
        ("upper_b", actual.upper_b(), expected.upper_b()),
    ];
    for &(name, av, ev) in fields {
        for (idx, (&a, &e)) in av.iter().zip(ev.iter()).enumerate() {
            assert!(
                floats_close_or_equal(a, e, tol),
                "{label}: {name}[{idx}] actual={a} expected={e} diff={:.2e} tol={tol}",
                (a - e).abs()
            );
        }
    }
}

#[test]
fn test_assert_close_accepts_matching_infinities_3974() {
    assert_close(f32::INFINITY, f32::INFINITY, 0.0);
    assert_close(f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0);
}

#[test]
fn test_assert_all_close_accepts_matching_infinities_3974() {
    let actual = arr1(&[f32::NEG_INFINITY, f32::INFINITY]).into_dyn();
    let expected = arr1(&[f32::NEG_INFINITY, f32::INFINITY]).into_dyn();
    assert_all_close(&actual, &expected, 0.0, "infinity array");
}

#[test]
fn test_assert_linear_bounds_close_accepts_matching_infinities_3974() {
    let actual = LinearBounds::new(
        arr2(&[[1.0_f32]]),
        arr1(&[f32::NEG_INFINITY]),
        arr2(&[[1.0_f32]]),
        arr1(&[f32::INFINITY]),
    )
    .expect("bias infinities are valid for LinearBounds");
    let expected = LinearBounds::new(
        arr2(&[[1.0_f32]]),
        arr1(&[f32::NEG_INFINITY]),
        arr2(&[[1.0_f32]]),
        arr1(&[f32::INFINITY]),
    )
    .expect("bias infinities are valid for LinearBounds");

    assert_linear_bounds_close(&actual, &expected, 0.0, "infinity linear bounds");
}

#[test]
fn test_assert_batched_bounds_close_accepts_matching_infinities_3974() {
    let actual = BatchedLinearBounds::new(
        arr2(&[[1.0_f32]]).into_dyn(),
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr2(&[[1.0_f32]]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
        vec![1],
        vec![1],
    )
    .expect("bias infinities are valid for BatchedLinearBounds");
    let expected = BatchedLinearBounds::new(
        arr2(&[[1.0_f32]]).into_dyn(),
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr2(&[[1.0_f32]]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
        vec![1],
        vec![1],
    )
    .expect("bias infinities are valid for BatchedLinearBounds");

    assert_batched_bounds_close(&actual, &expected, 0.0, "infinity batched bounds");
}

/// Assert end-to-end CROWN backward soundness: for each interval `[l, u]`,
/// call `propagate_linear_with_bounds` with identity coefficients, then
/// verify the returned linear bounds enclose `f(x)` for all sampled `x`.
///
/// Consolidated from identical copies in `periodic/tests.rs` and
/// `s_shaped.rs` (#2307).
#[allow(dead_code)]
pub fn assert_crown_backward_sound<F>(layer: &dyn BoundPropagation, intervals: &[(f32, f32)], f: F)
where
    F: Fn(f32) -> f32,
{
    use ny_tensor::BoundedTensor;

    for &(l, u) in intervals {
        let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn())
            .expect("invariant: valid test interval");
        let bounds = LinearBounds::identity(1);
        let result = layer
            .propagate_linear_with_bounds(&bounds, &pre)
            .expect("propagate_linear_with_bounds should not fail for valid test interval");

        let la = result.lower_a[[0, 0]];
        let lb = result.lower_b[0];
        let ua = result.upper_a[[0, 0]];
        let ub = result.upper_b[0];

        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = f(x);
            assert!(
                la * x + lb <= y + 1e-3,
                "CROWN lb violated at x={x} for [{l}, {u}]: {} > {y}",
                la * x + lb
            );
            assert!(
                ua * x + ub >= y - 1e-3,
                "CROWN ub violated at x={x} for [{l}, {u}]: {} < {y}",
                ua * x + ub
            );
        }
    }
}

/// Assert that a `LinearRelaxation` is sound for function `f` over `[l, u]`.
///
/// Samples 200 points uniformly in `[l, u]` and checks that
/// `lower(x) <= f(x) + tol` and `upper(x) >= f(x) - tol` everywhere.
///
/// Consolidated from 5 duplicate definitions across trigonometric/ and
/// softmax/gelu/ test modules (issue #2496).
#[allow(dead_code)]
pub fn assert_relaxation_sound<F>(
    l: f32,
    u: f32,
    relaxation: LinearRelaxation,
    f: F,
    tol: f32,
    label: &str,
) where
    F: Fn(f32) -> f32,
{
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = relaxation;
    let n = 200;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = (l + (u - l) * t).clamp(l, u);
        let y = f(x);
        let lower = ls * x + li;
        let upper = us * x + ui;
        assert!(
            lower <= y + tol,
            "{label} @ x={x}: lower={lower} > f(x)={y} (violation={:.2e})",
            lower - y,
        );
        assert!(
            upper >= y - tol,
            "{label} @ x={x}: upper={upper} < f(x)={y} (violation={:.2e})",
            y - upper,
        );
    }
}
