// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::arr1;
use std::collections::HashMap;

fn empty_alpha_state() -> AlphaState {
    AlphaState {
        alphas: vec![],
        alphas_upper: vec![],
        unstable_mask: vec![],
        velocity: vec![],
        adam_m: vec![],
        adam_v: vec![],
        velocity_upper: vec![],
        adam_m_upper: vec![],
        adam_v_upper: vec![],
        bilinear_alphas: HashMap::new(),
        bilinear_adam_m: HashMap::new(),
        bilinear_adam_v: HashMap::new(),
        invprop_state: None,
    }
}

fn scalar_bounds(lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn())
        .expect("scalar test bounds should be valid")
}

#[test]
fn invprop_spsa_directions_are_deterministic_and_not_a_two_direction_checkerboard() {
    let direction = |iter| {
        (0..64)
            .map(|parameter| invprop_spsa_sign(iter, parameter))
            .collect::<Vec<_>>()
    };
    let directions = [direction(0), direction(1), direction(2), direction(3)];
    assert_eq!(directions[0], direction(0));
    for left in 0..directions.len() {
        for right in (left + 1)..directions.len() {
            assert_ne!(directions[left], directions[right]);
            assert!(
                directions[left]
                    .iter()
                    .zip(&directions[right])
                    .any(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits()),
                "directions {left} and {right} must not be exact negatives"
            );
        }
    }
}

#[test]
fn invprop_spsa_score_normalization_is_bounded_and_restores_zero_direction() {
    let positive = invprop_bounded_spsa_step(-1.0, 1.0e300, 0.1, 0.5).unwrap();
    let negative = invprop_bounded_spsa_step(-1.0, -1.0e300, 0.1, 0.5).unwrap();
    let zero = invprop_bounded_spsa_step(-1.0, -1.0, 0.1, 0.5).unwrap();
    let wide_gap = invprop_bounded_spsa_step(-100.0, -99.95, 0.1, 0.5).unwrap();
    assert_eq!(positive, 0.5);
    assert_eq!(negative, -0.5);
    assert_eq!(zero.to_bits(), 0.0_f64.to_bits());
    assert!((wide_gap - 0.25).abs() < 1.0e-12);
    assert!(invprop_bounded_spsa_step(-1.0, f64::INFINITY, 0.1, 0.5).is_none());
    assert!(invprop_bounded_spsa_step(-1.0, 0.0, 0.0, 0.5).is_none());
}

#[test]
fn invprop_rowwise_spsa_moves_a_nonwinning_column_without_touching_other_rows() {
    let base = Array3::from_elem((2, 1, 2), 1.0_f32);
    // Row 0 wins the hard max and is unchanged. Row 1 improves substantially
    // but remains non-winning, so a max-only score would produce a zero update.
    let updated = invprop_projected_spsa_update(
        &base,
        &[Some(-1.0), Some(-100.0)],
        &[Some(-1.0), Some(-90.0)],
        0.1,
        1.0,
        0,
    )
    .expect("the non-winning row response should produce an update");
    assert_eq!(updated[[0, 0, 0]].to_bits(), base[[0, 0, 0]].to_bits());
    assert_eq!(updated[[1, 0, 0]].to_bits(), base[[1, 0, 0]].to_bits());
    assert!(
        updated[[0, 0, 1]].to_bits() != base[[0, 0, 1]].to_bits()
            || updated[[1, 0, 1]].to_bits() != base[[1, 0, 1]].to_bits()
    );
}

#[test]
fn invprop_shared_spsa_uses_mean_normalized_row_response() {
    let base = Array3::from_elem((2, 1, 1), 1.0_f32);
    let updated = invprop_projected_spsa_update(
        &base,
        &[Some(-1.0), Some(-100.0)],
        &[Some(-1.0), Some(-90.0)],
        0.1,
        1.0,
        0,
    )
    .expect("a shared gamma should aggregate the non-winning row response");
    // Probe-delta-normalized responses are [0, 1] after clipping, so their
    // mean gives |step|=0.5 without allowing either row to exceed the trust
    // radius on its own.
    assert!((updated[[0, 0, 0]] - 1.5).abs() < 1.0e-6);
    assert!((updated[[1, 0, 0]] - 0.5).abs() < 1.0e-6);
}

struct DeadlineInvpropProbeBackend {
    need_grad: std::cell::Cell<Option<bool>>,
}

impl AlphaCrownBackend for DeadlineInvpropProbeBackend {
    fn backward_iteration(
        &self,
        alpha_state: &AlphaState,
        _input: &BoundedTensor,
        _iter: usize,
        _invprop_enabled: bool,
        need_grad: bool,
    ) -> Result<Option<BackwardIterationResult>> {
        self.need_grad.set(Some(need_grad));
        let state = alpha_state.invprop_state.as_ref().unwrap();
        let gammas = state
            .layer_gammas(crate::invprop::INVPROP_OUTPUT_SEED)
            .unwrap();
        let (lower, upper) = gammas.checked_bound_gammas().unwrap();
        let _discarded_fold =
            crate::network::graph_alpha::invprop_backward::augment_bounds_with_constraints(
                &LinearBounds::identity(1),
                &state.constraints,
                &lower.to_owned(),
                &upper.to_owned(),
            );
        Err(NyError::DeadlineExceeded(
            "scripted optional gamma probe deadline".to_string(),
        ))
    }

    fn compute_gradients(
        &self,
        _config: &AlphaCrownConfig,
        _alpha_state: &mut AlphaState,
        _input: &BoundedTensor,
        _gradients: &[Array1<f32>],
        _gradients_upper: &[Array1<f32>],
        _iter: usize,
    ) -> Result<(Vec<Array1<f32>>, Vec<Array1<f32>>)> {
        unreachable!("a direct gamma-probe regression never requests alpha gradients")
    }

    fn crown_fallback(&self, _input: &BoundedTensor) -> Result<BoundedTensor> {
        unreachable!("a direct gamma-probe regression never requests CROWN fallback")
    }

    fn log_label(&self) -> &str {
        "deadline-invprop-probe"
    }
}

#[test]
fn invprop_deadline_probe_restores_seed_and_drops_evaluated_attribution() {
    let _serial = crate::execution_telemetry::TEST_LOCK.lock().unwrap();
    let _run = crate::execution_telemetry::begin_run();
    let mut alpha_state = empty_alpha_state();
    alpha_state
        .init_invprop_state(
            crate::invprop::OutputConstraints::le_threshold(1, 0, -0.5).unwrap(),
            1,
        )
        .unwrap();
    alpha_state.invprop_mut().unwrap().add_layer_gammas(
        crate::invprop::INVPROP_OUTPUT_SEED.to_string(),
        crate::invprop::LayerGammas::new(1, 1, false),
    );
    let original_seed = alpha_state
        .invprop_state
        .as_ref()
        .unwrap()
        .layer_gammas(crate::invprop::INVPROP_OUTPUT_SEED)
        .unwrap()
        .gammas
        .clone();
    let backend = DeadlineInvpropProbeBackend {
        need_grad: std::cell::Cell::new(None),
    };
    let config = AlphaCrownConfig {
        iterations: 2,
        invprop: crate::invprop::InvpropConfig {
            enabled: true,
            optimize_gammas: true,
            ..Default::default()
        },
        ..AlphaCrownConfig::default()
    };

    let outcome = invprop_seed_gamma_ascent_step(
        &backend,
        &config,
        &mut alpha_state,
        &scalar_bounds(-1.0, 1.0),
        0,
        &[Some(-2.0)],
        1,
    )
    .unwrap();
    assert_eq!(outcome, InvpropGammaStepOutcome::DeadlineExceeded);
    assert_eq!(backend.need_grad.get(), Some(false));
    let restored_seed = &alpha_state
        .invprop_state
        .as_ref()
        .unwrap()
        .layer_gammas(crate::invprop::INVPROP_OUTPUT_SEED)
        .unwrap()
        .gammas;
    assert!(params_bit_identical_for_test(restored_seed, &original_seed));

    let observed = crate::execution_telemetry::snapshot().invprop;
    assert_eq!(observed.gamma_steps_attempted, 1);
    assert_eq!(observed.gamma_steps_applied, 0);
    assert_eq!(observed.nonzero_output_seed_folds, 1);
    assert_eq!(observed.nonzero_evaluated_output_seed_folds, 0);
    assert!(!observed.attribution_conflict);
}

fn params_bit_identical_for_test(lhs: &Array3<f32>, rhs: &Array3<f32>) -> bool {
    lhs.dim() == rhs.dim()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

/// Mock backend for testing alpha_crown_optimize behavior with infinite bounds.
struct InfiniteBoundsBackend {
    /// Bounds returned by crown_fallback. Contains infinities to simulate
    /// clamp_inverted_best_bounds behavior.
    fallback_bounds: BoundedTensor,
}

impl AlphaCrownBackend for InfiniteBoundsBackend {
    fn backward_iteration(
        &self,
        _alpha_state: &AlphaState,
        _input: &BoundedTensor,
        _iter: usize,
        _invprop_enabled: bool,
        _need_grad: bool,
    ) -> Result<Option<BackwardIterationResult>> {
        // Return None to signal CROWN fallback — this path is only hit
        // if the loop body runs (iterations > 0).
        Ok(None)
    }

    fn compute_gradients(
        &self,
        _config: &AlphaCrownConfig,
        _alpha_state: &mut AlphaState,
        _input: &BoundedTensor,
        _gradients: &[Array1<f32>],
        _gradients_upper: &[Array1<f32>],
        _iter: usize,
    ) -> Result<(Vec<Array1<f32>>, Vec<Array1<f32>>)> {
        Ok((vec![], vec![]))
    }

    fn crown_fallback(&self, _input: &BoundedTensor) -> Result<BoundedTensor> {
        Ok(self.fallback_bounds.clone())
    }

    fn log_label(&self) -> &str {
        "test-alpha"
    }
}

/// Regression test for #2909: Sequential α-CROWN must not discard optimization
/// progress when bounds contain infinities from clamp_inverted_best_bounds.
///
/// The old `is_finite()` check treated [-inf, +inf] overapproximations as
/// invalid, falling back to plain CROWN and discarding all element-wise
/// optimization progress. The fix uses `!is_nan()` instead (cf. DAG path
/// fix in #2854).
#[test]
fn test_alpha_crown_infinite_bounds_no_fallback_regression_2909() {
    // Bounds with one finite element and one infinite element (from inversion widening).
    let lower = arr1(&[-1.0_f32, f32::NEG_INFINITY]).into_dyn();
    let upper = arr1(&[1.0_f32, f32::INFINITY]).into_dyn();
    let fallback_bounds =
        BoundedTensor::new_allow_infinite(lower, upper).expect("infinite bounds should be valid");

    let mut backend = InfiniteBoundsBackend { fallback_bounds };

    // 0 iterations: the loop body never runs, so best_lower/best_upper
    // remain as crown_fallback output (which includes infinities).
    let config = AlphaCrownConfig {
        iterations: 0,
        ..AlphaCrownConfig::default()
    };

    let mut alpha_state = empty_alpha_state();

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let result = alpha_crown_optimize(&mut backend, &config, &mut alpha_state, &input, false)
        .expect("should succeed with infinite bounds, not fall back to CROWN");

    // Verify the result preserves the infinite bounds (not CROWN fallback).
    // The key invariant: infinite bounds ARE valid overapproximations, not errors.
    assert!(
        result.lower().iter().any(|v| v.is_infinite()),
        "infinite lower bound should be preserved, not discarded"
    );
    assert!(
        result.upper().iter().any(|v| v.is_infinite()),
        "infinite upper bound should be preserved, not discarded"
    );
    // No NaN in result
    assert!(
        !result.lower().iter().any(|v| v.is_nan()),
        "result should have no NaN"
    );
    assert!(
        !result.upper().iter().any(|v| v.is_nan()),
        "result should have no NaN"
    );
}

/// Mock backend that injects NaN into crown_fallback via new_unchecked.
///
/// The first crown_fallback call returns NaN bounds (simulating a computation
/// error). The second call (from the fallback path after has_nan detection)
/// returns finite "safe" bounds. This verifies the has_nan → fallback
/// control flow actually executes.
struct NanFallbackBackend {
    call_count: std::cell::Cell<usize>,
}

impl AlphaCrownBackend for NanFallbackBackend {
    fn backward_iteration(
        &self,
        _alpha_state: &AlphaState,
        _input: &BoundedTensor,
        _iter: usize,
        _invprop_enabled: bool,
        _need_grad: bool,
    ) -> Result<Option<BackwardIterationResult>> {
        Ok(None)
    }

    fn compute_gradients(
        &self,
        _config: &AlphaCrownConfig,
        _alpha_state: &mut AlphaState,
        _input: &BoundedTensor,
        _gradients: &[Array1<f32>],
        _gradients_upper: &[Array1<f32>],
        _iter: usize,
    ) -> Result<(Vec<Array1<f32>>, Vec<Array1<f32>>)> {
        Ok((vec![], vec![]))
    }

    fn crown_fallback(&self, _input: &BoundedTensor) -> Result<BoundedTensor> {
        let n = self.call_count.get();
        self.call_count.set(n + 1);
        if n == 0 {
            // First call: return NaN bounds via new_unchecked to bypass validation.
            // In production, NaN enters best_lower/best_upper via
            // update_elementwise_best_bounds during backward iterations.
            BoundedTensor::new_unchecked(
                arr1(&[-1.0_f32, f32::NAN]).into_dyn(),
                arr1(&[1.0_f32, 2.0]).into_dyn(),
            )
        } else {
            // Second call (fallback): return finite safe bounds.
            BoundedTensor::new(
                arr1(&[-1.0_f32, -0.5]).into_dyn(),
                arr1(&[1.0_f32, 2.0]).into_dyn(),
            )
        }
    }

    fn log_label(&self) -> &str {
        "test-nan-fallback"
    }
}

/// Regression test: NaN bounds SHOULD trigger CROWN fallback.
/// This is the correct behavior — NaN indicates computation errors, not
/// valid overapproximations.
///
/// Uses NanFallbackBackend to inject NaN via new_unchecked, ensuring the
/// post-loop `has_nan` guard in `alpha_crown_optimize` is actually exercised
/// (not short-circuited by BoundedTensor constructor validation).
#[test]
fn test_alpha_crown_nan_bounds_trigger_fallback_2909() {
    let mut backend = NanFallbackBackend {
        call_count: std::cell::Cell::new(0),
    };

    let config = AlphaCrownConfig {
        iterations: 0,
        ..AlphaCrownConfig::default()
    };

    let mut alpha_state = empty_alpha_state();

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    // The function should detect NaN in best bounds and fall back to CROWN.
    let result = alpha_crown_optimize(&mut backend, &config, &mut alpha_state, &input, false)
        .expect("NaN bounds should trigger CROWN fallback, not error");

    // Verify the fallback was taken: result should be the finite bounds from
    // the second crown_fallback call, not the NaN bounds from the first.
    assert!(
        !result.lower().iter().any(|v| v.is_nan()),
        "result lower should have no NaN after fallback"
    );
    assert!(
        !result.upper().iter().any(|v| v.is_nan()),
        "result upper should have no NaN after fallback"
    );
    // Verify the specific values from the second crown_fallback call.
    // This ensures we got the fallback result, not some other code path.
    assert_eq!(result.lower().as_slice().unwrap(), &[-1.0_f32, -0.5]);
    assert_eq!(result.upper().as_slice().unwrap(), &[1.0_f32, 2.0]);
    // crown_fallback was called twice: initial + fallback
    assert_eq!(
        backend.call_count.get(),
        2,
        "crown_fallback should be called exactly twice (initial + NaN fallback)"
    );
}

/// Regression test for #2597: `finite_lower_sum` must produce a finite result
/// even when the input array contains NaN and ±Inf elements.
///
/// Without the finite filter, `[1.0, NaN, 2.0].iter().sum()` = NaN,
/// poisoning the early-stopping metric and causing the loop to waste
/// all remaining iterations.
#[test]
fn test_finite_lower_sum_strips_nan_and_inf_2597() {
    // NaN and ±Inf elements must be excluded from the sum.
    let arr = arr1(&[
        1.0_f32,
        f32::NAN,
        2.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        3.0,
    ])
    .into_dyn();
    let sum = finite_lower_sum(&arr);
    assert_eq!(sum, 6.0, "finite_lower_sum should sum only finite elements");
    assert!(sum.is_finite(), "result must be finite");

    // All-NaN array should sum to 0.0 (empty sum).
    let all_nan = arr1(&[f32::NAN, f32::NAN]).into_dyn();
    assert_eq!(
        finite_lower_sum(&all_nan),
        0.0,
        "all-NaN array should sum to 0.0"
    );

    // All-finite array should sum normally.
    let finite = arr1(&[1.0_f32, 2.0, 3.0]).into_dyn();
    assert_eq!(finite_lower_sum(&finite), 6.0);
}

/// Mock backend that produces -Inf in backward pass lower bounds.
///
/// `backward_iteration` returns `LinearBounds` with -Inf bias, simulating
/// the result after `concretize_sound` repairs NaN → ±Inf. `finite_lower_sum`
/// strips -Inf from the early-stopping metric (defense layer 2), and the
/// in-loop NaN check provides defense layer 3.
struct InfBackwardBackend {
    backward_calls: std::cell::Cell<usize>,
}

impl AlphaCrownBackend for InfBackwardBackend {
    fn backward_iteration(
        &self,
        _alpha_state: &AlphaState,
        _input: &BoundedTensor,
        _iter: usize,
        _invprop_enabled: bool,
        _need_grad: bool,
    ) -> Result<Option<BackwardIterationResult>> {
        self.backward_calls.set(self.backward_calls.get() + 1);

        // LinearBounds with -Inf in lower bias — simulates the post-repair state
        // when NaN enters the backward pass and concretize_sound repairs NaN → -Inf.
        // Without finite_lower_sum, the sum would be -Inf, poisoning early stopping.
        let linear_bounds = LinearBounds::from_parts_unchecked(
            ndarray::Array2::eye(2),         // lower_a: identity
            arr1(&[f32::NEG_INFINITY, 0.0]), // lower_b: -Inf in element 0
            ndarray::Array2::eye(2),         // upper_a: identity
            arr1(&[1.0_f32, 1.0]),           // upper_b: valid
        );

        Ok(Some(BackwardIterationResult {
            linear_bounds,
            gradients: vec![],
            gradients_upper: vec![],
            bounds_without_oc: None,
        }))
    }

    fn compute_gradients(
        &self,
        _config: &AlphaCrownConfig,
        _alpha_state: &mut AlphaState,
        _input: &BoundedTensor,
        _gradients: &[Array1<f32>],
        _gradients_upper: &[Array1<f32>],
        _iter: usize,
    ) -> Result<(Vec<Array1<f32>>, Vec<Array1<f32>>)> {
        Ok((vec![], vec![]))
    }

    fn crown_fallback(&self, _input: &BoundedTensor) -> Result<BoundedTensor> {
        BoundedTensor::new(
            arr1(&[-1.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
    }

    fn log_label(&self) -> &str {
        "test-inf-backward"
    }
}

/// Regression test for #2597: -Inf in backward pass lower bounds must not
/// cause the optimization loop to waste all iterations.
///
/// In production, NaN from backward pass instability is repaired to -Inf by
/// `concretize_sound`. Without `finite_lower_sum`, the early-stopping metric
/// becomes `-Inf + finite = -Inf`, then `(-Inf) - (-Inf) = NaN`, disabling
/// convergence detection entirely. With `finite_lower_sum`, -Inf elements
/// are excluded and early stopping works correctly.
#[test]
fn test_alpha_crown_inf_backward_early_convergence_2597() {
    let mut backend = InfBackwardBackend {
        backward_calls: std::cell::Cell::new(0),
    };

    let config = AlphaCrownConfig {
        iterations: 20,
        ..AlphaCrownConfig::default()
    };

    let mut alpha_state = empty_alpha_state();

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let result = alpha_crown_optimize(&mut backend, &config, &mut alpha_state, &input, false)
        .expect("-Inf in backward pass should not cause error");

    // Result should be valid (no NaN) — best bounds include the -Inf element
    // but that's a sound overapproximation.
    assert!(
        !result.lower().iter().any(|v| v.is_nan()),
        "result should have no NaN"
    );
    assert!(
        !result.upper().iter().any(|v| v.is_nan()),
        "result should have no NaN"
    );

    // The loop should NOT have run all 20 iterations — early stopping
    // should trigger because the bounds with -Inf produce no improvement
    // over CROWN bounds (finite_lower_sum strips the -Inf elements).
    let calls = backend.backward_calls.get();
    assert!(
        calls < 20,
        "loop should stop early (got {calls} backward calls out of 20 iterations)"
    );
}

struct ScriptedBoundsBackend {
    fallback: BoundedTensor,
    scripted_bounds: Vec<BoundedTensor>,
    backward_calls: std::cell::Cell<usize>,
}

impl ScriptedBoundsBackend {
    fn linear_bounds_for(bounds: &BoundedTensor) -> LinearBounds {
        LinearBounds::from_parts_unchecked(
            ndarray::Array2::zeros((1, 1)),
            arr1(&[bounds.lower()[[0]]]),
            ndarray::Array2::zeros((1, 1)),
            arr1(&[bounds.upper()[[0]]]),
        )
    }
}

impl AlphaCrownBackend for ScriptedBoundsBackend {
    fn backward_iteration(
        &self,
        _alpha_state: &AlphaState,
        _input: &BoundedTensor,
        iter: usize,
        _invprop_enabled: bool,
        _need_grad: bool,
    ) -> Result<Option<BackwardIterationResult>> {
        self.backward_calls.set(self.backward_calls.get() + 1);
        let bounds = self
            .scripted_bounds
            .get(iter)
            .expect("test backend missing scripted bounds for iteration");
        Ok(Some(BackwardIterationResult {
            linear_bounds: Self::linear_bounds_for(bounds),
            gradients: vec![],
            gradients_upper: vec![],
            bounds_without_oc: None,
        }))
    }

    fn compute_gradients(
        &self,
        _config: &AlphaCrownConfig,
        _alpha_state: &mut AlphaState,
        _input: &BoundedTensor,
        _gradients: &[Array1<f32>],
        _gradients_upper: &[Array1<f32>],
        _iter: usize,
    ) -> Result<(Vec<Array1<f32>>, Vec<Array1<f32>>)> {
        Ok((vec![], vec![]))
    }

    fn crown_fallback(&self, _input: &BoundedTensor) -> Result<BoundedTensor> {
        Ok(self.fallback.clone())
    }

    fn log_label(&self) -> &str {
        "test-scripted-bounds"
    }
}

#[test]
fn test_alpha_crown_patience_force_saves_during_warmup_4380() {
    let mut backend = ScriptedBoundsBackend {
        fallback: scalar_bounds(0.0, 100.0),
        scripted_bounds: vec![
            scalar_bounds(1.0, 99.0),
            scalar_bounds(5.0, 95.0),
            scalar_bounds(5.0, 95.0),
        ],
        backward_calls: std::cell::Cell::new(0),
    };
    let config = AlphaCrownConfig {
        iterations: 10,
        start_save_best: 0.5,
        early_stop_patience: 1,
        ..AlphaCrownConfig::default()
    };
    let input = scalar_bounds(0.0, 100.0);

    let result = alpha_crown_optimize(
        &mut backend,
        &config,
        &mut empty_alpha_state(),
        &input,
        false,
    )
    .expect("patience force-save regression should optimize successfully");

    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];
    assert!(
        (lower - 5.0).abs() < 1.0e-5,
        "patience exit inside warmup must force-save the current best bound (got {lower})"
    );
    assert!(
        (upper - 95.0).abs() < 1.0e-5,
        "patience exit should preserve the paired upper bound from the saved iteration (got {upper})"
    );
    assert_eq!(
        backend.backward_calls.get(),
        3,
        "regression should stop once patience is exhausted inside the warmup window"
    );
}

/// Two individually weak but complementary alpha iterates.  On x in [-1, 1],
/// each lower plane (`x` and `-x`) concretizes to -1, while their 1/2 mixture
/// is the exact constant zero lower certificate.
struct ComplementaryFacetBackend;

#[test]
fn test_alpha_facet_collector_seeds_finite_crown_constant_anchor() {
    let settings = AlphaFacetBankSettings {
        enabled: true,
        max_planes: 4,
        max_bytes: usize::MAX,
    };
    let mut collector =
        AlphaFacetBankCollector::new(2, 1, settings).expect("collector should fit test budget");
    collector.capture_constant_lower(&arr1(&[-2.0, f32::NEG_INFINITY]).into_dyn());

    assert_eq!(collector.rows[0].len(), 1);
    assert_eq!(collector.rows[0][0].coefficients(), &[0.0]);
    assert_eq!(collector.rows[0][0].bias(), -2.0);
    assert!(
        collector.rows[1].is_empty(),
        "a non-finite CROWN lower bound must not become a certificate"
    );
}

impl AlphaCrownBackend for ComplementaryFacetBackend {
    fn backward_iteration(
        &self,
        _alpha_state: &AlphaState,
        _input: &BoundedTensor,
        iter: usize,
        _invprop_enabled: bool,
        _need_grad: bool,
    ) -> Result<Option<BackwardIterationResult>> {
        let coefficient = if iter == 0 { 1.0 } else { -1.0 };
        let linear_bounds = LinearBounds::from_parts_unchecked(
            ndarray::arr2(&[[coefficient]]),
            arr1(&[0.0]),
            ndarray::arr2(&[[0.0]]),
            arr1(&[2.0]),
        );
        Ok(Some(BackwardIterationResult {
            linear_bounds,
            gradients: vec![],
            gradients_upper: vec![],
            bounds_without_oc: None,
        }))
    }

    fn compute_gradients(
        &self,
        _config: &AlphaCrownConfig,
        _alpha_state: &mut AlphaState,
        _input: &BoundedTensor,
        _gradients: &[Array1<f32>],
        _gradients_upper: &[Array1<f32>],
        _iter: usize,
    ) -> Result<(Vec<Array1<f32>>, Vec<Array1<f32>>)> {
        Ok((vec![], vec![]))
    }

    fn crown_fallback(&self, _input: &BoundedTensor) -> Result<BoundedTensor> {
        Ok(scalar_bounds(-2.0, 2.0))
    }

    fn log_label(&self) -> &str {
        "test-complementary-facets"
    }
}

#[test]
fn test_alpha_crown_facet_bank_convexifies_retained_iterates() {
    let config = AlphaCrownConfig {
        iterations: 2,
        tolerance: 0.0,
        early_stop_patience: 10,
        ..AlphaCrownConfig::default()
    };
    let input = scalar_bounds(-1.0, 1.0);

    let disabled = alpha_crown_optimize_impl(
        &mut ComplementaryFacetBackend,
        &config,
        &mut empty_alpha_state(),
        &input,
        false,
        AlphaFacetBankSettings {
            enabled: false,
            max_planes: 4,
            max_bytes: usize::MAX,
        },
        false,
    )
    .expect("baseline alpha-CROWN should succeed");

    let enabled = alpha_crown_optimize_impl(
        &mut ComplementaryFacetBackend,
        &config,
        &mut empty_alpha_state(),
        &input,
        false,
        AlphaFacetBankSettings {
            enabled: true,
            max_planes: 4,
            max_bytes: usize::MAX,
        },
        false,
    )
    .expect("FacetBank alpha-CROWN should succeed");

    let baseline_lower = disabled.lower()[[0]];
    let facet_lower = enabled.lower()[[0]];
    assert!(
        baseline_lower <= -1.0,
        "individual planes should concretize no tighter than -1 (got {baseline_lower})"
    );
    assert!(
        facet_lower > baseline_lower,
        "FacetBank mixture must strictly improve {baseline_lower} (got {facet_lower})"
    );
    assert!(
        facet_lower <= 0.0 && facet_lower > -1.0e-4,
        "the outward-certified half mixture should be zero up to downward rounding (got {facet_lower})"
    );
    assert_eq!(enabled.upper()[[0]], disabled.upper()[[0]]);
}
