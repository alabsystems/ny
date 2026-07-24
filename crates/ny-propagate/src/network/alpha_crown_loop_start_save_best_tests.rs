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
fn test_alpha_crown_warmup_skips_noisy_intermediate_best_bounds_4380() {
    let mut backend = ScriptedBoundsBackend {
        fallback: scalar_bounds(0.0, 100.0),
        scripted_bounds: vec![
            scalar_bounds(0.0, 100.0),
            scalar_bounds(10.0, 100.0),
            scalar_bounds(20.0, 100.0),
            scalar_bounds(1.0, 90.0),
        ],
        backward_calls: std::cell::Cell::new(0),
    };
    let config = AlphaCrownConfig {
        iterations: 4,
        start_save_best: 0.5,
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
    .expect("warmup skip regression should optimize successfully");

    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];
    assert!(
        (lower - 1.0).abs() < 1.0e-5,
        "warmup iterations must not overwrite best bounds with noisy intermediates (got {lower})"
    );
    assert!(
        (upper - 90.0).abs() < 1.0e-5,
        "final post-warmup iteration should still save element-wise best bounds (got {upper})"
    );
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
