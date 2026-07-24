// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::arr1;
use std::cell::{Cell, RefCell};
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

fn unit_interval_input() -> BoundedTensor {
    BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("unit interval input should be valid")
}

struct PilotAbortBackend {
    backward_calls: Cell<usize>,
    gradient_calls: Cell<usize>,
    pilot_calls: Cell<usize>,
    events: RefCell<Vec<&'static str>>,
}

impl AlphaCrownBackend for PilotAbortBackend {
    fn backward_iteration(
        &self,
        _alpha_state: &AlphaState,
        _input: &BoundedTensor,
        iter: usize,
        _invprop_enabled: bool,
        _need_grad: bool,
    ) -> Result<Option<BackwardIterationResult>> {
        self.backward_calls.set(self.backward_calls.get() + 1);
        self.events.borrow_mut().push("backward");

        Ok(Some(BackwardIterationResult {
            linear_bounds: LinearBounds::from_parts_unchecked(
                ndarray::Array2::zeros((1, 1)),
                arr1(&[1.0_f32 + (iter as f32)]),
                ndarray::Array2::zeros((1, 1)),
                arr1(&[3.0_f32]),
            ),
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
    ) -> Result<DualGradients> {
        self.gradient_calls.set(self.gradient_calls.get() + 1);
        self.events.borrow_mut().push("compute_gradients");
        Ok((vec![], vec![]))
    }

    fn crown_fallback(&self, _input: &BoundedTensor) -> Result<BoundedTensor> {
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[4.0_f32]).into_dyn())
    }

    fn log_label(&self) -> &str {
        "pilot-abort-test"
    }

    fn post_bounds_update(
        &mut self,
        iter: usize,
        _improved_output: bool,
    ) -> Result<Option<ReferenceBoundsCandidate>> {
        self.events.borrow_mut().push("post_bounds_update");
        Ok(Some(ReferenceBoundsCandidate {
            data: Box::new(iter),
        }))
    }

    fn apply_reference_refresh(
        &mut self,
        candidate: ReferenceBoundsCandidate,
        _iter: usize,
    ) -> Result<()> {
        let refresh_iter = *candidate
            .data
            .downcast::<usize>()
            .expect("candidate should carry the source iteration");
        assert_eq!(refresh_iter, 0, "pilot abort should skip iter 1 refresh");
        self.events.borrow_mut().push("apply_reference_refresh");
        Ok(())
    }

    fn pilot_check(
        &self,
        _config: &AlphaCrownConfig,
        _best_lower_sum: f32,
        _crown_bounds: &BoundedTensor,
    ) -> bool {
        self.pilot_calls.set(self.pilot_calls.get() + 1);
        self.events.borrow_mut().push("pilot_check");
        true
    }

    fn update_extended_alphas(
        &mut self,
        _config: &AlphaCrownConfig,
        _lr: f32,
        _iter: usize,
        _total_gradient_skips: &mut usize,
    ) -> Result<()> {
        self.events.borrow_mut().push("update_extended_alphas");
        Ok(())
    }

    fn log_iteration_telemetry(&self, _iter: usize) {
        self.events.borrow_mut().push("log_iteration_telemetry");
    }
}

#[test]
fn test_alpha_crown_pilot_abort_skips_iter1_gradient_update_3751() {
    let mut backend = PilotAbortBackend {
        backward_calls: Cell::new(0),
        gradient_calls: Cell::new(0),
        pilot_calls: Cell::new(0),
        events: RefCell::new(Vec::new()),
    };

    let config = AlphaCrownConfig {
        iterations: 3,
        ..AlphaCrownConfig::default()
    };

    let mut alpha_state = empty_alpha_state();
    let input = unit_interval_input();

    let result = alpha_crown_optimize(&mut backend, &config, &mut alpha_state, &input, false)
        .expect("pilot abort should return current best bounds");

    assert!(
        !result.lower().iter().any(|value| value.is_nan()),
        "pilot abort should still return a valid lower bound tensor"
    );
    assert!(
        !result.upper().iter().any(|value| value.is_nan()),
        "pilot abort should still return a valid upper bound tensor"
    );
    assert_eq!(backend.backward_calls.get(), 2);
    assert_eq!(
        backend.gradient_calls.get(),
        1,
        "iter 1 must abort before compute_gradients"
    );
    assert_eq!(backend.pilot_calls.get(), 1);
    assert_eq!(
        backend.events.into_inner(),
        vec![
            "backward",
            "post_bounds_update",
            "compute_gradients",
            "update_extended_alphas",
            "apply_reference_refresh",
            "backward",
            "pilot_check",
        ]
    );
}

struct RefreshHooksBackend {
    applied_candidates: RefCell<Vec<usize>>,
    events: RefCell<Vec<&'static str>>,
}

impl AlphaCrownBackend for RefreshHooksBackend {
    fn backward_iteration(
        &self,
        _alpha_state: &AlphaState,
        _input: &BoundedTensor,
        _iter: usize,
        _invprop_enabled: bool,
        _need_grad: bool,
    ) -> Result<Option<BackwardIterationResult>> {
        self.events.borrow_mut().push("backward");
        Ok(Some(BackwardIterationResult {
            linear_bounds: LinearBounds::from_parts_unchecked(
                ndarray::Array2::zeros((1, 1)),
                arr1(&[1.0_f32]),
                ndarray::Array2::zeros((1, 1)),
                arr1(&[2.0_f32]),
            ),
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
    ) -> Result<DualGradients> {
        self.events.borrow_mut().push("compute_gradients");
        Ok((vec![], vec![]))
    }

    fn crown_fallback(&self, _input: &BoundedTensor) -> Result<BoundedTensor> {
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
    }

    fn log_label(&self) -> &str {
        "refresh-hooks-test"
    }

    fn post_bounds_update(
        &mut self,
        iter: usize,
        _improved_output: bool,
    ) -> Result<Option<ReferenceBoundsCandidate>> {
        self.events.borrow_mut().push("post_bounds_update");
        Ok(Some(ReferenceBoundsCandidate {
            data: Box::new(iter),
        }))
    }

    fn apply_reference_refresh(
        &mut self,
        candidate: ReferenceBoundsCandidate,
        _iter: usize,
    ) -> Result<()> {
        let refresh_iter = *candidate
            .data
            .downcast::<usize>()
            .expect("candidate should carry the source iteration");
        self.applied_candidates.borrow_mut().push(refresh_iter);
        self.events.borrow_mut().push("apply_reference_refresh");
        Ok(())
    }

    fn update_extended_alphas(
        &mut self,
        _config: &AlphaCrownConfig,
        _lr: f32,
        _iter: usize,
        _total_gradient_skips: &mut usize,
    ) -> Result<()> {
        self.events.borrow_mut().push("update_extended_alphas");
        Ok(())
    }

    fn log_iteration_telemetry(&self, _iter: usize) {
        self.events.borrow_mut().push("log_iteration_telemetry");
    }
}

#[test]
fn test_alpha_crown_refresh_candidate_applies_after_updates_3751() {
    let mut backend = RefreshHooksBackend {
        applied_candidates: RefCell::new(Vec::new()),
        events: RefCell::new(Vec::new()),
    };

    let config = AlphaCrownConfig {
        iterations: 1,
        ..AlphaCrownConfig::default()
    };

    let mut alpha_state = empty_alpha_state();
    let input = unit_interval_input();

    let _ = alpha_crown_optimize(&mut backend, &config, &mut alpha_state, &input, false)
        .expect("single iteration should succeed");

    assert_eq!(*backend.applied_candidates.borrow(), vec![0]);
    assert_eq!(
        backend.events.into_inner(),
        vec![
            "backward",
            "post_bounds_update",
            "compute_gradients",
            "update_extended_alphas",
            "apply_reference_refresh",
        ],
        "telemetry hook should stay gated behind DEBUG tracing"
    );
}

struct TerminalBoundOnlyBackend {
    need_grad: RefCell<Vec<bool>>,
    evaluated_alpha: RefCell<Vec<f32>>,
    last_evaluated_state: RefCell<Option<AlphaState>>,
    gradient_calls: Cell<usize>,
    post_bounds_calls: Cell<usize>,
    refresh_calls: Cell<usize>,
    extended_update_calls: Cell<usize>,
}

impl TerminalBoundOnlyBackend {
    fn new() -> Self {
        Self {
            need_grad: RefCell::new(Vec::new()),
            evaluated_alpha: RefCell::new(Vec::new()),
            last_evaluated_state: RefCell::new(None),
            gradient_calls: Cell::new(0),
            post_bounds_calls: Cell::new(0),
            refresh_calls: Cell::new(0),
            extended_update_calls: Cell::new(0),
        }
    }
}

impl AlphaCrownBackend for TerminalBoundOnlyBackend {
    fn backward_iteration(
        &self,
        alpha_state: &AlphaState,
        _input: &BoundedTensor,
        _iter: usize,
        _invprop_enabled: bool,
        need_grad: bool,
    ) -> Result<Option<BackwardIterationResult>> {
        let alpha = alpha_state
            .alpha(0)
            .and_then(|values| values.first())
            .copied()
            .expect("terminal-pass test needs one alpha");
        self.need_grad.borrow_mut().push(need_grad);
        self.evaluated_alpha.borrow_mut().push(alpha);
        *self.last_evaluated_state.borrow_mut() = Some(alpha_state.clone());
        Ok(Some(BackwardIterationResult {
            linear_bounds: LinearBounds::from_parts_unchecked(
                ndarray::Array2::zeros((1, 1)),
                arr1(&[alpha]),
                ndarray::Array2::zeros((1, 1)),
                arr1(&[1.0]),
            ),
            gradients: if need_grad {
                vec![arr1(&[1.0])]
            } else {
                vec![]
            },
            gradients_upper: if need_grad {
                vec![arr1(&[1.0])]
            } else {
                vec![]
            },
            bounds_without_oc: None,
        }))
    }

    fn compute_gradients(
        &self,
        _config: &AlphaCrownConfig,
        _alpha_state: &mut AlphaState,
        _input: &BoundedTensor,
        gradients: &[Array1<f32>],
        gradients_upper: &[Array1<f32>],
        _iter: usize,
    ) -> Result<DualGradients> {
        self.gradient_calls.set(self.gradient_calls.get() + 1);
        Ok((gradients.to_vec(), gradients_upper.to_vec()))
    }

    fn crown_fallback(&self, _input: &BoundedTensor) -> Result<BoundedTensor> {
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn())
    }

    fn log_label(&self) -> &str {
        "terminal-bound-only-test"
    }

    fn post_bounds_update(
        &mut self,
        iter: usize,
        _improved_output: bool,
    ) -> Result<Option<ReferenceBoundsCandidate>> {
        self.post_bounds_calls.set(self.post_bounds_calls.get() + 1);
        Ok(Some(ReferenceBoundsCandidate {
            data: Box::new(iter),
        }))
    }

    fn apply_reference_refresh(
        &mut self,
        candidate: ReferenceBoundsCandidate,
        _iter: usize,
    ) -> Result<()> {
        drop(candidate);
        self.refresh_calls.set(self.refresh_calls.get() + 1);
        Ok(())
    }

    fn update_extended_alphas(
        &mut self,
        _config: &AlphaCrownConfig,
        _lr: f32,
        _iter: usize,
        _total_gradient_skips: &mut usize,
    ) -> Result<()> {
        self.extended_update_calls
            .set(self.extended_update_calls.get() + 1);
        Ok(())
    }
}

fn run_terminal_bound_only(
    iterations: usize,
    final_bound_only: bool,
) -> (BoundedTensor, AlphaState, TerminalBoundOnlyBackend) {
    let mut backend = TerminalBoundOnlyBackend::new();
    let mut alpha_state = AlphaState::from_preactivation_bounds(
        &[
            BoundedTensor::new(arr1(&[-2.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("pre-activation bounds should be valid"),
        ],
        &[0],
    )
    .expect("one-alpha state should initialize");
    let config = AlphaCrownConfig {
        iterations,
        learning_rate: 0.1,
        lr_decay: 1.0,
        tolerance: 0.0,
        use_momentum: false,
        momentum: 0.0,
        optimizer: Optimizer::Adam,
        early_stop_patience: usize::MAX,
        start_save_best: 0.0,
        ..AlphaCrownConfig::default()
    };
    let bounds = alpha_crown_optimize_impl(
        &mut backend,
        &config,
        &mut alpha_state,
        &unit_interval_input(),
        false,
        AlphaFacetBankSettings {
            enabled: false,
            max_planes: 4,
            max_bytes: usize::MAX,
        },
        final_bound_only,
    )
    .expect("terminal bound-only test optimization should succeed");
    (bounds, alpha_state, backend)
}

#[test]
fn final_alpha_bound_only_gate_and_schedule_are_strict() {
    assert!(!parse_final_alpha_bound_only(None));
    for raw in ["", "0", "true", "TRUE", "on", " 1 ", "2"] {
        assert!(
            !parse_final_alpha_bound_only(Some(raw)),
            "raw={raw:?} must preserve the default-off path"
        );
    }
    assert!(parse_final_alpha_bound_only(Some("1")));

    assert!(alpha_iteration_needs_gradient(0, 0, true));
    assert!(!alpha_iteration_needs_gradient(0, 1, true));
    assert_eq!(
        (0..3)
            .map(|iter| alpha_iteration_needs_gradient(iter, 3, true))
            .collect::<Vec<_>>(),
        vec![true, true, false]
    );
    assert_eq!(
        (0..3)
            .map(|iter| alpha_iteration_needs_gradient(iter, 3, false))
            .collect::<Vec<_>>(),
        vec![true, true, true]
    );
}

#[test]
fn final_alpha_bound_only_preserves_bound_and_last_evaluated_state() {
    let (legacy_bounds, legacy_state, legacy_backend) = run_terminal_bound_only(3, false);
    let (bound_only_bounds, bound_only_state, bound_only_backend) =
        run_terminal_bound_only(3, true);

    let legacy_lower_bits: Vec<u32> = legacy_bounds.lower().iter().map(|v| v.to_bits()).collect();
    let legacy_upper_bits: Vec<u32> = legacy_bounds.upper().iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        bound_only_bounds
            .lower()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        legacy_lower_bits,
        "the gate must not change returned certified lower-bound bytes"
    );
    assert_eq!(
        bound_only_bounds
            .upper()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        legacy_upper_bits,
        "the gate must not change returned certified upper-bound bytes"
    );

    assert_eq!(*legacy_backend.need_grad.borrow(), vec![true, true, true]);
    assert_eq!(
        *bound_only_backend.need_grad.borrow(),
        vec![true, true, false]
    );
    assert_eq!(legacy_backend.gradient_calls.get(), 3);
    assert_eq!(bound_only_backend.gradient_calls.get(), 2);
    assert_eq!(legacy_backend.post_bounds_calls.get(), 3);
    assert_eq!(bound_only_backend.post_bounds_calls.get(), 2);
    assert_eq!(legacy_backend.refresh_calls.get(), 3);
    assert_eq!(bound_only_backend.refresh_calls.get(), 2);
    assert_eq!(legacy_backend.extended_update_calls.get(), 3);
    assert_eq!(bound_only_backend.extended_update_calls.get(), 2);

    let last_evaluated = *bound_only_backend
        .evaluated_alpha
        .borrow()
        .last()
        .expect("three iterations should evaluate alpha");
    assert_eq!(
        bound_only_state.alpha(0).expect("alpha should persist")[0].to_bits(),
        last_evaluated.to_bits(),
        "persisted alpha must be the exact state that produced the terminal bound"
    );
    let evaluated_state = bound_only_backend.last_evaluated_state.borrow();
    let evaluated_state = evaluated_state
        .as_ref()
        .expect("terminal iteration should snapshot evaluated state");
    assert_eq!(&bound_only_state.alphas, &evaluated_state.alphas);
    assert_eq!(
        &bound_only_state.alphas_upper,
        &evaluated_state.alphas_upper
    );
    assert_eq!(
        &bound_only_state.unstable_mask,
        &evaluated_state.unstable_mask
    );
    assert_eq!(&bound_only_state.velocity, &evaluated_state.velocity);
    assert_eq!(
        &bound_only_state.velocity_upper,
        &evaluated_state.velocity_upper
    );
    assert_eq!(&bound_only_state.adam_m, &evaluated_state.adam_m);
    assert_eq!(&bound_only_state.adam_v, &evaluated_state.adam_v);
    assert_eq!(
        &bound_only_state.adam_m_upper,
        &evaluated_state.adam_m_upper
    );
    assert_eq!(
        &bound_only_state.adam_v_upper,
        &evaluated_state.adam_v_upper
    );
    assert_eq!(
        &bound_only_state.bilinear_alphas,
        &evaluated_state.bilinear_alphas
    );
    assert_eq!(
        &bound_only_state.bilinear_adam_m,
        &evaluated_state.bilinear_adam_m
    );
    assert_eq!(
        &bound_only_state.bilinear_adam_v,
        &evaluated_state.bilinear_adam_v
    );
    assert!(bound_only_state.invprop_state.is_none());
    assert!(evaluated_state.invprop_state.is_none());
    assert_ne!(
        legacy_state.alpha(0).expect("legacy alpha should persist")[0].to_bits(),
        last_evaluated.to_bits(),
        "legacy behavior performs one unusable post-evaluation update"
    );
}

#[test]
fn final_alpha_bound_only_handles_zero_and_one_iteration() {
    let (_zero_bounds, _zero_state, zero_backend) = run_terminal_bound_only(0, true);
    assert!(zero_backend.need_grad.borrow().is_empty());
    assert_eq!(zero_backend.gradient_calls.get(), 0);

    let (_one_bounds, one_state, one_backend) = run_terminal_bound_only(1, true);
    assert_eq!(*one_backend.need_grad.borrow(), vec![false]);
    assert_eq!(one_backend.gradient_calls.get(), 0);
    assert_eq!(one_backend.post_bounds_calls.get(), 0);
    assert_eq!(one_backend.refresh_calls.get(), 0);
    assert_eq!(one_backend.extended_update_calls.get(), 0);
    assert_eq!(
        one_state.alpha(0).expect("one alpha should persist")[0].to_bits(),
        one_backend.evaluated_alpha.borrow()[0].to_bits()
    );
}
