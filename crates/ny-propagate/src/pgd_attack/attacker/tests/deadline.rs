// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Actual attack/Network deadline boundaries; fault injection is test-only.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ndarray::{arr1, arr2};
use ny_core::{
    GemmEngine, GpuIbpForward, GpuIbpForwardExt, GpuIbpLayer, GpuIbpModelPlan, GpuIbpResult,
    NyError, Result,
};
use ny_tensor::BoundedTensor;

use super::super::PgdAttacker;
use super::common::CachedPlanCountingEngine;
use crate::layers::{Layer, LinearLayer};
use crate::network::ibp::{propagate_concrete_point_until, LeadingAxisMode};
use crate::pgd_attack::PgdConfig;
use crate::Network;

thread_local! {
    static EXPIRE_AT: RefCell<Option<(&'static str, usize)>> = const { RefCell::new(None) };
}

pub(in crate::pgd_attack) fn force_expiry_at(stage: &str) -> bool {
    EXPIRE_AT.with(|slot| {
        let mut state = slot.borrow_mut();
        let Some((selected, visits)) = state.as_mut() else {
            return false;
        };
        if *selected != stage {
            return false;
        }
        *visits = visits.saturating_sub(1);
        *visits == 0
    })
}

struct ExpiryFault(Option<(&'static str, usize)>);
impl ExpiryFault {
    fn at(stage: &'static str, visit: usize) -> Self {
        assert!(visit > 0);
        Self(EXPIRE_AT.with(|slot| slot.replace(Some((stage, visit)))))
    }
}
impl Drop for ExpiryFault {
    fn drop(&mut self) {
        EXPIRE_AT.with(|slot| {
            slot.replace(self.0);
        });
    }
}

fn network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), None).unwrap(),
    ));
    network
}

fn bounds() -> BoundedTensor {
    BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn()).unwrap()
}

fn expired() -> Instant {
    Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap()
}

#[test]
fn expired_attack_refuses_before_restart_allocation_or_seed_work() {
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: usize::MAX,
        deadline: Some(expired()),
        ..Default::default()
    });
    // This restart count cannot be allocated: an admission check must precede
    // every batched allocation, not just the later gradient loop.
    let error = attacker
        .attack(&network(), &bounds(), 0, 0.0, true)
        .unwrap_err();
    assert!(error.is_deadline_exceeded(), "{error:?}");
    assert!(attacker.deadline_exhausted());
}

#[test]
fn batched_initialization_polls_before_completing_all_restart_seeds() {
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 65,
        num_steps: 0,
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        ..Default::default()
    });
    let _fault = ExpiryFault::at("during batched seed initialization", 2);
    let error = attacker
        .attack(&network(), &bounds(), 0, 0.0, true)
        .unwrap_err();
    assert!(error.is_deadline_exceeded(), "{error:?}");
    assert!(error
        .to_string()
        .contains("during batched seed initialization"));
    assert!(attacker.deadline_exhausted());
}

#[test]
fn batched_expiry_before_final_evaluation_offers_no_unvalidated_witness() {
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 2,
        num_steps: 0,
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        ..Default::default()
    });
    // All prepared points satisfy y>=1>0. An expired final evaluation must
    // return typed expiry, rather than offer any unevaluated point as SAT.
    let _fault = ExpiryFault::at("before final batched evaluation", 1);
    let error = attacker
        .attack(&network(), &bounds(), 0, 0.0, true)
        .unwrap_err();
    assert!(error.is_deadline_exceeded(), "{error:?}");
    assert!(error
        .to_string()
        .contains("before final batched evaluation"));
}

#[test]
fn genuine_network_point_evaluation_honors_an_earlier_expired_phase() {
    let point = BoundedTensor::concrete(arr1(&[1.5_f32]).into_dyn()).unwrap();
    for mode in [LeadingAxisMode::Plain, LeadingAxisMode::PreserveLeadingAxis] {
        let error = propagate_concrete_point_until(&network(), &point, None, mode, Some(expired()))
            .unwrap_err();
        assert!(error.is_deadline_exceeded(), "{error:?}");
    }
}

#[test]
fn finite_phase_declines_legacy_cached_plan_and_evaluates_the_original_network() {
    let engine = CachedPlanCountingEngine::new();
    let attacker = PgdAttacker::new(PgdConfig {
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        ..Default::default()
    })
    .with_engine(&engine);
    let points = arr2(&[[1.0_f32], [2.0]]).into_dyn();
    let actual = attacker.evaluate_batch(&network(), &points).unwrap();
    assert_eq!(actual, points);
    assert_eq!(engine.plan_preparations(), 0);
    assert_eq!(engine.cached_calls(), 0);
    assert_eq!(engine.gemm_calls(), 0);
}

#[derive(Default)]
struct DeadlinePlanCounters {
    preparations: AtomicUsize,
    executions: AtomicUsize,
    deadlines: Mutex<Vec<Instant>>,
}

struct DeadlinePlan(Arc<DeadlinePlanCounters>);

impl GpuIbpModelPlan for DeadlinePlan {
    fn ibp_forward_cached(&self, _: &[f32], _: &[f32], _: &[usize]) -> Result<GpuIbpResult> {
        panic!("finite PGD must not call the unbounded resident method");
    }

    fn supports_deadline_execution(&self) -> bool {
        true
    }

    fn ibp_forward_cached_with_deadline(
        &self,
        lower: &[f32],
        upper: &[f32],
        shape: &[usize],
        deadline: Instant,
    ) -> Result<GpuIbpResult> {
        assert!(Instant::now() < deadline);
        assert_eq!(lower, upper);
        assert_eq!(shape, &[2, 1]);
        self.0.executions.fetch_add(1, Ordering::SeqCst);
        self.0.deadlines.lock().unwrap().push(deadline);
        // Actual authored source is y=x. This counting backend only supplies
        // an attack proposal; no GPU/UNSAT qualification is asserted.
        Ok(GpuIbpResult {
            lower_bounds: lower.to_vec(),
            upper_bounds: upper.to_vec(),
            output_shape: shape.to_vec(),
        })
    }
}

struct DeadlineEngine(Arc<DeadlinePlanCounters>);

impl GpuIbpForward for DeadlineEngine {
    fn ibp_forward_gpu(
        &self,
        _: &[GpuIbpLayer],
        _: &[f32],
        _: &[f32],
        _: &[usize],
    ) -> Result<GpuIbpResult> {
        panic!("deadline-qualified cached evaluation must not dispatch one-shot resident work");
    }
}

impl GpuIbpForwardExt for DeadlineEngine {
    fn prepare_model_plan(
        &self,
        _: &[GpuIbpLayer],
        _: &[usize],
    ) -> Result<Option<Box<dyn GpuIbpModelPlan>>> {
        panic!("finite PGD must not prepare an unbounded resident plan");
    }

    fn prepare_model_plan_with_deadline(
        &self,
        layers: &[GpuIbpLayer],
        shape: &[usize],
        deadline: Instant,
    ) -> Result<Option<Box<dyn GpuIbpModelPlan>>> {
        assert!(Instant::now() < deadline);
        assert_eq!(shape, &[2, 1]);
        assert_eq!(layers.len(), 1);
        self.0.preparations.fetch_add(1, Ordering::SeqCst);
        self.0.deadlines.lock().unwrap().push(deadline);
        Ok(Some(Box::new(DeadlinePlan(Arc::clone(&self.0)))))
    }
}

impl GemmEngine for DeadlineEngine {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("bounded resident dense chain must avoid per-layer fallback");
    }
    fn as_gpu_ibp_forward_ext(&self) -> Option<&dyn GpuIbpForwardExt> {
        Some(self)
    }
}

#[test]
fn finite_phase_reuses_a_deadline_capable_plan_with_the_original_absolute_cap() {
    let counts = Arc::new(DeadlinePlanCounters::default());
    let engine = DeadlineEngine(Arc::clone(&counts));
    let attacker = PgdAttacker::new(PgdConfig {
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        ..Default::default()
    })
    .with_engine(&engine);
    let original = attacker.effective_deadline().unwrap();
    let points = arr2(&[[1.0_f32], [2.0]]).into_dyn();
    // Retain the same Network allocation: the cache key binds its layer store.
    let source = network();
    assert_eq!(attacker.evaluate_batch(&source, &points).unwrap(), points);
    assert_eq!(attacker.evaluate_batch(&source, &points).unwrap(), points);
    assert_eq!(counts.preparations.load(Ordering::SeqCst), 1);
    assert_eq!(counts.executions.load(Ordering::SeqCst), 2);
    assert_eq!(*counts.deadlines.lock().unwrap(), vec![original; 3]);
}

#[test]
fn single_restart_expiry_preserves_typed_source_without_a_new_incumbent() {
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 1,
        num_steps: 0,
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        ..Default::default()
    });
    let _fault = ExpiryFault::at("before final single evaluation", 1);
    let error = attacker
        .attack(&network(), &bounds(), 0, 3.0, true)
        .unwrap_err();
    assert!(error.is_deadline_exceeded(), "{error:?}");
    assert!(attacker.deadline_exhausted());
}

#[test]
fn sequential_expiry_retains_only_the_completed_finite_incumbent() {
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 2,
        num_steps: 0,
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        ..Default::default()
    });
    let _fault = ExpiryFault::at("before final single evaluation", 2);
    let result = attacker
        .attack_sequential(&network(), &bounds(), 0, 3.0, true)
        .unwrap();
    assert!(!result.found_counterexample);
    let incumbent = result.counterexample.unwrap();
    assert!((1.0..=2.0).contains(&incumbent[[0]]));
    assert_eq!(result.output.unwrap(), incumbent);
    assert_eq!(result.total_evaluations, 1);
    assert_eq!(result.failed_restarts, 0);
    assert!(attacker.deadline_exhausted());
}

#[test]
fn batched_non_deadline_errors_keep_their_existing_failure_contract() {
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 2,
        num_steps: 0,
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        ..Default::default()
    });
    let error = attacker
        .attack(&network(), &bounds(), 7, 0.0, true)
        .unwrap_err();
    assert!(matches!(error, NyError::InternalError(_)), "{error:?}");
    assert!(!error.is_deadline_exceeded());
    assert!(error
        .to_string()
        .contains("all 2 restarts failed (batched lockstep)"));
    assert!(error.to_string().contains("output_idx 7 out of range"));
}

#[test]
fn live_batched_witness_retains_finite_input_and_actual_network_output() {
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 2,
        num_steps: 0,
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        ..Default::default()
    });
    let result = attacker
        .attack(&network(), &bounds(), 0, 0.0, true)
        .unwrap();
    assert!(result.found_counterexample);
    let input = result.counterexample.unwrap();
    let output = result.output.unwrap();
    assert!((1.0..=2.0).contains(&input[[0]]));
    assert_eq!(output, input);
    assert_eq!(result.restarts_completed, 2);
    assert_eq!(result.failed_restarts, 0);
}
