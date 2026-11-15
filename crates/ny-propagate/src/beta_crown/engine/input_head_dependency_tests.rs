// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Actual numerical consumer regressions. A negative scalar MAX is scheduling
//! information in the exact zero/lower route; no mocked Verified result,
//! borrowed parent plane, or manufactured Source leaf supplies authority.

use super::*;
use crate::beta_crown::bab_cuts::CutPool;
use crate::beta_crown::engine::cut_gate::CutGateState;
use crate::beta_crown::engine::verify_phases::InitialPhaseOutcome;
use crate::layers::{LinearLayer, MaxPool2dLayer, ReLULayer, ReshapeLayer};
use crate::margin_row::{
    test_verify_input_head_retained_with_provider, verify_input_head_retained, HeadGroupPolicy,
    InputHeadOutcome, InputHeadSpec,
};
use crate::{BabVerificationStatus, BetaCrownConfig, BranchingHeuristic, Layer};
use ndarray::{arr1, arr2};
use ny_core::head_functional_advice::{HeadFunctionalLpHint, HeadFunctionalLpRequest};
use ny_core::head_phase_advice::{HeadPhaseAdvice, HeadPhaseLpHint, HeadPhaseLpRequest};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

fn config() -> BetaCrownConfig {
    BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        use_alpha_crown: false,
        use_crown_ibp: false,
        // Dense networks run full CROWN: their configured backward cap is
        // deliberately ignored. The improving fixture instead supplies a
        // completed Source-owned joint row, not a fabricated shallow bound.
        max_crown_ibp_nodes: Some(1),
        enable_cuts: false,
        enable_proactive_cuts: false,
        enable_relaxed_clip: false,
        enable_pgd_attack: false,
        max_domains: 4,
        max_depth: 1,
        timeout: Duration::from_secs(10),
        ..Default::default()
    }
}

fn root() -> BoundedTensor {
    BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap()
}

fn point() -> BoundedTensor {
    BoundedTensor::new(arr1(&[1.0, 0.0]).into_dyn(), arr1(&[1.0, 0.0]).into_dyn()).unwrap()
}

fn network(bias: f32, tail_sign: f32) -> Network {
    let mut net = Network::new();
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0], [1.0, -1.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
    ));
    net.add_layer(Layer::ReLU(ReLULayer::new()));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[-1.0, -1.0], [0.0, 0.0]]), Some(arr1(&[bias, 0.0]))).unwrap(),
    ));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[tail_sign, -tail_sign]]), None).unwrap(),
    ));
    net.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 1, 1])));
    net.add_layer(Layer::MaxPool2d(MaxPool2dLayer::new(
        (1, 1),
        (1, 1),
        (0, 0),
    )));
    net.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1])));
    net
}

fn fixture(bias: f32) -> (BetaCrownVerifier, Network, BoundedTensor, Instant) {
    let end = Instant::now() + Duration::from_secs(10);
    let spec = InputHeadSpec::new(
        2,
        2,
        vec![1.0, 1.0, 1.0, -1.0],
        vec![0.0, 0.0],
        vec![-1.0, 0.0, -1.0, 0.0],
        vec![bias, 0.0],
        [121; 32],
    )
    .unwrap();
    let (outcome, carry) = verify_input_head_retained(
        &spec,
        &[-1.0, -1.0],
        &[1.0, 1.0],
        0,
        1,
        HeadGroupPolicy::new(1.0, 1, 1).unwrap(),
        end,
        &[-1.0, -1.0],
        &[1.0, 1.0],
        end,
    );
    let InputHeadOutcome::Unknown(report) = outcome else {
        panic!("the actual native root scalar relaxation must remain open")
    };
    assert!(report.lower.unwrap() <= 0.0);
    assert_eq!(report.advice_calls, 0, "no provider or extra advice grant");
    let carry = carry.expect("actual completed native open functional");
    let net = network(bias, 1.0);
    assert!(carry.matches_scalar_network(&net));
    (
        BetaCrownVerifier::new(config()).with_input_head_continuation(carry),
        net,
        root(),
        end,
    )
}

/// Deterministic finite multipliers only. The actual Source first certifies
/// all four masks of -rho(x+z)-rho(x-z), including independent first-layer
/// error balls. No returned bound, solver status or manufactured leaf exists.
struct JointAdvice {
    calls: AtomicUsize,
}

impl HeadPhaseAdvice for JointAdvice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }

    fn supports_functional_advice(&self) -> bool {
        true
    }

    fn propose_functional(
        &self,
        request: &HeadFunctionalLpRequest,
        deadline: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        assert!(request.validate() && Instant::now() < deadline);
        assert_eq!(request.feature_count, 2);
        assert_eq!(request.bounds.len(), 4);
        self.calls.fetch_add(1, Ordering::SeqCst);
        let row = request
            .rows
            .iter()
            .position(|row| {
                row.coefficients.len() == 2
                    && row
                        .coefficients
                        .iter()
                        .enumerate()
                        .all(|(at, &(j, a))| j == 2 + at && a < -0.99)
            })
            .expect("the complete Source-owned two-negative-head row must arrive");
        assert!(
            request.rows[row].lower <= -2.0,
            "the joint RHS must retain independent first-layer execution debt"
        );
        let mut row_duals = vec![0.0; request.rows.len()];
        row_duals[row] = 1.0;
        Some(HeadFunctionalLpHint { row_duals })
    }
}

fn joint_fixture(
    bias: f32,
) -> (
    BetaCrownVerifier,
    Network,
    BoundedTensor,
    Instant,
    Arc<JointAdvice>,
) {
    let end = Instant::now() + Duration::from_secs(10);
    let spec = InputHeadSpec::new(
        2,
        2,
        vec![1.0, 1.0, 1.0, -1.0],
        vec![0.0, 0.0],
        vec![-1.0, 0.0, -1.0, 0.0],
        vec![bias, 0.0],
        [123; 32],
    )
    .unwrap();
    let advice = Arc::new(JointAdvice {
        calls: AtomicUsize::new(0),
    });
    let (outcome, carry) = test_verify_input_head_retained_with_provider(
        &spec,
        &[-1.0, -1.0],
        &[1.0, 1.0],
        0,
        1,
        HeadGroupPolicy::new(1.0, 1, 1).unwrap(),
        end,
        &[-1.0, -1.0],
        &[1.0, 1.0],
        end,
        || Some(Arc::clone(&advice) as Arc<dyn HeadPhaseAdvice>),
    );
    let InputHeadOutcome::Unknown(report) = outcome else {
        panic!("completed joint replay must remain open under ROOT execution debt")
    };
    assert!(report.lower.unwrap().is_finite() && report.lower.unwrap() < 0.0);
    assert_eq!(
        report.advice_calls, 1,
        "all-negative legacy export declines"
    );
    assert_eq!(report.triangle_advice_calls, 1);
    assert_eq!(report.triangle_negative_groups, 1);
    assert_eq!(report.triangle_negative_group_masks, 4);
    assert_eq!(report.triangle_classes_improved, 1);
    assert!(
        report.triangle_after_lower.unwrap() > report.triangle_before_lower.unwrap(),
        "actual complete-mask replay must improve the completed native floor"
    );
    assert_eq!(advice.calls.load(Ordering::SeqCst), 1);
    let carry = carry.expect("actual completed joint residual must be retained");
    let net = network(bias, 1.0);
    assert!(carry.matches_scalar_network(&net));
    (
        BetaCrownVerifier::new(config()).with_input_head_continuation(carry),
        net,
        root(),
        end,
        advice,
    )
}

fn small_margin() -> f32 {
    // True mathematical minimum2^-20. The Source deliberately retains its
    // much larger ROOT FP32 E1/Eout/tail debt, even on the later point box.
    // This makes the independent generic point proof genuinely necessary.
    2.0 + 1.0 / 1_048_576.0
}

fn complete_generic_point_proof(v: &BetaCrownVerifier, net: &Network, end: Instant) {
    let full = config();
    let independent = BetaCrownVerifier::new(full.clone());
    let p = point();
    let own = independent
        .verify_with_engine(net, &p, 0.0, None, Some(end))
        .unwrap();
    assert!(matches!(own.result, BabVerificationStatus::Verified));
    let bounds = own
        .output_bounds
        .as_ref()
        .expect("actual generic numerical output");
    assert!(bounds.lower_scalar() > 0.0);
    assert!(
        bounds.lower_scalar() <= small_margin() - 2.0,
        "actual generic lower cannot exceed the exact dyadic point margin"
    );
    assert!(
        v.input_head_support(net, &p, 0.0, Some(end)).unwrap() < 0.0,
        "ROOT execution-error debt keeps every retained Source floor open"
    );
    let with_source = v
        .with_config_from(full)
        .verify_with_engine(net, &p, 0.0, None, Some(end))
        .unwrap();
    assert!(matches!(
        with_source.result,
        BabVerificationStatus::Verified
    ));
    assert_eq!(with_source.domains_explored, own.domains_explored);
    let supplied = with_source.output_bounds.unwrap();
    assert_eq!(
        supplied.lower_scalar().to_bits(),
        bounds.lower_scalar().to_bits()
    );
    assert_eq!(
        supplied.upper_scalar().to_bits(),
        bounds.upper_scalar().to_bits()
    );
    assert!(!v.input_head_continuation().unwrap().dependent_use());
    assert!(v
        .input_head_continuation()
        .unwrap()
        .try_finish(end)
        .is_none());
}

#[test]
fn negative_root_max_preserves_actual_independent_generic_point_proof() {
    let (v, net, root, end, advice) = joint_fixture(small_margin());
    let generic = BetaCrownVerifier::new(config())
        .compute_initial_bounds_and_layer_bounds_engine(
            &net,
            &root,
            Some((0.0, false)),
            None,
            Some(end),
        )
        .unwrap();
    let source_floor = v.input_head_support(&net, &root, 0.0, Some(end)).unwrap();
    assert!(
        source_floor <= 0.0 && source_floor > generic.output_bounds.lower_scalar(),
        "actual nonpositive joint Source floor {source_floor} must improve full CROWN {}",
        generic.output_bounds.lower_scalar()
    );
    let gate = CutGateState::new(&v.config);
    let mut pool = CutPool::new(0);
    let outcome = v
        .evaluate_initial_phase(
            &net,
            &root,
            0.0,
            None,
            Instant::now(),
            &gate,
            &mut pool,
            Some(end),
        )
        .unwrap();
    let InitialPhaseOutcome::Proceed { initial_bounds, .. } = outcome else {
        panic!("a completed nonpositive MAX cannot close the root")
    };
    assert_eq!(
        initial_bounds.lower_scalar().to_bits(),
        source_floor.to_bits()
    );
    assert!(!v.input_head_continuation().unwrap().dependent_use());
    complete_generic_point_proof(&v, &net, end);
    assert_eq!(
        advice.calls.load(Ordering::SeqCst),
        1,
        "root/point consumers cannot request fresh advice"
    );
}

#[test]
fn negative_child_max_preserves_actual_independent_generic_point_proof() {
    let (v, net, root, end, advice) = joint_fixture(small_margin());
    let parent = BabDomain::root_with_input(vec![], -3.0, 3.0, &root).unwrap();
    // A genuine proper input split. Neither the Source nor full CROWN
    // can close the [−1,1/2]×[−1,1] child.
    let generic = BetaCrownVerifier::new(config())
        .create_input_split_child(&net, &root, &parent, 0, -1.0, 0.5, 0.0, Some(end), None)
        .unwrap()
        .unwrap();
    let with_source = v
        .create_input_split_child(&net, &root, &parent, 0, -1.0, 0.5, 0.0, Some(end), None)
        .unwrap()
        .unwrap();
    assert!(with_source.lower_bound <= 0.0);
    assert!(
        with_source.lower_bound > generic.lower_bound,
        "actual nonpositive child MAX {} must improve full CROWN {}",
        with_source.lower_bound,
        generic.lower_bound
    );
    assert_eq!(
        with_source.upper_bound.to_bits(),
        generic.upper_bound.to_bits()
    );
    assert_eq!(with_source.layer_bounds.len(), generic.layer_bounds.len());
    for (source, own) in with_source.layer_bounds.iter().zip(&generic.layer_bounds) {
        assert_eq!(
            source.lower(),
            own.lower(),
            "the scalar MAX never changes generic intermediate boxes"
        );
        assert_eq!(source.upper(), own.upper());
    }
    assert!(!v.input_head_continuation().unwrap().dependent_use());
    complete_generic_point_proof(&v, &net, end);
    assert_eq!(
        advice.calls.load(Ordering::SeqCst),
        1,
        "child/point consumers cannot request fresh advice"
    );
}

#[test]
fn actual_positive_source_prune_and_missing_sibling_still_taint_generic_proof() {
    let end = Instant::now() + Duration::from_secs(10);
    let spec = InputHeadSpec::new(
        1,
        1,
        vec![1.0],
        vec![0.0],
        vec![-1.0, 0.0],
        vec![0.5, 0.0],
        [122; 32],
    )
    .unwrap();
    let (outcome, carry) = verify_input_head_retained(
        &spec,
        &[-1.0],
        &[1.0],
        0,
        1,
        HeadGroupPolicy::new(1.0, 1, 1).unwrap(),
        end,
        &[-1.0],
        &[1.0],
        end,
    );
    assert!(matches!(outcome, InputHeadOutcome::Unknown(_)));
    let mut net = Network::new();
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    net.add_layer(Layer::ReLU(ReLULayer::new()));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[-1.0], [0.0]]), Some(arr1(&[0.5, 0.0]))).unwrap(),
    ));
    net.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, -1.0]]), None).unwrap(),
    ));
    net.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 1, 1])));
    net.add_layer(Layer::MaxPool2d(MaxPool2dLayer::new(
        (1, 1),
        (1, 1),
        (0, 0),
    )));
    net.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1])));
    let root = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let safe = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[-0.25]).into_dyn()).unwrap();
    let source = carry.unwrap();
    let v = BetaCrownVerifier::new(config()).with_input_head_continuation(Arc::clone(&source));
    let parent = BabDomain::root_with_input(vec![], -1.0, 1.0, &root).unwrap();
    assert!(v
        .create_input_split_child(&net, &root, &parent, 0, -1.0, -0.25, 0.0, Some(end), None)
        .unwrap()
        .is_none());
    assert!(source.dependent_use());
    assert!(
        source.try_finish(end).is_none(),
        "unsafe sibling remains outside the private cover"
    );
    let own = BetaCrownVerifier::new(config())
        .verify_with_engine(&net, &safe, 0.0, None, Some(end))
        .unwrap();
    assert!(
        matches!(own.result, BabVerificationStatus::Verified),
        "real generic numerical proof exists on this child"
    );
    let dependent = v
        .verify_with_engine(&net, &safe, 0.0, None, Some(end))
        .unwrap();
    assert!(matches!(
        dependent.result,
        BabVerificationStatus::Unknown { .. }
    ));
    assert!(source.try_finish(end).is_none());
}

#[test]
fn preexisting_positive_or_unknown_cache_taint_is_never_cleared_by_negative_max() {
    let (v, net, root, end) = fixture(small_margin());
    let source = v.input_head_continuation().unwrap();
    // The unconditional API remains the conservative contract for any
    // unsupported/future cache path. A late negative scalar cannot untaint it.
    source.mark_dependent_use();
    v.mark_input_head_scalar_max_dependency(&net, 0.0, -0.25, 1, Some(end));
    assert!(source.dependent_use());
    let mut full = config();
    full.crown_backward_layers = None;
    let own = BetaCrownVerifier::new(full.clone())
        .verify_with_engine(&net, &point(), 0.0, None, Some(end))
        .unwrap();
    assert!(matches!(own.result, BabVerificationStatus::Verified));
    let result = v
        .with_config_from(full)
        .verify_with_engine(&net, &point(), 0.0, None, Some(end))
        .unwrap();
    assert!(matches!(
        result.result,
        BabVerificationStatus::Unknown { .. }
    ));
    assert!(source.try_finish(end).is_none());
    assert!(v.input_head_support(&net, &root, 0.0, Some(end)).unwrap() < 0.0);
}

#[test]
fn foreign_tail_mode_threshold_and_nonscalar_consumption_keep_conservative_taint() {
    for case in 0..6 {
        let (mut v, mut net, _, end) = fixture(small_margin());
        let mut threshold = 0.0;
        let mut count = 1;
        match case {
            0 => net = network(small_margin(), -1.0),
            1 => {
                let mut c = config();
                c.verify_upper_bound = true;
                v = v.with_config_from(c);
            }
            2 => {
                let mut c = config();
                c.branching_heuristic = BranchingHeuristic::LargestBoundWidth;
                v = v.with_config_from(c);
            }
            3 => threshold = -0.5,
            4 => count = 2,
            _ => net = network(2.5, 1.0),
        }
        v.mark_input_head_scalar_max_dependency(&net, threshold, -0.25, count, Some(end));
        assert!(
            v.input_head_continuation().unwrap().dependent_use(),
            "case{case} must decline proof-neutral classification"
        );
    }
}

#[test]
fn zero_signedzero_and_equality_do_not_close_unsafe_original_instance() {
    for floor in [0.0, -0.0, -f32::from_bits(1)] {
        let (v, net, root, end) = fixture(2.0);
        v.mark_input_head_scalar_max_dependency(&net, 0.0, floor, 1, Some(end));
        assert!(!v.input_head_continuation().unwrap().dependent_use());
        assert!(!v.config.domain_is_verified(floor, 1.0, 0.0));
        assert!(
            v.input_head_support(&net, &point(), 0.0, Some(end))
                .unwrap()
                <= 0.0
        );
        let result = v
            .verify_with_engine(&net, &point(), 0.0, None, Some(end))
            .unwrap();
        assert!(
            !matches!(result.result, BabVerificationStatus::Verified),
            "Y0==Y1 remains unsafe"
        );
        assert!(v
            .input_head_continuation()
            .unwrap()
            .try_finish(end)
            .is_none());
        assert!(v.input_head_support(&net, &root, 0.0, Some(end)).unwrap() <= 0.0);
    }
    for invalid in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
    ] {
        let (v, net, _, end) = fixture(small_margin());
        v.mark_input_head_scalar_max_dependency(&net, 0.0, invalid, 1, Some(end));
        assert!(v.input_head_continuation().unwrap().dependent_use());
    }
}

#[test]
fn absent_or_expired_phase_cap_never_exempts_consumed_floor() {
    for cap in [
        None,
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
    ] {
        let (v, net, _, _) = fixture(small_margin());
        v.mark_input_head_scalar_max_dependency(&net, 0.0, -0.25, 1, cap);
        assert!(v.input_head_continuation().unwrap().dependent_use());
    }
    // The explicit caller cap cannot extend the producer's sealed authority.
    let (v, net, _, end) = fixture(small_margin());
    let later = end + Duration::from_secs(10);
    assert!(
        v.input_head_live(Some(later)),
        "same live original authority, no fresh allowance"
    );
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(!v.input_head_live(Some(expired)));
    assert!(v
        .input_head_support(&net, &root(), 0.0, Some(expired))
        .is_none());
    v.mark_input_head_scalar_max_dependency(&net, 0.0, -0.25, 1, Some(expired));
    assert!(v.input_head_continuation().unwrap().dependent_use());
}

#[cfg(target_arch = "x86_64")]
#[test]
#[allow(unsafe_code, deprecated)]
fn foreign_worker_fp_cannot_exempt_a_delayed_negative_max() {
    let (v, net, _, end) = fixture(small_margin());
    std::thread::spawn(move || {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        struct Restore(u32);
        impl Drop for Restore {
            fn drop(&mut self) {
                unsafe {
                    _mm_setcsr(self.0);
                }
            }
        }
        let saved = unsafe { _mm_getcsr() };
        let _restore = Restore(saved);
        unsafe {
            _mm_setcsr(saved | (1 << 15) | (1 << 6));
        }
        assert!(ny_core::require_f64_interval_proof_environment().is_err());
        assert!(v
            .input_head_support(&net, &root(), 0.0, Some(end))
            .is_none());
        v.mark_input_head_scalar_max_dependency(&net, 0.0, -0.25, 1, Some(end));
        assert!(v.input_head_continuation().unwrap().dependent_use());
    })
    .join()
    .unwrap();
}
