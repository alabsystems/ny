// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use ny_core::head_functional_advice::HeadFunctionalLpHint;
use ny_core::head_phase_advice::{HeadPhaseLpHint, HeadPhaseLpRequest};

#[path = "input_head_bridge_compact_tests.rs"]
mod compact_tail;

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(3)
}
fn expired() -> Instant {
    Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
}
fn policy() -> HeadGroupPolicy {
    HeadGroupPolicy::new(1.0, 1, 1).unwrap()
}

// M(x)=rho(x)-rho(-x)+B=x+B. It is genuinely unsafe at x=-1.
fn unsafe_spec(bias: f32) -> InputHeadSpec {
    InputHeadSpec::new(
        1,
        2,
        vec![1.0, -1.0],
        vec![0.0; 2],
        vec![1.0, 0.0, -1.0, 0.0],
        vec![bias, 0.0],
        [91; 32],
    )
    .unwrap()
}
fn unsafe_source() -> Source {
    Source::build(&unsafe_spec(1.0 / 16.0), &[-1.0], &[1.0], 0, 1, deadline()).unwrap()
}
fn continuation(s: &Source, slopes: &[Vec<f64>]) -> Arc<InputHeadContinuation> {
    let forms = slopes
        .iter()
        .map(|v| s.replay_retained(v, deadline()).unwrap().1)
        .collect::<Vec<_>>();
    let recipes = slopes
        .iter()
        .cloned()
        .map(Recipe::Slopes)
        .collect::<Vec<_>>();
    InputHeadContinuation::build(s, &forms, &recipes, &s.lo, &s.hi, deadline(), deadline()).unwrap()
}

#[test]
fn unsafe_whole_instance_has_a_genuine_positive_child_but_no_complete_cover() {
    let s = unsafe_source();
    let (_, form) = s.replay_retained(&[1.0, -0.5], deadline()).unwrap();
    let root = vec![I { lo: -1.0, hi: 1.0 }];
    let relu = s.relu_boxes_on(&root, deadline()).unwrap();
    assert!(form.support(&s, &root, &relu, deadline()).unwrap() <= -31.0 / 16.0);
    let child = vec![I { lo: 0.5, hi: 1.0 }];
    let r = s.relu_boxes_on(&child, deadline()).unwrap();
    let restricted = form.support(&s, &child, &r, deadline()).unwrap();
    assert!(restricted > 0.30 && restricted <= 5.0 / 16.0);
    let carry = continuation(&s, &[vec![1.0, -0.5]]);
    assert!(carry.try_support_box(&[0.5], &[1.0], deadline()).unwrap() > 0.30);
    assert!(carry.try_support_box(&[-1.0], &[0.5], deadline()).unwrap() < 0.0);
    // The unsafe sibling is never evidence, even if a generic consumer has
    // removed it for some unrelated reason or marked the queue empty.
    carry.mark_dependent_use();
    assert!(carry.dependent_use());
    assert!(!carry.has_complete_source_cover(deadline()));
    assert!(carry.try_finish(deadline()).is_none());
    assert_eq!((-1.0f64).max(0.0) - 1.0 + 1.0 / 16.0, -15.0 / 16.0);
}

#[test]
fn two_source_owned_affine_cells_authorize_the_original_singleton_instance() {
    // This is a cover/authorizer oracle, not an old-producer separation claim.
    // M=rho(x+1/2)+rho(-x+1/2)-1/2+1/16 is strictly positive.
    let spec = InputHeadSpec::new(
        1,
        2,
        vec![1.0, -1.0],
        vec![0.5, 0.5],
        vec![1.0, 0.0, 1.0, 0.0],
        vec![-7.0 / 16.0, 0.0],
        [92; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0], &[1.0], 0, 1, deadline()).unwrap();
    let carry = continuation(&s, &[vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert!(carry.try_support_box(&[-1.0], &[1.0], deadline()).unwrap() < 0.0);
    assert!(carry.try_support_box(&[-1.0], &[0.0], deadline()).unwrap() > 1.0 / 32.0);
    assert!(!carry.has_complete_source_cover(deadline()));
    assert!(carry.try_finish(deadline()).is_none());
    assert!(carry.try_support_box(&[0.0], &[1.0], deadline()).unwrap() > 1.0 / 32.0);
    assert!(carry.has_complete_source_cover(deadline()));
    let token = carry
        .try_finish(deadline())
        .expect("both genuine Source cells cover the entire root");
    assert_eq!(token.scope(), TokenScope::Instance);
}

#[test]
fn missing_duplicate_and_one_ulp_gap_never_complete_a_source_cover() {
    let leaf = |lo, hi| Leaf {
        lo: vec![lo],
        hi: vec![hi],
        lower: 1.0 / 16.0,
    };
    assert!(complete_cover_lower(
        &[leaf(-1.0, 0.0), leaf(-1.0, 0.0)],
        &[-1.0],
        &[1.0],
        deadline()
    )
    .is_none());
    assert!(complete_cover_lower(
        &[leaf(-1.0, 0.0), leaf(f64::from(f32::from_bits(1)), 1.0)],
        &[-1.0],
        &[1.0],
        deadline()
    )
    .is_none());
    assert_eq!(
        complete_cover_lower(
            &[leaf(-1.0, 0.0), leaf(0.0, 1.0)],
            &[-1.0],
            &[1.0],
            deadline()
        ),
        Some(1.0 / 16.0)
    );
    assert!(complete_cover_lower(&[leaf(-1.0, 1.0)], &[-1.0], &[1.0], expired()).is_none());
}

#[test]
fn wider_evaluation_source_reprices_slopes_instead_of_extending_old_bounds() {
    let s = Source::build(&unsafe_spec(1.0 / 16.0), &[-0.5], &[0.5], 0, 1, deadline()).unwrap();
    let v = vec![1.0, -0.5];
    let (_, f) = s.replay_retained(&v, deadline()).unwrap();
    let carry = InputHeadContinuation::build(
        &s,
        &[f],
        &[Recipe::Slopes(v)],
        &[-1.0],
        &[1.0],
        deadline(),
        deadline(),
    )
    .unwrap();
    assert!(!Arc::ptr_eq(&s.spec, &carry.source.spec));
    assert!(carry.root_box_matches(&[-1.0], &[1.0]));
    let bound = carry.try_support_box(&[-1.0], &[1.0], deadline()).unwrap();
    assert!(f64::from(bound) <= -15.0 / 16.0);
    assert!(
        carry
            .try_support_box(&[-1.0], &[-0.75], deadline())
            .unwrap()
            < 0.0
    );
    assert!(carry
        .try_support_box(&[-1.01], &[1.0], deadline())
        .is_none());
    assert!(carry
        .try_support_box(&[f32::NAN], &[1.0], deadline())
        .is_none());
}

#[test]
fn foreign_generation_and_missing_last_coordinates_refuse_completed_forms() {
    let s = unsafe_source();
    let (_, mut form) = s.replay_retained(&[1.0, -0.5], deadline()).unwrap();
    let foreign = unsafe_source();
    assert!(!Arc::ptr_eq(&s.spec, &foreign.spec));
    let x = vec![I { lo: -1.0, hi: 1.0 }];
    let r = foreign.relu_boxes_on(&x, deadline()).unwrap();
    assert!(form.support(&foreign, &x, &r, deadline()).is_none());
    assert!(InputHeadContinuation::build(
        &foreign,
        &[form.clone()],
        &[],
        &foreign.lo,
        &foreign.hi,
        deadline(),
        deadline()
    )
    .is_none());
    form.input.pop();
    assert!(!form.valid(&s));
    let (_, mut form) = s.replay_retained(&[1.0, -0.5], deadline()).unwrap();
    form.relu.pop();
    assert!(!form.valid(&s));
}

#[test]
fn every_last_input_and_negative_last_head_residual_stays_in_child_support() {
    let spec = InputHeadSpec::new(
        2,
        2,
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.0; 2],
        vec![1.0, 0.0, -1.0, 0.0],
        vec![1.0 / 16.0, 0.0],
        [93; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    let carry = continuation(&s, &[vec![1.0, -0.5]]);
    // Last input makes the original margin negative at (.5,1); omitting it
    // or its negative head would yield a false positive child.
    let lower = carry
        .try_support_box(&[0.5, 1.0], &[1.0, 1.0], deadline())
        .unwrap();
    assert!(f64::from(lower) <= -7.0 / 16.0);
    assert!(carry.ledger.try_lock().unwrap().leaves.is_empty());
    assert!(carry.try_finish(deadline()).is_none());
}

#[test]
fn wider_negative_group_recipe_replays_all_masks_and_keeps_joint_cancellation() {
    let spec = InputHeadSpec::new(
        2,
        2,
        vec![1.0, 1.0, 1.0, -1.0],
        vec![0.0; 2],
        vec![-1.0, 0.0, -1.0, 0.0],
        vec![33.0 / 16.0, 0.0],
        [94; 32],
    )
    .unwrap();
    let narrow = Source::build(&spec, &[-0.5; 2], &[0.5; 2], 0, 1, deadline()).unwrap();
    let wide = Source::build(&spec, &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    let coefficients = vec![(2, I::p(-1.0)), (3, I::p(-1.0))];
    let recipe = Recipe::Dual {
        rows: vec![RowRecipe {
            coefficients: coefficients.clone(),
            mass: None,
            group: Some(vec![(0, 1.0), (1, 1.0)]),
        }],
        points: vec![1.0],
    };
    let form = recipe.replay(&wide, deadline()).unwrap();
    let x = vec![I { lo: -1.0, hi: 1.0 }; 2];
    let r = wide.relu_boxes_on(&x, deadline()).unwrap();
    let lower = form.support(&wide, &x, &r, deadline()).unwrap();
    assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
    // The ordinary scalar replay already contracts the full affine input
    // residual: slopes(-1/2,-1/2), intercepts(-1,-1), residual(-1,0).
    // Its ideal wide-box lower is -3, rather than the naive independent
    // ReLU-box lower -4. Four complete masks give the stronger group -2.
    let scalar = wide.row_lhs_lower(&coefficients, deadline()).unwrap();
    let group = wide
        .negative_group_lower(&[(0, 1.0), (1, 1.0)], deadline())
        .unwrap();
    assert!(scalar > -3.01 && scalar <= -3.0);
    assert!(group > scalar + 0.9 && group <= -2.0);
    assert!(
        narrow
            .negative_group_lower(&[(0, 1.0), (1, 1.0)], deadline())
            .unwrap()
            > -1.01
    );
    assert!(wide
        .negative_group_lower(&[(0, 1.0), (1, 1.0)], expired())
        .is_none());
}

#[test]
fn triangle_replay_retains_every_owned_residual_before_box_support() {
    let s = unsafe_source();
    let model = triangle::Model::build(&s, deadline()).unwrap();
    let hint = HeadFunctionalLpHint {
        row_duals: vec![0.0; model.request().rows.len()],
    };
    let (lower, form, recipe) = model
        .replay_recorded(&s, &hint, 1.0, deadline(), true)
        .unwrap();
    let form = form.unwrap();
    assert_eq!(form.input.len(), 1);
    assert_eq!(form.relu.len(), 2);
    assert_eq!(form.relu[1].lo, s.q[1].lo);
    assert_eq!(form.relu[1].hi, s.q[1].hi);
    let x = vec![I { lo: -1.0, hi: 1.0 }];
    let r = s.relu_boxes_on(&x, deadline()).unwrap();
    assert_eq!(
        form.support(&s, &x, &r, deadline()).unwrap().to_bits(),
        lower.to_bits()
    );
    assert!(recipe.unwrap().replay(&s, expired()).is_none());
}

#[test]
fn independent_first_output_and_bias_uncertainties_prevent_false_child_closure() {
    let mut s = unsafe_source();
    s.output_error = 2.0;
    s.first_error[0] = 1.0;
    let carry = continuation(&s, &[vec![1.0, -0.5]]);
    assert!(carry.try_support_box(&[0.5], &[1.0], deadline()).unwrap() < 0.0);
    assert!(carry.try_finish(deadline()).is_none());
}

#[test]
fn amplified_subnormal_coefficient_debt_survives_before_f32_conversion() {
    let spec = InputHeadSpec::new(
        1,
        1,
        vec![0.0],
        vec![0.0],
        vec![0.0; 2],
        vec![0.0; 2],
        [95; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0e30], &[-1.0e30], 0, 1, deadline()).unwrap();
    let form = ResidualForm {
        generation: Arc::clone(&s.spec),
        constant: I::p(0.0),
        input: vec![I {
            lo: 0.0,
            hi: f64::from_bits(1),
        }],
        relu: vec![I::p(0.0)],
    };
    let x = vec![I {
        lo: -1.0e30,
        hi: -1.0e30,
    }];
    let r = s.relu_boxes_on(&x, deadline()).unwrap();
    let lower = form.support(&s, &x, &r, deadline()).unwrap();
    assert!(lower < -f64::from_bits(1));
    assert!(lower_f32(lower).unwrap() < 0.0);
    assert_eq!(lower_f32(f64::from_bits(1)), Some(0.0));
}

#[test]
fn completed_child_then_authority_expiry_cannot_publish_or_return_another_floor() {
    let s = unsafe_source();
    let carry = continuation(&s, &[vec![1.0, -0.5]]);
    assert!(carry.try_support_box(&[0.5], &[1.0], deadline()).unwrap() > 0.0);
    assert_eq!(carry.ledger.try_lock().unwrap().leaves.len(), 1);
    assert!(carry.try_support_box(&[0.5], &[1.0], expired()).is_none());
    assert!(!carry.has_complete_source_cover(expired()));
    assert!(carry.try_finish(expired()).is_none());
    assert!(!carry.is_live(expired()));
}

#[test]
fn ledger_contention_and_capacity_refuse_authority_without_importing_generic_leaves() {
    let s = unsafe_source();
    let carry = continuation(&s, &[vec![1.0, -0.5]]);
    {
        let _locked = carry.ledger.try_lock().unwrap();
        assert!(carry.try_support_box(&[0.5], &[1.0], deadline()).is_none());
        assert!(carry.try_finish(deadline()).is_none());
    }
    carry.ledger.try_lock().unwrap().overflow = true;
    assert!(!carry.has_complete_source_cover(deadline()));
    assert!(carry.try_finish(deadline()).is_none());
}

struct BrokenAdvice {
    calls: Arc<AtomicUsize>,
}
impl HeadPhaseAdvice for BrokenAdvice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }
    fn supports_functional_advice(&self) -> bool {
        true
    }
    fn propose_functional(
        &self,
        r: &HeadFunctionalLpRequest,
        d: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        assert!(r.validate() && live(d));
        self.calls.fetch_add(1, Ordering::Relaxed);
        Some(HeadFunctionalLpHint {
            row_duals: vec![0.0; r.rows.len() - 1],
        })
    }
}

#[test]
fn malformed_advice_keeps_native_child_continuation_without_a_third_call() {
    let spec = unsafe_spec(1.0 / 16.0);
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(BrokenAdvice {
        calls: Arc::clone(&calls),
    });
    let (outcome, carry) = verify_with_provider_retained(
        &spec,
        &[-1.0],
        &[1.0],
        0,
        1,
        policy(),
        deadline(),
        || Some(provider.clone()),
        Some((&[-1.0], &[1.0], deadline())),
    );
    let InputHeadOutcome::Unknown(report) = outcome else {
        panic!("unsafe whole source closed")
    };
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(report.advice_calls, 2);
    assert_eq!(report.replayed_candidates, 1);
    assert_eq!(report.triangle_replayed_candidates, 0);
    let carry = carry.expect("completed native source function survives malformed proposals");
    assert!(carry.try_support_box(&[0.5], &[1.0], deadline()).unwrap() > 0.0);
    assert!(carry.try_finish(deadline()).is_none());
}

#[test]
fn original_immediate_instance_proof_survives_invalid_evaluation_box() {
    let spec = InputHeadSpec::new(
        1,
        1,
        vec![1.0],
        vec![0.0],
        vec![1.0, 0.0],
        vec![1.0 / 16.0, 0.0],
        [96; 32],
    )
    .unwrap();
    let (outcome, carry) = verify_with_provider_retained(
        &spec,
        &[-1.0],
        &[1.0],
        0,
        1,
        policy(),
        deadline(),
        || None,
        Some((&[f64::NAN], &[1.0], deadline())),
    );
    let InputHeadOutcome::Unsat(report, token) = outcome else {
        panic!("old immediate proof was lost")
    };
    assert_eq!(report.advice_calls, 0);
    assert_eq!(token.scope(), TokenScope::Instance);
    assert!(carry.is_none());
}

#[test]
fn unsafe_tie_and_negative_sources_never_complete_the_instance_partition() {
    for bias in [0.0, -1.0 / 16.0] {
        let s = Source::build(&unsafe_spec(bias), &[-1.0], &[1.0], 0, 1, deadline()).unwrap();
        let carry = continuation(&s, &[vec![1.0, -0.5]]);
        assert!(carry.try_support_box(&[0.0], &[0.0], deadline()).unwrap() <= 0.0);
        assert!(carry.try_support_box(&[-1.0], &[0.0], deadline()).unwrap() < 0.0);
        assert!(carry.try_finish(deadline()).is_none());
    }
}

#[test]
fn consuming_worker_rechecks_fp_and_cannot_borrow_the_producer_threads_qualification() {
    let s = unsafe_source();
    let carry = continuation(&s, &[vec![1.0, -0.5]]);
    let worker = std::thread::spawn(move || {
        assert!(ny_core::require_f64_interval_proof_environment().is_ok());
        assert!(carry.is_live(deadline()));
        let bound = carry.try_support_box(&[0.5], &[1.0], deadline()).unwrap();
        assert!(bound > 0.0);
        assert!(carry.try_finish(deadline()).is_none());
    });
    worker.join().unwrap();
}

#[cfg(target_arch = "x86_64")]
#[test]
#[allow(unsafe_code, deprecated)]
fn foreign_worker_ftz_daz_refuses_all_source_arithmetic_and_publication() {
    let s = unsafe_source();
    let carry = continuation(&s, &[vec![1.0, -0.5]]);
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
        // Only this test worker's FP control state changes; RAII restores it
        // even if an assertion unwinds. The producer ran on the qualified
        // parent thread and cannot lend that qualification to this worker.
        let saved = unsafe { _mm_getcsr() };
        let _restore = Restore(saved);
        unsafe {
            _mm_setcsr(saved | (1 << 15) | (1 << 6));
        }
        assert!(ny_core::require_f64_interval_proof_environment().is_err());
        assert!(!carry.is_live(deadline()));
        assert!(carry.try_support_box(&[0.5], &[1.0], deadline()).is_none());
        assert!(!carry.has_complete_source_cover(deadline()));
        assert!(carry.try_finish(deadline()).is_none());
    })
    .join()
    .unwrap();
}

struct ZeroAdvice {
    calls: Arc<AtomicUsize>,
}
impl HeadPhaseAdvice for ZeroAdvice {
    fn propose(&self, _: &HeadPhaseLpRequest, _: Instant) -> Option<HeadPhaseLpHint> {
        None
    }
    fn supports_functional_advice(&self) -> bool {
        true
    }
    fn propose_functional(
        &self,
        r: &HeadFunctionalLpRequest,
        d: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        assert!(r.validate() && live(d));
        self.calls.fetch_add(1, Ordering::Relaxed);
        Some(HeadFunctionalLpHint {
            row_duals: vec![0.0; r.rows.len()],
        })
    }
}

#[test]
fn real_open_negative_group_producer_closes_only_after_complete_expanded_root_child_cover() {
    // The actual native root stays open with zero-dual advice; the owned
    // joint row exists but supplies no multiplier authority. Re-concretizing
    // EVERY head on64 genuine z-strips closes an entire evaluation root.
    for bias in [33.0 / 16.0, 2.0] {
        let spec = InputHeadSpec::new(
            2,
            2,
            vec![1.0, 1.0, 1.0, -1.0],
            vec![0.0; 2],
            vec![-1.0, 0.0, -1.0, 0.0],
            vec![bias, 0.0],
            [97; 32],
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(ZeroAdvice {
            calls: Arc::clone(&calls),
        });
        let outer_lo = (-1.0f32).next_down();
        let outer_hi = (1.0f32).next_up();
        let elo = [f64::from(outer_lo); 2];
        let ehi = [f64::from(outer_hi); 2];
        let (outcome, carry) = verify_with_provider_retained(
            &spec,
            &[-1.0; 2],
            &[1.0; 2],
            0,
            1,
            policy(),
            deadline(),
            || Some(provider.clone()),
            Some((&elo, &ehi, deadline())),
        );
        let InputHeadOutcome::Unknown(report) = outcome else {
            panic!("zero duals must leave the actual root open")
        };
        // The legacy all-negative export has no epigraph and is declined
        // before a call. The genuine triangle is the sole provider call.
        assert_eq!(report.advice_calls, 1);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(report.replayed_candidates, 3);
        assert_eq!(report.triangle_negative_groups, 1);
        assert_eq!(report.triangle_negative_group_masks, 4);
        assert!(report.lower.unwrap() < 0.0);
        let carry = carry.unwrap();
        assert!(carry.root_box_matches(&[outer_lo; 2], &[outer_hi; 2]));
        assert!(
            carry
                .try_support_box(&[outer_lo; 2], &[outer_hi; 2], deadline())
                .unwrap()
                < 0.0
        );
        for index in 0..64 {
            let zlo = if index == 0 {
                outer_lo
            } else {
                -1.0 + index as f32 / 32.0
            };
            let zhi = if index == 63 {
                outer_hi
            } else {
                -1.0 + (index + 1) as f32 / 32.0
            };
            let bound = carry
                .try_support_box(&[outer_lo, zlo], &[outer_hi, zhi], deadline())
                .unwrap();
            if bias > 2.0 {
                assert!(bound > 1.0 / 64.0 && bound <= 1.0 / 16.0);
            }
        }
        carry.mark_dependent_use();
        if bias > 2.0 {
            assert!(carry.has_complete_source_cover(deadline()));
            assert_eq!(
                carry.try_finish(deadline()).unwrap().scope(),
                TokenScope::Instance
            );
        } else {
            // Equality is unsafe at the ORIGINAL point x=1,z=0. A slightly
            // wider evaluation root also cannot be certified as safe.
            assert!(
                carry
                    .try_support_box(&[1.0, 0.0], &[1.0, 0.0], deadline())
                    .unwrap()
                    <= 0.0
            );
            assert!(!carry.has_complete_source_cover(deadline()));
            assert!(carry.try_finish(deadline()).is_none());
        }
    }
}

fn scalar_network(spec: &InputHeadSpec, unfused: bool) -> crate::Network {
    use crate::layers::{AddConstantLayer, LinearLayer, MaxPool2dLayer, ReLULayer, ReshapeLayer};
    use crate::Layer;
    use ndarray::{Array1, Array2, ArrayD, IxDyn};
    let mut network = crate::Network::new();
    let first = Array2::from_shape_fn((spec.heads, spec.inputs), |(j, k)| {
        spec.first[k * spec.heads + j]
    });
    let last = Array2::from_shape_fn((2, spec.heads), |(o, j)| spec.final_weight[j * 2 + o]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            first,
            (!unfused).then(|| Array1::from(spec.first_bias.clone())),
        )
        .unwrap(),
    ));
    if unfused {
        network.add_layer(Layer::AddConstant(AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[1, spec.heads]), spec.first_bias.clone()).unwrap(),
        )));
    }
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            last,
            (!unfused).then(|| Array1::from(spec.final_bias.clone())),
        )
        .unwrap(),
    ));
    if unfused {
        network.add_layer(Layer::AddConstant(AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), spec.final_bias.clone()).unwrap(),
        )));
    }
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            Array2::from_shape_vec((1, 2), vec![1.0, -1.0]).unwrap(),
            None,
        )
        .unwrap(),
    ));
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 1, 1])));
    network.add_layer(Layer::MaxPool2d(MaxPool2dLayer::new(
        (1, 1),
        (1, 1),
        (0, 0),
    )));
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1])));
    network
}

#[test]
fn actual_unfused_and_fused_scalar_sequences_match_but_wrong_sign_offset_and_tail_refuse() {
    use crate::layers::{LinearLayer, ReshapeLayer};
    use crate::Layer;
    use ndarray::{Array1, Array2};
    let s = unsafe_source();
    let carry = continuation(&s, &[vec![1.0, -0.5]]);
    for unfused in [true, false] {
        let n = scalar_network(&s.spec, unfused);
        assert!(carry.matches_scalar_network(&n));
        let tail = n.num_layers() - 4;
        let mut wrong = n.clone();
        wrong.layers_mut()[tail] = Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 2), vec![-1.0, 1.0]).unwrap(),
                None,
            )
            .unwrap(),
        );
        assert!(!carry.matches_scalar_network(&wrong));
        let mut offset = n.clone();
        offset.layers_mut()[tail] = Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 2), vec![1.0, -1.0]).unwrap(),
                Some(Array1::from(vec![1.0 / 16.0])),
            )
            .unwrap(),
        );
        assert!(!carry.matches_scalar_network(&offset));
        let mut shape = n;
        shape.layers_mut()[tail + 1] = Layer::Reshape(ReshapeLayer::new(vec![1, 1, 2]));
        assert!(!carry.matches_scalar_network(&shape));
    }
}

#[test]
fn authentic_original_network_and_unused_carry_do_not_create_source_dependency() {
    // M(x)=x+3 is safe on the entire root. Merely registering an opaque
    // continuation or authenticating the original MIP input is not a prune,
    // numerical consumption, MIP verdict or new Instance authorization.
    let spec = unsafe_spec(3.0);
    let source = Source::build(&spec, &[-1.0], &[1.0], 0, 1, deadline()).unwrap();
    let carry = continuation(&source, &[vec![1.0, -0.5]]);
    for unfused in [true, false] {
        let scalar = scalar_network(&spec, unfused);
        let mut original = crate::Network::new();
        for layer in &scalar.layers()[..scalar.num_layers() - 4] {
            original.add_layer(layer.clone());
        }
        assert!(carry.matches_original_network(&original));
        assert!(!carry.matches_scalar_network(&original));
        assert!(carry.matches_scalar_network(&scalar));
        assert!(!carry.matches_original_network(&scalar));
        let mut wrong = original.clone();
        let last = wrong.num_layers() - 1;
        wrong.layers_mut()[last] = crate::Layer::ReLU(crate::layers::ReLULayer);
        assert!(!carry.matches_original_network(&wrong));
    }
    assert!(!carry.dependent_use());
    assert_eq!(carry.supports.load(Ordering::Relaxed), 0);
    assert!(carry.ledger.lock().unwrap().leaves.is_empty());
    assert!(!carry.has_complete_source_cover(deadline()));
    assert!(carry.try_finish(deadline()).is_none());
}

#[test]
fn malformed_group_recipe_cannot_reuse_an_unrelated_joint_source_rhs() {
    let source = unsafe_source();
    let recipe = |coefficients: Vec<(usize, I)>, group: Vec<(usize, f64)>| Recipe::Dual {
        rows: vec![RowRecipe {
            coefficients,
            mass: None,
            group: Some(group),
        }],
        points: vec![1.0],
    };
    assert!(recipe(
        vec![(1, I::p(-1.0)), (2, I::p(-0.5))],
        vec![(0, 1.0), (1, 1.0)]
    )
    .replay(&source, deadline())
    .is_none());
    assert!(recipe(
        vec![(1, I::p(-1.0)), (2, I::p(-1.0))],
        vec![(1, 1.0), (0, 1.0)]
    )
    .replay(&source, deadline())
    .is_none());
    assert!(recipe(vec![(1, I::p(-1.0))], vec![(usize::MAX, 1.0)])
        .replay(&source, deadline())
        .is_none());
}

#[test]
fn original_authority_expiry_after_full_positive_cover_refuses_a_later_caller_grant() {
    // First complete a genuine positive whole-Source cover, while all clocks
    // are live. Changing the private original authority is a deterministic
    // clock fault for this test only; production exposes no such mutator.
    let spec = unsafe_spec(3.0); // M=x+3 >=2 on the entire root.
    let source = Source::build(&spec, &[-1.0], &[1.0], 0, 1, deadline()).unwrap();
    let mut carry = continuation(&source, &[vec![1.0, -0.5]]);
    assert!(carry.try_support_box(&[-1.0], &[1.0], deadline()).unwrap() > 1.0);
    assert!(carry.has_complete_source_cover(deadline()));
    assert_eq!(carry.ledger.try_lock().unwrap().leaves.len(), 1);
    let already_completed = carry.ledger.try_lock().unwrap().leaves[0].lower;
    assert!(already_completed > 1.0);
    Arc::get_mut(&mut carry)
        .expect("the test owns this invocation")
        .authority = expired();
    let later_caller_grant = deadline();
    assert!(!carry.is_live(later_caller_grant));
    assert!(carry
        .try_support_box(&[-1.0], &[1.0], later_caller_grant)
        .is_none());
    assert!(!carry.has_complete_source_cover(later_caller_grant));
    assert!(carry.try_finish(later_caller_grant).is_none());
    // Expiry refuses consumption/publication; it never fabricates or erases
    // the fact that a complete positive certificate had actually finished.
    assert_eq!(
        carry.ledger.try_lock().unwrap().leaves[0].lower,
        already_completed
    );
}
