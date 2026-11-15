// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_core::head_functional_advice::HeadFunctionalLpHint;
use ny_core::head_phase_advice::{HeadPhaseLpHint, HeadPhaseLpRequest};

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(3)
}
fn expired() -> Instant {
    Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
}
fn inputs(lo: &[f64], hi: &[f64]) -> Vec<I> {
    lo.iter().zip(hi).map(|(&lo, &hi)| I { lo, hi }).collect()
}
fn negative_twins(bias: f32) -> InputHeadSpec {
    InputHeadSpec::new(
        1,
        2,
        vec![1.0, -1.0],
        vec![0.0; 2],
        vec![-1.0, 0.0, -1.0, 0.0],
        vec![bias, 0.0],
        [121; 32],
    )
    .unwrap()
}
fn source(bias: f32) -> Source {
    Source::build(&negative_twins(bias), &[-1.0], &[1.0], 0, 1, deadline()).unwrap()
}
fn carry(s: &Source) -> Arc<InputHeadContinuation> {
    let slopes = s.slopes().unwrap();
    let (_, form) = s.replay_retained(&slopes, deadline()).unwrap();
    InputHeadContinuation::build(
        s,
        &[form],
        &[Recipe::Slopes(slopes)],
        &s.lo,
        &s.hi,
        deadline(),
        deadline(),
    )
    .unwrap()
}
fn old_floor(carry: &InputHeadContinuation, x: &[I]) -> f64 {
    let relu = ChildBox::new(&carry.source, x, deadline()).unwrap().relu();
    carry
        .forms
        .iter()
        .map(|f| f.support(&carry.source, x, &relu, deadline()).unwrap())
        .fold(f64::NEG_INFINITY, f64::max)
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
        assert_eq!(r.feature_count, 1);
        assert_eq!(r.bounds.len(), 3);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Some(HeadFunctionalLpHint {
            row_duals: vec![0.0; r.rows.len()],
        })
    }
}

#[test]
fn real_open_producer_retains_only_nonpositive_forms_but_child_chords_close_the_central_cell() {
    let spec = negative_twins(15.0 / 16.0);
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(ZeroAdvice {
        calls: Arc::clone(&calls),
    });
    let (outcome, continuation) = verify_with_provider_retained(
        &spec,
        &[-1.0],
        &[1.0],
        0,
        1,
        HeadGroupPolicy::new(1.0, 1, 1).unwrap(),
        deadline(),
        || Some(Arc::clone(&provider) as Arc<dyn HeadPhaseAdvice>),
        Some((&[-1.0], &[1.0], deadline())),
    );
    let InputHeadOutcome::Unknown(report) = outcome else {
        panic!("unsafe original root closed")
    };
    // Legacy all-negative export is honestly declined before any call.
    // The genuine triangle request receives zero multipliers, not authority.
    assert_eq!(report.advice_calls, 1);
    assert_eq!(report.triangle_advice_calls, 1);
    assert_eq!(report.replayed_candidates, 3);
    assert_eq!(report.triangle_negative_groups, 1);
    assert_eq!(report.triangle_negative_group_masks, 4);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(report.lower.unwrap() <= -1.0 / 16.0);
    let carry = continuation.unwrap();
    let central = inputs(&[-0.5], &[0.5]);
    let relu = ChildBox::new(&carry.source, &central, deadline())
        .unwrap()
        .relu();
    assert!(carry.forms.len() >= 4);
    for form in &carry.forms {
        assert!(
            form.support(&carry.source, &central, &relu, deadline())
                .unwrap()
                <= 0.0
        );
    }
    let root_before = old_floor(&carry, &inputs(&[-1.0], &[1.0]));
    assert_eq!(
        carry
            .support(&[-1.0], &[1.0], deadline())
            .unwrap()
            .0
            .to_bits(),
        root_before.to_bits()
    );
    let lower = carry.try_support_box(&[-0.5], &[0.5], deadline()).unwrap();
    assert!(lower > 3.0 / 8.0 && lower <= 7.0 / 16.0);
    assert!(carry.try_support_box(&[-1.0], &[-0.5], deadline()).unwrap() <= -1.0 / 16.0);
    assert!(carry.try_support_box(&[0.5], &[1.0], deadline()).unwrap() <= -1.0 / 16.0);
    carry.mark_dependent_use();
    assert!(!carry.has_complete_source_cover(deadline()));
    assert!(carry.try_finish(deadline()).is_none());
    // Per-child work does not ask the provider or reset its allowance.
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn unsafe_outer_siblings_and_equality_never_become_original_instance_authority() {
    for bias in [15.0 / 16.0, 0.5] {
        let c = carry(&source(bias));
        let central = c.try_support_box(&[-0.5], &[0.5], deadline()).unwrap();
        if bias == 0.5 {
            assert!(central <= 0.0);
        } else {
            assert!(central > 3.0 / 8.0);
        }
        assert!(c.try_support_box(&[-1.0], &[-0.5], deadline()).unwrap() < 0.0);
        assert!(c.try_support_box(&[0.5], &[1.0], deadline()).unwrap() < 0.0);
        c.mark_dependent_use();
        assert!(!c.has_complete_source_cover(deadline()));
        assert!(c.try_finish(deadline()).is_none());
    }
}

#[test]
fn complete_source_owned_child_cover_uses_the_real_singleton_instance_authorizer() {
    // Cover oracle only: root full native already could prove this safe
    // source. We deliberately retain baseline/zero slopes, not claim an AY
    // or whole-root-relaxation separation.
    let s = source(17.0 / 16.0);
    let (_, f) = s.replay_retained(&[0.0, 0.0], deadline()).unwrap();
    let c = InputHeadContinuation::build(
        &s,
        &[f],
        &[Recipe::Slopes(vec![0.0; 2])],
        &s.lo,
        &s.hi,
        deadline(),
        deadline(),
    )
    .unwrap();
    assert!(c.try_support_box(&[-1.0], &[1.0], deadline()).unwrap() < 0.0);
    assert!(c.try_support_box(&[-1.0], &[0.0], deadline()).unwrap() > 1.0 / 32.0);
    assert!(c.try_finish(deadline()).is_none());
    assert!(c.try_support_box(&[0.0], &[1.0], deadline()).unwrap() > 1.0 / 32.0);
    c.mark_dependent_use();
    let token = c.try_finish(deadline()).unwrap();
    assert_eq!(token.scope(), TokenScope::Instance);
    assert_eq!(token.classes(), &[1]);
    assert!(token.covers_all(&[1]));
    assert!(!token.covers_all(&[0, 1]));
}

#[test]
fn new_pass_refusal_and_completed_prefix_local_expiry_preserve_exact_old_floor_bits() {
    let c = carry(&source(15.0 / 16.0));
    let x = inputs(&[-0.5], &[0.5]);
    let before = old_floor(&c, &x);
    assert!(before < 0.0);
    REFUSE_NATIVE.with(|f| f.set(true));
    let refused = c.support(&[-0.5], &[0.5], deadline()).unwrap().0;
    assert_eq!(before.to_bits(), refused.to_bits());
    EXPIRE_AFTER_HEAD.with(|f| f.set(Some(1)));
    let partial = c.support(&[-0.5], &[0.5], deadline()).unwrap().0;
    assert_eq!(before.to_bits(), partial.to_bits());
    assert!(EXPIRE_AFTER_HEAD.with(|f| f.get()).is_none());
    assert!(c.is_live(deadline()));
    assert!(c.ledger.try_lock().unwrap().leaves.is_empty());
    assert!(c.try_support_box(&[-0.5], &[0.5], deadline()).unwrap() > 3.0 / 8.0);
}

#[test]
fn child_intercept_form_refuses_foreign_equal_source_generation_and_every_other_domain() {
    let s = source(15.0 / 16.0);
    let x = inputs(&[-0.5], &[0.5]);
    let child = ChildBox::new(&s, &x, deadline()).unwrap();
    let v = child.slopes(deadline()).unwrap();
    let f = child
        .replay(
            &s,
            &v,
            s.difference_tail_error(deadline()).unwrap(),
            deadline(),
        )
        .unwrap();
    assert!(f.support(&s, &x, deadline()).unwrap() > 3.0 / 8.0);
    let cloned = s.clone(); // same Arc is insufficient: exact owned borrow too
    assert!(Arc::ptr_eq(&s.spec, &cloned.spec));
    assert!(f.support(&cloned, &x, deadline()).is_none());
    let foreign = source(15.0 / 16.0);
    assert!(!Arc::ptr_eq(&s.spec, &foreign.spec));
    assert!(child.replay(&foreign, &v, 0.0, deadline()).is_none());
    for other in [
        inputs(&[-1.0], &[1.0]),
        inputs(&[-0.25], &[0.25]),
        inputs(&[0.5], &[1.0]),
    ] {
        assert!(f.support(&s, &other, deadline()).is_none());
    }
}

#[test]
fn root_error_balls_full_q_and_output_bias_debt_survive_every_child_replay() {
    let mut s = source(15.0 / 16.0);
    // Independent source uncertainty is widened in this private adversarial
    // oracle, never shrunk or canceled based on duplicate nominal heads.
    s.first_error[0] += 0.5;
    s.first_error[1] += 0.5;
    s.q[0] = I {
        lo: -1.25,
        hi: -0.75,
    };
    s.q[1] = I { lo: -1.5, hi: 0.25 };
    s.bias = I {
        lo: 0.75,
        hi: 15.0 / 16.0,
    };
    s.output_error += 0.125;
    let child = ChildBox::new(&s, &inputs(&[-0.5], &[0.5]), deadline()).unwrap();
    let lower = child.native_lower(&s, 0.125, deadline()).unwrap();
    // At x=.5 choose both independent +.5 errors, Q lower endpoints,
    // B lower and full output/tail debt. That admitted value is negative.
    let admitted = 0.75 - 1.25 * 1.0 - 1.5 * 0.0 - s.output_error - 0.125;
    assert!(lower <= admitted && lower < 0.0);
}

#[test]
fn last_original_input_and_negative_last_head_cannot_disappear_in_child_slopes() {
    let spec = InputHeadSpec::new(
        2,
        2,
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.0; 2],
        vec![1.0, 0.0, -1.0, 0.0],
        vec![1.0 / 16.0, 0.0],
        [122; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    let x = inputs(&[0.5, 1.0], &[1.0, 1.0]);
    let child = ChildBox::new(&s, &x, deadline()).unwrap();
    assert_eq!(child.y.len(), 2);
    let lower = child
        .native_lower(&s, s.difference_tail_error(deadline()).unwrap(), deadline())
        .unwrap();
    assert!(lower <= -7.0 / 16.0);
    let c = carry(&s);
    assert!(
        c.try_support_box(&[0.5, 1.0], &[1.0, 1.0], deadline())
            .unwrap()
            <= -7.0 / 16.0
    );
    assert!(c.ledger.try_lock().unwrap().leaves.is_empty());
}

#[test]
fn root_scalar_tail_error_is_stored_and_charged_once_in_a_fresh_native_form() {
    let c = carry(&source(15.0 / 16.0));
    assert!(c.tail_error > 0.0);
    assert_eq!(
        c.tail_error.to_bits(),
        c.source
            .difference_tail_error(deadline())
            .unwrap()
            .to_bits()
    );
    let child = ChildBox::new(&c.source, &inputs(&[-0.5], &[0.5]), deadline()).unwrap();
    let without = child.native_lower(&c.source, 0.0, deadline()).unwrap();
    let charged = child
        .native_lower(&c.source, c.tail_error, deadline())
        .unwrap();
    assert!(charged < without);
    let twice = child
        .native_lower(&c.source, 2.0 * c.tail_error, deadline())
        .unwrap();
    assert!(charged > twice);
    // A dyadic tail envelope separates one charge from accidental doubling
    // without assuming reassociation of directed interval operations.
    let one_eighth = child.native_lower(&c.source, 0.125, deadline()).unwrap();
    assert!(one_eighth > 9.0 / 32.0 && one_eighth <= 5.0 / 16.0);
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(child.native_lower(&c.source, invalid, deadline()).is_none());
    }
}

#[test]
fn widened_f32_evaluation_root_rebuilds_errors_before_child_native_support() {
    let s = source(15.0 / 16.0);
    let v = s.slopes().unwrap();
    let (_, f) = s.replay_retained(&v, deadline()).unwrap();
    let lo = f64::from((-1.0f32).next_down());
    let hi = f64::from((1.0f32).next_up());
    let c = InputHeadContinuation::build(
        &s,
        &[f],
        &[Recipe::Slopes(v)],
        &[lo],
        &[hi],
        deadline(),
        deadline(),
    )
    .unwrap();
    assert!(!Arc::ptr_eq(&s.spec, &c.source.spec));
    assert_eq!(c.source.lo.as_slice(), &[lo]);
    assert!(c.source.first_error[0] >= s.first_error[0]);
    assert!(c.try_support_box(&[-0.5], &[0.5], deadline()).unwrap() > 3.0 / 8.0);
    assert!(c.try_support_box(&[1.0], &[hi as f32], deadline()).unwrap() < 0.0);
    assert!(c.try_finish(deadline()).is_none());
}

#[test]
fn complete_child_shape_box_and_source_errors_are_required_before_any_new_pass() {
    let s = source(15.0 / 16.0);
    for x in [
        vec![],
        inputs(&[-1.01], &[1.0]),
        vec![I {
            lo: f64::NAN,
            hi: 0.5,
        }],
        vec![I { lo: 0.5, hi: -0.5 }],
        vec![I {
            lo: 0.0,
            hi: f64::INFINITY,
        }],
    ] {
        assert!(ChildBox::new(&s, &x, deadline()).is_none());
    }
    let x = inputs(&[-0.5], &[0.5]);
    for fault in 0..5 {
        let mut malformed = s.clone();
        match fault {
            0 => {
                malformed.first_error.pop();
            }
            1 => {
                malformed.q.pop();
            }
            2 => {
                malformed.y.pop();
            }
            3 => malformed.first_error[0] = f64::NAN,
            _ => malformed.output_error = -1.0,
        }
        assert!(ChildBox::new(&malformed, &x, deadline()).is_none());
    }
    let child = ChildBox::new(&s, &x, deadline()).unwrap();
    for v in [vec![0.0], vec![f64::NAN, 0.0], vec![0.0, f64::INFINITY]] {
        assert!(child.replay(&s, &v, 0.0, deadline()).is_none());
    }
}

#[test]
fn consuming_deadline_and_original_authority_expiry_refuse_child_and_partition_authority() {
    let c = carry(&source(17.0 / 16.0));
    assert!(c.support(&[-0.5], &[0.5], expired()).is_none());
    assert!(c.try_support_box(&[-0.5], &[0.5], expired()).is_none());
    let child = ChildBox::new(&c.source, &inputs(&[-0.5], &[0.5]), deadline()).unwrap();
    assert!(child
        .native_lower(&c.source, c.tail_error, expired())
        .is_none());
    let mut c = Arc::try_unwrap(c).unwrap();
    c.authority = expired();
    assert!(c.try_support_box(&[-0.5], &[0.5], deadline()).is_none());
    assert!(c.try_finish(deadline()).is_none());
    assert!(!c.has_complete_source_cover(deadline()));
}

#[test]
fn directed_subnormal_product_is_repaired_before_large_child_input_amplifies_it() {
    let eta = f64::from_bits(1);
    let spec = InputHeadSpec::new(
        1,
        1,
        vec![f32::from_bits(1)],
        vec![0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        [123; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0e30], &[1.0e30], 0, 1, deadline()).unwrap();
    let x = inputs(&[-1.0e30], &[0.0]);
    let child = ChildBox::new(&s, &x, deadline()).unwrap();
    let form = child
        .replay(
            &s,
            &[eta],
            s.difference_tail_error(deadline()).unwrap(),
            deadline(),
        )
        .unwrap();
    assert!(form.input[0].lo < 0.0 && form.input[0].hi > 0.0);
    let amplified = form.input[0].mul(x[0]).unwrap();
    assert!(amplified.lo < -eta);
    assert!(form.support(&s, &x, deadline()).unwrap() <= 0.0);
    assert!(s.first_error[0] > 0.0 && s.output_error > 0.0);
}

#[test]
fn every_child_native_slope_and_arbitrary_point_replay_is_below_complete_dyadic_network_values() {
    let spec = InputHeadSpec::new(
        2,
        3,
        vec![1.0, 0.0, 0.5, 0.0, 1.0, -0.5],
        vec![0.25, -0.25, 0.0],
        vec![0.5, 0.0, -1.0, 0.0, 0.25, 0.0],
        vec![1.0 / 16.0, 0.0],
        [124; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    let box_ = inputs(&[-0.5, -0.25], &[0.5, 0.75]);
    let child = ChildBox::new(&s, &box_, deadline()).unwrap();
    let native = child.slopes(deadline()).unwrap();
    for v in [
        native,
        vec![0.0; 3],
        vec![1.5, -2.0, 0.125],
        vec![-1.0, 0.75, 2.0],
    ] {
        let form = child
            .replay(
                &s,
                &v,
                s.difference_tail_error(deadline()).unwrap(),
                deadline(),
            )
            .unwrap();
        let lower = form.support(&s, &box_, deadline()).unwrap();
        for ix in 0..=4 {
            for iz in 0..=4 {
                let x = -0.5 + f64::from(ix) / 4.0;
                let z = -0.25 + f64::from(iz) / 4.0;
                let value = 1.0 / 16.0 + 0.5 * (x + 0.25).max(0.0) - (z - 0.25).max(0.0)
                    + 0.25 * (0.5 * x - 0.5 * z).max(0.0);
                assert!(lower <= value);
            }
        }
    }
}

#[test]
fn genuine_thirty_input_128_head_child_retains_all_errors_and_final_feature() {
    let mut w = vec![0.0; 30 * 128];
    w[0] = 1.0;
    w[29 * 128 + 127] = -1.0;
    let mut q = vec![0.0; 256];
    q[0] = -1.0;
    q[254] = -1.0;
    let spec = InputHeadSpec::new(
        30,
        128,
        w,
        vec![0.0; 128],
        q,
        vec![15.0 / 16.0, 0.0],
        [125; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0; 30], &[1.0; 30], 0, 1, deadline()).unwrap();
    let mut lo = vec![-0.5; 30];
    let mut hi = vec![0.5; 30];
    lo[0] = 0.5;
    hi[0] = 0.5;
    lo[29] = -1.0;
    hi[29] = -1.0;
    let child = ChildBox::new(&s, &inputs(&lo, &hi), deadline()).unwrap();
    assert_eq!(child.y.len(), 128);
    assert_eq!(child.slopes(deadline()).unwrap().len(), 128);
    let lower = child
        .native_lower(&s, s.difference_tail_error(deadline()).unwrap(), deadline())
        .unwrap();
    assert!(lower <= -9.0 / 16.0);
}

#[cfg(target_arch = "x86_64")]
#[test]
#[allow(unsafe_code, deprecated)]
fn actual_consuming_worker_ftz_daz_refuses_new_child_support_and_source_authority() {
    let c = carry(&source(15.0 / 16.0));
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
        let old = unsafe { _mm_getcsr() };
        let _restore = Restore(old);
        unsafe {
            _mm_setcsr(old | (1 << 15) | (1 << 6));
        }
        assert!(!fp_qualified());
        assert!(ChildBox::new(&c.source, &inputs(&[-0.5], &[0.5]), deadline()).is_none());
        assert!(c.try_support_box(&[-0.5], &[0.5], deadline()).is_none());
        assert!(c.try_finish(deadline()).is_none());
    })
    .join()
    .unwrap();
}
