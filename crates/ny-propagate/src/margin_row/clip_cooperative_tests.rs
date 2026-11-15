// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Arithmetic adapters do not authenticate the constructed halfspaces below.
//! Rational controls certify their stated numerical premises independently;
//! caller-lease controls price this helper's local overlap, not an entire
//! original network. Actual original-source consumers own the wider census.

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{
    LeaseClass, WorkBudget, WorkLimits, WorkRefusal, WorkRequest, WorkReserve,
};
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};
use std::cell::{Cell, RefCell};
use std::time::{Duration, Instant};

fn q(value: f64) -> Q {
    Q::from_float(value).unwrap()
}
fn check() -> Result<()> {
    Ok(())
}
fn dot(a: &[f64], x: &[f64]) -> Q {
    a.iter()
        .zip(x)
        .fold(Q::zero(), |sum, (&a, &x)| sum + q(a) * q(x))
}
fn zero_constraint(n: usize) -> Halfspace {
    Halfspace {
        a: vec![0.0; n],
        d: 0.0,
    }
}
fn selected_problem(n: usize) -> (Vec<f64>, Halfspace, Vec<f64>, Vec<f64>) {
    (
        (1..=n).rev().map(|j| j as f64 / 8.0).collect(),
        Halfspace {
            a: vec![-1.0; n],
            d: -(n as f64) / 8.0,
        },
        vec![0.0; n],
        vec![1.0; n],
    )
}

fn resource(
    scratch: usize,
    work: u64,
    deadline: Option<Instant>,
) -> (WorkBudget, CancellationToken) {
    let memory = MemoryAuthority::new(scratch.max(1));
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline,
            scratch_bytes: scratch,
            persistent_bytes: 0,
            max_work: work,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, cancel)
}
fn request(quote: DualArithmeticRequest) -> WorkRequest {
    WorkRequest {
        scratch_bytes: quote.scratch_bytes,
        work: quote.work,
        ..WorkRequest::default()
    }
}
fn request_with_checkpoint(quote: DualArithmeticRequest) -> WorkRequest {
    // These actual ledger callbacks have a fixed <=64-unit clock/cancel
    // check. Every successful helper callback also has a paid 64-unit live
    // environment probe, so calls <= quote.work/64 + 1. Keep the callback's
    // separate allowance alongside the helper quote on the SAME ledger.
    WorkRequest {
        work: quote.work.checked_mul(2).unwrap().checked_add(64).unwrap(),
        ..request(quote)
    }
}
fn refusal(error: WorkRefusal) -> NyError {
    match error {
        WorkRefusal::Deadline => NyError::DeadlineExceeded("dual original clock".into()),
        _ => NyError::UnsupportedConfiguration(format!("dual original ledger: {error}")),
    }
}

#[test]
fn clip_cooperative_distinct_breakpoints_preserve_legacy_nominal_and_directed_bits() {
    let _env = crate::tests::lock_env_shared();
    for n in [1, 3, 19, 129] {
        let (obj, h, mid, rad) = selected_problem(n);
        let old = super::super::dual_lower_outward(&obj, &[h.clone()], &mid, &rad).unwrap();
        let new = dual_lower_outward_with_checkpoint(&obj, &[h], &mid, &rad, &check)
            .unwrap()
            .unwrap();
        assert_eq!(new.nominal.to_bits(), old.nominal.to_bits());
        assert_eq!(new.lower.to_bits(), old.lower.to_bits());
        assert_eq!(new.magnitude.to_bits(), old.magnitude.to_bits());
    }
}

#[test]
fn clip_cooperative_equal_breakpoints_keep_original_coordinate_order_and_rational_dual() {
    let _env = crate::tests::lock_env_shared();
    let mut points = [
        (2.0, -1.0),
        (1.0, -10.0),
        (1.0, -20.0),
        (2.0, -2.0),
        (1.0, -30.0),
    ];
    let mut tmp = [(0.0, 0.0); 5];
    let mut poll = DualPoll::new(&check).unwrap();
    sort_breakpoints(&mut points, &mut tmp, &mut poll).unwrap();
    assert_eq!(
        points,
        [
            (1.0, -10.0),
            (1.0, -20.0),
            (1.0, -30.0),
            (2.0, -1.0),
            (2.0, -2.0)
        ]
    );

    let obj = [0.5, 1.0, 2.0];
    let h = Halfspace {
        a: vec![-0.5, -1.0, -2.0],
        d: 0.0,
    };
    let mid = [0.0; 3];
    let rad = [1.0, 0.5, 0.25];
    let result = dual_lower_outward_with_checkpoint(&obj, &[h], &mid, &rad, &check)
        .unwrap()
        .unwrap();
    // All three breakpoints are exactly 1. At beta=1 the exact Lagrangian
    // coefficients and bias are zero, independently of any tied slope order.
    assert_eq!(result.nominal, 0.0);
    assert!(q(result.lower) <= Q::zero());
    assert!(result.lower > -1e-12);
    for point in [[0.0; 3], [1.0, 0.5, 0.25], [0.0, 0.5, 0.25]] {
        assert!(dot(&obj, &point) >= Q::zero());
        assert!(q(result.lower) <= dot(&obj, &point));
    }
}

#[test]
fn clip_cooperative_fractional_folds_enclose_exact_selected_rational_lagrangian() {
    let _env = crate::tests::lock_env_shared();
    let obj = [0.1, -0.3, 0.7];
    let mid = [0.0, 0.1, -0.2];
    let rad = [1.0, 0.9, 0.8];
    let hs = [
        Halfspace {
            a: vec![-0.3, 0.0, 0.0],
            d: 0.0375,
        },
        Halfspace {
            a: vec![0.0, 0.7, 0.0],
            d: -0.175,
        },
    ];
    let mut nominal = obj.to_vec();
    let mut coefficients: Vec<Q> = obj.iter().map(|&v| q(v)).collect();
    let mut magnitudes: Vec<Q> = obj.iter().map(|&v| q(v).abs()).collect();
    let (mut bias, mut magnitude) = (Q::zero(), Q::zero());
    let mut points = [(0.0, 0.0); 3];
    let mut tmp = [(0.0, 0.0); 3];
    let mut poll = DualPoll::new(&check).unwrap();
    let mut positive = 0;
    for h in &hs {
        let beta = argmax(&nominal, h, &mid, &rad, &mut points, &mut tmp, &mut poll).unwrap();
        assert!(beta.is_finite() && beta >= 0.0);
        if beta > 0.0 {
            positive += 1;
            for j in 0..obj.len() {
                nominal[j] = beta.mul_add(h.a[j], nominal[j]);
                coefficients[j] += q(beta) * q(h.a[j]);
                magnitudes[j] += q(beta) * q(h.a[j]).abs();
            }
            bias += q(beta) * q(h.d);
            magnitude += q(beta) * q(h.d).abs();
        }
    }
    assert_eq!(positive, 2);
    let mut exact = bias;
    for j in 0..obj.len() {
        let lo = q(mid[j]) - q(rad[j]);
        let hi = q(mid[j]) + q(rad[j]);
        exact += (&coefficients[j] * lo).min(&coefficients[j] * hi);
        magnitude += &magnitudes[j] * (q(mid[j]).abs() + q(rad[j]));
    }
    let result = dual_lower_outward_with_checkpoint(&obj, &hs, &mid, &rad, &check)
        .unwrap()
        .unwrap();
    assert!(q(result.lower) <= exact);
    assert!(q(result.magnitude) >= magnitude);
    assert!(result.lower > -1.0);
}

#[test]
fn clip_cooperative_original_halfspace_premises_bound_independent_feasible_grid() {
    let _env = crate::tests::lock_env_shared();
    let obj = [3.0, -5.0, 2.0];
    let hs = [
        Halfspace {
            a: vec![-1.0, 0.0, 0.0],
            d: 0.0,
        },
        Halfspace {
            a: vec![0.0, 1.0, 0.0],
            d: -0.25,
        },
    ];
    let result = dual_lower_outward_with_checkpoint(&obj, &hs, &[0.0; 3], &[1.0; 3], &check)
        .unwrap()
        .unwrap();
    assert!(q(result.lower) <= q(-3.25));
    assert!(result.lower > -3.250_000_000_001);
    let mut checked = 0;
    for a in 0..=4 {
        for b in -4..=1 {
            for c in -4..=4 {
                let x = [f64::from(a) / 4.0, f64::from(b) / 4.0, f64::from(c) / 4.0];
                assert!(hs.iter().all(|h| dot(&h.a, &x) + q(h.d) <= Q::zero()));
                assert!(q(result.lower) <= dot(&obj, &x));
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 270);
}

#[test]
fn clip_cooperative_large_cancellation_retains_negative_exact_unit() {
    let _env = crate::tests::lock_env_shared();
    let obj = [1e16, -1.0, -1e16];
    let x = [1.0; 3];
    let result =
        dual_lower_outward_with_checkpoint(&obj, &[zero_constraint(3)], &x, &[0.0; 3], &check)
            .unwrap()
            .unwrap();
    assert_eq!(result.nominal, 0.0);
    assert_eq!(dot(&obj, &x), q(-1.0));
    assert!(q(result.lower) <= q(-1.0));
}

#[test]
fn clip_cooperative_negative_half_min_subnormal_never_becomes_zero_lower_bound() {
    let _env = crate::tests::lock_env_shared();
    let tiny = f64::from_bits(1);
    let result = dual_lower_outward_with_checkpoint(&[-tiny], &[], &[0.5], &[0.0], &check)
        .unwrap()
        .unwrap();
    let exact = -q(tiny) * q(0.5);
    assert!(exact < Q::zero());
    assert_eq!(result.nominal, 0.0);
    assert!(result.lower < 0.0 && q(result.lower) <= exact);
    assert!(result.magnitude > 0.0 && q(result.magnitude) >= -exact);
}

#[test]
fn clip_cooperative_signed_endpoint_halfway_rounding_is_independently_enclosed() {
    let _env = crate::tests::lock_env_shared();
    let obj = [0.1, -0.3];
    let mid = [1.0, -1.0];
    let rad = [f64::EPSILON / 4.0, f64::EPSILON / 2.0];
    let exact = (q(mid[0]) - q(rad[0])) * q(obj[0]) + (q(mid[1]) + q(rad[1])) * q(obj[1]);
    let result = dual_lower_outward_with_checkpoint(&obj, &[], &mid, &rad, &check)
        .unwrap()
        .unwrap();
    assert!(q(result.lower) <= exact);
    assert!(result.lower > 0.399_999_999_999);
}

#[test]
fn clip_cooperative_exact_dyadic_constraint_keeps_useful_zero_and_prior_bound() {
    let _env = crate::tests::lock_env_shared();
    let h = Halfspace {
        a: vec![-1.0],
        d: 0.0,
    };
    let result = dual_lower_outward_with_checkpoint(&[1.0], &[h], &[0.0], &[1.0], &check)
        .unwrap()
        .unwrap();
    assert_eq!(result.nominal, 0.0);
    assert_eq!(result.lower, 0.0);
    // The consuming owner combines independently certified candidates. A
    // weaker (or absent) new proposal cannot replace its prior valid bound.
    let previous = -0.25_f64;
    assert_eq!(previous.max(result.lower), 0.0);
    let weak = dual_lower_outward_with_checkpoint(&[1.0], &[], &[0.0], &[1.0], &check)
        .unwrap()
        .unwrap();
    assert_eq!(previous.max(weak.lower), previous);
}

#[test]
fn clip_cooperative_signed_zero_and_empty_frames_preserve_valid_zero() {
    let _env = crate::tests::lock_env_shared();
    for (obj, mid, rad) in [
        (&[][..], &[][..], &[][..]),
        (&[-0.0][..], &[0.0][..], &[-0.0][..]),
    ] {
        let result = dual_lower_outward_with_checkpoint(obj, &[], mid, rad, &check)
            .unwrap()
            .unwrap();
        assert_eq!(q(result.lower), Q::zero());
        assert_eq!(q(result.magnitude), Q::zero());
    }
}

#[test]
fn clip_cooperative_invalid_shapes_nonfinite_and_negative_radii_refuse_atomically() {
    let _env = crate::tests::lock_env_shared();
    let obj = [1.0, -2.0];
    let before = obj.map(f64::to_bits);
    let h = Halfspace {
        a: vec![1.0],
        d: 0.0,
    };
    assert!(
        dual_lower_outward_with_checkpoint(&obj, &[h], &[0.0; 2], &[1.0; 2], &check)
            .unwrap()
            .is_none()
    );
    for (o, m, r) in [
        (&obj[..], &[0.0][..], &[1.0; 2][..]),
        (&[f64::NAN][..], &[0.0][..], &[1.0][..]),
        (&[1.0][..], &[f64::INFINITY][..], &[1.0][..]),
        (&[1.0][..], &[0.0][..], &[-1.0][..]),
    ] {
        assert!(dual_lower_outward_with_checkpoint(o, &[], m, r, &check)
            .unwrap()
            .is_none());
    }
    assert_eq!(obj.map(f64::to_bits), before);
}

#[test]
fn clip_cooperative_unrepresentable_intermediates_refuse_without_authority() {
    let _env = crate::tests::lock_env_shared();
    assert!(
        dual_lower_outward_with_checkpoint(&[f64::MAX], &[], &[2.0], &[0.0], &check)
            .unwrap()
            .is_none()
    );
    assert!(
        dual_lower_outward_with_checkpoint(&[1.0], &[], &[f64::MAX], &[f64::MAX], &check)
            .unwrap()
            .is_none()
    );
    assert!(dual_lower_outward_request(usize::MAX, 1).is_err());
    assert!(dual_lower_outward_request(1, usize::MAX).is_err());
}

#[test]
fn clip_cooperative_quote_covers_actual_blocks_sort_and_complete_overlap() {
    let _env = crate::tests::lock_env_shared();
    for n in [0, 1, 2, 3, 19, 129, 513] {
        for count in [0, 1, 3] {
            let (obj, h, mid, rad) = selected_problem(n);
            let hs = vec![h; count];
            let quote = dual_lower_outward_request(n, count).unwrap();
            assert_eq!(quote.scratch_bytes, META_BYTES + n * (4 * 8 + 2 * 16));
            let mut poll = DualPoll::new(&check).unwrap();
            let result = calculate(&obj, &hs, &mid, &rad, &mut poll).unwrap();
            poll.checkpoint().unwrap();
            assert!(result.is_some());
            assert!(
                poll.total as u64 <= quote.work,
                "n={n}, k={count}, actual={}, quote={}",
                poll.total,
                quote.work
            );
            assert!(poll.max_gap <= POLL_UNITS);
        }
    }
}

#[test]
fn clip_cooperative_scratch_and_work_refuse_before_any_helper_allocation() {
    let quote = dual_lower_outward_request(129, 3).unwrap();
    for (cap, work, expected) in [
        (
            quote.scratch_bytes - 1,
            quote.work,
            WorkRefusal::ScratchLimit,
        ),
        (quote.scratch_bytes, quote.work - 1, WorkRefusal::WorkLimit),
    ] {
        let (budget, _) = resource(cap, work, None);
        let result = budget.lease_for(LeaseClass::Fallback, request(quote));
        assert_eq!(result.err(), Some(expected));
        let receipt = budget.receipt().unwrap();
        assert_eq!(receipt.admitted, 0);
        assert_eq!(receipt.live_scratch_bytes, 0);
        assert_eq!(receipt.work_reserved, 0);
    }
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn clip_cooperative_real_shared_cancellation_stops_inner_work_and_keeps_spent_allowance() {
    let _env = crate::tests::lock_env_shared();
    let (obj, h, mid, rad) = selected_problem(513);
    let hs = [h];
    let quote = dual_lower_outward_request(obj.len(), hs.len()).unwrap();
    let paid = request_with_checkpoint(quote);
    let (budget, cancel) = resource(quote.scratch_bytes, paid.work, None);
    let lease = budget.lease_for(LeaseClass::Fallback, paid).unwrap();
    let calls = Cell::new(0);
    let callback = || {
        calls.set(calls.get() + 1);
        if calls.get() == 500 {
            cancel.cancel();
        }
        lease.checkpoint().map_err(refusal)
    };
    let original: Vec<u64> = obj.iter().map(|v| v.to_bits()).collect();
    assert!(dual_lower_outward_with_checkpoint(&obj, &hs, &mid, &rad, &callback).is_err());
    assert_eq!(calls.get(), 500);
    assert_eq!(
        obj.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        original
    );
    let held = budget.receipt().unwrap();
    assert_eq!(held.live_scratch_bytes, quote.scratch_bytes);
    assert_eq!(held.completed, 0);
    drop(lease);
    let released = budget.receipt().unwrap();
    assert_eq!(released.live_scratch_bytes, 0);
    assert_eq!(released.work_reserved, paid.work);
    assert!(budget
        .clone()
        .lease_for(LeaseClass::Fallback, paid)
        .is_err());
}

#[test]
fn clip_cooperative_sort_is_cooperative_inside_large_equal_key_runs() {
    let _env = crate::tests::lock_env_shared();
    let mut points: Vec<(f64, f64)> = (0..4097).map(|j| ((j % 3) as f64, j as f64)).collect();
    let original = points.clone();
    let mut tmp = vec![(0.0, 0.0); points.len()];
    let calls = Cell::new(0);
    let callback = || {
        calls.set(calls.get() + 1);
        if calls.get() == 10 {
            return Err(NyError::DeadlineExceeded("sort inherited clock".into()));
        }
        Ok(())
    };
    let mut poll = DualPoll::new(&callback).unwrap();
    assert!(sort_breakpoints(&mut points, &mut tmp, &mut poll).is_err());
    assert_eq!(calls.get(), 10);
    assert!(poll.max_gap <= POLL_UNITS);
    // Cancellation occurs within the first merge pass, before copying it
    // back. Temporary changes cannot publish a completed candidate.
    assert_eq!(points, original);
}

#[test]
fn clip_cooperative_successful_local_lease_releases_buffers_without_refunding_work() {
    let _env = crate::tests::lock_env_shared();
    let (obj, h, mid, rad) = selected_problem(19);
    let quote = dual_lower_outward_request(obj.len(), 1).unwrap();
    let paid = request_with_checkpoint(quote);
    let (budget, _) = resource(quote.scratch_bytes, paid.work, None);
    let mut lease = budget.lease_for(LeaseClass::Fallback, paid).unwrap();
    let result = dual_lower_outward_with_checkpoint(&obj, &[h], &mid, &rad, &|| {
        lease.checkpoint().map_err(refusal)
    })
    .unwrap()
    .unwrap();
    assert!(result.lower.is_finite());
    lease.complete().unwrap();
    drop(lease);
    let receipt = budget.receipt().unwrap();
    assert_eq!(receipt.completed, 1);
    assert_eq!(receipt.live_scratch_bytes, 0);
    assert_eq!(receipt.work_reserved, paid.work);
    assert_eq!(
        budget.lease_for(LeaseClass::Fallback, paid).err(),
        Some(WorkRefusal::WorkLimit)
    );
}

#[test]
fn clip_cooperative_expired_original_clock_refuses_before_admission() {
    let quote = dual_lower_outward_request(19, 1).unwrap();
    let (budget, _) = resource(
        quote.scratch_bytes,
        quote.work,
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
    );
    assert_eq!(
        budget.lease_for(LeaseClass::Fallback, request(quote)).err(),
        Some(WorkRefusal::Deadline)
    );
    assert_eq!(budget.receipt().unwrap().admitted, 0);
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
struct UnqualifiedEnvironment {
    #[cfg(target_arch = "aarch64")]
    saved: u64,
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl UnqualifiedEnvironment {
    #[allow(unsafe_code)]
    fn enter(flush: bool) -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let saved: u64;
            std::arch::asm!("mrs {saved}, fpcr", saved=out(reg) saved, options(nomem, nostack));
            let altered = if flush {
                saved | (1_u64 << 24)
            } else {
                (saved & !(3_u64 << 22)) | (1_u64 << 22)
            };
            std::arch::asm!("msr fpcr, {altered}", "isb", altered=in(reg) altered, options(nomem, nostack));
            Self { saved }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let saved = std::arch::x86_64::_mm_getcsr();
            let altered = if flush {
                saved | (1 << 15) | (1 << 6)
            } else {
                (saved & !(3 << 13)) | (1 << 13)
            };
            std::arch::x86_64::_mm_setcsr(altered);
            Self { saved }
        }
    }
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl Drop for UnqualifiedEnvironment {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("msr fpcr, {saved}", "isb", saved=in(reg) self.saved, options(nomem, nostack));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.saved);
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn clip_cooperative_directed_and_flushed_environment_refuse_before_interval_work() {
    let _env = crate::tests::lock_env_shared();
    for flush in [false, true] {
        let altered = UnqualifiedEnvironment::enter(flush);
        assert!(dual_lower_outward_with_checkpoint(&[1.0], &[], &[0.0], &[1.0], &check).is_err());
        drop(altered);
        ny_core::require_f64_interval_proof_environment().unwrap();
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn clip_cooperative_requalifies_environment_after_caller_checkpoint_returns() {
    let _env = crate::tests::lock_env_shared();
    let state = RefCell::new(None);
    let calls = Cell::new(0);
    let callback = || {
        calls.set(calls.get() + 1);
        if calls.get() == 3 {
            *state.borrow_mut() = Some(UnqualifiedEnvironment::enter(true));
        }
        Ok(())
    };
    assert!(dual_lower_outward_with_checkpoint(&[1.0], &[], &[0.0], &[1.0], &callback).is_err());
    assert_eq!(calls.get(), 3);
    drop(state.into_inner());
    ny_core::require_f64_interval_proof_environment().unwrap();
}
