// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::certify::{
    residency::{CancellationToken, MemoryAuthority},
    work::{WorkLimits, WorkReserve},
};
use std::time::Duration;

fn counts() -> DecoderMathCounts {
    DecoderMathCounts {
        elementary: 4096,
        gelu: 16,
        trig: 16,
        roots: 16,
        exponentials: 16,
    }
}
fn budget() -> WorkBudget {
    WorkBudget::new(
        Some(Instant::now() + Duration::from_secs(30)),
        1 << 20,
        1 << 20,
        1 << 32,
        0,
        Duration::ZERO,
    )
    .unwrap()
}
fn contains(x: DecoderInterval, value: f64) {
    assert!(
        x.lo <= value && value <= x.hi,
        "{x:?} does not contain {value}"
    );
}

#[test]
fn decoder_scalar_exact_dyadic_signs_zero_underflow_and_division() {
    let b = budget();
    let mut m = CheckedDecoderMath::new(&b, None, counts()).unwrap();
    contains(
        m.add(DecoderInterval::point(0.5), DecoderInterval::point(-0.25))
            .unwrap(),
        0.25,
    );
    contains(
        m.sub(DecoderInterval::point(-0.5), DecoderInterval::point(0.25))
            .unwrap(),
        -0.75,
    );
    contains(
        m.mul(
            DecoderInterval { lo: -0.5, hi: 0.75 },
            DecoderInterval { lo: -2.0, hi: 1.0 },
        )
        .unwrap(),
        -1.5,
    );
    contains(
        m.div_positive(DecoderInterval::point(0.75), DecoderInterval::point(3.0))
            .unwrap(),
        0.25,
    );
    let tiny = f64::from_bits(1);
    let half = m
        .mul(DecoderInterval::point(-tiny), DecoderInterval::point(0.5))
        .unwrap();
    assert!(half.lo < 0.0 && half.hi >= 0.0);
    contains(
        m.mul(DecoderInterval::point(-0.0), DecoderInterval::point(2.0))
            .unwrap(),
        0.0,
    );
    assert!(m
        .div_positive(
            DecoderInterval::point(1.0),
            DecoderInterval { lo: 0.0, hi: 1.0 }
        )
        .is_err());
    assert!(m
        .add(
            DecoderInterval { lo: 1.0, hi: 0.0 },
            DecoderInterval::point(0.0)
        )
        .is_err());
    assert!(m
        .mul(
            DecoderInterval::point(f64::INFINITY),
            DecoderInterval::point(0.0)
        )
        .is_err());
}

#[test]
fn decoder_scalar_rope_analytic_rational_series_and_periodic_reduction() {
    let b = budget();
    let mut m = CheckedDecoderMath::new(&b, None, counts()).unwrap();
    let (s, c) = m.sin_cos(DecoderInterval::point(0.0)).unwrap();
    contains(s, 0.0);
    contains(c, 1.0);
    // Independent rational alternating-series brackets for sin(1),cos(1).
    let (s, c) = m.sin_cos(DecoderInterval::point(1.0)).unwrap();
    assert!(s.lo <= 0.8414709848078965 && s.hi >= 0.8414709848078966);
    assert!(c.lo <= 0.5403023058681397 && c.hi >= 0.5403023058681398);
    let (s, c) = m.sin_cos(DecoderInterval::point(-1.0)).unwrap();
    assert!(s.lo <= -0.8414709848078966 && s.hi >= -0.8414709848078965);
    assert!(c.lo <= 0.5403023058681397 && c.hi >= 0.5403023058681398);
    // Integer-valued original angle1000 requires genuine checked reduction.
    let (s, c) = m.sin_cos(DecoderInterval::point(1000.0)).unwrap();
    assert!(s.lo <= 0.8268795405320025 && s.hi >= 0.8268795405320026);
    assert!(c.lo <= 0.5623790762907029 && c.hi >= 0.562_379_076_290_703);
    assert!(m.sin_cos(DecoderInterval { lo: -5.0, hi: 5.0 }).is_err());
    assert!(m.sin_cos(DecoderInterval::point(5000.0)).is_err());
}

#[test]
fn decoder_scalar_exact_math_sqrt_gelu_and_softmax_exponent() {
    let b = budget();
    let mut m = CheckedDecoderMath::new(&b, None, counts()).unwrap();
    contains(m.sqrt(DecoderInterval::point(4.0)).unwrap(), 2.0);
    let root = m.sqrt(DecoderInterval::point(2.0)).unwrap();
    assert!(root.lo * root.lo <= 2.0 && root.hi * root.hi >= 2.0);
    let tiny = m.sqrt(DecoderInterval::point(f64::from_bits(1))).unwrap();
    assert!(tiny.lo >= 0.0 && tiny.hi > 0.0);
    contains(m.gelu(DecoderInterval::point(0.0)).unwrap(), 0.0);
    let g = m.gelu(DecoderInterval { lo: -0.1, hi: 0.1 }).unwrap();
    assert!(g.lo < 0.0 && g.hi > 0.0);
    contains(
        m.exponential_nonpositive(DecoderInterval::point(0.0))
            .unwrap(),
        1.0,
    );
    let e = m
        .exponential_nonpositive(DecoderInterval::point(-1.0))
        .unwrap();
    assert!(e.lo <= 0.3678794411714423 && e.hi >= 0.3678794411714424);
    assert!(m
        .exponential_nonpositive(DecoderInterval::point(0.01))
        .is_err());
    assert!(m.sqrt(DecoderInterval::point(-1.0)).is_err());
}

#[test]
fn decoder_scalar_sqrt_exact_rational_squares_cover_zero_underflow_and_overflow_bins() {
    use num_rational::BigRational;

    let b = budget();
    let mut m = CheckedDecoderMath::new(&b, None, counts()).unwrap();
    for bits in [
        0u64,
        1,
        2,
        3,
        (1u64 << 52) - 1,
        1u64 << 52,
        ((1023u64 - 512) << 52) - 1,
        (1023u64 + 512) << 52,
        f64::MAX.to_bits(),
    ] {
        let x = f64::from_bits(bits);
        let y = m.sqrt(DecoderInterval::point(x)).unwrap();
        let exact = BigRational::from_float(x).unwrap();
        let lo = BigRational::from_float(y.lo).unwrap();
        let hi = BigRational::from_float(y.hi).unwrap();
        assert!(y.lo >= 0.0 && y.hi >= y.lo);
        assert!(
            &lo * &lo <= exact && &hi * &hi >= exact,
            "bits={bits:x}: {y:?}"
        );
    }
    let tiny = f64::from_bits(1);
    let y = m.sqrt(DecoderInterval { lo: 0.0, hi: tiny }).unwrap();
    let hi = BigRational::from_float(y.hi).unwrap();
    assert_eq!(y.lo, 0.0);
    assert!(&hi * &hi >= BigRational::from_float(tiny).unwrap());
    m.checkpoint().unwrap();
}

#[test]
fn decoder_scalar_paid_floor_inherited_clock_and_last_owner() {
    let request = counts().request().unwrap();
    let b = WorkBudget::new(
        Some(Instant::now() + Duration::from_secs(30)),
        request.scratch_bytes - 1,
        request.persistent_bytes,
        request.work,
        0,
        Duration::ZERO,
    )
    .unwrap();
    assert!(CheckedDecoderMath::new(&b, None, counts()).is_err());
    assert_eq!(b.receipt().unwrap().live_persistent_bytes, 0);
    let earlier = Instant::now() + Duration::from_secs(10);
    let b = WorkBudget::new(Some(earlier), 1 << 20, 1 << 20, 1 << 32, 0, Duration::ZERO).unwrap();
    let m = CheckedDecoderMath::new(&b, Some(earlier + Duration::from_secs(10)), counts()).unwrap();
    assert_eq!(m.deadline(), Some(earlier));
    let claim = m.finish().unwrap();
    let alias = claim.clone();
    drop(claim);
    assert!(b.receipt().unwrap().live_persistent_bytes > 0);
    drop(alias);
    assert_eq!(b.receipt().unwrap().live_persistent_bytes, 0);
    assert!(CheckedDecoderMath::new(&b, Some(Instant::now()), counts()).is_err());
}

#[test]
fn decoder_scalar_same_original_cancellation_between_real_operations() {
    let cancel = CancellationToken::default();
    let memory = MemoryAuthority::new(2 << 20);
    let b = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(Instant::now() + Duration::from_secs(30)),
            scratch_bytes: 1 << 20,
            persistent_bytes: 1 << 20,
            max_work: 1 << 32,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let mut m = CheckedDecoderMath::new(&b, None, counts()).unwrap();
    m.sin_cos(DecoderInterval::point(1.0)).unwrap();
    cancel.cancel();
    assert!(m
        .add(DecoderInterval::point(0.0), DecoderInterval::point(1.0))
        .is_err());
    drop(m);
    assert_eq!(b.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(b.receipt().unwrap().live_scratch_bytes, 0);
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
struct DirectedRounding {
    #[cfg(target_arch = "aarch64")]
    saved: u64,
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl DirectedRounding {
    #[allow(unsafe_code)]
    fn enter() -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let saved: u64;
            std::arch::asm!("mrs {saved}, fpcr",saved=out(reg)saved,options(nomem,nostack));
            let directed = (saved & !(3 << 22)) | (1 << 22);
            std::arch::asm!("msr fpcr, {directed}","isb",directed=in(reg)directed,options(nomem,nostack));
            Self { saved }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let saved = std::arch::x86_64::_mm_getcsr();
            std::arch::x86_64::_mm_setcsr((saved & !(3 << 13)) | (1 << 13));
            Self { saved }
        }
    }
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl Drop for DirectedRounding {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("msr fpcr, {saved}","isb",saved=in(reg)self.saved,options(nomem,nostack));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.saved);
        }
    }
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn decoder_scalar_live_changed_environment_cannot_continue_or_publish() {
    let b = budget();
    let mut math = CheckedDecoderMath::new(&b, None, counts()).unwrap();
    math.sin_cos(DecoderInterval::point(1.0)).unwrap();
    let second = CheckedDecoderMath::new(&b, None, counts()).unwrap();
    let guard = DirectedRounding::enter();
    assert!(math.gelu(DecoderInterval::point(0.5)).is_err());
    assert!(math.sin_cos(DecoderInterval::point(0.0)).is_err());
    assert!(CheckedDecoderMath::new(&b, None, counts()).is_err());
    assert!(math.checkpoint().is_err());
    assert!(second.finish().is_err());
    drop(guard);
    math.add(DecoderInterval::point(1.0), DecoderInterval::point(1.0))
        .unwrap();
    drop(math);
    assert_eq!(b.receipt().unwrap().live_persistent_bytes, 0);
}
