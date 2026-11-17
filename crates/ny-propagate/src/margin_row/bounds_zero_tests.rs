// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exact-zero seed-error controls. Constructed carriers test arithmetic only;
//! compiled fixtures use their own y-boxes and an original-network Q oracle.

use super::*;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use std::hint::black_box;

fn q(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}

fn network(w: f64, b: f64, final_w: [f64; 2], final_b: [f64; 2]) -> TwinNet {
    TwinNet::compile(&TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![w],
                bias: vec![b],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: final_w.to_vec(),
                bias: final_b.to_vec(),
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    })
    .unwrap()
}

#[test]
fn head_exact_zero_seed_error_requires_original_finite_zero_operand() {
    let values = [
        0.0,
        -0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        0.5,
        -1.0,
        f64::MAX,
        -f64::MAX,
    ];
    for z in [0.0, -0.0] {
        for v in values {
            for (a, b) in [(z, v), (v, z)] {
                let p = black_box(a) * black_box(b);
                assert_eq!(head_seed_error(a, b, p).to_bits(), 0);
                assert_eq!(q(a) * q(b) - q(p), BigRational::zero());
            }
        }
    }
    for (a, b) in [(f64::from_bits(1), 0.5), (-f64::from_bits(1), 0.5)] {
        let p = black_box(a) * black_box(b);
        let error = head_seed_error(a, b, p);
        assert_eq!(p.to_bits() & 0x7fff_ffff_ffff_ffff, 0);
        assert_eq!(error.to_bits(), 1);
        let exact_gap = (q(a) * q(b) - q(p)).abs();
        assert!(exact_gap > BigRational::zero());
        assert!(q(error) >= exact_gap);
    }
    // An inconsistent stored product cannot use the exemption either.
    assert_eq!(head_seed_error(0.0, 1.0, 1.0), slack16(3.0 * UNIT));
    for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        for (a, b) in [(0.0, bad), (bad, -0.0)] {
            assert!(head_seed_error(a, b, a * b).is_nan());
            assert_eq!(head_seed_error(a, b, 0.0).to_bits(), 1);
        }
    }
}

#[test]
fn head_exact_zero_compiled_identity_removes_artificial_confirmation() {
    let net = network(1.0, 0.0, [1.0, 0.0], [0.0; 2]);
    for mode in [RoundMode::Parity, RoundMode::Outward] {
        for hi in [1.0, 2.0] {
            let root = RootGates::build(&net, &[1.0], &[hi], mode, None).unwrap();
            let eng = BackwardEngine::new(&net, &root);
            let (al, au) = eng.y_rows(None).unwrap();
            let gates = head_gates(&YBox::from_rows(&eng, &al, &au), mode);
            let mb = MarginBatch::new(&net, 0, &[1]).unwrap();
            let ad = row_dots(&root, &al);
            let ud = row_dots(&root, &au);
            assert_eq!((mb.wp[0], mb.wn[0]), (1.0, 0.0));
            assert_eq!((gates.alpha[0], gates.s[0], gates.c[0]), (1.0, 1.0, 0.0));
            let (old, old_counts) = confirm::with_counts(|| {
                compose_viay_impl::<true, false>(&eng, &mb, &gates, &al, &au, &ad, &ud, mode)
            });
            let (new, new_counts) =
                confirm::with_counts(|| compose_viay(&eng, &mb, &gates, &al, &au, &ad, &ud, mode));
            assert!(new[0].is_finite() && old[0].is_finite());
            // Exact original minimum of ReLU(ReLU(x)) on [1, hi] is 1.
            assert!(q(new[0]) <= q(1.0));
            assert!(q(old[0]) <= q(1.0));
            if mode.outward() {
                assert_eq!(old_counts, [1, 1, 1]);
                assert!(new[0] >= old[0]);
                // The outward point radius is eta. Its genuine subnormal
                // concretization product must still request confirmation.
                // The non-point interval isolates the artificial seed dust.
                assert_eq!(head_underflow(1.0, root.rad[0]), hi == 1.0);
                assert_eq!(new_counts, if hi == 1.0 { [1, 1, 1] } else { [0; 3] });
            } else {
                assert_eq!(old_counts, [0; 3]);
                assert_eq!(new_counts, [0; 3]);
                assert_eq!(new[0].to_bits(), old[0].to_bits());
            }
        }
    }
}

#[test]
fn head_exact_zero_keeps_genuine_seed_and_intercept_underflow_confirmation() {
    let net = network(1.0, 0.0, [1.0, 0.0], [0.0; 2]);
    let root = RootGates::build(&net, &[1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let pack = PassOut {
        a: Array2::ones((1, 1)),
        e: Some(Array2::zeros((1, 1))),
        b: vec![0.0],
        eb: vec![0.0],
        coll: None,
        coll_rows: None,
        unst_rows: None,
    };
    let dots = row_dots(&root, &pack);
    let eta = f64::from_bits(1);
    // Arithmetic carriers, not authenticated neural-network margins.
    for (wp, wn, alpha, slope, intercept) in [
        (eta, 0.0, 0.5, 1.0, 0.0),
        (0.0, -eta, 1.0, 0.5, 0.0),
        (0.0, -0.5, 1.0, 0.0, eta),
    ] {
        let mb = MarginBatch {
            t: 0,
            adv: vec![1],
            wp: vec![wp],
            wn: vec![wn],
            cst: vec![0.0],
            n_y: 1,
        };
        let gates = HeadGates {
            alpha: vec![alpha],
            s: vec![slope],
            c: vec![intercept],
        };
        assert!(
            head_underflow(wp, alpha) || head_underflow(wn, slope) || head_underflow(wn, intercept)
        );
        let (new, counts) = confirm::with_counts(|| {
            compose_viay(
                &eng,
                &mb,
                &gates,
                &pack,
                &pack,
                &dots,
                &dots,
                RoundMode::Outward,
            )
        });
        assert_eq!(counts, [1, 1, 1]);
        let old = compose_viay_impl::<false, false>(
            &eng,
            &mb,
            &gates,
            &pack,
            &pack,
            &dots,
            &dots,
            RoundMode::Outward,
        );
        let truth = q(wp) * q(alpha) + q(wn) * q(slope) + q(wn) * q(intercept);
        assert!(new[0].is_finite() && old[0].is_finite());
        assert!(q(new[0]) <= truth);
        assert!(new[0] >= old[0]);
    }
}

#[test]
fn head_exact_zero_compiled_piecewise_minimum_rational_containment() {
    // These are all breakpoints of this 1D original network. Its output is
    // affine between consecutive points, so their minimum is the exact
    // whole-domain minimum, not a sampled lower-bound oracle.
    let lo = q(-1.0);
    let hi = q(2.0);
    for (w, b) in [(1.0, -0.5), (-1.0, 0.75), (0.25, -0.125)] {
        for final_w in [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, -0.25]] {
            let final_b = [0.125, -0.25];
            let net = network(w, b, final_w, final_b);
            let root = RootGates::build(&net, &[-1.0], &[2.0], RoundMode::Outward, None).unwrap();
            let eng = BackwardEngine::new(&net, &root);
            let (al, au) = eng.y_rows(None).unwrap();
            let ybox = YBox::from_rows(&eng, &al, &au);
            let mb = MarginBatch::new(&net, 0, &[1]).unwrap();
            let ad = row_dots(&root, &al);
            let ud = row_dots(&root, &au);
            let mut points = vec![lo.clone(), hi.clone(), BigRational::zero()];
            let kink = -q(b) / q(w);
            if kink >= BigRational::zero() && kink >= lo && kink <= hi {
                points.push(kink);
            }
            let truth = points
                .into_iter()
                .map(|x| {
                    let h = (q(w) * x.max(BigRational::zero()) + q(b)).max(BigRational::zero());
                    (q(final_w[0]) - q(final_w[1])) * h + q(final_b[0]) - q(final_b[1])
                })
                .min()
                .unwrap();
            for alpha in [0.0, 0.25, 1.0] {
                let gates = head_gates_with(&ybox, RoundMode::Outward, Some(&[alpha]));
                let new = compose_viay(&eng, &mb, &gates, &al, &au, &ad, &ud, RoundMode::Outward);
                let old = compose_viay_impl::<false, false>(
                    &eng,
                    &mb,
                    &gates,
                    &al,
                    &au,
                    &ad,
                    &ud,
                    RoundMode::Outward,
                );
                let cad = row_dots_with_checkpoint(&root, &al, &|| Ok(())).unwrap();
                let cud = row_dots_with_checkpoint(&root, &au, &|| Ok(())).unwrap();
                let cooperative = compose_viay_with_checkpoint(
                    &eng,
                    &mb,
                    &gates,
                    &al,
                    &au,
                    &cad,
                    &cud,
                    RoundMode::Outward,
                    &|| Ok(()),
                )
                .unwrap();
                assert!(new[0].is_finite() && old[0].is_finite() && cooperative[0].is_finite());
                assert!(new[0] >= old[0]);
                assert!(
                    q(new[0]) <= truth,
                    "w={w} b={b} alpha={alpha} final={final_w:?}"
                );
                assert!(q(cooperative[0]) <= truth);
            }
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
struct FlushedEnvironment {
    #[cfg(target_arch = "aarch64")]
    saved: u64,
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl FlushedEnvironment {
    #[allow(unsafe_code)]
    fn enter() -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let saved: u64;
            std::arch::asm!("mrs {saved}, fpcr", saved=out(reg) saved, options(nomem, nostack));
            let altered = saved | (1_u64 << 24);
            std::arch::asm!("msr fpcr, {altered}", "isb", altered=in(reg) altered, options(nomem, nostack));
            Self { saved }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let saved = std::arch::x86_64::_mm_getcsr();
            std::arch::x86_64::_mm_setcsr(saved | (1 << 15) | (1 << 6));
            Self { saved }
        }
    }
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl Drop for FlushedEnvironment {
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
fn head_exact_zero_daz_does_not_reclassify_nonzero_subnormal_operands() {
    let _env = crate::tests::lock_env_shared();
    ny_core::require_f64_interval_proof_environment().unwrap();
    let altered = FlushedEnvironment::enter();
    assert!(ny_core::require_f64_interval_proof_environment().is_err());
    for tiny in [f64::from_bits(1), -f64::from_bits(1)] {
        for (a, b) in [(tiny, 0.5), (0.5, tiny)] {
            let a = black_box(a);
            let b = black_box(b);
            let p = black_box(a * b);
            assert_ne!(a.to_bits() & 0x7fff_ffff_ffff_ffff, 0);
            assert_ne!(b.to_bits() & 0x7fff_ffff_ffff_ffff, 0);
            assert_eq!(head_seed_error(a, b, p).to_bits(), 1);
            for z in [0.0, -0.0] {
                for (left, right) in [(z, tiny), (tiny, z)] {
                    let left = black_box(left);
                    let right = black_box(right);
                    assert_eq!(
                        head_seed_error(left, right, black_box(left * right)).to_bits(),
                        0
                    );
                }
            }
        }
    }
    drop(altered);
    ny_core::require_f64_interval_proof_environment().unwrap();
    assert_eq!(head_seed_error(f64::from_bits(1), 0.5, 0.0).to_bits(), 1);
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn head_exact_zero_composer_refuses_unqualified_environment_and_recovers() {
    let _env = crate::tests::lock_env_shared();
    let net = network(1.0, 0.0, [1.0, 0.0], [0.0; 2]);
    let root = RootGates::build(&net, &[1.0], &[2.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).unwrap();
    let gates = head_gates(&YBox::from_rows(&eng, &al, &au), RoundMode::Outward);
    let mb = MarginBatch::new(&net, 0, &[1]).unwrap();
    let ad = row_dots(&root, &al);
    let ud = row_dots(&root, &au);
    let compose = |mode| compose_viay(&eng, &mb, &gates, &al, &au, &ad, &ud, mode);
    let (qualified, counts) = confirm::with_counts(|| compose(RoundMode::Outward));
    assert!(qualified[0].is_finite());
    assert_eq!(counts, [0; 3]);
    for flush in [false, true] {
        let rounding = (!flush).then(cooperative_tests::HeadDirectedRounding::enter);
        let flushing = flush.then(FlushedEnvironment::enter);
        assert!(ny_core::require_f64_interval_proof_environment().is_err());
        let (refused, counts) = confirm::with_counts(|| compose(RoundMode::Outward));
        assert!(refused.iter().all(|x| x.is_nan()));
        assert_eq!(counts, [0; 3]);
        // Parity retains its original behavior in the same environment; its
        // numerical answer need not be invariant to a changed rounding mode.
        let parity = compose_viay_impl::<false, false>(
            &eng,
            &mb,
            &gates,
            &al,
            &au,
            &ad,
            &ud,
            RoundMode::Parity,
        );
        assert_eq!(compose(RoundMode::Parity)[0].to_bits(), parity[0].to_bits());
        drop(flushing);
        drop(rounding);
        ny_core::require_f64_interval_proof_environment().unwrap();
        assert_eq!(
            compose(RoundMode::Outward)[0].to_bits(),
            qualified[0].to_bits()
        );
    }
}
