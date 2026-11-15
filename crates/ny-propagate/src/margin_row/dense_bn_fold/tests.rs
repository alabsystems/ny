// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact-rational oracle for the certified `Gemm -> BatchNormalization` fold
//! (#cct-margin-row-admission).
//!
//! The exact scale `s = gamma / sqrt(var + eps)` is irrational, but its SQUARE
//! `gamma^2 / (var + eps)` is an exact rational of the f64 inputs, and so is
//! the square of every product `s * x` with `x` rational. Each claim below is
//! therefore decided EXACTLY: a comparison `a <= s*x` becomes a sign case
//! split plus a comparison of squares, with no rounding anywhere in the
//! oracle.

use num_rational::BigRational;
use num_traits::Zero;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::{fold_dense_batchnorm, DenseBnFold, DenseBnParams};
use crate::margin_row::rounding::RHO_FOLD;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, deprecated)]
mod fp_environment {
    use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
    use std::marker::PhantomData;
    use std::rc::Rc;

    // Only rounding, FTZ and DAZ are changed; exception masks are preserved.
    const CONTROLS: u32 = (3 << 13) | (1 << 15) | (1 << 6);

    struct Restore {
        saved: u32,
        // FP control belongs to this thread, including during unwinding.
        _thread: PhantomData<Rc<()>>,
    }

    impl Drop for Restore {
        fn drop(&mut self) {
            // SAFETY: the value was read from this same thread's MXCSR and
            // this guard cannot be moved to another thread.
            unsafe { _mm_setcsr(self.saved) };
        }
    }

    pub(super) fn with_controls<T>(controls: u32, body: impl FnOnce() -> T) -> T {
        assert_eq!(controls & !CONTROLS, 0);
        // SAFETY: x86_64 has SSE; reading the calling thread's FP control
        // register accesses no memory owned by another thread.
        let saved = unsafe { _mm_getcsr() };
        let restore = Restore {
            saved,
            _thread: PhantomData,
        };
        // SAFETY: retain all reserved bits and exception masks from a valid
        // register value, changing only documented FP controls for this test.
        unsafe { _mm_setcsr((saved & !CONTROLS) | controls) };
        let result = body();
        drop(restore);
        result
    }
}

fn q(x: f64) -> BigRational {
    BigRational::from_float(x).expect("finite f64 is an exact dyadic")
}

fn rabs(x: &BigRational) -> BigRational {
    if *x < BigRational::zero() {
        -x.clone()
    } else {
        x.clone()
    }
}

fn rsign(x: &BigRational) -> i8 {
    let zero = BigRational::zero();
    if *x > zero {
        1
    } else if *x < zero {
        -1
    } else {
        0
    }
}

fn fsign(x: f64) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// Exact `a <= r` for the real `r = sign * sqrt(r2)` (`r2 >= 0`, `sign` the
/// exact sign of `r`, so `sign == 0` iff `r2 == 0`).
fn le_root(a: &BigRational, sign: i8, r2: &BigRational) -> bool {
    let zero = BigRational::zero();
    if sign >= 0 {
        *a <= zero || &(a * a) <= r2
    } else {
        *a < zero && &(a * a) >= r2
    }
}

/// Exact `r <= b` for the real `r = sign * sqrt(r2)`.
fn root_le(sign: i8, r2: &BigRational, b: &BigRational) -> bool {
    let zero = BigRational::zero();
    if sign <= 0 {
        *b >= zero || &(b * b) <= r2
    } else {
        *b > zero && r2 <= &(b * b)
    }
}

/// `s^2 = gamma^2 / (var + eps)`, exact.
fn s_squared(gamma: f64, var: f64, eps: f64) -> BigRational {
    let g = q(gamma);
    &(&g * &g) / &(q(var) + q(eps))
}

/// Draw an f32-representable value in `[lo, hi)` (every ONNX payload is f32).
fn f32v(rng: &mut StdRng, lo: f64, hi: f64) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    let v = rng.random_range(lo..hi) as f32;
    f64::from(v)
}

struct Case {
    n_out: usize,
    n_in: usize,
    weight: Vec<f64>,
    bias: Vec<f64>,
    gamma: Vec<f64>,
    beta: Vec<f64>,
    mean: Vec<f64>,
    var: Vec<f64>,
    eps: f64,
}

impl Case {
    fn random(rng: &mut StdRng, n_out: usize, n_in: usize) -> Self {
        let mut weight = Vec::with_capacity(n_out * n_in);
        for _ in 0..n_out * n_in {
            // ~10% exact zeros: a zero weight must fold exactly.
            weight.push(if rng.random_range(0.0..1.0) < 0.1 {
                0.0
            } else {
                f32v(rng, -1.0, 1.0)
            });
        }
        let bias = (0..n_out).map(|_| f32v(rng, -1.0, 1.0)).collect();
        let gamma = (0..n_out)
            .map(|_| {
                let r = rng.random_range(0.0..1.0);
                if r < 0.05 {
                    0.0 // s = 0: the row folds exactly
                } else if r < 0.5 {
                    -f32v(rng, 0.01, 3.0) // negative scales flip the head sign
                } else {
                    f32v(rng, 0.01, 3.0)
                }
            })
            .collect();
        let beta = (0..n_out).map(|_| f32v(rng, -2.0, 2.0)).collect();
        let mean = (0..n_out).map(|_| f32v(rng, -2.0, 2.0)).collect();
        let var = (0..n_out)
            .map(|_| {
                if rng.random_range(0.0..1.0) < 0.1 {
                    f32v(rng, 0.0, 1e-6) // variance near the eps floor
                } else {
                    f32v(rng, 1e-3, 5.0)
                }
            })
            .collect();
        Self {
            n_out,
            n_in,
            weight,
            bias,
            gamma,
            beta,
            mean,
            var,
            // ONNX's float attribute default 1e-5, widened exactly (cnn7 carries
            // exactly this value as an explicit attribute).
            eps: f64::from(1e-5_f32),
        }
    }

    fn params(&self) -> DenseBnParams<'_> {
        DenseBnParams {
            gamma: &self.gamma,
            beta: &self.beta,
            mean: &self.mean,
            var: &self.var,
            eps: self.eps,
        }
    }

    fn fold(&self) -> DenseBnFold {
        fold_dense_batchnorm(
            &self.weight,
            &self.bias,
            (self.n_out, self.n_in),
            &self.params(),
        )
        .expect("finite, normal, well-shaped case folds")
    }
}

/// Count violations of the per-entry weight claim
/// `|w' - s w| <= rel |w'|` under a CLAIMED `rel`.
fn weight_violations(case: &Case, fold: &DenseBnFold, rel: f64) -> usize {
    let rel_q = q(rel);
    let mut bad = 0usize;
    for o in 0..case.n_out {
        let s2 = s_squared(case.gamma[o], case.var[o], case.eps);
        for i in 0..case.n_in {
            let w = case.weight[o * case.n_in + i];
            let wp = q(fold.weight[o * case.n_in + i]);
            let sign = fsign(case.gamma[o]) * fsign(w);
            let wq = q(w);
            let r2 = &s2 * &(&wq * &wq);
            let slack = &rel_q * &rabs(&wp);
            let lo = &wp - &slack;
            let hi = &wp + &slack;
            if !(le_root(&lo, sign, &r2) && root_le(sign, &r2, &hi)) {
                bad += 1;
            }
        }
    }
    bad
}

/// Count violations of the per-row bias claim
/// `|b' - (beta + s (b - mean))| <= e` under CLAIMED errors `e`.
fn bias_violations(case: &Case, fold: &DenseBnFold, errs: &[f64]) -> usize {
    let mut bad = 0usize;
    for o in 0..case.n_out {
        let s2 = s_squared(case.gamma[o], case.var[o], case.eps);
        let d = q(case.bias[o]) - q(case.mean[o]);
        let sign = fsign(case.gamma[o]) * rsign(&d);
        let r2 = &s2 * &(&d * &d);
        let beta = q(case.beta[o]);
        let bp = q(fold.bias[o]);
        let e = q(errs[o]);
        let lo = &(&bp - &e) - &beta;
        let hi = &(&bp + &e) - &beta;
        if !(le_root(&lo, sign, &r2) && root_le(sign, &r2, &hi)) {
            bad += 1;
        }
    }
    bad
}

/// THE SOUNDNESS TEST. For random f32-valued Gemm -> BatchNormalization
/// parameters (negative, zero and near-eps-floor scales included):
///
///   1. every folded weight is within `weight_rel_err` of the exact `s * w`
///      (relative to the STORED weight — the engine's charge form);
///   2. every folded bias is within its `bias_err` of the exact
///      `beta + s (b - mean)`;
///   3. on random input boxes, the folded affine map's range, widened by the
///      charge the lane actually applies (`rel * sum |w'| |z| + bias_err`),
///      ENCLOSES the exact Gemm -> BN map's range;
///
/// all decided in exact rational arithmetic. NEGATIVE CONTROL: the same
/// oracle with a claimed error of zero must find violations, so a vacuous
/// oracle cannot pass this test.
#[test]
fn fold_dense_batchnorm_encloses_exact_gemm_bn_rational() {
    let mut rng = StdRng::seed_from_u64(0xCC7_B4F0);
    let mut weights_checked = 0usize;
    let mut control_weight_violations = 0usize;
    let mut control_bias_violations = 0usize;
    for _ in 0..120 {
        let case = Case::random(&mut rng, 6, 10);
        let fold = case.fold();
        assert_eq!(fold.weight_rel_err, RHO_FOLD);
        assert!(fold.bias_err.iter().all(|e| e.is_finite() && *e > 0.0));

        // (1) + (2): per-parameter claims at the CERTIFIED budgets.
        assert_eq!(
            weight_violations(&case, &fold, fold.weight_rel_err),
            0,
            "a folded weight escapes its certified relative error"
        );
        assert_eq!(
            bias_violations(&case, &fold, &fold.bias_err),
            0,
            "a folded bias escapes its certified absolute error"
        );
        weights_checked += case.n_out * case.n_in;

        // Negative control: claim the fold is EXACT.
        control_weight_violations += weight_violations(&case, &fold, 0.0);
        control_bias_violations += bias_violations(&case, &fold, &vec![0.0; case.n_out]);

        // (3) box enclosure under the lane's charge form.
        let rel = q(fold.weight_rel_err);
        for _ in 0..4 {
            let mut zl = Vec::with_capacity(case.n_in);
            let mut zu = Vec::with_capacity(case.n_in);
            for _ in 0..case.n_in {
                let a = f32v(&mut rng, -1.5, 1.5);
                let b = f32v(&mut rng, -1.5, 1.5);
                zl.push(a.min(b));
                zu.push(a.max(b));
            }
            for o in 0..case.n_out {
                let row = &case.weight[o * case.n_in..(o + 1) * case.n_in];
                let frow = &fold.weight[o * case.n_in..(o + 1) * case.n_in];
                let d = q(case.bias[o]) - q(case.mean[o]);
                // Exact u = W z + (b - mean): min / max over the box.
                let mut u_min = d.clone();
                let mut u_max = d;
                // Folded y' = W' z + b': min / max, and the lane's charge.
                let mut yp_min = q(fold.bias[o]);
                let mut yp_max = q(fold.bias[o]);
                let mut charge = q(fold.bias_err[o]);
                for i in 0..case.n_in {
                    let (lo_i, hi_i) = (q(zl[i]), q(zu[i]));
                    let w = q(row[i]);
                    if row[i] >= 0.0 {
                        u_min += &w * &lo_i;
                        u_max += &w * &hi_i;
                    } else {
                        u_min += &w * &hi_i;
                        u_max += &w * &lo_i;
                    }
                    let wp = q(frow[i]);
                    if frow[i] >= 0.0 {
                        yp_min += &wp * &lo_i;
                        yp_max += &wp * &hi_i;
                    } else {
                        yp_min += &wp * &hi_i;
                        yp_max += &wp * &lo_i;
                    }
                    let zmax = if rabs(&lo_i) > rabs(&hi_i) {
                        rabs(&lo_i)
                    } else {
                        rabs(&hi_i)
                    };
                    charge += &(&rel * &rabs(&wp)) * &zmax;
                }
                let s2 = s_squared(case.gamma[o], case.var[o], case.eps);
                let beta = q(case.beta[o]);
                // Exact y = beta + s*u is monotone in u with the sign of s.
                let (u_at_min, u_at_max) = if case.gamma[o] >= 0.0 {
                    (&u_min, &u_max)
                } else {
                    (&u_max, &u_min)
                };
                let lower = &(&yp_min - &charge) - &beta;
                let upper = &(&yp_max + &charge) - &beta;
                let sign_lo = fsign(case.gamma[o]) * rsign(u_at_min);
                let sign_hi = fsign(case.gamma[o]) * rsign(u_at_max);
                assert!(
                    le_root(&lower, sign_lo, &(&s2 * &(u_at_min * u_at_min))),
                    "charged folded lower end sits ABOVE the exact Gemm->BN minimum (row {o})"
                );
                assert!(
                    root_le(sign_hi, &(&s2 * &(u_at_max * u_at_max)), &upper),
                    "charged folded upper end sits BELOW the exact Gemm->BN maximum (row {o})"
                );
            }
        }
    }
    assert!(
        weights_checked >= 7000,
        "oracle checked {weights_checked} weights"
    );
    assert!(
        control_weight_violations > 0,
        "NEGATIVE CONTROL: claiming zero weight error found no violation — the \
         oracle cannot distinguish a charged fold from an uncharged one"
    );
    assert!(
        control_bias_violations > 0,
        "NEGATIVE CONTROL: claiming zero bias error found no violation"
    );
}

/// Every refusal path returns `None` — never a partial fold that the caller
/// could mistake for a certified one.
#[test]
fn fold_dense_batchnorm_refuses_what_it_cannot_certify() {
    let w = vec![0.5, -0.25, 1.0, 2.0];
    let b = vec![0.1, -0.2];
    let ok = |gamma: &[f64], var: &[f64], eps: f64| {
        fold_dense_batchnorm(
            &w,
            &b,
            (2, 2),
            &DenseBnParams {
                gamma,
                beta: &[0.0, 0.0],
                mean: &[0.0, 0.0],
                var,
                eps,
            },
        )
    };
    // Positive control: a plain, well-conditioned fold succeeds.
    assert!(ok(&[1.0, -2.0], &[1.0, 4.0], 1e-5).is_some());
    // var + eps <= 0.
    assert!(ok(&[1.0, 1.0], &[-1e-5, 1.0], 1e-5).is_none());
    assert!(ok(&[1.0, 1.0], &[-2.0, 1.0], 1e-5).is_none());
    // Non-finite parameters.
    assert!(ok(&[f64::NAN, 1.0], &[1.0, 1.0], 1e-5).is_none());
    assert!(ok(&[1.0, 1.0], &[f64::INFINITY, 1.0], 1e-5).is_none());
    assert!(ok(&[1.0, 1.0], &[1.0, 1.0], f64::NAN).is_none());
    // A nonzero product that lands subnormal (a ~ 3.2e-308 is normal, a * 0.5
    // is not): the relative bound would fail, so the fold refuses.
    assert!(ok(&[1e-300, 1.0], &[1e15, 1.0], 1e-5).is_none());
    // A nonzero scale that is itself subnormal.
    assert!(ok(&[1e-300, 1.0], &[1e17, 1.0], 1e-5).is_none());
    // Shape mismatch.
    assert!(fold_dense_batchnorm(
        &w,
        &b,
        (2, 3),
        &DenseBnParams {
            gamma: &[1.0, 1.0],
            beta: &[0.0, 0.0],
            mean: &[0.0, 0.0],
            var: &[1.0, 1.0],
            eps: 1e-5,
        },
    )
    .is_none());
    // gamma == 0 folds EXACTLY: zero weights, bias == beta.
    let exact = fold_dense_batchnorm(
        &w,
        &b,
        (2, 2),
        &DenseBnParams {
            gamma: &[0.0, 1.0],
            beta: &[0.75, 0.0],
            mean: &[3.0, 0.0],
            var: &[1.0, 1.0],
            eps: 1e-5,
        },
    )
    .expect("gamma == 0 is a legal (exact) row");
    assert_eq!(&exact.weight[..2], &[0.0, 0.0]);
    assert_eq!(exact.bias[0], 0.75);
}

/// The smallest binary32 payload is normal in binary64 and must survive the
/// actual certified fold. Amplification makes erasing it reverse a margin.
#[test]
fn fold_retains_smallest_binary32_weight_and_its_negative_amplified_margin() {
    ny_core::require_f64_interval_proof_environment().expect("qualified source producer");
    let delta = ny_core::f32_to_f64_exact(f32::from_bits(1));
    let fold = fold_dense_batchnorm(
        &[delta],
        &[0.0],
        (1, 1),
        &DenseBnParams {
            gamma: &[1.0],
            beta: &[0.0],
            mean: &[0.0],
            var: &[1.0],
            eps: 0.0,
        },
    )
    .expect("an identity BN preserves this normal binary64 weight");
    assert_eq!(fold.weight[0].to_bits(), delta.to_bits());

    let epsilon = f64::from_bits((1023 - 50) << 52);
    let amplifier = f64::from_bits((1023 + 100) << 52);
    assert_eq!(q(epsilon) - q(amplifier) * q(delta), -q(epsilon));
    // Exact rational pricing of the returned fold ball at input x=1:
    // even its largest possible final margin remains strictly negative.
    let lower_y = q(fold.weight[0]) - q(fold.weight_rel_err) * q(fold.weight[0].abs())
        + q(fold.bias[0])
        - q(fold.bias_err[0]);
    assert!(q(epsilon) - q(amplifier) * lower_y < BigRational::zero());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn fold_refuses_unqualified_producer_modes_and_restores_after_unwind() {
    ny_core::require_f64_interval_proof_environment().expect("qualified test entry");
    let produce = || {
        fold_dense_batchnorm(
            &[1.0],
            &[0.0],
            (1, 1),
            &DenseBnParams {
                gamma: &[1.0],
                beta: &[0.0],
                mean: &[0.0],
                var: &[1.0],
                eps: 0.0,
            },
        )
    };
    assert!(produce().is_some());
    for controls in [1 << 15, 1 << 6, 1 << 13, 2 << 13, 3 << 13] {
        let (qualified, refused) = fp_environment::with_controls(controls, || {
            (
                ny_core::has_f64_interval_proof_environment(),
                produce().is_none(),
            )
        });
        assert!(!qualified, "invalid control bits {controls:#x}");
        assert!(refused, "the actual producer accepted {controls:#x}");
        assert!(produce().is_some(), "the original FP mode was not restored");
    }
    let unwind = std::panic::catch_unwind(|| {
        fp_environment::with_controls(1 << 6, || panic!("test scoped FP-control unwind"))
    });
    assert!(unwind.is_err());
    assert!(
        produce().is_some(),
        "the original FP mode was not restored on unwind"
    );
}
