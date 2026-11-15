// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Rational arithmetic controls and actual original-root clipping consumers.
//! The primitive counterexamples are not original-model source certificates
//! and do not claim to reproduce a final NY verdict.

use super::*;
use crate::margin_row::net::TwinNet;
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};
use std::time::{Duration, Instant};

fn q(value: f64) -> Q {
    Q::from_float(value).unwrap()
}

fn exact_dot(a: &[f64], x: &[f64]) -> Q {
    a.iter()
        .zip(x)
        .fold(Q::zero(), |s, (&a, &x)| s + q(a) * q(x))
}

fn tautology(n: usize) -> Halfspace {
    Halfspace {
        a: vec![0.0; n],
        d: 0.0,
    }
}

#[test]
fn clip_outward_cancellation_retains_the_exact_negative_unit() {
    let _env = crate::tests::lock_env_shared();
    let obj = [1e16, -1.0, -1e16];
    let x = [1.0; 3];
    let hs = [tautology(3)];
    let exact = exact_dot(&obj, &x);
    assert_eq!(exact, q(-1.0));
    // Independent arithmetic reproducer: the old nominal accumulation loses
    // the middle term. No fabricated ClipRows or solver verdict is involved.
    assert_eq!(dual_lower(&obj, &hs, &x, &[0.0; 3]), Some(0.0));
    let certified = dual_lower_outward(&obj, &hs, &x, &[0.0; 3]).unwrap();
    assert_eq!(certified.nominal, 0.0);
    assert!(q(certified.lower) <= exact);
    assert!(q(certified.magnitude) >= obj.iter().fold(Q::zero(), |s, &v| s + q(v).abs()));
}

#[test]
fn clip_outward_negative_half_subnormal_does_not_publish_zero() {
    let _env = crate::tests::lock_env_shared();
    let obj = [-f64::from_bits(1)];
    let x = [0.5];
    let exact = exact_dot(&obj, &x);
    assert!(exact < Q::zero());
    assert_eq!(exact * q(2.0), q(obj[0]));
    assert_eq!(dual_lower(&obj, &[tautology(1)], &x, &[0.0]), Some(0.0));
    let certified = dual_lower_outward(&obj, &[tautology(1)], &x, &[0.0]).unwrap();
    assert!(q(certified.lower) <= exact_dot(&obj, &x));
    assert!(certified.lower < 0.0 && certified.magnitude > 0.0);
}

#[test]
fn clip_outward_nonnegative_dual_point_has_rational_feasible_bounds() {
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
    let certified = dual_lower_outward(&obj, &hs, &[0.0; 3], &[1.0; 3]).unwrap();
    // The exact optimum is x=(0,1/4,-1), objective -13/4. The search
    // selects beta=(3,5), so its whole operand magnitude is at least 19.25.
    assert!(q(certified.lower) <= q(-3.25));
    assert!(certified.lower > -3.250_000_000_001);
    assert!(q(certified.magnitude) >= q(19.25));
    let mut checked = 0;
    for a in 0..=4 {
        for b in -4..=1 {
            for c in -4..=4 {
                let point = [f64::from(a) / 4.0, f64::from(b) / 4.0, f64::from(c) / 4.0];
                assert!(hs
                    .iter()
                    .all(|h| exact_dot(&h.a, &point) + q(h.d) <= Q::zero()));
                assert!(q(certified.lower) <= exact_dot(&obj, &point));
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 270);
}

#[test]
fn clip_outward_exact_dyadic_phase_constraint_keeps_useful_zero_bound() {
    let _env = crate::tests::lock_env_shared();
    let active = Halfspace {
        a: vec![-1.0],
        d: 0.0,
    };
    let result = dual_lower_outward(&[1.0], &[active], &[0.0], &[1.0]).unwrap();
    assert_eq!(result.lower, 0.0);
    assert_eq!(result.nominal, 0.0);
    assert!(result.magnitude >= 2.0);
}

#[test]
fn clip_outward_fractional_beta_folds_enclose_the_independent_rational_dual() {
    let _env = crate::tests::lock_env_shared();
    let obj = [0.1, -0.3, 0.7];
    let x0 = [0.0, 0.1, -0.2];
    let eps = [1.0, 0.9, 0.8];
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
    let mut exact_coefficients: Vec<Q> = obj.iter().map(|&v| q(v)).collect();
    let mut exact_magnitudes: Vec<Q> = obj.iter().map(|&v| q(v).abs()).collect();
    let mut scratch = Vec::new();
    let (mut bias, mut magnitude) = (Q::zero(), Q::zero());
    let mut positive = 0;
    for h in &hs {
        // The proposal is deliberately still nominal. This oracle certifies
        // the exact real Lagrangian at the ACTUAL selected binary64 beta;
        // it does not reuse any directed production arithmetic.
        let beta = dual_argmax(&nominal, h, &x0, &eps, &mut scratch);
        assert!(beta.is_finite() && beta >= 0.0);
        if beta > 0.0 {
            positive += 1;
            for j in 0..obj.len() {
                nominal[j] = beta.mul_add(h.a[j], nominal[j]);
                exact_coefficients[j] += q(beta) * q(h.a[j]);
                exact_magnitudes[j] += q(beta) * q(h.a[j]).abs();
            }
            bias += q(beta) * q(h.d);
            magnitude += q(beta) * q(h.d).abs();
        }
    }
    assert_eq!(positive, 2);
    let mut exact_dual = bias;
    for j in 0..obj.len() {
        let l = q(x0[j]) - q(eps[j]);
        let u = q(x0[j]) + q(eps[j]);
        exact_dual += (&exact_coefficients[j] * &l).min(&exact_coefficients[j] * &u);
        magnitude += &exact_magnitudes[j] * (q(x0[j]).abs() + q(eps[j]));
    }
    let result = dual_lower_outward(&obj, &hs, &x0, &eps).unwrap();
    assert!(q(result.lower) <= exact_dual);
    assert!(q(result.magnitude) >= magnitude);
    assert!(result.lower > -1.0);
}

#[test]
fn clip_outward_twosum_and_products_enclose_signed_ties_and_subnormals() {
    let _env = crate::tests::lock_env_shared();
    let tiny = f64::from_bits(1);
    for (a, b) in [
        (1e16, -1.0),
        (1e16, 1.0),
        (1.0, -1.0),
        (1.0, f64::EPSILON / 2.0),
        (-1.0, -f64::EPSILON / 2.0),
        (tiny, -2.0 * tiny),
        (-tiny, tiny),
        (0.0, -0.0),
    ] {
        let (center, residual) = clip_two_sum(a, b).unwrap();
        let exact = q(a) + q(b);
        assert_eq!(q(center) + q(residual), exact);
        assert!(q(clip_add_down(a, b).unwrap()) <= exact);
        assert!(q(clip_add_up(a, b).unwrap()) >= exact);
    }
    for (a, b) in [
        (-tiny, 0.5),
        (tiny, 0.5),
        (-3.0 * tiny, 0.5),
        (0.3, -0.7),
        (-0.3, -0.7),
        (0.0, f64::MAX),
        (1.0, -tiny),
    ] {
        let exact = q(a) * q(b);
        let (lower, upper) = clip_mul_bounds(a, b).unwrap();
        assert!(q(lower) <= exact && exact <= q(upper));
    }
}

#[test]
fn clip_outward_refuses_invalid_shapes_nonfinite_values_and_exceptional_products() {
    let _env = crate::tests::lock_env_shared();
    assert!(dual_lower_outward(&[1.0], &[], &[], &[1.0]).is_none());
    assert!(dual_lower_outward(&[1.0], &[], &[0.0], &[-1.0]).is_none());
    assert!(dual_lower_outward(&[f64::NAN], &[], &[0.0], &[1.0]).is_none());
    assert!(
        dual_lower_outward(&[1.0], &[Halfspace { a: vec![], d: 0.0 }], &[0.0], &[1.0]).is_none()
    );
    assert!(dual_lower_outward(
        &[1.0],
        &[Halfspace {
            a: vec![0.0],
            d: f64::INFINITY
        }],
        &[0.0],
        &[1.0]
    )
    .is_none());
    assert!(dual_lower_outward(&[f64::MAX], &[], &[0.0], &[2.0]).is_none());
    assert!(clip_two_sum(f64::MAX, f64::MAX).is_none());
}

fn original_root(env: &mut ny_test_utils::env::EnvEditor, fold: bool) -> RootGates {
    env.set("NY_MARGIN_ROW_CLIP_ROWS", "1");
    env.set("NY_MARGIN_ROW_CLIP_INTERM", "1");
    env.set("NY_MARGIN_ROW_CLIP_TOPK", "20");
    env.set("NY_MARGIN_ROW_CLIP_STATEFUL", "0");
    let spec = TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0, 2.0],
                bias: if fold {
                    vec![0.125, -0.25]
                } else {
                    vec![0.0; 2]
                },
                bias_err: if fold {
                    vec![0.003, 0.005]
                } else {
                    vec![0.0; 2]
                },
                weight_rel_err: if fold { 0.025 } else { 0.0 },
                kernel: (2, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (2, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![1.0, 0.0, 0.0, 1.0],
                bias: vec![0.0; 2],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![1.0, -1.0, -1.0, 1.0],
                bias: vec![0.0; 2],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    };
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build_prec_store(
        &net,
        &[-1.0],
        &[1.0],
        RoundMode::Outward,
        Some(Instant::now() + Duration::from_secs(30)),
        false,
        false,
    )
    .unwrap();
    assert_eq!(root.layers[0].unst, vec![0, 1]);
    assert!(root.layers[0].clip_rows.is_some());
    root
}

#[test]
fn clip_original_root_halfspaces_contain_every_exact_phase_member() {
    ny_test_utils::env::with_env_edits(|env| {
        let root = original_root(env, false);
        for pos in 0..2 {
            for sign in [-1, 1] {
                let hs = halfspaces_for_splits(&root, &[(0, pos, sign)]);
                assert_eq!(hs.len(), 1);
                let mut checked = 0;
                for k in -16..=16 {
                    let x = f64::from(k) / 16.0;
                    let z = q(if pos == 0 { 1.0 } else { 2.0 }) * q(x);
                    if (sign > 0 && z >= Q::zero()) || (sign < 0 && z <= Q::zero()) {
                        assert!(exact_dot(&hs[0].a, &[x]) + q(hs[0].d) <= Q::zero());
                        checked += 1;
                    }
                }
                assert_eq!(checked, 17);
            }
        }
    });
}

#[test]
fn clip_original_fold_family_halfspaces_keep_bias_and_weight_charges() {
    ny_test_utils::env::with_env_edits(|env| {
        let root = original_root(env, true);
        let rows = root.layers[0].clip_rows.as_ref().unwrap();
        for pos in 0..2 {
            for sign in [-1, 1] {
                let hs = halfspaces_for_splits(&root, &[(0, pos, sign)]);
                assert_eq!(hs.len(), 1);
                let mut objective = [0.0];
                let (_, error) =
                    legacy_row_objective(rows, pos, sign < 0, &mut objective, |_| Some(1.0))
                        .unwrap();
                assert!(
                    error
                        >= if sign < 0 {
                            rows.sl_lo[pos]
                        } else {
                            rows.sl_up[pos]
                        }
                );
                let mut checked = 0;
                for weight_dir in [-1.0, 0.0, 1.0] {
                    for bias_dir in [-1.0, 0.0, 1.0] {
                        let w = q(if pos == 0 { 1.0 } else { 2.0 })
                            * (q(1.0) + q(weight_dir) * q(0.025));
                        let b = q(if pos == 0 { 0.125 } else { -0.25 })
                            + q(bias_dir) * q(if pos == 0 { 0.003 } else { 0.005 });
                        for k in -16..=16 {
                            let x = f64::from(k) / 16.0;
                            let z = &w * q(x) + &b;
                            if (sign > 0 && z >= Q::zero()) || (sign < 0 && z <= Q::zero()) {
                                assert!(exact_dot(&hs[0].a, &[x]) + q(hs[0].d) <= Q::zero());
                                checked += 1;
                            }
                        }
                    }
                }
                assert!(checked > 100);
            }
        }
    });
}

#[test]
fn clip_original_root_crossing_consumer_tightens_and_preserves_nonempty_domains() {
    ny_test_utils::env::with_env_edits(|env| {
        let root = original_root(env, false);
        for sign in [-1, 1] {
            let splits = [(0, 0, sign)];
            let hs = halfspaces_for_splits(&root, &splits);
            let (tightened, crossed, margin) = tighten_intermediate_acc_crossing(
                &root.layers[0],
                0,
                &hs,
                &root.mid,
                &root.rad,
                20,
                None,
            );
            assert_eq!(crossed, 0);
            assert_eq!(margin, f64::NEG_INFINITY);
            let other = tightened
                .iter()
                .find(|t| t.idx == 1)
                .expect("dependent actual original row must tighten");
            if sign > 0 {
                assert!(other.l > -0.001);
            } else {
                assert!(other.u < 0.001);
            }
            let mut checked = 0;
            for k in -16..=16 {
                let x = f64::from(k) / 16.0;
                if (sign > 0 && x >= 0.0) || (sign < 0 && x <= 0.0) {
                    let z = q(2.0) * q(x);
                    assert!(q(other.l) <= z && z <= q(other.u));
                    checked += 1;
                }
            }
            assert_eq!(checked, 17);
            // Invoke the actual pruning caller on this genuine completed root.
            // No source row or completion token is fabricated.
            let domain = crate::margin_row::engine::domain_gates(&root, &splits);
            assert!(!domain.infeasible);
        }
    });
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
    // Thread-local register, saved before alteration and restored by Drop
    // during both return and unwinding. No kernel runs while altered.
    #[allow(unsafe_code)]
    fn enter(flush: bool) -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let saved: u64;
            std::arch::asm!("mrs {saved}, fpcr", saved = out(reg) saved, options(nomem, nostack));
            let altered = if flush {
                saved | (1_u64 << 24)
            } else {
                (saved & !(3_u64 << 22)) | (1_u64 << 22)
            };
            std::arch::asm!("msr fpcr, {altered}", "isb", altered = in(reg) altered, options(nomem, nostack));
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
            std::arch::asm!("msr fpcr, {saved}", "isb", saved = in(reg) self.saved, options(nomem, nostack));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.saved);
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn clip_outward_and_original_consumers_refuse_directed_or_flushed_environment() {
    ny_test_utils::env::with_env_edits(|env| {
        let root = original_root(env, false);
        let hs = halfspaces_for_splits(&root, &[(0, 0, 1)]);
        assert_eq!(hs.len(), 1);
        for flush in [false, true] {
            let environment = UnqualifiedEnvironment::enter(flush);
            assert!(!ny_core::has_f64_interval_proof_environment());
            assert!(dual_lower_outward(&[1.0], &hs, &[0.0], &[1.0]).is_none());
            assert!(halfspaces_for_splits(&root, &[(0, 0, 1)]).is_empty());
            let (tight, crossed, margin) = tighten_intermediate_acc_crossing(
                &root.layers[0],
                0,
                &hs,
                &root.mid,
                &root.rad,
                20,
                None,
            );
            assert!(tight.is_empty());
            assert_eq!((crossed, margin), (0, f64::NEG_INFINITY));
            drop(environment);
            ny_core::require_f64_interval_proof_environment().unwrap();
        }
    });
}

#[test]
fn clip_outward_observation_preserves_kernel_bits_and_exact_rational_containment() {
    let _env = crate::tests::lock_env_shared();
    let tiny = f64::from_bits(1);
    for (obj, mid, rad) in [
        ([1e16, -1.0, -1e16], [1.0; 3], [0.0; 3]),
        ([1.0, -2.0, 0.5], [-0.5, 0.0, 0.25], [0.125, 0.5, 0.25]),
        ([tiny, -tiny, 0.0], [1.0, 1.0, 0.0], [0.0; 3]),
    ] {
        let hs = [tautology(3)];
        let kernel = dual_lower_outward(&obj, &hs, &mid, &rad).unwrap();
        let observed = dual_lower_outward_observation(&obj, &hs, &mid, &rad).unwrap();
        assert_eq!(observed.directed_lower.to_bits(), kernel.lower.to_bits());
        assert_eq!(observed.nominal.to_bits(), kernel.nominal.to_bits());
        assert_eq!(
            observed.operand_magnitude.to_bits(),
            kernel.magnitude.to_bits()
        );
        let exact_minimum = (0..3).fold(Q::zero(), |sum, i| {
            sum + q(obj[i]) * q(mid[i]) - q(obj[i]).abs() * q(rad[i])
        });
        assert!(q(observed.directed_lower) <= exact_minimum);
        assert!(observed.operand_magnitude >= observed.nominal.abs());
    }
}

#[test]
fn clip_outward_observation_preserves_nonfinite_shape_and_radius_refusals() {
    let _env = crate::tests::lock_env_shared();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(dual_lower_outward_observation(&[bad], &[], &[0.0], &[1.0]).is_none());
        assert!(dual_lower_outward_observation(&[1.0], &[], &[bad], &[1.0]).is_none());
        assert!(dual_lower_outward_observation(&[1.0], &[], &[0.0], &[bad]).is_none());
    }
    assert!(dual_lower_outward_observation(&[1.0], &[], &[0.0], &[-1.0]).is_none());
    assert!(dual_lower_outward_observation(&[1.0], &[], &[], &[1.0]).is_none());
    assert!(dual_lower_outward_observation(&[1.0], &[tautology(2)], &[0.0], &[1.0]).is_none());
    assert!(dual_lower_outward_observation(
        &[1.0],
        &[Halfspace {
            a: vec![1.0],
            d: f64::NAN,
        }],
        &[0.0],
        &[1.0],
    )
    .is_none());
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn clip_outward_observation_preserves_directed_and_flushed_environment_refusals() {
    let _env = crate::tests::lock_env_shared();
    for flush in [false, true] {
        let environment = UnqualifiedEnvironment::enter(flush);
        assert!(dual_lower_outward_observation(&[1.0], &[], &[0.0], &[1.0]).is_none());
        drop(environment);
        ny_core::require_f64_interval_proof_environment().unwrap();
    }
}
