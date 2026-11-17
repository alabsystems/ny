// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational as Q;

fn q(v: f64) -> Q {
    Q::from_float(v).expect("finite exact dyadic")
}

fn fixture(w: f64) -> TwinNet {
    let conv = |input, weight| TwinOpSpec::Conv {
        input,
        weight: vec![weight],
        bias: vec![0.0],
        bias_err: vec![0.0],
        weight_rel_err: 0.0,
        kernel: (1, 1, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (1, 1, 1),
        oshape: (1, 1, 1),
    };
    TwinNet::compile(&TwinSpec {
        n_in: 1,
        ops: vec![
            conv(0, 2.0_f64.powi(126)),
            TwinOpSpec::Relu { input: 1 },
            conv(2, w),
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Flatten { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, 1),
            },
            TwinOpSpec::Relu { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![1.0, -1.0],
                bias: vec![0.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    })
    .unwrap()
}

#[test]
fn exceptional_weight_conversion_encloses_exact_root_hull_and_rejects_mutant() {
    for w in [
        2.0_f64.powi(-150),
        -2.0_f64.powi(-150),
        3.0 * 2.0_f64.powi(-151),
        -3.0 * 2.0_f64.powi(-151),
    ] {
        let net = fixture(w);
        let v = q(2.0_f64.powi(126)) * q(w);
        let endpoints = [v.clone(), &v * q(2.0)];
        let lo = endpoints.iter().min().unwrap();
        let hi = endpoints.iter().max().unwrap();
        for store in [false, true] {
            let run = || {
                RootGates::build_prec_store(
                    &net,
                    &[1.0],
                    &[2.0],
                    RoundMode::Outward,
                    None,
                    true,
                    store,
                )
                .unwrap()
            };
            let repaired = run();
            let old = without_correction(run);
            let fixed = &repaired.layers[1];
            let mutant = &old.layers[1];
            assert!(
                q(fixed.l[0]) <= *lo && q(fixed.u[0]) >= *hi,
                "exceptional conversion escaped: w={w:e}, store={store}"
            );
            assert!(
                q(mutant.l[0]) > *lo || q(mutant.u[0]) < *hi,
                "oracle must reject disabled correction: w={w:e}, store={store}"
            );
        }
    }
}

#[test]
fn normal_weight_and_source_conversions_keep_original_root_bits() {
    // Includes the smallest normal f32 weight; that case still has a valid
    // relative conversion model and must not pay the exceptional charge.
    for w in [0.0, -0.0, f64::from(f32::MIN_POSITIVE), -0.375, 0.5] {
        let net = fixture(w);
        for store in [false, true] {
            let run = || {
                RootGates::build_prec_store(
                    &net,
                    &[1.0],
                    &[2.0],
                    RoundMode::Outward,
                    None,
                    true,
                    store,
                )
                .unwrap()
            };
            let fixed = run();
            let old = without_correction(run);
            for (x, y) in fixed.layers.iter().zip(&old.layers) {
                for (a, b) in [
                    (&x.l, &y.l),
                    (&x.u, &y.u),
                    (&x.alpha, &y.alpha),
                    (&x.s, &y.s),
                    (&x.c, &y.c),
                ] {
                    assert!(a.iter().zip(b).all(|(u, v)| u.to_bits() == v.to_bits()));
                }
            }
        }
    }
}

#[test]
fn exceptional_f64_source_conversion_encloses_exact_root_hull_and_rejects_mutant() {
    let conv = |input, weight| TwinOpSpec::Conv {
        input,
        weight: vec![weight],
        bias: vec![0.0],
        bias_err: vec![0.0],
        weight_rel_err: 0.0,
        kernel: (1, 1, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (1, 1, 1),
        oshape: (1, 1, 1),
    };
    for (scale, sign) in [
        (2.0_f64.powi(-24), 1.0),
        (2.0_f64.powi(-24), -1.0),
        (3.0 * 2.0_f64.powi(-25), 1.0),
        (3.0 * 2.0_f64.powi(-25), -1.0),
    ] {
        let net = TwinNet::compile(&TwinSpec {
            n_in: 1,
            ops: vec![
                conv(0, 2.0_f64.powi(-126)),
                TwinOpSpec::ChannelAffine {
                    input: 1,
                    scale: vec![scale],
                    shift: vec![0.0],
                    scale_rel_err: 0.0,
                    shift_err: vec![0.0],
                    shape: (1, 1, 1),
                },
                TwinOpSpec::Relu { input: 2 },
                conv(3, sign * 2.0_f64.powi(126)),
                TwinOpSpec::Relu { input: 4 },
                TwinOpSpec::Flatten { input: 5 },
                TwinOpSpec::Gemm {
                    input: 6,
                    weight: vec![1.0],
                    bias: vec![0.0],
                    bias_err: vec![],
                    weight_rel_err: 0.0,
                    shape: (1, 1),
                },
                TwinOpSpec::Relu { input: 7 },
                TwinOpSpec::Gemm {
                    input: 8,
                    weight: vec![1.0, -1.0],
                    bias: vec![0.0; 2],
                    bias_err: vec![],
                    weight_rel_err: 0.0,
                    shape: (2, 1),
                },
            ],
        })
        .unwrap();
        let run = || {
            RootGates::build_prec_store(&net, &[1.0], &[2.0], RoundMode::Outward, None, true, false)
                .unwrap()
        };
        let repaired = run();
        let mutant = without_correction(run);
        assert!(repaired.layers[0].l[0] > 0.0 && mutant.layers[0].l[0] > 0.0);
        assert_eq!(repaired.layers[0].alpha[0], 1.0);
        let v = q(2.0_f64.powi(-126)) * q(scale) * q(sign * 2.0_f64.powi(126));
        let end = &v * q(2.0);
        let (lo, hi) = (v.clone().min(end.clone()), v.max(end));
        assert!(q(repaired.layers[1].l[0]) <= lo && q(repaired.layers[1].u[0]) >= hi);
        assert!(
            q(mutant.layers[1].l[0]) > lo || q(mutant.layers[1].u[0]) < hi,
            "input-cast oracle must reject the correction-disabled run"
        );
    }
}

#[test]
#[allow(unsafe_code, deprecated)]
fn correction_covers_exact_subnormal_source_functional() {
    let w = 2.0_f64.powi(-150);
    let net = fixture(w);
    let TwinOp::Conv(c) = &net.ops[2] else {
        unreachable!()
    };
    let h = 2.0_f64.powi(200);
    let run = || {
        for bits in [1, 0x007f_ffff, 0x8000_0001, 0x807f_ffff] {
            let v = f32::from_bits(bits);
            let m = Array2::from_shape_vec((1, 2), vec![v, 0.0]).unwrap();
            let d = Array2::from_shape_vec((1, 2), vec![-v, 0.0]).unwrap();
            let charged = correct(c, &m, &d, &Array2::zeros((1, 1)), &[h], 1, vec![0.0]);
            // q_effective=0 is an allowed execution. Its exact loss includes
            // BOTH signed tableau lanes over the entire [-h,h] input box.
            let exact = q(w) * q(ny_core::f32_to_f64_exact(v).abs()) * q(2.0) * q(h);
            assert!(charged[0] > 0.0 && q(charged[0]) >= exact);
        }
    };
    run();
    // Exceptional movement in D alone must be charged even with normal W
    // and a zero M lane. Exercise both the original input and bias columns.
    let normal_net = fixture(2.0_f64.powi(126));
    let TwinOp::Conv(normal) = &normal_net.ops[2] else {
        unreachable!()
    };
    for sign in [-1.0, 1.0] {
        for column in [0, 1] {
            let m = Array2::<f64>::zeros((1, 2));
            let mut d = m.clone();
            d[[0, column]] = sign * 2.0_f64.powi(-150);
            let charge = correct(normal, &m, &d, &Array2::zeros((1, 1)), &[4.0], 1, vec![0.0]);
            let exact = q(2.0_f64.powi(126))
                * q(2.0_f64.powi(-150))
                * q(if column == 0 { 4.0 } else { 1.0 });
            assert!(q(charge[0]) >= exact);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        // Dedicated workers only: never alter the caller or global pool.
        let saved = unsafe { std::arch::x86_64::_mm_getcsr() };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .start_handler(move |_| unsafe {
                std::arch::x86_64::_mm_setcsr(saved | (1 << 6) | (1 << 15))
            })
            .build()
            .unwrap();
        pool.install(|| {
            let v = std::hint::black_box(f32::from_bits(1));
            assert_eq!(
                std::hint::black_box(f64::from(v)),
                0.0,
                "ordinary widening must demonstrate the DAZ erasure mutant"
            );
            run();
        });
    }
}

#[test]
#[allow(unsafe_code, deprecated)]
fn f64_subnormal_conversions_refuse_overflowed_sources_without_zero_erasure() {
    let conv = |input, weight| TwinOpSpec::Conv {
        input,
        weight: vec![weight],
        bias: vec![0.0],
        bias_err: vec![0.0],
        weight_rel_err: 0.0,
        kernel: (1, 1, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (1, 1, 1),
        oshape: (1, 1, 1),
    };
    let make_net = |first_weight, scale, consumer_weight| {
        TwinNet::compile(&TwinSpec {
            n_in: 1,
            ops: vec![
                conv(0, first_weight),
                TwinOpSpec::ChannelAffine {
                    input: 1,
                    scale: vec![scale],
                    shift: vec![0.0],
                    scale_rel_err: 0.0,
                    shift_err: vec![0.0],
                    shape: (1, 1, 1),
                },
                TwinOpSpec::Relu { input: 2 },
                conv(3, consumer_weight),
                TwinOpSpec::Relu { input: 4 },
                TwinOpSpec::Flatten { input: 5 },
                TwinOpSpec::Gemm {
                    input: 6,
                    weight: vec![1.0],
                    bias: vec![0.0],
                    bias_err: vec![],
                    weight_rel_err: 0.0,
                    shape: (1, 1),
                },
                TwinOpSpec::Relu { input: 7 },
                TwinOpSpec::Gemm {
                    input: 8,
                    weight: vec![1.0, -1.0],
                    bias: vec![0.0; 2],
                    bias_err: vec![],
                    weight_rel_err: 0.0,
                    shape: (2, 1),
                },
            ],
        })
        .unwrap()
    };
    let dead = make_net(-1.0, 1.0, 1.0);
    for sign_bit in [0, 1_u64 << 63] {
        // Bit construction survives DAZ/FTZ; this is exactly +/-2^-1023.
        let w = f64::from_bits(sign_bit | (1_u64 << 51));
        let net = make_net(1.0, 2.0_f64.powi(1021), w);
        let TwinOp::Conv(consumer) = &net.ops[3] else {
            unreachable!()
        };
        assert_eq!(consumer.wmat[0].to_bits(), w.to_bits());
        let run = || {
            RootGates::build_prec_store(&net, &[1.0], &[2.0], RoundMode::Outward, None, true, false)
        };
        let check = || {
            assert!(conversion_gap(std::hint::black_box(w)).is_infinite());
            // Source subnormals get a finite normal gap: ordinary rounding-floor
            // D entries must not force refusal when the consumer is normal.
            let normal = fixture(1.0);
            let TwinOp::Conv(c) = &normal.ops[2] else {
                unreachable!()
            };
            let mut source_charges = Vec::new();
            for lane in [false, true] {
                for column in [0, 1] {
                    let mut m = Array2::<f64>::zeros((1, 2));
                    let mut d = m.clone();
                    if lane {
                        d[[0, column]] = w;
                    } else {
                        m[[0, column]] = w;
                    }
                    let charge =
                        correct(c, &m, &d, &Array2::zeros((1, 1)), &[2.0], 1, vec![0.0])[0];
                    assert!(charge.is_finite());
                    source_charges.push((charge, column));
                }
            }
            // Ordinary dead ReLUs can retain tiny f64 rounding-floor D.
            // A following normal Conv must still return finite root bounds.
            let dead_root = RootGates::build_prec_store(
                &dead,
                &[1.0],
                &[2.0],
                RoundMode::Outward,
                None,
                true,
                false,
            )
            .unwrap();
            assert!(dead_root.layers[0].u[0] < 0.0);
            assert!(dead_root.layers[1].l[0].is_finite() && dead_root.layers[1].u[0].is_finite());
            assert!(dead_root.layers[1].l[0] <= 0.0 && dead_root.layers[1].u[0] >= 0.0);
            assert!(matches!(run(), Err(NyError::NumericalInstability(_))));
            // Fail-closed must be load-bearing: the uncorrected root returns
            // finite but wrong bounds for a nonzero exact preactivation.
            (without_correction(run).unwrap(), source_charges)
        };
        let mut mutants = vec![check()];
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .start_handler(|_| unsafe { _mm_setcsr(_mm_getcsr() | 0x8040) })
                .build()
                .unwrap();
            mutants.push(pool.install(|| {
                assert_eq!(unsafe { _mm_getcsr() } & 0x8040, 0x8040);
                assert_eq!(std::hint::black_box(w), 0.0);
                check()
            }));
        }
        // Exact comparison runs outside the altered floating-point context.
        let (lo, hi) = if sign_bit == 0 {
            (0.25, 0.5)
        } else {
            (-0.5, -0.25)
        };
        for (mutant, source_charges) in mutants {
            for (charge, column) in source_charges {
                // |source|=2^-1023; H=2 for an input and H=1 for bias.
                let exact = q(f64::MIN_POSITIVE) / q(if column == 0 { 1.0 } else { 2.0 });
                assert!(q(charge) >= exact);
            }
            let layer = &mutant.layers[1];
            assert!(layer.l[0].is_finite() && layer.u[0].is_finite());
            assert!(q(layer.l[0]) > q(lo) || q(layer.u[0]) < q(hi));
        }
    }
}
