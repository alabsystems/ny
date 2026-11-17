// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};

fn q(v: f64) -> Q {
    Q::from_float(v).expect("finite oracle operand")
}

// Check the emitted coefficients, not a second implementation of their cast.
// Injective gathers make the exact functional norm separable by input column.
fn exact_conversion<T: TabElem>(c: &ConvOp, direct: &Array2<T>, xabs: &[f64]) -> Vec<Q> {
    let p = c.oshape.1 * c.oshape.2;
    (0..direct.nrows())
        .map(|j| {
            let mut exact = Q::zero();
            for t in 0..c.k_fwd {
                let gi = c.gather[t * p + j % p];
                if gi != usize::MAX {
                    exact += (q(c.wmat[j / p * c.k_fwd + t]) - q(direct[[j, gi]].wide())).abs()
                        * q(next_up(xabs[gi]));
                }
            }
            exact
        })
        .collect()
}

fn fixture(transposed: bool, padded: bool) -> ConvOp {
    let (ishape, oshape, kernel, stride, pads) = if padded {
        ((1, 1, 1), (2, 3, 3), (2, 1, 1, 1), (1, 1), (1, 1, 1, 1))
    } else if transposed {
        ((2, 3, 3), (2, 6, 6), (2, 2, 3, 3), (2, 2), (1, 1, 1, 1))
    } else {
        ((2, 9, 7), (3, 3, 4), (3, 2, 4, 2), (3, 2), (2, 1, 1, 0))
    };
    let values = [-1.25, 0.0, -0.0, 1.0 / 3.0, 1e-50, -1e-50, 2.5];
    let weight = (0..kernel.0 * kernel.1 * kernel.2 * kernel.3)
        .map(|i| values[i % values.len()])
        .collect();
    let first = if transposed {
        TwinOpSpec::ConvTranspose {
            input: 0,
            weight,
            bias: vec![0.1; kernel.0],
            bias_err: vec![1e-18; kernel.0],
            weight_rel_err: 1e-15,
            kernel,
            stride,
            pads,
            ishape,
            oshape,
            out_pad: (1, 1),
        }
    } else {
        TwinOpSpec::Conv {
            input: 0,
            weight,
            bias: vec![0.1; kernel.0],
            bias_err: vec![1e-18; kernel.0],
            weight_rel_err: 1e-15,
            kernel,
            stride,
            pads,
            ishape,
            oshape,
        }
    };
    let n_out = oshape.0 * oshape.1 * oshape.2;
    let spec = TwinSpec {
        n_in: ishape.0 * ishape.1 * ishape.2,
        ops: vec![
            first,
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![1.0; n_out],
                bias: vec![0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, n_out),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![1.0, -1.0],
                bias: vec![0.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    };
    let net = TwinNet::compile(&spec).expect("valid fixture");
    match net.ops.into_iter().next().unwrap() {
        TwinOp::Conv(c) => *c,
        _ => unreachable!(),
    }
}

fn parity<T: TabElem>(c: &ConvOp, xabs: &[f64], masked: bool) {
    let n = xabs.len();
    let na = n + 1;
    let out = c.oshape.0 * c.oshape.1 * c.oshape.2;
    let sum = next_up(xabs.iter().sum::<f64>() + 1.0);
    let plan = plan(c, n, out, xabs, sum).expect("eligible");
    let mut mi = Array2::from_elem((n, na), T::ZERO);
    for i in 0..n {
        mi[[i, i]] = T::ONE;
    }
    let di = Array2::from_elem((n, na), T::ZERO);
    let blk = support_blk(na);
    let si = support_input(n, blk);
    let so = support_conv(c, &si, out, support_bias_bit(n, blk));
    let mask = masked.then_some((si.as_slice(), so.as_slice(), SUPPORT_BLOCKS));
    let mut dense = Array2::from_elem((out, na), T::ZERO);
    let mut direct = Array2::from_elem((out, na), T::ZERO);
    T::conv_forward_masked(c, &mi, &mut dense, false, true, mask);
    center(c, &plan, &mut direct);
    assert!(dense
        .iter()
        .zip(&direct)
        .all(|(a, b)| a.wide().to_bits() == b.wide().to_bits()));
    T::conv_forward_masked(c, &di, &mut dense, true, true, mask);
    assert!(dense.iter().all(|v| v.wide().to_bits() == 0));
    let old = conv_f32_gerr(
        c,
        &mi,
        &di,
        &vec![0.0; n],
        xabs,
        sum,
        n,
        masked.then_some(si.as_slice()),
        blk,
        T::F32_STORE,
    );
    let new = error(c, &plan, xabs);
    assert_eq!(new.len(), out);
    for (j, exact) in exact_conversion(c, &direct, xabs).iter().enumerate() {
        assert!(*exact <= q(new[j]), "row {j}: exact conversion escapes");
        assert!(
            new[j] <= old[j],
            "finite fixture must tighten the local charge"
        );
    }
}

#[test]
fn identity_conv_local_bits_and_exact_error_cover_padding_transpose_and_masks() {
    for c in [
        fixture(false, false),
        fixture(true, false),
        fixture(false, true),
    ] {
        let n = c.ishape.0 * c.ishape.1 * c.ishape.2;
        for x in [
            vec![0.0; n],
            (0..n).map(|i| [0.0, 0.1, 0.5, 3.0][i % 4]).collect(),
        ] {
            for masked in [false, true] {
                parity::<f32>(&c, &x, masked);
                parity::<f64>(&c, &x, masked);
            }
        }
    }
}

// A one-output, one-spatial-position Conv with one tap per input channel.
fn taps(weights: &[f64]) -> ConvOp {
    let mut c = fixture(false, true);
    c.kernel = (1, weights.len(), 1, 1);
    c.ishape = (weights.len(), 1, 1);
    c.oshape = (1, 1, 1);
    c.k_fwd = weights.len();
    c.wmat = weights.to_vec();
    c.gather = (0..weights.len()).collect();
    c.bias = vec![0.0];
    c.bias_err = vec![0.0];
    c
}

#[test]
fn identity_conv_exact_error_covers_cast_zero_underflow_and_scalar_reduction() {
    let cases = [
        (vec![1.0 + 2.0_f64.powi(-24); 10], {
            let mut x = vec![2.0_f64.powi(-30); 10];
            x[0] = 2.0_f64.powi(24);
            x
        }),
        (
            vec![
                2.0_f64.powi(-150),
                -2.0_f64.powi(-150),
                1e-50,
                f64::from_bits(1),
            ],
            vec![1.0; 4],
        ),
        (vec![2.0_f64.powi(-150)], vec![2.0_f64.powi(-1000)]),
    ];
    for (weights, xabs) in cases {
        let c = taps(&weights);
        let p = plan(&c, weights.len(), 1, &xabs, next_up(xabs.iter().sum())).unwrap();
        let mut direct = Array2::<f32>::zeros((1, weights.len() + 1));
        center(&c, &p, &mut direct);
        let exact = exact_conversion(&c, &direct, &xabs);
        let actual = error(&c, &p, &xabs);
        assert!(actual[0] > 0.0, "nonzero conversion cannot disappear");
        assert!(exact[0] <= q(actual[0]), "exact conversion norm escapes");
    }
    // One final outward step cannot recover nine small terms lost during
    // nearest-rounded summation; the exact oracle must detect that shortcut.
    let mut naive = 1.0_f64;
    let small = 2.0_f64.powi(-54);
    let mut exact = q(1.0);
    for _ in 0..9 {
        naive += small;
        exact += q(small);
    }
    assert!(
        q(next_up(naive)) < exact,
        "oracle must detect a lost-term reduction"
    );
}

#[test]
fn identity_conv_exact_weights_keep_zero_vector_and_bias_store_charge() {
    let mut c = taps(&[1.0, -2.0, 0.0, -0.0]);
    let xabs = vec![1.0; 4];
    let p = plan(&c, 4, 1, &xabs, 5.0).unwrap();
    for bias in [0.0, 0.1] {
        c.bias[0] = bias;
        let zero = error(&c, &p, &xabs);
        assert_eq!(
            zero.len(),
            1,
            "empty vector would disable the bias-store charge"
        );
        assert_eq!(zero[0].to_bits(), 0);
        let mut m64 = Array2::<f64>::zeros((1, 5));
        let mut d64 = m64.clone();
        let mut m32 = Array2::<f32>::zeros((1, 5));
        let mut d32 = m32.clone();
        let mut g64 = zero.clone();
        let mut g32 = zero;
        conv_bias_fold::<f64>(
            &mut m64,
            &mut d64,
            &c.bias,
            &c.bias_err,
            1,
            4,
            RoundMode::Outward,
            &mut g64,
        );
        conv_bias_fold::<f32>(
            &mut m32,
            &mut d32,
            &c.bias,
            &c.bias_err,
            1,
            4,
            RoundMode::Outward,
            &mut g32,
        );
        let need = (q(f64::from(m32[[0, 4]])) - q(m64[[0, 4]])).abs()
            + (q(f64::from(d32[[0, 4]])) - q(d64[[0, 4]])).abs();
        assert!(g32[0] > 0.0 && need <= q(g32[0]));
        assert_eq!(g64[0], 0.0);
    }
}

#[test]
fn identity_conv_error_retains_overflow() {
    let c = taps(&[1e30 + 1e23]);
    // A finite magnitude sum is still eligible; its conversion product need
    // not be finite. The error helper must retain infinity in that case.
    let xabs = vec![1e300];
    let p = plan(&c, 1, 1, &xabs, next_up(xabs[0])).unwrap();
    assert_eq!(error(&c, &p, &xabs), vec![f64::INFINITY]);
}

#[test]
fn identity_conv_plan_refuses_ambiguous_gathers_and_unsupported_numbers() {
    let c = fixture(false, false);
    let n = c.ishape.0 * c.ishape.1 * c.ishape.2;
    let out = c.oshape.0 * c.oshape.1 * c.oshape.2;
    let x = vec![1.0; n];
    let admitted = |v: &ConvOp| plan(v, n, out, &x, n as f64 + 2.0).is_some();
    assert!(admitted(&c));
    for weight in [
        f64::from(f32::from_bits(1)),
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ] {
        let mut bad = c.clone();
        bad.wmat[0] = weight;
        assert!(!admitted(&bad));
    }
    for value in [-1.0, f64::from_bits(1), f64::INFINITY, f64::NAN] {
        let mut bad = x.clone();
        bad[0] = value;
        assert!(plan(&c, n, out, &bad, n as f64 + 2.0).is_none());
    }
    for sum in [-1.0, f64::INFINITY, f64::NAN] {
        assert!(plan(&c, n, out, &x, sum).is_none());
    }
    let mut bad = c.clone();
    bad.gather[0] = n;
    assert!(!admitted(&bad));
    let mut bad = c.clone();
    let p = c.oshape.1 * c.oshape.2;
    bad.gather[0] = 0;
    bad.gather[p] = 0;
    assert!(!admitted(&bad));
    let mut bad = c.clone();
    bad.gather.pop();
    assert!(!admitted(&bad));
    let mut bad = c.clone();
    bad.wmat.pop();
    assert!(!admitted(&bad));
    let mut bad = c.clone();
    bad.ishape.0 = usize::MAX;
    assert!(!admitted(&bad));
    let mut bad = c.clone();
    bad.oshape.1 = usize::MAX;
    assert!(!admitted(&bad));
    let mut bad = c.clone();
    bad.input = 1;
    assert!(!admitted(&bad));
    let mut bad = c.clone();
    bad.k_fwd += 1;
    assert!(!admitted(&bad));
}

#[test]
#[allow(unsafe_code, deprecated)]
fn identity_conv_refuses_changed_caller_and_worker_fp_controls() {
    assert!(qualified_thread());
    #[cfg(target_arch = "x86_64")]
    std::thread::spawn(|| {
        struct Restore(u32);
        impl Drop for Restore {
            fn drop(&mut self) {
                unsafe { std::arch::x86_64::_mm_setcsr(self.0) };
            }
        }
        let c = fixture(false, false);
        let n = c.ishape.0 * c.ishape.1 * c.ishape.2;
        let out = c.oshape.0 * c.oshape.1 * c.oshape.2;
        let x = vec![1.0; n];
        let saved = unsafe { std::arch::x86_64::_mm_getcsr() };
        let _restore = Restore(saved);
        for controls in [(1 << 13), (1 << 6) | (1 << 15)] {
            unsafe { std::arch::x86_64::_mm_setcsr(saved | controls) };
            assert!(!qualified_thread());
            assert!(plan(&c, n, out, &x, n as f64 + 2.0).is_none());
            unsafe { std::arch::x86_64::_mm_setcsr(saved) };
            // Dedicated pool; altered worker state never reaches global Rayon.
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .start_handler(move |_| unsafe { std::arch::x86_64::_mm_setcsr(saved | controls) })
                .build()
                .unwrap();
            assert!(pool
                .install(|| {
                    // Restore only the calling worker. Its peer remains
                    // altered, so refusal now requires the broadcast guard.
                    unsafe { std::arch::x86_64::_mm_setcsr(saved) };
                    assert!(qualified_thread());
                    plan(&c, n, out, &x, n as f64 + 2.0)
                })
                .is_none());
        }
        assert!(qualified_thread());
    })
    .join()
    .expect("isolated FP control test");
}
