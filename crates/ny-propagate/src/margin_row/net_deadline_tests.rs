// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cancellation-contract and exact-bit regression tests for optional kernels.

use super::{
    axpy, axpy_with_deadline, check_deadline, compile_conv, compile_conv_transpose,
    conv_apply_backward_with_deadline, conv_backward_blocked, conv_backward_icgrain,
    envelope_error_conv_without_deadline, error_conv_with_deadline, error_conv_without_deadline,
    nominal_product_dust, underflow_product_up, ConvOp,
};
use crate::margin_row::rounding::{certify_up, gamma_n};
use ndarray::Array2;
use std::time::{Duration, Instant};

fn bits(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

fn values(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            match i % 13 {
                0 => 0.0,
                1 => -0.0,
                2 => f64::from_bits(1),
                3 => -f64::from_bits(1),
                _ => ((state >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0,
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn conv(
    kernel: (usize, usize, usize, usize),
    stride: (usize, usize),
    pads: (usize, usize, usize, usize),
    ishape: (usize, usize, usize),
    oshape: (usize, usize, usize),
    transposed: bool,
) -> ConvOp {
    let (co, ci, kh, kw) = kernel;
    let weights = values(co * ci * kh * kw, 0xDEAD_11AE);
    let bias = vec![0.0; co];
    // Nonzero fold metadata is deliberately retained. A kernel computes the
    // stored contraction; the engine still owns ALL existing fold charges.
    let bias_err = vec![f64::from_bits(1); co];
    let compile = if transposed {
        compile_conv_transpose
    } else {
        compile_conv
    };
    compile(
        0,
        &weights,
        &bias,
        &bias_err,
        f64::EPSILON,
        kernel,
        stride,
        pads,
        ishape,
        oshape,
    )
}

#[test]
fn expired_deadline_refuses_without_touching_scratch() {
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(check_deadline(expired).unwrap_err().is_deadline_exceeded());
    let mut acc = vec![1.0, -0.0, 7.0];
    let before = bits(acc.clone());
    assert!(
        axpy_with_deadline(&mut acc, -2.0, &[3.0, 0.0, 1.0], expired)
            .unwrap_err()
            .is_deadline_exceeded()
    );
    assert_eq!(bits(acc), before);

    let c = conv(
        (3, 2, 3, 3),
        (1, 1),
        (1, 1, 1, 1),
        (2, 4, 5),
        (3, 4, 5),
        false,
    );
    let src = Array2::from_shape_vec((60, 5), values(300, 123)).unwrap();
    let mut dst =
        Array2::from_shape_fn((40, 5), |(i, j)| if (i + j) % 2 == 0 { -0.0 } else { 17.0 });
    let before = bits(dst.iter().copied());
    assert!(
        conv_apply_backward_with_deadline(&c, &src, &mut dst, false, expired)
            .unwrap_err()
            .is_deadline_exceeded()
    );
    assert_eq!(bits(dst.iter().copied()), before);
}

#[test]
fn live_axpy_is_bit_identical_across_poll_tiles() {
    for n in [0, 1, 64, 255, 256, 257, 1025] {
        for w in [-2.0, -0.0, 0.0, 0.125, 1.0 + f64::EPSILON] {
            let src = values(n, 54);
            let mut want = values(n, 71);
            let mut got = want.clone();
            axpy(&mut want, w, &src);
            axpy_with_deadline(&mut got, w, &src, Instant::now() + Duration::from_secs(30))
                .unwrap();
            assert_eq!(bits(got), bits(want), "length={n} weight={w}");
        }
    }
}

#[test]
fn live_deadline_conv_matches_both_cpu_grains_bit_for_bit() {
    let cases = [
        conv(
            (5, 3, 3, 3),
            (1, 1),
            (1, 1, 1, 1),
            (3, 4, 5),
            (5, 4, 5),
            false,
        ),
        conv(
            (7, 5, 2, 3),
            (2, 1),
            (0, 1, 1, 0),
            (5, 7, 8),
            (7, 4, 7),
            false,
        ),
        conv(
            (5, 3, 3, 2),
            (2, 1),
            (1, 0, 1, 1),
            (3, 4, 5),
            (5, 7, 5),
            true,
        ),
    ];
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for c in &cases {
                let n_src = c.oshape.0 * c.oshape.1 * c.oshape.2;
                let n_dst = c.ishape.0 * c.ishape.1 * c.ishape.2;
                for r in [1, 4, 16, 64, 257] {
                    let src =
                        Array2::from_shape_vec((n_src, r), values(n_src * r, 0xB175_0ADC)).unwrap();
                    for abs_w in [false, true] {
                        let mut blocked = Array2::zeros((n_dst, r));
                        let mut rowgrain = Array2::zeros((n_dst, r));
                        let mut got = Array2::from_elem((n_dst, r), f64::NAN);
                        conv_backward_blocked(c, &src, &mut blocked, abs_w);
                        conv_backward_icgrain(c, &src, &mut rowgrain, abs_w);
                        conv_apply_backward_with_deadline(
                            c,
                            &src,
                            &mut got,
                            abs_w,
                            Instant::now() + Duration::from_secs(30),
                        )
                        .unwrap();
                        let expected = bits(blocked.iter().copied());
                        assert_eq!(bits(rowgrain.iter().copied()), expected);
                        assert_eq!(
                            bits(got.iter().copied()),
                            expected,
                            "threads={threads} rows={r} abs={abs_w} transpose={}",
                            c.transposed
                        );
                    }
                }
            }
        });
    }
}

#[test]
fn live_deadline_helpers_refuse_bad_shapes() {
    let live = Instant::now() + Duration::from_secs(30);
    let mut acc = [11.0];
    assert!(axpy_with_deadline(&mut acc, 1.0, &[], live).is_err());
    assert_eq!(acc, [11.0]);
    let c = conv(
        (3, 2, 1, 1),
        (1, 1),
        (0, 0, 0, 0),
        (2, 2, 2),
        (3, 2, 2),
        false,
    );
    let src = Array2::zeros((12, 4));
    let mut dst = Array2::from_elem((7, 4), 11.0);
    assert!(conv_apply_backward_with_deadline(&c, &src, &mut dst, false, live).is_err());
    assert!(dst.iter().all(|x| *x == 11.0));

    let mut nonstandard_dst = Array2::from_elem((4, 8), 11.0).reversed_axes();
    assert_eq!(nonstandard_dst.dim(), (8, 4));
    assert!(
        conv_apply_backward_with_deadline(&c, &src, &mut nonstandard_dst, false, live).is_err()
    );
    assert!(nonstandard_dst.iter().all(|x| *x == 11.0));

    let nonstandard_src = Array2::zeros((4, 12)).reversed_axes();
    let mut correct_dst = Array2::from_elem((8, 4), 11.0);
    assert!(
        conv_apply_backward_with_deadline(&c, &nonstandard_src, &mut correct_dst, false, live)
            .is_err()
    );
    assert!(correct_dst.iter().all(|x| *x == 11.0));
}

#[test]
fn optional_product_helpers_preserve_underflow_budget_and_normal_bits() {
    let eta = f64::from_bits(1);
    for sign in [-1.0, 1.0] {
        assert_eq!(nominal_product_dust(sign * eta, 0.375), eta);
        assert_eq!(nominal_product_dust(sign * eta, 1.0), eta);
    }
    assert_eq!(underflow_product_up(eta, 0.375), eta);
    assert_eq!(underflow_product_up(eta, 1.0), 2.0 * eta);
    for a in [0.0, 0.125, 1.0, 2.0] {
        for b in [0.0, 0.125, 1.0, 2.0] {
            assert_eq!(underflow_product_up(a, b).to_bits(), (a * b).to_bits());
            assert_eq!(nominal_product_dust(a, -b), 0.0);
        }
    }
    assert_eq!(nominal_product_dust(0.0, eta), 0.0);
    assert_eq!(nominal_product_dust(eta, -0.0), 0.0);
    assert_eq!(nominal_product_dust(f64::MIN_POSITIVE, 1.0), 0.0);
}

fn unit_spatial_conv(weights: &[f64], co: usize, ci: usize) -> ConvOp {
    compile_conv(
        0,
        weights,
        &vec![0.0; co],
        &vec![0.0; co],
        0.0,
        (co, ci, 1, 1),
        (1, 1),
        (0, 0, 0, 0),
        (ci, 1, 1),
        (co, 1, 1),
    )
}

/// Analytic, exactly representable post-amplification oracle: four products
/// eta*(-3/8) have exact sum -3*eta/2, while all four nominal products round
/// to zero. Multiplication by 2^256 makes the true -1.5*2^-818 representable
/// as a NORMAL binary64 number. The old one-unit final widening misses it.
#[test]
fn optional_error_conv_covers_lost_signed_products_before_amplification() {
    let eta = f64::from_bits(1);
    let first = unit_spatial_conv(&[eta; 4], 4, 1);
    let nominal = Array2::from_elem((4, 1), -0.375);
    let combined = Array2::from_elem((4, 1), gamma_n(6) * 0.375);
    let live = Instant::now() + Duration::from_secs(30);
    let mut l1 = Array2::zeros((1, 1));
    let mut repaired = Array2::zeros((1, 1));
    let mut legacy = Array2::zeros((1, 1));
    conv_apply_backward_with_deadline(&first, &nominal, &mut l1, false, live).unwrap();
    error_conv_with_deadline(&first, &combined, &nominal, &mut repaired, live).unwrap();
    let mut untimed = Array2::zeros((1, 1));
    error_conv_without_deadline(&first, &combined, &nominal, &mut untimed).unwrap();
    assert_eq!(
        bits(untimed.iter().copied()),
        bits(repaired.iter().copied())
    );
    conv_backward_blocked(&first, &combined, &mut legacy, true);
    assert_eq!(l1[[0, 0]], 0.0);
    assert_eq!(legacy[[0, 0]], 0.0);
    assert!(repaired[[0, 0]] >= 2.0 * eta);

    let amplifier = f64::from_bits((1023u64 + 256) << 52);
    let exact_scaled = -1.5 * f64::from_bits((1023u64 - 818) << 52);
    assert!(exact_scaled.is_normal() && exact_scaled < 0.0);
    let old_error = certify_up(legacy[[0, 0]], gamma_n(first.k_bwd + 8)) * amplifier;
    let new_error = certify_up(repaired[[0, 0]], gamma_n(first.k_bwd + 8)) * amplifier;
    assert!(
        -old_error > exact_scaled,
        "the old budget misses the exact row"
    );
    assert!(
        -new_error <= exact_scaled,
        "the repaired budget contains it"
    );

    // The repaired absolute budget itself survives the next native error
    // contraction before its existing post-contraction widening is applied.
    let second = unit_spatial_conv(&[amplifier], 1, 1);
    let mut e2 = Array2::zeros((1, 1));
    error_conv_with_deadline(&second, &repaired, &l1, &mut e2, live).unwrap();
    assert!(-e2[[0, 0]] <= exact_scaled);
}

#[test]
fn optional_error_conv_preserves_multiple_underflowed_error_products() {
    let eta = f64::from_bits(1);
    let c = unit_spatial_conv(&[0.375; 4], 4, 1);
    let nominal = Array2::zeros((4, 1));
    let combined = Array2::from_elem((4, 1), eta);
    let mut repaired = Array2::zeros((1, 1));
    let mut old = Array2::zeros((1, 1));
    error_conv_with_deadline(
        &c,
        &combined,
        &nominal,
        &mut repaired,
        Instant::now() + Duration::from_secs(30),
    )
    .unwrap();
    conv_backward_blocked(&c, &combined, &mut old, true);
    assert_eq!(old[[0, 0]], 0.0);
    assert_eq!(repaired[[0, 0]], 4.0 * eta);
    let amplifier = f64::from_bits((1023u64 + 256) << 52);
    let exact_scaled = 1.5 * f64::from_bits((1023u64 - 818) << 52);
    assert!(repaired[[0, 0]] * amplifier >= exact_scaled);
    assert!(certify_up(old[[0, 0]], gamma_n(c.k_bwd + 8)) * amplifier < exact_scaled);
}

#[test]
fn optional_error_conv_is_bit_identical_when_no_product_underflows() {
    let c = unit_spatial_conv(&[-0.5, 0.125, 0.0, 1.0, -0.0, 2.0], 3, 2);
    for r in [1, 4, 64, 257, 512] {
        let nominal =
            Array2::from_shape_fn((3, r), |(i, j)| if (i + j) % 3 == 0 { 0.0 } else { -0.375 });
        let combined =
            Array2::from_shape_fn((3, r), |(i, j)| if (i + j) % 4 == 0 { 0.0 } else { 0.125 });
        let mut ordinary = Array2::zeros((2, r));
        let mut repaired = Array2::zeros((2, r));
        let mut untimed = Array2::zeros((2, r));
        conv_backward_blocked(&c, &combined, &mut ordinary, true);
        error_conv_with_deadline(
            &c,
            &combined,
            &nominal,
            &mut repaired,
            Instant::now() + Duration::from_secs(30),
        )
        .unwrap();
        assert_eq!(
            bits(repaired.iter().copied()),
            bits(ordinary.iter().copied())
        );
        error_conv_without_deadline(&c, &combined, &nominal, &mut untimed).unwrap();
        assert_eq!(
            bits(untimed.iter().copied()),
            bits(ordinary.iter().copied())
        );
    }
}

#[test]
fn optional_error_conv_refuses_expiry_and_misaligned_nominal_source() {
    let c = unit_spatial_conv(&[1.0], 1, 1);
    let combined = Array2::from_elem((1, 4), 0.125);
    let nominal = Array2::from_elem((1, 3), 0.5);
    let mut dst = Array2::from_elem((1, 4), 11.0);
    assert!(error_conv_with_deadline(
        &c,
        &combined,
        &nominal,
        &mut dst,
        Instant::now() + Duration::from_secs(30),
    )
    .is_err());
    assert!(dst.iter().all(|v| *v == 11.0));
    assert!(error_conv_with_deadline(
        &c,
        &combined,
        &combined,
        &mut dst,
        Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
    )
    .unwrap_err()
    .is_deadline_exceeded());
    assert!(dst.iter().all(|v| *v == 11.0));
}

#[test]
fn invalid_error_products_propagate_an_infinite_budget_without_panicking() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        for other in [0.0, 1.0, f64::from_bits(1)] {
            assert_eq!(underflow_product_up(bad, other), f64::INFINITY);
            assert_eq!(underflow_product_up(other, bad), f64::INFINITY);
        }
    }
    assert_eq!(underflow_product_up(1e200, 1e200), f64::INFINITY);
    // Returning infinity leaves the existing pass firewall responsible for
    // refusal, rather than panicking in the middle of a Rayon error update.
    let c = unit_spatial_conv(&[1.0], 1, 1);
    let invalid = Array2::from_elem((1, 1), f64::NAN);
    let nominal = Array2::zeros((1, 1));
    let mut error = Array2::zeros((1, 1));
    error_conv_without_deadline(&c, &invalid, &nominal, &mut error).unwrap();
    assert_eq!(error[[0, 0]], f64::INFINITY);
}

/// Isolate the grouped term's non-monotone nominal-dust indicator: the max
/// nominal column is normal while a smaller column underflows. Grouped
/// incoming error is identical to the per-column incoming error here; the
/// caller's independent gamma charges are applied after each contraction.
#[test]
fn envelope_dust_dominates_smaller_underflowing_columns_before_amplification() {
    let eta = f64::from_bits(1);
    let c = unit_spatial_conv(&[0.375; 4], 4, 1);
    let nominal = Array2::from_shape_fn((4, 2), |(_, col)| if col == 0 { -eta } else { -1.0 });
    let combined = Array2::zeros((4, 2));
    let group_error = Array2::zeros((4, 1));
    let lambda = Array2::from_elem((4, 1), 1.0);
    let mut per_col = Array2::zeros((1, 2));
    let mut envelope = Array2::zeros((1, 1));
    let mut old_at_max = Array2::zeros((1, 1));
    error_conv_without_deadline(&c, &combined, &nominal, &mut per_col).unwrap();
    envelope_error_conv_without_deadline(&c, &group_error, &lambda, &mut envelope).unwrap();
    // This is the invalid majorant: inspecting underflow only at lambda
    // misses the actual small column even though lambda >= both magnitudes.
    error_conv_without_deadline(&c, &group_error, &lambda, &mut old_at_max).unwrap();
    assert_eq!(old_at_max[[0, 0]], 0.0);
    assert!(per_col[[0, 0]] > 0.0);
    assert!(envelope[[0, 0]] >= per_col[[0, 0]]);
    assert!(envelope[[0, 0]] >= per_col[[0, 1]]);

    let amplifier = f64::from_bits((1023u64 + 256) << 52);
    let exact_scaled = -1.5 * f64::from_bits((1023u64 - 818) << 52);
    let old_certified = certify_up(old_at_max[[0, 0]], gamma_n(c.k_bwd + 8));
    assert!(-old_certified * amplifier > exact_scaled);
    assert!(-envelope[[0, 0]] * amplifier <= exact_scaled);

    let second = unit_spatial_conv(&[amplifier], 1, 1);
    let nominal_after_first = Array2::from_shape_vec((1, 2), vec![0.0, -1.5]).unwrap();
    let mut per_col_second = Array2::zeros((1, 2));
    let mut env_second = Array2::zeros((1, 1));
    error_conv_without_deadline(&second, &per_col, &nominal_after_first, &mut per_col_second)
        .unwrap();
    envelope_error_conv_without_deadline(
        &second,
        &envelope,
        &Array2::from_elem((1, 1), 1.5),
        &mut env_second,
    )
    .unwrap();
    assert!(env_second[[0, 0]] >= per_col_second[[0, 0]]);
    assert!(env_second[[0, 0]] >= per_col_second[[0, 1]]);
    assert!(-env_second[[0, 0]] <= exact_scaled);
}

#[test]
fn envelope_zero_groups_need_no_nominal_dust_and_invalid_groups_refuse() {
    let c = unit_spatial_conv(&[-0.5, 0.125, 2.0], 3, 1);
    let errors = Array2::from_elem((3, 1), 0.125);
    let zero = Array2::zeros((3, 1));
    let mut per_col = Array2::zeros((1, 1));
    let mut envelope = Array2::zeros((1, 1));
    error_conv_without_deadline(&c, &errors, &zero, &mut per_col).unwrap();
    envelope_error_conv_without_deadline(&c, &errors, &zero, &mut envelope).unwrap();
    assert_eq!(
        bits(envelope.iter().copied()),
        bits(per_col.iter().copied())
    );
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        let invalid = Array2::from_elem((3, 1), bad);
        let mut untouched = Array2::from_elem((1, 1), 11.0);
        assert!(
            envelope_error_conv_without_deadline(&c, &errors, &invalid, &mut untouched).is_err()
        );
        assert_eq!(untouched[[0, 0]], 11.0);
    }
}
