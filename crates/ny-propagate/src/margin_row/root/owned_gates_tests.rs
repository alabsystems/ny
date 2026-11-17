// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

trait Bits: TabElem {
    fn bits(self) -> u64;
}

impl Bits for f32 {
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl Bits for f64 {
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

fn check<T: Bits>(mode: RoundMode, reuse: bool) {
    // The last coefficient and bias share support block 63. The bias must
    // still be processed exactly once, including for an empty support mask.
    let n_in = 127;
    let naug = n_in + 1;
    let blk = support_blk(naug);
    let alpha = [0.0, 1.0, 1.0, 0.0, 0.3, 0.999_999];
    let slope = [0.0, 1.0, 0.5, 0.5, 0.5, 1e-6];
    let intercept = [0.0, 0.0, 0.7, 0.7, 0.7, 0.2];
    let values = [
        0.0,
        -0.0,
        -2.5,
        1.0 / 3.0,
        f64::from(f32::from_bits(1)),
        f64::from(f32::MIN_POSITIVE),
        1.0 + f64::from(f32::EPSILON),
        -1e20,
    ];
    let masks = [
        0,
        u64::MAX,
        1 | (1 << 63),
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        1 << 63,
    ];
    let xabs: Vec<f64> = (0..n_in)
        .map(|i| [0.0, 1e-6, 1.0, 37.5][i % 4])
        .chain(std::iter::once(1.0))
        .collect();
    for masked in [false, true] {
        let mask = masked.then_some(masks.as_slice());
        let mut m = Array2::from_shape_fn((alpha.len(), naug), |(j, i)| {
            T::store_center(values[(i + j) % values.len()]).0
        });
        let mut d = Array2::from_shape_fn((alpha.len(), naug), |(j, i)| {
            T::store_center([0.0, -0.375, 0.125][(i + j) % 3]).0
        });
        if masked {
            for j in 0..alpha.len() {
                for i in 0..n_in {
                    if !support_has(masks[j], i / blk) {
                        // Root writers preserve raw +0 outside support. Keep
                        // signed zero and subnormals in the live blocks only.
                        m[[j, i]] = T::ZERO;
                        d[[j, i]] = T::ZERO;
                    }
                }
            }
        }
        let expected = apply_gates(
            &m, &d, &alpha, &slope, &intercept, n_in, mode, mask, blk, &xabs,
        );
        let (mp, dp) = (m.as_ptr(), d.as_ptr());
        let actual = apply_gates_owned(
            m, d, &alpha, &slope, &intercept, n_in, mode, mask, blk, &xabs,
        );
        assert_eq!(actual.0.as_ptr() == mp, reuse);
        assert_eq!(actual.1.as_ptr() == dp, reuse);
        for (a, b) in [(&actual.0, &expected.0), (&actual.1, &expected.1)] {
            assert_eq!(a.dim(), b.dim());
            assert!(a.iter().zip(b).all(|(&x, &y)| x.bits() == y.bits()));
        }
        assert!(actual
            .2
            .iter()
            .zip(&expected.2)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

#[test]
fn owned_gates_reuse_allocations_with_identical_site_and_dust_bits() {
    check::<f32>(RoundMode::Outward, true);
    check::<f64>(RoundMode::Outward, true);
}

#[test]
fn owned_gates_preserve_parity_and_legacy_fallbacks() {
    check::<f64>(RoundMode::Parity, false);
    with_legacy_rounding_charge_for_test(|| {
        check::<f32>(RoundMode::Outward, false);
        check::<f64>(RoundMode::Outward, false);
    });
    with_borrowed_gates_for_test(|| check::<f32>(RoundMode::Outward, false));
    assert!(!borrowed_gates_forced());
}
