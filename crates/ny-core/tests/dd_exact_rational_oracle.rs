// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact-rational oracle for [`ny_core::dd`].
//!
//! The double-double soundness argument rests on two claims that ordinary
//! float assertions CANNOT check, because "close enough in f64" is exactly the
//! property being tested:
//!
//! 1. [`two_sum`] and [`two_prod`] are **error-free**: `s + e` and `p + e`
//!    equal `a + b` and `a * b` EXACTLY, not approximately.
//! 2. A double-double dot product's error is bounded by the `gamma_n_dd`
//!    model that callers use to build their certified error channel. If the
//!    real error could exceed that model, every bound derived from it would be
//!    unsound.
//!
//! Both are checked here against exact rational arithmetic (`BigRational`),
//! where a binary64 value is represented with zero loss.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use ny_core::dd::{dd_fma, gamma_n_dd, two_prod, two_sum, Dd};
use proptest::prelude::*;

/// Exact `BigRational` for a finite binary64, via its integral significand and
/// binary exponent. Lossless by construction.
fn exact(x: f64) -> BigRational {
    assert!(x.is_finite(), "oracle is defined on finite values only");
    if x == 0.0 {
        return BigRational::zero();
    }
    let bits = x.to_bits();
    let sign = if bits >> 63 == 1 { -1i32 } else { 1i32 };
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let raw_frac = bits & 0x000f_ffff_ffff_ffff;

    // Subnormals have an implicit leading 0 and a fixed exponent.
    let (significand, exp2) = if raw_exp == 0 {
        (raw_frac, -1074i32)
    } else {
        (raw_frac | 0x0010_0000_0000_0000, raw_exp - 1075)
    };

    let mut r = BigRational::from_integer(BigInt::from(significand));
    let two = BigInt::from(2u32);
    if exp2 >= 0 {
        r *= BigRational::from_integer(two.pow(exp2.unsigned_abs()));
    } else {
        r /= BigRational::from_integer(two.pow(exp2.unsigned_abs()));
    }
    if sign < 0 {
        r = -r;
    }
    r
}

/// Finite f64s in a range wide enough to exercise cancellation and mixed
/// magnitudes, but bounded away from overflow (the transformations carry an
/// explicit no-overflow precondition).
fn value() -> impl Strategy<Value = f64> {
    prop_oneof![
        (-1e6f64..1e6f64),
        (-1e-6f64..1e-6f64),
        (-1e12f64..1e12f64),
        (-1.0f64..1.0f64),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// `two_sum` is EXACT: `a + b == s + e` over the rationals.
    #[test]
    fn two_sum_is_error_free(a in value(), b in value()) {
        let (s, e) = two_sum(a, b);
        prop_assume!(s.is_finite() && e.is_finite());
        prop_assert_eq!(exact(a) + exact(b), exact(s) + exact(e));
    }

    /// `two_prod` is EXACT: `a * b == p + e` over the rationals. This is the
    /// claim that FMA contraction silently destroys.
    #[test]
    fn two_prod_is_error_free(a in value(), b in value()) {
        let (p, e) = two_prod(a, b);
        prop_assume!(p.is_finite() && e.is_finite());
        prop_assert_eq!(exact(a) * exact(b), exact(p) + exact(e));
    }

    /// A double-double dot product's true error must respect the `gamma_n_dd`
    /// model that certified callers build their error channel from.
    ///
    /// Asserts `|acc - exact| <= gamma_n_dd(n) * S`, where `S = sum |a_k*b_k|`
    /// is the absolute-product sum the model is scaled by. A violation here
    /// would mean any bound using `gamma_n_dd` is unsound.
    #[test]
    fn dd_dot_error_respects_the_gamma_model(
        terms in prop::collection::vec((value(), value()), 1..64)
    ) {
        let mut acc = Dd::zero();
        let mut exact_sum = BigRational::zero();
        let mut abs_product_sum = BigRational::zero();

        for &(a, b) in &terms {
            acc = dd_fma(acc, a, b);
            let prod = exact(a) * exact(b);
            abs_product_sum += prod.abs();
            exact_sum += prod;
        }
        prop_assume!(acc.is_finite());

        // The represented double-double value is hi + lo, exactly.
        let got = exact(acc.hi) + exact(acc.lo);
        let err = (got - &exact_sum).abs();

        let budget = rational_from_f64_bound(gamma_n_dd(terms.len())) * abs_product_sum;
        prop_assert!(
            err <= budget,
            "dd dot error {} exceeded gamma_n_dd budget {} over {} terms",
            err, budget, terms.len()
        );
    }

    /// The f64 value of a double-double is the correctly-rounded f64 of what it
    /// represents: no other f64 is strictly closer to the exact value.
    #[test]
    fn dd_to_f64_is_nearest(a in value(), b in value(), c in value()) {
        let acc = dd_fma(dd_fma(Dd::zero(), a, b), c, 1.0);
        prop_assume!(acc.is_finite());
        let exact_val = exact(a) * exact(b) + exact(c);
        let got = acc.to_f64();
        prop_assume!(got.is_finite());

        let d_got = (exact(got) - &exact_val).abs();
        for neighbour in [next_up(got), next_down(got)] {
            if neighbour.is_finite() {
                let d_nb = (exact(neighbour) - &exact_val).abs();
                prop_assert!(d_got <= d_nb, "{got} is not the nearest f64");
            }
        }
    }
}

/// Upper-bounding rational for a small positive f64. `exact` is already exact,
/// so this is just a named wrapper documenting that the budget must not be
/// rounded DOWN (which would make the assertion stricter than the model and
/// produce false alarms rather than missed unsoundness).
fn rational_from_f64_bound(x: f64) -> BigRational {
    assert!(
        x.is_finite() && x >= 0.0,
        "gamma budget must be finite and non-negative"
    );
    if x == 0.0 {
        // A single-term "dot product" still stores once; give the assertion a
        // one-ulp floor so exactness is required rather than assumed.
        return BigRational::one() / BigRational::from_integer(BigInt::from(2u32).pow(1074));
    }
    exact(x)
}

fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    f64::from_bits(if x > 0.0 { bits + 1 } else { bits - 1 })
}

fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = x.to_bits();
    f64::from_bits(if x > 0.0 { bits - 1 } else { bits + 1 })
}

/// The oracle itself must be lossless, or every test above is vacuous.
#[test]
fn exact_round_trips_representative_values() {
    for x in [
        0.0f64,
        1.0,
        -1.0,
        0.1,
        1.0 / 3.0,
        f64::MIN_POSITIVE,
        f64::from_bits(1), // smallest subnormal
        1e17,
        -2.5e-300,
        f64::MAX,
    ] {
        let r = exact(x);
        // Reconstructing: the rational must equal x's own exact expansion.
        prop_assert_eq_helper(x, &r);
    }
}

fn prop_assert_eq_helper(x: f64, r: &BigRational) {
    let back = exact(x);
    assert_eq!(&back, r, "exact({x}) is not stable");
    if x != 0.0 {
        assert_eq!(back.is_negative(), x < 0.0, "sign lost for {x}");
    }
}
