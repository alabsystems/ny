// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::BoundedTensor;
use ndarray::{arr1, arr2, Array, Array2, IxDyn};
use proptest::prelude::*;

/// Strategy to generate valid interval bounds [lower, upper] where lower <= upper
fn valid_interval() -> impl Strategy<Value = (f32, f32)> {
    // Use ranges that avoid infinity and extreme values
    (-1000.0f32..1000.0f32)
        .prop_flat_map(|a| (-1000.0f32..1000.0f32).prop_map(move |b| (a.min(b), a.max(b))))
}

/// Strategy to generate valid interval bounds including near-extreme values.
///
/// Unlike `valid_interval()`, this includes values near `f32::MAX`/`f32::MIN`
/// which can trigger saturation to ±inf in `round_for_soundness_n_ulps`.
fn valid_interval_with_extremes() -> impl Strategy<Value = (f32, f32)> {
    (
        prop_oneof![
            -1000.0f32..1000.0f32,
            // Near-extreme values that trigger saturation with moderate n
            Just(f32::MAX),
            Just(f32::MAX - 1e30),
            Just(f32::MIN),
            Just(f32::MIN + 1e30),
        ],
        prop_oneof![
            -1000.0f32..1000.0f32,
            Just(f32::MAX),
            Just(f32::MAX - 1e30),
            Just(f32::MIN),
            Just(f32::MIN + 1e30),
        ],
    )
        .prop_map(|(a, b)| if a <= b { (a, b) } else { (b, a) })
}

/// Strategy to generate valid interval bounds with optional infinities.
///
/// Includes ±`f32::MAX` so scalar ops that overflow an endpoint to ±inf
/// (e.g. `[2, MAX].scale(1e10)`) are exercised by the containment checks.
fn valid_interval_with_infinities() -> impl Strategy<Value = (f32, f32)> {
    (
        prop_oneof![
            Just(f32::NEG_INFINITY),
            Just(f32::MIN),
            -1000.0f32..1000.0f32,
            Just(f32::MAX),
            Just(f32::INFINITY),
        ],
        prop_oneof![
            Just(f32::NEG_INFINITY),
            Just(f32::MIN),
            -1000.0f32..1000.0f32,
            Just(f32::MAX),
            Just(f32::INFINITY),
        ],
    )
        .prop_map(|(a, b)| if a <= b { (a, b) } else { (b, a) })
}

/// Sample points within an interval for verification.
/// Uses clamping to ensure FP rounding doesn't produce out-of-bounds samples.
fn sample_points(lower: f32, upper: f32, num_samples: usize) -> Vec<f32> {
    if lower == upper {
        return vec![lower];
    }
    (0..=num_samples)
        .map(|i| {
            let t = i as f32 / num_samples as f32;
            let sample = lower + (upper - lower) * t;
            // Clamp to handle FP rounding that could exceed bounds
            sample.clamp(lower, upper)
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(256) })]
    /// Interval addition soundness: [a,b] + [c,d] must contain a+c and b+d
    #[test]
    fn soundness_interval_add(
        (a, b) in valid_interval(),
        (c, d) in valid_interval()
    ) {
        let t1 = BoundedTensor::new(
            arr1(&[a]).into_dyn(),
            arr1(&[b]).into_dyn()
        ).expect("invariant: test construction");
        let t2 = BoundedTensor::new(
            arr1(&[c]).into_dyn(),
            arr1(&[d]).into_dyn()
        ).expect("invariant: test construction");

        let result = t1.add(&t2).expect("invariant: test construction");

        // Test that for any x1 in [a,b] and x2 in [c,d], x1+x2 is in result bounds
        for x1 in sample_points(a, b, 10) {
            for x2 in sample_points(c, d, 10) {
                let sum = x1 + x2;
                prop_assert!(
                    result.lower()[[0]] <= sum && sum <= result.upper()[[0]],
                    "Addition unsound: {}+{}={} not in [{}, {}]",
                    x1, x2, sum, result.lower()[[0]], result.upper()[[0]]
                );
            }
        }
    }

    /// Interval multiplication soundness:
    /// [a,b] * [c,d] must contain all products x*y for x in [a,b], y in [c,d]
    #[test]
    fn soundness_interval_mul(
        (a, b) in valid_interval(),
        (c, d) in valid_interval()
    ) {
        let t1 = BoundedTensor::new(
            arr1(&[a]).into_dyn(),
            arr1(&[b]).into_dyn()
        ).expect("invariant: test construction");
        let t2 = BoundedTensor::new(
            arr1(&[c]).into_dyn(),
            arr1(&[d]).into_dyn()
        ).expect("invariant: test construction");

        let result = t1.mul(&t2).expect("invariant: test construction");

        // Test that for any x1 in [a,b] and x2 in [c,d], x1*x2 is in result bounds
        for x1 in sample_points(a, b, 10) {
            for x2 in sample_points(c, d, 10) {
                let product = x1 * x2;
                prop_assert!(
                    result.lower()[[0]] <= product && product <= result.upper()[[0]],
                    "Multiplication unsound: {}*{}={} not in [{}, {}]",
                    x1, x2, product, result.lower()[[0]], result.upper()[[0]]
                );
            }
        }
    }

    /// Scalar multiplication soundness: s * [a,b] must contain s*x for all x in [a,b]
    #[test]
    fn soundness_scalar_mul(
        (a, b) in valid_interval(),
        s in -100.0f32..100.0f32
    ) {
        let t = BoundedTensor::new(
            arr1(&[a]).into_dyn(),
            arr1(&[b]).into_dyn()
        ).expect("invariant: test construction");

        let result = t.scale(s);

        // Test that for any x in [a,b], s*x is in result bounds
        for x in sample_points(a, b, 20) {
            let scaled = s * x;
            prop_assert!(
                result.lower()[[0]] <= scaled && scaled <= result.upper()[[0]],
                "Scalar mul unsound: {}*{}={} not in [{}, {}]",
                s, x, scaled, result.lower()[[0]], result.upper()[[0]]
            );
        }
    }

    /// Scalar multiplication with inf endpoints and zero scalars must remain valid,
    /// and the result must contain s*x for every finite x in the interval —
    /// including x large enough that the endpoint product overflows to ±inf.
    #[test]
    fn soundness_scalar_mul_infinite_bounds_and_zero_scalar(
        (a, b) in valid_interval_with_infinities(),
        s in prop_oneof![Just(0.0f32), -100.0f32..100.0f32]
    ) {
        let t = BoundedTensor::from_parts_unchecked(
            arr1(&[a]).into_dyn(),
            arr1(&[b]).into_dyn()
        );

        let result = t.scale(s);
        let lower = result.lower()[[0]];
        let upper = result.upper()[[0]];

        prop_assert!(!lower.is_nan(), "scale lower must not be NaN: [{a}, {b}] * {s} -> {lower}");
        prop_assert!(!upper.is_nan(), "scale upper must not be NaN: [{a}, {b}] * {s} -> {upper}");
        prop_assert!(lower <= upper, "scale produced inverted bounds: [{lower}, {upper}]");

        // Containment over the finite points of the interval. An unbounded-below
        // (resp. -above) endpoint is replaced by a large finite proxy inside the
        // interval; the proxy magnitude keeps the sampling span (hi - lo) finite.
        // Degenerate all-infinite intervals like [inf, inf] have no finite points.
        let sample_lo = if a == f32::NEG_INFINITY { -f32::MAX / 2.0 } else { a };
        let sample_hi = if b == f32::INFINITY { f32::MAX / 2.0 } else { b };
        if sample_lo.is_finite()
            && sample_hi.is_finite()
            && sample_lo <= sample_hi
            && (sample_hi - sample_lo).is_finite()
        {
            for x in sample_points(sample_lo, sample_hi, 20) {
                let scaled = s * x;
                prop_assert!(
                    lower <= scaled && scaled <= upper,
                    "Scalar mul unsound: {}*{}={} not in [{}, {}] (input [{}, {}])",
                    s, x, scaled, lower, upper, a, b
                );
            }
        }
    }

    /// Scalar addition soundness: [a,b] + s must contain x+s for all x in [a,b]
    #[test]
    fn soundness_scalar_add(
        (a, b) in valid_interval(),
        s in -100.0f32..100.0f32
    ) {
        let t = BoundedTensor::new(
            arr1(&[a]).into_dyn(),
            arr1(&[b]).into_dyn()
        ).expect("invariant: test construction");

        let result = t.shift(s);

        // Test that for any x in [a,b], x+s is in result bounds
        for x in sample_points(a, b, 20) {
            let shifted = x + s;
            prop_assert!(
                result.lower()[[0]] <= shifted && shifted <= result.upper()[[0]],
                "Scalar add unsound: {}+{}={} not in [{}, {}]",
                x, s, shifted, result.lower()[[0]], result.upper()[[0]]
            );
        }
    }

    /// Scalar addition with infinite bounds and infinite scalars must remain valid,
    /// and the result must contain x+s for every finite x in the interval —
    /// including infinite s, where x+s is ±inf and must still be enclosed.
    #[test]
    fn soundness_scalar_add_infinite_bounds_and_scalar(
        (a, b) in valid_interval_with_infinities(),
        s in prop_oneof![Just(f32::NEG_INFINITY), Just(f32::INFINITY), -100.0f32..100.0f32]
    ) {
        let t = BoundedTensor::from_parts_unchecked(
            arr1(&[a]).into_dyn(),
            arr1(&[b]).into_dyn()
        );

        let result = t.shift(s);
        let lower = result.lower()[[0]];
        let upper = result.upper()[[0]];

        prop_assert!(!lower.is_nan(), "shift lower must not be NaN: [{a}, {b}] + {s} -> {lower}");
        prop_assert!(!upper.is_nan(), "shift upper must not be NaN: [{a}, {b}] + {s} -> {upper}");
        prop_assert!(lower <= upper, "shift produced inverted bounds: [{lower}, {upper}]");

        // Containment over the finite points of the interval. An unbounded-below
        // (resp. -above) endpoint is replaced by a large finite proxy inside the
        // interval; the proxy magnitude keeps the sampling span (hi - lo) finite.
        // Degenerate all-infinite intervals like [inf, inf] have no finite points.
        let sample_lo = if a == f32::NEG_INFINITY { -f32::MAX / 2.0 } else { a };
        let sample_hi = if b == f32::INFINITY { f32::MAX / 2.0 } else { b };
        if sample_lo.is_finite()
            && sample_hi.is_finite()
            && sample_lo <= sample_hi
            && (sample_hi - sample_lo).is_finite()
        {
            for x in sample_points(sample_lo, sample_hi, 20) {
                let shifted = x + s;
                prop_assert!(
                    lower <= shifted && shifted <= upper,
                    "Scalar add unsound: {}+{}={} not in [{}, {}] (input [{}, {}])",
                    x, s, shifted, lower, upper, a, b
                );
            }
        }
    }

    /// Directed rounding must always widen bounds (for finite values)
    #[test]
    fn soundness_directed_rounding(
        (a, b) in valid_interval()
    ) {
        let t = BoundedTensor::new(
            arr1(&[a]).into_dyn(),
            arr1(&[b]).into_dyn()
        ).expect("invariant: test construction");

        let rounded = t.round_for_soundness();

        // Rounded bounds must be wider or equal (contain original)
        prop_assert!(
            rounded.lower()[[0]] <= t.lower()[[0]],
            "Directed rounding should decrease lower: {} > {}",
            rounded.lower()[[0]], t.lower()[[0]]
        );
        prop_assert!(
            rounded.upper()[[0]] >= t.upper()[[0]],
            "Directed rounding should increase upper: {} < {}",
            rounded.upper()[[0]], t.upper()[[0]]
        );
    }

    /// new_allow_infinite must accept all outputs of round_for_soundness_n_ulps (#2944).
    ///
    /// round_for_soundness_n_ulps can produce ±inf via saturation at f32::MAX/MIN.
    /// new_allow_infinite must accept these as valid bounds. Uses
    /// valid_interval_with_extremes() to actually exercise the saturation path.
    #[test]
    fn new_allow_infinite_accepts_n_ulp_rounded_bounds(
        (a, b) in valid_interval_with_extremes(),
        n in 0u32..100u32,
    ) {
        // Use new_allow_infinite for construction since extremes may include ±MAX
        let t = BoundedTensor::new_allow_infinite(
            arr1(&[a]).into_dyn(),
            arr1(&[b]).into_dyn()
        ).expect("invariant: valid_interval_with_extremes() produces valid bounds");

        let rounded = t.round_for_soundness_n_ulps(n);
        let result = BoundedTensor::new_allow_infinite(
            rounded.lower().clone(),
            rounded.upper().clone(),
        );
        prop_assert!(
            result.is_ok(),
            "new_allow_infinite must accept round_for_soundness_n_ulps output: \
             n={}, input=[{}, {}], rounded=[{}, {}], err={:?}",
            n, a, b, rounded.lower()[[0]], rounded.upper()[[0]], result.err()
        );
    }

    /// round_for_soundness_n_ulps must always produce valid bounds (lower <= upper) (#2944).
    #[test]
    fn n_ulps_preserves_bound_ordering(
        (a, b) in valid_interval(),
        n in 0u32..1000u32,
    ) {
        let t = BoundedTensor::new(
            arr1(&[a]).into_dyn(),
            arr1(&[b]).into_dyn()
        ).expect("invariant: valid_interval() produces valid bounds");

        let rounded = t.round_for_soundness_n_ulps(n);
        prop_assert!(
            rounded.lower()[[0]] <= rounded.upper()[[0]],
            "round_for_soundness_n_ulps({n}) produced inverted bounds: [{}, {}] from [{a}, {b}]",
            rounded.lower()[[0]], rounded.upper()[[0]]
        );
    }

    /// round_for_soundness_n_ulps must always contain the original bounds (#2944).
    #[test]
    fn n_ulps_contains_original(
        (a, b) in valid_interval(),
        n in 0u32..100u32,
    ) {
        let t = BoundedTensor::new(
            arr1(&[a]).into_dyn(),
            arr1(&[b]).into_dyn()
        ).expect("invariant: valid_interval() produces valid bounds");

        let rounded = t.round_for_soundness_n_ulps(n);
        prop_assert!(
            rounded.lower()[[0]] <= a,
            "rounded lower ({}) should be <= original lower ({a})",
            rounded.lower()[[0]]
        );
        prop_assert!(
            rounded.upper()[[0]] >= b,
            "rounded upper ({}) should be >= original upper ({b})",
            rounded.upper()[[0]]
        );
    }

    /// Per-element intersection soundness (#2935):
    /// For overlapping elements, result is the intersection (subset of both).
    /// For disjoint elements, result is the union (superset of both).
    /// Result always has lower <= upper.
    #[test]
    fn soundness_intersection_per_element(
        (a_lo, a_hi) in valid_interval(),
        (b_lo, b_hi) in valid_interval(),
        (c_lo, c_hi) in valid_interval(),
        (d_lo, d_hi) in valid_interval(),
    ) {
        let t1 = BoundedTensor::new(
            arr1(&[a_lo, c_lo]).into_dyn(),
            arr1(&[a_hi, c_hi]).into_dyn(),
        ).expect("invariant: test construction");
        let t2 = BoundedTensor::new(
            arr1(&[b_lo, d_lo]).into_dyn(),
            arr1(&[b_hi, d_hi]).into_dyn(),
        ).expect("invariant: test construction");

        let (result, _disjoint) = t1.intersection_per_element(&t2)
            .expect("invariant: valid intervals (no NaN, shapes match) always return Some");

        // Result must have valid bounds
        for i in 0..2 {
            prop_assert!(
                result.lower()[[i]] <= result.upper()[[i]],
                "intersection_per_element produced inverted bounds at element {}",
                i,
            );
        }

        // For each element, check soundness properties
        for (i, ((al, ah), (bl, bh))) in [(a_lo, a_hi), (c_lo, c_hi)].iter()
            .zip([(b_lo, b_hi), (d_lo, d_hi)].iter())
            .enumerate()
        {
            let rl = result.lower()[[i]];
            let ru = result.upper()[[i]];
            let tl = al.max(*bl);
            let tu = ah.min(*bh);
            if tl <= tu {
                // Overlapping: result is intersection (tighter)
                prop_assert!(
                    rl >= *al && rl >= *bl,
                    "Overlapping element {}: result lower {} should be >= both {} and {}",
                    i, rl, al, bl,
                );
                prop_assert!(
                    ru <= *ah && ru <= *bh,
                    "Overlapping element {}: result upper {} should be <= both {} and {}",
                    i, ru, ah, bh,
                );
            } else {
                // Disjoint: result is union (conservative)
                prop_assert!(
                    rl <= al.min(*bl),
                    "Disjoint element {}: result lower {} should be <= min({}, {})",
                    i, rl, al, bl,
                );
                prop_assert!(
                    ru >= ah.max(*bh),
                    "Disjoint element {}: result upper {} should be >= max({}, {})",
                    i, ru, ah, bh,
                );
            }
        }
    }

    /// Bounds should remain valid (lower <= upper) after operations
    #[test]
    fn validity_bounds_ordering(
        (a, b) in valid_interval(),
        (c, d) in valid_interval(),
        s in -100.0f32..100.0f32
    ) {
        let t1 = BoundedTensor::new(
            arr1(&[a]).into_dyn(),
            arr1(&[b]).into_dyn()
        ).expect("invariant: test construction");
        let t2 = BoundedTensor::new(
            arr1(&[c]).into_dyn(),
            arr1(&[d]).into_dyn()
        ).expect("invariant: test construction");

        // After add
        let added = t1.add(&t2).expect("invariant: test construction");
        prop_assert!(added.lower()[[0]] <= added.upper()[[0]], "Add produced inverted bounds");

        // After mul
        let multed = t1.mul(&t2).expect("invariant: test construction");
        prop_assert!(multed.lower()[[0]] <= multed.upper()[[0]], "Mul produced inverted bounds");

        // After scale
        let scaled = t1.scale(s);
        prop_assert!(scaled.lower()[[0]] <= scaled.upper()[[0]], "Scale produced inverted bounds");

        // After shift
        let shifted = t1.shift(s);
        prop_assert!(shifted.lower()[[0]] <= shifted.upper()[[0]], "Shift produced inverted bounds");

        // After rounding
        let rounded = t1.round_for_soundness();
        prop_assert!(rounded.lower()[[0]] <= rounded.upper()[[0]], "Rounding produced inverted bounds");
    }
}

#[test]
fn test_new_sanitized_with_nan() {
    let lower = arr1(&[1.0, f32::NAN, -1.0]).into_dyn();
    let upper = arr1(&[2.0, f32::NAN, 0.0]).into_dyn();
    let clamp_val = 100.0;

    let t = BoundedTensor::new_sanitized(lower, upper, clamp_val)
        .expect("invariant: test construction");

    // NaN in lower should become -clamp_val
    assert_eq!(t.lower()[[1]], -clamp_val);
    // NaN in upper should become +clamp_val
    assert_eq!(t.upper()[[1]], clamp_val);
    // Finite values should be unchanged
    assert_eq!(t.lower()[[0]], 1.0);
    assert_eq!(t.upper()[[0]], 2.0);
}

#[test]
fn test_new_sanitized_with_infinity() {
    let lower = arr1(&[f32::NEG_INFINITY, 0.0, f32::INFINITY]).into_dyn();
    let upper = arr1(&[0.0, f32::INFINITY, f32::INFINITY]).into_dyn();
    let clamp_val = 1e10;

    let t = BoundedTensor::new_sanitized(lower, upper, clamp_val)
        .expect("invariant: test construction");

    // -Inf should become -clamp_val
    assert_eq!(t.lower()[[0]], -clamp_val);
    // +Inf should become +clamp_val
    assert_eq!(t.upper()[[1]], clamp_val);
    // +Inf in lower should become +clamp_val (then possibly swapped)
    assert!(
        t.lower()[[2]] <= t.upper()[[2]],
        "sanitized bounds at index 2 must be ordered: lower={}, upper={}",
        t.lower()[[2]],
        t.upper()[[2]]
    );
}

#[test]
fn test_new_sanitized_ensures_ordering() {
    // Simulate case where sanitization might invert bounds
    // (e.g., NaN replaced differently in lower vs upper)
    let lower = arr1(&[f32::NAN]).into_dyn();
    let upper = arr1(&[f32::NAN]).into_dyn();
    let clamp_val = 100.0;

    let t = BoundedTensor::new_sanitized(lower, upper, clamp_val)
        .expect("invariant: test construction");

    // Even after NaN replacement, lower <= upper must hold
    assert!(
        t.lower()[[0]] <= t.upper()[[0]],
        "NaN-sanitized bounds must be ordered: lower={}, upper={}",
        t.lower()[[0]],
        t.upper()[[0]]
    );
}

#[test]
fn test_new_sanitized_preserves_shape() {
    let lower = arr2(&[[1.0, 2.0], [3.0, f32::NAN]]).into_dyn();
    let upper = arr2(&[[2.0, 3.0], [4.0, f32::INFINITY]]).into_dyn();
    let clamp_val = 100.0;

    let t = BoundedTensor::new_sanitized(lower.clone(), upper, clamp_val)
        .expect("invariant: test construction");

    assert_eq!(t.shape(), lower.shape());
}

#[test]
fn test_sanitize_method() {
    let lower = arr1(&[f32::NAN, 1.0, f32::NEG_INFINITY]).into_dyn();
    let upper = arr1(&[f32::INFINITY, 2.0, f32::NAN]).into_dyn();

    // Use new_unchecked to create tensor with NaN/Inf (bypasses assertions)
    let t = BoundedTensor::new_unchecked(lower, upper).expect("invariant: test construction");

    assert!(
        t.has_overflow(),
        "tensor with NaN/Inf should report overflow"
    );

    let sanitized = t.sanitize(1000.0);

    assert!(
        !sanitized.has_overflow(),
        "sanitized tensor should not have overflow"
    );
    assert!(
        sanitized.lower()[[0]] <= sanitized.upper()[[0]],
        "sanitized bounds[0] must be ordered"
    );
    assert!(
        sanitized.lower()[[1]] <= sanitized.upper()[[1]],
        "sanitized bounds[1] must be ordered"
    );
    assert!(
        sanitized.lower()[[2]] <= sanitized.upper()[[2]],
        "sanitized bounds[2] must be ordered"
    );
}

#[test]
fn test_has_overflow() {
    // Normal tensor should not have overflow
    let normal = BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn())
        .expect("invariant: test construction");
    assert!(
        !normal.has_overflow(),
        "normal finite tensor should not have overflow"
    );

    // Tensor with Inf should have overflow
    let with_inf = BoundedTensor::new_unchecked(
        arr1(&[1.0, f32::NEG_INFINITY]).into_dyn(),
        arr1(&[3.0, f32::INFINITY]).into_dyn(),
    )
    .expect("invariant: test construction");
    assert!(
        with_inf.has_overflow(),
        "tensor with Inf bounds should report overflow"
    );

    // Tensor with NaN should have overflow
    let with_nan = BoundedTensor::new_unchecked(
        arr1(&[f32::NAN, 2.0]).into_dyn(),
        arr1(&[3.0, 4.0]).into_dyn(),
    )
    .expect("invariant: test construction");
    assert!(
        with_nan.has_overflow(),
        "tensor with NaN bounds should report overflow"
    );
}

// ========================================
// Mutation-killing tests for BoundedTensor
// ========================================

#[test]
fn test_ndim_exact_values() {
    // Test that ndim returns exact dimension count, not 0 or 1
    let t1d = BoundedTensor::concrete(arr1(&[1.0, 2.0, 3.0]).into_dyn())
        .expect("invariant: test construction");
    assert_eq!(t1d.ndim(), 1);

    let t2d = BoundedTensor::concrete(
        Array2::from_shape_vec((2, 3), vec![1.0; 6])
            .expect("invariant: test construction")
            .into_dyn(),
    )
    .expect("invariant: test construction");
    assert_eq!(t2d.ndim(), 2);

    let t3d = BoundedTensor::concrete(
        Array::from_shape_vec(IxDyn(&[2, 3, 4]), vec![1.0; 24])
            .expect("invariant: test construction"),
    )
    .expect("invariant: test construction");
    assert_eq!(t3d.ndim(), 3);

    let t4d = BoundedTensor::concrete(
        Array::from_shape_vec(IxDyn(&[2, 3, 4, 5]), vec![1.0; 120])
            .expect("invariant: test construction"),
    )
    .expect("invariant: test construction");
    assert_eq!(t4d.ndim(), 4);
}

#[test]
fn test_len_exact_values() {
    // Test that len returns exact element count, not 1
    let t1 =
        BoundedTensor::concrete(arr1(&[1.0]).into_dyn()).expect("invariant: test construction");
    assert_eq!(t1.len(), 1);

    let t6 = BoundedTensor::concrete(arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).into_dyn())
        .expect("invariant: test construction");
    assert_eq!(t6.len(), 6);

    let t12 = BoundedTensor::concrete(
        Array2::from_shape_vec((3, 4), vec![1.0; 12])
            .expect("invariant: test construction")
            .into_dyn(),
    )
    .expect("invariant: test construction");
    assert_eq!(t12.len(), 12);
}

#[test]
fn test_is_empty_exact() {
    // Non-empty tensor should return false, not true
    let non_empty =
        BoundedTensor::concrete(arr1(&[1.0]).into_dyn()).expect("invariant: test construction");
    assert!(
        !non_empty.is_empty(),
        "single-element tensor should not be empty"
    );

    // Empty tensor should return true
    let empty = BoundedTensor::concrete(
        Array::from_shape_vec(IxDyn(&[0]), vec![]).expect("invariant: test construction"),
    )
    .expect("invariant: test construction");
    assert!(empty.is_empty(), "zero-element tensor should be empty");
}

#[test]
fn test_has_unbounded_distinguish_bounds() {
    // Test all combinations to distinguish || vs && and false vs true

    // Neither bound infinite - should return false
    let finite =
        BoundedTensor::new_unchecked(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn())
            .expect("invariant: test construction");
    assert!(
        !finite.has_unbounded(),
        "finite bounds should not be unbounded"
    );

    // Only lower bound infinite - should return true (|| catches this)
    let lower_inf = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY, 2.0]).into_dyn(),
        arr1(&[3.0, 4.0]).into_dyn(),
    )
    .expect("invariant: test construction");
    assert!(
        lower_inf.has_unbounded(),
        "tensor with -Inf lower should be unbounded"
    );

    // Only upper bound infinite - should return true (|| catches this)
    let upper_inf = BoundedTensor::new_unchecked(
        arr1(&[1.0, 2.0]).into_dyn(),
        arr1(&[3.0, f32::INFINITY]).into_dyn(),
    )
    .expect("invariant: test construction");
    assert!(
        upper_inf.has_unbounded(),
        "tensor with +Inf upper should be unbounded"
    );

    // Both bounds infinite - should return true
    let both_inf = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .expect("invariant: test construction");
    assert!(
        both_inf.has_unbounded(),
        "tensor with both Inf bounds should be unbounded"
    );
}

#[test]
fn test_center_exact_computation() {
    // Test that center is exactly (lower + upper) / 2
    // Using values where mutations would give different results
    let t = BoundedTensor::new(
        arr1(&[0.0, 2.0, 10.0]).into_dyn(),
        arr1(&[4.0, 6.0, 20.0]).into_dyn(),
    )
    .expect("invariant: test construction");
    let center = t.center();

    // (0+4)/2 = 2, (2+6)/2 = 4, (10+20)/2 = 15
    assert_eq!(center[[0]], 2.0); // If + replaced with -, would get -2.0
    assert_eq!(center[[1]], 4.0); // If + replaced with *, would get 12.0
    assert_eq!(center[[2]], 15.0); // If / replaced with *, would get 30.0
}

#[test]
fn test_mul_exact_interval_bounds() {
    // Test element-wise multiplication computes correct interval bounds
    // For intervals [a,b] * [c,d], result is [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]

    // Positive * Positive: [2,3] * [4,5] = [8, 15]
    let a = BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[3.0]).into_dyn())
        .expect("invariant: test construction");
    let b = BoundedTensor::new(arr1(&[4.0]).into_dyn(), arr1(&[5.0]).into_dyn())
        .expect("invariant: test construction");
    let result = a.mul(&b).expect("invariant: test construction");
    assert_eq!(result.lower()[[0]], 8.0); // min(8, 10, 12, 15) = 8
    assert_eq!(result.upper()[[0]], 15.0); // max(8, 10, 12, 15) = 15

    // Mixed signs: [-1,2] * [-3,4] = [-8, 8]
    // ac=-1*-3=3, ad=-1*4=-4, bc=2*-3=-6, bd=2*4=8
    let c = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[2.0]).into_dyn())
        .expect("invariant: test construction");
    let d = BoundedTensor::new(arr1(&[-3.0]).into_dyn(), arr1(&[4.0]).into_dyn())
        .expect("invariant: test construction");
    let result2 = c.mul(&d).expect("invariant: test construction");
    assert_eq!(result2.lower()[[0]], -6.0); // min(3, -4, -6, 8) = -6
    assert_eq!(result2.upper()[[0]], 8.0); // max(3, -4, -6, 8) = 8
}

#[test]
fn test_new_sanitized_boundary_conditions() {
    // Test that > vs >= distinctions matter in new_sanitized
    // The function clamps values using: if x > 0.0 / if x > 0.0

    // Exactly 0.0 should be treated differently from > 0.0
    let clamp = 10.0;
    let result = BoundedTensor::new_sanitized(
        arr1(&[f32::NEG_INFINITY, 0.0, f32::INFINITY]).into_dyn(),
        arr1(&[f32::NEG_INFINITY, 0.0, f32::INFINITY]).into_dyn(),
        clamp,
    )
    .expect("invariant: test construction");

    // NEG_INFINITY on lower should become -clamp_val
    assert_eq!(result.lower()[[0]], -clamp);
    // INFINITY on upper should become clamp_val
    assert_eq!(result.upper()[[2]], clamp);
    // 0.0 should remain 0.0 (clamped to valid range)
    assert_eq!(result.lower()[[1]], 0.0);
    assert_eq!(result.upper()[[1]], 0.0);
}

#[test]
fn test_slice_axis_range_boundary() {
    // Test boundary condition in slice_axis_range: end > dim[axis]
    let t = BoundedTensor::concrete(
        Array::from_shape_vec(IxDyn(&[2, 5, 3]), (0..30).map(|x| x as f32).collect())
            .expect("invariant: test construction"),
    )
    .expect("invariant: test construction");

    // Valid slice
    let slice = t
        .slice_axis_range(1, 1, 4)
        .expect("invariant: test construction");
    assert_eq!(slice.shape(), &[2, 3, 3]);

    // End == dim should work (boundary case)
    let slice2 = t
        .slice_axis_range(1, 0, 5)
        .expect("invariant: test construction");
    assert_eq!(slice2.shape(), &[2, 5, 3]);

    // End > dim should fail
    let err = t.slice_axis_range(1, 0, 6);
    assert!(err.is_err(), "slice_axis_range with end > dim should fail");
}

#[test]
fn test_concat_shape_validation() {
    // Test that concat validates shapes correctly (!= vs ==)
    let t1 = BoundedTensor::concrete(Array2::zeros((2, 3)).into_dyn())
        .expect("invariant: test construction");
    let t2 = BoundedTensor::concrete(Array2::zeros((2, 3)).into_dyn())
        .expect("invariant: test construction");
    let t3 = BoundedTensor::concrete(Array2::zeros((2, 4)).into_dyn())
        .expect("invariant: test construction"); // Different shape

    // Same shapes should concat fine
    let result = BoundedTensor::concat(&[t1.clone(), t2], 0);
    assert!(
        result.is_ok(),
        "concat of same-shaped tensors should succeed"
    );
    assert_eq!(
        result.expect("invariant: test construction").shape(),
        &[4, 3]
    );

    // Different shapes should fail (except on concat axis)
    let err = BoundedTensor::concat(&[t1, t3], 0);
    assert!(
        err.is_err(),
        "concat with mismatched non-concat dimensions should fail"
    );
}

#[test]
fn test_stack_ndim_boundary() {
    // Test that stack validates ndim > 0
    let t1 = BoundedTensor::concrete(arr1(&[1.0, 2.0]).into_dyn())
        .expect("invariant: test construction");
    let t2 = BoundedTensor::concrete(arr1(&[3.0, 4.0]).into_dyn())
        .expect("invariant: test construction");

    let result = BoundedTensor::stack(&[t1, t2], 0).expect("invariant: test construction");
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.ndim(), 2);
}

#[test]
fn test_transpose_validation() {
    // Test transpose axes validation (shape[i] != shape[axes[i]])
    let t = BoundedTensor::concrete(
        Array::from_shape_vec(IxDyn(&[2, 3, 4]), vec![1.0; 24])
            .expect("invariant: test construction"),
    )
    .expect("invariant: test construction");

    // Valid transpose
    let result = t
        .transpose(&[2, 0, 1])
        .expect("invariant: test construction");
    assert_eq!(result.shape(), &[4, 2, 3]);

    // Wrong number of axes should fail
    let err = t.transpose(&[0, 1]);
    assert!(
        err.is_err(),
        "transpose with wrong number of axes should fail"
    );
}

#[test]
fn test_transpose_last_two_minimum_dims() {
    // Test that transpose_last_two requires at least 2 dimensions
    let t1d = BoundedTensor::concrete(arr1(&[1.0, 2.0]).into_dyn())
        .expect("invariant: test construction");
    let err = t1d.transpose_last_two();
    assert!(err.is_err(), "transpose_last_two on 1D tensor should fail");

    let t2d = BoundedTensor::concrete(
        Array2::from_shape_vec((2, 3), vec![1.0; 6])
            .expect("invariant: test construction")
            .into_dyn(),
    )
    .expect("invariant: test construction");
    let result = t2d
        .transpose_last_two()
        .expect("invariant: test construction");
    assert_eq!(result.shape(), &[3, 2]);
}

// ========== Mutation-killing tests for bounded_tensor ==========

#[test]
fn test_new_sanitized_positive_infinity_exact_values() {
    // Target: new_sanitized infinity sign handling.
    // Tests that +INFINITY becomes +clamp_val (not -clamp_val).
    let clamp = 1e6;

    // Positive infinity in LOWER bound should become +clamp_val
    let result = BoundedTensor::new_sanitized(
        arr1(&[f32::INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
        clamp,
    )
    .expect("invariant: test construction");
    // If `> 0.0` mutated to `== 0.0`, INFINITY would give -clamp_val
    assert_eq!(
        result.lower()[[0]],
        clamp,
        "INFINITY in lower must become +clamp_val"
    );
    assert_eq!(
        result.upper()[[0]],
        clamp,
        "INFINITY in upper must become +clamp_val"
    );

    // Positive infinity in lower with finite upper
    let result2 = BoundedTensor::new_sanitized(
        arr1(&[f32::INFINITY]).into_dyn(),
        arr1(&[2e6]).into_dyn(), // Larger than clamp
        clamp,
    )
    .expect("invariant: test construction");
    // Lower should be clamped to +clamp_val, then may be swapped if needed
    // After sanitization, lower should be clamp (1e6), upper clamped to clamp (1e6)
    assert!(
        result2.lower()[[0]] <= result2.upper()[[0]],
        "sanitized +Inf lower must be <= upper: lower={}, upper={}",
        result2.lower()[[0]],
        result2.upper()[[0]]
    );
    assert_eq!(result2.lower()[[0]], clamp);
}

#[test]
fn test_new_sanitized_negative_infinity_upper() {
    // Target: new_sanitized negative infinity handling.
    // Tests that -INFINITY in upper becomes -clamp_val.
    let clamp = 500.0;

    let result = BoundedTensor::new_sanitized(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        clamp,
    )
    .expect("invariant: test construction");
    // Both should become -clamp_val
    // If `delete -` mutation, NEG_INFINITY would give +clamp_val (wrong)
    assert_eq!(
        result.lower()[[0]],
        -clamp,
        "NEG_INFINITY in lower must become -clamp_val"
    );
    assert_eq!(
        result.upper()[[0]],
        -clamp,
        "NEG_INFINITY in upper must become -clamp_val"
    );
}

#[test]
fn test_new_sanitized_swap_only_when_inverted() {
    // Target: new_sanitized swap condition.
    // Ensure swap only happens when lower > upper (not when equal).
    let clamp = 100.0;

    // Case: lower == upper (should NOT swap, values should stay as-is)
    let result = BoundedTensor::new_sanitized(
        arr1(&[5.0, 10.0]).into_dyn(),
        arr1(&[5.0, 10.0]).into_dyn(),
        clamp,
    )
    .expect("invariant: test construction");
    assert_eq!(result.lower()[[0]], 5.0);
    assert_eq!(result.upper()[[0]], 5.0);
    assert_eq!(result.lower()[[1]], 10.0);
    assert_eq!(result.upper()[[1]], 10.0);

    // Case: lower > upper (should swap)
    let result2 =
        BoundedTensor::new_sanitized(arr1(&[20.0]).into_dyn(), arr1(&[10.0]).into_dyn(), clamp)
            .expect("invariant: test construction");
    assert_eq!(result2.lower()[[0]], 10.0, "After swap, lower should be 10");
    assert_eq!(result2.upper()[[0]], 20.0, "After swap, upper should be 20");

    // Case: lower < upper (should NOT swap)
    let result3 =
        BoundedTensor::new_sanitized(arr1(&[5.0]).into_dyn(), arr1(&[15.0]).into_dyn(), clamp)
            .expect("invariant: test construction");
    assert_eq!(result3.lower()[[0]], 5.0);
    assert_eq!(result3.upper()[[0]], 15.0);
}

#[test]
fn test_concat_validates_all_non_concat_dimensions() {
    // Target: concat shape validation.
    // Ensure concat validates ALL dimensions except concat axis.

    // Create tensors with matching first dim but different second dim
    let t1 = BoundedTensor::concrete(Array2::zeros((3, 4)).into_dyn())
        .expect("invariant: test construction");
    let t2 = BoundedTensor::concrete(Array2::zeros((3, 5)).into_dyn())
        .expect("invariant: test construction"); // dim 1 differs

    // Concat on axis 0: dimension 1 differs (4 vs 5), should fail
    let err = BoundedTensor::concat(&[t1.clone(), t2.clone()], 0);
    assert!(
        err.is_err(),
        "Concat should fail when non-concat dimension differs"
    );

    // But concat on axis 1 should work (dimension 0 matches)
    let result = BoundedTensor::concat(&[t1, t2], 1);
    assert!(
        result.is_ok(),
        "concat on axis 1 with matching dim 0 should succeed"
    );
    assert_eq!(
        result.expect("invariant: test construction").shape(),
        &[3, 9]
    ); // 4 + 5 = 9
}

#[test]
fn test_stack_axis_at_ndim_boundary() {
    // Target: stack axis bounds check.
    // Ensure stack allows axis == ndim (insert at end).

    let t1 = BoundedTensor::concrete(arr1(&[1.0, 2.0]).into_dyn())
        .expect("invariant: test construction"); // shape [2]
    let t2 = BoundedTensor::concrete(arr1(&[3.0, 4.0]).into_dyn())
        .expect("invariant: test construction");

    // axis = 0: new axis at beginning -> [2, 2]
    let r0 =
        BoundedTensor::stack(&[t1.clone(), t2.clone()], 0).expect("invariant: test construction");
    assert_eq!(r0.shape(), &[2, 2]);

    // axis = 1 = ndim: new axis at end -> [2, 2]
    // With `> ndim`, this should work. With `>= ndim`, it would fail.
    let r1 =
        BoundedTensor::stack(&[t1.clone(), t2.clone()], 1).expect("invariant: test construction");
    assert_eq!(r1.shape(), &[2, 2]);

    // axis = 2 > ndim: should fail
    let err = BoundedTensor::stack(&[t1, t2], 2);
    assert!(err.is_err(), "Stack axis beyond ndim should fail");
}

#[test]
fn test_mul_non_contiguous_arrays() {
    // Target: mul fallback uses products (not + or /) for non-contiguous arrays.
    // Force non-contiguous arrays by using permuted_axes without as_standard_layout.

    // Test case 1: ad (a*d) is the minimum.
    // a=-2, b=1, c=2, d=3
    // Products: ac=-4, ad=-6, bc=2, bd=3
    // With *: min=-6, max=3
    // With + on ad: ad=-2+3=1, min would be -4 (DIFFERENT!)

    // Create non-contiguous arrays using permuted_axes on owned 2D arrays
    // permuted_axes on owned Array2 reorders axes but keeps same data buffer
    let arr1_lower = Array2::<f32>::from_shape_vec((2, 3), vec![-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test construction");
    let arr1_upper = Array2::<f32>::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test construction");
    // Transpose: [2,3] -> [3,2], makes it non-contiguous
    let lower1_t = arr1_lower.permuted_axes([1, 0]).into_dyn();
    let upper1_t = arr1_upper.permuted_axes([1, 0]).into_dyn();

    // Verify these are actually non-contiguous (as_slice returns None)
    assert!(
        lower1_t.as_slice().is_none(),
        "Array should be non-contiguous after permute"
    );

    let t1 = BoundedTensor::new(lower1_t, upper1_t).expect("invariant: test construction");

    let arr2_lower = Array2::<f32>::from_shape_vec((2, 3), vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test construction");
    let arr2_upper = Array2::<f32>::from_shape_vec((2, 3), vec![3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test construction");
    let lower2_t = arr2_lower.permuted_axes([1, 0]).into_dyn();
    let upper2_t = arr2_upper.permuted_axes([1, 0]).into_dyn();
    let t2 = BoundedTensor::new(lower2_t, upper2_t).expect("invariant: test construction");

    let result = t1.mul(&t2).expect("invariant: test construction");
    assert_eq!(
        result.lower()[[0, 0]],
        -6.0,
        "min should be ad=-6 (a*d=-2*3)"
    );
    assert_eq!(result.upper()[[0, 0]], 3.0, "max should be bd=3 (b*d=1*3)");

    // Test case 2: bc (b*c) is the minimum.
    // a=-1, b=2, c=-3, d=4
    // Products: ac=3, ad=-4, bc=-6, bd=8
    // With *: min=-6, max=8
    // With + on bc: bc=2+-3=-1, min would be -4 (DIFFERENT!)
    // Must use non-contiguous arrays to hit the fallback path.
    let arr3_lower = Array2::<f32>::from_shape_vec((2, 3), vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test construction");
    let arr3_upper = Array2::<f32>::from_shape_vec((2, 3), vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test construction");
    let lower3_t = arr3_lower.permuted_axes([1, 0]).into_dyn();
    let upper3_t = arr3_upper.permuted_axes([1, 0]).into_dyn();
    let t3 = BoundedTensor::new(lower3_t, upper3_t).expect("invariant: test construction");

    let arr4_lower = Array2::<f32>::from_shape_vec((2, 3), vec![-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test construction");
    let arr4_upper = Array2::<f32>::from_shape_vec((2, 3), vec![4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test construction");
    let lower4_t = arr4_lower.permuted_axes([1, 0]).into_dyn();
    let upper4_t = arr4_upper.permuted_axes([1, 0]).into_dyn();
    let t4 = BoundedTensor::new(lower4_t, upper4_t).expect("invariant: test construction");

    let result2 = t3.mul(&t4).expect("invariant: test construction");
    assert_eq!(
        result2.lower()[[0, 0]],
        -6.0,
        "min should be bc=-6 (b*c=2*-3)"
    );
    assert_eq!(result2.upper()[[0, 0]], 8.0, "max should be bd=8 (b*d=2*4)");

    // Test case 3: ac (a*c) is the maximum.
    // a=-3, b=-1, c=-2, d=-1
    // Products: ac=6, ad=3, bc=2, bd=1
    // With *: min=1, max=6
    // With + on ac: ac=-3+-2=-5, products are [-5, 3, 2, 1], max=3 (DIFFERENT!)
    let arr5_lower = Array2::<f32>::from_shape_vec((2, 3), vec![-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test construction");
    let arr5_upper = Array2::<f32>::from_shape_vec((2, 3), vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test construction");
    let lower5_t = arr5_lower.permuted_axes([1, 0]).into_dyn();
    let upper5_t = arr5_upper.permuted_axes([1, 0]).into_dyn();
    let t5 = BoundedTensor::new(lower5_t, upper5_t).expect("invariant: test construction");

    let arr6_lower = Array2::<f32>::from_shape_vec((2, 3), vec![-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test construction");
    let arr6_upper = Array2::<f32>::from_shape_vec((2, 3), vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test construction");
    let lower6_t = arr6_lower.permuted_axes([1, 0]).into_dyn();
    let upper6_t = arr6_upper.permuted_axes([1, 0]).into_dyn();
    let t6 = BoundedTensor::new(lower6_t, upper6_t).expect("invariant: test construction");

    let result3 = t5.mul(&t6).expect("invariant: test construction");
    assert_eq!(
        result3.lower()[[0, 0]],
        1.0,
        "min should be bd=1 (b*d=-1*-1)"
    );
    assert_eq!(
        result3.upper()[[0, 0]],
        6.0,
        "max should be ac=6 (a*c=-3*-2)"
    );
}

#[test]
fn test_mul_exact_products_negative_intervals() {
    // Additional mul test with negative values to catch * with + or / mutations
    // For intervals with negatives, the products differ significantly from sums/quotients

    // [-2, -1] * [3, 4] should give [-8, -3]
    // ac=-2*3=-6, ad=-2*4=-8, bc=-1*3=-3, bd=-1*4=-4
    // min=-8, max=-3
    let a = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[-1.0]).into_dyn())
        .expect("invariant: test construction");
    let b = BoundedTensor::new(arr1(&[3.0]).into_dyn(), arr1(&[4.0]).into_dyn())
        .expect("invariant: test construction");
    let result = a.mul(&b).expect("invariant: test construction");

    // With +: ac=-2+3=1, ad=-2+4=2, bc=-1+3=2, bd=-1+4=3 → [1, 3] (wrong sign!)
    // With /: ac=-2/3=-0.67, ad=-2/4=-0.5, bc=-1/3=-0.33, bd=-1/4=-0.25 → [-0.67, -0.25] (wrong magnitude!)
    assert_eq!(
        result.lower()[[0]],
        -8.0,
        "min of products with negative interval"
    );
    assert_eq!(
        result.upper()[[0]],
        -3.0,
        "max of products with negative interval"
    );

    // [0, 1] * [0, 1] should give [0, 1]
    // Products: 0*0=0, 0*1=0, 1*0=0, 1*1=1
    let c = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("invariant: test construction");
    let result2 = c.mul(&c.clone()).expect("invariant: test construction");
    // With +: 0+0=0, 0+1=1, 1+0=1, 1+1=2 → [0, 2] (wrong upper!)
    assert_eq!(result2.lower()[[0]], 0.0);
    assert_eq!(
        result2.upper()[[0]],
        1.0,
        "[0,1]*[0,1] upper must be 1, not 2"
    );
}
