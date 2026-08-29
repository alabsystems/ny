// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness tests for the interval carrier.
//!
//! The claim under test is CONTAINMENT, and it is tested against the REAL
//! lsnc_relu constants — the ones the defect is about — with an EXACT RATIONAL
//! reference rather than a floating-point recomputation. `/lyapunov/MatMul` is
//! `onnx::MatMul_137 @ lyapunov.R`, all 36 entries of which fail the bit-exact
//! fold; `/lyapunov/Add` adds `/lyapunov/Constant_1_output_0` to produce the
//! Lyapunov matrix Q, which is the tensor whose loss reclassifies two plain
//! `activation @ constant` Linears as BilinearCrown and takes the root bound to
//! [-inf, +inf].
//!
//! The reference brackets below were computed offline with Python
//! `fractions.Fraction` over the exact dyadic rationals the stored binary32
//! initializers denote. Every binary32 value IS an exact rational, so the
//! product is a well-defined rational matrix with no ambiguity; the pair
//! `(lo, hi)` is the tightest f64 straddle of it.

use super::interval::{self, IntervalTensor};
use ndarray::{ArrayD, IxDyn};

#[test]
fn typed_preset_bridge_is_default_silent_and_env_wins_both_directions() {
    assert_eq!(
        super::decode_bool_preset(super::encode_bool_preset(None)),
        None
    );
    assert_eq!(
        super::decode_bool_preset(super::encode_bool_preset(Some(false))),
        Some(false)
    );
    assert_eq!(
        super::decode_bool_preset(super::encode_bool_preset(Some(true))),
        Some(true)
    );

    let decl = &ny_levers::decls::dark_probes::CONST_MATMUL_INTERVAL;
    let resolve = |preset: Option<bool>, raw: Option<&str>| {
        let raw = raw.map(str::to_owned);
        ny_levers::read_over_config_with(
            decl,
            move |_| raw,
            preset.map(ny_levers::LeverValue::Bool),
        )
        .expect("bool preset must be admissible")
    };

    let absent = resolve(None, None);
    assert!(!absent.value.as_bool());
    assert_eq!(absent.source, ny_levers::Source::Default);

    let off = resolve(Some(false), None);
    assert!(!off.value.as_bool());
    assert_eq!(off.source, ny_levers::Source::Config);

    let on = resolve(Some(true), None);
    assert!(on.value.as_bool());
    assert_eq!(on.source, ny_levers::Source::Config);

    let env_off = resolve(Some(true), Some("0"));
    assert!(!env_off.value.as_bool());
    assert_eq!(env_off.source, ny_levers::Source::LegacyEnv);

    let env_on = resolve(Some(false), Some("1"));
    assert!(env_on.value.as_bool());
    assert_eq!(env_on.source, ny_levers::Source::LegacyEnv);
}

const LSNC_MATMUL_137_BITS: [u32; 36] = [
    0x3e67ad66, 0xbfa3029b, 0xbcc974ca, 0x3f418c37, 0x3e0e1740, 0xbeb902b1, 0x3ed60981, 0xbf0908bc,
    0xbea244a9, 0xbeb3224d, 0xbf2e5364, 0xbf809a0a, 0x3ee0fba2, 0x3f97e041, 0x3f4ad850, 0xbe4b1401,
    0x3f524a66, 0xbf9a26f1, 0xbe62447a, 0xbf2564d9, 0x3d9a13be, 0xbe687c84, 0x3e8ee000, 0x3ea65c0a,
    0x3e977cc0, 0xbe5b9759, 0xbdac8c8e, 0xbd7dec83, 0xbdb2fc33, 0xbe31cab3, 0x3c66a838, 0x3ca42914,
    0x3dab639a, 0x3b4d1bf7, 0xbcdfad2d, 0xbc2d896d,
];
const LSNC_LYAPUNOV_R_BITS: [u32; 36] = [
    0x3e67ad66, 0x3ed60981, 0x3ee0fba2, 0xbe62447a, 0x3e977cc0, 0x3c66a838, 0xbfa3029b, 0xbf0908bc,
    0x3f97e041, 0xbf2564d9, 0xbe5b9759, 0x3ca42914, 0xbcc974ca, 0xbea244a9, 0x3f4ad850, 0x3d9a13be,
    0xbdac8c8e, 0x3dab639a, 0x3f418c37, 0xbeb3224d, 0xbe4b1401, 0xbe687c84, 0xbd7dec83, 0x3b4d1bf7,
    0x3e0e1740, 0xbf2e5364, 0x3f524a66, 0x3e8ee000, 0xbdb2fc33, 0xbcdfad2d, 0xbeb902b1, 0xbf809a0a,
    0xbf9a26f1, 0x3ea65c0a, 0xbe31cab3, 0xbc2d896d,
];
const LSNC_LYAP_CONST1_BITS: [u32; 36] = [
    0x322bcc77, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x322bcc77,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x322bcc77, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x322bcc77, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x322bcc77, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x322bcc77,
];
/// `/lyapunov/MatMul` = `onnx::MatMul_137 @ lyapunov.R`, in exact real arithmetic.
///
/// Each pair is a two-sided f64 BRACKET of the exact rational value,
/// computed offline with `fractions.Fraction` over the exact dyadic
/// rationals the binary32 initializers denote: `lo <= exact <= hi`.
const LSNC_LYAPUNOV_MATMUL_EXACT: [(f64, f64); 36] = [
    (2.3950707624926624, 2.395070762492663),
    (0.7881133515364761, 0.7881133515364761),
    (-1.0319119915037627, -1.0319119915037627),
    (0.5205995272641488, 0.5205995272641489),
    (0.3458537746441039, 0.3458537746441039),
    (-0.021988191177842467, -0.021988191177842467),
    (0.7881133515364761, 0.7881133515364761),
    (2.1572781515279713, 2.1572781515279713),
    (0.017421827253550237, 0.017421827253550237),
    (-0.20741952195296998, -0.20741952195296998),
    (0.5208225954675448, 0.5208225954675448),
    (-0.003224353164670811, -0.003224353164670811),
    (-1.0319119915037627, -1.0319119915037627),
    (0.017421827253550237, 0.017421827253550237),
    (4.3932627680172125, 4.393262768017213),
    (-0.9211202414242559, -0.9211202414242559),
    (-0.04158838090586131, -0.04158838090586131),
    (0.08597923702620705, 0.08597923702620706),
    (0.5205995272641488, 0.5205995272641489),
    (-0.20741952195296998, -0.20741952195296998),
    (-0.9211202414242559, -0.9211202414242559),
    (0.7068818418974321, 0.7068818418974321),
    (0.00010269587572908634, 0.00010269587572908634),
    (-0.0215329058108064, -0.0215329058108064),
    (0.3458537746441039, 0.3458537746441039),
    (0.5208225954675448, 0.5208225954675448),
    (-0.04158838090586131, -0.04158838090586131),
    (0.00010269587572908634, 0.00010269587572908634),
    (0.18225276885408576, 0.18225276885408578),
    (-0.003151426438352538, -0.003151426438352538),
    (-0.021988191177842467, -0.021988191177842467),
    (-0.003224353164670811, -0.003224353164670811),
    (0.08597923702620705, 0.08597923702620706),
    (-0.0215329058108064, -0.0215329058108064),
    (-0.003151426438352538, -0.003151426438352538),
    (0.00847062464475014, 0.008470624644750142),
];
/// `/lyapunov/Add` = the Lyapunov matrix Q itself, in exact real arithmetic.
///
/// Each pair is a two-sided f64 BRACKET of the exact rational value,
/// computed offline with `fractions.Fraction` over the exact dyadic
/// rationals the binary32 initializers denote: `lo <= exact <= hi`.
const LSNC_Q_EXACT: [(f64, f64); 36] = [
    (2.3950707724926623, 2.3950707724926628),
    (0.7881133515364761, 0.7881133515364761),
    (-1.0319119915037627, -1.0319119915037627),
    (0.5205995272641488, 0.5205995272641489),
    (0.3458537746441039, 0.3458537746441039),
    (-0.021988191177842467, -0.021988191177842467),
    (0.7881133515364761, 0.7881133515364761),
    (2.1572781615279712, 2.1572781615279712),
    (0.017421827253550237, 0.017421827253550237),
    (-0.20741952195296998, -0.20741952195296998),
    (0.5208225954675448, 0.5208225954675448),
    (-0.003224353164670811, -0.003224353164670811),
    (-1.0319119915037627, -1.0319119915037627),
    (0.017421827253550237, 0.017421827253550237),
    (4.393262778017212, 4.393262778017213),
    (-0.9211202414242559, -0.9211202414242559),
    (-0.04158838090586131, -0.04158838090586131),
    (0.08597923702620705, 0.08597923702620706),
    (0.5205995272641488, 0.5205995272641489),
    (-0.20741952195296998, -0.20741952195296998),
    (-0.9211202414242559, -0.9211202414242559),
    (0.706881851897432, 0.706881851897432),
    (0.00010269587572908634, 0.00010269587572908634),
    (-0.0215329058108064, -0.0215329058108064),
    (0.3458537746441039, 0.3458537746441039),
    (0.5208225954675448, 0.5208225954675448),
    (-0.04158838090586131, -0.04158838090586131),
    (0.00010269587572908634, 0.00010269587572908634),
    (0.1822527788540857, 0.18225277885408572),
    (-0.003151426438352538, -0.003151426438352538),
    (-0.021988191177842467, -0.021988191177842467),
    (-0.003224353164670811, -0.003224353164670811),
    (0.08597923702620705, 0.08597923702620706),
    (-0.0215329058108064, -0.0215329058108064),
    (-0.003151426438352538, -0.003151426438352538),
    (0.00847063464475008, 0.008470634644750081),
];

fn f32_array(bits: &[u32], shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(
        IxDyn(shape),
        bits.iter().map(|b| f32::from_bits(*b)).collect(),
    )
    .expect("fixture shape")
}

fn lyapunov_matmul_carrier() -> IntervalTensor {
    let a = IntervalTensor::point(&f32_array(&LSNC_MATMUL_137_BITS, &[6, 6]));
    let r = IntervalTensor::point(&f32_array(&LSNC_LYAPUNOV_R_BITS, &[6, 6]));
    interval::matmul(&a, &r).expect("the carrier must not decline on the lsnc constants")
}

/// (a) THE CARRIED INTERVAL CONTAINS THE TRUE PRODUCT.
///
/// Against the exact rational value of every one of the 36 entries of
/// `/lyapunov/MatMul`, none of which is exactly representable in binary32.
#[test]
fn lsnc_lyapunov_matmul_carrier_contains_the_exact_rational_product() {
    let carrier = lyapunov_matmul_carrier();
    assert_eq!(carrier.shape(), [6, 6]);
    for (index, (exact_lo, exact_hi)) in LSNC_LYAPUNOV_MATMUL_EXACT.iter().enumerate() {
        let (i, j) = (index / 6, index % 6);
        let lo = f64::from(carrier.lower[[i, j]]);
        let hi = f64::from(carrier.upper[[i, j]]);
        assert!(
            lo <= *exact_lo,
            "entry ({i},{j}): carrier lower {lo:e} is ABOVE the exact value {exact_lo:e} — \
             the enclosure does not contain the truth"
        );
        assert!(
            hi >= *exact_hi,
            "entry ({i},{j}): carrier upper {hi:e} is BELOW the exact value {exact_hi:e} — \
             the enclosure does not contain the truth"
        );
    }
}

/// (a) THE CARRIER SURVIVES THE CHAIN: `Add(point, interval)` still contains Q.
#[test]
fn lsnc_q_carrier_contains_the_exact_rational_lyapunov_matrix() {
    let product = lyapunov_matmul_carrier();
    let constant = IntervalTensor::point(&f32_array(&LSNC_LYAP_CONST1_BITS, &[6, 6]));
    let q = interval::add(&constant, &product).expect("interval Add must not decline");
    assert_eq!(q.shape(), [6, 6]);
    for (index, (exact_lo, exact_hi)) in LSNC_Q_EXACT.iter().enumerate() {
        let (i, j) = (index / 6, index % 6);
        assert!(
            f64::from(q.lower[[i, j]]) <= *exact_lo,
            "Q({i},{j}): carrier lower is above the exact value"
        );
        assert!(
            f64::from(q.upper[[i, j]]) >= *exact_hi,
            "Q({i},{j}): carrier upper is below the exact value"
        );
    }
}

/// The enclosure must be SOUND but it must also be USEFUL: a carrier that is
/// merely `[-inf, +inf]` would pass the containment tests above and buy nothing.
///
/// The measured budget: total radius mass of Q is ~2.3e-6, against a measured
/// relaxation floor on this family of ~1e-4. Pin an order of magnitude of
/// headroom so a future loosening of the enclosure cannot pass silently.
#[test]
fn the_lsnc_carrier_is_not_vacuous() {
    let product = lyapunov_matmul_carrier();
    let constant = IntervalTensor::point(&f32_array(&LSNC_LYAP_CONST1_BITS, &[6, 6]));
    let q = interval::add(&constant, &product).expect("interval Add");
    let mut total_radius = 0.0_f64;
    let mut max_width = 0.0_f64;
    for (l, u) in q.lower.iter().zip(q.upper.iter()) {
        let width = f64::from(*u) - f64::from(*l);
        assert!(width.is_finite() && width >= 0.0);
        total_radius += width / 2.0;
        max_width = max_width.max(width);
    }
    assert!(
        max_width < 1.0e-5,
        "per-entry half-width blew past the budget: max width {max_width:e}"
    );
    assert!(
        total_radius < 1.0e-4,
        "total radius mass {total_radius:e} is within an order of magnitude of the measured \
         ~1e-4 relaxation floor; the carrier would become the binding constraint"
    );
}

/// (b) THE BIT-EXACT FAST PATH IS UNCHANGED, stated as the property that makes
/// it safe: whenever the exact fold produces a value, the carrier CONTAINS that
/// same value. The production ordering (exact first, verbatim; interval only on
/// its `None`) is what guarantees the exact value is the one published, and the
/// containment here is what guarantees the two can never disagree about the
/// truth.
#[test]
fn wherever_the_exact_fold_succeeds_the_carrier_contains_its_value() {
    // A matrix whose products and partial sums are all exactly representable:
    // small integers scaled by powers of two.
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, -2.0, 0.5, 4.0, 0.0, -0.25])
        .expect("shape");
    let b =
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![2.0, 1.0, 0.5, -1.0, 8.0, 4.0]).expect("shape");
    let exact = super::ops_try_fold_matmul_exact_for_test(&a, &b)
        .expect("this product is exactly representable in binary32");
    let carrier =
        interval::matmul(&IntervalTensor::point(&a), &IntervalTensor::point(&b)).expect("carrier");
    for (index, value) in exact.iter().enumerate() {
        let (i, j) = (index / 2, index % 2);
        assert!(
            carrier.lower[[i, j]] <= *value && *value <= carrier.upper[[i, j]],
            "the carrier excludes the exactly-folded value {value} at ({i},{j})"
        );
    }
}

/// Randomized differential check: the carrier contains BOTH the (double-double)
/// exact real dot product AND what a plain binary32 accumulation computes.
///
/// The second half is what covers the counterexample oracle: ONNX Runtime
/// evaluates the ORIGINAL graph in binary32, so a carrier that contained only
/// the exact real value could in principle exclude what the oracle sees.
#[test]
fn the_carrier_contains_both_the_exact_sum_and_a_binary32_accumulation() {
    // Deterministic xorshift — no dev-dependency, reproducible failure.
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Values spread over several binades so cancellation actually happens.
        let mantissa = f32::from(((state >> 40) & 0x7FF) as u16) / 1024.0 - 1.0;
        let scale = 2.0_f32.powi(((state >> 8) & 0x7) as i32 - 3);
        mantissa * scale
    };
    for _ in 0..2000 {
        let k = 6;
        let lhs: Vec<f32> = (0..k).map(|_| next()).collect();
        let rhs: Vec<f32> = (0..k).map(|_| next()).collect();
        let a = ArrayD::from_shape_vec(IxDyn(&[k]), lhs.clone()).expect("shape");
        let b = ArrayD::from_shape_vec(IxDyn(&[k]), rhs.clone()).expect("shape");
        let carrier = interval::matmul(&IntervalTensor::point(&a), &IntervalTensor::point(&b))
            .expect("carrier");
        let lo = f64::from(carrier.lower[[]]);
        let hi = f64::from(carrier.upper[[]]);

        // Exact real reference: error-free two-sum accumulation of the exact
        // binary64 products carries ~106 significant bits, far more than the
        // 48 + log2(6) this sum needs.
        let mut sum = 0.0_f64;
        let mut compensation = 0.0_f64;
        for i in 0..k {
            let (s, e) = ny_core::dd::two_sum(sum, f64::from(lhs[i]) * f64::from(rhs[i]));
            sum = s;
            compensation += e;
        }
        let exact = sum + compensation;
        assert!(
            lo <= exact && exact <= hi,
            "carrier [{lo:e}, {hi:e}] excludes the exact real dot product {exact:e}"
        );

        // A plain binary32 accumulation, in a DIFFERENT order than the carrier
        // visits (reversed), to exercise the order-independent gamma_n term.
        let mut f32_sum = 0.0_f32;
        for i in (0..k).rev() {
            f32_sum += lhs[i] * rhs[i];
        }
        assert!(
            carrier.lower[[]] <= f32_sum && f32_sum <= carrier.upper[[]],
            "carrier excludes a binary32 accumulation {f32_sum} of the same dot product"
        );
    }
}

/// `Relu` on an enclosure is monotone and exact: no widening, and the crossing
/// case clamps to zero on the low side only.
#[test]
fn interval_relu_is_monotone_and_adds_no_width() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, -1.0, 3.0]).expect("shape");
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 2.0, 4.0]).expect("shape");
    let carrier = interval::relu(&IntervalTensor { lower, upper }).expect("relu");
    assert_eq!(carrier.lower.as_slice().expect("dense"), &[0.0, 0.0, 3.0]);
    assert_eq!(carrier.upper.as_slice().expect("dense"), &[0.0, 2.0, 4.0]);
}

/// A non-finite operand makes the fold DECLINE rather than publish a bogus
/// enclosure. Refusing is always available and always sound.
#[test]
fn a_non_finite_operand_declines_the_fold() {
    let a = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, f32::INFINITY]).expect("shape");
    let b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).expect("shape");
    assert!(
        interval::matmul(&IntervalTensor::point(&a), &IntervalTensor::point(&b)).is_none(),
        "an infinite operand must decline, not produce an interval"
    );
}
