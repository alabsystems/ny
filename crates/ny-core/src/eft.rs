// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! f32 error-free transformations (EFT) and the a-posteriori compensated
//! certified-error channel — increment 1 (CPU reference) of
//! `docs/EFT_COMPENSATED_CERTIFIED_ERROR_DESIGN.md`.
//!
//! The sound GPU CROWN fold today charges the A-PRIORI Higham worst-case
//! rounding bound `γ_k·(|A|·|W|)`, measured ~10⁴× above the ACTUAL f32 error
//! on the cifar100 resnet fold (certified deficit 0.088–0.126 vs actual
//! ~1e-5 — see the errprobe in `wide_alpha_true.rs`). The EFT channel instead
//! MEASURES the rounding error of the executed computation exactly:
//!
//! - [`two_prod_f32`]: `p = fl(a·b)`, `e = fma(a, b, −p)` with `a·b = p + e`
//!   EXACTLY (requires a fused, single-rounding FMA — `f32::mul_add` is
//!   guaranteed fused-semantics by Rust).
//! - [`two_sum_f32`] (Knuth, branch-free): `s = fl(a+b)`, residual `t` with
//!   `a + b = s + t` EXACTLY under round-to-nearest.
//!
//! For a dot product these telescope: `Σ a_i·b_i = value + Σ e_prod + Σ e_sum`
//! is an *identity*, so `R = Σ|e_prod| + Σ|e_sum|` bounds the actual error of
//! the executed fold, and the certified error becomes `R` (plus outward
//! rounding and underflow floors) instead of the no-cancellation worst case.
//!
//! Soundness preconditions (fail-closed via [`eft_self_check`], mirroring
//! `dd::dd_self_check`): fused FMA, round-to-nearest f32, and no FTZ/DAZ on
//! this compilation target. The GPU (WGSL) twin in increment 2 must run the
//! same probes ON DEVICE and refuse the channel (falling back to the Higham
//! bound unchanged) when they fail. Product underflow, where TwoProdFMA's
//! exactness theorem does not apply, is charged a per-term floor instead.

/// Knuth two-sum: `s = fl(a+b)` and the exact residual `t` with `a+b = s+t`.
///
/// Exact for all finite f32 (including subnormals) under round-to-nearest
/// with no FTZ. 6 flops, branch-free — the same shape the WGSL twin will use.
#[inline]
pub fn two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    let t = (a - (s - bb)) + (b - bb);
    (s, t)
}

/// Dekker/FMA two-prod: `p = fl(a·b)` and the exact residual `e = fma(a,b,−p)`
/// with `a·b = p + e`, provided the product does not underflow (see
/// [`prod_underflow_floor_f32`]) and the FMA is fused (single rounding).
#[inline]
pub fn two_prod_f32(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// Smallest positive NORMAL f32; below this the TwoProdFMA exactness theorem
/// can fail (the true residual may not be representable), so [`eft_dot_f32`]
/// charges this as a sound per-term floor instead of trusting the residual.
pub const PROD_UNDERFLOW_FLOOR_F32: f32 = f32::MIN_POSITIVE; // 2^-126

/// Fail-closed self-check for the f32 EFT preconditions on THIS target:
/// fused FMA (the residual probe distinguishes a two-rounding emulation),
/// round-to-nearest two-sum exactness, and residual sign preservation.
///
/// The WGSL twin must run the same probes on-device before the compensated
/// channel is trusted; any failure keeps the Higham channel byte-identically.
pub fn eft_self_check() -> Result<(), &'static str> {
    // FMA fusedness: with a = 1 + 2^-12, fl(a·a) = 1 + 2^-11 exactly (the
    // 2^-24 cross term is the last significand bit), so the fused residual is
    // exactly 2^-24 while a two-rounding emulation `fl(a·a) - p` cancels to 0.
    let a = 1.0f32 + f32::from_bits(0x3980_0000); // 1 + 2^-12
    let (p, e) = two_prod_f32(a, a);
    if e != f32::from_bits(0x3380_0000) {
        // 2^-24
        return Err("f32 FMA is not fused (two_prod residual wrong)");
    }
    if (a * a) - p != 0.0 {
        return Err("f32 multiply is not deterministic round-to-nearest");
    }
    // two_sum exactness on a residual f32 addition drops entirely.
    let tiny = f32::from_bits(0x0D80_0000); // 2^-100
    let (s, t) = two_sum_f32(1.0, tiny);
    if s != 1.0 || t != tiny {
        return Err("f32 two_sum is not exact (RN violated or FTZ active)");
    }
    // Subnormal honored (no DAZ/FTZ): the smallest subnormal must survive
    // an identity add.
    let sub = f32::from_bits(1); // 2^-149
    if sub + 0.0 != sub || sub == 0.0 {
        return Err("f32 subnormals are flushed (FTZ/DAZ active)");
    }
    Ok(())
}

/// A dot product's value (byte-identical to the plain left-to-right f32 fold)
/// with an a-posteriori certified error bound from the EFT residual channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EftDot {
    /// `fl(Σ a_i·b_i)` in index order — exactly what the uncompensated fold
    /// produces. The EFT channel never changes the value path.
    pub value: f32,
    /// Certified `|exact − value| ≤ err`: the exactly-measured residual sum,
    /// outward-rounded, plus underflow floors.
    pub err: f32,
}

/// Compensated dot: the plain f32 fold's value plus the EFT-certified error
/// of that exact fold. Returns `None` on any non-finite intermediate (caller
/// keeps its existing a-priori channel — fail-closed).
///
/// CPU reference contract (increment 1): the residual magnitudes are
/// accumulated in f64 — for n ≤ 2^30 terms the f64 accumulation of f32-sized
/// residuals is exact to well below 1 ulp(f32) relative, and the final
/// outward rounding charges it. The WGSL twin accumulates in f32 and must
/// charge `γ_{2n}·R` instead (increment 2).
pub fn eft_dot_f32(a: &[f32], b: &[f32]) -> Option<EftDot> {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    let mut resid = 0.0f64; // Σ|e_prod| + Σ|e_sum| (+ underflow floors)
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (p, ep) = two_prod_f32(x, y);
        if !p.is_finite() {
            return None;
        }
        // TwoProdFMA exactness needs the product away from underflow; charge
        // the sound floor there instead of trusting the residual. The
        // `x != 0 && y != 0` form (not `p != 0`) is soundness-critical: a
        // nonzero exact product can underflow ALL the way to `p == 0`, where
        // the residual channel would silently miss it.
        if x != 0.0 && y != 0.0 && p.abs() < PROD_UNDERFLOW_FLOOR_F32 {
            resid += PROD_UNDERFLOW_FLOOR_F32 as f64;
        } else {
            resid += (ep as f64).abs();
        }
        let (s, es) = two_sum_f32(acc, p);
        if !s.is_finite() {
            return None;
        }
        resid += (es as f64).abs();
        acc = s;
    }
    if !resid.is_finite() {
        return None;
    }
    // Outward-round the f64 residual sum into the f32 certified channel:
    // f64 accumulation error over 2n terms is < 2n·2^-53 relative (≪ 2^-30
    // for any fold NY runs), covered together with the f64→f32 rounding by
    // one next-up step after a (1 + 2^-30) inflation.
    let err64 = resid * (1.0 + f64::from_bits(0x3E10_0000_0000_0000)); // 1+2^-30
    let mut err = err64 as f32;
    if (err as f64) < err64 {
        err = f32::from_bits(err.to_bits() + 1); // next_up on the magnitude
    }
    if !err.is_finite() {
        return None;
    }
    Some(EftDot { value: acc, err })
}

/// The a-priori Higham worst-case comparator for the same dot:
/// `γ_{n+1}·Σ|a_i·b_i|` with `γ_k = k·u/(1−k·u)`, `u = 2^-24` — the bound the
/// sound fold charges today. Measurement-only (bench + tests).
pub fn higham_dot_err_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let u = f64::from_bits(0x3E70_0000_0000_0000); // 2^-24
    let ku = ((n + 1) as f64) * u;
    let gamma = if ku < 0.5 { ku / (1.0 - ku) } else { 2.0 * ku };
    let abs_sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((x as f64) * (y as f64)).abs())
        .sum();
    (gamma * abs_sum) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};
    use proptest::prelude::*;

    fn rat(x: f32) -> BigRational {
        BigRational::from_float(x as f64).expect("finite f32 is a rational")
    }

    fn exact_dot(a: &[f32], b: &[f32]) -> BigRational {
        let mut s = BigRational::zero();
        for (&x, &y) in a.iter().zip(b.iter()) {
            s += rat(x) * rat(y);
        }
        s
    }

    /// The whole module is unsound if this fails; it must run first and loudly.
    #[test]
    fn self_check_passes_on_this_target() {
        assert_eq!(
            eft_self_check(),
            Ok(()),
            "f32 error-free transformations are broken on this target; the \
             compensated certified-error channel would be silently unsound"
        );
    }

    /// A fail-closed check that never fires is worthless: each probe must
    /// DISCRIMINATE against the broken transform it guards (dd.rs idiom).
    #[test]
    fn self_check_probes_discriminate() {
        // Fused vs two-rounding emulation on the probe case.
        let a = 1.0f32 + f32::from_bits(0x3980_0000); // 1 + 2^-12
        let p = a * a;
        let fused = a.mul_add(a, -p);
        let emulated = (a * a) - p;
        assert_eq!(fused, f32::from_bits(0x3380_0000), "2^-24");
        assert_eq!(emulated, 0.0);
        assert_ne!(fused, emulated, "the FMA probe must discriminate");

        // two_sum residual vs what reassociation would fold it to.
        let tiny = f32::from_bits(0x0D80_0000); // 2^-100
        let (_s, t) = two_sum_f32(1.0, tiny);
        assert_eq!(t, tiny);
        assert_ne!(t, 0.0, "the two_sum probe must discriminate");
    }

    #[test]
    fn two_sum_exact_against_rationals_on_adversarial_pairs() {
        let cases: &[(f32, f32)] = &[
            (1.0, f32::from_bits(0x0D80_0000)),
            (1e30, -1e30),
            (1e30, 1.0),
            (3.0, 1.0 / 3.0),
            (f32::MIN_POSITIVE, -f32::from_bits(1)),
            (f32::from_bits(1), f32::from_bits(1)), // smallest-subnormal pair
        ];
        for &(a, b) in cases {
            let (s, t) = two_sum_f32(a, b);
            assert_eq!(
                rat(a) + rat(b),
                rat(s) + rat(t),
                "two_sum must be EXACT: a={a:e} b={b:e}"
            );
        }
    }

    #[test]
    fn two_prod_exact_against_rationals_away_from_underflow() {
        let cases: &[(f32, f32)] = &[
            (3.0, 1.0 / 3.0),
            (
                1.0 + f32::from_bits(0x3980_0000),
                1.0 - f32::from_bits(0x3980_0000),
            ),
            (1e10, 1e-10),
            (-7.0, 0.142_857_15),
        ];
        for &(a, b) in cases {
            let (p, e) = two_prod_f32(a, b);
            assert_eq!(
                rat(a) * rat(b),
                rat(p) + rat(e),
                "two_prod must be EXACT: a={a:e} b={b:e}"
            );
        }
    }

    // THE soundness property: the exact rational dot lies within the
    // certified error of the plain-fold value, on adversarial mixes of
    // magnitude and cancellation.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]
        #[test]
        fn certified_error_encloses_exact_dot(
            pairs in proptest::collection::vec(
                (
                    prop_oneof![
                        -1e3f32..1e3f32,
                        -1e-3f32..1e-3f32,
                        -1e6f32..1e6f32,
                        Just(0.0f32),
                    ],
                    prop_oneof![
                        -1e3f32..1e3f32,
                        -1e-6f32..1e-6f32,
                        Just(1.0f32),
                        Just(-1.0f32),
                    ],
                ),
                0..300,
            )
        ) {
            let a: Vec<f32> = pairs.iter().map(|p| p.0).collect();
            let b: Vec<f32> = pairs.iter().map(|p| p.1).collect();
            let Some(EftDot { value, err }) = eft_dot_f32(&a, &b) else {
                return Ok(()); // non-finite refusal is fail-closed by design
            };
            let exact = exact_dot(&a, &b);
            let diff = (exact - rat(value)).abs();
            prop_assert!(
                diff <= rat(err),
                "certified enclosure violated: |exact - value| = {} > err = {err:e}",
                diff,
            );
        }
    }

    /// Cancellation-heavy structured case: the value path must be
    /// byte-identical to the plain fold, and the certified error must land
    /// FAR below the a-priori Higham bound (the design's entire point).
    /// Mixed-sign coefficient rows are exactly the CROWN fold's regime.
    #[test]
    fn compensated_error_beats_higham_by_orders_on_cancellation_heavy_dot() {
        // Deterministic LCG so the test is reproducible.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let n = 4096usize;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for _ in 0..n {
            // magnitudes ~U[0.5, 2), signs alternating pseudo-randomly:
            // large |A|·|W| mass, small running sums — the CROWN regime.
            let ra = 0.5 + (next() % 1_000_000) as f32 / 666_667.0;
            let rb = 0.5 + (next() % 1_000_000) as f32 / 666_667.0;
            let sa = if next() & 1 == 0 { 1.0 } else { -1.0 };
            let sb = if next() & 1 == 0 { 1.0 } else { -1.0 };
            a.push(sa * ra);
            b.push(sb * rb);
        }
        let eft = eft_dot_f32(&a, &b).expect("finite");

        // Value path is byte-identical to the plain fold.
        let mut plain = 0.0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            plain += x * y;
        }
        assert_eq!(eft.value.to_bits(), plain.to_bits());

        // Certified enclosure holds...
        let exact = exact_dot(&a, &b);
        let diff = (exact - rat(eft.value)).abs();
        assert!(diff <= rat(eft.err));

        // ...and the compensated bound is orders below the a-priori one.
        let higham = higham_dot_err_f32(&a, &b);
        assert!(
            eft.err < higham / 50.0,
            "expected ≥50x tightening on the cancellation-heavy regime, got \
             eft={:e} vs higham={:e} (ratio {:.1}x)",
            eft.err,
            higham,
            higham / eft.err,
        );
    }

    /// Underflow-range products take the sound floor path and still enclose.
    #[test]
    fn underflow_products_are_floored_soundly() {
        let a = vec![1.0e-30f32, -1.0e-30, 2.0e-25];
        let b = vec![1.0e-20f32, 1.0e-20, -3.0e-25];
        let EftDot { value, err } = eft_dot_f32(&a, &b).expect("finite");
        let exact = exact_dot(&a, &b);
        let diff = (exact - rat(value)).abs();
        assert!(diff <= rat(err), "underflow floor must keep the enclosure");
        // The floors themselves stay negligible at fold scale.
        assert!(err < 1.0e-20);
    }

    /// Guard the exactness claim used in the doc: value + residuals telescopes
    /// to the exact dot when no product underflows (identity, not a bound).
    #[test]
    fn residual_telescoping_is_an_identity_away_from_underflow() {
        let a = [3.0f32, -1.0e4, 0.125, 7.5];
        let b = [1.0 / 3.0, 1.0e-4, 8.0, -2.5];
        let mut acc = 0.0f32;
        let mut correction = BigRational::zero();
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (p, ep) = two_prod_f32(x, y);
            let (s, es) = two_sum_f32(acc, p);
            correction += rat(ep) + rat(es);
            acc = s;
        }
        assert_eq!(
            exact_dot(&a, &b),
            rat(acc) + correction,
            "Σ a·b = value + Σe_prod + Σe_sum must hold EXACTLY"
        );
    }

    /// num-bigint is in the dep tree for the oracle; touch it so an unused-dep
    /// lint never bites (and BigInt exactness backs BigRational).
    #[test]
    fn rational_oracle_is_exact_backed() {
        assert_eq!(BigInt::from(1i32) + BigInt::from(2i32), BigInt::from(3i32));
    }
}
