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
//! Soundness preconditions (fail-closed via [`eft_self_check`], cached by
//! [`eft_available`], mirroring `dd_selfcheck`): fused FMA, round-to-nearest
//! f32, and no FTZ/DAZ on this compilation target. The GPU (WGSL) twin runs the
//! same probes ON DEVICE and refuses the channel (falling back to the Higham
//! bound unchanged) when they fail. Product underflow, where TwoProdFMA's
//! exactness theorem does not apply, is charged a per-term floor instead — the
//! guard for that path is [`TWO_PROD_EXACT_FLOOR_F32`], NOT the charged floor.
//!
//! # The shipped contract
//!
//! S2's safety argument is `max(lb_higham, lb_eft)`, which on the certified
//! RADII is `min`. It lives in exactly one place, [`combine_downgrade_only`]
//! (plus its f64 twin), and callers should reach for
//! [`eft_dot_f32_downgrade_only`] rather than combining by hand: that function
//! owns the value and both arms, so the "both radii must certify the same
//! value" precondition cannot be violated. `min` is sound **iff both arms are
//! sound**, so [`higham_dot_err_f32`] carries an absolute underflow floor of its
//! own — a purely relative charge is not enclosing in the subnormal range, and
//! `min` would preferentially publish it.

use crate::dd::{next_up_f64, two_sum};

#[inline]
fn add_nonnegative_f64_up(accumulator: f64, term: f64) -> f64 {
    let (sum, residual) = two_sum(accumulator, term);
    if residual > 0.0 {
        next_up_f64(sum)
    } else {
        sum
    }
}

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

/// Smallest positive NORMAL f32 — the per-term error CHARGED whenever the
/// TwoProdFMA exactness theorem is unavailable (see
/// [`TWO_PROD_EXACT_FLOOR_F32`], which is the *guard* that selects this path).
///
/// SOUND as a charge because the guard fires only for `|p| < 2^-101`, and there
/// `|a·b − p| ≤ ½·ulp(p) ≤ ½·2^-125 = 2^-126` (the largest binade admitted by
/// the guard is `[2^-102, 2^-101)`, whose ulp is `2^-125`). Below the normal
/// range the rounding error is `≤ 2^-150`, smaller still.
pub const PROD_UNDERFLOW_FLOOR_F32: f32 = f32::MIN_POSITIVE; // 2^-126

/// The TwoProdFMA exactness GUARD: `fma(a, b, −fl(a·b))` returns the residual
/// EXACTLY only when `e_a + e_b ≥ e_min + p − 1 = −126 + 24 − 1 = −103`.
///
/// If `e_a + e_b ≤ −104` then `|a·b| < 4·2^-104 = 2^-102`, so requiring
/// `|p| ≥ 2^-101` implies the theorem's hypothesis with a full binade to spare.
/// In the band `|p| ∈ [2^-126, 2^-101)` the residual is itself rounded (often
/// all the way to `0`) — using `f32::MIN_POSITIVE` as the guard, as this module
/// did before 2026-08-02, therefore published a radius that does NOT enclose.
/// Minimal witness (from the exact-rational oracle in
/// `tests/eft_certified_error_exact_rational_oracle.rs`): `a = 1 + 2^-23`,
/// `b = 2^-126·(1 + 2^-23)` gives `e = 0` and `err = 0` while the exact product
/// exceeds the folded value by `2^-172`.
///
/// Raising the guard is a STRICT WIDENING: every term it newly diverts is
/// charged [`PROD_UNDERFLOW_FLOOR_F32`] `= 2^-126`, which dominates the `|e|`
/// that band could ever produce.
pub const TWO_PROD_EXACT_FLOOR_F32: f32 = f32::from_bits(0x0D00_0000); // 2^-101

/// Fail-closed self-check for the f32 EFT preconditions on THIS target:
/// fused FMA (the residual probe distinguishes a two-rounding emulation),
/// round-to-nearest two-sum exactness, and residual sign preservation.
///
/// The WGSL twin must run the same probes on-device before the compensated
/// channel is trusted; any failure keeps the Higham channel byte-identically.
///
/// `#[inline(never)]` + [`std::hint::black_box`] on every probe operand, exactly
/// as `dd_selfcheck::run_probes` does: the probe values are compile-time
/// constants, so without the barriers a constant-folding pass can evaluate the
/// probe with exact semantics while the runtime kernel is reassociated — the
/// probe would pass on a target where the channel is broken.
#[inline(never)]
pub fn eft_self_check() -> Result<(), &'static str> {
    use std::hint::black_box;

    // FMA fusedness: with a = 1 + 2^-12, fl(a·a) = 1 + 2^-11 exactly (the
    // 2^-24 cross term is the last significand bit), so the fused residual is
    // exactly 2^-24 while a two-rounding emulation `fl(a·a) - p` cancels to 0.
    let a = black_box(1.0f32 + f32::from_bits(0x3980_0000)); // 1 + 2^-12
    let (p, e) = two_prod_f32(black_box(a), black_box(a));
    if e != f32::from_bits(0x3380_0000) {
        // 2^-24
        return Err("f32 FMA is not fused (two_prod residual wrong)");
    }
    if black_box(a * a) - black_box(p) != 0.0 {
        return Err("f32 multiply is not deterministic round-to-nearest");
    }
    // two_sum exactness on a residual f32 addition drops entirely.
    let tiny = black_box(f32::from_bits(0x0D80_0000)); // 2^-100
    let (s, t) = two_sum_f32(black_box(1.0f32), tiny);
    if s != 1.0 || t != tiny {
        return Err("f32 two_sum is not exact (RN violated or FTZ active)");
    }
    // Subnormal honored (no DAZ/FTZ): the smallest subnormal must survive
    // an identity add. The absolute underflow floors charged by BOTH arms of
    // this module assume gradual underflow (`η ≤ 2^-150` per rounding); under
    // FTZ the loss is `2^-126` per rounding and neither floor would cover it.
    let sub = black_box(f32::from_bits(1)); // 2^-149
    if black_box(sub + 0.0) != sub || sub == 0.0 {
        return Err("f32 subnormals are flushed (FTZ/DAZ active)");
    }
    Ok(())
}

/// Cached, process-wide authorization for the compensated channel.
///
/// Mirrors `dd_selfcheck::dd_selfcheck_ok`: caching is not (only) an
/// optimization — it guarantees that two consumers in the same process can
/// never disagree about whether the channel is authorized. Every consumer MUST
/// consult this and keep its incumbent a-priori channel when it is `false`.
#[must_use]
pub fn eft_available() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| eft_self_check().is_ok())
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
/// CPU reference contract (increment 1): residual magnitudes are accumulated
/// in f64 with every addition rounded toward +∞, then converted outward to
/// f32. The enclosure is therefore independent of the fold length. The WGSL
/// twin accumulates in f32 and must charge `γ_{2n}·R` instead (increment 2).
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
        //
        // The guard is [`TWO_PROD_EXACT_FLOOR_F32`] (2^-101), NOT the charged
        // floor 2^-126: in the band between them the residual is itself
        // rounded and the published radius did not enclose (fixed 2026-08-02;
        // witness pinned in the exact-rational oracle test).
        if x != 0.0 && y != 0.0 && p.abs() < TWO_PROD_EXACT_FLOOR_F32 {
            resid = add_nonnegative_f64_up(resid, PROD_UNDERFLOW_FLOOR_F32 as f64);
        } else {
            resid = add_nonnegative_f64_up(resid, (ep as f64).abs());
        }
        let (s, es) = two_sum_f32(acc, p);
        if !s.is_finite() {
            return None;
        }
        resid = add_nonnegative_f64_up(resid, (es as f64).abs());
        acc = s;
    }
    if !resid.is_finite() {
        return None;
    }
    // Every non-negative f64 addition above is rounded toward +∞. This is
    // dimension-independent: unlike a fixed relative inflation, it remains an
    // enclosure even for the documented n <= 2^30 contract.
    let err = publish_directed_err_up_f32(resid)?;
    Some(EftDot { value: acc, err })
}

/// Publish an already upward-directed non-negative f64 reduction as f32.
///
/// Unlike [`publish_err_up_f32`], this needs no fixed relative inflation: every
/// source addition was already rounded toward +∞ by
/// [`add_nonnegative_f64_up`].
#[inline]
fn publish_directed_err_up_f32(term: f64) -> Option<f32> {
    if !term.is_finite() || term < 0.0 {
        return None;
    }
    let mut err = term as f32;
    if (err as f64) < term {
        err = f32::from_bits(err.to_bits() + 1);
    }
    err.is_finite().then_some(err)
}

/// Publish a non-negative f64 certificate term as an f32 that is `≥` it.
///
/// Inflates by `(1 + 2^-30)` — covering the f64 accumulation error of the
/// residual sum itself, `< 2n·2^-53` relative, which is `≪ 2^-30` for any fold
/// NY runs — then rounds the f64→f32 cast OUTWARD. Both steps are required:
/// dropping the `next_up` is one of the mutants the exact-rational oracle
/// catches (`Inject::NoOutwardRounding`).
///
/// Returns `None` on a non-finite input or a term that overflows f32, so every
/// caller fails closed onto its own a-priori channel.
#[inline]
fn publish_err_up_f32(term: f64) -> Option<f32> {
    if !term.is_finite() || term < 0.0 {
        return None;
    }
    let inflated = term * (1.0 + f64::from_bits(0x3E10_0000_0000_0000)); // 1+2^-30
    if !inflated.is_finite() {
        return None;
    }
    let mut err = inflated as f32;
    if (err as f64) < inflated {
        err = f32::from_bits(err.to_bits() + 1); // next_up on the magnitude
    }
    if !err.is_finite() {
        return None;
    }
    Some(err)
}

/// The a-priori Higham worst-case comparator for the same dot:
/// `γ_{n+1}·Σ|a_i·b_i| + underflow floor`, with `γ_k = k·u/(1−k·u)`,
/// `u = 2^-24` — the shape of the charge the sound folds carry today.
///
/// # Why the absolute floor is not optional
///
/// `γ·Σ|a_i b_i|` is a purely RELATIVE model, and Higham's Thm 3.1 assumes NO
/// underflow. Under gradual underflow each of the `≤ 2n` roundings contributes
/// an ABSOLUTE `|η| ≤ 2^-150` that no relative multiple of a subnormal-scale
/// `Σ|ab|` can cover, so the bare relative form is NOT enclosing there — caught
/// by the exact-rational oracle (`higham_arm_encloses_under_underflow`, which
/// was a *characterization of the defect* before this floor landed on
/// 2026-08-02 and is its regression gate now). This matters
/// for [`combine_downgrade_only`], which preferentially publishes the SMALLER of
/// the two arms and would therefore publish the broken one.
///
/// The charge is `4n·2^-149`: `2n` roundings (n products, n accumulating adds),
/// each `≤ 2^-150 = ½·2^-149`, times the `1/(1 − γ) ≤ 2` amplification the
/// `γ < 1/2` admission guard below bounds. Both factors are operation counts of
/// this dot, not tuned constants.
///
/// Returns `f32::INFINITY` (sound, maximally loose) rather than a smaller
/// number whenever the growth factor degenerates — `(n+1)·u ≥ 1/2`, where
/// `k·u/(1 − k·u)` exceeds the `2·k·u` approximation this function used to
/// return.
///
/// PRECONDITION (shared with the whole module, verified by [`eft_self_check`]):
/// gradual underflow. Under FTZ each rounding loses up to `2^-126`, not
/// `2^-150`, and this floor does not cover it.
pub fn higham_dot_err_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let u = f64::from_bits(0x3E70_0000_0000_0000); // 2^-24
    let ku = ((n + 1) as f64) * u;
    // NaN-aware "not (ku < 0.5)": `ku >= 0.5` would let a NaN through.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(ku < 0.5) {
        return f32::INFINITY;
    }
    let gamma = ku / (1.0 - ku);
    let mut abs_sum = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        // f32×f32 is EXACT in f64 (48 < 53 significand bits); only the sum
        // rounds, and `publish_err_up_f32`'s (1 + 2^-30) inflation covers it.
        abs_sum += ((x as f64) * (y as f64)).abs();
    }
    // ETA = 2^-149, the smallest positive f32 subnormal (exactly widened).
    let eta = f64::from(f32::from_bits(1));
    let floor = 4.0 * (n as f64) * eta;
    publish_err_up_f32(gamma * abs_sum + floor).unwrap_or(f32::INFINITY)
}

/// **THE downgrade-only contract**, in one place, for f32 certified radii.
///
/// `S2`'s safety argument is `max(lb_higham, lb_eft)` on the induced LOWER
/// BOUND `value − err`; on the RADII that is `min`, and applying `max` to the
/// radii instead is a strict no-op that recovers nothing (pinned by
/// `max_on_radii_is_sound_but_recovers_nothing`). Routing every combination
/// through this function is what makes the contract STRUCTURAL:
///
/// * the result is always one of the two inputs — this function cannot
///   synthesize a radius, so it cannot synthesize a narrower one;
/// * the result is never greater than `err_higham`, so the certified RADIUS can
///   only shrink. Be precise about what that means: a smaller certified radius
///   is a TIGHTER bound, and a tighter bound is sound only if the EFT arm is
///   itself a valid bound on the same error. This function guarantees the
///   direction of travel, never the validity of the arm it travels toward —
///   that is the caller's obligation, stated as the precondition below;
/// * a refused, non-finite, or negative EFT arm returns `err_higham`
///   **bit-identically**, so a channel that cannot be computed leaves the
///   a-priori charge exactly as it stood.
///
/// SOUNDNESS PRECONDITION, and it is the caller's whole obligation: both radii
/// must certify the SAME published `value`. `min` of two radii around different
/// values is meaningless. Prefer [`eft_dot_f32_downgrade_only`], which owns the
/// value and both arms so the precondition cannot be violated.
#[inline]
#[must_use]
pub fn combine_downgrade_only(err_higham: f32, err_eft: f32) -> f32 {
    // NaN/negative/infinite EFT arm ⇒ keep the incumbent, byte-identically.
    // Written as an explicit comparison rather than `f32::min` so that a NaN
    // `err_higham` also survives unchanged into the caller's own guards.
    // `is_sign_positive()`, NOT `>= 0.0`: IEEE says `-0.0 >= 0.0` is TRUE, and
    // `-0.0 < err_higham` is also true, so a `-0.0` arm would be published as the
    // certified radius and ZERO the entire error charge — a false-proof
    // generator. `+0.0` still passes, so a genuinely zero error is unaffected.
    if err_eft.is_finite() && err_eft.is_sign_positive() && err_eft < err_higham {
        err_eft
    } else {
        err_higham
    }
}

/// f64 twin of [`combine_downgrade_only`], for the certificate arithmetic that
/// runs in f64 before its final directed publication (the conv/Linear CROWN
/// error matrices). Identical contract, identical fail-closed direction.
#[inline]
#[must_use]
pub fn combine_downgrade_only_f64(err_higham: f64, err_eft: f64) -> f64 {
    // See `combine_downgrade_only`: `-0.0 >= 0.0` is true and would zero the charge.
    if err_eft.is_finite() && err_eft.is_sign_positive() && err_eft < err_higham {
        err_eft
    } else {
        err_higham
    }
}

/// The shipped S2 channel for one dot: the plain f32 fold's value, carrying
/// `min(higham, eft)` as its certified radius.
///
/// This is the form callers should use, because it makes
/// [`combine_downgrade_only`]'s same-value precondition unviolatable: the
/// value, the a-priori arm and the a-posteriori arm all come from THIS fold.
///
/// Fails closed to `None` — leaving the caller's own a-priori channel untouched
/// — when the target's EFT preconditions do not hold ([`eft_available`]) or the
/// fold is non-finite. `None` must never be read as "zero error".
///
/// GUARANTEES (pinned by tests):
/// * `value` is bit-identical to the plain left-to-right f32 fold;
/// * `err ≤ higham_dot_err_f32(a, b)`;
/// * `|Σ a_i·b_i − value| ≤ err` exactly (exact-rational oracle).
pub fn eft_dot_f32_downgrade_only(a: &[f32], b: &[f32]) -> Option<EftDot> {
    if !eft_available() {
        return None;
    }
    let eft = eft_dot_f32(a, b)?;
    let higham = higham_dot_err_f32(a, b);
    Some(EftDot {
        value: eft.value,
        err: combine_downgrade_only(higham, eft.err),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};
    use proptest::prelude::*;

    fn rat(x: f32) -> BigRational {
        BigRational::from_float(x).expect("finite f32 is a rational")
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
    fn residual_reduction_cannot_use_a_fixed_relative_inflation() {
        let term = 2.0_f64.powi(-54);
        assert_eq!(1.0 + term, 1.0, "nearest-f64 addition loses this term");

        // 2^25 such non-negative f32-representable residuals are within the
        // documented n <= 2^30 contract, yet their lost mass exceeds the old
        // fixed 2^-30 inflation.
        let exact = 1.0 + ((1_u64 << 25) as f64) * term;
        let legacy_envelope = 1.0 * (1.0 + 2.0_f64.powi(-30));
        assert!(legacy_envelope < exact);

        // The production reducer now directs every addition upward, so even a
        // term lost by round-to-nearest immediately enlarges the enclosure.
        let directed = add_nonnegative_f64_up(1.0, term);
        assert!(directed >= 1.0 + term);
        assert!(directed > 1.0);
        assert_eq!(add_nonnegative_f64_up(1.0, 0.0), 1.0);
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
            let EftDot { value, err } = eft_dot_f32(&a, &b)
                .expect("the bounded finite strategy must publish an EFT enclosure");
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

    // -----------------------------------------------------------------------
    // The two defects the exact-rational oracle caught, now pinned as FIXED.
    // -----------------------------------------------------------------------

    /// The constants must stay in the relationship the exactness derivation
    /// needs: the GUARD is `2^-101`, the CHARGE is `2^-126`, guard > charge.
    #[test]
    fn two_prod_guard_and_charge_are_the_derived_constants() {
        assert_eq!(TWO_PROD_EXACT_FLOOR_F32, 2f32.powi(-101));
        assert_eq!(PROD_UNDERFLOW_FLOOR_F32, 2f32.powi(-126));
        const { assert!(TWO_PROD_EXACT_FLOOR_F32 > PROD_UNDERFLOW_FLOOR_F32) };
    }

    /// The minimal witness from the oracle: `a = 1 + 2^-23`,
    /// `b = 2^-126·(1 + 2^-23)`. The product is NORMAL (so the old `2^-126`
    /// guard let it through) but below the TwoProdFMA exactness threshold, so
    /// the residual is `0` and the old channel certified `err = 0` while the
    /// exact product exceeds the value.
    #[test]
    fn tiny_normal_product_is_charged_the_floor_not_a_rounded_residual() {
        let a = 1.0f32 + f32::from_bits(0x3400_0000); // 1 + 2^-23
        let b = f32::MIN_POSITIVE * (1.0f32 + f32::from_bits(0x3400_0000));
        let (p, e) = two_prod_f32(a, b);
        assert!(
            p.abs() >= PROD_UNDERFLOW_FLOOR_F32 && p.abs() < TWO_PROD_EXACT_FLOOR_F32,
            "the witness must sit inside the previously-unguarded band"
        );
        assert_eq!(
            e, 0.0,
            "this is why the band was unsound: the residual is 0"
        );

        let EftDot { value, err } = eft_dot_f32(&[a], &[b]).expect("finite");
        assert!(
            err >= PROD_UNDERFLOW_FLOOR_F32,
            "the raised guard must charge the 2^-126 floor, got {err:e}"
        );
        let exact = rat(a) * rat(b);
        assert!(
            (exact - rat(value)).abs() <= rat(err),
            "the witness must now enclose"
        );
    }

    /// The Higham comparator arm must also enclose under gradual underflow —
    /// `min` publishes the SMALLER arm, so a broken comparator is a broken
    /// channel.
    #[test]
    fn higham_arm_encloses_under_gradual_underflow() {
        let a = vec![f32::from_bits(3), 1.5e-30f32];
        let b = vec![1.0f32 + f32::from_bits(0x3400_0000), 1.5e-15];
        let mut value = 0.0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            value += x * y;
        }
        let higham = higham_dot_err_f32(&a, &b);
        let exact = exact_dot(&a, &b);
        assert!(
            (exact - rat(value)).abs() <= rat(higham),
            "the Higham arm is not enclosing under underflow: value={value:e} \
             err={higham:e}"
        );
    }

    /// A degenerate growth factor must widen to infinity, never to the smaller
    /// `2·k·u` approximation (which is BELOW `k·u/(1−k·u)` once `k·u ≥ 1/2`).
    #[test]
    fn higham_arm_refuses_rather_than_undercharging_a_degenerate_gamma() {
        // n + 1 = 2^23 makes (n+1)·2^-24 = 1/2 exactly, the first degenerate
        // width. One buffer, aliased on both sides, keeps the fixture at 32 MiB.
        let n = (1usize << 23) - 1;
        let a = vec![0.0f32; n];
        assert_eq!(
            higham_dot_err_f32(&a, &a),
            f32::INFINITY,
            "a degenerate γ must publish +inf, not a smaller approximation"
        );
    }

    // -----------------------------------------------------------------------
    // The downgrade-only contract
    // -----------------------------------------------------------------------

    /// Structural property 1: the combinator can only ever RETURN one of its
    /// two inputs, and never one larger than the incumbent.
    #[test]
    fn combinator_returns_an_input_and_never_exceeds_the_incumbent() {
        let radii = [
            0.0f32,
            f32::MIN_POSITIVE,
            1e-30,
            1e-7,
            1.0,
            3.4e38,
            f32::INFINITY,
            f32::NAN,
            -1.0,
            -0.0,
        ];
        for &h in &radii {
            for &e in &radii {
                let out = combine_downgrade_only(h, e);
                assert!(
                    out.to_bits() == h.to_bits() || out.to_bits() == e.to_bits(),
                    "combinator synthesized a radius: h={h:e} e={e:e} -> {out:e}"
                );
                if h.is_finite() {
                    assert!(
                        out <= h,
                        "combinator weakened the certified bound: h={h:e} e={e:e}"
                    );
                }
            }
        }
    }

    /// Structural property 2: EVERY refusal shape degrades to the incumbent
    /// BIT-identically, so a channel that cannot be computed changes nothing.
    #[test]
    fn combinator_refusal_is_bit_identical_to_the_incumbent() {
        for h in [1e-7f32, 1.0, 3.4e38, 0.0, f32::MIN_POSITIVE] {
            for broken in [f32::NAN, f32::INFINITY, -1.0f32, f32::NEG_INFINITY] {
                assert_eq!(
                    combine_downgrade_only(h, broken).to_bits(),
                    h.to_bits(),
                    "refusal must be byte-identical: h={h:e} broken={broken:e}"
                );
                assert_eq!(
                    combine_downgrade_only_f64(f64::from(h), f64::from(broken)).to_bits(),
                    f64::from(h).to_bits(),
                );
            }
        }
    }

    /// THE test the brief demands: a case where the EFT arm is LOOSER, so the
    /// a-priori Higham charge must win. An all-same-sign fold has no
    /// cancellation for the a-posteriori channel to exploit, and the residual
    /// sum is inflated and outward-rounded, so it can land ABOVE the relative
    /// bound on short folds.
    #[test]
    fn higham_wins_when_the_eft_arm_is_looser() {
        // A one-term fold whose product is in the floored band: the EFT arm
        // charges the full 2^-126 floor while Higham charges γ_2·|a·b| ≈
        // 2^-23·2^-126 — six orders tighter. The combinator must publish
        // Higham, unchanged.
        let a = [1.0f32 + f32::from_bits(0x3400_0000)];
        let b = [f32::MIN_POSITIVE * (1.0f32 + f32::from_bits(0x3400_0000))];
        let eft = eft_dot_f32(&a, &b).expect("finite");
        let higham = higham_dot_err_f32(&a, &b);
        assert!(
            eft.err > higham,
            "the fixture is not exercising the EFT-looser branch: \
             eft={:e} higham={higham:e}",
            eft.err
        );
        let combined = combine_downgrade_only(higham, eft.err);
        assert_eq!(
            combined.to_bits(),
            higham.to_bits(),
            "Higham must win when the EFT arm is looser"
        );
        // And the published channel agrees.
        let shipped = eft_dot_f32_downgrade_only(&a, &b).expect("channel available");
        assert_eq!(shipped.err.to_bits(), higham.to_bits());
        assert_eq!(shipped.value.to_bits(), eft.value.to_bits());
        // Still enclosing with the tighter arm.
        let exact = exact_dot(&a, &b);
        assert!((exact - rat(shipped.value)).abs() <= rat(shipped.err));
    }

    // The shipped channel never weakens the incumbent, keeps the value path
    // byte-identical, and still encloses — over the cancellation regime S2
    // targets AND over adversarial magnitude mixes.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        #[test]
        fn shipped_channel_is_downgrade_only_and_encloses(
            pairs in proptest::collection::vec(
                (
                    prop_oneof![
                        -1e3f32..1e3f32,
                        -1e-3f32..1e-3f32,
                        -1e6f32..1e6f32,
                        (1u32..=0x007f_ffffu32).prop_map(f32::from_bits),
                        Just(0.0f32),
                    ],
                    prop_oneof![
                        -1e3f32..1e3f32,
                        -1e-6f32..1e-6f32,
                        -1e-20f32..1e-20f32,
                        Just(1.0f32),
                        Just(-1.0f32),
                    ],
                ),
                1..300,
            )
        ) {
            let a: Vec<f32> = pairs.iter().map(|p| p.0).collect();
            let b: Vec<f32> = pairs.iter().map(|p| p.1).collect();
            let EftDot { value, err } = eft_dot_f32_downgrade_only(&a, &b)
                .expect("the bounded finite strategy must publish a downgrade-only enclosure");
            // Value path untouched.
            let mut plain = 0.0f32;
            for (&x, &y) in a.iter().zip(b.iter()) {
                plain += x * y;
            }
            prop_assert_eq!(value.to_bits(), plain.to_bits());
            // Never worse than the incumbent.
            let higham = higham_dot_err_f32(&a, &b);
            prop_assert!(err <= higham, "err={err:e} > higham={higham:e}");
            // Still encloses the exact rational dot.
            let exact = exact_dot(&a, &b);
            prop_assert!(
                (exact - rat(value)).abs() <= rat(err),
                "downgraded radius stopped enclosing"
            );
        }
    }

    /// The gate is fail-closed: when the target's preconditions do not hold the
    /// channel must vanish, not degrade to a smaller radius.
    #[test]
    fn availability_gate_is_cached_and_agrees_with_the_self_check() {
        assert_eq!(eft_available(), eft_self_check().is_ok());
        assert_eq!(eft_available(), eft_available());
    }
}
