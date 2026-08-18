// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Floating-point precision tags for mixed-precision verification (P8).
//!
//! NY's bound propagation is an f32 *idealization*. Real models execute in
//! lower precisions on the GPU: f16 (Metal), bf16 (CUDA), or quantized GGUF
//! blocks. A bound proven in f32 is not automatically valid for the bits that
//! actually run, so verdicts must be widened to the deployed precision.
//!
//! [`FloatPrecision`] is the precision tag those widenings key off of. It is a
//! lightweight enum carrying only the IEEE-754-style format parameters needed
//! to *over-approximate* a deployed-precision value with an f32 interval. The
//! actual directed-rounding / round-to-nearest-representable widening primitives
//! live in the precision-prims component (see crate-level integration notes);
//! this enum is the shared vocabulary both components agree on.
//!
//! SOUNDNESS: any value produced at a given [`FloatPrecision`] is, after
//! conversion to f32, contained in an interval whose half-width is bounded by
//! the relative + absolute rounding error implied by that format. Helpers here
//! report those format parameters; they never narrow a bound.

use crate::Bound;
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

/// A floating-point compute/accumulate precision.
///
/// Today's default verification path is [`FloatPrecision::F32`], which matches
/// NY's historical idealization exactly (no widening). The non-F32 variants
/// describe lower-precision deployment targets that require SOUND widening
/// before an f32-proven bound may be claimed for them.
///
/// This enum is intentionally minimal: it is the shared tag referenced by both
/// the dtype-tagged-weights component (this crate's consumers in `ny-build`) and
/// the precision-prims component. If precision-prims lands a richer definition,
/// the two must be merged into this single location rather than duplicated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum FloatPrecision {
    /// IEEE-754 binary32. NY's idealized verification precision (the default).
    #[default]
    F32,
    /// IEEE-754 binary16 (half). 1 sign + 5 exponent + 10 stored mantissa bits.
    /// Used by Metal GPU inference.
    F16,
    /// bfloat16. 1 sign + 8 exponent + 7 stored mantissa bits (truncated f32).
    /// Used by CUDA GPU inference.
    Bf16,
}

impl FloatPrecision {
    /// Number of *stored* mantissa (fraction) bits for this format.
    ///
    /// The full significand precision is `mantissa_bits + 1` (the implicit
    /// leading bit). Returns the f32 value (23) for [`FloatPrecision::F32`].
    #[must_use]
    pub const fn mantissa_bits(self) -> u32 {
        match self {
            FloatPrecision::F32 => 23,
            FloatPrecision::F16 => 10,
            FloatPrecision::Bf16 => 7,
        }
    }

    /// Number of exponent bits for this format.
    #[must_use]
    pub const fn exponent_bits(self) -> u32 {
        match self {
            FloatPrecision::F32 => 8,
            FloatPrecision::F16 => 5,
            FloatPrecision::Bf16 => 8,
        }
    }

    /// The unit roundoff (machine epsilon / 2) for round-to-nearest in this
    /// format: `2^-(mantissa_bits + 1)`.
    ///
    /// A round-to-nearest conversion of a real value `x` into this format
    /// produces a result within a relative error of at most this much (away
    /// from overflow/subnormal regions). A SOUND widening of an f32 bound to
    /// account for deployed-precision rounding must inflate by at least this
    /// relative factor; this method only *reports* the figure — it does not
    /// itself widen anything.
    #[must_use]
    pub fn unit_roundoff(self) -> f32 {
        // 2^-(p) where p = mantissa_bits + 1 (implicit leading bit).
        // Exact in f32 for all formats here (small negative powers of two).
        let p = self.mantissa_bits() + 1;
        (2.0_f32).powi(-(p as i32))
    }

    /// Whether this precision is NY's idealized default (no widening required).
    #[must_use]
    pub const fn is_idealized_f32(self) -> bool {
        matches!(self, FloatPrecision::F32)
    }

    /// The largest finite (max-normal) magnitude representable in this format,
    /// expressed exactly in f32.
    ///
    /// SOUNDNESS (overflow test, F2): a deployed running sum whose magnitude can
    /// reach or exceed this value may overflow round-to-nearest to ±inf on the
    /// hardware. A SOUND deployed-precision bound for such a reduction must admit
    /// ±inf (a finite widened interval would EXCLUDE the deployed value). Callers
    /// use this threshold to decide when to widen an accumulating layer's output
    /// to `[-inf, +inf]`. Every finite 16-bit value is exactly representable in
    /// f32, so the returned f32 equals the format's max-normal exactly.
    #[must_use]
    pub fn max_normal(self) -> f32 {
        match self {
            FloatPrecision::F32 => f32::MAX,
            FloatPrecision::F16 => f16::MAX.to_f32(),
            FloatPrecision::Bf16 => bf16::MAX.to_f32(),
        }
    }

    /// The smallest positive subnormal magnitude representable in this format,
    /// expressed exactly in f32.
    ///
    /// Used to floor absolute-error bounds in the subnormal region (where the
    /// relative-error model breaks down). Returns the f32 smallest subnormal for
    /// [`FloatPrecision::F32`].
    #[must_use]
    pub fn smallest_subnormal(self) -> f32 {
        match self {
            FloatPrecision::F32 => f32::from_bits(1),
            FloatPrecision::F16 => f16::from_bits(1).to_f32(),
            FloatPrecision::Bf16 => bf16::from_bits(1).to_f32(),
        }
    }

    /// The *coarser* (more lossy / fewer-mantissa-bit) of two precisions.
    ///
    /// Used to combine the two precisions of a mixed-precision policy (compute and
    /// accumulate) into the single precision a SOUND output-rounding widening must
    /// key off of. SOUNDNESS: a deployed value that passes through *either* a
    /// `self`-precision or an `other`-precision step is rounded at least as coarsely
    /// as the coarser of the two; widening an f32 bound to the coarser grid is
    /// therefore an over-approximation of the rounding error of *both* steps (and a
    /// strict superset of widening to the finer grid). Ties (equal mantissa bits)
    /// return `self`, which is harmless since equal-coarseness formats induce the
    /// same widening.
    ///
    /// If either operand is non-F32, the result is non-F32 (the coarser one), so a
    /// policy that is non-idealized in *any* component still triggers widening.
    #[must_use]
    pub const fn coarser(self, other: FloatPrecision) -> FloatPrecision {
        // Fewer mantissa bits => coarser => larger rounding error => wider widening.
        if other.mantissa_bits() < self.mantissa_bits() {
            other
        } else {
            self
        }
    }

    /// SOUND interval bracketing the deployed-precision representation of `x`.
    ///
    /// Rounding an f32 weight/activation `x` into this precision (and reading it
    /// back as f32) is not exact for F16/Bf16: the hardware stores the nearest
    /// representable value, which may lie above or below `x`. To stay SOUND we
    /// return a `Bound` that contains *both* `x` and its rounded representation,
    /// so any value the deployed hardware could hold for this element is inside.
    ///
    /// For [`FloatPrecision::F32`] the representation is exact, so the returned
    /// bound is the degenerate `[x, x]` — preserving today's behavior exactly.
    ///
    /// SOUNDNESS: round-to-nearest gives a single value `r` (possibly ±inf on
    /// overflow); the returned interval is `[min(x, r), max(x, r)]`, which
    /// contains `x` (the idealized value) and `r` (the deployed value), including
    /// the case where `r` is ±inf. It is never narrower than a point and never
    /// excludes either endpoint.
    ///
    /// # Overflow
    /// If `|x|` exceeds the round-to-nearest overflow threshold of a 16-bit
    /// format, round-to-nearest overflows to ±inf on the deployed hardware. The
    /// deployed value the hardware holds is then *literally* ±inf, so a SOUND
    /// bracket of "every value the hardware could hold for this element" MUST
    /// contain ±inf. We therefore return a bound whose affected endpoint is the
    /// IEEE infinity itself, constructed via [`Bound::new_allow_infinite`].
    ///
    /// A prior version saturated the endpoint to `±f32::MAX` instead. That was
    /// UNSOUND: `+inf` is not `<= f32::MAX`, so the returned interval excluded
    /// the actual deployed value (`+inf`), violating containment. Returning
    /// `±inf` is the only sound choice once the deployed rounding overflows.
    ///
    /// # Panics
    /// Panics (via [`Bound::new_allow_infinite`]) only if `x` is NaN. Finite or
    /// infinite `x` is accepted; for non-finite `x` callers typically handle the
    /// element separately, but a finite `x` that overflows in `p` is the case
    /// this method specifically makes sound.
    #[must_use]
    pub fn representation_bound(self, x: f32) -> Bound {
        let r = match self {
            FloatPrecision::F32 => x,
            FloatPrecision::F16 => f16::from_f32(x).to_f32(),
            FloatPrecision::Bf16 => bf16::from_f32(x).to_f32(),
        };
        // `r` may be ±inf when the deployed-precision round-to-nearest overflows
        // the finite range. That is the *actual* value the hardware holds, so the
        // sound bracket must include it. `Bound::new_allow_infinite` admits the
        // infinite endpoint (it only rejects NaN / inverted intervals); the
        // min/max still order the endpoints correctly with an infinite `r`.
        Bound::new_allow_infinite(x.min(r), x.max(r))
    }
}

// ---------------------------------------------------------------------------
// 16-bit sign-magnitude bit stepping
//
// Both `half::f16` and `half::bf16` are 16-bit IEEE-754-style sign-magnitude
// formats: bit 15 is the sign, bits 0..=14 are the magnitude (exponent then
// mantissa). Consequently, treating the low 15 bits as an unsigned magnitude:
//   * incrementing the magnitude moves AWAY from zero (toward ±inf),
//   * decrementing the magnitude moves TOWARD zero,
//   * the smallest positive subnormal is magnitude `1`,
//   * the largest finite magnitude is `inf_magnitude - 1`,
//   * magnitude `inf_magnitude` is ±inf, and any larger magnitude is NaN.
// This is the same monotonic-bit-pattern trick `ny_tensor::rounding` uses for
// f32, specialized to 16 bits and parameterized over the inf magnitude so a
// single implementation serves both formats.
// ---------------------------------------------------------------------------

/// Magnitude (low 15 bits) at which an f16 becomes ±inf (`0x7C00`).
const F16_INF_MAGNITUDE: u16 = 0x7C00;
/// Magnitude (low 15 bits) at which a bf16 becomes ±inf (`0x7F80`).
const BF16_INF_MAGNITUDE: u16 = 0x7F80;
/// Sign bit for a 16-bit sign-magnitude float.
const SIGN16: u16 = 0x8000;
/// Magnitude mask for a 16-bit sign-magnitude float.
const MAG16: u16 = 0x7FFF;

/// The next representable value above a *finite* 16-bit sign-magnitude pattern
/// (toward +inf). Returns the resulting bit pattern.
///
/// `inf_mag` is the format's inf magnitude (`F16_INF_MAGNITUDE` / `BF16_INF_MAGNITUDE`).
/// When the step reaches the end of the finite range, the result is the format's
/// +inf pattern (`inf_mag`) — a SOUND over-approximation for an upper bound.
#[inline]
fn next_up_bits16(bits: u16, inf_mag: u16) -> u16 {
    let sign = bits & SIGN16;
    let mag = bits & MAG16;
    if sign == 0 {
        // Positive (or +0): move away from zero. Saturate at +inf.
        (mag + 1).min(inf_mag) // sign == 0
    } else {
        // Negative: move toward zero by decreasing magnitude.
        if mag == 0 {
            // -0 → smallest positive subnormal.
            1
        } else {
            // Still negative, smaller magnitude (closer to zero).
            SIGN16 | (mag - 1)
        }
    }
}

/// The next representable value below a *finite* 16-bit sign-magnitude pattern
/// (toward -inf). Returns the resulting bit pattern.
///
/// When the step reaches the end of the finite range, the result is the format's
/// -inf pattern (`SIGN16 | inf_mag`) — a SOUND over-approximation for a lower
/// bound.
#[inline]
fn next_down_bits16(bits: u16, inf_mag: u16) -> u16 {
    let sign = bits & SIGN16;
    let mag = bits & MAG16;
    if sign != 0 {
        // Negative (or -0): move away from zero (more negative). Saturate at -inf.
        let new_mag = (mag + 1).min(inf_mag);
        SIGN16 | new_mag
    } else {
        // Positive or +0: move toward zero by decreasing magnitude.
        if mag == 0 {
            // +0 → smallest negative subnormal.
            SIGN16 | 1
        } else {
            mag - 1 // sign == 0, smaller positive
        }
    }
}

/// Round a single finite f32 `x` outward to the bracketing values of a 16-bit
/// format, returning `(floor_p, ceil_p)` expressed back in f32.
///
/// `floor_p` is the largest p-representable value `<= x`; `ceil_p` is the
/// smallest p-representable value `>= x`. The conversion `to_bits`
/// (round-to-nearest into p) may move `x` in either direction, so each endpoint
/// is corrected by one ULP in p when it landed on the wrong side. Because every
/// finite 16-bit value is exactly representable in f32, the returned f32 values
/// equal their p-grid points exactly.
///
/// Pre: `x` is finite (callers handle NaN/±inf before dispatching here).
#[inline]
fn round_finite_outward_16bit(
    x: f32,
    inf_mag: u16,
    to_bits: impl Fn(f32) -> u16,
    from_bits: impl Fn(u16) -> f32,
) -> (f32, f32) {
    // Round-to-nearest into p, then read the exact f32 value of that p-point.
    let nearest_bits = to_bits(x);
    // `from_bits` of a finite p pattern is the exact f32 of that grid point. If
    // round-to-nearest overflowed to ±inf, `nearest_val` is ±inf, which the
    // comparisons below still handle correctly (it brackets `x` on that side).
    let nearest_val = from_bits(nearest_bits);

    // floor_p: largest p value <= x.
    let floor_p = if nearest_val <= x {
        nearest_val
    } else {
        // Rounded up past x: step down one ULP in p.
        from_bits(next_down_bits16(nearest_bits, inf_mag))
    };

    // ceil_p: smallest p value >= x.
    let ceil_p = if nearest_val >= x {
        nearest_val
    } else {
        // Rounded down below x: step up one ULP in p.
        from_bits(next_up_bits16(nearest_bits, inf_mag))
    };

    (floor_p, ceil_p)
}

/// Round an f32 interval outward to be representable-aware for precision `p`.
///
/// Returns the smallest interval, expressed in f32, that CONTAINS `[lower, upper]`
/// while having both endpoints exactly representable in `p`:
/// `lower'` is the largest p-representable value `<= lower` (round DOWN) and
/// `upper'` is the smallest p-representable value `>= upper` (round UP).
///
/// SOUNDNESS: every value a deployed-precision-`p` computation could produce in
/// `[lower, upper]`, once read back as f32, lies within `[lower', upper']`. The
/// result is never narrower than the input: `lower' <= lower <= upper <= upper'`.
///
/// # Behavior
/// - [`FloatPrecision::F32`]: identity — returns `(lower, upper)` unchanged
///   (preserves NY's exact f32 path; no widening, no regression).
/// - [`FloatPrecision::F16`] / [`FloatPrecision::Bf16`]: converts each endpoint
///   with `half`, correcting by one ULP in `p` if round-to-nearest moved the
///   endpoint the wrong way, so containment is guaranteed.
///
/// # Special cases (F16/Bf16)
/// - `NaN` endpoints are passed through unchanged (callers treat NaN as upstream
///   corruption; this never invents a finite bound from NaN).
/// - `lower = -inf` stays `-inf`; `upper = +inf` stays `+inf`.
/// - A finite endpoint whose magnitude exceeds the format's max-normal rounds
///   OUTWARD to `±inf` (the only SOUND choice — no finite p value contains it on
///   that side).
/// - Signed zero is normalized through the bit stepping (`-0.0 == 0.0`).
/// - Subnormals are handled by the bit stepping (magnitude `1` is the smallest
///   positive subnormal).
#[must_use]
pub fn round_to_precision_outward(lower: f32, upper: f32, p: FloatPrecision) -> (f32, f32) {
    match p {
        // Identity: today's exact f32 idealization. No widening, no regression.
        FloatPrecision::F32 => (lower, upper),
        FloatPrecision::F16 => (
            round_lower_16bit(
                lower,
                F16_INF_MAGNITUDE,
                |v| f16::from_f32(v).to_bits(),
                |b| f16::from_bits(b).to_f32(),
            ),
            round_upper_16bit(
                upper,
                F16_INF_MAGNITUDE,
                |v| f16::from_f32(v).to_bits(),
                |b| f16::from_bits(b).to_f32(),
            ),
        ),
        FloatPrecision::Bf16 => (
            round_lower_16bit(
                lower,
                BF16_INF_MAGNITUDE,
                |v| bf16::from_f32(v).to_bits(),
                |b| bf16::from_bits(b).to_f32(),
            ),
            round_upper_16bit(
                upper,
                BF16_INF_MAGNITUDE,
                |v| bf16::from_f32(v).to_bits(),
                |b| bf16::from_bits(b).to_f32(),
            ),
        ),
    }
}

/// Round a single endpoint DOWN to the largest p value `<= x` (the lower side).
#[inline]
fn round_lower_16bit(
    x: f32,
    inf_mag: u16,
    to_bits: impl Fn(f32) -> u16,
    from_bits: impl Fn(u16) -> f32,
) -> f32 {
    if x.is_nan() {
        // Never fabricate a finite bound from NaN — pass it through.
        return x;
    }
    if x == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if x == f32::INFINITY {
        // +inf as a lower endpoint: the largest p value <= +inf is the format's
        // own +inf, which is f32 +inf. (Degenerate; preserved for totality.)
        return f32::INFINITY;
    }
    let (floor_p, _ceil_p) = round_finite_outward_16bit(x, inf_mag, to_bits, from_bits);
    floor_p
}

/// Round a single endpoint UP to the smallest p value `>= x` (the upper side).
#[inline]
fn round_upper_16bit(
    x: f32,
    inf_mag: u16,
    to_bits: impl Fn(f32) -> u16,
    from_bits: impl Fn(u16) -> f32,
) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x == f32::INFINITY {
        return f32::INFINITY;
    }
    if x == f32::NEG_INFINITY {
        // -inf as an upper endpoint: smallest p value >= -inf is the format's
        // own -inf, which is f32 -inf. (Degenerate; preserved for totality.)
        return f32::NEG_INFINITY;
    }
    let (_floor_p, ceil_p) = round_finite_outward_16bit(x, inf_mag, to_bits, from_bits);
    ceil_p
}

/// A SOUND absolute error bound for representing a value of the given
/// `magnitude` in precision `p` (round-to-nearest into `p`).
///
/// Returns an upper bound on `|round_p(v) - v|` for any real `v` with
/// `|v| <= magnitude`. This is the quantity by which an accumulation result must
/// be widened to remain SOUND once it is realized in precision `p`.
///
/// The bound dominates the exact half-ULP-at-magnitude error:
/// - For a normal value, the round-to-nearest error is at most half a ULP,
///   i.e. `|v| * 2^-(mantissa+1) = |v| * unit_roundoff`. Using `nextUp(magnitude)`
///   and rounding the product up keeps the bound a strict over-approximation
///   despite f32 evaluation error.
/// - In the subnormal region the error is at most half the smallest subnormal;
///   we floor the bound at that value so it never underestimates near zero
///   (where the relative model breaks down).
///
/// # Behavior
/// - [`FloatPrecision::F32`]: returns `0.0` — the f32 idealization is exact by
///   definition, so no accumulation widening is applied (no regression).
/// - Non-finite or NaN `magnitude`: returns `+inf` (cannot bound the error of a
///   non-finite magnitude; widening by `+inf` is the only SOUND answer).
#[must_use]
pub fn precision_round_error_bound(magnitude: f32, p: FloatPrecision) -> f32 {
    if matches!(p, FloatPrecision::F32) {
        return 0.0;
    }
    if !magnitude.is_finite() {
        return f32::INFINITY;
    }
    let mag = magnitude.abs();

    // Relative half-ULP component: |v| * 2^-(mantissa+1). Push the magnitude up
    // one f32 ULP and round the product up so f32 evaluation error cannot make
    // the returned bound too small.
    let mag_hi = next_up_f32_local(mag);
    let rel = next_up_f32_local(mag_hi * p.unit_roundoff());

    // Subnormal floor: half the smallest positive subnormal of p (rounded up).
    // Guarantees a nonzero, never-underestimating bound near zero.
    let subnormal_half = half_smallest_subnormal(p);

    // The sound error bound is at least the larger of the two regimes.
    let bound = rel.max(subnormal_half);
    // Final guard: never return NaN.
    if bound.is_nan() {
        f32::INFINITY
    } else {
        bound
    }
}

/// A SOUND absolute error bound for the result of summing `n_terms` values in
/// precision `p`, given an upper bound `abs_term_sum` on the sum of the terms'
/// absolute values (i.e. `abs_term_sum >= sum_i |t_i|`).
///
/// This is the missing ACCUMULATION-error primitive. The representation helpers
/// above ([`FloatPrecision::representation_bound`], [`round_to_precision_outward`])
/// bracket the rounding of *storing one value* at precision `p`; they do NOT model
/// the rounding that happens *inside* a reduction as the running sum is formed.
/// A dot product / GEMM / reduction realized in f16 or bf16 accumulates rounding
/// error term by term, and that error can be far larger than a single ULP of the
/// output — large enough that an f16 sum of 5000 ones saturates near 2048 rather
/// than reaching 5000. This function bounds exactly that drift.
///
/// # Model (Higham, *Accuracy and Stability of Numerical Algorithms*, §4.2)
///
/// For recursive (sequential) summation of `N` floating-point numbers in a format
/// with unit roundoff `u`, the computed sum `ŝ` satisfies
/// `|ŝ - s| <= gamma_{N-1} * sum_i |t_i|`, where `s = sum_i t_i` is the exact sum
/// and `gamma_k = k*u / (1 - k*u)` for `k*u < 1`. The bound is independent of the
/// summation order in the sense that any order is dominated by `gamma_{N-1}`
/// (recursive order is the standard worst case used here, and it dominates
/// pairwise/blocked orders too). When `k*u >= 1` the classical bound breaks down
/// and the only SOUND answer is `+inf` (no finite bound is guaranteed); we
/// saturate to `+inf` in that regime.
///
/// # Product rounding
///
/// If the deployed compute precision also rounds each *term* before it is added
/// (the usual case for an f16-multiply / f16-accumulate dot product), each `t_i`
/// is first perturbed by a relative `u`. Folding that pre-rounding into the
/// backward-error chain replaces the `N-1` additions' worth of error with `N`
/// roundings' worth, i.e. the factor becomes `gamma_N` rather than `gamma_{N-1}`.
/// To stay SOUND for either deployment (terms exact vs. terms pre-rounded), we use
/// the larger factor `gamma_{n_terms}` here. Callers that *know* terms are exact
/// in `p` may pass a separately-rounded `abs_term_sum`; the `+1` in the factor is
/// a conservative, documented over-approximation that never underestimates.
///
/// # Behavior
/// - [`FloatPrecision::F32`]: returns `0.0`. Under NY's f32-idealization contract
///   the f32 path is exact by definition, so no accumulation widening applies
///   (strict no-op; no regression). Callers that genuinely want to model f32
///   accumulation must use a non-F32 precision.
/// - `n_terms == 0`: returns `0.0` (no terms, nothing is stored or accumulated).
/// - `n_terms == 1`: a single term is NOT pairwise-accumulated, but a left-to-right
///   reduction `acc = round_p(0 + t_0)` still performs ONE store rounding of the
///   running sum into the accumulate precision `p`. The classical `gamma_{N-1}`
///   term is zero for `N=1`, but that single store is a real rounding the relative
///   model would charge nothing for. We therefore return the sound single-store
///   charge `precision_round_error_bound(abs_term_sum, p)` (`abs_term_sum >= |t_0|`
///   and the per-store bound is monotone in magnitude), guaranteeing the primitive
///   itself never under-charges a one-term reduction. (This is the 3rd-audit
///   corner (a); the SOUND verify path no longer relies on this primitive for
///   Linear, but the primitive must be sound on its own.)
/// - Non-finite / NaN / negative `abs_term_sum`: returns `+inf` (cannot bound;
///   only sound answer).
/// - `n_terms * u >= 1` (factor denominator non-positive): returns `+inf`.
///
/// # Subnormal floor (F4)
/// The relative `gamma_N * sum|t_i|` model collapses toward zero as the term
/// magnitudes shrink: for tiny / subnormal terms it can return a bound far below
/// the TRUE deployed error. Concretely, if every term rounds to `0` in `p`
/// (because each `|t_i| <= s/2`, where `s` is the smallest positive subnormal of
/// `p`), the deployed sum is `0` while the idealized sum is up to `N * s/2`, so
/// the true error is up to `N * s/2` — which the relative model underestimates.
/// More generally each of the (up to `N`) roundings in the reduction contributes
/// an absolute error of at most half a ULP, and in the subnormal region a ULP is
/// `s`, so the total absolute error is at most `N * s/2`. We therefore FLOOR the
/// returned bound at `n_terms * half_smallest_subnormal(p)` (a sound `>= N*s/2`),
/// mirroring the subnormal floor already present in
/// [`precision_round_error_bound`]. This guarantees tiny / subnormal-magnitude
/// reductions cannot escape the widened interval.
///
/// # Soundness
/// Every arithmetic step is evaluated in f64 (so f32 rounding cannot shrink the
/// result mid-computation) and the final value is rounded UP to f32 via
/// [`next_up_f32_local`], guaranteeing the returned f32 is `>=` the true real
/// bound. The result is the larger of the relative `gamma_N` bound and the
/// subnormal floor, so it is a SOUND upper bound on the accumulation error in
/// BOTH the normal and the subnormal regime.
#[must_use]
pub fn summation_error_bound(abs_term_sum: f32, n_terms: usize, p: FloatPrecision) -> f32 {
    // f32-idealization: exact, no accumulation error modeled.
    if matches!(p, FloatPrecision::F32) {
        return 0.0;
    }
    // No terms: nothing is stored or accumulated.
    if n_terms == 0 {
        return 0.0;
    }
    // Cannot bound a non-finite or negative term-magnitude sum.
    if !abs_term_sum.is_finite() || abs_term_sum < 0.0 {
        return f32::INFINITY;
    }
    // A single term is not pairwise-accumulated, but a left-to-right reduction
    // still performs ONE store rounding `acc = round_p(0 + t_0)` of the running
    // sum into `p`. Charge that single store (sound corner-(a) fix). The relative
    // gamma model would charge 0 here, so this is strictly additive widening.
    if n_terms == 1 {
        return precision_round_error_bound(abs_term_sum, p);
    }

    // Subnormal floor: each of up to `n_terms` reduction roundings contributes at
    // most half a ULP, and a ULP in the subnormal region is the smallest positive
    // subnormal `s` of `p`, so the total absolute error is at most `n_terms*s/2`.
    // `half_smallest_subnormal(p)` is a sound (rounded-up) `>= s/2`. This floor is
    // independent of `abs_term_sum`, so it never underestimates tiny/subnormal
    // reductions even when the relative `gamma_N` term collapses toward zero.
    // Evaluated in f64 and rounded up so it strictly dominates the real value.
    let subnormal_floor = {
        let per_step = f64::from(half_smallest_subnormal(p));
        let floor_raw = (n_terms as f64) * per_step;
        if floor_raw.is_finite() {
            let f = floor_raw as f32;
            if f.is_finite() {
                next_up_f32_local(f)
            } else {
                f32::INFINITY
            }
        } else {
            f32::INFINITY
        }
    };

    // Use the larger factor gamma_{n_terms} to cover product (per-term) rounding
    // as well as the N-1 additions (see "Product rounding" above).
    let k = n_terms as f64;
    let u = f64::from(p.unit_roundoff());
    let ku = k * u;

    // Classical bound is only valid while k*u < 1; otherwise (including the
    // degenerate NaN case) saturate to +inf — the only sound answer.
    if ku >= 1.0 || ku.is_nan() {
        return f32::INFINITY;
    }

    // gamma_k = k*u / (1 - k*u), evaluated in f64.
    let gamma = ku / (1.0 - ku);
    let raw = gamma * f64::from(abs_term_sum);
    if !raw.is_finite() {
        return f32::INFINITY;
    }

    // Round UP to f32: take the f64 value, cast (round-to-nearest may go down),
    // then nudge up one f32 ULP so the f32 result strictly dominates `raw`.
    let as_f32 = raw as f32;
    if !as_f32.is_finite() {
        return f32::INFINITY;
    }
    let relative = next_up_f32_local(as_f32);

    // SOUND in both regimes: take the larger of the relative bound and the
    // subnormal floor (NaN-safe: a NaN compares false, so fall back to the floor).
    if relative.is_nan() || subnormal_floor > relative {
        subnormal_floor
    } else {
        relative
    }
}

/// Half the smallest positive subnormal of `p`, expressed exactly in f32, then
/// nudged up one f32 ULP so it strictly dominates the true half-subnormal error.
#[inline]
fn half_smallest_subnormal(p: FloatPrecision) -> f32 {
    // smallest * 0.5 is exact (power-of-two scaling); nudge up for safety.
    next_up_f32_local(p.smallest_subnormal() * 0.5)
}

/// Local copy of `next_up` for f32 (toward +inf), so this module needs no
/// dependency on `ny_tensor`. Matches `ny_tensor::rounding::next_up_f32`.
#[inline]
fn next_up_f32_local(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= f32::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return f32::from_bits(1);
    }
    if bits & 0x8000_0000 == 0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

/// Widen a validated [`Bound`] to remain SOUND for precision `p`.
///
/// Convenience wrapper over [`round_to_precision_outward`]: rounds the bound's
/// endpoints outward to the `p` grid and reconstructs a [`Bound`]. For
/// [`FloatPrecision::F32`] this is the identity (returns an equal bound).
///
/// SOUNDNESS: the returned bound contains the input bound
/// (`result.lower() <= b.lower()` and `result.upper() >= b.upper()`), so any
/// value the original bound admitted — and any deployed-precision-`p` realization
/// of it — is still admitted.
///
/// Endpoints that round outward to `±inf` (magnitude beyond the format's
/// max-normal) are preserved via the infinite-allowing constructor.
#[must_use]
pub fn widen_bound(b: &Bound, p: FloatPrecision) -> Bound {
    let (lower, upper) = round_to_precision_outward(b.lower(), b.upper(), p);
    // Outward rounding can only move endpoints apart, so `lower <= upper` holds
    // and neither is NaN (a `Bound`'s endpoints are never NaN). Use the
    // infinite-allowing constructor since outward rounding may saturate to ±inf.
    Bound::new_allow_infinite(lower, upper)
}

/// Widen every bound in `bounds` IN PLACE to remain SOUND for precision `p`.
///
/// Applies [`widen_bound`] to each element. This is the bulk post-processing
/// primitive used by policy-aware verification: after f32 propagation produces a
/// vector of (intermediate or output) bounds, calling this rounds each one
/// outward to the deployed-precision grid so the whole vector remains valid for
/// the bits that actually run.
///
/// # Behavior
/// - [`FloatPrecision::F32`]: a strict no-op — every bound is left byte-for-byte
///   unchanged (preserves NY's exact f32 path; no regression).
/// - [`FloatPrecision::F16`] / [`FloatPrecision::Bf16`]: each bound's lower
///   endpoint can only decrease-or-stay and its upper can only increase-or-stay.
///
/// SOUNDNESS: each output bound contains its input bound, so the widened vector is
/// an element-wise superset of the input vector. It is an OVER-approximation: it
/// models output/representation rounding into `p`, not exact per-op accumulation
/// order (a documented follow-on). It NEVER narrows a bound.
pub fn widen_bounds_for_precision(bounds: &mut [Bound], p: FloatPrecision) {
    // Fast path: the idealized f32 case must not touch the data at all.
    if p.is_idealized_f32() {
        return;
    }
    for b in bounds.iter_mut() {
        *b = widen_bound(b, p);
    }
}

/// Returning variant of [`widen_bounds_for_precision`].
///
/// Produces a fresh `Vec<Bound>` in which every bound has been widened outward to
/// the precision-`p` grid, leaving the input untouched. For
/// [`FloatPrecision::F32`] the result is an exact copy of the input.
///
/// SOUNDNESS: identical guarantee to [`widen_bounds_for_precision`] — every
/// returned bound contains the corresponding input bound.
#[must_use]
pub fn widen_bounds_for_precision_owned(bounds: &[Bound], p: FloatPrecision) -> Vec<Bound> {
    if p.is_idealized_f32() {
        return bounds.to_vec();
    }
    bounds.iter().map(|b| widen_bound(b, p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_f32_idealization() {
        assert_eq!(FloatPrecision::default(), FloatPrecision::F32);
        assert!(FloatPrecision::default().is_idealized_f32());
        assert!(!FloatPrecision::F16.is_idealized_f32());
        assert!(!FloatPrecision::Bf16.is_idealized_f32());
    }

    #[test]
    fn mantissa_and_exponent_bits_match_ieee_formats() {
        assert_eq!(FloatPrecision::F32.mantissa_bits(), 23);
        assert_eq!(FloatPrecision::F32.exponent_bits(), 8);
        assert_eq!(FloatPrecision::F16.mantissa_bits(), 10);
        assert_eq!(FloatPrecision::F16.exponent_bits(), 5);
        assert_eq!(FloatPrecision::Bf16.mantissa_bits(), 7);
        assert_eq!(FloatPrecision::Bf16.exponent_bits(), 8);
    }

    #[test]
    fn unit_roundoff_ordering_is_monotone_in_precision() {
        // Fewer mantissa bits => larger rounding error.
        let f32_u = FloatPrecision::F32.unit_roundoff();
        let f16_u = FloatPrecision::F16.unit_roundoff();
        let bf16_u = FloatPrecision::Bf16.unit_roundoff();
        assert!(f32_u < f16_u, "f32 must be the most precise");
        assert!(f16_u < bf16_u, "bf16 (7 mantissa bits) is coarser than f16");
        // Exact expected values.
        assert_eq!(f32_u, 2.0_f32.powi(-24));
        assert_eq!(f16_u, 2.0_f32.powi(-11));
        assert_eq!(bf16_u, 2.0_f32.powi(-8));
    }

    #[test]
    fn f32_representation_bound_is_exact_point() {
        // ADDITIVE: f32 must reproduce today's behavior — a degenerate [x, x].
        for &x in &[-3.5_f32, 0.0, 1.0, 12345.678, -0.001] {
            let b = FloatPrecision::F32.representation_bound(x);
            assert_eq!(b.lower(), x);
            assert_eq!(b.upper(), x);
        }
    }

    #[test]
    fn f16_representation_bound_contains_value_and_rounding() {
        // For values that are NOT exactly representable in f16, the bound must
        // straddle the true f32 value (contain it) with positive width.
        for &x in &[0.1_f32, 1.0 / 3.0, 2.7182817, 100.001, -0.123_456] {
            let b = FloatPrecision::F16.representation_bound(x);
            let r = f16::from_f32(x).to_f32();
            assert!(
                b.lower() <= x && x <= b.upper(),
                "f16 bound must contain x={x}"
            );
            assert!(
                b.lower() <= r && r <= b.upper(),
                "f16 bound must contain the deployed rounding r={r} of x={x}"
            );
        }
    }

    #[test]
    fn bf16_representation_bound_contains_value_and_rounding() {
        for &x in &[0.1_f32, 1.0 / 3.0, 2.7182817, 100.001, -0.123_456] {
            let b = FloatPrecision::Bf16.representation_bound(x);
            let r = bf16::from_f32(x).to_f32();
            assert!(
                b.lower() <= x && x <= b.upper(),
                "bf16 bound must contain x={x}"
            );
            assert!(
                b.lower() <= r && r <= b.upper(),
                "bf16 bound must contain the deployed rounding r={r} of x={x}"
            );
        }
    }

    #[test]
    fn f16_overflow_returns_infinite_bound_containing_value() {
        // SOUNDNESS (Part B regression): f16 max finite is ~65504; 70000 rounds
        // to f16 +inf on the deployed hardware. The deployed value IS +inf, so a
        // sound bracket MUST contain +inf. A finite f32::MAX upper would EXCLUDE
        // the actual deployed value (+inf is not <= f32::MAX) — that was the bug.
        let x = 70_000.0_f32;
        let b = FloatPrecision::F16.representation_bound(x);
        assert_eq!(
            b.upper(),
            f32::INFINITY,
            "overflow upper must be +inf, not f32::MAX"
        );
        assert!(b.upper().is_infinite() && b.upper() > 0.0);
        // Containment of the idealized value (finite lower <= x).
        assert!(b.lower() <= x, "overflow lower must stay below x");
        assert!(x <= b.upper(), "overflow bound must contain x");
        // The deployed value (+inf) is contained: not <= f32::MAX, but <= +inf.
        let deployed = f16::from_f32(x).to_f32();
        assert!(deployed.is_infinite());
        assert!(b.contains(deployed), "bound must contain the deployed +inf");

        let xn = -70_000.0_f32;
        let bn = FloatPrecision::F16.representation_bound(xn);
        assert_eq!(
            bn.lower(),
            f32::NEG_INFINITY,
            "negative overflow lower must be -inf"
        );
        assert!(bn.lower() <= xn && xn <= bn.upper());
        let deployed_n = f16::from_f32(xn).to_f32();
        assert!(deployed_n.is_infinite() && deployed_n < 0.0);
        assert!(
            bn.contains(deployed_n),
            "bound must contain the deployed -inf"
        );
    }

    #[test]
    fn float_precision_round_trips_through_serde() {
        for p in [
            FloatPrecision::F32,
            FloatPrecision::F16,
            FloatPrecision::Bf16,
        ] {
            let json = serde_json::to_string(&p).expect("serialize");
            let back: FloatPrecision = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(p, back);
        }
    }

    // ---------------------------------------------------------------------
    // round_to_precision_outward / widen_bound / precision_round_error_bound
    // SOUNDNESS + tightness tests.
    // ---------------------------------------------------------------------

    /// Deterministic xorshift64* PRNG so randomized soundness checks are
    /// reproducible without a `rand` dependency.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            // Avoid the all-zero fixed point of xorshift.
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// A pseudo-random *finite, non-NaN* f32 spanning a wide dynamic range,
        /// including subnormals, normals, and large magnitudes of both signs.
        fn next_finite_f32(&mut self) -> f32 {
            loop {
                let bits = (self.next_u64() & 0xFFFF_FFFF) as u32;
                let f = f32::from_bits(bits);
                if f.is_finite() {
                    return f;
                }
            }
        }
    }

    /// Round-trip an f32 through precision `p` (round-to-nearest).
    fn round_nearest(x: f32, p: FloatPrecision) -> f32 {
        match p {
            FloatPrecision::F32 => x,
            FloatPrecision::F16 => f16::from_f32(x).to_f32(),
            FloatPrecision::Bf16 => bf16::from_f32(x).to_f32(),
        }
    }

    /// True iff `v` is exactly representable in precision `p` (finite case):
    /// rounding it into `p` and back is the identity.
    fn is_representable(v: f32, p: FloatPrecision) -> bool {
        if !v.is_finite() {
            return false;
        }
        round_nearest(v, p) == v
    }

    /// One ULP step up in `p` from a finite, representable f32 value.
    fn p_next_up(v: f32, p: FloatPrecision) -> f32 {
        match p {
            FloatPrecision::F32 => next_up_f32_local(v),
            FloatPrecision::F16 => f16::from_bits(next_up_bits16(
                f16::from_f32(v).to_bits(),
                F16_INF_MAGNITUDE,
            ))
            .to_f32(),
            FloatPrecision::Bf16 => bf16::from_bits(next_up_bits16(
                bf16::from_f32(v).to_bits(),
                BF16_INF_MAGNITUDE,
            ))
            .to_f32(),
        }
    }

    /// One ULP step down in `p` from a finite, representable f32 value.
    fn p_next_down(v: f32, p: FloatPrecision) -> f32 {
        match p {
            FloatPrecision::F32 => next_down_f32_local(v),
            FloatPrecision::F16 => f16::from_bits(next_down_bits16(
                f16::from_f32(v).to_bits(),
                F16_INF_MAGNITUDE,
            ))
            .to_f32(),
            FloatPrecision::Bf16 => bf16::from_bits(next_down_bits16(
                bf16::from_f32(v).to_bits(),
                BF16_INF_MAGNITUDE,
            ))
            .to_f32(),
        }
    }

    fn next_down_f32_local(x: f32) -> f32 {
        let bits = x.to_bits();
        let magnitude = bits & 0x7fff_ffff;
        if magnitude >= f32::INFINITY.to_bits() {
            return x;
        }
        if magnitude == 0 {
            return f32::from_bits(0x8000_0001);
        }
        if bits & 0x8000_0000 == 0 {
            f32::from_bits(bits - 1)
        } else {
            f32::from_bits(bits + 1)
        }
    }

    #[test]
    fn f32_round_outward_is_identity() {
        // ADDITIVE: F32 must be a byte-for-byte identity (no widening).
        for &x in &[-12345.678_f32, 0.0, -0.0, 1.0, 0.1, f32::MAX, f32::MIN] {
            for &y in &[x, x + 1.0, 1e30] {
                if x <= y {
                    let (lo, hi) = round_to_precision_outward(x, y, FloatPrecision::F32);
                    assert_eq!(lo.to_bits(), x.to_bits(), "F32 lower must be identity");
                    assert_eq!(hi.to_bits(), y.to_bits(), "F32 upper must be identity");
                }
            }
        }
        // Infinities and NaN pass through unchanged on the F32 path.
        let (lo, hi) =
            round_to_precision_outward(f32::NEG_INFINITY, f32::INFINITY, FloatPrecision::F32);
        assert_eq!(lo, f32::NEG_INFINITY);
        assert_eq!(hi, f32::INFINITY);
    }

    #[test]
    fn point_round_outward_brackets_and_is_representable() {
        // Core soundness + tightness: for a point [x, x], the result (lo, hi)
        // must satisfy lo <= x <= hi, both lo/hi representable in p, AND be the
        // TIGHTEST such bracket (one ULP up from lo is > x unless lo == x, and
        // symmetrically for hi).
        for p in [FloatPrecision::F16, FloatPrecision::Bf16] {
            let mut rng = Rng::new(0xC0FF_EE12_3456_789A ^ p as u64);
            for _ in 0..200_000 {
                let x = rng.next_finite_f32();
                let (lo, hi) = round_to_precision_outward(x, x, p);

                // Containment (the soundness core).
                assert!(
                    lo <= x && x <= hi,
                    "p={p:?} x={x} ({:#010x}) not contained in [{lo}, {hi}]",
                    x.to_bits()
                );

                // Representability of finite endpoints.
                if lo.is_finite() {
                    assert!(
                        is_representable(lo, p),
                        "p={p:?} lo={lo} not representable (x={x})"
                    );
                }
                if hi.is_finite() {
                    assert!(
                        is_representable(hi, p),
                        "p={p:?} hi={hi} not representable (x={x})"
                    );
                }

                // Tightness of the lower endpoint: no p-value strictly between
                // lo and x. If lo < x, stepping lo up by one ULP must exceed x.
                if lo.is_finite() && lo < x {
                    let lo_up = p_next_up(lo, p);
                    assert!(
                        lo_up > x,
                        "p={p:?} lo={lo} not tight: next-up {lo_up} <= x={x}"
                    );
                }
                // Tightness of the upper endpoint: stepping hi down must drop below x.
                if hi.is_finite() && hi > x {
                    let hi_down = p_next_down(hi, p);
                    assert!(
                        hi_down < x,
                        "p={p:?} hi={hi} not tight: next-down {hi_down} >= x={x}"
                    );
                }
            }
        }
    }

    #[test]
    fn point_round_brackets_the_nearest_rounding() {
        // The deployed value is round_nearest(x). The bracket must contain it,
        // and the nearest rounding must equal lo or hi (it is one of the two
        // adjacent grid points). This pins down "no representable value strictly
        // between lo and f16/bf16(x) on the wrong side".
        for p in [FloatPrecision::F16, FloatPrecision::Bf16] {
            let mut rng = Rng::new(0x1234_5678_9ABC_DEF0 ^ p as u64);
            for _ in 0..200_000 {
                let x = rng.next_finite_f32();
                let r = round_nearest(x, p);
                let (lo, hi) = round_to_precision_outward(x, x, p);
                if r.is_finite() {
                    assert!(
                        lo <= r && r <= hi,
                        "p={p:?} nearest rounding r={r} of x={x} not in [{lo}, {hi}]"
                    );
                    // The nearest rounding is one of the two bracket endpoints.
                    assert!(
                        r == lo || r == hi,
                        "p={p:?} nearest r={r} is neither lo={lo} nor hi={hi} (x={x})"
                    );
                }
            }
        }
    }

    #[test]
    fn interval_round_outward_contains_input_interval() {
        // Property test: for random ordered [a, b], the widened interval contains
        // [a, b] and both endpoints are p-representable (when finite).
        for p in [FloatPrecision::F16, FloatPrecision::Bf16] {
            let mut rng = Rng::new(0xDEAD_BEEF_F00D_BAAD ^ p as u64);
            for _ in 0..200_000 {
                let mut a = rng.next_finite_f32();
                let mut b = rng.next_finite_f32();
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                let (lo, hi) = round_to_precision_outward(a, b, p);
                assert!(lo <= a, "p={p:?} lo={lo} > a={a}");
                assert!(hi >= b, "p={p:?} hi={hi} < b={b}");
                assert!(lo <= hi, "p={p:?} inverted result [{lo}, {hi}]");
                if lo.is_finite() {
                    assert!(is_representable(lo, p), "p={p:?} lo={lo} unrepresentable");
                }
                if hi.is_finite() {
                    assert!(is_representable(hi, p), "p={p:?} hi={hi} unrepresentable");
                }
            }
        }
    }

    #[test]
    fn exact_representable_endpoints_are_unchanged() {
        // If x is already representable in p, outward rounding must NOT move it
        // (tightest = exact). Enumerate every finite f16 value and check.
        for mag in 0u16..F16_INF_MAGNITUDE {
            for &sign in &[0u16, SIGN16] {
                let v = f16::from_bits(sign | mag).to_f32();
                let (lo, hi) = round_to_precision_outward(v, v, FloatPrecision::F16);
                assert_eq!(lo, v, "exact f16 {v} moved down to {lo}");
                assert_eq!(hi, v, "exact f16 {v} moved up to {hi}");
            }
        }
    }

    #[test]
    fn special_endpoints_f16_and_bf16() {
        for p in [FloatPrecision::F16, FloatPrecision::Bf16] {
            // Zero (both signs) → contains 0, both endpoints representable.
            for &z in &[0.0_f32, -0.0_f32] {
                let (lo, hi) = round_to_precision_outward(z, z, p);
                assert!(lo <= 0.0 && 0.0 <= hi, "p={p:?} zero not contained");
                assert_eq!(lo, 0.0, "p={p:?} zero lo should be 0");
                assert_eq!(hi, 0.0, "p={p:?} zero hi should be 0");
            }

            // Smallest positive subnormal of p is exactly representable.
            let tiny = match p {
                FloatPrecision::F16 => f16::from_bits(1).to_f32(),
                FloatPrecision::Bf16 => bf16::from_bits(1).to_f32(),
                FloatPrecision::F32 => unreachable!(),
            };
            let (lo, hi) = round_to_precision_outward(tiny, tiny, p);
            assert_eq!(lo, tiny);
            assert_eq!(hi, tiny);

            // Max finite normal of p is exactly representable.
            let max_normal = match p {
                FloatPrecision::F16 => f16::MAX.to_f32(),
                FloatPrecision::Bf16 => bf16::MAX.to_f32(),
                FloatPrecision::F32 => unreachable!(),
            };
            let (lo, hi) = round_to_precision_outward(max_normal, max_normal, p);
            assert_eq!(lo, max_normal);
            assert_eq!(hi, max_normal);

            // A FINITE f32 magnitude beyond p's finite range must round OUTWARD
            // to ±inf on the overflowing side, never collapse to a finite value
            // that excludes the true point. `f32::MAX` (3.40e38) exceeds the
            // round-to-nearest overflow threshold of BOTH f16 (~65520) and bf16
            // (~3.396e38) while being a finite f32, so it is a valid "huge"
            // beyond-range probe for either format.
            let huge = f32::MAX;
            assert!(
                huge > max_normal,
                "test setup: huge must exceed p max-normal"
            );
            let (lo, hi) = round_to_precision_outward(huge, huge, p);
            assert_eq!(hi, f32::INFINITY, "p={p:?} upper of {huge} must be +inf");
            assert!(lo <= huge, "p={p:?} lower {lo} must stay <= {huge}");
            assert!(
                lo.is_finite(),
                "p={p:?} lower of finite huge should be finite"
            );
            assert_eq!(
                lo, max_normal,
                "p={p:?} lower of overflow should be max-normal"
            );

            let neg_huge = -huge;
            let (lo, hi) = round_to_precision_outward(neg_huge, neg_huge, p);
            assert_eq!(
                lo,
                f32::NEG_INFINITY,
                "p={p:?} lower of {neg_huge} must be -inf"
            );
            assert!(hi >= neg_huge && hi.is_finite());
            assert_eq!(hi, -max_normal, "p={p:?} upper of negative overflow");

            // Infinities pass through on the correct side.
            let (lo, hi) = round_to_precision_outward(f32::NEG_INFINITY, f32::INFINITY, p);
            assert_eq!(lo, f32::NEG_INFINITY);
            assert_eq!(hi, f32::INFINITY);

            // NaN endpoints pass through (never fabricate a finite bound).
            let (lo, _) = round_to_precision_outward(f32::NAN, 1.0, p);
            assert!(lo.is_nan(), "p={p:?} NaN lower must stay NaN");
            let (_, hi) = round_to_precision_outward(-1.0, f32::NAN, p);
            assert!(hi.is_nan(), "p={p:?} NaN upper must stay NaN");
        }
    }

    #[test]
    fn widen_bound_contains_input_bound() {
        for p in [
            FloatPrecision::F32,
            FloatPrecision::F16,
            FloatPrecision::Bf16,
        ] {
            let mut rng = Rng::new(0x0BAD_C0DE_1234_5678 ^ p as u64);
            for _ in 0..100_000 {
                let mut a = rng.next_finite_f32();
                let mut b = rng.next_finite_f32();
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                let input = Bound::new_allow_infinite(a, b);
                let widened = widen_bound(&input, p);
                assert!(
                    widened.lower() <= input.lower(),
                    "p={p:?} widened lower {} > input lower {}",
                    widened.lower(),
                    input.lower()
                );
                assert!(
                    widened.upper() >= input.upper(),
                    "p={p:?} widened upper {} < input upper {}",
                    widened.upper(),
                    input.upper()
                );
                if p.is_idealized_f32() {
                    // F32 is the identity.
                    assert_eq!(widened.lower(), input.lower());
                    assert_eq!(widened.upper(), input.upper());
                }
            }
        }
    }

    #[test]
    fn round_error_bound_dominates_actual_rounding_error() {
        // The reported error bound must be >= the true |round_p(x) - x| for any
        // x with |x| <= magnitude. We sample x and use magnitude = |x|.
        for p in [FloatPrecision::F16, FloatPrecision::Bf16] {
            let mut rng = Rng::new(0xFACE_FEED_CAFE_B0BA ^ p as u64);
            for _ in 0..200_000 {
                let x = rng.next_finite_f32();
                let r = round_nearest(x, p);
                if !r.is_finite() {
                    // Overflow: error bound is allowed to be anything finite/inf;
                    // skip (handled by the round_to_precision_outward inf tests).
                    continue;
                }
                let actual_err = (r as f64 - x as f64).abs();
                let bound = precision_round_error_bound(x.abs(), p) as f64;
                assert!(
                    bound >= actual_err,
                    "p={p:?} x={x}: bound {bound} < actual error {actual_err}"
                );
            }
        }
    }

    #[test]
    fn round_error_bound_special_cases() {
        // F32 is exact: zero error.
        assert_eq!(precision_round_error_bound(1.0, FloatPrecision::F32), 0.0);
        assert_eq!(precision_round_error_bound(1e30, FloatPrecision::F32), 0.0);
        // Non-finite magnitude → +inf (cannot bound; only sound answer).
        assert_eq!(
            precision_round_error_bound(f32::INFINITY, FloatPrecision::F16),
            f32::INFINITY
        );
        assert_eq!(
            precision_round_error_bound(f32::NAN, FloatPrecision::Bf16),
            f32::INFINITY
        );
        // Always strictly positive and finite for finite non-F32 magnitudes,
        // and monotone non-decreasing in magnitude.
        for p in [FloatPrecision::F16, FloatPrecision::Bf16] {
            let e_small = precision_round_error_bound(0.0, p);
            let e_mid = precision_round_error_bound(1.0, p);
            let e_big = precision_round_error_bound(1000.0, p);
            assert!(e_small > 0.0 && e_small.is_finite());
            assert!(e_mid >= e_small);
            assert!(e_big >= e_mid);
        }
    }

    #[test]
    fn coarser_picks_fewer_mantissa_bits() {
        use FloatPrecision::{Bf16, F16, F32};
        // bf16 (7) is coarser than f16 (10) is coarser than f32 (23).
        assert_eq!(F32.coarser(F16), F16);
        assert_eq!(F16.coarser(F32), F16);
        assert_eq!(F16.coarser(Bf16), Bf16);
        assert_eq!(Bf16.coarser(F16), Bf16);
        assert_eq!(F32.coarser(Bf16), Bf16);
        // Idempotent on equal inputs.
        assert_eq!(F16.coarser(F16), F16);
        assert_eq!(F32.coarser(F32), F32);
        // Any non-F32 component makes the combination non-idealized.
        assert!(!F32.coarser(F16).is_idealized_f32());
        assert!(!F16.coarser(F32).is_idealized_f32());
        assert!(F32.coarser(F32).is_idealized_f32());
    }

    #[test]
    fn widen_bounds_for_precision_f32_is_strict_noop() {
        // ADDITIVE: F32 must leave every bound byte-for-byte unchanged.
        let original = vec![
            Bound::new(-3.5, 2.25),
            Bound::new(0.1, 0.3000001),
            Bound::new(-1.0 / 3.0, 1234.5678),
        ];
        let mut bounds = original.clone();
        widen_bounds_for_precision(&mut bounds, FloatPrecision::F32);
        for (got, exp) in bounds.iter().zip(original.iter()) {
            assert_eq!(got.lower().to_bits(), exp.lower().to_bits());
            assert_eq!(got.upper().to_bits(), exp.upper().to_bits());
        }
        // Returning variant is an exact copy on F32.
        let owned = widen_bounds_for_precision_owned(&original, FloatPrecision::F32);
        for (got, exp) in owned.iter().zip(original.iter()) {
            assert_eq!(got.lower().to_bits(), exp.lower().to_bits());
            assert_eq!(got.upper().to_bits(), exp.upper().to_bits());
        }
    }

    #[test]
    fn widen_bounds_for_precision_only_widens_outward() {
        for p in [FloatPrecision::F16, FloatPrecision::Bf16] {
            let original = vec![
                Bound::new(0.1, 0.2),
                Bound::new(-2.7182817, 3.1400003),
                Bound::new(-0.123_456, 0.123_456),
                Bound::new(100.001, 100.002),
            ];
            let mut bounds = original.clone();
            widen_bounds_for_precision(&mut bounds, p);
            for (got, exp) in bounds.iter().zip(original.iter()) {
                // Lower never increases; upper never decreases (never narrows).
                assert!(
                    got.lower() <= exp.lower(),
                    "p={p:?} lower widened the wrong way: {} > {}",
                    got.lower(),
                    exp.lower()
                );
                assert!(
                    got.upper() >= exp.upper(),
                    "p={p:?} upper widened the wrong way: {} < {}",
                    got.upper(),
                    exp.upper()
                );
                // Containment of the original interval.
                assert!(got.lower() <= exp.lower() && got.upper() >= exp.upper());
            }
            // In-place and returning variants agree.
            let owned = widen_bounds_for_precision_owned(&original, p);
            for (a, b) in bounds.iter().zip(owned.iter()) {
                assert_eq!(a.lower(), b.lower());
                assert_eq!(a.upper(), b.upper());
            }
        }
    }

    #[test]
    fn widen_bounds_for_precision_empty_slice_is_fine() {
        let mut empty: Vec<Bound> = vec![];
        widen_bounds_for_precision(&mut empty, FloatPrecision::F16);
        assert!(empty.is_empty());
        let owned = widen_bounds_for_precision_owned(&[], FloatPrecision::Bf16);
        assert!(owned.is_empty());
    }

    #[test]
    fn next_up_down_bits16_round_trip_inverse_on_finite() {
        // next_down(next_up(v)) == v for every finite f16 not at the +inf edge,
        // confirming the bit-stepping helpers are exact ULP neighbors.
        for mag in 0u16..F16_INF_MAGNITUDE {
            for &sign in &[0u16, SIGN16] {
                let bits = sign | mag;
                let up = next_up_bits16(bits, F16_INF_MAGNITUDE);
                // Skip the case where we saturated to +inf (no finite inverse).
                if (up & MAG16) == F16_INF_MAGNITUDE {
                    continue;
                }
                let back = next_down_bits16(up, F16_INF_MAGNITUDE);
                let orig = f16::from_bits(bits).to_f32();
                let round = f16::from_bits(back).to_f32();
                assert_eq!(
                    orig, round,
                    "f16 down(up(bits={bits:#06x})) changed value {orig} -> {round}"
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // summation_error_bound (Part C): SOUND accumulation-error primitive.
    // ---------------------------------------------------------------------

    /// A REAL f16 recursive (left-to-right) sum of `n` copies of `term`,
    /// rounding to f16 after every addition — the deployed accumulation.
    fn f16_recursive_sum_of(term: f32, n: usize) -> f32 {
        let mut acc = f16::from_f32(0.0);
        let t = f16::from_f32(term);
        for _ in 0..n {
            acc = f16::from_f32(acc.to_f32() + t.to_f32());
        }
        acc.to_f32()
    }

    /// A REAL bf16 recursive sum (deployed accumulation in bf16).
    fn bf16_recursive_sum_of(term: f32, n: usize) -> f32 {
        let mut acc = bf16::from_f32(0.0);
        let t = bf16::from_f32(term);
        for _ in 0..n {
            acc = bf16::from_f32(acc.to_f32() + t.to_f32());
        }
        acc.to_f32()
    }

    #[test]
    fn summation_error_bound_f32_is_zero_and_trivial_cases() {
        // f32-idealization: exact, no accumulation widening.
        assert_eq!(
            summation_error_bound(5000.0, 5000, FloatPrecision::F32),
            0.0
        );
        // 0 terms: nothing stored or accumulated.
        assert_eq!(summation_error_bound(123.0, 0, FloatPrecision::F16), 0.0);
        // F32 is exact even for the single-store case.
        assert_eq!(summation_error_bound(123.0, 1, FloatPrecision::F32), 0.0);
        // n=1 (corner (a)): a single left-to-right store still rounds once into p,
        // so the primitive now charges exactly that single-store error (not 0).
        let single = summation_error_bound(123.0, 1, FloatPrecision::F16);
        assert_eq!(
            single,
            precision_round_error_bound(123.0, FloatPrecision::F16)
        );
        assert!(
            single > 0.0,
            "n=1 single-store charge must be strictly positive"
        );
        // And it must dominate the real f16 single store of any |t| <= 123.
        for &t in &[123.0_f32, 100.0, -77.5, 0.3, -0.001] {
            let r = f16::from_f32(t).to_f32();
            let actual = (f64::from(r) - f64::from(t)).abs();
            assert!(
                f64::from(single) >= actual,
                "n=1 charge {single} < real single-store error {actual} (t={t})"
            );
        }
        // Non-finite / negative term-sum cannot be bounded.
        assert_eq!(
            summation_error_bound(f32::INFINITY, 10, FloatPrecision::F16),
            f32::INFINITY
        );
        assert_eq!(
            summation_error_bound(f32::NAN, 10, FloatPrecision::Bf16),
            f32::INFINITY
        );
        assert_eq!(
            summation_error_bound(-1.0, 10, FloatPrecision::F16),
            f32::INFINITY
        );
    }

    #[test]
    fn summation_error_bound_saturates_to_inf_when_ku_ge_1() {
        // For f16 (u = 2^-11), n*u >= 1 once n >= 2048. The classical bound
        // breaks down; the only sound answer is +inf.
        let e = summation_error_bound(5000.0, 5000, FloatPrecision::F16);
        assert!(e.is_infinite(), "f16 n=5000 must saturate to +inf, got {e}");
        // bf16 (u = 2^-8): n*u >= 1 once n >= 256.
        let e2 = summation_error_bound(281.6, 512, FloatPrecision::Bf16);
        assert!(
            e2.is_infinite(),
            "bf16 n=512 must saturate to +inf, got {e2}"
        );
    }

    #[test]
    fn summation_error_bound_contains_f16_5000_ones_drift() {
        // ACCEPTANCE (counterexample 1): summing 5000 ones in f16 saturates near
        // 2048. The idealized f32 sum is 5000 (point bound [5000, 5000]). The
        // sound widened interval [5000 - E, 5000 + E] MUST contain the deployed
        // value (~2048).
        let n = 5000usize;
        let abs_term_sum = 5000.0_f32; // sum |t_i| = 5000 * 1.0
        let e = summation_error_bound(abs_term_sum, n, FloatPrecision::F16);
        let deployed = f16_recursive_sum_of(1.0, n);
        // The deployed value is well below 5000 (saturation), demonstrating the
        // bug the representation-only widening missed.
        assert!(
            deployed < 2100.0,
            "f16 5000-ones should saturate, got {deployed}"
        );
        let lo = 5000.0_f32 - e;
        let hi = 5000.0_f32 + e;
        assert!(
            lo <= deployed && deployed <= hi,
            "widened [{lo}, {hi}] must contain deployed {deployed} (E={e})"
        );
        // Specifically must contain 2048 per the acceptance criterion.
        assert!(
            lo <= 2048.0 && 2048.0 <= hi,
            "widened bound must contain 2048"
        );
    }

    #[test]
    fn summation_error_bound_dominates_real_f16_sum_for_moderate_n() {
        // Numerically verify the bound DOMINATES the actual f16 recursive-sum
        // error for several N in the regime where the bound is finite (n*u < 1,
        // i.e. n < 2048 for f16). Use terms that are exactly f16-representable so
        // the only error is accumulation (isolating what this primitive models).
        for &term in &[1.0_f32, 0.5, 0.25, 0.125] {
            for &n in &[2usize, 4, 8, 16, 64, 256, 1024, 2000] {
                let abs_term_sum = (n as f32) * term.abs();
                let e = summation_error_bound(abs_term_sum, n, FloatPrecision::F16);
                if !e.is_finite() {
                    continue; // saturated regime; trivially dominates.
                }
                let ideal = (n as f64) * f64::from(term);
                let deployed = f16_recursive_sum_of(term, n);
                let actual_err = (f64::from(deployed) - ideal).abs();
                assert!(
                    f64::from(e) >= actual_err,
                    "f16 term={term} n={n}: bound {e} < actual error {actual_err} \
                     (deployed={deployed}, ideal={ideal})"
                );
            }
        }
    }

    #[test]
    fn summation_error_bound_dominates_real_bf16_sum_for_moderate_n() {
        // Same domination check for bf16 (finite regime n*u < 1, i.e. n < 256).
        for &term in &[1.0_f32, 0.5, 0.25] {
            for &n in &[2usize, 4, 8, 16, 64, 128, 200] {
                let abs_term_sum = (n as f32) * term.abs();
                let e = summation_error_bound(abs_term_sum, n, FloatPrecision::Bf16);
                if !e.is_finite() {
                    continue;
                }
                let ideal = (n as f64) * f64::from(term);
                let deployed = bf16_recursive_sum_of(term, n);
                let actual_err = (f64::from(deployed) - ideal).abs();
                assert!(
                    f64::from(e) >= actual_err,
                    "bf16 term={term} n={n}: bound {e} < actual error {actual_err}"
                );
            }
        }
    }

    #[test]
    fn summation_error_bound_has_subnormal_floor_f4() {
        // F4: the relative gamma_N * sum|t| model collapses toward zero for tiny
        // term magnitudes. The subnormal floor guarantees the bound is at least
        // n_terms * (half the smallest subnormal of p), so tiny/subnormal-magnitude
        // reductions cannot escape.
        for p in [FloatPrecision::F16, FloatPrecision::Bf16] {
            let s = p.smallest_subnormal();
            for &n in &[2usize, 4, 8, 16, 100, 1000] {
                // abs_term_sum near zero (where the relative model underestimates).
                let e = summation_error_bound(s * 0.0, n, p); // abs_term_sum == 0
                let expected_floor = (n as f32) * (s * 0.5);
                assert!(
                    e >= expected_floor,
                    "p={p:?} n={n}: bound {e} below subnormal floor {expected_floor}"
                );
                assert!(e > 0.0, "p={p:?} n={n}: floor must be strictly positive");
            }
        }
    }

    /// REAL deployed recursive sum at precision `p`, terms pre-rounded to `p`.
    fn p_recursive_sum_of(term: f32, n: usize, p: FloatPrecision) -> f32 {
        match p {
            FloatPrecision::F16 => f16_recursive_sum_of(term, n),
            FloatPrecision::Bf16 => bf16_recursive_sum_of(term, n),
            FloatPrecision::F32 => {
                let mut acc = 0.0_f32;
                for _ in 0..n {
                    acc += term;
                }
                acc
            }
        }
    }

    #[test]
    fn summation_error_bound_dominates_real_subnormal_sums_f4() {
        // F4 numeric verification: a REAL half-crate recursive sum across tiny /
        // subnormal-magnitude terms must always lie inside [ideal - E, ideal + E].
        // We use abs_term_sum = n * |term| (terms pre-rounded into p) so the bound
        // is the one the layer-aware path actually feeds it.
        for p in [FloatPrecision::F16, FloatPrecision::Bf16] {
            let s = p.smallest_subnormal();
            // A spread of tiny magnitudes: smallest subnormal, a few subnormals,
            // and just above the subnormal/normal boundary. Includes magnitudes
            // that round to ZERO when pre-rounded into p (term magnitude < s/2).
            let terms = [
                s * 0.25, // rounds to 0 in p (idealized nonzero, deployed 0)
                s,        // smallest subnormal
                s * 3.0,
                s * 17.0,
                s * 1000.0,
            ];
            for &raw_term in &terms {
                // Pre-round the term into p (what the deployed hardware stores).
                let term = round_nearest(raw_term, p);
                for &n in &[2usize, 4, 8, 16, 64, 200, 1000] {
                    // abs_term_sum bounds sum_i |t_i| using the rounded term.
                    let abs_term_sum = next_up_f32_local((n as f32) * term.abs());
                    let e = summation_error_bound(abs_term_sum, n, p);
                    if !e.is_finite() {
                        continue; // saturated regime trivially dominates.
                    }
                    // Idealized sum uses the (rounded) term value, in f64.
                    let ideal = (n as f64) * f64::from(term);
                    let deployed = p_recursive_sum_of(term, n, p);
                    let actual_err = (f64::from(deployed) - ideal).abs();
                    assert!(
                        f64::from(e) >= actual_err,
                        "p={p:?} raw_term={raw_term} term={term} n={n}: bound {e} < actual \
                         error {actual_err} (deployed={deployed}, ideal={ideal})"
                    );
                    // Also stress the case where the IDEALIZED (un-rounded) tiny
                    // term is summed and the deployed rounds each to 0: the bound
                    // must still contain the deployed 0 relative to the un-rounded
                    // idealized sum. Here ideal_unrounded can be up to n*s/2.
                    let ideal_unrounded = (n as f64) * f64::from(raw_term);
                    let deployed_unrounded = p_recursive_sum_of(raw_term, n, p);
                    let err_unrounded = (f64::from(deployed_unrounded) - ideal_unrounded).abs();
                    let e_unrounded =
                        summation_error_bound(next_up_f32_local((n as f32) * raw_term.abs()), n, p);
                    if e_unrounded.is_finite() {
                        assert!(
                            f64::from(e_unrounded) >= err_unrounded,
                            "p={p:?} raw_term={raw_term} n={n}: unrounded bound \
                             {e_unrounded} < err {err_unrounded} \
                             (deployed={deployed_unrounded}, ideal={ideal_unrounded})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn max_normal_and_smallest_subnormal_match_formats() {
        assert_eq!(FloatPrecision::F16.max_normal(), f16::MAX.to_f32());
        assert_eq!(FloatPrecision::Bf16.max_normal(), bf16::MAX.to_f32());
        assert_eq!(FloatPrecision::F32.max_normal(), f32::MAX);
        assert_eq!(
            FloatPrecision::F16.smallest_subnormal(),
            f16::from_bits(1).to_f32()
        );
        assert_eq!(
            FloatPrecision::Bf16.smallest_subnormal(),
            bf16::from_bits(1).to_f32()
        );
        // Sanity: f16 max-normal is ~65504; bf16 max-normal ~3.39e38.
        assert!((FloatPrecision::F16.max_normal() - 65504.0).abs() < 1.0);
        assert!(FloatPrecision::Bf16.max_normal() > 3.0e38);
    }

    #[test]
    fn summation_error_bound_is_monotone_and_rounded_up() {
        // Monotone non-decreasing in both abs_term_sum and n_terms (within the
        // finite regime), and strictly positive once there is real accumulation.
        let p = FloatPrecision::F16;
        let a = summation_error_bound(100.0, 10, p);
        let b = summation_error_bound(200.0, 10, p);
        let c = summation_error_bound(100.0, 20, p);
        assert!(a > 0.0 && a.is_finite());
        assert!(b >= a, "larger term-sum must not decrease the bound");
        assert!(c >= a, "more terms must not decrease the bound");
    }
}
