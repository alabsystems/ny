// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Directed-rounding helpers for the margin-row lane (#twinwall).
//!
//! Two arithmetic modes drive every bound in this module tree:
//!
//! * [`RoundMode::Parity`] — plain f64 round-to-nearest, algebra identical to
//!   the verified Python reference engine (part of the development-time
//!   falsifier harness, not shipped here). MEASUREMENT
//!   grade: used only by differential tests, never for verdicts.
//! * [`RoundMode::Outward`] — certified-outward: every quantity that feeds a
//!   verdict carries a rigorous error term (Higham `gamma_n` dot-product
//!   envelopes + per-step elementwise widening + a folded-weight relative
//!   error), and final bounds are rounded TOWARD -inf (lower) / +inf (upper).
//!   A domain may only be closed on a bound computed in this mode.
//!
//! The discipline mirrors `network/graph_ibp_f64_cell.rs` (naive f64
//! accumulation + `gamma_n * sum(|terms|)` widening, Higham sec. 3.5) and the
//! `LinearBounds64` certified coefficient-error carry (#vnncomp-aw-soundness).

/// f64 unit roundoff (2^-53).
pub const UNIT: f64 = 1.110_223_024_625_156_5e-16;

/// f32 unit roundoff (2^-24). Relative error bound of a single f32
/// round-to-nearest op (used only by the optional `NY_MARGIN_ROW_ROOT_F32`
/// fast-path conv lanes; every f32 rounding is re-absorbed into a certified
/// additive concretize slack, never into a verdict coefficient directly).
pub const UNIT_F32: f64 = 5.960_464_477_539_063e-8;

/// Smallest positive f32 subnormal (2^-149). The absolute rounding error of any
/// single f32 op is `<= UNIT_F32 * |x|` for normals and `<= SUBNORMAL_F32` in
/// the gradual-underflow range, so charging `k * SUBNORMAL_F32` per accumulated
/// conv output dominates any FTZ / subnormal effect (Rust f32 does gradual
/// underflow, so this floor is a rigor belt, not a load-bearing term).
pub const SUBNORMAL_F32: f64 = 1.401_298_464_324_817e-45;

/// Relative error bound on BN-folded conv weights and biases produced by the
/// f64 fold `W' = W * w_bn / sqrt(var + eps)` (4 nearest-rounded ops:
/// add, sqrt, div, mul => relative error <= (1+u)^4 - 1 ~= 4.44e-16).
/// 1e-15 covers it with >2x headroom. Biases carry their own absolute error
/// vectors computed by the spec builder (cancellation-safe), so this constant
/// only needs to cover the multiplicative kernel chain.
pub const RHO_FOLD: f64 = 1e-15;

/// Rounding mode for a full pass (see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundMode {
    /// Round-to-nearest measurement semantics (Python parity; tests only).
    Parity,
    /// Certified-outward semantics (production; the only verdict-grade mode).
    Outward,
}

impl RoundMode {
    /// True in the certified-outward mode.
    #[inline]
    pub fn outward(self) -> bool {
        matches!(self, Self::Outward)
    }
}

/// Next representable f64 toward +inf. Non-finite inputs pass through.
#[inline]
pub fn next_up(x: f64) -> f64 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude >= f64::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Next representable f64 toward -inf. Non-finite inputs pass through.
#[inline]
pub fn next_down(x: f64) -> f64 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude >= f64::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return -f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

/// Higham `gamma_n = n*u / (1 - n*u)` rounded UP one ulp, so the returned
/// value is >= the real gamma. Saturates (returns 1.0) for absurd n; callers
/// treat gamma >= 1 as "degrade to unbounded" via the NaN/Inf firewall.
#[inline]
pub fn gamma_n(n: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let nu = (n as f64) * UNIT;
    if nu >= 0.5 {
        return 1.0;
    }
    next_up(nu / (1.0 - nu))
}

/// f32 analogue of [`gamma_n`]: `n*u32 / (1 - n*u32)` rounded UP one ulp, with
/// `u32 = 2^-24`. Bounds the relative error of an f32 dot product / conv row of
/// length `n` (Higham sec. 3.5, valid for ANY summation order — so the f32 conv
/// grains need not be bit-identical). Saturates to 1.0 for absurd `n` (the
/// caller's NaN/Inf firewall then degrades the domain to Unknown).
#[inline]
pub fn gamma_n_f32(n: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let nu = (n as f64) * UNIT_F32;
    if nu >= 0.5 {
        return 1.0;
    }
    next_up(nu / (1.0 - nu))
}

/// Certified upper bound on the REAL value of a nonneg-magnitude quantity
/// whose f64 nearest-rounded computation `x` accumulated relative error
/// <= `rel` (i.e. real <= x/(1-rel)): returns `next_up(x * (1 + 2*rel))`,
/// which dominates `x/(1-rel)` for `2u <= rel <= 0.25` (the `2*rel` headroom
/// swallows the multiply's own rounding, `next_up` the boundary case).
#[inline]
pub fn certify_up(x: f64, rel: f64) -> f64 {
    debug_assert!((2.0 * UNIT..=0.25).contains(&rel));
    next_up(x * (1.0 + 2.0 * rel))
}

/// Upper-widen a nonneg error EXPRESSION result whose own evaluation used a
/// handful (< 8) of nearest-rounded ops: `next_up(x * (1 + 16u))` dominates
/// `x / (1 - 8u)`.
#[inline]
pub fn slack16(x: f64) -> f64 {
    next_up(x * (1.0 + 16.0 * UNIT))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_up_down_are_one_ulp_and_directed() {
        let x = 1.0_f64;
        assert!(next_up(x) > x);
        assert!(next_down(x) < x);
        assert_eq!(next_down(next_up(x)), x);
        assert!(next_up(0.0) > 0.0);
        assert!(next_down(0.0) < 0.0);
        assert!(next_up(-1.0) > -1.0);
        assert_eq!(next_up(f64::INFINITY), f64::INFINITY);
        assert!(next_up(f64::NAN).is_nan());
    }

    #[test]
    fn gamma_dominates_real_gamma() {
        for n in [1usize, 10, 1000, 100_000] {
            #[allow(clippy::cast_precision_loss)]
            let nu = (n as f64) * UNIT;
            assert!(gamma_n(n) >= nu / (1.0 - nu));
        }
        assert_eq!(gamma_n(1 << 55), 1.0);
    }

    #[test]
    fn gamma_f32_dominates_real_gamma_and_exceeds_f64() {
        for n in [1usize, 10, 1000, 100_000] {
            #[allow(clippy::cast_precision_loss)]
            let nu = (n as f64) * UNIT_F32;
            assert!(gamma_n_f32(n) >= nu / (1.0 - nu));
            // f32 grade is strictly coarser than f64 grade (the whole point).
            assert!(gamma_n_f32(n) > gamma_n(n));
        }
        assert_eq!(gamma_n_f32(1 << 30), 1.0);
    }

    #[test]
    fn certify_up_dominates_inverse_factor() {
        for &(x, rel) in &[(1.0, 1e-13), (1e6, 1e-10), (3.5e-4, 3e-16)] {
            assert!(certify_up(x, rel) >= x / (1.0 - rel));
        }
    }
}
