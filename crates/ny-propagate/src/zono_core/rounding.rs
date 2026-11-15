// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Directed-rounding primitives for the certified zonotope core.
//!
//! Every quantity the core publishes is computed in plain IEEE-754 binary64
//! round-to-nearest and then made rigorous by an EXPLICIT charge: Higham
//! `gamma_n` envelopes for dot products and element-wise ops, one-ulp outward
//! steps (`next_up` / `next_down`) wherever a value becomes an endpoint, and a
//! tiny absolute floor ([`TINY`]) that absorbs gradual-underflow error. The
//! discipline is the one `margin_row::rounding` and the Z1 numpy replica use;
//! it is duplicated here (not imported) so the shared core carries no
//! dependency on a lane module.

/// Unit roundoff of binary64 round-to-nearest (`2^-53`).
pub const UNIT: f64 = f64::EPSILON / 2.0;

/// Absolute floor added to every charge. It dominates the total error of any
/// realistic number of gradual-underflow events (each at most `2^-1074`), so a
/// product that underflows can never escape its certificate.
pub const TINY: f64 = 1e-300;

/// Next representable binary64 toward `+inf`. Non-finite inputs pass through.
#[inline]
#[must_use]
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

/// Next representable binary64 toward `-inf`. Non-finite inputs pass through.
#[inline]
#[must_use]
pub fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

/// Higham `gamma_n = n u / (1 - n u)`, rounded UP one ulp so the returned
/// value dominates the real one. Absurd `n` (where `n u >= 1/2`) returns
/// `+inf`: every charge built from it becomes non-finite and the caller's
/// finiteness firewall refuses the computation (fail closed), instead of a
/// saturated constant silently under-charging.
#[inline]
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn gamma(n: usize) -> f64 {
    let nu = n as f64 * UNIT;
    if nu >= 0.5 {
        return f64::INFINITY;
    }
    next_up(nu / (1.0 - nu))
}

/// `next_up(a + b)`: an upper bound on the exact sum of two binary64 values.
#[inline]
#[must_use]
pub fn add_up(a: f64, b: f64) -> f64 {
    next_up(a + b)
}

/// `next_up(a * b)`: an upper bound on the exact product of two NON-NEGATIVE
/// binary64 values.
#[inline]
#[must_use]
pub fn mul_up(a: f64, b: f64) -> f64 {
    next_up(a * b)
}

/// Widen an `f32` interval endpoint pair by one binary64 ulp outward. The
/// published child boxes are `f32`; their exact `f64` images are widened so
/// every downstream midpoint/radius rounding is covered with room to spare.
#[inline]
#[must_use]
pub fn widen_f32_interval(lower: f32, upper: f32) -> (f64, f64) {
    (next_down(f64::from(lower)), next_up(f64::from(upper)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_up_and_down_step_one_ulp_and_pass_non_finite() {
        assert!(next_up(1.0) > 1.0);
        assert!(next_down(1.0) < 1.0);
        assert_eq!(next_up(0.0), f64::from_bits(1));
        assert_eq!(next_down(0.0), -f64::from_bits(1));
        assert!(next_up(f64::INFINITY).is_infinite());
        assert!(next_up(f64::NAN).is_nan());
        assert!(next_down(-1.0) < -1.0);
    }

    #[test]
    fn gamma_dominates_the_real_value_and_refuses_absurd_n() {
        #[allow(clippy::cast_precision_loss)]
        for n in [1usize, 2, 10, 1000, 1 << 20] {
            let real = (n as f64 * UNIT) / (1.0 - n as f64 * UNIT);
            assert!(gamma(n) >= real);
        }
        assert!(gamma(usize::MAX).is_infinite());
    }
}
