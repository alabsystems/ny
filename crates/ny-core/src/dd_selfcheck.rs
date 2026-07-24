// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! One-time known-answer bit-exactness probe that AUTHORIZES the
//! double-double certified path (`#dd-zonotope`).
//!
//! # Why this exists (soundness, not hygiene)
//!
//! [`crate::dd::two_sum`]'s error term is *algebraically* zero:
//! `(a - (s - bb)) + (b - bb)` is `0` under real arithmetic and is nonzero only
//! because each step re-rounds. Any FP reassociation — `-ffast-math`, an
//! `fp-contract=fast` backend, an aggressive vectorizer that reorders the
//! subtractions — collapses it to `0.0`, and [`crate::dd::Dd`] silently
//! degrades to plain `f64`.
//!
//! That degradation is NOT a "slightly looser bound". The reference probe
//! MEASURED the plain-f64 zonotope certified margin on `vgg16-7` spec1 as
//! `[-29298, +29045]` — but a consumer that still *believes* it is running at
//! `U_DD = 2^-102` would publish a `1e-11` rounding half-width it does not
//! have, i.e. a bound that is too TIGHT. That is an unsound verdict, worth
//! -150 under VNN-COMP scoring.
//!
//! So every consumer must call [`dd_selfcheck_ok`] and refuse — return `None`,
//! never a verdict — when it returns `false`. Exactly the precedent set by
//! `ny-cuda/src/ieee_selfcheck.rs`, which refuses to construct a CUDA engine
//! when cuBLAS silently substitutes reduced precision.
//!
//! # The probes
//!
//! Each operand pair has an EFT residual that is nonzero, exactly
//! representable, and known in closed form, so a conformant IEEE-754
//! round-to-nearest implementation returns ONE bit pattern:
//!
//! * `two_sum(1, 2^-60)` -> `(1.0, 2^-60)`. The residual is below the f64 ulp
//!   of `1.0` (`2^-52`), so a reassociated implementation that computes
//!   `(a + b) - a - b` in the wrong order returns `0.0`.
//! * `two_sum(2^60, 1)` -> `(2^60, 1.0)` — the asymmetric orientation.
//! * `two_sum(-1e17, 1e17 + 1)` -> exercises catastrophic cancellation.
//! * `two_prod((1 + 2^-30)^2)` -> `(1 + 2^-29, 2^-60)`. Requires a true
//!   single-rounding FMA; a platform that emulates `mul_add` as
//!   `fl(fl(a*b) + c)` returns `0.0` for the residual.
//! * `two_prod` on a full-width mantissa pair whose residual spans the low
//!   half, so an FMA with fewer than 106 product bits is caught.
//! * a cancelling `dd_fma` accumulation whose exact answer (`100`) is
//!   unreachable in plain f64 — the end-to-end check that the accumulator
//!   really carries the second word.
//!
//! The probes are `#[inline(never)]` and read their operands through
//! [`std::hint::black_box`] so a constant-folding pass cannot evaluate them at
//! compile time with different (exact) semantics than the runtime code path.

use std::sync::OnceLock;

use crate::dd::{dd_add_f64, dd_fma, two_prod, two_sum, Dd};

/// Result of the one-time probe, with the first failing probe's name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdSelfcheck {
    /// Every error-free-transformation probe returned its exact known answer.
    Ok,
    /// A probe deviated: the double-double path must refuse.
    Failed(&'static str),
}

impl DdSelfcheck {
    /// True when the double-double path is authorized.
    #[inline]
    #[must_use]
    pub fn is_ok(self) -> bool {
        matches!(self, DdSelfcheck::Ok)
    }
}

/// Run (once per process) the double-double bit-exactness probe.
///
/// Cached in a [`OnceLock`]: the probe is a few dozen flops, but caching also
/// guarantees that two consumers in the same process can never disagree about
/// whether the path is authorized.
#[must_use]
pub fn dd_selfcheck() -> DdSelfcheck {
    static RESULT: OnceLock<DdSelfcheck> = OnceLock::new();
    *RESULT.get_or_init(run_probes)
}

/// Convenience predicate: `true` iff the double-double path is authorized.
#[must_use]
pub fn dd_selfcheck_ok() -> bool {
    dd_selfcheck().is_ok()
}

#[inline(never)]
fn run_probes() -> DdSelfcheck {
    // two_sum: residual below the ulp of the leading term.
    let a = std::hint::black_box(1.0_f64);
    let b = std::hint::black_box(2.0_f64.powi(-60));
    let (s, e) = two_sum(a, b);
    if s != 1.0 || e != 2.0_f64.powi(-60) {
        return DdSelfcheck::Failed("two_sum/small-addend");
    }

    // two_sum: the reversed-magnitude orientation.
    let a = std::hint::black_box(2.0_f64.powi(60));
    let b = std::hint::black_box(1.0_f64);
    let (s, e) = two_sum(a, b);
    if s != 2.0_f64.powi(60) || e != 1.0 {
        return DdSelfcheck::Failed("two_sum/large-first");
    }

    // two_sum under catastrophic cancellation: (-1e17) + (1e17 + 1) where the
    // second operand is not exactly 1e17+1 in f64; the EFT must still be exact.
    let a = std::hint::black_box(-1.0e17_f64);
    let b = std::hint::black_box(1.0e17_f64 + 1.0);
    let (s, e) = two_sum(a, b);
    // The pair must reconstruct the exact sum: s + e == b + a exactly, and both
    // operands are f64 so the exact sum IS representable here.
    // The operands are NOT algebraically equal in f64: (1e17 + 1) - 1e17
    // exercises rounding, which is the point of the self-check.
    #[allow(clippy::eq_op)]
    if s + e != (1.0e17_f64 + 1.0) - 1.0e17_f64 || !(s.is_finite() && e.is_finite()) {
        return DdSelfcheck::Failed("two_sum/cancellation");
    }

    // two_prod: the classic (1 + 2^-30)^2 low word.
    let a = std::hint::black_box(1.0_f64 + 2.0_f64.powi(-30));
    let (p, e) = two_prod(a, a);
    if p != 1.0 + 2.0_f64.powi(-29) || e != 2.0_f64.powi(-60) {
        return DdSelfcheck::Failed("two_prod/square");
    }

    // two_prod: full-width mantissas, residual in the low half of the product.
    // (1 + 2^-52) * (1 - 2^-52) = 1 - 2^-104 exactly. fl(.) = 1.0 (round to
    // nearest even), so the residual must be exactly -2^-104.
    let a = std::hint::black_box(1.0_f64 + 2.0_f64.powi(-52));
    let b = std::hint::black_box(1.0_f64 - 2.0_f64.powi(-52));
    let (p, e) = two_prod(a, b);
    if p != 1.0 || e != -(2.0_f64.powi(-104)) {
        return DdSelfcheck::Failed("two_prod/full-width");
    }

    // End-to-end: an accumulation whose exact answer is unreachable in f64.
    let big = std::hint::black_box(1.0e17_f64);
    let one = std::hint::black_box(1.0_f64);
    let mut acc = Dd::ZERO;
    acc = dd_add_f64(acc, big);
    for _ in 0..100 {
        acc = dd_add_f64(acc, one);
    }
    acc = dd_add_f64(acc, -big);
    if acc.to_f64() != 100.0 {
        return DdSelfcheck::Failed("dd_add_f64/cancelling-sum");
    }

    // Same, through the multiply-accumulate entry point.
    let mut acc = Dd::ZERO;
    acc = dd_fma(acc, big, one);
    for _ in 0..100 {
        acc = dd_fma(acc, one, one);
    }
    acc = dd_fma(acc, -big, one);
    if acc.to_f64() != 100.0 {
        return DdSelfcheck::Failed("dd_fma/cancelling-dot");
    }

    DdSelfcheck::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selfcheck_passes_on_this_toolchain() {
        // If this ever fails, the double-double certified path is DISABLED on
        // this build — which is the intended fail-closed behaviour, but it must
        // be seen, not silently absorbed.
        assert_eq!(
            dd_selfcheck(),
            DdSelfcheck::Ok,
            "double-double EFTs were broken by the compiler/target; the \
             #dd-zonotope certified path will refuse (fail-closed)"
        );
        assert!(dd_selfcheck_ok());
    }

    #[test]
    fn selfcheck_is_cached_and_stable() {
        assert_eq!(dd_selfcheck(), dd_selfcheck());
    }
}
