// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Deterministic random generation of [`Relu1Problem`] instances.
//!
//! Used by the differential test harness and the cross-repo Clean round-trip to
//! exercise the certificate generator over a wide distribution of small
//! exact-rational ReLU networks. Determinism (a seeded LCG, no `rand` crate, no
//! wall-clock) keeps every run reproducible: a failing seed is a permanent,
//! shrinkable witness.

use crate::crown::Relu1Problem;
use crate::rational::Rat;

/// A tiny deterministic linear-congruential generator (Numerical Recipes
/// constants). Not cryptographic — only a reproducible stream of bits.
#[derive(Debug, Clone)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Seed the generator. Distinct seeds give independent-looking streams.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        // Mix the seed so small seeds (0, 1, 2…) don't start in lockstep.
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        // behavior-identical: `s` is u64 and 31 < 64, so `wrapping_shr(31)`
        // equals `>> 31`; it drops the spurious shift-overflow VC.
        s ^= s.wrapping_shr(31);
        Lcg { state: s | 1 }
    }

    // Trust: keep this as an ordinary local implementation. Trust bundles local
    // callees and analyzes their bodies; a vacuous `#[trust::ensures(|_| true)]`
    // neither makes the call opaque nor grants reusable postcondition evidence,
    // and instead creates another contract obligation. A genuine opaque-total
    // boundary would require authenticated verifier semantics; `assume_total`
    // is deliberately not used because it is a recorded assumption rejected by
    // strict verification. This body is total on its own: wrapping arithmetic,
    // a constant in-range shift, and xor cannot panic.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Return the high bits, which have the best statistical quality.
        self.state ^ (self.state >> 33)
    }

    /// Uniform integer in `[lo, hi]` inclusive (`lo <= hi` required — now a
    /// contract-checked precondition; every in-repo caller passes a
    /// constant-bounded range).
    #[trust::requires(lo <= hi)]
    #[trust::ensures(move |r: &i128| lo <= *r && *r <= hi)]
    pub fn range_i128(&mut self, lo: i128, hi: i128) -> i128 {
        debug_assert!(lo <= hi);
        // Exact width-minus-one of [lo, hi], computed in u128 two's complement.
        // Given `lo <= hi` the mathematical difference hi - lo is in [0, 2^128),
        // and (hi as u128).wrapping_sub(lo as u128) equals (hi - lo) mod 2^128,
        // so this wrapping subtraction is exact — unlike the previous
        // `(hi - lo + 1) as u128`, which overflowed i128 for wide ranges and
        // hit a division-by-zero on the full-range call (a latent panic).
        let width = (hi as u128).wrapping_sub(lo as u128);
        let raw = u128::from(self.next_u64());
        // width == u128::MAX iff [lo, hi] is the full i128 range; then every
        // 64-bit draw is a valid offset and `width + 1` would wrap, so branch
        // instead of adding 1. In the else-branch width < u128::MAX, so
        // `wrapping_add(1)` equals `width + 1 >= 1` exactly and `raw % (width+1)
        // <= width`. The `.max(1)` pins the divisor `>= 1` for the verifier
        // (which does not thread the `width < u128::MAX` guard into the modulo,
        // so it cannot otherwise rule out a `% 0` div-by-zero): it is the
        // identity here (`wrapping_add(1) >= 1` already) and removes both the
        // Add-overflow (wrapping) and the div-by-zero obligation.
        let offset = if width == u128::MAX {
            raw
        } else {
            // `checked_rem` instead of `%`: the divisor `width.wrapping_add(1)
            // .max(1)` is provably `>= 1` (the `.max(1)` pins it), so the
            // remainder is always `Some` and the `raw` fallback is unreachable —
            // behaviour-identical to `raw % divisor` on every reachable path.
            // But `checked_rem` returns `Option` with NO MIR zero-divisor
            // assert, so it carries no RemainderByZero panic obligation at all
            // (the u128-width divzero VC the SMT lanes could not discharge).
            let divisor = width.wrapping_add(1).max(1);
            raw.checked_rem(divisor).unwrap_or(raw)
        };
        // offset <= width == hi - lo, so lo + offset is in [lo, hi], which is
        // representable in i128; the u128 two's-complement add therefore
        // round-trips to that exact value (no wrap occurs mathematically).
        //
        // Branch-clamp (not `.clamp`): the two clamping arms are statically
        // UNREACHABLE (candidate is already in [lo, hi] per the argument above)
        // so the value is identical — but each arm lets the `lo <= r <= hi`
        // postcondition discharge from its OWN guard, per-arm, without any
        // width-128 modular reasoning about the wrapping construction.
        let candidate = (lo as u128).wrapping_add(offset) as i128;
        if candidate < lo {
            lo
        } else if candidate > hi {
            hi
        } else {
            candidate
        }
    }

    #[trust::requires(lo <= hi)]
    #[trust::ensures(move |r: &usize| lo <= *r && *r <= hi)]
    fn range_usize(&mut self, lo: usize, hi: usize) -> usize {
        let lo_i = lo as i128;
        let hi_i = hi as i128;
        // The `lo <= hi` contract above implies `lo_i <= hi_i` (usize -> i128
        // is a value-preserving zero-extension). The L0 lane consumes requires
        // only in the assert-refutation lane, so restate the bound as an
        // assert: the refutation lane discharges it from the requires, and the
        // surviving path condition discharges `range_i128`'s precondition in
        // the body VC (a no-op panic for every in-repo caller).
        assert!(lo_i <= hi_i, "range_usize: lo <= hi is contract-checked");
        // `.clamp(lo, hi)` makes the `lo <= r <= hi` postcondition PROVABLE: the
        // verifier models `lo <= hi ⟹ lo <= clamp(_, lo, hi) <= hi` directly, and
        // `lo <= hi` holds here (the assert above, over the value-preserving
        // `lo_i/hi_i`). It is a NO-OP for behavior — `range_i128` already returns
        // a value in `[lo_i, hi_i] = [lo, hi]`, and `try_from` of an in-`[lo,hi]`
        // value succeeds (so `unwrap_or(lo)` is unreachable), leaving `r` already
        // in range — but it lets the postcondition prove WITHOUT consuming
        // `range_i128`'s (u128-wrapping) `#[ensures]`, which the verifier cannot
        // yet establish.
        // Branch-clamp (not `.clamp`): identical value on every reachable path
        // (`r` is already in `[lo, hi]`, and `lo <= hi` is asserted above so the
        // `.clamp` panic edge was unreachable too) — and each arm discharges the
        // `lo <= r <= hi` postcondition from its own guard.
        let r = usize::try_from(self.range_i128(lo_i, hi_i)).unwrap_or(lo);
        if r < lo {
            lo
        } else if r > hi {
            hi
        } else {
            r
        }
    }

    /// A denominator drawn uniformly from `{1, 2, 4}`. Total match instead of
    /// `[1, 2, 4][idx]`: no bounds obligation, and the (statically unreachable)
    /// fallthrough stays inside the {1, 2, 4} set.
    fn small_den(&mut self) -> i128 {
        match self.range_usize(0, 2) {
            0 => 1,
            1 => 2,
            _ => 4,
        }
    }

    /// A small rational with numerator in `[-num_mag, num_mag]` and denominator
    /// drawn from `{1, 2, 4}` — enough to exercise non-integer arithmetic while
    /// keeping the certificate within Clean's `i64` encoding most of the time.
    /// `num_mag >= 0` (contract-checked) keeps the `-num_mag` negation and the
    /// `range_i128(-num_mag, num_mag)` bounds well-formed; all callers pass
    /// small positive literals.
    #[trust::requires(num_mag >= 0)]
    fn small_rat(&mut self, num_mag: i128) -> Rat {
        // `wrapping_neg`: for every contract-legal input (num_mag >= 0) this is
        // exact negation; only the unreachable i128::MIN wraps (to itself),
        // where the old `-num_mag` panicked in debug builds. The wrapping form
        // is total, so no negation-overflow VC is emitted — needed because the
        // L0 lane does not consume the requires yet (contract-assumed=0).
        let neg = num_mag.wrapping_neg();
        // Branch-dup (same idiom as `random_simplex_lp`'s slack draw): the
        // `lo <= hi` precondition of `range_i128` discharges from the guard on
        // EVERY path, without consuming this fn's `num_mag >= 0` requires
        // (which the L0 lane does not consume yet). For every contract-legal
        // input `neg = -num_mag <= 0 <= num_mag`, so the else-arm is
        // statically unreachable and this is behaviour-identical: exactly one
        // draw either way (same LCG stream), same arguments on every
        // reachable path.
        let num = if neg <= num_mag {
            self.range_i128(neg, num_mag)
        } else {
            self.range_i128(num_mag, neg)
        };
        let den = self.small_den();
        // Total match instead of `expect`: `den` is drawn from small_den()'s
        // {1, 2, 4}, so the Err arm is statically unreachable; the fail-soft
        // ZERO keeps the function total (no panic VC / hardened boundary),
        // matching the fail-soft style of range_usize and small_den.
        match Rat::new(num, den) {
            Ok(r) => r,
            Err(_) => Rat::ZERO,
        }
    }
}

/// Generate a random small ReLU-1 problem.
///
/// `max_input` / `max_hidden` bound the layer widths (each width is at least 1).
/// Weights and biases are small rationals; the input box has integer-ish bounds
/// with `lower < upper` on every axis so the box is non-degenerate.
///
/// Both caps must be at most 16 (contract-checked): the generator targets tiny
/// exact-rational networks, and the bound makes every layer `Vec` allocation
/// below provably in range. All in-repo callers pass widths `<= 4`.
#[must_use]
#[trust::requires(max_input <= 16 && max_hidden <= 16)]
pub fn random_problem(seed: u64, max_input: usize, max_hidden: usize) -> Relu1Problem {
    let mut g = Lcg::new(seed);
    // Width bounds: the #[trust::requires] cap (16) documents the contract;
    // the `.min(16)` structural caps below make the bound DOMINATING in the
    // body so the collection-allocation obligations discharge without
    // consuming range_usize's ensures (whose proof chain bottoms out in
    // range_i128's 128-bit wrapping arithmetic, still beyond the verifier).
    // `.min(16)` was havoc'd in earlier toolchains (Ord::min unmodeled at
    // these VC sites); the verifier now emits `min(a, C) <= C` result bounds
    // as function-wide facts on every VC, so the cap is exactly the
    // "dominating check" the allocation checker asks for. Semantics are
    // unchanged for every contract-legal input (n, h <= 16 already).
    let n = g.range_usize(1, max_input.max(1)).min(16);
    let h = g.range_usize(1, max_hidden.max(1)).min(16);

    // `Vec::new()` + push loops (not `collect`/`with_capacity`): the count-form
    // allocation obligations (`min(count, C)` at the collect/capacity site) did
    // not reliably discharge in every lane, so use the crate's proven idiom —
    // `Vec::new()` carries no allocation-size obligation and push growth is
    // amortized noise. Loop counts keep the previous `.min(1_048_576)` caps
    // (identity — every count is <= 16), so draw order and element values are
    // identical. Same convention as `exact::solve_system`.
    let mut w1: Vec<Vec<Rat>> = Vec::new();
    for _ in 0..h.min(1_048_576) {
        let mut row = Vec::new();
        // `.min(16)` is a no-op (`n <= 16` from the dominating cap above),
        // kept so the loop count matches the previous per-row bound exactly.
        for _ in 0..n.min(16) {
            row.push(g.small_rat(3));
        }
        w1.push(row);
    }
    let mut b1 = Vec::new();
    for _ in 0..h.min(1_048_576) {
        b1.push(g.small_rat(2));
    }
    let mut w2 = Vec::new();
    for _ in 0..h.min(1_048_576) {
        w2.push(g.small_rat(3));
    }
    let b2 = g.small_rat(3);

    // `Vec::new()` (not `with_capacity`): same idiom as above — no
    // capacity-hint allocation obligation; the `.min(1_048_576)` loop cap
    // (no-op, `n <= 16`) keeps the push count identical.
    let mut input_lower = Vec::new();
    let mut input_upper = Vec::new();
    for _ in 0..n.min(1_048_576) {
        let lo = g.range_i128(-2, 1);
        let width = g.range_i128(1, 3);
        input_lower.push(Rat::from_int(lo));
        input_upper.push(Rat::from_int(lo.wrapping_add(width)));
    }

    // Half the time use the adaptive default slope; otherwise a random α∈[0,1]
    // per unit (numerator 0..=den so the value stays in range).
    let alpha = if g.range_i128(0, 1) == 0 {
        None
    } else {
        // `Vec::new()` + push loop (not `collect`): same allocation idiom as
        // the layer vectors above; the `.min(1_048_576)` loop cap (no-op,
        // `h <= 16`) keeps the per-unit draw count identical.
        let mut alphas = Vec::new();
        for _ in 0..h.min(1_048_576) {
            // Dominating lower cap: small_den() returns only {1, 2, 4},
            // but its result is an unmodeled call to the verifier, so
            // `range_i128(0, den)`'s `lo <= hi` requires and Rat::new's
            // nonzero-den precondition were spuriously refutable with
            // den <= 0. `.max(1)` re-establishes `den >= 1` structurally.
            // No-op for every real value.
            let den = g.small_den().max(1);
            let num = g.range_i128(0, den);
            // Total: `den >= 1` (small_den ∈ {1,2,4}, capped by max(1)),
            // so `Rat::new` is always `Ok` and the fallback is
            // unreachable — removes the panic obligation the verifier
            // can't discharge over the opaque `small_den()` result.
            alphas.push(Rat::new(num, den).unwrap_or(Rat::ZERO));
        }
        Some(alphas)
    };

    Relu1Problem {
        w1,
        b1,
        w2,
        b2,
        input_lower,
        input_upper,
        alpha,
    }
}

/// Generate a random *feasible* box-truncated-simplex support LP (for SBAR /
/// Pillar 2). Positions `2..=max_pos`, small-rational `g`, and weight bounds
/// `0 ≤ p_lo_j ≤ p_hi_j` constructed so that `Σ p_lo ≤ 1 ≤ Σ p_hi` (feasible).
#[must_use]
#[trust::requires(max_pos <= 16)]
pub fn random_simplex_lp(seed: u64, max_pos: usize) -> crate::sbar::SimplexSupportLp {
    let mut g = Lcg::new(seed ^ 0x5B_A12);
    // Caps mirror random_problem's convention: the requires documents the
    // generator bound (all in-repo callers pass <= 6); `.min(16)` makes the
    // upper bound DOMINATING in the body (position-vector allocations
    // provably bounded) and `.max(2)` the lower bound (`denom = 4m != 0` and
    // downstream `Rat::new` preconditions provable) — both without consuming
    // range_usize's ensures. No-ops for every contract-legal input
    // (range_usize(2, ..) already returns >= 2 and the requires caps at 16).
    #[allow(clippy::manual_clamp)]
    let m = g.range_usize(2, max_pos.max(2)).min(16).max(2);

    // `Vec::new()` + push loop (not `collect`): the crate's proven allocation
    // idiom — no allocation-size obligation, amortized growth. The
    // `.min(1_048_576)` loop cap (no-op — `m <= 16` from `.min(16)` above)
    // keeps the draw count identical. Same convention as `random_problem`.
    let mut gv = Vec::new();
    for _ in 0..m.min(1_048_576) {
        gv.push(g.small_rat(4));
    }

    // Lower bounds: each p_lo_j = a_j / (4m) with a_j ∈ {0,1,2}, so Σ p_lo ≤
    // 2m/(4m) = 1/2 ≤ 1. Upper bounds: p_hi_j = p_lo_j + slack_j with slack_j ≥
    // (something) so that Σ p_hi ≥ 1.
    // `(m as i128).saturating_mul(4)`: `m <= 16` (the `.min(16)` above), so the
    // product is at most 64 — the saturating form is exact for every real input
    // while discharging the hardened i128-Mul overflow boundary (the BV path
    // declines width > 64, so the plain `4 * m` cannot be proven bound-free).
    let denom = (m as i128).saturating_mul(4);
    // `Vec::new()` (not `with_capacity`): no capacity-hint allocation
    // obligation; the `.min(1_048_576)` loop cap keeps the push count identical.
    let mut p_lo = Vec::new();
    let mut p_hi = Vec::new();
    for _ in 0..m.min(1_048_576) {
        let lo_num = g.range_i128(0, 2);
        // `.unwrap_or(Rat::ZERO)`: `denom = 4m != 0`, so `Rat::new` is always
        // `Ok` and the fallback is unreachable — total (no panic obligation).
        let lo = Rat::new(lo_num, denom).unwrap_or(Rat::ZERO);
        // slack numerator in [m, 3m] over denom so each slack ≈ [1/4, 3/4];
        // Σ slack ≥ m·(m/denom) = m/4 ... ensure total ≥ 1 by a generous floor.
        // `saturating_mul` (exact: `m <= 16`, products <= 64): same i128-Mul
        // overflow-boundary discharge as `denom` above.
        // Branch-dup so the `range_i128` precondition `lo <= hi` discharges from
        // the guard on every path: `a = 2m <= 4m = b` always holds (`m >= 0`), so
        // the `else` is unreachable and this is behaviour-identical — but the
        // verifier cannot thread `2m <= 4m` through the i128 `saturating_mul`
        // havoc, whereas each branch establishes the precondition locally.
        let a = (m as i128).saturating_mul(2);
        let b = (m as i128).saturating_mul(4);
        let slack_num = if a <= b {
            g.range_i128(a, b)
        } else {
            g.range_i128(b, a)
        };
        // `denom = 4m` with `m >= 2` (never 0), so `Rat::new` is always `Ok` and
        // the `Rat::ZERO` fallback is unreachable — exact for every generated
        // value, but total (no unwrap panic boundary). Generator-only path.
        let hi = Rat::new(lo_num.wrapping_add(slack_num), denom).unwrap_or(Rat::ZERO);
        p_lo.push(lo);
        p_hi.push(hi);
    }
    crate::sbar::SimplexSupportLp { g: gv, p_lo, p_hi }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        let a = random_problem(42, 3, 4);
        let b = random_problem(42, 3, 4);
        assert_eq!(a.w1, b.w1);
        assert_eq!(a.b2, b.b2);
        assert_eq!(a.input_lower, b.input_lower);
    }

    #[test]
    fn boxes_are_non_degenerate() {
        for seed in 0..200u64 {
            let p = random_problem(seed, 3, 4);
            for (l, u) in p.input_lower.iter().zip(&p.input_upper) {
                assert!(l < u, "degenerate box at seed {seed}");
            }
        }
    }

    #[test]
    fn next_u64_has_no_vacuous_opacity_contract() {
        let source = include_str!("generate.rs");
        let signature = source
            .find("fn next_u64")
            .expect("next_u64 declaration exists");
        let previous_body_end = source[..signature]
            .rfind("\n    }\n")
            .expect("Lcg::new ends before next_u64");
        let declaration_preamble = &source[previous_body_end..signature];

        assert!(
            !declaration_preamble
                .lines()
                .any(|line| line.trim_start().starts_with("#[trust::ensures")),
            "a postcondition does not make next_u64 opaque; keep it as an ordinary bundled callee"
        );
        assert!(
            declaration_preamble.contains("ordinary local implementation"),
            "the source must document Trust's current bundled-callee semantics"
        );
    }
}
