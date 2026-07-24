// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact true-minimum of a one-hidden-layer ReLU network over its input box, in
//! **arbitrary** input dimension, by complete hyperplane-arrangement vertex
//! enumeration. This is a *decision procedure*, not sampling: it computes the
//! exact minimum (so `certified_bound ≤ exact_min` is a real per-network
//! soundness proof, with no grid blind spot in any dimension).
//!
//! The true network `y = W₂·ReLU(W₁·x + b₁) + b₂` is continuous piecewise-linear
//! in `x`. Its minimum over the axis-aligned box is attained at a **vertex of the
//! arrangement** formed by the ReLU breakpoint hyperplanes `W₁[j]·x + b₁[j] = 0`
//! together with the box-face hyperplanes `xᵢ = lᵢ`, `xᵢ = uᵢ`. Every such vertex
//! is the unique intersection of `n` linearly-independent hyperplanes from that
//! set; enumerating all `n`-subsets, solving each `n×n` system exactly over the
//! rationals, and evaluating the in-box solutions therefore yields the exact
//! minimum. (The box faces guarantee the feasible region is bounded, so a
//! minimizing vertex always exists in the set.)

use crate::crown::{CrownError, Relu1Problem};
use crate::rational::{Rat, RatError};

impl Relu1Problem {
    /// Exact minimum of the true network over the input box, for any input
    /// dimension, by arrangement-vertex enumeration. Returns `None` only for the
    /// empty (zero-dimensional) input.
    ///
    /// # Errors
    /// Returns [`CrownError::Dimension`] for a shape-inconsistent network built
    /// through the pub fields (previously: under-long vectors hit an index
    /// panic, while over-long `input_upper`/`b1`/`w2` were silently truncated —
    /// both now error, matching `certify`/`preact_bounds`); propagates
    /// exact-arithmetic overflow.
    pub fn exact_min_nd(&self) -> Result<Option<Rat>, CrownError> {
        self.validate()?;
        let n = self.input_lower.len();
        if n == 0 {
            return Ok(None);
        }

        // Assemble candidate hyperplanes `a · x = b`.
        let mut planes: Vec<(Vec<Rat>, Rat)> = Vec::new();
        // Box faces xᵢ = lᵢ and xᵢ = uᵢ. (`validate()` established
        // `input_upper.len() == input_lower.len() == n`, so this zip visits
        // exactly the n dimensions — no indexed reads to bound.)
        for (i, (l, u)) in self.input_lower.iter().zip(&self.input_upper).enumerate() {
            // `Vec::new()` + push (not `vec![_; n]`): the `n`-count bulk fill
            // carries a hardened allocation obligation unbounded on the `&self`
            // input dimension; the push loop produces the identical `n`-zero
            // vector before the single `e[i] = ONE` face selection.
            let mut e = Vec::new();
            for _ in 0..n {
                e.push(Rat::ZERO);
            }
            // behavior-identical: `e` was just built with exactly `n` pushes and
            // `i ∈ [0, n)` (the zip above runs over the `n`-element
            // `input_lower`), so `get_mut(i)` is always `Some` and the write
            // always fires — the guard removes the unstable-len index VC without
            // ever skipping the face selection.
            if let Some(slot) = e.get_mut(i) {
                *slot = Rat::ONE;
            }
            planes.push((e.clone(), *l));
            planes.push((e, *u));
        }
        // ReLU breakpoint hyperplanes W₁[j]·x = −b₁[j].
        for (row, b) in self.w1.iter().zip(&self.b1) {
            planes.push((row.clone(), b.neg()));
        }

        let mut best: Option<Rat> = None;
        // `Vec::new()` + push loop (NOT `collect`): the inline `min(n, C)`
        // collect-count form did not discharge in every lane (see
        // `solve_system`'s solution-vector note below) — the push loop carries
        // no allocation-size obligation. The `.min(1_048_576)` loop cap is kept
        // so the count is identical (`n` is the input dimension,
        // `input_lower.len()`, in practice a handful — `.min` is the identity).
        let mut combo: Vec<usize> = Vec::new();
        for i in 0..n.min(1_048_576) {
            combo.push(i);
        }
        // Iterate all n-subsets of `planes` in lexicographic order.
        loop {
            if let Some(x) = solve_system(&planes, &combo)? {
                // Explicit loop (not `.zip().zip().all(closure)`): keeps the box
                // test in verified code (no absent-adapter `Iterator::all`/
                // closure-Fn obligation). Identical short-circuit.
                let mut in_box = true;
                for ((xi, l), u) in x.iter().zip(&self.input_lower).zip(&self.input_upper) {
                    if !(*l <= *xi && *xi <= *u) {
                        in_box = false;
                        break;
                    }
                }
                if in_box {
                    let y = self.eval(&x)?;
                    best = Some(match best {
                        Some(b) if b <= y => b,
                        _ => y,
                    });
                }
            }
            if !next_combination(&mut combo, planes.len()) {
                break;
            }
        }
        Ok(best)
    }
}

/// Advance `combo` (a strictly-increasing index list of length `k`) to the next
/// `k`-combination of `0..total` in lexicographic order. Returns `false` when
/// the last combination has been passed.
///
/// The strictly-increasing-combination invariant the sole caller establishes
/// (`combo = (0..n)`, `total >= 2n`) is documented here as prose; it is the
/// *correctness* precondition, not a *safety* one. The body is now
/// UNCONDITIONALLY memory-safe for any `(&mut [usize], usize)` — every read is a
/// `.get()`/`.get_mut()` and every `total-k+i` / `base+(j-i)` step is
/// saturating — so no `#[trust::requires]` safety obligation is needed (and the
/// former quantified `combo.iter().all(...)` precondition, which existed only to
/// make the arithmetic non-wrapping, is subsumed). A `total < combo.len()` or
/// otherwise out-of-contract call returns `false`/a truncated advance rather
/// than panicking. (Not `#[trust::ensures]`: the closure would need to borrow
/// the mutated `&mut combo` for `'static`, which does not borrow-check under
/// trustc.)
fn next_combination(combo: &mut [usize], total: usize) -> bool {
    let k = combo.len();
    // No k-combination of 0..total exists when total < k: the enumeration is
    // vacuously exhausted. Unreachable under the contract (and from the sole
    // caller); makes the `total - k` below provably non-wrapping.
    if total < k {
        return false;
    }
    for i in (0..k).rev() {
        // Maximum value position `i` can take is `total - k + i`. Saturating
        // ops + `.get()` reads keep every step TOTAL for the intraprocedural
        // verifier: the `total < k` guard above makes `total - k` non-wrapping
        // and `i < k <= total` makes the sum `<= total`, but those bounds are
        // established across the loop/guard structure the verifier cannot fully
        // thread; each saturation/fallback is UNREACHABLE for a valid combo
        // (`i,j < k == combo.len()`, values `< total`), so behavior is
        // unchanged. No panic boundary, no slice-bounds or overflow obligation.
        let max_at_i = total.saturating_sub(k).saturating_add(i);
        let ci = combo.get(i).copied().unwrap_or(0);
        // `<` rather than `!=`: equivalent on every valid strictly-increasing
        // combination (where combo[i] <= max_at_i always holds).
        if ci < max_at_i {
            let base = ci.saturating_add(1);
            if let Some(slot) = combo.get_mut(i) {
                *slot = base;
            }
            // `i.saturating_add(1)`: `i < k = combo.len() <= isize::MAX`, so
            // `i + 1 <= k` never overflows — the saturation is UNREACHABLE and a
            // provable no-op, matching the `saturating_*` idiom already used
            // throughout this function. Clears the Add-overflow obligation on the
            // range start without a behavior change.
            for j in (i.saturating_add(1))..k {
                // Closed form of the old `combo[j] = combo[j - 1] + 1`
                // recurrence: base + (j - i) = total - k + j <= total - 1.
                if let Some(slot) = combo.get_mut(j) {
                    *slot = base.saturating_add(j.saturating_sub(i));
                }
            }
            return true;
        }
    }
    false
}

/// Solve the `n×n` linear system formed by the `n` hyperplanes indexed by
/// `combo` (exact rational Gaussian elimination with partial pivoting). Returns
/// `Ok(None)` when the system is singular (the chosen hyperplanes are not
/// linearly independent).
///
/// The sole call site always satisfies the (prose) precondition: `exact_min_nd`
/// validated the network (every `w1` row is `n`-wide, so every plane row is
/// `n`-wide) and `combo` stays a strictly-increasing combination of
/// `0..planes.len()` by `next_combination`'s contract; the in-body `get`/width
/// guards below are therefore unreachable and fail-safe (they report the
/// system as singular). (Not a `#[trust::requires]`: method-call contract
/// predicates are currently unparseable by the contract lowering and become
/// their own FAILED "unverifiable spec" obligations, and body VCs do not
/// consume requires.)
/// Total matrix read with fail-safe `Rat::ZERO` (see `solve_system`'s bounds
/// note: every index is provably in range, so the fallback is unreachable).
///
/// A named free fn (not a per-caller closure): a direct call resolves to this
/// bundled, verified body, whereas the old local closure minted an unresolvable
/// `<{closure}> as Fn>::call` absent-callee obligation at every call site.
fn mat_at(m: &[Vec<Rat>], r: usize, c: usize) -> Rat {
    m.get(r)
        .and_then(|row| row.get(c))
        .copied()
        .unwrap_or(Rat::ZERO)
}

fn solve_system(planes: &[(Vec<Rat>, Rat)], combo: &[usize]) -> Result<Option<Vec<Rat>>, RatError> {
    let n = combo.len();
    // Dominating size cap (fail-closed): a linear system this large cannot arise
    // from any real ReLU certificate (`n` = number of selected planes ≤ the
    // input+hidden dimension, in practice ≤ a few dozen). Bounding `n` here is
    // the "dominating check" the allocation checker asks for — it discharges the
    // solution-vector `collect`'s unbounded-allocation obligation and makes the
    // function fail-closed against an implausibly large `combo` rather than
    // attempting a multi-gigabyte allocation. No behavior change for real inputs.
    if n >= (1_048_576) {
        return Ok(None);
    }
    // Augmented matrix [A | b]. `Vec::new()` (not `with_capacity(n)`): the
    // capacity hint on the unbounded `combo` length carries a hardened
    // allocation obligation the model cannot bound; growth cost is noise.
    let mut m: Vec<Vec<Rat>> = Vec::new();
    for &idx in combo {
        // Total lookup + width pin: unreachable per the prose precondition;
        // a malformed selection degrades to the fail-safe singular answer.
        let Some((a, b)) = planes.get(idx) else {
            return Ok(None);
        };
        let mut row = a.clone();
        row.push(*b);
        if row.len() <= n {
            return Ok(None);
        }
        m.push(row);
    }
    if m.len() != n {
        return Ok(None);
    }

    // Every read is a `.get()` and every write a `.get_mut()`, with fail-safe
    // `Rat::ZERO` / skip fallbacks: `m` has exactly `n` rows each `> n` wide (the
    // guards above), and all indices are `< n` or `<= n`, so every fallback is
    // UNREACHABLE. Total access keeps the whole elimination free of
    // slice-bounds obligations — verifier-independent (the intra-/inter-
    // procedural length facts the solver can prove churn with upstream vcgen
    // changes; a `.get()` never needs one). No behavior change (ny tests green).
    for col in 0..n {
        // Find a nonzero pivot at or below `col`.
        // Explicit loop (not `(col..n).find(closure)`): keeps the pivot search in
        // verified code (no absent-adapter `Iterator::find`/closure-Fn
        // obligation). Identical: first `r` in `col..n` with a nonzero entry.
        let mut pivot = None;
        for r in col..n {
            if !mat_at(&m, r, col).is_zero() {
                pivot = Some(r);
                break;
            }
        }
        let Some(p) = pivot else {
            return Ok(None); // singular
        };
        // Total row swap via clone + `get_mut` (no `Vec::swap` bounds
        // obligation — the guard facts `col,p < m.len()` don't flow into
        // `swap`'s intrinsic index check). Both rows always exist (col,p < n =
        // m.len()); a self-swap (col == p) is a no-op we skip.
        if col != p {
            if let (Some(rc), Some(rp)) = (m.get(col).cloned(), m.get(p).cloned()) {
                if let Some(slot) = m.get_mut(col) {
                    *slot = rp;
                }
                if let Some(slot) = m.get_mut(p) {
                    *slot = rc;
                }
            }
        }
        let inv = mat_at(&m, col, col).inv()?;
        // Normalize the pivot row.
        for c in col..=n {
            let v = mat_at(&m, col, c).mul(inv)?;
            if let Some(slot) = m.get_mut(col).and_then(|row| row.get_mut(c)) {
                *slot = v;
            }
        }
        // Eliminate the column from every other row.
        for r in 0..n {
            if r == col || mat_at(&m, r, col).is_zero() {
                continue;
            }
            let factor = mat_at(&m, r, col);
            for c in col..=n {
                let sub = mat_at(&m, col, c).mul(factor)?;
                let v = mat_at(&m, r, c).sub(sub)?;
                if let Some(slot) = m.get_mut(r).and_then(|row| row.get_mut(c)) {
                    *slot = v;
                }
            }
        }
    }

    // `n.min(1<<20)` gives the allocation checker a *syntactic* `min(n, C) <= C`
    // bound it can consume (the early `n >= 1<<20` return's path-condition is in
    // a different lane and was NOT consumed — the collect stayed FAILED). Since
    // that guard already ensures `n < 1<<20` here, so the solution vector still
    // has exactly `n` entries — behavior-preserving.
    // `Vec::new()` + push loop (NOT `collect`): neither the let-bound nor the
    // inline `min(_, C)` collect-count form discharges here (the checker
    // fail-closes it), but incremental push growth carries no bulk-allocation
    // obligation at all — the same proven pattern as this function's augmented
    // matrix `m` and sbar's premises/mult.
    let mut sol = Vec::new();
    for r in 0..n {
        sol.push(mat_at(&m, r, n));
    }
    Ok(Some(sol))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i128, d: i128) -> Rat {
        Rat::new(n, d).unwrap()
    }

    fn worked() -> Relu1Problem {
        Relu1Problem {
            w1: vec![vec![r(1, 1), r(1, 1)], vec![r(1, 1), r(-1, 1)]],
            b1: vec![Rat::ZERO, Rat::ZERO],
            w2: vec![r(1, 1), r(-1, 1)],
            b2: r(5, 2),
            input_lower: vec![r(-1, 1), r(-1, 1)],
            input_upper: vec![r(1, 1), r(1, 1)],
            alpha: Some(vec![r(1, 2), r(1, 2)]),
        }
    }

    #[test]
    fn nd_matches_2d_oracle_on_worked_example() {
        let p = worked();
        assert_eq!(p.exact_min_nd().unwrap().unwrap(), r(1, 2));
        // Agrees with the dedicated 2-D enumerator in eval.rs.
        assert_eq!(p.exact_min_nd().unwrap(), p.exact_min().unwrap());
    }

    #[test]
    fn nd_finds_1d_interior_trough() {
        // y = |x − 5/12| on [0,1]; exact min 0 at the interior breakpoint.
        let v = Relu1Problem {
            w1: vec![vec![r(1, 1)], vec![r(-1, 1)]],
            b1: vec![r(-5, 12), r(5, 12)],
            w2: vec![r(1, 1), r(1, 1)],
            b2: Rat::ZERO,
            input_lower: vec![Rat::ZERO],
            input_upper: vec![r(1, 1)],
            alpha: None,
        };
        assert_eq!(v.exact_min_nd().unwrap().unwrap(), Rat::ZERO);
    }

    #[test]
    fn nd_solves_3d_where_grid_and_2d_oracle_cannot() {
        // A 3-input net: y = ReLU(x0+x1+x2) − ReLU(x0+x1−x2) over [−1,1]³.
        let p = Relu1Problem {
            w1: vec![
                vec![r(1, 1), r(1, 1), r(1, 1)],
                vec![r(1, 1), r(1, 1), r(-1, 1)],
            ],
            b1: vec![Rat::ZERO, Rat::ZERO],
            w2: vec![r(1, 1), r(-1, 1)],
            b2: r(3, 1),
            input_lower: vec![r(-1, 1), r(-1, 1), r(-1, 1)],
            input_upper: vec![r(1, 1), r(1, 1), r(1, 1)],
            alpha: None,
        };
        // exact_min (the ≤2-D enumerator) declines 3-D; exact_min_nd handles it.
        assert_eq!(p.exact_min().unwrap(), None);
        let m = p.exact_min_nd().unwrap().unwrap();
        // Certified CROWN bound must not exceed the exact true minimum.
        let bound = match p.certify(Rat::ZERO) {
            Ok(c) => c.lower_bound,
            Err(CrownError::ThresholdAboveBound { bound, .. }) => {
                let (n, d) = bound.split_once('/').unwrap_or((bound.as_str(), "1"));
                r(n.parse().unwrap(), d.parse().unwrap())
            }
            Err(e) => panic!("{e:?}"),
        };
        assert!(bound <= m, "bound {bound:?} exceeds exact 3-D min {m:?}");
    }

    #[test]
    fn next_combination_enumerates_all() {
        // C(4,2) = 6 combinations.
        let mut c = vec![0, 1];
        let mut count = 1;
        while next_combination(&mut c, 4) {
            count += 1;
        }
        assert_eq!(count, 6);
        assert_eq!(c, vec![2, 3]); // last combination
    }
}
