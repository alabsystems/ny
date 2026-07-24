// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact forward evaluation of a [`Relu1Problem`]'s *true* (un-relaxed) network.
//!
//! This is the independent oracle that makes the proof-carrying pipeline
//! *sound end-to-end*. Clean's external-certificate verifier only checks the
//! linear-program half — that the non-negative multipliers really do derive the
//! claimed bound from the supplied premises. It cannot know whether those
//! premises faithfully over-approximate the actual ReLU network; that obligation
//! lives entirely on the NY side (`crown.rs`).
//!
//! Evaluating the genuine network at points of the input box and checking that
//! its output never dips below the certified lower bound is the differential
//! test of that obligation: if NY ever emitted a premise that was *not* a sound
//! relaxation, the combined certificate could still pass Clean while the true
//! network violated the bound. The checks here would catch exactly that.

use crate::crown::Relu1Problem;
use crate::rational::{Rat, RatError};

impl Relu1Problem {
    /// Evaluate the true network `y = W₂·ReLU(W₁·x + b₁) + b₂` at an exact point.
    ///
    /// `x` must have exactly `input_lower.len()` entries; every in-repo caller
    /// builds `x` with exactly that many pushes. A mis-sized point is rejected
    /// fail-CLOSED with [`RatError::Dimension`] (was a fail-loud `assert_eq!`,
    /// which the strict verifier could not discharge — the length equality is
    /// an interprocedural precondition, and a `#[trust::requires]` method-call
    /// predicate like `x.len() == self.input_lower.len()` is currently
    /// unparseable by the contract lowering).
    ///
    /// # Errors
    /// [`RatError::Dimension`] for a wrong-length point; propagates
    /// exact-arithmetic overflow.
    pub fn eval(&self, x: &[Rat]) -> Result<Rat, RatError> {
        // Fail-CLOSED dimension guard (was a fail-loud `assert_eq!`): sound `Err`
        // on a mis-sized point instead of a panic the strict verifier can't
        // discharge (the length equality is interprocedural). Unreachable for
        // in-repo callers (they size `x` to `input_lower.len()` exactly).
        if x.len() != self.input_lower.len() {
            return Err(RatError::Dimension {
                expected: self.input_lower.len(),
                got: x.len(),
            });
        }
        let h = self.w1.len();
        let mut y = self.b2;
        for j in 0..h {
            // zⱼ = Σ W₁[j][i]·xᵢ + b₁[j]. `b1`/`w2` reads are totalized to
            // fail-CLOSED `.get(j).ok_or(..)?` (never a silent default: these
            // feed the certificate-witness output). `j < h == self.w1.len()`, so
            // for any well-formed problem (`b1.len() == w2.len() == w1.len()`)
            // the `None` arm is unreachable — exact for every real input, but it
            // removes the interprocedural slice-bounds obligations at these reads.
            let mut z = self.b1.get(j).copied().ok_or(RatError::Dimension {
                expected: h,
                got: self.b1.len(),
            })?;
            // Totalize the `w1[j]` read to the same fail-CLOSED `.get(j).ok_or(..)?`
            // idiom as `b1`/`w2` above: `j < h == self.w1.len()`, so the `None` arm
            // is UNREACHABLE for any well-formed problem (exact, no silent default),
            // but making the bound explicit here removes the interprocedural
            // slice-bounds obligation the verifier cannot otherwise connect to
            // `h == self.w1.len()` across the loop.
            let w1_row = self.w1.get(j).ok_or(RatError::Dimension {
                expected: h,
                got: self.w1.len(),
            })?;
            for (wji, xi) in w1_row.iter().zip(x) {
                z = z.add(wji.mul(*xi)?)?;
            }
            // aⱼ = ReLU(zⱼ)
            let a = if z.is_positive() { z } else { Rat::ZERO };
            y = y.add(
                self.w2
                    .get(j)
                    .copied()
                    .ok_or(RatError::Dimension {
                        expected: h,
                        got: self.w2.len(),
                    })?
                    .mul(a)?,
            )?;
        }
        Ok(y)
    }

    /// Sample the input box on a regular rational grid with `steps`+1 points per
    /// axis (so `(steps+1)^dim` points total), returning the exact minimum of the
    /// true network over the grid, or `None` if the dimension is empty.
    ///
    /// Used as a *necessary* soundness witness: the CROWN lower bound must not
    /// exceed the true minimum, hence must not exceed this sampled minimum.
    ///
    /// # Errors
    /// Propagates exact-arithmetic overflow.
    pub fn grid_min(&self, steps: u32) -> Result<Option<Rat>, RatError> {
        let n = self.input_lower.len();
        if n == 0 {
            return Ok(None);
        }
        let denom = Rat::from_int(i128::from(steps.max(1)));
        // `Vec::new()` + push (not `vec![_; n]`): the `n`-count bulk fill carries
        // a hardened allocation obligation the model cannot bound on the
        // unbounded `&self` input dimension; the push loop yields the identical
        // `n`-zero multi-index and its amortized growth is noise next to the Rat
        // grid sweep.
        let mut idx = Vec::new();
        #[allow(clippy::same_item_push)]
        for _ in 0..n {
            idx.push(0u32);
        }
        let mut best: Option<Rat> = None;
        loop {
            // Build the point for the current multi-index. `Vec::new()` (not
            // `with_capacity(n)`): the capacity hint on the unbounded `&self`
            // dimension carries a hardened allocation obligation the model
            // cannot bound; amortized growth is noise next to the Rat math.
            let mut x = Vec::new();
            // Index-free build over the parallel (digit, lo, hi) triple. `idx`
            // has length `n == input_lower.len()`, so for a well-formed problem
            // (`input_upper.len() == input_lower.len()`) this zip visits exactly
            // the `n` dimensions — identical to the old `for i in 0..n` — while
            // removing the three slice-bounds obligations. A length mismatch just
            // truncates `x`, which `eval` then rejects fail-closed via its
            // existing `x.len() != input_lower.len()` dimension guard.
            for ((&digit, lo), hi) in idx.iter().zip(&self.input_lower).zip(&self.input_upper) {
                let frac = Rat::from_int(i128::from(digit)).mul(denom.inv()?)?;
                let span = hi.sub(*lo)?;
                x.push(lo.add(frac.mul(span)?)?);
            }
            let y = self.eval(&x)?;
            best = Some(match best {
                Some(b) if b <= y => b,
                _ => y,
            });
            // Increment the mixed-radix counter (radix steps+1 per axis).
            // A digit is bumped only while strictly below `steps`, so no digit
            // ever exceeds `steps` — no u32 overflow even for
            // `steps == u32::MAX` (the old bump-then-test cursor wrapped
            // there) — and `iter_mut` replaces the manually indexed carry
            // cursor, leaving no bounds obligation.
            let mut advanced = false;
            for digit in &mut idx {
                if *digit < steps {
                    *digit = (*digit).saturating_add(1);
                    advanced = true;
                    break;
                }
                *digit = 0;
            }
            if !advanced {
                return Ok(best);
            }
        }
    }

    /// Exact minimum of the true network over the input box, for `input_dim ∈
    /// {1, 2}`, returning `None` for higher dimensions.
    ///
    /// The network is continuous piecewise-linear, so its minimum over the box
    /// is attained at a *vertex of the ReLU breakpoint hyperplane arrangement*:
    /// a box corner, a box-edge ∩ breakpoint-line intersection, or a
    /// breakpoint-line ∩ breakpoint-line intersection. Enumerating those exact
    /// rational points and evaluating each yields the true minimum with **no
    /// grid blind spot** — unlike [`grid_min`], a fixed grid can miss an
    /// interior trough that sits between sample points (see the harness-adequacy
    /// finding in `crates/ny-cert/SPEC.md`).
    ///
    /// [`grid_min`]: Self::grid_min
    ///
    /// # Errors
    /// Propagates exact-arithmetic overflow.
    pub fn exact_min(&self) -> Result<Option<Rat>, RatError> {
        let n = self.input_lower.len();
        let candidates = match n {
            1 => self.candidates_1d(),
            2 => self.candidates_2d()?,
            _ => return Ok(None),
        };
        let mut best: Option<Rat> = None;
        for x in candidates {
            // Only points inside the closed box are feasible.
            if !self.in_box(&x) {
                continue;
            }
            let y = self.eval(&x)?;
            best = Some(match best {
                Some(b) if b <= y => b,
                _ => y,
            });
        }
        Ok(best)
    }

    fn in_box(&self, x: &[Rat]) -> bool {
        // Explicit loop (not `.zip().zip().all(closure)`): keeps the box test in
        // verified code (no absent-adapter `Iterator::all`/closure-Fn obligation).
        // `zip` in the `for` header is recognized; identical short-circuit.
        for ((xi, l), u) in x.iter().zip(&self.input_lower).zip(&self.input_upper) {
            if !(*l <= *xi && *xi <= *u) {
                return false;
            }
        }
        true
    }

    /// 1-D candidates: box endpoints + each unit's breakpoint `x = −b₁[j]/W₁[j]`.
    fn candidates_1d(&self) -> Vec<Vec<Rat>> {
        // Total `.get()` reads with fail-safe `Rat::ZERO`: `exact_min` calls this
        // only for `input_dim == 1`, so `input_lower/upper` have ≥1 elem and each
        // `w1` row has ≥1 weight — every fallback is unreachable. Keeps the
        // constant-index reads free of slice-bounds obligations, which upstream
        // vcgen changes intermittently stop proving (verifier-independent).
        let il0 = self.input_lower.first().copied().unwrap_or(Rat::ZERO);
        let iu0 = self.input_upper.first().copied().unwrap_or(Rat::ZERO);
        // `Vec::new()` + push via `point1` (not `vec![…]` literals): the macro's
        // internal boxed-slice `into_vec` inlines hardened alloc/arith
        // obligations into this fn; identical points, identical order.
        let mut pts: Vec<Vec<Rat>> = Vec::new();
        pts.push(point1(il0));
        pts.push(point1(iu0));
        for (row, b) in self.w1.iter().zip(&self.b1) {
            let w = row.first().copied().unwrap_or(Rat::ZERO);
            if !w.is_zero() {
                if let Ok(x) = b.neg().mul(w.inv().unwrap_or(Rat::ZERO)) {
                    pts.push(point1(x));
                }
            }
        }
        pts
    }

    /// 2-D candidates: the 4 box corners, each box-edge ∩ breakpoint-line, and
    /// each breakpoint-line ∩ breakpoint-line intersection.
    #[allow(clippy::vec_init_then_push)] // deliberate: avoids the vec! macro (see pts below)
    fn candidates_2d(&self) -> Result<Vec<Vec<Rat>>, RatError> {
        // Total `.get()` reads via `elem_or_zero` (fail-safe `Rat::ZERO`):
        // reached only for `input_dim == 2`, so `input_lower/upper` have ≥2
        // elems and each `w1` row ≥2 weights — fallbacks unreachable.
        // Verifier-independent bounds.
        let (l0, u0) = (
            elem_or_zero(&self.input_lower, 0),
            elem_or_zero(&self.input_upper, 0),
        );
        let (l1, u1) = (
            elem_or_zero(&self.input_lower, 1),
            elem_or_zero(&self.input_upper, 1),
        );
        // `Vec::new()` + push via `point2` (not `vec![…]` literals): the macro's
        // internal boxed-slice `into_vec` inlines hardened alloc/arith
        // obligations into this fn; identical corners, identical order.
        let mut pts: Vec<Vec<Rat>> = Vec::new();
        pts.push(point2(l0, l1));
        pts.push(point2(l0, u1));
        pts.push(point2(u0, l1));
        pts.push(point2(u0, u1));
        // Breakpoint line j: a·x0 + b·x1 = c, with a=W1[j][0], b=W1[j][1], c=−b1[j].
        // Explicit Vec::new()+push (not `.collect()`): the length is the (zipped)
        // hidden-layer count, an input-derived count the verifier cannot bound,
        // so a bulk `.collect()` raises an UnboundedAllocation obligation. The
        // loop has no bulk-alloc obligation — identical elements and order.
        let mut lines: Vec<(Rat, Rat, Rat)> = Vec::new();
        for (row, bias) in self.w1.iter().zip(&self.b1) {
            lines.push((elem_or_zero(row, 0), elem_or_zero(row, 1), bias.neg()));
        }

        // Line ∩ vertical/horizontal box edges.
        for &(a, b, c) in &lines {
            // x0 fixed at l0 / u0  ⇒ solve for x1 (needs b ≠ 0).
            if !b.is_zero() {
                for &x0 in &[l0, u0] {
                    let x1 = c.sub(a.mul(x0)?)?.mul(b.inv()?)?;
                    pts.push(point2(x0, x1));
                }
            }
            // x1 fixed at l1 / u1  ⇒ solve for x0 (needs a ≠ 0).
            if !a.is_zero() {
                for &x1 in &[l1, u1] {
                    let x0 = c.sub(b.mul(x1)?)?.mul(a.inv()?)?;
                    pts.push(point2(x0, x1));
                }
            }
        }
        // Line ∩ line: 2×2 solve where the determinant is non-zero.
        let z3 = (Rat::ZERO, Rat::ZERO, Rat::ZERO);
        for i in 0..lines.len() {
            for j in (i.saturating_add(1))..lines.len() {
                let (a1, b1, c1) = lines.get(i).copied().unwrap_or(z3);
                let (a2, b2, c2) = lines.get(j).copied().unwrap_or(z3);
                let det = a1.mul(b2)?.sub(a2.mul(b1)?)?;
                if det.is_zero() {
                    continue;
                }
                let x0 = c1.mul(b2)?.sub(c2.mul(b1)?)?.mul(det.inv()?)?;
                let x1 = a1.mul(c2)?.sub(a2.mul(c1)?)?.mul(det.inv()?)?;
                pts.push(point2(x0, x1));
            }
        }
        Ok(pts)
    }
}

/// Total slice read with fail-safe `Rat::ZERO` (see `candidates_2d`'s bounds
/// note: every index is provably in range, so the fallback is unreachable).
///
/// A named free fn (not a per-caller closure): a direct call resolves to this
/// bundled, verified body, whereas the old local closure minted an unresolvable
/// `<{closure}> as Fn>::call` absent-callee obligation at every call site.
fn elem_or_zero(v: &[Rat], i: usize) -> Rat {
    v.get(i).copied().unwrap_or(Rat::ZERO)
}

/// Single-point candidate `[x]` built by `Vec::new()` + push (not `vec![x]`):
/// the macro's internal boxed-slice `into_vec` carries hardened alloc/arith
/// obligations inlined at every call site; the push builds the identical
/// one-element point. A named free fn (not a per-caller closure) for the same
/// absent-callee reason as [`elem_or_zero`].
#[allow(clippy::vec_init_then_push)] // deliberate: avoids the vec! macro (see above)
fn point1(x: Rat) -> Vec<Rat> {
    let mut p = Vec::new();
    p.push(x);
    p
}

/// Two-coordinate candidate `[x0, x1]`; same rationale as [`point1`].
#[allow(clippy::vec_init_then_push)] // deliberate: avoids the vec! macro (see point1)
fn point2(x0: Rat, x1: Rat) -> Vec<Rat> {
    let mut p = Vec::new();
    p.push(x0);
    p.push(x1);
    p
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
    fn eval_matches_hand_values() {
        let p = worked();
        // At x = (0,0): z=(0,0), a=(0,0), y = 5/2.
        assert_eq!(p.eval(&[Rat::ZERO, Rat::ZERO]).unwrap(), r(5, 2));
        // At x = (1,0): z=(1,1), a=(1,1), y = 1 - 1 + 5/2 = 5/2.
        assert_eq!(p.eval(&[r(1, 1), Rat::ZERO]).unwrap(), r(5, 2));
        // At x = (-1,0): z=(-1,-1), a=(0,0), y = 5/2.
        assert_eq!(p.eval(&[r(-1, 1), Rat::ZERO]).unwrap(), r(5, 2));
        // At x = (0,1): z=(1,-1), a=(1,0), y = 1 - 0 + 5/2 = 7/2.
        assert_eq!(p.eval(&[Rat::ZERO, r(1, 1)]).unwrap(), r(7, 2));
        // At x = (0,-1): z=(-1,1), a=(0,1), y = 0 - 1 + 5/2 = 3/2.
        assert_eq!(p.eval(&[Rat::ZERO, r(-1, 1)]).unwrap(), r(3, 2));
    }

    #[test]
    fn exact_min_finds_interior_trough_a_grid_misses() {
        // y = |x − 5/12| = ReLU(x − 5/12) + ReLU(5/12 − x), box [0,1]. The true
        // minimum is 0 at the interior breakpoint x = 5/12, which a coarse
        // regular grid steps over. This is the harness-adequacy witness.
        let v_shape = Relu1Problem {
            w1: vec![vec![r(1, 1)], vec![r(-1, 1)]],
            b1: vec![r(-5, 12), r(5, 12)],
            w2: vec![r(1, 1), r(1, 1)],
            b2: Rat::ZERO,
            input_lower: vec![Rat::ZERO],
            input_upper: vec![r(1, 1)],
            alpha: None,
        };
        // grid_min(6) samples {0,1/6,…,1} and never sees x=5/12: it over-reports.
        assert_eq!(v_shape.grid_min(6).unwrap().unwrap(), r(1, 12));
        // exact_min enumerates the breakpoint x=5/12 and returns the true 0.
        assert_eq!(v_shape.exact_min().unwrap().unwrap(), Rat::ZERO);
    }

    #[test]
    fn exact_min_matches_grid_when_grid_is_aligned() {
        // For the worked 2-D example the true min 1/2 is attained at a corner,
        // so both oracles agree — and exact_min is the off-grid-safe witness.
        let p = worked();
        assert_eq!(p.exact_min().unwrap().unwrap(), r(1, 2));
        let bound = p.certify(Rat::ZERO).unwrap().lower_bound;
        assert!(bound <= p.exact_min().unwrap().unwrap());
    }

    #[test]
    fn grid_min_is_above_certified_bound() {
        let p = worked();
        let bound = p.certify(Rat::ZERO).unwrap().lower_bound; // 1/2
        let gmin = p.grid_min(8).unwrap().unwrap();
        // The certified lower bound never exceeds the true (sampled) minimum.
        assert!(bound <= gmin, "bound {bound:?} exceeded grid min {gmin:?}");
        // For this net the true min 1/2 is attained on the grid (e.g. corner).
        assert_eq!(gmin, r(1, 2));
    }
}
