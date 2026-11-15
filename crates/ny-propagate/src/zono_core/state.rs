// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The certified zonotope state.
//!
//! A [`ZonoState`] over `n` scalar values represents the set
//!
//! ```text
//! { c + sum_k G[k] * e_k + rho  :  e_k in [-1, 1],  |rho_j| <= r_j }
//! ```
//!
//! in EXACT real arithmetic, where `c` is the centre, `G[k]` is generator row
//! `k` (one row per noise symbol, stored row-major `K x n`), and `r >= 0` is an
//! independent per-element interval remainder that absorbs every rounding
//! error, every folded generator and every model error (e.g. the certified
//! BatchNorm coefficient radii). Each generator row carries an explicit
//! [`SymbolId`]: two states are combined by ID, never by row position (the
//! Vibecheck U8 defect class, same shape as ny's retracted N-ary Concat).
//!
//! INVARIANT (what every transformer must preserve): if the true value of the
//! tracked quantity lies in the represented set for the current symbol
//! assignment, it still does after the transformer, for the SAME assignment
//! of every pre-existing symbol and SOME value of each fresh symbol.

use rayon::prelude::*;

use super::ids::{SymbolAllocator, SymbolId};
use super::kernels::RowPar;
use super::rounding::{gamma, next_down, next_up, TINY, UNIT};
use super::ZonoRefusal;

/// Columns per task of the column-parallel passes (#m8-cgan-leaf-batch). A
/// scheduling constant only: every column is computed by exactly one task, in
/// the reference's row order, whatever this value is.
pub(crate) const COL_BLOCK: usize = 2048;

/// One input noise symbol: the flat input dimension it spans and the exact
/// binary64 midpoint/radius it was built from, so a caller can map a concrete
/// input `x` to its symbol value `e = (x_dim - center) / radius`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputSymbol {
    /// The symbol's explicit ID.
    pub id: SymbolId,
    /// Flat input dimension the symbol spans.
    pub dim: usize,
    /// Binary64 midpoint used as the centre of that dimension.
    pub center: f64,
    /// Binary64 radius stored as the generator entry.
    pub radius: f64,
}

/// Certified zonotope over `n` values. See the module docs for the semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct ZonoState {
    n: usize,
    center: Vec<f64>,
    gens: Vec<f64>,
    ids: Vec<SymbolId>,
    rem: Vec<f64>,
}

impl ZonoState {
    /// Build a state from its parts, validating shapes, ID uniqueness and the
    /// remainder's sign and finiteness.
    pub fn from_parts(
        n: usize,
        center: Vec<f64>,
        gens: Vec<f64>,
        ids: Vec<SymbolId>,
        rem: Vec<f64>,
    ) -> Result<Self, ZonoRefusal> {
        if center.len() != n || rem.len() != n {
            return Err(ZonoRefusal::Shape(format!(
                "zonotope parts: centre {} / remainder {} for n = {n}",
                center.len(),
                rem.len()
            )));
        }
        let expected = ids
            .len()
            .checked_mul(n)
            .ok_or_else(|| ZonoRefusal::Shape("zonotope generator size overflows".to_string()))?;
        if gens.len() != expected {
            return Err(ZonoRefusal::Shape(format!(
                "zonotope parts: {} generator values for {} symbols x {n}",
                gens.len(),
                ids.len()
            )));
        }
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ZonoRefusal::Internal(
                "zonotope parts: duplicate symbol id".to_string(),
            ));
        }
        if rem.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(ZonoRefusal::NonFinite("zonotope remainder"));
        }
        Ok(Self {
            n,
            center,
            gens,
            ids,
            rem,
        })
    }

    /// Unchecked constructor for transformers that build parts they already
    /// know to be consistent (shape and ID uniqueness are debug-asserted).
    pub(crate) fn from_parts_trusted(
        n: usize,
        center: Vec<f64>,
        gens: Vec<f64>,
        ids: Vec<SymbolId>,
        rem: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(center.len(), n);
        debug_assert_eq!(rem.len(), n);
        debug_assert_eq!(gens.len(), ids.len() * n);
        Self {
            n,
            center,
            gens,
            ids,
            rem,
        }
    }

    /// The zonotope of an axis-aligned box `[lower, upper]`.
    ///
    /// The box is first widened by one binary64 ulp on each side (the child
    /// boxes are `f32`; this also covers any decimal-to-binary slack a caller
    /// may have left). The centre is the midpoint and every NON-degenerate
    /// dimension gets its own input symbol whose generator is the radius. The
    /// remainder `8u(|c| + rad) + TINY` covers the rounding of the midpoint and
    /// radius, and the enclosure of the widened box is re-checked explicitly.
    ///
    /// Input symbols are allocated FIRST, so they hold IDs `0..n_var` of
    /// `alloc`'s sequence when `alloc` is fresh.
    pub fn from_box(
        lower: &[f64],
        upper: &[f64],
        alloc: &mut SymbolAllocator,
    ) -> Result<(Self, Vec<InputSymbol>), ZonoRefusal> {
        if lower.len() != upper.len() {
            return Err(ZonoRefusal::Shape(format!(
                "input box: lower {} vs upper {}",
                lower.len(),
                upper.len()
            )));
        }
        let n = lower.len();
        let mut center = Vec::with_capacity(n);
        let mut rem = Vec::with_capacity(n);
        let mut radius = Vec::with_capacity(n);
        for (index, (&lo_raw, &hi_raw)) in lower.iter().zip(upper).enumerate() {
            if !lo_raw.is_finite() || !hi_raw.is_finite() || lo_raw > hi_raw {
                return Err(ZonoRefusal::Shape(format!(
                    "input box dimension {index} is not a finite interval: [{lo_raw}, {hi_raw}]"
                )));
            }
            // A dimension the caller fixed (`lo == hi`) gets no symbol: its
            // one-ulp widening goes into the remainder instead.
            let has_symbol = hi_raw > lo_raw;
            let lo = next_down(lo_raw);
            let hi = next_up(hi_raw);
            let mid = f64::midpoint(lo, hi);
            let rad = (hi - lo) / 2.0;
            let generator = if has_symbol { rad } else { 0.0 };
            // The midpoint and radius each carry at most one rounding (the
            // halvings are exact); 8u(|c| + rad) covers them AND the ulp steps
            // of the explicit check below with a wide margin.
            let folded = if has_symbol { 0.0 } else { rad };
            let r = next_up(folded + 8.0 * UNIT * (mid.abs() + rad) + TINY);
            // Explicit enclosure check of the widened box: an UPPER bound on
            // the exact `mid - g - r` must not exceed `lo`, and a LOWER bound
            // on the exact `mid + g + r` must reach `hi`.
            if next_up(next_up(mid - generator) - r) > lo
                || next_down(next_down(mid + generator) + r) < hi
            {
                return Err(ZonoRefusal::Internal(format!(
                    "input box dimension {index}: midpoint/radius enclosure check failed"
                )));
            }
            center.push(mid);
            rem.push(r);
            radius.push(generator);
        }
        let mut symbols = Vec::new();
        let mut ids = Vec::new();
        let mut gens = Vec::new();
        for (dim, &rad) in radius.iter().enumerate() {
            if rad > 0.0 {
                let id = alloc.fresh();
                ids.push(id);
                let start = gens.len();
                gens.resize(start + n, 0.0);
                gens[start + dim] = rad;
                symbols.push(InputSymbol {
                    id,
                    dim,
                    center: center[dim],
                    radius: rad,
                });
            }
        }
        Ok((Self::from_parts_trusted(n, center, gens, ids, rem), symbols))
    }

    /// Number of represented scalar values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// True when the state represents no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Number of generator rows (noise symbols).
    #[must_use]
    pub fn num_generators(&self) -> usize {
        self.ids.len()
    }

    /// Centre vector.
    #[must_use]
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Row-major `K x n` generator matrix.
    #[must_use]
    pub fn generators(&self) -> &[f64] {
        &self.gens
    }

    /// Generator row `k`.
    #[must_use]
    pub fn generator_row(&self, k: usize) -> &[f64] {
        &self.gens[k * self.n..(k + 1) * self.n]
    }

    /// Symbol IDs, one per generator row.
    #[must_use]
    pub fn ids(&self) -> &[SymbolId] {
        &self.ids
    }

    /// Per-element remainder radius.
    #[must_use]
    pub fn remainder(&self) -> &[f64] {
        &self.rem
    }

    /// Largest remainder radius (the state's certified charge).
    #[must_use]
    pub fn max_remainder(&self) -> f64 {
        self.rem.iter().copied().fold(0.0, f64::max)
    }

    /// Certified upper bound on `sum_k |G[k]_j|` for every element `j`.
    ///
    /// The binary64 sum of `K` non-negative terms is below the exact one by at
    /// most a relative `gamma_K`; multiplying by `1 + gamma_{K+2}` and stepping
    /// one ulp up dominates it.
    #[must_use]
    pub fn abs_generator_sum(&self) -> Vec<f64> {
        let mut acc = vec![0.0f64; self.n];
        let k = self.ids.len();
        if k == 0 {
            return acc;
        }
        for row in self.gens.chunks_exact(self.n.max(1)) {
            for (slot, &value) in acc.iter_mut().zip(row) {
                *slot += value.abs();
            }
        }
        let factor = 1.0 + gamma(k + 2);
        for slot in &mut acc {
            *slot = next_up(*slot * factor);
        }
        acc
    }

    /// [`Self::abs_generator_sum`] under a row schedule (#m8-cgan-leaf-batch).
    ///
    /// [`RowPar::Rayon`] splits the COLUMNS into blocks of [`COL_BLOCK`]; each
    /// task sums every column of its block over ALL rows in the reference's
    /// row order and applies the same factor, so every entry is bitwise the
    /// sequential one. A reduction is never split across tasks.
    #[must_use]
    pub fn abs_generator_sum_par(&self, par: RowPar) -> Vec<f64> {
        let (n, k) = (self.n, self.ids.len());
        if par == RowPar::Seq || k == 0 || n <= COL_BLOCK {
            return self.abs_generator_sum();
        }
        let factor = 1.0 + gamma(k + 2);
        let mut acc = vec![0.0f64; n];
        acc.par_chunks_mut(COL_BLOCK)
            .enumerate()
            .for_each(|(block, slots)| {
                let (c0, width) = (block * COL_BLOCK, slots.len());
                for row in self.gens.chunks_exact(n) {
                    for (slot, &value) in slots.iter_mut().zip(&row[c0..c0 + width]) {
                        *slot += value.abs();
                    }
                }
                for slot in slots {
                    *slot = next_up(*slot * factor);
                }
            });
        acc
    }

    /// Certified magnitude bound `m_j >= |c_j| + sum_k |G[k]_j| + r_j`, hence
    /// `m_j >= |x_j|` for every represented value.
    #[must_use]
    pub fn magnitude(&self, abs_g: &[f64]) -> Vec<f64> {
        self.center
            .iter()
            .zip(abs_g)
            .zip(&self.rem)
            .map(|((&c, &g), &r)| next_up(c.abs() + next_up(g + r)))
            .collect()
    }

    /// [`Self::magnitude`] under a row schedule: element-wise, so the column
    /// blocks of [`RowPar::Rayon`] compute the identical expression per entry.
    #[must_use]
    pub fn magnitude_par(&self, abs_g: &[f64], par: RowPar) -> Vec<f64> {
        let n = self.n;
        if par == RowPar::Seq || n <= COL_BLOCK || abs_g.len() != n {
            return self.magnitude(abs_g);
        }
        let mut m = vec![0.0f64; n];
        m.par_chunks_mut(COL_BLOCK)
            .enumerate()
            .for_each(|(block, slots)| {
                let c0 = block * COL_BLOCK;
                for (offset, slot) in slots.iter_mut().enumerate() {
                    let j = c0 + offset;
                    *slot = next_up(self.center[j].abs() + next_up(abs_g[j] + self.rem[j]));
                }
            });
        m
    }

    /// Refuse a state carrying a NaN or infinity anywhere. `abs_g` finite
    /// implies every generator entry is finite.
    pub fn ensure_finite(&self, abs_g: &[f64]) -> Result<(), ZonoRefusal> {
        if self.center.iter().any(|v| !v.is_finite()) {
            return Err(ZonoRefusal::NonFinite("zonotope centre"));
        }
        if self.rem.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(ZonoRefusal::NonFinite("zonotope remainder"));
        }
        if abs_g.iter().any(|v| !v.is_finite()) {
            return Err(ZonoRefusal::NonFinite("zonotope generators"));
        }
        Ok(())
    }

    /// Enforce a generator budget by folding the smallest-L1 unprotected rows
    /// into the remainder. Protected rows (the input symbols) are always kept,
    /// so the result can exceed `budget` only by the protected count.
    ///
    /// SOUND: `|G[k]_j e_k| <= |G[k]_j|` for `e_k in [-1, 1]`, so replacing a
    /// folded row by `r_j += sum_folded |G[k]_j|` (rounded up with its own
    /// `gamma` factor) over-approximates it. Kept rows retain their relative
    /// order; ties in the L1 ranking break by row index, so the fold is
    /// deterministic. Returns the number of folded rows.
    pub fn fold_to_budget<P>(&mut self, budget: usize, protected: P) -> usize
    where
        P: Fn(SymbolId) -> bool,
    {
        let k = self.ids.len();
        if k <= budget || self.n == 0 {
            return 0;
        }
        let n = self.n;
        let mut keep = vec![false; k];
        let mut candidates: Vec<(f64, usize)> = Vec::new();
        let mut protected_count = 0usize;
        for (row, &id) in self.ids.iter().enumerate() {
            if protected(id) {
                keep[row] = true;
                protected_count += 1;
            } else {
                let l1: f64 = self.gens[row * n..(row + 1) * n]
                    .iter()
                    .map(|v| v.abs())
                    .sum();
                candidates.push((l1, row));
            }
        }
        let keep_unprotected = budget.saturating_sub(protected_count);
        if candidates.len() <= keep_unprotected {
            return 0;
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(_, row) in candidates.iter().take(keep_unprotected) {
            keep[row] = true;
        }
        let folded = candidates.len() - keep_unprotected;
        let mut acc = vec![0.0f64; n];
        for (row, _) in keep.iter().enumerate().filter(|(_, kept)| !**kept) {
            for (slot, &value) in acc.iter_mut().zip(&self.gens[row * n..(row + 1) * n]) {
                *slot += value.abs();
            }
        }
        let factor = 1.0 + gamma(folded + 2);
        for (r, a) in self.rem.iter_mut().zip(&acc) {
            *r = next_up(*r + next_up(*a * factor));
        }
        let mut gens = Vec::with_capacity((k - folded) * n);
        let mut ids = Vec::with_capacity(k - folded);
        for (row, &kept) in keep.iter().enumerate() {
            if kept {
                gens.extend_from_slice(&self.gens[row * n..(row + 1) * n]);
                ids.push(self.ids[row]);
            }
        }
        self.gens = gens;
        self.ids = ids;
        folded
    }

    /// Consume the state into its parts `(n, centre, generators, ids, remainder)`.
    #[must_use]
    pub fn into_parts(self) -> (usize, Vec<f64>, Vec<f64>, Vec<SymbolId>, Vec<f64>) {
        (self.n, self.center, self.gens, self.ids, self.rem)
    }

    /// Live bytes held by the generator matrix (for admission accounting).
    #[must_use]
    pub fn generator_bytes(&self) -> u64 {
        u64::try_from(self.gens.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(8)
    }
}
