// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit noise-symbol identities and ID-aligned binary operations.
//!
//! Every generator row of a [`ZonoState`] names the noise symbol it scales.
//! Two forms that share a symbol MUST combine that symbol's rows; rows of
//! symbols only one side knows are carried over unchanged. Aligning rows by
//! POSITION instead (zero-padding to a common width, or a positional common
//! prefix) silently adds generator `k` of one form to an unrelated generator
//! `k` of the other. That is Vibecheck's documented concat-aliasing defect
//! (plan U8) and the same class as ny's retracted N-ary Concat soundness bug.
//! The cGAN lane is sequential and never combines two forms, but the shared
//! core's later users (V4 ml4acopf, vit) do, so the invariant lives here.

use std::collections::HashMap;

use super::rounding::{gamma, next_up, TINY, UNIT};
use super::state::ZonoState;
use super::ZonoRefusal;

/// Identity of one noise symbol `e in [-1, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolId(pub u64);

/// Monotone allocator of fresh [`SymbolId`]s for one forward pass.
#[derive(Debug, Default, Clone)]
pub struct SymbolAllocator {
    next: u64,
}

impl SymbolAllocator {
    /// A fresh allocator whose first symbol is `SymbolId(0)`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate one fresh symbol.
    pub fn fresh(&mut self) -> SymbolId {
        let id = SymbolId(self.next);
        self.next += 1;
        id
    }

    /// Number of symbols allocated so far.
    #[must_use]
    pub fn allocated(&self) -> u64 {
        self.next
    }
}

/// Element-wise `a + b` (or `a - b` when `subtract`) of two forms over the
/// same values, aligned BY SYMBOL ID.
///
/// Output rows: every symbol of `a` in `a`'s order, then every symbol only `b`
/// knows in `b`'s order. A shared symbol's row is `a_row +/- b_row`.
///
/// SOUND: in exact arithmetic `(c_a + G_a e + rho_a) +/- (c_b + G_b e + rho_b)
/// = (c_a +/- c_b) + (G_a +/- G_b) e + (rho_a +/- rho_b)` for the SAME
/// assignment `e` of every symbol (this is exactly what ID alignment
/// guarantees). The centre and each shared row are rounded once, at most
/// `u (|x| + |y|)` each; those errors and both remainders go into the new
/// remainder, which is rounded up. Copied rows are exact (negation is exact).
pub fn add_aligned(a: &ZonoState, b: &ZonoState, subtract: bool) -> Result<ZonoState, ZonoRefusal> {
    let n = a.len();
    if b.len() != n {
        return Err(ZonoRefusal::Shape(format!(
            "aligned add of {n} values with {} values",
            b.len()
        )));
    }
    let sign = if subtract { -1.0 } else { 1.0 };
    if let Some(mutated) = positional_mutation(a, b, sign) {
        return mutated;
    }
    let b_row_of: HashMap<_, _> = b
        .ids()
        .iter()
        .enumerate()
        .map(|(row, &id)| (id, row))
        .collect();
    let mut ids = Vec::with_capacity(a.num_generators() + b.num_generators());
    let mut gens = Vec::with_capacity((a.num_generators() + b.num_generators()) * n);
    let mut shared_abs = vec![0.0f64; n];
    let mut b_used = vec![false; b.num_generators()];
    for (row, &id) in a.ids().iter().enumerate() {
        let a_row = a.generator_row(row);
        ids.push(id);
        match b_row_of.get(&id) {
            Some(&b_row) => {
                b_used[b_row] = true;
                let b_vals = b.generator_row(b_row);
                for j in 0..n {
                    gens.push(a_row[j] + sign * b_vals[j]);
                    shared_abs[j] += a_row[j].abs() + b_vals[j].abs();
                }
            }
            None => gens.extend_from_slice(a_row),
        }
    }
    for (row, &id) in b.ids().iter().enumerate() {
        if !b_used[row] {
            ids.push(id);
            gens.extend(b.generator_row(row).iter().map(|v| sign * v));
        }
    }
    let center: Vec<f64> = a
        .center()
        .iter()
        .zip(b.center())
        .map(|(x, y)| x + sign * y)
        .collect();
    // `shared_abs` is itself a binary64 sum of up to 2K terms; the gamma factor
    // lifts it back above the exact sum before it is charged.
    let sum_factor = 1.0 + gamma(2 * ids.len() + 4);
    let rem: Vec<f64> = (0..n)
        .map(|j| {
            let rounding =
                UNIT * (a.center()[j].abs() + b.center()[j].abs() + shared_abs[j] * sum_factor);
            next_up((a.remainder()[j] + b.remainder()[j] + rounding) * (1.0 + 8.0 * UNIT) + TINY)
        })
        .collect();
    ZonoState::from_parts(n, center, gens, ids, rem)
}

/// The positional-alignment mutation when the test switch is armed.
#[cfg(test)]
fn positional_mutation(
    a: &ZonoState,
    b: &ZonoState,
    sign: f64,
) -> Option<Result<ZonoState, ZonoRefusal>> {
    fault::positional_alignment().then(|| add_positional_for_mutation_test(a, b, sign))
}

/// Never a mutation in production.
#[cfg(not(test))]
#[inline]
fn positional_mutation(
    _a: &ZonoState,
    _b: &ZonoState,
    _sign: f64,
) -> Option<Result<ZonoState, ZonoRefusal>> {
    None
}

/// The deliberately WRONG positional alignment, reachable only through the
/// test fault switch. Rows are paired by index and the result takes `a`'s IDs
/// for paired rows, exactly the U8 defect shape.
#[cfg(test)]
fn add_positional_for_mutation_test(
    a: &ZonoState,
    b: &ZonoState,
    sign: f64,
) -> Result<ZonoState, ZonoRefusal> {
    let n = a.len();
    let k = a.num_generators().max(b.num_generators());
    let mut ids = Vec::with_capacity(k);
    let mut gens = Vec::with_capacity(k * n);
    let mut next_extra = a
        .ids()
        .iter()
        .chain(b.ids())
        .map(|id| id.0)
        .max()
        .unwrap_or(0)
        + 1;
    for row in 0..k {
        let a_row = (row < a.num_generators()).then(|| a.generator_row(row));
        let b_row = (row < b.num_generators()).then(|| b.generator_row(row));
        let id = if row < a.num_generators() {
            a.ids()[row]
        } else if !a.ids().contains(&b.ids()[row]) {
            b.ids()[row]
        } else {
            next_extra += 1;
            SymbolId(next_extra)
        };
        ids.push(id);
        for j in 0..n {
            let x = a_row.map_or(0.0, |r| r[j]);
            let y = b_row.map_or(0.0, |r| r[j]);
            gens.push(x + sign * y);
        }
    }
    let center = a
        .center()
        .iter()
        .zip(b.center())
        .map(|(x, y)| x + sign * y)
        .collect();
    let rem = a
        .remainder()
        .iter()
        .zip(b.remainder())
        .map(|(x, y)| next_up((x + y) * (1.0 + 8.0 * UNIT) + TINY))
        .collect();
    ZonoState::from_parts(n, center, gens, ids, rem)
}

/// Test-only fault injection for the ID-alignment mutation pin
/// (`child_zono_symbol_ids_align_by_id`): when armed, [`add_aligned`] pairs
/// rows by POSITION, which the pin must catch.
#[cfg(test)]
pub(crate) mod fault {
    use std::cell::Cell;

    thread_local! {
        static POSITIONAL: Cell<bool> = const { Cell::new(false) };
    }

    /// Arm or disarm the positional-alignment mutation on this thread.
    pub(crate) fn set_positional_alignment(on: bool) {
        POSITIONAL.with(|cell| cell.set(on));
    }

    pub(crate) fn positional_alignment() -> bool {
        POSITIONAL.with(Cell::get)
    }
}
