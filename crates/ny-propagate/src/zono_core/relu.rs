// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DeepZ ReLU with directed rounding and one fresh explicit-ID symbol per
//! unstable neuron.
//!
//! For a pre-activation enclosure `x in [l, u]` with `l < 0 < u` and ANY slope
//! `lambda in [0, 1]`:
//!
//! ```text
//! relu(x) - lambda x in [0, M],   M = max(-lambda l, (1 - lambda) u)
//! ```
//!
//! (on `[l, 0]` the difference is `-lambda x in [0, -lambda l]`; on `[0, u]` it
//! is `(1 - lambda) x in [0, (1 - lambda) u]`). Hence `relu(x) = lambda x + mu
//! + mu e_new` with `mu >= M / 2` and a FRESH symbol `e_new in [-1, 1]`. The
//! identity needs only that `[l, u]` encloses `x` — which is why the caller may
//! first intersect the zonotope's own concretisation with any other certified
//! enclosure (the root CROWN map, the parent's refined map): a tighter valid
//! `[l, u]` gives a tighter `M` and more stable neurons, never an unsound one.
//! Stable neurons are exact: active (`l >= 0`) passes through unchanged, dead
//! (`u <= 0`) is identically zero.

use rayon::prelude::*;

use super::ids::{SymbolAllocator, SymbolId};
use super::kernels::RowPar;
use super::rounding::{gamma, next_up, TINY, UNIT};
use super::state::ZonoState;
use super::ZonoRefusal;

/// What one ReLU did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReluReport {
    /// Unstable neurons (`l < 0 < u` after any intersection).
    pub unstable: usize,
    /// Neurons fixed active.
    pub active: usize,
    /// Neurons fixed dead.
    pub dead: usize,
}

/// `(flip mu, drop remainder)` for the mutation pin; both `false` outside
/// `cfg(test)`.
#[cfg(test)]
fn fault_flags() -> (bool, bool) {
    let armed = fault::get();
    (
        armed == fault::ReluFault::FlipMuSign,
        armed == fault::ReluFault::DropRemainder,
    )
}

/// `(flip mu, drop remainder)`: always `(false, false)` in production.
#[cfg(not(test))]
#[inline]
const fn fault_flags() -> (bool, bool) {
    (false, false)
}

/// Push a state through DeepZ ReLU using the certified pre-activation
/// enclosure `[lo, hi]` (which the caller has already intersected with any
/// inherited map). `abs_g` must be `state.abs_generator_sum()`.
///
/// ROUNDING. `lambda = hi / (hi - lo)` is clipped into `[0, 1]`; whatever
/// binary64 value results, `M` is computed from THAT value with every product
/// rounded up, and `mu = M / 2` is bumped one ulp if the halving was inexact
/// (subnormal `M`). The centre `lambda c + mu` and the scaled rows `lambda G`
/// are rounded once each; `gamma_2 (|lambda c| + mu) + u lambda sum|G|` plus
/// `lambda r` becomes the new remainder, times `(1 + 4u)` and rounded up.
/// Active neurons keep `c`, their generator column and `r` bit-for-bit
/// (multiplication by 1 is exact); dead neurons become exactly zero.
pub fn apply_relu(
    state: &ZonoState,
    abs_g: &[f64],
    lo: &[f64],
    hi: &[f64],
    alloc: &mut SymbolAllocator,
) -> Result<(ZonoState, ReluReport), ZonoRefusal> {
    apply_relu_par(state, abs_g, lo, hi, alloc, RowPar::Seq)
}

/// [`apply_relu`] under a row schedule (#m8-cgan-leaf-batch). Only the
/// scaling of the `K` inherited generator rows is row-parallel under
/// [`RowPar::Rayon`]: every row is independent and every entry is the same
/// single product (or pass-through) as the sequential loop, so the result is
/// bitwise identical. Slopes, centre, remainder and fresh rows are unchanged.
pub fn apply_relu_par(
    state: &ZonoState,
    abs_g: &[f64],
    lo: &[f64],
    hi: &[f64],
    alloc: &mut SymbolAllocator,
    par: RowPar,
) -> Result<(ZonoState, ReluReport), ZonoRefusal> {
    let n = state.len();
    if lo.len() != n || hi.len() != n || abs_g.len() != n {
        return Err(ZonoRefusal::Shape(format!(
            "relu bounds of {} / {} values for {n}",
            lo.len(),
            hi.len()
        )));
    }
    let mut report = ReluReport::default();
    let mut lambda = vec![0.0f64; n];
    let mut mu = vec![0.0f64; n];
    let mut unstable_idx = Vec::new();
    for j in 0..n {
        let (l, u) = (lo[j], hi[j]);
        if !l.is_finite() || !u.is_finite() || l > u {
            return Err(ZonoRefusal::NonFinite("relu pre-activation bounds"));
        }
        if l >= 0.0 {
            lambda[j] = 1.0;
            report.active += 1;
        } else if u <= 0.0 {
            report.dead += 1;
        } else {
            let lam = (u / (u - l)).clamp(0.0, 1.0);
            let big_m = next_up(next_up(-lam * l).max(next_up(next_up(1.0 - lam) * u)));
            let mut half = big_m * 0.5;
            if half * 2.0 < big_m {
                half = next_up(half);
            }
            lambda[j] = lam;
            mu[j] = half;
            unstable_idx.push(j);
            report.unstable += 1;
        }
    }

    // Production: both flags are constant `false` and `1.0 * mu` is exact, so
    // the arithmetic below is unchanged; only the test fault switch flips them.
    let (flip_mu, drop_remainder) = fault_flags();
    let mu_sign = if flip_mu { -1.0f64 } else { 1.0f64 };

    let mut center = Vec::with_capacity(n);
    for j in 0..n {
        let c = state.center()[j];
        let value = if lo[j] >= 0.0 {
            c
        } else if hi[j] <= 0.0 {
            0.0
        } else {
            lambda[j] * c + mu_sign * mu[j]
        };
        center.push(value);
    }

    let mut rem = Vec::with_capacity(n);
    for j in 0..n {
        let r_old = state.remainder()[j];
        let value = if lo[j] >= 0.0 {
            r_old
        } else if hi[j] <= 0.0 {
            0.0
        } else {
            let lam = lambda[j];
            let charge = lam * r_old
                + gamma(2) * ((lam * state.center()[j]).abs() + mu[j])
                + UNIT * lam * abs_g[j];
            next_up(charge * (1.0 + 4.0 * UNIT) + TINY)
        };
        rem.push(value);
    }
    if drop_remainder {
        rem.iter_mut().for_each(|r| *r = 0.0);
    }

    let k = state.num_generators();
    let new_rows = unstable_idx.len();
    let mut gens = if par == RowPar::Rayon && n > 0 && k > 1 {
        // Independent rows: row r of the output is row r of the input scaled
        // entry-wise by the SAME expression as the sequential arm below.
        let mut gens = vec![0.0f64; (k + new_rows) * n];
        gens[..k * n]
            .par_chunks_mut(n)
            .zip(state.generators().par_chunks_exact(n))
            .for_each(|(dst, row)| {
                for ((slot, g), lam) in dst.iter_mut().zip(row).zip(&lambda) {
                    *slot = if *lam == 1.0 { *g } else { g * lam };
                }
            });
        gens.truncate(k * n);
        gens
    } else {
        let mut gens = Vec::with_capacity((k + new_rows) * n);
        for row in state.generators().chunks_exact(n.max(1)) {
            gens.extend(row.iter().zip(&lambda).map(|(g, lam)| {
                // Active: exact pass-through; dead: exact zero; unstable: scaled.
                if *lam == 1.0 {
                    *g
                } else {
                    g * lam
                }
            }));
        }
        gens
    };
    let mut ids: Vec<SymbolId> = state.ids().to_vec();
    for &j in &unstable_idx {
        let start = gens.len();
        gens.resize(start + n, 0.0);
        gens[start + j] = mu[j];
        ids.push(alloc.fresh());
    }
    Ok((
        ZonoState::from_parts_trusted(n, center, gens, ids, rem),
        report,
    ))
}

/// Test-only fault injection for the form-mutation pin
/// (`child_zono_form_mutation_fails`).
#[cfg(test)]
pub(crate) mod fault {
    use std::cell::Cell;

    /// The mutations the pin must catch.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum ReluFault {
        /// No mutation (production behaviour).
        None,
        /// Flip the sign of the DeepZ offset `mu` in the centre.
        FlipMuSign,
        /// Drop the remainder entirely after the ReLU.
        DropRemainder,
    }

    thread_local! {
        static FAULT: Cell<ReluFault> = const { Cell::new(ReluFault::None) };
    }

    /// Arm (or with `ReluFault::None`, disarm) a ReLU mutation on this thread.
    pub(crate) fn set(fault: ReluFault) {
        FAULT.with(|cell| cell.set(fault));
    }

    pub(crate) fn get() -> ReluFault {
        FAULT.with(Cell::get)
    }
}
