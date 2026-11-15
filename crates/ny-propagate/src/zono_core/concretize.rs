// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Outward concretisation of a zonotope to an interval box, the
//! keep-the-looser-side intersection with inherited enclosures, and the
//! outward store to `f32`.

use ny_core::{f64_to_f32_down, f64_to_f32_up};

use super::rounding::{next_down, next_up};
use super::state::ZonoState;

/// Certified box of every represented value: `[c - s, c + s]` with
/// `s >= sum|G| + r`, both endpoints rounded outward. `abs_g` must be
/// `state.abs_generator_sum()`.
#[must_use]
pub fn concretize(state: &ZonoState, abs_g: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut lo = Vec::with_capacity(state.len());
    let mut hi = Vec::with_capacity(state.len());
    for ((&c, &g), &r) in state.center().iter().zip(abs_g).zip(state.remainder()) {
        let spread = next_up(g + r);
        lo.push(next_down(c - spread));
        hi.push(next_up(c + spread));
    }
    (lo, hi)
}

/// Intersect `[lo, hi]` element-wise with another certified enclosure of the
/// same values (an inherited `f32` map entry is exact in binary64).
///
/// FAIL CLOSED per element: a NaN endpoint is ignored, and a crossed result
/// (the two enclosures disjoint, which exact arithmetic forbids but rounding
/// or an upstream defect could produce) keeps the ORIGINAL pair — never the
/// collapse-to-a-point of Vibecheck's U9. Returns the number of elements whose
/// interval strictly shrank.
pub fn intersect_keep_looser(
    lo: &mut [f64],
    hi: &mut [f64],
    other_lo: &[f64],
    other_hi: &[f64],
) -> usize {
    debug_assert_eq!(lo.len(), other_lo.len());
    debug_assert_eq!(hi.len(), other_hi.len());
    let mut tightened = 0usize;
    for j in 0..lo.len().min(other_lo.len()).min(other_hi.len()) {
        let (ol, oh) = (other_lo[j], other_hi[j]);
        if ol.is_nan() || oh.is_nan() {
            continue;
        }
        let nl = lo[j].max(ol);
        let nh = hi[j].min(oh);
        if nl > nh {
            continue;
        }
        if nl > lo[j] || nh < hi[j] {
            tightened += 1;
        }
        lo[j] = nl;
        hi[j] = nh;
    }
    tightened
}

/// Store a binary64 box to `f32` outward (lower rounded down, upper up; the
/// store never publishes an `f32` subnormal, see `ny_core`).
#[must_use]
pub fn to_f32_outward(lo: &[f64], hi: &[f64]) -> (Vec<f32>, Vec<f32>) {
    (
        lo.iter().map(|&v| f64_to_f32_down(v)).collect(),
        hi.iter().map(|&v| f64_to_f32_up(v)).collect(),
    )
}

/// Count elements with `lo < 0 < hi`.
#[must_use]
pub fn count_unstable(lo: &[f64], hi: &[f64]) -> usize {
    lo.iter()
        .zip(hi)
        .filter(|(&l, &h)| l < 0.0 && h > 0.0)
        .count()
}
