// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Ordered via-y composition across independent input coordinates.
//!
//! Each lane retains the original ascending-j mul/mul/add/add sequence.
//! The transpose only puts independent coordinates next to one another, so
//! vectorization needs no reassociation or change to the gamma/error proof.

use super::head_underflow;

const LANES: usize = 8;
const MAX_BYTES: usize = 128 << 20;
const MIN_WORK: usize = 1 << 18;

pub(super) struct TiledHeadRows {
    lower: Vec<f64>,
    upper: Vec<f64>,
    lower_min: Vec<f64>,
    upper_min: Vec<f64>,
    columns: usize,
}

fn zeros(n: usize) -> Option<Vec<f64>> {
    let mut out = Vec::new();
    out.try_reserve_exact(n).ok()?;
    if out.capacity() != n {
        return None;
    }
    out.resize(n, 0.0);
    Some(out)
}

#[cfg(test)]
#[path = "bounds_tiled_tests.rs"]
mod tests;

impl TiledHeadRows {
    pub(super) fn for_composition(
        lower: &[f64],
        upper: &[f64],
        inputs: usize,
        columns: usize,
        margins: usize,
    ) -> Option<Self> {
        // A single partial tile can multiply arithmetic by up to eight while
        // barely reducing product checks. Keep those shapes on the scalar path.
        // prepare remains available to exercise all padding tails in controls.
        if inputs < LANES {
            return None;
        }
        Self::prepare(lower, upper, inputs, columns, margins)
    }

    pub(super) fn prepare(
        lower: &[f64],
        upper: &[f64],
        inputs: usize,
        columns: usize,
        margins: usize,
    ) -> Option<Self> {
        let elements = inputs.checked_mul(columns)?;
        if columns == 0
            || margins < 4
            || elements.checked_mul(margins)? < MIN_WORK
            || lower.len() != elements
            || upper.len() != elements
        {
            return None;
        }
        let padded_inputs = inputs.checked_add(LANES - 1)? / LANES * LANES;
        let padded = padded_inputs.checked_mul(columns)?;
        if padded.checked_add(columns)?.checked_mul(16)? > MAX_BYTES {
            return None;
        }
        let mut out = Self {
            lower: zeros(padded)?,
            upper: zeros(padded)?,
            lower_min: zeros(columns)?,
            upper_min: zeros(columns)?,
            columns,
        };
        for k in 0..inputs {
            for j in 0..columns {
                let (a, b) = (lower[k * columns + j], upper[k * columns + j]);
                if !a.is_finite() || !b.is_finite() {
                    return None;
                }
                let slot = (k / LANES * columns + j) * LANES + k % LANES;
                out.lower[slot] = a;
                out.upper[slot] = b;
                for (minimum, value) in [(&mut out.lower_min[j], a), (&mut out.upper_min[j], b)] {
                    let value = value.abs();
                    if value != 0.0 && (*minimum == 0.0 || value < *minimum) {
                        *minimum = value;
                    }
                }
            }
        }
        Some(out)
    }

    /// EXACT OR of the original per-product dust predicates, for finite
    /// coefficients in the qualified RN/gradual-underflow environment.
    /// Each positive minimum is attained in its original column. Monotonic
    /// rounding of magnitudes makes a dust-producing product exist iff the
    /// minimum produces dust. Zero means the entire column is zero; those
    /// products never satisfy the original nonzero-operand predicate.
    /// Bias/coefficient-generation/concretization checks stay with the caller.
    pub(super) fn product_underflow(&self, ca: &[f64], cb: &[f64]) -> Option<bool> {
        if ca.len() != self.columns || cb.len() != self.columns {
            return None;
        }
        let mut underflow = false;
        for j in 0..self.columns {
            if !ca[j].is_finite() || !cb[j].is_finite() {
                return None;
            }
            underflow |= head_underflow(ca[j], self.lower_min[j])
                || head_underflow(cb[j], self.upper_min[j]);
        }
        Some(underflow)
    }

    pub(super) fn compose(&self, ca: &[f64], cb: &[f64], out: &mut [f64]) {
        for (tile, destination) in out.chunks_mut(LANES).enumerate() {
            let base = tile * self.columns * LANES;
            let mut acc = [0.0; LANES];
            for j in 0..self.columns {
                let offset = base + j * LANES;
                for (lane, value) in acc.iter_mut().enumerate() {
                    *value += ca[j] * self.lower[offset + lane] + cb[j] * self.upper[offset + lane];
                }
            }
            destination.copy_from_slice(&acc[..destination.len()]);
        }
    }
}
