// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The declared input box, and the float32 snapping that makes a candidate
//! mean the same thing to the search and to the oracle.
//!
//! Ported from `scripts/audit_unsat_by_falsification.py` (`Searcher.__init__`,
//! `snap`, `materialise`, `_f32_above`, `_f32_below`). The comment there states
//! the contract and it is preserved exactly:
//!
//! > Every point is generated so that ORT sees exactly the float64 values the
//! > assertions were checked against: pinned coordinates keep their exact
//! > (possibly non-float32-representable) constant -- which is what makes
//! > cctsdb-style equality-pinned specs falsifiable at all -- and free
//! > coordinates are snapped onto the float32 grid inside the box.
//!
//! Getting this wrong is not a quality issue. A candidate that leaves the box
//! by one ULP after float32 rounding is rejected by the trusted oracle, and a
//! candidate that the search believed was at the bound but that ORT sees one
//! ULP inside it is a different point than the one that was scored.

/// Why a box could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoxError {
    /// Bound vectors disagree in length.
    LengthMismatch {
        /// Lower-bound length.
        lo: usize,
        /// Upper-bound length.
        hi: usize,
    },
    /// Some coordinate has `lo > hi`.
    EmptyInterval {
        /// The offending coordinate.
        index: usize,
    },
    /// The box has no coordinates at all.
    Empty,
}

/// Smallest `f32` that is `>= value` (infinite when none exists).
fn f32_above(value: f64) -> f32 {
    if !value.is_finite() {
        return if value > 0.0 {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
    }
    let candidate = value as f32;
    if f64::from(candidate) < value {
        candidate.next_up()
    } else {
        candidate
    }
}

/// Largest `f32` that is `<= value` (infinite when none exists).
fn f32_below(value: f64) -> f32 {
    if !value.is_finite() {
        return if value > 0.0 {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
    }
    let candidate = value as f32;
    if f64::from(candidate) > value {
        candidate.next_down()
    } else {
        candidate
    }
}

/// The declared input box, split into free and pinned coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchBox {
    lo: Vec<f64>,
    hi: Vec<f64>,
    free: Vec<usize>,
    free_lo: Vec<f64>,
    free_hi: Vec<f64>,
    use_f32: Vec<bool>,
    f32_lo: Vec<f32>,
    f32_hi: Vec<f32>,
    base: Vec<f64>,
    centre_free: Vec<f64>,
}

impl SearchBox {
    /// Build a box from declared per-coordinate bounds.
    pub fn new(lo: &[f64], hi: &[f64]) -> Result<Self, BoxError> {
        if lo.len() != hi.len() {
            return Err(BoxError::LengthMismatch {
                lo: lo.len(),
                hi: hi.len(),
            });
        }
        if lo.is_empty() {
            return Err(BoxError::Empty);
        }
        for (index, (&l, &h)) in lo.iter().zip(hi).enumerate() {
            if l > h {
                return Err(BoxError::EmptyInterval { index });
            }
        }

        let free: Vec<usize> = (0..lo.len()).filter(|&i| hi[i] > lo[i]).collect();
        let free_lo: Vec<f64> = free.iter().map(|&i| lo[i]).collect();
        let free_hi: Vec<f64> = free.iter().map(|&i| hi[i]).collect();

        let mut use_f32 = vec![false; free.len()];
        let mut f32_lo = vec![0.0f32; free.len()];
        let mut f32_hi = vec![0.0f32; free.len()];
        for position in 0..free.len() {
            let low = f32_above(free_lo[position]);
            let high = f32_below(free_hi[position]);
            if low.is_finite() && high.is_finite() && low <= high {
                use_f32[position] = true;
                f32_lo[position] = low;
                f32_hi[position] = high;
            }
        }

        // base: lo where finite else 0, replaced by the midpoint where both
        // bounds are finite; pinned coordinates then forced back to their exact
        // constant. Exactly the Python order of operations.
        let mut base: Vec<f64> = lo
            .iter()
            .map(|&l| if l.is_finite() { l } else { 0.0 })
            .collect();
        // NOT `f64::midpoint`: the comment above promises exactly the Python
        // reference's order of operations, and `0.5 * (lo + hi)` overflows to
        // `inf` where `midpoint` would return a finite interior value —
        // keeping the reference's arithmetic is the contract here.
        #[allow(unknown_lints)] // stock 1.95 clippy (public pin) may not know the lint below
        #[allow(clippy::manual_midpoint)]
        for i in 0..lo.len() {
            if lo[i].is_finite() && hi[i].is_finite() {
                base[i] = 0.5 * (lo[i] + hi[i]);
            }
        }

        let mut this = Self {
            lo: lo.to_vec(),
            hi: hi.to_vec(),
            free,
            free_lo,
            free_hi,
            use_f32,
            f32_lo,
            f32_hi,
            base,
            centre_free: Vec::new(),
        };

        let raw_centre: Vec<f64> = this.free.iter().map(|&i| this.base[i]).collect();
        this.centre_free = this.snap(&raw_centre);
        for (position, &index) in this.free.iter().enumerate() {
            this.base[index] = this.centre_free[position];
        }
        for index in 0..this.lo.len() {
            if this.hi[index] == this.lo[index] {
                this.base[index] = this.lo[index];
            }
        }
        Ok(this)
    }

    /// Number of declared input coordinates.
    pub fn dims(&self) -> usize {
        self.lo.len()
    }

    /// Free coordinates (`hi > lo`).
    pub fn free_dims(&self) -> usize {
        self.free.len()
    }

    /// Pinned coordinates (`hi == lo`).
    pub fn pinned_dims(&self) -> usize {
        self.lo.len() - self.free.len()
    }

    /// Lower bounds of the free coordinates.
    pub fn free_lo(&self) -> &[f64] {
        &self.free_lo
    }

    /// Upper bounds of the free coordinates.
    pub fn free_hi(&self) -> &[f64] {
        &self.free_hi
    }

    /// Snapped centre of the free coordinates.
    pub fn centre_free(&self) -> &[f64] {
        &self.centre_free
    }

    /// Snap a free-coordinate vector onto the float32 grid inside the box,
    /// clipping to the declared bounds where no float32 grid point exists.
    pub fn snap(&self, free_values: &[f64]) -> Vec<f64> {
        assert_eq!(
            free_values.len(),
            self.free.len(),
            "snap expects a free-coordinate vector"
        );
        let mut out = free_values.to_vec();
        for position in 0..out.len() {
            if self.use_f32[position] {
                let narrowed =
                    (out[position] as f32).clamp(self.f32_lo[position], self.f32_hi[position]);
                out[position] = f64::from(narrowed);
            } else {
                out[position] = out[position].clamp(self.free_lo[position], self.free_hi[position]);
            }
        }
        out
    }

    /// Expand a free-coordinate vector into a full declared input vector,
    /// snapping the free coordinates and leaving pinned ones exact.
    pub fn materialise(&self, free_values: &[f64]) -> Vec<f64> {
        let snapped = self.snap(free_values);
        let mut point = self.base.clone();
        for (position, &index) in self.free.iter().enumerate() {
            point[index] = snapped[position];
        }
        point
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_degenerate_box_is_refused_rather_than_silently_repaired() {
        assert_eq!(
            SearchBox::new(&[0.0, 1.0], &[1.0]),
            Err(BoxError::LengthMismatch { lo: 2, hi: 1 })
        );
        assert_eq!(SearchBox::new(&[], &[]), Err(BoxError::Empty));
        assert_eq!(
            SearchBox::new(&[0.0, 5.0], &[1.0, 2.0]),
            Err(BoxError::EmptyInterval { index: 1 })
        );
    }

    #[test]
    fn a_fully_pinned_box_still_has_exactly_one_point_in_it() {
        // cctsdb-style equality-pinned specs are falsifiable precisely because
        // that point gets checked, and because the pinned constant reaches the
        // oracle EXACTLY rather than snapped.
        let constant = 0.123_456_789_012_345_68_f64;
        let domain = SearchBox::new(&[constant, 2.0], &[constant, 2.0]).unwrap();
        assert_eq!(domain.free_dims(), 0);
        assert_eq!(domain.pinned_dims(), 2);
        let point = domain.materialise(&[]);
        assert_eq!(point[0].to_bits(), constant.to_bits());
        assert_ne!(point[0].to_bits(), f64::from(constant as f32).to_bits());
    }

    #[test]
    fn snapping_stays_inside_the_declared_box_in_both_directions() {
        // The bounds are chosen so that naive `as f32` rounding leaves the box:
        // the lower bound rounds DOWN and the upper bound rounds UP in f32.
        let lo = -0.30353115610000002_f64;
        let hi = 0.679_857_768_699_999_9_f64;
        let domain = SearchBox::new(&[lo], &[hi]).unwrap();
        for request in [lo, hi, lo - 1.0, hi + 1.0, 0.0] {
            let point = domain.materialise(&[request]);
            assert!(
                point[0] >= lo && point[0] <= hi,
                "{request:?} snapped to {:?}, outside [{lo:?}, {hi:?}]",
                point[0]
            );
            assert_eq!(f64::from(point[0] as f32).to_bits(), point[0].to_bits());
        }
    }
}
