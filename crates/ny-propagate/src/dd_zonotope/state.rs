// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The certified double-double zonotope state (`#dd-zonotope`).

use ny_core::dd::{next_down_f64, next_up_f64, Dd, U_F64};

/// Relative inflation applied to every error-channel value at the end of each
/// op, covering the plain-f64 rounding committed *while computing the error
/// channel itself*.
///
/// The error channel is a sum of nonnegative terms whose longest reduction is
/// one conv/gemm dot of length `K <= 4608`, i.e. relative error
/// `gamma_4608(2^-53) = 5.1e-13`. Even chaining every reduction in one layer
/// stays under `gamma_20000 = 2.2e-12`. `1e-9` covers that with ~450x margin
/// while contributing `(1 + 1e-9)^40 - 1 = 4e-8` relative over the whole
/// VGG16 pass — nothing, against a channel that must stay 10 orders below the
/// margin.
pub(crate) const ERR_INFLATE: f64 = 1.0 + 1e-9;

/// Round a computed nonnegative error-channel value OUTWARD (up).
#[inline]
pub(crate) fn err_up(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::INFINITY;
    }
    // Exact zero stays exact: every error-channel expression is a sum of
    // nonnegative terms, so a computed 0.0 means every term was 0.0 and 0.0 is
    // already an upper bound. Rounding it up to the smallest subnormal would
    // make `mu > 0` fire on structurally-exact ops (dominated max-pool windows,
    // stable ReLUs) and manufacture generator columns out of nothing.
    if x <= 0.0 {
        return 0.0;
    }
    next_up_f64(x * ERR_INFLATE)
}

/// A zonotope over `n` elements with a double-double center and a certified
/// interval remainder.
///
/// The set represented is
///
/// ```text
/// Z = { center + sum_j gens[j] * e_j + d
///       : e in [-1,1]^m,  |d_i| <= ec_i + eg_i }
/// ```
///
/// with the invariant that the network's true reachable set at this node is a
/// subset of `Z`. `ec` carries the accumulated CENTER error (double-double
/// working precision), `eg` the accumulated GENERATOR error (plain-f64 working
/// precision). They are split only for diagnostics — every concretization adds
/// both.
///
/// Generator layout is COLUMN-major: `gens[j]` is a full `n`-length vector.
/// That is the layout the affine transformers want (one independent
/// convolution/gemm per column) and it is what the reference probe measured.
#[derive(Debug, Clone)]
pub(crate) struct DdZono {
    /// Element shape (unbatched, C-order), e.g. `[64, 224, 224]`.
    pub(crate) shape: Vec<usize>,
    /// Double-double center, length `numel`.
    pub(crate) center: Vec<Dd>,
    /// Generator columns, each of length `numel`.
    pub(crate) gens: Vec<Vec<f64>>,
    /// Nonnegative certified center-error channel, length `numel`.
    pub(crate) ec: Vec<f64>,
    /// Nonnegative certified generator-error channel, length `numel`.
    pub(crate) eg: Vec<f64>,
}

impl DdZono {
    /// Number of elements.
    #[inline]
    pub(crate) fn numel(&self) -> usize {
        self.center.len()
    }

    /// Number of generator columns.
    #[inline]
    pub(crate) fn n_gens(&self) -> usize {
        self.gens.len()
    }

    /// Validate the shape/value layout before any transformer performs indexed
    /// access. A malformed internal state must cause the default-on lane to
    /// refuse, never panic or let a short generator silently truncate a zip.
    pub(crate) fn has_valid_layout(&self) -> bool {
        let Some(shape_numel) = self
            .shape
            .iter()
            .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        else {
            return false;
        };
        let n = self.center.len();
        shape_numel == n
            && self.ec.len() == n
            && self.eg.len() == n
            && self.gens.iter().all(|generator| generator.len() == n)
    }

    /// Approximate live bytes held by this state.
    pub(crate) fn bytes(&self) -> usize {
        let n = self.numel();
        // center is 2 x f64; ec/eg are f64; each generator column is f64.
        n.saturating_mul(2 * 8 + 2 * 8)
            .saturating_add(n.saturating_mul(8).saturating_mul(self.gens.len()))
    }

    /// Per-element generator radius `sum_j |gens[j][i]|`, rounded OUTWARD.
    pub(crate) fn radius(&self) -> Vec<f64> {
        let n = self.numel();
        let mut rad = vec![0.0_f64; n];
        for g in &self.gens {
            for (r, v) in rad.iter_mut().zip(g.iter()) {
                *r += v.abs();
            }
        }
        for r in rad.iter_mut() {
            *r = err_up(*r);
        }
        rad
    }

    /// Certified elementwise enclosure `[lo, up]`, rounded OUTWARD.
    ///
    /// `slack_i = rad_i + ec_i + eg_i + 2u|c_i|`; the last term pays for the
    /// single `hi + lo` collapse of the double-double center.
    pub(crate) fn concretize(&self) -> (Vec<f64>, Vec<f64>) {
        let rad = self.radius();
        self.concretize_with_radius(&rad)
    }

    /// [`Self::concretize`] with a precomputed radius (saves a full re-scan).
    pub(crate) fn concretize_with_radius(&self, rad: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.numel();
        let mut lo = vec![0.0_f64; n];
        let mut up = vec![0.0_f64; n];
        for i in 0..n {
            let cv = self.center[i].to_f64();
            let slack = err_up(rad[i] + self.ec[i] + self.eg[i] + 2.0 * U_F64 * cv.abs());
            lo[i] = next_down_f64(cv - slack);
            up[i] = next_up_f64(cv + slack);
        }
        (lo, up)
    }

    /// Per-element NON-RADIUS half-width of the certified enclosure, i.e.
    /// `ec_i + eg_i + 2u|c_i|` as [`Self::concretize_with_radius`] actually
    /// applied it.
    ///
    /// The relaxation transformers size their fold budgets against THIS, not
    /// against `ec + eg` alone: the `2u|c|` term pays for the single
    /// double-double collapse and, at VGG16 activation magnitudes (`|c| ~ 100`),
    /// it is `~2.2e-14` — some 500x larger than the accumulated channel. A fold
    /// budget that omits it declares real rounding artefacts to be genuine
    /// relaxations. MEASURED: 9670 spurious generator columns at the first
    /// max-pool of `vgg16-7` spec1.
    pub(crate) fn error_half_width(&self, rad: &[f64], lo: &[f64], up: &[f64]) -> Vec<f64> {
        (0..self.numel())
            .map(|i| (0.5 * (up[i] - lo[i]) - rad[i]).max(0.0))
            .collect()
    }

    /// True when every stored quantity is finite (a non-finite entry makes the
    /// certificate meaningless, so consumers must refuse).
    pub(crate) fn all_finite(&self) -> bool {
        self.has_valid_layout()
            && self.center.iter().all(|c| c.is_finite())
            && self.ec.iter().all(|v| v.is_finite() && *v >= 0.0)
            && self.eg.iter().all(|v| v.is_finite() && *v >= 0.0)
            && self.gens.iter().all(|g| g.iter().all(|v| v.is_finite()))
    }

    /// Append one generator column that is zero everywhere except at the given
    /// `(index, coefficient)` pairs.
    pub(crate) fn push_sparse_generator(&mut self, entries: &[(usize, f64)]) {
        let mut col = vec![0.0_f64; self.numel()];
        for &(i, v) in entries {
            col[i] = v;
        }
        self.gens.push(col);
    }

    /// Reinterpret the element shape without touching any value (Flatten /
    /// Reshape / Squeeze / Unsqueeze).
    pub(crate) fn reshape(&mut self, shape: Vec<usize>) {
        self.shape = shape;
    }
}

#[cfg(test)]
mod layout_tests {
    use ny_core::dd::Dd;

    use super::DdZono;

    fn state(shape: Vec<usize>, n: usize) -> DdZono {
        DdZono {
            shape,
            center: vec![Dd::ZERO; n],
            gens: vec![vec![0.0; n]],
            ec: vec![0.0; n],
            eg: vec![0.0; n],
        }
    }

    #[test]
    fn layout_validation_rejects_shape_overflow_and_short_channels() {
        assert!(state(vec![2, 3], 6).has_valid_layout());

        let mut malformed = state(vec![2, 3], 6);
        malformed.gens[0].pop();
        assert!(!malformed.has_valid_layout());
        assert!(!malformed.all_finite());

        let overflowing = state(vec![usize::MAX, 2], 0);
        assert!(!overflowing.has_valid_layout());
    }
}
