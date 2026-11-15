// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified additive widening: y = x + e with e in [-epsilon, +epsilon].
//!
//! The Minkowski-sum enclosure of a per-element perturbation budget. Unlike
//! `QdqPerturbationLayer` — whose envelope certifies the typed FLOAT32 QDQ
//! rounding program (division, ties-to-even, saturation, reconstruction) —
//! this layer carries no quantization story at all: the additive widening IS
//! the declared op semantics, so the interval image below is the exact set
//! image and every propagation path is an enclosure by construction.
//!
//! Downstream consumers (nn) use it for index-agnostic Embedding relaxations:
//! a midpoint row plus a per-dimension epsilon budget covering every row the
//! lookup could select.
//!
//! Epsilons are PER-ELEMENT and apply to the propagated tensor's TRAILING
//! dimensions (a rank-0 epsilon is the uniform-scalar case). In row-major
//! layout that trailing-shape broadcast is exactly cyclic indexing of the
//! flattened epsilons, which is how the CROWN paths apply them. Shape
//! validation is exact where the shape is visible: IBP and batched CROWN
//! (via the recorded input shape) enforce the exact trailing-shape rule and
//! fail closed on mismatch; the scalar CROWN path sees only the flattened
//! input dimension, so it can enforce only whole-tile divisibility — its
//! residual assumption is that the layer was built for the tensor it is
//! attached to, which both in-tree producers guarantee by tiling epsilons to
//! the exact flattened length.
//!
//! Certified coefficient-error carriers (`lower_a_err`/`upper_a_err`,
//! `#vnncomp-aw-soundness`) pass through unchanged — this layer keeps the
//! coefficient matrices verbatim, so a carried error bound remains valid —
//! and the bias widening uses `(|A| + err) · eps`, the largest `|Ã·e|` any
//! true coefficient `Ã` within the certified error band could produce.
//! Silently dropping the carrier would publish the coefficients as exact,
//! the false-proof class the carrier exists to close. The layer still does
//! not REGISTER as a carrier (`propagates_coeff_err`), so gated dispatchers
//! keep discharging error before it — the passthrough here covers the
//! ungated backward paths.

use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn, Zip};
use ny_core::{NyError, Result};
use ny_tensor::{add_up_f32, mul_up_f32, sub_down_f32, BoundedTensor};
use std::borrow::Cow;

use crate::layers::common::BoundPropagation;
use crate::{BatchedLinearBounds, LinearBounds};

/// Additive widening layer: y = x + e with e in [-epsilon, +epsilon] element-wise.
#[derive(Debug, Clone)]
pub struct EpsilonExpandLayer {
    /// Per-element perturbation radii applied to the propagated tensor's
    /// trailing dimensions. Every entry is finite and >= 0 (`try_new`).
    epsilons: ArrayD<f32>,
}

/// Upper bound on `mag * eps` rounded toward +inf, for a non-negative
/// coefficient magnitude `mag` (already `|A|`, or `|A| + err` when a
/// certified coefficient-error carrier is present).
///
/// `eps == 0.0` pins the perturbation of that element to exactly 0, so the
/// contribution is a TRUE zero even for an infinite magnitude (a conservative
/// ±inf coefficient or a sanitized +inf error entry, whose float product
/// would otherwise be NaN): whatever the true coefficient is, it multiplies
/// an exactly-zero perturbation. For `eps > 0` an infinite magnitude
/// propagates to an infinite — maximally conservative — delta.
fn mag_mul_eps_up(mag: f32, eps: f32) -> f32 {
    if eps == 0.0 {
        0.0
    } else {
        mul_up_f32(mag, eps)
    }
}

/// Per-element coefficient magnitude for the widening term: `|A|`, plus the
/// certified per-coefficient error when a carrier is present (rounded toward
/// +inf so the magnitude never understates what the true coefficient could
/// be).
fn coeff_magnitudes<D: ndarray::Dimension>(
    coeffs: &ndarray::Array<f32, D>,
    err: Option<&ndarray::Array<f32, D>>,
) -> ndarray::Array<f32, D> {
    match err {
        Some(err) => Zip::from(coeffs)
            .and(err)
            .map_collect(|&coeff, &err| add_up_f32(coeff.abs(), err)),
        None => coeffs.mapv(f32::abs),
    }
}

/// `bias - delta` rounded toward -inf, with an infinite widening collapsing to
/// -inf instead of the NaN that `(+inf) - (+inf)` would produce.
fn widen_bias_down(bias: f32, delta: f32) -> f32 {
    if delta == f32::INFINITY {
        f32::NEG_INFINITY
    } else {
        sub_down_f32(bias, delta)
    }
}

/// `bias + delta` rounded toward +inf; mirror of [`widen_bias_down`].
fn widen_bias_up(bias: f32, delta: f32) -> f32 {
    if delta == f32::INFINITY {
        f32::INFINITY
    } else {
        add_up_f32(bias, delta)
    }
}

impl EpsilonExpandLayer {
    /// Create the layer from per-element epsilons.
    ///
    /// Every entry must be finite and >= 0.0 — a negative or non-finite radius
    /// has no enclosure semantics — and the tensor must be non-empty (a rank-0
    /// scalar is fine and means one radius for every element).
    pub fn try_new(epsilons: ArrayD<f32>) -> Result<Self> {
        if epsilons.is_empty() {
            return Err(NyError::InvalidSpec(
                "EpsilonExpand requires a non-empty epsilon tensor".to_string(),
            ));
        }
        if let Some(bad) = epsilons
            .iter()
            .find(|value| !(value.is_finite() && **value >= 0.0))
        {
            return Err(NyError::InvalidSpec(format!(
                "EpsilonExpand requires finite non-negative epsilons, got {bad}"
            )));
        }
        Ok(Self { epsilons })
    }

    /// Create the layer with one uniform scalar epsilon for every element.
    pub fn uniform(epsilon: f32) -> Result<Self> {
        Self::try_new(ArrayD::from_elem(IxDyn(&[]), epsilon))
    }

    /// The per-element perturbation radii.
    pub fn epsilons(&self) -> &ArrayD<f32> {
        &self.epsilons
    }

    /// Fail closed unless the epsilons match the tensor's trailing dimensions
    /// exactly (no size-1 stretching — that would silently reinterpret the
    /// budget's element mapping).
    fn validate_trailing_shape(&self, shape: &[usize], context: &str) -> Result<()> {
        let eps_shape = self.epsilons.shape();
        if shape.len() < eps_shape.len() || shape[shape.len() - eps_shape.len()..] != *eps_shape {
            return Err(NyError::InvalidSpec(format!(
                "EpsilonExpand {context}: epsilon shape {eps_shape:?} must match the trailing dimensions of tensor shape {shape:?}"
            )));
        }
        Ok(())
    }

    /// Fail closed unless the flattened input dimension is a whole number of
    /// epsilon tiles. CROWN backward only sees the flattened dimension, where
    /// the trailing-shape broadcast is cyclic indexing (row-major layout).
    fn validate_flat_input_dim(&self, in_dim: usize, context: &str) -> Result<()> {
        let eps_len = self.epsilons.len();
        if eps_len == 0 || !in_dim.is_multiple_of(eps_len) {
            return Err(NyError::InvalidSpec(format!(
                "EpsilonExpand {context}: input dimension {in_dim} is not a multiple of the {eps_len} epsilons; refusing to guess the element mapping"
            )));
        }
        Ok(())
    }

    /// Per-row bias widening `Σ_j mag[.., j] * eps[j mod n]` over non-negative
    /// coefficient magnitudes, every product and partial sum rounded toward
    /// +inf so the widening can never round inward.
    fn bias_delta_up(&self, magnitudes: &ArrayD<f32>, axis: Axis) -> ArrayD<f32> {
        let eps_flat: Vec<f32> = self.epsilons.iter().copied().collect();
        magnitudes.map_axis(axis, |row| {
            row.iter().enumerate().fold(0.0_f32, |acc, (col, &mag)| {
                add_up_f32(acc, mag_mul_eps_up(mag, eps_flat[col % eps_flat.len()]))
            })
        })
    }

    /// Batched CROWN backward: coefficients pass through unchanged, biases
    /// widen outward by `Σ (|A| + err) · eps` per output row, and any certified
    /// coefficient-error carrier passes through unchanged (the coefficient
    /// matrices are kept verbatim, so the carried bound remains valid).
    ///
    /// Same shape as `QdqPerturbationLayer::propagate_linear_batched_with_bounds`
    /// minus the saturation gate — this layer's semantics hold on the whole of
    /// R, so no pre-activation bounds are needed.
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        let ndim = bounds.lower_a().ndim();
        if ndim < 2 {
            return Err(NyError::InvalidSpec(
                "EpsilonExpand batched CROWN: bounds must have at least 2 dimensions".to_string(),
            ));
        }
        let in_dim = bounds.lower_a().shape()[ndim - 1];
        self.validate_flat_input_dim(in_dim, "batched CROWN")?;
        // The batched path records the true (unflattened) input tensor shape,
        // so the exact trailing-shape rule is enforceable here — divisibility
        // alone would accept cyclic misassignments of the per-element budget.
        self.validate_trailing_shape(bounds.input_shape(), "batched CROWN")?;

        let lower_mag = coeff_magnitudes(bounds.lower_a(), bounds.lower_a_err.as_ref());
        let upper_mag = coeff_magnitudes(bounds.upper_a(), bounds.upper_a_err.as_ref());
        let lower_delta = self.bias_delta_up(&lower_mag, Axis(ndim - 1));
        let upper_delta = self.bias_delta_up(&upper_mag, Axis(ndim - 1));
        let lower_b = Zip::from(bounds.lower_b())
            .and(&lower_delta)
            .map_collect(|&bias, &delta| widen_bias_down(bias, delta));
        let upper_b = Zip::from(bounds.upper_b())
            .and(&upper_delta)
            .map_collect(|&bias, &delta| widen_bias_up(bias, delta));

        let mut result = BatchedLinearBounds::new(
            bounds.lower_a().clone(),
            lower_b,
            bounds.upper_a().clone(),
            upper_b,
            bounds.input_shape().to_vec(),
            bounds.output_shape().to_vec(),
        )?;
        if bounds.lower_a_err.is_some() || bounds.upper_a_err.is_some() {
            // `None` on one side means "exact" (error 0); materialize zeros so
            // the carrier on the other side is never silently dropped.
            let lower_err = bounds
                .lower_a_err
                .clone()
                .unwrap_or_else(|| ArrayD::zeros(bounds.lower_a().raw_dim()));
            let upper_err = bounds
                .upper_a_err
                .clone()
                .unwrap_or_else(|| ArrayD::zeros(bounds.upper_a().raw_dim()));
            result.set_coeff_err(lower_err, upper_err);
        }
        Ok(result)
    }
}

impl BoundPropagation for EpsilonExpandLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.validate_trailing_shape(input.shape(), "IBP")?;
        let eps = self
            .epsilons
            .broadcast(input.lower().raw_dim())
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "EpsilonExpand IBP: epsilon shape {:?} does not broadcast over input shape {:?}",
                    self.epsilons.shape(),
                    input.shape()
                ))
            })?;
        // DIRECTED: `value - epsilon` under round-to-nearest can land ABOVE the
        // true difference, so the widened lower bound could sit inside the
        // interval it is meant to enlarge — the perturbation would be silently
        // smaller than requested. Exact whenever the shift is representable
        // (same rationale as QdqPerturbationLayer). `new_allow_infinite` (NaN
        // still rejected) lets a widening that overflows f32 — or an already
        // conservative ±inf input box — saturate to a sound infinite bound
        // instead of hard-erroring.
        BoundedTensor::new_allow_infinite(
            Zip::from(input.lower())
                .and(&eps)
                .map_collect(|&value, &eps| sub_down_f32(value, eps)),
            Zip::from(input.upper())
                .and(&eps)
                .map_collect(|&value, &eps| add_up_f32(value, eps)),
        )
    }

    /// CROWN backward: identity linear relaxation. `A(x + e) + b` with
    /// `|e_j| <= eps_j` is enclosed by keeping `A` and widening the bias
    /// interval by `± Σ_j (|A[i,j]| + err[i,j]) · eps_j` — no pre-activation
    /// bounds needed (the semantics hold on the whole of R, unlike the
    /// saturating QDQ path).
    ///
    /// A certified coefficient-error carrier (`#vnncomp-aw-soundness`) passes
    /// through UNCHANGED: the coefficient matrices are kept verbatim, so the
    /// carried per-coefficient bound remains valid, and the widening term uses
    /// `|A| + err` so no true coefficient within the certified band can
    /// produce a larger `|Ã·e|` than accounted for. Dropping the carrier here
    /// would publish the coefficients as exact on the ungated backward paths
    /// (sequential alpha-CROWN, beta-CROWN fallback) — the false-proof class
    /// the carrier exists to close.
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        self.validate_flat_input_dim(bounds.lower_a().ncols(), "CROWN")?;
        let eps_flat: Vec<f32> = self.epsilons.iter().copied().collect();
        let delta_up = |magnitudes: &Array2<f32>| -> Array1<f32> {
            magnitudes.map_axis(Axis(1), |row| {
                row.iter().enumerate().fold(0.0_f32, |acc, (col, &mag)| {
                    add_up_f32(acc, mag_mul_eps_up(mag, eps_flat[col % eps_flat.len()]))
                })
            })
        };
        let lower_mag = coeff_magnitudes(bounds.lower_a(), bounds.lower_a_err());
        let upper_mag = coeff_magnitudes(bounds.upper_a(), bounds.upper_a_err());
        let lower_delta = delta_up(&lower_mag);
        let upper_delta = delta_up(&upper_mag);

        let mut result = LinearBounds::new(
            bounds.lower_a().clone(),
            Zip::from(bounds.lower_b())
                .and(&lower_delta)
                .map_collect(|&bias, &delta| widen_bias_down(bias, delta)),
            bounds.upper_a().clone(),
            Zip::from(bounds.upper_b())
                .and(&upper_delta)
                .map_collect(|&bias, &delta| widen_bias_up(bias, delta)),
        )?;
        if bounds.lower_a_err().is_some() || bounds.upper_a_err().is_some() {
            // `None` on one side means "exact" (error 0); materialize zeros so
            // the carrier on the other side is never silently dropped.
            let lower_err = bounds
                .lower_a_err()
                .cloned()
                .unwrap_or_else(|| Array2::zeros(bounds.lower_a().raw_dim()));
            let upper_err = bounds
                .upper_a_err()
                .cloned()
                .unwrap_or_else(|| Array2::zeros(bounds.upper_a().raw_dim()));
            result.set_coeff_err(lower_err, upper_err);
        }
        Ok(Cow::Owned(result))
    }
}

#[cfg(test)]
mod tests {
    use super::EpsilonExpandLayer;
    use crate::layers::common::BoundPropagation;
    use crate::{BatchedLinearBounds, LinearBounds};
    use ndarray::{arr1, arr2, ArrayD, Axis, IxDyn};
    use ny_tensor::BoundedTensor;

    /// Deterministic xorshift for sampled soundness checks.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        fn u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 29) as u32
        }
        /// Uniform-ish sample in [lo, hi] on a 1/64 grid (exact in f32/f64 for
        /// the small dyadic endpoints used below, so test arithmetic is exact).
        fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (self.u32() % 65) as f32 / 64.0
        }
    }

    fn eps_1d(values: &[f32]) -> ArrayD<f32> {
        arr1(values).into_dyn()
    }

    #[test]
    fn try_new_rejects_invalid_epsilons() {
        for bad in [-1.0e-6_f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = EpsilonExpandLayer::try_new(eps_1d(&[0.5, bad]))
                .expect_err("non-finite or negative epsilon must fail closed");
            assert!(error.to_string().contains("finite non-negative"), "{error}");
        }
        let error = EpsilonExpandLayer::try_new(ArrayD::zeros(IxDyn(&[0])))
            .expect_err("empty epsilon tensor must fail closed");
        assert!(error.to_string().contains("non-empty"), "{error}");

        let error =
            EpsilonExpandLayer::uniform(-0.5).expect_err("negative uniform epsilon must fail");
        assert!(error.to_string().contains("finite non-negative"), "{error}");
    }

    #[test]
    fn ibp_widens_exactly_when_shift_is_representable() {
        let layer = EpsilonExpandLayer::try_new(eps_1d(&[0.5, 0.0, 2.0])).unwrap();
        let input = BoundedTensor::new(
            arr1(&[1.0_f32, -3.0, 0.25]).into_dyn(),
            arr1(&[2.0_f32, -1.0, 0.75]).into_dyn(),
        )
        .unwrap();
        let output = layer.propagate_ibp(&input).unwrap();
        assert_eq!(
            output.lower().iter().copied().collect::<Vec<_>>(),
            vec![0.5, -3.0, -1.75]
        );
        assert_eq!(
            output.upper().iter().copied().collect::<Vec<_>>(),
            vec![2.5, -1.0, 2.75]
        );
    }

    #[test]
    fn ibp_broadcasts_over_leading_dimensions() {
        let layer = EpsilonExpandLayer::try_new(eps_1d(&[1.0, 0.25])).unwrap();
        let input = BoundedTensor::new(
            arr2(&[[0.0_f32, 0.0], [4.0, 4.0]]).into_dyn(),
            arr2(&[[1.0_f32, 1.0], [5.0, 5.0]]).into_dyn(),
        )
        .unwrap();
        let output = layer.propagate_ibp(&input).unwrap();
        assert_eq!(
            output.lower().iter().copied().collect::<Vec<_>>(),
            vec![-1.0, -0.25, 3.0, 3.75]
        );
        assert_eq!(
            output.upper().iter().copied().collect::<Vec<_>>(),
            vec![2.0, 1.25, 6.0, 5.25]
        );

        // Uniform (rank-0) epsilons broadcast over every shape.
        let uniform = EpsilonExpandLayer::uniform(0.5).unwrap();
        let output = uniform.propagate_ibp(&input).unwrap();
        assert_eq!(
            output.lower().iter().copied().collect::<Vec<_>>(),
            vec![-0.5, -0.5, 3.5, 3.5]
        );
    }

    #[test]
    fn ibp_rounds_outward_at_extreme_magnitudes() {
        // At 2^26 the f32 grid spacing is 4: the true shifted endpoints
        // 2^26 ∓ 1 are non-representable and round-to-nearest would round
        // BOTH back to 2^26 — the widening would silently vanish. Directed
        // rounding must land strictly outside instead.
        let value = (1u32 << 26) as f32;
        let layer = EpsilonExpandLayer::uniform(1.0).unwrap();
        let input =
            BoundedTensor::new(arr1(&[value]).into_dyn(), arr1(&[value]).into_dyn()).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();
        let lower = output.lower()[[0]];
        let upper = output.upper()[[0]];
        assert!(
            lower <= value - 1.0 && (lower as f64) <= (value as f64) - 1.0,
            "lower {lower} must enclose {value} - 1"
        );
        assert!(
            upper >= value + 1.0 && (upper as f64) >= (value as f64) + 1.0,
            "upper {upper} must enclose {value} + 1"
        );
        assert!(lower < value, "widening must not round back inward");
        assert!(upper > value, "widening must not round back inward");
    }

    #[test]
    fn ibp_fails_closed_on_trailing_shape_mismatch() {
        let layer = EpsilonExpandLayer::try_new(eps_1d(&[0.5, 0.5, 0.5])).unwrap();
        let input = BoundedTensor::new(
            arr1(&[0.0_f32; 4]).into_dyn(),
            arr1(&[1.0_f32; 4]).into_dyn(),
        )
        .unwrap();
        let error = layer
            .propagate_ibp(&input)
            .expect_err("trailing-shape mismatch must fail closed");
        assert!(error.to_string().contains("trailing dimensions"), "{error}");

        // Size-1 stretching is deliberately NOT accepted as a match.
        let one = EpsilonExpandLayer::try_new(eps_1d(&[0.5])).unwrap();
        let error = one
            .propagate_ibp(&input)
            .expect_err("size-1 stretching must fail closed (use the rank-0 uniform form)");
        assert!(error.to_string().contains("trailing dimensions"), "{error}");
    }

    #[test]
    fn crown_fails_closed_on_flat_dim_mismatch() {
        let layer = EpsilonExpandLayer::try_new(eps_1d(&[0.5, 0.5, 0.5])).unwrap();
        let bounds = LinearBounds::identity(4);
        let error = layer
            .propagate_linear(&bounds)
            .expect_err("4 columns over 3 epsilons must fail closed");
        assert!(error.to_string().contains("not a multiple"), "{error}");
    }

    #[test]
    fn crown_identity_widens_bias_by_epsilon() {
        let layer = EpsilonExpandLayer::try_new(eps_1d(&[0.5, 0.0, 2.0])).unwrap();
        let bounds = LinearBounds::identity(3);
        let result = layer.propagate_linear(&bounds).unwrap();
        assert_eq!(result.lower_a(), bounds.lower_a());
        assert_eq!(result.upper_a(), bounds.upper_a());
        assert_eq!(
            result.lower_b().iter().copied().collect::<Vec<_>>(),
            vec![-0.5, 0.0, -2.0]
        );
        assert_eq!(
            result.upper_b().iter().copied().collect::<Vec<_>>(),
            vec![0.5, 0.0, 2.0]
        );
    }

    /// CROWN soundness against the concrete op: for every x in the box and
    /// every |e_j| <= eps_j, the widened bounds over x must enclose A(x+e)+b.
    /// Dyadic 1/64-grid samples over small magnitudes keep every dot product
    /// exact in f64, so the assertion tolerates no rounding slack at all.
    #[test]
    fn crown_sound_against_sampled_concrete_perturbations() {
        let eps = [0.25_f32, 0.0, 1.0, 0.5];
        let layer = EpsilonExpandLayer::try_new(eps_1d(&eps)).unwrap();
        // Mixed-sign coefficients exercise the |A| folding on both rows.
        let a = arr2(&[[1.0_f32, -2.0, 0.5, 0.0], [-0.25, 0.0, -1.0, 2.0]]);
        let b = arr1(&[0.5_f32, -1.0]);
        let bounds = LinearBounds::new(a.clone(), b.clone(), a.clone(), b.clone()).unwrap();
        let result = layer.propagate_linear(&bounds).unwrap();

        let x_lower = [-1.0_f32, 0.0, -0.5, 2.0];
        let x_upper = [1.0_f32, 1.0, 0.5, 3.0];
        let mut rng = Rng::new(0x5eed);
        for _ in 0..500 {
            let x: Vec<f64> = (0..4)
                .map(|j| rng.f32_in(x_lower[j], x_upper[j]) as f64)
                .collect();
            let e: Vec<f64> = (0..4).map(|j| rng.f32_in(-eps[j], eps[j]) as f64).collect();
            for row in 0..2 {
                let concrete: f64 = (0..4)
                    .map(|j| a[[row, j]] as f64 * (x[j] + e[j]))
                    .sum::<f64>()
                    + b[row] as f64;
                let lower: f64 = (0..4)
                    .map(|j| result.lower_a()[[row, j]] as f64 * x[j])
                    .sum::<f64>()
                    + result.lower_b()[row] as f64;
                let upper: f64 = (0..4)
                    .map(|j| result.upper_a()[[row, j]] as f64 * x[j])
                    .sum::<f64>()
                    + result.upper_b()[row] as f64;
                assert!(
                    lower <= concrete && concrete <= upper,
                    "SOUNDNESS VIOLATION row={row}: {concrete} not in [{lower}, {upper}] for x={x:?} e={e:?}"
                );
            }
        }
    }

    #[test]
    fn crown_cyclic_tiling_matches_trailing_broadcast() {
        // Input shape [2, 2] flattened to 4 columns over epsilons [a, b]:
        // column j takes eps[j mod 2], the row-major trailing broadcast.
        let layer = EpsilonExpandLayer::try_new(eps_1d(&[1.0, 0.25])).unwrap();
        let bounds = LinearBounds::identity(4);
        let result = layer.propagate_linear(&bounds).unwrap();
        assert_eq!(
            result.upper_b().iter().copied().collect::<Vec<_>>(),
            vec![1.0, 0.25, 1.0, 0.25]
        );
    }

    #[test]
    fn batched_crown_matches_scalar_widening() {
        let layer = EpsilonExpandLayer::try_new(eps_1d(&[0.25, 0.0, 1.0, 0.5])).unwrap();
        let a = arr2(&[[1.0_f32, -2.0, 0.5, 0.0], [-0.25, 0.0, -1.0, 2.0]]);
        let b = arr1(&[0.5_f32, -1.0]);
        let scalar_in = LinearBounds::new(a.clone(), b.clone(), a.clone(), b.clone()).unwrap();
        let scalar = layer.propagate_linear(&scalar_in).unwrap();

        let batched_a = a.clone().into_dyn().insert_axis(Axis(0));
        let batched_b = b.into_dyn().insert_axis(Axis(0));
        let batched = layer
            .propagate_linear_batched(
                &BatchedLinearBounds::new(
                    batched_a.clone(),
                    batched_b.clone(),
                    batched_a,
                    batched_b,
                    vec![1, 4],
                    vec![1, 2],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            batched.lower_b().iter().copied().collect::<Vec<_>>(),
            scalar.lower_b().iter().copied().collect::<Vec<_>>()
        );
        assert_eq!(
            batched.upper_b().iter().copied().collect::<Vec<_>>(),
            scalar.upper_b().iter().copied().collect::<Vec<_>>()
        );
        assert_eq!(batched.lower_a(), &a.into_dyn().insert_axis(Axis(0)));
    }

    #[test]
    fn batched_crown_infinite_coefficient_stays_sound_without_nan() {
        // Conservative ±inf coefficients are legal in BatchedLinearBounds.
        // eps == 0 pins that element's perturbation to zero (true-zero
        // contribution); eps > 0 must widen the bias all the way out instead
        // of manufacturing NaN.
        let a = arr2(&[[f32::INFINITY, 1.0]])
            .into_dyn()
            .insert_axis(Axis(0));
        let b = arr1(&[0.0_f32]).into_dyn().insert_axis(Axis(0));
        let make = |eps: &[f32]| -> (EpsilonExpandLayer, BatchedLinearBounds) {
            (
                EpsilonExpandLayer::try_new(eps_1d(eps)).unwrap(),
                BatchedLinearBounds::new(
                    a.clone(),
                    b.clone(),
                    a.clone(),
                    b.clone(),
                    vec![1, 2],
                    vec![1, 1],
                )
                .unwrap(),
            )
        };

        let (layer, bounds) = make(&[0.0, 0.5]);
        let result = layer.propagate_linear_batched(&bounds).unwrap();
        assert_eq!(result.lower_b()[[0, 0]], -0.5);
        assert_eq!(result.upper_b()[[0, 0]], 0.5);

        let (layer, bounds) = make(&[0.5, 0.5]);
        let result = layer.propagate_linear_batched(&bounds).unwrap();
        assert_eq!(result.lower_b()[[0, 0]], f32::NEG_INFINITY);
        assert_eq!(result.upper_b()[[0, 0]], f32::INFINITY);
    }

    #[test]
    fn crown_backward_needs_no_pre_activation_bounds() {
        let layer = EpsilonExpandLayer::uniform(0.5).unwrap();
        assert!(!layer.requires_pre_activation_bounds());
        let result = layer
            .propagate_crown_backward(&LinearBounds::identity(2), None)
            .unwrap();
        assert_eq!(
            result.upper_b().iter().copied().collect::<Vec<_>>(),
            vec![0.5, 0.5]
        );
    }

    /// A certified coefficient-error carrier must pass through UNCHANGED (the
    /// coefficient matrices are kept verbatim) and the bias widening must use
    /// `(|A| + err) · eps`: the true coefficient can sit anywhere in the
    /// certified band, so the widening accounts for the largest `|Ã·e|`.
    /// Silently dropping the carrier is the false-proof class this pins.
    #[test]
    fn crown_carries_coefficient_error_and_widens_by_it() {
        let layer = EpsilonExpandLayer::try_new(eps_1d(&[0.5, 2.0])).unwrap();
        let a = arr2(&[[1.0_f32, -1.0]]);
        let b = arr1(&[0.0_f32]);
        let mut bounds = LinearBounds::new(a.clone(), b.clone(), a, b).unwrap();
        let err = arr2(&[[0.25_f32, 0.125]]);
        bounds.set_coeff_err(err.clone(), err.clone());

        let result = layer.propagate_linear(&bounds).unwrap();
        // err_out == err_in on both sides — the carrier is never dropped.
        assert_eq!(result.lower_a_err(), Some(&err));
        assert_eq!(result.upper_a_err(), Some(&err));
        // Widening: (|1| + 0.25) * 0.5 + (|-1| + 0.125) * 2.0 = 0.625 + 2.25
        // = 2.875 — exact dyadic arithmetic, so no rounding slack in the pin.
        assert_eq!(
            result.lower_b().iter().copied().collect::<Vec<_>>(),
            vec![-2.875]
        );
        assert_eq!(
            result.upper_b().iter().copied().collect::<Vec<_>>(),
            vec![2.875]
        );
    }

    /// Batched parity for the carrier passthrough: err survives, and the bias
    /// widening includes the certified error band.
    #[test]
    fn batched_crown_carries_coefficient_error_and_widens_by_it() {
        let layer = EpsilonExpandLayer::try_new(eps_1d(&[0.5, 2.0])).unwrap();
        let a = arr2(&[[1.0_f32, -1.0]]).into_dyn().insert_axis(Axis(0));
        let b = arr1(&[0.0_f32]).into_dyn().insert_axis(Axis(0));
        let mut bounds =
            BatchedLinearBounds::new(a.clone(), b.clone(), a, b, vec![1, 2], vec![1, 1]).unwrap();
        let err = arr2(&[[0.25_f32, 0.125]]).into_dyn().insert_axis(Axis(0));
        bounds.set_coeff_err(err.clone(), err.clone());

        let result = layer.propagate_linear_batched(&bounds).unwrap();
        assert_eq!(result.lower_a_err.as_ref(), Some(&err));
        assert_eq!(result.upper_a_err.as_ref(), Some(&err));
        assert_eq!(result.lower_b()[[0, 0]], -2.875);
        assert_eq!(result.upper_b()[[0, 0]], 2.875);
    }

    /// Batched CROWN sees the recorded (unflattened) input shape, so it must
    /// enforce the exact trailing-shape rule — flat-dim divisibility alone
    /// would cyclically misassign per-element budgets (an under-widening).
    #[test]
    fn batched_crown_fails_closed_on_trailing_shape_mismatch() {
        // eps len 2 divides flat dim 6, but the true tensor shape [2, 3] does
        // not end in [2] — must refuse rather than tile a,b,a,b,a,b.
        let layer = EpsilonExpandLayer::try_new(eps_1d(&[1.0, 0.25])).unwrap();
        let a = ArrayD::zeros(IxDyn(&[1, 1, 6]));
        let b = ArrayD::zeros(IxDyn(&[1, 1]));
        let bounds =
            BatchedLinearBounds::new(a.clone(), b.clone(), a, b, vec![2, 3], vec![1, 1]).unwrap();
        let error = layer
            .propagate_linear_batched(&bounds)
            .expect_err("divisible-but-mismatched trailing shape must fail closed");
        assert!(error.to_string().contains("trailing dimensions"), "{error}");
    }

    /// A widening that overflows f32 saturates to a sound infinite bound
    /// instead of hard-erroring (conservative-bounds convention).
    #[test]
    fn ibp_saturates_to_infinite_bounds_instead_of_erroring() {
        let layer = EpsilonExpandLayer::uniform(f32::MAX).unwrap();
        let input = BoundedTensor::new(arr1(&[-f32::MAX]).into_dyn(), arr1(&[f32::MAX]).into_dyn())
            .unwrap();
        let output = layer
            .propagate_ibp(&input)
            .expect("overflowed widening must saturate, not error");
        assert_eq!(output.lower()[[0]], f32::NEG_INFINITY);
        assert_eq!(output.upper()[[0]], f32::INFINITY);
    }
}
