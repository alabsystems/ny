// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Double-precision CROWN linear bounds for f64 propagation.
//!
//! Parallel to [`LinearBounds`] but operates entirely in f64.
//! Used when `double_fp: true` for VNN-COMP soundnessbench/sat_relu.
//!
//! Reference: alpha-beta-CROWN `double_fp` (`abcrown.py:81-82`).

use ndarray::{Array1, Array2};
use ny_core::{
    dd::{two_prod, two_sum},
    NyError, Result,
};
use ny_tensor::BoundedTensor64;
use tracing::warn;

use super::LinearBounds;

/// Next representable f64 strictly below a finite `x` (toward -∞), for outward
/// (lower-bound) widening. Non-finite input is returned unchanged so ±inf / NaN
/// sentinels are preserved. `0.0` maps to the smallest negative subnormal.
#[inline]
fn next_down_f64_ulp(x: f64) -> f64 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude >= f64::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return -f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

/// Next representable f64 strictly above a finite `x` (toward +∞), for outward
/// (upper-bound) widening. Non-finite input is returned unchanged so ±inf / NaN
/// sentinels are preserved. `0.0` maps to the smallest positive subnormal.
#[inline]
fn next_up_f64_ulp(x: f64) -> f64 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude >= f64::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

#[inline]
fn add_down_f64(a: f64, b: f64) -> f64 {
    let (sum, residual) = two_sum(a, b);
    if !sum.is_finite() {
        sum
    } else if residual < 0.0 {
        next_down_f64_ulp(sum)
    } else {
        sum
    }
}

#[inline]
fn add_up_f64(a: f64, b: f64) -> f64 {
    let (sum, residual) = two_sum(a, b);
    if !sum.is_finite() {
        sum
    } else if residual > 0.0 {
        next_up_f64_ulp(sum)
    } else {
        sum
    }
}

#[inline]
fn mul_down_f64(a: f64, b: f64) -> f64 {
    let (product, residual) = two_prod(a, b);
    if !product.is_finite() {
        product
    } else if residual < 0.0 {
        next_down_f64_ulp(product)
    } else {
        product
    }
}

#[inline]
fn mul_up_f64(a: f64, b: f64) -> f64 {
    let (product, residual) = two_prod(a, b);
    if !product.is_finite() {
        product
    } else if residual > 0.0 {
        next_up_f64_ulp(product)
    } else {
        product
    }
}

/// Double-precision CROWN linear bounds.
///
/// Represents: `lA @ x + lb <= y <= uA @ x + ub` where all arithmetic is f64.
/// Shape: A matrices are (num_outputs, num_inputs), b vectors are (num_outputs,).
#[derive(Debug, Clone)]
#[must_use = "LinearBounds64 from CROWN propagation should not be silently discarded"]
pub struct LinearBounds64 {
    pub(crate) lower_a: Array2<f64>,
    pub(crate) lower_b: Array1<f64>,
    pub(crate) upper_a: Array2<f64>,
    pub(crate) upper_b: Array1<f64>,
    /// Certified per-coefficient error bound on the *lower* relaxation's stored
    /// coefficients, accumulated in f64 through the graph CROWN merge
    /// (#vnncomp-aw-soundness). Invariant: for every (i, j),
    /// `|lower_a[i,j] - true_real_coeff| <= lower_a_err[i,j]`, where `true_real`
    /// is the exact real-arithmetic coefficient the backward pass intends.
    /// All entries are non-negative; non-finite means "degrade this row".
    /// `None` means exact (error 0). Mirror of `LinearBounds::lower_a_err`,
    /// widened/accumulated in f64 so DAG merges never silently drop the
    /// per-contribution coefficient error the f32 carrier brought in.
    pub(crate) lower_a_err: Option<Array2<f64>>,
    /// Certified per-coefficient error bound on the *upper* relaxation's stored
    /// coefficients. Mirror of [`lower_a_err`](Self::lower_a_err) for `upper_a`.
    pub(crate) upper_a_err: Option<Array2<f64>>,
}

impl LinearBounds64 {
    /// Create with validation: shapes and NaN rejection.
    pub fn new(
        lower_a: Array2<f64>,
        lower_b: Array1<f64>,
        upper_a: Array2<f64>,
        upper_b: Array1<f64>,
    ) -> Result<Self> {
        let bounds = Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err: None,
            upper_a_err: None,
        };
        bounds.validate_shapes()?;
        bounds.validate_no_nan()?;
        Ok(bounds)
    }

    /// Identity linear bounds (output = input).
    pub fn identity(dim: usize) -> Self {
        Self {
            lower_a: Array2::eye(dim),
            lower_b: Array1::zeros(dim),
            upper_a: Array2::eye(dim),
            upper_b: Array1::zeros(dim),
            lower_a_err: None,
            upper_a_err: None,
        }
    }

    /// Conservative fallback: A = 0, b = ±∞.
    pub fn conservative(num_outputs: usize, num_inputs: usize) -> Self {
        Self {
            lower_a: Array2::zeros((num_outputs, num_inputs)),
            lower_b: Array1::from_elem(num_outputs, f64::NEG_INFINITY),
            upper_a: Array2::zeros((num_outputs, num_inputs)),
            upper_b: Array1::from_elem(num_outputs, f64::INFINITY),
            lower_a_err: None,
            upper_a_err: None,
        }
    }

    /// Convert from f32 LinearBounds (exact widening).
    ///
    /// The f32→f64 widening of the coefficients, biases AND the certified
    /// coefficient-error matrices is exact, so the carried error invariant
    /// `|coeff - true_real| <= err` is preserved bit-for-bit. Carrying the error
    /// here is what stops the graph CROWN merge from silently dropping the
    /// per-contribution coefficient error at a DAG merge (#vnncomp-aw-soundness).
    pub fn from_f32(lb: &LinearBounds) -> Self {
        Self {
            lower_a: lb.lower_a().mapv(|x| x as f64),
            lower_b: lb.lower_b().mapv(|x| x as f64),
            upper_a: lb.upper_a().mapv(|x| x as f64),
            upper_b: lb.upper_b().mapv(|x| x as f64),
            lower_a_err: lb.lower_a_err().map(|e| e.mapv(|x| x as f64)),
            upper_a_err: lb.upper_a_err().map(|e| e.mapv(|x| x as f64)),
        }
    }

    /// Certified per-coefficient error on the lower relaxation coefficients.
    pub(crate) fn lower_a_err(&self) -> Option<&Array2<f64>> {
        self.lower_a_err.as_ref()
    }

    /// Certified per-coefficient error on the upper relaxation coefficients.
    pub(crate) fn upper_a_err(&self) -> Option<&Array2<f64>> {
        self.upper_a_err.as_ref()
    }

    /// Validated construction with NaN firewall (falls back to conservative).
    pub fn new_or_conservative(
        lower_a: Array2<f64>,
        lower_b: Array1<f64>,
        upper_a: Array2<f64>,
        upper_b: Array1<f64>,
    ) -> Result<Self> {
        let bounds = Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err: None,
            upper_a_err: None,
        };
        bounds.validate_shapes()?;
        match bounds.validate_no_nan() {
            Ok(()) => Ok(bounds),
            Err(_) => {
                let (n_out, n_in) = (bounds.lower_a.nrows(), bounds.lower_a.ncols());
                warn!(
                    "LinearBounds64 NaN firewall: non-finite coefficients ({n_out}x{n_in}), \
                     falling back to conservative bounds"
                );
                Ok(Self::conservative(n_out, n_in))
            }
        }
    }

    /// Concretize against f64 input bounds.
    ///
    /// Computes concrete bounds: for each output i,
    ///   lower_i = lA[i] @ x_best_lower + lb[i]
    ///   upper_i = uA[i] @ x_best_upper + ub[i]
    ///
    /// where x_best is chosen via positive/negative coefficient splitting.
    /// All arithmetic in f64, no intermediate f32 cast.
    pub fn concretize(&self, input: &BoundedTensor64) -> Result<BoundedTensor64> {
        self.validate_shapes()?;
        self.validate_no_nan()?;

        let (in_l, in_u) = input.flatten_to_1d();
        let m = self.lower_a.nrows();
        let n = self.lower_a.ncols();

        if in_l.len() != n {
            return Err(NyError::shape_mismatch(vec![n], vec![in_l.len()]));
        }

        let (mut lower, mut upper) = self.concretize_dot_products(m, n, &in_l, &in_u);

        // The inner loop directs every product/addition using exact EFT residuals.
        // Retain one publication ULP for compatibility with the historical API
        // contract and as a final boundary guard.
        for v in lower.iter_mut() {
            *v = next_down_f64_ulp(*v);
        }
        for v in upper.iter_mut() {
            *v = next_up_f64_ulp(*v);
        }

        Self::repair_inversions(&mut lower, &mut upper, m);

        // Use checked constructor: concretize_dot_products() widens NaN to +/-Inf and
        // repair_inversions() restores lower <= upper, so new() should always succeed.
        // The f64 path is not performance-critical (soundnessbench only). (#4253)
        BoundedTensor64::new(lower.into_dyn(), upper.into_dyn())
    }

    /// Inner dot-product loop for concretization.
    fn concretize_dot_products(
        &self,
        m: usize,
        n: usize,
        in_l: &Array1<f64>,
        in_u: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut lower = Array1::<f64>::zeros(m);
        let mut upper = Array1::<f64>::zeros(m);

        for i in 0..m {
            let lb = self.lower_b[i];
            let ub = self.upper_b[i];

            // Skip degraded rows
            if lb == f64::NEG_INFINITY && ub == f64::INFINITY {
                lower[i] = f64::NEG_INFINITY;
                upper[i] = f64::INFINITY;
                continue;
            }

            let mut sum_l = lb;
            let mut sum_u = ub;

            // Certified coefficient-error penalty (#vnncomp-aw-soundness): for any
            // true coefficient within `[A-err, A+err]` over the input box, the
            // worst case subtracts `Σ_j max(|x_l|,|x_u|)·err` from the lower bound
            // and adds it to the upper bound. Accumulated in f64 alongside the dot.
            let lower_err = self.lower_a_err.as_ref();
            let upper_err = self.upper_a_err.as_ref();
            let mut err_penalty_l = 0.0f64;
            let mut err_penalty_u = 0.0f64;

            for j in 0..n {
                let (x_l, x_u) = (in_l[j], in_u[j]);
                let la = self.lower_a[[i, j]];
                let ua = self.upper_a[[i, j]];

                if lower_err.is_some() || upper_err.is_some() {
                    let mag = x_l.abs().max(x_u.abs());
                    if let Some(le) = lower_err {
                        err_penalty_l = add_up_f64(err_penalty_l, mul_up_f64(le[[i, j]], mag));
                    }
                    if let Some(ue) = upper_err {
                        err_penalty_u = add_up_f64(err_penalty_u, mul_up_f64(ue[[i, j]], mag));
                    }
                }

                // Positive/negative split; 0*inf=0 by skipping zero coefficients.
                if la > 0.0 {
                    sum_l = add_down_f64(sum_l, mul_down_f64(la, x_l));
                } else if la < 0.0 {
                    sum_l = add_down_f64(sum_l, mul_down_f64(la, x_u));
                }
                if ua > 0.0 {
                    sum_u = add_up_f64(sum_u, mul_up_f64(ua, x_u));
                } else if ua < 0.0 {
                    sum_u = add_up_f64(sum_u, mul_up_f64(ua, x_l));
                }
            }

            // Apply the certified-error penalty: lower DOWN, upper UP. A non-finite
            // penalty drives the bound to ±inf (sound, maximally loose).
            if err_penalty_l != 0.0 {
                sum_l = add_down_f64(sum_l, -err_penalty_l);
            }
            if err_penalty_u != 0.0 {
                sum_u = add_up_f64(sum_u, err_penalty_u);
            }

            // NaN guard
            lower[i] = if sum_l.is_nan() {
                f64::NEG_INFINITY
            } else {
                sum_l
            };
            upper[i] = if sum_u.is_nan() { f64::INFINITY } else { sum_u };
        }
        (lower, upper)
    }

    /// Repair inversions (lower > upper) to [-inf, +inf].
    fn repair_inversions(lower: &mut Array1<f64>, upper: &mut Array1<f64>, m: usize) {
        let mut inversions = 0usize;
        for i in 0..m {
            if lower[i] > upper[i] {
                lower[i] = f64::NEG_INFINITY;
                upper[i] = f64::INFINITY;
                inversions += 1;
            }
        }
        if inversions > 0 {
            tracing::debug!(
                inversions,
                num_outputs = m,
                "LinearBounds64::concretize repaired {inversions} inverted elements"
            );
        }
    }

    /// Number of outputs (rows).
    pub fn num_outputs(&self) -> usize {
        self.lower_a.nrows()
    }

    /// Number of inputs (columns).
    pub fn num_inputs(&self) -> usize {
        self.lower_a.ncols()
    }

    // --- Read-only accessors ---

    /// Lower bound coefficient matrix.
    pub fn lower_a(&self) -> &Array2<f64> {
        &self.lower_a
    }

    /// Upper bound coefficient matrix.
    pub fn upper_a(&self) -> &Array2<f64> {
        &self.upper_a
    }

    /// Lower bound bias.
    pub fn lower_b(&self) -> &Array1<f64> {
        &self.lower_b
    }

    /// Upper bound bias.
    pub fn upper_b(&self) -> &Array1<f64> {
        &self.upper_b
    }

    // --- Mutable accessors ---

    /// Mutable reference to lower bound coefficient matrix.
    pub fn lower_a_mut(&mut self) -> &mut Array2<f64> {
        &mut self.lower_a
    }

    /// Mutable reference to upper bound coefficient matrix.
    pub fn upper_a_mut(&mut self) -> &mut Array2<f64> {
        &mut self.upper_a
    }

    /// Mutable reference to lower bound bias.
    pub fn lower_b_mut(&mut self) -> &mut Array1<f64> {
        &mut self.lower_b
    }

    /// Mutable reference to upper bound bias.
    pub fn upper_b_mut(&mut self) -> &mut Array1<f64> {
        &mut self.upper_b
    }

    /// Consume self and return the four components.
    pub fn into_parts(self) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
        (self.lower_a, self.lower_b, self.upper_a, self.upper_b)
    }

    // --- Internal ---

    fn validate_shapes(&self) -> Result<()> {
        if self.lower_a.shape() != self.upper_a.shape() {
            return Err(NyError::InvalidSpec(format!(
                "LinearBounds64: lower_a shape {:?} != upper_a shape {:?}",
                self.lower_a.shape(),
                self.upper_a.shape()
            )));
        }
        let expected = self.lower_a.nrows();
        if self.lower_b.len() != expected {
            return Err(NyError::InvalidSpec(format!(
                "LinearBounds64: lower_b len {} != nrows {}",
                self.lower_b.len(),
                expected
            )));
        }
        if self.upper_b.len() != expected {
            return Err(NyError::InvalidSpec(format!(
                "LinearBounds64: upper_b len {} != nrows {}",
                self.upper_b.len(),
                expected
            )));
        }
        if self
            .lower_a_err
            .as_ref()
            .is_some_and(|error| error.shape() != self.lower_a.shape())
        {
            return Err(NyError::InvalidSpec(
                "LinearBounds64 lower_a_err shape mismatch".into(),
            ));
        }
        if self
            .upper_a_err
            .as_ref()
            .is_some_and(|error| error.shape() != self.upper_a.shape())
        {
            return Err(NyError::InvalidSpec(
                "LinearBounds64 upper_a_err shape mismatch".into(),
            ));
        }
        Ok(())
    }

    fn validate_no_nan(&self) -> Result<()> {
        if self.lower_a.iter().any(|v| !v.is_finite()) {
            return Err(NyError::NumericalInstability(
                "LinearBounds64 lower_a contains NaN or Inf".into(),
            ));
        }
        if self.upper_a.iter().any(|v| !v.is_finite()) {
            return Err(NyError::NumericalInstability(
                "LinearBounds64 upper_a contains NaN or Inf".into(),
            ));
        }
        if self.lower_b.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "LinearBounds64 lower_b contains NaN".into(),
            ));
        }
        if self.upper_b.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "LinearBounds64 upper_b contains NaN".into(),
            ));
        }
        if self
            .lower_a_err
            .iter()
            .chain(self.upper_a_err.iter())
            .flat_map(|error| error.iter())
            .any(|&value| value.is_nan() || value < 0.0)
        {
            return Err(NyError::NumericalInstability(
                "LinearBounds64 coefficient error must be non-negative and non-NaN".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_identity() {
        let lb = LinearBounds64::identity(3);
        assert_eq!(lb.num_outputs(), 3);
        assert_eq!(lb.num_inputs(), 3);
        assert_eq!(lb.lower_a()[[0, 0]], 1.0);
        assert_eq!(lb.lower_a()[[0, 1]], 0.0);
        assert_eq!(lb.lower_b()[0], 0.0);
    }

    #[test]
    fn test_concretize_identity() {
        let input = BoundedTensor64::new(
            arr1(&[-1.0f64, 0.0, 1.0]).into_dyn(),
            arr1(&[1.0f64, 2.0, 3.0]).into_dyn(),
        )
        .unwrap();
        let lb = LinearBounds64::identity(3);
        let result = lb.concretize(&input).unwrap();
        // Identity concretization returns the input bounds, widened outward by one
        // f64 ULP for soundness (#concretize-soundness-hardening): lower rounds down,
        // upper rounds up. Assert sound enclosure within 1 ULP rather than exact eq.
        assert!(result.lower()[0] <= -1.0 && result.lower()[0] >= next_down_f64_ulp(-1.0));
        assert!(result.upper()[0] >= 1.0 && result.upper()[0] <= next_up_f64_ulp(1.0));
        assert!(result.lower()[2] <= 1.0 && result.lower()[2] >= next_down_f64_ulp(1.0));
        assert!(result.upper()[2] >= 3.0 && result.upper()[2] <= next_up_f64_ulp(3.0));
    }

    #[test]
    fn test_concretize_linear() {
        // bounds: y = 2*x + 1 (both lower and upper)
        let la = arr2(&[[2.0f64]]);
        let lb_bias = arr1(&[1.0f64]);
        let lb = LinearBounds64::new(la.clone(), lb_bias.clone(), la, lb_bias).unwrap();

        let input =
            BoundedTensor64::new(arr1(&[0.0f64]).into_dyn(), arr1(&[3.0f64]).into_dyn()).unwrap();
        let result = lb.concretize(&input).unwrap();
        // lower = 2*0 + 1 = 1, upper = 2*3 + 1 = 7. Outward 1-f64-ULP widening
        // (#concretize-soundness-hardening): assert sound enclosure within 1 ULP.
        assert!(result.lower()[0] <= 1.0 && result.lower()[0] >= next_down_f64_ulp(1.0));
        assert!(result.upper()[0] >= 7.0 && result.upper()[0] <= next_up_f64_ulp(7.0));
    }

    #[test]
    fn concretize_directs_each_f64_operation_under_cancellation() {
        let large = 2.0_f64.powi(60);
        let coefficients = arr2(&[[large, 1.0, -large]]);
        let bounds = LinearBounds64::new(
            coefficients.clone(),
            arr1(&[0.0]),
            coefficients,
            arr1(&[0.0]),
        )
        .unwrap();
        let point = arr1(&[1.0, 1.0, 1.0]).into_dyn();
        let input = BoundedTensor64::new(point.clone(), point).unwrap();
        let result = bounds.concretize(&input).unwrap();

        assert!(result.lower()[[0]] <= 1.0, "lower={}", result.lower()[[0]]);
        assert!(result.upper()[[0]] >= 1.0, "upper={}", result.upper()[[0]]);
    }

    #[test]
    fn malformed_coefficient_error_is_rejected_before_concretization() {
        let mut negative = LinearBounds64::identity(1);
        negative.lower_a_err = Some(arr2(&[[-1.0]]));
        negative.upper_a_err = Some(arr2(&[[0.0]]));
        let point = arr1(&[1.0]).into_dyn();
        let input = BoundedTensor64::new(point.clone(), point).unwrap();
        assert!(negative.concretize(&input).is_err());

        let mut wrong_shape = LinearBounds64::identity(1);
        wrong_shape.lower_a_err = Some(Array2::zeros((2, 1)));
        assert!(wrong_shape.concretize(&input).is_err());
    }

    /// #concretize-soundness-hardening: the f64 concretize endpoints are widened
    /// OUTWARD by one f64 ULP, so they strictly enclose the round-to-nearest f64
    /// linear form (lower never above, upper never below the un-widened value).
    #[test]
    fn test_concretize_widens_outward_one_ulp() {
        // y = (1/3) * x over x in [1, 2]: 1/3 is not f64-representable, so the
        // round-to-nearest dot product can be optimistic; the directed widening must
        // push the stored lower down and upper up.
        let third = 1.0f64 / 3.0;
        let la = arr2(&[[third]]);
        let bias = arr1(&[0.0f64]);
        let lb = LinearBounds64::new(la.clone(), bias.clone(), la, bias).unwrap();
        let input =
            BoundedTensor64::new(arr1(&[1.0f64]).into_dyn(), arr1(&[2.0f64]).into_dyn()).unwrap();
        let result = lb.concretize(&input).unwrap();

        // Round-to-nearest (un-widened) endpoints.
        let rn_lower = third * 1.0;
        let rn_upper = third * 2.0;
        // Stored endpoints must enclose the round-to-nearest values (sound), and be
        // no looser than one ULP outward.
        assert!(result.lower()[0] <= rn_lower);
        assert!(result.lower()[0] >= next_down_f64_ulp(rn_lower));
        assert!(result.upper()[0] >= rn_upper);
        assert!(result.upper()[0] <= next_up_f64_ulp(rn_upper));
        assert!(result.lower()[0] <= result.upper()[0]);
    }

    #[test]
    fn test_conservative() {
        let lb = LinearBounds64::conservative(2, 3);
        assert_eq!(lb.num_outputs(), 2);
        assert_eq!(lb.num_inputs(), 3);
        assert!(lb.lower_a().iter().all(|v| *v == 0.0));
        assert!(lb.lower_b().iter().all(|v| *v == f64::NEG_INFINITY));
        assert!(lb.upper_b().iter().all(|v| *v == f64::INFINITY));
    }

    #[test]
    fn test_new_rejects_nan_coefficient() {
        let la = arr2(&[[f64::NAN, 1.0]]);
        let lb = arr1(&[0.0]);
        let ua = arr2(&[[1.0, 1.0]]);
        let ub = arr1(&[0.0]);
        assert!(LinearBounds64::new(la, lb, ua, ub).is_err());
    }

    #[test]
    fn test_new_or_conservative_nan_falls_back() {
        let la = arr2(&[[f64::NAN, 1.0]]);
        let lb = arr1(&[0.0]);
        let ua = arr2(&[[1.0, 1.0]]);
        let ub = arr1(&[0.0]);
        let result = LinearBounds64::new_or_conservative(la, lb, ua, ub).unwrap();
        assert!(result.lower_a().iter().all(|v| *v == 0.0));
        assert!(result.lower_b().iter().all(|v| *v == f64::NEG_INFINITY));
    }
}
