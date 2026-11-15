// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array, Array1, Array2, ArrayBase, ArrayD, ArrayViewD, Data, Dimension, Zip};
use ny_core::{
    checked_shape_product,
    dd::{next_down_f64, next_up_f64, two_sum},
};
use ny_tensor::{next_down_f32, next_up_f32};
use std::mem::size_of;

use super::BatchedLinearBounds;
use crate::layers::linear::bias::{
    f32_to_f64_exact, f64_to_f32_up_no_subnormal, nonnegative_f32_error_or_infinity,
};

/// Four reshaped views (lower_a, lower_b, upper_a, upper_b) for accumulation.
type BoundsViewQuad<'a> = (
    ArrayViewD<'a, f32>,
    ArrayViewD<'a, f32>,
    ArrayViewD<'a, f32>,
    ArrayViewD<'a, f32>,
);

/// Double-precision batched linear bounds for merge-point accumulation.
///
/// This stays crate-private because the carrier exists only to preserve
/// precision at batched graph-CROWN DAG fan-in points before materializing
/// back to `BatchedLinearBounds`.
#[derive(Debug, Clone)]
#[must_use = "BatchedLinearBounds64 from merge accumulation should not be discarded"]
pub(crate) struct BatchedLinearBounds64 {
    lower_a: ArrayD<f64>,
    lower_b: ArrayD<f64>,
    upper_a: ArrayD<f64>,
    upper_b: ArrayD<f64>,
    /// Certified absolute coefficient error carried across f64 fan-in.
    /// `None` means the stored coefficients are exact at this point.
    lower_a_err: Option<ArrayD<f64>>,
    /// Upper-coefficient counterpart of [`Self::lower_a_err`].
    upper_a_err: Option<ArrayD<f64>>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl BatchedLinearBounds64 {
    pub(crate) fn from_f32(bounds: &BatchedLinearBounds) -> Self {
        Self {
            lower_a: bounds.lower_a().mapv(f32_to_f64_exact),
            lower_b: bounds.lower_b().mapv(f32_to_f64_exact),
            upper_a: bounds.upper_a().mapv(f32_to_f64_exact),
            upper_b: bounds.upper_b().mapv(f32_to_f64_exact),
            lower_a_err: bounds
                .lower_a_err
                .as_ref()
                .map(|error| Self::widen_coefficient_error(error, bounds.lower_a())),
            upper_a_err: bounds
                .upper_a_err
                .as_ref()
                .map(|error| Self::widen_coefficient_error(error, bounds.upper_a())),
            input_shape: bounds.input_shape().to_vec(),
            output_shape: bounds.output_shape().to_vec(),
        }
    }

    fn widen_coefficient_error(error: &ArrayD<f32>, coefficients: &ArrayD<f32>) -> ArrayD<f64> {
        if error.shape() == coefficients.shape() {
            error.mapv(nonnegative_f32_error_or_infinity)
        } else {
            ArrayD::from_elem(coefficients.raw_dim(), f64::INFINITY)
        }
    }

    /// Materialize the ACTUAL stored carrier, preserving global rows as well
    /// as legacy position-local rows. Semantic input/output metadata does not
    /// specify its storage layout: a global A may have width prod(input_shape),
    /// rather than input_shape.last(). Failed rows retain this same layout.
    pub(crate) fn into_f32(self) -> BatchedLinearBounds {
        let mut result = self.conservative_storage();
        // Nearest nominal casts and TwoSum gap certificates require the same
        // binary64 rounding/underflow premise as fan-in accumulation. Under an
        // unsupported environment the zero coefficients and infinite biases
        // above are the entire result; no source arithmetic is authoritative.
        if ny_core::require_f64_interval_proof_environment().is_err() {
            tracing::warn!("BatchedLinearBounds64 unsupported proof environment; returning conservative storage");
            return result;
        }
        if !self.storage_shapes_match() {
            tracing::warn!(
                "BatchedLinearBounds64 invalid storage shapes; returning conservative stored layout"
            );
            return result;
        }
        let in_dim = self.lower_a.shape().last().copied().unwrap_or(0);
        let num_rows = self.lower_b.len();
        let Some(lower_a) = Self::reshape_coefficients(self.lower_a, num_rows, in_dim, "lower_a")
        else {
            return result;
        };
        let Some(lower_b) = Self::reshape_bias(self.lower_b, num_rows, "lower_b") else {
            return result;
        };
        let Some(upper_a) = Self::reshape_coefficients(self.upper_a, num_rows, in_dim, "upper_a")
        else {
            return result;
        };
        let Some(upper_b) = Self::reshape_bias(self.upper_b, num_rows, "upper_b") else {
            return result;
        };
        let lower_a_err = match self.lower_a_err {
            Some(error) => {
                let Some(error) =
                    Self::reshape_coefficients(error, num_rows, in_dim, "lower_a_err")
                else {
                    return result;
                };
                Some(error)
            }
            None => None,
        };
        let upper_a_err = match self.upper_a_err {
            Some(error) => {
                let Some(error) =
                    Self::reshape_coefficients(error, num_rows, in_dim, "upper_a_err")
                else {
                    return result;
                };
                Some(error)
            }
            None => None,
        };

        if !Self::populate_result(
            &mut result,
            &lower_a,
            &lower_b,
            &upper_a,
            &upper_b,
            num_rows,
            in_dim,
        ) {
            return result;
        }
        // Coefficients multiply SIGNED inputs. Store a nearest nominal center,
        // and pay its complete actual gap plus incoming error independently on
        // each side. Exact zero planes stay absent, rather than allocating them
        // because a directed one-ULP nominal shift created artificial error.
        result.lower_a_err =
            Self::downcast_coefficient_error(&lower_a, result.lower_a(), lower_a_err.as_ref());
        result.upper_a_err =
            Self::downcast_coefficient_error(&upper_a, result.upper_a(), upper_a_err.as_ref());
        result
    }

    fn storage_shapes_match(&self) -> bool {
        self.lower_a.ndim() >= 2
            && self.lower_a.shape() == self.upper_a.shape()
            && self.lower_b.shape() == self.upper_b.shape()
            && self.lower_b.shape() == &self.lower_a.shape()[..self.lower_a.ndim() - 1]
            && self
                .lower_a_err
                .as_ref()
                .is_none_or(|error| error.shape() == self.lower_a.shape())
            && self
                .upper_a_err
                .as_ref()
                .is_none_or(|error| error.shape() == self.upper_a.shape())
    }

    /// A coherent stored-layout fallback. For valid carriers bias dimensions
    /// are preserved exactly. A poisoned bias/error layout cannot publish an
    /// invalid carrier: its bias dimensions come from the actual A prefix.
    fn conservative_storage(&self) -> BatchedLinearBounds {
        if self.lower_a.ndim() < 2 {
            return BatchedLinearBounds::conservative(
                self.input_shape.clone(),
                self.output_shape.clone(),
            );
        }
        let a_shape = self.lower_a.raw_dim();
        let b_shape = ndarray::IxDyn(&self.lower_a.shape()[..self.lower_a.ndim() - 1]);
        BatchedLinearBounds {
            lower_a: ArrayD::zeros(a_shape.clone()),
            lower_b: ArrayD::from_elem(b_shape.clone(), f32::NEG_INFINITY),
            upper_a: ArrayD::zeros(a_shape),
            upper_b: ArrayD::from_elem(b_shape, f32::INFINITY),
            input_shape: self.input_shape.clone(),
            output_shape: self.output_shape.clone(),
            lower_a_err: None,
            upper_a_err: None,
        }
    }

    /// Reconstitute an f32 coefficient certificate after the f64 sidecar is
    /// downcast. The certificate includes both carried fan-in error and the
    /// exact gap between the f64 coefficient and its stored f32 value.
    fn downcast_coefficient_error(
        source: &Array2<f64>,
        stored: &ArrayD<f32>,
        carried: Option<&Array2<f64>>,
    ) -> Option<ArrayD<f32>> {
        let mut result: Option<ArrayD<f32>> = None;
        let mut carried_values = carried.map(|error| error.iter());
        for (index, (&stored_value, &source_value)) in stored.iter().zip(source.iter()).enumerate()
        {
            let carried_value = carried_values
                .as_mut()
                .and_then(|values| values.next())
                .copied()
                .unwrap_or(0.0);
            let stored_value = f32_to_f64_exact(stored_value);
            let cast_gap = if stored_value.is_finite() && source_value.is_finite() {
                // two_sum accounts for a possibly inexact f64 subtraction.
                let (gap, residual) = two_sum(stored_value, -source_value);
                Self::add_nonnegative_f64_up(gap.abs(), residual.abs())
            } else {
                f64::INFINITY
            };
            let error = Self::err_to_f32(Self::add_nonnegative_f64_up(carried_value, cast_gap));
            if error != 0.0 {
                // A full dense uncertainty plane is needed by the current
                // carrier only after the first nonzero certificate. Its shape
                // is exactly the stored A shape; err_to_f32 has already paid
                // upward and sanitized poison to +infinity, without a second
                // map/copy through set_coeff_err.
                let plane = result.get_or_insert_with(|| ArrayD::zeros(stored.raw_dim()));
                plane
                    .as_slice_mut()
                    .expect("fresh coefficient-error plane is contiguous")[index] = error;
            }
        }
        result
    }

    pub(crate) fn accumulate(&mut self, new_bounds: &BatchedLinearBounds) {
        if ny_core::require_f64_interval_proof_environment().is_err() {
            tracing::warn!("BatchedLinearBounds64 unsupported proof environment; widening accumulator to infinities");
            Self::widen_to_infinities(self);
            return;
        }
        if new_bounds
            .lower_a_err
            .as_ref()
            .is_some_and(|error| error.shape() != new_bounds.lower_a().shape())
            || new_bounds
                .upper_a_err
                .as_ref()
                .is_some_and(|error| error.shape() != new_bounds.upper_a().shape())
        {
            tracing::warn!(
                "BatchedLinearBounds64 incoming coefficient-error shape mismatch; widening accumulator to infinities"
            );
            Self::widen_to_infinities(self);
            return;
        }

        if self.matches(new_bounds) {
            let lower_roundoff =
                Self::accumulate_coeff_array(&mut self.lower_a, new_bounds.lower_a());
            Self::accumulate_array(&mut self.lower_b, new_bounds.lower_b(), true);
            let upper_roundoff =
                Self::accumulate_coeff_array(&mut self.upper_a, new_bounds.upper_a());
            Self::accumulate_array(&mut self.upper_b, new_bounds.upper_b(), false);
            let lower_shape = self.lower_a.raw_dim();
            let upper_shape = self.upper_a.raw_dim();
            Self::accumulate_err(
                &mut self.lower_a_err,
                new_bounds.lower_a_err.as_ref(),
                &lower_roundoff,
                lower_shape,
            );
            Self::accumulate_err(
                &mut self.upper_a_err,
                new_bounds.upper_a_err.as_ref(),
                &upper_roundoff,
                upper_shape,
            );
            return;
        }

        if let Some((lower_a, lower_b, upper_a, upper_b)) =
            self.reshape_compatible_views(new_bounds)
        {
            tracing::debug!(
                existing_lower_a = ?self.lower_a.shape(),
                new_lower_a = ?new_bounds.lower_a().shape(),
                existing_output_shape = ?self.output_shape,
                new_output_shape = ?new_bounds.output_shape(),
                "BatchedLinearBounds64 reshaping same-count contribution before accumulation"
            );
            let lower_roundoff = Self::accumulate_coeff_array(&mut self.lower_a, &lower_a);
            Self::accumulate_array(&mut self.lower_b, &lower_b, true);
            let upper_roundoff = Self::accumulate_coeff_array(&mut self.upper_a, &upper_a);
            Self::accumulate_array(&mut self.upper_b, &upper_b, false);
            let lower_shape = self.lower_a.raw_dim();
            let upper_shape = self.upper_a.raw_dim();
            Self::accumulate_err(
                &mut self.lower_a_err,
                new_bounds.lower_a_err.as_ref(),
                &lower_roundoff,
                lower_shape,
            );
            Self::accumulate_err(
                &mut self.upper_a_err,
                new_bounds.upper_a_err.as_ref(),
                &upper_roundoff,
                upper_shape,
            );
            return;
        }

        tracing::warn!(
            existing_lower_a = ?self.lower_a.shape(),
            new_lower_a = ?new_bounds.lower_a().shape(),
            existing_lower_b = ?self.lower_b.shape(),
            new_lower_b = ?new_bounds.lower_b().shape(),
            existing_input_shape = ?self.input_shape,
            new_input_shape = ?new_bounds.input_shape(),
            existing_output_shape = ?self.output_shape,
            new_output_shape = ?new_bounds.output_shape(),
            "BatchedLinearBounds64 shape mismatch; widening accumulator to infinities"
        );
        Self::widen_to_infinities(self);
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        (self.lower_a.len()
            + self.lower_b.len()
            + self.upper_a.len()
            + self.upper_b.len()
            + self.lower_a_err.as_ref().map_or(0, ArrayD::len)
            + self.upper_a_err.as_ref().map_or(0, ArrayD::len))
            * size_of::<f64>()
    }

    fn matches(&self, new_bounds: &BatchedLinearBounds) -> bool {
        self.lower_a.shape() == new_bounds.lower_a().shape()
            && self.lower_b.shape() == new_bounds.lower_b().shape()
            && self.upper_a.shape() == new_bounds.upper_a().shape()
            && self.upper_b.shape() == new_bounds.upper_b().shape()
            && self.input_shape == new_bounds.input_shape()
            && self.output_shape == new_bounds.output_shape()
    }

    fn reshape_compatible_views<'a>(
        &self,
        new_bounds: &'a BatchedLinearBounds,
    ) -> Option<BoundsViewQuad<'a>> {
        if self.lower_a.len() != new_bounds.lower_a().len()
            || self.lower_b.len() != new_bounds.lower_b().len()
            || self.upper_a.len() != new_bounds.upper_a().len()
            || self.upper_b.len() != new_bounds.upper_b().len()
        {
            return None;
        }

        let input_elems = checked_shape_product(&self.input_shape)?;
        let new_input_elems = checked_shape_product(new_bounds.input_shape())?;
        let output_elems = checked_shape_product(&self.output_shape)?;
        let new_output_elems = checked_shape_product(new_bounds.output_shape())?;
        if input_elems != new_input_elems || output_elems != new_output_elems {
            return None;
        }

        let lower_a = new_bounds
            .lower_a()
            .view()
            .into_shape_with_order(self.lower_a.raw_dim())
            .ok()?;
        let lower_b = new_bounds
            .lower_b()
            .view()
            .into_shape_with_order(self.lower_b.raw_dim())
            .ok()?;
        let upper_a = new_bounds
            .upper_a()
            .view()
            .into_shape_with_order(self.upper_a.raw_dim())
            .ok()?;
        let upper_b = new_bounds
            .upper_b()
            .view()
            .into_shape_with_order(self.upper_b.raw_dim())
            .ok()?;
        Some((lower_a, lower_b, upper_a, upper_b))
    }

    fn widen_to_infinities(existing: &mut Self) {
        existing.lower_a = Array::from_elem(existing.lower_a.raw_dim(), f64::NEG_INFINITY);
        existing.lower_b = Array::from_elem(existing.lower_b.raw_dim(), f64::NEG_INFINITY);
        existing.upper_a = Array::from_elem(existing.upper_a.raw_dim(), f64::INFINITY);
        existing.upper_b = Array::from_elem(existing.upper_b.raw_dim(), f64::INFINITY);
        existing.lower_a_err = None;
        existing.upper_a_err = None;
    }

    fn accumulate_coeff_array<D, S>(
        existing: &mut ArrayD<f64>,
        new: &ArrayBase<S, D>,
    ) -> ArrayD<f64>
    where
        D: Dimension,
        S: Data<Elem = f32>,
    {
        let mut roundoff = ArrayD::zeros(existing.raw_dim());
        for ((existing_value, &new_value), roundoff_value) in
            existing.iter_mut().zip(new.iter()).zip(roundoff.iter_mut())
        {
            if existing_value.is_nan() || new_value.is_nan() {
                *existing_value = f64::NAN;
                *roundoff_value = 0.0;
                continue;
            }
            let (sum, residual) = two_sum(*existing_value, f32_to_f64_exact(new_value));
            *existing_value = sum;
            *roundoff_value = if sum.is_finite() { residual.abs() } else { 0.0 };
        }
        roundoff
    }

    fn accumulate_err(
        existing_err: &mut Option<ArrayD<f64>>,
        new_err: Option<&ArrayD<f32>>,
        roundoff: &ArrayD<f64>,
        expected_shape: ndarray::IxDyn,
    ) {
        let roundoff_has = roundoff.iter().any(|&value| value != 0.0);
        if existing_err.is_none() && new_err.is_none() && !roundoff_has {
            return;
        }
        if existing_err
            .as_ref()
            .is_some_and(|error| error.shape() != expected_shape.slice())
            || new_err.is_some_and(|error| error.len() != roundoff.len())
        {
            *existing_err = Some(ArrayD::from_elem(expected_shape, f64::INFINITY));
            return;
        }
        let accumulator =
            existing_err.get_or_insert_with(|| ArrayD::<f64>::zeros(expected_shape.clone()));
        for (value, &roundoff_value) in accumulator.iter_mut().zip(roundoff.iter()) {
            *value = Self::add_nonnegative_f64_up(*value, roundoff_value);
        }
        if let Some(new_error) = new_err {
            for (value, &incoming) in accumulator.iter_mut().zip(new_error.iter()) {
                *value = Self::add_nonnegative_f64_up(
                    *value,
                    nonnegative_f32_error_or_infinity(incoming),
                );
            }
        }
    }

    #[inline]
    fn add_nonnegative_f64_up(a: f64, b: f64) -> f64 {
        if !a.is_finite() || !b.is_finite() || a < 0.0 || b < 0.0 {
            return f64::INFINITY;
        }
        let (sum, residual) = two_sum(a, b);
        if !sum.is_finite() {
            f64::INFINITY
        } else if residual > 0.0 {
            next_up_f64(sum)
        } else {
            sum
        }
    }

    #[inline]
    fn err_to_f32(error: f64) -> f32 {
        if !error.is_finite() || error < 0.0 {
            return f32::INFINITY;
        }
        // A flushable subnormal error would lose its certificate on a later
        // binary32 kernel. Exact zero stays zero; every positive error is
        // published upward at a normal endpoint (or conservative +infinity).
        // `error` already encloses the carried uncertainty, the full cast
        // gap and every arithmetic residual. Directed publication pays the
        // final storage gap; an additional unconditional step would widen
        // exact incoming f32 errors on every otherwise unchanged roundtrip.
        f64_to_f32_up_no_subnormal(error)
    }

    fn accumulate_array<D, S>(existing: &mut Array<f64, D>, new: &ArrayBase<S, D>, is_lower: bool)
    where
        D: Dimension,
        S: Data<Elem = f32>,
    {
        let nan_fallback = if is_lower {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        Zip::from(existing)
            .and(new)
            .for_each(|existing_value, &new_value| {
                if existing_value.is_nan() || new_value.is_nan() {
                    *existing_value = nan_fallback;
                    return;
                }
                let (sum, residual) = two_sum(*existing_value, f32_to_f64_exact(new_value));
                *existing_value = if sum.is_nan() {
                    nan_fallback
                } else if !sum.is_finite() {
                    sum
                } else if is_lower && residual < 0.0 {
                    next_down_f64(sum)
                } else if !is_lower && residual > 0.0 {
                    next_up_f64(sum)
                } else {
                    sum
                };
            });
    }

    fn try_downcast_row(
        lower_a: &Array2<f64>,
        lower_b: &Array1<f64>,
        upper_a: &Array2<f64>,
        upper_b: &Array1<f64>,
        row: usize,
        num_inputs: usize,
    ) -> Option<(Vec<f32>, f32, Vec<f32>, f32)> {
        let lower_bias = Self::downcast_lower_bias(lower_b[row])?;
        let upper_bias = Self::downcast_upper_bias(upper_b[row])?;
        let mut lower_row = Vec::with_capacity(num_inputs);
        let mut upper_row = Vec::with_capacity(num_inputs);

        for col in 0..num_inputs {
            lower_row.push(Self::downcast_coeff(lower_a[[row, col]])?);
            upper_row.push(Self::downcast_coeff(upper_a[[row, col]])?);
        }

        Some((lower_row, lower_bias, upper_row, upper_bias))
    }

    fn downcast_coeff(value: f64) -> Option<f32> {
        if !value.is_finite() {
            return None;
        }

        let cast = value as f32;
        if !cast.is_finite() {
            return None;
        }

        // Flushable nominal centers are deliberately zero, with their FULL
        // original gap paid by downcast_coefficient_error. All nonzero emitted
        // uncertainty is normal/upward, so a later binary32 consumer cannot
        // erase the certificate by flushing a subnormal error.
        Some(
            if cast.to_bits() & 0x7f80_0000 == 0 && cast.to_bits() & 0x007f_ffff != 0 {
                0.0
            } else {
                cast
            },
        )
    }

    fn downcast_lower_bias(value: f64) -> Option<f32> {
        if value == f64::NEG_INFINITY {
            return Some(f32::NEG_INFINITY);
        }
        let cast = value as f32;
        (value.is_finite() && cast.is_finite()).then(|| next_down_f32(cast))
    }

    fn downcast_upper_bias(value: f64) -> Option<f32> {
        if value == f64::INFINITY {
            return Some(f32::INFINITY);
        }
        let cast = value as f32;
        (value.is_finite() && cast.is_finite()).then(|| next_up_f32(cast))
    }

    fn reshape_coefficients(
        array: ArrayD<f64>,
        num_rows: usize,
        in_dim: usize,
        label: &'static str,
    ) -> Option<Array2<f64>> {
        let shape = array.shape().to_vec();
        let Ok(reshaped) = array.into_shape_with_order((num_rows, in_dim)) else {
            tracing::warn!(
                shape = ?shape,
                "BatchedLinearBounds64 {label} reshape failed; returning conservative bounds"
            );
            return None;
        };
        Some(reshaped)
    }

    fn reshape_bias(
        array: ArrayD<f64>,
        num_rows: usize,
        label: &'static str,
    ) -> Option<Array1<f64>> {
        let shape = array.shape().to_vec();
        let Ok(reshaped) = array.into_shape_with_order(num_rows) else {
            tracing::warn!(
                shape = ?shape,
                "BatchedLinearBounds64 {label} reshape failed; returning conservative bounds"
            );
            return None;
        };
        Some(reshaped)
    }

    fn populate_result(
        result: &mut BatchedLinearBounds,
        lower_a: &Array2<f64>,
        lower_b: &Array1<f64>,
        upper_a: &Array2<f64>,
        upper_b: &Array1<f64>,
        num_rows: usize,
        in_dim: usize,
    ) -> bool {
        let Ok(mut result_lower_a) = result
            .lower_a
            .view_mut()
            .into_shape_with_order((num_rows, in_dim))
        else {
            tracing::warn!(
                "BatchedLinearBounds64 result lower_a reshape failed; returning conservative bounds"
            );
            return false;
        };
        let Ok(mut result_lower_b) = result.lower_b.view_mut().into_shape_with_order(num_rows)
        else {
            tracing::warn!(
                "BatchedLinearBounds64 result lower_b reshape failed; returning conservative bounds"
            );
            return false;
        };
        let Ok(mut result_upper_a) = result
            .upper_a
            .view_mut()
            .into_shape_with_order((num_rows, in_dim))
        else {
            tracing::warn!(
                "BatchedLinearBounds64 result upper_a reshape failed; returning conservative bounds"
            );
            return false;
        };
        let Ok(mut result_upper_b) = result.upper_b.view_mut().into_shape_with_order(num_rows)
        else {
            tracing::warn!(
                "BatchedLinearBounds64 result upper_b reshape failed; returning conservative bounds"
            );
            return false;
        };

        for row in 0..num_rows {
            let Some((lower_row, lower_bias, upper_row, upper_bias)) =
                Self::try_downcast_row(lower_a, lower_b, upper_a, upper_b, row, in_dim)
            else {
                tracing::warn!(
                    row,
                    "BatchedLinearBounds64 row downcast failed; returning conservative row"
                );
                continue;
            };

            for (col, value) in lower_row.into_iter().enumerate() {
                result_lower_a[[row, col]] = value;
            }
            result_lower_b[row] = lower_bias;
            for (col, value) in upper_row.into_iter().enumerate() {
                result_upper_a[[row, col]] = value;
            }
            result_upper_b[row] = upper_bias;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;
    use ny_tensor::{next_down_f32, next_up_f32};

    fn zero_bounds(input_shape: &[usize], output_shape: &[usize]) -> BatchedLinearBounds {
        let in_dim = *input_shape.last().expect("input shape should be non-empty");
        let out_dim = *output_shape
            .last()
            .expect("output shape should be non-empty");
        let mut a_shape = output_shape[..output_shape.len() - 1].to_vec();
        a_shape.push(out_dim);
        a_shape.push(in_dim);

        BatchedLinearBounds::from_parts_unchecked(
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(output_shape)),
            ArrayD::zeros(IxDyn(&a_shape)),
            ArrayD::zeros(IxDyn(output_shape)),
            input_shape.to_vec(),
            output_shape.to_vec(),
        )
    }

    fn filled_bounds(
        input_shape: &[usize],
        output_shape: &[usize],
        start: f32,
    ) -> BatchedLinearBounds {
        let in_dim = *input_shape.last().expect("input shape should be non-empty");
        let out_dim = *output_shape
            .last()
            .expect("output shape should be non-empty");
        let mut a_shape = output_shape[..output_shape.len() - 1].to_vec();
        a_shape.push(out_dim);
        a_shape.push(in_dim);
        let a_len: usize = a_shape.iter().product();
        let b_len: usize = output_shape.iter().product();

        let lower_a = ArrayD::from_shape_vec(
            IxDyn(&a_shape),
            (0..a_len).map(|idx| start + idx as f32).collect(),
        )
        .expect("lower_a shape should be valid");
        let lower_b = ArrayD::from_shape_vec(
            IxDyn(output_shape),
            (0..b_len).map(|idx| start + 1000.0 + idx as f32).collect(),
        )
        .expect("lower_b shape should be valid");
        let upper_a = ArrayD::from_shape_vec(
            IxDyn(&a_shape),
            (0..a_len).map(|idx| start + 2000.0 + idx as f32).collect(),
        )
        .expect("upper_a shape should be valid");
        let upper_b = ArrayD::from_shape_vec(
            IxDyn(output_shape),
            (0..b_len).map(|idx| start + 3000.0 + idx as f32).collect(),
        )
        .expect("upper_b shape should be valid");

        BatchedLinearBounds::from_parts_unchecked(
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            input_shape.to_vec(),
            output_shape.to_vec(),
        )
    }

    #[test]
    fn test_accumulate_reshape_compatible_bounds_preserves_values_4243() {
        let input_shape = [1, 2, 4, 8];
        let existing_output_shape = [1, 4, 16];
        let reshaped_output_shape = [1, 2, 4, 8];

        let existing = zero_bounds(&input_shape, &existing_output_shape);
        let new_bounds = filled_bounds(&input_shape, &reshaped_output_shape, 10.0);

        let mut accumulator = BatchedLinearBounds64::from_f32(&existing);
        accumulator.accumulate(&new_bounds);
        let merged = accumulator.into_f32();

        assert_eq!(merged.output_shape(), &existing_output_shape);
        assert!(
            merged
                .lower_a()
                .iter()
                .chain(merged.lower_b().iter())
                .chain(merged.upper_a().iter())
                .chain(merged.upper_b().iter())
                .all(|value| value.is_finite()),
            "reshape-compatible merge should stay finite"
        );

        let expected_lower_a = new_bounds
            .lower_a()
            .view()
            .into_shape_with_order(IxDyn(merged.lower_a().shape()))
            .expect("lower_a reshape should succeed");
        let expected_lower_b = new_bounds
            .lower_b()
            .view()
            .into_shape_with_order(IxDyn(merged.lower_b().shape()))
            .expect("lower_b reshape should succeed");
        let expected_upper_a = new_bounds
            .upper_a()
            .view()
            .into_shape_with_order(IxDyn(merged.upper_a().shape()))
            .expect("upper_a reshape should succeed");
        let expected_upper_b = new_bounds
            .upper_b()
            .view()
            .into_shape_with_order(IxDyn(merged.upper_b().shape()))
            .expect("upper_b reshape should succeed");

        // The signed nominal coefficients are exact f32 sums. Publishing them
        // bit for bit needs no uncertainty plane; a directed coefficient nudge
        // would require paying a gap even for this exact sum.
        assert!(
            !merged.has_coeff_err(),
            "an exact fan-in sum needs no plane"
        );
        for (actual, expected) in merged.lower_a().iter().zip(expected_lower_a.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        for (actual, expected) in merged.lower_b().iter().zip(expected_lower_b.iter()) {
            assert_eq!(*actual, next_down_f32(*expected));
        }
        for (actual, expected) in merged.upper_a().iter().zip(expected_upper_a.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        for (actual, expected) in merged.upper_b().iter().zip(expected_upper_b.iter()) {
            assert_eq!(*actual, next_up_f32(*expected));
        }
    }

    fn global_bounds() -> BatchedLinearBounds {
        let a = ndarray::array![[1.0, -0.5, 0.25, -2.0], [-0.125, 3.0, -1.0, 0.0]].into_dyn();
        let b = ndarray::array![-0.25, 0.125].into_dyn();
        BatchedLinearBounds::new(a.clone(), b.clone(), a, b, vec![1, 2, 2], vec![1, 2]).unwrap()
    }
    fn coherent(bounds: &BatchedLinearBounds) {
        bounds.validate_internal_shapes().unwrap();
        bounds.validate_no_nan().unwrap();
        assert_eq!(bounds.input_shape(), &[1, 2, 2]);
        assert_eq!(bounds.output_shape(), &[1, 2]);
    }
    fn rational_bits(sign: bool, mantissa: u64, power: i32) -> num_rational::BigRational {
        use num_bigint::BigInt;
        use num_rational::BigRational as Q;
        let mut n = BigInt::from(mantissa);
        if sign {
            n = -n;
        }
        if power >= 0 {
            Q::from_integer(n << power as usize)
        } else {
            Q::new(n, BigInt::from(1) << (-power) as usize)
        }
    }
    pub(super) fn q32(value: f32) -> num_rational::BigRational {
        let bits = value.to_bits();
        let exponent = (bits >> 23) & 255;
        assert_ne!(exponent, 255);
        let fraction = bits & 0x7f_ffff;
        let (mantissa, power) = if exponent == 0 {
            (u64::from(fraction), -149)
        } else {
            (u64::from(fraction | (1 << 23)), exponent as i32 - 127 - 23)
        };
        rational_bits(bits >> 31 != 0, mantissa, power)
    }
    pub(super) fn q64(value: f64) -> num_rational::BigRational {
        let bits = value.to_bits();
        let exponent = (bits >> 52) & 2047;
        assert_ne!(exponent, 2047);
        let fraction = bits & 0x000f_ffff_ffff_ffff;
        let (mantissa, power) = if exponent == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1 << 52), exponent as i32 - 1023 - 52)
        };
        rational_bits(bits >> 63 != 0, mantissa, power)
    }

    #[test]
    fn batched_f64_global_roundtrip_keeps_actual_rows_errors_and_signed_affine_bounds() {
        use num_traits::{Signed, Zero};
        let mut original = global_bounds();
        original.set_coeff_err(
            ndarray::array![[0.125, 0.0, 0.25, 0.0625], [0.0, 0.125, 0.0, 0.5]].into_dyn(),
            ndarray::array![[0.25, 0.0625, 0.0, 0.0], [0.5, 0.0, 0.125, 0.0]].into_dyn(),
        );
        let sidecar = BatchedLinearBounds64::from_f32(&original);
        let source = sidecar.clone();
        let restored = sidecar.into_f32();
        coherent(&restored);
        // The old metadata-derived constructor used [1,2,2] and lost half of
        // these global columns. Actual storage and its error planes are [2,4].
        assert_eq!(restored.lower_a().shape(), &[2, 4]);
        assert_eq!(restored.upper_a().shape(), &[2, 4]);
        assert_eq!(restored.lower_b().shape(), &[2]);
        assert_eq!(restored.upper_b().shape(), &[2]);
        for (source, stored, carried, error) in [
            (
                &source.lower_a,
                restored.lower_a(),
                source.lower_a_err.as_ref().unwrap(),
                restored.lower_a_err.as_ref().unwrap(),
            ),
            (
                &source.upper_a,
                restored.upper_a(),
                source.upper_a_err.as_ref().unwrap(),
                restored.upper_a_err.as_ref().unwrap(),
            ),
        ] {
            assert_eq!(error.shape(), &[2, 4]);
            for (((&exact, &stored), &carried), &error) in
                source.iter().zip(stored).zip(carried).zip(error)
            {
                assert!(q32(error) >= q64(carried) + (q32(stored) - q64(exact)).abs());
            }
        }
        for point in [[-1.0, 2.0, -3.0, 4.0], [0.125, -0.25, 8.0, -16.0]] {
            for row in 0..2 {
                let mut old_lower = q32(original.lower_b()[row]);
                let mut old_upper = q32(original.upper_b()[row]);
                let mut new_lower = q32(restored.lower_b()[row]);
                let mut new_upper = q32(restored.upper_b()[row]);
                for (column, &x) in point.iter().enumerate() {
                    let x = q32(x);
                    old_lower += q32(original.lower_a()[[row, column]]) * &x
                        - q32(original.lower_a_err.as_ref().unwrap()[[row, column]]) * x.abs();
                    old_upper += q32(original.upper_a()[[row, column]]) * &x
                        + q32(original.upper_a_err.as_ref().unwrap()[[row, column]]) * x.abs();
                    new_lower += q32(restored.lower_a()[[row, column]]) * &x
                        - q32(restored.lower_a_err.as_ref().unwrap()[[row, column]]) * x.abs();
                    new_upper += q32(restored.upper_a()[[row, column]]) * &x
                        + q32(restored.upper_a_err.as_ref().unwrap()[[row, column]]) * x.abs();
                }
                assert!(new_lower <= old_lower);
                assert!(new_upper >= old_upper);
            }
        }
        assert!(!restored.lower_a().iter().all(|&a| a == 0.0));
        assert!(restored
            .lower_b()
            .iter()
            .chain(restored.upper_b())
            .all(|b| b.is_finite()));
        assert!(!q32(restored.lower_a()[[0, 0]]).is_zero());
    }

    #[test]
    fn batched_f64_global_inexact_negative_input_requires_new_cast_error_plane() {
        use num_traits::Signed;
        let original = BatchedLinearBounds::new(
            ndarray::array![[1.0]].into_dyn(),
            ndarray::array![0.0].into_dyn(),
            ndarray::array![[1.0]].into_dyn(),
            ndarray::array![0.0].into_dyn(),
            vec![1],
            vec![1],
        )
        .unwrap();
        assert!(!original.has_coeff_err());
        let mut source = BatchedLinearBounds64::from_f32(&original);
        source.lower_a[[0, 0]] = 1.0 + 2f64.powi(-25);
        source.upper_a[[0, 0]] = 1.0 - 2f64.powi(-25);
        let exact_lower = q64(source.lower_a[[0, 0]]) * q32(-1.0);
        let exact_upper = q64(source.upper_a[[0, 0]]) * q32(-1.0);
        let restored = source.into_f32();
        restored.validate_internal_shapes().unwrap();
        restored.validate_no_nan().unwrap();
        // Nearest nominal coefficients are centers, whose independently paid
        // gap is still essential on signed inputs.
        assert_eq!(restored.lower_a()[[0, 0]].to_bits(), 1.0f32.to_bits());
        assert_eq!(restored.upper_a()[[0, 0]].to_bits(), 1.0f32.to_bits());
        let x = q32(-1.0);
        let nominal_lower = q32(restored.lower_a()[[0, 0]]) * &x + q32(restored.lower_b()[0]);
        let nominal_upper = q32(restored.upper_a()[[0, 0]]) * &x + q32(restored.upper_b()[0]);
        // Concrete exclusion: rounding to these nominal centers loses a real
        // coefficient contribution on negative x. Bias widening cannot pay it.
        assert!(nominal_lower > exact_lower);
        assert!(nominal_upper < exact_upper);
        let lower_error = q32(restored.lower_a_err.as_ref().unwrap()[[0, 0]]);
        let upper_error = q32(restored.upper_a_err.as_ref().unwrap()[[0, 0]]);
        assert!(nominal_lower - lower_error * x.abs() <= exact_lower);
        assert!(nominal_upper + upper_error * x.abs() >= exact_upper);
    }

    #[test]
    fn batched_f64_global_conservative_rows_and_malformed_sidecar_keep_storage() {
        for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
            let mut source = BatchedLinearBounds64::from_f32(&global_bounds());
            source.lower_a[[0, 1]] = poison;
            let restored = source.into_f32();
            coherent(&restored);
            assert_eq!(restored.lower_a().shape(), &[2, 4]);
            assert_eq!(restored.lower_b().shape(), &[2]);
            for column in 0..4 {
                assert_eq!(restored.lower_a()[[0, column]], 0.0);
                assert_eq!(restored.upper_a()[[0, column]], 0.0);
            }
            assert_eq!(restored.lower_b()[0], f32::NEG_INFINITY);
            assert_eq!(restored.upper_b()[0], f32::INFINITY);
            assert!(restored.lower_b()[1].is_finite() && restored.upper_b()[1].is_finite());
            assert_ne!(restored.lower_a()[[1, 1]], 0.0);
        }
        for wrong_lower in [true, false] {
            let mut source = BatchedLinearBounds64::from_f32(&global_bounds());
            if wrong_lower {
                source.lower_b[0] = f64::INFINITY;
            } else {
                source.upper_b[0] = f64::NEG_INFINITY;
            }
            let restored = source.into_f32();
            coherent(&restored);
            assert_eq!(restored.lower_a().shape(), &[2, 4]);
            assert_eq!(restored.lower_b()[0], f32::NEG_INFINITY);
            assert_eq!(restored.upper_b()[0], f32::INFINITY);
            assert!(restored.lower_b()[1].is_finite());
        }
        for error_plane in [true, false] {
            let mut source = BatchedLinearBounds64::from_f32(&global_bounds());
            if error_plane {
                source.lower_a_err = Some(ArrayD::zeros(IxDyn(&[2, 2])));
            } else {
                // Same count with incompatible authored storage must fail
                // closed instead of silently accepting an invalid sidecar.
                source.upper_a = ArrayD::zeros(IxDyn(&[1, 8]));
            }
            let restored = source.into_f32();
            coherent(&restored);
            assert_eq!(restored.lower_a().shape(), &[2, 4]);
            assert_eq!(restored.lower_b().shape(), &[2]);
            assert!(restored
                .lower_a()
                .iter()
                .chain(restored.upper_a())
                .all(|&a| a == 0.0));
            assert!(restored.lower_b().iter().all(|&b| b == f32::NEG_INFINITY));
            assert!(restored.upper_b().iter().all(|&b| b == f32::INFINITY));
        }
    }

    #[test]
    fn batched_f64_global_legacy_local_and_singleton_global_layouts_remain_distinct() {
        let local = filled_bounds(&[1, 2, 2], &[1, 2], 1.0);
        let restored = BatchedLinearBounds64::from_f32(&local).into_f32();
        coherent(&restored);
        assert_eq!(restored.lower_a().shape(), &[1, 2, 2]);
        assert_eq!(restored.lower_b().shape(), &[1, 2]);
        for (old, new) in local.lower_a().iter().zip(restored.lower_a()) {
            assert_eq!(new.to_bits(), old.to_bits());
        }
        for (old, new) in local.upper_a().iter().zip(restored.upper_a()) {
            assert_eq!(new.to_bits(), old.to_bits());
        }
        let global = global_bounds();
        let singleton = BatchedLinearBounds::new(
            global
                .lower_a()
                .clone()
                .into_shape_with_order(IxDyn(&[1, 2, 4]))
                .unwrap(),
            global
                .lower_b()
                .clone()
                .into_shape_with_order(IxDyn(&[1, 2]))
                .unwrap(),
            global
                .upper_a()
                .clone()
                .into_shape_with_order(IxDyn(&[1, 2, 4]))
                .unwrap(),
            global
                .upper_b()
                .clone()
                .into_shape_with_order(IxDyn(&[1, 2]))
                .unwrap(),
            vec![1, 2, 2],
            vec![1, 2],
        )
        .unwrap();
        let restored_singleton = BatchedLinearBounds64::from_f32(&singleton).into_f32();
        let restored_flat = BatchedLinearBounds64::from_f32(&global).into_f32();
        coherent(&restored_singleton);
        assert_eq!(restored_singleton.lower_a().shape(), &[1, 2, 4]);
        assert_eq!(restored_singleton.lower_b().shape(), &[1, 2]);
        for (a, b) in restored_flat
            .lower_a()
            .iter()
            .zip(restored_singleton.lower_a())
        {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert!(restored_flat.lower_a_err.is_none() && restored_flat.upper_a_err.is_none());
        assert!(
            restored_singleton.lower_a_err.is_none() && restored_singleton.upper_a_err.is_none()
        );
        // Ordinary conservative metadata-local carriers remain metadata-local.
        let conservative = BatchedLinearBounds::conservative(vec![1, 2, 2], vec![1, 2]);
        let restored = BatchedLinearBounds64::from_f32(&conservative).into_f32();
        coherent(&restored);
        assert_eq!(restored.lower_a().shape(), &[1, 2, 2]);
        assert_eq!(restored.lower_b().shape(), &[1, 2]);
        assert!(restored.lower_b().iter().all(|&b| b == f32::NEG_INFINITY));
        assert!(restored.upper_b().iter().all(|&b| b == f32::INFINITY));
    }

    #[test]
    fn batched_f64_global_subnormal_originals_and_poisoned_error_bits_are_preserved() {
        use num_traits::{Signed, Zero};
        let tiny = f32::from_bits(1);
        let mut original = global_bounds();
        original.lower_a[[0, 0]] = tiny;
        original.upper_a[[0, 0]] = -tiny;
        original.lower_a_err = Some(ArrayD::from_elem(IxDyn(&[2, 4]), tiny));
        original.upper_a_err = Some(ArrayD::from_elem(IxDyn(&[2, 4]), -tiny));
        let mut source = BatchedLinearBounds64::from_f32(&original);
        assert_eq!(q64(source.lower_a[[0, 0]]), q32(tiny));
        assert_eq!(q64(source.upper_a[[0, 0]]), -q32(tiny));
        assert_eq!(q64(source.lower_a_err.as_ref().unwrap()[[0, 0]]), q32(tiny));
        assert!(source
            .upper_a_err
            .as_ref()
            .unwrap()
            .iter()
            .all(|e| *e == f64::INFINITY));
        // The incoming exact subnormal error is also retained during a fan-in,
        // rather than read as signed zero by an ordinary hardware conversion.
        source.accumulate(&original);
        assert!(q64(source.lower_a_err.as_ref().unwrap()[[0, 0]]) >= q32(tiny) * q32(2.0));
        let exact = q64(source.lower_a[[0, 0]]);
        let carried = q64(source.lower_a_err.as_ref().unwrap()[[0, 0]]);
        let restored = source.into_f32();
        coherent(&restored);
        assert_eq!(restored.lower_a().shape(), &[2, 4]);
        let stored = q32(restored.lower_a()[[0, 0]]);
        let error = restored.lower_a_err.as_ref().unwrap()[[0, 0]];
        assert!(q32(error) >= carried + (stored - exact).abs());
        assert!(error >= f32::MIN_POSITIVE);
        assert_ne!(q32(error), num_rational::BigRational::zero());
        assert!(restored
            .upper_a_err
            .as_ref()
            .unwrap()
            .iter()
            .all(|e| *e == f32::INFINITY));
        for error in restored.lower_a_err.as_ref().unwrap().iter() {
            assert!(error.to_bits() & 0x7f80_0000 != 0 || *error == 0.0);
        }
    }
}

#[cfg(test)]
mod nearest_tests;
