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
            lower_a: bounds.lower_a().mapv(f64::from),
            lower_b: bounds.lower_b().mapv(f64::from),
            upper_a: bounds.upper_a().mapv(f64::from),
            upper_b: bounds.upper_b().mapv(f64::from),
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
            error.mapv(|value| {
                if value.is_finite() && value >= 0.0 {
                    f64::from(value)
                } else {
                    f64::INFINITY
                }
            })
        } else {
            ArrayD::from_elem(coefficients.raw_dim(), f64::INFINITY)
        }
    }

    pub(crate) fn into_f32(self) -> BatchedLinearBounds {
        let in_dim = self.lower_a.shape().last().copied().unwrap_or(0);
        let num_rows = self.lower_b.len();
        let input_shape = self.input_shape;
        let output_shape = self.output_shape;
        let has_err = self.lower_a_err.is_some() || self.upper_a_err.is_some();
        let Some(lower_a) = Self::reshape_coefficients(self.lower_a, num_rows, in_dim, "lower_a")
        else {
            return BatchedLinearBounds::conservative(input_shape, output_shape);
        };
        let Some(lower_b) = Self::reshape_bias(self.lower_b, num_rows, "lower_b") else {
            return BatchedLinearBounds::conservative(input_shape, output_shape);
        };
        let Some(upper_a) = Self::reshape_coefficients(self.upper_a, num_rows, in_dim, "upper_a")
        else {
            return BatchedLinearBounds::conservative(input_shape, output_shape);
        };
        let Some(upper_b) = Self::reshape_bias(self.upper_b, num_rows, "upper_b") else {
            return BatchedLinearBounds::conservative(input_shape, output_shape);
        };
        let lower_a_err = match self.lower_a_err {
            Some(error) => {
                let Some(error) =
                    Self::reshape_coefficients(error, num_rows, in_dim, "lower_a_err")
                else {
                    return BatchedLinearBounds::conservative(input_shape, output_shape);
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
                    return BatchedLinearBounds::conservative(input_shape, output_shape);
                };
                Some(error)
            }
            None => None,
        };

        let mut result =
            BatchedLinearBounds::conservative(input_shape.clone(), output_shape.clone());
        if Self::populate_result(
            &mut result,
            &lower_a,
            &lower_b,
            &upper_a,
            &upper_b,
            num_rows,
            in_dim,
        ) {
            if has_err {
                let lower_err = Self::downcast_coefficient_error(
                    &lower_a,
                    result.lower_a(),
                    lower_a_err.as_ref(),
                );
                let upper_err = Self::downcast_coefficient_error(
                    &upper_a,
                    result.upper_a(),
                    upper_a_err.as_ref(),
                );
                result.set_coeff_err(lower_err, upper_err);
            }
            result
        } else {
            BatchedLinearBounds::conservative(input_shape, output_shape)
        }
    }

    /// Reconstitute an f32 coefficient certificate after the f64 sidecar is
    /// downcast. The certificate includes both carried fan-in error and the
    /// exact gap between the f64 coefficient and its stored f32 value.
    fn downcast_coefficient_error(
        source: &Array2<f64>,
        stored: &ArrayD<f32>,
        carried: Option<&Array2<f64>>,
    ) -> ArrayD<f32> {
        let mut result = ArrayD::zeros(stored.raw_dim());
        let mut carried_values = carried.map(|error| error.iter());
        for ((out, &stored_value), &source_value) in
            result.iter_mut().zip(stored.iter()).zip(source.iter())
        {
            let carried_value = carried_values
                .as_mut()
                .and_then(|values| values.next())
                .copied()
                .unwrap_or(0.0);
            let cast_gap = (f64::from(stored_value) - source_value).abs();
            *out = Self::err_to_f32(Self::add_nonnegative_f64_up(carried_value, cast_gap));
        }
        result
    }

    pub(crate) fn accumulate(&mut self, new_bounds: &BatchedLinearBounds) {
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
            let (sum, residual) = two_sum(*existing_value, f64::from(new_value));
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
                *value = Self::add_nonnegative_f64_up(*value, f64::from(incoming));
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
        let widened = next_up_f32(error as f32);
        if widened.is_finite() {
            widened
        } else {
            f32::INFINITY
        }
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
                let (sum, residual) = two_sum(*existing_value, f64::from(new_value));
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
            lower_row.push(Self::downcast_coeff(lower_a[[row, col]], true)?);
            upper_row.push(Self::downcast_coeff(upper_a[[row, col]], false)?);
        }

        Some((lower_row, lower_bias, upper_row, upper_bias))
    }

    fn downcast_coeff(value: f64, is_lower: bool) -> Option<f32> {
        if !value.is_finite() {
            return None;
        }

        let cast = value as f32;
        if !cast.is_finite() {
            return None;
        }

        Some(if is_lower {
            next_down_f32(cast)
        } else {
            next_up_f32(cast)
        })
    }

    fn downcast_lower_bias(value: f64) -> Option<f32> {
        if value == f64::NEG_INFINITY {
            return Some(f32::NEG_INFINITY);
        }
        Self::downcast_coeff(value, true)
    }

    fn downcast_upper_bias(value: f64) -> Option<f32> {
        if value == f64::INFINITY {
            return Some(f32::INFINITY);
        }
        Self::downcast_coeff(value, false)
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

        for (actual, expected) in merged.lower_a().iter().zip(expected_lower_a.iter()) {
            assert_eq!(*actual, next_down_f32(*expected));
        }
        for (actual, expected) in merged.lower_b().iter().zip(expected_lower_b.iter()) {
            assert_eq!(*actual, next_down_f32(*expected));
        }
        for (actual, expected) in merged.upper_a().iter().zip(expected_upper_a.iter()) {
            assert_eq!(*actual, next_up_f32(*expected));
        }
        for (actual, expected) in merged.upper_b().iter().zip(expected_upper_b.iter()) {
            assert_eq!(*actual, next_up_f32(*expected));
        }
    }
}
