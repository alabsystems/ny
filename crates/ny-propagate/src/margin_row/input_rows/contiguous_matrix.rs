// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! The sealed row carrier needs only zero-origin, contiguous owned matrices.
//! Keep their original allocation and shape without redundant stride headers.

use ndarray::Array2;
use std::ops::Index;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MatrixRefusal {
    Layout,
    Origin,
    Shape,
}

pub(super) struct ContiguousMatrix<T> {
    shape: (usize, usize),
    values: Vec<T>,
}

impl<T> ContiguousMatrix<T> {
    pub(super) fn from_shape_vec(
        shape: (usize, usize),
        values: Vec<T>,
    ) -> Result<Self, MatrixRefusal> {
        if shape.0.checked_mul(shape.1) != Some(values.len()) {
            return Err(MatrixRefusal::Shape);
        }
        Ok(Self { shape, values })
    }

    pub(super) fn from_array(matrix: Array2<T>) -> Result<Self, MatrixRefusal> {
        if !matrix.is_standard_layout() {
            return Err(MatrixRefusal::Layout);
        }
        let shape = matrix.dim();
        let cells = matrix.len();
        let (values, offset) = matrix.into_raw_vec_and_offset();
        if offset != Some(0) || values.len() != cells {
            return Err(MatrixRefusal::Origin);
        }
        Self::from_shape_vec(shape, values)
    }

    pub(super) fn dim(&self) -> (usize, usize) {
        self.shape
    }

    pub(super) fn len(&self) -> usize {
        self.values.len()
    }

    pub(super) fn capacity(&self) -> usize {
        self.values.capacity()
    }

    pub(super) fn is_standard_layout(&self) -> bool {
        self.shape.0.checked_mul(self.shape.1) == Some(self.values.len())
    }

    pub(super) fn iter(&self) -> std::slice::Iter<'_, T> {
        self.values.iter()
    }

    pub(super) fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.values.iter_mut()
    }

    #[cfg(test)]
    pub(super) fn as_ptr(&self) -> *const T {
        self.values.as_ptr()
    }
}

impl<T> Index<[usize; 2]> for ContiguousMatrix<T> {
    type Output = T;

    fn index(&self, [row, column]: [usize; 2]) -> &Self::Output {
        // Check both coordinates, preserving Array2's refusal of a column
        // which would otherwise alias the next row in the flat allocation.
        assert!(row < self.shape.0 && column < self.shape.1);
        &self.values[row * self.shape.1 + column]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_vector_pointer_capacity_shape_and_bits_move_without_a_copy() {
        let values = vec![0.0f64, -0.0, f64::from_bits(1), 1.25, -2.5, f64::MAX];
        let pointer = values.as_ptr();
        let capacity = values.capacity();
        let array = Array2::from_shape_vec((2, 3), values).unwrap();
        let matrix = ContiguousMatrix::from_array(array).unwrap();
        assert_eq!(matrix.as_ptr(), pointer);
        assert_eq!(matrix.capacity(), capacity);
        assert_eq!(matrix.dim(), (2, 3));
        assert_eq!(matrix.len(), 6);
        assert!(matrix.is_standard_layout());
        assert_eq!(matrix[[0, 1]].to_bits(), (-0.0f64).to_bits());
        assert_eq!(matrix[[0, 2]].to_bits(), 1);
        assert_eq!(matrix[[1, 2]], f64::MAX);
        assert!(size_of::<ContiguousMatrix<f64>>() < size_of::<Array2<f64>>());
    }

    #[test]
    fn binary32_bits_and_mutable_error_iteration_keep_row_major_order() {
        let values = vec![0.0f32, -0.0, f32::from_bits(1), f32::MAX];
        let bits: Vec<_> = values.iter().map(|value| value.to_bits()).collect();
        let pointer = values.as_ptr();
        let mut matrix = ContiguousMatrix::from_shape_vec((2, 2), values).unwrap();
        assert_eq!(matrix.as_ptr(), pointer);
        assert_eq!(
            matrix
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
        for (value, replacement) in matrix.iter_mut().zip([0.0, 1.0, 2.0, 3.0]) {
            *value = replacement;
        }
        assert_eq!(matrix[[0, 0]], 0.0);
        assert_eq!(matrix[[0, 1]], 1.0);
        assert_eq!(matrix[[1, 0]], 2.0);
        assert_eq!(matrix[[1, 1]], 3.0);
    }

    #[test]
    fn nonstandard_nonzero_and_oversized_origins_refuse_before_a_carrier_exists() {
        let nonstandard = Array2::from_shape_vec((2, 3), vec![0.0f64; 6])
            .unwrap()
            .reversed_axes();
        assert!(matches!(
            ContiguousMatrix::from_array(nonstandard),
            Err(MatrixRefusal::Layout)
        ));
        let nonzero = Array2::from_shape_vec((2, 3), vec![0.0f64; 6])
            .unwrap()
            .slice_move(ndarray::s![1.., ..]);
        assert!(matches!(
            ContiguousMatrix::from_array(nonzero),
            Err(MatrixRefusal::Origin)
        ));
        let oversized = Array2::from_shape_vec((2, 3), vec![0.0f64; 6])
            .unwrap()
            .slice_move(ndarray::s![..1, ..]);
        assert!(matches!(
            ContiguousMatrix::from_array(oversized),
            Err(MatrixRefusal::Origin)
        ));
        assert!(matches!(
            ContiguousMatrix::from_shape_vec((usize::MAX, 2), Vec::<f64>::new()),
            Err(MatrixRefusal::Shape)
        ));
        assert!(matches!(
            ContiguousMatrix::from_shape_vec((2, 3), vec![0.0f64; 5]),
            Err(MatrixRefusal::Shape)
        ));
    }

    #[test]
    fn each_coordinate_refuses_instead_of_aliasing_another_row() {
        let matrix = ContiguousMatrix::from_shape_vec((2, 3), vec![0u8; 6]).unwrap();
        for coordinate in [[0, 3], [2, 0], [usize::MAX, 0], [0, usize::MAX]] {
            assert!(std::panic::catch_unwind(|| matrix[coordinate]).is_err());
        }
    }
}
