// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;

use ndarray::{ArrayBase, Data, Dimension, OwnedRepr};

use crate::{NyError, Result};

/// Return a flat slice view of the array, copying only when the source layout is
/// non-contiguous (for example after reshape, broadcast, or transpose).
#[must_use]
pub(crate) fn contiguous_flat_slice<S, D>(arr: &ArrayBase<S, D>) -> Cow<'_, [f32]>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    if let Some(slice) = arr.as_slice() {
        Cow::Borrowed(slice)
    } else {
        Cow::Owned(arr.iter().copied().collect())
    }
}

/// Ensure the array is contiguous before returning a mutable flat slice.
pub(crate) fn contiguous_flat_slice_mut<T, D>(
    arr: &mut ArrayBase<OwnedRepr<T>, D>,
) -> Result<&mut [T]>
where
    T: Clone,
    D: Dimension,
{
    if arr.as_slice_mut().is_none() {
        let normalized = arr.as_standard_layout().into_owned();
        *arr = normalized;
    }

    if let Some(slice) = arr.as_slice_mut() {
        Ok(slice)
    } else {
        Err(NyError::InternalError(
            "contiguous_flat_slice_mut: layout normalization still produced a non-contiguous array"
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};

    use super::{contiguous_flat_slice, contiguous_flat_slice_mut};

    fn make_non_contiguous_2d() -> ArrayD<f32> {
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .expect("shape should be valid")
            .view()
            .reversed_axes()
            .to_owned()
    }

    #[test]
    fn test_contiguous_flat_slice_reads_non_contiguous_array_4250() {
        let arr = make_non_contiguous_2d();
        assert!(
            arr.as_slice().is_none(),
            "precondition: reversed-axes owned array should be non-contiguous"
        );

        let flat = contiguous_flat_slice(&arr);
        assert_eq!(flat.as_ref(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_contiguous_flat_slice_mut_normalizes_non_contiguous_array_4250() {
        let mut arr = make_non_contiguous_2d();
        assert!(
            arr.as_slice().is_none(),
            "precondition: reversed-axes owned array should be non-contiguous"
        );

        {
            let flat = contiguous_flat_slice_mut(&mut arr)
                .expect("helper should normalize mutable non-contiguous arrays");
            flat[0] = 42.0;
            flat[5] = -7.0;
        }

        assert!(
            arr.is_standard_layout(),
            "array should be normalized in-place"
        );
        assert_eq!(
            arr.as_slice()
                .expect("normalized array should be contiguous"),
            &[42.0, 2.0, 3.0, 4.0, 5.0, -7.0]
        );
    }
}
