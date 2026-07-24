// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP propagation for ReLU activation.

use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::bounds::nan_propagating_max_zero;

/// IBP propagation for ReLU activation.
/// Uses NaN-propagating max(0) so NaN bounds poison rather than silently vanish (#2432).
pub(crate) fn relu_ibp(input: &BoundedTensor) -> Result<BoundedTensor> {
    let lower = input.lower().mapv(nan_propagating_max_zero);
    let upper = input.upper().mapv(nan_propagating_max_zero);
    BoundedTensor::new_allow_infinite(lower, upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[ntest::timeout(5000)]
    #[test]
    fn test_relu_ibp_positive() {
        let input = BoundedTensor::new(
            array![1.0, 2.0, 3.0].into_dyn(),
            array![2.0, 3.0, 4.0].into_dyn(),
        )
        .unwrap();
        let output = relu_ibp(&input).unwrap();
        assert_eq!(output.lower(), array![1.0, 2.0, 3.0].into_dyn());
        assert_eq!(output.upper(), array![2.0, 3.0, 4.0].into_dyn());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_relu_ibp_negative() {
        let input = BoundedTensor::new(
            array![-3.0, -2.0, -1.0].into_dyn(),
            array![-2.0, -1.0, -0.5].into_dyn(),
        )
        .unwrap();
        let output = relu_ibp(&input).unwrap();
        assert_eq!(output.lower(), array![0.0, 0.0, 0.0].into_dyn());
        assert_eq!(output.upper(), array![0.0, 0.0, 0.0].into_dyn());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_relu_ibp_crossing() {
        let input = BoundedTensor::new(
            array![-1.0, -2.0, 0.0].into_dyn(),
            array![1.0, 3.0, 2.0].into_dyn(),
        )
        .unwrap();
        let output = relu_ibp(&input).unwrap();
        assert_eq!(output.lower(), array![0.0, 0.0, 0.0].into_dyn());
        assert_eq!(output.upper(), array![1.0, 3.0, 2.0].into_dyn());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_relu_ibp_mixed() {
        let input = BoundedTensor::new(
            array![1.0, -3.0, -1.0].into_dyn(),
            array![2.0, -1.0, 1.0].into_dyn(),
        )
        .unwrap();
        let output = relu_ibp(&input).unwrap();
        assert_eq!(output.lower(), array![1.0, 0.0, 0.0].into_dyn());
        assert_eq!(output.upper(), array![2.0, 0.0, 1.0].into_dyn());
    }
}
