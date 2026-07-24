// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::ArrayD;
use ny_core::{NyError, Result};

pub(super) fn array_slice<'a>(name: &str, array: &'a ArrayD<f32>) -> Result<&'a [f32]> {
    // Use as_slice() (not as_slice_memory_order()) to require C-contiguous layout.
    // GPU shaders index as row-major: a_offset + i * k + kk for A[i, kk].
    // as_slice_memory_order() accepts Fortran-layout arrays but returns data in
    // column-major order, which the shader misinterprets as row-major, producing
    // silently wrong bounds. Part of #2257.
    array.as_slice().ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "attention_ibp_fused: {name} must be C-contiguous (standard layout)"
        ))
    })
}

pub(super) fn checked_mul_usize(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| NyError::InvalidSpec(format!("attention_ibp_fused: {label} overflow")))
}

pub(super) fn to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| NyError::InvalidSpec(format!("attention_ibp_fused: {label} exceeds u32")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{IxDyn, ShapeBuilder};

    #[test]
    fn test_array_slice_c_contiguous_succeeds() {
        let arr =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(arr.as_slice().is_some(), "precondition: C-contiguous");
        let result = array_slice("test", &arr);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// Regression test for #2257: Fortran-layout arrays must be rejected.
    /// as_slice_memory_order() would accept these and return column-major data,
    /// which GPU shaders (expecting row-major) would silently misinterpret.
    #[test]
    fn test_array_slice_fortran_layout_rejected_2257() {
        // Create C-order array, then copy into Fortran-order allocation.
        let c_arr =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut f_arr = ArrayD::zeros(IxDyn(&[2, 3]).f());
        f_arr.assign(&c_arr);

        // Precondition: Fortran-layout must fail as_slice() but pass as_slice_memory_order()
        assert!(
            f_arr.as_slice().is_none(),
            "precondition: Fortran-layout array should fail as_slice()"
        );
        assert!(
            f_arr.as_slice_memory_order().is_some(),
            "precondition: Fortran-layout array should pass as_slice_memory_order()"
        );

        // array_slice must reject Fortran-layout arrays
        let result = array_slice("fortran_test", &f_arr);
        assert!(result.is_err(), "Fortran-layout array should be rejected");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("C-contiguous"),
            "error should mention C-contiguous, got: {err_msg}"
        );
    }
}
