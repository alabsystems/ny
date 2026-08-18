// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Narrow selected-column materialization for beta-gradient intermediates.
//!
//! This is not a general Patches-to-dense replacement. It deliberately admits
//! only materialized contiguous 7D explicit-row carriers whose lower/upper
//! geometry agrees. Every refusal returns `None`, allowing the caller to execute
//! the historical full `to_dense()` path unchanged.

use ndarray::Array2;
use ny_core::checked_shape_product;

use super::scatter::{
    compute_unfold_index_map, scatter_rows_selected_columns_with_unfold_map, validate_patches_shape,
};
use super::PatchesLinearBounds;

impl PatchesLinearBounds {
    /// Try to materialize only the lower-A input-flat columns needed by beta
    /// sensitivity.
    ///
    /// Returns normalized (ascending, duplicate-free) global neuron indices and
    /// a `(row_count, selected_columns)` compact matrix. `None` means the caller
    /// must retain the historical full-dense capture.
    pub(crate) fn try_lower_a_beta_columns(
        &self,
        requested_columns: &[usize],
    ) -> Option<(Vec<usize>, Array2<f32>)> {
        if self.lower_a.identity
            || self.upper_a.identity
            || self.lower_a.unstable_idx.is_some()
            || self.upper_a.unstable_idx.is_some()
        {
            return None;
        }

        self.validate_row_count().ok()?;
        self.lower_a.validate_common_geometry(&self.upper_a).ok()?;

        let (out_c, out_h, out_w) = self.lower_a.output_shape;
        let (in_c, in_h, in_w) = self.lower_a.input_shape;
        let _out_dim = checked_shape_product(&[out_c, out_h, out_w])?;
        let in_dim = checked_shape_product(&[in_c, in_h, in_w])?;

        let (lower, lower_explicit_rows) =
            validate_patches_shape(&self.lower_a, self.row_count, out_c, out_h, out_w, in_c)
                .ok()?;
        let (upper, upper_explicit_rows) =
            validate_patches_shape(&self.upper_a, self.row_count, out_c, out_h, out_w, in_c)
                .ok()?;

        // M6 intentionally excludes legacy 6D and every sparse/identity layout.
        // Contiguous equal shapes make the flattened 7D address arithmetic
        // identical to the canonical full scatter.
        if !lower_explicit_rows
            || !upper_explicit_rows
            || lower.shape() != upper.shape()
            || lower.as_slice().is_none()
            || upper.as_slice().is_none()
        {
            return None;
        }

        let shape = lower.shape();
        let (kh, kw) = (shape[5], shape[6]);
        let _block = checked_shape_product(&[in_c, kh, kw])?;
        let _patch_len =
            checked_shape_product(&[self.row_count, out_c, out_h, out_w, in_c, kh, kw])?;
        // Match the 7D full materializer's hard coeff-error length precondition.
        for err in [
            self.lower_a.coeff_err.as_ref(),
            self.upper_a.coeff_err.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if err.len() != self.row_count {
                return None;
            }
        }

        // The historical constructor turns a NaN bias into a conservative
        // all-zero A relation. Keep that rare case on the historical path.
        if self.lower_b.iter().any(|value| value.is_nan())
            || self.upper_b.iter().any(|value| value.is_nan())
        {
            return None;
        }

        let mut selected_columns: Vec<usize> = requested_columns
            .iter()
            .copied()
            .filter(|&column| column < in_dim)
            .collect();
        selected_columns.sort_unstable();
        selected_columns.dedup();

        // Empty/all-column captures provide no benefit. Refusing also makes
        // malformed all-out-of-range beta metadata take the historical path.
        if selected_columns.is_empty() || selected_columns.len() >= in_dim {
            return None;
        }

        let index_map = compute_unfold_index_map(&self.lower_a, kh, kw).ok()?;
        let expected_map_len = checked_shape_product(&[out_h, out_w, in_c, kh, kw])?;
        if index_map.len() != expected_map_len {
            return None;
        }

        let mut compact = Array2::<f32>::zeros((self.row_count, selected_columns.len()));
        scatter_rows_selected_columns_with_unfold_map(
            &mut compact,
            &selected_columns,
            lower,
            &index_map,
            self.row_count,
            out_c,
            out_h,
            out_w,
            in_c,
            kh,
            kw,
        );

        // Selected non-finite coefficients would have triggered the historical
        // LinearBounds firewall. Refuse and let that exact path handle them.
        if compact.iter().any(|value| !value.is_finite()) {
            return None;
        }

        Some((selected_columns, compact))
    }
}

#[cfg(test)]
mod anchored_tests {
    use super::*;
    use crate::bounds::patches::{PatchGeometry, PatchesData};
    use ndarray::{Array1, ArrayD, IxDyn};

    #[test]
    fn anchored_selected_columns_uses_the_same_exact_7d_map() {
        let patches =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 2, 1, 1, 1]), vec![2.0, 5.0]).unwrap();
        let data = PatchesData {
            coeff_err: None,
            patches: Some(patches),
            geometry: PatchGeometry::anchored(vec![0], vec![0, 3]).unwrap(),
            identity: false,
            output_shape: (1, 1, 2),
            input_shape: (1, 1, 4),
            unstable_idx: None,
        };
        let bounds = PatchesLinearBounds {
            row_count: 1,
            lower_a: data.clone(),
            lower_b: Array1::zeros(1),
            upper_a: data,
            upper_b: Array1::zeros(1),
        };

        let (columns, compact) = bounds
            .try_lower_a_beta_columns(&[3, 0, 3, usize::MAX])
            .expect("anchored 7D carrier should use the exact selected-column plan");
        assert_eq!(columns, vec![0, 3]);
        assert_eq!(
            compact,
            Array2::from_shape_vec((1, 2), vec![2.0, 5.0]).unwrap()
        );
    }
}
