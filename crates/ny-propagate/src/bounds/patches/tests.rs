// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, Array1, ArrayD, Dimension, IxDyn};
use ny_core::Result;
use std::time::Instant;

use super::*;

#[test]
fn test_identity_patches_to_dense() -> Result<()> {
    let shape = (2, 3, 3); // 2 channels, 3x3 spatial
    let dim = shape.0 * shape.1 * shape.2; // 18
    let patches_data = PatchesData {
        coeff_err: None,
        patches: None,
        geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
        identity: true,
        output_shape: shape,
        input_shape: shape,
        unstable_idx: None,
    };
    let plb = PatchesLinearBounds {
        row_count: dim,
        lower_a: patches_data.clone(),
        lower_b: Array1::zeros(dim),
        upper_a: patches_data,
        upper_b: Array1::zeros(dim),
    };
    let dense = plb.to_dense()?;
    assert_eq!(dense.num_outputs(), dim);
    assert_eq!(dense.num_inputs(), dim);
    for i in 0..dim {
        for j in 0..dim {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(
                dense.lower_a()[[i, j]],
                expected,
                "lower_a[{i},{j}] mismatch"
            );
            assert_eq!(
                dense.upper_a()[[i, j]],
                expected,
                "upper_a[{i},{j}] mismatch"
            );
        }
    }
    Ok(())
}

#[test]
fn test_crown_bounds_ensure_dense_noop_for_dense() -> Result<()> {
    let lb = LinearBounds::identity(4);
    let mut cb = CrownBounds::Dense(lb);
    let dense_ref = cb.ensure_dense()?;
    assert_eq!(dense_ref.num_outputs(), 4);
    assert_eq!(dense_ref.num_inputs(), 4);
    Ok(())
}

#[test]
fn test_crown_bounds_ensure_dense_converts_patches() -> Result<()> {
    let shape = (1, 2, 2); // 1 channel, 2x2
    let dim = 4;
    let patches_data = PatchesData {
        coeff_err: None,
        patches: None,
        geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
        identity: true,
        output_shape: shape,
        input_shape: shape,
        unstable_idx: None,
    };
    let plb = PatchesLinearBounds {
        row_count: dim,
        lower_a: patches_data.clone(),
        lower_b: Array1::zeros(dim),
        upper_a: patches_data,
        upper_b: Array1::zeros(dim),
    };
    let mut cb = CrownBounds::Patches(Box::new(plb));
    let dense_ref = cb.ensure_dense()?;
    assert_eq!(dense_ref.num_outputs(), dim);
    assert_eq!(dense_ref.num_inputs(), dim);
    assert!(matches!(cb, CrownBounds::Dense(_)));
    Ok(())
}

#[test]
fn test_patches_to_dense_1x1_conv_identity() -> Result<()> {
    let patches_arr = ArrayD::ones(IxDyn(&[1, 2, 2, 1, 1, 1]));
    let patches_data = PatchesData {
        coeff_err: None,
        patches: Some(patches_arr),
        geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
        identity: false,
        output_shape: (1, 2, 2),
        input_shape: (1, 2, 2),
        unstable_idx: None,
    };
    let dim = 4;
    let plb = PatchesLinearBounds {
        row_count: dim,
        lower_a: patches_data.clone(),
        lower_b: Array1::zeros(dim),
        upper_a: patches_data,
        upper_b: arr1(&[1.0, 2.0, 3.0, 4.0]),
    };
    let dense = plb.to_dense()?;
    assert_eq!(dense.num_outputs(), 4);
    assert_eq!(dense.num_inputs(), 4);
    for i in 0..4 {
        assert_eq!(dense.lower_a()[[i, i]], 1.0);
        assert_eq!(dense.upper_a()[[i, i]], 1.0);
    }
    assert_eq!(dense.upper_b()[[2]], 3.0);
    Ok(())
}

#[test]
fn test_patches_to_dense_3x3_conv_with_padding() -> Result<()> {
    let (in_c, in_h, in_w) = (1, 3, 3);
    let (out_c, out_h, out_w) = (1, 3, 3);
    let (kh, kw) = (3, 3);
    let in_dim = in_c * in_h * in_w;
    let out_dim = out_c * out_h * out_w;

    let patches_arr = ArrayD::ones(IxDyn(&[out_c, out_h, out_w, in_c, kh, kw]));
    let patches_data = PatchesData {
        coeff_err: None,
        patches: Some(patches_arr),
        geometry: PatchGeometry::affine((1, 1), (1, 1, 1, 1)),
        identity: false,
        output_shape: (out_c, out_h, out_w),
        input_shape: (in_c, in_h, in_w),
        unstable_idx: None,
    };
    let plb = PatchesLinearBounds {
        row_count: out_dim,
        lower_a: patches_data.clone(),
        lower_b: Array1::zeros(out_dim),
        upper_a: patches_data,
        upper_b: Array1::zeros(out_dim),
    };
    let dense = plb.to_dense()?;
    assert_eq!(dense.lower_a().shape(), &[out_dim, in_dim]);

    let row0: Vec<f32> = (0..in_dim).map(|j| dense.lower_a()[[0, j]]).collect();
    assert_eq!(row0, vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    let row4: Vec<f32> = (0..in_dim).map(|j| dense.lower_a()[[4, j]]).collect();
    assert_eq!(row4, vec![1.0; 9]);

    Ok(())
}

fn explicit_row_overlap_bounds() -> PatchesLinearBounds {
    let shape = IxDyn(&[3, 2, 2, 2, 1, 3, 3]);
    let lower = ArrayD::from_shape_fn(shape, |index| {
        let ordinal = index
            .slice()
            .iter()
            .fold(0usize, |acc, &part| acc.wrapping_mul(7).wrapping_add(part));
        if ordinal % 29 == 0 {
            -0.0
        } else {
            ((ordinal % 19) as f32 - 9.0) * 0.125
        }
    });
    let upper = lower.mapv(|value| value * -0.5);
    let side = |patches| PatchesData {
        coeff_err: Some(arr1(&[1.0e-4, 2.0e-4, 3.0e-4])),
        patches: Some(patches),
        geometry: PatchGeometry::affine((1, 1), (1, 1, 1, 1)),
        identity: false,
        output_shape: (2, 2, 2),
        input_shape: (1, 2, 2),
        unstable_idx: None,
    };
    PatchesLinearBounds {
        row_count: 3,
        lower_a: side(lower),
        lower_b: arr1(&[-0.0, 0.25, -0.5]),
        upper_a: side(upper),
        upper_b: arr1(&[0.0, -0.25, 0.5]),
    }
}

#[test]
fn test_explicit_rows_selected_beta_columns_match_full_dense_bitwise() -> Result<()> {
    let bounds = explicit_row_overlap_bounds();
    let dense = bounds.to_dense()?;
    let (indices, compact) = bounds
        .try_lower_a_beta_columns(&[3, 0, 3, usize::MAX, 2])
        .expect("narrow explicit-row carrier should admit selected-column capture");

    assert_eq!(indices, vec![0, 2, 3]);
    assert_eq!(compact.shape(), &[3, 3]);
    for row in 0..compact.nrows() {
        for (compact_col, &global_col) in indices.iter().enumerate() {
            assert_eq!(
                compact[[row, compact_col]].to_bits(),
                dense.lower_a()[[row, global_col]].to_bits(),
                "selected scatter must preserve dense += order at row={row}, global_col={global_col}"
            );
        }
    }
    Ok(())
}

#[test]
fn test_selected_beta_columns_refuse_non_narrow_or_malformed_carriers() {
    let bounds = explicit_row_overlap_bounds();
    assert!(
        bounds.try_lower_a_beta_columns(&[0, 1, 2, 3]).is_none(),
        "capturing every input column is not a sparse win"
    );

    let mut malformed_err = bounds.clone();
    malformed_err.lower_a.coeff_err = Some(arr1(&[1.0e-4]));
    assert!(
        malformed_err.try_lower_a_beta_columns(&[0]).is_none(),
        "malformed 7D coeff-error metadata must use historical materialization"
    );

    let mut selected_non_finite = bounds;
    selected_non_finite
        .lower_a
        .patches
        .as_mut()
        .expect("explicit lower patches")[[0, 0, 0, 0, 0, 1, 1]] = f32::INFINITY;
    assert!(
        selected_non_finite.try_lower_a_beta_columns(&[0]).is_none(),
        "non-finite selected results must use the historical firewall"
    );

    let legacy_shape = IxDyn(&[2, 2, 2, 1, 3, 3]);
    let legacy_side = PatchesData {
        coeff_err: None,
        patches: Some(ArrayD::zeros(legacy_shape)),
        geometry: PatchGeometry::affine((1, 1), (1, 1, 1, 1)),
        identity: false,
        output_shape: (2, 2, 2),
        input_shape: (1, 2, 2),
        unstable_idx: None,
    };
    let legacy = PatchesLinearBounds {
        row_count: 8,
        lower_a: legacy_side.clone(),
        lower_b: Array1::zeros(8),
        upper_a: legacy_side,
        upper_b: Array1::zeros(8),
    };
    assert!(
        legacy.try_lower_a_beta_columns(&[0]).is_none(),
        "legacy 6D layout stays on the established full-dense path"
    );
}

#[test]
fn test_should_fallback_to_dense() {
    let data = PatchesData {
        coeff_err: None,
        patches: Some(ArrayD::zeros(IxDyn(&[1, 3, 3, 1, 5, 5]))),
        geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
        identity: false,
        output_shape: (1, 3, 3),
        input_shape: (1, 5, 5),
        unstable_idx: None,
    };
    assert!(data.should_fallback_to_dense());

    let data2 = PatchesData {
        coeff_err: None,
        patches: Some(ArrayD::zeros(IxDyn(&[1, 6, 6, 1, 3, 3]))),
        geometry: PatchGeometry::affine((1, 1), (1, 1, 1, 1)),
        identity: false,
        output_shape: (1, 6, 6),
        input_shape: (1, 6, 6),
        unstable_idx: None,
    };
    assert!(!data2.should_fallback_to_dense());
}

/// Helper: build a non-identity PatchesData with the given effective kernel and
/// input spatial extent, for exercising the area-crossover heuristics.
fn patches_with_kernel(kh: usize, kw: usize, in_h: usize, in_w: usize) -> PatchesData {
    PatchesData {
        coeff_err: None,
        patches: Some(ArrayD::zeros(IxDyn(&[1, 1, 1, 1, kh, kw]))),
        geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
        identity: false,
        output_shape: (1, 1, 1),
        input_shape: (1, in_h, in_w),
        unstable_idx: None,
    }
}

/// The pre-check (`would_conv_compose_cover_input`) must use the MEMORY-area
/// crossover, not the old fixed 75%-per-dimension threshold (#hotpath).
///
/// An effective 7x7 kernel over a 9x9 input has area 49 < 81 (patches still
/// ~1.65x cheaper than dense) but the OLD rule bailed (7 >= ceil(9*3/4)=7). The
/// new area rule keeps it in patches; composing further to 9x9 (area 81 >= 81)
/// reaches the dense element count and bails.
#[test]
fn test_compose_cover_uses_area_crossover_not_75pct() {
    let pd = patches_with_kernel(7, 7, 9, 9);
    assert!(
        !pd.would_conv_compose_cover_input((1, 1), (1, 1), 9, 9),
        "7x7 over 9x9 (area 49<81) should stay in patches under area crossover"
    );
    assert!(
        pd.would_conv_compose_cover_input((1, 1), (3, 3), 9, 9),
        "composing to 9x9 (area 81>=81) reaches dense area -> bail to dense"
    );
}

/// Regression: at the old 75% boundary the new heuristic keeps patches active.
/// effective 6x6 over 8x8 input: old rule bailed (6 >= ceil(8*3/4)=6), new rule
/// keeps patches (area 36 < 64, ~1.78x cheaper).
#[test]
fn test_compose_cover_keeps_patches_at_old_75pct_boundary() {
    let pd = patches_with_kernel(6, 6, 8, 8);
    assert!(
        !pd.would_conv_compose_cover_input((1, 1), (1, 1), 8, 8),
        "6x6 over 8x8 (area 36<64) was bailed by old 75% rule; area crossover keeps it"
    );
}

/// Anisotropic receptive fields: a tall-thin kernel (covers full height, narrow
/// width) still saves memory and must stay in patches even though one dimension
/// fully covers the input; the full-area case bails.
#[test]
fn test_compose_cover_anisotropic() {
    let pd = patches_with_kernel(9, 3, 9, 9);
    assert!(
        !pd.would_conv_compose_cover_input((1, 1), (1, 1), 9, 9),
        "9x3 over 9x9 (area 27<81) should stay in patches"
    );
    let pd_full = patches_with_kernel(9, 9, 9, 9);
    assert!(
        pd_full.would_conv_compose_cover_input((1, 1), (1, 1), 9, 9),
        "9x9 over 9x9 (full area) should bail to dense"
    );
}

/// `should_fallback_to_dense` (post-composition check) must use the same area
/// crossover so the pre-check decision is consistent with the post-check.
#[test]
fn test_should_fallback_uses_area_crossover() {
    let pd = patches_with_kernel(7, 7, 9, 9);
    assert!(!pd.should_fallback_to_dense());
    let pd_full = patches_with_kernel(9, 9, 9, 9);
    assert!(pd_full.should_fallback_to_dense());
    let pd_thin = patches_with_kernel(9, 3, 9, 9);
    assert!(!pd_thin.should_fallback_to_dense());
}

/// Gate test (#2613): Patches mode uses dramatically less memory than Dense.
#[test]
fn test_patches_vs_dense_memory_savings_2613() {
    type CnnConfig<'a> = (
        &'a str,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    );
    let configs: Vec<CnnConfig<'_>> = vec![
        ("mnist_conv", 2, 6, 6, 1, 3, 3, 8, 8),
        ("moderate_cnn", 8, 8, 8, 4, 3, 3, 10, 10),
        ("cifar10_first", 32, 30, 30, 3, 3, 3, 32, 32),
        ("deep_cnn", 64, 14, 14, 32, 3, 3, 16, 16),
    ];

    for (name, oc, oh, ow, ic, kh, kw, ih, iw) in &configs {
        let out_dim = oc * oh * ow;
        let dense_bytes = LinearBounds::identity(out_dim).memory_bytes();
        assert_eq!(dense_bytes, (2 * out_dim * out_dim + 2 * out_dim) * 4);

        let patches_arr = ArrayD::zeros(IxDyn(&[*oc, *oh, *ow, *ic, *kh, *kw]));
        let pd = PatchesData {
            coeff_err: None,
            patches: Some(patches_arr),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (*oc, *oh, *ow),
            input_shape: (*ic, *ih, *iw),
            unstable_idx: None,
        };
        let plb = PatchesLinearBounds {
            row_count: out_dim,
            lower_a: pd.clone(),
            lower_b: Array1::zeros(out_dim),
            upper_a: pd,
            upper_b: Array1::zeros(out_dim),
        };
        let patches_bytes = plb.memory_bytes();

        let ratio = dense_bytes as f64 / patches_bytes as f64;
        assert!(
            ratio > 1.0,
            "{}: Patches should use less memory than Dense (ratio={:.1}x)",
            name,
            ratio
        );
        if out_dim > 100 {
            assert!(
                ratio > 3.0,
                "{}: Expected >3x memory savings, got {:.1}x (dense={}B, patches={}B)",
                name,
                ratio,
                dense_bytes,
                patches_bytes
            );
        }
    }
}

#[test]
fn test_sparse_identity_to_dense() -> Result<()> {
    let shape = (1, 3, 3);
    let dim = 9;
    let idx = UnstableIdx {
        channels: vec![0, 0, 0, 0],
        heights: vec![0, 1, 1, 2],
        widths: vec![0, 1, 2, 2],
    };
    let sparse = PatchesLinearBounds::sparse_identity(shape, shape, idx);
    assert_eq!(sparse.lower_b.len(), 4);
    assert_eq!(sparse.upper_b.len(), 4);

    let dense = sparse.to_dense()?;
    assert_eq!(dense.num_outputs(), dim);
    assert_eq!(dense.num_inputs(), dim);

    let unstable_flats = [0usize, 4, 5, 8];
    for i in 0..dim {
        for j in 0..dim {
            let expected = if unstable_flats.contains(&i) && i == j {
                1.0
            } else {
                0.0
            };
            assert_eq!(
                dense.lower_a()[[i, j]],
                expected,
                "lower_a[{i},{j}] expected {expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn test_filter_to_unstable_roundtrip() -> Result<()> {
    let (out_c, out_h, out_w) = (1, 2, 2);
    let (in_c, in_h, in_w) = (1, 3, 3);
    let kh = 2;
    let kw = 2;

    let mut patches_arr = ArrayD::zeros(IxDyn(&[out_c, out_h, out_w, in_c, kh, kw]));
    for i in 0..patches_arr.len() {
        patches_arr.as_slice_mut().unwrap()[i] = (i as f32 + 1.0) * 0.1;
    }

    let patches_data = PatchesData {
        coeff_err: None,
        patches: Some(patches_arr),
        geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
        identity: false,
        output_shape: (out_c, out_h, out_w),
        input_shape: (in_c, in_h, in_w),
        unstable_idx: None,
    };
    let plb = PatchesLinearBounds {
        row_count: out_c * out_h * out_w,
        lower_a: patches_data.clone(),
        lower_b: arr1(&[1.0, 2.0, 3.0, 4.0]),
        upper_a: patches_data,
        upper_b: arr1(&[5.0, 6.0, 7.0, 8.0]),
    };

    let dense_full = plb.to_dense()?;

    let mut mask = ndarray::Array3::<bool>::from_elem((out_c, out_h, out_w), false);
    mask[[0, 0, 0]] = true;
    mask[[0, 1, 0]] = true;

    let sparse = plb.filter_to_unstable(&mask, 0.9).expect("should filter");
    assert_eq!(sparse.lower_b.len(), 2);
    assert!(sparse.lower_a.unstable_idx.is_some());

    let dense_sparse = sparse.to_dense()?;
    let out_dim = out_c * out_h * out_w;
    assert_eq!(dense_sparse.num_outputs(), out_dim);

    let in_dim = in_c * in_h * in_w;
    for j in 0..in_dim {
        assert!(
            (dense_sparse.lower_a()[[0, j]] - dense_full.lower_a()[[0, j]]).abs() < 1e-9,
            "row 0, col {j}: sparse {}, full {}",
            dense_sparse.lower_a()[[0, j]],
            dense_full.lower_a()[[0, j]]
        );
        assert!(
            (dense_sparse.lower_a()[[2, j]] - dense_full.lower_a()[[2, j]]).abs() < 1e-9,
            "row 2, col {j}: sparse {}, full {}",
            dense_sparse.lower_a()[[2, j]],
            dense_full.lower_a()[[2, j]]
        );
        assert_eq!(dense_sparse.lower_a()[[1, j]], 0.0, "row 1 should be zero");
        assert_eq!(dense_sparse.lower_a()[[3, j]], 0.0, "row 3 should be zero");
    }

    assert_eq!(dense_sparse.lower_b()[[0]], 1.0);
    assert_eq!(dense_sparse.lower_b()[[2]], 3.0);
    assert_eq!(dense_sparse.lower_b()[[1]], 0.0);
    assert_eq!(dense_sparse.lower_b()[[3]], 0.0);

    Ok(())
}

#[test]
fn test_filter_to_unstable_all_unstable_returns_none() {
    let plb = PatchesLinearBounds::identity((1, 2, 2), (1, 2, 2));
    let mask = ndarray::Array3::<bool>::from_elem((1, 2, 2), true);
    assert!(plb.filter_to_unstable(&mask, 0.9).is_none());
}

#[test]
fn test_filter_to_unstable_below_sparsity_threshold() {
    let plb = PatchesLinearBounds::identity((1, 3, 3), (1, 3, 3));
    let mut mask = ndarray::Array3::<bool>::from_elem((1, 3, 3), true);
    mask[[0, 0, 0]] = false;
    assert!(plb.filter_to_unstable(&mask, 0.9).is_some());
}

// ---------------------------------------------------------------------------
// Equivalence tests for the optimized scatter / to_dense path (#hotpath).
//
// The optimized scatter functions in `scatter.rs` walk flat row-major slices
// and hoist the per-(oh,ow) unfold geometry out of the oc/row loops. These
// tests assert the optimized output is BIT-IDENTICAL to the pre-optimization
// reference algorithm (naive strided ndarray indexing), across randomized
// kernels / strides / paddings / specs and all four scatter variants, plus the
// full `to_dense` end-to-end conversion.
// ---------------------------------------------------------------------------
use super::scatter::{
    compute_unfold_index_map, scatter_rows_with_unfold_map, scatter_sparse_rows_with_unfold_map,
    scatter_sparse_with_unfold_map, scatter_with_unfold_map,
};
use ndarray::Array2;

/// Tiny deterministic xorshift PRNG (no external rng dependency).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Random f32 in [-1, 1), quantized to a few bits so repeated overlapping
    /// scatters into the same cell produce non-trivial summation order effects.
    fn next_f32(&mut self) -> f32 {
        let v = (self.next_u64() >> 40) as i32; // 24-bit
        (v - (1 << 23)) as f32 / (1 << 23) as f32
    }
    fn next_range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo + 1)
    }
}

fn random_arr(rng: &mut Rng, shape: &[usize]) -> ArrayD<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    ArrayD::from_shape_vec(IxDyn(shape), data).unwrap()
}

/// Reference (pre-optimization) dense scatter — naive strided ndarray indexing.
#[allow(clippy::too_many_arguments)]
fn ref_scatter(
    dense: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) {
    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let out_flat = oc * out_h * out_w + oh * out_w + ow;
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let idx_1based = index_map[[oh, ow, ic, ki, kj]];
                            if idx_1based > 0.0 {
                                let in_flat = (idx_1based as usize) - 1;
                                dense[[out_flat, in_flat]] += patches[[oc, oh, ow, ic, ki, kj]];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Reference row-aware dense scatter.
#[allow(clippy::too_many_arguments)]
fn ref_scatter_rows(
    dense: &mut Array2<f32>,
    patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    row_count: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) {
    for row in 0..row_count {
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let idx_1based = index_map[[oh, ow, ic, ki, kj]];
                                if idx_1based > 0.0 {
                                    let in_flat = (idx_1based as usize) - 1;
                                    dense[[row, in_flat]] += patches[[row, oc, oh, ow, ic, ki, kj]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Reference sparse scatter.
#[allow(clippy::too_many_arguments)]
fn ref_scatter_sparse(
    dense: &mut Array2<f32>,
    sparse_patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    idx: &UnstableIdx,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) {
    for (i, ((&c, &h), &w)) in idx
        .channels
        .iter()
        .zip(idx.heights.iter())
        .zip(idx.widths.iter())
        .enumerate()
    {
        let out_flat = c * out_h * out_w + h * out_w + w;
        for ic in 0..in_c {
            for ki in 0..kh {
                for kj in 0..kw {
                    let idx_1based = index_map[[h, w, ic, ki, kj]];
                    if idx_1based > 0.0 {
                        let in_flat = (idx_1based as usize) - 1;
                        dense[[out_flat, in_flat]] += sparse_patches[[i, ic, ki, kj]];
                    }
                }
            }
        }
    }
}

/// Reference row-aware sparse scatter.
#[allow(clippy::too_many_arguments)]
fn ref_scatter_sparse_rows(
    dense: &mut Array2<f32>,
    sparse_patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    row_count: usize,
    idx: &UnstableIdx,
    in_c: usize,
    kh: usize,
    kw: usize,
) {
    for row in 0..row_count {
        for (i, ((&_, &h), &w)) in idx
            .channels
            .iter()
            .zip(idx.heights.iter())
            .zip(idx.widths.iter())
            .enumerate()
        {
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let idx_1based = index_map[[h, w, ic, ki, kj]];
                        if idx_1based > 0.0 {
                            let in_flat = (idx_1based as usize) - 1;
                            dense[[row, in_flat]] += sparse_patches[[row, i, ic, ki, kj]];
                        }
                    }
                }
            }
        }
    }
}

fn assert_arr2_bit_identical(a: &Array2<f32>, b: &Array2<f32>, ctx: &str) {
    assert_eq!(a.shape(), b.shape(), "{ctx}: shape mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{ctx}: element {i} differs: opt={x} ref={y}"
        );
    }
}

/// Compute output spatial extent so the index_map covers the receptive field.
fn unfold_out_dims(
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    padding: (usize, usize, usize, usize),
) -> (usize, usize) {
    ny_tensor::unfold_output_size(in_h, in_w, (kh, kw), stride, padding)
}

#[test]
fn test_scatter_dense_equiv_random() {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    for trial in 0..40 {
        let in_c = rng.next_range(1, 3);
        let in_h = rng.next_range(3, 7);
        let in_w = rng.next_range(3, 7);
        let kh = rng.next_range(1, in_h.min(4));
        let kw = rng.next_range(1, in_w.min(4));
        let sh = rng.next_range(1, 2);
        let sw = rng.next_range(1, 2);
        let pad = rng.next_range(0, 1);
        let stride = (sh, sw);
        let padding = (pad, pad, pad, pad);
        let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
        if out_h == 0 || out_w == 0 {
            continue;
        }
        let out_c = rng.next_range(1, 3);
        let in_dim = in_c * in_h * in_w;

        let patches = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
        let pd = PatchesData {
            coeff_err: None,
            patches: Some(patches.clone()),
            geometry: PatchGeometry::affine(stride, padding),
            identity: false,
            output_shape: (out_c, out_h, out_w),
            input_shape: (in_c, in_h, in_w),
            unstable_idx: None,
        };
        let index_map = compute_unfold_index_map(&pd, kh, kw).unwrap();

        let mut opt = Array2::<f32>::zeros((out_c * out_h * out_w, in_dim));
        let mut reference = Array2::<f32>::zeros((out_c * out_h * out_w, in_dim));
        scatter_with_unfold_map(
            &mut opt,
            &patches,
            &index_map,
            out_c,
            out_h,
            out_w,
            in_c,
            kh,
            kw,
            0,
            out_c * out_h * out_w,
        )
        .unwrap();
        ref_scatter(
            &mut reference,
            &patches,
            &index_map,
            out_c,
            out_h,
            out_w,
            in_c,
            kh,
            kw,
        );
        assert_arr2_bit_identical(&opt, &reference, &format!("dense trial {trial}"));
    }
}

#[test]
fn test_scatter_rows_equiv_random() {
    let mut rng = Rng(0x0bad_c0de_dead_beef);
    for trial in 0..40 {
        let in_c = rng.next_range(1, 3);
        let in_h = rng.next_range(3, 7);
        let in_w = rng.next_range(3, 7);
        let kh = rng.next_range(1, in_h.min(4));
        let kw = rng.next_range(1, in_w.min(4));
        let stride = (rng.next_range(1, 2), rng.next_range(1, 2));
        let pad = rng.next_range(0, 1);
        let padding = (pad, pad, pad, pad);
        let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
        if out_h == 0 || out_w == 0 {
            continue;
        }
        let out_c = rng.next_range(1, 3);
        let row_count = rng.next_range(1, 5);
        let in_dim = in_c * in_h * in_w;

        let patches = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
        let pd = PatchesData {
            coeff_err: None,
            patches: Some(patches.clone()),
            geometry: PatchGeometry::affine(stride, padding),
            identity: false,
            output_shape: (out_c, out_h, out_w),
            input_shape: (in_c, in_h, in_w),
            unstable_idx: None,
        };
        let index_map = compute_unfold_index_map(&pd, kh, kw).unwrap();

        let mut opt = Array2::<f32>::zeros((row_count, in_dim));
        let mut reference = Array2::<f32>::zeros((row_count, in_dim));
        scatter_rows_with_unfold_map(
            &mut opt, &patches, &index_map, 0, row_count, out_c, out_h, out_w, in_c, kh, kw,
        )
        .unwrap();
        ref_scatter_rows(
            &mut reference,
            &patches,
            &index_map,
            row_count,
            out_c,
            out_h,
            out_w,
            in_c,
            kh,
            kw,
        );
        assert_arr2_bit_identical(&opt, &reference, &format!("rows trial {trial}"));
    }
}

#[test]
fn test_scatter_sparse_equiv_random() {
    let mut rng = Rng(0xfeed_face_cafe_babe);
    for trial in 0..40 {
        let in_c = rng.next_range(1, 3);
        let in_h = rng.next_range(3, 7);
        let in_w = rng.next_range(3, 7);
        let kh = rng.next_range(1, in_h.min(4));
        let kw = rng.next_range(1, in_w.min(4));
        let stride = (rng.next_range(1, 2), rng.next_range(1, 2));
        let pad = rng.next_range(0, 1);
        let padding = (pad, pad, pad, pad);
        let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
        if out_h == 0 || out_w == 0 {
            continue;
        }
        let out_c = rng.next_range(1, 3);
        let in_dim = in_c * in_h * in_w;
        let out_dim = out_c * out_h * out_w;

        // Build a random subset of unstable positions.
        let mut channels = Vec::new();
        let mut heights = Vec::new();
        let mut widths = Vec::new();
        for c in 0..out_c {
            for h in 0..out_h {
                for w in 0..out_w {
                    if rng.next_u64() & 1 == 0 {
                        channels.push(c);
                        heights.push(h);
                        widths.push(w);
                    }
                }
            }
        }
        if channels.is_empty() {
            channels.push(0);
            heights.push(0);
            widths.push(0);
        }
        let idx = UnstableIdx {
            channels,
            heights,
            widths,
        };
        let n = idx.len();
        let sparse = random_arr(&mut rng, &[n, in_c, kh, kw]);
        let pd = PatchesData {
            coeff_err: None,
            patches: Some(sparse.clone()),
            geometry: PatchGeometry::affine(stride, padding),
            identity: false,
            output_shape: (out_c, out_h, out_w),
            input_shape: (in_c, in_h, in_w),
            unstable_idx: Some(idx.clone()),
        };
        let index_map = compute_unfold_index_map(&pd, kh, kw).unwrap();

        let mut opt = Array2::<f32>::zeros((out_dim, in_dim));
        let mut reference = Array2::<f32>::zeros((out_dim, in_dim));
        scatter_sparse_with_unfold_map(
            &mut opt, &sparse, &index_map, &idx, out_h, out_w, in_c, kh, kw, 0, out_dim,
        )
        .unwrap();
        ref_scatter_sparse(
            &mut reference,
            &sparse,
            &index_map,
            &idx,
            out_h,
            out_w,
            in_c,
            kh,
            kw,
        );
        assert_arr2_bit_identical(&opt, &reference, &format!("sparse trial {trial}"));
    }
}

#[test]
fn test_scatter_sparse_rows_equiv_random() {
    let mut rng = Rng(0xa5a5_5a5a_1234_4321);
    for trial in 0..40 {
        let in_c = rng.next_range(1, 3);
        let in_h = rng.next_range(3, 7);
        let in_w = rng.next_range(3, 7);
        let kh = rng.next_range(1, in_h.min(4));
        let kw = rng.next_range(1, in_w.min(4));
        let stride = (rng.next_range(1, 2), rng.next_range(1, 2));
        let pad = rng.next_range(0, 1);
        let padding = (pad, pad, pad, pad);
        let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
        if out_h == 0 || out_w == 0 {
            continue;
        }
        let out_c = rng.next_range(1, 3);
        let row_count = rng.next_range(1, 4);
        let in_dim = in_c * in_h * in_w;

        let mut channels = Vec::new();
        let mut heights = Vec::new();
        let mut widths = Vec::new();
        for c in 0..out_c {
            for h in 0..out_h {
                for w in 0..out_w {
                    if rng.next_u64() & 1 == 0 {
                        channels.push(c);
                        heights.push(h);
                        widths.push(w);
                    }
                }
            }
        }
        if channels.is_empty() {
            channels.push(0);
            heights.push(0);
            widths.push(0);
        }
        let idx = UnstableIdx {
            channels,
            heights,
            widths,
        };
        let n = idx.len();
        let sparse = random_arr(&mut rng, &[row_count, n, in_c, kh, kw]);
        let pd = PatchesData {
            coeff_err: None,
            patches: Some(sparse.clone()),
            geometry: PatchGeometry::affine(stride, padding),
            identity: false,
            output_shape: (out_c, out_h, out_w),
            input_shape: (in_c, in_h, in_w),
            unstable_idx: Some(idx.clone()),
        };
        let index_map = compute_unfold_index_map(&pd, kh, kw).unwrap();

        let mut opt = Array2::<f32>::zeros((row_count, in_dim));
        let mut reference = Array2::<f32>::zeros((row_count, in_dim));
        scatter_sparse_rows_with_unfold_map(
            &mut opt, &sparse, &index_map, 0, row_count, &idx, in_c, kh, kw,
        )
        .unwrap();
        ref_scatter_sparse_rows(
            &mut reference,
            &sparse,
            &index_map,
            row_count,
            &idx,
            in_c,
            kh,
            kw,
        );
        assert_arr2_bit_identical(&opt, &reference, &format!("sparse rows trial {trial}"));
    }
}

/// End-to-end: `to_dense()` must produce bit-identical A-matrices versus the
/// reference scatter applied independently, across randomized dense and sparse
/// patches bounds (exercises the full conversion, not just the scatter kernel).
#[test]
fn test_to_dense_equiv_random_dense() -> Result<()> {
    let mut rng = Rng(0x7e57_0011_2233_4455);
    for trial in 0..30 {
        let in_c = rng.next_range(1, 3);
        let in_h = rng.next_range(3, 7);
        let in_w = rng.next_range(3, 7);
        let kh = rng.next_range(1, in_h.min(4));
        let kw = rng.next_range(1, in_w.min(4));
        let stride = (rng.next_range(1, 2), rng.next_range(1, 2));
        let pad = rng.next_range(0, 1);
        let padding = (pad, pad, pad, pad);
        let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
        if out_h == 0 || out_w == 0 {
            continue;
        }
        let out_c = rng.next_range(1, 3);
        let out_dim = out_c * out_h * out_w;
        let in_dim = in_c * in_h * in_w;

        let lower = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
        let upper = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
        let make_pd = |p: &ArrayD<f32>| PatchesData {
            coeff_err: None,
            patches: Some(p.clone()),
            geometry: PatchGeometry::affine(stride, padding),
            identity: false,
            output_shape: (out_c, out_h, out_w),
            input_shape: (in_c, in_h, in_w),
            unstable_idx: None,
        };
        let plb = PatchesLinearBounds {
            row_count: out_dim,
            lower_a: make_pd(&lower),
            lower_b: Array1::zeros(out_dim),
            upper_a: make_pd(&upper),
            upper_b: Array1::zeros(out_dim),
        };
        let dense = plb.to_dense()?;

        let index_map = compute_unfold_index_map(&plb.lower_a, kh, kw)?;
        let mut ref_lower = Array2::<f32>::zeros((out_dim, in_dim));
        let mut ref_upper = Array2::<f32>::zeros((out_dim, in_dim));
        ref_scatter(
            &mut ref_lower,
            &lower,
            &index_map,
            out_c,
            out_h,
            out_w,
            in_c,
            kh,
            kw,
        );
        ref_scatter(
            &mut ref_upper,
            &upper,
            &index_map,
            out_c,
            out_h,
            out_w,
            in_c,
            kh,
            kw,
        );
        assert_arr2_bit_identical(
            dense.lower_a(),
            &ref_lower,
            &format!("to_dense lower {trial}"),
        );
        assert_arr2_bit_identical(
            dense.upper_a(),
            &ref_upper,
            &format!("to_dense upper {trial}"),
        );
    }
    Ok(())
}

#[test]
fn test_sparse_patches_memory_savings() {
    let shape = (2, 4, 4);
    let idx = UnstableIdx {
        channels: vec![0, 0, 0, 0, 1, 1, 1, 1],
        heights: vec![0, 1, 2, 3, 0, 1, 2, 3],
        widths: vec![0, 0, 0, 0, 0, 0, 0, 0],
    };
    let sparse = PatchesLinearBounds::sparse_identity(shape, shape, idx);
    let dense = PatchesLinearBounds::identity(shape, shape);

    assert!(sparse.memory_bytes() < dense.memory_bytes());
}

/// Crash guard: a sparse identity layout whose `(c,h,w)` triple lands outside
/// the declared output grid must return a clean error instead of panicking when
/// `to_dense()` turns it into unchecked flat row indices.
#[test]
fn test_sparse_to_dense_out_of_bounds_index_returns_error_not_panic() {
    let shape = (1, 3, 3); // out grid has 9 positions; valid c<1, h<3, w<3
                           // Channel 5 is past out_c=1 -> flat_index would be out of bounds.
    let idx = UnstableIdx {
        channels: vec![5],
        heights: vec![0],
        widths: vec![0],
    };
    let sparse = PatchesLinearBounds::sparse_identity(shape, shape, idx);
    let result = sparse.to_dense();
    assert!(
        matches!(result, Err(NyError::ShapeMismatch { .. })),
        "expected ShapeMismatch for out-of-bounds sparse index, got {result:?}"
    );
}

/// Crash guard: a sparse layout whose parallel index vectors disagree in length
/// (heights shorter than channels) must error cleanly, since downstream code
/// indexes `heights[i]`/`widths[i]` for every channel index.
#[test]
fn test_sparse_to_dense_ragged_index_vectors_return_error_not_panic() {
    let shape = (1, 3, 3);
    let idx = UnstableIdx {
        channels: vec![0, 0],
        heights: vec![0], // shorter than channels
        widths: vec![0, 1],
    };
    let sparse = PatchesLinearBounds::sparse_identity(shape, shape, idx);
    let result = sparse.to_dense();
    assert!(
        matches!(result, Err(NyError::ShapeMismatch { .. })),
        "expected ShapeMismatch for ragged sparse index vectors, got {result:?}"
    );
}

/// Crash guard: a sparse non-identity layout whose `(c,h,w)` exceeds the output
/// grid must error cleanly. This exercises the scatter path
/// (`materialize_sparse_dense_pair` -> `scatter_sparse_with_unfold_map`) plus
/// `expand_sparse_bias`, all of which index dense rows / bias by the unchecked
/// flat index. Without the guard, `dense.row_mut(out_flat)` panics out of bounds.
#[test]
fn test_sparse_to_dense_nonidentity_out_of_bounds_returns_error_not_panic() {
    let (out_c, out_h, out_w) = (1, 2, 2);
    let (in_c, in_h, in_w) = (1, 3, 3);
    let (kh, kw) = (2, 2);
    // One sparse row whose height (9) is far outside out_h=2.
    let idx = UnstableIdx {
        channels: vec![0],
        heights: vec![9],
        widths: vec![0],
    };
    // 4D sparse patches tensor: (unstable_size=1, in_c, kH, kW).
    let sparse_patches = ArrayD::<f32>::zeros(IxDyn(&[1, in_c, kh, kw]));
    let make_data = || PatchesData {
        coeff_err: None,
        patches: Some(sparse_patches.clone()),
        geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
        identity: false,
        output_shape: (out_c, out_h, out_w),
        input_shape: (in_c, in_h, in_w),
        unstable_idx: Some(idx.clone()),
    };
    let plb = PatchesLinearBounds {
        row_count: out_c * out_h * out_w,
        lower_a: make_data(),
        lower_b: Array1::zeros(1),
        upper_a: make_data(),
        upper_b: Array1::zeros(1),
    };
    let result = plb.to_dense();
    assert!(
        matches!(result, Err(NyError::ShapeMismatch { .. })),
        "expected ShapeMismatch for out-of-bounds non-identity sparse layout, got {result:?}"
    );
}

/// Soundness: the crash guard must NOT reject a well-formed sparse layout — a
/// valid layout still materializes correctly (regression guard so the validation
/// is a pure crash guard with no behavior change on the happy path).
#[test]
fn test_sparse_to_dense_valid_layout_still_succeeds() -> Result<()> {
    let shape = (1, 3, 3);
    let idx = UnstableIdx {
        channels: vec![0, 0, 0, 0],
        heights: vec![0, 1, 1, 2],
        widths: vec![0, 1, 2, 2],
    };
    let sparse = PatchesLinearBounds::sparse_identity(shape, shape, idx);
    let dense = sparse.to_dense()?;
    assert_eq!(dense.num_outputs(), 9);
    assert_eq!(dense.num_inputs(), 9);
    Ok(())
}

/// Small well-formed 4D sparse pair used to exercise the shared lower/upper
/// authentication boundary.  The 2x2 kernel over a 3x3 input has a 2x2 output.
fn sparse_pair_validation_fixture() -> PatchesLinearBounds {
    let idx = UnstableIdx {
        channels: vec![0, 0],
        heights: vec![0, 1],
        widths: vec![0, 1],
    };
    let patches = ArrayD::<f32>::ones(IxDyn(&[2, 1, 2, 2]));
    let side = || PatchesData {
        coeff_err: None,
        patches: Some(patches.clone()),
        geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
        identity: false,
        output_shape: (1, 2, 2),
        input_shape: (1, 3, 3),
        unstable_idx: Some(idx.clone()),
    };
    PatchesLinearBounds {
        row_count: 2,
        lower_a: side(),
        lower_b: Array1::zeros(2),
        upper_a: side(),
        upper_b: Array1::zeros(2),
    }
}

/// Both sparse terminal consumers must reject a malformed pair before their
/// unchecked scatter/concretize loops run.
fn assert_sparse_pair_rejected(bounds: &PatchesLinearBounds, context: &str) {
    let dense = bounds.to_dense();
    assert!(dense.is_err(), "{context}: to_dense unexpectedly succeeded");

    let input = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&[1, 3, 3])),
        ArrayD::ones(IxDyn(&[1, 3, 3])),
    )
    .expect("valid input box");
    let concrete = bounds.concretize_sound_sparse(&input, None);
    assert!(
        concrete.is_err(),
        "{context}: sparse concretize unexpectedly succeeded"
    );
}

#[test]
fn sparse_pair_rejects_missing_or_permuted_upper_index() {
    let mut missing = sparse_pair_validation_fixture();
    missing.upper_a.unstable_idx = None;
    assert_sparse_pair_rejected(&missing, "missing upper unstable_idx");

    let mut permuted = sparse_pair_validation_fixture();
    permuted.upper_a.unstable_idx = Some(UnstableIdx {
        channels: vec![0, 0],
        heights: vec![1, 0],
        widths: vec![1, 0],
    });
    assert_sparse_pair_rejected(&permuted, "permuted upper unstable_idx");
}

#[test]
fn sparse_identity_pair_authenticates_upper_index_and_bias_length() {
    let idx = UnstableIdx {
        channels: vec![0, 0],
        heights: vec![0, 2],
        widths: vec![0, 2],
    };
    let mut permuted = PatchesLinearBounds::sparse_identity((1, 3, 3), (1, 3, 3), idx);
    permuted.upper_a.unstable_idx = Some(UnstableIdx {
        channels: vec![0, 0],
        heights: vec![2, 0],
        widths: vec![2, 0],
    });
    assert_sparse_pair_rejected(&permuted, "sparse identity upper index mismatch");

    let idx = UnstableIdx {
        channels: vec![0, 0],
        heights: vec![0, 2],
        widths: vec![0, 2],
    };
    let mut short_bias = PatchesLinearBounds::sparse_identity((1, 3, 3), (1, 3, 3), idx);
    short_bias.upper_b = Array1::zeros(1);
    assert_sparse_pair_rejected(&short_bias, "sparse identity upper bias length mismatch");
}

#[test]
fn sparse_pair_rejects_upper_geometry_mismatch() {
    let mut bounds = sparse_pair_validation_fixture();
    bounds.upper_a.geometry = PatchGeometry::affine((2, 1), (0, 0, 0, 0));
    assert_sparse_pair_rejected(&bounds, "upper stride mismatch");
}

#[test]
fn sparse_pair_rejects_sparse_axis_or_rank_mismatch() {
    let mut short_axis = sparse_pair_validation_fixture();
    short_axis.lower_a.patches = Some(ArrayD::zeros(IxDyn(&[1, 1, 2, 2])));
    short_axis.upper_a.patches = Some(ArrayD::zeros(IxDyn(&[1, 1, 2, 2])));
    assert_sparse_pair_rejected(&short_axis, "short sparse tensor axis");

    let mut mixed_rank = sparse_pair_validation_fixture();
    mixed_rank.upper_a.patches = Some(ArrayD::zeros(IxDyn(&[2, 2, 1, 2, 2])));
    assert_sparse_pair_rejected(&mixed_rank, "4D/5D lower/upper rank mismatch");
}

#[test]
fn sparse_pair_rejects_duplicate_output_indices() {
    let mut bounds = sparse_pair_validation_fixture();
    let duplicate = UnstableIdx {
        channels: vec![0, 0],
        heights: vec![1, 1],
        widths: vec![0, 0],
    };
    bounds.lower_a.unstable_idx = Some(duplicate.clone());
    bounds.upper_a.unstable_idx = Some(duplicate);
    assert_sparse_pair_rejected(&bounds, "duplicate sparse output index");
}

// ---------------------------------------------------------------------------
// Byte-identity PIN tests for the 7D explicit-rows coeff_err closure
// (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §3.4 T3, §10.4 T2; validation gate
// §13.1). These are committed against the UNMODIFIED tree and must pass
// UNMODIFIED after the closure lands: they pin that every 6D path and every
// err-free value path stays bit-for-bit unchanged.
// ---------------------------------------------------------------------------

fn assert_arr1_bit_identical(a: &Array1<f32>, b: &Array1<f32>, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{ctx}: element {i} differs: {x} vs {y}"
        );
    }
}

fn assert_arrd_bit_identical(a: &ArrayD<f32>, b: &ArrayD<f32>, ctx: &str) {
    assert_eq!(a.shape(), b.shape(), "{ctx}: shape mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{ctx}: element {i} differs: {x} vs {y}"
        );
    }
}

/// LOCKED inline reference for the CURRENT 6D `patches_err_matrix`
/// (`to_dense.rs`): naive strided scatter of per-cell tap `count`/`absacc` into
/// **f32** accumulators — deliberately f32, pinning that the 7D closure does
/// NOT migrate the 6D arm to the new f64 accumulators (spec §3.4 T3(b), R1) —
/// followed by the same valid-certificate second phase. Malformed and
/// non-finite certificates have separate fail-closed tests below.
#[allow(clippy::too_many_arguments)]
fn ref_err_matrix_6d_f32(
    patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    err_row: &Array1<f32>,
    row_count: usize,
    in_dim: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) -> Array2<f32> {
    let mut count = Array2::<f32>::zeros((row_count, in_dim));
    let mut absacc = Array2::<f32>::zeros((row_count, in_dim));
    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let out_flat = oc * out_h * out_w + oh * out_w + ow;
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let idx_1based = index_map[[oh, ow, ic, ki, kj]];
                            if idx_1based > 0.0 {
                                let in_flat = (idx_1based as usize) - 1;
                                count[[out_flat, in_flat]] += 1.0;
                                absacc[[out_flat, in_flat]] +=
                                    patches[[oc, oh, ow, ic, ki, kj]].abs();
                            }
                        }
                    }
                }
            }
        }
    }
    let mut err = Array2::<f32>::zeros((row_count, in_dim));
    for i in 0..row_count {
        let carried = err_row[i];
        assert!(carried.is_finite() && carried >= 0.0);
        let er = f64::from(carried);
        for j in 0..in_dim {
            let c = count[[i, j]];
            if c > 0.0 {
                let gamma = crate::layers::linear::crown_single_gamma_n_f32(c as usize);
                let term = c as f64 * er + gamma * f64::from(absacc[[i, j]]);
                err[[i, j]] = ny_tensor::next_up_f32(term as f32);
            }
        }
    }
    err
}

/// PIN (spec §3.4 T3): 6D `to_dense` behavior is byte-identical before/after
/// the 7D closure.
/// (a) 6D `(None,None)` keeps the exact fast path: both dense errs `None`.
/// (b) 6D-with-err: emitted dense err matrices are bit-identical to the locked
///     inline 6D reference with **f32** accumulators (pins that 6D was NOT
///     migrated to f64), and the dense value path (A-matrices, biases) is
///     bit-identical to the err-free run.
#[test]
fn test_to_dense_6d_behavior_byte_identical() -> Result<()> {
    let mut rng = Rng(0x6d6d_70d3_2153_0001);
    for trial in 0..12 {
        // Trial 0: fixed overlap fixture (in_c=2, 5x5 input, k3x3, stride 1,
        // pad 1 -> out 5x5, out_c=2), padding taps exercised. Rest: randomized
        // geometries from the equivalence-test generator ranges.
        let (in_c, in_h, in_w, kh, kw, stride, padding, out_c) = if trial == 0 {
            (
                2usize,
                5usize,
                5usize,
                3usize,
                3usize,
                (1usize, 1usize),
                (1usize, 1, 1, 1),
                2usize,
            )
        } else {
            let in_c = rng.next_range(1, 3);
            let in_h = rng.next_range(3, 7);
            let in_w = rng.next_range(3, 7);
            let kh = rng.next_range(1, in_h.min(4));
            let kw = rng.next_range(1, in_w.min(4));
            let stride = (rng.next_range(1, 2), rng.next_range(1, 2));
            let pad = rng.next_range(0, 1);
            (
                in_c,
                in_h,
                in_w,
                kh,
                kw,
                stride,
                (pad, pad, pad, pad),
                rng.next_range(1, 3),
            )
        };
        let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
        if out_h == 0 || out_w == 0 {
            continue;
        }
        let out_dim = out_c * out_h * out_w;
        let in_dim = in_c * in_h * in_w;

        let lower = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
        let upper = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
        let lower_bias = Array1::from_shape_fn(out_dim, |_| rng.next_f32());
        let upper_bias = Array1::from_shape_fn(out_dim, |_| rng.next_f32());

        let mut lower_err_row = Array1::<f32>::zeros(out_dim);
        let mut upper_err_row = Array1::<f32>::zeros(out_dim);
        for i in 0..out_dim {
            lower_err_row[i] = match i % 4 {
                0 => 0.0,
                1 => 1e-3,
                2 => 1e-6,
                _ => 3.25e-2,
            };
            upper_err_row[i] = match i % 3 {
                0 => 5e-4,
                1 => 0.0,
                _ => 7.5e-7,
            };
        }

        let make_pd = |p: &ArrayD<f32>, err: Option<Array1<f32>>| PatchesData {
            coeff_err: err,
            patches: Some(p.clone()),
            geometry: PatchGeometry::affine(stride, padding),
            identity: false,
            output_shape: (out_c, out_h, out_w),
            input_shape: (in_c, in_h, in_w),
            unstable_idx: None,
        };
        let make_plb = |le: Option<Array1<f32>>, ue: Option<Array1<f32>>| PatchesLinearBounds {
            row_count: out_dim,
            lower_a: make_pd(&lower, le),
            lower_b: lower_bias.clone(),
            upper_a: make_pd(&upper, ue),
            upper_b: upper_bias.clone(),
        };

        // (a) (None,None) 6D pair stays on the exact fast path: no err emitted.
        let dense_plain = make_plb(None, None).to_dense()?;
        assert!(
            dense_plain.lower_a_err().is_none(),
            "trial {trial}: 6D (None,None) fast path must emit no lower err"
        );
        assert!(
            dense_plain.upper_a_err().is_none(),
            "trial {trial}: 6D (None,None) fast path must emit no upper err"
        );

        // (b) 6D-with-err: value path untouched, errs match the locked f32
        // reference bit-for-bit.
        let dense_err =
            make_plb(Some(lower_err_row.clone()), Some(upper_err_row.clone())).to_dense()?;
        assert_arr2_bit_identical(
            dense_err.lower_a(),
            dense_plain.lower_a(),
            &format!("trial {trial}: err attach must not change lower_a"),
        );
        assert_arr2_bit_identical(
            dense_err.upper_a(),
            dense_plain.upper_a(),
            &format!("trial {trial}: err attach must not change upper_a"),
        );
        assert_arr1_bit_identical(
            dense_err.lower_b(),
            dense_plain.lower_b(),
            &format!("trial {trial}: err attach must not change lower_b"),
        );
        assert_arr1_bit_identical(
            dense_err.upper_b(),
            dense_plain.upper_b(),
            &format!("trial {trial}: err attach must not change upper_b"),
        );

        let index_map = compute_unfold_index_map(&make_pd(&lower, None), kh, kw)?;
        let ref_lower_err = ref_err_matrix_6d_f32(
            &lower,
            &index_map,
            &lower_err_row,
            out_dim,
            in_dim,
            out_c,
            out_h,
            out_w,
            in_c,
            kh,
            kw,
        );
        let ref_upper_err = ref_err_matrix_6d_f32(
            &upper,
            &index_map,
            &upper_err_row,
            out_dim,
            in_dim,
            out_c,
            out_h,
            out_w,
            in_c,
            kh,
            kw,
        );
        assert_arr2_bit_identical(
            dense_err
                .lower_a_err()
                .expect("6D lower Some err must materialize a dense err matrix"),
            &ref_lower_err,
            &format!("trial {trial}: 6D lower err vs locked f32 reference"),
        );
        assert_arr2_bit_identical(
            dense_err
                .upper_a_err()
                .expect("6D upper Some err must materialize a dense err matrix"),
            &ref_upper_err,
            &format!("trial {trial}: 6D upper err vs locked f32 reference"),
        );

        // Mixed (Some, None): the exact 6D None side gets the all-zeros err
        // matrix, the Some side the same reference matrix; values unchanged.
        let dense_mixed = make_plb(Some(lower_err_row.clone()), None).to_dense()?;
        assert_arr2_bit_identical(
            dense_mixed.lower_a(),
            dense_plain.lower_a(),
            &format!("trial {trial}: mixed-err attach must not change lower_a"),
        );
        assert_arr2_bit_identical(
            dense_mixed
                .lower_a_err()
                .expect("mixed pair: Some side must carry err"),
            &ref_lower_err,
            &format!("trial {trial}: mixed pair lower err vs locked f32 reference"),
        );
        assert_arr2_bit_identical(
            dense_mixed
                .upper_a_err()
                .expect("mixed pair: 6D None side materializes exact zeros"),
            &Array2::<f32>::zeros((out_dim, in_dim)),
            &format!("trial {trial}: mixed pair 6D-None upper err must be exact zeros"),
        );
    }
    Ok(())
}

/// PIN (spec §10.4 T2): `from_dense_spatial_rows` value path is byte-identical
/// for err-free vs err-carrying dense sources — patches tensors, biases and
/// metadata bit-for-bit equal; the err-free source keeps `coeff_err: None` on
/// both sides. (The err-carrying source's `coeff_err` is deliberately NOT
/// pinned: the closure attaches `Some(row_max)` there, changing nothing else.)
#[test]
fn from_dense_spatial_rows_no_err_source_byte_identical() -> Result<()> {
    let mut rng = Rng(0xf20d_e45e_5f2a_0001);
    let output_shape = (2usize, 2usize, 3usize);
    let out_dim = output_shape.0 * output_shape.1 * output_shape.2; // 12
    let row_count = 5usize; // spec rows != out_dim: exercises the re-entry case

    let lower_a = Array2::from_shape_fn((row_count, out_dim), |_| rng.next_f32());
    let upper_a = Array2::from_shape_fn((row_count, out_dim), |_| rng.next_f32());
    let lower_b = Array1::from_shape_fn(row_count, |_| rng.next_f32());
    let upper_b = Array1::from_shape_fn(row_count, |_| rng.next_f32());

    let src_plain = LinearBounds::new(lower_a, lower_b, upper_a, upper_b)?;
    let mut src_err = src_plain.clone();
    let lower_err = Array2::from_shape_fn((row_count, out_dim), |(i, j)| match (i + j) % 4 {
        0 => 0.0,
        1 => 1e-3,
        2 => 1e-6,
        _ => 2.5e-2,
    });
    let upper_err = Array2::from_shape_fn((row_count, out_dim), |(i, j)| match (i + 2 * j) % 3 {
        0 => 4.0e-4,
        1 => 0.0,
        _ => 8.5e-7,
    });
    src_err.set_coeff_err(lower_err, upper_err);

    let p_plain = PatchesLinearBounds::from_dense_spatial_rows(&src_plain, output_shape)?;
    let p_err = PatchesLinearBounds::from_dense_spatial_rows(&src_err, output_shape)?;

    // Value path + metadata byte-identical between the two sources.
    assert_eq!(p_plain.row_count, row_count);
    assert_eq!(p_err.row_count, row_count);
    for (pa, pb, side) in [
        (&p_plain.lower_a, &p_err.lower_a, "lower"),
        (&p_plain.upper_a, &p_err.upper_a, "upper"),
    ] {
        assert_arrd_bit_identical(
            pa.patches.as_ref().expect("plain patches tensor"),
            pb.patches.as_ref().expect("err-source patches tensor"),
            &format!("from_dense {side} patches tensor"),
        );
        assert_eq!(pa.geometry, pb.geometry, "{side}: geometry");
        assert_eq!(pa.identity, pb.identity, "{side}: identity flag");
        assert_eq!(pa.output_shape, pb.output_shape, "{side}: output_shape");
        assert_eq!(pa.input_shape, pb.input_shape, "{side}: input_shape");
        assert!(
            pa.unstable_idx.is_none() && pb.unstable_idx.is_none(),
            "{side}: unstable_idx must stay None"
        );
    }
    assert_arr1_bit_identical(&p_plain.lower_b, &p_err.lower_b, "from_dense lower_b");
    assert_arr1_bit_identical(&p_plain.upper_b, &p_err.upper_b, "from_dense upper_b");

    // Biases are bitwise clones of the source bias vectors (zero discharge).
    assert_arr1_bit_identical(&p_plain.lower_b, src_plain.lower_b(), "lower_b == source");
    assert_arr1_bit_identical(&p_plain.upper_b, src_plain.upper_b(), "upper_b == source");

    // The err-free source stays exact: coeff_err None on both sides
    // (rule `E_s = None => coeff_err_s = None`, unchanged by the closure).
    assert!(
        p_plain.lower_a.coeff_err.is_none(),
        "err-free source must keep lower coeff_err None"
    );
    assert!(
        p_plain.upper_a.coeff_err.is_none(),
        "err-free source must keep upper coeff_err None"
    );
    Ok(())
}

/// SITE 1 oracle (spec §10.4 T1): the carried per-spec-row err over-bounds the
/// TRUE deviation of every stored 7D coefficient. Source dense carries inexact
/// f32 coefficients with an exact-f64 ground truth and a valid per-cell
/// certificate `E = next_up(|exact − stored|)`; after re-entry the emitted
/// `coeff_err[r]` must (a) equal the rule `if max_j sanitize(E[r,j]) == 0 { 0 }
/// else next_up(max)` bit-exactly, and (b) cover `|exact[r,j] − stored[r,j]|`
/// for EVERY column (the diagonal coefficients; off-diagonal 7D entries are
/// structural zeros with deviation 0).
#[test]
fn from_dense_spatial_rows_err_carry_covers_true_deviation_oracle() -> Result<()> {
    let output_shape = (2usize, 1usize, 2usize);
    let out_dim = output_shape.0 * output_shape.1 * output_shape.2; // 4
    let row_count = 3usize;

    // Exact-f64 ground-truth coefficients chosen to round lossily to f32.
    let exact_lower = Array2::<f64>::from_shape_fn((row_count, out_dim), |(i, j)| {
        (1.0 + (i as f64) + (j as f64)) / 3.0 - 0.5 * (j as f64)
    });
    let exact_upper = Array2::<f64>::from_shape_fn((row_count, out_dim), |(i, j)| {
        2.0 / (7.0 + i as f64) + 0.1 * (j as f64)
    });
    let stored_lower = exact_lower.mapv(|v| v as f32);
    let stored_upper = exact_upper.mapv(|v| v as f32);
    // Valid per-cell certificate: E[r,j] = next_up(|exact − stored|) >= true dev.
    let e_lower = Array2::<f32>::from_shape_fn((row_count, out_dim), |(i, j)| {
        ny_tensor::next_up_f32((exact_lower[[i, j]] - f64::from(stored_lower[[i, j]])).abs() as f32)
    });
    let e_upper = Array2::<f32>::from_shape_fn((row_count, out_dim), |(i, j)| {
        ny_tensor::next_up_f32((exact_upper[[i, j]] - f64::from(stored_upper[[i, j]])).abs() as f32)
    });

    let mut src = LinearBounds::new(
        stored_lower.clone(),
        Array1::zeros(row_count),
        stored_upper.clone(),
        Array1::zeros(row_count),
    )?;
    src.set_coeff_err(e_lower.clone(), e_upper.clone());

    let p = PatchesLinearBounds::from_dense_spatial_rows(&src, output_shape)?;
    let lerr = p.lower_a.coeff_err.as_ref().expect("lower coeff_err Some");
    let uerr = p.upper_a.coeff_err.as_ref().expect("upper coeff_err Some");
    assert_eq!(lerr.len(), row_count, "lower err len == row_count");
    assert_eq!(uerr.len(), row_count, "upper err len == row_count");

    let mut any_nonzero = false;
    for (e, err_out, exact, stored, side) in [
        (&e_lower, lerr, &exact_lower, &stored_lower, "lower"),
        (&e_upper, uerr, &exact_upper, &stored_upper, "upper"),
    ] {
        for r in 0..row_count {
            // (a) rule: bit-exact per-row max fold (all entries here finite >= 0).
            let mut m = 0.0f32;
            for j in 0..out_dim {
                if e[[r, j]] > m {
                    m = e[[r, j]];
                }
            }
            let expected = if m == 0.0 {
                0.0
            } else {
                ny_tensor::next_up_f32(m)
            };
            assert_eq!(
                err_out[r].to_bits(),
                expected.to_bits(),
                "{side} row {r}: emitted err must equal the rule bit-for-bit"
            );
            if err_out[r] > 0.0 {
                any_nonzero = true;
            }
            // (b) coverage: err_row[r] >= true deviation of every diagonal coeff.
            for j in 0..out_dim {
                let true_dev = (exact[[r, j]] - f64::from(stored[[r, j]])).abs();
                assert!(
                    f64::from(err_out[r]) >= true_dev,
                    "{side} row {r} col {j}: err {} must cover true deviation {}",
                    err_out[r],
                    true_dev
                );
            }
        }
    }
    assert!(
        any_nonzero,
        "fixture must be non-vacuous (some emitted err > 0)"
    );
    Ok(())
}

/// SITE 1 (spec §10.4 T3): a source with err on ONE side only carries err on
/// that side and keeps `None` on the exact side (sides mapped independently).
#[test]
fn from_dense_spatial_rows_one_sided_err() -> Result<()> {
    let output_shape = (2usize, 1usize, 2usize);
    let out_dim = 4usize;
    let row_count = 2usize;
    let mut src = LinearBounds::new(
        Array2::from_shape_fn((row_count, out_dim), |(i, j)| 0.1 * (i + j) as f32),
        Array1::zeros(row_count),
        Array2::from_shape_fn((row_count, out_dim), |(i, j)| 0.2 * (i + j) as f32),
        Array1::zeros(row_count),
    )?;
    // Direct field write: lower side gets a certificate, upper stays exact.
    src.lower_a_err = Some(Array2::from_elem((row_count, out_dim), 3.0e-3f32));

    let p = PatchesLinearBounds::from_dense_spatial_rows(&src, output_shape)?;
    assert!(
        p.lower_a.coeff_err.is_some(),
        "lower side with source err must carry Some"
    );
    assert!(
        p.upper_a.coeff_err.is_none(),
        "upper side (exact source) must stay None"
    );
    let lerr = p.lower_a.coeff_err.as_ref().unwrap();
    assert_eq!(lerr.len(), row_count);
    let expect = ny_tensor::next_up_f32(3.0e-3f32);
    for r in 0..row_count {
        assert_eq!(lerr[r].to_bits(), expect.to_bits(), "row {r} uniform max");
    }
    Ok(())
}

/// SITE 1 (spec §10.4 T4): sanitization — a row with any non-finite/negative
/// entry poisons to `+INF`; a clean row stays finite; `-0.0` does NOT poison
/// (IEEE `-0.0 >= 0.0`).
#[test]
fn from_dense_spatial_rows_err_sanitization() -> Result<()> {
    let output_shape = (2usize, 1usize, 2usize);
    let out_dim = 4usize;
    let row_count = 4usize;
    let mut src = LinearBounds::new(
        Array2::zeros((row_count, out_dim)),
        Array1::zeros(row_count),
        Array2::zeros((row_count, out_dim)),
        Array1::zeros(row_count),
    )?;
    // Row 0: NaN → +INF. Row 1: negative → +INF. Row 2: +INF → +INF.
    // Row 3: clean finite mix incl. -0.0 (must NOT poison) → finite next_up(max).
    let mut e = Array2::<f32>::zeros((row_count, out_dim));
    e[[0, 1]] = f32::NAN;
    e[[1, 2]] = -1.0;
    e[[2, 0]] = f32::INFINITY;
    e[[3, 0]] = -0.0;
    e[[3, 1]] = 5.0e-4;
    src.lower_a_err = Some(e);

    let p = PatchesLinearBounds::from_dense_spatial_rows(&src, output_shape)?;
    let lerr = p.lower_a.coeff_err.as_ref().expect("Some");
    assert!(lerr[0].is_infinite() && lerr[0] > 0.0, "NaN row => +INF");
    assert!(
        lerr[1].is_infinite() && lerr[1] > 0.0,
        "negative row => +INF"
    );
    assert!(lerr[2].is_infinite() && lerr[2] > 0.0, "+INF row => +INF");
    assert_eq!(
        lerr[3].to_bits(),
        ny_tensor::next_up_f32(5.0e-4f32).to_bits(),
        "clean row (with -0.0) stays finite next_up(max), no poison"
    );
    Ok(())
}

/// SITE 1 (spec §10.4 T5): a wrong-shaped source err returns `ShapeMismatch`
/// so the caller skips re-entry and stays Dense (never silently zero-filled).
#[test]
fn from_dense_spatial_rows_err_shape_mismatch_errors() -> Result<()> {
    let output_shape = (2usize, 1usize, 2usize);
    let out_dim = 4usize;
    let row_count = 2usize;
    let mut src = LinearBounds::new(
        Array2::zeros((row_count, out_dim)),
        Array1::zeros(row_count),
        Array2::zeros((row_count, out_dim)),
        Array1::zeros(row_count),
    )?;
    // Wrong column count (out_dim + 1) — bypasses set_coeff_err's shape guard.
    src.lower_a_err = Some(Array2::zeros((row_count, out_dim + 1)));
    let res = PatchesLinearBounds::from_dense_spatial_rows(&src, output_shape);
    assert!(
        matches!(res, Err(NyError::ShapeMismatch { .. })),
        "wrong-shaped source err must return ShapeMismatch, got {res:?}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// SITE I tests — to_dense 7D explicit-rows err materialization
// (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §3.4: T1/T2/T4/T5 f64-oracle,
// T6/T7/T8 guards). The byte-identity pin T3 lives above
// (`test_to_dense_6d_behavior_byte_identical`, committed with the pin set).
// ---------------------------------------------------------------------------
use super::to_dense::patches_err_matrix_rows;

/// Naive strided reference for the 7D err accumulators: per dense cell
/// `(spec_row, in_flat)`, tap `count` and the **f64** sum of absolute tap
/// magnitudes, in the exact `row -> oc -> oh -> ow -> (ic, ki, kj)` tap order
/// of `scatter_rows_err_accumulators`.
#[allow(clippy::too_many_arguments)]
fn ref_rows_count_absacc_f64(
    patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    row_count: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    in_dim: usize,
) -> (Array2<f64>, Array2<f64>) {
    let mut count = Array2::<f64>::zeros((row_count, in_dim));
    let mut absacc = Array2::<f64>::zeros((row_count, in_dim));
    for row in 0..row_count {
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let idx_1based = index_map[[oh, ow, ic, ki, kj]];
                                if idx_1based > 0.0 {
                                    let in_flat = (idx_1based as usize) - 1;
                                    count[[row, in_flat]] += 1.0;
                                    absacc[[row, in_flat]] +=
                                        f64::from(patches[[row, oc, oh, ow, ic, ki, kj]].abs());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (count, absacc)
}

/// Independent naive reference for `patches_err_matrix_rows`: f64 accumulators
/// (naive strided tap walk), identical second phase (sanitize to +INF, γ from
/// the f64 count, `absacc == 0` short-circuit, one outward `next_up_f32`).
#[allow(clippy::too_many_arguments)]
fn ref_err_matrix_rows_f64(
    patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    err_row: Option<&Array1<f32>>,
    row_count: usize,
    in_dim: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) -> Array2<f32> {
    let (count, absacc) = ref_rows_count_absacc_f64(
        patches, index_map, row_count, out_c, out_h, out_w, in_c, kh, kw, in_dim,
    );
    let mut err = Array2::<f32>::zeros((row_count, in_dim));
    for r in 0..row_count {
        let er = match err_row {
            None => 0.0f64,
            Some(e) => {
                let v = e[r];
                if v.is_finite() && v >= 0.0 {
                    f64::from(v)
                } else {
                    f64::INFINITY
                }
            }
        };
        for j in 0..in_dim {
            let c = count[[r, j]];
            if c > 0.0 {
                let gamma = crate::layers::linear::crown_single_gamma_n_f32(c as usize);
                let oe = absacc[[r, j]];
                let acc = if oe == 0.0 { 0.0 } else { gamma * oe };
                let term = c * er + acc;
                err[[r, j]] = ny_tensor::next_up_f32(term as f32);
            }
        }
    }
    err
}

/// f64 oracle: the true dense matrix `T(r,j) = Σ_taps (stored + delta)` for an
/// adversarial true-coefficient deviation pattern within the carried per-row
/// budget `e_r` (`|delta| <= e_r`). Patterns: 0 ⇒ `+e_r` at every tap;
/// 1 ⇒ alternating `±e_r` in tap-scan order; 2 ⇒ `delta = 0` (exact taps).
#[allow(clippy::too_many_arguments)]
fn ref_rows_true_dense_f64(
    patches: &ArrayD<f32>,
    index_map: &ArrayD<f32>,
    err_row: Option<&Array1<f32>>,
    pattern: u8,
    row_count: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    in_dim: usize,
) -> Array2<f64> {
    let mut true_dense = Array2::<f64>::zeros((row_count, in_dim));
    let mut tap_parity = false;
    for row in 0..row_count {
        let er = err_row.map_or(0.0f64, |e| f64::from(e[row]));
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let idx_1based = index_map[[oh, ow, ic, ki, kj]];
                                if idx_1based > 0.0 {
                                    let in_flat = (idx_1based as usize) - 1;
                                    let v = f64::from(patches[[row, oc, oh, ow, ic, ki, kj]]);
                                    let delta = match pattern {
                                        0 => er,
                                        1 => {
                                            if tap_parity {
                                                -er
                                            } else {
                                                er
                                            }
                                        }
                                        _ => 0.0,
                                    };
                                    tap_parity = !tap_parity;
                                    true_dense[[row, in_flat]] += v + delta;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    true_dense
}

/// Build a 7D explicit-rows `PatchesData`.
fn make_pd_7d(
    patches: &ArrayD<f32>,
    err: Option<Array1<f32>>,
    stride: (usize, usize),
    padding: (usize, usize, usize, usize),
    output_shape: (usize, usize, usize),
    input_shape: (usize, usize, usize),
) -> PatchesData {
    PatchesData {
        coeff_err: err,
        patches: Some(patches.clone()),
        geometry: PatchGeometry::affine(stride, padding),
        identity: false,
        output_shape,
        input_shape,
        unstable_idx: None,
    }
}

/// T1 (spec §3.4): the emitted 7D lower err covers the true deviation for
/// three adversarial true-coefficient patterns, on 30 random geometries plus
/// the hand-pinned overlap fixture. Strict `>=` is valid: the f64 oracle noise
/// is >= 2^19 below the γ(1) term.
#[test]
fn test_to_dense_rows_err_covers_true_deviation_oracle() -> Result<()> {
    let mut rng = Rng(0x7d51_0e2a_c0a3_2001);
    for trial in 0..31usize {
        // Trial 0: hand-pinned overlap fixture (interior per-cell count =
        // out_c * kh * kw = 27; asserts real multi-tap accumulation).
        let (in_c, in_h, in_w, kh, kw, stride, padding, out_c, row_count) = if trial == 0 {
            (
                1usize,
                5usize,
                5usize,
                3usize,
                3usize,
                (1usize, 1usize),
                (1usize, 1, 1, 1),
                3usize,
                2usize,
            )
        } else {
            let in_c = rng.next_range(1, 3);
            let in_h = rng.next_range(3, 7);
            let in_w = rng.next_range(3, 7);
            let kh = rng.next_range(1, in_h.min(4));
            let kw = rng.next_range(1, in_w.min(4));
            let stride = (rng.next_range(1, 2), rng.next_range(1, 2));
            let pad = rng.next_range(0, 1);
            (
                in_c,
                in_h,
                in_w,
                kh,
                kw,
                stride,
                (pad, pad, pad, pad),
                rng.next_range(1, 3),
                rng.next_range(1, 5),
            )
        };
        let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
        if out_h == 0 || out_w == 0 {
            continue;
        }
        let in_dim = in_c * in_h * in_w;

        let lower = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
        let upper = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
        let mut err_row = Array1::<f32>::zeros(row_count);
        for r in 0..row_count {
            err_row[r] = match r % 4 {
                0 => 0.0,
                1 => 1e-3,
                2 => 1e-6,
                _ => 3.25e-2,
            };
        }

        let plb = PatchesLinearBounds {
            row_count,
            lower_a: make_pd_7d(
                &lower,
                Some(err_row.clone()),
                stride,
                padding,
                (out_c, out_h, out_w),
                (in_c, in_h, in_w),
            ),
            lower_b: Array1::zeros(row_count),
            upper_a: make_pd_7d(
                &upper,
                None,
                stride,
                padding,
                (out_c, out_h, out_w),
                (in_c, in_h, in_w),
            ),
            upper_b: Array1::zeros(row_count),
        };
        let dense = plb.to_dense()?;
        let lower_err = dense
            .lower_a_err()
            .expect("7D explicit-rows side must emit a dense err matrix");

        let pd_geom = make_pd_7d(
            &lower,
            None,
            stride,
            padding,
            (out_c, out_h, out_w),
            (in_c, in_h, in_w),
        );
        let index_map = compute_unfold_index_map(&pd_geom, kh, kw)?;
        let (count, _absacc) = ref_rows_count_absacc_f64(
            &lower, &index_map, row_count, out_c, out_h, out_w, in_c, kh, kw, in_dim,
        );
        if trial == 0 {
            let maxc = count.iter().cloned().fold(0.0f64, f64::max);
            assert!(
                maxc > 1.0,
                "fixture must exercise multi-tap accumulation, max count = {maxc}"
            );
            assert_eq!(
                maxc, 27.0,
                "interior fixture count must be out_c*kh*kw = 27"
            );
        }

        // Structure: err == 0 exactly where count == 0; err > 0 where
        // count > 0; all entries finite (finite carried errs, sane counts).
        for r in 0..row_count {
            for j in 0..in_dim {
                let e = lower_err[[r, j]];
                assert!(
                    e.is_finite(),
                    "trial {trial}: err[{r},{j}] = {e} not finite"
                );
                if count[[r, j]] == 0.0 {
                    assert_eq!(
                        e.to_bits(),
                        0.0f32.to_bits(),
                        "trial {trial}: err[{r},{j}] must be exactly 0 where no tap lands"
                    );
                } else {
                    assert!(
                        e > 0.0,
                        "trial {trial}: err[{r},{j}] must be > 0 where count > 0"
                    );
                }
            }
        }

        // Coverage oracle: for each adversarial pattern, the emitted err
        // dominates |stored − true| everywhere.
        for pattern in 0..3u8 {
            let true_dense = ref_rows_true_dense_f64(
                &lower,
                &index_map,
                Some(&err_row),
                pattern,
                row_count,
                out_c,
                out_h,
                out_w,
                in_c,
                kh,
                kw,
                in_dim,
            );
            for r in 0..row_count {
                for j in 0..in_dim {
                    let stored = f64::from(dense.lower_a()[[r, j]]);
                    let dev = (stored - true_dense[[r, j]]).abs();
                    assert!(
                        f64::from(lower_err[[r, j]]) >= dev,
                        "trial {trial} pattern {pattern}: err[{r},{j}] = {} < deviation {dev}",
                        lower_err[[r, j]]
                    );
                }
            }
        }
    }
    Ok(())
}

/// T2 (spec §3.4): `patches_err_matrix_rows` is bit-identical to an
/// independent naive strided reference (f64 accumulators, identical second
/// phase, same tap order), for both `Some` and `None` carried err.
#[test]
fn test_patches_err_matrix_rows_matches_naive_reference() -> Result<()> {
    let mut rng = Rng(0x0e22_4a7c_ff64_2002);
    for trial in 0..12 {
        let in_c = rng.next_range(1, 3);
        let in_h = rng.next_range(3, 7);
        let in_w = rng.next_range(3, 7);
        let kh = rng.next_range(1, in_h.min(4));
        let kw = rng.next_range(1, in_w.min(4));
        let stride = (rng.next_range(1, 2), rng.next_range(1, 2));
        let pad = rng.next_range(0, 1);
        let padding = (pad, pad, pad, pad);
        let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
        if out_h == 0 || out_w == 0 {
            continue;
        }
        let out_c = rng.next_range(1, 3);
        let row_count = rng.next_range(1, 5);
        let in_dim = in_c * in_h * in_w;

        let patches = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
        let pd = make_pd_7d(
            &patches,
            None,
            stride,
            padding,
            (out_c, out_h, out_w),
            (in_c, in_h, in_w),
        );
        let index_map = compute_unfold_index_map(&pd, kh, kw)?;
        let mut err_row = Array1::<f32>::zeros(row_count);
        for r in 0..row_count {
            err_row[r] = match r % 4 {
                0 => 0.0,
                1 => 1e-3,
                2 => 7.5e-7,
                _ => 2.5e-2,
            };
        }

        for err_opt in [None, Some(&err_row)] {
            let got = patches_err_matrix_rows(
                &patches, &index_map, err_opt, 0, row_count, in_dim, out_c, out_h, out_w, in_c, kh,
                kw,
            );
            let want = ref_err_matrix_rows_f64(
                &patches, &index_map, err_opt, row_count, in_dim, out_c, out_h, out_w, in_c, kh, kw,
            );
            assert_arr2_bit_identical(
                &got,
                &want,
                &format!("trial {trial} err={:?}", err_opt.is_some()),
            );
        }
    }
    Ok(())
}

/// T4 (spec §3.4, R2): a 7D `(None,None)` pair emits `Some` err on BOTH sides
/// — the γ(N)·S_hat scatter-accumulation rounding exists at `e_r = 0` — with
/// `err > 0` iff `count >= 1`, covered by the exact-tap (delta = 0) oracle.
/// Plus the 1x1/stride-1 re-entry fixture: every cell has `count == out_c` and
/// `err >= γ(out_c)·absacc`.
#[test]
fn test_to_dense_7d_none_none_emits_accumulation_err() -> Result<()> {
    let mut rng = Rng(0x7d40_0e40_0e40_2004);

    // (a) Overlapping random geometry, both sides 7D, no carried err.
    let (in_c, in_h, in_w, kh, kw, out_c, row_count) =
        (2usize, 6usize, 6usize, 3usize, 3usize, 2usize, 3usize);
    let (stride, padding) = ((1usize, 1usize), (1usize, 1, 1, 1));
    let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
    let in_dim = in_c * in_h * in_w;
    let lower = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
    let upper = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
    let plb = PatchesLinearBounds {
        row_count,
        lower_a: make_pd_7d(
            &lower,
            None,
            stride,
            padding,
            (out_c, out_h, out_w),
            (in_c, in_h, in_w),
        ),
        lower_b: Array1::zeros(row_count),
        upper_a: make_pd_7d(
            &upper,
            None,
            stride,
            padding,
            (out_c, out_h, out_w),
            (in_c, in_h, in_w),
        ),
        upper_b: Array1::zeros(row_count),
    };
    let dense = plb.to_dense()?;
    let index_map = compute_unfold_index_map(&plb.lower_a, kh, kw)?;
    for (tensor, dense_a, dense_err, side) in [
        (&lower, dense.lower_a(), dense.lower_a_err(), "lower"),
        (&upper, dense.upper_a(), dense.upper_a_err(), "upper"),
    ] {
        let err =
            dense_err.unwrap_or_else(|| panic!("{side}: 7D (None,None) must still emit Some err"));
        let (count, _absacc) = ref_rows_count_absacc_f64(
            tensor, &index_map, row_count, out_c, out_h, out_w, in_c, kh, kw, in_dim,
        );
        assert!(
            count.iter().any(|&c| c > 1.0),
            "{side}: fixture must have overlapping taps"
        );
        // Exact-tap oracle (delta = 0): err covers pure accumulation rounding.
        let true_dense = ref_rows_true_dense_f64(
            tensor, &index_map, None, 2, row_count, out_c, out_h, out_w, in_c, kh, kw, in_dim,
        );
        for r in 0..row_count {
            for j in 0..in_dim {
                let e = err[[r, j]];
                if count[[r, j]] == 0.0 {
                    assert_eq!(
                        e.to_bits(),
                        0.0f32.to_bits(),
                        "{side}: err[{r},{j}] must be 0 where count == 0"
                    );
                } else {
                    assert!(e > 0.0, "{side}: err[{r},{j}] must be > 0 where count >= 1");
                }
                let dev = (f64::from(dense_a[[r, j]]) - true_dense[[r, j]]).abs();
                assert!(
                    f64::from(e) >= dev,
                    "{side}: err[{r},{j}] = {e} < accumulation deviation {dev}"
                );
            }
        }
    }

    // (b) No-overlap 1x1/stride-1 re-entry fixture (from_dense_spatial_rows):
    // every dense cell receives exactly out_c plan taps (structural zeros for
    // the off-diagonal channels count too, spec I7), so
    // err >= γ(out_c)·absacc everywhere.
    let output_shape = (2usize, 2usize, 3usize);
    let out_dim = output_shape.0 * output_shape.1 * output_shape.2;
    let rows = 4usize;
    let src = LinearBounds::new(
        Array2::from_shape_fn((rows, out_dim), |_| rng.next_f32()),
        Array1::from_shape_fn(rows, |_| rng.next_f32()),
        Array2::from_shape_fn((rows, out_dim), |_| rng.next_f32()),
        Array1::from_shape_fn(rows, |_| rng.next_f32()),
    )?;
    let reentry = PatchesLinearBounds::from_dense_spatial_rows(&src, output_shape)?;
    let rd = reentry.to_dense()?;
    let rerr = rd
        .lower_a_err()
        .expect("re-entry 7D (None,None) must emit Some err");
    let rpatches = reentry.lower_a.patches.as_ref().unwrap();
    let rindex_map = compute_unfold_index_map(&reentry.lower_a, 1, 1)?;
    let (rcount, rabsacc) = ref_rows_count_absacc_f64(
        rpatches,
        &rindex_map,
        rows,
        output_shape.0,
        output_shape.1,
        output_shape.2,
        output_shape.0,
        1,
        1,
        out_dim,
    );
    let gamma_oc = crate::layers::linear::crown_single_gamma_n_f32(output_shape.0);
    for r in 0..rows {
        for j in 0..out_dim {
            assert_eq!(
                rcount[[r, j]],
                output_shape.0 as f64,
                "re-entry: every cell receives exactly out_c plan taps"
            );
            assert!(
                f64::from(rerr[[r, j]]) >= gamma_oc * rabsacc[[r, j]],
                "re-entry: err[{r},{j}] = {} < gamma(out_c)*absacc = {}",
                rerr[[r, j]],
                gamma_oc * rabsacc[[r, j]]
            );
            assert!(rerr[[r, j]] > 0.0, "re-entry: err must be > 0 (count >= 1)");
        }
    }
    Ok(())
}

/// T5 (spec §3.4): mixed 6D/7D pair dispatches PER SIDE — the exact 6D `None`
/// side gets the all-zero err matrix (bit-exact), the 7D `None` side gets the
/// accumulation-rounding err (> 0 wherever a tap lands).
#[test]
fn test_to_dense_mixed_layout_pair_err_dispatch() -> Result<()> {
    let mut rng = Rng(0x5013_ed70_7000_2005);
    let (in_c, in_h, in_w, kh, kw, out_c) = (1usize, 5usize, 5usize, 3usize, 3usize, 1usize);
    let (stride, padding) = ((1usize, 1usize), (1usize, 1, 1, 1));
    let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
    let out_dim = out_c * out_h * out_w;
    let row_count = out_dim; // 6D side requires row_count == out_dim
    let in_dim = in_c * in_h * in_w;

    let lower6 = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
    let upper7 = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
    let plb = PatchesLinearBounds {
        row_count,
        lower_a: PatchesData {
            coeff_err: None,
            patches: Some(lower6),
            geometry: PatchGeometry::affine(stride, padding),
            identity: false,
            output_shape: (out_c, out_h, out_w),
            input_shape: (in_c, in_h, in_w),
            unstable_idx: None,
        },
        lower_b: Array1::zeros(row_count),
        upper_a: make_pd_7d(
            &upper7,
            None,
            stride,
            padding,
            (out_c, out_h, out_w),
            (in_c, in_h, in_w),
        ),
        upper_b: Array1::zeros(row_count),
    };
    let dense = plb.to_dense()?;

    // 6D None side: exact — all-zero err bits.
    assert_arr2_bit_identical(
        dense
            .lower_a_err()
            .expect("mixed pair must attach err matrices"),
        &Array2::<f32>::zeros((row_count, in_dim)),
        "mixed pair: 6D None side must be the exact zeros matrix",
    );

    // 7D None side: accumulation err wherever a tap lands, incl. count > 1.
    let index_map = compute_unfold_index_map(&plb.lower_a, kh, kw)?;
    let (count, _absacc) = ref_rows_count_absacc_f64(
        &upper7, &index_map, row_count, out_c, out_h, out_w, in_c, kh, kw, in_dim,
    );
    assert!(
        count.iter().any(|&c| c > 1.0),
        "fixture must have a multi-tap cell"
    );
    let uerr = dense.upper_a_err().expect("7D side must carry err");
    for r in 0..row_count {
        for j in 0..in_dim {
            if count[[r, j]] == 0.0 {
                assert_eq!(uerr[[r, j]].to_bits(), 0.0f32.to_bits());
            } else {
                assert!(
                    uerr[[r, j]] > 0.0,
                    "7D side err[{r},{j}] must be > 0 where count >= 1"
                );
            }
        }
    }
    Ok(())
}

/// A row-wide certificate has exactly one entry per logical row. Both 6D and
/// 7D materializers reject a malformed length before their infallible scatter
/// kernels, so missing rows can never be reinterpreted as exact-zero error.
#[test]
fn test_to_dense_err_row_length_mismatch_is_error() {
    let mut rng = Rng(0x1e46_7bad_1346_2006);
    let (in_c, in_h, in_w, kh, kw, out_c) = (1usize, 4usize, 4usize, 2usize, 2usize, 2usize);
    let (stride, padding) = ((1usize, 1usize), (0usize, 0, 0, 0));
    let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
    let out_dim = out_c * out_h * out_w;

    // 6D pair, lower err too long.
    let p6 = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
    let make_pd6 = |err: Option<Array1<f32>>| PatchesData {
        coeff_err: err,
        patches: Some(p6.clone()),
        geometry: PatchGeometry::affine(stride, padding),
        identity: false,
        output_shape: (out_c, out_h, out_w),
        input_shape: (in_c, in_h, in_w),
        unstable_idx: None,
    };
    let plb6 = PatchesLinearBounds {
        row_count: out_dim,
        lower_a: make_pd6(Some(Array1::zeros(out_dim + 1))),
        lower_b: Array1::zeros(out_dim),
        upper_a: make_pd6(None),
        upper_b: Array1::zeros(out_dim),
    };
    let res6 = plb6.to_dense();
    assert!(
        matches!(res6, Err(NyError::ShapeMismatch { .. })),
        "6D wrong-length Some err must be a hard ShapeMismatch, got {res6:?}"
    );
    let input6 = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&[in_c, in_h, in_w])),
        ArrayD::ones(IxDyn(&[in_c, in_h, in_w])),
    )
    .expect("valid input box");
    let sparse6 = plb6.concretize_sound_sparse(&input6, None);
    assert!(
        matches!(sparse6, Err(NyError::ShapeMismatch { .. })),
        "sparse 6D concretize must reject the same malformed certificate, got {sparse6:?}"
    );

    // 7D pair, upper err too short.
    let row_count = 3usize;
    let p7 = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
    let plb7 = PatchesLinearBounds {
        row_count,
        lower_a: make_pd_7d(
            &p7,
            None,
            stride,
            padding,
            (out_c, out_h, out_w),
            (in_c, in_h, in_w),
        ),
        lower_b: Array1::zeros(row_count),
        upper_a: make_pd_7d(
            &p7,
            Some(Array1::zeros(row_count - 1)),
            stride,
            padding,
            (out_c, out_h, out_w),
            (in_c, in_h, in_w),
        ),
        upper_b: Array1::zeros(row_count),
    };
    let res7 = plb7.to_dense();
    assert!(
        matches!(res7, Err(NyError::ShapeMismatch { .. })),
        "7D wrong-length Some err must be a hard ShapeMismatch, got {res7:?}"
    );
}

/// Invalid 6D row certificates poison every represented cell outward. This is
/// the 6D counterpart of the 7D I5/R3 test below and specifically prevents
/// `NaN.max(0)` from erasing an unknown error.
#[test]
fn test_to_dense_6d_nonfinite_err_degrades() -> Result<()> {
    let shape = (1usize, 2usize, 2usize);
    let rows = 4usize;
    let patches = ArrayD::from_elem(IxDyn(&[1, 2, 2, 1, 1, 1]), 1.0f32);
    let make_pd = |err: Array1<f32>| PatchesData {
        coeff_err: Some(err),
        patches: Some(patches.clone()),
        geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
        identity: false,
        output_shape: shape,
        input_shape: shape,
        unstable_idx: None,
    };
    let errors = Array1::from_vec(vec![f32::NAN, f32::INFINITY, -1.0, 1e-4]);
    let bounds = PatchesLinearBounds {
        row_count: rows,
        lower_a: make_pd(errors.clone()),
        lower_b: Array1::zeros(rows),
        upper_a: make_pd(errors),
        upper_b: Array1::zeros(rows),
    };
    let dense = bounds.to_dense()?;
    for err in [
        dense.lower_a_err().expect("lower error matrix"),
        dense.upper_a_err().expect("upper error matrix"),
    ] {
        for row in 0..3 {
            assert_eq!(
                err[[row, row]],
                f32::INFINITY,
                "invalid certificate row {row} must poison its represented cell"
            );
        }
        assert!(err.iter().all(|value| !value.is_nan()));
        assert!(err[[3, 3]].is_finite(), "valid row must remain finite");
    }

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0f32),
    )?;
    let sparse = bounds.concretize_sound_sparse(&input, None)?;
    let sparse_lower = sparse.lower().as_slice().expect("contiguous lower");
    let sparse_upper = sparse.upper().as_slice().expect("contiguous upper");
    for row in 0..3 {
        assert_eq!(sparse_lower[row], f32::NEG_INFINITY);
        assert_eq!(sparse_upper[row], f32::INFINITY);
    }
    assert!(sparse_lower[3].is_finite());
    assert!(sparse_upper[3].is_finite());
    Ok(())
}

/// T7 (spec §3.4, I5/R3): non-finite or negative carried err rows degrade to
/// `+INF` at every count>0 cell, matching the 6D policy; finite rows are
/// unaffected.
#[test]
fn test_patches_err_matrix_rows_nonfinite_err_degrades() -> Result<()> {
    let mut rng = Rng(0x404f_1417_e444_2007);
    let (in_c, in_h, in_w, kh, kw, out_c, row_count) =
        (1usize, 5usize, 5usize, 3usize, 3usize, 2usize, 4usize);
    let (stride, padding) = ((1usize, 1usize), (1usize, 1, 1, 1));
    let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
    let in_dim = in_c * in_h * in_w;

    let patches = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
    let pd = make_pd_7d(
        &patches,
        None,
        stride,
        padding,
        (out_c, out_h, out_w),
        (in_c, in_h, in_w),
    );
    let index_map = compute_unfold_index_map(&pd, kh, kw)?;
    let (count, _absacc) = ref_rows_count_absacc_f64(
        &patches, &index_map, row_count, out_c, out_h, out_w, in_c, kh, kw, in_dim,
    );

    // Rows: NaN, +INF, negative — all must poison; row 3 finite.
    let bad = Array1::from(vec![f32::NAN, f32::INFINITY, -1.0f32, 1e-4]);
    let err = patches_err_matrix_rows(
        &patches,
        &index_map,
        Some(&bad),
        0,
        row_count,
        in_dim,
        out_c,
        out_h,
        out_w,
        in_c,
        kh,
        kw,
    );
    for r in 0..3 {
        for j in 0..in_dim {
            if count[[r, j]] > 0.0 {
                assert_eq!(
                    err[[r, j]],
                    f32::INFINITY,
                    "poisoned row {r}: err[{r},{j}] must be +INF at count > 0 (got {})",
                    err[[r, j]]
                );
            } else {
                assert_eq!(err[[r, j]].to_bits(), 0.0f32.to_bits());
            }
            assert!(!err[[r, j]].is_nan(), "err must never be NaN");
        }
    }
    // Finite row is unaffected: rows are independent, so it must be
    // bit-identical to the same computation with only-finite carried errs.
    let clean = Array1::from(vec![0.0f32, 0.0, 0.0, 1e-4]);
    let err_clean = patches_err_matrix_rows(
        &patches,
        &index_map,
        Some(&clean),
        0,
        row_count,
        in_dim,
        out_c,
        out_h,
        out_w,
        in_c,
        kh,
        kw,
    );
    for j in 0..in_dim {
        assert!(err[[3, j]].is_finite(), "finite row must stay finite");
        assert_eq!(
            err[[3, j]].to_bits(),
            err_clean[[3, j]].to_bits(),
            "finite row {j} must be unaffected by poison in other rows"
        );
    }
    Ok(())
}

/// T8 (spec §3.4, B6): a `Some` coeff_err leaking onto the sparse path is a
/// hard error (was: silent drop — the false-VERIFIED direction).
#[test]
fn test_sparse_to_dense_carried_coeff_err_is_error() {
    let shape = (1, 3, 3);
    let idx = UnstableIdx {
        channels: vec![0, 0],
        heights: vec![0, 1],
        widths: vec![0, 2],
    };
    let mut sparse = PatchesLinearBounds::sparse_identity(shape, shape, idx.clone());
    sparse.lower_a.coeff_err = Some(Array1::zeros(2));
    let res = sparse.to_dense();
    assert!(
        matches!(res, Err(NyError::InternalError(_))),
        "sparse lower Some coeff_err must hard-error, got {res:?}"
    );

    let mut sparse_u = PatchesLinearBounds::sparse_identity(shape, shape, idx);
    sparse_u.upper_a.coeff_err = Some(Array1::zeros(2));
    let res_u = sparse_u.to_dense();
    assert!(
        matches!(res_u, Err(NyError::InternalError(_))),
        "sparse upper Some coeff_err must hard-error, got {res_u:?}"
    );
}

/// Identity patches must be exact. A carried certificate is a hard error in
/// every build profile rather than a debug-only assertion and release drop.
#[test]
fn test_identity_to_dense_carried_coeff_err_is_error() {
    let shape = (1, 2, 2);
    let mut plb = PatchesLinearBounds::identity(shape, shape);
    plb.lower_a.coeff_err = Some(Array1::zeros(4));
    let result = plb.to_dense();
    assert!(
        matches!(result, Err(NyError::InternalError(_))),
        "identity coeff_err must hard-error, got {result:?}"
    );
    let input = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&[1, 2, 2])),
        ArrayD::ones(IxDyn(&[1, 2, 2])),
    )
    .expect("valid input box");
    let sparse_result = plb.concretize_sound_sparse(&input, None);
    assert!(
        matches!(sparse_result, Err(NyError::InternalError(_))),
        "sparse identity coeff_err must hard-error, got {sparse_result:?}"
    );
}

// ---------------------------------------------------------------------------
// #patches-row-range: to_dense_rows / concretize_sound_chunked
// ---------------------------------------------------------------------------

use ny_tensor::BoundedTensor;

/// Assert that `part` is bit-identical to rows `[row_start, row_start + n)` of
/// `full` across the A pair, the bias pair, and the (optional) err pair.
fn assert_dense_row_slice(full: &LinearBounds, part: &LinearBounds, row_start: usize, ctx: &str) {
    let n = part.num_outputs();
    assert_eq!(
        part.num_inputs(),
        full.num_inputs(),
        "{ctx}: in_dim mismatch"
    );
    assert!(
        row_start + n <= full.num_outputs(),
        "{ctx}: slice out of range"
    );
    for r in 0..n {
        let fr = row_start + r;
        for j in 0..full.num_inputs() {
            assert_eq!(
                part.lower_a()[[r, j]].to_bits(),
                full.lower_a()[[fr, j]].to_bits(),
                "{ctx}: lower_a[{r},{j}] vs full[{fr},{j}]"
            );
            assert_eq!(
                part.upper_a()[[r, j]].to_bits(),
                full.upper_a()[[fr, j]].to_bits(),
                "{ctx}: upper_a[{r},{j}] vs full[{fr},{j}]"
            );
        }
        assert_eq!(
            part.lower_b()[r].to_bits(),
            full.lower_b()[fr].to_bits(),
            "{ctx}: lower_b[{r}] vs full[{fr}]"
        );
        assert_eq!(
            part.upper_b()[r].to_bits(),
            full.upper_b()[fr].to_bits(),
            "{ctx}: upper_b[{r}] vs full[{fr}]"
        );
    }
    for (side, p_err, f_err) in [
        ("lower", part.lower_a_err(), full.lower_a_err()),
        ("upper", part.upper_a_err(), full.upper_a_err()),
    ] {
        match (p_err, f_err) {
            (None, None) => {}
            (Some(p), Some(f)) => {
                for r in 0..n {
                    let fr = row_start + r;
                    for j in 0..f.ncols() {
                        assert_eq!(
                            p[[r, j]].to_bits(),
                            f[[fr, j]].to_bits(),
                            "{ctx}: {side}_a_err[{r},{j}] vs full[{fr},{j}]"
                        );
                    }
                }
            }
            (p, f) => panic!(
                "{ctx}: {side}_a_err presence differs: part {} full {}",
                p.is_some(),
                f.is_some()
            ),
        }
    }
}

/// #patches-row-range core property for one fixture: `to_dense_rows(r0, r1)`
/// equals rows `[r0, r1)` of `to_dense()` bit-for-bit (A pair, biases, err
/// pair) over full, leading, interior, trailing, unaligned, and single-row
/// ranges — so a partition reassembles the full form exactly — and malformed
/// ranges are rejected.
fn check_row_range_variant(plb: &PatchesLinearBounds, ctx: &str) -> Result<()> {
    let full = plb.to_dense()?;
    let total = full.num_outputs();
    assert!(total >= 3, "{ctx}: fixture too small ({total} rows)");

    // Full range IS to_dense (one code path).
    let all = plb.to_dense_rows(0, total)?;
    assert_dense_row_slice(&full, &all, 0, &format!("{ctx}: full range"));

    let cut_a = total / 3;
    let cut_b = (2 * total) / 3;
    for (r0, r1) in [
        (0, cut_a),
        (cut_a, cut_b),
        (cut_b, total),
        (1, total - 1),
        (total / 2, total / 2 + 1),
    ] {
        let part = plb.to_dense_rows(r0, r1)?;
        assert_eq!(part.num_outputs(), r1 - r0, "{ctx}: [{r0},{r1}) rows");
        assert_dense_row_slice(&full, &part, r0, &format!("{ctx}: range [{r0},{r1})"));
    }

    assert!(
        plb.to_dense_rows(0, total + 1).is_err(),
        "{ctx}: past-end range must be rejected"
    );
    assert!(
        plb.to_dense_rows(2, 1).is_err(),
        "{ctx}: inverted range must be rejected"
    );
    Ok(())
}

/// #patches-row-range: `to_dense_rows` reproduces the exact row slice of
/// `to_dense()` for every materialization variant — 6D dense patches (exact
/// and with carried coeff_err), 7D explicit rows (err matrices always
/// emitted), 4D sparse, 5D sparse explicit rows, dense identity, and sparse
/// identity.
#[test]
fn test_to_dense_rows_slices_match_to_dense_all_variants() -> Result<()> {
    let mut rng = Rng(0x511c_e5f0_44a1_2026);

    // Shared overlapping conv geometry: in 2x5x5, k3x3, stride 1, pad 1 ->
    // out spatial 5x5, out_c = 2 (interior cells receive multiple taps).
    let (in_c, in_h, in_w, kh, kw) = (2usize, 5usize, 5usize, 3usize, 3usize);
    let (stride, padding) = ((1usize, 1usize), (1usize, 1, 1, 1));
    let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
    let out_c = 2usize;
    let out_dim = out_c * out_h * out_w;

    let random_bias = |rng: &mut Rng, n: usize| -> Array1<f32> {
        Array1::from((0..n).map(|_| rng.next_f32()).collect::<Vec<_>>())
    };

    // --- 6D dense patches, exact (no err matrices emitted). ---
    let patches_l = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
    let patches_u = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
    let pd_6d = |patches: &ArrayD<f32>, err: Option<Array1<f32>>| PatchesData {
        coeff_err: err,
        patches: Some(patches.clone()),
        geometry: PatchGeometry::affine(stride, padding),
        identity: false,
        output_shape: (out_c, out_h, out_w),
        input_shape: (in_c, in_h, in_w),
        unstable_idx: None,
    };
    let plb_6d = PatchesLinearBounds {
        row_count: out_dim,
        lower_a: pd_6d(&patches_l, None),
        lower_b: random_bias(&mut rng, out_dim),
        upper_a: pd_6d(&patches_u, None),
        upper_b: random_bias(&mut rng, out_dim),
    };
    check_row_range_variant(&plb_6d, "6D exact")?;

    // --- 6D dense patches with carried coeff_err on both sides. ---
    let err_l = Array1::from(
        (0..out_dim)
            .map(|r| match r % 3 {
                0 => 0.0,
                1 => 1e-4,
                _ => 2.5e-3,
            })
            .collect::<Vec<f32>>(),
    );
    let err_u = Array1::from(
        (0..out_dim)
            .map(|r| if r % 2 == 0 { 5e-6 } else { 1e-3 })
            .collect::<Vec<f32>>(),
    );
    let plb_6d_err = PatchesLinearBounds {
        row_count: out_dim,
        lower_a: pd_6d(&patches_l, Some(err_l)),
        lower_b: plb_6d.lower_b,
        upper_a: pd_6d(&patches_u, Some(err_u)),
        upper_b: plb_6d.upper_b,
    };
    check_row_range_variant(&plb_6d_err, "6D with coeff_err")?;

    // --- 7D explicit spec rows: err matrices emitted on BOTH sides even for
    // the (Some, None) pair (the 7D scatter accumulates multiple taps/cell). ---
    let row_count = 5usize;
    let rows_l = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
    let rows_u = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
    let err_rows = Array1::from(vec![0.0f32, 1e-3, 7.5e-7, 2.5e-2, 0.0]);
    let plb_7d = PatchesLinearBounds {
        row_count,
        lower_a: make_pd_7d(
            &rows_l,
            Some(err_rows),
            stride,
            padding,
            (out_c, out_h, out_w),
            (in_c, in_h, in_w),
        ),
        lower_b: random_bias(&mut rng, row_count),
        upper_a: make_pd_7d(
            &rows_u,
            None,
            stride,
            padding,
            (out_c, out_h, out_w),
            (in_c, in_h, in_w),
        ),
        upper_b: random_bias(&mut rng, row_count),
    };
    check_row_range_variant(&plb_7d, "7D explicit rows")?;

    // --- 4D sparse (broadcast rows over the full output grid). ---
    let idx = UnstableIdx {
        channels: vec![0, 0, 1, 1],
        heights: vec![0, 2, 1, 4],
        widths: vec![1, 3, 0, 4],
    };
    let n = idx.len();
    let sparse_l = random_arr(&mut rng, &[n, in_c, kh, kw]);
    let sparse_u = random_arr(&mut rng, &[n, in_c, kh, kw]);
    let pd_sparse = |patches: &ArrayD<f32>| PatchesData {
        coeff_err: None,
        patches: Some(patches.clone()),
        geometry: PatchGeometry::affine(stride, padding),
        identity: false,
        output_shape: (out_c, out_h, out_w),
        input_shape: (in_c, in_h, in_w),
        unstable_idx: Some(idx.clone()),
    };
    let plb_sparse = PatchesLinearBounds {
        row_count: n,
        lower_a: pd_sparse(&sparse_l),
        lower_b: random_bias(&mut rng, n),
        upper_a: pd_sparse(&sparse_u),
        upper_b: random_bias(&mut rng, n),
    };
    check_row_range_variant(&plb_sparse, "4D sparse")?;

    // --- 5D sparse explicit spec rows. ---
    let srows = 4usize;
    let sparse_rows_l = random_arr(&mut rng, &[srows, n, in_c, kh, kw]);
    let sparse_rows_u = random_arr(&mut rng, &[srows, n, in_c, kh, kw]);
    let plb_sparse_rows = PatchesLinearBounds {
        row_count: srows,
        lower_a: pd_sparse(&sparse_rows_l),
        lower_b: random_bias(&mut rng, srows),
        upper_a: pd_sparse(&sparse_rows_u),
        upper_b: random_bias(&mut rng, srows),
    };
    check_row_range_variant(&plb_sparse_rows, "5D sparse explicit rows")?;

    // --- Dense identity. ---
    let ident_shape = (2usize, 3usize, 3usize);
    let ident_dim = 18usize;
    let mut plb_ident = PatchesLinearBounds::identity(ident_shape, ident_shape);
    plb_ident.lower_b = random_bias(&mut rng, ident_dim);
    plb_ident.upper_b = random_bias(&mut rng, ident_dim);
    check_row_range_variant(&plb_ident, "identity")?;

    // --- Sparse identity. ---
    let sidx = UnstableIdx {
        channels: vec![0, 0, 1],
        heights: vec![0, 2, 1],
        widths: vec![1, 0, 2],
    };
    let mut plb_sident = PatchesLinearBounds::sparse_identity(ident_shape, ident_shape, sidx);
    plb_sident.lower_b = random_bias(&mut rng, 3);
    plb_sident.upper_b = random_bias(&mut rng, 3);
    check_row_range_variant(&plb_sident, "sparse identity")?;

    Ok(())
}

/// #patches-row-range: `concretize_sound_chunked` is bit-identical to the
/// single-shot `to_dense()?.concretize_sound(input)` — per-row independence of
/// the scatter, the certified-err fold (coeff_err set on BOTH sides, so the
/// err-penalty path runs), and the concretize dot products. A 1-byte block
/// budget forces one row per block (minimum clamp); a 4096-byte budget covers
/// multi-row blocks with a ragged tail. An already-expired deadline surfaces
/// as `DeadlineExceeded`, never a partial result.
#[test]
fn test_concretize_sound_chunked_matches_unchunked() -> Result<()> {
    let mut rng = Rng(0xc04c_4e71_2e50_2026);
    let (in_c, in_h, in_w, kh, kw) = (2usize, 5usize, 5usize, 3usize, 3usize);
    let (stride, padding) = ((1usize, 1usize), (1usize, 1, 1, 1));
    let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
    let out_c = 2usize;
    let out_dim = out_c * out_h * out_w;

    let patches_l = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
    let patches_u = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
    let err_l = Array1::from(
        (0..out_dim)
            .map(|r| if r % 3 == 0 { 0.0 } else { 1e-4 })
            .collect::<Vec<f32>>(),
    );
    let err_u = Array1::from_elem(out_dim, 2.5e-5f32);
    let pd = |patches: &ArrayD<f32>, err: Array1<f32>| PatchesData {
        coeff_err: Some(err),
        patches: Some(patches.clone()),
        geometry: PatchGeometry::affine(stride, padding),
        identity: false,
        output_shape: (out_c, out_h, out_w),
        input_shape: (in_c, in_h, in_w),
        unstable_idx: None,
    };
    let bias = |rng: &mut Rng, n: usize| -> Array1<f32> {
        Array1::from((0..n).map(|_| rng.next_f32()).collect::<Vec<_>>())
    };
    let plb = PatchesLinearBounds {
        row_count: out_dim,
        lower_a: pd(&patches_l, err_l),
        lower_b: bias(&mut rng, out_dim),
        upper_a: pd(&patches_u, err_u),
        upper_b: bias(&mut rng, out_dim),
    };

    // Random input box with lower <= upper.
    let mut lo = ArrayD::<f32>::zeros(IxDyn(&[in_c, in_h, in_w]));
    let mut hi = ArrayD::<f32>::zeros(IxDyn(&[in_c, in_h, in_w]));
    for (l, u) in lo.iter_mut().zip(hi.iter_mut()) {
        let a = rng.next_f32();
        let b = rng.next_f32();
        *l = a.min(b);
        *u = a.max(b);
    }
    let input = BoundedTensor::new(lo, hi)?;

    let unchunked = plb.to_dense()?.concretize_sound(&input);
    assert_eq!(unchunked.len(), out_dim);
    for block_bytes in [1usize, 4096] {
        let chunked = plb.concretize_sound_chunked(&input, block_bytes, None)?;
        assert_eq!(chunked.shape(), &[out_dim], "chunked shape");
        for (i, (c, u)) in chunked
            .lower()
            .iter()
            .zip(unchunked.lower().iter())
            .enumerate()
        {
            assert_eq!(
                c.to_bits(),
                u.to_bits(),
                "lower[{i}] (block_bytes={block_bytes})"
            );
        }
        for (i, (c, u)) in chunked
            .upper()
            .iter()
            .zip(unchunked.upper().iter())
            .enumerate()
        {
            assert_eq!(
                c.to_bits(),
                u.to_bits(),
                "upper[{i}] (block_bytes={block_bytes})"
            );
        }
    }

    // An already-expired deadline is a clean DeadlineExceeded (checked before
    // the first block), matching the walk's per-node budget handling.
    let expired = Instant::now();
    let res = plb.concretize_sound_chunked(&input, 1, Some(expired));
    assert!(
        matches!(res, Err(NyError::DeadlineExceeded(_))),
        "expired deadline must surface as DeadlineExceeded, got {res:?}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// #patches-sparse-concretize: concretize_sound_sparse oracle
// ---------------------------------------------------------------------------

/// Build a random input box of shape `(in_c, in_h, in_w)` where column `j` is
/// **zero-width** (lower == upper == fixed) iff `is_fixed(j)`, otherwise a
/// random non-degenerate interval. Models the VGG16 spec structure (k perturbed
/// pixels, the rest fixed).
fn sparse_input_box(
    rng: &mut Rng,
    in_c: usize,
    in_h: usize,
    in_w: usize,
    is_fixed: impl Fn(usize) -> bool,
) -> BoundedTensor {
    let n = in_c * in_h * in_w;
    let mut lo = ArrayD::<f32>::zeros(IxDyn(&[in_c, in_h, in_w]));
    let mut hi = ArrayD::<f32>::zeros(IxDyn(&[in_c, in_h, in_w]));
    {
        let lo_s = lo.as_slice_mut().unwrap();
        let hi_s = hi.as_slice_mut().unwrap();
        for j in 0..n {
            let a = rng.next_f32();
            if is_fixed(j) {
                lo_s[j] = a;
                hi_s[j] = a;
            } else {
                let b = rng.next_f32();
                lo_s[j] = a.min(b);
                hi_s[j] = a.max(b);
            }
        }
    }
    BoundedTensor::new(lo, hi).unwrap()
}

/// Assert `concretize_sound_sparse` is BIT-FOR-BIT identical to
/// `to_dense()?.concretize_sound(input)` (lower and upper) over `input`.
fn assert_sparse_matches_dense(
    plb: &PatchesLinearBounds,
    input: &BoundedTensor,
    ctx: &str,
) -> Result<()> {
    let dense = plb.to_dense()?.concretize_sound(input);
    let sparse = plb.concretize_sound_sparse(input, None)?;
    assert_eq!(sparse.shape(), dense.shape(), "{ctx}: shape mismatch");
    for (i, (s, d)) in sparse.lower().iter().zip(dense.lower().iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            d.to_bits(),
            "{ctx}: lower[{i}] sparse={s} dense={d}"
        );
    }
    for (i, (s, d)) in sparse.upper().iter().zip(dense.upper().iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            d.to_bits(),
            "{ctx}: upper[{i}] sparse={s} dense={d}"
        );
    }
    Ok(())
}

/// Run the bit-identical assertion over several zero-width column patterns
/// (the soundness-proof obligation): all-nonzero, all-fixed-but-k for k in
/// {0, 1, few}, all-fixed, and a random subset.
fn assert_sparse_matches_dense_all_widths(
    plb: &PatchesLinearBounds,
    rng: &mut Rng,
    in_c: usize,
    in_h: usize,
    in_w: usize,
    ctx: &str,
) -> Result<()> {
    let n = in_c * in_h * in_w;
    // all-nonzero-width (full perturbation)
    let full = sparse_input_box(rng, in_c, in_h, in_w, |_| false);
    assert_sparse_matches_dense(plb, &full, &format!("{ctx}: full-width"))?;
    // all-fixed (k = 0 perturbed): every column zero-width
    let none = sparse_input_box(rng, in_c, in_h, in_w, |_| true);
    assert_sparse_matches_dense(plb, &none, &format!("{ctx}: all-fixed"))?;
    // exactly one perturbed pixel (k = 1, the spec1 shape)
    let k1_col = rng.next_range(0, n - 1);
    let k1 = sparse_input_box(rng, in_c, in_h, in_w, |j| j != k1_col);
    assert_sparse_matches_dense(plb, &k1, &format!("{ctx}: k=1"))?;
    // a few perturbed pixels (k = 5)
    let mut perturbed = std::collections::HashSet::new();
    for _ in 0..5 {
        perturbed.insert(rng.next_range(0, n - 1));
    }
    let kfew = sparse_input_box(rng, in_c, in_h, in_w, |j| !perturbed.contains(&j));
    assert_sparse_matches_dense(plb, &kfew, &format!("{ctx}: k=few"))?;
    // random ~50% zero-width subset (precompute the mask so the closure does
    // not double-borrow `rng`).
    let mask: Vec<bool> = (0..n).map(|_| rng.next_u64() & 1 == 0).collect();
    let rand = sparse_input_box(rng, in_c, in_h, in_w, |j| mask[j]);
    assert_sparse_matches_dense(plb, &rand, &format!("{ctx}: random-subset"))?;
    Ok(())
}

/// #patches-sparse-concretize: `concretize_sound_sparse` reproduces
/// `to_dense()?.concretize_sound(input)` BIT-FOR-BIT for every fast-pathed
/// layout — 6D dense (exact and with carried coeff_err on one/both sides), 4D
/// sparse, dense identity, and sparse identity — over random input boxes with a
/// RANDOM SUBSET of zero-width columns (k = 0, 1, few, ~half, and all-nonzero).
/// The zero-width columns provably contribute exactly 0 to the interval radius
/// and their center contribution is carried by the receptive-field point-apply,
/// so the sparse result equals the certified dense result.
#[test]
fn sparse_concretize_matches_dense_bit_identical() -> Result<()> {
    let mut rng = Rng(0x59a3_c0de_2026_0721);

    // Overlapping conv geometry: in 2x5x5, k3x3, stride 1, pad 1 -> out 5x5.
    let (in_c, in_h, in_w, kh, kw) = (2usize, 5usize, 5usize, 3usize, 3usize);
    let (stride, padding) = ((1usize, 1usize), (1usize, 1, 1, 1));
    let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
    let out_c = 2usize;
    let out_dim = out_c * out_h * out_w;

    let bias = |rng: &mut Rng, n: usize| -> Array1<f32> {
        Array1::from((0..n).map(|_| rng.next_f32()).collect::<Vec<_>>())
    };

    let patches_l = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
    let patches_u = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
    let pd_6d = |patches: &ArrayD<f32>, err: Option<Array1<f32>>| PatchesData {
        coeff_err: err,
        patches: Some(patches.clone()),
        geometry: PatchGeometry::affine(stride, padding),
        identity: false,
        output_shape: (out_c, out_h, out_w),
        input_shape: (in_c, in_h, in_w),
        unstable_idx: None,
    };

    // --- 6D dense, exact (no err matrices). ---
    let plb_6d = PatchesLinearBounds {
        row_count: out_dim,
        lower_a: pd_6d(&patches_l, None),
        lower_b: bias(&mut rng, out_dim),
        upper_a: pd_6d(&patches_u, None),
        upper_b: bias(&mut rng, out_dim),
    };
    assert_sparse_matches_dense_all_widths(&plb_6d, &mut rng, in_c, in_h, in_w, "6D exact")?;

    // --- 6D dense with carried coeff_err on BOTH sides. ---
    let err_l = Array1::from(
        (0..out_dim)
            .map(|r| match r % 3 {
                0 => 0.0,
                1 => 1e-4,
                _ => 2.5e-3,
            })
            .collect::<Vec<f32>>(),
    );
    let err_u = Array1::from(
        (0..out_dim)
            .map(|r| if r % 2 == 0 { 5e-6 } else { 1e-3 })
            .collect::<Vec<f32>>(),
    );
    let plb_6d_err = PatchesLinearBounds {
        row_count: out_dim,
        lower_a: pd_6d(&patches_l, Some(err_l.clone())),
        lower_b: plb_6d.lower_b.clone(),
        upper_a: pd_6d(&patches_u, Some(err_u)),
        upper_b: plb_6d.upper_b.clone(),
    };
    assert_sparse_matches_dense_all_widths(
        &plb_6d_err,
        &mut rng,
        in_c,
        in_h,
        in_w,
        "6D coeff_err both",
    )?;

    // --- 6D dense with coeff_err on the LOWER side ONLY (mixed Some/None: the
    //     upper materializes a zero err matrix, so the err penalty pass still
    //     runs). ---
    let plb_6d_err_lo = PatchesLinearBounds {
        row_count: out_dim,
        lower_a: pd_6d(&patches_l, Some(err_l)),
        lower_b: plb_6d.lower_b.clone(),
        upper_a: pd_6d(&patches_u, None),
        upper_b: plb_6d.upper_b.clone(),
    };
    assert_sparse_matches_dense_all_widths(
        &plb_6d_err_lo,
        &mut rng,
        in_c,
        in_h,
        in_w,
        "6D coeff_err lower-only",
    )?;

    // --- 4D sparse (broadcast rows over the full output grid). ---
    let idx = UnstableIdx {
        channels: vec![0, 0, 1, 1],
        heights: vec![0, 2, 1, 4],
        widths: vec![1, 3, 0, 4],
    };
    let nsp = idx.len();
    let sparse_l = random_arr(&mut rng, &[nsp, in_c, kh, kw]);
    let sparse_u = random_arr(&mut rng, &[nsp, in_c, kh, kw]);
    let pd_sparse = |patches: &ArrayD<f32>| PatchesData {
        coeff_err: None,
        patches: Some(patches.clone()),
        geometry: PatchGeometry::affine(stride, padding),
        identity: false,
        output_shape: (out_c, out_h, out_w),
        input_shape: (in_c, in_h, in_w),
        unstable_idx: Some(idx.clone()),
    };
    let plb_sparse = PatchesLinearBounds {
        row_count: nsp,
        lower_a: pd_sparse(&sparse_l),
        lower_b: bias(&mut rng, nsp),
        upper_a: pd_sparse(&sparse_u),
        upper_b: bias(&mut rng, nsp),
    };
    assert_sparse_matches_dense_all_widths(&plb_sparse, &mut rng, in_c, in_h, in_w, "4D sparse")?;

    // --- Dense identity (out == in). ---
    let ident_shape = (2usize, 3usize, 3usize);
    let ident_dim = 18usize;
    let mut plb_ident = PatchesLinearBounds::identity(ident_shape, ident_shape);
    plb_ident.lower_b = bias(&mut rng, ident_dim);
    plb_ident.upper_b = bias(&mut rng, ident_dim);
    assert_sparse_matches_dense_all_widths(&plb_ident, &mut rng, 2, 3, 3, "identity")?;

    // --- Sparse identity. ---
    let sidx = UnstableIdx {
        channels: vec![0, 0, 1],
        heights: vec![0, 2, 1],
        widths: vec![1, 0, 2],
    };
    let mut plb_sident = PatchesLinearBounds::sparse_identity(ident_shape, ident_shape, sidx);
    plb_sident.lower_b = bias(&mut rng, 3);
    plb_sident.upper_b = bias(&mut rng, 3);
    assert_sparse_matches_dense_all_widths(&plb_sident, &mut rng, 2, 3, 3, "sparse identity")?;

    Ok(())
}

/// #patches-sparse-concretize: the sparse bounds SOUNDLY ENCLOSE the exact f64
/// evaluation `A·x + b` for random concrete `x` inside the box, on the k=1
/// (spec1-shaped) fixed-input case. (Bit-identity to the certified dense path is
/// asserted separately; this is an independent, direct soundness witness.)
#[test]
fn sparse_concretize_encloses_exact_eval() -> Result<()> {
    let mut rng = Rng(0x50f7_e2c1_0721_2026);
    let (in_c, in_h, in_w, kh, kw) = (2usize, 5usize, 5usize, 3usize, 3usize);
    let (stride, padding) = ((1usize, 1usize), (1usize, 1, 1, 1));
    let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
    let out_c = 2usize;
    let out_dim = out_c * out_h * out_w;
    let n = in_c * in_h * in_w;

    let bias = |rng: &mut Rng, n: usize| -> Array1<f32> {
        Array1::from((0..n).map(|_| rng.next_f32()).collect::<Vec<_>>())
    };
    let pd_6d = |patches: &ArrayD<f32>| PatchesData {
        coeff_err: None,
        patches: Some(patches.clone()),
        geometry: PatchGeometry::affine(stride, padding),
        identity: false,
        output_shape: (out_c, out_h, out_w),
        input_shape: (in_c, in_h, in_w),
        unstable_idx: None,
    };
    let patches_l = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
    let patches_u = random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw]);
    let plb = PatchesLinearBounds {
        row_count: out_dim,
        lower_a: pd_6d(&patches_l),
        lower_b: bias(&mut rng, out_dim),
        upper_a: pd_6d(&patches_u),
        upper_b: bias(&mut rng, out_dim),
    };
    let dense = plb.to_dense()?; // exact dense A for the direct evaluation

    for _ in 0..8 {
        // k = 1 perturbed pixel; the rest fixed.
        let k1_col = rng.next_range(0, n - 1);
        let input = sparse_input_box(&mut rng, in_c, in_h, in_w, |j| j != k1_col);
        let concrete = plb.concretize_sound_sparse(&input, None)?;
        let flat = input.flatten();
        let lo = flat.lower();
        let lo = lo.as_slice().unwrap();
        let hi = flat.upper();
        let hi = hi.as_slice().unwrap();
        // Sample interior points, incl. both corners on the perturbed pixel.
        for t in 0..6 {
            let mut x = vec![0.0f64; n];
            for (j, xj) in x.iter_mut().enumerate() {
                let frac = match t {
                    0 => 0.0,
                    1 => 1.0,
                    _ => (rng.next_u64() >> 40) as f64 / (1u64 << 24) as f64,
                };
                *xj = lo[j] as f64 + frac * (hi[j] as f64 - lo[j] as f64);
            }
            let cl = concrete.lower();
            let cl = cl.as_slice().unwrap();
            let cu = concrete.upper();
            let cu = cu.as_slice().unwrap();
            for i in 0..out_dim {
                // lower relaxation eval: lower_a·x + lower_b <= true, must be >= sparse.lower.
                let mut yl = dense.lower_b()[i] as f64;
                let mut yu = dense.upper_b()[i] as f64;
                for j in 0..n {
                    yl += dense.lower_a()[[i, j]] as f64 * x[j];
                    yu += dense.upper_a()[[i, j]] as f64 * x[j];
                }
                assert!(
                    cl[i] as f64 <= yl + 1e-6,
                    "row {i}: sparse.lower {} exceeds lower_a·x+b {yl}",
                    cl[i]
                );
                assert!(
                    cu[i] as f64 + 1e-6 >= yu,
                    "row {i}: sparse.upper {} below upper_a·x+b {yu}",
                    cu[i]
                );
            }
        }
    }
    Ok(())
}

/// #patches-sparse-concretize: unsupported layouts (7D explicit-rows, 5D sparse
/// explicit-rows) return `UnsupportedOp` so the caller falls back to the
/// certified dense-chunked path — never a silently wrong or panicking result.
#[test]
fn sparse_concretize_rejects_explicit_rows_layouts() -> Result<()> {
    let mut rng = Rng(0x7e57_fa11_bac6_2026);
    let (in_c, in_h, in_w, kh, kw) = (2usize, 5usize, 5usize, 3usize, 3usize);
    let (stride, padding) = ((1usize, 1usize), (1usize, 1, 1, 1));
    let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
    let out_c = 2usize;

    let bias = |rng: &mut Rng, n: usize| -> Array1<f32> {
        Array1::from((0..n).map(|_| rng.next_f32()).collect::<Vec<_>>())
    };
    let input = sparse_input_box(&mut rng, in_c, in_h, in_w, |_| false);

    // 7D explicit spec rows.
    let row_count = 5usize;
    let rows_l = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
    let rows_u = random_arr(&mut rng, &[row_count, out_c, out_h, out_w, in_c, kh, kw]);
    let plb_7d = PatchesLinearBounds {
        row_count,
        lower_a: make_pd_7d(
            &rows_l,
            None,
            stride,
            padding,
            (out_c, out_h, out_w),
            (in_c, in_h, in_w),
        ),
        lower_b: bias(&mut rng, row_count),
        upper_a: make_pd_7d(
            &rows_u,
            None,
            stride,
            padding,
            (out_c, out_h, out_w),
            (in_c, in_h, in_w),
        ),
        upper_b: bias(&mut rng, row_count),
    };
    assert!(
        matches!(
            plb_7d.concretize_sound_sparse(&input, None),
            Err(NyError::UnsupportedOp(_))
        ),
        "7D explicit-rows must return UnsupportedOp"
    );

    // 5D sparse explicit spec rows.
    let idx = UnstableIdx {
        channels: vec![0, 0, 1, 1],
        heights: vec![0, 2, 1, 4],
        widths: vec![1, 3, 0, 4],
    };
    let nsp = idx.len();
    let srows = 4usize;
    let sparse_rows_l = random_arr(&mut rng, &[srows, nsp, in_c, kh, kw]);
    let sparse_rows_u = random_arr(&mut rng, &[srows, nsp, in_c, kh, kw]);
    let pd_sparse = |patches: &ArrayD<f32>| PatchesData {
        coeff_err: None,
        patches: Some(patches.clone()),
        geometry: PatchGeometry::affine(stride, padding),
        identity: false,
        output_shape: (out_c, out_h, out_w),
        input_shape: (in_c, in_h, in_w),
        unstable_idx: Some(idx.clone()),
    };
    let plb_5d = PatchesLinearBounds {
        row_count: srows,
        lower_a: pd_sparse(&sparse_rows_l),
        lower_b: bias(&mut rng, srows),
        upper_a: pd_sparse(&sparse_rows_u),
        upper_b: bias(&mut rng, srows),
    };
    assert!(
        matches!(
            plb_5d.concretize_sound_sparse(&input, None),
            Err(NyError::UnsupportedOp(_))
        ),
        "5D sparse explicit-rows must return UnsupportedOp"
    );
    Ok(())
}

/// #patches-sparse-concretize: an already-expired deadline surfaces as
/// `DeadlineExceeded`, never a partial or wrong result (matches the chunked
/// path's between-block deadline handling).
#[test]
fn sparse_concretize_honors_expired_deadline() -> Result<()> {
    let mut rng = Rng(0xdead_11fe_2026_0721);
    let (in_c, in_h, in_w, kh, kw) = (2usize, 5usize, 5usize, 3usize, 3usize);
    let (stride, padding) = ((1usize, 1usize), (1usize, 1, 1, 1));
    let (out_h, out_w) = unfold_out_dims(in_h, in_w, kh, kw, stride, padding);
    let out_c = 2usize;
    let out_dim = out_c * out_h * out_w;
    let bias = |rng: &mut Rng, n: usize| -> Array1<f32> {
        Array1::from((0..n).map(|_| rng.next_f32()).collect::<Vec<_>>())
    };
    let plb = PatchesLinearBounds {
        row_count: out_dim,
        lower_a: PatchesData {
            coeff_err: None,
            patches: Some(random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw])),
            geometry: PatchGeometry::affine(stride, padding),
            identity: false,
            output_shape: (out_c, out_h, out_w),
            input_shape: (in_c, in_h, in_w),
            unstable_idx: None,
        },
        lower_b: bias(&mut rng, out_dim),
        upper_a: PatchesData {
            coeff_err: None,
            patches: Some(random_arr(&mut rng, &[out_c, out_h, out_w, in_c, kh, kw])),
            geometry: PatchGeometry::affine(stride, padding),
            identity: false,
            output_shape: (out_c, out_h, out_w),
            input_shape: (in_c, in_h, in_w),
            unstable_idx: None,
        },
        upper_b: bias(&mut rng, out_dim),
    };
    let input = sparse_input_box(&mut rng, in_c, in_h, in_w, |_| false);
    let expired = Instant::now();
    assert!(
        matches!(
            plb.concretize_sound_sparse(&input, Some(expired)),
            Err(NyError::DeadlineExceeded(_))
        ),
        "expired deadline must surface as DeadlineExceeded"
    );
    Ok(())
}

/// #patches-row-range: the Dense->Patches 7D re-entry pair is `out_c` times
/// larger than the dense pair it replaces (VGG16: 250 rows over the 64x224x224
/// conv1 grid is a 411 GB pair — an allocation abort). An over-budget re-entry
/// must be refused with `CpuMemoryExceeded` BEFORE allocating (the caller
/// stays Dense — sound and more precise); the same fixture succeeds once the
/// budget accommodates it.
#[test]
fn test_from_dense_spatial_rows_budget_guard() -> Result<()> {
    let (out_c, out_h, out_w) = (8usize, 32usize, 32usize);
    let out_dim = out_c * out_h * out_w;
    let rows = 40usize;
    let lb = LinearBounds::new(
        Array2::zeros((rows, out_dim)),
        Array1::zeros(rows),
        Array2::zeros((rows, out_dim)),
        Array1::zeros(rows),
    )?;
    // 7D pair: 2 * 40 * 8 * 32 * 32 * 8 * 4 bytes = 20 MiB.
    crate::tests::with_crown_dense_budget_mb("1", || {
        let res = PatchesLinearBounds::from_dense_spatial_rows(&lb, (out_c, out_h, out_w));
        assert!(
            matches!(res, Err(NyError::CpuMemoryExceeded { .. })),
            "over-budget 7D re-entry must be refused, got {:?}",
            res.map(|pb| pb.row_count)
        );
    });
    crate::tests::with_crown_dense_budget_mb("64", || {
        let res = PatchesLinearBounds::from_dense_spatial_rows(&lb, (out_c, out_h, out_w));
        assert!(
            res.is_ok(),
            "in-budget re-entry must succeed: {:?}",
            res.err()
        );
    });
    Ok(())
}
