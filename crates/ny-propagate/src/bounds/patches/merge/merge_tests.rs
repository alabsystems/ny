// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, ArrayD, Axis, IxDyn};

use super::reconcile_padding;
use crate::bounds::patches::types::{PatchesData, UnstableIdx};
use crate::bounds::patches::PatchesLinearBounds;

fn make_simple_patches(
    output_shape: (usize, usize, usize),
    input_shape: (usize, usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize, usize, usize),
    fill_lower: f32,
    fill_upper: f32,
) -> PatchesLinearBounds {
    let (oc, oh, ow) = output_shape;
    let (ic, _, _) = input_shape;
    let (kh, kw) = kernel;
    let row_count = oc * oh * ow;

    PatchesLinearBounds {
        row_count,
        lower_a: PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(
                IxDyn(&[oc, oh, ow, ic, kh, kw]),
                fill_lower,
            )),
            stride,
            padding,
            identity: false,
            output_shape,
            input_shape,
            unstable_idx: None,
        },
        lower_b: Array1::from_elem(row_count, fill_lower),
        upper_a: PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::from_elem(
                IxDyn(&[oc, oh, ow, ic, kh, kw]),
                fill_upper,
            )),
            stride,
            padding,
            identity: false,
            output_shape,
            input_shape,
            unstable_idx: None,
        },
        upper_b: Array1::from_elem(row_count, fill_upper),
    }
}

#[test]
fn test_clone_with_zero_bias() {
    let p = make_simple_patches((2, 3, 3), (2, 5, 5), (3, 3), (1, 1), (1, 1, 1, 1), 0.5, 1.0);
    let zeroed = p.clone_with_zero_bias();
    assert_eq!(zeroed.row_count, p.row_count);
    assert!(zeroed.lower_b.iter().all(|&v| v == 0.0));
    assert!(zeroed.upper_b.iter().all(|&v| v == 0.0));
    assert_eq!(
        zeroed.lower_a.patches.as_ref().unwrap().shape(),
        p.lower_a.patches.as_ref().unwrap().shape()
    );
}

#[test]
fn test_negated_swapped_zero_bias() {
    let p = make_simple_patches((1, 2, 2), (1, 4, 4), (3, 3), (1, 1), (1, 1, 1, 1), 0.5, 1.5);
    let neg = p.negated_swapped_zero_bias();
    assert!(neg.lower_b.iter().all(|&v| v == 0.0));
    assert!(neg.upper_b.iter().all(|&v| v == 0.0));
    assert!(neg
        .lower_a
        .patches
        .as_ref()
        .unwrap()
        .iter()
        .all(|&v| (v - (-1.5)).abs() < 1e-7));
    assert!(neg
        .upper_a
        .patches
        .as_ref()
        .unwrap()
        .iter()
        .all(|&v| (v - (-0.5)).abs() < 1e-7));
}

#[test]
fn test_negated_swapped_with_identity() {
    let p = PatchesLinearBounds::identity((2, 3, 3), (2, 5, 5));
    let neg = p.negated_swapped_zero_bias();
    assert!(!neg.lower_a.identity);
    assert!(!neg.upper_a.identity);
    assert!(neg.lower_a.patches.is_some());
    assert!(neg.upper_a.patches.is_some());
}

#[test]
fn test_try_merge_compatible_patches() {
    let mut a = make_simple_patches(
        (2, 3, 3),
        (2, 5, 5),
        (3, 3),
        (1, 1),
        (1, 1, 1, 1),
        0.25,
        0.75,
    );
    let b = make_simple_patches(
        (2, 3, 3),
        (2, 5, 5),
        (3, 3),
        (1, 1),
        (1, 1, 1, 1),
        0.125,
        0.5,
    );
    assert!(a.try_merge_inplace(&b).unwrap());
    for &v in a.lower_b.iter() {
        assert!(v <= 0.375, "lower bias {v} > exact 0.375");
    }
    for &v in a.upper_b.iter() {
        assert!(v >= 1.25, "upper bias {v} < exact 1.25");
    }
}

#[test]
fn test_try_merge_incompatible_row_count() {
    let mut a = make_simple_patches((2, 3, 3), (2, 5, 5), (3, 3), (1, 1), (1, 1, 1, 1), 0.5, 1.0);
    let b = make_simple_patches((1, 3, 3), (2, 5, 5), (3, 3), (1, 1), (1, 1, 1, 1), 0.5, 1.0);
    assert!(!a.try_merge_inplace(&b).unwrap());
}

#[test]
fn test_try_merge_incompatible_stride() {
    let mut a = make_simple_patches((2, 3, 3), (2, 5, 5), (3, 3), (1, 1), (1, 1, 1, 1), 0.5, 1.0);
    let b = make_simple_patches((2, 3, 3), (2, 5, 5), (3, 3), (2, 2), (1, 1, 1, 1), 0.5, 1.0);
    assert!(!a.try_merge_inplace(&b).unwrap());
}

#[test]
fn test_try_merge_incomparable_padding() {
    let mut a = make_simple_patches((2, 3, 3), (2, 5, 5), (3, 3), (1, 1), (2, 0, 1, 1), 0.5, 1.0);
    let b = make_simple_patches((2, 3, 3), (2, 5, 5), (3, 3), (1, 1), (0, 2, 1, 1), 0.5, 1.0);
    assert!(!a.try_merge_inplace(&b).unwrap());
}

#[test]
fn test_try_merge_padding_dominance() {
    // When one padding dominates, the smaller-padded kernel is zero-padded
    // to match. Both must produce the same shape after padding for merge to succeed.
    // Here a's kernel=5x5 (3+1+1 each side) and b's kernel=3x3 padded to 5x5.
    let mut a = make_simple_patches((1, 2, 2), (1, 4, 4), (5, 5), (1, 1), (1, 1, 1, 1), 0.5, 1.0);
    let b = make_simple_patches((1, 2, 2), (1, 4, 4), (3, 3), (1, 1), (0, 0, 0, 0), 0.5, 1.0);
    assert!(a.try_merge_inplace(&b).unwrap());
    assert_eq!(a.lower_a.padding, (1, 1, 1, 1));
}

#[test]
fn test_try_merge_identity_with_explicit() {
    let mut a = PatchesLinearBounds::identity((2, 3, 3), (2, 5, 5));
    let b = make_simple_patches(
        (2, 3, 3),
        (2, 5, 5),
        (1, 1),
        (1, 1),
        (0, 0, 0, 0),
        0.25,
        0.75,
    );
    assert!(a.try_merge_inplace(&b).unwrap());
    assert!(!a.lower_a.identity);
}

#[test]
fn test_try_merge_mixed_sparse_dense_falls_back() {
    let mut a = make_simple_patches((2, 3, 3), (2, 5, 5), (3, 3), (1, 1), (1, 1, 1, 1), 0.5, 1.0);
    let idx = UnstableIdx {
        channels: vec![0, 1],
        heights: vec![0, 1],
        widths: vec![0, 0],
    };
    let b = PatchesLinearBounds::sparse_identity((2, 3, 3), (2, 5, 5), idx);
    assert!(!a.try_merge_inplace(&b).unwrap());
}

#[test]
fn test_outward_rounding_soundness() {
    let mut a = make_simple_patches((1, 1, 1), (1, 3, 3), (3, 3), (1, 1), (0, 0, 0, 0), 0.1, 0.3);
    let b = make_simple_patches((1, 1, 1), (1, 3, 3), (3, 3), (1, 1), (0, 0, 0, 0), 0.2, 0.7);
    let exact_lower = 0.1_f32 + 0.2;
    let exact_upper = 0.3_f32 + 0.7;
    a.try_merge_inplace(&b).unwrap();
    for &v in a.lower_a.patches.as_ref().unwrap().iter() {
        assert!(v <= exact_lower, "lower {v} > exact {exact_lower}");
    }
    for &v in a.upper_a.patches.as_ref().unwrap().iter() {
        assert!(v >= exact_upper, "upper {v} < exact {exact_upper}");
    }
}

#[test]
fn test_reconcile_padding_equal() {
    assert_eq!(
        reconcile_padding((1, 1, 1, 1), (1, 1, 1, 1)),
        Some((1, 1, 1, 1))
    );
}

#[test]
fn test_reconcile_padding_a_dominates() {
    assert_eq!(
        reconcile_padding((2, 2, 2, 2), (1, 1, 1, 1)),
        Some((2, 2, 2, 2))
    );
}

#[test]
fn test_reconcile_padding_incomparable() {
    assert_eq!(reconcile_padding((2, 0, 1, 1), (0, 2, 1, 1)), None);
}

// ---------------------------------------------------------------------------
// Byte-identity PIN tests for the 7D explicit-rows coeff_err closure
// (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §4.4 T3/T3a; validation gate §13.1).
// Committed against the UNMODIFIED tree and must pass UNMODIFIED after the
// closure lands: they pin that the 6D merge arm (directed outward adds +
// `merge_coeff_err`) stays bit-for-bit unchanged when the 7D arm is added.
// ---------------------------------------------------------------------------

use ny_tensor::{next_down_f32, next_up_f32};

fn assert_arr1_bits_eq(a: &Array1<f32>, b: &Array1<f32>, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{ctx}: element {i} differs: {x} vs {y}"
        );
    }
}

fn assert_arrd_bits_eq(a: &ArrayD<f32>, b: &ArrayD<f32>, ctx: &str) {
    assert_eq!(a.shape(), b.shape(), "{ctx}: shape mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{ctx}: element {i} differs: {x} vs {y}"
        );
    }
}

/// Verbatim inline reference of the CURRENT directed outward elementwise add
/// (`outward_round_add_lower`/`_upper` in merge/mod.rs), locked for the pin.
fn ref_outward_add(a: &ArrayD<f32>, b: &ArrayD<f32>, upper: bool) -> ArrayD<f32> {
    let mut out = a.clone();
    for (o, (&av, &bv)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        let sum = av + bv;
        *o = if sum.is_finite() {
            if upper {
                next_up_f32(sum)
            } else {
                next_down_f32(sum)
            }
        } else if upper {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
    }
    out
}

fn ref_outward_add_1d(a: &Array1<f32>, b: &Array1<f32>, upper: bool) -> Array1<f32> {
    let mut out = a.clone();
    for (o, (&av, &bv)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        let sum = av + bv;
        *o = if sum.is_finite() {
            if upper {
                next_up_f32(sum)
            } else {
                next_down_f32(sum)
            }
        } else if upper {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
    }
    out
}

/// Verbatim inline reference of the CURRENT 6D `merge_coeff_err` body
/// (merge/mod.rs), including the silent `.get(i).unwrap_or(0.0)` reads kept
/// for 6D byte-identity (spec I6): the 6D arm must move VERBATIM into the
/// ndim-6 match arm of the lifted function.
fn ref_merge_coeff_err_6d(
    merged: &ArrayD<f32>,
    a_err: Option<&Array1<f32>>,
    b_err: Option<&Array1<f32>>,
) -> Array1<f32> {
    const U: f64 = 1.0 / (1u64 << 24) as f64;
    let sh = merged.shape();
    let (out_c, out_h, out_w, in_c, kh, kw) = (sh[0], sh[1], sh[2], sh[3], sh[4], sh[5]);
    let mut ne = Array1::<f32>::zeros(out_c * out_h * out_w);
    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let i = oc * out_h * out_w + oh * out_w + ow;
                let mut rowmax = 0.0f64;
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let a = f64::from(merged[[oc, oh, ow, ic, ki, kj]]).abs();
                            if a > rowmax {
                                rowmax = a;
                            }
                        }
                    }
                }
                let ae = a_err.map_or(0.0, |e| f64::from(e.get(i).copied().unwrap_or(0.0)));
                let be = b_err.map_or(0.0, |e| f64::from(e.get(i).copied().unwrap_or(0.0)));
                ne[i] = next_up_f32((ae + be + 3.0 * U * rowmax) as f32);
            }
        }
    }
    ne
}

/// PIN (spec §4.4 T3(a), G5): hardcoded-bit micro-case. Merging 0.1-filled with
/// 0.2-filled 6D patches must produce EXACTLY these bits, before and after the
/// 7D closure:
///   merged lower  = next_down(RN(0.1f32 + 0.2f32)) = 0x3E999999
///   merged upper  = next_up(RN(0.1f32 + 0.2f32))   = 0x3E99999B
///   lower err     = next_up((3·2^-24·|0x3E999999|) as f32) = 0x33666667
///   upper err     = next_up((3·2^-24·|0x3E99999B|) as f32) = 0x33666669
/// The merged-value literals are the spec's hand literals (§4.4 T3); the err
/// literals were confirmed from a run of the unmodified tree per §12 G5.
#[test]
fn test_merge_6d_byte_identical_micro_bits() {
    let mut a = make_simple_patches((1, 2, 2), (1, 4, 4), (3, 3), (1, 1), (1, 1, 1, 1), 0.1, 0.1);
    let b = make_simple_patches((1, 2, 2), (1, 4, 4), (3, 3), (1, 1), (1, 1, 1, 1), 0.2, 0.2);
    assert!(a.try_merge_inplace(&b).unwrap());

    for (i, &v) in a.lower_a.patches.as_ref().unwrap().iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            0x3E99_9999,
            "merged lower coeff {i}: {v} has bits {:#010X}",
            v.to_bits()
        );
    }
    for (i, &v) in a.upper_a.patches.as_ref().unwrap().iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            0x3E99_999B,
            "merged upper coeff {i}: {v} has bits {:#010X}",
            v.to_bits()
        );
    }
    // Biases go through the same directed outward add.
    for (i, &v) in a.lower_b.iter().enumerate() {
        assert_eq!(v.to_bits(), 0x3E99_9999, "merged lower bias {i}: {v}");
    }
    for (i, &v) in a.upper_b.iter().enumerate() {
        assert_eq!(v.to_bits(), 0x3E99_999B, "merged upper bias {i}: {v}");
    }

    // Err formula bits (None+None inputs -> pure 3U·RowMaxAbs term).
    let le = a
        .lower_a
        .coeff_err
        .as_ref()
        .expect("6D merge emits Some even for exact (None+None) inputs");
    let ue = a
        .upper_a
        .coeff_err
        .as_ref()
        .expect("6D merge emits Some even for exact (None+None) inputs");
    assert_eq!(le.len(), 4);
    assert_eq!(ue.len(), 4);
    for (i, &e) in le.iter().enumerate() {
        assert_eq!(
            e.to_bits(),
            0x3366_6667,
            "lower err {i}: {e:e} has bits {:#010X}",
            e.to_bits()
        );
    }
    for (i, &e) in ue.iter().enumerate() {
        assert_eq!(
            e.to_bits(),
            0x3366_6669,
            "upper err {i}: {e:e} has bits {:#010X}",
            e.to_bits()
        );
    }
}

/// PIN (spec §4.4 T3(b)): 6D merge output — merged coefficient tensors, merged
/// biases AND emitted per-row err arrays — is bitwise identical to an inline
/// verbatim copy of the current 6D implementation, for both Some+Some input
/// errs (asymmetric, exercising the ae/be carry) and None+None input errs
/// (intrinsic rounding term only). Mixed-sign non-representable fills.
#[test]
fn test_merge_6d_byte_identical_reference() {
    let output_shape = (2, 2, 2);
    let input_shape = (2, 4, 4);
    let row_count = output_shape.0 * output_shape.1 * output_shape.2; // 8

    let make_filled = |seed: f32, err_seed: Option<f32>| {
        let mut p = make_simple_patches(
            output_shape,
            input_shape,
            (3, 3),
            (1, 1),
            (1, 1, 1, 1),
            0.0,
            0.0,
        );
        for (k, v) in p.lower_a.patches.as_mut().unwrap().iter_mut().enumerate() {
            let s = if k % 2 == 0 { 1.0 } else { -1.0 };
            *v = s * (0.017 + seed * ((k % 13) as f32) * 0.093);
        }
        for (k, v) in p.upper_a.patches.as_mut().unwrap().iter_mut().enumerate() {
            let s = if k % 3 == 0 { -1.0 } else { 1.0 };
            *v = s * (0.029 + seed * ((k % 11) as f32) * 0.071);
        }
        for (k, v) in p.lower_b.iter_mut().enumerate() {
            *v = seed * 0.3 - (k as f32) * 0.077;
        }
        for (k, v) in p.upper_b.iter_mut().enumerate() {
            *v = seed * 0.4 + (k as f32) * 0.061;
        }
        if let Some(es) = err_seed {
            p.lower_a.coeff_err = Some(Array1::from_shape_fn(row_count, |i| match i % 4 {
                0 => 0.0,
                1 => es * 1e-3,
                2 => es * 1e-6,
                _ => es * 3.1e-2,
            }));
            p.upper_a.coeff_err = Some(Array1::from_shape_fn(row_count, |i| match i % 3 {
                0 => es * 7e-4,
                1 => 0.0,
                _ => es * 4.3e-7,
            }));
        }
        p
    };

    // Case 1: Some+Some asymmetric input errs. Case 2: None+None (the 6D
    // always-Some intrinsic-rounding behavior).
    for (case, a_err_seed, b_err_seed) in [(1, Some(1.0), Some(0.37)), (2, None, None)] {
        let mut a = make_filled(0.83, a_err_seed);
        let b = make_filled(-0.41, b_err_seed);

        // Snapshot the operands (equal paddings -> no pad step in the merge).
        let a_low = a.lower_a.patches.clone().unwrap();
        let a_up = a.upper_a.patches.clone().unwrap();
        let b_low = b.lower_a.patches.clone().unwrap();
        let b_up = b.upper_a.patches.clone().unwrap();
        let a_low_err = a.lower_a.coeff_err.clone();
        let a_up_err = a.upper_a.coeff_err.clone();
        let b_low_err = b.lower_a.coeff_err.clone();
        let b_up_err = b.upper_a.coeff_err.clone();
        let a_lb = a.lower_b.clone();
        let a_ub = a.upper_b.clone();

        assert!(
            a.try_merge_inplace(&b).unwrap(),
            "case {case}: merge failed"
        );

        let ref_ml = ref_outward_add(&a_low, &b_low, false);
        let ref_mu = ref_outward_add(&a_up, &b_up, true);
        assert_arrd_bits_eq(
            a.lower_a.patches.as_ref().unwrap(),
            &ref_ml,
            &format!("case {case}: merged lower tensor"),
        );
        assert_arrd_bits_eq(
            a.upper_a.patches.as_ref().unwrap(),
            &ref_mu,
            &format!("case {case}: merged upper tensor"),
        );

        assert_arr1_bits_eq(
            &a.lower_b,
            &ref_outward_add_1d(&a_lb, &b.lower_b, false),
            &format!("case {case}: merged lower bias"),
        );
        assert_arr1_bits_eq(
            &a.upper_b,
            &ref_outward_add_1d(&a_ub, &b.upper_b, true),
            &format!("case {case}: merged upper bias"),
        );

        let ref_le = ref_merge_coeff_err_6d(&ref_ml, a_low_err.as_ref(), b_low_err.as_ref());
        let ref_ue = ref_merge_coeff_err_6d(&ref_mu, a_up_err.as_ref(), b_up_err.as_ref());
        assert_arr1_bits_eq(
            a.lower_a
                .coeff_err
                .as_ref()
                .expect("6D merge always emits Some lower err"),
            &ref_le,
            &format!("case {case}: lower err array"),
        );
        assert_arr1_bits_eq(
            a.upper_a
                .coeff_err
                .as_ref()
                .expect("6D merge always emits Some upper err"),
            &ref_ue,
            &format!("case {case}: upper err array"),
        );
    }
}

// ---------------------------------------------------------------------------
// 7D explicit-rows merge/negate tests
// (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §4.4 T1/T1b/T2/T4/T5/T6; the T3
// byte-identity pins for the 6D arm live above and stay unmodified).
// ---------------------------------------------------------------------------

/// Build a 7D explicit-rows carrier `[rows, out_c, out_h, out_w, in_c, kh, kw]`
/// mirroring the metadata `PatchesLinearBounds::from_dense_spatial_rows`
/// produces (stride `(1,1)`, `in_c == out_c`, `input_shape == output_shape`,
/// dense `unstable_idx: None`, `coeff_err: None`), with parameterizable
/// kernel/padding and per-flat-index (row-major) fill functions.
fn make_7d_patches(
    rows: usize,
    output_shape: (usize, usize, usize),
    kernel: (usize, usize),
    padding: (usize, usize, usize, usize),
    lower_fill: impl Fn(usize) -> f32,
    upper_fill: impl Fn(usize) -> f32,
) -> PatchesLinearBounds {
    let (oc, oh, ow) = output_shape;
    let (kh, kw) = kernel;
    let ic = oc;
    let n = rows * oc * oh * ow * ic * kh * kw;
    let shape = IxDyn(&[rows, oc, oh, ow, ic, kh, kw]);
    let lower = ArrayD::from_shape_vec(shape.clone(), (0..n).map(lower_fill).collect())
        .expect("shape/vec length agree");
    let upper = ArrayD::from_shape_vec(shape, (0..n).map(upper_fill).collect())
        .expect("shape/vec length agree");
    let side = |patches: ArrayD<f32>| PatchesData {
        coeff_err: None,
        patches: Some(patches),
        stride: (1, 1),
        padding,
        identity: false,
        output_shape,
        input_shape: output_shape,
        unstable_idx: None,
    };
    PatchesLinearBounds {
        row_count: rows,
        lower_a: side(lower),
        lower_b: Array1::zeros(rows),
        upper_a: side(upper),
        upper_b: Array1::zeros(rows),
    }
}

/// Per-element soundness oracle for one merged 7D side (spec §4.4 T1): the
/// emitted spec-row err must cover the stored merged value's deviation from
/// the exact (f64) sum of the two stored operands PLUS both carried input
/// errs, and the merged value itself must be rounded outward.
#[allow(clippy::too_many_arguments)]
fn check_7d_merged_side_oracle(
    merged: &ArrayD<f32>,
    orig_a: &ArrayD<f32>,
    orig_b: &ArrayD<f32>,
    a_err: Option<&Array1<f32>>,
    b_err: Option<&Array1<f32>>,
    ne: &Array1<f32>,
    upper: bool,
    ctx: &str,
) {
    let rows = merged.shape()[0];
    assert_eq!(
        ne.len(),
        rows,
        "{ctx}: err length must be the spec-row count (I1)"
    );
    for (r, ((m_row, a_row), b_row)) in merged
        .axis_iter(Axis(0))
        .zip(orig_a.axis_iter(Axis(0)))
        .zip(orig_b.axis_iter(Axis(0)))
        .enumerate()
    {
        let er = f64::from(ne[r]);
        assert!(
            er.is_finite() && er >= 0.0,
            "{ctx}: row {r}: err {er} not finite/nonnegative"
        );
        let ae = a_err.map_or(0.0, |e| f64::from(e[r]));
        let be = b_err.map_or(0.0, |e| f64::from(e[r]));
        for (k, (&m, (&av, &bv))) in m_row.iter().zip(a_row.iter().zip(b_row.iter())).enumerate() {
            let exact = f64::from(av) + f64::from(bv);
            if upper {
                assert!(
                    f64::from(m) >= exact,
                    "{ctx}: row {r} elem {k}: merged {m} not outward-above exact {exact}"
                );
            } else {
                assert!(
                    f64::from(m) <= exact,
                    "{ctx}: row {r} elem {k}: merged {m} not outward-below exact {exact}"
                );
            }
            let dev = (f64::from(m) - exact).abs();
            assert!(
                dev + ae + be <= er,
                "{ctx}: row {r} elem {k}: dev {dev:e} + carried {ae:e}+{be:e} > err {er:e}"
            );
        }
    }
}

/// T1 (spec §4.4): Some+Some oracle — rows=3, mixed-sign non-representable
/// fills, asymmetric per-row errs on all four sides. The emitted spec-row err
/// covers deviation + both carried errs; merged values are outward.
#[test]
fn test_merge_7d_some_some_err_oracle() {
    let rows = 3;
    let mut a = make_7d_patches(
        rows,
        (2, 2, 2),
        (2, 2),
        (0, 0, 0, 0),
        |k| {
            let s = if k % 2 == 0 { 1.0f32 } else { -1.0 };
            s * (0.017 + ((k % 13) as f32) * 0.093)
        },
        |k| {
            let s = if k % 3 == 0 { -1.0f32 } else { 1.0 };
            s * (0.029 + ((k % 11) as f32) * 0.071)
        },
    );
    let mut b = make_7d_patches(
        rows,
        (2, 2, 2),
        (2, 2),
        (0, 0, 0, 0),
        |k| 0.1 - ((k % 7) as f32) * 0.033,
        |k| -0.2 + ((k % 5) as f32) * 0.057,
    );
    a.lower_a.coeff_err = Some(Array1::from(vec![0.0f32, 1e-3, 3.1e-2]));
    a.upper_a.coeff_err = Some(Array1::from(vec![7e-4f32, 0.0, 4.3e-7]));
    b.lower_a.coeff_err = Some(Array1::from(vec![2e-5f32, 0.0, 1e-2]));
    b.upper_a.coeff_err = Some(Array1::from(vec![0.0f32, 5e-6, 2.2e-3]));

    let a_low = a.lower_a.patches.clone().unwrap();
    let a_up = a.upper_a.patches.clone().unwrap();
    let b_low = b.lower_a.patches.clone().unwrap();
    let b_up = b.upper_a.patches.clone().unwrap();
    let (al_err, au_err) = (a.lower_a.coeff_err.clone(), a.upper_a.coeff_err.clone());
    let (bl_err, bu_err) = (b.lower_a.coeff_err.clone(), b.upper_a.coeff_err.clone());

    assert!(a.try_merge_inplace(&b).unwrap(), "7D merge must succeed");

    check_7d_merged_side_oracle(
        a.lower_a.patches.as_ref().unwrap(),
        &a_low,
        &b_low,
        al_err.as_ref(),
        bl_err.as_ref(),
        a.lower_a
            .coeff_err
            .as_ref()
            .expect("7D merge emits Some lower err"),
        false,
        "T1 lower",
    );
    check_7d_merged_side_oracle(
        a.upper_a.patches.as_ref().unwrap(),
        &a_up,
        &b_up,
        au_err.as_ref(),
        bu_err.as_ref(),
        a.upper_a
            .coeff_err
            .as_ref()
            .expect("7D merge emits Some upper err"),
        true,
        "T1 upper",
    );
}

/// T1b (spec §4.4): padding-reconcile oracle — 3x3/pad(1,1,1,1) merged with
/// 1x1/pad(0,0,0,0). The padded-in cells (exact sum 0) step exactly one
/// subnormal outward (lower == `-f32::from_bits(1)`), the emitted err is
/// `>= 2^-149` on every row, and the reconciled center cells satisfy the
/// deviation oracle.
#[test]
fn test_merge_7d_padding_reconcile_err_oracle() {
    let rows = 2;
    // a: 3x3 kernel, padding (1,1,1,1); only the kernel center is written
    // (border taps are structural zeros).
    let mut a = make_7d_patches(rows, (1, 2, 2), (3, 3), (1, 1, 1, 1), |_| 0.0, |_| 0.0);
    for side in [&mut a.lower_a, &mut a.upper_a] {
        let p = side.patches.as_mut().unwrap();
        for r in 0..rows {
            for oh in 0..2 {
                for ow in 0..2 {
                    p[[r, 0, oh, ow, 0, 1, 1]] =
                        0.7 + (r as f32) * 0.13 + (oh as f32) * 0.011 + (ow as f32) * 0.003;
                }
            }
        }
    }
    // b: 1x1 kernel, no padding -> zero-padded to 3x3 during the merge.
    let b = make_7d_patches(
        rows,
        (1, 2, 2),
        (1, 1),
        (0, 0, 0, 0),
        |k| 0.1 + (k as f32) * 0.021,
        |k| -0.3 + (k as f32) * 0.017,
    );
    let a_low = a.lower_a.patches.clone().unwrap();
    let a_up = a.upper_a.patches.clone().unwrap();
    let b_low = b.lower_a.patches.clone().unwrap();
    let b_up = b.upper_a.patches.clone().unwrap();

    assert!(
        a.try_merge_inplace(&b).unwrap(),
        "7D pad-merge must succeed"
    );
    assert_eq!(a.lower_a.padding, (1, 1, 1, 1));

    let tiny = f32::from_bits(1); // 2^-149, smallest positive subnormal
    let sides = [
        (
            "lower",
            a.lower_a.patches.as_ref().unwrap(),
            &a_low,
            &b_low,
            a.lower_a
                .coeff_err
                .as_ref()
                .expect("7D merge emits Some lower err"),
            false,
        ),
        (
            "upper",
            a.upper_a.patches.as_ref().unwrap(),
            &a_up,
            &b_up,
            a.upper_a
                .coeff_err
                .as_ref()
                .expect("7D merge emits Some upper err"),
            true,
        ),
    ];
    for (side_name, merged, orig_a, orig_b, err, upper) in sides {
        assert_eq!(merged.shape(), &[rows, 1, 2, 2, 1, 3, 3]);
        assert_eq!(err.len(), rows);
        for r in 0..rows {
            assert!(
                err[r] >= tiny,
                "{side_name} row {r}: err {:e} < 2^-149",
                err[r]
            );
            for oh in 0..2 {
                for ow in 0..2 {
                    for ki in 0..3 {
                        for kj in 0..3 {
                            let m = merged[[r, 0, oh, ow, 0, ki, kj]];
                            if (ki, kj) == (1, 1) {
                                // Reconciled center: a center + b's 1x1 tap.
                                let exact = f64::from(orig_a[[r, 0, oh, ow, 0, 1, 1]])
                                    + f64::from(orig_b[[r, 0, oh, ow, 0, 0, 0]]);
                                let dev = (f64::from(m) - exact).abs();
                                assert!(
                                    dev <= f64::from(err[r]),
                                    "{side_name} row {r} center ({oh},{ow}): \
                                     dev {dev:e} > err {:e}",
                                    err[r]
                                );
                            } else {
                                // Padded-in / structural-zero cell: exact sum 0,
                                // directed round steps one subnormal outward.
                                let want = if upper {
                                    tiny.to_bits()
                                } else {
                                    (-tiny).to_bits()
                                };
                                assert_eq!(
                                    m.to_bits(),
                                    want,
                                    "{side_name} row {r} padded cell ({ki},{kj}): {m:e}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// T2 (spec §4.4, the live-gap test): None+None on the dense 7D path must emit
/// `Some` — the directed outward add rounding is intrinsic and was previously
/// dropped (`coeff_err` forced to None for every non-6D layout, so per-op f32
/// rounding reached the verdict uncertified).
#[test]
fn test_merge_7d_none_none_emits_intrinsic_err() {
    let rows = 3;
    let mut a = make_7d_patches(
        rows,
        (2, 2, 2),
        (2, 2),
        (0, 0, 0, 0),
        |k| 0.1 + (k as f32) * 0.007,
        |k| 0.3 - (k as f32) * 0.011,
    );
    let b = make_7d_patches(
        rows,
        (2, 2, 2),
        (2, 2),
        (0, 0, 0, 0),
        |k| 0.2 - (k as f32) * 0.013,
        |k| 0.7 + (k as f32) * 0.003,
    );
    assert!(a.lower_a.coeff_err.is_none() && a.upper_a.coeff_err.is_none());

    let a_low = a.lower_a.patches.clone().unwrap();
    let a_up = a.upper_a.patches.clone().unwrap();
    let b_low = b.lower_a.patches.clone().unwrap();
    let b_up = b.upper_a.patches.clone().unwrap();

    assert!(a.try_merge_inplace(&b).unwrap());

    let le = a
        .lower_a
        .coeff_err
        .as_ref()
        .expect("None+None 7D merge must still emit Some lower err (intrinsic rounding)");
    let ue = a
        .upper_a
        .coeff_err
        .as_ref()
        .expect("None+None 7D merge must still emit Some upper err (intrinsic rounding)");
    check_7d_merged_side_oracle(
        a.lower_a.patches.as_ref().unwrap(),
        &a_low,
        &b_low,
        None,
        None,
        le,
        false,
        "T2 lower",
    );
    check_7d_merged_side_oracle(
        a.upper_a.patches.as_ref().unwrap(),
        &a_up,
        &b_up,
        None,
        None,
        ue,
        true,
        "T2 upper",
    );
    // Hauser/grid floor: the intrinsic err is >= 2^-149 on every row.
    let tiny = f32::from_bits(1);
    assert!(le.iter().chain(ue.iter()).all(|&e| e >= tiny));
}

/// T4 (spec §4.4): Sub-negate — the per-row err arrays swap lower<->upper
/// BITWISE on the 7D path (negation is exact; each source carries its own
/// err through `negated_swapped_zero_bias`).
#[test]
fn test_negated_swapped_7d_err_swapped_bitwise() {
    let rows = 3;
    let mut p = make_7d_patches(
        rows,
        (1, 2, 2),
        (1, 1),
        (0, 0, 0, 0),
        |k| 0.4 + (k as f32) * 0.019,
        |k| 0.9 - (k as f32) * 0.023,
    );
    let le = Array1::from(vec![1.5e-4f32, 0.0, 3.25e-2]);
    let ue = Array1::from(vec![0.0f32, 2.75e-5, 6.5e-3]);
    p.lower_a.coeff_err = Some(le.clone());
    p.upper_a.coeff_err = Some(ue.clone());
    let low_vals = p.lower_a.patches.clone().unwrap();
    let up_vals = p.upper_a.patches.clone().unwrap();

    let neg = p.negated_swapped_zero_bias();

    assert_arr1_bits_eq(
        neg.lower_a
            .coeff_err
            .as_ref()
            .expect("negated 7D lower must carry the old upper err"),
        &ue,
        "neg lower err = old upper err",
    );
    assert_arr1_bits_eq(
        neg.upper_a
            .coeff_err
            .as_ref()
            .expect("negated 7D upper must carry the old lower err"),
        &le,
        "neg upper err = old lower err",
    );
    // Values: exact sign flip of the swapped source, bitwise.
    for (k, (&nv, &ov)) in neg
        .lower_a
        .patches
        .as_ref()
        .unwrap()
        .iter()
        .zip(up_vals.iter())
        .enumerate()
    {
        assert_eq!(nv.to_bits(), (-ov).to_bits(), "neg lower elem {k}");
    }
    for (k, (&nv, &ov)) in neg
        .upper_a
        .patches
        .as_ref()
        .unwrap()
        .iter()
        .zip(low_vals.iter())
        .enumerate()
    {
        assert_eq!(nv.to_bits(), (-ov).to_bits(), "neg upper elem {k}");
    }
    assert!(neg.lower_b.iter().all(|&v| v == 0.0));
    assert!(neg.upper_b.iter().all(|&v| v == 0.0));
}

/// T4 (spec §4.4, sparse scope guard): negate on a sparse (unstable_idx Some)
/// carrier keeps `coeff_err = None` even if a `Some` was (illegally) attached
/// — sparse stays outside the err channel (I2).
#[test]
fn test_negated_sparse_err_stays_none() {
    let idx = UnstableIdx {
        channels: vec![0, 1],
        heights: vec![0, 1],
        widths: vec![0, 0],
    };
    let sparse = PatchesData {
        coeff_err: Some(Array1::from(vec![1e-3f32, 2e-3])),
        patches: Some(ArrayD::from_elem(IxDyn(&[2, 2, 1, 1]), 0.5f32)),
        stride: (1, 1),
        padding: (0, 0, 0, 0),
        identity: false,
        output_shape: (2, 2, 2),
        input_shape: (2, 2, 2),
        unstable_idx: Some(idx),
    };
    let neg = sparse.negated();
    assert!(
        neg.coeff_err.is_none(),
        "sparse negate must keep coeff_err None"
    );
}

/// T5 (spec §4.4, alignment gate): 7D tensors whose spec-row axis disagrees
/// with the carrier `row_count` are rejected with `Ok(false)` BEFORE any
/// mutation. Both carriers here have matching tensor shapes (shape[0]=4) but
/// claim row_count=3 — without the gate the merge would succeed with
/// mis-indexed row accounting.
#[test]
fn test_merge_7d_gate_rejects_row_count_mismatch() {
    let mk = || {
        let mut p = make_7d_patches(
            4,
            (1, 2, 2),
            (1, 1),
            (0, 0, 0, 0),
            |k| 0.1 + (k as f32) * 0.01,
            |k| 0.5 - (k as f32) * 0.01,
        );
        p.row_count = 3;
        p.lower_b = Array1::zeros(3);
        p.upper_b = Array1::zeros(3);
        p
    };
    let mut a = mk();
    let b = mk();
    let snap_low = a.lower_a.patches.clone().unwrap();
    let snap_up = a.upper_a.patches.clone().unwrap();

    assert!(
        !a.try_merge_inplace(&b).unwrap(),
        "gate must reject spec-row axis != row_count"
    );
    // Rejection happens before take_or_materialize_pair: self is unmutated.
    assert_arrd_bits_eq(
        a.lower_a.patches.as_ref().unwrap(),
        &snap_low,
        "self lower unmutated",
    );
    assert_arrd_bits_eq(
        a.upper_a.patches.as_ref().unwrap(),
        &snap_up,
        "self upper unmutated",
    );
    assert!(a.lower_a.coeff_err.is_none() && a.upper_a.coeff_err.is_none());
    assert!(!a.lower_a.identity && !a.upper_a.identity);
}

/// T5 (spec §4.4, alignment gate): a carried err whose length disagrees with
/// `row_count` on ANY of the four 7D sides is rejected with `Ok(false)`;
/// correct-length errs merge fine (control).
#[test]
fn test_merge_7d_gate_rejects_err_length_mismatch() {
    let rows = 3;
    let mk = || {
        make_7d_patches(
            rows,
            (1, 2, 2),
            (1, 1),
            (0, 0, 0, 0),
            |k| 0.1 + (k as f32) * 0.01,
            |k| 0.5 - (k as f32) * 0.01,
        )
    };
    // Mismatched Some on self.lower.
    let mut a = mk();
    a.lower_a.coeff_err = Some(Array1::from(vec![1e-3f32, 2e-3])); // len 2 != 3
    let b = mk();
    assert!(!a.try_merge_inplace(&b).unwrap());

    // Mismatched Some on other.upper.
    let mut a2 = mk();
    let mut b2 = mk();
    b2.upper_a.coeff_err = Some(Array1::zeros(4)); // len 4 != 3
    assert!(!a2.try_merge_inplace(&b2).unwrap());

    // Control: correct-length errs pass the gate and merge.
    let mut a3 = mk();
    a3.lower_a.coeff_err = Some(Array1::zeros(rows));
    let b3 = mk();
    assert!(a3.try_merge_inplace(&b3).unwrap());
}

/// T5 (spec §4.4): a 6D/7D pair is incompatible (representation-family check)
/// and falls back to dense promotion with `Ok(false)`.
#[test]
fn test_merge_7d_with_6d_pair_falls_back() {
    // row_counts agree (8) but the layouts differ.
    let mut a = make_simple_patches(
        (2, 2, 2),
        (2, 2, 2),
        (1, 1),
        (1, 1),
        (0, 0, 0, 0),
        0.25,
        0.75,
    );
    let b = make_7d_patches(8, (2, 2, 2), (1, 1), (0, 0, 0, 0), |_| 0.1, |_| 0.2);
    assert!(!a.try_merge_inplace(&b).unwrap());
}

/// T6 (spec §4.4): degenerate all-zero rows with None errs — the directed
/// outward add still steps one subnormal (`next_down(0+0) = -2^-149` on the
/// lower side), and the emitted err is EXACTLY `f32::from_bits(1)` per row
/// (the `2^-149` grid floor; the `3U·RowMaxAbs` f64 term underflows the f32
/// cast to 0 and `next_up` regains one full subnormal step).
#[test]
fn test_merge_7d_all_zero_rows_emit_smallest_subnormal_err() {
    let rows = 2;
    let mut a = make_7d_patches(rows, (1, 2, 2), (2, 2), (0, 0, 0, 0), |_| 0.0, |_| 0.0);
    let b = make_7d_patches(rows, (1, 2, 2), (2, 2), (0, 0, 0, 0), |_| 0.0, |_| 0.0);
    assert!(a.try_merge_inplace(&b).unwrap());
    let sides = [
        ("lower", a.lower_a.coeff_err.as_ref().expect("Some")),
        ("upper", a.upper_a.coeff_err.as_ref().expect("Some")),
    ];
    for (name, err) in sides {
        assert_eq!(err.len(), rows);
        for (r, &e) in err.iter().enumerate() {
            assert_eq!(
                e.to_bits(),
                1,
                "{name} row {r}: expected exactly 2^-149, got {e:e}"
            );
        }
    }
}
