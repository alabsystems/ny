// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::patches::{CrownBounds, PatchesData, PatchesLinearBounds, UnstableIdx};
use crate::layers::activations::LinearRelaxation;
use crate::layers::common::crown_elementwise_backward_patches;
use ndarray::{array, ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

fn patches_relaxation(lower: f32, upper: f32) -> LinearRelaxation {
    LinearRelaxation::new(
        lower.abs() + 1.0,
        lower.abs() + 0.25,
        upper + 0.5,
        upper + 1.25,
    )
}

#[test]
fn test_crown_elementwise_backward_patches_dense_mixed_sign_coefficients() {
    let bounds = PatchesLinearBounds {
        row_count: 1,
        lower_a: PatchesData {
            coeff_err: None,
            patches: Some(
                ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 1, 2]), vec![1.5_f32, -2.0]).unwrap(),
            ),
            stride: (1, 1),
            padding: (0, 0, 0, 0),
            identity: false,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, 2),
            unstable_idx: None,
        },
        lower_b: array![0.25_f32],
        upper_a: PatchesData {
            coeff_err: None,
            patches: Some(
                ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 1, 2]), vec![-0.75_f32, 3.0]).unwrap(),
            ),
            stride: (1, 1),
            padding: (0, 0, 0, 0),
            identity: false,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, 2),
            unstable_idx: None,
        },
        upper_b: array![-0.5_f32],
    };
    let pre_activation = BoundedTensor::new(
        array![-1.0_f32, -2.0].into_dyn(),
        array![2.0_f32, 4.0].into_dyn(),
    )
    .unwrap();

    let result =
        crown_elementwise_backward_patches(&bounds, &pre_activation, patches_relaxation).unwrap();
    let CrownBounds::Patches(result) = result else {
        panic!("expected patches output");
    };

    let lower_patches = result.lower_a.patches.as_ref().expect("lower patches");
    let upper_patches = result.upper_a.patches.as_ref().expect("upper patches");

    assert_eq!(lower_patches[[0, 0, 0, 0, 0, 0]], next_down_f32(3.0));
    assert_eq!(lower_patches[[0, 0, 0, 0, 0, 1]], next_down_f32(-9.0));
    assert_eq!(upper_patches[[0, 0, 0, 0, 0, 0]], next_up_f32(-1.5));
    assert_eq!(upper_patches[[0, 0, 0, 0, 0, 1]], next_up_f32(13.5));

    assert_eq!(result.lower_b[0], next_down_f32(-8.375));
    assert_eq!(result.upper_b[0], next_up_f32(14.3125));
}

#[test]
fn test_crown_elementwise_backward_patches_sparse_identity_tracks_unstable_positions() {
    let bounds = PatchesLinearBounds::sparse_identity(
        (1, 1, 3),
        (1, 1, 3),
        UnstableIdx {
            channels: vec![0, 0],
            heights: vec![0, 0],
            widths: vec![0, 2],
        },
    );
    let bounds = PatchesLinearBounds {
        row_count: 2,
        lower_b: array![0.1_f32, -0.2],
        upper_b: array![0.3_f32, 0.4],
        ..bounds
    };
    let pre_activation = BoundedTensor::new(
        array![-1.0_f32, -2.0, -3.0].into_dyn(),
        array![2.0_f32, 4.0, 6.0].into_dyn(),
    )
    .unwrap();

    let result =
        crown_elementwise_backward_patches(&bounds, &pre_activation, patches_relaxation).unwrap();
    let CrownBounds::Patches(result) = result else {
        panic!("expected patches output");
    };

    let lower_patches = result.lower_a.patches.as_ref().expect("lower patches");
    let upper_patches = result.upper_a.patches.as_ref().expect("upper patches");

    assert_eq!(lower_patches.shape(), &[2, 1, 1, 1]);
    assert_eq!(upper_patches.shape(), &[2, 1, 1, 1]);

    assert_eq!(lower_patches[[0, 0, 0, 0]], next_down_f32(2.0));
    assert_eq!(lower_patches[[1, 0, 0, 0]], next_down_f32(4.0));
    assert_eq!(upper_patches[[0, 0, 0, 0]], next_up_f32(2.5));
    assert_eq!(upper_patches[[1, 0, 0, 0]], next_up_f32(6.5));

    assert_eq!(result.lower_b[0], next_down_f32(1.35));
    assert_eq!(result.lower_b[1], next_down_f32(3.05));
    assert_eq!(result.upper_b[0], next_up_f32(3.55));
    assert_eq!(result.upper_b[1], next_up_f32(7.65));

    let unstable_idx = result
        .lower_a
        .unstable_idx
        .as_ref()
        .expect("sparse output should retain unstable_idx");
    assert_eq!(unstable_idx.widths, vec![0, 2]);
}

// =====================================================================
// Byte-identity pin (#patches-coeff-err-soundness; 7D explicit-rows
// closure spec §6.4 T4, docs/PATCHES_7D_COEFF_ERR_CLOSURE.md).
//
// Committed against the UNMODIFIED tree: pins the exact bit patterns the
// CURRENT 6D activation backward emits (both coeff_err arrays, both
// biases including the incoming-err intercept discharge, and the full
// composed coefficient tensors). The 7D closure adds a 7D arm beside the
// 6D arm and must keep the 6D path byte-for-byte unchanged — this test
// must pass unmodified after it lands.
// =====================================================================

use ndarray::Array1;

/// Deterministic non-dyadic mixed-sign fill with exact zeros for the pin.
fn pin_fill(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let k = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            if k.is_multiple_of(9) {
                0.0
            } else {
                (((k >> 7) % 4000) as f32 - 2000.0) * 0.000_731
            }
        })
        .collect()
}

/// Fixed 6D fixture for the byte-identity pin: [2,2,2,2,2,2] patches
/// (64 coefficients/side), stride 1, padding (1,1,1,1) (padding taps
/// exercised), nonzero incoming coeff_err on both sides (with exact-zero
/// rows), non-dyadic biases, mixed-regime pre-activation bounds, and the
/// nonzero-intercept `patches_relaxation` so the intercept discharge is live.
fn run_6d_err_pin_fixture() -> Box<PatchesLinearBounds> {
    let shape = [2usize, 2, 2, 2, 2, 2];
    let n: usize = shape.iter().product();
    let mk = |seed: u32, err: Vec<f32>| PatchesData {
        coeff_err: Some(Array1::from_vec(err)),
        patches: Some(ArrayD::from_shape_vec(IxDyn(&shape), pin_fill(n, seed)).unwrap()),
        stride: (1, 1),
        padding: (1, 1, 1, 1),
        identity: false,
        output_shape: (2, 2, 2),
        input_shape: (2, 2, 2),
        unstable_idx: None,
    };
    let bounds = PatchesLinearBounds {
        row_count: 8,
        lower_a: mk(
            1,
            vec![1.0e-3, 0.0, 5.0e-4, 2.0e-3, 0.0, 1.0e-6, 7.0e-4, 3.0e-3],
        ),
        lower_b: array![0.13_f32, -0.7, 0.29, -0.011, 0.53, 0.0, -1.21, 0.077],
        upper_a: mk(
            2,
            vec![2.0e-3, 1.0e-4, 0.0, 5.0e-5, 4.0e-4, 0.0, 6.0e-4, 8.0e-4],
        ),
        upper_b: array![0.41_f32, 0.09, -0.33, 0.72, -0.005, 1.13, 0.0, -0.86],
    };
    let pre_activation = BoundedTensor::new(
        array![-1.3_f32, 0.2, -0.45, -2.7, 0.9, -0.05, -1.15, 0.33].into_dyn(),
        array![0.7_f32, 1.9, 0.6, -0.9, 2.1, 1.4, 0.02, 0.87].into_dyn(),
    )
    .unwrap();
    let result =
        crown_elementwise_backward_patches(&bounds, &pre_activation, patches_relaxation).unwrap();
    let CrownBounds::Patches(res) = result else {
        panic!("expected patches output");
    };
    res
}

fn assert_pinned_bits(label: &str, actual: &[f32], expected_bits: &[u32]) {
    assert_eq!(
        actual.len(),
        expected_bits.len(),
        "{label}: length mismatch"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected_bits.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e,
            "{label}[{i}]: got {a:?} (bits {:#010x}), pinned {:#010x}",
            a.to_bits(),
            e
        );
    }
}

/// T4 (spec §6.4): byte-identity pin for the 6D activation backward. Bit
/// literals captured from the UNMODIFIED (pre-closure) tree via the (now
/// deleted) capture harness; the 7D closure must keep this green unmodified.
#[test]
fn test_patches_backward_6d_err_byte_identical_pin() {
    const PIN_LOWER_ERR_BITS: [u32; 8] = [
        0x3b9375fc, 0x34366b3d, 0x3b1378de, 0x3c1375fc, 0x34066193, 0x369b1e99, 0x3b4e7535,
        0x3c5d3037,
    ];
    const PIN_UPPER_ERR_BITS: [u32; 8] = [
        0x3c13751c, 0x39ebfb8f, 0x348effab, 0x396c4c82, 0x3aebf15e, 0x33b5d201, 0x3b30f73f,
        0x3b6bf0e9,
    ];
    const PIN_LOWER_B_BITS: [u32; 8] = [
        0x3f3bafe1, 0xc03fa67b, 0xc057abb6, 0xc138b07a, 0x3fb25935, 0xc01ba0c0, 0xc0a12d1a,
        0xc0f2fc28,
    ];
    const PIN_UPPER_B_BITS: [u32; 8] = [
        0x405fdb7e, 0x3f3f51ff, 0x3ec504a9, 0xc01e7001, 0x40876eb0, 0x3ed50231, 0x3e2b57a7,
        0xbf111e35,
    ];
    const PIN_LOWER_PATCH_BITS: [u32; 64] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x3f813ed6, 0x00000000, 0x00000000, 0x40013b3f, 0x00000000, 0x00000000, 0x00000000,
        0xbf8495fb, 0xbfd4fac0, 0x00000000, 0xbd7f1a93, 0x00000000, 0x00000000, 0x00000000,
        0xc05d0458, 0x00000000, 0x3fb06d59, 0xbf96cb51, 0xc0601d35, 0x3fb74b86, 0x00000000,
        0xc0392237, 0x3e4ac7d1, 0xbe19c03e, 0xbf865759, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3fb69447, 0x00000000, 0x00000000,
        0xbd6d2386, 0x00000000, 0x00000000, 0x00000000, 0xbeee6949, 0xbf9f77cc, 0x00000000,
        0x3ec52108, 0x00000000, 0x00000000, 0x00000000, 0xc0388685, 0x00000000, 0xbe17ce03,
        0xbf6a382d, 0xc03e6dfb, 0x3db65866, 0x00000000, 0xc014a463, 0x3edb4933, 0xbd13e98d,
        0x3fd1b0c8,
    ];
    const PIN_UPPER_PATCH_BITS: [u32; 64] = [
        0x00000000, 0x00000000, 0x00000000, 0x3f4b744c, 0x00000000, 0x00000000, 0x00000000,
        0x3fb0dcbb, 0x00000000, 0x00000000, 0x3f86d9a8, 0xbf64d473, 0x00000000, 0x00000000,
        0xbf41c781, 0xbf6b65fa, 0x00000000, 0x00000000, 0x00000000, 0x3f596090, 0x00000000,
        0xc021832b, 0x00000000, 0x3eaaaeda, 0xc01082d6, 0x00000000, 0x3f8b0d2a, 0xc0166cb1,
        0xc0074a3a, 0x3eb777d8, 0xbf1eecdc, 0xbf826b37, 0x00000000, 0x00000000, 0x00000000,
        0x3f876960, 0x00000000, 0x00000000, 0x00000000, 0x3ff9d863, 0x00000000, 0x00000000,
        0xbde34208, 0xbf2175ff, 0x00000000, 0x00000000, 0xbeae393e, 0xbf304107, 0x00000000,
        0x00000000, 0x00000000, 0x3f8b90e8, 0x00000000, 0xc006d873, 0x00000000, 0xbf1ce9dd,
        0xbfe075d3, 0x00000000, 0x3d8a54b9, 0xbfc4fd1a, 0xbfd93f05, 0x3f4666cf, 0xbe18e3de,
        0x3fd7ff40,
    ];

    let res = run_6d_err_pin_fixture();
    assert_pinned_bits(
        "PIN_LOWER_ERR_BITS",
        res.lower_a.coeff_err.as_ref().unwrap().as_slice().unwrap(),
        &PIN_LOWER_ERR_BITS,
    );
    assert_pinned_bits(
        "PIN_UPPER_ERR_BITS",
        res.upper_a.coeff_err.as_ref().unwrap().as_slice().unwrap(),
        &PIN_UPPER_ERR_BITS,
    );
    assert_pinned_bits(
        "PIN_LOWER_B_BITS",
        res.lower_b.as_slice().unwrap(),
        &PIN_LOWER_B_BITS,
    );
    assert_pinned_bits(
        "PIN_UPPER_B_BITS",
        res.upper_b.as_slice().unwrap(),
        &PIN_UPPER_B_BITS,
    );
    assert_pinned_bits(
        "PIN_LOWER_PATCH_BITS",
        res.lower_a.patches.as_ref().unwrap().as_slice().unwrap(),
        &PIN_LOWER_PATCH_BITS,
    );
    assert_pinned_bits(
        "PIN_UPPER_PATCH_BITS",
        res.upper_a.patches.as_ref().unwrap().as_slice().unwrap(),
        &PIN_UPPER_PATCH_BITS,
    );
}

// =====================================================================
// 7D explicit-rows err lift (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §6.4):
// T1 f64 oracle coverage, T2 length-mismatch hard error, T3 gap-only
// emission on exact inputs, T5 serial-vs-parallel bitwise equality.
// The err index on this layout is the SPEC row (axis 0).
// =====================================================================

use ny_core::NyError;

/// 7D fixture geometry: [row=2, oc=2, oh=1, ow=2, ic=2, ki=1, kj=2],
/// stride (1,1), padding (left=1, 0, 0, 0) — the (ow=0, kj=0) taps map to
/// iw = −1 (out-of-bounds padding taps, exercised per §6.4 T1).
const F7_SHAPE: [usize; 7] = [2, 2, 1, 2, 2, 1, 2];
const F7_INPUT_SHAPE: (usize, usize, usize) = (2, 1, 2);
const F7_PADDING: (usize, usize, usize, usize) = (1, 0, 0, 0);

/// Input-neuron flat index for a tap of the T1 fixture, `None` for padding
/// taps. Mirrors the padding predicate + `input_flat` mapping of the
/// production loop.
fn f7_input_flat(oh: usize, ow: usize, ki: usize, kj: usize) -> Option<usize> {
    let (_, in_h, in_w) = F7_INPUT_SHAPE;
    let ih_raw = (oh + ki) as isize; // stride 1, pad_top 0
    let iw_raw = (ow + kj) as isize - 1; // stride 1, pad_left 1
    if ih_raw < 0 || (ih_raw as usize) >= in_h || iw_raw < 0 || (iw_raw as usize) >= in_w {
        None
    } else {
        Some(ih_raw as usize * in_w + iw_raw as usize)
    }
}

fn mk_7d_fixture(
    lower_err: Option<Vec<f32>>,
    upper_err: Option<Vec<f32>>,
) -> (PatchesLinearBounds, BoundedTensor) {
    let n: usize = F7_SHAPE.iter().product();
    let mk = |seed: u32, err: Option<Vec<f32>>| PatchesData {
        coeff_err: err.map(Array1::from_vec),
        patches: Some(ArrayD::from_shape_vec(IxDyn(&F7_SHAPE), pin_fill(n, seed)).unwrap()),
        stride: (1, 1),
        padding: F7_PADDING,
        identity: false,
        output_shape: (2, 1, 2),
        input_shape: F7_INPUT_SHAPE,
        unstable_idx: None,
    };
    let bounds = PatchesLinearBounds {
        row_count: 2,
        lower_a: mk(3, lower_err),
        lower_b: array![0.25_f32, -0.5],
        upper_a: mk(4, upper_err),
        upper_b: array![-0.125_f32, 0.375],
    };
    // 4 input neurons (2, 1, 2); mixed-regime bounds so patches_relaxation
    // yields 4 distinct relaxations with nonzero intercepts.
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), vec![-1.0_f32, 0.5, -0.75, 0.25]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), vec![2.0_f32, 1.5, 0.5, 0.8]).unwrap(),
    )
    .unwrap();
    (bounds, pre_activation)
}

/// f64 mirrors of the fixture's per-neuron relaxations.
struct RelaxF64 {
    ls: f64,
    li: f64,
    us: f64,
    ui: f64,
}

fn f7_relaxations() -> Vec<RelaxF64> {
    let pre_l = [-1.0_f32, 0.5, -0.75, 0.25];
    let pre_u = [2.0_f32, 1.5, 0.5, 0.8];
    pre_l
        .iter()
        .zip(pre_u.iter())
        .map(|(&l, &u)| {
            let r = patches_relaxation(l, u);
            RelaxF64 {
                ls: f64::from(r.lower_slope),
                li: f64::from(r.lower_intercept),
                us: f64::from(r.upper_slope),
                ui: f64::from(r.upper_intercept),
            }
        })
        .collect()
}

/// Composed coefficient and bias contribution for a candidate true incoming
/// coefficient `c`, LOWER side (sign selection exactly as `compose_lower`).
fn comp_lower_f64(c: f64, r: &RelaxF64) -> (f64, f64) {
    if c > 0.0 {
        (c * r.ls, c * r.li)
    } else if c < 0.0 {
        (c * r.us, c * r.ui)
    } else {
        (0.0, 0.0)
    }
}

/// UPPER-side mirror (sign selection exactly as `compose_upper`).
fn comp_upper_f64(c: f64, r: &RelaxF64) -> (f64, f64) {
    if c > 0.0 {
        (c * r.us, c * r.ui)
    } else if c < 0.0 {
        (c * r.ls, c * r.li)
    } else {
        (0.0, 0.0)
    }
}

/// Candidate true incoming coefficients for stored `a` under row err `e`:
/// the endpoints of `[a−e, a+e]` plus 0 when the interval straddles it.
/// Sufficient for extremal composed coefficient/intercept: both maps are
/// piecewise linear in the true coefficient with the only breakpoint at 0.
fn err_candidates(a: f64, e: f64) -> Vec<f64> {
    let mut c = vec![a - e, a + e];
    if a - e < 0.0 && 0.0 < a + e {
        c.push(0.0);
    }
    c
}

/// Shared f64 oracle for the 7D arm (§6.4 T1 semantics, no tolerance
/// epsilon anywhere): per valid tap every candidate composed coefficient is
/// within the emitted row err of the stored one; padding taps are stored 0
/// exactly; the output biases are outside `b + Σ min/max candidate
/// intercept folds`. `strict_bias` asserts strict inequality (T3: proves
/// the gbar·ABS discharge + directed cast move outward even at e = 0).
fn check_7d_oracle(input: &PatchesLinearBounds, output: &PatchesLinearBounds, strict_bias: bool) {
    let relax = f7_relaxations();
    let [rows, out_c, out_h, out_w, in_c, kh, kw] = F7_SHAPE;
    let (_, in_h, in_w) = F7_INPUT_SHAPE;
    let old_l = input.lower_a.patches.as_ref().unwrap();
    let old_u = input.upper_a.patches.as_ref().unwrap();
    let new_l = output.lower_a.patches.as_ref().unwrap();
    let new_u = output.upper_a.patches.as_ref().unwrap();
    let err_l = output.lower_a.coeff_err.as_ref().expect("lower err Some");
    let err_u = output.upper_a.coeff_err.as_ref().expect("upper err Some");
    assert_eq!(
        err_l.len(),
        rows,
        "err index is the spec row (len == row_count)"
    );
    assert_eq!(err_u.len(), rows);

    for row in 0..rows {
        let e_l = input
            .lower_a
            .coeff_err
            .as_ref()
            .map_or(0.0, |e| f64::from(e[row]));
        let e_u = input
            .upper_a
            .coeff_err
            .as_ref()
            .map_or(0.0, |e| f64::from(e[row]));
        let ne_l = f64::from(err_l[row]);
        let ne_u = f64::from(err_u[row]);
        let mut bias_min_l = f64::from(input.lower_b[row]);
        let mut bias_max_u = f64::from(input.upper_b[row]);
        let mut saw_padding = false;
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let idx = [row, oc, oh, ow, ic, ki, kj];
                                let Some(pos) = f7_input_flat(oh, ow, ki, kj) else {
                                    // Padding taps: never composed, stored 0
                                    // exactly on both sides.
                                    assert_eq!(new_l[idx], 0.0, "padding tap not 0 (lower)");
                                    assert_eq!(new_u[idx], 0.0, "padding tap not 0 (upper)");
                                    saw_padding = true;
                                    continue;
                                };
                                let input_flat = ic * in_h * in_w + pos;
                                let r = &relax[input_flat];

                                let a_l = f64::from(old_l[idx]);
                                let stored_l = f64::from(new_l[idx]);
                                let mut min_h = f64::INFINITY;
                                for cand in err_candidates(a_l, e_l) {
                                    let (c_ideal, h) = comp_lower_f64(cand, r);
                                    assert!(
                                        (stored_l - c_ideal).abs() <= ne_l,
                                        "lower coeff row {row} tap {idx:?}: \
                                         |{stored_l} - {c_ideal}| > err {ne_l}",
                                    );
                                    if h < min_h {
                                        min_h = h;
                                    }
                                }
                                bias_min_l += min_h;

                                let a_u = f64::from(old_u[idx]);
                                let stored_u = f64::from(new_u[idx]);
                                let mut max_h = f64::NEG_INFINITY;
                                for cand in err_candidates(a_u, e_u) {
                                    let (c_ideal, h) = comp_upper_f64(cand, r);
                                    assert!(
                                        (stored_u - c_ideal).abs() <= ne_u,
                                        "upper coeff row {row} tap {idx:?}: \
                                         |{stored_u} - {c_ideal}| > err {ne_u}",
                                    );
                                    if h > max_h {
                                        max_h = h;
                                    }
                                }
                                bias_max_u += max_h;
                            }
                        }
                    }
                }
            }
        }
        assert!(saw_padding, "fixture must exercise out-of-bounds taps");

        let out_bl = f64::from(output.lower_b[row]);
        let out_bu = f64::from(output.upper_b[row]);
        if strict_bias {
            assert!(
                out_bl < bias_min_l,
                "lower bias row {row}: {out_bl} not strictly below oracle {bias_min_l} \
                 (gbar·ABS discharge / directed cast missing?)",
            );
            assert!(
                out_bu > bias_max_u,
                "upper bias row {row}: {out_bu} not strictly above oracle {bias_max_u}",
            );
        } else {
            assert!(
                out_bl <= bias_min_l,
                "lower bias row {row}: {out_bl} > oracle min {bias_min_l}",
            );
            assert!(
                out_bu >= bias_max_u,
                "upper bias row {row}: {out_bu} < oracle max {bias_max_u}",
            );
        }
    }
}

fn run_7d(bounds: &PatchesLinearBounds, pre: &BoundedTensor) -> Box<PatchesLinearBounds> {
    let result = crown_elementwise_backward_patches(bounds, pre, patches_relaxation).unwrap();
    let CrownBounds::Patches(res) = result else {
        panic!("expected patches output");
    };
    res
}

/// T1 (spec §6.4): 7D explicit-rows arm emits per-SPEC-row errs that cover
/// the f64 oracle over every admissible true incoming coefficient (endpoint +
/// sign-flip candidates), with padding taps stored exactly 0 and biases
/// outside the candidate-extremal intercept folds. Row 0 carries nonzero
/// incoming err on both sides (0.75 / 0.5 — sign flips live); row 1 is
/// exact. No tolerance epsilon anywhere.
#[test]
fn test_patches_backward_7d_err_covers_f64_oracle() {
    let (bounds, pre) = mk_7d_fixture(Some(vec![0.75, 0.0]), Some(vec![0.5, 0.0]));
    // Fixture sanity: mixed-sign values with exact zeros among the taps.
    let lp = bounds.lower_a.patches.as_ref().unwrap();
    assert!(lp.iter().any(|&v| v > 0.0) && lp.iter().any(|&v| v < 0.0));
    assert!(lp.iter().any(|&v| v == 0.0));

    let res = run_7d(&bounds, &pre);
    check_7d_oracle(&bounds, &res, false);
}

/// T2 (spec §6.4 / I6): a carried 7D err whose length != row_count is a
/// construction bug — hard Err(ShapeMismatch), never a silent under-count.
#[test]
fn test_patches_backward_7d_err_wrong_length_rejected() {
    for (le, ue) in [
        (Some(vec![0.1_f32, 0.2, 0.3]), None),
        (None, Some(vec![0.1_f32])),
    ] {
        let (bounds, pre) = mk_7d_fixture(le, ue);
        let err = crown_elementwise_backward_patches(&bounds, &pre, patches_relaxation)
            .expect_err("length-mismatched 7D err must be rejected");
        assert!(
            matches!(err, NyError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got {err:?}",
        );
    }
}

/// T3 (spec §6.4): exact inputs (None err both sides) still emit Some errs —
/// the directed-rounding gap terms are intrinsic — with each entry tightly
/// bounded by the computed per-row max gap, and biases strictly outside the
/// e=0 oracle (proves the gbar·ABS compose-fold discharge is present and
/// outward).
#[test]
fn test_patches_backward_7d_none_err_emits_gap_only() {
    let (bounds, pre) = mk_7d_fixture(None, None);
    let res = run_7d(&bounds, &pre);

    // Entries: gap-covering from below, within one outward f32 step of the
    // f64 row max gap from above.
    let relax = f7_relaxations();
    let [rows, out_c, out_h, out_w, in_c, kh, kw] = F7_SHAPE;
    let (_, in_h, in_w) = F7_INPUT_SHAPE;
    for row in 0..rows {
        let mut gap_l = 0.0f64;
        let mut gap_u = 0.0f64;
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let Some(pos) = f7_input_flat(oh, ow, ki, kj) else {
                                    continue;
                                };
                                let r = &relax[ic * in_h * in_w + pos];
                                let idx = [row, oc, oh, ow, ic, ki, kj];
                                let a_l = f64::from(bounds.lower_a.patches.as_ref().unwrap()[idx]);
                                let (ideal_l, _) = comp_lower_f64(a_l, r);
                                let stored_l =
                                    f64::from(res.lower_a.patches.as_ref().unwrap()[idx]);
                                gap_l = gap_l.max((stored_l - ideal_l).abs());
                                let a_u = f64::from(bounds.upper_a.patches.as_ref().unwrap()[idx]);
                                let (ideal_u, _) = comp_upper_f64(a_u, r);
                                let stored_u =
                                    f64::from(res.upper_a.patches.as_ref().unwrap()[idx]);
                                gap_u = gap_u.max((stored_u - ideal_u).abs());
                            }
                        }
                    }
                }
            }
        }
        let ne_l = f64::from(res.lower_a.coeff_err.as_ref().expect("Some out")[row]);
        let ne_u = f64::from(res.upper_a.coeff_err.as_ref().expect("Some out")[row]);
        assert!(ne_l >= gap_l, "row {row} lower err {ne_l} < gap {gap_l}");
        assert!(ne_u >= gap_u, "row {row} upper err {ne_u} < gap {gap_u}");
        assert!(
            ne_l <= f64::from(next_up_f32(gap_l as f32)),
            "row {row} lower err {ne_l} looser than the 1-ulp gap bound",
        );
        assert!(
            ne_u <= f64::from(next_up_f32(gap_u as f32)),
            "row {row} upper err {ne_u} looser than the 1-ulp gap bound",
        );
    }

    // Biases: e = 0 oracle, strict (discharge present and outward).
    check_7d_oracle(&bounds, &res, true);
}

/// Rebuild a logically-identical 7D tensor with reversed memory layout so
/// `as_slice()` returns `None` (forces the serial indexed fallback).
fn to_noncontiguous_7d(arr: &ArrayD<f32>) -> ArrayD<f32> {
    let shape = arr.shape().to_vec();
    let rev_shape: Vec<usize> = shape.iter().rev().copied().collect();
    let perm: Vec<usize> = (0..shape.len()).rev().collect();
    let mut buf = ArrayD::<f32>::zeros(IxDyn(&rev_shape));
    buf.view_mut().permuted_axes(perm.clone()).assign(arr);
    let nc = buf.permuted_axes(perm);
    assert_eq!(nc.shape(), arr.shape());
    assert!(nc.as_slice().is_none(), "fixture must be non-contiguous");
    assert_eq!(&nc, arr);
    nc
}

/// T5 (spec §6.4): the parallel row driver and the serial indexed fallback
/// (reached via a non-contiguous, permuted-axes input) produce bitwise
/// identical patches, biases, and err arrays.
#[test]
fn test_patches_backward_7d_serial_fallback_matches_parallel() {
    let (bounds, pre) = mk_7d_fixture(Some(vec![0.75, 0.0]), Some(vec![0.5, 1.0e-3]));
    let res_par = run_7d(&bounds, &pre);

    let mut bounds_nc = bounds.clone();
    bounds_nc.lower_a.patches = Some(to_noncontiguous_7d(
        bounds.lower_a.patches.as_ref().unwrap(),
    ));
    bounds_nc.upper_a.patches = Some(to_noncontiguous_7d(
        bounds.upper_a.patches.as_ref().unwrap(),
    ));
    let res_ser = run_7d(&bounds_nc, &pre);

    let bits = |xs: &[f32]| xs.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let iter_bits = |a: &ArrayD<f32>| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        iter_bits(res_par.lower_a.patches.as_ref().unwrap()),
        iter_bits(res_ser.lower_a.patches.as_ref().unwrap()),
        "lower patches diverge",
    );
    assert_eq!(
        iter_bits(res_par.upper_a.patches.as_ref().unwrap()),
        iter_bits(res_ser.upper_a.patches.as_ref().unwrap()),
        "upper patches diverge",
    );
    assert_eq!(
        bits(res_par.lower_b.as_slice().unwrap()),
        bits(res_ser.lower_b.as_slice().unwrap()),
        "lower bias diverges",
    );
    assert_eq!(
        bits(res_par.upper_b.as_slice().unwrap()),
        bits(res_ser.upper_b.as_slice().unwrap()),
        "upper bias diverges",
    );
    assert_eq!(
        bits(
            res_par
                .lower_a
                .coeff_err
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap()
        ),
        bits(
            res_ser
                .lower_a
                .coeff_err
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap()
        ),
        "lower err diverges",
    );
    assert_eq!(
        bits(
            res_par
                .upper_a
                .coeff_err
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap()
        ),
        bits(
            res_ser
                .upper_a
                .coeff_err
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap()
        ),
        "upper err diverges",
    );
}
