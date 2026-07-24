// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `crown_relu_backward_patches_with_alpha`.
//!
//! Covers: alpha composition for crossing neurons, directed-rounding bias,
//! gradient d(lower_bound_sum)/d(alpha[i]) accumulation, and identity
//! pass-through for always-active neurons.
//!
//! Part of #3463

use crate::bounds::patches::{CrownBounds, PatchesData, PatchesLinearBounds, UnstableIdx};
use crate::layers::activations::relu::relu_linear_relaxation;
use crate::layers::common::{
    crown_relu_backward_patches_with_alpha, crown_relu_backward_patches_with_alpha_bound_only,
};
use ndarray::{array, ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

/// Dense crossing neurons: both inputs cross zero.
///
/// Verifies alpha-parametric lower slope composition, chord upper slope
/// composition, directed-rounding on patches coefficients and bias, and
/// gradient accumulation only for positive lower-A crossing neurons.
///
/// Pre-activation: l = [-1, -2], u = [3, 2]  → both crossing
/// Alpha: [0.5, 0.25]
///
/// Relaxation per neuron:
///   n0: lower_slope=0.5 (alpha), upper_slope=3/4=0.75, upper_intercept=0.75
///   n1: lower_slope=0.25 (alpha), upper chord rounds slightly above 0.5 / 1.0
///       so the stored f32 line stays conservative
///
/// lower_a = [1.5, -2.0]  →  compose_lower:
///   la0=+1.5: 1.5*0.5=0.75,  intercept=0.0
///   la1=-2.0: -2.0*0.5=-1.0, intercept=-2.0*1.0=-2.0
/// lower_b = 0.25 + 0.0 + (-2.0) = -1.75
///
/// upper_a = [-0.75, 3.0]  →  compose_upper:
///   ua0=-0.75: -0.75*0.5=-0.375, intercept=0.0
///   ua1=+3.0:  3.0*0.5=1.5,      intercept=3.0*1.0=3.0
/// upper_b = -0.5 + 0.0 + 3.0 = 2.5
///
/// Gradient: la0=1.5>0, crossing → 1.5*(-1.0)=-1.5;  la1=-2.0≤0 → 0.0
#[test]
fn test_crown_relu_backward_patches_with_alpha_dense_crossing_neurons() {
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
        array![3.0_f32, 2.0].into_dyn(),
    )
    .unwrap();
    let alpha = array![0.5_f32, 0.25];

    let (result, gradient) =
        crown_relu_backward_patches_with_alpha(&bounds, &pre_activation, &alpha).unwrap();
    let bound_only =
        crown_relu_backward_patches_with_alpha_bound_only(&bounds, &pre_activation, &alpha)
            .unwrap();
    let CrownBounds::Patches(result) = result else {
        panic!("expected patches output");
    };
    let CrownBounds::Patches(bound_only) = bound_only else {
        panic!("expected bound-only patches output");
    };

    assert_eq!(
        bound_only.lower_a.patches.as_ref(),
        result.lower_a.patches.as_ref()
    );
    assert_eq!(
        bound_only.lower_a.coeff_err.as_ref(),
        result.lower_a.coeff_err.as_ref()
    );
    assert_eq!(bound_only.lower_b, result.lower_b);
    assert_eq!(
        bound_only.upper_a.patches.as_ref(),
        result.upper_a.patches.as_ref()
    );
    assert_eq!(
        bound_only.upper_a.coeff_err.as_ref(),
        result.upper_a.coeff_err.as_ref()
    );
    assert_eq!(bound_only.upper_b, result.upper_b);

    let lower_patches = result.lower_a.patches.as_ref().expect("lower patches");
    let upper_patches = result.upper_a.patches.as_ref().expect("upper patches");

    let n1_relax = relu_linear_relaxation(-2.0, 2.0);

    // Coefficient assertions (directed rounding).
    assert_eq!(lower_patches[[0, 0, 0, 0, 0, 0]], next_down_f32(0.75));
    assert_eq!(
        lower_patches[[0, 0, 0, 0, 0, 1]],
        next_down_f32(-2.0 * n1_relax.upper_slope)
    );
    assert_eq!(upper_patches[[0, 0, 0, 0, 0, 0]], next_up_f32(-0.375));
    assert_eq!(
        upper_patches[[0, 0, 0, 0, 0, 1]],
        next_up_f32(3.0 * n1_relax.upper_slope)
    );

    // Bias assertions (f64 accumulation → next_down/next_up).
    let expected_lower_b = 0.25_f64 + (-2.0_f64) * n1_relax.upper_intercept as f64;
    let expected_upper_b = -0.5_f64 + 3.0_f64 * n1_relax.upper_intercept as f64;
    assert_eq!(result.lower_b[0], next_down_f32(expected_lower_b as f32));
    assert_eq!(result.upper_b[0], next_up_f32(expected_upper_b as f32));

    // Gradient: d(lower_bound_sum)/d(alpha[i]).
    // Only positive lower-A coefficients on crossing neurons contribute.
    assert_eq!(gradient[0], -1.5); // la0=1.5 * pre_lower[0]=-1.0
    assert_eq!(gradient[1], 0.0); // la1=-2.0 ≤ 0, no contribution
}

/// Mixed neuron states: one always-active, one crossing.
///
/// Verifies identity pass-through for always-active neurons (alpha ignored),
/// alpha composition for crossing neurons, and gradient accumulation
/// restricted to crossing neurons only.
///
/// Pre-activation: l = [1, -2], u = [3, 2]
///   n0: l≥0 → identity (lower_slope=1, upper_slope=1, intercepts=0)
///   n1: crossing → alpha=0.75 lower, chord=0.5 upper, upper_intercept=1.0
/// Alpha: [0.5, 0.75]
///
/// lower_a = [2.0, 1.5]  →  compose_lower:
///   la0=+2.0: 2.0*1.0=2.0,     intercept=0.0
///   la1=+1.5: 1.5*0.75=1.125,  intercept=0.0
/// lower_b = 0.0
///
/// upper_a = [1.0, -0.5]  →  compose_upper:
///   ua0=+1.0: 1.0*1.0=1.0,       intercept=0.0
///   ua1=-0.5: -0.5*0.75=-0.375,  intercept=0.0
/// upper_b = 0.0
///
/// Gradient: la0=2.0>0 but n0 not crossing → 0;  la1=1.5>0, n1 crossing → 1.5*(-2.0)=-3.0
#[test]
fn test_crown_relu_backward_patches_with_alpha_dense_active_plus_crossing() {
    let bounds = PatchesLinearBounds {
        row_count: 1,
        lower_a: PatchesData {
            coeff_err: None,
            patches: Some(
                ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 1, 2]), vec![2.0_f32, 1.5]).unwrap(),
            ),
            stride: (1, 1),
            padding: (0, 0, 0, 0),
            identity: false,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, 2),
            unstable_idx: None,
        },
        lower_b: array![0.0_f32],
        upper_a: PatchesData {
            coeff_err: None,
            patches: Some(
                ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 1, 2]), vec![1.0_f32, -0.5]).unwrap(),
            ),
            stride: (1, 1),
            padding: (0, 0, 0, 0),
            identity: false,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, 2),
            unstable_idx: None,
        },
        upper_b: array![0.0_f32],
    };
    let pre_activation = BoundedTensor::new(
        array![1.0_f32, -2.0].into_dyn(),
        array![3.0_f32, 2.0].into_dyn(),
    )
    .unwrap();
    let alpha = array![0.5_f32, 0.75];

    let (result, gradient) =
        crown_relu_backward_patches_with_alpha(&bounds, &pre_activation, &alpha).unwrap();
    let CrownBounds::Patches(result) = result else {
        panic!("expected patches output");
    };

    let lower_patches = result.lower_a.patches.as_ref().expect("lower patches");
    let upper_patches = result.upper_a.patches.as_ref().expect("upper patches");

    // Always-active neuron: identity composition (alpha[0] not used).
    assert_eq!(lower_patches[[0, 0, 0, 0, 0, 0]], next_down_f32(2.0));
    assert_eq!(upper_patches[[0, 0, 0, 0, 0, 0]], next_up_f32(1.0));

    // Crossing neuron: alpha=0.75 used for lower slope.
    assert_eq!(lower_patches[[0, 0, 0, 0, 0, 1]], next_down_f32(1.125));
    // Upper uses lower relaxation (ua1=-0.5 negative → flip direction).
    assert_eq!(upper_patches[[0, 0, 0, 0, 0, 1]], next_up_f32(-0.375));

    // Bias: zero intercepts from identity + zero lower_intercept.
    assert_eq!(result.lower_b[0], next_down_f32(0.0));
    assert_eq!(result.upper_b[0], next_up_f32(0.0));

    // Gradient: only crossing neurons with positive lower-A contribute.
    assert_eq!(gradient[0], 0.0); // n0 always-active, no gradient
    assert_eq!(gradient[1], -3.0); // la1=1.5 * pre_lower[1]=-2.0
}

/// Sparse identity patches delegate to Dense propagate_linear_with_alpha.
///
/// When `unstable_idx` is present, the function converts to Dense and
/// delegates. Verifies the output is `CrownBounds::Dense` (not Patches)
/// and that the gradient vector has the correct length.
#[test]
fn test_crown_relu_backward_patches_with_alpha_sparse_delegates_to_dense() {
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
        array![-1.0_f32, 0.5, -2.0].into_dyn(),
        array![2.0_f32, 3.0, 4.0].into_dyn(),
    )
    .unwrap();
    let alpha = array![0.5_f32, 0.5, 0.5];

    let (result, gradient) =
        crown_relu_backward_patches_with_alpha(&bounds, &pre_activation, &alpha).unwrap();

    // Sparse path delegates to Dense — output must be CrownBounds::Dense.
    assert!(
        matches!(result, CrownBounds::Dense(_)),
        "sparse patches should delegate to Dense"
    );

    // Gradient vector covers all input neurons.
    assert_eq!(gradient.len(), 3);
}

// =====================================================================
// Equivalence test: optimized `crown_relu_backward_patches_with_alpha`
// must be bit-identical to the pre-optimization indexed reference.
//
// Perf change (#3293): the production loop was switched from per-element
// `ArrayD[[...]]` dynamic indexing to contiguous flat-slice iteration with a
// monotonic cursor, and the patches-tensor deep clone was removed in favor of
// borrowing. These tests pin every f32 of both patches tensors, both bias
// vectors, and the gradient to a self-contained reference re-implementation of
// the ORIGINAL indexed algorithm (using the unchanged `compose_*` math
// helpers). Any drift in accumulation order or value is caught bit-for-bit.
// =====================================================================

use crate::layers::activations::{relu_crossing_upper_chord, LinearRelaxation};
use crate::layers::common::compose;
use ndarray::Array1;

/// Deterministic xorshift-based PRNG (no external dep) producing f32 in a range.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform f32 in [lo, hi).
    fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        lo + u * (hi - lo)
    }
}

/// Build the same per-neuron relaxation the production code uses for alpha mode.
fn reference_relaxation(l: f32, u: f32, alpha_i: f32) -> LinearRelaxation {
    if l.is_nan() || u.is_nan() {
        LinearRelaxation::new(0.0, 0.0, 0.0, f32::INFINITY)
    } else if l >= 0.0 {
        LinearRelaxation::identity()
    } else if u <= 0.0 {
        LinearRelaxation::zero()
    } else if l.is_infinite() && u.is_infinite() {
        LinearRelaxation::new(alpha_i, 0.0, 0.0, f32::INFINITY)
    } else if u.is_infinite() {
        LinearRelaxation::new(alpha_i, 0.0, 1.0, -l)
    } else if l.is_infinite() {
        LinearRelaxation::new(alpha_i, 0.0, 0.0, u)
    } else {
        let (lambda, lambda_intercept) = relu_crossing_upper_chord(l, u, None);
        LinearRelaxation::new(alpha_i, 0.0, lambda, lambda_intercept)
    }
}

/// Independent re-implementation of the ORIGINAL indexed `crown_relu_backward_
/// patches_with_alpha` for the dense (non-sparse) 6-D path. Returns the new
/// lower/upper patches tensors, the two bias vectors, and the gradient.
///
/// Intentionally uses fully dynamic `ArrayD[[...]]` indexing (the pre-change
/// structure) so it does not share the optimized fast path's flat-cursor logic.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn reference_patches_alpha(
    bounds: &PatchesLinearBounds,
    pre_lower: &[f32],
    pre_upper: &[f32],
    alpha: &[f32],
) -> (
    ArrayD<f32>,
    Array1<f32>,
    ArrayD<f32>,
    Array1<f32>,
    Array1<f32>,
) {
    let (out_c, out_h, out_w) = bounds.lower_a.output_shape;
    let (in_c_shape, in_h_shape, in_w_shape) = bounds.lower_a.input_shape;
    let num_outputs = out_c * out_h * out_w;
    let num_input_neurons = in_c_shape * in_h_shape * in_w_shape;

    let relaxations: Vec<LinearRelaxation> = (0..num_input_neurons)
        .map(|i| reference_relaxation(pre_lower[i], pre_upper[i], alpha[i]))
        .collect();
    let is_crossing: Vec<bool> = (0..num_input_neurons)
        .map(|i| {
            let (l, u) = (pre_lower[i], pre_upper[i]);
            !l.is_nan() && !u.is_nan() && l < 0.0 && u > 0.0
        })
        .collect();

    let lower_patches = bounds.lower_a.patches.as_ref().unwrap();
    let upper_patches = bounds.upper_a.patches.as_ref().unwrap();
    let shape = lower_patches.shape();
    let (in_c, kh, kw) = (shape[3], shape[4], shape[5]);

    let (sh, sw) = bounds.lower_a.stride;
    let (pad_left, _pr, pad_top, _pb) = bounds.lower_a.padding;

    let mut new_lower_patches = ArrayD::<f32>::zeros(lower_patches.raw_dim());
    let mut new_upper_patches = ArrayD::<f32>::zeros(upper_patches.raw_dim());
    let mut new_lower_b_f64 = bounds.lower_b.mapv(|x| x as f64);
    let mut new_upper_b_f64 = bounds.upper_b.mapv(|x| x as f64);
    let mut lower_nonfinite = vec![false; num_outputs];
    let mut upper_nonfinite = vec![false; num_outputs];
    let mut gradient = Array1::<f32>::zeros(num_input_neurons);

    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let j = oc * out_h * out_w + oh * out_w + ow;
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                            let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                            if ih_raw < 0
                                || (ih_raw as usize) >= in_h_shape
                                || iw_raw < 0
                                || (iw_raw as usize) >= in_w_shape
                            {
                                continue;
                            }
                            let ih = ih_raw as usize;
                            let iw = iw_raw as usize;
                            let input_flat = ic * in_h_shape * in_w_shape + ih * in_w_shape + iw;
                            let relax = &relaxations[input_flat];

                            let la = lower_patches[[oc, oh, ow, ic, ki, kj]];
                            let lr = compose::compose_lower(la, relax);
                            new_lower_patches[[oc, oh, ow, ic, ki, kj]] = lr.new_coeff;
                            new_lower_b_f64[j] += lr.intercept_contrib;
                            lower_nonfinite[j] |= lr.nonfinite;

                            if la > 0.0 && is_crossing[input_flat] {
                                gradient[input_flat] += la * pre_lower[input_flat];
                            }

                            let ua = upper_patches[[oc, oh, ow, ic, ki, kj]];
                            let ur = compose::compose_upper(ua, relax);
                            new_upper_patches[[oc, oh, ow, ic, ki, kj]] = ur.new_coeff;
                            new_upper_b_f64[j] += ur.intercept_contrib;
                            upper_nonfinite[j] |= ur.nonfinite;
                        }
                    }
                }
            }
        }
    }

    let mut new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
    let mut new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));
    for j in 0..num_outputs {
        let oc = j / (out_h * out_w);
        let oh = (j % (out_h * out_w)) / out_w;
        let ow = j % out_w;
        if lower_nonfinite[j] {
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        new_lower_patches[[oc, oh, ow, ic, ki, kj]] = 0.0;
                    }
                }
            }
            new_lower_b[j] = f32::NEG_INFINITY;
        }
        if upper_nonfinite[j] {
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        new_upper_patches[[oc, oh, ow, ic, ki, kj]] = 0.0;
                    }
                }
            }
            new_upper_b[j] = f32::INFINITY;
        }
    }

    (
        new_lower_patches,
        new_lower_b,
        new_upper_patches,
        new_upper_b,
        gradient,
    )
}

/// Bit-identical comparison of two f32 slices (NaN bit patterns must also match).
fn assert_bits_eq(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}[{i}]: optimized {x:?} (bits {:#x}) != reference {y:?} (bits {:#x})",
            x.to_bits(),
            y.to_bits()
        );
    }
}

/// Build a random dense (6-D) patches-mode ReLU input and assert the optimized
/// production function is bit-identical to the indexed reference across a range
/// of geometries, strides, paddings, and pre-activation regimes.
#[test]
fn test_patches_alpha_optimized_matches_indexed_reference_bit_identical() {
    let mut rng = Lcg::new(0xC0FF_EE12_3456_789A);

    // (out_c, out_h, out_w, in_c, in_h, in_w, kh, kw, stride, pad_l, pad_t)
    let configs: &[(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    )] = &[
        (1, 1, 1, 1, 1, 2, 1, 2, 1, 0, 0), // matches existing dense test geometry
        (2, 3, 3, 2, 3, 3, 1, 1, 1, 0, 0), // 1x1, no stride/pad
        (3, 4, 4, 2, 6, 6, 3, 3, 1, 1, 1), // 3x3 kernel, padding
        (2, 2, 2, 3, 5, 5, 3, 3, 2, 1, 1), // stride 2 + padding
        (1, 3, 5, 1, 3, 5, 1, 1, 1, 0, 0), // non-square spatial
    ];

    for (cfg_idx, &(out_c, out_h, out_w, in_c, in_h, in_w, kh, kw, stride, pad_l, pad_t)) in
        configs.iter().enumerate()
    {
        // Iterate over several pre-activation regimes to exercise every branch:
        // 0=straddling, 1=all positive (identity), 2=all negative (zero), 3=mixed.
        for regime in 0..4 {
            let num_outputs = out_c * out_h * out_w;
            let num_in = in_c * in_h * in_w;

            let lower_vec: Vec<f32> = (0..num_outputs * in_c * kh * kw)
                .map(|_| rng.f32_in(-3.0, 3.0))
                .collect();
            let upper_vec: Vec<f32> = (0..num_outputs * in_c * kh * kw)
                .map(|_| rng.f32_in(-3.0, 3.0))
                .collect();

            let lower_patches =
                ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w, in_c, kh, kw]), lower_vec)
                    .unwrap();
            let upper_patches =
                ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w, in_c, kh, kw]), upper_vec)
                    .unwrap();

            let lower_b: Vec<f32> = (0..num_outputs).map(|_| rng.f32_in(-1.0, 1.0)).collect();
            let upper_b: Vec<f32> = (0..num_outputs).map(|_| rng.f32_in(-1.0, 1.0)).collect();

            let bounds = PatchesLinearBounds {
                row_count: num_outputs,
                lower_a: PatchesData {
                    coeff_err: None,
                    patches: Some(lower_patches),
                    stride: (stride, stride),
                    padding: (pad_l, pad_l, pad_t, pad_t),
                    identity: false,
                    output_shape: (out_c, out_h, out_w),
                    input_shape: (in_c, in_h, in_w),
                    unstable_idx: None,
                },
                lower_b: Array1::from_vec(lower_b),
                upper_a: PatchesData {
                    coeff_err: None,
                    patches: Some(upper_patches),
                    stride: (stride, stride),
                    padding: (pad_l, pad_l, pad_t, pad_t),
                    identity: false,
                    output_shape: (out_c, out_h, out_w),
                    input_shape: (in_c, in_h, in_w),
                    unstable_idx: None,
                },
                upper_b: Array1::from_vec(upper_b),
            };

            // Pre-activation bounds per regime.
            let mut pl = vec![0.0f32; num_in];
            let mut pu = vec![0.0f32; num_in];
            for i in 0..num_in {
                match regime {
                    1 => {
                        pl[i] = rng.f32_in(0.1, 2.0);
                        pu[i] = pl[i] + rng.f32_in(0.1, 2.0);
                    }
                    2 => {
                        pu[i] = rng.f32_in(-2.0, -0.1);
                        pl[i] = pu[i] - rng.f32_in(0.1, 2.0);
                    }
                    3 => {
                        // Mix: alternate straddling / stable per neuron.
                        if i % 2 == 0 {
                            pl[i] = rng.f32_in(-2.0, -0.1);
                            pu[i] = rng.f32_in(0.1, 2.0);
                        } else {
                            pl[i] = rng.f32_in(0.1, 1.0);
                            pu[i] = pl[i] + rng.f32_in(0.1, 1.0);
                        }
                    }
                    _ => {
                        // Straddling: l < 0 < u for all.
                        pl[i] = rng.f32_in(-3.0, -0.1);
                        pu[i] = rng.f32_in(0.1, 3.0);
                    }
                }
            }

            let pre_activation = BoundedTensor::new(
                Array1::from_vec(pl.clone()).into_dyn(),
                Array1::from_vec(pu.clone()).into_dyn(),
            )
            .unwrap();

            // Vary alpha across [0, 1] (and a few exact 0.0 / 1.0 endpoints).
            let alpha_vec: Vec<f32> = (0..num_in)
                .map(|i| match i % 3 {
                    0 => 0.0,
                    1 => 1.0,
                    _ => rng.f32_in(0.0, 1.0),
                })
                .collect();
            let alpha = Array1::from_vec(alpha_vec);

            let (result, gradient) =
                crown_relu_backward_patches_with_alpha(&bounds, &pre_activation, &alpha).unwrap();
            let CrownBounds::Patches(result) = result else {
                panic!("config {cfg_idx} regime {regime}: expected Patches output");
            };

            let (ref_lp, ref_lb, ref_up, ref_ub, ref_grad) =
                reference_patches_alpha(&bounds, &pl, &pu, alpha.as_slice().unwrap());

            let opt_lp = result.lower_a.patches.as_ref().unwrap();
            let opt_up = result.upper_a.patches.as_ref().unwrap();

            let tag = format!("cfg {cfg_idx} regime {regime}");
            assert_bits_eq(
                &format!("{tag} lower_patches"),
                opt_lp.as_slice().unwrap(),
                ref_lp.as_slice().unwrap(),
            );
            assert_bits_eq(
                &format!("{tag} upper_patches"),
                opt_up.as_slice().unwrap(),
                ref_up.as_slice().unwrap(),
            );
            assert_bits_eq(
                &format!("{tag} lower_b"),
                result.lower_b.as_slice().unwrap(),
                ref_lb.as_slice().unwrap(),
            );
            assert_bits_eq(
                &format!("{tag} upper_b"),
                result.upper_b.as_slice().unwrap(),
                ref_ub.as_slice().unwrap(),
            );
            assert_bits_eq(
                &format!("{tag} gradient"),
                gradient.as_slice().unwrap(),
                ref_grad.as_slice().unwrap(),
            );

            // Output geometry must be preserved.
            assert_eq!(result.lower_a.stride, (stride, stride));
            assert_eq!(result.lower_a.padding, (pad_l, pad_l, pad_t, pad_t));
            assert_eq!(result.lower_a.output_shape, (out_c, out_h, out_w));
            assert_eq!(result.lower_a.input_shape, (in_c, in_h, in_w));
        }
    }
}

/// Identity-input equivalence: when `lower_a`/`upper_a` are virtual identity,
/// the optimized path materializes them and must still match a reference built
/// from the explicitly-materialized identity tensors.
#[test]
fn test_patches_alpha_optimized_identity_input_matches_reference() {
    // Square identity patches (out_c == in_c) so materialize_identity is exact.
    let (c, h, w) = (3usize, 4usize, 4usize);
    let num_outputs = c * h * w;
    let num_in = c * h * w;

    let identity = PatchesData {
        coeff_err: None,
        patches: None,
        stride: (1, 1),
        padding: (0, 0, 0, 0),
        identity: true,
        output_shape: (c, h, w),
        input_shape: (c, h, w),
        unstable_idx: None,
    };
    let bounds = PatchesLinearBounds {
        row_count: num_outputs,
        lower_a: identity.clone(),
        lower_b: Array1::zeros(num_outputs),
        upper_a: identity,
        upper_b: Array1::zeros(num_outputs),
    };

    // Mix of regimes per neuron.
    let mut rng = Lcg::new(0x1234_5678_9ABC_DEF0);
    let mut pl = vec![0.0f32; num_in];
    let mut pu = vec![0.0f32; num_in];
    for i in 0..num_in {
        match i % 3 {
            0 => {
                pl[i] = rng.f32_in(-2.0, -0.1);
                pu[i] = rng.f32_in(0.1, 2.0);
            }
            1 => {
                pl[i] = rng.f32_in(0.1, 1.0);
                pu[i] = pl[i] + rng.f32_in(0.1, 1.0);
            }
            _ => {
                pu[i] = rng.f32_in(-2.0, -0.1);
                pl[i] = pu[i] - rng.f32_in(0.1, 1.0);
            }
        }
    }
    let pre_activation = BoundedTensor::new(
        Array1::from_vec(pl.clone()).into_dyn(),
        Array1::from_vec(pu.clone()).into_dyn(),
    )
    .unwrap();
    let alpha = Array1::from_vec((0..num_in).map(|i| (i as f32 % 7.0) / 7.0).collect());

    let (result, gradient) =
        crown_relu_backward_patches_with_alpha(&bounds, &pre_activation, &alpha).unwrap();
    let CrownBounds::Patches(result) = result else {
        panic!("expected Patches output");
    };

    // Reference: materialize the identity tensors, then run the indexed reference
    // over a bounds struct holding the materialized patches.
    let lower_mat = bounds.lower_a.materialize_identity();
    let upper_mat = bounds.upper_a.materialize_identity();
    let ref_bounds = PatchesLinearBounds {
        row_count: num_outputs,
        lower_a: lower_mat,
        lower_b: Array1::zeros(num_outputs),
        upper_a: upper_mat,
        upper_b: Array1::zeros(num_outputs),
    };
    let (ref_lp, ref_lb, ref_up, ref_ub, ref_grad) =
        reference_patches_alpha(&ref_bounds, &pl, &pu, alpha.as_slice().unwrap());

    assert_bits_eq(
        "identity lower_patches",
        result.lower_a.patches.as_ref().unwrap().as_slice().unwrap(),
        ref_lp.as_slice().unwrap(),
    );
    assert_bits_eq(
        "identity upper_patches",
        result.upper_a.patches.as_ref().unwrap().as_slice().unwrap(),
        ref_up.as_slice().unwrap(),
    );
    assert_bits_eq(
        "identity lower_b",
        result.lower_b.as_slice().unwrap(),
        ref_lb.as_slice().unwrap(),
    );
    assert_bits_eq(
        "identity upper_b",
        result.upper_b.as_slice().unwrap(),
        ref_ub.as_slice().unwrap(),
    );
    assert_bits_eq(
        "identity gradient",
        gradient.as_slice().unwrap(),
        ref_grad.as_slice().unwrap(),
    );
}

// =====================================================================
// Byte-identity regression pin (#patches-coeff-err-soundness; 7D
// explicit-rows closure spec §7.4 T2, docs/PATCHES_7D_COEFF_ERR_CLOSURE.md).
//
// Committed against the UNMODIFIED tree: with nonzero incoming coeff_err
// on BOTH sides, the production alpha backward must be bit-identical —
// coeff_err arrays, biases (including the intercept-error discharge
// applied BEFORE the directed cast), patch tensors, and gradient — to an
// independent in-test transcription of the CURRENT 6D rule. The 7D
// closure adds a 7D arm beside this one and must keep the 6D arm
// byte-for-byte unchanged, so this test must pass unmodified after it
// lands.
// =====================================================================

/// Independent transcription of the CURRENT production 6-D alpha path
/// INCLUDING the certified coeff_err rule: compose loop, per-row error block
/// (slope-envelope + exact directed-rounding gap max + intercept discharge
/// into the f64 bias BEFORE the directed cast), directed casts, and the
/// non-finite row zeroing.
///
/// Returns (lower_patches, lower_b, upper_patches, upper_b, gradient,
/// lower_coeff_err, upper_coeff_err).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn reference_patches_alpha_with_err(
    bounds: &PatchesLinearBounds,
    pre_lower: &[f32],
    pre_upper: &[f32],
    alpha: &[f32],
) -> (
    ArrayD<f32>,
    Array1<f32>,
    ArrayD<f32>,
    Array1<f32>,
    Array1<f32>,
    Array1<f32>,
    Array1<f32>,
) {
    let (out_c, out_h, out_w) = bounds.lower_a.output_shape;
    let (in_c_shape, in_h_shape, in_w_shape) = bounds.lower_a.input_shape;
    let num_outputs = out_c * out_h * out_w;
    let num_input_neurons = in_c_shape * in_h_shape * in_w_shape;

    let relaxations: Vec<LinearRelaxation> = (0..num_input_neurons)
        .map(|i| reference_relaxation(pre_lower[i], pre_upper[i], alpha[i]))
        .collect();
    let is_crossing: Vec<bool> = (0..num_input_neurons)
        .map(|i| {
            let (l, u) = (pre_lower[i], pre_upper[i]);
            !l.is_nan() && !u.is_nan() && l < 0.0 && u > 0.0
        })
        .collect();

    let lower_patches = bounds.lower_a.patches.as_ref().unwrap();
    let upper_patches = bounds.upper_a.patches.as_ref().unwrap();
    let shape = lower_patches.shape();
    let (in_c, kh, kw) = (shape[3], shape[4], shape[5]);

    let (sh, sw) = bounds.lower_a.stride;
    let (pad_left, _pr, pad_top, _pb) = bounds.lower_a.padding;

    let mut new_lower_patches = ArrayD::<f32>::zeros(lower_patches.raw_dim());
    let mut new_upper_patches = ArrayD::<f32>::zeros(upper_patches.raw_dim());
    let mut new_lower_b_f64 = bounds.lower_b.mapv(|x| x as f64);
    let mut new_upper_b_f64 = bounds.upper_b.mapv(|x| x as f64);
    let mut lower_nonfinite = vec![false; num_outputs];
    let mut upper_nonfinite = vec![false; num_outputs];
    let mut gradient = Array1::<f32>::zeros(num_input_neurons);

    // --- compose loop (verbatim structure of the production 6D branch) ---
    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let j = oc * out_h * out_w + oh * out_w + ow;
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                            let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                            if ih_raw < 0
                                || (ih_raw as usize) >= in_h_shape
                                || iw_raw < 0
                                || (iw_raw as usize) >= in_w_shape
                            {
                                continue;
                            }
                            let ih = ih_raw as usize;
                            let iw = iw_raw as usize;
                            let input_flat = ic * in_h_shape * in_w_shape + ih * in_w_shape + iw;
                            let relax = &relaxations[input_flat];

                            let la = lower_patches[[oc, oh, ow, ic, ki, kj]];
                            let lr = compose::compose_lower(la, relax);
                            new_lower_patches[[oc, oh, ow, ic, ki, kj]] = lr.new_coeff;
                            new_lower_b_f64[j] += lr.intercept_contrib;
                            lower_nonfinite[j] |= lr.nonfinite;

                            if la > 0.0 && is_crossing[input_flat] {
                                gradient[input_flat] += la * pre_lower[input_flat];
                            }

                            let ua = upper_patches[[oc, oh, ow, ic, ki, kj]];
                            let ur = compose::compose_upper(ua, relax);
                            new_upper_patches[[oc, oh, ow, ic, ki, kj]] = ur.new_coeff;
                            new_upper_b_f64[j] += ur.intercept_contrib;
                            upper_nonfinite[j] |= ur.nonfinite;
                        }
                    }
                }
            }
        }
    }

    // --- coeff_err block (independent transcription of the CURRENT 6D rule) ---
    let old_lower_err = bounds.lower_a.coeff_err.as_ref();
    let old_upper_err = bounds.upper_a.coeff_err.as_ref();
    let mut new_lower_err = Array1::<f32>::zeros(num_outputs);
    let mut new_upper_err = Array1::<f32>::zeros(num_outputs);
    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let j = oc * out_h * out_w + oh * out_w + ow;
                let oe_l =
                    old_lower_err.map_or(0.0, |e| f64::from(e.get(j).copied().unwrap_or(0.0)));
                let oe_u =
                    old_upper_err.map_or(0.0, |e| f64::from(e.get(j).copied().unwrap_or(0.0)));

                let mut max_slope_sum = 0.0f64;
                let mut int_sum = 0.0f64;
                let mut max_lower_gap = 0.0f64;
                let mut max_upper_gap = 0.0f64;
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                            let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                            if ih_raw < 0
                                || (ih_raw as usize) >= in_h_shape
                                || iw_raw < 0
                                || (iw_raw as usize) >= in_w_shape
                            {
                                continue;
                            }
                            let ih = ih_raw as usize;
                            let iw = iw_raw as usize;
                            let input_flat = ic * in_h_shape * in_w_shape + ih * in_w_shape + iw;
                            let relax = &relaxations[input_flat];

                            let ss = f64::from(relax.lower_slope).abs()
                                + f64::from(relax.upper_slope).abs();
                            if ss > max_slope_sum {
                                max_slope_sum = ss;
                            }
                            int_sum += f64::from(relax.lower_intercept).abs()
                                + f64::from(relax.upper_intercept).abs();

                            let la = lower_patches[[oc, oh, ow, ic, ki, kj]];
                            if la != 0.0 {
                                let slope_used = if la > 0.0 {
                                    f64::from(relax.lower_slope)
                                } else {
                                    f64::from(relax.upper_slope)
                                };
                                let stored = f64::from(new_lower_patches[[oc, oh, ow, ic, ki, kj]]);
                                let gap = (f64::from(la) * slope_used - stored).abs();
                                if gap > max_lower_gap {
                                    max_lower_gap = gap;
                                }
                            }
                            let ua = upper_patches[[oc, oh, ow, ic, ki, kj]];
                            if ua != 0.0 {
                                let slope_used = if ua > 0.0 {
                                    f64::from(relax.upper_slope)
                                } else {
                                    f64::from(relax.lower_slope)
                                };
                                let stored = f64::from(new_upper_patches[[oc, oh, ow, ic, ki, kj]]);
                                let gap = (f64::from(ua) * slope_used - stored).abs();
                                if gap > max_upper_gap {
                                    max_upper_gap = gap;
                                }
                            }
                        }
                    }
                }

                if oe_l != 0.0 {
                    let disc_l = oe_l * int_sum;
                    if disc_l.is_finite() {
                        new_lower_b_f64[j] -= disc_l;
                    } else {
                        new_lower_b_f64[j] = f64::NEG_INFINITY;
                    }
                }
                if oe_u != 0.0 {
                    let disc_u = oe_u * int_sum;
                    if disc_u.is_finite() {
                        new_upper_b_f64[j] += disc_u;
                    } else {
                        new_upper_b_f64[j] = f64::INFINITY;
                    }
                }

                let lterm = if oe_l != 0.0 {
                    oe_l * max_slope_sum
                } else {
                    0.0
                };
                let uterm = if oe_u != 0.0 {
                    oe_u * max_slope_sum
                } else {
                    0.0
                };
                new_lower_err[j] = if lower_nonfinite[j] {
                    0.0
                } else {
                    next_up_f32((lterm + max_lower_gap) as f32)
                };
                new_upper_err[j] = if upper_nonfinite[j] {
                    0.0
                } else {
                    next_up_f32((uterm + max_upper_gap) as f32)
                };
            }
        }
    }

    // --- directed casts + non-finite row zeroing (verbatim) ---
    let mut new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
    let mut new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));
    for j in 0..num_outputs {
        let oc = j / (out_h * out_w);
        let oh = (j % (out_h * out_w)) / out_w;
        let ow = j % out_w;
        if lower_nonfinite[j] {
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        new_lower_patches[[oc, oh, ow, ic, ki, kj]] = 0.0;
                    }
                }
            }
            new_lower_b[j] = f32::NEG_INFINITY;
        }
        if upper_nonfinite[j] {
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        new_upper_patches[[oc, oh, ow, ic, ki, kj]] = 0.0;
                    }
                }
            }
            new_upper_b[j] = f32::INFINITY;
        }
    }

    (
        new_lower_patches,
        new_lower_b,
        new_upper_patches,
        new_upper_b,
        gradient,
        new_lower_err,
        new_upper_err,
    )
}

/// Spec §7.4 T2 pin: 6D geometries WITH nonzero incoming errs — production
/// output must be bit-identical to the in-test transcription of the CURRENT
/// 6D coeff_err rule (errs, biases, patch tensors, gradient).
#[test]
fn test_patches_alpha_6d_coeff_err_regression_bit_identical() {
    let mut rng = Lcg::new(0xA11C_E5D4_71B2_9F03);

    // Same geometry family as the optimized-vs-reference test above.
    // (out_c, out_h, out_w, in_c, in_h, in_w, kh, kw, stride, pad_l, pad_t)
    let configs: &[(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    )] = &[
        (1, 1, 1, 1, 1, 2, 1, 2, 1, 0, 0),
        (2, 3, 3, 2, 3, 3, 1, 1, 1, 0, 0),
        (3, 4, 4, 2, 6, 6, 3, 3, 1, 1, 1),
        (2, 2, 2, 3, 5, 5, 3, 3, 2, 1, 1),
        (1, 3, 5, 1, 3, 5, 1, 1, 1, 0, 0),
    ];

    for (cfg_idx, &(out_c, out_h, out_w, in_c, in_h, in_w, kh, kw, stride, pad_l, pad_t)) in
        configs.iter().enumerate()
    {
        for regime in 0..4 {
            let num_outputs = out_c * out_h * out_w;
            let num_in = in_c * in_h * in_w;

            let lower_vec: Vec<f32> = (0..num_outputs * in_c * kh * kw)
                .map(|_| rng.f32_in(-3.0, 3.0))
                .collect();
            let upper_vec: Vec<f32> = (0..num_outputs * in_c * kh * kw)
                .map(|_| rng.f32_in(-3.0, 3.0))
                .collect();

            let lower_patches =
                ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w, in_c, kh, kw]), lower_vec)
                    .unwrap();
            let upper_patches =
                ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w, in_c, kh, kw]), upper_vec)
                    .unwrap();

            let lower_b: Vec<f32> = (0..num_outputs).map(|_| rng.f32_in(-1.0, 1.0)).collect();
            let upper_b: Vec<f32> = (0..num_outputs).map(|_| rng.f32_in(-1.0, 1.0)).collect();

            // Nonzero incoming coeff_err on both sides, with exact-zero rows
            // sprinkled in so the `oe == 0` short-circuit is also pinned.
            let lower_err: Vec<f32> = (0..num_outputs)
                .map(|j| {
                    if j % 5 == 4 {
                        0.0
                    } else {
                        rng.f32_in(1e-6, 1e-3)
                    }
                })
                .collect();
            let upper_err: Vec<f32> = (0..num_outputs)
                .map(|j| {
                    if j % 7 == 3 {
                        0.0
                    } else {
                        rng.f32_in(1e-6, 2e-3)
                    }
                })
                .collect();

            let bounds = PatchesLinearBounds {
                row_count: num_outputs,
                lower_a: PatchesData {
                    coeff_err: Some(Array1::from_vec(lower_err)),
                    patches: Some(lower_patches),
                    stride: (stride, stride),
                    padding: (pad_l, pad_l, pad_t, pad_t),
                    identity: false,
                    output_shape: (out_c, out_h, out_w),
                    input_shape: (in_c, in_h, in_w),
                    unstable_idx: None,
                },
                lower_b: Array1::from_vec(lower_b),
                upper_a: PatchesData {
                    coeff_err: Some(Array1::from_vec(upper_err)),
                    patches: Some(upper_patches),
                    stride: (stride, stride),
                    padding: (pad_l, pad_l, pad_t, pad_t),
                    identity: false,
                    output_shape: (out_c, out_h, out_w),
                    input_shape: (in_c, in_h, in_w),
                    unstable_idx: None,
                },
                upper_b: Array1::from_vec(upper_b),
            };

            let mut pl = vec![0.0f32; num_in];
            let mut pu = vec![0.0f32; num_in];
            for i in 0..num_in {
                match regime {
                    1 => {
                        pl[i] = rng.f32_in(0.1, 2.0);
                        pu[i] = pl[i] + rng.f32_in(0.1, 2.0);
                    }
                    2 => {
                        pu[i] = rng.f32_in(-2.0, -0.1);
                        pl[i] = pu[i] - rng.f32_in(0.1, 2.0);
                    }
                    3 => {
                        if i % 2 == 0 {
                            pl[i] = rng.f32_in(-2.0, -0.1);
                            pu[i] = rng.f32_in(0.1, 2.0);
                        } else {
                            pl[i] = rng.f32_in(0.1, 1.0);
                            pu[i] = pl[i] + rng.f32_in(0.1, 1.0);
                        }
                    }
                    _ => {
                        pl[i] = rng.f32_in(-3.0, -0.1);
                        pu[i] = rng.f32_in(0.1, 3.0);
                    }
                }
            }

            let pre_activation = BoundedTensor::new(
                Array1::from_vec(pl.clone()).into_dyn(),
                Array1::from_vec(pu.clone()).into_dyn(),
            )
            .unwrap();

            let alpha_vec: Vec<f32> = (0..num_in)
                .map(|i| match i % 3 {
                    0 => 0.0,
                    1 => 1.0,
                    _ => rng.f32_in(0.0, 1.0),
                })
                .collect();
            let alpha = Array1::from_vec(alpha_vec);

            let (result, gradient) =
                crown_relu_backward_patches_with_alpha(&bounds, &pre_activation, &alpha).unwrap();
            let CrownBounds::Patches(result) = result else {
                panic!("cfg {cfg_idx} regime {regime}: expected Patches output");
            };

            let (ref_lp, ref_lb, ref_up, ref_ub, ref_grad, ref_le, ref_ue) =
                reference_patches_alpha_with_err(&bounds, &pl, &pu, alpha.as_slice().unwrap());

            let opt_le =
                result.lower_a.coeff_err.as_ref().unwrap_or_else(|| {
                    panic!("cfg {cfg_idx} regime {regime}: lower coeff_err None")
                });
            let opt_ue =
                result.upper_a.coeff_err.as_ref().unwrap_or_else(|| {
                    panic!("cfg {cfg_idx} regime {regime}: upper coeff_err None")
                });

            let tag = format!("err cfg {cfg_idx} regime {regime}");
            assert_bits_eq(
                &format!("{tag} lower coeff_err"),
                opt_le.as_slice().unwrap(),
                ref_le.as_slice().unwrap(),
            );
            assert_bits_eq(
                &format!("{tag} upper coeff_err"),
                opt_ue.as_slice().unwrap(),
                ref_ue.as_slice().unwrap(),
            );
            assert_bits_eq(
                &format!("{tag} lower_b"),
                result.lower_b.as_slice().unwrap(),
                ref_lb.as_slice().unwrap(),
            );
            assert_bits_eq(
                &format!("{tag} upper_b"),
                result.upper_b.as_slice().unwrap(),
                ref_ub.as_slice().unwrap(),
            );
            assert_bits_eq(
                &format!("{tag} lower_patches"),
                result.lower_a.patches.as_ref().unwrap().as_slice().unwrap(),
                ref_lp.as_slice().unwrap(),
            );
            assert_bits_eq(
                &format!("{tag} upper_patches"),
                result.upper_a.patches.as_ref().unwrap().as_slice().unwrap(),
                ref_up.as_slice().unwrap(),
            );
            assert_bits_eq(
                &format!("{tag} gradient"),
                gradient.as_slice().unwrap(),
                ref_grad.as_slice().unwrap(),
            );
        }
    }
}

// =====================================================================
// 7D explicit-rows coeff_err closure (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md
// §7.4): f64 oracle coverage (T1 + None-in sub-case), guard tests (T3,
// mixed-ndim D3, non-finite incoming err I5), and non-finite coefficient
// row handling (T4).
// =====================================================================

use ny_core::NyError;

/// Build a 7D explicit-rows `PatchesLinearBounds` fixture (stride 1).
///
/// `shape7 = [rows, out_c, out_h, out_w, in_c, kh, kw]`; both sides share the
/// geometry (metadata mirrors `from_dense_spatial_rows` output plus padding).
#[allow(clippy::too_many_arguments)]
fn make_alpha_7d_bounds(
    shape7: &[usize],
    input_shape: (usize, usize, usize),
    padding: (usize, usize, usize, usize),
    lower_vec: Vec<f32>,
    upper_vec: Vec<f32>,
    lower_b: Vec<f32>,
    upper_b: Vec<f32>,
    lower_err: Option<Array1<f32>>,
    upper_err: Option<Array1<f32>>,
) -> PatchesLinearBounds {
    let rows = shape7[0];
    let output_shape = (shape7[1], shape7[2], shape7[3]);
    PatchesLinearBounds {
        row_count: rows,
        lower_a: PatchesData {
            coeff_err: lower_err,
            patches: Some(ArrayD::from_shape_vec(IxDyn(shape7), lower_vec).unwrap()),
            stride: (1, 1),
            padding,
            identity: false,
            output_shape,
            input_shape,
            unstable_idx: None,
        },
        lower_b: Array1::from_vec(lower_b),
        upper_a: PatchesData {
            coeff_err: upper_err,
            patches: Some(ArrayD::from_shape_vec(IxDyn(shape7), upper_vec).unwrap()),
            stride: (1, 1),
            padding,
            identity: false,
            output_shape,
            input_shape,
            unstable_idx: None,
        },
        upper_b: Array1::from_vec(upper_b),
    }
}

/// f64-oracle certificate check for one side of a 7D explicit-rows alpha
/// backward result (spec §7.4 T1).
///
/// The composed true coefficient `c̃(ã) = ã·σ(ã)` and intercept contribution
/// `ã·i(ã)` are piecewise linear in the admissible true coefficient
/// `ã ∈ [a−e, a+e]` with their ONLY breakpoint at 0, so the interval extremes
/// are attained at the endpoints or the kink — the candidate set
/// `{a−e, a+e} ∪ {0 if straddling}` is exhaustive. Asserts, per spec row:
/// - every candidate composed coefficient is covered by the emitted row err;
/// - out-of-bounds (padding) taps stay exactly `+0.0`;
/// - the stored bias is OUTWARD of the worst-case true intercept fold
///   (min over candidates for lower, max for upper) — with `e = 0` this
///   degenerates to the exact f64 fold and pins the `γ̄·ABS` fold discharge
///   as present-and-outward (spec §14 A1).
///
/// Oracle-noise note: the oracle folds/gaps carry ~2^-53-relative f64
/// rounding while the production values sit at least one directed-f32-cast
/// step (≥ 2^-25 relative) outward — strict comparisons are safe.
#[allow(clippy::too_many_lines)]
fn check_alpha_7d_side(
    tag: &str,
    is_lower: bool,
    input: &PatchesLinearBounds,
    result: &PatchesLinearBounds,
    pre_lower: &[f32],
    pre_upper: &[f32],
    alpha: &[f32],
) {
    let (side_in, side_out, old_b, new_b) = if is_lower {
        (
            &input.lower_a,
            &result.lower_a,
            &input.lower_b,
            &result.lower_b,
        )
    } else {
        (
            &input.upper_a,
            &result.upper_a,
            &input.upper_b,
            &result.upper_b,
        )
    };
    let old_patches = side_in.patches.as_ref().unwrap();
    let new_patches = side_out.patches.as_ref().unwrap();
    let err_out = side_out
        .coeff_err
        .as_ref()
        .unwrap_or_else(|| panic!("{tag}: 7D output must carry coeff_err Some"));
    assert_eq!(err_out.len(), input.row_count, "{tag}: err length");

    let (sh, sw) = side_in.stride;
    let (pad_left, _pr, pad_top, _pb) = side_in.padding;
    let (out_c, out_h, out_w) = side_in.output_shape;
    let (in_c_shape, in_h, in_w) = side_in.input_shape;
    let shp = old_patches.shape();
    let (in_c, kh, kw) = (shp[4], shp[5], shp[6]);

    let relax: Vec<LinearRelaxation> = (0..in_c_shape * in_h * in_w)
        .map(|i| reference_relaxation(pre_lower[i], pre_upper[i], alpha[i]))
        .collect();

    for row in 0..input.row_count {
        let e = side_in
            .coeff_err
            .as_ref()
            .map_or(0.0f64, |a| f64::from(a[row]));
        let err_r = f64::from(err_out[row]);
        assert!(
            err_r.is_finite() && err_r >= 0.0,
            "{tag} row {row}: err must be finite and >= 0, got {err_r}"
        );
        // Worst-case true intercept fold (min for lower / max for upper).
        let mut fold_extreme = f64::from(old_b[row]);
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                                let stored_f32 = new_patches[[row, oc, oh, ow, ic, ki, kj]];
                                if ih_raw < 0
                                    || (ih_raw as usize) >= in_h
                                    || iw_raw < 0
                                    || (iw_raw as usize) >= in_w
                                {
                                    assert_eq!(
                                        stored_f32.to_bits(),
                                        0.0f32.to_bits(),
                                        "{tag} row {row} tap ({oc},{oh},{ow},{ic},{ki},{kj}): \
                                         padding tap must stay exactly +0.0"
                                    );
                                    continue;
                                }
                                let input_flat =
                                    ic * in_h * in_w + (ih_raw as usize) * in_w + iw_raw as usize;
                                let r = &relax[input_flat];
                                let a = f64::from(old_patches[[row, oc, oh, ow, ic, ki, kj]]);
                                let stored = f64::from(stored_f32);

                                let mut cands = vec![a - e, a + e];
                                if a - e < 0.0 && a + e > 0.0 {
                                    cands.push(0.0);
                                }
                                let mut contrib_min = f64::INFINITY;
                                let mut contrib_max = f64::NEG_INFINITY;
                                for &c in &cands {
                                    // Slope/intercept selection per
                                    // compose_lower / compose_upper on the
                                    // TRUE coefficient sign.
                                    let (slope, intercept) = if c > 0.0 {
                                        if is_lower {
                                            (r.lower_slope, r.lower_intercept)
                                        } else {
                                            (r.upper_slope, r.upper_intercept)
                                        }
                                    } else if c < 0.0 {
                                        if is_lower {
                                            (r.upper_slope, r.upper_intercept)
                                        } else {
                                            (r.lower_slope, r.lower_intercept)
                                        }
                                    } else {
                                        (0.0f32, 0.0f32)
                                    };
                                    let ctrue = c * f64::from(slope);
                                    let dev = (stored - ctrue).abs();
                                    assert!(
                                        err_r >= dev,
                                        "{tag} row {row} tap \
                                         ({oc},{oh},{ow},{ic},{ki},{kj}) cand {c}: \
                                         err {err_r:e} < |stored {stored:e} - true \
                                         {ctrue:e}| = {dev:e}"
                                    );
                                    let contrib = c * f64::from(intercept);
                                    contrib_min = contrib_min.min(contrib);
                                    contrib_max = contrib_max.max(contrib);
                                }
                                fold_extreme += if is_lower { contrib_min } else { contrib_max };
                            }
                        }
                    }
                }
            }
        }
        if is_lower {
            assert!(
                f64::from(new_b[row]) <= fold_extreme,
                "{tag} row {row}: lower_b {} not outward of worst-case true fold {}",
                new_b[row],
                fold_extreme
            );
        } else {
            assert!(
                f64::from(new_b[row]) >= fold_extreme,
                "{tag} row {row}: upper_b {} not outward of worst-case true fold {}",
                new_b[row],
                fold_extreme
            );
        }
    }
}

/// Spec §7.4 T1: 7D explicit-rows with padding (out-of-bounds corner taps),
/// mixed relaxation regimes, alpha ∈ {0, 1, 1/3, rng}, exact-zero and tiny
/// (sign-straddling) coefficients, asymmetric per-row incoming errs on both
/// sides. Kink-argument f64 oracle over coefficients and biases.
///
/// Sub-case None-in: output errs still Some (the exact directed-rounding gap
/// is intrinsic); coefficient tensors and gradient BIT-IDENTICAL to the
/// Some-in run (the err pass is read-only over coefficients — spec I3); the
/// e=0 oracle pins the `γ̄·ABS` fold discharge outward (spec §14 A1 — note
/// this deliberately supersedes the pre-adjudication §7.4 wording that
/// None-in biases stay bit-identical: the adopted fold discharge moves them
/// outward); and the Some-in discharge is strictly live vs the None-in run.
#[test]
fn test_patches_alpha_7d_coeff_err_covers_true_deviation() {
    let mut rng = Lcg::new(0x7D5E_C7A1_0B3F_D219);
    let shape7 = [2usize, 2, 2, 2, 2, 2, 2];
    let (in_c, in_h, in_w) = (2usize, 2usize, 2usize);
    let num_in = in_c * in_h * in_w;
    let n_coeff: usize = shape7.iter().product();

    fn fill(rng: &mut Lcg, i: usize) -> f32 {
        if i % 7 == 3 {
            0.0 // exact structural-style zeros (a==0 envelope case)
        } else if i % 5 == 2 {
            rng.f32_in(-8e-4, 8e-4) // tiny: straddles zero under e ~ 1e-3
        } else {
            rng.f32_in(-2.0, 2.0)
        }
    }
    let lower_vec: Vec<f32> = (0..n_coeff).map(|i| fill(&mut rng, i)).collect();
    let upper_vec: Vec<f32> = (0..n_coeff).map(|i| fill(&mut rng, i)).collect();
    let lower_b = vec![0.375f32, -0.25];
    let upper_b = vec![0.5f32, -0.125];
    let errs = Array1::from_vec(vec![1e-3f32, 5e-4]);

    let mut pl = vec![0.0f32; num_in];
    let mut pu = vec![0.0f32; num_in];
    let mut alpha_vec = vec![0.0f32; num_in];
    for i in 0..num_in {
        match i % 4 {
            0 => {
                pl[i] = -1.25;
                pu[i] = 0.75;
            }
            1 => {
                pl[i] = 0.3;
                pu[i] = 1.7;
            }
            2 => {
                pl[i] = -1.5;
                pu[i] = -0.2;
            }
            _ => {
                pl[i] = rng.f32_in(-2.0, -0.1);
                pu[i] = rng.f32_in(0.1, 2.0);
            }
        }
        alpha_vec[i] = match i % 4 {
            0 => 0.0,
            1 => 1.0,
            2 => 1.0 / 3.0,
            _ => rng.f32_in(0.05, 0.95),
        };
    }
    let pre = BoundedTensor::new(
        Array1::from_vec(pl.clone()).into_dyn(),
        Array1::from_vec(pu.clone()).into_dyn(),
    )
    .unwrap();
    let alpha = Array1::from_vec(alpha_vec);

    // Padding (1,1,1,1) with 2x2 input: corner taps go out of bounds.
    let bounds = make_alpha_7d_bounds(
        &shape7,
        (in_c, in_h, in_w),
        (1, 1, 1, 1),
        lower_vec.clone(),
        upper_vec.clone(),
        lower_b.clone(),
        upper_b.clone(),
        Some(errs.clone()),
        Some(errs),
    );
    let (result, gradient) = crown_relu_backward_patches_with_alpha(&bounds, &pre, &alpha).unwrap();
    let CrownBounds::Patches(result) = result else {
        panic!("expected Patches output");
    };

    check_alpha_7d_side(
        "T1 lower",
        true,
        &bounds,
        &result,
        &pl,
        &pu,
        alpha.as_slice().unwrap(),
    );
    check_alpha_7d_side(
        "T1 upper",
        false,
        &bounds,
        &result,
        &pl,
        &pu,
        alpha.as_slice().unwrap(),
    );

    // --- Sub-case: None-in ---
    let bounds_none = make_alpha_7d_bounds(
        &shape7,
        (in_c, in_h, in_w),
        (1, 1, 1, 1),
        lower_vec,
        upper_vec,
        lower_b,
        upper_b,
        None,
        None,
    );
    let (result_none, gradient_none) =
        crown_relu_backward_patches_with_alpha(&bounds_none, &pre, &alpha).unwrap();
    let CrownBounds::Patches(result_none) = result_none else {
        panic!("expected Patches output (None-in)");
    };

    // Gap-only certificate with e = 0 (also pins the γ̄·ABS discharge outward).
    check_alpha_7d_side(
        "T1 lower None-in",
        true,
        &bounds_none,
        &result_none,
        &pl,
        &pu,
        alpha.as_slice().unwrap(),
    );
    check_alpha_7d_side(
        "T1 upper None-in",
        false,
        &bounds_none,
        &result_none,
        &pl,
        &pu,
        alpha.as_slice().unwrap(),
    );

    // The err pass is read-only over coefficients (spec I3): the composed
    // patch tensors and the gradient must be bit-identical across Some-in
    // and None-in runs.
    assert_bits_eq(
        "T1 lower patches Some-in vs None-in",
        result.lower_a.patches.as_ref().unwrap().as_slice().unwrap(),
        result_none
            .lower_a
            .patches
            .as_ref()
            .unwrap()
            .as_slice()
            .unwrap(),
    );
    assert_bits_eq(
        "T1 upper patches Some-in vs None-in",
        result.upper_a.patches.as_ref().unwrap().as_slice().unwrap(),
        result_none
            .upper_a
            .patches
            .as_ref()
            .unwrap()
            .as_slice()
            .unwrap(),
    );
    assert_bits_eq(
        "T1 gradient Some-in vs None-in",
        gradient.as_slice().unwrap(),
        gradient_none.as_slice().unwrap(),
    );

    // Discharge liveness: with nonzero incoming errs the intercept-envelope
    // discharge `oe·(IS·(1+γ̄))` must move the biases strictly outward of the
    // None-in run (which only carries the tiny γ̄·ABS fold discharge).
    let mut lower_strict = false;
    let mut upper_strict = false;
    for row in 0..2 {
        assert!(
            result.lower_b[row] <= result_none.lower_b[row],
            "row {row}: Some-in lower_b must be <= None-in lower_b"
        );
        assert!(
            result.upper_b[row] >= result_none.upper_b[row],
            "row {row}: Some-in upper_b must be >= None-in upper_b"
        );
        lower_strict |= result.lower_b[row] < result_none.lower_b[row];
        upper_strict |= result.upper_b[row] > result_none.upper_b[row];
    }
    assert!(
        lower_strict && upper_strict,
        "incoming-err discharge must be live on at least one row per side"
    );
}

/// Spec §7.4 T3: a `Some` incoming coeff_err whose length differs from
/// `row_count` is a hard `Err(ShapeMismatch)` on the 7D path — never the
/// 6D-style silent `.get(i).unwrap_or(0.0)` under-count (spec I6). Both sides.
#[test]
fn test_patches_alpha_7d_coeff_err_length_mismatch_rejected() {
    let shape7 = [2usize, 1, 1, 2, 1, 1, 2];
    let n_coeff: usize = shape7.iter().product();
    let vals: Vec<f32> = (0..n_coeff).map(|i| 0.1 + i as f32 * 0.05).collect();
    let pre = BoundedTensor::new(
        Array1::from_vec(vec![-1.0f32; 3]).into_dyn(),
        Array1::from_vec(vec![1.0f32; 3]).into_dyn(),
    )
    .unwrap();
    let alpha = Array1::from_vec(vec![0.5f32; 3]);

    for (lower_err, upper_err, side) in [
        (Some(Array1::from_vec(vec![1e-3f32; 3])), None, "lower"),
        (None, Some(Array1::from_vec(vec![1e-3f32; 1])), "upper"),
    ] {
        let bounds = make_alpha_7d_bounds(
            &shape7,
            (1, 1, 3),
            (0, 0, 0, 0),
            vals.clone(),
            vals.clone(),
            vec![0.0f32; 2],
            vec![0.0f32; 2],
            lower_err,
            upper_err,
        );
        let err = crown_relu_backward_patches_with_alpha(&bounds, &pre, &alpha)
            .expect_err(&format!("{side}: wrong-length err must be rejected"));
        assert!(
            matches!(err, NyError::ShapeMismatch { .. }),
            "{side}: expected ShapeMismatch, got {err:?}"
        );
    }
}

/// Mixed-layout guard (spec §14 D3): a 6D/7D side pair must be rejected with
/// a clean `ShapeMismatch` instead of panicking inside the compose loop.
#[test]
fn test_patches_alpha_mixed_ndim_pair_rejected() {
    let shape7 = [2usize, 1, 1, 2, 1, 1, 2];
    let shape6 = [1usize, 1, 2, 1, 1, 2];
    let n7: usize = shape7.iter().product();
    let n6: usize = shape6.iter().product();
    let vals7: Vec<f32> = (0..n7).map(|i| 0.1 + i as f32 * 0.05).collect();
    let vals6: Vec<f32> = (0..n6).map(|i| 0.1 + i as f32 * 0.05).collect();
    let pre = BoundedTensor::new(
        Array1::from_vec(vec![-1.0f32; 3]).into_dyn(),
        Array1::from_vec(vec![1.0f32; 3]).into_dyn(),
    )
    .unwrap();
    let alpha = Array1::from_vec(vec![0.5f32; 3]);

    // (lower shape, upper shape) in both orders.
    for (ls, lv, us, uv) in [
        (&shape7[..], &vals7, &shape6[..], &vals6),
        (&shape6[..], &vals6, &shape7[..], &vals7),
    ] {
        let bounds = PatchesLinearBounds {
            row_count: 2,
            lower_a: PatchesData {
                coeff_err: None,
                patches: Some(ArrayD::from_shape_vec(IxDyn(ls), lv.clone()).unwrap()),
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: false,
                output_shape: (1, 1, 2),
                input_shape: (1, 1, 3),
                unstable_idx: None,
            },
            lower_b: Array1::from_vec(vec![0.0f32; 2]),
            upper_a: PatchesData {
                coeff_err: None,
                patches: Some(ArrayD::from_shape_vec(IxDyn(us), uv.clone()).unwrap()),
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: false,
                output_shape: (1, 1, 2),
                input_shape: (1, 1, 3),
                unstable_idx: None,
            },
            upper_b: Array1::from_vec(vec![0.0f32; 2]),
        };
        let err = crown_relu_backward_patches_with_alpha(&bounds, &pre, &alpha)
            .expect_err("mixed 6D/7D side pair must be rejected");
        assert!(
            matches!(err, NyError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got {err:?}"
        );
    }
}

/// Spec §7.4 T4: an injected non-finite coefficient (|slope| <= 1, so
/// overflow only via non-finite input) makes the row non-finite: coefficients
/// zeroed, bias -INF, err exactly 0.0 (the vacuous certificate); the other
/// row and the other side are unaffected. No NaN anywhere — the `∞·0`
/// intercept-magnitude hazard inside ABS must resolve to the -INF poison.
#[test]
fn test_patches_alpha_7d_nonfinite_row_err_zero() {
    let shape7 = [2usize, 1, 1, 2, 1, 1, 2];
    let n_coeff: usize = shape7.iter().product();
    let mut lower_vec: Vec<f32> = vec![0.75, -0.5, 0.25, 0.6, 1.5, -0.75, 0.6, 0.9];
    assert_eq!(lower_vec.len(), n_coeff);
    lower_vec[0] = f32::INFINITY; // row 0, first tap: INF * alpha -> nonfinite
    let upper_vec: Vec<f32> = vec![0.5, -0.25, 0.8, -0.4, 1.1, 0.3, -0.9, 0.2];

    let pre = BoundedTensor::new(
        Array1::from_vec(vec![-1.0f32; 3]).into_dyn(),
        Array1::from_vec(vec![1.0f32; 3]).into_dyn(),
    )
    .unwrap();
    let alpha = Array1::from_vec(vec![0.5f32; 3]);

    let errs = Array1::from_vec(vec![1e-3f32, 1e-3]);
    let bounds = make_alpha_7d_bounds(
        &shape7,
        (1, 1, 3),
        (0, 0, 0, 0),
        lower_vec,
        upper_vec,
        vec![0.25f32, -0.5],
        vec![0.5f32, 0.75],
        Some(errs.clone()),
        Some(errs),
    );
    let (result, _gradient) =
        crown_relu_backward_patches_with_alpha(&bounds, &pre, &alpha).unwrap();
    let CrownBounds::Patches(result) = result else {
        panic!("expected Patches output");
    };

    let le = result.lower_a.coeff_err.as_ref().unwrap();
    let ue = result.upper_a.coeff_err.as_ref().unwrap();
    let lp = result.lower_a.patches.as_ref().unwrap();

    // Non-finite lower row 0: zeroed coefficients, -INF bias, err exactly 0.0.
    assert_eq!(
        le[0].to_bits(),
        0.0f32.to_bits(),
        "nonfinite row err must be 0.0"
    );
    assert_eq!(result.lower_b[0], f32::NEG_INFINITY);
    for tap in 0..4usize {
        assert_eq!(
            lp[[0, 0, 0, tap / 2, 0, 0, tap % 2]].to_bits(),
            0.0f32.to_bits(),
            "nonfinite row coefficients must be zeroed"
        );
    }
    // Other lower row unaffected: finite err > 0, finite bias.
    assert!(
        le[1].is_finite() && le[1] > 0.0,
        "row 1 lower err: {}",
        le[1]
    );
    assert!(result.lower_b[1].is_finite());
    // Upper side fully finite on both rows.
    for row in 0..2 {
        assert!(ue[row].is_finite() && ue[row] >= 0.0);
        assert!(result.upper_b[row].is_finite());
    }
    // No NaN anywhere.
    for v in result
        .lower_b
        .iter()
        .chain(result.upper_b.iter())
        .chain(le.iter())
        .chain(ue.iter())
    {
        assert!(!v.is_nan(), "NaN leaked into outputs");
    }
    for v in lp
        .iter()
        .chain(result.upper_a.patches.as_ref().unwrap().iter())
    {
        assert!(!v.is_nan(), "NaN leaked into patch tensors");
    }
}

/// I5 poison guard: a non-finite (NaN) or negative incoming per-row err is
/// sanitized to +INF at consumption — the affected row emits err +INF with a
/// ∓INF (vacuous) bias, other rows stay finite, and NaN NEVER reaches any
/// output (the NaN->0 false-proof hazard). The coefficient tensors are
/// untouched by the poison (the err pass is read-only over them).
#[test]
fn test_patches_alpha_7d_nonfinite_incoming_err_poisons_row() {
    let shape7 = [2usize, 1, 1, 2, 1, 1, 2];
    let n_coeff: usize = shape7.iter().product();
    let lower_vec: Vec<f32> = vec![0.75, -0.5, 0.25, 0.6, 1.5, -0.75, 0.6, 0.9];
    let upper_vec: Vec<f32> = vec![0.5, -0.25, 0.8, -0.4, 1.1, 0.3, -0.9, 0.2];
    assert_eq!(lower_vec.len(), n_coeff);

    let pre = BoundedTensor::new(
        Array1::from_vec(vec![-1.0f32; 3]).into_dyn(),
        Array1::from_vec(vec![1.0f32; 3]).into_dyn(),
    )
    .unwrap();
    let alpha = Array1::from_vec(vec![0.5f32; 3]);

    let build = |le: Option<Array1<f32>>, ue: Option<Array1<f32>>| {
        make_alpha_7d_bounds(
            &shape7,
            (1, 1, 3),
            (0, 0, 0, 0),
            lower_vec.clone(),
            upper_vec.clone(),
            vec![0.25f32, -0.5],
            vec![0.5f32, 0.75],
            le,
            ue,
        )
    };

    // NaN on lower row 0; NEGATIVE err on upper row 1.
    let bounds = build(
        Some(Array1::from_vec(vec![f32::NAN, 1e-3])),
        Some(Array1::from_vec(vec![1e-3f32, -1.0])),
    );
    let (result, _) = crown_relu_backward_patches_with_alpha(&bounds, &pre, &alpha).unwrap();
    let CrownBounds::Patches(result) = result else {
        panic!("expected Patches output");
    };
    let le = result.lower_a.coeff_err.as_ref().unwrap();
    let ue = result.upper_a.coeff_err.as_ref().unwrap();

    assert_eq!(le[0], f32::INFINITY, "NaN incoming err must poison to +INF");
    assert_eq!(result.lower_b[0], f32::NEG_INFINITY, "poisoned lower bias");
    assert!(
        le[1].is_finite() && result.lower_b[1].is_finite(),
        "clean lower row"
    );

    assert_eq!(
        ue[1],
        f32::INFINITY,
        "negative incoming err must poison to +INF"
    );
    assert_eq!(result.upper_b[1], f32::INFINITY, "poisoned upper bias");
    assert!(
        ue[0].is_finite() && result.upper_b[0].is_finite(),
        "clean upper row"
    );

    // NaN must never reach any output.
    for v in result
        .lower_b
        .iter()
        .chain(result.upper_b.iter())
        .chain(le.iter())
        .chain(ue.iter())
    {
        assert!(!v.is_nan(), "NaN leaked into outputs");
    }

    // The poison is bias/err-only: coefficient tensors bit-identical to an
    // err-free run (read-only err pass, spec I3).
    let bounds_clean = build(None, None);
    let (result_clean, _) =
        crown_relu_backward_patches_with_alpha(&bounds_clean, &pre, &alpha).unwrap();
    let CrownBounds::Patches(result_clean) = result_clean else {
        panic!("expected Patches output");
    };
    assert_bits_eq(
        "poison lower patches vs clean",
        result.lower_a.patches.as_ref().unwrap().as_slice().unwrap(),
        result_clean
            .lower_a
            .patches
            .as_ref()
            .unwrap()
            .as_slice()
            .unwrap(),
    );
    assert_bits_eq(
        "poison upper patches vs clean",
        result.upper_a.patches.as_ref().unwrap().as_slice().unwrap(),
        result_clean
            .upper_a
            .patches
            .as_ref()
            .unwrap()
            .as_slice()
            .unwrap(),
    );
}

// =====================================================================
// Operator parity: patches-mode alpha-CROWN == dense-mode alpha-CROWN.
//
// The wiring landed in `try_patches_target_step_core` (#conv-patches-collect
// alpha) routes crossing-ReLU nodes through
// `ReLULayer::propagate_patches_with_alpha` (the patches operator) instead of
// the dense `propagate_linear_with_alpha`. This test pins the SOUNDNESS
// contract the wiring relies on: for the SAME seed coefficients, pre-activation
// box, and lower-alpha vector, the patches operator's concretized output must
// equal the dense operator's (with `alpha_upper = None`, the single-alpha
// relaxation the patches path implements) within f32 tolerance, AND must be a
// valid ENCLOSURE (never tighter than the dense bound beyond tolerance) — over
// several alpha vectors and random seeds. The two operators multiply the
// per-neuron slope and col2im-scatter in opposite orders, which commute
// mathematically (one slope per input neuron) but round at different points; the
// patches path's certified coeff_err covers exactly that gap.
// =====================================================================
#[test]
fn test_patches_alpha_matches_dense_alpha_operator_conv_seed() {
    use crate::bounds::LinearBounds;
    use crate::layers::ReLULayer;
    use ndarray::Array1;

    // Small overlapping-receptive-field conv geometry so col2im sums multiple
    // taps per input neuron (the case where multiply/scatter order matters).
    // output (1,2,2) -> 4 rows; kernel 2x2, stride 1, pad 0; input (1,3,3) -> 9.
    let (out_c, out_h, out_w) = (1usize, 2usize, 2usize);
    let (in_c, in_h, in_w) = (1usize, 3usize, 3usize);
    let (kh, kw) = (2usize, 2usize);
    let num_outputs = out_c * out_h * out_w; // 4
    let num_inputs = in_c * in_h * in_w; // 9
    let patch_len = out_c * out_h * out_w * in_c * kh * kw; // 16

    // Local deterministic PRNG (independent of the equivalence module's Lcg).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
            let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
            lo + u * (hi - lo)
        }
    }

    let mut max_lower_diff = 0.0f32;
    let mut max_upper_diff = 0.0f32;
    let mut finite_cmp = 0usize;

    for seed in 0..6u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));

        // Random seed coefficients for the incoming patches bounds.
        let lower_patch: Vec<f32> = (0..patch_len).map(|_| rng.f32_in(-1.5, 1.5)).collect();
        let upper_patch: Vec<f32> = (0..patch_len).map(|_| rng.f32_in(-1.5, 1.5)).collect();
        let lower_b: Vec<f32> = (0..num_outputs).map(|_| rng.f32_in(-0.5, 0.5)).collect();
        let upper_b: Vec<f32> = (0..num_outputs).map(|_| rng.f32_in(-0.5, 0.5)).collect();

        // Pre-activation box: a mix of crossing / active / inactive neurons.
        let mut pre_lower = vec![0.0f32; num_inputs];
        let mut pre_upper = vec![0.0f32; num_inputs];
        for i in 0..num_inputs {
            match i % 3 {
                0 => {
                    // crossing
                    pre_lower[i] = rng.f32_in(-2.0, -0.2);
                    pre_upper[i] = rng.f32_in(0.2, 2.5);
                }
                1 => {
                    // always active
                    pre_lower[i] = rng.f32_in(0.1, 1.5);
                    pre_upper[i] = pre_lower[i] + rng.f32_in(0.2, 1.5);
                }
                _ => {
                    // always inactive
                    pre_upper[i] = rng.f32_in(-1.5, -0.1);
                    pre_lower[i] = pre_upper[i] - rng.f32_in(0.2, 1.5);
                }
            }
        }
        let pre = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[num_inputs]), pre_lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[num_inputs]), pre_upper.clone()).unwrap(),
        )
        .unwrap();

        let make_bounds = || PatchesLinearBounds {
            row_count: num_outputs,
            lower_a: PatchesData {
                coeff_err: None,
                patches: Some(
                    ArrayD::from_shape_vec(
                        IxDyn(&[out_c, out_h, out_w, in_c, kh, kw]),
                        lower_patch.clone(),
                    )
                    .unwrap(),
                ),
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: false,
                output_shape: (out_c, out_h, out_w),
                input_shape: (in_c, in_h, in_w),
                unstable_idx: None,
            },
            lower_b: Array1::from(lower_b.clone()),
            upper_a: PatchesData {
                coeff_err: None,
                patches: Some(
                    ArrayD::from_shape_vec(
                        IxDyn(&[out_c, out_h, out_w, in_c, kh, kw]),
                        upper_patch.clone(),
                    )
                    .unwrap(),
                ),
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: false,
                output_shape: (out_c, out_h, out_w),
                input_shape: (in_c, in_h, in_w),
                unstable_idx: None,
            },
            upper_b: Array1::from(upper_b.clone()),
        };

        // Dense seed = the same coefficients scattered to a [num_outputs x
        // num_inputs] matrix — the exact input the dense operator would receive.
        let dense_seed: LinearBounds = make_bounds().to_dense().unwrap();

        for (ai, alpha_fill) in [Some(0.0f32), Some(0.25), Some(0.5), Some(1.0), None]
            .into_iter()
            .enumerate()
        {
            // alpha per INPUT neuron, in [0,1] (only crossing neurons use it).
            let alpha: Array1<f32> = (0..num_inputs)
                .map(|_i| match alpha_fill {
                    Some(v) => v,
                    None => rng.f32_in(0.0, 1.0), // random-per-neuron sweep
                })
                .collect();
            let _ = ai;

            // ---- patches operator ----
            let (patches_result, _grad) = ReLULayer
                .propagate_patches_with_alpha(&make_bounds(), &pre, &alpha)
                .unwrap();
            let patches_dense = patches_result.into_dense().unwrap();
            let conc_p = patches_dense.concretize_sound(&pre);

            // ---- dense operator (single-alpha: alpha_upper = None) ----
            let (dense_result, _g, _gu) = ReLULayer
                .propagate_linear_with_alpha(&dense_seed, &pre, &alpha, None)
                .unwrap();
            let conc_d = dense_result.concretize_sound(&pre);

            let pl = conc_p.lower();
            let pu = conc_p.upper();
            let dl = conc_d.lower();
            let du = conc_d.upper();
            // Bit-equal values (including both -inf / both +inf) are exact
            // agreement; only finite-vs-finite pairs are checked against `tol`.
            let close = |a: f32, b: f32, tol: f32| -> bool {
                a == b || (a.is_finite() && b.is_finite() && (a - b).abs() <= tol)
            };
            for j in 0..num_outputs {
                if pl[[j]].is_finite() && dl[[j]].is_finite() {
                    let ld = (pl[[j]] - dl[[j]]).abs();
                    max_lower_diff = max_lower_diff.max(ld);
                    finite_cmp += 1;
                }
                if pu[[j]].is_finite() && du[[j]].is_finite() {
                    let ud = (pu[[j]] - du[[j]]).abs();
                    max_upper_diff = max_upper_diff.max(ud);
                }

                // Parity within f32 tolerance.
                assert!(
                    close(pl[[j]], dl[[j]], 1e-2),
                    "seed {seed} alpha {alpha_fill:?} row {j}: lower parity {} vs {}",
                    pl[[j]],
                    dl[[j]]
                );
                assert!(
                    close(pu[[j]], du[[j]], 1e-2),
                    "seed {seed} alpha {alpha_fill:?} row {j}: upper parity {} vs {}",
                    pu[[j]],
                    du[[j]]
                );
                // Enclosure: patches never tighter than dense beyond tolerance
                // (no tighter-than-true intermediate). inf-safe by construction.
                assert!(
                    pl[[j]] <= dl[[j]] + 1e-3,
                    "seed {seed} alpha {alpha_fill:?} row {j}: patches lower {} tighter than dense {}",
                    pl[[j]],
                    dl[[j]]
                );
                assert!(
                    pu[[j]] >= du[[j]] - 1e-3,
                    "seed {seed} alpha {alpha_fill:?} row {j}: patches upper {} tighter than dense {}",
                    pu[[j]],
                    du[[j]]
                );
            }
        }
    }

    // Sanity: the operators are genuinely close over the finite rows, and the
    // test actually exercised a non-trivial number of finite comparisons.
    assert!(
        finite_cmp >= 20,
        "too few finite comparisons ({finite_cmp}) — test is near-vacuous"
    );
    assert!(
        max_lower_diff < 1e-2 && max_upper_diff < 1e-2,
        "max diffs lower={max_lower_diff} upper={max_upper_diff}"
    );
}
