// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bit-equivalence + soundness proptests for the patches-native
//! ConvTranspose2d CROWN backward.
//!
//! The gate (crown_patches.rs:29 contract): the patches result converted via
//! `to_dense()` must reproduce the dense `ConvTranspose2dLayer::
//! propagate_linear_with_engine` bound within `1e-5` relative. These tests
//! assert that gate for identity AND non-identity incoming bounds, plus:
//!   - end-to-end soundness (sampled true inputs stay inside the
//!     `concretize_sound` box built from the certified `coeff_err`);
//!   - the certified `coeff_err` is at least as conservative as the dense
//!     path's (a sound over-bound — the patches per-row bound is looser than the
//!     dense per-cell composition, so exact err MATCHING is not the contract;
//!     see the module-level note in `bound_transpose_patches.rs`);
//!   - stride-2/3 identity bounds use exact anchored geometry, including
//!     padding, dilation, and output-padding boundary cases;
//!   - certified materialized 6D/7D stride>1 composition, including duplicate
//!     destinations and carried coefficient error;
//!   - stride>1 finite-authority identity/general production routes agree with
//!     their no-deadline twins, while finite stride 1 uses the certified
//!     Anchored route and preserves the historical no-deadline reduction;
//!   - expired authority and unsupported inputs remain atomic;
//!   - sparse/mixed/malformed and over-budget inputs return a typed error before
//!     mutating their input; the historical no-deadline stored-size crossover
//!     remains a typed refusal, while finite authority may cross it only through
//!     the total-live admitted Anchored transaction.
//!
//! Design: no-deadline stride 1 retains the stage-2a reduction to the proven
//! Conv2d patches path (flip+swap kernel, adjusted padding, same bias). Finite
//! stride 1 uses the same cooperative, budgeted Anchored planners as stride>1.

use crate::bounds::patches::{
    CrownBounds, PatchGeometry, PatchesData, PatchesLinearBounds, UnstableIdx,
};
use crate::layers::activations::ReLULayer;
use crate::layers::common::{BoundPropagation, PatchesPropagation};
use crate::layers::convolution::conv2d::ConvTranspose2dLayer;
use crate::layers::normalization::BatchNormLayer;
use crate::LinearBounds;
use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;
use std::time::{Duration, Instant};

/// Tolerance for the DAZ-stable identity Patches-vs-Dense comparison. Identity
/// incoming yields per-cell single-kernel-tap coefficients on both paths, so
/// ordinary finite-normal fixtures agree bit-for-bit; `1e-5` relative leaves
/// ULP headroom. Subnormal kernels have a separate interval oracle below.
const IDENTITY_TOL: f32 = 1e-5;

/// Tolerance for non-identity composition. Both paths f64-recompute the SAME
/// contraction and cast to f32, but via different accumulation orders (im2col
/// GEMM vs transpose scatter), so they agree to a few ULP. Matches the Conv2d
/// chain proptest tolerance (`crown_patches.rs` Test 3).
const COMPOSE_TOL: f32 = 1e-4;

/// Deterministic xorshift64 fill in `[-scale, scale)`.
fn det_fill(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = seed | 1;
    (0..len)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32); // [0, 1)
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Random ConvTranspose2d kernel of ONNX layout `(in_c, out_c, kh, kw)`.
fn make_convt_kernel(in_c: usize, out_c: usize, kh: usize, kw: usize, seed: u64) -> ArrayD<f32> {
    ArrayD::from_shape_vec(
        IxDyn(&[in_c, out_c, kh, kw]),
        det_fill(in_c * out_c * kh * kw, seed, 2.0),
    )
    .expect("kernel shape")
}

fn make_bias(out_c: usize, seed: u64) -> Array1<f32> {
    Array1::from_vec(det_fill(out_c, seed.wrapping_add(0x9E37_79B9), 1.0))
}

/// Generate exactly the stride-1 patches-native kernel/padding domain:
/// `padding <= kernel - 1`, while retaining every kernel size 1..=3.
fn supported_kernel_padding() -> impl Strategy<Value = (usize, usize)> {
    (1usize..=3).prop_flat_map(|kernel| (Just(kernel), 0usize..kernel.min(2)))
}

/// Assert `patches`'s dense A/b reproduce `dense`'s within `tol` relative
/// (scale floored at 1.0), mirroring the Conv2d patches equivalence tests.
fn assert_linear_bounds_close(
    dense: &LinearBounds,
    patches: &LinearBounds,
    tol: f32,
) -> Result<(), TestCaseError> {
    prop_assert_eq!(
        dense.lower_a().shape(),
        patches.lower_a().shape(),
        "lower_a shape"
    );
    prop_assert_eq!(
        dense.upper_a().shape(),
        patches.upper_a().shape(),
        "upper_a shape"
    );
    prop_assert_eq!(dense.num_outputs(), patches.num_outputs(), "num_outputs");
    prop_assert_eq!(dense.num_inputs(), patches.num_inputs(), "num_inputs");

    for ((idx, &d), &p) in dense.lower_a().indexed_iter().zip(patches.lower_a().iter()) {
        let diff = (d - p).abs();
        let scale = d.abs().max(p.abs()).max(1.0);
        prop_assert!(
            diff <= tol * scale,
            "lower_a mismatch at {:?}: dense={}, patches={}, diff={}",
            idx,
            d,
            p,
            diff
        );
    }
    for ((idx, &d), &p) in dense.upper_a().indexed_iter().zip(patches.upper_a().iter()) {
        let diff = (d - p).abs();
        let scale = d.abs().max(p.abs()).max(1.0);
        prop_assert!(
            diff <= tol * scale,
            "upper_a mismatch at {:?}: dense={}, patches={}, diff={}",
            idx,
            d,
            p,
            diff
        );
    }
    for i in 0..dense.num_outputs() {
        let dl = dense.lower_b()[i];
        let pl = patches.lower_b()[i];
        let diff_l = (dl - pl).abs();
        let scale_l = dl.abs().max(pl.abs()).max(1.0);
        prop_assert!(
            diff_l <= tol * scale_l,
            "lower_b mismatch at {}: dense={}, patches={}, diff={}",
            i,
            dl,
            pl,
            diff_l
        );
        let du = dense.upper_b()[i];
        let pu = patches.upper_b()[i];
        let diff_u = (du - pu).abs();
        let scale_u = du.abs().max(pu.abs()).max(1.0);
        prop_assert!(
            diff_u <= tol * scale_u,
            "upper_b mismatch at {}: dense={}, patches={}, diff={}",
            i,
            du,
            pu,
            diff_u
        );
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(400) })]

    /// STAGE 2a gate — identity incoming. `propagate_patches` -> `to_dense` must
    /// reproduce the dense `propagate_linear_with_engine` A/b within 1e-5.
    #[ntest::timeout(30000)]
    #[test]
    fn proptest_convtranspose2d_patches_vs_dense_identity(
        in_c in 1usize..=4,
        out_c in 1usize..=4,
        (kh, pad_h) in supported_kernel_padding(),
        (kw, pad_w) in supported_kernel_padding(),
        // Keep the equivalent Conv2d below its Dense area crossover.
        in_h in 4usize..=8,
        in_w in 4usize..=8,
        use_bias in proptest::bool::ANY,
        seed in any::<u64>(),
    ) {
        // The strategy guarantees valid stride-1, dilation-1, output-padding-0
        // geometry in the patches-native stage-2a domain.
        let out_h = in_h + kh - 1 - 2 * pad_h;
        let out_w = in_w + kw - 1 - 2 * pad_w;

        let out_dim = out_c * out_h * out_w;
        let kernel = make_convt_kernel(in_c, out_c, kh, kw, seed);
        let bias = if use_bias { Some(make_bias(out_c, seed)) } else { None };
        let convt = ConvTranspose2dLayer::with_input_shape(
            kernel, bias, (1, 1), (pad_h, pad_w), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("ConvTranspose2d creation failed: {e}")))?;

        // Ground truth: dense CROWN backward.
        let identity_lb = LinearBounds::identity(out_dim);
        let dense_out = convt.propagate_linear(&identity_lb)
            .map_err(|e| TestCaseError::fail(format!("dense propagate_linear failed: {e}")))?
            .into_owned();

        // Patches path.
        let patches_id = PatchesLinearBounds::identity(
            (out_c, out_h, out_w), (out_c, out_h, out_w),
        );
        let patches_out = convt.propagate_patches(&patches_id)
            .map_err(|e| TestCaseError::fail(format!(
                "supported ConvTranspose2d patches identity path failed: {e}"
            )))?;
        prop_assert!(
            matches!(&patches_out, CrownBounds::Patches(_)),
            "supported below-crossover ConvTranspose2d identity path returned Dense"
        );
        let patches_dense = patches_out.into_dense()
            .map_err(|e| TestCaseError::fail(format!("patches to_dense failed: {e}")))?;

        assert_linear_bounds_close(&dense_out, &patches_dense, IDENTITY_TOL)?;
    }
}

/// Independent exact-binary64 evaluator for the unpadded, dilation-one,
/// stride-two ConvTranspose links used by the official cGAN seam below.
/// Every fixture operand is dyadic, so these sums are exact real evaluations
/// rather than a second f32 implementation of either CROWN path.
fn official_stride2_convt_forward_f64(
    input: &[f64],
    input_shape: (usize, usize, usize),
    kernel: &ArrayD<f32>,
    bias: &Array1<f32>,
) -> (Vec<f64>, (usize, usize, usize)) {
    let (in_c, in_h, in_w) = input_shape;
    assert_eq!(input.len(), in_c * in_h * in_w);
    assert_eq!(kernel.ndim(), 4);
    assert_eq!(kernel.shape()[0], in_c);
    let out_c = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    assert_eq!(bias.len(), out_c);
    let out_h = (in_h - 1) * 2 + kh;
    let out_w = (in_w - 1) * 2 + kw;
    let mut output = vec![0.0f64; out_c * out_h * out_w];

    for oc in 0..out_c {
        let exact_bias = ny_core::f32_to_f64_exact(bias[oc]);
        for oh in 0..out_h {
            for ow in 0..out_w {
                output[(oc * out_h + oh) * out_w + ow] = exact_bias;
            }
        }
    }
    for ic in 0..in_c {
        for ih in 0..in_h {
            for iw in 0..in_w {
                let value = input[(ic * in_h + ih) * in_w + iw];
                for oc in 0..out_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let oh = ih * 2 + ki;
                            let ow = iw * 2 + kj;
                            let weight = ny_core::f32_to_f64_exact(kernel[[ic, oc, ki, kj]]);
                            output[(oc * out_h + oh) * out_w + ow] += value * weight;
                        }
                    }
                }
            }
        }
    }

    (output, (out_c, out_h, out_w))
}

fn assert_anchored_cgan_stage(stage: &str, bounds: &PatchesLinearBounds) {
    assert!(
        matches!(&bounds.lower_a.geometry, PatchGeometry::Anchored(_)),
        "{stage} lower side lost Anchored geometry"
    );
    assert!(
        matches!(&bounds.upper_a.geometry, PatchGeometry::Anchored(_)),
        "{stage} upper side lost Anchored geometry"
    );
    bounds
        .lower_a
        .validate_common_geometry(&bounds.upper_a)
        .unwrap_or_else(|error| panic!("{stage} published unauthenticated geometry: {error}"));
}

/// End-to-end regression for the first fully sparse official cGAN backward
/// seam whose second composition remains below the deliberate Dense crossover:
///
///   identity(30x30) -> ConvTranspose(14->30) -> ReLU -> BatchNorm
///                    -> ConvTranspose(6->14)
///
/// The terminal 14->6->2 seam has a raw composed extent of 3x3 over a 2x2
/// input and therefore correctly takes the established Dense crossover.  The
/// adjacent 30->14->6 seam has the same real operator topology while retaining
/// a 3x3 patch over a 6x6 input, so it is the bounded hermetic fixture for the
/// native consumer chain.
#[test]
fn convtranspose2d_official_cgan_30_14_6_finite_chain_stays_anchored_and_sound() {
    use crate::bounds::patches::{patches_to_dense_call_sites, reset_patches_to_dense_call_count};

    // Forward 6x6 -> 14x14, producing the two BatchNorm/ReLU channels.
    let input_kernel_values: Vec<f32> = (0..32)
        .map(|index| if index % 3 == 0 { -0.0625 } else { 0.0625 })
        .collect();
    let input_kernel = ArrayD::from_shape_vec(IxDyn(&[1, 2, 4, 4]), input_kernel_values).unwrap();
    let input_bias = Array1::from_vec(vec![0.0625, -0.0625]);
    let mut input_convt = ConvTranspose2dLayer::new_full(
        input_kernel.clone(),
        Some(input_bias.clone()),
        (2, 2),
        (0, 0),
        (1, 1),
        (0, 0),
    )
    .unwrap();
    input_convt.set_input_shape(6, 6);
    assert_eq!(input_convt.output_size(6, 6).unwrap(), (14, 14));

    // Forward 14x14 -> 30x30, returning to one image channel.
    let output_kernel_values: Vec<f32> = (0..32)
        .map(|index| if index % 5 < 2 { -0.125 } else { 0.125 })
        .collect();
    let output_kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 4, 4]), output_kernel_values).unwrap();
    let output_bias = Array1::from_vec(vec![0.03125]);
    let mut output_convt = ConvTranspose2dLayer::new_full(
        output_kernel.clone(),
        Some(output_bias.clone()),
        (2, 2),
        (0, 0),
        (1, 1),
        (0, 0),
    )
    .unwrap();
    output_convt.set_input_shape(14, 14);
    assert_eq!(output_convt.output_size(14, 14).unwrap(), (30, 30));

    let batch_norm = BatchNormLayer::from_scale_bias(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, -0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.125, -0.125]).unwrap(),
    )
    .unwrap();

    // Channel 0 alternates exact-active and deliberately loose unstable
    // relaxations; channel 1 is exact-inactive.  The independent interval
    // oracle below authenticates the entire input box before the test relies on
    // the resulting relaxation.
    let mut relu_lower = Vec::with_capacity(2 * 14 * 14);
    let mut relu_upper = Vec::with_capacity(2 * 14 * 14);
    for channel in 0..2 {
        for row in 0..14 {
            for column in 0..14 {
                if channel == 0 && (row + column) % 2 == 0 {
                    relu_lower.push(0.0);
                    relu_upper.push(0.5);
                } else if channel == 0 {
                    relu_lower.push(-0.5);
                    relu_upper.push(0.5);
                } else {
                    relu_lower.push(-0.5);
                    relu_upper.push(0.0);
                }
            }
        }
    }
    let relu_pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 14, 14]), relu_lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 14, 14]), relu_upper).unwrap(),
    )
    .unwrap();

    // Authenticate the relaxation domain for the entire 6x6 input box, not
    // merely for the samples below.  For a linear map over a symmetric box,
    // center +/- sum(|weight| * radius) is its exact coordinate-wise range.
    let zero_input = [0.0f64; 36];
    let (hidden_center, center_shape) =
        official_stride2_convt_forward_f64(&zero_input, (1, 6, 6), &input_kernel, &input_bias);
    let absolute_input_kernel = input_kernel.mapv(|value| value.abs());
    let uniform_radius = [0.25f64; 36];
    let (hidden_radius, radius_shape) = official_stride2_convt_forward_f64(
        &uniform_radius,
        (1, 6, 6),
        &absolute_input_kernel,
        &Array1::zeros(2),
    );
    assert_eq!(center_shape, (2, 14, 14));
    assert_eq!(radius_shape, center_shape);
    for flat in 0..hidden_center.len() {
        let channel = flat / (14 * 14);
        let hidden_lower = hidden_center[flat] - hidden_radius[flat];
        let hidden_upper = hidden_center[flat] + hidden_radius[flat];
        let scale = ny_core::f32_to_f64_exact(batch_norm.scale[[channel]]);
        let bias = ny_core::f32_to_f64_exact(batch_norm.bias[[channel]]);
        let endpoint_a = hidden_lower * scale + bias;
        let endpoint_b = hidden_upper * scale + bias;
        let exact_lower = endpoint_a.min(endpoint_b);
        let exact_upper = endpoint_a.max(endpoint_b);
        let supplied_lower =
            ny_core::f32_to_f64_exact(relu_pre_activation.lower().as_slice().unwrap()[flat]);
        let supplied_upper =
            ny_core::f32_to_f64_exact(relu_pre_activation.upper().as_slice().unwrap()[flat]);
        assert!(
            supplied_lower <= exact_lower && exact_upper <= supplied_upper,
            "ReLU domain {flat} misses exact box range [{exact_lower}, {exact_upper}] with [{supplied_lower}, {supplied_upper}]"
        );
    }

    // Completely independent Dense CROWN route over the same four backward
    // operators. BatchNorm's precomputed affine has zero parameter error here,
    // so the supplied box contributes authenticated shape metadata only.
    let dense_identity = LinearBounds::identity(30 * 30);
    let dense_after_output = output_convt
        .propagate_linear(&dense_identity)
        .unwrap()
        .into_owned();
    let dense_after_relu = ReLULayer
        .propagate_linear_with_bounds(&dense_after_output, &relu_pre_activation)
        .unwrap();
    let dense_after_batch_norm = batch_norm
        .propagate_linear_with_bounds(&dense_after_relu, &relu_pre_activation)
        .unwrap();
    let dense_final = input_convt
        .propagate_linear(&dense_after_batch_norm)
        .unwrap()
        .into_owned();

    reset_patches_to_dense_call_count();
    let deadline = Instant::now() + Duration::from_secs(30);
    let patches_identity = PatchesLinearBounds::identity((1, 30, 30), (1, 30, 30));
    let patches_identity_before = patches_identity.clone();
    let after_output = match output_convt
        .propagate_patches_engine_and_deadline(&patches_identity, None, Some(deadline))
        .unwrap()
    {
        CrownBounds::Patches(bounds) => bounds,
        CrownBounds::Dense(_) => {
            panic!("finite official identity ConvTranspose unexpectedly densified")
        }
    };
    assert_patches_input_unchanged(&patches_identity, &patches_identity_before);
    assert_anchored_cgan_stage("identity ConvTranspose", &after_output);

    let after_output_before = (*after_output).clone();
    let after_relu = match ReLULayer
        .propagate_patches_with_bounds_and_deadline(&after_output, &relu_pre_activation, deadline)
        .unwrap()
    {
        CrownBounds::Patches(bounds) => bounds,
        CrownBounds::Dense(_) => panic!("Anchored ReLU unexpectedly densified"),
    };
    assert_patches_input_unchanged(&after_output, &after_output_before);
    assert_anchored_cgan_stage("ReLU", &after_relu);

    let after_relu_before = (*after_relu).clone();
    let after_batch_norm = match batch_norm
        .propagate_patches_with_deadline(&after_relu, deadline)
        .unwrap()
    {
        CrownBounds::Patches(bounds) => bounds,
        CrownBounds::Dense(_) => panic!("Anchored BatchNorm unexpectedly densified"),
    };
    assert_patches_input_unchanged(&after_relu, &after_relu_before);
    assert_anchored_cgan_stage("BatchNorm", &after_batch_norm);

    let after_batch_norm_before = (*after_batch_norm).clone();
    let final_patches = match input_convt
        .propagate_patches_engine_and_deadline(&after_batch_norm, None, Some(deadline))
        .unwrap()
    {
        CrownBounds::Patches(bounds) => bounds,
        CrownBounds::Dense(_) => {
            panic!("finite official general ConvTranspose unexpectedly densified")
        }
    };
    assert_patches_input_unchanged(&after_batch_norm, &after_batch_norm_before);
    assert_anchored_cgan_stage("general ConvTranspose", &final_patches);
    assert_eq!(final_patches.lower_a.input_shape, (1, 6, 6));
    assert_eq!(final_patches.lower_a.output_shape, (1, 30, 30));
    assert_eq!(
        final_patches.lower_a.patches.as_ref().unwrap().shape(),
        &[1, 30, 30, 1, 3, 3]
    );

    // No operator in the native chain may hide a Patches->Dense conversion.
    // The sole explicit materialization below is delayed until after observing
    // this recorder window and exists only to compare against the Dense oracle.
    let implicit_dense_sites = patches_to_dense_call_sites();
    assert!(
        implicit_dense_sites.is_empty(),
        "official Anchored chain materialized Dense internally: {implicit_dense_sites:?}"
    );
    let patches_final_dense = final_patches.to_dense().unwrap();
    assert_linear_bounds_close(&dense_final, &patches_final_dense, COMPOSE_TOL).unwrap();

    let input_box = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 6, 6]), -0.25),
        ArrayD::from_elem(IxDyn(&[1, 6, 6]), 0.25),
    )
    .unwrap();
    let patches_box = patches_final_dense.concretize_sound(&input_box);
    let dense_box = dense_final.concretize_sound(&input_box);
    assert_eq!(patches_box.shape(), dense_box.shape());
    for (index, (((&patch_lower, &patch_upper), &dense_lower), &dense_upper)) in patches_box
        .lower()
        .iter()
        .zip(patches_box.upper().iter())
        .zip(dense_box.lower().iter())
        .zip(dense_box.upper().iter())
        .enumerate()
    {
        assert!(
            patch_lower.is_finite()
                && patch_upper.is_finite()
                && dense_lower.is_finite()
                && dense_upper.is_finite(),
            "non-finite certificate at output {index}: patches=[{patch_lower}, {patch_upper}] Dense=[{dense_lower}, {dense_upper}]"
        );
        let scale = patch_lower
            .abs()
            .max(patch_upper.abs())
            .max(dense_lower.abs())
            .max(dense_upper.abs())
            .max(1.0);
        let tolerance = COMPOSE_TOL * scale;
        assert!(
            patch_lower <= dense_lower + tolerance,
            "patch lower misses Dense certificate at output {index}: {patch_lower} > {dense_lower}"
        );
        assert!(
            patch_upper + tolerance >= dense_upper,
            "patch upper misses Dense certificate at output {index}: {patch_upper} < {dense_upper}"
        );
        assert!(
            (patch_lower - dense_lower).abs() <= tolerance
                && (patch_upper - dense_upper).abs() <= tolerance,
            "Patches/Dense concrete mismatch at output {index}: patches=[{patch_lower}, {patch_upper}] Dense=[{dense_lower}, {dense_upper}]"
        );
    }

    // Three exact dyadic points cover zero, both box corners in alternating
    // coordinates, and a four-level interior pattern.  The f64 evaluator uses
    // the mathematical forward operators, independently authenticates every
    // ReLU pre-activation, and must be enclosed by both CROWN certificates.
    let samples: Vec<Vec<f32>> = vec![
        vec![0.0; 36],
        (0..36)
            .map(|index| if index % 2 == 0 { -0.25 } else { 0.25 })
            .collect(),
        (0..36)
            .map(|index| [-0.25, -0.125, 0.125, 0.25][index % 4])
            .collect(),
    ];
    for (sample_index, sample) in samples.into_iter().enumerate() {
        let exact_input: Vec<f64> = sample
            .iter()
            .map(|&value| ny_core::f32_to_f64_exact(value))
            .collect();
        let (hidden, hidden_shape) =
            official_stride2_convt_forward_f64(&exact_input, (1, 6, 6), &input_kernel, &input_bias);
        assert_eq!(hidden_shape, (2, 14, 14));
        let mut pre_relu = Vec::with_capacity(hidden.len());
        for (flat, &value) in hidden.iter().enumerate() {
            let channel = flat / (14 * 14);
            let affine = value * ny_core::f32_to_f64_exact(batch_norm.scale[[channel]])
                + ny_core::f32_to_f64_exact(batch_norm.bias[[channel]]);
            let certified_lower =
                ny_core::f32_to_f64_exact(relu_pre_activation.lower().as_slice().unwrap()[flat]);
            let certified_upper =
                ny_core::f32_to_f64_exact(relu_pre_activation.upper().as_slice().unwrap()[flat]);
            assert!(
                certified_lower <= affine && affine <= certified_upper,
                "sample {sample_index} pre-activation {flat}={affine} escapes [{certified_lower}, {certified_upper}]"
            );
            pre_relu.push(affine.max(0.0));
        }
        let (truth, truth_shape) = official_stride2_convt_forward_f64(
            &pre_relu,
            (2, 14, 14),
            &output_kernel,
            &output_bias,
        );
        assert_eq!(truth_shape, (1, 30, 30));

        let point = ArrayD::from_shape_vec(IxDyn(&[1, 6, 6]), sample).unwrap();
        let point_box = BoundedTensor::new(point.clone(), point).unwrap();
        let patches_point = patches_final_dense.concretize_sound(&point_box);
        let dense_point = dense_final.concretize_sound(&point_box);
        for (output, &exact) in truth.iter().enumerate() {
            for (name, concrete) in [("Patches", &patches_point), ("Dense", &dense_point)] {
                let lower = ny_core::f32_to_f64_exact(concrete.lower().as_slice().unwrap()[output]);
                let upper = ny_core::f32_to_f64_exact(concrete.upper().as_slice().unwrap()[output]);
                assert!(lower.is_finite() && upper.is_finite());
                assert!(
                    lower <= exact && exact <= upper,
                    "sample {sample_index} {name} output {output} excludes exact {exact} from [{lower}, {upper}]"
                );
            }
        }
    }
}

#[test]
fn convtranspose2d_official_grid_2_6_14_30_32_is_exact() {
    for &(input, output, stride, padding, kernel) in &[
        (2, 6, 2, 0, 4),
        (6, 14, 2, 0, 4),
        (14, 30, 2, 0, 4),
        (30, 32, 1, 0, 3),
    ] {
        assert_official_grid_link(input, output, stride, padding, kernel);
    }
}

#[test]
fn convtranspose2d_official_grid_2_6_14_30_62_64_is_exact() {
    for &(input, output, stride, padding, kernel) in &[
        (2, 6, 2, 0, 4),
        (6, 14, 2, 0, 4),
        (14, 30, 2, 0, 4),
        (30, 62, 2, 0, 4),
        (62, 64, 1, 0, 3),
    ] {
        assert_official_grid_link(input, output, stride, padding, kernel);
    }
}

#[test]
fn convtranspose2d_padding_one_grid_4_8_16_32_is_exact() {
    for &(input, output) in &[(4, 8), (8, 16), (16, 32)] {
        assert_official_grid_link(input, output, 2, 1, 4);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(400) })]

    /// STAGE 2a gate — NON-identity incoming (real composition, EXACT incoming).
    /// Build a non-identity incoming `PatchesLinearBounds` with an exact
    /// (`coeff_err = None`) small receptive field, take its `to_dense()` as the
    /// dense incoming, and assert the two ConvTranspose backward paths commute
    /// with `to_dense` on the coefficients AND the bias within `1e-4`.
    ///
    /// The incoming `coeff_err` is deliberately `None`: with a nonzero incoming
    /// err the two paths distribute that error differently (patches folds part
    /// into an outward bias widen via `compute_patches_bias`'s HOLE2 discharge;
    /// dense keeps it in the `coeff_err` matrix) — both sound, but not
    /// bit-identical. The incoming-err composition is inherited verbatim from the
    /// (proptested) Conv2d patches path and is exercised end-to-end against
    /// sampled truth by `proptest_convtranspose2d_relu_chain_soundness`.
    #[ntest::timeout(30000)]
    #[test]
    fn proptest_convtranspose2d_patches_vs_dense_nonidentity(
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        (kh, pad_h) in supported_kernel_padding(),
        (kw, pad_w) in supported_kernel_padding(),
        // The composed kernel is at most 4x4; a 5x5 input keeps it sparse.
        in_h in 5usize..=8,
        in_w in 5usize..=8,
        prev_k in 1usize..=2,
        use_bias in proptest::bool::ANY,
        seed in any::<u64>(),
    ) {
        let out_h = in_h + kh - 1 - 2 * pad_h;
        let out_w = in_w + kw - 1 - 2 * pad_w;
        // The incoming (prev_k x prev_k, stride 1, pad 0) receptive field over the
        // ConvTranspose OUTPUT grid yields spec spatial dims = unfold output size.
        let spec_h = out_h - prev_k + 1;
        let spec_w = out_w - prev_k + 1;

        let kernel = make_convt_kernel(in_c, out_c, kh, kw, seed);
        let bias = if use_bias { Some(make_bias(out_c, seed)) } else { None };
        let convt = ConvTranspose2dLayer::with_input_shape(
            kernel, bias, (1, 1), (pad_h, pad_w), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("ConvTranspose2d creation failed: {e}")))?;

        // Non-identity incoming: coefficients over the ConvTranspose OUTPUT space
        // ((out_c, out_h, out_w) = the patch input_shape), one spec row per
        // (spec_c=out_c, spec_h, spec_w) neuron, a small (prev_k x prev_k)
        // receptive field, stride 1 / padding 0. `in_c-of-patch` (axis 3) ==
        // out_c. Exact (coeff_err None).
        let spec_dim = out_c * spec_h * spec_w;
        let patch_shape = [out_c, spec_h, spec_w, out_c, prev_k, prev_k];
        let patch_len: usize = patch_shape.iter().product();
        let make_side = |salt: u64| -> PatchesData {
            PatchesData {
                coeff_err: None,
                patches: Some(
                    ArrayD::from_shape_vec(IxDyn(&patch_shape), det_fill(patch_len, salt, 1.5)).unwrap(),
                ),
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: false,
                output_shape: (out_c, spec_h, spec_w),
                input_shape: (out_c, out_h, out_w),
                unstable_idx: None,
            }
        };
        let incoming = PatchesLinearBounds {
            row_count: spec_dim,
            lower_a: make_side(seed ^ 0xA1),
            lower_b: Array1::from_vec(det_fill(spec_dim, seed ^ 0xB2, 1.0)),
            upper_a: make_side(seed ^ 0xC3),
            upper_b: Array1::from_vec(det_fill(spec_dim, seed ^ 0xD4, 1.0)),
        };

        // Dense incoming = to_dense of the SAME mathematical bound.
        let dense_in = incoming.to_dense()
            .map_err(|e| TestCaseError::fail(format!("incoming to_dense failed: {e}")))?;

        let dense_out = convt.propagate_linear_with_engine(&dense_in, None)
            .map_err(|e| TestCaseError::fail(format!("dense propagate failed: {e}")))?
            .into_owned();

        let patches_out = convt.propagate_patches(&incoming)
            .map_err(|e| TestCaseError::fail(format!(
                "supported ConvTranspose2d nonidentity patches path failed: {e}"
            )))?;
        prop_assert!(
            matches!(&patches_out, CrownBounds::Patches(_)),
            "supported below-crossover ConvTranspose2d composition returned Dense"
        );
        let patches_dense = patches_out.into_dense()
            .map_err(|e| TestCaseError::fail(format!("patches to_dense failed: {e}")))?;

        assert_linear_bounds_close(&dense_out, &patches_dense, COMPOSE_TOL)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// End-to-end soundness: sampled true ConvTranspose outputs stay inside BOTH
    /// the dense and the patches `concretize_sound` boxes (built from the
    /// certified `coeff_err`). Mirrors the Conv2d patches soundness proptest.
    #[ntest::timeout(20000)]
    #[test]
    fn proptest_convtranspose2d_patches_soundness(
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        (kh, pad_h) in supported_kernel_padding(),
        (kw, pad_w) in supported_kernel_padding(),
        in_h in 4usize..=6,
        in_w in 4usize..=6,
        seed in any::<u64>(),
    ) {
        let out_h = in_h + kh - 1 - 2 * pad_h;
        let out_w = in_w + kw - 1 - 2 * pad_w;

        let out_dim = out_c * out_h * out_w;
        let in_dim = in_c * in_h * in_w;
        let kernel = make_convt_kernel(in_c, out_c, kh, kw, seed);
        let convt = ConvTranspose2dLayer::with_input_shape(
            kernel, Some(make_bias(out_c, seed)), (1, 1), (pad_h, pad_w), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("ConvTranspose2d creation failed: {e}")))?;

        // Input box over the ConvTranspose INPUT space.
        let in_shape = [in_c, in_h, in_w];
        let lower_vals = det_fill(in_dim, seed ^ 0x1234, 1.0);
        let upper_vals: Vec<f32> = lower_vals.iter().map(|&l| l + 0.2).collect();
        let input_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&in_shape), lower_vals.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&in_shape), upper_vals.clone()).unwrap(),
        ).map_err(|e| TestCaseError::fail(format!("BoundedTensor failed: {e}")))?;
        let flat_input = input_bt.flatten();

        let identity_lb = LinearBounds::identity(out_dim);
        let dense_out = convt.propagate_linear(&identity_lb)
            .map_err(|e| TestCaseError::fail(format!("dense failed: {e}")))?
            .into_owned();
        let dense_bounds = dense_out.concretize_sound(&flat_input);

        let patches_out = convt.propagate_patches(
            &PatchesLinearBounds::identity((out_c, out_h, out_w), (out_c, out_h, out_w)),
        ).map_err(|e| TestCaseError::fail(format!(
            "supported ConvTranspose2d patches soundness path failed: {e}"
        )))?;
        prop_assert!(
            matches!(&patches_out, CrownBounds::Patches(_)),
            "supported below-crossover ConvTranspose2d soundness path returned Dense"
        );
        let patches_lb = patches_out.into_dense()
            .map_err(|e| TestCaseError::fail(format!("patches to_dense failed: {e}")))?;
        let patches_bounds = patches_lb.concretize_sound(&flat_input);

        let tol = 1e-5_f32;
        for s in 0..16 {
            let sample: Vec<f32> = lower_vals.iter().zip(upper_vals.iter()).enumerate()
                .map(|(i, (&l, &u))| {
                    let t = ((s as f32 * 0.618_034) + (i as f32 * 0.414_213)) % 1.0;
                    l + (u - l) * t
                })
                .collect();
            let sample_nd = ArrayD::from_shape_vec(IxDyn(&in_shape), sample).unwrap();
            let sample_pt = BoundedTensor::new(sample_nd.clone(), sample_nd).unwrap();
            let true_out = convt.propagate_ibp(&sample_pt)
                .map_err(|e| TestCaseError::fail(format!("eval failed: {e}")))?;
            let true_flat: Vec<f32> = true_out.lower().iter().copied().collect();
            for (j, &tv) in true_flat.iter().enumerate().take(out_dim) {
                prop_assert!(
                    dense_bounds.lower()[[j]] <= tv + tol && dense_bounds.upper()[[j]] >= tv - tol,
                    "dense bound violation at {j}: [{}, {}] excludes {tv}",
                    dense_bounds.lower()[[j]], dense_bounds.upper()[[j]]
                );
                prop_assert!(
                    patches_bounds.lower()[[j]] <= tv + tol && patches_bounds.upper()[[j]] >= tv - tol,
                    "patches bound violation at {j}: [{}, {}] excludes {tv}",
                    patches_bounds.lower()[[j]], patches_bounds.upper()[[j]]
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// End-to-end soundness through a REAL non-identity incoming carrying the
    /// full certified `coeff_err` machinery. Network (forward):
    ///   Input -> ConvTranspose2d -> ReLU -> output.
    /// Backward: identity -> ReLU backward (non-identity patches, err-carrying)
    /// -> ConvTranspose backward. Both the dense and the patches chains must
    /// enclose every sampled true output `ReLU(ConvTranspose(x))`. This exercises
    /// the incoming-`coeff_err` composition (which the bit-equivalence tests hold
    /// exact-only) against sampled truth via `concretize_sound`.
    #[ntest::timeout(20000)]
    #[test]
    fn proptest_convtranspose2d_relu_chain_soundness(
        in_c in 1usize..=2,
        out_c in 1usize..=3,
        (kh, pad_h) in supported_kernel_padding(),
        (kw, pad_w) in supported_kernel_padding(),
        in_h in 4usize..=5,
        in_w in 4usize..=5,
        seed in any::<u64>(),
    ) {
        let out_h = in_h + kh - 1 - 2 * pad_h;
        let out_w = in_w + kw - 1 - 2 * pad_w;

        let out_dim = out_c * out_h * out_w;
        let in_dim = in_c * in_h * in_w;
        let kernel = make_convt_kernel(in_c, out_c, kh, kw, seed);
        let convt = ConvTranspose2dLayer::with_input_shape(
            kernel, Some(make_bias(out_c, seed)), (1, 1), (pad_h, pad_w), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("ConvTranspose2d creation failed: {e}")))?;
        let relu = ReLULayer::new();

        // Input box over the ConvTranspose INPUT space; ReLU pre-activation = the
        // ConvTranspose output box over that input box (sound enclosure of the
        // true pre-activation for every sampled input).
        let in_shape = [in_c, in_h, in_w];
        let lower_vals = det_fill(in_dim, seed ^ 0x1234, 1.0);
        let upper_vals: Vec<f32> = lower_vals.iter().map(|&l| l + 0.4).collect();
        let input_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&in_shape), lower_vals.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&in_shape), upper_vals.clone()).unwrap(),
        ).map_err(|e| TestCaseError::fail(format!("BoundedTensor failed: {e}")))?;
        let flat_input = input_bt.flatten();
        let pre_relu = convt.propagate_ibp(&input_bt)
            .map_err(|e| TestCaseError::fail(format!("ConvT IBP failed: {e}")))?;

        // ---- Dense chain ----
        let dense_after_relu = relu
            .propagate_linear_with_bounds(&LinearBounds::identity(out_dim), &pre_relu)
            .map_err(|e| TestCaseError::fail(format!("dense ReLU backward failed: {e}")))?;
        let dense_out = convt.propagate_linear(&dense_after_relu)
            .map_err(|e| TestCaseError::fail(format!("dense ConvT backward failed: {e}")))?
            .into_owned();
        let dense_bounds = dense_out.concretize_sound(&flat_input);

        // ---- Patches chain ----
        let patches_after_relu = relu
            .propagate_patches_with_bounds(
                &PatchesLinearBounds::identity((out_c, out_h, out_w), (out_c, out_h, out_w)),
                &pre_relu,
            )
            .map_err(|e| TestCaseError::fail(format!("patches ReLU backward failed: {e}")))?;
        let patches_bounds = match patches_after_relu {
            CrownBounds::Patches(ref pb) => {
                let result = convt.propagate_patches(pb)
                    .map_err(|e| TestCaseError::fail(format!(
                        "supported ConvTranspose2d ReLU-chain patches path failed: {e}"
                    )))?;
                prop_assert!(
                    matches!(&result, CrownBounds::Patches(_)),
                    "supported below-crossover ConvTranspose2d ReLU chain returned Dense"
                );
                result.into_dense()
                    .map_err(|e| TestCaseError::fail(format!("patches to_dense failed: {e}")))?
                    .concretize_sound(&flat_input)
            }
            CrownBounds::Dense(_) => return Err(TestCaseError::fail(
                "ReLU backward unexpectedly left patches mode before ConvTranspose2d",
            )),
        };

        let tol = 1e-4_f32;
        for s in 0..16 {
            let sample: Vec<f32> = lower_vals.iter().zip(upper_vals.iter()).enumerate()
                .map(|(i, (&l, &u))| {
                    let t = ((s as f32 * 0.618_034) + (i as f32 * 0.414_213)) % 1.0;
                    l + (u - l) * t
                })
                .collect();
            let sample_nd = ArrayD::from_shape_vec(IxDyn(&in_shape), sample).unwrap();
            let sample_pt = BoundedTensor::new(sample_nd.clone(), sample_nd).unwrap();
            let hidden = convt.propagate_ibp(&sample_pt)
                .map_err(|e| TestCaseError::fail(format!("ConvT eval failed: {e}")))?;
            let out = relu.propagate_ibp(&hidden)
                .map_err(|e| TestCaseError::fail(format!("ReLU eval failed: {e}")))?;
            let true_flat: Vec<f32> = out.lower().iter().copied().collect();
            for (j, &tv) in true_flat.iter().enumerate().take(out_dim) {
                prop_assert!(
                    dense_bounds.lower()[[j]] <= tv + tol && dense_bounds.upper()[[j]] >= tv - tol,
                    "dense chain violation at {j}: [{}, {}] excludes {tv}",
                    dense_bounds.lower()[[j]], dense_bounds.upper()[[j]]
                );
                prop_assert!(
                    patches_bounds.lower()[[j]] <= tv + tol && patches_bounds.upper()[[j]] >= tv - tol,
                    "patches chain violation at {j}: [{}, {}] excludes {tv}",
                    patches_bounds.lower()[[j]], patches_bounds.upper()[[j]]
                );
            }
        }
    }
}

// =====================================================================
// Direct anchored identity and materialized 6D/7D composition routes plus
// typed fallbacks. Sparse, mixed, and malformed forms still refuse before
// arithmetic. The no-deadline stored-size crossover likewise remains a refusal
// so its dispatcher can take SOUND dense CROWN; finite authority instead keeps
// the cooperative Anchored transaction when its total-live receipt is admitted.
// =====================================================================

/// Build a valid identity incoming for a ConvTranspose with the given params.
fn convt_identity_for(
    kernel: ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_padding: (usize, usize),
    in_h: usize,
    in_w: usize,
) -> (ConvTranspose2dLayer, PatchesLinearBounds) {
    let mut convt =
        ConvTranspose2dLayer::new_full(kernel, None, stride, padding, dilation, output_padding)
            .expect("layer");
    convt.set_input_shape(in_h, in_w);
    let out_c = convt.out_channels();
    let (out_h, out_w) = convt.output_size(in_h, in_w).expect("output size");
    let id = PatchesLinearBounds::identity((out_c, out_h, out_w), (out_c, out_h, out_w));
    (convt, id)
}

#[derive(Clone, Copy, Debug)]
struct DirectIdentityGeometry {
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_padding: (usize, usize),
    input: (usize, usize),
}

fn ceil_div_i128(value: i128, divisor: i128) -> i128 {
    debug_assert!(divisor > 0);
    -((-value).div_euclid(divisor))
}

fn canonical_f32_bits(value: f32) -> u32 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

/// Independent definition of one ConvTranspose coefficient.  This deliberately
/// does not use the patches planner or forward implementation: a coefficient
/// from output `(oc, qh, qw)` to input `(ic, ih, iw)` exists exactly when
/// `q + padding - input * stride` is a non-negative dilated kernel coordinate.
fn direct_identity_coefficient(
    kernel: &ArrayD<f32>,
    geometry: DirectIdentityGeometry,
    output: (usize, usize, usize),
    input: (usize, usize, usize),
) -> f32 {
    let (oc, qh, qw) = output;
    let (ic, ih, iw) = input;
    let (sh, sw) = geometry.stride;
    let (ph, pw) = geometry.padding;
    let (dh, dw) = geometry.dilation;
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    let kernel_h = qh as i128 + ph as i128 - (ih as i128 * sh as i128);
    let kernel_w = qw as i128 + pw as i128 - (iw as i128 * sw as i128);
    if kernel_h < 0
        || kernel_w < 0
        || kernel_h.rem_euclid(dh as i128) != 0
        || kernel_w.rem_euclid(dw as i128) != 0
    {
        return 0.0;
    }
    let ki = (kernel_h / dh as i128) as usize;
    let kj = (kernel_w / dw as i128) as usize;
    if ki >= kh || kj >= kw {
        0.0
    } else {
        // ONNX ConvTranspose layout: (input channel, output channel, kh, kw).
        kernel[[ic, oc, ki, kj]]
    }
}

fn assert_exact_coefficient(actual: f32, expected: f32, context: &str) {
    assert!(
        actual.is_finite(),
        "{context}: non-finite coefficient {actual}"
    );
    assert_eq!(
        canonical_f32_bits(actual),
        canonical_f32_bits(expected),
        "{context}: got {actual}, expected {expected}"
    );
}

/// Validate the native carrier against an independent inverse-map oracle.
///
/// The raw-patch pass proves every admitted tap denotes the correct input
/// coefficient and that the tight canonical anchor/extent is used.  Counting
/// all valid forward contributions proves completeness without allocating a
/// quadratic matrix, which keeps the 64x64 official-grid tests memory-light.
/// When `dense_reference` is present, a second pass checks every materialized
/// lower/upper coefficient and bias bit-for-bit (modulo signed zero).
fn assert_native_identity_matches_oracle(
    result: CrownBounds,
    dense_reference: Option<&LinearBounds>,
    kernel: &ArrayD<f32>,
    bias: Option<&Array1<f32>>,
    geometry: DirectIdentityGeometry,
    incoming_lower_b: &[f32],
    incoming_upper_b: &[f32],
) -> Box<PatchesLinearBounds> {
    let patches = match result {
        CrownBounds::Patches(patches) => patches,
        CrownBounds::Dense(_) => panic!(
            "identity ConvTranspose {:?} must remain in native Patches",
            geometry
        ),
    };
    let in_c = kernel.shape()[0];
    let out_c = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    let (in_h, in_w) = geometry.input;
    let effective_h = (kh - 1) * geometry.dilation.0;
    let effective_w = (kw - 1) * geometry.dilation.1;
    let out_h = (in_h - 1) * geometry.stride.0 + effective_h + 1 - 2 * geometry.padding.0
        + geometry.output_padding.0;
    let out_w = (in_w - 1) * geometry.stride.1 + effective_w + 1 - 2 * geometry.padding.1
        + geometry.output_padding.1;
    let out_dim = out_c * out_h * out_w;

    assert_eq!(patches.row_count, out_dim);
    assert_eq!(patches.lower_a.output_shape, (out_c, out_h, out_w));
    assert_eq!(patches.upper_a.output_shape, (out_c, out_h, out_w));
    assert_eq!(patches.lower_a.input_shape, (in_c, in_h, in_w));
    assert_eq!(patches.upper_a.input_shape, (in_c, in_h, in_w));
    assert!(!patches.lower_a.identity && !patches.upper_a.identity);
    assert!(patches.lower_a.unstable_idx.is_none());
    assert!(patches.upper_a.unstable_idx.is_none());
    assert!(patches.lower_a.coeff_err.is_none());
    assert!(patches.upper_a.coeff_err.is_none());
    patches
        .lower_a
        .validate_common_geometry(&patches.upper_a)
        .expect("lower/upper identity result must share authenticated geometry");

    if geometry.stride != (1, 1) || geometry.dilation != (1, 1) {
        assert!(
            matches!(&patches.lower_a.geometry, PatchGeometry::Anchored(_)),
            "stride/dilation direct route must publish Anchored geometry"
        );
    }

    let patch_h = effective_h / geometry.stride.0 + 1;
    let patch_w = effective_w / geometry.stride.1 + 1;
    let lower = patches
        .lower_a
        .patches
        .as_ref()
        .expect("native identity result has coefficients");
    let upper = patches
        .upper_a
        .patches
        .as_ref()
        .expect("native identity result has upper coefficients");
    assert_eq!(
        lower.shape(),
        &[out_c, out_h, out_w, in_c, patch_h, patch_w]
    );
    assert_eq!(upper.shape(), lower.shape());

    for qh in 0..out_h {
        let expected = ceil_div_i128(
            qh as i128 + geometry.padding.0 as i128 - effective_h as i128,
            geometry.stride.0 as i128,
        );
        let actual = patches.lower_a.geometry.origin((qh, 0)).unwrap().0;
        assert_eq!(actual, expected, "row anchor at output row {qh}");
    }
    for qw in 0..out_w {
        let expected = ceil_div_i128(
            qw as i128 + geometry.padding.1 as i128 - effective_w as i128,
            geometry.stride.1 as i128,
        );
        let actual = patches.lower_a.geometry.origin((0, qw)).unwrap().1;
        assert_eq!(actual, expected, "column anchor at output column {qw}");
    }

    let mut represented_nonzero = 0usize;
    for oc in 0..out_c {
        for qh in 0..out_h {
            for qw in 0..out_w {
                for ic in 0..in_c {
                    for ti in 0..patch_h {
                        for tj in 0..patch_w {
                            let index = [oc, qh, qw, ic, ti, tj];
                            let lower_value = lower[index];
                            let upper_value = upper[index];
                            assert_exact_coefficient(
                                upper_value,
                                lower_value,
                                "lower/upper raw patch parity",
                            );
                            let Some(flat) = patches
                                .lower_a
                                .geometry
                                .input_flat_index((qh, qw), ic, (ti, tj), (in_c, in_h, in_w))
                                .unwrap()
                            else {
                                // Out-of-image taps are semantically zero and are
                                // discarded by the authenticated scatter map.
                                continue;
                            };
                            let mapped_ic = flat / (in_h * in_w);
                            let rem = flat % (in_h * in_w);
                            let ih = rem / in_w;
                            let iw = rem % in_w;
                            assert_eq!(mapped_ic, ic);
                            let expected = direct_identity_coefficient(
                                kernel,
                                geometry,
                                (oc, qh, qw),
                                (ic, ih, iw),
                            );
                            assert_exact_coefficient(
                                lower_value,
                                expected,
                                "raw patch versus inverse-map oracle",
                            );
                            if expected != 0.0 {
                                represented_nonzero += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut expected_nonzero = 0usize;
    for ic in 0..in_c {
        for ih in 0..in_h {
            for iw in 0..in_w {
                for oc in 0..out_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let qh = ih * geometry.stride.0 + ki * geometry.dilation.0;
                            let qw = iw * geometry.stride.1 + kj * geometry.dilation.1;
                            if qh >= geometry.padding.0
                                && qw >= geometry.padding.1
                                && qh - geometry.padding.0 < out_h
                                && qw - geometry.padding.1 < out_w
                                && kernel[[ic, oc, ki, kj]] != 0.0
                            {
                                expected_nonzero += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        represented_nonzero, expected_nonzero,
        "native carrier omitted or duplicated an in-range nonzero coefficient"
    );

    assert_eq!(incoming_lower_b.len(), out_dim);
    assert_eq!(incoming_upper_b.len(), out_dim);
    assert_eq!(patches.lower_b.len(), out_dim);
    assert_eq!(patches.upper_b.len(), out_dim);
    for row in 0..out_dim {
        let oc = row / (out_h * out_w);
        let layer_bias = bias.map_or(0.0, |values| values[oc]);
        let exact_lower = incoming_lower_b[row] as f64 + layer_bias as f64;
        let exact_upper = incoming_upper_b[row] as f64 + layer_bias as f64;
        assert!(
            patches.lower_b[row] as f64 <= exact_lower,
            "lower bias row {row} is not outward: {} > {exact_lower}",
            patches.lower_b[row]
        );
        assert!(
            patches.upper_b[row] as f64 >= exact_upper,
            "upper bias row {row} is not outward: {} < {exact_upper}",
            patches.upper_b[row]
        );
    }

    if let Some(dense) = dense_reference {
        let materialized = patches.to_dense().expect("anchored to_dense");
        assert_eq!(materialized.lower_a().shape(), dense.lower_a().shape());
        assert_eq!(materialized.upper_a().shape(), dense.upper_a().shape());
        for oc in 0..out_c {
            for qh in 0..out_h {
                for qw in 0..out_w {
                    let row = (oc * out_h + qh) * out_w + qw;
                    for ic in 0..in_c {
                        for ih in 0..in_h {
                            for iw in 0..in_w {
                                let column = (ic * in_h + ih) * in_w + iw;
                                let expected = direct_identity_coefficient(
                                    kernel,
                                    geometry,
                                    (oc, qh, qw),
                                    (ic, ih, iw),
                                );
                                for (name, actual) in [
                                    ("patch lower", materialized.lower_a()[[row, column]]),
                                    ("patch upper", materialized.upper_a()[[row, column]]),
                                    ("dense lower", dense.lower_a()[[row, column]]),
                                    ("dense upper", dense.upper_a()[[row, column]]),
                                ] {
                                    assert_exact_coefficient(actual, expected, name);
                                }
                            }
                        }
                    }
                }
            }
        }
        for row in 0..out_dim {
            assert_eq!(
                materialized.lower_b()[row].to_bits(),
                dense.lower_b()[row].to_bits(),
                "lower bias dense parity at row {row}"
            );
            assert_eq!(
                materialized.upper_b()[row].to_bits(),
                dense.upper_b()[row].to_bits(),
                "upper bias dense parity at row {row}"
            );
        }
    }
    patches
}

fn assert_patches_input_unchanged(actual: &PatchesLinearBounds, before: &PatchesLinearBounds) {
    assert_eq!(actual.row_count, before.row_count);
    assert_eq!(actual.lower_b, before.lower_b);
    assert_eq!(actual.upper_b, before.upper_b);
    for (side, old) in [
        (&actual.lower_a, &before.lower_a),
        (&actual.upper_a, &before.upper_a),
    ] {
        assert_eq!(side.patches, old.patches);
        assert_eq!(side.geometry, old.geometry);
        assert_eq!(side.identity, old.identity);
        assert_eq!(side.output_shape, old.output_shape);
        assert_eq!(side.input_shape, old.input_shape);
        assert_eq!(side.unstable_idx, old.unstable_idx);
        assert_eq!(side.coeff_err, old.coeff_err);
    }
}

fn assert_patches_bounds_bitwise_equal(
    actual: &PatchesLinearBounds,
    expected: &PatchesLinearBounds,
) {
    assert_eq!(actual.row_count, expected.row_count);
    assert_eq!(actual.lower_b, expected.lower_b);
    assert_eq!(actual.upper_b, expected.upper_b);
    for (side, reference) in [
        (&actual.lower_a, &expected.lower_a),
        (&actual.upper_a, &expected.upper_a),
    ] {
        assert_eq!(side.patches, reference.patches);
        assert_eq!(side.geometry, reference.geometry);
        assert_eq!(side.identity, reference.identity);
        assert_eq!(side.output_shape, reference.output_shape);
        assert_eq!(side.input_shape, reference.input_shape);
        assert_eq!(side.unstable_idx, reference.unstable_idx);
        assert_eq!(side.coeff_err, reference.coeff_err);
    }
}

fn expect_native_patches(result: CrownBounds, context: &str) -> Box<PatchesLinearBounds> {
    match result {
        CrownBounds::Patches(bounds) => bounds,
        CrownBounds::Dense(_) => panic!("{context} unexpectedly materialized Dense bounds"),
    }
}

fn assert_small_native_identity_case(
    kernel: ArrayD<f32>,
    bias: Option<Array1<f32>>,
    geometry: DirectIdentityGeometry,
    incoming_lower_b: Vec<f32>,
    incoming_upper_b: Vec<f32>,
) -> Box<PatchesLinearBounds> {
    let mut layer = ConvTranspose2dLayer::new_full(
        kernel.clone(),
        bias.clone(),
        geometry.stride,
        geometry.padding,
        geometry.dilation,
        geometry.output_padding,
    )
    .expect("valid direct identity geometry");
    layer.set_input_shape(geometry.input.0, geometry.input.1);
    let (out_h, out_w) = layer
        .output_size(geometry.input.0, geometry.input.1)
        .expect("valid direct identity output size");
    let out_shape = (layer.out_channels(), out_h, out_w);
    let out_dim = out_shape.0 * out_shape.1 * out_shape.2;
    assert_eq!(incoming_lower_b.len(), out_dim);
    assert_eq!(incoming_upper_b.len(), out_dim);

    let mut dense_in = LinearBounds::identity(out_dim);
    dense_in
        .lower_b_mut()
        .assign(&Array1::from_vec(incoming_lower_b.clone()));
    dense_in
        .upper_b_mut()
        .assign(&Array1::from_vec(incoming_upper_b.clone()));
    let dense = layer
        .propagate_linear(&dense_in)
        .expect("dense identity oracle route")
        .into_owned();

    let mut patch_in = PatchesLinearBounds::identity(out_shape, out_shape);
    patch_in.lower_b.assign(&Array1::from_vec(incoming_lower_b));
    patch_in.upper_b.assign(&Array1::from_vec(incoming_upper_b));
    let result = layer
        .propagate_patches(&patch_in)
        .expect("supported direct identity patches route");
    assert_native_identity_matches_oracle(
        result,
        Some(&dense),
        &kernel,
        bias.as_ref(),
        geometry,
        patch_in.lower_b.as_slice().unwrap(),
        patch_in.upper_b.as_slice().unwrap(),
    )
}

fn assert_dense_identity_coefficients(
    dense: &LinearBounds,
    kernel: &ArrayD<f32>,
    geometry: DirectIdentityGeometry,
) {
    let in_c = kernel.shape()[0];
    let out_c = kernel.shape()[1];
    let (in_h, in_w) = geometry.input;
    let effective_h = (kernel.shape()[2] - 1) * geometry.dilation.0;
    let effective_w = (kernel.shape()[3] - 1) * geometry.dilation.1;
    let out_h = (in_h - 1) * geometry.stride.0 + effective_h + 1 - 2 * geometry.padding.0
        + geometry.output_padding.0;
    let out_w = (in_w - 1) * geometry.stride.1 + effective_w + 1 - 2 * geometry.padding.1
        + geometry.output_padding.1;
    for oc in 0..out_c {
        for qh in 0..out_h {
            for qw in 0..out_w {
                let row = (oc * out_h + qh) * out_w + qw;
                for ic in 0..in_c {
                    for ih in 0..in_h {
                        for iw in 0..in_w {
                            let column = (ic * in_h + ih) * in_w + iw;
                            let expected = direct_identity_coefficient(
                                kernel,
                                geometry,
                                (oc, qh, qw),
                                (ic, ih, iw),
                            );
                            assert_exact_coefficient(
                                dense.lower_a()[[row, column]],
                                expected,
                                "dense-crossover lower coefficient",
                            );
                            assert_exact_coefficient(
                                dense.upper_a()[[row, column]],
                                expected,
                                "dense-crossover upper coefficient",
                            );
                        }
                    }
                }
            }
        }
    }
}

fn assert_official_grid_link(
    input: usize,
    expected_output: usize,
    stride: usize,
    padding: usize,
    kernel_size: usize,
) {
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[1, 1, kernel_size, kernel_size]),
        (1..=kernel_size * kernel_size)
            .map(|value| value as f32)
            .collect(),
    )
    .unwrap();
    let geometry = DirectIdentityGeometry {
        stride: (stride, stride),
        padding: (padding, padding),
        dilation: (1, 1),
        output_padding: (0, 0),
        input: (input, input),
    };
    let mut layer = ConvTranspose2dLayer::new_full(
        kernel.clone(),
        None,
        geometry.stride,
        geometry.padding,
        geometry.dilation,
        geometry.output_padding,
    )
    .unwrap();
    layer.set_input_shape(input, input);
    assert_eq!(
        layer.output_size(input, input).unwrap(),
        (expected_output, expected_output)
    );
    let out_dim = expected_output * expected_output;
    let incoming = PatchesLinearBounds::identity(
        (1, expected_output, expected_output),
        (1, expected_output, expected_output),
    );

    // The first 2->6 link has a 2x2 anchored patch over a 2x2 input: at equal
    // area the native route refuses before allocation so the dispatcher can
    // densify the still-virtual identity and run the exact dense operator.
    if input == 2 && stride == 2 && kernel_size == 4 && padding == 0 {
        let before = incoming.clone();
        assert!(matches!(
            layer.propagate_patches(&incoming),
            Err(NyError::UnsupportedConfiguration(_))
        ));
        assert_patches_input_unchanged(&incoming, &before);
        let dense = layer
            .propagate_linear(&LinearBounds::identity(out_dim))
            .unwrap()
            .into_owned();
        assert_dense_identity_coefficients(&dense, &kernel, geometry);
    } else {
        let result = layer.propagate_patches(&incoming).unwrap();
        let zero_bias = vec![0.0; out_dim];
        assert_native_identity_matches_oracle(
            result, None, &kernel, None, geometry, &zero_bias, &zero_bias,
        );
    }
}

fn direct_identity_refusal_fixture() -> (ConvTranspose2dLayer, PatchesLinearBounds) {
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[1, 1, 3, 3]),
        vec![0.5, -1.0, 1.5, 2.0, -2.5, 3.0, 3.5, -4.0, 4.5],
    )
    .unwrap();
    let mut layer = ConvTranspose2dLayer::new_full(
        kernel,
        Some(Array1::from_vec(vec![0.2])),
        (2, 2),
        (1, 1),
        (1, 1),
        (1, 1),
    )
    .unwrap();
    layer.set_input_shape(4, 4);
    let (out_h, out_w) = layer.output_size(4, 4).unwrap();
    let output = (1, out_h, out_w);
    (layer, PatchesLinearBounds::identity(output, output))
}

fn subnormal_identity_fixture(source: f32) -> (ConvTranspose2dLayer, PatchesLinearBounds) {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![source]).unwrap();
    let mut layer =
        ConvTranspose2dLayer::new_full(kernel, None, (2, 2), (0, 0), (1, 1), (0, 0)).unwrap();
    layer.set_input_shape(4, 4);
    let (out_h, out_w) = layer.output_size(4, 4).unwrap();
    assert_eq!((out_h, out_w), (7, 7));
    let output = (1, out_h, out_w);
    (layer, PatchesLinearBounds::identity(output, output))
}

#[test]
fn convtranspose2d_stride2_identity_uses_exact_negative_anchors() {
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[1, 1, 4, 4]),
        (1..=16).map(|value| value as f32).collect(),
    )
    .unwrap();
    let geometry = DirectIdentityGeometry {
        stride: (2, 2),
        padding: (0, 0),
        dilation: (1, 1),
        output_padding: (0, 0),
        input: (3, 3),
    };
    let shape_layer = ConvTranspose2dLayer::new_full(
        kernel.clone(),
        None,
        geometry.stride,
        geometry.padding,
        geometry.dilation,
        geometry.output_padding,
    )
    .unwrap();
    let (out_h, out_w) = shape_layer
        .output_size(geometry.input.0, geometry.input.1)
        .unwrap();
    assert_eq!((out_h, out_w), (8, 8));
    // The former 6x6 fixture accidentally counted only the stride-expanded
    // input and omitted the k4 overlap tail.
    let out_dim = out_h * out_w;
    let patches = assert_small_native_identity_case(
        kernel,
        None,
        geometry,
        vec![0.0; out_dim],
        vec![0.0; out_dim],
    );
    assert_eq!(patches.lower_a.geometry.origin((0, 0)).unwrap(), (-1, -1));
}

#[test]
fn convtranspose2d_asymmetric_dilation_and_gcd_phase_gaps_are_exact() {
    // gcd(stride, dilation) > 1 leaves entire output residue classes with no
    // input coefficient. They are real bias-only rows, not omitted test cases.
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[1, 2, 3, 2]),
        (1..=12).map(|value| value as f32 * 0.125).collect(),
    )
    .unwrap();
    let bias = Array1::from_vec(vec![0.1, -0.3]);
    let geometry = DirectIdentityGeometry {
        stride: (2, 3),
        padding: (1, 2),
        dilation: (2, 3),
        output_padding: (1, 2),
        input: (3, 3),
    };
    let out_h = 8;
    let out_w = 8;
    let out_dim = 2 * out_h * out_w;
    let lower_b = det_fill(out_dim, 0xA55A, 0.4);
    let upper_b: Vec<f32> = lower_b.iter().map(|value| value + 0.25).collect();
    let patches = assert_small_native_identity_case(kernel, Some(bias), geometry, lower_b, upper_b);
    let dense = patches.to_dense().unwrap();
    let bias_only_rows = (0..out_dim)
        .filter(|&row| dense.lower_a().row(row).iter().all(|value| *value == 0.0))
        .count();
    assert!(
        bias_only_rows > 0,
        "gcd phase-gap fixture must contain bias-only rows"
    );
    assert!(
        bias_only_rows < out_dim,
        "fixture must also contain real coefficients"
    );
}

#[test]
fn convtranspose2d_output_padding_high_edge_keeps_real_anchored_coefficient() {
    // input 3x3, k3, stride2, pad1, output_padding1 -> output 6x6.  The high
    // edge y[5,5] is reached by x[2,2] * W[2,2]; it is not bias-only.
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[1, 1, 3, 3]),
        (1..=9).map(|value| value as f32).collect(),
    )
    .unwrap();
    let bias = Array1::from_vec(vec![0.2]);
    let geometry = DirectIdentityGeometry {
        stride: (2, 2),
        padding: (1, 1),
        dilation: (1, 1),
        output_padding: (1, 1),
        input: (3, 3),
    };
    let out_dim = 36;
    let patches = assert_small_native_identity_case(
        kernel,
        Some(bias),
        geometry,
        vec![0.1; out_dim],
        vec![0.3; out_dim],
    );
    let dense = patches.to_dense().unwrap();
    let high_edge_row = 5 * 6 + 5;
    let high_edge_input = 2 * 3 + 2;
    assert_eq!(
        dense.lower_a()[[high_edge_row, high_edge_input]].to_bits(),
        9.0_f32.to_bits()
    );
    assert_eq!(
        dense.upper_a()[[high_edge_row, high_edge_input]].to_bits(),
        9.0_f32.to_bits()
    );
}

#[test]
fn convtranspose2d_padding_beyond_kernel_minus_one_is_directly_representable() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![2.0]).unwrap();
    let geometry = DirectIdentityGeometry {
        stride: (2, 2),
        padding: (2, 1),
        dilation: (1, 1),
        output_padding: (0, 0),
        input: (4, 4),
    };
    let out_dim = 3 * 5;
    assert_small_native_identity_case(
        kernel,
        None,
        geometry,
        vec![0.0; out_dim],
        vec![0.0; out_dim],
    );
}

#[test]
fn convtranspose2d_asymmetric_raw_area_crossover_refuses_before_allocation() {
    // Raw anchored area is 7x1 while the input area is 2x3=6. Clamping each
    // patch axis to the corresponding input axis would incorrectly compare
    // 2x1 against 6 and admit an allocation that cannot save memory.
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[1, 1, 13, 1]),
        (1..=13).map(|value| value as f32).collect(),
    )
    .unwrap();
    let geometry = DirectIdentityGeometry {
        stride: (2, 2),
        padding: (0, 0),
        dilation: (1, 1),
        output_padding: (0, 0),
        input: (2, 3),
    };
    let mut layer = ConvTranspose2dLayer::new_full(
        kernel.clone(),
        None,
        geometry.stride,
        geometry.padding,
        geometry.dilation,
        geometry.output_padding,
    )
    .unwrap();
    layer.set_input_shape(2, 3);
    let (out_h, out_w) = layer.output_size(2, 3).unwrap();
    assert_eq!((out_h, out_w), (15, 5));
    let out_dim = out_h * out_w;
    let incoming = PatchesLinearBounds::identity((1, out_h, out_w), (1, out_h, out_w));
    let before = incoming.clone();
    assert!(matches!(
        layer.propagate_patches(&incoming),
        Err(NyError::UnsupportedConfiguration(_))
    ));
    assert_patches_input_unchanged(&incoming, &before);

    // The refusal is an optimization decision, not a skipped correctness case:
    // independently validate the exact dense fallback relation.
    let dense = layer
        .propagate_linear(&LinearBounds::identity(out_dim))
        .unwrap()
        .into_owned();
    assert_dense_identity_coefficients(&dense, &kernel, geometry);
}

#[test]
fn convtranspose2d_stride2_materialized_6d_composes_natively() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let (convt, mut incoming) = convt_identity_for(kernel, (2, 2), (0, 0), (1, 1), (0, 0), 4, 4);
    incoming.lower_a = incoming.lower_a.try_materialize_identity().unwrap();
    incoming.upper_a = incoming.upper_a.try_materialize_identity().unwrap();
    incoming.lower_b[0] = -0.0;
    incoming.upper_b[0] = 0.0;
    incoming.lower_b[1] = f32::from_bits(1);
    incoming.upper_b[1] = f32::from_bits(0x8000_0001);
    let before = incoming.clone();
    let dense_in = incoming.to_dense().unwrap();
    let dense = convt
        .propagate_linear_with_engine(&dense_in, None)
        .unwrap()
        .into_owned();
    let result = convt.propagate_patches(&incoming).unwrap();
    let native = match &result {
        CrownBounds::Patches(native) => native,
        CrownBounds::Dense(_) => panic!("below-crossover 6D composition must stay native"),
    };
    assert!(matches!(
        &native.lower_a.geometry,
        PatchGeometry::Anchored(_)
    ));
    assert!(native.lower_a.coeff_err.is_some());
    assert!(native.upper_a.coeff_err.is_some());
    assert_eq!(
        native
            .lower_b
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        incoming
            .lower_b
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        native
            .upper_b
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        incoming
            .upper_b
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    let materialized = result.into_dense().unwrap();
    assert_linear_bounds_close(&dense, &materialized, COMPOSE_TOL).unwrap();
    assert_patches_input_unchanged(&incoming, &before);
}

fn anchored_general_side(
    values: Vec<f32>,
    shape: &[usize],
    output_shape: (usize, usize, usize),
    input_shape: (usize, usize, usize),
    row_origins: Vec<i128>,
    column_origins: Vec<i128>,
    coeff_err: Option<Array1<f32>>,
) -> PatchesData {
    PatchesData {
        patches: Some(ArrayD::from_shape_vec(IxDyn(shape), values).unwrap()),
        geometry: PatchGeometry::anchored(row_origins, column_origins).unwrap(),
        identity: false,
        output_shape,
        input_shape,
        unstable_idx: None,
        coeff_err,
    }
}

/// The first official cGAN transpose link has this same representation ratio:
/// a 2x2 incoming window through k4/s2 becomes a stored 3x3 Anchored window over
/// a 2x2 Dense input. Thus each row stores nine coefficients rather than four.
fn finite_above_dense_crossover_6d_fixture() -> (ConvTranspose2dLayer, PatchesLinearBounds) {
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[1, 1, 4, 4]),
        vec![
            0.5, -0.25, 0.75, 1.0, -1.5, 0.125, 0.375, -0.625, 1.25, -0.5, 0.25, 0.875, -1.125,
            0.625, -0.375, 1.5,
        ],
    )
    .unwrap();
    let mut layer = ConvTranspose2dLayer::new_full(
        kernel,
        Some(Array1::from_vec(vec![0.25])),
        (2, 2),
        (0, 0),
        (1, 1),
        (0, 0),
    )
    .unwrap();
    layer.set_input_shape(2, 2);
    assert_eq!(layer.output_size(2, 2).unwrap(), (6, 6));

    let shape = [1, 1, 1, 1, 2, 2];
    let make_side = |values| {
        anchored_general_side(values, &shape, (1, 1, 1), (1, 6, 6), vec![1], vec![1], None)
    };
    (
        layer,
        PatchesLinearBounds {
            row_count: 1,
            lower_a: make_side(vec![0.25, -0.5, 0.75, 1.0]),
            lower_b: Array1::from_vec(vec![-0.125]),
            upper_a: make_side(vec![-0.125, 0.375, -0.625, 0.875]),
            upper_b: Array1::from_vec(vec![0.375]),
        },
    )
}

#[test]
fn convtranspose2d_finite_above_dense_crossover_6d_defers_materialization() {
    let (layer, incoming) = finite_above_dense_crossover_6d_fixture();
    let before = incoming.clone();

    // Preserve the historical compatibility face. With no deadline, the
    // stored-size optimization still declines this 9-coefficient row in favor
    // of its 4-coefficient Dense equivalent before allocating a result.
    let refusal = layer
        .propagate_patches_engine_and_deadline(&incoming, None, None)
        .expect_err("no-deadline route must retain the stored-size crossover");
    assert!(
        matches!(&refusal, NyError::UnsupportedConfiguration(message)
        if message.contains("stores 9 coefficients") && message.contains("dense size 4"))
    );
    assert_patches_input_unchanged(&incoming, &before);

    let dense_in = incoming.to_dense().unwrap();
    let dense = layer
        .propagate_linear_with_engine(&dense_in, None)
        .unwrap()
        .into_owned();
    let result = layer
        .propagate_patches_engine_and_deadline(
            &incoming,
            None,
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .expect("finite authority may retain the admitted Anchored carrier");
    let native = match &result {
        CrownBounds::Patches(native) => native,
        CrownBounds::Dense(_) => panic!("finite crossover route materialized Dense prematurely"),
    };
    assert_eq!(
        native.lower_a.patches.as_ref().unwrap().shape(),
        &[1, 1, 1, 1, 3, 3]
    );
    assert_eq!(
        native.upper_a.patches.as_ref().unwrap().shape(),
        &[1, 1, 1, 1, 3, 3]
    );
    assert!(matches!(
        &native.lower_a.geometry,
        PatchGeometry::Anchored(_)
    ));
    assert!(native.lower_a.coeff_err.is_some());
    assert!(native.upper_a.coeff_err.is_some());
    let materialized = result.into_dense().unwrap();
    assert_linear_bounds_close(&dense, &materialized, COMPOSE_TOL).unwrap();
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_finite_above_crossover_receipt_and_expiry_are_atomic() {
    use crate::layers::convolution::conv2d::ConvTransposePatchesDeadlineFailpoint;

    let (layer, incoming) = finite_above_dense_crossover_6d_fixture();
    let before = incoming.clone();
    let live_deadline = || Instant::now() + Duration::from_secs(30);

    let required_bytes = match layer
        .propagate_anchored_composition_with_deadline_and_budget_for_test(
            &incoming,
            Some(live_deadline()),
            0,
        )
        .expect_err("zero-byte authority must expose the truthful resident receipt")
    {
        NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes: 0,
            ..
        } => required_bytes,
        other => panic!("unexpected zero-budget result: {other}"),
    };
    assert!(required_bytes > 0);
    assert_patches_input_unchanged(&incoming, &before);

    let exact = expect_native_patches(
        layer
            .propagate_anchored_composition_with_deadline_and_budget_for_test(
                &incoming,
                Some(live_deadline()),
                required_bytes,
            )
            .expect("exact total-live budget must admit the finite transaction"),
        "exact-budget finite crossover route",
    );
    assert_patches_input_unchanged(&incoming, &before);

    match layer.propagate_anchored_composition_with_deadline_and_budget_for_test(
        &incoming,
        Some(live_deadline()),
        required_bytes - 1,
    ) {
        Err(NyError::CpuMemoryExceeded {
            required_bytes: actual_required,
            budget_bytes,
            ..
        }) => {
            assert_eq!(actual_required, required_bytes);
            assert_eq!(budget_bytes, required_bytes - 1);
        }
        other => panic!("budget-minus-one transaction must refuse, got {other:?}"),
    }
    assert_patches_input_unchanged(&incoming, &before);

    let unbounded_budget = expect_native_patches(
        layer
            .propagate_anchored_composition_with_deadline_and_budget_for_test(
                &incoming,
                Some(live_deadline()),
                usize::MAX,
            )
            .expect("larger budget must preserve the exact published carrier"),
        "large-budget finite crossover route",
    );
    assert_patches_bounds_bitwise_equal(&unbounded_budget, &exact);

    {
        // Expire after validation has entered the finite transaction. Any
        // locally allocated work remains unpublished and the borrowed source
        // carrier stays bit-identical.
        let _failpoint = ConvTransposePatchesDeadlineFailpoint::after_successful_polls(10);
        let error = layer
            .propagate_patches_engine_and_deadline(&incoming, None, Some(live_deadline()))
            .expect_err("mid-transaction expiry must discard local work");
        assert!(matches!(error, NyError::DeadlineExceeded(ref message)
            if message.contains("ConvTranspose2d Patches backward")
                && !message.contains("before dispatch")));
    }
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_stride2_explicit_row_7d_composes_natively() {
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[1, 1, 3, 3]),
        vec![0.5, -0.25, 0.75, 1.0, -1.5, 0.125, 0.375, -0.625, 1.25],
    )
    .unwrap();
    let mut layer = ConvTranspose2dLayer::new_full(
        kernel,
        Some(Array1::from_vec(vec![0.375])),
        (2, 2),
        (0, 0),
        (1, 1),
        (0, 0),
    )
    .unwrap();
    layer.set_input_shape(5, 5);
    let (out_h, out_w) = layer.output_size(5, 5).unwrap();
    assert_eq!((out_h, out_w), (11, 11));

    let rows = 2usize;
    let shape = [rows, 1, 1, 2, 1, 3, 3];
    let len: usize = shape.iter().product();
    let make_side = |seed| {
        anchored_general_side(
            det_fill(len, seed, 0.75),
            &shape,
            (1, 1, 2),
            (1, out_h, out_w),
            vec![0],
            vec![0, 2],
            None,
        )
    };
    let incoming = PatchesLinearBounds {
        row_count: rows,
        lower_a: make_side(0x7101),
        lower_b: Array1::from_vec(vec![-0.25, 0.5]),
        upper_a: make_side(0x7102),
        upper_b: Array1::from_vec(vec![0.25, 0.75]),
    };
    let before = incoming.clone();
    let dense_in = incoming.to_dense().unwrap();
    let dense = layer
        .propagate_linear_with_engine(&dense_in, None)
        .unwrap()
        .into_owned();
    let result = layer.propagate_patches(&incoming).unwrap();
    let native = match &result {
        CrownBounds::Patches(native) => native,
        CrownBounds::Dense(_) => panic!("below-crossover explicit-row composition returned Dense"),
    };
    assert_eq!(
        native.lower_a.patches.as_ref().unwrap().shape(),
        &[rows, 1, 1, 2, 1, 3, 3]
    );
    assert_eq!(native.lower_a.coeff_err.as_ref().unwrap().len(), rows);
    assert_eq!(native.upper_a.coeff_err.as_ref().unwrap().len(), rows);
    let materialized = result.into_dense().unwrap();
    assert_linear_bounds_close(&dense, &materialized, COMPOSE_TOL).unwrap();
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_explicit_row_crossover_counts_the_repeated_position_slab() {
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[1, 1, 3, 3]),
        (1..=9).map(|value| value as f32 * 0.125).collect(),
    )
    .unwrap();
    let mut layer =
        ConvTranspose2dLayer::new_full(kernel, None, (2, 2), (0, 0), (1, 1), (0, 0)).unwrap();
    layer.set_input_shape(5, 5);
    let (out_h, out_w) = layer.output_size(5, 5).unwrap();
    assert_eq!((out_h, out_w), (11, 11));

    // Each composed patch is only 3x3, below the 5x5 dense input area. But a
    // 7D explicit row repeats that patch for all three spec positions, so its
    // actual stored slab is 27 coefficients versus 25 in the dense row.
    let shape = [1, 1, 1, 3, 1, 3, 3];
    let make_side = || {
        anchored_general_side(
            vec![0.25; shape.iter().product()],
            &shape,
            (1, 1, 3),
            (1, out_h, out_w),
            vec![0],
            vec![0, 2, 4],
            None,
        )
    };
    let incoming = PatchesLinearBounds {
        row_count: 1,
        lower_a: make_side(),
        lower_b: Array1::zeros(1),
        upper_a: make_side(),
        upper_b: Array1::zeros(1),
    };
    let before = incoming.clone();
    assert!(matches!(
        layer.propagate_patches(&incoming),
        Err(NyError::UnsupportedConfiguration(_))
    ));
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_duplicate_destination_rational_oracle_covers_intrinsic_carry_and_bias() {
    // For x=(0,0), q=(ki,kj) for all nine taps. Thus nine distinct incoming
    // coefficients consolidate into one emitted destination. All values are
    // dyadic, so the f64 oracle below is an exact rational evaluation.
    let weights = vec![0.5, -0.25, 0.75, 1.0, -1.5, 0.125, 0.375, -0.625, 1.25];
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), weights.clone()).unwrap();
    let layer_bias = 0.5f32;
    let mut layer = ConvTranspose2dLayer::new_full(
        kernel,
        Some(Array1::from_vec(vec![layer_bias])),
        (2, 2),
        (0, 0),
        (1, 1),
        (0, 0),
    )
    .unwrap();
    layer.set_input_shape(4, 4);
    let (out_h, out_w) = layer.output_size(4, 4).unwrap();
    assert_eq!((out_h, out_w), (9, 9));

    let stored = vec![0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -1.75, 2.0, 0.125];
    let old_error = 0.0625f32;
    let shape = [1, 1, 1, 1, 3, 3];
    let make_side = || {
        anchored_general_side(
            stored.clone(),
            &shape,
            (1, 1, 1),
            (1, out_h, out_w),
            vec![0],
            vec![0],
            Some(Array1::from_vec(vec![old_error])),
        )
    };
    let incoming = PatchesLinearBounds {
        row_count: 1,
        lower_a: make_side(),
        lower_b: Array1::from_vec(vec![-0.125]),
        upper_a: make_side(),
        upper_b: Array1::from_vec(vec![0.25]),
    };
    let result = match layer.propagate_patches(&incoming).unwrap() {
        CrownBounds::Patches(result) => result,
        CrownBounds::Dense(_) => panic!("duplicate-consolidation fixture crossed to Dense"),
    };
    let origin = result.lower_a.geometry.origin((0, 0)).unwrap();
    assert_eq!(origin, (-1, -1));
    let x0_tap = [0, 0, 0, 0, 1, 1];
    let mut exact_stored = 0.0f64;
    for index in 0..9 {
        exact_stored += f64::from(stored[index]) * f64::from(weights[index]);
    }
    assert_eq!(
        exact_stored, 2.125,
        "nine dyadic products consolidate exactly"
    );

    // Exhaust the rational interval's 2^9 corners. Every operand, weight,
    // error radius, and bias is dyadic, so these binary64 calculations are
    // exact rather than a second rounded implementation of the production
    // contraction. This catches duplicate loss as well as a carried-error or
    // repeated-bias factor that was counted only once.
    for mask in 0usize..(1usize << stored.len()) {
        let mut exact_true = 0.0f64;
        let mut true_bias_fold = 0.0f64;
        for index in 0..stored.len() {
            let perturbation = if mask & (1usize << index) == 0 {
                -f64::from(old_error)
            } else {
                f64::from(old_error)
            };
            let true_coefficient = f64::from(stored[index]) + perturbation;
            exact_true += true_coefficient * f64::from(weights[index]);
            true_bias_fold += true_coefficient * f64::from(layer_bias);
        }
        for side in [&result.lower_a, &result.upper_a] {
            let stored_result = f64::from(side.patches.as_ref().unwrap()[x0_tap]);
            let certificate = f64::from(side.coeff_err.as_ref().unwrap()[0]);
            assert!(certificate.is_finite() && certificate > 0.0);
            assert!(
                (stored_result - exact_true).abs() <= certificate,
                "published coefficient {stored_result} +/- {certificate} excludes exact rational corner {exact_true} (mask {mask:#x})"
            );
        }
        assert!(
            result.lower_b[0] as f64 <= -0.125 + true_bias_fold,
            "lower bias excludes exact rational corner at mask {mask:#x}"
        );
        assert!(
            result.upper_b[0] as f64 >= 0.25 + true_bias_fold,
            "upper bias excludes exact rational corner at mask {mask:#x}"
        );
    }
}

#[test]
fn convtranspose2d_explicit_row_error_max_and_bias_carry_cover_entire_position_slab_per_side() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![2.0]).unwrap();
    let mut layer = ConvTranspose2dLayer::new_full(
        kernel,
        Some(Array1::from_vec(vec![0.5])),
        (2, 2),
        (0, 0),
        (1, 1),
        (0, 0),
    )
    .unwrap();
    layer.set_input_shape(5, 5);
    let (out_h, out_w) = layer.output_size(5, 5).unwrap();
    assert_eq!((out_h, out_w), (9, 9));

    let lower_old_error = 0.125f32;
    let upper_old_error = 0.375f32;
    let shape = [1, 1, 1, 2, 1, 1, 1];
    let make_side = |values, old_error| {
        anchored_general_side(
            values,
            &shape,
            (1, 1, 2),
            (1, out_h, out_w),
            vec![0],
            vec![0, 2],
            Some(Array1::from_vec(vec![old_error])),
        )
    };
    let incoming = PatchesLinearBounds {
        row_count: 1,
        lower_a: make_side(vec![1.0, 3.0], lower_old_error),
        lower_b: Array1::from_vec(vec![-0.25]),
        upper_a: make_side(vec![2.0, -4.0], upper_old_error),
        upper_b: Array1::from_vec(vec![0.25]),
    };
    let result = match layer.propagate_patches(&incoming).unwrap() {
        CrownBounds::Patches(result) => result,
        CrownBounds::Dense(_) => panic!("explicit-row error slab fixture crossed to Dense"),
    };
    // Choose independent true coefficients at opposite ends of each side's
    // shared per-row certificate. Each emitted certificate must cover every
    // destination in its own position slab, without borrowing the other side.
    for (side, true_values) in [
        (
            &result.lower_a,
            [
                f64::from(1.0 + lower_old_error) * 2.0,
                f64::from(3.0 - lower_old_error) * 2.0,
            ],
        ),
        (
            &result.upper_a,
            [
                f64::from(2.0 - upper_old_error) * 2.0,
                f64::from(-4.0 + upper_old_error) * 2.0,
            ],
        ),
    ] {
        let values = side.patches.as_ref().unwrap();
        let error = f64::from(side.coeff_err.as_ref().unwrap()[0]);
        for (position, truth) in true_values.into_iter().enumerate() {
            let stored = f64::from(values[[0, 0, 0, position, 0, 0, 0]]);
            assert!(
                (stored - truth).abs() <= error,
                "row error {error} does not cover position {position}: {stored} vs {truth}"
            );
        }
    }
    let lower_true_bias_fold = f64::from(1.0 + lower_old_error + 3.0 - lower_old_error) * 0.5;
    let upper_true_bias_fold = f64::from(2.0 - upper_old_error - 4.0 + upper_old_error) * 0.5;
    assert!(result.lower_b[0] as f64 <= -0.25 + lower_true_bias_fold);
    assert!(result.upper_b[0] as f64 >= 0.25 + upper_true_bias_fold);
}

#[test]
fn convtranspose2d_general_f32_overflow_uses_finite_center_and_infinite_certificate() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![f32::MAX]).unwrap();
    let mut layer =
        ConvTranspose2dLayer::new_full(kernel, None, (2, 2), (0, 0), (1, 1), (0, 0)).unwrap();
    layer.set_input_shape(4, 4);
    let (out_h, out_w) = layer.output_size(4, 4).unwrap();
    let shape = [1, 1, 1, 1, 1, 1];
    let make_side = || {
        anchored_general_side(
            vec![f32::MAX],
            &shape,
            (1, 1, 1),
            (1, out_h, out_w),
            vec![0],
            vec![0],
            None,
        )
    };
    let incoming = PatchesLinearBounds {
        row_count: 1,
        lower_a: make_side(),
        lower_b: Array1::zeros(1),
        upper_a: make_side(),
        upper_b: Array1::zeros(1),
    };
    let result = match layer.propagate_patches(&incoming).unwrap() {
        CrownBounds::Patches(result) => result,
        CrownBounds::Dense(_) => panic!("overflow fixture crossed to Dense"),
    };
    for side in [&result.lower_a, &result.upper_a] {
        let center = side.patches.as_ref().unwrap()[[0, 0, 0, 0, 0, 0]];
        assert!(
            center.is_finite(),
            "overflow must never publish an infinite center"
        );
        assert_eq!(center, 0.0);
        assert_eq!(side.coeff_err.as_ref().unwrap()[0], f32::INFINITY);
    }
}

#[test]
fn convtranspose2d_general_subnormal_coefficients_publish_zero_center_with_normal_certificate() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0]).unwrap();
    let mut layer =
        ConvTranspose2dLayer::new_full(kernel, None, (2, 2), (0, 0), (1, 1), (0, 0)).unwrap();
    layer.set_input_shape(4, 4);
    let (out_h, out_w) = layer.output_size(4, 4).unwrap();
    let shape = [1, 1, 1, 1, 1, 1];
    let exact_min_subnormal = f64::from_bits(874_u64 << 52); // exactly 2^-149

    for (source, exact) in [
        (f32::from_bits(1), exact_min_subnormal),
        (f32::from_bits(0x8000_0001), -exact_min_subnormal),
    ] {
        let make_side = || {
            anchored_general_side(
                vec![source],
                &shape,
                (1, 1, 1),
                (1, out_h, out_w),
                vec![0],
                vec![0],
                None,
            )
        };
        let incoming = PatchesLinearBounds {
            row_count: 1,
            lower_a: make_side(),
            lower_b: Array1::zeros(1),
            upper_a: make_side(),
            upper_b: Array1::zeros(1),
        };
        let result = match layer.propagate_patches(&incoming).unwrap() {
            CrownBounds::Patches(result) => result,
            CrownBounds::Dense(_) => panic!("subnormal fixture crossed to Dense"),
        };
        for side in [&result.lower_a, &result.upper_a] {
            let center = side.patches.as_ref().unwrap()[[0, 0, 0, 0, 0, 0]];
            let error = side.coeff_err.as_ref().unwrap()[0];
            assert_eq!(center.to_bits(), 0.0f32.to_bits());
            assert!(error.is_finite());
            assert!(error >= f32::MIN_POSITIVE);
            assert!((f64::from(center) - exact).abs() <= f64::from(error));
        }
    }
}

#[test]
fn convtranspose2d_general_zero_kernel_short_circuits_infinite_carry() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![0.0]).unwrap();
    let mut layer =
        ConvTranspose2dLayer::new_full(kernel, None, (2, 2), (0, 0), (1, 1), (0, 0)).unwrap();
    layer.set_input_shape(4, 4);
    let (out_h, out_w) = layer.output_size(4, 4).unwrap();
    let shape = [1, 1, 1, 1, 1, 1];
    let make_side = || {
        anchored_general_side(
            vec![1.0],
            &shape,
            (1, 1, 1),
            (1, out_h, out_w),
            vec![0],
            vec![0],
            Some(Array1::from_vec(vec![f32::INFINITY])),
        )
    };
    let incoming = PatchesLinearBounds {
        row_count: 1,
        lower_a: make_side(),
        lower_b: Array1::zeros(1),
        upper_a: make_side(),
        upper_b: Array1::zeros(1),
    };
    let result = match layer.propagate_patches(&incoming).unwrap() {
        CrownBounds::Patches(result) => result,
        CrownBounds::Dense(_) => panic!("zero-kernel fixture crossed to Dense"),
    };
    for side in [&result.lower_a, &result.upper_a] {
        assert_eq!(side.patches.as_ref().unwrap()[[0, 0, 0, 0, 0, 0]], 0.0);
        assert_eq!(side.coeff_err.as_ref().unwrap()[0], 0.0);
    }
}

#[test]
fn convtranspose2d_general_invalid_carried_error_poison_is_never_nan() {
    for poison in [f32::NAN, -1.0, f32::INFINITY] {
        let (layer, mut incoming) = general_malformed_fixture();
        incoming.lower_a.coeff_err = Some(Array1::from_vec(vec![poison]));
        incoming.upper_a.coeff_err = Some(Array1::from_vec(vec![poison]));
        let result = match layer.propagate_patches(&incoming).unwrap() {
            CrownBounds::Patches(result) => result,
            CrownBounds::Dense(_) => panic!("poison fixture crossed to Dense"),
        };
        assert_eq!(result.lower_a.coeff_err.as_ref().unwrap()[0], f32::INFINITY);
        assert_eq!(result.upper_a.coeff_err.as_ref().unwrap()[0], f32::INFINITY);
    }
}

fn assert_official_grid_general_probe(
    input: usize,
    expected_output: usize,
    stride: usize,
    padding: usize,
    kernel_size: usize,
) {
    assert!(stride > 1, "stage-4 probe is the anchored stride>1 route");
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[1, 1, kernel_size, kernel_size]),
        (1..=kernel_size * kernel_size)
            .map(|value| (value as f32) * 0.125)
            .collect(),
    )
    .unwrap();
    let mut layer = ConvTranspose2dLayer::new_full(
        kernel,
        Some(Array1::from_vec(vec![0.25])),
        (stride, stride),
        (padding, padding),
        (1, 1),
        (0, 0),
    )
    .unwrap();
    layer.set_input_shape(input, input);
    assert_eq!(
        layer.output_size(input, input).unwrap(),
        (expected_output, expected_output)
    );
    let shape = [1, 1, 1, 1, 1, 1, 1];
    let make_side = |value| {
        anchored_general_side(
            vec![value],
            &shape,
            (1, 1, 1),
            (1, expected_output, expected_output),
            vec![(expected_output - 1) as i128],
            vec![(expected_output - 1) as i128],
            None,
        )
    };
    let incoming = PatchesLinearBounds {
        row_count: 1,
        lower_a: make_side(-0.75),
        lower_b: Array1::from_vec(vec![-0.125]),
        upper_a: make_side(1.25),
        upper_b: Array1::from_vec(vec![0.375]),
    };
    let before = incoming.clone();
    let dense_in = incoming.to_dense().unwrap();
    let dense = layer
        .propagate_linear_with_engine(&dense_in, None)
        .unwrap()
        .into_owned();
    let patch_extent = (kernel_size - 1) / stride + 1;
    let native = layer.propagate_patches(&incoming);
    if patch_extent * patch_extent >= input * input {
        assert!(matches!(native, Err(NyError::UnsupportedConfiguration(_))));
    } else {
        let native = native.unwrap();
        assert!(matches!(&native, CrownBounds::Patches(_)));
        let materialized = native.into_dense().unwrap();
        assert_linear_bounds_close(&dense, &materialized, COMPOSE_TOL).unwrap();
    }
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_official_stride2_grids_have_general_7d_high_edge_probes() {
    for &(input, output, padding) in &[
        (2, 6, 0),
        (6, 14, 0),
        (14, 30, 0),
        (30, 62, 0),
        (4, 8, 1),
        (8, 16, 1),
        (16, 32, 1),
    ] {
        assert_official_grid_general_probe(input, output, 2, padding, 4);
    }
}

#[test]
fn convtranspose2d_general_asymmetric_dilation_padding_output_padding_and_negative_origins() {
    let kernel = make_convt_kernel(2, 2, 3, 2, 0xA5A5_7711);
    let mut layer = ConvTranspose2dLayer::new_full(
        kernel,
        Some(Array1::from_vec(vec![0.25, -0.375])),
        (2, 3),
        (2, 1),
        (2, 1),
        (1, 2),
    )
    .unwrap();
    layer.set_input_shape(6, 7);
    let (out_h, out_w) = layer.output_size(6, 7).unwrap();
    assert_eq!((out_h, out_w), (12, 20));

    let shape = [1, 1, 2, 2, 3, 4];
    let len: usize = shape.iter().product();
    let make_side = |seed| {
        anchored_general_side(
            det_fill(len, seed, 0.5),
            &shape,
            (1, 1, 2),
            (2, out_h, out_w),
            vec![-2],
            vec![-1, (out_w - 2) as i128],
            None,
        )
    };
    let incoming = PatchesLinearBounds {
        row_count: 2,
        lower_a: make_side(0xA501),
        lower_b: Array1::from_vec(vec![-0.5, 0.125]),
        upper_a: make_side(0xA502),
        upper_b: Array1::from_vec(vec![0.5, 0.625]),
    };
    let before = incoming.clone();
    let dense_in = incoming.to_dense().unwrap();
    let dense = layer
        .propagate_linear_with_engine(&dense_in, None)
        .unwrap()
        .into_owned();
    let result = layer.propagate_patches(&incoming).unwrap();
    let native = match &result {
        CrownBounds::Patches(native) => native,
        CrownBounds::Dense(_) => panic!("asymmetric below-crossover route returned Dense"),
    };
    assert!(matches!(
        &native.lower_a.geometry,
        PatchGeometry::Anchored(_)
    ));
    assert!(native.lower_a.geometry.origin((0, 0)).unwrap().0 < 0);
    let materialized = result.into_dense().unwrap();
    assert_linear_bounds_close(&dense, &materialized, COMPOSE_TOL).unwrap();
    assert_patches_input_unchanged(&incoming, &before);
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(120) })]

    /// Stage-4 native composition gate over both carrier families. A one-position
    /// incoming carrier keeps the independent dense oracle small while varying
    /// every ConvTranspose geometry parameter and anchors on/beyond both edges.
    #[ntest::timeout(30000)]
    #[test]
    fn proptest_convtranspose2d_general_6d_7d_dense_parity(
        explicit_rows in proptest::bool::ANY,
        in_c in 1usize..=2,
        out_c in 1usize..=2,
        in_h in 5usize..=7,
        in_w in 5usize..=7,
        sh in 2usize..=3,
        sw in 2usize..=3,
        kh in 1usize..=3,
        kw in 1usize..=3,
        dh in 1usize..=2,
        dw in 1usize..=2,
        ph in 0usize..=2,
        pw in 0usize..=2,
        raw_oph in 0usize..=7,
        raw_opw in 0usize..=7,
        previous_h in 1usize..=3,
        previous_w in 1usize..=3,
        high_row in proptest::bool::ANY,
        high_column in proptest::bool::ANY,
        row_offset in -2i32..=2,
        column_offset in -2i32..=2,
        seed in any::<u64>(),
    ) {
        let output_padding = (raw_oph % sh, raw_opw % sw);
        let kernel = make_convt_kernel(in_c, out_c, kh, kw, seed);
        let mut layer = ConvTranspose2dLayer::new_full(
            kernel,
            Some(make_bias(out_c, seed)),
            (sh, sw),
            (ph, pw),
            (dh, dw),
            output_padding,
        ).map_err(|error| TestCaseError::fail(format!("layer: {error}")))?;
        layer.set_input_shape(in_h, in_w);
        let (out_h, out_w) = layer.output_size(in_h, in_w)
            .map_err(|error| TestCaseError::fail(format!("output: {error}")))?;
        let row_base = if high_row { out_h as i128 - 1 } else { 0 };
        let column_base = if high_column { out_w as i128 - 1 } else { 0 };
        let row_origin = row_base + i128::from(row_offset);
        let column_origin = column_base + i128::from(column_offset);
        let shape = if explicit_rows {
            vec![1, 1, 1, 1, out_c, previous_h, previous_w]
        } else {
            vec![1, 1, 1, out_c, previous_h, previous_w]
        };
        let len: usize = shape.iter().product();
        let make_side = |salt| {
            anchored_general_side(
                det_fill(len, salt, 0.5),
                &shape,
                (1, 1, 1),
                (out_c, out_h, out_w),
                vec![row_origin],
                vec![column_origin],
                None,
            )
        };
        let incoming = PatchesLinearBounds {
            row_count: 1,
            lower_a: make_side(seed ^ 0x61),
            lower_b: Array1::from_vec(vec![-0.25]),
            upper_a: make_side(seed ^ 0x72),
            upper_b: Array1::from_vec(vec![0.375]),
        };
        let before = incoming.clone();
        let dense_in = incoming.to_dense()
            .map_err(|error| TestCaseError::fail(format!("incoming dense: {error}")))?;
        let dense = layer.propagate_linear_with_engine(&dense_in, None)
            .map_err(|error| TestCaseError::fail(format!("dense propagation: {error}")))?
            .into_owned();
        let result = layer.propagate_patches(&incoming)
            .map_err(|error| TestCaseError::fail(format!("native propagation: {error}")))?;
        prop_assert!(matches!(&result, CrownBounds::Patches(_)));
        let materialized = result.into_dense()
            .map_err(|error| TestCaseError::fail(format!("result dense: {error}")))?;
        assert_linear_bounds_close(&dense, &materialized, COMPOSE_TOL)?;
        assert_patches_input_unchanged(&incoming, &before);
    }
}

#[test]
fn convtranspose2d_stride2_mixed_identity_sides_refuse_atomically() {
    let (layer, identity) = direct_identity_refusal_fixture();
    for materialize_lower in [true, false] {
        let mut mixed = identity.clone();
        if materialize_lower {
            mixed.lower_a = mixed.lower_a.try_materialize_identity().unwrap();
        } else {
            mixed.upper_a = mixed.upper_a.try_materialize_identity().unwrap();
        }
        let before = mixed.clone();
        assert!(matches!(
            layer.propagate_patches(&mixed),
            Err(NyError::InvalidSpec(_) | NyError::UnsupportedConfiguration(_))
        ));
        assert_patches_input_unchanged(&mixed, &before);
    }
}

#[test]
fn convtranspose2d_stride2_sparse_identity_refuses_without_mutation() {
    let (layer, full) = direct_identity_refusal_fixture();
    let output = full.lower_a.output_shape;
    let sparse = PatchesLinearBounds::sparse_identity(
        output,
        output,
        UnstableIdx {
            channels: vec![0, 0],
            heights: vec![0, output.1 - 1],
            widths: vec![0, output.2 - 1],
        },
    );
    let before = sparse.clone();
    assert!(matches!(
        layer.propagate_patches(&sparse),
        Err(NyError::UnsupportedConfiguration(_))
    ));
    assert_patches_input_unchanged(&sparse, &before);
}

#[test]
fn convtranspose2d_stride2_identity_coeff_err_refuses_without_mutation() {
    let (layer, mut incoming) = direct_identity_refusal_fixture();
    incoming.lower_a.coeff_err = Some(Array1::zeros(incoming.row_count));
    incoming.upper_a.coeff_err = Some(Array1::zeros(incoming.row_count));
    let before = incoming.clone();
    assert!(matches!(
        layer.propagate_anchored_identity_with_budget_for_test(&incoming, usize::MAX),
        Err(NyError::InternalError(_) | NyError::UnsupportedConfiguration(_))
    ));
    assert!(matches!(
        layer.propagate_patches(&incoming),
        Err(NyError::InternalError(_) | NyError::UnsupportedConfiguration(_))
    ));
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_stride2_nonfinite_sources_refuse_without_mutation() {
    let (layer, mut bad_incoming) = direct_identity_refusal_fixture();
    bad_incoming.lower_b[0] = f32::NAN;
    let before = bad_incoming.clone();
    assert!(matches!(
        layer.propagate_patches(&bad_incoming),
        Err(NyError::NumericalInstability(_))
    ));
    // Compare all non-NaN carrier state explicitly; NaN is intentionally not
    // reflexive, so the generic equality helper cannot represent this case.
    assert_eq!(
        bad_incoming.lower_b[0].to_bits(),
        before.lower_b[0].to_bits()
    );
    assert_eq!(bad_incoming.upper_b, before.upper_b);
    assert_eq!(bad_incoming.lower_a.patches, before.lower_a.patches);
    assert_eq!(bad_incoming.upper_a.patches, before.upper_a.patches);
    assert_eq!(bad_incoming.lower_a.geometry, before.lower_a.geometry);
    assert_eq!(bad_incoming.upper_a.geometry, before.upper_a.geometry);

    for (source, kernel_value, bias_value) in [
        ("NaN kernel", f32::NAN, 0.0),
        ("+Inf kernel", f32::INFINITY, 0.0),
        ("-Inf kernel", f32::NEG_INFINITY, 0.0),
        ("NaN bias", 1.0, f32::NAN),
        ("+Inf bias", 1.0, f32::INFINITY),
        ("-Inf bias", 1.0, f32::NEG_INFINITY),
    ] {
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![kernel_value]).unwrap();
        let mut nonfinite_layer = ConvTranspose2dLayer::new_full(
            kernel,
            Some(Array1::from_vec(vec![bias_value])),
            (2, 2),
            (0, 0),
            (1, 1),
            (0, 0),
        )
        .unwrap();
        nonfinite_layer.set_input_shape(4, 4);
        let (out_h, out_w) = nonfinite_layer.output_size(4, 4).unwrap();
        let identity = PatchesLinearBounds::identity((1, out_h, out_w), (1, out_h, out_w));
        let before = identity.clone();
        let deadline = Instant::now() + Duration::from_secs(30);
        assert!(
            matches!(
                nonfinite_layer.propagate_patches_engine_and_deadline(
                    &identity,
                    None,
                    Some(deadline),
                ),
                Err(NyError::NumericalInstability(_))
            ),
            "{source}"
        );
        assert!(
            Instant::now() < deadline,
            "{source} exhausted live authority"
        );
        assert_patches_input_unchanged(&identity, &before);
    }
}

#[test]
fn convtranspose2d_stride2_without_layer_bias_preserves_incoming_bias_bits() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0]).unwrap();
    let geometry = DirectIdentityGeometry {
        stride: (2, 2),
        padding: (0, 0),
        dilation: (1, 1),
        output_padding: (0, 0),
        input: (4, 4),
    };
    let out_dim: usize = 7 * 7;
    let lower: Vec<f32> = (0..out_dim)
        .map(|index| if index.is_multiple_of(2) { -0.0 } else { 0.25 })
        .collect();
    let upper: Vec<f32> = (0..out_dim)
        .map(|index| if index.is_multiple_of(2) { 0.0 } else { 0.5 })
        .collect();
    let expected_lower: Vec<u32> = lower.iter().map(|value| value.to_bits()).collect();
    let expected_upper: Vec<u32> = upper.iter().map(|value| value.to_bits()).collect();
    let patches = assert_small_native_identity_case(kernel, None, geometry, lower, upper);
    assert_eq!(
        patches
            .lower_b
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_lower
    );
    assert_eq!(
        patches
            .upper_b
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_upper
    );
}

#[test]
fn convtranspose2d_stride2_subnormal_bias_sum_publishes_normal_or_zero_outward() {
    let min_subnormal = f32::from_bits(1);
    let min_subnormal_f64 = f64::from_bits(874_u64 << 52); // exactly 2^-149
    for (operand, exact_operand) in [
        (min_subnormal, min_subnormal_f64),
        (f32::from_bits(0x8000_0001), -min_subnormal_f64),
    ] {
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0]).unwrap();
        let geometry = DirectIdentityGeometry {
            stride: (2, 2),
            padding: (0, 0),
            dilation: (1, 1),
            output_padding: (0, 0),
            input: (4, 4),
        };
        let out_dim = 7 * 7;
        let patches = assert_small_native_identity_case(
            kernel,
            Some(Array1::from_vec(vec![operand])),
            geometry,
            vec![operand; out_dim],
            vec![operand; out_dim],
        );
        let exact_sum = exact_operand + exact_operand;
        for row in 0..out_dim {
            let lower = patches.lower_b[row];
            let upper = patches.upper_b[row];
            for endpoint in [lower, upper] {
                let magnitude = endpoint.to_bits() & 0x7fff_ffff;
                assert!(
                    magnitude == 0 || magnitude >= f32::MIN_POSITIVE.to_bits(),
                    "published endpoint must not be binary32-subnormal: {endpoint:e}"
                );
            }
            assert!(
                lower as f64 <= exact_sum && upper as f64 >= exact_sum,
                "published [{lower:e}, {upper:e}] excludes exact subnormal sum {exact_sum:e}"
            );
        }
    }
}

#[test]
fn convtranspose2d_identity_normalizes_signed_subnormal_kernel_with_per_row_certificate() {
    let exact_min_subnormal = f64::from_bits(874_u64 << 52); // exactly 2^-149
    for (source, exact) in [
        (f32::from_bits(1), exact_min_subnormal),
        (f32::from_bits(0x8000_0001), -exact_min_subnormal),
    ] {
        let (layer, incoming) = subnormal_identity_fixture(source);
        let before = incoming.clone();
        let result = match layer.propagate_patches(&incoming).unwrap() {
            CrownBounds::Patches(result) => result,
            CrownBounds::Dense(_) => panic!("subnormal identity fixture crossed to Dense"),
        };
        assert_patches_input_unchanged(&incoming, &before);
        let lower = result.lower_a.patches.as_ref().unwrap();
        let upper = result.upper_a.patches.as_ref().unwrap();
        let lower_error = result.lower_a.coeff_err.as_ref().unwrap();
        let upper_error = result.upper_a.coeff_err.as_ref().unwrap();
        assert_eq!(lower_error.len(), 49);
        assert_eq!(upper_error.len(), 49);

        for row in 0usize..7 {
            for column in 0usize..7 {
                let logical_row = row * 7 + column;
                let receives_source = row.is_multiple_of(2) && column.is_multiple_of(2);
                for (coefficients, errors) in [(lower, lower_error), (upper, upper_error)] {
                    let center = coefficients[[0, row, column, 0, 0, 0]];
                    let error = errors[logical_row];
                    assert_eq!(
                        center.to_bits(),
                        0.0f32.to_bits(),
                        "stored identity center must be DAZ-stable"
                    );
                    if receives_source {
                        assert!(error.is_finite() && error >= f32::MIN_POSITIVE);
                        assert!(
                            (f64::from(center) - exact).abs() <= f64::from(error),
                            "row {logical_row} certificate excludes exact signed subnormal"
                        );
                    } else {
                        assert_eq!(error.to_bits(), 0.0f32.to_bits());
                    }
                }
            }
        }
    }
}

#[test]
fn convtranspose2d_identity_subnormal_certificate_survives_relu_bn_materialize_and_concretize() {
    let exact_min_subnormal = f64::from_bits(874_u64 << 52); // exactly 2^-149
    for (source, exact_weight) in [
        (f32::from_bits(1), exact_min_subnormal),
        (f32::from_bits(0x8000_0001), -exact_min_subnormal),
    ] {
        let (layer, incoming) = subnormal_identity_fixture(source);
        let after_identity = match layer.propagate_patches(&incoming).unwrap() {
            CrownBounds::Patches(result) => result,
            CrownBounds::Dense(_) => panic!("subnormal identity fixture crossed to Dense"),
        };
        // Model a DAZ consumer explicitly: every source-bearing center is zero,
        // so ReLU sign selection sees exactly the value DAZ hardware would see.
        assert!(after_identity
            .lower_a
            .patches
            .as_ref()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0));

        let relu_domain = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 4, 4]), -1.0),
            ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1.0),
        )
        .unwrap();
        let after_relu = match ReLULayer
            .propagate_patches_with_bounds(&after_identity, &relu_domain)
            .unwrap()
        {
            CrownBounds::Patches(result) => result,
            CrownBounds::Dense(_) => panic!("subnormal ReLU consumer crossed to Dense"),
        };

        let assert_encloses = |concrete: &BoundedTensor, stage: &str| {
            assert_eq!(concrete.lower().len(), 49);
            for row in 0usize..7 {
                for column in 0usize..7 {
                    let output = row * 7 + column;
                    let receives_source = row.is_multiple_of(2) && column.is_multiple_of(2);
                    let (exact_lower, exact_upper) = if !receives_source {
                        (0.0, 0.0)
                    } else if exact_weight.is_sign_negative() {
                        (exact_weight, 0.0)
                    } else {
                        (0.0, exact_weight)
                    };
                    let actual_lower =
                        ny_core::f32_to_f64_exact(concrete.lower().as_slice().unwrap()[output]);
                    let actual_upper =
                        ny_core::f32_to_f64_exact(concrete.upper().as_slice().unwrap()[output]);
                    assert!(
                        actual_lower <= exact_lower && actual_upper >= exact_upper,
                        "{stage} output ({row},{column}) [{actual_lower:e}, {actual_upper:e}] excludes exact [{exact_lower:e}, {exact_upper:e}]"
                    );
                }
            }
        };

        let relu_dense = after_relu.to_dense().unwrap();
        let relu_concrete = relu_dense.concretize_sound(&relu_domain);
        assert_encloses(&relu_concrete, "ReLU/materialize/concretize");

        // BatchNorm z=2x maps x in [-0.5,0.5] back to the authenticated ReLU
        // domain [-1,1]. Its Patches consumer must transport the same identity
        // certificate before the second explicit materialization.
        let batch_norm = BatchNormLayer::from_scale_bias(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        )
        .unwrap();
        let after_batch_norm = match batch_norm.propagate_patches(&after_relu).unwrap() {
            CrownBounds::Patches(result) => result,
            CrownBounds::Dense(_) => panic!("subnormal BatchNorm consumer crossed to Dense"),
        };
        let input_box = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.5),
            ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.5),
        )
        .unwrap();
        let batch_norm_dense = after_batch_norm.to_dense().unwrap();
        let batch_norm_concrete = batch_norm_dense.concretize_sound(&input_box);
        assert_encloses(
            &batch_norm_concrete,
            "ReLU/BatchNorm/materialize/concretize",
        );
    }
}

#[test]
fn convtranspose2d_stride2_malformed_identity_refuses_without_mutation() {
    let (layer, incoming) = direct_identity_refusal_fixture();

    let mut geometry_mismatch = incoming.clone();
    geometry_mismatch.upper_a.geometry = PatchGeometry::affine((1, 1), (1, 1, 1, 1));
    let before = geometry_mismatch.clone();
    assert!(matches!(
        layer.propagate_patches(&geometry_mismatch),
        Err(NyError::InvalidSpec(_)
            | NyError::ShapeMismatch { .. }
            | NyError::UnsupportedConfiguration(_))
    ));
    assert_patches_input_unchanged(&geometry_mismatch, &before);

    let mut unequal_upper_identity = incoming.clone();
    let (_, out_h, out_w) = unequal_upper_identity.upper_a.output_shape;
    unequal_upper_identity.upper_a.output_shape = (1, out_h - 1, out_w);
    unequal_upper_identity.upper_a.input_shape = (1, out_h - 1, out_w);
    let before = unequal_upper_identity.clone();
    assert!(matches!(
        layer
            .propagate_anchored_identity_with_budget_for_test(&unequal_upper_identity, usize::MAX,),
        Err(NyError::InvalidSpec(_) | NyError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        layer.propagate_patches(&unequal_upper_identity),
        Err(NyError::InvalidSpec(_) | NyError::ShapeMismatch { .. })
    ));
    assert_patches_input_unchanged(&unequal_upper_identity, &before);

    let mut wrong_output = incoming.clone();
    let (_, out_h, out_w) = wrong_output.lower_a.output_shape;
    let wrong_shape = (1, out_h - 1, out_w);
    let wrong_rows = wrong_shape.0 * wrong_shape.1 * wrong_shape.2;
    for side in [&mut wrong_output.lower_a, &mut wrong_output.upper_a] {
        side.output_shape = wrong_shape;
        side.input_shape = wrong_shape;
    }
    wrong_output.row_count = wrong_rows;
    wrong_output.lower_b = Array1::zeros(wrong_rows);
    wrong_output.upper_b = Array1::zeros(wrong_rows);
    let before = wrong_output.clone();
    assert!(matches!(
        layer.propagate_patches(&wrong_output),
        Err(NyError::ShapeMismatch { .. })
    ));
    assert_patches_input_unchanged(&wrong_output, &before);

    let mut bad_rows = incoming.clone();
    bad_rows.row_count -= 1;
    let before = bad_rows.clone();
    assert!(matches!(
        layer.propagate_patches(&bad_rows),
        Err(NyError::ShapeMismatch { .. } | NyError::InvalidSpec(_))
    ));
    assert_patches_input_unchanged(&bad_rows, &before);

    let mut bad_bias = incoming;
    bad_bias.lower_b = Array1::zeros(bad_bias.row_count - 1);
    let before = bad_bias.clone();
    assert!(matches!(
        layer.propagate_patches(&bad_bias),
        Err(NyError::ShapeMismatch { .. } | NyError::InvalidSpec(_))
    ));
    assert_patches_input_unchanged(&bad_bias, &before);
}

fn general_malformed_fixture() -> (ConvTranspose2dLayer, PatchesLinearBounds) {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5, -1.0, 1.5, 0.25]).unwrap();
    let mut layer =
        ConvTranspose2dLayer::new_full(kernel, None, (2, 2), (0, 0), (1, 1), (0, 0)).unwrap();
    layer.set_input_shape(4, 4);
    let (out_h, out_w) = layer.output_size(4, 4).unwrap();
    let shape = [1, 1, 1, 1, 1, 1];
    let make_side = || {
        anchored_general_side(
            vec![0.75],
            &shape,
            (1, 1, 1),
            (1, out_h, out_w),
            vec![0],
            vec![0],
            None,
        )
    };
    (
        layer,
        PatchesLinearBounds {
            row_count: 1,
            lower_a: make_side(),
            lower_b: Array1::from_vec(vec![0.0]),
            upper_a: make_side(),
            upper_b: Array1::from_vec(vec![0.0]),
        },
    )
}

#[test]
fn convtranspose2d_general_unequal_tap_extents_refuse_atomically() {
    let (layer, mut incoming) = general_malformed_fixture();
    incoming.upper_a.patches =
        Some(ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 2, 1]), vec![0.75, -0.25]).unwrap());
    let before = incoming.clone();
    assert!(matches!(
        layer.propagate_patches(&incoming),
        Err(NyError::ShapeMismatch { .. })
    ));
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_general_mixed_geometry_variants_refuse_even_when_origins_match() {
    let (layer, mut incoming) = general_malformed_fixture();
    // For input extent 8 and a 1x1 patch, affine stride 8 has one output at
    // origin zero: the same effective origin as the Anchored lower side. The
    // typed common-geometry invariant still requires an exact variant match.
    incoming.upper_a.geometry = PatchGeometry::affine((8, 8), (0, 0, 0, 0));
    let before = incoming.clone();
    assert!(matches!(
        layer.propagate_patches(&incoming),
        Err(NyError::InvalidSpec(_))
    ));
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_general_excess_anchored_axis_metadata_refuses_atomically() {
    let (layer, mut incoming) = general_malformed_fixture();
    incoming.upper_a.geometry = PatchGeometry::anchored(vec![0, 1], vec![0]).unwrap();
    let before = incoming.clone();
    assert!(matches!(
        layer.propagate_patches(&incoming),
        Err(NyError::ShapeMismatch { .. })
    ));
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_general_sparse_carrier_refuses_atomically() {
    let (layer, mut incoming) = general_malformed_fixture();
    for side in [&mut incoming.lower_a, &mut incoming.upper_a] {
        side.patches = Some(ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![0.75]).unwrap());
        side.unstable_idx = Some(UnstableIdx {
            channels: vec![0],
            heights: vec![0],
            widths: vec![0],
        });
    }
    let before = incoming.clone();
    assert!(matches!(
        layer.propagate_patches(&incoming),
        Err(NyError::UnsupportedConfiguration(_))
    ));
    assert_patches_input_unchanged(&incoming, &before);
}

fn finite_stride1_identity_fixture() -> (ConvTranspose2dLayer, PatchesLinearBounds) {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3, 3]), det_fill(2 * 3 * 3, 0x51_1D_EA, 0.75))
            .unwrap();
    let mut layer = ConvTranspose2dLayer::new_full(
        kernel,
        Some(Array1::from_vec(vec![0.125])),
        (1, 1),
        (0, 0),
        (1, 1),
        (0, 0),
    )
    .unwrap();
    layer.set_input_shape(6, 6);
    let (out_h, out_w) = layer.output_size(6, 6).unwrap();
    assert_eq!((out_h, out_w), (8, 8));
    let output = (1, out_h, out_w);
    (layer, PatchesLinearBounds::identity(output, output))
}

fn finite_stride1_general_fixture(
    explicit_rows: bool,
) -> (ConvTranspose2dLayer, PatchesLinearBounds) {
    let (layer, _) = finite_stride1_identity_fixture();
    let (shape, geometry, output_shape, row_count) = if explicit_rows {
        (
            vec![2, 1, 1, 2, 1, 2, 2],
            PatchGeometry::affine((7, 4), (0, 0, 0, 0)),
            (1, 1, 2),
            2,
        )
    } else {
        // Production-shaped Conv_14 relation over the terminal 8x8 image:
        // stride 2, padding 1, and a 3x3 receptive field yield a 4x4 spec.
        (
            vec![1, 4, 4, 1, 3, 3],
            PatchGeometry::affine((2, 2), (1, 1, 1, 1)),
            (1, 4, 4),
            16,
        )
    };
    let elements: usize = shape.iter().product();
    let make_side = |seed| PatchesData {
        patches: Some(
            ArrayD::from_shape_vec(IxDyn(&shape), det_fill(elements, seed, 0.75)).unwrap(),
        ),
        geometry: geometry.clone(),
        identity: false,
        output_shape,
        input_shape: (1, 8, 8),
        unstable_idx: None,
        coeff_err: None,
    };
    let lower_bias = det_fill(row_count, 0x51_B1_A5, 0.25);
    let upper_bias = lower_bias.iter().map(|value| value + 0.5).collect();
    (
        layer,
        PatchesLinearBounds {
            row_count,
            lower_a: make_side(0x51_6D_10 + explicit_rows as u64),
            lower_b: Array1::from_vec(lower_bias),
            upper_a: make_side(0x51_7D_20 + explicit_rows as u64),
            upper_b: Array1::from_vec(upper_bias),
        },
    )
}

#[test]
fn convtranspose2d_stride1_finite_identity_is_anchored_and_matches_dense() {
    let (layer, incoming) = finite_stride1_identity_fixture();
    let before = incoming.clone();
    let dense = layer
        .propagate_linear(&LinearBounds::identity(incoming.row_count))
        .unwrap()
        .into_owned();

    // Preserve the historical compatibility face: no deadline still uses the
    // equivalent Conv2d and therefore publishes regular Affine geometry.
    let legacy = expect_native_patches(
        layer
            .propagate_patches_engine_and_deadline(&incoming, None, None)
            .expect("no-deadline stride-1 identity route"),
        "no-deadline stride-1 identity route",
    );
    assert!(matches!(&legacy.lower_a.geometry, PatchGeometry::Affine(_)));
    assert!(matches!(&legacy.upper_a.geometry, PatchGeometry::Affine(_)));
    let legacy_dense = legacy.to_dense().unwrap();
    assert_linear_bounds_close(&dense, &legacy_dense, IDENTITY_TOL).unwrap();

    let finite = expect_native_patches(
        layer
            .propagate_patches_engine_and_deadline(
                &incoming,
                None,
                Some(Instant::now() + Duration::from_secs(30)),
            )
            .expect("finite stride-1 identity route"),
        "finite stride-1 identity route",
    );
    assert!(matches!(
        &finite.lower_a.geometry,
        PatchGeometry::Anchored(_)
    ));
    assert!(matches!(
        &finite.upper_a.geometry,
        PatchGeometry::Anchored(_)
    ));
    assert!(finite.lower_a.coeff_err.is_none());
    assert!(finite.upper_a.coeff_err.is_none());
    let finite_dense = finite.to_dense().unwrap();
    assert_linear_bounds_close(&dense, &finite_dense, IDENTITY_TOL).unwrap();
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_stride1_finite_general_6d_7d_matches_dense() {
    for explicit_rows in [false, true] {
        let (layer, incoming) = finite_stride1_general_fixture(explicit_rows);
        let before = incoming.clone();
        let dense_in = incoming.to_dense().unwrap();
        let dense = layer
            .propagate_linear_with_engine(&dense_in, None)
            .unwrap()
            .into_owned();
        let result = layer
            .propagate_patches_engine_and_deadline(
                &incoming,
                None,
                Some(Instant::now() + Duration::from_secs(30)),
            )
            .expect("finite stride-1 materialized route");
        let native = match &result {
            CrownBounds::Patches(native) => native,
            CrownBounds::Dense(_) => panic!(
                "finite stride-1 {} route returned Dense",
                if explicit_rows { "7D" } else { "6D" }
            ),
        };
        assert!(matches!(
            &native.lower_a.geometry,
            PatchGeometry::Anchored(_)
        ));
        assert!(matches!(
            &native.upper_a.geometry,
            PatchGeometry::Anchored(_)
        ));
        let expected_shape = if explicit_rows {
            vec![2, 1, 1, 2, 2, 4, 4]
        } else {
            vec![1, 4, 4, 2, 5, 5]
        };
        assert_eq!(
            native.lower_a.patches.as_ref().unwrap().shape(),
            expected_shape.as_slice()
        );
        assert_eq!(
            native.upper_a.patches.as_ref().unwrap().shape(),
            expected_shape.as_slice()
        );
        assert!(native.lower_a.coeff_err.is_some());
        assert!(native.upper_a.coeff_err.is_some());

        let materialized = result.into_dense().unwrap();
        assert_linear_bounds_close(&dense, &materialized, COMPOSE_TOL).unwrap();
        assert_patches_input_unchanged(&incoming, &before);
    }
}

#[test]
fn convtranspose2d_stride1_expired_deadline_refuses_identity_and_general_atomically() {
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("one millisecond before now must be representable");
    for (context, (layer, incoming)) in [
        ("identity", finite_stride1_identity_fixture()),
        ("general", finite_stride1_general_fixture(true)),
    ] {
        let before = incoming.clone();
        let error = layer
            .propagate_patches_engine_and_deadline(&incoming, None, Some(expired))
            .unwrap_err();
        assert!(
            matches!(&error, NyError::DeadlineExceeded(message)
            if message.contains("ConvTranspose2d Patches backward")
                && message.contains("before dispatch")),
            "unexpected expired {context} result: {error}"
        );
        assert_patches_input_unchanged(&incoming, &before);
    }
}

#[test]
fn convtranspose2d_stride1_general_midwork_deadline_is_atomic() {
    use crate::layers::convolution::conv2d::ConvTransposePatchesDeadlineFailpoint;

    let (layer, incoming) = finite_stride1_general_fixture(true);
    let before = incoming.clone();
    // Reach beyond entry and source/geometry validation so the failure occurs
    // after the finite stride-1 Anchored transaction has begun local work.
    let _failpoint = ConvTransposePatchesDeadlineFailpoint::after_successful_polls(10);
    let error = layer
        .propagate_patches_engine_and_deadline(
            &incoming,
            None,
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .expect_err("deterministic finite stride-1 expiry must discard local work");
    assert!(matches!(error, NyError::DeadlineExceeded(ref message)
        if message.contains("ConvTranspose2d Patches backward")
            && !message.contains("before dispatch")));
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_stride1_finite_zero_budget_refuses_atomically() {
    let _env_lock = ny_test_utils::env::lock_env();
    let _budget = ny_test_utils::env::ScopedEnvVar::set("NY_DENSE_BUDGET_MB", "0");

    for (context, (layer, incoming)) in [
        ("identity", finite_stride1_identity_fixture()),
        ("general", finite_stride1_general_fixture(false)),
    ] {
        let before = incoming.clone();
        let error = layer
            .propagate_patches_engine_and_deadline(
                &incoming,
                None,
                Some(Instant::now() + Duration::from_secs(30)),
            )
            .unwrap_err();
        assert!(
            matches!(&error, NyError::CpuMemoryExceeded { budget_bytes, .. }
                if *budget_bytes == 0),
            "unexpected zero-budget {context} result: {error}"
        );
        assert_patches_input_unchanged(&incoming, &before);
    }
}

#[test]
fn convtranspose2d_stride2_expired_deadline_refuses_without_mutation() {
    let (layer, incoming) = direct_identity_refusal_fixture();
    let before = incoming.clone();
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("one millisecond before now must be representable");
    let error = layer
        .propagate_patches_engine_and_deadline(&incoming, None, Some(expired))
        .expect_err("expired finite authority must not publish a carrier");
    assert!(matches!(error, NyError::DeadlineExceeded(ref message)
        if message.contains("ConvTranspose2d Patches backward")
            && message.contains("before dispatch")));
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_stride2_midwork_deadline_is_atomic() {
    use crate::layers::convolution::conv2d::ConvTransposePatchesDeadlineFailpoint;

    let (layer, incoming) = direct_identity_refusal_fixture();
    let before = incoming.clone();
    let _failpoint = ConvTransposePatchesDeadlineFailpoint::after_successful_polls(7);
    let error = layer
        .propagate_patches_engine_and_deadline(
            &incoming,
            None,
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .expect_err("deterministic mid-work expiry must discard partial output");
    assert!(matches!(error, NyError::DeadlineExceeded(ref message)
        if message.contains("ConvTranspose2d Patches backward")
            && !message.contains("before dispatch")));
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_stride2_live_deadline_matches_no_deadline_bitwise() {
    let (layer, incoming) = direct_identity_refusal_fixture();
    let before = incoming.clone();
    let expected = expect_native_patches(
        layer
            .propagate_patches_engine_and_deadline(&incoming, None, None)
            .expect("no-deadline identity route"),
        "no-deadline identity route",
    );
    let actual = expect_native_patches(
        layer
            .propagate_patches_engine_and_deadline(
                &incoming,
                None,
                Some(Instant::now() + Duration::from_secs(30)),
            )
            .expect("live finite identity route"),
        "live finite identity route",
    );
    assert_patches_bounds_bitwise_equal(&actual, &expected);
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_general_live_deadline_matches_no_deadline_bitwise() {
    let (layer, incoming) = general_malformed_fixture();
    let before = incoming.clone();
    let expected = expect_native_patches(
        layer
            .propagate_patches_engine_and_deadline(&incoming, None, None)
            .expect("no-deadline general route"),
        "no-deadline general route",
    );
    let actual = expect_native_patches(
        layer
            .propagate_patches_engine_and_deadline(
                &incoming,
                None,
                Some(Instant::now() + Duration::from_secs(30)),
            )
            .expect("live finite general route"),
        "live finite general route",
    );
    assert_patches_bounds_bitwise_equal(&actual, &expected);
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_general_helper_is_deadline_cooperative_and_atomic() {
    let (layer, incoming) = general_malformed_fixture();
    let before = incoming.clone();
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap();
    assert!(matches!(
        layer.propagate_anchored_composition_with_deadline(&incoming, Some(expired)),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_patches_input_unchanged(&incoming, &before);

    let live = layer
        .propagate_anchored_composition_with_deadline(
            &incoming,
            Some(Instant::now() + Duration::from_secs(30)),
        )
        .expect("private composition helper must already be cooperatively pollable");
    assert!(matches!(live, CrownBounds::Patches(_)));
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_stride2_zero_budget_refuses_before_allocation() {
    let _env_lock = ny_test_utils::env::lock_env();
    let _budget = ny_test_utils::env::ScopedEnvVar::set("NY_DENSE_BUDGET_MB", "0");
    let (layer, incoming) = direct_identity_refusal_fixture();
    let before = incoming.clone();
    match layer.propagate_patches(&incoming) {
        Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            ..
        }) => {
            assert!(required_bytes > 0);
            assert_eq!(budget_bytes, 0);
        }
        other => panic!("zero-budget direct planner must fail before allocation, got {other:?}"),
    }
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_identity_total_live_receipt_refuses_budget_minus_one_atomically() {
    let (layer, incoming) = direct_identity_refusal_fixture();
    let before = incoming.clone();
    // 8x8 output, 2x2 anchored patch, one input/output channel:
    //   retained source bias pair = 2*64*4 = 512
    //   fresh coefficient pair    = 2*64*4*4 = 2048
    //   fresh bias pair           = 2*64*4 = 512
    //   anchored axes             = (8+8)*16 = 256
    // Total live receipt = 3328 bytes. The one-byte-short authority must be
    // rejected by preflight, before any result allocation or publication.
    let required_bytes = 3_328usize;
    match layer.propagate_anchored_identity_with_budget_for_test(&incoming, required_bytes - 1) {
        Err(NyError::CpuMemoryExceeded {
            required_bytes: actual_required,
            budget_bytes,
            ..
        }) => {
            assert_eq!(actual_required, required_bytes);
            assert_eq!(budget_bytes, required_bytes - 1);
        }
        other => panic!("budget-minus-one identity receipt must refuse, got {other:?}"),
    }
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_identity_subnormal_error_receipt_refuses_budget_minus_one_atomically() {
    let (layer, incoming) = subnormal_identity_fixture(f32::from_bits(1));
    let before = incoming.clone();
    // 7x7 output, 1x1 anchored patch, one input/output channel:
    //   retained source bias pair = 2*49*4 = 392
    //   fresh coefficient pair    = 2*49*1*4 = 392
    //   fresh coefficient errors  = 2*49*4 = 392
    //   fresh bias pair           = 2*49*4 = 392
    //   anchored axes             = (7+7)*16 = 224
    // Total live receipt = 1792 bytes. This preflight boundary proves the new
    // certificates are not an unreceipted allocation.
    let required_bytes = 1_792usize;
    match layer.propagate_anchored_identity_with_budget_for_test(&incoming, required_bytes - 1) {
        Err(NyError::CpuMemoryExceeded {
            required_bytes: actual_required,
            budget_bytes,
            ..
        }) => {
            assert_eq!(actual_required, required_bytes);
            assert_eq!(budget_bytes, required_bytes - 1);
        }
        other => panic!("subnormal budget-minus-one receipt must refuse, got {other:?}"),
    }
    assert_patches_input_unchanged(&incoming, &before);
}

#[test]
fn convtranspose2d_general_zero_budget_refuses_before_allocation() {
    let _env_lock = ny_test_utils::env::lock_env();
    let _budget = ny_test_utils::env::ScopedEnvVar::set("NY_DENSE_BUDGET_MB", "0");
    let (layer, incoming) = general_malformed_fixture();
    let before = incoming.clone();
    assert!(matches!(
        layer.propagate_patches(&incoming),
        Err(NyError::CpuMemoryExceeded {
            budget_bytes: 0,
            ..
        })
    ));
    assert_patches_input_unchanged(&incoming, &before);
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(160) })]

    /// Direct stride>1 identity planner gate.  Random signed kernels, independent
    /// lower/upper incoming biases, asymmetric stride/padding/dilation, and every
    /// legal output-padding residue must agree with both the algebraic inverse-map
    /// oracle and the existing dense CROWN implementation.
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_convtranspose2d_anchored_identity_exact_dense_parity(
        in_c in 1usize..=2,
        out_c in 1usize..=2,
        sh in 2usize..=3,
        sw in 2usize..=3,
        kh in 1usize..=3,
        kw in 1usize..=3,
        dh in 1usize..=2,
        dw in 1usize..=2,
        in_h in 4usize..=5,
        in_w in 4usize..=5,
        raw_ph in 0usize..=15,
        raw_pw in 0usize..=15,
        raw_oph in 0usize..=7,
        raw_opw in 0usize..=7,
        use_bias in proptest::bool::ANY,
        seed in any::<u64>(),
    ) {
        let effective_h = (kh - 1) * dh;
        let effective_w = (kw - 1) * dw;
        let output_padding = (raw_oph % sh, raw_opw % sw);
        // Keep each output non-empty while including padding beyond kernel-1.
        let max_ph = (((in_h - 1) * sh + effective_h + output_padding.0) / 2)
            .min(effective_h + 2);
        let max_pw = (((in_w - 1) * sw + effective_w + output_padding.1) / 2)
            .min(effective_w + 2);
        let padding = (raw_ph % (max_ph + 1), raw_pw % (max_pw + 1));
        let geometry = DirectIdentityGeometry {
            stride: (sh, sw),
            padding,
            dilation: (dh, dw),
            output_padding,
            input: (in_h, in_w),
        };
        let out_h = (in_h - 1) * sh + effective_h + 1 - 2 * padding.0
            + output_padding.0;
        let out_w = (in_w - 1) * sw + effective_w + 1 - 2 * padding.1
            + output_padding.1;
        let out_dim = out_c * out_h * out_w;
        let kernel = make_convt_kernel(in_c, out_c, kh, kw, seed);
        let bias = use_bias.then(|| make_bias(out_c, seed ^ 0xB1A5));
        let lower_b = det_fill(out_dim, seed ^ 0x10_0B, 0.75);
        let gaps = det_fill(out_dim, seed ^ 0x20_0B, 0.5);
        let upper_b = lower_b
            .iter()
            .zip(gaps)
            .map(|(&lower, gap)| lower + gap.abs() + 0.01)
            .collect();

        assert_small_native_identity_case(
            kernel,
            bias,
            geometry,
            lower_b,
            upper_b,
        );
    }
}

/// Unit check of the reduction core: the equivalent Conv2d (flip+swap kernel,
/// Cp = kh-1-pad) dense backward must equal the ConvTranspose dense backward
/// exactly (the identity case is bit-exact, single-tap coefficients).
#[test]
fn convtranspose2d_reduction_matches_dense_identity_fixed() {
    use crate::layers::convolution::conv2d::Conv2dLayer;
    for &(in_c, out_c, kh, kw, ph, pw, in_h, in_w) in &[
        (
            1usize, 1usize, 2usize, 2usize, 0usize, 0usize, 4usize, 4usize,
        ),
        (2, 3, 3, 3, 1, 1, 5, 5),
        (3, 2, 3, 2, 0, 1, 6, 4),
        (2, 2, 1, 3, 0, 1, 5, 5),
    ] {
        let kernel = make_convt_kernel(in_c, out_c, kh, kw, 0xDEAD_BEEF ^ (in_c as u64));
        let convt = ConvTranspose2dLayer::with_input_shape(
            kernel.clone(),
            None,
            (1, 1),
            (ph, pw),
            in_h,
            in_w,
        )
        .unwrap();
        let (out_h, out_w) = convt.output_size(in_h, in_w).unwrap();
        let out_dim = out_c * out_h * out_w;

        // Dense ConvTranspose.
        let dense_ct = convt
            .propagate_linear(&LinearBounds::identity(out_dim))
            .unwrap()
            .into_owned();

        // Equivalent Conv2d: Kc[oc,ic,ki,kj] = W[ic,oc,kh-1-ki,kw-1-kj], Cp=(kh-1-ph,kw-1-pw).
        let mut kc = ArrayD::<f32>::zeros(IxDyn(&[out_c, in_c, kh, kw]));
        for oc in 0..out_c {
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        kc[[oc, ic, ki, kj]] = kernel[[ic, oc, kh - 1 - ki, kw - 1 - kj]];
                    }
                }
            }
        }
        let conv2d =
            Conv2dLayer::with_input_shape(kc, None, (1, 1), (kh - 1 - ph, kw - 1 - pw), in_h, in_w)
                .unwrap();
        let dense_c2 = conv2d
            .propagate_linear(&LinearBounds::identity(out_dim))
            .unwrap()
            .into_owned();

        assert_eq!(dense_ct.lower_a().shape(), dense_c2.lower_a().shape());
        for (a, b) in dense_ct.lower_a().iter().zip(dense_c2.lower_a().iter()) {
            assert!(
                (a - b).abs() <= 1e-5 * (1.0 + a.abs()),
                "reduction coeff mismatch: {a} vs {b}"
            );
        }
    }
}

// =====================================================================
// STAGE 2b — phase-partition of a STRIDE-s ConvTranspose2d CROWN backward.
//
// A forward stride-s ConvTranspose *upsamples* (insert s-1 zeros between inputs,
// then a stride-1 conv), so its CROWN backward (the adjoint) *downsamples* the
// coefficient grid: output pixels (oh, ow) -> input pixels (oh/s, ow/s). The
// PHASE-PARTITION splits the output/spec grid into the s^2 residue classes
// (a, b) = (oh mod s, ow mod s); within each class the backward is a STRIDE-1
// ConvTranspose backward on the DECIMATED sub-grid with the per-phase kernel
// slice `W[:, :, (ph+a) % s :: s, (pw+b) % s :: s]` and per-phase padding
// `(⌊(ph+a)/s⌋, ⌊(pw+b)/s⌋)`. Each phase therefore reduces (STAGE 2a) to a plain
// Conv2d PATCHES backward with the flip+swap kernel, and the s^2 phase results
// SUM (over disjoint spec rows) to the dense ConvTranspose backward.
//
// The tests below VALIDATE that reduction end-to-end against the *dense*
// `propagate_linear_with_engine` bound, bit-exact within 1e-5 (coeffs), for
// stride-2 AND stride-3, identity AND non-identity incoming — routing every
// phase through the PROVEN `Conv2dLayer::propagate_patches` machinery.
//
// The reduction is a valid *computation*. The historical affine-only carrier
// could not reassemble it, which exposed two obligations:
//
//   (1) FRACTIONAL-STRIDE OBSTRUCTION. `PatchesData` positions each spec row's
//       receptive field at `input_pos = spec_pos * stride + tap - pad` with an
//       INTEGER `stride >= 1` (an *upsampling* map — every existing producer,
//       Conv2d/pool backward, upsamples). The ConvTranspose backward needs
//       `input_pos = ⌊(oh + ph)/s⌋ + tap` — a *downsampling* (floor) map, which
//       is a step function of the spec index and is NOT expressible as
//       `spec_pos * stride + ...` for any integer stride. `PatchesData` now has
//       an exact separable `Anchored` geometry capable of storing this map, and
//       generic dense scatter/materialization supports it. The direct identity
//       planner constructs that mapping. The stage-4 directed-f64 planner also
//       composes materialized 6D/7D carriers on the same inverse grid with a
//       certified per-row error; unsupported consumers retain typed refusal.
//
//   (2) PADDING BOUNDARY. Even the per-phase equiv-Conv2d reduction only
//       size-matches the phase sub-grid when `padding == 0`; nonzero ConvTranspose
//       padding shifts the phase boundaries so the equiv Conv2d over/under-produces
//       rows (empirically: ~half of padded configs). The direct inverse-map
//       planner handles these cases rather than treating the old
//       equivalent-Conv2d reduction as an admission proof.
//
// Production therefore admits strict full identity bounds and nonsparse
// materialized 6D/7D bounds through direct anchored inverse-map planners
// (including nonzero padding, dilation, and output padding). No-deadline
// composition retains its historical stored-size crossover. A finite request
// may continue beyond that optimization crossover only after the Anchored
// planner's truthful total-live resident admission; expiry is terminal and
// never launches a same-relation Dense retry. Sparse, mixed, and malformed
// carriers retain their typed refusal or established checked fallback. These
// phase tests remain an independent zero-padding oracle for the stride-2/3
// coefficient math rather than the implementation under test.
// =====================================================================

/// Reassemble the stride-s ConvTranspose IDENTITY CROWN backward from its s^2
/// phase reductions, routing each phase through `Conv2dLayer::propagate_patches`.
/// Returns `(reconstructed lower_a [out_dim x in_dim], covered)` where
/// `covered[out_flat]` counts how many phases wrote that output row (must be
/// exactly 1 — the disjoint+complete partition). These callers use padding 0,
/// so a phase-size mismatch is a failed test invariant, not a skipped corner.
fn phase_reduce_identity(
    convt: &ConvTranspose2dLayer,
    kernel: &ArrayD<f32>,
    s: usize,
    in_h: usize,
    in_w: usize,
) -> Result<(ndarray::Array2<f32>, Vec<u32>), NyError> {
    use crate::layers::convolution::conv2d::Conv2dLayer;
    let in_c = kernel.shape()[0];
    let out_c = kernel.shape()[1];
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];
    let (out_h, out_w) = convt.output_size(in_h, in_w)?;
    let in_dim = in_c * in_h * in_w;
    let out_dim = out_c * out_h * out_w;
    let mut recon = ndarray::Array2::<f32>::zeros((out_dim, in_dim));
    let mut covered = vec![0u32; out_dim];

    for a in 0..s {
        for b in 0..s {
            let hs: Vec<usize> = (a..out_h).step_by(s).collect();
            let ws: Vec<usize> = (b..out_w).step_by(s).collect();
            if hs.is_empty() || ws.is_empty() {
                continue;
            }
            let (out_h_a, out_w_b) = (hs.len(), ws.len());
            // Every output pixel of this phase is "covered" (it belongs to
            // exactly this residue class) even if the phase has no kernel taps.
            for oc in 0..out_c {
                for &oh in &hs {
                    for &ow in &ws {
                        covered[(oc * out_h + oh) * out_w + ow] += 1;
                    }
                }
            }
            // padding == 0 in these tests => t0 = a (or b), c_h = c_w = 0.
            let kis: Vec<usize> = (a..kh).step_by(s).collect();
            let kjs: Vec<usize> = (b..kw).step_by(s).collect();
            if kis.is_empty() || kjs.is_empty() {
                continue; // no taps for this phase -> zero coefficients (already covered)
            }
            let (kh_a, kw_b) = (kis.len(), kjs.len());
            // equiv Conv2d kernel (flip+swap of the per-phase slice), STAGE 2a form:
            //   Kc[oc, ic, u, v] = W[ic, oc, kis[kh_a-1-u], kjs[kw_b-1-v]]
            let mut kc = ArrayD::<f32>::zeros(IxDyn(&[out_c, in_c, kh_a, kw_b]));
            for oc in 0..out_c {
                for ic in 0..in_c {
                    for u in 0..kh_a {
                        for v in 0..kw_b {
                            kc[[oc, ic, u, v]] =
                                kernel[[ic, oc, kis[kh_a - 1 - u], kjs[kw_b - 1 - v]]];
                        }
                    }
                }
            }
            // per-phase equiv padding Cp = (kh_a-1-c_h, kw_b-1-c_w), c_* = 0 (pad 0).
            let equiv =
                Conv2dLayer::with_input_shape(kc, None, (1, 1), (kh_a - 1, kw_b - 1), in_h, in_w)?;
            let (e_oh, e_ow) = equiv.output_size(in_h, in_w)?;
            if (e_oh, e_ow) != (out_h_a, out_w_b) {
                return Err(NyError::InternalError(format!(
                    "padding-zero phase reduction size mismatch: equivalent Conv2d={e_oh}x{e_ow}, phase={out_h_a}x{out_w_b}"
                )));
            }
            let id = PatchesLinearBounds::identity((out_c, e_oh, e_ow), (out_c, e_oh, e_ow));
            let p_ab = match equiv.propagate_patches(&id)? {
                CrownBounds::Patches(pb) => pb.to_dense()?,
                CrownBounds::Dense(_) => {
                    return Err(NyError::InternalError(
                        "phase reduction unexpectedly crossed to Dense".to_string(),
                    ));
                }
            };
            let pa = p_ab.lower_a();
            for oc in 0..out_c {
                for (ii, &oh) in hs.iter().enumerate() {
                    for (jj, &ow) in ws.iter().enumerate() {
                        let full = (oc * out_h + oh) * out_w + ow;
                        let sub = (oc * out_h_a + ii) * out_w_b + jj;
                        for j in 0..in_dim {
                            recon[[full, j]] += pa[[sub, j]];
                        }
                    }
                }
            }
        }
    }
    Ok((recon, covered))
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// STAGE 2b gate — IDENTITY incoming, stride-2 AND stride-3, padding 0. The
    /// s^2 phase reductions (each through the proven Conv2d patches path) must:
    ///   - PARTITION the output/spec grid (every output row written exactly once);
    ///   - reassemble bit-exact to the dense ConvTranspose `propagate_linear`
    ///     coefficient (within 1e-5 relative).
    // Hang sentinel with scheduler headroom for the 300-case property on a
    // shared builder; equivalence assertions, not elapsed time, are the gate.
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_convtranspose2d_phase_partition_identity(
        s in 2usize..=3,
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        kh in 1usize..=4,
        kw in 1usize..=4,
        // Per-phase kernels are at most 2x2; 3x3 keeps every reduction in Patches.
        in_h in 3usize..=6,
        in_w in 3usize..=6,
        seed in any::<u64>(),
    ) {
        let kernel = make_convt_kernel(in_c, out_c, kh, kw, seed);
        let convt = ConvTranspose2dLayer::with_input_shape(
            kernel.clone(), None, (s, s), (0, 0), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("ConvTranspose2d creation failed: {e}")))?;
        let (out_h, out_w) = convt.output_size(in_h, in_w)
            .map_err(|e| TestCaseError::fail(format!("output_size: {e}")))?;
        let out_dim = out_c * out_h * out_w;

        // Ground truth: dense ConvTranspose CROWN backward on identity.
        let dense = convt.propagate_linear(&LinearBounds::identity(out_dim))
            .map_err(|e| TestCaseError::fail(format!("dense propagate_linear failed: {e}")))?
            .into_owned();

        let (recon, covered) = phase_reduce_identity(&convt, &kernel, s, in_h, in_w)
            .map_err(|e| TestCaseError::fail(format!("phase reduce failed: {e}")))?;

        // Disjoint + complete partition of the spec rows.
        for (i, &c) in covered.iter().enumerate() {
            prop_assert_eq!(c, 1u32, "spec row {} covered {} times (phases must partition)", i, c);
        }

        // Bit-exact reassembly vs dense.
        let m = dense.lower_a();
        prop_assert_eq!(m.shape(), recon.shape(), "reconstructed shape");
        for ((idx, &d), &p) in m.indexed_iter().zip(recon.iter()) {
            let diff = (d - p).abs();
            let scale = d.abs().max(p.abs()).max(1.0);
            prop_assert!(
                diff <= IDENTITY_TOL * scale,
                "phase-partition coeff mismatch at {:?}: dense={}, phase={}, diff={}",
                idx, d, p, diff
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// STAGE 2b gate — NON-identity incoming, stride-2 AND stride-3, padding 0.
    /// The dense ConvTranspose backward of an arbitrary incoming `A` equals
    /// `A @ M`, where `M` is the (phase-reassembled) identity backward validated
    /// above. Assert the two agree within 1e-4 relative on the coefficients —
    /// i.e. the phase reduction commutes with an arbitrary non-identity
    /// composition, exactly the dense path's result.
    #[ntest::timeout(20000)]
    #[test]
    fn proptest_convtranspose2d_phase_partition_nonidentity(
        s in 2usize..=3,
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        kh in 1usize..=4,
        kw in 1usize..=4,
        in_h in 3usize..=5,
        in_w in 3usize..=5,
        rows in 1usize..=4,
        seed in any::<u64>(),
    ) {
        let kernel = make_convt_kernel(in_c, out_c, kh, kw, seed);
        let convt = ConvTranspose2dLayer::with_input_shape(
            kernel.clone(), None, (s, s), (0, 0), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("ConvTranspose2d creation failed: {e}")))?;
        let (out_h, out_w) = convt.output_size(in_h, in_w)
            .map_err(|e| TestCaseError::fail(format!("output_size: {e}")))?;
        let out_dim = out_c * out_h * out_w;

        // M = phase-reassembled identity backward (== dense M, pinned by the
        // identity test); use it as the exact reference operator matrix.
        let (recon, _covered) = phase_reduce_identity(&convt, &kernel, s, in_h, in_w)
            .map_err(|e| TestCaseError::fail(format!("phase reduce failed: {e}")))?;

        // Arbitrary non-identity incoming A (lower/upper) over the ConvT output.
        let a_lower = ndarray::Array2::from_shape_vec(
            (rows, out_dim), det_fill(rows * out_dim, seed ^ 0xF00D, 1.5),
        ).map_err(|e| TestCaseError::fail(format!("a_lower shape: {e}")))?;
        let a_upper = ndarray::Array2::from_shape_vec(
            (rows, out_dim), det_fill(rows * out_dim, seed ^ 0x1DEA, 1.5),
        ).map_err(|e| TestCaseError::fail(format!("a_upper shape: {e}")))?;
        let b_lower = Array1::from_vec(det_fill(rows, seed ^ 0x0B01, 1.0));
        let b_upper = Array1::from_vec(det_fill(rows, seed ^ 0x0B02, 1.0));
        let incoming = LinearBounds::new(a_lower.clone(), b_lower, a_upper.clone(), b_upper)
            .map_err(|e| TestCaseError::fail(format!("incoming LinearBounds: {e}")))?;

        let dense_out = convt.propagate_linear_with_engine(&incoming, None)
            .map_err(|e| TestCaseError::fail(format!("dense propagate failed: {e}")))?
            .into_owned();

        // Reference = A @ M for each side.
        let expect_lower = a_lower.dot(&recon);
        let expect_upper = a_upper.dot(&recon);
        let cmp = |dense_a: &ndarray::Array2<f32>, expect: &ndarray::Array2<f32>, name: &str|
            -> Result<(), TestCaseError> {
            prop_assert_eq!(dense_a.shape(), expect.shape(), "{} shape", name);
            for ((idx, &d), &e) in dense_a.indexed_iter().zip(expect.iter()) {
                let diff = (d - e).abs();
                let scale = d.abs().max(e.abs()).max(1.0);
                prop_assert!(
                    diff <= COMPOSE_TOL * scale,
                    "{} phase-compose mismatch at {:?}: dense={}, A@M={}, diff={}",
                    name, idx, d, e, diff
                );
            }
            Ok(())
        };
        cmp(dense_out.lower_a(), &expect_lower, "lower_a")?;
        cmp(dense_out.upper_a(), &expect_upper, "upper_a")?;
    }
}
