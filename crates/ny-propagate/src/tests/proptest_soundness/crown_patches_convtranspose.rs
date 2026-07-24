// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bit-equivalence + soundness proptests for the STRIDE-1 patches-native
//! ConvTranspose2d CROWN backward (LEVER 2 stage 2a).
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
//!   - every non-stride-1 / unsupported corner returns
//!     `UnsupportedConfiguration` so the dispatcher falls back to the exact
//!     dense path (proving no silent wrongness).
//!
//! Design: stage-2a reduction of the stride-1 ConvTranspose backward to the
//! proven Conv2d patches path (flip+swap kernel, adjusted padding, same bias).

use crate::bounds::patches::{CrownBounds, PatchesData, PatchesLinearBounds};
use crate::layers::activations::ReLULayer;
use crate::layers::common::{BoundPropagation, PatchesPropagation};
use crate::layers::convolution::conv2d::ConvTranspose2dLayer;
use crate::LinearBounds;
use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::NyError;
use proptest::prelude::*;

/// Tolerance for the identity (bit-exact) Patches-vs-Dense comparison. Identity
/// incoming yields per-cell single-kernel-tap coefficients on both paths, so
/// they agree bit-for-bit; `1e-5` relative leaves ULP headroom.
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
        kh in 1usize..=3,
        kw in 1usize..=3,
        in_h in 3usize..=8,
        in_w in 3usize..=8,
        pad_h in 0usize..=1,
        pad_w in 0usize..=1,
        use_bias in proptest::bool::ANY,
        seed in any::<u64>(),
    ) {
        // Stage-2a supports padding <= kernel-1 per dim; otherwise the layer is
        // valid but routes to the dense fallback (covered by the fallback test).
        prop_assume!(pad_h < kh && pad_w < kw);
        // Valid ConvTranspose output size (stride 1, dilation 1, output_padding 0).
        let out_h = (in_h + kh).checked_sub(1 + 2 * pad_h);
        let out_w = (in_w + kw).checked_sub(1 + 2 * pad_w);
        prop_assume!(out_h.is_some() && out_w.is_some());
        let (out_h, out_w) = (out_h.unwrap(), out_w.unwrap());
        prop_assume!(out_h >= 1 && out_w >= 1);

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
        let patches_out = match convt.propagate_patches(&patches_id) {
            Ok(r) => r,
            Err(NyError::UnsupportedConfiguration(_)) => return Ok(()), // dense fallback corner
            Err(e) => return Err(TestCaseError::fail(format!("patches failed: {e}"))),
        };
        let patches_dense = patches_out.into_dense()
            .map_err(|e| TestCaseError::fail(format!("patches to_dense failed: {e}")))?;

        assert_linear_bounds_close(&dense_out, &patches_dense, IDENTITY_TOL)?;
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
        kh in 1usize..=3,
        kw in 1usize..=3,
        in_h in 4usize..=8,
        in_w in 4usize..=8,
        pad_h in 0usize..=1,
        pad_w in 0usize..=1,
        prev_k in 1usize..=2,
        use_bias in proptest::bool::ANY,
        seed in any::<u64>(),
    ) {
        prop_assume!(pad_h < kh && pad_w < kw);
        let out_h = (in_h + kh).checked_sub(1 + 2 * pad_h);
        let out_w = (in_w + kw).checked_sub(1 + 2 * pad_w);
        prop_assume!(out_h.is_some() && out_w.is_some());
        let (out_h, out_w) = (out_h.unwrap(), out_w.unwrap());
        // The incoming (prev_k x prev_k, stride 1, pad 0) receptive field over the
        // ConvTranspose OUTPUT grid yields spec spatial dims = unfold output size.
        prop_assume!(out_h >= prev_k && out_w >= prev_k);
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
                stride: (1, 1),
                padding: (0, 0, 0, 0),
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

        let patches_out = match convt.propagate_patches(&incoming) {
            Ok(r) => r,
            Err(NyError::UnsupportedConfiguration(_)) => return Ok(()),
            Err(e) => return Err(TestCaseError::fail(format!("patches failed: {e}"))),
        };
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
        kh in 1usize..=3,
        kw in 1usize..=3,
        in_h in 3usize..=6,
        in_w in 3usize..=6,
        pad_h in 0usize..=1,
        pad_w in 0usize..=1,
        seed in any::<u64>(),
    ) {
        prop_assume!(pad_h < kh && pad_w < kw);
        let out_h = (in_h + kh).checked_sub(1 + 2 * pad_h);
        let out_w = (in_w + kw).checked_sub(1 + 2 * pad_w);
        prop_assume!(out_h.is_some() && out_w.is_some());
        let (out_h, out_w) = (out_h.unwrap(), out_w.unwrap());
        prop_assume!(out_h >= 1 && out_w >= 1);

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
        let input_bt = ny_tensor::BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&in_shape), lower_vals.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&in_shape), upper_vals.clone()).unwrap(),
        ).map_err(|e| TestCaseError::fail(format!("BoundedTensor failed: {e}")))?;
        let flat_input = input_bt.flatten();

        let identity_lb = LinearBounds::identity(out_dim);
        let dense_out = convt.propagate_linear(&identity_lb)
            .map_err(|e| TestCaseError::fail(format!("dense failed: {e}")))?
            .into_owned();
        let dense_bounds = dense_out.concretize_sound(&flat_input);

        let patches_out = match convt.propagate_patches(
            &PatchesLinearBounds::identity((out_c, out_h, out_w), (out_c, out_h, out_w)),
        ) {
            Ok(r) => r,
            Err(NyError::UnsupportedConfiguration(_)) => return Ok(()),
            Err(e) => return Err(TestCaseError::fail(format!("patches failed: {e}"))),
        };
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
            let sample_pt = ny_tensor::BoundedTensor::new(sample_nd.clone(), sample_nd).unwrap();
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
        kh in 1usize..=3,
        kw in 1usize..=3,
        in_h in 3usize..=5,
        in_w in 3usize..=5,
        pad_h in 0usize..=1,
        pad_w in 0usize..=1,
        seed in any::<u64>(),
    ) {
        prop_assume!(pad_h < kh && pad_w < kw);
        let out_h = (in_h + kh).checked_sub(1 + 2 * pad_h);
        let out_w = (in_w + kw).checked_sub(1 + 2 * pad_w);
        prop_assume!(out_h.is_some() && out_w.is_some());
        let (out_h, out_w) = (out_h.unwrap(), out_w.unwrap());
        prop_assume!(out_h >= 1 && out_w >= 1);

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
        let input_bt = ny_tensor::BoundedTensor::new(
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
                match convt.propagate_patches(pb) {
                    Ok(r) => r.into_dense()
                        .map_err(|e| TestCaseError::fail(format!("patches to_dense failed: {e}")))?
                        .concretize_sound(&flat_input),
                    Err(NyError::UnsupportedConfiguration(_)) => return Ok(()),
                    Err(e) => return Err(TestCaseError::fail(format!("patches ConvT failed: {e}"))),
                }
            }
            CrownBounds::Dense(_) => return Ok(()), // ReLU may terminate to dense
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
            let sample_pt = ny_tensor::BoundedTensor::new(sample_nd.clone(), sample_nd).unwrap();
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
// Corner fallbacks: every unsupported configuration returns
// UnsupportedConfiguration so the dispatcher falls back to the SOUND dense
// path (no silent wrongness).
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

#[test]
fn convtranspose2d_stride2_takes_dense_fallback() {
    // kernel (in_c=1, out_c=1, 2, 2), stride 2 -> must NOT be handled by patches.
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let (convt, id) = convt_identity_for(kernel, (2, 2), (0, 0), (1, 1), (0, 0), 4, 4);
    match convt.propagate_patches(&id) {
        Err(NyError::UnsupportedConfiguration(_)) => {}
        other => panic!("stride-2 must return UnsupportedConfiguration, got {other:?}"),
    }
    // And the DENSE path succeeds (the sound fallback the dispatcher would take).
    let (out_h, out_w) = convt.output_size(4, 4).unwrap();
    let out_dim = convt.out_channels() * out_h * out_w;
    let identity = LinearBounds::identity(out_dim);
    let dense = convt.propagate_linear(&identity);
    assert!(dense.is_ok(), "dense fallback must succeed for stride-2");
}

#[test]
fn convtranspose2d_dilation_takes_dense_fallback() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let (convt, id) = convt_identity_for(kernel, (1, 1), (0, 0), (2, 2), (0, 0), 4, 4);
    match convt.propagate_patches(&id) {
        Err(NyError::UnsupportedConfiguration(_)) => {}
        other => panic!("dilation!=1 must return UnsupportedConfiguration, got {other:?}"),
    }
}

#[test]
fn convtranspose2d_output_padding_takes_dense_fallback() {
    // output_padding requires stride>2; with stride 2, op 1 is valid and must
    // also fall back (stride!=1 already forces it, plus output_padding!=0).
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let (convt, id) = convt_identity_for(kernel, (2, 2), (0, 0), (1, 1), (1, 1), 4, 4);
    match convt.propagate_patches(&id) {
        Err(NyError::UnsupportedConfiguration(_)) => {}
        other => panic!("output_padding!=0 must return UnsupportedConfiguration, got {other:?}"),
    }
}

#[test]
fn convtranspose2d_padding_gt_kernel_takes_dense_fallback() {
    // kh=1, pad_h=1 => Cp = kh-1-pad = -1 (not representable) => dense fallback.
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![2.0]).unwrap();
    let (convt, id) = convt_identity_for(kernel, (1, 1), (1, 1), (1, 1), (0, 0), 4, 4);
    match convt.propagate_patches(&id) {
        Err(NyError::UnsupportedConfiguration(_)) => {}
        other => panic!("padding>kernel-1 must return UnsupportedConfiguration, got {other:?}"),
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
// WHY STRIDE>1 STAYS ON THE SOUND DENSE FALLBACK IN PRODUCTION. The reduction
// is a valid *computation*, but its result cannot be reassembled into the
// memory-light `PatchesData` representation, for two independent reasons this
// analysis pinned down:
//
//   (1) FRACTIONAL-STRIDE OBSTRUCTION. `PatchesData` positions each spec row's
//       receptive field at `input_pos = spec_pos * stride + tap - pad` with an
//       INTEGER `stride >= 1` (an *upsampling* map — every existing producer,
//       Conv2d/pool backward, upsamples). The ConvTranspose backward needs
//       `input_pos = ⌊(oh + ph)/s⌋ + tap` — a *downsampling* (floor) map, which
//       is a step function of the spec index and is NOT expressible as
//       `spec_pos * stride + ...` for any integer stride. A single `PatchesData`
//       (6D or 7D) therefore cannot hold the reassembled coefficient; the phases
//       must be re-interleaved through a new multi-phase geometry that would have
//       to be threaded — soundly — through EVERY geometry-aware patches consumer
//       (`to_dense`, all element-wise activation backwards which look up
//       per-input-neuron slopes via `ih = oh*stride + ki - pad`, pooling, the
//       Conv2d/ConvTranspose compose, the coeff_err scatter, merge). Any consumer
//       that saw the new geometry without handling it would be SILENTLY unsound.
//
//   (2) PADDING BOUNDARY. Even the per-phase equiv-Conv2d reduction only
//       size-matches the phase sub-grid when `padding == 0`; nonzero ConvTranspose
//       padding shifts the phase boundaries so the equiv Conv2d over/under-produces
//       rows (empirically: ~half of padded configs). Those reduce cleanly only
//       via the raw forward-conv form, which the (transposed-conv) Conv2d patches
//       path does not implement.
//
// So per the soundness mandate ("a correct dense-fallback-only result is better
// than a wrong patches bound") the production `propagate_patches_engine` keeps
// returning `UnsupportedConfiguration` for stride>1 -> the exact dense CROWN
// backward. These tests lock the phase-partition MATH (the validated foundation
// for any future memory-light multi-phase representation) and the zero-padding
// scope where the per-phase reduction is bit-exact.
// =====================================================================

/// Reassemble the stride-s ConvTranspose IDENTITY CROWN backward from its s^2
/// phase reductions, routing each phase through `Conv2dLayer::propagate_patches`.
/// Returns `None` if any phase's equiv Conv2d output size does not match its
/// phase sub-grid (the nonzero-padding boundary corner; never happens for
/// `padding == 0`). Otherwise returns `(reconstructed lower_a [out_dim x in_dim],
/// covered)` where `covered[out_flat]` counts how many phases wrote that output
/// row (must be exactly 1 — the disjoint+complete partition).
fn phase_reduce_identity(
    convt: &ConvTranspose2dLayer,
    kernel: &ArrayD<f32>,
    s: usize,
    in_h: usize,
    in_w: usize,
) -> Result<Option<(ndarray::Array2<f32>, Vec<u32>)>, NyError> {
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
                return Ok(None); // padding boundary corner -> not cleanly reducible
            }
            let id = PatchesLinearBounds::identity((out_c, e_oh, e_ow), (out_c, e_oh, e_ow));
            let p_ab = equiv.propagate_patches(&id)?.into_dense()?;
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
    Ok(Some((recon, covered)))
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
        in_h in 2usize..=6,
        in_w in 2usize..=6,
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

        let reduced = phase_reduce_identity(&convt, &kernel, s, in_h, in_w)
            .map_err(|e| TestCaseError::fail(format!("phase reduce failed: {e}")))?;
        // padding == 0 always size-matches; guard defensively.
        let Some((recon, covered)) = reduced else { return Ok(()); };

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
        in_h in 2usize..=5,
        in_w in 2usize..=5,
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
        let reduced = phase_reduce_identity(&convt, &kernel, s, in_h, in_w)
            .map_err(|e| TestCaseError::fail(format!("phase reduce failed: {e}")))?;
        let Some((recon, _covered)) = reduced else { return Ok(()); };

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
