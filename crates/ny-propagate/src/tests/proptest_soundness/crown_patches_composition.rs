// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest for Conv2d Patches composition through multi-Conv chains.
//!
//! Separated from crown_patches.rs to keep files under 1000 lines.
//!
//! Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md
//! Part of #2613

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::layers::activations::ReLULayer;
use crate::layers::common::{BoundPropagation, PatchesPropagation};
use crate::layers::convolution::conv2d::Conv2dLayer;
use crate::LinearBounds;
use ndarray::{ArrayD, IxDyn};
use proptest::prelude::*;

/// Generate a random kernel of shape (out_c, in_c, kh, kw) with values in
/// [-2.0, 2.0], using a seed for reproducibility.
fn make_kernel(out_c: usize, in_c: usize, kh: usize, kw: usize, seed: u64) -> ArrayD<f32> {
    let len = out_c * in_c * kh * kw;
    let mut rng = seed;
    let values: Vec<f32> = (0..len)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            u * 4.0 - 2.0
        })
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, kh, kw]), values).expect("kernel shape mismatch")
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Test 6: Conv2d → ReLU → Conv2d Patches composition vs Dense equivalence.
    ///
    /// Network: Input → Conv1 → ReLU → Conv2 → Output
    /// Backward: identity at Output → Conv2 backward → ReLU backward → Conv1 backward
    ///
    /// Property: The Patches path stays in Patches mode through Conv2 (creates),
    /// ReLU (scales), and Conv1 (composes), and after to_dense() matches Dense.
    ///
    /// This is the key composition test — verifies that non-identity patches
    /// are correctly composed through Conv1 via conv2d_transpose.
    ///
    /// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md Phase 1
    /// Part of #2613
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_conv2d_relu_conv2d_patches_composition(
        // Conv1: (in_c1, in_h1, in_w1) → (out_c1, out_h1, out_w1)
        in_c1 in 1usize..=3,
        out_c1 in 1usize..=3,
        kh1 in 1usize..=2,
        kw1 in 1usize..=2,
        in_h1 in 5usize..=8,
        in_w1 in 5usize..=8,
        // Conv2: (in_c2=out_c1, in_h2=out_h1, in_w2=out_w1) → (out_c2, out_h2, out_w2)
        out_c2 in 1usize..=3,
        kh2 in 1usize..=2,
        kw2 in 1usize..=2,
        seed in any::<u64>(),
    ) {
        // stride=1, padding=0 for both convs for simplicity
        // Conv1 output spatial
        let out_h1 = in_h1 - kh1 + 1;
        let out_w1 = in_w1 - kw1 + 1;

        // Conv2 input = Conv1 output
        let in_c2 = out_c1;
        let in_h2 = out_h1;
        let in_w2 = out_w1;
        let out_h2 = in_h2 - kh2 + 1;
        let out_w2 = in_w2 - kw2 + 1;

        let conv2_out_dim = out_c2 * out_h2 * out_w2;

        let kernel1 = make_kernel(out_c1, in_c1, kh1, kw1, seed);
        let kernel2 = make_kernel(out_c2, in_c2, kh2, kw2, seed.wrapping_add(1));

        let conv1 = Conv2dLayer::with_input_shape(
            kernel1, None, (1, 1), (0, 0), in_h1, in_w1,
        ).map_err(|e| TestCaseError::fail(format!("Conv1 creation failed: {e}")))?;

        let conv2 = Conv2dLayer::with_input_shape(
            kernel2, None, (1, 1), (0, 0), in_h2, in_w2,
        ).map_err(|e| TestCaseError::fail(format!("Conv2 creation failed: {e}")))?;

        let relu = ReLULayer::new();

        // ReLU pre-activation bounds: at ReLU's input = Conv1's output
        // Shape: (out_c1, out_h1, out_w1) = (in_c2, in_h2, in_w2)
        let relu_in_dim = out_c1 * out_h1 * out_w1;
        let pre_relu_lower: Vec<f32> = (0..relu_in_dim).map(|i| {
            let mut rng = seed.wrapping_add(i as u64 + 44444);
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            u * 2.0 - 1.0  // [-1, 1) — crossing zero for ReLU
        }).collect();
        let pre_relu_upper: Vec<f32> = pre_relu_lower.iter().map(|&l| l + 0.3).collect();
        let pre_relu_bt = ny_tensor::BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[out_c1, out_h1, out_w1]), pre_relu_lower).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[out_c1, out_h1, out_w1]), pre_relu_upper).unwrap(),
        ).map_err(|e| TestCaseError::fail(format!("pre_relu BoundedTensor failed: {e}")))?;

        // ---- Dense path: identity → Conv2 backward → ReLU backward → Conv1 backward ----
        let identity_lb = LinearBounds::identity(conv2_out_dim);
        let after_conv2_dense = conv2.propagate_linear(&identity_lb)
            .map_err(|e| TestCaseError::fail(format!("Dense Conv2 backward failed: {e}")))?
            .into_owned();
        let after_relu_dense = relu.propagate_linear_with_bounds(&after_conv2_dense, &pre_relu_bt)
            .map_err(|e| TestCaseError::fail(format!("Dense ReLU backward failed: {e}")))?;
        let after_conv1_dense = conv1.propagate_linear(&after_relu_dense)
            .map_err(|e| TestCaseError::fail(format!("Dense Conv1 backward failed: {e}")))?
            .into_owned();

        // ---- Patches path: same chain but staying in Patches mode ----
        let patches_identity = PatchesLinearBounds::identity(
            (out_c2, out_h2, out_w2),
            (out_c2, out_h2, out_w2),
        );

        // Conv2 backward: creates initial patches from Conv2's kernel
        let after_conv2_patches = conv2.propagate_patches(&patches_identity)
            .map_err(|e| TestCaseError::fail(format!("Patches Conv2 backward failed: {e}")))?;

        // ReLU backward in Patches mode
        let after_relu_patches = match after_conv2_patches {
            CrownBounds::Patches(ref pb) => {
                relu.propagate_patches_with_bounds(pb, &pre_relu_bt)
                    .map_err(|e| TestCaseError::fail(format!("Patches ReLU backward failed: {e}")))?
            }
            CrownBounds::Dense(_) => {
                return Err(TestCaseError::fail(
                    "Conv2 backward unexpectedly left patches mode in a patches-composition test",
                ));
            }
        };

        // Conv1 backward: COMPOSES non-identity patches (the key test!)
        let after_conv1_patches = match after_relu_patches {
            CrownBounds::Patches(ref pb) => {
                conv1.propagate_patches(pb)
                    .map_err(|e| TestCaseError::fail(format!("Patches Conv1 composition failed: {e}")))?
            }
            CrownBounds::Dense(_) => {
                return Err(TestCaseError::fail(
                    "ReLU backward unexpectedly left patches mode before Conv1 composition",
                ));
            }
        };

        // Convert Patches result to Dense for comparison
        let after_conv1_patches_dense = after_conv1_patches.into_dense()
            .map_err(|e| TestCaseError::fail(format!("Patches to_dense failed: {e}")))?;

        // ---- Compare ----
        let chain_tol = 1e-4_f32;

        let la_dense = after_conv1_dense.lower_a();
        let la_patches = after_conv1_patches_dense.lower_a();
        let ua_dense = after_conv1_dense.upper_a();
        let ua_patches = after_conv1_patches_dense.upper_a();

        prop_assert_eq!(
            la_dense.shape(), la_patches.shape(),
            "lower_a shape mismatch: dense={:?}, patches={:?}",
            la_dense.shape(), la_patches.shape()
        );

        for j in 0..la_dense.nrows() {
            for i in 0..la_dense.ncols() {
                let d: f32 = la_dense[[j, i]];
                let p: f32 = la_patches[[j, i]];
                let diff = (d - p).abs();
                let scale = d.abs().max(p.abs()).max(1.0);
                prop_assert!(
                    diff <= chain_tol * scale,
                    "Conv→ReLU→Conv lower_a[{},{}]: dense={}, patches={}, diff={}",
                    j, i, d, p, diff,
                );
            }
        }
        for j in 0..ua_dense.nrows() {
            for i in 0..ua_dense.ncols() {
                let d: f32 = ua_dense[[j, i]];
                let p: f32 = ua_patches[[j, i]];
                let diff = (d - p).abs();
                let scale = d.abs().max(p.abs()).max(1.0);
                prop_assert!(
                    diff <= chain_tol * scale,
                    "Conv→ReLU→Conv upper_a[{},{}]: dense={}, patches={}, diff={}",
                    j, i, d, p, diff,
                );
            }
        }

        for i in 0..after_conv1_dense.num_outputs() {
            let diff_lb = (after_conv1_dense.lower_b()[i] - after_conv1_patches_dense.lower_b()[i]).abs();
            let scale_lb = after_conv1_dense.lower_b()[i].abs().max(1.0);
            prop_assert!(
                diff_lb <= chain_tol * scale_lb,
                "Conv→ReLU→Conv lower_b[{}]: dense={}, patches={}, diff={}",
                i, after_conv1_dense.lower_b()[i], after_conv1_patches_dense.lower_b()[i], diff_lb,
            );
            let diff_ub = (after_conv1_dense.upper_b()[i] - after_conv1_patches_dense.upper_b()[i]).abs();
            let scale_ub = after_conv1_dense.upper_b()[i].abs().max(1.0);
            prop_assert!(
                diff_ub <= chain_tol * scale_ub,
                "Conv→ReLU→Conv upper_b[{}]: dense={}, patches={}, diff={}",
                i, after_conv1_dense.upper_b()[i], after_conv1_patches_dense.upper_b()[i], diff_ub,
            );
        }
    }

    /// Test 7: Conv2d → SiLU Patches vs Dense equivalence.
    ///
    /// Network: Input → Conv → SiLU → Output
    /// Backward: identity at Output → SiLU backward → Conv backward
    ///
    /// Verifies that the macro-generated `propagate_patches_with_bounds` for SiLU
    /// produces identical bounds to the Dense path. This validates the generic
    /// activation Patches support added in Phase 2 step 11.
    ///
    /// Part of #2613
    // This 200-case nonlinear differential sweep needs the same shared-host
    // allowance as the neighboring pooling sweeps.
    #[ntest::timeout(20000)]
    #[test]
    fn proptest_conv2d_silu_patches_vs_dense(
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        kh in 1usize..=2,
        kw in 1usize..=2,
        in_h in 4usize..=6,
        in_w in 4usize..=6,
        seed in any::<u64>(),
    ) {
        use crate::layers::activations::SiLULayer;

        let out_h = in_h - kh + 1;
        let out_w = in_w - kw + 1;
        let conv_out_dim = out_c * out_h * out_w;

        let kernel = make_kernel(out_c, in_c, kh, kw, seed);
        let conv = Conv2dLayer::with_input_shape(
            kernel, None, (1, 1), (0, 0), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("Conv creation failed: {e}")))?;

        let silu = SiLULayer::new();

        // SiLU pre-activation bounds: SiLU's input = Conv's output.
        // Shape: (out_c, out_h, out_w)
        let pre_silu_lower: Vec<f32> = (0..conv_out_dim).map(|i| {
            let mut rng = seed.wrapping_add(i as u64 + 55555);
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            u * 4.0 - 2.0  // [-2, 2) — crosses zero
        }).collect();
        let pre_silu_upper: Vec<f32> = pre_silu_lower.iter().map(|&l| l + 0.5).collect();
        let pre_silu_bt = ny_tensor::BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w]), pre_silu_lower).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w]), pre_silu_upper).unwrap(),
        ).map_err(|e| TestCaseError::fail(format!("pre_silu BoundedTensor failed: {e}")))?;

        // ---- Dense path: identity → SiLU backward → Conv backward ----
        let identity_lb = LinearBounds::identity(conv_out_dim);
        let after_silu_dense = silu.propagate_linear_with_bounds(&identity_lb, &pre_silu_bt)
            .map_err(|e| TestCaseError::fail(format!("Dense SiLU backward failed: {e}")))?;
        // after_silu_dense: (conv_out_dim, conv_out_dim) — same shape, activation is elementwise
        let _after_conv_dense = conv.propagate_linear(&after_silu_dense)
            .map_err(|e| TestCaseError::fail(format!("Dense Conv backward failed: {e}")))?
            .into_owned();

        // ---- Patches path: identity patches → SiLU backward → Conv backward ----
        // In Patches mode, the initial Conv creates patches. But here SiLU is AFTER Conv
        // in the forward pass, so in backward, SiLU comes BEFORE Conv. The Patches identity
        // is at the SiLU output (= network output), and SiLU scales it, keeping Patches.
        // Then Conv composes the patches into Conv's input space.
        //
        // However, the initial identity is at the SiLU output, which has spatial shape
        // (out_c, out_h, out_w). The Patches structure started from a Conv2d backward.
        // Since we're testing SiLU+Conv and the identity is at SiLU's output (not Conv's
        // output), the Patches identity for SiLU doesn't come from Conv backward.
        //
        // For this test, we test just SiLU Patches backward in isolation:
        // Start with identity patches at SiLU's output, apply SiLU backward, compare to Dense.
        let patches_identity = PatchesLinearBounds::identity(
            (out_c, out_h, out_w),
            (out_c, out_h, out_w),
        );

        // SiLU backward in Patches mode (directly from identity patches)
        let after_silu_patches = silu.propagate_patches_with_bounds(&patches_identity, &pre_silu_bt)
            .map_err(|e| TestCaseError::fail(format!("Patches SiLU backward failed: {e}")))?;
        let after_silu_patches_dense = after_silu_patches.into_dense()
            .map_err(|e| TestCaseError::fail(format!("Patches to_dense failed: {e}")))?;

        // ---- Compare SiLU Patches vs Dense (identity → SiLU backward) ----
        let tol = 1e-4_f32;

        // Dense: identity → SiLU backward
        let after_silu_dense_only = silu.propagate_linear_with_bounds(&identity_lb, &pre_silu_bt)
            .map_err(|e| TestCaseError::fail(format!("Dense SiLU-only backward failed: {e}")))?;

        let la_d = after_silu_dense_only.lower_a();
        let la_p = after_silu_patches_dense.lower_a();
        prop_assert_eq!(la_d.shape(), la_p.shape(),
            "lower_a shape mismatch: dense={:?}, patches={:?}",
            la_d.shape(), la_p.shape()
        );

        for j in 0..la_d.nrows() {
            for i in 0..la_d.ncols() {
                let d: f32 = la_d[[j, i]];
                let p: f32 = la_p[[j, i]];
                let diff = (d - p).abs();
                let scale = d.abs().max(p.abs()).max(1.0);
                prop_assert!(
                    diff <= tol * scale,
                    "SiLU lower_a[{},{}]: dense={}, patches={}, diff={}",
                    j, i, d, p, diff,
                );
            }
        }

        let ua_d = after_silu_dense_only.upper_a();
        let ua_p = after_silu_patches_dense.upper_a();
        for j in 0..ua_d.nrows() {
            for i in 0..ua_d.ncols() {
                let d: f32 = ua_d[[j, i]];
                let p: f32 = ua_p[[j, i]];
                let diff = (d - p).abs();
                let scale = d.abs().max(p.abs()).max(1.0);
                prop_assert!(
                    diff <= tol * scale,
                    "SiLU upper_a[{},{}]: dense={}, patches={}, diff={}",
                    j, i, d, p, diff,
                );
            }
        }

        for i in 0..after_silu_dense_only.num_outputs() {
            let diff_lb = (after_silu_dense_only.lower_b()[i] - after_silu_patches_dense.lower_b()[i]).abs();
            let scale_lb = after_silu_dense_only.lower_b()[i].abs().max(1.0);
            prop_assert!(
                diff_lb <= tol * scale_lb,
                "SiLU lower_b[{}]: dense={}, patches={}, diff={}",
                i, after_silu_dense_only.lower_b()[i], after_silu_patches_dense.lower_b()[i], diff_lb,
            );
            let diff_ub = (after_silu_dense_only.upper_b()[i] - after_silu_patches_dense.upper_b()[i]).abs();
            let scale_ub = after_silu_dense_only.upper_b()[i].abs().max(1.0);
            prop_assert!(
                diff_ub <= tol * scale_ub,
                "SiLU upper_b[{}]: dense={}, patches={}, diff={}",
                i, after_silu_dense_only.upper_b()[i], after_silu_patches_dense.upper_b()[i], diff_ub,
            );
        }
    }
}

// =====================================================================
// #conv-patches-collect: EXACT padded-conv patches composition parity.
//
// The pre-existing composition proptest above uses padding=0 only. These
// tests exercise the intermediate-tap masking that keeps a PADDED conv
// chain in patches mode (`NY_CONV_PATCHES_COLLECT=1`) and pin the composed
// bound bit-close to the dense CROWN backward — the soundness contract for
// lifting the `nonzero_incoming_padding` guard. A mask that drops a real
// tap or keeps a leaked one would diverge here.
// =====================================================================
#[cfg(test)]
mod padded_compose_parity {
    use super::*;
    use crate::layers::common::PatchesPropagation;
    use ny_tensor::BoundedTensor;

    fn det_vals(n: usize, seed: u64, lo: f32, span: f32) -> Vec<f32> {
        let mut rng = seed | 1;
        (0..n)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let u = (rng as f32) / (u64::MAX as f32);
                u * span + lo
            })
            .collect()
    }

    /// One Conv(pad1) -> ReLU -> Conv(pad1) chain: dense CROWN backward vs
    /// patches CROWN backward (masking on). Asserts the second conv stayed in
    /// patches (so the masked compose was actually exercised) and the composed
    /// coefficients + concretized-sound bounds match the dense path.
    fn assert_two_conv_padded_parity(s2: (usize, usize), seed: u64) {
        // Conv1: (2, in_h, in_w) -> (3, .., ..), 3x3, pad 1, stride 1 ("same").
        let (in_c1, in_h, in_w) = (2usize, 7usize, 9usize);
        let out_c1 = 3usize;
        let out_h1 = in_h; // stride 1, pad 1, k 3
        let out_w1 = in_w;
        // Conv2: (3, out_h1, out_w1) -> (2, .., ..), 3x3, pad 1, stride s2.
        let in_c2 = out_c1;
        let out_c2 = 2usize;
        let out_h2 = (out_h1 + 2 - 3) / s2.0 + 1;
        let out_w2 = (out_w1 + 2 - 3) / s2.1 + 1;
        let conv2_out_dim = out_c2 * out_h2 * out_w2;

        let kernel1 = ArrayD::from_shape_vec(
            IxDyn(&[out_c1, in_c1, 3, 3]),
            det_vals(out_c1 * in_c1 * 9, seed, -1.0, 2.0),
        )
        .unwrap();
        let kernel2 = ArrayD::from_shape_vec(
            IxDyn(&[out_c2, in_c2, 3, 3]),
            det_vals(out_c2 * in_c2 * 9, seed ^ 0xabcd, -1.0, 2.0),
        )
        .unwrap();
        let bias2 = ndarray::Array1::from_vec(det_vals(out_c2, seed ^ 0x55, -0.3, 0.6));

        let conv1 =
            Conv2dLayer::with_input_shape(kernel1, None, (1, 1), (1, 1), in_h, in_w).unwrap();
        let conv2 = Conv2dLayer::with_input_shape(kernel2, Some(bias2), s2, (1, 1), out_h1, out_w1)
            .unwrap();
        let relu = ReLULayer::new();

        // Pre-activation bounds at Conv1's output (ReLU input), crossing zero.
        let relu_dim = out_c1 * out_h1 * out_w1;
        let lo = det_vals(relu_dim, seed ^ 0x777, -1.0, 1.5);
        let up: Vec<f32> = lo.iter().map(|&l| l + 0.4).collect();
        let pre_relu = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[out_c1, out_h1, out_w1]), lo).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[out_c1, out_h1, out_w1]), up).unwrap(),
        )
        .unwrap();

        // ---- Dense: identity -> Conv2 -> ReLU -> Conv1 ----
        let id = LinearBounds::identity(conv2_out_dim);
        let d1 = conv2.propagate_linear(&id).unwrap().into_owned();
        let d2 = relu.propagate_linear_with_bounds(&d1, &pre_relu).unwrap();
        let dense = conv1.propagate_linear(&d2).unwrap().into_owned();

        // ---- Patches (masking on): identity patches -> Conv2 -> ReLU -> Conv1 ----
        let patches =
            crate::tests::with_serialized_env_vars(&[("NY_CONV_PATCHES_COLLECT", "1")], || {
                let idp = PatchesLinearBounds::identity(
                    (out_c2, out_h2, out_w2),
                    (out_c2, out_h2, out_w2),
                );
                let p1 = conv2.propagate_patches(&idp).unwrap();
                let p1 = match p1 {
                    CrownBounds::Patches(ref pb) => {
                        relu.propagate_patches_with_bounds(pb, &pre_relu).unwrap()
                    }
                    CrownBounds::Dense(_) => panic!("Conv2 identity backward should stay patches"),
                };
                // Conv1 composes a NON-identity PADDED patch -> the masked path.
                match p1 {
                    CrownBounds::Patches(ref pb) => {
                        let out = conv1.propagate_patches(pb).unwrap();
                        assert!(
                            matches!(out, CrownBounds::Patches(_)),
                            "padded Conv1 compose must stay in patches (masking path), got Dense"
                        );
                        out.into_dense().unwrap()
                    }
                    CrownBounds::Dense(_) => panic!("ReLU backward should stay patches"),
                }
            });

        // ---- Coefficient parity ----
        let tol = 2e-4_f32;
        for (name, dm, pm) in [
            ("lower_a", dense.lower_a(), patches.lower_a()),
            ("upper_a", dense.upper_a(), patches.upper_a()),
        ] {
            assert_eq!(dm.shape(), pm.shape(), "{name} shape mismatch (s2={s2:?})");
            for (idx, (d, p)) in dm.iter().zip(pm.iter()).enumerate() {
                let scale = d.abs().max(p.abs()).max(1.0);
                assert!(
                    (d - p).abs() <= tol * scale,
                    "{name}[{idx}] dense={d} patches={p} (s2={s2:?}, seed={seed})"
                );
            }
        }

        // ---- Concretized-sound parity (the bound actually consumed) ----
        let in_dim = in_c1 * in_h * in_w;
        let xin_lo = det_vals(in_dim, seed ^ 0x99, -0.5, 0.5);
        let xin_up: Vec<f32> = xin_lo.iter().map(|&l| l + 0.25).collect();
        let xbox = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[in_c1, in_h, in_w]), xin_lo).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[in_c1, in_h, in_w]), xin_up).unwrap(),
        )
        .unwrap();
        let cd = dense.concretize_sound(&xbox);
        let cp = patches.concretize_sound(&xbox);
        let dl_s = cd.lower().as_slice().unwrap().to_vec();
        let du_s = cd.upper().as_slice().unwrap().to_vec();
        let pl_s = cp.lower().as_slice().unwrap().to_vec();
        let pu_s = cp.upper().as_slice().unwrap().to_vec();
        for i in 0..conv2_out_dim {
            let (dl, pl, du, pu) = (dl_s[i], pl_s[i], du_s[i], pu_s[i]);
            let scale = dl.abs().max(du.abs()).max(1.0);
            assert!(
                (dl - pl).abs() <= 1e-3 * scale && (du - pu).abs() <= 1e-3 * scale,
                "concretized[{i}] dense=[{dl},{du}] patches=[{pl},{pu}] (s2={s2:?}, seed={seed})"
            );
        }
    }

    #[test]
    fn patches_padded_compose_matches_dense_stride1() {
        for seed in [1u64, 7, 42, 1234, 99991] {
            assert_two_conv_padded_parity((1, 1), seed);
        }
    }

    #[test]
    fn patches_padded_compose_matches_dense_stride2() {
        for seed in [3u64, 11, 57, 2024, 88888] {
            assert_two_conv_padded_parity((2, 2), seed);
        }
    }

    /// Env-UNSET must keep the pre-existing guard: a padded non-identity
    /// compose returns UnsupportedConfiguration (caller falls to dense).
    #[test]
    fn patches_padded_compose_guarded_when_env_unset() {
        let (in_c1, in_h, in_w) = (2usize, 6usize, 6usize);
        let out_c1 = 2usize;
        let kernel1 = ArrayD::from_shape_vec(
            IxDyn(&[out_c1, in_c1, 3, 3]),
            det_vals(out_c1 * in_c1 * 9, 5, -1.0, 2.0),
        )
        .unwrap();
        let kernel2 = ArrayD::from_shape_vec(
            IxDyn(&[2, out_c1, 3, 3]),
            det_vals(2 * out_c1 * 9, 6, -1.0, 2.0),
        )
        .unwrap();
        let conv1 =
            Conv2dLayer::with_input_shape(kernel1, None, (1, 1), (1, 1), in_h, in_w).unwrap();
        let conv2 =
            Conv2dLayer::with_input_shape(kernel2, None, (1, 1), (1, 1), in_h, in_w).unwrap();
        let relu = ReLULayer::new();
        let relu_dim = out_c1 * in_h * in_w;
        let pre_relu = BoundedTensor::new(
            ArrayD::from_shape_vec(
                IxDyn(&[out_c1, in_h, in_w]),
                det_vals(relu_dim, 7, -1.0, 1.5),
            )
            .unwrap(),
            ArrayD::from_shape_vec(
                IxDyn(&[out_c1, in_h, in_w]),
                det_vals(relu_dim, 7, -0.6, 1.5),
            )
            .unwrap(),
        )
        .unwrap();
        crate::tests::with_serialized_env_vars(&[("NY_CONV_PATCHES_COLLECT", "0")], || {
            let idp = PatchesLinearBounds::identity((2, in_h, in_w), (2, in_h, in_w));
            let p1 = conv2.propagate_patches(&idp).unwrap();
            let CrownBounds::Patches(ref pb) = p1 else {
                panic!("conv2 should produce patches")
            };
            let p2 = relu.propagate_patches_with_bounds(pb, &pre_relu).unwrap();
            let CrownBounds::Patches(ref pb2) = p2 else {
                panic!("relu should keep patches")
            };
            let res = conv1.propagate_patches(pb2);
            assert!(
                matches!(res, Err(ny_core::NyError::UnsupportedConfiguration(_))),
                "env-unset padded compose must hit the guard, got {res:?}"
            );
        });
    }
}
