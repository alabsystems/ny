// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! AddConstant broadcast-axis agreement and fail-closed CROWN tests.

use super::AddConstantLayer;
use crate::layers::BoundPropagation;
use crate::{BatchedLinearBounds, Layer, LinearBounds};
use ndarray::{array, Array1, Array2, ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;

fn assert_close(actual: f32, expected: f32, tol: f32, label: impl std::fmt::Display) {
    assert!(
        (actual - expected).abs() < tol,
        "{label}: expected {expected}, got {actual}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// AddConstant broadcast-AXIS agreement (soundness).
//
// `propagate_concrete_point` (network/graph_ibp.rs) evaluates the graph by
// running IBP on a degenerate box, so `AddConstantLayer::propagate_ibp` IS
// ny's reference for what an Add-with-constant node computes. CROWN backward
// used to tile a short constant as `c[i % const_len]` — an unchecked LAST-AXIS
// assumption — while a per-channel constant varies along the SLOWEST axis
// (`c[i / (H*W)]`). The two paths therefore certified DIFFERENT FUNCTIONS, and
// nothing made the mis-placed bias conservative. These tests pin the agreement
// and the fail-closed refusals; every one of them fails on the tiling code.
// ═══════════════════════════════════════════════════════════════════════

/// `[C,1,1]` over a `[C,H,W]` operand — an ONNX-legal per-channel bias.
/// Returns the layer, an input box `[0,1]^{2,3,3}`, and the ONNX-correct flat
/// constant `[c0 ×9, c1 ×9]`.
fn channel_bias_case() -> (AddConstantLayer, BoundedTensor, Vec<f32>) {
    let constant = ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![10.0_f32, 20.0]).unwrap();
    let layer = AddConstantLayer::new(constant);
    let input = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&[2, 3, 3])),
        ArrayD::ones(IxDyn(&[2, 3, 3])),
    )
    .unwrap();
    let onnx: Vec<f32> = (0..18).map(|i| if i < 9 { 10.0 } else { 20.0 }).collect();
    (layer, input, onnx)
}

#[test]
fn add_constant_ibp_channel_bias_follows_onnx_broadcasting() {
    // Right-aligning [2,1,1] against [2,3,3] pairs axis 0 with axis 0, so the
    // constant varies along the SLOWEST axis. This is the ONNX answer, and it
    // is the answer `propagate_concrete_point` gives for the real network.
    let (layer, input, onnx) = channel_bias_case();
    let ibp = layer
        .propagate_ibp(&input)
        .expect("channel bias broadcasts");
    let got: Vec<f32> = ibp.lower().iter().copied().collect();
    for (i, want) in onnx.iter().enumerate() {
        assert_close(got[i], *want, 1e-6, format!("ibp lower[{i}]"));
    }
}

#[test]
fn add_constant_crown_backward_channel_bias_agrees_with_ibp_and_encloses_it() {
    let (layer, input, onnx) = channel_bias_case();
    let ibp = layer
        .propagate_ibp(&input)
        .expect("channel bias broadcasts");

    let crown = layer
        .propagate_crown_backward(&LinearBounds::identity(18), Some(&input))
        .expect("a per-channel bias is an ordinary elementwise Add");

    // A = I and b = 0, so the new bias IS the broadcast constant.
    for (i, want) in onnx.iter().enumerate() {
        assert_close(crown.lower_b[i], *want, 1e-4, format!("crown lower_b[{i}]"));
        assert_close(crown.upper_b[i], *want, 1e-4, format!("crown upper_b[{i}]"));
    }

    // The certificate must ENCLOSE what the network computes. The old tiling put
    // c1 = 20 at flat position 1, whose true range is [10, 11]: the certified
    // lower bound EXCLUDED the true value — a false proof, not a loose bound.
    let concrete = crown.concretize(&input);
    let cl: Vec<f32> = concrete.lower().iter().copied().collect();
    let cu: Vec<f32> = concrete.upper().iter().copied().collect();
    let il: Vec<f32> = ibp.lower().iter().copied().collect();
    let iu: Vec<f32> = ibp.upper().iter().copied().collect();
    for i in 0..18 {
        assert!(
            cl[i] <= il[i] + 1e-4,
            "CROWN lower {} excludes the true lower {} at flat index {i}",
            cl[i],
            il[i]
        );
        assert!(
            cu[i] >= iu[i] - 1e-4,
            "CROWN upper {} excludes the true upper {} at flat index {i}",
            cu[i],
            iu[i]
        );
    }
}

#[test]
fn add_constant_crown_backward_matches_the_rank1_cnn_bias_reading() {
    // A rank-1 `[C]` constant on a rank-3 `[C,H,W]` operand is read as a CHANNEL
    // bias (`effective_constant_shape`) — the same reading `propagate_ibp`,
    // `propagate_concrete_point` and `one_axis_phase.rs::add_constant_tensor`
    // use, and a deliberate departure from strict ONNX right-alignment. CROWN
    // must MIRROR the forward reference, not out-vote it with a last-axis guess.
    let layer = AddConstantLayer::new(array![10.0_f32, 20.0].into_dyn());
    let input = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&[2, 3, 3])),
        ArrayD::ones(IxDyn(&[2, 3, 3])),
    )
    .unwrap();
    let ibp: Vec<f32> = layer
        .propagate_ibp(&input)
        .expect("cnn bias broadcasts")
        .lower()
        .iter()
        .copied()
        .collect();

    let crown = layer
        .propagate_crown_backward(&LinearBounds::identity(18), Some(&input))
        .expect("the operand shape resolves the CNN-bias reading");
    for i in 0..18 {
        assert_close(
            crown.lower_b[i],
            ibp[i],
            1e-4,
            format!("cnn-bias crown lower_b[{i}]"),
        );
    }
}

#[test]
fn add_constant_refuses_rank1_layout_ambiguous_between_channel_and_last_axis() {
    // Operand [2,1,2] with a rank-1 [2] constant admits two incompatible
    // readings. Without source provenance selecting one, both IBP and CROWN
    // must refuse rather than certify different authored functions.
    let layer = AddConstantLayer::new(array![10.0_f32, 20.0].into_dyn());
    let input = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&[2, 1, 2])),
        ArrayD::zeros(IxDyn(&[2, 1, 2])),
    )
    .unwrap();
    let ibp_error = layer
        .propagate_ibp(&input)
        .expect_err("ambiguous rank-1 layout must fail closed");
    assert!(matches!(ibp_error, NyError::UnsupportedOp(_)));

    let crown_error = layer
        .propagate_crown_backward(&LinearBounds::identity(4), Some(&input))
        .expect_err("CROWN must share the same ambiguity refusal");
    assert!(matches!(crown_error, NyError::UnsupportedOp(_)));
}

#[test]
fn add_constant_crown_backward_without_a_shape_refuses_an_ambiguous_rank1_constant() {
    // With no operand shape, `[2]` against 18 output positions is ambiguous
    // between the channel reading (`c[i / 9]`) and the last-axis reading
    // (`c[i % 2]`). Guessing is what produced the wrong-function certificate.
    let layer = AddConstantLayer::new(array![10.0_f32, 20.0].into_dyn());
    let err = layer
        .propagate_linear(&LinearBounds::identity(18))
        .expect_err("an ambiguous broadcast axis must fail closed, not be guessed");
    assert!(
        matches!(err, NyError::UnsupportedOp(_)),
        "refusal must be UnsupportedOp so callers degrade to IBP, got {err:?}"
    );
}

#[test]
fn add_constant_crown_backward_without_shape_refuses_even_an_exact_length_non_scalar() {
    // `const_len == layer_dim` proves that the constant itself is not
    // replicated, but does not prove that the bounded operand is not. The same
    // output `[2,3]` could come from operand `[1,3]` plus constant `[2,3]`, in
    // which case passing six coefficient columns to a three-position
    // predecessor is invalid. Only the pre-activation shape can rule this out.
    let constant =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let layer = AddConstantLayer::new(constant);
    let err = layer
        .propagate_linear(&LinearBounds::identity(6))
        .expect_err("constant/output length equality does not authenticate the operand shape");
    assert!(matches!(err, NyError::UnsupportedOp(_)), "got {err:?}");
}

#[test]
fn add_constant_crown_backward_keeps_a_last_axis_broadcast_exact() {
    // `[1,W]` over `[H,W]`: the case where flat tiling happened to be the ONNX
    // answer. It stays exact when the operand shape proves no expansion.
    let constant = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0_f32, 2.0, 3.0]).unwrap();
    let layer = AddConstantLayer::new(constant);
    let input =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&[2, 3])), ArrayD::ones(IxDyn(&[2, 3]))).unwrap();
    let expected = [1.0_f32, 2.0, 3.0, 1.0, 2.0, 3.0];

    let ibp: Vec<f32> = layer
        .propagate_ibp(&input)
        .expect("last-axis broadcast")
        .lower()
        .iter()
        .copied()
        .collect();
    for i in 0..6 {
        assert_close(ibp[i], expected[i], 1e-6, format!("ibp lower[{i}]"));
    }

    let with_shape = layer
        .propagate_crown_backward(&LinearBounds::identity(6), Some(&input))
        .expect("last-axis broadcast stays supported with the operand shape");
    let without_shape = layer
        .propagate_linear(&LinearBounds::identity(6))
        .expect_err("a non-scalar cannot prove that it did not expand an unseen operand");
    assert!(matches!(without_shape, NyError::UnsupportedOp(_)));
    for i in 0..6 {
        assert_close(
            with_shape.lower_b[i],
            expected[i],
            1e-4,
            format!("with-shape lower_b[{i}]"),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// #acasxu-addconst: FC-bias layout certification.
//
// acasxu's Add nodes are rank-1 FC biases — constant `[50]` (Operation_1..6)
// and `[5]` (linear_7) — whose true operand is the post-MatMul `[1,D]`. That
// layout is right-aligned, non-expanding and unambiguous, so the proven
// from-shape arm must certify it EXACTLY: `A` passes through unchanged and the
// bias moves by exactly `A @ c`. The batched graph-BaB lane used to hand this
// layer the NETWORK INPUT (`[1,1,1,5]`) instead of the operand; that shape is
// not in the proven set and must KEEP declining — the admission below is a
// property of the true layout, never a loosening of the binding.
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn add_constant_crown_backward_certifies_the_fc_bias_layout_exactly() {
    // The acasxu shape family: rank-1 [D] bias against a [1,D] operand.
    for d in [50usize, 5] {
        let c: Vec<f32> = (0..d).map(|i| 0.25 * i as f32 - 3.0).collect();
        let constant = ArrayD::from_shape_vec(IxDyn(&[d]), c.clone()).unwrap();
        let layer = AddConstantLayer::new(constant);
        let operand =
            BoundedTensor::new(ArrayD::zeros(IxDyn(&[1, d])), ArrayD::ones(IxDyn(&[1, d])))
                .unwrap();

        // Non-trivial coefficients and non-zero incoming bias so the exactness
        // claim is not vacuous under A = I, b = 0.
        let n_out = 3;
        let lower_a = Array2::from_shape_fn((n_out, d), |(i, j)| {
            0.125 * (i as f32 + 1.0) * (j as f32 - 2.0)
        });
        let upper_a = Array2::from_shape_fn((n_out, d), |(i, j)| {
            0.0625 * (i as f32 - 1.0) * (j as f32 + 1.0)
        });
        let lower_b = Array1::from_shape_fn(n_out, |i| 0.5 * i as f32 - 1.0);
        let upper_b = Array1::from_shape_fn(n_out, |i| 0.5 * i as f32 + 1.0);
        let bounds = LinearBounds::new(
            lower_a.clone(),
            lower_b.clone(),
            upper_a.clone(),
            upper_b.clone(),
        )
        .unwrap();

        let out = layer
            .propagate_crown_backward(&bounds, Some(&operand))
            .expect("the FC-bias layout [D] against [1,D] is in the proven set");

        // Coefficients pass through UNCHANGED — bit-exact.
        assert_eq!(out.lower_a, lower_a, "d={d}: lower_a must pass through");
        assert_eq!(out.upper_a, upper_a, "d={d}: upper_a must pass through");

        // Bias moves by exactly A @ c: reference computed independently in f64.
        for i in 0..n_out {
            let dot_l: f64 = (0..d).map(|j| lower_a[[i, j]] as f64 * c[j] as f64).sum();
            let dot_u: f64 = (0..d).map(|j| upper_a[[i, j]] as f64 * c[j] as f64).sum();
            let want_l = (lower_b[i] as f64 + dot_l) as f32;
            let want_u = (upper_b[i] as f64 + dot_u) as f32;
            // The implementation applies ONE directed-rounding step per
            // endpoint, so agreement must hold to a few ulps of the magnitude.
            let tol_l = (want_l.abs() * 4.0 * f32::EPSILON).max(1e-6);
            let tol_u = (want_u.abs() * 4.0 * f32::EPSILON).max(1e-6);
            assert_close(out.lower_b[i], want_l, tol_l, format!("d={d} lower_b[{i}]"));
            assert_close(out.upper_b[i], want_u, tol_u, format!("d={d} upper_b[{i}]"));
            // Directed rounding may move each endpoint OUTWARD only.
            assert!(
                out.lower_b[i] <= want_l && out.upper_b[i] >= want_u,
                "d={d} bias[{i}]: endpoints moved inward"
            );
        }
    }
}

#[test]
fn add_constant_crown_backward_fc_bias_still_declines_unproven_layouts() {
    // (a) The exact wrong-operand shape from the batched-lane bug: bias [50]
    // against the acasxu NETWORK INPUT [1,1,1,5]. Not broadcast-compatible;
    // must keep refusing with ShapeMismatch. Certifying the true [1,50] layout
    // must not admit a caller that lies about the operand.
    let c50: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let layer = AddConstantLayer::new(ArrayD::from_shape_vec(IxDyn(&[50]), c50).unwrap());
    let wrong_operand = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&[1, 1, 1, 5])),
        ArrayD::ones(IxDyn(&[1, 1, 1, 5])),
    )
    .unwrap();
    let err = layer
        .propagate_crown_backward(&LinearBounds::identity(50), Some(&wrong_operand))
        .expect_err("the network-input shape is not the operand and must decline");
    assert!(
        matches!(err, NyError::ShapeMismatch { .. }),
        "wrong-operand refusal must stay ShapeMismatch, got {err:?}"
    );

    // (b) A genuinely-ambiguous layout in the same rank-1 family: [5] against
    // [5,1,5] admits both the channel reading (c[i/5]) and the last-axis
    // reading (c[i%5]) — two different functions. Still refuses.
    let c5: Vec<f32> = (0..5).map(|i| i as f32 + 1.0).collect();
    let ambiguous_layer = AddConstantLayer::new(ArrayD::from_shape_vec(IxDyn(&[5]), c5).unwrap());
    let ambiguous_operand = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&[5, 1, 5])),
        ArrayD::ones(IxDyn(&[5, 1, 5])),
    )
    .unwrap();
    let err = ambiguous_layer
        .propagate_crown_backward(&LinearBounds::identity(25), Some(&ambiguous_operand))
        .expect_err("an ambiguous rank-1 layout must keep failing closed");
    assert!(
        matches!(err, NyError::UnsupportedOp(_)),
        "ambiguity refusal must stay UnsupportedOp so callers degrade, got {err:?}"
    );
}

#[test]
fn add_constant_batched_crown_refuses_a_leading_axis_constant() {
    // The batched substitution folds ONE shared column vector into every
    // position's bias, so it can only represent a constant that varies along the
    // LAST axis. `[2,1,1]` over `[2,3,2]` varies along axis 0, yet the old code
    // accepted it because `const_len == in_dim` and applied it to the wrong
    // axis. Refusing routes the node to the partial-CROWN / IBP fallback.
    let constant = ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![10.0_f32, 20.0]).unwrap();
    let layer = AddConstantLayer::new(constant);
    let bounds = BatchedLinearBounds::identity(&[2, 3, 2]).unwrap();
    let pre_activation = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&[2, 3, 2])),
        ArrayD::zeros(IxDyn(&[2, 3, 2])),
    )
    .unwrap();
    let err = Layer::AddConstant(layer)
        .propagate_crown_backward_batched(&bounds, Some(&pre_activation), None)
        .expect_err("a leading-axis constant must fail closed in the batched path");
    assert!(
        matches!(err, NyError::UnsupportedOp(_)),
        "refusal must be UnsupportedOp so the dispatcher degrades, got {err:?}"
    );
}

#[test]
fn add_constant_batched_direct_without_pre_activation_refuses_non_scalar() {
    let layer = AddConstantLayer::new(array![1.0_f32, 2.0].into_dyn());
    let bounds = BatchedLinearBounds::identity(&[2]).unwrap();
    let err = layer
        .propagate_linear_batched(&bounds)
        .expect_err("direct non-scalar batched CROWN lacks operand-shape evidence");
    assert!(matches!(err, NyError::UnsupportedOp(_)), "got {err:?}");
}

#[test]
fn add_constant_batched_crown_keeps_a_last_axis_constant_exact() {
    // `[3]` over `[2,3]` right-aligns onto the last axis — exactly what the
    // batched substitution represents. Must stay supported and exact.
    let layer = AddConstantLayer::new(array![1.0_f32, 2.0, 3.0].into_dyn());
    let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();
    let pre_activation =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&[2, 3])), ArrayD::zeros(IxDyn(&[2, 3]))).unwrap();
    let result = Layer::AddConstant(layer)
        .propagate_crown_backward_batched(&bounds, Some(&pre_activation), None)
        .expect("a last-axis constant is representable in batched mode");
    for b in 0..2 {
        for j in 0..3 {
            assert_close(
                result.lower_b[[b, j]],
                (j + 1) as f32,
                1e-4,
                format!("batched lower_b[{b},{j}]"),
            );
        }
    }
}

#[test]
fn add_constant_batched_crown_refuses_operand_expansion_with_real_pre_activation_shape() {
    // Operand `[1,3]` plus constant `[2,3]` produces `[2,3]`. The backward
    // would need to sum the two replicated output rows into one predecessor
    // row; passing the carrier through unchanged is not a valid certificate.
    let constant =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();
    let pre_activation =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&[1, 3])), ArrayD::zeros(IxDyn(&[1, 3]))).unwrap();
    let err = Layer::AddConstant(AddConstantLayer::new(constant))
        .propagate_crown_backward_batched(&bounds, Some(&pre_activation), None)
        .expect_err("operand replication requires a column reduction and must refuse");
    assert!(matches!(err, NyError::UnsupportedOp(_)), "got {err:?}");
}
