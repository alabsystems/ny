// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for weight inspection and norm computation.

use super::norms::{
    extract_gguf_block_number, extract_hf_block_number, frobenius_norm, spectral_norm_approx,
};

// ============================================================
// frobenius_norm tests
// ============================================================

#[test]
fn test_frobenius_norm_1d() {
    // [3, 4] -> sqrt(9 + 16) = 5
    let tensor = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![3.0f32, 4.0]).unwrap();
    let norm = frobenius_norm(&tensor);
    assert!((norm - 5.0).abs() < 1e-6);
}

#[test]
fn test_frobenius_norm_2d() {
    // [[1, 2], [3, 4]] -> sqrt(1 + 4 + 9 + 16) = sqrt(30)
    let tensor =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
    let norm = frobenius_norm(&tensor);
    assert!((norm - 30.0f64.sqrt()).abs() < 1e-6);
}

#[test]
fn test_frobenius_norm_zeros() {
    let tensor = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.0f32; 3]).unwrap();
    let norm = frobenius_norm(&tensor);
    assert!(norm.abs() < 1e-10);
}

#[test]
fn test_frobenius_norm_negative() {
    // [-3, 4] -> sqrt(9 + 16) = 5 (squares make negatives positive)
    let tensor = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![-3.0f32, 4.0]).unwrap();
    let norm = frobenius_norm(&tensor);
    assert!((norm - 5.0).abs() < 1e-6);
}

// ============================================================
// spectral_norm_approx tests
// ============================================================

#[test]
fn test_spectral_norm_identity_2x2() {
    // Identity matrix has spectral norm = 1
    let tensor =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0f32, 0.0, 0.0, 1.0])
            .unwrap();
    // Power iteration on 2x2 identity converges in 1 iteration (1e-4 tolerance)
    let norm = spectral_norm_approx(&tensor, 20);
    assert!((norm - 1.0).abs() < 1e-4, "spectral norm of I = {}", norm);
}

#[test]
fn test_spectral_norm_scaled_identity() {
    // 3*I has spectral norm = 3
    let tensor =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![3.0f32, 0.0, 0.0, 3.0])
            .unwrap();
    // Power iteration on 2x2 scaled identity converges in 1 iteration (1e-4 tolerance)
    let norm = spectral_norm_approx(&tensor, 20);
    assert!((norm - 3.0).abs() < 1e-4, "spectral norm of 3I = {}", norm);
}

#[test]
fn test_spectral_norm_non_square() {
    // 2x3 matrix - check it doesn't crash and returns reasonable value
    let tensor = ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&[2, 3]),
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let norm = spectral_norm_approx(&tensor, 20);
    // Known: singular values are approximately 9.508 and 0.773, largest is ~9.5
    assert!(norm > 8.0 && norm < 11.0, "spectral norm = {}", norm);
}

#[test]
fn test_spectral_norm_fallback_1d() {
    // 1D tensor should fallback to Frobenius norm
    let tensor =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![3.0f32, 4.0, 0.0, 0.0]).unwrap();
    let norm = spectral_norm_approx(&tensor, 20);
    assert!((norm - 5.0).abs() < 1e-6);
}

// ============================================================
// extract_gguf_block_number tests
// ============================================================

#[test]
fn test_extract_gguf_block_basic() {
    assert_eq!(extract_gguf_block_number("blk.5.attn_q.weight"), Some(5));
    assert_eq!(extract_gguf_block_number("blk.0.ffn_gate.weight"), Some(0));
    assert_eq!(extract_gguf_block_number("blk.31.attn_v.weight"), Some(31));
}

#[test]
fn test_extract_gguf_block_large_number() {
    assert_eq!(
        extract_gguf_block_number("blk.127.attn_k.weight"),
        Some(127)
    );
}

#[test]
fn test_extract_gguf_block_not_a_block() {
    assert_eq!(extract_gguf_block_number("token_embd.weight"), None);
    assert_eq!(extract_gguf_block_number("output.weight"), None);
    assert_eq!(extract_gguf_block_number("output_norm.weight"), None);
}

#[test]
fn test_extract_gguf_block_malformed() {
    assert_eq!(extract_gguf_block_number("blk."), None);
    assert_eq!(extract_gguf_block_number("blk"), None);
    assert_eq!(extract_gguf_block_number("block.5.weight"), None);
}

// ============================================================
// extract_hf_block_number tests
// ============================================================

#[test]
fn test_extract_hf_block_encoder_layers() {
    assert_eq!(
        extract_hf_block_number("encoder.layers.5.self_attn.q_proj.weight"),
        Some(5)
    );
    assert_eq!(
        extract_hf_block_number("encoder.layers.11.mlp.fc1.weight"),
        Some(11)
    );
}

#[test]
fn test_extract_hf_block_decoder_layers() {
    assert_eq!(
        extract_hf_block_number("decoder.layers.3.self_attn.k_proj.weight"),
        Some(3)
    );
    assert_eq!(
        extract_hf_block_number("decoder.layers.0.mlp.fc2.weight"),
        Some(0)
    );
}

#[test]
fn test_extract_hf_block_model_layers() {
    assert_eq!(
        extract_hf_block_number("model.layers.15.self_attn.q_proj.weight"),
        Some(15)
    );
}

#[test]
fn test_extract_hf_block_layers_only() {
    assert_eq!(
        extract_hf_block_number("layers.7.attention.weight"),
        Some(7)
    );
}

#[test]
fn test_extract_hf_block_not_a_block() {
    assert_eq!(extract_hf_block_number("embed_tokens.weight"), None);
    assert_eq!(extract_hf_block_number("lm_head.weight"), None);
    assert_eq!(extract_hf_block_number("norm.weight"), None);
}

#[test]
fn test_extract_hf_block_malformed() {
    assert_eq!(extract_hf_block_number("encoder.layers."), None);
    assert_eq!(extract_hf_block_number("layers"), None);
}
