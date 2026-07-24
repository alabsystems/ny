// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::dequant::{
    dequantize_q2_k, dequantize_q3_k, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k,
    dequantize_q5_k, dequantize_q6_k, dequantize_q8_0, get_block_elements, get_block_size,
};
use super::load::{is_dequantizable, is_quantized_type, load_tensor_data};
use super::metadata::format_metadata_value;
use super::parser::{align_up, compute_data_section_offset};
use super::{gguf_info, load_gguf};
use gguf::{GGMLType, GGUFFile, GGUFMetadataValue, GGUFTensorInfo};

fn push_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn push_u64(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn push_string(v: &mut Vec<u8>, s: &str) {
    push_u64(v, s.len() as u64);
    v.extend_from_slice(s.as_bytes());
}

#[ntest::timeout(5000)]
#[test]
fn test_gguf_tensor_offsets_are_relative_to_data_section() {
    // Minimal GGUF v3 file with one F32 tensor at offset 0 (relative to the data section).
    let mut buf = Vec::<u8>::new();
    buf.extend_from_slice(b"GGUF");
    push_u32(&mut buf, 3); // version
    push_u64(&mut buf, 1); // tensor_count
    push_u64(&mut buf, 1); // metadata_count

    // metadata: general.alignment = 32 (Uint32)
    push_string(&mut buf, "general.alignment");
    push_u32(&mut buf, 4);
    push_u32(&mut buf, 32);

    // tensor info: name, ndims, dims, type, offset
    push_string(&mut buf, "test.weight");
    push_u32(&mut buf, 1);
    push_u64(&mut buf, 4);
    push_u32(&mut buf, GGMLType::F32 as u32);
    push_u64(&mut buf, 0);

    let expected_data_start = align_up(buf.len(), 32);
    buf.resize(expected_data_start, 0);

    for f in [1.0f32, 2.0, 3.0, 4.0] {
        buf.extend_from_slice(&f.to_le_bytes());
    }

    let gguf_file = GGUFFile::read(&buf).unwrap().unwrap();
    let tensor = &gguf_file.tensors[0];
    assert_eq!(tensor.offset, 0);

    let data_start = compute_data_section_offset(&buf).unwrap();
    assert_eq!(data_start, expected_data_start);

    let arr = load_tensor_data(&buf, data_start, tensor, 4).unwrap();
    let (raw, _) = arr.into_raw_vec_and_offset();
    assert_eq!(raw, vec![1.0, 2.0, 3.0, 4.0]);
}

// Helper: convert f32 to f16 bits (for test data creation)
fn f32_to_half(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf/NaN
        return sign | 0x7C00 | ((mant != 0) as u16);
    }
    if exp == 0 {
        // Zero/Denormal
        return sign;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00; // Overflow to inf
    }
    if new_exp <= 0 {
        return sign; // Underflow to zero
    }

    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

fn pack_scale_min_k4(scales: &mut [u8], j: usize, scale: u8, min: u8) {
    assert!(scales.len() >= 12, "K4 scales require 12 bytes");
    let ls = scale & 0x3F;
    let lm = min & 0x3F;
    if j < 4 {
        scales[j] = ls;
        scales[j + 4] = lm;
    } else {
        scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
        scales[j - 4] = (scales[j - 4] & 0x3F) | (((ls >> 4) & 0x03) << 6);
        scales[j] = (scales[j] & 0x3F) | (((lm >> 4) & 0x03) << 6);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_get_scale_min_k4_packing() {
    let mut scales = [0u8; 12];
    for j in 0..8 {
        pack_scale_min_k4(&mut scales, j, (j + 1) as u8, (j + 2) as u8);
    }

    for j in 0..8 {
        let (scale, min) = super::dequant::get_scale_min_k4(j, &scales);
        assert_eq!(scale, (j + 1) as u8);
        assert_eq!(min, (j + 2) as u8);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_is_quantized_type() {
    assert!(!is_quantized_type(&GGMLType::F32));
    assert!(!is_quantized_type(&GGMLType::F16));
    assert!(is_quantized_type(&GGMLType::Q4_0));
    assert!(is_quantized_type(&GGMLType::Q8_0));
    assert!(is_quantized_type(&GGMLType::Q8_1));
    // K-quants are quantized types.
    assert!(is_quantized_type(&GGMLType::Q2K));
    assert!(is_quantized_type(&GGMLType::Q3K));
    assert!(is_quantized_type(&GGMLType::Q4K));
    assert!(is_quantized_type(&GGMLType::Q5K));
    assert!(is_quantized_type(&GGMLType::Q6K));
}

#[ntest::timeout(5000)]
#[test]
fn test_is_dequantizable() {
    assert!(is_dequantizable(&GGMLType::F32));
    assert!(is_dequantizable(&GGMLType::F16));
    assert!(is_dequantizable(&GGMLType::Q8_0));
    assert!(is_dequantizable(&GGMLType::Q4_0));
    assert!(is_dequantizable(&GGMLType::Q4_1));
    assert!(is_dequantizable(&GGMLType::Q5_0));
    assert!(is_dequantizable(&GGMLType::Q5_1));
    assert!(is_dequantizable(&GGMLType::Q8_1));
    // K-quants are now supported
    assert!(is_dequantizable(&GGMLType::Q2K));
    assert!(is_dequantizable(&GGMLType::Q3K));
    assert!(is_dequantizable(&GGMLType::Q4K));
    assert!(is_dequantizable(&GGMLType::Q5K));
    assert!(is_dequantizable(&GGMLType::Q6K));
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q8_0_basic() {
    // Create a Q8_0 block: 2 bytes (f16 scale) + 32 bytes (int8 quants)
    let scale: f32 = 0.5;
    let scale_bits = f32_to_half(scale);

    let mut data = Vec::with_capacity(34);
    data.extend_from_slice(&scale_bits.to_le_bytes());

    // Create int8 values: -127 to +127 range, we'll use simple values
    // Let's use values 0, 1, 2, ..., 31 as signed int8
    for i in 0..32i8 {
        data.push(i as u8);
    }

    let result = dequantize_q8_0(&data, 32).unwrap();
    assert_eq!(result.len(), 32);

    // Verify: y[i] = qs[i] * scale
    for (i, &val) in result.iter().enumerate() {
        let expected = i as f32 * scale;
        assert!(
            (val - expected).abs() < 0.01,
            "Mismatch at {}: got {}, expected {}",
            i,
            val,
            expected
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q8_0_negative() {
    // Test with negative scale and negative values
    let scale: f32 = -0.25;
    let scale_bits = f32_to_half(scale);

    let mut data = Vec::with_capacity(34);
    data.extend_from_slice(&scale_bits.to_le_bytes());

    // Use negative values
    for i in 0..32i8 {
        data.push((i - 16) as u8); // -16 to +15
    }

    let result = dequantize_q8_0(&data, 32).unwrap();
    assert_eq!(result.len(), 32);

    for (i, &val) in result.iter().enumerate() {
        let qs = (i as i8 - 16) as f32;
        let expected = qs * scale;
        assert!(
            (val - expected).abs() < 0.01,
            "Mismatch at {}: got {}, expected {}",
            i,
            val,
            expected
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q4_0_basic() {
    // Create a Q4_0 block: 2 bytes (f16 scale) + 16 bytes (nibble pairs)
    let scale: f32 = 1.0;
    let scale_bits = f32_to_half(scale);

    let mut data = Vec::with_capacity(18);
    data.extend_from_slice(&scale_bits.to_le_bytes());

    // Each byte stores two nibbles. Lower nibble = first half, upper = second half.
    // Values are 0-15, offset by -8 to get -8 to +7.
    // Let's use byte values where lower nibble = 8, upper nibble = 8 (both = 0 after offset)
    data.extend(std::iter::repeat_n(0x88, 16)); // Both nibbles = 8, so (8-8)=0 and (8-8)=0

    let result = dequantize_q4_0(&data, 32).unwrap();
    assert_eq!(result.len(), 32);

    // All values should be 0.0
    for (i, &val) in result.iter().enumerate() {
        assert!(val.abs() < 0.01, "Expected 0.0 at {}, got {}", i, val);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q4_0_varied() {
    // Test with varied nibble values
    let scale: f32 = 2.0;
    let scale_bits = f32_to_half(scale);

    let mut data = Vec::with_capacity(18);
    data.extend_from_slice(&scale_bits.to_le_bytes());

    // Create 16 bytes with pattern: low=0, high=15 (after offset: -8 and +7)
    data.extend(std::iter::repeat_n(0xF0, 16)); // low nibble = 0, high nibble = 15

    let result = dequantize_q4_0(&data, 32).unwrap();
    assert_eq!(result.len(), 32);

    // First 16 values should be (0-8)*2 = -16
    for (i, &val) in result[..16].iter().enumerate() {
        assert!(
            (val - (-16.0)).abs() < 0.01,
            "Expected -16.0 at {}, got {}",
            i,
            val
        );
    }
    // Next 16 values should be (15-8)*2 = 14
    for (i, &val) in result[16..32].iter().enumerate() {
        assert!(
            (val - 14.0).abs() < 0.01,
            "Expected 14.0 at {}, got {}",
            i + 16,
            val
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q4_1_basic() {
    // Q4_1: scale + min, no offset subtraction
    let scale: f32 = 1.0;
    let min: f32 = -5.0;
    let scale_bits = f32_to_half(scale);
    let min_bits = f32_to_half(min);

    let mut data = Vec::with_capacity(20);
    data.extend_from_slice(&scale_bits.to_le_bytes());
    data.extend_from_slice(&min_bits.to_le_bytes());

    // All nibbles = 0
    data.extend(std::iter::repeat_n(0x00, 16));

    let result = dequantize_q4_1(&data, 32).unwrap();
    assert_eq!(result.len(), 32);

    // y = 0 * 1.0 + (-5.0) = -5.0
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - (-5.0)).abs() < 0.01,
            "Expected -5.0 at {}, got {}",
            i,
            val
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q8_0_multiple_blocks() {
    // Test with 2 blocks (64 elements)
    let scale1: f32 = 1.0;
    let scale2: f32 = 2.0;

    let mut data = Vec::with_capacity(68);

    // Block 1
    data.extend_from_slice(&f32_to_half(scale1).to_le_bytes());
    for i in 0..32u8 {
        data.push(i);
    }

    // Block 2
    data.extend_from_slice(&f32_to_half(scale2).to_le_bytes());
    for i in 0..32u8 {
        data.push(i);
    }

    let result = dequantize_q8_0(&data, 64).unwrap();
    assert_eq!(result.len(), 64);

    // First block: y = i * 1.0
    for (i, &val) in result[..32].iter().enumerate() {
        assert!(
            (val - i as f32).abs() < 0.01,
            "Block1 mismatch at {}: got {}, expected {}",
            i,
            val,
            i as f32
        );
    }
    // Second block: y = i * 2.0
    for (i, &val) in result[32..64].iter().enumerate() {
        assert!(
            (val - (i as f32 * 2.0)).abs() < 0.01,
            "Block2 mismatch at {}: got {}, expected {}",
            i,
            val,
            i as f32 * 2.0
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_invalid_element_count() {
    // Q8_0 requires elements divisible by 32
    let data = vec![0u8; 34];
    let result = dequantize_q8_0(&data, 31);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("divisible by 32"));

    // Q4_0 requires elements divisible by 32
    let data = vec![0u8; 18];
    let result = dequantize_q4_0(&data, 16);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("divisible by 32"));
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_data_too_short() {
    // Q8_0: needs 34 bytes per block
    let data = vec![0u8; 33];
    let result = dequantize_q8_0(&data, 32);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too short"));

    // Q4_0: needs 18 bytes per block
    let data = vec![0u8; 17];
    let result = dequantize_q4_0(&data, 32);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too short"));
}

#[ntest::timeout(5000)]
#[test]
fn test_block_size_functions() {
    // Simple quants
    assert_eq!(get_block_size(&GGMLType::Q8_0), Some(34));
    assert_eq!(get_block_size(&GGMLType::Q4_0), Some(18));
    assert_eq!(get_block_size(&GGMLType::Q4_1), Some(20));
    assert_eq!(get_block_size(&GGMLType::Q5_0), Some(22));
    assert_eq!(get_block_size(&GGMLType::Q5_1), Some(24));
    assert_eq!(get_block_size(&GGMLType::Q8_1), Some(36));
    // K-quants
    assert_eq!(get_block_size(&GGMLType::Q2K), Some(84));
    assert_eq!(get_block_size(&GGMLType::Q3K), Some(110));
    assert_eq!(get_block_size(&GGMLType::Q4K), Some(144));
    assert_eq!(get_block_size(&GGMLType::Q5K), Some(176));
    assert_eq!(get_block_size(&GGMLType::Q6K), Some(210));

    // Elements per block
    assert_eq!(get_block_elements(&GGMLType::Q8_0), Some(32));
    assert_eq!(get_block_elements(&GGMLType::Q4_0), Some(32));
    assert_eq!(get_block_elements(&GGMLType::Q4_1), Some(32));
    assert_eq!(get_block_elements(&GGMLType::Q5_0), Some(32));
    assert_eq!(get_block_elements(&GGMLType::Q5_1), Some(32));
    assert_eq!(get_block_elements(&GGMLType::Q8_1), Some(32));
    // K-quants all use 256 elements per super-block
    assert_eq!(get_block_elements(&GGMLType::Q2K), Some(256));
    assert_eq!(get_block_elements(&GGMLType::Q3K), Some(256));
    assert_eq!(get_block_elements(&GGMLType::Q4K), Some(256));
    assert_eq!(get_block_elements(&GGMLType::Q5K), Some(256));
    assert_eq!(get_block_elements(&GGMLType::Q6K), Some(256));

    // Non-quantized types return None.
    assert_eq!(get_block_size(&GGMLType::F32), None);
    assert_eq!(get_block_elements(&GGMLType::F16), None);
}

#[ntest::timeout(5000)]
#[test]
fn test_format_metadata_value() {
    assert_eq!(format_metadata_value(&GGUFMetadataValue::Uint32(42)), "42");
    assert_eq!(
        format_metadata_value(&GGUFMetadataValue::String("test".to_string())),
        "test"
    );
    assert_eq!(
        format_metadata_value(&GGUFMetadataValue::Bool(true)),
        "true"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_file_not_found() {
    let result = load_gguf("/nonexistent/path/model.gguf");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("File not found"));
}

#[ntest::timeout(5000)]
#[test]
fn test_info_file_not_found() {
    let result = gguf_info("/nonexistent/path/model.gguf");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("File not found"));
}

// ==========================================================================
// K-Quant Dequantization Tests
// ==========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q2_k_basic() {
    // Q2_K: 84 bytes per 256 elements
    // Layout: scales[16], qs[64], d(2), dmin(2)
    let mut data = vec![0u8; 84];

    // Set d = 1.0, dmin = 0.0
    let d_bits = f32_to_half(1.0);
    let dmin_bits = f32_to_half(0.0);
    data[80..82].copy_from_slice(&d_bits.to_le_bytes());
    data[82..84].copy_from_slice(&dmin_bits.to_le_bytes());

    // Set scales: first scale nibble = 1, second (min) nibble = 0
    data[0..16].fill(0x01); // scale = 1, min = 0

    // Set all quants to 0 (2-bit values = 0)
    data[16..80].fill(0);

    let result = dequantize_q2_k(&data, 256).unwrap();
    assert_eq!(result.len(), 256);
    // All values should be 0 (d * scale * quant - dmin * min = 1*1*0 - 0*0*0 = 0)
    for v in &result {
        assert!(v.abs() < 1e-5, "Expected ~0, got {}", v);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q2_k_all_ones() {
    // Q2_K: choose a pattern that produces quant=1 for all 256 values:
    // 0x55 = 0b01010101, so each 2-bit lane (shift 0/2/4/6) yields 1.
    let mut data = vec![0u8; 84];

    // d = 1.0, dmin = 0.0
    let d_bits = f32_to_half(1.0);
    let dmin_bits = f32_to_half(0.0);
    data[80..82].copy_from_slice(&d_bits.to_le_bytes());
    data[82..84].copy_from_slice(&dmin_bits.to_le_bytes());

    // dl = d * 1, ml = dmin * 0
    data[0..16].fill(0x01);
    data[16..80].fill(0x55);

    let result = dequantize_q2_k(&data, 256).unwrap();
    assert_eq!(result.len(), 256);
    for v in &result {
        assert!((v - 1.0).abs() < 1e-5, "Expected ~1.0, got {}", v);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q3_k_basic() {
    // Q3_K: 110 bytes per 256 elements
    // Layout: hmask[32], qs[64], scales[12], d(2)
    let mut data = vec![0u8; 110];

    // Set d = 1.0
    let d_bits = f32_to_half(1.0);
    data[108..110].copy_from_slice(&d_bits.to_le_bytes());

    // Set scales to 32 (so scale - 32 = 0)
    // The scales are packed in a complex way, but setting all to 32 gives zero scales
    data[96..108].fill(0x20); // This approximates scales of 32

    // Set hmask to all 1s (all high-bit flags set)
    data[0..32].fill(0xFF);

    let result = dequantize_q3_k(&data, 256).unwrap();
    assert_eq!(result.len(), 256);
    // Values depend on the complex scale unpacking, but should be finite
    for v in &result {
        assert!(v.is_finite(), "Got non-finite value: {}", v);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q4_k_basic() {
    // Q4_K: 144 bytes per 256 elements
    // Layout: d(2), dmin(2), scales[12], qs[128]
    let mut data = vec![0u8; 144];

    // Set d = 1.0, dmin = 0.0
    let d_bits = f32_to_half(1.0);
    let dmin_bits = f32_to_half(0.0);
    data[0..2].copy_from_slice(&d_bits.to_le_bytes());
    data[2..4].copy_from_slice(&dmin_bits.to_le_bytes());

    // Set scales to encode scale=1, min=0 for all groups (ggml packing).
    let scales = &mut data[4..16];
    scales.fill(0);
    for j in 0..8 {
        pack_scale_min_k4(scales, j, 1, 0);
    }

    // Set qs: 4-bit values packed in nibbles
    // Set all nibbles to 2 (so value = 2 * scale = 2)
    data[16..144].fill(0x22); // low nibble = 2, high nibble = 2

    let result = dequantize_q4_k(&data, 256).unwrap();
    assert_eq!(result.len(), 256);
    // All values should be close to 2.0
    for v in &result {
        assert!((v - 2.0).abs() < 1e-5, "Expected ~2.0, got {}", v);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q5_k_basic() {
    // Q5_K: 176 bytes per 256 elements
    // Layout: d(2), dmin(2), scales[12], qh[32], qs[128]
    let mut data = vec![0u8; 176];

    // Set d = 1.0, dmin = 0.0
    let d_bits = f32_to_half(1.0);
    let dmin_bits = f32_to_half(0.0);
    data[0..2].copy_from_slice(&d_bits.to_le_bytes());
    data[2..4].copy_from_slice(&dmin_bits.to_le_bytes());

    // Set scales to encode scale=1, min=0 for all groups (ggml packing).
    let scales = &mut data[4..16];
    scales.fill(0);
    for j in 0..8 {
        pack_scale_min_k4(scales, j, 1, 0);
    }

    // Set qh to 0 (no high bits)
    data[16..48].fill(0x00);

    // Set qs nibbles to 3
    data[48..176].fill(0x33);

    let result = dequantize_q5_k(&data, 256).unwrap();
    assert_eq!(result.len(), 256);
    for v in &result {
        assert!((v - 3.0).abs() < 1e-5, "Expected ~3.0, got {}", v);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q6_k_basic() {
    // Q6_K: 210 bytes per 256 elements
    // Layout: ql[128], qh[64], scales[16], d(2)
    let mut data = vec![0u8; 210];

    // Set d = 1.0
    let d_bits = f32_to_half(1.0);
    data[208..210].copy_from_slice(&d_bits.to_le_bytes());

    // Set scales to 1
    data[192..208].fill(1);

    // Set ql and qh to zeros: 6-bit value = 0, minus 32 = -32
    data[0..128].fill(0);
    data[128..192].fill(0);

    let result = dequantize_q6_k(&data, 256).unwrap();
    assert_eq!(result.len(), 256);
    for v in &result {
        assert!((v + 32.0).abs() < 0.5, "Expected ~-32, got {}", v);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_dequantize_q6_k_with_scales() {
    // Test Q6_K with varying scales
    let mut data = vec![0u8; 210];

    // Set d = 1.0
    let d_bits = f32_to_half(1.0);
    data[208..210].copy_from_slice(&d_bits.to_le_bytes());

    // Set scales to 2 (signed i8)
    data[192..208].fill(2);

    // Set ql to all zeros, qh to all zeros
    // So 6-bit value = 0, minus 32 = -32
    // Result = d * scale * (0 - 32) = 1 * 2 * -32 = -64

    let result = dequantize_q6_k(&data, 256).unwrap();
    assert_eq!(result.len(), 256);

    // Check that values are close to expected
    for v in &result {
        assert!((v - (-64.0)).abs() < 0.5, "Expected ~-64, got {}", v);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_k_quant_data_too_short() {
    // Q2_K needs 84 bytes per 256 elements
    let data = vec![0u8; 83]; // 1 byte short
    let result = dequantize_q2_k(&data, 256);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too short"));

    // Q3_K needs 110 bytes per 256 elements
    let data = vec![0u8; 109];
    let result = dequantize_q3_k(&data, 256);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too short"));

    // Q4_K needs 144 bytes per 256 elements
    let data = vec![0u8; 143];
    let result = dequantize_q4_k(&data, 256);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too short"));

    // Q5_K needs 176 bytes per 256 elements
    let data = vec![0u8; 175];
    let result = dequantize_q5_k(&data, 256);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too short"));

    // Q6_K needs 210 bytes per 256 elements
    let data = vec![0u8; 209];
    let result = dequantize_q6_k(&data, 256);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too short"));
}

#[ntest::timeout(5000)]
#[test]
fn test_k_quant_data_too_short_multi_block() {
    // 2 blocks expected, but data is 1 byte short.
    let data = vec![0u8; 84 * 2 - 1];
    let result = dequantize_q2_k(&data, 512);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too short"));

    let data = vec![0u8; 110 * 2 - 1];
    let result = dequantize_q3_k(&data, 512);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too short"));

    let data = vec![0u8; 144 * 2 - 1];
    let result = dequantize_q4_k(&data, 512);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too short"));

    let data = vec![0u8; 176 * 2 - 1];
    let result = dequantize_q5_k(&data, 512);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too short"));

    let data = vec![0u8; 210 * 2 - 1];
    let result = dequantize_q6_k(&data, 512);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too short"));
}

#[ntest::timeout(5000)]
#[test]
fn test_k_quant_invalid_element_count() {
    // K-quants require elements divisible by 256
    let data_q2 = vec![0u8; 84];
    let result = dequantize_q2_k(&data_q2, 255); // Not divisible by 256
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("divisible by 256"));

    let data_q3 = vec![0u8; 110];
    let result = dequantize_q3_k(&data_q3, 255); // Not divisible by 256
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("divisible by 256"));

    let data_q4 = vec![0u8; 144];
    let result = dequantize_q4_k(&data_q4, 255); // Not divisible by 256
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("divisible by 256"));

    let data_q5 = vec![0u8; 176];
    let result = dequantize_q5_k(&data_q5, 255); // Not divisible by 256
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("divisible by 256"));

    let data_q6 = vec![0u8; 210];
    let result = dequantize_q6_k(&data_q6, 255); // Not divisible by 256
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("divisible by 256"));

    let result = dequantize_q2_k(&data_q2, 128); // Not divisible by 256
    assert!(result.is_err());
}

#[ntest::timeout(5000)]
#[test]
fn test_k_quant_multiple_blocks() {
    // Test with 2 super-blocks (512 elements)
    let data = vec![0u8; 84 * 2]; // 2 Q2_K blocks
    let result = dequantize_q2_k(&data, 512).unwrap();
    assert_eq!(result.len(), 512);

    let data = vec![0u8; 110 * 2]; // 2 Q3_K blocks
    let result = dequantize_q3_k(&data, 512).unwrap();
    assert_eq!(result.len(), 512);

    let data = vec![0u8; 144 * 2]; // 2 Q4_K blocks
    let result = dequantize_q4_k(&data, 512).unwrap();
    assert_eq!(result.len(), 512);

    let mut data = vec![0u8; 176 * 2]; // 2 Q5_K blocks
    for block in 0..2 {
        let base = block * 176;
        let d_bits = f32_to_half(1.0);
        data[base..base + 2].copy_from_slice(&d_bits.to_le_bytes());
        data[base + 2..base + 4].copy_from_slice(&0u16.to_le_bytes());
        let scales = &mut data[base + 4..base + 16];
        scales.fill(0);
        for j in 0..8 {
            pack_scale_min_k4(scales, j, 1, 0);
        }
        data[base + 16..base + 48].fill(0x00);
        let fill = if block == 0 { 0x22 } else { 0x33 };
        data[base + 48..base + 176].fill(fill);
    }
    let result = dequantize_q5_k(&data, 512).unwrap();
    assert_eq!(result.len(), 512);
    assert!(
        (result[0] - 2.0).abs() < 1e-5,
        "Expected ~2.0, got {}",
        result[0]
    );
    assert!(
        (result[255] - 2.0).abs() < 1e-5,
        "Expected ~2.0, got {}",
        result[255]
    );
    assert!(
        (result[256] - 3.0).abs() < 1e-5,
        "Expected ~3.0, got {}",
        result[256]
    );
    assert!(
        (result[511] - 3.0).abs() < 1e-5,
        "Expected ~3.0, got {}",
        result[511]
    );

    let data = vec![0u8; 210 * 2]; // 2 Q6_K blocks
    let result = dequantize_q6_k(&data, 512).unwrap();
    assert_eq!(result.len(), 512);
}

// ==========================================================================
// Overflow regression tests (#3280)
//
// Verify that checked arithmetic in load_tensor_data rejects crafted inputs
// that would overflow usize. Without these guards, integer overflow in
// untrusted file parsers could bypass bounds checks.
// ==========================================================================

/// Regression test (#3280): data_section_offset + tensor.offset overflows usize.
///
/// Crafts a GGUFTensorInfo with offset near usize::MAX so that
/// `data_section_offset.checked_add(tensor.offset)` returns None.
#[ntest::timeout(5000)]
#[test]
fn test_gguf_offset_overflow_rejected_3280() {
    let file_data = vec![0u8; 64];
    let tensor = GGUFTensorInfo {
        name: "overflow.weight".to_string(),
        dimensions: vec![4],
        tensor_type: GGMLType::F32,
        offset: u64::MAX, // forces checked_add(usize::MAX) to overflow
    };
    let result = load_tensor_data(&file_data, 1, &tensor, 4);
    assert!(result.is_err(), "#3280: offset overflow must be rejected");
    assert!(
        result.unwrap_err().contains("offset overflow"),
        "#3280: error message should mention offset overflow"
    );
}

/// Regression test (#3280): elements * 4 overflows usize for F32 tensors.
///
/// Passes element count near usize::MAX/4 so that `elements.checked_mul(4)`
/// returns None.
#[ntest::timeout(5000)]
#[test]
fn test_gguf_f32_byte_size_overflow_rejected_3280() {
    let file_data = vec![0u8; 64];
    let tensor = GGUFTensorInfo {
        name: "overflow.weight".to_string(),
        dimensions: vec![usize::MAX as u64 / 4 + 1],
        tensor_type: GGMLType::F32,
        offset: 0,
    };
    let elements = usize::MAX / 4 + 1;
    let result = load_tensor_data(&file_data, 0, &tensor, elements);
    assert!(
        result.is_err(),
        "#3280: F32 byte size overflow must be rejected"
    );
    assert!(
        result.unwrap_err().contains("byte size overflow"),
        "#3280: error message should mention byte size overflow"
    );
}

/// Regression test (#3280): checked_shape_product rejects overflowing tensor dimensions.
///
/// Verifies that `checked_shape_product` returns None for shapes whose
/// product exceeds usize::MAX, which the GGUF loader maps to NyError.
#[ntest::timeout(5000)]
#[test]
fn test_gguf_shape_product_overflow_rejected_3280() {
    use ny_core::checked_shape_product;
    // Two large dimensions whose product overflows: (2^32) * (2^32) = 2^64 > usize::MAX
    let shape = vec![1usize << 32, 1usize << 32];
    assert_eq!(
        checked_shape_product(&shape),
        None,
        "#3280: shape product overflow must return None"
    );
    // Single dimension at usize::MAX with multiplier 2
    let shape2 = vec![usize::MAX, 2];
    assert_eq!(
        checked_shape_product(&shape2),
        None,
        "#3280: usize::MAX * 2 must overflow"
    );
    // Sanity: normal shape should succeed
    let shape3 = vec![4, 8, 16];
    assert_eq!(checked_shape_product(&shape3), Some(512));
}

/// Regression test (#3280): elements * 2 overflows usize for F16 tensors.
///
/// Passes element count near usize::MAX/2 so that `elements.checked_mul(2)`
/// returns None in the F16 branch of load_tensor_data.
#[ntest::timeout(5000)]
#[test]
fn test_gguf_f16_byte_size_overflow_rejected_3280() {
    let file_data = vec![0u8; 64];
    let tensor = GGUFTensorInfo {
        name: "overflow_f16.weight".to_string(),
        dimensions: vec![usize::MAX as u64 / 2 + 1],
        tensor_type: GGMLType::F16,
        offset: 0,
    };
    let elements = usize::MAX / 2 + 1;
    let result = load_tensor_data(&file_data, 0, &tensor, elements);
    assert!(
        result.is_err(),
        "#3280: F16 byte size overflow must be rejected"
    );
    assert!(
        result.unwrap_err().contains("byte size overflow"),
        "#3280: error message should mention byte size overflow"
    );
}

/// Regression test (#3280): num_blocks * block_size overflows usize for quantized tensors.
///
/// For Q8_0: block_elements=32, block_size=34. Constructs an element count
/// that is divisible by 32 but whose (elements/32)*34 overflows usize.
#[ntest::timeout(5000)]
#[test]
fn test_gguf_quantized_byte_size_overflow_rejected_3280() {
    let file_data = vec![0u8; 64];

    // Q8_0: block_elements=32, block_size=34
    // We need num_blocks * 34 > usize::MAX.
    // num_blocks = elements / 32, so we need (elements / 32) * 34 > usize::MAX.
    // Choose elements = ((usize::MAX / 34) + 1) * 32 which makes num_blocks overflow on mul.
    let num_blocks_overflow = usize::MAX / 34 + 1;
    let elements = num_blocks_overflow * 32;
    // Verify our elements value is divisible by block_elements (32)
    assert_eq!(
        elements % 32,
        0,
        "elements must be divisible by Q8_0 block_elements"
    );

    let tensor = GGUFTensorInfo {
        name: "overflow_q8.weight".to_string(),
        dimensions: vec![elements as u64],
        tensor_type: GGMLType::Q8_0,
        offset: 0,
    };
    let result = load_tensor_data(&file_data, 0, &tensor, elements);
    assert!(
        result.is_err(),
        "#3280: quantized byte size overflow must be rejected"
    );
    assert!(
        result.unwrap_err().contains("byte size overflow"),
        "#3280: error message should mention byte size overflow"
    );
}
