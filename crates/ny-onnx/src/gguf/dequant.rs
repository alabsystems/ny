// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::safetensors::half_to_f32;
use gguf::GGMLType;

// =============================================================================
// GGML Dequantization
// =============================================================================
// Implements dequantization for common GGML quantization formats.
// Reference: llama.cpp ggml-quants.c
//
// Block sizes:
// - Q8_0: 32 elements, 34 bytes (2 byte half scale + 32 int8 quants)
// - Q4_0: 32 elements, 18 bytes (2 byte half scale + 16 bytes nibbles)
// - Q4_1: 32 elements, 20 bytes (2 byte half scale + 2 byte half min + 16 bytes nibbles)
// - Q5_0: 32 elements, 22 bytes (2 byte half scale + 4 bytes high bits + 16 bytes low nibbles)
// - Q5_1: 32 elements, 24 bytes (2 byte half d + 2 byte half m + 4 bytes high + 16 bytes low)
// - Q8_1: 32 elements, 36 bytes (2 byte half d + 2 byte half s + 32 int8 quants)

const QK8_0: usize = 32; // Elements per Q8_0 block
const QK4_0: usize = 32; // Elements per Q4_0 block
const QK4_1: usize = 32; // Elements per Q4_1 block
const QK5_0: usize = 32; // Elements per Q5_0 block
const QK5_1: usize = 32; // Elements per Q5_1 block
const QK8_1: usize = 32; // Elements per Q8_1 block

// =============================================================================
// K-Quant Constants
// =============================================================================
// K-quants use super-blocks of 256 elements with per-group scales.
// Reference: llama.cpp ggml-quants.c, ggml-common.h

const QK_K: usize = 256; // Elements per K-quant super-block

// K-quant block sizes (bytes per super-block of 256 elements):
// Q2_K: scales[16] + qs[64] + d(2) + dmin(2) = 84 bytes
// Q3_K: hmask[32] + qs[64] + scales[12] + d(2) = 110 bytes
// Q4_K: d(2) + dmin(2) + scales[12] + qs[128] = 144 bytes
// Q5_K: d(2) + dmin(2) + scales[12] + qh[32] + qs[128] = 176 bytes
// Q6_K: ql[128] + qh[64] + scales[16] + d(2) = 210 bytes

/// Dequantize Q8_0 format: 32 elements per block, each stored as int8.
/// Block layout: [f16 delta (2 bytes)][32 x int8 quants]
/// Formula: y[i] = qs[i] * d
pub(super) fn dequantize_q8_0(data: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 2 + QK8_0; // 34 bytes per block

    if !elements.is_multiple_of(QK8_0) {
        return Err(format!(
            "Q8_0 requires element count divisible by {}, got {}",
            QK8_0, elements
        ));
    }

    let num_blocks = elements / QK8_0;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q8_0 data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;

        // Read f16 delta (scale)
        let d_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let d = half_to_f32(d_bits);

        // Dequantize each int8 value
        for j in 0..QK8_0 {
            let qs = data[block_start + 2 + j] as i8;
            result.push(qs as f32 * d);
        }
    }

    Ok(result)
}

/// Dequantize Q4_0 format: 32 elements per block, packed as nibbles.
/// Block layout: [f16 delta (2 bytes)][16 x uint8 nibble pairs]
/// Formula: y[i] = (nibble - 8) * d
pub(super) fn dequantize_q4_0(data: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 2 + QK4_0 / 2; // 18 bytes per block

    if !elements.is_multiple_of(QK4_0) {
        return Err(format!(
            "Q4_0 requires element count divisible by {}, got {}",
            QK4_0, elements
        ));
    }

    let num_blocks = elements / QK4_0;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q4_0 data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;

        // Read f16 delta (scale)
        let d_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let d = half_to_f32(d_bits);

        // Unpack nibbles: lower nibble goes to first half, upper to second half
        // Note: llama.cpp interleaves: y[j] and y[j + qk/2]
        let half_qk = QK4_0 / 2;
        // First pass: lower nibbles (first half of block output)
        for j in 0..half_qk {
            let byte = data[block_start + 2 + j];
            let x0 = (byte & 0x0F) as i32 - 8;
            result.push(x0 as f32 * d);
        }
        // Second pass: upper nibbles (second half of block output)
        for j in 0..half_qk {
            let byte = data[block_start + 2 + j];
            let x1 = (byte >> 4) as i32 - 8;
            result.push(x1 as f32 * d);
        }
    }

    Ok(result)
}

/// Dequantize Q4_1 format: 32 elements per block, packed as nibbles with min.
/// Block layout: [f16 d (2 bytes)][f16 m (2 bytes)][16 x uint8 nibble pairs]
/// Formula: y[i] = nibble * d + m
pub(super) fn dequantize_q4_1(data: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 2 + 2 + QK4_1 / 2; // 20 bytes per block

    if !elements.is_multiple_of(QK4_1) {
        return Err(format!(
            "Q4_1 requires element count divisible by {}, got {}",
            QK4_1, elements
        ));
    }

    let num_blocks = elements / QK4_1;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q4_1 data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;

        // Read f16 delta (scale) and min
        let d_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let m_bits = u16::from_le_bytes([data[block_start + 2], data[block_start + 3]]);
        let d = half_to_f32(d_bits);
        let m = half_to_f32(m_bits);

        // Unpack nibbles
        let half_qk = QK4_1 / 2;
        for j in 0..half_qk {
            let byte = data[block_start + 4 + j];
            let x0 = (byte & 0x0F) as f32;
            result.push(x0 * d + m);
        }
        for j in 0..half_qk {
            let byte = data[block_start + 4 + j];
            let x1 = (byte >> 4) as f32;
            result.push(x1 * d + m);
        }
    }

    Ok(result)
}

/// Dequantize Q5_0 format: 32 elements per block, 5 bits per element.
/// Block layout: [f16 d (2 bytes)][4 bytes high bits][16 bytes low nibbles]
/// Formula: y[i] = ((nibble | (high_bit << 4)) - 16) * d
pub(super) fn dequantize_q5_0(data: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 2 + 4 + QK5_0 / 2; // 22 bytes per block

    if !elements.is_multiple_of(QK5_0) {
        return Err(format!(
            "Q5_0 requires element count divisible by {}, got {}",
            QK5_0, elements
        ));
    }

    let num_blocks = elements / QK5_0;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q5_0 data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;

        // Read f16 delta (scale)
        let d_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let d = half_to_f32(d_bits);

        // High bits are packed in 4 bytes (32 bits)
        let mut high_bits: u32 = 0;
        for j in 0..4 {
            high_bits |= (data[block_start + 2 + j] as u32) << (j * 8);
        }

        // Low nibbles (16 bytes)
        let half_qk = QK5_0 / 2;
        for j in 0..half_qk {
            let byte = data[block_start + 6 + j];
            let low0 = (byte & 0x0F) as i32;
            let low1 = (byte >> 4) as i32;

            let high0 = ((high_bits >> j) & 1) as i32;
            let high1 = ((high_bits >> (j + half_qk)) & 1) as i32;

            let q0 = (low0 | (high0 << 4)) - 16;
            let q1 = (low1 | (high1 << 4)) - 16;

            result.push(q0 as f32 * d);
            result.push(q1 as f32 * d);
        }
    }

    Ok(result)
}

/// Dequantize Q5_1 format: 32 elements per block, 5 bits per element with min.
/// Block layout: [f16 d][f16 m][4 bytes high bits][16 bytes low nibbles]
/// Formula: y[i] = (q * d) + m
pub(super) fn dequantize_q5_1(data: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 2 + 2 + 4 + QK5_1 / 2; // 24 bytes per block

    if !elements.is_multiple_of(QK5_1) {
        return Err(format!(
            "Q5_1 requires element count divisible by {}, got {}",
            QK5_1, elements
        ));
    }

    let num_blocks = elements / QK5_1;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q5_1 data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;

        // Read f16 delta (scale) and min
        let d_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let m_bits = u16::from_le_bytes([data[block_start + 2], data[block_start + 3]]);
        let d = half_to_f32(d_bits);
        let m = half_to_f32(m_bits);

        // High bits are packed in 4 bytes
        let mut high_bits: u32 = 0;
        for j in 0..4 {
            high_bits |= (data[block_start + 4 + j] as u32) << (j * 8);
        }

        // Low nibbles (16 bytes)
        let half_qk = QK5_1 / 2;
        for j in 0..half_qk {
            let byte = data[block_start + 8 + j];
            let low0 = (byte & 0x0F) as i32;
            let low1 = (byte >> 4) as i32;

            let high0 = ((high_bits >> j) & 1) as i32;
            let high1 = ((high_bits >> (j + half_qk)) & 1) as i32;

            let q0 = (low0 | (high0 << 4)) as f32;
            let q1 = (low1 | (high1 << 4)) as f32;

            result.push(q0 * d + m);
            result.push(q1 * d + m);
        }
    }

    Ok(result)
}

/// Dequantize Q8_1 format: 32 elements per block, stored as int8 with sum.
/// Block layout: [f16 d][f16 s][32 x int8 quants]
/// Formula: y[i] = (qs[i] * d) + s / 32
pub(super) fn dequantize_q8_1(data: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 2 + 2 + QK8_1; // 36 bytes per block

    if !elements.is_multiple_of(QK8_1) {
        return Err(format!(
            "Q8_1 requires element count divisible by {}, got {}",
            QK8_1, elements
        ));
    }

    let num_blocks = elements / QK8_1;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q8_1 data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;

        // Read f16 delta and sum
        let d_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let s_bits = u16::from_le_bytes([data[block_start + 2], data[block_start + 3]]);
        let d = half_to_f32(d_bits);
        let s = half_to_f32(s_bits);

        let base = s / QK8_1 as f32;
        for j in 0..QK8_1 {
            let qs = data[block_start + 4 + j] as i8;
            result.push(qs as f32 * d + base);
        }
    }

    Ok(result)
}

pub(super) fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    debug_assert!(q.len() >= 12, "K4 scales require 12 bytes");
    debug_assert!(j < 8, "K4 scales index out of range");
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        // Matches ggml get_scale_min_k4 packing: low 4 bits in q[j+4],
        // high 2 bits in q[j-4] (scale) and q[j] (min).
        let d = (q[j + 4] & 0x0F) | (((q[j - 4] >> 6) & 0x03) << 4);
        let m = (q[j + 4] >> 4) | (((q[j] >> 6) & 0x03) << 4);
        (d, m)
    }
}

fn unpack_q3_k_scales(scales: &[u8]) -> [i8; 16] {
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0F0F0F0F;

    let mut aux = [0u32; 4];
    let mut word = [0u8; 4];
    word.copy_from_slice(&scales[0..4]);
    aux[0] = u32::from_le_bytes(word);
    word.copy_from_slice(&scales[4..8]);
    aux[1] = u32::from_le_bytes(word);
    word.copy_from_slice(&scales[8..12]);
    aux[2] = u32::from_le_bytes(word);

    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&aux[0].to_le_bytes());
    bytes[4..8].copy_from_slice(&aux[1].to_le_bytes());
    bytes[8..12].copy_from_slice(&aux[2].to_le_bytes());
    bytes[12..16].copy_from_slice(&aux[3].to_le_bytes());

    let mut out = [0i8; 16];
    for (dst, src) in out.iter_mut().zip(bytes.iter()) {
        *dst = *src as i8;
    }
    out
}

pub(super) fn dequantize_q2_k(data: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    if !elements.is_multiple_of(QK_K) {
        return Err(format!(
            "Q2_K requires element count divisible by {}, got {}",
            QK_K, elements
        ));
    }

    const BLOCK_SIZE: usize = 84;
    let num_blocks = elements / QK_K;
    let expected_bytes = num_blocks * BLOCK_SIZE;
    if data.len() < expected_bytes {
        return Err(format!(
            "Q2_K data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(elements);
    let mut offset = 0usize;

    for _ in 0..num_blocks {
        let scales = &data[offset..offset + 16];
        let qs = &data[offset + 16..offset + 80];
        let d_bits = u16::from_le_bytes([data[offset + 80], data[offset + 81]]);
        let dmin_bits = u16::from_le_bytes([data[offset + 82], data[offset + 83]]);
        let d = half_to_f32(d_bits);
        let dmin = half_to_f32(dmin_bits);

        for (group, &scale_byte) in scales[..16].iter().enumerate() {
            let scale_nibble = scale_byte & 0x0F;
            let min_nibble = (scale_byte >> 4) & 0x0F;
            let scale = scale_nibble as f32;
            let min = min_nibble as f32;
            let base = -dmin * min;

            for i in 0..16 {
                let idx = group * 16 + i;
                let qbyte = qs[idx / 4];
                let shift = (idx % 4) * 2;
                let q = ((qbyte >> shift) & 0x03) as f32;
                out.push(d * scale * q + base);
            }
        }

        offset += BLOCK_SIZE;
    }

    Ok(out)
}

pub(super) fn dequantize_q3_k(data: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    if !elements.is_multiple_of(QK_K) {
        return Err(format!(
            "Q3_K requires element count divisible by {}, got {}",
            QK_K, elements
        ));
    }

    const BLOCK_SIZE: usize = 110;
    let num_blocks = elements / QK_K;
    let expected_bytes = num_blocks * BLOCK_SIZE;
    if data.len() < expected_bytes {
        return Err(format!(
            "Q3_K data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(elements);
    let mut offset = 0usize;

    for _ in 0..num_blocks {
        let hmask = &data[offset..offset + 32];
        let qs = &data[offset + 32..offset + 96];
        let scales = &data[offset + 96..offset + 108];
        let d_bits = u16::from_le_bytes([data[offset + 108], data[offset + 109]]);
        let d = half_to_f32(d_bits);

        let scales = unpack_q3_k_scales(scales);
        let mut q_offset = 0usize;
        let mut is = 0usize;
        let mut m: u8 = 1;

        for _ in 0..(QK_K / 128) {
            let mut shift = 0usize;
            for _ in 0..4 {
                let dl = d * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let q = ((qs[q_offset + l] >> shift) & 0x03) as i8;
                    let h = hmask[l] & m;
                    let q = q - if h != 0 { 0 } else { 4 };
                    out.push(dl * q as f32);
                }

                let dl = d * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let q = ((qs[q_offset + l + 16] >> shift) & 0x03) as i8;
                    let h = hmask[l + 16] & m;
                    let q = q - if h != 0 { 0 } else { 4 };
                    out.push(dl * q as f32);
                }

                shift += 2;
                m <<= 1;
            }

            q_offset += 32;
        }

        offset += BLOCK_SIZE;
    }

    Ok(out)
}

pub(super) fn dequantize_q4_k(data: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    if !elements.is_multiple_of(QK_K) {
        return Err(format!(
            "Q4_K requires element count divisible by {}, got {}",
            QK_K, elements
        ));
    }

    const BLOCK_SIZE: usize = 144;
    let num_blocks = elements / QK_K;
    let expected_bytes = num_blocks * BLOCK_SIZE;
    if data.len() < expected_bytes {
        return Err(format!(
            "Q4_K data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(elements);
    let mut offset = 0usize;

    for _ in 0..num_blocks {
        let d_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let dmin_bits = u16::from_le_bytes([data[offset + 2], data[offset + 3]]);
        let d = half_to_f32(d_bits);
        let dmin = half_to_f32(dmin_bits);
        let scales = &data[offset + 4..offset + 16];
        let qs = &data[offset + 16..offset + 144];

        let mut q_offset = 0usize;
        let mut is = 0usize;
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            for l in 0..32 {
                let byte = qs[q_offset + l];
                out.push(d1 * (byte & 0x0F) as f32 - m1);
            }
            for l in 0..32 {
                let byte = qs[q_offset + l];
                out.push(d2 * (byte >> 4) as f32 - m2);
            }

            q_offset += 32;
            is += 2;
        }

        offset += BLOCK_SIZE;
    }

    Ok(out)
}

pub(super) fn dequantize_q5_k(data: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    if !elements.is_multiple_of(QK_K) {
        return Err(format!(
            "Q5_K requires element count divisible by {}, got {}",
            QK_K, elements
        ));
    }

    const BLOCK_SIZE: usize = 176;
    let num_blocks = elements / QK_K;
    let expected_bytes = num_blocks * BLOCK_SIZE;
    if data.len() < expected_bytes {
        return Err(format!(
            "Q5_K data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(elements);
    let mut offset = 0usize;

    for _ in 0..num_blocks {
        let d_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let dmin_bits = u16::from_le_bytes([data[offset + 2], data[offset + 3]]);
        let d = half_to_f32(d_bits);
        let dmin = half_to_f32(dmin_bits);
        let scales = &data[offset + 4..offset + 16];
        let qh = &data[offset + 16..offset + 48];
        let qs = &data[offset + 48..offset + 176];

        let mut q_offset = 0usize;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            for l in 0..32 {
                let byte = qs[q_offset + l];
                let q = (byte & 0x0F) + if (qh[l] & u1) != 0 { 16 } else { 0 };
                out.push(d1 * q as f32 - m1);
            }
            for l in 0..32 {
                let byte = qs[q_offset + l];
                let q = (byte >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
                out.push(d2 * q as f32 - m2);
            }

            q_offset += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }

        offset += BLOCK_SIZE;
    }

    Ok(out)
}

pub(super) fn dequantize_q6_k(data: &[u8], elements: usize) -> Result<Vec<f32>, String> {
    if !elements.is_multiple_of(QK_K) {
        return Err(format!(
            "Q6_K requires element count divisible by {}, got {}",
            QK_K, elements
        ));
    }

    const BLOCK_SIZE: usize = 210;
    let num_blocks = elements / QK_K;
    let expected_bytes = num_blocks * BLOCK_SIZE;
    if data.len() < expected_bytes {
        return Err(format!(
            "Q6_K data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(elements);
    let mut offset = 0usize;

    for _ in 0..num_blocks {
        let ql = &data[offset..offset + 128];
        let qh = &data[offset + 128..offset + 192];
        let scales = &data[offset + 192..offset + 208];
        let d_bits = u16::from_le_bytes([data[offset + 208], data[offset + 209]]);
        let d = half_to_f32(d_bits);

        let mut ql_offset = 0usize;
        let mut qh_offset = 0usize;
        let mut sc_offset = 0usize;
        for _ in 0..(QK_K / 128) {
            let mut block = [0.0f32; 128];
            for l in 0..32 {
                let is = l / 16;
                let ql0 = ql[ql_offset + l];
                let ql1 = ql[ql_offset + l + 32];
                let qh_byte = qh[qh_offset + l];
                let q1 = (ql0 & 0x0F) | ((qh_byte & 0x03) << 4);
                let q2 = (ql1 & 0x0F) | (((qh_byte >> 2) & 0x03) << 4);
                let q3 = (ql0 >> 4) | (((qh_byte >> 4) & 0x03) << 4);
                let q4 = (ql1 >> 4) | (((qh_byte >> 6) & 0x03) << 4);

                let s0 = scales[sc_offset + is] as i8 as f32;
                let s1 = scales[sc_offset + is + 2] as i8 as f32;
                let s2 = scales[sc_offset + is + 4] as i8 as f32;
                let s3 = scales[sc_offset + is + 6] as i8 as f32;

                block[l] = d * s0 * (q1 as i8 - 32) as f32;
                block[l + 32] = d * s1 * (q2 as i8 - 32) as f32;
                block[l + 64] = d * s2 * (q3 as i8 - 32) as f32;
                block[l + 96] = d * s3 * (q4 as i8 - 32) as f32;
            }
            out.extend_from_slice(&block);

            ql_offset += 64;
            qh_offset += 32;
            sc_offset += 8;
        }

        offset += BLOCK_SIZE;
    }

    Ok(out)
}

/// Get block size in bytes for a quantized type.
pub(super) fn get_block_size(dtype: &GGMLType) -> Option<usize> {
    match dtype {
        GGMLType::Q8_0 => Some(2 + QK8_0),             // 34
        GGMLType::Q4_0 => Some(2 + QK4_0 / 2),         // 18
        GGMLType::Q4_1 => Some(2 + 2 + QK4_1 / 2),     // 20
        GGMLType::Q5_0 => Some(2 + 4 + QK5_0 / 2),     // 22
        GGMLType::Q5_1 => Some(2 + 2 + 4 + QK5_1 / 2), // 24
        GGMLType::Q8_1 => Some(2 + 2 + QK8_1),         // 36
        // K-quants: 256 elements per super-block
        GGMLType::Q2K => Some(84),  // scales[16] + qs[64] + d(2) + dmin(2)
        GGMLType::Q3K => Some(110), // hmask[32] + qs[64] + scales[12] + d(2)
        GGMLType::Q4K => Some(144), // d(2) + dmin(2) + scales[12] + qs[128]
        GGMLType::Q5K => Some(176), // d(2) + dmin(2) + scales[12] + qh[32] + qs[128]
        GGMLType::Q6K => Some(210), // ql[128] + qh[64] + scales[16] + d(2)
        _ => None,
    }
}

/// Get elements per block for a quantized type.
pub(super) fn get_block_elements(dtype: &GGMLType) -> Option<usize> {
    match dtype {
        GGMLType::Q8_0 => Some(QK8_0),
        GGMLType::Q4_0 => Some(QK4_0),
        GGMLType::Q4_1 => Some(QK4_1),
        GGMLType::Q5_0 => Some(QK5_0),
        GGMLType::Q5_1 => Some(QK5_1),
        GGMLType::Q8_1 => Some(QK8_1),
        // K-quants all use QK_K = 256 elements per super-block
        GGMLType::Q2K | GGMLType::Q3K | GGMLType::Q4K | GGMLType::Q5K | GGMLType::Q6K => Some(QK_K),
        _ => None,
    }
}
