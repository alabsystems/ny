// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

pub(in crate::gguf) fn align_up(value: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        // `alignment` is attacker-controlled (GGUF `general.alignment`), so guard
        // the round-up against usize overflow rather than panicking. Saturating is
        // sound here: the data-section offset is bounds-checked against the file
        // length by the caller before any slice uses it.
        value.saturating_add(alignment - rem)
    }
}

pub(in crate::gguf) fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if pos.checked_add(4).is_none_or(|end| end > data.len()) {
        return Err("Unexpected EOF while reading u32".to_string());
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

pub(in crate::gguf) fn read_u64_le(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    if pos.checked_add(8).is_none_or(|end| end > data.len()) {
        return Err("Unexpected EOF while reading u64".to_string());
    }
    let v = u64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos += 8;
    Ok(v)
}

pub(in crate::gguf) fn read_gguf_string(data: &[u8], pos: &mut usize) -> Result<Vec<u8>, String> {
    let len = read_u64_le(data, pos)? as usize;
    // `len` is taken verbatim from the file (untrusted); use a checked add so a
    // crafted length cannot overflow `*pos + len` and abort the process — it must
    // fail closed with the structured EOF error instead.
    if pos.checked_add(len).is_none_or(|end| end > data.len()) {
        return Err("Unexpected EOF while reading GGUF string".to_string());
    }
    let bytes = data[*pos..*pos + len].to_vec();
    *pos += len;
    Ok(bytes)
}

fn skip_bytes(data: &[u8], pos: &mut usize, n: usize) -> Result<(), String> {
    // `n` is derived from untrusted lengths (string/array values); checked add so
    // a crafted length fails closed instead of overflowing.
    if pos.checked_add(n).is_none_or(|end| end > data.len()) {
        return Err("Unexpected EOF while skipping bytes".to_string());
    }
    *pos += n;
    Ok(())
}

fn skip_metadata_value(
    data: &[u8],
    pos: &mut usize,
    value_type: u32,
    depth: usize,
) -> Result<(), String> {
    if depth > 4 {
        return Err("GGUF metadata nesting too deep".to_string());
    }

    match value_type {
        0 | 1 | 7 => skip_bytes(data, pos, 1), // u8, i8, bool
        2 | 3 => skip_bytes(data, pos, 2),     // u16, i16
        4..=6 => skip_bytes(data, pos, 4),     // u32, i32, f32
        10..=12 => skip_bytes(data, pos, 8),   // u64, i64, f64
        8 => {
            let len = read_u64_le(data, pos)? as usize;
            skip_bytes(data, pos, len)
        }
        9 => {
            let inner_type = read_u32_le(data, pos)?;
            let len = read_u64_le(data, pos)? as usize;
            for _ in 0..len {
                skip_metadata_value(data, pos, inner_type, depth + 1)?;
            }
            Ok(())
        }
        other => Err(format!("Unknown GGUF metadata value type: {}", other)),
    }
}

pub(in crate::gguf) fn compute_data_section_offset(file_data: &[u8]) -> Result<usize, String> {
    // GGUF layout:
    // - Header + metadata + tensor infos
    // - Tensor data section begins after tensor infos, aligned to `general.alignment` (default 32).
    //
    // TensorInfo offsets are *relative* to the start of the data section, not absolute file
    // offsets. Many files have the first tensor offset = 0, so treating it as absolute would
    // read the GGUF header bytes as tensor data.
    let mut pos = 0usize;

    if file_data.len() < 4 || &file_data[0..4] != b"GGUF" {
        return Err("Invalid GGUF magic".to_string());
    }
    pos += 4;
    let _version = read_u32_le(file_data, &mut pos)?;
    let tensor_count = read_u64_le(file_data, &mut pos)? as usize;
    let metadata_count = read_u64_le(file_data, &mut pos)? as usize;

    let mut alignment: usize = 32;

    for _ in 0..metadata_count {
        let key_bytes = read_gguf_string(file_data, &mut pos)?;
        let value_type = read_u32_le(file_data, &mut pos)?;

        if key_bytes == b"general.alignment" {
            match value_type {
                4 => alignment = read_u32_le(file_data, &mut pos)? as usize,
                10 => alignment = read_u64_le(file_data, &mut pos)? as usize,
                _ => skip_metadata_value(file_data, &mut pos, value_type, 0)?,
            }
        } else {
            skip_metadata_value(file_data, &mut pos, value_type, 0)?;
        }
    }

    for _ in 0..tensor_count {
        let _name = read_gguf_string(file_data, &mut pos)?;
        let n_dimensions = read_u32_le(file_data, &mut pos)? as usize;
        // 8 * n_dimensions cannot overflow on 64-bit (n_dimensions <= u32::MAX),
        // but guard it so 32-bit targets fail closed rather than wrapping.
        let dims_bytes = 8usize
            .checked_mul(n_dimensions)
            .ok_or("GGUF tensor dimension count too large")?;
        skip_bytes(file_data, &mut pos, dims_bytes)?; // dims u64
        skip_bytes(file_data, &mut pos, 4)?; // tensor_type u32
        skip_bytes(file_data, &mut pos, 8)?; // offset u64
    }

    Ok(align_up(pos, alignment.max(1)))
}
