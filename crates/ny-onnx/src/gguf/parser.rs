// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use gguf::{GGMLType, GGUFTensorInfo};
use ny_core::{NyError, Result as NyResult};
use std::{
    fs::File,
    io::{BufReader, Read, Seek},
    mem::size_of,
    path::Path,
};

/// Hard bound for the metadata + tensor-descriptor prefix.
///
/// Typical GGUF descriptors are much smaller. The cap prevents a crafted file
/// from making `gguf_info` mirror an arbitrarily large tensor payload into RAM,
/// while leaving room for large tokenizer metadata.
const MAX_GGUF_DESCRIPTOR_BYTES: u64 = 256 * 1024 * 1024;
const MAX_GGUF_METADATA_ENTRIES: usize = 1_000_000;
const MAX_GGUF_TENSORS: usize = 1_000_000;
const MAX_GGUF_ARRAY_ENTRIES: usize = 10_000_000;
const MAX_GGUF_TENSOR_DIMS: usize = 1_024;
const GGUF_DESCRIPTOR_SITE: &str = "ny-onnx::gguf::stream_descriptor";

pub(in crate::gguf) struct StreamedGgufDescriptor {
    pub version: u32,
    pub architecture: Option<String>,
    pub model_name: Option<String>,
    pub metadata: Vec<(String, String)>,
    pub tensors: Vec<GGUFTensorInfo>,
    pub data_section_offset: u64,
    #[cfg(test)]
    pub descriptor_bytes: u64,
}

struct DescriptorReader<R> {
    inner: R,
    consumed: u64,
    limit: u64,
    file_len: u64,
    path: String,
}

impl<R: Read> DescriptorReader<R> {
    fn new(inner: R, file_len: u64, path: &Path) -> Self {
        Self {
            inner,
            consumed: 0,
            limit: file_len.min(MAX_GGUF_DESCRIPTOR_BYTES),
            file_len,
            path: path.display().to_string(),
        }
    }

    fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.consumed)
    }

    fn require_bytes(&self, len: u64, what: &str) -> NyResult<()> {
        let end = self.consumed.checked_add(len).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "GGUF {what} length overflows while parsing '{}'",
                self.path
            ))
        })?;
        if end <= self.limit {
            return Ok(());
        }
        if self.limit < self.file_len {
            Err(NyError::ModelLoad(format!(
                "GGUF descriptor in '{}' exceeds the {} MiB safety limit",
                self.path,
                MAX_GGUF_DESCRIPTOR_BYTES / (1024 * 1024)
            )))
        } else {
            Err(NyError::ModelLoad(format!(
                "Truncated GGUF descriptor in '{}' while reading {what}",
                self.path
            )))
        }
    }

    fn read_array<const N: usize>(&mut self, what: &str) -> NyResult<[u8; N]> {
        self.require_bytes(N as u64, what)?;
        let mut bytes = [0u8; N];
        self.inner.read_exact(&mut bytes).map_err(|e| {
            NyError::ModelLoad(format!(
                "Failed to read GGUF {what} from '{}': {e}",
                self.path
            ))
        })?;
        self.consumed += N as u64;
        Ok(bytes)
    }

    fn read_u8(&mut self, what: &str) -> NyResult<u8> {
        Ok(self.read_array::<1>(what)?[0])
    }

    fn read_u16(&mut self, what: &str) -> NyResult<u16> {
        Ok(u16::from_le_bytes(self.read_array(what)?))
    }

    fn read_u32(&mut self, what: &str) -> NyResult<u32> {
        Ok(u32::from_le_bytes(self.read_array(what)?))
    }

    fn read_u64(&mut self, what: &str) -> NyResult<u64> {
        Ok(u64::from_le_bytes(self.read_array(what)?))
    }

    fn read_bytes(&mut self, len: u64, what: &str) -> NyResult<Vec<u8>> {
        self.require_bytes(len, what)?;
        let len = usize::try_from(len).map_err(|_| {
            NyError::ModelLoad(format!(
                "GGUF {what} in '{}' is too large for this platform",
                self.path
            ))
        })?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: len,
                budget_bytes: usize::MAX,
                site: GGUF_DESCRIPTOR_SITE,
            })?;
        bytes.resize(len, 0);
        self.inner.read_exact(&mut bytes).map_err(|e| {
            NyError::ModelLoad(format!(
                "Failed to read GGUF {what} from '{}': {e}",
                self.path
            ))
        })?;
        self.consumed += len as u64;
        Ok(bytes)
    }

    fn read_string(&mut self, what: &str, version: u32) -> NyResult<String> {
        let len = if version == 1 {
            u64::from(self.read_u32(&format!("{what} length"))?)
        } else {
            self.read_u64(&format!("{what} length"))?
        };
        let bytes = self.read_bytes(len, what)?;
        String::from_utf8(bytes).map_err(|e| {
            NyError::ModelLoad(format!(
                "Invalid UTF-8 in GGUF {what} from '{}': {e}",
                self.path
            ))
        })
    }
}

fn checked_collection_len(raw: u64, maximum: usize, what: &str) -> NyResult<usize> {
    let value = usize::try_from(raw)
        .map_err(|_| NyError::ModelLoad(format!("GGUF {what} does not fit usize: {raw}")))?;
    if value > maximum {
        return Err(NyError::ModelLoad(format!(
            "GGUF {what} {value} exceeds safety limit {maximum}"
        )));
    }
    Ok(value)
}

fn read_metadata_value<R: Read>(
    reader: &mut DescriptorReader<R>,
    value_type: u32,
    want_display: bool,
    depth: usize,
    version: u32,
) -> NyResult<(Option<String>, Option<u64>)> {
    if depth > 4 {
        return Err(NyError::ModelLoad(
            "GGUF metadata nesting exceeds safety limit".into(),
        ));
    }

    let result = match value_type {
        0 => {
            let value = reader.read_u8("metadata u8")?;
            (want_display.then(|| value.to_string()), None)
        }
        1 => {
            let value = i8::from_le_bytes(reader.read_array("metadata i8")?);
            (want_display.then(|| value.to_string()), None)
        }
        2 => {
            let value = reader.read_u16("metadata u16")?;
            (want_display.then(|| value.to_string()), None)
        }
        3 => {
            let value = i16::from_le_bytes(reader.read_array("metadata i16")?);
            (want_display.then(|| value.to_string()), None)
        }
        4 => {
            let value = reader.read_u32("metadata u32")?;
            (
                want_display.then(|| value.to_string()),
                Some(u64::from(value)),
            )
        }
        5 => {
            let value = i32::from_le_bytes(reader.read_array("metadata i32")?);
            (want_display.then(|| value.to_string()), None)
        }
        6 => {
            let value = f32::from_le_bytes(reader.read_array("metadata f32")?);
            (want_display.then(|| value.to_string()), None)
        }
        7 => {
            let raw = reader.read_u8("metadata bool")?;
            let value = match raw {
                0 => false,
                1 => true,
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "Invalid GGUF boolean value {raw}"
                    )))
                }
            };
            (want_display.then(|| value.to_string()), None)
        }
        8 => {
            let value = reader.read_string("metadata string", version)?;
            (want_display.then_some(value), None)
        }
        9 => {
            let inner_type = reader.read_u32("metadata array element type")?;
            let raw_len = if version == 1 {
                u64::from(reader.read_u32("metadata array length")?)
            } else {
                reader.read_u64("metadata array length")?
            };
            let len =
                checked_collection_len(raw_len, MAX_GGUF_ARRAY_ENTRIES, "metadata array length")?;
            for _ in 0..len {
                let _ = read_metadata_value(reader, inner_type, false, depth + 1, version)?;
            }
            (want_display.then(|| format!("[{len} elements]")), None)
        }
        10 => {
            let value = reader.read_u64("metadata u64")?;
            (want_display.then(|| value.to_string()), Some(value))
        }
        11 => {
            let value = i64::from_le_bytes(reader.read_array("metadata i64")?);
            (want_display.then(|| value.to_string()), None)
        }
        12 => {
            let value = f64::from_le_bytes(reader.read_array("metadata f64")?);
            (want_display.then(|| value.to_string()), None)
        }
        _ => {
            return Err(NyError::ModelLoad(format!(
                "Unknown GGUF metadata value type: {value_type}"
            )))
        }
    };
    Ok(result)
}

fn is_interesting_metadata(key: &str) -> bool {
    key.starts_with("general.")
        || key.contains(".context_length")
        || key.contains(".embedding_length")
        || key.contains(".block_count")
        || key.contains(".attention.head_count")
}

fn reserve_one<T>(values: &mut Vec<T>, required_bytes: usize) -> NyResult<()> {
    values
        .try_reserve(1)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes: usize::MAX,
            site: GGUF_DESCRIPTOR_SITE,
        })
}

fn align_up_u64(value: u64, alignment: u64) -> NyResult<u64> {
    let alignment = alignment.max(1);
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| NyError::ModelLoad("GGUF data-section alignment overflows u64".into()))
    }
}

/// Stream just the GGUF metadata and tensor descriptors.
///
/// Tensor payload bytes are never read by this parser. The descriptor itself
/// has a hard byte/entry bound so malformed lengths cannot request unbounded
/// allocations or loops.
pub(in crate::gguf) fn read_streamed_gguf_descriptor(
    file: &mut File,
    path: &Path,
    file_len: u64,
) -> NyResult<StreamedGgufDescriptor> {
    file.rewind().map_err(|e| {
        NyError::ModelLoad(format!(
            "Failed to seek GGUF file '{}': {e}",
            path.display()
        ))
    })?;
    let buffered = BufReader::with_capacity(64 * 1024, file);
    let mut reader = DescriptorReader::new(buffered, file_len, path);

    if reader.read_array::<4>("magic")? != *b"GGUF" {
        return Err(NyError::ModelLoad("Invalid GGUF magic".into()));
    }
    let version = reader.read_u32("version")?;
    if !(1..=3).contains(&version) {
        return Err(NyError::ModelLoad(format!(
            "Unsupported GGUF version {version}; supported versions are 1 through 3"
        )));
    }
    let raw_tensor_count = if version == 1 {
        u64::from(reader.read_u32("tensor count")?)
    } else {
        reader.read_u64("tensor count")?
    };
    let raw_metadata_count = if version == 1 {
        u64::from(reader.read_u32("metadata count")?)
    } else {
        reader.read_u64("metadata count")?
    };
    let tensor_count = checked_collection_len(raw_tensor_count, MAX_GGUF_TENSORS, "tensor count")?;
    let metadata_count = checked_collection_len(
        raw_metadata_count,
        MAX_GGUF_METADATA_ENTRIES,
        "metadata count",
    )?;

    let mut architecture = None;
    let mut model_name = None;
    let mut metadata = Vec::new();
    let mut alignment = 32u64;

    for _ in 0..metadata_count {
        let key = reader.read_string("metadata key", version)?;
        let value_type = reader.read_u32("metadata value type")?;
        let interesting = is_interesting_metadata(&key);
        let (display, unsigned_value) =
            read_metadata_value(&mut reader, value_type, interesting, 0, version)?;

        if key == "general.alignment" && matches!(value_type, 4 | 10) {
            alignment = unsigned_value.expect("u32/u64 metadata carries unsigned value");
        }
        if key == "general.architecture" && value_type == 8 {
            architecture.clone_from(&display);
        }
        if key == "general.name" && value_type == 8 {
            model_name.clone_from(&display);
        }
        if interesting {
            let required_bytes = metadata
                .len()
                .saturating_add(1)
                .saturating_mul(size_of::<(String, String)>());
            reserve_one(&mut metadata, required_bytes)?;
            metadata.push((
                key,
                display.expect("interesting metadata requests a display value"),
            ));
        }
    }

    let minimum_tensor_size = if version == 1 { 20 } else { 24 };
    let minimum_tensor_bytes = raw_tensor_count
        .checked_mul(minimum_tensor_size)
        .ok_or_else(|| NyError::ModelLoad("GGUF tensor descriptor size overflows u64".into()))?;
    if minimum_tensor_bytes > reader.remaining() {
        return Err(NyError::ModelLoad(format!(
            "Truncated or oversized GGUF tensor descriptor table in '{}'",
            path.display()
        )));
    }

    let tensor_reservation = tensor_count
        .checked_mul(size_of::<GGUFTensorInfo>())
        .ok_or_else(|| NyError::ModelLoad("GGUF tensor table allocation overflows usize".into()))?;
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(tensor_count)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes: tensor_reservation,
            budget_bytes: usize::MAX,
            site: GGUF_DESCRIPTOR_SITE,
        })?;

    for _ in 0..tensor_count {
        let name = reader.read_string("tensor name", version)?;
        let dimension_count = checked_collection_len(
            u64::from(reader.read_u32("tensor dimension count")?),
            MAX_GGUF_TENSOR_DIMS,
            "tensor dimension count",
        )?;
        let dimension_width = if version == 1 {
            size_of::<u32>()
        } else {
            size_of::<u64>()
        };
        let dimension_bytes = dimension_count
            .checked_mul(dimension_width)
            .ok_or_else(|| {
                NyError::ModelLoad("GGUF dimension allocation overflows usize".into())
            })?;
        if dimension_bytes as u64 > reader.remaining() {
            return Err(NyError::ModelLoad(format!(
                "Truncated GGUF dimensions for tensor '{name}'"
            )));
        }
        let mut dimensions = Vec::new();
        dimensions
            .try_reserve_exact(dimension_count)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: dimension_bytes,
                budget_bytes: usize::MAX,
                site: GGUF_DESCRIPTOR_SITE,
            })?;
        for _ in 0..dimension_count {
            let dimension = if version == 1 {
                u64::from(reader.read_u32("tensor dimension")?)
            } else {
                reader.read_u64("tensor dimension")?
            };
            dimensions.push(dimension);
        }
        let raw_type = reader.read_u32("tensor type")?;
        let tensor_type = GGMLType::try_from(raw_type).map_err(|e| {
            NyError::ModelLoad(format!("Invalid GGUF tensor type for '{name}': {e}"))
        })?;
        let offset = reader.read_u64("tensor offset")?;
        tensors.push(GGUFTensorInfo {
            name,
            dimensions,
            tensor_type,
            offset,
        });
    }

    let data_section_offset = align_up_u64(reader.consumed, alignment)?;
    if data_section_offset > file_len {
        return Err(NyError::ModelLoad(format!(
            "GGUF data-section offset {data_section_offset} is beyond file end {file_len}"
        )));
    }
    let available_payload = file_len - data_section_offset;
    for tensor in &tensors {
        if tensor.offset > available_payload {
            return Err(NyError::ModelLoad(format!(
                "GGUF tensor '{}' offset {} is beyond the {}-byte data section",
                tensor.name, tensor.offset, available_payload
            )));
        }
    }
    Ok(StreamedGgufDescriptor {
        version,
        architecture,
        model_name,
        metadata,
        tensors,
        data_section_offset,
        #[cfg(test)]
        descriptor_bytes: reader.consumed,
    })
}

#[cfg(test)]
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

#[cfg(test)]
pub(in crate::gguf) fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if pos.checked_add(4).is_none_or(|end| end > data.len()) {
        return Err("Unexpected EOF while reading u32".to_string());
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

#[cfg(test)]
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

#[cfg(test)]
pub(in crate::gguf) fn read_gguf_string(data: &[u8], pos: &mut usize) -> Result<Vec<u8>, String> {
    let raw_len = read_u64_le(data, pos)?;
    let len = usize::try_from(raw_len)
        .map_err(|_| format!("GGUF string length {raw_len} does not fit usize"))?;
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

#[cfg(test)]
fn skip_bytes(data: &[u8], pos: &mut usize, n: usize) -> Result<(), String> {
    // `n` is derived from untrusted lengths (string/array values); checked add so
    // a crafted length fails closed instead of overflowing.
    if pos.checked_add(n).is_none_or(|end| end > data.len()) {
        return Err("Unexpected EOF while skipping bytes".to_string());
    }
    *pos += n;
    Ok(())
}

#[cfg(test)]
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
            let raw_len = read_u64_le(data, pos)?;
            let len = usize::try_from(raw_len)
                .map_err(|_| format!("GGUF string length {raw_len} does not fit usize"))?;
            skip_bytes(data, pos, len)
        }
        9 => {
            let inner_type = read_u32_le(data, pos)?;
            let raw_len = read_u64_le(data, pos)?;
            let len = usize::try_from(raw_len)
                .map_err(|_| format!("GGUF array length {raw_len} does not fit usize"))?;
            for _ in 0..len {
                skip_metadata_value(data, pos, inner_type, depth + 1)?;
            }
            Ok(())
        }
        other => Err(format!("Unknown GGUF metadata value type: {}", other)),
    }
}

#[cfg(test)]
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
    let raw_tensor_count = read_u64_le(file_data, &mut pos)?;
    let tensor_count = usize::try_from(raw_tensor_count)
        .map_err(|_| format!("GGUF tensor count {raw_tensor_count} does not fit usize"))?;
    let raw_metadata_count = read_u64_le(file_data, &mut pos)?;
    let metadata_count = usize::try_from(raw_metadata_count)
        .map_err(|_| format!("GGUF metadata count {raw_metadata_count} does not fit usize"))?;

    let mut alignment: usize = 32;

    for _ in 0..metadata_count {
        let key_bytes = read_gguf_string(file_data, &mut pos)?;
        let value_type = read_u32_le(file_data, &mut pos)?;

        if key_bytes == b"general.alignment" {
            match value_type {
                4 => alignment = read_u32_le(file_data, &mut pos)? as usize,
                10 => {
                    let raw_alignment = read_u64_le(file_data, &mut pos)?;
                    alignment = usize::try_from(raw_alignment).map_err(|_| {
                        format!("GGUF alignment {raw_alignment} does not fit usize")
                    })?;
                }
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
