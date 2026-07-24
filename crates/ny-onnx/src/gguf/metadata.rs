// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use gguf::GGUFMetadataValue;

/// Format a metadata value for display.
pub(super) fn format_metadata_value(value: &GGUFMetadataValue) -> String {
    match value {
        GGUFMetadataValue::Uint8(v) => v.to_string(),
        GGUFMetadataValue::Int8(v) => v.to_string(),
        GGUFMetadataValue::Uint16(v) => v.to_string(),
        GGUFMetadataValue::Int16(v) => v.to_string(),
        GGUFMetadataValue::Uint32(v) => v.to_string(),
        GGUFMetadataValue::Int32(v) => v.to_string(),
        GGUFMetadataValue::Float32(v) => v.to_string(),
        GGUFMetadataValue::Uint64(v) => v.to_string(),
        GGUFMetadataValue::Int64(v) => v.to_string(),
        GGUFMetadataValue::Float64(v) => v.to_string(),
        GGUFMetadataValue::Bool(v) => v.to_string(),
        GGUFMetadataValue::String(v) => v.clone(),
        GGUFMetadataValue::Array(arr) => format!("[{} elements]", arr.value.len()),
    }
}
