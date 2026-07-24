// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use memmap2::Mmap;
use ny_core::{NyError, Result};
use std::{fs::File, path::Path};

/// Create a read-only memory map for an immutable GGUF file.
///
/// `memmap2` marks file-backed mappings as `unsafe` because dereferencing the
/// map is UB if the underlying file is modified while the map is live, whether
/// by this process or another one. Callers must only use this helper for GGUF
/// files that are stable for the full lifetime of the returned map.
pub(super) fn map_read_only_gguf(file: &File, path: &Path) -> Result<Mmap> {
    // SAFETY: `file` is opened read-only and the returned map stays scoped to
    // the current loader/inspector call. The caller also guarantees the GGUF
    // path is immutable while the map is live, matching memmap2's contract for
    // file-backed mappings under concurrent writers.
    #[allow(unsafe_code)]
    unsafe { Mmap::map(file) }.map_err(|e| {
        NyError::ModelLoad(format!(
            "Failed to mmap GGUF file '{}': {}",
            path.display(),
            e
        ))
    })
}
