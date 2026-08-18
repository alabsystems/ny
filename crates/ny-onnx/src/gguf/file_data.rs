// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use std::{
    fs::{File, Metadata},
    path::Path,
    time::SystemTime,
};

#[cfg(unix)]
type PlatformFileIdentity = (u64, u64);
/// `(creation_time, file_attributes)` — see [`platform_file_identity`] for why
/// this rather than the volume-serial/file-index pair.
#[cfg(windows)]
type PlatformFileIdentity = (u64, u32);
#[cfg(not(any(unix, windows)))]
type PlatformFileIdentity = ();

/// Best-effort identity and mutation stamp for an open GGUF file.
///
/// A caller must still keep the model immutable while it is being inspected or
/// loaded: portable filesystems do not provide a mandatory snapshot read. The
/// stamp catches normal replacement, truncation, growth, and writes whose
/// modification time changes, turning those races into a load error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FileStamp {
    identity: PlatformFileIdentity,
    len: u64,
    modified: Option<SystemTime>,
}

impl FileStamp {
    pub(super) fn len(&self) -> u64 {
        self.len
    }
}

#[cfg(unix)]
fn platform_file_identity(metadata: &Metadata) -> PlatformFileIdentity {
    use std::os::unix::fs::MetadataExt;
    (metadata.dev(), metadata.ino())
}

/// Windows identity, from the STABLE surface of `MetadataExt`.
///
/// The exact analogue of Unix `(dev, ino)` is
/// `(volume_serial_number(), file_index())`, but both sit behind the unstable
/// `windows_by_handle` feature, so naming them does not compile on stable —
/// this crate did not build on Windows at all until they were replaced.
///
/// `creation_time()` is the strongest stable substitute, and it is NOT
/// sufficient for replacement detection — the claim previously recorded here,
/// that "a REPLACED file carries a new creation time", is false on NTFS.
/// FILE-SYSTEM TUNNELING reuses a deleted name's creation time when the name
/// reappears within ~15 s, so the replacement inherits it. Measured: creation
/// time, attributes and length all identical across a same-length
/// rename-replace.
///
/// This tuple therefore detects a MUTATED file only. Replacement is answered by
/// [`path_still_names_open_file`], which compares real file identity.
/// `file_attributes()` rides along to notice a type change (e.g. file swapped
/// for a reparse point).
#[cfg(windows)]
fn platform_file_identity(metadata: &Metadata) -> PlatformFileIdentity {
    use std::os::windows::fs::MetadataExt;
    (metadata.creation_time(), metadata.file_attributes())
}

#[cfg(not(any(unix, windows)))]
fn platform_file_identity(_metadata: &Metadata) -> PlatformFileIdentity {}

fn stamp_from_metadata(metadata: &Metadata) -> FileStamp {
    FileStamp {
        identity: platform_file_identity(metadata),
        len: metadata.len(),
        // Some virtual filesystems do not expose modification timestamps. In
        // that case identity and length checks still apply.
        modified: metadata.modified().ok(),
    }
}

pub(super) fn capture_file_stamp(file: &File, path: &Path) -> Result<FileStamp> {
    let metadata = file.metadata().map_err(|e| {
        NyError::ModelLoad(format!(
            "Failed to inspect GGUF file '{}': {e}",
            path.display()
        ))
    })?;
    Ok(stamp_from_metadata(&metadata))
}

/// Does `path` still name the SAME FILE as the open handle?
///
/// The metadata stamp cannot answer this on Windows. NTFS FILE-SYSTEM
/// TUNNELING reuses a deleted name's original creation time when the name
/// reappears within a short window (~15 s by default), so a replacement is
/// born carrying the ORIGINAL creation time. Measured on this repository's own
/// exhibit — write, open, rename a same-length file over it — the handle and
/// the replacement agreed on creation_time, file_attributes AND len, leaving
/// the stamp nothing to compare. `modified` did not separate them either: both
/// writes landed in one timer tick.
///
/// `same_file` asks the real question instead: `(volume serial, file index)` on
/// Windows via `GetFileInformationByHandle`, `(dev, ino)` on Unix — the exact
/// analogue std keeps behind the unstable `windows_by_handle` feature.
fn path_still_names_open_file(file: &File, path: &Path, operation: &str) -> Result<bool> {
    let identity_error = |what: &str, e: std::io::Error| {
        NyError::ModelLoad(format!(
            "Failed to resolve {what} identity for GGUF file '{}' while {operation}: {e}",
            path.display()
        ))
    };
    let open = file
        .try_clone()
        .and_then(same_file::Handle::from_file)
        .map_err(|e| identity_error("open-handle", e))?;
    let named = same_file::Handle::from_path(path).map_err(|e| identity_error("path", e))?;
    Ok(open == named)
}

/// Refuse a result if the open file or the path naming it changed during use.
pub(super) fn ensure_file_unchanged(
    file: &File,
    path: &Path,
    before: &FileStamp,
    operation: &str,
) -> Result<()> {
    let after = capture_file_stamp(file, path)?;
    let path_metadata = std::fs::metadata(path).map_err(|e| {
        NyError::ModelLoad(format!(
            "GGUF file '{}' became unavailable while {operation}: {e}",
            path.display()
        ))
    })?;
    let path_after = stamp_from_metadata(&path_metadata);

    if &after != before || path_after != after {
        return Err(NyError::ModelLoad(format!(
            "GGUF file '{}' changed while {operation}; refusing a potentially mixed read",
            path.display()
        )));
    }
    // Identity LAST, and separately: the stamp above catches a mutated file,
    // this catches a swapped one. On Windows the stamp provably cannot.
    if !path_still_names_open_file(file, path, operation)? {
        return Err(NyError::ModelLoad(format!(
            "GGUF file '{}' was REPLACED by a different file while {operation}; \
             refusing a potentially mixed read",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn stable_file_stamp_is_accepted() {
        let mut temp = tempfile::NamedTempFile::new().expect("temporary GGUF");
        temp.write_all(b"stable").expect("write fixture");
        temp.flush().expect("flush fixture");

        let path = temp.path().to_path_buf();
        let file = File::open(&path).expect("open fixture");
        let stamp = capture_file_stamp(&file, &path).expect("capture stamp");
        ensure_file_unchanged(&file, &path, &stamp, "testing").expect("stable file");
    }

    #[test]
    fn size_change_is_rejected() {
        let mut temp = tempfile::NamedTempFile::new().expect("temporary GGUF");
        temp.write_all(b"12345678").expect("write fixture");
        temp.flush().expect("flush fixture");

        let path = temp.path().to_path_buf();
        let file = File::open(&path).expect("open fixture");
        let stamp = capture_file_stamp(&file, &path).expect("capture stamp");
        temp.as_file_mut().set_len(2).expect("truncate fixture");

        let error = ensure_file_unchanged(&file, &path, &stamp, "testing")
            .expect_err("truncation must be rejected");
        assert!(matches!(error, NyError::ModelLoad(_)));
    }

    #[test]
    fn path_replacement_is_rejected() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("model.gguf");
        std::fs::write(&path, b"original").expect("write original");

        let file = File::open(&path).expect("open original");
        let stamp = capture_file_stamp(&file, &path).expect("capture stamp");
        let replacement = directory.path().join("replacement.gguf");
        std::fs::write(&replacement, b"replaced").expect("write replacement");
        std::fs::rename(&replacement, &path).expect("replace path");

        let error = ensure_file_unchanged(&file, &path, &stamp, "testing")
            .expect_err("path replacement must be rejected");
        assert!(matches!(error, NyError::ModelLoad(_)));
    }
}
