// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use flate2::read::GzDecoder;
use ny_core::{NyError, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub fn read_bytes_maybe_gzip(path: &Path) -> Result<Vec<u8>> {
    if !path.exists() {
        return Err(NyError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    let is_gzip = path.extension().and_then(|e| e.to_str()) == Some("gz");
    if !is_gzip {
        return std::fs::read(path)
            .map_err(|e| NyError::ModelLoad(format!("Failed to read file: {}", e)));
    }

    let file =
        File::open(path).map_err(|e| NyError::ModelLoad(format!("Failed to open file: {}", e)))?;
    let mut decoder = GzDecoder::new(file);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| NyError::ModelLoad(format!("Failed to decode gzip: {}", e)))?;
    Ok(out)
}

pub fn read_string_maybe_gzip(path: &Path) -> Result<String> {
    let bytes = read_bytes_maybe_gzip(path)?;
    String::from_utf8(bytes)
        .map_err(|e| NyError::ModelLoad(format!("Failed to decode UTF-8: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{write::GzEncoder, Compression};
    use std::io::Write;
    use tempfile::{NamedTempFile, TempPath};

    // Explicitly select AsRef<Path> to avoid TempPath inference ambiguity in tests.
    fn temp_path_ref(path: &TempPath) -> &Path {
        <TempPath as AsRef<Path>>::as_ref(path)
    }

    fn write_temp_file(bytes: &[u8]) -> TempPath {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(bytes).unwrap();
        file.flush().unwrap();
        file.into_temp_path()
    }

    fn write_temp_gz_file(bytes: &[u8]) -> TempPath {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(bytes).unwrap();
        let gz_bytes = encoder.finish().unwrap();

        let mut file = tempfile::Builder::new().suffix(".gz").tempfile().unwrap();
        file.write_all(&gz_bytes).unwrap();
        file.flush().unwrap();
        file.into_temp_path()
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_read_bytes_plain() {
        let file = write_temp_file(b"hello");
        let bytes = read_bytes_maybe_gzip(temp_path_ref(&file)).unwrap();
        assert_eq!(bytes, b"hello");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_read_bytes_gzip() {
        let file = write_temp_gz_file(b"hello gzip");
        let bytes = read_bytes_maybe_gzip(temp_path_ref(&file)).unwrap();
        assert_eq!(bytes, b"hello gzip");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_read_string_plain() {
        let file = write_temp_file(b"hello");
        let s = read_string_maybe_gzip(temp_path_ref(&file)).unwrap();
        assert_eq!(s, "hello");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_read_string_gzip() {
        let file = write_temp_gz_file(b"hello gzip");
        let s = read_string_maybe_gzip(temp_path_ref(&file)).unwrap();
        assert_eq!(s, "hello gzip");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_missing_file_is_error() {
        let err = read_bytes_maybe_gzip(Path::new("/nonexistent/file.bin"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("File not found"), "{err}");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_invalid_gzip_is_error() {
        let mut file = tempfile::Builder::new().suffix(".gz").tempfile().unwrap();
        file.write_all(b"not a gzip stream").unwrap();
        file.flush().unwrap();
        let path = file.into_temp_path();

        let err = read_bytes_maybe_gzip(temp_path_ref(&path))
            .unwrap_err()
            .to_string();
        assert!(err.contains("Failed to decode gzip"), "{err}");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_invalid_utf8_is_error() {
        let file = write_temp_file(&[0xff, 0xfe, 0xfd]);
        let err = read_string_maybe_gzip(temp_path_ref(&file))
            .unwrap_err()
            .to_string();
        assert!(err.contains("Failed to decode UTF-8"), "{err}");
    }
}
