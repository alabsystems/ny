// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Helper functions for types module.

use ny_core::{NyError, Result};

pub use ny_core::truncate_name;

/// Compute SHA256 hash of the first 64KB of a file for fast model identification.
///
/// Hashes only the first 64KB (65536 bytes) for performance: model files can be
/// gigabytes, but the header region contains format magic, tensor metadata, and
/// weight data that changes whenever the model is retrained or converted. This
/// gives collision resistance (SHA256) while keeping checkpoint hashing fast.
///
/// Returns a lowercase 64-character hex string (256 bits).
pub fn compute_model_hash(path: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|e| NyError::InvalidSpec(format!("Failed to open model for hashing: {}", e)))?;

    let mut buffer = vec![0u8; 65536]; // 64KB
    let bytes_read = file
        .read(&mut buffer)
        .map_err(|e| NyError::InvalidSpec(format!("Failed to read model for hashing: {}", e)))?;

    let mut hasher = Sha256::new();
    hasher.update(&buffer[..bytes_read]);
    let digest = hasher.finalize();

    // Format as lowercase hex (64 chars for SHA256)
    use std::fmt::Write;
    Ok(digest.iter().fold(String::with_capacity(64), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    }))
}

/// Simple ISO 8601 timestamp without chrono dependency.
pub(crate) fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Convert to UTC components (simplified, not accounting for leap seconds)
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since 1970-01-01
    let mut year = 1970i32;
    let mut remaining_days = days as i32;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let month_days = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for &days_in_month in &month_days {
        if remaining_days < days_in_month {
            break;
        }
        remaining_days -= days_in_month;
        month += 1;
    }
    let day = remaining_days + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

pub(crate) fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ntest::timeout(5000)]
    #[test]
    fn test_truncate_name_short() {
        let name = "layer0";
        assert_eq!(truncate_name(name, 20), "layer0");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_truncate_name_exact() {
        let name = "exactly_ten";
        assert_eq!(truncate_name(name, 11), "exactly_ten");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_truncate_name_long() {
        let name = "very_long_layer_name_that_needs_truncation";
        let truncated = truncate_name(name, 20);
        assert_eq!(truncated.len(), 20);
        assert!(truncated.starts_with("..."));
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_is_leap_year_common() {
        assert!(!is_leap_year(2023));
        assert!(!is_leap_year(2021));
        assert!(!is_leap_year(1999));
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_is_leap_year_divisible_by_4() {
        assert!(is_leap_year(2024));
        assert!(is_leap_year(2020));
        assert!(is_leap_year(2016));
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_is_leap_year_divisible_by_100_not_400() {
        assert!(!is_leap_year(1900));
        assert!(!is_leap_year(2100));
        assert!(!is_leap_year(2200));
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_is_leap_year_divisible_by_400() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(1600));
        assert!(is_leap_year(2400));
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_chrono_lite_now_format() {
        let timestamp = chrono_lite_now();
        // Should be ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
        assert!(timestamp.contains('T'));
        assert!(timestamp.ends_with('Z'));
        assert_eq!(timestamp.len(), 20);
        // Parse to verify structure
        let parts: Vec<&str> = timestamp[..10].split('-').collect();
        assert_eq!(parts.len(), 3);
        let year: i32 = parts[0].parse().unwrap();
        let month: i32 = parts[1].parse().unwrap();
        let day: i32 = parts[2].parse().unwrap();
        assert!(year >= 2020);
        assert!((1..=12).contains(&month));
        assert!((1..=31).contains(&day));
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_compute_model_hash_nonexistent() {
        let result = compute_model_hash(std::path::Path::new("/nonexistent/file.onnx"));
        assert!(result.is_err());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_compute_model_hash_returns_valid_sha256_hex() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("ny_hash_test_sha256");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model_a.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04])
            .unwrap();
        drop(f);

        let hash = compute_model_hash(&path).expect("hash should succeed");
        // SHA256 produces 64 lowercase hex characters
        assert_eq!(
            hash.len(),
            64,
            "SHA256 hex must be 64 chars, got {}",
            hash.len()
        );
        assert!(
            hash.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "hash must be lowercase hex: {hash}"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Regression test for #1930: differing model bytes must produce differing hashes.
    #[ntest::timeout(5000)]
    #[test]
    fn test_compute_model_hash_different_bytes_different_hash() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("ny_hash_test_diff");
        std::fs::create_dir_all(&dir).unwrap();

        let path_a = dir.join("model_a.bin");
        let path_b = dir.join("model_b.bin");

        let mut fa = std::fs::File::create(&path_a).unwrap();
        fa.write_all(&[0x00; 128]).unwrap();
        drop(fa);

        let mut fb = std::fs::File::create(&path_b).unwrap();
        fb.write_all(&[0xFF; 128]).unwrap();
        drop(fb);

        let hash_a = compute_model_hash(&path_a).expect("hash_a");
        let hash_b = compute_model_hash(&path_b).expect("hash_b");
        assert_ne!(
            hash_a, hash_b,
            "different file contents must produce different hashes"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Regression test for #1930: single-byte change must be detected.
    #[ntest::timeout(5000)]
    #[test]
    fn test_compute_model_hash_single_byte_change_detected() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("ny_hash_test_1byte");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.bin");

        // Write initial content
        let mut data = vec![0x42u8; 1024];
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&data)
            .unwrap();
        let hash_before = compute_model_hash(&path).expect("hash_before");

        // Flip one byte
        data[512] = 0x43;
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&data)
            .unwrap();
        let hash_after = compute_model_hash(&path).expect("hash_after");

        assert_ne!(
            hash_before, hash_after,
            "single-byte change must produce a different hash"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Verify that compute_model_hash produces a known SHA256 digest.
    /// SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    #[ntest::timeout(5000)]
    #[test]
    fn test_compute_model_hash_empty_file_known_digest() {
        let dir = std::env::temp_dir().join("ny_hash_test_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.bin");
        std::fs::File::create(&path).unwrap(); // 0 bytes

        let hash = compute_model_hash(&path).expect("hash of empty file");
        assert_eq!(
            hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "SHA256 of empty input must match known digest"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
