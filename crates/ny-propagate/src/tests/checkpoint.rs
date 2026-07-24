// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint atomic write tests

use crate::{BlockBoundsInfo, VerificationCheckpoint};

// ============================================================
// CHECKPOINT ATOMIC WRITE TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_checkpoint_save_creates_file() {
    let temp_dir = std::env::temp_dir();
    let unique_id = std::process::id();
    let checkpoint_path = temp_dir.join(format!("test_checkpoint_save_{}.json", unique_id));

    // Ensure clean state
    let _ = std::fs::remove_file(&checkpoint_path);
    let temp_path = checkpoint_path.with_extension("json.tmp");
    let _ = std::fs::remove_file(&temp_path);

    let checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("test_model.gguf"),
        "abc123".to_string(),
        0.001,
        "ibp",
        "cpu",
        10,
    );

    checkpoint.save(&checkpoint_path).unwrap();

    assert!(checkpoint_path.exists(), "Checkpoint file should exist");
    assert!(!temp_path.exists(), "Temp file should not exist after save");

    // Cleanup
    let _ = std::fs::remove_file(&checkpoint_path);
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpoint_save_load_roundtrip() {
    let temp_dir = std::env::temp_dir();
    let unique_id = std::process::id();
    let checkpoint_path = temp_dir.join(format!("test_checkpoint_roundtrip_{}.json", unique_id));

    // Ensure clean state
    let _ = std::fs::remove_file(&checkpoint_path);

    let mut checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("test_model.gguf"),
        "abc123".to_string(),
        0.001,
        "ibp",
        "cpu",
        10,
    );

    // Add some blocks
    checkpoint.update(
        BlockBoundsInfo {
            block_index: 0,
            block_name: "block0".to_string(),
            nodes: vec![],
            input_width: 1.0,
            output_width: 1.5,
            sensitivity: 1.5,
            qk_matmul_width: None,
            swiglu_width: None,
            degraded: false,
        },
        1000,
    );

    checkpoint.save(&checkpoint_path).unwrap();

    // Load and verify
    let loaded = VerificationCheckpoint::load(&checkpoint_path).unwrap();
    assert_eq!(loaded.model_hash, "abc123");
    assert_eq!(loaded.epsilon, 0.001);
    assert_eq!(loaded.method, "ibp");
    assert_eq!(loaded.backend, "cpu");
    assert_eq!(loaded.total_blocks, 10);
    assert_eq!(loaded.next_block_index, 1);
    assert_eq!(loaded.completed_blocks.len(), 1);
    assert!((loaded.max_sensitivity - 1.5).abs() < 1e-6);

    // Cleanup
    let _ = std::fs::remove_file(&checkpoint_path);
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpoint_load_cleans_stale_temp_file() {
    let temp_dir = std::env::temp_dir();
    let unique_id = std::process::id();
    let checkpoint_path = temp_dir.join(format!("test_checkpoint_clean_temp_{}.json", unique_id));
    let temp_path = checkpoint_path.with_extension("json.tmp");

    // Create checkpoint file
    let checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("test_model.gguf"),
        "abc123".to_string(),
        0.001,
        "ibp",
        "cpu",
        10,
    );
    checkpoint.save(&checkpoint_path).unwrap();

    // Create stale temp file (simulating interrupted save)
    std::fs::write(&temp_path, "stale data").unwrap();
    assert!(temp_path.exists(), "Temp file should exist before load");

    // Load should clean up temp file and preserve content
    let loaded = VerificationCheckpoint::load(&checkpoint_path).unwrap();
    assert!(
        !temp_path.exists(),
        "Temp file should be cleaned up after load"
    );
    assert_eq!(
        loaded.epsilon, 0.001,
        "Loaded checkpoint should preserve epsilon"
    );

    // Cleanup
    let _ = std::fs::remove_file(&checkpoint_path);
}

#[ntest::timeout(10000)]
#[test]
fn test_checkpoint_atomic_no_temp_file_left() {
    let temp_dir = std::env::temp_dir();
    let unique_id = std::process::id();
    let checkpoint_path = temp_dir.join(format!("test_checkpoint_no_temp_{}.json", unique_id));
    let temp_path = checkpoint_path.with_extension("json.tmp");

    // Ensure clean state
    let _ = std::fs::remove_file(&checkpoint_path);
    let _ = std::fs::remove_file(&temp_path);

    let checkpoint = VerificationCheckpoint::new(
        std::path::PathBuf::from("test_model.gguf"),
        "abc123".to_string(),
        0.001,
        "ibp",
        "cpu",
        10,
    );

    // Multiple saves should not leave temp files
    for _ in 0..5 {
        checkpoint.save(&checkpoint_path).unwrap();
        assert!(checkpoint_path.exists(), "Checkpoint should exist");
        assert!(!temp_path.exists(), "Temp file should not exist after save");
    }

    // Cleanup
    let _ = std::fs::remove_file(&checkpoint_path);
}
