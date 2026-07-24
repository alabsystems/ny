// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

fn is_rust_fence(fence: &str) -> bool {
    let fence = fence.trim();
    if fence.is_empty() {
        return true;
    }
    let mut tokens = fence.split(',').map(str::trim);
    let first = tokens.next().unwrap_or("");
    if first.is_empty() || first == "rust" {
        return true;
    }
    matches!(
        first,
        "ignore"
            | "no_run"
            | "should_panic"
            | "compile_fail"
            | "edition2015"
            | "edition2018"
            | "edition2021"
            | "edition2024"
    )
}

fn scan_dir(dir: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            scan_dir(&path, offenders)?;
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let contents = fs::read_to_string(&path)?;
        let mut in_code = false;
        let mut check_code = false;
        for (index, line) in contents.lines().enumerate() {
            let trimmed = line.trim_start();
            let doc_line = if trimmed.starts_with("///") {
                Some(trimmed.trim_start_matches("///"))
            } else if trimmed.starts_with("//!") {
                Some(trimmed.trim_start_matches("//!"))
            } else {
                None
            };
            if let Some(doc_line) = doc_line {
                let doc = doc_line.trim_start();
                if doc.starts_with("```") {
                    if in_code {
                        in_code = false;
                        check_code = false;
                    } else {
                        in_code = true;
                        let fence = doc.trim_start_matches("```");
                        check_code = is_rust_fence(fence);
                    }
                } else if in_code && check_code && doc.contains("crate::") {
                    offenders.push((path.clone(), index + 1, doc.to_string()));
                }
            } else if in_code {
                in_code = false;
                check_code = false;
            }
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn doctest_examples_avoid_crate_imports() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src_dir = manifest_dir.join("src");
    let mut offenders = Vec::new();
    scan_dir(&src_dir, &mut offenders).expect("scan ny-propagate src dir");
    if !offenders.is_empty() {
        let mut message =
            String::from("Doctest Rust code blocks must avoid crate:: imports. Found:\n");
        for (path, line, text) in offenders {
            message.push_str(&format!("- {}:{}: {}\n", path.display(), line, text));
        }
        panic!("{message}");
    }
}
