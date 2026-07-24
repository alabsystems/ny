// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ACAS-Xu benchmark problem discovery from instances CSV or file enumeration.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use super::{AcasxuBenchmarkArgs, AcasxuProblem, AcasxuProblemSet, DEFAULT_TIMEOUT_SECS};

/// Discover ACAS-Xu problems from the instances CSV or by enumeration.
pub(crate) fn discover_problems(args: &AcasxuBenchmarkArgs) -> Result<AcasxuProblemSet> {
    let bench_dir = resolve_benchmark_dir(args.year, "acasxu")?;
    let csv_path = bench_dir.join("instances.csv");
    let fallback_csv_path = bench_dir.join("acasxu_instances.csv");

    if csv_path.exists() {
        discover_from_csv(&csv_path, &bench_dir, args)
    } else if fallback_csv_path.exists() {
        discover_from_csv(&fallback_csv_path, &bench_dir, args)
    } else {
        discover_from_files(&bench_dir, args)
    }
}

/// Discover problems from the VNN-COMP instances CSV.
fn discover_from_csv(
    csv_path: &Path,
    bench_dir: &Path,
    args: &AcasxuBenchmarkArgs,
) -> Result<AcasxuProblemSet> {
    let content = std::fs::read_to_string(csv_path)
        .with_context(|| format!("Failed to read instances CSV: {}", csv_path.display()))?;

    let mut problems = Vec::new();
    let mut instance_count = 0usize;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() < 2 {
            continue;
        }

        let model_name = parts[0].trim();
        if model_name.eq_ignore_ascii_case("network") {
            continue;
        }
        let property_name = parts[1].trim();
        let parsed_timeout = parts
            .get(2)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);
        let timeout = args.timeout_override.unwrap_or(parsed_timeout);

        instance_count += 1;

        // Apply filters
        if let Some(ref filter) = args.model_filter {
            if !model_name.contains(filter) {
                continue;
            }
        }
        if let Some(ref filter) = args.property_filter {
            // Match "2" against "prop_2.vnnlib" or "vnnlib/prop_2.vnnlib".
            // CSV paths may include directory prefixes (e.g., "vnnlib/prop_2.vnnlib").
            // Extract just the filename before stripping prop_/suffix.
            let basename = property_name.rsplit('/').next().unwrap_or(property_name);
            let prop_num = basename
                .strip_prefix("prop_")
                .and_then(|s| s.strip_suffix(".vnnlib"))
                .unwrap_or(basename);
            if prop_num != filter {
                continue;
            }
        }

        let model_path = resolve_instance_path(bench_dir, model_name, "onnx");
        let property_path = resolve_instance_path(bench_dir, property_name, "vnnlib");

        if let (Some(model_path), Some(property_path)) = (model_path, property_path) {
            problems.push(AcasxuProblem {
                model_path,
                property_path,
                model_name: model_name.to_string(),
                property_name: property_name.to_string(),
                timeout,
            });
        }
    }

    Ok(AcasxuProblemSet {
        problems,
        instance_count,
    })
}

/// Discover problems by enumerating model and property files.
fn discover_from_files(bench_dir: &Path, args: &AcasxuBenchmarkArgs) -> Result<AcasxuProblemSet> {
    // Find all ONNX models
    let models: Vec<_> = std::fs::read_dir(bench_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "onnx"))
        .collect();

    // Find all property files
    let properties: Vec<_> = std::fs::read_dir(bench_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "vnnlib")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("prop_"))
        })
        .collect();

    let mut problems = Vec::new();
    let instance_count = models.len().saturating_mul(properties.len());
    for model_path in &models {
        let model_name = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        // Apply model filter
        if let Some(ref filter) = args.model_filter {
            if !model_name.contains(filter) {
                continue;
            }
        }

        for property_path in &properties {
            let property_name = property_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            // Apply property filter
            if let Some(ref filter) = args.property_filter {
                let prop_num = property_name
                    .strip_prefix("prop_")
                    .and_then(|s| s.strip_suffix(".vnnlib"))
                    .unwrap_or(&property_name);
                if prop_num != filter {
                    continue;
                }
            }

            problems.push(AcasxuProblem {
                model_path: model_path.clone(),
                property_path: property_path.clone(),
                model_name: model_name.clone(),
                property_name: property_name.clone(),
                timeout: args.timeout_override.unwrap_or(args.timeout),
            });
        }
    }

    Ok(AcasxuProblemSet {
        problems,
        instance_count,
    })
}

fn resolve_benchmark_dir(year: u32, benchmark: &str) -> Result<PathBuf> {
    let base = match year {
        2021 | 2023 | 2024 | 2025 => PathBuf::from(format!("benchmarks/vnncomp{year}/benchmarks")),
        _ => anyhow::bail!("Unsupported VNN-COMP year: {}", year),
    };

    if !base.exists() {
        anyhow::bail!(
            "Benchmark directory missing for year {}: {}",
            year,
            base.display()
        );
    }

    // VNN-COMP reuses category names across years (e.g., "acasxu_2023" in 2025 benchmarks).
    // Try multiple year suffixes to find the actual directory.
    let mut candidates = vec![
        base.join(benchmark),
        base.join(format!("{benchmark}_{year}")),
        base.join(format!("{benchmark}_{}", year.saturating_sub(1))),
    ];
    // Also try earlier year suffixes (categories may be named e.g. "acasxu_2023" in 2025)
    for offset in 2..=4 {
        candidates.push(base.join(format!("{benchmark}_{}", year.saturating_sub(offset))));
    }

    for path in candidates {
        if path.exists() {
            return Ok(path);
        }
    }

    anyhow::bail!(
        "No benchmark directory found for {} in {}",
        benchmark,
        base.display()
    );
}

fn resolve_instance_path(bench_dir: &Path, entry: &str, subdir: &str) -> Option<PathBuf> {
    let entry_path = Path::new(entry);
    let mut candidates = Vec::new();

    if entry_path.is_absolute() {
        candidates.push(entry_path.to_path_buf());
    } else {
        candidates.push(bench_dir.join(entry_path));

        let starts_with_subdir = entry_path
            .components()
            .next()
            .and_then(|c| c.as_os_str().to_str())
            .map(|c| c == subdir)
            .unwrap_or(false);
        if !starts_with_subdir {
            candidates.push(bench_dir.join(subdir).join(entry_path));
        }
    }

    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }

        if candidate
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gz"))
            .unwrap_or(false)
        {
            let without_gz = candidate.with_extension("");
            if without_gz.exists() {
                return Some(without_gz);
            }
        } else {
            let ext = candidate.extension().and_then(|e| e.to_str());
            let gz_ext = match ext {
                Some(existing) if !existing.is_empty() => format!("{existing}.gz"),
                _ => "gz".to_string(),
            };
            let gz_candidate = candidate.with_extension(gz_ext);
            if gz_candidate.exists() {
                return Some(gz_candidate);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_problems_with_filter() {
        // This test requires benchmark files to exist
        let args = AcasxuBenchmarkArgs {
            model_filter: Some("1_1".to_string()),
            property_filter: Some("1".to_string()),
            ..Default::default()
        };

        // Just verify it doesn't panic - actual results depend on file system
        let _ = discover_problems(&args);
    }

    // Issue #165: Year-aware benchmark discovery tests

    #[test]
    fn test_resolve_benchmark_dir_supported_years() {
        // Verify all supported years are accepted (2021, 2023, 2024, 2025)
        // This tests the year validation logic without requiring actual directories
        for year in [2021, 2023, 2024, 2025] {
            let result = resolve_benchmark_dir(year, "acasxu");
            // Result will be Err if directory doesn't exist, but should NOT be
            // "Unsupported VNN-COMP year" error
            if let Err(e) = result {
                let msg = e.to_string();
                assert!(
                    !msg.contains("Unsupported VNN-COMP year"),
                    "Year {} should be supported: {}",
                    year,
                    msg
                );
            }
        }
    }

    #[test]
    fn test_resolve_benchmark_dir_unsupported_year() {
        // Verify unsupported year returns appropriate error
        let result = resolve_benchmark_dir(2020, "acasxu");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Unsupported VNN-COMP year"),
            "Should reject year 2020: {}",
            msg
        );
    }

    #[test]
    fn test_resolve_instance_path_gz_fallback() {
        // Test the .gz fallback logic by verifying path resolution patterns
        use std::fs;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let bench_dir = dir.path();

        // Create onnx subdirectory
        let onnx_dir = bench_dir.join("onnx");
        fs::create_dir(&onnx_dir).unwrap();

        // Test 1: Direct file exists
        let model_file = onnx_dir.join("test_model.onnx");
        fs::write(&model_file, b"dummy").unwrap();
        let result = resolve_instance_path(bench_dir, "test_model.onnx", "onnx");
        assert!(result.is_some(), "Should find direct file");

        // Test 2: .gz fallback (create .gz, look for non-.gz)
        let gz_file = onnx_dir.join("test_model2.onnx.gz");
        fs::write(&gz_file, b"dummy gz").unwrap();
        let result = resolve_instance_path(bench_dir, "test_model2.onnx", "onnx");
        assert!(result.is_some(), "Should find .gz fallback");
        assert!(
            result.unwrap().to_string_lossy().ends_with(".gz"),
            "Should return .gz path"
        );

        // Test 3: File with onnx/ prefix in CSV entry
        let result = resolve_instance_path(bench_dir, "onnx/test_model.onnx", "onnx");
        assert!(result.is_some(), "Should find file with subdir prefix");
    }

    #[test]
    fn test_csv_parsing_header_skip() {
        // Verify CSV parsing skips header line with "network" column
        use std::fs;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let bench_dir = dir.path();

        // Create minimal directory structure
        let onnx_dir = bench_dir.join("onnx");
        fs::create_dir(&onnx_dir).unwrap();
        let vnnlib_dir = bench_dir.join("vnnlib");
        fs::create_dir(&vnnlib_dir).unwrap();

        // Create test files
        fs::write(onnx_dir.join("model1.onnx"), b"dummy").unwrap();
        fs::write(vnnlib_dir.join("prop_1.vnnlib"), b"dummy").unwrap();

        // Create instances.csv with header
        let csv_content = "network,property,timeout\nmodel1.onnx,prop_1.vnnlib,60\n";
        fs::write(bench_dir.join("instances.csv"), csv_content).unwrap();

        let args = AcasxuBenchmarkArgs::default();
        let result = discover_from_csv(&bench_dir.join("instances.csv"), bench_dir, &args);

        assert!(result.is_ok());
        let problem_set = result.unwrap();
        // Should have 1 instance (header skipped)
        assert_eq!(problem_set.instance_count, 1, "Header should be skipped");
        // Should also have 1 problem discovered (matching files exist)
        assert_eq!(problem_set.problems.len(), 1, "Should discover 1 problem");
        assert_eq!(problem_set.problems[0].model_name, "model1.onnx");
        assert_eq!(problem_set.problems[0].property_name, "prop_1.vnnlib");
    }
}
