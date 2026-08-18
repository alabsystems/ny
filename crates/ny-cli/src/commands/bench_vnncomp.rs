// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNN-COMP category load-coverage audit.
//!
//! Discovers VNN-COMP benchmark categories and probes whether each category's
//! sample model and property load and convert to the verifier representations.
//! No verification is run: the audit measures LOAD coverage, not verdicts.
//! Part of #1475, #114.

use anyhow::{Context, Result};
use clap::ValueEnum;
use ny_onnx::vnnlib::load_vnnlib;
use ny_onnx::{is_multi_output_split, load_onnx, OnnxModel};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Status of a benchmark category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum CategoryStatus {
    /// Sample model and property load and convert successfully
    /// (verification itself is not probed)
    Supported,
    /// Model loads but has unsupported operators
    PartiallySupported { missing_ops: Vec<String> },
    /// Cannot load model (format error, missing files)
    UnsupportedLoad { reason: String },
    /// Property parsing fails
    UnsupportedProperty { reason: String },
    /// Audit panicked or exceeded the per-category timeout
    UnsupportedRuntime { reason: String },
    /// No benchmark files found
    Empty,
}

impl CategoryStatus {
    fn icon(&self) -> &'static str {
        match self {
            CategoryStatus::Supported => "✅",
            CategoryStatus::PartiallySupported { .. } => "⚠️",
            CategoryStatus::UnsupportedLoad { .. } => "❌",
            CategoryStatus::UnsupportedProperty { .. } => "❌",
            CategoryStatus::UnsupportedRuntime { .. } => "⏱",
            CategoryStatus::Empty => "🔲",
        }
    }

    fn label(&self) -> &'static str {
        match self {
            CategoryStatus::Supported => "Supported",
            CategoryStatus::PartiallySupported { .. } => "Partial",
            CategoryStatus::UnsupportedLoad { .. } => "Load Error",
            CategoryStatus::UnsupportedProperty { .. } => "Property Error",
            CategoryStatus::UnsupportedRuntime { .. } => "Runtime Error",
            CategoryStatus::Empty => "Empty",
        }
    }
}

/// Result of auditing a single category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CategoryAuditResult {
    /// Category name (directory name)
    pub(crate) name: String,
    /// VNN-COMP year
    pub(crate) year: u32,
    /// Number of instances in instances.csv
    pub(crate) instance_count: usize,
    /// Category status
    pub(crate) status: CategoryStatus,
    /// Sample model tested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sample_model: Option<String>,
    /// Sample property tested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sample_property: Option<String>,
    /// Time to test category in milliseconds
    pub(crate) test_time_ms: u64,
    /// ONNX operators used in sample model
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) operators: Vec<String>,
    /// Error details if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error_details: Option<String>,
}

/// Summary of full audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VnncompAuditSummary {
    /// VNN-COMP year
    pub(crate) year: u32,
    /// Total categories found
    pub(crate) total_categories: usize,
    /// Fully supported categories
    pub(crate) supported: usize,
    /// Partially supported (missing ops but loadable)
    pub(crate) partial: usize,
    /// Unsupported categories
    pub(crate) unsupported: usize,
    /// Empty/no-files categories
    pub(crate) empty: usize,
    /// Load-coverage percentage (supported + partial) / total. Measures how
    /// many categories load, not how many verify.
    pub(crate) coverage_pct: f64,
    /// Git commit hash
    pub(crate) commit: String,
    /// Individual category results
    pub(crate) categories: Vec<CategoryAuditResult>,
}

/// Arguments for VNN-COMP audit.
#[derive(Debug, Clone)]
pub(crate) struct VnncompAuditArgs {
    /// VNN-COMP year (2021, 2023, 2024, 2025)
    pub(crate) year: u32,
    /// Test timeout per category in seconds
    pub(crate) timeout: u64,
    /// JSON output
    pub(crate) json: bool,
    /// Filter to specific category
    pub(crate) category_filter: Option<String>,
}

impl Default for VnncompAuditArgs {
    fn default() -> Self {
        Self {
            year: 2021,
            timeout: 30,
            json: false,
            category_filter: None,
        }
    }
}

/// VNN-LIB corpus version used by versioned VNN-COMP benchmark layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub(crate) enum VnnlibVersion {
    #[serde(rename = "1.0")]
    #[value(name = "1.0")]
    V1,
    #[serde(rename = "2.0")]
    #[value(name = "2.0")]
    V2,
}

impl VnnlibVersion {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "1.0",
            Self::V2 => "2.0",
        }
    }
}

/// Discover benchmark categories for a given year.
pub(crate) fn discover_categories(year: u32) -> Result<Vec<(String, PathBuf)>> {
    let bench_dir = get_benchmark_dir(year)?;

    let mut categories = Vec::new();
    for entry in std::fs::read_dir(&bench_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            // Skip test directories and hidden directories
            if name.starts_with('.') || name == "test" {
                continue;
            }

            categories.push((name, path));
        }
    }

    categories.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(categories)
}

/// Get the benchmark directory for a year.
fn get_benchmark_dir(year: u32) -> Result<PathBuf> {
    let base = PathBuf::from(format!("benchmarks/vnncomp{year}/benchmarks"));
    if !base.exists() {
        anyhow::bail!(
            "Benchmark directory missing for year {}: {}. Run benchmarks/download_benchmarks.sh first.",
            year,
            base.display()
        );
    }
    Ok(base)
}

/// Find the instances CSV for a category.
///
/// A top-level `instances.csv` is authoritative: setup scripts can unpack
/// unrelated payloads with that name below data directories. Otherwise only
/// direct version children are considered. Parallel version lists are never
/// guessed between.
pub(crate) fn instances_csv_for(
    category_dir: &Path,
    version: Option<VnnlibVersion>,
) -> Result<Option<PathBuf>> {
    fn lists_in(dir: &Path) -> Result<Vec<PathBuf>> {
        let exact = dir.join("instances.csv");
        if exact.is_file() {
            return Ok(vec![exact]);
        }

        let mut legacy = Vec::new();
        for entry in std::fs::read_dir(dir)
            .with_context(|| format!("read benchmark category directory {}", dir.display()))?
        {
            let path = entry?.path();
            if path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with("_instances.csv"))
            {
                legacy.push(path);
            }
        }
        legacy.sort();
        Ok(legacy)
    }

    let render_candidates = |candidates: &[PathBuf]| {
        candidates
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };

    if let Some(version) = version {
        let selected_dir = category_dir.join(version.as_str());
        let candidates = if selected_dir.is_dir() {
            lists_in(&selected_dir)?
        } else {
            Vec::new()
        };
        return match candidates.as_slice() {
            [only] => Ok(Some(only.clone())),
            [] => anyhow::bail!(
                "requested VNN-LIB version {} has no instances.csv under {}",
                version.as_str(),
                category_dir.display()
            ),
            _ => anyhow::bail!(
                "VNN-LIB version {} has multiple instance lists under {}: {}",
                version.as_str(),
                selected_dir.display(),
                render_candidates(&candidates)
            ),
        };
    }

    // Flat layouts are authoritative even if setup unpacked nested files.
    let top_level = category_dir.join("instances.csv");
    if top_level.is_file() {
        return Ok(Some(top_level));
    }

    let mut candidates = lists_in(category_dir)?;
    for known_version in [VnnlibVersion::V1, VnnlibVersion::V2] {
        let path = category_dir.join(known_version.as_str());
        if path.is_dir() {
            candidates.extend(lists_in(&path)?);
        }
    }
    candidates.sort();
    candidates.dedup();

    match candidates.as_slice() {
        [] => Ok(None),
        [only] => Ok(Some(only.clone())),
        _ => anyhow::bail!(
            "multiple VNN-COMP instance lists found under {}: {}; pass \
             --vnnlib-version 1.0 or --vnnlib-version 2.0",
            category_dir.display(),
            render_candidates(&candidates)
        ),
    }
}

/// Parse first instance from instances CSV.
fn parse_first_instance(
    csv_path: &Path,
    category_dir: &Path,
) -> Result<(Vec<PathBuf>, PathBuf, usize)> {
    let content = std::fs::read_to_string(csv_path)
        .with_context(|| format!("Failed to read instances CSV: {}", csv_path.display()))?;
    let instance_base = csv_path.parent().unwrap_or(category_dir);

    let mut instance_count = 0usize;
    let mut first_models: Option<Vec<PathBuf>> = None;
    let mut first_prop: Option<PathBuf> = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts = parse_csv_line(trimmed)
            .with_context(|| format!("invalid CSV row in {}", csv_path.display()))?;
        if parts.len() < 2 {
            continue;
        }

        let model_names = network_paths(&parts[0]);
        let model_name = model_names
            .first()
            .cloned()
            .unwrap_or_else(|| parts[0].trim().to_string());
        // Skip header line
        if model_name.eq_ignore_ascii_case("network") {
            continue;
        }
        let property_name = parts[1].trim();

        instance_count += 1;

        if first_models.is_none() {
            let mut resolved = Vec::new();
            for model_name in model_names {
                if let Some(path) = resolve_file(instance_base, &model_name) {
                    resolved.push(path);
                } else {
                    anyhow::bail!("No model file found for {model_name}");
                }
            }
            if resolved.is_empty() {
                anyhow::bail!("No model file found");
            }
            first_models = Some(resolved);
            first_prop = resolve_file(instance_base, property_name);
        }
    }

    match (first_models, first_prop) {
        (Some(m), Some(p)) => Ok((m, p, instance_count)),
        (Some(_), None) => anyhow::bail!("No property file found"),
        (None, Some(_)) => anyhow::bail!("No model file found"),
        (None, None) => anyhow::bail!("No model or property files found"),
    }
}

/// Parse one CSV line, honoring quoted fields and rejecting malformed quoting.
///
/// Shared with `ny benchmarks run`: the paired/relational ONNX field embeds
/// commas inside quotes, so a plain `split(',')` corrupts those rows.
///
/// Benchmark paths cannot contain record separators in the VNN-COMP protocol,
/// so this deliberately parses exactly one physical line rather than accepting
/// multiline CSV fields.
pub(crate) fn parse_csv_line(line: &str) -> Result<Vec<String>> {
    #[derive(Clone, Copy)]
    enum State {
        Start,
        Unquoted,
        Quoted,
        AfterQuote,
    }

    let mut fields = Vec::new();
    let mut field = String::new();
    let mut state = State::Start;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match state {
            State::Start => match ch {
                ',' => {
                    fields.push(String::new());
                }
                '"' => {
                    state = State::Quoted;
                }
                '\n' | '\r' => anyhow::bail!("CSV record contains a line separator"),
                _ => {
                    field.push(ch);
                    state = State::Unquoted;
                }
            },
            State::Unquoted => match ch {
                ',' => {
                    fields.push(field.trim().to_string());
                    field.clear();
                    state = State::Start;
                }
                '"' => anyhow::bail!("quote appears inside an unquoted CSV field"),
                '\n' | '\r' => anyhow::bail!("CSV record contains a line separator"),
                _ => field.push(ch),
            },
            State::Quoted => match ch {
                '"' if chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => state = State::AfterQuote,
                '\n' | '\r' => anyhow::bail!("CSV record contains a line separator"),
                _ => field.push(ch),
            },
            State::AfterQuote => match ch {
                ',' => {
                    fields.push(field.trim().to_string());
                    field.clear();
                    state = State::Start;
                }
                ch if ch.is_whitespace() => {}
                _ => anyhow::bail!("non-whitespace data follows a closing CSV quote"),
            },
        }
    }

    if matches!(state, State::Quoted) {
        anyhow::bail!("unterminated quoted CSV field");
    }
    fields.push(field.trim().to_string());
    Ok(fields)
}

/// Extract every `*.onnx` path embedded in an instance-CSV model field,
/// including the paired/relational python-literal form. Shared with
/// `ny benchmarks reseed`.
pub(crate) fn network_paths(field: &str) -> Vec<String> {
    let trimmed = field.trim();
    let mut paths = Vec::new();
    let mut rest = trimmed;
    while let Some(end_rel) = rest.find(".onnx") {
        let end_idx = end_rel + ".onnx".len();
        let prefix = &rest[..end_idx];
        let start_idx = prefix
            .rfind(|ch: char| {
                ch == '\'' || ch == '"' || ch == '(' || ch == '[' || ch == ',' || ch.is_whitespace()
            })
            .map(|idx| idx + 1)
            .unwrap_or(0);
        paths.push(prefix[start_idx..].trim().to_string());
        rest = &rest[end_idx..];
    }
    if paths.is_empty() {
        paths.push(trimmed.to_string());
    }
    paths
}

#[cfg(test)]
fn first_network_path(field: &str) -> String {
    network_paths(field)
        .into_iter()
        .next()
        .unwrap_or_else(|| field.trim().to_string())
}

/// Resolve a file in the category directory.
fn resolve_file(category_dir: &Path, name: &str) -> Option<PathBuf> {
    let path = Path::new(name);

    // Try direct path
    let direct = category_dir.join(path);
    if direct.exists() {
        return Some(direct);
    }
    let direct_gz = category_dir.join(format!("{name}.gz"));
    if direct_gz.exists() {
        return Some(direct_gz);
    }

    // Try in onnx/ or vnnlib/ subdirs
    for subdir in &["onnx", "vnnlib", ""] {
        let candidate = category_dir.join(subdir).join(path);
        if candidate.exists() {
            return Some(candidate);
        }
        let candidate_gz = category_dir.join(subdir).join(format!("{name}.gz"));
        if candidate_gz.exists() {
            return Some(candidate_gz);
        }
    }

    // Some 2026 categories list onnx/original/... while setup.sh leaves
    // identical ACAS models directly under onnx/.
    if name.starts_with("onnx/original/") {
        if let Some(file_name) = path.file_name() {
            let candidate = category_dir.join("onnx").join(file_name);
            if candidate.exists() {
                return Some(candidate);
            }
            let candidate_gz = category_dir
                .join("onnx")
                .join(format!("{}.gz", file_name.to_string_lossy()));
            if candidate_gz.exists() {
                return Some(candidate_gz);
            }
        }
    }

    // Try with .gz
    let gz_path = category_dir.join(format!("{}.gz", name));
    if gz_path.exists() {
        return Some(gz_path);
    }

    None
}

/// Audit a single category.
fn audit_category(name: &str, path: &Path, year: u32) -> CategoryAuditResult {
    let start = Instant::now();

    // Find instances CSV
    let csv_path = match instances_csv_for(path, None) {
        Ok(Some(p)) => p,
        Err(error) => {
            return CategoryAuditResult {
                name: name.to_string(),
                year,
                instance_count: 0,
                status: CategoryStatus::UnsupportedLoad {
                    reason: error.to_string(),
                },
                sample_model: None,
                sample_property: None,
                test_time_ms: start.elapsed().as_millis() as u64,
                operators: vec![],
                error_details: Some(error.to_string()),
            };
        }
        Ok(None) => {
            // Try to find any ONNX model directly
            let has_onnx = std::fs::read_dir(path)
                .map(|entries| {
                    entries
                        .flatten()
                        .any(|e| e.path().extension().is_some_and(|ext| ext == "onnx"))
                })
                .unwrap_or(false);

            if !has_onnx {
                return CategoryAuditResult {
                    name: name.to_string(),
                    year,
                    instance_count: 0,
                    status: CategoryStatus::Empty,
                    sample_model: None,
                    sample_property: None,
                    test_time_ms: start.elapsed().as_millis() as u64,
                    operators: vec![],
                    error_details: Some("No instances.csv or ONNX files found".to_string()),
                };
            }

            // Find first ONNX/vnnlib pair manually
            return audit_from_files(name, path, year, start);
        }
    };

    // Parse instances
    let (model_paths, property_path, instance_count) = match parse_first_instance(&csv_path, path) {
        Ok(result) => result,
        Err(e) => {
            return CategoryAuditResult {
                name: name.to_string(),
                year,
                instance_count: 0,
                status: CategoryStatus::UnsupportedLoad {
                    reason: e.to_string(),
                },
                sample_model: None,
                sample_property: None,
                test_time_ms: start.elapsed().as_millis() as u64,
                operators: vec![],
                error_details: Some(e.to_string()),
            };
        }
    };

    let model_path = &model_paths[0];
    let model_name = model_paths
        .iter()
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
        .collect::<Vec<_>>()
        .join(",");
    let model_name = if model_name.is_empty() {
        model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string()
    } else {
        model_name
    };
    let property_name = property_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Try to load model
    let onnx_model = match load_onnx(&model_path) {
        Ok(m) => m,
        Err(e) => {
            return CategoryAuditResult {
                name: name.to_string(),
                year,
                instance_count,
                status: CategoryStatus::UnsupportedLoad {
                    reason: format!("ONNX load failed: {}", e),
                },
                sample_model: Some(model_name),
                sample_property: Some(property_name),
                test_time_ms: start.elapsed().as_millis() as u64,
                operators: vec![],
                error_details: Some(e.to_string()),
            };
        }
    };

    // Collect operators (from layer types)
    let operators: Vec<String> = onnx_model
        .network
        .layers
        .iter()
        .map(|l| format!("{:?}", l.layer_type))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    // Try to convert to the verifier load representation.
    let load_result = probe_model_load(&onnx_model);
    match load_result {
        Ok(()) => {}
        Err(e) => {
            // Check if it's an unsupported operator
            let err_str = e.to_string();
            let missing_ops = extract_missing_ops(&err_str);

            let status = if !missing_ops.is_empty() {
                CategoryStatus::PartiallySupported { missing_ops }
            } else {
                CategoryStatus::UnsupportedLoad { reason: err_str }
            };

            return CategoryAuditResult {
                name: name.to_string(),
                year,
                instance_count,
                status,
                sample_model: Some(model_name),
                sample_property: Some(property_name),
                test_time_ms: start.elapsed().as_millis() as u64,
                operators,
                error_details: Some(e.to_string()),
            };
        }
    }

    for extra_model_path in model_paths.iter().skip(1) {
        let extra = match load_onnx(extra_model_path) {
            Ok(m) => m,
            Err(e) => {
                return CategoryAuditResult {
                    name: name.to_string(),
                    year,
                    instance_count,
                    status: CategoryStatus::UnsupportedLoad {
                        reason: format!("ONNX load failed: {}", e),
                    },
                    sample_model: Some(model_name),
                    sample_property: Some(property_name),
                    test_time_ms: start.elapsed().as_millis() as u64,
                    operators,
                    error_details: Some(e.to_string()),
                };
            }
        };
        if let Err(e) = probe_model_load(&extra) {
            return CategoryAuditResult {
                name: name.to_string(),
                year,
                instance_count,
                status: CategoryStatus::UnsupportedLoad {
                    reason: e.to_string(),
                },
                sample_model: Some(model_name),
                sample_property: Some(property_name),
                test_time_ms: start.elapsed().as_millis() as u64,
                operators,
                error_details: Some(e.to_string()),
            };
        }
    }

    // Try to load property
    let _vnnlib = match load_vnnlib(&property_path) {
        Ok(v) => v,
        Err(e) => {
            return CategoryAuditResult {
                name: name.to_string(),
                year,
                instance_count,
                status: CategoryStatus::UnsupportedProperty {
                    reason: e.to_string(),
                },
                sample_model: Some(model_name),
                sample_property: Some(property_name),
                test_time_ms: start.elapsed().as_millis() as u64,
                operators,
                error_details: Some(e.to_string()),
            };
        }
    };

    // Success - model and property load
    CategoryAuditResult {
        name: name.to_string(),
        year,
        instance_count,
        status: CategoryStatus::Supported,
        sample_model: Some(model_name),
        sample_property: Some(property_name),
        test_time_ms: start.elapsed().as_millis() as u64,
        operators,
        error_details: None,
    }
}

/// Audit a category by finding ONNX/vnnlib files directly (no instances.csv).
fn audit_from_files(name: &str, path: &Path, year: u32, start: Instant) -> CategoryAuditResult {
    // Find first ONNX file
    let onnx_file = std::fs::read_dir(path)
        .ok()
        .and_then(|entries| {
            entries
                .flatten()
                .find(|e| e.path().extension().is_some_and(|ext| ext == "onnx"))
        })
        .map(|e| e.path());

    let model_path = match onnx_file {
        Some(p) => p,
        None => {
            return CategoryAuditResult {
                name: name.to_string(),
                year,
                instance_count: 0,
                status: CategoryStatus::Empty,
                sample_model: None,
                sample_property: None,
                test_time_ms: start.elapsed().as_millis() as u64,
                operators: vec![],
                error_details: Some("No ONNX files found".to_string()),
            };
        }
    };

    // Find first vnnlib file
    let vnnlib_file = std::fs::read_dir(path)
        .ok()
        .and_then(|entries| {
            entries
                .flatten()
                .find(|e| e.path().extension().is_some_and(|ext| ext == "vnnlib"))
        })
        .map(|e| e.path());

    let model_name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Try to load model
    let onnx_model = match load_onnx(&model_path) {
        Ok(m) => m,
        Err(e) => {
            return CategoryAuditResult {
                name: name.to_string(),
                year,
                instance_count: 0,
                status: CategoryStatus::UnsupportedLoad {
                    reason: format!("ONNX load failed: {}", e),
                },
                sample_model: Some(model_name),
                sample_property: None,
                test_time_ms: start.elapsed().as_millis() as u64,
                operators: vec![],
                error_details: Some(e.to_string()),
            };
        }
    };

    // Collect operators (from layer types)
    let operators: Vec<String> = onnx_model
        .network
        .layers
        .iter()
        .map(|l| format!("{:?}", l.layer_type))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    // Try to convert to the verifier load representation.
    if let Err(e) = probe_model_load(&onnx_model) {
        let err_str = e.to_string();
        let missing_ops = extract_missing_ops(&err_str);

        let status = if !missing_ops.is_empty() {
            CategoryStatus::PartiallySupported { missing_ops }
        } else {
            CategoryStatus::UnsupportedLoad { reason: err_str }
        };

        return CategoryAuditResult {
            name: name.to_string(),
            year,
            instance_count: 0,
            status,
            sample_model: Some(model_name),
            sample_property: vnnlib_file
                .as_ref()
                .and_then(|p| p.file_name().and_then(|n| n.to_str()).map(String::from)),
            test_time_ms: start.elapsed().as_millis() as u64,
            operators,
            error_details: Some(e.to_string()),
        };
    }

    // Check property if available
    if let Some(ref prop_path) = vnnlib_file {
        if let Err(e) = load_vnnlib(prop_path) {
            return CategoryAuditResult {
                name: name.to_string(),
                year,
                instance_count: 0,
                status: CategoryStatus::UnsupportedProperty {
                    reason: e.to_string(),
                },
                sample_model: Some(model_name),
                sample_property: prop_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(String::from),
                test_time_ms: start.elapsed().as_millis() as u64,
                operators,
                error_details: Some(e.to_string()),
            };
        }
    }

    CategoryAuditResult {
        name: name.to_string(),
        year,
        instance_count: 0,
        status: CategoryStatus::Supported,
        sample_model: Some(model_name),
        sample_property: vnnlib_file
            .as_ref()
            .and_then(|p| p.file_name().and_then(|n| n.to_str()).map(String::from)),
        test_time_ms: start.elapsed().as_millis() as u64,
        operators,
        error_details: None,
    }
}

fn probe_model_load(onnx_model: &OnnxModel) -> ny_core::Result<()> {
    if onnx_model.network.layers.iter().any(is_multi_output_split) {
        onnx_model.to_graph_network().map(|_| ())
    } else {
        onnx_model.to_propagate_network().map(|_| ())
    }
}

/// Extract missing operator names from error message.
fn extract_missing_ops(error: &str) -> Vec<String> {
    let mut ops = Vec::new();

    // Common patterns: "unsupported op: Xyz", "Unsupported operator 'Xyz'"
    // All patterns must be lowercase since we search in lowercased error string
    let patterns = [
        "unsupported op: ",
        "unsupported operator ",
        "unsupported operator: ",
        "unknown operator: ",
        "not supported: ",
    ];

    let error_lower = error.to_lowercase();
    for pattern in patterns {
        if let Some(idx) = error_lower.find(pattern) {
            let start = idx + pattern.len();
            let rest = &error[start..];
            // Extract operator name (word or quoted string)
            let op = rest
                .trim_start_matches(['\'', '"'])
                .split(|c: char| c.is_whitespace() || c == '\'' || c == '"' || c == ',')
                .next()
                .unwrap_or("")
                .to_string();
            if !op.is_empty() && !ops.contains(&op) {
                ops.push(op);
            }
        }
    }

    ops
}

/// Run [`audit_category`] on a worker thread, bounded by `timeout`.
///
/// The audit body is synchronous file parsing with no interruption point, so
/// a category that exceeds its budget is abandoned (the detached worker is
/// left to finish or hang harmlessly) and reported as a runtime failure
/// instead of stalling the whole sweep.
fn audit_category_with_timeout(
    name: &str,
    path: &Path,
    year: u32,
    timeout: Duration,
) -> CategoryAuditResult {
    let start = Instant::now();
    let (sender, receiver) = std::sync::mpsc::channel();
    let worker_name = name.to_string();
    let worker_path = path.to_path_buf();
    std::thread::spawn(move || {
        let _ = sender.send(audit_category(&worker_name, &worker_path, year));
    });

    match receiver.recv_timeout(timeout) {
        Ok(result) => result,
        Err(err) => {
            let reason = match err {
                std::sync::mpsc::RecvTimeoutError::Timeout => {
                    format!("audit exceeded the {}s category timeout", timeout.as_secs())
                }
                // The worker dropped the sender without sending a result: it
                // panicked mid-audit.
                std::sync::mpsc::RecvTimeoutError::Disconnected => {
                    "audit worker panicked".to_string()
                }
            };
            CategoryAuditResult {
                name: name.to_string(),
                year,
                instance_count: 0,
                status: CategoryStatus::UnsupportedRuntime {
                    reason: reason.clone(),
                },
                sample_model: None,
                sample_property: None,
                test_time_ms: start.elapsed().as_millis() as u64,
                operators: vec![],
                error_details: Some(reason),
            }
        }
    }
}

/// Run the VNN-COMP category audit.
pub(crate) fn run_vnncomp_audit(args: VnncompAuditArgs) -> Result<VnncompAuditSummary> {
    let categories = discover_categories(args.year)?;

    if categories.is_empty() {
        anyhow::bail!(
            "No benchmark categories found for year {}. Run benchmarks/download_benchmarks.sh first.",
            args.year
        );
    }

    info!(
        "Auditing {} VNN-COMP {} categories...",
        categories.len(),
        args.year
    );

    let timeout = Duration::from_secs(args.timeout);
    let mut results = Vec::new();

    for (name, path) in &categories {
        // Apply filter if specified
        if let Some(ref filter) = args.category_filter {
            if !name.contains(filter) {
                continue;
            }
        }

        debug!("Auditing category: {}", name);
        let result = audit_category_with_timeout(name, path, args.year, timeout);

        if !args.json {
            eprint!(
                "\r{} {:20} {:15} ({} instances, {}ms)    ",
                result.status.icon(),
                result.name,
                result.status.label(),
                result.instance_count,
                result.test_time_ms
            );
        }

        results.push(result);
    }

    if !args.json {
        eprintln!(); // Clear progress line
    }

    // Calculate summary
    let supported = results
        .iter()
        .filter(|r| matches!(r.status, CategoryStatus::Supported))
        .count();
    let partial = results
        .iter()
        .filter(|r| matches!(r.status, CategoryStatus::PartiallySupported { .. }))
        .count();
    let unsupported = results
        .iter()
        .filter(|r| {
            matches!(
                r.status,
                CategoryStatus::UnsupportedLoad { .. }
                    | CategoryStatus::UnsupportedProperty { .. }
                    | CategoryStatus::UnsupportedRuntime { .. }
            )
        })
        .count();
    let empty = results
        .iter()
        .filter(|r| matches!(r.status, CategoryStatus::Empty))
        .count();

    let total = results.len();
    let coverage_pct = if total > 0 {
        ((supported + partial) as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    Ok(VnncompAuditSummary {
        year: args.year,
        total_categories: total,
        supported,
        partial,
        unsupported,
        empty,
        coverage_pct,
        commit: get_git_commit(),
        categories: results,
    })
}

/// Get current git commit hash.
fn get_git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Print summary in text format.
pub(crate) fn print_audit_summary(summary: &VnncompAuditSummary, verbose: bool) {
    println!("\nVNN-COMP {} Category Audit", summary.year);
    println!("===========================");
    println!();

    // Print category table
    println!("| Status | Category             | Instances | Model               | Notes");
    println!("|--------|----------------------|-----------|---------------------|-------");

    for cat in &summary.categories {
        let notes = match &cat.status {
            CategoryStatus::PartiallySupported { missing_ops } => {
                format!("Missing: {}", missing_ops.join(", "))
            }
            CategoryStatus::UnsupportedLoad { reason } => truncate_str(reason, 40),
            CategoryStatus::UnsupportedProperty { reason } => truncate_str(reason, 40),
            CategoryStatus::UnsupportedRuntime { reason } => truncate_str(reason, 40),
            CategoryStatus::Empty => "No files".to_string(),
            CategoryStatus::Supported => "OK".to_string(),
        };

        println!(
            "| {}    | {:20} | {:>9} | {:19} | {}",
            cat.status.icon(),
            truncate_str(&cat.name, 20),
            cat.instance_count,
            cat.sample_model
                .as_ref()
                .map(|s| truncate_str(s, 19))
                .unwrap_or_else(|| "-".to_string()),
            notes
        );

        if verbose && !cat.operators.is_empty() {
            println!("|        | Operators: {}", cat.operators.join(", "));
        }
    }

    println!();
    println!("Summary:");
    println!("  Total categories: {}", summary.total_categories);
    println!(
        "  ✅ Supported:     {} ({:.0}%)",
        summary.supported,
        (summary.supported as f64 / summary.total_categories as f64) * 100.0
    );
    println!(
        "  ⚠️  Partial:       {} ({:.0}%)",
        summary.partial,
        (summary.partial as f64 / summary.total_categories as f64) * 100.0
    );
    println!(
        "  ❌ Unsupported:   {} ({:.0}%)",
        summary.unsupported,
        (summary.unsupported as f64 / summary.total_categories as f64) * 100.0
    );
    if summary.empty > 0 {
        println!(
            "  🔲 Empty:         {} ({:.0}%)",
            summary.empty,
            (summary.empty as f64 / summary.total_categories as f64) * 100.0
        );
    }
    println!();
    println!(
        "Load coverage: {:.1}% (supported + partial; verification not probed)",
        summary.coverage_pct
    );
    println!("Commit: {}", summary.commit);
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_missing_ops() {
        assert_eq!(
            extract_missing_ops("unsupported op: MaxPool"),
            vec!["MaxPool"]
        );
        assert_eq!(
            extract_missing_ops("Unsupported operator 'Conv'"),
            vec!["Conv"]
        );
        assert!(extract_missing_ops("some random error").is_empty());
    }

    #[test]
    fn test_category_status_labels() {
        assert_eq!(CategoryStatus::Supported.label(), "Supported");
        assert_eq!(
            CategoryStatus::PartiallySupported {
                missing_ops: vec![]
            }
            .label(),
            "Partial"
        );
    }

    #[test]
    fn test_truncate_str() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello world", 8), "hello...");
    }

    #[test]
    fn test_parse_csv_line_preserves_quoted_model_tuple() {
        let fields = parse_csv_line(
            "\"[('f', 'onnx/original/model.onnx'), ('g', 'onnx/perturbed/model.onnx')]\",./vnnlib/instance_0.vnnlib,100",
        )
        .expect("valid CSV");

        assert_eq!(fields.len(), 3);
        assert_eq!(first_network_path(&fields[0]), "onnx/original/model.onnx");
        assert_eq!(fields[1], "./vnnlib/instance_0.vnnlib");
    }

    #[test]
    fn parse_csv_line_rejects_malformed_quoting() {
        assert!(parse_csv_line("\"unterminated,p.vnnlib,10").is_err());
        assert!(parse_csv_line("bad\"quote,p.vnnlib,10").is_err());
        assert!(parse_csv_line("\"closed\"junk,p.vnnlib,10").is_err());
    }

    #[test]
    fn instances_csv_resolver_preserves_unambiguous_flat_layout() {
        let category = tempfile::tempdir().expect("category");
        let expected = category.path().join("instances.csv");
        std::fs::write(&expected, "m.onnx,p.vnnlib,10\n").expect("instances");

        assert_eq!(
            instances_csv_for(category.path(), None).expect("resolve"),
            Some(expected)
        );
    }

    #[test]
    fn instances_csv_resolver_treats_top_level_list_as_authoritative() {
        let category = tempfile::tempdir().expect("category");
        let expected = category.path().join("instances.csv");
        std::fs::write(&expected, "m.onnx,p.vnnlib,10\n").expect("instances");
        std::fs::create_dir(category.path().join("vnnlib")).expect("payload directory");
        std::fs::write(
            category.path().join("vnnlib/instances.csv"),
            "payload,not,a-list\n",
        )
        .expect("payload");

        assert_eq!(
            instances_csv_for(category.path(), None).expect("resolve"),
            Some(expected)
        );
    }

    #[test]
    fn instances_csv_resolver_ignores_nested_payload_lists() {
        let category = tempfile::tempdir().expect("category");
        std::fs::create_dir(category.path().join("vnnlib")).expect("payload directory");
        std::fs::write(
            category.path().join("vnnlib/instances.csv"),
            "payload,not,a-list\n",
        )
        .expect("payload");

        assert_eq!(
            instances_csv_for(category.path(), None).expect("resolve"),
            None
        );
    }

    #[test]
    fn instances_csv_resolver_rejects_parallel_versions_without_selector() {
        let category = tempfile::tempdir().expect("category");
        for version in ["1.0", "2.0"] {
            std::fs::create_dir(category.path().join(version)).expect("version directory");
            std::fs::write(
                category.path().join(version).join("instances.csv"),
                "m.onnx,p.vnnlib,10\n",
            )
            .expect("instances");
        }

        let error = instances_csv_for(category.path(), None)
            .expect_err("parallel versions must be ambiguous");
        // The message renders paths with `Path::display`, i.e. the HOST
        // separator, so normalize before matching POSIX-shaped needles.
        let message = error.to_string().replace('\\', "/");
        assert!(message.contains("--vnnlib-version"));
        assert!(message.contains("1.0/instances.csv"));
        assert!(message.contains("2.0/instances.csv"));
    }

    #[test]
    fn instances_csv_resolver_selects_requested_version_only() {
        let category = tempfile::tempdir().expect("category");
        for version in ["1.0", "2.0"] {
            std::fs::create_dir(category.path().join(version)).expect("version directory");
            std::fs::write(
                category.path().join(version).join("instances.csv"),
                "m.onnx,p.vnnlib,10\n",
            )
            .expect("instances");
        }

        assert_eq!(
            instances_csv_for(category.path(), Some(VnnlibVersion::V1)).expect("select 1.0"),
            Some(category.path().join("1.0/instances.csv"))
        );
        assert_eq!(
            instances_csv_for(category.path(), Some(VnnlibVersion::V2)).expect("select 2.0"),
            Some(category.path().join("2.0/instances.csv"))
        );
    }

    #[test]
    fn instances_csv_resolver_never_falls_back_from_missing_requested_version() {
        let category = tempfile::tempdir().expect("category");
        std::fs::create_dir(category.path().join("1.0")).expect("version directory");
        std::fs::write(
            category.path().join("1.0/instances.csv"),
            "m.onnx,p.vnnlib,10\n",
        )
        .expect("instances");

        let error = instances_csv_for(category.path(), Some(VnnlibVersion::V2))
            .expect_err("a requested version must never fall back");
        assert!(error.to_string().contains("version 2.0"));
    }
}
