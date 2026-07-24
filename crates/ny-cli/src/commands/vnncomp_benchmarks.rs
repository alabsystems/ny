// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNN-COMP benchmark download/status helpers.

use anyhow::{anyhow, bail, Result};
use clap::Subcommand;
use serde::Serialize;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_YEARS: [u32; 5] = [2021, 2023, 2024, 2025, 2026];
const ALL_YEARS: [u32; 6] = [2021, 2022, 2023, 2024, 2025, 2026];

#[derive(Subcommand)]
pub(crate) enum BenchmarkAssetsAction {
    /// Download VNN-COMP benchmark repositories.
    Download {
        /// VNN-COMP years to download. Defaults to 2021 and 2023-2026.
        years: Vec<u32>,

        /// Include optional/historical years, including 2022.
        #[arg(long, default_value_t = false)]
        all: bool,
    },

    /// Report locally available benchmark assets.
    Status {
        /// VNN-COMP year to include. Repeat to include multiple years.
        #[arg(long = "year")]
        years: Vec<u32>,

        /// Output as JSON.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// Exit non-zero when any present category is missing models or properties.
        #[arg(long, default_value_t = false)]
        strict: bool,
    },
}

pub(crate) fn handle_benchmark_assets_command(action: BenchmarkAssetsAction) -> Result<()> {
    match action {
        BenchmarkAssetsAction::Download { years, all } => {
            handle_vnncomp_benchmarks_command(years, all, false)
        }
        BenchmarkAssetsAction::Status {
            years,
            json,
            strict,
        } => run_status(years, json, strict),
    }
}

pub(crate) fn handle_vnncomp_benchmarks_command(
    years: Vec<u32>,
    all: bool,
    json_output: bool,
) -> Result<()> {
    let repo_root = find_repo_root(&std::env::current_dir()?)?;
    let script = repo_root.join("benchmarks/download_benchmarks.sh");
    if !script.is_file() {
        bail!("missing benchmark downloader: {}", script.display());
    }

    let years = selected_years(years, all);
    let args: Vec<String> = years.iter().map(u32::to_string).collect();
    let status = Command::new(&script)
        .args(&args)
        .current_dir(&repo_root)
        .status()?;
    if !status.success() {
        bail!("benchmark download failed with status {status}");
    }

    let summary = benchmark_summary(&repo_root, &years);
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "command": "vnncomp-benchmarks",
                "repo_root": repo_root,
                "years": years,
                "benchmarks": summary,
            }))?
        );
    } else {
        println!();
        println!("VNN-COMP benchmark summary");
        for entry in summary {
            println!(
                "  {}: {} ONNX files ({})",
                entry.year,
                entry.onnx_files,
                entry.path.display()
            );
        }
    }

    Ok(())
}

fn run_status(years: Vec<u32>, json_output: bool, strict: bool) -> Result<()> {
    let repo_root = find_repo_root(&std::env::current_dir()?)?;
    let script = repo_root.join("scripts/vnncomp_coverage.py");
    if !script.is_file() {
        bail!("missing benchmark coverage script: {}", script.display());
    }

    let mut command = Command::new("python3");
    command.arg(script);
    for year in years {
        command.arg("--year").arg(year.to_string());
    }
    command.arg("--root").arg(repo_root.join("benchmarks"));
    if json_output {
        command.arg("--json").arg("--pretty");
    }
    if strict {
        command.arg("--strict");
    }

    let status = command.current_dir(&repo_root).status()?;
    if !status.success() {
        bail!("benchmark status failed with status {status}");
    }
    Ok(())
}

fn selected_years(years: Vec<u32>, all: bool) -> Vec<u32> {
    let mut selected = if all {
        ALL_YEARS.to_vec()
    } else if years.is_empty() {
        DEFAULT_YEARS.to_vec()
    } else {
        years
    };
    selected.sort_unstable();
    selected.dedup();
    selected
}

#[derive(Debug, Serialize)]
struct BenchmarkSummary {
    year: u32,
    path: PathBuf,
    onnx_files: usize,
}

fn benchmark_summary(repo_root: &Path, years: &[u32]) -> Vec<BenchmarkSummary> {
    years
        .iter()
        .map(|&year| {
            let path = repo_root.join(format!("benchmarks/vnncomp{year}"));
            BenchmarkSummary {
                year,
                onnx_files: count_onnx_files(&path),
                path,
            }
        })
        .collect()
}

fn count_onnx_files(path: &Path) -> usize {
    let Ok(output) = Command::new("find")
        .arg(path)
        .arg("-name")
        .arg("*.onnx")
        .current_dir(path.parent().unwrap_or(path))
        .output()
    else {
        return 0;
    };
    if !output.status.success() {
        return 0;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.is_empty())
        .count()
}

fn find_repo_root(start: &Path) -> Result<PathBuf> {
    for ancestor in start.ancestors() {
        if ancestor.join("Cargo.toml").is_file() && ancestor.join("benchmarks").is_dir() {
            return Ok(ancestor.to_path_buf());
        }
    }
    Err(anyhow!(
        "could not find ny repo root from {}",
        start.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_years_defaults_to_current_set() {
        assert_eq!(selected_years(Vec::new(), false), DEFAULT_YEARS);
    }

    #[test]
    fn selected_years_all_includes_optional_2022() {
        assert_eq!(selected_years(Vec::new(), true), ALL_YEARS);
    }

    #[test]
    fn selected_years_sorts_and_deduplicates_explicit_years() {
        assert_eq!(
            selected_years(vec![2026, 2025, 2026], false),
            vec![2025, 2026]
        );
    }
}
