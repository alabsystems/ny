// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ACAS-Xu benchmark runner for measuring VISION.md success metrics.
//!
//! Runs verification on ACAS-Xu benchmark instances from VNN-COMP (year-selectable).
//! Primary success metric: >95% verified rate on ACAS-Xu benchmark.
//!
//! Part of #154.

mod discovery;
mod verification;

use anyhow::{anyhow, Result};
use ny_core::GemmEngine;
use ny_gpu::{Backend, ComputeDevice};
use ny_propagate::{BabVerificationStatus, BranchingHeuristic};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::info;

pub(crate) use discovery::discover_problems;
use verification::{resolve_branching_heuristic, run_verification};

/// Default timeout when not specified in instances.csv.
const DEFAULT_TIMEOUT_SECS: u64 = 60;
type VerificationOutcome = (BabVerificationStatus, usize, usize);

/// Runtime GEMM engine context for ACAS-Xu GPU BaB benchmarking.
///
/// Uses wgpu ComputeDevice when available, otherwise falls back to
/// `NaiveCpuGemmEngine` from ny-core.
///
/// Stores `Arc<ComputeDevice>` so the engine can be shared with
/// `BetaCrownVerifier::new_with_engine` for stored-engine construction (#3643).
struct GpuBabEngineRuntime {
    compute_device: Option<Arc<ComputeDevice>>,
    fallback_reason: Option<String>,
}

impl GpuBabEngineRuntime {
    fn initialize() -> Self {
        match ComputeDevice::new(Backend::Wgpu) {
            Ok(device) => Self {
                compute_device: Some(Arc::new(device)),
                fallback_reason: None,
            },
            Err(e) => Self {
                compute_device: None,
                fallback_reason: Some(e.to_string()),
            },
        }
    }

    /// Get the engine as an `Arc<dyn GemmEngine>` for verifier construction.
    fn engine_arc(&self) -> Option<Arc<dyn GemmEngine>> {
        self.compute_device
            .as_ref()
            .map(|device| Arc::clone(device) as Arc<dyn GemmEngine>)
    }

    fn engine_summary(&self) -> String {
        match self.compute_device.as_deref() {
            Some(device) => format!(
                "requested=wgpu,effective={},gemm=compute-device",
                device.backend()
            ),
            None => "requested=wgpu,effective=cpu-fallback,gemm=cpu-naive".to_string(),
        }
    }

    fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }

    fn clear_crown_working_set(&self) -> Result<()> {
        if let Some(device) = self.compute_device.as_deref() {
            device.clear_crown_working_set()?;
        }
        Ok(())
    }
}

/// Result of a single verification problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AcasxuResult {
    /// Model filename (e.g., "ACASXU_run2a_1_1_batch_2000.onnx")
    pub(crate) model: String,
    /// Property filename (e.g., "prop_1.vnnlib")
    pub(crate) property: String,
    /// Verification status
    pub(crate) status: String,
    /// Time taken in milliseconds
    pub(crate) time_ms: u64,
    /// Number of domains explored (if available)
    pub(crate) domains: usize,
    /// Number of domains that were verified/pruned (if available)
    pub(crate) domains_verified: usize,
    /// Error message if verification failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
}

/// Summary of benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AcasxuSummary {
    /// Benchmark name
    pub(crate) benchmark: String,
    /// Benchmark year
    pub(crate) benchmark_year: u32,
    /// Total instances defined in instances.csv (before filtering)
    pub(crate) instance_count: usize,
    /// Total number of problems
    pub(crate) total: usize,
    /// Number verified
    pub(crate) verified: usize,
    /// Number falsified (counterexample found)
    pub(crate) falsified: usize,
    /// Number unknown (timeout or other)
    pub(crate) unknown: usize,
    /// Number that hit timeout
    pub(crate) timeout_count: usize,
    /// Number that had errors
    pub(crate) error_count: usize,
    /// Pass rate (verified / total)
    pub(crate) pass_rate: f64,
    /// Target pass rate
    pub(crate) target_rate: f64,
    /// Total time in milliseconds
    pub(crate) total_time_ms: u64,
    /// Average time per problem in milliseconds
    pub(crate) avg_time_ms: u64,
    /// Timeout per problem in seconds
    pub(crate) timeout_seconds: u64,
    /// Git commit hash
    pub(crate) commit: String,
    /// Individual results
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) results: Vec<AcasxuResult>,
}

impl AcasxuSummary {
    /// Create a new summary for the given number of problems.
    pub(crate) fn new(
        total: usize,
        timeout_seconds: u64,
        benchmark_year: u32,
        instance_count: usize,
    ) -> Self {
        Self {
            benchmark: "acasxu".to_string(),
            benchmark_year,
            instance_count,
            total,
            verified: 0,
            falsified: 0,
            unknown: 0,
            timeout_count: 0,
            error_count: 0,
            pass_rate: 0.0,
            // 98.4% (183/186) measured 2025-02-03 with --branching input.
            // Set 0.4pp below best to catch real regressions while tolerating noise.
            target_rate: 0.98,
            total_time_ms: 0,
            avg_time_ms: 0,
            timeout_seconds,
            commit: get_git_commit(),
            results: Vec::with_capacity(total),
        }
    }

    /// Add a result and update counts.
    pub(crate) fn add_result(&mut self, result: AcasxuResult) {
        match result.status.as_str() {
            "Verified" => self.verified += 1,
            "Falsified" => self.falsified += 1,
            "Timeout" => {
                self.timeout_count += 1;
                self.unknown += 1;
            }
            "Error" => {
                self.error_count += 1;
                self.unknown += 1;
            }
            _ => self.unknown += 1,
        }
        self.total_time_ms += result.time_ms;
        self.results.push(result);
        self.update_rates();
    }

    fn update_rates(&mut self) {
        let completed = self.results.len();
        if completed > 0 {
            self.pass_rate = self.verified as f64 / self.total as f64;
            self.avg_time_ms = self.total_time_ms / completed as u64;
        }
    }

    /// Number of completed problems.
    pub(crate) fn completed(&self) -> usize {
        self.results.len()
    }
}

/// A verification problem (model + property pair).
#[derive(Debug, Clone)]
pub(crate) struct AcasxuProblem {
    /// Model file path
    pub(crate) model_path: PathBuf,
    /// Property file path
    pub(crate) property_path: PathBuf,
    /// Model filename
    pub(crate) model_name: String,
    /// Property filename
    pub(crate) property_name: String,
    /// Timeout in seconds (from CSV or default)
    pub(crate) timeout: u64,
}

/// Arguments for the ACAS-Xu benchmark.
#[derive(Debug, Clone)]
pub(crate) struct AcasxuBenchmarkArgs {
    /// Timeout per problem in seconds (default: 60 unless instances.csv provides one)
    pub(crate) timeout: u64,
    /// Override timeout per problem in seconds (forces all problems to this timeout).
    pub(crate) timeout_override: Option<u64>,
    /// VNN-COMP year (2021, 2023, 2024, 2025)
    pub(crate) year: u32,
    /// Filter to specific model (e.g., "1_1" for ACASXU_run2a_1_1)
    pub(crate) model_filter: Option<String>,
    /// Filter to specific property (e.g., "1" for prop_1.vnnlib)
    pub(crate) property_filter: Option<String>,
    /// JSON output
    pub(crate) json: bool,
    /// Include individual results in JSON output.
    pub(crate) include_results: bool,
    /// Branching heuristic (width, impact/babsr, fsb/kfsb, sequential, input).
    /// With `gpu_bab=true`, only `impact`/`babsr` is currently supported.
    pub(crate) branching: String,
    /// Maximum domains to explore
    pub(crate) max_domains: usize,
    /// Enable proactive cut generation (BICCOS-lite).
    pub(crate) proactive_cuts: bool,
    /// Maximum number of proactive cuts to generate.
    pub(crate) max_proactive_cuts: usize,
    /// Enable relaxed clipping for input splitting (Clip-and-Verify).
    pub(crate) relaxed_clip: bool,
    /// Enable PGD attack to find counterexamples.
    pub(crate) pgd_attack: bool,
    /// Number of PGD attack restarts.
    pub(crate) pgd_restarts: usize,
    /// Use GPU-accelerated BaB with DomainList storage.
    /// Routes through verify_graph_gpu_domain_list instead of sequential verify().
    pub(crate) gpu_bab: bool,
    /// Disable lA warm-start in GPU BaB backward pass (for A/B benchmarking).
    pub(crate) no_la_warm_start: bool,
}

impl Default for AcasxuBenchmarkArgs {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT_SECS,
            timeout_override: None,
            year: 2021,
            model_filter: None,
            property_filter: None,
            json: false,
            include_results: true,
            // Default to input branching for ACAS-Xu: achieves 98.4% vs 28% with width.
            // This matches α,β-CROWN's ACAS-Xu configuration.
            // See: reports/experiments/2026-02-03-acasxu-config-experiments.md
            branching: "input".to_string(),
            max_domains: 10000,
            proactive_cuts: false,
            max_proactive_cuts: 100,
            // Auto-enabled when branching=input (see run_verification)
            relaxed_clip: false,
            pgd_attack: false,
            pgd_restarts: 100,
            gpu_bab: false,
            no_la_warm_start: false,
        }
    }
}

/// Set of discovered problems and metadata.
pub(crate) struct AcasxuProblemSet {
    pub(crate) problems: Vec<AcasxuProblem>,
    pub(crate) instance_count: usize,
}

/// Run verification on a single problem.
fn run_single_problem(
    problem: &AcasxuProblem,
    args: &AcasxuBenchmarkArgs,
    gpu_bab_engine: Option<&GpuBabEngineRuntime>,
) -> AcasxuResult {
    run_single_problem_with_hooks(
        problem,
        || run_verification(problem, args, gpu_bab_engine),
        || match gpu_bab_engine {
            Some(runtime) => runtime.clear_crown_working_set(),
            None => Ok(()),
        },
    )
}

fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

// The catch_unwind pair below records a panicking instance as a per-problem
// `status = "Error"` result and continues the sweep. The workspace release
// profile deliberately unwinds, so the shipped binary retains this recovery
// path. Native aborts, faults, and hangs still require process-level isolation.
fn run_single_problem_with_hooks<Verify, Cleanup>(
    problem: &AcasxuProblem,
    verify: Verify,
    cleanup: Cleanup,
) -> AcasxuResult
where
    Verify: FnOnce() -> Result<VerificationOutcome>,
    Cleanup: FnOnce() -> Result<()>,
{
    let start = Instant::now();
    let verification = std::panic::catch_unwind(std::panic::AssertUnwindSafe(verify))
        .map_err(|payload| {
            anyhow!(
                "verification panicked: {}",
                panic_payload_message(payload.as_ref())
            )
        })
        .and_then(|result| result);
    let cleanup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(cleanup))
        .map_err(|payload| {
            anyhow!(
                "cleanup panicked: {}",
                panic_payload_message(payload.as_ref())
            )
        })
        .and_then(|result| result);
    let elapsed = start.elapsed();
    let time_ms = elapsed.as_millis() as u64;

    match (verification, cleanup) {
        (Ok((status, domains, domains_verified)), Ok(())) => {
            let status_str = match &status {
                BabVerificationStatus::Verified => "Verified",
                BabVerificationStatus::Violated { .. } => "Falsified",
                BabVerificationStatus::PotentialViolation { .. } => "PotentialViolation",
                BabVerificationStatus::Unknown { reason } => {
                    if reason.contains("timeout") || elapsed >= Duration::from_secs(problem.timeout)
                    {
                        "Timeout"
                    } else {
                        "Unknown"
                    }
                }
                BabVerificationStatus::Timeout => "Timeout",
            };
            AcasxuResult {
                model: problem.model_name.clone(),
                property: problem.property_name.clone(),
                status: status_str.to_string(),
                time_ms,
                domains,
                domains_verified,
                error: None,
            }
        }
        (Ok(_), Err(cleanup_err)) => AcasxuResult {
            model: problem.model_name.clone(),
            property: problem.property_name.clone(),
            status: "Error".to_string(),
            time_ms,
            domains: 0,
            domains_verified: 0,
            error: Some(format!(
                "failed to clear GPU CROWN working set after verification: {cleanup_err}"
            )),
        },
        (Err(err), Ok(())) => AcasxuResult {
            model: problem.model_name.clone(),
            property: problem.property_name.clone(),
            status: "Error".to_string(),
            time_ms,
            domains: 0,
            domains_verified: 0,
            error: Some(err.to_string()),
        },
        (Err(err), Err(cleanup_err)) => AcasxuResult {
            model: problem.model_name.clone(),
            property: problem.property_name.clone(),
            status: "Error".to_string(),
            time_ms,
            domains: 0,
            domains_verified: 0,
            error: Some(format!(
                "{err}; failed to clear GPU CROWN working set after verification: {cleanup_err}"
            )),
        },
    }
}

pub(crate) fn run_acasxu_benchmark(args: AcasxuBenchmarkArgs) -> Result<AcasxuSummary> {
    let branching_heuristic = resolve_branching_heuristic(&args.branching, args.gpu_bab)?;
    let gpu_bab_engine = args.gpu_bab.then(GpuBabEngineRuntime::initialize);
    let AcasxuProblemSet {
        problems,
        instance_count,
    } = discover_problems(&args)?;

    if problems.is_empty() {
        anyhow::bail!("No ACAS-Xu problems found matching filters");
    }

    let log_timeout = args.timeout_override.unwrap_or(args.timeout);
    info!(
        "Running ACAS-Xu benchmark: {} problems ({} instances), year={}, timeout={}s",
        problems.len(),
        instance_count,
        args.year,
        log_timeout
    );
    // Log auto-enabled options when using input splitting (α,β-CROWN config match).
    let input_split_note = if matches!(branching_heuristic, BranchingHeuristic::InputSplit) {
        " [auto: relaxed_clip=true, pgd_attack=true, pgd_restarts=10000]"
    } else {
        ""
    };
    let gpu_bab_note = if args.gpu_bab {
        " [gpu_bab: GraphNetwork + DomainList]"
    } else {
        ""
    };
    info!(
        "ACAS-Xu config: branching={}, max_domains={}, filter={}/{}{}{}",
        &args.branching,
        args.max_domains,
        args.model_filter.as_deref().unwrap_or("*"),
        args.property_filter.as_deref().unwrap_or("*"),
        input_split_note,
        gpu_bab_note
    );
    if let Some(engine_runtime) = gpu_bab_engine.as_ref() {
        let summary = engine_runtime.engine_summary();
        info!("ACAS-Xu GPU BaB engine: {summary}");
        if let Some(reason) = engine_runtime.fallback_reason() {
            info!("ACAS-Xu GPU BaB fallback reason: {reason}");
        }
    }

    let timeout_seconds = if let Some(override_timeout) = args.timeout_override {
        override_timeout
    } else if problems
        .first()
        .map(|p| problems.iter().all(|other| other.timeout == p.timeout))
        .unwrap_or(false)
    {
        problems[0].timeout
    } else {
        args.timeout
    };

    let mut summary =
        AcasxuSummary::new(problems.len(), timeout_seconds, args.year, instance_count);

    for problem in &problems {
        let result = run_single_problem(problem, &args, gpu_bab_engine.as_ref());

        if !args.json {
            let status_indicator = match result.status.as_str() {
                "Verified" => "\x1b[32m✓\x1b[0m",
                "Falsified" => "\x1b[33m✗\x1b[0m",
                "Timeout" => "\x1b[31m⏱\x1b[0m",
                "Error" => "\x1b[31m!\x1b[0m",
                _ => "?",
            };
            eprint!(
                "\r[{}/{}] {} {} × {}: {} ({:.2}s)    ",
                summary.completed() + 1,
                summary.total,
                status_indicator,
                problem.model_name,
                problem.property_name,
                result.status,
                result.time_ms as f64 / 1000.0
            );
        }

        summary.add_result(result);
    }

    if !args.json {
        eprintln!(); // Clear progress line
    }

    // Clear individual results if not requested in JSON output
    if !args.include_results {
        summary.results.clear();
    }

    Ok(summary)
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
pub(crate) fn print_summary(summary: &AcasxuSummary) {
    println!("\nACAS-Xu Benchmark Results");
    println!("========================");
    println!("Year: {}", summary.benchmark_year);
    println!("Instances (CSV): {}", summary.instance_count);
    println!("Total problems: {}", summary.total);
    println!();
    println!(
        "Verified:   {:>4}/{} ({:.1}%)",
        summary.verified,
        summary.total,
        (summary.verified as f64 / summary.total as f64) * 100.0
    );
    println!(
        "Falsified:  {:>4}/{} ({:.1}%)",
        summary.falsified,
        summary.total,
        (summary.falsified as f64 / summary.total as f64) * 100.0
    );
    println!(
        "Unknown:    {:>4}/{} ({:.1}%)",
        summary.unknown,
        summary.total,
        (summary.unknown as f64 / summary.total as f64) * 100.0
    );
    if summary.timeout_count > 0 {
        println!("  (timeout: {})", summary.timeout_count);
    }
    if summary.error_count > 0 {
        println!("  (errors:  {})", summary.error_count);
    }
    println!();
    let target_met = if summary.pass_rate >= summary.target_rate {
        "\x1b[32m✓\x1b[0m"
    } else {
        "\x1b[31m✗\x1b[0m"
    };
    println!(
        "Pass rate: {:.1}% (target: >{:.0}%) {}",
        summary.pass_rate * 100.0,
        summary.target_rate * 100.0,
        target_met
    );
    println!("Total time: {:.1}s", summary.total_time_ms as f64 / 1000.0);
    println!(
        "Avg time: {:.2}s/problem",
        summary.avg_time_ms as f64 / 1000.0
    );
    println!();
    println!("Commit: {}", summary.commit);
    println!("Timeout: {}s", summary.timeout_seconds);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_core::NaiveCpuGemmEngine;
    use std::cell::Cell;

    #[test]
    fn test_args_default_year() {
        let args = AcasxuBenchmarkArgs::default();
        assert_eq!(args.year, 2021, "Default year should be 2021");
    }

    #[test]
    fn test_args_default_branching() {
        // CRITICAL: Default branching is "input" to match α,β-CROWN's ACAS-Xu config.
        // This achieves 98.4% pass rate vs 28% with "width".
        // See reports/experiments/2026-02-03-acasxu-config-experiments.md
        let args = AcasxuBenchmarkArgs::default();
        assert_eq!(
            args.branching, "input",
            "Default branching should be 'input' to match α,β-CROWN"
        );
    }

    #[test]
    fn test_summary_add_result() {
        let mut summary = AcasxuSummary::new(3, 60, 2021, 3);

        summary.add_result(AcasxuResult {
            model: "model1.onnx".to_string(),
            property: "prop_1.vnnlib".to_string(),
            status: "Verified".to_string(),
            time_ms: 100,
            domains: 10,
            domains_verified: 6,
            error: None,
        });

        assert_eq!(summary.verified, 1);
        assert_eq!(summary.completed(), 1);

        summary.add_result(AcasxuResult {
            model: "model2.onnx".to_string(),
            property: "prop_1.vnnlib".to_string(),
            status: "Falsified".to_string(),
            time_ms: 200,
            domains: 5,
            domains_verified: 1,
            error: None,
        });

        assert_eq!(summary.falsified, 1);
        assert_eq!(summary.completed(), 2);
    }

    #[test]
    fn test_summary_benchmark_year_tracking() {
        // Verify benchmark year is correctly tracked in summary
        let summary_2021 = AcasxuSummary::new(10, 60, 2021, 10);
        assert_eq!(summary_2021.benchmark_year, 2021);

        let summary_2025 = AcasxuSummary::new(10, 60, 2025, 10);
        assert_eq!(summary_2025.benchmark_year, 2025);
    }

    #[test]
    fn test_summary_instance_count_vs_total() {
        // Verify instance_count (from CSV) vs total (after filtering) distinction
        let mut summary = AcasxuSummary::new(5, 60, 2021, 10);
        // instance_count = 10 (total in CSV)
        // total = 5 (after filtering)
        assert_eq!(summary.instance_count, 10);
        assert_eq!(summary.total, 5);

        // Adding results shouldn't change instance_count
        summary.add_result(AcasxuResult {
            model: "model.onnx".to_string(),
            property: "prop.vnnlib".to_string(),
            status: "Verified".to_string(),
            time_ms: 100,
            domains: 1,
            domains_verified: 1,
            error: None,
        });
        assert_eq!(
            summary.instance_count, 10,
            "instance_count should not change"
        );
    }

    #[test]
    fn test_include_results_json_serialization() {
        // Test that include_results=true includes results in JSON
        let mut summary = AcasxuSummary::new(2, 60, 2021, 2);
        summary.add_result(AcasxuResult {
            model: "model1.onnx".to_string(),
            property: "prop_1.vnnlib".to_string(),
            status: "Verified".to_string(),
            time_ms: 1000,
            domains: 10,
            domains_verified: 7,
            error: None,
        });

        // With results present, JSON should include "results" field
        let json_with_results = serde_json::to_string(&summary).unwrap();
        assert!(
            json_with_results.contains("\"results\""),
            "JSON should contain results field when not empty"
        );
        assert!(
            json_with_results.contains("model1.onnx"),
            "JSON should contain result details"
        );

        // After clearing results (simulating include_results=false), JSON should omit "results"
        summary.results.clear();
        let json_without_results = serde_json::to_string(&summary).unwrap();
        assert!(
            !json_without_results.contains("\"results\""),
            "JSON should omit results field when empty (skip_serializing_if)"
        );
    }

    #[test]
    fn test_gpu_bab_runtime_provides_concrete_engine_fallback_2343() {
        let runtime = GpuBabEngineRuntime {
            compute_device: None,
            fallback_reason: Some("wgpu unavailable".to_string()),
        };

        let summary = runtime.engine_summary();
        assert!(
            summary.contains("gemm=cpu-naive"),
            "Fallback summary must disclose concrete cpu engine: {summary}"
        );

        // Regression guard for #2343: GPU BaB dispatch must never receive `None`
        // for GEMM engine wiring. Even fallback mode exposes a concrete adapter.
        // Post-#3643: engine_arc returns None when no compute device — the verifier
        // construction path handles this by using BetaCrownVerifier::new (no engine).
        assert!(
            runtime.engine_arc().is_none(),
            "Fallback runtime should return None from engine_arc"
        );

        // CPU fallback GEMM is still functional when constructed directly.
        let cpu = NaiveCpuGemmEngine;
        let product = cpu
            .gemm_f32(1, 1, 1, &[2.0], &[3.0])
            .expect("cpu fallback GEMM should be executable");
        assert_eq!(product, vec![6.0]);
    }

    fn test_problem() -> AcasxuProblem {
        AcasxuProblem {
            model_path: PathBuf::from("model.onnx"),
            property_path: PathBuf::from("prop.vnnlib"),
            model_name: "model.onnx".to_string(),
            property_name: "prop.vnnlib".to_string(),
            timeout: 60,
        }
    }

    #[test]
    fn test_run_single_problem_cleans_up_after_success_3515() {
        let problem = test_problem();
        let cleanup_calls = Cell::new(0usize);

        let result = run_single_problem_with_hooks(
            &problem,
            || Ok((BabVerificationStatus::Verified, 7, 5)),
            || {
                cleanup_calls.set(cleanup_calls.get() + 1);
                Ok(())
            },
        );

        assert_eq!(
            cleanup_calls.get(),
            1,
            "per-problem GPU cleanup must run after successful verification",
        );
        assert_eq!(result.status, "Verified");
        assert_eq!(result.domains, 7);
        assert_eq!(result.domains_verified, 5);
        assert!(
            result.error.is_none(),
            "unexpected error: {:?}",
            result.error
        );
    }

    #[test]
    fn test_run_single_problem_cleans_up_after_error_3515() {
        let problem = test_problem();
        let cleanup_calls = Cell::new(0usize);

        let result = run_single_problem_with_hooks(
            &problem,
            || Err(anyhow::anyhow!("verification failed")),
            || {
                cleanup_calls.set(cleanup_calls.get() + 1);
                Ok(())
            },
        );

        assert_eq!(
            cleanup_calls.get(),
            1,
            "per-problem GPU cleanup must still run after verification errors",
        );
        assert_eq!(result.status, "Error");
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("verification failed")),
            "expected propagated verification error, got {:?}",
            result.error
        );
    }

    #[test]
    fn test_run_single_problem_reports_cleanup_error_3515() {
        let problem = test_problem();
        let cleanup_calls = Cell::new(0usize);

        let result = run_single_problem_with_hooks(
            &problem,
            || Ok((BabVerificationStatus::Verified, 7, 5)),
            || {
                cleanup_calls.set(cleanup_calls.get() + 1);
                Err(anyhow::anyhow!("cleanup failed"))
            },
        );

        assert_eq!(
            cleanup_calls.get(),
            1,
            "cleanup-error path must still invoke the per-problem cleanup closure",
        );
        assert_eq!(result.status, "Error");
        assert!(
            result.error.as_deref().is_some_and(|error| error.contains(
                "failed to clear GPU CROWN working set after verification: cleanup failed"
            )),
            "expected cleanup failure in error message, got {:?}",
            result.error
        );
    }

    #[test]
    fn test_run_single_problem_cleans_up_after_verification_panic_3515() {
        let problem = test_problem();
        let cleanup_calls = Cell::new(0usize);

        let result = run_single_problem_with_hooks(
            &problem,
            || panic!("verification panic"),
            || {
                cleanup_calls.set(cleanup_calls.get() + 1);
                Ok(())
            },
        );

        assert_eq!(
            cleanup_calls.get(),
            1,
            "cleanup must still run when verification unwinds",
        );
        assert_eq!(result.status, "Error");
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("verification panicked: verification panic")),
            "expected caught verification panic in error message, got {:?}",
            result.error
        );
    }

    #[test]
    fn test_run_single_problem_reports_cleanup_panic_3515() {
        let problem = test_problem();
        let cleanup_calls = Cell::new(0usize);

        let result = run_single_problem_with_hooks(
            &problem,
            || Ok((BabVerificationStatus::Verified, 7, 5)),
            || {
                cleanup_calls.set(cleanup_calls.get() + 1);
                panic!("cleanup panic");
            },
        );

        assert_eq!(
            cleanup_calls.get(),
            1,
            "cleanup panic must still prove the cleanup closure was entered exactly once",
        );
        assert_eq!(result.status, "Error");
        assert!(
            result.error.as_deref().is_some_and(|error| error.contains(
                "failed to clear GPU CROWN working set after verification: cleanup panicked: cleanup panic"
            )),
            "expected caught cleanup panic in error message, got {:?}",
            result.error
        );
    }

    #[test]
    fn test_run_single_problem_reports_verify_and_cleanup_failures_3515() {
        let problem = test_problem();
        let cleanup_calls = Cell::new(0usize);

        let result = run_single_problem_with_hooks(
            &problem,
            || Err(anyhow::anyhow!("verification failed")),
            || {
                cleanup_calls.set(cleanup_calls.get() + 1);
                Err(anyhow::anyhow!("cleanup failed"))
            },
        );

        assert_eq!(
            cleanup_calls.get(),
            1,
            "combined verification/cleanup failures must still run cleanup once",
        );
        assert_eq!(result.status, "Error");
        assert!(
            result.error.as_deref().is_some_and(|error| error.contains(
                "verification failed; failed to clear GPU CROWN working set after verification: cleanup failed"
            )),
            "expected combined verification and cleanup failures, got {:?}",
            result.error
        );
    }

    #[test]
    fn test_run_single_problem_reports_verify_panic_and_cleanup_failure_3515() {
        let problem = test_problem();
        let cleanup_calls = Cell::new(0usize);

        let result = run_single_problem_with_hooks(
            &problem,
            || panic!("verification panic"),
            || {
                cleanup_calls.set(cleanup_calls.get() + 1);
                Err(anyhow::anyhow!("cleanup failed"))
            },
        );

        assert_eq!(
            cleanup_calls.get(),
            1,
            "cleanup failure must still be reported when verification unwinds",
        );
        assert_eq!(result.status, "Error");
        assert!(
            result.error.as_deref().is_some_and(|error| {
                error.contains("verification panicked: verification panic")
                    && error.contains(
                        "failed to clear GPU CROWN working set after verification: cleanup failed",
                    )
            }),
            "expected combined verification panic and cleanup failure, got {:?}",
            result.error
        );
    }

    #[test]
    fn test_target_rate_catches_regressions_2605() {
        // Best measured: 98.4% (183/186) on 2025-02-03. Target must be close enough
        // to detect meaningful regressions (>= 0.98) but leave room for noise.
        let summary = AcasxuSummary::new(186, 300, 2024, 186);
        assert!(
            summary.target_rate >= 0.98,
            "target_rate {:.2} is too low — regressions from 98.4% would be invisible",
            summary.target_rate
        );
        assert!(
            summary.target_rate <= 0.984,
            "target_rate {:.3} exceeds best measurement — would false-alarm on noise",
            summary.target_rate
        );
    }
}
