// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use ny_core::{NyError, Result};

use crate::benchmark_support::crown_backward_cases::BenchCase;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementArgs {
    case_filter: Option<String>,
    output_path: Option<PathBuf>,
    profile_gpu: bool,
    profile_host: bool,
    production_only: bool,
    /// Enable GraphNetwork measurement phases (#3716).
    ///
    /// When set, the benchmark converts each sequential workload to a
    /// `GraphNetwork` and measures the graph CROWN-IBP collection path that
    /// real VNN-COMP ONNX models exercise. Includes IBP forward and
    /// CROWN-IBP collection (with and without engine) — the actual code path
    /// that competition workloads run through `verify_graph`.
    graph: bool,
    /// Skip the known-slow CPU graph collection lane while still measuring the
    /// graph IBP + engine-accelerated collection path used by the regression
    /// checker for GPU evidence refreshes.
    graph_engine_only: bool,
    /// Also run the full graph CROWN backward (`--graph-full`). Extremely slow
    /// (43+ min for metaroom) and NOT the VNN-COMP code path (#3716).
    /// Requires `--case <name>` so the benchmark cannot accidentally launch a
    /// multi-hour sweep across every workload.
    graph_full: bool,
}

impl MeasurementArgs {
    pub fn parse_from<I>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = String>,
    {
        let mut case_filter = None;
        let mut output_path = None;
        let mut profile_gpu = false;
        let mut profile_host = false;
        let mut production_only = false;
        let mut graph = false;
        let mut graph_engine_only = false;
        let mut graph_full = false;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--case" => {
                    let value = args.next().ok_or_else(|| {
                        NyError::InvalidSpec("expected a case name after --case".into())
                    })?;
                    case_filter = Some(value);
                }
                "--output" => {
                    let value = args.next().ok_or_else(|| {
                        NyError::InvalidSpec("expected a file path after --output".into())
                    })?;
                    output_path = Some(PathBuf::from(value));
                }
                "--profile-gpu" => {
                    profile_gpu = true;
                }
                "--profile-host" => {
                    profile_host = true;
                }
                "--production-only" => {
                    production_only = true;
                }
                "--graph" => {
                    graph = true;
                }
                "--graph-engine-only" => {
                    graph = true;
                    graph_engine_only = true;
                }
                "--graph-full" => {
                    graph = true;
                    graph_full = true;
                }
                unknown => {
                    return Err(NyError::InvalidSpec(format!(
                        "unsupported argument `{unknown}`"
                    )));
                }
            }
        }

        if production_only && graph {
            return Err(NyError::InvalidSpec(
                "--production-only cannot be combined with --graph or --graph-full".into(),
            ));
        }

        if graph_full && case_filter.is_none() {
            return Err(NyError::InvalidSpec(
                "--graph-full requires --case <name> to avoid a multi-hour full graph sweep".into(),
            ));
        }

        if graph_engine_only && graph_full {
            return Err(NyError::InvalidSpec(
                "--graph-engine-only cannot be combined with --graph-full".into(),
            ));
        }

        Ok(Self {
            case_filter,
            output_path,
            profile_gpu,
            profile_host,
            production_only,
            graph,
            graph_engine_only,
            graph_full,
        })
    }

    #[must_use]
    pub fn case_filter(&self) -> Option<&str> {
        self.case_filter.as_deref()
    }

    #[must_use]
    pub fn output_path(&self) -> Option<&Path> {
        self.output_path.as_deref()
    }

    #[must_use]
    pub fn profile_gpu(&self) -> bool {
        self.profile_gpu
    }

    #[must_use]
    pub fn profile_host(&self) -> bool {
        self.profile_host
    }

    #[must_use]
    pub fn production_only(&self) -> bool {
        self.production_only
    }

    /// Whether to include GraphNetwork measurement phases (#3716).
    #[must_use]
    pub fn graph(&self) -> bool {
        self.graph
    }

    /// Whether to skip the CPU graph collection lane and keep only the graph
    /// GPU evidence rows.
    #[must_use]
    pub fn graph_engine_only(&self) -> bool {
        self.graph_engine_only
    }

    /// Whether to include the expensive full graph CROWN backward (#3716).
    #[must_use]
    pub fn graph_full(&self) -> bool {
        self.graph_full
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementStatus {
    Measured,
    Skipped,
    Failed,
}

impl MeasurementStatus {
    #[must_use]
    fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Skipped => "skipped",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuPhaseOutcome {
    Measured,
    Skipped,
}

pub struct MeasurementRow<'a> {
    case: &'a str,
    phase: &'a str,
    seconds: Option<f64>,
    parameter_count: usize,
    estimated_cpu_peak_bytes: usize,
    cpu_dense_budget_bytes: usize,
    status: MeasurementStatus,
    detail: &'a str,
}

impl<'a> MeasurementRow<'a> {
    #[must_use]
    pub fn measured(
        case: &'a BenchCase,
        phase: &'a str,
        seconds: f64,
        cpu_dense_budget_bytes: usize,
    ) -> Self {
        Self {
            case: case.name(),
            phase,
            seconds: Some(seconds),
            parameter_count: case.parameter_count(),
            estimated_cpu_peak_bytes: case.estimated_cpu_peak_bytes(),
            cpu_dense_budget_bytes,
            status: MeasurementStatus::Measured,
            detail: "",
        }
    }

    #[must_use]
    pub fn measured_with_detail(
        case: &'a BenchCase,
        phase: &'a str,
        seconds: f64,
        cpu_dense_budget_bytes: usize,
        detail: &'a str,
    ) -> Self {
        Self {
            case: case.name(),
            phase,
            seconds: Some(seconds),
            parameter_count: case.parameter_count(),
            estimated_cpu_peak_bytes: case.estimated_cpu_peak_bytes(),
            cpu_dense_budget_bytes,
            status: MeasurementStatus::Measured,
            detail,
        }
    }

    #[must_use]
    pub fn skipped(
        case: &'a BenchCase,
        phase: &'a str,
        cpu_dense_budget_bytes: usize,
        detail: &'a str,
    ) -> Self {
        Self {
            case: case.name(),
            phase,
            seconds: None,
            parameter_count: case.parameter_count(),
            estimated_cpu_peak_bytes: case.estimated_cpu_peak_bytes(),
            cpu_dense_budget_bytes,
            status: MeasurementStatus::Skipped,
            detail,
        }
    }

    #[must_use]
    pub fn failed(
        case: &'a BenchCase,
        phase: &'a str,
        cpu_dense_budget_bytes: usize,
        detail: &'a str,
    ) -> Self {
        Self {
            case: case.name(),
            phase,
            seconds: None,
            parameter_count: case.parameter_count(),
            estimated_cpu_peak_bytes: case.estimated_cpu_peak_bytes(),
            cpu_dense_budget_bytes,
            status: MeasurementStatus::Failed,
            detail,
        }
    }
}

pub fn create_output_file(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| {
                NyError::InternalError(format!(
                    "failed to create output directory `{}`: {err}",
                    parent.display()
                ))
            })?;
        }
    }

    let file = File::create(path).map_err(|err| {
        NyError::InternalError(format!(
            "failed to create output file `{}`: {err}",
            path.display()
        ))
    })?;
    Ok(BufWriter::new(file))
}

pub fn write_csv_header(out: &mut impl Write) -> Result<()> {
    writeln!(
        out,
        "case,phase,seconds,parameter_count,estimated_cpu_peak_bytes,cpu_dense_budget_bytes,status,detail"
    )
    .map_err(|err| NyError::InternalError(format!("failed to write csv header: {err}")))?;
    out.flush()
        .map_err(|err| NyError::InternalError(format!("failed to flush csv header: {err}")))
}

pub fn write_csv_row(out: &mut impl Write, row: &MeasurementRow<'_>) -> Result<()> {
    let seconds = row
        .seconds
        .map(|value| format!("{value:.6}"))
        .unwrap_or_default();
    writeln!(
        out,
        "{case},{phase},{seconds},{parameter_count},{estimated_cpu_peak_bytes},{cpu_dense_budget_bytes},{status},{detail}",
        case = row.case,
        phase = row.phase,
        seconds = seconds,
        parameter_count = row.parameter_count,
        estimated_cpu_peak_bytes = row.estimated_cpu_peak_bytes,
        cpu_dense_budget_bytes = row.cpu_dense_budget_bytes,
        status = row.status.as_str(),
        detail = row.detail,
    )
    .map_err(|err| NyError::InternalError(format!("failed to write measurement row: {err}")))?;
    out.flush()
        .map_err(|err| NyError::InternalError(format!("failed to flush measurement row: {err}")))
}

pub fn write_measured_phase(
    out: &mut impl Write,
    case: &BenchCase,
    phase: &str,
    seconds: f64,
    cpu_dense_budget_bytes: usize,
) -> Result<()> {
    write_csv_row(
        out,
        &MeasurementRow::measured(case, phase, seconds, cpu_dense_budget_bytes),
    )
}

pub fn write_measured_phase_with_detail(
    out: &mut impl Write,
    case: &BenchCase,
    phase: &str,
    seconds: f64,
    cpu_dense_budget_bytes: usize,
    detail: &str,
) -> Result<()> {
    write_csv_row(
        out,
        &MeasurementRow::measured_with_detail(case, phase, seconds, cpu_dense_budget_bytes, detail),
    )
}

pub fn measure_or_skip_cpu_phase(
    out: &mut impl Write,
    case: &BenchCase,
    cpu_dense_budget_bytes: usize,
    measure_seconds: impl FnOnce() -> Result<f64>,
) -> Result<CpuPhaseOutcome> {
    if case.estimated_cpu_peak_bytes() > cpu_dense_budget_bytes {
        write_csv_row(
            out,
            &MeasurementRow::skipped(
                case,
                "cpu_production",
                cpu_dense_budget_bytes,
                "dense_peak_exceeds_budget",
            ),
        )?;
        return Ok(CpuPhaseOutcome::Skipped);
    }

    let seconds = measure_seconds()?;
    write_measured_phase(out, case, "cpu_production", seconds, cpu_dense_budget_bytes)?;
    Ok(CpuPhaseOutcome::Measured)
}

pub fn write_failed_phase(
    out: &mut impl Write,
    case: &BenchCase,
    phase: &str,
    cpu_dense_budget_bytes: usize,
    detail: &str,
) -> Result<()> {
    write_csv_row(
        out,
        &MeasurementRow::failed(case, phase, cpu_dense_budget_bytes, detail),
    )
}
