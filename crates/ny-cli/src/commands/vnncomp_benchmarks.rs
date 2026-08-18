// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNN-COMP benchmark download/status helpers.

use anyhow::{anyhow, bail, Result};
use clap::Subcommand;
use serde::Serialize;
use serde_json::json;
use std::collections::VecDeque;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const DEFAULT_YEARS: [u32; 5] = [2021, 2023, 2024, 2025, 2026];
const ALL_YEARS: [u32; 6] = [2021, 2022, 2023, 2024, 2025, 2026];

/// Result of a child process whose output and wall time were bounded by the
/// benchmark harness.
#[derive(Debug)]
pub(crate) struct BoundedChildOutput {
    pub(crate) success: bool,
    pub(crate) timed_out: bool,
    /// Wall time through direct-child completion, excluding stderr-drain cleanup.
    pub(crate) elapsed: Duration,
    pub(crate) stderr_tail: String,
}

#[cfg(unix)]
fn isolate_child_process_group(command: &mut Command) {
    std::os::unix::process::CommandExt::process_group(command, 0);
}

#[cfg(not(unix))]
fn isolate_child_process_group(_command: &mut Command) {}

/// Terminate descendants that inherited the benchmark child's process group.
///
/// The direct child may already have exited; POSIX process groups remain
/// addressable while any descendant is alive. A plain `Child::kill` cannot
/// close pipes held by those descendants and can leak both work and drainer
/// threads across a long sweep.
#[cfg(unix)]
fn kill_child_process_group(child: &std::process::Child) {
    let Some(pid) = i32::try_from(child.id())
        .ok()
        .and_then(rustix::process::Pid::from_raw)
    else {
        return;
    };
    let _ = rustix::process::kill_process_group(pid, rustix::process::Signal::KILL);
}

#[cfg(not(unix))]
fn kill_child_process_group(_child: &std::process::Child) {}

fn drain_bounded_tail<R: Read>(mut reader: R, limit: usize) -> io::Result<Vec<u8>> {
    let mut tail = VecDeque::with_capacity(limit);
    let mut chunk = [0_u8; 4096];
    loop {
        let read = reader.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        for &byte in &chunk[..read] {
            if tail.len() == limit {
                tail.pop_front();
            }
            if limit != 0 {
                tail.push_back(byte);
            }
        }
    }
    Ok(tail.into_iter().collect())
}

/// Run a benchmark child without allowing it to retain unbounded output or
/// outlive the harness watchdog.
///
/// Stdout is discarded (the VNN-COMP result file is the protocol); stderr is
/// continuously drained on a helper thread while retaining only its final
/// `stderr_limit` bytes. This avoids both pipe deadlock and the unbounded memory
/// allocation performed by [`Command::output`].
pub(crate) fn run_bounded_child(
    command: &mut Command,
    timeout: Duration,
    stderr_limit: usize,
) -> io::Result<BoundedChildOutput> {
    command.stdout(Stdio::null()).stderr(Stdio::piped());
    isolate_child_process_group(command);
    let mut child = command.spawn()?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("benchmark child stderr pipe was not created"))?;
    let (stderr_sender, stderr_receiver) = std::sync::mpsc::sync_channel(1);
    if let Err(error) = std::thread::Builder::new()
        .name("ny-benchmark-stderr".into())
        .spawn(move || {
            let _ = stderr_sender.send(drain_bounded_tail(stderr, stderr_limit));
        })
    {
        kill_child_process_group(&child);
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }

    let started = Instant::now();
    let (success, timed_out) = loop {
        let status = match child.try_wait() {
            Ok(status) => status,
            Err(error) => {
                kill_child_process_group(&child);
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };
        if let Some(status) = status {
            kill_child_process_group(&child);
            break (status.success(), false);
        }
        if started.elapsed() >= timeout {
            // The child may exit between try_wait and kill. Either way, wait
            // reaps it and the harness records that its watchdog elapsed. Kill
            // the isolated process group first so descendants cannot survive
            // with inherited pipes or computation.
            kill_child_process_group(&child);
            let _ = child.kill();
            let _ = child.wait()?;
            break (false, true);
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let elapsed = started.elapsed();

    // A grandchild can inherit the pipe after the direct child exits. Never
    // let that defeat the wall-time bound: wait briefly for the normal EOF
    // path, then detach the still-bounded drain thread.
    let stderr = match stderr_receiver.recv_timeout(Duration::from_millis(100)) {
        Ok(result) => result?,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            b"[stderr pipe remained open in a descendant]".to_vec()
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            return Err(io::Error::other(
                "benchmark stderr reader stopped unexpectedly",
            ));
        }
    };
    Ok(BoundedChildOutput {
        success,
        timed_out,
        elapsed,
        stderr_tail: String::from_utf8_lossy(&stderr).into_owned(),
    })
}

// `Run` carries more flags than its siblings and now exceeds the 64-byte spread
// clippy.toml sets. Boxing is the lint's remedy for many live values; exactly one
// of these exists per process, parsed from argv and destructured immediately, so
// the indirection would buy nothing and cost a level of pattern matching.
#[allow(clippy::large_enum_variant)]
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

    /// Sweep a VNN-COMP category (or all of them) and write an official-format
    /// `results.csv`.
    ///
    /// Per-instance budgets come from the benchmark's own `instances.csv`.
    /// This reproduces the budget policy, but not the organizers' hardware,
    /// environment, or witness validation. Handles both the flat (2021-2025)
    /// and versioned (2026) corpus layouts, and both single-model and
    /// paired/relational instance rows.
    Run {
        /// VNN-COMP year to sweep.
        #[arg(long, default_value_t = 2026)]
        year: u32,

        /// Select a versioned VNN-LIB corpus (`1.0` or `2.0`). Required when a
        /// category contains parallel instance lists; unambiguous layouts need
        /// no selector.
        #[arg(long, value_enum)]
        vnnlib_version: Option<super::bench_vnncomp::VnnlibVersion>,

        /// Category to run. Repeat for several; omit to sweep every category.
        #[arg(long = "category")]
        categories: Vec<String>,

        /// Official six-column results CSV to write. Because this runner has no
        /// separate preparation phase, `prepare_time` is recorded as zero.
        /// Authoritative row state is written as `<name>.metadata.jsonl`; pinned
        /// provenance is written as `<name>.manifest.json`.
        /// Defaults to `reports/sweeps/vnncomp<year>-results.csv`.
        #[arg(long)]
        output: Option<PathBuf>,

        /// LOWER the per-instance budget to at most this many seconds. Never
        /// raises it. Any capped row is flagged, and the summary states that the
        /// result is a lower bound rather than a competition-comparable score.
        #[arg(long)]
        timeout_cap: Option<u64>,

        /// Run only the first N instances of each category.
        #[arg(long)]
        limit: Option<usize>,

        /// Run EXACTLY the rows named in this file: one `category,onnx,vnnlib`
        /// per line, `#` comments and blank lines ignored.
        ///
        /// The three fields are the first three columns of the `results.csv`
        /// this command emits and must match the category's `instances.csv`
        /// verbatim. A named row the corpus does not contain is an ERROR, never
        /// a skip: a gate set that quietly shrinks reports the same clean sweep
        /// as one that ran in full.
        ///
        /// Refused together with `--limit` — a subset that then takes the first
        /// N of itself depends on CSV order, which is not what either flag says.
        #[arg(long)]
        instances: Option<PathBuf>,

        /// Preset directory (`vnncomp*/{category}.yaml`), forwarded to each run.
        #[arg(long)]
        configs_dir: Option<PathBuf>,

        /// Continue a provenance-compatible authoritative metadata bank,
        /// regenerating the CSV and retrying prior harness-error occurrences.
        #[arg(long, default_value_t = false)]
        resume: bool,

        /// Replace an existing CSV/state/manifest bank. Refused together with
        /// `--resume`; without either flag, pre-existing artifacts are protected.
        #[arg(long, default_value_t = false)]
        overwrite: bool,

        /// Emit a machine-readable summary instead of per-instance lines.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// List what would run, without running anything.
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Apply a lever assignment to every child of this sweep: `--lever
        /// NAME=VALUE`, repeatable.
        ///
        /// Use this instead of exporting a variable into the shell. A shell
        /// export is invisible to the artifact, is inherited by every later
        /// process, and silently mixes arms across a `--resume`. Passing the arm
        /// here makes it data: applied per child, recorded on every row, and
        /// sealed into the manifest so a resume under a different arm refuses.
        ///
        /// Each assignment is validated against `ny_levers::space` before
        /// anything runs. An axis the interaction lattice proves INERT is
        /// rejected rather than measured, because measuring it would spend a full
        /// instance budget re-measuring the baseline and reporting it as a
        /// treatment. `Class::Unsafe` axes are refused outright.
        #[arg(long = "lever", value_name = "NAME=VALUE")]
        levers: Vec<String>,

        /// Verdict cache participation: `off` (default), `read`, `read-write`.
        ///
        /// A cached verdict is only sound because the key carries everything it
        /// depends on — binary, configuration, CATEGORY (it names the preset the
        /// child loads), model, property, budget, arm, backend, host, and the
        /// ambient lever set. Because the budget is in the key, a `timeout`
        /// measured at a SMALLER budget cannot even be addressed by a larger
        /// request; on top of the key, a row measured under heavy load is never
        /// served and never stored. See `vnncomp_sweep/cache.rs` for the exact
        /// scope of both admission rules.
        ///
        /// A served row is marked `from_cache` in the metadata bank: a replay
        /// carries the flight record of a different process, and no gate may be
        /// forced to read it as evidence that a child ran.
        #[arg(
            long,
            default_value = "off",
            value_name = "off|read|read-write",
            value_parser = parse_cache_mode
        )]
        cache: super::vnncomp_sweep::cache::CacheMode,

        /// Verdict cache root. Defaults to `reports/sweeps/verdict-cache`.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },

    /// Report progress from a sweep's results: solved counts, NEW SOLVES,
    /// REGRESSIONS, and (with `--official`) normalized per-benchmark scores.
    ///
    /// With `--official`, raw CSVs are modeled under the +10/-150 rule and
    /// benchmark scores are normalized winner-relative; the report states its
    /// witness/timeout limitations. Without it, the command reports solve
    /// counts and baseline deltas only. A sat<->unsat flip is a soundness
    /// alarm, never progress.
    Score {
        /// Results CSV (or a directory of them, e.g. `reports/measured`).
        /// Repeatable.
        #[arg(long = "results", required = true)]
        results: Vec<PathBuf>,

        /// Baseline to diff against — the previous bank. Omit to treat every
        /// solved row as new.
        #[arg(long)]
        baseline: Option<PathBuf>,

        /// Released results corpus supplying the comparison field for a modeled,
        /// winner-relative category total. Without it, solve counts are reported
        /// and NO normalized total is invented.
        #[arg(long, requires = "year")]
        official: Option<PathBuf>,

        /// VNN-COMP edition whose released category split to use. Required with
        /// `--official`; it is explicit because category membership changed
        /// between 2025 and 2026.
        #[arg(long, value_enum)]
        year: Option<super::vnncomp_score::ScoreYear>,

        /// Leaderboard to model. Regular and extended categories have separate
        /// normalized totals and must never be combined.
        #[arg(long, value_enum, default_value = "regular")]
        track: super::vnncomp_score::ScoreTrack,

        /// Emit the full report as JSON.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// Max instances to list per benchmark in the new-solve / regression
        /// sections.
        #[arg(long, default_value_t = 10)]
        show_rows: usize,

        /// How SAT rows are treated when scoring, worth more than 100 points.
        ///
        /// `assumed-valid` (default) treats every SAT as a strictly valid
        /// counterexample, so an UNSAT it contradicts takes the real -150.
        /// `unvalidated` opens no counterexample, so nothing convicts and every
        /// claimed verdict earns +10 — this reproduces the PUBLISHED 2025 board's
        /// totals numerically, and therefore rewards unsound tools. Use it to
        /// answer "where would NY have placed", nothing else. (It does NOT mean
        /// the organizers skipped validation: they checked 5,744 counterexamples
        /// and applied 62 -150 penalties. See vnncomp_score::Witnesses.)
        #[arg(long, value_enum, default_value = "assumed-valid")]
        witnesses: super::vnncomp_score::Witnesses,

        /// Enforce official per-instance timeouts from this corpus year's
        /// `instances.csv` files. Repeatable. VNN-COMP budgets are PER
        /// INSTANCE (nn4sys spans 20s-800s), so a solved row recorded above
        /// its own budget is a phantom point: it is reported and excluded from
        /// the modeled score. Omit to leave runtimes unchecked, which the
        /// report then states.
        #[arg(long = "budget-year")]
        budget_years: Vec<u32>,
    },

    /// Metamorphic overfitting check: re-run each instance under a
    /// semantics-preserving rename and require a decided verdict to be unchanged.
    ///
    /// Competition instances are regenerated from a seed, so any behaviour keyed
    /// to an instance's name or path is inert on the scored set and is
    /// overfitting. A verdict that changes under rename is either that, or
    /// nondeterminism — both invalidate progress measured from a sweep. Missing
    /// assets, child errors, timeouts/unknowns, and a zero-valid-case run fail
    /// the command instead of being reported as a pass.
    Reseed {
        /// VNN-COMP year.
        #[arg(long, default_value_t = 2026)]
        year: u32,

        /// Select a versioned VNN-LIB corpus (`1.0` or `2.0`). Required when a
        /// category contains parallel instance lists; unambiguous layouts need
        /// no selector.
        #[arg(long, value_enum)]
        vnnlib_version: Option<super::bench_vnncomp::VnnlibVersion>,

        /// Category to check. Repeat for several; omit for all.
        #[arg(long = "category")]
        categories: Vec<String>,

        /// Check only the first N instances of each category.
        #[arg(long)]
        limit: Option<usize>,

        /// LOWER the per-instance budget (each instance is run twice).
        #[arg(long)]
        timeout_cap: Option<u64>,

        /// Preset directory forwarded to both runs.
        #[arg(long)]
        configs_dir: Option<PathBuf>,

        /// Emit the report as JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
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

/// `--cache` goes through the cache module's OWN parser, not a clap `ValueEnum`.
///
/// The accepted spellings are part of that module's contract, and a second,
/// derived parser is how `read-write` quietly becomes `ReadWrite` or `Read-Write`
/// as well. This adapter only converts the error type clap needs.
fn parse_cache_mode(
    raw: &str,
) -> std::result::Result<super::vnncomp_sweep::cache::CacheMode, String> {
    super::vnncomp_sweep::cache::CacheMode::parse(raw).map_err(|error| error.to_string())
}

/// Parse `--lever NAME=VALUE` pairs and refuse anything the search space says is
/// unsafe or inert, BEFORE any instance runs.
///
/// Refusing early is the whole point. An inert assignment is not a cheap
/// mistake: it spends a full instance budget re-measuring the baseline and then
/// reports the result as a treatment. That failure mode has produced several
/// confident-but-empty conclusions in this repository, which is why
/// `ny_levers::space::expand` exists and why it is consulted here rather than in
/// the runner.
fn parse_and_validate_arm(levers: &[String]) -> Result<Vec<(String, String)>> {
    use std::collections::BTreeMap;
    if levers.is_empty() {
        return Ok(Vec::new());
    }
    let axes: BTreeMap<&str, &ny_levers::space::Axis> = ny_levers::space::axes()
        .iter()
        .map(|axis| (axis.name, axis))
        .collect();

    let mut sample: BTreeMap<&'static str, String> = BTreeMap::new();
    let mut seen: Vec<String> = Vec::new();
    for raw in levers {
        let (name, value) = raw
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--lever expects NAME=VALUE, got {raw:?}"))?;
        if seen.iter().any(|previous| previous == name) {
            anyhow::bail!("--lever {name} given more than once; an arm must be unambiguous");
        }
        seen.push(name.to_string());
        // Resolve to the 'static name from the space so the sample can borrow it,
        // and so a typo is caught here instead of becoming a silent no-op.
        let axis = axes.get(name).copied().ok_or_else(|| {
            let unsafe_hit = ny_levers::space::unsafe_axes()
                .iter()
                .find(|axis| axis.name == name);
            match unsafe_hit {
                Some(axis) => anyhow::anyhow!(
                    "--lever {name} is Class::Unsafe and must never be set by an automated \
                     search: {}",
                    axis.why
                ),
                None => anyhow::anyhow!(
                    "--lever {name} is not a searchable axis. It may be undeclared, \
                     instrument-only (telemetry perturbs a deadline-sensitive run and is \
                     never a treatment arm), or test-only. See ny_levers::space."
                ),
            }
        })?;
        sample.insert(axis.name, value.to_string());
    }

    let expanded = ny_levers::space::expand(&sample)
        .map_err(|inert| anyhow::anyhow!("refusing to measure an inert arm: {inert}"))?;

    // Delivery is a warning, not an error: measuring an EnvOnly axis is a
    // legitimate experiment. Shipping its result is what requires a preset key.
    for (name, _) in &expanded {
        if let Some(axis) = axes.get(name.as_str()) {
            if axis.deliver == ny_levers::space::Deliver::EnvOnly {
                eprintln!(
                    "[sweep-arm] WARNING: {name} is EnvOnly. The scored entry point exports \
                     exactly one NY_* variable, so whatever this measures CANNOT reach a \
                     scored run until a typed preset key exists for it."
                );
            }
        }
    }
    Ok(expanded)
}

pub(crate) fn handle_benchmark_assets_command(action: BenchmarkAssetsAction) -> Result<()> {
    match action {
        BenchmarkAssetsAction::Download { years, all } => {
            handle_vnncomp_benchmarks_command(years, all, false)
        }
        BenchmarkAssetsAction::Run {
            year,
            vnnlib_version,
            categories,
            output,
            timeout_cap,
            limit,
            instances,
            configs_dir,
            resume,
            overwrite,
            json,
            dry_run,
            levers,
            cache,
            cache_dir,
        } => {
            let arm = parse_and_validate_arm(&levers)?;
            let opts = super::vnncomp_sweep::SweepOptions {
                year,
                vnnlib_version,
                categories,
                output,
                timeout_cap,
                limit,
                instances,
                configs_dir,
                resume,
                overwrite,
                json,
                dry_run,
                arm,
                cache,
                cache_dir,
            };
            let summary = super::vnncomp_sweep::run_sweep(&opts)?;
            if summary.error > 0 {
                bail!(
                    "{} benchmark instance(s) failed at the harness boundary; \
                     results and metadata were preserved for diagnosis and --resume",
                    summary.error
                );
            }
            Ok(())
        }
        BenchmarkAssetsAction::Score {
            results,
            baseline,
            official,
            year,
            track,
            json,
            show_rows,
            budget_years,
            witnesses,
        } => {
            let opts = super::vnncomp_score::ScoreOptions {
                results,
                baseline,
                official,
                year,
                track,
                json,
                show_rows,
                budget_years,
                witnesses,
            };
            super::vnncomp_score::run_score(&opts).map(|_| ())
        }
        BenchmarkAssetsAction::Reseed {
            year,
            vnnlib_version,
            categories,
            limit,
            timeout_cap,
            configs_dir,
            json,
        } => {
            let opts = super::vnncomp_reseed::ReseedOptions {
                year,
                vnnlib_version,
                categories,
                limit,
                timeout_cap,
                configs_dir,
                json,
            };
            super::vnncomp_reseed::run_reseed(&opts).map(|_| ())
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

    use super::super::vnncomp_sweep::cache::CacheMode;
    use clap::{CommandFactory, Parser};

    #[derive(Parser)]
    struct AssetsCli {
        #[command(subcommand)]
        action: BenchmarkAssetsAction,
    }

    /// `--cache` must reach `CacheMode::parse` unmodified, and its default must
    /// be a value that parser accepts — a `default_value` clap cannot parse is a
    /// panic on first invocation, not a compile error.
    #[test]
    fn sweep_subset_and_cache_flags_parse_through_the_modules_own_parser() {
        AssetsCli::command().debug_assert();

        let parsed = AssetsCli::try_parse_from([
            "assets",
            "run",
            "--instances",
            "moat41.txt",
            "--cache",
            "read-write",
        ])
        .expect("subset and cache flags parse");
        match parsed.action {
            BenchmarkAssetsAction::Run {
                instances,
                cache,
                cache_dir,
                limit,
                ..
            } => {
                assert_eq!(instances, Some(PathBuf::from("moat41.txt")));
                assert_eq!(cache, CacheMode::ReadWrite);
                assert_eq!(cache_dir, None);
                assert_eq!(limit, None);
            }
            _ => panic!("expected the run subcommand"),
        }

        let default = AssetsCli::try_parse_from(["assets", "run"]).expect("defaults parse");
        match default.action {
            BenchmarkAssetsAction::Run {
                cache, instances, ..
            } => {
                assert_eq!(cache, CacheMode::Off, "the cache must be opt-in");
                assert_eq!(instances, None);
            }
            _ => panic!("expected the run subcommand"),
        }

        // A spelling the cache module does not accept must be refused here, not
        // silently widened into one it does.
        let rejected = match AssetsCli::try_parse_from(["assets", "run", "--cache", "ReadWrite"]) {
            Ok(_) => panic!("an unknown cache mode must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(
            rejected.contains("off|read|read-write"),
            "the refusal must state the accepted spellings: {rejected}"
        );
    }

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

    #[cfg(unix)]
    #[test]
    fn bounded_child_retains_only_the_stderr_tail() {
        let mut command = Command::new("sh");
        command.arg("-c").arg(
            "printf 'discard-this-prefix'; \
             printf 'abcdefghijklmnopqrstuvwxyz' >&2",
        );
        let output =
            run_bounded_child(&mut command, Duration::from_secs(2), 8).expect("bounded child");
        assert!(output.success);
        assert!(!output.timed_out);
        assert_eq!(output.stderr_tail, "stuvwxyz");
    }

    #[cfg(unix)]
    #[test]
    fn bounded_child_watchdog_kills_a_hung_process() {
        let mut command = Command::new("sleep");
        command.arg("30");
        let started = Instant::now();
        let output =
            run_bounded_child(&mut command, Duration::from_millis(50), 128).expect("watchdog run");
        assert!(!output.success);
        assert!(output.timed_out);
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "watchdog failed to stop the child promptly"
        );
    }

    #[cfg(unix)]
    #[test]
    fn inherited_stderr_pipe_in_a_descendant_cannot_block_return() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_file = dir.path().join("descendant.pid");
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg("(trap '' HUP; exec sleep 30) >&2 & printf '%s' \"$!\" > \"$1\"; exit 0")
            .arg("sh")
            .arg(&pid_file);
        let started = Instant::now();
        let output = run_bounded_child(&mut command, Duration::from_secs(2), 128)
            .expect("bounded child with inherited pipe");
        let elapsed = started.elapsed();

        let pid = std::fs::read_to_string(&pid_file).expect("descendant pid");
        let mut alive = true;
        for _ in 0..50 {
            let status = Command::new("kill")
                .arg("-0")
                .arg(pid.trim())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .expect("probe descendant");
            alive = status.success();
            if !alive {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        if alive {
            let _ = Command::new("kill")
                .args(["-9", pid.trim()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }

        assert!(output.success);
        assert!(!alive, "the benchmark child's process tree must be reaped");
        assert!(
            elapsed < Duration::from_secs(1),
            "inherited stderr pipe blocked bounded return for {elapsed:?}"
        );
    }
}
