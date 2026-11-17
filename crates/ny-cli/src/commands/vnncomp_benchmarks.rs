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
    /// Print the tunable-parameter search space (input to tuning scripts)
    ///
    /// Read from the binary, so a search always matches the executable that
    /// will run the benchmarks.
    Space {
        /// Emit the complete machine-readable search contract.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Download VNN-COMP benchmark repositories.
    Download {
        /// VNN-COMP years to download. Defaults to 2021 and 2023-2026.
        years: Vec<u32>,

        /// Include optional/historical years, including 2022.
        #[arg(long, default_value_t = false)]
        all: bool,
    },

    /// Run VNN-COMP instances at their official budgets and write results.csv
    ///
    /// Budgets come from each benchmark's `instances.csv`. This matches the
    /// competition's budget policy, not its hardware, environment, or witness
    /// validation.
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

        /// Results CSV to write (default: `reports/sweeps/vnncomp<YEAR>-results.csv`)
        ///
        /// The CSV uses the official six columns, with `prepare_time` always 0.
        /// Row state goes to `<name>.metadata.jsonl` and provenance to
        /// `<name>.manifest.json`.
        #[arg(long)]
        output: Option<PathBuf>,

        /// Lower each instance's budget to at most this many seconds (never
        /// raises it); capped results are flagged as lower bounds
        #[arg(long)]
        timeout_cap: Option<u64>,

        /// Run only the first N instances of each category.
        #[arg(long)]
        limit: Option<usize>,

        /// Run exactly the rows in this file: one `category,onnx,vnnlib` per line
        ///
        /// Each line must match the category's `instances.csv` exactly; a row
        /// the corpus does not contain is an error, not a skip. `#` comments
        /// and blank lines are ignored. Cannot be combined with --limit.
        #[arg(long)]
        instances: Option<PathBuf>,

        /// Preset directory (`vnncomp*/{category}.yaml`), forwarded to each run.
        #[arg(long)]
        configs_dir: Option<PathBuf>,

        /// Continue an existing sweep from its metadata file; rows that errored
        /// are retried
        #[arg(long, default_value_t = false)]
        resume: bool,

        /// Replace an existing sweep's files (without this or --resume, existing
        /// files are never overwritten)
        #[arg(long, default_value_t = false)]
        overwrite: bool,

        /// Emit a machine-readable summary instead of per-instance lines.
        #[arg(long, default_value_t = false)]
        json: bool,

        /// List what would run, without running anything.
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Set a tuning parameter for every instance: NAME=VALUE, repeatable
        ///
        /// Prefer this to exporting an environment variable: the value is
        /// validated before anything runs, recorded on every row, and sealed
        /// into the manifest, so a --resume with different values refuses.
        /// Unsafe parameters, and parameters that provably have no effect, are
        /// rejected.
        #[arg(long = "lever", value_name = "NAME=VALUE")]
        levers: Vec<String>,

        /// Verdict cache: off (default), read, or read-write
        ///
        /// Entries are keyed on the binary, configuration, category, model,
        /// property, budget, parameters, backend, host, and environment. Rows
        /// measured under heavy load are never cached. Rows served from the
        /// cache are marked `from_cache`.
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

    /// Summarize sweep results: solve counts, changes against a baseline, and
    /// modeled scores
    ///
    /// With --official, rows are scored +10 correct / -150 incorrect and
    /// normalized to each category's winner. A sat/unsat flip against the
    /// baseline is reported as a soundness alarm.
    Score {
        /// Results CSV (or a directory of them, e.g. `reports/measured`).
        /// Repeatable.
        #[arg(long = "results", required = true)]
        results: Vec<PathBuf>,

        /// Baseline to diff against — the previous bank. Omit to treat every
        /// solved row as new.
        #[arg(long)]
        baseline: Option<PathBuf>,

        /// Released VNN-COMP results to score against (requires --year); without
        /// it, only solve counts are reported
        #[arg(long, requires = "year")]
        official: Option<PathBuf>,

        /// VNN-COMP edition to score against (required with --official, because
        /// category membership changed between 2025 and 2026)
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

        /// How sat claims are scored: adjudicated (default), assumed-valid, or
        /// unvalidated
        ///
        /// For 2026, `adjudicated` uses published per-run counterexample checks
        /// and source-bound --ny-ce-checks. Tolerant acceptance credits its own
        /// SAT; only strict acceptance convicts UNSAT. Missing checks are
        /// unqualified. `assumed-valid` and `unvalidated` are raw projections.
        #[arg(long, value_enum, default_value = "adjudicated")]
        witnesses: super::vnncomp_score::Witnesses,

        /// Historical organizer truth summary: a released results corpus, or the
        /// `SCORING/results.txt` inside it
        ///
        /// Used by the 2025 model. Found automatically when --official points at an unsanitized
        /// corpus. A sanitized copy usually lacks `SCORING/`; pass the raw
        /// corpus here instead.
        #[arg(long)]
        ground_truth: Option<PathBuf>,

        /// Source-bound NY replay checks; raw SAT claims receive no 2026
        /// adjudicated credit without them
        #[arg(long, requires = "ny_ce_source")]
        ny_ce_checks: Option<PathBuf>,

        /// Exact NY source commit required by every supplied replay check
        #[arg(long, requires = "ny_ce_checks")]
        ny_ce_source: Option<String>,

        /// Check solved rows against this year's official per-instance timeouts
        /// (repeatable). For 2026 official fields, subtract each tool's startup
        /// minimum from its complete TOOL/results.csv; NY uses actual wall time
        #[arg(long = "budget-year")]
        budget_years: Vec<u32>,
    },

    /// Check that verdicts do not change when instance files are renamed
    ///
    /// A verdict that depends on a file name or path is overfitting or
    /// nondeterminism. Missing assets, child errors, timeouts, unknowns, or no
    /// valid pairs at all fail the command.
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

fn domain_json(domain: ny_levers::space::Domain) -> (&'static str, Vec<String>) {
    match domain {
        ny_levers::space::Domain::Bool => ("bool", vec!["0".into(), "1".into()]),
        ny_levers::space::Domain::Enum(values) => (
            "enum",
            values.iter().map(|value| (*value).to_string()).collect(),
        ),
        ny_levers::space::Domain::Grid(values) => {
            ("grid", values.iter().map(u64::to_string).collect())
        }
    }
}

fn class_name(class: ny_levers::space::Class) -> &'static str {
    match class {
        ny_levers::space::Class::Unsafe => "Unsafe",
        ny_levers::space::Class::VerdictAffecting => "VerdictAffecting",
        ny_levers::space::Class::SafeToSearch => "SafeToSearch",
    }
}

fn delivery_name(delivery: ny_levers::space::Deliver) -> String {
    match delivery {
        ny_levers::space::Deliver::PresetKey(key) => format!("preset:{key}"),
        ny_levers::space::Deliver::EnvOnly => "env-only".to_string(),
    }
}

fn absence_baseline_json(baseline: ny_levers::space::AbsenceBaseline) -> serde_json::Value {
    match baseline {
        ny_levers::space::AbsenceBaseline::DeclarationDefault => {
            json!({ "kind": "declaration_default" })
        }
        ny_levers::space::AbsenceBaseline::Distinct => json!({ "kind": "distinct" }),
        ny_levers::space::AbsenceBaseline::Equivalent(token) => {
            json!({ "kind": "equivalent", "token": token })
        }
        ny_levers::space::AbsenceBaseline::Contextual => json!({ "kind": "contextual" }),
    }
}

fn default_token(default: ny_levers::DefaultSpec) -> Option<String> {
    match default {
        ny_levers::DefaultSpec::Bool(value) => Some(if value { "1" } else { "0" }.into()),
        ny_levers::DefaultSpec::U64(value) => Some(value.to_string()),
        ny_levers::DefaultSpec::F64(value) | ny_levers::DefaultSpec::Secs(value) => {
            Some(value.to_string())
        }
        ny_levers::DefaultSpec::Text(value) => Some(value.to_string()),
        ny_levers::DefaultSpec::Unset => None,
    }
}

fn declaration_json(decl: &ny_levers::LeverDecl) -> serde_json::Value {
    let (kind, min, max, members) = match decl.kind {
        ny_levers::LeverKind::Bool => ("Bool", None, None, None),
        ny_levers::LeverKind::BoolOverride => ("BoolOverride", None, None, None),
        ny_levers::LeverKind::BoolKillOverride => ("BoolKillOverride", None, None, None),
        ny_levers::LeverKind::TruthyTrimmed => ("TruthyTrimmed", None, None, None),
        ny_levers::LeverKind::TrimmedExactOne => ("TrimmedExactOne", None, None, None),
        ny_levers::LeverKind::Presence => ("Presence", None, None, None),
        ny_levers::LeverKind::U64 => ("U64", None, None, None),
        ny_levers::LeverKind::U64Override => ("U64Override", None, None, None),
        ny_levers::LeverKind::U64Trimmed => ("U64Trimmed", None, None, None),
        ny_levers::LeverKind::UsizeTrimmed => ("UsizeTrimmed", None, None, None),
        ny_levers::LeverKind::UsizeTrimmedUnicodeOwned => {
            ("UsizeTrimmedUnicodeOwned", None, None, None)
        }
        ny_levers::LeverKind::UsizeTrimmedOverride { min } => (
            "UsizeTrimmedOverride",
            Some(min as f64),
            Some(usize::MAX as f64),
            None,
        ),
        ny_levers::LeverKind::UsizeOverride { min } => (
            "UsizeOverride",
            Some(min as f64),
            Some(usize::MAX as f64),
            None,
        ),
        ny_levers::LeverKind::Usize { min } => {
            ("Usize", Some(min as f64), Some(usize::MAX as f64), None)
        }
        ny_levers::LeverKind::F32Positive => (
            "F32Positive",
            Some(f64::from(f32::from_bits(1))),
            Some(f64::from(f32::MAX)),
            None,
        ),
        ny_levers::LeverKind::F32NonNegative => {
            ("F32NonNegative", Some(0.0), Some(f64::from(f32::MAX)), None)
        }
        ny_levers::LeverKind::F64Open { min, max } => ("F64Open", Some(min), Some(max), None),
        ny_levers::LeverKind::F64OpenUnicodeOwned { min, max } => {
            ("F64OpenUnicodeOwned", Some(min), Some(max), None)
        }
        ny_levers::LeverKind::F64OpenOverride { min, max } => {
            ("F64OpenOverride", Some(min), Some(max), None)
        }
        ny_levers::LeverKind::F64ClosedTrimmed { min, max } => {
            ("F64ClosedTrimmed", Some(min), Some(max), None)
        }
        ny_levers::LeverKind::Secs => ("Secs", None, None, None),
        ny_levers::LeverKind::Enum(members) => ("Enum", None, None, Some(members.to_vec())),
        ny_levers::LeverKind::LayeredEnum { env_members, .. } => {
            ("LayeredEnum", None, None, Some(env_members.to_vec()))
        }
        ny_levers::LeverKind::Text => ("Text", None, None, None),
    };
    json!({
        "kind": kind,
        "default": default_token(decl.default),
        "min": min,
        "max": max,
        "members": members,
    })
}

fn benchmark_space_payload() -> serde_json::Value {
    use std::collections::BTreeMap;

    let axes = ny_levers::space::axes()
        .iter()
        .map(|axis| {
            let (domain, values) = domain_json(axis.domain);
            json!({
                "name": axis.name,
                "domain": domain,
                "values": values,
                "class": class_name(axis.class),
                "deliver": delivery_name(axis.deliver),
                "absence_baseline": absence_baseline_json(
                    ny_levers::space::absence_baseline(axis.name)
                ),
                "preset_baseline_keys": ny_levers::space::preset_baseline_keys(axis.name),
                "category_prefixes": ny_levers::space::category_prefixes(axis.name),
                "why": axis.why,
            })
        })
        .collect::<Vec<_>>();
    let edges = ny_levers::space::edges()
        .iter()
        .map(|edge| {
            let (kind, parent, parents, bound, token) = match edge.requires {
                ny_levers::space::Requirement::Armed(parent) => {
                    ("armed", Some(parent), None, None, None)
                }
                ny_levers::space::Requirement::AnyArmed(parents) => {
                    ("any_armed", None, Some(parents), None, None)
                }
                ny_levers::space::Requirement::NonZero(parent) => {
                    ("non_zero", Some(parent), None, None, None)
                }
                ny_levers::space::Requirement::GreaterThan(parent, bound) => {
                    ("greater_than", Some(parent), None, Some(bound), None)
                }
                ny_levers::space::Requirement::NotArmed(parent) => {
                    ("not_armed", Some(parent), None, None, None)
                }
                ny_levers::space::Requirement::NotEquals(parent, token) => {
                    ("not_equals", Some(parent), None, None, Some(token))
                }
                ny_levers::space::Requirement::Absent(parent) => {
                    ("absent", Some(parent), None, None, None)
                }
                ny_levers::space::Requirement::Equals(parent, token) => {
                    ("equals", Some(parent), None, None, Some(token))
                }
            };
            json!({
                "child": edge.child,
                "kind": kind,
                "parent": parent,
                "parents": parents,
                "bound": bound,
                "token": token,
                "site": edge.site,
            })
        })
        .collect::<Vec<_>>();
    let declarations = ny_levers::all()
        .all()
        .iter()
        .map(|decl| (decl.name.to_string(), declaration_json(decl)))
        .collect::<BTreeMap<_, _>>();

    json!({
        "schema": "ny-levers/search-space/v6",
        "axes": axes,
        "edges": edges,
        "unsafe": ny_levers::space::unsafe_axes()
            .iter()
            .map(|axis| axis.name)
            .collect::<Vec<_>>(),
        "instrument_only": ny_levers::space::instrument_only(),
        "test_only": ny_levers::space::test_only(),
        "declarations": declarations,
    })
}

fn print_benchmark_space(json_output: bool) -> Result<()> {
    let payload = benchmark_space_payload();
    if json_output {
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    let axes = ny_levers::space::axes();
    let deliverable = axes
        .iter()
        .filter(|axis| matches!(axis.deliver, ny_levers::space::Deliver::PresetKey(_)))
        .count();
    println!("compiled searchable axes: {}", axes.len());
    println!("preset-deliverable: {deliverable}");
    println!("environment-only: {}", axes.len() - deliverable);
    println!("dependency edges: {}", ny_levers::space::edges().len());
    println!("use --json for the complete machine-readable contract");
    Ok(())
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
                     no NY_* variables, so whatever this measures CANNOT reach a \
                     scored run until a typed preset key exists for it."
                );
            }
        }
    }
    Ok(expanded)
}

pub(crate) fn handle_benchmark_assets_command(action: BenchmarkAssetsAction) -> Result<()> {
    match action {
        BenchmarkAssetsAction::Space { json } => print_benchmark_space(json),
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
            ground_truth,
            ny_ce_checks,
            ny_ce_source,
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
                ground_truth,
                ny_ce_checks,
                ny_ce_source,
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

    #[test]
    fn compiled_schema_serializes_scheduler_usize_ownership_exactly() {
        let declaration = |name: &'static str, kind: ny_levers::LeverKind| ny_levers::LeverDecl {
            name,
            kind,
            default: ny_levers::DefaultSpec::Unset,
            bucket: ny_levers::Bucket::Debug,
            moat: ny_levers::MoatRisk::None,
            doc: "unit-test fixture",
            provenance: ny_levers::Provenance::Unmeasured { why_ok: "test" },
            owner: ny_levers::Scope {
                package: "ny-cli",
                subsystem: "compiled-schema-test",
            },
            readers: &[],
        };

        let unicode_owned = declaration(
            "FIXTURE_VERIFIER_THREADS",
            ny_levers::LeverKind::UsizeTrimmedUnicodeOwned,
        );
        assert_eq!(
            declaration_json(&unicode_owned),
            json!({
                "kind": "UsizeTrimmedUnicodeOwned",
                "default": null,
                "min": null,
                "max": null,
                "members": null,
            })
        );

        let override_kind = declaration(
            "FIXTURE_CONCURRENT_THREADS",
            ny_levers::LeverKind::UsizeOverride { min: 0 },
        );
        assert_eq!(
            declaration_json(&override_kind),
            json!({
                "kind": "UsizeOverride",
                "default": null,
                "min": 0.0,
                "max": usize::MAX as f64,
                "members": null,
            })
        );
    }

    #[test]
    fn compiled_space_subcommand_and_json_contract_are_complete() {
        AssetsCli::command().debug_assert();
        let parsed = AssetsCli::try_parse_from(["assets", "space", "--json"])
            .expect("compiled space command parses");
        assert!(matches!(
            parsed.action,
            BenchmarkAssetsAction::Space { json: true }
        ));

        let payload = benchmark_space_payload();
        assert_eq!(payload["schema"], "ny-levers/search-space/v6");
        assert_eq!(
            payload["axes"].as_array().map(Vec::len),
            Some(ny_levers::space::axes().len())
        );
        assert_eq!(
            payload["edges"].as_array().map(Vec::len),
            Some(ny_levers::space::edges().len())
        );
        let declarations = payload["declarations"]
            .as_object()
            .expect("declaration map");
        assert_eq!(declarations.len(), ny_levers::all().len());
        let upfront_master = declarations
            .get("NY_UPFRONT_ATTACK")
            .expect("upfront master declaration");
        assert_eq!(upfront_master["kind"], "BoolOverride");
        assert!(upfront_master["default"].is_null());
        let upfront_fraction = declarations
            .get("NY_UPFRONT_ATTACK_FRAC")
            .expect("upfront fraction declaration");
        assert_eq!(upfront_fraction["kind"], "F64ClosedTrimmed");
        assert_eq!(
            upfront_fraction["min"].as_f64().map(f64::to_bits),
            Some(1),
            "compiled schema must preserve the strict-positive lower endpoint"
        );
        assert_eq!(upfront_fraction["max"], 1.0);
        for axis in ny_levers::space::axes() {
            let declaration = declarations
                .get(axis.name)
                .unwrap_or_else(|| panic!("searchable axis {} has no declaration", axis.name));
            assert!(declaration["kind"].is_string());
            let emitted = payload["axes"]
                .as_array()
                .expect("axis list")
                .iter()
                .find(|entry| entry["name"] == axis.name)
                .expect("axis is serialized");
            assert!(
                emitted["values"]
                    .as_array()
                    .is_some_and(|values| !values.is_empty()),
                "{} has no emitted values",
                axis.name
            );
            assert!(
                emitted["absence_baseline"]["kind"].is_string(),
                "{} has no effective absence contract",
                axis.name
            );
            assert_eq!(
                emitted["preset_baseline_keys"],
                json!(ny_levers::space::preset_baseline_keys(axis.name)),
                "{} has the wrong compiled preset baseline chain",
                axis.name
            );
            assert!(
                emitted["category_prefixes"].is_array(),
                "{} has no category applicability contract",
                axis.name
            );
        }
        let reserve = payload["axes"]
            .as_array()
            .expect("axis list")
            .iter()
            .find(|entry| entry["name"] == "NY_BAB_RESERVE_FRAC")
            .expect("reserve axis");
        assert_eq!(reserve["absence_baseline"]["kind"], "equivalent");
        assert_eq!(reserve["absence_baseline"]["token"], "0.0");

        let kfsb_candidates = payload["axes"]
            .as_array()
            .expect("axis list")
            .iter()
            .find(|entry| entry["name"] == "NY_MO_KFSB_K")
            .expect("multi-objective kFSB candidate axis");
        assert_eq!(
            kfsb_candidates["preset_baseline_keys"],
            json!([
                "bab.branching.kfsb_multi_candidates",
                "bab.branching.candidates",
            ])
        );
        let kfsb_reduce = payload["axes"]
            .as_array()
            .expect("axis list")
            .iter()
            .find(|entry| entry["name"] == "NY_MO_KFSB_REDUCE")
            .expect("multi-objective kFSB reduction axis");
        assert_eq!(
            kfsb_reduce["preset_baseline_keys"],
            json!([
                "bab.branching.kfsb_multi_reduceop",
                "bab.branching.reduceop",
            ])
        );
        let kfsb_master = payload["axes"]
            .as_array()
            .expect("axis list")
            .iter()
            .find(|entry| entry["name"] == "NY_MO_KFSB")
            .expect("multi-objective kFSB master axis");
        assert_eq!(
            kfsb_master["preset_baseline_keys"],
            json!(["bab.branching.kfsb_multi"])
        );

        let eta_mode = payload["edges"]
            .as_array()
            .expect("edge list")
            .iter()
            .find(|entry| entry["child"] == "NY_MARGIN_ROW_BETA_ETA" && entry["kind"] == "equals")
            .expect("exact beta-mode edge");
        assert_eq!(eta_mode["parent"], "NY_MARGIN_ROW_BETA_POLYAK");
        assert!(eta_mode["parents"].is_null());
        assert_eq!(eta_mode["token"], "0");
        assert!(eta_mode["bound"].is_null());

        let upfront_cap_gate = payload["edges"]
            .as_array()
            .expect("edge list")
            .iter()
            .find(|entry| {
                entry["child"] == "NY_UPFRONT_ATTACK_CAP" && entry["kind"] == "not_equals"
            })
            .expect("upfront kill-switch edge");
        assert_eq!(upfront_cap_gate["parent"], "NY_UPFRONT_ATTACK");
        assert_eq!(upfront_cap_gate["token"], "0");
        let upfront_auto_gate = payload["edges"]
            .as_array()
            .expect("edge list")
            .iter()
            .find(|entry| {
                entry["child"] == "NY_UPFRONT_ATTACK_AUTO_CAP" && entry["kind"] == "absent"
            })
            .expect("upfront contextual-auto edge");
        assert_eq!(upfront_auto_gate["parent"], "NY_UPFRONT_ATTACK");
        assert!(upfront_auto_gate["token"].is_null());

        let eta_arming = payload["edges"]
            .as_array()
            .expect("edge list")
            .iter()
            .find(|entry| {
                entry["child"] == "NY_MARGIN_ROW_BETA_ETA" && entry["kind"] == "any_armed"
            })
            .expect("OR-capable beta arming edge");
        assert!(eta_arming["parent"].is_null());
        assert_eq!(
            eta_arming["parents"],
            json!(["NY_MARGIN_ROW_BETA", "NY_MARGIN_ROW_CLOSURE_ADAPTIVE"])
        );
        assert!(eta_arming["bound"].is_null());
        assert!(eta_arming["token"].is_null());
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
