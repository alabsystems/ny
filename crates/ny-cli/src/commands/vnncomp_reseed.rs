// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny benchmarks reseed` — metamorphic overfitting check (`#no-instance-overfit`).
//!
//! # The problem this exists for
//!
//! VNN-COMP instances are GENERATED FROM A SEED and the organizers regenerate
//! them for the scored run (`relusplitter_2026` filenames still carry
//! `seed850851855`). A solver that keys any behaviour on instance identity —
//! a filename, a property index, a model basename — gains nothing on the scored
//! set and is overfitting in exactly the sense the organizers raise.
//!
//! ny shipped one such case until 2026-07-25: seven exact
//! `CIFAR100_resnet_medium_prop_idx_*.vnnlib` names compiled into the binary
//! that changed the reserve policy. It bought zero points and was removed. This
//! command exists so the next one cannot go unnoticed.
//!
//! # The check
//!
//! For each instance we build a RESEEDED twin: byte-identical model and property
//! content under fresh, content-derived names, in a fresh directory. Semantics
//! are unchanged, so:
//!
//! > **the verdict on the twin MUST equal the verdict on the original.**
//!
//! Any difference is one of two things, and both matter:
//! * **OVERFIT** — behaviour that depended on the instance's name or path.
//! * **NONDETERMINISM** — the same problem answered two ways, which undermines
//!   every measurement taken from a sweep.
//!
//! A `sat`/`unsat` disagreement between original and twin is additionally a
//! SOUNDNESS alarm: at most one can be right.
//!
//! # What this does not do
//!
//! It does not resample the underlying data point (that needs the benchmark
//! author's generator and dataset, which the corpus does not ship). Renaming is
//! the transformation whose correct answer is knowable with certainty, which is
//! what makes the check trustworthy rather than merely suggestive.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use super::bench_vnncomp::{instances_csv_for, VnnlibVersion};
use super::vnncomp_benchmarks::run_bounded_child;
use super::vnncomp_sweep::{
    enforce_scoring_deadline, parse_instances, SweepInstance, SweepVerdict,
};

const CHILD_STDERR_TAIL_BYTES: usize = 16 * 1024;
const CHILD_WATCHDOG_GRACE: Duration = Duration::from_secs(30);
const RESULT_FIRST_LINE_BYTES: u64 = 4096;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ReseedRow {
    pub(crate) benchmark: String,
    pub(crate) instance: String,
    pub(crate) original: SweepVerdict,
    pub(crate) reseeded: SweepVerdict,
    pub(crate) agrees: bool,
    /// Set when the pair disagrees AND both were decided — at most one is right.
    pub(crate) soundness_alarm: bool,
    /// Harness or asset failure detail. Never used to turn the row into a pass.
    pub(crate) detail: Option<String>,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct ReseedReport {
    pub(crate) rows: Vec<ReseedRow>,
    pub(crate) checked: usize,
    /// Pairs for which both executions produced a decided SAT/UNSAT verdict.
    pub(crate) valid: usize,
    /// Pairs containing at least one timeout/unknown verdict.
    pub(crate) inconclusive: usize,
    /// Pairs containing a child or harness error.
    pub(crate) errors: usize,
    /// Instances whose property or model assets were absent.
    pub(crate) missing: usize,
    pub(crate) disagreements: usize,
    pub(crate) soundness_alarms: usize,
}

pub(crate) struct ReseedOptions {
    pub(crate) year: u32,
    pub(crate) vnnlib_version: Option<VnnlibVersion>,
    pub(crate) categories: Vec<String>,
    pub(crate) limit: Option<usize>,
    pub(crate) timeout_cap: Option<u64>,
    pub(crate) configs_dir: Option<PathBuf>,
    pub(crate) json: bool,
}

/// A content-derived name that carries no benchmark, model, or property
/// identity — the point is that nothing about it can be recognised.
fn reseeded_stem(payload: &[u8], salt: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in payload.iter().chain(salt.as_bytes()) {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    format!("reseeded_{h:016x}")
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ModelField {
    Single(String),
    Relational(Vec<(String, String)>),
}

impl ModelField {
    fn paths(&self) -> Vec<&str> {
        match self {
            Self::Single(path) => vec![path],
            Self::Relational(entries) => entries.iter().map(|(_, path)| path.as_str()).collect(),
        }
    }

    fn rewrite_paths(&self, replacements: &[String]) -> Result<String> {
        if replacements.len() != self.paths().len() {
            bail!(
                "model rewrite supplied {} path(s) for {} model(s)",
                replacements.len(),
                self.paths().len()
            );
        }
        match self {
            Self::Single(_) => Ok(replacements[0].clone()),
            Self::Relational(entries) => {
                let rendered = entries
                    .iter()
                    .zip(replacements)
                    .map(|((label, _), path)| {
                        format!(
                            "('{}', '{}')",
                            python_single_quote(label),
                            python_single_quote(path)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                Ok(format!("[{rendered}]"))
            }
        }
    }

    fn absolute_argument(&self, base: &Path) -> Result<PathBuf> {
        let absolute_paths = self
            .paths()
            .into_iter()
            .map(|path| {
                let absolute = std::fs::canonicalize(base.join(path)).with_context(|| {
                    format!("resolve original model {}", base.join(path).display())
                })?;
                absolute
                    .to_str()
                    .map(str::to_string)
                    .context("original model path is not valid UTF-8")
            })
            .collect::<Result<Vec<_>>>()?;
        match self {
            Self::Single(_) => Ok(PathBuf::from(&absolute_paths[0])),
            Self::Relational(_) => Ok(PathBuf::from(self.rewrite_paths(&absolute_paths)?)),
        }
    }
}

fn python_single_quote(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\'', "\\'")
}

struct LiteralParser<'a> {
    input: &'a str,
    cursor: usize,
}

impl<'a> LiteralParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, cursor: 0 }
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.input[self.cursor..].chars().next() {
            if !ch.is_whitespace() {
                break;
            }
            self.cursor += ch.len_utf8();
        }
    }

    fn consume(&mut self, expected: char) -> bool {
        self.skip_ws();
        if self.input[self.cursor..].starts_with(expected) {
            self.cursor += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn quoted(&mut self) -> Result<String> {
        self.skip_ws();
        let quote = self.input[self.cursor..]
            .chars()
            .next()
            .filter(|ch| *ch == '\'' || *ch == '"')
            .context("expected quoted string in relational ONNX field")?;
        self.cursor += quote.len_utf8();
        let mut value = String::new();
        loop {
            let ch = self.input[self.cursor..]
                .chars()
                .next()
                .context("unterminated quoted string in relational ONNX field")?;
            self.cursor += ch.len_utf8();
            if ch == quote {
                return Ok(value);
            }
            if ch == '\\' {
                let escaped = self.input[self.cursor..]
                    .chars()
                    .next()
                    .context("trailing escape in relational ONNX field")?;
                self.cursor += escaped.len_utf8();
                value.push(escaped);
            } else {
                value.push(ch);
            }
        }
    }

    fn finished(&mut self) -> bool {
        self.skip_ws();
        self.cursor == self.input.len()
    }
}

fn onnx_suffix(path: &Path) -> Option<&'static str> {
    let extension_is = |candidate: &Path, expected: &str| {
        candidate
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case(expected))
    };
    if extension_is(path, "onnx") {
        Some(".onnx")
    } else if extension_is(path, "gz")
        && path
            .file_stem()
            .map(Path::new)
            .is_some_and(|stem| extension_is(stem, "onnx"))
    {
        Some(".onnx.gz")
    } else {
        None
    }
}

fn validate_onnx_path(path: &str) -> Result<()> {
    if path.is_empty() || onnx_suffix(Path::new(path)).is_none() {
        bail!("model path must end in .onnx or .onnx.gz, got {path:?}");
    }
    Ok(())
}

fn parse_model_field(field: &str) -> Result<ModelField> {
    let trimmed = field.trim();
    if trimmed.is_empty() {
        bail!("empty ONNX field");
    }
    if !trimmed.starts_with('[') {
        validate_onnx_path(trimmed)?;
        return Ok(ModelField::Single(trimmed.to_string()));
    }

    let mut parser = LiteralParser::new(trimmed);
    if !parser.consume('[') {
        bail!("relational ONNX field must start with '['");
    }
    let mut entries = Vec::new();
    if parser.consume(']') {
        bail!("relational ONNX field cannot be empty");
    }
    loop {
        if !parser.consume('(') {
            bail!("expected '(' in relational ONNX field");
        }
        let label = parser.quoted()?;
        if label.is_empty() {
            bail!("relational network label cannot be empty");
        }
        if !parser.consume(',') {
            bail!("expected ',' between relational network label and path");
        }
        let path = parser.quoted()?;
        validate_onnx_path(&path)?;
        if !parser.consume(')') {
            bail!("expected ')' after relational network entry");
        }
        entries.push((label, path));
        if parser.consume(']') {
            break;
        }
        if !parser.consume(',') {
            bail!("expected ',' between relational network entries");
        }
    }
    if !parser.finished() {
        bail!("trailing data after relational ONNX field");
    }
    let mut labels = std::collections::HashSet::new();
    if entries
        .iter()
        .any(|(label, _)| !labels.insert(label.clone()))
    {
        bail!("relational ONNX field contains a duplicate network label");
    }
    Ok(ModelField::Relational(entries))
}

/// Parse a model field into label/path pairs for path-stable result identities.
///
/// A single-model field has a `None` label; relational fields retain every
/// network label and path in order.
pub(crate) fn model_identity_entries(field: &str) -> Result<Vec<(Option<String>, String)>> {
    Ok(match parse_model_field(field)? {
        ModelField::Single(path) => vec![(None, path)],
        ModelField::Relational(entries) => entries
            .into_iter()
            .map(|(label, path)| (Some(label), path))
            .collect(),
    })
}

pub(crate) fn absolute_model_argument(base: &Path, field: &str) -> Result<PathBuf> {
    parse_model_field(field)?.absolute_argument(base)
}

fn model_suffix(path: &Path) -> Result<&'static str> {
    onnx_suffix(path).with_context(|| {
        format!(
            "model file must end in .onnx or .onnx.gz: {}",
            path.display()
        )
    })
}

fn link_or_copy(source: &Path, destination: &Path) -> Result<()> {
    match std::fs::hard_link(source, destination) {
        Ok(()) => Ok(()),
        Err(_) => {
            std::fs::copy(source, destination).with_context(|| {
                format!(
                    "copy model {} to {}",
                    source.display(),
                    destination.display()
                )
            })?;
            Ok(())
        }
    }
}

fn read_result_first_line(path: &Path) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    let mut line = String::new();
    BufReader::new(file)
        .take(RESULT_FIRST_LINE_BYTES)
        .read_line(&mut line)
        .ok()?;
    Some(line)
}

#[derive(Debug)]
struct ReseedRun {
    verdict: SweepVerdict,
    detail: Option<String>,
}

fn run_one(
    exe: &Path,
    category: &str,
    onnx: &Path,
    vnnlib: &Path,
    budget: u64,
    configs_dir: Option<&Path>,
    scratch: &Path,
    tag: &str,
) -> ReseedRun {
    let rf = scratch.join(format!("{tag}-result.txt"));
    let mut cmd = Command::new(exe);
    cmd.arg("vnncomp")
        .arg("v1")
        .arg(category)
        .arg(onnx)
        .arg(vnnlib)
        .arg(&rf)
        .arg(budget.to_string());
    if let Some(d) = configs_dir {
        cmd.arg("--configs-dir").arg(d);
    }
    cmd.env("RUST_LOG", "error");
    let watchdog = Duration::from_secs(budget).saturating_add(CHILD_WATCHDOG_GRACE);
    match run_bounded_child(&mut cmd, watchdog, CHILD_STDERR_TAIL_BYTES) {
        Ok(output) if output.timed_out => ReseedRun {
            verdict: SweepVerdict::Timeout,
            detail: Some(format!(
                "outer watchdog stopped child after {:.1}s",
                watchdog.as_secs_f64()
            )),
        },
        Ok(output) => {
            let initial_verdict =
                SweepVerdict::classify(read_result_first_line(&rf).as_deref(), output.success);
            let (verdict, deadline_detail) =
                enforce_scoring_deadline(initial_verdict, output.elapsed, budget);
            let detail = deadline_detail.or_else(|| {
                (verdict == SweepVerdict::Error).then(|| {
                    let mut lines: Vec<&str> = output.stderr_tail.lines().rev().take(3).collect();
                    lines.reverse();
                    lines.join(" | ").chars().take(240).collect()
                })
            });
            ReseedRun { verdict, detail }
        }
        Err(error) => ReseedRun {
            verdict: SweepVerdict::Error,
            detail: Some(error.to_string()),
        },
    }
}

fn is_decided(verdict: SweepVerdict) -> bool {
    matches!(verdict, SweepVerdict::Sat | SweepVerdict::Unsat)
}

fn is_inconclusive(verdict: SweepVerdict) -> bool {
    matches!(verdict, SweepVerdict::Timeout | SweepVerdict::Unknown)
}

impl ReseedReport {
    fn record_run(
        &mut self,
        benchmark: &str,
        instance: &str,
        original: ReseedRun,
        reseeded: ReseedRun,
    ) {
        self.checked += 1;
        let agrees = original.verdict == reseeded.verdict;
        let soundness_alarm =
            !agrees && is_decided(original.verdict) && is_decided(reseeded.verdict);
        if original.verdict == SweepVerdict::Error || reseeded.verdict == SweepVerdict::Error {
            self.errors += 1;
        } else if is_inconclusive(original.verdict) || is_inconclusive(reseeded.verdict) {
            self.inconclusive += 1;
        } else {
            self.valid += 1;
        }
        if !agrees {
            self.disagreements += 1;
        }
        if soundness_alarm {
            self.soundness_alarms += 1;
        }
        let detail = match (original.detail, reseeded.detail) {
            (Some(original), Some(reseeded)) => {
                Some(format!("original: {original}; reseeded: {reseeded}"))
            }
            (Some(detail), None) => Some(format!("original: {detail}")),
            (None, Some(detail)) => Some(format!("reseeded: {detail}")),
            (None, None) => None,
        };
        self.rows.push(ReseedRow {
            benchmark: benchmark.to_string(),
            instance: instance.to_string(),
            original: original.verdict,
            reseeded: reseeded.verdict,
            agrees,
            soundness_alarm,
            detail,
        });
    }

    fn record_preparation_failure(
        &mut self,
        benchmark: &str,
        instance: &str,
        detail: String,
        missing: bool,
    ) {
        if missing {
            self.missing += 1;
        } else {
            self.errors += 1;
        }
        self.rows.push(ReseedRow {
            benchmark: benchmark.to_string(),
            instance: instance.to_string(),
            original: SweepVerdict::Error,
            reseeded: SweepVerdict::Error,
            agrees: false,
            soundness_alarm: false,
            detail: Some(detail),
        });
    }

    fn passes(&self) -> bool {
        self.valid > 0
            && self.inconclusive == 0
            && self.errors == 0
            && self.missing == 0
            && self.disagreements == 0
    }

    fn failure_summary(&self) -> String {
        format!(
            "valid={}, inconclusive={}, errors={}, missing={}, disagreements={}",
            self.valid, self.inconclusive, self.errors, self.missing, self.disagreements
        )
    }
}

struct PreparedTwin {
    directory: tempfile::TempDir,
    original_onnx: PathBuf,
    twin_onnx: PathBuf,
    twin_vnnlib: PathBuf,
}

fn prepare_twin(
    work: &Path,
    base: &Path,
    model_field: &ModelField,
    vnnlib: &Path,
) -> Result<PreparedTwin> {
    let directory = tempfile::Builder::new()
        .prefix("instance-")
        .tempdir_in(work)
        .context("create private reseed instance directory")?;
    let twin_onnx_dir = directory.path().join("onnx");
    let twin_vnnlib_dir = directory.path().join("vnnlib");
    std::fs::create_dir_all(&twin_onnx_dir)?;
    std::fs::create_dir_all(&twin_vnnlib_dir)?;

    let vnnlib_bytes =
        std::fs::read(vnnlib).with_context(|| format!("read property {}", vnnlib.display()))?;
    // The production name depends only on content plus a constant domain
    // separator, never on the original instance identity.
    let stem = reseeded_stem(&vnnlib_bytes, "vnnlib-property");
    let property_suffix = if vnnlib
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".vnnlib.gz"))
    {
        ".vnnlib.gz"
    } else {
        ".vnnlib"
    };
    let twin_vnnlib = twin_vnnlib_dir.join(format!("{stem}{property_suffix}"));
    std::fs::write(&twin_vnnlib, &vnnlib_bytes)
        .with_context(|| format!("write reseeded property {}", twin_vnnlib.display()))?;

    let mut replacement_paths = Vec::new();
    for (index, relative) in model_field.paths().into_iter().enumerate() {
        let source = base.join(relative);
        let suffix = model_suffix(&source)?;
        let destination = twin_onnx_dir.join(format!("{stem}_m{index}{suffix}"));
        // Link/copy the resolved file rather than a symlink inode. A relative
        // symlink recreated in this private directory could point somewhere
        // else (or nowhere), changing the model instead of merely renaming it.
        let resolved_source = std::fs::canonicalize(&source)
            .with_context(|| format!("resolve model {}", source.display()))?;
        link_or_copy(&resolved_source, &destination)?;
        let absolute = std::fs::canonicalize(&destination)
            .with_context(|| format!("resolve reseeded model {}", destination.display()))?;
        replacement_paths.push(
            absolute
                .to_str()
                .map(str::to_string)
                .context("reseeded model path is not valid UTF-8")?,
        );
    }

    let rewritten = model_field.rewrite_paths(&replacement_paths)?;
    let twin_onnx = PathBuf::from(rewritten);
    Ok(PreparedTwin {
        original_onnx: model_field.absolute_argument(base)?,
        twin_onnx,
        twin_vnnlib,
        directory,
    })
}

pub(crate) fn run_reseed(opts: &ReseedOptions) -> Result<ReseedReport> {
    if opts.timeout_cap == Some(0) {
        bail!("--timeout-cap must be greater than zero");
    }
    if opts.limit == Some(0) {
        bail!("--limit must be greater than zero");
    }

    let all = super::bench_vnncomp::discover_categories(opts.year)?;
    if all.is_empty() {
        bail!(
            "no VNN-COMP {} benchmark categories were discovered",
            opts.year
        );
    }
    let chosen: Vec<_> = if opts.categories.is_empty() {
        all
    } else {
        let mut v = Vec::new();
        let mut selected = std::collections::HashSet::new();
        for want in &opts.categories {
            match all.iter().find(|(n, _)| n == want) {
                Some(hit) if selected.insert(hit.0.clone()) => v.push(hit.clone()),
                Some(_) => {}
                None => bail!("unknown category {want:?} for VNN-COMP {}", opts.year),
            }
        }
        v
    };

    let exe = std::env::current_exe().context("locate the running ny executable")?;
    let work = tempfile::Builder::new()
        .prefix("ny-reseed-")
        .tempdir()
        .context("create private reseed workspace")?;
    let mut report = ReseedReport::default();

    for (category, dir) in chosen {
        let Some(csv) = instances_csv_for(&dir, opts.vnnlib_version)? else {
            bail!(
                "category {category:?} has no instances.csv under {}",
                dir.display()
            );
        };
        let body =
            std::fs::read_to_string(&csv).with_context(|| format!("read {}", csv.display()))?;
        let instances: Vec<SweepInstance> =
            parse_instances(&body).with_context(|| format!("parse {}", csv.display()))?;
        if instances.is_empty() {
            bail!("category {category:?} has an empty {}", csv.display());
        }
        let base = csv.parent().unwrap_or(&dir).to_path_buf();

        for inst in instances.iter().take(opts.limit.unwrap_or(usize::MAX)) {
            let instance = inst.vnnlib_field.as_str();
            let vnnlib = base.join(&inst.vnnlib_field);
            if !vnnlib.is_file() {
                report.record_preparation_failure(
                    &category,
                    instance,
                    format!("missing property {}", vnnlib.display()),
                    true,
                );
                continue;
            }
            let model_field = match parse_model_field(&inst.onnx_field) {
                Ok(field) => field,
                Err(error) => {
                    report.record_preparation_failure(
                        &category,
                        instance,
                        format!("invalid ONNX field {:?}: {error}", inst.onnx_field),
                        false,
                    );
                    continue;
                }
            };
            let missing_models: Vec<PathBuf> = model_field
                .paths()
                .into_iter()
                .map(|path| base.join(path))
                .filter(|path| !path.is_file())
                .collect();
            if !missing_models.is_empty() {
                report.record_preparation_failure(
                    &category,
                    instance,
                    format!(
                        "missing model asset(s): {}",
                        missing_models
                            .iter()
                            .map(|path| path.display().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    true,
                );
                continue;
            }

            let prepared = match prepare_twin(work.path(), &base, &model_field, &vnnlib) {
                Ok(prepared) => prepared,
                Err(error) => {
                    report.record_preparation_failure(
                        &category,
                        instance,
                        format!("prepare reseeded twin: {error:#}"),
                        false,
                    );
                    continue;
                }
            };
            let budget = opts
                .timeout_cap
                .map_or(inst.budget_secs, |cap| cap.min(inst.budget_secs));
            let original = run_one(
                &exe,
                &category,
                &prepared.original_onnx,
                &vnnlib,
                budget,
                opts.configs_dir.as_deref(),
                prepared.directory.path(),
                "original",
            );
            let reseeded = run_one(
                &exe,
                &category,
                &prepared.twin_onnx,
                &prepared.twin_vnnlib,
                budget,
                opts.configs_dir.as_deref(),
                prepared.directory.path(),
                "reseeded",
            );
            report.record_run(&category, instance, original, reseeded);
            let row = report
                .rows
                .last()
                .context("reseed report lost the row it just recorded")?;
            if !opts.json {
                let mark =
                    if row.original == SweepVerdict::Error || row.reseeded == SweepVerdict::Error {
                        "ERROR"
                    } else if row.agrees && is_decided(row.original) && is_decided(row.reseeded) {
                        "valid"
                    } else if row.agrees {
                        "INCONCLUSIVE"
                    } else {
                        "MISMATCH"
                    };
                println!(
                    "  {category}: {} | original={} reseeded={} [{mark}]",
                    inst.vnnlib_field,
                    row.original.as_str(),
                    row.reseeded.as_str()
                );
                if let Some(detail) = row.detail.as_deref() {
                    println!("    {detail}");
                }
            }
        }
    }

    if opts.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!();
        println!(
            "reseed: {} pair(s) run | valid {} | inconclusive {} | errors {} | missing {} | \
             disagreements {} | soundness alarms {}",
            report.checked,
            report.valid,
            report.inconclusive,
            report.errors,
            report.missing,
            report.disagreements,
            report.soundness_alarms
        );
        if report.passes() {
            println!(
                "PASS — all {} decided verdict pair(s) survived reseeding; no run was \
                 inconclusive, erroneous, missing, or identity-dependent.",
                report.valid
            );
        } else {
            println!(
                "FAIL — reseeding was not a complete clean check ({})",
                report.failure_summary()
            );
            for row in report
                .rows
                .iter()
                .filter(|row| !row.agrees || row.detail.is_some())
            {
                println!(
                    "  {}: {} original={} reseeded={}{}",
                    row.benchmark,
                    row.instance,
                    row.original.as_str(),
                    row.reseeded.as_str(),
                    if row.soundness_alarm {
                        "   *** SOUNDNESS: sat/unsat disagreement, at most one is right ***"
                    } else {
                        ""
                    }
                );
                if let Some(detail) = row.detail.as_deref() {
                    println!("    {detail}");
                }
            }
        }
    }

    if !report.passes() {
        bail!("reseed check failed: {}", report.failure_summary());
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reseeded_names_carry_no_recognisable_identity() {
        let name = reseeded_stem(b"(assert (<= X_0 1.0))", "vnnlib-property");
        assert!(name.starts_with("reseeded_"));
        assert!(!name.contains("CIFAR100"));
        assert!(!name.contains("prop_idx"));
        assert!(!name.contains("815"));
    }

    #[test]
    fn reseeded_names_are_deterministic_and_content_sensitive() {
        let a = reseeded_stem(b"body-a", "s");
        let b = reseeded_stem(b"body-a", "s");
        let c = reseeded_stem(b"body-b", "s");
        assert_eq!(a, b, "same content must reseed to the same twin");
        assert_ne!(a, c, "different content must reseed differently");
    }

    #[test]
    fn model_field_parser_preserves_gzip_suffixes() {
        assert_eq!(
            parse_model_field("models/a.onnx.gz").expect("single gzip"),
            ModelField::Single("models/a.onnx.gz".into())
        );
        let relational = parse_model_field("[('f', 'onnx/a.onnx.gz'), ('g', \"onnx/b.onnx\")]")
            .expect("relational gzip");
        assert_eq!(relational.paths(), vec!["onnx/a.onnx.gz", "onnx/b.onnx"]);
    }

    #[test]
    fn relational_rewrite_is_structural_not_substring_replacement() {
        let field = parse_model_field("[('f', 'models/a.onnx'), ('g', 'models/a.onnx.gz')]")
            .expect("relational field");
        let rewritten = field
            .rewrite_paths(&["onnx/fresh0.onnx".into(), "onnx/fresh1.onnx.gz".into()])
            .expect("rewrite");
        assert_eq!(
            rewritten,
            "[('f', 'onnx/fresh0.onnx'), ('g', 'onnx/fresh1.onnx.gz')]"
        );
        assert!(!rewritten.contains("models/a.onnx"));
    }

    #[test]
    fn relational_parser_rejects_malformed_or_ambiguous_fields() {
        assert!(parse_model_field("[]").is_err());
        assert!(parse_model_field("[('f', 'a.onnx')] trailing").is_err());
        assert!(
            parse_model_field("[('f', 'a.onnx'), ('f', 'b.onnx')]").is_err(),
            "duplicate labels make a structural rewrite ambiguous"
        );
        assert!(parse_model_field("[('f', 'not-a-model.bin')]").is_err());
    }

    #[test]
    fn prepared_gzip_twin_uses_raii_cleanup_and_preserves_bytes() {
        let corpus = tempfile::tempdir().expect("corpus");
        let onnx_dir = corpus.path().join("onnx");
        let vnnlib_dir = corpus.path().join("vnnlib");
        std::fs::create_dir_all(&onnx_dir).expect("onnx dir");
        std::fs::create_dir_all(&vnnlib_dir).expect("vnnlib dir");
        let model = onnx_dir.join("model.onnx.gz");
        let property = vnnlib_dir.join("property.vnnlib");
        std::fs::write(&model, b"gzip-model-bytes").expect("model");
        std::fs::write(&property, b"(assert true)").expect("property");
        let field = parse_model_field("onnx/model.onnx.gz").expect("field");

        let prepared =
            prepare_twin(corpus.path(), corpus.path(), &field, &property).expect("prepare");
        let private_dir = prepared.directory.path().to_path_buf();
        assert!(prepared.twin_onnx.to_string_lossy().ends_with(".onnx.gz"));
        assert_eq!(
            std::fs::read(&prepared.twin_onnx).expect("twin bytes"),
            b"gzip-model-bytes"
        );
        drop(prepared);
        assert!(
            !private_dir.exists(),
            "TempDir must clean copied/hardlinked assets"
        );
    }

    #[cfg(unix)]
    #[test]
    fn prepared_twin_copies_a_symlink_target_not_its_relative_link() {
        let corpus = tempfile::tempdir().expect("corpus");
        let onnx_dir = corpus.path().join("onnx");
        let vnnlib_dir = corpus.path().join("vnnlib");
        std::fs::create_dir_all(&onnx_dir).expect("onnx dir");
        std::fs::create_dir_all(&vnnlib_dir).expect("vnnlib dir");
        std::fs::write(onnx_dir.join("actual.onnx"), b"model-bytes").expect("model");
        std::os::unix::fs::symlink("actual.onnx", onnx_dir.join("model.onnx")).expect("symlink");
        let property = vnnlib_dir.join("property.vnnlib");
        std::fs::write(&property, b"(assert true)").expect("property");
        let field = parse_model_field("onnx/model.onnx").expect("field");

        let prepared =
            prepare_twin(corpus.path(), corpus.path(), &field, &property).expect("prepare");
        assert_eq!(
            std::fs::read(&prepared.twin_onnx).expect("twin bytes"),
            b"model-bytes"
        );
        assert!(
            !prepared.twin_onnx.is_symlink(),
            "a relative source symlink must not be recreated in the private twin directory"
        );
    }

    #[test]
    fn relational_original_and_twin_arguments_use_absolute_member_paths() {
        let corpus = tempfile::tempdir().expect("corpus");
        let onnx_dir = corpus.path().join("onnx");
        let vnnlib_dir = corpus.path().join("vnnlib");
        std::fs::create_dir_all(&onnx_dir).expect("onnx dir");
        std::fs::create_dir_all(&vnnlib_dir).expect("vnnlib dir");
        std::fs::write(onnx_dir.join("f.onnx"), b"f").expect("f");
        std::fs::write(onnx_dir.join("g.onnx.gz"), b"g").expect("g");
        let property = vnnlib_dir.join("property.vnnlib");
        std::fs::write(&property, b"(assert true)").expect("property");
        let field =
            parse_model_field("[('f', 'onnx/f.onnx'), ('g', 'onnx/g.onnx.gz')]").expect("field");

        let prepared =
            prepare_twin(corpus.path(), corpus.path(), &field, &property).expect("prepare");
        for argument in [&prepared.original_onnx, &prepared.twin_onnx] {
            let parsed = parse_model_field(argument.to_str().expect("UTF-8 argument"))
                .expect("rewritten relational field");
            assert!(
                parsed
                    .paths()
                    .iter()
                    .all(|path| Path::new(path).is_absolute()),
                "relational members must not depend on the child process cwd: {argument:?}"
            );
        }
    }

    fn run(verdict: SweepVerdict) -> ReseedRun {
        ReseedRun {
            verdict,
            detail: None,
        }
    }

    #[test]
    fn report_requires_at_least_one_valid_decided_pair() {
        let empty = ReseedReport::default();
        assert!(!empty.passes());

        let mut timeout = ReseedReport::default();
        timeout.record_run(
            "c",
            "p",
            run(SweepVerdict::Timeout),
            run(SweepVerdict::Timeout),
        );
        assert_eq!(timeout.inconclusive, 1);
        assert_eq!(timeout.valid, 0);
        assert!(
            !timeout.passes(),
            "matching timeouts are not a passing check"
        );
    }

    #[test]
    fn report_fails_errors_missing_assets_and_disagreements() {
        let mut report = ReseedReport::default();
        report.record_run(
            "c",
            "valid",
            run(SweepVerdict::Unsat),
            run(SweepVerdict::Unsat),
        );
        report.record_run(
            "c",
            "error",
            run(SweepVerdict::Error),
            run(SweepVerdict::Error),
        );
        report.record_preparation_failure("c", "missing", "missing model".into(), true);
        report.record_run(
            "c",
            "mismatch",
            run(SweepVerdict::Sat),
            run(SweepVerdict::Unsat),
        );
        assert_eq!(report.valid, 2);
        assert_eq!(report.errors, 1);
        assert_eq!(report.missing, 1);
        assert_eq!(report.disagreements, 1);
        assert_eq!(report.soundness_alarms, 1);
        assert!(!report.passes());
    }

    #[test]
    fn report_passes_only_clean_decided_agreement() {
        let mut report = ReseedReport::default();
        report.record_run("c", "p", run(SweepVerdict::Sat), run(SweepVerdict::Sat));
        assert_eq!(report.valid, 1);
        assert!(report.passes());
    }

    #[test]
    fn missing_result_file_is_not_a_decided_verdict() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("missing-result.txt");
        assert!(read_result_first_line(&path).is_none());
        assert_eq!(
            SweepVerdict::classify(read_result_first_line(&path).as_deref(), true),
            SweepVerdict::Error
        );
    }
}
