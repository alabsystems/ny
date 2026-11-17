// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny benchmarks score` — compare sweep results and model VNN-COMP scoring.
//!
//! The baseline passed with `--baseline` is a historical **reference**, not a
//! source of truth. It is used only to report new solves, regressions, and
//! SAT/UNSAT flips. When `--official` is supplied, the current run is modeled in
//! the selected competition year's regular or extended portion of that field
//! using the VNN-COMP raw-point rules:
//!
//! * a correct `sat`/`unsat` result earns +10 points;
//! * an `unsat` contradicted by a valid SAT witness earns -150 points;
//! * timeout, unknown, and error rows earn zero;
//! * each benchmark is normalized winner-relative as
//!   `max(0, 100 * current_raw / best_field_raw)`.
//!
//! For 2026, adjudicated scoring reads the published per-run counterexample
//! checks. Tolerant acceptance credits its own SAT without convicting UNSAT;
//! only a strict witness does both. Unevidenced SAT claims receive no credit.
//! Source/build eligibility and cross-version equivalence remain separate gates.
//! With `--year 2026 --budget-year 2026`, field verdicts also apply the pinned
//! organizer's global per-tool startup subtraction before the timeout check.
//! NY's measured wall time keeps its separate strict original-budget check.
//!
//! # Budget enforcement
//!
//! `--budget-year` loads per-instance timeouts from the corpus's `instances.csv`
//! files. Budgets vary within categories: nn4sys alone spans 20 s to 800 s.
//! NY uses recorded wall time; normalized 2026 field rows use startup-adjusted
//! time. Solves above the applicable budget receive zero modeled credit.
//!
//! Coverage is reported alongside the violations. A row whose identity has no
//! budget in the supplied corpora is counted as unchecked, never as passing;
//! "0 over budget" out of 0 checked is not a clean bill of health.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::bench_vnncomp::{discover_categories, parse_csv_line};
use super::vnncomp_2025_tracks::{EXTENDED_TRACK_2025, REGULAR_TRACK_2025};
use super::vnncomp_2026_tracks::{EXTENDED_TRACK_2026, REGULAR_TRACK_2026};
use super::vnncomp_reseed::model_identity_entries;
use super::vnncomp_sweep::parse_instances;

#[path = "vnncomp_score_field_timeouts.rs"]
mod field_timeouts;

const POINTS_CORRECT: i64 = 10;
const PENALTY_INCORRECT: i64 = -150;
const MODELED_SCORE_CAVEAT: &str = "raw CSV: SAT witnesses are assumed valid and are not \
replayed; normalized identity matching does not reproduce organizer cross-version positional \
checks";
const MODELED_2026_SCORE_CAVEAT: &str = "local projection: published per-run counterexample \
checks are used only in adjudicated mode; unevidenced SAT claims are unqualified. Source/build \
eligibility, corpus pins, organizer field timeout normalization and cross-version positional \
checks are not established";
const NORMALIZED_2026_SCORE_CAVEAT: &str = "local projection: supplied aggregate field rows \
apply the pinned 2026 per-tool startup/timeout policy; published per-run counterexample checks \
are used only in adjudicated mode. Source/build eligibility, complete corpus provenance and \
cross-version positional equivalence are not established";
const UNCHECKED_BUDGET_CAVEAT: &str =
    "recorded runtimes are NOT checked against official per-instance budgets; pass \
--budget-year to enforce them";

/// Competition edition whose released category split is being modeled.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) enum ScoreYear {
    #[value(name = "2025")]
    #[serde(rename = "2025")]
    Y2025,
    #[value(name = "2026")]
    #[serde(rename = "2026")]
    Y2026,
}

impl ScoreYear {
    fn number(self) -> u32 {
        match self {
            Self::Y2025 => 2025,
            Self::Y2026 => 2026,
        }
    }

    fn from_category_prefix(category: &str) -> (Option<Self>, &str) {
        if let Some(category) = category.strip_prefix("2025_") {
            (Some(Self::Y2025), category)
        } else if let Some(category) = category.strip_prefix("2026_") {
            (Some(Self::Y2026), category)
        } else {
            (None, category)
        }
    }
}

/// How SAT rows in a raw results CSV are treated when scoring.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Witnesses {
    /// For 2026, use each published per-run counterexample check and explicit
    /// source-bound NY replay checks. Only strict acceptance convicts UNSAT;
    /// tolerant acceptance credits its own SAT. Missing evidence is neutral.
    /// The historical 2025 model retains published-truth/corroboration rules.
    Adjudicated,
    /// Projection: assume every raw SAT has a strictly valid witness.
    AssumedValid,
    /// Projection: credit every decided claim without checking witnesses.
    Unvalidated,
}

impl Witnesses {
    fn label(self) -> &'static str {
        match self {
            Self::Adjudicated => {
                "2026 per-run checks (strict and tolerant acceptance differ); \
                 historical 2025 truth/corroboration model"
            }
            Self::AssumedValid => "SAT witnesses assumed valid (contradicted UNSAT scores -150)",
            Self::Unvalidated => "counterexamples not checked; every decided claim is credited",
        }
    }
}

/// The separate regular or extended leaderboard whose total is being modeled.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ScoreTrack {
    Regular,
    Extended,
}

impl ScoreTrack {
    fn includes(self, year: ScoreYear, benchmark: &str) -> bool {
        let (prefix_year, benchmark) = ScoreYear::from_category_prefix(benchmark);
        if prefix_year.is_some_and(|prefix_year| prefix_year != year) {
            return false;
        }
        match (year, self) {
            (ScoreYear::Y2025, Self::Regular) => REGULAR_TRACK_2025.contains(&benchmark),
            (ScoreYear::Y2025, Self::Extended) => EXTENDED_TRACK_2025.contains(&benchmark),
            (ScoreYear::Y2026, Self::Regular) => REGULAR_TRACK_2026.contains(&benchmark),
            (ScoreYear::Y2026, Self::Extended) => EXTENDED_TRACK_2026.contains(&benchmark),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Extended => "extended",
        }
    }
}

fn known_benchmark(year: ScoreYear, benchmark: &str) -> bool {
    ScoreTrack::Regular.includes(year, benchmark) || ScoreTrack::Extended.includes(year, benchmark)
}

/// A decided verdict, or the absence of one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Verdict {
    Sat,
    Unsat,
    Undecided,
}

impl Verdict {
    fn parse(raw: &str) -> Result<Self> {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "sat" | "violated" => Ok(Self::Sat),
            "unsat" | "holds" => Ok(Self::Unsat),
            "timeout" | "timed-out" | "unknown" | "error" | "no_result_in_file"
            | "error_nonmaximal" => Ok(Self::Undecided),
            value
                if value.starts_with("error_exit_code_")
                    || value.starts_with("prepare_instance_error_")
                    || value.starts_with("run_instance_timeout")
                    || value.starts_with("prepare_instance_timeout") =>
            {
                Ok(Self::Undecided)
            }
            _ => bail!("unrecognized result token {raw:?}"),
        }
    }

    fn solved(self) -> bool {
        matches!(self, Self::Sat | Self::Unsat)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Sat => "sat",
            Self::Unsat => "unsat",
            Self::Undecided => "undecided",
        }
    }
}

/// Stable identity of one instance.
///
/// The category is part of the key, and model/property paths retain their full
/// suffix beginning at `onnx/` or `vnnlib/`. This matches relative sweep paths
/// to absolute official paths without merging same-basename instances in
/// nested categories such as safenlp's `medical/` and `ruarobot/`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct InstanceKey {
    benchmark: String,
    onnx: String,
    vnnlib: String,
    /// Zero-based occurrence of a repeated instance key in one result file.
    ///
    /// Some official instance banks intentionally repeat the same model/spec
    /// pair. The repetitions are separately scored and must not be overwritten.
    occurrence: usize,
}

impl InstanceKey {
    fn new(benchmark: &str, onnx: &str, vnnlib: &str) -> Result<Self> {
        let raw_benchmark = benchmark.trim();
        let (_, canonical_benchmark) = ScoreYear::from_category_prefix(raw_benchmark);
        let benchmark_aliases = [raw_benchmark, canonical_benchmark];
        let benchmark_aliases = if raw_benchmark == canonical_benchmark {
            &benchmark_aliases[..1]
        } else {
            &benchmark_aliases[..]
        };
        let benchmark = canonical_benchmark.to_string();
        let onnx_entries = model_identity_entries(onnx)
            .with_context(|| format!("invalid model identity field {onnx:?}"))?;
        let onnx_entries = onnx_entries
            .into_iter()
            .map(|(label, path)| {
                (
                    label,
                    normalize_identity_field(&path, "onnx", benchmark_aliases),
                )
            })
            .collect::<Vec<_>>();
        let onnx = match onnx_entries.as_slice() {
            [(None, path)] => path.clone(),
            _ => serde_json::to_string(&onnx_entries)
                .context("serialize normalized relational model identity")?,
        };
        let vnnlib = normalize_identity_field(vnnlib, "vnnlib", benchmark_aliases);
        if benchmark.is_empty() || onnx.is_empty() || vnnlib.is_empty() {
            bail!(
                "result row has an empty identity field: category={benchmark:?}, \
                 onnx={onnx:?}, vnnlib={vnnlib:?}"
            );
        }
        Ok(Self {
            benchmark,
            onnx,
            vnnlib,
            occurrence: 0,
        })
    }

    fn display_name(&self) -> String {
        if self.occurrence == 0 {
            format!("{} | {}", self.onnx, self.vnnlib)
        } else {
            format!(
                "{} | {} [occurrence {}]",
                self.onnx,
                self.vnnlib,
                self.occurrence + 1
            )
        }
    }

    fn is_harness_test(&self) -> bool {
        let model_is_test = serde_json::from_str::<Vec<(Option<String>, String)>>(&self.onnx)
            .map_or_else(
                |_| normalized_basename(&self.onnx).starts_with("test_"),
                |entries| {
                    entries
                        .iter()
                        .any(|(_, path)| normalized_basename(path).starts_with("test_"))
                },
            );
        normalized_basename(&self.vnnlib).starts_with("test_") || model_is_test
    }

    fn semantic_key(&self) -> SemanticKey {
        (
            self.benchmark.clone(),
            self.onnx.clone(),
            self.vnnlib.clone(),
        )
    }

    /// Identity in historical organiser scoring logs: `<model-stem>` and
    /// `<property-stem>` joined by `-`, with directories and extensions dropped.
    ///
    /// Relational instances hold an ordered named-model list, with no single
    /// stem; return `None` rather than inventing a log identity.
    fn ground_truth_key(&self) -> Option<GroundTruthKey> {
        if self.onnx.starts_with('[') {
            return None;
        }
        let model = identity_stem(&self.onnx);
        let property = identity_stem(&self.vnnlib);
        if model.is_empty() || property.is_empty() {
            return None;
        }
        Some((self.benchmark.clone(), format!("{model}-{property}")))
    }
}

type SemanticKey = (String, String, String);

/// `(benchmark, "<model-stem>-<property-stem>")`, the organisers' row identity.
type GroundTruthKey = (String, String);

/// Basename with the `.onnx`, `.onnx.gz` or `.vnnlib` suffix removed.
fn identity_stem(path: &str) -> &str {
    let base = normalized_basename(path.trim());
    let base = base.strip_suffix(".gz").unwrap_or(base);
    let base = base.strip_suffix(".onnx").unwrap_or(base);
    base.strip_suffix(".vnnlib").unwrap_or(base)
}

fn normalized_basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Lexically normalize a result identity without consulting the local
/// filesystem (official result paths generally do not exist on this host).
fn normalize_identity_field(raw: &str, anchor: &str, benchmark_aliases: &[&str]) -> String {
    let slash_normalized = raw.trim().replace('\\', "/");
    let mut components = Vec::<String>::new();
    for component in slash_normalized.split('/') {
        let component = component.trim();
        if component.is_empty() || component == "." {
            continue;
        }
        if component == ".." {
            if components.last().is_some_and(|last| last != "..") {
                components.pop();
            } else {
                components.push(component.to_string());
            }
        } else {
            components.push(component.to_string());
        }
    }

    // Absolute official paths and relative sweep paths share this suffix.
    if let Some(position) = components
        .iter()
        .rposition(|component| component.eq_ignore_ascii_case(anchor))
    {
        components.drain(..position);
        // Anchor spelling is structural, not filesystem data. Canonicalizing
        // it also matches official Windows paths to sweep-relative Unix paths.
        components[0] = anchor.to_string();
    } else if let Some(position) = components.iter().rposition(|component| {
        benchmark_aliases
            .iter()
            .any(|benchmark| component == benchmark)
    }) {
        // Some categories store models outside literal `onnx/` and `vnnlib/`
        // directories. Absolute official and relative sweep paths still share
        // the suffix beneath the category directory.
        components.drain(..=position);
    }

    let normalized = components.join("/");
    if normalized.is_empty() {
        slash_normalized
    } else {
        normalized
    }
}

#[derive(Debug, Clone)]
struct RecordedVerdict {
    verdict: Verdict,
    source: String,
    source_csv: PathBuf,
    /// Explicit `2025_`/`2026_` category prefix, when the source uses one.
    ///
    /// Most released CSVs use unprefixed ids. Retaining an explicit prefix
    /// lets an explicitly selected scoring year reject accidentally mixed
    /// corpora even though the canonical instance key is prefix-free.
    year_hint: Option<ScoreYear>,
    /// Recorded wall-clock seconds, already validated finite and non-negative.
    seconds: f64,
    prepare_seconds: Option<f64>,
    /// Run tag from a measured-7 row (`...,run_id`), when present. A row
    /// without one cannot tie its verdict back to a sealed run, which matters
    /// when its runtime is otherwise indistinguishable from a hand-edited
    /// value (see `audit_budgets` on at-budget solves).
    tag: Option<String>,
    /// Complete six-column row hash after the organizer's verdict substitution.
    organizer_row_identity: Option<String>,
    network_field: String,
    property_field: String,
    /// Version retained independently of the normalized model/property key.
    vnnlib_version: Option<String>,
    counterexample: Option<CounterexampleCheck>,
}

/// Published `cex_checks.py` outcomes: own credit and strict SAT authority differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CounterexampleCheck {
    Strict,
    Tolerant,
    Missing,
    Malformed,
    Invalid,
    Unsupported,
}

impl CounterexampleCheck {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "correct" => Ok(Self::Strict),
            "correct_up_to_tolerance" => Ok(Self::Tolerant),
            "no_ce" => Ok(Self::Missing),
            "malformed_ce" => Ok(Self::Malformed),
            "exec_doesnt_match" | "spec_not_violated" | "wrong_shape" => Ok(Self::Invalid),
            "unsupported" => Ok(Self::Unsupported),
            _ => bail!("unrecognized organizer counterexample outcome {value:?}"),
        }
    }

    fn metadata(self) -> (&'static str, bool, bool) {
        match self {
            Self::Strict => ("correct", true, true),
            Self::Tolerant => ("correct_with_tolerance", false, true),
            Self::Missing => ("missing", false, false),
            Self::Malformed => ("malformed", false, false),
            Self::Invalid => ("incorrect", false, false),
            Self::Unsupported => ("unsupported", false, false),
        }
    }
}

fn organizer_row_identity(fields: &[String]) -> Result<String> {
    let mut row = fields.to_vec();
    row[4] = "violated".into();
    // Python's json.dumps(..., separators=(',', ':')) defaults to ensure_ascii.
    let json = serde_json::to_string(&row)?;
    let mut canonical = String::with_capacity(json.len());
    for ch in json.chars() {
        if ch.is_ascii() {
            canonical.push(ch);
        } else {
            use std::fmt::Write;
            let mut units = [0; 2];
            for unit in ch.encode_utf16(&mut units) {
                write!(&mut canonical, "\\u{unit:04x}")?;
            }
        }
    }
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

fn row_vnnlib_version(property: &str) -> Option<String> {
    let parts = property.split('/').collect::<Vec<_>>();
    parts.windows(2).find_map(|pair| {
        (pair[1] == "vnnlib" && matches!(pair[0], "1.0" | "2.0")).then(|| pair[0].to_owned())
    })
}

type ResultRows = BTreeMap<InstanceKey, RecordedVerdict>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CsvSchema {
    /// Legacy NY sweep: `category,onnx,vnnlib,verdict,seconds`.
    LegacySweep5,
    /// Official aggregate:
    /// `category,network,property,prepare_time,result,run_time`.
    Official6,
    /// Historical NY measurements:
    /// `category,onnx,vnnlib,prepare_status,verdict,seconds`.
    Measured6,
    /// `category,onnx,vnnlib,prepare_status,verdict,seconds,run_id`
    Measured7,
}

impl CsvSchema {
    fn for_fields(fields: &[String], path: &Path, line: usize) -> Result<Self> {
        let schema = match fields.len() {
            5 => Self::LegacySweep5,
            6 if fields[3].trim().eq_ignore_ascii_case("prepared") => Self::Measured6,
            6 => Self::Official6,
            7 => Self::Measured7,
            _ => bail!(
                "{}:{}: unsupported results schema with {} columns; expected \
                 legacy sweep-5 (category,onnx,vnnlib,verdict,seconds), official-6 \
                 (category,network,property,prepare_time,result,run_time), measured-6 \
                 (...,prepare_status,verdict,seconds), or \
                 measured-7 (...,prepare_status,verdict,seconds,run_id)",
                path.display(),
                line,
                fields.len()
            ),
        };

        if matches!(schema, Self::Official6) {
            validate_nonnegative_finite(&fields[3], "prepare_time", path, line)?;
        } else if matches!(schema, Self::Measured7)
            && !fields[3].trim().eq_ignore_ascii_case("prepared")
        {
            validate_nonnegative_finite(&fields[3], "prepare_time/status", path, line)?;
        }
        validate_nonnegative_finite(&fields[schema.runtime_index()], "run_time", path, line)?;
        Ok(schema)
    }

    fn verdict_index(self) -> usize {
        match self {
            Self::LegacySweep5 => 3,
            Self::Official6 | Self::Measured6 | Self::Measured7 => 4,
        }
    }

    /// Index of the recorded wall-clock runtime column.
    ///
    /// Kept beside `verdict_index` because getting these two confused is how a
    /// seven-column run ID once became a verdict.
    fn runtime_index(self) -> usize {
        match self {
            Self::LegacySweep5 => 4,
            Self::Official6 | Self::Measured6 | Self::Measured7 => 5,
        }
    }
}

fn validate_nonnegative_finite(raw: &str, field: &str, path: &Path, line: usize) -> Result<f64> {
    let value = raw
        .trim()
        .parse::<f64>()
        .with_context(|| format!("{}:{line}: invalid {field} {raw:?}", path.display()))?;
    if !value.is_finite() || value < 0.0 {
        bail!(
            "{}:{line}: {field} must be finite and non-negative, got {raw:?}",
            path.display()
        );
    }
    Ok(value)
}

fn is_results_header(fields: &[String]) -> bool {
    let Some(category) = fields.first() else {
        return false;
    };
    let Some(network) = fields.get(1) else {
        return false;
    };
    let Some(property) = fields.get(2) else {
        return false;
    };
    let category = category.trim().to_ascii_lowercase();
    let network = network.trim().to_ascii_lowercase();
    let property = property.trim().to_ascii_lowercase();
    matches!(category.as_str(), "category" | "cat" | "benchmark")
        && matches!(
            network.as_str(),
            "network" | "onnx" | "onnx_path" | "model" | "model_path"
        )
        && matches!(
            property.as_str(),
            "property" | "vnnlib" | "vnnlib_path" | "spec" | "specification"
        )
}

fn insert_unique(rows: &mut ResultRows, key: InstanceKey, entry: RecordedVerdict) -> Result<()> {
    if let Some(previous) = rows.get(&key) {
        bail!(
            "duplicate result identity for category {:?}, model {:?}, property {:?}: \
             first at {}, again at {}",
            key.benchmark,
            key.onnx,
            key.vnnlib,
            previous.source,
            entry.source
        );
    }
    rows.insert(key, entry);
    Ok(())
}

/// Merge banks idempotently. The same identity/verdict may be present in a
/// canonical bank and an immutable sealed snapshot; a disagreement is
/// ambiguous and is refused instead of depending on filesystem iteration.
fn merge_compatible(target: &mut ResultRows, source: ResultRows) -> Result<()> {
    for (key, entry) in source {
        if let Some(previous) = target.get_mut(&key) {
            if previous.verdict != entry.verdict {
                bail!(
                    "conflicting result identity for category {:?}, model {:?}, property {:?}, \
                     occurrence {}: {:?} at {}, {:?} at {}",
                    key.benchmark,
                    key.onnx,
                    key.vnnlib,
                    key.occurrence + 1,
                    previous.verdict,
                    previous.source,
                    entry.verdict,
                    entry.source
                );
            }
            if let (Some(previous_year), Some(entry_year)) = (previous.year_hint, entry.year_hint) {
                if previous_year != entry_year {
                    bail!(
                        "mixed competition-year prefixes for category {:?}, model {:?}, \
                         property {:?}: {} at {}, {} at {}",
                        key.benchmark,
                        key.onnx,
                        key.vnnlib,
                        previous_year.number(),
                        previous.source,
                        entry_year.number(),
                        entry.source
                    );
                }
            } else if previous.year_hint.is_none() {
                previous.year_hint = entry.year_hint;
            }
            continue;
        }
        target.insert(key, entry);
    }
    Ok(())
}

/// Read one supported result CSV. Schemas are positional and explicit so a
/// seven-column run ID can never be mistaken for the verdict.
fn read_results(path: &Path) -> Result<ResultRows> {
    let body = std::fs::read_to_string(path)
        .with_context(|| format!("read results {}", path.display()))?;
    let mut out = BTreeMap::new();
    let mut occurrences = BTreeMap::<SemanticKey, usize>::new();
    for (offset, raw) in body.lines().enumerate() {
        let line_number = offset + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = parse_csv_line(line)
            .with_context(|| format!("{}:{line_number}: malformed CSV", path.display()))?;
        if is_results_header(&fields) {
            if !matches!(fields.len(), 5..=7) {
                bail!(
                    "{}:{line_number}: malformed results header with {} columns",
                    path.display(),
                    fields.len()
                );
            }
            continue;
        }
        let schema = CsvSchema::for_fields(&fields, path, line_number)?;
        let verdict = Verdict::parse(&fields[schema.verdict_index()])
            .with_context(|| format!("{}:{line_number}", path.display()))?;
        let (year_hint, _) = ScoreYear::from_category_prefix(fields[0].trim());
        let mut key = InstanceKey::new(&fields[0], &fields[1], &fields[2])
            .with_context(|| format!("{}:{line_number}", path.display()))?;
        if key.is_harness_test() {
            continue;
        }
        let occurrence = occurrences.entry(key.semantic_key()).or_default();
        key.occurrence = *occurrence;
        *occurrence += 1;
        let source = format!("{}:{line_number}", path.display());
        let seconds = validate_nonnegative_finite(
            &fields[schema.runtime_index()],
            "run_time",
            path,
            line_number,
        )?;
        let tag = (fields.len() == 7)
            .then(|| fields[6].trim().to_string())
            .filter(|tag| !tag.is_empty());
        insert_unique(
            &mut out,
            key,
            RecordedVerdict {
                verdict,
                source,
                source_csv: path.to_path_buf(),
                year_hint,
                seconds,
                prepare_seconds: (schema == CsvSchema::Official6)
                    .then(|| fields[3].parse::<f64>())
                    .transpose()?,
                tag,
                organizer_row_identity: (schema == CsvSchema::Official6 && verdict == Verdict::Sat)
                    .then(|| organizer_row_identity(&fields))
                    .transpose()?,
                network_field: fields[1].clone(),
                property_field: fields[2].clone(),
                vnnlib_version: row_vnnlib_version(&fields[2]),
                counterexample: None,
            },
        )?;
    }
    Ok(out)
}

/// Read every direct `*.csv` child under a directory (the shape of
/// `reports/measured/`) in deterministic path order.
fn read_results_dir(dir: &Path) -> Result<ResultRows> {
    let mut paths = Vec::new();
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("read results dir {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("read entry under {}", dir.display()))?;
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("csv"))
        {
            paths.push(path);
        }
    }
    paths.sort();

    let mut merged = BTreeMap::new();
    for path in paths {
        merge_compatible(&mut merged, read_results(&path)?)?;
    }
    Ok(merged)
}

fn load(path: &Path) -> Result<ResultRows> {
    if path.is_dir() {
        read_results_dir(path)
    } else {
        read_results(path)
    }
}

/// Official per-instance timeouts, keyed by the same normalized identity the
/// result rows use so official and sweep path spellings both match.
type BudgetTable = BTreeMap<SemanticKey, u64>;

/// Load official per-instance budgets from every category's `instances.csv`.
///
/// Budgets are per INSTANCE. Assuming one budget for a whole benchmark is how
/// seven nn4sys rows with 20-100 s timeouts were once "solved" in 75-724 s.
///
/// An identity that appears in several instance lists (the 2026 corpus has
/// parallel VNN-LIB 1.0/2.0 lists) must agree on its budget. Disagreement is
/// refused rather than resolved by picking one: guessing here would either
/// invent phantom points or discard real ones.
fn load_budgets(years: &[u32]) -> Result<BudgetTable> {
    let mut categories = Vec::new();
    for &year in years {
        categories.extend(
            discover_categories(year)
                .with_context(|| format!("discover VNN-COMP {year} benchmark categories"))?,
        );
    }
    let table = budgets_from_categories(&categories)?;
    if table.is_empty() {
        bail!(
            "no official per-instance budgets found for year(s) {}; refusing to report a \
             budget audit that checked nothing",
            years
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(table)
}

/// Build the budget table from already-resolved `(category, directory)` pairs.
fn budgets_from_categories(categories: &[(String, PathBuf)]) -> Result<BudgetTable> {
    let mut out = BudgetTable::new();
    for (category, dir) in categories {
        for list in instance_lists_under(dir)? {
            let body = std::fs::read_to_string(&list)
                .with_context(|| format!("read {}", list.display()))?;
            let instances =
                parse_instances(&body).with_context(|| format!("parse {}", list.display()))?;
            for instance in instances {
                let key = InstanceKey::new(category, &instance.onnx_field, &instance.vnnlib_field)
                    .with_context(|| list.display().to_string())?
                    .semantic_key();
                match out.get(&key) {
                    Some(&previous) if previous != instance.budget_secs => bail!(
                        "conflicting official budgets for {} / {} / {}: {previous}s and {}s \
                         (seen while reading {})",
                        key.0,
                        key.1,
                        key.2,
                        instance.budget_secs,
                        list.display()
                    ),
                    Some(_) => {}
                    None => {
                        out.insert(key, instance.budget_secs);
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Every instance list under a category, including both members of a versioned
/// 1.0/2.0 pair.
///
/// `instances_csv_for` deliberately refuses an ambiguous versioned layout when
/// choosing ONE list to sweep. A budget audit wants them all, and they are
/// required to agree, so ambiguity is not a problem here.
fn instance_lists_under(category_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let direct = category_dir.join("instances.csv");
    if direct.is_file() {
        out.push(direct);
    }
    for entry in std::fs::read_dir(category_dir)
        .with_context(|| format!("read category directory {}", category_dir.display()))?
    {
        let path = entry
            .with_context(|| format!("read entry under {}", category_dir.display()))?
            .path();
        if path.is_dir() {
            let nested = path.join("instances.csv");
            if nested.is_file() {
                out.push(nested);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Official field results, keyed by the true tool/team identity.
type OfficialField = BTreeMap<String, ResultRows>;

fn collect_official_csvs(dir: &Path, root: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("read official results dir {}", dir.display()))?
    {
        entries.push(
            entry
                .with_context(|| format!("read entry under {}", dir.display()))?
                .path(),
        );
    }
    entries.sort();

    for path in entries {
        let metadata = std::fs::symlink_metadata(&path)
            .with_context(|| format!("inspect official result path {}", path.display()))?;
        if metadata.file_type().is_symlink() {
            bail!(
                "official results corpus contains a symlink, which is refused to avoid \
                 traversal outside the selected corpus: {}",
                path.display()
            );
        }
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let under_hidden_directory = relative
            .components()
            .any(|component| component.as_os_str().to_string_lossy().starts_with('.'));
        let under_scoring_output = relative.components().any(|component| {
            component
                .as_os_str()
                .to_string_lossy()
                .to_ascii_uppercase()
                .starts_with("SCORING")
        });
        if under_hidden_directory || under_scoring_output {
            continue;
        }
        if metadata.is_dir() {
            collect_official_csvs(&path, root, paths)?;
        } else if metadata.is_file() {
            let direct_csv = path.parent() == Some(root)
                && path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("csv"));
            let nested_results = path
                .file_name()
                .is_some_and(|name| name.eq_ignore_ascii_case("results.csv"));
            if direct_csv || nested_results {
                paths.push(path);
            }
        }
    }
    Ok(())
}

fn official_tool_name(root: &Path, path: &Path) -> Result<String> {
    let relative = path
        .strip_prefix(root)
        .with_context(|| format!("{} is not under {}", path.display(), root.display()))?;
    let components = relative.components().collect::<Vec<_>>();
    let name = if components.len() > 1 {
        components[0].as_os_str().to_string_lossy().into_owned()
    } else {
        path.file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    if name.is_empty() {
        bail!("cannot derive official tool name from {}", path.display());
    }
    Ok(name)
}

fn load_official_field(root: &Path) -> Result<OfficialField> {
    if !root.is_dir() {
        bail!(
            "official results corpus is not a directory: {}",
            root.display()
        );
    }
    let mut paths = Vec::new();
    collect_official_csvs(root, root, &mut paths)?;
    paths.sort();

    let mut field = BTreeMap::<String, ResultRows>::new();
    for path in paths {
        let tool = official_tool_name(root, &path)?;
        if tool.eq_ignore_ascii_case("rover") {
            continue;
        }
        let rows = read_results(&path)?;
        merge_compatible(field.entry(tool).or_default(), rows)?;
    }
    field.retain(|_, rows| !rows.is_empty());
    if field.is_empty() {
        bail!(
            "no tool result rows found under official corpus {}",
            root.display()
        );
    }
    Ok(field)
}

struct CounterexampleRecord {
    value: serde_json::Value,
    row_identity: String,
    category: String,
    version: String,
    outcome: CounterexampleCheck,
}

fn hex_digest(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn read_counterexample_check(path: &Path) -> Result<CounterexampleRecord> {
    let value: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path)
            .with_context(|| format!("read counterexample check {}", path.display()))?,
    )?;
    let string = |pointer: &str| {
        value
            .pointer(pointer)
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("{}: missing string {pointer}", path.display()))
    };
    if value["schema_version"] != 1
        || string("/checker_version")? != "vnncomp2026-cex-1"
        || string("/provider")? != "CPUExecutionProvider"
    {
        bail!(
            "{}: unsupported organizer counterexample check",
            path.display()
        );
    }
    for pointer in [
        "/dependency_versions/python",
        "/dependency_versions/onnx",
        "/dependency_versions/onnxruntime",
    ] {
        if string(pointer)?.is_empty() {
            bail!("{}: empty checker dependency version", path.display());
        }
    }
    let outcome = CounterexampleCheck::parse(string("/result")?)?;
    let (classification, strict, credit) = outcome.metadata();
    if string("/classification")? != classification
        || value["establishes_sat"] != strict
        || value["score_credit"] != credit
    {
        bail!(
            "{}: counterexample result and derived flags disagree",
            path.display()
        );
    }
    let row_identity = string("/results_row_identity")?.to_owned();
    let category = string("/benchmark/category")?.to_owned();
    let version = string("/benchmark/version")?.to_owned();
    if !hex_digest(&row_identity, 64) || !matches!(version.as_str(), "1.0" | "2.0") {
        bail!(
            "{}: invalid result-row identity or VNN-LIB version",
            path.display()
        );
    }
    Ok(CounterexampleRecord {
        value,
        row_identity,
        category,
        version,
        outcome,
    })
}

fn counterexample_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        if path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().ends_with(".check.json"))
        {
            let metadata = std::fs::symlink_metadata(&path)?;
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                bail!(
                    "counterexample check is not a regular file: {}",
                    path.display()
                );
            }
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn load_published_counterexample_checks(root: &Path, field: &mut OfficialField) -> Result<()> {
    for (tool, rows) in field {
        let categories = rows
            .keys()
            .map(|key| key.benchmark.clone())
            .collect::<BTreeSet<_>>();
        let mut checks = BTreeMap::new();
        for category in categories {
            let directory = root.join(tool.as_str()).join(format!("2026_{category}"));
            if !directory.is_dir() {
                continue;
            }
            for path in counterexample_files(&directory)? {
                let check = read_counterexample_check(&path)?;
                if check.category != format!("2026_{category}") {
                    bail!("{}: check belongs to another category", path.display());
                }
                let identity = (check.category, check.version, check.row_identity);
                if let Some(previous) = checks.insert(identity, check.outcome) {
                    if previous != check.outcome {
                        bail!(
                            "{}: conflicting per-run counterexample checks",
                            path.display()
                        );
                    }
                }
            }
        }
        for (key, entry) in rows {
            if let (Some(version), Some(identity)) =
                (&entry.vnnlib_version, &entry.organizer_row_identity)
            {
                entry.counterexample = checks
                    .get(&(
                        format!("2026_{}", key.benchmark),
                        version.clone(),
                        identity.clone(),
                    ))
                    .copied();
            }
        }
    }
    Ok(())
}

fn checked_replay_pin(value: &serde_json::Value, label: &str) -> Result<PathBuf> {
    let path = value["path"]
        .as_str()
        .with_context(|| format!("{label}: missing path"))?;
    let expected = value["sha256"]
        .as_str()
        .with_context(|| format!("{label}: missing SHA"))?;
    let bytes = value["bytes"]
        .as_u64()
        .with_context(|| format!("{label}: missing size"))?;
    let (canonical, actual, size) = crate::build_id::sha256_regular_file_nofollow(Path::new(path))?;
    if !hex_digest(expected, 64)
        || canonical != Path::new(path)
        || actual != expected
        || size != bytes
    {
        bail!("{label}: replay artifact path, digest or size differs");
    }
    Ok(canonical)
}

fn pinned_json(value: &serde_json::Value, label: &str) -> Result<serde_json::Value> {
    let path = checked_replay_pin(value, label)?;
    let bytes = std::fs::read(path)?;
    if format!("{:x}", Sha256::digest(&bytes)) != value["sha256"].as_str().unwrap_or("") {
        bail!("{label}: JSON changed after pin validation");
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn same_replay_pin(left: &serde_json::Value, right: &serde_json::Value) -> bool {
    left["path"] == right["path"]
        && left["sha256"] == right["sha256"]
        && left.get("bytes").or_else(|| left.get("size_bytes"))
            == right.get("bytes").or_else(|| right.get("size_bytes"))
}

fn completed_replay_process(value: &serde_json::Value, budget: Option<f64>) -> bool {
    value["returncode"] == 0
        && value["timed_out"] == false
        && value["interrupted"] == false
        && value["error"].is_null()
        && value["wall_seconds"].as_f64().is_some_and(|seconds| {
            seconds.is_finite()
                && seconds >= 0.0
                && budget.is_none_or(|limit| {
                    seconds <= limit && value["timeout_seconds"].as_f64() == Some(limit)
                })
        })
}

fn validate_ny_counterexample_check(
    check: &CounterexampleRecord,
    source: &str,
    key: &InstanceKey,
    entry: &RecordedVerdict,
) -> Result<String> {
    let evidence = &check.value["ny_evidence"];
    if evidence["schema"] != "ny_vnncomp2026_row_replay_evidence_v1"
        || evidence["tool"] != "ny"
        || evidence["validation_only"] != false
        || evidence["source_commit"] != source
        || evidence["occurrence"].as_u64() != Some(key.occurrence as u64)
        || check.value["benchmark"]["network_field"] != entry.network_field
        || check.value["benchmark"]["property_field"] != entry.property_field
        || entry.prepare_seconds != Some(0.0)
    {
        bail!("NY check is validation-only or belongs to another source/row");
    }
    let csv = checked_replay_pin(&evidence["results_csv"], "results CSV")?;
    if csv != std::fs::canonicalize(&entry.source_csv)? {
        bail!("NY replay check belongs to another result CSV");
    }
    for name in ["model", "property", "image", "original_result", "witness"] {
        checked_replay_pin(&evidence[name], name)?;
    }
    let replay = pinned_json(&evidence["replay"], "replay")?;
    let start = pinned_json(&evidence["pilot_start"], "pilot start")?;
    let observation = pinned_json(&evidence["pilot_observation"], "pilot observation")?;
    let request = pinned_json(&evidence["replay_request"], "replay request")?;
    let process = pinned_json(&evidence["replay_process"], "replay process")?;
    let response = pinned_json(&evidence["replay_stdout"], "replay response")?;
    let dependencies = pinned_json(&evidence["replay_dependencies"], "replay dependencies")?;
    let batch = pinned_json(&evidence["batch_result"], "replay batch")?;
    let plan = pinned_json(&evidence["batch_plan"], "replay batch plan")?;
    let exec_start = pinned_json(&evidence["exec_start"], "original exec")?;
    let empty = format!("{:x}", Sha256::digest(b""));
    let fields = &start["build_identity"]["fields"];
    let image = &evidence["image"];
    let run_dir = Path::new(
        evidence["run_id"]
            .as_str()
            .context("missing original run directory")?,
    );
    if start["schema"] != "ny_cct_pilot_start_v2"
        || start["arm"] != "candidate"
        || start["source"]["head"] != source
        || fields["commit"] != source
        || fields["source"] != "clean"
        || fields["dirty"] != "false"
        || start["source"]["status_sha256"] != empty
        || start["source"]["tracked_diff_sha256"] != empty
        || fields["exe"] != image["sha256"]
        || fields["features"] != "mip,cuda"
        || fields["profile"] != "release"
        || !same_replay_pin(&start["binary"], image)
        || !same_replay_pin(&start["row"]["model"], &evidence["model"])
        || !same_replay_pin(&start["row"]["property"], &evidence["property"])
        || evidence["model"]["path"] != entry.network_field
        || evidence["property"]["path"] != entry.property_field
        || start["argv"] != exec_start["argv"]
        || evidence["pilot_start"]["path"].as_str().map(Path::new)
            != Some(run_dir.join("start.json").as_path())
        || evidence["pilot_observation"]["path"]
            .as_str()
            .map(Path::new)
            != Some(run_dir.join("observation.json").as_path())
        || evidence["original_result"]["path"].as_str().map(Path::new)
            != Some(run_dir.join("result.txt").as_path())
    {
        bail!("NY replay source, image or original inputs differ");
    }
    let budget = start["row"]["timeout_seconds"]
        .as_f64()
        .context("missing original budget")?;
    if !budget.is_finite()
        || budget <= 0.0
        || observation["status"] != "completed"
        || observation["protocol_verdict"] != "sat"
        || observation["drained"] != true
        || observation
            .get("outer_timeout_censored")
            .is_some_and(|value| value.as_bool() != Some(false))
        || !completed_replay_process(&observation["process"], Some(budget))
        || observation["process"]["wall_seconds"].as_f64() != Some(entry.seconds)
        || !completed_replay_process(&process, None)
        || evidence["original_timeout_seconds"].as_f64() != Some(budget)
        || evidence["original_wall_seconds"].as_f64() != Some(entry.seconds)
    {
        bail!("NY SAT was late, censored, incomplete or has a different runtime");
    }
    let expected_prefix = [
        image["path"]
            .as_str()
            .context("missing image path")?
            .to_owned(),
        "-v".into(),
        "vnncomp".into(),
        "v1".into(),
        key.benchmark.clone(),
        entry.network_field.clone(),
        entry.property_field.clone(),
        evidence["original_result"]["path"]
            .as_str()
            .context("missing original result")?
            .to_owned(),
        budget.to_string(),
    ];
    let argv = start["argv"]
        .as_array()
        .context("missing original solver argv")?;
    if argv.len() < 11
        || !argv
            .iter()
            .zip(&expected_prefix)
            .all(|(actual, expected)| actual.as_str() == Some(expected.as_str()))
    {
        bail!("NY original solver invocation differs from the scored row/source/budget");
    }
    if replay["schema"] != "ny_cct_direct_official_sat_replay_v1"
        || !same_replay_pin(&replay["pilot_start"], &evidence["pilot_start"])
        || !same_replay_pin(&replay["pilot_observation"], &evidence["pilot_observation"])
        || !same_replay_pin(&replay["original_result"], &evidence["original_result"])
        || replay["response"] != response
        || replay["dependencies"] != dependencies
        || response["ok"] != true
        || response["provider"] != "CPUExecutionProvider"
        || response["result"] != check.value["result"]
        || replay["independent_local_strict_sat"] != check.value["establishes_sat"]
        || replay["official_score_credit"] != false
        || dependencies["ready"] != true
        || dependencies["checker_commit"] != "eea564f5d0a8751d8321a409214db14d03f452b8"
        || request["onnx_path"] != evidence["model"]["path"]
        || request["vnnlib_path"] != evidence["property"]["path"]
        || request["vnnlib_version"] != check.version
        || request["witness_path"] != evidence["witness"]["path"]
        || check.value["witness_sha256"] != evidence["witness"]["sha256"]
        || check.value["witness_path"] != evidence["witness"]["path"]
    {
        bail!("NY checker response is not bound to the original source/witness");
    }
    for name in ["python", "onnx", "onnxruntime"] {
        if response["dependency_versions"][name] != check.value["dependency_versions"][name] {
            bail!("NY checker dependency version differs: {name}");
        }
    }
    let checker_sources = evidence["checker_sources"]
        .as_array()
        .context("missing checker sources")?;
    if checker_sources.len() != 3 {
        bail!("incomplete final checker source pins");
    }
    let required_names = BTreeSet::from([
        "cex_checks.py",
        "counterexamples.py",
        "counterexamples_v2.py",
    ]);
    let mut actual_names = BTreeSet::new();
    for item in checker_sources {
        let path = checked_replay_pin(item, "checker source")?;
        actual_names.insert(
            path.file_name()
                .and_then(|name| name.to_str())
                .context("invalid checker source name")?
                .to_owned(),
        );
        if !dependencies["checker_sources"]
            .as_array()
            .context("missing replay checker sources")?
            .iter()
            .any(|source| same_replay_pin(source, item))
        {
            bail!("NY record checker sources differ from the replay");
        }
    }
    if actual_names
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != required_names
    {
        bail!("NY record does not bind the three final checker source files");
    }
    let contract = "ny_research_actual_sat_replay_adapter_v2";
    if source == "f3a24a21a07233f16d92e1e390cb4ce93d6659c2"
        || batch["schema"] != "ny_cct_actual_sat_replay_batch_result_v2"
        || batch["all_completed_sat_discovered"] != true
        || batch["official_score_credit"] != false
        || evidence["official_score_credit"] != false
        || plan["adapter_contract"] != contract
        || batch["collection"]["adapter_contract"] != contract
        || plan["old_replay_passes_transferred"] != 0
        || batch["collection"]["old_replay_passes_transferred"] != 0
        || plan["source_commits"]["candidate"] != source
        || batch["collection"]["source_commits"]["candidate"] != source
        || plan["images"]["candidate"]["sha256"] != image["sha256"]
    {
        bail!("old/control replay evidence cannot supply fresh NY score credit");
    }
    let matching = batch["adjudications"]
        .as_array()
        .context("missing actual batch adjudications")?
        .iter()
        .filter(|item| {
            item["arm"] == "candidate"
                && item["run_dir"] == evidence["run_id"]
                && item["row"] == start["row"]["index_zero_based"]
                && same_replay_pin(&item["replay"], &evidence["replay"])
                && same_replay_pin(&item["result"], &evidence["original_result"])
                && item["adjudication"] == check.value["result"]
                && item["strict_correct"] == check.value["establishes_sat"]
                && item["input_tolerance_only"] == (check.outcome == CounterexampleCheck::Tolerant)
                && completed_replay_process(&item["command"], None)
        })
        .count();
    if matching != 1 {
        bail!("NY check does not bind one actual replay-batch adjudication");
    }
    let raw = std::fs::read(
        evidence["original_result"]["path"]
            .as_str()
            .context("missing result path")?,
    )?;
    let Some(newline) = raw.iter().position(|&byte| byte == b'\n') else {
        bail!("NY result has no SAT header");
    };
    if !raw[..newline].trim_ascii().eq_ignore_ascii_case(b"sat")
        || std::fs::read(
            evidence["witness"]["path"]
                .as_str()
                .context("missing witness path")?,
        )? != raw[newline + 1..]
    {
        bail!("NY replay changed the original witness assignment");
    }
    Ok(image["sha256"]
        .as_str()
        .context("missing image SHA")?
        .to_owned())
}

fn load_ny_counterexample_checks(
    directory: &Path,
    source: &str,
    current: &mut ResultRows,
) -> Result<()> {
    let paths = counterexample_files(directory)?;
    if paths.is_empty() {
        bail!("--ny-ce-checks contains no replay checks");
    }
    let mut image = None;
    let mut used = BTreeSet::new();
    for path in paths {
        let check = read_counterexample_check(&path)?;
        let matched = current
            .iter()
            .filter(|(key, entry)| {
                entry.verdict == Verdict::Sat
                    && check.category == format!("2026_{}", key.benchmark)
                    && entry.vnnlib_version.as_deref() == Some(check.version.as_str())
                    && entry.organizer_row_identity.as_deref() == Some(check.row_identity.as_str())
                    && check.value["ny_evidence"]["occurrence"].as_u64()
                        == Some(key.occurrence as u64)
            })
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        let [key] = matched.as_slice() else {
            bail!(
                "{}: NY check does not bind exactly one current SAT row",
                path.display()
            );
        };
        if !used.insert(key.clone()) {
            bail!("duplicate NY replay check for one result occurrence");
        }
        let actual_image = validate_ny_counterexample_check(&check, source, key, &current[key])?;
        if image
            .as_ref()
            .is_some_and(|expected| expected != &actual_image)
        {
            bail!("NY replay checks mix executable images");
        }
        image = Some(actual_image);
        current.get_mut(key).expect("matched row").counterexample = Some(check.outcome);
    }
    Ok(())
}

/// Where a released results corpus keeps the organisers' own scoring log.
const GROUND_TRUTH_RELATIVE_PATH: &str = "SCORING/results.txt";

/// Historical stem-based summary used by the 2025 model. Ambiguous names are
/// dropped and reported. The 2026 model uses full-row per-run checks instead:
/// a `True Result` summary cannot express tolerant own-tool acceptance.
#[derive(Debug, Default)]
pub(crate) struct GroundTruth {
    verdicts: BTreeMap<GroundTruthKey, Verdict>,
    /// Keys seen with two DIFFERENT verdicts, hence unusable for scoring.
    ambiguous: BTreeSet<String>,
    /// Human-readable provenance, printed with the score.
    source: String,
}

impl GroundTruth {
    fn get(&self, key: &InstanceKey) -> Option<Verdict> {
        self.verdicts.get(&key.ground_truth_key()?).copied()
    }

    /// Not named `len`: clippy's `len_without_is_empty` would then demand an
    /// `is_empty` whose only honest answer is already a refusal at parse time.
    fn instance_count(&self) -> usize {
        self.verdicts.len()
    }
}

/// Extract the first quoted element of a Python list literal, e.g. the instance
/// name in `Row: ['lindex-lindex_400', '1.0 (h)', ...]`.
fn first_quoted_element(rest: &str) -> Option<String> {
    let rest = rest.trim_start();
    let quote = rest.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let body = &rest[quote.len_utf8()..];
    let end = body.find(quote)?;
    Some(body[..end].to_string())
}

/// Parse the organisers' scoring log into per-instance truth.
///
/// The log is line-oriented: a `Category <id>:` line opens a category, a
/// `Row: [...]` line names an instance, and the `True Result:` line that follows
/// it carries the adjudicated verdict (or `-` when the organisers left the row
/// undecided, which claims nothing and is skipped).
fn parse_ground_truth_log(body: &str, source: &str) -> Result<GroundTruth> {
    let mut seen = BTreeMap::<GroundTruthKey, BTreeSet<Verdict>>::new();
    let mut benchmark: Option<String> = None;
    let mut pending: Option<String> = None;
    for line in body.lines() {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix("Category ") {
            if let Some(name) = rest.strip_suffix(':') {
                let (_, canonical) = ScoreYear::from_category_prefix(name.trim());
                benchmark = Some(canonical.to_string());
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("Row: [") {
            pending = first_quoted_element(rest);
            continue;
        }
        let Some(rest) = line.strip_prefix("True Result:") else {
            continue;
        };
        let (Some(benchmark), Some(instance)) = (benchmark.as_ref(), pending.take()) else {
            continue;
        };
        let verdict = match rest.trim() {
            "sat" => Verdict::Sat,
            "unsat" => Verdict::Unsat,
            // `-` means the organisers themselves recorded no truth here.
            _ => continue,
        };
        seen.entry((benchmark.clone(), instance))
            .or_default()
            .insert(verdict);
    }
    if seen.is_empty() {
        bail!(
            "no `True Result:` rows parsed from organiser ground truth at {source}; an empty \
             truth table is a PARSE FAILURE, not a clean absence of truth"
        );
    }
    let mut truth = GroundTruth {
        source: source.to_string(),
        ..Default::default()
    };
    for (key, verdicts) in seen {
        match verdicts.iter().copied().collect::<Vec<_>>().as_slice() {
            [only] => {
                truth.verdicts.insert(key, *only);
            }
            _ => {
                truth.ambiguous.insert(format!("{} | {}", key.0, key.1));
            }
        }
    }
    Ok(truth)
}

/// Resolve a `--ground-truth` argument, which may be the released corpus root,
/// the `SCORING` directory inside it, or the log file itself.
fn load_ground_truth(path: &Path) -> Result<GroundTruth> {
    let file = if path.is_dir() {
        let nested = path.join(GROUND_TRUTH_RELATIVE_PATH);
        if nested.is_file() {
            nested
        } else {
            let direct = path.join("results.txt");
            if direct.is_file() {
                direct
            } else {
                bail!(
                    "no organiser scoring log under {}: expected {} or results.txt",
                    path.display(),
                    GROUND_TRUTH_RELATIVE_PATH
                )
            }
        }
    } else {
        path.to_path_buf()
    };
    let body = std::fs::read_to_string(&file)
        .with_context(|| format!("read organiser ground truth {}", file.display()))?;
    parse_ground_truth_log(&body, &file.display().to_string())
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct BenchProgress {
    pub(crate) benchmark: String,
    pub(crate) current_solved: usize,
    /// Historical comparison count. This is a reference, not ground truth.
    pub(crate) reference_solved: usize,
    pub(crate) new_solves: Vec<String>,
    pub(crate) regressions: Vec<String>,
    pub(crate) verdict_flips: Vec<String>,
    /// Solved rows whose recorded runtime exceeds their official per-instance
    /// budget. These score zero in the real harness and are excluded below.
    pub(crate) over_budget: Vec<String>,
    /// Undecided rows recorded below 90% of their official budget — invalid
    /// measurements that must not be read as capability limits.
    pub(crate) under_measured: Vec<String>,
    /// Solved rows recorded exactly at their budget, surfaced for provenance
    /// review (legal at the wire; also what a hand-edited row looks like).
    pub(crate) at_budget_solves: Vec<String>,
    /// Solved rows actually compared against a budget.
    pub(crate) budget_checked: usize,
    /// Solved rows with no budget in the supplied corpora — unchecked, and
    /// deliberately not reported as passing.
    pub(crate) budget_unchecked: usize,
    /// Union of current and reference identities considered for the diff.
    pub(crate) rows_compared: usize,
    /// Current raw points under the VNN-COMP +10/-150 model.
    pub(crate) current_raw_score: Option<i64>,
    pub(crate) current_correct: Option<usize>,
    pub(crate) current_incorrect: Option<usize>,
    /// Decided current rows withheld from scoring because the instance is
    /// disputed: one participant claims SAT, others hold UNSAT, and no
    /// published organiser truth settles it.
    pub(crate) current_disputed: Option<usize>,
    pub(crate) unqualified_sat: usize,
    pub(crate) field_unqualified_sat: usize,
    pub(crate) field_scores: BTreeMap<String, FieldCategoryScore>,
    /// Every disputed instance in this benchmark, both sides named. Never
    /// summarized away — a withheld row is a claim about the field, not a
    /// rounding detail.
    pub(crate) disputed_rows: Vec<String>,
    /// Instances scored against published organiser truth rather than inferred
    /// witness credibility.
    pub(crate) truth_resolved_instances: Option<usize>,
    /// What the current rows would score if every field SAT were assumed valid.
    /// Reported whenever it differs from `current_raw_score`, so declining to
    /// score a disputed row can never quietly become a point gain.
    pub(crate) current_raw_score_assumed_valid: Option<i64>,
    /// Strongest participating tool's raw score, including current.
    pub(crate) field_best_raw_score: Option<i64>,
    /// `max(0, 100 * current_raw / field_best_raw)`.
    pub(crate) normalized: Option<f64>,
}

impl BenchProgress {
    fn delta(&self) -> i64 {
        self.current_solved as i64 - self.reference_solved as i64
    }
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct ProgressReport {
    pub(crate) benchmarks: Vec<BenchProgress>,
    pub(crate) total_current_solved: usize,
    pub(crate) total_reference_solved: usize,
    pub(crate) total_new_solves: usize,
    pub(crate) total_regressions: usize,
    pub(crate) total_verdict_flips: usize,
    /// Solved rows recorded above their official per-instance budget.
    pub(crate) total_over_budget: usize,
    /// Undecided rows recorded below 90% of their official budget.
    pub(crate) total_under_measured: usize,
    /// Solved rows recorded exactly at their official budget.
    pub(crate) total_at_budget_solves: usize,
    pub(crate) total_budget_checked: usize,
    pub(crate) total_budget_unchecked: usize,
    /// Sum of winner-relative scores for benchmarks represented in the field.
    pub(crate) normalized_total: Option<f64>,
    pub(crate) field_normalized_totals: BTreeMap<String, f64>,
    pub(crate) total_unqualified_sat: usize,
    pub(crate) total_field_unqualified_sat: usize,
    pub(crate) officially_scored_benchmarks: usize,
    /// Competition edition whose released track split was selected.
    pub(crate) score_year: Option<ScoreYear>,
    /// Scored categories with ZERO measured rows. Each is a guaranteed 0, and in
    /// practice the cause is a bank directory that was not supplied rather than a
    /// real absence of results.
    pub(crate) scored_categories_with_no_rows: Vec<String>,
    /// The separate leaderboard modeled by `normalized_total`.
    pub(crate) score_track: Option<ScoreTrack>,
    /// Present whenever `--official` requested a modeled raw-CSV scoreboard.
    pub(crate) modeled_score_caveat: Option<&'static str>,
    /// Applied only with the explicit 2026 field and original-budget corpus.
    pub(crate) field_timeout_normalization: Option<field_timeouts::FieldTimeoutNormalization>,
    /// Present when no `--budget-year` was supplied, so runtimes went unchecked.
    pub(crate) unchecked_budget_caveat: Option<&'static str>,
    /// Decided current rows withheld from scoring across all benchmarks.
    pub(crate) total_disputed: usize,
    /// Disputed instances found across all benchmarks (current rows or not).
    pub(crate) total_disputed_instances: usize,
    /// Provenance of the published organiser truth, when one was loaded.
    pub(crate) ground_truth_source: Option<String>,
    /// Instances in the loaded truth table, before matching.
    pub(crate) ground_truth_instances: usize,
    /// Instances actually scored against published truth.
    pub(crate) ground_truth_resolved_instances: usize,
    /// Organiser row identities that two instances share with DIFFERENT
    /// verdicts. These are dropped, never guessed, and are listed so the
    /// ambiguity is visible rather than silently resolved.
    pub(crate) ground_truth_ambiguous_keys: Vec<String>,
}

#[derive(Debug)]
struct ModeledOfficialScore {
    current_raw: i64,
    current_correct: usize,
    current_incorrect: usize,
    /// Decided current rows withheld from scoring as disputed.
    current_disputed: usize,
    current_unqualified_sat: usize,
    field_scores: BTreeMap<String, FieldCategoryScore>,
    /// One line per disputed instance in this benchmark, both sides named.
    disputes: Vec<String>,
    /// Instances in this benchmark settled by published organiser truth.
    truth_resolved: usize,
    field_best_raw: i64,
    normalized: f64,
    /// What `current` would score if every field SAT were taken at face value.
    current_raw_assumed_valid: i64,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct FieldCategoryScore {
    raw: i64,
    normalized: f64,
    unqualified_sat: usize,
}

/// Label used for the locally measured rows wherever a participant is named.
const CURRENT_LABEL: &str = "ny (current)";

/// How one instance's verdicts are scored, decided ONCE from every
/// participant's claim on that instance rather than row by row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstanceRuling {
    /// Per-run 2026 checks: only strict SAT convicts other participants.
    PerRun { strict_sat: bool },
    /// The organisers published a verdict for this instance. Match it for +10,
    /// contradict it for -150; no other evidence is consulted.
    Truth(Verdict),
    /// A credible SAT witness exists, so a contradicting UNSAT is wrong.
    WitnessCredible,
    /// Nothing convicts here: either no SAT was claimed at all, or the only SAT
    /// claim is uncontested. Every decided verdict earns +10.
    Uncontested,
    /// Exactly one participant claims SAT, at least one claims UNSAT, and no
    /// published truth settles it. Scoring either side would let one
    /// unreproduced witness set the category denominator, so NOBODY scores and
    /// the conflict is reported instead.
    Disputed,
}

/// Per-instance rulings for one benchmark, and what reaching them surfaced.
#[derive(Debug, Default)]
struct Adjudication {
    rulings: BTreeMap<SemanticKey, InstanceRuling>,
    per_run: BTreeMap<InstanceKey, bool>,
    /// One line per disputed instance, naming BOTH sides.
    disputes: Vec<String>,
    /// Instances settled by published organiser truth.
    truth_resolved: usize,
}

impl Adjudication {
    fn ruling(&self, key: &InstanceKey) -> InstanceRuling {
        if let Some(&strict_sat) = self.per_run.get(key) {
            return InstanceRuling::PerRun { strict_sat };
        }
        self.rulings
            .get(&key.semantic_key())
            .copied()
            .unwrap_or(InstanceRuling::Uncontested)
    }
}

/// Decide, for every instance of one benchmark, what its verdicts are worth.
///
/// The claim census spans the locally measured rows AND every field tool, so a
/// witness is judged by the whole field rather than by whoever happens to be
/// scored at the moment. That matters: the ruling is shared by `current` and by
/// each competitor when the winner-relative denominator is computed, which is
/// what stops one tool from being scored under a different model than the tool
/// it is normalized against.
fn adjudicate(
    benchmark: &str,
    current: &ResultRows,
    field: &OfficialField,
    scored_instances: &BTreeSet<InstanceKey>,
    truth: Option<&GroundTruth>,
    witnesses: Witnesses,
) -> Adjudication {
    // Only instances the field actually scores can be adjudicated; a row that
    // exists solely in the local bank earns nothing either way, and counting it
    // would inflate the reported truth coverage.
    let scored = scored_instances
        .iter()
        .map(InstanceKey::semantic_key)
        .collect::<BTreeSet<_>>();
    type Claims = (BTreeSet<String>, BTreeSet<String>, InstanceKey);
    let mut claims = BTreeMap::<SemanticKey, Claims>::new();
    let mut participants: Vec<(&str, &ResultRows)> = vec![(CURRENT_LABEL, current)];
    participants.extend(field.iter().map(|(tool, rows)| (tool.as_str(), rows)));
    for (tool, rows) in participants {
        for (key, entry) in rows {
            if key.benchmark != benchmark || !scored.contains(&key.semantic_key()) {
                continue;
            }
            let slot = claims
                .entry(key.semantic_key())
                .or_insert_with(|| (BTreeSet::new(), BTreeSet::new(), key.clone()));
            match entry.verdict {
                Verdict::Sat => {
                    slot.0.insert(tool.to_string());
                }
                Verdict::Unsat => {
                    slot.1.insert(tool.to_string());
                }
                Verdict::Undecided => {}
            }
        }
    }

    let mut out = Adjudication::default();
    for (semantic, (sats, unsats, key)) in claims {
        let ruling = match witnesses {
            Witnesses::Unvalidated => InstanceRuling::Uncontested,
            Witnesses::AssumedValid => {
                if sats.is_empty() {
                    InstanceRuling::Uncontested
                } else {
                    InstanceRuling::WitnessCredible
                }
            }
            Witnesses::Adjudicated => match truth.and_then(|truth| truth.get(&key)) {
                Some(verdict) => {
                    out.truth_resolved += 1;
                    InstanceRuling::Truth(verdict)
                }
                None if sats.len() >= 2 => InstanceRuling::WitnessCredible,
                None if sats.len() == 1 && !unsats.is_empty() => {
                    out.disputes.push(format!(
                        "{} [SAT claimed ONLY by {}; UNSAT held by {}; no published truth \
                         settles it, so NOBODY scores this row and it sets no denominator]",
                        key.display_name(),
                        sats.iter().cloned().collect::<Vec<_>>().join(", "),
                        unsats.iter().cloned().collect::<Vec<_>>().join(", "),
                    ));
                    InstanceRuling::Disputed
                }
                None => InstanceRuling::Uncontested,
            },
        };
        out.rulings.insert(semantic, ruling);
    }
    out
}

fn adjudicate_per_run(
    benchmark: &str,
    current: &ResultRows,
    field: &OfficialField,
    scored_instances: &BTreeSet<InstanceKey>,
    over_budget: &BTreeSet<InstanceKey>,
) -> Adjudication {
    let mut out = Adjudication::default();
    for key in scored_instances
        .iter()
        .filter(|key| key.benchmark == benchmark)
    {
        let strict_sat = current.get(key).is_some_and(|entry| {
            !over_budget.contains(key)
                && entry.verdict == Verdict::Sat
                && entry.counterexample == Some(CounterexampleCheck::Strict)
        }) || field.values().any(|rows| {
            rows.get(key).is_some_and(|entry| {
                entry.verdict == Verdict::Sat
                    && entry.counterexample == Some(CounterexampleCheck::Strict)
            })
        });
        out.per_run.insert(key.clone(), strict_sat);
    }
    out
}

#[derive(Debug, Default, Clone, Copy)]
struct RowScore {
    raw: i64,
    correct: usize,
    incorrect: usize,
    /// Decided rows withheld from scoring because the instance is disputed.
    disputed: usize,
    unqualified_sat: usize,
}

fn score_rows(
    rows: &ResultRows,
    scored_instances: &BTreeSet<InstanceKey>,
    adjudication: &Adjudication,
    over_budget: &BTreeSet<InstanceKey>,
) -> RowScore {
    let mut score = RowScore::default();
    for (key, entry) in rows {
        if !scored_instances.contains(key) {
            continue;
        }
        // Past its own timeout the harness reports a timeout, worth zero. It is
        // not a wrong answer either, so an over-budget UNSAT contradicted by a
        // witness must not draw the -150 penalty here.
        if over_budget.contains(key) {
            continue;
        }
        if entry.verdict == Verdict::Undecided {
            continue;
        }
        match adjudication.ruling(key) {
            InstanceRuling::PerRun { strict_sat } => match entry.verdict {
                Verdict::Sat => match entry.counterexample {
                    Some(CounterexampleCheck::Strict | CounterexampleCheck::Tolerant) => {
                        score.raw += POINTS_CORRECT;
                        score.correct += 1;
                    }
                    Some(
                        CounterexampleCheck::Missing
                        | CounterexampleCheck::Malformed
                        | CounterexampleCheck::Invalid,
                    ) => {
                        score.raw += PENALTY_INCORRECT;
                        score.incorrect += 1;
                    }
                    Some(CounterexampleCheck::Unsupported) | None => score.unqualified_sat += 1,
                },
                Verdict::Unsat => {
                    if strict_sat {
                        score.raw += PENALTY_INCORRECT;
                        score.incorrect += 1;
                    } else {
                        score.raw += POINTS_CORRECT;
                        score.correct += 1;
                    }
                }
                Verdict::Undecided => {}
            },
            InstanceRuling::Truth(truth) => {
                if entry.verdict == truth {
                    score.raw += POINTS_CORRECT;
                    score.correct += 1;
                } else {
                    score.raw += PENALTY_INCORRECT;
                    score.incorrect += 1;
                }
            }
            InstanceRuling::WitnessCredible => match entry.verdict {
                Verdict::Sat => {
                    score.raw += POINTS_CORRECT;
                    score.correct += 1;
                }
                _ => {
                    score.raw += PENALTY_INCORRECT;
                    score.incorrect += 1;
                }
            },
            InstanceRuling::Uncontested => {
                score.raw += POINTS_CORRECT;
                score.correct += 1;
            }
            InstanceRuling::Disputed => score.disputed += 1,
        }
    }
    score
}

/// Outcome of checking one benchmark's solved rows against official budgets.
#[derive(Debug, Default)]
struct BudgetAudit {
    /// Human-readable violations, one per over-budget row.
    rows: Vec<String>,
    /// Identities to exclude from scoring.
    keys: BTreeSet<InstanceKey>,
    /// Solved rows that had a budget to compare against.
    checked: usize,
    /// Solved rows with no budget in the corpora — unchecked, not passing.
    unchecked: usize,
    /// Undecided rows recorded below `UNDER_MEASURED_FRACTION` of their own
    /// budget — invalid measurements, never capability limits.
    under_measured: Vec<String>,
    /// Solved rows recorded exactly at their budget: legal at the wire, but
    /// also what a hand-edited ceiling looks like, so they are surfaced.
    at_budget: Vec<String>,
}

/// An undecided row is an invalid measurement when it was given less than this
/// fraction of its official budget. The slack keeps watchdog-grace rounding
/// and sub-second harness overhead from manufacturing candidates, matching the
/// 90% convention in `BUDGET_AUDIT_CORRECTION_2026-07-29.md`.
const UNDER_MEASURED_FRACTION: f64 = 0.90;

/// Compare each row's recorded runtime against its own official timeout.
///
/// Solved rows past their budget are phantom points and are excluded from
/// scoring. Undecided rows recorded well BELOW their budget are the inverse
/// defect: the row never got the time the competition gives it (the class the
/// FORMER `NY_MEASURE_CAP=120` default in `scripts/measure_ny_scorecard.sh`
/// produced against 900s cgan budgets, until `866e21946` made no-cap the
/// default), so it must not be read as a capability limit. The audit stays,
/// because the BANK still holds those rows even though the generator no
/// longer makes new ones.
/// Neither direction is a formality — one invents points, the other buries
/// them.
fn audit_budgets(current: &ResultRows, budgets: &BudgetTable, benchmark: &str) -> BudgetAudit {
    let mut audit = BudgetAudit::default();
    for (key, entry) in current {
        if key.benchmark != benchmark {
            continue;
        }
        let Some(&budget) = budgets.get(&key.semantic_key()) else {
            if entry.verdict.solved() {
                audit.unchecked += 1;
            }
            continue;
        };
        let budget_secs = budget as f64;
        if entry.verdict.solved() {
            audit.checked += 1;
            if entry.seconds > budget_secs {
                audit.rows.push(format!(
                    "{} [{} in {:.1}s vs {}s budget, {:.2}x over, at {}]",
                    key.display_name(),
                    entry.verdict.as_str(),
                    entry.seconds,
                    budget,
                    entry.seconds / budget_secs,
                    entry.source
                ));
                audit.keys.insert(key.clone());
            } else if entry.seconds == budget_secs {
                audit.at_budget.push(format!(
                    "{} [{} at exactly its {}s budget; run tag: {}]",
                    key.display_name(),
                    entry.verdict.as_str(),
                    budget,
                    entry.tag.as_deref().unwrap_or("NONE")
                ));
            }
        } else if entry.seconds < budget_secs * UNDER_MEASURED_FRACTION {
            audit.under_measured.push(format!(
                "{} [{} after {:.1}s vs {}s budget — measured at {:.0}% of its official time, at {}]",
                key.display_name(),
                entry.verdict.as_str(),
                entry.seconds,
                budget,
                100.0 * entry.seconds / budget_secs,
                entry.source
            ));
        }
    }
    audit
}

fn modeled_official_score(
    benchmark: &str,
    current: &ResultRows,
    field: &OfficialField,
    original_cohort: Option<&BTreeSet<InstanceKey>>,
    year: ScoreYear,
    track: ScoreTrack,
    over_budget: &BTreeSet<InstanceKey>,
    witnesses: Witnesses,
    truth: Option<&GroundTruth>,
) -> Option<ModeledOfficialScore> {
    if !track.includes(year, benchmark) {
        return None;
    }
    let scored_instances = match original_cohort {
        Some(keys) => keys
            .iter()
            .filter(|key| key.benchmark == benchmark)
            .cloned()
            .collect(),
        None => field
            .values()
            .flat_map(|rows| rows.keys())
            .filter(|key| key.benchmark == benchmark)
            .cloned()
            .collect::<BTreeSet<_>>(),
    };
    if scored_instances.is_empty() {
        return None;
    }
    // Removing inactive incumbent participants must not remove their original
    // eligible rows: an in-budget NY decision can be the sole new participant.
    // A row outside that supplied cohort cannot invent a scored category.
    if original_cohort.is_some()
        && !field
            .values()
            .any(|rows| rows.keys().any(|key| key.benchmark == benchmark))
        && !current.iter().any(|(key, entry)| {
            scored_instances.contains(key) && entry.verdict.solved() && !over_budget.contains(key)
        })
    {
        return None;
    }

    // One ruling per instance, shared by `current` and by every competitor, so
    // the denominator is computed under the SAME model as the numerator.
    let adjudication = if year == ScoreYear::Y2026 && witnesses == Witnesses::Adjudicated {
        adjudicate_per_run(benchmark, current, field, &scored_instances, over_budget)
    } else {
        adjudicate(
            benchmark,
            current,
            field,
            &scored_instances,
            truth,
            witnesses,
        )
    };
    let current_score = score_rows(current, &scored_instances, &adjudication, over_budget);
    // Field verdicts include any explicitly requested 2026 timeout normalization.
    let no_exclusions = BTreeSet::new();
    let mut field_best_raw = current_score.raw;
    let mut field_scores = BTreeMap::new();
    for (tool, rows) in field {
        if !rows.keys().any(|key| key.benchmark == benchmark) {
            continue;
        }
        let score = score_rows(rows, &scored_instances, &adjudication, &no_exclusions);
        field_best_raw = field_best_raw.max(score.raw);
        field_scores.insert(
            tool.clone(),
            FieldCategoryScore {
                raw: score.raw,
                unqualified_sat: score.unqualified_sat,
                ..Default::default()
            },
        );
    }
    for score in field_scores.values_mut() {
        if field_best_raw > 0 {
            score.normalized = (100.0 * score.raw as f64 / field_best_raw as f64).max(0.0);
        }
    }
    let normalized = if field_best_raw > 0 {
        (100.0 * current_score.raw as f64 / field_best_raw as f64).max(0.0)
    } else {
        0.0
    };

    // Also report the score obtained by accepting every SAT claim, including
    // NY's. This can increase the score when a NY witness lacks adjudication.
    let assumed_valid = adjudicate(
        benchmark,
        current,
        field,
        &scored_instances,
        None,
        Witnesses::AssumedValid,
    );
    let assumed_valid_raw = score_rows(current, &scored_instances, &assumed_valid, over_budget).raw;

    Some(ModeledOfficialScore {
        current_raw: current_score.raw,
        current_correct: current_score.correct,
        current_incorrect: current_score.incorrect,
        current_disputed: current_score.disputed,
        current_unqualified_sat: current_score.unqualified_sat,
        field_scores,
        disputes: adjudication.disputes,
        truth_resolved: adjudication.truth_resolved,
        field_best_raw,
        normalized,
        current_raw_assumed_valid: assumed_valid_raw,
    })
}

pub(crate) struct ScoreOptions {
    pub(crate) results: Vec<PathBuf>,
    pub(crate) baseline: Option<PathBuf>,
    pub(crate) official: Option<PathBuf>,
    pub(crate) year: Option<ScoreYear>,
    pub(crate) track: ScoreTrack,
    pub(crate) json: bool,
    pub(crate) show_rows: usize,
    /// Corpus years whose `instances.csv` budgets are enforced against the
    /// recorded runtimes. Empty leaves runtimes unchecked, and says so.
    pub(crate) budget_years: Vec<u32>,
    /// How SAT rows are treated.
    pub(crate) witnesses: Witnesses,
    /// Historical 2025 truth summary: `SCORING/results.txt` or its corpus root.
    /// Found automatically under `--official`; sanitized corpora need an
    /// explicit path to the original summary.
    pub(crate) ground_truth: Option<PathBuf>,
    pub(crate) ny_ce_checks: Option<PathBuf>,
    pub(crate) ny_ce_source: Option<String>,
}

fn validate_year_hints(rows: &ResultRows, selected: ScoreYear, corpus: &str) -> Result<()> {
    if let Some((benchmark, hinted, source)) = rows.iter().find_map(|(key, entry)| {
        entry
            .year_hint
            .filter(|hint| *hint != selected)
            .map(|hint| (key.benchmark.as_str(), hint, entry.source.as_str()))
    }) {
        bail!(
            "{corpus} contains a {}-prefixed category {:?} at {}, but --year {} was selected",
            hinted.number(),
            benchmark,
            source,
            selected.number()
        );
    }
    Ok(())
}

pub(crate) fn run_score(opts: &ScoreOptions) -> Result<ProgressReport> {
    if opts.results.is_empty() {
        bail!("at least one --results path is required");
    }
    let score_year = match (opts.official.as_ref(), opts.year) {
        (Some(_), Some(year)) => Some(year),
        (Some(_), None) => {
            bail!(
                "--year <2025|2026> is required with --official; category track membership \
                 changed between editions and is never inferred from ambiguous category ids"
            )
        }
        (None, year) => year,
    };
    let mut current = BTreeMap::new();
    for path in &opts.results {
        merge_compatible(&mut current, load(path)?)?;
    }
    let reference = match &opts.baseline {
        Some(path) => load(path)?,
        None => BTreeMap::new(),
    };
    let mut official = opts
        .official
        .as_deref()
        .map(load_official_field)
        .transpose()?;
    if opts.ny_ce_checks.is_some() != opts.ny_ce_source.is_some() {
        bail!("--ny-ce-checks and --ny-ce-source must be supplied together");
    }
    if let Some(source) = &opts.ny_ce_source {
        if !hex_digest(source, 40)
            || score_year != Some(ScoreYear::Y2026)
            || opts.witnesses != Witnesses::Adjudicated
            || official.is_none()
        {
            bail!("NY counterexample checks require --official --year 2026 --witnesses adjudicated and an exact --ny-ce-source commit");
        }
    }
    if score_year == Some(ScoreYear::Y2026) && opts.witnesses == Witnesses::Adjudicated {
        if let (Some(root), Some(field)) = (&opts.official, &mut official) {
            load_published_counterexample_checks(root, field)?;
        }
        if let (Some(directory), Some(source)) = (&opts.ny_ce_checks, &opts.ny_ce_source) {
            load_ny_counterexample_checks(directory, source, &mut current)?;
        }
    }
    // Published organiser truth, when one is reachable. An explicit path wins;
    // otherwise the released corpus is asked for its own scoring log, which the
    // field loader already skips when collecting result CSVs.
    let ground_truth = match (&opts.ground_truth, &opts.official) {
        (Some(path), _) => Some(load_ground_truth(path)?),
        (None, Some(root)) => {
            let log = root.join(GROUND_TRUTH_RELATIVE_PATH);
            if log.is_file() {
                Some(load_ground_truth(&log)?)
            } else {
                None
            }
        }
        (None, None) => None,
    };
    if let Some(year) = score_year {
        validate_year_hints(&current, year, "current results")?;
        validate_year_hints(&reference, year, "baseline results")?;
        if let Some(field) = &official {
            for (tool, rows) in field {
                validate_year_hints(rows, year, &format!("official results for tool {tool:?}"))?;
            }
        }
    }
    if let (Some(field), Some(year)) = (&official, score_year) {
        let unknown = field
            .values()
            .flat_map(|rows| rows.keys())
            .map(|key| key.benchmark.as_str())
            .filter(|benchmark| !known_benchmark(year, benchmark))
            .collect::<BTreeSet<_>>();
        if !unknown.is_empty() {
            bail!(
                "official field contains categories not present in the pinned VNN-COMP {} \
                 regular/extended track lists: {}",
                year.number(),
                unknown.into_iter().collect::<Vec<_>>().join(", ")
            );
        }
    }
    if current.is_empty() && reference.is_empty() && official.is_none() {
        bail!("no result rows parsed from the supplied current/reference paths");
    }

    let budgets = if opts.budget_years.is_empty() {
        None
    } else {
        let mut years = opts.budget_years.clone();
        years.sort_unstable();
        years.dedup();
        Some(load_budgets(&years)?)
    };
    let original_field_cohort = official.as_ref().map(|field| {
        field
            .values()
            .flat_map(|rows| rows.keys())
            .cloned()
            .collect::<BTreeSet<_>>()
    });
    let field_timeout_normalization =
        if score_year == Some(ScoreYear::Y2026) && opts.budget_years.contains(&2026) {
            match (&opts.official, &mut official, &budgets) {
                (Some(root), Some(field), Some(table)) => {
                    Some(field_timeouts::normalize(root, field, table)?)
                }
                _ => None,
            }
        } else {
            None
        };

    let all_keys = current
        .keys()
        .chain(reference.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut benchmarks = all_keys
        .iter()
        .map(|key| key.benchmark.clone())
        .collect::<BTreeSet<_>>();
    if let Some(field) = &official {
        benchmarks.extend(
            field
                .values()
                .flat_map(|rows| rows.keys())
                .map(|key| key.benchmark.clone()),
        );
    }

    let mut report = ProgressReport {
        modeled_score_caveat: official.as_ref().map(|_| match score_year {
            Some(ScoreYear::Y2026) if field_timeout_normalization.is_some() => {
                NORMALIZED_2026_SCORE_CAVEAT
            }
            Some(ScoreYear::Y2026) => MODELED_2026_SCORE_CAVEAT,
            _ => MODELED_SCORE_CAVEAT,
        }),
        field_timeout_normalization,
        score_year: official.as_ref().and(score_year),
        score_track: official.as_ref().map(|_| opts.track),
        unchecked_budget_caveat: budgets
            .as_ref()
            .map_or(Some(UNCHECKED_BUDGET_CAVEAT), |_| None),
        ground_truth_source: ground_truth.as_ref().map(|truth| truth.source.clone()),
        ground_truth_instances: ground_truth.as_ref().map_or(0, GroundTruth::instance_count),
        ground_truth_ambiguous_keys: ground_truth
            .as_ref()
            .map(|truth| truth.ambiguous.iter().cloned().collect())
            .unwrap_or_default(),
        ..Default::default()
    };
    let mut normalized_total = 0.0;
    for benchmark in benchmarks {
        let mut progress = BenchProgress {
            benchmark: benchmark.clone(),
            ..Default::default()
        };
        let over_budget_keys = match budgets.as_ref() {
            Some(table) => {
                let audit = audit_budgets(&current, table, &benchmark);
                progress.over_budget = audit.rows;
                progress.under_measured = audit.under_measured;
                progress.at_budget_solves = audit.at_budget;
                progress.budget_checked = audit.checked;
                progress.budget_unchecked = audit.unchecked;
                audit.keys
            }
            None => BTreeSet::new(),
        };
        for key in all_keys.iter().filter(|key| key.benchmark == benchmark) {
            progress.rows_compared += 1;
            let current_verdict = current
                .get(key)
                .map_or(Verdict::Undecided, |entry| entry.verdict);
            let reference_verdict = reference
                .get(key)
                .map_or(Verdict::Undecided, |entry| entry.verdict);
            if current_verdict.solved() {
                progress.current_solved += 1;
            }
            if reference_verdict.solved() {
                progress.reference_solved += 1;
            }
            match (reference_verdict.solved(), current_verdict.solved()) {
                (false, true) => progress.new_solves.push(key.display_name()),
                (true, false) => progress.regressions.push(key.display_name()),
                (true, true) if reference_verdict != current_verdict => {
                    progress.verdict_flips.push(key.display_name());
                }
                _ => {}
            }
        }

        if let Some(score) = official.as_ref().and_then(|field| {
            modeled_official_score(
                &benchmark,
                &current,
                field,
                report
                    .field_timeout_normalization
                    .as_ref()
                    .and(original_field_cohort.as_ref()),
                score_year.expect("official score year validated above"),
                opts.track,
                &over_budget_keys,
                opts.witnesses,
                ground_truth.as_ref(),
            )
        }) {
            progress.current_raw_score = Some(score.current_raw);
            progress.current_correct = Some(score.current_correct);
            progress.current_incorrect = Some(score.current_incorrect);
            progress.current_disputed = Some(score.current_disputed);
            progress.unqualified_sat = score.current_unqualified_sat;
            report.total_unqualified_sat += score.current_unqualified_sat;
            for (tool, value) in &score.field_scores {
                progress.field_unqualified_sat += value.unqualified_sat;
                *report
                    .field_normalized_totals
                    .entry(tool.clone())
                    .or_default() += value.normalized;
            }
            report.total_field_unqualified_sat += progress.field_unqualified_sat;
            progress.field_scores = score.field_scores;
            progress.truth_resolved_instances = Some(score.truth_resolved);
            progress.current_raw_score_assumed_valid = Some(score.current_raw_assumed_valid);
            progress.disputed_rows = score.disputes;
            progress.field_best_raw_score = Some(score.field_best_raw);
            progress.normalized = Some(score.normalized);
            report.total_disputed += score.current_disputed;
            report.total_disputed_instances += progress.disputed_rows.len();
            report.ground_truth_resolved_instances += score.truth_resolved;
            normalized_total += score.normalized;
            report.officially_scored_benchmarks += 1;
            if progress.rows_compared == 0 {
                report.scored_categories_with_no_rows.push(benchmark);
            }
        }

        report.total_current_solved += progress.current_solved;
        report.total_reference_solved += progress.reference_solved;
        report.total_new_solves += progress.new_solves.len();
        report.total_regressions += progress.regressions.len();
        report.total_verdict_flips += progress.verdict_flips.len();
        report.total_over_budget += progress.over_budget.len();
        report.total_under_measured += progress.under_measured.len();
        report.total_at_budget_solves += progress.at_budget_solves.len();
        report.total_budget_checked += progress.budget_checked;
        report.total_budget_unchecked += progress.budget_unchecked;
        report.benchmarks.push(progress);
    }
    if report.officially_scored_benchmarks > 0 {
        report.normalized_total = Some(normalized_total);
        report
            .field_normalized_totals
            .insert(CURRENT_LABEL.into(), normalized_total);
    } else if official.is_some() {
        bail!(
            "official field contains no categories in the selected VNN-COMP {} {} track",
            score_year
                .expect("official score year validated above")
                .number(),
            opts.track.label(),
        );
    }

    if opts.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&report, opts);
    }
    Ok(report)
}

fn print_report(report: &ProgressReport, opts: &ScoreOptions) {
    let has_reference = opts.baseline.is_some();
    println!();
    if has_reference {
        println!(
            "{:<28}{:>8}{:>10}{:>7}{:>6}{:>6}{:>8}{:>7}{:>8}",
            "benchmark", "current", "reference", "delta", "new", "regr", "raw", "wrong", "norm"
        );
    } else {
        println!(
            "{:<34}{:>8}{:>7}{:>8}{:>7}{:>8}",
            "benchmark", "current", "rows", "raw", "wrong", "norm"
        );
    }
    for benchmark in &report.benchmarks {
        let raw = benchmark
            .current_raw_score
            .map_or_else(|| "-".to_string(), |value| value.to_string());
        let wrong = benchmark
            .current_incorrect
            .map_or_else(|| "-".to_string(), |value| value.to_string());
        let normalized = benchmark
            .normalized
            .map_or_else(|| "-".to_string(), |value| format!("{value:.1}"));
        if has_reference {
            println!(
                "{:<28}{:>8}{:>10}{:>+7}{:>6}{:>6}{:>8}{:>7}{:>8}",
                benchmark.benchmark,
                benchmark.current_solved,
                benchmark.reference_solved,
                benchmark.delta(),
                benchmark.new_solves.len(),
                benchmark.regressions.len(),
                raw,
                wrong,
                normalized
            );
        } else {
            println!(
                "{:<34}{:>8}{:>7}{:>8}{:>7}{:>8}",
                benchmark.benchmark,
                benchmark.current_solved,
                benchmark.rows_compared,
                raw,
                wrong,
                normalized
            );
        }
    }
    println!("{}", "-".repeat(90));
    if has_reference {
        println!(
            "TOTAL current {} (reference {}, delta {:+})   NEW SOLVES {}   REGRESSIONS {}",
            report.total_current_solved,
            report.total_reference_solved,
            report.total_current_solved as i64 - report.total_reference_solved as i64,
            report.total_new_solves,
            report.total_regressions
        );
    } else {
        println!("TOTAL current solved {}", report.total_current_solved);
    }
    match report.normalized_total {
        Some(total) => {
            let year = report
                .score_year
                .map_or_else(|| "unknown".to_string(), |year| year.number().to_string());
            let track = report
                .score_track
                .map_or("unknown", ScoreTrack::label)
                .to_ascii_uppercase();
            println!(
                "MODELED VNN-COMP {year} SUPPLIED-FIELD {track}-CATEGORY TOTAL {total:.1} across \
                 {} benchmark(s)",
                report.officially_scored_benchmarks,
            );
            println!("witness model: {}", opts.witnesses.label());
            if report.score_year == Some(ScoreYear::Y2026)
                && opts.witnesses == Witnesses::Adjudicated
            {
                println!("Per-run checks: strict SAT convicts UNSAT; tolerant SAT earns only its own credit.");
                println!("Unqualified SAT rows: NY={}, field={}; these receive zero and cannot convict other tools.",
                    report.total_unqualified_sat, report.total_field_unqualified_sat);
                let mut totals = report.field_normalized_totals.iter().collect::<Vec<_>>();
                totals.sort_by(|a, b| b.1.total_cmp(a.1).then_with(|| a.0.cmp(b.0)));
                for (tool, score) in totals {
                    println!("  {tool}: {score:.6}");
                }
            } else {
                match &report.ground_truth_source {
                    Some(source) => {
                        println!(
                            "ground truth: {} instance(s) from {source}; {} scored against it",
                            report.ground_truth_instances, report.ground_truth_resolved_instances
                        );
                        println!(
                            "  note: a raw results CSV cannot express two organiser rules (a \
                         rejected counterexample is charged -150; a counterexample correct only \
                         up to tolerance still earns +10), so modeled totals can differ \
                         slightly from the published board."
                        );
                    }
                    None if opts.witnesses == Witnesses::Adjudicated => println!(
                    "ground truth: NONE SUPPLIED — no organiser adjudication was reachable, so \
                     every instance fell back to corroboration. Pass --ground-truth <corpus> \
                     (the released tree's {}); a sanitized field copy usually drops it. An \
                     absent truth table is a WEAKER model, not a clean one.",
                    GROUND_TRUTH_RELATIVE_PATH
                ),
                    None => {}
                }
            }
            if !report.ground_truth_ambiguous_keys.is_empty() {
                println!(
                    "  {} organiser row id(s) are shared by instances with different published \
                     verdicts and were dropped rather than guessed: {}",
                    report.ground_truth_ambiguous_keys.len(),
                    report.ground_truth_ambiguous_keys.join(", ")
                );
            }
            if opts.witnesses == Witnesses::Unvalidated {
                println!(
                    "  unvalidated projection: every decided claim earns credit without a witness check."
                );
            }
            println!(
                "{}; this is not an organizer-validated scoreboard.",
                report.modeled_score_caveat.unwrap_or(MODELED_SCORE_CAVEAT)
            );
        }
        None => println!(
            "MODELED SUPPLIED-FIELD TOTAL: not computed — pass --official <results-dir> with a \
             matching competition field. Solve counts above remain exact."
        ),
    }

    if !report.scored_categories_with_no_rows.is_empty() {
        println!(
            "\n*** {} SCORED CATEGOR{} HAVE NO MEASURED ROWS and score zero: {}. This \
             usually means a results directory was not passed; pass every bank with \
             repeated --results before reading a total. ***",
            report.scored_categories_with_no_rows.len(),
            if report.scored_categories_with_no_rows.len() == 1 {
                "Y"
            } else {
                "IES"
            },
            report.scored_categories_with_no_rows.join(", ")
        );
    }

    match &report.unchecked_budget_caveat {
        Some(caveat) => println!("BUDGET AUDIT: not run — {caveat}."),
        None => println!(
            "BUDGET AUDIT: {} solved row(s) checked against official per-instance timeouts, \
             {} over budget, {} with no budget in the supplied corpora (unchecked, NOT passing); \
             {} undecided row(s) measured below 90% of their budget (INVALID as capability \
             limits), {} solve(s) recorded exactly at their budget.",
            report.total_budget_checked,
            report.total_over_budget,
            report.total_budget_unchecked,
            report.total_under_measured,
            report.total_at_budget_solves
        ),
    }
    if let Some(normalization) = &report.field_timeout_normalization {
        println!(
            "FIELD TIMEOUT NORMALIZATION: {} tool aggregate(s), {} late decision(s) changed to \
             timeout, {} all-undecided tool categor{} removed; raw runtimes retained.",
            normalization.tools.len(),
            normalization.forced_timeouts,
            normalization.removed_categories,
            if normalization.removed_categories == 1 {
                "y"
            } else {
                "ies"
            },
        );
    }

    let show = |label: &str, pick: fn(&BenchProgress) -> &[String]| {
        let mut any = false;
        for benchmark in &report.benchmarks {
            let rows = pick(benchmark);
            if rows.is_empty() {
                continue;
            }
            if !any {
                println!("\n{label}");
                any = true;
            }
            for row in rows.iter().take(opts.show_rows) {
                println!("  {}: {}", benchmark.benchmark, row);
            }
            if rows.len() > opts.show_rows {
                println!(
                    "  {}: ... and {} more",
                    benchmark.benchmark,
                    rows.len() - opts.show_rows
                );
            }
        }
    };
    show("NEW SOLVES:", |benchmark| &benchmark.new_solves);
    show(
        "REGRESSIONS (solved in reference, not current):",
        |benchmark| &benchmark.regressions,
    );

    if report.total_over_budget > 0 {
        println!(
            "\n*** PHANTOM POINTS: {} solved row(s) were recorded ABOVE their own official \
             per-instance timeout. VNN-COMP budgets are per instance, so these would be \
             timeouts worth zero and are EXCLUDED from the modeled score above. Re-measure \
             them at the budget from instances.csv before treating any as progress. ***",
            report.total_over_budget
        );
        for benchmark in &report.benchmarks {
            for row in benchmark.over_budget.iter().take(opts.show_rows) {
                println!("  {}: {}", benchmark.benchmark, row);
            }
            if benchmark.over_budget.len() > opts.show_rows {
                println!(
                    "  {}: ... and {} more",
                    benchmark.benchmark,
                    benchmark.over_budget.len() - opts.show_rows
                );
            }
        }
    }

    if report.total_under_measured > 0 {
        println!(
            "\n*** INVALID MEASUREMENTS: {} undecided row(s) were recorded below 90% of \
             their own official per-instance timeout, so they say nothing about \
             capability. Re-measure each at its instances.csv budget (`ny benchmarks \
             run` and scripts/measure_ny_scorecard.sh both do so by default). ***",
            report.total_under_measured
        );
        for benchmark in &report.benchmarks {
            for row in benchmark.under_measured.iter().take(opts.show_rows) {
                println!("  {}: {}", benchmark.benchmark, row);
            }
            if benchmark.under_measured.len() > opts.show_rows {
                println!(
                    "  {}: ... and {} more",
                    benchmark.benchmark,
                    benchmark.under_measured.len() - opts.show_rows
                );
            }
        }
    }

    if report.total_at_budget_solves > 0 {
        println!(
            "\nAT-BUDGET SOLVES: {} solved row(s) recorded exactly at their official \
             timeout. A real solve can finish at the wire, but a runtime equal to the \
             ceiling is also what a hand-edited row looks like — verify each row's run \
             tag and evidence before trusting it.",
            report.total_at_budget_solves
        );
        for benchmark in &report.benchmarks {
            for row in benchmark.at_budget_solves.iter().take(opts.show_rows) {
                println!("  {}: {}", benchmark.benchmark, row);
            }
            if benchmark.at_budget_solves.len() > opts.show_rows {
                println!(
                    "  {}: ... and {} more",
                    benchmark.benchmark,
                    benchmark.at_budget_solves.len() - opts.show_rows
                );
            }
        }
    }

    if report.total_disputed_instances > 0 {
        println!(
            "\n*** DISPUTED WITNESSES: {} instance(s) carry a SAT claimed by exactly ONE \
             participant and contradicted by others, with no published organiser truth to \
             settle them. Under --witnesses assumed-valid each of these would convict every \
             tool that disagrees at -150 and would set this category's winner-relative \
             DENOMINATOR off a single unreproduced witness. They are withheld from scoring \
             instead — for BOTH sides — and listed here. {} decided current row(s) are among \
             them. Adjudicate them against evidence; do not read a withheld row as agreement. \
             ***",
            report.total_disputed_instances, report.total_disputed
        );
        for benchmark in &report.benchmarks {
            for row in benchmark.disputed_rows.iter().take(opts.show_rows) {
                println!("  {}: {}", benchmark.benchmark, row);
            }
            if benchmark.disputed_rows.len() > opts.show_rows {
                println!(
                    "  {}: ... and {} more",
                    benchmark.benchmark,
                    benchmark.disputed_rows.len() - opts.show_rows
                );
            }
        }
        let mut moved = report
            .benchmarks
            .iter()
            .filter_map(|benchmark| {
                let raw = benchmark.current_raw_score?;
                let assumed_valid = benchmark.current_raw_score_assumed_valid?;
                (raw != assumed_valid).then(|| {
                    format!(
                        "{} {raw} (assumed-valid: {assumed_valid})",
                        benchmark.benchmark
                    )
                })
            })
            .peekable();
        if moved.peek().is_some() {
            println!(
                "  ASSUMED-VALID COMPARISON — current raw under this model vs. \
                 accepting every SAT claim: {}",
                moved.collect::<Vec<_>>().join("; ")
            );
        }
    }

    if report.total_verdict_flips > 0 {
        println!(
            "\n*** SOUNDNESS ALARM: {} row(s) flipped sat<->unsat between the reference \
             and current run. The reference is not treated as truth; investigate the \
             underlying evidence before trusting either verdict. ***",
            report.total_verdict_flips
        );
        for benchmark in &report.benchmarks {
            for row in &benchmark.verdict_flips {
                println!("  {}: {}", benchmark.benchmark, row);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_file(dir: &TempDir, relative: &str, body: &str) -> PathBuf {
        let path = dir.path().join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create parent");
        }
        std::fs::write(&path, body).expect("write fixture");
        path
    }

    fn opts(results: PathBuf, baseline: Option<PathBuf>) -> ScoreOptions {
        ScoreOptions {
            results: vec![results],
            baseline,
            official: None,
            year: None,
            track: ScoreTrack::Regular,
            json: false,
            show_rows: 5,
            budget_years: Vec::new(),
            witnesses: Witnesses::AssumedValid,
            ground_truth: None,
            ny_ce_checks: None,
            ny_ce_source: None,
        }
    }

    /// Give every instance in `rows` the same ruling, for the score_rows unit
    /// tests that predate per-instance adjudication.
    fn ruling_for_all(rows: &ResultRows, ruling: InstanceRuling) -> Adjudication {
        Adjudication {
            rulings: rows
                .keys()
                .map(|key| (key.semantic_key(), ruling))
                .collect(),
            ..Default::default()
        }
    }

    fn publish_check(
        root: &TempDir,
        tool: &str,
        row: &str,
        outcome: CounterexampleCheck,
    ) -> PathBuf {
        let fields = parse_csv_line(row.trim_end_matches(['\r', '\n'])).unwrap();
        let (classification, strict, credit) = outcome.metadata();
        let result = match outcome {
            CounterexampleCheck::Strict => "correct",
            CounterexampleCheck::Tolerant => "correct_up_to_tolerance",
            CounterexampleCheck::Missing => "no_ce",
            CounterexampleCheck::Malformed => "malformed_ce",
            CounterexampleCheck::Invalid => "spec_not_violated",
            CounterexampleCheck::Unsupported => "unsupported",
        };
        let identity = organizer_row_identity(&fields).unwrap();
        let value = serde_json::json!({
            "schema_version": 1, "checker_version": "vnncomp2026-cex-1",
            "provider": "CPUExecutionProvider",
            "dependency_versions": {"python": "3.12.13", "onnx": "1.21.0", "onnxruntime": "1.26.0"},
            "benchmark": {"category": format!("2026_{}", fields[0]), "version": "2.0",
                          "network_field": fields[1], "property_field": fields[2]},
            "results_row_identity": identity, "result": result,
            "classification": classification, "establishes_sat": strict, "score_credit": credit,
        });
        write_file(
            root,
            &format!(
                "{tool}/2026_{}/{identity}.counterexample.check.json",
                fields[0]
            ),
            &serde_json::to_string(&value).unwrap(),
        )
    }

    #[test]
    fn organizer_row_hash_matches_pinned_tolerant_acc_run_and_python_unicode_encoding() {
        let row = vec![
            "adaptive_cruise_control_non_linear_2026",
            "vnncomp2026_benchmarks/benchmarks/adaptive_cruise_control_non_linear_2026/2.0/onnx/acc-2000000-64-64-64-64-retrain-100000-200000-0.9.onnx",
            "vnncomp2026_benchmarks/benchmarks/adaptive_cruise_control_non_linear_2026/2.0/vnnlib/instance_15.vnnlib",
            "4.453935807", "sat", "7.936269926",
        ].into_iter().map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(
            organizer_row_identity(&row).unwrap(),
            "da9b41c6efa0dc45213f5d0c8edc46964906a7b45d39459eb37aa77dfed6ad6f"
        );
        let row = [
            "cgan2026",
            "onnx/é😀.onnx",
            "vnnlib/p.vnnlib",
            "0",
            "violated",
            "1",
        ]
        .map(str::to_owned);
        assert_eq!(
            organizer_row_identity(&row).unwrap(),
            "3ce4327bd135282fdee05cc51b053b4e6b00347c0d2056bb11fd7e474f77dc97"
        );
    }

    #[test]
    fn published_tolerant_sat_earns_own_credit_without_convicting_unsat() {
        let dir = TempDir::new().unwrap();
        let official = TempDir::new().unwrap();
        let sat = "acasxu_2023,corpus/2.0/onnx/m.onnx,corpus/2.0/vnnlib/p.vnnlib,0,sat,1\n";
        write_file(&official, "tolerant/results.csv", sat);
        write_file(
            &official,
            "holds/results.csv",
            &sat.replace(",sat,", ",unsat,"),
        );
        publish_check(&official, "tolerant", sat, CounterexampleCheck::Tolerant);
        let current = write_file(&dir, "current.csv", &sat.replace(",sat,", ",unsat,"));
        let mut options = opts(current, None);
        options.official = Some(official.path().into());
        options.year = Some(ScoreYear::Y2026);
        options.witnesses = Witnesses::Adjudicated;
        let report = run_score(&options).unwrap();
        let score = bench(&report, "acasxu_2023");
        assert_eq!(score.current_raw_score, Some(10));
        assert_eq!(score.field_scores["tolerant"].raw, 10);
        assert_eq!(score.field_scores["holds"].raw, 10);
        assert_eq!(score.field_best_raw_score, Some(10));
        assert_eq!(score.field_unqualified_sat, 0);
    }

    #[test]
    fn strict_ny_check_rescores_incumbent_penalties_and_shared_denominators() {
        let dir = TempDir::new().unwrap();
        let path = write_file(&dir, "ny.csv", "acasxu_2023,onnx/m.onnx,vnnlib/p.vnnlib,0,sat,1\nacasxu_2023,onnx/m.onnx,vnnlib/q.vnnlib,0,unsat,1\n");
        let mut current = load(&path).unwrap();
        let incumbent_path = write_file(&dir, "incumbent.csv", "acasxu_2023,onnx/m.onnx,vnnlib/p.vnnlib,0,unsat,1\nacasxu_2023,onnx/m.onnx,vnnlib/q.vnnlib,0,unsat,1\n");
        let field = BTreeMap::from([("incumbent".into(), load(&incumbent_path).unwrap())]);
        let published = GroundTruth {
            verdicts: current
                .keys()
                .map(|key| (key.ground_truth_key().unwrap(), Verdict::Unsat))
                .collect(),
            ..Default::default()
        };
        let evaluate = |current: &ResultRows, excluded: &BTreeSet<InstanceKey>| {
            modeled_official_score(
                "acasxu_2023",
                current,
                &field,
                None,
                ScoreYear::Y2026,
                ScoreTrack::Regular,
                excluded,
                Witnesses::Adjudicated,
                Some(&published),
            )
            .unwrap()
        };
        let before = evaluate(&current, &BTreeSet::new());
        assert_eq!(
            (
                before.current_raw,
                before.current_unqualified_sat,
                before.field_best_raw
            ),
            (10, 1, 20)
        );
        assert_eq!(before.normalized, 50.0);
        let key = current
            .iter()
            .find(|(_, entry)| entry.verdict == Verdict::Sat)
            .unwrap()
            .0
            .clone();
        current.get_mut(&key).unwrap().counterexample = Some(CounterexampleCheck::Strict);
        let after = evaluate(&current, &BTreeSet::new());
        assert_eq!(after.field_scores["incumbent"].raw, -140);
        assert_eq!(
            (
                after.current_raw,
                after.current_unqualified_sat,
                after.field_best_raw
            ),
            (20, 0, 20)
        );
        assert_eq!(
            (after.normalized, after.field_scores["incumbent"].normalized),
            (100.0, 0.0)
        );
        let late = evaluate(&current, &BTreeSet::from([key]));
        assert_eq!(
            late.field_scores["incumbent"].raw, 20,
            "late NY SAT cannot convict another tool"
        );
        assert_eq!(late.normalized, 50.0);
    }

    #[test]
    fn per_run_unknown_missing_invalid_and_unsupported_are_distinct() {
        let dir = TempDir::new().unwrap();
        let path = write_file(
            &dir,
            "rows.csv",
            "acasxu_2023,onnx/m.onnx,vnnlib/p.vnnlib,0,sat,1\n",
        );
        let mut rows = load(&path).unwrap();
        let key = rows.keys().next().unwrap().clone();
        let scored = BTreeSet::from([key.clone()]);
        let adjudication = Adjudication {
            per_run: BTreeMap::from([(key.clone(), false)]),
            ..Default::default()
        };
        for (check, points, incorrect, unqualified) in [
            (None, 0, 0, 1),
            (Some(CounterexampleCheck::Unsupported), 0, 0, 1),
            (Some(CounterexampleCheck::Missing), -150, 1, 0),
            (Some(CounterexampleCheck::Malformed), -150, 1, 0),
            (Some(CounterexampleCheck::Invalid), -150, 1, 0),
            (Some(CounterexampleCheck::Tolerant), 10, 0, 0),
            (Some(CounterexampleCheck::Strict), 10, 0, 0),
        ] {
            rows.get_mut(&key).unwrap().counterexample = check;
            let score = score_rows(&rows, &scored, &adjudication, &BTreeSet::new());
            assert_eq!(
                (score.raw, score.incorrect, score.unqualified_sat),
                (points, incorrect, unqualified)
            );
        }
    }

    #[test]
    fn per_run_checks_bind_complete_paths_version_runtime_and_derived_flags() {
        let official = TempDir::new().unwrap();
        let medical = "safenlp_2024,corpus/2.0/onnx/medical/perturbations_0.onnx,corpus/2.0/vnnlib/medical/hyperrectangle_570.vnnlib,0,sat,1\n";
        let robot = medical.replace("medical", "ruarobot");
        write_file(&official, "tool/results.csv", &format!("{medical}{robot}"));
        let check = publish_check(&official, "tool", medical, CounterexampleCheck::Tolerant);
        publish_check(&official, "tool", &robot, CounterexampleCheck::Strict);
        let mut field = load_official_field(official.path()).unwrap();
        load_published_counterexample_checks(official.path(), &mut field).unwrap();
        let statuses = field["tool"]
            .values()
            .map(|row| row.counterexample)
            .collect::<Vec<_>>();
        assert_eq!(
            statuses,
            [
                Some(CounterexampleCheck::Tolerant),
                Some(CounterexampleCheck::Strict)
            ]
        );
        let mut changed =
            serde_json::from_slice::<serde_json::Value>(&std::fs::read(&check).unwrap()).unwrap();
        changed["establishes_sat"] = true.into();
        std::fs::write(&check, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(load_published_counterexample_checks(official.path(), &mut field).is_err());
        changed["establishes_sat"] = false.into();
        changed["benchmark"]["version"] = "1.0".into();
        std::fs::write(&check, serde_json::to_vec(&changed).unwrap()).unwrap();
        load_published_counterexample_checks(official.path(), &mut field).unwrap();
        assert_eq!(
            field["tool"].values().next().unwrap().counterexample,
            None,
            "a different version must not authenticate the row"
        );
        assert_ne!(
            organizer_row_identity(
                &parse_csv_line(medical.trim_end_matches(['\r', '\n'])).unwrap()
            )
            .unwrap(),
            organizer_row_identity(
                &parse_csv_line(
                    &medical
                        .trim_end_matches(['\r', '\n'])
                        .replace(",sat,1", ",sat,1.0"),
                )
                .unwrap()
            )
            .unwrap()
        );
    }

    #[test]
    fn raw_ny_sat_and_validation_only_checks_never_acquire_fresh_credit() {
        let dir = TempDir::new().unwrap();
        let official = TempDir::new().unwrap();
        let row = "acasxu_2023,corpus/2.0/onnx/m.onnx,corpus/2.0/vnnlib/p.vnnlib,0,sat,1\n";
        let current = write_file(&dir, "current.csv", row);
        write_file(
            &official,
            "holds/results.csv",
            &row.replace(",sat,", ",unsat,"),
        );
        let mut options = opts(current.clone(), None);
        options.official = Some(official.path().into());
        options.year = Some(ScoreYear::Y2026);
        options.witnesses = Witnesses::Adjudicated;
        let report = run_score(&options).unwrap();
        assert_eq!(report.normalized_total, Some(0.0));
        assert_eq!(report.total_unqualified_sat, 1);
        assert_eq!(report.field_normalized_totals["holds"], 100.0);

        let check = publish_check(&official, "validation", row, CounterexampleCheck::Strict);
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&check).unwrap()).unwrap();
        let source = "a".repeat(40);
        value["ny_evidence"] = serde_json::json!({
            "schema": "ny_vnncomp2026_row_replay_evidence_v1", "tool": "ny",
            "validation_only": true, "source_commit": source, "occurrence": 0
        });
        let checks = TempDir::new().unwrap();
        std::fs::write(
            checks.path().join("row.check.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        let mut rows = load(&current).unwrap();
        let error = load_ny_counterexample_checks(checks.path(), &source, &mut rows).unwrap_err();
        assert!(error.to_string().contains("validation-only"));
        assert!(rows.values().all(|entry| entry.counterexample.is_none()));
    }

    #[test]
    fn a_strict_check_on_one_duplicate_occurrence_does_not_relabel_another() {
        let dir = TempDir::new().unwrap();
        let sat = "acasxu_2023,onnx/m.onnx,vnnlib/p.vnnlib,0,sat,1\n";
        let current_path = write_file(&dir, "ny.csv", &sat.repeat(2));
        let incumbent_path = write_file(
            &dir,
            "field.csv",
            &sat.replace(",sat,", ",unsat,").repeat(2),
        );
        let mut current = load(&current_path).unwrap();
        current.values_mut().next().unwrap().counterexample = Some(CounterexampleCheck::Strict);
        let field = BTreeMap::from([("incumbent".into(), load(&incumbent_path).unwrap())]);
        let score = modeled_official_score(
            "acasxu_2023",
            &current,
            &field,
            None,
            ScoreYear::Y2026,
            ScoreTrack::Regular,
            &BTreeSet::new(),
            Witnesses::Adjudicated,
            None,
        )
        .unwrap();
        assert_eq!(score.current_raw, 10);
        assert_eq!(score.current_unqualified_sat, 1);
        assert_eq!(score.field_scores["incumbent"].raw, -140);
    }

    #[test]
    fn counts_new_solves_and_regressions_separately() {
        let dir = TempDir::new().expect("tempdir");
        let reference = write_file(
            &dir,
            "reference.csv",
            "b,onnx/m.onnx,vnnlib/a.vnnlib,unsat,1.0\n\
             b,onnx/m.onnx,vnnlib/c.vnnlib,timeout,9.0\n\
             b,onnx/m.onnx,vnnlib/d.vnnlib,unsat,1.0\n",
        );
        let current = write_file(
            &dir,
            "current.csv",
            "b,onnx/m.onnx,vnnlib/a.vnnlib,unsat,1.0\n\
             b,onnx/m.onnx,vnnlib/c.vnnlib,unsat,2.0\n\
             b,onnx/m.onnx,vnnlib/d.vnnlib,timeout,9.0\n",
        );
        let report = run_score(&opts(current, Some(reference))).expect("score");
        assert_eq!(report.total_new_solves, 1);
        assert_eq!(report.total_regressions, 1);
        assert_eq!(report.total_current_solved, 2);
        assert_eq!(report.total_reference_solved, 2);
    }

    #[test]
    fn baseline_only_rows_are_regressions_and_categories_are_not_dropped() {
        let dir = TempDir::new().expect("tempdir");
        let reference = write_file(
            &dir,
            "reference.csv",
            "a,onnx/m.onnx,vnnlib/gone.vnnlib,unsat,1\n\
             b,onnx/m.onnx,vnnlib/also-gone.vnnlib,sat,1\n",
        );
        let current = write_file(
            &dir,
            "current.csv",
            "a,onnx/m.onnx,vnnlib/still.vnnlib,unknown,1\n",
        );

        let report = run_score(&opts(current, Some(reference))).expect("score");

        assert_eq!(report.total_regressions, 2);
        assert_eq!(report.total_reference_solved, 2);
        assert_eq!(report.total_current_solved, 0);
        assert_eq!(
            report
                .benchmarks
                .iter()
                .map(|benchmark| benchmark.benchmark.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
    }

    #[test]
    fn a_sat_unsat_flip_is_an_alarm_not_progress() {
        let dir = TempDir::new().expect("tempdir");
        let reference = write_file(
            &dir,
            "reference.csv",
            "b,onnx/m.onnx,vnnlib/a.vnnlib,unsat,1\n",
        );
        let current = write_file(&dir, "current.csv", "b,onnx/m.onnx,vnnlib/a.vnnlib,sat,1\n");
        let report = run_score(&opts(current, Some(reference))).expect("score");
        assert_eq!(report.total_verdict_flips, 1);
        assert_eq!(report.total_new_solves, 0);
        assert_eq!(report.total_regressions, 0);
    }

    #[test]
    fn without_a_reference_everything_solved_is_new() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "b,onnx/m.onnx,vnnlib/a.vnnlib,unsat,1\n\
             b,onnx/m.onnx,vnnlib/c.vnnlib,timeout,9\n",
        );
        let report = run_score(&opts(current, None)).expect("score");
        assert_eq!(report.total_current_solved, 1);
        assert_eq!(report.total_new_solves, 1);
        assert_eq!(report.total_regressions, 0);
    }

    #[test]
    fn normalized_total_is_absent_without_an_official_field() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "b,onnx/m.onnx,vnnlib/a.vnnlib,unsat,1\n",
        );
        let report = run_score(&opts(current, None)).expect("score");
        assert!(report.normalized_total.is_none());
    }

    #[test]
    fn parses_legacy_five_official_six_and_measured_seven_column_schemas() {
        let dir = TempDir::new().expect("tempdir");
        let five = write_file(
            &dir,
            "five.csv",
            "a,onnx/m.onnx,vnnlib/five.vnnlib,sat,1.5\n",
        );
        let six = write_file(
            &dir,
            "six.csv",
            "category,network,property,prepare_time,result,run_time\n\
             a,onnx/m.onnx,vnnlib/six.vnnlib,0,unsat,2.5\n",
        );
        let seven = write_file(
            &dir,
            "seven.csv",
            "a,onnx/m.onnx,vnnlib/seven.vnnlib,0,sat,3.5,run-2026\n",
        );

        assert_eq!(
            read_results(&five)
                .expect("five")
                .values()
                .next()
                .expect("row")
                .verdict,
            Verdict::Sat
        );
        let parsed_six = read_results(&six).expect("six");
        assert_eq!(
            parsed_six.len(),
            1,
            "the official network/property header must not become a result row"
        );
        assert_eq!(
            parsed_six.values().next().expect("row").verdict,
            Verdict::Unsat
        );
        assert_eq!(
            read_results(&seven)
                .expect("seven")
                .values()
                .next()
                .expect("row")
                .verdict,
            Verdict::Sat,
            "the seven-column verdict is index 4, not the second-to-last runtime"
        );
    }

    #[test]
    fn unsupported_or_ambiguous_schema_is_rejected() {
        let dir = TempDir::new().expect("tempdir");
        let path = write_file(
            &dir,
            "bad.csv",
            "a,onnx/m.onnx,vnnlib/p.vnnlib,prepared,unsat,1,run,extra\n",
        );
        let error = read_results(&path).expect_err("must reject");
        assert!(error.to_string().contains("unsupported results schema"));
    }

    #[test]
    fn nested_same_basename_instances_do_not_collide() {
        let dir = TempDir::new().expect("tempdir");
        let path = write_file(
            &dir,
            "nested.csv",
            "safenlp_2024,onnx/medical/perturbations_0.onnx,\
             vnnlib/medical/hyperrectangle_1215.vnnlib,unsat,1\n\
             safenlp_2024,onnx/ruarobot/perturbations_0.onnx,\
             vnnlib/ruarobot/hyperrectangle_1215.vnnlib,sat,1\n",
        );
        let rows = read_results(&path).expect("parse");
        assert_eq!(rows.len(), 2);
        assert_ne!(
            rows.keys().next().expect("first"),
            rows.keys().next_back().expect("last")
        );
    }

    #[test]
    fn repeated_instances_are_preserved_as_stable_occurrences() {
        let dir = TempDir::new().expect("tempdir");
        let path = write_file(
            &dir,
            "repeated.csv",
            "safenlp_2024,onnx/medical/m.onnx,vnnlib/medical/p.vnnlib,unsat,1\n\
             safenlp_2024,/root/corpus/onnx/medical/m.onnx,\
             /root/corpus/vnnlib/medical/p.vnnlib,sat,1\n",
        );
        let rows = read_results(&path).expect("repeated rows");
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows.keys().map(|key| key.occurrence).collect::<Vec<_>>(),
            [0, 1],
            "repeated official instances must each remain scoreable"
        );

        let relative =
            InstanceKey::new("category-a", "models/m.onnx", "properties/p.vnnlib").expect("key");
        let absolute = InstanceKey::new(
            "category-a",
            "/root/corpus/category-a/models/m.onnx",
            "/root/corpus/category-a/properties/p.vnnlib",
        )
        .expect("key");
        assert_eq!(
            relative, absolute,
            "category-relative identities must match even without onnx/vnnlib directories"
        );
    }

    #[test]
    fn identical_banks_merge_idempotently_but_conflicts_fail_loudly() {
        let dir = TempDir::new().expect("tempdir");
        write_file(&dir, "a.csv", "b,onnx/m.onnx,vnnlib/p.vnnlib,unsat,1\n");
        write_file(
            &dir,
            "b.csv",
            "b,/root/onnx/m.onnx,/root/vnnlib/p.vnnlib,unsat,2\n",
        );
        assert_eq!(
            read_results_dir(dir.path())
                .expect("idempotent merge")
                .len(),
            1
        );

        write_file(&dir, "c.csv", "b,onnx/m.onnx,vnnlib/p.vnnlib,sat,3\n");
        let error = read_results_dir(dir.path()).expect_err("conflict must fail");
        assert!(error.to_string().contains("conflicting result identity"));
    }

    #[test]
    fn directory_loading_and_row_reporting_are_deterministic() {
        let dir = TempDir::new().expect("tempdir");
        write_file(&dir, "z.csv", "z,onnx/z.onnx,vnnlib/z.vnnlib,unsat,1\n");
        write_file(
            &dir,
            "a.csv",
            "a,onnx/a.onnx,vnnlib/c.vnnlib,unsat,1\n\
             a,onnx/a.onnx,vnnlib/b.vnnlib,unsat,1\n",
        );
        let rows = read_results_dir(dir.path()).expect("load");
        let identities = rows
            .keys()
            .map(|key| (key.benchmark.clone(), key.onnx.clone(), key.vnnlib.clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            identities,
            [
                (
                    "a".to_string(),
                    "onnx/a.onnx".to_string(),
                    "vnnlib/b.vnnlib".to_string()
                ),
                (
                    "a".to_string(),
                    "onnx/a.onnx".to_string(),
                    "vnnlib/c.vnnlib".to_string()
                ),
                (
                    "z".to_string(),
                    "onnx/z.onnx".to_string(),
                    "vnnlib/z.vnnlib".to_string()
                ),
            ]
        );
    }

    #[test]
    fn test_instances_are_excluded_like_the_official_processor() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "b,onnx/m.onnx,vnnlib/test_nano.vnnlib,unsat,1\n\
             b,onnx/m.onnx,vnnlib/a.vnnlib,unsat,1\n",
        );
        let report = run_score(&opts(current, None)).expect("score");
        assert_eq!(report.total_current_solved, 1);
    }

    #[test]
    fn quoted_paired_model_rows_parse_as_one_instance() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "paired.csv",
            "mono,\"[('f', 'a.onnx'), ('g', 'b.onnx')]\",i0.vnnlib,sat,1.1\n",
        );
        let report = run_score(&opts(current, None)).expect("score");
        assert_eq!(report.total_current_solved, 1);
        assert_eq!(report.benchmarks[0].benchmark, "mono");
    }

    #[test]
    fn official_model_applies_positive_points_and_incorrect_penalties() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "acasxu_2023,onnx/m.onnx,vnnlib/a.vnnlib,unsat,1\n\
             acasxu_2023,onnx/m.onnx,vnnlib/c.vnnlib,sat,1\n",
        );
        let official = TempDir::new().expect("official tempdir");
        write_file(
            &official,
            "tool-a/b/results.csv",
            "acasxu_2023,/corpus/onnx/m.onnx,/corpus/vnnlib/a.vnnlib,0,sat,1\n\
             acasxu_2023,/corpus/onnx/m.onnx,/corpus/vnnlib/c.vnnlib,0,unsat,1\n",
        );
        write_file(
            &official,
            "tool-b/b/results.csv",
            "acasxu_2023,onnx/m.onnx,vnnlib/a.vnnlib,0,sat,1\n\
             acasxu_2023,onnx/m.onnx,vnnlib/c.vnnlib,0,sat,1\n",
        );
        let mut options = opts(current, None);
        options.official = Some(official.path().to_path_buf());
        options.year = Some(ScoreYear::Y2026);

        let report = run_score(&options).expect("score");
        let benchmark = &report.benchmarks[0];
        assert_eq!(benchmark.current_correct, Some(1));
        assert_eq!(benchmark.current_incorrect, Some(1));
        assert_eq!(
            benchmark.current_raw_score,
            Some(POINTS_CORRECT + PENALTY_INCORRECT)
        );
        assert_eq!(benchmark.field_best_raw_score, Some(20));
        assert_eq!(benchmark.normalized, Some(0.0));
        assert_eq!(report.normalized_total, Some(0.0));
    }

    #[test]
    fn official_normalization_uses_best_raw_score_not_solve_count() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "acasxu_2023,onnx/m.onnx,vnnlib/a.vnnlib,unsat,1\n",
        );
        let official = TempDir::new().expect("official tempdir");
        write_file(
            &official,
            "winner/b/results.csv",
            "acasxu_2023,onnx/m.onnx,vnnlib/a.vnnlib,prepared,unsat,1,run-a\n\
             acasxu_2023,onnx/m.onnx,vnnlib/b.vnnlib,prepared,unsat,1,run-a\n",
        );
        let mut options = opts(current, None);
        options.official = Some(official.path().to_path_buf());
        options.year = Some(ScoreYear::Y2026);

        let report = run_score(&options).expect("score");
        let benchmark = &report.benchmarks[0];
        assert_eq!(benchmark.current_raw_score, Some(10));
        assert_eq!(benchmark.field_best_raw_score, Some(20));
        assert_eq!(benchmark.normalized, Some(50.0));
    }

    #[test]
    fn official_scoring_ignores_rows_outside_the_supplied_field() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "acasxu_2023,onnx/m.onnx,vnnlib/official.vnnlib,unsat,1\n\
             acasxu_2023,onnx/m.onnx,vnnlib/not-in-field.vnnlib,unsat,1\n",
        );
        let official = TempDir::new().expect("official tempdir");
        write_file(
            &official,
            "winner/b/results.csv",
            "acasxu_2023,onnx/m.onnx,vnnlib/official.vnnlib,0,unsat,1\n",
        );
        let mut options = opts(current, None);
        options.official = Some(official.path().to_path_buf());
        options.year = Some(ScoreYear::Y2026);

        let report = run_score(&options).expect("score");
        let benchmark = &report.benchmarks[0];
        assert_eq!(
            benchmark.current_solved, 2,
            "solve-count reporting still covers every current row"
        );
        assert_eq!(
            benchmark.current_raw_score,
            Some(10),
            "a non-field row must not earn modeled official points"
        );
        assert_eq!(benchmark.current_correct, Some(1));
        assert_eq!(benchmark.field_best_raw_score, Some(10));
        assert_eq!(benchmark.normalized, Some(100.0));
    }

    #[test]
    fn official_only_categories_are_visible_zeroes_not_silently_omitted() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(&dir, "current.csv", "# intentionally empty current bank\n");
        let official = TempDir::new().expect("official tempdir");
        write_file(
            &official,
            "winner/acasxu_2023/results.csv",
            "acasxu_2023,onnx/m.onnx,vnnlib/a.vnnlib,0,unsat,1\n",
        );
        let mut options = opts(current, None);
        options.official = Some(official.path().to_path_buf());
        options.year = Some(ScoreYear::Y2026);

        let report = run_score(&options).expect("score");

        assert_eq!(report.benchmarks.len(), 1);
        let benchmark = &report.benchmarks[0];
        assert_eq!(benchmark.benchmark, "acasxu_2023");
        assert_eq!(benchmark.rows_compared, 0);
        assert_eq!(benchmark.current_solved, 0);
        assert_eq!(benchmark.current_raw_score, Some(0));
        assert_eq!(benchmark.field_best_raw_score, Some(10));
        assert_eq!(benchmark.normalized, Some(0.0));
        assert_eq!(report.officially_scored_benchmarks, 1);
        assert_eq!(report.normalized_total, Some(0.0));
        assert_eq!(report.modeled_score_caveat, Some(MODELED_2026_SCORE_CAVEAT));
    }

    #[test]
    fn historical_reference_is_not_treated_as_truth() {
        let dir = TempDir::new().expect("tempdir");
        let reference = write_file(
            &dir,
            "reference.csv",
            "acasxu_2023,onnx/m.onnx,vnnlib/a.vnnlib,sat,1\n",
        );
        let current = write_file(
            &dir,
            "current.csv",
            "acasxu_2023,onnx/m.onnx,vnnlib/a.vnnlib,unsat,1\n",
        );
        let official = TempDir::new().expect("official tempdir");
        write_file(
            &official,
            "reference-tool/b/results.csv",
            "acasxu_2023,onnx/m.onnx,vnnlib/a.vnnlib,0,unsat,1\n",
        );
        let mut options = opts(current, Some(reference));
        options.official = Some(official.path().to_path_buf());
        options.year = Some(ScoreYear::Y2026);

        let report = run_score(&options).expect("score");
        let benchmark = &report.benchmarks[0];
        assert_eq!(report.total_verdict_flips, 1);
        assert_eq!(benchmark.current_correct, Some(1));
        assert_eq!(benchmark.current_incorrect, Some(0));
        assert_eq!(benchmark.current_raw_score, Some(10));
        assert_eq!(benchmark.normalized, Some(100.0));
    }

    #[test]
    fn relational_model_identity_matches_absolute_and_relative_member_paths() {
        let relative = InstanceKey::new(
            "relusplitter",
            "[('f', 'onnx/a.onnx'), ('g', 'onnx/nested/b.onnx')]",
            "vnnlib/p.vnnlib",
        )
        .expect("relative identity");
        let absolute = InstanceKey::new(
            "relusplitter",
            "[('f', '/corpus/onnx/a.onnx'), ('g', '/corpus/onnx/nested/b.onnx')]",
            "/corpus/vnnlib/p.vnnlib",
        )
        .expect("absolute identity");
        assert_eq!(relative, absolute);
    }

    #[test]
    fn malformed_csv_unknown_verdicts_and_nonfinite_times_fail_closed() {
        let dir = TempDir::new().expect("tempdir");
        for (name, body) in [
            (
                "unterminated.csv",
                "\"acasxu_2023,onnx/m.onnx,vnnlib/p.vnnlib,unsat,1\n",
            ),
            (
                "verdict.csv",
                "acasxu_2023,onnx/m.onnx,vnnlib/p.vnnlib,satt,1\n",
            ),
            (
                "runtime.csv",
                "acasxu_2023,onnx/m.onnx,vnnlib/p.vnnlib,unsat,NaN\n",
            ),
            (
                "prepare.csv",
                "acasxu_2023,onnx/m.onnx,vnnlib/p.vnnlib,not-prepared,unsat,1\n",
            ),
        ] {
            let path = write_file(&dir, name, body);
            assert!(read_results(&path).is_err(), "{name} must be rejected");
        }
    }

    #[test]
    fn pinned_released_2025_metadata_matches_the_compiled_track_tables() {
        let snapshot = include_str!("../../testdata/vnncomp2025_track_membership.csv");
        let mut regular = BTreeSet::new();
        let mut extended = BTreeSet::new();
        for line in snapshot.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (track, benchmark) = line.split_once(',').expect("track snapshot row");
            match track {
                "regular" => {
                    assert!(regular.insert(benchmark), "duplicate regular {benchmark}");
                }
                "extended" => {
                    assert!(extended.insert(benchmark), "duplicate extended {benchmark}");
                }
                other => panic!("unknown snapshot track {other:?}"),
            }
        }

        assert_eq!(
            regular,
            REGULAR_TRACK_2025.iter().copied().collect(),
            "compiled regular table must match the pinned released scorer"
        );
        assert_eq!(
            extended,
            EXTENDED_TRACK_2025.iter().copied().collect(),
            "compiled extended table must match the pinned released scorer"
        );
        assert!(
            regular.is_disjoint(&extended),
            "a category cannot be in both 2025 tracks"
        );
    }

    #[test]
    fn pinned_2026_vote_metadata_matches_the_compiled_track_tables() {
        let snapshot = include_str!("../../testdata/vnncomp2026_track_membership.csv");
        let mut regular = BTreeSet::new();
        let mut extended = BTreeSet::new();
        for line in snapshot.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (track, benchmark) = line.split_once(',').expect("track snapshot row");
            match track {
                "regular" => {
                    assert!(regular.insert(benchmark), "duplicate regular {benchmark}");
                }
                "extended" => {
                    assert!(extended.insert(benchmark), "duplicate extended {benchmark}");
                }
                other => panic!("unknown snapshot track {other:?}"),
            }
        }

        assert_eq!(
            regular,
            REGULAR_TRACK_2026.iter().copied().collect(),
            "compiled regular table must match the frozen official vote snapshot"
        );
        assert_eq!(
            extended,
            EXTENDED_TRACK_2026.iter().copied().collect(),
            "compiled extended table must match the frozen official vote snapshot"
        );
        assert!(
            regular.is_disjoint(&extended),
            "a category cannot be in both 2026 tracks"
        );
    }

    #[test]
    fn categories_that_changed_tracks_are_not_aliased_between_years() {
        for benchmark in [
            "lsnc_relu",
            "ml4acopf_2024",
            "traffic_signs_recognition_2023",
            "vggnet16_2022",
            "vit_2023",
            "yolo_2023",
        ] {
            assert!(
                ScoreTrack::Extended.includes(ScoreYear::Y2025, benchmark),
                "{benchmark} was extended in 2025"
            );
            assert!(
                !ScoreTrack::Regular.includes(ScoreYear::Y2025, benchmark),
                "{benchmark} must not leak into the 2025 regular score"
            );
            assert!(
                ScoreTrack::Regular.includes(ScoreYear::Y2026, benchmark),
                "{benchmark} moved to regular in 2026"
            );
            assert!(
                !ScoreTrack::Extended.includes(ScoreYear::Y2026, benchmark),
                "{benchmark} must not remain extended in 2026"
            );
        }
    }

    #[test]
    fn released_2025_only_categories_are_scored_in_their_actual_tracks() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "cgan_2023,onnx/c.onnx,vnnlib/c.vnnlib,unsat,1\n\
             soundnessbench,onnx/s.onnx,vnnlib/s.vnnlib,unsat,1\n\
             relusplitter,onnx/r.onnx,vnnlib/r.vnnlib,unsat,1\n",
        );
        let official = TempDir::new().expect("official");
        write_file(
            &official,
            "winner/results.csv",
            "cgan_2023,onnx/c.onnx,vnnlib/c.vnnlib,0,unsat,1\n\
             soundnessbench,onnx/s.onnx,vnnlib/s.vnnlib,0,unsat,1\n\
             relusplitter,onnx/r.onnx,vnnlib/r.vnnlib,0,unsat,1\n",
        );

        let mut regular = opts(current.clone(), None);
        regular.official = Some(official.path().to_path_buf());
        regular.year = Some(ScoreYear::Y2025);
        let regular_report = run_score(&regular).expect("2025 regular");
        assert_eq!(regular_report.score_year, Some(ScoreYear::Y2025));
        assert_eq!(regular_report.officially_scored_benchmarks, 2);
        assert_eq!(regular_report.normalized_total, Some(200.0));

        let mut extended = opts(current, None);
        extended.official = Some(official.path().to_path_buf());
        extended.year = Some(ScoreYear::Y2025);
        extended.track = ScoreTrack::Extended;
        let extended_report = run_score(&extended).expect("2025 extended");
        assert_eq!(extended_report.score_year, Some(ScoreYear::Y2025));
        assert_eq!(extended_report.officially_scored_benchmarks, 1);
        assert_eq!(extended_report.normalized_total, Some(100.0));
    }

    #[test]
    fn official_scoring_requires_an_explicit_supported_year() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,unsat,1\n",
        );
        let mut options = opts(current, None);
        options.official = Some(dir.path().join("not-consulted"));

        let error = run_score(&options).expect_err("year must be explicit");
        assert!(error.to_string().contains("--year <2025|2026> is required"));
    }

    #[test]
    fn explicit_year_rejects_wrong_prefixes_and_other_year_only_categories() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,unsat,1\n",
        );
        let prefixed = TempDir::new().expect("official");
        write_file(
            &prefixed,
            "winner/results.csv",
            "2026_acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,1\n",
        );
        let mut options = opts(current.clone(), None);
        options.official = Some(prefixed.path().to_path_buf());
        options.year = Some(ScoreYear::Y2025);
        let error = run_score(&options).expect_err("wrong prefix must fail");
        assert!(error.to_string().contains("2026-prefixed"));
        assert!(error.to_string().contains("--year 2025"));

        let other_year = TempDir::new().expect("official");
        write_file(
            &other_year,
            "winner/results.csv",
            "cgan_2023,onnx/c.onnx,vnnlib/c.vnnlib,0,unsat,1\n",
        );
        let mut options = opts(current, None);
        options.official = Some(other_year.path().to_path_buf());
        options.year = Some(ScoreYear::Y2026);
        let error = run_score(&options).expect_err("2025-only category must fail");
        assert!(error
            .to_string()
            .contains("pinned VNN-COMP 2026 regular/extended track lists"));
        assert!(error.to_string().contains("cgan_2023"));
    }

    #[test]
    fn result_banks_with_conflicting_explicit_year_prefixes_fail_closed() {
        let dir = TempDir::new().expect("tempdir");
        let y2025 = write_file(
            &dir,
            "2025.csv",
            "2025_acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,unsat,1\n",
        );
        let y2026 = write_file(
            &dir,
            "2026.csv",
            "2026_acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,unsat,1\n",
        );
        let mut options = opts(y2025, None);
        options.results.push(y2026);

        let error = run_score(&options).expect_err("mixed years must fail");
        assert!(error
            .to_string()
            .contains("mixed competition-year prefixes"));
    }

    #[test]
    fn regular_and_extended_tracks_are_never_combined() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "relusplitter_2026,onnx/r.onnx,vnnlib/r.vnnlib,unsat,1\n\
             smart_turn_multimodal_2026,onnx/s.onnx,vnnlib/s.vnnlib,unsat,1\n",
        );
        let official = TempDir::new().expect("official");
        write_file(
            &official,
            "winner/relusplitter_2026/results.csv",
            "relusplitter_2026,onnx/r.onnx,vnnlib/r.vnnlib,0,unsat,1\n",
        );
        write_file(
            &official,
            "winner/smart_turn_multimodal_2026/results.csv",
            "smart_turn_multimodal_2026,onnx/s.onnx,vnnlib/s.vnnlib,0,unsat,1\n",
        );

        let mut regular = opts(current.clone(), None);
        regular.official = Some(official.path().to_path_buf());
        regular.year = Some(ScoreYear::Y2026);
        let regular_report = run_score(&regular).expect("regular score");
        assert_eq!(regular_report.officially_scored_benchmarks, 1);
        assert_eq!(regular_report.score_year, Some(ScoreYear::Y2026));
        assert_eq!(regular_report.score_track, Some(ScoreTrack::Regular));

        let mut extended = opts(current, None);
        extended.official = Some(official.path().to_path_buf());
        extended.year = Some(ScoreYear::Y2026);
        extended.track = ScoreTrack::Extended;
        let extended_report = run_score(&extended).expect("extended score");
        assert_eq!(extended_report.officially_scored_benchmarks, 1);
        assert_eq!(extended_report.score_track, Some(ScoreTrack::Extended));
    }

    /// A corpus fixture with the real nn4sys shape: per-instance budgets that
    /// differ by an order of magnitude within one benchmark.
    fn nn4sys_corpus(dir: &TempDir) -> Vec<(String, PathBuf)> {
        write_file(
            dir,
            "nn4sys/instances.csv",
            "onnx/mscn_2048d.onnx,vnnlib/cardinality_0_500_2048.vnnlib,20\n\
             onnx/mscn_2048d.onnx,vnnlib/cardinality_0_3800_2048.vnnlib,100\n\
             onnx/mscn_2048d.onnx,vnnlib/cardinality_1_11080_2048_dual.vnnlib,800\n",
        );
        vec![("nn4sys".to_string(), dir.path().join("nn4sys"))]
    }

    #[test]
    fn budgets_are_read_per_instance_not_per_benchmark() {
        let dir = TempDir::new().expect("tempdir");
        let table = budgets_from_categories(&nn4sys_corpus(&dir)).expect("load budgets");
        assert_eq!(table.len(), 3);
        let budget = |spec: &str| {
            table[&(
                "nn4sys".to_string(),
                "onnx/mscn_2048d.onnx".to_string(),
                format!("vnnlib/{spec}.vnnlib"),
            )]
        };
        // The defect this guards: inferring ONE budget for the benchmark from a
        // competitor's runtime on the 800s dual row, then applying it to the
        // 20s cardinality_0 rows.
        assert_eq!(budget("cardinality_0_500_2048"), 20);
        assert_eq!(budget("cardinality_0_3800_2048"), 100);
        assert_eq!(budget("cardinality_1_11080_2048_dual"), 800);
    }

    #[test]
    fn versioned_instance_lists_are_all_read_and_must_agree() {
        let dir = TempDir::new().expect("tempdir");
        write_file(
            &dir,
            "nn4sys/1.0/instances.csv",
            "onnx/m.onnx,vnnlib/a.vnnlib,40\n",
        );
        write_file(
            &dir,
            "nn4sys/2.0/instances.csv",
            "onnx/m.onnx,vnnlib/b.vnnlib,60\n",
        );
        let categories = vec![("nn4sys".to_string(), dir.path().join("nn4sys"))];
        let table = budgets_from_categories(&categories).expect("both versioned lists");
        assert_eq!(table.len(), 2, "both 1.0 and 2.0 lists must contribute");

        // Same identity, disagreeing budgets: refuse instead of picking one,
        // since guessing invents phantom points or discards real solves.
        write_file(
            &dir,
            "nn4sys/2.0/instances.csv",
            "onnx/m.onnx,vnnlib/a.vnnlib,600\n",
        );
        let error = budgets_from_categories(&categories).expect_err("conflict must fail closed");
        assert!(
            error.to_string().contains("conflicting official budgets"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_solve_past_its_own_budget_is_a_phantom_point_not_progress() {
        let corpus = TempDir::new().expect("corpus");
        let budgets = budgets_from_categories(&nn4sys_corpus(&corpus)).expect("load budgets");

        let results = TempDir::new().expect("results");
        // Exactly the claim that motivated this check: 75.5s and 624.4s solves
        // on rows whose official budgets are 20s and 100s.
        let bank = write_file(
            &results,
            "nn4sys.csv",
            "nn4sys,onnx/mscn_2048d.onnx,vnnlib/cardinality_0_500_2048.vnnlib,0,unsat,75.5\n\
             nn4sys,onnx/mscn_2048d.onnx,vnnlib/cardinality_0_3800_2048.vnnlib,0,unsat,624.4\n\
             nn4sys,onnx/mscn_2048d.onnx,vnnlib/cardinality_1_11080_2048_dual.vnnlib,0,unsat,78.5\n",
        );
        let rows = load(&bank).expect("load bank");
        let audit = audit_budgets(&rows, &budgets, "nn4sys");
        assert_eq!(audit.checked, 3);
        assert_eq!(audit.unchecked, 0);
        assert_eq!(
            audit.rows.len(),
            2,
            "the 20s and 100s rows are over budget; the 800s row is not: {:?}",
            audit.rows
        );
        assert_eq!(audit.keys.len(), 2);
        // Each violation must quantify the overrun. Matched loosely: the exact
        // last digit is float-formatting rounding (75.5/20 renders 3.77x), which
        // is not what this test is about.
        assert!(
            audit
                .rows
                .iter()
                .all(|row| row.contains("x over, at ") && row.contains("s budget")),
            "violation must state its budget and how far over it ran: {:?}",
            audit.rows
        );
        assert!(
            audit
                .rows
                .iter()
                .any(|row| row.contains("75.5s vs 20s budget")),
            "the 20s row must be reported against its OWN budget: {:?}",
            audit.rows
        );

        // Excluded rows earn nothing: a solve past the timeout scores zero.
        let scored = rows.keys().cloned().collect::<BTreeSet<_>>();
        let uncontested = ruling_for_all(&rows, InstanceRuling::Uncontested);
        let score = score_rows(&rows, &scored, &uncontested, &audit.keys);
        assert_eq!(score.correct, 1, "only the in-budget row scores");
        assert_eq!(score.raw, POINTS_CORRECT);
    }

    #[test]
    fn an_over_budget_unsat_contradicted_by_a_witness_is_zero_not_minus_150() {
        let corpus = TempDir::new().expect("corpus");
        let budgets = budgets_from_categories(&nn4sys_corpus(&corpus)).expect("load budgets");
        let results = TempDir::new().expect("results");
        let bank = write_file(
            &results,
            "nn4sys.csv",
            "nn4sys,onnx/mscn_2048d.onnx,vnnlib/cardinality_0_500_2048.vnnlib,0,unsat,75.5\n",
        );
        let rows = load(&bank).expect("load bank");
        let audit = audit_budgets(&rows, &budgets, "nn4sys");
        let scored = rows.keys().cloned().collect::<BTreeSet<_>>();
        let convicting = ruling_for_all(&rows, InstanceRuling::WitnessCredible);

        // Past the timeout the harness reports a timeout, so the row is neither
        // a solve nor a wrong answer.
        let score = score_rows(&rows, &scored, &convicting, &audit.keys);
        assert_eq!((score.raw, score.correct, score.incorrect), (0, 0, 0));

        // Without the exclusion the same contradiction is the real -150.
        let score = score_rows(&rows, &scored, &convicting, &BTreeSet::new());
        assert_eq!((score.raw, score.incorrect), (PENALTY_INCORRECT, 1));
    }

    #[test]
    fn unbudgeted_rows_are_counted_unchecked_never_as_passing() {
        let corpus = TempDir::new().expect("corpus");
        let budgets = budgets_from_categories(&nn4sys_corpus(&corpus)).expect("load budgets");
        let results = TempDir::new().expect("results");
        let bank = write_file(
            &results,
            "nn4sys.csv",
            "nn4sys,onnx/mscn_2048d.onnx,vnnlib/not_in_the_corpus.vnnlib,0,unsat,9999\n",
        );
        let rows = load(&bank).expect("load bank");
        let audit = audit_budgets(&rows, &budgets, "nn4sys");
        assert_eq!(audit.checked, 0);
        assert_eq!(audit.unchecked, 1);
        assert!(
            audit.rows.is_empty(),
            "an unknown budget is not a violation, but it is also not a pass"
        );
    }

    #[test]
    fn undecided_rows_are_not_budget_violations() {
        let corpus = TempDir::new().expect("corpus");
        let budgets = budgets_from_categories(&nn4sys_corpus(&corpus)).expect("load budgets");
        let results = TempDir::new().expect("results");
        let bank = write_file(
            &results,
            "nn4sys.csv",
            "nn4sys,onnx/mscn_2048d.onnx,vnnlib/cardinality_0_500_2048.vnnlib,0,timeout,900\n",
        );
        let rows = load(&bank).expect("load bank");
        let audit = audit_budgets(&rows, &budgets, "nn4sys");
        assert_eq!((audit.checked, audit.unchecked), (0, 0));
        assert!(audit.rows.is_empty(), "a timeout already scores zero");
        assert!(
            audit.under_measured.is_empty(),
            "a timeout at/past its budget is a genuine measurement, not an invalid one"
        );
    }

    #[test]
    fn a_timeout_below_its_budget_is_an_invalid_measurement_not_a_capability_limit() {
        let corpus = TempDir::new().expect("corpus");
        let budgets = budgets_from_categories(&nn4sys_corpus(&corpus)).expect("load budgets");
        let results = TempDir::new().expect("results");
        // The defect this guards: scripts/measure_ny_scorecard.sh's FORMER
        // default NY_MEASURE_CAP=120 banked ~101s "timeouts" against 800-1200s
        // budgets, which were then read as capability limits. The default is
        // now 0 (no cap), but the banked rows outlive the generator fix, so
        // the audit must keep catching them.
        let bank = write_file(
            &results,
            "nn4sys.csv",
            "nn4sys,onnx/mscn_2048d.onnx,vnnlib/cardinality_1_11080_2048_dual.vnnlib,0,timeout,101\n\
             nn4sys,onnx/mscn_2048d.onnx,vnnlib/cardinality_0_500_2048.vnnlib,0,timeout,19\n\
             nn4sys,onnx/mscn_2048d.onnx,vnnlib/cardinality_0_3800_2048.vnnlib,0,unknown,40\n",
        );
        let rows = load(&bank).expect("load bank");
        let audit = audit_budgets(&rows, &budgets, "nn4sys");
        // 101s vs 800s (13%) and 40s vs 100s (40%) are invalid; 19s vs 20s
        // (95%) is inside the watchdog-grace slack and is a real measurement.
        assert_eq!(
            audit.under_measured.len(),
            2,
            "the 800s and 100s rows were never given their time: {:?}",
            audit.under_measured
        );
        assert!(
            audit
                .under_measured
                .iter()
                .any(|row| row.contains("101.0s vs 800s budget") && row.contains("13%")),
            "the violation must quantify how little of the budget was used: {:?}",
            audit.under_measured
        );
        // Invalid measurements are a lint, not a scoring change: an undecided
        // row scores zero either way.
        assert!(audit.keys.is_empty());
        assert_eq!((audit.checked, audit.unchecked), (0, 0));
    }

    #[test]
    fn a_solve_recorded_exactly_at_its_budget_is_surfaced_for_provenance_review() {
        let corpus = TempDir::new().expect("corpus");
        let budgets = budgets_from_categories(&nn4sys_corpus(&corpus)).expect("load budgets");
        let results = TempDir::new().expect("results");
        // The defect this guards: bank rows restored by hand with the runtime
        // set to the budget ceiling (safenlp `sat,20` on a 20s budget) and no
        // run tag tying them to a sealed run.
        let bank = write_file(
            &results,
            "nn4sys.csv",
            "nn4sys,onnx/mscn_2048d.onnx,vnnlib/cardinality_0_500_2048.vnnlib,0,unsat,20\n\
             nn4sys,onnx/mscn_2048d.onnx,vnnlib/cardinality_0_3800_2048.vnnlib,0,unsat,100,20260729-tagged-run\n",
        );
        let rows = load(&bank).expect("load bank");
        let audit = audit_budgets(&rows, &budgets, "nn4sys");
        assert_eq!(
            audit.at_budget.len(),
            2,
            "both rows sit exactly at their own ceilings: {:?}",
            audit.at_budget
        );
        assert!(
            audit
                .at_budget
                .iter()
                .any(|row| row.contains("run tag: NONE")),
            "an untagged at-budget solve must say so: {:?}",
            audit.at_budget
        );
        assert!(
            audit
                .at_budget
                .iter()
                .any(|row| row.contains("run tag: 20260729-tagged-run")),
            "a tagged at-budget solve must carry its tag for review: {:?}",
            audit.at_budget
        );
        // Surfaced, not excluded: finishing at the wire is legal, so both rows
        // still score until evidence says otherwise.
        assert!(audit.keys.is_empty());
        assert_eq!(audit.checked, 2);
    }

    // ---------------------------------------------------------------------
    // Witness adjudication: an uncorroborated SAT must not set a denominator.
    // ---------------------------------------------------------------------

    /// Synthetic controls for the retained historical 2025 adjudication model.
    fn planted_witness_corpus(official: &TempDir) {
        // Decides all five correctly.
        write_file(
            official,
            "honest/results.csv",
            "cgan_2023,onnx/m.onnx,vnnlib/p1.vnnlib,0,unsat,1\n\
             cgan_2023,onnx/m.onnx,vnnlib/p2.vnnlib,0,unsat,1\n\
             cgan_2023,onnx/m.onnx,vnnlib/p3.vnnlib,0,unsat,1\n\
             cgan_2023,onnx/m.onnx,vnnlib/p4.vnnlib,0,unsat,1\n\
             cgan_2023,onnx/m.onnx,vnnlib/p5.vnnlib,0,unsat,1\n",
        );
        // Claims SAT on four of them and nobody corroborates it.
        write_file(
            official,
            "planted/results.csv",
            "cgan_2023,onnx/m.onnx,vnnlib/p1.vnnlib,0,sat,1\n\
             cgan_2023,onnx/m.onnx,vnnlib/p2.vnnlib,0,sat,1\n\
             cgan_2023,onnx/m.onnx,vnnlib/p3.vnnlib,0,sat,1\n\
             cgan_2023,onnx/m.onnx,vnnlib/p4.vnnlib,0,sat,1\n",
        );
    }

    /// NY holds three of the contested rows plus the uncontested one.
    fn planted_witness_current(dir: &TempDir) -> PathBuf {
        write_file(
            dir,
            "current.csv",
            "cgan_2023,onnx/m.onnx,vnnlib/p1.vnnlib,0,unsat,1\n\
             cgan_2023,onnx/m.onnx,vnnlib/p2.vnnlib,0,unsat,1\n\
             cgan_2023,onnx/m.onnx,vnnlib/p3.vnnlib,0,unsat,1\n\
             cgan_2023,onnx/m.onnx,vnnlib/p5.vnnlib,0,unsat,1\n",
        )
    }

    fn organiser_log(dir: &TempDir, body: &str) -> PathBuf {
        write_file(dir, "SCORING/results.txt", body)
    }

    fn cgan_truth_log() -> String {
        let mut body = String::from("Category 2025_cgan_2023:\n");
        for property in ["p1", "p2", "p3", "p4", "p5"] {
            body.push_str(&format!("Row: ['m-{property}', '1.0 (h)', '1.0 (v)']\n"));
            body.push_str("True Result: unsat\n");
        }
        body
    }

    fn bench<'a>(report: &'a ProgressReport, name: &str) -> &'a BenchProgress {
        report
            .benchmarks
            .iter()
            .find(|benchmark| benchmark.benchmark == name)
            .unwrap_or_else(|| panic!("{name} present in report"))
    }

    /// THE REGRESSION THIS FIX EXISTS FOR.
    ///
    /// Under `assumed-valid` the planted tool's four unreproduced SAT claims
    /// convict everyone who disagrees, which makes the planted tool the field
    /// leader and hands it the winner-relative DENOMINATOR. Scoring against the
    /// organisers' published adjudication has to leave that denominator with the
    /// tool that answered correctly, and must not charge NY for the same
    /// refuted witness.
    #[test]
    fn a_single_tool_witness_contradicting_published_truth_does_not_set_the_denominator() {
        let dir = TempDir::new().expect("tempdir");
        let current = planted_witness_current(&dir);
        let official = TempDir::new().expect("official");
        planted_witness_corpus(&official);
        let truth_dir = TempDir::new().expect("truth");
        let log = organiser_log(&truth_dir, &cgan_truth_log());

        // Arm A — the defect, pinned so the fix is measured against it.
        let mut broken = opts(current.clone(), None);
        broken.official = Some(official.path().to_path_buf());
        broken.year = Some(ScoreYear::Y2025);
        broken.witnesses = Witnesses::AssumedValid;
        let report = run_score(&broken).expect("assumed-valid score");
        let cgan = bench(&report, "cgan_2023");
        assert_eq!(
            cgan.field_best_raw_score,
            Some(40),
            "assumed-valid hands the denominator to the planted tool's four \
             unreproduced witnesses"
        );
        assert_eq!(
            cgan.current_raw_score,
            Some(3 * PENALTY_INCORRECT + POINTS_CORRECT)
        );
        assert_eq!(cgan.current_incorrect, Some(3));

        // Arm B — the fix. Same rows, same field, organiser truth supplied.
        let mut fixed = opts(current, None);
        fixed.official = Some(official.path().to_path_buf());
        fixed.year = Some(ScoreYear::Y2025);
        fixed.witnesses = Witnesses::Adjudicated;
        fixed.ground_truth = Some(log);
        let report = run_score(&fixed).expect("adjudicated score");
        let cgan = bench(&report, "cgan_2023");
        assert_eq!(
            cgan.field_best_raw_score,
            Some(50),
            "the denominator belongs to the tool that matches published truth"
        );
        assert_eq!(
            cgan.current_raw_score,
            Some(4 * POINTS_CORRECT),
            "NY's four unsat rows all match published truth"
        );
        assert_eq!(cgan.current_incorrect, Some(0));
        assert_eq!(cgan.current_disputed, Some(0));
        assert_eq!(cgan.truth_resolved_instances, Some(5));
        assert_eq!(report.ground_truth_resolved_instances, 5);
        assert!(
            report
                .ground_truth_source
                .as_deref()
                .is_some_and(|source| source.contains("results.txt")),
            "the truth's provenance must be reported: {:?}",
            report.ground_truth_source
        );
        // The verdict counts are untouched; only their price changed.
        assert_eq!(cgan.current_solved, 4);
    }

    /// With no published truth to appeal to, a lone contested SAT settles
    /// nothing — so it scores nobody, in EITHER direction, and is reported.
    #[test]
    fn an_uncorroborated_contested_sat_is_disputed_and_scores_no_participant() {
        let dir = TempDir::new().expect("tempdir");
        let current = planted_witness_current(&dir);
        let official = TempDir::new().expect("official");
        planted_witness_corpus(&official);

        let mut adjudicated = opts(current, None);
        adjudicated.official = Some(official.path().to_path_buf());
        adjudicated.year = Some(ScoreYear::Y2025);
        adjudicated.witnesses = Witnesses::Adjudicated;
        let report = run_score(&adjudicated).expect("adjudicated score");
        let cgan = bench(&report, "cgan_2023");

        assert_eq!(
            cgan.disputed_rows.len(),
            4,
            "each contested row is reported: {:?}",
            cgan.disputed_rows
        );
        assert_eq!(
            cgan.current_disputed,
            Some(3),
            "NY decided three of the four"
        );
        assert_eq!(
            cgan.current_raw_score,
            Some(POINTS_CORRECT),
            "only the uncontested row scores"
        );
        assert_eq!(cgan.current_incorrect, Some(0));
        assert_eq!(
            cgan.field_best_raw_score,
            Some(POINTS_CORRECT),
            "the lone witness cannot lift the planted tool above the field"
        );
        // The withheld rows must not read as a quiet gain.
        assert_eq!(
            cgan.current_raw_score_assumed_valid,
            Some(3 * PENALTY_INCORRECT + POINTS_CORRECT),
            "the assumed-valid comparison stays visible"
        );
        assert!(
            cgan.disputed_rows
                .iter()
                .all(|row| row.contains("planted") && row.contains("honest")),
            "a dispute must name BOTH sides: {:?}",
            cgan.disputed_rows
        );
        assert_eq!(report.total_disputed_instances, 4);
        assert_eq!(report.total_disputed, 3);
        assert!(report.ground_truth_source.is_none());
    }

    /// The fix is not an amnesty. A SAT two participants agree on is credible
    /// evidence and still convicts an UNSAT that contradicts it — including
    /// NY's own.
    #[test]
    fn a_corroborated_sat_still_convicts_a_contradicting_unsat() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "cgan_2023,onnx/m.onnx,vnnlib/p1.vnnlib,0,unsat,1\n",
        );
        let official = TempDir::new().expect("official");
        write_file(
            &official,
            "first/results.csv",
            "cgan_2023,onnx/m.onnx,vnnlib/p1.vnnlib,0,sat,1\n",
        );
        write_file(
            &official,
            "second/results.csv",
            "cgan_2023,onnx/m.onnx,vnnlib/p1.vnnlib,0,sat,1\n",
        );

        let mut adjudicated = opts(current, None);
        adjudicated.official = Some(official.path().to_path_buf());
        adjudicated.year = Some(ScoreYear::Y2025);
        adjudicated.witnesses = Witnesses::Adjudicated;
        let report = run_score(&adjudicated).expect("adjudicated score");
        let cgan = bench(&report, "cgan_2023");
        assert_eq!(cgan.current_raw_score, Some(PENALTY_INCORRECT));
        assert_eq!(cgan.current_incorrect, Some(1));
        assert_eq!(cgan.current_disputed, Some(0));
        assert_eq!((cgan.disputed_rows).len(), 0);
    }

    /// Published truth outranks corroboration in BOTH directions: when the
    /// organisers say `sat`, an UNSAT majority does not save a wrong answer.
    #[test]
    fn published_truth_outranks_a_disagreeing_majority() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "cgan_2023,onnx/m.onnx,vnnlib/p1.vnnlib,0,unsat,1\n",
        );
        let official = TempDir::new().expect("official");
        write_file(
            &official,
            "first/results.csv",
            "cgan_2023,onnx/m.onnx,vnnlib/p1.vnnlib,0,unsat,1\n",
        );
        write_file(
            &official,
            "second/results.csv",
            "cgan_2023,onnx/m.onnx,vnnlib/p1.vnnlib,0,unsat,1\n",
        );
        let truth_dir = TempDir::new().expect("truth");
        let log = organiser_log(
            &truth_dir,
            "Category 2025_cgan_2023:\nRow: ['m-p1', '1.0 (h)']\nTrue Result: sat\n",
        );

        let mut adjudicated = opts(current, None);
        adjudicated.official = Some(official.path().to_path_buf());
        adjudicated.year = Some(ScoreYear::Y2025);
        adjudicated.witnesses = Witnesses::Adjudicated;
        adjudicated.ground_truth = Some(log);
        let report = run_score(&adjudicated).expect("adjudicated score");
        let cgan = bench(&report, "cgan_2023");
        assert_eq!(
            cgan.current_raw_score,
            Some(PENALTY_INCORRECT),
            "no SAT witness exists anywhere in the field, yet the published truth \
             still convicts the contradicting UNSAT"
        );
        assert_eq!(cgan.current_incorrect, Some(1));
    }

    /// The organisers' row id drops directories, and `safenlp_2024` reuses
    /// property filenames across `medical/` and `ruarobot/`. Where that collapse
    /// puts two DIFFERENT published verdicts on one id, the id is unusable: it
    /// must be dropped and named, never resolved by taking whichever row parsed
    /// last. A vnnlib-only join once manufactured 123 phantom contradictions.
    #[test]
    fn an_ambiguous_organiser_row_id_is_dropped_and_named_not_guessed() {
        let log = "Category 2026_safenlp_2024:\n\
                   Row: ['perturbations_0-hyperrectangle_570', '1.0 (v)']\n\
                   True Result: sat\n\
                   Row: ['perturbations_0-hyperrectangle_570', '1.0 (h)']\n\
                   True Result: unsat\n\
                   Row: ['perturbations_0-hyperrectangle_1', '1.0 (h)']\n\
                   True Result: unsat\n";
        let truth = parse_ground_truth_log(log, "fixture").expect("parse truth");
        assert_eq!(
            truth.instance_count(),
            1,
            "only the unambiguous row survives"
        );
        assert!(truth.verdicts.contains_key(&(
            "safenlp_2024".to_string(),
            "perturbations_0-hyperrectangle_1".to_string()
        )));
        assert_eq!(
            truth.ambiguous.iter().cloned().collect::<Vec<_>>(),
            vec!["safenlp_2024 | perturbations_0-hyperrectangle_570".to_string()],
            "the conflicted id must be named, not silently absent"
        );
    }

    /// A truth table that parsed nothing is a parse failure, not a clean absence
    /// of truth — the same shape as an empty grep read as evidence.
    #[test]
    fn an_organiser_log_with_no_true_result_rows_is_a_parse_failure() {
        let error = parse_ground_truth_log(
            "Category 2025_cgan_2023:\nsomething else entirely\n",
            "fixture",
        )
        .expect_err("empty truth must refuse");
        assert!(
            error.to_string().contains("PARSE FAILURE"),
            "the refusal must say why: {error}"
        );
    }

    /// `True Result: -` is the organisers recording that they have no verdict.
    /// It must claim nothing rather than being read as a decided row.
    #[test]
    fn an_undecided_organiser_row_claims_no_truth() {
        let log = "Category 2025_cgan_2023:\n\
                   Row: ['m-p1', '1.0 (v)']\n\
                   True Result: -\n\
                   Row: ['m-p2', '1.0 (h)']\n\
                   True Result: unsat\n";
        let truth = parse_ground_truth_log(log, "fixture").expect("parse truth");
        assert_eq!(truth.instance_count(), 1);
        assert!(truth.ambiguous.is_empty());
        assert!(!truth
            .verdicts
            .contains_key(&("cgan_2023".to_string(), "m-p1".to_string())));
    }

    /// A released corpus carries its own scoring log, and pointing `--official`
    /// at it must find that log without a second flag.
    #[test]
    fn organiser_truth_is_found_inside_an_unsanitized_official_corpus() {
        let dir = TempDir::new().expect("tempdir");
        let current = planted_witness_current(&dir);
        let official = TempDir::new().expect("official");
        planted_witness_corpus(&official);
        organiser_log(&official, &cgan_truth_log());

        let mut adjudicated = opts(current, None);
        adjudicated.official = Some(official.path().to_path_buf());
        adjudicated.year = Some(ScoreYear::Y2025);
        adjudicated.witnesses = Witnesses::Adjudicated;
        let report = run_score(&adjudicated).expect("adjudicated score");
        assert_eq!(report.ground_truth_resolved_instances, 5);
        let cgan = bench(&report, "cgan_2023");
        assert_eq!(cgan.field_best_raw_score, Some(50));
        assert_eq!(cgan.current_disputed, Some(0));
    }

    /// Assumed-valid witnesses penalize a contradicted UNSAT by -150;
    /// the published 2025 board's unvalidated model retains its +10.
    #[test]
    fn the_witness_model_decides_whether_a_contradicted_unsat_is_punished() {
        let dir = TempDir::new().expect("tempdir");
        // NY says unsat; the field says sat on the same instance.
        let current = write_file(
            &dir,
            "current.csv",
            "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,1\n",
        );
        let official = TempDir::new().expect("official");
        write_file(
            &official,
            "winner/acasxu_2023/results.csv",
            "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,0,sat,1\n",
        );

        let mut punished = opts(current.clone(), None);
        punished.official = Some(official.path().to_path_buf());
        punished.year = Some(ScoreYear::Y2025);
        punished.witnesses = Witnesses::AssumedValid;
        let report = run_score(&punished).expect("assumed-valid score");
        let bench = report
            .benchmarks
            .iter()
            .find(|b| b.benchmark == "acasxu_2023")
            .expect("acasxu present");
        assert_eq!(bench.current_raw_score, Some(PENALTY_INCORRECT));
        assert_eq!(bench.current_incorrect, Some(1));

        let mut published = opts(current, None);
        published.official = Some(official.path().to_path_buf());
        published.year = Some(ScoreYear::Y2025);
        published.witnesses = Witnesses::Unvalidated;
        let report = run_score(&published).expect("unvalidated score");
        let bench = report
            .benchmarks
            .iter()
            .find(|b| b.benchmark == "acasxu_2023")
            .expect("acasxu present");
        assert_eq!(
            bench.current_raw_score,
            Some(POINTS_CORRECT),
            "with counterexamples never opened, a contradicted UNSAT still earns +10 — \
             this is how the published board scored, not an endorsement of it"
        );
        assert_eq!(bench.current_incorrect, Some(0));
    }

    /// A scored category with no measured rows is a guaranteed zero, and the
    /// usual cause is an unsupplied bank directory. Scoring only
    /// `reports/measured/` and not `reports/measured-ext/` under-counted the
    /// extended track by 182 normalized points, silently.
    #[test]
    fn a_scored_category_with_no_measured_rows_is_flagged_not_silently_zeroed() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,1\n",
        );
        let official = TempDir::new().expect("official");
        write_file(
            &official,
            "winner/acasxu_2023/results.csv",
            "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,1\n",
        );
        // Scored, present in the field, but absent from the supplied bank.
        write_file(
            &official,
            "winner/cersyve/results.csv",
            "cersyve,onnx/c.onnx,vnnlib/c.vnnlib,0,unsat,1\n",
        );

        let mut o = opts(current, None);
        o.official = Some(official.path().to_path_buf());
        o.year = Some(ScoreYear::Y2025);
        let report = run_score(&o).expect("score");
        assert_eq!(
            report.scored_categories_with_no_rows,
            vec!["cersyve".to_string()],
            "a scored category with zero rows must be named, since it is \
             indistinguishable from a real zero in the totals"
        );
        // The measured category is not flagged.
        assert!(!report
            .scored_categories_with_no_rows
            .contains(&"acasxu_2023".to_string()));
    }

    /// EVERY TRACKED BANK MUST PARSE. Standing guard for damage that happened.
    ///
    /// `reports/measured/cctsdb_yolo_2023.csv` was written with the 5-column ext
    /// layout into a 6-column file, putting the verdict in the `prepare_time`
    /// slot. Field 3 then failed to parse as f64 and the load died on line 1 --
    /// and because `read_results_dir` walks paths in sorted order, that was the
    /// SECOND file read and it aborted the ENTIRE load. `ny benchmarks score`
    /// could not run at all, in any configuration, for as long as that row
    /// existed, so every standing quoted meanwhile came from somewhere else.
    ///
    /// Deliberately narrow. It does NOT compare verdicts across banks: the
    /// scorer already refuses to merge banks that genuinely disagree, and a
    /// naive comparison here would fire on `timeout` vs `unknown`, which are
    /// both undecided rather than contradictory. One unparseable file is the
    /// failure that silences everything, so that is what this pins.
    #[test]
    fn every_tracked_results_bank_parses() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let mut tracked_banks = 0usize;
        let mut tracked_rows = 0usize;
        for bank in ["reports/measured", "reports/measured-ext"] {
            let dir = root.join(bank);
            assert!(
                dir.is_dir(),
                "required tracked results bank is missing: {}; a partial checkout must not \
                 report that the repository's score banks passed validation",
                dir.display()
            );
            let rows = read_results_dir(&dir).unwrap_or_else(|e| {
                panic!(
                    "{bank} failed to load: {e}\n\n\
                     A single unparseable bank file silently disables scoring for \
                     EVERY bank, because the reader aborts on the first bad one. \
                     Check the column layout: the 6-column form is \
                     category,onnx,vnnlib,prepare_time,result,time and the \
                     5-column form omits prepare_time. Never write one bank's \
                     layout into the other's file, and never write bank rows by \
                     row POSITION -- join on the (onnx, vnnlib) instance key."
                )
            });
            assert!(
                !rows.is_empty(),
                "tracked results bank {} contains zero parseable rows; parsing an empty \
                 directory is not score coverage",
                dir.display()
            );
            tracked_banks += 1;
            tracked_rows += rows.len();
        }
        assert_eq!(
            tracked_banks, 2,
            "both tracked score-bank roots must be checked"
        );
        assert!(
            tracked_rows > 0,
            "tracked score-bank validation checked zero rows"
        );
    }

    #[test]
    fn omitting_budget_years_states_that_runtimes_went_unchecked() {
        let dir = TempDir::new().expect("tempdir");
        let current = write_file(
            &dir,
            "current.csv",
            "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,99999\n",
        );
        let report = run_score(&opts(current, None)).expect("score");
        assert_eq!(
            report.unchecked_budget_caveat,
            Some(UNCHECKED_BUDGET_CAVEAT)
        );
        assert_eq!(report.total_budget_checked, 0);
        assert_eq!(report.total_over_budget, 0);
    }
}
