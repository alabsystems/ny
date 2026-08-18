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
//! Plain result CSVs do not retain organizer witness-validation outcomes.
//! Consistent with `scripts/vnncomp_competitive_score.py`, every SAT row in
//! this raw-CSV model is therefore assumed to carry a strictly valid witness.
//! This exposes contradictory UNSAT rows to the real -150 penalty, but it is
//! still a modeled score rather than an organizer-validated scoreboard.
//!
//! # Budget enforcement
//!
//! `--budget-year` loads the official per-instance timeouts from that corpus's
//! `instances.csv` files and re-checks every solved row's recorded runtime
//! against its own budget. This is not a formality: VNN-COMP timeouts are
//! **per instance**, not per benchmark, and nn4sys alone spans 20 s to 800 s.
//! A solved row recorded above its budget is a *phantom point* — the real
//! harness kills it at the timeout and awards zero — so such rows are reported
//! and EXCLUDED from the modeled score rather than silently banked as progress.
//!
//! Coverage is reported alongside the violations. A row whose identity has no
//! budget in the supplied corpora is counted as unchecked, never as passing;
//! "0 over budget" out of 0 checked is not a clean bill of health.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::bench_vnncomp::{discover_categories, parse_csv_line};
use super::vnncomp_2025_tracks::{EXTENDED_TRACK_2025, REGULAR_TRACK_2025};
use super::vnncomp_2026_tracks::{EXTENDED_TRACK_2026, REGULAR_TRACK_2026};
use super::vnncomp_reseed::model_identity_entries;
use super::vnncomp_sweep::parse_instances;

const POINTS_CORRECT: i64 = 10;
const PENALTY_INCORRECT: i64 = -150;
const MODELED_SCORE_CAVEAT: &str = "raw CSV: SAT witnesses are assumed valid and are not \
replayed; normalized identity matching does not reproduce organizer cross-version positional \
checks";
const MODELED_2026_SCORE_CAVEAT: &str = "projection only: no released organizer VNN-COMP 2026 \
team-result corpus or generated score tables are bound; supplied raw CSVs assume SAT witnesses \
valid and do not reproduce organizer cross-version \
positional checks";
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
///
/// This choice moves the answer by more than 100 points, so it is explicit.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Witnesses {
    /// Every SAT row is assumed to carry a strictly valid counterexample, so an
    /// UNSAT contradicted by one takes the real -150. Conservative: it is the
    /// model that punishes NY for a wrong answer, and the default.
    AssumedValid,
    /// No counterexample is ever opened, so nothing convicts a contradicting
    /// UNSAT and every claimed verdict earns +10.
    ///
    /// This reproduces the **published** VNN-COMP 2025 board numerically.
    /// Validated against the organizers' own
    /// `SCORING-ZERO-TOL/plots_scored/latex/total.tex`: this model returns
    /// alpha-beta-CROWN 1566.9 and PyRAT 1228.4, matching exactly.
    ///
    /// CORRECTION (2026-08-02): an earlier version of this comment justified the
    /// match by claiming the published run "scored with `SKIP_CE_FILES = True`",
    /// i.e. never validated counterexamples. That inference is FALSE and is
    /// retracted. `settings.py:63` sets `ALWAYS_CHECK_COUNTEREXAMPLES = True`;
    /// `SKIP_CE_FILES` only relaxes a file-exists `assert` at
    /// `process_results.py:330` and does not disable checking. The published run
    /// checked 5,744 counterexamples across 1,197 instances and applied 62 live
    /// -150 penalties (`results.txt:111602`). The numerical agreement above is
    /// real; only the old explanation for it was wrong.
    ///
    /// Use it to answer "where would NY have placed", and ONLY that. It rewards
    /// unsound tools: on cifar100 it makes NNV the winner at 190/200 over
    /// alpha-beta-CROWN's 129, while NNV holds 19 UNSAT rows that another tool
    /// contradicts with a SAT witness and every other tool holds none.
    Unvalidated,
}

impl Witnesses {
    fn label(self) -> &'static str {
        match self {
            Self::AssumedValid => "SAT witnesses assumed valid (contradicted UNSAT scores -150)",
            Self::Unvalidated => {
                "counterexamples not opened by this model (reproduces the published totals)"
            }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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
    /// Zero-based occurrence among identical rows in one result file.
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
}

type SemanticKey = (String, String, String);

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
    /// Explicit `2025_`/`2026_` category prefix, when the source uses one.
    ///
    /// Most released CSVs use unprefixed ids. Retaining an explicit prefix
    /// lets an explicitly selected scoring year reject accidentally mixed
    /// corpora even though the canonical instance key is prefix-free.
    year_hint: Option<ScoreYear>,
    /// Recorded wall-clock seconds, already validated finite and non-negative.
    seconds: f64,
    /// Run tag from a measured-7 row (`...,run_id`), when present. A row
    /// without one cannot tie its verdict back to a sealed run, which matters
    /// when its runtime is otherwise indistinguishable from a hand-edited
    /// value (see `audit_budgets` on at-budget solves).
    tag: Option<String>,
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
                verdict: Verdict::parse(&fields[schema.verdict_index()])
                    .with_context(|| format!("{}:{line_number}", path.display()))?,
                source,
                year_hint,
                seconds,
                tag,
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
    /// Present when no `--budget-year` was supplied, so runtimes went unchecked.
    pub(crate) unchecked_budget_caveat: Option<&'static str>,
}

#[derive(Debug)]
struct ModeledOfficialScore {
    current_raw: i64,
    current_correct: usize,
    current_incorrect: usize,
    field_best_raw: i64,
    normalized: f64,
}

fn score_rows(
    rows: &ResultRows,
    scored_instances: &BTreeSet<InstanceKey>,
    valid_sat_witnesses: &BTreeSet<SemanticKey>,
    over_budget: &BTreeSet<InstanceKey>,
) -> (i64, usize, usize) {
    let mut raw = 0_i64;
    let mut correct = 0_usize;
    let mut incorrect = 0_usize;
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
        match entry.verdict {
            Verdict::Sat => {
                raw += POINTS_CORRECT;
                correct += 1;
            }
            Verdict::Unsat if valid_sat_witnesses.contains(&key.semantic_key()) => {
                raw += PENALTY_INCORRECT;
                incorrect += 1;
            }
            Verdict::Unsat => {
                raw += POINTS_CORRECT;
                correct += 1;
            }
            Verdict::Undecided => {}
        }
    }
    (raw, correct, incorrect)
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
/// `NY_MEASURE_CAP=120` default in `scripts/measure_ny_scorecard.sh` produced
/// against 900s cgan budgets), so it must not be read as a capability limit.
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
    year: ScoreYear,
    track: ScoreTrack,
    over_budget: &BTreeSet<InstanceKey>,
    witnesses: Witnesses,
) -> Option<ModeledOfficialScore> {
    if !track.includes(year, benchmark) {
        return None;
    }
    let scored_instances = field
        .values()
        .flat_map(|rows| rows.keys())
        .filter(|key| key.benchmark == benchmark)
        .cloned()
        .collect::<BTreeSet<_>>();
    if scored_instances.is_empty() {
        return None;
    }

    // Which SAT rows are treated as counterexamples that can convict a
    // contradicting UNSAT. Under `Witnesses::Unvalidated` nothing convicts,
    // because the published board never opened the counterexample files.
    let valid_sat_witnesses = match witnesses {
        Witnesses::AssumedValid => current
            .iter()
            .chain(field.values().flat_map(|rows| rows.iter()))
            .filter(|(key, entry)| key.benchmark == benchmark && entry.verdict == Verdict::Sat)
            .map(|(key, _)| key.semantic_key())
            .collect::<BTreeSet<_>>(),
        Witnesses::Unvalidated => BTreeSet::new(),
    };

    let (current_raw, current_correct, current_incorrect) = score_rows(
        current,
        &scored_instances,
        &valid_sat_witnesses,
        over_budget,
    );
    // The official field's runtimes came from the real harness, which enforces
    // the timeout itself, so no exclusion applies to the competitors' rows. Only
    // locally measured results can be over budget.
    let no_exclusions = BTreeSet::new();
    let mut field_best_raw = current_raw;
    for rows in field.values() {
        let (raw, _, _) = score_rows(
            rows,
            &scored_instances,
            &valid_sat_witnesses,
            &no_exclusions,
        );
        field_best_raw = field_best_raw.max(raw);
    }
    let normalized = if field_best_raw > 0 {
        (100.0 * current_raw as f64 / field_best_raw as f64).max(0.0)
    } else {
        0.0
    };
    Some(ModeledOfficialScore {
        current_raw,
        current_correct,
        current_incorrect,
        field_best_raw,
        normalized,
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
    /// How SAT rows are treated. Worth >100 points, so never implicit.
    pub(crate) witnesses: Witnesses,
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
    let official = opts
        .official
        .as_deref()
        .map(load_official_field)
        .transpose()?;
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

    let budgets = if opts.budget_years.is_empty() {
        None
    } else {
        let mut years = opts.budget_years.clone();
        years.sort_unstable();
        years.dedup();
        Some(load_budgets(&years)?)
    };

    let mut report = ProgressReport {
        modeled_score_caveat: official.as_ref().map(|_| match score_year {
            Some(ScoreYear::Y2026) => MODELED_2026_SCORE_CAVEAT,
            _ => MODELED_SCORE_CAVEAT,
        }),
        score_year: official.as_ref().and(score_year),
        score_track: official.as_ref().map(|_| opts.track),
        unchecked_budget_caveat: budgets
            .as_ref()
            .map_or(Some(UNCHECKED_BUDGET_CAVEAT), |_| None),
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
                score_year.expect("official score year validated above"),
                opts.track,
                &over_budget_keys,
                opts.witnesses,
            )
        }) {
            progress.current_raw_score = Some(score.current_raw);
            progress.current_correct = Some(score.current_correct);
            progress.current_incorrect = Some(score.current_incorrect);
            progress.field_best_raw_score = Some(score.field_best_raw);
            progress.normalized = Some(score.normalized);
            normalized_total += score.normalized;
            report.officially_scored_benchmarks += 1;
            if progress.rows_compared == 0 {
                report
                    .scored_categories_with_no_rows
                    .push(benchmark.clone());
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
            if opts.witnesses == Witnesses::Unvalidated {
                println!(
                    "  this reproduces the PUBLISHED 2025 board (validated: alpha-beta-CROWN \
                     1566.9, PyRAT 1228.4) and therefore rewards unsound tools — on cifar100 it \
                     ranks NNV first at 190/200 while NNV holds 19 UNSAT rows another tool \
                     contradicts with a SAT witness."
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
            "\n*** {} SCORED CATEGOR{} HAVE NO MEASURED ROWS and therefore score a guaranteed \
             ZERO: {}. The usual cause is a bank directory that was not supplied, NOT an absence \
             of results — this repository keeps TWO banks, `reports/measured/` and \
             `reports/measured-ext/`, and scoring only the first silently under-counted the \
             extended track by 182 normalized points. Pass every bank with repeated --results \
             before reading a total. ***",
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
             their own official per-instance timeout. These rows were never given the time \
             the competition gives them, so they say NOTHING about capability — do not \
             prioritize (or deprioritize) work from them. Re-measure each at the budget \
             from instances.csv (`ny benchmarks run` does; scripts/measure_ny_scorecard.sh \
             caps at NY_MEASURE_CAP=120s by default). ***",
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
        }
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
        let witnesses = BTreeSet::new();
        let (raw, correct, _) = score_rows(&rows, &scored, &witnesses, &audit.keys);
        assert_eq!(correct, 1, "only the in-budget row scores");
        assert_eq!(raw, POINTS_CORRECT);
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
        let witnesses = rows.keys().map(InstanceKey::semantic_key).collect();

        // Past the timeout the harness reports a timeout, so the row is neither
        // a solve nor a wrong answer.
        let (raw, correct, incorrect) = score_rows(&rows, &scored, &witnesses, &audit.keys);
        assert_eq!((raw, correct, incorrect), (0, 0, 0));

        // Without the exclusion the same contradiction is the real -150.
        let (raw, _, incorrect) = score_rows(&rows, &scored, &witnesses, &BTreeSet::new());
        assert_eq!((raw, incorrect), (PENALTY_INCORRECT, 1));
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
        // The defect this guards: scripts/measure_ny_scorecard.sh's default
        // NY_MEASURE_CAP=120 banked ~101s "timeouts" against 800-1200s budgets,
        // which were then read as capability limits.
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

    /// The witness model is worth more than 100 normalized points, so both arms
    /// are pinned. `unvalidated` is what the published 2025 board did, and it is
    /// exactly what lets an unsound UNSAT keep its +10.
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
