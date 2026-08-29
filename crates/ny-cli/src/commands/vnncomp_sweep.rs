// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny benchmarks run` — sweep a VNN-COMP corpus and emit an official-format
//! `results.csv`.
//!
//! This is the supported way to run a whole benchmark category. It replaces the
//! ad-hoc shell loops that previously lived in `scripts/` and in scratch
//! directories, and it fixes the two failure modes those loops kept producing:
//!
//! * **silent argument rejection** — a mistyped flag made every instance exit
//!   instantly and record `timeout`, which reads exactly like a real negative
//!   result. This runner accepts a verdict only after a successful child exit
//!   and an explicit protocol token; every partial, missing, or invalid result
//!   is `error`, with a bounded stderr tail retained for diagnosis.
//! * **budget drift** — loops hard-coded a cap and silently under-measured
//!   benchmarks whose official budget is larger. Here the per-instance budget
//!   comes from `instances.csv`; `--timeout-cap` may only LOWER it, and when it
//!   does, the emitted CSV and the summary both say so.
//! * **environment drift** — the scored path is
//!   `vnncomp_scripts/run_instance.sh`, which exports the deterministic
//!   throughput setting before calling `ny vnncomp`. A sweep that omits it
//!   measures a different performance environment than the submission.
//!   Verdict-affecting lane settings travel through typed presets instead.
//!   `submission_environment` below replicates the wrapper, and
//!   `sweep_environment_matches_the_submission_wrapper` parses the wrapper and
//!   fails when the two drift apart.
//!
//! Layouts handled (both are real in-tree today):
//! * flat — `benchmarks/vnncomp2025/benchmarks/<category>/instances.csv`
//! * versioned — `benchmarks/vnncomp2026/benchmarks/<category>/<version>/instances.csv`
//!
//! Instance row formats handled:
//! * single model — `onnx,vnnlib,budget`
//! * paired/relational — `"[('f', 'a.onnx'), ('g', 'b.onnx')]",vnnlib,budget`
//!   (used by `isomorphic_acasxu_2026` and `monotonic_acasxu_2026`). The field
//!   is forwarded verbatim, joined to the category directory, exactly as the
//!   organizers' `run_all_categories.sh` does — `ny vnncomp` parses the literal
//!   and resolves its members relative to that directory.
//!
//! Each instance runs in its own child process (`ny vnncomp`). That is
//! deliberate, not a shortcut: a verifier crash, a driver fault, or an OOM kill
//! on one instance must not abort the sweep, and the competition harness
//! isolates instances the same way.
//!
//! Every banked `sat` row RETAINS its witness (#witness-retention-gap): the
//! `counterexample_vnnlib` block the child appended to its result file (the
//! run_instance protocol shape, `sat\n<witness>`) is copied out of the
//! per-instance scratch into `<output-stem>.witnesses/<category>/` before that
//! scratch is deleted, and the row's metadata records the bank-relative path
//! plus sha256. A sat row whose witness could not be extracted or copied banks
//! an explicit `"witness": null` and is counted in the sweep summary —
//! visible, never silent. The witness directory's content is sealed into the
//! manifest at sweep end.

use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use prost::Message;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
/// Content-addressed verdict cache (#sweep-cache): turns an arm x row matrix into
/// new-pairs-only. See the module docs for the two admission rules, both of which
/// encode a class of fake result this repository has produced.
pub(crate) mod cache;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use super::bench_vnncomp::{discover_categories, instances_csv_for, parse_csv_line, VnnlibVersion};
use super::vnncomp_benchmarks::run_bounded_child;

const CHILD_STDERR_TAIL_BYTES: usize = 16 * 1024;
const CHILD_WATCHDOG_GRACE: Duration = Duration::from_secs(30);
const RESULT_FIRST_LINE_BYTES: u64 = 4096;

/// One instance row, already resolved against its category directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SweepInstance {
    /// The ONNX field verbatim from `instances.csv` (may be a paired literal).
    pub(crate) onnx_field: String,
    /// The VNN-LIB field verbatim from `instances.csv`.
    pub(crate) vnnlib_field: String,
    /// Official per-instance budget in whole seconds.
    pub(crate) budget_secs: u64,
}

fn parse_budget_secs(raw: &str, lineno: usize) -> Result<u64> {
    let value = raw
        .parse::<f64>()
        .with_context(|| format!("instances.csv line {lineno}: unparsable budget {raw:?}"))?;
    if !value.is_finite() || value <= 0.0 || value >= u64::MAX as f64 || value.fract() != 0.0 {
        bail!(
            "instances.csv line {lineno}: budget must be finite, positive, integral, and \
             representable as whole seconds; got {raw:?}"
        );
    }
    let seconds = value as u64;
    if seconds as f64 != value {
        bail!(
            "instances.csv line {lineno}: budget is not exactly representable as whole \
             seconds; got {raw:?}"
        );
    }
    Ok(seconds)
}

/// Unconditional environment exported by `vnncomp_scripts/run_instance.sh`
/// before it execs `ny vnncomp`.
///
/// A sweep MUST reproduce this or it measures a different verifier than the one
/// being submitted. Historically the `safenlp_2024` upfront lane lived only in
/// a wrapper export; it now travels through the frozen typed preset instead.
///
/// `NY_MARGIN_ROW_CONV_BWD_BLOCKED=1` / `NY_MARGIN_ROW_PARALLEL=1` were removed
/// from this list (and from the wrapper and the scorecard script) because 1 has
/// been the compiled default since 7b004fba (parallel frontier) and 2eaa6b13
/// (cache-blocked backward conv): exporting =1 was a bit-exact no-op — MEASURED
/// by `margin_row/tests.rs`, which asserts unset == `"1"` for both
/// `blocked_backward_enabled_from_env` and `margin_row_frontier_from_env`. The
/// `=0` serial kill switches in ny-propagate remain available for A/Bs.
///
/// Category-scoped wrapper settings deliberately do NOT belong here. They belong
/// in the category's preset, where every entry point picks them up — and that is
/// now literally true, not aspirational: `attack.upfront_attack`
/// (`configs/vnncomp26/safenlp_2024.yaml`) is consumed by
/// `commands::vnncomp::upfront_wrapper_route` (`ForcedByPreset`), so a sweep
/// arms the same lane the submission wrapper arms without any category-fenced
/// export. The redundant SafeNLP shell force was removed once the same frozen
/// LeverSet became the execution and receipt authority.
pub(crate) fn submission_environment() -> &'static [(&'static str, &'static str)] {
    &[
        // ny uses faer/ndarray matrixmultiply, not OpenBLAS; single-threaded OMP
        // keeps it from fighting rayon's pool.
        ("OMP_NUM_THREADS", "1"),
    ]
}

/// Parse an `instances.csv` body into instance rows.
///
/// Blank lines and `#` comments are skipped. A row whose budget field is absent
/// or unparsable is REFUSED rather than defaulted: silently substituting a
/// budget is how a sweep ends up measuring something other than the competition.
pub(crate) fn parse_instances(body: &str) -> Result<Vec<SweepInstance>> {
    let mut out = Vec::new();
    for (lineno, raw) in body.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = parse_csv_line(line)
            .with_context(|| format!("instances.csv line {}: malformed CSV", lineno + 1))?;
        if fields.len() != 3 {
            bail!(
                "instances.csv line {}: expected exactly 3 fields \
                 (onnx,vnnlib,budget), got {}: {line}",
                lineno + 1,
                fields.len()
            );
        }
        if fields[0].is_empty() || fields[1].is_empty() {
            bail!(
                "instances.csv line {}: model and property fields must be non-empty",
                lineno + 1
            );
        }
        let budget_raw = fields[2].trim();
        let budget_secs = parse_budget_secs(budget_raw, lineno + 1)?;
        out.push(SweepInstance {
            onnx_field: fields[0].clone(),
            vnnlib_field: fields[1].clone(),
            budget_secs,
        });
    }
    Ok(out)
}

/// One `--instances` selector: a row identity from `instances.csv`, prefixed
/// with the category that owns it.
type InstanceSelector = (String, String, String);

/// Cross-tool schema for the semantic identity of an exact row selector.
///
/// `scripts/ny_search.py` uses the same schema and canonical JSON payload, so
/// it can compare the selector it requested with the selector the native sweep
/// sealed without depending on a temporary file's path or byte layout.
const INSTANCE_SELECTOR_IDENTITY_SCHEMA: &str = "ny-search/instance-selectors/v1";

/// Cross-tool schema for the exact selected row set materialized from a plan.
const MEASURED_ROW_SET_IDENTITY_SCHEMA: &str = "ny-search/measured-row-set/v1";

/// Schema for the explicit resolved-file companion to the semantic row set.
const SELECTED_ROW_INPUTS_SCHEMA: &str = "ny-cli/selected-row-inputs/v2";
const ONNX_CACHE_INPUT_SCHEMA: &str = "ny-cli/onnx-cache-input/v1";

/// Parse an `--instances` file into row selectors.
///
/// One `category,onnx,vnnlib` triple per line — the first three columns of the
/// `results.csv` this runner emits, so a subset can be cut straight out of a
/// bank. Blank lines and `#` comments are skipped. Fields are CSV-parsed rather
/// than split, so a paired/relational ONNX literal (which contains commas)
/// survives, and they are trimmed exactly as `parse_instances` trims the
/// `instances.csv` they must match.
///
/// The category is REQUIRED because the sets this flag exists to run span
/// categories (the moat gate's 41 rows cover six: blocks A, B and C of §3 of
/// docs/VNNCOMP_MEASUREMENT_PLATFORM_DESIGN.md are all `cifar100_2024`, then
/// `soundnessbench` and the four in block E), and a bare model/property pair
/// cannot say which. The full pair is required for the same reason a
/// property basename is refused: `safenlp_2024` nests `hyperrectangle_883.vnnlib`
/// under two directories whose verdicts DISAGREE, and keying on the basename
/// collapsed them into one row twice, each time producing a false moat alarm
/// (docs/MOAT_AUDIT_VS_STRICT_FIELD_CE_2026-08-10.md).
fn parse_instance_selectors(body: &str) -> Result<Vec<InstanceSelector>> {
    let mut out: Vec<InstanceSelector> = Vec::new();
    for (index, raw) in body.lines().enumerate() {
        let lineno = index + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields =
            parse_csv_line(line).with_context(|| format!("line {lineno}: malformed CSV"))?;
        if fields.len() != 3 {
            bail!(
                "line {lineno}: expected exactly 3 fields (category,onnx,vnnlib), got {}: {line}",
                fields.len()
            );
        }
        if fields.iter().any(String::is_empty) {
            bail!("line {lineno}: category, model, and property fields must be non-empty");
        }
        let selector = (fields[0].clone(), fields[1].clone(), fields[2].clone());
        if out.contains(&selector) {
            bail!(
                "line {lineno}: {},{},{} is listed twice; a row subset must say exactly \
                 which rows it means",
                selector.0,
                selector.1,
                selector.2
            );
        }
        out.push(selector);
    }
    if out.is_empty() {
        bail!(
            "the file names no rows. An empty subset would run nothing and report a clean \
             sweep, which is indistinguishable from a passing one"
        );
    }
    Ok(out)
}

/// Refuse a subset that did not fully resolve against the selected corpus.
///
/// A named row missing from `instances.csv` is an ERROR, never a skip: a gate
/// set that quietly shrinks reports the same "all rows fine" as one that ran in
/// full, which is exactly how a regression hides.
fn ensure_every_selector_matched(
    selectors: &[InstanceSelector],
    matched: &HashSet<InstanceSelector>,
    selected_categories: &[String],
) -> Result<()> {
    const SHOWN: usize = 10;
    let missing = selectors
        .iter()
        .filter(|selector| !matched.contains(*selector))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }
    let rendered = missing
        .iter()
        .take(SHOWN)
        .map(|(category, onnx, vnnlib)| format!("{category},{onnx},{vnnlib}"))
        .collect::<Vec<_>>()
        .join("; ");
    let elided = missing.len().saturating_sub(SHOWN);
    bail!(
        "--instances names {} row(s) that no selected category's instances.csv contains: {}{}. \
         Fields must match the CSV verbatim, and the category must be among the swept ones \
         [{}].",
        missing.len(),
        rendered,
        if elided > 0 {
            format!("; and {elided} more")
        } else {
            String::new()
        },
        selected_categories.join(", ")
    );
}

/// Whether one corpus row belongs to this invocation's requested subset.
///
/// Exact selectors are identities, not occurrence selectors: if an official
/// `instances.csv` repeats the same model/property pair, selecting that pair
/// selects every occurrence. The full `instances.csv` fingerprint in the
/// manifest pins how many such occurrences exist and where they occur.
fn instance_is_selected(
    wanted: Option<&HashSet<InstanceSelector>>,
    identity: &InstanceSelector,
    index: usize,
    take: usize,
) -> bool {
    wanted.map_or(index < take, |wanted| wanted.contains(identity))
}

/// Quote a CSV field when it contains a comma, quote, or newline.
///
/// REQUIRED for paired/relational rows: their ONNX field is a python literal
/// containing commas, so writing it bare produces a row that neither the
/// organizers' tooling nor this runner's own `--resume` can parse back.
pub(crate) fn csv_escape(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// Verdict of one instance, in the organizers' vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SweepVerdict {
    Sat,
    Unsat,
    Timeout,
    Unknown,
    Error,
}

impl SweepVerdict {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Sat => "sat",
            Self::Unsat => "unsat",
            Self::Timeout => "timeout",
            Self::Unknown => "unknown",
            Self::Error => "error",
        }
    }

    fn parse_token(token: &str) -> Option<Self> {
        match token.trim() {
            "sat" => Some(Self::Sat),
            "unsat" => Some(Self::Unsat),
            "timeout" => Some(Self::Timeout),
            "unknown" => Some(Self::Unknown),
            "error" => Some(Self::Error),
            _ => None,
        }
    }

    /// Classify a child run. `ran_ok` is the child's exit status.
    ///
    /// A verdict is accepted only from a successful child that explicitly
    /// writes one of the four protocol tokens. Non-success, missing/empty
    /// output, and unrecognized tokens are harness errors; none may masquerade
    /// as a timeout or unknown result.
    pub(crate) fn classify(result_first_line: Option<&str>, ran_ok: bool) -> Self {
        if !ran_ok {
            return Self::Error;
        }
        result_first_line
            .and_then(Self::parse_token)
            .unwrap_or(Self::Error)
    }
}

/// Retained SAT witness provenance for one banked row (#witness-retention-gap).
///
/// `path` is relative to the bank directory (the official CSV's parent), so a
/// bank moved or archived as a unit stays organizer-replayable.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct WitnessRecord {
    pub(crate) path: String,
    pub(crate) sha256: String,
}

/// Witness disposition of one banked SAT row.
///
/// `Retained` serializes as the record object; `Missing` serializes as an
/// EXPLICIT `null` — the visible-never-silent marker for a sat row whose
/// witness could not be extracted or copied. Non-sat rows (and pre-retention
/// banks) omit the field entirely, which is the third, absent state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WitnessDisposition {
    Retained(WitnessRecord),
    Missing,
}

impl Serialize for WitnessDisposition {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Retained(record) => record.serialize(serializer),
            Self::Missing => serializer.serialize_none(),
        }
    }
}

/// `skip_serializing_if` predicate for a default-false flag.
fn is_false(value: &bool) -> bool {
    !*value
}

/// Deserialize a PRESENT `witness` key, keeping it distinct from an absent
/// one: `"witness": null` becomes `Some(Missing)`, an object becomes
/// `Some(Retained)`, and a missing key stays `None` via `#[serde(default)]`.
fn witness_if_present<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<WitnessDisposition>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<WitnessRecord>::deserialize(deserializer).map(|witness| {
        Some(match witness {
            Some(record) => WitnessDisposition::Retained(record),
            None => WitnessDisposition::Missing,
        })
    })
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct SweepRow {
    pub(crate) category: String,
    pub(crate) onnx: String,
    pub(crate) vnnlib: String,
    /// Zero-based occurrence among identical category/model/property rows.
    ///
    /// This makes retries stable even when an `instances.csv` intentionally
    /// repeats the same semantic instance.
    #[serde(default)]
    pub(crate) occurrence: usize,
    /// Zero-based row position in this category's pinned `instances.csv`.
    #[serde(default)]
    pub(crate) instance_index: usize,
    pub(crate) verdict: SweepVerdict,
    pub(crate) seconds: f64,
    pub(crate) budget_secs: u64,
    /// Set when the budget actually applied was below the official one.
    pub(crate) capped_from: Option<u64>,
    /// Short diagnostic detail for harness errors or outer-watchdog timeouts.
    pub(crate) detail: Option<String>,
    /// The child's flight-recorder sidecar (`<result>.flight.json`,
    /// #flight-record), embedded verbatim so each bank row carries its own
    /// execution trace — the sweep scratch tempdir that held the sidecar is
    /// deleted when the sweep ends. Absent when the child wrote no sidecar or
    /// wrote an unparsable one; either way the row itself must never fail
    /// (the recorder is best-effort by contract, and so is this copy).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) flight: Option<serde_json::Value>,
    /// #arm-overrides: the lever assignment this row was measured under, sorted.
    ///
    /// A verdict is only interpretable together with the configuration that
    /// produced it. Recording the arm ON THE ROW lets legacy manifests prove a
    /// one-time arm adoption and makes any historical mixed bank diagnosable;
    /// the manifest seal now prevents different arms from entering one bank.
    /// Empty (and omitted) for a default-configuration row, so existing banks
    /// round-trip unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) arm: Vec<(String, String)>,
    /// #sweep-cache: true when this row was SERVED from the verdict cache rather
    /// than measured by a child process of this sweep.
    ///
    /// This is a gate-integrity field, not a convenience. Without it a bank in
    /// which zero children ran is byte-indistinguishable from one that ran in
    /// full: the served row carries the earlier run's flight record verbatim, so
    /// every artifact-level receipt check (`scripts/ny_search.py` T1.2/T1.4)
    /// still passes against evidence produced by a DIFFERENT process. A replay
    /// may be a legitimate saving, but nothing downstream may be forced to
    /// mistake it for a measurement. `ny_search.py`'s `rows_were_measured` check
    /// reads exactly this field.
    ///
    /// Omitted when false, so a measured row and every pre-cache bank round-trip
    /// byte-identically. Never stored INTO the cache as true: only the measured
    /// branch writes entries.
    #[serde(default, skip_serializing_if = "is_false")]
    pub(crate) from_cache: bool,
    /// True when this row's `timeout` is a BUDGET OVERRUN — the child was
    /// hard-stopped at its own deadline after committing and verifying its
    /// `timeout` token — rather than a clean in-budget timeout.
    ///
    /// This is what makes an un-measured row distinguishable from a genuine
    /// crash in the bank. It is a FLAG rather than a fifth `SweepVerdict`
    /// variant deliberately: the official CSV must stay inside the organizers'
    /// four-token vocabulary (`budgetoverrun` is not a legal token, `timeout`
    /// is), and every consumer that reasons about timeouts — the reseed
    /// selector, `solved()`, the scoring deadline — must keep treating an
    /// overrun as the unsolved row it is, with no new arm to forget.
    ///
    /// Omitted when false, so a clean row and every pre-flag bank round-trip
    /// byte-identically.
    #[serde(default, skip_serializing_if = "is_false")]
    pub(crate) budget_overrun: bool,
    /// Retained SAT witness (#witness-retention-gap). Absent on non-sat rows
    /// and on banks written before retention existed; an explicit `null` on a
    /// sat row whose witness could not be extracted or copied — that gap is
    /// also counted in the sweep summary, so it is visible, never silent.
    #[serde(
        default,
        deserialize_with = "witness_if_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) witness: Option<WitnessDisposition>,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct SweepSummary {
    pub(crate) rows: usize,
    pub(crate) sat: usize,
    pub(crate) unsat: usize,
    pub(crate) timeout: usize,
    pub(crate) unknown: usize,
    pub(crate) error: usize,
    /// Of `timeout`, how many were BUDGET OVERRUNS (hard-stopped at their own
    /// deadline) rather than clean in-budget timeouts. Counted separately so an
    /// operator can see "this family is bankable, but N rows only just fit"
    /// instead of it vanishing into the timeout bucket — or, as before this
    /// existed, into `error`.
    pub(crate) budget_overrun: usize,
    pub(crate) capped_rows: usize,
    /// Sat rows banked without a retained witness (#witness-retention-gap):
    /// organizer-style replay cannot revalidate these rows.
    pub(crate) sat_rows_without_witness: usize,
    /// #sweep-cache: rows this invocation SERVED from the verdict cache, and
    /// rows it measured because the cache could not serve them. Always emitted,
    /// zeros included: a summary that cannot say how much of it was actually
    /// run cannot be checked for silently-served stale rows.
    pub(crate) cache_hits: usize,
    pub(crate) cache_misses: usize,
    pub(crate) wall_secs: f64,
}

impl SweepSummary {
    fn record(&mut self, row: &SweepRow) {
        self.rows += 1;
        match row.verdict {
            SweepVerdict::Sat => self.sat += 1,
            SweepVerdict::Unsat => self.unsat += 1,
            SweepVerdict::Timeout => self.timeout += 1,
            SweepVerdict::Unknown => self.unknown += 1,
            SweepVerdict::Error => self.error += 1,
        }
        if row.budget_overrun {
            self.budget_overrun += 1;
        }
        if row.capped_from.is_some() {
            self.capped_rows += 1;
        }
        if row.verdict == SweepVerdict::Sat
            && !matches!(row.witness, Some(WitnessDisposition::Retained(_)))
        {
            self.sat_rows_without_witness += 1;
        }
    }

    /// Instances decided one way or the other — the only rows worth points.
    pub(crate) fn solved(&self) -> usize {
        self.sat + self.unsat
    }
}

/// Options for a sweep, mirroring the CLI flags one-for-one.
#[derive(Debug, Clone)]
pub(crate) struct SweepOptions {
    pub(crate) year: u32,
    pub(crate) vnnlib_version: Option<VnnlibVersion>,
    pub(crate) categories: Vec<String>,
    pub(crate) output: Option<PathBuf>,
    pub(crate) timeout_cap: Option<u64>,
    pub(crate) limit: Option<usize>,
    /// File naming exactly which rows to run (`category,onnx,vnnlib` per line).
    ///
    /// Restricting an arm to a named subset is what makes a gate set affordable:
    /// a whole-category arm re-measures every row to learn about the handful
    /// that can move. Mutually exclusive with `--limit` — two selectors that
    /// silently intersect would make the swept set unpredictable.
    pub(crate) instances: Option<PathBuf>,
    pub(crate) configs_dir: Option<PathBuf>,
    pub(crate) resume: bool,
    pub(crate) overwrite: bool,
    pub(crate) json: bool,
    pub(crate) dry_run: bool,
    /// #arm-overrides: lever assignments applied to EVERY child of this sweep,
    /// as `(NAME, value)` pairs already validated against `ny_levers::space`.
    ///
    /// This exists so an A/B does not have to be run by exporting a variable into
    /// the parent shell. A shell export is invisible to the artifact, is
    /// inherited by every later process, and silently mixes arms across a
    /// `--resume` — the hole `ambient_env` sealing closes. Passing the arm here
    /// instead makes it DATA: it is applied per child, recorded on every row, and
    /// sealed into the manifest.
    pub(crate) arm: Vec<(String, String)>,
    /// #sweep-cache: how the content-addressed verdict cache participates.
    pub(crate) cache: cache::CacheMode,
    /// Cache root; defaults to [`DEFAULT_VERDICT_CACHE_DIR`]. Deliberately NOT
    /// derived from the output bank: the whole value of the cache is that a new
    /// bank reuses the pairs an older one already measured.
    pub(crate) cache_dir: Option<PathBuf>,
}

/// Default verdict-cache root, beside the default sweep output directory.
const DEFAULT_VERDICT_CACHE_DIR: &str = "reports/sweeps/verdict-cache";

/// Resolve which categories to sweep, erroring on an unknown name rather than
/// quietly sweeping nothing.
fn selected_categories(year: u32, requested: &[String]) -> Result<Vec<(String, PathBuf)>> {
    let all = discover_categories(year)?;
    if all.is_empty() {
        bail!("no VNN-COMP {year} benchmark categories were discovered");
    }
    if requested.is_empty() {
        return Ok(all);
    }
    let mut chosen = Vec::new();
    let mut selected = HashSet::new();
    for want in requested {
        match all.iter().find(|(name, _)| name == want) {
            Some(hit) if selected.insert(hit.0.clone()) => chosen.push(hit.clone()),
            Some(_) => {}
            None => {
                let mut names: Vec<&str> = all.iter().map(|(n, _)| n.as_str()).collect();
                names.sort_unstable();
                bail!(
                    "unknown category {want:?} for VNN-COMP {year}. Available: {}",
                    names.join(", ")
                );
            }
        }
    }
    Ok(chosen)
}

type SweepBaseIdentity = (String, String, String);
type SweepIdentity = (String, String, String, usize);

fn occurrence_identity(
    counts: &mut HashMap<SweepBaseIdentity, usize>,
    base: SweepBaseIdentity,
) -> SweepIdentity {
    let occurrence = counts.entry(base.clone()).or_default();
    let identity = (base.0, base.1, base.2, *occurrence);
    *occurrence += 1;
    identity
}

fn metadata_path_for(output: &Path) -> PathBuf {
    output.with_extension("metadata.jsonl")
}

fn manifest_path_for(output: &Path) -> PathBuf {
    output.with_extension("manifest.json")
}

type SweepState = BTreeMap<SweepIdentity, SweepRow>;

/// Whether a banked witness path is anything other than a contained,
/// bank-relative location.
///
/// A witness record carries a BANK-RELATIVE POSIX path, so it must be judged by
/// POSIX rules on every host. The previous guard asked
/// `Path::new(&witness.path).is_absolute()`, whose answer is HOST-dependent:
/// on Windows that is `false` for `/absolute/w.counterexample`, because a
/// Windows absolute path wants a drive prefix — so a rooted record passed
/// containment there. The `..` scan had the mirror gap, splitting only on `/`.
///
/// Judged here by explicit rules instead, so the verdict is identical on every
/// platform and strictly no weaker than before:
///   - a leading `/` or `\` (POSIX-rooted, or Windows root/UNC),
///   - anything the host itself calls absolute (`C:\...`, `\\server\share`),
///   - a `..` component under EITHER separator.
fn witness_path_escapes_bank(witness_path: &str) -> bool {
    witness_path.starts_with('/')
        || witness_path.starts_with('\\')
        || Path::new(witness_path).is_absolute()
        || witness_path
            .split(['/', '\\'])
            .any(|component| component == "..")
}

fn validate_metadata_row(row: &SweepRow, path: &Path, line: usize) -> Result<()> {
    if row.category.is_empty()
        || row.onnx.is_empty()
        || row.vnnlib.is_empty()
        || !row.seconds.is_finite()
        || row.seconds < 0.0
        || row.budget_secs == 0
        || row
            .capped_from
            .is_some_and(|official| official <= row.budget_secs)
    {
        bail!(
            "{} line {line}: invalid identity, elapsed time, or budget metadata",
            path.display()
        );
    }
    match &row.witness {
        Some(_) if row.verdict != SweepVerdict::Sat => {
            bail!(
                "{} line {line}: a witness field on a non-sat row has no meaning",
                path.display()
            );
        }
        Some(WitnessDisposition::Retained(witness))
            if witness.path.is_empty()
                || witness.sha256.is_empty()
                || witness_path_escapes_bank(&witness.path) =>
        {
            bail!(
                "{} line {line}: a witness record must carry a non-empty bank-relative \
                 path and content hash",
                path.display()
            );
        }
        _ => {}
    }
    Ok(())
}

fn metadata_state(path: &Path) -> Result<SweepState> {
    let body = std::fs::read_to_string(path)
        .with_context(|| format!("read sweep metadata {}", path.display()))?;
    let mut state = BTreeMap::new();
    for (index, line) in body.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: SweepRow = serde_json::from_str(line).with_context(|| {
            format!(
                "{} line {}: invalid sweep metadata JSON",
                path.display(),
                index + 1
            )
        })?;
        validate_metadata_row(&row, path, index + 1)?;
        let identity = (
            row.category.clone(),
            row.onnx.clone(),
            row.vnnlib.clone(),
            row.occurrence,
        );
        if state.insert(identity.clone(), row).is_some() {
            bail!(
                "{} line {}: duplicate authoritative metadata identity {identity:?}",
                path.display(),
                index + 1
            );
        }
    }
    state_rows_in_plan_order(&state)?;
    Ok(state)
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct ArtifactFingerprint {
    canonical_path: String,
    sha256: String,
}

/// One external tensor-data file named by an ONNX model.
///
/// `location` is the normalized relative POSIX path authored in the protobuf;
/// `artifact` binds that logical dependency to the exact resolved file bytes.
/// Hashing only the `.onnx` container is incomplete because ONNX permits tensor
/// payloads to live in sibling files whose contents can change independently.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct ExternalDataInputFingerprint {
    location: String,
    artifact: ArtifactFingerprint,
}

/// One resolved ONNX member and the exact bytes planned for this sweep.
///
/// A relational model has one entry per labelled network, in literal order;
/// a single-model row has one entry whose label is absent. Keeping the label in
/// the seal prevents two byte-identical members from being silently exchanged.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct ModelInputFingerprint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    artifact: ArtifactFingerprint,
    /// Affirmative closure of every external tensor-data location this model
    /// can read. `Some([])` is meaningful; `None` identifies an incomplete
    /// legacy/tampered record and is never valid under schema v2.
    #[serde(default)]
    external_data: Option<Vec<ExternalDataInputFingerprint>>,
}

/// Exact resolved inputs for one planned corpus row (#instance-content-seal).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct ResolvedInstanceInputs {
    onnx: Vec<ModelInputFingerprint>,
    vnnlib: ArtifactFingerprint,
}

/// Audit-visible binding from a selected semantic row to its resolved bytes.
///
/// `selected_rows` remains the cross-language v1 identity consumed by
/// `ny_search.py`. This companion seal closes the content hole without changing
/// that wire contract: the CSV field strings can stay identical while a model
/// or property is edited in place.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct SelectedRowInputSeal {
    onnx: String,
    vnnlib: String,
    occurrence: usize,
    instance_index: usize,
    inputs: ResolvedInstanceInputs,
}

/// Compact, stable, order-insensitive identity of a semantic set.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct SemanticSetIdentity {
    schema: String,
    count: usize,
    sha256: String,
}

/// The row-selection operation applied independently to every selected
/// category. This is part of a sweep bank's resume identity.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum InstanceSelection {
    /// Run every row in each selected category.
    All,
    /// Run the first `per_category` rows in each selected category.
    Limit { per_category: usize },
    /// Run the exact semantic set named by `--instances`.
    Exact { identity: SemanticSetIdentity },
}

fn canonical_arm(arm: &[(String, String)]) -> Vec<(String, String)> {
    let mut canonical = arm.to_vec();
    canonical.sort();
    canonical
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct SweepManifest {
    schema_version: u32,
    year: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    vnnlib_version: Option<VnnlibVersion>,
    timeout_cap: Option<u64>,
    configs: Option<ArtifactFingerprint>,
    executable: ArtifactFingerprint,
    /// The canonical `NY_BUILD_ID_V1 …` line of the binary that measured this
    /// bank (#binary-identity).
    ///
    /// `executable.sha256` above already content-addresses the file, but a
    /// digest cannot be read. This line names the DERIVED VERSION
    /// (`<major>.<minor>.<rev-list count>[+dirty.<commit>.<diff>]`), the cargo feature
    /// tier, the compiled backend class, and the impl markers — the things a
    /// human compares when two banks disagree. Two rows whose versions differ
    /// were not produced by the same verifier, full stop.
    ///
    /// Empty on banks written before #binary-identity existed.
    #[serde(default)]
    build_id: String,
    build_provenance: String,
    target_os: String,
    target_arch: String,
    /// Detected host compute regime at sweep start (#backend-detect):
    /// cpu-only vs metal vs cuda decides what the rows MEAN, so it is sealed
    /// here. Empty on banks written before detection existed.
    #[serde(default)]
    compute_backend: String,
    /// Host identity at sweep start (#host-provenance): hostname, CPU model,
    /// core count, RAM. Same-backend timings from different machine classes
    /// are still not comparable. Empty on banks written before the probe.
    #[serde(default)]
    host: String,
    /// Canonical explicit `--lever` arm applied to every child in this bank.
    ///
    /// `Some([])` is an affirmative baseline seal. `None` means only that the
    /// manifest predates this field; a legacy resume may adopt the requested
    /// arm only after every retained measurement's row provenance agrees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    arm: Option<Vec<(String, String)>>,
    /// Version of the ambient capture contract used by `ambient_env`.
    ///
    /// Absent manifests used the pre-Rayon capture and are legacy even when
    /// they contain an `ambient_env` object. An unknown present schema is never
    /// silently migrated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ambient_env_schema: Option<String>,
    /// Ambient `NY_*` / thread-control environment at sweep start
    /// (#arm-sealing).
    ///
    /// `run_instance` only ADDS `RUST_LOG` and the submission environment to each
    /// child, so any lever exported into the parent shell is inherited by every
    /// row and was previously INVISIBLE in the artifact. Two consequences, both
    /// of which have produced fake results in this repo: a `--resume` could
    /// silently continue a bank under a different arm than it started with, and
    /// any cache keyed on this manifest would serve rows measured under an
    /// environment the key never saw. `Some({})` is affirmative evidence that
    /// the parent was clean; `None` is reserved for manifests written before
    /// this presence-sensitive seal existed. A legacy resume may adopt only
    /// after every retained measurement's child flight record proves the same
    /// effective environment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ambient_env: Option<BTreeMap<String, String>>,
    /// Content fingerprint of the retained-witness directory, sealed at sweep
    /// end (#witness-retention-gap) — the directory grows during the run, so
    /// the start-of-run manifest cannot carry it. Derived output provenance
    /// only: deliberately NOT part of the `--resume` compatibility comparison.
    /// Absent when no witnesses were banked and on manifests written before
    /// retention existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    witness_dir: Option<ArtifactFingerprint>,
    /// Semantic row-selection request (#instance-selection-seal).
    ///
    /// This must be sealed separately from `instances_csv`: the latter pins the
    /// full source corpus, but says nothing about whether this bank ran all
    /// rows, a first-N prefix, or an exact named subset. Absent only on manifests
    /// written before selection sealing; a legacy resume adopts the current
    /// request once, after which every change is refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    instance_selection: Option<InstanceSelection>,
    /// Exact selected row set for every category planned by this invocation.
    ///
    /// Each identity includes category/model/property, duplicate occurrence,
    /// and source `instance_index`. A present category with `count: 0` means a
    /// proven empty selection; an absent category is deliberately unsealed
    /// legacy provenance. Compatible resumes compare overlapping categories
    /// and may add new ones without relabeling omitted historical categories.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    selected_rows: BTreeMap<String, SemanticSetIdentity>,
    /// Presence-sensitive version of `selected_row_inputs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    selected_row_inputs_schema: Option<String>,
    /// Resolved ONNX/VNN-LIB content for every selected row.
    ///
    /// The semantic row seal above deliberately uses the stable cross-tool v1
    /// tuple and therefore cannot notice an in-place edit to a referenced file.
    /// These explicit fingerprints are the missing half of the identity. A
    /// retained non-error row from a manifest without this field is unproven and
    /// is never adopted on resume.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    selected_row_inputs: BTreeMap<String, Vec<SelectedRowInputSeal>>,
    instances_csv: BTreeMap<String, ArtifactFingerprint>,
}

const SWEEP_MANIFEST_SCHEMA_VERSION: u32 = 1;

fn canonical_utf8(path: &Path, what: &str) -> Result<(PathBuf, String)> {
    let canonical = std::fs::canonicalize(path)
        .with_context(|| format!("resolve {what} {}", path.display()))?;
    let rendered = canonical
        .to_str()
        .map(str::to_string)
        .with_context(|| format!("{what} path is not valid UTF-8: {}", canonical.display()))?;
    Ok((canonical, rendered))
}

fn hash_file_into(path: &Path, hasher: &mut Sha256) -> Result<()> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("open {} for hashing", path.display()))?;
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if count == 0 {
            return Ok(());
        }
        hasher.update(&buffer[..count]);
    }
}

fn finish_sha256(hasher: Sha256) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

/// Build a semantic-set digest shared with `scripts/ny_search.py`.
///
/// Python canonicalizes `{"entries":[...],"schema":"..."}` with sorted keys,
/// compact separators and UTF-8 output. Declaring `entries` before `schema`
/// gives serde the same key order; sorting entries gives both tools the same
/// set-like identity even when input order changes.
fn semantic_set_identity<T>(schema: &str, entries: &[T]) -> Result<SemanticSetIdentity>
where
    T: Clone + Ord + Serialize,
{
    #[derive(Serialize)]
    struct CanonicalIdentity<'a, T> {
        entries: &'a [T],
        schema: &'a str,
    }

    let mut entries = entries.to_vec();
    entries.sort();
    if entries.windows(2).any(|pair| pair[0] == pair[1]) {
        bail!("semantic set {schema:?} cannot contain duplicate identities");
    }
    let canonical = serde_json::to_vec(&CanonicalIdentity {
        entries: &entries,
        schema,
    })?;
    let mut hasher = Sha256::new();
    hasher.update(canonical);
    Ok(SemanticSetIdentity {
        schema: schema.to_string(),
        count: entries.len(),
        sha256: finish_sha256(hasher),
    })
}

fn exact_instance_selection_identity(
    selectors: &[InstanceSelector],
) -> Result<SemanticSetIdentity> {
    if selectors.is_empty() {
        bail!("an exact instance selection cannot be empty");
    }
    semantic_set_identity(INSTANCE_SELECTOR_IDENTITY_SCHEMA, selectors)
}

fn requested_instance_selection(
    limit: Option<usize>,
    selectors: Option<&[InstanceSelector]>,
) -> Result<InstanceSelection> {
    match (limit, selectors) {
        (None, None) => Ok(InstanceSelection::All),
        (Some(per_category), None) => Ok(InstanceSelection::Limit { per_category }),
        (None, Some(selectors)) => Ok(InstanceSelection::Exact {
            identity: exact_instance_selection_identity(selectors)?,
        }),
        (Some(_), Some(_)) => {
            bail!("--instances and --limit cannot both define the instance selection")
        }
    }
}

fn fingerprint_file(path: &Path, what: &str) -> Result<ArtifactFingerprint> {
    let (canonical, canonical_path) = canonical_utf8(path, what)?;
    if !canonical.is_file() {
        bail!("{what} is not a regular file: {}", canonical.display());
    }
    let mut hasher = Sha256::new();
    hash_file_into(&canonical, &mut hasher)?;
    Ok(ArtifactFingerprint {
        canonical_path,
        sha256: finish_sha256(hasher),
    })
}

/// Project the one early executable guard into manifest identity without
/// reopening the mutable source path.
fn fingerprint_from_binary_identity(
    identity: &crate::build_id::BinaryIdentity,
) -> Result<ArtifactFingerprint> {
    let canonical_path = identity
        .path
        .to_str()
        .map(str::to_string)
        .with_context(|| {
            format!(
                "ny executable path is not valid UTF-8: {}",
                identity.path.display()
            )
        })?;
    Ok(ArtifactFingerprint {
        canonical_path,
        sha256: identity.exe_sha256.clone(),
    })
}

/// Copy the guarded executable into the sweep's private directory, prove the
/// copy has the guarded bytes, then remove write permission.
///
/// The original executable path is a build target and is therefore mutable.
/// Re-hashing it before each launch narrows the race but does not close the gap
/// between the final hash read and `exec`. The child path returned here is a
/// private, content-verified copy that ordinary rebuilds cannot name or replace.
/// On Unix it is mode 0500 inside tempfile's mode-0700 directory; elsewhere it
/// is marked read-only. This is the safest portable copy-and-hash design: it
/// does not claim filesystem immutability against the sweep's own privileged
/// process, but it removes the live build path from every child launch.
fn sealed_executable_copy(
    guard: &crate::build_id::BinaryIdentityGuard,
    scratch: &Path,
) -> Result<PathBuf> {
    guard
        .reverify("creating the private executable copy")
        .context("the ny executable changed before it could be copied")?;
    let extension = guard.sealed().path.extension();
    let mut file_name = PathBuf::from("sealed-ny");
    if let Some(extension) = extension {
        file_name.set_extension(extension);
    }
    let destination = scratch.join(file_name);
    let mut source = std::fs::File::open(&guard.sealed().path).with_context(|| {
        format!(
            "open sealed ny executable {} for private copy",
            guard.sealed().path.display()
        )
    })?;
    let mut copy = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&destination)
        .with_context(|| format!("create private executable copy {}", destination.display()))?;
    std::io::copy(&mut source, &mut copy)
        .with_context(|| format!("copy ny executable to {}", destination.display()))?;
    copy.flush()
        .with_context(|| format!("flush private executable copy {}", destination.display()))?;
    copy.sync_all()
        .with_context(|| format!("sync private executable copy {}", destination.display()))?;
    drop(copy);

    // Detect both atomic replacement of the source while it was open and an
    // in-place write during the copy. Either race aborts before planning.
    guard
        .reverify("finalizing the private executable copy")
        .context("the ny executable changed while it was being copied")?;
    let copied = fingerprint_file(&destination, "private ny executable copy")?;
    if copied.sha256 != guard.sealed().exe_sha256 {
        bail!(
            "private ny executable copy does not match the early binary seal\n  sealed: {}\n  \
             copied: {}",
            guard.sealed().exe_sha256,
            copied.sha256,
        );
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&destination, std::fs::Permissions::from_mode(0o500))
            .with_context(|| {
                format!(
                    "make private executable copy read/execute-only {}",
                    destination.display()
                )
            })?;
    }
    #[cfg(not(unix))]
    {
        let mut permissions = std::fs::metadata(&destination)
            .with_context(|| format!("inspect private executable copy {}", destination.display()))?
            .permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(&destination, permissions).with_context(|| {
            format!(
                "make private executable copy read-only {}",
                destination.display()
            )
        })?;
    }
    Ok(destination)
}

#[derive(Debug, Default)]
struct InputFingerprintMemo {
    artifacts: BTreeMap<PathBuf, ArtifactFingerprint>,
    onnx_external_data: BTreeMap<PathBuf, Vec<ExternalDataInputFingerprint>>,
}

impl InputFingerprintMemo {
    fn new() -> Self {
        Self::default()
    }
}

fn fingerprint_input_file(
    path: &Path,
    what: &str,
    memo: &mut InputFingerprintMemo,
) -> Result<ArtifactFingerprint> {
    let canonical = std::fs::canonicalize(path)
        .with_context(|| format!("resolve {what} {}", path.display()))?;
    if let Some(fingerprint) = memo.artifacts.get(&canonical) {
        return Ok(fingerprint.clone());
    }
    let fingerprint = fingerprint_file(&canonical, what)?;
    memo.artifacts.insert(canonical, fingerprint.clone());
    Ok(fingerprint)
}

fn normalize_external_data_location(raw: &str, tensor_name: &str) -> Result<String> {
    if raw.is_empty() || raw.as_bytes().contains(&0) {
        bail!("ONNX tensor {tensor_name:?} has an empty or NUL-containing external-data location");
    }
    let bytes = raw.as_bytes();
    let looks_like_windows_prefix = raw.starts_with('\\')
        || (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':');
    if raw.contains('\\') || looks_like_windows_prefix {
        bail!(
            "ONNX tensor {tensor_name:?} external-data location {raw:?} is not a portable relative POSIX path"
        );
    }
    let path = Path::new(raw);
    if path.is_absolute() {
        bail!(
            "ONNX tensor {tensor_name:?} external-data location {raw:?} must be relative to the model directory"
        );
    }

    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::Normal(part) => {
                parts.push(part.to_str().with_context(|| {
                    format!(
                        "ONNX tensor {tensor_name:?} external-data location {raw:?} is not UTF-8"
                    )
                })?);
            }
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => {
                bail!(
                    "ONNX tensor {tensor_name:?} external-data location {raw:?} attempts to escape the model directory"
                );
            }
        }
    }
    if parts.is_empty() {
        bail!("ONNX tensor {tensor_name:?} external-data location {raw:?} does not name a file");
    }
    Ok(parts.join("/"))
}

fn tensor_external_data_location(
    tensor: &ny_onnx::onnx_proto::TensorProto,
) -> Result<Option<String>> {
    const DATA_LOCATION_DEFAULT: i32 = 0;
    const DATA_LOCATION_EXTERNAL: i32 = 1;

    match tensor.data_location {
        DATA_LOCATION_DEFAULT if tensor.external_data.is_empty() => return Ok(None),
        DATA_LOCATION_DEFAULT => {
            bail!(
                "ONNX tensor {:?} has external_data metadata but data_location is DEFAULT",
                tensor.name
            );
        }
        DATA_LOCATION_EXTERNAL => {}
        other => {
            bail!(
                "ONNX tensor {:?} has unknown data_location {other}",
                tensor.name
            );
        }
    }

    let mut seen = HashSet::new();
    let mut location = None;
    for entry in &tensor.external_data {
        if !seen.insert(entry.key.as_str()) {
            bail!(
                "ONNX tensor {:?} repeats external_data key {:?}",
                tensor.name,
                entry.key
            );
        }
        match entry.key.as_str() {
            "location" => {
                location = Some(normalize_external_data_location(
                    &entry.value,
                    &tensor.name,
                )?)
            }
            // These fields select or authenticate bytes inside the same file;
            // the closure conservatively hashes the complete file.
            "offset" | "length" | "checksum" | "basepath" => {}
            other => {
                bail!(
                    "ONNX tensor {:?} has unsupported external_data key {other:?}",
                    tensor.name
                );
            }
        }
    }
    location
        .map(Some)
        .context("ONNX external_data is missing its required location")
}

fn onnx_external_data_locations(model_path: &Path, container_bytes: &[u8]) -> Result<Vec<String>> {
    let mut decoded = Vec::new();
    let protobuf = if model_path.extension().and_then(std::ffi::OsStr::to_str) == Some("gz") {
        GzDecoder::new(container_bytes)
            .read_to_end(&mut decoded)
            .with_context(|| format!("decode gzipped ONNX model {}", model_path.display()))?;
        decoded.as_slice()
    } else {
        container_bytes
    };
    let model = ny_onnx::onnx_proto::ModelProto::decode(protobuf).with_context(|| {
        format!(
            "decode ONNX model {} for dependency sealing",
            model_path.display()
        )
    })?;
    let Some(graph) = model.graph.as_ref() else {
        return Ok(Vec::new());
    };
    if !graph.sparse_initializer.is_empty() {
        bail!(
            "ONNX model {} contains unsupported sparse initializers; its input dependency closure cannot be proven",
            model_path.display()
        );
    }

    let mut locations = std::collections::BTreeSet::new();
    for tensor in &graph.initializer {
        if let Some(location) = tensor_external_data_location(tensor)? {
            locations.insert(location);
        }
    }
    for node in &graph.node {
        for attribute in &node.attribute {
            let has_tensor = attribute.t.is_some();
            if has_tensor != (attribute.r#type == ny_onnx::onnx_proto::attribute_type::TENSOR) {
                bail!(
                    "ONNX attribute {:?} on node {:?} has inconsistent tensor payload/type metadata",
                    attribute.name,
                    node.name
                );
            }
            let has_unmodeled_container = attribute.g.is_some()
                || !attribute.tensors.is_empty()
                || !attribute.graphs.is_empty()
                || attribute.sparse_tensor.is_some()
                || !attribute.sparse_tensors.is_empty();
            if has_unmodeled_container || matches!(attribute.r#type, 5 | 9..=12) {
                bail!(
                    "ONNX attribute {:?} on node {:?} uses an unsupported nested tensor container; its input dependency closure cannot be proven",
                    attribute.name,
                    node.name
                );
            }
            if let Some(tensor) = attribute.t.as_ref() {
                if let Some(location) = tensor_external_data_location(tensor)? {
                    locations.insert(location);
                }
            }
        }
    }
    Ok(locations.into_iter().collect())
}

fn ensure_external_data_path_has_no_symlinks(model_dir: &Path, location: &str) -> Result<PathBuf> {
    let mut path = model_dir.to_path_buf();
    let parts = location.split('/').collect::<Vec<_>>();
    for (index, part) in parts.iter().enumerate() {
        path.push(part);
        let metadata = std::fs::symlink_metadata(&path).with_context(|| {
            format!(
                "inspect ONNX external-data dependency {location:?} at {}",
                path.display()
            )
        })?;
        if metadata.file_type().is_symlink() {
            bail!(
                "ONNX external-data dependency {location:?} contains a symlink at {}; the verifier rejects sidecar symlinks",
                path.display()
            );
        }
        if index + 1 < parts.len() && !metadata.is_dir() {
            bail!(
                "ONNX external-data dependency {location:?} traverses non-directory {}",
                path.display()
            );
        }
        if index + 1 == parts.len() && !metadata.is_file() {
            bail!(
                "ONNX external-data dependency {location:?} does not resolve to a regular file at {}",
                path.display()
            );
        }
    }
    Ok(path)
}

fn fingerprint_onnx_input(
    path: &Path,
    label: Option<String>,
    memo: &mut InputFingerprintMemo,
) -> Result<ModelInputFingerprint> {
    let canonical = std::fs::canonicalize(path)
        .with_context(|| format!("resolve ONNX input {}", path.display()))?;
    let external_data = if let Some(found) = memo.onnx_external_data.get(&canonical) {
        found.clone()
    } else {
        // Hash and decode the SAME container bytes. Separate opens allow an
        // atomic replacement to combine model A's hash with model B's sidecar
        // closure, which is not a valid identity even though the next row
        // preflight would normally catch a stable replacement.
        let container_bytes = std::fs::read(&canonical)
            .with_context(|| format!("read ONNX input {}", canonical.display()))?;
        let canonical_path = canonical
            .to_str()
            .map(str::to_string)
            .with_context(|| format!("ONNX input path is not UTF-8: {}", canonical.display()))?;
        let mut hasher = Sha256::new();
        hasher.update(&container_bytes);
        memo.artifacts.insert(
            canonical.clone(),
            ArtifactFingerprint {
                canonical_path,
                sha256: finish_sha256(hasher),
            },
        );
        let model_dir = canonical
            .parent()
            .context("resolved ONNX model has no parent directory")?;
        let mut found = Vec::new();
        for location in onnx_external_data_locations(&canonical, &container_bytes)? {
            let dependency = ensure_external_data_path_has_no_symlinks(model_dir, &location)?;
            found.push(ExternalDataInputFingerprint {
                location,
                artifact: fingerprint_input_file(
                    &dependency,
                    "resolved ONNX external-data input",
                    memo,
                )?,
            });
        }
        memo.onnx_external_data
            .insert(canonical.clone(), found.clone());
        found
    };
    let artifact = memo
        .artifacts
        .get(&canonical)
        .cloned()
        .context("ONNX dependency memo has no paired container fingerprint")?;
    Ok(ModelInputFingerprint {
        label,
        artifact,
        external_data: Some(external_data),
    })
}

/// Resolve and content-address every file the child will read for one row.
fn fingerprint_instance_inputs(
    category_dir: &Path,
    instance: &SweepInstance,
    memo: &mut InputFingerprintMemo,
) -> Result<ResolvedInstanceInputs> {
    let models = super::vnncomp_reseed::model_identity_entries(&instance.onnx_field)
        .context("parse ONNX field for content sealing")?;
    let mut onnx = Vec::with_capacity(models.len());
    for (label, model) in models {
        let path = category_dir.join(&model);
        onnx.push(fingerprint_onnx_input(&path, label, memo)?);
    }
    let vnnlib = fingerprint_input_file(
        &category_dir.join(&instance.vnnlib_field),
        "resolved VNN-LIB input",
        memo,
    )?;
    Ok(ResolvedInstanceInputs { onnx, vnnlib })
}

fn collect_tree_files(dir: &Path, what: &str, files: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = std::fs::read_dir(dir)
        .with_context(|| format!("read {what} {}", dir.display()))?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .with_context(|| format!("read entry under {}", dir.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    entries.sort();
    for path in entries {
        let metadata = std::fs::symlink_metadata(&path)
            .with_context(|| format!("inspect {what} path {}", path.display()))?;
        if metadata.file_type().is_symlink() {
            bail!(
                "{what} contains a symlink; use a resolved, self-contained tree: {}",
                path.display()
            );
        }
        if metadata.is_dir() {
            collect_tree_files(&path, what, files)?;
        } else if metadata.is_file() {
            files.push(path);
        } else {
            bail!("{what} contains a non-file entry: {}", path.display());
        }
    }
    Ok(())
}

/// Content-address a directory tree: relative names, sizes, and bytes, in
/// sorted order. Used both for the configuration tree (sealed at sweep start)
/// and for the retained-witness directory (sealed at sweep end).
fn fingerprint_tree(path: &Path, what: &str) -> Result<ArtifactFingerprint> {
    let (canonical, canonical_path) = canonical_utf8(path, what)?;
    if !canonical.is_dir() {
        bail!("{what} is not a directory: {}", canonical.display());
    }
    let mut files = Vec::new();
    collect_tree_files(&canonical, what, &mut files)?;
    let mut hasher = Sha256::new();
    for file in files {
        let relative = file
            .strip_prefix(&canonical)
            .with_context(|| format!("{what} file escaped its canonical root"))?;
        let relative = relative
            .to_str()
            .with_context(|| format!("{what} path is not valid UTF-8: {}", file.display()))?;
        hasher.update((relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        let size = std::fs::metadata(&file)
            .with_context(|| format!("inspect {what} file {}", file.display()))?
            .len();
        hasher.update(size.to_le_bytes());
        hash_file_into(&file, &mut hasher)?;
    }
    Ok(ArtifactFingerprint {
        canonical_path,
        sha256: finish_sha256(hasher),
    })
}

/// Fingerprint the ORDERED set of preset directories this sweep may consult.
///
/// #sweep-preset-parity: a `--year 2026` sweep resolves each category
/// newest-first from `vnncomp26` down through the older year directories (see
/// [`vnncomp::benchmark_preset_search_dirs`]), so pinning only the exact-year
/// directory would leave every preset that actually shipped from `vnncomp25`
/// outside the manifest — a bank could then be reproduced against different
/// preset bytes than it ran on. A single-directory set keeps the previous
/// fingerprint byte for byte, so nothing that really was year-scoped changes
/// identity.
fn fingerprint_preset_dirs(dirs: &[PathBuf]) -> Result<Option<ArtifactFingerprint>> {
    let Some(primary) = dirs.first() else {
        return Ok(None);
    };
    let primary_fingerprint = fingerprint_tree(primary, "configuration directory")?;
    if dirs.len() == 1 {
        return Ok(Some(primary_fingerprint));
    }
    let mut hasher = Sha256::new();
    for dir in dirs {
        let fingerprint = fingerprint_tree(dir, "configuration directory")?;
        let name = dir
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| {
                format!(
                    "configuration directory has no UTF-8 name: {}",
                    dir.display()
                )
            })?;
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(fingerprint.sha256.as_bytes());
    }
    Ok(Some(ArtifactFingerprint {
        canonical_path: primary_fingerprint.canonical_path,
        sha256: finish_sha256(hasher),
    }))
}

fn read_manifest(path: &Path) -> Result<SweepManifest> {
    let body = std::fs::read_to_string(path)
        .with_context(|| format!("read sweep manifest {}", path.display()))?;
    serde_json::from_str(&body).with_context(|| format!("parse sweep manifest {}", path.display()))
}

#[cfg(unix)]
fn same_existing_file(left: &Path, right: &Path) -> Result<bool> {
    use std::os::unix::fs::MetadataExt;

    if !left.exists() || !right.exists() {
        return Ok(false);
    }
    let left_metadata =
        std::fs::metadata(left).with_context(|| format!("inspect {}", left.display()))?;
    let right_metadata =
        std::fs::metadata(right).with_context(|| format!("inspect {}", right.display()))?;
    Ok(left_metadata.dev() == right_metadata.dev() && left_metadata.ino() == right_metadata.ino())
}

#[cfg(not(unix))]
fn same_existing_file(left: &Path, right: &Path) -> Result<bool> {
    if !left.exists() || !right.exists() {
        return Ok(false);
    }
    Ok(std::fs::canonicalize(left)? == std::fs::canonicalize(right)?)
}

fn ensure_distinct_outputs(paths: &[&Path]) -> Result<()> {
    for (index, left) in paths.iter().enumerate() {
        for right in &paths[index + 1..] {
            if left == right || same_existing_file(left, right)? {
                bail!(
                    "sweep output artifacts must be distinct files, but {} and {} alias",
                    left.display(),
                    right.display()
                );
            }
        }
    }
    Ok(())
}

fn output_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn atomic_write(path: &Path, write: impl FnOnce(&mut std::fs::File) -> Result<()>) -> Result<()> {
    let parent = output_parent(path);
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("create temporary file beside {}", path.display()))?;
    write(temporary.as_file_mut())?;
    temporary
        .as_file_mut()
        .flush()
        .with_context(|| format!("flush temporary {}", path.display()))?;
    temporary
        .as_file()
        .sync_all()
        .with_context(|| format!("sync temporary {}", path.display()))?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("atomically replace {}", path.display()))?;
    #[cfg(unix)]
    std::fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("sync output directory {}", parent.display()))?;
    Ok(())
}

fn write_manifest_atomic(path: &Path, manifest: &SweepManifest) -> Result<()> {
    atomic_write(path, |file| {
        serde_json::to_writer_pretty(&mut *file, manifest)?;
        writeln!(file)?;
        Ok(())
    })
}

fn state_rows_in_plan_order(state: &SweepState) -> Result<Vec<&SweepRow>> {
    let mut rows = state.values().collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        (&left.category, left.instance_index)
            .cmp(&(&right.category, right.instance_index))
            .then_with(|| left.occurrence.cmp(&right.occurrence))
    });
    for pair in rows.windows(2) {
        if pair[0].category == pair[1].category && pair[0].instance_index == pair[1].instance_index
        {
            bail!(
                "authoritative metadata assigns category {:?} instance index {} more than once",
                pair[0].category,
                pair[0].instance_index
            );
        }
    }
    Ok(rows)
}

fn write_metadata_atomic(path: &Path, state: &SweepState) -> Result<()> {
    let rows = state_rows_in_plan_order(state)?;
    atomic_write(path, |file| {
        for row in rows {
            serde_json::to_writer(&mut *file, row)?;
            writeln!(file)?;
        }
        Ok(())
    })
}

fn write_results_atomic(path: &Path, state: &SweepState) -> Result<()> {
    let rows = state_rows_in_plan_order(state)?;
    atomic_write(path, |file| {
        for row in rows {
            writeln!(
                file,
                "{},{},{},0.0,{},{:.1}",
                csv_escape(&row.category),
                csv_escape(&row.onnx),
                csv_escape(&row.vnnlib),
                row.verdict.as_str(),
                row.seconds
            )?;
        }
        Ok(())
    })
}

/// Commit authoritative state first, then regenerate the official CSV view.
///
/// A crash between the two replacements can leave a stale CSV, but a future
/// `--resume` always trusts metadata and regenerates the CSV before running.
fn publish_state(metadata: &Path, output: &Path, state: &SweepState) -> Result<()> {
    write_metadata_atomic(metadata, state)?;
    write_results_atomic(output, state)
}

fn expected_child_ambient_env(
    ambient_env: &BTreeMap<String, String>,
    arm: &[(String, String)],
) -> BTreeMap<String, String> {
    let mut effective = ambient_env.clone();
    for (name, value) in submission_environment() {
        effective.insert((*name).to_string(), (*value).to_string());
    }
    // The explicit arm is applied last in `run_instance` and therefore wins
    // over both the inherited parent environment and submission defaults.
    for (name, value) in arm {
        effective.insert(name.clone(), value.clone());
    }
    crate::flight::ambient_env_from(effective)
}

fn row_flight_ambient_env(
    identity: &SweepIdentity,
    row: &SweepRow,
) -> Result<BTreeMap<String, String>> {
    let flight = row.flight.as_ref().with_context(|| {
        format!(
            "cannot safely --resume row {identity:?}: \
             the retained measurement has no child flight record"
        )
    })?;
    let schema = flight
        .get("ambient_env_schema")
        .and_then(serde_json::Value::as_str);
    if schema != Some(crate::flight::AMBIENT_ENV_SCHEMA) {
        bail!(
            "cannot safely --resume row {identity:?}: \
             the child flight ambient_env_schema is {:?}, expected {:?}; pre-v3 captures did \
             not record every Rayon worker-count variable and cannot prove a clean environment",
            schema,
            crate::flight::AMBIENT_ENV_SCHEMA,
        );
    }
    let ambient = flight.get("ambient_env").with_context(|| {
        format!(
            "cannot safely --resume row {identity:?}: \
             the child flight record has no ambient_env"
        )
    })?;
    serde_json::from_value(ambient.clone()).with_context(|| {
        format!(
            "cannot safely --resume row {identity:?}: \
             the child flight ambient_env is not a string map"
        )
    })
}

/// Validate every retained measurement against the current execution-input
/// seals. This both proves a one-time legacy-manifest adoption and detects a
/// corrupt or historically mixed bank even when its manifest already claims
/// current seals. Error rows are retried and are not measurements.
fn validate_retained_execution_seals(
    existing: &SweepManifest,
    desired: &SweepManifest,
    state: &SweepState,
) -> Result<()> {
    let desired_arm = desired
        .arm
        .as_deref()
        .context("current sweep plan is missing its required explicit-arm seal")?;
    let desired_ambient_schema = desired
        .ambient_env_schema
        .as_deref()
        .context("current sweep plan is missing its required ambient-environment schema")?;
    if desired_ambient_schema != crate::flight::AMBIENT_ENV_SCHEMA {
        bail!(
            "current sweep plan uses unsupported ambient-environment schema {:?}; expected {:?}",
            desired_ambient_schema,
            crate::flight::AMBIENT_ENV_SCHEMA,
        );
    }
    let desired_ambient = desired
        .ambient_env
        .as_ref()
        .context("current sweep plan is missing its required ambient-environment seal")?;
    let expected_ambient = expected_child_ambient_env(desired_ambient, desired_arm);

    for (identity, row) in state {
        if row.verdict == SweepVerdict::Error {
            continue;
        }
        let row_arm = canonical_arm(&row.arm);
        if row_arm != desired_arm {
            bail!(
                "cannot safely --resume row {identity:?}: the retained measurement records \
                 explicit arm {:?}, expected bank arm {:?}{}; use a new output bank",
                row_arm,
                desired_arm,
                if existing.arm.is_none() {
                    " (legacy manifest adoption requires every row to agree)"
                } else {
                    ""
                },
            );
        }
        let observed = row_flight_ambient_env(identity, row)?;
        if observed != expected_ambient {
            bail!(
                "cannot safely --resume row {identity:?}: its child flight records effective \
                 ambient environment {:?}, expected {:?} from the bank seals{}; use a new \
                 output bank",
                observed,
                expected_ambient,
                if existing.ambient_env_schema.is_none() {
                    " (legacy manifest adoption requires every row to agree)"
                } else {
                    ""
                },
            );
        }
    }
    Ok(())
}

fn row_set_identity_from_input_seals(
    category: &str,
    seals: &[SelectedRowInputSeal],
) -> Result<SemanticSetIdentity> {
    let rows = seals
        .iter()
        .map(|seal| {
            (
                category.to_string(),
                seal.onnx.clone(),
                seal.vnnlib.clone(),
                seal.occurrence,
                seal.instance_index,
            )
        })
        .collect::<Vec<_>>();
    selected_row_set_identity(&rows)
}

/// Validate the self-consistency of the explicit row-to-file-content map.
fn validate_selected_row_input_manifest(manifest: &SweepManifest, which: &str) -> Result<()> {
    match manifest.selected_row_inputs_schema.as_deref() {
        Some(SELECTED_ROW_INPUTS_SCHEMA) => {}
        Some(schema) => {
            bail!(
                "cannot safely use {which} sweep manifest: unsupported selected-row input \
                 schema {schema:?}; expected {SELECTED_ROW_INPUTS_SCHEMA:?}"
            );
        }
        None if manifest.selected_row_inputs.is_empty() => return Ok(()),
        None => {
            bail!(
                "cannot safely use {which} sweep manifest: selected_row_inputs is present \
                 without its schema"
            );
        }
    }

    if manifest.selected_row_inputs.len() != manifest.selected_rows.len()
        || manifest
            .selected_rows
            .keys()
            .any(|category| !manifest.selected_row_inputs.contains_key(category))
    {
        bail!(
            "cannot safely use {which} sweep manifest: selected_rows and \
             selected_row_inputs do not cover the same categories"
        );
    }
    for (category, seals) in &manifest.selected_row_inputs {
        let semantic = manifest.selected_rows.get(category).with_context(|| {
            format!(
                "cannot safely use {which} sweep manifest: category {category:?} has input \
                 seals but no semantic row seal"
            )
        })?;
        let reconstructed =
            row_set_identity_from_input_seals(category, seals).with_context(|| {
                format!(
                    "cannot safely use {which} sweep manifest: invalid selected-row input seals for \
                     category {category:?}"
                )
            })?;
        if &reconstructed != semantic {
            bail!(
                "cannot safely use {which} sweep manifest: category {category:?} input seals do \
                 not describe its selected_rows identity"
            );
        }
        for seal in seals {
            if seal.inputs.onnx.is_empty()
                || seal.inputs.vnnlib.canonical_path.is_empty()
                || seal.inputs.vnnlib.sha256.is_empty()
            {
                bail!(
                    "cannot safely use {which} sweep manifest: category {category:?} row \
                     {}/{:?} has an empty resolved input path or content hash",
                    seal.instance_index,
                    seal.vnnlib,
                );
            }
            for model in &seal.inputs.onnx {
                if model.artifact.canonical_path.is_empty() || model.artifact.sha256.is_empty() {
                    bail!(
                        "cannot safely use {which} sweep manifest: category {category:?} row \
                         {}/{:?} has an empty resolved ONNX path or content hash",
                        seal.instance_index,
                        seal.vnnlib,
                    );
                }
                let dependencies = model.external_data.as_ref().with_context(|| {
                    format!(
                        "cannot safely use {which} sweep manifest: category {category:?} row \
                         {}/{:?} has no affirmative ONNX external-data closure",
                        seal.instance_index, seal.vnnlib,
                    )
                })?;
                if dependencies.iter().any(|dependency| {
                    dependency.location.is_empty()
                        || dependency.artifact.canonical_path.is_empty()
                        || dependency.artifact.sha256.is_empty()
                }) || dependencies
                    .windows(2)
                    .any(|pair| pair[0].location >= pair[1].location)
                {
                    bail!(
                        "cannot safely use {which} sweep manifest: category {category:?} row \
                         {}/{:?} has malformed, duplicate, or noncanonical ONNX external-data seals",
                        seal.instance_index,
                        seal.vnnlib,
                    );
                }
                let model_parent = Path::new(&model.artifact.canonical_path)
                    .parent()
                    .with_context(|| {
                        format!(
                            "cannot safely use {which} sweep manifest: ONNX path {:?} has no parent",
                            model.artifact.canonical_path
                        )
                    })?;
                for dependency in dependencies {
                    let normalized =
                        normalize_external_data_location(&dependency.location, "manifest seal")?;
                    let expected_path = dependency
                        .location
                        .split('/')
                        .fold(model_parent.to_path_buf(), |path, part| path.join(part));
                    if normalized != dependency.location
                        || expected_path != Path::new(&dependency.artifact.canonical_path)
                    {
                        bail!(
                            "cannot safely use {which} sweep manifest: ONNX external-data \
                             location {:?} is noncanonical or does not resolve from model {:?} \
                             to sealed artifact {:?}",
                            dependency.location,
                            model.artifact.canonical_path,
                            dependency.artifact.canonical_path,
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// Every reusable historical verdict must point to a current-format content
/// seal. The present files cannot retroactively prove which bytes a legacy row
/// measured, so absence is a hard refusal rather than one-time adoption.
fn validate_retained_input_seals(manifest: &SweepManifest, state: &SweepState) -> Result<()> {
    validate_selected_row_input_manifest(manifest, "banked")?;
    for (identity, row) in state {
        if row.verdict == SweepVerdict::Error {
            // Harness errors are retried and never reused or cached.
            continue;
        }
        let seal = manifest
            .selected_row_inputs
            .get(&row.category)
            .and_then(|seals| {
                seals.iter().find(|seal| {
                    seal.onnx == row.onnx
                        && seal.vnnlib == row.vnnlib
                        && seal.occurrence == row.occurrence
                        && seal.instance_index == row.instance_index
                })
            });
        if seal.is_none() {
            bail!(
                "cannot safely --resume retained row {identity:?}: its manifest does not prove \
                 the resolved ONNX/VNN-LIB content hashes that produced the verdict. Legacy \
                 rows cannot adopt today's file bytes retroactively; use a new output bank."
            );
        }
    }
    Ok(())
}

fn merge_resume_manifest(
    existing: SweepManifest,
    desired: &SweepManifest,
    state: &SweepState,
) -> Result<SweepManifest> {
    if existing.schema_version != SWEEP_MANIFEST_SCHEMA_VERSION {
        bail!(
            "unsupported sweep manifest schema {}; expected {}",
            existing.schema_version,
            SWEEP_MANIFEST_SCHEMA_VERSION
        );
    }
    validate_selected_row_input_manifest(desired, "current")?;
    validate_retained_input_seals(&existing, state)?;
    let same_run = existing.year == desired.year
        && existing.vnnlib_version == desired.vnnlib_version
        && existing.timeout_cap == desired.timeout_cap
        && existing.configs == desired.configs
        && existing.executable == desired.executable
        // #binary-identity: a resumed bank must be the same VERIFIER, not just
        // the same bytes at the same path. Legacy banks carry an empty string
        // and are compared on the fields they do have, exactly as the backend
        // and host fields already work.
        && (existing.build_id.is_empty() || existing.build_id == desired.build_id)
        && existing.build_provenance == desired.build_provenance
        && existing.target_os == desired.target_os
        && existing.target_arch == desired.target_arch;
    if !same_run {
        bail!(
            "cannot safely --resume: year, VNN-LIB version, timeout cap, configuration content, \
             executable, binary identity, build provenance, or target differs from the pinned \
             manifest; use a new output bank.\n  banked: {}\n  current: {}",
            if existing.build_id.is_empty() {
                "<no binary identity recorded>"
            } else {
                existing.build_id.as_str()
            },
            desired.build_id,
        );
    }
    let desired_arm = desired
        .arm
        .as_ref()
        .context("current sweep plan is missing its required explicit-arm seal")?;
    let desired_ambient_schema = desired
        .ambient_env_schema
        .as_deref()
        .context("current sweep plan is missing its required ambient-environment schema")?;
    if desired_ambient_schema != crate::flight::AMBIENT_ENV_SCHEMA {
        bail!(
            "current sweep plan uses unsupported ambient-environment schema {:?}; expected {:?}",
            desired_ambient_schema,
            crate::flight::AMBIENT_ENV_SCHEMA,
        );
    }
    let desired_ambient = desired
        .ambient_env
        .as_ref()
        .context("current sweep plan is missing its required ambient-environment seal")?;
    if existing.arm.as_ref().is_some_and(|arm| arm != desired_arm) {
        bail!(
            "cannot safely --resume: this bank was measured with explicit arm {:?}, but the \
             current request is {:?}; rows from mixed treatment arms cannot share a bank. Use a \
             new output bank.",
            existing.arm,
            desired.arm,
        );
    }
    // #instance-selection-seal. `instances_csv` fingerprints only the full
    // source corpus. A bank resumed with `--limit 4` after starting with
    // `--limit 8` (or with another exact selector file) would otherwise remain
    // byte-for-byte compatible here while silently measuring a different row
    // set. Manifests predating this field adopt the current request once, just
    // like the binary/backend/host seals; a recorded request must match.
    if existing.instance_selection.is_some()
        && existing.instance_selection != desired.instance_selection
    {
        bail!(
            "cannot safely --resume: this bank was measured with instance selection {:?}, but \
             the current request is {:?}; rows from different subsets cannot share a bank. Use \
             a new output bank.",
            existing.instance_selection,
            desired.instance_selection,
        );
    }
    // A bank half-measured on cuda and half on cpu-only/metal is exactly the
    // cross-regime contamination the backend field exists to prevent. Legacy
    // manifests (empty field) adopt the current regime rather than blocking
    // resume; recorded regimes must match.
    if !existing.compute_backend.is_empty() && existing.compute_backend != desired.compute_backend {
        bail!(
            "cannot safely --resume: this bank was measured on a different compute backend \
             [{}] than this host detects now [{}]; rows from mixed regimes are not comparable. \
             Use a new output bank.",
            existing.compute_backend,
            desired.compute_backend
        );
    }
    // #arm-sealing: the same rule for the AMBIENT LEVER SET. A bank half-measured
    // with `NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN=1` in the parent shell and half
    // without is two different verifiers wearing one bank's name, and nothing
    // downstream can tell them apart -- `run_instance` only adds to the child
    // environment, so the export is inherited silently by every row. A legacy
    // manifest has no ambient schema. Its map, even when present, used an older
    // capture contract and is not current proof. An unknown present schema is
    // refused; a current v3 schema makes an empty map affirmative cleanliness.
    let existing_ambient_is_current = match existing.ambient_env_schema.as_deref() {
        Some(schema) if schema == crate::flight::AMBIENT_ENV_SCHEMA => true,
        Some(schema) => {
            bail!(
                "cannot safely --resume: this bank uses unsupported ambient-environment schema \
                 {:?}; expected {:?}",
                schema,
                crate::flight::AMBIENT_ENV_SCHEMA,
            );
        }
        None => false,
    };
    if existing_ambient_is_current && existing.ambient_env.is_none() {
        bail!(
            "cannot safely --resume: this bank claims ambient-environment schema {:?} but has no \
             ambient_env seal",
            crate::flight::AMBIENT_ENV_SCHEMA,
        );
    }
    if existing_ambient_is_current
        && existing
            .ambient_env
            .as_ref()
            .is_some_and(|ambient| ambient != desired_ambient)
    {
        let render = |env: &BTreeMap<String, String>| {
            env.iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        bail!(
            "cannot safely --resume: this bank was measured with a different ambient lever set              [{}] than the current environment provides [{}]; rows from mixed arms are not              comparable. Unset the difference or use a new output bank.",
            render(existing.ambient_env.as_ref().expect("checked present")),
            render(desired_ambient)
        );
    }
    validate_retained_execution_seals(&existing, desired, state)?;
    // Same rule for host identity: two machines can both detect "metal" and
    // still have incomparable timings (M-series laptop vs Mac Studio). The
    // cgan parity sweep proved this the hard way — a sealed 636s UNSAT from
    // one host, a 900s timeout on another, same category and budget.
    if !existing.host.is_empty() && existing.host != desired.host {
        bail!(
            "cannot safely --resume: this bank was measured on host [{}], but this machine is \
             [{}]; same-bank rows from different machines are not comparable timings. Use a \
             new output bank.",
            existing.host,
            desired.host
        );
    }

    let mut merged = existing;
    if merged.build_id.is_empty() {
        // Legacy bank: adopt the current identity so rows from here on name
        // their verifier, instead of the bank staying unattributable forever.
        merged.build_id.clone_from(&desired.build_id);
    }
    if merged.compute_backend.is_empty() {
        // Legacy bank: adopt the detected regime so rows from here on are
        // attributed, instead of staying unrecorded forever.
        merged.compute_backend = desired.compute_backend.clone();
    }
    if merged.host.is_empty() {
        merged.host = desired.host.clone();
    }
    if merged.arm.is_none() {
        merged.arm.clone_from(&desired.arm);
    }
    if merged.ambient_env_schema.is_none() {
        merged
            .ambient_env_schema
            .clone_from(&desired.ambient_env_schema);
        merged.ambient_env.clone_from(&desired.ambient_env);
    }
    if merged.instance_selection.is_none() {
        // Legacy bank: make this resume's explicit selection authoritative from
        // now on instead of leaving the bank permanently unsealed.
        merged
            .instance_selection
            .clone_from(&desired.instance_selection);
    }
    for (category, identity) in &desired.selected_rows {
        match merged.selected_rows.get(category) {
            Some(previous) if previous != identity => {
                bail!(
                    "cannot safely --resume: selected row set changed for category {category:?} \
                     (banked count/digest: {}/{}, current: {}/{}); use a new output bank",
                    previous.count,
                    previous.sha256,
                    identity.count,
                    identity.sha256,
                );
            }
            Some(_) => {}
            None => {
                // Compatible category expansion, or a one-category-at-a-time
                // migration of a legacy bank. Omitted historical categories
                // deliberately remain absent rather than receiving a seal this
                // process could not prove.
                merged
                    .selected_rows
                    .insert(category.clone(), identity.clone());
            }
        }
    }
    if merged.selected_row_inputs_schema.is_none()
        && merged
            .selected_rows
            .keys()
            .any(|category| !desired.selected_row_inputs.contains_key(category))
    {
        bail!(
            "cannot safely --resume this legacy bank: it has selected-row provenance for a \
             category omitted from the current plan, but no resolved input-content seal for \
             that category. Include every historical category or use a new output bank."
        );
    }
    for (category, seals) in &desired.selected_row_inputs {
        match merged.selected_row_inputs.get(category) {
            Some(previous) if previous != seals => {
                bail!(
                    "cannot safely --resume: resolved ONNX/VNN-LIB content changed for selected \
                     category {category:?} while its CSV row identity stayed the same; use a new \
                     output bank"
                );
            }
            Some(_) => {}
            None => {
                // Safe only for a newly planned category or an empty/error-only
                // legacy bank. `validate_retained_input_seals` above already
                // refused every reusable historical row that lacks this proof.
                merged
                    .selected_row_inputs
                    .insert(category.clone(), seals.clone());
            }
        }
    }
    if merged.selected_row_inputs_schema.is_none() {
        merged.selected_row_inputs_schema = Some(SELECTED_ROW_INPUTS_SCHEMA.to_string());
    }
    for (category, fingerprint) in &desired.instances_csv {
        match merged.instances_csv.get(category) {
            Some(previous) if previous != fingerprint => {
                bail!(
                    "cannot safely --resume: instances.csv changed for category {category:?}; \
                     use a new output bank"
                );
            }
            Some(_) => {}
            None => {
                merged
                    .instances_csv
                    .insert(category.clone(), fingerprint.clone());
            }
        }
    }
    validate_selected_row_input_manifest(&merged, "merged")?;
    Ok(merged)
}

#[derive(Debug, Clone)]
struct PlannedInstance {
    identity: SweepIdentity,
    category: String,
    base: PathBuf,
    instance: SweepInstance,
    instance_index: usize,
    budget_secs: u64,
    capped_from: Option<u64>,
    /// Whether this row is in the swept subset (`--instances`, else `--limit`).
    /// Unselected rows stay in the plan because `--resume` validates persisted
    /// state against the whole pinned `instances.csv`, not against the subset.
    selected: bool,
    /// Exact resolved bytes this selected row is allowed to use.
    ///
    /// Unselected rows intentionally carry no input seal: they are retained in
    /// the plan only to validate semantic identity/order and are never executed.
    inputs: Option<ResolvedInstanceInputs>,
}

/// Python `measured_row_set_identity` tuple, including the occurrence that
/// distinguishes duplicate official rows and the source CSV index that pins
/// their position.
type SelectedRowSetEntry = (String, String, String, usize, usize);

fn selected_row_set_identity(entries: &[SelectedRowSetEntry]) -> Result<SemanticSetIdentity> {
    semantic_set_identity(MEASURED_ROW_SET_IDENTITY_SCHEMA, entries)
}

fn selected_rows_for_plan(
    plan: &[PlannedInstance],
) -> Result<BTreeMap<String, SemanticSetIdentity>> {
    let mut rows_by_category = BTreeMap::<String, Vec<SelectedRowSetEntry>>::new();
    for entry in plan {
        let rows = rows_by_category.entry(entry.category.clone()).or_default();
        if entry.selected {
            rows.push((
                entry.identity.0.clone(),
                entry.identity.1.clone(),
                entry.identity.2.clone(),
                entry.identity.3,
                entry.instance_index,
            ));
        }
    }
    rows_by_category
        .into_iter()
        .map(|(category, rows)| Ok((category, selected_row_set_identity(&rows)?)))
        .collect()
}

fn selected_row_inputs_for_plan(
    plan: &[PlannedInstance],
) -> Result<BTreeMap<String, Vec<SelectedRowInputSeal>>> {
    let mut seals = BTreeMap::<String, Vec<SelectedRowInputSeal>>::new();
    for entry in plan {
        let category = seals.entry(entry.category.clone()).or_default();
        if !entry.selected {
            continue;
        }
        let inputs = entry.inputs.clone().with_context(|| {
            format!(
                "selected row {:?} has no resolved input-content seal",
                entry.identity
            )
        })?;
        category.push(SelectedRowInputSeal {
            onnx: entry.instance.onnx_field.clone(),
            vnnlib: entry.instance.vnnlib_field.clone(),
            occurrence: entry.identity.3,
            instance_index: entry.instance_index,
            inputs,
        });
    }
    for category in seals.values_mut() {
        category.sort_by(|left, right| {
            left.instance_index
                .cmp(&right.instance_index)
                .then_with(|| left.occurrence.cmp(&right.occurrence))
                .then_with(|| left.onnx.cmp(&right.onnx))
                .then_with(|| left.vnnlib.cmp(&right.vnnlib))
        });
    }
    Ok(seals)
}

/// Re-hash a planned row immediately before reuse/launch and, for a fresh
/// child, again before banking its outcome.
fn verify_planned_inputs(entry: &PlannedInstance, context: &str) -> Result<()> {
    let expected = entry.inputs.as_ref().with_context(|| {
        format!(
            "selected row {:?} has no input-content seal before {context}",
            entry.identity
        )
    })?;
    // Deliberately fresh, with no cross-row metadata memo. The adversarial case
    // is an in-place replacement whose field strings, file size, and coarse
    // mtime can all remain unchanged.
    let found = fingerprint_instance_inputs(
        &entry.base,
        &entry.instance,
        &mut InputFingerprintMemo::new(),
    )
    .with_context(|| format!("re-resolve input files before {context}"))?;
    if &found != expected {
        bail!(
            "selected row {:?} changed resolved ONNX/VNN-LIB content before {context}; the \
             sweep plan sealed different file paths or hashes. Refusing to retain or mix rows; \
             restore the inputs or start a new bank.",
            entry.identity,
        );
    }
    Ok(())
}

fn build_plan(
    opts: &SweepOptions,
    categories: Vec<(String, PathBuf)>,
    binary: &crate::build_id::BinaryIdentity,
    preset_search_dirs: &[PathBuf],
    selectors: Option<&[InstanceSelector]>,
) -> Result<(Vec<PlannedInstance>, SweepManifest)> {
    let mut plan = Vec::new();
    let mut occurrences = HashMap::new();
    let mut instances_fingerprints = BTreeMap::new();
    let wanted = selectors.map(|selectors| selectors.iter().cloned().collect::<HashSet<_>>());
    let mut matched = HashSet::new();
    let mut input_fingerprints = InputFingerprintMemo::new();

    for (category, category_dir) in categories {
        let Some(csv_path) = instances_csv_for(&category_dir, opts.vnnlib_version)? else {
            bail!(
                "category {category:?} has no instances.csv under {}",
                category_dir.display()
            );
        };
        let fingerprint = fingerprint_file(&csv_path, "instances.csv")?;
        if instances_fingerprints
            .insert(category.clone(), fingerprint)
            .is_some()
        {
            bail!("duplicate selected category {category:?}");
        }
        let body = std::fs::read_to_string(&csv_path)
            .with_context(|| format!("read {}", csv_path.display()))?;
        let instances =
            parse_instances(&body).with_context(|| format!("parse {}", csv_path.display()))?;
        if instances.is_empty() {
            bail!("category {category:?} has an empty {}", csv_path.display());
        }
        let base = csv_path.parent().unwrap_or(&category_dir).to_path_buf();
        let take = opts.limit.unwrap_or(instances.len());
        for (index, instance) in instances.into_iter().enumerate() {
            let base_identity = (
                category.clone(),
                instance.onnx_field.clone(),
                instance.vnnlib_field.clone(),
            );
            // A selector names a model/property pair, so it selects EVERY
            // occurrence of that pair — an `instances.csv` that repeats a row
            // means it, and half of a repeated row is not the row.
            let selected = instance_is_selected(wanted.as_ref(), &base_identity, index, take);
            if selected && wanted.is_some() {
                matched.insert(base_identity.clone());
            }
            let identity = occurrence_identity(&mut occurrences, base_identity);
            let capped_from = opts
                .timeout_cap
                .filter(|cap| *cap < instance.budget_secs)
                .map(|_| instance.budget_secs);
            let budget_secs = capped_from.map_or(instance.budget_secs, |_| {
                opts.timeout_cap.unwrap_or(instance.budget_secs)
            });
            let inputs = selected
                .then(|| {
                    fingerprint_instance_inputs(&base, &instance, &mut input_fingerprints)
                        .with_context(|| {
                            format!("seal resolved inputs for category {category:?} row {index}")
                        })
                })
                .transpose()?;
            plan.push(PlannedInstance {
                identity,
                category: category.clone(),
                base: base.clone(),
                instance,
                instance_index: index,
                budget_secs,
                capped_from,
                selected,
                inputs,
            });
        }
    }

    if let Some(selectors) = selectors {
        let names = instances_fingerprints.keys().cloned().collect::<Vec<_>>();
        ensure_every_selector_matched(selectors, &matched, &names)?;
    }
    let selected_rows = selected_rows_for_plan(&plan)?;
    let selected_row_inputs = selected_row_inputs_for_plan(&plan)?;

    let manifest = SweepManifest {
        schema_version: SWEEP_MANIFEST_SCHEMA_VERSION,
        year: opts.year,
        vnnlib_version: opts.vnnlib_version,
        timeout_cap: opts.timeout_cap,
        configs: fingerprint_preset_dirs(preset_search_dirs)?,
        executable: fingerprint_from_binary_identity(binary)?,
        build_id: binary.canonical_line(),
        build_provenance: crate::VNNCOMP_BUILD_PROVENANCE.to_string(),
        target_os: std::env::consts::OS.to_string(),
        target_arch: std::env::consts::ARCH.to_string(),
        compute_backend: crate::compute_backend::detect().summary.clone(),
        host: crate::compute_backend::host().summary(),
        arm: Some(canonical_arm(&opts.arm)),
        ambient_env_schema: Some(crate::flight::AMBIENT_ENV_SCHEMA.to_string()),
        // Reuse flight v3's presence-preserving `vars_os()` snapshot. Calling
        // `std::env::vars()` here would panic on one non-Unicode entry while
        // claiming the same ambient schema as the lossy-but-visible flight
        // record, and could leave manifest/cache provenance weaker than rows.
        ambient_env: Some(crate::flight::ambient_env_snapshot()),
        witness_dir: None,
        instance_selection: Some(requested_instance_selection(opts.limit, selectors)?),
        selected_rows,
        selected_row_inputs_schema: Some(SELECTED_ROW_INPUTS_SCHEMA.to_string()),
        selected_row_inputs,
        instances_csv: instances_fingerprints,
    };
    Ok((plan, manifest))
}

fn validate_state_against_plan(state: &SweepState, plan: &[PlannedInstance]) -> Result<()> {
    let planned = plan
        .iter()
        .map(|entry| (entry.identity.clone(), entry))
        .collect::<HashMap<_, _>>();
    let selected_categories = plan
        .iter()
        .map(|entry| entry.category.as_str())
        .collect::<HashSet<_>>();

    for (identity, row) in state {
        if !selected_categories.contains(row.category.as_str()) {
            continue;
        }
        let entry = planned.get(identity).with_context(|| {
            format!(
                "authoritative metadata identity {identity:?} no longer exists in the \
                 selected instances.csv"
            )
        })?;
        if !entry.selected {
            bail!(
                "cannot safely --resume {identity:?}: this bank already contains a row outside \
                 the current instance selection. A legacy unsealed bank may only adopt a \
                 selection compatible with every row it has measured; use a new output bank"
            );
        }
        let official_budget = row.capped_from.unwrap_or(row.budget_secs);
        if row.budget_secs != entry.budget_secs
            || row.capped_from != entry.capped_from
            || official_budget != entry.instance.budget_secs
            || row.instance_index != entry.instance_index
        {
            bail!(
                "cannot safely --resume {identity:?}: persisted order/budget/cap \
                 ({}, {}, {:?}) differs from the current plan ({}, {}, {:?}); \
                 use a new output bank",
                row.instance_index,
                row.budget_secs,
                row.capped_from,
                entry.instance_index,
                entry.budget_secs,
                entry.capped_from
            );
        }
    }
    Ok(())
}

/// Prove that historical categories omitted from the current invocation do not
/// make a one-time legacy selection adoption lie.
///
/// Banks may intentionally accumulate categories across resumes, so the normal
/// plan validator ignores persisted categories not requested this time. A
/// legacy manifest has no selection seal to protect those rows, however. `All`
/// is compatible by definition. `Limit` and `Exact` both require the omitted
/// category to be loaded: old metadata rows default a missing `instance_index`
/// to zero, so the stored number alone is not proof that a pre-index row belonged
/// to a first-N prefix, and exact membership likewise needs the selector join.
fn validate_legacy_selection_adoption(
    state: &SweepState,
    plan: &[PlannedInstance],
    selection: &InstanceSelection,
) -> Result<()> {
    let planned_categories = plan
        .iter()
        .map(|entry| entry.category.as_str())
        .collect::<HashSet<_>>();
    for (identity, row) in state {
        if planned_categories.contains(row.category.as_str()) {
            continue;
        }
        match selection {
            InstanceSelection::All => {}
            InstanceSelection::Limit { per_category } => {
                bail!(
                    "cannot safely --resume legacy row {identity:?}: category {:?} is omitted \
                     from the current plan, so membership in adopted --limit {per_category} \
                     cannot be proven (pre-index rows deserialize an absent index as zero); \
                     include the category or use a new output bank",
                    row.category,
                );
            }
            InstanceSelection::Exact { .. } => {
                bail!(
                    "cannot safely --resume legacy row {identity:?}: category {:?} is omitted \
                     from the current plan, so membership in the adopted exact --instances \
                     selection cannot be proven; include the category or use a new output bank",
                    row.category,
                );
            }
        }
    }
    Ok(())
}

fn summarize_state(state: &SweepState, wall_secs: f64) -> SweepSummary {
    let mut summary = SweepSummary {
        wall_secs,
        ..SweepSummary::default()
    };
    for row in state.values() {
        summary.record(row);
    }
    summary
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

fn stderr_tail_lines(stderr: &str) -> String {
    let mut lines: Vec<&str> = stderr.lines().rev().take(3).collect();
    lines.reverse();
    lines.join(" | ")
}

pub(crate) fn enforce_scoring_deadline(
    verdict: SweepVerdict,
    elapsed: Duration,
    budget_secs: u64,
) -> (SweepVerdict, Option<String>) {
    if matches!(verdict, SweepVerdict::Sat | SweepVerdict::Unsat)
        && elapsed > Duration::from_secs(budget_secs)
    {
        (
            SweepVerdict::Timeout,
            Some(format!(
                "decided result completed in {:.3}s, beyond the official {budget_secs}s \
                 scoring deadline; watchdog grace does not extend the score",
                elapsed.as_secs_f64()
            )),
        )
    } else {
        (verdict, None)
    }
}

/// The token the child's own deadline machinery writes, verifies on disk, and
/// then hard-stops on. Only this exact token can be promoted below.
const CHILD_DEADLINE_TOKEN: &str = "timeout";

/// Promote a HARD-STOPPED-AT-ITS-OWN-BUDGET row out of `Error`.
///
/// # The defect
///
/// A row that overruns its per-instance budget is stopped by the CHILD's
/// in-process watchdog at `budget + WATCHDOG_GRACE_SECS`. That watchdog writes
/// `timeout` into the result file, RE-READS it to confirm the token landed, and
/// then hard-stops the process by signal. The sweep's outer watchdog is a
/// wider `budget + CHILD_WATCHDOG_GRACE`, so it has NOT fired; the child simply
/// exited by signal, `ran_ok` is `false`, and [`SweepVerdict::classify`]
/// correctly refuses to take a verdict from a failed child — collapsing the row
/// to `Error`.
///
/// Measured consequence: seven `traffic_signs_recognition_2023` rows exited at
/// 485.2-485.9s against a 480s budget and were banked as `error`. Those rows
/// are UN-MEASURED, not crashed, and a family containing them is not bankable.
/// The asymmetry is the bug: [`enforce_scoring_deadline`] already demotes a
/// LATE DECIDED result to `Timeout`, and does nothing at all for `Error`.
///
/// # Why this is the right seam
///
/// [`SweepVerdict::classify`] is left ALONE. Its rule — "a partial verdict from
/// a failed child has no authority" — is correct, and is what
/// `missing_or_partial_results_are_errors_even_after_child_exit` protects: it
/// exists because recording a crashed sweep as `timeout` once turned a wholly
/// broken run into a plausible-looking negative result. So this is a sibling of
/// `enforce_scoring_deadline` at the same call site, applied AFTER it.
///
/// # What survives as a genuine error
///
/// Promotion needs EVERY one of: the row already classified `Error`; the outer
/// watchdog did NOT fire (that path is already `Timeout`); the child exited
/// non-successfully; the result file's first line is EXACTLY
/// [`CHILD_DEADLINE_TOKEN`], which is the only token the deadline machinery
/// self-writes and confirms; and the child ran to at least its own budget.
/// Everything else stays `Error`: a rejected flag, a segfault in teardown, an
/// OOM kill, a missing or empty result file, an unrecognized token, and a
/// `sat`/`unsat`/`unknown` token from a failed child are all untouched. A crash
/// BEFORE the budget cannot be promoted because of the elapsed test, and a
/// crash after it cannot be promoted unless the child had already committed and
/// verified its own `timeout`.
///
/// `timeout` is a no-score token, so promotion can never manufacture a verdict:
/// the worst case is that an unmeasured row reads as unsolved instead of broken,
/// which is exactly what it is.
pub(crate) fn enforce_budget_overrun(
    verdict: SweepVerdict,
    result_first_line: Option<&str>,
    ran_ok: bool,
    watchdog_fired: bool,
    elapsed: Duration,
    budget_secs: u64,
) -> (SweepVerdict, Option<String>, bool) {
    let promotable = verdict == SweepVerdict::Error
        && !watchdog_fired
        && !ran_ok
        && result_first_line.map(str::trim) == Some(CHILD_DEADLINE_TOKEN)
        && elapsed >= Duration::from_secs(budget_secs);
    if !promotable {
        return (verdict, None, false);
    }
    (
        SweepVerdict::Timeout,
        Some(format!(
            "budget overrun: child hard-stopped at {:.3}s of a {budget_secs}s budget after \
             committing `{CHILD_DEADLINE_TOKEN}`; the row is UN-MEASURED, not a crash",
            elapsed.as_secs_f64()
        )),
        true,
    )
}

fn resolve_instance_arguments(
    category_dir: &Path,
    inst: &SweepInstance,
) -> Result<(PathBuf, PathBuf)> {
    let onnx = super::vnncomp_reseed::absolute_model_argument(category_dir, &inst.onnx_field)
        .context("resolve ONNX field")?;
    let property =
        std::fs::canonicalize(category_dir.join(&inst.vnnlib_field)).with_context(|| {
            format!(
                "resolve property {}",
                category_dir.join(&inst.vnnlib_field).display()
            )
        })?;
    Ok((onnx, property))
}

/// The bank's witness directory: a sibling of the official CSV, named after
/// its stem, mirroring `metadata_path_for`/`manifest_path_for`.
fn witness_dir_for(output: &Path) -> PathBuf {
    output.with_extension("witnesses")
}

/// Deterministic witness file name for one planned row.
///
/// The property stem alone is ambiguous — acasxu-style categories run one
/// property file against many networks — so the pinned `instances.csv` row
/// position prefixes it. The stem is reduced to filesystem-safe characters,
/// and path separators in the field never reach the name (only the stem is
/// used), so a hostile field cannot escape its category directory.
fn witness_file_name(instance_index: usize, vnnlib_field: &str) -> String {
    let stem = Path::new(vnnlib_field)
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .filter(|stem| !stem.is_empty())
        .unwrap_or("instance");
    let sanitized = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!("{instance_index:04}-{sanitized}.counterexample")
}

/// Read the witness block a sat child appended to its result file
/// (`sat\n<counterexample_vnnlib>` — the run_instance protocol shape rendered
/// by `commands/vnncomp.rs`). `None` for an unreadable file, a non-sat token,
/// or an empty block: the caller banks the gap visibly instead of failing.
fn read_result_witness(result_file: &Path) -> Option<String> {
    let body = std::fs::read_to_string(result_file).ok()?;
    let (token, witness) = body.split_once('\n')?;
    if SweepVerdict::parse_token(token) != Some(SweepVerdict::Sat) || witness.trim().is_empty() {
        return None;
    }
    Some(witness.to_string())
}

/// Copy one sat row's witness into the bank's witness directory and return
/// the bank-relative path plus content hash for the row's metadata record.
fn persist_witness(
    output: &Path,
    category: &str,
    instance_index: usize,
    vnnlib_field: &str,
    witness: &str,
) -> Result<WitnessRecord> {
    let witness_dir = witness_dir_for(output);
    let dir_name = witness_dir
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .context("derive the witness directory name")?
        .to_string();
    let category_dir = witness_dir.join(category);
    std::fs::create_dir_all(&category_dir)
        .with_context(|| format!("create witness directory {}", category_dir.display()))?;
    let file_name = witness_file_name(instance_index, vnnlib_field);
    let file = category_dir.join(&file_name);
    atomic_write(&file, |handle| {
        handle
            .write_all(witness.as_bytes())
            .with_context(|| format!("write witness {}", file.display()))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(witness.as_bytes());
    Ok(WitnessRecord {
        path: format!("{dir_name}/{category}/{file_name}"),
        sha256: finish_sha256(hasher),
    })
}

/// Witness retention for one banked sat row. NEVER fails the row: an
/// unextractable or uncopyable witness banks as `Missing` (an explicit
/// `null`) and is counted in the sweep summary — the verdict is the product,
/// the witness is its replayable evidence.
fn retain_witness(
    output: &Path,
    category: &str,
    instance_index: usize,
    vnnlib_field: &str,
    witness_text: Option<&str>,
) -> WitnessDisposition {
    let Some(text) = witness_text else {
        return WitnessDisposition::Missing;
    };
    match persist_witness(output, category, instance_index, vnnlib_field, text) {
        Ok(record) => WitnessDisposition::Retained(record),
        Err(error) => {
            eprintln!(
                "WARNING: sat witness for {category}/{vnnlib_field} was not retained: {error:#}"
            );
            WitnessDisposition::Missing
        }
    }
}

/// Best-effort read of the child's flight sidecar next to its results file.
///
/// Missing or unparsable sidecars yield `None`: a row must never fail because
/// its execution trace did — the verdict is the product, the trace is
/// evidence.
fn read_flight_sidecar(result_file: &Path) -> Option<serde_json::Value> {
    let body = std::fs::read_to_string(crate::flight::sidecar_path(result_file)).ok()?;
    serde_json::from_str(&body).ok()
}

/// Cross-check a child's self-reported compiled stamp against the executable
/// seal that selected its private copy.
///
/// A missing sidecar remains best-effort by the flight-recorder contract (hard
/// stops can prevent it). Once a parseable sidecar exists, however, a missing or
/// different build id is affirmative contradictory provenance and may not be
/// banked or served from cache.
fn validate_child_build_receipt(
    flight: Option<&serde_json::Value>,
    expected_build_stamp: &str,
    context: &str,
) -> Result<()> {
    let Some(flight) = flight else {
        return Ok(());
    };
    let observed = flight
        .get("build_id")
        .and_then(serde_json::Value::as_str)
        .with_context(|| {
            format!(
                "{context} has a child flight record without a string build_id; refusing \
                 contradictory executable provenance"
            )
        })?;
    if observed != expected_build_stamp {
        bail!(
            "{context} child flight build identity differs from the early executable seal\n  \
             expected: {expected_build_stamp}\n  observed: {observed}"
        );
    }
    Ok(())
}

/// Everything one child run hands back to the banking loop.
struct InstanceOutcome {
    verdict: SweepVerdict,
    /// See [`SweepRow::budget_overrun`].
    budget_overrun: bool,
    seconds: f64,
    detail: Option<String>,
    flight: Option<serde_json::Value>,
    /// The witness block a sat child appended to its result file, extracted
    /// here because the private instance tempdir is deleted on return.
    witness_text: Option<String>,
}

/// Run one instance in its own child process and classify the outcome.
///
/// Returns the verdict, elapsed seconds, optional harness detail, the child's
/// flight record if one was written, and — for a sat verdict — the witness
/// block from the result file.
fn run_instance(
    exe: &Path,
    category: &str,
    category_dir: &Path,
    inst: &SweepInstance,
    budget: u64,
    configs_dir: Option<&Path>,
    scratch: &Path,
    arm: &[(String, String)],
) -> Result<InstanceOutcome> {
    let instance_dir = tempfile::Builder::new()
        .prefix("instance-")
        .tempdir_in(scratch)
        .context("create private sweep instance directory")?;
    let result_file = instance_dir.path().join("result.txt");

    let (onnx_arg, vnnlib_arg) = match resolve_instance_arguments(category_dir, inst) {
        Ok(arguments) => arguments,
        Err(error) => {
            return Ok(InstanceOutcome {
                verdict: SweepVerdict::Error,
                budget_overrun: false,
                seconds: 0.0,
                detail: Some(format!("resolve instance arguments: {error:#}")),
                flight: None,
                witness_text: None,
            });
        }
    };

    let mut cmd = Command::new(exe);
    cmd.arg("vnncomp")
        .arg("v1")
        .arg(category)
        .arg(&onnx_arg)
        .arg(&vnnlib_arg)
        .arg(&result_file)
        .arg(budget.to_string());
    if let Some(dir) = configs_dir {
        cmd.arg("--configs-dir").arg(dir);
    }
    cmd.env("RUST_LOG", "error");
    for (name, value) in submission_environment() {
        cmd.env(name, value);
    }
    // #arm-overrides: applied AFTER the submission environment, deliberately.
    // The submission set is what the competition wrapper exports and is the
    // baseline every row must share; the arm is the one thing being varied, so it
    // must win any collision rather than be silently dropped. A collision is also
    // worth seeing, so it is reported rather than hidden.
    let submission: BTreeMap<&str, &str> = submission_environment().iter().copied().collect();
    for (name, value) in arm {
        if let Some(base) = submission.get(name.as_str()) {
            if *base != value.as_str() {
                eprintln!(
                    "[sweep-arm] {name}: arm value {value:?} overrides the submission \
                     environment value {base:?} for this row"
                );
            }
        }
        cmd.env(name, value);
    }

    let started = Instant::now();
    let watchdog = Duration::from_secs(budget).saturating_add(CHILD_WATCHDOG_GRACE);
    let output = run_bounded_child(&mut cmd, watchdog, CHILD_STDERR_TAIL_BYTES);

    let (ran_ok, watchdog_fired, elapsed, stderr_tail) = match output {
        Ok(out) => (
            out.success,
            out.timed_out,
            out.elapsed,
            stderr_tail_lines(&out.stderr_tail),
        ),
        Err(err) => (false, false, started.elapsed(), err.to_string()),
    };

    let first_line = read_result_first_line(&result_file);
    let initial_verdict = if watchdog_fired {
        SweepVerdict::Timeout
    } else {
        SweepVerdict::classify(first_line.as_deref(), ran_ok)
    };
    let (verdict, deadline_detail) = enforce_scoring_deadline(initial_verdict, elapsed, budget);
    // Applied AFTER the scoring deadline, and only ever to what is still
    // `Error`: a row the deadline check already demoted is a DECIDED result and
    // must not be relabelled as an overrun.
    let (verdict, overrun_detail, budget_overrun) = enforce_budget_overrun(
        verdict,
        first_line.as_deref(),
        ran_ok,
        watchdog_fired,
        elapsed,
        budget,
    );
    let detail = if watchdog_fired {
        Some(format!(
            "outer watchdog stopped child after {:.1}s",
            watchdog.as_secs_f64()
        ))
    } else if overrun_detail.is_some() {
        overrun_detail
    } else if deadline_detail.is_some() {
        deadline_detail
    } else {
        (verdict == SweepVerdict::Error && !stderr_tail.is_empty())
            .then(|| stderr_tail.chars().take(240).collect::<String>())
    };
    let flight = read_flight_sidecar(&result_file);
    // Extract the witness NOW: the result file lives in the per-instance
    // tempdir, which is deleted when this function returns.
    let witness_text = (verdict == SweepVerdict::Sat)
        .then(|| read_result_witness(&result_file))
        .flatten();
    Ok(InstanceOutcome {
        verdict,
        budget_overrun,
        seconds: elapsed.as_secs_f64(),
        detail,
        flight,
        witness_text,
    })
}

/// The process-wide half of a [`cache::CacheKey`], captured once per sweep.
///
/// Read from the DESIRED manifest, never from a resumed one. A legacy bank may
/// lack `compute_backend`/`host`/`ambient_env` and `--resume` can adopt current
/// values only after the retained row provenance is compatible; keying a cache
/// off the merged manifest would nevertheless describe historical state rather
/// than the process about to run. The desired manifest is the cache input.
#[derive(Debug, Clone)]
struct CacheIdentity {
    exe_sha256: String,
    /// #binary-identity. `exe_sha256` alone would already refuse a swapped
    /// binary, but the identity line also separates two builds that hash the
    /// same file path from different feature tiers or commits in a future
    /// reproducible-build world, and it is what a human reads in the manifest.
    build_id: String,
    build_provenance: String,
    configs_sha256: String,
    compute_backend: String,
    host: String,
    ambient_env: BTreeMap<String, String>,
}

/// Content digest used by cache-key v7 for the complete ONNX dependency
/// closure. Paths are deliberately excluded: cache identity follows bytes and
/// logical external-data locations, while the manifest retains canonical paths
/// for audit and resume checks.
fn onnx_cache_input_digest(models: &[ModelInputFingerprint]) -> Result<String> {
    #[derive(Serialize)]
    struct Closure<'a> {
        schema: &'a str,
        models: Vec<(Option<&'a str>, &'a str, Vec<(&'a str, &'a str)>)>,
    }

    let mut cache_models = Vec::with_capacity(models.len());
    for model in models {
        let dependencies = model
            .external_data
            .as_deref()
            .context("ONNX cache input has no affirmative external-data closure")?;
        cache_models.push((
            model.label.as_deref(),
            model.artifact.sha256.as_str(),
            dependencies
                .iter()
                .map(|dependency| {
                    (
                        dependency.location.as_str(),
                        dependency.artifact.sha256.as_str(),
                    )
                })
                .collect(),
        ));
    }
    let canonical = serde_json::to_vec(&Closure {
        schema: ONNX_CACHE_INPUT_SCHEMA,
        models: cache_models,
    })
    .context("serialize complete ONNX cache input")?;
    let mut hasher = Sha256::new();
    hasher.update(canonical);
    Ok(finish_sha256(hasher))
}

impl CacheIdentity {
    fn from_manifest(manifest: &SweepManifest) -> Self {
        Self {
            exe_sha256: manifest.executable.sha256.clone(),
            build_id: manifest.build_id.clone(),
            build_provenance: manifest.build_provenance.clone(),
            // "No configuration tree" is its own identity: the empty string is
            // not a sha256, so it cannot collide with a real digest.
            configs_sha256: manifest
                .configs
                .as_ref()
                .map_or_else(String::new, |configs| configs.sha256.clone()),
            compute_backend: manifest.compute_backend.clone(),
            host: manifest.host.clone(),
            // `CacheIdentity` is built from the freshly planned (therefore
            // presence-sealed) manifest, never directly from a legacy one.
            ambient_env: manifest.ambient_env.clone().unwrap_or_default(),
        }
    }

    /// Content-address one planned row, or `None` when it cannot be addressed.
    ///
    /// A paired/relational ONNX field is a python literal rather than a file, so
    /// those rows have no model digest and are never cached. That is a MISS —
    /// counted, and followed by a fresh measurement — not an error.
    fn key_for(
        &self,
        entry: &PlannedInstance,
        arm: &[(String, String)],
    ) -> Option<cache::CacheKey> {
        let inputs = entry.inputs.as_ref()?;
        // Relational rows are deliberately not cached. Their literal contains
        // multiple independently labelled models, and the cache has never
        // claimed a cross-process measurement contract for that protocol.
        if inputs.onnx.len() != 1 || inputs.onnx[0].label.is_some() {
            return None;
        }
        cache::CacheKey::new(
            self.exe_sha256.clone(),
            self.build_id.clone(),
            self.build_provenance.clone(),
            self.configs_sha256.clone(),
            // The category is an INPUT, not a label: it names the preset the
            // child loads and gates the category-fenced routes inside it.
            entry.category.clone(),
            onnx_cache_input_digest(&inputs.onnx).ok()?,
            inputs.vnnlib.sha256.clone(),
            entry.budget_secs,
            entry.capped_from,
            arm,
            self.compute_backend.clone(),
            self.host.clone(),
            self.ambient_env.clone(),
        )
        .ok()
    }
}

/// Serve one planned row from the verdict cache.
///
/// Every failure — an admission refusal, an unreadable entry, a stored row this
/// build cannot parse — returns `None`, which the caller counts as a miss and
/// follows with a fresh measurement. A rule refusal is announced rather than
/// swallowed: a cache that is silently refusing everything must not look like a
/// cache that is simply cold.
fn serve_from_cache(
    verdict_cache: &cache::VerdictCache,
    key: &cache::CacheKey,
    entry: &PlannedInstance,
    expected_build_stamp: &str,
) -> Option<(SweepRow, Option<String>)> {
    match verdict_cache.get(key) {
        Ok(measurement) => match serde_json::from_value::<SweepRow>(measurement.row) {
            Ok(row) => {
                if let Err(error) = validate_child_build_receipt(
                    row.flight.as_ref(),
                    expected_build_stamp,
                    "cached measurement",
                ) {
                    eprintln!(
                        "[sweep-cache] {}: {} refused ({error}); measuring it fresh",
                        entry.category, entry.instance.vnnlib_field
                    );
                    None
                } else {
                    Some((row, measurement.witness_text))
                }
            }
            Err(_) => None,
        },
        Err(cache::MissReason::NotPresent) => None,
        Err(reason) => {
            eprintln!(
                "[sweep-cache] {}: {} refused ({reason:?}); measuring it fresh",
                entry.category, entry.instance.vnnlib_field
            );
            None
        }
    }
}

pub(crate) fn run_sweep(opts: &SweepOptions) -> Result<SweepSummary> {
    // State WHICH VERIFIER is about to run, before anything else. This is the
    // line that was missing on 2026-08-21 (#binary-identity).
    crate::build_id::log_once();
    // State the compute regime before planning: it is sealed into the
    // manifest, and a human watching the sweep should not have to wait for
    // the first instance to learn which regime is being measured.
    crate::compute_backend::log_once();
    if opts.timeout_cap == Some(0) {
        bail!("--timeout-cap must be greater than zero");
    }
    if opts.limit == Some(0) {
        bail!("--limit must be greater than zero");
    }
    if opts.resume && opts.overwrite {
        bail!("--resume and --overwrite are mutually exclusive");
    }
    if opts.instances.is_some() && opts.limit.is_some() {
        bail!(
            "--instances and --limit are mutually exclusive: --instances names exactly which \
             rows to run, and layering a first-N cut on top of it makes the swept set depend \
             on CSV order"
        );
    }

    // Parse the subset BEFORE any fingerprinting: a malformed list should cost
    // nothing.
    let selectors = opts
        .instances
        .as_deref()
        .map(|path| {
            let body = std::fs::read_to_string(path)
                .with_context(|| format!("read --instances {}", path.display()))?;
            parse_instance_selectors(&body)
                .with_context(|| format!("parse --instances {}", path.display()))
        })
        .transpose()?;

    let exe = std::env::current_exe().context("locate the running ny executable")?;
    // One authoritative binary identity, captured before configuration/plan
    // provenance is derived. Everything below — manifest, cache namespace,
    // human-readable build id, child receipt expectation — projects from this
    // guard instead of reopening the mutable build target independently.
    let identity_guard = crate::build_id::BinaryIdentityGuard::seal(&exe)
        .context("seal the measuring ny binary before planning the sweep")?;
    let scratch = tempfile::Builder::new()
        .prefix("ny-sweep-")
        .tempdir()
        .context("create private sweep workspace")?;
    let child_exe = sealed_executable_copy(&identity_guard, scratch.path())?;
    let configs_root = opts
        .configs_dir
        .clone()
        .or_else(super::vnncomp::configs_dir_from_env)
        .or_else(|| {
            let mut starts = vec![identity_guard.sealed().path.clone()];
            if let Ok(cwd) = std::env::current_dir() {
                starts.push(cwd);
            }
            super::vnncomp::auto_derive_configs_dir(&starts)
        });
    // #year-scoped-preset: a corpus sweep names its VNN-COMP year explicitly.
    // Passing the parent `configs/` directory through to the native child used
    // to re-enter its yearless newest-first resolver, so once vnncomp26 existed
    // a 2025 safenlp_2024 sweep silently loaded the 2026 preset. Scope the child
    // to exactly `vnncompYY`; if a configuration root was found/requested but
    // that year is absent, fail before measuring a single row.
    let effective_configs_dir = configs_root
        .as_deref()
        .map(|root| {
            super::vnncomp::benchmark_configs_dir(root, opts.year).with_context(|| {
                format!(
                    "configuration root {} has no exact vnncomp{:02} directory for the requested \
                     VNN-COMP {} corpus; refusing newest-year fallback",
                    root.display(),
                    opts.year % 100,
                    opts.year,
                )
            })
        })
        .transpose()?;
    // #sweep-preset-parity: the exact-year check above is the fail-closed gate
    // (a corpus whose year directory is absent is not measurable). Preset
    // RESOLUTION then walks that year and every OLDER one, newest-first —
    // byte-for-byte the order the scored `run_instance.sh` path walks for the
    // newest corpus, and still blind to any NEWER directory for an older one.
    let preset_search_dirs = configs_root
        .as_deref()
        .map(|root| super::vnncomp::benchmark_preset_search_dirs(root, opts.year))
        .unwrap_or_default();
    let categories = selected_categories(opts.year, &opts.categories)?;
    let (plan, desired_manifest) = build_plan(
        opts,
        categories,
        identity_guard.sealed(),
        &preset_search_dirs,
        selectors.as_deref(),
    )?;
    let cache_identity = CacheIdentity::from_manifest(&desired_manifest);
    let cache_dir = opts
        .cache_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_VERDICT_CACHE_DIR));
    let verdict_cache = cache::VerdictCache::new(cache_dir.clone(), opts.cache);

    let output_path = opts.output.clone().unwrap_or_else(|| {
        PathBuf::from(format!("reports/sweeps/vnncomp{}-results.csv", opts.year))
    });
    let metadata_path = metadata_path_for(&output_path);
    let manifest_path = manifest_path_for(&output_path);
    let witness_dir_path = witness_dir_for(&output_path);
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create sweep output directory {}", parent.display()))?;
        }
    }
    ensure_distinct_outputs(&[
        &output_path,
        &metadata_path,
        &manifest_path,
        &witness_dir_path,
    ])?;
    for path in [&output_path, &metadata_path, &manifest_path] {
        if path.exists() && !path.is_file() {
            bail!("sweep artifact is not a regular file: {}", path.display());
        }
    }
    if witness_dir_path.exists() && !witness_dir_path.is_dir() {
        bail!(
            "sweep witness path is not a directory: {}",
            witness_dir_path.display()
        );
    }
    if !opts.resume
        && !opts.overwrite
        && !opts.dry_run
        && ([&output_path, &metadata_path, &manifest_path]
            .iter()
            .any(|path| path.exists())
            || witness_dir_path.exists())
    {
        bail!(
            "refusing to replace an existing sweep artifact for {}; pass --resume to continue \
             a compatible bank or --overwrite to start over explicitly",
            output_path.display()
        );
    }

    let (mut state, mut manifest) = if opts.resume {
        match (metadata_path.exists(), manifest_path.exists()) {
            (true, true) => {
                let state = metadata_state(&metadata_path)?;
                let existing_manifest = read_manifest(&manifest_path)?;
                let adopts_legacy_selection = existing_manifest.instance_selection.is_none();
                if adopts_legacy_selection {
                    validate_legacy_selection_adoption(
                        &state,
                        &plan,
                        desired_manifest.instance_selection.as_ref().context(
                            "current sweep plan is missing its required instance-selection seal",
                        )?,
                    )?;
                }
                let manifest = merge_resume_manifest(existing_manifest, &desired_manifest, &state)?;
                if let Some(category) = state
                    .values()
                    .map(|row| row.category.as_str())
                    .find(|category| !manifest.instances_csv.contains_key(*category))
                {
                    bail!(
                        "authoritative metadata contains category {category:?} absent from its \
                         provenance manifest"
                    );
                }
                validate_state_against_plan(&state, &plan)?;
                if adopts_legacy_selection {
                    eprintln!(
                        "[sweep] warning: legacy manifest had no row-selection seal; adopting \
                         {:?}. Existing rows all belong to this selection, and future changes \
                         will be refused.",
                        manifest.instance_selection
                    );
                }
                (state, manifest)
            }
            (false, false) if !output_path.exists() => (BTreeMap::new(), desired_manifest),
            (false, false) => {
                bail!(
                    "cannot safely --resume {}: an official CSV exists without authoritative \
                     metadata and a provenance manifest",
                    output_path.display()
                );
            }
            _ => {
                bail!(
                    "cannot safely --resume: authoritative metadata {} and manifest {} must \
                     either both exist or both be absent",
                    metadata_path.display(),
                    manifest_path.display()
                );
            }
        }
    } else {
        (BTreeMap::new(), desired_manifest)
    };

    if !opts.dry_run {
        if opts.overwrite && witness_dir_path.exists() {
            // A fresh bank must not inherit witness files from the bank it
            // replaces: stale files would be sealed into the new manifest as
            // if this sweep produced them.
            std::fs::remove_dir_all(&witness_dir_path).with_context(|| {
                format!(
                    "clear stale witness directory {}",
                    witness_dir_path.display()
                )
            })?;
        }
        write_manifest_atomic(&manifest_path, &manifest)?;
        // Metadata is authoritative. Always regenerate the CSV before running;
        // this repairs a crash that happened after metadata publication.
        publish_state(&metadata_path, &output_path, &state)?;
    }

    // Every child below executes `child_exe`, the private read/execute-only copy
    // proven against the ONE early guard. The mutable source path may now be
    // rebuilt without changing which verifier this already-planned bank runs.
    if !opts.dry_run && !opts.json {
        println!("  binary: {}", identity_guard.sealed().short());
    }
    if !opts.json {
        match &effective_configs_dir {
            Some(dir) => println!(
                "  configs: {} (exact VNN-COMP {} preset directory)",
                dir.display(),
                opts.year
            ),
            None => println!(
                "  configs: none (VNN-COMP {} runs use auto defaults)",
                opts.year
            ),
        }
    }
    // #sweep-preset-parity: scope each child to the directory that actually
    // holds ITS category's preset. A category whose preset ships in the exact
    // year directory is unchanged; one whose preset ships only in an older
    // year directory now loads that file, which is what the scored run loads.
    // A category with no preset at or before this year keeps the exact-year
    // directory and therefore keeps auto defaults, unchanged.
    let mut category_configs_dirs: BTreeMap<String, PathBuf> = BTreeMap::new();
    if let Some(root) = configs_root.as_deref() {
        for entry in plan.iter().filter(|entry| entry.selected) {
            if category_configs_dirs.contains_key(&entry.category) {
                continue;
            }
            if let Some(dir) = super::vnncomp::sweep_preset_dir(root, opts.year, &entry.category) {
                category_configs_dirs.insert(entry.category.clone(), dir);
            }
        }
    }
    if !opts.json {
        for (category, dir) in &category_configs_dirs {
            if Some(dir.as_path()) != effective_configs_dir.as_deref() {
                println!(
                    "  preset[{category}]: {} (no vnncomp{:02} preset; taking the scored path's \
                     newest-first fallback)",
                    dir.display(),
                    opts.year % 100
                );
            }
        }
    }

    let sweep_started = Instant::now();
    let mut cache_hits = 0_usize;
    let mut cache_misses = 0_usize;
    let expected_child_build_stamp = identity_guard.sealed().stamp.canonical_line();
    for entry in plan.iter().filter(|entry| entry.selected) {
        verify_planned_inputs(entry, "considering resume reuse or execution")?;
        let reusable = state
            .get(&entry.identity)
            .is_some_and(|row| row.verdict != SweepVerdict::Error);
        if reusable {
            continue;
        }

        if opts.dry_run {
            if !opts.json {
                println!(
                    "  {category}: would run {} [{}s{}]",
                    entry.instance.vnnlib_field,
                    entry.budget_secs,
                    entry.capped_from.map_or(String::new(), |official| format!(
                        ", capped from {official}s"
                    )),
                    category = entry.category,
                );
            }
            continue;
        }

        // #sweep-cache. An unkeyable row counts as a miss rather than vanishing
        // from the accounting: hits + misses must equal the rows this loop
        // considered, or the summary cannot be read as evidence of anything.
        let cache_key = (opts.cache != cache::CacheMode::Off)
            .then(|| cache_identity.key_for(entry, &opts.arm))
            .flatten();
        let served = cache_key.as_ref().and_then(|key| {
            serve_from_cache(&verdict_cache, key, entry, &expected_child_build_stamp)
        });
        if opts.cache != cache::CacheMode::Off {
            if served.is_some() {
                cache_hits += 1;
            } else {
                cache_misses += 1;
            }
        }

        let from_cache = served.is_some();
        let row = match served {
            Some((cached, witness_text)) => {
                // Close the interval around cache lookup. A same-path edit
                // after the preflight check must not be banked under the
                // planned hashes even though the cached verdict itself is for
                // those older bytes.
                verify_planned_inputs(entry, "banking a cached verdict")?;
                // The MEASUREMENT is the cached row's; the IDENTITY is this
                // plan's. The key carries the category (it selects the preset,
                // so it is an input) but deliberately NOT the row position: one
                // model/property pair repeated in an `instances.csv` is the same
                // measurement twice and shares an entry, so serving the stored
                // identity would file the row under the wrong occurrence.
                //
                // The witness is re-banked from its retained TEXT, because the
                // stored `witness` record names a path inside the bank that
                // measured it. Copying that record would claim a counterexample
                // file this bank does not contain.
                let witness = (cached.verdict == SweepVerdict::Sat).then(|| {
                    retain_witness(
                        &output_path,
                        &entry.category,
                        entry.instance_index,
                        &entry.instance.vnnlib_field,
                        witness_text.as_deref(),
                    )
                });
                SweepRow {
                    arm: opts.arm.clone(),
                    category: entry.category.clone(),
                    onnx: entry.instance.onnx_field.clone(),
                    vnnlib: entry.instance.vnnlib_field.clone(),
                    occurrence: entry.identity.3,
                    instance_index: entry.instance_index,
                    verdict: cached.verdict,
                    // Carried from the cached measurement: the flag describes
                    // HOW that row ended, which a replay does not change.
                    budget_overrun: cached.budget_overrun,
                    seconds: cached.seconds,
                    budget_secs: entry.budget_secs,
                    capped_from: entry.capped_from,
                    detail: cached.detail,
                    flight: cached.flight,
                    // The flight record above belongs to a DIFFERENT process.
                    // Mark the row so no downstream gate can read it as
                    // evidence that a child ran here.
                    from_cache: true,
                    witness,
                }
            }
            None => {
                let outcome = run_instance(
                    &child_exe,
                    &entry.category,
                    &entry.base,
                    &entry.instance,
                    entry.budget_secs,
                    category_configs_dirs
                        .get(&entry.category)
                        .map(PathBuf::as_path)
                        .or(effective_configs_dir.as_deref()),
                    scratch.path(),
                    &opts.arm,
                )?;
                validate_child_build_receipt(
                    outcome.flight.as_ref(),
                    &expected_child_build_stamp,
                    "fresh measurement",
                )?;
                // Detect edits that landed while the child was running. The
                // verdict is not published or cached unless the resolved files
                // still hash to the exact pre-run plan.
                verify_planned_inputs(entry, "banking a fresh verdict")?;
                // Every banked sat row carries its witness disposition: a
                // retained record, or an explicit null the summary counts.
                // Non-sat rows carry no witness field at all.
                let witness = (outcome.verdict == SweepVerdict::Sat).then(|| {
                    retain_witness(
                        &output_path,
                        &entry.category,
                        entry.instance_index,
                        &entry.instance.vnnlib_field,
                        outcome.witness_text.as_deref(),
                    )
                });
                let row = SweepRow {
                    arm: opts.arm.clone(),
                    category: entry.category.clone(),
                    onnx: entry.instance.onnx_field.clone(),
                    vnnlib: entry.instance.vnnlib_field.clone(),
                    occurrence: entry.identity.3,
                    instance_index: entry.instance_index,
                    verdict: outcome.verdict,
                    budget_overrun: outcome.budget_overrun,
                    seconds: outcome.seconds,
                    budget_secs: entry.budget_secs,
                    capped_from: entry.capped_from,
                    detail: outcome.detail,
                    flight: outcome.flight,
                    from_cache: false,
                    witness,
                };
                // A harness error is a failure to measure, not a measurement —
                // `--resume` retries one — so storing it would make a transient
                // fault permanent for every future sweep sharing this cache.
                if let Some(key) = cache_key.filter(|_| row.verdict != SweepVerdict::Error) {
                    if let Ok(value) = serde_json::to_value(&row) {
                        verdict_cache.put(
                            &key,
                            &cache::CachedMeasurement {
                                row: value,
                                witness_text: outcome.witness_text,
                            },
                        );
                    }
                }
                row
            }
        };
        state.insert(entry.identity.clone(), row.clone());
        publish_state(&metadata_path, &output_path, &state)?;
        if !opts.json {
            println!(
                "  {}: {} -> {} ({:.1}s/{}s){}{}",
                row.category,
                row.vnnlib,
                row.verdict.as_str(),
                row.seconds,
                row.budget_secs,
                row.detail
                    .as_deref()
                    .map_or(String::new(), |detail| format!("  [{detail}]")),
                if from_cache { "  [cache hit]" } else { "" }
            );
        }
    }

    let mut reported_state = state;
    if opts.dry_run {
        for entry in plan.iter().filter(|entry| entry.selected) {
            reported_state
                .entry(entry.identity.clone())
                .or_insert_with(|| SweepRow {
                    arm: opts.arm.clone(),
                    category: entry.category.clone(),
                    onnx: entry.instance.onnx_field.clone(),
                    vnnlib: entry.instance.vnnlib_field.clone(),
                    occurrence: entry.identity.3,
                    instance_index: entry.instance_index,
                    verdict: SweepVerdict::Unknown,
                    budget_overrun: false,
                    seconds: 0.0,
                    budget_secs: entry.budget_secs,
                    capped_from: entry.capped_from,
                    detail: Some("dry run: instance was not executed".into()),
                    flight: None,
                    from_cache: false,
                    witness: None,
                });
        }
    }
    if !opts.dry_run {
        // Seal the witness directory content into the manifest at sweep end —
        // it grows during the run, so the start-of-run manifest cannot carry
        // it. Derived output provenance only; `--resume` never compares it.
        manifest.witness_dir = witness_dir_path
            .is_dir()
            .then(|| fingerprint_tree(&witness_dir_path, "witness directory"))
            .transpose()?;
        write_manifest_atomic(&manifest_path, &manifest)?;
    }
    let mut summary = summarize_state(&reported_state, sweep_started.elapsed().as_secs_f64());
    summary.cache_hits = cache_hits;
    summary.cache_misses = cache_misses;

    if opts.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!();
        println!(
            "sweep: {} rows | solved {} (sat {}, unsat {}) | timeout {} | unknown {} | error {} | {:.1}s",
            summary.rows,
            summary.solved(),
            summary.sat,
            summary.unsat,
            summary.timeout,
            summary.unknown,
            summary.error,
            summary.wall_secs
        );
        if summary.capped_rows > 0 {
            println!(
                "NOTE: {} row(s) ran BELOW their official budget (--timeout-cap). \
                 This is a lower bound, not a competition-comparable score.",
                summary.capped_rows
            );
        }
        if summary.error > 0 {
            println!(
                "WARNING: {} row(s) failed to run. These are NOT negative results — \
                 investigate before reading anything into this sweep.",
                summary.error
            );
        }
        if summary.budget_overrun > 0 {
            println!(
                "NOTE: {} of those timeout(s) were BUDGET OVERRUNS — the child was \
                 hard-stopped at its own deadline after committing `timeout`. They are \
                 un-measured rows, not crashes, and the bank marks each one \
                 `budget_overrun`.",
                summary.budget_overrun
            );
        }
        if summary.sat_rows_without_witness > 0 {
            println!(
                "WARNING: {} sat row(s) without retained witnesses — organizer-style \
                 replay cannot revalidate them (#witness-retention-gap).",
                summary.sat_rows_without_witness
            );
        }
        if opts.cache != cache::CacheMode::Off {
            // Say it even when it is all misses: a cache that served nothing and
            // a cache that served everything must not print the same thing.
            println!(
                "cache: {} served, {} measured [{}]",
                summary.cache_hits,
                summary.cache_misses,
                cache_dir.display()
            );
            if summary.cache_hits > 0 && summary.cache_misses == 0 {
                // Say it out loud: this invocation started no child. The rows
                // are a replay of earlier measurements, and every flight record
                // in the bank belongs to a different process. Machine readers
                // use the per-row `from_cache` marker.
                println!(
                    "WARNING: every row was SERVED from the cache — this invocation \
                     measured nothing. The bank is a replay; its rows are marked \
                     \"from_cache\": true."
                );
            }
        }
        if !opts.dry_run {
            println!("results: {}", output_path.display());
            println!("metadata: {}", metadata_path.display());
            println!("manifest: {}", manifest_path.display());
        }
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Most manifest-unit tests exercise already sealed inputs and need no
    /// historical rows. Tests of legacy execution-seal adoption call the parent
    /// function directly with an adversarial state.
    fn merge_resume_manifest(
        existing: SweepManifest,
        desired: &SweepManifest,
    ) -> Result<SweepManifest> {
        super::merge_resume_manifest(existing, desired, &SweepState::new())
    }

    fn external_tensor(name: &str, location: &str) -> ny_onnx::onnx_proto::TensorProto {
        ny_onnx::onnx_proto::TensorProto {
            dims: vec![1],
            data_type: 1,
            name: name.to_string(),
            external_data: vec![ny_onnx::onnx_proto::StringStringEntryProto {
                key: "location".into(),
                value: location.into(),
            }],
            data_location: 1,
            ..Default::default()
        }
    }

    fn write_test_onnx(path: &Path, initializer_locations: &[&str], attribute_locations: &[&str]) {
        let initializer = initializer_locations
            .iter()
            .enumerate()
            .map(|(index, location)| external_tensor(&format!("initializer_{index}"), location))
            .collect();
        let attributes = attribute_locations
            .iter()
            .enumerate()
            .map(|(index, location)| ny_onnx::onnx_proto::AttributeProto {
                name: format!("attribute_{index}"),
                r#type: ny_onnx::onnx_proto::attribute_type::TENSOR,
                t: Some(external_tensor(
                    &format!("attribute_tensor_{index}"),
                    location,
                )),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let nodes = (!attributes.is_empty())
            .then(|| ny_onnx::onnx_proto::NodeProto {
                name: "constant_node".into(),
                attribute: attributes,
                ..Default::default()
            })
            .into_iter()
            .collect();
        let model = ny_onnx::onnx_proto::ModelProto {
            graph: Some(ny_onnx::onnx_proto::GraphProto {
                name: "dependency_seal_test".into(),
                initializer,
                node: nodes,
                ..Default::default()
            }),
            ..Default::default()
        };
        let encoded = model.encode_to_vec();
        if path.extension().and_then(std::ffi::OsStr::to_str) == Some("gz") {
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(&encoded).expect("compress test ONNX");
            std::fs::write(path, encoder.finish().expect("finish test ONNX gzip"))
                .expect("write test ONNX gzip");
        } else {
            std::fs::write(path, encoded).expect("write test ONNX");
        }
    }

    #[test]
    fn parses_single_model_rows() {
        let got = parse_instances("onnx/a.onnx,vnnlib/p.vnnlib,100\n").expect("parse");
        assert_eq!(
            got,
            vec![SweepInstance {
                onnx_field: "onnx/a.onnx".into(),
                vnnlib_field: "vnnlib/p.vnnlib".into(),
                budget_secs: 100,
            }]
        );
    }

    #[test]
    fn parses_paired_relational_rows_without_splitting_inside_the_literal() {
        // isomorphic_acasxu_2026 / monotonic_acasxu_2026 shape: the ONNX field
        // contains commas inside quotes, so naive front-counting breaks it.
        let line = "\"[('f', 'onnx/original/m.onnx'), ('g', 'onnx/perturbed/m.onnx')]\",./vnnlib/instance_0.vnnlib,100";
        let got = parse_instances(line).expect("parse");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].vnnlib_field, "./vnnlib/instance_0.vnnlib");
        assert_eq!(got[0].budget_secs, 100);
        assert!(got[0].onnx_field.contains("('f'"));
        assert!(got[0].onnx_field.contains("('g'"));
    }

    #[test]
    fn relational_child_argument_contains_absolute_member_paths() {
        let category = tempfile::tempdir().expect("category");
        std::fs::create_dir_all(category.path().join("onnx")).expect("onnx dir");
        std::fs::create_dir_all(category.path().join("vnnlib")).expect("vnnlib dir");
        let f = category.path().join("onnx/f.onnx");
        let g = category.path().join("onnx/g.onnx.gz");
        let property = category.path().join("vnnlib/p.vnnlib");
        std::fs::write(&f, b"f").expect("f");
        std::fs::write(&g, b"g").expect("g");
        std::fs::write(&property, b"(assert true)").expect("property");
        let inst = SweepInstance {
            onnx_field: "[('f', 'onnx/f.onnx'), ('g', 'onnx/g.onnx.gz')]".into(),
            vnnlib_field: "vnnlib/p.vnnlib".into(),
            budget_secs: 10,
        };

        let (onnx_argument, property_argument) =
            resolve_instance_arguments(category.path(), &inst).expect("arguments");
        let rendered = onnx_argument.to_string_lossy();
        // A relational field renders as a PYTHON LITERAL, so each member path is
        // single-quote escaped on the way in — and on Windows a canonicalized
        // path is full of backslashes (`\\?\C:\...`), every one of which the
        // literal doubles. Comparing against the raw canonical string therefore
        // failed there while the rendering was perfectly correct. Escape the
        // expectation the same way the renderer does, so this asserts the member
        // path is PRESENT rather than asserting a separator convention.
        let as_rendered = |path: &Path| {
            std::fs::canonicalize(path)
                .expect("canonical member")
                .to_str()
                .expect("UTF-8 member")
                .replace('\\', "\\\\")
        };
        assert!(
            rendered.contains(&as_rendered(&f)),
            "rendered={rendered} missing f"
        );
        assert!(
            rendered.contains(&as_rendered(&g)),
            "rendered={rendered} missing g"
        );
        assert_eq!(
            property_argument,
            std::fs::canonicalize(property).expect("canonical property")
        );
    }

    #[test]
    fn accepts_integral_decimal_budgets_like_the_competition_csvs() {
        // metaroom_2023 ships 210.0; traffic_signs ships 480.0.
        let got = parse_instances("a.onnx,p.vnnlib,210.0\n").expect("parse");
        assert_eq!(got[0].budget_secs, 210);
    }

    #[test]
    fn refuses_a_row_with_an_unparsable_budget_instead_of_defaulting() {
        assert!(parse_instances("a.onnx,p.vnnlib,\n").is_err());
        assert!(parse_instances("a.onnx,p.vnnlib,soon\n").is_err());
        assert!(
            parse_instances("a.onnx,ignored,p.vnnlib,10\n").is_err(),
            "extra columns must not silently shift the property and budget"
        );
        assert!(parse_instances("\"unterminated,p.vnnlib,10\n").is_err());
    }

    #[test]
    fn refuses_nonpositive_nonfinite_and_unrepresentable_budgets() {
        for budget in ["0", "-1", "NaN", "inf", "-inf", "1e100"] {
            assert!(
                parse_instances(&format!("a.onnx,p.vnnlib,{budget}\n")).is_err(),
                "budget {budget:?} must be rejected"
            );
        }
        assert!(
            parse_instances("a.onnx,p.vnnlib,0.25\n").is_err(),
            "the integer-only child CLI must not silently round a fractional official budget"
        );
    }

    #[test]
    fn skips_blank_and_comment_lines() {
        let got = parse_instances("\n# header\na.onnx,p.vnnlib,30\n\n").expect("parse");
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn instance_selectors_parse_like_the_csv_rows_they_must_match() {
        let got = parse_instance_selectors(
            "# the moat set\n\
             \n\
             cifar100_2024,onnx/m.onnx,vnnlib/idx_9694.vnnlib\n\
             \"monotonic_acasxu_2026\",\"[('f', 'onnx/a.onnx'), ('g', 'onnx/b.onnx')]\",./vnnlib/instance_0.vnnlib\n",
        )
        .expect("parse");
        assert_eq!(got.len(), 2);
        assert_eq!(
            got[0],
            (
                "cifar100_2024".into(),
                "onnx/m.onnx".into(),
                "vnnlib/idx_9694.vnnlib".into()
            )
        );
        // The relational ONNX field contains commas; splitting instead of
        // CSV-parsing would make those two categories unselectable.
        assert_eq!(got[1].1, "[('f', 'onnx/a.onnx'), ('g', 'onnx/b.onnx')]");
        assert_eq!(got[1].2, "./vnnlib/instance_0.vnnlib");
    }

    #[test]
    fn instance_selectors_refuse_ambiguous_or_empty_subsets() {
        // Two fields cannot say which category a row belongs to, and the sets
        // this flag exists for span categories.
        assert!(parse_instance_selectors("m.onnx,p.vnnlib\n").is_err());
        assert!(parse_instance_selectors("c,m.onnx,p.vnnlib,100\n").is_err());
        assert!(parse_instance_selectors("c,,p.vnnlib\n").is_err());
        assert!(parse_instance_selectors("\"unterminated,m.onnx,p.vnnlib\n").is_err());
        assert!(
            parse_instance_selectors("c,m.onnx,p.vnnlib\nc,m.onnx,p.vnnlib\n").is_err(),
            "a row listed twice means the file does not say what its author thinks"
        );
        assert!(
            parse_instance_selectors("# nothing but a comment\n").is_err(),
            "an empty subset would run nothing and report a clean sweep"
        );
    }

    /// A named row the corpus does not contain must ABORT the sweep. A subset
    /// that silently shrinks reports the same "all rows fine" as a full one,
    /// which is how a regression hides behind a gate that never ran it.
    #[test]
    fn an_unmatched_selector_is_a_hard_error_not_a_skip() {
        let selectors = vec![
            (
                "c".to_string(),
                "m.onnx".to_string(),
                "p.vnnlib".to_string(),
            ),
            (
                "c".to_string(),
                "typo.onnx".to_string(),
                "p.vnnlib".to_string(),
            ),
        ];
        let matched = HashSet::from([selectors[0].clone()]);
        let categories = vec!["c".to_string()];
        let error = ensure_every_selector_matched(&selectors, &matched, &categories)
            .expect_err("an absent row must abort the sweep");
        let rendered = format!("{error}");
        assert!(
            rendered.contains("typo.onnx"),
            "the refusal must NAME the missing row: {rendered}"
        );
        assert!(
            !rendered.contains("c,m.onnx,p.vnnlib"),
            "a matched row is not missing: {rendered}"
        );

        let matched = selectors.iter().cloned().collect::<HashSet<_>>();
        ensure_every_selector_matched(&selectors, &matched, &categories)
            .expect("a fully resolved subset runs");
    }

    #[test]
    fn missing_or_partial_results_are_errors_even_after_child_exit() {
        // The bug this encodes: a rejected CLI flag exits non-zero and writes no
        // result. Recording that as `timeout` turned a wholly broken 96-row
        // sweep into a plausible-looking negative result.
        assert_eq!(SweepVerdict::classify(None, true), SweepVerdict::Error);
        assert_eq!(SweepVerdict::classify(None, false), SweepVerdict::Error);
        assert_eq!(SweepVerdict::classify(Some(""), false), SweepVerdict::Error);
        assert_eq!(
            SweepVerdict::classify(Some("sat"), false),
            SweepVerdict::Error,
            "a partial verdict from a failed child has no authority"
        );
        assert_eq!(
            SweepVerdict::classify(Some("not-a-verdict"), true),
            SweepVerdict::Error
        );
    }

    /// The exact measured signature: a `traffic_signs_recognition_2023` row at a
    /// 480s budget whose child's in-process watchdog committed `timeout` at
    /// 485s and then hard-stopped the process by signal. The outer watchdog
    /// (budget + 30s = 510s) has NOT fired, so `watchdog_fired` is false and
    /// `ran_ok` is false.
    fn overrun_signature() -> (
        SweepVerdict,
        Option<&'static str>,
        bool,
        bool,
        Duration,
        u64,
    ) {
        (
            SweepVerdict::Error,
            Some("timeout"),
            false,
            false,
            Duration::from_secs_f64(485.4),
            480,
        )
    }

    /// (c) A BUDGET OVERRUN classifies as `Timeout`, flagged `budget_overrun`,
    /// with a detail string that says so. It is an UN-MEASURED row, not a crash.
    #[test]
    fn a_budget_overrun_classifies_as_a_flagged_timeout() {
        let (verdict, line, ran_ok, watchdog, elapsed, budget) = overrun_signature();
        let (promoted, detail, overrun) =
            enforce_budget_overrun(verdict, line, ran_ok, watchdog, elapsed, budget);

        assert_eq!(
            promoted,
            SweepVerdict::Timeout,
            "a row hard-stopped at its own budget is unsolved, not broken"
        );
        assert!(overrun, "the row must be FLAGGED as a budget overrun");
        let detail = detail.expect("a promoted row must carry its reason");
        assert!(
            detail.contains("budget overrun") && detail.contains("485.400"),
            "detail must name the defect and the overrun time: {detail}"
        );

        // The scored token stays inside the organizers' vocabulary: an overrun
        // is an unsolved row, and `budgetoverrun` is not a legal CSV token.
        assert_eq!(promoted.as_str(), "timeout");

        // Every trailing-whitespace shape of the committed token is the same
        // token; the child writes it with a newline.
        for written in ["timeout", "timeout\n", " timeout ", "timeout\r\n"] {
            let (promoted, _, overrun) = enforce_budget_overrun(
                SweepVerdict::Error,
                Some(written),
                false,
                false,
                elapsed,
                budget,
            );
            assert_eq!(promoted, SweepVerdict::Timeout, "{written:?}");
            assert!(overrun, "{written:?}");
        }

        // Exactly at the budget is still an overrun (the child's own watchdog
        // fires at budget + grace, so `elapsed >= budget` always holds there).
        let (promoted, _, overrun) = enforce_budget_overrun(
            SweepVerdict::Error,
            Some("timeout"),
            false,
            false,
            Duration::from_secs(budget),
            budget,
        );
        assert_eq!(promoted, SweepVerdict::Timeout);
        assert!(overrun);
    }

    /// (c) A GENUINE CRASH still classifies as `Error`, and is never flagged.
    /// Each case below removes exactly one leg of the overrun signature.
    #[test]
    fn a_genuine_crash_still_classifies_as_an_error() {
        let (_, _, _, _, late, budget) = overrun_signature();
        let early = Duration::from_secs_f64(12.0);

        let cases: Vec<(&str, SweepVerdict, Option<&str>, bool, bool, Duration)> = vec![
            // A rejected CLI flag: exits non-zero, writes no result file at all.
            (
                "no result file",
                SweepVerdict::Error,
                None,
                false,
                false,
                late,
            ),
            // A crashed child that produced an empty result file.
            (
                "empty result",
                SweepVerdict::Error,
                Some(""),
                false,
                false,
                late,
            ),
            // A crashed child whose result file holds an unrecognized token.
            (
                "unrecognized token",
                SweepVerdict::Error,
                Some("not-a-verdict"),
                false,
                false,
                late,
            ),
            // A partial DECIDED verdict from a failed child: never rehabilitated.
            (
                "partial sat",
                SweepVerdict::Error,
                Some("sat"),
                false,
                false,
                late,
            ),
            (
                "partial unsat",
                SweepVerdict::Error,
                Some("unsat"),
                false,
                false,
                late,
            ),
            (
                "partial unknown",
                SweepVerdict::Error,
                Some("unknown"),
                false,
                false,
                late,
            ),
            // The child's own explicit `error` token.
            (
                "explicit error token",
                SweepVerdict::Error,
                Some("error"),
                false,
                false,
                late,
            ),
            // A SIGSEGV at 12s on a row with a stale `timeout` on disk: far
            // inside the budget, so the elapsed test refuses it.
            (
                "crash before budget",
                SweepVerdict::Error,
                Some("timeout"),
                false,
                false,
                early,
            ),
            // A child that SUCCEEDED but wrote nothing recognizable: a harness
            // fault, not an overrun, even though it ran long.
            (
                "successful child, no token",
                SweepVerdict::Error,
                None,
                true,
                false,
                late,
            ),
        ];

        for (name, verdict, line, ran_ok, watchdog, elapsed) in cases {
            let (out, detail, overrun) =
                enforce_budget_overrun(verdict, line, ran_ok, watchdog, elapsed, budget);
            assert_eq!(
                out,
                SweepVerdict::Error,
                "{name}: must stay a genuine error"
            );
            assert!(detail.is_none(), "{name}: must carry no overrun detail");
            assert!(!overrun, "{name}: must not be flagged as a budget overrun");
        }

        // A row the OUTER watchdog stopped is already `Timeout` on its own
        // path; the promoter must not relabel it as a budget overrun.
        let (out, _, overrun) = enforce_budget_overrun(
            SweepVerdict::Timeout,
            Some("timeout"),
            false,
            true,
            late,
            budget,
        );
        assert_eq!(out, SweepVerdict::Timeout);
        assert!(
            !overrun,
            "an outer-watchdog timeout is not a budget overrun"
        );

        // And a DECIDED verdict is never touched, at any elapsed time.
        for decided in [
            SweepVerdict::Sat,
            SweepVerdict::Unsat,
            SweepVerdict::Unknown,
        ] {
            let (out, detail, overrun) =
                enforce_budget_overrun(decided, Some("timeout"), false, false, late, budget);
            assert_eq!(out, decided, "{decided:?} must be left alone");
            assert!(detail.is_none() && !overrun);
        }
    }

    /// (c) The two are DISTINGUISHABLE in the bank: an overrun row serializes
    /// `budget_overrun: true` into the metadata sidecar while keeping the
    /// organizer-legal `timeout` token in the official CSV; a clean timeout and
    /// a genuine error both omit the field entirely, so every pre-existing bank
    /// round-trips byte-identically.
    #[test]
    fn the_bank_distinguishes_a_budget_overrun_from_a_crash_and_a_clean_timeout() {
        let mut overrun = metadata_row("traffic", 0, SweepVerdict::Timeout);
        overrun.budget_overrun = true;
        overrun.detail = Some("budget overrun: child hard-stopped at 485.400s".into());
        let clean = metadata_row("traffic", 1, SweepVerdict::Timeout);
        let crashed = metadata_row("traffic", 2, SweepVerdict::Error);

        let overrun_json = serde_json::to_string(&overrun).expect("serialize");
        assert!(
            overrun_json.contains("\"budget_overrun\":true"),
            "an overrun must be machine-readable in the bank: {overrun_json}"
        );
        for row in [&clean, &crashed] {
            let json = serde_json::to_string(row).expect("serialize");
            assert!(
                !json.contains("budget_overrun"),
                "a non-overrun row must omit the flag entirely: {json}"
            );
        }

        // Round-trip, including a legacy row written before the flag existed.
        let parsed: SweepRow = serde_json::from_str(&overrun_json).expect("round-trip");
        assert!(parsed.budget_overrun);
        let legacy = serde_json::to_string(&clean).expect("serialize");
        let parsed_legacy: SweepRow = serde_json::from_str(&legacy).expect("legacy round-trip");
        assert!(
            !parsed_legacy.budget_overrun,
            "a bank written before the flag existed must read as not-an-overrun"
        );

        // The summary counts overruns separately from timeouts and errors,
        // WITHOUT removing them from the timeout bucket — an overrun is still
        // an unsolved row for every downstream consumer.
        let mut summary = SweepSummary::default();
        summary.record(&overrun);
        summary.record(&clean);
        summary.record(&crashed);
        assert_eq!(summary.timeout, 2);
        assert_eq!(summary.error, 1);
        assert_eq!(summary.budget_overrun, 1);
        assert_eq!(summary.solved(), 0);
    }

    #[test]
    fn classifies_the_organizer_verdict_vocabulary() {
        assert_eq!(SweepVerdict::classify(Some("sat"), true), SweepVerdict::Sat);
        assert_eq!(
            SweepVerdict::classify(Some("unsat\n"), true),
            SweepVerdict::Unsat
        );
        assert_eq!(
            SweepVerdict::classify(Some("unknown"), true),
            SweepVerdict::Unknown
        );
        assert_eq!(
            SweepVerdict::classify(Some("timeout"), true),
            SweepVerdict::Timeout
        );
        assert_eq!(
            SweepVerdict::classify(Some("error"), true),
            SweepVerdict::Error
        );
        assert_eq!(
            SweepVerdict::classify(Some("error_exit_code_1"), true),
            SweepVerdict::Error
        );
    }

    #[test]
    fn emitted_rows_round_trip_through_the_csv_parser() {
        // Regression: the paired/relational ONNX field contains commas. Written
        // bare it produced a 6-field row that split back into the wrong columns,
        // corrupting both downstream scoring and this runner's own --resume.
        let onnx = "[('f', 'onnx/a.onnx'), ('g', 'onnx/b.onnx')]";
        let line = format!(
            "{},{},{},0.0,{},{:.1}",
            csv_escape("monotonic_acasxu_2026"),
            csv_escape(onnx),
            csv_escape("./vnnlib/instance_0.vnnlib"),
            SweepVerdict::Sat.as_str(),
            1.1
        );
        let fields = parse_csv_line(&line).expect("valid emitted CSV");
        assert_eq!(fields.len(), 6, "row must have exactly 6 columns: {line}");
        assert_eq!(fields[0], "monotonic_acasxu_2026");
        assert_eq!(fields[1], onnx);
        assert_eq!(fields[2], "./vnnlib/instance_0.vnnlib");
        assert_eq!(fields[3], "0.0");
        assert_eq!(fields[4], "sat");
    }

    fn metadata_row(category: &str, occurrence: usize, verdict: SweepVerdict) -> SweepRow {
        SweepRow {
            arm: Vec::new(),
            budget_overrun: false,
            category: category.into(),
            onnx: "model.onnx".into(),
            vnnlib: "property.vnnlib".into(),
            occurrence,
            instance_index: occurrence,
            verdict,
            seconds: 1.0,
            budget_secs: 10,
            capped_from: None,
            detail: None,
            flight: None,
            from_cache: false,
            witness: None,
        }
    }

    fn planned_row(
        category: &str,
        onnx: &str,
        vnnlib: &str,
        occurrence: usize,
        instance_index: usize,
        selected: bool,
    ) -> PlannedInstance {
        PlannedInstance {
            identity: (category.into(), onnx.into(), vnnlib.into(), occurrence),
            category: category.into(),
            base: PathBuf::from("."),
            instance: SweepInstance {
                onnx_field: onnx.into(),
                vnnlib_field: vnnlib.into(),
                budget_secs: 10,
            },
            instance_index,
            budget_secs: 10,
            capped_from: None,
            selected,
            inputs: selected.then(|| ResolvedInstanceInputs {
                onnx: vec![ModelInputFingerprint {
                    label: None,
                    artifact: ArtifactFingerprint {
                        canonical_path: format!("/corpus/{category}/{onnx}"),
                        sha256: format!("onnx-{onnx}"),
                    },
                    external_data: Some(Vec::new()),
                }],
                vnnlib: ArtifactFingerprint {
                    canonical_path: format!("/corpus/{category}/{vnnlib}"),
                    sha256: format!("vnnlib-{vnnlib}"),
                },
            }),
        }
    }

    #[test]
    fn authoritative_metadata_preserves_duplicate_occurrences_during_error_retry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = dir.path().join("results.csv");
        let metadata = metadata_path_for(&output);
        let mut state = SweepState::new();
        state.insert(
            ("c".into(), "model.onnx".into(), "property.vnnlib".into(), 0),
            metadata_row("c", 0, SweepVerdict::Error),
        );
        state.insert(
            ("c".into(), "model.onnx".into(), "property.vnnlib".into(), 1),
            metadata_row("c", 1, SweepVerdict::Unsat),
        );
        publish_state(&metadata, &output, &state).expect("initial state");

        let first = state
            .get_mut(&("c".into(), "model.onnx".into(), "property.vnnlib".into(), 0))
            .expect("first occurrence");
        first.verdict = SweepVerdict::Sat;
        publish_state(&metadata, &output, &state).expect("replacement state");

        let loaded = metadata_state(&metadata).expect("load replacement");
        assert_eq!(loaded.len(), 2);
        assert_eq!(
            loaded
                .get(&("c".into(), "model.onnx".into(), "property.vnnlib".into(), 0))
                .expect("occurrence zero")
                .verdict,
            SweepVerdict::Sat
        );
        assert_eq!(
            loaded
                .get(&("c".into(), "model.onnx".into(), "property.vnnlib".into(), 1))
                .expect("occurrence one")
                .verdict,
            SweepVerdict::Unsat
        );
    }

    #[test]
    fn authoritative_metadata_regenerates_a_stale_or_corrupt_official_csv() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = dir.path().join("results.csv");
        let metadata = metadata_path_for(&output);
        let row = metadata_row("c", 0, SweepVerdict::Unsat);
        let mut state = SweepState::new();
        state.insert(
            ("c".into(), "model.onnx".into(), "property.vnnlib".into(), 0),
            row,
        );
        write_metadata_atomic(&metadata, &state).expect("metadata");
        std::fs::write(&output, "corrupt,stale,csv\n").expect("corrupt CSV");

        write_results_atomic(
            &output,
            &metadata_state(&metadata).expect("authoritative state"),
        )
        .expect("regenerate");
        assert_eq!(
            std::fs::read_to_string(output).expect("results"),
            "c,model.onnx,property.vnnlib,0.0,unsat,1.0\n"
        );
    }

    #[test]
    fn publication_uses_pinned_instance_order_not_identity_sort_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = dir.path().join("results.csv");
        let mut first = metadata_row("c", 0, SweepVerdict::Unsat);
        first.onnx = "z-first.onnx".into();
        first.vnnlib = "z-first.vnnlib".into();
        let mut second = metadata_row("c", 0, SweepVerdict::Sat);
        second.onnx = "a-second.onnx".into();
        second.vnnlib = "a-second.vnnlib".into();
        second.instance_index = 1;
        let mut state = SweepState::new();
        state.insert(
            ("c".into(), first.onnx.clone(), first.vnnlib.clone(), 0),
            first,
        );
        state.insert(
            ("c".into(), second.onnx.clone(), second.vnnlib.clone(), 0),
            second,
        );

        write_results_atomic(&output, &state).expect("results");
        let body = std::fs::read_to_string(output).expect("read results");
        assert!(body.starts_with("c,z-first.onnx,z-first.vnnlib"));
    }

    #[cfg(unix)]
    #[test]
    fn output_artifacts_must_not_alias_the_same_inode() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = dir.path().join("results.csv");
        let metadata = dir.path().join("results.metadata.jsonl");
        std::fs::write(&output, b"existing").expect("output");
        std::fs::hard_link(&output, &metadata).expect("hard link");
        assert!(ensure_distinct_outputs(&[&output, &metadata]).is_err());
    }

    #[test]
    fn full_bank_summary_keeps_prior_errors_and_caps_visible() {
        let mut state = SweepState::new();
        let mut error = metadata_row("c", 0, SweepVerdict::Error);
        error.budget_secs = 5;
        error.capped_from = Some(10);
        state.insert(
            ("c".into(), "model.onnx".into(), "property.vnnlib".into(), 0),
            error,
        );
        let summary = summarize_state(&state, 0.5);
        assert_eq!(summary.rows, 1);
        assert_eq!(summary.error, 1);
        assert_eq!(summary.capped_rows, 1);
    }

    #[test]
    fn resume_refuses_budget_or_cap_drift() {
        let identity = ("c".into(), "model.onnx".into(), "property.vnnlib".into(), 0);
        let plan = vec![PlannedInstance {
            identity: identity.clone(),
            category: "c".into(),
            base: PathBuf::from("."),
            instance: SweepInstance {
                onnx_field: "model.onnx".into(),
                vnnlib_field: "property.vnnlib".into(),
                budget_secs: 10,
            },
            instance_index: 0,
            budget_secs: 5,
            capped_from: Some(10),
            selected: true,
            inputs: None,
        }];
        let mut state = SweepState::new();
        state.insert(identity, metadata_row("c", 0, SweepVerdict::Unsat));
        assert!(
            validate_state_against_plan(&state, &plan).is_err(),
            "an uncapped persisted row must not enter a capped bank"
        );
    }

    #[test]
    fn legacy_adoption_refuses_rows_outside_the_current_exact_or_limit_selection() {
        let row_identity: InstanceSelector =
            ("c".into(), "model.onnx".into(), "property.vnnlib".into());
        let other = HashSet::from([("c".into(), "other.onnx".into(), "other.vnnlib".into())]);
        for (label, instance_index, selected) in [
            (
                "exact",
                0,
                instance_is_selected(Some(&other), &row_identity, 0, usize::MAX),
            ),
            ("limit", 1, instance_is_selected(None, &row_identity, 1, 1)),
        ] {
            assert!(!selected, "test fixture must be outside {label} selection");
            let identity = (
                row_identity.0.clone(),
                row_identity.1.clone(),
                row_identity.2.clone(),
                0,
            );
            let plan = vec![PlannedInstance {
                identity: identity.clone(),
                category: "c".into(),
                base: PathBuf::from("."),
                instance: SweepInstance {
                    onnx_field: row_identity.1.clone(),
                    vnnlib_field: row_identity.2.clone(),
                    budget_secs: 10,
                },
                instance_index,
                budget_secs: 10,
                capped_from: None,
                selected,
                inputs: None,
            }];
            let mut row = metadata_row("c", 0, SweepVerdict::Unsat);
            row.instance_index = instance_index;
            let state = SweepState::from([(identity, row)]);
            let error = validate_state_against_plan(&state, &plan)
                .expect_err("legacy adoption must not relabel existing rows");
            assert!(error
                .to_string()
                .contains("outside the current instance selection"));
        }
    }

    #[test]
    fn legacy_adoption_checks_rows_from_categories_omitted_by_this_resume() {
        let mut old_row = metadata_row("old", 0, SweepVerdict::Unsat);
        old_row.instance_index = 1;
        let state = SweepState::from([(
            (
                "old".into(),
                "model.onnx".into(),
                "property.vnnlib".into(),
                0,
            ),
            old_row,
        )]);
        let plan = vec![PlannedInstance {
            identity: ("current".into(), "m.onnx".into(), "p.vnnlib".into(), 0),
            category: "current".into(),
            base: PathBuf::from("."),
            instance: SweepInstance {
                onnx_field: "m.onnx".into(),
                vnnlib_field: "p.vnnlib".into(),
                budget_secs: 10,
            },
            instance_index: 0,
            budget_secs: 10,
            capped_from: None,
            selected: true,
            inputs: None,
        }];

        validate_legacy_selection_adoption(&state, &plan, &InstanceSelection::All)
            .expect("all rows is compatible with accumulated historical categories");
        let limit_error = validate_legacy_selection_adoption(
            &state,
            &plan,
            &InstanceSelection::Limit { per_category: 2 },
        )
        .expect_err("an omitted category cannot prove whether its index was recorded");
        assert!(limit_error
            .to_string()
            .contains("pre-index rows deserialize an absent index as zero"));

        let exact = exact_selection("current,m.onnx,p.vnnlib\n");
        let exact_error = validate_legacy_selection_adoption(&state, &plan, &exact)
            .expect_err("an omitted category's exact membership cannot be reconstructed");
        assert!(exact_error.to_string().contains("cannot be proven"));
    }

    #[test]
    fn pre_index_legacy_row_cannot_masquerade_as_a_limit_prefix_member() {
        let row = metadata_row("old", 0, SweepVerdict::Unsat);
        let mut json = serde_json::to_value(row).expect("row JSON");
        json.as_object_mut()
            .expect("row object")
            .remove("instance_index");
        let legacy: SweepRow =
            serde_json::from_value(json).expect("pre-index metadata remains readable");
        assert_eq!(
            legacy.instance_index, 0,
            "this default is why the omitted category cannot prove Limit membership"
        );
        let identity = (
            legacy.category.clone(),
            legacy.onnx.clone(),
            legacy.vnnlib.clone(),
            legacy.occurrence,
        );
        let state = SweepState::from([(identity, legacy)]);
        let plan = vec![planned_row("current", "m.onnx", "p.vnnlib", 0, 0, true)];

        let error = validate_legacy_selection_adoption(
            &state,
            &plan,
            &InstanceSelection::Limit { per_category: 1 },
        )
        .expect_err("defaulted index zero is not affirmative first-one provenance");
        assert!(error.to_string().contains("include the category"));
    }

    /// #arm-sealing: a resume must REFUSE when the ambient lever set differs.
    ///
    /// `run_instance` only ADDS to the child environment, so a lever exported in
    /// the parent shell is inherited by every row while being invisible in the
    /// artifact. Without this guard a bank can be half-measured under one arm and
    /// half under another, and nothing downstream can tell.
    #[test]
    fn resume_refuses_a_different_ambient_lever_set() {
        let mut sealed = test_manifest(None);
        sealed.ambient_env = Some(BTreeMap::from([(
            "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN".to_string(),
            "1".to_string(),
        )]));
        let desired = test_manifest(None); // empty ambient set
        let error = merge_resume_manifest(sealed, &desired)
            .expect_err("a resume under a different arm must refuse");
        let rendered = format!("{error}");
        assert!(
            rendered.contains("ambient lever set"),
            "unexpected error: {rendered}"
        );
        assert!(
            rendered.contains("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"),
            "the refusal must NAME the difference: {rendered}"
        );
    }

    /// A legacy bank (sealed before this field existed) adopts the current set
    /// rather than blocking resume — the same rule the backend and host fields
    /// already use, so old banks stay resumable.
    #[test]
    fn resume_allows_a_legacy_bank_with_no_sealed_ambient_set() {
        let mut legacy = test_manifest(None);
        legacy.ambient_env_schema = None;
        // Pre-v2 manifests may already contain an ambient map; without the
        // schema token it is still legacy capture, not current proof.
        legacy.ambient_env = Some(BTreeMap::new());
        let mut desired = test_manifest(None);
        desired.ambient_env = Some(BTreeMap::from([(
            "OMP_NUM_THREADS".to_string(),
            "1".to_string(),
        )]));
        let merged =
            merge_resume_manifest(legacy, &desired).expect("an empty legacy bank is safe to adopt");
        assert_eq!(merged.ambient_env, desired.ambient_env);
    }

    /// An identical ambient set resumes cleanly.
    #[test]
    fn resume_allows_an_identical_ambient_lever_set() {
        let arm = BTreeMap::from([("OMP_NUM_THREADS".to_string(), "1".to_string())]);
        let mut sealed = test_manifest(None);
        sealed.ambient_env = Some(arm.clone());
        let mut desired = test_manifest(None);
        desired.ambient_env = Some(arm);
        merge_resume_manifest(sealed, &desired).expect("same arm must resume");
    }

    #[test]
    fn affirmative_empty_execution_seals_are_present_and_refuse_drift() {
        let clean = test_manifest(None);
        let json = serde_json::to_value(&clean).expect("manifest JSON");
        assert_eq!(json["arm"], serde_json::json!([]));
        assert_eq!(
            json["ambient_env_schema"],
            crate::flight::AMBIENT_ENV_SCHEMA
        );
        assert_eq!(json["ambient_env"], serde_json::json!({}));

        let mut dirty = test_manifest(None);
        dirty.ambient_env = Some(BTreeMap::from([(
            "NY_MARGIN_ROW_CLASSWISE".to_string(),
            "1".to_string(),
        )]));
        let error = merge_resume_manifest(clean, &dirty).expect_err(
            "a clean current bank must not masquerade as legacy and adopt contamination",
        );
        assert!(error.to_string().contains("ambient lever set"));

        let mut unknown = test_manifest(None);
        unknown.ambient_env_schema = Some("ny-cli/ambient-env/v999".into());
        let error = merge_resume_manifest(unknown, &test_manifest(None))
            .expect_err("an unknown present ambient schema must never be treated as legacy");
        assert!(error
            .to_string()
            .contains("unsupported ambient-environment schema"));

        let mut legacy_json = json;
        let object = legacy_json.as_object_mut().expect("manifest object");
        object.remove("arm");
        object.remove("ambient_env_schema");
        let legacy: SweepManifest =
            serde_json::from_value(legacy_json).expect("pre-seal manifest remains readable");
        assert!(
            legacy.arm.is_none()
                && legacy.ambient_env_schema.is_none()
                && legacy.ambient_env == Some(BTreeMap::new())
        );
    }

    #[test]
    fn sealed_manifest_refuses_a_different_explicit_arm() {
        let mut existing = test_manifest(None);
        existing.arm = Some(vec![("NY_MARGIN_ROW_LRU".into(), "16".into())]);
        let mut desired = test_manifest(None);
        desired.arm = Some(vec![("NY_MARGIN_ROW_LRU".into(), "128".into())]);

        let error = merge_resume_manifest(existing, &desired)
            .expect_err("a partial bank cannot resume under another --lever treatment");
        assert!(error.to_string().contains("explicit arm"));
    }

    #[test]
    fn legacy_explicit_arm_adoption_requires_every_retained_row_to_match() {
        let arm = canonical_arm(&[
            ("NY_MARGIN_ROW_PARALLEL".into(), "8".into()),
            ("NY_MARGIN_ROW_LRU".into(), "16".into()),
        ]);
        let mut legacy = test_manifest(None);
        legacy.arm = None;
        let mut desired = test_manifest(None);
        desired.arm = Some(arm.clone());

        let identity = ("c".into(), "model.onnx".into(), "property.vnnlib".into(), 0);
        let mut matching = metadata_row("c", 0, SweepVerdict::Unsat);
        matching.arm = arm.iter().rev().cloned().collect();
        matching.flight = Some(serde_json::json!({
            "ambient_env_schema": crate::flight::AMBIENT_ENV_SCHEMA,
            "ambient_env": {
                "NY_MARGIN_ROW_LRU": "16",
                "NY_MARGIN_ROW_PARALLEL": "8",
                "OMP_NUM_THREADS": "1"
            }
        }));
        let state = SweepState::from([(identity.clone(), matching)]);
        let merged = super::merge_resume_manifest(legacy.clone(), &desired, &state)
            .expect("row arm proves a canonical legacy adoption");
        assert_eq!(merged.arm, Some(arm));

        let mismatched = metadata_row("c", 0, SweepVerdict::Unsat);
        let state = SweepState::from([(identity, mismatched)]);
        let error = super::merge_resume_manifest(legacy, &desired, &state)
            .expect_err("a baseline row cannot be relabeled as a treatment");
        assert!(error
            .to_string()
            .contains("retained measurement records explicit arm"));
    }

    #[test]
    fn legacy_ambient_adoption_requires_exact_child_flight_provenance() {
        let mut legacy = test_manifest(None);
        legacy.ambient_env_schema = None;
        // A pre-v3 manifest map is readable but cannot itself prove that every
        // Rayon worker-count variable was absent; only the row's v3 flight can
        // do that.
        legacy.ambient_env = Some(BTreeMap::new());
        let desired = test_manifest(None);
        let identity = ("c".into(), "model.onnx".into(), "property.vnnlib".into(), 0);

        let mut matching = metadata_row("c", 0, SweepVerdict::Unsat);
        matching.flight = Some(serde_json::json!({
            "ambient_env_schema": crate::flight::AMBIENT_ENV_SCHEMA,
            "ambient_env": {"OMP_NUM_THREADS": "1"}
        }));
        let state = SweepState::from([(identity.clone(), matching)]);
        let merged = super::merge_resume_manifest(legacy.clone(), &desired, &state)
            .expect("the child flight proves the effective clean environment");
        assert_eq!(merged.ambient_env, Some(BTreeMap::new()));
        assert_eq!(
            merged.ambient_env_schema.as_deref(),
            Some(crate::flight::AMBIENT_ENV_SCHEMA)
        );

        let mut pre_v3 = metadata_row("c", 0, SweepVerdict::Unsat);
        pre_v3.flight = Some(serde_json::json!({
            "schema_version": 3,
            "ambient_env_schema": "ny-cli/ambient-env/v2",
            "ambient_env": {"OMP_NUM_THREADS": "1"}
        }));
        let state = SweepState::from([(identity.clone(), pre_v3)]);
        let error = super::merge_resume_manifest(legacy.clone(), &desired, &state)
            .expect_err("a v2 flight omitted deprecated Rayon worker-count state");
        assert!(error.to_string().contains("pre-v3 captures"));

        let mut contaminated = metadata_row("c", 0, SweepVerdict::Unsat);
        contaminated.flight = Some(serde_json::json!({
            "ambient_env_schema": crate::flight::AMBIENT_ENV_SCHEMA,
            "ambient_env": {
                "NY_MARGIN_ROW_CLASSWISE": "1",
                "OMP_NUM_THREADS": "1"
            }
        }));
        let state = SweepState::from([(identity.clone(), contaminated)]);
        let error = super::merge_resume_manifest(legacy.clone(), &desired, &state)
            .expect_err("undeclared ambient contamination cannot be adopted as clean");
        assert!(error.to_string().contains("NY_MARGIN_ROW_CLASSWISE"));

        let missing_flight = metadata_row("c", 0, SweepVerdict::Unsat);
        let state = SweepState::from([(identity, missing_flight)]);
        let error = super::merge_resume_manifest(legacy, &desired, &state)
            .expect_err("a retained verdict without flight provenance must fail closed");
        assert!(error.to_string().contains("no child flight record"));
    }

    #[test]
    fn current_manifest_seals_are_rechecked_against_every_retained_measurement() {
        let sealed = test_manifest(None);
        let desired = sealed.clone();
        let identity = ("c".into(), "model.onnx".into(), "property.vnnlib".into(), 0);

        let mut matching = metadata_row("c", 0, SweepVerdict::Unsat);
        matching.flight = Some(serde_json::json!({
            "ambient_env_schema": crate::flight::AMBIENT_ENV_SCHEMA,
            "ambient_env": {"OMP_NUM_THREADS": "1"}
        }));
        let state = SweepState::from([(identity.clone(), matching)]);
        super::merge_resume_manifest(sealed.clone(), &desired, &state)
            .expect("a retained row matching both current seals resumes");

        let mut mixed_arm = metadata_row("c", 0, SweepVerdict::Unsat);
        mixed_arm.arm = vec![("NY_MARGIN_ROW_LRU".into(), "128".into())];
        let state = SweepState::from([(identity.clone(), mixed_arm)]);
        let error = super::merge_resume_manifest(sealed.clone(), &desired, &state)
            .expect_err("a sealed manifest must not hide a row measured under another arm");
        assert!(error.to_string().contains("expected bank arm"));

        let mut contaminated = metadata_row("c", 0, SweepVerdict::Unsat);
        contaminated.flight = Some(serde_json::json!({
            "ambient_env_schema": crate::flight::AMBIENT_ENV_SCHEMA,
            "ambient_env": {
                "NY_MARGIN_ROW_CLASSWISE": "1",
                "OMP_NUM_THREADS": "1"
            }
        }));
        let state = SweepState::from([(identity.clone(), contaminated)]);
        let error = super::merge_resume_manifest(sealed.clone(), &desired, &state)
            .expect_err("a sealed manifest must not hide ambient-contaminated rows");
        assert!(error.to_string().contains("NY_MARGIN_ROW_CLASSWISE"));

        let mut pre_v3 = metadata_row("c", 0, SweepVerdict::Unsat);
        pre_v3.flight = Some(serde_json::json!({
            "schema_version": 3,
            "ambient_env_schema": "ny-cli/ambient-env/v2",
            "ambient_env": {"OMP_NUM_THREADS": "1"}
        }));
        let state = SweepState::from([(identity, pre_v3)]);
        let error = super::merge_resume_manifest(sealed, &desired, &state)
            .expect_err("a current manifest token cannot launder a pre-v3 row flight");
        assert!(error.to_string().contains("pre-v3 captures"));
    }

    fn exact_selection(body: &str) -> InstanceSelection {
        let selectors = parse_instance_selectors(body).expect("valid exact selector");
        requested_instance_selection(None, Some(&selectors)).expect("selection identity")
    }

    #[test]
    fn exact_selector_identity_is_order_insensitive_and_python_compatible() {
        let forward = exact_selection(
            "c,m.onnx,p.vnnlib\n\
             d,x.onnx,q.vnnlib\n",
        );
        let reordered = exact_selection(
            "# comments and row order are not selection semantics\n\
             d,x.onnx,q.vnnlib\n\
             c,m.onnx,p.vnnlib\n",
        );
        assert_eq!(forward, reordered);

        let manifest_shape = serde_json::to_value(&forward).expect("selection JSON");
        assert_eq!(manifest_shape["mode"], "exact");
        assert_eq!(manifest_shape["identity"]["count"], 2);
        assert_eq!(
            manifest_shape["identity"]["schema"],
            INSTANCE_SELECTOR_IDENTITY_SCHEMA
        );

        let InstanceSelection::Exact { identity } = forward else {
            panic!("an --instances request must be exact");
        };
        assert_eq!(identity.schema, INSTANCE_SELECTOR_IDENTITY_SCHEMA);
        assert_eq!(identity.count, 2);
        assert_eq!(
            identity.sha256,
            // Python `json.dumps(..., ensure_ascii=False, sort_keys=True,
            // separators=(\",\", \":\"))`, locked so the search harness and
            // native manifest cannot silently drift apart.
            "158b2b4f72e17e621f7981fc7273a11d4b1aaf0f1502506cbdfb151f5e30b83b"
        );
    }

    #[test]
    fn selected_row_seals_match_python_and_distinguish_partial_duplicate_sets() {
        let plan = vec![
            planned_row("c", "m.onnx", "p.vnnlib", 0, 2, true),
            planned_row("c", "m.onnx", "p.vnnlib", 1, 5, true),
            planned_row("c", "unselected.onnx", "q.vnnlib", 0, 7, false),
            // A planned category with no selected rows is a proven empty set,
            // not missing legacy provenance.
            planned_row("d", "x.onnx", "q.vnnlib", 0, 0, false),
        ];
        let sealed = selected_rows_for_plan(&plan).expect("selected row seals");
        assert_eq!(sealed.len(), 2);

        let c = &sealed["c"];
        assert_eq!(c.schema, MEASURED_ROW_SET_IDENTITY_SCHEMA);
        assert_eq!(
            c.count, 2,
            "both duplicate corpus occurrences must be sealed"
        );
        assert_eq!(
            c.sha256,
            // Python `measured_row_set_identity` over
            // [(c,m,p,0,2), (c,m,p,1,5)].
            "99a3767547dda44cff5878fbd1a592f85ca38a271f704b69095ca5cc7a60be53"
        );
        let d = &sealed["d"];
        assert_eq!(d.count, 0);
        assert_eq!(
            d.sha256,
            "abea9155bcec1da8b43764abb29a251959149de590fb283625f3559ab8a5071a"
        );

        let partial =
            selected_row_set_identity(&[("c".into(), "m.onnx".into(), "p.vnnlib".into(), 0, 2)])
                .expect("partial row seal");
        assert_eq!(
            partial.sha256,
            "fdbb96ef4ecaeba8e80eb8f2c6c0155d24b6a1841102269468a00dfc198c2f35"
        );
        assert_ne!(
            c, &partial,
            "equal selector intent cannot make a partial bank look like the full duplicate set"
        );

        let reversed = selected_row_set_identity(&[
            ("c".into(), "m.onnx".into(), "p.vnnlib".into(), 1, 5),
            ("c".into(), "m.onnx".into(), "p.vnnlib".into(), 0, 2),
        ])
        .expect("reordered row seal");
        assert_eq!(c, &reversed, "plan enumeration order is not set identity");
    }

    #[test]
    fn resolved_input_seal_detects_in_place_model_and_property_edits() {
        let category = tempfile::tempdir().expect("category");
        std::fs::create_dir_all(category.path().join("onnx")).expect("onnx dir");
        std::fs::create_dir_all(category.path().join("vnnlib")).expect("vnnlib dir");
        let first = category.path().join("onnx/f.onnx");
        let second = category.path().join("onnx/g.onnx");
        let property = category.path().join("vnnlib/p.vnnlib");
        write_test_onnx(&first, &[], &[]);
        write_test_onnx(&second, &[], &[]);
        std::fs::write(&property, b"CCCC").expect("property");
        let instance = SweepInstance {
            onnx_field: "[('f', 'onnx/f.onnx'), ('g', 'onnx/g.onnx')]".into(),
            vnnlib_field: "vnnlib/p.vnnlib".into(),
            budget_secs: 10,
        };
        let inputs = fingerprint_instance_inputs(
            category.path(),
            &instance,
            &mut InputFingerprintMemo::new(),
        )
        .expect("initial content seal");
        assert_eq!(inputs.onnx.len(), 2, "every relational member is sealed");
        assert_eq!(inputs.onnx[0].label.as_deref(), Some("f"));
        assert_eq!(inputs.onnx[1].label.as_deref(), Some("g"));
        let entry = PlannedInstance {
            identity: (
                "relational".into(),
                instance.onnx_field.clone(),
                instance.vnnlib_field.clone(),
                0,
            ),
            category: "relational".into(),
            base: category.path().to_path_buf(),
            instance,
            instance_index: 0,
            budget_secs: 10,
            capped_from: None,
            selected: true,
            inputs: Some(inputs),
        };
        verify_planned_inputs(&entry, "unchanged test row").expect("unchanged inputs verify");

        let original_second = std::fs::read(&second).expect("read second model");
        let mut changed_second = original_second.clone();
        let last = changed_second.last_mut().expect("nonempty encoded model");
        *last ^= 1;
        // Same path and same byte length: identity must come from CONTENT, not
        // the CSV field, size, or a coarse metadata cache.
        std::fs::write(&second, changed_second).expect("mutate relational member in place");
        let error = verify_planned_inputs(&entry, "reusing an old verdict")
            .expect_err("an in-place ONNX edit must invalidate resume reuse");
        assert!(error
            .to_string()
            .contains("changed resolved ONNX/VNN-LIB content"));

        std::fs::write(&second, original_second).expect("restore model");
        verify_planned_inputs(&entry, "after model restore").expect("restored model verifies");
        std::fs::write(&property, b"EEEE").expect("mutate property in place");
        let error = verify_planned_inputs(&entry, "banking a fresh verdict")
            .expect_err("an in-place VNN-LIB edit must invalidate the measured row");
        assert!(error
            .to_string()
            .contains("changed resolved ONNX/VNN-LIB content"));
    }

    #[test]
    fn resolved_input_seal_and_cache_digest_include_external_onnx_data() {
        let category = tempfile::tempdir().expect("category");
        std::fs::create_dir_all(category.path().join("onnx/nested")).expect("onnx dir");
        std::fs::create_dir_all(category.path().join("vnnlib")).expect("vnnlib dir");
        let model = category.path().join("onnx/model.onnx.gz");
        let first_sidecar = category.path().join("onnx/weights.data");
        let second_sidecar = category.path().join("onnx/nested/constants.data");
        let property = category.path().join("vnnlib/p.vnnlib");
        std::fs::write(&first_sidecar, b"AAAA").expect("first sidecar");
        std::fs::write(&second_sidecar, b"BBBB").expect("second sidecar");
        std::fs::write(&property, b"(assert true)").expect("property");
        // The initializer and AttributeProto.t share one sidecar. The closure is
        // set-like, so that file appears once beside the distinct nested file.
        write_test_onnx(
            &model,
            &["weights.data", "nested/constants.data"],
            &["./weights.data"],
        );
        let instance = SweepInstance {
            onnx_field: "onnx/model.onnx.gz".into(),
            vnnlib_field: "vnnlib/p.vnnlib".into(),
            budget_secs: 10,
        };
        let inputs = fingerprint_instance_inputs(
            category.path(),
            &instance,
            &mut InputFingerprintMemo::new(),
        )
        .expect("seal gzipped external-data model");
        let dependencies = inputs.onnx[0]
            .external_data
            .as_ref()
            .expect("affirmative external-data closure");
        assert_eq!(
            dependencies
                .iter()
                .map(|dependency| dependency.location.as_str())
                .collect::<Vec<_>>(),
            ["nested/constants.data", "weights.data"],
            "initializer and attribute dependencies are normalized, sorted, and deduplicated"
        );
        let original_cache_digest =
            onnx_cache_input_digest(&inputs.onnx).expect("initial cache input digest");
        let entry = PlannedInstance {
            identity: (
                "external".into(),
                instance.onnx_field.clone(),
                instance.vnnlib_field.clone(),
                0,
            ),
            category: "external".into(),
            base: category.path().to_path_buf(),
            instance,
            instance_index: 0,
            budget_secs: 10,
            capped_from: None,
            selected: true,
            inputs: Some(inputs),
        };
        verify_planned_inputs(&entry, "unchanged external-data row")
            .expect("unchanged closure verifies");

        std::fs::write(&first_sidecar, b"CCCC").expect("mutate sidecar in place");
        let changed = fingerprint_instance_inputs(
            category.path(),
            &entry.instance,
            &mut InputFingerprintMemo::new(),
        )
        .expect("seal changed sidecar");
        assert_ne!(
            original_cache_digest,
            onnx_cache_input_digest(&changed.onnx).expect("changed cache input digest"),
            "a sidecar-only edit must select a different cache entry"
        );
        let error = verify_planned_inputs(&entry, "reusing a sidecar-backed verdict")
            .expect_err("a sidecar-only edit must invalidate resume/cache reuse");
        assert!(
            error
                .to_string()
                .contains("changed resolved ONNX/VNN-LIB content"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn manifest_exposes_and_compares_selected_row_input_hashes() {
        let existing = test_manifest(None);
        let json = serde_json::to_value(&existing).expect("manifest JSON");
        assert_eq!(
            json["selected_row_inputs_schema"],
            SELECTED_ROW_INPUTS_SCHEMA
        );
        assert_eq!(
            json["selected_row_inputs"]["c"][0]["inputs"]["onnx"][0]["artifact"]["sha256"],
            "model-bytes"
        );
        assert_eq!(
            json["selected_row_inputs"]["c"][0]["inputs"]["onnx"][0]["external_data"],
            serde_json::json!([]),
            "schema v2 must affirmatively seal even an empty external-data closure"
        );
        assert_eq!(
            json["selected_row_inputs"]["c"][0]["inputs"]["vnnlib"]["sha256"],
            "property-bytes"
        );

        let mut changed = existing.clone();
        changed
            .selected_row_inputs
            .get_mut("c")
            .expect("category")
            .get_mut(0)
            .expect("row")
            .inputs
            .onnx
            .get_mut(0)
            .expect("model")
            .artifact
            .sha256 = "different-model-bytes".into();
        let error = merge_resume_manifest(existing, &changed)
            .expect_err("same CSV row over changed bytes must not resume");
        assert!(
            error.to_string().contains("content changed"),
            "unexpected refusal: {error}"
        );
    }

    #[test]
    fn legacy_retained_verdict_without_input_hashes_fails_closed() {
        let desired = test_manifest(None);
        let mut legacy_json = serde_json::to_value(&desired).expect("manifest JSON");
        let object = legacy_json.as_object_mut().expect("manifest object");
        object.remove("selected_row_inputs_schema");
        object.remove("selected_row_inputs");
        let legacy: SweepManifest =
            serde_json::from_value(legacy_json).expect("legacy manifest parses");
        let identity = ("c".into(), "model.onnx".into(), "property.vnnlib".into(), 0);
        let state = SweepState::from([(identity, metadata_row("c", 0, SweepVerdict::Unsat))]);
        let error = super::merge_resume_manifest(legacy.clone(), &desired, &state)
            .expect_err("today's bytes cannot prove what a legacy verdict measured");
        assert!(
            error
                .to_string()
                .contains("cannot adopt today's file bytes retroactively"),
            "unexpected refusal: {error}"
        );

        let error_only = SweepState::from([(
            ("c".into(), "model.onnx".into(), "property.vnnlib".into(), 0),
            metadata_row("c", 0, SweepVerdict::Error),
        )]);
        let migrated = super::merge_resume_manifest(legacy, &desired, &error_only)
            .expect("an error is retried, never reused as a measurement");
        assert_eq!(
            migrated.selected_row_inputs_schema.as_deref(),
            Some(SELECTED_ROW_INPUTS_SCHEMA)
        );
    }

    #[test]
    fn selected_row_input_v1_or_missing_external_closure_fails_closed() {
        let mut v1 = test_manifest(None);
        v1.selected_row_inputs_schema = Some("ny-cli/selected-row-inputs/v1".into());
        let error = validate_selected_row_input_manifest(&v1, "legacy")
            .expect_err("v1 cannot prove that a model had no external tensor data");
        assert!(
            error.to_string().contains("expected"),
            "unexpected refusal: {error:#}"
        );

        let mut missing_json = serde_json::to_value(test_manifest(None)).expect("manifest JSON");
        missing_json["selected_row_inputs"]["c"][0]["inputs"]["onnx"][0]
            .as_object_mut()
            .expect("model seal")
            .remove("external_data");
        let missing: SweepManifest =
            serde_json::from_value(missing_json).expect("presence-sensitive legacy manifest");
        let error = validate_selected_row_input_manifest(&missing, "tampered")
            .expect_err("schema v2 cannot infer an omitted closure as empty");
        assert!(
            error
                .to_string()
                .contains("no affirmative ONNX external-data closure"),
            "unexpected refusal: {error:#}"
        );
    }

    #[test]
    fn resume_allows_the_same_exact_selector_and_refuses_a_different_one() {
        let selection = exact_selection("c,m.onnx,p.vnnlib\n");
        let mut existing = test_manifest(None);
        existing.instance_selection = Some(selection.clone());
        let mut same = test_manifest(None);
        same.instance_selection = Some(selection);
        merge_resume_manifest(existing.clone(), &same).expect("same selector resumes");

        let mut different = test_manifest(None);
        different.instance_selection = Some(exact_selection("c,m.onnx,other.vnnlib\n"));
        let error = merge_resume_manifest(existing, &different)
            .expect_err("a different exact subset must not enter the bank");
        assert!(
            error.to_string().contains("instance selection"),
            "unexpected refusal: {error}"
        );
    }

    #[test]
    fn resume_refuses_a_limit_change_but_allows_the_same_limit() {
        let mut existing = test_manifest(None);
        existing.instance_selection = Some(InstanceSelection::Limit { per_category: 4 });
        let same = existing.clone();
        merge_resume_manifest(existing.clone(), &same).expect("same first-N selection resumes");

        let mut different = test_manifest(None);
        different.instance_selection = Some(InstanceSelection::Limit { per_category: 8 });
        let error = merge_resume_manifest(existing, &different)
            .expect_err("changing --limit must not change a bank's measured prefix");
        assert!(error.to_string().contains("instance selection"));
    }

    #[test]
    fn a_legacy_manifest_adopts_the_current_selection_once() {
        let mut json = serde_json::to_value(test_manifest(None)).expect("manifest JSON");
        json.as_object_mut()
            .expect("manifest object")
            .remove("instance_selection");
        let legacy: SweepManifest =
            serde_json::from_value(json).expect("pre-selection manifest remains readable");
        assert!(legacy.instance_selection.is_none());

        let selection = exact_selection("c,m.onnx,p.vnnlib\n");
        let mut desired = test_manifest(None);
        desired.instance_selection = Some(selection.clone());
        let merged = merge_resume_manifest(legacy, &desired)
            .expect("a legacy bank adopts this resume's explicit selection");
        assert_eq!(merged.instance_selection, Some(selection));
    }

    #[test]
    fn a_legacy_manifest_adopts_only_currently_proven_category_row_seals() {
        let mut legacy = test_manifest(None);
        legacy.instances_csv.insert(
            "old".into(),
            ArtifactFingerprint {
                canonical_path: "/corpus/old/instances.csv".into(),
                sha256: "old-instances".into(),
            },
        );
        let mut json = serde_json::to_value(legacy).expect("legacy manifest JSON");
        json.as_object_mut()
            .expect("manifest object")
            .remove("selected_rows");
        json.as_object_mut()
            .expect("manifest object")
            .remove("selected_row_inputs_schema");
        json.as_object_mut()
            .expect("manifest object")
            .remove("selected_row_inputs");
        let legacy: SweepManifest =
            serde_json::from_value(json).expect("pre-row-seal manifest remains readable");
        assert!(legacy.selected_rows.is_empty());

        // This invocation planned only c. Its seal is provable and adopted;
        // historical category old stays deliberately unsealed rather than
        // receiving a digest for rows this process never loaded.
        let desired = test_manifest(None);
        let merged = merge_resume_manifest(legacy, &desired).expect("legacy row-seal migration");
        assert!(merged.selected_rows.contains_key("c"));
        assert!(!merged.selected_rows.contains_key("old"));
        assert!(merged.instances_csv.contains_key("old"));
    }

    #[test]
    fn an_exact_selector_selects_every_duplicate_corpus_occurrence() {
        let selector: InstanceSelector = ("c".into(), "m.onnx".into(), "p.vnnlib".into());
        let wanted = HashSet::from([selector.clone()]);
        let corpus = [selector.clone(), selector.clone()];
        assert_eq!(
            corpus
                .iter()
                .enumerate()
                .filter(|(index, row)| instance_is_selected(Some(&wanted), row, *index, 0))
                .count(),
            2,
            "a pair selector names the row, so every official duplicate occurrence runs"
        );

        let identity =
            exact_instance_selection_identity(&[selector.clone()]).expect("one semantic selector");
        assert_eq!(
            identity.count, 1,
            "selector count is semantic; duplicate multiplicity is pinned by instances.csv"
        );
        let mut occurrences = HashMap::new();
        assert_eq!(occurrence_identity(&mut occurrences, selector.clone()).3, 0);
        assert_eq!(occurrence_identity(&mut occurrences, selector).3, 1);
    }

    fn test_manifest(cap: Option<u64>) -> SweepManifest {
        SweepManifest {
            schema_version: SWEEP_MANIFEST_SCHEMA_VERSION,
            year: 2026,
            vnnlib_version: None,
            timeout_cap: cap,
            configs: None,
            executable: ArtifactFingerprint {
                canonical_path: "/bin/ny".into(),
                sha256: "exe".into(),
            },
            build_id: "NY_BUILD_ID_V1 version=0.3.4745 features=mip exe=exe".into(),
            build_provenance: "sealed".into(),
            target_os: "linux".into(),
            target_arch: "x86_64".into(),
            compute_backend: "cuda [test]".into(),
            host: "gb10-test | Test CPU | 20 cores | 128 GiB".into(),
            arm: Some(Vec::new()),
            ambient_env_schema: Some(crate::flight::AMBIENT_ENV_SCHEMA.to_string()),
            ambient_env: Some(BTreeMap::new()),
            witness_dir: None,
            instance_selection: Some(InstanceSelection::All),
            selected_rows: BTreeMap::from([(
                "c".into(),
                selected_row_set_identity(&[(
                    "c".into(),
                    "model.onnx".into(),
                    "property.vnnlib".into(),
                    0,
                    0,
                )])
                .expect("selected rows identity"),
            )]),
            selected_row_inputs_schema: Some(SELECTED_ROW_INPUTS_SCHEMA.to_string()),
            selected_row_inputs: BTreeMap::from([(
                "c".into(),
                vec![SelectedRowInputSeal {
                    onnx: "model.onnx".into(),
                    vnnlib: "property.vnnlib".into(),
                    occurrence: 0,
                    instance_index: 0,
                    inputs: ResolvedInstanceInputs {
                        onnx: vec![ModelInputFingerprint {
                            label: None,
                            artifact: ArtifactFingerprint {
                                canonical_path: "/corpus/c/model.onnx".into(),
                                sha256: "model-bytes".into(),
                            },
                            external_data: Some(Vec::new()),
                        }],
                        vnnlib: ArtifactFingerprint {
                            canonical_path: "/corpus/c/property.vnnlib".into(),
                            sha256: "property-bytes".into(),
                        },
                    },
                }],
            )]),
            instances_csv: BTreeMap::from([(
                "c".into(),
                ArtifactFingerprint {
                    canonical_path: "/corpus/c/instances.csv".into(),
                    sha256: "instances".into(),
                },
            )]),
        }
    }

    /// #sweep-cache: the cache identity is the DESIRED manifest's, so the arm a
    /// row is keyed under is the one this process will actually run — including
    /// the ambient set, whose absence from the key was the defect that let two
    /// runs differing only by an exported `NY_*` share an entry.
    #[test]
    fn cache_identity_carries_the_ambient_arm_and_a_distinct_no_configs_digest() {
        let mut manifest = test_manifest(None);
        manifest.ambient_env = Some(BTreeMap::from([(
            "NY_ROOT_GEMM".to_string(),
            "faer".to_string(),
        )]));
        let identity = CacheIdentity::from_manifest(&manifest);
        assert_eq!(identity.ambient_env, manifest.ambient_env.clone().unwrap());
        assert_eq!(identity.exe_sha256, "exe");
        assert_eq!(identity.host, manifest.host);
        assert!(
            identity.configs_sha256.is_empty(),
            "no configuration tree must key distinctly, and no real sha256 is empty"
        );
    }

    /// #sweep-cache, the wiring half of the category defect. `cache.rs` proves
    /// the field separates two keys; this proves `key_for` actually POPULATES it
    /// from the plan. Without it the two rows below — byte-identical model and
    /// property in two categories, which is what a synthetic corpus and several
    /// real VNN-COMP families look like — shared one entry, so a single cold
    /// `--cache read-write` sweep measured one row and served the other a
    /// verdict produced under the wrong preset.
    #[test]
    fn the_cache_key_separates_two_categories_over_identical_instance_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut planned = Vec::new();
        for category in ["demo_a", "demo_b"] {
            let base = dir.path().join(category);
            std::fs::create_dir_all(base.join("onnx")).expect("onnx dir");
            std::fs::create_dir_all(base.join("vnnlib")).expect("vnnlib dir");
            // Identical BYTES in both categories: only the category differs.
            std::fs::write(base.join("onnx/m.onnx"), b"same-model").expect("model");
            std::fs::write(base.join("vnnlib/p.vnnlib"), b"same-property").expect("property");
            planned.push(PlannedInstance {
                identity: (
                    category.into(),
                    "onnx/m.onnx".into(),
                    "vnnlib/p.vnnlib".into(),
                    0,
                ),
                category: category.into(),
                base,
                instance: SweepInstance {
                    onnx_field: "onnx/m.onnx".into(),
                    vnnlib_field: "vnnlib/p.vnnlib".into(),
                    budget_secs: 100,
                },
                instance_index: 0,
                budget_secs: 100,
                capped_from: None,
                selected: true,
                inputs: Some(ResolvedInstanceInputs {
                    onnx: vec![ModelInputFingerprint {
                        label: None,
                        artifact: ArtifactFingerprint {
                            canonical_path: format!("/{category}/m.onnx"),
                            sha256: "same-model".into(),
                        },
                        external_data: Some(Vec::new()),
                    }],
                    vnnlib: ArtifactFingerprint {
                        canonical_path: format!("/{category}/p.vnnlib"),
                        sha256: "same-property".into(),
                    },
                }),
            });
        }

        let identity = CacheIdentity::from_manifest(&test_manifest(None));
        let digest_of = |entry: &PlannedInstance| {
            identity
                .key_for(entry, &[])
                .expect("both rows are keyable")
                .digest()
                .expect("digest")
        };
        let a = digest_of(&planned[0]);
        let b = digest_of(&planned[1]);
        assert_ne!(
            a, b,
            "the category selects the preset the child loads; serving across it \
             is a verdict for a run that never happened"
        );
    }

    #[test]
    fn manifest_refuses_resume_under_a_different_binary_identity() {
        // #binary-identity. A bank half-measured by `0.3.4745 features=mip` and
        // half by `0.3.4746 features=mip,cuda` is two verifiers wearing one
        // bank's name — the 2026-08-21 failure, in resume form. The refusal
        // must print BOTH ids, because "they differ" is not actionable.
        let existing = test_manifest(None);
        let mut desired = test_manifest(None);
        desired.build_id = "NY_BUILD_ID_V1 version=0.3.4746 features=mip,cuda exe=exe".into();
        let error = merge_resume_manifest(existing, &desired)
            .expect_err("resume must not mix verifiers in one bank");
        let rendered = error.to_string();
        assert!(rendered.contains("binary identity"), "{rendered}");
        assert!(rendered.contains("version=0.3.4745"), "{rendered}");
        assert!(rendered.contains("version=0.3.4746"), "{rendered}");

        // Banks written before #binary-identity recorded nothing; they adopt
        // the current identity rather than being locked out of resume forever,
        // exactly as the backend and host fields already do.
        let mut legacy = test_manifest(None);
        legacy.build_id = String::new();
        let merged = merge_resume_manifest(legacy, &desired).expect("legacy bank resumes");
        assert_eq!(
            merged.build_id,
            "NY_BUILD_ID_V1 version=0.3.4746 features=mip,cuda exe=exe"
        );
    }

    #[test]
    fn a_swept_bank_seals_the_identity_of_the_binary_that_measured_it() {
        // The manifest must carry a READABLE identity, not only a digest.
        // `executable.sha256` already content-addressed the file on
        // 2026-08-21 and still could not tell anyone that the cuda binary had
        // become a mip-only wgpu one.
        let identity = crate::build_id::BinaryIdentity::current().expect("identity");
        let line = identity.canonical_line();
        assert!(line.contains(&format!("version={}", identity.stamp.version)));
        assert!(line.contains(&format!("features={}", identity.stamp.features)));
        assert!(line.contains(&format!("exe={}", identity.exe_sha256)));

        // And the cache must not serve a row measured by another verifier.
        let key = |build_id: &str| {
            cache::CacheKey::new(
                "exe".into(),
                build_id.into(),
                "sealed".into(),
                "cfg".into(),
                "acasxu_2023".into(),
                "onnx-closure".into(),
                "vnnlib".into(),
                100,
                None,
                &[],
                "cuda".into(),
                "gb10".into(),
                BTreeMap::new(),
            )
            .expect("key")
            .digest()
            .expect("digest")
        };
        assert_ne!(
            key("NY_BUILD_ID_V1 version=0.3.4745 features=mip"),
            key("NY_BUILD_ID_V1 version=0.3.4745 features=mip,cuda"),
            "two feature tiers must not share a cache entry"
        );
    }

    #[test]
    fn private_executable_copy_stays_bound_to_the_one_early_guard() {
        let source_dir = tempfile::tempdir().expect("source dir");
        let scratch = tempfile::tempdir().expect("private scratch");
        let source = source_dir.path().join("ny-under-test");
        std::fs::write(&source, b"first executable bytes").expect("source executable");
        let guard = crate::build_id::BinaryIdentityGuard::seal(&source).expect("early guard");
        let copied = sealed_executable_copy(&guard, scratch.path()).expect("private copy");

        let manifest_fingerprint =
            fingerprint_from_binary_identity(guard.sealed()).expect("manifest projection");
        assert_eq!(manifest_fingerprint.sha256, guard.sealed().exe_sha256);
        assert_eq!(
            manifest_fingerprint.canonical_path,
            guard.sealed().path.to_str().expect("UTF-8 source")
        );
        assert_ne!(
            Path::new(&manifest_fingerprint.canonical_path),
            copied,
            "the ephemeral child path is not durable manifest provenance"
        );
        assert_eq!(
            fingerprint_file(&copied, "copied executable")
                .expect("copy fingerprint")
                .sha256,
            guard.sealed().exe_sha256
        );

        // Replacing the live build target after setup changes neither the
        // bytes nor the path every child will execute.
        std::fs::write(&source, b"second executable bytes").expect("replace source");
        assert!(guard.reverify("adversarial source replacement").is_err());
        assert_eq!(
            fingerprint_file(&copied, "copied executable after source replacement")
                .expect("copy fingerprint")
                .sha256,
            manifest_fingerprint.sha256
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&copied)
                .expect("copy metadata")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o500, "private child path must not be writable");
        }
        #[cfg(not(unix))]
        assert!(std::fs::metadata(&copied)
            .expect("copy metadata")
            .permissions()
            .readonly());
    }

    #[test]
    fn child_flight_receipt_must_match_the_early_binary_stamp() {
        let expected = crate::build_id::BuildStamp::current().canonical_line();
        let matching = serde_json::json!({"build_id": expected});
        validate_child_build_receipt(Some(&matching), &expected, "matching child")
            .expect("matching receipt");
        validate_child_build_receipt(None, &expected, "hard stop without sidecar")
            .expect("missing best-effort sidecar remains allowed");

        let mismatched = serde_json::json!({
            "build_id": "NY_BUILD_STAMP_V1 version=other stamp=other"
        });
        let error = validate_child_build_receipt(Some(&mismatched), &expected, "fresh measurement")
            .expect_err("a different child binary must not be banked");
        assert!(error
            .to_string()
            .contains("differs from the early executable seal"));

        let malformed = serde_json::json!({"schema_version": 3});
        let error = validate_child_build_receipt(Some(&malformed), &expected, "cached measurement")
            .expect_err("a present sidecar cannot omit its program identity");
        assert!(error.to_string().contains("without a string build_id"));
    }

    #[test]
    fn manifest_refuses_cross_backend_resume_but_adopts_into_legacy_banks() {
        // A bank measured on one compute regime must not silently absorb rows
        // from another: cpu-only vs cuda decides what a timeout MEANS.
        let existing = test_manifest(None);
        let mut desired = test_manifest(None);
        desired.compute_backend = "cpu-only [test]".into();
        let error = merge_resume_manifest(existing, &desired)
            .expect_err("resume must not mix compute regimes");
        assert!(
            error.to_string().contains("compute backend"),
            "unexpected error: {error}"
        );

        // Pre-detection banks recorded nothing; they adopt the detected
        // regime instead of being locked out of resume forever.
        let mut legacy = test_manifest(None);
        legacy.compute_backend = String::new();
        let merged = merge_resume_manifest(legacy, &desired).expect("legacy bank resumes");
        assert_eq!(merged.compute_backend, "cpu-only [test]");
    }

    #[test]
    fn manifest_refuses_cross_host_resume_but_adopts_into_legacy_banks() {
        // Two machines can both detect "metal" and still be incomparable
        // (laptop vs studio): host identity is its own resume boundary.
        let existing = test_manifest(None);
        let mut desired = test_manifest(None);
        desired.host = "other-box | Other CPU | 8 cores | 16 GiB".into();
        let error = merge_resume_manifest(existing, &desired)
            .expect_err("resume must not mix machines in one bank");
        assert!(
            error.to_string().contains("different machines"),
            "unexpected error: {error}"
        );

        let mut legacy = test_manifest(None);
        legacy.host = String::new();
        let merged = merge_resume_manifest(legacy, &desired).expect("legacy bank resumes");
        assert_eq!(merged.host, "other-box | Other CPU | 8 cores | 16 GiB");
    }

    #[test]
    fn manifest_refuses_cap_drift_and_allows_compatible_category_expansion() {
        assert!(merge_resume_manifest(test_manifest(Some(5)), &test_manifest(Some(6))).is_err());

        let existing = test_manifest(Some(5));
        let mut desired = test_manifest(Some(5));
        desired.instances_csv.insert(
            "d".into(),
            ArtifactFingerprint {
                canonical_path: "/corpus/d/instances.csv".into(),
                sha256: "other".into(),
            },
        );
        let d_rows =
            selected_row_set_identity(&[("d".into(), "x.onnx".into(), "q.vnnlib".into(), 0, 0)])
                .expect("category d selected rows");
        desired.selected_rows.insert("d".into(), d_rows.clone());
        desired.selected_row_inputs.insert(
            "d".into(),
            vec![SelectedRowInputSeal {
                onnx: "x.onnx".into(),
                vnnlib: "q.vnnlib".into(),
                occurrence: 0,
                instance_index: 0,
                inputs: ResolvedInstanceInputs {
                    onnx: vec![ModelInputFingerprint {
                        label: None,
                        artifact: ArtifactFingerprint {
                            canonical_path: "/corpus/d/x.onnx".into(),
                            sha256: "x-model".into(),
                        },
                        external_data: Some(Vec::new()),
                    }],
                    vnnlib: ArtifactFingerprint {
                        canonical_path: "/corpus/d/q.vnnlib".into(),
                        sha256: "q-property".into(),
                    },
                },
            }],
        );
        let merged = merge_resume_manifest(existing, &desired).expect("compatible expansion");
        assert_eq!(merged.instances_csv.len(), 2);
        assert_eq!(merged.selected_rows.len(), 2);
        assert_eq!(merged.selected_rows["d"], d_rows);
    }

    #[test]
    fn manifest_refuses_a_changed_selected_row_set_in_an_existing_category() {
        let existing = test_manifest(None);
        let mut desired = test_manifest(None);
        desired.selected_rows.insert(
            "c".into(),
            selected_row_set_identity(&[(
                "c".into(),
                "model.onnx".into(),
                "property.vnnlib".into(),
                1,
                5,
            )])
            .expect("different c row set"),
        );
        let changed = desired
            .selected_row_inputs
            .get_mut("c")
            .and_then(|seals| seals.first_mut())
            .expect("c input seal");
        changed.occurrence = 1;
        changed.instance_index = 5;
        let error = merge_resume_manifest(existing, &desired)
            .expect_err("same category cannot silently reuse a different partial bank");
        assert!(error.to_string().contains("selected row set changed"));
    }

    #[test]
    fn manifest_pins_requested_vnnlib_version() {
        let existing = test_manifest(None);
        let mut desired = test_manifest(None);
        desired.vnnlib_version = Some(VnnlibVersion::V2);

        let error = merge_resume_manifest(existing, &desired)
            .expect_err("resume must not cross VNN-LIB versions");
        assert!(error.to_string().contains("VNN-LIB version"));

        let json = serde_json::to_value(&desired).expect("serialize manifest");
        assert_eq!(json["vnnlib_version"], "2.0");
    }

    #[test]
    fn scoring_deadline_overrides_a_late_decided_result() {
        let (late, detail) =
            enforce_scoring_deadline(SweepVerdict::Unsat, Duration::from_millis(10_001), 10);
        assert_eq!(late, SweepVerdict::Timeout);
        assert!(detail.expect("deadline detail").contains("does not extend"));

        let (on_time, detail) =
            enforce_scoring_deadline(SweepVerdict::Sat, Duration::from_secs(10), 10);
        assert_eq!(on_time, SweepVerdict::Sat);
        assert!(detail.is_none());
    }

    #[test]
    fn metadata_sidecar_preserves_non_official_run_details() {
        let output = Path::new("reports/sweep.csv");
        assert_eq!(
            metadata_path_for(output),
            PathBuf::from("reports/sweep.metadata.jsonl")
        );
        let row = SweepRow {
            // No arm: this fixture is about the metadata sidecar, not lever arming.
            budget_overrun: false,
            arm: Vec::new(),
            category: "c".into(),
            onnx: "m.onnx".into(),
            vnnlib: "p.vnnlib".into(),
            occurrence: 0,
            instance_index: 0,
            verdict: SweepVerdict::Error,
            seconds: 1.25,
            budget_secs: 10,
            capped_from: Some(100),
            detail: Some("child failed".into()),
            flight: Some(serde_json::json!({
                "schema_version": 3,
                "levers": {"status": "not_materialized"},
                "events": [{"method": "result_publish", "status": "ran"}],
            })),
            from_cache: false,
            witness: None,
        };
        let value = serde_json::to_value(&row).expect("serialize metadata");
        assert_eq!(value["budget_secs"], 10);
        assert_eq!(value["capped_from"], 100);
        assert_eq!(value["detail"], "child failed");
        assert_eq!(value["verdict"], "error");
        // The embedded child flight record rides along verbatim: the bank row
        // is the durable home of the trace once the sweep scratch dir is gone.
        assert_eq!(value["flight"]["schema_version"], 3);
        assert_eq!(value["flight"]["levers"]["status"], "not_materialized");
        assert_eq!(value["flight"]["events"][0]["method"], "result_publish");
    }

    /// #sweep-cache gate integrity. A served row carries the flight record of a
    /// DIFFERENT process, so every artifact-level receipt check still passes
    /// against it. `from_cache` is the one field that separates a bank in which
    /// children ran from a bank that was entirely replayed; `ny_search.py`'s
    /// `rows_were_measured` check reads it.
    #[test]
    fn a_cached_row_is_marked_in_the_bank_and_a_measured_one_is_not() {
        let measured = metadata_row("c", 0, SweepVerdict::Unsat);
        let value = serde_json::to_value(&measured).expect("serialize measured row");
        assert!(
            value.get("from_cache").is_none(),
            "a measured row omits the marker, so pre-cache banks stay byte-identical"
        );

        let mut served = metadata_row("c", 0, SweepVerdict::Unsat);
        served.from_cache = true;
        let value = serde_json::to_value(&served).expect("serialize served row");
        assert_eq!(
            value["from_cache"], true,
            "a served row must be visible as such in the bank"
        );

        // Both directions round-trip, and an absent marker reads as measured.
        let legacy = r#"{"category":"c","onnx":"m.onnx","vnnlib":"p.vnnlib","occurrence":0,
            "instance_index":0,"verdict":"unsat","seconds":1.0,"budget_secs":10,
            "capped_from":null,"detail":null}"#;
        let parsed: SweepRow = serde_json::from_str(legacy).expect("legacy bank row parses");
        assert!(!parsed.from_cache);
        let reparsed: SweepRow = serde_json::from_value(value).expect("served row round-trips");
        assert!(reparsed.from_cache);
    }

    #[test]
    fn a_row_without_a_flight_sidecar_omits_the_field_and_still_parses() {
        // Forward: no sidecar -> no key, keeping legacy consumers byte-stable.
        let row = metadata_row("c", 0, SweepVerdict::Unsat);
        let value = serde_json::to_value(&row).expect("serialize metadata");
        assert!(
            value.get("flight").is_none(),
            "a missing sidecar is an omitted field, not a null"
        );
        // Backward: pre-flight banks (no `flight` key) must keep resuming.
        let legacy = r#"{"category":"c","onnx":"m.onnx","vnnlib":"p.vnnlib","occurrence":0,
            "instance_index":0,"verdict":"unsat","seconds":1.0,"budget_secs":10,
            "capped_from":null,"detail":null}"#;
        let parsed: SweepRow = serde_json::from_str(legacy).expect("legacy bank row parses");
        assert!(parsed.flight.is_none());
    }

    #[test]
    fn flight_sidecar_copy_is_best_effort_never_a_row_failure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let result_file = dir.path().join("result.txt");

        // Missing sidecar: the field is simply absent.
        assert!(read_flight_sidecar(&result_file).is_none());

        // Unparsable sidecar: same outcome, no error escapes.
        let sidecar = crate::flight::sidecar_path(&result_file);
        std::fs::write(&sidecar, b"{ not json").expect("write corrupt sidecar");
        assert!(read_flight_sidecar(&result_file).is_none());

        // A valid sidecar is embedded as parsed JSON.
        std::fs::write(
            &sidecar,
            br#"{"schema_version":3,"levers":{"status":"not_materialized"},"events":[]}"#,
        )
        .expect("write valid sidecar");
        let flight = read_flight_sidecar(&result_file).expect("valid sidecar embeds");
        assert_eq!(flight["schema_version"], 3);
        assert_eq!(flight["levers"]["status"], "not_materialized");
    }

    #[test]
    fn witness_file_names_are_deterministic_and_index_disambiguated() {
        assert_eq!(
            witness_file_name(3, "./vnnlib/prop_1.vnnlib"),
            "0003-prop_1.counterexample"
        );
        // acasxu-style categories run one property file against many
        // networks; the pinned row position keeps those witnesses distinct.
        assert_ne!(
            witness_file_name(1, "vnnlib/prop_1.vnnlib"),
            witness_file_name(2, "vnnlib/prop_1.vnnlib")
        );
        // Hostile stems reduce to filesystem-safe names inside the category
        // directory; separators never reach the name.
        assert_eq!(
            witness_file_name(0, "props/a b*?.vnnlib"),
            "0000-a_b__.counterexample"
        );
        assert_eq!(witness_file_name(7, ".."), "0007-instance.counterexample");
    }

    #[test]
    fn witness_extraction_reads_the_block_after_the_sat_token() {
        let dir = tempfile::tempdir().expect("tempdir");
        let result_file = dir.path().join("result.txt");

        std::fs::write(&result_file, "sat\n(X_0 0.5)\n(Y_0 1.0)\n").expect("write sat");
        assert_eq!(
            read_result_witness(&result_file).as_deref(),
            Some("(X_0 0.5)\n(Y_0 1.0)\n")
        );

        // A bare token, a non-sat token, and an unreadable file all yield
        // None: the retention gap is the caller's to record, never a crash.
        std::fs::write(&result_file, "sat\n").expect("write bare sat");
        assert!(read_result_witness(&result_file).is_none());
        std::fs::write(&result_file, "unsat\n(X_0 0.5)\n").expect("write unsat");
        assert!(read_result_witness(&result_file).is_none());
        assert!(read_result_witness(&dir.path().join("missing.txt")).is_none());
    }

    #[test]
    fn banked_sat_witness_lands_in_the_bank_witness_dir_with_its_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = dir.path().join("results.csv");
        let witness = "(X_0 0.5)\n(Y_0 1.0)\n";

        let record = persist_witness(&output, "acasxu_2023", 7, "./vnnlib/prop_1.vnnlib", witness)
            .expect("persist witness");
        assert_eq!(
            record.path,
            "results.witnesses/acasxu_2023/0007-prop_1.counterexample"
        );
        // The recorded path is bank-relative and the recorded hash matches
        // the exact bytes on disk.
        let file = dir.path().join(&record.path);
        assert_eq!(
            std::fs::read_to_string(&file).expect("witness file"),
            witness
        );
        let mut hasher = Sha256::new();
        hasher.update(witness.as_bytes());
        assert_eq!(record.sha256, finish_sha256(hasher));

        // Overwrite-safe: an error-retry of the same row replaces its file
        // at the same deterministic path.
        let replaced = persist_witness(
            &output,
            "acasxu_2023",
            7,
            "./vnnlib/prop_1.vnnlib",
            "(X_0 1.0)\n",
        )
        .expect("replace witness");
        assert_eq!(replaced.path, record.path);
        assert_eq!(
            std::fs::read_to_string(&file).expect("replaced witness"),
            "(X_0 1.0)\n"
        );
    }

    #[test]
    fn witness_retention_failure_is_nonfatal_and_banks_null() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = dir.path().join("results.csv");
        // Absent witness text: the sat row banks a visible gap, not a failure.
        assert_eq!(
            retain_witness(&output, "c", 0, "p.vnnlib", None),
            WitnessDisposition::Missing
        );
        // Blocked witness directory (a FILE occupies its path): same outcome —
        // the row must never fail because its evidence copy did.
        std::fs::write(witness_dir_for(&output), b"in the way").expect("blocker");
        assert_eq!(
            retain_witness(&output, "c", 0, "p.vnnlib", Some("(X_0 0.0)\n")),
            WitnessDisposition::Missing
        );
    }

    #[test]
    fn sat_rows_serialize_witness_records_and_nulls_and_legacy_rows_omit() {
        let record = WitnessRecord {
            path: "results.witnesses/c/0000-property.counterexample".into(),
            sha256: "deadbeef".into(),
        };
        let mut row = metadata_row("c", 0, SweepVerdict::Sat);
        row.witness = Some(WitnessDisposition::Retained(record.clone()));
        let value = serde_json::to_value(&row).expect("serialize retained witness");
        assert_eq!(
            value["witness"]["path"],
            "results.witnesses/c/0000-property.counterexample"
        );
        assert_eq!(value["witness"]["sha256"], "deadbeef");
        let round: SweepRow = serde_json::from_value(value).expect("round trip");
        assert_eq!(round.witness, Some(WitnessDisposition::Retained(record)));

        // A sat row that retained nothing banks an EXPLICIT null...
        row.witness = Some(WitnessDisposition::Missing);
        let value = serde_json::to_value(&row).expect("serialize null witness");
        assert!(value.get("witness").is_some_and(serde_json::Value::is_null));
        // ...and the null round-trips as present-but-missing, not absent.
        let round: SweepRow = serde_json::from_value(value).expect("parse null witness");
        assert_eq!(round.witness, Some(WitnessDisposition::Missing));

        // Non-sat rows and pre-retention banks carry no witness key at all.
        let plain = metadata_row("c", 0, SweepVerdict::Unsat);
        assert!(serde_json::to_value(&plain)
            .expect("serialize plain row")
            .get("witness")
            .is_none());
        let legacy = r#"{"category":"c","onnx":"m.onnx","vnnlib":"p.vnnlib","occurrence":0,
            "instance_index":0,"verdict":"sat","seconds":1.0,"budget_secs":10,
            "capped_from":null,"detail":null}"#;
        let parsed: SweepRow = serde_json::from_str(legacy).expect("legacy bank row parses");
        assert!(parsed.witness.is_none());
    }

    #[test]
    fn summary_counts_sat_rows_without_retained_witnesses() {
        let mut retained = metadata_row("c", 0, SweepVerdict::Sat);
        retained.witness = Some(WitnessDisposition::Retained(WitnessRecord {
            path: "results.witnesses/c/0000-property.counterexample".into(),
            sha256: "deadbeef".into(),
        }));
        let mut gap = metadata_row("c", 1, SweepVerdict::Sat);
        gap.witness = Some(WitnessDisposition::Missing);
        // A legacy sat row (pre-retention bank, no field) is the same gap.
        let legacy_sat = metadata_row("c", 2, SweepVerdict::Sat);
        let unsat = metadata_row("c", 3, SweepVerdict::Unsat);

        let mut summary = SweepSummary::default();
        for row in [&retained, &gap, &legacy_sat, &unsat] {
            summary.record(row);
        }
        assert_eq!(summary.sat, 3);
        assert_eq!(summary.sat_rows_without_witness, 2);
        let json = serde_json::to_value(&summary).expect("summary json");
        assert_eq!(json["sat_rows_without_witness"], 2);
    }

    #[test]
    fn metadata_refuses_witnesses_on_non_sat_rows_and_malformed_records() {
        let path = Path::new("bank.metadata.jsonl");

        let mut row = metadata_row("c", 0, SweepVerdict::Unsat);
        row.witness = Some(WitnessDisposition::Missing);
        assert!(
            validate_metadata_row(&row, path, 1).is_err(),
            "a witness disposition on a non-sat row is corruption"
        );

        let mut row = metadata_row("c", 0, SweepVerdict::Sat);
        for bad in [
            WitnessRecord {
                path: String::new(),
                sha256: "ab".into(),
            },
            WitnessRecord {
                path: "results.witnesses/c/0000-p.counterexample".into(),
                sha256: String::new(),
            },
            WitnessRecord {
                path: "../outside/w.counterexample".into(),
                sha256: "ab".into(),
            },
            WitnessRecord {
                path: "/absolute/w.counterexample".into(),
                sha256: "ab".into(),
            },
        ] {
            row.witness = Some(WitnessDisposition::Retained(bad));
            assert!(validate_metadata_row(&row, path, 1).is_err());
        }

        row.witness = Some(WitnessDisposition::Retained(WitnessRecord {
            path: "results.witnesses/c/0000-p.counterexample".into(),
            sha256: "ab".into(),
        }));
        assert!(validate_metadata_row(&row, path, 1).is_ok());
        // A sat row with a missing witness is legal — a counted gap, not
        // corruption (that is the visible-never-silent contract).
        row.witness = Some(WitnessDisposition::Missing);
        assert!(validate_metadata_row(&row, path, 1).is_ok());
    }

    #[test]
    fn manifest_seals_witness_dir_content_and_resume_ignores_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = dir.path().join("results.csv");
        let record =
            persist_witness(&output, "c", 0, "p.vnnlib", "(X_0 0.5)\n").expect("persist witness");

        let sealed =
            fingerprint_tree(&witness_dir_for(&output), "witness directory").expect("fingerprint");
        // Content-addressed: the same retained bytes fingerprint identically...
        let again = fingerprint_tree(&witness_dir_for(&output), "witness directory")
            .expect("refingerprint");
        assert_eq!(sealed, again);
        // ...and a changed witness changes the seal.
        std::fs::write(dir.path().join(&record.path), "(X_0 1.0)\n").expect("mutate witness");
        let changed = fingerprint_tree(&witness_dir_for(&output), "witness directory")
            .expect("changed fingerprint");
        assert_ne!(sealed.sha256, changed.sha256);

        // The seal is derived output provenance: it must never block
        // --resume, and it rides through a merge untouched until re-sealed.
        let mut existing = test_manifest(None);
        existing.witness_dir = Some(sealed.clone());
        let desired = test_manifest(None);
        let merged = merge_resume_manifest(existing, &desired).expect("resume unaffected");
        assert_eq!(merged.witness_dir, Some(sealed));

        // No witnesses -> no key; pre-retention manifests (no key) parse.
        let json = serde_json::to_value(test_manifest(None)).expect("manifest json");
        assert!(json.get("witness_dir").is_none());
        let legacy: SweepManifest =
            serde_json::from_value(json).expect("pre-retention manifest parses");
        assert!(legacy.witness_dir.is_none());
    }

    #[test]
    fn csv_escape_only_quotes_when_needed_and_doubles_inner_quotes() {
        assert_eq!(csv_escape("plain/path.onnx"), "plain/path.onnx");
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
        assert_eq!(csv_escape("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn summary_counts_only_decided_rows_as_solved() {
        let mut s = SweepSummary::default();
        for v in [
            SweepVerdict::Sat,
            SweepVerdict::Unsat,
            SweepVerdict::Timeout,
            SweepVerdict::Error,
        ] {
            s.record(&SweepRow {
                arm: Vec::new(),
                budget_overrun: false,
                category: "c".into(),
                onnx: "o".into(),
                vnnlib: "v".into(),
                occurrence: 0,
                instance_index: 0,
                verdict: v,
                seconds: 1.0,
                budget_secs: 10,
                capped_from: None,
                detail: None,
                flight: None,
                from_cache: false,
                witness: None,
            });
        }
        assert_eq!(s.rows, 4);
        assert_eq!(s.solved(), 2);
        assert_eq!(s.error, 1);
    }

    /// The sweep must export exactly what the scored path exports.
    ///
    /// Parses `vnncomp_scripts/run_instance.sh` rather than restating its
    /// contents, so adding an `export` there and forgetting the sweep fails here
    /// instead of silently producing measurements of a different verifier.
    /// Category-scoped exports (inside a `case`) are excluded: those belong in
    /// presets, which every entry point already reads.
    #[test]
    fn sweep_environment_matches_the_submission_wrapper() {
        let wrapper = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vnncomp_scripts/run_instance.sh")
            .canonicalize()
            .expect("locate the submission wrapper");
        let body = std::fs::read_to_string(&wrapper).expect("read the submission wrapper");

        // Unconditional exports only: skip anything inside a `case` block, which
        // is how the wrapper scopes a setting to one category.
        let mut in_case = false;
        let mut wrapper_vars = BTreeMap::<String, String>::new();
        for raw in body.lines() {
            let line = raw.trim();
            if line.starts_with("case ") {
                in_case = true;
            } else if line == "esac" {
                in_case = false;
                continue;
            }
            if in_case {
                continue;
            }
            let Some(assignment) = line.strip_prefix("export ") else {
                continue;
            };
            let Some((name, value)) = assignment.split_once('=') else {
                continue;
            };
            // Only literal assignments are parity-relevant; a computed value
            // (WALL_TIMEOUT, NY_BIN) is wrapper plumbing, not verifier config.
            let value = value.trim().trim_matches('"');
            if value.is_empty() || value.contains('$') {
                continue;
            }
            wrapper_vars.insert(name.trim().to_string(), value.to_string());
        }

        assert!(
            !wrapper_vars.is_empty(),
            "parsed no unconditional exports from {} — the parser or the wrapper \
             changed shape, and silently parsing nothing would make this test vacuous",
            wrapper.display()
        );

        let sweep_vars = submission_environment()
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect::<BTreeMap<_, _>>();

        assert_eq!(
            sweep_vars,
            wrapper_vars,
            "`ny benchmarks run` and {} disagree on the scored environment. A sweep \
             that does not reproduce the submission environment measures a different \
             performance regime than the submitted verifier. Update \
             `submission_environment` (or move a verdict-affecting category setting \
             into that category's typed preset).",
            wrapper.display()
        );
    }

    /// The provenance-sealed measurement script must also match the wrapper.
    ///
    /// `sweep_environment_matches_the_submission_wrapper` binds `ny benchmarks
    /// run` to `run_instance.sh`, but `scripts/measure_ny_scorecard.sh` carries
    /// a third, hand-copied list of the same exports — the exact shape that
    /// produced #measure-submission-env-drift. This test parses that script
    /// too: every submission variable must appear in its caller-wins parity
    /// block with the same value, and every OTHER verifier knob it exports
    /// (`NY_*`/`OMP_*`) must be on the explicit allowlist below. A fourth knob
    /// added to the script without a decision about the wrapper fails here
    /// instead of silently measuring a different verifier.
    #[test]
    fn scorecard_script_environment_matches_the_submission_wrapper() {
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/measure_ny_scorecard.sh")
            .canonicalize()
            .expect("locate the measurement script");
        let body = std::fs::read_to_string(&script).expect("read the measurement script");

        // Every `export NAME=VALUE` anywhere on a line, including the
        // caller-wins form `[ "${NAME+x}" = x ] || export NAME=1` and exports
        // inside platform `case` blocks. Category scoping does not exist in
        // this script, so unlike the wrapper parser nothing is skipped.
        let mut script_vars = BTreeMap::<String, Vec<String>>::new();
        for raw in body.lines() {
            let line = raw.trim();
            if line.starts_with('#') {
                continue;
            }
            let Some(position) = line.find("export ") else {
                continue;
            };
            let assignment = &line[position + "export ".len()..];
            let Some((name, value)) = assignment.split_once('=') else {
                continue;
            };
            let name = name.trim();
            if !(name.starts_with("NY_") || name.starts_with("OMP_")) {
                continue;
            }
            let value = value.trim().trim_matches('"').to_string();
            script_vars.entry(name.to_string()).or_default().push(value);
        }

        assert!(
            !script_vars.is_empty(),
            "parsed no NY_*/OMP_* exports from {} — the parser or the script changed \
             shape, and silently parsing nothing would make this test vacuous",
            script.display()
        );

        for (name, value) in submission_environment() {
            let values = script_vars.get(*name).unwrap_or_else(|| {
                panic!(
                    "{} does not export {name}: the provenance-sealed scorecard would \
                     measure a different verifier than the submission wrapper",
                    script.display()
                )
            });
            assert!(
                values.iter().any(|found| found == value),
                "{} exports {name}={values:?}, but the submission wrapper sets \
                 {name}={value}",
                script.display()
            );
        }

        // Knobs the script may export beyond the wrapper, each with a reason.
        // NY_ROOT_GEMM: materializes the compile-time per-platform default into
        //   the provenance manifest — explicitness, not behavior. If the Rust
        //   default ever changes, the script's `case` must change with it.
        // NY_AY: binds an external ay executable for legacy lanes when one is
        //   discovered; the wrapper ships its own binary and never needs it.
        // NY_NO_CUDA: deliberately changes routing, but only in the explicit
        //   NY_ALLOW_NONCUDA_MEASURE=1 CPU-debug lane. Provenance rejects that
        //   lane as CUDA score evidence. Pin the exact gate and assignment so
        //   this exception cannot silently become an unconditional divergence.
        // NY_MEASURE_*: the script's own harness namespace (recursion markers,
        //   binary override, cap). Nothing in crates/ reads it, and the
        //   containment marker is `unset` before any verifier launches. The
        //   cap's measurement-corruption risk is guarded separately, by the
        //   under-measured lint in `ny benchmarks score --budget-year`.
        let noncuda_debug_gate = concat!(
            "if [ \"${NY_ALLOW_NONCUDA_MEASURE:-0}\" = \"1\" ]; then\n",
            "  export NY_NO_CUDA=1\n",
            "fi",
        );
        assert!(
            body.contains(noncuda_debug_gate),
            "{} may export NY_NO_CUDA only as NY_ALLOW_NONCUDA_MEASURE=1's explicit \
             CPU-debug routing decision",
            script.display()
        );
        assert!(
            matches!(
                script_vars.get("NY_NO_CUDA").map(Vec::as_slice),
                Some([value]) if value == "1"
            ),
            "{} must contain exactly one literal NY_NO_CUDA=1 export",
            script.display()
        );

        let allowed_beyond_wrapper = ["NY_ROOT_GEMM", "NY_AY", "NY_NO_CUDA"];
        let submission_names = submission_environment()
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>();
        for name in script_vars.keys() {
            assert!(
                submission_names.contains(&name.as_str())
                    || allowed_beyond_wrapper.contains(&name.as_str())
                    || name.starts_with("NY_MEASURE_"),
                "{} exports {name}, which the submission wrapper does not set and the \
                 allowlist does not explain. Either add it to run_instance.sh AND \
                 `submission_environment`, or allowlist it here with the exact \
                 provenance/debug boundary that makes the divergence intentional.",
                script.display()
            );
        }
    }
}
