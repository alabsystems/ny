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
//!   `vnncomp_scripts/run_instance.sh`, which exports throughput and lane
//!   settings before calling `ny vnncomp`. A sweep that omits them measures a
//!   DIFFERENT verifier than the one being submitted, and the difference shows
//!   up exactly where it hurts: those knobs can only turn a budget-edge timeout
//!   into its already-certified UNSAT, so omitting them makes budget-edge rows
//!   look capability-limited. `submission_environment` below replicates the
//!   wrapper, and `sweep_environment_matches_the_submission_wrapper` parses the
//!   wrapper and fails when the two drift apart.
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
/// being submitted. That is not hypothetical: the `safenlp_2024`
/// `NY_UPFRONT_ATTACK` lane lived only in the wrapper, so every measurement path
/// disagreed with the submission path on rows it could actually solve.
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
/// export. `run_instance.sh`'s `NY_UPFRONT_ATTACK=1` for safenlp_2024 is now a
/// redundant belt-and-braces force of the identical route, not a divergence.
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
    /// produced it. Recording the arm ON THE ROW means two arms of the same
    /// instance can coexist in one bank without ambiguity, and a cache can key on
    /// it. Empty (and omitted) for a default-configuration row, so existing banks
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct SweepManifest {
    schema_version: u32,
    year: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    vnnlib_version: Option<VnnlibVersion>,
    timeout_cap: Option<u64>,
    configs: Option<ArtifactFingerprint>,
    executable: ArtifactFingerprint,
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
    /// Ambient `NY_*` / `OMP_NUM_THREADS` at sweep start (#arm-sealing).
    ///
    /// `run_instance` only ADDS `RUST_LOG` and the submission environment to each
    /// child, so any lever exported into the parent shell is inherited by every
    /// row and was previously INVISIBLE in the artifact. Two consequences, both
    /// of which have produced fake results in this repo: a `--resume` could
    /// silently continue a bank under a different arm than it started with, and
    /// any cache keyed on this manifest would serve rows measured under an
    /// environment the key never saw. Sealing the ambient set makes the arm part
    /// of the bank's identity. Empty on banks written before sealing existed.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    ambient_env: BTreeMap<String, String>,
    /// Content fingerprint of the retained-witness directory, sealed at sweep
    /// end (#witness-retention-gap) — the directory grows during the run, so
    /// the start-of-run manifest cannot carry it. Derived output provenance
    /// only: deliberately NOT part of the `--resume` compatibility comparison.
    /// Absent when no witnesses were banked and on manifests written before
    /// retention existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    witness_dir: Option<ArtifactFingerprint>,
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

fn merge_resume_manifest(
    existing: SweepManifest,
    desired: &SweepManifest,
) -> Result<SweepManifest> {
    if existing.schema_version != SWEEP_MANIFEST_SCHEMA_VERSION {
        bail!(
            "unsupported sweep manifest schema {}; expected {}",
            existing.schema_version,
            SWEEP_MANIFEST_SCHEMA_VERSION
        );
    }
    let same_run = existing.year == desired.year
        && existing.vnnlib_version == desired.vnnlib_version
        && existing.timeout_cap == desired.timeout_cap
        && existing.configs == desired.configs
        && existing.executable == desired.executable
        && existing.build_provenance == desired.build_provenance
        && existing.target_os == desired.target_os
        && existing.target_arch == desired.target_arch;
    if !same_run {
        bail!(
            "cannot safely --resume: year, VNN-LIB version, timeout cap, configuration content, \
             executable, build provenance, or target differs from the pinned manifest; use a \
             new output bank"
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
    // manifest (empty map) adopts the current set rather than blocking resume,
    // exactly as the backend and host fields do; a RECORDED set must match.
    if !existing.ambient_env.is_empty() && existing.ambient_env != desired.ambient_env {
        let render = |env: &BTreeMap<String, String>| {
            env.iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        bail!(
            "cannot safely --resume: this bank was measured with a different ambient lever set              [{}] than the current environment provides [{}]; rows from mixed arms are not              comparable. Unset the difference or use a new output bank.",
            render(&existing.ambient_env),
            render(&desired.ambient_env)
        );
    }
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
    if merged.compute_backend.is_empty() {
        // Legacy bank: adopt the detected regime so rows from here on are
        // attributed, instead of staying unrecorded forever.
        merged.compute_backend = desired.compute_backend.clone();
    }
    if merged.host.is_empty() {
        merged.host = desired.host.clone();
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
}

fn build_plan(
    opts: &SweepOptions,
    categories: Vec<(String, PathBuf)>,
    exe: &Path,
    effective_configs_dir: Option<&Path>,
    selectors: Option<&[InstanceSelector]>,
) -> Result<(Vec<PlannedInstance>, SweepManifest)> {
    let mut plan = Vec::new();
    let mut occurrences = HashMap::new();
    let mut instances_fingerprints = BTreeMap::new();
    let wanted = selectors.map(|selectors| selectors.iter().cloned().collect::<HashSet<_>>());
    let mut matched = HashSet::new();

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
            let selected = match &wanted {
                Some(wanted) => wanted.contains(&base_identity),
                None => index < take,
            };
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
            plan.push(PlannedInstance {
                identity,
                category: category.clone(),
                base: base.clone(),
                instance,
                instance_index: index,
                budget_secs,
                capped_from,
                selected,
            });
        }
    }

    if let Some(selectors) = selectors {
        let names = instances_fingerprints.keys().cloned().collect::<Vec<_>>();
        ensure_every_selector_matched(selectors, &matched, &names)?;
    }

    let manifest = SweepManifest {
        schema_version: SWEEP_MANIFEST_SCHEMA_VERSION,
        year: opts.year,
        vnnlib_version: opts.vnnlib_version,
        timeout_cap: opts.timeout_cap,
        configs: effective_configs_dir
            .map(|dir| fingerprint_tree(dir, "configuration directory"))
            .transpose()?,
        executable: fingerprint_file(exe, "ny executable")?,
        build_provenance: crate::VNNCOMP_BUILD_PROVENANCE.to_string(),
        target_os: std::env::consts::OS.to_string(),
        target_arch: std::env::consts::ARCH.to_string(),
        compute_backend: crate::compute_backend::detect().summary.clone(),
        host: crate::compute_backend::host().summary(),
        ambient_env: crate::flight::ambient_env_from(std::env::vars()),
        witness_dir: None,
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
/// Read from the DESIRED manifest, never from a resumed one. A legacy bank
/// carries an empty `compute_backend`/`host`/`ambient_env` and `--resume`
/// deliberately lets it adopt the current values; keying a cache off the merged
/// manifest would therefore address rows under an environment this process is
/// not running in. The desired manifest describes the run that is about to
/// happen, which is exactly what a verdict is a function of.
#[derive(Debug, Clone)]
struct CacheIdentity {
    exe_sha256: String,
    build_provenance: String,
    configs_sha256: String,
    compute_backend: String,
    host: String,
    ambient_env: BTreeMap<String, String>,
}

impl CacheIdentity {
    fn from_manifest(manifest: &SweepManifest) -> Self {
        Self {
            exe_sha256: manifest.executable.sha256.clone(),
            build_provenance: manifest.build_provenance.clone(),
            // "No configuration tree" is its own identity: the empty string is
            // not a sha256, so it cannot collide with a real digest.
            configs_sha256: manifest
                .configs
                .as_ref()
                .map_or_else(String::new, |configs| configs.sha256.clone()),
            compute_backend: manifest.compute_backend.clone(),
            host: manifest.host.clone(),
            ambient_env: manifest.ambient_env.clone(),
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
        hashes: &mut cache::FileHashMemo,
    ) -> Option<cache::CacheKey> {
        let (onnx, vnnlib) = resolve_instance_arguments(&entry.base, &entry.instance).ok()?;
        cache::CacheKey::new(
            self.exe_sha256.clone(),
            self.build_provenance.clone(),
            self.configs_sha256.clone(),
            // The category is an INPUT, not a label: it names the preset the
            // child loads and gates the category-fenced routes inside it.
            entry.category.clone(),
            &onnx,
            &vnnlib,
            entry.budget_secs,
            entry.capped_from,
            arm,
            self.compute_backend.clone(),
            self.host.clone(),
            self.ambient_env.clone(),
            hashes,
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
) -> Option<(SweepRow, Option<String>)> {
    match verdict_cache.get(key) {
        Ok(measurement) => match serde_json::from_value::<SweepRow>(measurement.row) {
            Ok(row) => Some((row, measurement.witness_text)),
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
    let effective_configs_dir = opts
        .configs_dir
        .clone()
        .or_else(|| {
            std::env::var_os("NY_CONFIGS_DIR")
                .map(PathBuf::from)
                .filter(|path| path.is_dir())
        })
        .or_else(|| {
            let mut starts = vec![exe.clone()];
            if let Ok(cwd) = std::env::current_dir() {
                starts.push(cwd);
            }
            super::vnncomp::auto_derive_configs_dir(&starts)
        });
    let categories = selected_categories(opts.year, &opts.categories)?;
    let (plan, desired_manifest) = build_plan(
        opts,
        categories,
        &exe,
        effective_configs_dir.as_deref(),
        selectors.as_deref(),
    )?;
    let cache_identity = CacheIdentity::from_manifest(&desired_manifest);
    let cache_dir = opts
        .cache_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_VERDICT_CACHE_DIR));
    let verdict_cache = cache::VerdictCache::new(cache_dir.clone(), opts.cache);
    let mut cache_hashes = cache::FileHashMemo::default();
    let scratch = tempfile::Builder::new()
        .prefix("ny-sweep-")
        .tempdir()
        .context("create private sweep workspace")?;

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
                let manifest =
                    merge_resume_manifest(read_manifest(&manifest_path)?, &desired_manifest)?;
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

    let sweep_started = Instant::now();
    let mut cache_hits = 0_usize;
    let mut cache_misses = 0_usize;
    for entry in plan.iter().filter(|entry| entry.selected) {
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
            .then(|| cache_identity.key_for(entry, &opts.arm, &mut cache_hashes))
            .flatten();
        let served = cache_key
            .as_ref()
            .and_then(|key| serve_from_cache(&verdict_cache, key, entry));
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
                    &exe,
                    &entry.category,
                    &entry.base,
                    &entry.instance,
                    entry.budget_secs,
                    effective_configs_dir.as_deref(),
                    scratch.path(),
                    &opts.arm,
                )?;
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
        }];
        let mut state = SweepState::new();
        state.insert(identity, metadata_row("c", 0, SweepVerdict::Unsat));
        assert!(
            validate_state_against_plan(&state, &plan).is_err(),
            "an uncapped persisted row must not enter a capped bank"
        );
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
        sealed.ambient_env = BTreeMap::from([(
            "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN".to_string(),
            "1".to_string(),
        )]);
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
        let legacy = test_manifest(None); // ambient_env empty == legacy
        let mut desired = test_manifest(None);
        desired.ambient_env = BTreeMap::from([("OMP_NUM_THREADS".to_string(), "1".to_string())]);
        merge_resume_manifest(legacy, &desired).expect("a legacy bank must remain resumable");
    }

    /// An identical ambient set resumes cleanly.
    #[test]
    fn resume_allows_an_identical_ambient_lever_set() {
        let arm = BTreeMap::from([("OMP_NUM_THREADS".to_string(), "1".to_string())]);
        let mut sealed = test_manifest(None);
        sealed.ambient_env = arm.clone();
        let mut desired = test_manifest(None);
        desired.ambient_env = arm;
        merge_resume_manifest(sealed, &desired).expect("same arm must resume");
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
            build_provenance: "sealed".into(),
            target_os: "linux".into(),
            target_arch: "x86_64".into(),
            compute_backend: "cuda [test]".into(),
            host: "gb10-test | Test CPU | 20 cores | 128 GiB".into(),
            ambient_env: BTreeMap::new(),
            witness_dir: None,
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
        manifest.ambient_env = BTreeMap::from([("NY_ROOT_GEMM".to_string(), "faer".to_string())]);
        let identity = CacheIdentity::from_manifest(&manifest);
        assert_eq!(identity.ambient_env, manifest.ambient_env);
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
            });
        }

        let identity = CacheIdentity::from_manifest(&test_manifest(None));
        let mut hashes = cache::FileHashMemo::default();
        let digest_of = |entry: &PlannedInstance, hashes: &mut cache::FileHashMemo| {
            identity
                .key_for(entry, &[], hashes)
                .expect("both rows are keyable")
                .digest()
                .expect("digest")
        };
        let a = digest_of(&planned[0], &mut hashes);
        let b = digest_of(&planned[1], &mut hashes);
        assert_ne!(
            a, b,
            "the category selects the preset the child loads; serving across it \
             is a verdict for a run that never happened"
        );
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
        let merged = merge_resume_manifest(existing, &desired).expect("compatible expansion");
        assert_eq!(merged.instances_csv.len(), 2);
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
             verifier than the one being submitted; these knobs can only turn a \
             budget-edge timeout into an already-certified UNSAT, so the divergence \
             shows up as a phantom capability limit. Update `submission_environment` \
             (or move a category-scoped setting into that category's preset).",
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
