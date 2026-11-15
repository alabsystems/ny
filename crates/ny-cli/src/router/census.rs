// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #admission-census — `ny vnncomp census`: the rows the router WOULD route
//! each routable capability on, if it were sealed.
//!
//! The seal pipeline (`scripts/seal/`) must moat every row on which the
//! router routes a capability once it is sealed (ROUTER_PROGRAM §3.5). That
//! set is CR-RS's own decision, so this command computes it with the
//! router's own code, row by row, over a corpus manifest
//! (`# ny-admission-census-corpus/v1`: `row_id<TAB>model<TAB>property<TAB>budget`):
//!
//! 1. **The same facts.** Each row's router input is built exactly as the
//!    scored entry point builds it at router start: the relational probe
//!    ([`RelationalRouteDecision::for_vnnlib`], the SAME type the entry point
//!    asks) decides the relational route (facts `Unknown("relational-route")`,
//!    as there); otherwise the plan resolver's single-pass model scan
//!    ([`ModelFacts::from_onnx_file`], the call
//!    `plan_resolver::resolve_and_materialize_with_runtime` makes) and the
//!    model's on-disk size, into [`super::policy_input`] — the live router's
//!    own glue. A row whose files the entry point would refuse before the
//!    router starts gets no router at all (`no-router:`).
//!    #admission-precision: the property's archetype and routing shape
//!    come from the entry point's own read (`beta_crown::spec_shape::
//!    read_spec`, the lanes' parser and the one extractor the lanes route
//!    on), cached per competition-form property; the `spec` column records
//!    them.
//! 2. **The competition form.** The harness hands the verifier plain files;
//!    a gzip-compressed model or property in the corpus is decompressed into
//!    the work directory under a content-addressed name first, so the facts
//!    are the ones the router reads in competition (on the compressed path the
//!    scan reads nothing and every route is refused).
//! 3. **The same decision.** For every routable capability,
//!    [`route_policy::census_decision`] runs CR-RS's `commit` with that ONE
//!    capability pretended sealed under the census envelope (default
//!    [`CENSUS_WIDEST_ENVELOPE`]) and every other route removed: the
//!    capability's admission alone (precondition, parent, envelope).
//! 4. **The parent's own admission, where the census can ask it.** At router
//!    start a parent lane whose admission is inline is deferred
//!    (`unknown:inline-admission` passes: the parent re-checks at its own
//!    site). The census can load the model and the property, so for a parent
//!    whose structural admission is a callable function — today the margin-row
//!    lane ([`lane_model_admission`] / [`lane_property_admission`], the
//!    lane's own functions) — it asks that function and hands its answer to
//!    the same `commit`. A refusal there means the parent never runs on the
//!    row, so no consumer inside it can engage whatever the router routes.
//!    A probe that fails, times out or cannot read its input keeps the
//!    router's deferred answer (the row stays admitted: never a smaller moat
//!    than the truth) — the read is asked before the lane's own functions,
//!    which answer `None` for an unreadable file exactly as for a refused one
//!    (`census_probe_failure_keeps_the_routers_answer`). The `router` column
//!    is the decision WITHOUT any probe.
//!
//! No verification runs: per row it is one model scan, one property prelude
//! and, where a probe applies, one twin extraction (cached per model) and one
//! robustness parse, each probe under `--row-deadline-secs`. The output is
//! deterministic (manifest order, registry order, no clock) and bound by a
//! header (binary sha256, router and census schemas, registry version, corpus
//! sha256, host, envelope) and a trailing `census_sha256` over every byte
//! above it, which the seal pipeline checks before it trusts a row.
//!
//! NAME-BLIND: the row id is echoed, never read; paths are opened, never
//! parsed; a decompressed file is named by its content. Facts come from the
//! model and the property only (`census_is_name_blind`).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

use ny_core::router::decl::{OpEnvelope, OpKind, OpSet, SpecArchetype, SpecSet};
use ny_core::router::facts::{FactsStatus, HostClass, InstanceFacts};
use ny_core::router::plan::{AdmissionState, PolicyInput};
use ny_core::router::route::{
    self as route_policy, RouteDecision, CENSUS_SCHEMA, CENSUS_WIDEST_ENVELOPE, R5_SCHEMA,
};
use ny_core::router::ROUTER_SCHEMA;

use super::{policy_input, registry, Authority, StartContext};
use crate::commands::beta_crown::spec_shape::{read_spec, SpecRead, SPEC_READ_MAX_BYTES};
use crate::commands::margin_row_bab::{lane_model_admission, lane_property_admission};
use crate::commands::vnncomp::RelationalRouteDecision;
use crate::plan_resolver::ModelFacts;

/// The corpus manifest's first line.
pub(crate) const CORPUS_SCHEMA: &str = "ny-admission-census-corpus/v1";

/// The census columns, in order (#admission-precision: `spec`, the row's
/// archetype and routing shape as the router read them at start).
pub(crate) const COLUMNS: [&str; 10] = [
    "row_id",
    "cap",
    "admit",
    "router",
    "parent_probe",
    "reasons",
    "facts",
    "ops",
    "spec",
    "form",
];

/// #admission-precision: the router's start-time read of the property (the
/// lanes' parser, the one routing-shape extractor, the size ceiling), recorded
/// in the binding: a changed read is a different census.
pub(crate) const SPEC_READ: &str = "vnnlib-shape/v1";

/// The parent lanes whose structural admission the census can ask, with the
/// version of each probe (recorded in the binding: a changed probe is a
/// different census).
pub(crate) const PARENT_PROBES: [(&str, &str); 1] = [("margin-row", "structural/v1")];

/// The plan resolver's scan ceiling (`plan_resolver::SCAN_MAX_BYTES`): a
/// competition-form model above it has no facts, so the census never needs
/// its bytes on disk.
const SCAN_MAX_BYTES: u64 = 256 * 1024 * 1024;

/// A property the census decompresses for a probe: above this the probe is
/// `unknown` (the row keeps the router's deferred answer).
const PROPERTY_MAX_BYTES: u64 = 1 << 30;

/// Probe threads a census may abandon to their deadline before it stops
/// probing (each later probe is `unknown`, which keeps the router's answer).
const MAX_ABANDONED_PROBES: usize = 4;

/// What `ny vnncomp census` was asked to do.
pub(crate) struct CensusOptions {
    /// The corpus manifest.
    pub(crate) manifest: PathBuf,
    /// The census TSV (refused if it exists).
    pub(crate) out: PathBuf,
    /// The pretended seal envelope.
    pub(crate) envelope: OpEnvelope,
    /// A declared host (`None`: the running host, as the scored router).
    pub(crate) target_host: Option<HostClass>,
    /// Each probe's deadline.
    pub(crate) row_deadline: Duration,
    /// Where decompressed competition-form files go.
    pub(crate) work_dir: PathBuf,
}

/// One corpus row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CorpusRow {
    /// Opaque, echoed.
    pub(crate) row_id: String,
    /// The model argument the harness passes.
    pub(crate) model: PathBuf,
    /// The property.
    pub(crate) property: PathBuf,
    /// The scored budget (seconds).
    pub(crate) budget_secs: u64,
}

/// The corpus manifest, parsed (refused unless it carries [`CORPUS_SCHEMA`],
/// four fields per row and unique row ids).
pub(crate) fn parse_corpus(text: &str) -> Result<Vec<CorpusRow>> {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines.next().unwrap_or_default();
    if header.trim() != format!("# {CORPUS_SCHEMA}") {
        bail!("corpus manifest must start with `# {CORPUS_SCHEMA}`, got {header:?}");
    }
    let mut rows = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for line in lines {
        if line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let [row_id, model, property, budget] = fields[..] else {
            bail!("corpus row needs 4 tab-separated fields: {line:?}");
        };
        if row_id.is_empty() || !seen.insert(row_id.to_owned()) {
            bail!("corpus row id {row_id:?} is empty or repeated");
        }
        let secs = budget
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|secs| secs.is_finite() && *secs > 0.0)
            .with_context(|| format!("corpus row {row_id}: budget {budget:?}"))?;
        rows.push(CorpusRow {
            row_id: row_id.to_owned(),
            model: PathBuf::from(model),
            property: PathBuf::from(property),
            budget_secs: (secs.floor() as u64).max(1),
        });
    }
    if rows.is_empty() {
        bail!("corpus manifest has no row");
    }
    Ok(rows)
}

/// Parse `widest` or `OPS/SPECS` (`+`-joined operator and archetype tokens).
pub(crate) fn parse_envelope(text: &str) -> Result<OpEnvelope> {
    if text == "widest" {
        return Ok(CENSUS_WIDEST_ENVELOPE);
    }
    let (ops, specs) = text
        .split_once('/')
        .with_context(|| format!("--envelope must be `widest` or OPS/SPECS, got {text:?}"))?;
    let mut envelope = OpEnvelope::NONE;
    for token in ops.split('+').filter(|t| !t.is_empty()) {
        let kind = OpKind::ALL
            .iter()
            .copied()
            .find(|kind| kind.token() == token)
            .with_context(|| format!("--envelope: unknown operator kind {token:?}"))?;
        envelope.ops = envelope.ops.with(kind);
    }
    for token in specs.split('+').filter(|t| !t.is_empty()) {
        let spec = SPEC_ARCHETYPES
            .iter()
            .copied()
            .find(|spec| spec.token() == token)
            .with_context(|| format!("--envelope: unknown archetype {token:?}"))?;
        envelope.specs = envelope.specs.with(spec);
    }
    Ok(envelope)
}

const SPEC_ARCHETYPES: [SpecArchetype; 5] = [
    SpecArchetype::Robustness,
    SpecArchetype::Halfspace,
    SpecArchetype::Disjunctive,
    SpecArchetype::Relational,
    SpecArchetype::Nonlinear,
];

/// `+`-joined operator tokens of `ops` (`-` when empty).
fn ops_tokens(ops: OpSet) -> String {
    let tokens: Vec<&str> = OpKind::ALL
        .iter()
        .filter(|kind| ops.contains(**kind))
        .map(|kind| kind.token())
        .collect();
    if tokens.is_empty() {
        "-".to_owned()
    } else {
        tokens.join("+")
    }
}

/// `+`-joined archetype tokens of `specs` (`-` when empty).
fn spec_tokens(specs: SpecSet) -> String {
    let tokens: Vec<&str> = SPEC_ARCHETYPES
        .iter()
        .filter(|spec| specs.contains(**spec))
        .map(|spec| spec.token())
        .collect();
    if tokens.is_empty() {
        "-".to_owned()
    } else {
        tokens.join("+")
    }
}

/// The envelope as the binding records it.
pub(crate) fn envelope_token(envelope: OpEnvelope) -> String {
    format!(
        "ops={} specs={}",
        ops_tokens(envelope.ops),
        spec_tokens(envelope.specs)
    )
}

/// A field value with nothing that could break the TSV.
fn cell(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\t' | '\n' | '\r' | ',' => '_',
            other => other,
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    ny_core::router::hash::hex(&digest)
}

fn is_gz(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gz"))
}

/// A competition-form file: the path itself when plain; a gzip file
/// decompressed into the work directory under `<sha256>.<ext>` when its
/// decompressed size is at most `cap`; else only the decompressed size.
/// `NotRead`: the route the row takes never reads the model (the relational
/// route) or the entry point refuses the row first (a missing file).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Form {
    Plain(PathBuf),
    Decompressed(PathBuf),
    TooLarge(u64),
    Unreadable,
    NotRead,
}

impl Form {
    fn path(&self) -> Option<&Path> {
        match self {
            Self::Plain(path) | Self::Decompressed(path) => Some(path),
            Self::TooLarge(_) | Self::Unreadable | Self::NotRead => None,
        }
    }

    const fn token(&self) -> &'static str {
        match self {
            Self::Plain(_) => "plain",
            Self::Decompressed(_) => "gz",
            Self::TooLarge(_) => "gz-large",
            Self::Unreadable => "gz-unreadable",
            Self::NotRead => "unread",
        }
    }
}

fn competition_form(path: &Path, work: &Path, ext: &str, cap: u64) -> Form {
    if !is_gz(path) {
        return Form::Plain(path.to_path_buf());
    }
    let Ok(file) = fs::File::open(path) else {
        return Form::Unreadable;
    };
    let mut decoder = flate2::read::GzDecoder::new(std::io::BufReader::new(file));
    let mut kept: Vec<u8> = Vec::new();
    let mut total: u64 = 0;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = match decoder.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => return Form::Unreadable,
        };
        total = total.saturating_add(n as u64);
        if total <= cap {
            kept.extend_from_slice(&buf[..n]);
        } else if !kept.is_empty() {
            kept = Vec::new();
        }
    }
    if total > cap {
        return Form::TooLarge(total);
    }
    let name = format!("{}.{ext}", sha256_hex(&kept));
    let target = work.join(name);
    if !target.is_file() {
        let staged = work.join(format!(".{}.partial", std::process::id()));
        if fs::write(&staged, &kept).is_err() || fs::rename(&staged, &target).is_err() {
            return Form::Unreadable;
        }
    }
    Form::Decompressed(target)
}

/// One row's router input, exactly as the scored entry point would build it
/// (`None`: the entry point refuses the row before the router starts).
struct RowFacts {
    input: Option<PolicyInput>,
    no_router: Option<&'static str>,
    model_form: Form,
    property_form: Option<Form>,
    /// #admission-precision: the property read the router started with
    /// (`None`: no router, or the relational route, which reads none).
    spec: Option<SpecRead>,
}

/// The parent probe's answer for one row.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Probe {
    Admit,
    Decline(&'static str),
    Unknown(String),
}

impl Probe {
    fn token(&self) -> String {
        match self {
            Self::Admit => "admit".to_owned(),
            Self::Decline(why) => format!("decline:{why}"),
            Self::Unknown(why) => format!("unknown:{}", cell(why)),
        }
    }
}

/// Run `f` on its own thread under `deadline` (`Err` on a timeout, a panic or
/// a failed spawn; a timed-out thread is abandoned, never joined).
fn with_deadline<T: Send + 'static>(
    deadline: Duration,
    f: impl FnOnce() -> T + Send + 'static,
) -> std::result::Result<T, &'static str> {
    let (tx, rx) = mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("ny-census-probe".to_owned())
        .spawn(move || {
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            let _ = tx.send(out);
        });
    if spawned.is_err() {
        return Err("spawn-failed");
    }
    match rx.recv_timeout(deadline) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(_)) => Err("panic"),
        Err(_) => Err("deadline"),
    }
}

/// What differs between the scored entry point's router start sites (the
/// relational route's and the ordinary route's).
struct RowStart<'a> {
    model: Option<&'a ModelFacts>,
    model_bytes: Option<u64>,
    reason: Option<&'static str>,
    relational: bool,
    spec: SpecRead,
}

/// The router's start context for a census row: the scored entry point's
/// fields, with the lever set a row with NO category file freezes in the
/// competition form (declaration defaults, zero `NY_*`; the census plans
/// every rule and arms none, and that set pins nothing), no pins (it
/// describes no row file) and the row's own budget.
///
/// #prepared-root-route: a predicate whose input is the row's frozen policy
/// (`prepared-root`) is asked on that set. A category file that arms a dose
/// such a predicate refuses makes the lane decline on its rows at run time,
/// so the census admits a SUPERSET of the CORPUS rows the lane runs on.
/// review(prepared-root-route): of the corpus rows only — the corpus is the
/// m0 moat corpus (banked sat/unsat and field-SAT rows), and a default-on
/// lane also runs on the scored rows outside it (ROUTER_PROGRAM §10.13).
fn start_context<'a>(
    row: &CorpusRow,
    declared_networks: Option<u64>,
    target_host: Option<HostClass>,
    levers: &'a ny_levers::LeverSet,
    start: RowStart<'a>,
) -> StartContext<'a> {
    StartContext {
        levers: Some(levers),
        authority: Authority::Frozen,
        model: start.model,
        model_bytes: start.model_bytes,
        facts_reason: start.reason,
        declared_networks,
        relational_selected: Some(start.relational),
        official_budget_secs: row.budget_secs,
        internal_secs: row.budget_secs,
        join_margin: crate::commands::vnncomp::CONCURRENT_LANE_JOIN_MARGIN,
        anchor: Instant::now(),
        target_host,
        pinned_keys: Vec::new(),
        trial_pinned_keys: Vec::new(),
        spec: start.spec,
    }
}

/// The census engine (per-model probe cache, abandoned-probe budget).
struct Census {
    deadline: Duration,
    work: PathBuf,
    target_host: Option<HostClass>,
    model_probe: BTreeMap<PathBuf, std::result::Result<usize, Probe>>,
    scans: std::cell::RefCell<BTreeMap<PathBuf, ModelScan>>,
    /// #admission-precision: the router's property read, per
    /// competition-form property (a pure function of the file).
    specs: std::cell::RefCell<BTreeMap<PathBuf, SpecRead>>,
    /// #prepared-root-route: the frozen lever set of a row with no category
    /// file in the competition form ([`start_context`]).
    levers: ny_levers::LeverSet,
    abandoned: usize,
}

/// #prepared-root-route: the lever set a scored row with NO category file
/// freezes in the competition form: every declaration at its default, no
/// `NY_*` (`run_instance.sh` exports none) and no config layer. Never reads
/// the process environment.
pub(crate) fn no_category_levers() -> ny_levers::LeverSet {
    ny_levers::LeverSet::freeze_with(ny_levers::all(), |_| None)
}

/// One model's scan and on-disk size.
type ModelScan = (Option<ModelFacts>, Option<u64>);

impl Census {
    fn new(target_host: Option<HostClass>, deadline: Duration, work: PathBuf) -> Self {
        Self {
            deadline,
            work,
            target_host,
            model_probe: BTreeMap::new(),
            scans: std::cell::RefCell::new(BTreeMap::new()),
            specs: std::cell::RefCell::new(BTreeMap::new()),
            levers: no_category_levers(),
            abandoned: 0,
        }
    }

    /// #admission-precision: the router's start-time read of the property in
    /// its competition form — the entry point's own [`read_spec`] on the
    /// plain file (a gzip property decompressed first; one above the read
    /// ceiling is `unknown` exactly as its plain form would be).
    fn spec_read(&self, form: &Form) -> SpecRead {
        match form {
            Form::Plain(path) | Form::Decompressed(path) => *self
                .specs
                .borrow_mut()
                .entry(path.clone())
                .or_insert_with(|| read_spec(path)),
            Form::TooLarge(bytes) if *bytes > SPEC_READ_MAX_BYTES => {
                SpecRead::Unknown("spec>16MiB")
            }
            Form::TooLarge(_) | Form::Unreadable | Form::NotRead => {
                SpecRead::Unknown("spec-unreadable")
            }
        }
    }

    /// The router input for `row`, built as the scored entry point builds it.
    fn row_facts(&self, row: &CorpusRow) -> RowFacts {
        // The entry point asks the relational probe FIRST (the relational
        // route returns before anything else runs).
        let relational = RelationalRouteDecision::for_vnnlib(&row.property);
        let declared = u64::try_from(relational.declared_networks()).ok();
        let (host, levers) = (self.target_host, &self.levers);
        let at_start =
            |start: RowStart<'_>| policy_input(&start_context(row, declared, host, levers, start));
        if relational.selected() {
            return RowFacts {
                input: Some(at_start(RowStart {
                    model: None,
                    model_bytes: None,
                    reason: Some("relational-route"),
                    relational: true,
                    spec: SpecRead::Unknown("relational-route"),
                })),
                no_router: None,
                model_form: Form::NotRead,
                property_form: None,
                spec: None,
            };
        }
        // The entry point refuses missing files before the router starts.
        if !row.model.is_file() {
            return RowFacts {
                input: None,
                no_router: Some("model-missing"),
                model_form: Form::NotRead,
                property_form: None,
                spec: None,
            };
        }
        if !row.property.is_file() {
            return RowFacts {
                input: None,
                no_router: Some("property-missing"),
                model_form: Form::NotRead,
                property_form: None,
                spec: None,
            };
        }
        // #admission-precision: the entry point reads the property's shape
        // before the router starts (its competition form).
        let property_form =
            competition_form(&row.property, &self.work, "vnnlib", PROPERTY_MAX_BYTES);
        let spec = self.spec_read(&property_form);
        let model_form = competition_form(&row.model, &self.work, "onnx", SCAN_MAX_BYTES);
        let input = match &model_form {
            Form::Plain(path) | Form::Decompressed(path) => {
                // The scored path: `resolve_and_materialize_with_runtime`'s
                // `ModelFacts::from_onnx_file(onnx)` and the entry point's
                // `fs::metadata(&onnx)` — a pure function of the file, so it
                // is computed once per model and reused.
                let (model, model_bytes) = self
                    .scans
                    .borrow_mut()
                    .entry(path.clone())
                    .or_insert_with(|| {
                        (
                            ModelFacts::from_onnx_file(path),
                            fs::metadata(path).ok().map(|meta| meta.len()),
                        )
                    })
                    .clone();
                at_start(RowStart {
                    model: model.as_ref(),
                    model_bytes,
                    reason: None,
                    relational: false,
                    spec,
                })
            }
            // Above the scan ceiling the live router has no facts
            // (`scan>256MiB`), whatever the bytes are.
            Form::TooLarge(bytes) => at_start(RowStart {
                model: None,
                model_bytes: Some(*bytes),
                reason: None,
                relational: false,
                spec,
            }),
            // (`competition_form` never answers `NotRead`.)
            Form::Unreadable | Form::NotRead => at_start(RowStart {
                model: None,
                model_bytes: None,
                reason: Some("model-unreadable"),
                relational: false,
                spec,
            }),
        };
        RowFacts {
            input: Some(input),
            no_router: None,
            model_form,
            property_form: Some(property_form),
            spec: Some(spec),
        }
    }

    /// The structural admission of `parent` on this row, when the census can
    /// ask it (`None`: no probe for that parent).
    fn probe(&mut self, parent: &str, facts: &mut RowFacts, row: &CorpusRow) -> Option<Probe> {
        if !PARENT_PROBES.iter().any(|(id, _)| *id == parent) {
            return None;
        }
        let Some(model) = facts.model_form.path().map(Path::to_path_buf) else {
            return Some(Probe::Unknown("model-not-on-disk".to_owned()));
        };
        if self.abandoned >= MAX_ABANDONED_PROBES {
            return Some(Probe::Unknown("probe-budget-exhausted".to_owned()));
        }
        let n_in = match self.model_probe.get(&model) {
            Some(cached) => cached.clone(),
            None => {
                let path = model.clone();
                // Review fix (#admission-census): the lane's functions read
                // `None` both for a file outside the family AND for a file
                // they could not read, and the second is no refusal the lane
                // would make on a readable copy. The census asks the read
                // FIRST, so an I/O failure is `unknown` (the router's
                // deferred answer stands), never a decline.
                let answer = match with_deadline(self.deadline, move || {
                    if fs::read(&path).is_err() {
                        return Err(None);
                    }
                    lane_model_admission(&path).map_err(Some)
                }) {
                    Ok(Ok(n_in)) => Ok(n_in),
                    Ok(Err(Some(why))) => Err(Probe::Decline(why)),
                    Ok(Err(None)) => Err(Probe::Unknown("model-unreadable".to_owned())),
                    Err(fault) => {
                        self.abandoned += usize::from(fault == "deadline");
                        Err(Probe::Unknown(fault.to_owned()))
                    }
                };
                self.model_probe.insert(model, answer.clone());
                answer
            }
        };
        let n_in = match n_in {
            Ok(n_in) => n_in,
            Err(probe) => return Some(probe),
        };
        let property_form = facts.property_form.get_or_insert_with(|| {
            competition_form(&row.property, &self.work, "vnnlib", PROPERTY_MAX_BYTES)
        });
        let Some(property) = property_form.path().map(Path::to_path_buf) else {
            return Some(Probe::Unknown(format!(
                "property-{}",
                property_form.token()
            )));
        };
        // Review fix: as for the model, an unreadable property is `unknown`
        // (`parse_vnnlib_robustness` reads `None` for it, which is not the
        // lane's refusal of a non-robustness property). A readable file that
        // is not UTF-8 still reaches the lane's own parse, which refuses it
        // exactly as the lane does.
        match with_deadline(self.deadline, move || {
            if fs::read(&property).is_err() {
                return Err(None);
            }
            lane_property_admission(&property, n_in).map_err(Some)
        }) {
            Ok(Ok(())) => Some(Probe::Admit),
            Ok(Err(Some(why))) => Some(Probe::Decline(why)),
            Ok(Err(None)) => Some(Probe::Unknown("property-unreadable".to_owned())),
            Err(fault) => {
                self.abandoned += usize::from(fault == "deadline");
                Some(Probe::Unknown(fault.to_owned()))
            }
        }
    }
}

/// The census row for one capability.
struct CapRow {
    admit: bool,
    router: bool,
    probe: String,
    reasons: String,
}

fn decision_reasons(decision: &RouteDecision) -> String {
    if decision.reasons.is_empty() {
        "-".to_owned()
    } else {
        decision
            .reasons
            .iter()
            .map(|r| cell(r))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// The parent id of `cap` in `input` when its admission there is deferred
/// inline (the only case a probe can refine).
fn inline_parent(input: &PolicyInput, cap: &str) -> Option<String> {
    let parent = input
        .capabilities
        .iter()
        .find(|c| c.id == cap)?
        .route
        .as_ref()?
        .parent
        .clone()?;
    let state = &input
        .capabilities
        .iter()
        .find(|c| c.id == parent)?
        .admission;
    matches!(state, AdmissionState::Unknown(why) if why == ny_core::router::decl::UNKNOWN_INLINE)
        .then_some(parent)
}

/// `input` with `parent`'s admission replaced by the probe's answer (an
/// `unknown` probe keeps the router's deferred answer).
fn with_probed_parent(input: &PolicyInput, parent: &str, probe: &Probe) -> PolicyInput {
    let mut probed = input.clone();
    if let Some(cap) = probed.capabilities.iter_mut().find(|c| c.id == parent) {
        match probe {
            Probe::Admit => cap.admission = AdmissionState::Admit,
            Probe::Decline(why) => cap.admission = AdmissionState::Decline((*why).to_owned()),
            Probe::Unknown(_) => {}
        }
    }
    probed
}

fn facts_token(facts: &InstanceFacts) -> String {
    match &facts.status {
        FactsStatus::Known if facts.ops_complete => "known/complete".to_owned(),
        FactsStatus::Known => "known/lower-bound".to_owned(),
        FactsStatus::Unknown(why) => format!("unknown:{}", cell(why)),
    }
}

/// The census of one row: one [`CapRow`] per routable capability (registry
/// order), plus the row's facts, operator and form tokens.
fn census_row(
    census: &mut Census,
    row: &CorpusRow,
    caps: &[String],
    envelope: OpEnvelope,
) -> (Vec<CapRow>, String, String, String, String) {
    let mut facts = census.row_facts(row);
    let Some(input) = facts.input.clone() else {
        let why = facts.no_router.unwrap_or("no-router");
        let rows = caps
            .iter()
            .map(|_| CapRow {
                admit: false,
                router: false,
                probe: "-".to_owned(),
                reasons: format!("no-router:{why}"),
            })
            .collect();
        return (
            rows,
            format!("no-router:{why}"),
            "-".to_owned(),
            "-".to_owned(),
            facts.model_form.token().to_owned(),
        );
    };
    let mut probes: BTreeMap<String, Probe> = BTreeMap::new();
    let mut out = Vec::with_capacity(caps.len());
    for cap in caps {
        let router = route_policy::census_decision(&input, cap, envelope);
        let router_admits = router.as_ref().is_some_and(|d| d.routed);
        // A probe can only ADD a refusal (an admitting probe is what the
        // router's deferred answer already assumed), so it runs only where the
        // router alone would route: the probe's cost (one model load) is paid
        // on admitted rows only.
        let parent = inline_parent(&input, cap).filter(|_| router_admits);
        let (census_decision, probe_token) = match parent {
            Some(parent) => {
                let probe = match probes.get(&parent) {
                    Some(probe) => Some(probe.clone()),
                    None => {
                        let answer = census.probe(&parent, &mut facts, row);
                        if let Some(answer) = &answer {
                            probes.insert(parent.clone(), answer.clone());
                        }
                        answer
                    }
                };
                match probe {
                    Some(probe) => (
                        route_policy::census_decision(
                            &with_probed_parent(&input, &parent, &probe),
                            cap,
                            envelope,
                        ),
                        probe.token(),
                    ),
                    None => (router.clone(), "-".to_owned()),
                }
            }
            None if router_admits || inline_parent(&input, cap).is_none() => {
                (router.clone(), "-".to_owned())
            }
            None => (router.clone(), "skipped".to_owned()),
        };
        let (admit, reasons) = census_decision.as_ref().map_or_else(
            || (false, "unregistered".to_owned()),
            |d| (d.routed, decision_reasons(d)),
        );
        out.push(CapRow {
            admit,
            router: router.as_ref().is_some_and(|d| d.routed),
            probe: probe_token,
            reasons,
        });
    }
    let form = match &facts.property_form {
        Some(Form::Decompressed(_)) => format!("{}+property-gz", facts.model_form.token()),
        _ => facts.model_form.token().to_owned(),
    };
    let spec = facts
        .spec
        .as_ref()
        .map_or_else(|| "-".to_owned(), |s| cell(&s.token()));
    (
        out,
        facts_token(&input.facts),
        if input.facts.is_known() {
            ops_tokens(input.facts.ops)
        } else {
            "-".to_owned()
        },
        spec,
        form,
    )
}

/// What a census run reports.
pub(crate) struct CensusSummary {
    /// Manifest rows.
    pub(crate) rows: usize,
    /// Routable capabilities.
    pub(crate) caps: Vec<String>,
    /// Admitted rows per capability (census column), registry order.
    pub(crate) admitted: Vec<(String, usize)>,
    /// Admitted rows per capability without any probe (router column).
    pub(crate) router_admitted: Vec<(String, usize)>,
    /// The trailing sha256.
    pub(crate) census_sha256: String,
}

/// The binding header lines (`# key<TAB>value`), in order.
pub(crate) fn binding_lines(
    binary_sha256: &str,
    build_commit: &str,
    corpus_sha256: &str,
    corpus_rows: usize,
    host: &str,
    envelope: OpEnvelope,
    caps: &[String],
) -> Vec<String> {
    let probes: Vec<String> = PARENT_PROBES
        .iter()
        .map(|(id, version)| format!("{id}={version}"))
        .collect();
    vec![
        format!("# {CENSUS_SCHEMA}"),
        format!("# binary_sha256\t{binary_sha256}"),
        format!("# build\t{build_commit}"),
        format!("# router_schema\t{ROUTER_SCHEMA}"),
        format!("# r5_schema\t{R5_SCHEMA}"),
        format!("# registry_version\t{}", registry::registry_version()),
        format!("# corpus_schema\t{CORPUS_SCHEMA}"),
        format!("# corpus_sha256\t{corpus_sha256}"),
        format!("# corpus_rows\t{corpus_rows}"),
        format!("# host\t{host}"),
        format!("# envelope\t{}", envelope_token(envelope)),
        format!("# probes\t{}", probes.join(",")),
        format!("# spec_read\t{SPEC_READ} max_bytes={SPEC_READ_MAX_BYTES}"),
        format!("# caps\t{}", caps.join(",")),
        format!("# columns\t{}", COLUMNS.join("\t")),
    ]
}

/// Compute the census of `rows` (pure but for the files it reads and the
/// work directory it may write): the full TSV text, trailer included.
pub(crate) fn census_text(
    rows: &[CorpusRow],
    corpus_sha256: &str,
    binary_sha256: &str,
    build_commit: &str,
    options: &CensusOptions,
) -> (String, CensusSummary) {
    let mut census = Census::new(
        options.target_host,
        options.row_deadline,
        options.work_dir.clone(),
    );
    // The routable capabilities in registry order (the order `policy_input`
    // keeps them in).
    let mut caps: Vec<String> = Vec::new();
    let mut body = String::new();
    let mut admitted: BTreeMap<String, usize> = BTreeMap::new();
    let mut router_admitted: BTreeMap<String, usize> = BTreeMap::new();
    for decl in registry::REGISTRY {
        if decl.routable.is_some() {
            caps.push(decl.id.as_str().to_owned());
        }
    }
    for row in rows {
        let (cap_rows, facts, ops, spec, form) =
            census_row(&mut census, row, &caps, options.envelope);
        for (cap, cap_row) in caps.iter().zip(&cap_rows) {
            *admitted.entry(cap.clone()).or_default() += usize::from(cap_row.admit);
            *router_admitted.entry(cap.clone()).or_default() += usize::from(cap_row.router);
            let _ = writeln!(
                body,
                "{}\t{cap}\t{}\t{}\t{}\t{}\t{facts}\t{ops}\t{spec}\t{form}",
                cell(&row.row_id),
                u8::from(cap_row.admit),
                u8::from(cap_row.router),
                cap_row.probe,
                cap_row.reasons,
            );
        }
    }
    // The host the census planned against: the router's own envelope
    // function (`policy_input` calls the same one per row).
    let h = super::facts::host_envelope(options.target_host);
    let host = format!(
        "class={} threads={} device={} declared={}",
        h.class.token(),
        h.threads,
        h.device.token(),
        h.declared
    );
    let mut text = binding_lines(
        binary_sha256,
        build_commit,
        corpus_sha256,
        rows.len(),
        &host,
        options.envelope,
        &caps,
    )
    .join("\n");
    text.push('\n');
    text.push_str(&body);
    let census_sha256 = sha256_hex(text.as_bytes());
    let _ = writeln!(text, "# census_sha256\t{census_sha256}");
    let ordered = |m: &BTreeMap<String, usize>| {
        caps.iter()
            .map(|c| (c.clone(), m.get(c).copied().unwrap_or(0)))
            .collect()
    };
    let summary = CensusSummary {
        rows: rows.len(),
        caps: caps.clone(),
        admitted: ordered(&admitted),
        router_admitted: ordered(&router_admitted),
        census_sha256,
    };
    (text, summary)
}

/// Verify a census text's trailer: `Ok(census_sha256)` when the last line is
/// `# census_sha256<TAB><sha256 of every byte above it>`.
pub(crate) fn verify_census_text(text: &str) -> Result<String> {
    let body = text.strip_suffix('\n').unwrap_or(text);
    let (above, last) = body
        .rsplit_once('\n')
        .context("census has no trailer line")?;
    let want = last
        .strip_prefix("# census_sha256\t")
        .context("census trailer must be `# census_sha256<TAB><hex>`")?;
    let mut covered = above.to_owned();
    covered.push('\n');
    let got = sha256_hex(covered.as_bytes());
    if got != want {
        bail!("census_sha256 mismatch: trailer {want}, content {got} (the census was edited)");
    }
    if !above.starts_with(&format!("# {CENSUS_SCHEMA}\n")) {
        bail!("census does not start with `# {CENSUS_SCHEMA}`");
    }
    Ok(got)
}

/// `ny vnncomp census`.
pub(crate) fn handle_census_command(options: &CensusOptions) -> Result<CensusSummary> {
    let started = Instant::now();
    if options.out.exists() {
        bail!(
            "refusing to overwrite {}: a census is written once",
            options.out.display()
        );
    }
    let manifest_bytes = fs::read(&options.manifest)
        .with_context(|| format!("reading corpus manifest {}", options.manifest.display()))?;
    let corpus_sha256 = sha256_hex(&manifest_bytes);
    let rows = parse_corpus(
        std::str::from_utf8(&manifest_bytes).context("corpus manifest is not UTF-8")?,
    )?;
    fs::create_dir_all(&options.work_dir)
        .with_context(|| format!("creating work dir {}", options.work_dir.display()))?;
    let identity = crate::build_id::BinaryIdentity::current()
        .context("hashing the running binary for the census binding")?;
    let build = format!(
        "commit={} dirty={} features={}",
        identity.stamp.commit, identity.stamp.dirty, identity.stamp.features
    );
    let (text, summary) = census_text(&rows, &corpus_sha256, &identity.exe_sha256, &build, options);
    let staged = options.out.with_extension("tsv.partial");
    {
        let mut file =
            fs::File::create(&staged).with_context(|| format!("creating {}", staged.display()))?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
    }
    fs::rename(&staged, &options.out)
        .with_context(|| format!("publishing {}", options.out.display()))?;
    // Read the published file back: what the seal pipeline will verify.
    let written = verify_census_text(&fs::read_to_string(&options.out)?)?;
    if written != summary.census_sha256 {
        bail!("{} does not read back as written", options.out.display());
    }
    eprintln!(
        "[census] rows={} caps={} elapsed_s={:.1} work_dir={}",
        summary.rows,
        summary.caps.len(),
        started.elapsed().as_secs_f64(),
        options.work_dir.display()
    );
    Ok(summary)
}

/// The DONE line `ny vnncomp census` prints (one `rc=`).
pub(crate) fn done_line(summary: &CensusSummary, out: &Path) -> String {
    let fmt = |v: &[(String, usize)]| {
        v.iter()
            .map(|(cap, n)| format!("{cap}:{n}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        "CENSUS_DONE rc=0 rows={} caps={} admitted={} router_admitted={} census_sha256={} out={}",
        summary.rows,
        summary.caps.len(),
        fmt(&summary.admitted),
        fmt(&summary.router_admitted),
        summary.census_sha256,
        out.display()
    )
}
