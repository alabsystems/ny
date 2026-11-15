// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #graph-free-candidates (#smart-turn-x86-harvest): a DIAGNOSTIC-ONLY side
//! file of the counterexample candidates the graph-free falsification lane
//! (`try_graph_free_ort_falsify`) puts in front of its acceptance gates.
//!
//! # Why
//!
//! `smart_turn_multimodal_2026` is decided by that lane, because ny cannot
//! load its QDQ graph. MEASURED 2026-09-26 (`scratchpad/x86replay`): the
//! organisers grade witnesses with ONNX Runtime 1.26.0 on x86-64 AVX2+FMA,
//! and on this quantized model arm64 ONNX Runtime is NOT that checker — qemu
//! `Haswell-v4` reproduces the organiser's Y on 100/100 graded field
//! witnesses, native aarch64 on 11/100. So:
//!
//! * a point ny's arm64 forward calls violating may read Y = 0.5 on the
//!   checker (the dual-level and corroboration gates withhold such points,
//!   correctly: a withheld candidate is a forgone +10, a published one a
//!   possible -150);
//! * a point ny's arm64 forward calls SAFE may violate on the checker (71 of
//!   the 100 organiser-accepted field witnesses read exactly Y = 0.5 on
//!   arm64).
//!
//! Neither question can be answered on the host that runs ny. This file lets
//! both be answered OFFLINE: every candidate is written with the exact f32
//! tensor that ran, so `scratchpad/x86replay/harvest` can replay it on the
//! certified x86 configuration.
//!
//! # What is recorded (schema `ny.graph_free_candidates.v1`, one JSON object per line)
//!
//! * `header` — once per lane entry: model/spec paths, input width, the
//!   PINNED dims (index, declared f64) the witness would emit verbatim, host
//!   arch, encoding;
//! * `candidate` — every point handed to the lane's `try_accept`, with its
//!   outcome: `accepted_by_lane`, or which gate rejected it
//!   (`rejected_emit`, `rejected_dual_level`, `rejected_dual_unavailable`,
//!   `rejected_extent`, `rejected_corroboration`, `rejected_pin`,
//!   `rejected_forward`, `rejected_deadline`) plus the gate's own numbers;
//! * `near_miss` — a BOUNDED sample of the flat-objective surrogate's
//!   incumbents that the lane re-scored on the ORIGINAL model and found
//!   non-violating here (power-of-two improvement indices, at most
//!   [`MAX_NEAR_MISS_RECORDS`], plus the final incumbent);
//! * `summary` — the counters, once, when the lane exits;
//! * `trusted_oracle` — appended by the caller for the lane's accepted
//!   candidate: whether `gate_sat_with_trusted_oracle` upheld it.
//!
//! Every point carries `x`: the f32 input tensor as little-endian bytes,
//! base64 (RFC 4648, padded), with its sha256. The organiser-format witness
//! of a point is the f64 view `x64[i] = pinned[i]` for a pinned dim and the
//! exact widening of `x[i]` otherwise — the value `refine_emit_view` emits,
//! whose f32 cast is exactly the tensor that ran.
//!
//! # Soundness
//!
//! REPORT-ONLY. The recorder reads values the gates already computed; it
//! changes no gate, threshold, radius, RNG draw or probe, and ny never reads
//! the file back. Nothing written here is published, and nothing here can
//! produce `sat` or `unsat`: the lane's acceptance stack, and the caller's
//! trusted-oracle gate after it, are exactly what they were. The harvest
//! tool that consumes the file emits a witness only after replaying it on
//! the certified x86 configuration.
//!
//! # Dark by default
//!
//! `attack.graph_free_candidate_sidefile` is absent from every shipped yaml
//! (pinned by `no_shipped_preset_arms_a_promoted_lever_delivery_key`).
//! Unarmed, [`CandidateSidefile::from_gates`] returns `None`, every hook in
//! the lane is an `if let Some(..)` on that `None`, and the lane is
//! byte-identical — probe sequence and stderr included.

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};
use std::time::Instant;

use ny_core::f32_to_f64_exact;
use ny_onnx::vnnlib::VnnLibSpec;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::property_margin;

/// Line schema of the side file.
pub(super) const SCHEMA: &str = "ny.graph_free_candidates.v1";
/// The typed preset key that arms the side file.
pub(super) const PRESET_KEY: &str = "attack.graph_free_candidate_sidefile";
/// Encoding tag of every `x` payload.
pub(super) const X_ENCODING: &str = "f32le-base64";
/// Suffix appended (not substituted) to RESULTS_FILE.
pub(super) const SIDEFILE_SUFFIX: &str = ".graph_free_candidates.jsonl";
/// At most this many `candidate` records per row; later ones are counted in
/// `dropped_cap` (their outcomes are still counted).
pub(super) const MAX_CANDIDATE_RECORDS: u64 = 256;
/// At most this many non-final `near_miss` records per row (the final
/// incumbent may add one more).
pub(super) const MAX_NEAR_MISS_RECORDS: u64 = 16;
/// Byte budget of one row's side file.
pub(super) const MAX_SIDEFILE_BYTES: u64 = 2 << 30;

/// The scored entry's RESULTS_FILE, recorded once at entry. A plain path
/// store: nothing reads it unless the side file is armed.
static RESULTS_TARGET: RwLock<Option<PathBuf>> = RwLock::new(None);

/// The lane's accepted candidate, parked until the caller's trusted-oracle
/// gate has ruled on it. Written only by an armed, gate-constructed recorder.
static PENDING_ORACLE: Mutex<Option<PendingOracle>> = Mutex::new(None);

#[derive(Debug, Clone)]
struct PendingOracle {
    path: PathBuf,
    candidate_id: u64,
    candidate_recorded: bool,
}

/// Record the scored entry's RESULTS_FILE (called unconditionally at entry;
/// storing a path has no observable effect).
pub(super) fn set_results_target(results_file: &Path) {
    if let Ok(mut slot) = RESULTS_TARGET.write() {
        *slot = Some(results_file.to_path_buf());
    }
}

fn results_target() -> Option<PathBuf> {
    RESULTS_TARGET.read().ok().and_then(|slot| slot.clone())
}

/// `<RESULTS_FILE>.graph_free_candidates.jsonl` — appended, so `a.txt` and
/// `a.csv` can never share a side file.
pub(super) fn sidefile_path(results_file: &Path) -> PathBuf {
    let mut os = results_file.as_os_str().to_os_string();
    os.push(SIDEFILE_SUFFIX);
    PathBuf::from(os)
}

/// Receipt projection for the per-run frozen lever set: `Some(value)` iff
/// `decl` is the side-file lever (the key has no search axis — it is an
/// INSTRUMENT — so the generic axis projector cannot see it). Without this,
/// an armed run's flight receipt would record the declaration default.
#[expect(
    clippy::option_option,
    reason = "the lever projector protocol of `materialize_levers` (vnncomp.rs): outer `None` \
              = not the side-file lever, `Some(None)` = the side-file lever unset (the \
              declaration default), `Some(Some(v))` = the typed key's value"
)]
pub(super) fn preset_config_value(
    loaded: &crate::preset::PresetConfig,
    decl: &'static ny_levers::LeverDecl,
) -> Option<Option<ny_levers::LeverValue>> {
    std::ptr::eq(
        decl,
        &raw const ny_levers::decls::dark_probes::GRAPH_FREE_CANDIDATE_SIDEFILE,
    )
    .then(|| {
        loaded
            .attack
            .graph_free_candidate_sidefile
            .map(ny_levers::LeverValue::Bool)
    })
}

/// Why a candidate left the lane's acceptance stack. Mirrors, one-for-one,
/// the exits of `try_accept` in `try_graph_free_ort_falsify_observed`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Outcome {
    /// Passed every gate of the lane; handed to the caller's trusted-oracle
    /// gate (whose ruling is appended as a `trusted_oracle` record).
    AcceptedByLane { corroborated: usize },
    /// The zero-tolerance `property_violated_f64` re-check of the emit view
    /// (box membership included) refused it.
    RejectedEmit,
    /// The Disable-level session could not be built; the gate failed closed.
    RejectedDualUnavailable,
    /// Violates at the default level but not with optimizations disabled.
    RejectedDualLevel,
    /// The instance deadline expired inside the named gate phase.
    RejectedDeadline { phase: &'static str },
    /// A forward pass failed inside the named gate phase.
    RejectedForward { phase: &'static str },
    /// The violation ends within 1e-4 of the span along free dim `dim`.
    RejectedExtent { dim: usize },
    /// A resampled neighbour at `radius` of the corroboration span read
    /// `neighbour_margin <= 0` after `checked` forwards.
    RejectedCorroboration {
        radius: f64,
        neighbour_margin: f64,
        checked: usize,
    },
    /// A corroboration neighbour could not be pinned into the declared box.
    RejectedPin,
}

impl Outcome {
    /// Stable token written to the side file and counted in the receipt.
    pub(super) const fn token(self) -> &'static str {
        match self {
            Self::AcceptedByLane { .. } => "accepted_by_lane",
            Self::RejectedEmit => "rejected_emit",
            Self::RejectedDualUnavailable => "rejected_dual_unavailable",
            Self::RejectedDualLevel => "rejected_dual_level",
            Self::RejectedDeadline { .. } => "rejected_deadline",
            Self::RejectedForward { .. } => "rejected_forward",
            Self::RejectedExtent { .. } => "rejected_extent",
            Self::RejectedCorroboration { .. } => "rejected_corroboration",
            Self::RejectedPin => "rejected_pin",
        }
    }
}

/// Per-outcome counters (every candidate is counted, recorded or not).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct OutcomeCounts {
    pub(super) seen: u64,
    pub(super) accepted: u64,
    pub(super) rejected_emit: u64,
    pub(super) rejected_dual_unavailable: u64,
    pub(super) rejected_dual_level: u64,
    pub(super) rejected_deadline: u64,
    pub(super) rejected_forward: u64,
    pub(super) rejected_extent: u64,
    pub(super) rejected_corroboration: u64,
    pub(super) rejected_pin: u64,
}

impl OutcomeCounts {
    fn bump(&mut self, outcome: Outcome) {
        self.seen += 1;
        let slot = match outcome {
            Outcome::AcceptedByLane { .. } => &mut self.accepted,
            Outcome::RejectedEmit => &mut self.rejected_emit,
            Outcome::RejectedDualUnavailable => &mut self.rejected_dual_unavailable,
            Outcome::RejectedDualLevel => &mut self.rejected_dual_level,
            Outcome::RejectedDeadline { .. } => &mut self.rejected_deadline,
            Outcome::RejectedForward { .. } => &mut self.rejected_forward,
            Outcome::RejectedExtent { .. } => &mut self.rejected_extent,
            Outcome::RejectedCorroboration { .. } => &mut self.rejected_corroboration,
            Outcome::RejectedPin => &mut self.rejected_pin,
        };
        *slot += 1;
    }
}

/// The latest surrogate incumbent not yet written (kept for the final record).
struct PendingIncumbent {
    x: Vec<f32>,
    out: Vec<f32>,
    real_margin: f64,
    surrogate_margin: f64,
    improvement: u64,
}

/// The armed recorder. Constructed only by [`CandidateSidefile::from_gates`]
/// (armed scored run) or [`CandidateSidefile::for_path`] (tests); its `Drop`
/// writes the summary and prints the fixed-schema receipt on EVERY exit of
/// the lane that owns it.
pub(super) struct CandidateSidefile {
    path: Option<PathBuf>,
    source: &'static str,
    park_oracle: bool,
    writer: Option<BufWriter<File>>,
    status: &'static str,
    started: Instant,
    stage: &'static str,
    next_id: u64,
    candidate_records: u64,
    near_miss_records: u64,
    bytes: u64,
    write_errors: u64,
    dropped_cap: u64,
    counts: OutcomeCounts,
    accepted_id: Option<u64>,
    accepted_recorded: bool,
    improvements: u64,
    pending: Option<PendingIncumbent>,
}

impl CandidateSidefile {
    fn new(path: Option<PathBuf>, source: &'static str) -> Self {
        Self {
            path,
            source,
            park_oracle: false,
            writer: None,
            status: "armed_not_begun",
            started: Instant::now(),
            stage: "setup",
            next_id: 0,
            candidate_records: 0,
            near_miss_records: 0,
            bytes: 0,
            write_errors: 0,
            dropped_cap: 0,
            counts: OutcomeCounts::default(),
            accepted_id: None,
            accepted_recorded: false,
            improvements: 0,
            pending: None,
        }
    }

    /// The scored route: `None` unless the typed key (or its env override)
    /// arms the side file. This is the lever's single lane-side read.
    pub(super) fn from_gates() -> Option<Self> {
        if !crate::preset_gates::graph_free_candidate_sidefile_armed() {
            return None;
        }
        let source = crate::preset_gates::graph_free_candidate_sidefile_source();
        let path = results_target().map(|results| sidefile_path(&results));
        let mut sidefile = Self::new(path, source);
        sidefile.park_oracle = true;
        Some(sidefile)
    }

    /// An explicit recorder writing to `path` (tests). It never parks a
    /// trusted-oracle record, so it cannot leak into another run's funnel.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn for_path(path: PathBuf) -> Self {
        Self::new(Some(path), "explicit")
    }

    /// Counters so far (tests and the receipt).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn counts(&self) -> OutcomeCounts {
        self.counts
    }

    /// Create the file and write the header. Called once, after the lane has
    /// its spec, width and pinned dims; a lane that declines earlier leaves
    /// status `armed_not_begun` in its receipt.
    pub(super) fn begin(&mut self, onnx: &Path, vnnlib: &Path, n: usize, emit_pin: &[Option<f64>]) {
        let Some(path) = self.path.clone() else {
            self.status = "no_target";
            return;
        };
        match File::create(&path) {
            Ok(file) => {
                self.writer = Some(BufWriter::with_capacity(1 << 20, file));
                self.status = "recording";
            }
            Err(error) => {
                self.write_errors += 1;
                self.status = "open_failed";
                eprintln!(
                    "#graph-free-candidates: cannot create {} ({error}); nothing will be \
                     recorded, the verdict is unaffected",
                    path.display()
                );
                return;
            }
        }
        let pinned: Vec<Value> = emit_pin
            .iter()
            .enumerate()
            .filter_map(|(i, pin)| pin.map(|declared| json!([i, declared])))
            .collect();
        let header = json!({
            "schema": SCHEMA,
            "kind": "header",
            "onnx": onnx.display().to_string(),
            "vnnlib": vnnlib.display().to_string(),
            "sidefile": path.display().to_string(),
            "n_inputs": n,
            "pinned_count": pinned.len(),
            "pinned": pinned,
            "x_encoding": X_ENCODING,
            "x_semantics": "the f32 input tensor ny ran; the witness f64 view is the pinned \
                            declared value for a pinned dim and the exact widening of x \
                            otherwise",
            "outputs_semantics": "ny's own ONNX Runtime forward on THIS host at the default \
                                  graph-optimization level (disable_outputs: optimizations \
                                  disabled); NOT the organisers' checker",
            "arch": std::env::consts::ARCH,
            "os": std::env::consts::OS,
            "ny_version": env!("CARGO_PKG_VERSION"),
            "pid": std::process::id(),
            "unix_ms": unix_ms(),
            "source": self.source,
            "preset_key": PRESET_KEY,
            "verdict_authority": "none: diagnostic only; nothing in this file was accepted or \
                                  published because it is here",
        });
        self.write_value(&header);
    }

    /// Label the lane stage that proposes the next candidates.
    pub(super) fn set_stage(&mut self, stage: &'static str) {
        self.stage = stage;
    }

    /// Record one candidate and the outcome of its gates.
    pub(super) fn record_candidate(
        &mut self,
        spec: &VnnLibSpec,
        x: &[f32],
        out: &[f32],
        disable_out: Option<&[f32]>,
        outcome: Outcome,
    ) {
        self.counts.bump(outcome);
        let id = self.next_id;
        self.next_id += 1;
        let accepted = matches!(outcome, Outcome::AcceptedByLane { .. });
        if accepted {
            self.accepted_id = Some(id);
            self.accepted_recorded = false;
        }
        if !self.admit(self.candidate_records < MAX_CANDIDATE_RECORDS, x.len()) {
            return;
        }
        let mut record = self.point_record("candidate", id, spec, x, out);
        record.insert("outcome".into(), Value::from(outcome.token()));
        match outcome {
            Outcome::AcceptedByLane { corroborated } => {
                record.insert("corroborated".into(), Value::from(corroborated));
            }
            Outcome::RejectedDeadline { phase } | Outcome::RejectedForward { phase } => {
                record.insert("phase".into(), Value::from(phase));
            }
            Outcome::RejectedExtent { dim } => {
                record.insert("extent_dim".into(), Value::from(dim));
            }
            Outcome::RejectedCorroboration {
                radius,
                neighbour_margin,
                checked,
            } => {
                record.insert("corroboration_radius".into(), Value::from(radius));
                record.insert(
                    "corroboration_neighbour_margin".into(),
                    Value::from(neighbour_margin),
                );
                record.insert("corroboration_checked".into(), Value::from(checked));
            }
            Outcome::RejectedEmit
            | Outcome::RejectedDualUnavailable
            | Outcome::RejectedDualLevel
            | Outcome::RejectedPin => {}
        }
        if let Some(disable) = disable_out {
            let disable64: Vec<f64> = disable.iter().map(|&v| f32_to_f64_exact(v)).collect();
            record.insert(
                "disable_margin".into(),
                Value::from(property_margin(spec, x, &disable64)),
            );
            record.insert("disable_outputs".into(), Value::from(disable64));
            record.insert("disable_outputs_f32_bits".into(), f32_bits(disable));
        }
        if self.write_value(&Value::Object(record)) {
            self.candidate_records += 1;
            if accepted {
                self.accepted_recorded = true;
            }
        }
    }

    /// Observe one surrogate incumbent the flat-objective search re-scored on
    /// the ORIGINAL model (`search_end = false`), or the end of that search
    /// (`search_end = true`, slices empty). Violating incumbents are skipped:
    /// the lane offered them to `try_accept`, whose own record covers them.
    pub(super) fn observe_incumbent(
        &mut self,
        spec: &VnnLibSpec,
        x: &[f32],
        out: &[f32],
        real_margin: f64,
        surrogate_margin: f64,
        search_end: bool,
    ) {
        if self.writer.is_none() {
            return;
        }
        if search_end {
            if let Some(pending) = self.pending.take() {
                self.write_near_miss(
                    spec,
                    &pending.x,
                    &pending.out,
                    pending.real_margin,
                    pending.surrogate_margin,
                    pending.improvement,
                    true,
                );
            }
            return;
        }
        self.improvements += 1;
        if real_margin > 0.0 {
            self.pending = None;
            return;
        }
        if self.improvements.is_power_of_two() && self.near_miss_records < MAX_NEAR_MISS_RECORDS {
            self.pending = None;
            self.write_near_miss(
                spec,
                x,
                out,
                real_margin,
                surrogate_margin,
                self.improvements,
                false,
            );
            return;
        }
        let improvement = self.improvements;
        match self.pending.as_mut() {
            Some(pending) => {
                pending.x.clear();
                pending.x.extend_from_slice(x);
                pending.out.clear();
                pending.out.extend_from_slice(out);
                pending.real_margin = real_margin;
                pending.surrogate_margin = surrogate_margin;
                pending.improvement = improvement;
            }
            None => {
                self.pending = Some(PendingIncumbent {
                    x: x.to_vec(),
                    out: out.to_vec(),
                    real_margin,
                    surrogate_margin,
                    improvement,
                });
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write_near_miss(
        &mut self,
        spec: &VnnLibSpec,
        x: &[f32],
        out: &[f32],
        real_margin: f64,
        surrogate_margin: f64,
        improvement: u64,
        final_incumbent: bool,
    ) {
        let id = self.next_id;
        self.next_id += 1;
        let within = self.near_miss_records < MAX_NEAR_MISS_RECORDS + u64::from(final_incumbent);
        if !self.admit(within, x.len()) {
            return;
        }
        let mut record = self.point_record("near_miss", id, spec, x, out);
        record.insert("outcome".into(), Value::from("not_offered"));
        record.insert("real_margin".into(), Value::from(real_margin));
        record.insert("surrogate_margin".into(), Value::from(surrogate_margin));
        record.insert("improvement".into(), Value::from(improvement));
        record.insert("final_incumbent".into(), Value::from(final_incumbent));
        if self.write_value(&Value::Object(record)) {
            self.near_miss_records += 1;
        }
    }

    /// Cap check before a point record is built. `within_count` is the
    /// caller's per-kind count test.
    fn admit(&mut self, within_count: bool, n: usize) -> bool {
        if self.writer.is_none() {
            return false;
        }
        let approx = (n as u64).saturating_mul(4).saturating_mul(4) / 3 + 4096;
        if !within_count || self.bytes.saturating_add(approx) > MAX_SIDEFILE_BYTES {
            self.dropped_cap += 1;
            return false;
        }
        true
    }

    /// The fields every point record shares.
    fn point_record(
        &self,
        kind: &'static str,
        id: u64,
        spec: &VnnLibSpec,
        x: &[f32],
        out: &[f32],
    ) -> Map<String, Value> {
        let out64: Vec<f64> = out.iter().map(|&v| f32_to_f64_exact(v)).collect();
        let margin = property_margin(spec, x, &out64);
        let bytes = f32_le_bytes(x);
        let mut x_obj = Map::new();
        x_obj.insert("encoding".into(), Value::from(X_ENCODING));
        x_obj.insert("len".into(), Value::from(x.len()));
        x_obj.insert("sha256".into(), Value::String(sha256_hex(&bytes)));
        x_obj.insert("data".into(), Value::String(base64_encode(&bytes)));
        let mut record = Map::new();
        record.insert("schema".into(), Value::from(SCHEMA));
        record.insert("kind".into(), Value::from(kind));
        record.insert("id".into(), Value::from(id));
        record.insert(
            "t_ms".into(),
            Value::from(u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)),
        );
        record.insert("stage".into(), Value::from(self.stage));
        record.insert("default_margin".into(), Value::from(margin));
        record.insert("default_outputs".into(), Value::from(out64));
        record.insert("default_outputs_f32_bits".into(), f32_bits(out));
        record.insert("x".into(), Value::Object(x_obj));
        record
    }

    /// Serialize and append one line. Any I/O error stops recording (the
    /// verdict path never depends on it) and is counted in the receipt.
    fn write_value(&mut self, value: &Value) -> bool {
        let line = match serde_json::to_string(value) {
            Ok(line) => line,
            Err(_) => {
                self.write_errors += 1;
                return false;
            }
        };
        let result = match self.writer.as_mut() {
            None => return false,
            Some(writer) => writer
                .write_all(line.as_bytes())
                .and_then(|()| writer.write_all(b"\n"))
                .and_then(|()| writer.flush()),
        };
        match result {
            Ok(()) => {
                self.bytes = self.bytes.saturating_add(line.len() as u64 + 1);
                true
            }
            Err(error) => {
                self.write_errors += 1;
                self.writer = None;
                self.status = "write_failed";
                eprintln!(
                    "#graph-free-candidates: write failed ({error}); recording stops, the \
                     verdict is unaffected"
                );
                false
            }
        }
    }

    /// The fixed-schema receipt line. Every field is always present, so an
    /// armed null (`candidates=0`) is distinguishable from a lane that never
    /// engaged (no receipt at all after the `armed=1` line).
    pub(super) fn receipt(&self) -> String {
        let c = &self.counts;
        format!(
            "#graph-free-candidates: v1 armed=1 source={} status={} path={} records={} \
             candidates={} accepted={} rejected_emit={} rejected_dual_level={} \
             rejected_dual_unavailable={} rejected_extent={} rejected_corroboration={} \
             rejected_pin={} rejected_forward={} rejected_deadline={} near_miss={} \
             improvements={} dropped_cap={} write_errors={} bytes={} elapsed_ms={}",
            self.source,
            self.status,
            self.path
                .as_ref()
                .map_or_else(|| "none".to_string(), |p| p.display().to_string()),
            self.candidate_records + self.near_miss_records,
            c.seen,
            c.accepted,
            c.rejected_emit,
            c.rejected_dual_level,
            c.rejected_dual_unavailable,
            c.rejected_extent,
            c.rejected_corroboration,
            c.rejected_pin,
            c.rejected_forward,
            c.rejected_deadline,
            self.near_miss_records,
            self.improvements,
            self.dropped_cap,
            self.write_errors,
            self.bytes,
            self.started.elapsed().as_millis(),
        )
    }
}

impl Drop for CandidateSidefile {
    fn drop(&mut self) {
        if self.writer.is_some() {
            let c = self.counts;
            let summary = json!({
                "schema": SCHEMA,
                "kind": "summary",
                "status": self.status,
                "candidates": c.seen,
                "accepted": c.accepted,
                "rejected_emit": c.rejected_emit,
                "rejected_dual_level": c.rejected_dual_level,
                "rejected_dual_unavailable": c.rejected_dual_unavailable,
                "rejected_extent": c.rejected_extent,
                "rejected_corroboration": c.rejected_corroboration,
                "rejected_pin": c.rejected_pin,
                "rejected_forward": c.rejected_forward,
                "rejected_deadline": c.rejected_deadline,
                "candidate_records": self.candidate_records,
                "near_miss_records": self.near_miss_records,
                "improvements": self.improvements,
                "dropped_cap": self.dropped_cap,
                "accepted_id": self.accepted_id,
                "elapsed_ms": u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX),
                "unix_ms": unix_ms(),
            });
            self.write_value(&summary);
        }
        eprintln!("{}", self.receipt());
        if self.park_oracle {
            if let (Some(candidate_id), Some(path), true) = (
                self.accepted_id,
                self.path.clone(),
                self.status == "recording",
            ) {
                if let Ok(mut slot) = PENDING_ORACLE.lock() {
                    *slot = Some(PendingOracle {
                        path,
                        candidate_id,
                        candidate_recorded: self.accepted_recorded,
                    });
                }
            }
        }
    }
}

/// Lane hook: label the stage. No-op when dark.
pub(super) fn set_stage(sidefile: &Option<RefCell<CandidateSidefile>>, stage: &'static str) {
    if let Some(sidefile) = sidefile.as_ref() {
        sidefile.borrow_mut().set_stage(stage);
    }
}

/// Lane hook: one candidate left the acceptance stack. No-op when dark.
pub(super) fn note_candidate(
    sidefile: &Option<RefCell<CandidateSidefile>>,
    spec: &VnnLibSpec,
    x: &[f32],
    out: &[f32],
    disable_out: Option<&[f32]>,
    outcome: Outcome,
) {
    if let Some(sidefile) = sidefile.as_ref() {
        sidefile
            .borrow_mut()
            .record_candidate(spec, x, out, disable_out, outcome);
    }
}

/// Lane hook: a surrogate incumbent was re-scored (or the surrogate search
/// ended). No-op when dark.
pub(super) fn observe_incumbent(
    sidefile: &Option<RefCell<CandidateSidefile>>,
    spec: &VnnLibSpec,
    x: &[f32],
    out: &[f32],
    real_margin: f64,
    surrogate_margin: f64,
    search_end: bool,
) {
    if let Some(sidefile) = sidefile.as_ref() {
        sidefile.borrow_mut().observe_incumbent(
            spec,
            x,
            out,
            real_margin,
            surrogate_margin,
            search_end,
        );
    }
}

/// Caller hook, after `gate_sat_with_trusted_oracle`: append the ruling on the
/// lane's parked accepted candidate. `consulted` is whether a `sat` reached
/// the gate at all; `upheld` whether it came back `sat`. A no-op (no parked
/// candidate) on every dark run and on every armed run whose lane accepted
/// nothing.
pub(super) fn record_trusted_oracle_outcome(consulted: bool, upheld: bool) {
    let pending = PENDING_ORACLE.lock().ok().and_then(|mut slot| slot.take());
    let Some(pending) = pending else {
        return;
    };
    append_trusted_oracle(&pending, consulted, upheld);
}

/// Append one `trusted_oracle` record for `pending` and print its receipt.
/// Returns whether the line reached the file.
fn append_trusted_oracle(pending: &PendingOracle, consulted: bool, upheld: bool) -> bool {
    let line = json!({
        "schema": SCHEMA,
        "kind": "trusted_oracle",
        "candidate": pending.candidate_id,
        "candidate_recorded": pending.candidate_recorded,
        "consulted": consulted,
        "upheld": upheld,
        "unix_ms": unix_ms(),
    })
    .to_string();
    let appended = OpenOptions::new()
        .append(true)
        .open(&pending.path)
        .and_then(|mut file| {
            file.write_all(line.as_bytes())?;
            file.write_all(b"\n")
        })
        .is_ok();
    eprintln!(
        "#graph-free-candidates: trusted-oracle candidate={} consulted={} upheld={} appended={}",
        pending.candidate_id,
        u8::from(consulted),
        u8::from(upheld),
        u8::from(appended)
    );
    appended
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|d| u64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

fn f32_bits(values: &[f32]) -> Value {
    Value::from(
        values
            .iter()
            .map(|v| format!("{:08x}", v.to_bits()))
            .collect::<Vec<String>>(),
    )
}

/// Little-endian bytes of an f32 tensor (the exact payload that is hashed
/// and base64-encoded).
pub(super) fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Lowercase hex sha256.
pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// RFC 4648 base64 (standard alphabet, `=` padding). Hand-rolled because the
/// workspace carries no base64 crate; pinned against the RFC's test vectors.
pub(super) fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = chunk.get(1).copied().map_or(0, u32::from);
        let b2 = chunk.get(2).copied().map_or(0, u32::from);
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(char::from(ALPHABET[((triple >> 18) & 0x3f) as usize]));
        out.push(char::from(ALPHABET[((triple >> 12) & 0x3f) as usize]));
        if chunk.len() > 1 {
            out.push(char::from(ALPHABET[((triple >> 6) & 0x3f) as usize]));
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(char::from(ALPHABET[(triple & 0x3f) as usize]));
        } else {
            out.push('=');
        }
    }
    out
}

#[cfg(test)]
#[path = "graph_free_sidefile_tests.rs"]
mod tests;
