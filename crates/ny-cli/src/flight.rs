// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Flight recorder v0 (#flight-record, invariant I7 of
//! `docs/SYSTEM_DESIGN_ONE_PIPELINE_2026-07-30.md`).
//!
//! Every scored run must leave a machine-readable record of WHICH methods
//! ran/skipped and on what backend — in the result ARTIFACT, not the log
//! stream: the scored entry points run under `RUST_LOG=error`
//! (`run_instance.sh`, `vnncomp_sweep.rs`), so an info-level line is invisible
//! exactly where the record matters. Post-hoc questions like "did the upfront
//! attack lane ever arm on this row?" have repeatedly been unanswerable from
//! the bank (design doc §1 instances 3/6/7/9/10); the sidecar written here is
//! the artifact that answers them.
//!
//! v0 is deliberately shallow: the command layer notes the seams it already
//! knows (lane consulted/skipped, route chosen, preset resolved), with no
//! plumbing into ny-propagate. The record is process-global because the scored
//! contract is one instance per process (`ny vnncomp` is spawned per row by
//! both the competition harness and the sweep).
//!
//! Best-effort by construction: a sidecar write failure must NEVER affect the
//! verdict or the exit code — the recorder logs on stderr (which survives
//! `RUST_LOG=error` and lands in the per-instance logs) and continues.
//!
//! HARD-STOP COVERAGE (#flight-record-hard-stop). The sidecar was originally
//! written only from the command's post-verdict exit paths, so a row the
//! deadline watchdog HARD STOPS — the process is SIGKILLed/aborted, it never
//! returns through those paths — left no record at all. MEASURED on the
//! `isomorphic_acasxu_2026` sweep: sidecar coverage 32/50 rows and 1/19
//! TIMEOUT rows, i.e. the recorder was blind on exactly the rows where the
//! budget question has scoring consequence. [`FlightRecorder::write_hard_stop_sidecar`]
//! closes that hole from the watchdog thread, immediately before the stop.
//!
//! Why an early flush at the stop and not an incremental append:
//!
//! * cost — an append/rewrite per [`FlightRecorder::note`] pays file I/O
//!   inside every command-layer decision seam of every scored row; the
//!   pre-stop flush pays exactly one write, and only on the rows that are
//!   hard stopped. A row that finishes normally serializes exactly what it
//!   serialized before, byte for byte;
//! * torn files — the hard stop is a SIGKILL (helper) or `abort`, which can
//!   land mid-`write`. A file appended to as the run proceeds would be
//!   truncated in the middle of a JSON value precisely on the rows that
//!   matter. The flush is staged and `rename`d, so it lands whole or not at
//!   all;
//! * evidence quality — at the flush point the watchdog has already decided
//!   what it is committing to RESULTS_FILE, so the record can state it.
//!
//! A flush that runs before any verdict was recorded is marked partial in
//! three redundant places: the `record_status` envelope, a terminal
//! `run_incomplete` event, and the absence of the `run_complete` event that
//! closes every completed record. A reader must never mistake it for a
//! completed run: the lanes it lists are the lanes that ran BEFORE the stop,
//! and it says nothing about lanes that would have run after.
//!
//! What is STILL not covered, so a later audit does not assume otherwise:
//! the short-grace (safenlp) route, whose timer contract forbids any I/O
//! between its boundary and the abort, and a process killed only by the
//! out-of-process helper (budget + grace + 10 s) without its own watchdog
//! thread ever running. Both leave no record, exactly as before.

use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Bump when the sidecar shape changes; consumers (the sweep's metadata bank,
/// the expected-methods contracts) key on this.
///
/// v2: `ambient_env` (NY_* + OMP_NUM_THREADS snapshot, I10 flag-zero) and the
/// terminal `run_complete` event (lifecycle v0.5). That original ambient
/// capture contract is now called `ny-cli/ambient-env/v1`.
///
/// v3: adds a two-stage `levers` envelope. `begin` freezes the registered raw
/// environment inputs and records `not_materialized`; after the run context is
/// known and any preset snapshot is semantically validated, the command
/// replaces that state once with a layered registry receipt or an explicit
/// `invalid_config` reason. Older v2 sidecars remain a supported input to the
/// archival/qualification consumers.
///
/// `build_id` (#binary-identity) and `ambient_env_schema` were added to v3
/// WITHOUT a global bump, deliberately. Both are additive fields read by key,
/// and several Python gates refuse any record whose `schema_version` is not
/// exactly their pinned constant. A missing `ambient_env_schema` identifies
/// the legacy v1 capture; current proof requires v3, which captures both the
/// current and deprecated Rayon worker-count variables.
const SCHEMA_VERSION: u32 = 3;

/// Current semantic contract for [`FlightRecord::ambient_env`].
///
/// v1 was implicit and captured `NY_*` plus `OMP_NUM_THREADS`. v2 added
/// `RAYON_NUM_THREADS`, which controls the contextual default for margin-row
/// parallelism. v3 also captures deprecated `RAYON_RS_NUM_CPUS`, which
/// rayon-core 1.13 still honors whenever `RAYON_NUM_THREADS` is absent. Keeping
/// this version separate from the broad flight schema lets old v3 records remain
/// readable without treating them as current proof of a clean environment.
pub(crate) const AMBIENT_ENV_SCHEMA: &str = "ny-cli/ambient-env/v3";

/// What happened to one method/lane at a seam the command layer controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FlightStatus {
    /// The lane/decision was entered or consulted.
    Ran,
    /// The lane was reachable but a gate declined it; `reason` says which.
    Skipped,
    /// Control flow never got there (an earlier stage terminated the run).
    #[allow(dead_code)] // Wire-contract sentinel exercised by recorder tests.
    NotReached,
    /// Terminal disposition (lifecycle v0.5): used only by the one
    /// `run_complete` event that closes a record, so consumers can find the
    /// final verdict by type instead of by grepping method names.
    Complete,
    /// Terminal disposition of a record flushed because the run was about to
    /// be HARD STOPPED without ever reaching a verdict. Used only by the one
    /// `run_incomplete` event that closes such a record. Distinct from
    /// [`Self::Complete`] on the wire so a consumer that types the terminal
    /// event cannot read a truncated trace as a finished one.
    Incomplete,
}

/// One recorded seam: method name, disposition, why, and when.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct FlightEvent {
    pub(crate) method: String,
    pub(crate) status: FlightStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reason: Option<String>,
    /// Seconds since `begin` — the scored instance start, near enough for v0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) at_secs: Option<f64>,
}

/// Whether this record reached the point where layered registry resolution was
/// possible.
///
/// The scored command begins recording before it validates input files or
/// loads a category preset. An early error must not invent a default-only
/// receipt, because a valid run might later have received a typed preset.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum LeverReceiptState {
    /// Preset/config resolution did not complete. This is affirmative evidence
    /// that no layered receipt is being claimed, not a missing field.
    NotMaterialized,
    /// Preset/config resolution failed, or the typed receipt projection
    /// disagreed with its declaration. The reason is machine-readable
    /// evidence that no layered receipt is being claimed.
    InvalidConfig { reason: String },
    /// The registered raw environment snapshot was layered over the exact
    /// validated preset/config values for this run.
    Resolved { receipt: serde_json::Value },
}

/// Whether a serialized record describes a run that reached its own exit path
/// or one that was cut off (#flight-record-hard-stop).
///
/// Absent — the overwhelmingly common case — means the record was written by a
/// terminal exit path, and this is the shape every v2/v3 consumer already
/// reads. The field appears ONLY on a record flushed ahead of a hard stop, so
/// a completed row's sidecar is unchanged byte for byte.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum RecordStatus {
    /// The deadline watchdog was about to stop the process and no verdict had
    /// been recorded. `events` is a PREFIX of the run: every lane it lists did
    /// run, and a lane it does not list may simply never have been reached.
    PartialHardStop {
        /// What fired, and what it committed to RESULTS_FILE.
        reason: String,
        /// Seconds since `begin` at which the record was flushed — i.e. how
        /// far the row actually got before it was cut off.
        flushed_at_secs: f64,
    },
}

/// The per-run record serialized to `<results_file>.flight.json`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct FlightRecord {
    pub(crate) schema_version: u32,
    /// Partial-record marker; see [`RecordStatus`]. Skipped when the record is
    /// complete, which keeps the completed-row schema exactly as consumers
    /// (bank metadata, qualification scripts) already parse it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) record_status: Option<RecordStatus>,
    /// WHICH VERIFIER produced this row (#binary-identity): the canonical
    /// `NY_BUILD_STAMP_V1 …` line — derived version, cargo feature tier,
    /// compiled backend class, impl markers, commit + dirty digest, toolchain.
    ///
    /// Every other field here describes the CONDITIONS of a measurement; this
    /// one describes the PROGRAM. Four times this campaign a bank was
    /// attributed to the wrong binary — most recently 330 rows sealed against
    /// `cfe490f1… cuda` and measured by `e3d81cc2… mip-only wgpu` — and in
    /// none of those cases could the artifact have told anyone, because
    /// `ny --version` answered `0.3.0` on both sides.
    ///
    /// The STAMP, not the content-addressed `NY_BUILD_ID_V1`: hashing the
    /// executable in the scored child would be charged against a per-instance
    /// budget as small as 10 s. The `stamp=` digest is shared with the full id
    /// the sweep manifest carries, so the two cross-link exactly.
    pub(crate) build_id: String,
    /// `cuda` / `metal` / `gpu` / `cpu-only` from `compute_backend::detect()`.
    pub(crate) backend_kind: String,
    /// The one-line provenance summary sealed alongside the kind.
    pub(crate) backend_summary: String,
    /// Static host identity (#host-provenance): which machine class produced
    /// this timing.
    pub(crate) host: crate::compute_backend::HostProbe,
    /// 1/5/15-minute load averages sampled at `begin` — whether the box was
    /// quiet when the clock started.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) load_avg_at_begin: Option<[f64; 3]>,
    /// Sampled again when the sidecar is written, i.e. after the verdict —
    /// together with `load_avg_at_begin` this brackets the run. A timing
    /// whose brackets show heavy contention (load ≫ cores) is evidence of a
    /// measurement problem, recorded instead of remembered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) load_avg_at_end: Option<[f64; 3]>,
    pub(crate) category: String,
    /// The scored competition budget, not any internal tier.
    pub(crate) budget_secs: u64,
    /// Semantic version of the following ambient snapshot. Always present on
    /// current records; absence identifies the legacy implicit v1 contract.
    pub(crate) ambient_env_schema: &'static str,
    /// Every `NY_*` environment variable plus `OMP_NUM_THREADS`,
    /// `RAYON_NUM_THREADS`, and `RAYON_RS_NUM_CPUS`, snapshotted
    /// at `begin` (I10 #flag-zero, week-one enforcement): a dev-override that
    /// shaped this run must be visible in the artifact, because the bank
    /// linter's later refusal of non-allowlisted-flag rows can only judge
    /// what the record shows. BTreeMap so the serialization is sorted and
    /// deterministic. Always present — an empty map is affirmative evidence
    /// the run was flag-clean, not a missing measurement.
    pub(crate) ambient_env: BTreeMap<String, String>,
    /// Two-stage declared-input evidence. See [`LeverReceiptState`].
    pub(crate) levers: LeverReceiptState,
    pub(crate) events: Vec<FlightEvent>,
}

struct RecorderState {
    record: FlightRecord,
    /// Exact registered environment values captured at `begin`, including
    /// present non-UTF-8 values. Preset materialization must resolve against
    /// this frozen input rather than reopening the process environment.
    raw_levers: ny_levers::RawLeverInputs,
    /// Runtime authority paired with `record.levers`. A duplicate
    /// materialization receives this same allocation rather than constructing
    /// an unrecorded second policy.
    lever_authority: Option<Arc<ny_levers::LeverSet>>,
    /// Anchor for `at_secs`; monotonic so a wall-clock step cannot produce
    /// negative or wildly wrong offsets in the artifact.
    started: std::time::Instant,
    /// Terminal verdict supplied by `finish`. Held OUT of `record.events` so
    /// the `run_complete` event is appended exactly once per serialized
    /// sidecar, however many times `write_sidecar` runs.
    verdict: Option<String>,
    /// #stop-accounting — the engine's OWN stop reason, as constructed by
    /// `GraphBabLifecycle::unresolved_reason()` and carried on
    /// `BabVerificationStatus::Unknown`.
    ///
    /// That string names a depth cap, a queue eviction, an unsplittable
    /// domain — and it was CONSTRUCTED AND THEN THROWN AWAY: the verdict seam
    /// matches `Unknown { .. }` and returns a reasonless `VnncompResult`, so at
    /// competition verbosity an `unknown` row carried no stop reason at all and
    /// a row that quit on a depth cap with 94% of its budget unspent was
    /// indistinguishable in the record from one that ran out of clock.
    stop_reason: Option<String>,
}

/// Filter an arbitrary `(name, value)` stream down to the ambient flags the
/// record captures: every `NY_*` variable plus `OMP_NUM_THREADS`,
/// `RAYON_NUM_THREADS`, and deprecated `RAYON_RS_NUM_CPUS`.
///
/// Takes an iterator instead of reading the process environment so unit tests
/// need no env mutation — the real environment is process-global and tests run
/// in parallel; injected pairs sidestep that entirely.
pub(crate) fn ambient_env_from(
    vars: impl IntoIterator<Item = (String, String)>,
) -> BTreeMap<String, String> {
    vars.into_iter()
        .filter(|(name, _)| {
            name.starts_with("NY_")
                || matches!(
                    name.as_str(),
                    "OMP_NUM_THREADS" | "RAYON_NUM_THREADS" | "RAYON_RS_NUM_CPUS"
                )
        })
        .collect()
}

/// The production capture: the real process environment, lossily decoded so a
/// non-UTF-8 value stays visible in the artifact rather than silently vanishing.
fn ambient_env_from_os(
    vars: impl IntoIterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
) -> BTreeMap<String, String> {
    ambient_env_from(vars.into_iter().map(|(name, value)| {
        (
            name.to_string_lossy().into_owned(),
            value.to_string_lossy().into_owned(),
        )
    }))
}

pub(crate) fn ambient_env_snapshot() -> BTreeMap<String, String> {
    ambient_env_from_os(std::env::vars_os())
}

/// Event sink with interior mutability so call sites need no plumbing.
///
/// Tests construct their own instances; production code uses [`global`]. Lock
/// poisoning is deliberately shrugged off (`into_inner`): the recorder must
/// never convert someone else's panic into a second failure on the verdict
/// path.
pub(crate) struct FlightRecorder {
    state: Mutex<Option<RecorderState>>,
}

impl FlightRecorder {
    pub(crate) const fn new() -> Self {
        Self {
            state: Mutex::new(None),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<RecorderState>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Arm the recorder for this run, replacing any previous record.
    ///
    /// Until `begin` runs, `note` and `write_sidecar` are no-ops — entry
    /// points that never opted into flight recording cannot leave partial or
    /// misattributed sidecars.
    pub(crate) fn begin(
        &self,
        backend_kind: &str,
        backend_summary: &str,
        category: &str,
        budget_secs: u64,
    ) {
        // Anchor timings before provenance capture: enumerating the host and
        // environment is part of the run whose timing the record describes.
        let started = std::time::Instant::now();
        let raw_levers = ny_levers::RawLeverInputs::capture(ny_levers::all());
        *self.lock() = Some(RecorderState {
            record: FlightRecord {
                schema_version: SCHEMA_VERSION,
                record_status: None,
                build_id: crate::build_id::current_line().to_string(),
                backend_kind: backend_kind.to_string(),
                backend_summary: backend_summary.to_string(),
                host: crate::compute_backend::host().clone(),
                load_avg_at_begin: crate::compute_backend::load_average(),
                load_avg_at_end: None,
                category: category.to_string(),
                budget_secs,
                ambient_env_schema: AMBIENT_ENV_SCHEMA,
                ambient_env: ambient_env_snapshot(),
                levers: LeverReceiptState::NotMaterialized,
                events: Vec::new(),
            },
            raw_levers,
            lever_authority: None,
            started,
            verdict: None,
            stop_reason: None,
        });
    }

    /// Materialize the layered registry receipt after typed preset/config
    /// validation succeeds.
    ///
    /// `config_value` supplies the contextual lower layer. The raw environment
    /// captured by [`Self::begin`] has precedence, including a present invalid
    /// value that deliberately suppresses a preset. A no-preset run calls this
    /// with an all-`None` lookup and therefore still gets a resolved receipt.
    pub(crate) fn materialize_levers(
        &self,
        config_value: impl FnMut(&'static ny_levers::LeverDecl) -> Option<ny_levers::LeverValue>,
    ) -> Option<Arc<ny_levers::LeverSet>> {
        let mut guard = self.lock();
        let state = guard.as_mut()?;
        if let Some(levers) = &state.lever_authority {
            // Like `finish`, first authority wins. A duplicate call is a
            // recorder bug, but instrumentation must not rewrite evidence.
            // Returning the same Arc also prevents that bug from suppressing
            // verifier execution or handing runtime an unrecorded snapshot.
            return Some(Arc::clone(levers));
        }
        if !matches!(state.record.levers, LeverReceiptState::NotMaterialized) {
            return None;
        }
        match ny_levers::LeverSet::resolve_layered(
            ny_levers::all(),
            &state.raw_levers,
            config_value,
        ) {
            Ok(levers) => {
                let levers = Arc::new(levers);
                state.record.levers = LeverReceiptState::Resolved {
                    receipt: levers.receipt(),
                };
                state.lever_authority = Some(Arc::clone(&levers));
                Some(levers)
            }
            Err(error) => {
                state.record.levers = LeverReceiptState::InvalidConfig {
                    reason: error.to_string(),
                };
                None
            }
        }
    }

    /// Record that the frozen preset/config was invalid before a layered
    /// registry receipt could be materialized.
    ///
    /// This is evidence only. It must never replace an already-resolved state
    /// or become a second error path for the verifier.
    pub(crate) fn mark_levers_invalid_config(&self, reason: &str) {
        if let Some(state) = self.lock().as_mut() {
            if matches!(state.record.levers, LeverReceiptState::NotMaterialized) {
                state.record.levers = LeverReceiptState::InvalidConfig {
                    reason: reason.to_owned(),
                };
            }
        }
    }

    /// Append one event. A no-op before `begin` (see there for why).
    pub(crate) fn note(&self, method: &str, status: FlightStatus, reason: Option<String>) {
        if let Some(state) = self.lock().as_mut() {
            let at_secs = state.started.elapsed().as_secs_f64();
            state.record.events.push(FlightEvent {
                method: method.to_string(),
                status,
                reason,
                at_secs: Some(at_secs),
            });
        }
    }

    /// #stop-accounting — record the engine's own stop reason for the terminal
    /// accounting line. Last writer wins: the reason that reaches the verdict
    /// seam is the one that decided the published verdict.
    pub(crate) fn note_stop_reason(&self, reason: &str) {
        let reason = reason.trim();
        if reason.is_empty() {
            return;
        }
        if let Some(state) = self.lock().as_mut() {
            state.stop_reason = Some(reason.to_string());
        }
    }

    /// A copy of the current record, for tests and the final serialization.
    pub(crate) fn snapshot(&self) -> Option<FlightRecord> {
        self.lock().as_ref().map(|state| state.record.clone())
    }

    /// Record the terminal verdict (lifecycle v0.5). A no-op before `begin`.
    ///
    /// The first verdict wins: the command's exit paths are mutually
    /// exclusive, so a second call can only be a bug, and the record must
    /// keep the verdict that was actually published, not the latest caller's
    /// opinion of it.
    pub(crate) fn finish(&self, verdict: &str) {
        let mut accounting = None;
        if let Some(state) = self.lock().as_mut() {
            if state.verdict.is_none() {
                state.verdict = Some(verdict.to_string());
                accounting = Some((
                    state.started.elapsed().as_secs_f64(),
                    state.record.budget_secs,
                    state.stop_reason.clone(),
                ));
            }
        }
        if let Some((wall_s, budget_secs, stop_reason)) = accounting {
            eprintln!(
                "{}",
                stop_accounting_line(verdict, wall_s, budget_secs, stop_reason.as_deref())
            );
        }
    }

    /// Serialize the record to `<results_file>.flight.json`, best-effort.
    ///
    /// INFALLIBLE BY CONTRACT: whatever goes wrong (record never begun, JSON
    /// failure, unwritable path), the verdict and exit code are already
    /// decided elsewhere and must not change — log on stderr and return. The
    /// sidecar sits NEXT TO the results file because that is the one location
    /// every harness (competition wrapper, sweep child tempdir) already
    /// collects.
    pub(crate) fn write_sidecar(&self, results_file: &Path) {
        let Some(mut record) = self.snapshot() else {
            return;
        };
        let terminal = self.lock().as_ref().and_then(|state| {
            state
                .verdict
                .clone()
                .map(|verdict| (verdict, state.started.elapsed().as_secs_f64()))
        });
        // Terminal disposition (lifecycle v0.5): appended to the serialized
        // copy, never the live record, so the sidecar carries exactly one
        // `run_complete` however many exit paths write it. `at_secs` here is
        // the total elapsed since `begin` — the run's wall time.
        if let Some((verdict, elapsed)) = terminal {
            record.events.push(FlightEvent {
                method: "run_complete".to_string(),
                status: FlightStatus::Complete,
                reason: Some(verdict),
                at_secs: Some(elapsed),
            });
        }
        // Close the load bracket at write time — after the verdict, so the
        // sample covers the run it describes.
        record.load_avg_at_end = crate::compute_backend::load_average();
        let path = sidecar_path(results_file);
        let body = match serde_json::to_string_pretty(&record) {
            Ok(body) => body,
            Err(error) => {
                eprintln!(
                    "flight recorder: serialization failed ({error}); the verdict is unaffected"
                );
                return;
            }
        };
        if let Err(error) = std::fs::write(&path, body) {
            eprintln!(
                "flight recorder: could not write {} ({error}); the verdict is unaffected",
                path.display()
            );
        }
    }

    /// Flush the record from the deadline watchdog, immediately BEFORE the
    /// process is hard stopped (#flight-record-hard-stop).
    ///
    /// The command's post-verdict `write_sidecar` calls can only run on paths
    /// that return; a watchdog stop never returns through them, which is why
    /// hard-stopped rows had no sidecar at all. This is the same record, taken
    /// one instant earlier.
    ///
    /// Two dispositions, decided by whether a terminal verdict was recorded:
    ///
    /// * `finish` HAS run — the row reached its answer and the stop is only
    ///   forcing teardown (the measured "published `sat` at 1.7 s, then wedged
    ///   in exit handlers" class). The record is COMPLETE and is written as
    ///   such; it is byte-equivalent to what the wedged thread would have
    ///   written, so this rescues rather than degrades that row's evidence.
    /// * no verdict yet — the run was cut off. The record is marked
    ///   [`RecordStatus::PartialHardStop`] and closed with a `run_incomplete`
    ///   terminal event.
    ///
    /// SAFETY OF THE DEADLINE, which outranks the record:
    ///
    /// * it never blocks — the state lock is polled for a bounded moment and
    ///   the flush is abandoned rather than waited on, so a hypothetical lock
    ///   holder cannot delay the stop;
    /// * it writes a staged file and `rename`s it, so a stop that lands during
    ///   the write leaves either the previous sidecar or the new one, never a
    ///   half-serialized one;
    /// * it touches only the sidecar path, never RESULTS_FILE, and the caller
    ///   invokes it only AFTER the watchdog has committed its verdict.
    pub(crate) fn write_hard_stop_sidecar(&self, results_file: &Path, reason: &str) {
        // The lock is only ever held for a push or a receipt projection — no
        // I/O, no waiting — so contention here is a microsecond-scale race,
        // not a wedge. Poll it briefly, then give up: the stop owns the clock.
        const LOCK_ATTEMPTS: u32 = 5;
        const LOCK_POLL: std::time::Duration = std::time::Duration::from_millis(5);
        let mut guard = None;
        for attempt in 0..LOCK_ATTEMPTS {
            match self.state.try_lock() {
                Ok(locked) => {
                    guard = Some(locked);
                    break;
                }
                // Someone else's panic must not cost the record.
                Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                    guard = Some(poisoned.into_inner());
                    break;
                }
                Err(std::sync::TryLockError::WouldBlock) => {
                    if attempt + 1 < LOCK_ATTEMPTS {
                        std::thread::sleep(LOCK_POLL);
                    }
                }
            }
        }
        let Some(mut guard) = guard else {
            return;
        };
        // Before `begin`, exactly as elsewhere: an entry point that never
        // opted into flight recording leaves no sidecar.
        let Some(state) = guard.as_mut() else {
            return;
        };
        let at_secs = state.started.elapsed().as_secs_f64();
        let mut record = state.record.clone();
        match state.verdict.clone() {
            Some(verdict) => record.events.push(FlightEvent {
                method: "run_complete".to_string(),
                status: FlightStatus::Complete,
                reason: Some(verdict),
                at_secs: Some(at_secs),
            }),
            None => {
                record.record_status = Some(RecordStatus::PartialHardStop {
                    reason: reason.to_string(),
                    flushed_at_secs: at_secs,
                });
                record.events.push(FlightEvent {
                    method: "run_incomplete".to_string(),
                    status: FlightStatus::Incomplete,
                    reason: Some(reason.to_string()),
                    at_secs: Some(at_secs),
                });
            }
        }
        record.load_avg_at_end = crate::compute_backend::load_average();
        // Release the lock before touching the filesystem: a note() racing the
        // stop should never queue behind a write.
        drop(guard);
        write_record_atomically(&record, results_file);
    }
}

/// Serialize `record` next to the results file so a concurrent hard stop can
/// never expose a partially written sidecar: stage in the same directory, then
/// `rename` (atomic on every platform this ships to). Infallible by the same
/// contract as [`FlightRecorder::write_sidecar`], and silent — the watchdog
/// path forbids stderr writes, which can block on a full pipe.
fn write_record_atomically(record: &FlightRecord, results_file: &Path) {
    let path = sidecar_path(results_file);
    let Ok(body) = serde_json::to_string_pretty(record) else {
        return;
    };
    let mut staged = path.clone().into_os_string();
    staged.push(format!(".{}.staging", std::process::id()));
    let staged = PathBuf::from(staged);
    if std::fs::write(&staged, body).is_ok() {
        if std::fs::rename(&staged, &path).is_err() {
            // Leave nothing behind that a harness could collect as a sidecar.
            let _ = std::fs::remove_file(&staged);
        }
    } else {
        let _ = std::fs::remove_file(&staged);
    }
}

/// The process-global recorder the scored `ny vnncomp` path writes through.
pub(crate) fn global() -> &'static FlightRecorder {
    static GLOBAL: FlightRecorder = FlightRecorder::new();
    &GLOBAL
}

/// `<results_file>.flight.json` — appended, not `with_extension`, so
/// `result.txt` maps to `result.txt.flight.json` and can never collide with
/// the results file itself.
pub(crate) fn sidecar_path(results_file: &Path) -> PathBuf {
    let mut os = results_file.as_os_str().to_os_string();
    os.push(".flight.json");
    PathBuf::from(os)
}

/// Convenience for call sites deep in the command layer.
pub(crate) fn note(method: &str, status: FlightStatus, reason: Option<String>) {
    global().note(method, status, reason);
}

/// Convenience for the command's exit paths: record the terminal verdict on
/// the global recorder before the sidecar write.
pub(crate) fn finish(verdict: &str) {
    global().finish(verdict);
}

/// #stop-accounting — record the engine's own stop reason on the global
/// recorder. See [`FlightRecorder::note_stop_reason`].
pub(crate) fn note_stop_reason(reason: &str) {
    global().note_stop_reason(reason);
}

/// #stop-accounting — the terminal budget-accounting line, as a PURE function
/// so the arithmetic is testable without a clock, a model, or a box slot.
///
/// One line, on stderr, at every publish path, at default verbosity. It exists
/// because this campaign's bank cannot currently distinguish the three things a
/// non-verdict row can be:
///
/// * a GENUINE TIMEOUT — `unspent_s` near zero;
/// * an EARLY QUIT — `unspent_s` large AND a `reason=` naming an effort cap
///   (depth, queue eviction, unsplittable domain);
/// * a FAST REFUSAL — `unspent_s` large and a structural reason.
///
/// A banked `timeout` whose wall is far under budget is not a capability fact,
/// and until this line existed the only way to tell was to re-run the row with
/// `-v` and a dark telemetry variable.
pub(crate) fn stop_accounting_line(
    verdict: &str,
    wall_s: f64,
    budget_secs: u64,
    stop_reason: Option<&str>,
) -> String {
    let budget = budget_secs as f64;
    let unspent = (budget - wall_s).max(0.0);
    let unspent_frac = if budget > 0.0 { unspent / budget } else { 0.0 };
    format!(
        "[stop] verdict={verdict} wall_s={wall_s:.2} budget_s={budget_secs} \
         unspent_s={unspent:.2} unspent_frac={unspent_frac:.3} reason={}",
        stop_reason.unwrap_or("none-recorded"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #stop-accounting — CONTAINMENT on the arithmetic, not finiteness.
    ///
    /// The three shapes this line has to separate are exactly the three causes
    /// the board audit could not tell apart from a banked row: a genuine
    /// timeout, an early quit, and a fast refusal.
    #[test]
    fn stop_accounting_line_reports_the_unspent_budget_and_the_engine_reason() {
        // EARLY QUIT: an effort cap ended the search with 94% of the budget
        // still on the clock. This is the ml4acopf 118_ieee_prop2 shape.
        let early = stop_accounting_line("unknown", 6.14, 600, Some("Max depth 100 reached"));
        assert!(early.contains("verdict=unknown"), "{early}");
        assert!(early.contains("wall_s=6.14"), "{early}");
        assert!(early.contains("budget_s=600"), "{early}");
        assert!(early.contains("unspent_s=593.86"), "{early}");
        assert!(early.contains("unspent_frac=0.990"), "{early}");
        assert!(
            early.contains("reason=Max depth 100 reached"),
            "the engine's own stop reason must survive to the verdict seam: {early}"
        );

        // GENUINE TIMEOUT: the wall really is the budget.
        let genuine = stop_accounting_line("timeout", 98.30, 100, None);
        assert!(genuine.contains("unspent_s=1.70"), "{genuine}");
        assert!(genuine.contains("unspent_frac=0.017"), "{genuine}");
        assert!(
            genuine.contains("reason=none-recorded"),
            "an unrecorded reason must SAY SO rather than read as an empty one: {genuine}"
        );

        // A run that overran its budget must report zero unspent, never a
        // negative one that would read as extra capacity.
        let overrun = stop_accounting_line("timeout", 105.10, 30, None);
        assert!(overrun.contains("unspent_s=0.00"), "{overrun}");
        assert!(overrun.contains("unspent_frac=0.000"), "{overrun}");

        // An unbounded run has no budget to be under.
        let unbounded = stop_accounting_line("unsat", 12.0, 0, None);
        assert!(unbounded.contains("unspent_frac=0.000"), "{unbounded}");
    }

    /// #stop-accounting — the reason recorded by the verdict seam must reach
    /// the terminal line. A recorder that silently drops it reproduces the very
    /// defect this exists to remove.
    #[test]
    fn recorder_carries_the_engine_stop_reason_to_the_terminal_line() {
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "ml4acopf_2024", 600);
        recorder.note_stop_reason("   Max depth 100 reached   ");
        let carried = recorder
            .lock()
            .as_ref()
            .and_then(|state| state.stop_reason.clone());
        assert_eq!(
            carried.as_deref(),
            Some("Max depth 100 reached"),
            "the reason must be recorded, trimmed, on the live recorder"
        );

        // An empty reason must never overwrite a real one.
        recorder.note_stop_reason("  ");
        let after = recorder
            .lock()
            .as_ref()
            .and_then(|state| state.stop_reason.clone());
        assert_eq!(after.as_deref(), Some("Max depth 100 reached"));
    }

    #[test]
    fn recorder_captures_events_in_order_with_timing_and_header() {
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "acasxu_2023", 116);
        recorder.note("upfront_attack", FlightStatus::Ran, None);
        recorder.note(
            "margin_row_concurrent",
            FlightStatus::Skipped,
            Some("not armed".into()),
        );
        recorder.note("postbab_leftover", FlightStatus::NotReached, None);

        let record = recorder.snapshot().expect("record exists after begin");
        assert_eq!(record.schema_version, 3);
        assert_eq!(record.backend_kind, "cpu-only");
        assert_eq!(record.category, "acasxu_2023");
        assert_eq!(record.budget_secs, 116);
        let methods: Vec<&str> = record
            .events
            .iter()
            .map(|event| event.method.as_str())
            .collect();
        assert_eq!(
            methods,
            [
                "upfront_attack",
                "margin_row_concurrent",
                "postbab_leftover"
            ],
            "events must keep call order — the record IS the execution trace"
        );
        assert_eq!(record.events[1].status, FlightStatus::Skipped);
        assert_eq!(record.events[1].reason.as_deref(), Some("not armed"));
        for event in &record.events {
            let at = event.at_secs.expect("every noted event is timestamped");
            assert!(at >= 0.0, "monotonic anchor cannot go negative: {at}");
        }
    }

    #[test]
    fn sidecar_lands_next_to_the_results_file_as_machine_readable_json() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.begin("metal", "metal [test]", "cgan_2023", 300);
        recorder.note(
            "preset",
            FlightStatus::Ran,
            Some("configs/vnncomp26/cgan_2023.yaml".into()),
        );
        recorder.note("internal_verifier", FlightStatus::NotReached, None);
        recorder.write_sidecar(&results_file);

        let sidecar = tmp.path().join("result.txt.flight.json");
        let body = std::fs::read_to_string(&sidecar).expect("sidecar written next to results");
        let json: serde_json::Value = serde_json::from_str(&body).expect("sidecar is valid JSON");
        assert_eq!(json["schema_version"], 3);
        assert_eq!(json["levers"]["status"], "not_materialized");
        assert_eq!(json["backend_kind"], "metal");
        assert_eq!(json["budget_secs"], 300);
        assert_eq!(json["ambient_env_schema"], AMBIENT_ENV_SCHEMA);
        assert!(
            json["ambient_env"].is_object(),
            "the flag snapshot is always present, even when empty (flag-clean is a measurement)"
        );
        assert_eq!(json["events"][0]["method"], "preset");
        assert_eq!(json["events"][0]["status"], "ran");
        // The wire spelling of NotReached is contract: consumers grep for it.
        assert_eq!(json["events"][1]["status"], "not_reached");
        // No `finish` was called, so no terminal event may be invented.
        let events = json["events"].as_array().expect("events array");
        assert!(
            events.iter().all(|event| event["method"] != "run_complete"),
            "run_complete requires an explicit verdict from the caller"
        );
    }

    #[test]
    fn a_failed_sidecar_write_is_non_fatal() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // A regular file where a directory is needed makes every write under
        // it fail with ENOTDIR on every platform we build for.
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"not a directory").expect("write blocker");
        let results_file = blocker.join("result.txt");

        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "safenlp_2024", 20);
        recorder.note("internal_verifier", FlightStatus::Ran, None);
        // Must return normally: the write failure may cost the artifact but
        // never the verdict or the exit code.
        recorder.write_sidecar(&results_file);
        assert!(
            recorder.snapshot().is_some(),
            "the in-memory record survives a failed write"
        );
    }

    #[test]
    fn before_begin_the_recorder_is_inert() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.note("upfront_attack", FlightStatus::Ran, None);
        recorder.write_sidecar(&results_file);
        assert!(
            recorder.snapshot().is_none(),
            "note before begin is a no-op"
        );
        assert!(
            !sidecar_path(&results_file).exists(),
            "no partial sidecar may appear for a run that never began recording"
        );
    }

    #[test]
    fn ambient_capture_keeps_ny_flags_and_thread_controls_only_sorted() {
        // Injected pairs, not the process environment: env is process-global
        // and tests run in parallel, so the capture is tested at the seam
        // that filters, with production supplying the real iterator.
        let captured = ambient_env_from([
            ("PATH".to_string(), "/usr/bin".to_string()),
            ("NY_TEST_FLAG".to_string(), "1".to_string()),
            ("NY_ATTACK_ARMING_BLOCK".to_string(), "1".to_string()),
            ("OMP_NUM_THREADS".to_string(), "8".to_string()),
            ("RAYON_NUM_THREADS".to_string(), "16".to_string()),
            ("RAYON_RS_NUM_CPUS".to_string(), "4".to_string()),
            ("NY_A_EARLIER".to_string(), "yes".to_string()),
            // Prefix must anchor at the start: an embedded NY_ is not a flag.
            ("SOME_NY_LOOKALIKE".to_string(), "no".to_string()),
            ("HOME".to_string(), "/var/empty".to_string()),
        ]);
        let pairs: Vec<(&str, &str)> = captured
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(
            pairs,
            [
                ("NY_ATTACK_ARMING_BLOCK", "1"),
                ("NY_A_EARLIER", "yes"),
                ("NY_TEST_FLAG", "1"),
                ("OMP_NUM_THREADS", "8"),
                ("RAYON_NUM_THREADS", "16"),
                ("RAYON_RS_NUM_CPUS", "4"),
            ],
            "NY_* plus OMP/current-and-legacy-Rayon thread controls, sorted by name — deterministic artifact bytes"
        );
    }

    #[cfg(unix)]
    #[test]
    fn ambient_capture_keeps_non_utf8_ny_presence_visible() {
        use std::os::unix::ffi::OsStringExt;

        let captured = ambient_env_from_os([
            (
                std::ffi::OsString::from_vec(b"NY_NON_UTF8_\xff".to_vec()),
                std::ffi::OsString::from_vec(vec![0xfe]),
            ),
            (
                std::ffi::OsString::from_vec(b"UNRELATED_\xff".to_vec()),
                std::ffi::OsString::from("hidden"),
            ),
        ]);
        assert_eq!(captured.len(), 1);
        assert_eq!(
            captured.get("NY_NON_UTF8_\u{fffd}").map(String::as_str),
            Some("\u{fffd}"),
            "schema v3 preserves non-Unicode NY_* presence with a deterministic lossy spelling"
        );
    }

    #[test]
    fn begin_records_that_layered_levers_are_not_yet_materialized() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "acasxu_2023", 116);
        recorder.write_sidecar(&results_file);

        let body = std::fs::read_to_string(sidecar_path(&results_file)).expect("sidecar written");
        let json: serde_json::Value = serde_json::from_str(&body).expect("sidecar is valid JSON");
        assert_eq!(json["levers"]["status"], "not_materialized");
        assert!(
            json["levers"].get("receipt").is_none(),
            "begin cannot claim a layered default before preset resolution"
        );
    }

    #[test]
    fn materialize_levers_projects_the_contextual_config_layer() {
        ny_test_utils::env::with_serialized_env_vars_removed(&["NY_ALPHA_ZERO_YIELD_FRAC"], || {
            let recorder = FlightRecorder::new();
            recorder.begin("cpu-only", "cpu-only [test]", "custom", 116);
            let frozen = recorder
                .materialize_levers(|decl| {
                    std::ptr::eq(
                        decl,
                        &raw const ny_levers::decls::root_alpha::ALPHA_ZERO_YIELD_FRAC,
                    )
                    .then_some(ny_levers::LeverValue::F64(0.25))
                })
                .expect("valid first materialization returns runtime authority");
            let duplicate = recorder
                .materialize_levers(|decl| {
                    std::ptr::eq(
                        decl,
                        &raw const ny_levers::decls::root_alpha::ALPHA_ZERO_YIELD_FRAC,
                    )
                    .then_some(ny_levers::LeverValue::F64(0.5))
                })
                .expect("duplicate receives the first frozen authority");
            assert!(
                Arc::ptr_eq(&frozen, &duplicate),
                "receipt and every runtime caller must share one allocation"
            );
            let record = recorder.snapshot().expect("record exists");
            let LeverReceiptState::Resolved { receipt } = record.levers else {
                panic!("validated config must resolve the receipt");
            };
            assert_eq!(receipt["schema"], "ny-levers/receipt/v3");
            let levers = receipt["levers"].as_array().expect("levers array");
            let zero_yield = levers
                .iter()
                .find(|l| l["name"] == "NY_ALPHA_ZERO_YIELD_FRAC")
                .expect("declared lever present");
            assert_eq!(zero_yield["value"], 0.25);
            assert_eq!(zero_yield["source"], "config");
            assert_eq!(
                frozen.get(&ny_levers::decls::root_alpha::ALPHA_ZERO_YIELD_FRAC),
                ny_levers::LeverValue::F64(0.25),
                "the object handed to runtime must be the object serialized into the receipt"
            );
        });
    }

    #[test]
    fn inadmissible_typed_projection_is_recorded_without_panicking() {
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "custom", 116);
        let materialized = recorder.materialize_levers(|decl| {
            std::ptr::eq(
                decl,
                &raw const ny_levers::decls::root_alpha::ALPHA_ZERO_YIELD_FRAC,
            )
            .then_some(ny_levers::LeverValue::F64(0.95))
        });
        assert!(materialized.is_none());

        let record = recorder.snapshot().expect("record exists");
        let LeverReceiptState::InvalidConfig { reason } = record.levers else {
            panic!("inadmissible typed config must be explicit evidence");
        };
        assert!(reason.contains("NY_ALPHA_ZERO_YIELD_FRAC"));
    }

    #[test]
    fn invalid_preset_reason_is_explicit_and_first_authority_wins() {
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "custom", 116);
        recorder.mark_levers_invalid_config("preset field was out of range");
        assert!(recorder.materialize_levers(|_| None).is_none());

        let record = recorder.snapshot().expect("record exists");
        let LeverReceiptState::InvalidConfig { reason } = record.levers else {
            panic!("invalid preset must remain explicit evidence");
        };
        assert_eq!(reason, "preset field was out of range");
    }

    #[test]
    fn finish_appends_exactly_one_terminal_event_with_verdict_and_elapsed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "acasxu_2023", 116);
        recorder.note("result_publish", FlightStatus::Ran, Some("unsat".into()));
        recorder.finish("unsat");
        // A buggy second finish must not rewrite the published verdict.
        recorder.finish("sat");
        // Two writes model two exit paths racing to serialize; each sidecar
        // must still end with exactly one run_complete.
        recorder.write_sidecar(&results_file);
        recorder.write_sidecar(&results_file);

        let body = std::fs::read_to_string(sidecar_path(&results_file)).expect("sidecar written");
        let json: serde_json::Value = serde_json::from_str(&body).expect("sidecar is valid JSON");
        let events = json["events"].as_array().expect("events array");
        let terminals: Vec<_> = events
            .iter()
            .filter(|event| event["method"] == "run_complete")
            .collect();
        assert_eq!(terminals.len(), 1, "exactly one terminal disposition");
        let terminal = events.last().expect("at least the terminal event");
        assert_eq!(
            terminal["method"], "run_complete",
            "the terminal event closes the record"
        );
        assert_eq!(terminal["status"], "complete");
        assert_eq!(terminal["reason"], "unsat", "first verdict wins");
        let elapsed = terminal["at_secs"].as_f64().expect("total elapsed");
        assert!(elapsed >= 0.0, "elapsed is measured from begin: {elapsed}");
    }

    #[test]
    fn finish_before_begin_is_inert() {
        let recorder = FlightRecorder::new();
        recorder.finish("unsat");
        assert!(
            recorder.snapshot().is_none(),
            "a verdict cannot arm a recorder that never began"
        );
    }

    #[test]
    fn a_hard_stopped_run_still_records_which_lanes_ran_marked_partial() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "isomorphic_acasxu_2026", 100);
        recorder.note("internal_timeout_route", FlightStatus::Ran, None);
        recorder.note(
            "relational_route",
            FlightStatus::Ran,
            Some("dual-network category".into()),
        );
        // No `finish`: the watchdog is stopping a run that never got a verdict.
        recorder.write_hard_stop_sidecar(&results_file, "in-process watchdog; timeout committed");

        let body = std::fs::read_to_string(sidecar_path(&results_file))
            .expect("a hard-stopped row now leaves a sidecar");
        let json: serde_json::Value = serde_json::from_str(&body).expect("sidecar is valid JSON");
        // The record answers the question the audit could not: which lanes ran.
        let methods: Vec<&str> = json["events"]
            .as_array()
            .expect("events array")
            .iter()
            .map(|event| event["method"].as_str().expect("method"))
            .collect();
        assert_eq!(
            methods,
            [
                "internal_timeout_route",
                "relational_route",
                "run_incomplete"
            ],
            "the prefix that ran, closed by an explicit incomplete terminal"
        );
        // Marked partial in three independent places, because a reader that
        // took this for a completed run would conclude the unlisted lanes were
        // skipped rather than never reached.
        assert_eq!(json["record_status"]["status"], "partial_hard_stop");
        assert_eq!(
            json["record_status"]["reason"],
            "in-process watchdog; timeout committed"
        );
        assert!(
            json["record_status"]["flushed_at_secs"].as_f64().is_some(),
            "a partial record must say how far the row got"
        );
        let terminal = json["events"].as_array().expect("events").last().cloned();
        let terminal = terminal.expect("terminal event");
        assert_eq!(terminal["status"], "incomplete");
        assert!(
            json["events"]
                .as_array()
                .expect("events")
                .iter()
                .all(|event| event["method"] != "run_complete"),
            "run_complete is the completed-run marker and must never appear here"
        );
        // Still a v3 record in every other respect, so existing consumers parse it.
        assert_eq!(json["schema_version"], 3);
        assert_eq!(json["category"], "isomorphic_acasxu_2026");
        assert_eq!(json["budget_secs"], 100);
    }

    #[test]
    fn a_hard_stop_after_the_verdict_writes_a_complete_record_not_a_partial_one() {
        // The measured "published sat at 1.7 s, then wedged in exit handlers"
        // class: the row reached its answer and the stop only forces teardown.
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "cctsdb_yolo_2023", 350);
        recorder.note("upfront_attack", FlightStatus::Ran, Some("sat".into()));
        recorder.finish("sat");
        recorder.write_hard_stop_sidecar(&results_file, "in-process watchdog; verdict preserved");

        let body = std::fs::read_to_string(sidecar_path(&results_file)).expect("sidecar written");
        let json: serde_json::Value = serde_json::from_str(&body).expect("sidecar is valid JSON");
        assert!(
            json.get("record_status").is_none(),
            "a run that reached its verdict is not partial, whatever ended the process"
        );
        let terminal = json["events"]
            .as_array()
            .expect("events")
            .last()
            .cloned()
            .expect("terminal event");
        assert_eq!(terminal["method"], "run_complete");
        assert_eq!(terminal["status"], "complete");
        assert_eq!(terminal["reason"], "sat");
    }

    #[test]
    fn the_hard_stop_flush_leaves_a_complete_record_untouched_by_a_later_partial() {
        // Ordering safety: the ordinary post-verdict write and the watchdog
        // flush can race. Whichever runs second, the sidecar must still carry
        // the verdict — never a partial that erases it.
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "cersyve", 100);
        recorder.note("internal_verifier", FlightStatus::Ran, None);
        recorder.finish("unsat");
        recorder.write_sidecar(&results_file);
        recorder.write_hard_stop_sidecar(&results_file, "in-process watchdog");

        let body = std::fs::read_to_string(sidecar_path(&results_file)).expect("sidecar written");
        let json: serde_json::Value = serde_json::from_str(&body).expect("sidecar is valid JSON");
        assert!(json.get("record_status").is_none());
        let terminals: Vec<_> = json["events"]
            .as_array()
            .expect("events")
            .iter()
            .filter(|event| event["method"] == "run_complete")
            .collect();
        assert_eq!(terminals.len(), 1, "exactly one terminal disposition");
        assert_eq!(terminals[0]["reason"], "unsat");
    }

    #[test]
    fn the_hard_stop_flush_leaves_no_staging_file_behind() {
        // A harness collects the instance directory; a leftover staging file
        // would be a second, unversioned artifact claiming to be a record.
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "acasxu_2023", 116);
        recorder.write_hard_stop_sidecar(&results_file, "in-process watchdog");

        let entries: Vec<String> = std::fs::read_dir(tmp.path())
            .expect("readdir")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(
            entries,
            ["result.txt.flight.json"],
            "the record lands under exactly one name, atomically"
        );
    }

    #[test]
    fn before_begin_a_hard_stop_flush_writes_nothing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.note("upfront_attack", FlightStatus::Ran, None);
        recorder.write_hard_stop_sidecar(&results_file, "in-process watchdog");
        assert!(
            !sidecar_path(&results_file).exists(),
            "a run that never armed the recorder cannot be attributed a partial record"
        );
    }

    #[test]
    fn a_hard_stop_flush_onto_an_unwritable_path_is_non_fatal() {
        // Same contract as the post-verdict write: the deadline outranks the
        // record, and an unwritable sidecar path must cost neither.
        let tmp = tempfile::tempdir().expect("tempdir");
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"not a directory").expect("write blocker");
        let results_file = blocker.join("result.txt");

        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "safenlp_2024", 20);
        recorder.note("internal_verifier", FlightStatus::Ran, None);
        recorder.write_hard_stop_sidecar(&results_file, "in-process watchdog");
        assert!(
            recorder.snapshot().is_some(),
            "the in-memory record survives a failed flush"
        );
        assert!(
            recorder.snapshot().expect("record").record_status.is_none(),
            "the flush marks the SERIALIZED copy, never the live record"
        );
    }

    #[test]
    fn a_completed_record_is_unchanged_by_the_partial_marker_field() {
        // The completed-row schema is a consumer contract (bank metadata,
        // qualification scripts key on v3). Adding the partial envelope must
        // not add a key to records that are not partial.
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "acasxu_2023", 116);
        recorder.note("result_publish", FlightStatus::Ran, Some("unsat".into()));
        recorder.finish("unsat");
        recorder.write_sidecar(&results_file);

        let body = std::fs::read_to_string(sidecar_path(&results_file)).expect("sidecar written");
        let json: serde_json::Value = serde_json::from_str(&body).expect("sidecar is valid JSON");
        let keys: Vec<&str> = json
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        // `serde_json::Value` sorts object keys, so this is the key SET the
        // completed schema promises — nothing added, nothing dropped.
        assert_eq!(
            keys,
            [
                "ambient_env",
                "ambient_env_schema",
                "backend_kind",
                "backend_summary",
                "budget_secs",
                // #binary-identity: WHICH VERIFIER produced the row. Not
                // optional in the key set — a record that cannot name its
                // binary is the exact artifact that let 330 rows be
                // misattributed on 2026-08-21.
                "build_id",
                "category",
                "events",
                "host",
                "levers",
                "load_avg_at_begin",
                "load_avg_at_end",
                "schema_version",
            ],
            "no new key may appear on a completed record"
        );
    }

    #[test]
    fn sidecar_path_appends_rather_than_replacing_the_extension() {
        assert_eq!(
            sidecar_path(Path::new("/tmp/result.txt")),
            Path::new("/tmp/result.txt.flight.json"),
            "with_extension would map result.txt and result.csv to the same sidecar"
        );
    }
}
